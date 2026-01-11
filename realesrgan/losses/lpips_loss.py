"""LPIPS (Learned Perceptual Image Patch Similarity) Loss

Better alternative to VGG perceptual loss that:
- Trained on human perceptual judgments
- Better preserves textures and details
- Less over-smoothing than raw VGG features
- Combines features from multiple layers

Reference: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
https://arxiv.org/abs/1801.03924
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LPIPS(nn.Module):
    """LPIPS perceptual loss network

    Uses VGG19 features with learned channel weighting to match human perception.
    """
    def __init__(self, net_type='vgg', use_dropout=True):
        super(LPIPS, self).__init__()

        # Load pretrained VGG19
        if net_type == 'vgg':
            vgg = models.vgg19(pretrained=True).features

            # Extract feature layers (after ReLU before pooling)
            self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])   # conv1_2
            self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])  # conv2_2
            self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 18)]) # conv3_4
            self.slice4 = nn.Sequential(*[vgg[i] for i in range(18, 27)]) # conv4_4
            self.slice5 = nn.Sequential(*[vgg[i] for i in range(27, 36)]) # conv5_4

            # Freeze feature extraction
            for param in self.parameters():
                param.requires_grad = False

            # Channel dimensions for each layer
            channels = [64, 128, 256, 512, 512]

        else:
            raise ValueError(f"Unsupported network type: {net_type}")

        # Learnable channel weights (1x1 convolutions)
        # These learn which channels are perceptually important
        self.lin0 = NetLinLayer(channels[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(channels[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(channels[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(channels[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(channels[4], use_dropout=use_dropout)

        # Normalization shift and scale (ImageNet stats)
        self.register_buffer('shift', torch.tensor([-.030, -.088, -.188]).view(1, 3, 1, 1))
        self.register_buffer('scale', torch.tensor([.458, .448, .450]).view(1, 3, 1, 1))

    def forward(self, input, target):
        """Compute LPIPS distance between input and target

        Args:
            input: Predicted image [0, 1]
            target: Ground truth image [0, 1]

        Returns:
            LPIPS distance (lower is more similar)
        """
        # Normalize to [-1, 1] then apply ImageNet normalization
        input_norm = (input - 0.5) * 2.0
        target_norm = (target - 0.5) * 2.0

        input_norm = (input_norm - self.shift) / self.scale
        target_norm = (target_norm - self.shift) / self.scale

        # Extract features at multiple scales
        in_feats = self.extract_features(input_norm)
        target_feats = self.extract_features(target_norm)

        # Compute normalized feature differences and apply learned weights
        diffs = []
        diffs.append(self.lin0(normalize_tensor(in_feats[0] - target_feats[0])))
        diffs.append(self.lin1(normalize_tensor(in_feats[1] - target_feats[1])))
        diffs.append(self.lin2(normalize_tensor(in_feats[2] - target_feats[2])))
        diffs.append(self.lin3(normalize_tensor(in_feats[3] - target_feats[3])))
        diffs.append(self.lin4(normalize_tensor(in_feats[4] - target_feats[4])))

        # Spatially average and sum across layers
        return sum([diff.mean([2, 3], keepdim=True) for diff in diffs]).mean()

    def extract_features(self, x):
        """Extract multi-scale VGG features"""
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h

        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class NetLinLayer(nn.Module):
    """Learnable 1x1 conv to weight feature channels by perceptual importance"""
    def __init__(self, in_channels, use_dropout=True):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(in_channels, 1, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def normalize_tensor(x, eps=1e-10):
    """L2 normalize tensor along channel dimension"""
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


class LPIPSLoss(nn.Module):
    """LPIPS Loss wrapper for BasicSR

    Replaces VGG perceptual loss with learned perceptual similarity.
    Better preserves details and textures without over-smoothing.

    Args:
        loss_weight: Weight for this loss term (default: 1.0)
        net_type: Network backbone ('vgg' only currently)
        use_dropout: Use dropout in channel weighting (default: True)
        pretrained_path: Optional path to pretrained LPIPS weights
        reduction: Loss reduction method ('mean' or 'sum')
    """
    def __init__(self, loss_weight=1.0, net_type='vgg', use_dropout=True,
                 pretrained_path=None, reduction='mean', **kwargs):
        super(LPIPSLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.lpips_net = LPIPS(net_type=net_type, use_dropout=use_dropout)

        # Load pretrained LPIPS weights if provided
        # Otherwise uses VGG features with random channel weights (will be trained)
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.lpips_net.load_state_dict(state_dict, strict=False)
            print(f'Loaded pretrained LPIPS weights from {pretrained_path}')

        # Move to GPU if available
        if torch.cuda.is_available():
            self.lpips_net = self.lpips_net.cuda()

    def forward(self, pred, target):
        """Compute LPIPS loss

        Args:
            pred: Predicted SR image in [0, 1] range
            target: Ground truth image in [0, 1] range

        Returns:
            LPIPS loss value (always positive)
        """
        loss = self.lpips_net(pred, target)

        # Ensure non-negative (can get small negatives due to numerical precision)
        loss = torch.abs(loss)

        if self.reduction == 'sum':
            loss = loss * pred.shape[0]  # multiply by batch size for sum reduction

        return loss * self.loss_weight
