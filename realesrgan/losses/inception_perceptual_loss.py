"""InceptionV3 Perceptual Loss

Complementary to VGG perceptual loss with different architecture.
InceptionV3 uses inception modules with parallel convolutions at multiple scales,
providing different feature representations than sequential VGG layers.

Reference: "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class InceptionV3Features(nn.Module):
    """Extract features from InceptionV3 at multiple layers

    InceptionV3 architecture has mixed layers (inception modules) that we'll use:
    - Mixed_5b: Early features (similar to VGG conv2)
    - Mixed_6a: Mid-level features (similar to VGG conv3)
    - Mixed_7a: High-level features (similar to VGG conv4/5)
    """

    def __init__(self):
        super(InceptionV3Features, self).__init__()

        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()

        # Freeze all parameters
        for param in inception.parameters():
            param.requires_grad = False

        # Extract the feature extraction layers (before aux classifier and final pooling)
        # InceptionV3 structure: Conv2d_1a -> ... -> Mixed_7c -> avgpool -> fc

        # We'll split at key mixed layers for multi-scale features
        self.layers = nn.ModuleDict({
            # Early features (after Mixed_5d) - ~35x35 spatial resolution
            'mixed_5d': nn.Sequential(
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                inception.maxpool1,
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                inception.maxpool2,
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d
            ),
            # Mid-level features (after Mixed_6e) - ~17x17 spatial resolution
            'mixed_6e': nn.Sequential(
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e
            ),
            # High-level features (after Mixed_7c) - ~8x8 spatial resolution
            'mixed_7c': nn.Sequential(
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c
            )
        })

        # Channel dimensions for each layer
        self.channels = {
            'mixed_5d': 288,
            'mixed_6e': 768,
            'mixed_7c': 2048
        }

    def forward(self, x):
        """Extract multi-scale features from InceptionV3

        Args:
            x: Input image tensor, expected range [0, 1]

        Returns:
            dict: Features at different scales
        """
        # InceptionV3 expects input in range [-1, 1] with ImageNet normalization
        # Input is [0, 1], convert to [-1, 1]
        x = x * 2.0 - 1.0

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # InceptionV3 expects 299x299 input
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        features = {}

        # Extract features sequentially
        feat = self.layers['mixed_5d'](x)
        features['mixed_5d'] = feat

        feat = self.layers['mixed_6e'](feat)
        features['mixed_6e'] = feat

        feat = self.layers['mixed_7c'](feat)
        features['mixed_7c'] = feat

        return features


class InceptionPerceptualLoss(nn.Module):
    """InceptionV3-based perceptual loss

    Complements VGG19 perceptual loss with different architectural features.
    Uses inception modules that capture multi-scale patterns simultaneously.

    Args:
        layer_weights (dict): Weights for each layer. Default uses equal weighting.
        perceptual_weight (float): Overall weight for this loss term
        criterion (str): Distance metric, 'l1' or 'l2'
    """

    def __init__(self, layer_weights=None, perceptual_weight=1.0, criterion='l1', **kwargs):
        super(InceptionPerceptualLoss, self).__init__()

        self.inception_net = InceptionV3Features()
        self.perceptual_weight = perceptual_weight

        # Default layer weights if not specified
        if layer_weights is None:
            self.layer_weights = {
                'mixed_5d': 1.0,  # Early features
                'mixed_6e': 1.0,  # Mid-level features
                'mixed_7c': 1.0   # High-level features
            }
        else:
            self.layer_weights = layer_weights

        # Loss criterion
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")

    def forward(self, pred, target):
        """Compute InceptionV3 perceptual loss

        Args:
            pred: Predicted SR image [0, 1]
            target: Ground truth image [0, 1]

        Returns:
            Perceptual loss value
        """
        # Extract features
        pred_features = self.inception_net(pred)
        target_features = self.inception_net(target)

        # Compute weighted loss across layers
        percep_loss = 0
        for layer_name, weight in self.layer_weights.items():
            if weight > 0:
                loss = self.criterion(pred_features[layer_name], target_features[layer_name])
                percep_loss += weight * loss

        return percep_loss * self.perceptual_weight
