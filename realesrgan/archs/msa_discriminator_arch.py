from basicsr.utils.registry import ARCH_REGISTRY
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

class ChannelAttention(nn.Module):
    def __init__(self, num_in_ch, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_in_ch, num_in_ch // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_in_ch // reduction, num_in_ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x))

class AttentionBlock(nn.Module):
    def __init__(self, num_in_ch):
        super(AttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(num_in_ch)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

@ARCH_REGISTRY.register()
class UNetDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super(UNetDiscriminator, self).__init__()
        self.skip_connection = skip_connection

        # Encoding (Downsampling)
        self.enc1 = spectral_norm(nn.Conv2d(num_in_ch, num_feat, 3, stride=1, padding=1))
        self.enc2 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 3, stride=2, padding=1))
        self.enc3 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 3, stride=2, padding=1))
        self.enc4 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 3, stride=2, padding=1))

        # Attention blocks
        self.att1 = AttentionBlock(num_feat * 2)
        self.att2 = AttentionBlock(num_feat * 4)
        self.att3 = AttentionBlock(num_feat * 8)

        # Decoding (Upsampling)
        self.up1 = nn.ConvTranspose2d(num_feat * 8, num_feat * 4, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(num_feat * 4, num_feat * 2, 3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(num_feat * 2, num_feat, 3, stride=2, padding=1, output_padding=1)

        # Final classification layer
        self.final_conv = spectral_norm(nn.Conv2d(num_feat, 1, 3, stride=1, padding=1))

    def forward(self, x):

        # FIXME: THERE ARE NO SKIPCONNECTIONS?
        # in the previous UNetDiscriminiator code the skipconnections between x5 and x6 had code that
        # looked like:
        #
        # x4 = ...
        # x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        # if self.skip_connection:
        #     x5 = x5 + x1
        # x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        # x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        #
        # if self.skip_connection:
        #     x6 = x6 + x0
        # # extra convolutions
        # out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        #
        #
        # I can see from that example that skipconnections are implemented by over-writting interim
        # encoded results as well as final outputs

        e1 = F.leaky_relu(self.enc1(x), 0.2)
        e2 = F.leaky_relu(self.enc2(e1), 0.2)
        e3 = F.leaky_relu(self.enc3(e2), 0.2)
        e4 = F.leaky_relu(self.enc4(e3), 0.2)

        # Apply attention
        a1 = self.att1(e2)
        a2 = self.att2(e3)
        a3 = self.att3(e4)

        d1 = F.leaky_relu(self.up1(a3), 0.2) + a2
        d2 = F.leaky_relu(self.up2(d1), 0.2) + a1
        d3 = F.leaky_relu(self.up3(d2), 0.2) + e1

        return self.final_conv(d3)
