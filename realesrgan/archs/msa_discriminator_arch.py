from basicsr.utils.registry import ARCH_REGISTRY
import torch
from torch import nn as nn
# from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class ChannelAttention(nn.Module):
    def __init__(self, num_in_ch):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(num_in_ch, num_in_ch // 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(num_in_ch // 2, num_in_ch, 1, bias=False)
        # Use GroupNorm instead of BatchNorm - more stable for GANs
        num_groups = min(32, num_in_ch // 2)  # Ensure num_groups divides num_in_ch
        self.group_norm = nn.GroupNorm(num_groups, num_in_ch)

        self.conv3 = nn.Conv2d(num_in_ch, num_in_ch, 1, bias=False)
        num_groups2 = min(32, num_in_ch // 2)
        self.group_norm2 = nn.GroupNorm(num_groups2, num_in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv1(self.avg_pool(x))
        avg_out = self.lrelu(avg_out)
        avg_out = self.group_norm(self.conv2(avg_out))

        conv_out = self.conv3(x)
        conv_out = self.group_norm2(self.lrelu(conv_out))

        combined = avg_out + conv_out
        attention = self.sigmoid(combined)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class CSAFM(nn.Module):
    def __init__(self, num_in_ch):
        super(CSAFM, self).__init__()
        self.channel_attention = ChannelAttention(num_in_ch)
        self.spatial_attention = SpatialAttention()

    def forward(self, x, y):
        combined = x + y
        ca_out = self.channel_attention(combined)
        sa_out = self.spatial_attention(ca_out)
        return (sa_out * x) + (sa_out * y) + sa_out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.conv(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.conv(self.lrelu(self.upsample(x))))


@ARCH_REGISTRY.register()
class UNetDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=False):
        super(UNetDiscriminator, self).__init__()
        self.skip_connection = skip_connection
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.encoder1 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.encoder2 = DownBlock(num_feat, num_feat * 2)
        self.encoder3 = DownBlock(num_feat * 2, num_feat * 4)
        self.encoder4 = DownBlock(num_feat * 4, num_feat * 8)

        self.up1 = UpBlock(num_feat * 8, num_feat * 4)
        self.csafm1 = CSAFM(num_feat * 4)
        self.up2 = UpBlock(num_feat * 4, num_feat * 2)
        self.csafm2 = CSAFM(num_feat * 2)
        self.up3 = UpBlock(num_feat * 2, num_feat)
        self.csafm3 = CSAFM(num_feat)

        self.final_conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.final_conv2 = spectral_norm(nn.Conv2d(num_feat, 1, 3, 1, 1))

    def forward(self, x):
        # NOTE: prior to Wasserstein GAN with gradient penalty, no skip connections
        # existed even if skip_connection was set to True.

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d1 = self.up1(e4)
        # optional residual skip: add corresponding encoder feature
        if self.skip_connection:
            # e3 has same channel dim as d1 (num_feat*4)
            d1 = d1 + e3
        d1 = self.csafm1(d1, e3)

        d2 = self.up2(d1)
        if self.skip_connection:
            # e2 has same channel dim as d2 (num_feat*2)
            d2 = d2 + e2
        d2 = self.csafm2(d2, e2)

        d3 = self.up3(d2)
        if self.skip_connection:
            # e1 has same channel dim as d3 (num_feat)
            d3 = d3 + e1
        d3 = self.csafm3(d3, e1)

        out = self.lrelu(self.final_conv1(d3))
        out = self.lrelu(self.final_conv1(out))
        out = self.lrelu(self.final_conv2(out))

        return out
