"""LPIPS metric for validation

LPIPS (Learned Perceptual Image Patch Similarity) is better than PSNR/SSIM for
perceptual quality assessment as it correlates better with human judgments.

Lower LPIPS = better perceptual similarity (range: 0-1, typical 0.01-0.5)
"""

import torch
import numpy as np
from realesrgan.losses.lpips_loss import LPIPS
from basicsr.utils.registry import METRIC_REGISTRY


# Global LPIPS network for reuse across validation calls
_lpips_net = None


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border=0, test_y_channel=False, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity)

    Args:
        img (ndarray): Images with range [0, 255], shape (H, W, C)
        img2 (ndarray): Images with range [0, 255], shape (H, W, C)
        crop_border (int): Cropped pixels in each edge of an image. Default: 0
        test_y_channel (bool): Not used for LPIPS, kept for API compatibility
        **kwargs: Other arguments (ignored)

    Returns:
        float: LPIPS distance (lower is better, 0 = identical)
    """
    global _lpips_net

    assert img.shape == img2.shape, (
        f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert to torch tensor [0, 1] range
    img = torch.from_numpy(img / 255.0).float().permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2 / 255.0).float().permute(2, 0, 1).unsqueeze(0)

    # Move to GPU if available
    if torch.cuda.is_available():
        img = img.cuda()
        img2 = img2.cuda()

    # Initialize LPIPS network once
    if _lpips_net is None:
        _lpips_net = LPIPS(net_type='vgg', use_dropout=False)
        if torch.cuda.is_available():
            _lpips_net = _lpips_net.cuda()
        _lpips_net.eval()

    # Compute LPIPS
    with torch.no_grad():
        lpips_val = _lpips_net(img, img2)

    # Return absolute value to ensure positive (sometimes gets small negatives due to numerical precision)
    return abs(lpips_val.item())
