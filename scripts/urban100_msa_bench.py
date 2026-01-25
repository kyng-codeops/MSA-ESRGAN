#!/usr/bin/env python3
"""
Benchmark MSA-ESRGAN generator on validation dataset.

Usage:
    python scripts/urban100_msa_bench.py --model_path experiments/MSA-ESRGAN-Wgp_x4_ip14v33-d3-cc128lpip-v4D/models/net_g_100000.pth
    python scripts/urban100_msa_bench.py --model_path experiments/MSA-ESRGAN-Wgp_x4_ip14v33-d3-cc128lpip-v4B/models/net_g_58940.pth --tile 256
    python scripts/urban100_msa_bench.py --model_path experiments/.../net_g_100000.pth --lq_dir datasets/ds_root/validation/lq2_auto_85 --hq_dir datasets/ds_root/validation/hq2
"""

import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.metrics import calculate_niqe
from PIL import Image
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.metrics import calculate_lpips


def main():
    parser = argparse.ArgumentParser(description='Benchmark MSA-ESRGAN on validation dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model weights (e.g., experiments/.../models/net_g_100000.pth)')
    parser.add_argument('--lq_dir', type=str, default='datasets/ds_root/validation/lq2_auto_85',
                       help='Directory containing low-quality input images')
    parser.add_argument('--hq_dir', type=str, default='datasets/ds_root/validation/hq2',
                       help='Directory containing high-quality ground truth images')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling factor (default: 4)')
    parser.add_argument('--num_feat', type=int, default=64, help='Number of features (default: 64)')
    parser.add_argument('--num_block', type=int, default=23, help='Number of RRDB blocks (default: 23)')
    parser.add_argument('--num_grow_ch', type=int, default=32, help='Number of grow channels (default: 32)')
    parser.add_argument('--tile', type=int, default=0,
                        help='Tile size for processing large images (0=no tiling, default: 0)')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding (default: 10)')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre-padding (default: 0)')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 precision (default: FP16 if GPU available)')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU device ID (default: None for auto)')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for metrics (default: 4)')
    parser.add_argument('--save_images', action='store_true', help='Save output images')
    parser.add_argument('--output_dir', type=str, default='outputs/urban100_benchmark',
                        help='Directory to save output images')

    args = parser.parse_args()

    # Create RRDBNet model with your MSA-ESRGAN architecture
    print(f'Creating RRDBNet model (feat={args.num_feat}, blocks={args.num_block}, grow_ch={args.num_grow_ch}, scale={args.scale})')
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=args.num_feat,
        num_block=args.num_block,
        num_grow_ch=args.num_grow_ch,
        scale=args.scale
    )

    # Load weights
    print(f'Loading model weights from: {args.model_path}')
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model weights not found: {args.model_path}')

    # Initialize RealESRGANer with your model
    upsampler = RealESRGANer(
        scale=args.scale,
        model_path=args.model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id
    )

    # Load Urban100 dataset
    print(f'Loading validation dataset from {args.lq_dir}...')

    # Get list of LQ images
    lq_images = sorted(glob.glob(os.path.join(args.lq_dir, '*.png')) +
                      glob.glob(os.path.join(args.lq_dir, '*.jpg')))

    if len(lq_images) == 0:
        raise FileNotFoundError(f'No images found in {args.lq_dir}')

    print(f'Found {len(lq_images)} images')

    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'Output images will be saved to: {args.output_dir}')

    # Benchmark metrics
    lpips_values = []
    niqe_values = []

    print(f'\nBenchmarking on {len(lq_images)} images...')

    for idx, lq_path in enumerate(tqdm(lq_images, desc='Processing')):
        # Get corresponding HQ image path
        lq_filename = os.path.basename(lq_path)
        hq_path = os.path.join(args.hq_dir, lq_filename)

        # Skip if HQ image doesn't exist
        if not os.path.exists(hq_path):
            print(f'\nWarning: HQ image not found for {lq_filename}, skipping...')
            continue

        # Load images with PIL
        lr_pil = Image.open(lq_path).convert('RGB')
        hr_pil = Image.open(hq_path).convert('RGB')

        # Convert PIL to BGR for OpenCV
        lr_img_bgr = cv2.cvtColor(np.array(lr_pil), cv2.COLOR_RGB2BGR)
        hr_img_bgr = cv2.cvtColor(np.array(hr_pil), cv2.COLOR_RGB2BGR)

        # Perform super-resolution
        try:
            sr_img_bgr, _ = upsampler.enhance(lr_img_bgr, outscale=args.scale)
        except Exception as e:
            print(f'\nError processing image {idx}: {e}')
            continue

        # Ensure same size for comparison (crop if needed)
        h_hr, w_hr = hr_img_bgr.shape[:2]
        h_sr, w_sr = sr_img_bgr.shape[:2]
        if (h_sr, w_sr) != (h_hr, w_hr):
            sr_img_bgr = sr_img_bgr[:h_hr, :w_hr]

        # Calculate metrics
        # LPIPS: Lower is better (0 = identical, typical range 0.01-0.5)
        lpips = calculate_lpips(sr_img_bgr, hr_img_bgr, crop_border=args.crop_border)
        # NIQE: Lower is better (no-reference metric, typical range 3-10 for good images)
        niqe = calculate_niqe(sr_img_bgr, crop_border=args.crop_border)

        lpips_values.append(lpips)
        niqe_values.append(niqe)

        # Save image if requested
        if args.save_images:
            output_path = os.path.join(args.output_dir, f'urban100_{idx:03d}.png')
            cv2.imwrite(output_path, sr_img_bgr)

    # Print results
    print('\n' + '='*60)
    print('BENCHMARK RESULTS')
    print('='*60)
    print(f'Model: {args.model_path}')
    print(f'LQ Directory: {args.lq_dir}')
    print(f'HQ Directory: {args.hq_dir}')
    print(f'Images processed: {len(lpips_values)}')
    print(f'Crop border: {args.crop_border} pixels')
    print('-'*60)
    if len(lpips_values) > 0:
        print(f'Average LPIPS: {np.mean(lpips_values):.4f} (std: {np.std(lpips_values):.4f}) [lower is better]')
        print(f'Average NIQE:  {np.mean(niqe_values):.4f} (std: {np.std(niqe_values):.4f}) [lower is better]')
        print(f'Min LPIPS: {np.min(lpips_values):.4f}')
        print(f'Max LPIPS: {np.max(lpips_values):.4f}')
        print(f'Min NIQE:  {np.min(niqe_values):.4f}')
        print(f'Max NIQE:  {np.max(niqe_values):.4f}')
    else:
        print('No images were processed successfully!')
    print('='*60)


if __name__ == '__main__':
    main()
