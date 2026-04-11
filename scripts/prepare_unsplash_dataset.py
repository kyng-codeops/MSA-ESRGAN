#!/usr/bin/env python3
"""
Prepare unsplash-4k-photos dataset for training.

This script:
1. Converts JPG images to PNG format (lossless)
2. Creates multi-scale variants (0.75x, 0.5x, 0.33x, shortest_edge=400)

Usage:
    conda activate cuda-torch
    python scripts/prepare_unsplash_dataset.py
"""

import argparse
import glob
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_and_scale(input_dir, output_original, output_multiscale, scale_list, shortest_edge):
    """Convert JPGs to PNG and create multi-scale variants."""

    os.makedirs(output_original, exist_ok=True)
    os.makedirs(output_multiscale, exist_ok=True)

    # Find all images (jpg, jpeg, png)
    extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    path_list = []
    for ext in extensions:
        path_list.extend(glob.glob(os.path.join(input_dir, ext)))
    path_list = sorted(set(path_list))

    print(f"Found {len(path_list)} images to process")

    for path in tqdm(path_list, desc="Processing images"):
        basename = Path(path).stem  # filename without extension

        try:
            img = Image.open(path)
            # Convert to RGB if necessary (handles RGBA, palette modes)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size

            # Strip ICC profile to avoid libpng warnings during training
            img.info.pop('icc_profile', None)

            # Save original as PNG
            original_path = os.path.join(output_original, f'{basename}.png')
            img.save(original_path, 'PNG')

            # Generate multi-scale versions
            for idx, scale in enumerate(scale_list):
                rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
                rlt.save(os.path.join(output_multiscale, f'{basename}T{idx}.png'))

            # Save smallest image (shortest edge = 400)
            if width < height:
                ratio = height / width
                new_width = shortest_edge
                new_height = int(new_width * ratio)
            else:
                ratio = width / height
                new_height = shortest_edge
                new_width = int(new_height * ratio)

            rlt = img.resize((new_width, new_height), resample=Image.LANCZOS)
            rlt.save(os.path.join(output_multiscale, f'{basename}T{len(scale_list)}.png'))

        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description='''Prepare Unsplash 4K photos dataset for training.

Converts JPG images to lossless PNG and creates multi-scale variants.
Strips ICC profiles to avoid libpng warnings during training.

Creates 4 scaled versions of each input image:
  - Original: Full resolution PNG (in output_original folder)
  - T0: 0.75x scale (in output_multiscale folder)
  - T1: 0.50x scale
  - T2: 0.33x scale
  - T3: shortest edge = 400px''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Example:
  python scripts/prepare_unsplash_dataset.py \\
      --input datasets/ds_root/unsplash-4k-photos \\
      --output_original datasets/ds_root/unsplash-4k-photos_png \\
      --output_multiscale datasets/ds_root/unsplash-4k-photos_multiscale

After running this script:
  1. Run add_unsplash_to_metadata.py to append to existing metadata
     OR run generate_meta_info.py to create new metadata file
  2. Update training config to use the new metadata file
''')
    parser.add_argument('--input', type=str,
                        default='datasets/ds_root/unsplash-4k-photos',
                        help='Input folder with source images (JPG/PNG)')
    parser.add_argument('--output_original', type=str,
                        default='datasets/ds_root/unsplash-4k-photos_png',
                        help='Output folder for lossless PNG originals')
    parser.add_argument('--output_multiscale', type=str,
                        default='datasets/ds_root/unsplash-4k-photos_multiscale',
                        help='Output folder for multi-scale variants (T0-T3)')
    args = parser.parse_args()

    # Multi-scale configuration (same as DF2K script)
    scale_list = [0.75, 0.5, 1/3]
    shortest_edge = 400

    print(f"Input: {args.input}")
    print(f"Output (originals): {args.output_original}")
    print(f"Output (multiscale): {args.output_multiscale}")
    print(f"Scale factors: {scale_list}")
    print(f"Shortest edge: {shortest_edge}")
    print()

    convert_and_scale(
        args.input,
        args.output_original,
        args.output_multiscale,
        scale_list,
        shortest_edge
    )

    print("\nDone! Next steps:")
    print("1. Run generate_meta_info.py to create updated metadata")
    print("2. Update v8e config to use the new metadata file")


if __name__ == '__main__':
    main()
