#!/usr/bin/env python3
"""
Add unsplash-4k-photos to the existing nostreak metadata file.

This script:
1. Reads the existing meta_info_ip14v33_nostreak.txt
2. Adds entries for unsplash-4k-photos_png (originals)
3. Adds entries for unsplash-4k-photos_multiscale (scaled versions)
4. Writes to meta_info_ip14v33_nostreak_unsplash.txt

Usage:
    conda activate cuda-torch
    python scripts/add_unsplash_to_metadata.py
"""

import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser(
        description='''Append Unsplash dataset entries to an existing metadata file.

This is a convenience script that combines:
  1. An existing base metadata file (e.g., meta_info_ip14v33_nostreak.txt)
  2. Unsplash PNG originals (from prepare_unsplash_dataset.py)
  3. Unsplash multiscale variants (from prepare_unsplash_dataset.py)

Prerequisite: Run prepare_unsplash_dataset.py first to create the PNG/multiscale folders.''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Example:
  # Default usage (appends to ip14v33_nostreak)
  python scripts/add_unsplash_to_metadata.py

  # Custom paths
  python scripts/add_unsplash_to_metadata.py \\
      --base_meta datasets/ds_root/meta_info/meta_info_ip14v33.txt \\
      --output datasets/ds_root/meta_info/meta_info_ip14v33_unsplash.txt

Workflow:
  1. Place source JPGs in datasets/ds_root/unsplash-4k-photos/
  2. Run: python scripts/prepare_unsplash_dataset.py
  3. Run: python scripts/add_unsplash_to_metadata.py
  4. Update training config to use the new metadata file
''')
    parser.add_argument('--base_meta', type=str,
                        default='datasets/ds_root/meta_info/meta_info_ip14v33_nostreak.txt',
                        help='Existing metadata file to append to')
    parser.add_argument('--unsplash_png', type=str,
                        default='datasets/ds_root/unsplash-4k-photos_png',
                        help='Folder with lossless PNG originals (from prepare_unsplash_dataset.py)')
    parser.add_argument('--unsplash_multiscale', type=str,
                        default='datasets/ds_root/unsplash-4k-photos_multiscale',
                        help='Folder with multiscale variants T0-T3 (from prepare_unsplash_dataset.py)')
    parser.add_argument('--ds_root', type=str,
                        default='datasets/ds_root',
                        help='Root folder for computing relative paths')
    parser.add_argument('--output', type=str,
                        default='datasets/ds_root/meta_info/meta_info_ip14v33_nostreak_unsplash.txt',
                        help='Output combined metadata file')
    args = parser.parse_args()

    # Check that source folders exist
    if not os.path.exists(args.unsplash_png):
        print(f"ERROR: {args.unsplash_png} does not exist!")
        print("Run prepare_unsplash_dataset.py first to convert the images.")
        return

    if not os.path.exists(args.unsplash_multiscale):
        print(f"ERROR: {args.unsplash_multiscale} does not exist!")
        print("Run prepare_unsplash_dataset.py first to convert the images.")
        return

    # Read existing metadata
    print(f"Reading existing metadata from: {args.base_meta}")
    with open(args.base_meta, 'r') as f:
        existing_lines = f.read().splitlines()
    print(f"  Found {len(existing_lines)} existing entries")

    # Collect unsplash PNG originals
    print(f"\nScanning unsplash PNG originals: {args.unsplash_png}")
    unsplash_png_files = sorted(glob.glob(os.path.join(args.unsplash_png, '*.png')))
    unsplash_png_rel = [os.path.relpath(f, args.ds_root) for f in unsplash_png_files]
    print(f"  Found {len(unsplash_png_rel)} PNG originals")

    # Collect unsplash multiscale
    print(f"\nScanning unsplash multiscale: {args.unsplash_multiscale}")
    unsplash_ms_files = sorted(glob.glob(os.path.join(args.unsplash_multiscale, '*.png')))
    unsplash_ms_rel = [os.path.relpath(f, args.ds_root) for f in unsplash_ms_files]
    print(f"  Found {len(unsplash_ms_rel)} multiscale variants")

    # Combine all entries
    all_entries = existing_lines + unsplash_png_rel + unsplash_ms_rel

    # Write output
    print(f"\nWriting combined metadata to: {args.output}")
    with open(args.output, 'w') as f:
        for entry in all_entries:
            f.write(f"{entry}\n")

    print(f"  Total entries: {len(all_entries)}")
    print(f"    - Original (nostreak): {len(existing_lines)}")
    print(f"    - New unsplash PNG: {len(unsplash_png_rel)}")
    print(f"    - New unsplash multiscale: {len(unsplash_ms_rel)}")

    print("\nDone! Update your v8e config with:")
    print(f"  meta_info: {args.output}")


if __name__ == '__main__':
    main()
