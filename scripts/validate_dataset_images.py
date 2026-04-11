#!/usr/bin/env python3
"""Validate all images in a metadata file for corruption.

Scans all images listed in the dataset metadata file and reports any
corrupted or truncated files that would cause training to crash.

Usage:
    python scripts/validate_dataset_images.py [meta_info_file]

Example:
    python scripts/validate_dataset_images.py datasets/ds_root/meta_info/meta_info_ip14v33_nostreak_unsplash.txt
"""

import sys
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_image(path):
    """Check if an image file is valid and can be fully decoded."""
    try:
        img = Image.open(path)
        img.load()  # Force full decode to catch truncation
        return None
    except Exception as e:
        return (path, str(e))


def main():
    # Default metadata file
    if len(sys.argv) > 1:
        meta_file = sys.argv[1]
    else:
        meta_file = 'datasets/ds_root/meta_info/meta_info_ip14v33_nostreak_unsplash.txt'

    ds_root = 'datasets/ds_root'

    if not os.path.exists(meta_file):
        print(f'Error: {meta_file} not found')
        sys.exit(1)

    print(f'Reading {meta_file}...')
    with open(meta_file) as f:
        lines = [l.strip().split()[0] for l in f if l.strip()]

    print(f'Scanning {len(lines)} images with 16 threads...')
    print(flush=True)

    corrupted = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(check_image, os.path.join(ds_root, p)): p for p in lines}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                corrupted.append(result)
                print(f'CORRUPT: {result[0]}')
                print(f'  Error: {result[1]}')
            if i % 2000 == 0:
                print(f'Progress: {i}/{len(lines)}', flush=True)

    print(f'\n{"="*60}')
    print(f'Scan complete: {len(lines)} images checked')

    if corrupted:
        print(f'\nFOUND {len(corrupted)} CORRUPTED FILES:')
        for path, err in corrupted:
            print(f'  {path}')
        sys.exit(1)
    else:
        print('All images validated successfully!')
        sys.exit(0)


if __name__ == '__main__':
    main()
