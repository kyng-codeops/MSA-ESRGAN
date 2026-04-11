#!/usr/bin/env python3
"""Strip ICC profiles from PNG files to eliminate libpng warnings."""

import glob
import os
from PIL import Image
from tqdm import tqdm


def strip_icc(directories):
    for d in directories:
        files = glob.glob(os.path.join(d, '*.png'))
        print(f'Processing {len(files)} files in {d}')

        stripped = 0
        for f in tqdm(files):
            img = Image.open(f)
            if 'icc_profile' in img.info:
                # Load to memory, save without ICC
                img.load()
                img.info.pop('icc_profile', None)
                img.save(f, 'PNG')
                stripped += 1

        print(f'  Stripped ICC from {stripped} files\n')


if __name__ == '__main__':
    dirs = [
        'datasets/ds_root/unsplash-4k-photos_png',
        'datasets/ds_root/unsplash-4k-photos_multiscale'
    ]
    strip_icc(dirs)
    print('Done!')
