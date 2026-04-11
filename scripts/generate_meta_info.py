import argparse
import cv2
import glob
import os


def main(args):
    txt_file = open(args.meta_info, 'w')
    for folder, root in zip(args.input, args.root):
        img_paths = sorted(glob.glob(os.path.join(folder, '*')))
        for img_path in img_paths:
            status = True
            if args.check:
                # read the image once for check, as some images may have errors
                try:
                    img = cv2.imread(img_path)
                except (IOError, OSError) as error:
                    print(f'Read {img_path} error: {error}')
                    status = False
                if img is None:
                    status = False
                    print(f'Img is None: {img_path}')
            if status:
                # get the relative path
                img_name = os.path.relpath(img_path, root)
                print(img_name)
                txt_file.write(f'{img_name}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Generate meta_info.txt file listing all GT images for training.

Creates a text file with relative paths (one per line) for use in training configs.
Can merge multiple input folders into a single metadata file.

The number of --input and --root arguments must match. Each input folder's paths
are computed relative to its corresponding root.''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  # Single folder (originals only)
  python scripts/generate_meta_info.py \\
      --input datasets/ds_root/iphone14+4kv2 \\
      --root datasets/ds_root \\
      --meta_info datasets/ds_root/meta_info/meta_info_iphone14.txt

  # Combine originals + multiscale (typical usage)
  python scripts/generate_meta_info.py \\
      --input datasets/ds_root/iphone14+4kv2 datasets/ds_root/iphone14+4kv2_multiscale \\
      --root datasets/ds_root datasets/ds_root \\
      --meta_info datasets/ds_root/meta_info/meta_info_iphone14_multiscale.txt

  # Multiple datasets combined
  python scripts/generate_meta_info.py \\
      --input datasets/ds_root/DIV2K_train_HR datasets/ds_root/DIV2K_train_HR_multiscale \\
             datasets/ds_root/4K_action-movies datasets/ds_root/4K_action-movies_multiscale \\
      --root datasets/ds_root datasets/ds_root datasets/ds_root datasets/ds_root \\
      --meta_info datasets/ds_root/meta_info/meta_info_combined.txt

  # With image validation (slower, checks for corrupted images)
  python scripts/generate_meta_info.py \\
      --input datasets/ds_root/unsplash-4k-photos_png \\
      --root datasets/ds_root \\
      --meta_info datasets/ds_root/meta_info/meta_info_unsplash.txt \\
      --check

Output format (one image path per line, relative to root):
  iphone14+4kv2/IMG_0001.png
  iphone14+4kv2/IMG_0002.png
  iphone14+4kv2_multiscale/IMG_0001T0.png
  ...
''')
    parser.add_argument(
        '--input',
        nargs='+',
        required=True,
        help='Input folder(s) containing images (can specify multiple)')
    parser.add_argument(
        '--root',
        nargs='+',
        required=True,
        help='Root folder(s) for computing relative paths (must match --input count)')
    parser.add_argument(
        '--meta_info',
        type=str,
        required=True,
        help='Output path for meta_info.txt file')
    parser.add_argument('--check', action='store_true',
                        help='Validate each image is readable (slower but catches corrupt files)')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)

    main(args)
