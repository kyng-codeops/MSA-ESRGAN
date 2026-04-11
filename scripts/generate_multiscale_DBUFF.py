import argparse
import glob
import os
import threading
from queue import Queue
from tqdm import tqdm
from PIL import Image


def save_images(buffer, output_path):
    """Save images in the buffer to disk using multiple threads."""
    while True:
        image_info = buffer.get()
        if image_info is None:
            buffer.task_done()
            break
        img, basename, idx = image_info
        img.save(os.path.join(output_path, f'{basename}T{idx}.png'))
        buffer.task_done()


def main(args):
    # Configuration
    scale_list = [0.75, 0.5, 1 / 3]
    shortest_edge = 400
    buffer_size = 96  # Set buffer size to 96

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Setup buffers and threads for double buffering
    buffer1, buffer2 = Queue(maxsize=buffer_size), Queue(maxsize=buffer_size)
    active_buffer, next_buffer = buffer1, buffer2

    # Initialize and start 24 threads for saving images in buffer1
    num_threads = 24
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=save_images, args=(buffer1, args.output))
        t.start()
        threads.append(t)

    # Process images
    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    total_images = len(path_list) * (len(scale_list) + 1)  # Total images processed
    progress_bar = tqdm(total=total_images, desc="Processing Images")

    for path in path_list:
        basename = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path)
        width, height = img.size

        # Strip ICC profile to avoid libpng warnings during training
        img.info.pop('icc_profile', None)

        # Generate and queue scaled images
        for idx, scale in enumerate(scale_list):
            rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            active_buffer.put((rlt, basename, idx))
            progress_bar.update(1)  # Update progress for each scaled image

        # Queue the smallest image (shortest edge set to 400)
        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)
        rlt = img.resize((width, height), resample=Image.LANCZOS)
        active_buffer.put((rlt, basename, len(scale_list)))
        progress_bar.update(1)  # Update progress for the smallest image

        # Switch buffers if the active buffer is full
        if active_buffer.full():
            active_buffer, next_buffer = next_buffer, active_buffer

            # Start new threads for the new active buffer
            for _ in range(num_threads):
                t = threading.Thread(target=save_images, args=(active_buffer, args.output))
                t.start()
                threads.append(t)

    # Signal threads to stop after all images are processed
    for _ in range(num_threads):
        active_buffer.put(None)
        next_buffer.put(None)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    progress_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Generate multi-scale image variants for training using double-buffered I/O.

Creates 4 scaled versions of each input image:
  - T0: 0.75x scale
  - T1: 0.50x scale
  - T2: 0.33x scale
  - T3: shortest edge = 400px

Output naming: {input_basename}T{0-3}.png

Uses 24 writer threads with 96-item double buffers for fast disk I/O.''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  # Process a new 4K dataset
  python scripts/generate_multiscale_DBUFF.py \\
      --input datasets/ds_root/4K_action-movies \\
      --output datasets/ds_root/4K_action-movies_multiscale

  # Process unsplash PNG originals
  python scripts/generate_multiscale_DBUFF.py \\
      --input datasets/ds_root/unsplash-4k-photos_png \\
      --output datasets/ds_root/unsplash-4k-photos_multiscale

  # Process DIV2K
  python scripts/generate_multiscale_DBUFF.py \\
      --input datasets/ds_root/DIV2K_train_HR \\
      --output datasets/ds_root/DIV2K_train_HR_multiscale

Typical workflow:
  1. Place HR images in datasets/ds_root/<dataset_name>/
  2. Run this script with --output datasets/ds_root/<dataset_name>_multiscale
  3. Run generate_meta_info.py to create metadata file for training
''')
    parser.add_argument('--input', type=str, required=True,
                        help='Input folder containing HR images (e.g., datasets/ds_root/iphone14+4kv2)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output folder for multiscale images (e.g., datasets/ds_root/iphone14+4kv2_multiscale)')
    args = parser.parse_args()

    main(args)
