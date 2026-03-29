#!/bin/env python

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Run RealESRGAN inference with user-specified paths.")
    parser.add_argument('--mppath', type=str, required=True, help='Path to the model file (MPPATH).')
    parser.add_argument('--vopath', type=str, required=True, help='Output directory path (VOPATH).')
    parser.add_argument('--bit16', action='store_true', help='Save output in 16-bit color depth (PNG format only).')
    args = parser.parse_args()

    # Define the base command with placeholders for dynamic arguments
    base_command = (
        "python inference_realesrgan.py -n RealESRGAN_x2plus -s 2 --fp32 -i {input_path} -o {output_path} --model_path {model_path}"
    )

    # Add 16-bit flag if requested
    if args.bit16:
        base_command += " --bit16"

    # Input directories
    input_dirs = [
        "~/Downloads/upscale/esta",
        "~/Downloads/upscale/orig/",
        "~/Downloads/upscale/estadmg/",
        "~/Documents/validation/lq2_x2_auto_85",
    ]

    # Run the commands for each input directory
    for input_dir in input_dirs:
        command = base_command.format(input_path=input_dir, output_path=args.vopath, model_path=args.mppath)
        os.system(command)

if __name__ == "__main__":
    main()
