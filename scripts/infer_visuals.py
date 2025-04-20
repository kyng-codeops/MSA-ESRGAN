import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run RealESRGAN inference with user-specified paths.")
    parser.add_argument('--mppath', type=str, required=True, help='Path to the model file (MPPATH).')
    parser.add_argument('--vopath', type=str, required=True, help='Output directory path (VOPATH).')
    args = parser.parse_args()

    # Define the base command with placeholders for dynamic arguments
    base_command = (
        "python inference_realesrgan.py -n RealESRGAN_x4plus -i {input_path} -o {output_path} --model_path {model_path}"
    )

    # Input directories
    input_dirs = [
        "~/Downloads/upscale/esta",
        "~/Downloads/upscale/orig/",
        "~/Downloads/upscale/estadmg/"
    ]

    # Run the commands for each input directory
    for input_dir in input_dirs:
        command = base_command.format(input_path=input_dir, output_path=args.vopath, model_path=args.mppath)
        os.system(command)

if __name__ == "__main__":
    main()