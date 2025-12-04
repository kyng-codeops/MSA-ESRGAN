import os
import subprocess
import sys
from pathlib import Path
from typing import List


class ScriptError(Exception):
    """Custom exception for script-specific errors."""
    pass


def validate_working_directory():
    """Ensure script is run from the MSA-ESRGAN root directory."""
    required_paths = [
        "scripts/generate_multiscale_DBUFF.py",
        "scripts/generate_meta_info.py",
        "realesrgan/train.py",
        "options"
    ]

    for path in required_paths:
        if not Path(path).exists():
            raise ScriptError(
                "Script must be run from MSA-ESRGAN root directory.\n"
                f"Missing required path: {path}\n"
                "Please run: cd /path/to/MSA-ESRGAN"
            )


def check_conda_env():
    """Verify correct conda environment is active."""
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    expected_env = os.path.join(os.path.expanduser('~'), '.conda/envs/cuda-torch')

    if conda_prefix != expected_env:
        raise ScriptError(
            "Incorrect conda environment!\n"
            "Please run: conda activate cuda-torch"
        )


def setup_datasets(ds_prefix: str, ds_names: List[str], meta_name: str):
    """Process multiple datasets and create meta info with custom name."""
    try:
        # Ensure dataset root and meta_info directories exist
        meta_dir = Path(ds_prefix) / "meta_info"
        meta_dir.mkdir(parents=True, exist_ok=True)

        for ds_name in ds_names:
            ds_path = Path(ds_prefix) / ds_name
            if not ds_path.exists():
                raise ScriptError(f"Dataset directory not found: {ds_path}")

            multiscale_path = Path(ds_prefix) / f"{ds_name}_multiscale"
            if multiscale_path.exists():
                print(f"Skipping {multiscale_path} - already exists")
                continue

            print(f"Generating {ds_name}")
            result = subprocess.run([
                "python", "scripts/generate_multiscale_DBUFF.py",
                "--input", str(ds_path),
                "--output", str(multiscale_path)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise ScriptError(f"Failed to generate multiscale dataset for {ds_name}:\n{result.stderr}")

            result = subprocess.run([
                "python", "scripts/generate_meta_info.py",
                "--input", str(ds_path), str(multiscale_path),
                "--root", ds_prefix, ds_prefix,
                "--meta_info", f"{ds_prefix}/meta_info/meta_info_{ds_name}.txt"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise ScriptError(f"Failed to generate meta info for {ds_name}:\n{result.stderr}")

        # Create combined meta info with custom name
        meta_file = meta_dir / f"meta_info_{meta_name}.txt"
        with open(meta_file, 'w') as outfile:
            for ds_name in ds_names:
                source_meta = meta_dir / f"meta_info_{ds_name}.txt"
                if not source_meta.exists():
                    raise ScriptError(f"Meta info file not found: {source_meta}")
                with open(source_meta, 'r') as infile:
                    outfile.write(infile.read())
        print(f"Created combined meta info: {meta_file}")

    except (OSError, IOError) as e:
        raise ScriptError(f"File system error: {str(e)}")


def main():
    try:
        validate_working_directory()
        check_conda_env()

        ds_prefix = 'datasets/ds_root'
        ds_names = [
            'iphone14+4kv2',
            '4K_dbj_closeups'
        ]
        meta_name = 'ip14v33'

        setup_datasets(ds_prefix, ds_names, meta_name)
        print("Dataset processing completed successfully!")

    except ScriptError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
