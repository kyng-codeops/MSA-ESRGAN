# flake8: noqa
import os.path as osp
import torch
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    # Limit VRAM allocation to 97.5% of total GPU memory
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.975)

    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
