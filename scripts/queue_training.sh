#!/bin/bash

#python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp2.yml
#python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp.yml --auto_resume



#python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3.yml --auto_resume

### Notes: Wgp2 vs Wgp3
## Wgp3 is return to the loss weights of the original RealESRGAN work but with the
## added gv_loss (so perceptual loss set back to 1 and wgan increased to 0.15).
## the wgan increase matches the original percentage of the 0.1 before gv_loss
## to make the weighting of gan losses similar to the original RealESRGAN work.
## slight improvements to details were observed between Wgp2 and Wgp3 (iter 50 the best).

### Notes:
## Wgp3a-3d are the first to use multiple net_d_iters per net_g. GPU OOM errors
## forced changes to the batch_size and queue_size (smaller batch forced change in queue_size)
## and next divisable integer longer queue_size was used since it uses main memory vs gpu_mem.
## This increases the diversity of training vs all prior runs.

# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3c.yml --auto_resume
# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3b.yml --auto_resume
# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3a.yml --auto_resume
# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3d.yml --auto_resume

### Wgp3a vs Wgp3b
## 3b uses an net_d_iters of 5 and the details like spots on skin are washed out
## 3a (most comparable to 3b) uses net_d_iters of 3 and maintains more details

# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3e.yml --auto_resume
# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3c.yml --auto_resume


### Wgp3c vs Wgp3d
## 3c (with gp_lambda 15) at 30k iterations or epoch juast after 10 shows more detail vs 3d (gp_lambda 5)
## has slightly weaker details, but still more details than 3b.

### Observations:
## Too much pre-training on net_d makes the wgan washout details mistaking gradient features as noise.
## Stronger gp_lambda is better than weaker lambda for maintaining gradient features?

# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3g.yml --auto_resume
# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3h.yml --auto_resume

## 3h attempts to initialize with a stronger net_d from 3g (260k iters) and continue training
## 3h also uses a lesser trained net_g (120k iters) vs 3g (2.95M iters).
## but the 3h run suffered from an overly dominant net_g (still lots of negative loss values)
## the lesser trained init net_g also lead to washed out details (reverting to basic RealESRGAN anime quality)
##
## 3i further strengthens the net_d by using a pretrained net_d from a the 3h run (65k iters)
## in the 3h run, net_g is switched back to the stronger 2.95M iters pretrained net_g.
python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3j.yml --auto_resume
# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3i.yml --auto_resume
# python realesrgan/train.py -opt options/train_realesrnet_x4cust.yml --auto_resume
## esrnet with higher ssim and psnr values than 3g, but wgan-gp trained 3g images look subjectively better
## having high ssim and psnr sometimes indicates not much image enhancement has occurred?
## However when comparing iterations within the same model, higher ssim and psnr does correlate sharper clearer images.


# Run-out Wgp3d even with weaker gp_lambda, weaker wgan maybe letting other losses maintain more noise
# which maybe looks more detailed? To confirm, expecting 3d to have details but with more noise/artifacts.
# python realesrgan/train.py -opt options/train_msaesrgan_x4v3.2gvWgp3d.yml --auto_resume

