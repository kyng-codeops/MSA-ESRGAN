# MSA-ESRGAN (Nov 2024 modifications to Real-ESRGAN+)
### Work In-Progress 3/10/2025

## <div align="center"><b><a href="README.md">English</a></b></div>

<div align="center">

[**Demos**](#-demos-videos) **|** [**Updates**](#-updates) **|** [**Usage**](#-quick-inference) **|** [**Model Zoo**](docs/model_zoo.md) **|** [Install](#-dependencies-and-installation)  **|** [Train](docs/Training.md) **|** [FAQ](docs/FAQ.md) **|** [Contribution](docs/CONTRIBUTING.md)

[![download](https://img.shields.io/github/downloads/xinntao/Real-ESRGAN/total.svg)](https://github.com/xinntao/Real-ESRGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/realesrgan)](https://pypi.org/project/realesrgan/)
[![Open issue](https://img.shields.io/github/issues/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/Real-ESRGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/Real-ESRGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/publish-pip.yml)

</div>
MSA-ESRGAN is a forked modification from Real-ESRGAN.  This is a pytorch and BasicSR implementation of the NIH paper published at the
National Library of Medicine's National Center for Biotechnology Information--with a modification on activation functions.

MSA stands for Multi-Scale Attention and refers to the discriminator network architecture of the GAN.  This GAN uses the
exact same generator as ESRGAN, Real-ESRGAN, and others (I believe BSRGAN also uses the same generator).  The new discriminator
is meant to improve the GAN's ability to focus attention in areas that might or should improve perceptual quality but
using a combination of Channel Attention and Spatial Attention. From the various versions of the same paper, the authors
only called for the use or ReLU as the activation function.  However, the first couple rounds of training demonstrated
the need to use Leaky ReLU.


I became interested in this work while training Real-ESRGAN from the ground up
on a custom built 4K natural image dataset that included the artistic use of focused foregrounds and blurred backgrounds.
My goal was to see if a custom data set could teach Real-ESRGAN to maintain more details on images of human faces (i.e. getting
rid of the airbrushed over smoothing on humans).  With a dataset of nearly 20,000 4K and higher images of mostly people, places,
and more, I was able to make Real-ESRGAN generate skin textures and skin tone gradients. However, Real-ESRGAN had a tendency
to attempt sharpening blurred backgrounds and blurring some foregrounds.  This repo is my first functioning MSA (Multi-Scale Attention) discriminator
modification to Real-ESRGAN (the generators between ESRGAN, Real-ESRGAN, and MSA-ESRGAN are all identical, only the discriminator is changed).  Since this is a work in-progress,
I will eventually fix-up this repo's documentation.  For now, there will be tons of old references and links and typos.

Inferencing this model is identical to Real-ESRGAN so I'm keeping much of that documentation for now.  The training procedures
are also the same but the MSA discriminator is unique.



If MSA-ESRGAN is helpful, please help to ⭐ this repo<br>
Other recommended projects:<br>
▶️ [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for General Image/Video Restoration <br>
▶️ [GFPGAN](https://github.com/TencentARC/GFPGAN): A practical algorithm for real-world face restoration <br>
▶️ [BasicSR](https://github.com/xinntao/BasicSR): An open-source image and video restoration toolbox<br>
▶️ [facexlib](https://github.com/xinntao/facexlib): A collection that provides useful face-relation functions.<br>
▶️ [HandyView](https://github.com/xinntao/HandyView): A PyQt5-based image viewer that is handy for view and comparison <br>
▶️ [HandyFigure](https://github.com/xinntao/HandyFigure): Open source of paper figures <br>

---
### MSA-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data
> [[Paper](https://pubmed.ncbi.nlm.nih.gov/39580539/)] &emsp; [[Paper](https://www.nature.com/articles/s41598-024-78813-5)]

<p align="center">
  <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-024-78813-5/MediaObjects/41598_2024_78813_Fig4_HTML.png?as=webp">
</p>

---

<!---------------------------------- Updates --------------------------->
## Updates

- ✅ Support finetuning on your own data or paired data (*i.e.*, finetuning ESRGAN). See [here](docs/Training.md#Finetune-Real-ESRGAN-on-your-own-dataset)
- ✅ Integrate [GFPGAN](https://github.com/TencentARC/GFPGAN) to support **face enhancement**.
- ✅ Support arbitrary scale with `--outscale` (It actually further resizes outputs with `LANCZOS4`). Add *RealESRGAN_x2plus.pth* model.
- ✅ [The inference code](inference_realesrgan.py) supports: 1) **tile** options; 2) images with **alpha channel**; 3) **gray** images; 4) **16-bit** images.
- ✅ The training codes have been released. A detailed guide can be found in [Training.md](docs/Training.md).

---

<!---------------------------------- Demo videos --------------------------->
## Demos Videos

#### YouTube

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/kyng-codeops/MSA-ESRGAN
    cd MSA-ESRGAN
    ```

1. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr
    # facexlib and gfpgan are for face enhancement
    pip install facexlib
    pip install gfpgan
    pip install -r requirements.txt
    python setup.py develop
    ```

---

## Quick Inference

Unlike the Real-ESRGAN project, there is only one way to inference MSA-ESRGAN.

1. [Python script](#python-script)

### Online inference

TBD maybe?

### Portable executable files (NCNN)

No plans... however, the generators are identical between Real-ESRGAN, ESRGAN etc so you could
plug these models into the windows executable in xinntao's repo (if I end up posting my trained models).
They are just <filename>.pth and you only run the generator (i.e. the net_g_xxxxxxx.pth).  The discriminator
models are only used during training so all of the MSA benefits are reflected in the model weights.
Just go up the the "Other Recommended Projects" section of this README to find a link to their portable exe file.

### Python script

#### Usage of python script

1. You can use X4 to train your own models... if I find great results I might share my pretrained models (so far so good)

```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

#### Inference general images

Download pre-trained models: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```

Inference!

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```

Results are in the `results` folder


## Projects that use Real-ESRGAN

These projects could also make use of any pretrained models from the MSA-ESRGAN work here (if I post any) or ones that
you train yourself.

- NCNN-Android: [RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android) by [tumuyan](https://github.com/tumuyan)
- VapourSynth: [vs-realesrgan](https://github.com/HolyWu/vs-realesrgan) by [HolyWu](https://github.com/HolyWu)
- NCNN: [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)

&nbsp;&nbsp;&nbsp;&nbsp;**GUI**

- [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) by [AaronFeng753](https://github.com/AaronFeng753)
- [Squirrel-RIFE](https://github.com/Justin62628/Squirrel-RIFE) by [Justin62628](https://github.com/Justin62628)
- [Real-GUI](https://github.com/scifx/Real-GUI) by [scifx](https://github.com/scifx)
- [Real-ESRGAN_GUI](https://github.com/net2cn/Real-ESRGAN_GUI) by [net2cn](https://github.com/net2cn)
- [Real-ESRGAN-EGUI](https://github.com/WGzeyu/Real-ESRGAN-EGUI) by [WGzeyu](https://github.com/WGzeyu)
- [anime_upscaler](https://github.com/shangar21/anime_upscaler) by [shangar21](https://github.com/shangar21)
- [Upscayl](https://github.com/upscayl/upscayl) by [Nayam Amarshe](https://github.com/NayamAmarshe) and [TGS963](https://github.com/TGS963)

## Acknowledgement

tbd