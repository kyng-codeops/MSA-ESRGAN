import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F


@MODEL_REGISTRY.register()
class RealESRNetModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRNetModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)

            # ----------------------- Color jitter augmentation (GPU) ----------------------- #
            # Applies brightness/contrast jitter to GT to improve generalization on flat regions.
            # This teaches the model that flat areas at different brightness levels should stay flat.
            color_jitter_prob = self.opt.get('color_jitter_prob', 0)
            if color_jitter_prob > 0 and np.random.uniform() < color_jitter_prob:
                brightness_range = self.opt.get('brightness_range', [0.9, 1.1])
                contrast_range = self.opt.get('contrast_range', [0.9, 1.1])

                # Brightness: multiplicative adjustment
                brightness = np.random.uniform(brightness_range[0], brightness_range[1])
                self.gt = self.gt * brightness

                # Contrast: adjust deviation from mean
                contrast = np.random.uniform(contrast_range[0], contrast_range[1])
                mean = self.gt.mean()
                self.gt = (self.gt - mean) * contrast + mean

                # Clamp to valid range [0, 1]
                self.gt = torch.clamp(self.gt, 0, 1)

            # USM sharpen the GT images
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # ----------------------- Fake upscale degradation (optional) ----------------------- #
            # Simulates cheap SD-to-HD stretching (e.g., 480->720/1080, 320->480)
            # This degrades by downscaling then upscaling with cheap interpolation
            fake_upscale_prob = self.opt.get('fake_upscale_prob', 0)
            if fake_upscale_prob > 0 and np.random.uniform() < fake_upscale_prob:
                fake_upscale_range = self.opt.get('fake_upscale_range', [1.5, 2.25])
                fake_upscale_modes = self.opt.get('fake_upscale_modes', ['bilinear', 'bicubic', 'nearest'])
                fake_upscale_mode_weights = self.opt.get('fake_upscale_mode_weights', None)
                # Random upscale factor in range
                upscale_factor = np.random.uniform(fake_upscale_range[0], fake_upscale_range[1])
                current_h, current_w = out.size()[2:4]
                # Downsample first (simulate the original SD source)
                down_h, down_w = int(current_h / upscale_factor), int(current_w / upscale_factor)
                if down_h > 4 and down_w > 4:  # ensure minimum size
                    out = F.interpolate(out, size=(down_h, down_w), mode='area')
                    # Cheap upscale back (simulate the fake HD stretch)
                    fake_mode = random.choices(fake_upscale_modes, weights=fake_upscale_mode_weights)[0]
                    out = F.interpolate(out, size=(current_h, current_w), mode=fake_mode)

            # ----------------------- Combing artifact degradation (optional) ----------------------- #
            # Simulates poor deinterlacing where alternating scanlines show slight displacement
            combing_prob = self.opt.get('combing_prob', 0)
            if combing_prob > 0 and np.random.uniform() < combing_prob:
                combing_strength = self.opt.get('combing_strength', [0.5, 2.0])  # pixel shift range
                combing_blend = self.opt.get('combing_blend', 0.3)  # blend factor with adjacent lines

                b, c, h, w = out.size()
                shift_pixels = np.random.uniform(combing_strength[0], combing_strength[1])
                # Normalize shift to [-1, 1] range for grid_sample
                shift_norm = (shift_pixels / w) * 2

                # Create base grid
                theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=out.dtype, device=out.device)
                theta = theta.unsqueeze(0).expand(b, -1, -1)
                grid = F.affine_grid(theta, out.size(), align_corners=False)

                # Shift odd scanlines horizontally
                grid_shifted = grid.clone()
                grid_shifted[:, 1::2, :, 0] += shift_norm  # shift x coordinate of odd rows

                # Apply the shifted sampling
                out_combed = F.grid_sample(out, grid_shifted, mode='bilinear', padding_mode='border', align_corners=False)

                # Optional: blend with vertically adjacent lines to simulate field blending artifacts
                if combing_blend > 0 and h > 2:
                    # Create a blended version where odd lines blend with even neighbors
                    out_blended = out_combed.clone()
                    # Calculate how many odd lines can be safely blended (have both above and below neighbors)
                    n_blend = (h - 1) // 2  # number of odd lines with both neighbors
                    if n_blend > 0:
                        end_idx = 2 * n_blend  # exclusive end index for slicing
                        # odd lines: 1, 3, ..., end_idx-1
                        # above neighbors: 0, 2, ..., end_idx-2
                        # below neighbors: 2, 4, ..., end_idx
                        out_blended[:, :, 1:end_idx:2, :] = (
                            (1 - combing_blend) * out_combed[:, :, 1:end_idx:2, :] +
                            combing_blend * 0.5 * (out_combed[:, :, 0:end_idx-1:2, :] + out_combed[:, :, 2:end_idx+1:2, :])
                        )
                    out = out_blended
                else:
                    out = out_combed

            # ----------------------- Video transcoding degradation (optional) ----------------------- #
            # Simulates color bleed from video transcoding: RGB → YUV (with chroma subsampling) → RGB
            # Common in video re-encoding where chroma is subsampled (4:2:0 or 4:2:2) causing color fringing
            transcode_prob = self.opt.get('transcode_prob', 0)
            if transcode_prob > 0 and np.random.uniform() < transcode_prob:
                transcode_subsample = self.opt.get('transcode_subsample', ['420', '422'])
                transcode_upsample_mode = self.opt.get('transcode_upsample_mode', 'bilinear')

                # Choose random subsampling mode
                subsample_mode = random.choice(transcode_subsample)

                b, c, h, w = out.size()

                # RGB to YCbCr conversion matrix (BT.601 standard, common in older video)
                # Y  =  0.299*R + 0.587*G + 0.114*B
                # Cb = -0.169*R - 0.331*G + 0.500*B + 0.5
                # Cr =  0.500*R - 0.419*G - 0.081*B + 0.5
                rgb_to_ycbcr = torch.tensor([
                    [0.299, 0.587, 0.114],
                    [-0.169, -0.331, 0.500],
                    [0.500, -0.419, -0.081]
                ], dtype=out.dtype, device=out.device)

                # YCbCr to RGB conversion matrix
                ycbcr_to_rgb = torch.tensor([
                    [1.0, 0.0, 1.402],
                    [1.0, -0.344, -0.714],
                    [1.0, 1.772, 0.0]
                ], dtype=out.dtype, device=out.device)

                # Convert to YCbCr
                out_flat = out.permute(0, 2, 3, 1)  # (B, H, W, C)
                ycbcr = torch.matmul(out_flat, rgb_to_ycbcr.T)
                ycbcr[:, :, :, 1:] += 0.5  # offset Cb, Cr
                ycbcr = ycbcr.permute(0, 3, 1, 2)  # (B, C, H, W)

                # Separate Y and chroma
                y_channel = ycbcr[:, 0:1, :, :]
                chroma = ycbcr[:, 1:3, :, :]

                # Subsample chroma based on mode
                if subsample_mode == '420':
                    # 4:2:0: half resolution in both H and W
                    down_h, down_w = h // 2, w // 2
                elif subsample_mode == '422':
                    # 4:2:2: half resolution in W only
                    down_h, down_w = h, w // 2
                else:
                    down_h, down_w = h // 2, w // 2

                if down_h > 2 and down_w > 2:
                    # Downsample chroma (simulates encoder)
                    chroma_down = F.interpolate(chroma, size=(down_h, down_w), mode='area')

                    # Upsample chroma back (simulates decoder with cheap interpolation)
                    chroma_up = F.interpolate(chroma_down, size=(h, w), mode=transcode_upsample_mode)

                    # Reconstruct YCbCr with degraded chroma
                    ycbcr_degraded = torch.cat([y_channel, chroma_up], dim=1)

                    # Convert back to RGB
                    ycbcr_degraded[:, 1:3, :, :] -= 0.5  # remove Cb, Cr offset
                    ycbcr_flat = ycbcr_degraded.permute(0, 2, 3, 1)  # (B, H, W, C)
                    out = torch.matmul(ycbcr_flat, ycbcr_to_rgb.T)
                    out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
                    out = torch.clamp(out, 0, 1)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRNetModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
