import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from basicsr.losses import build_loss


@MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

        train_opt = opt['train']
        # Add this block to initialize grad_opt
        if train_opt.get('grad_opt'):
            self.cri_grad = build_loss(train_opt['grad_opt']).to(self.device)
        else:
            self.cri_grad = None

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
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
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
                    # Odd lines: 1, 3, 5, ... - the last odd line may not have a below neighbor
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
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
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
        super(RealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # gradient variance loss
            if self.cri_grad:
                l_grad = self.cri_grad(self.output, self.gt)
                loss_dict['l_grad'] = l_grad
                l_g_total += l_grad
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            real_d_pred = self.net_d(gan_gt)
            l_g_gan = self.cri_gan(real_d_pred, fake_g_pred, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        # Discriminator loss (relativistic)
        real_d_pred = self.net_d(gan_gt)
        fake_d_pred = self.net_d(self.output.detach().clone())
        l_d_gan = self.cri_gan(
            real_d_pred, fake_d_pred, is_disc=True,
            real_img=gan_gt, fake_img=self.output.detach().clone(), net_d=self.net_d
        )
        loss_dict['l_d_gan'] = l_d_gan
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_gan.backward()

        self.optimizer_d.step()

        # WGAN regularization
        if getattr(self.cri_gan, 'loss_type', None) == 'wgan':
            clip_value = 0.01  # You can tune this value
            for p in self.net_d.parameters():
                p.data.clamp_(-clip_value, clip_value)
        elif getattr(self.cri_gan, 'loss_type', None) == 'wgan-gp':
            # No weight clipping; gradient penalty is added in the loss
            pass

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # Compute d_gap from already-logged values
        if 'out_d_real' in self.log_dict and 'out_d_fake' in self.log_dict:
            # Use abs() instead of torch.abs() since log_dict contains floats
            self.log_dict['d_gap'] = abs(
                self.log_dict['out_d_real'] - self.log_dict['out_d_fake']
            )
        """
        How It Works for GANs & WGANs:
            d_real and d_fake: Average discriminator predictions for real and fake images
            d_gap: Absolute difference between them
            Added to self.log_dict: Automatically logged to TensorBoard by the parent class

        What to Look For in TensorBoard (GANs):
            d_gap close to 1.0: Discriminator is well-separated (good sign)
            d_gap close to 0: Discriminator can't distinguish real from fake (bad sign)
            d_gap decreasing over time: Discriminator becoming confused (generator winning)
            d_gap increasing over time: Discriminator getting stronger (may overpower generator)

            Ideal Range:
                For stable GAN training, aim for d_gap to stay in the range 0.5-0.8 and remain
                relatively stable over iterations.

        What to Look For in TensorBoard (WGANs):
            d_gap close to 0: Discriminator (critic) is well-balanced (good sign
            d_gap significantly above 0: Critic may be overpowering generator (bad sign)
            d_gap decreasing over time: Critic should slowly decay to 0 (where 0 is perfect generation)

            Scale l_g_gan magnitude to balance with other loss magnitudes. Critic requires gradient magnitude
            scaling both to get started and to actively participate in training. Critic needs this to learn
            from the generator to get started (always start net_d from random because net_d is
            custom fit to net_g at each state -- they are matched pairs).
        """
