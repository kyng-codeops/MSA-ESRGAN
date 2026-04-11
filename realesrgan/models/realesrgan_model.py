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


def compute_gradient_penalty(discriminator, real_data, fake_data, weight=10.0):
    """Compute gradient penalty for WGAN-GP with proper device handling.

    Args:
        discriminator: The discriminator network
        real_data: Real images tensor (on GPU)
        fake_data: Fake images tensor (on GPU)
        weight: Gradient penalty weight (lambda)

    Returns:
        Gradient penalty loss value
    """
    batch_size = real_data.size(0)
    device = real_data.device

    # Generate random interpolation weight on the same device as real_data
    alpha = torch.rand(batch_size, 1, 1, 1, device=device, dtype=real_data.dtype)

    # Create interpolated images
    interpolates = alpha * real_data + (1.0 - alpha) * fake_data
    interpolates.requires_grad_(True)

    # Get discriminator output for interpolated images
    disc_interpolates = discriminator(interpolates)

    # Compute gradients with respect to interpolated images
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten and compute gradient norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Gradient penalty: (||grad|| - 1)^2
    gradient_penalty = weight * torch.mean((gradient_norm - 1.0) ** 2)

    return gradient_penalty


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

        # Initialize LPIPS loss
        if train_opt.get('lpips_opt'):
            self.cri_lpips = build_loss(train_opt['lpips_opt']).to(self.device)
        else:
            self.cri_lpips = None

        # Initialize InceptionV3 perceptual loss
        if train_opt.get('inception_opt'):
            self.cri_inception = build_loss(train_opt['inception_opt']).to(self.device)
        else:
            self.cri_inception = None

        # Initialize Dynamic Loss Balancer for perceptual losses
        if train_opt.get('dynamic_balance_opt'):
            from realesrgan.losses.dynamic_loss_balancer import DynamicLossBalancer
            balance_opt = train_opt['dynamic_balance_opt']

            # Extract loss names to balance (e.g., ['perceptual', 'inception'])
            loss_names = balance_opt.get('loss_names', ['perceptual', 'inception'])

            # Extract initial weights from individual loss configs
            initial_weights = {}
            for name in loss_names:
                if name == 'perceptual' and self.cri_perceptual:
                    initial_weights[name] = train_opt.get('perceptual_opt', {}).get('perceptual_weight', 1.0)
                elif name == 'inception' and self.cri_inception:
                    initial_weights[name] = train_opt.get('inception_opt', {}).get('perceptual_weight', 1.0)

            self.loss_balancer = DynamicLossBalancer(
                loss_names=loss_names,
                initial_weights=initial_weights,
                balance_method=balance_opt.get('method', 'loss_ratio'),
                momentum=balance_opt.get('momentum', 0.9),
                update_freq=balance_opt.get('update_freq', 10),
                target_ratio=balance_opt.get('target_ratio', None)
            ).to(self.device)
        else:
            self.loss_balancer = None

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
                out_combed = F.grid_sample(
                    out,
                    grid_shifted,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False)

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
                            combing_blend * 0.5 *
                            (out_combed[:, :, 0:end_idx - 1:2, :] + out_combed[:, :, 2:end_idx + 1:2, :])
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

            # Dynamic perceptual loss balancing
            if self.loss_balancer is not None and (self.cri_perceptual or self.cri_inception):
                # Collect raw perceptual losses (unweighted)
                perceptual_losses = {}

                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                    if l_g_percep is not None:
                        # Remove the weight already applied, balancer will reweight
                        percep_weight = self.opt['train'].get('perceptual_opt', {}).get('perceptual_weight', 1.0)
                        perceptual_losses['perceptual'] = l_g_percep / percep_weight
                        loss_dict['l_g_percep'] = l_g_percep  # Log original weighted value

                if self.cri_inception:
                    l_g_inception = self.cri_inception(self.output, percep_gt)
                    # Remove the weight already applied
                    incep_weight = self.opt['train'].get('inception_opt', {}).get('perceptual_weight', 1.0)
                    perceptual_losses['inception'] = l_g_inception / incep_weight
                    loss_dict['l_g_inception'] = l_g_inception  # Log original weighted value

                # Apply dynamic balancing
                balanced_loss, weighted_losses = self.loss_balancer(perceptual_losses, model=self.net_g)
                l_g_total += balanced_loss

                # Log dynamic weights
                current_weights = self.loss_balancer.get_current_weights()
                for name, weight in current_weights.items():
                    loss_dict[f'dyn_weight_{name}'] = torch.tensor(weight, device=self.device)

                # Log effective contributions
                contributions = self.loss_balancer.get_effective_contributions()
                for name, contrib in contributions.items():
                    loss_dict[f'dyn_contrib_{name}'] = torch.tensor(contrib, device=self.device)
            else:
                # Standard fixed-weight perceptual losses
                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style
                # InceptionV3 perceptual loss
                if self.cri_inception:
                    l_g_inception = self.cri_inception(self.output, percep_gt)
                    l_g_total += l_g_inception
                    loss_dict['l_g_inception'] = l_g_inception

            # LPIPS loss
            if self.cri_lpips:
                l_g_lpips = self.cri_lpips(self.output, percep_gt)
                l_g_total += l_g_lpips
                loss_dict['l_g_lpips'] = l_g_lpips
            # gan loss
            fake_g_pred = self.net_d(self.output)

            # Check if using relativistic loss (RaGAN, RaHinge) or hinge loss
            is_relativistic = (
                hasattr(self.cri_gan, '__class__') and
                ('Relativistic' in self.cri_gan.__class__.__name__ or 'Hinge' in self.cri_gan.__class__.__name__)
            )
            is_hinge = hasattr(self.cri_gan, '__class__') and 'Hinge' in self.cri_gan.__class__.__name__

            if is_relativistic or is_hinge:
                # Relativistic loss or Hinge loss requires both real and fake predictions
                real_d_pred = self.net_d(gan_gt)
                l_g_gan = self.cri_gan(real_d_pred, fake_g_pred, is_disc=False)
            else:
                # Non-relativistic: generator wants fake predictions to be classified as real
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        # Discriminator loss
        real_d_pred = self.net_d(gan_gt)
        fake_d_pred = self.net_d(self.output.detach().clone())

        # Check if using relativistic loss (RaGAN, RaHinge) or hinge loss
        is_relativistic = (
            hasattr(self.cri_gan, '__class__') and
            ('Relativistic' in self.cri_gan.__class__.__name__ or 'Hinge' in self.cri_gan.__class__.__name__)
        )
        is_hinge = hasattr(self.cri_gan, '__class__') and 'Hinge' in self.cri_gan.__class__.__name__

        if is_relativistic or is_hinge:
            # Relativistic or Hinge loss - no GP needed (Hinge is stable without GP)
            l_d_gan = self.cri_gan(real_d_pred, fake_d_pred, is_disc=True)
        else:
            # Non-relativistic: compute base loss + GP separately
            l_d_gan = self.cri_gan(real_d_pred, True) + self.cri_gan(fake_d_pred, False)

            # Add gradient penalty only for WGAN-GP (not needed for Hinge Loss)
            gp_lambda = self.opt['train'].get('gp_lambda', 0)
            if gp_lambda > 0:
                gp = compute_gradient_penalty(
                    self.net_d, gan_gt, self.output.detach().clone(),
                    weight=gp_lambda
                )
                l_d_gan = l_d_gan + gp
                loss_dict['l_d_gp'] = gp

        loss_dict['l_d_gan'] = l_d_gan
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_gan.backward()

        # Gradient clipping for discriminator stability
        # Prevents extreme gradient spikes while allowing normal gradient flow
        max_grad_norm = self.opt['train'].get('max_grad_norm_d', None)
        if max_grad_norm is not None:
            # Log gradient norm before clipping for monitoring
            grad_norm_before = torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), max_grad_norm)
            loss_dict['grad_norm_d'] = grad_norm_before

        self.optimizer_d.step()

        # No weight clipping needed for Hinge Loss (only for vanilla WGAN without GP)
        # Spectral normalization handles discriminator regularization for Hinge Loss

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
