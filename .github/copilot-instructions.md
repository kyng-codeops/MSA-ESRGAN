# MSA-ESRGAN Copilot Instructions

## Project Overview
MSA-ESRGAN is a fork of Real-ESRGAN implementing **Multi-Scale Attention** discriminator architecture for improved super-resolution. The generator remains identical to ESRGAN/Real-ESRGAN, but the custom MSA discriminator with Channel/Spatial Attention modules (CSAFM) improves adversarial training quality. This is a **research-focused training codebase** built on BasicSR framework.

## Architecture & Core Components

### Training Pipeline (Two-Stage)
1. **Stage 1: ESRNet pretraining** - L1 loss only to establish base generator
2. **Stage 2: GAN training** - Full adversarial training with perceptual + GAN losses

### Key Discriminators
- `UNetDiscriminator` (in `realesrgan/archs/msa_discriminator_arch.py`): Custom MSA discriminator with CSAFM attention blocks
  - Uses **spectral normalization** for stability
  - Implements ChannelAttention + SpatialAttention + CSAFM fusion modules
  - U-Net structure with skip connections for multi-scale feature extraction
- `UNetDiscriminatorSN`: Spectral-normalized variant for standard training

### Loss Systems (evolving across versions)
- **v3.2 series**: WGAN with gradient penalty (`compute_gradient_penalty` in `realesrgan_model.py`)
- **v5-v7 series**: Relativistic Average Hinge Loss (`hinge_loss.py`) - preferred for stability
  - No gradient penalty needed with spectral norm
  - Standard in modern GANs (BigGAN, StyleGAN2)
  - Check `RAHINGE_IMPLEMENTATION.md` for migration details

### Perceptual Loss Architecture
- Dual perceptual losses: **VGG19** (texture) + **InceptionV3** (structure)
- **Dynamic Loss Balancing** (`dynamic_loss_balancer.py`): Automatically adjusts VGG19/Inception weight ratios during training
  - Solves magnitude imbalance (VGG19 ~0.1 vs Inception ~0.002)
  - Configure in `.yml` with `dynamic_balance_opt` section
  - See `docs/dynamic_loss_balancing.md` for detailed explanation
- Additional specialized losses:
  - `GradientVarianceLoss`: Sharpens boundaries blurred by L1
  - `LPIPSLoss`: Alternative perceptual metric

### Degradation Model (Critical for Real-World SR)
Configs specify extensive degradation pipeline parameters in YAML (e.g., `train_msaesrgan_RaHinge_x4_v7a.yml`):
- **Two-stage degradation**: Each stage applies blur + noise + JPEG compression
- **Fake upscale degradation** (`fake_upscale_prob`): Simulates SD-to-HD cheap interpolation
- **Combing artifacts** (`combing_prob`): Interlacing simulation (disabled in v6/v7 for stability)
- **Noise ranges**: Wide ranges (1-40) accommodate both modern sensors and older devices
- **JPEG ranges**: [15, 95] for heavy compression to high quality

### Data Pipeline
- `RealESRGANDataset` (`realesrgan/data/`): Loads GT images, generates degraded pairs **on-the-fly in GPU tensors**
- Uses BasicSR degradation primitives: `random_mixed_kernels`, `random_add_gaussian_noise_pt`, etc.
- Supports LMDB or disk backends with meta_info.txt file listing
- Custom datasets should follow Real-ESRGAN paper/README patterns for meta_info.txt and directory structure

## Development Workflows

### Training Commands
```bash
# ALWAYS activate cuda-torch environment first
conda activate cuda-torch

# Single GPU training
python realesrgan/train.py -opt options/train_msaesrgan_RaHinge_x4_v7a.yml

# Auto-detect multi-GPU (set num_gpu: auto in yml)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/your_config.yml
```

### Configuration Patterns (`.yml` structure)
- **Experiment naming**: `name: MSA-ESRGAN-<loss_type>_x4_<dataset>-<version>` (e.g., `MSA-ESRGAN-RaHinge_x4_ip14v33-v7a`)
- **Loss hierarchy**: Three-level system
  1. **Pixel loss** (L1): Reconstruction anchor (~1.0 weight, rarely change)
  2. **Perceptual losses**: Dynamic balancer maintains internal VGG19↔Inception 1:1 ratio
  3. **GAN loss**: Texture refinement (0.08-0.10 typical for Hinge)
- **Learning rates**: Generator (8e-5), Discriminator (2x generator, but v7a uses 5e-6 due to gradient spikes)
- **Resume training**: Use `resume_state` path pointing to `.state` file in `experiments/<name>/training_states/`

### Critical: Updating Learning Rates in Resumed Training

**When changing LR in config for resumed training**, optimizer state must be updated:

1. **Update the `.state` file** with new learning rates:
   ```bash
   python scripts/update_state_lrs.py \
     -state experiments/<name>/training_states/<iter>.state \
     -opt options/your_config.yml
   ```
   Example:
   ```bash
   python scripts/update_state_lrs.py \
     -state experiments/MSA-ESRGAN-RaHinge_x4_ip14v33-v7a/training_states/151560.state \
     -opt options/train_msaesrgan_RaHinge_x4_v7a.yml
   ```
   This creates `<iter>.state.updated` with LRs from your config, adjusted for current iteration and scheduler

2. **Update config** to point to `.updated` file:
   ```yaml
   path:
     resume_state: experiments/<name>/training_states/<iter>.state.updated
   ```

3. **Why this is required**:
   - Optimizer state (Adam momentum, etc.) stores LR internally
   - Changing LR in YAML alone won't update optimizer state
   - Training will use old LR from state file, ignoring YAML config
   - Script reads YAML config and applies scheduler logic to compute correct LR at current iteration

**Common mistake**: Setting `resume_state` to plain `.state` after changing LR in config - this silently uses the old LR from the checkpoint!

### Validation & Metrics
- Validation runs every `val_freq` iterations (typically 4210 = 1 epoch)
- Metrics computed: LPIPS, NIQE (perceptual quality - no PSNR/SSIM in GAN training)
- Results saved to `experiments/<name>/visualization/` with comparison images

### Monitoring & Debugging Training

**Critical TensorBoard Metrics to Monitor:**
- **l_g_total**: Total generator loss (sum of all components)
  - Should trend downward gradually over training
  - Sudden spikes indicate instability (check discriminator)
- **l_g_pix** (L1 Loss): ~0.05 magnitude typical
  - Stable anchor metric; sudden increases = mode collapse
- **l_g_percep** (VGG19) + **l_g_inception**: ~0.05 combined after dynamic balancing
  - Monitor ratio between them (should approach 1:1 with balancer)
  - Check `weight_vgg19` and `weight_inception` for balancer adjustments
- **l_g_gan**: Generator adversarial loss
  - RaHinge: Expect small negative values (-0.1 to -0.01)
  - WGAN: Meaningful loss magnitudes, adjust weight based on scale
  - **Warning**: Spikes to -700+ indicate discriminator gradient explosion
- **l_d_real** + **l_d_fake**: Discriminator losses
  - Should stay relatively balanced (neither dominating)
  - If l_d_real >> l_d_fake: Discriminator too strong, reduce D learning rate
  - If l_d_fake >> l_d_real: Generator overwhelming D, increase D training
- **D(x)** and **D(G(z))**: Discriminator output statistics
  - Healthy range: D(x) slightly > D(G(z)) but not too far apart
  - D(x) → 1, D(G(z)) → 0: Discriminator winning (reduce optim_d.lr)
  - D(x) ≈ D(G(z)) ≈ 0.5: Training balanced

**Log File Patterns (experiments/<name>/train_<name>.log):**
- Search for "lr:" to verify learning rate schedule changes
- "epoch:" markers show dataset passes
- Loss spikes: Use `grep "l_g_gan" | awk '$NF < -100'` to find gradient explosions
- Memory errors: Reduce `batch_size_per_gpu` or `gt_size` (256→192)

**Training Success Indicators:**
1. **Visual quality progression**: Check `visualization/` folder regularly
   - Early iterations: Blurry but structurally correct
   - Mid-training: Sharper details emerging, some artifacts
   - Late training: Texture refinement, reduced artifacts
2. **LPIPS trending down**: Lower = better perceptual similarity
3. **NIQE stabilization**: ~3-5 typical for good quality (lower not always better)
4. **Loss stability**: Generator losses should decrease smoothly
5. **Discriminator balance**: Neither G nor D should dominate metrics

**Common Warning Signs:**
- **Mode collapse**: l_g_pix suddenly jumps, outputs become uniform
- **Discriminator domination**: D(x)→1, D(G(z))→0, l_g_gan stops improving
- **Generator domination**: D(x)≈D(G(z)), discriminator can't distinguish
- **Checkerboard artifacts**: Increase queue_size or add anti-aliasing
- **Gradient explosion**: l_g_gan spikes to -700+, discriminator LR too high (see v7a fix)

**Discriminator Instability Without Generator Collapse (Case Study: v7a iter 154k):**

A subtle form of training degradation can occur where the discriminator becomes unstable **without** obvious gradient explosion, yet still damages model quality:

**Symptoms observed at iter 151k-154k:**
- Validation metrics peak (LPIPS: 0.0083, NIQE: 6.5672 @ iter 151,560 - best ever)
- Shortly after, `out_d_real` and `out_d_fake` show increased volatility
- At iter 151,800: `out_d_real: 0.761`, `out_d_fake: 0.410` (both positive, discriminator saturating)
- At iter 153,100: `out_d_real: 0.745`, `out_d_fake: 0.800` (**fake scores HIGHER than real!**)
- At iter 153,600: `d_gap: 0.037` (discriminator can't tell real from fake)
- Generator losses remain stable (no obvious spike), **masking the problem**
- Next validation at 155,770: LPIPS: 0.0106 (worse), NIQE: 6.8163 (worse)
- By iter 160,000: LPIPS: 0.0116, NIQE: 6.9303 (continued degradation)

**Root Cause - Discriminator Saturation:**
- Discriminator outputs drifting toward extreme values (large positive/negative)
- Loss becomes numerically stable (Hinge Loss plateaus) so no spike appears in `l_d_gan`
- Discriminator provides weak/misleading gradients to generator
- Generator continues training with poor adversarial signal → quality degrades
- This is **more insidious than gradient explosion** because logs look "fine"

**How to Detect:**
1. **Watch `out_d_real` and `out_d_fake`** in logs (not just loss values):
   - Healthy: Both negative or slightly positive, with clear gap (e.g., -0.8 vs -1.4)
   - Warning: Both climb toward +1 or higher (discriminator becoming overconfident)
   - Danger: Values approaching each other (discriminator confused)
2. **Monitor validation metrics trend**:
   - If LPIPS stops improving or reverses after previously decreasing → discriminator issue
   - Compare visual quality in `visualization/` folder before/after suspected iteration
3. **Check `d_gap` metric**: `d_gap = |out_d_real - out_d_fake|`
   - Healthy: 0.3-0.7 (clear discrimination with room to improve)
   - Too high (>1.2): Discriminator too strong, generator struggles
   - Too low (<0.15): Discriminator confused, generator getting misleading gradients

**Recovery Strategies:**
1. **Revert to checkpoint before saturation** (e.g., load iter 151,560 in this case)
2. **Reduce discriminator learning rate** by 2-5x (e.g., 1e-5 → 5e-6 or lower)
3. **Increase queue_size** to provide more diversity to discriminator
4. **Consider reducing `num_feat` in discriminator** if repeatedly overshooting (128→64)
5. **Add discriminator dropout** (0.1-0.2) to prevent overconfident predictions

**Lesson**: Monitor **discriminator output values**, not just loss magnitudes. Hinge Loss can hide saturation issues.

### Model Checkpoints
- Generator: `experiments/<name>/models/net_g_<iter>.pth`
- Discriminator: `experiments/<name>/models/net_d_<iter>.pth` (only for resuming training)
- EMA models: Exponential moving average weights (`ema_decay: 0.999`)
- Pretrained ESRNet: Load with `pretrain_network_g` path in config

### Inference
```bash
# ALWAYS activate cuda-torch environment first
conda activate cuda-torch

python inference_realesrgan.py -n <model_name> -i inputs/ -o results/ --outscale 4
# Add --face_enhance for GFPGAN face restoration (real faces only, not anime)
# Add --fp32 if CPU inference fails with Half precision errors
```

## Project-Specific Conventions

### File Organization
- `realesrgan/archs/`: Network architectures (discriminators, generators)
- `realesrgan/losses/`: Custom loss implementations
- `realesrgan/models/`: Training models (esp. `realesrgan_model.py` - core training logic)
- `realesrgan/data/`: Dataset classes
- `options/`: YAML configs for all experiments (version-named)
- `experiments/`: Training outputs (logs, models, visualizations, states)

### Code Patterns
- **Registry system**: All components registered via `@ARCH_REGISTRY`, `@MODEL_REGISTRY`, `@DATASET_REGISTRY` decorators
- **Loss routing in `optimize_parameters`**: Detects loss type by class name (e.g., checks `'Hinge' in class.__name__`)
- **Device handling**: Training uses `.cuda()` extensively; GPU tensors assumed throughout
- **Config access**: Use `opt.get('key', default)` pattern for optional config parameters

### Version Evolution Notes
- **v3.2 series**: WGAN-GP era, includes gradient penalty computation
- **v5-v7 series**: Migrated to RaHinge loss, removed GP code paths
- **v6/v7 differences**: v6 had combing artifacts enabled, v7/v7a disabled (`combing_prob: 0.0`)
- **v7a adjustments**: Lowered discriminator LR (1e-5 → 5e-6) to prevent gradient explosions seen at iter 331.9k

### Critical Implementation Details
- **Spectral norm + Beta1=0**: Common pairing for discriminator stability (see `optim_d: betas: [0.0, 0.99]`)
- **Queue-based training**: Uses `queue_size: 180` for discriminator input diversity
- **USM sharpening flags**: `l1_gt_usm`, `percep_gt_usm`, `gan_gt_usm` control selective GT sharpening per loss
- **Gradient penalty removal**: When using Hinge Loss, DO NOT apply WGAN gradient penalty (incompatible)

## Hardware & VRAM Considerations

**GPU Memory Requirements:**
- **Minimum**: 12GB VRAM (single GPU, reduced settings)
- **Recommended**: 24GB VRAM (RTX 3090/4090, A5000)
- **Optimal**: 40GB+ VRAM (A100, allows larger batches)

**VRAM-Parameter Trade-offs:**

| Configuration | VRAM Usage | Impact |
|---------------|------------|--------|
| `batch_size_per_gpu: 5` | ~22GB | Standard v7a setting, good diversity |
| `batch_size_per_gpu: 3` | ~14GB | Reduced diversity per iteration |
| `gt_size: 256` | Baseline | Standard crop size for training |
| `gt_size: 192` | -30% VRAM | Smaller crops, less context but faster |
| `num_feat: 128` (discriminator) | High | Better discrimination, more VRAM |
| `num_feat: 64` (discriminator) | -40% VRAM | v7a setting, prevents D dominance |
| `queue_size: 180` | Moderate | Discriminator input diversity buffer |

**VRAM Optimization Strategy:**
1. **If OOM (Out of Memory)**:
   - Reduce `batch_size_per_gpu` (5→3→2)
   - Reduce `gt_size` (256→192)
   - Reduce discriminator `num_feat` (128→64)
   - Check for memory leaks in custom losses

2. **Stability vs Capacity Trade-offs** (Historical Context):
   - **v3.2-v5**: Used `num_feat: 128` discriminator with WGAN-GP
     - Required smaller batches (batch_size: 3) due to VRAM limits
     - Gradient penalty computation added VRAM overhead
   - **v7a**: Reduced to `num_feat: 64` discriminator
     - Allowed `batch_size: 5` with better diversity
     - RaHinge Loss + spectral norm (no GP) freed VRAM
     - Side benefit: Weaker discriminator improved G/D balance

3. **Multi-GPU Training**:
   - Set `num_gpu: auto` in config to auto-detect GPUs
   - Effective batch size = `batch_size_per_gpu × num_gpu`
   - Linear scaling up to 4 GPUs; diminishing returns beyond 8

4. **Batch Size Impact on Training**:
   - **Larger batches** (5+): More stable gradients, better diversity per iteration
   - **Smaller batches** (2-3): Noisier gradients, may need LR adjustment
   - **Critical for GAN training**: Queue diversity matters more than batch size
     - `queue_size: 180` maintains diversity even with smaller batches

**Inference VRAM:**
- FP16 (default): ~2-4GB for 4K input
- FP32 (--fp32 flag): ~4-8GB for 4K input
- Tiling (--tile flag): Enables processing of >4K images on low VRAM

## Installation & Dependencies

### Conda Environment Setup
**CRITICAL**: This project requires a CUDA-enabled conda environment:

```bash
# Activate the cuda-torch conda environment BEFORE any operations
conda activate cuda-torch
```

**Important Notes:**
- **DO NOT use the `(base)` conda environment** - it lacks CUDA-enabled PyTorch
- **ALL commands** (training, inference, testing, package installation) require `cuda-torch` environment
- The `cuda-torch` environment contains CUDA-enabled PyTorch and all GPU dependencies
- If you see `(base)` in the terminal prompt, you're in the wrong environment

### Package Installation
Built on BasicSR framework (install within `cuda-torch` environment):
```bash
conda activate cuda-torch
pip install basicsr facexlib gfpgan
pip install -r requirements.txt
python setup.py develop  # Editable install for development
```

**Environment Requirements:**
- **CUDA-enabled PyTorch environment required** for all training, inference, and testing
- Tests use GPU operations (`.cuda()` calls throughout)
- CPU-only environments not supported for this project

## Testing Framework

### Structure & Organization
- **Framework**: Python `unittest` (NOT pytest)
- **Location**: `tests/` directory with pattern `test_*.py`
- **Test Data**: `tests/data/` contains YAML configs, sample images (GT/LQ), and LMDB databases
- **Execution**: Run individual test files directly: `python tests/test_<module>.py`
- **CUDA Requirement**: All tests require CUDA-enabled PyTorch environment

### Test File Organization
```
tests/
├── test_hinge_loss.py           # Loss function tests
├── test_lpips_loss.py           # LPIPS loss tests
├── test_discriminator_arch.py  # MSA discriminator architecture tests
├── test_model.py                # RealESRGAN/ESRNet model tests
├── test_dataset.py              # Dataset loading/degradation tests
├── test_utils.py                # Utility function tests
├── test_validation_metrics.py  # LPIPS/NIQE metric tests
└── data/
    ├── test_realesrgan_model.yml      # Config for GAN model tests
    ├── test_realesrnet_model.yml      # Config for ESRNet tests
    ├── test_realesrgan_dataset.yml    # Config for dataset tests
    ├── gt/, lq/                        # Sample image directories
    ├── gt.lmdb/, lq.lmdb/              # LMDB database samples
    ├── meta_info_gt.txt                # GT image metadata
    └── meta_info_pair.txt              # Paired GT-LQ metadata
```

### Test Patterns & Conventions

**1. Test Class Structure** (from `test_hinge_loss.py`):
```python
import torch
import unittest
from realesrgan.losses import HingeLoss

class TestHingeLoss(unittest.TestCase):
    """Test suite for HingeLoss and RelativisticHingeLoss."""

    def test_hinge_loss_initialization(self):
        """Test HingeLoss initialization with default parameters."""
        loss_fn = HingeLoss(relativistic=True, loss_weight=0.1)

        self.assertEqual(loss_fn.__class__.__name__, 'HingeLoss')
        self.assertEqual(loss_fn.loss_weight, 0.1)
        self.assertTrue(loss_fn.relativistic)

    def test_hinge_loss_cuda(self):
        """Test HingeLoss works on CUDA if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        loss_fn = HingeLoss(relativistic=True, loss_weight=1.0)
        real_pred = torch.rand(4).cuda()
        fake_pred = torch.rand(4).cuda()

        loss = loss_fn(real_pred, fake_pred, is_disc=True)

        self.assertEqual(loss.device.type, 'cuda')
        self.assertTrue(torch.isfinite(loss))

if __name__ == '__main__':
    unittest.main()
```

**2. Architecture Testing Pattern** (from `test_discriminator_arch.py`):
```python
class TestUNetDiscriminator(unittest.TestCase):

    def test_unetdiscriminator_relativistic(self):
        """Test arch: UNetDiscriminator (relativistic)."""
        net = UNetDiscriminator(num_in_ch=3, num_feat=4, skip_connection=True)
        net.eval()  # Set to eval mode to avoid batch norm issues
        img = torch.rand((2, 3, 32, 32), dtype=torch.float32)  # Batch size >= 2
        output = net(img)

        # Check output shape: (batch_size, 1, height, width)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 1)
        self.assertGreater(output.shape[2], 0)

    def test_unetdiscriminator_gradient_flow(self):
        """Test that gradients flow properly through the discriminator."""
        net = UNetDiscriminator(num_in_ch=3, num_feat=4, skip_connection=True)
        net.train()
        img = torch.rand((2, 3, 32, 32), requires_grad=True)

        output = net(img)
        loss = output.mean()
        loss.backward()

        self.assertIsNotNone(img.grad)
        self.assertFalse(torch.all(img.grad == 0))
```

**3. Model Testing with YAML Configs** (from `test_model.py`):
```python
class TestRealESRNetModel(unittest.TestCase):

    def test_realesrnet_model_initialization(self):
        """Test RealESRNetModel initialization and attributes."""
        with open('tests/data/test_realesrnet_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRNetModel(opt)

        self.assertEqual(model.__class__.__name__, 'RealESRNetModel')
        self.assertIsInstance(model.net_g, RRDBNet)
        self.assertIsInstance(model.cri_pix, L1Loss)
```

### Guidelines for New Tests

**When adding custom losses:**
1. Create `tests/test_<loss_name>_loss.py`
2. Test initialization with various parameters
3. Test forward pass with random tensors (check scalar output, finite values)
4. Test identical inputs (should return ~0 for distance metrics)
5. Test loss weight application (verify scaling)
6. Test CUDA compatibility (`if torch.cuda.is_available()`)
7. Test gradient flow (`requires_grad=True`, check `.grad` exists)
8. Test batch size consistency (try multiple batch sizes)

**When adding discriminator architectures:**
1. Create test class in `tests/test_discriminator_arch.py`
2. Test output shape: `(batch_size, 1, H, W)` for UNet-style
3. Test with batch_size >= 2 (avoid batch norm issues)
4. Test gradient flow in train mode
5. Test both eval and train modes
6. Test CUDA compatibility if available

**When adding model changes:**
1. Add test methods to `tests/test_model.py`
2. Create YAML config in `tests/data/` if needed
3. Test initialization, feed_data, optimization step, validation
4. Use small dimensions (e.g., `gt_size: 32`, `num_feat: 4`) for fast tests
5. Test with minimal degradation options for speed

**Common Test Assertions:**
- `self.assertEqual(tensor.shape, expected_shape)` - Shape validation
- `self.assertTrue(torch.isfinite(loss))` - Check for NaN/Inf
- `self.assertGreater(loss.item(), 0)` - Loss positivity
- `self.assertIsNotNone(tensor.grad)` - Gradient existence
- `self.assertEqual(loss.device.type, 'cuda')` - Device validation
- `self.assertAlmostEqual(a, b, places=5)` - Numerical precision

### Running Tests

**Individual test file:**
```bash
conda activate cuda-torch
python tests/test_hinge_loss.py
```

**Specific test method:**
```bash
conda activate cuda-torch
python tests/test_hinge_loss.py TestHingeLoss.test_hinge_loss_cuda
```

**All tests (from project root):**
```bash
conda activate cuda-torch
# Run all test files manually
for test in tests/test_*.py; do python "$test"; done
```

**Note**: Project uses unittest framework, not pytest. Tests require CUDA-enabled environment.

## Key Documentation Files
- `RAHINGE_IMPLEMENTATION.md`: Migration guide from WGAN-GP to Hinge Loss
- `docs/dynamic_loss_balancing.md`: Dual perceptual loss balancing system
- `docs/Training.md`: Dataset preparation, two-stage training procedures
- `docs/hierarchical_loss_balancing.md`: Three-level loss hierarchy strategy
- Config comments (e.g., lines 148-190 in `train_msaesrgan_RaHinge_x4_v7a.yml`): Inline tuning guidelines

## Common Pitfalls
- **Gradient explosions**: Monitor discriminator gradients; reduce D learning rate if l_g_gan spikes to -700+ (see v7a fix)
- **Perceptual imbalance**: Don't manually tune VGG19/Inception weights; use dynamic balancer instead
- **Loss compatibility**: WGAN-GP and Hinge Loss are mutually exclusive; check model version carefully
- **CPU inference**: Add `--fp32` flag if encountering "slow_conv2d_cpu not implemented for Half" error
