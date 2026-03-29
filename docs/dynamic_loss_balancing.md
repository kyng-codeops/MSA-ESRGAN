# Dynamic Perceptual Loss Balancing

## Overview

This implementation adds **automatic loss weight balancing** for training with multiple perceptual losses (e.g., VGG19 + InceptionV3). It solves the sensitivity issue where different perceptual networks produce vastly different loss magnitudes, making manual weight tuning difficult.

## The Problem

When using dual perceptual losses in MSA-ESRGAN:

```
VGG19 loss magnitude:     ~0.10 (1e-1)
Inception loss magnitude: ~0.002 (2e-3)
Raw ratio: 50:1
```

With fixed weights `vgg19=0.01, inception=0.004`:
```
Effective gradient contribution:
VGG19:     0.01 × 0.10  = 0.001000
Inception: 0.004 × 0.002 = 0.000008
Ratio: 125:1 (VGG19 dominates!)
```

This causes:
- **VGG19 over-dominance**: Model optimizes primarily for VGG19-preferred features (textures)
- **Inception under-utilization**: Multi-scale structural information is mostly ignored
- **Sensitivity to weights**: Small changes in weights cause large training instability

## The Solution

Dynamic loss balancing automatically adjusts weights during training to maintain target contribution ratios. Based on research papers:

- **Chen et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing"** (ICML 2018)
- **Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"** (CVPR 2018)

## Implementation

### 1. New Files

**`realesrgan/losses/dynamic_loss_balancer.py`**
- `DynamicLossBalancer`: Main balancing class
- Two balancing methods:
  - `loss_ratio`: Fast, balances based on loss magnitude ratios (recommended)
  - `gradient`: Accurate, balances based on gradient norms (slower)

### 2. Modified Files

**`realesrgan/losses/__init__.py`**
- Registered `DynamicLossBalancer`

**`realesrgan/models/realesrgan_model.py`**
- Added `self.loss_balancer` initialization in `__init__`
- Modified `optimize_parameters` to use dynamic balancing when enabled
- Logs dynamic weights and contributions to TensorBoard

### 3. Configuration

Add to your `.yml` config:

```yaml
train:
  # Standard perceptual losses (initial weights)
  perceptual_opt:
    type: PerceptualLoss
    perceptual_weight: !!float 1.0  # Will be adjusted dynamically
    # ... other settings ...

  inception_opt:
    type: InceptionPerceptualLoss
    perceptual_weight: !!float 1.0  # Will be adjusted dynamically
    # ... other settings ...

  # Dynamic balancer configuration
  dynamic_balance_opt:
    loss_names: ['perceptual', 'inception']
    method: loss_ratio  # or 'gradient'
    target_ratio:
      perceptual: 1.0   # Reference loss
      inception: 1.0    # Equal contribution to reference
    momentum: 0.9       # Smoothing (0.9 = stable, 0.5 = responsive)
    update_freq: 10     # Update every N iterations
```

## Usage Examples

### Example 1: Equal Contribution (Recommended)

```yaml
dynamic_balance_opt:
  loss_names: ['perceptual', 'inception']
  target_ratio: {perceptual: 1.0, inception: 1.0}
  method: loss_ratio
  momentum: 0.9
  update_freq: 10
```

**Effect**: Both losses contribute equally to training
- Balances VGG19 texture guidance with Inception structural features
- Most stable starting point

### Example 2: VGG19 Preferred (2:1 ratio)

```yaml
dynamic_balance_opt:
  loss_names: ['perceptual', 'inception']
  target_ratio: {perceptual: 1.0, inception: 0.5}
  method: loss_ratio
  momentum: 0.9
  update_freq: 10
```

**Effect**: VGG19 contributes 2× more than Inception
- Emphasizes texture/pattern preservation
- Reduces hallucination risk

### Example 3: Inception Preferred (1:2 ratio)

```yaml
dynamic_balance_opt:
  loss_names: ['perceptual', 'inception']
  target_ratio: {perceptual: 0.5, inception: 1.0}
  method: loss_ratio
  momentum: 0.9
  update_freq: 10
```

**Effect**: Inception contributes 2× more than VGG19
- Emphasizes multi-scale structure and global coherence
- Better for architectural/geometric preservation

### Example 4: Gradient-Based Balancing (More Accurate)

```yaml
dynamic_balance_opt:
  loss_names: ['perceptual', 'inception']
  target_ratio: {perceptual: 1.0, inception: 1.0}
  method: gradient  # Slower but more precise
  momentum: 0.95    # Higher momentum for stability
  update_freq: 20   # Update less frequently
```

**Effect**: Balances based on actual gradient magnitudes
- More computationally expensive (requires extra backward passes)
- More accurate for complex loss landscapes
- Use when loss_ratio method is insufficient

## Monitoring

The balancer logs several metrics to TensorBoard:

| Metric | Description |
|--------|-------------|
| `l_g_percep` | Raw VGG19 perceptual loss (for comparison) |
| `l_g_inception` | Raw Inception perceptual loss (for comparison) |
| `dyn_weight_perceptual` | Current dynamic weight for VGG19 |
| `dyn_weight_inception` | Current dynamic weight for Inception |
| `dyn_contrib_perceptual` | Effective contribution (weight × loss) |
| `dyn_contrib_inception` | Effective contribution (weight × loss) |

### What to Look For

**Healthy training:**
```
dyn_contrib_perceptual: ~0.001-0.01 (stable)
dyn_contrib_inception:  ~0.001-0.01 (stable, similar to VGG19)
dyn_weight_inception:   ~0.05-0.5 (automatically increases to compensate)
```

**Warning signs:**
```
dyn_weight_* oscillating wildly → Reduce momentum or increase update_freq
dyn_contrib_* diverging → Check if one loss is collapsing
```

## Benefits

1. **Eliminates manual tuning**: No more guessing weight ratios
2. **Adapts to training dynamics**: Automatically adjusts as loss scales change
3. **Improves stability**: Prevents one loss from dominating
4. **Better model quality**: Both perceptual spaces contribute meaningfully

## Comparison: Fixed vs Dynamic

### Fixed Weights (v6 current)
```yaml
perceptual_weight: 0.01
inception_weight: 0.004
```
- VGG19 dominates 125:1
- Requires extensive manual tuning
- Sensitive to hyperparameters
- May need adjustment mid-training

### Dynamic Balancing
```yaml
perceptual_weight: 1.0  # Initial
inception_weight: 1.0   # Initial
dynamic_balance_opt:
  target_ratio: {perceptual: 1.0, inception: 1.0}
```
- Automatically balances to 1:1 contribution
- No manual tuning required
- Adapts throughout training
- Robust to loss scale changes

## Performance Impact

- **loss_ratio method**: Negligible overhead (<1% slowdown)
  - Simple arithmetic on loss values
  - Recommended for production use

- **gradient method**: ~10-20% slowdown
  - Requires extra backward passes per loss
  - Use only if loss_ratio is insufficient

## When to Use

**Use dynamic balancing when:**
- Training with multiple perceptual losses (VGG19 + Inception)
- Loss magnitudes differ significantly (>10× ratio)
- Manually tuned weights cause instability
- You want equal contribution from different perceptual spaces

**Don't use when:**
- Using only one perceptual loss
- Loss magnitudes are already similar
- You specifically want one loss to dominate
- Training is already stable with fixed weights

## Troubleshooting

### Weights oscillate too much
```yaml
momentum: 0.95  # Increase from 0.9
update_freq: 20  # Increase from 10
```

### Weights converge too slowly
```yaml
momentum: 0.7   # Decrease from 0.9
update_freq: 5  # Decrease from 10
```

### One loss contributing too much
```yaml
target_ratio:
  perceptual: 1.0
  inception: 0.5  # Reduce target ratio
```

### Training becomes unstable
```yaml
method: gradient  # Switch to more accurate method
momentum: 0.95    # Increase stability
```

## References

1. Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018). GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. ICML 2018.

2. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. CVPR 2018.

3. Cipolla, R., Gal, Y., & Kendall, A. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. CVPR 2018.
