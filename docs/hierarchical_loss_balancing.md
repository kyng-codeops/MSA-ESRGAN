# Hierarchical Loss Balancing in MSA-ESRGAN v7

## Overview

V7 introduces **two-level hierarchical balancing**:

1. **Level 1 (Internal)**: VGG19 ↔ Inception - **Automatic** via dynamic balancer
2. **Level 2 (External)**: L1 ↔ Perceptuals ↔ GAN - **Manual** tuning

## The Balancing Hierarchy

```
Total Generator Loss
│
├─ L1 Loss (weight: 1.0)
│  └─ Pixel-level reconstruction anchor
│     Magnitude: ~0.05
│     Contribution: ~0.05 (55%)
│
├─ Perceptual Losses (combined)
│  │  Initial weight: 1.0 each
│  │  Combined magnitude: ~0.05
│  │  Contribution: ~0.05 (55%)
│  │
│  ├─ VGG19 (dynamically balanced)
│  │  └─ Loss: ~0.10 × Dynamic weight: ~0.01 = ~0.025
│  │
│  └─ Inception (dynamically balanced)
│     └─ Loss: ~0.002 × Dynamic weight: ~0.50 = ~0.025
│
│     ⚡ AUTOMATIC BALANCING ⚡
│     Dynamic balancer adjusts weights to maintain 1:1 ratio
│     (VGG19 ~0.025 = Inception ~0.025)
│
└─ GAN Loss (weight: 0.08)
   └─ Adversarial texture refinement
      Magnitude: ~0.08
      Contribution: ~0.006 (7%)
```

## Current Effective Balance

| Loss Type | Weight | Magnitude | Contribution | % of Total |
|-----------|--------|-----------|--------------|------------|
| **L1** | 1.0 | ~0.05 | **0.05** | 55% |
| **Perceptuals** (total) | - | - | **0.05** | 55% |
| ├─ VGG19 | ~0.01 (dynamic) | ~0.10 | 0.025 | 27.5% |
| └─ Inception | ~0.50 (dynamic) | ~0.002 | 0.025 | 27.5% |
| **GAN** | 0.08 | ~0.08 | **0.006** | 7% |

**Key Insight**: L1 and Perceptuals dominate equally (~55% each), GAN provides subtle refinement (~7%)

## How the Dynamic Balancer Works

### Problem (v6):
```
VGG19:     0.01  × 0.10  = 0.001   (125× larger)
Inception: 0.004 × 0.002 = 0.000008
```
→ VGG19 dominates, Inception barely contributes

### Solution (v7):
```
VGG19:     ~0.01 × 0.10  = 0.001   (balanced)
Inception: ~0.50 × 0.002 = 0.001   (balanced)
```
→ Both contribute equally, balancer adjusts Inception weight automatically

### Configuration:
```yaml
dynamic_balance_opt:
  loss_names: ['perceptual', 'inception']
  target_ratio:
    perceptual: 1.0   # Reference
    inception: 1.0    # Equal to reference
  momentum: 0.9       # Smooth updates
  update_freq: 10     # Update every 10 iters
```

## Manual Balancing Strategies

### Strategy 1: Increase Total Perceptual Influence

**Goal**: Richer textures and structure

**Change**:
```yaml
perceptual_weight: 2.0   # Was 1.0
inception_weight: 2.0    # Was 1.0
```

**Result**:
- L1: ~0.05 (33%)
- Perceptuals: ~0.10 (66%) ← doubled
- GAN: ~0.006 (4%)

**Effect**: More feature-driven, less pixel-perfect

---

### Strategy 2: Increase GAN Influence

**Goal**: Sharper, more textured outputs

**Change**:
```yaml
gan_opt.loss_weight: 0.12  # Was 0.08
```

**Result**:
- L1: ~0.05 (45%)
- Perceptuals: ~0.05 (45%)
- GAN: ~0.01 (10%) ← increased

**Effect**: Sharper details, more texture
**Warning**: Monitor for hallucinations if too high

---

### Strategy 3: More GT Fidelity

**Goal**: Safer, more faithful to ground truth

**Change**:
```yaml
perceptual_weight: 0.5   # Was 1.0
inception_weight: 0.5    # Was 1.0
gan_opt.loss_weight: 0.04  # Was 0.08
```

**Result**:
- L1: ~0.05 (67%) ← dominant
- Perceptuals: ~0.025 (33%)
- GAN: ~0.003 (5%)

**Effect**: More GT-faithful, potentially less detailed

---

### Strategy 4: Prefer Structure Over Texture

**Goal**: Better geometry, less texture emphasis

**Change**:
```yaml
dynamic_balance_opt:
  target_ratio:
    perceptual: 0.5   # VGG19 reduced
    inception: 1.0    # Inception preferred
```

**Result**:
- Within perceptuals: Inception contributes 2× more than VGG19
- Overall balance: L1 ~0.05, Perceptuals ~0.05, GAN ~0.006

**Effect**: Better structural/geometric preservation

## When to Adjust What

### Symptoms & Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| **Over-smooth outputs** | L1 too dominant | Increase perceptual weights (1.0 → 2.0) |
| **Too much texture noise** | GAN too strong | Decrease gan_opt.loss_weight (0.08 → 0.04) |
| **Missing fine details** | GAN too weak | Increase gan_opt.loss_weight (0.08 → 0.12) |
| **Hallucinations/artifacts** | GAN too strong | Decrease gan_opt.loss_weight (0.08 → 0.04) |
| **Poor structure** | Inception under-weighted | Change target_ratio: {perceptual: 0.5, inception: 1.0} |
| **Loss texture quality** | VGG19 under-weighted | Change target_ratio: {perceptual: 1.0, inception: 0.5} |
| **Training unstable** | Perceptuals too high | Reduce both weights (1.0 → 0.5) |

## Monitoring in TensorBoard

### Key Metrics to Watch

**Dynamic Balancing (Level 1)**:
```
dyn_weight_perceptual:   ~0.01  (VGG19 weight)
dyn_weight_inception:    ~0.50  (Inception weight, grows automatically)
dyn_contrib_perceptual:  ~0.025 (should match inception)
dyn_contrib_inception:   ~0.025 (should match perceptual)
```

**Total Loss Components**:
```
l_g_pix:      ~0.05  (L1 loss)
l_g_percep:   ~0.10  (raw VGG19, before dynamic weighting)
l_g_inception:~0.002 (raw Inception, before dynamic weighting)
l_g_gan:      ~0.08  (GAN loss)
```

### Healthy Training Signs

✅ **Good**:
- `dyn_contrib_perceptual` ≈ `dyn_contrib_inception` (balanced)
- `dyn_weight_inception` gradually increases to ~0.50
- All losses decreasing smoothly
- `d_gap` oscillating around 0 (discriminator balanced)

⚠️ **Warning**:
- `dyn_weight_*` oscillating wildly → Increase momentum (0.9 → 0.95)
- `dyn_contrib_*` diverging → Check if one loss is collapsing
- `l_g_gan` very large → GAN too strong, reduce weight
- `d_gap` very high → Discriminator overpowering generator

## Why Not Balance Everything Dynamically?

**Could you extend dynamic balancing to L1 and GAN?**

Technically yes:
```yaml
dynamic_balance_opt:
  loss_names: ['pixel', 'perceptual', 'inception', 'gan']
  target_ratio: {pixel: 1.0, perceptual: 1.0, inception: 1.0, gan: 0.1}
```

**But DON'T do this because:**

1. **Different roles**: L1 (reconstruction), Perceptuals (features), GAN (adversarial)
   - They serve fundamentally different purposes
   - You WANT asymmetric contributions

2. **Loss of control**: Over-automation hides important training signals
   - Manual tuning lets you steer training direction
   - Different stages may need different balances

3. **Dynamic balancer best for similar losses**:
   - VGG19 and Inception are both perceptual feature extractors
   - They have the same role, just different feature spaces
   - Balancing them makes sense

4. **Established best practices**:
   - L1/Perceptual dominant (55% each) is well-tested
   - GAN subtle (7%) prevents artifacts
   - Deviating requires careful experimentation

## Recommended Workflow

### Phase 1: Start with v7 defaults
```yaml
pixel_opt.loss_weight: 1.0
perceptual_weight: 1.0
inception_weight: 1.0
gan_opt.loss_weight: 0.08
dynamic_balance_opt.target_ratio: {perceptual: 1.0, inception: 1.0}
```

### Phase 2: Monitor first 10 epochs
- Check TensorBoard
- Verify `dyn_contrib_*` are balanced
- Verify losses decreasing smoothly
- Check validation images

### Phase 3: Adjust if needed
- Too smooth? → Increase GAN weight (0.08 → 0.10)
- Too textured? → Increase perceptual weights (1.0 → 2.0)
- Poor structure? → Adjust target_ratio

### Phase 4: Fine-tune
- Small adjustments (±20% at a time)
- Wait 5-10 epochs between changes
- Document what works in your dataset

## Summary

**Two-Level Balancing**:
1. **Automatic** (VGG19 ↔ Inception): Handled by dynamic balancer
2. **Manual** (L1 ↔ Perceptuals ↔ GAN): You control the high-level balance

**Default v7 Balance**:
- L1 and Perceptuals contribute equally (~55% each)
- VGG19 and Inception balanced internally (27.5% each)
- GAN provides subtle refinement (~7%)

**When to Adjust**:
- Adjust perceptual weights to control total perceptual influence
- Adjust GAN weight to control texture detail
- Adjust target_ratio to prefer VGG19 vs Inception internally
- Keep L1 at 1.0 as stable anchor

**Key Advantage**:
- No more sensitivity to VGG19 vs Inception weight tuning
- Focus on high-level balance (L1 vs Perceptuals vs GAN)
- Both perceptual spaces contribute meaningfully
- Adapts automatically throughout training
