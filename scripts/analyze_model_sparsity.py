#!/usr/bin/env python3
"""
Analyze weight sparsity in trained models to gauge learning capacity.

Sparsity analysis helps determine:
- Remaining learning capacity (high sparsity = unused weights)
- Overtraining risk (weights collapsing to similar values)
- Network utilization (effective vs total parameters)

Usage:
    python scripts/analyze_model_sparsity.py -m experiments/MSA-ESRGAN-RaHinge_x4_ip14v33-v7a/models/net_g_884100.pth
    python scripts/analyze_model_sparsity.py -m experiments/MSA-ESRGAN-RaHinge_x4_ip14v33-v7a/models/net_d_884100.pth --threshold 1e-3
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict


def analyze_tensor_sparsity(tensor, thresholds=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]):
    """Analyze sparsity of a tensor at different thresholds."""
    tensor_flat = tensor.abs().flatten()
    total_params = tensor_flat.numel()

    sparsity_info = {}
    for threshold in thresholds:
        near_zero = (tensor_flat < threshold).sum().item()
        sparsity = (near_zero / total_params) * 100
        sparsity_info[threshold] = {
            'count': near_zero,
            'percent': sparsity
        }

    # Statistical measures
    stats = {
        'mean': tensor_flat.mean().item(),
        'std': tensor_flat.std().item(),
        'min': tensor_flat.min().item(),
        'max': tensor_flat.max().item(),
        'median': tensor_flat.median().item(),
        'q25': tensor_flat.quantile(0.25).item(),
        'q75': tensor_flat.quantile(0.75).item(),
    }

    return sparsity_info, stats, total_params


def load_checkpoint(checkpoint_path, param_key='params'):
    """Load model checkpoint, handling different formats."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if param_key in checkpoint:
        state_dict = checkpoint[param_key]
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    return state_dict


def print_layer_analysis(name, sparsity_info, stats, total_params, threshold):
    """Print formatted analysis for a single layer."""
    sparsity_at_threshold = sparsity_info[threshold]['percent']

    print(f"\n{name}")
    print(f"  Total params: {total_params:,}")
    print(f"  Sparsity @ {threshold:.0e}: {sparsity_at_threshold:.2f}%")
    print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
    print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"  Quartiles: Q25={stats['q25']:.6f}, Median={stats['median']:.6f}, Q75={stats['q75']:.6f}")


def analyze_model(checkpoint_path, threshold=1e-4, verbose=False, param_key='params'):
    """Main analysis function."""
    print(f"\n{'='*80}")
    print(f"Weight Sparsity Analysis: {checkpoint_path}")
    print(f"{'='*80}")

    state_dict = load_checkpoint(checkpoint_path, param_key)

    total_model_params = 0
    total_sparse_params = 0
    layer_stats = []

    # Thresholds to analyze
    thresholds = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue

        # Skip non-weight parameters (batch norm stats, etc.)
        if any(skip in name for skip in ['num_batches_tracked', 'running_mean', 'running_var']):
            continue

        sparsity_info, stats, num_params = analyze_tensor_sparsity(param, thresholds)

        total_model_params += num_params
        total_sparse_params += sparsity_info[threshold]['count']

        layer_stats.append({
            'name': name,
            'params': num_params,
            'sparsity': sparsity_info[threshold]['percent'],
            'mean': stats['mean'],
            'std': stats['std'],
            'sparsity_info': sparsity_info,
            'stats': stats
        })

        if verbose:
            print_layer_analysis(name, sparsity_info, stats, num_params, threshold)

    # Sort by sparsity (descending) to find most sparse layers
    layer_stats.sort(key=lambda x: x['sparsity'], reverse=True)

    # Overall statistics
    overall_sparsity = (total_sparse_params / total_model_params) * 100

    print(f"\n{'='*80}")
    print(f"OVERALL MODEL STATISTICS")
    print(f"{'='*80}")
    print(f"Total parameters: {total_model_params:,}")
    print(f"Sparse parameters (< {threshold:.0e}): {total_sparse_params:,}")
    print(f"Overall sparsity: {overall_sparsity:.2f}%")
    print(f"Active parameters: {total_model_params - total_sparse_params:,} ({100-overall_sparsity:.2f}%)")

    # Sparsity at different thresholds
    print(f"\n{'='*80}")
    print(f"SPARSITY AT DIFFERENT THRESHOLDS")
    print(f"{'='*80}")
    for thresh in thresholds:
        count = sum(layer['sparsity_info'][thresh]['count'] for layer in layer_stats)
        percent = (count / total_model_params) * 100
        print(f"  < {thresh:.0e}: {percent:6.2f}% ({count:,} params)")

    # Top 10 most sparse layers
    print(f"\n{'='*80}")
    print(f"TOP 10 MOST SPARSE LAYERS (@ {threshold:.0e})")
    print(f"{'='*80}")
    for i, layer in enumerate(layer_stats[:10], 1):
        print(f"{i:2d}. {layer['name']:50s} {layer['sparsity']:6.2f}% ({layer['params']:,} params)")

    # Top 10 least sparse layers (most utilized)
    print(f"\n{'='*80}")
    print(f"TOP 10 LEAST SPARSE LAYERS (Most Active)")
    print(f"{'='*80}")
    for i, layer in enumerate(reversed(layer_stats[-10:]), 1):
        print(f"{i:2d}. {layer['name']:50s} {layer['sparsity']:6.2f}% ({layer['params']:,} params)")

    # Learning capacity assessment
    print(f"\n{'='*80}")
    print(f"LEARNING CAPACITY ASSESSMENT")
    print(f"{'='*80}")

    if overall_sparsity > 70:
        assessment = "HIGH UNUSED CAPACITY"
        recommendation = "Model has significant room for learning. Consider:\n" \
                        "  - Longer training\n" \
                        "  - Higher learning rates\n" \
                        "  - More complex data augmentation"
    elif overall_sparsity > 40:
        assessment = "MODERATE CAPACITY"
        recommendation = "Model is well-utilized with room to grow. Training appears healthy."
    elif overall_sparsity > 20:
        assessment = "LOW CAPACITY"
        recommendation = "Model is highly utilized. Consider:\n" \
                        "  - Lower learning rates for fine-tuning\n" \
                        "  - Weight decay to prevent overfitting\n" \
                        "  - Monitor validation metrics closely"
    else:
        assessment = "NEAR SATURATION"
        recommendation = "Model weights are densely packed. Risk of:\n" \
                        "  - Overtraining (weights overwriting each other)\n" \
                        "  - Diminishing returns from further training\n" \
                        "  - Consider early stopping or architecture expansion"

    print(f"Assessment: {assessment}")
    print(f"Sparsity: {overall_sparsity:.2f}% (threshold: {threshold:.0e})")
    print(f"\nRecommendation:\n{recommendation}")

    # Weight distribution uniformity check
    all_means = [layer['mean'] for layer in layer_stats]
    all_stds = [layer['std'] for layer in layer_stats]
    mean_of_means = np.mean(all_means)
    std_of_stds = np.mean(all_stds)

    print(f"\n{'='*80}")
    print(f"WEIGHT DISTRIBUTION HEALTH")
    print(f"{'='*80}")
    print(f"Average layer mean: {mean_of_means:.6f}")
    print(f"Average layer std: {std_of_stds:.6f}")

    if std_of_stds < 0.01:
        print("⚠ WARNING: Low weight variance detected. Possible overtraining or vanishing gradients.")
    elif std_of_stds > 0.5:
        print("⚠ WARNING: High weight variance. Check for exploding gradients or unstable training.")
    else:
        print("✓ Weight distribution appears healthy.")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze weight sparsity in trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze generator model at default threshold (1e-4)
  python scripts/analyze_model_sparsity.py -m experiments/MSA-ESRGAN-RaHinge_x4_ip14v33-v7a/models/net_g_884100.pth

  # Analyze discriminator with custom threshold
  python scripts/analyze_model_sparsity.py -m experiments/MSA-ESRGAN-RaHinge_x4_ip14v33-v7a/models/net_d_884100.pth -t 1e-3

  # Verbose output with per-layer details
  python scripts/analyze_model_sparsity.py -m experiments/MSA-ESRGAN-RaHinge_x4_ip14v33-v7a/models/net_g_884100.pth -v

  # Use specific parameter key for checkpoint format
  python scripts/analyze_model_sparsity.py -m model.pth --param-key params_ema
        """
    )

    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('-t', '--threshold', type=float, default=1e-4,
                        help='Threshold for considering weights as sparse (default: 1e-4)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed per-layer analysis')
    parser.add_argument('--param-key', type=str, default='params',
                        help='Parameter key in checkpoint (default: params, can be params_ema, state_dict, etc.)')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1

    analyze_model(args.model, args.threshold, args.verbose, args.param_key)
    return 0


if __name__ == '__main__':
    exit(main())
