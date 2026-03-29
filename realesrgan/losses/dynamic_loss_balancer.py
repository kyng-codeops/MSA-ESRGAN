"""
Dynamic Loss Balancer for Multi-Perceptual Loss Training

Implements adaptive weight balancing for multiple perceptual losses based on:
1. Gradient magnitude normalization (GradNorm-style)
2. Loss magnitude ratio balancing
3. Exponential moving average smoothing

References:
- Chen et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing" ICML 2018
- Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" CVPR 2018
"""

import torch
import torch.nn as nn


class DynamicLossBalancer(nn.Module):
    """
    Dynamically balances multiple loss terms based on their gradient magnitudes or loss ratios.

    This helps when different losses have vastly different scales (e.g., VGG19 ~0.1, Inception ~0.002).

    Args:
        loss_names (list): Names of losses to balance (e.g., ['vgg19', 'inception'])
        initial_weights (dict): Initial loss weights (e.g., {'vgg19': 1.0, 'inception': 1.0})
        balance_method (str): 'gradient' (balance by gradient magnitude) or
                              'loss_ratio' (balance by loss magnitude ratio)
        momentum (float): EMA momentum for smoothing (0.9 = slow adaptation, 0.1 = fast)
        update_freq (int): Update weights every N iterations
        target_ratio (dict, optional): Target ratios for each loss relative to first loss
                                       e.g., {'vgg19': 1.0, 'inception': 1.0} for equal contribution
    """

    def __init__(self,
                 loss_names,
                 initial_weights,
                 balance_method='loss_ratio',
                 momentum=0.9,
                 update_freq=10,
                 target_ratio=None):
        super(DynamicLossBalancer, self).__init__()

        self.loss_names = loss_names
        self.balance_method = balance_method
        self.momentum = momentum
        self.update_freq = update_freq
        self.iter_count = 0

        # Initialize weights as learnable parameters (but we'll update manually)
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(initial_weights[name]), requires_grad=False)
            for name in loss_names
        })

        # Target ratios (default: equal contribution)
        if target_ratio is None:
            self.target_ratio = {name: 1.0 for name in loss_names}
        else:
            self.target_ratio = target_ratio

        # EMA tracking for loss magnitudes
        self.loss_ema = {name: None for name in loss_names}

        # Gradient magnitude tracking (if using gradient balancing)
        self.grad_ema = {name: None for name in loss_names}

    def forward(self, losses_dict, model=None):
        """
        Compute weighted total loss and optionally update weights.

        Args:
            losses_dict (dict): Dictionary of loss tensors {'vgg19': tensor, 'inception': tensor}
            model (nn.Module, optional): Model for gradient computation (needed for gradient balancing)

        Returns:
            total_loss (tensor): Weighted sum of losses
            weighted_losses (dict): Individual weighted losses for logging
        """
        weighted_losses = {}
        total_loss = 0

        # Compute weighted losses
        for name in self.loss_names:
            if name in losses_dict:
                loss_val = losses_dict[name]
                weight = self.weights[name].item()
                weighted_loss = weight * loss_val
                weighted_losses[f'weighted_{name}'] = weighted_loss
                total_loss += weighted_loss

                # Update EMA of loss magnitude
                if self.loss_ema[name] is None:
                    self.loss_ema[name] = loss_val.item()
                else:
                    self.loss_ema[name] = (self.momentum * self.loss_ema[name] +
                                           (1 - self.momentum) * loss_val.item())

        # Update weights periodically
        self.iter_count += 1
        if self.iter_count % self.update_freq == 0:
            self._update_weights(losses_dict, model)

        return total_loss, weighted_losses

    @torch.no_grad()
    def _update_weights(self, losses_dict, model=None):
        """Update loss weights based on balancing method."""

        if self.balance_method == 'loss_ratio':
            # Balance based on loss magnitude ratios
            self._update_by_loss_ratio(losses_dict)
        elif self.balance_method == 'gradient' and model is not None:
            # Balance based on gradient magnitudes (more expensive)
            self._update_by_gradient(losses_dict, model)

    @torch.no_grad()
    def _update_by_loss_ratio(self, losses_dict):
        """
        Adjust weights so effective contributions match target ratios.

        Effective contribution = weight × loss_magnitude
        Goal: Make (w1 × L1) / (w2 × L2) = target_ratio
        """
        # Use EMA smoothed loss magnitudes for stability
        loss_mags = {name: self.loss_ema[name] for name in self.loss_names if name in losses_dict}

        if len(loss_mags) < 2:
            return  # Need at least 2 losses to balance

        # Use first loss as reference
        ref_name = self.loss_names[0]
        ref_mag = loss_mags[ref_name]
        ref_weight = self.weights[ref_name].item()
        ref_contribution = ref_weight * ref_mag

        # Adjust other losses to match target ratios
        for name in self.loss_names[1:]:
            if name in loss_mags:
                target_ratio = self.target_ratio[name]
                current_mag = loss_mags[name]

                # Target: (w_new × current_mag) = target_ratio × ref_contribution
                new_weight = (target_ratio * ref_contribution) / (current_mag + 1e-8)

                # Smooth weight update with momentum
                old_weight = self.weights[name].item()
                smoothed_weight = self.momentum * old_weight + (1 - self.momentum) * new_weight

                # Clamp to reasonable range [0.0001, 100]
                smoothed_weight = max(0.0001, min(100.0, smoothed_weight))

                self.weights[name].data = torch.tensor(smoothed_weight)

    @torch.no_grad()
    def _update_by_gradient(self, losses_dict, model):
        """
        Balance based on gradient magnitudes (GradNorm-style).

        This is more accurate but computationally expensive - requires computing
        gradients for each loss individually.
        """
        # Compute gradient norm for each loss
        grad_norms = {}

        for name in self.loss_names:
            if name in losses_dict:
                loss = losses_dict[name] * self.weights[name]

                # Compute gradients for this loss only
                model.zero_grad()
                loss.backward(retain_graph=True)

                # Compute gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norms[name] = (total_norm ** 0.5)

                # Update EMA
                if self.grad_ema[name] is None:
                    self.grad_ema[name] = grad_norms[name]
                else:
                    self.grad_ema[name] = (self.momentum * self.grad_ema[name] +
                                           (1 - self.momentum) * grad_norms[name])

        # Adjust weights to equalize gradient contributions
        if len(grad_norms) < 2:
            return

        ref_name = self.loss_names[0]
        ref_grad = self.grad_ema[ref_name]

        for name in self.loss_names[1:]:
            if name in grad_norms:
                target_ratio = self.target_ratio[name]
                current_grad = self.grad_ema[name]

                # Adjust weight to balance gradients
                grad_ratio = current_grad / (ref_grad + 1e-8)
                adjustment = target_ratio / (grad_ratio + 1e-8)

                old_weight = self.weights[name].item()
                new_weight = old_weight * adjustment
                smoothed_weight = self.momentum * old_weight + (1 - self.momentum) * new_weight

                # Clamp to reasonable range
                smoothed_weight = max(0.0001, min(100.0, smoothed_weight))

                self.weights[name].data = torch.tensor(smoothed_weight)

    def get_current_weights(self):
        """Return current weights as a dictionary."""
        return {name: self.weights[name].item() for name in self.loss_names}

    def get_effective_contributions(self):
        """Return effective gradient contributions (weight × loss_magnitude)."""
        return {name: self.weights[name].item() * (self.loss_ema[name] or 0)
                for name in self.loss_names}
