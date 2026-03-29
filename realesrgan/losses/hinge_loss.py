"""
Hinge Loss for GANs with Relativistic GAN (RaGAN) framework support.

Hinge loss provides better training stability compared to standard GAN losses.
When combined with RaGAN, it uses relativistic averaging for both real and fake predictions.

References:
    - Geometric GAN (Lim & Ye, 2017)
    - Spectral Normalization GAN (Miyato et al., 2018)
    - The relativistic discriminator (Jolicoeur-Martineau, 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    """
    Hinge Loss for GAN training.

    For discriminator:
        L_D = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]

    For generator:
        L_G = -E[D(fake)]

    Args:
        relativistic (bool): Whether to use Relativistic Average Hinge Loss. Default: True
        loss_weight (float): Weight for the loss. Default: 1.0
        margin (float): Margin/target for hinge loss. Default: 1.0. Use 0.9 for label smoothing to prevent discriminator overconfidence.
    """

    def __init__(self, relativistic=True, loss_weight=1.0, margin=1.0):
        super(HingeLoss, self).__init__()
        self.relativistic = relativistic
        self.loss_weight = loss_weight
        self.margin = margin

    def forward(self, real_pred, fake_pred, is_disc):
        """
        Calculate hinge loss.

        Args:
            real_pred (Tensor): Discriminator predictions for real images
            fake_pred (Tensor): Discriminator predictions for fake images
            is_disc (bool): Whether this is discriminator training (True) or generator training (False)

        Returns:
            Tensor: Calculated loss value
        """
        if self.relativistic:
            # Relativistic Average Hinge Loss (RaHinge)
            if is_disc:
                # Discriminator loss: wants D(real) - avg(D(fake)) > margin and D(fake) - avg(D(real)) < -margin
                real_loss = F.relu(self.margin - (real_pred - fake_pred.mean()))
                fake_loss = F.relu(self.margin + (fake_pred - real_pred.mean()))
                loss = (real_loss.mean() + fake_loss.mean()) / 2
            else:
                # Generator loss: wants D(fake) - avg(D(real)) > margin and D(real) - avg(D(fake)) < -margin
                # This is the adversarial loss for the generator
                real_loss = F.relu(self.margin + (real_pred - fake_pred.mean()))
                fake_loss = F.relu(self.margin - (fake_pred - real_pred.mean()))
                loss = (real_loss.mean() + fake_loss.mean()) / 2
        else:
            # Standard Hinge Loss
            if is_disc:
                # Discriminator loss
                real_loss = F.relu(self.margin - real_pred).mean()
                fake_loss = F.relu(self.margin + fake_pred).mean()
                loss = real_loss + fake_loss
            else:
                # Generator loss: maximize D(fake) = minimize -D(fake)
                loss = -fake_pred.mean()

        return loss * self.loss_weight


class RelativisticHingeLoss(nn.Module):
    """
    Alias for HingeLoss with relativistic=True for backward compatibility.
    """

    def __init__(self, loss_weight=1.0):
        super(RelativisticHingeLoss, self).__init__()
        self.hinge_loss = HingeLoss(relativistic=True, loss_weight=loss_weight)

    def forward(self, real_pred, fake_pred, is_disc):
        return self.hinge_loss(real_pred, fake_pred, is_disc)
