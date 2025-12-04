import torch
import torch.nn as nn


class RelativisticAverageGANLoss(nn.Module):
    def __init__(self, loss_type='bce', gp_lambda=10, loss_weight=1.0, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.gp_lambda = gp_lambda
        self.loss_weight = loss_weight  # Store the weight
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type in ['wgan', 'wgan-gp']:
            self.criterion = None
        else:
            raise NotImplementedError(f'Unknown loss_type: {loss_type}')

    def forward(self, real_pred, fake_pred, is_disc, real_img=None, fake_img=None, net_d=None):
        if self.loss_type == 'wgan':
            if is_disc:
                real_loss = -(torch.mean(real_pred - fake_pred.mean()))
                fake_loss = torch.mean(fake_pred - real_pred.mean())
                loss = (real_loss + fake_loss) / 2
            else:
                real_loss = torch.mean(real_pred - fake_pred.mean())
                fake_loss = -(torch.mean(fake_pred - real_pred.mean()))
                loss = (real_loss + fake_loss) / 2
        elif self.loss_type == 'wgan-gp':
            # WGAN-GP: add gradient penalty for discriminator
            if is_disc:
                real_loss = -(torch.mean(real_pred - fake_pred.mean()))
                fake_loss = torch.mean(fake_pred - real_pred.mean())
                gp = self.gradient_penalty(real_img, fake_img, net_d)
                loss = ((real_loss + fake_loss) / 2) + self.gp_lambda * gp
            else:
                real_loss = torch.mean(real_pred - fake_pred.mean())
                fake_loss = -(torch.mean(fake_pred - real_pred.mean()))
                loss = (real_loss + fake_loss) / 2
        else:
            if is_disc:
                real_loss = self.criterion(real_pred - fake_pred.mean(), torch.ones_like(real_pred))
                fake_loss = self.criterion(fake_pred - real_pred.mean(), torch.zeros_like(fake_pred))
                loss = (real_loss + fake_loss) / 2
            else:
                real_loss = self.criterion(real_pred - fake_pred.mean(), torch.zeros_like(real_pred))
                fake_loss = self.criterion(fake_pred - real_pred.mean(), torch.ones_like(fake_pred))
                loss = (real_loss + fake_loss) / 2

        # Apply loss weight
        return loss * self.loss_weight

    def gradient_penalty(self, real_img, fake_img, net_d):
        batch_size = real_img.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_img.device)
        interpolates = alpha * real_img + (1 - alpha) * fake_img
        interpolates.requires_grad_(True)
        disc_interpolates = net_d(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp
