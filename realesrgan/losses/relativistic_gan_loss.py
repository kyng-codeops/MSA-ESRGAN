import torch
import torch.nn as nn

class RelativisticAverageGANLoss(nn.Module):
    def __init__(self, loss_type='bce', **kwargs):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'wgan':
            self.criterion = None  # WGAN does not use BCE or MSE
        else:
            raise NotImplementedError(f'Unknown loss_type: {loss_type}')

    def forward(self, real_pred, fake_pred, is_disc):
        if self.loss_type == 'wgan':
            # Relativistic WGAN loss
            if is_disc:
                real_loss = -(torch.mean(real_pred - fake_pred.mean()))
                fake_loss = torch.mean(fake_pred - real_pred.mean())
                return (real_loss + fake_loss) / 2
            else:
                real_loss = torch.mean(real_pred - fake_pred.mean())
                fake_loss = -(torch.mean(fake_pred - real_pred.mean()))
                return (real_loss + fake_loss) / 2
        else:
            if is_disc:
                real_loss = self.criterion(real_pred - fake_pred.mean(), torch.ones_like(real_pred))
                fake_loss = self.criterion(fake_pred - real_pred.mean(), torch.zeros_like(fake_pred))
                return (real_loss + fake_loss) / 2
            else:
                real_loss = self.criterion(real_pred - fake_pred.mean(), torch.zeros_like(real_pred))
                fake_loss = self.criterion(fake_pred - real_pred.mean(), torch.ones_like(fake_pred))
                return (real_loss + fake_loss) / 2