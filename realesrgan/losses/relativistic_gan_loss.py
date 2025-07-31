import torch
import torch.nn as nn

class RelativisticAverageGANLoss(nn.Module):
    def __init__(self, loss_type='bce', **kwargs):
        super().__init__()
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f'Unknown loss_type: {loss_type}')

    def forward(self, real_pred, fake_pred, is_disc):
        # real_pred and fake_pred are logits from the discriminator
        if is_disc:
            # Discriminator loss
            real_loss = self.criterion(real_pred - fake_pred.mean(), torch.ones_like(real_pred))
            fake_loss = self.criterion(fake_pred - real_pred.mean(), torch.zeros_like(fake_pred))
            return (real_loss + fake_loss) / 2
        else:
            # Generator loss
            real_loss = self.criterion(real_pred - fake_pred.mean(), torch.zeros_like(real_pred))
            fake_loss = self.criterion(fake_pred - real_pred.mean(), torch.ones_like(fake_pred))
            return (real_loss + fake_loss) / 2