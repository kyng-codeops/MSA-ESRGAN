from .gradient_variance_loss import GradientVarianceLoss
from .relativistic_gan_loss import RelativisticAverageGANLoss
from .lpips_loss import LPIPSLoss
from basicsr.utils.registry import LOSS_REGISTRY

LOSS_REGISTRY.register(GradientVarianceLoss)
LOSS_REGISTRY.register(RelativisticAverageGANLoss)
LOSS_REGISTRY.register(LPIPSLoss)
