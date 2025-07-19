from .gradient_variance_loss import GradientVarianceLoss
from basicsr.utils.registry import LOSS_REGISTRY

LOSS_REGISTRY.register(GradientVarianceLoss)
