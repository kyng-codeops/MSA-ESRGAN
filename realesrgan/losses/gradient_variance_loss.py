import torch
import torch.nn as nn


class GradientVarianceLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, **kwargs):
        super(GradientVarianceLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight  # Store the loss weight

    def forward(self, pred, target):
        # Compute gradients (Sobel operator)
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=pred.dtype,
            device=pred.device
        ).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=pred.dtype,
            device=pred.device
        ).unsqueeze(0).unsqueeze(0)

        def gradient(img, kernel):
            # Apply to each channel
            grad = []
            for c in range(img.shape[1]):
                grad.append(torch.nn.functional.conv2d(img[:, c:c+1], kernel, padding=1))
            return torch.cat(grad, dim=1)

        pred_grad_x = gradient(pred, sobel_x)
        pred_grad_y = gradient(pred, sobel_y)
        target_grad_x = gradient(target, sobel_x)
        target_grad_y = gradient(target, sobel_y)

        # Compute variance of gradients
        pred_var = torch.var(pred_grad_x, dim=[2, 3]) + torch.var(pred_grad_y, dim=[2, 3])
        target_var = torch.var(target_grad_x, dim=[2, 3]) + torch.var(target_grad_y, dim=[2, 3])

        loss = torch.abs(pred_var - target_var)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        # Apply loss weight
        return loss * self.loss_weight
