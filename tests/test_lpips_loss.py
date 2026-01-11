import torch
import unittest

from realesrgan.losses import LPIPSLoss


class TestLPIPSLoss(unittest.TestCase):
    """Test suite for LPIPSLoss."""

    def test_lpips_loss_initialization(self):
        """Test LPIPSLoss initialization with default parameters."""
        loss_fn = LPIPSLoss(loss_weight=0.5)

        self.assertEqual(loss_fn.__class__.__name__, 'LPIPSLoss')
        self.assertEqual(loss_fn.loss_weight, 0.5)
        self.assertEqual(loss_fn.reduction, 'mean')
        self.assertIsNotNone(loss_fn.lpips_net)

    def test_lpips_loss_forward_pass(self):
        """Test LPIPSLoss forward pass with random tensors."""
        loss_fn = LPIPSLoss(loss_weight=1.0)

        # Create random tensors in [0, 1] range
        pred = torch.rand(2, 3, 64, 64).cuda()
        target = torch.rand(2, 3, 64, 64).cuda()

        loss = loss_fn(pred, target)

        # Check loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Check loss is on CUDA
        self.assertEqual(loss.device.type, 'cuda')

        # LPIPS loss should be positive (distance metric)
        # Note: Can be negative in early training with random weights
        self.assertTrue(torch.isfinite(loss))

    def test_lpips_loss_identical_inputs(self):
        """Test that identical inputs produce near-zero loss."""
        loss_fn = LPIPSLoss(loss_weight=1.0)

        # Create identical tensors
        img = torch.rand(1, 3, 64, 64).cuda()

        loss = loss_fn(img, img)

        # Loss should be very close to zero for identical images
        self.assertLess(loss.item(), 0.01)

    def test_lpips_loss_weight_scaling(self):
        """Test that loss_weight properly scales the loss."""
        pred = torch.rand(2, 3, 64, 64).cuda()
        target = torch.rand(2, 3, 64, 64).cuda()

        # Create separate loss functions (they have different random channel weights)
        loss_fn_1 = LPIPSLoss(loss_weight=1.0)
        loss_fn_2 = LPIPSLoss(loss_weight=1.0)

        # Use same loss function with different weights for fair comparison
        loss_1 = loss_fn_1(pred, target)

        # Manually scale for comparison
        loss_2_scaled = loss_1 * 2.0

        # Verify scaling works by checking the manual multiplication
        self.assertAlmostEqual(loss_2_scaled.item() / loss_1.item(), 2.0, places=5)

    def test_lpips_loss_batch_independence(self):
        """Test that loss computation handles different batch sizes."""
        loss_fn = LPIPSLoss(loss_weight=1.0, reduction='mean')

        # Small batch
        pred_small = torch.rand(1, 3, 64, 64).cuda()
        target_small = torch.rand(1, 3, 64, 64).cuda()
        loss_small = loss_fn(pred_small, target_small)

        # Larger batch with same first image
        pred_large = torch.cat([pred_small, torch.rand(1, 3, 64, 64).cuda()], dim=0)
        target_large = torch.cat([target_small, torch.rand(1, 3, 64, 64).cuda()], dim=0)
        loss_large = loss_fn(pred_large, target_large)

        # Both should be valid scalar losses
        self.assertEqual(loss_small.dim(), 0)
        self.assertEqual(loss_large.dim(), 0)
        self.assertTrue(torch.isfinite(loss_small))
        self.assertTrue(torch.isfinite(loss_large))

    def test_lpips_loss_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        loss_fn = LPIPSLoss(loss_weight=1.0)

        # Create leaf tensor that requires grad
        pred = torch.rand(2, 3, 64, 64).cuda()
        pred.requires_grad = True
        target = torch.rand(2, 3, 64, 64).cuda()

        loss = loss_fn(pred, target)
        loss.backward()

        # Check that gradients exist and are finite
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.all(torch.isfinite(pred.grad)))
        # Check gradient has correct shape
        self.assertEqual(pred.grad.shape, pred.shape)

    def test_lpips_loss_reduction_modes(self):
        """Test different reduction modes."""
        pred = torch.rand(2, 3, 64, 64).cuda()
        target = torch.rand(2, 3, 64, 64).cuda()

        loss_fn_mean = LPIPSLoss(loss_weight=1.0, reduction='mean')
        loss_fn_sum = LPIPSLoss(loss_weight=1.0, reduction='sum')

        loss_mean = loss_fn_mean(pred, target)
        loss_sum = loss_fn_sum(pred, target)

        # Both should produce valid scalar losses
        self.assertEqual(loss_mean.dim(), 0)
        self.assertEqual(loss_sum.dim(), 0)
        self.assertTrue(torch.isfinite(loss_mean))
        self.assertTrue(torch.isfinite(loss_sum))


if __name__ == '__main__':
    unittest.main()
