import torch
import unittest

from realesrgan.losses import HingeLoss, RelativisticHingeLoss


class TestHingeLoss(unittest.TestCase):
    """Test suite for HingeLoss and RelativisticHingeLoss."""

    def test_hinge_loss_initialization(self):
        """Test HingeLoss initialization with default parameters."""
        loss_fn = HingeLoss(relativistic=True, loss_weight=0.1)

        self.assertEqual(loss_fn.__class__.__name__, 'HingeLoss')
        self.assertEqual(loss_fn.loss_weight, 0.1)
        self.assertTrue(loss_fn.relativistic)

    def test_hinge_loss_non_relativistic_discriminator(self):
        """Test non-relativistic Hinge loss for discriminator."""
        loss_fn = HingeLoss(relativistic=False, loss_weight=1.0)

        # Create fake discriminator predictions
        real_pred = torch.tensor([1.5, 2.0, 1.0])  # D(real)
        fake_pred = torch.tensor([-0.5, -1.0, 0.5])  # D(fake)

        loss = loss_fn(real_pred, fake_pred, is_disc=True)

        # Check loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Loss should be positive (max(0, 1-real) + max(0, 1+fake))
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_hinge_loss_non_relativistic_generator(self):
        """Test non-relativistic Hinge loss for generator."""
        loss_fn = HingeLoss(relativistic=False, loss_weight=1.0)

        # Create fake discriminator predictions
        real_pred = torch.tensor([1.5, 2.0, 1.0])  # D(real) - not used for non-relativistic G
        fake_pred = torch.tensor([-0.5, -1.0, 0.5])  # D(fake)

        loss = loss_fn(real_pred, fake_pred, is_disc=False)

        # Check loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Generator loss should be -mean(D(fake))
        expected = -fake_pred.mean()
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_relativistic_hinge_loss_discriminator(self):
        """Test relativistic Hinge loss for discriminator."""
        loss_fn = HingeLoss(relativistic=True, loss_weight=1.0)

        # Create fake discriminator predictions with some margin violations
        # to ensure non-zero loss
        real_pred = torch.tensor([0.8, 0.5, 1.0])  # Some below margin
        fake_pred = torch.tensor([0.2, 0.5, -0.2])  # Some close to real

        loss = loss_fn(real_pred, fake_pred, is_disc=True)

        # Check loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Loss should be non-negative
        self.assertGreaterEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_relativistic_hinge_loss_generator(self):
        """Test relativistic Hinge loss for generator."""
        loss_fn = HingeLoss(relativistic=True, loss_weight=1.0)

        # Create fake discriminator predictions
        real_pred = torch.tensor([2.0, 1.5, 1.8])
        fake_pred = torch.tensor([0.5, -0.5, 0.2])

        loss = loss_fn(real_pred, fake_pred, is_disc=False)

        # Check loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Loss should be positive
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_relativistic_hinge_loss_alias(self):
        """Test that RelativisticHingeLoss is an alias for HingeLoss with relativistic=True."""
        loss_fn = RelativisticHingeLoss(loss_weight=0.5)

        self.assertEqual(loss_fn.__class__.__name__, 'RelativisticHingeLoss')
        self.assertEqual(loss_fn.hinge_loss.loss_weight, 0.5)
        self.assertTrue(loss_fn.hinge_loss.relativistic)

    def test_hinge_loss_weight_application(self):
        """Test that loss_weight is properly applied."""
        loss_fn_1 = HingeLoss(relativistic=True, loss_weight=1.0)
        loss_fn_2 = HingeLoss(relativistic=True, loss_weight=2.0)

        real_pred = torch.tensor([2.0, 1.5])
        fake_pred = torch.tensor([0.5, -0.5])

        loss_1 = loss_fn_1(real_pred, fake_pred, is_disc=True)
        loss_2 = loss_fn_2(real_pred, fake_pred, is_disc=True)

        # loss_2 should be exactly 2x loss_1
        self.assertAlmostEqual(loss_2.item(), 2.0 * loss_1.item(), places=5)

    def test_hinge_loss_cuda(self):
        """Test HingeLoss works on CUDA if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        loss_fn = HingeLoss(relativistic=True, loss_weight=1.0)

        real_pred = torch.rand(4).cuda()
        fake_pred = torch.rand(4).cuda()

        loss = loss_fn(real_pred, fake_pred, is_disc=True)

        # Check loss is on CUDA
        self.assertEqual(loss.device.type, 'cuda')
        self.assertTrue(torch.isfinite(loss))

    def test_hinge_loss_batch_consistency(self):
        """Test that Hinge loss handles different batch sizes correctly."""
        loss_fn = HingeLoss(relativistic=True, loss_weight=1.0)

        for batch_size in [2, 4, 8, 16]:
            real_pred = torch.randn(batch_size)
            fake_pred = torch.randn(batch_size)

            loss = loss_fn(real_pred, fake_pred, is_disc=True)

            # Loss should be a scalar
            self.assertEqual(loss.dim(), 0)
            self.assertTrue(torch.isfinite(loss))

    def test_hinge_loss_gradient_flow(self):
        """Test that gradients flow properly through Hinge loss."""
        loss_fn = HingeLoss(relativistic=True, loss_weight=1.0)

        real_pred = torch.randn(4, requires_grad=True)
        fake_pred = torch.randn(4, requires_grad=True)

        loss = loss_fn(real_pred, fake_pred, is_disc=True)
        loss.backward()

        # Check gradients exist and are finite
        self.assertIsNotNone(real_pred.grad)
        self.assertIsNotNone(fake_pred.grad)
        self.assertTrue(torch.all(torch.isfinite(real_pred.grad)))
        self.assertTrue(torch.all(torch.isfinite(fake_pred.grad)))


if __name__ == '__main__':
    unittest.main()
