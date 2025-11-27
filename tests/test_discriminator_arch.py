import torch
import unittest

from realesrgan.archs.msa_discriminator_arch import UNetDiscriminator


class TestUNetDiscriminator(unittest.TestCase):

    def test_unetdiscriminator_relativistic(self):
        """Test arch: UNetDiscriminator (relativistic)."""

        # model init and forward (cpu)
        net = UNetDiscriminator(num_in_ch=3, num_feat=4, skip_connection=True)
        net.eval()  # Set to eval mode to avoid batch norm issues
        img = torch.rand((2, 3, 32, 32), dtype=torch.float32)  # Use batch size of 2
        output = net(img)

        # Check output shape: (batch_size, 1, height, width) for UNet discriminator
        self.assertEqual(output.shape[0], 2)  # batch size
        self.assertEqual(output.shape[1], 1)  # single channel output
        self.assertGreater(output.shape[2], 0)  # height > 0
        self.assertGreater(output.shape[3], 0)  # width > 0

        # model init and forward (gpu)
        if torch.cuda.is_available():
            net.cuda()
            output = net(img.cuda())
            self.assertEqual(output.shape[0], 2)  # batch size
            self.assertEqual(output.shape[1], 1)  # single channel output

    def test_unetdiscriminator_batch_sizes(self):
        """Test UNetDiscriminator with different batch sizes."""
        net = UNetDiscriminator(num_in_ch=3, num_feat=4, skip_connection=True)
        net.eval()  # Set to eval mode

        for batch_size in [2, 4, 8]:  # Start with batch_size >= 2
            img = torch.rand((batch_size, 3, 32, 32), dtype=torch.float32)
            output = net(img)

            # Output batch size should match input batch size
            self.assertEqual(output.shape[0], batch_size)
            # Output should have 1 channel
            self.assertEqual(output.shape[1], 1)

    def test_unetdiscriminator_gradient_flow(self):
        """Test that gradients flow properly through the discriminator."""
        net = UNetDiscriminator(num_in_ch=3, num_feat=4, skip_connection=True)
        net.train()  # Set to train mode for gradient computation
        img = torch.rand((2, 3, 32, 32), dtype=torch.float32, requires_grad=True)  # Batch size 2

        output = net(img)
        loss = output.mean()
        loss.backward()

        # Check that input gradients are computed
        self.assertIsNotNone(img.grad)
        self.assertFalse(torch.all(img.grad == 0))

    def test_unetdiscriminator_output_shape(self):
        """Test that output shape is as expected: (batch, 1, H, W)."""
        net = UNetDiscriminator(num_in_ch=3, num_feat=4, skip_connection=True)
        net.eval()

        img = torch.rand((2, 3, 32, 32), dtype=torch.float32)
        output = net(img)

        # Verify 4D output tensor
        self.assertEqual(output.dim(), 4)
        self.assertEqual(output.shape, (2, 1, 32, 32))


if __name__ == '__main__':
    unittest.main()
