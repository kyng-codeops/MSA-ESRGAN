import torch
import yaml
import unittest
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.losses import GANLoss, L1Loss, PerceptualLoss

from realesrgan.archs.discriminator_arch import UNetDiscriminatorSN
from realesrgan.models.realesrgan_model import RealESRGANModel
from realesrgan.models.realesrnet_model import RealESRNetModel


class TestRealESRNetModel(unittest.TestCase):
    """Test suite for RealESRNetModel."""

    def test_realesrnet_model_initialization(self):
        """Test RealESRNetModel initialization and attributes."""
        with open('tests/data/test_realesrnet_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRNetModel(opt)

        self.assertEqual(model.__class__.__name__, 'RealESRNetModel')
        self.assertIsInstance(model.net_g, RRDBNet)
        self.assertIsInstance(model.cri_pix, L1Loss)
        self.assertIsInstance(model.optimizers[0], torch.optim.Adam)

    def test_realesrnet_model_feed_data(self):
        """Test RealESRNetModel feed_data with various configurations."""
        with open('tests/data/test_realesrnet_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRNetModel(opt)

        # prepare data
        gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        kernel1 = torch.rand((1, 5, 5), dtype=torch.float32)
        kernel2 = torch.rand((1, 5, 5), dtype=torch.float32)
        sinc_kernel = torch.rand((1, 5, 5), dtype=torch.float32)
        data = dict(gt=gt, kernel1=kernel1, kernel2=kernel2, sinc_kernel=sinc_kernel)

        model.feed_data(data)
        model.feed_data(data)  # check dequeue

        self.assertEqual(model.lq.shape, (1, 3, 8, 8))
        self.assertEqual(model.gt.shape, (1, 3, 32, 32))

    def test_realesrnet_model_degradation_options(self):
        """Test RealESRNetModel with disabled degradation options."""
        with open('tests/data/test_realesrnet_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRNetModel(opt)

        gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        kernel1 = torch.rand((1, 5, 5), dtype=torch.float32)
        kernel2 = torch.rand((1, 5, 5), dtype=torch.float32)
        sinc_kernel = torch.rand((1, 5, 5), dtype=torch.float32)
        data = dict(gt=gt, kernel1=kernel1, kernel2=kernel2, sinc_kernel=sinc_kernel)

        # disable degradation
        model.opt['gaussian_noise_prob'] = 0
        model.opt['gray_noise_prob'] = 0
        model.opt['second_blur_prob'] = 0
        model.opt['gaussian_noise_prob2'] = 0
        model.opt['gray_noise_prob2'] = 0

        model.feed_data(data)

        self.assertEqual(model.lq.shape, (1, 3, 8, 8))
        self.assertEqual(model.gt.shape, (1, 3, 32, 32))

    def test_realesrnet_model_validation(self):
        """Test RealESRNetModel nondist_validation."""
        with open('tests/data/test_realesrnet_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRNetModel(opt)

        dataset_opt = dict(
            name='Demo',
            dataroot_gt='tests/data/gt',
            dataroot_lq='tests/data/lq',
            io_backend=dict(type='disk'),
            scale=4,
            phase='val'
        )
        dataset = PairedImageDataset(dataset_opt)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=0
        )

        self.assertTrue(model.is_train)
        model.nondist_validation(dataloader, 1, None, False)
        self.assertTrue(model.is_train)


class TestRealESRGANModel(unittest.TestCase):
    """Test suite for RealESRGANModel."""

    def test_realesrgan_model_initialization(self):
        """Test RealESRGANModel initialization and attributes."""
        with open('tests/data/test_realesrgan_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRGANModel(opt)

        self.assertEqual(model.__class__.__name__, 'RealESRGANModel')
        self.assertIsInstance(model.net_g, RRDBNet)
        self.assertIsInstance(model.net_d, UNetDiscriminatorSN)
        self.assertIsInstance(model.cri_pix, L1Loss)
        self.assertIsInstance(model.cri_perceptual, PerceptualLoss)
        self.assertIsInstance(model.cri_gan, GANLoss)
        self.assertEqual(len(model.optimizers), 2)
        self.assertIsInstance(model.optimizers[0], torch.optim.Adam)
        self.assertIsInstance(model.optimizers[1], torch.optim.Adam)

    def test_realesrgan_model_feed_data(self):
        """Test RealESRGANModel feed_data with various configurations."""
        with open('tests/data/test_realesrgan_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRGANModel(opt)

        gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        kernel1 = torch.rand((1, 5, 5), dtype=torch.float32)
        kernel2 = torch.rand((1, 5, 5), dtype=torch.float32)
        sinc_kernel = torch.rand((1, 5, 5), dtype=torch.float32)
        data = dict(gt=gt, kernel1=kernel1, kernel2=kernel2, sinc_kernel=sinc_kernel)

        model.feed_data(data)
        model.feed_data(data)  # check dequeue

        self.assertEqual(model.lq.shape, (1, 3, 8, 8))
        self.assertEqual(model.gt.shape, (1, 3, 32, 32))

    def test_realesrgan_model_degradation_options(self):
        """Test RealESRGANModel with disabled degradation options."""
        with open('tests/data/test_realesrgan_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRGANModel(opt)

        gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        kernel1 = torch.rand((1, 5, 5), dtype=torch.float32)
        kernel2 = torch.rand((1, 5, 5), dtype=torch.float32)
        sinc_kernel = torch.rand((1, 5, 5), dtype=torch.float32)
        data = dict(gt=gt, kernel1=kernel1, kernel2=kernel2, sinc_kernel=sinc_kernel)

        # disable degradation
        model.opt['gaussian_noise_prob'] = 0
        model.opt['gray_noise_prob'] = 0
        model.opt['second_blur_prob'] = 0
        model.opt['gaussian_noise_prob2'] = 0
        model.opt['gray_noise_prob2'] = 0

        model.feed_data(data)

        self.assertEqual(model.lq.shape, (1, 3, 8, 8))
        self.assertEqual(model.gt.shape, (1, 3, 32, 32))

    def test_realesrgan_model_validation(self):
        """Test RealESRGANModel nondist_validation."""
        with open('tests/data/test_realesrgan_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRGANModel(opt)

        dataset_opt = dict(
            name='Demo',
            dataroot_gt='tests/data/gt',
            dataroot_lq='tests/data/lq',
            io_backend=dict(type='disk'),
            scale=4,
            phase='val'
        )
        dataset = PairedImageDataset(dataset_opt)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=0
        )

        self.assertTrue(model.is_train)
        model.nondist_validation(dataloader, 1, None, False)
        self.assertTrue(model.is_train)

    def test_realesrgan_model_optimize_parameters(self):
        """Test RealESRGANModel optimize_parameters and loss tracking."""
        with open('tests/data/test_realesrgan_model.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        model = RealESRGANModel(opt)

        gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        kernel1 = torch.rand((1, 5, 5), dtype=torch.float32)
        kernel2 = torch.rand((1, 5, 5), dtype=torch.float32)
        sinc_kernel = torch.rand((1, 5, 5), dtype=torch.float32)
        data = dict(gt=gt, kernel1=kernel1, kernel2=kernel2, sinc_kernel=sinc_kernel)

        model.feed_data(data)
        model.optimize_parameters(1)

        self.assertEqual(model.output.shape, (1, 3, 32, 32))
        self.assertIsInstance(model.log_dict, dict)

        # check returned keys
        expected_keys = ['l_g_pix', 'l_g_percep', 'l_g_gan', 'l_d_real', 'out_d_real', 'l_d_fake', 'out_d_fake']
        for key in expected_keys:
            self.assertIn(key, model.log_dict.keys())


if __name__ == '__main__':
    unittest.main()
