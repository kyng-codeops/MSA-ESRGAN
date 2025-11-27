import unittest
import yaml

from realesrgan.data.realesrgan_dataset import RealESRGANDataset
from realesrgan.data.realesrgan_paired_dataset import RealESRGANPairedDataset


class TestRealESRGANDataset(unittest.TestCase):
    """Test suite for RealESRGANDataset."""

    def test_realesrgan_dataset_disk_backend(self):
        """Test RealESRGANDataset with disk backend."""
        with open('tests/data/test_realesrgan_dataset.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        dataset = RealESRGANDataset(opt)

        # Check io backend
        self.assertEqual(dataset.io_backend_opt['type'], 'disk')
        # Check correct meta info reading
        self.assertEqual(len(dataset), 2)
        # Check degradation configurations
        self.assertEqual(dataset.kernel_list, [
            'iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'
        ])
        self.assertEqual(dataset.betag_range2, [0.5, 4])

        # Test __getitem__
        result = dataset.__getitem__(0)
        # Check returned keys
        expected_keys = ['gt', 'kernel1', 'kernel2', 'sinc_kernel', 'gt_path']
        self.assertTrue(set(expected_keys).issubset(set(result.keys())))
        # Check shape and contents
        self.assertEqual(result['gt'].shape, (3, 400, 400))
        self.assertEqual(result['kernel1'].shape, (21, 21))
        self.assertEqual(result['kernel2'].shape, (21, 21))
        self.assertEqual(result['sinc_kernel'].shape, (21, 21))
        self.assertEqual(result['gt_path'], 'tests/data/gt/baboon.png')

    def test_realesrgan_dataset_lmdb_backend(self):
        """Test RealESRGANDataset with lmdb backend."""
        with open('tests/data/test_realesrgan_dataset.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        opt['dataroot_gt'] = 'tests/data/gt.lmdb'
        opt['io_backend']['type'] = 'lmdb'

        dataset = RealESRGANDataset(opt)

        # Check io backend
        self.assertEqual(dataset.io_backend_opt['type'], 'lmdb')
        # Check correct meta info reading
        self.assertEqual(len(dataset.paths), 2)
        # Check degradation configurations
        self.assertEqual(dataset.kernel_list, [
            'iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'
        ])
        self.assertEqual(dataset.betag_range2, [0.5, 4])

        # Test __getitem__
        result = dataset.__getitem__(1)
        # Check returned keys
        expected_keys = ['gt', 'kernel1', 'kernel2', 'sinc_kernel', 'gt_path']
        self.assertTrue(set(expected_keys).issubset(set(result.keys())))
        # Check shape and contents
        self.assertEqual(result['gt'].shape, (3, 400, 400))
        self.assertEqual(result['kernel1'].shape, (21, 21))
        self.assertEqual(result['kernel2'].shape, (21, 21))
        self.assertEqual(result['sinc_kernel'].shape, (21, 21))
        self.assertEqual(result['gt_path'], 'comic')

    def test_realesrgan_dataset_sinc_prob_zero(self):
        """Test RealESRGANDataset with sinc_prob set to 0."""
        with open('tests/data/test_realesrgan_dataset.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        opt['dataroot_gt'] = 'tests/data/gt.lmdb'
        opt['io_backend']['type'] = 'lmdb'
        opt['sinc_prob'] = 0
        opt['sinc_prob2'] = 0
        opt['final_sinc_prob'] = 0

        dataset = RealESRGANDataset(opt)
        result = dataset.__getitem__(0)

        # Check returned keys
        expected_keys = ['gt', 'kernel1', 'kernel2', 'sinc_kernel', 'gt_path']
        self.assertTrue(set(expected_keys).issubset(set(result.keys())))
        # Check shape and contents
        self.assertEqual(result['gt'].shape, (3, 400, 400))
        self.assertEqual(result['kernel1'].shape, (21, 21))
        self.assertEqual(result['kernel2'].shape, (21, 21))
        self.assertEqual(result['sinc_kernel'].shape, (21, 21))
        self.assertEqual(result['gt_path'], 'baboon')

    def test_realesrgan_dataset_lmdb_invalid_path(self):
        """Test RealESRGANDataset raises ValueError for invalid lmdb path."""
        with open('tests/data/test_realesrgan_dataset.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        opt['dataroot_gt'] = 'tests/data/gt'
        opt['io_backend']['type'] = 'lmdb'

        # lmdb backend should have paths ending with .lmdb
        with self.assertRaises(ValueError):
            dataset = RealESRGANDataset(opt)


class TestRealESRGANPairedDataset(unittest.TestCase):
    """Test suite for RealESRGANPairedDataset."""

    def test_realesrgan_paired_dataset_disk_backend(self):
        """Test RealESRGANPairedDataset with disk backend."""
        with open('tests/data/test_realesrgan_paired_dataset.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        dataset = RealESRGANPairedDataset(opt)

        # Check io backend
        self.assertEqual(dataset.io_backend_opt['type'], 'disk')
        # Check correct meta info reading
        self.assertEqual(len(dataset), 2)

        # Test __getitem__
        result = dataset.__getitem__(0)
        # Check returned keys
        expected_keys = ['gt', 'lq', 'gt_path', 'lq_path']
        self.assertTrue(set(expected_keys).issubset(set(result.keys())))
        # Check shape and contents
        self.assertEqual(result['gt'].shape, (3, 128, 128))
        self.assertEqual(result['lq'].shape, (3, 32, 32))
        self.assertEqual(result['gt_path'], 'tests/data/gt/baboon.png')
        self.assertEqual(result['lq_path'], 'tests/data/lq/baboon.png')

    def test_realesrgan_paired_dataset_lmdb_backend(self):
        """Test RealESRGANPairedDataset with lmdb backend."""
        with open('tests/data/test_realesrgan_paired_dataset.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        opt['dataroot_gt'] = 'tests/data/gt.lmdb'
        opt['dataroot_lq'] = 'tests/data/lq.lmdb'
        opt['io_backend']['type'] = 'lmdb'

        dataset = RealESRGANPairedDataset(opt)

        # Check io backend
        self.assertEqual(dataset.io_backend_opt['type'], 'lmdb')
        # Check correct meta info reading
        self.assertEqual(len(dataset), 2)

        # Test __getitem__
        result = dataset.__getitem__(1)
        # Check returned keys
        expected_keys = ['gt', 'lq', 'gt_path', 'lq_path']
        self.assertTrue(set(expected_keys).issubset(set(result.keys())))
        # Check shape and contents
        self.assertEqual(result['gt'].shape, (3, 128, 128))
        self.assertEqual(result['lq'].shape, (3, 32, 32))
        self.assertEqual(result['gt_path'], 'comic')
        self.assertEqual(result['lq_path'], 'comic')

    def test_realesrgan_paired_dataset_from_folder(self):
        """Test RealESRGANPairedDataset with paired_paths_from_folder."""
        with open('tests/data/test_realesrgan_paired_dataset.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        opt['dataroot_gt'] = 'tests/data/gt'
        opt['dataroot_lq'] = 'tests/data/lq'
        opt['io_backend'] = dict(type='disk')
        opt['meta_info'] = None

        dataset = RealESRGANPairedDataset(opt)

        # Check io backend
        self.assertEqual(dataset.io_backend_opt['type'], 'disk')
        # Check correct meta info reading
        self.assertEqual(len(dataset), 2)

        # Test __getitem__
        result = dataset.__getitem__(0)
        # Check returned keys
        expected_keys = ['gt', 'lq', 'gt_path', 'lq_path']
        self.assertTrue(set(expected_keys).issubset(set(result.keys())))
        # Check shape and contents
        self.assertEqual(result['gt'].shape, (3, 128, 128))
        self.assertEqual(result['lq'].shape, (3, 32, 32))

    def test_realesrgan_paired_dataset_normalization(self):
        """Test RealESRGANPairedDataset with normalization."""
        with open('tests/data/test_realesrgan_paired_dataset.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        opt['dataroot_gt'] = 'tests/data/gt'
        opt['dataroot_lq'] = 'tests/data/lq'
        opt['io_backend'] = dict(type='disk')
        opt['meta_info'] = None

        dataset = RealESRGANPairedDataset(opt)

        # Set normalization parameters
        dataset.mean = [0.5, 0.5, 0.5]
        dataset.std = [0.5, 0.5, 0.5]

        # Test __getitem__
        result = dataset.__getitem__(0)
        # Check returned keys
        expected_keys = ['gt', 'lq', 'gt_path', 'lq_path']
        self.assertTrue(set(expected_keys).issubset(set(result.keys())))
        # Check shape and contents
        self.assertEqual(result['gt'].shape, (3, 128, 128))
        self.assertEqual(result['lq'].shape, (3, 32, 32))


if __name__ == '__main__':
    unittest.main()
