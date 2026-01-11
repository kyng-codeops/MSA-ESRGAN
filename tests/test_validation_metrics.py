"""Test custom validation metrics"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realesrgan.metrics import calculate_lpips


class TestValidationMetrics(unittest.TestCase):

    def test_lpips_identical_images(self):
        """Test LPIPS with identical images (should be ~0)"""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        lpips_val = calculate_lpips(img, img)

        # Identical images should have very low LPIPS
        self.assertLess(lpips_val, 0.01)
        self.assertGreaterEqual(lpips_val, 0.0)

    def test_lpips_different_images(self):
        """Test LPIPS with different images"""
        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        lpips_val = calculate_lpips(img1, img2)

        # Different random images should have some LPIPS distance
        self.assertGreaterEqual(lpips_val, 0.0)
        self.assertLess(lpips_val, 1.0)

    def test_lpips_crop_border(self):
        """Test LPIPS with crop_border"""
        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        lpips_val = calculate_lpips(img1, img2, crop_border=4)

        # Should compute without error
        self.assertIsInstance(lpips_val, float)
        self.assertTrue(np.isfinite(lpips_val))


if __name__ == '__main__':
    unittest.main()


class TestValidationMetrics(unittest.TestCase):

    def test_lpips_identical_images(self):
        """Test LPIPS with identical images (should be ~0)"""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        lpips_val = calculate_lpips(img, img)

        # Identical images should have very low LPIPS
        self.assertLess(lpips_val, 0.01)
        self.assertGreaterEqual(lpips_val, 0.0)

    def test_lpips_different_images(self):
        """Test LPIPS with different images"""
        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        lpips_val = calculate_lpips(img1, img2)

        # Different random images should have some LPIPS distance
        self.assertGreaterEqual(lpips_val, 0.0)
        self.assertLess(lpips_val, 1.0)

    def test_lpips_crop_border(self):
        """Test LPIPS with crop_border"""
        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        lpips_val = calculate_lpips(img1, img2, crop_border=4)

        # Should compute without error
        self.assertIsInstance(lpips_val, float)
        self.assertTrue(np.isfinite(lpips_val))

    def test_niqe_random_image(self):
        """Test NIQE with random image"""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        niqe_val = calculate_niqe(img)

        # NIQE should be in reasonable range
        self.assertGreater(niqe_val, 0.0)
        self.assertLess(niqe_val, 15.0)

    def test_niqe_natural_pattern(self):
        """Test NIQE with smooth gradient (more natural)"""
        # Create smooth gradient (more natural than random noise)
        x = np.linspace(0, 255, 256)
        y = np.linspace(0, 255, 256)
        xx, yy = np.meshgrid(x, y)
        img = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.uint8)

        niqe_val = calculate_niqe(img)

        # Should produce a valid score
        self.assertIsInstance(niqe_val, (float, np.floating))
        self.assertTrue(np.isfinite(niqe_val))

    def test_niqe_crop_border(self):
        """Test NIQE with crop_border"""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        niqe_val = calculate_niqe(img, crop_border=4)

        # Should compute without error
        self.assertIsInstance(niqe_val, (float, np.floating))
        self.assertTrue(np.isfinite(niqe_val))


if __name__ == '__main__':
    unittest.main()
