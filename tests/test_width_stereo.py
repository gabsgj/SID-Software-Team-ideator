"""
Tests for Stereo Camera Width Estimation
=========================================
Validates depth-based and 3D-projection width measurement.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crack_detection.width_estimation.stereo import StereoWidthEstimator
from tests.fixtures.generate_test_data import (
    generate_synthetic_crack_image,
    generate_synthetic_depth_map,
)


@pytest.fixture
def estimator():
    """StereoWidthEstimator with a typical camera matrix."""
    cam = np.array([[800, 0, 320], [0, 800, 320], [0, 0, 1]], dtype=np.float64)
    return StereoWidthEstimator(camera_matrix=cam)


@pytest.fixture
def crack_and_depth():
    """5px crack + depth map at 2 m."""
    _, mask, meta = generate_synthetic_crack_image(width_px=5, length_px=200)
    depth = generate_synthetic_depth_map(base_depth=2000.0, noise_std=2.0)
    return mask, depth, meta


class TestStereoWidth:

    def test_stereo_width_with_known_depth(self, estimator, crack_and_depth):
        """Known depth + known pixel width → mm result must be non-None."""
        mask, depth, meta = crack_and_depth
        cam = np.array([[800, 0, 320], [0, 800, 320], [0, 0, 1]], dtype=np.float64)
        result = estimator.estimate_width(mask, depth, camera_matrix=cam)

        assert result is not None, "Stereo estimation returned None"
        assert result.width_mm is not None, "width_mm should be set with stereo"
        assert result.width_mm > 0, "width_mm should be positive"

        # Sanity check: at 2 m distance with fx=800, 5px ≈ 12.5 mm
        # width_mm = width_px * depth / fx ≈ 5 * 2000 / 800 = 12.5
        assert 2.0 < result.width_mm < 50.0, (
            f"Width {result.width_mm:.2f}mm seems out of range for 5px crack at 2m"
        )

    def test_stereo_3d_width(self, estimator, crack_and_depth):
        """3D projection method should also return valid width."""
        mask, depth, _ = crack_and_depth
        cam = np.array([[800, 0, 320], [0, 800, 320], [0, 0, 1]], dtype=np.float64)

        result = estimator.estimate_width_3d(mask, depth, camera_matrix=cam)

        assert result is not None
        if result.width_mm is not None:
            assert result.width_mm > 0

    def test_robust_depth_estimation(self, estimator):
        """Median filtering of noisy depth should be more stable."""
        depth = np.full((100, 100), 2000.0, dtype=np.float32)
        # Add outliers
        depth[50, 50] = 0.0  # dead pixel
        depth[51, 50] = 9999.0  # outlier

        robust_d = estimator._get_robust_depth(depth, (50, 50), window=5)
        assert abs(robust_d - 2000.0) < 50.0, (
            f"Robust depth {robust_d} should be close to 2000"
        )

    def test_missing_depth_handling(self, estimator):
        """NaN / zero depth values should be handled gracefully."""
        _, mask, _ = generate_synthetic_crack_image(width_px=5, length_px=100)

        # Depth map with many NaN values
        depth = np.full((640, 640), np.nan, dtype=np.float32)
        # Set some valid values in the crack region
        depth[mask > 0] = 2000.0

        cam = np.array([[800, 0, 320], [0, 800, 320], [0, 0, 1]], dtype=np.float64)

        # Should not crash
        result = estimator.estimate_width(mask, depth, camera_matrix=cam)
        assert result is not None

    def test_zero_depth_map(self, estimator):
        """Completely zero depth map should return gracefully."""
        _, mask, _ = generate_synthetic_crack_image(width_px=5, length_px=100)
        depth = np.zeros((640, 640), dtype=np.float32)

        cam = np.array([[800, 0, 320], [0, 800, 320], [0, 0, 1]], dtype=np.float64)
        result = estimator.estimate_width(mask, depth, camera_matrix=cam)
        assert result is not None


class TestStereo3DProjection:

    def test_pixel_to_3d(self, estimator):
        """Verify pixel-to-3D projection math."""
        cam = np.array([[800, 0, 320], [0, 800, 320], [0, 0, 1]], dtype=np.float64)

        # Center pixel at 2m depth
        pt = estimator._pixel_to_3d(320, 320, 2000.0, cam)
        assert pt is not None
        assert len(pt) == 3
        assert abs(pt[0]) < 1.0  # X should be near 0 (center pixel)
        assert abs(pt[1]) < 1.0  # Y should be near 0 (center pixel)
        assert abs(pt[2] - 2000.0) < 1.0  # Z = depth

    def test_off_center_projection(self, estimator):
        """Off-center pixel should have non-zero X, Y."""
        cam = np.array([[800, 0, 320], [0, 800, 320], [0, 0, 1]], dtype=np.float64)

        pt = estimator._pixel_to_3d(420, 220, 2000.0, cam)
        assert pt[0] > 0  # right of center
        assert pt[1] < 0  # above center
