"""
Pytest configuration and shared fixtures for SID crack detection tests.
"""

import sys
import os
import pytest
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.generate_test_data import (
    generate_synthetic_crack_image,
    generate_crack_with_known_width,
    generate_branching_crack,
    generate_curved_crack,
    generate_synthetic_depth_map,
    generate_test_ifc_model,
    generate_calibration_image,
)


@pytest.fixture
def sample_crack_image():
    """A single synthetic crack image (width=5px, 45°)."""
    image, mask, meta = generate_synthetic_crack_image(
        width_px=5, length_px=200, orientation_deg=45
    )
    return image, mask, meta


@pytest.fixture
def sample_mask():
    """Binary mask of a 5px-wide crack."""
    _, mask, _ = generate_synthetic_crack_image(width_px=5, length_px=200)
    return mask


@pytest.fixture
def sample_depth_map():
    """Synthetic depth map at 2 m distance."""
    return generate_synthetic_depth_map(
        image_size=(640, 640), base_depth=2000.0, noise_std=5.0
    )


@pytest.fixture
def sample_ifc_path(tmp_path):
    """Path to a generated minimal test IFC file."""
    ifc_path = str(tmp_path / "test_bridge.ifc")
    generate_test_ifc_model(ifc_path)
    return ifc_path


@pytest.fixture
def known_width_cracks():
    """Set of crack images with known widths for benchmarking."""
    return generate_crack_with_known_width(widths=[2, 5, 10, 15, 20])


@pytest.fixture
def branching_crack():
    """Y-shaped branching crack."""
    return generate_branching_crack(main_width=5, branch_width=3)


@pytest.fixture
def curved_crack():
    """Arc-shaped curved crack."""
    return generate_curved_crack(width_px=5, curvature_radius=300)


@pytest.fixture
def scale_info():
    """A sample ScaleInfo for GSD-based conversion."""
    from crack_detection.width_estimation.common import ScaleInfo, MeasurementMethod
    return ScaleInfo(
        gsd_mm_per_px=0.5,
        method=MeasurementMethod.MONOCULAR_GSD,
        distance_mm=2000.0,
        focal_length_px=800.0,
    )


@pytest.fixture
def camera_matrix():
    """Typical camera intrinsic matrix."""
    fx, fy = 800.0, 800.0
    cx, cy = 320.0, 320.0
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)
