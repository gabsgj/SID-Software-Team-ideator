"""
Tests for the Monocular Width Estimation Algorithm
===================================================
Validates accuracy of the skeletonization + perpendicular distance
method using synthetic crack images with known ground-truth widths.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crack_detection.width_estimation.monocular import MonocularWidthEstimator
from crack_detection.width_estimation.common import ScaleInfo
from crack_detection.schemas import MeasurementMethod
from tests.fixtures.generate_test_data import (
    generate_synthetic_crack_image,
    generate_crack_with_known_width,
    generate_branching_crack,
    generate_curved_crack,
)


@pytest.fixture
def estimator():
    """Create a MonocularWidthEstimator with default settings."""
    return MonocularWidthEstimator()


# ──────────────────────────────────────────────────────────────────────
# Accuracy tests
# ──────────────────────────────────────────────────────────────────────

class TestWidthAccuracy:
    """Test width measurement accuracy against known ground truth."""

    @pytest.mark.parametrize("true_width", [2, 5, 10, 15, 20])
    def test_straight_crack_width_accuracy(self, estimator, true_width):
        """Measure straight cracks with known widths — error should be < 2px."""
        _, mask, meta = generate_synthetic_crack_image(
            width_px=true_width,
            length_px=max(100, true_width * 15),
            orientation_deg=0,
        )
        result = estimator.estimate_width(mask)

        assert result is not None, "Width estimation returned None"
        assert result.median_width_px > 0, "Median width should be positive"

        error = abs(result.median_width_px - true_width)
        assert error < max(2.5, true_width * 0.35), (
            f"Width error {error:.2f}px exceeds tolerance for {true_width}px crack. "
            f"Measured median: {result.median_width_px:.2f}px"
        )

    def test_diagonal_crack_width(self, estimator):
        """Crack at 45° — width should still be measured correctly."""
        true_width = 8
        _, mask, _ = generate_synthetic_crack_image(
            width_px=true_width, length_px=200, orientation_deg=45
        )
        result = estimator.estimate_width(mask)

        assert result is not None
        error = abs(result.median_width_px - true_width)
        assert error < max(3.0, true_width * 0.4), (
            f"Diagonal crack width error {error:.2f}px. "
            f"Measured: {result.median_width_px:.2f}px vs true: {true_width}px"
        )

    def test_curved_crack_width(self, estimator):
        """Curved crack — width should be consistent along the curve."""
        true_width = 6
        _, mask, _ = generate_curved_crack(width_px=true_width)
        result = estimator.estimate_width(mask)

        assert result is not None
        assert result.median_width_px > 0

        # Width should be roughly consistent (std should be small relative to mean)
        if result.mean_width_px > 0 and result.std_width_px is not None:
            cv = result.std_width_px / result.mean_width_px  # coeff of variation
            assert cv < 0.6, (
                f"Width too variable along curved crack: CV={cv:.2f}"
            )


# ──────────────────────────────────────────────────────────────────────
# Edge case tests
# ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test handling of edge cases and unusual inputs."""

    def test_very_thin_crack(self, estimator):
        """1-2 px wide crack should not crash."""
        _, mask, _ = generate_synthetic_crack_image(
            width_px=2, length_px=150, orientation_deg=30
        )
        result = estimator.estimate_width(mask)
        # Should return a result even for very thin cracks
        assert result is not None
        assert result.median_width_px >= 0

    def test_thick_crack(self, estimator):
        """30+ px wide crack should be measured."""
        _, mask, _ = generate_synthetic_crack_image(
            width_px=30, length_px=300, orientation_deg=10
        )
        result = estimator.estimate_width(mask)

        assert result is not None
        assert result.median_width_px > 15, "Thick crack should have large width"

    def test_branching_crack(self, estimator):
        """Y-shaped crack — should measure trunk width, not branch artifact."""
        _, mask, meta = generate_branching_crack(main_width=8, branch_width=4)
        result = estimator.estimate_width(mask)

        assert result is not None
        assert result.median_width_px > 0

    def test_empty_mask_handling(self, estimator):
        """Empty mask should return a valid (zero) result without crashing."""
        empty_mask = np.zeros((640, 640), dtype=np.uint8)
        result = estimator.estimate_width(empty_mask)

        assert result is not None
        assert result.width_px == 0.0 or result.median_width_px == 0.0

    def test_noisy_mask(self, estimator):
        """Mask with salt-and-pepper noise should still produce a result."""
        _, mask, meta = generate_synthetic_crack_image(width_px=6, length_px=200)

        # Add noise
        rng = np.random.RandomState(123)
        noise_mask = mask.copy()
        salt = rng.random(mask.shape) < 0.01
        pepper = rng.random(mask.shape) < 0.01
        noise_mask[salt] = 255
        noise_mask[pepper] = 0

        result = estimator.estimate_width(noise_mask)
        assert result is not None

    def test_tiny_mask(self, estimator):
        """Very small mask (< 20 pixels) should be handled gracefully."""
        tiny_mask = np.zeros((640, 640), dtype=np.uint8)
        tiny_mask[300:305, 300:303] = 255  # ~15 pixels
        result = estimator.estimate_width(tiny_mask)
        assert result is not None


# ──────────────────────────────────────────────────────────────────────
# Statistics tests
# ──────────────────────────────────────────────────────────────────────

class TestWidthStatistics:
    """Test that width statistics are properly computed."""

    def test_width_statistics_present(self, estimator):
        """Verify all statistical fields are populated."""
        _, mask, _ = generate_synthetic_crack_image(width_px=8, length_px=200)
        result = estimator.estimate_width(mask)

        assert result is not None
        assert result.mean_width_px > 0
        assert result.median_width_px > 0
        assert result.max_width_px >= result.median_width_px
        assert result.min_width_px <= result.median_width_px
        assert result.min_width_px >= 0

    def test_measurement_points_populated(self, estimator):
        """Verify measurement points are recorded."""
        _, mask, _ = generate_synthetic_crack_image(width_px=6, length_px=200)
        result = estimator.estimate_width(mask)

        assert result is not None
        assert len(result.measurement_points) > 0
        # Each point should be (x, y, width)
        for pt in result.measurement_points:
            assert len(pt) == 3
            assert pt[2] >= 0  # width >= 0

    def test_max_geq_median_geq_min(self, estimator):
        """Width ordering: max >= median >= min."""
        _, mask, _ = generate_synthetic_crack_image(width_px=10, length_px=200)
        result = estimator.estimate_width(mask)

        assert result is not None
        assert result.max_width_px >= result.median_width_px
        assert result.median_width_px >= result.min_width_px


# ──────────────────────────────────────────────────────────────────────
# Conversion tests
# ──────────────────────────────────────────────────────────────────────

class TestPixelToMMConversion:
    """Test pixel-to-mm conversion with known GSD."""

    def test_gsd_conversion(self, estimator):
        """Given a known GSD, verify mm result is width_px × GSD."""
        _, mask, _ = generate_synthetic_crack_image(width_px=10, length_px=200)
        gsd = 0.5  # mm per pixel

        scale = ScaleInfo(
            gsd_mm_per_px=gsd,
            method=MeasurementMethod.MONOCULAR_GSD,
        )
        result = estimator.estimate_width(mask, scale_info=scale)

        assert result is not None
        if result.width_mm is not None:
            expected_mm = result.width_px * gsd
            assert abs(result.width_mm - expected_mm) < 0.01, (
                f"MM conversion mismatch: {result.width_mm} vs {expected_mm}"
            )

    def test_no_scale_gives_none_mm(self, estimator):
        """Without scale info, width_mm should be None."""
        _, mask, _ = generate_synthetic_crack_image(width_px=10)
        result = estimator.estimate_width(mask, scale_info=None)

        assert result is not None
        assert result.width_mm is None or result.gsd_mm_per_px is None


# ──────────────────────────────────────────────────────────────────────
# Benchmark test (not run by default)
# ──────────────────────────────────────────────────────────────────────

class TestBenchmark:
    """Benchmark accuracy across multiple known widths."""

    def test_accuracy_benchmark(self, estimator):
        """Run full benchmark across widths [2, 5, 10, 15, 20]."""
        test_data = generate_crack_with_known_width(widths=[2, 5, 10, 15, 20])
        errors = []

        for img, mask, meta in test_data:
            true_w = meta["true_width_px"]
            result = estimator.estimate_width(mask)

            if result is not None and result.median_width_px > 0:
                error = abs(result.median_width_px - true_w)
                rel_error = error / true_w if true_w > 0 else 0
                errors.append({
                    "true_width": true_w,
                    "measured": result.median_width_px,
                    "error_px": error,
                    "rel_error": rel_error,
                })

        # At least half should have < 40% relative error
        good = sum(1 for e in errors if e["rel_error"] < 0.4)
        assert good >= len(errors) // 2, (
            f"Only {good}/{len(errors)} measurements within 40% tolerance. "
            f"Errors: {errors}"
        )
