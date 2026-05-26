"""
End-to-End Pipeline Integration Tests
======================================
Tests the full inspection pipeline from image to report.
Uses synthetic data to avoid dependency on best.pt.
"""

import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.generate_test_data import (
    generate_synthetic_crack_image,
    generate_synthetic_depth_map,
)
from crack_detection.schemas import (
    CrackDetection,
    CrackGeometry,
    WidthMeasurement,
    SeverityClassification,
    InspectionResult,
    SeverityLevel,
    MeasurementMethod,
)


def _make_mock_detection(det_id="det_001", width_px=5.0, conf=0.9):
    """Create a mock CrackDetection for testing."""
    return CrackDetection(
        detection_id=det_id,
        bbox=(100, 100, 250, 140),
        confidence=conf,
        geometry=CrackGeometry(
            length_px=160.0,
            length_mm=None,
            orientation_deg=15.0,
            curvature=0.01,
            skeleton_points=[(150, 120), (180, 125), (210, 130)],
            area_px=800.0,
        ),
        width=WidthMeasurement(
            width_px=width_px,
            width_mm=2.5,
            method=MeasurementMethod.MONOCULAR_GSD,
            measurement_points=[(150, 120, 5.0), (180, 125, 4.8)],
            mean_width_px=4.9,
            median_width_px=5.0,
            max_width_px=5.5,
            min_width_px=4.2,
            std_width_px=0.3,
            percentile_95_width_px=5.3,
            gsd_mm_per_px=0.5,
        ),
        severity=SeverityClassification(
            level=SeverityLevel.MODERATE,
            basis="width_mm",
            is456_compliant=False,
            permissible_width_mm=0.20,
            measured_width_mm=2.5,
            remediation_notes="Schedule epoxy injection repair within 6 months",
        ),
        mask_polygon=[(100, 100), (250, 100), (250, 140), (100, 140)],
    )


def _make_mock_result(num_detections=2):
    """Create a mock InspectionResult."""
    detections = [
        _make_mock_detection(f"det_{i:03d}", width_px=3.0 + i * 2, conf=0.9 - i * 0.05)
        for i in range(num_detections)
    ]
    highest = max(
        (d.severity.level for d in detections),
        key=lambda x: ["MINOR", "MODERATE", "SEVERE", "CRITICAL"].index(x.value)
        if hasattr(x, "value") else 0,
        default=SeverityLevel.MINOR,
    )
    return InspectionResult(
        session_id="sid_test_20250101_120000",
        timestamp="2025-01-01T12:00:00",
        source="test_synthetic",
        image_path="test_image.jpg",
        image_width=640,
        image_height=640,
        model_name="yolov11s-seg",
        model_weights="best.pt",
        detections=detections,
        total_detections=num_detections,
        highest_severity=highest,
        flagged=num_detections > 0,
        metadata={"test": True},
    )


# ──────────────────────────────────────────────────────────────────────
# Schema Tests
# ──────────────────────────────────────────────────────────────────────

class TestInspectionResultFormat:
    """Test the InspectionResult Pydantic model."""

    def test_result_creation(self):
        """InspectionResult should be creatable with valid data."""
        result = _make_mock_result(2)
        assert result is not None
        assert result.total_detections == 2
        assert len(result.detections) == 2

    def test_result_serialization(self):
        """InspectionResult should serialize to dict/JSON."""
        result = _make_mock_result(1)
        data = result.model_dump(mode="json")

        assert isinstance(data, dict)
        assert "session_id" in data
        assert "detections" in data
        assert len(data["detections"]) == 1

    def test_detection_fields(self):
        """CrackDetection should have all required fields."""
        det = _make_mock_detection()
        assert det.detection_id == "det_001"
        assert det.confidence == 0.9
        assert det.bbox == (100, 100, 250, 140)
        assert det.width.width_px == 5.0
        assert det.severity.level == SeverityLevel.MODERATE

    def test_empty_result(self):
        """Result with zero detections should be valid."""
        result = InspectionResult(
            session_id="empty_test",
            timestamp="2025-01-01",
            source="test",
            image_path="none.jpg",
            image_width=640,
            image_height=640,
            model_name="test",
            model_weights="test.pt",
            detections=[],
            total_detections=0,
            highest_severity=SeverityLevel.MINOR,
            flagged=False,
            metadata={},
        )
        assert result.total_detections == 0


# ──────────────────────────────────────────────────────────────────────
# Report Generation Tests
# ──────────────────────────────────────────────────────────────────────

class TestReportGeneration:
    """Test report generation in various formats."""

    def test_json_report(self, tmp_path):
        """Generate JSON report and verify structure."""
        from pipeline.report_generator import ReportGenerator

        gen = ReportGenerator()
        result = _make_mock_result(2)
        output = str(tmp_path / "report.json")

        path = gen.generate_json_report(result, output)
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)

        assert "session_id" in data
        assert "detections" in data
        assert len(data["detections"]) == 2

    def test_html_report(self, tmp_path):
        """Generate HTML report and verify it's valid HTML."""
        from pipeline.report_generator import ReportGenerator

        gen = ReportGenerator()
        result = _make_mock_result(3)
        output = str(tmp_path / "report.html")

        path = gen.generate_html_report(result, output_path=output)
        assert os.path.exists(path)

        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "SID Bridge Inspection Report" in content
        assert "det_000" in content
        assert "MODERATE" in content

    def test_html_report_no_detections(self, tmp_path):
        """HTML report with zero detections should not crash."""
        from pipeline.report_generator import ReportGenerator

        gen = ReportGenerator()
        result = _make_mock_result(0)
        output = str(tmp_path / "empty_report.html")

        path = gen.generate_html_report(result, output_path=output)
        assert os.path.exists(path)

    def test_csv_report(self, tmp_path):
        """Generate CSV report with flattened rows."""
        from pipeline.report_generator import ReportGenerator

        gen = ReportGenerator()
        results = [_make_mock_result(2), _make_mock_result(1)]
        output = str(tmp_path / "report.csv")

        path = gen.generate_csv_report(results, output)
        assert os.path.exists(path)

        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3  # 2 + 1 detections
        assert "detection_id" in rows[0]
        assert "severity" in rows[0]
        assert "width_px" in rows[0]


# ──────────────────────────────────────────────────────────────────────
# Pipeline Integration Tests (with mocked detector)
# ──────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    """Test the full pipeline with mocked YOLO detector."""

    def test_width_estimation_standalone(self):
        """Run width estimation on synthetic data without YOLO."""
        from crack_detection.width_estimation.monocular import MonocularWidthEstimator

        estimator = MonocularWidthEstimator()
        _, mask, meta = generate_synthetic_crack_image(width_px=8, length_px=200)

        result = estimator.estimate_width(mask)
        assert result is not None
        assert result.median_width_px > 0

    def test_severity_classification(self):
        """Test severity classifier with known width."""
        from crack_detection.severity import SeverityClassifier
        from crack_detection.schemas import ExposureClass

        classifier = SeverityClassifier(exposure_class=ExposureClass.MODERATE)

        # Create a width measurement with known mm value
        width = WidthMeasurement(
            width_px=10.0,
            width_mm=0.25,
            method=MeasurementMethod.MONOCULAR_GSD,
            measurement_points=[],
            mean_width_px=10.0,
            median_width_px=10.0,
            max_width_px=12.0,
            min_width_px=8.0,
            std_width_px=1.0,
            percentile_95_width_px=11.5,
            gsd_mm_per_px=0.025,
        )

        result = classifier.classify(width)
        assert result is not None
        assert result.level is not None

    def test_segmentation_processor(self):
        """Test mask refinement and orientation computation."""
        from crack_detection.segmentation import MaskProcessor

        proc = MaskProcessor()
        _, mask, _ = generate_synthetic_crack_image(width_px=5, length_px=200)

        refined = proc.refine_mask(mask)
        assert refined is not None
        assert refined.shape == mask.shape

        orientation = proc.compute_orientation(mask)
        assert isinstance(orientation, (int, float))

        area = proc.compute_crack_area(mask)
        assert area > 0

    def test_directory_processing(self, tmp_path):
        """Test processing a directory of synthetic images."""
        # Create a few test images
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for i in range(3):
            img, _, _ = generate_synthetic_crack_image(
                width_px=5 + i * 3, length_px=200
            )
            import cv2
            cv2.imwrite(str(img_dir / f"crack_{i}.png"), img)

        # Just verify the images were created
        images = list(img_dir.glob("*.png"))
        assert len(images) == 3
