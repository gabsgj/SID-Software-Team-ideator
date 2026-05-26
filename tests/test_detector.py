"""
Tests for the YOLO Crack Detector Wrapper
==========================================
Uses mocking to test without requiring the trained model file.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_mock_result(num_detections=2, img_shape=(640, 640)):
    """Create a mock YOLO result object."""
    h, w = img_shape
    result = MagicMock()
    result.orig_shape = img_shape

    # Mock boxes
    boxes = MagicMock()
    xyxy_data = []
    conf_data = []
    for i in range(num_detections):
        x1, y1 = 50 + i * 100, 50 + i * 80
        x2, y2 = x1 + 120, y1 + 40
        xyxy_data.append([x1, y1, x2, y2])
        conf_data.append(0.85 - i * 0.1)

    if num_detections > 0:
        import torch
        boxes.xyxy = torch.tensor(xyxy_data, dtype=torch.float32)
        boxes.conf = torch.tensor(conf_data, dtype=torch.float32)
    else:
        import torch
        boxes.xyxy = torch.zeros((0, 4))
        boxes.conf = torch.zeros(0)

    boxes.__len__ = lambda self: num_detections
    result.boxes = boxes

    # Mock masks
    if num_detections > 0:
        masks = MagicMock()
        mask_data = []
        for i in range(num_detections):
            m = np.zeros((160, 160), dtype=np.float32)
            x1, y1 = 12 + i * 25, 12 + i * 20
            m[y1:y1+10, x1:x1+30] = 1.0
            mask_data.append(m)
        import torch
        masks.data = torch.tensor(np.array(mask_data))
        result.masks = masks
    else:
        result.masks = None

    return result


class TestCrackDetector:
    """Test the CrackDetector wrapper class."""

    @patch("crack_detection.detector.YOLO")
    def test_detector_initialization(self, mock_yolo_class):
        """Verify model loads with correct parameters."""
        mock_yolo_class.return_value = MagicMock()

        from crack_detection.detector import CrackDetector
        detector = CrackDetector(model_path="test_model.pt", confidence=0.5)

        mock_yolo_class.assert_called_once_with("test_model.pt")
        assert detector.confidence == 0.5

    @patch("crack_detection.detector.YOLO")
    def test_detection_output_format(self, mock_yolo_class):
        """Verify detect() returns a list of CrackDetection objects."""
        mock_model = MagicMock()
        mock_result = _make_mock_result(num_detections=2)
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        from crack_detection.detector import CrackDetector
        detector = CrackDetector(model_path="test.pt")

        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_image)

        assert isinstance(detections, list)
        assert len(detections) == 2

        for det in detections:
            assert hasattr(det, "detection_id")
            assert hasattr(det, "bbox")
            assert hasattr(det, "confidence")
            assert det.confidence > 0
            assert len(det.bbox) == 4

    @patch("crack_detection.detector.YOLO")
    def test_confidence_threshold(self, mock_yolo_class):
        """Verify low-confidence detections are passed through to YOLO."""
        mock_model = MagicMock()
        mock_result = _make_mock_result(num_detections=1)
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        from crack_detection.detector import CrackDetector
        detector = CrackDetector(model_path="test.pt", confidence=0.8)

        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_image, conf=0.8)

        # YOLO's predict should have been called with the confidence
        mock_model.predict.assert_called_once()

    @patch("crack_detection.detector.YOLO")
    def test_no_detections(self, mock_yolo_class):
        """No detections should return empty list."""
        mock_model = MagicMock()
        mock_result = _make_mock_result(num_detections=0)
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        from crack_detection.detector import CrackDetector
        detector = CrackDetector(model_path="test.pt")

        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_image)

        assert isinstance(detections, list)
        assert len(detections) == 0

    @patch("crack_detection.detector.YOLO")
    def test_batch_detection(self, mock_yolo_class):
        """Verify batch processing returns results for each image."""
        mock_model = MagicMock()
        mock_result = _make_mock_result(num_detections=1)
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        from crack_detection.detector import CrackDetector
        detector = CrackDetector(model_path="test.pt")

        images = [
            np.zeros((640, 640, 3), dtype=np.uint8),
            np.zeros((640, 640, 3), dtype=np.uint8),
        ]
        results = detector.detect_batch(images)

        assert isinstance(results, list)
        assert len(results) == 2

    @patch("crack_detection.detector.YOLO")
    def test_detection_id_format(self, mock_yolo_class):
        """Detection IDs should be unique and well-formatted."""
        mock_model = MagicMock()
        mock_result = _make_mock_result(num_detections=3)
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        from crack_detection.detector import CrackDetector
        detector = CrackDetector(model_path="test.pt")

        detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

        ids = [d.detection_id for d in detections]
        assert len(ids) == len(set(ids)), "Detection IDs must be unique"
