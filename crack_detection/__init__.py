"""
SID Crack Detection Engine
==========================
Automated crack detection, segmentation, width estimation, and severity
classification for concrete bridge inspection using drone imagery.
"""

from crack_detection.detector import CrackDetector
from crack_detection.segmentation import MaskProcessor
from crack_detection.severity import SeverityClassifier
from crack_detection.schemas import (
    CrackDetection,
    WidthMeasurement,
    SeverityClassification,
    InspectionResult,
)

__all__ = [
    "CrackDetector",
    "MaskProcessor",
    "SeverityClassifier",
    "CrackDetection",
    "WidthMeasurement",
    "SeverityClassification",
    "InspectionResult",
]
