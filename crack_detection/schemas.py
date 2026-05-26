"""
Crack Detection Schemas
=======================
Pydantic v2 data models for crack detection results, width measurements,
severity classification, and inspection reports.

These models are the canonical data contract for the entire crack-detection
pipeline — from raw YOLO output through width estimation to the final
inspection report.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SeverityLevel(str, Enum):
    """Crack severity classification levels."""

    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"


class ExposureClass(str, Enum):
    """Exposure classification per IS 456:2000 Table 3."""

    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


class MeasurementMethod(str, Enum):
    """Method used to estimate real-world crack width."""

    MONOCULAR_GSD = "MONOCULAR_GSD"
    MONOCULAR_REFERENCE = "MONOCULAR_REFERENCE"
    STEREO_DEPTH = "STEREO_DEPTH"
    RELATIVE_ONLY = "RELATIVE_ONLY"


# ---------------------------------------------------------------------------
# Width Measurement
# ---------------------------------------------------------------------------

class WidthMeasurement(BaseModel):
    """
    Aggregated crack-width measurements taken along the crack skeleton.

    Each entry in ``measurement_points`` is a *(x, y, width_px)* tuple
    representing a single perpendicular-width sample on the crack mask.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    width_px: float = Field(
        ..., description="Representative width in pixels (median of samples)."
    )
    width_mm: Optional[float] = Field(
        None, description="Representative width in millimetres (if calibrated)."
    )
    method: MeasurementMethod = Field(
        ..., description="Method used to obtain the width."
    )

    # Per-sample measurement points: list of (x, y, width_px)
    measurement_points: List[Tuple[float, float, float]] = Field(
        default_factory=list,
        description="Individual (x, y, width_px) samples along the skeleton.",
    )

    # Statistical aggregates (all in pixels)
    mean_width_px: float = Field(0.0, description="Mean width across samples.")
    median_width_px: float = Field(0.0, description="Median width across samples.")
    max_width_px: float = Field(0.0, description="Maximum width across samples.")
    min_width_px: float = Field(0.0, description="Minimum width across samples.")
    std_width_px: float = Field(0.0, description="Std-dev of width across samples.")
    percentile_95_width_px: float = Field(
        0.0, description="95th-percentile width across samples."
    )

    # Calibration info
    gsd_mm_per_px: Optional[float] = Field(
        None, description="Ground sampling distance (mm/px) used for conversion."
    )
    scale_factor: Optional[float] = Field(
        None,
        description="Generic scale factor (mm/px) derived from reference object.",
    )


# ---------------------------------------------------------------------------
# Crack Geometry
# ---------------------------------------------------------------------------

class CrackGeometry(BaseModel):
    """Geometric properties of a single detected crack."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    length_px: float = Field(..., description="Crack length in pixels.")
    length_mm: Optional[float] = Field(
        None, description="Crack length in millimetres (if calibrated)."
    )
    orientation_deg: float = Field(
        ..., description="Dominant orientation in degrees [0, 180)."
    )
    curvature: float = Field(
        0.0, description="Curvature metric (sum of angular change / length)."
    )

    # Ordered skeleton coordinates
    skeleton_points: List[Tuple[float, float]] = Field(
        default_factory=list,
        description="Ordered (x, y) points along the crack skeleton.",
    )

    area_px: float = Field(0.0, description="Crack area in pixels.")


# ---------------------------------------------------------------------------
# Severity Classification
# ---------------------------------------------------------------------------

class SeverityClassification(BaseModel):
    """Result of severity classification for a single crack."""

    level: SeverityLevel = Field(
        ..., description="Assigned severity level."
    )
    basis: str = Field(
        ...,
        description="Human-readable description of the measurement basis "
        "(e.g. 'IS 456 width comparison', 'relative pixel width').",
    )
    exposure_class: Optional[ExposureClass] = Field(
        None, description="Exposure class used for IS 456 comparison."
    )
    is456_compliant: Optional[bool] = Field(
        None,
        description="True if measured width is within IS 456 permissible limit.",
    )
    permissible_width_mm: Optional[float] = Field(
        None, description="IS 456 permissible width for the exposure class (mm)."
    )
    measured_width_mm: Optional[float] = Field(
        None, description="Measured crack width used for classification (mm)."
    )
    remediation_notes: str = Field(
        "", description="Recommended remediation action."
    )


# ---------------------------------------------------------------------------
# Single Crack Detection
# ---------------------------------------------------------------------------

class CrackDetection(BaseModel):
    """Complete data for one detected crack in an image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    detection_id: str = Field(
        ..., description="Unique identifier (e.g. 'frame001_det003')."
    )
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box (x1, y1, x2, y2) in pixel coordinates."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="YOLO detection confidence."
    )
    geometry: CrackGeometry = Field(
        ..., description="Geometric properties of the crack."
    )
    width: WidthMeasurement = Field(
        ..., description="Width-measurement results."
    )
    severity: SeverityClassification = Field(
        ..., description="Severity classification."
    )
    mask_polygon: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Simplified contour polygon as list of (x, y) points.",
    )


# ---------------------------------------------------------------------------
# Full Inspection Result
# ---------------------------------------------------------------------------

class InspectionResult(BaseModel):
    """Top-level result for a single image inspection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(..., description="Inspection session identifier.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the inspection.",
    )
    source: str = Field(
        "drone", description="Image source (e.g. 'drone', 'webcam', 'file')."
    )
    image_path: str = Field(
        ..., description="Path or URI of the inspected image."
    )
    image_width: int = Field(..., description="Image width in pixels.")
    image_height: int = Field(..., description="Image height in pixels.")

    model_name: str = Field(
        "YOLOv11s-seg", description="Detection model architecture."
    )
    model_weights: str = Field(
        "best.pt", description="Path to model weights used."
    )

    detections: List[CrackDetection] = Field(
        default_factory=list, description="List of crack detections."
    )
    total_detections: int = Field(
        0, description="Total number of cracks detected."
    )
    highest_severity: Optional[SeverityLevel] = Field(
        None, description="Highest severity among all detections."
    )
    flagged: bool = Field(
        False,
        description="True if any detection is SEVERE or CRITICAL.",
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Arbitrary metadata (camera info, drone telemetry, etc.).",
    )
