"""
Width Estimation — Common Utilities
====================================
Shared data structures and helper functions used by both the monocular and
stereo width-estimation modules.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from crack_detection.schemas import MeasurementMethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scale / calibration data
# ---------------------------------------------------------------------------

class ScaleInfo(BaseModel):
    """
    Encapsulates the pixel-to-real-world scaling information required for
    converting crack widths from pixels to millimetres.
    """

    gsd_mm_per_px: Optional[float] = Field(
        None,
        description="Ground Sampling Distance in mm per pixel.",
    )
    method: MeasurementMethod = Field(
        MeasurementMethod.RELATIVE_ONLY,
        description="Method used to derive the scale.",
    )
    distance_mm: Optional[float] = Field(
        None,
        description="Distance from the camera to the target surface (mm).",
    )
    focal_length_px: Optional[float] = Field(
        None,
        description="Camera focal length expressed in pixels.",
    )


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def pixels_to_mm(width_px: float, scale_info: ScaleInfo) -> Optional[float]:
    """Convert a pixel-domain width to millimetres.

    Parameters
    ----------
    width_px:
        Width measured in pixels.
    scale_info:
        Scaling information (must contain ``gsd_mm_per_px``).

    Returns
    -------
    Optional[float]
        Width in millimetres, or ``None`` if no valid scale is available.
    """
    if scale_info is None:
        logger.debug("No scale_info provided — cannot convert to mm.")
        return None

    if scale_info.gsd_mm_per_px is not None and scale_info.gsd_mm_per_px > 0:
        return width_px * scale_info.gsd_mm_per_px

    logger.debug(
        "ScaleInfo lacks a valid gsd_mm_per_px — returning None."
    )
    return None


def compute_gsd(
    altitude_m: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: float,
) -> float:
    """Compute Ground Sampling Distance (GSD) in mm/pixel.

    Formula
    -------
    ``GSD = (altitude × sensor_width) / (focal_length × image_width)``

    All inputs are in consistent units (altitude in **metres** is first
    converted to mm internally).

    Parameters
    ----------
    altitude_m:
        Drone altitude above the target surface in metres.
    focal_length_mm:
        Camera focal length in millimetres.
    sensor_width_mm:
        Physical width of the camera sensor in millimetres.
    image_width_px:
        Image width in pixels.

    Returns
    -------
    float
        GSD in mm/pixel.

    Raises
    ------
    ValueError
        If any input is non-positive.
    """
    if altitude_m <= 0:
        raise ValueError(f"altitude_m must be positive, got {altitude_m}")
    if focal_length_mm <= 0:
        raise ValueError(
            f"focal_length_mm must be positive, got {focal_length_mm}"
        )
    if sensor_width_mm <= 0:
        raise ValueError(
            f"sensor_width_mm must be positive, got {sensor_width_mm}"
        )
    if image_width_px <= 0:
        raise ValueError(
            f"image_width_px must be positive, got {image_width_px}"
        )

    altitude_mm = altitude_m * 1000.0
    gsd = (altitude_mm * sensor_width_mm) / (focal_length_mm * image_width_px)
    logger.info(
        "Computed GSD = %.4f mm/px  (alt=%.1fm, fl=%.1fmm, sw=%.1fmm, iw=%dpx)",
        gsd,
        altitude_m,
        focal_length_mm,
        sensor_width_mm,
        int(image_width_px),
    )
    return gsd
