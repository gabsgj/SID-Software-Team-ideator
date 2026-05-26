"""
Severity Classification
=======================
IS 456:2000-compliant crack severity classification with fall-back
heuristic grading when absolute millimetre measurements are unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

from crack_detection.schemas import (
    ExposureClass,
    MeasurementMethod,
    SeverityClassification,
    SeverityLevel,
    WidthMeasurement,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IS 456:2000 — Table 35 — Permissible crack widths (mm)
# ---------------------------------------------------------------------------
IS456_LIMITS: dict[ExposureClass, float] = {
    ExposureClass.MILD: 0.30,      # mm
    ExposureClass.MODERATE: 0.20,  # mm
    ExposureClass.SEVERE: 0.10,    # mm
}


class SeverityClassifier:
    """Classify crack severity using IS 456:2000 limits or pixel heuristics.

    Parameters
    ----------
    exposure_class:
        Environmental exposure class of the bridge element being
        inspected.  Defaults to ``MODERATE`` (the most common field
        condition for Indian bridges).
    """

    def __init__(
        self,
        exposure_class: ExposureClass = ExposureClass.MODERATE,
    ) -> None:
        self.exposure_class = exposure_class

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        width_measurement: WidthMeasurement,
    ) -> SeverityClassification:
        """Classify severity from a width measurement.

        If calibrated millimetre values are available the classification
        is based on IS 456:2000 limits.  Otherwise a heuristic on the
        pixel width relative to image diagonal is used.

        Parameters
        ----------
        width_measurement:
            Completed :class:`WidthMeasurement` from the width-estimation
            stage.

        Returns
        -------
        SeverityClassification
        """
        if (
            width_measurement.width_mm is not None
            and width_measurement.method != MeasurementMethod.RELATIVE_ONLY
        ):
            return self._classify_by_mm(
                width_measurement.width_mm, self.exposure_class
            )

        # Fall back to relative (pixel-based) classification
        return self._classify_by_relative(width_measurement.width_px)

    # ------------------------------------------------------------------
    # IS 456:2000 classification
    # ------------------------------------------------------------------

    def _classify_by_mm(
        self,
        width_mm: float,
        exposure_class: ExposureClass,
    ) -> SeverityClassification:
        """Classify based on absolute width in millimetres.

        Thresholds
        ----------
        * **MINOR**: < 0.10 mm
        * **MODERATE**: < permissible limit for the exposure class
        * **SEVERE**: < 2 × permissible limit
        * **CRITICAL**: ≥ 2 × permissible limit

        Parameters
        ----------
        width_mm:
            Measured crack width in millimetres.
        exposure_class:
            IS 456 exposure class.

        Returns
        -------
        SeverityClassification
        """
        permissible = IS456_LIMITS[exposure_class]

        if width_mm < 0.10:
            level = SeverityLevel.MINOR
        elif width_mm < permissible:
            level = SeverityLevel.MODERATE
        elif width_mm < 2.0 * permissible:
            level = SeverityLevel.SEVERE
        else:
            level = SeverityLevel.CRITICAL

        compliant = width_mm <= permissible

        notes = self._get_remediation_notes(level)
        logger.info(
            "Severity (mm): %.3f mm → %s (permissible=%.2f mm, class=%s, "
            "compliant=%s).",
            width_mm,
            level.value,
            permissible,
            exposure_class.value,
            compliant,
        )

        return SeverityClassification(
            level=level,
            basis=f"IS 456:2000 width comparison (exposure={exposure_class.value})",
            exposure_class=exposure_class,
            is456_compliant=compliant,
            permissible_width_mm=permissible,
            measured_width_mm=width_mm,
            remediation_notes=notes,
        )

    # ------------------------------------------------------------------
    # Relative / heuristic classification
    # ------------------------------------------------------------------

    def _classify_by_relative(
        self,
        relative_width: float,
    ) -> SeverityClassification:
        """Classify when only pixel-domain width is available.

        The ``relative_width`` is expected to be the crack width in pixels
        (or a normalised ratio).  The thresholds here are intentionally
        conservative so that cracks are not under-classified.

        Thresholds (pixel-based)
        ------------------------
        * **MINOR**: < 5 px
        * **MODERATE**: < 15 px
        * **SEVERE**: < 30 px
        * **CRITICAL**: ≥ 30 px

        Parameters
        ----------
        relative_width:
            Width value (pixels or normalised ratio).

        Returns
        -------
        SeverityClassification
        """
        if relative_width < 5.0:
            level = SeverityLevel.MINOR
        elif relative_width < 15.0:
            level = SeverityLevel.MODERATE
        elif relative_width < 30.0:
            level = SeverityLevel.SEVERE
        else:
            level = SeverityLevel.CRITICAL

        notes = self._get_remediation_notes(level)
        logger.info(
            "Severity (relative): %.2f px → %s (heuristic).",
            relative_width,
            level.value,
        )

        return SeverityClassification(
            level=level,
            basis="Relative pixel width (heuristic — no calibration)",
            exposure_class=None,
            is456_compliant=None,
            permissible_width_mm=None,
            measured_width_mm=None,
            remediation_notes=notes,
        )

    # ------------------------------------------------------------------
    # Remediation notes
    # ------------------------------------------------------------------

    @staticmethod
    def _get_remediation_notes(level: SeverityLevel) -> str:
        """Return human-readable remediation guidance for a severity level.

        Parameters
        ----------
        level:
            Severity level.

        Returns
        -------
        str
        """
        notes = {
            SeverityLevel.MINOR: (
                "Monitor during next inspection cycle."
            ),
            SeverityLevel.MODERATE: (
                "Schedule epoxy injection repair within 6 months."
            ),
            SeverityLevel.SEVERE: (
                "Urgent repair required — structural epoxy injection or patch."
            ),
            SeverityLevel.CRITICAL: (
                "IMMEDIATE action — structural assessment and emergency "
                "repair required."
            ),
        }
        return notes.get(level, "")
