"""
Crack Detector
==============
YOLO-based crack detection and instance segmentation using a trained
YOLOv11s-seg model.  Accepts single images or batches and returns
structured :class:`CrackDetection` results.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
from ultralytics import YOLO

from crack_detection.schemas import (
    CrackDetection,
    CrackGeometry,
    MeasurementMethod,
    SeverityClassification,
    SeverityLevel,
    WidthMeasurement,
)

logger = logging.getLogger(__name__)


class CrackDetector:
    """High-level crack detector wrapping the Ultralytics YOLO API.

    Parameters
    ----------
    model_path:
        Path to the trained ``.pt`` weights file.
    confidence:
        Default confidence threshold for detections.
    device:
        Inference device — ``'auto'``, ``'cpu'``, ``'cuda'``, or a device id.
    """

    def __init__(
        self,
        model_path: str = "best.pt",
        confidence: float = 0.45,
        device: str = "auto",
    ) -> None:
        self.model_path = str(model_path)
        self.confidence = confidence
        self.default_confidence = confidence
        self.device = device if device != "auto" else ""

        logger.info("Loading YOLO model from %s …", self.model_path)
        self.model = YOLO(self.model_path)
        logger.info("YOLO model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
    ) -> List[CrackDetection]:
        """Run crack detection on a single image.

        Parameters
        ----------
        image:
            Either a file-system path (``str`` / ``Path``) or a BGR
            ``numpy.ndarray`` as returned by ``cv2.imread``.
        conf:
            Confidence threshold override.  Falls back to
            ``self.default_confidence`` when ``None``.

        Returns
        -------
        List[CrackDetection]
            One entry per detected crack.  The ``width`` field contains
            only pixel-level placeholders — real width estimation is
            performed downstream by the pipeline.
        """
        conf = conf if conf is not None else self.default_confidence

        # Load image if a path was given
        if isinstance(image, (str, Path)):
            img_path = str(image)
            img_array = cv2.imread(img_path)
            if img_array is None:
                raise FileNotFoundError(
                    f"Could not read image at '{img_path}'."
                )
        else:
            img_array = image

        img_shape = img_array.shape[:2]  # (H, W)

        # Run YOLO inference
        results = self.model.predict(
            source=img_array,
            conf=conf,
            device=self.device or None,
            verbose=False,
        )

        detections: List[CrackDetection] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for idx in range(len(result.boxes)):
                det = self._build_detection(result, idx, img_shape, frame_idx=0)
                detections.append(det)

        logger.info("Detected %d crack(s) at conf >= %.2f.", len(detections), conf)
        return detections

    def detect_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        conf: Optional[float] = None,
    ) -> List[List[CrackDetection]]:
        """Run crack detection on a batch of images.

        Parameters
        ----------
        images:
            Iterable of images (paths or arrays).
        conf:
            Confidence threshold override.

        Returns
        -------
        List[List[CrackDetection]]
            Outer list corresponds to each input image.
        """
        conf = conf if conf is not None else self.default_confidence
        batch_results: List[List[CrackDetection]] = []

        for frame_idx, image in enumerate(images):
            try:
                dets = self.detect(image, conf=conf)
                # Re-stamp detection IDs with correct frame index
                for det_idx, det in enumerate(dets):
                    det.detection_id = self._generate_detection_id(
                        frame_idx, det_idx
                    )
                batch_results.append(dets)
            except Exception:
                logger.exception(
                    "Failed to process image at batch index %d.", frame_idx
                )
                batch_results.append([])

        return batch_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_detection(
        self,
        result,
        idx: int,
        img_shape: tuple,
        frame_idx: int = 0,
    ) -> CrackDetection:
        """Construct a :class:`CrackDetection` from a single YOLO result row."""
        # Bounding box (xyxy)
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[idx].tolist())
        confidence = float(result.boxes.conf[idx])

        # Segmentation mask
        mask = self._extract_mask(result, idx, img_shape)

        # Contour polygon (simplified)
        polygon = self._mask_to_polygon(mask)

        # Placeholder geometry — real values computed downstream
        geometry = CrackGeometry(
            length_px=0.0,
            orientation_deg=0.0,
            curvature=0.0,
            skeleton_points=[],
            area_px=float(np.count_nonzero(mask)) if mask is not None else 0.0,
        )

        # Placeholder width (to be filled by width-estimation stage)
        width = WidthMeasurement(
            width_px=0.0,
            method=MeasurementMethod.RELATIVE_ONLY,
            measurement_points=[],
        )

        # Placeholder severity
        severity = SeverityClassification(
            level=SeverityLevel.MINOR,
            basis="pending",
            remediation_notes="",
        )

        detection_id = self._generate_detection_id(frame_idx, idx)

        return CrackDetection(
            detection_id=detection_id,
            bbox=(x1, y1, x2, y2),
            confidence=confidence,
            geometry=geometry,
            width=width,
            severity=severity,
            mask_polygon=polygon,
        )

    def _extract_mask(
        self,
        result,
        idx: int,
        img_shape: tuple,
    ) -> Optional[np.ndarray]:
        """Extract a full-resolution binary mask for detection ``idx``.

        The YOLO segmentation model stores masks in ``result.masks.data``
        at the model's internal resolution.  This method resizes the mask
        to match the original image dimensions.

        Parameters
        ----------
        result:
            A single Ultralytics ``Results`` object.
        idx:
            Index of the detection within the result.
        img_shape:
            ``(H, W)`` of the original image.

        Returns
        -------
        Optional[np.ndarray]
            Binary mask of shape ``(H, W)`` with dtype ``uint8``,
            or ``None`` if no mask data is available.
        """
        if result.masks is None:
            logger.warning(
                "No segmentation masks in result — model may be detection-only."
            )
            return None

        try:
            mask_tensor = result.masks.data[idx]  # shape: (mH, mW)
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8)

            # Resize to original image resolution
            h, w = img_shape[:2]
            if mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(
                    mask_np, (w, h), interpolation=cv2.INTER_NEAREST
                )

            return mask_np
        except (IndexError, AttributeError) as exc:
            logger.warning("Could not extract mask for idx %d: %s", idx, exc)
            return None

    @staticmethod
    def _mask_to_polygon(
        mask: Optional[np.ndarray],
        simplify_epsilon: float = 2.0,
    ) -> Optional[List[tuple]]:
        """Convert a binary mask to a simplified contour polygon.

        Parameters
        ----------
        mask:
            Binary mask ``(H, W)``.
        simplify_epsilon:
            Douglas–Peucker approximation tolerance in pixels.

        Returns
        -------
        Optional[List[Tuple[float, float]]]
            Simplified polygon vertices, or ``None`` if extraction fails.
        """
        if mask is None:
            return None

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Pick the largest contour
        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, simplify_epsilon, closed=True)
        return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]

    @staticmethod
    def _generate_detection_id(frame_idx: int, det_idx: int) -> str:
        """Generate a deterministic, human-readable detection identifier.

        Format: ``frame{frame_idx:04d}_det{det_idx:03d}_{short_uuid}``
        """
        short = uuid.uuid4().hex[:8]
        return f"frame{frame_idx:04d}_det{det_idx:03d}_{short}"
