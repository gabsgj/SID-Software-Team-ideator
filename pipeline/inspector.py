"""
Bridge Inspector — End-to-End Orchestrator
============================================
Chains the full inspection pipeline:

    Image → Detection → Mask Refinement → Width Estimation
          → Severity Classification → BIM Mapping → Report

Supports single images, directories, and video inputs.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from crack_detection.detector import CrackDetector
from crack_detection.schemas import (
    CrackDetection,
    CrackGeometry,
    ExposureClass,
    InspectionResult,
    MeasurementMethod,
    SeverityClassification,
    SeverityLevel,
    WidthMeasurement,
)
from crack_detection.segmentation import MaskProcessor
from crack_detection.severity import SeverityClassifier
from crack_detection.width_estimation.calibration import CameraCalibrator
from crack_detection.width_estimation.common import ScaleInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — these may be built by parallel agents
# ---------------------------------------------------------------------------
try:
    from crack_detection.width_estimation.monocular import MonocularWidthEstimator
except ImportError:
    MonocularWidthEstimator = None  # type: ignore[assignment,misc]
    logger.debug("MonocularWidthEstimator not available.")

try:
    from crack_detection.width_estimation.stereo import StereoWidthEstimator
except ImportError:
    StereoWidthEstimator = None  # type: ignore[assignment,misc]
    logger.debug("StereoWidthEstimator not available.")

try:
    from bim_mapping.mapper import CrackBIMMapper
except ImportError:
    CrackBIMMapper = None  # type: ignore[assignment,misc]
    logger.debug("CrackBIMMapper not available.")

try:
    from bim_mapping.visualizer import BIMVisualizer
except ImportError:
    BIMVisualizer = None  # type: ignore[assignment,misc]
    logger.debug("BIMVisualizer not available.")

try:
    from pipeline.report_generator import ReportGenerator
except ImportError:
    ReportGenerator = None  # type: ignore[assignment,misc]
    logger.debug("ReportGenerator not available.")


# =====================================================================
# BridgeInspector
# =====================================================================

class BridgeInspector:
    """End-to-end orchestrator for drone-based bridge crack inspection.

    Initialises all sub-components and exposes high-level ``inspect_*``
    methods that run the full detection → estimation → classification
    → mapping → reporting pipeline.

    Parameters
    ----------
    model_path : str
        Path to YOLOv11s-seg weights (``.pt`` file).
    confidence : float
        Detection confidence threshold.
    method : str
        Width-estimation strategy: ``'monocular'`` or ``'stereo'``.
    exposure_class : str
        IS 456 exposure class name (``MILD``, ``MODERATE``, ``SEVERE``).
    camera_config : dict, optional
        Camera intrinsics with keys ``focal_length_mm``,
        ``sensor_width_mm``, and ``image_width_px``.
    ifc_path : str, optional
        Path to an IFC model for BIM mapping.
    """

    # Supported image extensions (case-insensitive)
    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        model_path: str = "best.pt",
        confidence: float = 0.45,
        method: str = "monocular",
        exposure_class: str = "MODERATE",
        camera_config: Optional[Dict] = None,
        ifc_path: Optional[str] = None,
    ) -> None:
        # ---- Session ----
        self.session_id: str = (
            f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:6]}"
        )
        logger.info("Initialising BridgeInspector — session %s", self.session_id)

        # ---- Core detector ----
        self.detector = CrackDetector(
            model_path=model_path,
            confidence=confidence,
        )

        # ---- Mask processing ----
        self.mask_processor = MaskProcessor()

        # ---- Width estimation ----
        self.method = method.lower()
        self.width_estimator = None
        if self.method == "monocular" and MonocularWidthEstimator is not None:
            self.width_estimator = MonocularWidthEstimator()
            logger.info("Using MonocularWidthEstimator.")
        elif self.method == "stereo" and StereoWidthEstimator is not None:
            self.width_estimator = StereoWidthEstimator()
            logger.info("Using StereoWidthEstimator.")
        else:
            logger.warning(
                "Width estimator for method '%s' is not available. "
                "Width measurements will be pixel-only.",
                self.method,
            )

        # ---- Camera calibration ----
        self.calibrator: Optional[CameraCalibrator] = None
        if camera_config is not None:
            self.calibrator = CameraCalibrator(
                focal_length_mm=camera_config.get("focal_length_mm"),
                sensor_width_mm=camera_config.get("sensor_width_mm"),
                image_width_px=camera_config.get("image_width_px"),
            )
            logger.info("CameraCalibrator initialised with provided config.")

        # ---- Severity ----
        try:
            exp_cls = ExposureClass(exposure_class.upper())
        except ValueError:
            logger.warning(
                "Unknown exposure class '%s' — defaulting to MODERATE.",
                exposure_class,
            )
            exp_cls = ExposureClass.MODERATE
        self.severity_classifier = SeverityClassifier(exposure_class=exp_cls)

        # ---- BIM mapping ----
        self.bim_mapper = None
        if ifc_path is not None and CrackBIMMapper is not None:
            try:
                self.bim_mapper = CrackBIMMapper(ifc_path=ifc_path)
                logger.info("CrackBIMMapper loaded IFC model: %s", ifc_path)
            except Exception as exc:
                logger.error("Failed to load IFC model: %s", exc)

        # ---- Visualiser ----
        self.visualizer = None
        if BIMVisualizer is not None:
            self.visualizer = BIMVisualizer()

        # ---- Report generator ----
        self.report_generator = None
        if ReportGenerator is not None:
            self.report_generator = ReportGenerator()

        logger.info("BridgeInspector ready (session=%s).", self.session_id)

    # ==================================================================
    # Public API
    # ==================================================================

    def inspect_image(
        self,
        image_path: str,
        altitude_m: Optional[float] = None,
        depth_map: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None,
    ) -> InspectionResult:
        """Run the full inspection pipeline on a single image.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        altitude_m : float, optional
            Drone altitude in metres (used for monocular GSD calibration).
        depth_map : np.ndarray, optional
            Per-pixel depth map for stereo width estimation.
        output_dir : str, optional
            If provided, annotated visualisations and reports are saved here.

        Returns
        -------
        InspectionResult
            Complete inspection record with detections, measurements,
            severity classifications, and BIM mappings.
        """
        image_path = str(image_path)
        logger.info("Inspecting image: %s", image_path)

        # 1. Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at '{image_path}'.")
        img_h, img_w = image.shape[:2]

        # 2. Run crack detection
        raw_detections = self.detector.detect(image)
        logger.info("Raw detections: %d crack(s) found.", len(raw_detections))

        # 3. Build scale info (if altitude provided)
        scale_info = self._build_scale_info(altitude_m)

        # 4. Process each detection through the pipeline
        processed_detections: List[CrackDetection] = []
        severity_levels: List[SeverityLevel] = []
        bim_mappings: List[Dict] = []

        for det in raw_detections:
            try:
                enriched_det = self._process_single_detection(
                    detection=det,
                    image=image,
                    scale_info=scale_info,
                    depth_map=depth_map,
                )
                processed_detections.append(enriched_det)
                severity_levels.append(enriched_det.severity.level)

                # BIM mapping
                if self.bim_mapper is not None:
                    bbox = enriched_det.bbox
                    centroid_x = (bbox[0] + bbox[2]) / 2.0
                    centroid_y = (bbox[1] + bbox[3]) / 2.0
                    mapping = self.bim_mapper.map_crack_to_element(
                        crack_detection={
                            "detection_id": enriched_det.detection_id,
                            "confidence": enriched_det.confidence,
                        },
                        point_3d=np.array([centroid_x, centroid_y, 0.0]),
                    )
                    bim_mappings.append(mapping)
            except Exception as exc:
                logger.error(
                    "Failed to process detection %s: %s",
                    det.detection_id,
                    exc,
                    exc_info=True,
                )
                processed_detections.append(det)

        # 5. Compute aggregate severity
        highest_severity = self._compute_highest_severity(severity_levels)
        flagged = highest_severity in (
            SeverityLevel.SEVERE,
            SeverityLevel.CRITICAL,
        ) if highest_severity else False

        # 6. Build InspectionResult
        result = InspectionResult(
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc),
            source="drone",
            image_path=image_path,
            image_width=img_w,
            image_height=img_h,
            model_name="YOLOv11s-seg",
            model_weights=self.detector.model_path,
            detections=processed_detections,
            total_detections=len(processed_detections),
            highest_severity=highest_severity,
            flagged=flagged,
            metadata={
                "altitude_m": altitude_m,
                "method": self.method,
                "bim_mappings": bim_mappings if bim_mappings else None,
            },
        )

        # 7. Generate visualisations and reports
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            image_name = Path(image_path).stem
            self._save_visualizations(
                image, processed_detections, output_dir, image_name,
            )

            # Generate reports
            if self.report_generator is not None:
                annotated_path = os.path.join(
                    output_dir, f"{image_name}_annotated.jpg"
                )
                if not os.path.exists(annotated_path):
                    annotated_path = None

                try:
                    json_path = os.path.join(
                        output_dir, f"{image_name}_report.json"
                    )
                    self.report_generator.generate_json_report(result, json_path)
                    logger.info("JSON report saved: %s", json_path)
                except Exception as exc:
                    logger.error("JSON report generation failed: %s", exc)

                try:
                    html_path = os.path.join(
                        output_dir, f"{image_name}_report.html"
                    )
                    self.report_generator.generate_html_report(
                        result,
                        annotated_image_path=annotated_path,
                        output_path=html_path,
                    )
                    logger.info("HTML report saved: %s", html_path)
                except Exception as exc:
                    logger.error("HTML report generation failed: %s", exc)

        logger.info(
            "Inspection complete: %d detection(s), highest_severity=%s, flagged=%s",
            len(processed_detections),
            highest_severity.value if highest_severity else "N/A",
            flagged,
        )
        return result

    def inspect_directory(
        self,
        image_dir: str,
        output_dir: str,
        altitude_m: Optional[float] = None,
        extensions: tuple = (".jpg", ".png", ".jpeg"),
    ) -> List[InspectionResult]:
        """Process all images in a directory.

        Parameters
        ----------
        image_dir : str
            Path to the directory containing images.
        output_dir : str
            Directory where results and reports are saved.
        altitude_m : float, optional
            Drone altitude in metres.
        extensions : tuple
            Accepted image file extensions (case-insensitive).

        Returns
        -------
        List[InspectionResult]
            One result per processed image.
        """
        image_dir = str(image_dir)
        output_dir = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        ext_set = {e.lower() for e in extensions}
        image_paths = sorted(
            p
            for p in Path(image_dir).iterdir()
            if p.is_file() and p.suffix.lower() in ext_set
        )

        if not image_paths:
            logger.warning("No images found in '%s' with extensions %s.", image_dir, extensions)
            return []

        logger.info(
            "Inspecting %d image(s) in '%s'.", len(image_paths), image_dir,
        )

        results: List[InspectionResult] = []
        for idx, img_path in enumerate(image_paths, start=1):
            logger.info(
                "[%d/%d] Processing %s …", idx, len(image_paths), img_path.name,
            )
            try:
                img_output_dir = os.path.join(output_dir, img_path.stem)
                result = self.inspect_image(
                    image_path=str(img_path),
                    altitude_m=altitude_m,
                    output_dir=img_output_dir,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Failed to inspect '%s': %s", img_path.name, exc, exc_info=True,
                )

        # Generate aggregate CSV report
        if results and self.report_generator is not None:
            try:
                csv_path = os.path.join(output_dir, "inspection_summary.csv")
                self.report_generator.generate_csv_report(results, csv_path)
                logger.info("Aggregate CSV report saved: %s", csv_path)
            except Exception as exc:
                logger.error("CSV report generation failed: %s", exc)

        logger.info(
            "Directory inspection complete: %d/%d image(s) processed.",
            len(results),
            len(image_paths),
        )
        return results

    def inspect_video(
        self,
        video_path: str,
        output_dir: str,
        altitude_m: Optional[float] = None,
        frame_interval: int = 30,
    ) -> List[InspectionResult]:
        """Process a video file frame by frame.

        Parameters
        ----------
        video_path : str
            Path to the input video.
        output_dir : str
            Directory where per-frame results are saved.
        altitude_m : float, optional
            Drone altitude in metres.
        frame_interval : int
            Process every *n*-th frame (default: 30).

        Returns
        -------
        List[InspectionResult]
            One result per processed frame.
        """
        video_path = str(video_path)
        output_dir = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video at '{video_path}'.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info(
            "Video: %s — %d frames @ %.1f FPS, processing every %d frames.",
            video_path, total_frames, fps, frame_interval,
        )

        results: List[InspectionResult] = []
        frame_idx = 0
        processed_count = 0

        scale_info = self._build_scale_info(altitude_m)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Save frame to disk for traceability
                frame_name = f"frame_{frame_idx:06d}"
                frame_path = os.path.join(output_dir, f"{frame_name}.jpg")
                cv2.imwrite(frame_path, frame)

                try:
                    frame_output_dir = os.path.join(output_dir, frame_name)
                    result = self.inspect_image(
                        image_path=frame_path,
                        altitude_m=altitude_m,
                        output_dir=frame_output_dir,
                    )
                    results.append(result)
                    processed_count += 1
                except Exception as exc:
                    logger.error(
                        "Failed to inspect frame %d: %s",
                        frame_idx, exc, exc_info=True,
                    )

            frame_idx += 1

        cap.release()

        # Aggregate CSV
        if results and self.report_generator is not None:
            try:
                csv_path = os.path.join(output_dir, "video_inspection_summary.csv")
                self.report_generator.generate_csv_report(results, csv_path)
                logger.info("Video CSV report saved: %s", csv_path)
            except Exception as exc:
                logger.error("Video CSV report generation failed: %s", exc)

        logger.info(
            "Video inspection complete: %d/%d frame(s) processed.",
            processed_count,
            total_frames,
        )
        return results

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_scale_info(
        self,
        altitude_m: Optional[float] = None,
    ) -> Optional[ScaleInfo]:
        """Compute :class:`ScaleInfo` from calibrator and altitude.

        Parameters
        ----------
        altitude_m : float, optional
            Drone altitude in metres.

        Returns
        -------
        ScaleInfo or None
            Scale information for pixel-to-mm conversion, or ``None`` if
            calibration data is insufficient.
        """
        if altitude_m is not None and self.calibrator is not None:
            try:
                scale = self.calibrator.calibrate_from_altitude(altitude_m)
                logger.info(
                    "Scale info: GSD=%.4f mm/px at altitude %.1f m.",
                    scale.gsd_mm_per_px,
                    altitude_m,
                )
                return scale
            except ValueError as exc:
                logger.warning("Altitude calibration failed: %s", exc)
        return None

    def _process_single_detection(
        self,
        detection: CrackDetection,
        image: np.ndarray,
        scale_info: Optional[ScaleInfo],
        depth_map: Optional[np.ndarray],
    ) -> CrackDetection:
        """Run mask refinement, width estimation, and severity on one detection.

        Parameters
        ----------
        detection : CrackDetection
            Raw detection from the YOLO model.
        image : np.ndarray
            Original BGR image.
        scale_info : ScaleInfo or None
            Pixel-to-mm calibration.
        depth_map : np.ndarray or None
            Per-pixel depth (for stereo mode).

        Returns
        -------
        CrackDetection
            Enriched detection with geometry, width, and severity filled in.
        """
        x1, y1, x2, y2 = detection.bbox
        img_h, img_w = image.shape[:2]

        # --- (a) Extract / reconstruct binary mask ---
        mask = self._get_detection_mask(detection, img_h, img_w)
        if mask is None:
            logger.warning(
                "No mask for detection %s — skipping refinement.",
                detection.detection_id,
            )
            return detection

        # --- (b) Refine mask ---
        refined_mask = self.mask_processor.refine_mask(mask)

        # --- (c) Compute geometry ---
        from skimage.morphology import skeletonize

        skeleton = skeletonize(refined_mask > 0).astype(np.uint8)

        orientation = self.mask_processor.compute_orientation(refined_mask)
        crack_area = float(np.count_nonzero(refined_mask))
        crack_length = float(np.count_nonzero(skeleton))

        # Extract ordered skeleton points
        skel_ys, skel_xs = np.nonzero(skeleton)
        skeleton_pts = list(zip(skel_xs.astype(float).tolist(),
                                skel_ys.astype(float).tolist()))

        # Compute length in mm if scale available
        length_mm = None
        if scale_info is not None and scale_info.gsd_mm_per_px is not None:
            length_mm = crack_length * scale_info.gsd_mm_per_px

        geometry = CrackGeometry(
            length_px=crack_length,
            length_mm=length_mm,
            orientation_deg=orientation,
            curvature=0.0,
            skeleton_points=skeleton_pts[:500],  # cap to avoid bloat
            area_px=crack_area,
        )

        # --- (d) Estimate width ---
        width_measurement = self._estimate_width(
            refined_mask, scale_info, depth_map,
        )

        # --- (e) Classify severity ---
        severity = self.severity_classifier.classify(width_measurement)

        # --- Assemble enriched detection ---
        enriched = CrackDetection(
            detection_id=detection.detection_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            geometry=geometry,
            width=width_measurement,
            severity=severity,
            mask_polygon=detection.mask_polygon,
        )
        return enriched

    def _get_detection_mask(
        self,
        detection: CrackDetection,
        img_h: int,
        img_w: int,
    ) -> Optional[np.ndarray]:
        """Reconstruct a binary mask from the detection's polygon or bbox.

        Parameters
        ----------
        detection : CrackDetection
            Detection with optional ``mask_polygon``.
        img_h, img_w : int
            Image dimensions.

        Returns
        -------
        np.ndarray or None
            Binary mask of shape ``(img_h, img_w)`` or ``None``.
        """
        if detection.mask_polygon is not None and len(detection.mask_polygon) >= 3:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            pts = np.array(detection.mask_polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
            return mask

        # Fallback: create a simple mask from the bounding box
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    def _estimate_width(
        self,
        mask: np.ndarray,
        scale_info: Optional[ScaleInfo],
        depth_map: Optional[np.ndarray],
    ) -> WidthMeasurement:
        """Run width estimation using the configured method.

        Parameters
        ----------
        mask : np.ndarray
            Refined binary crack mask.
        scale_info : ScaleInfo or None
            Calibration data.
        depth_map : np.ndarray or None
            Per-pixel depth map (stereo only).

        Returns
        -------
        WidthMeasurement
        """
        # Attempt calibrated estimation
        if self.width_estimator is not None:
            try:
                if self.method == "monocular" and scale_info is not None:
                    return self.width_estimator.estimate_width(mask, scale_info)
                elif self.method == "stereo" and depth_map is not None:
                    camera_matrix = (
                        self.calibrator.camera_matrix
                        if self.calibrator is not None
                        else None
                    )
                    return self.width_estimator.estimate_width(
                        mask, depth_map, camera_matrix,
                    )
            except Exception as exc:
                logger.warning(
                    "Width estimation (%s) failed, falling back to pixel-only: %s",
                    self.method, exc,
                )

        # Fallback: pixel-only measurement from mask
        return self._pixel_only_width(mask)

    @staticmethod
    def _pixel_only_width(mask: np.ndarray) -> WidthMeasurement:
        """Compute width in pixels from the mask using distance transform.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask.

        Returns
        -------
        WidthMeasurement
            Pixel-only measurement.
        """
        binary = (mask > 0).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            return WidthMeasurement(
                width_px=0.0,
                method=MeasurementMethod.RELATIVE_ONLY,
            )

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        # Width ≈ 2× max distance-transform value (diameter)
        max_dist = float(dist.max())
        width_px = 2.0 * max_dist

        # Sample multiple points along the skeleton for statistics
        from skimage.morphology import skeletonize

        skeleton = skeletonize(binary > 0).astype(np.uint8)
        skel_ys, skel_xs = np.nonzero(skeleton)

        measurement_points = []
        widths = []
        for sx, sy in zip(skel_xs, skel_ys):
            w = 2.0 * float(dist[sy, sx])
            measurement_points.append((float(sx), float(sy), w))
            widths.append(w)

        widths_arr = np.array(widths) if widths else np.array([width_px])

        return WidthMeasurement(
            width_px=float(np.median(widths_arr)),
            width_mm=None,
            method=MeasurementMethod.RELATIVE_ONLY,
            measurement_points=measurement_points[:200],  # cap for serialisation
            mean_width_px=float(np.mean(widths_arr)),
            median_width_px=float(np.median(widths_arr)),
            max_width_px=float(np.max(widths_arr)),
            min_width_px=float(np.min(widths_arr)),
            std_width_px=float(np.std(widths_arr)),
            percentile_95_width_px=float(np.percentile(widths_arr, 95)),
        )

    @staticmethod
    def _compute_highest_severity(
        levels: List[SeverityLevel],
    ) -> Optional[SeverityLevel]:
        """Return the most severe level from a list.

        Parameters
        ----------
        levels : list[SeverityLevel]

        Returns
        -------
        SeverityLevel or None
        """
        if not levels:
            return None
        order = {
            SeverityLevel.MINOR: 0,
            SeverityLevel.MODERATE: 1,
            SeverityLevel.SEVERE: 2,
            SeverityLevel.CRITICAL: 3,
        }
        return max(levels, key=lambda lv: order.get(lv, 0))

    def _save_visualizations(
        self,
        image: np.ndarray,
        detections: List[CrackDetection],
        output_dir: str,
        image_name: str,
    ) -> None:
        """Save annotated image and individual crack crops.

        Parameters
        ----------
        image : np.ndarray
            Original BGR image.
        detections : list[CrackDetection]
            Processed detections.
        output_dir : str
            Output directory.
        image_name : str
            Base name for saved files (no extension).
        """
        os.makedirs(output_dir, exist_ok=True)

        # --- Annotated overview ---
        annotated = image.copy()

        severity_colours = {
            SeverityLevel.MINOR: (0, 200, 0),       # green
            SeverityLevel.MODERATE: (0, 200, 255),   # yellow/orange
            SeverityLevel.SEVERE: (0, 100, 255),     # orange-red
            SeverityLevel.CRITICAL: (0, 0, 255),     # red
        }

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            colour = severity_colours.get(det.severity.level, (255, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            # Label
            label = (
                f"{det.detection_id[:16]} | "
                f"{det.severity.level.value} | "
                f"W={det.width.width_px:.1f}px"
            )
            if det.width.width_mm is not None:
                label += f" ({det.width.width_mm:.2f}mm)"

            font_scale = 0.45
            thickness = 1
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                colour,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            # Overlay mask polygon
            if det.mask_polygon is not None and len(det.mask_polygon) >= 3:
                pts = np.array(det.mask_polygon, dtype=np.int32).reshape(-1, 1, 2)
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], colour)
                cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        annotated_path = os.path.join(output_dir, f"{image_name}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)
        logger.info("Annotated image saved: %s", annotated_path)

        # --- Individual crack crops ---
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(image.shape[1], x2 + 10)
            y2 = min(image.shape[0], y2 + 10)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crop_path = os.path.join(
                    crops_dir, f"{det.detection_id}.jpg"
                )
                cv2.imwrite(crop_path, crop)

        logger.info(
            "Saved %d crack crop(s) to %s", len(detections), crops_dir,
        )

        # --- Use BIMVisualizer if available ---
        if self.visualizer is not None:
            try:
                vis_path = os.path.join(
                    output_dir, f"{image_name}_bim_overlay.jpg"
                )
                vis_img = self.visualizer.visualize_crack_on_image(
                    image, detections, [d.width for d in detections],
                )
                if vis_img is not None:
                    cv2.imwrite(vis_path, vis_img)
                    logger.info("BIM overlay saved: %s", vis_path)
            except Exception as exc:
                logger.warning("BIM visualisation failed: %s", exc)
