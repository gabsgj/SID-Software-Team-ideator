#!/usr/bin/env python3
"""
SID Webcam Real-Time Crack Inspector
====================================
Real-time crack detection, width estimation, and severity classification
using a webcam. Supports real-time camera calibration using ArUco markers.

Usage:
    ./venv/bin/python run_webcam.py --model best.pt --confidence 0.40

Keyboard Shortcuts:
    'q' : Quit the application
    'c' : Reset calibration to pixels-only
    's' : Save current frame annotation as a screenshot
    '+' : Increase confidence threshold by 0.05
    '-' : Decrease confidence threshold by 0.05

Author: IDEATOR GECT — SID Structural Inspection Drone
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from crack_detection.detector import CrackDetector
    from crack_detection.segmentation import MaskProcessor
    from crack_detection.severity import SeverityClassifier
    from crack_detection.width_estimation.calibration import CameraCalibrator
    from crack_detection.width_estimation.common import ScaleInfo
    from crack_detection.schemas import MeasurementMethod, SeverityLevel, ExposureClass
    from crack_detection.width_estimation.monocular import MonocularWidthEstimator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you run this script using the virtual environment:")
    print("  ./venv/bin/python run_webcam.py")
    sys.exit(1)


class WebcamInspector:
    """Real-time webcam inspector class."""

    def __init__(
        self,
        model_path: str = "best.pt",
        confidence: float = 0.40,
        exposure_class: str = "MODERATE",
        aruco_width_mm: float = 50.0,
    ):
        self.confidence = confidence
        self.aruco_width_mm = aruco_width_mm
        
        # Load model and components
        print(f"[*] Initializing CrackDetector with model: {model_path}...")
        self.detector = CrackDetector(model_path=model_path, confidence=confidence)
        self.mask_processor = MaskProcessor()
        self.width_estimator = MonocularWidthEstimator()
        self.severity_classifier = SeverityClassifier(
            exposure_class=ExposureClass(exposure_class.upper())
        )
        self.calibrator = CameraCalibrator()
        self.scale_info: Optional[ScaleInfo] = None
        
        # Performance tracking
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        
        # Visualization colors
        self.color_map = {
            "MINOR": (0, 255, 0),        # Green
            "MODERATE": (0, 255, 255),    # Yellow
            "SEVERE": (0, 0, 255),        # Red
            "CRITICAL": (255, 0, 255),    # Magenta
        }

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return the annotated frame."""
        h, w = frame.shape[:2]
        
        # 1. Check for ArUco marker for calibration
        self._check_calibration(frame)
        
        # 2. Run detector
        detections = []
        try:
            # Run detection using override confidence
            detections = self.detector.detect(frame, conf=self.confidence)
        except Exception as e:
            logging.error(f"Detection failed: {e}")
            
        annotated_frame = frame.copy()
        
        # Draw ArUco calibration marker indicators if calibrating
        if self.scale_info is not None:
            # We are calibrated
            calib_text = f"Scale: {self.scale_info.gsd_mm_per_px:.3f} mm/px (ArUco)"
            cv2.putText(annotated_frame, calib_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "Scale: Uncalibrated (pixels only)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 3. Process and draw each detection
        highest_sev = "MINOR"
        sev_rank = {"MINOR": 0, "MODERATE": 1, "SEVERE": 2, "CRITICAL": 3}
        
        for det in detections:
            # Extract and refine mask
            mask = self.detector._extract_mask(self.detector.model.predict(frame, conf=self.confidence, verbose=False)[0], 0, (h, w)) # Fallback or direct extraction
            # To be safe and clean, let's extract the mask from the detection object itself
            # In schemas, mask_polygon is stored
            mask = np.zeros((h, w), dtype=np.uint8)
            if det.mask_polygon and len(det.mask_polygon) >= 3:
                pts = np.array(det.mask_polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts], 1)
            else:
                x1, y1, x2, y2 = det.bbox
                mask[y1:y2, x1:x2] = 1

            if np.count_nonzero(mask) < 20:
                continue

            refined_mask = self.mask_processor.refine_mask(mask)
            
            # Estimate width
            try:
                width_measure = self.width_estimator.estimate_width(refined_mask, scale_info=self.scale_info)
            except Exception as e:
                # Fallback to distance transform
                width_measure = self.detector._pixel_only_width(refined_mask)
                
            # Classify severity
            severity = self.severity_classifier.classify(width_measure)
            sev_level = severity.level.value if hasattr(severity.level, 'value') else str(severity.level)
            
            if sev_rank.get(sev_level, 0) > sev_rank.get(highest_sev, 0):
                highest_sev = sev_level
                
            color = self.color_map.get(sev_level, (255, 255, 255))
            
            # Draw semi-transparent mask overlay
            overlay = annotated_frame.copy()
            overlay[refined_mask > 0] = color
            cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
            
            # Draw skeleton
            from skimage.morphology import skeletonize
            skeleton = skeletonize(refined_mask > 0).astype(np.uint8)
            annotated_frame[skeleton > 0] = (255, 0, 0) # Draw skeleton in Blue
            
            # Draw bbox
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Label width and confidence
            label = f"Crack: "
            if width_measure.width_mm is not None:
                label += f"{width_measure.width_mm:.2f}mm"
            else:
                label += f"{width_measure.median_width_px:.1f}px"
            label += f" ({det.confidence:.0%})"
            
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            # Draw perpendicular width measurement indicators at sample points
            if hasattr(width_measure, 'measurement_points') and width_measure.measurement_points:
                # Sample a few points to draw perpendicular lines
                step = max(1, len(width_measure.measurement_points) // 10)
                for pt in width_measure.measurement_points[::step]:
                    px, py, pw = pt
                    # Let's draw a small dot at skeleton points
                    cv2.circle(annotated_frame, (int(px), int(py)), 2, (0, 0, 255), -1)

        # Draw HUD overlay
        self._draw_hud(annotated_frame, len(detections), highest_sev)
        
        return annotated_frame

    def _check_calibration(self, frame: np.ndarray):
        """Detect ArUco markers to dynamically calibrate the scaling factor."""
        try:
            markers = self.calibrator.detect_aruco_markers(frame)
            if markers:
                corners, marker_id = markers[0]
                # Draw marker borders
                cv2.polylines(
                    frame,
                    [corners.astype(np.int32)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
                # Compute scaling info
                self.scale_info = self.calibrator.calibrate_from_reference(
                    frame, known_width_mm=self.aruco_width_mm
                )
                # Draw ID text
                cv2.putText(
                    frame,
                    f"Calib Marker ID: {marker_id}",
                    (int(corners[0][0]), int(corners[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
        except Exception:
            pass

    def _draw_hud(self, frame: np.ndarray, num_detections: int, highest_sev: str):
        """Draw a professional head-up display showing stats."""
        # Calculate FPS
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = now

        # Draw HUD semi-transparent background
        h, w = frame.shape[:2]
        hud_w, hud_h = 320, 160
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (10, 10), (hud_w, hud_h), (30, 30, 30), -1)
        cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0, frame)

        # Draw title
        cv2.putText(frame, "SID REALTIME INSPECTOR", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.line(frame, (20, 45), (hud_w - 10, 45), (100, 100, 100), 1)

        # Draw details
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Confidence: {self.confidence:.2f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Severity text and color
        sev_color = self.color_map.get(highest_sev, (255, 255, 255))
        # BGR to RGB conversion for displaying in cv2 (it expects BGR, so keep as is)
        cv2.putText(frame, f"Cracks Found: {num_detections}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if num_detections > 0:
            cv2.putText(frame, f"Max Severity: {highest_sev}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sev_color, 2)
        else:
            cv2.putText(frame, "Max Severity: NONE", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def main():
    parser = argparse.ArgumentParser(
        description="SID Real-time Webcam Crack Detector and Width Estimator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="best.pt", help="Path to YOLOv11s-seg model weights")
    parser.add_argument("--confidence", type=float, default=0.40, help="Initial confidence threshold (default: 0.40)")
    parser.add_argument("--exposure", default="MODERATE", choices=["MILD", "MODERATE", "SEVERE"], help="IS 456 exposure class")
    parser.add_argument("--aruco-width", type=float, default=50.0, help="True width of ArUco marker in mm (default: 50.0)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    # Load inspector
    try:
        inspector = WebcamInspector(
            model_path=args.model,
            confidence=args.confidence,
            exposure_class=args.exposure,
            aruco_width_mm=args.aruco_width,
        )
    except Exception as e:
        print(f"Error loading inspector: {e}")
        sys.exit(1)

    print("[*] Opening camera stream...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {args.camera}.")
        print("Suggestions:")
        print("  1. Check if the webcam is connected.")
        print("  2. If using a Mac, make sure the terminal application has Camera Permissions enabled in System Settings.")
        print("  3. Try another camera index, e.g., --camera 1")
        sys.exit(1)

    # Set camera resolution to 720p or 1080p if supported for better width accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 50)
    print("  SID Structural Inspection Drone — Realtime Webcam Test")
    print("=" * 50)
    print("  Instructions:")
    print("   1. Point the camera at a surface with a crack.")
    print("   2. For calibration (converting pixels to mm):")
    print("      - Print a standard ArUco marker (DICT_4X4_50, e.g. ID 0).")
    print("      - Place it in the camera frame next to the crack.")
    print("      - The system will detect it, draw a green boundary, and calibrate.")
    print("   3. Controls:")
    print("      - Press [q] to Quit")
    print("      - Press [c] to Clear scale calibration (resets to pixels)")
    print("      - Press [s] to Save a screenshot of the output")
    print("      - Press [+] to Increase detection confidence threshold")
    print("      - Press [-] to Decrease detection confidence threshold")
    print("=" * 50 + "\n")

    window_name = "SID Structural Inspection Drone — Real-time Crack Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera. Exiting.")
            break

        # Process frame
        annotated = inspector.process_frame(frame)

        # Show frame
        cv2.imshow(window_name, annotated)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[*] Exiting...")
            break
        elif key == ord('c'):
            inspector.scale_info = None
            print("[*] Scale calibration cleared.")
        elif key == ord('s'):
            screenshot_dir = Path("inspection_output/screenshots")
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = screenshot_dir / f"screenshot_{timestamp}.jpg"
            cv2.imwrite(str(filename), annotated)
            print(f"[*] Screenshot saved to: {filename}")
        elif key == ord('+') or key == ord('='):
            inspector.confidence = min(0.95, inspector.confidence + 0.05)
            print(f"[*] Confidence threshold increased to {inspector.confidence:.2f}")
        elif key == ord('-') or key == ord('_'):
            inspector.confidence = max(0.10, inspector.confidence - 0.05)
            print(f"[*] Confidence threshold decreased to {inspector.confidence:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
