#!/usr/bin/env python3
"""
SID Crack Detection Demo
=========================
Run the full inspection pipeline on synthetic or real images.

Usage:
    python run_demo.py --mode demo              # Synthetic data benchmark
    python run_demo.py --mode image --image x.jpg --altitude 5
    python run_demo.py --mode image --image x.jpg --bim bridge.ifc

Author: IDEATOR GECT — SID Structural Inspection Drone
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# ANSI color helpers
# ──────────────────────────────────────────────────────────────────────

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"

SEVERITY_COLORS = {
    "MINOR": C.GREEN,
    "MODERATE": C.YELLOW,
    "SEVERE": C.RED,
    "CRITICAL": C.BG_RED + C.WHITE,
}


def print_header():
    """Print the SID banner."""
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🔍  SID — Structural Inspection Drone                      ║
║       Crack Detection & Width Estimation Demo                ║
║                                                              ║
║   IDEATOR GECT · Government Engineering College Thrissur     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


def print_section(title):
    """Print a section header."""
    print(f"\n{C.BOLD}{C.BLUE}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{C.RESET}\n")


def print_result_row(label, value, color=C.WHITE):
    """Print a formatted key-value row."""
    print(f"  {C.DIM}{label:.<30}{C.RESET} {color}{value}{C.RESET}")


# ──────────────────────────────────────────────────────────────────────
# Demo Mode — Synthetic Data Benchmark
# ──────────────────────────────────────────────────────────────────────

def run_demo_mode(args):
    """Run width estimation on synthetic cracks with known ground truth."""
    from tests.fixtures.generate_test_data import (
        generate_crack_with_known_width,
        generate_branching_crack,
        generate_curved_crack,
    )
    from crack_detection.width_estimation.monocular import MonocularWidthEstimator
    from crack_detection.width_estimation.common import ScaleInfo
    from crack_detection.schemas import MeasurementMethod
    from crack_detection.severity import SeverityClassifier
    from crack_detection.schemas import ExposureClass

    print_section("DEMO MODE — Synthetic Crack Width Benchmark")

    # Setup
    estimator = MonocularWidthEstimator()
    classifier = SeverityClassifier(exposure_class=ExposureClass.MODERATE)

    # GSD for testing: 0.5 mm/px (simulating 2m altitude)
    scale = ScaleInfo(
        gsd_mm_per_px=0.5,
        method=MeasurementMethod.MONOCULAR_GSD,
        distance_mm=2000.0,
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Benchmark across known widths
    print(f"  {C.CYAN}Generating synthetic cracks with known widths...{C.RESET}")
    test_widths = [2, 3, 5, 8, 10, 15, 20]
    test_data = generate_crack_with_known_width(widths=test_widths)

    print(f"\n  {'True (px)':>10}  {'Measured (px)':>14}  {'Error (px)':>11}  {'Width (mm)':>11}  {'Severity':>10}")
    print(f"  {'─' * 10}  {'─' * 14}  {'─' * 11}  {'─' * 11}  {'─' * 10}")

    errors = []
    for img, mask, meta in test_data:
        true_w = meta["true_width_px"]

        t0 = time.time()
        result = estimator.estimate_width(mask, scale_info=scale)
        elapsed = (time.time() - t0) * 1000

        if result and result.median_width_px > 0:
            measured = result.median_width_px
            error = abs(measured - true_w)
            width_mm = result.width_mm if result.width_mm else measured * 0.5
            errors.append(error)

            # Classify severity
            sev = classifier.classify(result)
            sev_level = sev.level.value if hasattr(sev.level, 'value') else str(sev.level)
            sev_color = SEVERITY_COLORS.get(sev_level, C.WHITE)

            error_color = C.GREEN if error < 2.0 else (C.YELLOW if error < 3.0 else C.RED)

            print(
                f"  {true_w:>10d}  "
                f"{measured:>14.2f}  "
                f"{error_color}{error:>11.2f}{C.RESET}  "
                f"{width_mm:>10.2f}mm  "
                f"{sev_color}{sev_level:>10}{C.RESET}"
            )

            # Save annotated mask
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                vis, f"W={measured:.1f}px ({width_mm:.2f}mm)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imwrite(str(output_dir / f"crack_w{true_w}px.png"), vis)
        else:
            print(f"  {true_w:>10d}  {'FAILED':>14}  {'-':>11}  {'-':>11}  {'-':>10}")

    # Summary statistics
    if errors:
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        print(f"\n  {C.BOLD}Average error:{C.RESET} {avg_error:.2f} px")
        print(f"  {C.BOLD}Max error:{C.RESET}     {max_error:.2f} px")

        if avg_error < 2.0:
            print(f"  {C.BG_GREEN}{C.WHITE}{C.BOLD} ✅ EXCELLENT — Average error < 2 px {C.RESET}")
        elif avg_error < 3.0:
            print(f"  {C.YELLOW}{C.BOLD} ⚠️  GOOD — Average error < 3 px {C.RESET}")
        else:
            print(f"  {C.RED}{C.BOLD} ❌ NEEDS IMPROVEMENT — Average error ≥ 3 px {C.RESET}")

    # 2. Branching crack test
    print_section("Branching Crack Test")
    img, mask, meta = generate_branching_crack(main_width=8, branch_width=4)
    result = estimator.estimate_width(mask, scale_info=scale)
    if result:
        print_result_row("Main trunk width (true)", "8 px")
        print_result_row("Measured median", f"{result.median_width_px:.2f} px")
        print_result_row("Measured mean", f"{result.mean_width_px:.2f} px")
        print_result_row("Max width", f"{result.max_width_px:.2f} px")
        print_result_row("Measurement points", str(len(result.measurement_points)))

    # 3. Curved crack test
    print_section("Curved Crack Test")
    img, mask, meta = generate_curved_crack(width_px=6, curvature_radius=300)
    result = estimator.estimate_width(mask, scale_info=scale)
    if result:
        print_result_row("True width", "6 px")
        print_result_row("Measured median", f"{result.median_width_px:.2f} px")
        if result.std_width_px is not None:
            cv = result.std_width_px / result.mean_width_px if result.mean_width_px > 0 else 0
            print_result_row("Width consistency (CV)", f"{cv:.3f}")
            if cv < 0.3:
                print(f"  {C.GREEN}✅ Width is consistent along the curve{C.RESET}")

    # Generate HTML report
    print_section("Report Generation")
    try:
        from pipeline.report_generator import ReportGenerator
        from crack_detection.schemas import (
            InspectionResult, CrackDetection, CrackGeometry,
            WidthMeasurement, SeverityClassification, SeverityLevel,
        )

        gen = ReportGenerator()

        # Build a mock result for the report
        detections = []
        for i, (img, mask, meta) in enumerate(test_data[:3]):
            true_w = meta["true_width_px"]
            r = estimator.estimate_width(mask, scale_info=scale)
            if r:
                sev = classifier.classify(r)
                det = CrackDetection(
                    detection_id=f"demo_{i:03d}",
                    bbox=(50, 50, 250, 100),
                    confidence=0.92,
                    geometry=CrackGeometry(
                        length_px=200.0,
                        orientation_deg=meta.get("true_orientation_deg", 0),
                        curvature=0.0,
                        skeleton_points=[],
                        area_px=float(np.sum(mask > 0)),
                    ),
                    width=r,
                    severity=sev,
                )
                detections.append(det)

        result_obj = InspectionResult(
            session_id="demo_benchmark",
            timestamp="2025-01-01T00:00:00",
            source="synthetic_generator",
            image_path="synthetic",
            image_width=640,
            image_height=640,
            model_name="yolov11s-seg",
            model_weights="best.pt",
            detections=detections,
            total_detections=len(detections),
            highest_severity=max(
                (d.severity.level for d in detections),
                default=SeverityLevel.MINOR,
            ),
            flagged=len(detections) > 0,
            metadata={"mode": "demo"},
        )

        html_path = gen.generate_html_report(
            result_obj, output_path=str(output_dir / "demo_report.html")
        )
        json_path = gen.generate_json_report(
            result_obj, str(output_dir / "demo_report.json")
        )
        print_result_row("HTML report", html_path, C.GREEN)
        print_result_row("JSON report", json_path, C.GREEN)
    except Exception as e:
        print(f"  {C.YELLOW}Report generation skipped: {e}{C.RESET}")

    print(f"\n  {C.BOLD}Output saved to:{C.RESET} {output_dir.resolve()}")
    print(f"\n{C.GREEN}{C.BOLD}  ✅ Demo complete!{C.RESET}\n")


# ──────────────────────────────────────────────────────────────────────
# Image Mode — Process Real Images
# ──────────────────────────────────────────────────────────────────────

def run_image_mode(args):
    """Process a real image through the full pipeline."""
    print_section(f"IMAGE MODE — Processing {args.image}")

    if not os.path.exists(args.image):
        print(f"  {C.RED}Error: Image not found: {args.image}{C.RESET}")
        sys.exit(1)

    try:
        from pipeline.inspector import BridgeInspector

        camera_config = None
        if args.altitude:
            camera_config = {
                "focal_length_mm": 12.29,
                "sensor_width_mm": 17.3,
                "image_width_px": 5280,
            }

        inspector = BridgeInspector(
            model_path=args.model,
            confidence=args.confidence,
            method=args.method,
            exposure_class=args.exposure.upper(),
            camera_config=camera_config,
            ifc_path=args.bim,
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = inspector.inspect_image(
            image_path=args.image,
            altitude_m=args.altitude,
            output_dir=str(output_dir),
        )

        # Print results
        print_result_row("Session ID", result.session_id)
        print_result_row("Total cracks detected", str(result.total_detections))
        highest = result.highest_severity
        hs = highest.value if hasattr(highest, "value") else str(highest)
        print_result_row("Highest severity", hs,
                        SEVERITY_COLORS.get(hs, C.WHITE))
        print_result_row("Flagged", str(result.flagged))

        for det in result.detections:
            sev = det.severity.level
            sv = sev.value if hasattr(sev, "value") else str(sev)
            print(f"\n  {C.BOLD}Detection: {det.detection_id}{C.RESET}")
            print_result_row("  Confidence", f"{det.confidence:.2%}")
            print_result_row("  Width (px)", f"{det.width.median_width_px:.2f}")
            if det.width.width_mm:
                print_result_row("  Width (mm)", f"{det.width.width_mm:.3f}")
            print_result_row("  Severity",
                           f"{SEVERITY_COLORS.get(sv, '')}{sv}{C.RESET}")

        # Generate reports
        from pipeline.report_generator import ReportGenerator
        gen = ReportGenerator()
        gen.generate_html_report(result, output_path=str(output_dir / "report.html"))
        gen.generate_json_report(result, str(output_dir / "report.json"))

        print(f"\n  {C.BOLD}Output saved to:{C.RESET} {output_dir.resolve()}")
        print(f"\n{C.GREEN}{C.BOLD}  ✅ Inspection complete!{C.RESET}\n")

    except Exception as e:
        print(f"  {C.RED}Error: {e}{C.RESET}")
        logging.exception("Pipeline error")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Directory Mode — Batch Processing
# ──────────────────────────────────────────────────────────────────────

def run_directory_mode(args):
    """Process all images in a directory."""
    print_section(f"DIRECTORY MODE — Processing {args.directory}")

    if not os.path.isdir(args.directory):
        print(f"  {C.RED}Error: Directory not found: {args.directory}{C.RESET}")
        sys.exit(1)

    try:
        from pipeline.inspector import BridgeInspector

        inspector = BridgeInspector(
            model_path=args.model,
            confidence=args.confidence,
            method=args.method,
            exposure_class=args.exposure.upper(),
        )

        output_dir = Path(args.output)
        results = inspector.inspect_directory(
            image_dir=args.directory,
            output_dir=str(output_dir),
            altitude_m=args.altitude,
        )

        print(f"  Processed {C.BOLD}{len(results)}{C.RESET} images")
        total_cracks = sum(r.total_detections for r in results)
        print(f"  Total cracks found: {C.BOLD}{total_cracks}{C.RESET}")

    except Exception as e:
        print(f"  {C.RED}Error: {e}{C.RESET}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SID Crack Detection Demo — Bridge Inspection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py --mode demo
  python run_demo.py --mode image --image bridge_photo.jpg --altitude 5
  python run_demo.py --mode image --image photo.jpg --bim bridge.ifc
  python run_demo.py --mode directory --directory ./images --altitude 3
        """,
    )

    parser.add_argument(
        "--mode", choices=["demo", "image", "directory", "video"],
        default="demo",
        help="Operation mode (default: demo)",
    )
    parser.add_argument("--image", help="Path to image file (for image mode)")
    parser.add_argument("--directory", help="Path to image directory (for directory mode)")
    parser.add_argument("--video", help="Path to video file (for video mode)")
    parser.add_argument(
        "--method", choices=["monocular", "stereo"], default="monocular",
        help="Width estimation method (default: monocular)",
    )
    parser.add_argument("--model", default="best.pt", help="Path to YOLO model weights")
    parser.add_argument("--confidence", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--altitude", type=float, help="Drone altitude in meters (for GSD)")
    parser.add_argument("--bim", help="Path to IFC file for BIM mapping")
    parser.add_argument("--output", default="./inspection_output", help="Output directory")
    parser.add_argument(
        "--exposure", choices=["MILD", "MODERATE", "SEVERE"], default="MODERATE",
        help="IS 456:2000 exposure class (default: MODERATE)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print_header()

    if args.mode == "demo":
        run_demo_mode(args)
    elif args.mode == "image":
        if not args.image:
            parser.error("--image is required for image mode")
        run_image_mode(args)
    elif args.mode == "directory":
        if not args.directory:
            parser.error("--directory is required for directory mode")
        run_directory_mode(args)
    elif args.mode == "video":
        print(f"  {C.YELLOW}Video mode: use --mode image with individual frames{C.RESET}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
