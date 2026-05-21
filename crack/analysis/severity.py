"""
severity.py — Crack Severity Classifier & Output Generator
crack/analysis/severity.py

Consumes geometry.py output (extract_geometry dict).
Produces JSON records that validate against output_schema.json v2.0.

geometry.py output shape:
    {
        "length_px":      float,   # bbox diagonal
        "width_px":       float,   # min(bbox_w, bbox_h)
        "relative_width": float,   # width_px / image_width_px  (0.0 – 1.0)
    }

Output schema (one record per frame):
    {
      "session_id", "timestamp", "source", "frame_id",
      "image_width_px", "image_height_px",
      "location":   { label, latitude, longitude },
      "model":      { name, weights },
      "detections": [ { detection_id, bbox_px, confidence,
                        geometry_px, geometry_mm, severity } ],
      "summary":    { total_detections, highest_severity, flagged }
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Severity thresholds  (relative_width based)
# ---------------------------------------------------------------------------
THRESHOLD_MINOR_MAX = 0.01  # relative_width  < 0.01          → Minor
THRESHOLD_MODERATE_MAX = 0.03  # 0.01 ≤ relative_width < 0.03   → Moderate
# relative_width ≥ 0.03           → Severe

# Severity priority for summary ranking
_SEVERITY_RANK = {"Minor": 0, "Moderate": 1, "Severe": 2}

# Flag frame if any detection reaches this level or above
FLAG_THRESHOLD = "Severe"


# ---------------------------------------------------------------------------
# Threshold recalibration
# ---------------------------------------------------------------------------


def recalibrate_thresholds(
    new_minor_max: float,
    new_moderate_max: float,
) -> None:
    """
    Adjust global thresholds in-place.
    Call once after visually checking 20 detections if labels look wrong.

    Parameters
    ----------
    new_minor_max    : float  e.g. 0.008
    new_moderate_max : float  e.g. 0.025
    """
    global THRESHOLD_MINOR_MAX, THRESHOLD_MODERATE_MAX

    if not (0 < new_minor_max < new_moderate_max <= 1.0):
        raise ValueError("Thresholds must satisfy 0 < minor_max < moderate_max ≤ 1")
    THRESHOLD_MINOR_MAX = new_minor_max
    THRESHOLD_MODERATE_MAX = new_moderate_max
    print(
        f"[severity] Thresholds recalibrated → "
        f"Minor < {THRESHOLD_MINOR_MAX}, "
        f"Moderate [{THRESHOLD_MINOR_MAX}, {THRESHOLD_MODERATE_MAX}), "
        f"Severe ≥ {THRESHOLD_MODERATE_MAX}"
    )


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------


def classify_severity(geometry_px: dict[str, Any]) -> dict[str, str]:
    """
    Classify crack severity from geometry_px dict.

    Parameters
    ----------
    geometry_px : dict
        Must contain ``relative_width`` (float, 0–1).

    Returns
    -------
    dict:
        { "level": "Minor"|"Moderate"|"Severe", "basis": "relative_width" }
    """
    rw = geometry_px["relative_width"]

    if not (0.0 <= rw <= 1.0):
        raise ValueError(
            f"relative_width must be in [0, 1], got {rw}. Check geometry.py output."
        )

    if rw < THRESHOLD_MINOR_MAX:
        level = "Minor"
    elif rw < THRESHOLD_MODERATE_MAX:
        level = "Moderate"
    else:
        level = "Severe"

    return {"level": level, "basis": "relative_width"}


# ---------------------------------------------------------------------------
# Output generator — one record per frame
# ---------------------------------------------------------------------------


def generate_output(
    detections: list[dict[str, Any]],
    geometries: list[dict[str, Any]],
    session_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Assemble one JSON frame record matching output_schema.json v2.0.

    Parameters
    ----------
    detections : list of dicts, one per bbox::

        {
            "detection_id": "det_142_001",   # unique within frame
            "bbox_px":      [x1, y1, x2, y2],
            "confidence":   0.94,
        }

    geometries : list of dicts from extract_geometry(), same order::

        {
            "length_px":      158.0,
            "width_px":       40.0,
            "relative_width": 0.031,
        }

    session_meta : dict with frame / session context::

        {
            "session_id":     "sid_20250316_143022",
            "timestamp":      "2025-03-16T14:30:22Z",   # ISO-8601 UTC
            "source":         "usb_camera",
            "frame_id":       142,
            "image_width_px": 1280,
            "image_height_px":720,
            "location": {
                "label":     "bridge_deck_faceA",
                "latitude":  null / float,
                "longitude": null / float,
            },
            "model": {
                "name":    "yolov11s-seg",
                "weights": "best.pt",
            },
        }

    Returns
    -------
    dict matching output_schema.json
    """
    if len(detections) != len(geometries):
        raise ValueError(
            f"detections ({len(detections)}) and geometries ({len(geometries)}) "
            "must be the same length."
        )

    det_records = []
    severity_levels: list[str] = []

    for det, geo in zip(detections, geometries):
        sev = classify_severity(geo)
        severity_levels.append(sev["level"])

        det_records.append(
            {
                "detection_id": det["detection_id"],
                "bbox_px": det["bbox_px"],
                "confidence": round(float(det["confidence"]), 4),
                "geometry_px": {
                    "length_px": round(float(geo["length_px"]), 2),
                    "width_px": round(float(geo["width_px"]), 2),
                    "relative_width": round(float(geo["relative_width"]), 6),
                },
                "geometry_mm": {
                    "width_mm": geo.get("width_mm", None),  # null unless stereo
                    "stereo_available": bool(geo.get("stereo_available", False)),
                },
                "severity": sev,
            }
        )

    # Frame-level summary
    if severity_levels:
        highest = max(severity_levels, key=lambda s: _SEVERITY_RANK[s])
    else:
        highest = None

    flagged = (
        highest is not None
        and _SEVERITY_RANK[highest] >= _SEVERITY_RANK[FLAG_THRESHOLD]
    )

    location = session_meta.get("location", {})
    model = session_meta.get("model", {})

    record = {
        "session_id": session_meta["session_id"],
        "timestamp": session_meta.get(
            "timestamp", datetime.now(timezone.utc).isoformat()
        ),
        "source": session_meta.get("source", "unknown"),
        "frame_id": int(session_meta["frame_id"]),
        "image_width_px": int(session_meta["image_width_px"]),
        "image_height_px": int(session_meta["image_height_px"]),
        "location": {
            "label": location.get("label", ""),
            "latitude": location.get("latitude", None),
            "longitude": location.get("longitude", None),
        },
        "model": {
            "name": model.get("name", "unknown"),
            "weights": model.get("weights", "best.pt"),
        },
        "detections": det_records,
        "summary": {
            "total_detections": len(det_records),
            "highest_severity": highest,
            "flagged": flagged,
        },
    }

    return record


# ---------------------------------------------------------------------------
# Batch runner — 50-image test → results_batch_test.json
# ---------------------------------------------------------------------------


def run_batch(
    frames: list[dict[str, Any]],
    output_path: str = "results_batch_test.json",
) -> list[dict[str, Any]]:
    """
    Run generate_output() over a batch of frames and export results.

    Parameters
    ----------
    frames : list of dicts, each with keys:
        - "session_meta" : dict   (session / frame context)
        - "detections"   : list   (bbox dicts)
        - "geometries"   : list   (geometry dicts, same order as detections)

    output_path : str
        Destination JSON file path.

    Returns
    -------
    list of per-frame output records
    """
    results = []
    severity_tally = {"Minor": 0, "Moderate": 0, "Severe": 0}

    for frame in frames:
        record = generate_output(
            frame["detections"],
            frame["geometries"],
            frame["session_meta"],
        )
        results.append(record)
        for det in record["detections"]:
            severity_tally[det["severity"]["level"]] += 1

    batch_output = {
        "batch_summary": {
            "total_frames": len(results),
            "total_detections": sum(severity_tally.values()),
            "severity_counts": severity_tally,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "thresholds": {
                "minor_max": THRESHOLD_MINOR_MAX,
                "moderate_max": THRESHOLD_MODERATE_MAX,
            },
        },
        "frames": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_output, f, indent=2)

    print(
        f"[severity] Batch complete → {len(results)} frames, "
        f"{sum(severity_tally.values())} detections written to '{output_path}'\n"
        f"           Minor: {severity_tally['Minor']}  |  "
        f"Moderate: {severity_tally['Moderate']}  |  "
        f"Severe: {severity_tally['Severe']}"
    )
    return results


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mock_frames = [
        {
            "session_meta": {
                "session_id": "sid_20250316_143022",
                "timestamp": "2025-03-16T14:30:22Z",
                "source": "usb_camera",
                "frame_id": 142,
                "image_width_px": 1280,
                "image_height_px": 720,
                "location": {
                    "label": "bridge_deck_faceA",
                    "latitude": None,
                    "longitude": None,
                },
                "model": {
                    "name": "yolov11s-seg",
                    "weights": "best.pt",
                },
            },
            "detections": [
                {
                    "detection_id": "det_142_001",
                    "bbox_px": [100, 120, 250, 160],
                    "confidence": 0.94,
                },
            ],
            "geometries": [
                {
                    "length_px": 158.0,
                    "width_px": 40.0,
                    "relative_width": 0.031,  # → Moderate
                    "width_mm": None,
                    "stereo_available": False,
                },
            ],
        },
        {
            "session_meta": {
                "session_id": "sid_20250316_143022",
                "timestamp": "2025-03-16T14:30:23Z",
                "source": "usb_camera",
                "frame_id": 143,
                "image_width_px": 1280,
                "image_height_px": 720,
                "location": {
                    "label": "bridge_deck_faceA",
                    "latitude": None,
                    "longitude": None,
                },
                "model": {
                    "name": "yolov11s-seg",
                    "weights": "best.pt",
                },
            },
            "detections": [
                {
                    "detection_id": "det_143_001",
                    "bbox_px": [200, 300, 220, 400],
                    "confidence": 0.87,
                },
                {
                    "detection_id": "det_143_002",
                    "bbox_px": [600, 100, 680, 150],
                    "confidence": 0.73,
                },
            ],
            "geometries": [
                {
                    "length_px": 102.0,
                    "width_px": 20.0,
                    "relative_width": 0.0156,  # → Moderate
                    "width_mm": None,
                    "stereo_available": False,
                },
                {
                    "length_px": 94.9,
                    "width_px": 5.0,
                    "relative_width": 0.0039,  # → Minor
                    "width_mm": None,
                    "stereo_available": False,
                },
            ],
        },
    ]

    records = run_batch(mock_frames, output_path="results_batch_test.json")

    print("\nSample frame output:")
    print(json.dumps(records[0], indent=2))
