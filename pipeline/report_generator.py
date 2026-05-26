"""
Report Generator
================
Generates structured inspection reports in JSON, HTML, and CSV formats.
Produces professional, print-ready reports with severity color coding,
crack measurements, and IS 456:2000 compliance status.

Author: IDEATOR GECT — SID Structural Inspection Drone
"""

import json
import csv
import logging
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates inspection reports in multiple formats."""

    # Severity color mapping (CSS colors)
    SEVERITY_COLORS = {
        "MINOR": "#22c55e",       # green
        "MODERATE": "#eab308",    # yellow
        "SEVERE": "#f97316",      # orange
        "CRITICAL": "#ef4444",    # red
    }

    SEVERITY_BG_COLORS = {
        "MINOR": "#f0fdf4",
        "MODERATE": "#fefce8",
        "SEVERE": "#fff7ed",
        "CRITICAL": "#fef2f2",
    }

    def __init__(self):
        """Initialize the report generator."""
        logger.info("ReportGenerator initialized")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    def generate_json_report(self, result, output_path: str) -> str:
        """Serialize an InspectionResult to a JSON file.

        Args:
            result: InspectionResult pydantic model.
            output_path: Destination file path.

        Returns:
            The absolute path of the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = self._result_to_dict(result)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        logger.info("JSON report saved to %s", output_path)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    def generate_csv_report(self, results, output_path: str) -> str:
        """Flatten one or more InspectionResults into a CSV file.

        Args:
            results: Single InspectionResult or list of them.
            output_path: Destination CSV path.

        Returns:
            The absolute path of the written file.
        """
        if not isinstance(results, list):
            results = [results]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        fieldnames = [
            "session_id", "image_path", "detection_id", "confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "width_px", "width_mm", "mean_width_px", "median_width_px",
            "max_width_px", "min_width_px", "length_px",
            "orientation_deg", "area_px",
            "severity", "is456_compliant", "exposure_class",
            "measurement_method", "gsd_mm_per_px",
            "remediation_notes",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for result in results:
                rd = self._result_to_dict(result)
                for det in rd.get("detections", []):
                    row = {
                        "session_id": rd.get("session_id", ""),
                        "image_path": rd.get("image_path", ""),
                        "detection_id": det.get("detection_id", ""),
                        "confidence": det.get("confidence", ""),
                    }
                    bbox = det.get("bbox", (0, 0, 0, 0))
                    if bbox and len(bbox) == 4:
                        row["bbox_x1"], row["bbox_y1"] = bbox[0], bbox[1]
                        row["bbox_x2"], row["bbox_y2"] = bbox[2], bbox[3]

                    width = det.get("width", {})
                    row["width_px"] = width.get("width_px", "")
                    row["width_mm"] = width.get("width_mm", "")
                    row["mean_width_px"] = width.get("mean_width_px", "")
                    row["median_width_px"] = width.get("median_width_px", "")
                    row["max_width_px"] = width.get("max_width_px", "")
                    row["min_width_px"] = width.get("min_width_px", "")
                    row["measurement_method"] = width.get("method", "")
                    row["gsd_mm_per_px"] = width.get("gsd_mm_per_px", "")

                    geom = det.get("geometry", {})
                    row["length_px"] = geom.get("length_px", "")
                    row["orientation_deg"] = geom.get("orientation_deg", "")
                    row["area_px"] = geom.get("area_px", "")

                    sev = det.get("severity", {})
                    row["severity"] = sev.get("level", "")
                    row["is456_compliant"] = sev.get("is456_compliant", "")
                    row["exposure_class"] = sev.get("exposure_class", "")
                    row["remediation_notes"] = sev.get("remediation_notes", "")

                    writer.writerow(row)

        logger.info("CSV report saved to %s (%d results)", output_path, len(results))
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    def generate_html_report(
        self,
        result,
        annotated_image_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a professional HTML inspection report.

        Args:
            result: InspectionResult pydantic model.
            annotated_image_path: Optional path to annotated image to embed.
            output_path: Destination HTML path. Auto-generated if None.

        Returns:
            The absolute path of the written file.
        """
        rd = self._result_to_dict(result)

        if output_path is None:
            sid = rd.get("session_id", "report")
            output_path = f"inspection_report_{sid}.html"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        # Embed image as base64 if provided
        image_data_uri = ""
        if annotated_image_path and os.path.exists(annotated_image_path):
            try:
                with open(annotated_image_path, "rb") as img_f:
                    b64 = base64.b64encode(img_f.read()).decode("utf-8")
                    ext = Path(annotated_image_path).suffix.lower().replace(".", "")
                    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "png")
                    image_data_uri = f"data:image/{mime};base64,{b64}"
            except Exception as exc:
                logger.warning("Could not embed image: %s", exc)

        html = self._render_html(rd, image_data_uri)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("HTML report saved to %s", output_path)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _result_to_dict(self, result) -> dict:
        """Convert an InspectionResult (or dict) to a plain dict."""
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        if hasattr(result, "dict"):
            return result.dict()
        if isinstance(result, dict):
            return result
        return {}

    def _render_html(self, data: dict, image_data_uri: str = "") -> str:
        """Render the full HTML report string."""

        session_id = data.get("session_id", "N/A")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        source = data.get("source", "drone_camera")
        image_path = data.get("image_path", "N/A")
        img_w = data.get("image_width", "N/A")
        img_h = data.get("image_height", "N/A")
        model_name = data.get("model_name", "YOLOv11s-seg")
        model_weights = data.get("model_weights", "best.pt")
        total = data.get("total_detections", 0)
        highest = data.get("highest_severity", "NONE")
        flagged = data.get("flagged", False)
        detections = data.get("detections", [])

        # Count severities
        sev_counts: Dict[str, int] = {}
        for det in detections:
            lvl = det.get("severity", {}).get("level", "UNKNOWN")
            sev_counts[lvl] = sev_counts.get(lvl, 0) + 1

        severity_badges = ""
        for lvl in ["CRITICAL", "SEVERE", "MODERATE", "MINOR"]:
            cnt = sev_counts.get(lvl, 0)
            if cnt > 0:
                color = self.SEVERITY_COLORS.get(lvl, "#888")
                severity_badges += (
                    f'<span style="display:inline-block;padding:4px 14px;'
                    f'border-radius:20px;background:{color};color:#fff;'
                    f'font-weight:600;font-size:0.85rem;margin-right:6px;">'
                    f'{lvl}: {cnt}</span>\n'
                )

        # Build detection cards
        detection_cards = ""
        for idx, det in enumerate(detections):
            detection_cards += self._render_detection_card(idx, det)

        # Image section
        image_section = ""
        if image_data_uri:
            image_section = (
                '<div style="margin:24px 0;text-align:center;">'
                f'<img src="{image_data_uri}" style="max-width:100%;border-radius:8px;'
                'box-shadow:0 2px 12px rgba(0,0,0,0.12);" alt="Annotated inspection image"/>'
                '<p style="color:#64748b;margin-top:8px;font-size:0.85rem;">Annotated Inspection Image</p>'
                '</div>'
            )

        flagged_banner = ""
        if flagged:
            flagged_banner = (
                '<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;'
                'padding:12px 20px;margin-bottom:20px;color:#991b1b;font-weight:600;">'
                '⚠️  This inspection is FLAGGED — critical or severe cracks detected. '
                'Immediate review required.</div>'
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SID Bridge Inspection Report — {session_id}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f8fafc; color: #1e293b; line-height: 1.6;
  }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 32px 24px; }}
  .header {{
    background: linear-gradient(135deg, #1e3a5f 0%, #0f766e 100%);
    color: #fff; padding: 32px 28px; border-radius: 12px;
    margin-bottom: 24px;
  }}
  .header h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }}
  .header p {{ opacity: 0.85; font-size: 0.9rem; }}
  .meta-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px; margin: 20px 0;
  }}
  .meta-card {{
    background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 14px 18px;
  }}
  .meta-card .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 4px; }}
  .meta-card .value {{ font-size: 1rem; font-weight: 600; }}
  .section-title {{
    font-size: 1.15rem; font-weight: 700; margin: 28px 0 14px;
    padding-bottom: 8px; border-bottom: 2px solid #e2e8f0;
  }}
  .det-card {{
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 16px;
    transition: box-shadow 0.2s;
  }}
  .det-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
  .det-header {{ display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px; }}
  .det-header h3 {{ font-size: 1rem; }}
  .badge {{
    display: inline-block; padding: 3px 12px; border-radius: 16px;
    font-size: 0.78rem; font-weight: 600; color: #fff;
  }}
  .props-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 8px;
  }}
  .prop {{ padding: 8px 12px; background: #f8fafc; border-radius: 6px; }}
  .prop .p-label {{ font-size: 0.72rem; color: #64748b; text-transform: uppercase; }}
  .prop .p-value {{ font-size: 0.92rem; font-weight: 600; }}
  .compliance {{
    margin-top: 12px; padding: 10px 16px; border-radius: 6px;
    font-size: 0.85rem;
  }}
  .compliance.pass {{ background: #f0fdf4; border: 1px solid #86efac; color: #166534; }}
  .compliance.fail {{ background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; }}
  .compliance.unknown {{ background: #f8fafc; border: 1px solid #e2e8f0; color: #64748b; }}
  .footer {{
    text-align: center; margin-top: 40px; padding-top: 20px;
    border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.8rem;
  }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 0.88rem; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.03em; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>🔍 SID Bridge Inspection Report</h1>
    <p>Structural Inspection Drone — Automated Crack Analysis</p>
  </div>

  {flagged_banner}

  <div class="meta-grid">
    <div class="meta-card">
      <div class="label">Session ID</div>
      <div class="value">{session_id}</div>
    </div>
    <div class="meta-card">
      <div class="label">Timestamp</div>
      <div class="value">{timestamp}</div>
    </div>
    <div class="meta-card">
      <div class="label">Source</div>
      <div class="value">{source}</div>
    </div>
    <div class="meta-card">
      <div class="label">Image</div>
      <div class="value">{img_w} × {img_h} px</div>
    </div>
    <div class="meta-card">
      <div class="label">Model</div>
      <div class="value">{model_name}</div>
    </div>
    <div class="meta-card">
      <div class="label">Total Cracks</div>
      <div class="value">{total}</div>
    </div>
  </div>

  <h2 class="section-title">Severity Summary</h2>
  <div style="margin-bottom:8px;">{severity_badges if severity_badges else '<span style="color:#64748b;">No cracks detected.</span>'}</div>

  {image_section}

  <h2 class="section-title">Detection Details</h2>
  {detection_cards if detection_cards else '<p style="color:#64748b;">No cracks detected in this image.</p>'}

  <div class="footer">
    <p>Generated by SID — Structural Inspection Drone · IDEATOR GECT</p>
    <p>Government Engineering College Thrissur · Centre for Innovation</p>
    <p style="margin-top:4px;">Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  </div>

</div>
</body>
</html>"""
        return html

    def _render_detection_card(self, idx: int, det: dict) -> str:
        """Render a single detection card."""
        det_id = det.get("detection_id", f"det_{idx}")
        conf = det.get("confidence", 0)

        sev = det.get("severity", {})
        sev_level = sev.get("level", "UNKNOWN")
        sev_color = self.SEVERITY_COLORS.get(sev_level, "#888")
        is456 = sev.get("is456_compliant")
        remediation = sev.get("remediation_notes", "")
        exposure = sev.get("exposure_class", "")
        permissible = sev.get("permissible_width_mm")
        measured_mm = sev.get("measured_width_mm")

        width = det.get("width", {})
        w_px = width.get("width_px", "N/A")
        w_mm = width.get("width_mm")
        mean_w = width.get("mean_width_px", "N/A")
        median_w = width.get("median_width_px", "N/A")
        max_w = width.get("max_width_px", "N/A")
        method = width.get("method", "N/A")
        gsd = width.get("gsd_mm_per_px")

        geom = det.get("geometry", {})
        length = geom.get("length_px", "N/A")
        orientation = geom.get("orientation_deg", "N/A")
        area = geom.get("area_px", "N/A")

        bbox = det.get("bbox", [])
        bbox_str = f"[{', '.join(str(b) for b in bbox)}]" if bbox else "N/A"

        # Compliance indicator
        if is456 is True:
            comp_class = "pass"
            comp_text = "✅ IS 456:2000 Compliant"
        elif is456 is False:
            comp_class = "fail"
            comp_text = "❌ IS 456:2000 Non-Compliant"
        else:
            comp_class = "unknown"
            comp_text = "ℹ️ IS 456:2000 compliance not evaluated (mm measurement unavailable)"

        width_display = f"{w_px}"
        if w_mm is not None:
            width_display += f" ({w_mm:.3f} mm)"

        extra_info = ""
        if permissible is not None and measured_mm is not None:
            extra_info = (
                f'<div style="font-size:0.82rem;margin-top:6px;color:#475569;">'
                f'Measured: {measured_mm:.3f} mm · Permissible ({exposure}): {permissible:.2f} mm</div>'
            )

        return f"""
  <div class="det-card">
    <div class="det-header">
      <h3>🔎 {det_id}</h3>
      <span class="badge" style="background:{sev_color};">{sev_level}</span>
    </div>
    <div class="props-grid">
      <div class="prop"><div class="p-label">Width</div><div class="p-value">{width_display}</div></div>
      <div class="prop"><div class="p-label">Confidence</div><div class="p-value">{conf:.1%}</div></div>
      <div class="prop"><div class="p-label">Mean Width (px)</div><div class="p-value">{mean_w}</div></div>
      <div class="prop"><div class="p-label">Median Width (px)</div><div class="p-value">{median_w}</div></div>
      <div class="prop"><div class="p-label">Max Width (px)</div><div class="p-value">{max_w}</div></div>
      <div class="prop"><div class="p-label">Length (px)</div><div class="p-value">{length}</div></div>
      <div class="prop"><div class="p-label">Orientation</div><div class="p-value">{orientation}°</div></div>
      <div class="prop"><div class="p-label">Area (px²)</div><div class="p-value">{area}</div></div>
      <div class="prop"><div class="p-label">Bbox</div><div class="p-value">{bbox_str}</div></div>
      <div class="prop"><div class="p-label">Method</div><div class="p-value">{method}</div></div>
      <div class="prop"><div class="p-label">GSD (mm/px)</div><div class="p-value">{gsd if gsd else 'N/A'}</div></div>
    </div>
    <div class="compliance {comp_class}">{comp_text}</div>
    {extra_info}
    {f'<div style="margin-top:10px;padding:8px 14px;background:#f8fafc;border-radius:6px;font-size:0.84rem;"><strong>Remediation:</strong> {remediation}</div>' if remediation else ''}
  </div>"""
