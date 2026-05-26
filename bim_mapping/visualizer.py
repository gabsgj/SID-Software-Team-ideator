"""
BIM Visualizer
==============
Visualises IFC model geometry with crack detection overlays.

Supports:
- **Interactive 3-D** (Plotly) — structural elements + crack locations
- **Static 3-D** (Matplotlib) — publication-quality PNG
- **Crack-on-image overlay** — bounding boxes, skeletons, width markers
- **2-D crack map** — floor-plan style element outlines with crack positions
- **Severity heatmap** — interpolated 2-D heatmap of crack severity
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional heavy dependencies — degrade gracefully
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning(
        "matplotlib not installed. Static visualisations unavailable."
    )

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning(
        "plotly not installed. Interactive visualisations unavailable."
    )

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning(
        "OpenCV not installed. Image overlay visualisations unavailable."
    )

# ---- Severity colour scheme ----
_SEVERITY_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "MINOR":    (0, 200, 0),       # green
    "MODERATE": (230, 230, 0),     # yellow
    "SEVERE":   (255, 140, 0),     # orange
    "CRITICAL": (220, 0, 0),       # red
}

_SEVERITY_COLORS_NORM: Dict[str, Tuple[float, float, float]] = {
    k: (r / 255.0, g / 255.0, b / 255.0)
    for k, (r, g, b) in _SEVERITY_COLORS_RGB.items()
}

_ELEMENT_COLORS_NORM: Dict[str, str] = {
    "IfcBeam":       "royalblue",
    "IfcColumn":     "mediumseagreen",
    "IfcSlab":       "silver",
    "IfcWall":       "tan",
    "IfcMember":     "mediumpurple",
    "IfcPlate":      "lightskyblue",
    "IfcFooting":    "peru",
    "IfcBridgePart": "steelblue",
}


class BIMVisualizer:
    """Visualise structural elements and crack detections.

    No IFC file is required — all inputs are plain dicts / arrays so that
    pre-computed data can be visualised even without ``ifcopenshell``.
    """

    def __init__(self) -> None:
        logger.info(
            "BIMVisualizer initialised (matplotlib=%s, plotly=%s, cv2=%s).",
            HAS_MPL, HAS_PLOTLY, HAS_CV2,
        )

    # ------------------------------------------------------------------
    # 3-D element + crack visualisation
    # ------------------------------------------------------------------
    def visualize_elements(
        self,
        elements: List[Dict],
        cracks: Optional[List[Dict]] = None,
        output_path: Optional[str] = None,
        interactive: bool = False,
    ) -> Optional[str]:
        """Render structural elements as 3-D bounding boxes with optional cracks.

        Parameters
        ----------
        elements : list[dict]
            Each dict must contain ``Type`` and ``bbox`` with keys
            ``min`` ([x,y,z]) and ``max`` ([x,y,z]).
        cracks : list[dict], optional
            Each dict should have ``position`` ([x,y,z]),
            ``severity_level``, and optionally ``width_mm``.
        output_path : str, optional
            Where to save the output file.  Extension determines format.
        interactive : bool
            If *True*, produce an interactive Plotly HTML; otherwise a
            static Matplotlib PNG.

        Returns
        -------
        str or None
            Path to the saved file, or *None* if rendering is skipped.
        """
        if interactive:
            return self._create_plotly_visualization(
                elements, cracks, output_path,
            )
        return self._create_matplotlib_visualization(
            elements, cracks, output_path,
        )

    # ------------------------------------------------------------------
    # Plotly interactive
    # ------------------------------------------------------------------
    def _create_plotly_visualization(
        self,
        elements: List[Dict],
        cracks: Optional[List[Dict]],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Build an interactive 3-D Plotly visualisation.

        Parameters
        ----------
        elements : list[dict]
            Structural element records with ``bbox``.
        cracks : list[dict] or None
            Crack detection records with ``position``.
        output_path : str or None
            Save location (HTML).

        Returns
        -------
        str or None
            Saved file path.
        """
        if not HAS_PLOTLY:
            logger.error("Plotly is required for interactive visualisation.")
            return None

        fig = go.Figure()

        # ---- Elements as translucent boxes ----
        for elem in elements:
            bbox = elem.get("bbox")
            if bbox is None:
                continue
            bmin = np.asarray(bbox["min"])
            bmax = np.asarray(bbox["max"])
            etype = elem.get("Type", "Unknown")
            color = _ELEMENT_COLORS_NORM.get(etype, "gray")
            name = elem.get("Name", etype)

            # 8 vertices of the AABB
            verts = np.array([
                [bmin[0], bmin[1], bmin[2]],
                [bmax[0], bmin[1], bmin[2]],
                [bmax[0], bmax[1], bmin[2]],
                [bmin[0], bmax[1], bmin[2]],
                [bmin[0], bmin[1], bmax[2]],
                [bmax[0], bmin[1], bmax[2]],
                [bmax[0], bmax[1], bmax[2]],
                [bmin[0], bmax[1], bmax[2]],
            ])

            # 12 triangles for 6 faces
            i_idx = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
            j_idx = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
            k_idx = [2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 6, 7]

            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=i_idx, j=j_idx, k=k_idx,
                color=color,
                opacity=0.25,
                name=name,
                hoverinfo="name",
            ))

        # ---- Cracks as coloured spheres ----
        if cracks:
            for severity in ("MINOR", "MODERATE", "SEVERE", "CRITICAL"):
                subset = [
                    c for c in cracks
                    if c.get("severity_level") == severity and "position" in c
                ]
                if not subset:
                    continue
                xs = [c["position"][0] for c in subset]
                ys = [c["position"][1] for c in subset]
                zs = [c["position"][2] for c in subset]
                hover = [
                    (
                        f"ID: {c.get('detection_id', '?')}<br>"
                        f"Width: {c.get('width_mm', 'N/A')} mm<br>"
                        f"Severity: {severity}"
                    )
                    for c in subset
                ]
                r, g, b = _SEVERITY_COLORS_RGB.get(severity, (128, 128, 128))
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=f"rgb({r},{g},{b})",
                        symbol="circle",
                    ),
                    name=f"Cracks – {severity}",
                    text=hover,
                    hoverinfo="text",
                ))

        fig.update_layout(
            title="BIM Crack Visualisation",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        out = output_path or "bim_cracks_3d.html"
        fig.write_html(out)
        logger.info("Interactive 3-D visualisation saved to: %s", out)
        return out

    # ------------------------------------------------------------------
    # Matplotlib static
    # ------------------------------------------------------------------
    def _create_matplotlib_visualization(
        self,
        elements: List[Dict],
        cracks: Optional[List[Dict]],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Create a static Matplotlib 3-D visualisation.

        Parameters
        ----------
        elements : list[dict]
            Structural element records.
        cracks : list[dict] or None
            Crack records.
        output_path : str or None
            Save location (PNG).

        Returns
        -------
        str or None
            Saved file path.
        """
        if not HAS_MPL:
            logger.error(
                "matplotlib is required for static visualisation."
            )
            return None

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # ---- Elements ----
        for elem in elements:
            bbox = elem.get("bbox")
            if bbox is None:
                continue
            bmin = np.asarray(bbox["min"])
            bmax = np.asarray(bbox["max"])
            etype = elem.get("Type", "Unknown")
            color = _ELEMENT_COLORS_NORM.get(etype, "gray")

            # Build 6 faces of the AABB
            faces = _bbox_faces(bmin, bmax)
            poly = Poly3DCollection(
                faces, alpha=0.15, facecolor=color,
                edgecolor=color, linewidth=0.5,
            )
            ax.add_collection3d(poly)

        # ---- Cracks ----
        if cracks:
            for c in cracks:
                pos = c.get("position")
                if pos is None:
                    continue
                sev = c.get("severity_level", "MINOR")
                col = _SEVERITY_COLORS_NORM.get(sev, (0.5, 0.5, 0.5))
                ax.scatter(
                    pos[0], pos[1], pos[2],
                    c=[col], s=60, marker="o", edgecolors="black",
                    linewidths=0.5, depthshade=True,
                )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("BIM Crack Visualisation")

        # Auto-scale
        _auto_scale_3d(ax, elements)

        out = output_path or "bim_cracks_3d.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Static 3-D visualisation saved to: %s", out)
        return out

    # ------------------------------------------------------------------
    # Crack overlay on camera image
    # ------------------------------------------------------------------
    def visualize_crack_on_image(
        self,
        image: np.ndarray,
        crack_detections: List[Dict],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw crack detections on a camera image.

        Renders bounding boxes, skeleton overlays, perpendicular width
        measurement lines, and text labels.

        Parameters
        ----------
        image : np.ndarray
            BGR or RGB image (H×W×3).
        crack_detections : list[dict]
            Each detection should include ``bbox``, ``severity_level``,
            and optionally ``skeleton_points``, ``width_mm``, ``mask_polygon``.
        output_path : str, optional
            If provided, save the annotated image.

        Returns
        -------
        np.ndarray
            Annotated image (same colour space as input).
        """
        if not HAS_CV2:
            logger.error(
                "OpenCV is required for image overlay visualisation."
            )
            return image

        annotated = image.copy()

        for det in crack_detections:
            severity = det.get("severity_level", "MINOR")
            bgr = _SEVERITY_COLORS_RGB.get(severity, (128, 128, 128))
            # OpenCV uses BGR
            color_bgr = (bgr[2], bgr[1], bgr[0])

            # Bounding box
            bbox = det.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

                # Label
                width_text = (
                    f"{det['width_mm']:.2f} mm"
                    if det.get("width_mm") is not None
                    else f"{x2 - x1} px"
                )
                label = f"{severity} | {width_text}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - 8), (x1 + tw + 4, y1),
                    color_bgr, -1,
                )
                cv2.putText(
                    annotated, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

            # Skeleton overlay
            skeleton = det.get("skeleton_points")
            if skeleton and len(skeleton) >= 2:
                pts = [tuple(int(c) for c in p) for p in skeleton]
                for i in range(len(pts) - 1):
                    cv2.line(annotated, pts[i], pts[i + 1], color_bgr, 2)

                # Width measurement lines (perpendicular to skeleton)
                self._draw_width_markers(annotated, pts, color_bgr)

            # Mask polygon
            mask_poly = det.get("mask_polygon")
            if mask_poly and len(mask_poly) >= 3:
                poly_pts = np.array(mask_poly, dtype=np.int32).reshape(-1, 1, 2)
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [poly_pts], color_bgr)
                cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
                cv2.polylines(annotated, [poly_pts], True, color_bgr, 1)

        if output_path is not None:
            cv2.imwrite(output_path, annotated)
            logger.info("Crack overlay image saved to: %s", output_path)

        return annotated

    # ------------------------------------------------------------------
    # 2-D floor-plan crack map
    # ------------------------------------------------------------------
    def generate_crack_map(
        self,
        all_cracks: List[Dict],
        element_mappings: List[Dict],
        output_path: str,
    ) -> str:
        """Generate a 2-D floor-plan style crack map.

        Elements are drawn as rectangles (XY projection of their AABB).
        Crack positions are marked with severity-coloured dots.

        Parameters
        ----------
        all_cracks : list[dict]
            Crack records with ``position`` and ``severity_level``.
        element_mappings : list[dict]
            Element records with ``bbox`` and ``Type``.
        output_path : str
            Path to save the map (PNG or HTML).

        Returns
        -------
        str
            Path to the saved file.
        """
        if not HAS_MPL:
            logger.error("matplotlib is required for crack map generation.")
            return output_path

        fig, ax = plt.subplots(figsize=(14, 10))

        # ---- Element outlines ----
        for elem in element_mappings:
            bbox = elem.get("bbox")
            if bbox is None:
                continue
            bmin = np.asarray(bbox["min"])
            bmax = np.asarray(bbox["max"])
            etype = elem.get("Type", "Unknown")
            color = _ELEMENT_COLORS_NORM.get(etype, "gray")
            w = bmax[0] - bmin[0]
            h = bmax[1] - bmin[1]
            rect = Rectangle(
                (bmin[0], bmin[1]), w, h,
                linewidth=1.2, edgecolor=color,
                facecolor=color, alpha=0.15,
            )
            ax.add_patch(rect)
            ax.text(
                bmin[0] + w / 2, bmin[1] + h / 2,
                elem.get("Name", etype),
                fontsize=6, ha="center", va="center",
                color="dimgray",
            )

        # ---- Crack positions ----
        severity_counts: Dict[str, int] = {}
        for crack in all_cracks:
            pos = crack.get("position")
            if pos is None:
                continue
            sev = crack.get("severity_level", "MINOR")
            col = _SEVERITY_COLORS_NORM.get(sev, (0.5, 0.5, 0.5))
            ax.plot(
                pos[0], pos[1], "o",
                color=col, markersize=7,
                markeredgecolor="black", markeredgewidth=0.5,
            )
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # ---- Summary table ----
        if severity_counts:
            cell_text = [
                [sev, str(cnt)]
                for sev, cnt in sorted(severity_counts.items())
            ]
            table = ax.table(
                cellText=cell_text,
                colLabels=["Severity", "Count"],
                loc="upper right",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(0.5, 1.2)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("2-D Crack Map")
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale()
        ax.grid(True, alpha=0.3)

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Crack map saved to: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Severity heatmap
    # ------------------------------------------------------------------
    def create_severity_heatmap(
        self,
        cracks: List[Dict],
        grid_resolution: int = 50,
    ) -> np.ndarray:
        """Create a 2-D severity heatmap across the bridge surface.

        Severity levels are encoded numerically (MINOR=1 … CRITICAL=4)
        and interpolated over a regular grid using inverse-distance
        weighting.

        Parameters
        ----------
        cracks : list[dict]
            Crack records with ``position`` and ``severity_level``.
        grid_resolution : int
            Number of grid cells along each axis.

        Returns
        -------
        np.ndarray
            2-D heatmap array (float, shape ``(grid_resolution, grid_resolution)``).
        """
        severity_map = {"MINOR": 1, "MODERATE": 2, "SEVERE": 3, "CRITICAL": 4}

        valid = [
            c for c in cracks
            if c.get("position") is not None
        ]
        if not valid:
            logger.warning("No cracks with positions for heatmap.")
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float64)

        positions = np.array([c["position"][:2] for c in valid])
        severities = np.array([
            severity_map.get(c.get("severity_level", "MINOR"), 1)
            for c in valid
        ], dtype=np.float64)

        xmin, ymin = positions.min(axis=0) - 1.0
        xmax, ymax = positions.max(axis=0) + 1.0

        xi = np.linspace(xmin, xmax, grid_resolution)
        yi = np.linspace(ymin, ymax, grid_resolution)
        xx, yy = np.meshgrid(xi, yi)
        grid = np.zeros_like(xx)

        # Inverse-distance weighting
        eps = 1e-8
        for gx in range(grid_resolution):
            for gy in range(grid_resolution):
                dists = np.sqrt(
                    (positions[:, 0] - xx[gy, gx]) ** 2
                    + (positions[:, 1] - yy[gy, gx]) ** 2
                )
                weights = 1.0 / (dists + eps) ** 2
                grid[gy, gx] = np.sum(weights * severities) / np.sum(weights)

        logger.info(
            "Severity heatmap generated (%d×%d, %d cracks).",
            grid_resolution, grid_resolution, len(valid),
        )
        return grid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_width_markers(
        image: np.ndarray,
        skeleton_pts: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        marker_length: int = 12,
    ) -> None:
        """Draw perpendicular width measurement lines along the skeleton.

        Markers are placed at regular intervals along the skeleton.
        """
        if len(skeleton_pts) < 2:
            return

        step = max(1, len(skeleton_pts) // 5)
        for idx in range(1, len(skeleton_pts) - 1, step):
            p_prev = np.array(skeleton_pts[idx - 1], dtype=np.float64)
            p_next = np.array(skeleton_pts[min(idx + 1, len(skeleton_pts) - 1)], dtype=np.float64)
            tangent = p_next - p_prev
            length = np.linalg.norm(tangent)
            if length < 1e-6:
                continue
            tangent /= length
            normal = np.array([-tangent[1], tangent[0]])

            centre = np.array(skeleton_pts[idx], dtype=np.float64)
            p1 = (centre + normal * marker_length / 2).astype(int)
            p2 = (centre - normal * marker_length / 2).astype(int)

            cv2.line(
                image,
                tuple(p1), tuple(p2),
                color, 1, cv2.LINE_AA,
            )


# ======================================================================
# Module-level helpers
# ======================================================================

def _bbox_faces(
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> List[np.ndarray]:
    """Return the 6 faces of an axis-aligned bounding box as vertex lists."""
    v = np.array([
        [bmin[0], bmin[1], bmin[2]],
        [bmax[0], bmin[1], bmin[2]],
        [bmax[0], bmax[1], bmin[2]],
        [bmin[0], bmax[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]],
        [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmax[2]],
        [bmin[0], bmax[1], bmax[2]],
    ])
    faces = [
        v[[0, 1, 2, 3]],  # bottom
        v[[4, 5, 6, 7]],  # top
        v[[0, 1, 5, 4]],  # front
        v[[2, 3, 7, 6]],  # back
        v[[0, 3, 7, 4]],  # left
        v[[1, 2, 6, 5]],  # right
    ]
    return faces


def _auto_scale_3d(ax, elements: List[Dict]) -> None:
    """Set equal aspect ratio for a Matplotlib 3-D axes from element bboxes."""
    all_mins, all_maxs = [], []
    for elem in elements:
        bbox = elem.get("bbox")
        if bbox is None:
            continue
        all_mins.append(bbox["min"])
        all_maxs.append(bbox["max"])

    if not all_mins:
        return

    mins = np.min(all_mins, axis=0)
    maxs = np.max(all_maxs, axis=0)
    centre = (mins + maxs) / 2.0
    span = max((maxs - mins).max(), 1.0) / 2.0

    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(centre[2] - span, centre[2] + span)
