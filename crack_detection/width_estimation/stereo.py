"""
Stereo Depth-Based Width Estimation
====================================
Crack-width measurement using a stereo depth map (e.g. ZED 2 camera).

Two estimation modes are provided:

* **2.5-D** — pixel width × depth / focal-length (``estimate_width``).
* **Full 3-D** — project perpendicular edge points to 3-D space and
  compute Euclidean distance (``estimate_width_3d``).

The core algorithm reuses the same skeletonisation + perpendicular
ray-casting approach as the monocular estimator.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from crack_detection.schemas import MeasurementMethod, WidthMeasurement
from crack_detection.width_estimation.common import ScaleInfo

# Attempt to import the ZED SDK — gracefully degrade if not installed.
try:
    import pyzed.sl as sl  # type: ignore[import-untyped]

    _ZED_AVAILABLE = True
except ImportError:
    _ZED_AVAILABLE = False

logger = logging.getLogger(__name__)


class StereoWidthEstimator:
    """Estimate crack width from a depth map and camera intrinsics.

    Parameters
    ----------
    camera_matrix:
        ``(3, 3)`` camera intrinsic matrix.  If ``None``, must be
        supplied per-call.
    baseline_mm:
        Stereo baseline distance in millimetres (default 120 mm for
        ZED 2).
    """

    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        baseline_mm: float = 120.0,
    ) -> None:
        self.camera_matrix = camera_matrix
        self.baseline_mm = baseline_mm

        # Internal: reuse monocular helpers lazily
        self._mono: Optional["_SkeletonHelper"] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_zed_camera(cls, resolution: str = "HD720") -> "StereoWidthEstimator":
        """Initialise from a live ZED camera (if the SDK is installed).

        Parameters
        ----------
        resolution:
            ZED resolution preset (``HD720``, ``HD1080``, ``HD2K``).

        Returns
        -------
        StereoWidthEstimator

        Raises
        ------
        ImportError
            If ``pyzed`` is not available.
        RuntimeError
            If the ZED camera cannot be opened.
        """
        if not _ZED_AVAILABLE:
            raise ImportError(
                "pyzed.sl is not installed — cannot initialise from ZED camera. "
                "Install the ZED SDK and its Python wrapper first."
            )

        zed = sl.Camera()
        init_params = sl.InitParameters()
        res_map = {
            "HD720": sl.RESOLUTION.HD720,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD2K": sl.RESOLUTION.HD2K,
        }
        init_params.camera_resolution = res_map.get(
            resolution, sl.RESOLUTION.HD720
        )
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA

        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Failed to open ZED camera: {status}"
            )

        calib = zed.get_camera_information().camera_configuration.calibration_parameters
        left = calib.left_cam
        fx, fy = left.fx, left.fy
        cx, cy = left.cx, left.cy
        baseline = calib.get_camera_baseline()

        camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )
        zed.close()

        logger.info(
            "ZED camera initialised: fx=%.1f, fy=%.1f, baseline=%.1f mm.",
            fx, fy, baseline,
        )
        return cls(camera_matrix=camera_matrix, baseline_mm=float(baseline))

    # ==================================================================
    # Public API — 2.5-D width
    # ==================================================================

    def estimate_width(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> WidthMeasurement:
        """Estimate crack width using pixel width × depth / focal_length.

        Parameters
        ----------
        mask:
            Binary crack mask ``(H, W)``.
        depth_map:
            Depth image ``(H, W)`` in **millimetres** (float or uint16).
        camera_matrix:
            Optional per-call ``(3, 3)`` intrinsic override.

        Returns
        -------
        WidthMeasurement
        """
        K = camera_matrix if camera_matrix is not None else self.camera_matrix
        if K is None:
            raise ValueError(
                "camera_matrix must be provided either at init or per-call."
            )

        fx = float(K[0, 0])

        helper = self._get_helper()

        binary = (mask > 0).astype(np.uint8)
        if binary.sum() < 20:
            return self._empty_measurement()

        # Morphological refinement
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)

        skeleton = skeletonize(refined > 0).astype(np.uint8)
        if skeleton.sum() == 0:
            return self._empty_measurement()

        ordered = helper.order_skeleton_points(skeleton)
        if len(ordered) < 3:
            return self._empty_measurement()

        raw: List[Tuple[float, float, float]] = []
        widths_mm_list: List[float] = []

        for i in range(0, len(ordered), 3):
            pt = ordered[i]
            angle = helper.compute_local_orientation(ordered, i, 15)
            width_px, e1, e2 = helper.measure_perpendicular_width(
                pt, angle, refined
            )
            if width_px <= 0:
                continue

            # Robust depth at skeleton point
            depth_mm = self._get_robust_depth(
                depth_map, (int(round(pt[0])), int(round(pt[1])))
            )
            if depth_mm is None or depth_mm <= 0:
                continue

            w_mm = float(width_px * depth_mm / fx)
            raw.append((float(pt[0]), float(pt[1]), width_px))
            widths_mm_list.append(w_mm)

        if not raw:
            return self._empty_measurement()

        widths_px = np.array([m[2] for m in raw])
        widths_mm = np.array(widths_mm_list)

        # Outlier removal
        med_px = float(np.median(widths_px))
        if med_px > 0:
            keep = widths_px <= 3.0 * med_px
            raw = [m for m, k in zip(raw, keep) if k]
            widths_px = widths_px[keep]
            widths_mm = widths_mm[keep]

        med_mm = float(np.median(widths_mm))

        return WidthMeasurement(
            width_px=float(np.median(widths_px)),
            width_mm=med_mm,
            method=MeasurementMethod.STEREO_DEPTH,
            measurement_points=raw,
            mean_width_px=float(np.mean(widths_px)),
            median_width_px=float(np.median(widths_px)),
            max_width_px=float(np.max(widths_px)),
            min_width_px=float(np.min(widths_px)),
            std_width_px=float(np.std(widths_px)),
            percentile_95_width_px=float(np.percentile(widths_px, 95)),
            gsd_mm_per_px=None,
            scale_factor=None,
        )

    # ==================================================================
    # Public API — full 3-D width
    # ==================================================================

    def estimate_width_3d(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> WidthMeasurement:
        """Estimate crack width by projecting edge points to 3-D.

        This is the most accurate method: for each perpendicular width
        sample, both edge points are back-projected to 3-D using the
        depth map and camera intrinsics, and the true Euclidean distance
        is computed.

        Parameters
        ----------
        mask:
            Binary crack mask ``(H, W)``.
        depth_map:
            Depth image ``(H, W)`` in millimetres.
        camera_matrix:
            ``(3, 3)`` camera intrinsic matrix.

        Returns
        -------
        WidthMeasurement
        """
        K = camera_matrix
        helper = self._get_helper()

        binary = (mask > 0).astype(np.uint8)
        if binary.sum() < 20:
            return self._empty_measurement()

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)

        skeleton = skeletonize(refined > 0).astype(np.uint8)
        if skeleton.sum() == 0:
            return self._empty_measurement()

        ordered = helper.order_skeleton_points(skeleton)
        if len(ordered) < 3:
            return self._empty_measurement()

        raw: List[Tuple[float, float, float]] = []
        widths_3d: List[float] = []

        for i in range(0, len(ordered), 3):
            pt = ordered[i]
            angle = helper.compute_local_orientation(ordered, i, 15)

            perp = angle + np.pi / 2.0
            direction = np.array([np.cos(perp), np.sin(perp)])

            e_pos = helper.cast_ray(pt, direction, refined)
            e_neg = helper.cast_ray(pt, -direction, refined)
            if e_pos is None or e_neg is None:
                continue

            width_px = float(np.hypot(e_pos[0] - e_neg[0], e_pos[1] - e_neg[1]))
            if width_px <= 0:
                continue

            # 3-D projection
            d1 = self._get_robust_depth(depth_map, e_pos)
            d2 = self._get_robust_depth(depth_map, e_neg)
            if d1 is None or d2 is None or d1 <= 0 or d2 <= 0:
                continue

            p3d_1 = self._pixel_to_3d(e_pos[0], e_pos[1], d1, K)
            p3d_2 = self._pixel_to_3d(e_neg[0], e_neg[1], d2, K)
            w3d = float(np.linalg.norm(p3d_1 - p3d_2))

            raw.append((float(pt[0]), float(pt[1]), width_px))
            widths_3d.append(w3d)

        if not raw:
            return self._empty_measurement()

        widths_px = np.array([m[2] for m in raw])
        widths_3d_arr = np.array(widths_3d)

        # Outlier removal
        med_px = float(np.median(widths_px))
        if med_px > 0:
            keep = widths_px <= 3.0 * med_px
            raw = [m for m, k in zip(raw, keep) if k]
            widths_px = widths_px[keep]
            widths_3d_arr = widths_3d_arr[keep]

        med_3d = float(np.median(widths_3d_arr))

        return WidthMeasurement(
            width_px=float(np.median(widths_px)),
            width_mm=med_3d,
            method=MeasurementMethod.STEREO_DEPTH,
            measurement_points=raw,
            mean_width_px=float(np.mean(widths_px)),
            median_width_px=float(np.median(widths_px)),
            max_width_px=float(np.max(widths_px)),
            min_width_px=float(np.min(widths_px)),
            std_width_px=float(np.std(widths_px)),
            percentile_95_width_px=float(np.percentile(widths_px, 95)),
            gsd_mm_per_px=None,
            scale_factor=None,
        )

    # ==================================================================
    # Depth helpers
    # ==================================================================

    @staticmethod
    def _get_robust_depth(
        depth_map: np.ndarray,
        point: Tuple[int, int],
        window: int = 5,
    ) -> Optional[float]:
        """Median depth in a window around a pixel, ignoring NaN / zero.

        Parameters
        ----------
        depth_map:
            ``(H, W)`` depth image in mm.
        point:
            ``(x, y)`` pixel coordinate.
        window:
            Side length of the neighbourhood (must be odd).

        Returns
        -------
        Optional[float]
            Robust depth in mm, or ``None`` if no valid depth is found.
        """
        h, w = depth_map.shape[:2]
        x, y = int(point[0]), int(point[1])
        half = window // 2

        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)

        patch = depth_map[y0:y1, x0:x1].astype(np.float64).ravel()
        valid = patch[(~np.isnan(patch)) & (patch > 0)]

        if len(valid) == 0:
            return None
        return float(np.median(valid))

    @staticmethod
    def _pixel_to_3d(
        u: int,
        v: int,
        depth: float,
        camera_matrix: np.ndarray,
    ) -> np.ndarray:
        """Back-project a pixel + depth to a 3-D point.

        Parameters
        ----------
        u, v:
            Pixel coordinates (x, y).
        depth:
            Depth at the pixel in mm.
        camera_matrix:
            ``(3, 3)`` intrinsic matrix ``[[fx, 0, cx], [0, fy, cy], …]``.

        Returns
        -------
        np.ndarray
            ``(X, Y, Z)`` 3-D point in camera frame (mm).
        """
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z], dtype=np.float64)

    # ==================================================================
    # Internal skeleton helper (reuses monocular logic)
    # ==================================================================

    def _get_helper(self) -> "_SkeletonHelper":
        """Lazily create a helper that wraps monocular skeleton routines."""
        if self._mono is None:
            self._mono = _SkeletonHelper()
        return self._mono

    @staticmethod
    def _empty_measurement() -> WidthMeasurement:
        """Return a zeroed-out measurement."""
        return WidthMeasurement(
            width_px=0.0,
            width_mm=None,
            method=MeasurementMethod.STEREO_DEPTH,
            measurement_points=[],
            mean_width_px=0.0,
            median_width_px=0.0,
            max_width_px=0.0,
            min_width_px=0.0,
            std_width_px=0.0,
            percentile_95_width_px=0.0,
        )


# ======================================================================
# Lightweight skeleton helper (avoids circular import with monocular)
# ======================================================================

class _SkeletonHelper:
    """Minimal skeleton-processing helpers replicating the core routines
    from :class:`MonocularWidthEstimator` without importing it, to avoid
    circular dependencies.
    """

    @staticmethod
    def order_skeleton_points(skeleton: np.ndarray) -> np.ndarray:
        """Nearest-neighbour ordering of skeleton pixels."""
        ys, xs = np.nonzero(skeleton)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float64)

        coords = np.column_stack([xs, ys]).astype(np.float64)

        # Find an endpoint to start from
        kernel = np.ones((3, 3), dtype=np.uint8)
        nbr_count = cv2.filter2D(
            skeleton, -1, kernel, borderType=cv2.BORDER_CONSTANT
        )
        ep_mask = (skeleton == 1) & (nbr_count == 2)
        ep_ys, ep_xs = np.nonzero(ep_mask)

        if len(ep_xs) > 0:
            dists = np.abs(coords[:, 0] - ep_xs[0]) + np.abs(coords[:, 1] - ep_ys[0])
            start = int(np.argmin(dists))
        else:
            start = 0

        n = len(coords)
        visited = np.zeros(n, dtype=bool)
        order = np.empty((n, 2), dtype=np.float64)
        cur = start

        for i in range(n):
            order[i] = coords[cur]
            visited[cur] = True
            if i == n - 1:
                break
            diff = coords[~visited] - coords[cur]
            sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
            uv = np.where(~visited)[0]
            cur = uv[int(np.argmin(sq))]

        return order

    @staticmethod
    def compute_local_orientation(
        points: np.ndarray, idx: int, window: int
    ) -> float:
        """PCA orientation at a skeleton point."""
        half = window // 2
        lo, hi = max(0, idx - half), min(len(points), idx + half + 1)
        w = points[lo:hi]
        if len(w) < 2:
            return 0.0
        c = w - w.mean(axis=0)
        cov = np.cov(c, rowvar=False)
        if cov.ndim < 2:
            return 0.0
        _, vecs = np.linalg.eigh(cov)
        p = vecs[:, -1]
        return float(np.arctan2(p[1], p[0]))

    @staticmethod
    def measure_perpendicular_width(
        point: np.ndarray,
        angle: float,
        mask: np.ndarray,
    ) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
        """Width via perpendicular ray casting."""
        perp = angle + np.pi / 2.0
        d = np.array([np.cos(perp), np.sin(perp)])
        origin = (0, 0)

        e_pos = _SkeletonHelper.cast_ray(point, d, mask)
        e_neg = _SkeletonHelper.cast_ray(point, -d, mask)
        if e_pos is None or e_neg is None:
            return 0.0, origin, origin

        width = float(np.hypot(e_pos[0] - e_neg[0], e_pos[1] - e_neg[1]))
        return width, e_pos, e_neg

    @staticmethod
    def cast_ray(
        start: np.ndarray,
        direction: np.ndarray,
        mask: np.ndarray,
        max_distance: int = 200,
    ) -> Optional[Tuple[int, int]]:
        """Step along *direction* until exiting the mask."""
        h, w = mask.shape
        x, y = float(start[0]), float(start[1])
        last: Optional[Tuple[int, int]] = None

        for _ in range(max_distance):
            ix, iy = int(round(x)), int(round(y))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                break
            if mask[iy, ix] > 0:
                last = (ix, iy)
            else:
                if last is not None:
                    return last
                return None
            x += direction[0]
            y += direction[1]

        return last
