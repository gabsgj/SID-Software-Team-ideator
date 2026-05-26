"""
Monocular Width Estimation
==========================
State-of-the-art crack-width measurement using **skeletonisation +
perpendicular edge distance**.

Algorithm overview
------------------
1. Morphological refinement of the binary crack mask.
2. Zhang–Suen thinning (``skimage.morphology.skeletonize``) to obtain
   the medial axis.
3. Skeleton pruning — removal of short spurious branches.
4. Ordered traversal of skeleton pixels (nearest-neighbour chain from an
   endpoint).
5. Local orientation via PCA on a sliding window of skeleton coordinates.
6. Perpendicular ray-casting to both edges of the crack mask.
7. Statistical aggregation with outlier filtering (> 3 × median).
8. Optional pixel → mm conversion using :class:`ScaleInfo`.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from crack_detection.schemas import MeasurementMethod, WidthMeasurement
from crack_detection.width_estimation.common import ScaleInfo, pixels_to_mm

logger = logging.getLogger(__name__)


class MonocularWidthEstimator:
    """Estimate crack width from a 2-D binary mask using monocular imagery.

    Parameters
    ----------
    scale_info:
        Pre-computed pixel-to-mm scaling.  If ``None``, results are
        reported in pixels only.
    neighborhood_size:
        Number of skeleton points used in the PCA window for local
        orientation estimation.
    sample_spacing:
        Spacing (in skeleton-point indices) between successive width
        samples.  A value of 3 means every 3rd ordered skeleton point
        is sampled.
    min_branch_length:
        Minimum number of pixels for a skeleton branch to survive
        pruning.
    """

    def __init__(
        self,
        scale_info: Optional[ScaleInfo] = None,
        neighborhood_size: int = 15,
        sample_spacing: int = 3,
        min_branch_length: int = 15,
    ) -> None:
        self.scale_info = scale_info
        self.neighborhood_size = max(3, neighborhood_size)
        self.sample_spacing = max(1, sample_spacing)
        self.min_branch_length = max(3, min_branch_length)

    # ==================================================================
    # Public API
    # ==================================================================

    def estimate_width(
        self,
        mask: np.ndarray,
        scale_info: Optional[ScaleInfo] = None,
    ) -> WidthMeasurement:
        """Run the full width-estimation pipeline on a binary crack mask.

        Parameters
        ----------
        mask:
            Binary crack mask of shape ``(H, W)``.  Values should be in
            ``{0, 1}`` or ``{0, 255}``.
        scale_info:
            Per-call override for the scaling information.

        Returns
        -------
        WidthMeasurement
            Fully populated measurement result.
        """
        active_scale = scale_info or self.scale_info

        # ------ 1. Input validation ------
        if mask is None or mask.size == 0:
            logger.warning("Empty mask — returning zero-width measurement.")
            return self._empty_measurement(active_scale)

        binary = (mask > 0).astype(np.uint8)
        crack_pixels = int(binary.sum())
        if crack_pixels < 20:
            logger.warning(
                "Insufficient crack pixels (%d < 20) — returning zero.",
                crack_pixels,
            )
            return self._empty_measurement(active_scale)

        # ------ 2. Morphological refinement ------
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)

        # ------ 3. Skeletonize (Zhang–Suen) ------
        skeleton = self._skeletonize(refined)
        if skeleton.sum() == 0:
            logger.warning("Skeleton is empty after thinning.")
            return self._empty_measurement(active_scale)

        # ------ 4. Prune short branches ------
        skeleton = self._prune_skeleton(skeleton, self.min_branch_length)
        if skeleton.sum() == 0:
            logger.warning("Skeleton is empty after pruning.")
            return self._empty_measurement(active_scale)

        # ------ 5. Order skeleton points ------
        ordered = self._order_skeleton_points(skeleton)
        if len(ordered) < 3:
            logger.warning("Too few ordered skeleton points (%d).", len(ordered))
            return self._empty_measurement(active_scale)

        # ------ 6 & 7. Sample perpendicular widths ------
        raw_measurements: List[Tuple[float, float, float]] = []
        half_win = self.neighborhood_size // 2

        for i in range(0, len(ordered), self.sample_spacing):
            pt = ordered[i]
            angle = self._compute_local_orientation(ordered, i, self.neighborhood_size)
            width_px, _e1, _e2 = self._measure_perpendicular_width(
                pt, angle, refined
            )
            if width_px > 0:
                raw_measurements.append((float(pt[0]), float(pt[1]), width_px))

        if not raw_measurements:
            logger.warning("No valid width measurements obtained.")
            return self._empty_measurement(active_scale)

        # ------ 8. Statistical aggregation ------
        widths = np.array([m[2] for m in raw_measurements], dtype=np.float64)

        # Outlier removal: discard > 3 × median
        median_w = float(np.median(widths))
        if median_w > 0:
            inlier_mask = widths <= 3.0 * median_w
            if inlier_mask.sum() > 0:
                raw_measurements = [
                    m for m, keep in zip(raw_measurements, inlier_mask) if keep
                ]
                widths = widths[inlier_mask]

        mean_w = float(np.mean(widths))
        median_w = float(np.median(widths))
        max_w = float(np.max(widths))
        min_w = float(np.min(widths))
        std_w = float(np.std(widths))
        p95_w = float(np.percentile(widths, 95))

        # ------ 9. Pixel → mm conversion ------
        width_mm: Optional[float] = None
        gsd: Optional[float] = None
        scale_factor: Optional[float] = None
        method = MeasurementMethod.RELATIVE_ONLY

        if active_scale is not None and active_scale.gsd_mm_per_px is not None:
            gsd = active_scale.gsd_mm_per_px
            scale_factor = gsd
            width_mm = median_w * gsd
            method = active_scale.method
        elif active_scale is not None:
            method = active_scale.method

        # ------ 10. Build result ------
        measurement = WidthMeasurement(
            width_px=median_w,
            width_mm=width_mm,
            method=method,
            measurement_points=raw_measurements,
            mean_width_px=mean_w,
            median_width_px=median_w,
            max_width_px=max_w,
            min_width_px=min_w,
            std_width_px=std_w,
            percentile_95_width_px=p95_w,
            gsd_mm_per_px=gsd,
            scale_factor=scale_factor,
        )

        logger.info(
            "Width estimation complete: median=%.2f px (%.3f mm), "
            "%d sample(s).",
            median_w,
            width_mm if width_mm is not None else float("nan"),
            len(raw_measurements),
        )
        return measurement

    # ==================================================================
    # Skeleton operations
    # ==================================================================

    @staticmethod
    def _skeletonize(mask: np.ndarray) -> np.ndarray:
        """Zhang–Suen thinning via scikit-image.

        Parameters
        ----------
        mask:
            Binary mask ``(H, W)`` with dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Skeleton image (bool cast to uint8).
        """
        skel = skeletonize(mask > 0)
        return skel.astype(np.uint8)

    def _prune_skeleton(
        self,
        skeleton: np.ndarray,
        min_length: int,
    ) -> np.ndarray:
        """Remove short branches from the skeleton.

        Strategy
        --------
        Iteratively identify *endpoint* pixels (pixels with exactly one
        8-connected skeleton neighbour).  For each endpoint, walk along
        the branch counting pixels until a junction (≥ 3 neighbours) is
        reached.  If the branch is shorter than ``min_length``, erase
        those pixels.  Repeat until no short branches remain.

        Parameters
        ----------
        skeleton:
            Binary skeleton ``(H, W)``, dtype ``uint8``.
        min_length:
            Branches shorter than this are removed.

        Returns
        -------
        np.ndarray
            Pruned skeleton.
        """
        skel = skeleton.copy()
        changed = True
        max_iterations = 50  # safety cap

        for _ in range(max_iterations):
            if not changed:
                break
            changed = False

            endpoints = self._find_endpoints(skel)
            if len(endpoints) == 0:
                break

            for ey, ex in endpoints:
                if skel[ey, ex] == 0:
                    continue  # already pruned in this pass

                branch = self._trace_branch(skel, ey, ex)
                if len(branch) < min_length:
                    for by, bx in branch:
                        skel[by, bx] = 0
                    changed = True

        return skel

    @staticmethod
    def _find_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Return coordinates of skeleton endpoints (1-neighbour pixels).

        Parameters
        ----------
        skeleton:
            Binary skeleton ``(H, W)``.

        Returns
        -------
        List[Tuple[int, int]]
            ``(row, col)`` for each endpoint.
        """
        # Convolve with a 3×3 all-ones kernel, then an endpoint has
        # kernel_sum == 2 (itself + 1 neighbour) where skeleton == 1.
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbour_count = cv2.filter2D(
            skeleton, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT
        )
        # Mask: skeleton pixel with exactly 2 in the sum (itself + 1 nbr)
        ep_mask = (skeleton == 1) & (neighbour_count == 2)
        ys, xs = np.nonzero(ep_mask)
        return list(zip(ys.tolist(), xs.tolist()))

    @staticmethod
    def _trace_branch(
        skeleton: np.ndarray,
        start_y: int,
        start_x: int,
    ) -> List[Tuple[int, int]]:
        """Walk along a skeleton branch from an endpoint to a junction.

        Parameters
        ----------
        skeleton:
            Binary skeleton ``(H, W)``.
        start_y, start_x:
            Starting endpoint coordinates.

        Returns
        -------
        List[Tuple[int, int]]
            Ordered ``(row, col)`` pixels comprising the branch (including
            the start, excluding the junction itself).
        """
        h, w = skeleton.shape
        visited = set()
        branch: List[Tuple[int, int]] = []
        cy, cx = start_y, start_x

        while True:
            visited.add((cy, cx))
            branch.append((cy, cx))

            # Find unvisited skeleton neighbours
            neighbours: List[Tuple[int, int]] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if skeleton[ny, nx] == 1 and (ny, nx) not in visited:
                            neighbours.append((ny, nx))

            if len(neighbours) == 0:
                # Dead end
                break
            elif len(neighbours) == 1:
                # Continue along the branch
                cy, cx = neighbours[0]
            else:
                # Junction reached — stop (don't include the junction)
                break

        return branch

    # ==================================================================
    # Skeleton ordering
    # ==================================================================

    def _order_skeleton_points(
        self,
        skeleton: np.ndarray,
    ) -> np.ndarray:
        """Order skeleton pixels along the crack path.

        Uses a greedy nearest-neighbour chain starting from an endpoint
        (or an arbitrary skeleton pixel if no endpoint exists).

        Parameters
        ----------
        skeleton:
            Binary skeleton ``(H, W)``.

        Returns
        -------
        np.ndarray
            ``(N, 2)`` array of ``(x, y)`` coordinates in traversal order.
        """
        ys, xs = np.nonzero(skeleton)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float64)

        coords = np.column_stack([xs, ys])  # (N, 2) as (x, y)

        # Pick an endpoint as the start (or first pixel otherwise)
        endpoints = self._find_endpoints(skeleton)
        if endpoints:
            start_y, start_x = endpoints[0]
            # Find the index in coords closest to this endpoint
            dists = np.abs(coords[:, 0] - start_x) + np.abs(coords[:, 1] - start_y)
            start_idx = int(np.argmin(dists))
        else:
            start_idx = 0

        # Greedy nearest-neighbour traversal
        n = len(coords)
        visited = np.zeros(n, dtype=bool)
        order = np.empty((n, 2), dtype=np.float64)

        current = start_idx
        for i in range(n):
            order[i] = coords[current]
            visited[current] = True
            if i == n - 1:
                break

            # Distances from current to all unvisited
            diff = coords[~visited] - coords[current]
            sq_dists = diff[:, 0] ** 2 + diff[:, 1] ** 2

            # Map back to global index
            unvisited_indices = np.where(~visited)[0]
            nearest_local = int(np.argmin(sq_dists))
            current = unvisited_indices[nearest_local]

        return order

    # ==================================================================
    # Orientation & width measurement
    # ==================================================================

    @staticmethod
    def _compute_local_orientation(
        points: np.ndarray,
        idx: int,
        window_size: int,
    ) -> float:
        """PCA on a local neighbourhood of skeleton points.

        Parameters
        ----------
        points:
            ``(N, 2)`` ordered skeleton coordinates ``(x, y)``.
        idx:
            Index of the query point.
        window_size:
            Number of points in the window (centred on ``idx``).

        Returns
        -------
        float
            Crack direction angle in **radians**.
        """
        half = window_size // 2
        lo = max(0, idx - half)
        hi = min(len(points), idx + half + 1)
        window = points[lo:hi]

        if len(window) < 2:
            return 0.0

        centered = window - window.mean(axis=0)
        cov = np.cov(centered, rowvar=False)

        # Handle degenerate (1-D) covariance
        if cov.ndim < 2:
            return 0.0

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = eigenvectors[:, -1]  # largest eigenvalue
        angle = float(np.arctan2(principal[1], principal[0]))
        return angle

    def _measure_perpendicular_width(
        self,
        point: np.ndarray,
        angle: float,
        mask: np.ndarray,
    ) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
        """Cast rays perpendicular to the crack at *point*.

        Parameters
        ----------
        point:
            ``(x, y)`` of the skeleton sample point.
        angle:
            Local crack direction in radians.
        mask:
            Refined binary mask ``(H, W)``.

        Returns
        -------
        Tuple[float, Tuple[int, int], Tuple[int, int]]
            ``(width_px, edge_point_positive, edge_point_negative)``.
            Width is the Euclidean distance between the two edge crossings.
            If measurement fails, width is 0.
        """
        perp_angle = angle + np.pi / 2.0
        direction = np.array([np.cos(perp_angle), np.sin(perp_angle)])

        origin = (0, 0)

        edge_pos = self._cast_ray(point, direction, mask)
        edge_neg = self._cast_ray(point, -direction, mask)

        if edge_pos is None or edge_neg is None:
            return 0.0, origin, origin

        width = float(
            np.sqrt(
                (edge_pos[0] - edge_neg[0]) ** 2
                + (edge_pos[1] - edge_neg[1]) ** 2
            )
        )
        return width, edge_pos, edge_neg

    @staticmethod
    def _cast_ray(
        start: np.ndarray,
        direction: np.ndarray,
        mask: np.ndarray,
        max_distance: int = 200,
    ) -> Optional[Tuple[int, int]]:
        """Step along a direction vector until leaving the mask.

        Parameters
        ----------
        start:
            ``(x, y)`` origin (float).
        direction:
            Unit direction vector ``(dx, dy)``.
        mask:
            Binary mask ``(H, W)``.
        max_distance:
            Maximum number of pixel steps.

        Returns
        -------
        Optional[Tuple[int, int]]
            ``(x, y)`` of the last in-mask pixel along the ray, or
            ``None`` if the start itself is out of bounds / outside the
            mask.
        """
        h, w = mask.shape
        x, y = float(start[0]), float(start[1])

        last_inside: Optional[Tuple[int, int]] = None

        for step in range(max_distance):
            ix, iy = int(round(x)), int(round(y))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                break

            if mask[iy, ix] > 0:
                last_inside = (ix, iy)
            else:
                # We just left the mask — the previous point was the edge
                if last_inside is not None:
                    return last_inside
                else:
                    # Started outside the mask
                    return None

            x += direction[0]
            y += direction[1]

        # Reached max_distance while still inside — return last seen point
        return last_inside

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _empty_measurement(
        scale_info: Optional[ScaleInfo] = None,
    ) -> WidthMeasurement:
        """Return a zeroed-out :class:`WidthMeasurement`."""
        method = (
            scale_info.method
            if scale_info is not None
            else MeasurementMethod.RELATIVE_ONLY
        )
        return WidthMeasurement(
            width_px=0.0,
            width_mm=None,
            method=method,
            measurement_points=[],
            mean_width_px=0.0,
            median_width_px=0.0,
            max_width_px=0.0,
            min_width_px=0.0,
            std_width_px=0.0,
            percentile_95_width_px=0.0,
            gsd_mm_per_px=(
                scale_info.gsd_mm_per_px if scale_info is not None else None
            ),
            scale_factor=None,
        )
