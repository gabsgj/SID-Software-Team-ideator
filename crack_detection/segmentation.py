"""
Mask / Segmentation Processing
==============================
Morphological refinement, connected-component analysis, orientation
estimation, contour extraction, and geometry helpers for binary crack masks.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


class MaskProcessor:
    """Utility class for post-processing binary crack masks."""

    # ------------------------------------------------------------------
    # Morphological refinement
    # ------------------------------------------------------------------

    @staticmethod
    def refine_mask(
        mask: np.ndarray,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Apply morphological closing then opening to clean a binary mask.

        Closing bridges small gaps inside the crack region; opening
        removes isolated noise pixels around it.

        Parameters
        ----------
        mask:
            Binary mask of shape ``(H, W)`` with values in {0, 1} or
            {0, 255}.
        kernel_size:
            Size of the structuring element (square).

        Returns
        -------
        np.ndarray
            Refined binary mask (values 0 or 1, dtype ``uint8``).
        """
        # Ensure binary {0, 1}
        binary = (mask > 0).astype(np.uint8)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        logger.debug(
            "refine_mask: %d → %d foreground pixels.",
            int(binary.sum()),
            int(opened.sum()),
        )
        return opened

    # ------------------------------------------------------------------
    # Connected-component analysis
    # ------------------------------------------------------------------

    @staticmethod
    def extract_connected_components(
        mask: np.ndarray,
        min_area: int = 50,
    ) -> List[np.ndarray]:
        """Label connected regions and return individual masks.

        Parameters
        ----------
        mask:
            Binary mask ``(H, W)``.
        min_area:
            Minimum area (in pixels) for a component to be kept.

        Returns
        -------
        List[np.ndarray]
            List of binary masks, one per qualifying connected component.
        """
        binary = (mask > 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

        components: List[np.ndarray] = []
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8)
            area = int(component.sum())
            if area >= min_area:
                components.append(component)
            else:
                logger.debug(
                    "Discarding component %d (area=%d < %d).",
                    label_id,
                    area,
                    min_area,
                )

        logger.info(
            "extract_connected_components: kept %d / %d component(s).",
            len(components),
            num_labels - 1,
        )
        return components

    # ------------------------------------------------------------------
    # Orientation (PCA)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_orientation(mask: np.ndarray) -> float:
        """Compute the dominant orientation of a crack via PCA.

        Parameters
        ----------
        mask:
            Binary mask ``(H, W)``.

        Returns
        -------
        float
            Angle in degrees in the range [0, 180).
        """
        ys, xs = np.nonzero(mask)
        if len(xs) < 2:
            return 0.0

        coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        mean = coords.mean(axis=0)
        centered = coords - mean

        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Principal eigenvector (largest eigenvalue is last after eigh)
        principal = eigenvectors[:, -1]
        angle_rad = np.arctan2(principal[1], principal[0])
        angle_deg = float(np.degrees(angle_rad)) % 180.0

        return angle_deg

    # ------------------------------------------------------------------
    # Contour extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_contour_polygon(
        mask: np.ndarray,
        simplify_epsilon: float = 2.0,
    ) -> Optional[List[Tuple[float, float]]]:
        """Return a simplified contour polygon for the mask.

        Parameters
        ----------
        mask:
            Binary mask ``(H, W)``.
        simplify_epsilon:
            Douglas–Peucker approximation tolerance (pixels).

        Returns
        -------
        Optional[List[Tuple[float, float]]]
            List of ``(x, y)`` vertices, or ``None`` if no contour found.
        """
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, simplify_epsilon, closed=True)
        return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]

    # ------------------------------------------------------------------
    # Area & length
    # ------------------------------------------------------------------

    @staticmethod
    def compute_crack_area(mask: np.ndarray) -> float:
        """Return the total foreground area in pixels.

        Parameters
        ----------
        mask:
            Binary mask ``(H, W)``.

        Returns
        -------
        float
            Number of non-zero pixels.
        """
        return float(np.count_nonzero(mask))

    @staticmethod
    def compute_crack_length(skeleton: np.ndarray) -> float:
        """Estimate crack length by counting skeleton pixels.

        A more accurate estimation could trace the skeleton path and sum
        Euclidean segment lengths, but pixel count is a fast and
        reasonable approximation (typically within ~5 % for smooth cracks).

        Parameters
        ----------
        skeleton:
            Binary skeleton image ``(H, W)`` (1-pixel-wide).

        Returns
        -------
        float
            Length estimate in pixels.
        """
        return float(np.count_nonzero(skeleton))
