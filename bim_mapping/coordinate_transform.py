"""
Coordinate Transformation Module
=================================
Handles transformations between image space, camera space, world space,
and IFC local space for mapping crack detections onto BIM models.

Supports:
- Pinhole camera model (image ↔ camera frame)
- Rigid body transforms (camera ↔ world)
- IFC local coordinate alignment (world ↔ IFC)
- Manual point-pair homography / affine mapping
- Marker-based SVD rigid transform estimation
"""

import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV (cv2) not installed. Manual 2D homography mapping unavailable.")


class CoordinateTransformer:
    """Transform coordinates between image, camera, world, and IFC spaces.

    The standard pipeline is::

        image (u, v, depth) → camera 3-D → world 3-D → IFC local 3-D

    Each stage can also be used independently.

    Parameters
    ----------
    camera_matrix : np.ndarray, optional
        3×3 camera intrinsic matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
    dist_coeffs : np.ndarray, optional
        Distortion coefficients compatible with OpenCV (k1, k2, p1, p2 [, k3 …]).
    """

    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> None:
        self.camera_matrix = (
            np.asarray(camera_matrix, dtype=np.float64)
            if camera_matrix is not None
            else None
        )
        self.dist_coeffs = (
            np.asarray(dist_coeffs, dtype=np.float64)
            if dist_coeffs is not None
            else None
        )
        logger.info(
            "CoordinateTransformer initialised (intrinsics %s).",
            "provided" if self.camera_matrix is not None else "not set",
        )

    # ------------------------------------------------------------------
    # Image → Camera
    # ------------------------------------------------------------------
    def image_to_camera(
        self,
        u: float,
        v: float,
        depth: float,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert a pixel coordinate + depth to a 3-D point in the camera frame.

        Uses the standard pinhole model::

            X = (u - cx) * depth / fx
            Y = (v - cy) * depth / fy
            Z = depth

        Parameters
        ----------
        u, v : float
            Pixel coordinates (column, row).
        depth : float
            Depth value in the same unit as the desired output (typically metres).
        camera_matrix : np.ndarray, optional
            3×3 intrinsic matrix.  Falls back to ``self.camera_matrix``.

        Returns
        -------
        np.ndarray
            Camera-frame 3-D point ``[X, Y, Z]``.

        Raises
        ------
        ValueError
            If no camera matrix is available.
        """
        K = camera_matrix if camera_matrix is not None else self.camera_matrix
        if K is None:
            raise ValueError(
                "Camera intrinsic matrix required but not provided."
            )
        K = np.asarray(K, dtype=np.float64)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = float(depth)

        point_camera = np.array([X, Y, Z], dtype=np.float64)
        logger.debug(
            "image_to_camera: (u=%.1f, v=%.1f, d=%.3f) → %s",
            u, v, depth, point_camera,
        )
        return point_camera

    # ------------------------------------------------------------------
    # Camera → World
    # ------------------------------------------------------------------
    def camera_to_world(
        self,
        point_camera: np.ndarray,
        extrinsic_matrix: np.ndarray,
    ) -> np.ndarray:
        """Transform a camera-frame 3-D point to world coordinates.

        Parameters
        ----------
        point_camera : np.ndarray
            ``[X, Y, Z]`` in the camera frame.
        extrinsic_matrix : np.ndarray
            4×4 homogeneous transformation ``[R | t; 0 0 0 1]`` that maps
            camera coordinates to world coordinates.

        Returns
        -------
        np.ndarray
            ``[X, Y, Z]`` in the world frame.
        """
        point_camera = np.asarray(point_camera, dtype=np.float64).ravel()
        E = np.asarray(extrinsic_matrix, dtype=np.float64)

        if E.shape != (4, 4):
            raise ValueError(
                f"Extrinsic matrix must be 4×4, got {E.shape}."
            )

        R = E[:3, :3]
        t = E[:3, 3]
        point_world = R @ point_camera + t

        logger.debug("camera_to_world: %s → %s", point_camera, point_world)
        return point_world

    # ------------------------------------------------------------------
    # World → IFC
    # ------------------------------------------------------------------
    def world_to_ifc(
        self,
        point_world: np.ndarray,
        ifc_transform_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform world coordinates to IFC local coordinate system.

        If no transform matrix is supplied the world frame is assumed to
        coincide with the IFC frame (identity transform).

        Parameters
        ----------
        point_world : np.ndarray
            ``[X, Y, Z]`` in the world frame.
        ifc_transform_matrix : np.ndarray, optional
            4×4 homogeneous transform from world to IFC local space.

        Returns
        -------
        np.ndarray
            ``[X, Y, Z]`` in the IFC local frame.
        """
        point_world = np.asarray(point_world, dtype=np.float64).ravel()

        if ifc_transform_matrix is None:
            logger.debug(
                "world_to_ifc: no IFC transform – using identity."
            )
            return point_world.copy()

        M = np.asarray(ifc_transform_matrix, dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError(
                f"IFC transform matrix must be 4×4, got {M.shape}."
            )

        R = M[:3, :3]
        t = M[:3, 3]
        point_ifc = R @ point_world + t

        logger.debug("world_to_ifc: %s → %s", point_world, point_ifc)
        return point_ifc

    # ------------------------------------------------------------------
    # Full chain: Image → IFC
    # ------------------------------------------------------------------
    def image_to_ifc(
        self,
        u: float,
        v: float,
        depth: float,
        extrinsic_matrix: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        ifc_transform: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Full pipeline: image pixel → camera → world → IFC local.

        Parameters
        ----------
        u, v : float
            Pixel coordinates.
        depth : float
            Depth value.
        extrinsic_matrix : np.ndarray
            4×4 camera-to-world transform.
        camera_matrix : np.ndarray, optional
            3×3 intrinsic matrix (falls back to stored).
        ifc_transform : np.ndarray, optional
            4×4 world-to-IFC transform (identity if *None*).

        Returns
        -------
        np.ndarray
            ``[X, Y, Z]`` in the IFC local frame.
        """
        pt_cam = self.image_to_camera(u, v, depth, camera_matrix)
        pt_world = self.camera_to_world(pt_cam, extrinsic_matrix)
        pt_ifc = self.world_to_ifc(pt_world, ifc_transform)
        logger.info(
            "image_to_ifc: (u=%.1f, v=%.1f, d=%.3f) → IFC %s",
            u, v, depth, pt_ifc,
        )
        return pt_ifc

    # ------------------------------------------------------------------
    # Manual mapping (homography / affine)
    # ------------------------------------------------------------------
    def create_manual_mapping(
        self,
        image_points: np.ndarray,
        ifc_points: np.ndarray,
    ) -> np.ndarray:
        """Compute a transformation from corresponding point pairs.

        * **2-D case** (Nx2 arrays, N ≥ 4): computes a homography via
          ``cv2.findHomography`` (RANSAC).
        * **3-D case** (Nx3 arrays, N ≥ 3): computes a rigid (Euclidean)
          transform using SVD-based least squares.

        Parameters
        ----------
        image_points : np.ndarray
            Nx2 or Nx3 source points (image / camera space).
        ifc_points : np.ndarray
            Nx2 or Nx3 destination points (IFC space).

        Returns
        -------
        np.ndarray
            Transformation matrix — 3×3 homography **or** 4×4 rigid
            transform depending on the dimensionality.

        Raises
        ------
        ValueError
            If point arrays are mismatched or too few correspondences.
        RuntimeError
            If OpenCV is required but not available.
        """
        src = np.asarray(image_points, dtype=np.float64)
        dst = np.asarray(ifc_points, dtype=np.float64)

        if src.shape != dst.shape:
            raise ValueError(
                f"Point arrays must match in shape: {src.shape} vs {dst.shape}."
            )

        ndim = src.shape[1] if src.ndim == 2 else 1

        # --- 2-D homography ---
        if ndim == 2:
            if src.shape[0] < 4:
                raise ValueError(
                    "At least 4 point pairs required for 2-D homography."
                )
            if not HAS_CV2:
                raise RuntimeError(
                    "OpenCV is required for 2-D homography computation."
                )
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                raise RuntimeError("Homography estimation failed.")
            inliers = int(mask.sum()) if mask is not None else src.shape[0]
            logger.info(
                "2-D homography computed (%d/%d inliers).",
                inliers, src.shape[0],
            )
            return H

        # --- 3-D rigid transform ---
        if ndim == 3:
            if src.shape[0] < 3:
                raise ValueError(
                    "At least 3 point pairs required for 3-D rigid transform."
                )
            return self._compute_rigid_transform(src, dst)

        raise ValueError(
            f"Points must be 2-D or 3-D, got dimensionality {ndim}."
        )

    def apply_manual_mapping(
        self,
        image_point: np.ndarray,
        transform_matrix: np.ndarray,
    ) -> np.ndarray:
        """Apply a pre-computed transform to a point.

        * If ``transform_matrix`` is 3×3, applies a 2-D homography.
        * If ``transform_matrix`` is 4×4, applies a 3-D rigid transform.

        Parameters
        ----------
        image_point : np.ndarray
            Source point (2-D or 3-D).
        transform_matrix : np.ndarray
            3×3 homography or 4×4 rigid transform.

        Returns
        -------
        np.ndarray
            Transformed point.
        """
        pt = np.asarray(image_point, dtype=np.float64).ravel()
        T = np.asarray(transform_matrix, dtype=np.float64)

        if T.shape == (3, 3):
            # 2-D homography
            pt_h = np.array([pt[0], pt[1], 1.0])
            result_h = T @ pt_h
            result = result_h[:2] / result_h[2]
            logger.debug("apply_manual_mapping 2-D: %s → %s", pt, result)
            return result

        if T.shape == (4, 4):
            # 3-D rigid
            R = T[:3, :3]
            t = T[:3, 3]
            result = R @ pt[:3] + t
            logger.debug("apply_manual_mapping 3-D: %s → %s", pt, result)
            return result

        raise ValueError(
            f"Transform matrix must be 3×3 or 4×4, got {T.shape}."
        )

    # ------------------------------------------------------------------
    # Marker-based SVD rigid transform
    # ------------------------------------------------------------------
    def estimate_transform_from_markers(
        self,
        marker_image_positions: np.ndarray,
        marker_ifc_positions: np.ndarray,
    ) -> np.ndarray:
        """Estimate the optimal rigid transform from marker correspondences.

        Uses SVD-based least-squares fitting (Arun *et al.*, 1987) to find
        the rotation ``R`` and translation ``t`` minimising::

            Σ ‖ R · p_i + t − q_i ‖²

        Parameters
        ----------
        marker_image_positions : np.ndarray
            Nx3 source marker positions (image / camera / world space).
        marker_ifc_positions : np.ndarray
            Nx3 corresponding marker positions in IFC space.

        Returns
        -------
        np.ndarray
            4×4 homogeneous rigid transformation matrix.
        """
        src = np.asarray(marker_image_positions, dtype=np.float64)
        dst = np.asarray(marker_ifc_positions, dtype=np.float64)

        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
            raise ValueError(
                "Marker positions must be Nx3 arrays with matching shapes."
            )
        if src.shape[0] < 3:
            raise ValueError(
                "At least 3 marker pairs required for rigid transform."
            )

        return self._compute_rigid_transform(src, dst)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_rigid_transform(
        src: np.ndarray,
        dst: np.ndarray,
    ) -> np.ndarray:
        """SVD-based rigid (Euclidean) transform estimation.

        Solves for ``R, t`` such that ``dst ≈ R @ src + t``.

        Parameters
        ----------
        src, dst : np.ndarray
            Nx3 corresponding point sets.

        Returns
        -------
        np.ndarray
            4×4 homogeneous transformation matrix.
        """
        centroid_src = src.mean(axis=0)
        centroid_dst = dst.mean(axis=0)

        src_centered = src - centroid_src
        dst_centered = dst - centroid_dst

        # Cross-covariance matrix
        H = src_centered.T @ dst_centered

        U, S, Vt = np.linalg.svd(H)
        V = Vt.T

        # Correct for reflection
        d = np.linalg.det(V @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
        R = V @ sign_matrix @ U.T

        t = centroid_dst - R @ centroid_src

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        residuals = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        logger.info(
            "Rigid transform computed: RMSE = %.6f (%d point pairs).",
            rmse, src.shape[0],
        )

        return T
