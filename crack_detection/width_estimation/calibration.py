"""
Camera Calibration
==================
Intrinsic calibration, GSD computation from drone altitude, and
reference-object-based scale estimation (ArUco markers, known-width
bounding boxes, and checkerboard calibration).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from crack_detection.schemas import MeasurementMethod
from crack_detection.width_estimation.common import ScaleInfo, compute_gsd

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """Derive pixel-to-mm scaling from camera intrinsics or reference objects.

    Parameters
    ----------
    focal_length_mm:
        Physical focal length of the camera lens (mm).
    sensor_width_mm:
        Physical width of the camera sensor (mm).
    image_width_px:
        Width of the captured image in pixels.
    """

    def __init__(
        self,
        focal_length_mm: Optional[float] = None,
        sensor_width_mm: Optional[float] = None,
        image_width_px: Optional[int] = None,
    ) -> None:
        self.focal_length_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm
        self.image_width_px = image_width_px

        # Internal state updated by the last calibration call
        self._scale_info: ScaleInfo = ScaleInfo()

        # Camera matrix & distortion (populated by checkerboard calibration)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Altitude-based GSD
    # ------------------------------------------------------------------

    def calibrate_from_altitude(self, altitude_m: float) -> ScaleInfo:
        """Compute GSD from drone altitude and stored camera intrinsics.

        Parameters
        ----------
        altitude_m:
            Height above the inspection surface in metres.

        Returns
        -------
        ScaleInfo
            Populated with ``gsd_mm_per_px`` and ``method = MONOCULAR_GSD``.

        Raises
        ------
        ValueError
            If camera intrinsics have not been set.
        """
        if (
            self.focal_length_mm is None
            or self.sensor_width_mm is None
            or self.image_width_px is None
        ):
            raise ValueError(
                "Camera intrinsics (focal_length_mm, sensor_width_mm, "
                "image_width_px) must be set before calling "
                "calibrate_from_altitude."
            )

        gsd = compute_gsd(
            altitude_m=altitude_m,
            focal_length_mm=self.focal_length_mm,
            sensor_width_mm=self.sensor_width_mm,
            image_width_px=float(self.image_width_px),
        )

        # Also compute focal length in pixels for downstream use
        focal_length_px = (
            self.focal_length_mm * self.image_width_px / self.sensor_width_mm
        )

        self._scale_info = ScaleInfo(
            gsd_mm_per_px=gsd,
            method=MeasurementMethod.MONOCULAR_GSD,
            distance_mm=altitude_m * 1000.0,
            focal_length_px=focal_length_px,
        )
        logger.info(
            "Altitude calibration complete: GSD=%.4f mm/px at %.1f m.",
            gsd,
            altitude_m,
        )
        return self._scale_info

    # ------------------------------------------------------------------
    # Reference-object based
    # ------------------------------------------------------------------

    def calibrate_from_reference(
        self,
        image: np.ndarray,
        known_width_mm: float,
        ref_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> ScaleInfo:
        """Derive scale from a reference object of known size.

        The method first attempts ArUco-marker detection.  If no markers
        are found (or ``ref_bbox`` is provided), it falls back to using
        the supplied bounding box directly.

        Parameters
        ----------
        image:
            BGR image containing the reference object.
        known_width_mm:
            True physical width of the reference object in mm.
        ref_bbox:
            Optional ``(x1, y1, x2, y2)`` bounding box of the reference
            object.  Skips ArUco detection when supplied.

        Returns
        -------
        ScaleInfo

        Raises
        ------
        ValueError
            If no reference could be found in the image.
        """
        marker_width_px: Optional[float] = None

        if ref_bbox is None:
            # Try ArUco detection
            markers = self.detect_aruco_markers(image)
            if markers:
                corners, marker_id = markers[0]
                # Marker width = distance between top-left and top-right
                tl, tr = corners[0], corners[1]
                marker_width_px = float(np.linalg.norm(tr - tl))
                logger.info(
                    "ArUco marker %d detected — width %.1f px.",
                    marker_id,
                    marker_width_px,
                )
            else:
                raise ValueError(
                    "No ArUco markers detected and no ref_bbox provided."
                )
        else:
            x1, y1, x2, y2 = ref_bbox
            marker_width_px = float(x2 - x1)
            logger.info(
                "Using provided ref_bbox — reference width %.1f px.",
                marker_width_px,
            )

        if marker_width_px is None or marker_width_px <= 0:
            raise ValueError(
                "Reference-object width in pixels is invalid."
            )

        gsd = known_width_mm / marker_width_px

        self._scale_info = ScaleInfo(
            gsd_mm_per_px=gsd,
            method=MeasurementMethod.MONOCULAR_REFERENCE,
        )
        logger.info(
            "Reference calibration complete: scale=%.5f mm/px.", gsd
        )
        return self._scale_info

    # ------------------------------------------------------------------
    # Checkerboard calibration
    # ------------------------------------------------------------------

    def calibrate_from_checkerboard(
        self,
        images: Sequence[np.ndarray],
        pattern_size: Tuple[int, int],
        square_size_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full camera calibration using a set of checkerboard images.

        Parameters
        ----------
        images:
            List of BGR images containing views of the checkerboard.
        pattern_size:
            Inner corner count as ``(cols, rows)``.
        square_size_mm:
            Physical side-length of each checkerboard square in mm.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(camera_matrix, dist_coeffs)``

        Raises
        ------
        RuntimeError
            If calibration fails (too few valid images).
        """
        obj_p = np.zeros(
            (pattern_size[0] * pattern_size[1], 3), dtype=np.float32
        )
        obj_p[:, :2] = (
            np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]]
            .T.reshape(-1, 2)
            * square_size_mm
        )

        obj_points: List[np.ndarray] = []
        img_points: List[np.ndarray] = []
        img_size: Optional[Tuple[int, int]] = None

        for idx, img in enumerate(images):
            gray = (
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if len(img.shape) == 3
                else img
            )
            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if found:
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )
                obj_points.append(obj_p)
                img_points.append(corners_refined)
                logger.debug("Checkerboard found in image %d.", idx)
            else:
                logger.debug("Checkerboard NOT found in image %d.", idx)

        if len(obj_points) < 3:
            raise RuntimeError(
                f"Checkerboard calibration requires ≥ 3 valid images, "
                f"got {len(obj_points)}."
            )

        ret, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None  # type: ignore[arg-type]
        )
        if not ret:
            raise RuntimeError("cv2.calibrateCamera returned failure.")

        self.camera_matrix = mtx
        self.dist_coeffs = dist
        logger.info(
            "Checkerboard calibration complete (RMS=%.3f). "
            "fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f",
            ret,
            mtx[0, 0],
            mtx[1, 1],
            mtx[0, 2],
            mtx[1, 2],
        )
        return mtx, dist

    # ------------------------------------------------------------------
    # ArUco helpers
    # ------------------------------------------------------------------

    @staticmethod
    def detect_aruco_markers(
        image: np.ndarray,
        dictionary: int = cv2.aruco.DICT_4X4_50,
    ) -> List[Tuple[np.ndarray, int]]:
        """Detect ArUco markers in an image.

        Parameters
        ----------
        image:
            BGR or grayscale image.
        dictionary:
            OpenCV ArUco dictionary constant.

        Returns
        -------
        List[Tuple[np.ndarray, int]]
            Each tuple is ``(corners, marker_id)`` where ``corners`` is
            a ``(4, 2)`` float array of the four corner coordinates.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )

        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()

        try:
            # OpenCV 4.7+
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners_list, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Fallback for older OpenCV
            corners_list, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=params
            )

        results: List[Tuple[np.ndarray, int]] = []
        if ids is not None:
            for corners, marker_id in zip(corners_list, ids.flatten()):
                results.append((corners[0], int(marker_id)))

        logger.info("Detected %d ArUco marker(s).", len(results))
        return results

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------

    def get_scale_info(self) -> ScaleInfo:
        """Return the most recently computed :class:`ScaleInfo`."""
        return self._scale_info
