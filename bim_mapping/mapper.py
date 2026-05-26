"""
Crack → BIM Element Mapper
===========================
Maps crack detections to specific IFC structural elements by spatial proximity.

Supports IFC 2×3, IFC 4, and IFC 4.3 schemas.  Bridge-specific element types
(``IfcBridgePart``) are used when available (IFC 4.3); the mapper falls back
to generic structural types for older schemas.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import ifcopenshell
    import ifcopenshell.geom
    HAS_IFC = True
except ImportError:
    HAS_IFC = False
    logger.warning(
        "ifcopenshell not installed. BIM mapping will be limited."
    )

# Structural element types to extract from the IFC model.
_STRUCTURAL_TYPES: Tuple[str, ...] = (
    "IfcBeam",
    "IfcColumn",
    "IfcSlab",
    "IfcWall",
    "IfcMember",
    "IfcPlate",
    "IfcFooting",
)

# Bridge-specific types available in IFC 4.3+
_BRIDGE_TYPES: Tuple[str, ...] = (
    "IfcBridgePart",
    "IfcBearing",
    "IfcDeepFoundation",
    "IfcCaissonFoundation",
)


class CrackBIMMapper:
    """Map detected cracks to IFC structural elements.

    Elements are looked up by spatial proximity — the crack's 3-D position
    (in IFC-local coordinates) is compared against every cached element's
    axis-aligned bounding box.

    Parameters
    ----------
    ifc_path : str, optional
        Path to an IFC model file.  The model is loaded immediately if
        provided.
    """

    def __init__(self, ifc_path: Optional[str] = None) -> None:
        self.ifc_model = None
        self.ifc_path: Optional[str] = None
        self.elements: Dict[str, Dict] = {}
        self._schema_version: Optional[str] = None

        if ifc_path is not None:
            self.load_model(ifc_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self, ifc_path: str) -> None:
        """Open an IFC file and extract all structural elements.

        Parameters
        ----------
        ifc_path : str
            Filesystem path to the ``.ifc`` file.

        Raises
        ------
        RuntimeError
            If ``ifcopenshell`` is not installed.
        FileNotFoundError
            If *ifc_path* does not exist (raised by ifcopenshell).
        """
        if not HAS_IFC:
            raise RuntimeError(
                "ifcopenshell is required to load IFC models. "
                "Install it with: pip install ifcopenshell"
            )

        logger.info("Loading IFC model: %s", ifc_path)
        self.ifc_model = ifcopenshell.open(ifc_path)
        self.ifc_path = ifc_path
        self._schema_version = self.ifc_model.schema

        logger.info(
            "IFC schema version: %s", self._schema_version
        )

        self._extract_all_elements()

    def _extract_all_elements(self) -> None:
        """Parse and cache every structural element in the loaded model."""
        if self.ifc_model is None:
            return

        self.elements.clear()

        # Standard structural types
        type_list = list(_STRUCTURAL_TYPES)

        # Try bridge-specific types (IFC 4.3)
        for btype in _BRIDGE_TYPES:
            try:
                if self.ifc_model.by_type(btype):
                    type_list.append(btype)
            except Exception:
                logger.debug(
                    "Element type %s not available in schema %s.",
                    btype, self._schema_version,
                )

        for type_name in type_list:
            try:
                elements = self.ifc_model.by_type(type_name)
            except Exception:
                logger.debug(
                    "Type %s not found in schema.", type_name,
                )
                continue

            for elem in elements:
                global_id = elem.GlobalId
                bbox_info = self._extract_element_bbox(elem)
                entry = {
                    "GlobalId": global_id,
                    "Name": getattr(elem, "Name", None) or "Unnamed",
                    "Type": type_name,
                    "bbox": bbox_info,
                    "centroid": (
                        bbox_info["centroid"] if bbox_info else None
                    ),
                }
                self.elements[global_id] = entry

        logger.info(
            "Extracted %d structural elements from IFC model.",
            len(self.elements),
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _extract_element_bbox(self, element) -> Optional[Dict]:
        """Compute an axis-aligned bounding box for *element*.

        Parameters
        ----------
        element
            An ifcopenshell entity.

        Returns
        -------
        dict or None
            ``{'min': [x,y,z], 'max': [x,y,z], 'centroid': [x,y,z]}``
            or ``None`` if geometry processing fails.
        """
        if not HAS_IFC:
            return None

        try:
            settings = ifcopenshell.geom.settings()
            shape = ifcopenshell.geom.create_shape(settings, element)
            verts = shape.geometry.verts
            # verts is a flat list [x0, y0, z0, x1, y1, z1, ...]
            coords = np.array(verts, dtype=np.float64).reshape(-1, 3)
            bbox_min = coords.min(axis=0).tolist()
            bbox_max = coords.max(axis=0).tolist()
            centroid = coords.mean(axis=0).tolist()
            return {
                "min": bbox_min,
                "max": bbox_max,
                "centroid": centroid,
            }
        except Exception as exc:
            logger.warning(
                "Failed to extract bbox for element %s (%s): %s",
                getattr(element, "GlobalId", "?"),
                getattr(element, "Name", "?"),
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------
    @staticmethod
    def _distance_to_bbox(
        point: np.ndarray,
        bbox: Dict,
    ) -> float:
        """Compute the Euclidean distance from *point* to an AABB.

        Returns 0 if the point is inside the box.
        """
        p = np.asarray(point, dtype=np.float64)
        bmin = np.asarray(bbox["min"], dtype=np.float64)
        bmax = np.asarray(bbox["max"], dtype=np.float64)

        # Per-axis clamped distance
        clamped = np.maximum(bmin - p, 0.0) + np.maximum(p - bmax, 0.0)
        return float(np.linalg.norm(clamped))

    def find_nearest_element(
        self,
        point_3d: np.ndarray,
    ) -> Optional[Dict]:
        """Find the structural element closest to a 3-D point.

        Parameters
        ----------
        point_3d : array-like
            ``[X, Y, Z]`` in IFC-local coordinates.

        Returns
        -------
        dict or None
            Element info with keys ``GlobalId``, ``Name``, ``Type``,
            ``distance``.  Returns ``None`` if no elements are loaded.
        """
        if not self.elements:
            logger.warning(
                "No structural elements loaded — cannot find nearest."
            )
            return None

        point_3d = np.asarray(point_3d, dtype=np.float64)
        best_dist = float("inf")
        best_elem = None

        for gid, info in self.elements.items():
            bbox = info.get("bbox")
            if bbox is None:
                continue
            dist = self._distance_to_bbox(point_3d, bbox)
            if dist < best_dist:
                best_dist = dist
                best_elem = info

        if best_elem is None:
            logger.warning(
                "No element with valid geometry found near %s.", point_3d,
            )
            return None

        return {
            "GlobalId": best_elem["GlobalId"],
            "Name": best_elem["Name"],
            "Type": best_elem["Type"],
            "distance": best_dist,
        }

    # ------------------------------------------------------------------
    # Crack-to-element mapping
    # ------------------------------------------------------------------
    def map_crack_to_element(
        self,
        crack_detection: Dict,
        point_3d: Optional[np.ndarray] = None,
        element_id: Optional[str] = None,
    ) -> Dict:
        """Map a single crack detection to a structural element.

        Priority:
        1. If *element_id* is provided the crack is directly assigned.
        2. If *point_3d* is provided the nearest element is found.
        3. Otherwise a mapping with ``element_id=None`` is returned.

        Parameters
        ----------
        crack_detection : dict
            Detection record (must include ``detection_id``).
        point_3d : np.ndarray, optional
            Crack location in IFC-local 3-D space.
        element_id : str, optional
            Directly assigned IFC element GlobalId.

        Returns
        -------
        dict
            Mapping record with keys: ``crack_id``, ``element_id``,
            ``element_name``, ``element_type``, ``distance``,
            ``confidence``, ``mapping_method``, ``timestamp``.
        """
        crack_id = crack_detection.get("detection_id", "unknown")
        mapping = {
            "crack_id": crack_id,
            "element_id": None,
            "element_name": None,
            "element_type": None,
            "distance": None,
            "confidence": crack_detection.get("confidence", 0.0),
            "mapping_method": "none",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Direct assignment
        if element_id is not None:
            elem_info = self.elements.get(element_id)
            if elem_info:
                mapping.update({
                    "element_id": element_id,
                    "element_name": elem_info["Name"],
                    "element_type": elem_info["Type"],
                    "distance": 0.0,
                    "mapping_method": "direct",
                })
                logger.info(
                    "Crack %s directly assigned to element %s.",
                    crack_id, element_id,
                )
            else:
                logger.warning(
                    "Element %s not found in model — "
                    "recording raw ID only.",
                    element_id,
                )
                mapping.update({
                    "element_id": element_id,
                    "mapping_method": "direct_unverified",
                })
            return mapping

        # Spatial proximity
        if point_3d is not None:
            nearest = self.find_nearest_element(point_3d)
            if nearest is not None:
                mapping.update({
                    "element_id": nearest["GlobalId"],
                    "element_name": nearest["Name"],
                    "element_type": nearest["Type"],
                    "distance": nearest["distance"],
                    "mapping_method": "spatial_proximity",
                })
                logger.info(
                    "Crack %s mapped to element %s (dist=%.4f).",
                    crack_id, nearest["GlobalId"], nearest["distance"],
                )
            return mapping

        logger.warning(
            "No point_3d or element_id given for crack %s.", crack_id,
        )
        return mapping

    def map_cracks_batch(
        self,
        crack_detections: List[Dict],
        points_3d: Optional[List[np.ndarray]] = None,
        element_ids: Optional[List[Optional[str]]] = None,
    ) -> List[Dict]:
        """Map multiple cracks to structural elements.

        Parameters
        ----------
        crack_detections : list[dict]
            List of detection records.
        points_3d : list[np.ndarray], optional
            Corresponding 3-D positions; ``None`` entries are skipped.
        element_ids : list[str | None], optional
            Corresponding element IDs for direct assignment.

        Returns
        -------
        list[dict]
            List of mapping records.
        """
        n = len(crack_detections)
        pts = points_3d or [None] * n
        eids = element_ids or [None] * n

        if len(pts) != n or len(eids) != n:
            raise ValueError(
                "All input lists must have the same length."
            )

        mappings: List[Dict] = []
        for det, pt, eid in zip(crack_detections, pts, eids):
            mappings.append(
                self.map_crack_to_element(det, point_3d=pt, element_id=eid)
            )

        logger.info("Batch mapped %d cracks.", len(mappings))
        return mappings

    # ------------------------------------------------------------------
    # Summaries & manual assignment
    # ------------------------------------------------------------------
    def get_element_summary(self) -> List[Dict]:
        """Return a summary list of all parsed structural elements.

        Returns
        -------
        list[dict]
            Each entry has ``GlobalId``, ``Name``, ``Type``, ``centroid``.
        """
        summaries = []
        for gid, info in self.elements.items():
            summaries.append({
                "GlobalId": gid,
                "Name": info["Name"],
                "Type": info["Type"],
                "centroid": info.get("centroid"),
            })
        return summaries

    def assign_crack_manually(
        self,
        crack_id: str,
        element_global_id: str,
        notes: str = "",
    ) -> Dict:
        """Manually assign a crack to an IFC element.

        Parameters
        ----------
        crack_id : str
            Detection identifier.
        element_global_id : str
            IFC ``GlobalId`` of the target element.
        notes : str, optional
            Free-text notes for the assignment.

        Returns
        -------
        dict
            Assignment record.
        """
        elem_info = self.elements.get(element_global_id)
        mapping = {
            "crack_id": crack_id,
            "element_id": element_global_id,
            "element_name": (
                elem_info["Name"] if elem_info else "Unknown"
            ),
            "element_type": (
                elem_info["Type"] if elem_info else "Unknown"
            ),
            "distance": 0.0,
            "mapping_method": "manual",
            "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "Manual assignment: crack %s → element %s (notes=%r).",
            crack_id, element_global_id, notes,
        )
        return mapping
