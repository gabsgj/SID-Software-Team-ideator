"""
IFC Annotator
=============
Writes crack detection data directly into an IFC model file as structured
annotations.

- **IFC 4.3**: Uses ``IfcSurfaceFeature`` with PredefinedType = DEFECT.
- **IFC 4 / IFC 2×3**: Falls back to ``IfcAnnotation``.

All writes are performed on a *working copy* — the original file is never
modified.
"""

import copy
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import ifcopenshell
    import ifcopenshell.guid
    HAS_IFC = True
except ImportError:
    HAS_IFC = False
    logger.warning(
        "ifcopenshell not installed. IFC annotation will be unavailable."
    )


class IFCAnnotator:
    """Annotate an IFC model with crack inspection results.

    All mutations happen on an in-memory copy of the model.  Call
    :meth:`save` to persist changes to disk.

    Parameters
    ----------
    ifc_path : str, optional
        Path to an IFC model file.  Loaded immediately if provided.
    """

    def __init__(self, ifc_path: Optional[str] = None) -> None:
        self.ifc_model = None
        self.ifc_path: Optional[str] = None
        self._schema_version: Optional[str] = None
        self._annotations_created: List[Dict] = []

        if ifc_path is not None:
            self.load_model(ifc_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self, ifc_path: str) -> None:
        """Open an IFC file and prepare a working copy.

        Parameters
        ----------
        ifc_path : str
            Path to the ``.ifc`` file.

        Raises
        ------
        RuntimeError
            If ``ifcopenshell`` is not available.
        """
        if not HAS_IFC:
            raise RuntimeError(
                "ifcopenshell is required for IFC annotation. "
                "Install it with: pip install ifcopenshell"
            )

        logger.info("Loading IFC model for annotation: %s", ifc_path)
        self.ifc_path = ifc_path

        # Work on a copy to protect the original
        self.ifc_model = ifcopenshell.open(ifc_path)
        self._schema_version = self.ifc_model.schema
        self._annotations_created.clear()

        logger.info(
            "IFC model loaded (schema: %s). Working on in-memory copy.",
            self._schema_version,
        )

    # ------------------------------------------------------------------
    # Public annotation API
    # ------------------------------------------------------------------
    def annotate_crack(
        self,
        element_global_id: str,
        crack_data: Dict,
        mapping_info: Dict,
    ) -> str:
        """Annotate a structural element with a crack detection.

        Creates an ``IfcSurfaceFeature`` (IFC 4.3) or ``IfcAnnotation``
        (IFC 2×3 / 4 fallback), attaches a property set with detailed
        crack measurements, and links the feature to the parent element.

        Parameters
        ----------
        element_global_id : str
            ``GlobalId`` of the target structural element.
        crack_data : dict
            Crack detection record.  Expected keys include:
            ``detection_id``, ``width_mm``, ``severity_level``,
            ``confidence``, and optionally ``length_mm``,
            ``measurement_method``, ``is456_compliant``,
            ``remediation_notes``, ``source_image``.
        mapping_info : dict
            Mapping record produced by :class:`CrackBIMMapper`.

        Returns
        -------
        str
            The ``GlobalId`` of the newly created IFC entity.

        Raises
        ------
        RuntimeError
            If no model is loaded.
        """
        if self.ifc_model is None:
            raise RuntimeError("No IFC model loaded.")

        crack_id = crack_data.get("detection_id", "unknown")
        severity = crack_data.get("severity_level", "UNKNOWN")
        name = f"Crack_{crack_id}"
        description = (
            f"Detected crack — severity: {severity}, "
            f"width: {crack_data.get('width_mm', 'N/A')} mm"
        )

        # Create the feature entity
        if self._supports_surface_feature():
            feature = self._create_surface_feature(name, description)
            logger.debug("Created IfcSurfaceFeature: %s", name)
        else:
            feature = self._create_annotation_fallback(name, description)
            logger.debug("Created IfcAnnotation fallback: %s", name)

        new_global_id = feature.GlobalId

        # Property set
        pset = self._create_property_set(crack_data)

        # Relationships
        self._link_to_element(feature, element_global_id)
        self._attach_property_set(feature, pset)

        # Book-keeping
        record = {
            "annotation_global_id": new_global_id,
            "element_global_id": element_global_id,
            "crack_id": crack_id,
            "severity": severity,
            "entity_type": feature.is_a(),
        }
        self._annotations_created.append(record)

        logger.info(
            "Annotated element %s with crack %s (entity %s, id=%s).",
            element_global_id, crack_id,
            feature.is_a(), new_global_id,
        )
        return new_global_id

    def annotate_batch(
        self,
        mappings: List[Dict],
        crack_detections: List[Dict],
    ) -> List[str]:
        """Annotate multiple cracks at once.

        Parameters
        ----------
        mappings : list[dict]
            Mapping records (one per crack).
        crack_detections : list[dict]
            Detection records (one per crack, same order).

        Returns
        -------
        list[str]
            ``GlobalId`` values for every created annotation.
        """
        if len(mappings) != len(crack_detections):
            raise ValueError(
                "mappings and crack_detections must have equal length."
            )

        ids: List[str] = []
        for mapping, crack in zip(mappings, crack_detections):
            elem_id = mapping.get("element_id")
            if elem_id is None:
                logger.warning(
                    "Skipping crack %s — no element mapping.",
                    crack.get("detection_id", "?"),
                )
                continue
            new_id = self.annotate_crack(elem_id, crack, mapping)
            ids.append(new_id)

        logger.info(
            "Batch annotation complete: %d/%d cracks annotated.",
            len(ids), len(crack_detections),
        )
        return ids

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, output_path: Optional[str] = None) -> str:
        """Persist the annotated model to disk.

        Parameters
        ----------
        output_path : str, optional
            Destination path.  If omitted, ``_annotated`` is appended to
            the original filename (before the extension).

        Returns
        -------
        str
            The path to the saved file.

        Raises
        ------
        RuntimeError
            If no model is loaded.
        """
        if self.ifc_model is None:
            raise RuntimeError("No IFC model loaded.")

        if output_path is None:
            base, ext = os.path.splitext(self.ifc_path)
            output_path = f"{base}_annotated{ext}"

        self.ifc_model.write(output_path)
        logger.info(
            "Annotated IFC model saved to: %s (%d annotations).",
            output_path, len(self._annotations_created),
        )
        return output_path

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def get_annotation_summary(self) -> Dict:
        """Return a summary of all annotations created in this session.

        Returns
        -------
        dict
            Keys: ``total``, ``by_severity``, ``by_element_type``,
            ``annotations``.
        """
        by_severity: Dict[str, int] = {}
        by_entity_type: Dict[str, int] = {}

        for rec in self._annotations_created:
            sev = rec.get("severity", "UNKNOWN")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            etype = rec.get("entity_type", "Unknown")
            by_entity_type[etype] = by_entity_type.get(etype, 0) + 1

        return {
            "total": len(self._annotations_created),
            "by_severity": by_severity,
            "by_entity_type": by_entity_type,
            "annotations": list(self._annotations_created),
        }

    # ------------------------------------------------------------------
    # Internal: IFC entity creation
    # ------------------------------------------------------------------
    def _supports_surface_feature(self) -> bool:
        """Return True if the schema supports IfcSurfaceFeature."""
        if self._schema_version is None:
            return False
        # IfcSurfaceFeature was introduced in IFC 4 ADD2 / IFC 4.3
        try:
            self.ifc_model.create_entity("IfcSurfaceFeature")
            return True
        except (RuntimeError, AttributeError):
            return False

    def _create_surface_feature(
        self,
        name: str,
        description: str,
    ):
        """Create an ``IfcSurfaceFeature`` entity.

        Parameters
        ----------
        name : str
            Feature name.
        description : str
            Human-readable description.

        Returns
        -------
        entity
            The newly created IFC entity.
        """
        new_guid = ifcopenshell.guid.new()
        owner_history = self._get_owner_history()

        try:
            feature = self.ifc_model.create_entity(
                "IfcSurfaceFeature",
                GlobalId=new_guid,
                OwnerHistory=owner_history,
                Name=name,
                Description=description,
                PredefinedType="DEFECT",
            )
        except TypeError:
            # Some builds don't accept PredefinedType as kwarg
            feature = self.ifc_model.create_entity(
                "IfcSurfaceFeature",
                GlobalId=new_guid,
                OwnerHistory=owner_history,
                Name=name,
                Description=description,
            )
            try:
                feature.PredefinedType = "DEFECT"
            except (AttributeError, RuntimeError):
                logger.debug("PredefinedType=DEFECT not supported.")

        return feature

    def _create_annotation_fallback(
        self,
        name: str,
        description: str,
    ):
        """Create an ``IfcAnnotation`` entity (fallback for older schemas).

        Parameters
        ----------
        name : str
            Annotation name.
        description : str
            Human-readable description.

        Returns
        -------
        entity
            The newly created IFC entity.
        """
        new_guid = ifcopenshell.guid.new()
        owner_history = self._get_owner_history()

        feature = self.ifc_model.create_entity(
            "IfcAnnotation",
            GlobalId=new_guid,
            OwnerHistory=owner_history,
            Name=name,
            Description=description,
        )
        return feature

    # ------------------------------------------------------------------
    # Internal: Property set
    # ------------------------------------------------------------------
    def _create_property_set(self, crack_data: Dict):
        """Build a ``Pset_CrackInspection`` property set.

        Parameters
        ----------
        crack_data : dict
            Detection record with measurement values.

        Returns
        -------
        entity
            The ``IfcPropertySet`` entity.
        """
        owner_history = self._get_owner_history()
        pset_guid = ifcopenshell.guid.new()

        properties = []

        # Helper to create a single-value property
        def _spv(name, value, ifc_type):
            """Create an IfcPropertySingleValue."""
            if value is None:
                return None
            try:
                wrapped = self.ifc_model.create_entity(ifc_type, value)
            except Exception:
                wrapped = self.ifc_model.create_entity(
                    "IfcLabel", str(value)
                )
            prop = self.ifc_model.create_entity(
                "IfcPropertySingleValue",
                Name=name,
                NominalValue=wrapped,
            )
            return prop

        # InspectionDate
        inspection_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prop = _spv("InspectionDate", inspection_date, "IfcDate")
        if prop is None:
            # IfcDate may not exist in IFC 2×3 — fall back to IfcLabel
            prop = _spv("InspectionDate", inspection_date, "IfcLabel")
        if prop:
            properties.append(prop)

        # CrackWidthMm
        width = crack_data.get("width_mm")
        prop = _spv("CrackWidthMm", float(width) if width is not None else None, "IfcReal")
        if prop:
            properties.append(prop)

        # CrackLengthMm
        length = crack_data.get("length_mm")
        prop = _spv("CrackLengthMm", float(length) if length is not None else None, "IfcReal")
        if prop:
            properties.append(prop)

        # CrackSeverity
        prop = _spv("CrackSeverity", crack_data.get("severity_level"), "IfcLabel")
        if prop:
            properties.append(prop)

        # MeasurementMethod
        prop = _spv(
            "MeasurementMethod",
            crack_data.get("measurement_method", "monocular"),
            "IfcLabel",
        )
        if prop:
            properties.append(prop)

        # Confidence
        conf = crack_data.get("confidence")
        prop = _spv("Confidence", float(conf) if conf is not None else None, "IfcReal")
        if prop:
            properties.append(prop)

        # IS456Compliant
        compliant = crack_data.get("is456_compliant")
        if compliant is not None:
            try:
                wrapped = self.ifc_model.create_entity(
                    "IfcBoolean", bool(compliant)
                )
                prop = self.ifc_model.create_entity(
                    "IfcPropertySingleValue",
                    Name="IS456Compliant",
                    NominalValue=wrapped,
                )
                properties.append(prop)
            except Exception:
                prop = _spv("IS456Compliant", str(compliant), "IfcLabel")
                if prop:
                    properties.append(prop)

        # RemediationNotes
        prop = _spv(
            "RemediationNotes",
            crack_data.get("remediation_notes"),
            "IfcText",
        )
        if prop:
            properties.append(prop)

        # SourceImage
        prop = _spv("SourceImage", crack_data.get("source_image"), "IfcLabel")
        if prop:
            properties.append(prop)

        # DetectionId
        prop = _spv("DetectionId", crack_data.get("detection_id"), "IfcLabel")
        if prop:
            properties.append(prop)

        # Build the property set
        pset = self.ifc_model.create_entity(
            "IfcPropertySet",
            GlobalId=pset_guid,
            OwnerHistory=owner_history,
            Name="Pset_CrackInspection",
            HasProperties=properties,
        )

        logger.debug(
            "Created Pset_CrackInspection with %d properties.", len(properties),
        )
        return pset

    # ------------------------------------------------------------------
    # Internal: Relationships
    # ------------------------------------------------------------------
    def _link_to_element(self, feature_entity, element_global_id: str) -> None:
        """Link an annotation/feature entity to a structural element.

        Uses ``IfcRelAggregates`` to create a parent–child relationship
        between the structural element and the crack feature.

        Parameters
        ----------
        feature_entity
            The feature / annotation entity to link.
        element_global_id : str
            ``GlobalId`` of the parent structural element.
        """
        parent = self._find_element_by_global_id(element_global_id)
        if parent is None:
            logger.warning(
                "Cannot link — element %s not found in model.",
                element_global_id,
            )
            return

        owner_history = self._get_owner_history()
        rel_guid = ifcopenshell.guid.new()

        try:
            self.ifc_model.create_entity(
                "IfcRelAggregates",
                GlobalId=rel_guid,
                OwnerHistory=owner_history,
                Name="CrackFeatureLink",
                Description="Links crack feature to structural element",
                RelatingObject=parent,
                RelatedObjects=[feature_entity],
            )
            logger.debug(
                "Linked %s to element %s via IfcRelAggregates.",
                feature_entity.GlobalId, element_global_id,
            )
        except Exception as exc:
            logger.warning(
                "IfcRelAggregates failed (%s). "
                "Trying IfcRelContainedInSpatialStructure.",
                exc,
            )
            try:
                self.ifc_model.create_entity(
                    "IfcRelContainedInSpatialStructure",
                    GlobalId=rel_guid,
                    OwnerHistory=owner_history,
                    Name="CrackFeatureContainment",
                    Description="Crack feature spatial containment",
                    RelatedElements=[feature_entity],
                    RelatingStructure=parent,
                )
                logger.debug(
                    "Linked via IfcRelContainedInSpatialStructure."
                )
            except Exception as exc2:
                logger.error(
                    "Failed to link feature to element: %s", exc2,
                )

    def _attach_property_set(self, entity, pset) -> None:
        """Attach a property set to an entity via ``IfcRelDefinesByProperties``.

        Parameters
        ----------
        entity
            The IFC entity to receive the property set.
        pset
            The ``IfcPropertySet`` entity.
        """
        owner_history = self._get_owner_history()
        rel_guid = ifcopenshell.guid.new()

        try:
            self.ifc_model.create_entity(
                "IfcRelDefinesByProperties",
                GlobalId=rel_guid,
                OwnerHistory=owner_history,
                Name="CrackPropertyLink",
                Description="Links crack property set to annotation",
                RelatedObjects=[entity],
                RelatingPropertyDefinition=pset,
            )
            logger.debug(
                "Attached Pset_CrackInspection to entity %s.",
                entity.GlobalId,
            )
        except Exception as exc:
            logger.error(
                "Failed to attach property set: %s", exc,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_owner_history(self):
        """Return the first ``IfcOwnerHistory`` in the model, or None."""
        try:
            histories = self.ifc_model.by_type("IfcOwnerHistory")
            return histories[0] if histories else None
        except Exception:
            return None

    def _find_element_by_global_id(self, global_id: str):
        """Find an IFC entity by its ``GlobalId``."""
        try:
            return self.ifc_model.by_guid(global_id)
        except Exception:
            logger.debug("Element with GlobalId %s not found.", global_id)
            return None
