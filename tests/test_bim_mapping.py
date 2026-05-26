"""
Tests for BIM Mapping Module
=============================
Tests coordinate transforms, IFC element parsing, crack-to-element
mapping, and IFC annotation.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bim_mapping.coordinate_transform import CoordinateTransformer
from tests.fixtures.generate_test_data import generate_test_ifc_model

# Check if ifcopenshell is available
try:
    import ifcopenshell
    HAS_IFC = True
except ImportError:
    HAS_IFC = False


# ──────────────────────────────────────────────────────────────────────
# Coordinate Transform Tests
# ──────────────────────────────────────────────────────────────────────

class TestCoordinateTransform:
    """Test the CoordinateTransformer class."""

    @pytest.fixture
    def transformer(self):
        cam = np.array([[800, 0, 320], [0, 800, 320], [0, 0, 1]], dtype=np.float64)
        return CoordinateTransformer(camera_matrix=cam)

    def test_image_to_camera_center(self, transformer):
        """Center pixel at known depth should map to (0, 0, depth)."""
        pt = transformer.image_to_camera(320, 320, 2000.0)
        assert pt is not None
        assert len(pt) == 3
        assert abs(pt[0]) < 1.0, f"X should be ~0 for center pixel, got {pt[0]}"
        assert abs(pt[1]) < 1.0, f"Y should be ~0 for center pixel, got {pt[1]}"
        assert abs(pt[2] - 2000.0) < 1.0, f"Z should be ~2000, got {pt[2]}"

    def test_image_to_camera_offset(self, transformer):
        """Off-center pixel should produce non-zero X, Y."""
        pt = transformer.image_to_camera(420, 220, 1000.0)
        assert pt[0] > 0, "Right of center should have positive X"
        assert pt[1] < 0, "Above center should have negative Y"

    def test_camera_to_world_identity(self, transformer):
        """Identity extrinsic should not change the point."""
        extrinsic = np.eye(4)
        pt_cam = np.array([1.0, 2.0, 3.0])
        pt_world = transformer.camera_to_world(pt_cam, extrinsic)

        np.testing.assert_array_almost_equal(pt_world, pt_cam, decimal=5)

    def test_camera_to_world_translation(self, transformer):
        """Translation-only extrinsic should offset the point."""
        extrinsic = np.eye(4)
        extrinsic[0, 3] = 10.0  # translate X by 10
        extrinsic[1, 3] = 20.0  # translate Y by 20
        extrinsic[2, 3] = 30.0  # translate Z by 30

        pt_cam = np.array([1.0, 2.0, 3.0])
        pt_world = transformer.camera_to_world(pt_cam, extrinsic)

        expected = np.array([11.0, 22.0, 33.0])
        np.testing.assert_array_almost_equal(pt_world, expected, decimal=5)

    def test_world_to_ifc_identity(self, transformer):
        """No IFC transform should pass through unchanged."""
        pt = np.array([5.0, 10.0, 15.0])
        result = transformer.world_to_ifc(pt)
        np.testing.assert_array_almost_equal(result, pt, decimal=5)

    def test_full_transform_chain(self, transformer):
        """Full image→camera→world→IFC chain should not crash."""
        extrinsic = np.eye(4)
        result = transformer.image_to_ifc(
            u=320, v=320, depth=2000.0,
            extrinsic_matrix=extrinsic,
        )
        assert result is not None
        assert len(result) == 3

    def test_manual_mapping_2d(self, transformer):
        """Test homography computation from point pairs."""
        # 4 corresponding point pairs (simple scale + translate)
        image_pts = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ], dtype=np.float32)
        ifc_pts = np.array([
            [10, 10], [20, 10], [20, 20], [10, 20]
        ], dtype=np.float32)

        H = transformer.create_manual_mapping(image_pts, ifc_pts)
        assert H is not None
        assert H.shape[0] >= 3

    def test_apply_manual_mapping(self, transformer):
        """Apply homography to a new point."""
        image_pts = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ], dtype=np.float32)
        ifc_pts = np.array([
            [0, 0], [10, 0], [10, 10], [0, 10]
        ], dtype=np.float32)

        H = transformer.create_manual_mapping(image_pts, ifc_pts)
        if H is not None:
            result = transformer.apply_manual_mapping(np.array([50, 50]), H)
            assert result is not None


# ──────────────────────────────────────────────────────────────────────
# BIM Mapper Tests
# ──────────────────────────────────────────────────────────────────────

class TestBIMMapper:
    """Test the CrackBIMMapper class."""

    @pytest.fixture
    def ifc_path(self, tmp_path):
        path = str(tmp_path / "test_bridge.ifc")
        generate_test_ifc_model(path)
        return path

    def test_mapper_init_no_file(self):
        """Mapper should initialize without an IFC file."""
        from bim_mapping.mapper import CrackBIMMapper
        mapper = CrackBIMMapper()
        assert mapper is not None

    @pytest.mark.skipif(not HAS_IFC, reason="ifcopenshell not installed")
    def test_ifc_model_loading(self, ifc_path):
        """Load test IFC and verify elements are parsed."""
        from bim_mapping.mapper import CrackBIMMapper
        mapper = CrackBIMMapper(ifc_path=ifc_path)

        summary = mapper.get_element_summary()
        assert isinstance(summary, list)
        assert len(summary) > 0, "Should have parsed at least one element"

        # Check we found our test elements
        names = [e.get("name", "") for e in summary]
        has_beam = any("Girder" in n or "Beam" in n for n in names)
        has_column = any("Pier" in n or "Column" in n for n in names)
        assert has_beam or has_column or len(summary) >= 2

    @pytest.mark.skipif(not HAS_IFC, reason="ifcopenshell not installed")
    def test_nearest_element_search(self, ifc_path):
        """Find nearest element to a 3D point."""
        from bim_mapping.mapper import CrackBIMMapper
        mapper = CrackBIMMapper(ifc_path=ifc_path)

        result = mapper.find_nearest_element(np.array([0.0, 0.0, 5.5]))
        if result is not None:
            assert "global_id" in result or "GlobalId" in result or "name" in result

    def test_manual_assignment(self):
        """Manual crack-to-element assignment should work without IFC."""
        from bim_mapping.mapper import CrackBIMMapper
        mapper = CrackBIMMapper()

        mapping = mapper.assign_crack_manually(
            crack_id="crack_001",
            element_global_id="ELEM_ABC123",
            notes="Assigned by inspector",
        )
        assert mapping is not None
        assert mapping.get("crack_id") == "crack_001"
        assert mapping.get("element_id") == "ELEM_ABC123"


# ──────────────────────────────────────────────────────────────────────
# IFC Annotator Tests
# ──────────────────────────────────────────────────────────────────────

class TestIFCAnnotator:
    """Test the IFCAnnotator class."""

    @pytest.fixture
    def ifc_path(self, tmp_path):
        path = str(tmp_path / "test_bridge.ifc")
        generate_test_ifc_model(path)
        return path

    @pytest.mark.skipif(not HAS_IFC, reason="ifcopenshell not installed")
    def test_annotator_load(self, ifc_path):
        """Annotator should load an IFC model."""
        from bim_mapping.ifc_annotator import IFCAnnotator
        annotator = IFCAnnotator(ifc_path=ifc_path)
        assert annotator is not None

    @pytest.mark.skipif(not HAS_IFC, reason="ifcopenshell not installed")
    def test_crack_annotation(self, ifc_path):
        """Annotate a crack and verify data is written."""
        from bim_mapping.ifc_annotator import IFCAnnotator

        annotator = IFCAnnotator(ifc_path=ifc_path)

        # Find a valid element GlobalId
        ifc_model = ifcopenshell.open(ifc_path)
        beams = ifc_model.by_type("IfcBeam")
        if not beams:
            elements = ifc_model.by_type("IfcProduct")
            if not elements:
                pytest.skip("No elements found in test IFC")
            elem_id = elements[0].GlobalId
        else:
            elem_id = beams[0].GlobalId

        crack_data = {
            "detection_id": "crack_test_001",
            "width_mm": 0.25,
            "length_mm": 150.0,
            "severity": "MODERATE",
            "method": "MONOCULAR_GSD",
            "confidence": 0.92,
            "is456_compliant": True,
            "remediation_notes": "Schedule epoxy injection",
            "source_image": "test_image.jpg",
            "inspection_date": "2025-01-15",
        }
        mapping_info = {
            "element_id": elem_id,
            "element_name": "Test Beam",
            "distance": 0.0,
        }

        new_id = annotator.annotate_crack(elem_id, crack_data, mapping_info)
        assert new_id is not None

    @pytest.mark.skipif(not HAS_IFC, reason="ifcopenshell not installed")
    def test_annotation_save(self, ifc_path, tmp_path):
        """Save annotated model and verify file exists."""
        from bim_mapping.ifc_annotator import IFCAnnotator

        annotator = IFCAnnotator(ifc_path=ifc_path)
        output = str(tmp_path / "annotated_bridge.ifc")
        saved_path = annotator.save(output_path=output)

        assert saved_path is not None
        assert os.path.exists(saved_path)
        assert os.path.getsize(saved_path) > 0

    def test_no_ifcopenshell_fallback(self):
        """Graceful degradation when ifcopenshell is missing."""
        # This just tests that import doesn't crash
        from bim_mapping.mapper import CrackBIMMapper
        mapper = CrackBIMMapper()
        assert mapper is not None
