"""
Synthetic Test Data Generator
=============================
Generates synthetic crack images, depth maps, and IFC models with known
ground-truth dimensions for benchmarking the width estimation algorithms.

Every generated crack has a **known true width** so that estimation accuracy
can be measured quantitatively.

Author: IDEATOR GECT — SID Structural Inspection Drone
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  1. Straight crack with known width
# ──────────────────────────────────────────────────────────────────────

def generate_synthetic_crack_image(
    width_px: int = 5,
    length_px: int = 200,
    orientation_deg: float = 45.0,
    image_size: Tuple[int, int] = (640, 640),
    noise_level: float = 0.02,
    curvature: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate a grayscale image containing a single synthetic crack.

    The crack is drawn as a filled polygon with the specified width and
    approximate length, centered in the image.  Minor edge roughness is
    added for realism.

    Args:
        width_px: True crack width in pixels.
        length_px: Approximate crack length in pixels.
        orientation_deg: Angle of the crack from horizontal (degrees).
        image_size: (height, width) of the output image.
        noise_level: Standard deviation of additive Gaussian noise (0–1 scale).
        curvature: Curvature parameter (0 = straight, higher = more curved).

    Returns:
        (image, mask, metadata) where:
            image: uint8 grayscale image (H×W).
            mask:  uint8 binary mask  (0 or 255).
            metadata: dict with ground-truth values.
    """
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = w // 2, h // 2
    angle_rad = math.radians(orientation_deg)

    # Direction vectors
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    # Perpendicular direction
    px = -dy
    py = dx

    half_len = length_px / 2.0
    half_w = width_px / 2.0

    # Number of sample points along the crack centerline
    n_samples = max(int(length_px), 60)

    rng = np.random.RandomState(42)

    # Build centerline points (with optional curvature)
    ts = np.linspace(-half_len, half_len, n_samples)
    center_x = []
    center_y = []
    for t in ts:
        # Add slight curvature
        curve_offset = curvature * (t ** 2) / (half_len if half_len > 0 else 1)
        cx_t = cx + t * dx + curve_offset * px
        cy_t = cy + t * dy + curve_offset * py
        center_x.append(cx_t)
        center_y.append(cy_t)

    # Build left and right edges with slight roughness
    roughness = max(0.3, width_px * 0.08)
    left_pts = []
    right_pts = []
    for i in range(n_samples):
        # Small random perturbation for edge roughness
        r_left = half_w + rng.normal(0, roughness)
        r_right = half_w + rng.normal(0, roughness)
        r_left = max(0.5, r_left)
        r_right = max(0.5, r_right)

        left_pts.append((
            int(round(center_x[i] + r_left * px)),
            int(round(center_y[i] + r_left * py)),
        ))
        right_pts.append((
            int(round(center_x[i] - r_right * px)),
            int(round(center_y[i] - r_right * py)),
        ))

    # Form a closed polygon (left edge forward, right edge backward)
    polygon = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    # Create the image: concrete-like background + dark crack
    image = rng.normal(180, 12, (h, w)).clip(0, 255).astype(np.uint8)

    # Darken the crack region
    crack_intensity = rng.normal(40, 8, (h, w)).clip(0, 80).astype(np.uint8)
    image[mask > 0] = crack_intensity[mask > 0]

    # Add Gaussian noise
    if noise_level > 0:
        noise = rng.normal(0, noise_level * 255, (h, w))
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Slight Gaussian blur for realism
    image = cv2.GaussianBlur(image, (3, 3), 0.5)

    metadata = {
        "true_width_px": width_px,
        "true_length_px": length_px,
        "true_orientation_deg": orientation_deg,
        "image_size": image_size,
        "curvature": curvature,
    }

    return image, mask, metadata


# ──────────────────────────────────────────────────────────────────────
#  2. Multiple known-width cracks for benchmarking
# ──────────────────────────────────────────────────────────────────────

def generate_crack_with_known_width(
    widths: Optional[List[int]] = None,
    image_size: Tuple[int, int] = (640, 640),
) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """Generate a set of crack images with different known widths.

    Args:
        widths: List of crack widths in pixels.
        image_size: Output image size.

    Returns:
        List of (image, mask, metadata) tuples.
    """
    if widths is None:
        widths = [1, 2, 3, 5, 8, 10, 15, 20]

    results = []
    for w in widths:
        # Vary orientation for diversity
        angle = 30 + (w * 7) % 120
        img, msk, meta = generate_synthetic_crack_image(
            width_px=w,
            length_px=max(100, w * 15),
            orientation_deg=angle,
            image_size=image_size,
        )
        results.append((img, msk, meta))

    return results


# ──────────────────────────────────────────────────────────────────────
#  3. Branching (Y-shaped) crack
# ──────────────────────────────────────────────────────────────────────

def generate_branching_crack(
    main_width: int = 5,
    branch_width: int = 3,
    image_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate a Y-shaped branching crack.

    The main trunk runs vertically with a branch splitting off at ~45°.
    Tests skeleton pruning and width measurement near branch points.

    Returns:
        (image, mask, metadata)
    """
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    rng = np.random.RandomState(99)

    cx, cy = w // 2, h // 2

    # Main trunk: vertical, from top-third to bottom-third
    trunk_top = cy - 150
    trunk_bottom = cy + 150
    trunk_half_w = main_width // 2

    # Draw main trunk
    pts_trunk = np.array([
        [cx - trunk_half_w, trunk_top],
        [cx + trunk_half_w, trunk_top],
        [cx + trunk_half_w, trunk_bottom],
        [cx - trunk_half_w, trunk_bottom],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts_trunk], 255)

    # Branch: starts at midpoint, goes to upper-right at 45°
    branch_start_y = cy - 30
    branch_len = 100
    branch_half_w = branch_width // 2
    angle_rad = math.radians(-45)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    px_dir = -dy
    py_dir = dx

    branch_pts = []
    for t in np.linspace(0, branch_len, 40):
        bx = cx + t * dx
        by = branch_start_y + t * dy
        branch_pts.append((bx, by))

    left_pts = [(int(bx + branch_half_w * px_dir), int(by + branch_half_w * py_dir))
                for bx, by in branch_pts]
    right_pts = [(int(bx - branch_half_w * px_dir), int(by - branch_half_w * py_dir))
                 for bx, by in branch_pts]
    poly = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)

    # Create image
    image = rng.normal(180, 12, (h, w)).clip(0, 255).astype(np.uint8)
    crack_intensity = rng.normal(40, 8, (h, w)).clip(0, 80).astype(np.uint8)
    image[mask > 0] = crack_intensity[mask > 0]
    image = cv2.GaussianBlur(image, (3, 3), 0.5)

    metadata = {
        "true_main_width_px": main_width,
        "true_branch_width_px": branch_width,
        "type": "branching",
        "image_size": image_size,
    }

    return image, mask, metadata


# ──────────────────────────────────────────────────────────────────────
#  4. Curved (arc-shaped) crack
# ──────────────────────────────────────────────────────────────────────

def generate_curved_crack(
    width_px: int = 5,
    curvature_radius: float = 300.0,
    image_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate a curved arc-shaped crack.

    Tests orientation estimation that must change along the crack.

    Returns:
        (image, mask, metadata)
    """
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    rng = np.random.RandomState(77)

    cx, cy = w // 2, h // 2
    half_w = width_px / 2.0

    # Arc parameters
    arc_center_x = cx
    arc_center_y = cy + curvature_radius
    start_angle = math.radians(-60)
    end_angle = math.radians(60)
    n_samples = 100

    angles = np.linspace(start_angle, end_angle, n_samples)

    left_pts = []
    right_pts = []
    for theta in angles:
        # Point on arc
        ax = arc_center_x + curvature_radius * math.sin(theta)
        ay = arc_center_y - curvature_radius * math.cos(theta)

        # Normal direction (radially outward)
        nx = math.sin(theta)
        ny = -math.cos(theta)

        roughness = max(0.3, width_px * 0.06)
        r1 = half_w + rng.normal(0, roughness)
        r2 = half_w + rng.normal(0, roughness)

        left_pts.append((int(round(ax + max(0.5, r1) * nx)),
                         int(round(ay + max(0.5, r1) * ny))))
        right_pts.append((int(round(ax - max(0.5, r2) * nx)),
                          int(round(ay - max(0.5, r2) * ny))))

    polygon = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    # Create image
    image = rng.normal(180, 12, (h, w)).clip(0, 255).astype(np.uint8)
    crack_intensity = rng.normal(40, 8, (h, w)).clip(0, 80).astype(np.uint8)
    image[mask > 0] = crack_intensity[mask > 0]
    image = cv2.GaussianBlur(image, (3, 3), 0.5)

    metadata = {
        "true_width_px": width_px,
        "curvature_radius": curvature_radius,
        "type": "curved",
        "image_size": image_size,
    }

    return image, mask, metadata


# ──────────────────────────────────────────────────────────────────────
#  5. Synthetic depth map
# ──────────────────────────────────────────────────────────────────────

def generate_synthetic_depth_map(
    image_size: Tuple[int, int] = (640, 640),
    base_depth: float = 2000.0,
    noise_std: float = 5.0,
) -> np.ndarray:
    """Generate a synthetic depth map simulating a roughly planar surface.

    Args:
        image_size: (height, width).
        base_depth: Mean depth in mm.
        noise_std: Standard deviation of depth noise in mm.

    Returns:
        float32 depth map (H×W), values in mm.
    """
    h, w = image_size
    rng = np.random.RandomState(55)

    # Slight planar tilt
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    tilt = 0.05 * (xs - w / 2) + 0.03 * (ys - h / 2)

    depth = np.full((h, w), base_depth, dtype=np.float32) + tilt
    depth += rng.normal(0, noise_std, (h, w)).astype(np.float32)

    return depth


# ──────────────────────────────────────────────────────────────────────
#  6. Minimal test IFC model
# ──────────────────────────────────────────────────────────────────────

def generate_test_ifc_model(output_path: str) -> Optional[str]:
    """Create a minimal IFC4 file with beam, column, and slab elements.

    Args:
        output_path: Where to write the .ifc file.

    Returns:
        Path to the created file, or None if ifcopenshell is unavailable.
    """
    try:
        import ifcopenshell
        import ifcopenshell.guid
    except ImportError:
        logger.warning("ifcopenshell not installed — cannot generate test IFC model.")
        # Fallback: write a minimal IFC text file for basic parsing tests
        _write_minimal_ifc_text(output_path)
        return output_path

    try:
        ifc = ifcopenshell.file(schema="IFC4")

        # Header
        person = ifc.createIfcPerson(
            Identification="SID",
            FamilyName="IDEATOR",
            GivenName="SID",
        )
        org = ifc.createIfcOrganization(Name="IDEATOR_GECT")
        person_org = ifc.createIfcPersonAndOrganization(
            ThePerson=person,
            TheOrganization=org,
        )
        app = ifc.createIfcApplication(
            ApplicationDeveloper=org,
            Version="1.0",
            ApplicationFullName="SID Crack Detector",
            ApplicationIdentifier="SID",
        )
        owner = ifc.createIfcOwnerHistory(
            OwningUser=person_org,
            OwningApplication=app,
            ChangeAction="ADDED",
            CreationDate=0,
        )

        # Context
        context = ifc.createIfcGeometricRepresentationContext(
            ContextType="Model",
            CoordinateSpaceDimension=3,
            Precision=1e-5,
            WorldCoordinateSystem=ifc.createIfcAxis2Placement3D(
                Location=ifc.createIfcCartesianPoint((0.0, 0.0, 0.0)),
            ),
        )

        # Project and site
        project = ifc.createIfcProject(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name="SID Test Bridge",
            RepresentationContexts=[context],
            UnitsInContext=ifc.createIfcUnitAssignment(
                Units=[
                    ifc.createIfcSIUnit(UnitType="LENGTHUNIT", Name="METRE"),
                    ifc.createIfcSIUnit(UnitType="AREAUNIT", Name="SQUARE_METRE"),
                    ifc.createIfcSIUnit(UnitType="VOLUMEUNIT", Name="CUBIC_METRE"),
                ]
            ),
        )

        site = ifc.createIfcSite(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name="Bridge Site",
        )

        building = ifc.createIfcBuilding(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name="Test Bridge",
        )

        storey = ifc.createIfcBuildingStorey(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name="Deck Level",
        )

        # Spatial hierarchy
        ifc.createIfcRelAggregates(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            RelatingObject=project,
            RelatedObjects=[site],
        )
        ifc.createIfcRelAggregates(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            RelatingObject=site,
            RelatedObjects=[building],
        )
        ifc.createIfcRelAggregates(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            RelatingObject=building,
            RelatedObjects=[storey],
        )

        # Helper to create a simple extruded box
        def create_box(x, y, z, dx, dy, dz):
            placement = ifc.createIfcLocalPlacement(
                RelativePlacement=ifc.createIfcAxis2Placement3D(
                    Location=ifc.createIfcCartesianPoint((float(x), float(y), float(z))),
                )
            )
            # Simple extruded area solid
            rect_profile = ifc.createIfcRectangleProfileDef(
                ProfileType="AREA",
                XDim=float(dx),
                YDim=float(dy),
            )
            direction = ifc.createIfcDirection((0.0, 0.0, 1.0))
            body = ifc.createIfcExtrudedAreaSolid(
                SweptArea=rect_profile,
                Depth=float(dz),
                ExtrudedDirection=direction,
            )
            shape = ifc.createIfcShapeRepresentation(
                ContextOfItems=context,
                RepresentationIdentifier="Body",
                RepresentationType="SweptSolid",
                Items=[body],
            )
            product_shape = ifc.createIfcProductDefinitionShape(
                Representations=[shape],
            )
            return placement, product_shape

        # Create structural elements
        elements = []

        # Beam (spanning the bridge deck)
        p, s = create_box(0, 0, 5, 10, 0.5, 0.8)
        beam = ifc.createIfcBeam(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name="Main_Girder_1",
            ObjectPlacement=p,
            Representation=s,
        )
        elements.append(beam)

        # Column (pier)
        p, s = create_box(-4, 0, 0, 0.6, 0.6, 5)
        column = ifc.createIfcColumn(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name="Pier_1",
            ObjectPlacement=p,
            Representation=s,
        )
        elements.append(column)

        # Another column
        p, s = create_box(4, 0, 0, 0.6, 0.6, 5)
        column2 = ifc.createIfcColumn(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name="Pier_2",
            ObjectPlacement=p,
            Representation=s,
        )
        elements.append(column2)

        # Slab (deck)
        p, s = create_box(0, -3, 5.8, 12, 6, 0.3)
        slab = ifc.createIfcSlab(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name="Bridge_Deck",
            ObjectPlacement=p,
            Representation=s,
        )
        elements.append(slab)

        # Contain elements in storey
        ifc.createIfcRelContainedInSpatialStructure(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            RelatingStructure=storey,
            RelatedElements=elements,
        )

        ifc.write(output_path)
        logger.info("Test IFC model written to %s", output_path)
        return output_path

    except Exception as exc:
        logger.error("Failed to generate IFC model: %s", exc)
        _write_minimal_ifc_text(output_path)
        return output_path


def _write_minimal_ifc_text(output_path: str) -> None:
    """Write a minimal valid IFC file as plain text (fallback)."""
    ifc_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('test_bridge.ifc','2025-01-01T00:00:00',('SID'),('IDEATOR GECT'),'','SID','');
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCPROJECT('0001',#2,'SID Test Bridge',$,$,$,$,(#10),#11);
#2=IFCOWNERHISTORY(#3,#5,$,.ADDED.,$,$,$,0);
#3=IFCPERSONANDORGANIZATION(#4,#6,$);
#4=IFCPERSON($,'SID','Test',$,$,$,$,$);
#5=IFCAPPLICATION(#6,'1.0','SID','SID');
#6=IFCORGANIZATION($,'IDEATOR_GECT',$,$,$);
#10=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-5,#12,$);
#11=IFCUNITASSIGNMENT((#13,#14,#15));
#12=IFCAXIS2PLACEMENT3D(#16,$,$);
#13=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);
#14=IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.);
#15=IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.);
#16=IFCCARTESIANPOINT((0.,0.,0.));
#20=IFCSITE('0020',#2,'Bridge Site',$,$,$,$,$,$,$,$,$,$,$);
#30=IFCBUILDING('0030',#2,'Test Bridge',$,$,$,$,$,$,$,$,$);
#40=IFCBUILDINGSTOREY('0040',#2,'Deck Level',$,$,$,$,$,$,$);
#50=IFCRELAGGREGATES('0050',#2,$,$,#1,(#20));
#51=IFCRELAGGREGATES('0051',#2,$,$,#20,(#30));
#52=IFCRELAGGREGATES('0052',#2,$,$,#30,(#40));
#100=IFCBEAM('0100',#2,'Main_Girder_1',$,$,$,$,$,$);
#101=IFCCOLUMN('0101',#2,'Pier_1',$,$,$,$,$,$);
#102=IFCCOLUMN('0102',#2,'Pier_2',$,$,$,$,$,$);
#103=IFCSLAB('0103',#2,'Bridge_Deck',$,$,$,$,$,$);
#200=IFCRELCONTAINEDINSPATIALSTRUCTURE('0200',#2,$,$,(#100,#101,#102,#103),#40);
ENDSEC;
END-ISO-10303-21;
"""
    with open(output_path, "w") as f:
        f.write(ifc_content)
    logger.info("Minimal IFC text file written to %s (fallback)", output_path)


# ──────────────────────────────────────────────────────────────────────
#  7. Calibration image with known-size marker
# ──────────────────────────────────────────────────────────────────────

def generate_calibration_image(
    marker_size_px: int = 100,
    marker_real_mm: float = 50.0,
    image_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, float]:
    """Generate an image with a known-size square marker for calibration.

    Args:
        marker_size_px: Marker size in pixels.
        marker_real_mm: Real-world marker size in mm.
        image_size: Output image size.

    Returns:
        (image, known_width_mm)
    """
    h, w = image_size
    rng = np.random.RandomState(33)

    # Concrete-like background
    image = rng.normal(175, 10, (h, w)).clip(0, 255).astype(np.uint8)

    # Draw black square marker in center
    cx, cy = w // 2, h // 2
    half = marker_size_px // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    cv2.rectangle(image, (x1, y1), (x2, y2), 0, -1)  # filled black

    # Add white border
    cv2.rectangle(image, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 255, 3)

    return image, marker_real_mm
