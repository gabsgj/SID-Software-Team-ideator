"""
BIM Mapping Module
==================
Maps detected cracks onto IFC bridge models using IfcOpenShell.
Supports IfcSurfaceFeature (IFC 4.3) and PropertySet-based annotation.
"""

from bim_mapping.mapper import CrackBIMMapper
from bim_mapping.ifc_annotator import IFCAnnotator
from bim_mapping.coordinate_transform import CoordinateTransformer
from bim_mapping.visualizer import BIMVisualizer

__all__ = [
    "CrackBIMMapper",
    "IFCAnnotator",
    "CoordinateTransformer",
    "BIMVisualizer",
]
