"""
SID — Structural Inspection Drone
Crack Detection, Width Estimation & BIM Mapping Module
IDEATOR GECT · Government Engineering College Thrissur
"""
from setuptools import setup, find_packages

setup(
    name="sid-crack-detection",
    version="1.0.0",
    description="Automated crack detection, width estimation, and BIM mapping for bridge inspection",
    author="IDEATOR GECT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8",
        "numpy>=1.24",
        "scikit-image>=0.21",
        "scipy>=1.11",
        "pydantic>=2.0",
        "ifcopenshell>=0.7",
        "matplotlib>=3.7",
        "plotly>=5.15",
        "Pillow>=10.0",
        "flask>=3.0",
    ],
    extras_require={
        "test": ["pytest>=7.4"],
        "stereo": ["pyzed"],  # ZED SDK Python wrapper
    },
)
