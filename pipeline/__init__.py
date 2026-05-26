"""
Inspection Pipeline
===================
End-to-end orchestration: Image → Detection → Width → Severity → BIM → Report
"""

from pipeline.inspector import BridgeInspector
from pipeline.report_generator import ReportGenerator

__all__ = [
    "BridgeInspector",
    "ReportGenerator",
]
