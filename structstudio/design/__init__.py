"""Design checks that work on analysis results (Design menu)."""
from .slab_column_drift import SlabColumnCheck, SlabColumnReport, check_slab_column_drift, earthquake_cases

__all__ = ["SlabColumnCheck", "SlabColumnReport", "check_slab_column_drift", "earthquake_cases"]
