"""Structural Studio - object-based building analysis & design engine."""
from .api import analyze
from .model import Model, ModelError
from .units import UnitSystem

__version__ = "0.1.0"
__all__ = ["Model", "ModelError", "UnitSystem", "analyze", "__version__"]
