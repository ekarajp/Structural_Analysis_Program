"""Design preferences stored with the model (Design menu)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

SLAB_COLUMN_CODES = ("ACI 318-19", "ACI 318-14")


@dataclass
class SlabColumnDriftPrefs:
    """Design > Concrete Slab Design > Slab-Column Drift Check (ACI 318 18.14.5).

    ``combos``: the load combinations giving the slab shear stress (318-19: vuv, the
    combinations with E; 318-14: vug, gravity) - empty = the default of the code edition.
    ``seismic_cases``: the earthquake cases whose drift is the design drift - empty = all."""

    code: str = "ACI 318-19"
    phi: float = 0.75  # strength reduction factor for shear (ACI 318 21.2.1)
    lam: float = 1.0  # lightweight concrete factor lambda
    combos: List[str] = field(default_factory=list)
    seismic_cases: List[str] = field(default_factory=list)


@dataclass
class DesignPreferences:
    slab_column_drift: SlabColumnDriftPrefs = field(default_factory=SlabColumnDriftPrefs)
