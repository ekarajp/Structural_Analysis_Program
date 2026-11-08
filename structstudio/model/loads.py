"""Load patterns, load cases, load combinations, functions and mass source.

Terminology (as in CSI products):

* **Load pattern** - a named *spatial* distribution of loads (dead, live ...)
  assigned to objects.  A pattern can also carry self weight via a multiplier.
* **Load case** - *how* patterns are applied and analysed (linear static,
  modal, response spectrum, P-Delta ...).
* **Load combination** - a post-processing combination of case results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import q

PATTERN_TYPES = (
    "Dead", "SuperDead", "Live", "ReducibleLive", "RoofLive", "Snow", "Wind", "Seismic",
    "Temperature", "Notional", "Other",
)


@dataclass
class LoadPattern:
    name: str
    type: str = "Other"
    self_weight_multiplier: float = 0.0
    auto_load: Optional[dict] = None  # code-generated lateral load definition (later phases)


@dataclass
class MassSource:
    """Which mass is used for dynamic analyses (CSI "Mass Source")."""

    name: str = "MsSrc1"
    include_element_mass: bool = True
    include_added_mass: bool = True
    include_lateral: bool = True
    include_vertical: bool = False  # CSI/ETABS default: lateral mass only
    lump_at_stories: bool = True
    load_patterns: Dict[str, float] = field(default_factory=dict)  # pattern -> multiplier (mass from loads)
    is_default: bool = True


@dataclass
class ResponseSpectrumFunction:
    """Pseudo-acceleration spectrum: period (s) vs acceleration.

    Accelerations are stored as *multiples of g* unless ``normalized`` is
    False, in which case they are in the model's acceleration units.
    """

    name: str
    periods: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)
    damping: float = 0.05
    normalized: bool = True
    # the analysis always uses accelerations x the load's scale factor, and the
    # scale factor carries the acceleration unit (it is converted with the model):
    # the function values themselves are never converted

    def value(self, period: float) -> float:
        """Linear interpolation; constant extrapolation (CSI convention)."""
        ps, acs = self.periods, self.accelerations
        if not ps:
            raise ValueError(f"Spectrum function {self.name!r} has no points")
        if period <= ps[0]:
            return acs[0]
        if period >= ps[-1]:
            return acs[-1]
        for k in range(1, len(ps)):
            if period <= ps[k]:
                t0, t1 = ps[k - 1], ps[k]
                a0, a1 = acs[k - 1], acs[k]
                return a0 + (a1 - a0) * (period - t0) / (t1 - t0)
        return acs[-1]


@dataclass
class TimeHistoryFunction:
    name: str
    times: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    kind: str = "acceleration"


# ----------------------------------------------------------------------------
# Load cases
# ----------------------------------------------------------------------------
@dataclass
class LinearStaticCase:
    name: str
    loads: Dict[str, float] = field(default_factory=dict)  # pattern -> scale factor
    acceleration_loads: Dict[str, float] = field(default_factory=dict)  # "UX"/"UY"/"UZ" -> factor of g
    stiffness_from: Optional[str] = None  # name of nonlinear case providing stiffness (P-Delta)
    design_type: Optional[str] = None
    type: str = "LinearStatic"
    # CSI nonlinear static case without geometric nonlinearity: solved iteratively when
    # members have tension / compression limits, otherwise identical to a linear case
    nonlinear: bool = False


@dataclass
class ModalCase:
    name: str
    max_modes: int = 12
    min_modes: int = 1
    method: str = "Eigen"  # Eigen | Ritz
    mass_source: Optional[str] = None
    shift: float = 0.0  # frequency shift (cyc/s)
    cutoff: float = 0.0  # cutoff frequency, 0 = none
    tolerance: float = 1e-9
    stiffness_from: Optional[str] = None
    ritz_vectors: List[str] = field(default_factory=lambda: ["UX", "UY", "UZ"])
    type: str = "Modal"


@dataclass
class SpectrumLoad:
    direction: str  # U1 | U2 | U3 (in the case's rotated coordinate system)
    function: str
    scale: float = q("acceleration", 1.0)  # multiplies the (normalized) function; usually g


@dataclass
class ResponseSpectrumCase:
    name: str
    modal_case: str
    loads: List[SpectrumLoad] = field(default_factory=list)
    modal_combination: str = "CQC"  # CQC | SRSS | ABS | GMC
    directional_combination: str = "SRSS"  # SRSS | ABS | CQC3
    directional_abs_factor: float = 1.0
    damping: float = 0.05  # constant modal damping ratio
    angle: float = 0.0  # rotation of U1/U2 from global X/Y (degrees)
    eccentricity_ratio: float = 0.0  # accidental eccentricity (later)
    rigid_response: Optional[Tuple[float, float]] = None  # (f1, f2) Gupta; None = all periodic
    type: str = "ResponseSpectrum"


@dataclass
class PDeltaCase:
    """Nonlinear static (P-Delta) case: geometric stiffness from the applied
    gravity loads; the resulting stiffness can be used by other cases."""

    name: str
    loads: Dict[str, float] = field(default_factory=dict)
    max_iterations: int = 40
    tolerance: float = 1e-8
    type: str = "PDelta"


@dataclass
class LinearBucklingCase:
    name: str
    loads: Dict[str, float] = field(default_factory=dict)
    modes: int = 6
    type: str = "Buckling"


LoadCase = LinearStaticCase | ModalCase | ResponseSpectrumCase | PDeltaCase | LinearBucklingCase


@dataclass
class ComboItem:
    name: str  # case or combo name
    scale: float = 1.0
    is_combo: bool = False


@dataclass
class LoadCombination:
    name: str
    type: str = "LinearAdd"  # LinearAdd | Envelope | Absolute | SRSS | Range
    items: List[ComboItem] = field(default_factory=list)
    is_auto: bool = False
    notes: str = ""

    def add(self, name: str, scale: float = 1.0, is_combo: bool = False) -> "LoadCombination":
        self.items.append(ComboItem(name, scale, is_combo))
        return self
