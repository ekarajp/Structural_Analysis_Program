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

from .base import ModelError, q

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
    # stiffness set: auto | gravity | lateral (see stiffness_set_of)
    stiffness: str = "auto"
    # CSI nonlinear static case without geometric nonlinearity: solved iteratively when
    # members have tension / compression limits, otherwise identical to a linear case
    nonlinear: bool = False
    # earthquake cases: deflection amplification Cd and importance factor Ie - the design
    # story drift is Cd / Ie x the elastic drift (ASCE 7 12.8.6, มยผ. 1301/1302-61);
    # None = not given (the slab-column drift check then asks for it)
    cd: Optional[float] = None
    ie: float = 1.0


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
    stiffness: str = "auto"  # stiffness set: auto (= lateral) | gravity | lateral


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
    cd: Optional[float] = None  # deflection amplification (see LinearStaticCase.cd)
    ie: float = 1.0


LATERAL_PATTERN_TYPES = ("Wind", "Seismic", "Notional")
STIFFNESS_SETS = ("auto", "gravity", "lateral")


def stiffness_set_of(model, name: str) -> str:
    """The stiffness set a load case is solved with: "gravity" (the shells' modifiers)
    or "lateral" (their lateral modifiers - a slab that is to take no part in the
    lateral system).  ``stiffness`` auto: modal -> lateral, a response spectrum -> its
    modal case's, a linear static case -> lateral when it loads only lateral patterns
    (wind / seismic / notional) or accelerations in X / Y, P-Delta / nonlinear / time
    history -> gravity."""
    case = model.load_cases[name]
    s = (getattr(case, "stiffness", "auto") or "auto").lower()
    if s not in STIFFNESS_SETS:
        raise ModelError("BAD_STIFFNESS_SET", f"Load case {name!r}: stiffness set {s!r} - use auto, gravity or lateral")
    if s != "auto":
        return s
    if isinstance(case, ModalCase):
        return "lateral"
    if isinstance(case, ResponseSpectrumCase):
        return stiffness_set_of(model, case.modal_case) if case.modal_case in model.load_cases else "lateral"
    if isinstance(case, LinearStaticCase):
        types = {model.load_patterns[p].type for p, f in case.loads.items() if f and p in model.load_patterns}
        acc = {d for d, f in (case.acceleration_loads or {}).items() if f}
        lateral = bool(types & set(LATERAL_PATTERN_TYPES)) or bool(acc & {"UX", "UY"})
        gravity = bool(types - set(LATERAL_PATTERN_TYPES) - {"Temperature", "Other"}) or "UZ" in acc
        return "lateral" if lateral and not gravity else "gravity"
    return "gravity"


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


def spectrum_case_problems(case) -> List[Tuple[str, str]]:
    """(code, message) of the options of a response spectrum case that are unknown or not
    implemented.  They are refused, never replaced by another method."""
    import math
    out: List[Tuple[str, str]] = []
    name = case.name
    mc = str(case.modal_combination).upper().replace(" ", "").replace("%", "PCT")
    if mc not in ("SRSS", "ABS", "ABSSUM", "NRC10", "10PCT", "NRC10PCT", "TENPERCENT", "CQC", "GMC"):
        out.append(("INVALID_COMBINATION", f"Spectrum {name}: unknown modal combination {case.modal_combination!r} "
                                           "(use CQC, SRSS, ABS, GMC or NRC 10%)"))
    dc = str(case.directional_combination).upper().replace(" ", "")
    if dc == "CQC3":
        out.append(("NOT_IMPLEMENTED", f"Spectrum {name}: the CQC3 directional combination is not implemented - "
                                       "use SRSS or ABS"))
    elif dc not in ("SRSS", "ABS", "ABSSUM"):
        out.append(("INVALID_COMBINATION", f"Spectrum {name}: unknown directional combination "
                                           f"{case.directional_combination!r} (use SRSS or ABS)"))
    if not (isinstance(case.damping, (int, float)) and math.isfinite(case.damping) and 0.0 <= case.damping < 1.0):
        out.append(("INVALID_DAMPING", f"Spectrum {name}: modal damping {case.damping!r} must be in [0, 1)"))
    e = case.eccentricity_ratio
    if not (isinstance(e, (int, float)) and math.isfinite(e) and 0.0 <= e <= 0.5):
        out.append(("INVALID_ECCENTRICITY", f"Spectrum {name}: eccentricity ratio {e!r} must be in [0, 0.5]"))
    f = case.directional_abs_factor
    if not (isinstance(f, (int, float)) and math.isfinite(f) and f >= 0):
        out.append(("INVALID_COMBINATION", f"Spectrum {name}: the directional ABS factor must be >= 0"))
    rr = case.rigid_response
    if rr is not None:
        try:
            f1, f2 = float(rr[0]), float(rr[1])
            ok = math.isfinite(f1) and math.isfinite(f2) and 0.0 < f1 < f2
        except (TypeError, ValueError, IndexError):
            f1 = f2 = float("nan")
            ok = False
        if not ok:
            out.append(("INVALID_RIGID_RESPONSE", f"Spectrum {name}: the rigid response frequencies need "
                                                  f"0 < f1 < f2 (got {rr!r} Hz)"))
    elif mc == "GMC":
        out.append(("INVALID_RIGID_RESPONSE", f"Spectrum {name}: the GMC (Gupta) combination needs the rigid "
                                              "response frequencies f1 < f2"))
    for ld in case.loads:
        if str(ld.direction).upper() not in ("U1", "U2", "U3"):
            out.append(("INVALID_DIRECTION", f"Spectrum {name}: unknown direction {ld.direction!r} (U1, U2 or U3)"))
    return out


def spectrum_function_problems(fn) -> List[Tuple[str, str]]:
    import math
    out: List[Tuple[str, str]] = []
    ps, acs = list(fn.periods), list(fn.accelerations)
    if not ps:
        out.append(("INVALID_FUNCTION", f"Spectrum function {fn.name}: no points"))
    if len(ps) != len(acs):
        out.append(("INVALID_FUNCTION", f"Spectrum function {fn.name}: {len(ps)} periods but {len(acs)} values"))
    if any(not math.isfinite(p) or p < 0 for p in ps):
        out.append(("INVALID_FUNCTION", f"Spectrum function {fn.name}: periods must be finite and >= 0"))
    if any(b <= a for a, b in zip(ps, ps[1:])):
        out.append(("INVALID_FUNCTION", f"Spectrum function {fn.name}: the periods must increase strictly "
                                        "(sorted, no repeated period)"))
    if not (math.isfinite(fn.damping) and 0.0 <= fn.damping < 1.0):
        out.append(("INVALID_DAMPING", f"Spectrum function {fn.name}: damping {fn.damping!r} must be in [0, 1)"))
    return out

