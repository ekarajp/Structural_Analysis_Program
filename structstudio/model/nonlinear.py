"""Nonlinear analysis definitions of the object model.

* :class:`NonlinearMaterial`  - uniaxial stress-strain law for fibers
  (Steel01/02, Concrete01/02, ElasticPP, Hysteretic, ...), plus helpers for
  Mander confined concrete and reinforcing steel.
* :class:`FiberSectionDef`    - fiber discretisation of an RC / steel section.
* :class:`FrameNonlinearProperty` - distributed plasticity assignment of a
  frame (force- or displacement-based element, integration rule, plastic
  hinge lengths).
* :class:`HingeProperty` / :class:`FrameHingeAssignment` - CSI-style
  concentrated hinges (backbone B-C-D-E, acceptance IO/LS/CP, hysteresis),
  with ASCE 41 automatic generation for steel and RC beams.
* :class:`LinkProperty` / :class:`LinkObject` - two-joint or grounded links
  (linear, Wen plastic, multilinear plastic, viscous damper).
* Load cases :class:`NonlinearStaticCase` (pushover, staged gravity) and
  :class:`TimeHistoryCase` (linear / nonlinear, modal or direct integration).

All values are in the owning model's units.  Classes whose parameters do not
carry one fixed physical quantity implement ``convert_units``.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import q

# ----------------------------------------------------------------------------
MATERIAL_PARAM_QUANTITY: Dict[str, Dict[str, str]] = {
    "Elastic": {"E": "stress", "eta": "stress", "Eneg": "stress"},
    "ElasticPP": {"E": "stress"},
    "Steel01": {"Fy": "stress", "E0": "stress"},
    "Steel02": {"Fy": "stress", "E0": "stress", "sigini": "stress"},
    "Concrete01": {"fpc": "stress", "fpcu": "stress"},
    "Concrete02": {"fpc": "stress", "fpcu": "stress", "ft": "stress", "Ets": "stress"},
    "Hysteretic": {k: "stress" for k in ("s1p", "s2p", "s3p", "s1n", "s2n", "s3n")},
    "BoucWen": {"ko": "stress"},
    "Viscous": {"C": "stress"},
}


def _conv(value, quantity, src, dst):
    if value is None or quantity is None:
        return value
    return value * src.factor(quantity) / dst.factor(quantity)


@dataclass
class NonlinearMaterial:
    """Uniaxial fiber material: ``type`` is an engine material type, ``params``
    its constructor arguments (stresses in model units, strains dimensionless)."""

    name: str
    type: str = "Steel02"
    params: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def convert_units(self, src, dst) -> "NonlinearMaterial":
        qmap = MATERIAL_PARAM_QUANTITY.get(self.type, {})
        p = {k: _conv(v, qmap.get(k), src, dst) for k, v in self.params.items()}
        return dataclasses.replace(self, params=p)

    def si_params(self, units) -> Dict[str, float]:
        qmap = MATERIAL_PARAM_QUANTITY.get(self.type, {})
        return {k: (v * units.factor(qmap[k]) if k in qmap and v is not None else v) for k, v in self.params.items()}

    # ---- helpers -----------------------------------------------------------
    @classmethod
    def rebar(cls, name: str, fy: float, E: float, b: float = 0.01, R0: float = 18.0) -> "NonlinearMaterial":
        """Reinforcing steel, Giuffre-Menegotto-Pinto (Steel02)."""
        return cls(name, "Steel02", {"Fy": fy, "E0": E, "b": b, "R0": R0, "cR1": 0.925, "cR2": 0.15})

    @classmethod
    def unconfined_concrete(cls, name: str, fc: float, eps_c0: float = 0.002, eps_cu: float = 0.005,
                            ft_ratio: float = 0.0) -> "NonlinearMaterial":
        """Cover concrete: Kent-Park parabola to fc at eps_c0, linear to 0.2 fc at eps_cu
        (Concrete02 with optional tension ft = ft_ratio * fc)."""
        fc = abs(fc)
        Ec = 2 * fc / eps_c0
        p = {"fpc": -fc, "epsc0": -eps_c0, "fpcu": -0.2 * fc, "epscu": -eps_cu, "lam": 0.1,
             "ft": ft_ratio * fc, "Ets": 0.1 * Ec}
        return cls(name, "Concrete02", p)

    @classmethod
    def confined_concrete_mander(cls, name: str, fc: float, fyh: float, rho_s: float, ke: float = 0.75,
                                 eps_c0: float = 0.002, eps_su: float = 0.09, Ec: Optional[float] = None,
                                 ft_ratio: float = 0.0) -> "NonlinearMaterial":
        """Confined core concrete from Mander, Priestley & Park (1988), fitted to the
        Concrete02 envelope: peak (f'cc, eps_cc), linear softening to the Mander stress at
        the ultimate strain eps_cu = 0.004 + 1.4 rho_s fyh eps_su / f'cc.
        ``fc``/``fyh`` in model stress units, ``rho_s`` volumetric ratio of transverse steel."""
        fco = abs(fc)
        fl = 0.5 * ke * rho_s * fyh
        fcc = fco * (-1.254 + 2.254 * math.sqrt(1 + 7.94 * fl / fco) - 2 * fl / fco)
        ecc = eps_c0 * (1 + 5 * (fcc / fco - 1))
        ecu = 0.004 + 1.4 * rho_s * fyh * eps_su / fcc
        Ec = Ec if Ec is not None else 2 * fco / eps_c0
        Esec = fcc / ecc
        r = Ec / (Ec - Esec) if Ec > Esec else 10.0
        x = ecu / ecc
        fcu = fcc * x * r / (r - 1 + x ** r)
        ecu = max(ecu, 1.05 * ecc)
        # unloading ratio: keep the Concrete02 reloading focus point on the tension side
        # (eps_r >= 0), otherwise unloading from the envelope is discontinuous
        Ec0 = 2 * fcc / ecc
        lam = min(max(0.1, 1.1 * fcu / (Ec0 * ecu)), 0.9)
        p = {"fpc": -fcc, "epsc0": -ecc, "fpcu": -fcu, "epscu": -ecu, "lam": lam,
             "ft": ft_ratio * fco, "Ets": 0.1 * 2 * fcc / ecc}
        return cls(name, "Concrete02", p, notes=f"Mander: f'cc={fcc:.4g}, ecc={ecc:.4g}, ecu={ecu:.4g}")

    @classmethod
    def bilinear_steel(cls, name: str, fy: float, E: float, b: float = 0.01) -> "NonlinearMaterial":
        return cls(name, "Steel01", {"Fy": fy, "E0": E, "b": b})


@dataclass
class FiberSectionDef:
    """Fiber section.  ``shape``: RCRectangle | RCCircle | SteelI | SteelBox | SteelPipe | Custom.

    Dimensions default to those of ``frame_section`` (an elastic
    :class:`~structstudio.model.FrameSection`); ``cover`` is measured to the
    longitudinal bar centres.  Custom sections list ``fibers`` as
    ``[y (local 2), z (local 3), area, material]``."""

    name: str
    shape: str = "RCRectangle"
    frame_section: Optional[str] = None
    depth: Optional[float] = q("length", None)  # local 2 (CSI t3)
    width: Optional[float] = q("length", None)  # local 3 (CSI t2)
    diameter: Optional[float] = q("length", None)
    tf: Optional[float] = q("length", None)
    tw: Optional[float] = q("length", None)
    cover: float = q("length", 0.05)
    core_material: Optional[str] = None
    cover_material: Optional[str] = None
    steel_material: Optional[str] = None  # rebar, or the steel of steel shapes
    n_top: int = 0
    n_bottom: int = 0
    n_side: int = 0
    n_bars: int = 0  # circular sections
    bar_area: float = q("area", 0.0)
    side_bar_area: Optional[float] = q("area", None)
    mesh: Tuple[int, int] = (12, 6)
    fibers: List[list] = field(default_factory=list)
    GJ: Optional[float] = q("flexural_rigidity", None)  # None: G J of the elastic frame section

    def convert_units(self, src, dst) -> "FiberSectionDef":
        L = src.factor("length") / dst.factor("length")
        out = dataclasses.replace(
            self,
            depth=None if self.depth is None else self.depth * L, width=None if self.width is None else self.width * L,
            diameter=None if self.diameter is None else self.diameter * L,
            tf=None if self.tf is None else self.tf * L, tw=None if self.tw is None else self.tw * L,
            cover=self.cover * L, bar_area=self.bar_area * L * L,
            side_bar_area=None if self.side_bar_area is None else self.side_bar_area * L * L,
            fibers=[[y * L, z * L, a * L * L, m] for y, z, a, m in self.fibers],
            GJ=_conv(self.GJ, "flexural_rigidity", src, dst))
        return out


@dataclass
class FrameNonlinearProperty:
    """Distributed plasticity of a frame object.

    ``formulation``: ForceBased | DispBased; ``integration``: Lobatto | Legendre |
    Radau | HingeRadau (plastic hinge integration with lengths ``hinge_length_i/j``
    and an elastic interior using the frame's elastic section)."""

    name: str
    section: str
    formulation: str = "ForceBased"
    integration: str = "Lobatto"
    n_points: int = 5
    hinge_length_i: float = q("length", 0.0)
    hinge_length_j: float = q("length", 0.0)
    section_j: Optional[str] = None  # HingeRadau: different section at end J
    max_iterations: int = 10


@dataclass
class HingeProperty:
    """CSI-style hinge property.

    ``points``: backbone B, C, D, E as (plastic deformation, force) on the
    positive side (B normally (0, yield force)); ``points_neg`` for an
    unsymmetric hinge (magnitudes).  When ``force_sf`` / ``deformation_sf`` are
    non-zero the points are normalised (CSI "Moment/SF", "Rotation/SF").
    ``acceptance``: (IO, LS, CP) plastic deformations in the same normalisation.
    ``dof``: M3 | M2 | P.  The hinge is rigid-plastic like CSI hinges: its
    elastic stiffness is ``rigid_factor`` times the member's 6EI/L (EA/L for P).
    Load drops (C-D, after E) are vertical in static cases with
    ``hinge_unloading = "Redistribute"`` (CSI behaviour, released at a fixed
    control displacement) and get a finite slope (``drop_regularisation``) in
    time history and in "Regularised" static cases.
    ``auto``: ASCE41-SteelBeam | ASCE41-RCBeam (points from the section, see
    :mod:`structstudio.engine.nonlinear.hinges`)."""

    name: str
    dof: str = "M3"
    points: List[Tuple[float, float]] = field(default_factory=list)
    points_neg: Optional[List[Tuple[float, float]]] = None
    force_sf: float = 0.0
    deformation_sf: float = 0.0
    acceptance: Optional[Tuple[float, float, float]] = None
    acceptance_neg: Optional[Tuple[float, float, float]] = None
    hysteresis: str = "Kinematic"  # Kinematic | Takeda | Elastic | Pivot-like via Hysteretic
    drop_after_e: bool = True
    residual: float = 0.0
    rigid_factor: float = 1000.0
    # load drops (C-D, after E) are regularised to a slope of k_member / drop_regularisation
    # in time history and in static cases with hinge_unloading = "Regularised"; static
    # cases with "Redistribute" (default) drop vertically as in CSI
    drop_regularisation: float = 10.0
    auto: Optional[str] = None
    auto_params: Dict[str, float] = field(default_factory=dict)
    # dof = "PMM": fiber P-M2-M3 hinge (CSI "fiber P-M2-M3"): the member becomes a
    # force-based element with plastic-hinge integration (Scott & Fenves 2006) using
    # ``fiber_section`` over ``hinge_length`` (0: half the section depth t3)
    fiber_section: Optional[str] = None
    hinge_length: float = q("length", 0.0)

    @property
    def is_pmm(self) -> bool:
        return self.dof.upper().replace("-", "") in ("PMM", "PM2M3", "FIBER")

    def _quantities(self):
        if self.dof.upper() == "P":
            return "force", "length"
        return "moment", None

    def convert_units(self, src, dst) -> "HingeProperty":
        out = self._convert_points(src, dst)
        return dataclasses.replace(out, hinge_length=_conv(self.hinge_length, "length", src, dst))

    def _convert_points(self, src, dst) -> "HingeProperty":
        if self.is_pmm:
            return self
        fq, dq = self._quantities()
        normalized = bool(self.force_sf)
        if normalized:
            return dataclasses.replace(self, force_sf=_conv(self.force_sf, fq, src, dst),
                                       deformation_sf=_conv(self.deformation_sf, dq, src, dst),
                                       auto_params=_conv_auto(self.auto_params, src, dst))

        def cp(pts):
            if pts is None:
                return None
            return [(_conv(d, dq, src, dst), _conv(f, fq, src, dst)) for d, f in pts]

        def ca(a):
            return None if a is None else tuple(_conv(v, dq, src, dst) for v in a)

        return dataclasses.replace(self, points=cp(self.points), points_neg=cp(self.points_neg),
                                   acceptance=ca(self.acceptance), acceptance_neg=ca(self.acceptance_neg),
                                   auto_params=_conv_auto(self.auto_params, src, dst))


_AUTO_QUANTITY = {"fy": "stress", "fc": "stress", "My": "moment", "Py": "force", "V": "force", "d": "length"}


def _conv_auto(p, src, dst):
    return {k: _conv(v, _AUTO_QUANTITY.get(k), src, dst) for k, v in p.items()}


@dataclass
class FrameHingeAssignment:
    property: str
    relative_distance: float = 0.0  # 0 = end I, 1 = end J (snapped to the nearest element end)


# ----------------------------------------------------------------------------
@dataclass
class LinkDOF:
    """One DOF of a link property.  ``dof``: U1 U2 U3 R1 R2 R3 (link local axes).

    type: Linear (k, c) | Wen (k, yield_force, post_yield_ratio, exponent) |
    MultiLinear (points [(deformation, force)], hysteresis) | Damper (c, exponent) |
    ElasticPP (k, yield_force)."""

    dof: str = "U1"
    type: str = "Linear"
    k: float = 0.0
    c: float = 0.0
    yield_force: float = 0.0
    post_yield_ratio: float = 0.0
    exponent: float = 1.0
    points: List[Tuple[float, float]] = field(default_factory=list)
    hysteresis: str = "Kinematic"

    def _q(self):
        rot = self.dof.upper().startswith("R")
        return ("rotational_stiffness" if rot else "translational_stiffness", "moment" if rot else "force",
                None if rot else "length")

    def convert_units(self, src, dst) -> "LinkDOF":
        kq, fq, dq = self._q()
        # damping coefficient: force / velocity^exponent (time is unchanged)
        fr = src.factor(fq) / dst.factor(fq)
        lr = src.factor(dq) / dst.factor(dq) if dq else 1.0
        c = self.c * fr / lr ** (self.exponent if self.type == "Damper" else 1.0)
        return dataclasses.replace(self, k=_conv(self.k, kq, src, dst), c=c,
                                   yield_force=_conv(self.yield_force, fq, src, dst),
                                   points=[(_conv(d, dq, src, dst), _conv(f, fq, src, dst)) for d, f in self.points])


@dataclass
class LinkProperty:
    name: str
    dofs: List[LinkDOF] = field(default_factory=list)
    mass: float = q("mass", 0.0)


@dataclass
class LinkObject:
    """Link between two points (``point_j``) or from a point to the ground.
    Axis 1 runs from I to J (two-joint links) or along global Z (grounded links);
    axis 2 follows the CSI frame rule; ``angle`` rotates 2-3 about 1."""

    name: str
    point_i: str
    point_j: Optional[str]
    property: str
    angle: float = 0.0


# ----------------------------------------------------------------------------
@dataclass
class NonlinearStaticCase:
    """Nonlinear static case (gravity staging, pushover).

    Loads: ``loads`` (pattern -> scale), ``acceleration_loads`` ("UX" -> scale of
    g, uniform pushover), ``mode_loads`` ((modal case, mode, scale): CSI "Mode"
    load, force pattern M phi).  ``control`` = "Load" applies the full load in
    ``n_steps``; "Displacement" pushes ``monitored_point`` / ``monitored_dof`` to
    ``target_displacement`` (displacement control, the load is the unknown).
    ``initial_case``: continue from the end of another nonlinear case (its loads
    stay applied)."""

    name: str
    loads: Dict[str, float] = field(default_factory=dict)
    acceleration_loads: Dict[str, float] = field(default_factory=dict)
    mode_loads: List[Tuple[str, int, float]] = field(default_factory=list)
    initial_case: Optional[str] = None
    geometric_nonlinearity: str = "PDelta"  # None | PDelta | Corotational (large displacements)
    control: str = "Load"
    monitored_point: Optional[str] = None
    monitored_dof: str = "UX"
    target_displacement: float = q("length", 0.0)
    n_steps: int = 10
    max_iterations: int = 25
    tolerance: float = 1e-6
    max_subdivisions: int = 10
    save_steps: str = "Multiple"  # Multiple | Final
    stop_on_failure: bool = False  # False: keep the steps that converged, flag the case
    # hinge strength loss (C-D, after E): "Redistribute" = vertical drops as in CSI; a
    # drop is released in sub-steps at a fixed control displacement so the structure
    # sheds the load (vertical step on the capacity curve).  "Regularised" = the drop
    # gets a finite slope of k_member / drop_regularisation (always used in time history)
    hinge_unloading: str = "Redistribute"
    type: str = "NonlinearStatic"


@dataclass
class TimeHistoryLoad:
    """``type`` = Acceleration: ``name`` is a direction (U1 U2 U3 rotated by
    ``angle``, or UX UY UZ) and ``scale`` an acceleration (model units, e.g. g
    for a function in g); ``type`` = Load: ``name`` is a load pattern and
    ``scale`` a plain multiplier."""

    type: str = "Acceleration"
    name: str = "U1"
    function: str = ""
    scale: float = 1.0
    angle: float = 0.0

    def convert_units(self, src, dst) -> "TimeHistoryLoad":
        if self.type.lower().startswith("acc"):
            return dataclasses.replace(self, scale=_conv(self.scale, "acceleration", src, dst))
        return self


@dataclass
class TimeHistoryCase:
    """Linear or nonlinear time history.

    ``method``: DirectIntegration (Newmark / HHT, nonlinear when ``nonlinear``),
    Modal (linear modal superposition, exact for piecewise linear loading,
    needs ``modal_case``) or FNA (fast nonlinear analysis, Wilson: linear
    structure + nonlinear links on a basis of ``max_modes`` eigenvectors of
    ``modal_case`` (12 without one) plus load-dependent Ritz vectors of the
    link deformations and ground loads; frame hinges are linear in FNA).  Damping: ``damping`` = Rayleigh (``damping_ratio``
    at ``damping_periods``, or explicit ``mass_coefficient`` /
    ``stiffness_coefficient``), Modal (constant ``damping_ratio`` on the modes
    of ``modal_case``) or None."""

    name: str
    loads: List[TimeHistoryLoad] = field(default_factory=list)
    method: str = "DirectIntegration"
    nonlinear: bool = True
    n_steps: int = 100
    dt: float = 0.01
    initial_case: Optional[str] = None
    modal_case: Optional[str] = None
    damping: str = "Rayleigh"
    damping_ratio: float = 0.05
    damping_periods: Optional[Tuple[float, float]] = None
    mass_coefficient: Optional[float] = None
    stiffness_coefficient: Optional[float] = None
    rayleigh_stiffness: str = "initial"
    integrator: str = "Newmark"
    gamma: float = 0.5
    beta: float = 0.25
    alpha: float = 1.0
    geometric_nonlinearity: str = "PDelta"
    max_iterations: int = 25
    tolerance: float = 1e-6
    max_subdivisions: int = 8
    monitor_points: List[str] = field(default_factory=list)
    save_every: int = 1
    type: str = "TimeHistory"


NONLINEAR_TYPES = (NonlinearMaterial, FiberSectionDef, FrameNonlinearProperty, HingeProperty, FrameHingeAssignment,
                   LinkDOF, LinkProperty, LinkObject, NonlinearStaticCase, TimeHistoryLoad, TimeHistoryCase)
