"""Structural *objects*: points, frames (beam/column/brace) and areas
(floor/wall/ramp) together with their assignments and object loads.

Objects are what the engineer draws.  The mesher converts them into finite
elements (see :mod:`structstudio.mesh`).  This separation mirrors ETABS/SAFE
"object based modelling" and is what makes auto-meshing, slab design and
drawing generation possible later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .base import q, qlist

Vec3 = Tuple[float, float, float]
Six = Tuple[float, float, float, float, float, float]
SixBool = Tuple[bool, bool, bool, bool, bool, bool]

DOF_NAMES = ("UX", "UY", "UZ", "RX", "RY", "RZ")
LOCAL_DOF_NAMES = ("U1", "U2", "U3", "R1", "R2", "R3")
FIXED: SixBool = (True, True, True, True, True, True)
PINNED: SixBool = (True, True, True, False, False, False)
FREE: SixBool = (False, False, False, False, False, False)
ZERO6: Six = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ----------------------------------------------------------------------------
# Object loads
# ----------------------------------------------------------------------------
@dataclass
class PointLoad:
    """Concentrated force/moment at a point object (global axes)."""

    pattern: str
    force: Vec3 = qlist("force", lambda: (0.0, 0.0, 0.0))
    moment: Vec3 = qlist("moment", lambda: (0.0, 0.0, 0.0))


@dataclass
class FrameLoad:
    """Distributed or concentrated load on a frame object.

    ``direction``: ``"Gravity"`` (global -Z), ``"X"``, ``"Y"``, ``"Z"`` (global),
    ``"1"``, ``"2"``, ``"3"`` (element local), or ``"GravityProjected"``,
    ``"XProjected"`` ... for projected loads.

    Distributed loads are trapezoidal between relative distances ``rel_a`` and
    ``rel_b`` (0..1 of the element length) with intensities ``value_a`` and
    ``value_b``.  Set ``absolute=True`` to interpret the distances as lengths
    from end I.  Point loads use ``rel_a``/``value_a`` only.
    """

    pattern: str
    kind: str = "distributed"  # distributed | point | temperature | gravity (value = self-weight multiplier)
    direction: str = "Gravity"
    # Force | Moment; for kind "temperature": Temperature (uniform change, deg) |
    # Gradient2 | Gradient3 (deg / length across local 2 / 3); value_a / value_b at ends I / J
    load_type: str = "Force"
    rel_a: float = 0.0
    rel_b: float = 1.0
    absolute: bool = False
    value_a: float = 0.0
    value_b: float = 0.0

    def __post_init__(self):
        if self.kind == "point":
            self.rel_b = self.rel_a
            self.value_b = self.value_a

    def quantity(self) -> Optional[str]:
        """Physical quantity of ``value_a`` / ``value_b`` (it depends on the kind)."""
        if self.kind == "temperature":
            return {"Temperature": "temperature_change", "Gradient2": "temperature_gradient",
                    "Gradient3": "temperature_gradient"}.get(self.load_type)
        if self.kind == "gravity":
            return None  # a self-weight multiplier
        if self.kind == "point":
            return "moment" if self.load_type == "Moment" else "force"
        return "moment_per_length" if self.load_type == "Moment" else "force_per_length"

    def convert_units(self, src, dst) -> "FrameLoad":
        import dataclasses as _dc
        qv = self.quantity()
        rv = src.factor(qv) / dst.factor(qv) if qv else 1.0
        rd = src.factor("length") / dst.factor("length") if self.absolute else 1.0
        return _dc.replace(self, value_a=self.value_a * rv, value_b=self.value_b * rv,
                           rel_a=self.rel_a * rd, rel_b=self.rel_b * rd)


@dataclass
class GroundDisplacement:
    """Prescribed support displacement (CSI "ground displacement") of a load
    pattern at a joint: imposed on its restrained DOFs, and acting through its
    springs.  Components in the joint local axes when the joint has them."""

    pattern: str
    translations: Vec3 = qlist("length", lambda: (0.0, 0.0, 0.0))
    rotations: Vec3 = (0.0, 0.0, 0.0)
    coord_sys: str = "Global"  # "Global" or "Local" (joint local axes)


@dataclass
class AreaLoad:
    """Uniform load on an area object.

    ``direction`` as in :class:`FrameLoad` (``"Gravity"``, global ``"X/Y/Z"``,
    local ``"3"`` for pressure normal to the surface).  Value is force/area.
    """

    pattern: str
    value: float = q("force_per_area", 0.0)
    direction: str = "Gravity"


# ----------------------------------------------------------------------------
# Objects
# ----------------------------------------------------------------------------
@dataclass
class Point:
    """Point (joint) object.

    ``name`` is unique in the model.  ``label``/``story`` are the ETABS style
    display handles (a label such as ``"C1"`` may repeat on every story).
    """

    name: str
    x: float = q("length", 0.0)
    y: float = q("length", 0.0)
    z: float = q("length", 0.0)
    label: Optional[str] = None
    story: Optional[str] = None
    restraints: SixBool = FREE
    springs: Tuple[float, float, float] = qlist("translational_stiffness", lambda: (0.0, 0.0, 0.0))  # UX,UY,UZ
    rotational_springs: Tuple[float, float, float] = qlist("rotational_stiffness", lambda: (0.0, 0.0, 0.0))  # RX,RY,RZ
    mass: Tuple[float, float, float] = qlist("mass", lambda: (0.0, 0.0, 0.0))
    rotational_mass: Tuple[float, float, float] = qlist("rotational_mass", lambda: (0.0, 0.0, 0.0))
    diaphragm: Optional[str] = None  # explicit diaphragm assignment
    panel_zone: Optional[str] = None
    loads: List[PointLoad] = field(default_factory=list)
    is_auto: bool = False  # created by meshing / auto-generation
    # joint local axes (CSI angles a, b, c in degrees): restraints, springs and ground
    # displacements act in these axes; None = global
    local_axes: Optional[Tuple[float, float, float]] = None
    ground_displacements: List[GroundDisplacement] = field(default_factory=list)

    @property
    def xyz(self) -> Vec3:
        return (self.x, self.y, self.z)


@dataclass
class Frame:
    """Line object: beam, column or brace."""

    name: str
    point_i: str
    point_j: str
    section: str
    type: str = "Beam"  # Beam | Column | Brace | Null
    label: Optional[str] = None
    story: Optional[str] = None
    angle: float = 0.0  # local axes rotation (degrees) about local 1
    releases_i: SixBool = FREE  # U1,U2,U3,R1,R2,R3 released at end I
    releases_j: SixBool = FREE
    # partial fixity: spring stiffness for a released DOF (0 = full release)
    fixity_i_trans: Tuple[float, float, float] = qlist("translational_stiffness", lambda: (0.0, 0.0, 0.0))
    fixity_i_rot: Tuple[float, float, float] = qlist("rotational_stiffness", lambda: (0.0, 0.0, 0.0))
    fixity_j_trans: Tuple[float, float, float] = qlist("translational_stiffness", lambda: (0.0, 0.0, 0.0))
    fixity_j_rot: Tuple[float, float, float] = qlist("rotational_stiffness", lambda: (0.0, 0.0, 0.0))
    end_offset_i: float = q("length", 0.0)
    end_offset_j: float = q("length", 0.0)
    # rigid-zone factor applied to end offsets; None = the model default for the
    # member's material (Analyze > Analysis Model Defaults: concrete 0.5, steel 0)
    rigid_factor: Optional[float] = None
    # compute offsets from connected member sizes; None = the model default (on)
    auto_end_offsets: Optional[bool] = None
    cardinal_point: int = 10  # 10 = centroid (CSI numbering); used by the analysis
    # cardinal point for drawing only (ETABS draws beams at "8 top centre" without
    # transforming the stiffness); None = cardinal_point, or 8 for beams in story models
    display_cardinal: Optional[int] = None
    joint_offset_i: Vec3 = qlist("length", lambda: (0.0, 0.0, 0.0))  # global
    joint_offset_j: Vec3 = qlist("length", lambda: (0.0, 0.0, 0.0))
    joint_offset_local: bool = False  # joint offsets given in the member local axes
    modifiers: Optional["FrameModifiers"] = None  # object level, multiplies section modifiers
    mass_per_length: float = q("mass_per_length", 0.0)  # additional (object level)
    tension_limit: Optional[float] = q("force", None)
    compression_limit: Optional[float] = q("force", None)
    auto_mesh: bool = True
    mesh_at_intersections: bool = True
    mesh_at_floor: bool = True  # split at floor mesh nodes along the member
    max_station_spacing: Optional[float] = q("length", None)
    min_stations: int = 3
    pier: Optional[str] = None
    spandrel: Optional[str] = None
    loads: List[FrameLoad] = field(default_factory=list)
    group: Optional[str] = None
    # elastic foundation along the member (CSI "frame springs", Winkler): stiffness per
    # unit length in local 1, 2, 3 (force / length / length), consistent element stiffness
    foundation: Vec3 = qlist("force_per_area", lambda: (0.0, 0.0, 0.0))
    # nonlinear analysis: distributed plasticity property and / or concentrated hinges
    nonlinear: Optional[str] = None
    hinges: list = field(default_factory=list)  # FrameHingeAssignment

    def __post_init__(self):
        # keep dataclass annotations light: import here to avoid cycles
        from .properties import FrameModifiers  # noqa: F401

    @property
    def is_column(self) -> bool:
        return self.type == "Column"


@dataclass
class Area:
    """Area object: floor slab, wall, ramp or opening."""

    name: str
    points: List[str]
    section: Optional[str] = None  # None => opening / null
    type: str = "Floor"  # Floor | Wall | Ramp | Null | Opening
    label: Optional[str] = None
    story: Optional[str] = None
    diaphragm: Optional[str] = None
    angle: float = 0.0  # local axes rotation (degrees)
    modifiers: Optional["ShellModifiers"] = None
    mass_per_area: float = q("mass_per_area", 0.0)
    auto_mesh: bool = True
    mesh_type: str = "Rectangular"  # Rectangular | General | None
    max_mesh_size: Optional[float] = q("length", None)  # manual maximum mesh size
    # use the fine mesh found by Analyze > Find Fine Mesh (model.options.fine_mesh_factor)
    fine_mesh: bool = False
    mesh_divisions: Optional[Tuple[int, int]] = None  # explicit n1 x n2 for quads
    edge_releases: Optional[List[bool]] = None
    # area spring (e.g. soil subgrade modulus, SAFE / ETABS "area springs") normal to the
    # area, force per length^3; distributed to the mesh nodes by tributary area
    spring: float = q("weight_per_volume", 0.0)
    pier: Optional[str] = None
    spandrel: Optional[str] = None
    loads: List[AreaLoad] = field(default_factory=list)
    group: Optional[str] = None

    @property
    def is_opening(self) -> bool:
        return self.section is None or self.type in ("Opening", "Null")


@dataclass
class TendonPoint:
    """A point of a tendon path: ``kind`` "Start", "Line" (straight to here),
    "ParabolaIntermediate" / "ParabolaEnd" (parabola through both)."""

    kind: str
    xyz: Vec3 = qlist("length", lambda: (0.0, 0.0, 0.0))  # global


@dataclass
class TendonLoad:
    """Post-tensioning of a tendon in a load pattern: jacking ``force`` with friction
    (``curvature`` = mu per radian, ``wobble`` = K per model length unit) and
    stress losses (anchorage set as stress, elastic shortening, creep, shrinkage,
    relaxation) subtracted uniformly, as in SAP2000 "Tendon Loads"."""

    pattern: str
    force: float = q("force", 0.0)
    jack_from: str = "I"  # I | J | Both
    curvature: float = 0.0
    wobble: float = q("inverse_length", 0.0)  # 1 / model length
    loss_stress: float = q("stress", 0.0)  # sum of uniform stress losses


@dataclass
class Tendon:
    """Prestressing tendon bonded to frame objects.  ``model_as``: "Loads" (the
    tendon acts on the frames through equivalent loads only) or "Elements" (the
    tendon is also a bonded axial member that adds stiffness and weight)."""

    name: str
    path: List[TendonPoint] = field(default_factory=list)
    area: float = q("area", 0.0)
    material: Optional[str] = None
    model_as: str = "Loads"
    max_discretization: Optional[float] = q("length", None)
    frames: List[str] = field(default_factory=list)  # bonded frames (empty: nearest frames)
    loads: List[TendonLoad] = field(default_factory=list)


@dataclass
class PanelLoadCase:
    """Load of a panel in one load pattern.  ``direction``: "Gravity" (global -Z),
    "X" / "Y" / "Z" (global), "Normal" (the panel normal, e.g. wind pressure) or
    "GravityProjected"...  Uniform (``value``) or varying linearly with height
    (``value_lower`` at the panel's lowest point, ``value_upper`` at its highest,
    as the CYPE 3D "vary depending on the height" loads)."""

    pattern: str
    value: float = q("force_per_area", 0.0)
    direction: str = "Gravity"
    varying: bool = False
    value_lower: float = q("force_per_area", 0.0)
    value_upper: float = q("force_per_area", 0.0)


@dataclass
class PanelLoad:
    """A *load panel*: a planar polygon that carries surface loads to the frames
    (and shells) under it without adding stiffness (CYPE 3D panels / surface
    loads, with a two-way option).

    ``distribution``: "one_way" (strips parallel to ``span_dir``, simply
    supported on every crossing bar), "two_way" (nearest-bar tributary areas,
    the ETABS / SAFE 45-degree split) or "shells" (pressure on the overlapping
    area objects).  ``span_dir``: global direction of the one-way span (set from
    an edge of the panel in the UI).  ``receivers``: names of the frames that may
    receive load (None = every frame in the panel, CYPE "Assign" option)."""

    name: str
    points: List[Vec3] = field(default_factory=list)  # global coordinates of the polygon
    distribution: str = "one_way"
    span_dir: Optional[Vec3] = None
    loads: List[PanelLoadCase] = field(default_factory=list)
    receivers: Optional[List[str]] = None
    story: Optional[str] = None
    label: Optional[str] = None
    group: Optional[str] = None
    # the area object the panel was made from (an *area load*, Load menu): the
    # panel follows that area's polygon and, with distribution "shells", loads only
    # that area (the ETABS / SAP area load); "one_way" / "two_way" carry the load to
    # the frames around it
    area: Optional[str] = None


@dataclass
class LineSpring:
    """Distributed spring support along a line (SAFE / ETABS "line springs",
    SAP2000 frame springs): stiffness per unit length in global X, Y, Z and a
    rotational stiffness per unit length about the line.  Distributed to the
    analysis nodes on the line by tributary length."""

    name: str
    point_i: str
    point_j: str
    stiffness: Vec3 = qlist("force_per_area", lambda: (0.0, 0.0, 0.0))  # F/L per unit length
    rotational: float = q("force", 0.0)  # F·L/rad per unit length


# ----------------------------------------------------------------------------
# Building organisation
# ----------------------------------------------------------------------------
@dataclass
class Story:
    name: str
    elevation: float = q("length", 0.0)
    height: float = q("length", 0.0)
    is_master: bool = False
    similar_to: Optional[str] = None
    color: Optional[str] = None


@dataclass
class GridLine:
    """A grid line in a Cartesian grid system."""

    system: str
    name: str
    direction: str  # X | Y | General | Z (Z = elevation grid line, SAP2000 grid mode)
    ordinate: float = q("length", 0.0)
    start: Optional[Tuple[float, float]] = qlist("length", lambda: None)  # for General lines
    end: Optional[Tuple[float, float]] = qlist("length", lambda: None)
    bubble_location: str = "End"
    visible: bool = True


@dataclass
class GridSystem:
    name: str = "G1"
    type: str = "Cartesian"
    origin: Tuple[float, float] = qlist("length", lambda: (0.0, 0.0))
    rotation: float = 0.0  # degrees
    bubble_size: float = q("length", 1.25)


@dataclass
class Diaphragm:
    name: str
    rigid: bool = True  # False => semi-rigid (meshed membrane)


@dataclass
class Group:
    name: str
    points: List[str] = field(default_factory=list)
    frames: List[str] = field(default_factory=list)
    areas: List[str] = field(default_factory=list)
