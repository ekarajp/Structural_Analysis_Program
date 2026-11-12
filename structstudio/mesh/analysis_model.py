"""Finite element *analysis model* produced by the mesher.

Everything here is in SI (m, N, kg, s).  The analysis model is deliberately
simple data (numpy arrays and dataclasses) so that a compiled kernel can be
dropped in later without touching the object model or the mesher.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Node:
    id: int
    xyz: np.ndarray
    restraints: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=bool))
    springs: np.ndarray = field(default_factory=lambda: np.zeros(6))  # N/m, N·m/rad (global)
    mass: np.ndarray = field(default_factory=lambda: np.zeros(6))  # kg, kg·m² (global, diagonal)
    point: Optional[str] = None  # source point object name (None for mesh-generated nodes)
    story: Optional[str] = None
    diaphragm: Optional[str] = None  # diaphragm name this node belongs to (if any)
    is_master: bool = False  # virtual master node of a constraint
    # joint local axes (rows = local 1, 2, 3 in global): restraints and springs act in them
    local_axes: Optional[np.ndarray] = None
    ground_disp: Dict[str, np.ndarray] = field(default_factory=dict)  # pattern -> (6,) SI, node axes


@dataclass
class ElementLoad:
    """A load on a frame element, resolved into the element local axes.

    ``x_a``..``x_b`` are absolute distances from the element's end I node
    (0..L).  ``values`` are the load intensities (force or moment per length
    for distributed, total force/moment for point) in the *local* 1-2-3 axes
    at ``x_a`` (``value_a``) and ``x_b`` (``value_b``).  Point loads have
    ``x_a == x_b``.
    """

    pattern: str
    kind: str  # distributed | point | temperature
    load_type: str  # Force | Moment | Temperature | Gradient2 | Gradient3
    x_a: float
    x_b: float
    value_a: np.ndarray  # local (3,)
    value_b: np.ndarray


@dataclass
class FrameElement:
    id: int
    object: str  # parent frame object name
    node_i: int
    node_j: int
    E: float
    G: float
    A: float
    As2: float  # 0 => shear deformation neglected
    As3: float
    J: float
    I22: float
    I33: float
    mass_per_length: float  # kg/m (incl. modifiers & additional)
    weight_per_length: float  # N/m (incl. modifiers & additional), acts in -Z
    axes: np.ndarray  # (3,3) rows = unit vectors of local 1,2,3 in global
    releases_i: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=bool))
    releases_j: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=bool))
    fixity_i: np.ndarray = field(default_factory=lambda: np.zeros(6))  # partial fixity springs (0 = free)
    fixity_j: np.ndarray = field(default_factory=lambda: np.zeros(6))
    offset_i: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rigid link node->element end (global)
    offset_j: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rigid_i: float = 0.0  # rigid zone length at end I (bending/shear only), along axis
    rigid_j: float = 0.0
    end_offset_i: float = 0.0  # full end offset (for output stations)
    end_offset_j: float = 0.0
    rel_start: float = 0.0  # position of this element within the parent object (0..1)
    rel_end: float = 1.0
    loads: List[ElementLoad] = field(default_factory=list)
    stations: int = 3  # number of output stations along the element
    tension_limit: Optional[float] = None
    compression_limit: Optional[float] = None
    foundation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # N/m per m, local 1 2 3
    alpha: float = 0.0  # coefficient of thermal expansion (1/deg)
    # nonprismatic members: moments (int 1/EI, int x/EI, int x^2/EI over the element,
    # x from end I) for bending about 3 ("33") and 2 ("22"); the bending stiffness then
    # follows exactly from the flexibility (Euler-Bernoulli, as SAP2000)
    bending_flex: Optional[dict] = None

    @property
    def nodes(self) -> Tuple[int, int]:
        return (self.node_i, self.node_j)


@dataclass
class ShellLoad:
    pattern: str
    kind: str  # pressure (local 3) | force (global uniform force/area) | temperature
    value: np.ndarray  # pressure: (1,) ; force: (3,) global


@dataclass
class ShellElement:
    id: int
    object: str
    nodes: Tuple[int, int, int, int]
    E: float
    nu: float
    G: float
    thickness: float
    bending_thickness: float
    model_type: str  # ShellThin | ShellThick | Membrane | Plate
    mass_per_area: float
    weight_per_area: float
    modifiers: np.ndarray  # (8,) f11,f22,f12,m11,m22,m12,v13,v23
    axes: Optional[np.ndarray] = None  # (3,3) rows = local 1,2,3 (3 = normal); None => default
    loads: List[ShellLoad] = field(default_factory=list)
    # the modifiers of the lateral load cases (stiffness set "lateral"); None = modifiers
    modifiers_lateral: Optional[np.ndarray] = None


@dataclass
class Tie:
    """Auto edge constraint (ETABS): ``node`` follows the element it lies in /
    on, u = sum(w * u_master) for all six DOFs (DOFs already constrained by a
    diaphragm or restrained keep that)."""

    node: int
    masters: List[Tuple[int, float]]
    object: str = ""  # the area object of the element
    dofs: Optional[Tuple[int, ...]] = None  # the tied DOFs (None = all six)


@dataclass
class Body:
    """Rigid plate zone (slab inside a column's cross-section): the out-of-plane
    DOFs of ``nodes`` follow ``master`` as a rigid body - w = w_m + rx_m*dy -
    ry_m*dx, rx = rx_m, ry = ry_m (horizontal slabs)."""

    master: int
    nodes: List[int]
    object: str = ""  # the column


@dataclass
class RigidLink:
    """Rigid link between two nodes (all six DOFs): the nodes move as one rigid
    body, u_b = u_a + r_a x (x_b - x_a), r_b = r_a.  From Rigid frame objects and
    from beams meshed on their true (offset) axis, joined to their joints."""

    node_i: int
    node_j: int
    object: str = ""


@dataclass
class Constraint:
    """Rigid diaphragm constraint on a set of nodes with a virtual master."""

    name: str  # diaphragm name
    story: Optional[str]
    nodes: List[int]
    master: int  # node id of the virtual master
    normal: int = 2  # axis index of the plane normal (2 = Z)
    rigid: bool = True
    # exact-minus-discrete rotational inertia corrections (kg·m²) about the master
    # normal axis, so that lumped area masses reproduce the analytical inertia
    inertia_correction_self: float = 0.0
    inertia_correction_loads: Dict[str, float] = field(default_factory=dict)  # pattern -> kg·m² per unit multiplier


@dataclass
class AnalysisModel:
    nodes: List[Node] = field(default_factory=list)
    frames: List[FrameElement] = field(default_factory=list)
    shells: List[ShellElement] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    ties: List[Tie] = field(default_factory=list)  # auto edge constraints
    bodies: List[Body] = field(default_factory=list)  # slab-column rigid zones
    rigid_links: List[RigidLink] = field(default_factory=list)  # exact rigid links (6 DOF)
    info: List[str] = field(default_factory=list)  # notes for the analysis log (not warnings)
    active_dofs: np.ndarray = field(default_factory=lambda: np.ones(6, dtype=bool))
    self_weight: Dict[str, float] = field(default_factory=dict)  # pattern -> multiplier
    mass_from_loads: Dict[str, float] = field(default_factory=dict)  # pattern -> multiplier
    mass_include_elements: bool = True
    mass_include_added: bool = True
    mass_lateral: bool = True
    mass_vertical: bool = True
    mass_lump_at_stories: bool = False  # lateral mass between story levels -> the levels above/below
    nodal_loads: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)  # pattern -> node -> (6,)
    object_to_frames: Dict[str, List[int]] = field(default_factory=dict)
    object_to_shells: Dict[str, List[int]] = field(default_factory=dict)
    # mass of areas that are no elements (membrane floors in a rigid diaphragm), lumped at
    # the joints under them: node -> mass (kg); element mass (Mass Source "element")
    area_node_mass: Dict[int, float] = field(default_factory=dict)
    # joints of such floors that no element uses (a free corner): held without a warning
    quiet_nodes: List[int] = field(default_factory=list)
    point_to_node: Dict[str, int] = field(default_factory=dict)
    story_elevations: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    # load panels: {panel name: {pattern: (applied, carried)}} totals (force, SI) for checks
    panel_report: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    gravity: float = 9.80665
    auto_sizes: Dict[str, float] = field(default_factory=dict)  # area -> automatic mesh size (SI)
    mesh_check: Optional[dict] = None  # result of the mesh refinement check (api.analyze)

    # -------------------------------------------------------------- helpers
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def coords(self) -> np.ndarray:
        return np.array([n.xyz for n in self.nodes]) if self.nodes else np.zeros((0, 3))

    def add_nodal_load(self, pattern: str, node: int, vec: np.ndarray):
        table = self.nodal_loads.setdefault(pattern, {})
        table[node] = table.get(node, np.zeros(6)) + np.asarray(vec, dtype=float)

    def summary(self) -> dict:
        return {
            "nodes": len(self.nodes),
            "frame_elements": len(self.frames),
            "shell_elements": len(self.shells),
            "constraints": len(self.constraints),
            "warnings": list(self.warnings),
        }
