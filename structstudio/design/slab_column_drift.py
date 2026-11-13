"""Slab-column connections of two-way slabs without beams under the design story drift
(ACI 318-19 18.14.5, ACI 318-14 18.14.5.1 - referred to by มยผ. 1301/1302-61 2.11.5).

A slab that is left out of the lateral system (a membrane floor, or a shell floor with
small lateral stiffness modifiers) still moves with the building.  The code requires slab
shear reinforcement at a slab-column connection where the design story drift ratio is
large for the gravity shear stress on the slab critical section:

* ACI 318-19 18.14.5.1:  (a) nonprestressed      Δx/hsx >= 0.035 - 0.05 vuv/(φvc)
                         (b) unbonded PT, fpc >= 0.9 MPa in each direction (8.6.2.1)
                                                 Δx/hsx >= 0.040 - 0.05 vuv/(φvc)
  vuv: the factored two-way shear stress without moment transfer, from the controlling
  combination with E; Δx/hsx the greater of the stories above and below; vc per 22.6.5
  with Vp = 0 for unbonded PT.  18.14.5.2: not needed when Δx/hsx <= 0.005 (a) or
  0.01 (b).  18.14.5.3: vs >= 0.29 √f'c (MPa), extended 4h from the support face.
* ACI 318-14 18.14.5.1:  Δx/hsx >= 0.035 - (1/20) vug/(φvc), vug from the gravity loads
  without moment transfer, not needed when Δx/hsx <= 0.005; vs >= 0.29 √f'c over 4h.

Everything is computed in N, mm, MPa (the SI edition, ACI 318M) and reported in SI.

What is found from the model:

* connections: a vertical column below a joint that lies in a floor slab (shell or
  membrane) and has no beams framing into it in two directions;
* the critical section at d/2 from the column faces (22.6.4.1), straight sides
  (22.6.4.1.1), a circular column as the square of the same area (22.6.4.1.2); where a
  slab edge is near a face the section may run to the edge - the arrangement with the
  smallest bo is used, and the continuous sides give αs (40 / 30 / 20, 22.6.5.3);
* d = h - (distance to the reinforcement centroid, the slab section's ``rebar_cover``,
  default 30 mm), and for prestressed slabs not less than 0.8h (22.6.2.2);
* Vu: the column axial force below minus above the joint, formed case by case (and mode
  by mode for spectra) before the combination - never from envelopes;
* the design drift: Cd / Ie x the elastic drift of the column itself (its two ends),
  each direction, greater of the stories above and below, the largest over the
  earthquake cases.

Not done (reported where it matters): openings within 4h of the column (22.6.4.3, bo is
not reduced), drop panels / capitals (only the section at the column face), the
conditions 22.6.5.4(a) (bonded reinforcement, assumed provided when fpc is given).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..model.base import ModelError
from ..model.design import SLAB_COLUMN_CODES, SlabColumnDriftPrefs
from ..model.loads import LinearStaticCase, ResponseSpectrumCase

DEFAULT_COVER = 0.030  # m, slab face to the reinforcement centroid when the section gives none
FPC_MIN = 0.9  # MPa, ACI 318 8.6.2.1
SQRT_FC_MAX = 8.3  # MPa, 22.6.3.1
SQRT_FC_MAX_PT = 5.8  # MPa, 22.6.5.5
FPC_MAX = 3.5  # MPa, 22.6.5.5
SIN_VERTICAL = 1e-3


# ---------------------------------------------------------------- results
@dataclass
class SlabColumnCheck:
    """One slab-column connection.  Lengths m, forces N, stresses MPa (SI)."""

    story: str
    point: str
    column: str  # the column below
    column_above: Optional[str]
    x: float  # model units (where the connection is)
    y: float
    z: float
    slab: str = ""  # slab section
    location: str = ""  # Interior | Edge | Corner
    c1: float = 0.0  # column dimensions along / across its local 2 axis
    c2: float = 0.0
    h: float = 0.0
    d: float = 0.0
    bo: float = 0.0
    beta: float = 1.0
    alpha_s: float = 40.0
    lambda_s: float = 1.0
    fc: float = 0.0
    prestressed: bool = False  # unbonded PT with fpc >= 0.9 MPa in each direction (8.6.2.1)
    vc_clause: str = ""
    vc: float = 0.0
    phi_vc: float = 0.0
    Vu: float = 0.0
    vu: float = 0.0
    combo: str = ""
    ratio: float = 0.0  # vu / (φ vc)
    drift_below: float = 0.0
    drift_above: float = 0.0
    drift: float = 0.0  # design story drift ratio used (the greater)
    drift_case: str = ""
    limit: float = 0.0  # the drift ratio above which shear reinforcement is required
    exempt: float = 0.0  # 18.14.5.2: not required at or below this drift ratio
    status: str = ""  # Not required | Required | Not checked
    vs_min: float = 0.0  # MPa, 18.14.5.3
    extent: float = 0.0  # m from the support face
    notes: List[str] = field(default_factory=list)

    @property
    def required(self) -> bool:
        return self.status == "Required"

    def row(self) -> dict:
        return {"Story": self.story, "Point": self.point, "Column": self.column, "Location": self.location,
                "Slab": self.slab, "h [mm]": self.h * 1e3, "d [mm]": self.d * 1e3, "bo [mm]": self.bo * 1e3,
                "αs": self.alpha_s, "β": self.beta, "λs": self.lambda_s, "f'c [MPa]": self.fc,
                "PT": self.prestressed, "vc eq": self.vc_clause, "φvc [MPa]": self.phi_vc,
                "Vu [kN]": self.Vu / 1e3, "vu [MPa]": self.vu, "Combo": self.combo, "vu/φvc": self.ratio,
                "Δ/h below": self.drift_below, "Δ/h above": self.drift_above, "Δ/h": self.drift,
                "Drift case": self.drift_case, "Limit": self.limit, "Exempt ≤": self.exempt,
                "Status": self.status, "vs ≥ [MPa]": self.vs_min if self.required else None,
                "Extent ≥ [mm]": self.extent * 1e3 if self.required else None, "Notes": "; ".join(self.notes)}


@dataclass
class SlabColumnReport:
    code: str
    checks: List[SlabColumnCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    combos: List[str] = field(default_factory=list)
    drift_cases: List[str] = field(default_factory=list)

    @property
    def required(self) -> List[SlabColumnCheck]:
        return [c for c in self.checks if c.required]

    def rows(self) -> List[dict]:
        return [c.row() for c in self.checks]


# --------------------------------------------------------- load cases
def _is_earthquake_case(model, name: str) -> bool:
    case = model.load_cases.get(name)
    if isinstance(case, ResponseSpectrumCase):
        return True
    if isinstance(case, LinearStaticCase):
        return any(f and model.load_patterns.get(p) is not None and model.load_patterns[p].type == "Seismic"
                   for p, f in case.loads.items())
    return False


def earthquake_cases(model) -> List[str]:
    """Load cases that are earthquake (E) cases: response spectra, and linear static cases
    loading a pattern of type Seismic."""
    return [n for n in model.load_cases if _is_earthquake_case(model, n)]


def _combo_has_e(model, name: str, eq: set, seen=None) -> bool:
    seen = set() if seen is None else seen
    if name in seen:
        return False
    seen.add(name)
    c = model.load_combinations.get(name)
    if c is None:
        return name in eq
    return any(_combo_has_e(model, it.name, eq, seen) for it in c.items)


def combinations_with_e(model) -> List[str]:
    eq = set(earthquake_cases(model))
    return [n for n in model.load_combinations if _combo_has_e(model, n, eq)]


# ----------------------------------------------------------- geometry
def _seg_dist(p, a, b) -> float:
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def _inside(poly: np.ndarray, p: np.ndarray, tol: float) -> bool:
    """Point in polygon, the boundary (within tol) counted inside."""
    n = len(poly)
    for k in range(n):
        if _seg_dist(p, poly[k], poly[(k + 1) % n]) <= tol:
            return True
    x, y = p
    c = False
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def _strictly_inside(poly: np.ndarray, p: np.ndarray, tol: float) -> bool:
    n = len(poly)
    if any(_seg_dist(p, poly[k], poly[(k + 1) % n]) <= tol for k in range(n)):
        return False
    return _inside(poly, p, -1.0)


class _Level:
    """The floor slabs (and openings) at one elevation, plan coordinates in m."""

    def __init__(self, floors, openings, tol):
        self.floors = floors  # [(poly (n,2) m, thickness m, section name, area name)]
        self.openings = openings
        self.tol = tol

    def slab_at(self, p) -> Optional[tuple]:
        for op in self.openings:
            if _strictly_inside(op, p, self.tol):
                return None
        for fl in self.floors:
            if _inside(fl[0], p, self.tol):
                return fl
        return None

    def edge_distance(self, start, u, from_t: float, until: float, step: float) -> Optional[float]:
        """Distance along u from the point start + from_t u to the first point that is
        not slab, None when there is slab all the way to start + until u."""
        if self.slab_at(start + from_t * u) is None:
            return 0.0
        t_prev, t = from_t, from_t
        while t < until:
            t = min(until, t + step)
            if self.slab_at(start + t * u) is None:
                lo, hi = t_prev, t
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if self.slab_at(start + mid * u) is None:
                        hi = mid
                    else:
                        lo = mid
                return lo - from_t
            t_prev = t
        return None

    def thickness_change_near(self, start, h, radius) -> bool:
        for k in range(16):
            a = 2 * math.pi * k / 16
            for r in (0.5 * radius, radius):
                fl = self.slab_at(start + r * np.array([math.cos(a), math.sin(a)]))
                if fl is not None and abs(fl[1] - h) > 1e-6:
                    return True
        return False

    def opening_near(self, start, radius) -> bool:
        for op in self.openings:
            if min(_seg_dist(start, op[k], op[(k + 1) % len(op)]) for k in range(len(op))) <= radius:
                return True
        return False


def _critical_section(c1, c2, d, edges) -> Tuple[float, int, Tuple[bool, bool, bool, bool], dict]:
    """bo (smallest) of the critical section at d/2 from a c1 x c2 column whose faces
    (+a, -a, +b, -b) are at ``edges`` (distance from the face to the slab edge, None = no
    edge near).  A side may run to the slab edge instead of closing at d/2; a side
    cannot close where the edge is nearer than d/2.  Returns bo, the number of
    continuous (closed) sides, which sides are closed, and the extents."""
    best = None
    opts = []
    for k in range(4):
        e = edges[k]
        if e is None:
            opts.append((True,))
        elif e < 0.5 * d - 1e-9:
            opts.append((False,))
        else:
            opts.append((True, False))
    half = (0.5 * c1, 0.5 * c1, 0.5 * c2, 0.5 * c2)
    import itertools
    for closed in itertools.product(*opts):
        ext = [half[k] + 0.5 * d if closed[k] else half[k] + edges[k] for k in range(4)]
        len_a = ext[2] + ext[3]  # a side runs along b
        len_b = ext[0] + ext[1]
        bo = (len_a if closed[0] else 0.0) + (len_a if closed[1] else 0.0) + \
             (len_b if closed[2] else 0.0) + (len_b if closed[3] else 0.0)
        if bo <= 0:
            continue
        if best is None or bo < best[0] - 1e-12:
            best = (bo, sum(closed), tuple(closed), {"ext": ext})
    return best


def _alpha_s(n_closed: int) -> Tuple[float, str]:
    if n_closed >= 4:
        return 40.0, "Interior"
    if n_closed == 3:
        return 30.0, "Edge"
    return 20.0, "Corner"


def lambda_s(d_mm: float) -> float:
    """ACI 318-19 22.5.5.1.3 size effect factor (d in mm)."""
    return min(1.0, math.sqrt(2.0 / (1.0 + 0.004 * d_mm)))


def two_way_vc(code: str, fc: float, d_mm: float, bo_mm: float, beta: float, alpha_s: float, lam: float = 1.0,
               fpc: Optional[float] = None) -> Tuple[float, str, float]:
    """vc (MPa) of a two-way slab without shear reinforcement: ACI 318-19 / 318-14
    Table 22.6.5.2, or 22.6.5.5 for a prestressed interior connection (``fpc`` given,
    Vp = 0).  Returns (vc, clause, λs)."""
    sq = min(math.sqrt(fc), SQRT_FC_MAX)
    if fpc is not None:
        sqp = min(math.sqrt(fc), SQRT_FC_MAX_PT)
        f = min(fpc, FPC_MAX)
        va = 0.29 * lam * sqp + 0.3 * f
        vb = 0.083 * (1.5 + alpha_s * d_mm / bo_mm) * lam * sqp + 0.3 * f
        return min(va, vb), "22.6.5.5", 1.0
    if code == "ACI 318-14":
        ls = 1.0
        a = 0.33 * lam * sq
        b = 0.17 * (1.0 + 2.0 / beta) * lam * sq
    else:
        ls = lambda_s(d_mm)
        a = 0.33 * ls * lam * sq
        b = (0.17 + 0.33 / beta) * ls * lam * sq
    c = 0.083 * (2.0 + alpha_s * d_mm / bo_mm) * ls * lam * sq
    return min(a, b, c), "22.6.5.2", ls


# ------------------------------------------------------------- checker
class _Checker:
    def __init__(self, results, prefs: SlabColumnDriftPrefs, design_model=None):
        self.res = results
        self.m = results.model
        self.dm = design_model if design_model is not None else results.model  # cases / sections data
        self.p = prefs
        U = self.m.units
        self.Lf = U.factor("length")
        self.Ff = U.factor("force")
        self.Sf = U.factor("stress")
        self.tol = max(self.m.options.merge_tolerance, 1e-6) * self.Lf
        self.rep = SlabColumnReport(code=prefs.code)
        self._levels: Dict[float, _Level] = {}
        self.adj: Dict[str, List[str]] = {}
        for f in self.m.frames.values():
            self.adj.setdefault(f.point_i, []).append(f.name)
            self.adj.setdefault(f.point_j, []).append(f.name)
        self.story_z = sorted(s.elevation for s in self.m.stories.values())
        # several towers: the story levels of each tower (a column's story is in its tower)
        self._tower_z = {t: [s.elevation for s in self.m.stories_of(t)] for t in self.m.tower_names()} \
            if self.m.multi_tower else None

    def levels_of(self, pt: str) -> List[float]:
        """The story elevations (model units) of the tower of joint ``pt``."""
        if self._tower_z is None:
            return self.story_z
        return self._tower_z.get(self.m.tower_of_point(pt), self.story_z)

    # ---- geometry
    def xyz(self, pt: str) -> np.ndarray:
        p = self.m.points[pt]
        return np.array([p.x, p.y, p.z], float) * self.Lf

    def level(self, z_si: float) -> _Level:
        key = round(z_si / max(self.tol, 1e-9))
        if key in self._levels:
            return self._levels[key]
        floors, openings = [], []
        for a in self.m.areas.values():
            if not a.points or any(p not in self.m.points for p in a.points):
                continue
            P = np.array([self.xyz(p) for p in a.points])
            if np.max(np.abs(P[:, 2] - z_si)) > self.tol:
                continue
            poly = P[:, :2]
            if a.type == "Opening" or a.section is None or getattr(a, "is_opening", False):
                openings.append(poly)
                continue
            if a.type not in ("Floor", "Ramp"):
                continue
            sec = self.m.shell_sections.get(a.section)
            if sec is None or sec.type not in ("Slab",):
                continue
            floors.append((poly, sec.thickness * self.Lf, sec.name, a.name))
        lv = _Level(floors, openings, self.tol)
        self._levels[key] = lv
        return lv

    def vertical(self, f) -> bool:
        a, b = self.xyz(f.point_i), self.xyz(f.point_j)
        v = b - a
        L = np.linalg.norm(v)
        return L > 0 and np.linalg.norm(v[:2]) / L < SIN_VERTICAL

    def column_end(self, pt: str, up: bool) -> Optional[str]:
        """The column object leaving joint pt upward (up=True) or downward."""
        for fn in self.adj.get(pt, []):
            f = self.m.frames[fn]
            if f.type in ("Null", "Rigid") or not self.vertical(f):
                continue
            other = f.point_j if f.point_i == pt else f.point_i
            dz = self.xyz(other)[2] - self.xyz(pt)[2]
            if (dz > 0) == up and abs(dz) > self.tol:
                return fn
        return None

    def chain_end(self, pt: str, up: bool) -> Optional[Tuple[str, float]]:
        """Point at the next story level along the column line from pt, and the height."""
        z0 = self.xyz(pt)[2]
        cur, seen = pt, {pt}
        while True:
            fn = self.column_end(cur, up)
            if fn is None:
                return None
            f = self.m.frames[fn]
            nxt = f.point_j if f.point_i == cur else f.point_i
            if nxt in seen:
                return None
            seen.add(nxt)
            z = self.xyz(nxt)[2]
            if any(abs(z - zs * self.Lf) <= self.tol for zs in self.levels_of(pt)):
                return nxt, abs(z - z0)
            cur = nxt

    def beam_directions(self, pt: str) -> List[np.ndarray]:
        out = []
        for fn in self.adj.get(pt, []):
            f = self.m.frames[fn]
            if f.type in ("Null", "Rigid") or self.vertical(f):
                continue
            v = self.xyz(f.point_j) - self.xyz(f.point_i)
            if np.linalg.norm(v[:2]) < 1e-9:
                continue
            out.append(v[:2] / np.linalg.norm(v[:2]))
        return out

    def column_plan(self, f) -> Tuple[float, float, np.ndarray, np.ndarray, List[str]]:
        """c1 (along local 2), c2 (along local 3), the plan directions of 2 and 3 (m)."""
        from ..mesh.geometry import frame_axes
        notes = []
        ax = frame_axes(self.xyz(f.point_i), self.xyz(f.point_j), f.angle)
        e2, e3 = ax[1][:2], ax[2][:2]
        e2 = e2 / np.linalg.norm(e2)
        e3 = e3 / np.linalg.norm(e3)
        sec = self.m.frame_sections.get(f.section)
        shape = (sec.shape if sec is not None else "").lower()
        if sec is not None and shape.startswith("circ") and sec.t3:
            s = math.sqrt(math.pi) / 2.0 * sec.t3 * self.Lf
            notes.append("circular column: square of the same area (22.6.4.1.2)")
            return s, s, e2, e3, notes
        if sec is not None and sec.t3 and sec.t2:
            if not shape.startswith("rect"):
                notes.append(f"{sec.shape} column: taken as its {sec.t3:g} x {sec.t2:g} outline")
            return sec.t3 * self.Lf, sec.t2 * self.Lf, e2, e3, notes
        if sec is not None and sec.A > 0:
            s = math.sqrt(sec.A * self.Lf ** 2)
            notes.append("column size unknown: a square of the section area assumed")
            return s, s, e2, e3, notes
        raise ModelError("NO_COLUMN_SIZE", f"Column {f.name}: section {f.section!r} has no dimensions")

    # ---- analysis quantities
    def _end_axial_u(self, frame: str, at_point: str):
        """function u (SI full vector) -> axial force (model units) at the end of the
        object at at_point (the face of an end offset), no element loads (spectrum modes)."""
        f = self.m.frames[frame]
        elems = self.res.am.object_to_frames.get(frame, [])
        if not elems:
            raise ModelError("NO_ELEMENTS", f"Frame {frame} is not in the analysis model")
        at_i = f.point_i == at_point
        eid = elems[0] if at_i else elems[-1]
        el = self.res.am.frames[eid]
        fe = self.res.engine.asm.frames[eid]
        x = np.array([el.end_offset_i if at_i else fe.kin.L - el.end_offset_j])
        d = self.res.engine.asm._element_dofs(fe.el.nodes)
        ff = 1.0 / self.Ff

        def val(u):
            return float(fe.internal_forces(u[d], [], x)[0][0]) * ff

        return val

    def _end_axial_case(self, frame: str, at_point: str):
        f = self.m.frames[frame]
        idx = 0 if f.point_i == at_point else -1

        def val(r):
            return float(self.res.frame_forces(frame, r.case).forces[idx][0])

        return val

    def shear_envelope(self, name: str, below: str, above: Optional[str], pt: str, strip_e: bool, eq: set):
        """Envelope (model force units) of Vu = P_above - P_below (the slab's shear
        delivered to the column, + downward) for a case or combination."""
        from ..results import Envelope
        ub = self._end_axial_u(below, pt)
        cb = self._end_axial_case(below, pt)
        if above:
            ua, ca = self._end_axial_u(above, pt), self._end_axial_case(above, pt)
        else:
            ua = ca = (lambda _x: 0.0)

        def vu_u(u):
            return np.array([ua(u) - ub(u)])

        def vu_c(r):
            return np.array([ca(r) - cb(r)])

        combos = self.m.load_combinations

        def env(n):
            if n in combos:
                c = combos[n]
                items = [it for it in c.items if not (strip_e and it.name not in combos and it.name in eq)]
                if not items:
                    return Envelope.single(np.zeros(1))
                return self.res._combine(dataclasses.replace(c, items=items), env)
            if strip_e and n in eq:
                return Envelope.single(np.zeros(1))
            return self.res._signed_envelope(n, vu_u, vu_c)

        return env(name)

    def drift_ratio(self, case: str, top: str, bottom: str, h: float) -> float:
        nt = self.res._node_of_point(top)
        nb = self.res._node_of_point(bottom)

        def val(u):
            return np.array([(u[6 * nt] - u[6 * nb]) / h, (u[6 * nt + 1] - u[6 * nb + 1]) / h])

        e = self.res._signed_envelope(case, val)
        return float(np.max(np.maximum(np.abs(e.max), np.abs(e.min))))

    # ---- run
    def run(self) -> SlabColumnReport:
        rep, p = self.rep, self.p
        if p.code not in SLAB_COLUMN_CODES:
            rep.errors.append(f"Unknown code {p.code!r}: use {' or '.join(SLAB_COLUMN_CODES)}")
            return rep
        if not (0 < p.phi <= 1) or not (0 < p.lam <= 1):
            rep.errors.append(f"φ = {p.phi:g} and λ = {p.lam:g} must be in (0, 1]")
            return rep
        eq_all = earthquake_cases(self.dm)
        eq = set(eq_all)
        # the cases of the design drift
        drift_cases = list(p.seismic_cases) or [c for c in eq_all if c in self.res.cases]
        missing = [c for c in drift_cases if c not in self.res.cases]
        if missing:
            rep.errors.append("No results for the drift case(s) " + ", ".join(missing) + " - run the analysis")
        if not drift_cases:
            rep.errors.append("No earthquake load case (a response spectrum, or a linear static case loading a "
                              "Seismic pattern) - the design drift cannot be found")
        no_cd = [c for c in drift_cases if c in self.dm.load_cases and getattr(self.dm.load_cases[c], "cd", None) is None]
        bad = [c for c in drift_cases if c not in self.dm.load_cases]
        if bad:
            rep.errors.append("Drift case(s) not defined: " + ", ".join(bad))
        if no_cd:
            rep.errors.append("Give the deflection amplification factor Cd (and Ie) of the earthquake case(s) "
                              + ", ".join(no_cd) + " (Define > Load Cases: cd, ie) - the design drift is "
                              "Cd/Ie x the elastic drift (ASCE 7 12.8.6, มยผ. 1301/1302-61)")
        # the combinations of the shear stress
        strip_e = False
        if p.combos:
            combos = list(p.combos)
            unknown = [c for c in combos if c not in self.m.load_combinations and c not in self.res.cases]
            if unknown:
                rep.errors.append("Unknown combination(s) / case(s) for the shear stress: " + ", ".join(unknown))
        else:
            combos = [c for c in combinations_with_e(self.dm) if c in self.m.load_combinations]
            if p.code == "ACI 318-14":
                strip_e = True
                rep.info.append("vug (ACI 318-14: gravity loads): the gravity part of each combination with E "
                                "(its earthquake cases left out) - choose other combinations in the preferences")
            else:
                rep.info.append("vuv (ACI 318-19): the combinations with E, the largest governs")
            if not combos:
                rep.errors.append("No load combination contains an earthquake case - define the strength "
                                  "combinations with E (or choose the combinations in the preferences)")
        rep.combos, rep.drift_cases = combos, drift_cases
        if rep.errors:
            return rep
        amp = {c: float(self.dm.load_cases[c].cd) / float(getattr(self.dm.load_cases[c], "ie", 1.0) or 1.0)
               for c in drift_cases}
        for c in drift_cases:
            rep.info.append(f"{c}: design drift = Cd/Ie = {amp[c]:.3g} x elastic drift")
        for pt in sorted(self.m.points, key=lambda n: (self.m.points[n].z, n)):
            below = self.column_end(pt, up=False)
            if below is None:
                continue
            xyz = self.xyz(pt)
            if not any(abs(xyz[2] - zs * self.Lf) <= self.tol for zs in self.levels_of(pt)):
                continue
            lv = self.level(xyz[2])
            if not lv.floors:
                continue
            fl = lv.slab_at(xyz[:2])
            if fl is None:
                continue
            chk = self._check(pt, below, xyz, lv, fl, combos, strip_e, eq, drift_cases, amp)
            if chk is not None:
                rep.checks.append(chk)
        if not rep.checks:
            rep.warnings.append("No slab-column connection found (a column below a joint inside a floor slab, "
                                "no beams framing in two directions)")
        return rep

    def _check(self, pt, below, xyz, lv, fl, combos, strip_e, eq, drift_cases, amp) -> Optional[SlabColumnCheck]:
        p = self.p
        P = self.m.points[pt]
        above = self.column_end(pt, up=True)
        st = self.m.story_at(P.z, tower=self.m.tower_of_point(pt)) if not P.story else None
        story = P.story or (st.name if st else "")
        c = SlabColumnCheck(story=story, point=pt, column=below, column_above=above, x=P.x, y=P.y, z=P.z)
        dirs = self.beam_directions(pt)
        if any(abs(float(a[0] * b[1] - a[1] * b[0])) > 0.17 for a in dirs for b in dirs):
            c.status = "Not checked"
            c.notes.append("beams frame in two directions: not a two-way slab without beams")
            return c
        if dirs:
            c.notes.append("a beam frames in along one direction (edge beam)")
        poly, h, sec_name, _ = fl
        sec = self.dm.shell_sections.get(sec_name) or self.m.shell_sections.get(sec_name)
        c.slab, c.h = sec_name, h
        mat = self.dm.materials.get(sec.material) or self.m.materials.get(sec.material)
        if mat is None or not mat.fc:
            c.status = "Not checked"
            c.notes.append(f"material {sec.material!r} of slab {sec_name} has no f'c")
            return c
        Sf = self.Sf
        c.fc = mat.fc * Sf / 1e6
        cover = sec.rebar_cover * self.Lf if sec.rebar_cover is not None else DEFAULT_COVER
        d = h - cover
        fpc1 = None if sec.fpc1 is None else sec.fpc1 * Sf / 1e6
        fpc2 = None if sec.fpc2 is None else sec.fpc2 * Sf / 1e6
        pt_slab = bool(sec.post_tensioned) and fpc1 is not None and fpc2 is not None
        if pt_slab:
            d = max(d, 0.8 * h)  # 22.6.2.2
        if d <= 0:
            c.status = "Not checked"
            c.notes.append(f"d = h - cover = {d * 1e3:.0f} mm")
            return c
        c.d = d
        fcol = self.m.frames[below]
        try:
            c1, c2, e2, e3, cn = self.column_plan(fcol)
        except ModelError as exc:
            c.status = "Not checked"
            c.notes.append(exc.message)
            return c
        c.notes += cn
        c.c1, c.c2 = c1, c2
        c.beta = max(c1, c2) / min(c1, c2)
        o = xyz[:2]
        search = max(4.0 * h, max(c1, c2) + 2.0 * d) + d
        step = max(min(0.25 * d, 0.05), 0.005)
        faces = ((e2, 0.5 * c1), (-e2, 0.5 * c1), (e3, 0.5 * c2), (-e3, 0.5 * c2))
        edges = [lv.edge_distance(o, u, f0, f0 + search, step) for u, f0 in faces]
        bo, n_closed, closed, _ = _critical_section(c1, c2, d, edges)
        c.bo = bo
        c.alpha_s, c.location = _alpha_s(n_closed)
        if n_closed < 2:
            c.notes.append("slab continuous on fewer than two sides: taken as a corner")
        if lv.opening_near(o, 4.0 * h + 0.5 * max(c1, c2)):
            c.notes.append("opening within 4h: bo not reduced (22.6.4.3) - check by hand")
        if lv.thickness_change_near(o, h, 0.5 * max(c1, c2) + 2.0 * h):
            c.notes.append("slab thickness changes near the column (drop panel?): only the section at the "
                           "column face is checked")
        # prestress
        meets = pt_slab and min(fpc1, fpc2) >= FPC_MIN
        interior_pt = meets and all(e is None or e > 4.0 * h for e in edges)  # 22.6.5.4(b)
        if sec.post_tensioned and not pt_slab:
            c.notes.append("post-tensioned slab without fpc1 / fpc2: designed as nonprestressed")
        elif pt_slab and not meets:
            c.notes.append(f"fpc {min(fpc1, fpc2):.2f} MPa < 0.9 MPa (8.6.2.1): designed as nonprestressed")
        if meets and not interior_pt:
            c.notes.append("PT connection within 4h of an edge: vc per 22.6.5.2 (22.6.5.4(b))")
        c.prestressed = meets if p.code == "ACI 318-19" else False
        fpc_avg = 0.5 * (fpc1 + fpc2) if interior_pt else None
        vc, clause, ls = two_way_vc(p.code, c.fc, d * 1e3, bo * 1e3, c.beta, c.alpha_s, p.lam, fpc_avg)
        c.vc, c.vc_clause, c.lambda_s = vc, clause, ls
        c.phi_vc = p.phi * vc
        # shear on the critical section
        best = (0.0, "")
        for n in combos:
            try:
                e = self.shear_envelope(n, below, above, pt, strip_e, eq)
            except ModelError as exc:
                c.status = "Not checked"
                c.notes.append(f"{n}: {exc.message}")
                return c
            v = float(np.max(e.max)) * self.Ff
            if v > best[0] or not best[1]:
                best = (max(v, 0.0), n)
        c.Vu, c.combo = best
        c.vu = c.Vu / (bo * d) / 1e6
        c.ratio = c.vu / c.phi_vc
        # design drift
        dn = self.chain_end(pt, up=False)
        up = self.chain_end(pt, up=True)
        for case in drift_cases:
            try:
                rb = self.drift_ratio(case, pt, dn[0], dn[1]) * amp[case] if dn else 0.0
                ra = self.drift_ratio(case, up[0], pt, up[1]) * amp[case] if up else 0.0
            except ModelError as exc:
                c.status = "Not checked"
                c.notes.append(f"{case}: {exc.message}")
                return c
            if max(rb, ra) > c.drift or not c.drift_case:
                c.drift_below, c.drift_above, c.drift, c.drift_case = rb, ra, max(rb, ra), case
        # the criterion
        if c.prestressed:
            c.limit, c.exempt = 0.040 - 0.05 * c.ratio, 0.01
        else:
            c.limit, c.exempt = 0.035 - 0.05 * c.ratio, 0.005
        need = c.drift > c.exempt and c.drift >= c.limit
        c.status = "Required" if need else "Not required"
        if c.drift <= c.exempt:
            clause = "18.14.5.2" if p.code == "ACI 318-19" else "18.14.5.1"
            c.notes.append(f"drift ratio <= {c.exempt:g}: shear reinforcement need not be provided ({clause})")
        c.vs_min = 0.29 * math.sqrt(c.fc)
        c.extent = 4.0 * h
        if c.ratio > 1.0:
            c.notes.append("vu > φvc under the shear alone: the punching check fails before the drift check")
        return c


def check_slab_column_drift(results, prefs: Optional[SlabColumnDriftPrefs] = None, model=None) -> SlabColumnReport:
    """ACI 318 18.14.5 slab-column drift check of every connection of an analysed model.
    ``model``: the model holding the case factors / slab data (default the analysed one)."""
    if prefs is None:
        src = model if model is not None else results.model
        prefs = getattr(getattr(src, "design", None), "slab_column_drift", None) or SlabColumnDriftPrefs()
    return _Checker(results, prefs, model).run()
