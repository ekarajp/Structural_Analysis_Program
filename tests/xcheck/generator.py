"""Random structures for the OpenSees cross-check.

``random_case(seed)`` returns a :class:`~structstudio.model.Model` (kN, m) and
a short description.  Families range from a single cantilever to towers,
domes, helices and random space frames, with skewed members, rotated local
axes, member end releases, springs, joint offsets, and unusual loads
(trapezoidal, partial span, projected, local and global directions, member
point forces and moments, self weight).  Every structure is built so that it
is stable (supports + no mechanisms) - the cross-check rejects singular ones.

The generator only uses features both programs model *exactly* (linear
elastic 3D frames, Euler-Bernoulli or Timoshenko members, nodal springs and
masses) so any difference is a bug, not a modelling choice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from structstudio.model import Model
from structstudio.model.objects import GroundDisplacement
from structstudio.model.properties import FrameModifiers, FrameSection

FIXED = (True,) * 6
PINNED = (True, True, True, False, False, False)
FREE = (False,) * 6


@dataclass
class Case:
    model: Model
    family: str
    seed: int
    notes: List[str] = field(default_factory=list)
    modal: bool = False
    # tolerance class: "exact" (both programs integrate the loads exactly) or
    # "strips" (trapezoids given to OpenSees as 64 uniform strips)
    tol_class: str = "exact"


# ------------------------------------------------------------------ helpers
def _material(m: Model, rng: np.random.Generator) -> str:
    kind = rng.choice(["Concrete", "Steel", "Aluminum", "Other"])
    E = {"Concrete": 2.5e7, "Steel": 2.0e8, "Aluminum": 7.0e7, "Other": float(10 ** rng.uniform(6, 8.5))}[kind]
    nu = float(rng.uniform(0.15, 0.33))
    name = f"M{len(m.materials) + 1}"
    mat = m.add_material(name, E=E * float(rng.uniform(0.8, 1.2)), nu=nu, weight_density=float(rng.uniform(0, 80)))
    mat.mass_density = 0.0  # masses only as point masses (both programs lump them identically)
    mat.type = str(kind)
    return name


def _section(m: Model, rng: np.random.Generator, mat: str, shear: Optional[bool] = None) -> str:
    """Random section; ``shear`` False forces As2 = As3 = 0 (Euler-Bernoulli)."""
    name = f"S{len(m.frame_sections) + 1}"
    shape = rng.choice(["rect", "circle", "I", "general"])
    if shape == "rect":
        s = FrameSection.rectangular(name, mat, float(rng.uniform(0.15, 1.2)), float(rng.uniform(0.15, 1.0)))
    elif shape == "circle":
        s = FrameSection.circular(name, mat, float(rng.uniform(0.1, 1.0)))
    elif shape == "I":
        d = float(rng.uniform(0.15, 0.9))
        bf = float(rng.uniform(0.1, 0.4))
        tf = float(rng.uniform(0.008, 0.04))
        tw = float(rng.uniform(0.005, 0.025))
        s = FrameSection.i_shape(name, mat, d, bf, tf, tw)
    else:
        A = float(10 ** rng.uniform(-3, -0.5))
        I33 = float(A * 10 ** rng.uniform(-3, -0.5))
        I22 = float(I33 * 10 ** rng.uniform(-1.5, 0.5))
        J = float((I33 + I22) * rng.uniform(0.02, 1.0))
        s = FrameSection.general(name, mat, A=A, I33=I33, I22=I22, J=J,
                                 As2=float(A * rng.uniform(0.3, 0.9)), As3=float(A * rng.uniform(0.3, 0.9)))
    if shear is None:
        shear = bool(rng.random() < 0.5)
    if not shear:
        s.As2 = s.As3 = 0.0
    if rng.random() < 0.15:
        s.modifiers = FrameModifiers(A=float(rng.uniform(0.5, 2)), J=float(rng.uniform(0.1, 1)),
                                     I22=float(rng.uniform(0.3, 1.5)), I33=float(rng.uniform(0.3, 1.5)),
                                     As2=1.0, As3=1.0)
    m.add_frame_section(s)
    return name


def _frame(m: Model, rng: np.random.Generator, a, b, sec: str, angle: Optional[float] = None, ftype=None):
    if angle is None:
        angle = float(rng.choice([0.0, 0.0, 90.0, float(rng.uniform(-180, 180))]))
    f = m.add_frame(a, b, sec, angle=angle, type=ftype)
    f.auto_mesh = False
    f.mesh_at_intersections = False
    f.mesh_at_floor = False
    return f


def _points_xyz(m: Model, xyz: np.ndarray):
    return [m.add_point(*map(float, p), merge=False) for p in xyz]


# ------------------------------------------------------------------ families
def fam_cantilever(m, rng):
    mat = _material(m, rng)
    sec = _section(m, rng, mat)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    if rng.random() < 0.3:  # exactly vertical / horizontal members are special cases of the axes rule
        d = np.array(rng.choice([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [1, 1, 0]]), float)
        d /= np.linalg.norm(d)
    L = float(rng.uniform(1, 12))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(*(d * L))
    _frame(m, rng, a, b, sec)
    return "cantilever"


def fam_chain(m, rng):
    """A crooked chain of members in 3D between two or three supports."""
    mat = _material(m, rng)
    n = int(rng.integers(3, 9))
    xyz = np.cumsum(rng.normal(scale=2.0, size=(n, 3)), axis=0)
    pts = _points_xyz(m, xyz)
    pts[0].restraints = FIXED
    pts[-1].restraints = tuple(bool(x) for x in rng.choice([FIXED, PINNED]))
    secs = [_section(m, rng, mat) for _ in range(int(rng.integers(1, 3)))]
    for k in range(n - 1):
        _frame(m, rng, pts[k], pts[k + 1], str(rng.choice(secs)))
    return "chain"


def fam_portal(m, rng):
    """Plane or 3D multi-bay, multi-story frame, optional sloped roof and inclined columns."""
    mat = _material(m, rng)
    col, beam = _section(m, rng, mat), _section(m, rng, mat)
    nx, ny, ns = int(rng.integers(1, 4)), int(rng.integers(0, 3)), int(rng.integers(1, 5))
    xs = np.concatenate([[0], np.cumsum(rng.uniform(3, 9, nx))])
    ys = np.concatenate([[0], np.cumsum(rng.uniform(3, 8, ny))]) if ny else np.array([0.0])
    zs = np.concatenate([[0], np.cumsum(rng.uniform(2.5, 5, ns))])
    lean = rng.uniform(-0.15, 0.15, 2) if rng.random() < 0.3 else np.zeros(2)
    roof = float(rng.uniform(0, 0.3)) if rng.random() < 0.3 else 0.0
    P = {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                dz = roof * min(x - xs[0], xs[-1] - x) if k == ns else 0.0
                p = m.add_point(x + lean[0] * z, y + lean[1] * z, z + dz, merge=False)
                if k == 0:
                    p.restraints = FIXED if rng.random() < 0.7 else PINNED
                P[i, j, k] = p
    pin_beams = rng.random() < 0.3
    if pin_beams:  # released members are Euler-Bernoulli (OpenSees' Timoshenko element has no releases)
        beam = _section(m, rng, mat, shear=False)
    for (i, j, k), p in P.items():
        if k > 0:
            _frame(m, rng, P[i, j, k - 1], p, col, angle=float(rng.choice([0, 90, rng.uniform(0, 90)])))
            if (i + 1, j, k) in P:
                f = _frame(m, rng, p, P[i + 1, j, k], beam, angle=0.0)
                if pin_beams and rng.random() < 0.5:
                    f.releases_i = (False, False, False, False, True, True)
            if (i, j + 1, k) in P:
                _frame(m, rng, p, P[i, j + 1, k], beam, angle=0.0)
    if rng.random() < 0.4:  # a few braces
        br = _section(m, rng, mat)
        for (i, j, k), p in list(P.items()):
            if k > 0 and (i + 1, j, k - 1) in P and rng.random() < 0.3:
                _frame(m, rng, P[i + 1, j, k - 1], p, br)
    return "portal"


def fam_truss(m, rng):
    """Pin-jointed plane or space truss: members released in M2, M3 at both ends,
    joint rotations restrained (the usual way to keep a truss model stable)."""
    mat = _material(m, rng)
    n = int(rng.integers(2, 7))
    h = float(rng.uniform(1, 3))
    span = float(rng.uniform(1.5, 4))
    space = rng.random() < 0.5
    w = float(rng.uniform(1, 3))
    bottom, top = [], []
    rows = [0.0, w] if space else [0.0]
    for y in rows:
        bottom.append([m.add_point(i * span, y, 0, merge=False) for i in range(n + 1)])
        top.append([m.add_point(i * span + span / 2, y, h, merge=False) for i in range(n)])
    sec = _section(m, rng, mat, shear=False)
    rel = (False, False, False, False, True, True)
    members = []
    for r in range(len(rows)):
        b, t = bottom[r], top[r]
        for i in range(n):
            members += [(b[i], b[i + 1]), (b[i], t[i]), (t[i], b[i + 1])]
            if i < n - 1:
                members.append((t[i], t[i + 1]))
    if space:
        for i in range(n + 1):
            members.append((bottom[0][i], bottom[1][i]))
        for i in range(n):
            members += [(top[0][i], top[1][i]), (bottom[0][i], bottom[1][i + 1]),
                        (bottom[0][i], top[1][i]), (bottom[0][i + 1], top[1][i])]  # brace the cross-section
        for i in range(n - 1):
            members.append((top[0][i], top[1][i + 1]))
    for a, b in members:
        f = _frame(m, rng, a, b, sec)
        f.releases_i = rel
        f.releases_j = rel
    for p in m.points.values():
        p.restraints = (False, False, False, True, True, True)
    for r in range(len(rows)):
        bottom[r][0].restraints = (True, True, True, True, True, True)
        bottom[r][-1].restraints = (False, True, True, True, True, True) if r == 0 else \
            (False, False, True, True, True, True)
    if not space:
        for p in m.points.values():
            p.restraints = tuple(bool(x) for x in (p.restraints[0], True, p.restraints[2], True, True, True))
    return "truss3d" if space else "truss2d"


def fam_arch(m, rng):
    """Circular / parabolic arch or a closed ring, in a tilted plane."""
    mat = _material(m, rng)
    sec = _section(m, rng, mat)
    n = int(rng.integers(6, 24))
    R = float(rng.uniform(3, 15))
    ring = rng.random() < 0.25
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) if ring else np.linspace(0, np.pi * rng.uniform(0.4, 1.0), n + 1)
    loc = np.stack([R * np.cos(th), np.zeros_like(th), R * np.sin(th)], axis=1)
    if rng.random() < 0.3 and not ring:  # parabolic arch
        loc[:, 2] = R * (1 - (loc[:, 0] / R) ** 2) * rng.uniform(0.3, 1)
    a = rng.uniform(0, 2 * np.pi)
    b = rng.uniform(-0.6, 0.6)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    xyz = loc @ (Rz @ Rx).T
    pts = _points_xyz(m, xyz)
    for k in range(len(pts) - (0 if ring else 1)):
        _frame(m, rng, pts[k], pts[(k + 1) % len(pts)], sec)
    if ring:
        pts[0].restraints = FIXED
        pts[n // 2].restraints = PINNED
    else:
        pts[0].restraints = FIXED if rng.random() < 0.6 else PINNED
        pts[-1].restraints = FIXED if rng.random() < 0.6 else PINNED
        if pts[0].restraints == PINNED and pts[-1].restraints == PINNED:
            pts[0].restraints = FIXED
    return "ring" if ring else "arch"


def fam_dome(m, rng):
    """Ribbed dome: meridians and rings on a sphere cap."""
    mat = _material(m, rng)
    rib, hoop = _section(m, rng, mat), _section(m, rng, mat)
    nm, nr = int(rng.integers(4, 9)), int(rng.integers(2, 5))
    R = float(rng.uniform(5, 15))
    phi = np.linspace(rng.uniform(0.6, 1.2), 0.15, nr + 1)
    crown = m.add_point(0, 0, R)
    rings = []
    for k, ph in enumerate(phi):
        ring = [m.add_point(R * math.sin(ph) * math.cos(t), R * math.sin(ph) * math.sin(t), R * math.cos(ph), merge=False)
                for t in np.linspace(0, 2 * np.pi, nm, endpoint=False)]
        rings.append(ring)
    for p in rings[0]:
        p.restraints = PINNED if rng.random() < 0.6 else FIXED
    for k in range(len(rings)):
        for i in range(nm):
            if k > 0:
                _frame(m, rng, rings[k][i], rings[k][(i + 1) % nm], hoop)
            nxt = rings[k + 1][i] if k + 1 < len(rings) else crown
            _frame(m, rng, rings[k][i], nxt, rib)
    return "dome"


def fam_helix(m, rng):
    """Spiral stair / helical spring: one or two helices with rungs."""
    mat = _material(m, rng)
    sec = _section(m, rng, mat)
    n = int(rng.integers(8, 30))
    R = float(rng.uniform(1, 6))
    pitch = float(rng.uniform(0.2, 1.0))
    turns = float(rng.uniform(0.5, 2.5))
    t = np.linspace(0, 2 * np.pi * turns, n + 1)
    a = _points_xyz(m, np.stack([R * np.cos(t), R * np.sin(t), pitch * t], axis=1))
    for k in range(n):
        _frame(m, rng, a[k], a[k + 1], sec)
    a[0].restraints = FIXED
    if rng.random() < 0.5:
        r2 = R * rng.uniform(0.3, 0.8)
        b = _points_xyz(m, np.stack([r2 * np.cos(t), r2 * np.sin(t), pitch * t], axis=1))
        rung = _section(m, rng, mat)
        for k in range(n):
            _frame(m, rng, b[k], b[k + 1], sec)
        for k in range(0, n + 1, int(rng.integers(1, 4))):
            _frame(m, rng, a[k], b[k], rung)
        b[0].restraints = FIXED
    elif rng.random() < 0.5:
        a[-1].restraints = PINNED
    return "helix"


def fam_tower(m, rng):
    """Lattice tower: 3 or 4 legs, tapering, X or K bracing."""
    mat = _material(m, rng)
    leg, br = _section(m, rng, mat), _section(m, rng, mat)
    legs = int(rng.choice([3, 4]))
    nlev = int(rng.integers(2, 7))
    H = float(rng.uniform(8, 40))
    b0, b1 = float(rng.uniform(3, 8)), float(rng.uniform(0.8, 3))
    lev = []
    for k in range(nlev + 1):
        z = H * k / nlev
        r = b0 + (b1 - b0) * k / nlev
        lev.append([m.add_point(r * math.cos(2 * math.pi * i / legs + 0.3), r * math.sin(2 * math.pi * i / legs + 0.3), z,
                                merge=False) for i in range(legs)])
    for p in lev[0]:
        p.restraints = FIXED if rng.random() < 0.5 else PINNED
    pattern = rng.choice(["X", "Z"])
    for k in range(nlev):
        for i in range(legs):
            j = (i + 1) % legs
            _frame(m, rng, lev[k][i], lev[k + 1][i], leg)
            _frame(m, rng, lev[k + 1][i], lev[k + 1][j], br)
            _frame(m, rng, lev[k][i], lev[k + 1][j], br)
            if pattern == "X":
                _frame(m, rng, lev[k][j], lev[k + 1][i], br)
    return "tower"


def fam_grillage(m, rng):
    """Horizontal beam grid (torsion and bending out of plane), skewed."""
    mat = _material(m, rng)
    s1, s2 = _section(m, rng, mat), _section(m, rng, mat)
    nx, ny = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    dx, dy = float(rng.uniform(1.5, 5)), float(rng.uniform(1.5, 5))
    skew = float(rng.uniform(-0.5, 0.5))
    P = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            P[i, j] = m.add_point(i * dx + skew * j * dy, j * dy, 0.0, merge=False)
    for (i, j), p in P.items():
        if (i + 1, j) in P:
            _frame(m, rng, p, P[i + 1, j], s1, angle=0.0)
        if (i, j + 1) in P:
            _frame(m, rng, p, P[i, j + 1], s2, angle=0.0)
    corners = [P[0, 0], P[nx, 0], P[0, ny], P[nx, ny]]
    for c in corners:
        c.restraints = (True, True, True, False, False, False)
    P[0, 0].restraints = FIXED
    return "grillage"


def fam_random_space(m, rng):
    """Random space frame: a random spanning tree on random points plus extra members."""
    mat = _material(m, rng)
    secs = [_section(m, rng, mat) for _ in range(int(rng.integers(1, 4)))]
    n = int(rng.integers(4, 16))
    xyz = rng.uniform(-8, 8, size=(n, 3))
    pts = _points_xyz(m, xyz)
    order = list(range(n))
    used = set()
    for k in range(1, n):
        j = int(rng.integers(0, k))
        _frame(m, rng, pts[order[k]], pts[order[j]], str(rng.choice(secs)))
        used.add((min(k, j), max(k, j)))
    for _ in range(int(rng.integers(0, n))):
        a, b = sorted(int(x) for x in rng.choice(n, 2, replace=False))
        if (a, b) not in used:
            used.add((a, b))
            _frame(m, rng, pts[a], pts[b], str(rng.choice(secs)))
    pts[0].restraints = FIXED
    if rng.random() < 0.5:
        pts[-1].restraints = tuple(bool(x) for x in rng.integers(0, 2, 6))
    return "random"


def fam_springs(m, rng):
    """Beam / frame on point springs (no rigid supports for some DOFs)."""
    mat = _material(m, rng)
    sec = _section(m, rng, mat)
    n = int(rng.integers(2, 8))
    pts = [m.add_point(i * float(rng.uniform(1, 4)), 0, 0, merge=False) for i in range(n + 1)]
    xs = np.cumsum([0] + [float(rng.uniform(1, 4)) for _ in range(n)])
    for p, x in zip(pts, xs):
        p.x = float(x)
    for k in range(n):
        _frame(m, rng, pts[k], pts[k + 1], sec)
    for p in pts:
        p.springs = tuple(float(10 ** rng.uniform(2, 6)) for _ in range(3))
        p.rotational_springs = tuple(float(10 ** rng.uniform(1, 5)) for _ in range(3))
    if rng.random() < 0.5:
        pts[0].restraints = PINNED
    return "springs"


def fam_plate(m, rng):
    """Flat plate (Mindlin, "ShellThick") of user-meshed quads, skewed and distorted,
    optionally with edge beams; in-plane DOFs restrained (pure plate bending)."""
    from structstudio.model.properties import ShellSection
    mat = _material(m, rng)
    t = float(rng.uniform(0.08, 0.6))
    m.add_shell_section(ShellSection("PL", mat, type="Slab", model_type="ShellThick", thickness=t))
    nx, ny = int(rng.integers(2, 7)), int(rng.integers(2, 7))
    lx, ly = float(rng.uniform(2, 10)), float(rng.uniform(2, 10))
    skew = float(rng.uniform(-0.4, 0.4))
    jitter = 0.0  # distorted quads: MITC4 variants differ ~0.1% (see compare.py notes)
    P = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = i * lx / nx + skew * j * ly / ny, j * ly / ny
            if 0 < i < nx and 0 < j < ny:
                x += jitter * lx / nx * float(rng.uniform(-1, 1))
                y += jitter * ly / ny * float(rng.uniform(-1, 1))
            P[i, j] = m.add_point(x, y, 0.0, merge=False, restraints=(True, True, False, False, False, True))
    for i in range(nx):
        for j in range(ny):
            a = m.add_area([P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]], "PL", type="Floor")
            a.mesh_divisions = (1, 1)
    edges = {"S": [P[i, 0] for i in range(nx + 1)], "N": [P[i, ny] for i in range(nx + 1)],
             "W": [P[0, j] for j in range(ny + 1)], "E": [P[nx, j] for j in range(ny + 1)]}
    chosen = [e for e in edges if rng.random() < 0.6] or ["S", "N"]
    if len(chosen) == 1:
        chosen.append({"S": "N", "N": "S", "W": "E", "E": "W"}[chosen[0]])
    for e in chosen:
        clamp = rng.random() < 0.4
        for p in edges[e]:
            p.restraints = (True, True, True, clamp or p.restraints[3], clamp or p.restraints[4], True)
    if rng.random() < 0.4:  # edge beams on the free edges
        sec = _section(m, rng, mat)
        for e in edges:
            if e not in chosen:
                pts = edges[e]
                for k in range(len(pts) - 1):
                    _frame(m, rng, pts[k], pts[k + 1], sec, angle=0.0)
    return "plate"


FAMILIES: List[Tuple[str, Callable, float]] = [
    ("cantilever", fam_cantilever, 1.0),
    ("chain", fam_chain, 1.0),
    ("portal", fam_portal, 2.0),
    ("truss", fam_truss, 1.0),
    ("arch", fam_arch, 1.0),
    ("dome", fam_dome, 0.7),
    ("helix", fam_helix, 0.8),
    ("tower", fam_tower, 1.0),
    ("grillage", fam_grillage, 1.0),
    ("random", fam_random_space, 1.5),
    ("springs", fam_springs, 0.7),
    ("plate", fam_plate, 1.2),
]


# --------------------------------------------------------------------- loads
def _snap(a: float, b: float) -> Tuple[float, float]:
    """Load ranges: no tiny pieces next to the member ends (OpenSees would need
    sub-elements of a few micrometres there - ill-conditioned)."""
    a = 0.0 if a < 0.03 else a
    b = 1.0 if b > 0.97 else b
    if b - a < 0.05:
        b = min(1.0, a + 0.1)
        if b > 0.97:
            b = 1.0
    return a, b


FRAME_DIRS = ["Gravity", "X", "Y", "Z", "1", "2", "3", "GravityProjected", "XProjected", "YProjected", "ZProjected"]


def _add_loads(m: Model, rng: np.random.Generator, case: Case):
    npat = int(rng.integers(1, 4))
    frames = list(m.frames)
    points = list(m.points)
    scale = 10.0 ** float(rng.uniform(-1, 2))
    for k in range(npat):
        pat = f"LP{k + 1}"
        sw = float(rng.choice([0.0, 0.0, 1.0, rng.uniform(-2, 2)])) if not m.areas else 0.0
        m.add_load_pattern(pat, "Other", self_weight_multiplier=sw)
        if sw:
            case.notes.append(f"{pat}: self weight x{sw:.3g}")
        # area loads (plates: uniform gravity / Z pressure)
        for an in rng.choice(list(m.areas), size=min(len(m.areas), int(rng.integers(0, 4))), replace=False) \
                if m.areas else []:
            m.add_area_load(str(an), pat, float(rng.normal(scale=scale)), direction=str(rng.choice(["Gravity", "Z"])))
        # joint loads
        for p in rng.choice(points, size=min(len(points), int(rng.integers(0, 5))), replace=False):
            f = rng.normal(scale=scale, size=3) * (rng.random(3) < 0.7)
            mm = rng.normal(scale=scale, size=3) * (rng.random(3) < 0.4)
            m.add_point_load(str(p), pat, force=tuple(map(float, f)), moment=tuple(map(float, mm)))
        # member loads
        for fname in rng.choice(frames, size=min(len(frames), int(rng.integers(0, 6))), replace=False):
            fname = str(fname)
            kind = rng.choice(["uniform", "partial", "trapezoid", "point", "point_moment"], p=[0.3, 0.2, 0.2, 0.2, 0.1])
            d = str(rng.choice(FRAME_DIRS))
            v = float(rng.normal(scale=scale))
            if kind == "uniform":
                m.add_frame_distributed_load(fname, pat, v, direction=d)
            elif kind == "partial":
                a, b = _snap(*sorted(float(x) for x in rng.uniform(0, 1, 2)))
                m.add_frame_distributed_load(fname, pat, v, direction=d, rel_a=a, rel_b=b)
            elif kind == "trapezoid":
                a, b = sorted(float(x) for x in rng.uniform(0, 1, 2))
                if rng.random() < 0.5:
                    a, b = 0.0, 1.0
                a, b = _snap(a, b)
                m.add_frame_distributed_load(fname, pat, v, value_b=float(rng.normal(scale=scale)), direction=d,
                                             rel_a=a, rel_b=b)
                case.tol_class = "strips"
            elif kind == "point":
                d = d.replace("Projected", "")
                m.add_frame_point_load(fname, pat, v, rel=float(rng.uniform(0.05, 0.95)), direction=d)
            else:
                d = str(rng.choice(["X", "Y", "Z", "1", "2", "3"]))
                m.add_frame_point_load(fname, pat, v, rel=float(rng.uniform(0.05, 0.95)), direction=d,
                                       load_type="Moment")
        m.add_linear_static_case(pat, {pat: 1.0})


def _add_masses(m: Model, rng: np.random.Generator):
    for p in m.points.values():
        if all(p.restraints[:3]):
            continue
        mm = float(10 ** rng.uniform(-1, 2))
        p.mass = (mm, mm * float(rng.uniform(0.5, 2)), mm * float(rng.uniform(0.5, 2)))
        p.rotational_mass = tuple(float(mm * rng.uniform(0.01, 1)) for _ in range(3))
    m.add_modal_case("MODAL", max_modes=8)


def random_case(seed: int, family: Optional[str] = None) -> Case:
    rng = np.random.default_rng(seed)
    m = Model(f"xcheck-{seed}", "kN m")
    m.use_csi_frame_defaults()  # OpenSees: members join at the joints
    names = [f[0] for f in FAMILIES]
    w = np.array([f[2] for f in FAMILIES])
    fam = family or str(rng.choice(names, p=w / w.sum()))
    fn = dict((f[0], f[1]) for f in FAMILIES)[fam]
    label = fn(m, rng)
    case = Case(m, label, seed)
    # extras on some models
    if rng.random() < 0.15:  # skewed spring at a free joint
        free = [p for p in m.points.values() if not any(p.restraints)]
        if free:
            p = free[int(rng.integers(0, len(free)))]
            p.springs = tuple(float(10 ** rng.uniform(2, 5)) for _ in range(3))
            case.notes.append(f"spring at {p.name}")
    diaphragm = False
    if label in ("portal", "tower", "dome") and rng.random() < 0.4:  # rigid floor diaphragms
        m.add_diaphragm("D1")
        levels: Dict[float, list] = {}
        for p in m.points.values():
            if not any(p.restraints):
                levels.setdefault(round(p.z, 6), []).append(p)
        for z, pts in levels.items():
            if len(pts) >= 2:
                for p in pts:
                    p.diaphragm = "D1"
                diaphragm = True
        if diaphragm:
            case.notes.append("rigid diaphragms")
    offset_frames = []
    if not diaphragm and rng.random() < 0.12:  # joint offsets (rigid links) on a few members
        for f in list(m.frames.values())[: int(rng.integers(1, 4))]:
            if any(f.releases_i) or any(f.releases_j):
                continue
            f.joint_offset_i = tuple(float(x) for x in rng.uniform(-0.3, 0.3, 3))
            f.joint_offset_j = tuple(float(x) for x in rng.uniform(-0.3, 0.3, 3))
            offset_frames.append(f.name)
        if offset_frames:
            case.notes.append(f"joint offsets on {', '.join(offset_frames)}")
    np_frames = []
    if rng.random() < 0.15:  # nonprismatic members (I varies, A and J constant)
        cand = [f for f in m.frames.values() if not any(f.releases_i) and not any(f.releases_j)
                and f.name not in offset_frames]
        rng.shuffle(cand)
        for f in cand[: int(rng.integers(1, 4))]:
            base = m.frame_sections[f.section]
            ends = []
            for k in range(2):
                sname = f"{base.name}_v{len(m.frame_sections)}"
                sec = FrameSection(**{**base.__dict__, "name": sname,
                                      "I33": base.I33 * float(rng.uniform(0.3, 3.0)),
                                      "I22": base.I22 * float(rng.uniform(0.3, 3.0)), "As2": 0.0, "As3": 0.0})
                m.frame_sections[sname] = sec
                ends.append(sname)
            nseg = int(rng.integers(1, 4))
            names = [base.name] + ends
            rows = []
            for k in range(nseg):
                rows.append(dict(start=names[k % 3], end=names[(k + 1) % 3], length=float(rng.uniform(0.5, 2.0)),
                                 length_type="Variable", ei33=str(rng.choice(["Linear", "Parabolic", "Cubic"])),
                                 ei22=str(rng.choice(["Linear", "Parabolic", "Cubic"]))))
            npname = f"NP{len(m.frame_sections)}"
            m.frame_sections[npname] = FrameSection(**{**m.frame_sections[rows[0]["start"]].__dict__, "name": npname,
                                                       "nonprismatic": rows, "As2": 0.0, "As3": 0.0})
            f.section = npname
            np_frames.append(f.name)
        if np_frames:
            case.notes.append(f"nonprismatic {', '.join(np_frames)}")
    xyz = np.array([p.xyz for p in m.points.values()])
    if len(xyz) > 1:
        d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1) + np.eye(len(xyz)) * 1e9
        if d.min() < 1e-4:
            case.notes.append("coincident points")
    _add_loads(m, rng, case)
    for n in offset_frames + np_frames:  # offset members: positions ambiguous; nonprismatic: forceBeamColumn
        m.frames[n].loads = []
    if rng.random() < 0.2:  # support settlements / spring ground displacements
        pats = list(m.load_patterns)
        sup = [p for p in m.points.values() if any(p.restraints) or any(v > 0 for v in p.springs)]
        for p in sup[: int(rng.integers(1, 4))]:
            pat = str(rng.choice(pats))
            tr = tuple(float(rng.uniform(-0.01, 0.01)) if rng.random() < 0.6 else 0.0 for _ in range(3))
            rot = tuple(float(rng.uniform(-0.002, 0.002)) if rng.random() < 0.3 else 0.0 for _ in range(3))
            if label == "plate":  # in-plane motion engages the fictitious (program dependent) drilling stiffness
                rot = (rot[0], rot[1], 0.0)
                tr = (0.0, 0.0, tr[2])
            p.ground_displacements.append(GroundDisplacement(pat, tr, rot))
            case.notes.append(f"ground displacement at {p.name} ({pat})")
    if rng.random() < 0.35:
        _add_masses(m, rng)
        case.modal = True
    return case
