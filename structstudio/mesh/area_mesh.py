"""Feature-aware meshing of floors and walls (2-D, in the shell's own plane).

ETABS meshes floors "at beams and other meshing lines, wall and ramp edges"
and SAFE meshes around every column, point load and line load.  Here:

* :func:`tensor_mesh` - a rectilinear shell (all edges along two perpendicular
  directions, the usual floor / wall) is cut along the lines through every
  corner, opening corner and *feature* (joint, beam, null line, wall edge,
  neighbouring mesh node), each interval subdivided to the maximum element
  size.  Features land exactly on mesh nodes / mesh lines and the mesh is
  conforming.  Feature lines closer than ``min_gap`` to a stronger line are
  not added (no sliver elements); those features are then reached by
* :func:`relocate` - the nearest movable mesh node is moved onto a feature
  point, or onto a feature line (sampled at the element size), as long as no
  element turns bad.  Nodes on the shell boundary only slide along their own
  edge; corners and nodes already holding a feature never move.

Everything works on ``P`` (n x 2 node coordinates) and ``cells`` (m x 4 node
indices, counter-clockwise).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import polygon2d as pg


@dataclass
class Features:
    points: List[Tuple[float, float]] = field(default_factory=list)  # must be mesh nodes
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)  # mesh along
    # wanted as mesh nodes where they fit (corners of column rigid zones): mesh
    # lines through them unless one is already close, never moved onto
    soft: List[Tuple[float, float]] = field(default_factory=list)
    # column cross-sections inside the shell (rigid zones): (polygon (k, 2) CCW,
    # element size h at its edges).  The mesh follows their outline exactly, at
    # every mesh size, and grows from h at the edges to the shell's size
    zones: List[Tuple[np.ndarray, float]] = field(default_factory=list)

    def __bool__(self):
        return bool(self.points or self.lines or self.soft or self.zones)


ZONE_GROWTH = 1.5  # element size ratio from one ring of elements to the next around a column


def _aligned(P: np.ndarray, tol: float) -> bool:
    return pg.axis_aligned(np.asarray(P, float), tol)


def _in_zone(p, feats, tol) -> bool:
    return any(_aligned(Z, tol) and pg.inside(np.asarray(p, float), Z, edge_tol=tol) for Z, _ in feats.zones)


def zone_grading(zones, tol: float):
    """({u level: h}, {v level: h}) of the axis-aligned zones: the element size
    wanted at those mesh lines (graded meshes start there)."""
    gu: Dict[float, float] = {}
    gv: Dict[float, float] = {}
    su: List[Tuple[float, float, float]] = []
    sv: List[Tuple[float, float, float]] = []
    for P, h in zones:
        if not _aligned(P, tol):
            continue
        u0, u1 = float(P[:, 0].min()), float(P[:, 0].max())
        v0, v1 = float(P[:, 1].min()), float(P[:, 1].max())
        for u in (u0, u1):
            gu[u] = min(h, gu.get(u, math.inf))
        for v in (v0, v1):
            gv[v] = min(h, gv.get(v, math.inf))
        su.append((u0, u1, h))
        sv.append((v0, v1, h))
    return gu, gv, su, sv


def _grade_at(grading: Dict[float, float], x: float, tol: float) -> Optional[float]:
    for k, h in grading.items():
        if abs(k - x) <= tol:
            return h
    return None


def graded_split(x0: float, x1: float, size: float, h0: Optional[float], h1: Optional[float],
                 growth: float = ZONE_GROWTH) -> List[float]:
    """Points dividing [x0, x1] (x1 included, x0 not) with elements of ``h0`` at
    x0 / ``h1`` at x1 growing by about ``growth`` per element up to ``size``
    (uniform when neither end is graded)."""
    L = x1 - x0
    if L <= 0:
        return [x1]
    if h0 is None and h1 is None:
        n = max(1, int(math.ceil(L / size - 1e-9)))
        return [x0 + L * k / n for k in range(1, n + 1)]
    g = growth - 1.0
    xs = np.linspace(0.0, L, 401)
    f = np.full_like(xs, size)
    if h0 is not None:
        f = np.minimum(f, h0 + g * xs)
    if h1 is not None:
        f = np.minimum(f, h1 + g * (L - xs))
    w = 1.0 / f
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(xs))])
    n = max(1, int(math.ceil(cum[-1] - 1e-6)))
    t = np.interp(np.arange(1, n + 1) * cum[-1] / n, cum, xs)
    t[-1] = L
    return [x0 + float(v) for v in t]


def _levels(base: Sequence[float], extra: Sequence[float], tol: float, min_gap: float) -> List[float]:
    """Sorted levels: every ``base`` value (corners), plus each ``extra`` value
    unless it is within ``min_gap`` of a level already taken."""
    out = sorted(pg._levels(base, tol))
    for v in sorted(extra):
        k = np.searchsorted(out, v)
        near = min([abs(v - out[j]) for j in (k - 1, k) if 0 <= j < len(out)], default=math.inf)
        if near > max(min_gap, tol):
            out.insert(k, float(v))
    return out


def feature_levels(polys: Sequence[np.ndarray], holes: Sequence[np.ndarray], feats: Features,
                   size: Optional[float], tol: float, min_gap_ratio: float = 0.3):
    """Mesh lines (u levels, v levels) through all corners, opening corners, joints
    on an edge and lines along u / v (always), and other joints (unless closer than
    ``min_gap_ratio * size`` to a line already taken)."""
    min_gap = (size or 0.0) * min_gap_ratio
    bu = [q[0] for P in [*polys, *holes] for q in P]
    bv = [q[1] for P in [*polys, *holes] for q in P]
    for Z, _h in feats.zones:  # column cross-sections: their outline is always a mesh line
        if _aligned(Z, tol):
            bu += list(Z[:, 0])
            bv += list(Z[:, 1])
    eu, ev = [p[0] for p in feats.soft], [p[1] for p in feats.soft]
    for p in feats.points:
        on_edge = any(_on_boundary(np.asarray(p, float), P, tol) for P in polys) or _in_zone(p, feats, tol)
        (bu if on_edge else eu).append(p[0])
        (bv if on_edge else ev).append(p[1])
    for a, b in feats.lines:
        if abs(a[0] - b[0]) <= tol:  # a line along v: always a u level (ETABS: mesh at beams)
            bu.append(a[0])
            ev += [a[1], b[1]]
        elif abs(a[1] - b[1]) <= tol:
            bv.append(a[1])
            eu += [a[0], b[0]]
        else:
            eu += [a[0], b[0]]
            ev += [a[1], b[1]]
    return _levels(bu, eu, tol, min_gap), _levels(bv, ev, tol, min_gap)


def tensor_mesh(Q: np.ndarray, holes: Sequence[np.ndarray], feats: Features, size: Optional[float],
                default_div: int, tol: float, min_gap_ratio: float = 0.3,
                fixed_div: Optional[Tuple[int, int]] = None, levels=None, grading=None):
    """Conforming quad mesh of a rectilinear polygon (minus rectilinear holes)
    with grid lines through the features (or through the given ``levels``, shared
    by all shells of one plane so neighbours match).  Column cross-sections
    (``feats.zones``, or the plane's ``grading``) add their outline as mesh lines,
    a fine fill inside and lines graded out from their faces (sizes h, 2h, 4h ...);
    the regular lines of the shell stay where they were away from the columns, so
    meshes of neighbouring floors and walls still nest.  Returns (P, cells)."""
    lu, lv = float(np.ptp(Q[:, 0])), float(np.ptp(Q[:, 1]))
    if size is None or size <= 0:
        size = max(lu, lv) / max(default_div, 1) if (feats or levels is not None) else None
    min_gap = (size or 0.0) * min_gap_ratio
    bu = [q[0] for P in [Q, *holes] for q in P]
    bv = [q[1] for P in [Q, *holes] for q in P]
    corners_u, corners_v = list(bu), list(bv)
    if grading is None:
        grading = zone_grading(feats.zones, tol)
    for p in feats.points:  # a column joint inside its cross-section: always a mesh node
        if _in_zone(p, feats, tol):
            bu.append(p[0])
            bv.append(p[1])
    eu = [p[0] for p in [*feats.points, *feats.soft]]
    ev = [p[1] for p in [*feats.points, *feats.soft]]
    for a, b in feats.lines:
        if abs(a[0] - b[0]) <= tol:  # line along v: a u level through it
            eu.append(a[0])
            ev += [a[1], b[1]]
        elif abs(a[1] - b[1]) <= tol:
            ev.append(a[1])
            eu += [a[0], b[0]]
        else:
            eu += [a[0], b[0]]
            ev += [a[1], b[1]]
    U = _levels(bu, [u for u in eu if Q[:, 0].min() - tol <= u <= Q[:, 0].max() + tol], tol, min_gap)
    V = _levels(bv, [v for v in ev if Q[:, 1].min() - tol <= v <= Q[:, 1].max() + tol], tol, min_gap)
    gu, gv, su, sv = (tuple(grading) + ([], []))[:4] if grading is not None else ({}, {}, [], [])
    if levels is not None:  # the plane's shared lines inside this shell (its corners are among them)
        lo, hi = Q.min(axis=0) - tol, Q.max(axis=0) + tol

        def plain(vals, grade, own):  # the zone outlines come back below, with their grading
            return [x for x in vals if _grade_at(grade, x, tol) is None
                    or any(abs(x - c) <= tol for c in own)]
        U = pg._levels(plain([u for u in levels[0] if lo[0] <= u <= hi[0]], gu, corners_u) + list(Q[:, 0]), tol)
        V = pg._levels(plain([v for v in levels[1] if lo[1] <= v <= hi[1]], gv, corners_v) + list(Q[:, 1]), tol)

    def subdivide(levels, n_total):
        out = [levels[0]]
        span = levels[-1] - levels[0]
        for x0, x1 in zip(levels[:-1], levels[1:]):
            if n_total is not None:  # explicit divisions shared out by length
                n = max(1, int(round(n_total * (x1 - x0) / span)))
            elif size is None:
                n = default_div
            else:
                n = max(1, int(math.ceil((x1 - x0) / size - 1e-9)))
            out += [x0 + (x1 - x0) * k / n for k in range(1, n + 1)]
        return out
    us = subdivide(U, fixed_div[0] if fixed_div else None)
    vs = subdivide(V, fixed_div[1] if fixed_div else None)
    if size is not None and fixed_div is None:
        us = _with_zones(us, U, gu, su, size, tol)
        vs = _with_zones(vs, V, gv, sv, size, tol)
    idx: Dict[Tuple[int, int], int] = {}
    P: List[Tuple[float, float]] = []
    cells = []
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            c = ((us[i] + us[i + 1]) / 2, (vs[j] + vs[j + 1]) / 2)
            if not pg.inside(c, Q) or any(pg.inside(c, h) for h in holes):
                continue
            quad = []
            for (a, b) in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                if (a, b) not in idx:
                    idx[(a, b)] = len(P)
                    P.append((us[a], vs[b]))
                quad.append(idx[(a, b)])
            cells.append(quad)
    return np.array(P, float).reshape(-1, 2), np.array(cells, int).reshape(-1, 4)


def _with_zones(xs: List[float], base: Sequence[float], grade: Dict[float, float],
                spans: Sequence[Tuple[float, float, float]], size: float, tol: float) -> List[float]:
    """Mesh lines ``xs`` (along one direction) plus the column cross-sections:
    their faces, a fill of ``h`` inside, and lines at h, 2h, 4h ... outward from
    each face until the regular size is reached.  Regular lines (not ``base``
    lines: corners, beams, joints) too close to the graded ones are dropped."""
    if not grade:
        return xs
    lo, hi = xs[0], xs[-1]
    added: List[float] = []
    clear: List[Tuple[float, float]] = []  # ranges where the regular lines give way

    def in_span(x):
        return any(a + tol < x < b - tol for a, b, _ in spans)
    for a, b, h in spans:
        if b < lo - tol or a > hi + tol:
            continue
        n = max(1, int(math.ceil((b - a) / h - 1e-9)))
        added += [a + (b - a) * k / n for k in range(n + 1)]
        clear.append((a, b))
    for z, h in grade.items():
        if z < lo - tol or z > hi + tol:
            continue
        added.append(z)
        for side in (-1.0, 1.0):
            if in_span(z + side * 2 * tol):
                continue  # the inside of the column: filled above
            d, step = 0.0, h
            while step < size - tol:
                d += step
                added.append(z + side * d)
                step *= 2.0
            reach = d + 0.25 * size  # regular lines a quarter size beyond the last graded one stay
            clear.append((z, z + side * reach) if side > 0 else (z - reach, z))
    keep_base = [x for x in base]

    def is_base(x):
        return any(abs(x - b) <= tol for b in keep_base)
    out = [x for x in xs if is_base(x) or not any(a - tol <= x <= b + tol for a, b in clear)]
    out += [x for x in added if lo - tol <= x <= hi + tol]
    out = sorted(out)
    hmin = min(grade.values())
    res: List[float] = []
    for x in out:  # merge lines closer than a fraction of the finest size (base lines win)
        if res and x - res[-1] <= tol:
            continue
        if res and x - res[-1] < 0.3 * hmin and not (is_base(x) and is_base(res[-1])):
            if is_base(x):
                res[-1] = x
            continue
        res.append(x)
    res[0], res[-1] = lo, hi
    return res


# ------------------------------------------------------------------ relocation
def _boundary(cells: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    """Boundary node -> its free edges (edges used by one cell)."""
    count: Dict[frozenset, List[Tuple[int, int]]] = {}
    for c in cells:
        for k in range(4):
            a, b = int(c[k]), int(c[(k + 1) % 4])
            count.setdefault(frozenset((a, b)), []).append((a, b))
    out: Dict[int, List[Tuple[int, int]]] = {}
    for es in count.values():
        if len(es) == 1:
            a, b = es[0]
            out.setdefault(a, []).append((a, b))
            out.setdefault(b, []).append((a, b))
    return out


MIN_ANGLE, MAX_ANGLE = 30.0, 150.0  # corner angles a moved node may leave (degrees)


def _quad_ok(P: np.ndarray, c) -> bool:
    """Convex, counter-clockwise, every corner angle within MIN_ANGLE..MAX_ANGLE."""
    X = P[list(c)]
    if pg.signed_area(X) <= 0:
        return False
    lo, hi = math.cos(math.radians(MAX_ANGLE)), math.cos(math.radians(MIN_ANGLE))
    for k in range(4):
        a, b, d = X[k - 1], X[k], X[(k + 1) % 4]
        u, v = a - b, d - b
        cr = u[1] * v[0] - u[0] * v[1]  # > 0 for a convex CCW corner
        nu, nv = float(np.hypot(*u)), float(np.hypot(*v))
        if nu <= 0 or nv <= 0 or cr <= 0:
            return False
        cs = float(u @ v) / (nu * nv)
        if not lo <= cs <= hi:
            return False
    return True


@dataclass
class Relocation:
    moved: int = 0
    missed: List[Tuple[float, float]] = field(default_factory=list)  # joints that got no node
    line_gap: float = 0.0  # largest distance between mesh nodes along a feature line


def relocate(P: np.ndarray, cells: np.ndarray, feats: Features, h: float, tol: float,
             fixed: Optional[Set[int]] = None, move_boundary: bool = False) -> Relocation:
    """Move mesh nodes onto feature points / lines that are not mesh nodes yet
    (in place).  ``h``: typical element size (reach and line sampling)."""
    rep = Relocation()
    if not feats or not len(P):
        return rep
    fixed = set(fixed or ())
    bnd = _boundary(cells)
    node_cells: Dict[int, List[int]] = {}
    for ci, c in enumerate(cells):
        for n in c:
            node_cells.setdefault(int(n), []).append(ci)
    corners = set()
    for n, es in bnd.items():
        if len(es) != 2:
            corners.add(n)
            continue
        (a, b), (c, d) = es
        e1 = P[b] - P[a]
        e2 = P[d] - P[c]
        if abs(e1[0] * e2[1] - e1[1] * e2[0]) > 1e-9 * (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-30):
            corners.add(n)  # the boundary turns here
    fixed |= corners

    def try_move(n: int, target) -> bool:
        if n in fixed:
            return False
        target = np.asarray(target, float)
        if n in bnd and not move_boundary:
            return False  # the edge may be shared with a neighbouring shell: keep it matching
        if n in bnd:  # boundary: only along its own (straight) edge
            (a, b) = bnd[n][0]
            e = P[b] - P[a]
            d = target - P[n]
            if abs(e[0] * d[1] - e[1] * d[0]) > tol * np.linalg.norm(e):
                return False
        old = P[n].copy()
        P[n] = target
        if all(_quad_ok(P, cells[ci]) for ci in node_cells.get(n, [])):
            fixed.add(n)
            rep.moved += 1
            return True
        P[n] = old
        return False

    def nearest(target, k=6):
        d = np.linalg.norm(P - np.asarray(target), axis=1)
        return list(np.argsort(d)[:k]), d

    # points first (exact positions), then lines
    for p in feats.points:
        cand, d = nearest(p)
        if d[cand[0]] <= tol:
            fixed.add(int(cand[0]))
            continue
        if not any(d[n] <= 1.5 * h and try_move(int(n), p) for n in cand):
            rep.missed.append(tuple(p))
    for a, b in feats.lines:
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if L <= tol:
            continue
        e = (b - a) / L
        # nodes already on the line
        rel = P - a
        t = rel @ e
        off = np.abs(rel[:, 0] * e[1] - rel[:, 1] * e[0])
        on = sorted(float(x) for x in t[(off <= tol) & (t >= -tol) & (t <= L + tol)])
        stations = [0.0] + on + [L]
        targets = []
        for s0, s1 in zip(stations[:-1], stations[1:]):
            gap = s1 - s0
            if gap > 1.5 * h:
                n = int(math.ceil(gap / h))
                targets += [s0 + gap * k / n for k in range(1, n)]
        for s in targets:
            q = a + s * e
            cand, d = nearest(q, k=10)
            for n in cand:
                if d[n] > 1.0 * h:
                    break
                proj = a + float(np.clip((P[n] - a) @ e, 0.0, L)) * e  # slide onto the line
                if try_move(int(n), proj):
                    break
        # how well is the line tied in now (loads on it go to these nodes)
        rel = P - a
        t = rel @ e
        off = np.abs(rel[:, 0] * e[1] - rel[:, 1] * e[0])
        st = sorted([0.0, L] + [float(x) for x in t[(off <= tol) & (t >= -tol) & (t <= L + tol)]])
        rep.line_gap = max(rep.line_gap, max(b_ - a_ for a_, b_ in zip(st[:-1], st[1:])))
    return rep


def clip_segment(a, b, Q: np.ndarray, tol: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """The parts of segment a-b inside polygon Q (on its boundary counts as inside).
    Parts lying on the boundary itself are dropped (they are edges already)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L <= tol:
        return []
    ts = [0.0, 1.0]
    n = len(Q)
    for i in range(n):
        c, e = Q[i], Q[(i + 1) % n]
        f = e - c
        den = d[0] * f[1] - d[1] * f[0]
        if abs(den) < 1e-12:
            continue
        w = c - a
        t = (w[0] * f[1] - w[1] * f[0]) / den
        s = (w[0] * d[1] - w[1] * d[0]) / den
        if -1e-9 <= s <= 1 + 1e-9 and 0 < t < 1:
            ts.append(float(t))
    ts = sorted(set(round(t, 12) for t in ts))
    out = []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        if (t1 - t0) * L <= tol:
            continue
        mid = a + d * (t0 + t1) / 2
        if not pg.inside(mid, Q, edge_tol=tol):
            continue
        if _on_boundary(mid, Q, tol):
            continue
        p0, p1 = a + d * t0, a + d * t1
        out.append(((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))))
    return out


def _on_boundary(p, Q: np.ndarray, tol: float) -> bool:
    for i in range(len(Q)):
        a, b = Q[i], Q[(i + 1) % len(Q)]
        e = b - a
        L2 = float(e @ e)
        t = 0.0 if L2 <= 0 else min(max(float((p - a) @ e) / L2, 0.0), 1.0)
        if np.linalg.norm(a + t * e - p) <= tol:
            return True
    return False


# ------------------------------------------------------------ freeform shells
def _sample_chain(pts: Sequence[np.ndarray], spacing: float, keep: Sequence[np.ndarray], tol: float):
    """Points along an open / closed polyline: its vertices, the ``keep`` points
    lying on it, and subdivisions no longer than ``spacing``."""
    out = []
    n = len(pts)
    for i in range(n):
        a, b = np.asarray(pts[i], float), np.asarray(pts[(i + 1) % n], float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= tol:
            continue
        ts = [0.0]
        for q in keep:
            t = float((np.asarray(q) - a) @ d) / (L * L)
            if tol / L < t < 1 - tol / L and np.linalg.norm(a + t * d - q) <= tol:
                ts.append(t)
        ts = sorted(set(ts)) + [1.0]
        seg = []
        for t0, t1 in zip(ts[:-1], ts[1:]):
            k = max(1, int(math.ceil((t1 - t0) * L / spacing - 1e-9)))
            seg += [t0 + (t1 - t0) * j / k for j in range(k)]
        out += [a + t * d for t in seg]
    return out


def freeform_mesh(Q: np.ndarray, holes: Sequence[np.ndarray], feats: Features, size: float, tol: float,
                  smooth: int = 8):
    """All-quad mesh of any simple polygon (non-convex, curved outline, holes) with
    the feature points / lines as mesh nodes / lines: a conforming Delaunay
    triangulation with points on the outline, openings and features at 2 x
    ``size`` and a fill lattice, each triangle split into three quads at its
    centroid and edge midpoints, then a few passes of quality-checked smoothing.
    Returns (P, cells)."""
    from scipy.spatial import Delaunay

    h2 = 2.0 * size
    fpts = [np.asarray(p, float) for p in feats.points]
    if feats.soft:  # rigid-zone corners: as nodes unless too close to the outline
        edges = [(Q[i], Q[(i + 1) % len(Q)]) for i in range(len(Q))] + \
                [(H[i], H[(i + 1) % len(H)]) for H in holes for i in range(len(H))]

        def edge_dist(c):
            out = math.inf
            for a, b in edges:
                d = b - a
                L2 = float(d @ d)
                t = 0.0 if L2 <= 0 else min(max(float((c - a) @ d) / L2, 0.0), 1.0)
                out = min(out, float(np.linalg.norm(a + t * d - c)))
            return out
        for p in feats.soft:
            c = np.asarray(p, float)
            if edge_dist(c) > 0.3 * size and all(np.linalg.norm(c - q) > 0.3 * size for q in fpts):
                fpts.append(c)
    # column cross-sections: their outline as mesh lines (points every 2 h - a
    # triangle side becomes two quads), crossings with beams as nodes, and rings
    # of points around them, spaced more widely outward, so the mesh grows from
    # h at the column face to the shell's size
    zlines = []
    for Z, hz in feats.zones:
        Z = np.asarray(Z, float)
        for i in range(len(Z)):
            za, zb = Z[i], Z[(i + 1) % len(Z)]
            zlines.append((za, zb))
            fpts += [p for p in _sample_chain([za, zb], 2.0 * hz, [], tol)[:1]]
            k = max(1, int(math.ceil(np.linalg.norm(zb - za) / (2.0 * hz) - 1e-9)))
            fpts += [za + (zb - za) * j / k for j in range(1, k)]
            for la, lb in feats.lines:
                x = _cross(za, zb, np.asarray(la, float), np.asarray(lb, float), tol)
                if x is not None:
                    fpts.append(x)
    if feats.zones:
        fpts += _zone_rings(Q, holes, feats.zones, size, fpts, tol)
    if zlines:  # outline, beams and column faces all held by the smoothing
        feats = Features(points=list(feats.points), lines=list(feats.lines) + [(tuple(a), tuple(b)) for a, b in zlines],
                         soft=list(feats.soft), zones=list(feats.zones))
    fpts = list(_dedupe(fpts, tol)) if fpts else fpts
    bnd = _sample_chain(list(Q), h2, fpts, tol)
    hole_pts = [p for H in holes for p in _sample_chain(list(H), h2, fpts, tol)]
    line_pts = []
    for a, b in feats.lines:
        line_pts += _sample_chain([np.asarray(a), np.asarray(b)], h2, fpts, tol)[:-0 or None]
        line_pts.append(np.asarray(b, float))
    fixed = [np.asarray(p) for p in bnd + hole_pts + line_pts] + [p for p in fpts if inside(p, Q, holes, tol)]

    def is_in(c):
        return pg.inside(c, Q) and not any(pg.inside(c, H) for H in holes)
    # fill: a triangular lattice, kept away from the fixed points and from the
    # outline, opening and feature segments themselves
    lo, hi = Q.min(axis=0), Q.max(axis=0)
    F = np.array(fixed) if fixed else np.zeros((0, 2))
    segs = [(Q[i], Q[(i + 1) % len(Q)]) for i in range(len(Q))]
    for H in holes:
        segs += [(H[i], H[(i + 1) % len(H)]) for i in range(len(H))]
    segs += [(np.asarray(a, float), np.asarray(b, float)) for a, b in feats.lines]
    SA = np.array([a for a, _ in segs], float).reshape(-1, 2)
    SB = np.array([b for _, b in segs], float).reshape(-1, 2)

    def seg_dist(c):
        d = SB - SA
        L2 = np.einsum("ij,ij->i", d, d)
        t = np.clip(np.einsum("ij,ij->i", c - SA, d) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        return float(np.min(np.linalg.norm(SA + d * t[:, None] - c, axis=1))) if len(SA) else math.inf
    fill = []
    dy = h2 * math.sqrt(3) / 2
    y, row = lo[1] + dy / 2, 0
    while y < hi[1]:
        x = lo[0] + (h2 / 2 if row % 2 else 0.0) + h2 / 4
        while x < hi[0]:
            c = np.array([x, y])
            if is_in(c) and (not len(F) or np.min(np.linalg.norm(F - c, axis=1)) > 0.6 * h2) \
                    and seg_dist(c) > 0.45 * h2 and not any(pg.inside(c, Z) for Z, _ in feats.zones):
                fill.append(c)
            x += h2
        y += dy
        row += 1
    pts = _dedupe(fixed + fill, tol)
    constraints = _segments(Q, holes, feats, pts, tol)
    for _ in range(12):  # conforming Delaunay: split segments missing from the triangulation
        tri = Delaunay(pts)
        edges = set()
        for s in tri.simplices:
            for k in range(3):
                edges.add(frozenset((int(s[k]), int(s[(k + 1) % 3]))))
        missing = [(i, j) for i, j in constraints if frozenset((i, j)) not in edges]
        if not missing:
            break
        new = []
        for i, j in missing:
            m = (pts[i] + pts[j]) / 2
            d = np.linalg.norm(pts - m, axis=1)
            k = int(np.argmin(d))
            if d[k] > tol:  # reuse a point already there
                pts = np.vstack([pts, m])
                k = len(pts) - 1
            if k in (i, j):
                continue
            new += [(i, k), (k, j)]
        constraints = [c for c in constraints if c not in missing] + new
    tris = [s for s in tri.simplices if is_in(pts[s].mean(axis=0)) and abs(pg.signed_area(pts[s])) > tol * tol]
    # triangles -> 3 quads each
    P = [p for p in pts]
    mid: Dict[frozenset, int] = {}

    def midpoint(i, j):
        k = frozenset((i, j))
        if k not in mid:
            mid[k] = len(P)
            P.append((P[i] + P[j]) / 2)
        return mid[k]
    cells = []
    for s in tris:
        a, b, c = (int(v) for v in s)
        if pg.signed_area(np.array([P[a], P[b], P[c]])) < 0:
            b, c = c, b
        g = len(P)
        P.append((P[a] + P[b] + P[c]) / 3)
        mab, mbc, mca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        cells += [(a, mab, g, mca), (b, mbc, g, mab), (c, mca, g, mbc)]
    P = np.array(P, float)
    used = sorted({i for c in cells for i in c})
    remap = {o: n for n, o in enumerate(used)}
    P = P[used]
    cells = np.array([[remap[i] for i in c] for c in cells], int).reshape(-1, 4)
    _smooth(P, cells, feats, tol, smooth)
    return P, cells


def _cross(a, b, c, d, tol):
    """Crossing point of segments ab and cd (not at their ends), or None."""
    r, q = b - a, d - c
    den = r[0] * q[1] - r[1] * q[0]
    if abs(den) <= 1e-12:
        return None
    t = ((c[0] - a[0]) * q[1] - (c[1] - a[1]) * q[0]) / den
    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
    La, Lc = float(np.linalg.norm(r)), float(np.linalg.norm(q))
    if tol / La < t < 1 - tol / La and -tol / Lc <= u <= 1 + tol / Lc:
        return a + t * r
    return None


def _zone_rings(Q, holes, zones, size, fixed, tol):
    """Points on rings around the column cross-sections (offsets of their
    outline), the spacing growing by ``ZONE_GROWTH`` from ring to ring until it
    reaches the shell's own (2 x size); and where the rings cross the shell
    outline, so the outline is graded too."""
    out = []
    F = [np.asarray(p, float) for p in fixed]
    edges = [(Q[i], Q[(i + 1) % len(Q)]) for i in range(len(Q))]
    for H in holes:
        edges += [(H[i], H[(i + 1) % len(H)]) for i in range(len(H))]

    def dist_outline(c):
        best = math.inf
        for a, b in edges:
            d = b - a
            L2 = float(d @ d)
            t = 0.0 if L2 <= 0 else min(max(float((c - a) @ d) / L2, 0.0), 1.0)
            best = min(best, float(np.linalg.norm(a + t * d - c)))
        return best
    g = ZONE_GROWTH - 1.0
    for Z, hz in zones:
        Z = np.asarray(Z, float)
        if len(Z) < 3:
            continue
        if not pg.is_convex(Z):  # an edge column cut by a curved / bent edge: rings around its hull
            from scipy.spatial import ConvexHull
            Z = Z[ConvexHull(Z).vertices]
        off, t = 0.0, 2.0 * hz
        while t < 1.9 * size:
            off += 0.87 * t
            t = min(2.0 * size, 2.0 * (hz + g * off))
            R = pg.offset_polygon(Z, off)
            ring = _sample_chain(list(R), t, [], tol)
            for i in range(len(R)):  # where the ring crosses the outline
                for a, b in edges:
                    x = _cross(R[i], R[(i + 1) % len(R)], a, b, tol)
                    if x is not None:
                        ring.append(x)
            for c in ring:
                c = np.asarray(c, float)
                on_out = dist_outline(c) <= tol
                if not on_out and (not pg.inside(c, Q) or any(pg.inside(c, H) for H in holes)
                                   or dist_outline(c) < 0.45 * t):
                    continue
                if any(pg.inside(c, Zo) for Zo, _ in zones):
                    continue
                if F and min(float(np.linalg.norm(c - p)) for p in F) < 0.55 * t:
                    continue
                out.append(c)
                F.append(c)
    return out


def inside(p, Q, holes, tol) -> bool:
    return pg.inside(p, Q, edge_tol=tol) and not any(pg.inside(p, H) and not _on_boundary(p, H, tol) for H in holes)


def _dedupe(pts, tol):
    out = []
    for p in pts:
        if not out or np.min(np.linalg.norm(np.array(out) - p, axis=1)) > tol:
            out.append(np.asarray(p, float))
    return np.array(out, float)


def _segments(Q, holes, feats, pts, tol):
    """Index pairs of consecutive points along the outline, openings and lines."""
    def chain(poly, closed):
        idx = []
        n = len(poly) if closed else len(poly) - 1
        for i in range(n):
            a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
            d = b - a
            L2 = float(d @ d)
            if L2 <= 0:
                continue
            t = ((pts - a) @ d) / L2
            off = np.linalg.norm(pts - (a + np.outer(t, d)), axis=1)
            on = np.nonzero((off <= tol) & (t >= -1e-9) & (t <= 1 + 1e-9))[0]
            on = on[np.argsort(t[on])]
            idx += [(int(i0), int(i1)) for i0, i1 in zip(on[:-1], on[1:])]
        return idx
    segs = chain(Q, True)
    for H in holes:
        segs += chain(H, True)
    for a, b in feats.lines:
        segs += chain([a, b], False)
    out, seen = [], set()
    for i, j in segs:  # one entry per segment, whichever way it was listed
        k = frozenset((i, j))
        if i != j and k not in seen:
            seen.add(k)
            out.append((i, j))
    return out


def _smooth(P, cells, feats, tol, passes):
    """Laplacian smoothing of interior, non-feature nodes; a move is kept only if
    every element around the node stays valid (or gets better)."""
    if passes <= 0 or not len(cells):
        return
    bnd = _boundary(cells)
    keep = set(bnd)
    F = [np.asarray(p, float) for p in feats.points]
    lines = [(np.asarray(a, float), np.asarray(b, float)) for a, b in feats.lines]
    for i, p in enumerate(P):
        if any(np.linalg.norm(p - f) <= tol for f in F):
            keep.add(i)
            continue
        for a, b in lines:
            d = b - a
            L2 = float(d @ d)
            t = float((p - a) @ d) / L2 if L2 > 0 else 0.0
            if -1e-9 <= t <= 1 + 1e-9 and np.linalg.norm(a + t * d - p) <= tol:
                keep.add(i)
                break
    nbr: Dict[int, Set[int]] = {}
    node_cells: Dict[int, List[int]] = {}
    for ci, c in enumerate(cells):
        for k in range(4):
            nbr.setdefault(int(c[k]), set()).update((int(c[k - 1]), int(c[(k + 1) % 4])))
            node_cells.setdefault(int(c[k]), []).append(ci)

    # pure-Python geometry on a list copy (numpy per-element calls dominate otherwise)
    Pl = P.tolist()
    cl = [tuple(int(v) for v in c) for c in cells]

    def cell_q(ci):
        """Worst corner sine, and edge balance (shortest / longest edge x 2.5) -
        a corner can look fine on an element with one collapsed edge."""
        X = [Pl[k] for k in cl[ci]]
        area2 = 0.0
        for k in range(4):
            x0, y0 = X[k]
            x1, y1 = X[(k + 1) % 4]
            area2 += x0 * y1 - x1 * y0
        if area2 <= 0:
            return -1.0
        L = []
        for k in range(4):
            x0, y0 = X[k]
            x1, y1 = X[(k + 1) % 4]
            L.append(math.hypot(x1 - x0, y1 - y0))
        q = min(1.0, 2.5 * min(L) / (max(L) + 1e-30))
        for k in range(4):
            px, py = X[k]
            ux, uy = X[k - 1][0] - px, X[k - 1][1] - py
            vx, vy = X[(k + 1) % 4][0] - px, X[(k + 1) % 4][1] - py
            q = min(q, (uy * vx - ux * vy) / (L[k - 1] * L[k] + 1e-30))
        return q

    def worst(n):
        return min(cell_q(ci) for ci in node_cells[n])
    for _ in range(passes):  # smart Laplacian: a move must not make the worst corner worse
        for n in nbr:
            if n in keep:
                continue
            old = Pl[n][:]
            before = worst(n)
            nb = nbr[n]
            tx = sum(Pl[m][0] for m in nb) / len(nb)
            ty = sum(Pl[m][1] for m in nb) / len(nb)
            for f in (1.0, 0.5, 0.25):
                Pl[n] = [old[0] + f * (tx - old[0]), old[1] + f * (ty - old[1])]
                if worst(n) >= before - 1e-12:
                    break
            else:
                Pl[n] = old
    # repair what is still poor (corners under ~25 deg / over ~155 deg): pattern search
    # on the movable nodes of those elements for the best worst-corner around them
    limit = math.sin(math.radians(25))
    dirs = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    for _ in range(6):
        bad = [ci for ci in range(len(cl)) if cell_q(ci) < limit]
        if not bad:
            break
        for ci in bad:
            for n in cl[ci]:
                if n in keep:
                    continue
                L = min(math.hypot(Pl[m][0] - Pl[n][0], Pl[m][1] - Pl[n][1]) for m in nbr[n])
                best, best_q = Pl[n][:], worst(n)
                for step in (0.5 * L, 0.25 * L, 0.1 * L):
                    for dx, dy in dirs:
                        Pl[n] = [best[0] + step * dx, best[1] + step * dy]
                        q = worst(n)
                        if q > best_q + 1e-9:
                            best, best_q = Pl[n][:], q
                    Pl[n] = best
                Pl[n] = best
    P[:] = np.asarray(Pl, float)


# ------------------------------------------------------------ automatic size
# Default rule "etabs": an approximate maximum element size of 0.5 m for floors and
# walls (the ETABS default of 48 in = 1.22 m was found too coarse for slab moments
# and wall stresses: 2026-11, user decision) with mesh lines through the columns,
# beams, walls and joints of the floor (the features of tensor_mesh): each span
# between them is split into equal elements no larger than that (6 m: 12 x 0.5 m;
# 7 m: 14 x 0.5 m; 5.5 m: 11 x 0.5 m).  Imported ETABS files and the size set in
# Analyze > Automatic Mesh Settings keep their own value.
# Rule "fine": the earlier, finer Structural Studio mesh (at least 8 elements
# across a shell and 6 between columns, never coarser than 1 m) - several times
# as many elements.  Analyze > Check Mesh refines either one.
AUTO_RULE = "etabs"
DEFAULT_MAX_SIZE = 0.5  # m
ETABS_MAX_SIZE = DEFAULT_MAX_SIZE  # (name kept for callers)
AUTO_MAX_SIZE = 1.0  # m, rule "fine": never coarser than this
AUTO_MIN_SIZE = 0.1  # m: never finer than this (unless the shell itself is smaller)
AUTO_ACROSS = 8  # elements across the narrowest width of a shell
AUTO_BETWEEN = 6  # elements between neighbouring supports / loaded joints
AUTO_MAX_ELEMENTS = 2500  # per shell object


def auto_size(Q: np.ndarray, feats: Optional[Features] = None, tol: float = 1e-6,
              max_size: Optional[float] = None, factor: float = 1.0, rule: Optional[str] = None) -> float:
    """Element size (SI) for one shell.

    Rule "etabs" (default, :data:`AUTO_RULE`): the approximate maximum mesh size
    :data:`DEFAULT_MAX_SIZE` (0.5 m) times ``factor`` (a mesh check halves it:
    0.25, 0.125 m...).

    Rule "fine", from the shell's geometry:

    * at least ``AUTO_ACROSS`` elements across the narrowest width, taken as
      the smaller of the short side of the bounding box and 4A/perimeter
      (= the side of a square, 2ab/(a+b) of a rectangle);
    * at least ``AUTO_BETWEEN`` elements between neighbouring features
      (columns, supports, loaded joints) - the spans of the slab;
    * no coarser than ``max_size`` and no finer than ``AUTO_MIN_SIZE`` (or
      whatever keeps a shell under ``AUTO_MAX_ELEMENTS``).

    ``factor`` scales the result (mesh refinement checks use 0.5, 0.25...).
    The size is rounded down to ``max_size / 2^k`` so that the meshes of
    neighbouring shells (floors and walls) nest into one another."""
    rule = rule or AUTO_RULE
    if rule == "etabs":
        top = DEFAULT_MAX_SIZE if max_size is None else max_size
        h = top * factor
        k = max(0, math.ceil(math.log2(top / h) - 1e-9))
        return top / 2 ** k
    if max_size is None:
        max_size = AUTO_MAX_SIZE
    Q = np.asarray(Q, float)
    ext = np.ptp(Q, axis=0)
    A = abs(pg.signed_area(Q))
    per = float(np.sum(np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1)))
    width = min(float(ext.min()), 4.0 * A / per if per > 0 else float(ext.min()))
    h = width / AUTO_ACROSS
    pts = list(feats.points) if feats else []
    if len(pts) >= 2:
        P = np.unique(np.round(np.asarray(pts, float) / max(tol, 1e-9)), axis=0) * max(tol, 1e-9)
        if len(P) >= 2:
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            D[D <= 10 * tol] = np.inf
            nn = D.min(axis=1)
            nn = nn[np.isfinite(nn)]
            if len(nn):
                # a typical span, not the closest pair (two joints 10 cm apart
                # should not refine the whole slab)
                h = min(h, float(np.percentile(nn, 25)) / AUTO_BETWEEN)
    h = min(h, max_size)
    floor = max(AUTO_MIN_SIZE, math.sqrt(A / AUTO_MAX_ELEMENTS) if A > 0 else 0.0)
    h = max(h, min(floor, width / 2))
    h *= factor
    # round down to max_size / 2^k (1, 0.5, 0.25, 0.125 m...): sizes of
    # neighbouring shells then divide one another and their mesh lines meet
    k = max(0, math.ceil(math.log2(max_size / h) - 1e-9))
    return max_size / 2 ** k
