"""Where and how to divide shells (Edit > Edit Shells > Divide Shells).

A shell is cut by planes: each plane that crosses it along a straight chord splits
it in two, and every piece is cut by the next plane.  The finders give the planes:

* equal pieces n1 x n2 / a largest piece size (a quadrilateral is divided along its
  own edges, any other shape along its local axes 1 and 2)
* frames drawn on the shell (beams over a slab, a frame across a wall) - the shell is
  cut along the frame where the frame runs right across it
* grid lines and arcs (in plan: a floor is cut along them - along an arc in short
  straight pieces -, a wall where they cross it); the grids shown, every grid or
  chosen grids
* story levels (walls and ramps through several stories)

A cut along a chord that would leave a sliver, or that crosses a concave outline more
than once, is not made.  Every piece keeps the section, loads (and the floor's area
loads), label and options of the shell it came from.  Lengths are in model units.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Plane = Tuple[np.ndarray, np.ndarray]  # (a point on it, its unit normal)
ARC_STEP_DEG = 5.0  # an arc grid cuts a floor in straight pieces of at most this angle


class Path:
    """A cut along a polyline lying on the shell (an arc grid line on a floor)."""

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, float)


# ------------------------------------------------------------- geometry
def _xyz(model, area) -> np.ndarray:
    return np.array([model.points[p].xyz for p in area.points], float)


def _normal(P: np.ndarray) -> np.ndarray:
    """Newell's normal of a polygon (unit; zero for a degenerate one)."""
    n = np.zeros(3)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                       (a[0] - b[0]) * (a[1] + b[1])])
    L = float(np.linalg.norm(n))
    return n / L if L > 0 else n


def _poly_area(P: np.ndarray) -> float:
    n = np.zeros(3)
    for i in range(len(P)):
        n += np.cross(P[i], P[(i + 1) % len(P)])
    return 0.5 * float(np.linalg.norm(n))


def _axes(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local axes 1, 2 (in the plane; 1 along the first edge) and 3 (the normal)."""
    n = _normal(P)
    e1 = P[1] - P[0]
    e1 = e1 - (e1 @ n) * n
    e1 = e1 / (np.linalg.norm(e1) or 1.0)
    return e1, np.cross(n, e1), n


def _signed(P: np.ndarray, plane: Plane, tol: float) -> np.ndarray:
    d = (P - plane[0]) @ plane[1]
    d[np.abs(d) <= tol] = 0.0
    return d


def _crossings(P: np.ndarray, d: np.ndarray) -> List[List[np.ndarray]]:
    """Where the outline passes from one side of the plane to the other: an edge
    crossing (one point), or a run of corners on the plane between the two sides (its
    points).  A run that only touches the plane (same side before and after) is not one."""
    k = len(P)
    out: List[List[np.ndarray]] = []
    start = next(i for i in range(k) if d[i] != 0)
    i = start
    for _ in range(k):
        j = (i + 1) % k
        if d[i] != 0 and d[j] != 0:
            if d[i] * d[j] < 0:
                out.append([P[i] + d[i] / (d[i] - d[j]) * (P[j] - P[i])])
        elif d[i] != 0 and d[j] == 0:  # a run of corners on the plane starts at j
            run, m = [], j
            while d[m] == 0:
                run.append(P[m])
                m = (m + 1) % k
            if d[i] * d[m] < 0:
                out.append(run)
        i = j
    return out


def chord(P: np.ndarray, plane: Plane, tol: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The two ends of the cut a plane makes across the shell, or None (it misses,
    only touches, runs along an edge, lies in the shell's plane, or crosses a concave
    outline more than once)."""
    d = _signed(P, plane, tol)
    if np.all(d >= 0) or np.all(d <= 0):
        return None
    xs = _crossings(P, d)
    if len(xs) != 2:
        return None
    (r1, r2) = xs
    # a run of corners on the plane (an edge along it): the cut starts at its corner
    # nearest the other side
    a = min(r1, key=lambda q: min(np.linalg.norm(q - x) for x in r2))
    b = min(r2, key=lambda q: np.linalg.norm(q - a))
    return a, b


def _tidy(Q: List[np.ndarray], tol: float) -> np.ndarray:
    """Drop repeated corners and spikes (a corner where the outline turns straight back)."""
    pts = [q for q in Q]
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        for i in range(len(pts)):
            a, v, b = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            u, w = v - a, b - v
            if np.linalg.norm(u) <= tol or (np.linalg.norm(np.cross(u, w)) <= tol * max(np.linalg.norm(u), np.linalg.norm(w), 1e-12)
                                             and float(u @ w) < 0):
                del pts[i]
                changed = True
                break
    return np.array(pts)


def split(P: np.ndarray, plane: Plane, tol: float, min_area: float) -> Optional[List[np.ndarray]]:
    """The two pieces a plane cuts a polygon into (outline order kept), or None."""
    if chord(P, plane, tol) is None:
        return None
    d = _signed(P, plane, tol)
    A: List[np.ndarray] = []
    B: List[np.ndarray] = []
    k = len(P)
    for i in range(k):
        j = (i + 1) % k
        if d[i] >= 0:
            A.append(P[i])
        if d[i] <= 0:
            B.append(P[i])
        if d[i] * d[j] < 0:
            x = P[i] + d[i] / (d[i] - d[j]) * (P[j] - P[i])
            A.append(x)
            B.append(x)
    out = [_tidy(q, tol) for q in (A, B)]
    if any(len(q) < 3 or _poly_area(q) <= min_area for q in out):
        return None
    if abs(sum(_poly_area(q) for q in out) - _poly_area(P)) > 1e-6 * _poly_area(P) + min_area:
        return None
    return out


def cut(P: np.ndarray, planes: Sequence[Plane], tol: float) -> List[np.ndarray]:
    """``P`` cut by every plane in turn."""
    min_area = max(tol * tol * 4, 1e-6 * _poly_area(P))
    pieces = [P]
    for pl in planes:
        nxt = []
        for q in pieces:
            s = split_path(q, pl.points, tol, min_area) if isinstance(pl, Path) else split(q, pl, tol, min_area)
            nxt += s if s else [q]
        pieces = nxt
    return pieces


def _seg_hits(a, b, c, d, eps=1e-12):
    """(s on a-b, r on c-d) where two 2-D segments cross, or None."""
    u, v, w = b - a, d - c, c - a
    den = u[0] * v[1] - u[1] * v[0]
    if abs(den) <= eps * (np.linalg.norm(u) * np.linalg.norm(v) or 1.0):
        return None
    s = (w[0] * v[1] - w[1] * v[0]) / den
    r = (w[0] * u[1] - w[1] * u[0]) / den
    if -1e-9 <= s <= 1 + 1e-9 and -1e-9 <= r <= 1 + 1e-9:
        return min(max(s, 0.0), 1.0), min(max(r, 0.0), 1.0)
    return None


def split_path(P: np.ndarray, path: np.ndarray, tol: float, min_area: float) -> Optional[List[np.ndarray]]:
    """The two pieces a polyline on the shell cuts it into, when it runs right across
    (in at one edge, out at another, once); None otherwise."""
    e1, e2, _ = _axes(P)
    O = P[0]
    P2 = np.c_[(P - O) @ e1, (P - O) @ e2]
    Q2 = np.c_[(path - O) @ e1, (path - O) @ e2]
    k, m = len(P2), len(Q2)
    hits = []  # (position along the path, edge, position on the edge, the 3-D point)
    for j in range(m - 1):
        for i in range(k):
            h = _seg_hits(Q2[j], Q2[j + 1], P2[i], P2[(i + 1) % k])
            if h is None:
                continue
            x = P[i] + h[1] * (P[(i + 1) % k] - P[i])
            if any(np.linalg.norm(x - y[3]) <= tol for y in hits):  # through a corner: once
                continue
            hits.append((j + h[0], i, h[1], x))
    if len(hits) != 2:
        return None
    (t1, i1, r1, X), (t2, i2, r2, Y) = sorted(hits, key=lambda z: z[0])
    inner = [path[j] for j in range(m) if t1 < j < t2]

    def along(i, r, x, i_to, r_to, y):  # the outline from x (on edge i) forward to y
        pts, n = [x], 0
        if not (i == i_to and r < r_to):
            n = 1
            while True:
                pts.append(P[(i + n) % k])
                if (i + n) % k == i_to:
                    break
                n += 1
        return pts + [y]
    A = along(i1, r1, X, i2, r2, Y) + inner[::-1]
    B = along(i2, r2, Y, i1, r1, X) + inner
    out = [_tidy(q, tol) for q in (A, B)]
    if any(len(q) < 3 or _poly_area(q) <= min_area for q in out):
        return None
    if abs(sum(_poly_area(q) for q in out) - _poly_area(P)) > 1e-6 * _poly_area(P) + min_area:
        return None
    return out


def _quad_grid(P: np.ndarray, n1: int, n2: int) -> List[np.ndarray]:
    """A quadrilateral divided n1 x n2 along its own edges (bilinear)."""
    c = P
    g = [[(1 - s) * (1 - t) * c[0] + s * (1 - t) * c[1] + s * t * c[2] + (1 - s) * t * c[3]
          for t in (j / n2 for j in range(n2 + 1))] for s in (i / n1 for i in range(n1 + 1))]
    return [np.array([g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]])
            for i in range(n1) for j in range(n2)]


def _axis_planes(P: np.ndarray, n1: int, n2: int) -> List[Plane]:
    e1, e2, _ = _axes(P)
    out: List[Plane] = []
    for e, n in ((e1, n1), (e2, n2)):
        u = (P - P[0]) @ e
        lo, hi = float(u.min()), float(u.max())
        out += [(P[0] + (lo + (hi - lo) * k / n) * e, e) for k in range(1, n)]
    return out


# ------------------------------------------------------------- the finders
def pieces_equal(model, area, n1: int, n2: int) -> List[np.ndarray]:
    P = _xyz(model, area)
    n1, n2 = max(1, int(n1)), max(1, int(n2))
    if n1 == 1 and n2 == 1:
        return [P]
    if len(P) == 4:
        return _quad_grid(P, n1, n2)
    return cut(P, _axis_planes(P, n1, n2), model.options.merge_tolerance)


def pieces_max_size(model, area, size: float) -> List[np.ndarray]:
    """The fewest equal pieces none larger than ``size`` along either side."""
    P = _xyz(model, area)
    if not size or size <= 0:
        return [P]
    if len(P) == 4:
        L1 = max(np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[2] - P[3]))
        L2 = max(np.linalg.norm(P[3] - P[0]), np.linalg.norm(P[2] - P[1]))
    else:
        e1, e2, _ = _axes(P)
        L1, L2 = [float(np.ptp((P - P[0]) @ e)) for e in (e1, e2)]
    return pieces_equal(model, area, int(math.ceil(L1 / size - 1e-9)), int(math.ceil(L2 / size - 1e-9)))


def planes_at_frames(model, area, frames: Optional[Sequence[str]] = None,
                     tol: Optional[float] = None) -> List[Plane]:
    """Along frames drawn on the shell (in its plane) that run right across it: the
    frame covers the whole chord.  A frame that stops inside the shell does not cut it."""
    tol = max(float(tol or 0.0), model.options.merge_tolerance)
    P = _xyz(model, area)
    n = _normal(P)
    if not np.any(n):
        return []
    lo, hi = P.min(axis=0) - tol, P.max(axis=0) + tol
    out: List[Plane] = []
    for fn in (frames if frames is not None else model.frames):
        f = model.frames.get(fn)
        if f is None or f.point_i not in model.points or f.point_j not in model.points:
            continue
        a = np.asarray(model.points[f.point_i].xyz, float)
        b = np.asarray(model.points[f.point_j].xyz, float)
        if np.any(np.minimum(a, b) > hi) or np.any(np.maximum(a, b) < lo):
            continue
        if abs((a - P[0]) @ n) > tol or abs((b - P[0]) @ n) > tol:  # not in the shell's plane
            continue
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= tol:
            continue
        u = d / L
        pl = (a, np.cross(u, n) / (np.linalg.norm(np.cross(u, n)) or 1.0))
        ch = chord(P, pl, tol)
        if ch is None:
            continue
        s = [float((q - a) @ u) for q in ch]
        if min(s) >= -tol and max(s) <= L + tol:  # the frame runs right across
            out.append(pl)
    return out


def planes_at_grids(model, area, tol: Optional[float] = None, grids="shown") -> list:
    """The cuts along grid lines that reach the shell: vertical planes through the
    straight ones; an arc cuts a floor along short straight pieces (a :class:`Path`) and
    a wall where it crosses it.  ``grids``: "shown", "all" or a list of (system, name)."""
    from .frame_divide import grid_set
    tol = max(float(tol or 0.0), model.options.merge_tolerance)
    P = _xyz(model, area)
    gg = grid_set(model, grids)
    out: list = []
    for p, q, _, _ in gg.lines:
        p, q = np.asarray(p, float), np.asarray(q, float)
        d = q - p
        L = float(np.linalg.norm(d))
        if L <= tol:
            continue
        u = d / L
        pl = (np.r_[p, 0.0], np.array([-u[1], u[0], 0.0]))
        ch = chord(P, pl, tol)
        if ch is None:
            continue
        s = sorted(float((c[:2] - p) @ u) for c in ch)
        if s[1] >= -tol and s[0] <= L + tol:  # the chord is where the grid line is drawn
            out.append(pl)
    n = _normal(P)
    for c, r, a0, a1, _, _ in gg.arcs:
        c = np.asarray(c, float)
        lo, hi = min(a0, a1), max(a0, a1)
        if abs(n[2]) > 1e-3:  # a floor (or a ramp): the arc on the shell's plane
            k = max(2, int(math.ceil((hi - lo) / ARC_STEP_DEG)))
            ang = np.radians(np.linspace(lo, hi, k + 1))
            xy = np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)]
            z = P[0][2] - ((xy[:, 0] - P[0][0]) * n[0] + (xy[:, 1] - P[0][1]) * n[1]) / n[2]
            out.append(Path(np.c_[xy, z]))
        else:  # a wall: vertical cuts where the arc crosses the wall's line in plan
            h = np.cross(n, [0.0, 0.0, 1.0])
            h2 = h[:2] / (np.linalg.norm(h[:2]) or 1.0)
            w = P[0][:2] - c
            B, C = 2 * float(w @ h2), float(w @ w) - r * r
            disc = B * B - 4 * C
            if disc < 0:
                continue
            for t in ((-B - math.sqrt(disc)) / 2, (-B + math.sqrt(disc)) / 2):
                x = P[0][:2] + t * h2
                ang = math.degrees(math.atan2(x[1] - c[1], x[0] - c[0]))
                if any(lo - 1e-6 <= ang + kk * 360 <= hi + 1e-6 for kk in (-1, 0, 1)):
                    out.append((np.r_[x, 0.0], np.r_[h2, 0.0]))
    return out


def planes_at_stories(model, area) -> List[Plane]:
    elev = [s.elevation for s in model.stories_bottom_up()] if hasattr(model, "stories_bottom_up") else []
    return [(np.array([0.0, 0.0, z]), np.array([0.0, 0.0, 1.0])) for z in elev]


def pieces_for(model, areas: Sequence[str], kind: str, v: dict,
               selected_frames: Sequence[str] = ()) -> Dict[str, List[np.ndarray]]:
    """The pieces of each shell for a choice of the Divide Shells dialog (``kind``
    "equal" / "size" / "cross", ``v`` its values) - the edit and its preview alike."""
    out: Dict[str, List[np.ndarray]] = {}
    for n in areas:
        a = model.areas.get(n)
        if a is None or a.type == "Opening":
            continue
        if kind == "equal":
            out[n] = pieces_equal(model, a, v["n1"], v["n2"])
        elif kind == "size":
            out[n] = pieces_max_size(model, a, v["size"])
        else:
            planes: list = []
            if v.get("frames"):
                pool = [f for f in selected_frames if f in model.frames] if v.get("selected_frames") else None
                planes += planes_at_frames(model, a, pool, v.get("tol"))
            if v.get("grids"):
                which = v.get("grid_which", "shown")
                if which in ("shown", "all") or which:
                    planes += planes_at_grids(model, a, v.get("tol"), grids=which)
            if v.get("stories"):
                planes += planes_at_stories(model, a)
            out[n] = cut(_xyz(model, a), planes, model.options.merge_tolerance)
    return out


def preview_2d(model, pieces: Dict[str, List[np.ndarray]]):
    """The shells and their pieces flattened for a picture: (outlines, pieces, frames),
    each a list of N x 2 arrays.  All in plan when every shell is (nearly) level, else
    in the plane of the first shell (seen from the side its axis 3 points to); the
    frames lying on the shells are drawn too."""
    names = [n for n in pieces if n in model.areas]
    if not names:
        return [], [], []
    Ps = {n: _xyz(model, model.areas[n]) for n in names}
    if all(abs(_normal(P)[2]) > 0.99 for P in Ps.values()):
        O, e1, e2 = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    else:
        first = Ps[names[0]]
        n3 = _normal(first)
        e1 = np.cross([0.0, 0.0, 1.0], n3)
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.array([1.0, 0, 0])
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n3, e1)
        O = first[0]
    flat = lambda Q: np.c_[(np.asarray(Q) - O) @ e1, (np.asarray(Q) - O) @ e2]
    outlines = [flat(P) for P in Ps.values()]
    parts = [flat(q) for n in names for q in pieces[n]]
    frames = []
    tol = model.options.merge_tolerance
    for f in model.frames.values():
        try:
            ab = np.array([model.points[f.point_i].xyz, model.points[f.point_j].xyz], float)
        except KeyError:
            continue
        for P in Ps.values():
            n3 = _normal(P)
            if np.all(np.abs((ab - P[0]) @ n3) <= max(tol, 1e-6)):
                lo, hi = P.min(axis=0) - tol, P.max(axis=0) + tol
                if np.all(ab.max(axis=0) >= lo) and np.all(ab.min(axis=0) <= hi):
                    frames.append(flat(ab))
                    break
    return outlines, parts, frames


# ------------------------------------------------------------- the edit
def divide(doc, pieces_of: Dict[str, List[np.ndarray]]) -> List[str]:
    """Replace each shell by its pieces (one Undo).  Returns the new shells."""
    from .commands import _area_kwargs
    created: List[str] = []
    todo = {n: ps for n, ps in pieces_of.items() if n in doc.model.areas and len(ps) > 1}
    if not todo:
        return created

    def mutate(model):
        for n, ps in todo.items():
            originals = [p for p in model.panel_loads.values() if p.area == n]
            for p in originals:
                del model.panel_loads[p.name]
            a = model.areas.pop(n)
            for q in ps:
                pts = []
                for xyz in q:
                    pt = model.add_point(*map(float, xyz)).name
                    if not pts or pts[-1] != pt:
                        pts.append(pt)
                if len(pts) > 1 and pts[0] == pts[-1]:
                    pts.pop()
                if len(pts) < 3:
                    continue
                b = model.add_area(pts, a.section, type=a.type, **_area_kwargs(a))
                b.loads = copy.deepcopy(a.loads)
                b.label = a.label
                for p in originals:  # every piece keeps the floor's area loads
                    r = model.add_area_panel(b.name, copy.deepcopy(p.loads), distribution=p.distribution,
                                             span_dir=p.span_dir)
                    r.receivers, r.label = p.receivers, p.label
                created.append(b.name)

    doc.edit(mutate)
    return created
