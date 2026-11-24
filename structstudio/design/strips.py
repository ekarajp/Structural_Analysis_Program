"""Slab design strips laid out from the structure (Edit > Add/Edit Design Strips).

A design strip only says how the slab forces are added up for design (the analysis
does not use it), so the layout follows where the slab is actually supported:

1. **Supports of the floor** at the level: columns and piles (frames running up or
   down from a joint of the floor), supports and springs of the joints, walls (their
   edges at the level), line springs and the beams that are stiff enough to act as a
   line of supports.  A beam counts when ``alpha_f * l2 / l1 >= beam_alpha`` (ACI 318
   8.10.2.7: alpha_f = Ecb Ib / (Ecs Is), Ib of the beam with its slab flange as ACI
   318 8.4.1.8, Is of the slab width l2 it carries); a shallow beam is part of the slab
   and the floor works as a flat slab there.
2. **Column strips** along the lines of supports, in two families (layer A along the
   first direction, B across it): the grid lines that have supports on them, then
   strips through the supports not yet on one - from each support to the next one
   ahead (turning at most ``align_angle`` from the direction), over the slab.  A strip
   runs on to the slab edge past its last support (end span, cantilever).
3. **Widths** (ACI 318 8.4.1.5): on each side 0.25 * min(l1, l2) - l1 the span between
   the supports along the strip, l2 the distance to the next line of supports on that
   side - and up to the slab edge on a side with no next line.
4. **Middle strips**: what lies between two column strips (and between a column strip
   and the slab edge).

Everything is in plan coordinates (x, y) at the level of the story; the strips made
are :class:`structstudio.model.objects.DesignStrip` objects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

XY = Tuple[float, float]
BIG = 1e30


# ------------------------------------------------------------------ options / result
@dataclass
class StripOptions:
    """Auto Design Strips settings.  Lengths in metres (converted to the model units)."""

    source: str = "auto"  # auto (grid lines with supports, then the supports) | grids | supports
    beam_alpha: float = 1.0  # a beam is a line of supports when alpha_f * l2 / l1 >= this
    align_angle: float = 15.0  # degrees: the most a column strip turns from its direction
    on_line_tol: float = 0.30  # m: a support this close to a grid line is on it
    middle: bool = True  # make the middle strips
    keep_user: bool = True  # keep the strips drawn by hand (auto ones are made again)
    layers: Tuple[str, ...] = ("A", "B")


@dataclass
class StripLayout:
    story: str
    strips: List[dict] = field(default_factory=list)  # kwargs of Model.add_design_strip
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    supports: Dict[str, int] = field(default_factory=dict)  # counts by kind


# ------------------------------------------------------------------ plan geometry
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _mul(a, k):
    return (a[0] * k, a[1] * k)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _norm(a):
    return math.hypot(a[0], a[1])


def _unit(a):
    L = _norm(a)
    return (a[0] / L, a[1] / L) if L > 0 else (1.0, 0.0)


def _left(d):
    return (-d[1], d[0])


def point_in_polygon(p: XY, poly: Sequence[XY]) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xc = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < xc:
                inside = not inside
        j = i
    return inside


def _seg_dist(p, a, b) -> float:
    ab = _sub(b, a)
    L2 = _dot(ab, ab)
    t = 0.0 if L2 <= 0 else max(0.0, min(1.0, _dot(_sub(p, a), ab) / L2))
    return _norm(_sub(p, _add(a, _mul(ab, t))))


def _ray_seg(o, d, a, b) -> Optional[float]:
    """Distance along the ray o + t d (d unit) to segment ab (None: no hit)."""
    e = _sub(b, a)
    den = _cross(d, e)
    if abs(den) < 1e-14:
        return None
    w = _sub(a, o)
    t = _cross(w, e) / den
    s = _cross(w, d) / den
    if t >= 0 and -1e-9 <= s <= 1 + 1e-9:
        return t
    return None


class Floor:
    """The slab of one level in plan: polygons (the floor objects), an outline test and
    distances to its edge."""

    def __init__(self, polys: List[List[XY]], tol: float):
        self.polys = [p for p in polys if len(p) >= 3]
        self.tol = tol
        self.near = tol  # (layout: the distance a support counts as on a line)
        edges = [(p[i], p[(i + 1) % len(p)]) for p in self.polys for i in range(len(p))]
        # the edge between two floor objects side by side (the same two corners, the other
        # way round) is inside the slab: only the outline is kept (far fewer edges for a
        # floor of many bays - the rays and the edge distances run over the outline)
        g = max(tol, 1e-9)

        def key(a, b):
            return (round(a[0] / g), round(a[1] / g), round(b[0] / g), round(b[1] / g))
        count: Dict[tuple, int] = {}
        for a, b in edges:
            count[key(a, b)] = count.get(key(a, b), 0) + 1
        self.edges = [(a, b) for a, b in edges if not count.get(key(b, a))]
        self.boxes = [(min(q[0] for q in p), max(q[0] for q in p), min(q[1] for q in p), max(q[1] for q in p))
                      for p in self.polys]
        xs = [q[0] for p in self.polys for q in p] or [0.0]
        ys = [q[1] for p in self.polys for q in p] or [0.0]
        self.box = (min(xs), max(xs), min(ys), max(ys))
        self.area = sum(abs(_poly_area(p)) for p in self.polys)

    def _in_poly(self, p: XY) -> bool:
        x, y = p
        for poly, (x0, x1, y0, y1) in zip(self.polys, self.boxes):
            if x0 <= x <= x1 and y0 <= y <= y1 and point_in_polygon(p, poly):
                return True
        return False

    def inside(self, p: XY, tol: Optional[float] = None) -> bool:
        tol = self.tol if tol is None else tol
        if self._in_poly(p):
            return True
        x, y = p
        b = self.box
        if x < b[0] - tol or x > b[1] + tol or y < b[2] - tol or y > b[3] + tol:
            return False
        return any(_seg_dist(p, a, b) <= tol for a, b in self.edges)

    def exit(self, o: XY, d: XY) -> float:
        """How far the ray from ``o`` along unit ``d`` runs over the slab (0: ``o`` is on
        the edge, going out).  A start just outside the edge (within ``near``: a strip on
        a circle grid line beside the faceted edge of a round slab) counts from where the
        ray comes onto the slab."""
        ts = sorted(t for a, b in self.edges for t in [_ray_seg(o, d, a, b)] if t is not None and t > 1e-9)
        prev = 0.0
        first = True
        for t in ts:
            mid = 0.5 * (prev + t)
            if not self.inside(_add(o, _mul(d, mid))):  # (a run along the slab edge is over the slab)
                if first and t <= self.near and not self._in_poly(o):
                    prev = t
                    first = False
                    continue
                return prev
            first = False
            prev = t
        return prev

    def nearest_edge_point(self, p: XY) -> Tuple[XY, float]:
        best, bd = p, BIG
        for a, b in self.edges:
            ab = _sub(b, a)
            L2 = _dot(ab, ab)
            t = 0.0 if L2 <= 0 else max(0.0, min(1.0, _dot(_sub(p, a), ab) / L2))
            q = _add(a, _mul(ab, t))
            d = _norm(_sub(p, q))
            if d < bd:
                best, bd = q, d
        return best, bd

    def snap_ends(self, pts: List[XY]) -> List[XY]:
        """A line whose ends lie just off the slab (within ``near``): its ends moved onto
        the slab edge."""
        pts = list(pts)
        for end in (0, -1):
            p = pts[end]
            if not self._in_poly(p):
                q, d = self.nearest_edge_point(p)
                if d <= self.near:
                    pts[end] = q
        return _simplify_keep(pts, self.tol)

    def covers(self, a: XY, b: XY, samples: int = 8) -> bool:
        """The segment ab lies over the slab (sampled)."""
        for k in range(1, samples):
            p = _add(a, _mul(_sub(b, a), k / samples))
            if not self.inside(p):
                return False
        return True

    def clip(self, pts: Sequence[XY], tol: Optional[float] = None) -> List[List[XY]]:
        """The parts of the polyline over the slab (``tol``: a line that far outside the
        edge counts as on it - a circle grid line along a faceted slab edge)."""
        near = self.tol if tol is None else tol
        out: List[List[XY]] = []
        cur: List[XY] = []
        for a, b in zip(pts[:-1], pts[1:]):
            d = _sub(b, a)
            L = _norm(d)
            if L <= 0:
                continue
            u = _unit(d)
            ts = sorted({0.0, L} | {t for e in self.edges for t in [_ray_seg(a, u, *e)] if t is not None and 0 < t < L})
            for t0, t1 in zip(ts[:-1], ts[1:]):
                mid = _add(a, _mul(u, 0.5 * (t0 + t1)))
                p0, p1 = _add(a, _mul(u, t0)), _add(a, _mul(u, t1))
                if self.inside(mid, near):  # (along the slab edge too: an edge strip)
                    if not cur:
                        cur = [p0]
                    elif _norm(_sub(cur[-1], p0)) > self.tol:
                        out.append(cur)
                        cur = [p0]
                    cur.append(p1)
                elif cur:
                    out.append(cur)
                    cur = []
        if cur:
            out.append(cur)
        return [_simplify(c, self.tol) for c in out if _length(c) > self.tol]


def _poly_area(p) -> float:
    return 0.5 * sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))


def _length(pts) -> float:
    return sum(_norm(_sub(b, a)) for a, b in zip(pts[:-1], pts[1:]))


def _simplify(pts: List[XY], tol: float) -> List[XY]:
    """Drop repeated points and points on a straight run (the middle of three in line)."""
    out: List[XY] = []
    for p in pts:
        if out and _norm(_sub(p, out[-1])) <= tol:
            continue
        out.append(p)
    k = 1
    while k < len(out) - 1:
        a, b, c = out[k - 1], out[k], out[k + 1]
        if _seg_dist(b, a, c) <= tol * 0.1 and _dot(_sub(b, a), _sub(c, b)) > 0:
            out.pop(k)
        else:
            k += 1
    return out


# ------------------------------------------------------------------ the floor of a story
def _length_unit(model) -> float:
    return 1.0 / model.units.factor("length")  # model length units in one metre


def _tol(model) -> float:
    return max(model.options.merge_tolerance, 1e-9)


def _slab_floor(model, a) -> bool:
    if a.type not in ("Floor", "Ramp") or not a.section:
        return False
    sec = model.shell_sections.get(a.section)
    return sec is not None and sec.type == "Slab"


def _at(model, names, z, tol) -> bool:
    return bool(names) and all(n in model.points and abs(model.points[n].z - z) <= tol for n in names)


def floor_of(model, story: str) -> Tuple[Floor, List[str]]:
    """The slab floor objects at the story's level (its tower only) and their outline."""
    tol = _tol(model)
    z = model.stories[story].elevation
    tower = model.tower_of_story(story) if model.multi_tower else None
    names = [a.name for a in model.areas.values() if _slab_floor(model, a) and _at(model, a.points, z, tol)
             and (tower is None or model.tower_of_object("area", a.name) == tower)]
    polys = [[(model.points[p].x, model.points[p].y) for p in model.areas[n].points] for n in names]
    return Floor(polys, tol), names


def _slab_thickness(model, areas) -> float:
    hs = []
    for n in areas:
        sec = model.shell_sections.get(model.areas[n].section)
        if sec is not None:
            hs.append(sec.bending_thickness or sec.thickness)
    return max(hs) if hs else 0.0


def _slab_E(model, areas) -> float:
    for n in areas:
        sec = model.shell_sections.get(model.areas[n].section)
        mat = model.materials.get(sec.material) if sec is not None else None
        if mat is not None and mat.E > 0:
            return mat.E
    return 0.0


# ------------------------------------------------------------------ supports
@dataclass
class _Line:
    a: XY
    b: XY
    kind: str  # wall | beam | spring
    name: str
    alpha: Optional[float] = None  # beams: alpha_f l2 / l1


def supports_of(model, story: str, floor: Optional[Floor] = None, areas: Optional[List[str]] = None,
                beam_alpha: float = 1.0):
    """(point supports {name: xy}, line supports [_Line], beams left out as too flexible
    [(name, alpha l2/l1)]) of the floor at ``story``."""
    if floor is None or areas is None:
        floor, areas = floor_of(model, story)
    tol = _tol(model)
    z = model.stories[story].elevation
    tower = model.tower_of_story(story) if model.multi_tower else None

    def mine(kind, name):
        return tower is None or model.tower_of_object(kind, name) == tower

    adj: Dict[str, List[str]] = {}
    for f in model.frames.values():
        if f.type == "Null":
            continue
        adj.setdefault(f.point_i, []).append(f.name)
        adj.setdefault(f.point_j, []).append(f.name)
    points: Dict[str, XY] = {}
    for p in model.points.values():
        if abs(p.z - z) > tol or not floor.inside((p.x, p.y)):
            continue
        support = bool(p.restraints[2]) or (p.springs[2] > 0)
        if not support:
            for fn in adj.get(p.name, []):
                f = model.frames[fn]
                o = model.points[f.point_j if f.point_i == p.name else f.point_i]
                dz = abs(o.z - p.z)
                if dz > tol and dz >= math.hypot(o.x - p.x, o.y - p.y) and mine("frame", fn):
                    support = True  # a column / pile / steep brace into the floor
                    break
        if support:
            points[p.name] = (p.x, p.y)
    lines: List[_Line] = []
    for a in model.areas.values():  # walls: their edges at the level
        if a.type != "Wall" or not mine("area", a.name):
            continue
        pts = a.points
        for i in range(len(pts)):
            m, n = pts[i], pts[(i + 1) % len(pts)]
            if m in model.points and n in model.points and abs(model.points[m].z - z) <= tol \
                    and abs(model.points[n].z - z) <= tol:
                pa, pb = (model.points[m].x, model.points[m].y), (model.points[n].x, model.points[n].y)
                if _norm(_sub(pb, pa)) > tol and floor.inside(_mul(_add(pa, pb), 0.5)):
                    lines.append(_Line(pa, pb, "wall", a.name))
    for n, ls in getattr(model, "line_springs", {}).items():
        if ls.point_i in model.points and ls.point_j in model.points and ls.stiffness[2] > 0:
            pi, pj = model.points[ls.point_i], model.points[ls.point_j]
            if abs(pi.z - z) <= tol and abs(pj.z - z) <= tol:
                lines.append(_Line((pi.x, pi.y), (pj.x, pj.y), "spring", n))
    flexible = []
    for run in _beam_runs(model, z, tol, floor, mine, set(points)):
        alpha = _beam_alpha(model, run, floor, areas, points, lines)
        if alpha is None or alpha >= beam_alpha:
            lines.append(_Line(run["a"], run["b"], "beam", ",".join(run["frames"]), alpha))
        else:
            flexible.append((",".join(run["frames"]), alpha))
    return points, lines, flexible


def _beam_runs(model, z, tol, floor, mine, columns):
    """The beams at the level joined into straight runs between supports (a beam cut by
    joints is one run until a column)."""
    beams = []
    for f in model.frames.values():
        if f.type == "Null" or f.section not in model.frame_sections or not mine("frame", f.name):
            continue
        pi, pj = model.points[f.point_i], model.points[f.point_j]
        if abs(pi.z - z) > tol or abs(pj.z - z) > tol:
            continue
        a, b = (pi.x, pi.y), (pj.x, pj.y)
        if _norm(_sub(b, a)) <= tol or not floor.inside(_mul(_add(a, b), 0.5), tol=max(tol, 0.5 * _beam_width(model, f))):
            continue
        beams.append(f)
    by_point: Dict[str, List] = {}
    for f in beams:
        by_point.setdefault(f.point_i, []).append(f)
        by_point.setdefault(f.point_j, []).append(f)
    used = set()
    runs = []

    def direction(f):
        pi, pj = model.points[f.point_i], model.points[f.point_j]
        return _unit((pj.x - pi.x, pj.y - pi.y))

    for f in beams:
        if f.name in used:
            continue
        used.add(f.name)
        chain = [f]
        ends = [f.point_i, f.point_j]
        d0 = direction(f)
        for side in (0, 1):  # grow both ways through joints that are not columns
            while True:
                end = ends[side]
                if end in columns:
                    break
                nxt = [g for g in by_point.get(end, []) if g.name not in used and abs(_cross(direction(g), d0)) < 0.02
                       and g.section == f.section]
                if len(nxt) != 1:
                    break
                g = nxt[0]
                used.add(g.name)
                chain.append(g)
                ends[side] = g.point_j if g.point_i == end else g.point_i
        pa, pb = model.points[ends[0]], model.points[ends[1]]
        runs.append({"frames": [g.name for g in chain], "a": (pa.x, pa.y), "b": (pb.x, pb.y), "frame": f})
    return runs


def _beam_width(model, f) -> float:
    sec = model.frame_sections.get(f.section)
    return float(sec.t2 or 0.0) if sec is not None else 0.0


def _beam_alpha(model, run, floor: Floor, areas, points, lines) -> Optional[float]:
    """alpha_f * l2 / l1 of a beam run (None: cannot tell - it is kept as a support)."""
    f = run["frame"]
    sec = model.frame_sections.get(f.section)
    if sec is None:
        return None
    mat = model.materials.get(sec.material)
    Eb = mat.E if mat is not None else 0.0
    Es = _slab_E(model, areas)
    hs = _slab_thickness(model, areas)
    if Eb <= 0 or Es <= 0 or hs <= 0:
        return None
    a, b = run["a"], run["b"]
    l1 = _norm(_sub(b, a))
    if l1 <= 0:
        return None
    u = _unit(_sub(b, a))
    n = _left(u)
    mid = _mul(_add(a, b), 0.5)
    # l2: half the distance to the next parallel support (or the slab edge) on each side
    others = [(q, q) for q in points.values()] + [(ln.a, ln.b) for ln in lines]
    # other beams too (whatever their stiffness): the panel the beam carries ends there
    halves = []
    sides = 0
    for s in (1.0, -1.0):
        d = _mul(n, s)
        e = floor.exit(mid, d)
        best = e
        for q0, q1 in others:
            for q in (q0, q1, _mul(_add(q0, q1), 0.5)):
                v = _sub(q, mid)
                t = _dot(v, d)
                if t > floor.tol and abs(_dot(v, u)) <= 0.5 * l1 + floor.tol:
                    best = min(best, t)
        beams_other = [ln for ln in _all_beam_lines(model, run, floor)]
        for q0, q1 in beams_other:
            t = _ray_seg(mid, d, q0, q1)
            if t is not None and t > floor.tol:
                best = min(best, t)
        if best > floor.tol:
            halves.append(0.5 * best if best < e else best)  # (an edge: the overhang itself)
            sides += 1
    l2 = sum(halves)
    if l2 <= 0:
        return None
    Ib = _beam_inertia(sec, hs, two_sided=(sides == 2))
    if Ib <= 0:
        return None
    Is = l2 * hs ** 3 / 12.0
    alpha = (Eb * Ib) / (Es * Is)
    return alpha * l2 / l1


def _all_beam_lines(model, run, floor):
    """Plan segments of the other beams at the run's level (cached on the run)."""
    cache = run.get("_others")
    if cache is None:
        z = model.points[run["frame"].point_i].z
        tol = floor.tol
        cache = []
        mine = set(run["frames"])
        for f in model.frames.values():
            if f.name in mine or f.type == "Null":
                continue
            pi, pj = model.points[f.point_i], model.points[f.point_j]
            if abs(pi.z - z) <= tol and abs(pj.z - z) <= tol:
                cache.append(((pi.x, pi.y), (pj.x, pj.y)))
        run["_others"] = cache
    return cache


def _beam_inertia(sec, hs: float, two_sided: bool = True) -> float:
    """Ib of ACI 318 8.4.1.8: a rectangular concrete beam with the slab on each side,
    as far as the beam projects below (or above) the slab, but not more than 4 hs;
    other sections: their own I33."""
    shape = (sec.shape or "").lower()
    b, h = sec.t2, sec.t3
    if shape in ("rectangular", "concrete rectangular") and b and h:
        hw = max(h - hs, 0.0)
        over = min(hw, 4.0 * hs) * (2.0 if two_sided else 1.0)
        if hw <= 0:
            return b * h ** 3 / 12.0
        # web b x h and flange (over) x hs at the top
        A1, y1, I1 = b * h, h / 2.0, b * h ** 3 / 12.0
        A2, y2, I2 = over * hs, h - hs / 2.0, over * hs ** 3 / 12.0
        A = A1 + A2
        yc = (A1 * y1 + A2 * y2) / A
        return I1 + A1 * (y1 - yc) ** 2 + I2 + A2 * (y2 - yc) ** 2
    return float(sec.I33 or 0.0)


# ------------------------------------------------------------------ chains (column strips)
@dataclass
class Chain:
    """A column strip being laid out: its centre line, the supports on it (distance
    along it) and where it came from."""

    pts: List[XY]
    family: str  # A | B
    source: str  # grid name | "supports"
    stations: List[float] = field(default_factory=list)  # supports, as distance along pts
    widths: List[Tuple[float, float, float, float]] = field(default_factory=list)
    l1: List[Optional[float]] = field(default_factory=list)


def _cum(pts) -> List[float]:
    out = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        out.append(out[-1] + _norm(_sub(b, a)))
    return out


def _project(pts, cum, p) -> Tuple[float, float]:
    """(distance along the polyline of the nearest point to p, distance from it)."""
    best = (0.0, BIG)
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        ab = _sub(b, a)
        L = _norm(ab)
        if L <= 0:
            continue
        t = max(0.0, min(L, _dot(_sub(p, a), ab) / L))
        d = _norm(_sub(p, _add(a, _mul(ab, t / L))))
        if d < best[1]:
            best = (cum[k] + t, d)
    return best


def _at_dist(pts, cum, s) -> Tuple[XY, XY, int]:
    """(point, unit direction, segment) at distance s along the polyline."""
    for k in range(len(pts) - 1):
        if s <= cum[k + 1] or k == len(pts) - 2:
            L = cum[k + 1] - cum[k]
            u = _unit(_sub(pts[k + 1], pts[k]))
            t = 0.0 if L <= 0 else (s - cum[k]) / L
            return _add(pts[k], _mul(_sub(pts[k + 1], pts[k]), max(0.0, min(1.0, t)))), u, k
    return pts[0], (1.0, 0.0), 0


def _split_at(pts: List[XY], cuts: Sequence[float], tol: float, gap: Optional[float] = None) -> List[XY]:
    """The polyline with vertices added at the distances ``cuts`` - not closer than
    ``gap`` to a vertex (a support just inside the slab edge: no stub of a segment)."""
    cum = _cum(pts)
    g = max(tol, gap or 0.0)
    extra = sorted(c for c in cuts if g < c < cum[-1] - g and all(abs(c - q) > g for q in cum))
    out = list(pts)
    for c in reversed(extra):
        p, _, k = _at_dist(pts, cum, c)
        out.insert(k + 1, p)
    # (inserted in reverse order of distance: positions stay right because each insert
    # is after its segment's start and the later cuts were inserted first)
    return _simplify_keep(out, tol)


def _simplify_keep(pts, tol):
    out = []
    for p in pts:
        if not out or _norm(_sub(p, out[-1])) > tol:
            out.append(p)
    return out


def _family_dirs(model, story, points, lines) -> Tuple[float, str]:
    """Angle (deg) of the layer-A direction and where it came from."""
    view = model.view(model.tower_of_story(story)) if model.multi_tower else model
    for name in sorted(view.grid_systems):
        gs = view.grid_systems[name]
        if gs.type != "Cylindrical" and any(g.system == name and g.direction in ("X", "Y") for g in view.grid_lines):
            return float(gs.rotation or 0.0) % 180.0, f"grid system {name}"
    hist: Dict[int, float] = {}
    for ln in lines:
        d = _sub(ln.b, ln.a)
        ang = math.degrees(math.atan2(d[1], d[0])) % 90.0
        hist[int(round(ang)) % 90] = hist.get(int(round(ang)) % 90, 0.0) + _norm(d)
    if not hist:
        pts = list(points.values())
        for i, p in enumerate(pts):
            near = sorted((q for j, q in enumerate(pts) if j != i), key=lambda q: _norm(_sub(q, p)))[:2]
            for q in near:
                d = _sub(q, p)
                if _norm(d) > 0:
                    ang = math.degrees(math.atan2(d[1], d[0])) % 90.0
                    hist[int(round(ang)) % 90] = hist.get(int(round(ang)) % 90, 0.0) + 1.0
    if not hist:
        return 0.0, "global X"
    best = max(hist, key=lambda k: (hist[k] + 0.5 * (hist.get((k - 1) % 90, 0) + hist.get((k + 1) % 90, 0)), -k))
    return float(best), "the walls / beams" if lines else "the column layout"


def _grid_polylines(model, story) -> List[Tuple[List[XY], str, str, bool]]:
    """(polyline, family, name, keep its direction) of every plan grid line of the story's
    tower shown (radial lines run outwards, circles counter-clockwise: the strips of a
    cylindrical grid all have their left side the same way round)."""
    from ..model.grids import grid_geometry
    view = model.view(model.tower_of_story(story)) if model.multi_tower else model
    gg = grid_geometry(view)
    dirs = {((g.system or "G1"), g.name): g.direction for g in view.grid_lines}
    out = []
    first_rot = None
    for name in sorted(view.grid_systems):
        gs = view.grid_systems[name]
        if gs.type != "Cylindrical":
            first_rot = float(gs.rotation or 0.0)
            break
    for a, b, name, system in gg.lines:
        d = dirs.get((system, name), "General")
        gs = view.grid_systems.get(system)
        if d == "X":
            fam = "B"  # an X line is at constant x: it runs along y
        elif d == "Y":
            fam = "A"
        elif d == "T":
            fam = "B"  # radial
        else:
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            rel = ((ang - (first_rot or 0.0)) + 90.0) % 180.0 - 90.0
            fam = "A" if abs(rel) <= 45.0 else "B"
        out.append(([a, b], fam, name, d in ("T",)))
    for c, r, a0, a1, name, system in gg.arcs:
        n = max(2, int(math.ceil(abs(a1 - a0) / 5.0)))
        pts = [(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
                c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]
        out.append((pts, "A", name, True))  # circumferential (counter-clockwise)
    return out


def _extend_line(pts: List[XY], floor: Floor) -> List[XY]:
    """A grid line run on past its ends as far as the slab reaches (grid lines stop a
    little past the outermost grid lines; the slab may go further, e.g. a cantilever)."""
    if len(pts) < 2:
        return pts
    x0, x1, y0, y1 = floor.box
    far = 2.0 * math.hypot(x1 - x0, y1 - y0) + 1.0
    d0 = _unit(_sub(pts[0], pts[1]))
    d1 = _unit(_sub(pts[-1], pts[-2]))
    return [_add(pts[0], _mul(d0, far))] + list(pts) + [_add(pts[-1], _mul(d1, far))]


def _orient(pts: List[XY], d: XY) -> List[XY]:
    """The polyline running along ``d`` (reversed when it runs the other way)."""
    v = _sub(pts[-1], pts[0])
    if _norm(v) <= 0:
        return pts
    if _dot(v, d) < -1e-9 or (abs(_dot(v, d)) <= 1e-9 and _cross(d, v) < 0):
        return list(reversed(pts))
    return pts


def _on_polyline(pts, cum, p, tol) -> Optional[float]:
    s, d = _project(pts, cum, p)
    return s if d <= tol else None


def story_signature(model, story: str):
    """What the strip layout of a story depends on, in plan: the objects at its level
    (each with what is above / below it, not how far) - two stories with the same
    signature get the same strips (a typical floor laid out once)."""
    if story not in model.stories:
        return None
    tol = _tol(model)
    z = model.stories[story].elevation
    g = max(tol, 1e-6) * 10

    def r(v):
        return round(v / g)

    def side(dz):
        return 0 if abs(dz) <= tol else (1 if dz > 0 else -1)
    pts = model.points
    at = {n for n, p in pts.items() if abs(p.z - z) <= tol}
    sig = []
    for n in at:
        p = pts[n]
        if any(p.restraints) or any(p.springs):
            sig.append(("J", r(p.x), r(p.y), tuple(p.restraints), tuple(p.springs)))
    for f in model.frames.values():
        if f.point_i in at or f.point_j in at:
            a, b = pts[f.point_i], pts[f.point_j]
            ends = sorted(((r(a.x), r(a.y), side(a.z - z)), (r(b.x), r(b.y), side(b.z - z))))
            sig.append(("F", f.type, f.section, tuple(ends)))
    for ar in model.areas.values():
        if any(q in at for q in ar.points):
            sig.append(("A", ar.type, ar.section,
                        tuple(sorted((r(pts[q].x), r(pts[q].y), side(pts[q].z - z)) for q in ar.points))))
    for ls in getattr(model, "line_springs", {}).values():
        sig.append(("S", repr(ls)))  # (rare: any line spring - no reuse across a change)
    tower = model.tower_of_story(story) if model.multi_tower else None
    return (tower, tuple(sorted(sig, key=repr)))


def layout(model, story: str, options: Optional[StripOptions] = None) -> StripLayout:
    """The design strips of the floor at ``story`` (not added to the model)."""
    opt = options or StripOptions()
    res = StripLayout(story=story)
    if story not in model.stories:
        res.warnings.append(f"Story {story!r} does not exist")
        return res
    floor, areas = floor_of(model, story)
    if not floor.polys:
        res.warnings.append(f"{story}: no floor slab at this level")
        return res
    tol = _tol(model)
    unit = _length_unit(model)
    on_tol = max(opt.on_line_tol * unit, tol)
    floor.near = on_tol
    z = model.stories[story].elevation
    points, lines, flexible = supports_of(model, story, floor, areas, opt.beam_alpha)
    res.supports = {"columns / supports": len(points),
                    "walls": sum(1 for ln in lines if ln.kind == "wall"),
                    "beams": sum(1 for ln in lines if ln.kind == "beam"),
                    "line springs": sum(1 for ln in lines if ln.kind == "spring")}
    if flexible:
        res.notes.append(f"{len(flexible)} beam run(s) too flexible to be a line of supports "
                         f"(alpha_f l2/l1 < {opt.beam_alpha:g}): the slab spans past them")
    theta, why = _family_dirs(model, story, points, lines)
    fam_dir = {"A": (math.cos(math.radians(theta)), math.sin(math.radians(theta))),
               "B": (-math.sin(math.radians(theta)), math.cos(math.radians(theta)))}
    res.notes.append(f"layer A along {theta:g} deg (from {why}), layer B across it")
    if not points and not lines and opt.source != "grids":
        res.warnings.append(f"{story}: no column, wall, pile or stiff beam supports the floor - "
                            "slab on ground: draw the strips (or lay them out on the grid lines)")
        return res
    cos_max = math.cos(math.radians(opt.align_angle))
    chains: Dict[str, List[Chain]] = {"A": [], "B": []}
    used: Dict[str, set] = {"A": set(), "B": set()}  # support keys on a chain of the family
    line_used: Dict[str, set] = {"A": set(), "B": set()}

    def line_family(ln) -> str:
        d = _unit(_sub(ln.b, ln.a))
        return max(("A", "B"), key=lambda f: abs(_dot(d, fam_dir[f])))

    # --- 1. grid lines (with supports on them, or all of them for source "grids")
    if opt.source in ("auto", "grids"):
        for pts0, fam, name, keep in _grid_polylines(model, story):
            for piece in floor.clip(_extend_line(pts0, floor), tol=on_tol):
                piece = floor.snap_ends(piece)
                if len(piece) < 2:
                    continue
                if not keep:
                    piece = _orient(piece, fam_dir[fam])
                cum = _cum(piece)
                on = []
                for key, p in points.items():
                    s = _on_polyline(piece, cum, p, on_tol)
                    if s is not None:
                        on.append((s, ("p", key)))
                for i, ln in enumerate(lines):
                    sa, sb = _on_polyline(piece, cum, ln.a, on_tol), _on_polyline(piece, cum, ln.b, on_tol)
                    if sa is not None and sb is not None and abs(sa - sb) > tol:
                        on += [(sa, ("l", i)), (sb, ("l", i))]
                if opt.source == "auto" and len({k for _, k in on}) < (1 if any(k[0] == "l" for _, k in on) else 2):
                    continue
                ch = Chain(piece, fam, name, stations=sorted({s for s, _ in on}))
                chains[fam].append(ch)
                for _, k in on:
                    (used if k[0] == "p" else line_used)[fam].add(k[1])
    # --- 2. the supports not on a chain yet: from each to the next one ahead
    if opt.source in ("auto", "supports"):
        for fam in ("A", "B"):
            d = fam_dir[fam]
            nodes: Dict[str, XY] = {k: p for k, p in points.items() if k not in used[fam]}
            forced = []
            for i, ln in enumerate(lines):
                if i in line_used[fam] or line_family(ln) != fam:
                    continue
                a, b = (ln.a, ln.b) if _dot(_sub(ln.b, ln.a), d) >= 0 else (ln.b, ln.a)
                ka, kb = _node_key(nodes, a, tol), _node_key(nodes, b, tol)
                forced.append((ka, kb, i))
            # candidate links: ahead of the support, within the turn angle, over the slab
            keys = list(nodes)
            cand = []
            for ka in keys:
                pa = nodes[ka]
                for kb in keys:
                    if ka == kb:
                        continue
                    v = _sub(nodes[kb], pa)
                    L = _norm(v)
                    if L <= tol:
                        continue
                    c = _dot(v, d) / L
                    if c < cos_max:
                        continue
                    if not floor.covers(pa, nodes[kb]):
                        continue
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
                    cand.append((L * (1.0 + ang / 45.0), ka, kb))
            succ: Dict[str, str] = {}
            pred: Dict[str, str] = {}
            for ka, kb, _ in forced:
                if ka not in succ and kb not in pred:
                    succ[ka], pred[kb] = kb, ka
            for _, ka, kb in sorted(cand):
                if ka in succ or kb in pred:
                    continue
                # no link that jumps over a nearer support in its way
                succ[ka], pred[kb] = kb, ka
            starts = [k for k in nodes if k not in pred]
            seen = set()
            for k0 in starts:
                seq = [k0]
                seen.add(k0)
                while seq[-1] in succ and succ[seq[-1]] not in seen:
                    seq.append(succ[seq[-1]])
                    seen.add(seq[-1])
                if len(seq) < 2 and (k0.startswith("@") or
                                     not _alone_needs_strip(nodes[k0], chains[fam], d, floor, on_tol)):
                    continue
                pts = [nodes[k] for k in seq]
                # past the end supports: on to the slab edge
                d0 = _unit(_sub(pts[0], pts[1])) if len(pts) > 1 else _mul(d, -1.0)
                d1 = _unit(_sub(pts[-1], pts[-2])) if len(pts) > 1 else d
                e0, e1 = floor.exit(pts[0], d0), floor.exit(pts[-1], d1)
                stub = 2.0 * on_tol  # (a support just inside the edge: its strip starts at the edge)
                if e0 > tol:
                    if e0 <= stub and len(pts) > 1:
                        pts = [_add(pts[0], _mul(d0, e0))] + pts[1:]
                    else:
                        pts = [_add(pts[0], _mul(d0, e0))] + pts
                if e1 > tol:
                    if e1 <= stub and len(pts) > 1:
                        pts = pts[:-1] + [_add(pts[-1], _mul(d1, e1))]
                    else:
                        pts = pts + [_add(pts[-1], _mul(d1, e1))]
                cum = _cum(pts)
                stations = [_project(pts, cum, nodes[k])[0] for k in seq]
                if len(pts) < 2:
                    continue
                chains[fam].append(Chain(pts, fam, "supports", stations=stations))
            for k in seen:
                used[fam].add(k)
    for fam in ("A", "B"):
        if fam not in opt.layers:
            chains[fam] = []
        for ch in chains[fam]:
            ch.stations = sorted(set(ch.stations) | set(_crossings(ch.pts, lines, tol)))
            ch.pts = _split_at(ch.pts, ch.stations, tol, gap=2.0 * on_tol)
        _drop_duplicates(chains[fam], tol, on_tol)
        column_widths(chains[fam], floor, tol, reach=2.0 * on_tol)
    # --- strips
    for fam in ("A", "B"):
        for ch in chains[fam]:
            res.strips.append(dict(points=[(p[0], p[1], z) for p in ch.pts], widths=list(ch.widths), layer=fam,
                                   kind="Column", auto=True))
        if opt.middle:
            for pts, ws in middle_strips(chains[fam], floor, tol, reach=2.0 * on_tol):
                res.strips.append(dict(points=[(p[0], p[1], z) for p in pts], widths=ws, layer=fam, kind="Middle",
                                       auto=True))
        if fam in opt.layers and not chains[fam]:
            res.warnings.append(f"{story}: no line of supports in layer {fam} - draw its strips")
    lone = [k for k in points if k not in used["A"] and k not in used["B"]]
    if lone:
        res.warnings.append(f"{story}: {len(lone)} support(s) not on a column strip (first: joint {lone[0]})")
    return res


def _crossings(pts, lines, tol) -> List[float]:
    """Where the polyline crosses walls / beams / line springs (supports of the strip
    there), as distances along it."""
    cum = _cum(pts)
    out = []
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        L = cum[k + 1] - cum[k]
        if L <= 0:
            continue
        u = _unit(_sub(b, a))
        for ln in lines:
            e = _sub(ln.b, ln.a)
            if abs(_cross(u, _unit(e))) < 0.05:
                continue  # along the strip, not across it
            t = _ray_seg(a, u, ln.a, ln.b)
            if t is not None and -tol <= t <= L + tol:
                out.append(cum[k] + max(0.0, min(L, t)))
    return out


def _node_key(nodes: Dict[str, XY], p: XY, tol: float) -> str:
    for k, q in nodes.items():
        if _norm(_sub(p, q)) <= tol:
            return k
    key = f"@{p[0]:.6g},{p[1]:.6g}"
    nodes[key] = p
    return key


def _alone_needs_strip(p, chains, d, floor, tol) -> bool:
    """A support with no other ahead or behind: it still gets a strip through it, unless
    a strip of the family already passes over it."""
    for ch in chains:
        if _project(ch.pts, _cum(ch.pts), p)[1] <= tol:
            return False
    return True


def _drop_duplicates(chains: List[Chain], tol: float, on_tol: float) -> None:
    """Two strips of a family along the same line: the shorter goes."""
    chains.sort(key=lambda c: -_length(c.pts))
    keep: List[Chain] = []
    for ch in chains:
        dup = False
        for k in keep:
            cum = _cum(k.pts)
            if all(_project(k.pts, cum, p)[1] <= on_tol for p in ch.pts):
                dup = True
                break
        if not dup:
            keep.append(ch)
    chains[:] = keep


# ------------------------------------------------------------------ widths
def _spans(ch: Chain, tol: float) -> List[Optional[float]]:
    """l1 of each segment: the span between the supports around it (end segments past
    the last support: the span next to them)."""
    cum = _cum(ch.pts)
    st = sorted(ch.stations)
    out: List[Optional[float]] = []
    for k in range(len(ch.pts) - 1):
        mid = 0.5 * (cum[k] + cum[k + 1])
        prev = [s for s in st if s <= mid + tol]
        nxt = [s for s in st if s >= mid - tol]
        if prev and nxt and nxt[0] - prev[-1] > tol:
            out.append(nxt[0] - prev[-1])
        else:
            out.append(None)
    # end spans / cantilevers: the span beside them
    for k in range(len(out)):
        if out[k] is None:
            near = [out[j] for j in sorted(range(len(out)), key=lambda j: abs(j - k)) if out[j] is not None]
            out[k] = near[0] if near else None
    return out


def _side_distance(o: XY, n: XY, others: List[Chain], floor: Floor, tol: float, reach: float = 0.0):
    """(distance, (the chain hit, its segment, distance) or None for the slab edge) across
    from o along n.  ``reach``: the other strips count as running that much past their
    ends (a ray from near the end of a strip passing just beside the end of the next)."""
    e = floor.exit(o, n)
    best, hit = e, None
    far = e + max(tol, floor.near)
    # (only the strips the ray can reach: a box test first)
    px, py = o[0] + n[0] * far, o[1] + n[1] * far
    pad = reach + tol
    x0, x1 = min(o[0], px) - pad, max(o[0], px) + pad
    y0, y1 = min(o[1], py) - pad, max(o[1], py) + pad
    for ch in others:
        pts = ch.pts
        if max(q[0] for q in pts) < x0 or min(q[0] for q in pts) > x1 or \
                max(q[1] for q in pts) < y0 or min(q[1] for q in pts) > y1:
            continue
        last = len(pts) - 2
        for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            if (a[0] < x0 and b[0] < x0) or (a[0] > x1 and b[0] > x1) or \
                    (a[1] < y0 and b[1] < y0) or (a[1] > y1 and b[1] > y1):
                continue
            if reach > 0 and (k == 0 or k == last):
                u = _unit(_sub(b, a))
                if k == 0:
                    a = _sub(a, _mul(u, reach))
                if k == last:
                    b = _add(b, _mul(u, reach))
            t = _ray_seg(o, n, a, b)
            if t is not None and tol < t <= e + max(tol, floor.near) and (hit is None or t < best):
                best, hit = t, (ch, k, t)  # (a strip along the slab edge counts - also just past a faceted one)
    return best, hit


SAMPLES = (0.1, 0.5, 0.9)


def _sides(a, b, n, others, floor, tol, reach):
    """The side distances (left, right) at the start and the end of segment ab: measured
    at 10 % and 90 % along it, each taken at mid-segment instead when it does not meet
    what the middle meets (a ray from a strip end slanting out through a slab corner)."""
    got = {}
    for frac in SAMPLES:
        o = _add(a, _mul(_sub(b, a), frac))
        got[frac] = [_side_distance(o, _mul(n, s), others, floor, tol, reach) for s in (1.0, -1.0)]
    out = []
    for frac in (SAMPLES[0], SAMPLES[2]):
        row = []
        for side in (0, 1):
            v, m = got[frac][side], got[SAMPLES[1]][side]
            same = (v[1] is None) == (m[1] is None) and (v[1] is None or v[1][0] is m[1][0])
            row.append((v if same else m, frac if same else SAMPLES[1]))
        out.append(row)
    return out  # [[(left, frac), (right, frac)] at start, [...] at end]


def _ends(v0: float, f0: float, v1: float, f1: float) -> Tuple[float, float]:
    """Values at the start and end of a segment from values at fractions f0 < f1 of it
    (straight-line extension; equal fractions: the same value at both ends)."""
    if f1 - f0 <= 1e-9:
        return v0, v1
    slope = (v1 - v0) / (f1 - f0)
    return v0 - slope * f0, v1 + slope * (1.0 - f1)


def column_widths(chains: List[Chain], floor: Floor, tol: float, reach: float = 0.0) -> None:
    """ACI 318 8.4.1.5: each side 0.25 min(l1, l2); a side with no next line of supports
    reaches the slab edge (but not more than 0.25 l1)."""
    for ch in chains:
        ch.l1 = _spans(ch, tol)
        others = [c for c in chains if c is not ch]
        ws = []
        for k, (a, b) in enumerate(zip(ch.pts[:-1], ch.pts[1:])):
            u = _unit(_sub(b, a))
            n = _left(u)
            l1 = ch.l1[k]
            vals = []  # [start, end] x [left, right] -> (width, frac)
            for pair in _sides(a, b, n, others, floor, tol, reach):
                sides = [v for v, _ in pair]
                l2s = [dist for dist, hit in sides if hit is not None]
                span = l1 if l1 is not None else (min(l2s) if l2s else None)
                row = []
                for (dist, hit), (_, frac) in zip(sides, pair):
                    if hit is None:  # slab edge: the strip reaches it, up to 0.25 l1
                        w = dist if span is None else min(dist, 0.25 * span)
                    else:
                        w = 0.25 * (dist if span is None else min(span, dist))
                    row.append((max(0.0, w), frac))
                vals.append(row)
            out = []
            for side in (0, 1):
                (v0, f0), (v1, f1) = vals[0][side], vals[1][side]
                w0, w1 = _ends(v0, f0, v1, f1)
                out.append((max(0.0, w0), max(0.0, w1)))
            # (start left, start right, end left, end right)
            ws.append((out[0][0], out[1][0], out[0][1], out[1][1]))
        ch.widths = ws


def _facing_width(ch: Chain, k: int, t_frac: float, toward: XY) -> float:
    """Width of chain ``ch`` (segment k, at t_frac along it) on the side facing ``toward``."""
    a, b = ch.pts[k], ch.pts[k + 1]
    n = _left(_unit(_sub(b, a)))
    w = ch.widths[k] if k < len(ch.widths) else (0, 0, 0, 0)
    t_frac = max(0.0, min(1.0, t_frac))
    left = w[0] + (w[2] - w[0]) * t_frac
    right = w[1] + (w[3] - w[1]) * t_frac
    return left if _dot(n, toward) > 0 else right


def _gap(o: XY, n: XY, own: float, found) -> Tuple[float, float, object]:
    """(inner offset, outer offset, what is across) of the slab between a column strip
    (half width ``own`` toward n) and what the ray along n meets."""
    dist, hit = found
    if hit is None:
        return own, dist, None
    oc, ok, ot = hit
    seg = _sub(oc.pts[ok + 1], oc.pts[ok])
    q = _add(o, _mul(n, ot))
    tf = _dot(_sub(q, oc.pts[ok]), seg) / max(_dot(seg, seg), 1e-30)
    # the other strip's width is square to it: along this ray (at an angle to it) longer
    c = abs(_dot(n, _left(_unit(seg)))) or 1.0
    return own, dist - _facing_width(oc, ok, tf, _mul(n, -1.0)) / max(c, 0.2), id(oc)


def middle_strips(chains: List[Chain], floor: Floor, tol: float, reach: float = 0.0):
    """[(centre line, widths per segment)] of the middle strips of one layer: between
    each column strip and the next one on its left, and to the slab edge on a side with
    no next one."""
    out = []
    for ch in chains:
        others = [c for c in chains if c is not ch]
        for side in (1.0, -1.0):
            pieces: List = []  # (across, a, b, inner0, outer0, inner1, outer1) | None (a break)
            for k, (a, b) in enumerate(zip(ch.pts[:-1], ch.pts[1:])):
                u = _unit(_sub(b, a))
                n = _mul(_left(u), side)
                L = _norm(_sub(b, a))
                sides = _sides(a, b, _left(u), others, floor, tol, reach)
                pick = 0 if side > 0 else 1
                ends = []
                for row in sides:
                    found, frac = row[pick]
                    o = _add(a, _mul(_sub(b, a), frac))
                    inner, outer, who = _gap(o, n, _facing_width(ch, k, frac, n), found)
                    ends.append((inner, outer, who, frac))
                (i0, o0, w0, f0), (i1, o1, w1, f1) = ends
                if side < 0 and (w0 is not None or w1 is not None):
                    pieces.append(None)  # (the gap to the next strip on the right: made by that strip)
                    continue
                i0, i1 = _ends(i0, f0, i1, f1)
                o0, o1 = _ends(o0, f0, o1, f1)
                small = max(tol, 0.02 * L)
                if o0 - i0 <= small and o1 - i1 <= small:
                    pieces.append(None)
                    continue
                pieces.append((w0 if w0 == w1 else ("mixed", w0, w1), a, b, i0, o0, i1, o1, n))
            run: List = []

            def flush():
                if not run:
                    return
                pts, ws = [], []
                for who, a, b, i0, o0, i1, o1, n in run:
                    c0 = _add(a, _mul(n, 0.5 * (i0 + o0)))
                    c1 = _add(b, _mul(n, 0.5 * (i1 + o1)))
                    # half widths measured along n; the strip's own widths are square to its
                    # centre line (which slants when the gap narrows / widens)
                    c = abs(_dot(n, _left(_unit(_sub(c1, c0))))) if _norm(_sub(c1, c0)) > tol else 1.0
                    h0, h1 = max(0.0, 0.5 * (o0 - i0)) * c, max(0.0, 0.5 * (o1 - i1)) * c
                    if not pts:
                        pts.append(c0)
                    elif _norm(_sub(pts[-1], c0)) > tol:
                        # (the next span's offset differs: the two pieces meet halfway)
                        pts[-1] = _mul(_add(pts[-1], c0), 0.5)
                    pts.append(c1)
                    ws.append((h0, h0, h1, h1))
                run.clear()
                pts, ws = _trim_ends(pts, ws, floor, tol, max(reach, 0.0) + 2.0 * tol)
                if len(pts) >= 2:
                    out.append((pts, ws))
            prev = None  # (what the run met at the end of its last segment)
            for pc in pieces:
                if pc is None:
                    flush()
                    prev = None
                    continue
                w = pc[0]
                w0, w1 = (w[1], w[2]) if isinstance(w, tuple) else (w, w)
                if run and not (w0 == prev or w1 == prev):
                    flush()  # (across is another strip / the edge: a new middle strip)
                run.append(pc)
                prev = w1
            flush()
    return out


def _trim_ends(pts: List[XY], ws: List, floor: Floor, tol: float, reach: float = 0.0):
    """A middle strip whose centre line starts / ends a little off the slab (offset from
    a slanting column strip): its end moved back onto the slab edge; zero-length
    segments dropped."""
    clean_p, clean_w = [pts[0]], []
    for p, w in zip(pts[1:], ws):
        if _norm(_sub(p, clean_p[-1])) > tol:
            clean_p.append(p)
            clean_w.append(w)
    pts, ws = clean_p, clean_w
    if len(pts) < 2:
        return pts, ws
    for end in (0, -1):
        p, q = pts[end], pts[1 if end == 0 else -2]
        d = _unit(_sub(p, q))
        if not floor.inside(p):
            e = floor.exit(q, d)
            if 0 < e < _norm(_sub(p, q)):
                pts[end] = _add(q, _mul(d, e))
        else:  # just short of the edge: run on to it (no sliver of slab left out)
            e = floor.exit(p, d)
            if tol < e <= max(reach, tol):
                pts[end] = _add(p, _mul(d, e))
    return pts, ws


# ------------------------------------------------------------------ existing strips
def strip_quads(ds) -> List[List[XY]]:
    """The plan outline of each segment of a strip (quadrilaterals)."""
    out = []
    pts = [(p[0], p[1]) for p in ds.points]
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        n = _left(_unit(_sub(b, a)))
        wl0, wr0, wl1, wr1 = ds.segment_widths(k)
        out.append([_add(a, _mul(n, wl0)), _add(b, _mul(n, wl1)), _add(b, _mul(n, -wr1)), _add(a, _mul(n, -wr0))])
    return out


def strip_outline(ds) -> List[XY]:
    """The plan outline of a whole strip: its left edge out, its right edge back."""
    pts = [(p[0], p[1]) for p in ds.points]
    left, right = [], []
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        n = _left(_unit(_sub(b, a)))
        wl0, wr0, wl1, wr1 = ds.segment_widths(k)
        left += [_add(a, _mul(n, wl0)), _add(b, _mul(n, wl1))]
        right += [_add(a, _mul(n, -wr0)), _add(b, _mul(n, -wr1))]
    return left + list(reversed(right))


def chains_from_strips(model, story: str, strips, floor: Floor, points, lines, tol, on_tol) -> List[Chain]:
    """Column strips of the model as chains (their supports found again) - Auto Strip
    Widths and strips drawn with automatic widths."""
    out = []
    for ds in strips:
        pts = [(p[0], p[1]) for p in ds.points]
        cum = _cum(pts)
        st = set()
        for p in points.values():
            s = _on_polyline(pts, cum, p, on_tol)
            if s is not None:
                st.add(s)
        for ln in lines:
            for q in (ln.a, ln.b):
                s = _on_polyline(pts, cum, q, on_tol)
                if s is not None:
                    st.add(s)
        ch = Chain(pts, ds.layer, ds.name, stations=sorted(st))
        ch.widths = list(ds.widths)
        out.append(ch)
    return out


def auto_widths(model, story: str, names: Optional[Sequence[str]] = None,
                options: Optional[StripOptions] = None) -> Tuple[Dict[str, list], List[Tuple[str, List, List]], List[str]]:
    """Widths worked out again for the column strips ``names`` of the story (default:
    those whose ``auto_width`` is on) from the supports and the other column strips of
    their layer: ({strip: widths}, [(layer, middle centre line, widths)], notes)."""
    opt = options or StripOptions()
    floor, areas = floor_of(model, story)
    tol = _tol(model)
    on_tol = max(opt.on_line_tol * _length_unit(model), tol)
    floor.near = on_tol
    notes = []
    if not floor.polys:
        return {}, [], [f"{story}: no floor slab at this level"]
    points, lines, _ = supports_of(model, story, floor, areas, opt.beam_alpha)
    strips = [ds for ds in model.design_strips.values() if ds.story == story]
    out: Dict[str, list] = {}
    middles = []
    for fam in ("A", "B", "Other"):
        cols = [ds for ds in strips if ds.layer == fam and ds.kind == "Column"]
        if not cols:
            continue
        chains = chains_from_strips(model, story, cols, floor, points, lines, tol, on_tol)
        target = set(names) if names is not None else {ds.name for ds in cols if ds.auto_width}
        saved = {ch.source: list(ch.widths) for ch in chains}
        column_widths(chains, floor, tol, reach=2.0 * on_tol)
        for ch in chains:
            if ch.source in target:
                out[ch.source] = list(ch.widths)
            else:
                ch.widths = saved[ch.source]  # (kept: the middle strips meet them as they are)
        for pts, ws in middle_strips(chains, floor, tol, reach=2.0 * on_tol):
            middles.append((fam, pts, ws))
    return out, middles, notes


def middle_widths(model, story: str, pts3: Sequence, layer: str) -> List[Tuple[float, float, float, float]]:
    """Automatic widths of a middle strip drawn along ``pts3``: to the column strips of
    its layer (or the slab edge) on each side."""
    floor, _ = floor_of(model, story)
    tol = _tol(model)
    cols = [ds for ds in model.design_strips.values() if ds.story == story and ds.layer == layer
            and ds.kind == "Column"]
    chains = []
    for ds in cols:
        ch = Chain([(p[0], p[1]) for p in ds.points], layer, ds.name)
        ch.widths = list(ds.widths)
        chains.append(ch)
    pts = [(p[0], p[1]) for p in pts3]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = _left(_unit(_sub(b, a)))
        row = []
        for frac in (0.02, 0.98):
            o = _add(a, _mul(_sub(b, a), frac))
            for s in (1.0, -1.0):
                d = _mul(n, s)
                dist, hit = _side_distance(o, d, chains, floor, tol) if floor.polys else (0.0, None)
                if hit is not None:
                    oc, ok, ot = hit
                    seg = _sub(oc.pts[ok + 1], oc.pts[ok])
                    tf = _dot(_sub(_add(o, _mul(d, ot)), oc.pts[ok]), seg) / max(_dot(seg, seg), 1e-30)
                    dist -= _facing_width(oc, ok, max(0.0, min(1.0, tf)), _mul(d, -1.0))
                row.append(max(0.0, dist))
        out.append((row[0], row[1], row[2], row[3]))
    return out


def _in_quad(p, q, gap: float) -> int:
    """1: p inside the quadrilateral by more than ``gap``; 0: within ``gap`` of its
    outline (a hairline between two strips is not a gap, a hairline overlap is not one
    either); -1: outside."""
    near = min(_seg_dist(p, q[k], q[(k + 1) % 4]) for k in range(4))
    if near <= gap:
        return 0
    return 1 if point_in_polygon(p, q) else -1


def coverage(model, story: str, layer: str, spacing: Optional[float] = None) -> dict:
    """How the strips of ``layer`` cover the floor: the share of the slab under no strip
    and under two or more (sampled on a square grid; slivers narrower than 2 cm - strips
    meeting along a curve - count as neither)."""
    floor, _ = floor_of(model, story)
    if not floor.polys:
        return {"samples": 0, "uncovered": 0.0, "overlap": 0.0, "where_uncovered": None, "where_overlap": None}
    quads = [(ds.name, q) for ds in model.design_strips.values() if ds.story == story and ds.layer == layer
             for q in strip_quads(ds)]
    x0, x1, y0, y1 = floor.box
    h = spacing or max(math.sqrt(max(floor.area, 1e-12) / 3000.0), 1e-6)
    gap = max(0.02 * _length_unit(model), floor.tol)
    # the quads in cells of a coarse grid (a sample tests only the quads of its cell)
    cell = max(h * 8, 1e-9)
    bins: Dict[Tuple[int, int], list] = {}
    for name, q in quads:
        qx0, qx1 = min(c[0] for c in q) - gap, max(c[0] for c in q) + gap
        qy0, qy1 = min(c[1] for c in q) - gap, max(c[1] for c in q) + gap
        for i in range(int(math.floor((qx0 - x0) / cell)), int(math.floor((qx1 - x0) / cell)) + 1):
            for j in range(int(math.floor((qy0 - y0) / cell)), int(math.floor((qy1 - y0) / cell)) + 1):
                bins.setdefault((i, j), []).append((name, q, qx0, qx1, qy0, qy1))
    n = 0
    un = ov = 0
    w_un = w_ov = None
    y = y0 + 0.5 * h
    while y < y1:
        x = x0 + 0.5 * h
        while x < x1:
            p = (x, y)
            if floor._in_poly(p):
                n += 1
                state = {}
                for name, q, qx0, qx1, qy0, qy1 in bins.get((int(math.floor((x - x0) / cell)),
                                                             int(math.floor((y - y0) / cell))), ()):
                    if x < qx0 or x > qx1 or y < qy0 or y > qy1:
                        continue
                    v = _in_quad(p, q, gap)
                    if v >= 0:
                        state[name] = max(state.get(name, -1), v)
                inside = [nm for nm, v in state.items() if v == 1]
                if not state:
                    un += 1
                    w_un = w_un or p
                elif len(inside) > 1:
                    ov += 1
                    w_ov = w_ov or p
            x += h
        y += h
    return {"samples": n, "uncovered": un / n if n else 0.0, "overlap": ov / n if n else 0.0,
            "where_uncovered": w_un, "where_overlap": w_ov}
