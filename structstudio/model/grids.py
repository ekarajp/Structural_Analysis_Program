"""Grid systems in plan: Cartesian (X / Y lines, turned and moved by the system's origin
and rotation) and cylindrical, as ETABS: "R" lines are circles of constant radius
(ordinate = R), "T" lines radial lines of constant angle theta (ordinate = degrees from
the system's local x axis, counter-clockwise).

``grid_geometry`` gives everything the program draws and snaps to, in global plan
coordinates: straight lines, arcs, the bubble positions and the grid points (the
intersections of the lines of each system).  Grid lines in the elevation sense
(direction "Z", SAP2000 grid mode) are levels and are not handled here.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

XY = Tuple[float, float]

CARTESIAN = "Cartesian"
CYLINDRICAL = "Cylindrical"
PLAN_DIRECTIONS = ("X", "Y", "T", "R", "General")


class GridGeometry:
    """What the grid systems draw and snap to, in global plan coordinates (computed
    from the model each time: never stored, so no unit rules)."""

    def __init__(self):
        self.lines: List[Tuple[XY, XY, str, str]] = []  # (a, b, name, system)
        self.arcs: List[Tuple[XY, float, float, float, str, str]] = []  # (centre, r, a0, a1 deg, name, system)
        self.bubbles: List[Tuple[XY, XY, str]] = []  # (end of the line, outward unit, name)
        self.points: List[XY] = []
        self.extent: Optional[Tuple[float, float, float, float]] = None  # x0, x1, y0, y1
        self.ext: float = 1.0  # how far the lines run past the outermost crossing lines
        self.general: List[Tuple[XY, XY]] = []  # General lines (their crossings are grid points too)

    def polylines(self, seg_deg: float = 6.0) -> List[List[XY]]:
        """Lines and arcs as polylines (arcs in steps of about ``seg_deg``)."""
        out = [[a, b] for a, b, _, _ in self.lines]
        for c, r, a0, a1, _, _ in self.arcs:
            n = max(4, int(math.ceil(abs(a1 - a0) / seg_deg)))
            out.append([(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
                         c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)])
        return out


def system_of(model, gl):
    gs = model.grid_systems.get(gl.system)
    if gs is None:
        from .objects import GridSystem
        gs = GridSystem(name=gl.system or "G1")
    return gs


def to_global(gs, u: float, v: float) -> XY:
    c, s = math.cos(math.radians(gs.rotation)), math.sin(math.radians(gs.rotation))
    ox, oy = (gs.origin or (0.0, 0.0))
    return ox + c * u - s * v, oy + s * u + c * v


def to_local(gs, x: float, y: float) -> XY:
    c, s = math.cos(math.radians(gs.rotation)), math.sin(math.radians(gs.rotation))
    ox, oy = (gs.origin or (0.0, 0.0))
    dx, dy = x - ox, y - oy
    return c * dx + s * dy, -s * dx + c * dy


def is_plain(gs) -> bool:
    """A Cartesian system along the global axes (its X / Y ordinates are global x / y
    after the origin): what elevation windows, fine grids and the like can use."""
    return gs.type != CYLINDRICAL and abs(((gs.rotation or 0.0) + 180.0) % 360.0 - 180.0) < 1e-9


def global_ordinate(model, gl) -> Optional[float]:
    """Global x (X lines) / y (Y lines) of a line of a Cartesian system along the axes;
    None for other lines."""
    gs = system_of(model, gl)
    if gl.direction not in ("X", "Y") or not is_plain(gs):
        return None
    o = gs.origin or (0.0, 0.0)
    return gl.ordinate + (o[0] if gl.direction == "X" else o[1])


def _lines_by_system(model) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for g in model.grid_lines:
        if g.direction in PLAN_DIRECTIONS and getattr(g, "visible", True):
            out.setdefault(g.system or "G1", []).append(g)
    return out


def grid_geometry(model) -> GridGeometry:
    out = GridGeometry()
    systems = _lines_by_system(model)
    if not systems:
        return out
    # the run-past distance: 10 % of the whole grid (as before), at least 1
    spans = []
    for name, lines in systems.items():
        gs = model.grid_systems.get(name) or system_of(model, lines[0])
        if gs.type == CYLINDRICAL:
            rs = [g.ordinate for g in lines if g.direction == "R"]
            spans.append(2 * max(rs) if rs else 1.0)
        else:
            us = [g.ordinate for g in lines if g.direction == "X"]
            vs = [g.ordinate for g in lines if g.direction == "Y"]
            spans.append(max((max(us) - min(us)) if us else 0.0, (max(vs) - min(vs)) if vs else 0.0))
    ext = max(max(spans) * 0.1, 1.0) if spans else 1.0
    out.ext = ext
    pts_all: List[XY] = []
    for name, lines in systems.items():
        gs = model.grid_systems.get(name) or system_of(model, lines[0])
        if gs.type == CYLINDRICAL:
            _cylindrical(gs, lines, ext, out)
        else:
            _cartesian(gs, lines, ext, out)
    _extra_crossings(out)
    for a, b, _, _ in out.lines:
        pts_all += [a, b]
    for c, r, a0, a1, _, _ in out.arcs:
        for a in (a0, a1, 0.5 * (a0 + a1)) + tuple(t for t in (0, 90, 180, 270, 360, -90, -180)
                                                  if min(a0, a1) <= t <= max(a0, a1)):
            pts_all.append((c[0] + r * math.cos(math.radians(a)), c[1] + r * math.sin(math.radians(a))))
    if pts_all:
        xs, ys = [p[0] for p in pts_all], [p[1] for p in pts_all]
        out.extent = (min(xs), max(xs), min(ys), max(ys))
    return out


def _extra_crossings(out: GridGeometry) -> None:
    """Grid points where lines meet that the systems' own crossings miss: a General line
    with any line, lines of two different systems, and straight lines with circles (and
    circles of two different systems) - so they snap like the X / Y crossings."""
    seen = {(round(x, 9), round(y, 9)) for x, y in out.points}

    def add(q):
        key = (round(q[0], 9), round(q[1], 9))
        if key not in seen:
            seen.add(key)
            out.points.append(q)
    gen = set(out.general)
    lines = out.lines
    for k in range(len(lines)):
        a, b, _, sa = lines[k]
        for m in range(k + 1, len(lines)):
            c, d, _, sc = lines[m]
            if sa != sc or (a, b) in gen or (c, d) in gen:
                q = _segment_crossing(a, b, c, d)
                if q is not None:
                    add(q)
    for c0, r, a0, a1, _, sc in out.arcs:
        for a, b, _, sa in lines:
            if sa == sc and (a, b) not in gen:
                continue  # (a system's own radial lines: its R x T points already)
            for q in _segment_circle(a, b, c0, r):
                if _on_arc(q, c0, a0, a1):
                    add(q)
    arcs = out.arcs
    for k in range(len(arcs)):
        for m in range(k + 1, len(arcs)):
            c1, r1, a10, a11, _, s1 = arcs[k]
            c2, r2, a20, a21, _, s2 = arcs[m]
            if s1 == s2:
                continue
            for q in _circle_circle(c1, r1, c2, r2):
                if _on_arc(q, c1, a10, a11) and _on_arc(q, c2, a20, a21):
                    add(q)


def _segment_circle(a: XY, b: XY, c: XY, r: float) -> List[XY]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    fx, fy = a[0] - c[0], a[1] - c[1]
    A = dx * dx + dy * dy
    if A <= 0:
        return []
    B = 2 * (fx * dx + fy * dy)
    C = fx * fx + fy * fy - r * r
    disc = B * B - 4 * A * C
    if disc < -1e-12 * max(1.0, B * B):
        return []
    sq = math.sqrt(max(disc, 0.0))
    out = []
    for t in sorted({(-B - sq) / (2 * A), (-B + sq) / (2 * A)}):
        if -1e-9 <= t <= 1 + 1e-9:
            out.append((a[0] + t * dx, a[1] + t * dy))
    return out


def _circle_circle(c1: XY, r1: float, c2: XY, r2: float) -> List[XY]:
    d = math.dist(c1, c2)
    if d <= 1e-12 or d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9:
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(max(r1 * r1 - a * a, 0.0))
    mx, my = c1[0] + a * (c2[0] - c1[0]) / d, c1[1] + a * (c2[1] - c1[1]) / d
    ox, oy = -h * (c2[1] - c1[1]) / d, h * (c2[0] - c1[0]) / d
    return [(mx + ox, my + oy)] if h <= 1e-12 else [(mx + ox, my + oy), (mx - ox, my - oy)]


def _on_arc(q: XY, c: XY, a0: float, a1: float) -> bool:
    if abs(a1 - a0) >= 360.0 - 1e-9:
        return True
    ang = math.degrees(math.atan2(q[1] - c[1], q[0] - c[0]))
    lo = min(a0, a1)
    return (ang - lo) % 360.0 <= abs(a1 - a0) + 1e-7


def _segment_crossing(a: XY, b: XY, c: XY, d: XY) -> Optional[XY]:
    """Where segment ab crosses segment cd (ends included), None if they do not."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    den = rx * sy - ry * sx
    if abs(den) <= 1e-12 * max(1.0, rx * rx + ry * ry, sx * sx + sy * sy):
        return None
    t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / den
    u = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / den
    eps = 1e-9
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        return a[0] + t * rx, a[1] + t * ry
    return None


def _cartesian(gs, lines, ext, out: GridGeometry):
    us = sorted({g.ordinate for g in lines if g.direction == "X"})
    vs = sorted({g.ordinate for g in lines if g.direction == "Y"})
    u0, u1 = (min(us), max(us)) if us else (0.0, 0.0)
    v0, v1 = (min(vs), max(vs)) if vs else (0.0, 0.0)
    rot = math.radians(gs.rotation or 0.0)
    eu = (math.cos(rot), math.sin(rot))
    ev = (-math.sin(rot), math.cos(rot))
    for g in lines:
        if g.direction == "X":
            a, b = to_global(gs, g.ordinate, v0 - ext), to_global(gs, g.ordinate, v1 + ext)
            out.lines.append((a, b, g.name, gs.name))
            start = getattr(g, "bubble_location", "End") == "Start"
            out.bubbles.append((a, (-ev[0], -ev[1]), g.name) if start else (b, ev, g.name))
        elif g.direction == "Y":
            a, b = to_global(gs, u0 - ext, g.ordinate), to_global(gs, u1 + ext, g.ordinate)
            out.lines.append((a, b, g.name, gs.name))
            # Y lines: the bubble at the start (left, as the program always drew them); "Start"
            # in the CSI sense flips it to the other end
            flip = getattr(g, "bubble_location", "End") == "Start"
            out.bubbles.append((b, eu, g.name) if flip else (a, (-eu[0], -eu[1]), g.name))
        elif g.direction == "General" and g.start and g.end:
            a, b = to_global(gs, *g.start), to_global(gs, *g.end)
            out.lines.append((a, b, g.name, gs.name))
            out.general.append((a, b))
            d = (b[0] - a[0], b[1] - a[1])
            L = math.hypot(*d) or 1.0
            out.bubbles.append((b, (d[0] / L, d[1] / L), g.name))
    out.points += [to_global(gs, u, v) for u in us for v in vs]


def _cylindrical(gs, lines, ext, out: GridGeometry):
    angs = sorted({g.ordinate % 360.0 for g in lines if g.direction == "T"})
    rads = sorted({g.ordinate for g in lines if g.direction == "R" and g.ordinate > 0})
    o = gs.origin or (0.0, 0.0)
    rot = gs.rotation or 0.0
    r_out = (max(rads) if rads else 0.0) + ext
    full = len(angs) < 2 or _covers_circle(angs)
    if full:
        a0, a1 = 0.0, 360.0
    else:
        lo, hi = _arc_range(angs)
        pad = math.degrees(ext / max(rads[-1] if rads else 1.0, 1e-9)) if rads else 10.0
        pad = min(pad, 20.0)
        a0, a1 = lo - pad, hi + pad
    for g in lines:
        if g.direction == "T":
            t = math.radians(g.ordinate + rot)
            d = (math.cos(t), math.sin(t))
            b = (o[0] + r_out * d[0], o[1] + r_out * d[1])
            out.lines.append(((o[0], o[1]), b, g.name, gs.name))
            out.bubbles.append((b, d, g.name))
        elif g.direction == "R" and g.ordinate > 0:
            out.arcs.append(((o[0], o[1]), g.ordinate, a0 + rot, a1 + rot, g.name, gs.name))
            t = math.radians(a0 + rot)
            if full:  # bubble between two radial lines, the gap nearest the bottom of the plan
                t = math.radians(_free_angle(angs, rot))
                p = (o[0] + g.ordinate * math.cos(t), o[1] + g.ordinate * math.sin(t))
                out.bubbles.append((p, (math.cos(t), math.sin(t)), g.name))
            else:  # at the start of the arc, along it (backwards)
                p = (o[0] + g.ordinate * math.cos(t), o[1] + g.ordinate * math.sin(t))
                out.bubbles.append((p, (math.sin(t), -math.cos(t)), g.name))
    for a in angs:
        t = math.radians(a + rot)
        out.points += [(o[0] + r * math.cos(t), o[1] + r * math.sin(t)) for r in rads]
    if len(angs) >= 2 or (angs and rads):
        out.points.append((o[0], o[1]))


def _free_angle(angs, rot: float) -> float:
    """Global angle (deg) halfway between two radial lines, the gap nearest straight down."""
    if not angs:
        return 270.0
    n = len(angs)
    mids = []
    for k in range(n):
        a, b = angs[k], angs[(k + 1) % n] + (360.0 if k == n - 1 else 0.0)
        mids.append(0.5 * (a + b) + rot)
    return min(mids, key=lambda m: abs(((m - 270.0) + 180.0) % 360.0 - 180.0))


def _covers_circle(angs) -> bool:
    """Radial lines all round (no gap wider than 120 degrees): full circles."""
    if len(angs) < 3:
        return False
    n = len(angs)
    gaps = [(angs[(k + 1) % n] - angs[k]) % 360.0 or 360.0 for k in range(n)]
    return max(gaps) <= 120.0 + 1e-9


def _arc_range(angs) -> Tuple[float, float]:
    """Smallest arc holding all the angles: (start, end) in degrees, end > start."""
    n = len(angs)
    gaps = [((angs[(k + 1) % n] - angs[k]) % 360.0, k) for k in range(n)]
    g, k = max(gaps)
    start = angs[(k + 1) % n]
    end = start + 360.0 - g
    return start, end


def grid_points(model) -> List[XY]:
    """The grid points of every system in plan (global x, y)."""
    return grid_geometry(model).points


def grid_names_at(model, x: float, y: float, tol: float) -> List[str]:
    """Names of the grid lines passing through (x, y) (for labels such as A-1)."""
    out = []
    for g in model.grid_lines:
        gs = system_of(model, g)
        if g.direction not in PLAN_DIRECTIONS:
            continue
        if gs.type == CYLINDRICAL:
            o = gs.origin or (0.0, 0.0)
            dx, dy = x - o[0], y - o[1]
            r = math.hypot(dx, dy)
            if g.direction == "R" and abs(r - g.ordinate) <= tol:
                out.append(g.name)
            elif g.direction == "T" and r > tol:
                a = (math.degrees(math.atan2(dy, dx)) - (gs.rotation or 0.0)) % 360.0
                if abs(((a - g.ordinate) + 180.0) % 360.0 - 180.0) * math.pi / 180.0 * r <= tol:
                    out.append(g.name)
        else:
            u, v = to_local(gs, x, y)
            if (g.direction == "X" and abs(u - g.ordinate) <= tol) or (g.direction == "Y" and abs(v - g.ordinate) <= tol):
                out.append(g.name)
    return out


def line_intersection(model, a, b) -> Optional[XY]:
    """Global plan point where two lines of the same system cross (X with Y, R with T,
    R with R at the centre), None otherwise."""
    gs = system_of(model, a)
    if gs.type == CYLINDRICAL:
        o = gs.origin or (0.0, 0.0)
        dirs = {a.direction, b.direction}
        if dirs == {"T", "R"}:
            ang = a.ordinate if a.direction == "T" else b.ordinate
            r = a.ordinate if a.direction == "R" else b.ordinate
            t = math.radians(ang + (gs.rotation or 0.0))
            return o[0] + r * math.cos(t), o[1] + r * math.sin(t)
        if dirs == {"T"}:
            return o[0], o[1]
        return None
    if {a.direction, b.direction} == {"X", "Y"}:
        u = a.ordinate if a.direction == "X" else b.ordinate
        v = a.ordinate if a.direction == "Y" else b.ordinate
        return to_global(gs, u, v)
    return None
