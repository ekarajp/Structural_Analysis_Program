"""Panel loads: surface loads on a polygon that are carried to the frames (and
shells) under it - the CYPE 3D *load panels / surface loads*, extended with a
two-way option.

A panel is a planar polygon (any orientation).  Its loads per pattern are
uniform or vary linearly with height (wind / earth pressure on a sloping or
vertical panel).  The panel does not add stiffness: the load is converted to
member loads before the analysis, so the model stays the same with or without
panels.  The resulting line loads can be displayed on the members.

Distribution
------------
* ``one_way``: the panel spans in one direction ``span_dir`` (a unit vector in
  the panel plane).  The panel is cut into thin strips parallel to the span.
  Each strip is a simply supported continuous "joist" resting on every receiving
  bar it crosses (bars with a component perpendicular to the span direction,
  CYPE rule); the strip load between two consecutive supports is split half to
  each (statically determinate, as CYPE's *simply supported simulation*).  Load
  outside the outermost supports is cantilevered to the nearest support.
* ``two_way``: ETABS / SAFE style tributary areas: every point of the panel
  loads the nearest supporting bar measured in the panel plane (the 45-degree
  yield-line split for a rectangular panel on four beams, trapezoids and
  triangles).  Implemented by sampling the panel on a fine grid and assigning
  each sample to its nearest receiving bar.
* ``shells``: no bar distribution; the load is applied as pressure to the area
  objects overlapping the panel (the CYPE "shell inside a panel" rule).

The output of :func:`distribute` is, per receiving frame object, a piecewise
linear line load along the member (``stations`` in relative 0..1 and ``values``
force / length) plus the resultant checks, so the caller can compare the total
applied load with the total on the bars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

XYZ = Tuple[float, float, float]


@dataclass
class BarLoad:
    frame: str
    stations: np.ndarray  # relative positions 0..1 along the member (sorted)
    values: np.ndarray  # load per unit length at the stations (sign: along ``direction``)
    direction: np.ndarray  # global unit vector of the load
    total: float = 0.0  # resultant force on the bar (for checks)


@dataclass
class PanelResult:
    bar_loads: List[BarLoad]
    shell_pressures: Dict[str, float]  # area name -> pressure (force / area along the direction)
    applied_total: float  # total load of the panel (force)
    carried_total: float  # sum carried by bars and shells
    # "shells": the panel polygon for an area only partly under the panel (None / absent:
    # the whole area is loaded) and the loaded part of each area (area units)
    shell_regions: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    shell_loaded_area: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegionAreaLoad:
    """An area load on the part of an area inside ``region`` (3D polygon in the area's
    plane): a "shells" load panel drawn over part of a floor or wall."""

    pattern: str
    value: float
    direction: str
    region: np.ndarray
    fraction: float = 1.0  # loaded part of the whole area


def overlap_area_2d(subject: np.ndarray, region: np.ndarray) -> float:
    """Area of ``subject`` (any simple polygon) inside ``region`` (any simple polygon), 2D."""
    from .polygon2d import clip_convex, signed_area, triangulate
    R = np.asarray(region, float)
    tot = 0.0
    for i, j, k in triangulate(R):
        piece = clip_convex(np.asarray(subject, float), R[[i, j, k]])
        if len(piece) >= 3:
            tot += abs(signed_area(piece))
    return tot


# --------------------------------------------------------------- geometry
def polygon_frame(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(origin, e1, e2, normal) of a planar polygon (Newell normal)."""
    P = np.asarray(poly, float)
    n = np.zeros(3)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n += np.cross(a, b)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        raise ValueError("Panel polygon is degenerate")
    n /= ln
    e1 = P[1] - P[0]
    e1 -= n * (e1 @ n)
    if np.linalg.norm(e1) < 1e-12:
        e1 = np.cross(n, (0, 0, 1))
        if np.linalg.norm(e1) < 1e-12:
            e1 = np.array([1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return P[0].copy(), e1, e2, n


def polygon_area(poly2: np.ndarray) -> float:
    x, y = poly2[:, 0], poly2[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_poly2(p, poly2: np.ndarray, edge_tol: float = 1e-9) -> bool:
    """Inside test; points on (or within ``edge_tol`` of) an edge count as inside,
    so beams along the panel edges receive load."""
    x, y = p
    inside = False
    n = len(poly2)
    for i in range(n):
        x1, y1 = poly2[i]
        x2, y2 = poly2[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    if inside or edge_tol <= 0:
        return inside
    for i in range(n):
        a, b = poly2[i], poly2[(i + 1) % n]
        e = b - a
        L2 = float(e @ e)
        if L2 <= 0:
            continue
        t = min(max(float(((x - a[0]) * e[0] + (y - a[1]) * e[1]) / L2), 0.0), 1.0)
        if math.hypot(a[0] + t * e[0] - x, a[1] + t * e[1] - y) <= edge_tol:
            return True
    return False


def _edge_tol(poly2: np.ndarray) -> float:
    ext = float(np.max(poly2.max(axis=0) - poly2.min(axis=0))) or 1.0
    return 1e-6 * ext


def clip_segment_to_polygon(a2, b2, poly2: np.ndarray) -> List[Tuple[float, float]]:
    """Parameter intervals [(t0, t1)] of segment a2->b2 inside a convex or concave
    polygon (crossing count along the segment); the boundary counts as inside."""
    tol = _edge_tol(poly2)
    ts = [0.0, 1.0]
    d = b2 - a2
    n = len(poly2)
    for i in range(n):
        p, q = poly2[i], poly2[(i + 1) % n]
        e = q - p
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-14:
            continue
        r = p - a2
        t = (r[0] * e[1] - r[1] * e[0]) / den
        u = (r[0] * d[1] - r[1] * d[0]) / den
        if 0.0 <= u <= 1.0 and 0.0 < t < 1.0:
            ts.append(float(t))
    ts = sorted(set(ts))
    out = []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        mid = a2 + d * 0.5 * (t0 + t1)
        if point_in_poly2(mid, poly2, tol):
            out.append((t0, t1))
    return out


# ------------------------------------------------------------- receivers
@dataclass
class Receiver:
    name: str
    a: np.ndarray  # 2D in the panel plane
    b: np.ndarray
    a3: np.ndarray
    b3: np.ndarray


def receivers_in_panel(poly2: np.ndarray, frame_ends: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       to2, plane_tol: float, normal: np.ndarray, origin: np.ndarray) -> List[Receiver]:
    """Frames lying in the panel plane (both ends within ``plane_tol`` of it) with
    some part inside the polygon."""
    out = []
    for name, (a3, b3) in frame_ends.items():
        if abs((a3 - origin) @ normal) > plane_tol or abs((b3 - origin) @ normal) > plane_tol:
            continue
        a2, b2 = to2(a3), to2(b3)
        if np.linalg.norm(b2 - a2) < 1e-12:
            continue
        if clip_segment_to_polygon(a2, b2, poly2):
            out.append(Receiver(name, a2, b2, a3, b3))
    return out


# ------------------------------------------------------------ distribution
def _height_value(p3: np.ndarray, load) -> float:
    """Load intensity at a point: uniform, or linear in global Z between the
    panel's lowest and highest points."""
    if load.get("varying"):
        z0, z1, v0, v1 = load["z0"], load["z1"], load["value_lower"], load["value_upper"]
        if abs(z1 - z0) < 1e-12:
            return 0.5 * (v0 + v1)
        t = (p3[2] - z0) / (z1 - z0)
        return v0 + (v1 - v0) * min(max(t, 0.0), 1.0)
    return float(load["value"])


def distribute(poly: Sequence[XYZ], distribution: str, span_dir: Optional[XYZ], load: dict,
               direction: XYZ, frame_ends: Dict[str, Tuple[np.ndarray, np.ndarray]],
               receivers_allowed: Optional[set] = None, strips: int = 60, plane_tol: float = 1e-3,
               areas: Optional[Dict[str, np.ndarray]] = None, n_stations: int = 11,
               within: Optional[Sequence[XYZ]] = None) -> PanelResult:
    """Distribute one load (``load``: {"value"} or {"varying": True, "value_lower",
    "value_upper", "z0", "z1"}) of a panel.  ``direction``: global unit vector of
    the load (e.g. (0, 0, -1) for gravity, the panel normal for wind)."""
    P = np.asarray(poly, float)
    origin, e1, e2, normal = polygon_frame(P)

    def to2(p):
        d = np.asarray(p, float) - origin
        return np.array([d @ e1, d @ e2])

    def to3(q):
        return origin + q[0] * e1 + q[1] * e2

    poly2 = np.array([to2(p) for p in P])
    area = polygon_area(poly2)
    dvec = np.asarray(direction, float)
    dvec = dvec / (np.linalg.norm(dvec) or 1.0)
    # total applied load (numerical integration for the varying case)
    applied = 0.0
    if load.get("varying"):
        g = 40
        lo, hi = poly2.min(axis=0), poly2.max(axis=0)
        cnt = 0
        for i in range(g):
            for j in range(g):
                q = lo + (hi - lo) * ((i + 0.5) / g, (j + 0.5) / g)
                if point_in_poly2(q, poly2):
                    applied += _height_value(to3(q), load)
                    cnt += 1
        applied *= np.prod(hi - lo) / (g * g)
    else:
        applied = float(load["value"]) * area
    if distribution == "shells":
        # every area in the panel's plane under (part of) the panel: the load acts on the
        # part inside the panel only (an area half under the panel gets half the load)
        pressures, regions, loaded = {}, {}, {}
        if areas:
            for name, apoly in areas.items():
                if not all(abs((np.asarray(p) - origin) @ normal) <= plane_tol for p in apoly):
                    continue
                a2 = np.array([to2(p) for p in apoly])
                full = polygon_area(a2)
                inter = overlap_area_2d(a2, poly2)
                if full <= 0 or inter <= 1e-9 * full:
                    continue
                pressures[name] = _height_value(np.asarray(apoly, float).mean(axis=0), load)
                loaded[name] = inter
                regions[name] = None if inter >= full * (1 - 1e-9) else P.copy()
        carried = sum(pressures[n] * loaded[n] for n in pressures)
        return PanelResult([], pressures, applied, carried, regions, loaded)
    # ``within``: a host polygon (two-way only) - the receivers are the host's and only the
    # part of the panel inside the host counts (an opening taken out of a membrane floor)
    host2 = None if within is None else np.array([to2(p) for p in within])
    recs = receivers_in_panel(poly2 if host2 is None else host2, frame_ends, to2, plane_tol, normal, origin)
    if receivers_allowed is not None:
        recs = [r for r in recs if r.name in receivers_allowed]
    if not recs:
        return PanelResult([], {}, applied, 0.0)
    # accumulated load per receiver at fixed stations
    stations = np.linspace(0.0, 1.0, n_stations)
    acc: Dict[str, np.ndarray] = {r.name: np.zeros(n_stations) for r in recs}  # force per station bin
    if distribution == "one_way":
        if span_dir is None:
            raise ValueError("one-way panel needs a span direction")
        s3 = np.asarray(span_dir, float)
        s2 = np.array([s3 @ e1, s3 @ e2])
        if np.linalg.norm(s2) < 1e-12:
            raise ValueError("span direction must lie in the panel plane")
        s2 /= np.linalg.norm(s2)
        t2 = np.array([-s2[1], s2[0]])  # strip offset direction (perpendicular to the span)
        # receivers that can carry a one-way load: not parallel to the span (CYPE rule)
        carriers = [r for r in recs if abs(np.cross(np.append(r.b - r.a, 0), np.append(s2, 0))[2])
                    > 1e-9 * np.linalg.norm(r.b - r.a)]
        if not carriers:
            return PanelResult([], {}, applied, 0.0)
        offs = poly2 @ t2
        o0, o1 = float(offs.min()), float(offs.max())
        n_strip = max(strips, 4)
        w = (o1 - o0) / n_strip
        along = poly2 @ s2
        a0, a1 = float(along.min()) - 1.0, float(along.max()) + 1.0
        for k in range(n_strip):
            oc = o0 + (k + 0.5) * w
            A = s2 * a0 + t2 * oc
            B = s2 * a1 + t2 * oc
            inside = clip_segment_to_polygon(A, B, poly2)
            if not inside:
                continue
            # supports: crossings of the strip line with the carriers
            sup: List[Tuple[float, Receiver, float]] = []  # (t along strip, receiver, u along receiver)
            d = B - A
            for r in carriers:
                e = r.b - r.a
                den = d[0] * e[1] - d[1] * e[0]
                if abs(den) < 1e-14:
                    continue
                rr = r.a - A
                t = (rr[0] * e[1] - rr[1] * e[0]) / den
                u = (rr[0] * d[1] - rr[1] * d[0]) / den
                if -1e-9 <= u <= 1 + 1e-9:
                    sup.append((t, r, min(max(u, 0.0), 1.0)))
            if not sup:
                # the strip passes beyond the ends of every carrier (a panel edge strip
                # past the last column line): its load goes to the nearest bar (the
                # two-way rule) instead of being lost
                for t0, t1 in inside:
                    nseg = 8
                    for i in range(nseg):
                        ta = t0 + (t1 - t0) * i / nseg
                        tb = t0 + (t1 - t0) * (i + 1) / nseg
                        pm = A + d * (0.5 * (ta + tb))
                        q = _height_value(to3(pm), load) * w * (tb - ta) * np.linalg.norm(d)
                        if q == 0.0:
                            continue
                        best = None
                        for r in recs:
                            e = r.b - r.a
                            L2 = float(e @ e)
                            u = 0.0 if L2 <= 0 else min(max(float((pm - r.a) @ e) / L2, 0.0), 1.0)
                            dist = float(np.linalg.norm(pm - (r.a + u * e)))
                            if best is None or dist < best[0]:
                                best = (dist, r, u)
                        if best is not None:
                            _add(acc, (0.0, best[1], best[2]), q, stations)
                continue
            sup.sort(key=lambda x: x[0])
            # integrate the strip load (piecewise) and hand it to the supports
            for t0, t1 in inside:
                nseg = 8
                for i in range(nseg):
                    ta = t0 + (t1 - t0) * i / nseg
                    tb = t0 + (t1 - t0) * (i + 1) / nseg
                    tm = 0.5 * (ta + tb)
                    pm = A + d * tm
                    q = _height_value(to3(pm), load) * w * (tb - ta) * np.linalg.norm(d)  # force of this piece
                    if q == 0.0:
                        continue
                    # supports left / right of the piece
                    left = [s for s in sup if s[0] <= tm]
                    right = [s for s in sup if s[0] > tm]
                    if left and right:
                        sl, sr = left[-1], right[0]
                        span = sr[0] - sl[0]
                        fr = (tm - sl[0]) / span if span > 0 else 0.5
                        _add(acc, sl, q * (1 - fr), stations)
                        _add(acc, sr, q * fr, stations)
                    elif left:
                        _add(acc, left[-1], q, stations)  # cantilever beyond the last support
                    else:
                        _add(acc, right[0], q, stations)
    elif distribution == "two_way":
        g = 64
        lo, hi = poly2.min(axis=0), poly2.max(axis=0)
        cell = np.prod(hi - lo) / (g * g)
        sub = 4  # sub-samples per cell: the fraction of a boundary cell inside the panel
        offs = [((a + 0.5) / sub, (b + 0.5) / sub) for a in range(sub) for b in range(sub)]
        for i in range(g):
            for j in range(g):
                corners_in = [point_in_poly2(lo + (hi - lo) * ((i + a) / g, (j + b) / g), poly2)
                              for a in (0, 1) for b in (0, 1)]
                if not any(corners_in):
                    q2 = lo + (hi - lo) * ((i + 0.5) / g, (j + 0.5) / g)
                    if not point_in_poly2(q2, poly2):
                        continue
                    frac = 1.0
                elif all(corners_in):
                    frac = 1.0
                else:
                    frac = sum(point_in_poly2(lo + (hi - lo) * ((i + a) / g, (j + b) / g), poly2)
                               for a, b in offs) / len(offs)
                    if frac == 0.0:
                        continue
                q2 = lo + (hi - lo) * ((i + 0.5) / g, (j + 0.5) / g)
                if host2 is not None and not point_in_poly2(q2, host2):
                    continue
                force = _height_value(to3(q2), load) * cell * frac
                best = None
                for r in recs:
                    e = r.b - r.a
                    L2 = float(e @ e)
                    u = float((q2 - r.a) @ e) / L2
                    uc = min(max(u, 0.0), 1.0)
                    dist = float(np.linalg.norm(r.a + uc * e - q2))
                    if best is None or dist < best[0]:
                        best = (dist, r, uc)
                _add(acc, (0.0, best[1], best[2]), force, stations)
    else:
        raise ValueError(f"unknown panel distribution {distribution!r}")
    out = []
    carried = 0.0
    for r in recs:
        forces = acc[r.name]
        if not np.any(forces):
            continue
        L = float(np.linalg.norm(r.b3 - r.a3))
        # bin forces -> load per unit length at the stations (bins half-width at the ends)
        widths = np.full(n_stations, L / (n_stations - 1))
        widths[0] *= 0.5
        widths[-1] *= 0.5
        values = forces / widths
        out.append(BarLoad(r.name, stations.copy(), values, dvec, float(forces.sum())))
        carried += float(forces.sum())
    if host2 is not None:
        applied = carried  # the part inside the host
    return PanelResult(out, {}, applied, carried)


def _add(acc: Dict[str, np.ndarray], sup, force: float, stations: np.ndarray):
    """Put ``force`` on receiver ``sup[1]`` at relative position ``sup[2]`` (shared
    linearly between the two neighbouring stations)."""
    _, r, u = sup
    n = len(stations)
    x = u * (n - 1)
    i = int(math.floor(x))
    if i >= n - 1:
        acc[r.name][n - 1] += force
        return
    f = x - i
    acc[r.name][i] += force * (1 - f)
    acc[r.name][i + 1] += force * f


def simplify(values: np.ndarray, stations: np.ndarray, tol: float = 1e-9) -> List[Tuple[float, float, float, float]]:
    """Piecewise linear segments (rel_a, rel_b, value_a, value_b) covering the
    stations (one per interval; zero intervals dropped)."""
    out = []
    for k in range(len(stations) - 1):
        va, vb = float(values[k]), float(values[k + 1])
        if abs(va) < tol and abs(vb) < tol:
            continue
        out.append((float(stations[k]), float(stations[k + 1]), va, vb))
    return out


# ----------------------------------------------------------- model bridge
def panel_direction(panel_points, text: str) -> Tuple[np.ndarray, bool]:
    """Global unit vector of a panel load direction ("Gravity", "X", "Normal",
    ...) and whether it is projected (only the area normal to the load counts)."""
    t = str(text).strip().lower()
    projected = t.endswith("projected")
    if projected:
        t = t[: -len("projected")].strip(" _-")
    if t in ("normal", "local 3", "local3", "3"):
        _, _, _, n = polygon_frame(np.asarray(panel_points, float))
        return n, False
    dirs = {"gravity": (0, 0, -1), "grav": (0, 0, -1), "-z": (0, 0, -1), "x": (1, 0, 0), "y": (0, 1, 0),
            "z": (0, 0, 1), "-x": (-1, 0, 0), "-y": (0, -1, 0)}
    if t not in dirs:
        raise ValueError(f"Unknown panel load direction {text!r}")
    return np.array(dirs[t], float), projected


def _overlap3(subject, region) -> float:
    """Area of the 3D polygon ``subject`` inside the 3D polygon ``region`` (same plane)."""
    S = np.asarray(subject, float)
    o, e1, e2, _ = polygon_frame(np.asarray(region, float))
    to = lambda Q: np.array([[(q - o) @ e1, (q - o) @ e2] for q in np.asarray(Q, float)])
    return overlap_area_2d(to(S), to(region))


def _poly3_area(P) -> float:
    P = np.asarray(P, float)
    n = np.zeros(3)
    for i in range(len(P)):
        n += np.cross(P[i], P[(i + 1) % len(P)])
    return 0.5 * float(np.linalg.norm(n))


def membrane_floors(model) -> Dict[str, float]:
    """Floors / ramps whose shell section is a Membrane -> their self weight per area
    (model units).  A membrane has no bending stiffness: the loads normal to it go
    to the frames that support it (two-way, ETABS), as does its self weight; the
    membrane itself keeps its in-plane stiffness (diaphragm) and its mass."""
    out = {}
    for a in model.areas.values():
        if a.type not in ("Floor", "Ramp") or a.is_opening:
            continue
        sec = model.shell_sections.get(a.section)
        if sec is None or sec.model_type != "Membrane":
            continue
        mat = model.materials.get(sec.material)
        mods = sec.modifiers.combined(a.modifiers)
        wd = float(getattr(mat, "weight_density", 0.0) or 0.0) if mat is not None else 0.0
        out[a.name] = (wd * float(sec.thickness or 0.0) + float(sec.weight_per_area or 0.0)) * float(mods.weight)
    return out


def panel_loads_to_members(model, strips: int = 60):
    """Convert every load panel of ``model`` into frame loads (model units, object
    relative coordinates) and area loads.  Returns ({frame: [FrameLoad]},
    {area: [AreaLoad]}, report) where report[panel][pattern] = (applied, carried)."""
    from ..model.area_panels import area_items
    from ..model.objects import AreaLoad, FrameLoad

    frame_loads: Dict[str, List] = {}
    area_loads: Dict[str, List] = {}
    report: Dict[str, Dict[str, Tuple[float, float]]] = {}
    membranes = membrane_floors(model)
    if not getattr(model, "panel_loads", None) and not membranes:
        return frame_loads, area_loads, report
    pts = model.points
    from .geometry import beam_axis_shift
    tol_model = max(getattr(model.options, "merge_tolerance", 1e-3), 1e-9)
    frame_ends, shifted_ends = {}, {}
    for f in model.frames.values():
        if f.type == "Rigid":
            continue  # a rigid link carries no load
        a, b = np.asarray(pts[f.point_i].xyz, float), np.asarray(pts[f.point_j].xyz, float)
        frame_ends[f.name] = (a, b)
        sh = beam_axis_shift(model, f, tol_model)  # beams meshed on their true axis are loaded there
        if sh is not None:
            shifted_ends[f.name] = (a + sh[0], b + sh[1])

    def ends_for(P):
        """The beams on their true axis where that axis is on the panel (else - a panel
        drawn only to the grid line - the joint-to-joint line, as before)."""
        if not shifted_ends:
            return frame_ends
        try:
            o, e1, e2, n = polygon_frame(P)
        except ValueError:
            return frame_ends
        poly2 = np.array([[(q - o) @ e1, (q - o) @ e2] for q in P])
        out = dict(frame_ends)
        for name, (a, b) in shifted_ends.items():
            if abs(float((a - o) @ n)) > tol or abs(float((b - o) @ n)) > tol:
                continue
            if clip_segment_to_polygon(np.array([(a - o) @ e1, (a - o) @ e2]),
                                       np.array([(b - o) @ e1, (b - o) @ e2]), poly2):
                out[name] = (a, b)
        return out
    area_polys = {a.name: np.array([pts[p].xyz for p in a.points], float) for a in model.areas.values()}
    tol = max(getattr(model.options, "merge_tolerance", 1e-3), 1e-6) * 2.0
    def to_frames(res, pattern, dvec):
        for bl in res.bar_loads:
            f = model.frames[bl.frame]
            for ra, rb, va, vb in simplify(bl.values, bl.stations):
                frame_loads.setdefault(f.name, []).append(
                    FrameLoad(pattern, "distributed", _direction_text(dvec), "Force", ra, rb, False, va, vb))

    hole_cache: Dict[str, list] = {}

    def holes_of(aname):
        """Openings in the plane of area ``aname`` with their centre inside it."""
        if aname not in hole_cache:
            A = area_polys[aname]
            out = []
            try:
                o, e1, e2, n = polygon_frame(A)
                A2 = np.array([[(q - o) @ e1, (q - o) @ e2] for q in A])
                for op in model.areas.values():
                    if not op.is_opening or op.name == aname or len(op.points) < 3:
                        continue
                    H = area_polys[op.name]
                    if np.max(np.abs((H - o) @ n)) > tol:
                        continue
                    c = H.mean(axis=0) - o
                    if point_in_poly2(np.array([c @ e1, c @ e2]), A2):
                        out.append(H)
            except ValueError:
                pass
            hole_cache[aname] = out
        return hole_cache[aname]

    def membrane_load(aname, pattern, value, direction, rep):
        """A load on a membrane floor: its part normal to the floor goes two-way to
        the frames (the membrane cannot carry it); a load in its plane stays on it."""
        A = area_polys[aname]
        dvec, _ = panel_direction(A, "Normal" if str(direction).strip() == "3" else direction)
        _, _, _, n = polygon_frame(A)
        if abs(float(n @ dvec)) <= 1e-9:
            area_loads.setdefault(aname, []).append(AreaLoad(pattern, value, direction))
            return
        ends = ends_for(A)
        res = distribute(A, "two_way", None, {"value": value}, tuple(dvec), ends, None, strips, tol, area_polys)
        to_frames(res, pattern, dvec)
        applied, carried = res.applied_total, res.carried_total
        if res.carried_total:
            for H in holes_of(aname):  # openings carry nothing: their share is taken back
                cut = distribute(H, "two_way", None, {"value": -value}, tuple(dvec), ends, None, strips, tol,
                                 area_polys, within=A)
                to_frames(cut, pattern, dvec)
                applied += cut.applied_total
                carried += cut.carried_total
        old = rep.get(pattern, (0.0, 0.0))
        rep[pattern] = (old[0] + applied, old[1] + carried)

    for aname, w in membranes.items():  # self weight of the membrane floors
        rep = {}
        for pat in model.load_patterns.values():
            mult = float(getattr(pat, "self_weight_multiplier", 0.0) or 0.0)
            if mult and w:
                membrane_load(aname, pat.name, w * mult, "Gravity", rep)
        for ld in getattr(model.areas[aname], "loads", []) or []:  # loads kept on the area object
            membrane_load(aname, ld.pattern, ld.value, ld.direction, rep)
        if rep:
            report[f"{aname} (membrane: self weight and area loads to the frames)"] = rep

    for panel in model.panel_loads.values():
        rep: Dict[str, Tuple[float, float]] = {}
        bound = model.areas.get(panel.area) if getattr(panel, "area", None) else None
        if bound is not None and panel.distribution == "shells" and bound.name in membranes:
            for ld in area_items(panel):
                membrane_load(bound.name, ld.pattern, ld.value, ld.direction, rep)
            report[panel.name] = rep
            continue
        if bound is not None and panel.distribution == "shells":
            # an area load on its own area (exactly the ETABS / SAP area load, local axes kept)
            from ..model.area_panels import area_items
            A = np.array([pts[p].xyz for p in bound.points], float)
            try:
                o, e1, e2, _ = polygon_frame(A)
                ar = polygon_area(np.array([[(q - o) @ e1, (q - o) @ e2] for q in A]))
            except ValueError:
                ar = 0.0
            for ld in area_items(panel):
                area_loads.setdefault(bound.name, []).append(ld)
                old = rep.get(ld.pattern, (0.0, 0.0))
                carried = ld.value * ar if not bound.is_opening else 0.0
                rep[ld.pattern] = (old[0] + ld.value * ar, old[1] + carried)
            report[panel.name] = rep
            continue
        # a panel made from an area follows the area's current polygon
        P = np.asarray([pts[p].xyz for p in bound.points] if bound is not None else panel.points, float)
        if len(P) < 3:
            continue
        zs = P[:, 2]
        allowed = set(panel.receivers) if panel.receivers is not None else None
        panel_ends = ends_for(P)
        for case in panel.loads:
            dvec, projected = panel_direction(P, case.direction)
            load = {"value": case.value}
            if case.varying:
                load = {"varying": True, "value_lower": case.value_lower, "value_upper": case.value_upper,
                        "z0": float(zs.min()), "z1": float(zs.max())}
            if projected:
                _, _, _, n = polygon_frame(P)
                cosang = abs(float(n @ dvec))
                if case.varying:
                    load = dict(load, value_lower=load["value_lower"] * cosang, value_upper=load["value_upper"] * cosang)
                else:
                    load = {"value": case.value * cosang}
            res = distribute(P, panel.distribution, panel.span_dir, load, tuple(dvec), panel_ends, allowed, strips,
                             tol, area_polys)
            to_frames(res, case.pattern, dvec)
            carried = res.carried_total
            if panel.distribution == "shells":
                # what really lands: the floors / walls under the panel less their openings
                # (an opening is no receiver; a membrane floor passes its share to its frames)
                carried = 0.0
                for aname, pressure in res.shell_pressures.items():
                    if model.areas[aname].is_opening:
                        continue
                    if aname in membranes:
                        sub: Dict[str, Tuple[float, float]] = {}
                        membrane_load(aname, case.pattern, pressure, _direction_text(dvec), sub)
                        carried += sum(c_ for _, c_ in sub.values())
                        continue
                    region = res.shell_regions.get(aname)
                    if region is None:
                        area_loads.setdefault(aname, []).append(AreaLoad(case.pattern, pressure, _direction_text(dvec)))
                        carried += pressure * (_poly3_area(area_polys[aname]) - sum(_poly3_area(H) for H in holes_of(aname)))
                    else:  # the part of the area under the panel only
                        full = _poly3_area(area_polys[aname])
                        part = res.shell_loaded_area.get(aname, 0.0) - sum(_overlap3(H, region) for H in holes_of(aname))
                        area_loads.setdefault(aname, []).append(RegionAreaLoad(
                            case.pattern, pressure, _direction_text(dvec), np.asarray(region, float),
                            part / full if full > 0 else 0.0))
                        carried += pressure * part
            old = rep.get(case.pattern, (0.0, 0.0))
            rep[case.pattern] = (old[0] + res.applied_total, old[1] + carried)
        report[panel.name] = rep
    return frame_loads, area_loads, report


def _direction_text(d: np.ndarray) -> str:
    """A FrameLoad direction string for a global unit vector: the axis name when
    it is one, else the vector itself ("vec:x,y,z")."""
    for name, v in (("Gravity", (0, 0, -1)), ("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1)),
                    ("-X", (-1, 0, 0)), ("-Y", (0, -1, 0))):
        if np.allclose(d, v, atol=1e-9):
            return name
    return "vec:" + ",".join(f"{float(x):.9g}" for x in d)
