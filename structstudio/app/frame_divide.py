"""Where to divide frames (Edit > Edit Frames > Divide Frames).

Each finder returns, per frame, the relative stations (0 < s < 1, from end I) at
which the frame is cut; :func:`structstudio.app.tools.divide_frames_at` does the
cutting (loads, releases, hinges and groups follow).  Lengths are in model units.

* equal pieces / maximum piece length / given distances from an end
* existing joints that lie on the frame (a frame drawn over a joint is not connected to it)
* other frames that cross it (in 3-D, within a tolerance; the crossing frames can be cut too)
* grid lines and arcs (in plan; vertical frames do not cross them)
* reference lines through joints (lines parallel to X, Y, Z through every joint, CYPE 3D style)
* story levels (columns and braces that run through several stories)
* shell edges (floor and wall edges that the frame crosses)
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Stations = Dict[str, List[float]]


def _ends(model, name):
    f = model.frames[name]
    return np.asarray(model.points[f.point_i].xyz, float), np.asarray(model.points[f.point_j].xyz, float)


def _clean(ss: Iterable[float], L: float, tol: float) -> List[float]:
    out: List[float] = []
    for s in sorted(float(x) for x in ss):
        if s * L > tol and (1 - s) * L > tol and (not out or (s - out[-1]) * L > tol):
            out.append(s)
    return out


def _tol(model, tol: Optional[float]) -> float:
    return max(float(tol) if tol else 0.0, model.options.merge_tolerance)


# ------------------------------------------------------------- by number / length
def equal(model, frames: Sequence[str], n: int) -> Stations:
    n = int(n)
    return {f: [k / n for k in range(1, n)] for f in frames if f in model.frames and n >= 2}


def max_length(model, frames: Sequence[str], lmax: float) -> Stations:
    """The fewest equal pieces none longer than ``lmax``."""
    out: Stations = {}
    if not lmax or lmax <= 0:
        return out
    for f in frames:
        if f not in model.frames:
            continue
        a, b = _ends(model, f)
        n = int(math.ceil(np.linalg.norm(b - a) / lmax - 1e-9))
        if n >= 2:
            out[f] = [k / n for k in range(1, n)]
    return out


def distances(model, frames: Sequence[str], values: Sequence[float], from_end: str = "I",
              relative: bool = False) -> Stations:
    """Cut at the given distances from end I (or J); ``relative`` = fractions of the length."""
    out: Stations = {}
    for f in frames:
        if f not in model.frames:
            continue
        a, b = _ends(model, f)
        L = float(np.linalg.norm(b - a))
        if L <= 0:
            continue
        ss = [(v if relative else v / L) for v in values]
        if str(from_end).upper().startswith("J"):
            ss = [1 - s for s in ss]
        out[f] = _clean(ss, L, model.options.merge_tolerance)
    return out


def parse_numbers(text: str) -> List[float]:
    """'1.5, 3 4.2' or '3@2.0' (three times 2.0, cumulative) -> distances."""
    vals: List[float] = []
    for tok in text.replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "@" in tok:
            k, d = tok.split("@", 1)
            base = vals[-1] if vals else 0.0
            vals += [base + float(d) * (i + 1) for i in range(int(k))]
        else:
            vals.append(float(tok))
    return vals


# ------------------------------------------------------------- geometry helpers
def _station_of_point(a, b, p, tol) -> Optional[float]:
    d = b - a
    L2 = float(d @ d)
    if L2 <= 0:
        return None
    s = float((p - a) @ d) / L2
    if not (0 < s < 1):
        return None
    return s if np.linalg.norm(a + s * d - p) <= tol else None


def _segment_crossing_3d(a, b, c, d, tol) -> Optional[Tuple[float, float]]:
    """Stations (s on a-b, t on c-d) where two segments pass within ``tol``; None if
    they are parallel or miss.  Ends included (0 <= s, t <= 1)."""
    u, v, w = b - a, d - c, a - c
    A, B, C = float(u @ u), float(u @ v), float(v @ v)
    D, E = float(u @ w), float(v @ w)
    den = A * C - B * B
    if A <= 0 or C <= 0 or den <= 1e-12 * A * C:  # parallel (or a point)
        return None
    s = (B * E - C * D) / den
    t = (A * E - B * D) / den
    eps_s, eps_t = tol / math.sqrt(A), tol / math.sqrt(C)
    if not (-eps_s <= s <= 1 + eps_s and -eps_t <= t <= 1 + eps_t):
        return None
    s, t = min(max(s, 0.0), 1.0), min(max(t, 0.0), 1.0)
    if np.linalg.norm((a + s * u) - (c + t * v)) > tol:
        return None
    return s, t


# ------------------------------------------------------------- by model objects
def at_joints(model, frames: Sequence[str], tol: Optional[float] = None,
              joints: Optional[Sequence[str]] = None) -> Stations:
    """Joints lying on the frame (all joints, or only ``joints``)."""
    tol = _tol(model, tol)
    pts = [model.points[n] for n in (joints if joints is not None else model.points) if n in model.points]
    P = np.array([p.xyz for p in pts], float) if pts else np.zeros((0, 3))
    out: Stations = {}
    for f in frames:
        if f not in model.frames or not len(P):
            continue
        a, b = _ends(model, f)
        d = b - a
        L2 = float(d @ d)
        if L2 <= 0:
            continue
        s = (P - a) @ d / L2
        near = np.linalg.norm(a + s[:, None] * d - P, axis=1) <= tol
        ss = s[near & (s > 0) & (s < 1)]
        if ss.size:
            out[f] = _clean(ss, math.sqrt(L2), model.options.merge_tolerance)
    return out


def at_crossing_frames(model, frames: Sequence[str], others: Optional[Sequence[str]] = None,
                       tol: Optional[float] = None, cut_others: bool = False) -> Stations:
    """Where other frames cross (or end on) each selected frame.  ``others``: the
    frames to test against (default: every frame).  With ``cut_others`` the crossing
    frames are cut there too, so both share the new joint."""
    tol = _tol(model, tol)
    sel = [f for f in frames if f in model.frames]
    pool = [f for f in (others if others is not None else model.frames) if f in model.frames]
    ends = {f: _ends(model, f) for f in set(sel) | set(pool)}
    lo = {f: np.minimum(*ends[f]) - tol for f in ends}
    hi = {f: np.maximum(*ends[f]) + tol for f in ends}
    found: Dict[str, List[float]] = {}
    for f in sel:
        a, b = ends[f]
        for g in pool:
            if g == f or np.any(lo[g] > hi[f]) or np.any(hi[g] < lo[f]):
                continue
            c, d = ends[g]
            st = _segment_crossing_3d(a, b, c, d, tol)
            if st is None:
                continue
            found.setdefault(f, []).append(st[0])
            if cut_others:
                found.setdefault(g, []).append(st[1])
    return {f: _clean(ss, float(np.linalg.norm(ends[f][1] - ends[f][0])), model.options.merge_tolerance)
            for f, ss in found.items()}


def at_grid_lines(model, frames: Sequence[str], tol: Optional[float] = None) -> Stations:
    """Where the frame crosses grid lines and arcs, in plan (vertical frames and frames
    lying along a grid line are not cut by it)."""
    from ..model.grids import grid_geometry
    tol = _tol(model, tol)
    gg = grid_geometry(model)
    out: Stations = {}
    for f in frames:
        if f not in model.frames:
            continue
        a, b = _ends(model, f)
        a2, b2 = a[:2], b[:2]
        d = b2 - a2
        Lp2 = float(d @ d)
        if Lp2 <= tol * tol:
            continue
        ss: List[float] = []
        for p, q, _, _ in gg.lines:
            st = _segment_crossing_3d(np.r_[a2, 0.0], np.r_[b2, 0.0], np.r_[p, 0.0], np.r_[q, 0.0], tol)
            if st is not None:
                ss.append(st[0])
        for c, r, a0, a1, _, _ in gg.arcs:
            # |a2 + s d - c| = r
            w = a2 - np.asarray(c, float)
            A, B, C = Lp2, 2 * float(w @ d), float(w @ w) - r * r
            disc = B * B - 4 * A * C
            if disc < 0:
                continue
            for s in ((-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A)):
                if 0 < s < 1:
                    q = a2 + s * d - np.asarray(c, float)
                    ang = math.degrees(math.atan2(q[1], q[0]))
                    lo_, hi_ = min(a0, a1), max(a0, a1)
                    if any(lo_ - 1e-6 <= ang + k * 360 <= hi_ + 1e-6 for k in (-1, 0, 1)):
                        ss.append(s)
        if ss:
            out[f] = _clean(ss, float(np.linalg.norm(b - a)), model.options.merge_tolerance)
    return out


def at_reference_lines(model, frames: Sequence[str], axes: str = "XYZ", tol: Optional[float] = None,
                       joints: Optional[Sequence[str]] = None) -> Stations:
    """Where the frame crosses a reference line: a line parallel to a global axis
    (``axes``) through a joint (every joint, or ``joints``).  A frame along such a
    line is not cut by it."""
    tol = _tol(model, tol)
    P = np.array([model.points[n].xyz for n in (joints if joints is not None else model.points)
                  if n in model.points], float)
    out: Stations = {}
    if not len(P):
        return out
    for f in frames:
        if f not in model.frames:
            continue
        a, b = _ends(model, f)
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= 0:
            continue
        ss: List[float] = []
        for k, ax in enumerate("XYZ"):
            if ax not in axes.upper():
                continue
            o = [i for i in range(3) if i != k]  # the line through joint p along axis k: the other two fixed
            dd = d[o]
            n2 = float(dd @ dd)
            if n2 <= (tol / L) ** 2 * L * L:  # the frame runs along this axis
                continue
            # station where the frame's two other coordinates equal the joint's
            s = ((P[:, o] - a[o]) @ dd) / n2
            ok = (s > 0) & (s < 1)
            res = np.linalg.norm(a[o] + s[:, None] * dd - P[:, o], axis=1)
            ss += s[ok & (res <= tol)].tolist()
        if ss:
            out[f] = _clean(ss, L, model.options.merge_tolerance)
    return out


def at_stories(model, frames: Sequence[str]) -> Stations:
    """Where columns and braces pass story levels."""
    tol = model.options.merge_tolerance
    elev = [s.elevation for s in model.stories_bottom_up()] if hasattr(model, "stories_bottom_up") else []
    out: Stations = {}
    for f in frames:
        if f not in model.frames:
            continue
        a, b = _ends(model, f)
        dz = b[2] - a[2]
        if abs(dz) <= tol:
            continue
        ss = [(z - a[2]) / dz for z in elev]
        out[f] = _clean([s for s in ss if 0 < s < 1], float(np.linalg.norm(b - a)), tol)
    return {f: s for f, s in out.items() if s}


def at_shell_edges(model, frames: Sequence[str], tol: Optional[float] = None,
                   areas: Optional[Sequence[str]] = None) -> Stations:
    """Where the frame crosses the edge of a floor or wall (every shell, or ``areas``)."""
    tol = _tol(model, tol)
    edges = []
    for an in (areas if areas is not None else model.areas):
        a = model.areas.get(an)
        if a is None:
            continue
        P = [np.asarray(model.points[p].xyz, float) for p in a.points]
        edges += [(P[i], P[(i + 1) % len(P)]) for i in range(len(P))]
    out: Stations = {}
    for f in frames:
        if f not in model.frames:
            continue
        a, b = _ends(model, f)
        ss = []
        for c, d in edges:
            st = _segment_crossing_3d(a, b, c, d, tol)
            if st is not None:
                ss.append(st[0])
        if ss:
            out[f] = _clean(ss, float(np.linalg.norm(b - a)), model.options.merge_tolerance)
    return out


def merge(*parts: Stations) -> Stations:
    out: Stations = {}
    for p in parts:
        for f, ss in p.items():
            out.setdefault(f, []).extend(ss)
    return out
