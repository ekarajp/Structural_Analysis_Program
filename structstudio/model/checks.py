"""Model check (Analyze > Check Model): geometry problems the analysis would not
stop on but that change the results - members drawn twice, overlapping
members, joints that should be one, shells or frames with no size.

:func:`geometry_issues` checks the whole model; :func:`duplicates_of` checks
only the given frames / areas (fast: used after every drawing / copy / move).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .point_ops import coincident_groups


@dataclass
class GeometryIssue:
    severity: str  # error | warning | info
    code: str
    message: str
    frames: List[str] = field(default_factory=list)
    areas: List[str] = field(default_factory=list)
    points: List[str] = field(default_factory=list)


def _xyz(model, n) -> np.ndarray:
    return np.array(model.points[n].xyz, float)


def area_size(model, a) -> float:
    pts = [_xyz(model, n) for n in a.points if n in model.points]
    if len(pts) < 3:
        return 0.0
    v = np.zeros(3)
    for k in range(len(pts)):
        v += np.cross(pts[k], pts[(k + 1) % len(pts)])
    return float(np.linalg.norm(v) / 2)


def _line_key(a: np.ndarray, b: np.ndarray, tol: float):
    """A key shared by every segment on the same infinite line (rounded)."""
    d = b - a
    n = np.linalg.norm(d)
    if n <= tol:
        return None
    d = d / n
    # canonical direction: first non-zero component positive
    for c in d:
        if abs(c) > 1e-9:
            if c < 0:
                d = -d
            break
    foot = a - np.dot(a, d) * d  # closest point of the line to the origin
    q = max(tol, 1e-9) * 10
    return (tuple(np.round(d, 6)), tuple(np.round(foot / q).astype(int)))


def _frame_index(model, tol: float):
    pairs: Dict[frozenset, List[str]] = {}
    lines: Dict[tuple, List[Tuple[str, float, float]]] = {}
    for f in model.frames.values():
        if f.point_i not in model.points or f.point_j not in model.points:
            continue
        pairs.setdefault(frozenset((f.point_i, f.point_j)), []).append(f.name)
        a, b = _xyz(model, f.point_i), _xyz(model, f.point_j)
        key = _line_key(a, b, tol)
        if key is None:
            continue
        d = np.array(key[0], float)
        t0, t1 = sorted((float(np.dot(a, d)), float(np.dot(b, d))))
        lines.setdefault(key, []).append((f.name, t0, t1))
    return pairs, lines


def _overlaps(seg: List[Tuple[str, float, float]], tol: float) -> List[Tuple[str, str, float]]:
    """(f, g, overlap length) of segments on one line that overlap by more than tol."""
    out = []
    seg = sorted(seg, key=lambda s: s[1])
    active: List[Tuple[str, float, float]] = []
    for s in seg:
        active = [a for a in active if a[2] > s[1] + tol]
        for a in active:
            ov = min(a[2], s[2]) - s[1]
            if ov > tol:
                out.append((a[0], s[0], ov))
        active.append(s)
    return out


def duplicates_of(model, frames: Iterable[str] = (), areas: Iterable[str] = ()) -> List[GeometryIssue]:
    """Duplicate / overlapping members and duplicate shells involving the given ones."""
    tol = max(model.options.merge_tolerance, 1e-9)
    frames = [n for n in frames if n in model.frames]
    areas = [n for n in areas if n in model.areas]
    out: List[GeometryIssue] = []
    if frames:
        pairs, lines = _frame_index(model, tol)
        mine = set(frames)
        seen: Set[frozenset] = set()
        for n in frames:
            f = model.frames[n]
            others = [o for o in pairs.get(frozenset((f.point_i, f.point_j)), []) if o != n]
            if others:
                key = frozenset([n] + others)
                if key not in seen:
                    seen.add(key)
                    out.append(GeometryIssue("warning", "DUPLICATE_FRAME",
                                             f"{n} lies exactly on {', '.join(others)} (same two joints): a member "
                                             "drawn twice doubles its stiffness and weight", [n] + others))
        for key, seg in lines.items():
            if not any(s[0] in mine for s in seg) or len(seg) < 2:
                continue
            for a, b, ov in _overlaps(seg, tol):
                if (a in mine or b in mine) and frozenset((a, b)) not in seen:
                    fa, fb = model.frames[a], model.frames[b]
                    if {fa.point_i, fa.point_j} == {fb.point_i, fb.point_j}:
                        continue
                    seen.add(frozenset((a, b)))
                    out.append(GeometryIssue("warning", "OVERLAPPING_FRAMES",
                                             f"{a} and {b} overlap along {ov:.4g} (same line): part of the member "
                                             "is there twice", [a, b]))
    if areas:
        sets: Dict[frozenset, List[str]] = {}
        for a in model.areas.values():
            sets.setdefault(frozenset(a.points), []).append(a.name)
        seen_a: Set[frozenset] = set()
        for n in areas:
            others = [o for o in sets.get(frozenset(model.areas[n].points), []) if o != n]
            key = frozenset([n] + others)
            if others and key not in seen_a:
                seen_a.add(key)
                out.append(GeometryIssue("warning", "DUPLICATE_AREA",
                                         f"{n} lies exactly on {', '.join(others)} (same corners): a slab / wall "
                                         "drawn twice doubles its stiffness and weight", areas=[n] + others))
    return out


def geometry_issues(model) -> List[GeometryIssue]:
    """Every geometry problem of the model (Check Model)."""
    tol = max(model.options.merge_tolerance, 1e-9)
    out: List[GeometryIssue] = []
    for f in model.frames.values():
        if f.point_i in model.points and f.point_j in model.points and \
                math.dist(model.points[f.point_i].xyz, model.points[f.point_j].xyz) <= tol:
            out.append(GeometryIssue("error", "ZERO_LENGTH_FRAME", f"{f.name} has no length (its two ends coincide)",
                                     [f.name]))
    for a in model.areas.values():
        if len(set(a.points)) != len(a.points):
            out.append(GeometryIssue("error", "REPEATED_CORNER", f"{a.name} uses a joint twice", areas=[a.name]))
        elif area_size(model, a) <= tol * tol * 10:
            out.append(GeometryIssue("error", "ZERO_AREA", f"{a.name} has no area (its corners are on one line)",
                                     areas=[a.name]))
    out += duplicates_of(model, list(model.frames), list(model.areas))
    for grp in coincident_groups(model.points.values(), tol):
        out.append(GeometryIssue("warning", "COINCIDENT_JOINTS",
                                 f"joints {', '.join(sorted(grp))} are at the same place but not connected "
                                 "(Edit > Merge Joints)", points=sorted(grp)))
    used = set()
    for f in model.frames.values():
        used |= {f.point_i, f.point_j}
    for a in model.areas.values():
        used |= set(a.points)
    for s in getattr(model, "line_springs", {}).values():
        used |= {s.point_i, s.point_j}
    for lk in getattr(model, "links", {}).values():
        used |= {lk.point_i, lk.point_j}
    from .point_ops import has_assignments
    lonely = [n for n, p in model.points.items() if n not in used and not has_assignments(p)]
    if lonely:
        out.append(GeometryIssue("info", "UNUSED_JOINTS",
                                 f"{len(lonely)} joint(s) are used by nothing and carry nothing "
                                 f"({', '.join(lonely[:8])}{'...' if len(lonely) > 8 else ''})", points=lonely))
    return out


__all__ = ["GeometryIssue", "geometry_issues", "duplicates_of", "area_size"]
