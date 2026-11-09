"""Joint operations shared by every editing command: copying a joint's
assignments, merging one joint into another and re-pointing every object that
refers to a joint.  One implementation, so Copy / Paste / Replicate / Merge /
Move / grip drag cannot each forget a different field (function audit FA05 /
FA06).

Assignment fields of a joint (:class:`~.objects.Point`):
restraints, springs, rotational springs, mass, rotational mass, diaphragm,
panel zone, local axes, loads and ground displacements.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Optional, Tuple

from .base import ModelError
from .objects import FREE, Point

ASSIGNMENT_FIELDS = ("restraints", "springs", "rotational_springs", "mass", "rotational_mass", "diaphragm",
                     "panel_zone", "local_axes", "loads", "ground_displacements")
_ZERO3 = (0.0, 0.0, 0.0)


def has_assignments(p: Point) -> bool:
    return bool(any(p.restraints) or any(p.springs) or any(p.rotational_springs) or any(p.mass)
                or any(p.rotational_mass) or p.diaphragm or p.panel_zone or p.local_axes is not None
                or p.loads or p.ground_displacements)


def _directional(p: Point) -> bool:
    """The joint has values that act in its local axes (restraints, springs, ground displacements)."""
    return bool(any(p.restraints) or any(p.springs) or any(p.rotational_springs) or p.ground_displacements)


# ------------------------------------------------------------------ copying
def clone_assignments(src: Point, dst: Point) -> None:
    """Give ``dst`` (a joint just made by a copy) every assignment of ``src``, as
    assigned: loads, supports and springs keep their global (or the joint's local)
    directions, as CSI Replicate / Paste do."""
    for k in ASSIGNMENT_FIELDS:
        setattr(dst, k, copy.deepcopy(getattr(src, k)))


def turns_directions(p: Point) -> bool:
    """A rotated / mirrored copy of this joint keeps directional values (loads in X / Y,
    supports or springs that differ between X and Y) in the same global directions."""
    if any(any(ld.force[:2]) or any(ld.moment[:2]) for ld in p.loads):
        return True
    r = p.restraints
    return r[0] != r[1] or r[3] != r[4] or p.springs[0] != p.springs[1] or \
        p.rotational_springs[0] != p.rotational_springs[1] or bool(p.ground_displacements)


# ------------------------------------------------------------------ merging
def merge_conflict(keep: Point, gone: Point) -> Optional[str]:
    """Why ``gone`` cannot be merged into ``keep`` without inventing or losing a
    value (None: no conflict).  Identical values are duplicates (kept once); a value
    set on one joint only is taken; two different values are a conflict."""
    for k, what in (("springs", "springs"), ("rotational_springs", "rotational springs"), ("mass", "masses"),
                    ("rotational_mass", "rotational masses")):
        a, b = tuple(getattr(keep, k)), tuple(getattr(gone, k))
        if any(a) and any(b) and any(abs(x - y) > 1e-12 * max(1.0, abs(x), abs(y)) for x, y in zip(a, b)):
            return f"joints {keep.name} and {gone.name} have different {what} ({_fmt(a)} / {_fmt(b)})"
    for k, what in (("diaphragm", "diaphragms"), ("panel_zone", "panel zones")):
        a, b = getattr(keep, k), getattr(gone, k)
        if a and b and a != b:
            return f"joints {keep.name} and {gone.name} belong to different {what} ({a} / {b})"
    la, lb = keep.local_axes, gone.local_axes
    if (la or _ZERO3) != (lb or _ZERO3):
        # different local axes matter when the joint whose axes would be lost acts in them
        if (la is not None and lb is not None) or (la is None and _directional(keep)) or \
                (lb is None and _directional(gone)):
            return f"joints {keep.name} and {gone.name} have restraints / springs in different local axes"
    return None


def _fmt(v) -> str:
    return ", ".join(f"{x:g}" for x in v)


def merge_into(keep: Point, gone: Point) -> None:
    """Assignments of ``gone`` added to ``keep`` (the joint that stays): a DOF
    restrained in either stays restrained; springs, masses, diaphragm, panel zone and
    local axes set on one joint are taken (identical values kept once - call
    :func:`merge_conflict` first for different ones); loads and ground
    displacements of both are kept, identical entries once."""
    keep.restraints = tuple(bool(a or b) for a, b in zip(keep.restraints, gone.restraints))
    for k in ("springs", "rotational_springs", "mass", "rotational_mass"):
        if not any(getattr(keep, k)):
            setattr(keep, k, tuple(getattr(gone, k)))
    keep.diaphragm = keep.diaphragm or gone.diaphragm
    keep.panel_zone = keep.panel_zone or gone.panel_zone
    if keep.local_axes is None and gone.local_axes is not None:
        keep.local_axes = gone.local_axes
    for k in ("loads", "ground_displacements"):
        mine = list(getattr(keep, k))
        for item in getattr(gone, k):
            if item not in mine:
                mine.append(copy.deepcopy(item))
        setattr(keep, k, mine)


def retarget(model, target: Dict[str, str]) -> Dict[str, List[str]]:
    """Re-point every reference to a joint in ``target`` (old -> new) and remove
    the old joints.  Frames that collapse to zero length and areas left with fewer
    than three corners are removed (returned as {"frames": [...], "areas": [...]})."""
    removed: Dict[str, List[str]] = {"frames": [], "areas": [], "line_springs": [], "links": []}
    if not target:
        return removed

    def t(n):
        return target.get(n, n)
    for f in list(model.frames.values()):
        f.point_i, f.point_j = t(f.point_i), t(f.point_j)
        if f.point_i == f.point_j:
            model.frames.pop(f.name)
            removed["frames"].append(f.name)
    for a in list(model.areas.values()):
        pts: List[str] = []
        for p in a.points:
            p = t(p)
            if not pts or pts[-1] != p:
                pts.append(p)
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts.pop()
        if len(set(pts)) < 3:
            model.areas.pop(a.name)
            removed["areas"].append(a.name)
        else:
            a.points = pts
    for s in list(getattr(model, "line_springs", {}).values()):
        s.point_i, s.point_j = t(s.point_i), t(s.point_j)
        if s.point_i == s.point_j:
            model.line_springs.pop(s.name)
            removed["line_springs"].append(s.name)
    for lk in list(getattr(model, "links", {}).values()):
        lk.point_i = t(lk.point_i)
        if lk.point_j is not None:
            lk.point_j = t(lk.point_j)
            if lk.point_j == lk.point_i:
                model.links.pop(lk.name)
                removed["links"].append(lk.name)
    for g in model.groups.values():
        out: List[str] = []
        for p in g.points:
            p = t(p)
            if p not in out:
                out.append(p)
        g.points = out
    for c in getattr(model, "load_cases", {}).values():
        if getattr(c, "monitored_point", None) in target:
            c.monitored_point = target[c.monitored_point]
        mp = getattr(c, "monitor_points", None)
        if mp:
            c.monitor_points = list(dict.fromkeys(t(p) for p in mp))
    for n in target:
        model.points.pop(n, None)
    return removed


def merge_points(model, target: Dict[str, str]) -> Dict[str, List[str]]:
    """Merge each joint ``old`` into ``target[old]`` (assignments combined, every
    reference re-pointed).  Raises ModelError before changing anything when two
    joints cannot be combined (see :func:`merge_conflict`)."""
    # follow chains a -> b -> c so every joint ends at a joint that stays
    final: Dict[str, str] = {}
    for n in target:
        k, seen = target[n], {n}
        while k in target and k not in seen:
            seen.add(k)
            k = target[k]
        if k != n:
            final[n] = k
    # preflight on copies: the model is untouched when anything conflicts
    acc: Dict[str, Point] = {}
    for n, k in final.items():
        keep = acc.setdefault(k, copy.deepcopy(model.points[k]))
        why = merge_conflict(keep, model.points[n])
        if why:
            raise ModelError("MERGE_CONFLICT", f"Cannot merge: {why}. Change one of them first.")
        merge_into(keep, model.points[n])
    for k, merged in acc.items():
        q = model.points[k]
        for f in ASSIGNMENT_FIELDS:
            setattr(q, f, getattr(merged, f))
    return retarget(model, final)


def coincident_groups(points: Iterable[Point], tol: float) -> List[List[str]]:
    """Joints closer than ``tol`` to each other (each group has 2+ names)."""
    cell = max(tol * 4, 1e-9)
    grid: Dict[Tuple[int, int, int], List[Point]] = {}
    pts = list(points)
    for p in pts:
        grid.setdefault((math.floor(p.x / cell), math.floor(p.y / cell), math.floor(p.z / cell)), []).append(p)
    parent = {p.name: p.name for p in pts}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for p in pts:
        k = (math.floor(p.x / cell), math.floor(p.y / cell), math.floor(p.z / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if q.name != p.name and math.dist(p.xyz, q.xyz) <= tol:
                            ra, rb = root(p.name), root(q.name)
                            if ra != rb:
                                parent[rb] = ra
    groups: Dict[str, List[str]] = {}
    for p in pts:
        groups.setdefault(root(p.name), []).append(p.name)
    return [g for g in groups.values() if len(g) > 1]


__all__ = ["ASSIGNMENT_FIELDS", "clone_assignments", "turns_directions", "merge_conflict", "merge_into", "merge_points",
           "retarget", "has_assignments", "coincident_groups", "FREE"]
