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
from dataclasses import dataclass
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
# what conflicts, as groups the user chooses between (Merge Joints dialog)
CONFLICT_GROUPS = {
    "axes": ("Local axes, supports and springs", ("local_axes", "restraints", "springs", "rotational_springs")),
    "springs": ("Springs", ("springs",)),
    "rotational_springs": ("Rotational springs", ("rotational_springs",)),
    "mass": ("Mass", ("mass",)),
    "rotational_mass": ("Rotational mass", ("rotational_mass",)),
    "diaphragm": ("Diaphragm", ("diaphragm",)),
    "panel_zone": ("Panel zone", ("panel_zone",)),
}


@dataclass
class MergeConflictItem:
    keep: str  # the joint that stays
    gone: str  # the joint merged into it
    group: str  # key of CONFLICT_GROUPS
    keep_value: str
    gone_value: str

    @property
    def label(self) -> str:
        return CONFLICT_GROUPS[self.group][0]


class MergeConflict(ModelError):
    """Two joints to be merged have different values; ``items`` says which, so the
    user can choose (MergeChoices) and the command can run again."""

    def __init__(self, items: List[MergeConflictItem]):
        why = "; ".join(f"{i.keep} / {i.gone}: different {i.label.lower()}" for i in items[:4])
        super().__init__("MERGE_CONFLICT", f"Cannot merge without choosing: {why}")
        self.items = items


# choices: (joint that stays, joint merged, group) -> the name of the joint whose value is kept
MergeChoices = Dict[Tuple[str, str, str], str]
# set by the window: called with the conflicts, returns the choices or None (cancel)
CONFLICT_RESOLVER = None


def _fmt(v) -> str:
    if v is None:
        return "none (global)"
    if isinstance(v, (tuple, list)):
        if v and isinstance(v[0], bool):
            return "".join("R" if x else "-" for x in v) if any(v) else "free"
        return ", ".join(f"{x:g}" for x in v)
    return str(v)


def _differ(a, b) -> bool:
    if isinstance(a, tuple) or isinstance(b, tuple):
        a, b = tuple(a or ()), tuple(b or ())
        return len(a) != len(b) or any(abs(float(x) - float(y)) > 1e-12 * max(1.0, abs(float(x)), abs(float(y)))
                                       for x, y in zip(a, b))
    return a != b


def merge_conflicts(keep: Point, gone: Point) -> List[MergeConflictItem]:
    """What cannot be merged without choosing: identical values are duplicates (kept
    once); a value set on one joint only is taken; two different values conflict."""
    out: List[MergeConflictItem] = []
    la, lb = keep.local_axes, gone.local_axes
    axes = (la or _ZERO3) != (lb or _ZERO3) and (
        (la is not None and lb is not None) or (la is None and _directional(keep)) or
        (lb is None and _directional(gone)))
    if axes:
        def show(p):
            return (f"axes {_fmt(p.local_axes)}; supports {_fmt(p.restraints)}; springs {_fmt(p.springs)}")
        out.append(MergeConflictItem(keep.name, gone.name, "axes", show(keep), show(gone)))
    for k in ("springs", "rotational_springs", "mass", "rotational_mass"):
        if axes and k in ("springs", "rotational_springs"):
            continue  # chosen with the axes
        a, b = tuple(getattr(keep, k)), tuple(getattr(gone, k))
        if any(a) and any(b) and _differ(a, b):
            out.append(MergeConflictItem(keep.name, gone.name, k, _fmt(a), _fmt(b)))
    for k in ("diaphragm", "panel_zone"):
        a, b = getattr(keep, k), getattr(gone, k)
        if a and b and a != b:
            out.append(MergeConflictItem(keep.name, gone.name, k, str(a), str(b)))
    return out


def merge_conflict(keep: Point, gone: Point) -> Optional[str]:
    """Text of the first conflict (None: none) - see :func:`merge_conflicts`."""
    items = merge_conflicts(keep, gone)
    return f"joints {keep.name} and {gone.name}: different {items[0].label.lower()}" if items else None


def merge_into(keep: Point, gone: Point, choose: Optional[Dict[str, bool]] = None) -> None:
    """Assignments of ``gone`` added to ``keep`` (the joint that stays): a DOF
    restrained in either stays restrained; springs, masses, diaphragm, panel zone and
    local axes set on one joint are taken (identical values kept once); loads and
    ground displacements of both are kept, identical entries once.  ``choose``:
    conflict group -> True to take ``gone``'s value (from the user's choice)."""
    choose = choose or {}
    take_axes = choose.get("axes")
    if take_axes is not None:  # the whole directional group of one joint
        src = gone if take_axes else keep
        keep.local_axes, keep.restraints = src.local_axes, tuple(src.restraints)
        keep.springs, keep.rotational_springs = tuple(src.springs), tuple(src.rotational_springs)
    else:
        keep.restraints = tuple(bool(a or b) for a, b in zip(keep.restraints, gone.restraints))
    for k in ("springs", "rotational_springs", "mass", "rotational_mass"):
        if take_axes is not None and k in ("springs", "rotational_springs"):
            continue
        if choose.get(k) or not any(getattr(keep, k)):
            if any(getattr(gone, k)) or choose.get(k):
                setattr(keep, k, tuple(getattr(gone, k)))
    for k in ("diaphragm", "panel_zone"):
        if choose.get(k):
            setattr(keep, k, getattr(gone, k))
        else:
            setattr(keep, k, getattr(keep, k) or getattr(gone, k))
    if take_axes is None and keep.local_axes is None and gone.local_axes is not None:
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


def _final_targets(target: Dict[str, str]) -> Dict[str, str]:
    final: Dict[str, str] = {}
    for n in target:  # follow chains a -> b -> c so every joint ends at a joint that stays
        k, seen = target[n], {n}
        while k in target and k not in seen:
            seen.add(k)
            k = target[k]
        if k != n:
            final[n] = k
    return final


def _preflight(model, final: Dict[str, str], choices: MergeChoices):
    acc: Dict[str, Point] = {}
    open_items: List[MergeConflictItem] = []
    for n, k in final.items():
        keep = acc.setdefault(k, copy.deepcopy(model.points[k]))
        gone = model.points[n]
        items = merge_conflicts(keep, gone)
        todo = [i for i in items if (k, n, i.group) not in choices]
        open_items += todo
        if not todo:
            merge_into(keep, gone, {i.group: choices[(k, n, i.group)] == n for i in items})
    return acc, open_items


def resolve_merge(model, target: Dict[str, str], choices: Optional[MergeChoices] = None) -> MergeChoices:
    """The choices a merge needs, asked from CONFLICT_RESOLVER (the window's dialog)
    when two joints have different values.  Changes nothing.  Raises MergeConflict
    without a resolver, ModelError("MERGE_CANCELLED") when the user cancels."""
    final = _final_targets(target)
    choices = dict(choices or {})
    _, open_items = _preflight(model, final, choices)
    if open_items:
        resolver = CONFLICT_RESOLVER
        picked = resolver(open_items) if resolver is not None else None
        if picked is None:
            if resolver is not None:
                raise ModelError("MERGE_CANCELLED", "Merge cancelled: nothing was changed")
            raise MergeConflict(open_items)
        choices.update(picked)
        _, open_items = _preflight(model, final, choices)
        if open_items:
            raise MergeConflict(open_items)
    return choices


def merge_points(model, target: Dict[str, str], choices: Optional[MergeChoices] = None) -> Dict[str, List[str]]:
    """Merge each joint ``old`` into ``target[old]`` (assignments combined, every
    reference re-pointed).  Different values on the two joints need a choice
    (``choices``, else asked through :func:`resolve_merge`); nothing changes when
    none is given."""
    final = _final_targets(target)
    choices = resolve_merge(model, target, choices)
    acc, _ = _preflight(model, final, choices)
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


__all__ = ["ASSIGNMENT_FIELDS", "clone_assignments", "turns_directions", "merge_conflict", "merge_conflicts",
           "MergeConflict", "MergeConflictItem", "CONFLICT_GROUPS", "merge_into", "merge_points", "resolve_merge",
           "retarget", "has_assignments", "coincident_groups", "FREE"]


# ---------------------------------------------------------- turning with a copy
def plan_map(T):
    """The 2x2 plan part of a transform (xyz -> xyz) that keeps vertical lines
    vertical and up up (rotation about Z, mirror in a vertical plane, translation);
    None for any other transform (rotation in an elevation ...)."""
    import numpy as np
    o = np.asarray(T(np.zeros(3)), float)
    R = np.column_stack([np.asarray(T(e), float) - o for e in np.eye(3)])
    if abs(R[2, 2] - 1.0) > 1e-9 or np.abs(R[2, :2]).max() > 1e-9 or np.abs(R[:2, 2]).max() > 1e-9:
        return None
    A = R[:2, :2]
    if not np.allclose(A @ A.T, np.eye(2), atol=1e-9):
        return None
    return A


def _anisotropic(p: Point) -> bool:
    r = p.restraints
    return (r[0] != r[1] or r[3] != r[4] or abs(p.springs[0] - p.springs[1]) > 0 or
            abs(p.rotational_springs[0] - p.rotational_springs[1]) > 0 or
            any(g.coord_sys != "Global" for g in p.ground_displacements))


def turn_assignments(p: Point, A) -> bool:
    """Turn a joint's directional assignments with the plan map A (from
    :func:`plan_map`): loads, global ground displacements and X / Y masses turn,
    supports and springs keep acting along the turned directions (through the
    joint's local axes).  False when the joint's local axes are tilted (b, c not 0)
    and could not be turned (the values are then kept as assigned)."""
    import numpy as np
    A = np.asarray(A, float)
    if np.allclose(A, np.eye(2)):
        return True
    det = float(np.linalg.det(A))

    def clean(x, scale):
        return 0.0 if abs(x) <= 1e-12 * max(scale, 1e-300) else float(x)

    def vec(v):
        x, y = A @ np.array(v[:2], float)
        sc = max(abs(v[0]), abs(v[1]))
        return (clean(x, sc), clean(y, sc), float(v[2]))

    def pseudo(v):
        x, y = det * (A @ np.array(v[:2], float))
        sc = max(abs(v[0]), abs(v[1]))
        return (clean(x, sc), clean(y, sc), float(det * v[2]) + 0.0)
    p.loads = [type(ld)(ld.pattern, vec(ld.force), pseudo(ld.moment)) for ld in p.loads]
    gd = []
    for g in p.ground_displacements:
        g = copy.deepcopy(g)
        if g.coord_sys == "Global":
            g.translations, g.rotations = vec(g.translations), pseudo(g.rotations)
        elif det < 0:  # local axis 2 turns over with a mirror
            t, r = g.translations, g.rotations
            g.translations, g.rotations = (t[0], -t[1], t[2]), (-r[0], r[1], -r[2])
        gd.append(g)
    p.ground_displacements = gd
    for k in ("mass", "rotational_mass"):
        mx, my, mz = getattr(p, k)
        if mx != my:
            M = A @ np.diag([mx, my]) @ A.T
            setattr(p, k, (float(M[0, 0]), float(M[1, 1]), mz))
    ok = True
    if p.local_axes is not None or _anisotropic(p):
        a, b, c = p.local_axes or (0.0, 0.0, 0.0)
        if abs(b) > 1e-9 or abs(c) > 1e-9:
            ok = False
        else:
            d = A @ np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
            na = math.degrees(math.atan2(d[1], d[0]))
            na = 0.0 if abs(na) < 1e-9 else na
            p.local_axes = (na, 0.0, 0.0) if (na or p.local_axes is not None) else None
    return ok
