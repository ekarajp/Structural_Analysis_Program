"""Frame load and end-condition mapping for the commands that change a frame's
ends: Divide, Join and Reverse Connectivity (function audit FA07 - FA09).

Every load is first placed at its physical distance from end I (metres along
the member, whatever "relative" / "absolute" it was given in), then written for
the new member(s) in the same convention it had.  So the resultant force and its
moment about any point stay the same.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import List, Optional, Tuple

from .objects import Frame, FrameLoad

EPS = 1e-9
# local axes after I and J are swapped: 1 and 3 reverse, 2 stays (CSI local-axis rule)
_FLIP_SIGN = {"1": -1.0, "3": -1.0}
_CARDINAL_MIRROR = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7}


def span(ld: FrameLoad, L: float) -> Tuple[float, float]:
    """(start, end) of a load in length units from end I."""
    if ld.kind in ("temperature", "gravity"):
        return 0.0, L
    a, b = (ld.rel_a, ld.rel_b) if ld.absolute else (ld.rel_a * L, ld.rel_b * L)
    if ld.kind == "point":
        return a, a
    return a, b


def _put(ld: FrameLoad, xa: float, xb: float, va: float, vb: float, L: float) -> FrameLoad:
    """``ld`` at [xa, xb] (length from end I of a member of length L) in its own convention."""
    out = dataclasses.replace(ld)
    if ld.absolute:
        out.rel_a, out.rel_b = xa, xb
    else:
        out.rel_a, out.rel_b = (xa / L, xb / L) if L > EPS else (0.0, 1.0)
    out.value_a, out.value_b = va, vb
    if ld.kind == "point":
        out.rel_b, out.value_b = out.rel_a, out.value_a
    return out


def reversed_load(ld: FrameLoad, L: float) -> FrameLoad:
    """The same physical load on the member with I and J swapped."""
    xa, xb = span(ld, L)
    out = dataclasses.replace(ld)
    if ld.kind in ("temperature",):
        out.value_a, out.value_b = ld.value_b, ld.value_a
        if ld.load_type == "Gradient3":
            out.value_a, out.value_b = -out.value_a, -out.value_b
        return out
    if ld.kind == "gravity":
        return out
    sign = _FLIP_SIGN.get(str(ld.direction), 1.0)
    if ld.kind == "point":
        return _put(ld, L - xa, L - xa, sign * ld.value_a, sign * ld.value_a, L)
    return _put(ld, L - xb, L - xa, sign * ld.value_b, sign * ld.value_a, L)


def reverse_frame(f: Frame, L: float) -> None:
    """Swap ends I and J of ``f`` (points, end conditions, loads, hinges, local axes)."""
    f.point_i, f.point_j = f.point_j, f.point_i
    f.releases_i, f.releases_j = f.releases_j, f.releases_i
    f.fixity_i_trans, f.fixity_j_trans = f.fixity_j_trans, f.fixity_i_trans
    f.fixity_i_rot, f.fixity_j_rot = f.fixity_j_rot, f.fixity_i_rot
    f.end_offset_i, f.end_offset_j = f.end_offset_j, f.end_offset_i
    f.joint_offset_i, f.joint_offset_j = f.joint_offset_j, f.joint_offset_i
    if f.joint_offset_local:  # local 1 and 3 reverse
        f.joint_offset_i = (-f.joint_offset_i[0], f.joint_offset_i[1], -f.joint_offset_i[2])
        f.joint_offset_j = (-f.joint_offset_j[0], f.joint_offset_j[1], -f.joint_offset_j[2])
    f.angle = -f.angle if f.angle else f.angle  # a rotation about a reversed axis 1
    f.cardinal_point = _CARDINAL_MIRROR.get(f.cardinal_point, f.cardinal_point)  # left / right swap with axis 3
    if f.display_cardinal is not None:
        f.display_cardinal = _CARDINAL_MIRROR.get(f.display_cardinal, f.display_cardinal)
    f.loads = [reversed_load(ld, L) for ld in f.loads]
    for h in f.hinges:
        if hasattr(h, "relative_distance"):
            h.relative_distance = 1.0 - h.relative_distance


def piece_loads(loads: List[FrameLoad], L: float, x0: float, x1: float, last: bool) -> List[FrameLoad]:
    """Loads of a member of length L that act on its part [x0, x1], written for a
    new member that runs from x0 to x1 (same direction).  A point load exactly at
    an inner division goes to the piece that starts there."""
    out: List[FrameLoad] = []
    Lp = x1 - x0
    for ld in loads:
        xa, xb = span(ld, L)
        if ld.kind == "gravity":
            out.append(dataclasses.replace(ld))
        elif ld.kind == "temperature":
            def at(x):
                return ld.value_a + (ld.value_b - ld.value_a) * (x / L if L > EPS else 0.0)
            q = dataclasses.replace(ld)
            q.value_a, q.value_b = at(x0), at(x1)
            out.append(q)
        elif ld.kind == "point":
            if x0 - EPS <= xa < x1 - EPS or (last and abs(xa - x1) <= EPS):
                out.append(_put(ld, xa - x0, xa - x0, ld.value_a, ld.value_a, Lp))
        else:
            a, b = max(xa, x0), min(xb, x1)
            if b - a <= EPS:
                continue

            def v(x):
                return ld.value_a if xb - xa <= EPS else ld.value_a + (ld.value_b - ld.value_a) * (x - xa) / (xb - xa)
            out.append(_put(ld, a - x0, b - x0, v(a), v(b), Lp))
    return out


def joined_loads(parts: List[Tuple[Frame, float, float, bool]], L: float) -> List[FrameLoad]:
    """Loads of frames joined into one of length L.  ``parts``: (frame, its length,
    distance of its start along the joined member, True when it runs the other way)."""
    out: List[FrameLoad] = []
    for f, Lf, off, flipped in parts:
        for ld in f.loads:
            if ld.kind == "gravity":
                if not any(o.kind == "gravity" and o.pattern == ld.pattern and o.value_a == ld.value_a for o in out):
                    out.append(dataclasses.replace(ld))
                continue
            q = reversed_load(ld, Lf) if flipped else ld
            xa, xb = span(q, Lf)
            if q.kind == "temperature":
                raise ValueError("temperature")
            out.append(_put(q, off + xa, off + xb, q.value_a, q.value_b, L))
    return out


def resultant(loads: List[FrameLoad], L: float, direction: str = "Gravity") -> Tuple[float, float]:
    """(total force, its moment about end I) of the force loads in ``direction``."""
    F = M = 0.0
    for ld in loads:
        if ld.direction != direction or ld.load_type != "Force" or ld.kind not in ("point", "distributed"):
            continue
        xa, xb = span(ld, L)
        if ld.kind == "point":
            F += ld.value_a
            M += ld.value_a * xa
        else:
            w = xb - xa
            f = (ld.value_a + ld.value_b) / 2 * w
            F += f
            M += f * xa + w * w * (ld.value_a + 2 * ld.value_b) / 6
    return F, M


def copy_frame_loads(loads: List[FrameLoad]) -> List[FrameLoad]:
    return [copy.deepcopy(ld) for ld in loads]


__all__ = ["span", "reversed_load", "reverse_frame", "piece_loads", "joined_loads", "resultant"]


# ---------------------------------------------------------- turning with a copy
_GLOBAL_PLAN = {"X": (1.0, 0.0), "Y": (0.0, 1.0), "XProjected": (1.0, 0.0), "YProjected": (0.0, 1.0)}


def _turned_direction(direction: str, A):
    """[(new direction, factor)] of a load in global X / Y turned with the plan map A."""
    import numpy as np
    proj = "Projected" if direction.endswith("Projected") else ""
    d = np.asarray(A, float) @ np.array(_GLOBAL_PLAN[direction])
    out = []
    for name, comp in (("X", d[0]), ("Y", d[1])):
        if abs(comp) > 1e-9:
            out.append((name + proj, float(comp)))
    return out


def turn_loads(loads, A, kind: str = "frame"):
    """Loads of a copied / moved frame or shell turned with the plan map A: global
    X / Y loads turn (a load between the axes becomes its X and Y parts); with a
    mirror, local loads across the member (axis 3 forces, axis 1 / 2 moments,
    gradients across 3) change sign because axis 3 turns over."""
    import numpy as np
    A = np.asarray(A, float)
    det = float(np.linalg.det(A))
    out = []
    for ld in loads:
        direction = str(getattr(ld, "direction", "Gravity"))
        moment = getattr(ld, "load_type", "Force") == "Moment"
        if direction in _GLOBAL_PLAN:
            parts = _turned_direction(direction, A)
            for name, f in parts:
                q = dataclasses.replace(ld, direction=name)
                f = f * (det if moment else 1.0)
                if kind == "frame":
                    q.value_a, q.value_b = ld.value_a * f, ld.value_b * f
                else:
                    q.value = ld.value * f
                out.append(q)
            continue
        q = dataclasses.replace(ld)
        if det < 0:
            flip = (direction in ("1", "2")) if moment else (direction == "3")
            if direction == "Z" and moment:
                flip = True
            if getattr(ld, "load_type", "") == "Gradient3":
                flip = True
            if flip:
                if kind == "frame":
                    q.value_a, q.value_b = -ld.value_a, -ld.value_b
                elif kind == "area" and direction != "3":  # area pressure follows the reversed corners
                    q.value = -ld.value
        out.append(q)
    return out


def mirror_frame_axes(f) -> None:
    """A mirrored copy of a frame: the rotation about axis 1 and the left / right
    cardinal points turn over (axis 3 reverses)."""
    if f.angle:
        f.angle = -f.angle
    f.cardinal_point = _CARDINAL_MIRROR.get(f.cardinal_point, f.cardinal_point)
    if f.display_cardinal is not None:
        f.display_cardinal = _CARDINAL_MIRROR.get(f.display_cardinal, f.display_cardinal)


def turn_frame(f, A, vertical: bool) -> None:
    """A frame copied / moved with the plan map A (see point_ops.plan_map): loads,
    global joint offsets and - for columns, whose local axes start from global X -
    the axis angle turn with it; a mirror also turns the section over."""
    import math
    import numpy as np
    A = np.asarray(A, float)
    if np.allclose(A, np.eye(2)):
        return
    det = float(np.linalg.det(A))
    f.loads = turn_loads(f.loads, A, "frame")
    if not f.joint_offset_local:
        for k in ("joint_offset_i", "joint_offset_j"):
            v = getattr(f, k)
            x, y = A @ np.array(v[:2], float)
            setattr(f, k, (float(x), float(y), v[2]))
    if vertical:
        a = math.radians(f.angle or 0.0)
        d = A @ np.array([math.cos(a), math.sin(a)])
        na = math.degrees(math.atan2(d[1], d[0]))
        f.angle = 0.0 if abs(na) < 1e-9 else na
        if det < 0:
            f.cardinal_point = _CARDINAL_MIRROR.get(f.cardinal_point, f.cardinal_point)
            if f.display_cardinal is not None:
                f.display_cardinal = _CARDINAL_MIRROR.get(f.display_cardinal, f.display_cardinal)
    elif det < 0:
        mirror_frame_axes(f)
