"""CAD-style modify commands (AutoCAD / Revit / ArchiCAD workflow) for the model windows.

Every command works on the selection; with nothing selected it first asks for
objects (click, window or crossing box, then Enter / right-click), the way the
CAD programs do.  Points are picked with the full snapping of the window
(object snaps, polar / ortho / alignment tracking) and every distance or angle
can be typed instead of clicked (see :func:`parse_typed`).

=============  ======================  ==============================================
command        Revit / CAD shortcut    clicks
=============  ======================  ==============================================
Move           MV                      base point, destination
Copy           CO                      base point, destination(s)  (multiple)
Rotate         RO                      centre, start direction, end direction / angle
Mirror         MM                      two points of the mirror axis
Paste          Ctrl+V                  insertion point (copied objects follow the cursor)
Trim           TR                      the part of a frame to cut away (quick trim)
Extend         EX                      a frame near the end to lengthen
Offset         OF                      a frame, on the side of the copy
=============  ======================  ==============================================

Model edits are single undo steps (``Document.edit``).  Moved objects detach
from unselected objects by default (CAD behaviour: a moved beam leaves its
columns where they are); *Stretch connected objects* keeps the shared joints
(ETABS *Move*).  Points that land on an existing joint are merged with it, so
moved / copied members connect to the structure they are placed on.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..model import Model, ModelError
from .commands import (_area_kwargs, _frame_kwargs, _restory, _sel_points, _unused_points, paste_objects,
                       replicate_transform)
from .document import Document
from .tools import next_step, TOOL_PROPERTIES, TOOLS, Plane, Preview, Tool, grid_ordinates, story_levels

XYZ = Tuple[float, float, float]
Transform = Callable[[np.ndarray], np.ndarray]


# ----------------------------------------------------------------- transforms
def translation(d) -> Transform:
    d = np.asarray(d, float)
    return lambda p: np.asarray(p, float) + d


def rotation(center, axis, angle_deg: float) -> Transform:
    c = np.asarray(center, float)
    k = np.asarray(axis, float)
    k = k / (np.linalg.norm(k) or 1.0)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)

    def T(p):
        v = np.asarray(p, float) - c
        return c + v * ca + np.cross(k, v) * sa + k * float(np.dot(k, v)) * (1 - ca)
    return T


def mirror(a, b, normal) -> Transform:
    """Reflection in the plane through the line a-b that contains ``normal``
    (the view direction: in a plan the mirror plane is vertical)."""
    a = np.asarray(a, float)
    m = np.cross(np.asarray(b, float) - a, np.asarray(normal, float))
    n = np.linalg.norm(m)
    if n < 1e-12:
        raise ModelError("BAD_AXIS", "The two points of the mirror axis coincide")
    m = m / n
    return lambda p: np.asarray(p, float) - 2.0 * float(np.dot(np.asarray(p, float) - a, m)) * m


def plane_normal(plane: Plane) -> np.ndarray:
    return np.array({"plan": (0, 0, 1), "xz": (0, 1, 0), "yz": (1, 0, 0)}.get(plane.kind, (0, 0, 1)), float)


# ------------------------------------------------------------ in-place edits
def _merge_points(model: Model, moved: Sequence[str], tol: float) -> int:
    """Merge every moved point lying on another (not moved) joint into that joint:
    assignments combined and every reference re-pointed (model.point_ops)."""
    from ..model.point_ops import merge_points
    moved = [n for n in moved if n in model.points]
    mset = set(moved)
    others = [p for n, p in model.points.items() if n not in mset]
    if not others or not moved:
        return 0
    cell = max(tol * 10, 1e-6)
    grid: Dict[Tuple[int, int, int], list] = {}
    for p in others:
        grid.setdefault((round(p.x / cell), round(p.y / cell), round(p.z / cell)), []).append(p)
    target: Dict[str, str] = {}
    for n in moved:
        p = model.points[n]
        k = (round(p.x / cell), round(p.y / cell), round(p.z / cell))
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if math.dist(p.xyz, q.xyz) <= tol:
                            best = q
        if best is not None:
            target[n] = best.name
    merge_points(model, target)  # raises MERGE_CONFLICT before changing anything
    return len(target)


def transform_in_place(doc: Document, frames: Sequence[str], areas: Sequence[str], points: Sequence[str],
                       T: Transform, panels: Sequence[str] = (), stretch: bool = False,
                       reverse_areas: bool = False, merge: bool = True) -> int:
    """Move / rotate / mirror the selection itself (one undo step); returns the
    number of joints moved.  Without ``stretch`` a joint shared with an
    unselected object is left for that object and the selection gets its own
    copy of the joint (the selection detaches)."""
    frames = [n for n in frames if n in doc.model.frames]
    areas = [n for n in areas if n in doc.model.areas]
    points = [n for n in points if n in doc.model.points]
    panels = [n for n in panels if n in doc.model.panel_loads and not doc.model.panel_loads[n].area]
    if not (frames or areas or points or panels):
        return 0
    count = [0]

    def mutate(model: Model):
        names = _sel_points(model, frames, areas, points)
        fset, aset = set(frames), set(areas)
        if not stretch:
            # joints used by an unselected object stay; the selection takes a copy
            shared = set()
            for f in model.frames.values():
                if f.name not in fset:
                    shared |= {f.point_i, f.point_j}
            for a in model.areas.values():
                if a.name not in aset:
                    shared |= set(a.points)
            shared -= set(points)
            remap: Dict[str, str] = {}
            for n in names:
                if n in shared:
                    p = model.points[n]
                    q = model.add_point(p.x, p.y, p.z, merge=False)
                    q.story = p.story
                    remap[n] = q.name
            for n in frames:
                f = model.frames[n]
                f.point_i = remap.get(f.point_i, f.point_i)
                f.point_j = remap.get(f.point_j, f.point_j)
            for n in areas:
                a = model.areas[n]
                a.points = [remap.get(p, p) for p in a.points]
            names = [remap.get(n, n) for n in names]
        for n in names:
            p = model.points[n]
            x, y, z = (float(v) for v in T(np.array(p.xyz, float)))
            p.x, p.y, p.z = x, y, z
        if reverse_areas:
            for n in areas:
                model.areas[n].points = model.areas[n].points[::-1]
        for n in panels:
            pl = model.panel_loads[n]
            pl.points = [tuple(float(v) for v in T(np.array(q, float))) for q in pl.points]
        if merge:
            _merge_points(model, names, max(model.options.merge_tolerance, 1e-6))
        _restory(model)
        count[0] = len(names)

    doc.edit(mutate)
    return count[0]


def copy_transformed(doc: Document, frames, areas, points, T: Transform, reverse_areas: bool = False,
                     panels: Sequence[str] = ()) -> Tuple[List[str], List[str], List[str]]:
    """Copy of the selection through ``T``; returns the new (frames, areas, points)."""
    before = (set(doc.model.frames), set(doc.model.areas), set(doc.model.points), set(doc.model.panel_loads))
    free = [n for n in panels if n in doc.model.panel_loads and not doc.model.panel_loads[n].area]

    def panels_too(model: Model):  # in the same undo step as the objects
        for n in free:
            pl = copy.deepcopy(model.panel_loads[n])
            pl.name = model._next_name("PL", model.panel_loads)
            pl.points = [tuple(float(v) for v in T(np.array(q, float))) for q in pl.points]
            model.panel_loads[pl.name] = pl
    if frames or areas or points:
        created = replicate_transform(doc, list(frames), list(areas), list(points), [T], reverse_areas=reverse_areas,
                                      extra=panels_too if free else None)
    else:
        created = []
        if free:
            doc.edit(panels_too)
    m = doc.model
    fr = [n for n in created if n in m.frames and n not in before[0]]
    ar = [n for n in created if n in m.areas and n not in before[1]]
    pt = [n for n in created if n in m.points and n not in before[2]]
    return fr, ar, pt


def move_joint(doc: Document, point: str, xyz: XYZ) -> str:
    """Grip drag: put one joint at ``xyz`` (connected members stretch); a joint
    already there takes its place (the members connect to it).  Returns the
    name of the joint at the new place."""
    out = [point]

    def mutate(model: Model):
        p = model.points[point]
        p.x, p.y, p.z = (float(v) for v in xyz)
        tol = max(model.options.merge_tolerance, 1e-6)
        before = set(model.points)
        _merge_points(model, [point], tol)
        if point not in model.points:  # merged into the joint that was there
            q = model.find_point(*xyz)
            out[0] = q.name if q is not None else point
        _restory(model)
        del before

    doc.edit(mutate)
    return out[0]


# -------------------------------------------------- trim / extend / offset
def _in_plane_frames(model: Model, plane: Plane, tol: float):
    for f in model.frames.values():
        a, b = model.points[f.point_i].xyz, model.points[f.point_j].xyz
        if plane.contains(a, tol) and plane.contains(b, tol):
            yield f, np.array(plane.to2(a), float), np.array(plane.to2(b), float)


def _boundaries(model: Model, plane: Plane, exclude: str, tol: float):
    """Cutting / boundary edges in the view plane: frames, shell edges, grid lines
    (and story levels in elevations).  Each is (p, d, finite): p + t d, t in [0, 1]
    for a finite edge, any t for a grid line."""
    out = []
    for f, a, b in _in_plane_frames(model, plane, tol):
        if f.name != exclude:
            out.append((a, b - a, True))
    for ar in model.areas.values():
        pts = [model.points[n].xyz for n in ar.points]
        if all(plane.contains(p, tol) for p in pts):
            q = [np.array(plane.to2(p), float) for p in pts]
            for k in range(len(q)):
                out.append((q[k], q[(k + 1) % len(q)] - q[k], True))
    xs, ys = grid_ordinates(model)
    levels = story_levels(model)
    if plane.kind == "plan":
        out += [(np.array([x, 0.0]), np.array([0.0, 1.0]), False) for x in xs]
        out += [(np.array([0.0, y]), np.array([1.0, 0.0]), False) for y in ys]
    else:
        us = xs if plane.kind == "xz" else ys
        out += [(np.array([u, 0.0]), np.array([0.0, 1.0]), False) for u in us]
        out += [(np.array([0.0, z]), np.array([1.0, 0.0]), False) for z in levels]
    return out


def _cut_params(a: np.ndarray, b: np.ndarray, edges, eps: float = 1e-9) -> List[float]:
    """Parameters t along a -> b (t = 0 at a, 1 at b, any value) where the line meets the edges."""
    d = b - a
    ts = []
    for p, e, finite in edges:
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < eps * (np.linalg.norm(d) * np.linalg.norm(e) or 1.0):
            continue  # parallel
        w = p - a
        t = (w[0] * e[1] - w[1] * e[0]) / den
        s = (w[0] * d[1] - w[1] * d[0]) / den
        if finite and not (-1e-9 <= s <= 1 + 1e-9):
            continue
        ts.append(float(t))
    return sorted(ts)


def _frame_param(plane: Plane, a2, b2, xyz) -> float:
    c = np.array(plane.to2(xyz), float)
    d = b2 - a2
    return float(np.dot(c - a2, d) / (np.dot(d, d) or 1.0))


def trim_plan(model: Model, frame: str, plane: Plane, click: XYZ) -> Optional[Tuple[float, float]]:
    """(t0, t1) of the part of ``frame`` a quick trim at ``click`` removes, or None."""
    tol = max(model.options.merge_tolerance, 1e-6)
    f = model.frames.get(frame)
    if f is None:
        return None
    a3, b3 = model.points[f.point_i].xyz, model.points[f.point_j].xyz
    if not (plane.contains(a3, tol) and plane.contains(b3, tol)):
        return None
    a, b = np.array(plane.to2(a3), float), np.array(plane.to2(b3), float)
    L = float(np.linalg.norm(b - a)) or 1.0
    e = tol / L * 10
    ts = [t for t in _cut_params(a, b, _boundaries(model, plane, frame, tol)) if e < t < 1 - e]
    tc = _frame_param(plane, a, b, click)
    lo = max((t for t in ts if t < tc), default=None)
    hi = min((t for t in ts if t > tc), default=None)
    if lo is None and hi is None:
        return None
    return (0.0 if lo is None else lo, 1.0 if hi is None else hi)


def trim_frame(doc: Document, frame: str, plane: Plane, click: XYZ) -> str:
    """Quick trim (AutoCAD TRIM / Revit split-trim): cut away the part of the
    frame between the two cutting edges around the click (frames, shell edges,
    grid lines).  A middle part splits the frame in two."""
    span = trim_plan(doc.model, frame, plane, click)
    if span is None:
        raise ModelError("NO_CUT", "Nothing crosses this frame here: no cutting edge to trim to")
    t0, t1 = span

    def mutate(model: Model):
        f = model.frames[frame]
        a, b = np.array(model.points[f.point_i].xyz), np.array(model.points[f.point_j].xyz)
        if t0 <= 0.0:
            f.point_i = model.add_point(*(a + (b - a) * t1)).name
        elif t1 >= 1.0:
            f.point_j = model.add_point(*(a + (b - a) * t0)).name
        else:
            pj = f.point_j
            f.point_j = model.add_point(*(a + (b - a) * t0)).name
            g = model.add_frame(model.add_point(*(a + (b - a) * t1)), pj, f.section, type=f.type,
                                **_frame_kwargs(f))
            g.loads = copy.deepcopy(f.loads)
        for n in _unused_points(model):
            model.points.pop(n, None)
        _restory(model)

    doc.edit(mutate)
    return "Trimmed" if (t0 <= 0.0 or t1 >= 1.0) else "Trimmed (frame split in two)"


def extend_plan(model: Model, frame: str, plane: Plane, click: XYZ) -> Optional[Tuple[str, XYZ]]:
    """(end "i" | "j", new end point) for extending ``frame`` from the end nearer the click."""
    tol = max(model.options.merge_tolerance, 1e-6)
    f = model.frames.get(frame)
    if f is None:
        return None
    a3, b3 = model.points[f.point_i].xyz, model.points[f.point_j].xyz
    if not (plane.contains(a3, tol) and plane.contains(b3, tol)):
        return None
    a, b = np.array(plane.to2(a3), float), np.array(plane.to2(b3), float)
    L = float(np.linalg.norm(b - a)) or 1.0
    e = tol / L * 10
    tc = _frame_param(plane, a, b, click)
    ts = _cut_params(a, b, _boundaries(model, plane, frame, tol))
    if tc >= 0.5:
        t = min((t for t in ts if t > 1 + e), default=None)
        end = "j"
    else:
        t = max((t for t in ts if t < -e), default=None)
        end = "i"
    if t is None:
        return None
    p = plane.to3(tuple(a + (b - a) * t))
    return end, p


def extend_frame(doc: Document, frame: str, plane: Plane, click: XYZ) -> str:
    """Extend (AutoCAD EXTEND): lengthen the frame from the end nearer the click
    to the next boundary (frame, shell edge, grid line)."""
    res = extend_plan(doc.model, frame, plane, click)
    if res is None:
        raise ModelError("NO_BOUNDARY", "No boundary ahead of this end to extend to")
    end, p = res

    def mutate(model: Model):
        f = model.frames[frame]
        q = model.add_point(*p).name
        if end == "i":
            f.point_i = q
        else:
            f.point_j = q
        for n in _unused_points(model):
            model.points.pop(n, None)
        _restory(model)

    doc.edit(mutate)
    return f"Extended end {end.upper()}"


def offset_line(model: Model, frame: str, plane: Plane, distance: float, side: XYZ):
    """End points of the offset copy of ``frame`` on the side of ``side`` (in the view plane)."""
    tol = max(model.options.merge_tolerance, 1e-6)
    f = model.frames.get(frame)
    if f is None:
        return None
    a3, b3 = np.array(model.points[f.point_i].xyz), np.array(model.points[f.point_j].xyz)
    n = plane_normal(plane)
    d = b3 - a3
    perp = np.cross(n, d)
    if np.linalg.norm(perp) < 1e-12:
        return None  # a column seen end-on in plan
    perp /= np.linalg.norm(perp)
    s = 1.0 if float(np.dot(np.asarray(side, float) - a3, perp)) >= 0 else -1.0
    off = perp * s * abs(distance)
    del tol
    return tuple(a3 + off), tuple(b3 + off)


def offset_frame(doc: Document, frame: str, plane: Plane, distance: float, side: XYZ) -> str:
    line = offset_line(doc.model, frame, plane, distance, side)
    if line is None or abs(distance) <= 0:
        raise ModelError("NO_OFFSET", "Set an offset distance (Properties of Object) and click a frame")
    out = [""]

    def mutate(model: Model):
        f = model.frames[frame]
        g = model.add_frame(model.add_point(*line[0]), model.add_point(*line[1]), f.section, type=f.type,
                            **_frame_kwargs(f))
        g.loads = copy.deepcopy(f.loads)
        out[0] = g.name
        _restory(model)

    doc.edit(mutate)
    return out[0]


# ------------------------------------------------------------- typed input
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_typed(text: str, base: XYZ, toward: Optional[XYZ], plane: Plane) -> XYZ:
    """A point typed relative to ``base`` (CAD dynamic input):

    * ``6``      - distance 6 in the direction of the cursor (direct distance entry);
    * ``6<30``   - distance 6 at 30 degrees from the plane's horizontal axis;
    * ``3,4``    - relative (du, dv) in the view plane (``@3,4`` also accepted);
    * ``3,4,1``  - relative (dx, dy, dz) in global axes.
    """
    t = text.strip().replace(" ", "").lstrip("@")
    if not t:
        raise ValueError("Type a distance, distance<angle or dx,dy")
    b = np.array(base, float)
    if re.fullmatch(_NUM, t):
        L = float(t)
        if toward is None or np.linalg.norm(np.asarray(toward, float) - b) < 1e-12:
            if plane.kind == "3d":
                raise ValueError("In 3D a typed length needs a direction: point the cursor where the line goes, then type "
                                 "the length and press Enter - or type dx,dy,dz (relative) or #x,y,z (absolute)")
            d = np.array(plane.to3((1.0, 0.0)), float) - np.array(plane.to3((0.0, 0.0)), float)
        else:
            d = np.asarray(toward, float) - b
        return tuple(b + d / np.linalg.norm(d) * L)
    m = re.fullmatch(rf"({_NUM})<({_NUM})", t)
    if m:
        if plane.kind == "3d":
            raise ValueError("Distance<angle works in plan and elevation windows")
        L, ang = float(m.group(1)), math.radians(float(m.group(2)))
        u, v = plane.to2(tuple(b))
        return plane.to3((u + L * math.cos(ang), v + L * math.sin(ang)))
    parts = t.split(",")
    if all(re.fullmatch(_NUM, p) for p in parts):
        vals = [float(p) for p in parts]
        if len(vals) == 3:
            return tuple(b + np.array(vals))
        if len(vals) == 2:
            if plane.kind == "3d":
                return tuple(b + np.array([vals[0], vals[1], 0.0]))
            u, v = plane.to2(tuple(b))
            return plane.to3((u + vals[0], v + vals[1]))
    raise ValueError(f"Cannot read {text!r}: type 6, 6<30, 3,4 or 3,4,1")


def parse_absolute(text: str, plane: Plane) -> XYZ:
    """Typed absolute coordinates: ``x,y,z`` (global), or ``u,v`` in a plan /
    elevation window (the window plane gives the third coordinate)."""
    t = text.strip().replace(" ", "")
    parts = t.split(",") if t else []
    if parts and all(re.fullmatch(_NUM, p) for p in parts):
        vals = [float(p) for p in parts]
        if len(vals) == 3:
            return (vals[0], vals[1], vals[2])
        if len(vals) == 2:
            if plane.kind == "3d":
                raise ValueError("In 3D type all three coordinates: x,y,z (for example 6,0,3) and press Enter")
            return tuple(float(c) for c in plane.to3((vals[0], vals[1])))
    raise ValueError(f"Type the point as x,y,z{'' if plane.kind == '3d' else ' (or u,v in this window)'}"
                     f" - not {text!r}")


def ghost_lines(model: Model, frames, areas, points, panels, T: Transform, limit: int = 4000):
    """Line segments of the objects mapped through ``T`` (the preview of a move / copy)."""
    m = model
    out = []
    cache: Dict[str, XYZ] = {}

    def P(n):
        if n not in cache:
            cache[n] = tuple(float(v) for v in T(np.array(m.points[n].xyz, float)))
        return cache[n]

    for n in list(frames)[:limit]:
        f = m.frames.get(n)
        if f is not None:
            out.append((P(f.point_i), P(f.point_j)))
    for n in list(areas)[:limit]:
        a = m.areas.get(n)
        if a is not None:
            q = [P(p) for p in a.points]
            out += [(q[k], q[(k + 1) % len(q)]) for k in range(len(q))]
    r = 0.004 * max(1.0, float(np.ptp(np.array([p.xyz for p in m.points.values()]), axis=0).max())
                    if m.points else 1.0)
    for n in list(points)[:limit]:
        if n in m.points:
            c = np.array(P(n))
            out += [(tuple(c - (r, 0, 0)), tuple(c + (r, 0, 0))), (tuple(c - (0, r, 0)), tuple(c + (0, r, 0)))]
    for n in list(panels):
        pl = m.panel_loads.get(n)
        if pl is not None and not pl.area:
            q = [tuple(float(v) for v in T(np.array(p, float))) for p in pl.points]
            out += [(q[k], q[(k + 1) % len(q)]) for k in range(len(q))]
    return out


# ------------------------------------------------------------------- tools
class ModifyTool(Tool):
    """Base of the selection commands: asks for objects when nothing is selected."""

    views = ("plan", "xz", "yz", "3d")
    selecting = False
    needs = "objects"

    def activate(self, state) -> None:
        self.state = state
        self.selecting = state.count == 0

    def done_selecting(self, state) -> bool:
        if state.count == 0:
            state.status.emit(f"{self.title}: select objects first (click / window), then Enter")
            return False
        self.selecting = False
        state.update_hint()
        return True

    def _sel(self):
        st = self.state
        return list(st.frames), list(st.areas), list(st.points), list(st.panels)

    def hint(self, state=None) -> str:
        if self.selecting:
            n = state.count if state is not None else 0
            if n:
                return next_step(self.title, f"press Enter (or right-click) to use the {n} selected object(s)",
                                 "click = add / remove", "Esc = cancel")
            return next_step(self.title, f"select the {self.needs} (click or drag a window)", "Esc = cancel")
        return self.step_hint()

    def step_hint(self) -> str:
        return next_step(self.title, self.prompt)

    # ghost of the selection through a transform (preview)
    def ghost(self, T: Transform, limit: int = 4000) -> List[Tuple[XYZ, XYZ]]:
        st = self.state
        return ghost_lines(self.model, st.frames, st.areas, st.points, st.panels, T, limit)

    def _end(self):
        """Back to the select tool (the result stays selected)."""
        self.points = []
        if hasattr(self, "state"):
            self.state.set_tool("select")

    @staticmethod
    def _measure(a, b) -> str:
        d = np.subtract(b, a)
        return f"L = {float(np.linalg.norm(d)):.3f}   dX {d[0]:.3f}  dY {d[1]:.3f}  dZ {d[2]:.3f}"


class MoveTool(ModifyTool):
    name = "move"
    title = "Move"
    prompt = "click the base point"
    next_prompt = "click the destination (or type a distance: 6, 6<30, 3,4)"

    def step_hint(self) -> str:
        return next_step(self.title, self.next_prompt if self.points else self.prompt,
                         *(("type a distance + Enter",) if self.points else ()), "Esc = cancel")

    def click(self, pt, snap=None) -> bool:
        if self.selecting:
            return False
        p = self._p3(pt)
        if not self.points:
            self.points = [p]
            return False
        d = np.subtract(p, self.points[0])
        f, a, pp, pan = self._sel()
        self.apply(translation(d), f, a, pp, pan)
        return True

    def apply(self, T, f, a, pp, pan):
        n = transform_in_place(self.doc, f, a, pp, T, panels=pan, stretch=self.settings.modify_stretch)
        self.state.status.emit(f"Moved {len(f)} frame(s), {len(a)} shell(s) ({n} joint(s))")
        self._end()

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=self._p3(cursor))
        if self.points and not self.selecting:
            c = self._p3(cursor)
            pv.lines = self.ghost(translation(np.subtract(c, self.points[0])))
            pv.guides = [(self.points[0], c)]
            pv.text = self._measure(self.points[0], c)
        return pv


class CopyTool(MoveTool):
    name = "copy"
    title = "Copy"
    next_prompt = "click where the copy goes (again for more copies; Enter / Esc = done)"

    def click(self, pt, snap=None) -> bool:
        if self.selecting:
            return False
        p = self._p3(pt)
        if not self.points:
            self.points = [p]
            return False
        f, a, pp, pan = self._sel()
        fr, ar, pts = copy_transformed(self.doc, f, a, pp, translation(np.subtract(p, self.points[0])), panels=pan)
        self.copies = getattr(self, "copies", 0) + 1
        self.state.status.emit(f"Copy {self.copies}: {len(fr)} frame(s), {len(ar)} shell(s)")
        if not self.settings.modify_multiple:
            self.state.set_selection(fr, ar, pts if not (fr or ar) else ())
            self._end()
        return True

    def finish(self) -> bool:
        if self.points:
            self._end()
        return False


class RotateTool(ModifyTool):
    name = "rotate"
    title = "Rotate"
    prompt = "click the centre of rotation"
    views = ("plan", "xz", "yz")

    def step_hint(self) -> str:
        if not self.points:
            return next_step(self.title, self.prompt)
        if len(self.points) == 1:
            return next_step(self.title, "click the start direction (or type the angle and press Enter)")
        return next_step(self.title, "click the end direction (or type the angle and press Enter)")

    def _angle(self, p) -> float:
        c = self.plane.to2(self.points[0])
        u, v = self.plane.to2(p)
        return math.degrees(math.atan2(v - c[1], u - c[0]))

    def _T(self, ang: float) -> Transform:
        return rotation(self.points[0], plane_normal(self.plane), ang)

    def click(self, pt, snap=None) -> bool:
        if self.selecting:
            return False
        p = self._p3(pt)
        if not self.points:
            self.points = [p]
            return False
        if len(self.points) == 1:
            self.points.append(p)
            return False
        self.rotate(self._angle(p) - self._angle(self.points[1]))
        return True

    def rotate(self, ang: float):
        f, a, pp, pan = self._sel()
        T = self._T(ang)
        if self.settings.modify_copy:
            fr, ar, pts = copy_transformed(self.doc, f, a, pp, T, panels=pan)
            self.state.set_selection(fr, ar, () if (fr or ar) else pts)
        else:
            transform_in_place(self.doc, f, a, pp, T, panels=pan, stretch=self.settings.modify_stretch)
        self.state.status.emit(f"Rotated by {ang:.2f}°" + (" (copy)" if self.settings.modify_copy else ""))
        self._end()

    def typed(self, text: str, toward=None):
        if not self.points:
            raise ValueError("Click the centre of rotation first")
        ang = float(text.replace("°", "").strip())
        self.rotate(ang)
        return None

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=self._p3(cursor))
        if not self.points or self.selecting:
            return pv
        c = self._p3(cursor)
        pv.guides = [(self.points[0], c)]
        if len(self.points) == 1:
            pv.text = f"start direction {self._angle(c):.1f}°"
            return pv
        ang = self._angle(c) - self._angle(self.points[1])
        pv.guides.append((self.points[0], self.points[1]))
        pv.lines = self.ghost(self._T(ang))
        pv.text = f"angle {((ang + 180) % 360) - 180:.2f}°"
        return pv


class MirrorTool(ModifyTool):
    name = "mirror"
    title = "Mirror"
    prompt = "click the first point of the mirror axis"
    views = ("plan", "xz", "yz")

    def step_hint(self) -> str:
        return next_step(self.title, "click the second point of the axis" if self.points else self.prompt,
                         "keep the original: Properties of Object", "Esc = cancel")

    def click(self, pt, snap=None) -> bool:
        if self.selecting:
            return False
        p = self._p3(pt)
        if not self.points:
            self.points = [p]
            return False
        T = mirror(self.points[0], p, plane_normal(self.plane))
        f, a, pp, pan = self._sel()
        if self.settings.mirror_copy:
            fr, ar, pts = copy_transformed(self.doc, f, a, pp, T, reverse_areas=True, panels=pan)
            self.state.set_selection(fr, ar, () if (fr or ar) else pts)
        else:
            transform_in_place(self.doc, f, a, pp, T, panels=pan, stretch=self.settings.modify_stretch,
                               reverse_areas=True)
        self.state.status.emit("Mirrored" + (" (copy)" if self.settings.mirror_copy else ""))
        self._end()
        return True

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=self._p3(cursor))
        if self.points and not self.selecting:
            c = self._p3(cursor)
            pv.guides = [(self.points[0], c)]
            try:
                pv.lines = self.ghost(mirror(self.points[0], c, plane_normal(self.plane)))
            except ModelError:
                pass
        return pv


class PasteTool(ModifyTool):
    """Ctrl+V: the copied objects follow the cursor (by the lower-left corner of
    their extent, or the base point given when copying); click to place.  In a
    plan the click's story sets the level, so a floor copied on one story is
    pasted on another by clicking in that story's plan."""

    name = "paste"
    title = "Paste"
    prompt = "click where the copied objects go (or type 3,4 for an offset)"

    def activate(self, state) -> None:
        self.state = state
        self.selecting = False
        clip = getattr(state, "clipboard", None)
        if clip is None or clip.empty:
            state.status.emit("Paste: nothing copied (select objects and Ctrl+C first)")
        self.base = self._base(clip)
        self.points = [tuple(self.base)] if self.base is not None else []

    @staticmethod
    def _base(clip):
        if clip is None or clip.empty:
            return None
        if getattr(clip, "base", None) is not None:
            return np.array(clip.base, float)
        xyz = np.array([p.xyz for p in clip.points.values()], float)
        return xyz.min(axis=0)

    def step_hint(self) -> str:
        return next_step(self.title, self.prompt)

    def _offset(self, p) -> np.ndarray:
        return np.subtract(p, self.base)

    def click(self, pt, snap=None) -> bool:
        if self.base is None:
            return False
        p = self._p3(pt)
        d = self._offset(p)
        clip = self.state.clipboard
        fr, ar, pts = paste_objects(self.doc, clip, float(d[0]), float(d[1]), float(d[2]))
        self.state.status.emit(f"Pasted {len(fr)} frame(s), {len(ar)} shell(s)")
        self.points = []
        self.state.set_tool("select")
        self.state.set_selection(fr, ar, () if (fr or ar) else pts)
        return True

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=self._p3(cursor))
        clip = getattr(self.state, "clipboard", None)
        if self.base is None or clip is None:
            return pv
        d = self._offset(self._p3(cursor))
        out = []
        for f in clip.frames[:4000]:
            a, b = clip.points.get(f.point_i), clip.points.get(f.point_j)
            if a is not None and b is not None:
                out.append((tuple(np.add(a.xyz, d)), tuple(np.add(b.xyz, d))))
        for ar in clip.areas[:4000]:
            q = [tuple(np.add(clip.points[n].xyz, d)) for n in ar.points if n in clip.points]
            out += [(q[k], q[(k + 1) % len(q)]) for k in range(len(q))]
        pv.lines = out
        return pv


class _FramePickTool(Tool):
    """Trim / Extend / Offset: hover a frame, click to apply (stays active)."""

    views = ("plan", "xz", "yz")
    picks_frames = True  # the viewport passes the frame under the cursor

    def activate(self, state) -> None:
        self.state = state
        self.hover: Optional[str] = None

    def frame_click(self, frame: Optional[str], pt) -> bool:
        return False

    def frame_preview(self, frame: Optional[str], cursor) -> Preview:
        return Preview(marker=None if cursor is None else self._p3(cursor))


class TrimTool(_FramePickTool):
    name = "trim"
    title = "Trim"
    prompt = "click the part of a beam / column to cut away (at crossing frames, shell edges, grid lines)"

    def hint(self, state=None) -> str:
        return next_step(self.title, self.prompt, "Esc = done")

    def frame_click(self, frame, pt) -> bool:
        if frame is None:
            return False
        msg = trim_frame(self.doc, frame, self.plane, self._p3(pt))
        self.state.status.emit(f"{msg}: {frame}")
        return True

    def frame_preview(self, frame, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if frame is None or cursor is None:
            return pv
        span = trim_plan(self.model, frame, self.plane, self._p3(cursor))
        if span is None:
            pv.text = "no cutting edge on this frame"
            return pv
        f = self.model.frames[frame]
        a, b = np.array(self.model.points[f.point_i].xyz), np.array(self.model.points[f.point_j].xyz)
        pv.remove = [(tuple(a + (b - a) * span[0]), tuple(a + (b - a) * span[1]))]
        pv.text = f"trim {float(np.linalg.norm(b - a)) * (span[1] - span[0]):.3f}"
        return pv


class ExtendTool(_FramePickTool):
    name = "extend"
    title = "Extend"
    prompt = "click a beam near the end to lengthen (to the next frame, shell edge or grid line)"

    def hint(self, state=None) -> str:
        return next_step(self.title, self.prompt, "Esc = done")

    def frame_click(self, frame, pt) -> bool:
        if frame is None:
            return False
        msg = extend_frame(self.doc, frame, self.plane, self._p3(pt))
        self.state.status.emit(f"{msg}: {frame}")
        return True

    def frame_preview(self, frame, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if frame is None or cursor is None:
            return pv
        res = extend_plan(self.model, frame, self.plane, self._p3(cursor))
        if res is None:
            pv.text = "no boundary ahead of this end"
            return pv
        end, p = res
        f = self.model.frames[frame]
        q = self.model.points[f.point_i if end == "i" else f.point_j].xyz
        pv.lines = [(q, p)]
        pv.text = f"extend {math.dist(q, p):.3f}"
        return pv


class OffsetTool(_FramePickTool):
    name = "offset"
    title = "Offset"
    prompt = "click a beam on the side of the copy (type a number + Enter = distance)"

    def hint(self, state=None) -> str:
        return next_step(f"{self.title} ({self.settings.offset_distance:g})", self.prompt,
                         "type a new distance + Enter", "Esc = done")

    def typed(self, text: str, toward=None):
        v = float(text)
        if v <= 0:
            raise ValueError("The offset distance must be positive")
        self.settings.offset_distance = v
        self.state.update_hint()
        self.state.status.emit(f"Offset distance {v:g}")
        return None

    def frame_click(self, frame, pt) -> bool:
        if frame is None:
            return False
        name = offset_frame(self.doc, frame, self.plane, self.settings.offset_distance, self._p3(pt))
        self.state.status.emit(f"Offset {frame} -> {name}")
        return True

    def frame_preview(self, frame, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if frame is None or cursor is None:
            return pv
        line = offset_line(self.model, frame, self.plane, self.settings.offset_distance, self._p3(cursor))
        if line is not None:
            pv.lines = [line]
            pv.text = f"offset {self.settings.offset_distance:g}"
        return pv


TOOLS.update({"move": MoveTool, "copy": CopyTool, "rotate": RotateTool, "mirror": MirrorTool,
              "paste": PasteTool, "trim": TrimTool, "extend": ExtendTool, "offset": OffsetTool})
_STRETCH = ("modify_stretch", "Stretch connected objects", "bool",
            "on: joints shared with unselected objects move too (stretch); off: the selection detaches")
TOOL_PROPERTIES.update({
    "move": [_STRETCH],
    "copy": [("modify_multiple", "Multiple copies", "bool", "keep placing copies until Enter / Esc")],
    "rotate": [("modify_copy", "Keep the original (rotate a copy)", "bool", ""), _STRETCH],
    "mirror": [("mirror_copy", "Keep the original (mirror a copy)", "bool", ""), _STRETCH],
    "offset": [("offset_distance", "Offset distance", "float", "or type the distance + Enter in the window")],
})
