"""Interactive modelling tools (ETABS / SAP2000 style draw, quick-draw and select).

The tools are pure Python so they can be unit tested without a window.  A
viewport feeds them points on its *drawing plane* (a story plan, a grid-line
elevation or, in 3D, snapped model points) and renders the preview geometry
they return.  Points may be given as (u, v) plane coordinates or as (x, y, z).
"""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..model import Model, ModelError
from ..model.objects import Area, Frame, Point
from .document import Document

XY = Tuple[float, float]
XYZ = Tuple[float, float, float]
FREE6 = (False,) * 6
PIN_ENDS = (False, False, False, False, True, True)  # M2, M3 released


# ----------------------------------------------------------------- geometry
def grid_ordinates(model: Model) -> Tuple[List[float], List[float]]:
    """(x ordinates of the X-direction lines, y ordinates of the Y-direction lines)."""
    xs = sorted({g.ordinate for g in model.grid_lines if g.direction == "X"})
    ys = sorted({g.ordinate for g in model.grid_lines if g.direction == "Y"})
    return xs, ys


def grid_intersections(model: Model) -> List[XY]:
    xs, ys = grid_ordinates(model)
    return [(x, y) for x in xs for y in ys]


def story_levels(model: Model) -> List[float]:
    return [s.elevation for s in model.stories_bottom_up()]


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))


def _seg_point_distance(p: XY, a: XY, b: XY) -> Tuple[float, float]:
    """Distance from p to segment ab and the parameter t of the foot."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 0:
        return _dist(p, a), 0.0
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    foot = (ax + t * dx, ay + t * dy)
    return _dist(p, foot), t


def segments_intersect(p1: XY, p2: XY, q1: XY, q2: XY) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def point_in_polygon(p: XY, poly: Sequence[XY]) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def next_step(title: str, step: str, *others: str) -> str:
    """The guide at the bottom left: the command, the step to do next, and the other
    choices at this point (Esc = cancel ...).  ``others`` default to Esc = cancel."""
    step = step.strip().rstrip(".")
    text = f"{title}  \u25B8  Next: {step[:1].lower() + step[1:]}"
    alts = [o for o in (others or ("Esc = cancel",)) if o]
    return text + ("      \u2022  " + "   \u2022  ".join(alts) if alts else "")


# A plan / elevation window shows (and snaps to) the objects within this many merge
# tolerances of its plane: one band for drawing, picking and snapping (D05)
PLANE_BAND = 10.0


def plane_band(model) -> float:
    """World tolerance of "in the plane of the window" (display, pick and snap)."""
    return max(model.options.merge_tolerance, 1e-6) * PLANE_BAND


# -------------------------------------------------------------------- planes
@dataclass
class Plane:
    """Drawing plane of a view.

    ``plan``: horizontal at z = value (story ``story``); ``xz``: vertical at
    y = value (elevation along an X-direction grid line such as "1");
    ``yz``: vertical at x = value; ``3d``: no plane (snap to model points).
    """

    kind: str = "plan"
    value: float = 0.0
    story: Optional[str] = None
    # 3D views: the direction towards the viewer (camera); arcs drawn in a 3D view
    # lie in the plane of their points and turn counter-clockwise as seen on screen
    normal: Optional[Tuple[float, float, float]] = field(default=None, compare=False)

    def to3(self, p) -> XYZ:
        if len(p) == 3:
            return (float(p[0]), float(p[1]), float(p[2]))
        u, v = float(p[0]), float(p[1])
        if self.kind == "plan":
            return (u, v, self.value)
        if self.kind == "xz":
            return (u, self.value, v)
        if self.kind == "yz":
            return (self.value, u, v)
        raise ModelError("NO_PLANE", "A 3D view needs snapped points")

    def to2(self, p) -> XY:
        x, y, z = self.to3(p)
        if self.kind == "xz":
            return (x, z)
        if self.kind == "yz":
            return (y, z)
        return (x, y)

    def contains(self, xyz, tol: float) -> bool:
        if self.kind == "plan":
            return abs(xyz[2] - self.value) <= tol
        if self.kind == "xz":
            return abs(xyz[1] - self.value) <= tol
        if self.kind == "yz":
            return abs(xyz[0] - self.value) <= tol
        return True

    def u_ordinates(self, model: Model) -> List[float]:
        """Grid ordinates along the plane's horizontal axis (elevations)."""
        xs, ys = grid_ordinates(model)
        return xs if self.kind == "xz" else ys if self.kind == "yz" else []

    def same(self, other: "Plane") -> bool:
        return self.kind == other.kind and abs(self.value - other.value) < 1e-9


# ------------------------------------------------------------------ snapping
@dataclass
class SnapResult:
    xy: tuple  # plan (x, y) for plan snaps; (x, y, z) otherwise
    kind: str  # grid | point | midpoint | free
    target: Optional[str] = None


class Snapper:
    """Finds the snap target nearest to a cursor position on a story plane."""

    def __init__(self, model: Model, story: Optional[str], tolerance: float,
                 grid: bool = True, points: bool = True, midpoints: bool = True):
        self.model = model
        self.story = story
        self.tol = tolerance
        self.use_grid = grid
        self.use_points = points
        self.use_midpoints = midpoints

    def _story_points(self) -> List[Point]:
        if self.story is None:
            return list(self.model.points.values())
        return [p for p in self.model.points.values() if p.story == self.story]

    def snap(self, xy: XY) -> SnapResult:
        best: Optional[SnapResult] = None
        best_d = self.tol
        if self.use_points:
            for p in self._story_points():
                d = _dist(xy, (p.x, p.y))
                if d < best_d:
                    best, best_d = SnapResult((p.x, p.y), "point", p.name), d
        if self.use_grid and best is None:
            for g in grid_intersections(self.model):
                d = _dist(xy, g)
                if d < best_d:
                    best, best_d = SnapResult(g, "grid"), d
        if self.use_midpoints and best is None:
            pts = self.model.points
            for f in self.model.frames.values():
                if self.story is not None and f.story != self.story:
                    continue
                a, b = pts[f.point_i], pts[f.point_j]
                if abs(a.z - b.z) > 1e-9:
                    continue
                m = ((a.x + b.x) / 2, (a.y + b.y) / 2)
                d = _dist(xy, m)
                if d < best_d:
                    best, best_d = SnapResult(m, "midpoint", f.name), d
        if best is None:
            # orthogonal snap onto a grid line (single ordinate)
            xs, ys = grid_ordinates(self.model)
            sx = min(xs, key=lambda v: abs(v - xy[0]), default=None) if self.use_grid else None
            sy = min(ys, key=lambda v: abs(v - xy[1]), default=None) if self.use_grid else None
            x = sx if sx is not None and abs(sx - xy[0]) < self.tol else xy[0]
            y = sy if sy is not None and abs(sy - xy[1]) < self.tol else xy[1]
            kind = "grid" if (x, y) != tuple(xy) else "free"
            return SnapResult((x, y), kind)
        return best


def snap_candidates(model: Model, plane: Plane, tol: float, grid: bool = True, points: bool = True,
                    midpoints: bool = True) -> List[Tuple[XYZ, str, Optional[str]]]:
    """3D snap candidates for elevation and 3D views: model points, grid-line x
    story-level intersections and member midpoints lying in the plane."""
    out: List[Tuple[XYZ, str, Optional[str]]] = []
    if points:
        for p in model.points.values():
            if plane.contains(p.xyz, tol):
                out.append((p.xyz, "point", p.name))
    if grid:
        xs, ys = grid_ordinates(model)
        levels = story_levels(model)
        if plane.kind == "xz":
            out += [((x, plane.value, z), "grid", None) for x in xs for z in levels]
        elif plane.kind == "yz":
            out += [((plane.value, y, z), "grid", None) for y in ys for z in levels]
        elif plane.kind == "3d":
            out += [((x, y, z), "grid", None) for x in xs for y in ys for z in levels]
    if midpoints:
        for f in model.frames.values():
            a, b = model.points[f.point_i].xyz, model.points[f.point_j].xyz
            if plane.contains(a, tol) and plane.contains(b, tol):
                out.append((tuple((np.add(a, b) / 2).tolist()), "midpoint", f.name))
    return out


# ------------------------------------------------------------- draw context
@dataclass
class DrawSettings:
    """What is drawn: the "Properties of Object" form of ETABS / SAP2000.

    Every draw tool shows the subset it uses (:data:`TOOL_PROPERTIES`)."""

    frame_section: Optional[str] = None
    brace_section: Optional[str] = None
    shell_section: Optional[str] = None
    wall_section: Optional[str] = None
    deck_section: Optional[str] = None  # Assign > Shell > Deck Section (command)
    all_stories: bool = False  # draw on all stories (ETABS "All Stories")
    column_angle: float = 0.0
    diaphragm: Optional[str] = None
    brace_pattern: str = "X"  # X | / | \\ | V | Inverted V
    secondary_count: int = 2
    secondary_direction: str = "X"  # secondary beams parallel to X or Y
    pinned_beams: bool = False
    # ETABS / SAP2000 Properties of Object items
    line_type: str = "Auto"  # Auto (from geometry) | Beam | Column | Brace | Null
    moment_releases: str = "Continuous"  # Continuous | Pinned | Pinned at start | Pinned at end
    line_shape: str = "Straight"  # Straight | Arc: 3 points | Arc: start, center, end | Arc: start, end, radius
    arc_segments: int = 8  # an arc is drawn as this many straight members (CSI curved frames)
    arc_radius: float = 0.0  # "Arc: start, end, radius" (model length units; sign: side of the bulge)
    frame_angle: float = 0.0  # local axis 2 angle of drawn beams / braces
    # insertion point (CSI numbering, 10 = centroid); None = Auto: beams 8 (top centre,
    # flush with the slab), columns / braces 10
    cardinal_point: Optional[int] = None
    # the offset from the centroid goes into the analysis (eccentric member); off = drawn
    # there only (ETABS "Do not transform frame stiffness for offsets from centroid")
    cardinal_transform: bool = False
    offset_x: float = 0.0  # plan offsets of the clicked locations (ETABS joints / columns)
    offset_y: float = 0.0
    offset_z: float = 0.0  # offset normal to the drawing plane (SAP2000 "XY Plane Offset Normal")
    drawing_control: str = "None"  # None | Parallel to X | Parallel to Y | Parallel to Z | Fixed dX dY | Fixed length | Fixed angle
    fixed_length: float = 0.0
    fixed_angle: float = 45.0
    area_type: str = "Auto"  # Auto | Floor | Wall | Ramp | Opening
    opening_shape: str = "Rectangle"  # Rectangle (2 corners / typed w,h) | Polygon (click the corners)
    quick_floor_edge: str = "Grid"  # Grid | Beams (quick slab bounded by beams)
    special_joint: bool = True
    # Load menu click tools (joint / frame / area loads); the panel tool asks in a form
    load_pattern: Optional[str] = None
    load_mode: str = "Add"  # Add | Replace (this pattern) | Delete (this pattern)
    load_fx: float = 0.0
    load_fy: float = 0.0
    load_fz: float = 0.0
    load_mx: float = 0.0
    load_my: float = 0.0
    load_mz: float = 0.0
    frame_load_kind: str = "Uniform"  # Uniform | Trapezoidal | Point
    load_direction: str = "Gravity"
    load_value: float = 0.0  # force / length (frames), force (point on a frame), force / area (areas)
    load_value_b: float = 0.0  # trapezoid: at the end
    load_rel: float = 0.5  # point load on a frame: relative distance from the start
    area_distribution: str = "On the area (shells)"  # | One-way to frames | Two-way to frames
    # 3D windows: where a point lands when nothing is snapped (D14)
    work_plane_3d: str = "Snap only"  # | XY / XZ / YZ through the last point
    area_span: str = "X"  # one-way span direction
    # CAD modify commands (Move / Copy / Rotate / Mirror / Offset)
    modify_stretch: bool = False  # joints shared with unselected objects move too (ETABS Move)
    modify_multiple: bool = True  # Copy: keep placing copies
    modify_copy: bool = False  # Rotate: rotate a copy
    mirror_copy: bool = True  # Mirror: keep the original (Revit / AutoCAD default)
    offset_distance: float = 1.0

    def offsets(self):
        return (self.offset_x, self.offset_y, self.offset_z)


WORK_PLANES_3D = ["Snap only", "XY through the last point", "XZ through the last point",
                  "YZ through the last point"]
_WORK_PLANE_ITEM = ("work_plane_3d", "3D: point off the snaps", "choice", WORK_PLANES_3D)

_OFFSET_HINT = "global, added to every picked point of this tool (0 = at the picked point)"
_OFFSETS = [("offset_x", "Offset X", "float", _OFFSET_HINT), ("offset_y", "Offset Y", "float", _OFFSET_HINT),
            ("offset_z", "Offset Z", "float", _OFFSET_HINT)]
# settings kept per tool: a value set in one tool's form never moves what another tool draws
TOOL_LOCAL_SETTINGS = ("offset_x", "offset_y", "offset_z", "drawing_control", "fixed_length", "fixed_angle",
                       "area_type", "work_plane_3d")

LINE_SHAPES = {  # line shape -> help shown in the status bar (AutoCAD ARC options)
    "Straight": "",
    "Arc: 3 points": "start point, a point on the arc, end point",
    "Arc: start, center, end": "start point, centre, then a point giving the end angle (counter-clockwise as seen on the screen; the radius is start - centre)",
    "Arc: start, end, radius": "start point, end point (Arc radius; 0 = the chord length, negative = bulge to the right)",
}


# properties shown for each tool (key, label, kind, choices / hint); kind: section_frame |
# section_shell | choice | float | int | bool | diaphragm
TOOL_PROPERTIES: Dict[str, List[tuple]] = {
    "point": [*_OFFSETS,
              ("special_joint", "Special joint", "bool", "keep the joint even when nothing connects to it"),
              ("all_stories", "All stories", "bool", "")],
    "beam": [("line_type", "Line object type", "choice", ["Auto", "Beam", "Column", "Brace", "Null"]),
             ("frame_section", "Section", "section_frame", ""),
             ("line_shape", "Line shape", "choice", list(LINE_SHAPES)),
             ("arc_segments", "Arc: number of members", "int", "the arc is made of this many straight members"),
             ("arc_radius", "Arc radius", "float", "Arc: start, end, radius (negative: bulge to the other side)"),
             ("moment_releases", "Moment releases", "choice",
              ["Continuous", "Pinned", "Pinned at start", "Pinned at end"]),
             ("frame_angle", "Local axis 2 angle", "float", "degrees"),
             ("cardinal_point", "Cardinal point", "cardinal", "the point of the section on the drawn line; ... = picture"),
             *_OFFSETS,
             ("drawing_control", "Drawing control", "choice",
              ["None", "Parallel to X", "Parallel to Y", "Parallel to Z", "Fixed length", "Fixed angle"]),
             ("fixed_length", "Fixed length", "float", ""), ("fixed_angle", "Fixed angle (deg)", "float", ""),
             _WORK_PLANE_ITEM, ("all_stories", "All stories", "bool", "")],
    "brace": [("brace_section", "Section", "section_frame", ""),
              ("moment_releases", "Moment releases", "choice", ["Continuous", "Pinned"]),
              ("frame_angle", "Local axis 2 angle", "float", "degrees"), *_OFFSETS,
              ("drawing_control", "Drawing control", "choice", ["None", "Fixed length", "Fixed angle"]),
              ("fixed_length", "Fixed length", "float", ""), ("fixed_angle", "Fixed angle (deg)", "float", ""),
              _WORK_PLANE_ITEM],
    "column": [("frame_section", "Section", "section_frame", ""), ("column_angle", "Angle (deg)", "float", ""),
               *_OFFSETS,
               ("cardinal_point", "Cardinal point", "cardinal", ""), ("all_stories", "All stories", "bool", "")],
    "quick_column": [("frame_section", "Section", "section_frame", ""), ("column_angle", "Angle (deg)", "float", ""),
                     ("offset_x", "Offset X", "float", _OFFSET_HINT), ("offset_y", "Offset Y", "float", _OFFSET_HINT),
                     ("all_stories", "All stories", "bool", "")],
    "quick_beam": [("frame_section", "Section", "section_frame", ""),
                   ("moment_releases", "Moment releases", "choice", ["Continuous", "Pinned"]),
                   ("all_stories", "All stories", "bool", "")],
    "secondary_beams": [("frame_section", "Section", "section_frame", ""),
                        ("secondary_count", "Number of beams", "int", ""),
                        ("secondary_direction", "Approx. orientation", "choice", ["X", "Y"]),
                        ("moment_releases", "Moment releases", "choice", ["Continuous", "Pinned"]),
                        ("all_stories", "All stories", "bool", "")],
    "quick_brace": [("brace_section", "Section", "section_frame", ""),
                    ("brace_pattern", "Bracing", "choice", ["X", "/", "\\", "V", "Inverted V"]),
                    ("moment_releases", "Moment releases", "choice", ["Continuous", "Pinned"])],
    "wall": [("wall_section", "Section", "section_shell", ""), *_OFFSETS, ("all_stories", "All stories", "bool", "")],
    "quick_wall": [("wall_section", "Section", "section_shell", ""), ("all_stories", "All stories", "bool", "")],
    "slab": [("area_type", "Area object type", "choice", ["Auto", "Floor", "Wall", "Ramp", "Opening"]),
             ("shell_section", "Section", "section_shell", ""), ("diaphragm", "Diaphragm", "diaphragm", ""),
             *_OFFSETS, ("all_stories", "All stories", "bool", "")],
    "rect_slab": [("shell_section", "Section", "section_shell", ""), ("diaphragm", "Diaphragm", "diaphragm", ""),
                  *_OFFSETS, ("all_stories", "All stories", "bool", "")],
    "quick_slab": [("shell_section", "Section", "section_shell", ""), ("diaphragm", "Diaphragm", "diaphragm", ""),
                   *_OFFSETS,
                   ("quick_floor_edge", "Bounded by", "choice", ["Grid", "Beams"]),
                   ("all_stories", "All stories", "bool", "")],
    "frame_section": [("frame_section", "Section", "section_list", "")],
    "opening": [("opening_shape", "Shape", "choice", ["Rectangle", "Polygon"]),
                ("all_stories", "All stories (plan)", "bool", "")],
}


def arc_polyline(mode: Optional[str], pts, n: int, radius: float, plane) -> List[XYZ]:
    """Points of an arc (n segments) from the clicked points, in the drawing plane
    (3D views: the plane of the three points)."""
    P = [np.asarray(p, float) for p in pts]
    view = None
    if plane is not None and plane.kind == "3d" and plane.normal is not None:
        view = np.asarray(plane.normal, float)
        view = view / (float(np.linalg.norm(view)) or 1.0)
    if mode == "Arc: start, end, radius":
        a, b = P[0], P[1]
        if plane is not None and plane.kind != "3d":
            u0, v0 = plane.to2(tuple(a))
            u1, v1 = plane.to2(tuple(b))
            ch = math.hypot(u1 - u0, v1 - v0)
            if ch <= 1e-12:
                return []
            r = abs(radius) if radius else ch  # default: radius = chord
            if r < ch / 2 * (1 - 1e-9):
                return []  # no circle of that radius passes through both ends (arc_problem says so)
            r = max(r, ch / 2)
            mx, my = (u0 + u1) / 2, (v0 + v1) / 2
            h = math.sqrt(max(r * r - ch * ch / 4, 0.0))
            nx, ny = -(v1 - v0) / ch, (u1 - u0) / ch
            sgn = 1.0 if radius >= 0 else -1.0
            # the minor arc, bulging to the left of start -> end for a positive radius
            mid = (mx + sgn * (r - h) * nx, my + sgn * (r - h) * ny)
            return arc_polyline("Arc: 3 points", [a, plane.to3(mid), b], n, 0.0, plane)
        # 3D view: the plane through the chord that faces the viewer best (horizontal
        # when nothing better is known); the bulge is to the left of start -> end on screen
        d = b - a
        ch = float(np.linalg.norm(d))
        if ch <= 1e-12:
            return []
        d = d / ch
        nrm = None
        for cand in ([view] if view is not None else []) + [np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])]:
            q = cand - float(cand @ d) * d
            if float(np.linalg.norm(q)) > 1e-6:
                nrm = q / float(np.linalg.norm(q))
                break
        left = np.cross(nrm, d)
        r = abs(radius) if radius else ch
        if r < ch / 2 * (1 - 1e-9):
            return []
        r = max(r, ch / 2)
        h = math.sqrt(max(r * r - ch * ch / 4, 0.0))
        sgn = 1.0 if radius >= 0 else -1.0
        mid = (a + b) / 2 + sgn * (r - h) * left
        return arc_polyline("Arc: 3 points", [a, mid, b], n, 0.0, plane)
    if len(P) < 3:
        return []
    if mode == "Arc: start, center, end":
        a, c, e = P
        r = float(np.linalg.norm(a - c))
        if r <= 1e-12:
            return []
        if plane is not None and plane.kind != "3d":
            u0, v0 = plane.to2(tuple(a))
            uc, vc = plane.to2(tuple(c))
            ue, ve = plane.to2(tuple(e))
            t0 = math.atan2(v0 - vc, u0 - uc)
            t1 = math.atan2(ve - vc, ue - uc)
            sweep = (t1 - t0) % (2 * math.pi) or 2 * math.pi  # counter-clockwise (AutoCAD)
            return [plane.to3((uc + r * math.cos(t0 + sweep * k / n), vc + r * math.sin(t0 + sweep * k / n)))
                    for k in range(n + 1)]
        # 3D view: the plane of the three points, counter-clockwise as seen by the viewer
        e1 = (a - c) / r
        nrm = np.cross(a - c, e - c)
        if float(np.linalg.norm(nrm)) <= 1e-9 * r * max(float(np.linalg.norm(e - c)), 1e-12):
            if view is None:
                return []
            nrm = view - float(view @ e1) * e1  # end straight through the centre: half circle
            if float(np.linalg.norm(nrm)) <= 1e-9:
                return []
        nrm = nrm / float(np.linalg.norm(nrm))
        if view is not None and float(nrm @ view) < 0:
            nrm = -nrm
        e2 = np.cross(nrm, e1)
        t1 = math.atan2(float((e - c) @ e2), float((e - c) @ e1)) % (2 * math.pi) or 2 * math.pi
        return [tuple((c + r * (math.cos(t1 * k / n) * e1 + math.sin(t1 * k / n) * e2)).tolist())
                for k in range(n + 1)]
    # three points on the arc
    a, m, b = P
    ab, am = b - a, m - a
    nrm = np.cross(ab, am)
    nn = float(nrm @ nrm)
    if nn <= 1e-18:
        return []
    c = a + (np.cross(nrm, ab) * float(am @ am) + np.cross(am, nrm) * float(ab @ ab)) / (2.0 * nn)
    r = float(np.linalg.norm(a - c))
    e1 = (a - c) / r
    e2 = np.cross(nrm / math.sqrt(nn), e1)
    ang = lambda p: math.atan2(float((p - c) @ e2), float((p - c) @ e1)) % (2 * math.pi)  # noqa: E731
    tb, tm = ang(b), ang(m)
    sweep = tb if tm <= tb else tb - 2 * math.pi  # through the middle point
    return [tuple((c + r * (math.cos(sweep * k / n) * e1 + math.sin(sweep * k / n) * e2)).tolist())
            for k in range(n + 1)]


def arc_info(chain) -> Tuple[float, float]:
    """(radius, sweep in degrees) of an arc polyline (its first, middle and last points)."""
    P = np.asarray(chain, float)
    if len(P) < 3:
        return 0.0, 0.0
    a, m, b = P[0], P[len(P) // 2], P[-1]
    ab, am = b - a, m - a
    nrm = np.cross(ab, am)
    nn = float(nrm @ nrm)
    if nn <= 1e-18:
        return float("inf"), 0.0
    c = a + (np.cross(nrm, ab) * float(am @ am) + np.cross(am, nrm) * float(ab @ ab)) / (2.0 * nn)
    R = float(np.linalg.norm(a - c))
    sweep = 0.0
    for k in range(len(P) - 1):
        u, v = P[k] - c, P[k + 1] - c
        sweep += math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(u @ v)))
    return R, sweep


def arc_problem(mode: Optional[str], pts, radius: float, plane) -> Optional[str]:
    """Why these picks give no arc (shown before / instead of drawing), or None."""
    P = [np.asarray(p, float) for p in pts]
    if mode == "Arc: start, end, radius" and len(P) >= 2:
        ch = float(np.linalg.norm(P[1] - P[0]))
        if ch <= 1e-12:
            return "The start and end points are the same"
        if radius and abs(radius) < ch / 2 * (1 - 1e-9):
            return (f"Radius {abs(radius):g} is smaller than half the chord ({ch / 2:.4g}): no arc joins these "
                    f"points - set Arc radius to {ch / 2:.4g} or more (0 = the chord length)")
        return None
    if mode == "Arc: start, center, end" and len(P) >= 3:
        if float(np.linalg.norm(P[0] - P[1])) <= 1e-12:
            return "The start point is on the centre"
        return None
    if len(P) >= 3 and float(np.linalg.norm(np.cross(P[2] - P[0], P[1] - P[0]))) <= 1e-12:
        return "The three points are on one line: no arc passes through them"
    return None


@dataclass
class Preview:
    """Geometry the viewport draws while a tool is in progress (3D coordinates)."""

    lines: List[Tuple[XYZ, XYZ]] = field(default_factory=list)
    polygon: List[XYZ] = field(default_factory=list)
    marker: Optional[XYZ] = None
    text: str = ""
    guides: List[Tuple[XYZ, XYZ]] = field(default_factory=list)  # thin dashed helper lines (base -> cursor)
    remove: List[Tuple[XYZ, XYZ]] = field(default_factory=list)  # what the command removes (trim), drawn red


class Tool:
    """Base class.  ``click`` receives the (snapped) point on the plane."""

    def _cardinal_extra(self, ftype: Optional[str] = "Beam") -> dict:
        return _cardinal_fields(self.settings, ftype)

    name = "tool"
    prompt = ""
    views = ("plan", "xz", "yz", "3d")  # views the tool works in
    tolerance = 1e9  # world tolerance for "near a grid segment" tests (set by the viewport)

    title = ""  # command name in the status-bar guidance (default: from ``name``)
    next_prompt = ""  # guidance once the first point is placed (chain tools)

    # a chain of points may continue in another window (its points are 3D) - polygon,
    # arc and single-plane tools start again when the plane changes
    keeps_points_across_planes = False

    def __init__(self, doc: Document, settings: DrawSettings):
        self.doc = doc
        self.settings = settings
        self.story: Optional[str] = None
        self.plane = Plane("plan", 0.0, None)
        self.points: List[XYZ] = []
        self.created: List[str] = []
        self._raw_last: Optional[XYZ] = None  # the last picked point before offsets (constraints use it)

    def activate(self, state) -> None:
        """Called when the command starts (state: the shared ViewState)."""

    def hint(self, state=None) -> str:
        """What to do next - shown at the bottom left of the window."""
        title = self.title or self.name.replace("_", " ").title()
        if self.points:
            return next_step(title, self.next_prompt or "click the next point",
                             "Enter / double-click = finish", "Esc = end this line", "type a length + Enter")
        return next_step(title, self.prompt, "Esc = cancel (back to Select)",
                         "Shift + right-click = snap menu")

    # ---- context
    @property
    def model(self) -> Model:
        return self.doc.model

    def set_story(self, story: Optional[str]):
        z = self.model.stories[story].elevation if story in self.model.stories else 0.0
        self.set_plane(Plane("plan", z, story))

    def set_plane(self, plane: Plane):
        if not plane.same(self.plane) and not (self.points and self.keeps_points_across_planes):
            self.cancel()
        self.plane = plane
        self.story = plane.story if plane.kind == "plan" else None

    def elevation(self) -> float:
        if self.story and self.story in self.model.stories:
            return self.model.stories[self.story].elevation
        return self.plane.value if self.plane.kind == "plan" else 0.0

    def _p3(self, pt) -> XYZ:
        return self.plane.to3(pt)

    def _story_at(self, z: float) -> Optional[str]:
        s = self.model.story_at(z, max(self.model.options.merge_tolerance, 1e-6))
        return s.name if s else None

    def _level_snapped(self, p: XYZ, skip_base: bool = False) -> XYZ:
        """In an elevation: the point moved to the nearest story level (columns and
        walls span whole stories, so a pick a little off a level means that level).
        ``skip_base``: the lowest level is not a top (a column needs a story below)."""
        if self.plane.kind not in ("xz", "yz"):
            return p
        lv = [z for _, z in self.model.levels()]
        if skip_base and len(lv) > 1:
            lv = lv[1:]
        if not lv:
            return p
        z = min(lv, key=lambda v: abs(v - p[2]))
        return (p[0], p[1], z)

    def _require_story(self, z: float) -> str:
        st = self._story_at(z)
        if st is None:
            raise ModelError("NOT_AT_STORY", "Click a point at a story level (Z grid level)")
        return st

    def story_names_to_draw(self, story: Optional[str] = None) -> List[str]:
        """Stories the drawn object is repeated on (bottom story excluded)."""
        story = story or self.story
        if not story:
            return []
        if not self.settings.all_stories:
            return [story]
        stories = self.model.stories_bottom_up()
        return [s.name for s in stories[1:]] if len(stories) > 1 else [story]

    @property
    def tol(self) -> float:
        return max(self.model.options.merge_tolerance, 1e-9)

    # ---- interaction
    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        """Return True when the tool completed an object."""
        return False

    def double_click(self, pt) -> bool:
        return self.finish()

    takes_typed_points = True  # tools that place points accept typed coordinates

    def typed(self, text: str, toward=None):
        """Typed input (CAD dynamic input): returns the point to click (see
        modify.parse_typed), or None when the tool used it itself.

        * before the first point: absolute coordinates ``x,y,z`` (plan / elevation:
          ``u,v`` in the window plane also works);
        * after it: ``6`` (towards the cursor), ``6<30``, relative ``dx,dy`` /
          ``dx,dy,dz``, or ``#x,y,z`` for an absolute point."""
        from .modify import parse_absolute, parse_typed
        t = text.strip().replace(" ", "")
        if not self.points or t.startswith("#"):
            return parse_absolute(t.lstrip("#"), self.plane)
        return parse_typed(text, self._raw_last if self._raw_last is not None else self.points[-1],
                           toward, self.plane)

    def finish(self) -> bool:
        return False

    def cancel(self):
        self.points = []
        self._raw_last = None

    def _uses(self, key: str) -> bool:
        """The tool's Properties of Object form shows this setting.  A setting it does
        not show (left over from another tool: offsets, drawing control ...) must not
        change what it draws."""
        return any(item[0] == key for item in TOOL_PROPERTIES.get(self.name, ()))

    def resolve(self, pt) -> XYZ:
        """The point a pick at ``pt`` (snapped / typed, before constraints) really
        places: drawing control first (relative to the last picked point, before its
        offset), then the offsets of the form - once.  Preview and click both use it."""
        return self._p3(pt)

    def preview(self, cursor) -> Preview:
        return Preview(marker=None if cursor is None else self.resolve(cursor))

    # ---- helpers
    def _point(self, model: Model, p, z: Optional[float] = None) -> Point:
        if len(p) == 2:
            return model.add_point(p[0], p[1], z if z is not None else self.elevation())
        return model.add_point(p[0], p[1], p[2] if z is None else z)

    def _section(self, attr: str, table: str, kind: Optional[str] = None) -> str:
        sec = getattr(self.settings, attr)
        tab = getattr(self.model, table)
        if sec is None or sec not in tab:
            if not tab:
                what = "frame" if table == "frame_sections" else "slab/wall"
                raise ModelError("NO_SECTION", f"Define a {what} section first (Define menu)")
            cands = [s.name for s in tab.values() if kind is None or getattr(s, "type", kind) == kind]
            sec = cands[0] if cands else next(iter(tab))
            setattr(self.settings, attr, sec)
        return sec

    def _frame_section(self) -> str:
        return self._section("frame_section", "frame_sections")

    def _brace_section(self) -> str:
        if self.settings.brace_section in self.model.frame_sections:
            return self.settings.brace_section
        return self._frame_section()

    def _shell_section(self, wall: bool = False) -> str:
        if wall:
            return self._section("wall_section", "shell_sections", "Wall")
        return self._section("shell_section", "shell_sections", "Slab")

    def _beam_releases(self):
        mode = self.settings.moment_releases
        if self.settings.pinned_beams or mode == "Pinned":
            return (PIN_ENDS, PIN_ENDS)
        if mode in ("Pinned at start", "Pinned at I"):  # I / J: files of earlier versions
            return (PIN_ENDS, FREE6)
        if mode in ("Pinned at end", "Pinned at J"):
            return (FREE6, PIN_ENDS)
        return (FREE6, FREE6)

    def _apply_drawing_control(self, p: XYZ) -> XYZ:
        """ETABS drawing constraints / SAP2000 drawing control on the next point
        (relative to the last picked point before its offsets)."""
        if not self.points or not self._uses("drawing_control"):
            return p
        ctl = self.settings.drawing_control
        a = self._raw_last if self._raw_last is not None else self.points[-1]
        if ctl == "None":
            return p
        if ctl == "Parallel to X":
            return (p[0], a[1], a[2])
        if ctl == "Parallel to Y":
            return (a[0], p[1], a[2])
        if ctl == "Parallel to Z":
            return (a[0], a[1], p[2])
        d = np.subtract(p, a)
        L = float(np.linalg.norm(d))
        if L <= 0:
            return p
        if ctl == "Fixed length" and self.settings.fixed_length > 0:
            return tuple((np.asarray(a) + d / L * self.settings.fixed_length).tolist())
        if ctl == "Fixed angle" and self.plane.kind != "3d":
            u0, v0 = self.plane.to2(a)
            u, v = self.plane.to2(p)
            ang = math.radians(self.settings.fixed_angle)
            e = np.array([math.cos(ang), math.sin(ang)])
            t = (u - u0) * e[0] + (v - v0) * e[1]
            return self.plane.to3((u0 + t * e[0], v0 + t * e[1]))
        return p

    def _offset_point(self, p: XYZ) -> XYZ:
        """Plan offsets of the Properties of Object form; the normal offset moves the
        point out of the drawing plane."""
        s = self.settings
        ox = s.offset_x if self._uses("offset_x") else 0.0
        oy = s.offset_y if self._uses("offset_y") else 0.0
        oz = s.offset_z if self._uses("offset_z") else 0.0
        if not (ox or oy or oz):
            return p
        x, y, z = p
        return (x + ox, y + oy, z + oz)  # global X / Y / Z, in every window

    def _offset_text(self) -> str:
        s = self.settings
        parts = [f"{lab} {getattr(s, k):g}" for k, lab in (("offset_x", "dX"), ("offset_y", "dY"),
                                                            ("offset_z", "dZ"))
                 if self._uses(k) and getattr(s, k)]
        return f"   [offset {', '.join(parts)}]" if parts else ""

    def _line_preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self.resolve(cursor))
        if self.points and cursor is not None:
            c = self.resolve(cursor)
            pv.marker = c
            pv.lines.append((self.points[-1], c))
            pv.text = f"L = {_dist(self.points[-1], c):.3f}"
            if self.plane.kind != "3d":
                (u0, v0), (u1, v1) = self.plane.to2(self.points[-1]), self.plane.to2(c)
                if abs(u1 - u0) + abs(v1 - v0) > 1e-12:
                    pv.text += f"   \u2220 {math.degrees(math.atan2(v1 - v0, u1 - u0)) % 360:.1f}\u00b0"
            if self.settings.drawing_control != "None":
                if self.settings.drawing_control == "Fixed angle" and self.plane.kind == "3d":
                    pv.text += "   [Fixed angle: plan / elevation windows only - not applied]"
                else:
                    pv.text += f"   [{self.settings.drawing_control}]"
            pv.text += self._offset_text()
        return pv


# ------------------------------------------------------------------- select
class SelectTool(Tool):
    name = "select"
    prompt = "Click to select (Ctrl adds); drag left-to-right = window, right-to-left = crossing"

    def hint(self, state=None) -> str:
        n = state.count if state is not None else 0
        if n:
            return next_step("Select", f"choose a command (Assign / Edit / Load) for the {n} selected object(s)",
                             "Ctrl + click = add / remove", "right-click = object info", "Delete = delete",
                             "Esc = clear the selection")
        return next_step("Select", "click an object or drag a window to select",
                         "drag right-to-left = crossing", "or choose a command first")


class SelectLineTool(Tool):
    """Select by intersecting line (ETABS "Intersecting Line")."""

    name = "select_line"
    prompt = "Drag a line: objects it crosses are selected"


# ---------------------------------------------------------------- draw point
class PointTool(Tool):
    title = "Draw Joints"
    name = "point"
    prompt = "Click to add joints"

    def resolve(self, pt) -> XYZ:
        return self._offset_point(self._p3(pt))

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self.resolve(pt)
        # a story plan repeats the joint on the chosen stories; a free plan (custom Z,
        # no story there), an elevation or 3D: the joint at the point itself
        on_story = self.plane.kind == "plan" and self.story and not self.settings.offset_z
        stories = self.story_names_to_draw() if on_story else [None]
        if not stories:
            raise ModelError("NOTHING_TO_DRAW", "No story to draw the joint on")

        def mutate(model: Model):
            for st in stories:
                z = model.stories[st].elevation if st else p[2]
                q = model.add_point(p[0], p[1], z)
                self.created.append(q.name)

        self.doc.edit(mutate)
        return True


# ---------------------------------------------------------- draw beam / line
class BeamTool(Tool):
    """Two clicks: a beam on the story plane, or any frame (beam / column /
    brace) between two points in an elevation or 3D view.  Chains from the
    last point; Esc or right-click ends the chain."""

    title = "Draw Beam / Column / Brace"
    next_prompt = "click the next point (the line goes on as a chain)"
    name = "beam"
    prompt = "Click the start point"
    frame_type: Optional[str] = "Beam"

    def hint(self, state=None) -> str:
        mode = self._arc_mode()
        if mode is None:
            return super().hint(state)
        return next_step(f"{self.title} ({mode})", self.prompt[:1].lower() + self.prompt[1:],
                         "Esc = cancel this arc" if self.points else "Esc = cancel (back to Select)",
                         "Line shape: Properties of Object")

    def cancel(self):
        super().cancel()
        if self._arc_mode() is not None:
            self.prompt = "Click the start point"

    def _arc_mode(self) -> Optional[str]:
        shape = getattr(self.settings, "line_shape", "Straight")
        return None if shape in (None, "", "Straight") or type(self) is not BeamTool else shape

    def arc_points(self, pts) -> List[XYZ]:
        """The points along the arc of the current line shape (start ... end)."""
        mode = self._arc_mode()
        n = max(2, int(self.settings.arc_segments or 8))
        return arc_polyline(mode, pts, n, float(self.settings.arc_radius or 0.0), self.plane)

    @property
    def keeps_points_across_planes(self) -> bool:  # a straight chain may go on in another window
        return self._arc_mode() is None

    def resolve(self, pt) -> XYZ:
        return self._offset_point(self._apply_drawing_control(self._p3(pt)))

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        raw = self._apply_drawing_control(self._p3(pt))
        p = self._offset_point(raw)
        self._raw_last = raw
        mode = self._arc_mode()
        if mode is not None:
            self.points.append(p)
            need = 2 if mode == "Arc: start, end, radius" else 3
            if len(self.points) < need:
                self.prompt = ("Click a point on the arc" if mode == "Arc: 3 points" and len(self.points) == 1
                               else "Click the centre" if mode == "Arc: start, center, end" and len(self.points) == 1
                               else "Click the end angle (the arc ends on the line centre -> this point)"
                               if mode == "Arc: start, center, end" else "Click the end point")
                return False
            chain = self.arc_points(self.points)
            picked = list(self.points)
            self.points = []
            self._raw_last = None
            self.prompt = "Click the start point"
            if len(chain) < 2:
                raise ModelError("BAD_ARC", arc_problem(mode, picked, float(self.settings.arc_radius or 0.0),
                                                        self.plane) or "These points do not define an arc")
            sec = self._frame_section()
            rel_i, rel_j = self._beam_releases()
            extra = dict(angle=self.settings.frame_angle,
                         **self._cardinal_extra(self._type_for(chain[0], chain[1]) or _geom_type(chain[0], chain[-1], self.tol)))

            def mutate_arc(model: Model):
                prev = model.add_point(*chain[0])
                for k in range(1, len(chain)):
                    q = model.add_point(*chain[k])
                    f = model.add_frame(prev, q, sec, type=self._type_for(chain[k - 1], chain[k]),
                                        releases_i=rel_i if k == 1 else FREE6,
                                        releases_j=rel_j if k == len(chain) - 1 else FREE6, **extra)
                    self.created.append(f.name)
                    prev = q
            self.doc.edit(mutate_arc)
            return True
        self.points.append(p)
        if len(self.points) < 2:
            self.prompt = "Click the end point"
            return False
        a, b = self.points[-2], self.points[-1]
        self.points = [b]  # chain: next member starts at the last point
        if _dist(a, b) <= self.tol:
            return False
        sec = self._section_for(a, b)
        rel_i, rel_j = self._releases_for(a, b)
        horizontal = abs(a[2] - b[2]) <= self.tol
        extra = dict(angle=self.settings.frame_angle,
                     **self._cardinal_extra(self._type_for(a, b) or _geom_type(a, b, self.tol)))
        if self.plane.kind == "plan" and horizontal and not self.settings.offset_z and self.story:
            stories = self.story_names_to_draw()
            ftype = self._type_for(a, b) or "Beam"

            def mutate(model: Model):
                for st in stories:
                    z = model.stories[st].elevation
                    f = model.add_frame(model.add_point(a[0], a[1], z), model.add_point(b[0], b[1], z), sec,
                                        type=ftype, story=st, releases_i=rel_i, releases_j=rel_j, **extra)
                    self.created.append(f.name)
        else:
            def mutate(model: Model):
                f = model.add_frame(model.add_point(*a), model.add_point(*b), sec, type=self._type_for(a, b),
                                    releases_i=rel_i, releases_j=rel_j, **extra)
                self.created.append(f.name)

        self.doc.edit(mutate)
        return True

    def _type_for(self, a, b) -> Optional[str]:
        t = self.settings.line_type
        return None if t in (None, "Auto") else t  # None: inferred from geometry (beam / column / brace)

    def _section_for(self, a, b) -> str:
        return self._frame_section()  # the section chosen in the form, whatever the slope

    def _releases_for(self, a, b):
        return self._beam_releases()

    def preview(self, cursor) -> Preview:
        mode = self._arc_mode()
        if mode is None or not self.points or cursor is None:
            return self._line_preview(cursor)
        c = self.resolve(cursor)
        pv = Preview(marker=c)
        pts = list(self.points) + [c]
        need = 2 if mode == "Arc: start, end, radius" else 3
        if len(pts) >= need:
            chain = self.arc_points(pts[:need])
            pv.lines = [(chain[k], chain[k + 1]) for k in range(len(chain) - 1)]
            if len(chain) >= 2:
                R, sweep = arc_info(chain)
                pv.text = (f"arc  L = {sum(_dist(chain[k], chain[k + 1]) for k in range(len(chain) - 1)):.3f}"
                           f"   R = {R:.3f}  sweep {sweep:.1f}\u00b0  ({len(chain) - 1} members)")
            else:
                pv.text = arc_problem(mode, pts[:need], float(self.settings.arc_radius or 0.0), self.plane) or ""
        else:
            pv.guides.append((self.points[-1], c))
        if mode == "Arc: start, center, end" and len(self.points) >= 2:
            pv.guides.append((self.points[1], self.points[0]))
            pv.guides.append((self.points[1], c))
        return pv

    def finish(self) -> bool:
        self.points = []
        self.prompt = "Click the start point"
        return False


def _cardinal_fields(settings, ftype: Optional[str] = "Beam") -> dict:
    """Frame attributes of the cardinal point chosen in Properties of Object: where the
    member is drawn (``display_cardinal``) and, if its offset goes into the analysis,
    ``cardinal_point``.  Auto (None): beams 8 (top centre), others 10."""
    cp = settings.cardinal_point
    if cp is None:
        cp = 8 if (ftype or "Beam") == "Beam" else 10
    cp = int(cp)
    transform = bool(getattr(settings, "cardinal_transform", False))
    return {"cardinal_point": cp if transform else 10, "display_cardinal": cp}


def _geom_type(a, b, tol: float = 1e-6) -> str:
    """Beam (level), Column (vertical) or Brace, as the model infers a line's type."""
    dx, dy, dz = (b[0] - a[0]), (b[1] - a[1]), (b[2] - a[2])
    if abs(dz) <= tol:
        return "Beam"
    if abs(dx) <= tol and abs(dy) <= tol:
        return "Column"
    return "Brace"


class BraceTool(BeamTool):
    """Draw braces between two points (elevation / 3D views)."""

    title = "Draw Braces"
    next_prompt = "click the end point of the brace"
    name = "brace"
    prompt = "Click the start point of the brace"
    views = ("xz", "yz", "3d", "plan")

    def _section_for(self, a, b) -> str:
        return self._brace_section()

    def _releases_for(self, a, b):
        return self._beam_releases()  # the form's Moment releases (Continuous / Pinned ...)


# -------------------------------------------------------------- draw column
class ColumnTool(Tool):
    """One click: column from the story below up to the clicked story level."""

    title = "Draw Columns"
    name = "column"
    prompt = "Click the column location"

    def resolve(self, pt) -> XYZ:
        return self._offset_point(self._level_snapped(self._p3(pt), skip_base=True))

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        raw = self._level_snapped(self._p3(pt), skip_base=True)  # elevation: the nearest story level
        q = self._offset_point(raw)
        dz = q[2] - raw[2]  # Offset Z moves the whole column (both ends)
        p = (q[0], q[1], raw[2])
        sec = self._frame_section()
        angle = self.settings.column_angle
        cardinal = self._cardinal_extra("Column")
        if self.plane.kind == "plan" and self.story is None:
            # free plan (any Z): the column runs from the level below the plane up to it
            top = p[2]
            lv = [z for _, z in self.model.levels() if z < top - self.tol]
            bot = max(lv) if lv else None
            if bot is None:
                raise ModelError("NO_LEVEL_BELOW", "No level below this plane for the column to start from")

            def mutate_free(model: Model):
                f = model.add_frame(model.add_point(p[0], p[1], bot + dz), model.add_point(p[0], p[1], top + dz), sec,
                                    type="Column", angle=angle, **cardinal)
                self.created.append(f.name)
            self.doc.edit(mutate_free)
            return True
        story = self.story if self.plane.kind == "plan" else self._require_story(p[2])
        stories = self.story_names_to_draw(story) if self.plane.kind == "plan" else [story]

        def mutate(model: Model):
            for st in stories:
                below = model.story_below(st)
                if below is None:
                    continue
                top, bot = model.stories[st].elevation, below.elevation
                if any(f.type == "Column" and f.story == st and _same_xy(model, f, p) for f in model.frames.values()):
                    continue
                f = model.add_frame(model.add_point(p[0], p[1], bot + dz), model.add_point(p[0], p[1], top + dz), sec,
                                    type="Column", story=st if not dz else None, angle=angle,
                                    **cardinal)
                self.created.append(f.name)

        self.doc.edit(mutate)
        return True


def _same_xy(model: Model, f: Frame, p, tol: float = 1e-6) -> bool:
    a = model.points[f.point_i]
    return abs(a.x - p[0]) <= tol and abs(a.y - p[1]) <= tol


# ---------------------------------------------------------------- draw wall
class WallTool(Tool):
    """Two clicks: wall panel between the points, from the story below up to
    the story of the clicked level (plan: the current story)."""

    title = "Draw Walls"
    next_prompt = "click the next point of the wall"
    name = "wall"
    prompt = "Click the start of the wall"

    def resolve(self, pt) -> XYZ:
        # elevation: the nearest story level (a wall spans whole stories)
        return self._apply_drawing_control(self._level_snapped(self._p3(pt)))

    def _spans(self, a, b) -> List[str]:
        """Stories the wall between picks a and b fills (each from the story below up
        to it).  Plan: the plan's story (and All stories).  Elevation / 3D: both picks
        at one level - the story below it; at two levels - every story between them."""
        if self.plane.kind == "plan":
            return self.story_names_to_draw(self.story) if self.story else []
        lo, hi = min(a[2], b[2]), max(a[2], b[2])
        if hi - lo <= self.tol:
            return [self._require_story(hi)]
        m = self.model
        levels = [z for _, z in m.levels()]
        for z in (lo, hi):
            if not any(abs(z - lv) <= self.tol for lv in levels):
                raise ModelError("NOT_AT_STORY", f"Wall: Z = {z:g} is not a story level - pick both ends at "
                                                 f"story levels (the wall spans whole stories)")
        out = []
        for st in m.stories_bottom_up():
            below = m.story_below(st.name)
            if below is None:
                continue
            if st.elevation > lo + self.tol and st.elevation <= hi + self.tol and below.elevation >= lo - self.tol:
                out.append(st.name)
        if not out:
            raise ModelError("NOT_AT_STORY", "Wall: no story between the two picked levels")
        return out

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        raw = self.resolve(pt)
        self.points.append(raw)
        self._raw_last = raw
        if len(self.points) < 2:
            self.prompt = "Click the end of the wall"
            return False
        a, b = self.points[-2], self.points[-1]
        self.points = [b]
        if math.hypot(a[0] - b[0], a[1] - b[1]) <= self.tol:
            return False
        sec = self._shell_section(wall=True)
        stories = self._spans(a, b)
        ox, oy, oz = self._offset_point((0.0, 0.0, 0.0))
        a2, b2 = (a[0] + ox, a[1] + oy, a[2]), (b[0] + ox, b[1] + oy, b[2])

        def mutate(model: Model):
            for st in stories:
                w = _add_wall(model, a2, b2, st, sec, dz=oz)
                if w:
                    self.created.append(w)

        self.doc.edit(mutate)
        return True

    def preview(self, cursor) -> Preview:
        """The outline of the wall(s) the next click makes, not only the top line."""
        pv = self._line_preview(cursor)
        if not self.points or cursor is None:
            return pv
        a, b = self.points[-1], self.resolve(cursor)
        if math.hypot(a[0] - b[0], a[1] - b[1]) <= self.tol:
            return pv
        try:
            stories = self._spans(a, b)
        except ModelError as exc:
            pv.text = (pv.text + "   " if pv.text else "") + exc.message
            return pv
        m = self.model
        rail = pv.lines[-1:] if pv.lines else []
        outline = []
        for st in stories:
            below = m.story_below(st)
            if below is None:
                continue
            ox, oy, oz = self._offset_point((0.0, 0.0, 0.0))
            top, bot = m.stories[st].elevation + oz, below.elevation + oz
            q = [(a[0] + ox, a[1] + oy, bot), (b[0] + ox, b[1] + oy, bot), (b[0] + ox, b[1] + oy, top),
                 (a[0] + ox, a[1] + oy, top)]
            outline += [(q[k], q[(k + 1) % 4]) for k in range(4)]
        pv.lines = outline + rail  # the outline of every wall, then the line to the cursor
        pv.text += f"   wall: {', '.join(stories)}" if stories else ""
        return pv


def _add_wall(model: Model, a, b, story: str, section: str, dz: float = 0.0) -> Optional[str]:
    below = model.story_below(story)
    if below is None:
        return None
    top, bot = model.stories[story].elevation + dz, below.elevation + dz
    pts = [model.add_point(a[0], a[1], bot), model.add_point(b[0], b[1], bot),
           model.add_point(b[0], b[1], top), model.add_point(a[0], a[1], top)]
    names = {p.name for p in pts}
    if any(set(ar.points) == names for ar in model.areas.values()):
        return None
    return model.add_area(pts, section, type="Wall", story=story if not dz else None).name


# ---------------------------------------------------------------- draw slab
class SlabTool(Tool):
    """Polygon floor / wall / ramp / opening: click the corners, double-click / Enter
    to close.

    * plan: a floor at the story (All stories repeats it), corners in the plan;
    * elevation: the polygon lies in the elevation plane (a wall by default);
    * 3D: the corners are snapped points; they must lie in one plane (checked before
      anything is made), the object type follows the plane unless the form says
      otherwise (horizontal: floor, vertical: wall, sloped: ramp).

    Holes are separate Opening objects drawn in the same plane (the analysis mesh
    cuts them out)."""

    title = "Draw Floor / Wall"
    next_prompt = "click the next corner"
    name = "slab"
    prompt = "Click the corners; Enter / right-click / double-click closes the outline"
    views = ("plan", "xz", "yz", "3d")
    area_type: Optional[str] = None  # fixed by a subclass (Opening); else from the form (Auto ...)
    closes_polygon = True  # Enter / right-click: the last corner joins the first (no need to click it)

    # ---- the object the outline makes
    def _requested_type(self) -> str:
        if type(self).area_type:
            return type(self).area_type
        if self._uses("area_type"):
            return getattr(self.settings, "area_type", "Auto") or "Auto"
        return "Auto"  # the form does not offer a type: floor in plan, wall in an elevation

    def _area_kind(self, corners3) -> str:
        """Floor / Wall / Ramp / Opening for these corners; refuses a contradiction
        (a horizontal wall, a vertical floor) instead of making something else."""
        want = self._requested_type()
        n = _polygon_normal(corners3)
        nz = abs(float(n[2])) if n is not None else 1.0
        horizontal, vertical = nz > 0.999, nz < 0.02
        if want == "Opening":
            return "Opening"
        if want == "Auto":
            return "Floor" if horizontal else "Wall" if vertical else "Ramp"
        if want == "Wall" and horizontal:
            raise ModelError("BAD_AREA_TYPE", "A wall is vertical: draw it in an elevation or 3D window "
                                              "(or choose Floor / Auto)")
        if want == "Floor" and not horizontal:
            raise ModelError("BAD_AREA_TYPE", "A floor is horizontal: these corners are not "
                                              "(choose Wall / Ramp / Auto)")
        return want

    def _section_for_kind(self, kind: str) -> Optional[str]:
        if kind == "Opening":
            return None
        return self._shell_section(wall=(kind == "Wall"))

    def _normal_shift(self) -> Tuple[float, float, float]:
        """The form's Offset X / Y / Z: a shift of the whole outline."""
        return self._offset_point((0.0, 0.0, 0.0))

    def _shifted(self, poly):
        d = self._normal_shift()
        return [(p[0] + d[0], p[1] + d[1], p[2] + d[2]) for p in poly]

    # ---- picking
    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self._p3(pt)
        if self.points and len(self.points) >= 3 and _dist(p, self.points[0]) <= self.tol:
            return self.finish()
        self.points.append(p)
        return False

    def double_click(self, pt) -> bool:
        p = self._p3(pt)
        if not (self.points and _dist(p, self.points[0]) <= self.tol) and \
                not (self.points and _dist(p, self.points[-1]) <= self.tol):
            self.points.append(p)
        return self.finish()

    def finish(self) -> bool:
        if len(self.points) < 3:
            self.points = []
            return False
        pts = list(self.points)
        self.points = []
        if self.plane.kind == "plan":
            return self._create([(p[0], p[1]) for p in pts])
        return self._create_planar(pts)

    # ---- making it
    def _create(self, corners: Sequence[XY]) -> bool:
        """Plan outline: at the story (repeated on All stories) or at the free plan's Z."""
        z_ref = self.plane.value
        kind = self._area_kind([(c[0], c[1], z_ref) for c in corners])
        sec = self._section_for_kind(kind)
        stories = self.story_names_to_draw()
        diaphragm = self.settings.diaphragm if kind == "Floor" else None
        ox, oy, dz = self._normal_shift()
        corners = [(c[0] + ox, c[1] + oy) for c in corners]
        z_free = self.plane.value if (self.plane.kind == "plan" and self.story is None) else None
        if not stories and z_free is None:
            raise ModelError("NOTHING_TO_DRAW", "No story to draw the floor on")

        def mutate(model: Model):
            for st in (stories or [None]):
                if st is None:
                    pts = [model.add_point(c[0], c[1], z_free + dz) for c in corners]
                    a = model.add_area(pts, sec, type=kind, diaphragm=diaphragm)
                    self.created.append(a.name)
                    continue
                z = model.stories[st].elevation + dz
                pts = [model.add_point(c[0], c[1], z) for c in corners]
                names = {p.name for p in pts}
                if any(set(ar.points) == names for ar in model.areas.values()):
                    continue
                a = model.add_area(pts, sec, type=kind, story=st if not dz else None, diaphragm=diaphragm)
                self.created.append(a.name)

        self.doc.edit(mutate)
        return True

    def _create_planar(self, pts) -> bool:
        """Elevation / 3D outline: one object through the corners, which must lie in
        one plane (within the window's plane band)."""
        P = np.asarray(pts, float) + np.asarray(self._normal_shift(), float)
        problem = _planarity_problem(P, plane_band(self.model))
        if problem:
            raise ModelError("NOT_PLANAR", problem)
        kind = self._area_kind(P)
        sec = self._section_for_kind(kind)
        diaphragm = self.settings.diaphragm if kind == "Floor" else None

        def mutate(model: Model):
            q = [model.add_point(*c) for c in P.tolist()]
            names = {x.name for x in q}
            if any(set(ar.points) == names for ar in model.areas.values()):
                return
            a = model.add_area(q, sec, type=kind, diaphragm=diaphragm)
            self.created.append(a.name)

        self.doc.edit(mutate)
        return True

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        poly = list(self.points)
        if cursor is not None and self.points:
            poly.append(self._p3(cursor))
        pv.polygon = self._shifted(poly)
        if len(poly) >= 3 and self.plane.kind != "plan":
            problem = _planarity_problem(np.asarray(poly, float), plane_band(self.model))
            try:
                kind = self._area_kind(poly)
            except ModelError as exc:
                kind, problem = None, problem or exc.message
            pv.text = problem or f"{kind}: {len(poly)} corners"
        return pv


def _polygon_normal(points) -> Optional[np.ndarray]:
    """Unit normal of a polygon (Newell's method), None if degenerate."""
    P = np.asarray(points, float)
    if len(P) < 3:
        return None
    n = np.zeros(3)
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    L = float(np.linalg.norm(n))
    return n / L if L > 1e-12 else None


def _planarity_problem(P: np.ndarray, tol: float) -> Optional[str]:
    """Why these corners cannot make one flat area, or None."""
    if len(P) < 3:
        return "An area needs at least three corners"
    n = _polygon_normal(P)
    if n is None:
        return "The corners are on one line: no area"
    off = np.abs((P - P.mean(axis=0)) @ n)
    if float(off.max()) > tol:
        k = int(np.argmax(off))
        return (f"The corners are not in one plane: corner {k + 1} is {float(off[k]):.3g} out of it "
                f"(snap the corners in one plane)")
    return None


class OpeningTool(SlabTool):
    """Openings (holes in floors and walls; the analysis mesh leaves them out).

    Rectangle (default): click two opposite corners, or click one corner and type
    ``width,height`` - in plan for shafts and stairs, in an elevation for doors
    and windows.  Polygon: click the corners, double-click / Enter to close."""

    title = "Draw Openings"
    next_prompt = "click the opposite corner (or type width,height)"
    name = "opening"
    prompt = "Click a corner of the opening (rectangle), or the corners (polygon)"
    area_type = "Opening"
    views = ("plan", "xz", "yz")

    @property
    def rectangle(self) -> bool:
        return getattr(self.settings, "opening_shape", "Rectangle") != "Polygon"

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        if not self.rectangle:
            return super().click(pt, snap)
        p = self._p3(pt)
        if not self.points:
            self.points = [p]
            return False
        a = self.points[0]
        self.points = []
        corners = self._rect(a, p)
        if corners is None:
            return False
        return self._create3(corners)

    def finish(self) -> bool:
        if self.rectangle:
            return False
        if len(self.points) < 3:
            self.points = []
            return False
        corners = list(self.points)
        self.points = []
        return self._create3(corners)

    def _rect(self, a, c):
        (u0, v0), (u1, v1) = self.plane.to2(a), self.plane.to2(c)
        if abs(u1 - u0) <= self.tol or abs(v1 - v0) <= self.tol:
            return None
        return [self.plane.to3(q) for q in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]

    def _create3(self, corners) -> bool:
        if self.plane.kind == "plan":
            return self._create([(c[0], c[1]) for c in corners])
        return self._create_planar(corners)  # elevation: one opening in the wall plane

    def preview(self, cursor) -> Preview:
        if not self.rectangle:
            return super().preview(cursor)
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if self.points and cursor is not None:
            a, c = self.points[0], self._p3(cursor)
            q = self._rect(a, c)
            if q is not None:
                pv.polygon = q
                (u0, v0), (u1, v1) = self.plane.to2(a), self.plane.to2(c)
                pv.text = f"{abs(u1 - u0):.3f} x {abs(v1 - v0):.3f}"
        return pv


class RectSlabTool(SlabTool):
    """Rectangle from two opposite corners: a floor in plan, a wall panel (or the
    form's type) in an elevation."""

    title = "Draw Rectangular Floor / Wall"
    next_prompt = "click the opposite corner"
    name = "rect_slab"
    prompt = "Click the first corner"
    views = ("plan", "xz", "yz")

    def _corners(self, a, c):
        (u0, v0), (u1, v1) = self.plane.to2(a), self.plane.to2(c)
        if abs(u1 - u0) <= self.tol or abs(v1 - v0) <= self.tol:
            return None
        return [self.plane.to3(q) for q in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self._p3(pt)
        if not self.points:
            self.points = [p]
            self.prompt = "Click the opposite corner"
            return False
        a = self.points[0]
        self.points = []
        self.prompt = "Click the first corner"
        q = self._corners(a, p)
        if q is None:
            return False
        if self.plane.kind == "plan":
            return self._create([(c[0], c[1]) for c in q])
        return self._create_planar(q)

    def finish(self) -> bool:
        return False

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if self.points and cursor is not None:
            a, c = self.points[0], self._p3(cursor)
            q = self._corners(a, c)
            if q is not None:
                pv.polygon = self._shifted(q)
                (u0, v0), (u1, v1) = self.plane.to2(a), self.plane.to2(c)
                pv.text = f"{abs(u1 - u0):.3f} x {abs(v1 - v0):.3f}"
        return pv


# --------------------------------------------------------------- quick draw
def _plan_segment_at(model: Model, xy: XY, tol: float) -> Optional[Tuple[XY, XY]]:
    xs, ys = grid_ordinates(model)
    best, best_d = None, tol
    for x in xs:  # segments along Y on grid line X
        for y0, y1 in zip(ys, ys[1:]):
            d, _ = _seg_point_distance(xy, (x, y0), (x, y1))
            if d < best_d:
                best, best_d = ((x, y0), (x, y1)), d
    for y in ys:
        for x0, x1 in zip(xs, xs[1:]):
            d, _ = _seg_point_distance(xy, (x0, y), (x1, y))
            if d < best_d:
                best, best_d = ((x0, y), (x1, y)), d
    return best


def _plan_cell_at(model: Model, xy: XY) -> Optional[List[XY]]:
    xs, ys = grid_ordinates(model)
    for x0, x1 in zip(xs, xs[1:]):
        if not (x0 <= xy[0] <= x1):
            continue
        for y0, y1 in zip(ys, ys[1:]):
            if y0 <= xy[1] <= y1:
                return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return None


def _elev_bay_at(model: Model, plane: Plane, uv: XY) -> Optional[Tuple[float, float, float, float]]:
    us = plane.u_ordinates(model)
    zs = story_levels(model)
    for u0, u1 in zip(us, us[1:]):
        if not (u0 <= uv[0] <= u1):
            continue
        for z0, z1 in zip(zs, zs[1:]):
            if z0 <= uv[1] <= z1:
                return u0, u1, z0, z1
    return None


def _has_frame(model: Model, pi: Point, pj: Point) -> bool:
    return any({f.point_i, f.point_j} == {pi.name, pj.name} for f in model.frames.values())


class QuickBeamTool(Tool):
    """Plan: click near a grid segment to add a beam on it.  Elevation: click
    near a story line between two grid lines."""

    title = "Quick Draw Beams"
    name = "quick_beam"
    prompt = "Click a grid line segment to draw a beam on it"
    views = ("plan", "xz", "yz")

    def segment_at(self, pt, tol: float) -> Optional[Tuple[XYZ, XYZ]]:
        if self.plane.kind == "plan":
            seg = _plan_segment_at(self.model, self.plane.to2(pt), tol)
            if seg is None:
                return None
            z = self.elevation()
            return (seg[0][0], seg[0][1], z), (seg[1][0], seg[1][1], z)
        u, v = self.plane.to2(pt)
        us = self.plane.u_ordinates(self.model)
        best, best_d = None, tol
        for z in story_levels(self.model)[1:]:
            for u0, u1 in zip(us, us[1:]):
                d, _ = _seg_point_distance((u, v), (u0, z), (u1, z))
                if d < best_d:
                    best, best_d = (self.plane.to3((u0, z)), self.plane.to3((u1, z))), d
        return best

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        seg = self.segment_at(pt, self.tolerance)
        if seg is None:
            return False
        a, b = seg
        sec = self._frame_section()
        rel_i, rel_j = self._beam_releases()
        if self.plane.kind == "plan":
            stories = self.story_names_to_draw()
        else:
            stories = [self._require_story(a[2])]

        def mutate(model: Model):
            for st in stories:
                z = model.stories[st].elevation
                pi, pj = model.add_point(a[0], a[1], z), model.add_point(b[0], b[1], z)
                if _has_frame(model, pi, pj):
                    continue
                self.created.append(model.add_frame(pi, pj, sec, type="Beam", story=st,
                                                    releases_i=rel_i, releases_j=rel_j).name)

        self.doc.edit(mutate)
        return True

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is not None:
            seg = self.segment_at(cursor, self.tolerance)
            if seg:
                pv.lines.append(seg)
        return pv


class QuickColumnTool(ColumnTool):
    """Plan: click near a grid intersection.  Elevation: click near a grid line
    between two story levels."""

    title = "Quick Draw Columns"
    name = "quick_column"
    prompt = "Click a grid intersection to draw a column"
    views = ("plan", "xz", "yz")

    def target(self, pt) -> Optional[XYZ]:
        if self.plane.kind == "plan":
            pts = grid_intersections(self.model)
            if not pts:
                return None
            xy = self.plane.to2(pt)
            g = min(pts, key=lambda q: _dist(q, xy))
            if _dist(g, xy) > self.tolerance:
                return None
            return (g[0], g[1], self.elevation())
        u, v = self.plane.to2(pt)
        us = self.plane.u_ordinates(self.model)
        zs = story_levels(self.model)
        for z0, z1 in zip(zs, zs[1:]):
            if z0 <= v <= z1:
                cands = [x for x in us if abs(x - u) <= self.tolerance]
                if cands:
                    uu = min(cands, key=lambda x: abs(x - u))
                    return self.plane.to3((uu, z1))
        return None

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        t = self.target(pt)
        if t is None:
            return False
        return super().click(t, snap)

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is not None:
            t = self.target(cursor)
            if t is not None:
                st = self._story_at(t[2])
                below = self.model.story_below(st) if st else None
                if below is not None:
                    pv.lines.append(((t[0], t[1], below.elevation), t))
                pv.marker = t
        return pv


class QuickWallTool(Tool):
    """Plan: click a grid segment -> wall below the current story.
    Elevation: click inside a bay -> wall panel filling it."""

    title = "Quick Draw Walls"
    name = "quick_wall"
    prompt = "Click a grid segment (plan) or a bay (elevation) to add a wall"
    views = ("plan", "xz", "yz")

    def _panel(self, pt):
        if self.plane.kind == "plan":
            seg = _plan_segment_at(self.model, self.plane.to2(pt), self.tolerance)
            if seg is None or not self.story:
                return None
            return (seg[0], seg[1], self.story_names_to_draw())
        bay = _elev_bay_at(self.model, self.plane, self.plane.to2(pt))
        if bay is None:
            return None
        u0, u1, z0, z1 = bay
        a, b = self.plane.to3((u0, z1)), self.plane.to3((u1, z1))
        return ((a[0], a[1]), (b[0], b[1]), [self._require_story(z1)])

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        panel = self._panel(pt)
        if panel is None:
            return False
        a, b, stories = panel
        sec = self._shell_section(wall=True)

        def mutate(model: Model):
            for st in stories:
                w = _add_wall(model, a, b, st, sec)
                if w:
                    self.created.append(w)

        self.doc.edit(mutate)
        return True

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is not None:
            panel = self._panel(cursor)
            if panel:
                a, b, stories = panel
                st = stories[0]
                below = self.model.story_below(st)
                if below is not None:
                    top, bot = self.model.stories[st].elevation, below.elevation
                    pv.polygon = [(a[0], a[1], bot), (b[0], b[1], bot), (b[0], b[1], top), (a[0], a[1], top)]
        return pv


class QuickBraceTool(Tool):
    """Elevation: click inside a bay to add braces (pattern from the settings)."""

    title = "Quick Draw Braces"
    name = "quick_brace"
    prompt = "Click inside a bay of an elevation view to add braces"
    views = ("xz", "yz")

    def _members(self, pt) -> List[Tuple[XYZ, XYZ]]:
        if self.plane.kind not in ("xz", "yz"):
            return []
        bay = _elev_bay_at(self.model, self.plane, self.plane.to2(pt))
        if bay is None:
            return []
        u0, u1, z0, z1 = bay
        P = self.plane.to3
        um = 0.5 * (u0 + u1)
        pattern = self.settings.brace_pattern
        if pattern == "/":
            return [(P((u0, z0)), P((u1, z1)))]
        if pattern == "\\":
            return [(P((u1, z0)), P((u0, z1)))]
        if pattern == "V":
            return [(P((u0, z1)), P((um, z0))), (P((um, z0)), P((u1, z1)))]
        if pattern.lower().startswith("inv"):
            return [(P((u0, z0)), P((um, z1))), (P((um, z1)), P((u1, z0)))]
        return [(P((u0, z0)), P((u1, z1))), (P((u1, z0)), P((u0, z1)))]

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        members = self._members(pt)
        if not members:
            return False
        sec = self._brace_section()
        rel_i, rel_j = self._beam_releases()  # the form's Moment releases

        def mutate(model: Model):
            for a, b in members:
                pi, pj = model.add_point(*a), model.add_point(*b)
                if _has_frame(model, pi, pj):
                    continue
                self.created.append(model.add_frame(pi, pj, sec, type="Brace", releases_i=rel_i,
                                                    releases_j=rel_j).name)

        self.doc.edit(mutate)
        return True

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is not None:
            pv.lines = self._members(cursor)
        return pv


class QuickSlabTool(SlabTool):
    """Click inside a cell: a floor bounded by the grid lines (Bounded by = Grid) or
    by the closed loop of beams around the click (Bounded by = Beams); in an
    elevation, a panel filling the grid / story bay."""

    title = "Quick Draw Floor"
    name = "quick_slab"
    prompt = "Click inside a cell (grid or beam loop) to draw a floor"
    views = ("plan", "xz", "yz")

    def cell_at(self, pt) -> Optional[List[XY]]:
        if self.plane.kind != "plan":
            bay = _elev_bay_at(self.model, self.plane, self.plane.to2(pt))
            if bay is None:
                return None
            u0, u1, z0, z1 = bay
            return [(u0, z0), (u1, z0), (u1, z1), (u0, z1)]
        if getattr(self.settings, "quick_floor_edge", "Grid") == "Beams":
            return _beam_loop_at(self.model, self.plane.to2(pt), self.plane.value, self.tol)
        return _plan_cell_at(self.model, self.plane.to2(pt))

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        cell = self.cell_at(pt)
        if cell is None:
            if getattr(self.settings, "quick_floor_edge", "Grid") == "Beams" and self.plane.kind == "plan":
                raise ModelError("NO_BEAM_LOOP", "No closed loop of beams around this point at this level")
            return False
        if self.plane.kind != "plan":
            return self._create_planar([self.plane.to3(c) for c in cell])
        return self._create(cell)

    def finish(self) -> bool:
        return False

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is not None:
            cell = self.cell_at(cursor)
            if cell:
                if self.plane.kind != "plan":
                    pv.polygon = self._shifted([self.plane.to3(c) for c in cell])
                else:
                    z = self.elevation()
                    pv.polygon = self._shifted([(c[0], c[1], z) for c in cell])
            elif getattr(self.settings, "quick_floor_edge", "Grid") == "Beams" and self.plane.kind == "plan":
                pv.text = "no closed loop of beams around the cursor"
        return pv


def _beam_loop_at(model: Model, xy: XY, z: float, tol: float) -> Optional[List[XY]]:
    """The smallest closed loop of horizontal frames at level z enclosing xy (the
    faces of the plan graph of the beams; beams meet at joints)."""
    pts = model.points
    edges = set()
    for f in model.frames.values():
        a, b = pts[f.point_i], pts[f.point_j]
        if abs(a.z - z) <= tol and abs(b.z - z) <= tol and f.point_i != f.point_j:
            edges.add((f.point_i, f.point_j))
            edges.add((f.point_j, f.point_i))
    if not edges:
        return None
    out = {}
    for u, v in edges:
        out.setdefault(u, []).append(v)
    P = {n: (pts[n].x, pts[n].y) for n in out}

    def ang(u, v):
        return math.atan2(P[v][1] - P[u][1], P[v][0] - P[u][0])

    for u in out:  # neighbours counter-clockwise
        out[u].sort(key=lambda v: ang(u, v))
    used, best = set(), None
    for e in edges:
        if e in used:
            continue
        face, (u, v) = [], e
        for _ in range(len(edges) + 1):
            used.add((u, v))
            face.append(u)
            nb = out[v]
            k = nb.index(u)
            w = nb[(k - 1) % len(nb)]  # the next edge turning most to the left of (u -> v)
            u, v = v, w
            if (u, v) == e:
                break
        else:
            continue
        poly = [P[n] for n in face]
        area = 0.5 * sum(poly[k][0] * poly[(k + 1) % len(poly)][1] - poly[(k + 1) % len(poly)][0] * poly[k][1]
                         for k in range(len(poly)))
        if area <= tol * tol or len(poly) < 3 or not point_in_polygon(xy, poly):
            continue
        if best is None or area < best[0]:
            best = (area, poly)
    return None if best is None else [tuple(map(float, c)) for c in best[1]]


class SecondaryBeamTool(Tool):
    """Click inside a grid cell: equally spaced secondary beams (pinned ends),
    parallel to X or Y (ETABS "Quick Draw Secondary Beams")."""

    title = "Secondary Beams"
    name = "secondary_beams"
    prompt = "Click inside a grid cell to add secondary beams"
    views = ("plan",)

    def _members(self, pt) -> List[Tuple[XYZ, XYZ]]:
        cell = _plan_cell_at(self.model, self.plane.to2(pt))
        if cell is None:
            return []
        (x0, y0), (x1, _), (_, y1), _ = cell
        n = max(1, int(self.settings.secondary_count))
        z = self.elevation()
        out = []
        for k in range(1, n + 1):
            t = k / (n + 1)
            if self.settings.secondary_direction.upper() == "X":
                y = y0 + (y1 - y0) * t
                out.append(((x0, y, z), (x1, y, z)))
            else:
                x = x0 + (x1 - x0) * t
                out.append(((x, y0, z), (x, y1, z)))
        return out

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        members = self._members(pt)
        if not members:
            return False
        sec = self._frame_section()
        stories = self.story_names_to_draw()
        z0 = self.elevation()

        def mutate(model: Model):
            for st in stories:
                dz = model.stories[st].elevation - z0
                for a, b in members:
                    pi = model.add_point(a[0], a[1], a[2] + dz)
                    pj = model.add_point(b[0], b[1], b[2] + dz)
                    if _has_frame(model, pi, pj):
                        continue
                    self.created.append(model.add_frame(pi, pj, sec, type="Beam", story=st,
                                                        releases_i=PIN_ENDS, releases_j=PIN_ENDS).name)

        self.doc.edit(mutate)
        return True

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is not None:
            pv.lines = self._members(cursor)
        return pv


class MeasureTool(Tool):
    """*View > Measure > Line / Area / Angle*: click points, the result goes to
    the status bar (Area: double-click the last point)."""

    title = "Measure"
    next_prompt = "click the next point"
    name = "measure_line"
    kind = "line"
    needed = 2
    prompt = "Measure Line: click two points"

    def __init__(self, doc: Document, settings: DrawSettings):
        super().__init__(doc, settings)
        self.result = ""

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        from .commands import measure
        self.points.append(self._p3(pt))
        if self.needed and len(self.points) >= self.needed:
            self.result = measure(self.points, self.kind)
            self.prompt = f"{self.result}   - click to measure again"
            self.points = []
            return True
        self.prompt = f"Measure {self.kind.title()}: point {len(self.points) + 1}"
        return False

    def finish(self) -> bool:
        from .commands import measure
        if self.kind == "area" and len(self.points) >= 3:
            self.result = measure(self.points, "area")
            self.prompt = f"{self.result}   - click to measure again"
            self.points = []
            return True
        return False

    def double_click(self, pt) -> bool:
        if self.kind == "area":
            return self.finish()
        return False

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        pts = list(self.points) + ([self._p3(cursor)] if cursor is not None else [])
        pv.lines = [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
        if self.kind == "area" and len(pts) >= 3:
            pv.lines.append((pts[-1], pts[0]))
        if self.points and cursor is not None:
            from .commands import measure
            pv.text = measure(pts, self.kind if self.kind != "angle" or len(pts) >= 3 else "line")
        return pv


class MeasureAreaTool(MeasureTool):
    name = "measure_area"
    kind = "area"
    needed = 0
    prompt = "Measure Area: click the corners, double-click (or Enter) on the last one"


class MeasureAngleTool(MeasureTool):
    name = "measure_angle"
    kind = "angle"
    needed = 3
    prompt = "Measure Angle: click three points, the second is the vertex"


TOOLS: Dict[str, type] = {
    "select": SelectTool,
    "select_line": SelectLineTool,
    "point": PointTool,
    "beam": BeamTool,
    "quick_beam": QuickBeamTool,
    "secondary_beams": SecondaryBeamTool,
    "column": ColumnTool,
    "quick_column": QuickColumnTool,
    "brace": BraceTool,
    "quick_brace": QuickBraceTool,
    "wall": WallTool,
    "quick_wall": QuickWallTool,
    "slab": SlabTool,
    "rect_slab": RectSlabTool,
    "quick_slab": QuickSlabTool,
    "opening": OpeningTool,
    "measure_line": MeasureTool,
    "measure_area": MeasureAreaTool,
    "measure_angle": MeasureAngleTool,
}


# ------------------------------------------------------------- selection ops
def delete_objects(doc: Document, frames: Sequence[str] = (), areas: Sequence[str] = (),
                   points: Sequence[str] = ()) -> int:
    """Delete objects; points are only removed when nothing references them."""
    frames, areas, points = set(frames), set(areas), set(points)
    count = len(frames) + len(areas)

    def mutate(model: Model):
        for n in frames:
            model.frames.pop(n, None)
        for n in areas:
            model.areas.pop(n, None)
        for pn in [p.name for p in model.panel_loads.values() if p.area in areas]:
            del model.panel_loads[pn]  # area loads (panels made from the areas) go with them
        used = {f.point_i for f in model.frames.values()} | {f.point_j for f in model.frames.values()}
        for a in model.areas.values():
            used |= set(a.points)
        for n in points:
            p = model.points.get(n)
            if p is not None and n not in used and not any(p.restraints) and not p.loads:
                model.points.pop(n)
        for g in model.groups.values():
            g.frames = [f for f in g.frames if f in model.frames]
            g.areas = [a for a in g.areas if a in model.areas]
            g.points = [p for p in g.points if p in model.points]

    if count or points:
        doc.edit(mutate)
    return count


def objects_in_window(model: Model, inside, story: Optional[str] = None) -> Tuple[List[str], List[str], List[str]]:
    """Window selection: ``inside(point) -> bool`` decides for each model point.
    Frames/areas are selected when all their points are inside (ETABS window rule)."""
    pts = {}
    for p in model.points.values():
        if story is None or p.story == story or any(
                f.story == story for f in model.frames.values() if p.name in (f.point_i, f.point_j)):
            pts[p.name] = inside(p)
    frames = [f.name for f in model.frames.values() if pts.get(f.point_i) and pts.get(f.point_j)]
    areas = [a.name for a in model.areas.values() if all(pts.get(n) for n in a.points)]
    points = [n for n, ok in pts.items() if ok]
    return frames, areas, points


# ------------------------------------------------------------ edit commands
def _copy_frame(model: Model, f: Frame, pi: Point, pj: Point, story: Optional[str]) -> Frame:
    import dataclasses
    kw = {k: v for k, v in dataclasses.asdict(f).items()
          if k not in ("name", "point_i", "point_j", "section", "type", "story", "loads", "label")}
    # dataclass children need real objects, not dicts
    kw["releases_i"], kw["releases_j"] = f.releases_i, f.releases_j
    kw["modifiers"] = copy.deepcopy(f.modifiers)
    kw["hinges"] = copy.deepcopy(f.hinges)
    new = model.add_frame(pi, pj, f.section, type=f.type, story=story, **kw)
    new.loads = [dataclasses.replace(ld) for ld in f.loads]
    return new


def _copy_area(model: Model, a: Area, pts: List[Point], story: Optional[str]) -> Area:
    import dataclasses
    new = model.add_area(pts, a.section, type=a.type, story=story, diaphragm=a.diaphragm, angle=a.angle,
                         modifiers=a.modifiers, mass_per_area=a.mass_per_area)
    new.loads = [dataclasses.replace(ld) for ld in a.loads]
    from ..model.area_panels import copy_area_panels
    copy_area_panels(model, a.name, new.name)  # its area loads (panels) come along
    return new


def move_panels(doc: Document, panels: Sequence[str], dx: float, dy: float, dz: float) -> int:
    """Move drawn load panels (free panels are load areas of their own: they move
    only when the user moves them; area-load panels follow their area)."""
    names = [n for n in panels if n in doc.model.panel_loads and not doc.model.panel_loads[n].area]
    if not names:
        return 0

    def mutate(model: Model):
        for n in names:
            p = model.panel_loads[n]
            p.points = [(x + dx, y + dy, z + dz) for x, y, z in p.points]

    doc.edit(mutate)
    return len(names)


def move_objects(doc: Document, frames: Sequence[str], areas: Sequence[str], points: Sequence[str],
                 dx: float, dy: float, dz: float) -> int:
    """Move the selected objects (their points) by (dx, dy, dz); points shared
    with unselected objects are moved too (ETABS "Move" behaviour)."""
    names = set(points)
    for n in frames:
        f = doc.model.frames[n]
        names |= {f.point_i, f.point_j}
    for n in areas:
        names |= set(doc.model.areas[n].points)
    if not names:
        return 0

    def mutate(model: Model):
        for n in names:
            p = model.points[n]
            p.x, p.y, p.z = p.x + dx, p.y + dy, p.z + dz
            s = model.story_at(p.z, model.options.merge_tolerance)
            p.story = s.name if s else p.story
        for f in model.frames.values():
            top = max(model.points[f.point_i].z, model.points[f.point_j].z)
            s = model.story_at(top, model.options.merge_tolerance)
            f.story = s.name if s else f.story
        for a in model.areas.values():
            top = max(model.points[p].z for p in a.points)
            s = model.story_at(top, model.options.merge_tolerance)
            a.story = s.name if s else a.story

    doc.edit(mutate)
    return len(names)


def replicate_linear(doc: Document, frames: Sequence[str], areas: Sequence[str], points: Sequence[str],
                     dx: float, dy: float, dz: float, count: int = 1) -> List[str]:
    """Copy the selection ``count`` times along (dx, dy, dz) (joints with all their
    assignments - see :func:`commands.replicate_transform`)."""
    from .commands import linear_transforms, replicate_transform
    return replicate_transform(doc, frames, areas, points, linear_transforms(dx, dy, dz, count))


def replicate_to_stories(doc: Document, frames: Sequence[str], areas: Sequence[str], points: Sequence[str],
                         stories: Sequence[str], delete_original: bool = False) -> List[str]:
    """Copy the selection to other stories (ETABS "Replicate > Story"): joints
    with all their assignments, members with theirs; ``delete_original`` removes
    the copied objects afterwards (one undo step)."""
    created: List[str] = []
    frames, areas, points = list(frames), list(areas), list(points)
    m = doc.model
    src_story = None
    for n in frames + areas:
        obj = m.frames.get(n) or m.areas.get(n)
        if obj is not None and obj.story:
            src_story = obj.story
            break
    if src_story is None:
        for n in points:
            if m.points[n].story:
                src_story = m.points[n].story
                break
    if src_story is None or src_story not in m.stories:
        return created
    z0 = m.stories[src_story].elevation
    from ..model.point_ops import clone_assignments, has_assignments
    from .commands import _delete_originals
    notes: List[str] = []

    def mutate(model: Model):
        kept = []
        targets = [st for st in stories if st != src_story and st in model.stories]
        for st in targets:
            dz = model.stories[st].elevation - z0
            cache: Dict[str, Point] = {}

            def moved(name: str) -> Point:
                if name not in cache:
                    p = model.points[name]
                    existing = model.find_point(p.x, p.y, p.z + dz)
                    q = existing or model.add_point(p.x, p.y, p.z + dz)
                    if existing is None:
                        clone_assignments(p, q)
                    elif has_assignments(p):
                        kept.append(existing.name)
                    cache[name] = q
                return cache[name]

            for n in points:
                created.append(moved(n).name)
            for n in frames:
                f = model.frames[n]
                created.append(_copy_frame(model, f, moved(f.point_i), moved(f.point_j), st).name)
            for n in areas:
                a = model.areas[n]
                created.append(_copy_area(model, a, [moved(p) for p in a.points], st).name)
        if kept:
            notes.append(f"{len(set(kept))} copied joint(s) landed on existing joints ({', '.join(sorted(set(kept))[:6])}): "
                         "those joints keep their own supports, springs and loads")
        if delete_original and targets:
            _delete_originals(model, frames, areas, points, created)

    doc.edit(mutate)
    for t in notes:
        doc.note(t, "info")
    return created


def divide_frames(doc: Document, frames: Sequence[str], n: int = 2) -> List[str]:
    """Divide frames into ``n`` equal objects (ETABS "Divide Frames").  Every load
    stays where it acts (each piece gets its part of distributed and temperature
    loads, a point load goes to the piece it is on); end releases, partial
    fixities and end / joint offsets stay at the original outer ends only, the new
    inner joints are continuous (function audit FA09)."""
    import math
    from ..model.frame_ops import piece_loads
    created: List[str] = []
    frames = [f for f in frames if f in doc.model.frames]
    if n < 2 or not frames:
        return created
    free3 = (0.0, 0.0, 0.0)

    def mutate(model: Model):
        for name in frames:
            f = model.frames.pop(name)
            a, b = model.points[f.point_i], model.points[f.point_j]
            L = math.dist(a.xyz, b.xyz)
            nodes = [a] + [model.add_point(a.x + (b.x - a.x) * k / n, a.y + (b.y - a.y) * k / n, a.z + (b.z - a.z) * k / n)
                           for k in range(1, n)] + [b]
            for k in range(n):
                new = _copy_frame(model, f, nodes[k], nodes[k + 1], f.story)
                new.loads = piece_loads(f.loads, L, L * k / n, L * (k + 1) / n, last=k == n - 1)
                if k > 0:  # inner end I
                    new.releases_i, new.fixity_i_trans, new.fixity_i_rot = (False,) * 6, free3, free3
                    new.end_offset_i, new.joint_offset_i = 0.0, free3
                if k < n - 1:  # inner end J
                    new.releases_j, new.fixity_j_trans, new.fixity_j_rot = (False,) * 6, free3, free3
                    new.end_offset_j, new.joint_offset_j = 0.0, free3
                lo, hi = k / n, (k + 1) / n
                new.hinges = [dataclasses.replace(h, relative_distance=(h.relative_distance - lo) * n)
                              for h in f.hinges if lo - 1e-9 <= h.relative_distance <= hi + 1e-9
                              and (k == n - 1 or h.relative_distance < hi - 1e-9)]
                created.append(new.name)
            for g in model.groups.values():
                if name in g.frames:
                    i = g.frames.index(name)
                    g.frames[i:i + 1] = created[-n:]

    doc.edit(mutate)
    return created


def set_point_coordinates(doc: Document, point: str, x: float, y: float, z: float):
    def mutate(model: Model):
        p = model.points[point]
        p.x, p.y, p.z = x, y, z
        s = model.story_at(z, model.options.merge_tolerance)
        p.story = s.name if s else p.story

    doc.edit(mutate)


# --------------------------------------------------------- screen selection
def _visible_objects(model: Model, visible):
    frames = [f for f in model.frames.values() if visible(f)]
    areas = [a for a in model.areas.values() if visible(a)]
    return frames, areas


def objects_crossing(model: Model, project, rect, visible=lambda o: True, points_visible=lambda p: True):
    """Crossing selection (drag right-to-left): every object touching the
    screen rectangle ``rect`` = (x0, y0, x1, y1).  ``project(xyz) -> (sx, sy)``."""
    x0, y0, x1, y1 = rect
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def inside(s):
        return x0 <= s[0] <= x1 and y0 <= s[1] <= y1

    def seg_hits(a, b):
        if inside(a) or inside(b):
            return True
        return any(segments_intersect(a, b, corners[k], corners[(k + 1) % 4]) for k in range(4))

    scr = {p.name: project(p.xyz) for p in model.points.values()}
    frames_v, areas_v = _visible_objects(model, visible)
    frames = [f.name for f in frames_v if seg_hits(scr[f.point_i], scr[f.point_j])]
    areas = []
    for a in areas_v:
        poly = [scr[p] for p in a.points]
        n = len(poly)
        if any(seg_hits(poly[k], poly[(k + 1) % n]) for k in range(n)) or point_in_polygon(((x0 + x1) / 2, (y0 + y1) / 2), poly):
            areas.append(a.name)
    points = [n for n, s in scr.items() if inside(s) and points_visible(model.points[n])]
    return frames, areas, points


def objects_on_line(model: Model, project, p0, p1, visible=lambda o: True):
    """Intersecting-line selection: frames and areas whose screen image the
    segment p0-p1 crosses."""
    scr = {p.name: project(p.xyz) for p in model.points.values()}
    frames_v, areas_v = _visible_objects(model, visible)
    frames = [f.name for f in frames_v if segments_intersect(p0, p1, scr[f.point_i], scr[f.point_j])]
    areas = []
    for a in areas_v:
        poly = [scr[p] for p in a.points]
        n = len(poly)
        if any(segments_intersect(p0, p1, poly[k], poly[(k + 1) % n]) for k in range(n)):
            areas.append(a.name)
    return frames, areas, []


def select_by_type(model: Model, kind: str) -> Tuple[List[str], List[str], List[str]]:
    """Columns | Beams | Braces | Floors | Walls | Openings | Supports | All points."""
    k = kind.lower()
    if k in ("columns", "beams", "braces"):
        t = {"columns": "Column", "beams": "Beam", "braces": "Brace"}[k]
        return [f.name for f in model.frames.values() if f.type == t], [], []
    if k in ("floors", "walls", "openings"):
        t = {"floors": "Floor", "walls": "Wall", "openings": "Opening"}[k]
        return [], [a.name for a in model.areas.values() if a.type == t], []
    if k in ("null lines", "null"):
        return [f.name for f in model.frames.values() if f.type == "Null"], [], []
    if k == "supports":
        return [], [], [p.name for p in model.points.values() if any(p.restraints)]
    return [], [], list(model.points)  # Joints


# ---------------------------------------------------------------- panel load
class PanelLoadTool(Tool):
    """Draw a load panel (CYPE 3D "Panels > New"): click the corners in a plan,
    elevation or 3D view (3D: snapped points), double-click / Enter to close.
    The panel's loads and distribution are then set in the Loads on Panel form
    (``pending`` holds the polygon until the window has shown the form)."""

    title = "Load Panel"
    next_prompt = "click the next corner; click the first corner, double-click or Enter = close"
    name = "panel_load"
    prompt = "Click the panel corners (any plane); Enter / right-click / double-click closes the outline"
    closes_polygon = True
    views = ("plan", "xz", "yz", "3d")

    def __init__(self, doc: Document, settings: DrawSettings):
        super().__init__(doc, settings)
        self.pending: Optional[List[XYZ]] = None
        self.reshape_of: Optional[str] = None  # Edit Panel Shape: the panel whose outline is redrawn

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self._p3(pt)
        if self.points and len(self.points) >= 3 and _dist(p, self.points[0]) <= self.tol:
            return self.finish()
        self.points.append(p)
        return False

    def double_click(self, pt) -> bool:
        p = self._p3(pt)
        if not (self.points and _dist(p, self.points[0]) <= self.tol) and \
                not (self.points and _dist(p, self.points[-1]) <= self.tol):
            self.points.append(p)
        return self.finish()

    def finish(self) -> bool:
        if len(self.points) < 3:
            self.points = []
            return False
        self.pending = list(self.points)
        self.points = []
        return True

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        poly = list(self.points)
        if cursor is not None and self.points:
            poly.append(self._p3(cursor))
        pv.polygon = poly
        return pv


TOOLS["panel_load"] = PanelLoadTool


# ------------------------------------------------------- Load menu click tools
AREA_DISTRIBUTIONS = {"On the area (shells)": "shells", "One-way to frames": "one_way",
                      "Two-way to frames": "two_way"}


class LoadPickTool(Tool):
    """One load command, two ways of using it (Load menu):

    * command first: set the load in *Properties of Object*, pick objects (click,
      or drag a window / crossing box), then Enter / right-click / Apply;
    * objects first: select them, then the command - the form opens with their
      current load of the pattern (edit), Enter / right-click / Apply applies.

    The picked objects are the selection (of this tool's kind), so they are
    highlighted like any selection.  :meth:`apply_many` changes them in one
    undoable step."""

    picks: Tuple[str, ...] = ()
    views = ("plan", "xz", "yz", "3d")
    noun = "object"
    nouns = ""  # plural (default noun + "s")
    editing = False  # opened on a selection: the form shows its current loads

    def pattern(self) -> str:
        m = self.model
        pat = self.settings.load_pattern
        if pat not in m.load_patterns:
            if not m.load_patterns:
                raise ModelError("NO_PATTERN", "Define a load pattern first (Define > Load Patterns)")
            pat = next(iter(m.load_patterns))
            self.settings.load_pattern = pat
        return pat

    # ---- the two ways in
    def activate(self, state) -> None:
        """Called when the command starts: with a selection of this kind, edit it."""
        names = self.picked(state)
        self.editing = bool(names)
        if self.editing:
            self.prefill(names)
            self.settings.load_mode = "Replace"  # editing replaces the pattern's load
        else:
            self.settings.load_mode = "Add"

    def picked(self, state) -> List[str]:
        sets = {"point": state.points, "frame": state.frames, "area": state.areas,
                "panel": getattr(state, "panels", set())}
        out = set()
        for k in self.picks:
            out |= set(sets[k])
        return sorted(out)

    def prefill(self, names: Sequence[str]) -> None:
        """Copy the current load (pattern of the form) of the first of ``names``
        that has one into the form."""

    # ---- apply
    def apply_many(self, names: Sequence[str]) -> str:
        raise NotImplementedError

    def apply_to(self, kind: str, name: str) -> str:
        return self.apply_many([name])

    # ---- guidance (status bar, bottom left)
    def plural(self, n: int = 2) -> str:
        return self.noun if n == 1 else (self.nouns or self.noun + "s")

    def hint(self, state=None) -> str:
        n = len(self.picked(state)) if state is not None else 0
        title = self.title
        if n == 0:
            return next_step(title, f"pick the {self.plural()} (click or drag a window)",
                             "set the values in Properties of Object", "Esc = cancel (back to Select)")
        verb = self.settings.load_mode.lower()
        return next_step(title, f"press Enter (or right-click) to {verb} the load on the {n} picked "
                                f"{self.plural(n)}", "click = add / remove a pick", "Esc = clear the picks")


class JointLoadTool(LoadPickTool):
    name = "joint_load"
    title = "Joint Load"
    noun = "joint"
    prompt = "Pick joints, then Enter / right-click to apply the joint load"
    picks = ("point",)

    def prefill(self, names):
        s, m = self.settings, self.model
        pat = self.pattern()
        for n in names:
            ld = next((ld for ld in m.points[n].loads if ld.pattern == pat), None)
            if ld is not None:
                s.load_fx, s.load_fy, s.load_fz = (float(v) for v in ld.force)
                s.load_mx, s.load_my, s.load_mz = (float(v) for v in ld.moment)
                return

    def apply_many(self, names) -> str:
        s, pat = self.settings, self.pattern()
        F = (s.load_fx, s.load_fy, s.load_fz)
        M = (s.load_mx, s.load_my, s.load_mz)
        names = list(names)

        def mutate(model: Model):
            for name in names:
                p = model.points[name]
                if s.load_mode != "Add":
                    p.loads = [ld for ld in p.loads if ld.pattern != pat]
                if s.load_mode != "Delete":
                    model.add_point_load(name, pat, F, M)

        self.doc.edit(mutate)
        return f"Joint load: {s.load_mode.lower()} {pat} on {len(names)} joint(s)"


class FrameLoadTool(LoadPickTool):
    name = "frame_load"
    title = "Frame Load"
    noun = "frame"
    prompt = "Pick frames, then Enter / right-click to apply the frame load"
    picks = ("frame",)

    def prefill(self, names):
        s, m = self.settings, self.model
        pat = self.pattern()
        for n in names:
            ld = next((ld for ld in m.frames[n].loads if ld.pattern == pat and ld.kind in ("distributed", "point")),
                      None)
            if ld is None:
                continue
            s.load_direction = ld.direction
            s.load_value = float(ld.value_a)
            if ld.kind == "point":
                s.frame_load_kind = "Point"
                s.load_rel = float(ld.rel_a)
            elif ld.value_a != ld.value_b:
                s.frame_load_kind = "Trapezoidal"
                s.load_value_b = float(ld.value_b)
            else:
                s.frame_load_kind = "Uniform"
            return

    def apply_many(self, names) -> str:
        s, pat = self.settings, self.pattern()
        point = s.frame_load_kind == "Point"
        vb = s.load_value_b if s.frame_load_kind == "Trapezoidal" else s.load_value
        names = list(names)

        def mutate(model: Model):
            for name in names:
                f = model.frames[name]
                if s.load_mode != "Add":  # replace / delete the loads of this pattern and kind
                    f.loads = [ld for ld in f.loads if not (ld.pattern == pat and (ld.kind == "point") == point)]
                if s.load_mode == "Delete":
                    continue
                if point:
                    model.add_frame_point_load(name, pat, s.load_value, rel=min(max(s.load_rel, 0.0), 1.0),
                                               direction=s.load_direction)
                else:
                    model.add_frame_distributed_load(name, pat, s.load_value, vb, direction=s.load_direction)

        self.doc.edit(mutate)
        return f"Frame load: {s.load_mode.lower()} {pat} {s.frame_load_kind.lower()} on {len(names)} frame(s)"


class FrameSectionTool(LoadPickTool):
    """Assign > Frame > Section Property, as a command: choose the section in
    Properties of Object (its own list, not the dialog of Define), pick frames by
    click or window, and only Enter / right-click assigns it (one undo step).  Esc
    clears the picks.  Frames selected before the command are shown as picked, and
    their section is the one shown in the list."""

    name = "frame_section"
    title = "Assign Frame Section"
    noun = "frame"
    prompt = "Choose a section, pick frames, then Enter / right-click to assign it"
    picks = ("frame",)
    apply_label = "Assign section"

    def activate(self, state) -> None:
        names = self.picked(state)
        self.editing = bool(names)
        if names:
            self.prefill(names)
        self._default_section()

    def _default_section(self) -> None:
        """No section chosen yet: the first defined one (so the command is ready)."""
        if self.settings.frame_section not in self.model.frame_sections and self.model.frame_sections:
            self.settings.frame_section = next(iter(self.model.frame_sections))

    def prefill(self, names: Sequence[str]) -> None:
        m = self.model
        for n in names:
            f = m.frames.get(n)
            if f is not None and f.section in m.frame_sections:
                self.settings.frame_section = f.section
                return

    def section(self) -> str:
        sec = self.settings.frame_section
        if sec not in self.model.frame_sections:
            raise ModelError("NO_SECTION", "Choose a frame section in Properties of Object "
                                           "(define one first: Define > Frame Sections)")
        return sec

    def apply_many(self, names: Sequence[str]) -> str:
        sec = self.section()
        names = list(names)

        def mutate(model: Model):
            for n in names:
                model.frames[n].section = sec

        self.doc.edit(mutate)
        return f"Section {sec} assigned to {len(names)} frame(s)"

    def hint(self, state=None) -> str:
        n = len(self.picked(state)) if state is not None else 0
        self._default_section()
        sec = self.settings.frame_section or "(none)"
        if n == 0:
            return next_step(f"{self.title} ({sec})", "pick the frames (click or drag a window)",
                             "change the section in Properties of Object", "Esc = cancel (back to Select)")
        return next_step(f"{self.title} ({sec})", f"press Enter (or right-click) to assign it to the {n} "
                                                  f"picked {self.plural(n)}", "click = add / remove a pick",
                                                  "Esc = clear the picks")


@dataclass
class AssignSpec:
    """An Assign command shown in the box at the lower left (Properties of Object):

    ``fields(obj)`` - the form (FormDialog field tuples) with the values of ``obj`` (the
    first picked object, or a new default one); or ``widget(obj)`` - (QWidget,
    values()) for a form of its own; ``apply(names, values)`` changes the model (one
    undo step) and may return the status text."""

    key: str
    title: str
    picks: Tuple[str, ...]
    apply: Callable
    fields: Optional[Callable] = None
    widget: Optional[Callable] = None
    note: Optional[str] = None
    noun: str = ""
    nouns: str = ""
    # OK runs this with the picked names instead of apply + close (e.g. Move Panel: the
    # Move command starts with them selected); ``apply`` may then be None (no Apply button)
    on_ok: Optional[Callable] = None


class AssignFormTool(LoadPickTool):
    """Assign > ... as a command in the box at the lower left: the form opens at once
    (with the values of the objects selected before, if any); pick / unpick objects;
    Apply assigns and keeps the box for the next set, OK / Enter assigns and ends,
    Cancel / Esc leaves."""

    name = "assign_form"
    apply_label = "Assign"

    def __init__(self, doc, settings, spec: AssignSpec):
        super().__init__(doc, settings)
        self.spec = spec
        self.name = "assign:" + spec.key
        self.title = spec.title
        self.picks = tuple(spec.picks)
        words = {"point": ("joint", "joints"), "frame": ("frame", "frames"), "area": ("shell", "shells"),
                 "panel": ("load panel", "load panels")}
        if spec.noun:
            self.noun, self.nouns = spec.noun, spec.nouns or spec.noun + "s"
        elif len(self.picks) == 1:
            self.noun, self.nouns = words[self.picks[0]]
        else:
            self.noun, self.nouns = "object", "objects"
        self.prompt = f"Set the values, pick {self.nouns}, then Apply / Enter"
        self.fields: list = []
        self.read_values: Optional[Callable] = None  # set by the box: the form's values
        self.first = None

    def first_object(self, names):
        m = self.model
        for n in names:
            for kind in self.picks:
                table = {"point": m.points, "frame": m.frames, "area": m.areas,
                         "panel": getattr(m, "panel_loads", {})}[kind]
                if n in table:
                    return table[n]
        from ..model.objects import Area, Frame, Point
        return {"point": Point(""), "frame": Frame("", "", "", ""), "area": Area("", []),
                "panel": None}[self.picks[0]]

    def activate(self, state) -> None:
        names = self.picked(state)
        self.editing = bool(names)
        self.first = self.first_object(names)
        if self.spec.fields is not None:
            self.fields = list(self.spec.fields(self.first))

    def apply_many(self, names: Sequence[str]) -> str:
        if self.spec.apply is None:
            raise ModelError("NO_APPLY", f"{self.title}: press OK")
        if self.read_values is None:
            raise ModelError("NO_FORM", f"{self.title}: the form is not open")
        try:
            values = self.read_values()
        except ValueError as exc:
            raise ModelError("BAD_VALUE", f"{self.title}: {exc}") from None
        msg = self.spec.apply(list(names), values)
        return msg if isinstance(msg, str) and msg else \
            f"{self.title}: assigned to {len(names)} {self.plural(len(names))}"

    def hint(self, state=None) -> str:
        n = len(self.picked(state)) if state is not None else 0
        if n == 0:
            return next_step(self.title, f"pick the {self.plural()} (click or drag a window)",
                             "set the values in the box at the left", "Cancel / Esc = leave")
        return next_step(self.title, f"Apply (assign, pick the next set) or Enter / OK (assign and finish) "
                                     f"for the {n} picked {self.plural(n)}", "click = add / remove a pick",
                                     "Esc = clear the picks")


class ShellSectionTool(FrameSectionTool):
    """Assign > Shell > Slab / Deck / Wall Section, as a command like Assign Frame
    Section: the sections of that type are listed in Properties of Object (with
    Define sections... to make a new one), pick floors / walls, Enter / right-click
    assigns it."""

    kind = "Slab"  # Slab | Deck | Wall (the ShellSection.type listed)
    attr = "shell_section"  # the DrawSettings item
    noun = "floor"
    picks = ("area",)

    def sections(self) -> List[str]:
        m = self.model
        return [n for n, x in m.shell_sections.items() if x.type == self.kind] or list(m.shell_sections)

    def _default_section(self) -> None:
        secs = self.sections()
        if getattr(self.settings, self.attr) not in secs and secs:
            setattr(self.settings, self.attr, secs[0])

    def prefill(self, names: Sequence[str]) -> None:
        m = self.model
        secs = self.sections()
        for n in names:
            a = m.areas.get(n)
            if a is not None and a.section in secs:
                setattr(self.settings, self.attr, a.section)
                return

    def section(self) -> str:
        sec = getattr(self.settings, self.attr)
        if sec not in self.model.shell_sections:
            raise ModelError("NO_SECTION", f"Choose a {self.kind.lower()} section in Properties of Object "
                                           "(define one first: Define sections...)")
        return sec

    def targets(self, names: Sequence[str]) -> List[str]:
        """The picked areas of this kind (walls for a wall section, floors otherwise); all
        of them when none is."""
        m = self.model
        wall = self.kind == "Wall"
        mine = [n for n in names if n in m.areas and (m.areas[n].type == "Wall") == wall]
        return mine or [n for n in names if n in m.areas]

    def apply_many(self, names: Sequence[str]) -> str:
        sec = self.section()
        names = self.targets(names)

        def mutate(model: Model):
            for n in names:
                model.areas[n].section = sec

        self.doc.edit(mutate)
        return f"Section {sec} assigned to {len(names)} {self.plural(len(names))}"

    def hint(self, state=None) -> str:
        n = len(self.picked(state)) if state is not None else 0
        self._default_section()
        sec = getattr(self.settings, self.attr) or "(none)"
        if n == 0:
            return next_step(f"{self.title} ({sec})", f"pick the {self.plural()} (click or drag a window)",
                             "change the section in Properties of Object", "Esc = cancel (back to Select)")
        return next_step(f"{self.title} ({sec})", f"press Enter (or right-click) to assign it to the {n} "
                                                  f"picked {self.plural(n)}", "click = add / remove a pick",
                                                  "Esc = clear the picks")


class SlabSectionTool(ShellSectionTool):
    name = "slab_section"
    title = "Assign Slab Section"
    prompt = "Choose a slab section, pick floors, then Enter / right-click to assign it"


class DeckSectionTool(ShellSectionTool):
    name = "deck_section"
    title = "Assign Deck Section"
    kind = "Deck"
    attr = "deck_section"
    prompt = "Choose a deck section, pick floors, then Enter / right-click to assign it"


class WallSectionTool(ShellSectionTool):
    name = "wall_section"
    title = "Assign Wall Section"
    kind = "Wall"
    attr = "wall_section"
    noun = "wall"
    prompt = "Choose a wall section, pick walls, then Enter / right-click to assign it"


class AreaLoadTool(LoadPickTool):
    """Area loads are load panels made from the areas (on the area itself, or
    one-way / two-way to the frames around it)."""

    name = "area_load"
    title = "Area Load"
    noun = "floor / wall"
    nouns = "floors / walls"
    prompt = "Pick floors / walls, then Enter / right-click to apply the area load (a load panel on each)"
    picks = ("area",)

    def prefill(self, names):
        s, m = self.settings, self.model
        pat = self.pattern()
        inv = {v: k for k, v in AREA_DISTRIBUTIONS.items()}
        for n in names:
            for p in m.panel_loads.values():
                if p.area != n:
                    continue
                c = next((c for c in p.loads if c.pattern == pat), None)
                if c is None:
                    continue
                s.load_value, s.load_direction = float(c.value), c.direction
                s.area_distribution = inv.get(p.distribution, s.area_distribution)
                if p.span_dir is not None:
                    s.area_span = "X" if abs(p.span_dir[0]) >= abs(p.span_dir[1]) else "Y"
                return

    def apply_many(self, names) -> str:
        from ..model.objects import PanelLoadCase
        s, pat = self.settings, self.pattern()
        dist = AREA_DISTRIBUTIONS.get(s.area_distribution, "shells")
        span = (1.0, 0.0, 0.0) if s.area_span == "X" else (0.0, 1.0, 0.0)
        names = list(names)

        def mutate(model: Model):
            for name in names:
                if s.load_mode != "Add":  # drop this pattern from the area's panels
                    for p in [p for p in model.panel_loads.values() if p.area == name]:
                        p.loads = [c for c in p.loads if c.pattern != pat]
                        if not p.loads:
                            del model.panel_loads[p.name]
                if s.load_mode == "Delete":
                    continue
                case = PanelLoadCase(pat, s.load_value, s.load_direction)
                same = next((p for p in model.panel_loads.values() if p.area == name and p.distribution == dist and
                             (dist != "one_way" or tuple(p.span_dir or ()) == span)), None)
                if same is not None:
                    same.loads.append(case)
                else:
                    model.add_area_panel(name, [case], distribution=dist,
                                         span_dir=span if dist == "one_way" else None)

        self.doc.edit(mutate)
        return f"Area load: {s.load_mode.lower()} {pat} on {len(names)} area(s) ({s.area_distribution.lower()})"


TOOLS.update({"joint_load": JointLoadTool, "frame_load": FrameLoadTool, "area_load": AreaLoadTool,
              "frame_section": FrameSectionTool, "slab_section": SlabSectionTool,
              "deck_section": DeckSectionTool, "wall_section": WallSectionTool})
TOOL_PROPERTIES["slab_section"] = [("shell_section", "Section", "section_list", "")]
TOOL_PROPERTIES["deck_section"] = [("deck_section", "Section", "section_list", "")]
TOOL_PROPERTIES["wall_section"] = [("wall_section", "Section", "section_list", "")]
_LOAD_MODE = ("load_mode", "Existing loads (this pattern)", "choice", ["Add", "Replace", "Delete"])
TOOL_PROPERTIES["joint_load"] = [
    ("load_pattern", "Load pattern", "pattern", ""), _LOAD_MODE,
    ("load_fx", "Force global X", "float", ""), ("load_fy", "Force global Y", "float", ""),
    ("load_fz", "Force global Z", "float", "gravity loads are negative"),
    ("load_mx", "Moment about global X", "float", ""), ("load_my", "Moment about global Y", "float", ""),
    ("load_mz", "Moment about global Z", "float", ""),
]
TOOL_PROPERTIES["frame_load"] = [
    ("load_pattern", "Load pattern", "pattern", ""), _LOAD_MODE,
    ("frame_load_kind", "Load", "choice", ["Uniform", "Trapezoidal", "Point"]),
    ("load_direction", "Direction", "choice", ["Gravity", "X", "Y", "Z", "Local 1", "Local 2", "Local 3",
                                                "GravityProjected", "XProjected", "YProjected"]),
    ("load_value", "Value (start / uniform / point)", "float", "force / length; point load: force"),
    ("load_value_b", "Value at the end (trapezoidal)", "float", ""),
    ("load_rel", "Relative distance (point load)", "float", "0 = start, 1 = end"),
]
TOOL_PROPERTIES["area_load"] = [
    ("load_pattern", "Load pattern", "pattern", ""), _LOAD_MODE,
    ("load_value", "Value (force / area)", "float", "gravity: positive downwards"),
    ("load_direction", "Direction", "choice", ["Gravity", "Normal", "X", "Y", "Z", "GravityProjected"]),
    ("area_distribution", "Distribution", "choice", list(AREA_DISTRIBUTIONS)),
    ("area_span", "One-way span direction", "choice", ["X", "Y"]),
]
TOOL_PROPERTIES["panel_load"] = [
    ("all_stories", "All stories", "bool", "repeat the panel on every story (plan views)"),
]
