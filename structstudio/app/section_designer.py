"""Section Designer: draw any frame cross-section like in CAD.

Tools: select / move / edit points, polygon, rectangle, circle, insert a
catalogue shape (I, T, channel ... as editable outlines), openings (holes).
Snaps to end points, mid points, centres, edges and the grid; ortho (F8 or
hold Shift); coordinates typed on the command line like CAD::

    0,0          absolute x,y           @30,0      relative to the last point
    @50<90       relative, length<angle 25         distance along the cursor
    c / close    close the polygon      u / redo   undo / redo
    p r ci i     polygon / rectangle / circle / insert   h  next shape is an opening
    g 5          grid spacing           f          zoom to fit

Overlapping or touching shapes are united (draw a T as two rectangles); the
properties (A, I33, I22, I23, J by finite elements, As2, As3, S, Z, r) are
updated as you draw.  x is the local 3 axis (width), y the local 2 axis (depth),
lengths in the section-dimension display unit.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..display_units import DisplayUnits
from ..model import FrameSection
from ..model.section_shapes import (COLD, CIRCLE_SEGMENTS, SHAPES, check_dims, designer_polygons, designer_properties,
                                    mesh_region, points_in_polygon, rect, shape_outline, signed_area)
from .section_manager import (STEEL_TYPES, edit_modifiers, fill_table, fmt, material_for, props_table, property_rows,
                              unique_name)

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
RE_XY = re.compile(rf"^(@)?\s*({NUM})\s*[,;]\s*({NUM})$")
RE_POLAR = re.compile(rf"^(@)?\s*({NUM})\s*<\s*({NUM})$")
RE_NUM = re.compile(rf"^({NUM})$")


def parse_point(text: str, last: Optional[np.ndarray], toward: Optional[np.ndarray], k: float):
    """A point typed on the command line (display units, ``k`` = shown per
    model unit) -> model coordinates, or None.  ``x,y`` absolute, ``@dx,dy``
    relative, ``@L<a`` / ``L<a`` polar, a bare number = that distance from
    ``last`` toward ``toward`` (the cursor)."""
    t = text.strip()
    m = RE_XY.match(t)
    if m:
        p = np.array([float(m.group(2)), float(m.group(3))]) / k
        if m.group(1):
            return None if last is None else last + p
        return p
    m = RE_POLAR.match(t)
    if m:
        L, a = float(m.group(2)) / k, math.radians(float(m.group(3)))
        d = np.array([L * math.cos(a), L * math.sin(a)])
        if m.group(1):
            return None if last is None else last + d
        return d
    m = RE_NUM.match(t)
    if m and last is not None and toward is not None:
        d = toward - last
        n = float(np.linalg.norm(d))
        if n <= 0:
            return None
        return last + d / n * (float(m.group(1)) / k)
    return None


def shape_points(g: dict) -> np.ndarray:
    if g.get("kind") == "circle":
        c, r = np.asarray(g["center"], float), float(g["radius"])
        t = np.linspace(0, 2 * math.pi, CIRCLE_SEGMENTS, endpoint=False)
        return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])
    return np.asarray(g.get("points") or [], float).reshape(-1, 2)


def nice_step(x: float) -> float:
    if x <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(x))
    for f in (1, 2, 2.5, 5, 10):
        if f * e >= x:
            return f * e
    return 10 * e


def self_intersecting(P: np.ndarray) -> bool:
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = P[j], P[(j + 1) % n]
            d1 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            d2 = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])
            d3 = (d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0])
            d4 = (d[0] - c[0]) * (b[1] - c[1]) - (d[1] - c[1]) * (b[0] - c[0])
            if d1 * d2 < 0 and d3 * d4 < 0:
                return True
    return False


def shape_from_catalogue(key: str, v: dict, at=(0.0, 0.0), mirror: bool = False) -> List[dict]:
    """Editable designer shapes of a catalogue section placed with its centroid
    at ``at`` (circles stay circles)."""
    x0, y0 = float(at[0]), float(at[1])
    if key == "Circular":
        return [{"kind": "circle", "center": [x0, y0], "radius": v["t3"] / 2, "hole": False}]
    if key in ("Pipe", "Concrete Pipe"):
        return [{"kind": "circle", "center": [x0, y0], "radius": v["t3"] / 2, "hole": False},
                {"kind": "circle", "center": [x0, y0], "radius": v["t3"] / 2 - v["tw"], "hole": True}]
    solids, holes = shape_outline(SHAPES[key], v, mirror)
    out = [{"kind": "polygon", "points": [[float(x) + x0, float(y) + y0] for x, y in P], "hole": False} for P in solids]
    out += [{"kind": "polygon", "points": [[float(x) + x0, float(y) + y0] for x, y in P], "hole": True} for P in holes]
    return out


def _apply(g: dict, fn) -> dict:
    """A copy of a designer shape with ``fn`` ((n, 2) -> (n, 2)) applied to it."""
    g = copy.deepcopy(g)
    if g.get("kind") == "circle":
        g["center"] = fn(np.asarray([g["center"]], float))[0].tolist()
    else:
        g["points"] = fn(np.asarray(g["points"], float)).tolist()
    return g


def _rotate_fn(c, a: float):
    ca, sa = math.cos(a), math.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    c = np.asarray(c, float)
    return lambda P: (np.asarray(P) - c) @ R.T + c


def _mirror_fn(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = (b - a) / (np.linalg.norm(b - a) or 1.0)
    M = 2.0 * np.outer(u, u) - np.eye(2)
    return lambda P: (np.asarray(P) - a) @ M.T + a


def _clip_halfplane(P: np.ndarray, a, b, sign: float) -> np.ndarray:
    """Sutherland-Hodgman: the part of polygon P on one side of the line a-b."""
    d = np.asarray(b, float) - np.asarray(a, float)
    f = lambda q: sign * (d[0] * (q[1] - a[1]) - d[1] * (q[0] - a[0]))
    out = []
    n = len(P)
    for i in range(n):
        p, q = P[i], P[(i + 1) % n]
        fp, fq = f(p), f(q)
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append(p + t * (q - p))
    return np.array(out) if out else np.zeros((0, 2))


def _intersections(polys: Sequence[np.ndarray]) -> np.ndarray:
    """Crossing points of the edges of the drawn shapes (intersection snap)."""
    polys = [np.asarray(P, float) for P in polys if len(P) >= 2]
    if len(polys) < 1:
        return np.zeros((0, 2))
    A = np.vstack([P for P in polys])
    B = np.vstack([np.roll(P, -1, axis=0) for P in polys])
    if len(A) > 1500:
        return np.zeros((0, 2))
    D = B - A
    den = D[:, None, 0] * D[None, :, 1] - D[:, None, 1] * D[None, :, 0]
    W = A[None, :, :] - A[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (W[..., 0] * D[None, :, 1] - W[..., 1] * D[None, :, 0]) / den
        u = (W[..., 0] * D[:, None, 1] - W[..., 1] * D[:, None, 0]) / den
    eps = 1e-9
    ok = (np.abs(den) > 1e-18) & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)
    ii, jj = np.nonzero(np.triu(ok, 1))
    if not len(ii):
        return np.zeros((0, 2))
    return A[ii] + t[ii, jj][:, None] * D[ii]


class DesignerCanvas(QtWidgets.QWidget):
    """The drawing area (see the module docstring)."""

    changed = QtCore.Signal()
    selection_changed = QtCore.Signal()
    modes_changed = QtCore.Signal()  # snap / grid / ortho toggled from the keyboard or the command line
    prompt_changed = QtCore.Signal(str)
    cursor_moved = QtCore.Signal(str)

    TOOLS = ("select", "polygon", "rect", "circle", "insert")
    PROMPTS = {"select": "Select: click a shape, drag a point or the shape, window from left / crossing from right",
               "polygon": "Polygon: first point (click, or type x,y)",
               "rect": "Rectangle: first corner (click, or type x,y)",
               "circle": "Circle: centre (click, or type x,y)",
               "insert": "Insert: pick the position of the centroid (click, or type x,y)"}

    def __init__(self, du: DisplayUnits, parent=None):
        super().__init__(parent)
        self.du = du
        self.k = du.factor("section_dim")  # shown per model length
        self.unit = du.label("section_dim")
        self.shapes: List[dict] = []
        self.selected: set = set()
        self.tool = "select"
        self.hole_mode = False
        self.grid = 0.05 / 1.0  # model units, set by the dialog
        # object snaps (F3 turns them all off / on), grid snap (F9), grid display (F7)
        self.snap = {"end": True, "mid": True, "centre": True, "intersection": True, "perpendicular": True,
                     "nearest": True, "grid": True}
        self.osnap_on = True
        self.show_grid = True
        self.ortho = False
        self.op: Optional[dict] = None  # a CAD edit command in progress (move, copy, mirror, rotate, trim, paste)
        self.clipboard: Optional[Tuple[List[dict], np.ndarray]] = None
        self._isect: np.ndarray = np.zeros((0, 2))
        self.scale = 400.0  # px per model unit
        self.center = np.array([0.0, 0.0])
        self.draft: List[np.ndarray] = []
        self.cursor = np.array([0.0, 0.0])
        self.snap_kind = ""
        self.insert_shapes: List[dict] = []
        self.undo_stack: List[List[dict]] = []
        self.redo_stack: List[List[dict]] = []
        self.region_loops: List[np.ndarray] = []
        self.centroid: Optional[np.ndarray] = None
        self.steel = False
        self._drag = None  # ("pan", pos, center) | ("vertex", i, j) | ("move", start) | ("band", start)
        self._shift = False
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(420, 360)
        self.setCursor(QtCore.Qt.CrossCursor)

    # ------------------------------------------------------------ geometry
    def set_shapes(self, shapes: Sequence[dict], fit: bool = True):
        self.shapes = [copy.deepcopy(dict(g)) for g in shapes]
        self.selected = set()
        self.update_region()
        if fit:
            self.fit()
        self.changed.emit()

    def push_undo(self):
        self.undo_stack.append(copy.deepcopy(self.shapes))
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if self.draft:
            self.draft.pop()
            self._prompt()
            self.update()
            return
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.shapes))
        self.shapes = self.undo_stack.pop()
        self.selected = {i for i in self.selected if i < len(self.shapes)}
        self._edited()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(copy.deepcopy(self.shapes))
        self.shapes = self.redo_stack.pop()
        self.selected = {i for i in self.selected if i < len(self.shapes)}
        self._edited()

    def _edited(self):
        self.update_region()
        self.update()
        self.changed.emit()
        self.selection_changed.emit()

    def update_region(self):
        self._isect = _intersections([shape_points(g) for g in self.shapes])
        solids, holes = designer_polygons(self.shapes)
        mesh = mesh_region(solids, holes, None) if solids else None
        self.region_loops = mesh.loops if mesh is not None else []
        from ..model.section_shapes import loop_moments
        if self.region_loops:
            m = sum(loop_moments(L) for L in self.region_loops)
            self.centroid = np.array([m[1] / m[0], m[2] / m[0]]) if m[0] > 0 else None
        else:
            self.centroid = None

    def add_shapes(self, new: Sequence[dict]):
        self.push_undo()
        n = len(self.shapes)
        self.shapes += [copy.deepcopy(g) for g in new]
        self.selected = set(range(n, len(self.shapes)))
        self._edited()

    def delete_selected(self):
        if not self.selected:
            return
        self.push_undo()
        self.shapes = [g for i, g in enumerate(self.shapes) if i not in self.selected]
        self.selected = set()
        self._edited()

    def _transform_selected(self, fn):
        if not self.selected:
            return
        self.push_undo()
        pts = np.vstack([shape_points(self.shapes[i]) for i in self.selected])
        c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        for i in self.selected:
            g = self.shapes[i]
            if g.get("kind") == "circle":
                g["center"] = list(map(float, fn(np.asarray(g["center"], float) - c) + c))
            else:
                P = np.array([fn(np.asarray(p, float) - c) + c for p in g["points"]])
                g["points"] = P.tolist()
        self._edited()

    def mirror_selected(self):
        self._transform_selected(lambda d: np.array([-d[0], d[1]]))

    def rotate_selected(self, deg: float = 90.0):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        self._transform_selected(lambda d: np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]]))

    def copy_selected(self):
        if not self.selected:
            return
        off = np.array([self.grid * 2, -self.grid * 2])
        new = []
        for i in sorted(self.selected):
            g = copy.deepcopy(self.shapes[i])
            if g.get("kind") == "circle":
                g["center"] = (np.asarray(g["center"]) + off).tolist()
            else:
                g["points"] = (np.asarray(g["points"]) + off).tolist()
            new.append(g)
        self.add_shapes(new)

    def toggle_hole_selected(self):
        if not self.selected:
            return
        self.push_undo()
        for i in self.selected:
            self.shapes[i]["hole"] = not self.shapes[i].get("hole")
        self._edited()

    def select_all(self):
        self.selected = set(range(len(self.shapes)))
        self.update()
        self.selection_changed.emit()

    # ------------------------------------------------------------ view
    def to_screen(self, p) -> QtCore.QPointF:
        return QtCore.QPointF(self.width() / 2 + (p[0] - self.center[0]) * self.scale,
                              self.height() / 2 - (p[1] - self.center[1]) * self.scale)

    def to_world(self, q: QtCore.QPointF) -> np.ndarray:
        return np.array([self.center[0] + (q.x() - self.width() / 2) / self.scale,
                         self.center[1] - (q.y() - self.height() / 2) / self.scale])

    def fit(self):
        pts = [shape_points(g) for g in self.shapes] + ([np.array(self.draft)] if self.draft else [])
        pts = [p for p in pts if len(p)]
        if not pts:
            span = 20 * self.grid
            lo, hi = np.array([-span / 2, -span / 2]), np.array([span / 2, span / 2])
        else:
            allp = np.vstack(pts)
            lo, hi = allp.min(axis=0), allp.max(axis=0)
        size = np.maximum(hi - lo, 1e-9)
        w, h = max(self.width(), 200), max(self.height(), 200)
        self.scale = 0.8 * min(w / size[0], h / size[1])
        self.center = 0.5 * (lo + hi)
        self.update()

    def wheelEvent(self, ev):
        f = 1.2 ** (ev.angleDelta().y() / 120.0)
        p = ev.position()
        before = self.to_world(p)
        self.scale = min(max(self.scale * f, 1e-3), 1e9)
        after = self.to_world(p)
        self.center += before - after
        self.update()

    # ------------------------------------------------------------ snapping
    def _tol(self) -> float:
        return 10.0 / self.scale

    def snapped(self, p: np.ndarray, ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
        """The point the cursor snaps to and the kind of snap: end point,
        centre / quadrant / centroid / origin, intersection, mid point,
        perpendicular (from ``ref``), nearest on an edge, ortho, grid."""
        tol = self._tol()
        sn = self.snap
        osnap = self.osnap_on
        moving = {self._drag[1]} if self._drag and self._drag[0] == "vertex" else set()
        if self.op and self.op["name"] in ("move", "copy", "paste") and self.op.get("pts"):
            moving = set(self.selected)  # do not snap to what is being moved
        best, kind, bd = None, "", tol

        def take(q, k):
            nonlocal best, kind, bd
            d = float(np.linalg.norm(np.asarray(q, float) - p))
            if d < bd:
                best, kind, bd = np.array(q, float), k, d
        if osnap and sn["end"]:
            for i, g in enumerate(self.shapes):
                if i in moving or g.get("kind") == "circle":
                    continue
                for q in g["points"]:
                    take(q, "end point")
            for q in self.draft:
                take(q, "end point")
        if osnap and sn["centre"]:
            for i, g in enumerate(self.shapes):
                if i in moving or g.get("kind") != "circle":
                    continue
                c, r = np.asarray(g["center"], float), float(g["radius"])
                take(c, "centre")
                for d in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    take(c + r * np.array(d), "quadrant")
            if self.centroid is not None:
                take(self.centroid, "centroid")
            take(np.zeros(2), "origin")
        if best is None and osnap and sn["intersection"] and len(self._isect):
            for q in self._isect:
                take(q, "intersection")
        if best is not None:
            return best, kind
        if osnap and sn["mid"]:
            for i, g in enumerate(self.shapes):
                if i in moving or g.get("kind") == "circle":
                    continue
                P = np.asarray(g["points"], float)
                for a, b in zip(P, np.roll(P, -1, axis=0)):
                    take(0.5 * (a + b), "mid point")
            if best is not None:
                return best, kind
        if osnap and sn["perpendicular"] and ref is not None:
            for i, g in enumerate(self.shapes):
                if i in moving:
                    continue
                P = shape_points(g)
                for a, b in zip(P, np.roll(P, -1, axis=0)):
                    ab = b - a
                    L2 = float(ab @ ab)
                    if L2 <= 0:
                        continue
                    t = float((ref - a) @ ab) / L2
                    if 0.0 <= t <= 1.0 and np.linalg.norm(a + t * ab - ref) > 1e-12:
                        take(a + t * ab, "perpendicular")
            if best is not None:
                return best, kind
        if self.ortho_on() and ref is not None:
            d = p - ref
            p = np.array([p[0], ref[1]]) if abs(d[0]) >= abs(d[1]) else np.array([ref[0], p[1]])
            if sn["grid"] and self.grid > 0:
                if abs(d[0]) >= abs(d[1]):
                    p[0] = ref[0] + round((p[0] - ref[0]) / self.grid) * self.grid
                else:
                    p[1] = ref[1] + round((p[1] - ref[1]) / self.grid) * self.grid
            return p, "ortho"
        if osnap and sn["nearest"]:
            bd = 0.6 * tol
            for i, g in enumerate(self.shapes):
                if i in moving:
                    continue
                P = shape_points(g)
                for a, b in zip(P, np.roll(P, -1, axis=0)):
                    ab = b - a
                    L2 = float(ab @ ab)
                    if L2 <= 0:
                        continue
                    t = min(max(float((p - a) @ ab) / L2, 0.0), 1.0)
                    take(a + t * ab, "on edge")
            if best is not None:
                return best, kind
        if sn["grid"] and self.grid > 0:
            return np.round(p / self.grid) * self.grid, "grid"
        return p, ""

    def ortho_on(self) -> bool:
        return self.ortho != self._shift

    def ref_point(self) -> Optional[np.ndarray]:
        if self.op and self.op.get("pts"):
            return self.op["pts"][-1]
        if self.draft:
            return self.draft[-1]
        return None

    # ------------------------------------------------------------ tools
    def set_tool(self, tool: str):
        self.tool = tool
        self.op = None
        self.draft = []
        if tool != "insert":
            self.insert_shapes = []
        self._prompt()
        self.update()

    def _prompt(self):
        if self.op:
            self.prompt_changed.emit(self._op_prompt())
            return
        if self.tool == "polygon" and self.draft:
            txt = (f"Polygon: next point ({len(self.draft)} so far) - type x,y  @dx,dy  @L<angle  or a distance; "
                   f"Enter / right-click / c to close, Backspace to undo a point")
        elif self.tool == "rect" and self.draft:
            txt = "Rectangle: opposite corner (click, x,y or @width,height)"
        elif self.tool == "circle" and self.draft:
            txt = "Circle: radius (type a number) or a point on the circle"
        else:
            txt = self.PROMPTS.get(self.tool, "")
        if self.hole_mode and self.tool in ("polygon", "rect", "circle"):
            txt = "[opening] " + txt
        self.prompt_changed.emit(txt)

    def input_point(self, p: np.ndarray):
        """A point picked or typed for the current tool / command."""
        if self.op:
            self.op_point(np.asarray(p, float))
            return
        if self.tool == "polygon":
            if len(self.draft) >= 3 and np.linalg.norm(p - self.draft[0]) <= max(self._tol() * 0.8, 1e-12):
                self.finish_polygon()
                return
            if self.draft and np.linalg.norm(p - self.draft[-1]) <= 1e-12:
                return
            self.draft.append(np.array(p, float))
        elif self.tool == "rect":
            if not self.draft:
                self.draft = [np.array(p, float)]
            else:
                a, b = self.draft[0], np.array(p, float)
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                if np.all(hi - lo > 1e-12):
                    self.add_shapes([{"kind": "polygon", "points": rect(lo[0], lo[1], *(hi - lo)).tolist(),
                                      "hole": self.hole_mode}])
                self.draft = []
        elif self.tool == "circle":
            if not self.draft:
                self.draft = [np.array(p, float)]
            else:
                self.finish_circle(float(np.linalg.norm(np.array(p, float) - self.draft[0])))
        elif self.tool == "insert" and self.insert_shapes:
            new = []
            for g in self.insert_shapes:
                g = copy.deepcopy(g)
                if g.get("kind") == "circle":
                    g["center"] = (np.asarray(g["center"]) + p).tolist()
                else:
                    g["points"] = (np.asarray(g["points"]) + p).tolist()
                new.append(g)
            self.add_shapes(new)
            self.set_tool("select")
        self._prompt()
        self.update()

    def finish_circle(self, r: float):
        if self.draft and r > 1e-12:
            self.add_shapes([{"kind": "circle", "center": self.draft[0].tolist(), "radius": r, "hole": self.hole_mode}])
        self.draft = []
        self._prompt()
        self.update()

    def finish_polygon(self) -> bool:
        if self.tool != "polygon" or len(self.draft) < 3:
            return False
        P = np.array(self.draft)
        if abs(signed_area(P)) <= 1e-14 or self_intersecting(P):
            self.prompt_changed.emit("The polygon crosses itself or has no area - not added (Esc to start again)")
            return False
        self.add_shapes([{"kind": "polygon", "points": P.tolist(), "hole": self.hole_mode}])
        self.draft = []
        self._prompt()
        self.update()
        return True

    def cancel(self):
        if self.op:
            self.op = None
            self.set_tool("select")
            return
        if self.draft:
            self.draft = []
        elif self.tool != "select":
            self.set_tool("select")
        else:
            self.selected = set()
            self.selection_changed.emit()
        self._prompt()
        self.update()

    def command(self, text: str) -> bool:
        """A line typed on the command line; True when understood."""
        t = text.strip()
        low = t.lower()
        if self.op:
            return self.op_command(t)
        if not t:
            if self.tool == "polygon" and self.draft:
                return self.finish_polygon()
            return True
        ops = {"m": "move", "move": "move", "co": "copy", "cp": "copy", "copy": "copy", "mi": "mirror",
               "mirror": "mirror", "ro": "rotate", "rotate": "rotate", "tr": "trim", "trim": "trim",
               "pa": "paste", "paste": "paste"}
        if low in ops and not self.draft:
            self.start_op(ops[low])
            return True
        if self.tool == "polygon" and self.draft and low in ("c", "close"):
            return self.finish_polygon()
        if self.tool == "circle" and self.draft and RE_NUM.match(t):
            self.finish_circle(float(t) / self.k)
            return True
        cmds = {"p": "polygon", "pl": "polygon", "poly": "polygon", "polygon": "polygon", "r": "rect", "rec": "rect",
                "rect": "rect", "rectangle": "rect", "ci": "circle", "circle": "circle", "s": "select",
                "select": "select", "i": "insert"}
        if low in cmds and not (self.draft and RE_NUM.match(t)):
            if cmds[low] == "insert":
                self.request_insert()
            else:
                self.set_tool(cmds[low])
            return True
        if low == "c" and not self.draft:
            self.set_tool("circle")
            return True
        if low in ("h", "hole", "opening"):
            self.hole_mode = not self.hole_mode
            self._prompt()
            return True
        if low in ("u", "undo"):
            self.undo()
            return True
        if low == "redo":
            self.redo()
            return True
        if low in ("f", "fit", "z", "zoom", "ze"):
            self.fit()
            return True
        if low in ("e", "erase", "del", "delete"):
            self.delete_selected()
            return True
        if low in ("o", "ortho"):
            self.ortho = not self.ortho
            self.modes_changed.emit()
            return True
        if low in ("os", "osnap", "snap"):
            self.osnap_on = not self.osnap_on
            self.modes_changed.emit()
            return True
        if low in ("fl", "flip"):
            self.mirror_selected()
            return True
        m = re.match(rf"^g(?:rid)?\s+({NUM})$", low)
        if m and float(m.group(1)) > 0:
            self.grid = float(m.group(1)) / self.k
            self.update()
            return True
        if self.tool in ("polygon", "rect", "circle", "insert"):
            p = parse_point(t, self.ref_point() if self.tool == "polygon" else (self.draft[0] if self.draft else None),
                            self.cursor, self.k)
            if p is not None:
                self.input_point(p)
                return True
        return False

    def request_insert(self):
        d = InsertShapeDialog(self.du, self)
        if d.exec() == QtWidgets.QDialog.Accepted and d.result_shapes:
            self.insert_shapes = d.result_shapes
            self.tool = "insert"
            self._prompt()
            self.update()

    # ------------------------------------------------------------ CAD edit commands
    OP_NEEDS_SELECTION = ("move", "copy", "mirror", "rotate")

    def start_op(self, name: str):
        """move / copy / mirror / rotate (on the selection: select first, or
        pick objects and press Enter), trim (cutting line), paste."""
        self.draft = []
        self.tool = "select"
        if name == "paste":
            if not self.clipboard:
                self.prompt_changed.emit("Nothing to paste - Ctrl+C copies the selection")
                return
            self.op = {"name": "paste", "stage": "point", "pts": []}
        elif name in self.OP_NEEDS_SELECTION:
            self.op = {"name": name, "stage": "point" if self.selected else "select", "pts": []}
        elif name == "trim":
            self.op = {"name": "trim", "stage": "point", "pts": []}
        self._prompt()
        self.update()

    def _op_prompt(self) -> str:
        op = self.op
        n, st, k = op["name"], op["stage"], len(op["pts"])
        if st == "select":
            return f"{n.capitalize()}: select objects (click / window), then Enter or right-click"
        if n in ("move", "copy"):
            if k == 0:
                return f"{n.capitalize()}: base point (click, or type x,y)"
            return (f"{n.capitalize()}: second point (click, @dx,dy, @L<angle or a distance)"
                    + ("; Enter / Esc to finish" if n == "copy" else ""))
        if n == "paste":
            return "Paste: insertion point (click, or type x,y)"
        if n == "mirror":
            if st == "erase":
                return "Mirror: erase the source objects? [y / N] (Enter = keep them)"
            return "Mirror: " + ("first point of the mirror line" if k == 0 else "second point of the mirror line")
        if n == "rotate":
            return "Rotate: " + ("base point" if k == 0 else "angle in degrees (type, or click a point)")
        if n == "trim":
            if st == "pick":
                return "Trim: click the parts to remove; Enter / Esc to finish"
            return "Trim: " + ("first point of the cutting line" if k == 0 else "second point of the cutting line")
        return ""

    def _ghost_transform(self):
        """(shapes, function) previewing the command at the cursor, or None."""
        op = self.op
        if not op:
            return None
        n, pts, cur = op["name"], op["pts"], self.cursor
        if n == "paste" and self.clipboard:
            shapes, base = self.clipboard
            return shapes, (lambda P, d=cur - base: P + d)
        if not pts or op["stage"] != "point":
            return None
        sel = [self.shapes[i] for i in sorted(self.selected)]
        if n in ("move", "copy"):
            return sel, (lambda P, d=cur - pts[0]: P + d)
        if n == "mirror":
            return sel, _mirror_fn(pts[0], cur)
        if n == "rotate":
            a = math.atan2(cur[1] - pts[0][1], cur[0] - pts[0][0])
            return sel, _rotate_fn(pts[0], a)
        return None

    def op_point(self, p: np.ndarray):
        op = self.op
        n = op["name"]
        if op["stage"] == "select":
            return
        op["pts"].append(p)
        pts = op["pts"]
        if n == "paste":
            shapes, base = self.clipboard
            fn = (lambda P, d=p - base: P + d)
            self.add_shapes([_apply(g, fn) for g in shapes])
            self.op = None
        elif n == "move" and len(pts) == 2:
            fn = (lambda P, d=pts[1] - pts[0]: P + d)
            self._apply_selected(fn)
            self.op = None
        elif n == "copy" and len(pts) >= 2:
            fn = (lambda P, d=pts[-1] - pts[0]: P + d)
            keep = set(self.selected)
            new = [_apply(self.shapes[i], fn) for i in sorted(keep)]
            self.push_undo()
            self.shapes += new
            self.selected = keep  # keep copying the same objects from the same base point
            op["pts"] = [pts[0]]
            self._edited()
        elif n == "mirror" and len(pts) == 2:
            if np.linalg.norm(pts[1] - pts[0]) <= 1e-12:
                op["pts"].pop()
                return
            op["stage"] = "erase"
        elif n == "rotate" and len(pts) == 2:
            a = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
            self._apply_selected(_rotate_fn(pts[0], a))
            self.op = None
        elif n == "trim" and len(pts) == 2:
            if np.linalg.norm(pts[1] - pts[0]) <= 1e-12:
                op["pts"].pop()
                return
            self.trim_by_line(pts[0], pts[1])
            op["stage"] = "pick"
        if self.op is None:
            self.set_tool("select")
        self._prompt()
        self.update()

    def op_click(self, w: np.ndarray):
        """A left click while a command runs."""
        op = self.op
        if op["stage"] == "select":
            i = self.shape_at(w)
            if i is not None:
                self.selected ^= {i}
                self.selection_changed.emit()
            else:
                self._drag = ("band", self.to_screen(w), self.to_screen(w))
            self.update()
            return
        if op["stage"] == "pick":  # trim: remove the clicked piece
            i = self.shape_at(w)
            if i is not None:
                self.push_undo()
                self.shapes.pop(i)
                self.selected = set()
                self._edited()
            return
        if op["stage"] == "erase":
            self.op_command("n")
            return
        p, _ = self.snapped(w, self.ref_point())
        self.op_point(p)

    def op_enter(self):
        op = self.op
        if op is None:
            return
        if op["stage"] == "select":
            if self.selected:
                op["stage"] = "point"
            else:
                self.op = None
                self.set_tool("select")
        elif op["stage"] == "erase":
            self.op_command("n")
            return
        else:
            self.op = None
            self.set_tool("select")
        self._prompt()
        self.update()

    def op_command(self, t: str) -> bool:
        op = self.op
        low = t.strip().lower()
        if not low:
            self.op_enter()
            return True
        if op["stage"] == "erase":
            a, b = op["pts"][0], op["pts"][1]
            fn = _mirror_fn(a, b)
            if low in ("y", "yes"):
                self._apply_selected(fn)
            else:
                keep = sorted(self.selected)
                self.add_shapes([_apply(self.shapes[i], fn) for i in keep])
            self.op = None
            self.set_tool("select")
            return True
        if op["name"] == "rotate" and len(op["pts"]) == 1 and RE_NUM.match(t):
            self._apply_selected(_rotate_fn(op["pts"][0], math.radians(float(t))))
            self.op = None
            self.set_tool("select")
            return True
        if op["stage"] != "point":
            return False
        p = parse_point(t, self.ref_point(), self.cursor, self.k)
        if p is None:
            return False
        self.op_point(p)
        return True

    def _apply_selected(self, fn):
        if not self.selected:
            return
        self.push_undo()
        for i in self.selected:
            self.shapes[i] = _apply(self.shapes[i], fn)
        self._edited()

    def trim_by_line(self, a: np.ndarray, b: np.ndarray):
        """Cut the shapes the line a-b crosses (the selected ones, if any) into
        the parts on either side; the parts can then be clicked away."""
        idx = sorted(self.selected) or list(range(len(self.shapes)))
        d = b - a
        new, changed = [], False
        for i, g in enumerate(self.shapes):
            P = shape_points(g)
            if i not in idx or len(P) < 3:
                new.append(g)
                continue
            side = d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])
            if not (side > 1e-12 * self.scale ** 0).any() or not (side < -1e-12).any():
                new.append(g)
                continue
            for sgn in (1.0, -1.0):
                Q = _clip_halfplane(P, a, b, sgn)
                if len(Q) >= 3 and abs(signed_area(Q)) > 1e-14:
                    h = {k: v for k, v in g.items() if k not in ("center", "radius", "segments", "dims", "role")}
                    h["kind"] = "polygon"
                    h["points"] = Q.tolist()
                    new.append(h)
            changed = True
        if changed:
            self.push_undo()
            self.shapes = new
            self.selected = set()
            self._edited()
        else:
            self.prompt_changed.emit("Trim: the line does not cut any shape")

    def copy_to_clipboard(self, cut: bool = False):
        if not self.selected:
            return
        shapes = [copy.deepcopy(self.shapes[i]) for i in sorted(self.selected)]
        pts = np.vstack([shape_points(g) for g in shapes])
        base = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        self.clipboard = (shapes, base)
        if cut:
            self.delete_selected()
        self.prompt_changed.emit(f"{len(shapes)} shape(s) copied - Ctrl+V to paste")


    # ------------------------------------------------------------ hit tests
    def shape_at(self, p: np.ndarray) -> Optional[int]:
        tol = self._tol() * 0.6
        for i in range(len(self.shapes) - 1, -1, -1):
            P = shape_points(self.shapes[i])
            if len(P) < 3:
                continue
            if points_in_polygon(p[None], P)[0]:
                return i
            for a, b in zip(P, np.roll(P, -1, axis=0)):
                ab = b - a
                L2 = float(ab @ ab)
                t = 0.0 if L2 <= 0 else min(max(float((p - a) @ ab) / L2, 0.0), 1.0)
                if np.linalg.norm(a + t * ab - p) <= tol:
                    return i
        return None

    def handle_at(self, p: np.ndarray):
        tol = self._tol() * 0.7
        for i in sorted(self.selected):
            g = self.shapes[i]
            if g.get("kind") == "circle":
                c, r = np.asarray(g["center"], float), float(g["radius"])
                if np.linalg.norm(c + np.array([r, 0]) - p) <= tol:
                    return (i, "radius")
                continue
            for j, q in enumerate(g["points"]):
                if np.linalg.norm(np.asarray(q) - p) <= tol:
                    return (i, j)
        return None

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, ev):
        self.setFocus()
        pos = ev.position()
        w = self.to_world(pos)
        if ev.button() == QtCore.Qt.MiddleButton:
            self._drag = ("pan", pos, self.center.copy())
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            return
        if ev.button() == QtCore.Qt.RightButton:
            if self.op:
                self.op_enter()
                return
            if self.tool == "polygon" and len(self.draft) >= 3:
                self.finish_polygon()
            elif self.draft:
                self.cancel()
            else:
                self._context_menu(ev.globalPosition().toPoint(), w)
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        if self.op:
            self.op_click(w)
            return
        if self.tool != "select":
            p, _ = self.snapped(w, self.ref_point() if self.tool != "rect" else (self.draft[0] if self.draft else None))
            self.input_point(p)
            return
        h = self.handle_at(w)
        if h is not None:
            self.push_undo()
            self._drag = ("vertex", h[0], h[1])
            return
        i = self.shape_at(w)
        if i is not None:
            if ev.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
                self.selected ^= {i}
            elif i not in self.selected:
                self.selected = {i}
            self.selection_changed.emit()
            p, _ = self.snapped(w)
            self._drag = ("move", p, copy.deepcopy(self.shapes), False)
            self.update()
            return
        if not ev.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
            self.selected = set()
            self.selection_changed.emit()
        self._drag = ("band", pos, pos)
        self.update()

    def mouseDoubleClickEvent(self, ev):
        if self.tool == "polygon" and len(self.draft) >= 3:
            self.finish_polygon()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        self._shift = bool(ev.modifiers() & QtCore.Qt.ShiftModifier) and self.tool != "select"
        w = self.to_world(pos)
        d = self._drag
        if d and d[0] == "pan":
            self.center = d[2] - np.array([(pos.x() - d[1].x()) / self.scale, -(pos.y() - d[1].y()) / self.scale])
            self.update()
            return
        ref = self.ref_point() if self.tool != "rect" else (self.draft[0] if self.draft else None)
        if d and d[0] == "vertex":
            i, j = d[1], d[2]
            g = self.shapes[i]
            p, self.snap_kind = self.snapped(w)
            if j == "radius":
                g["radius"] = max(float(np.linalg.norm(p - np.asarray(g["center"]))), 1e-9)
            else:
                g["points"][j] = [float(p[0]), float(p[1])]
            self.cursor = p
            self.update_region()
            self.changed.emit()
        elif d and d[0] == "move":
            p, self.snap_kind = self.snapped(w, d[1])
            delta = p - d[1]
            base = d[2]
            if np.linalg.norm(delta) > 0 and not d[3]:
                self.undo_stack.append(copy.deepcopy(base))
                self.redo_stack.clear()
                self._drag = d = ("move", d[1], d[2], True)
            for i in self.selected:
                g0, g = base[i], self.shapes[i]
                if g.get("kind") == "circle":
                    g["center"] = (np.asarray(g0["center"]) + delta).tolist()
                else:
                    g["points"] = (np.asarray(g0["points"]) + delta).tolist()
            self.cursor = p
            self.update_region()
            self.changed.emit()
        elif d and d[0] == "band":
            self._drag = ("band", d[1], pos)
        else:
            picking = self.op and self.op.get("stage") in ("select", "pick")
            self.cursor, self.snap_kind = self.snapped(w, ref) if (self.tool != "select" or self.op) and not picking \
                else (w, "")
        x, y = self.cursor * self.k
        txt = f"x = {fmt(x, 5)}   y = {fmt(y, 5)} {self.unit}"
        if ref is not None:
            dd = (self.cursor - ref) * self.k
            txt += f"    length {fmt(math.hypot(*dd), 5)}  angle {fmt(math.degrees(math.atan2(dd[1], dd[0])), 4)} deg"
        if self.snap_kind:
            txt += f"    [{self.snap_kind}]"
        self.cursor_moved.emit(txt)
        self.update()

    def mouseReleaseEvent(self, ev):
        d = self._drag
        self._drag = None
        self.setCursor(QtCore.Qt.CrossCursor)
        if not d:
            return
        if d[0] == "band":
            a, b = d[1], d[2]
            if abs(a.x() - b.x()) < 3 and abs(a.y() - b.y()) < 3:
                self.update()
                return
            p0, p1 = self.to_world(a), self.to_world(b)
            lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
            crossing = b.x() < a.x()  # right to left: everything touched
            for i, g in enumerate(self.shapes):
                P = shape_points(g)
                inside = (P >= lo).all(axis=1) & (P <= hi).all(axis=1)
                if inside.all() or (crossing and inside.any()):
                    self.selected.add(i)
            self.selection_changed.emit()
            self.update()
        elif d[0] in ("vertex", "move"):
            if d[0] == "move" and not d[3]:
                return
            for i in list(self.selected):
                g = self.shapes[i]
                if g.get("kind") != "circle" and len(g["points"]) >= 3 and self_intersecting(np.asarray(g["points"])):
                    self.prompt_changed.emit("Warning: a polygon now crosses itself - Ctrl+Z to undo")
            self._edited()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == QtCore.Qt.Key_Escape:
            self.cancel()
        elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if self.op:
                self.op_enter()
            else:
                self.finish_polygon()
        elif key == QtCore.Qt.Key_Backspace and self.draft:
            self.draft.pop()
            self._prompt()
            self.update()
        elif key == QtCore.Qt.Key_Delete:
            self.delete_selected()
        elif key == QtCore.Qt.Key_F8:
            self.ortho = not self.ortho
            self.update()
            self.modes_changed.emit()
        elif key == QtCore.Qt.Key_F3:
            self.osnap_on = not self.osnap_on
            self.update()
            self.modes_changed.emit()
        elif key == QtCore.Qt.Key_F7:
            self.show_grid = not self.show_grid
            self.update()
            self.modes_changed.emit()
        elif key == QtCore.Qt.Key_F9:
            self.snap["grid"] = not self.snap["grid"]
            self.update()
            self.modes_changed.emit()
        elif key == QtCore.Qt.Key_Shift:
            self._shift = self.tool != "select"
        else:
            if ev.text() and ev.text().isprintable():  # typing goes to the command line, like CAD
                self.parent_type(ev.text())
                return
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Shift:
            self._shift = False
        super().keyReleaseEvent(ev)

    def parent_type(self, text: str):
        """Hook: the dialog puts typed characters into the command line."""

    def _context_menu(self, gpos, w):
        menu = QtWidgets.QMenu(self)
        i = self.shape_at(w)
        if i is not None and i not in self.selected:
            self.selected = {i}
            self.selection_changed.emit()
            self.update()
        acts = []
        if self.selected:
            acts = [("Move", lambda: self.start_op("move")), ("Copy", lambda: self.start_op("copy")),
                    ("Mirror...", lambda: self.start_op("mirror")), ("Rotate...", lambda: self.start_op("rotate")),
                    ("Flip left / right", self.mirror_selected), ("Rotate 90 deg", lambda: self.rotate_selected(90)),
                    ("Copy to clipboard (Ctrl+C)", self.copy_to_clipboard),
                    ("Solid <-> opening", self.toggle_hole_selected), ("Delete", self.delete_selected)]
        if self.clipboard:
            acts.append(("Paste (Ctrl+V)", lambda: self.start_op("paste")))
        acts.append(("Trim with a line", lambda: self.start_op("trim")))
        acts += [("Select all", self.select_all), ("Zoom to fit", self.fit)]
        for t, f in acts:
            menu.addAction(t, f)
        menu.exec(gpos)

    # ------------------------------------------------------------ painting
    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.palette()
        base = pal.color(QtGui.QPalette.Base)
        dark = base.lightness() < 128
        text = pal.color(QtGui.QPalette.Text)
        p.fillRect(self.rect(), base)
        self._paint_grid(p, dark)
        # the region (union of solids minus openings)
        if self.region_loops:
            path = QtGui.QPainterPath()
            path.setFillRule(QtCore.Qt.OddEvenFill)
            for L in self.region_loops:
                pts = [self.to_screen(q) for q in L]
                path.addPolygon(QtGui.QPolygonF(pts + pts[:1]))
            fill = QtGui.QColor("#7fa3c7" if self.steel else "#c8c2b4")
            if dark:
                fill = QtGui.QColor("#4f7396" if self.steel else "#7b766c")
            fill.setAlpha(210)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(fill)
            p.drawPath(path)
        # shapes of another material (steel in concrete) in their own colour
        for g in self.shapes:
            if g.get("material") and not g.get("hole"):
                P = shape_points(g)
                if len(P) >= 3:
                    col = QtGui.QColor("#4f7396" if dark else "#7fa3c7")
                    p.setPen(QtCore.Qt.NoPen)
                    p.setBrush(col)
                    p.drawPolygon(QtGui.QPolygonF([self.to_screen(q) for q in P]))
        # every drawn shape: solids solid, openings dashed
        for i, g in enumerate(self.shapes):
            P = shape_points(g)
            if len(P) < 2:
                continue
            sel = i in self.selected
            col = QtGui.QColor("#e67e22") if sel else text
            pen = QtGui.QPen(col, 2.2 if sel else 1.2, QtCore.Qt.DashLine if g.get("hole") else QtCore.Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            pts = [self.to_screen(q) for q in P]
            p.drawPolygon(QtGui.QPolygonF(pts))
            if sel:
                p.setPen(QtGui.QPen(col, 1.0))
                p.setBrush(base)
                hs = [np.asarray(g["center"], float) + np.array([float(g["radius"]), 0])] if g.get("kind") == "circle" \
                    else [np.asarray(q, float) for q in g["points"]]
                if g.get("kind") == "circle":
                    hs.append(np.asarray(g["center"], float))
                for q in hs:
                    s = self.to_screen(q)
                    p.drawRect(QtCore.QRectF(s.x() - 3.5, s.y() - 3.5, 7, 7))
        # centroid and local axes
        if self.centroid is not None:
            c = self.to_screen(self.centroid)
            red = QtGui.QColor("#d43f3a")
            p.setPen(QtGui.QPen(red, 1.3))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(c, 6, 6)
            p.drawLine(QtCore.QPointF(c.x() - 9, c.y()), QtCore.QPointF(c.x() + 9, c.y()))
            p.drawLine(QtCore.QPointF(c.x(), c.y() - 9), QtCore.QPointF(c.x(), c.y() + 9))
            p.drawLine(c, QtCore.QPointF(c.x() + 45, c.y()))
            p.drawLine(c, QtCore.QPointF(c.x(), c.y() - 45))
            p.drawText(QtCore.QPointF(c.x() + 48, c.y() + 4), "3")
            p.drawText(QtCore.QPointF(c.x() - 3, c.y() - 49), "2")
        # the shape being drawn
        accent = QtGui.QColor("#1f6fd1") if not dark else QtGui.QColor("#6fb1ff")
        p.setPen(QtGui.QPen(accent, 1.6, QtCore.Qt.DashLine if self.hole_mode else QtCore.Qt.SolidLine))
        p.setBrush(QtCore.Qt.NoBrush)
        cur = self.cursor
        if self.tool == "polygon" and self.draft:
            pts = [self.to_screen(q) for q in self.draft] + [self.to_screen(cur)]
            p.drawPolyline(QtGui.QPolygonF(pts))
            if len(self.draft) >= 2:
                p.setPen(QtGui.QPen(accent, 0.8, QtCore.Qt.DotLine))
                p.drawLine(self.to_screen(cur), self.to_screen(self.draft[0]))
        elif self.tool == "rect" and self.draft:
            a, b = self.to_screen(self.draft[0]), self.to_screen(cur)
            p.drawRect(QtCore.QRectF(a, b).normalized())
            w, h = np.abs(cur - self.draft[0]) * self.k
            p.drawText(QtCore.QPointF(max(a.x(), b.x()) + 6, min(a.y(), b.y()) - 4), f"{fmt(w, 4)} x {fmt(h, 4)}")
        elif self.tool == "circle" and self.draft:
            c = self.to_screen(self.draft[0])
            r = float(np.linalg.norm(cur - self.draft[0])) * self.scale
            p.drawEllipse(c, r, r)
            p.drawLine(c, self.to_screen(cur))
        elif self.op:
            g = self._ghost_transform()
            if g is not None:
                shapes, fn = g
                p.setPen(QtGui.QPen(accent, 1.4, QtCore.Qt.DashLine))
                for sh in shapes:
                    P = fn(shape_points(sh))
                    p.drawPolygon(QtGui.QPolygonF([self.to_screen(q) for q in P]))
            pts = self.op.get("pts") or []
            if pts and self.op["stage"] == "point":
                p.setPen(QtGui.QPen(accent, 1.0, QtCore.Qt.DashDotLine))
                p.drawLine(self.to_screen(pts[-1]), self.to_screen(cur))
            if self.op["name"] in ("mirror", "trim") and len(pts) == 2:
                p.setPen(QtGui.QPen(QtGui.QColor("#c0392b"), 1.2, QtCore.Qt.DashLine))
                p.drawLine(self.to_screen(pts[0]), self.to_screen(pts[1]))
        elif self.op:
            g = self._ghost_transform()
            if g is not None:
                shapes, fn = g
                p.setPen(QtGui.QPen(accent, 1.4, QtCore.Qt.DashLine))
                for sh in shapes:
                    P = fn(shape_points(sh))
                    p.drawPolygon(QtGui.QPolygonF([self.to_screen(q) for q in P]))
            pts = self.op.get("pts") or []
            if pts and self.op["stage"] == "point":
                p.setPen(QtGui.QPen(accent, 1.0, QtCore.Qt.DashDotLine))
                p.drawLine(self.to_screen(pts[-1]), self.to_screen(cur))
            if self.op["name"] in ("mirror", "trim") and len(pts) == 2:
                p.setPen(QtGui.QPen(QtGui.QColor("#c0392b"), 1.2, QtCore.Qt.DashLine))
                p.drawLine(self.to_screen(pts[0]), self.to_screen(pts[1]))
        elif self.tool == "insert" and self.insert_shapes:
            for g in self.insert_shapes:
                P = shape_points(g) + cur
                p.drawPolygon(QtGui.QPolygonF([self.to_screen(q) for q in P]))
        # snap marker
        if self.tool != "select" or (self.op and self.op["stage"] == "point") or \
                (self._drag and self._drag[0] in ("vertex", "move")):
            s = self.to_screen(self.cursor)
            p.setPen(QtGui.QPen(QtGui.QColor("#16a085"), 1.6))
            p.setBrush(QtCore.Qt.NoBrush)
            k = self.snap_kind
            if k in ("end point", "origin", "quadrant"):
                p.drawRect(QtCore.QRectF(s.x() - 6, s.y() - 6, 12, 12))
            elif k == "mid point":
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(s.x(), s.y() - 7), QtCore.QPointF(s.x() - 7, s.y() + 5),
                                               QtCore.QPointF(s.x() + 7, s.y() + 5)]))
            elif k in ("centre", "centroid"):
                p.drawEllipse(s, 6, 6)
            elif k == "intersection":
                p.drawLine(QtCore.QPointF(s.x() - 7, s.y() - 7), QtCore.QPointF(s.x() + 7, s.y() + 7))
                p.drawLine(QtCore.QPointF(s.x() - 7, s.y() + 7), QtCore.QPointF(s.x() + 7, s.y() - 7))
            elif k == "perpendicular":
                p.drawLine(QtCore.QPointF(s.x() - 7, s.y() + 6), QtCore.QPointF(s.x() + 7, s.y() + 6))
                p.drawLine(QtCore.QPointF(s.x() - 3, s.y() + 6), QtCore.QPointF(s.x() - 3, s.y() - 7))
            elif k == "on edge":
                p.drawLine(QtCore.QPointF(s.x() - 6, s.y() - 6), QtCore.QPointF(s.x() + 6, s.y() + 6))
                p.drawLine(QtCore.QPointF(s.x() - 6, s.y() + 6), QtCore.QPointF(s.x() + 6, s.y() - 6))
                p.drawEllipse(s, 2, 2)
            else:
                p.drawLine(QtCore.QPointF(s.x() - 8, s.y()), QtCore.QPointF(s.x() + 8, s.y()))
                p.drawLine(QtCore.QPointF(s.x(), s.y() - 8), QtCore.QPointF(s.x(), s.y() + 8))
        # selection window
        d = self._drag
        if d and d[0] == "band":
            a, b = d[1], d[2]
            crossing = b.x() < a.x()
            col = QtGui.QColor(40, 160, 60, 50) if crossing else QtGui.QColor(30, 110, 210, 50)
            p.setBrush(col)
            p.setPen(QtGui.QPen(col.darker(250), 1, QtCore.Qt.DashLine if crossing else QtCore.Qt.SolidLine))
            p.drawRect(QtCore.QRectF(a, b).normalized())
        # overall size
        if self.region_loops:
            allp = np.vstack(self.region_loops)
            lo, hi = allp.min(axis=0), allp.max(axis=0)
            p.setPen(QtGui.QPen(accent, 1.0))
            a, b = self.to_screen((lo[0], lo[1])), self.to_screen((hi[0], lo[1]))
            y = a.y() + 18
            p.drawLine(QtCore.QPointF(a.x(), y), QtCore.QPointF(b.x(), y))
            p.drawText(QtCore.QRectF(a.x(), y + 1, b.x() - a.x(), 16), QtCore.Qt.AlignCenter,
                       f"{fmt((hi[0] - lo[0]) * self.k, 4)}")
            c, e = self.to_screen((lo[0], lo[1])), self.to_screen((lo[0], hi[1]))
            x = c.x() - 18
            p.drawLine(QtCore.QPointF(x, c.y()), QtCore.QPointF(x, e.y()))
            p.save()
            p.translate(x - 3, 0.5 * (c.y() + e.y()))
            p.rotate(-90)
            p.drawText(QtCore.QRectF(-60, -16, 120, 15), QtCore.Qt.AlignCenter, f"{fmt((hi[1] - lo[1]) * self.k, 4)}")
            p.restore()
        # status corner
        p.setPen(text)
        flags = []
        if not self.osnap_on:
            flags.append("SNAP OFF")
        if self.ortho_on():
            flags.append("ORTHO")
        if self.hole_mode:
            flags.append("OPENING")
        flags.append(f"grid {fmt(self.grid * self.k, 4)} {self.unit}" + ("" if self.snap["grid"] else " (no snap)"))
        p.drawText(QtCore.QRectF(6, 4, self.width() - 12, 16), QtCore.Qt.AlignRight, "  ".join(flags))
        p.end()

    def _paint_grid(self, p, dark):
        g = self.grid
        if g <= 0 or not self.show_grid:
            return
        step = g
        while step * self.scale < 7:
            step *= 5
        lo = self.to_world(QtCore.QPointF(0, self.height()))
        hi = self.to_world(QtCore.QPointF(self.width(), 0))
        minor = QtGui.QColor(255, 255, 255, 18) if dark else QtGui.QColor(0, 0, 0, 16)
        major = QtGui.QColor(255, 255, 255, 40) if dark else QtGui.QColor(0, 0, 0, 38)
        i0, i1 = int(math.floor(lo[0] / step)), int(math.ceil(hi[0] / step))
        j0, j1 = int(math.floor(lo[1] / step)), int(math.ceil(hi[1] / step))
        if (i1 - i0) > 600 or (j1 - j0) > 600:
            return
        for i in range(i0, i1 + 1):
            x = self.to_screen((i * step, 0)).x()
            p.setPen(QtGui.QPen(major if i % 5 == 0 else minor, 1))
            p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, self.height()))
        for j in range(j0, j1 + 1):
            y = self.to_screen((0, j * step)).y()
            p.setPen(QtGui.QPen(major if j % 5 == 0 else minor, 1))
            p.drawLine(QtCore.QPointF(0, y), QtCore.QPointF(self.width(), y))
        o = self.to_screen((0.0, 0.0))
        p.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120, 140), 1, QtCore.Qt.DashDotLine))
        p.drawLine(QtCore.QPointF(o.x(), 0), QtCore.QPointF(o.x(), self.height()))
        p.drawLine(QtCore.QPointF(0, o.y()), QtCore.QPointF(self.width(), o.y()))


class InsertShapeDialog(QtWidgets.QDialog):
    """Pick a catalogue shape and its dimensions to insert as editable outlines."""

    def __init__(self, du: DisplayUnits, parent=None):
        super().__init__(parent)
        self.du = du
        self.k = du.factor("section_dim")
        self.L = 1.0 / du.units.factor("length")
        self.setWindowTitle("Insert Shape")
        self.result_shapes: List[dict] = []
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.kind = QtWidgets.QComboBox()
        from .section_manager import HIDDEN_SHAPES
        for key, sh in SHAPES.items():
            if key not in HIDDEN_SHAPES:
                self.kind.addItem(f"{sh.label}  ({sh.group})", key)
        left.addWidget(self.kind)
        self.form_box = QtWidgets.QGroupBox(f"Dimensions [{du.label('section_dim')}]")
        self.form = QtWidgets.QFormLayout(self.form_box)
        left.addWidget(self.form_box)
        self.mirror = QtWidgets.QCheckBox("Mirror")
        left.addWidget(self.mirror)
        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet("color: #c0392b")
        left.addWidget(self.error)
        left.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        left.addWidget(bb)
        lay.addLayout(left, 1)
        from .section_manager import SectionPreview
        self.preview = SectionPreview()
        lay.addWidget(self.preview, 1)
        self.edits: Dict[str, QtWidgets.QLineEdit] = {}
        self.kind.currentIndexChanged.connect(self._build)
        self.mirror.toggled.connect(self._changed)
        self._build()

    def _build(self, *_):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.edits = {}
        sh = SHAPES[self.kind.currentData()]
        for dm in sh.dims:
            e = QtWidgets.QLineEdit("" if dm.default is None else fmt(dm.default * self.L * self.k, 6))
            if dm.optional:
                e.setPlaceholderText(dm.placeholder or "optional")
            e.textChanged.connect(self._changed)
            self.edits[dm.attr] = e
            self.form.addRow(dm.label, e)
        self.mirror.setEnabled(sh.mirror)
        self._changed()

    def _values(self):
        v = {}
        for a, e in self.edits.items():
            t = e.text().strip()
            if not t:
                v[a] = None
                continue
            try:
                v[a] = float(t) / self.k
            except ValueError:
                return None
        if self.kind.currentData() in COLD:
            v["tf"] = v.get("tw")
        return v

    def _changed(self, *_):
        sh = SHAPES[self.kind.currentData()]
        v = self._values()
        errs = ["Type numbers"] if v is None else check_dims(sh, v)
        self.error.setText("\n".join(errs))
        if errs:
            self.preview.set_outline([], [], False, self.du, "Check the dimensions")
            return
        solids, holes = shape_outline(sh, v, self.mirror.isChecked())
        self.preview.set_outline(solids, holes, sh.group == "Steel", self.du)

    def _ok(self):
        sh = SHAPES[self.kind.currentData()]
        v = self._values()
        if v is None or check_dims(sh, v):
            return
        self.result_shapes = shape_from_catalogue(sh.key, v, (0.0, 0.0), self.mirror.isChecked())
        self.accept()


def _tool_icon(kind: str) -> QtGui.QIcon:
    from . import icons
    names = {"select": "select", "polygon": "slab", "rect": "rect_slab", "insert": "frame_sections",
             "hole": "opening", "delete": "delete", "copy": "copy", "mirror": "mirror", "rotate": "rotate",
             "undo": "undo", "redo": "redo", "fit": "zoom_full", "grid": "grid"}
    if kind == "circle":
        pm = QtGui.QPixmap(32, 32)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QPen(QtGui.QColor("#30343a"), 2))
        p.setBrush(QtGui.QColor("#b9b4a8"))
        p.drawEllipse(QtCore.QPointF(16, 16), 11, 11)
        p.fillRect(QtCore.QRectF(14.5, 14.5, 3, 3), QtGui.QColor("#1f6fd1"))
        p.end()
        return QtGui.QIcon(pm)
    if kind in ("paste", "trim"):
        pm = QtGui.QPixmap(32, 32)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if kind == "paste":
            p.setPen(QtGui.QPen(QtGui.QColor("#30343a"), 1.5))
            p.setBrush(QtGui.QColor("#e6e2d6"))
            p.drawRect(QtCore.QRectF(7, 6, 16, 20))
            p.setBrush(QtGui.QColor("#b9b4a8"))
            p.drawRect(QtCore.QRectF(13, 12, 14, 16))
        else:
            p.setPen(QtGui.QPen(QtGui.QColor("#30343a"), 1.5))
            p.setBrush(QtGui.QColor("#b9b4a8"))
            p.drawRect(QtCore.QRectF(5, 8, 22, 16))
            p.setPen(QtGui.QPen(QtGui.QColor("#c0392b"), 2.0, QtCore.Qt.DashLine))
            p.drawLine(QtCore.QPointF(20, 3), QtCore.QPointF(12, 29))
        p.end()
        return QtGui.QIcon(pm)
    if kind == "hole_toggle":
        pm = QtGui.QPixmap(32, 32)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QPen(QtGui.QColor("#30343a"), 1.5))
        p.setBrush(QtGui.QColor("#b9b4a8"))
        p.drawRect(QtCore.QRectF(4, 4, 24, 24))
        p.setBrush(QtGui.QColor("white"))
        p.setPen(QtGui.QPen(QtGui.QColor("#1f6fd1"), 1.5, QtCore.Qt.DashLine))
        p.drawRect(QtCore.QRectF(10, 10, 12, 12))
        p.end()
        return QtGui.QIcon(pm)
    try:
        return icons.icon(names.get(kind, kind))
    except Exception:
        return QtGui.QIcon()


class SectionDesignerDialog(QtWidgets.QDialog):
    """Section Designer: name, material, the drawing (tools on the left, command
    line below), properties and the coordinates of the selected shape on the
    right."""

    def __init__(self, model, sec: Optional[FrameSection] = None, parent=None, existing: Sequence[str] = ()):
        super().__init__(parent)
        self.model = model
        self.du = DisplayUnits.of(model)
        self.k = self.du.factor("section_dim")
        sec = sec or FrameSection(name=unique_name("SD1", list(existing)), material=material_for(model, "Concrete"),
                                  shape="Section Designer")
        self.sec = copy.deepcopy(sec)
        self.existing = [n for n in existing if n != sec.name]
        self.setWindowTitle("Section Designer")
        self.resize(1280, 780)
        outer = QtWidgets.QVBoxLayout(self)
        # --- top row
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Section name"))
        self.name = QtWidgets.QLineEdit(sec.name)
        top.addWidget(self.name)
        top.addWidget(QtWidgets.QLabel("Material"))
        self.material = QtWidgets.QComboBox()
        self.material.addItems(list(model.materials))
        self.material.setCurrentText(sec.material)
        top.addWidget(self.material)
        top.addWidget(QtWidgets.QLabel(f"Grid [{self.du.label('section_dim')}]"))
        self.grid = QtWidgets.QDoubleSpinBox()
        self.grid.setDecimals(4)
        self.grid.setRange(1e-6, 1e9)
        top.addWidget(self.grid)
        self.osnap = QtWidgets.QCheckBox("Snap (F3)")
        self.osnap.setChecked(True)
        self.osnap.setToolTip("Object snaps on / off")
        top.addWidget(self.osnap)
        self.snap_button = QtWidgets.QToolButton()
        self.snap_button.setText("Snaps")
        self.snap_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self.snap_button)
        self.snap_actions: Dict[str, QtGui.QAction] = {}
        for key, label in (("end", "End point"), ("mid", "Mid point"), ("centre", "Centre / quadrant / centroid"),
                           ("intersection", "Intersection"), ("perpendicular", "Perpendicular"),
                           ("nearest", "Nearest (on an edge)")):
            a = menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(True)
            self.snap_actions[key] = a
        self.snap_button.setMenu(menu)
        top.addWidget(self.snap_button)
        self.show_grid = QtWidgets.QCheckBox("Grid (F7)")
        self.show_grid.setChecked(True)
        top.addWidget(self.show_grid)
        self.grid_snap = QtWidgets.QCheckBox("Grid snap (F9)")
        self.grid_snap.setChecked(True)
        top.addWidget(self.grid_snap)
        self.ortho = QtWidgets.QCheckBox("Ortho (F8)")
        top.addWidget(self.ortho)
        top.addStretch(1)
        outer.addLayout(top)
        # --- middle: tools | canvas | panel
        mid = QtWidgets.QHBoxLayout()
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setOrientation(QtCore.Qt.Vertical)
        self.toolbar.setIconSize(QtCore.QSize(26, 26))
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        mid.addWidget(self.toolbar)
        center = QtWidgets.QVBoxLayout()
        self.canvas = DesignerCanvas(self.du)
        self.canvas.parent_type = self._type_into_command
        center.addWidget(self.canvas, 1)
        self.prompt = QtWidgets.QLabel("")
        center.addWidget(self.prompt)
        cmd_row = QtWidgets.QHBoxLayout()
        cmd_row.addWidget(QtWidgets.QLabel("Command:"))
        self.command = QtWidgets.QLineEdit()
        self.command.setPlaceholderText("x,y   @dx,dy   @length<angle   a distance   c = close   p r ci i   h = opening   u = undo")
        self.command.returnPressed.connect(self._command)
        cmd_row.addWidget(self.command, 1)
        center.addLayout(cmd_row)
        self.coords = QtWidgets.QLabel("")
        center.addWidget(self.coords)
        mid.addLayout(center, 5)
        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("<b>Section properties</b>"))
        self.props = props_table()
        side.addWidget(self.props, 3)
        self.warn = QtWidgets.QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #c0392b")
        side.addWidget(self.warn)
        side.addWidget(QtWidgets.QLabel("<b>Selected shape</b>  (type to edit)"))
        self.points = QtWidgets.QTableWidget(0, 2)
        self.points.setHorizontalHeaderLabels([f"x [{self.du.label('section_dim')}]", f"y [{self.du.label('section_dim')}]"])
        self.points.horizontalHeader().setStretchLastSection(True)
        self.points.itemChanged.connect(self._point_edited)
        side.addWidget(self.points, 2)
        prow = QtWidgets.QHBoxLayout()
        self.btn_add_pt = QtWidgets.QPushButton("Add point")
        self.btn_add_pt.clicked.connect(self._add_point)
        self.btn_del_pt = QtWidgets.QPushButton("Remove point")
        self.btn_del_pt.clicked.connect(self._remove_point)
        prow.addWidget(self.btn_add_pt)
        prow.addWidget(self.btn_del_pt)
        side.addLayout(prow)
        mrow = QtWidgets.QHBoxLayout()
        mrow.addWidget(QtWidgets.QLabel("Material of selected"))
        self.shape_mat = QtWidgets.QComboBox()
        self.shape_mat.addItem("(section material)", "")
        for name in model.materials:
            self.shape_mat.addItem(name, name)
        self.shape_mat.setToolTip("Another material makes a composite section (e.g. a steel shape in concrete): "
                                  "properties transformed with n = E / E(section material)")
        self.shape_mat.activated.connect(self._set_shape_material)
        mrow.addWidget(self.shape_mat, 1)
        side.addLayout(mrow)
        mb = QtWidgets.QPushButton("Property Modifiers...")
        mb.clicked.connect(self._modifiers)
        side.addWidget(mb)
        mid.addLayout(side, 3)
        outer.addLayout(mid, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet("color: #c0392b")
        brow = QtWidgets.QHBoxLayout()
        brow.addWidget(self.error, 1)
        brow.addWidget(bb)
        outer.addLayout(brow)
        self._build_tools()
        # --- state
        geom = [dict(g) for g in sec.geometry] if sec.geometry else []
        size = max(sec.t3 or 0, sec.t2 or 0) if geom else 0.0
        if not size:
            size = 0.6 / model.units.factor("length")
        disp = nice_step(size * self.k / 30.0)
        self.grid.setValue(disp)
        self.canvas.grid = disp / self.k
        self.grid.valueChanged.connect(lambda v: (setattr(self.canvas, "grid", v / self.k), self.canvas.update()))
        for key, a in self.snap_actions.items():
            a.toggled.connect(lambda on, k=key: self.canvas.snap.__setitem__(k, on))
        self.osnap.toggled.connect(lambda on: (setattr(self.canvas, "osnap_on", on), self.canvas.update()))
        self.show_grid.toggled.connect(lambda on: (setattr(self.canvas, "show_grid", on), self.canvas.update()))
        self.grid_snap.toggled.connect(lambda on: (self.canvas.snap.__setitem__("grid", on), self.canvas.update()))
        self.ortho.toggled.connect(lambda on: (setattr(self.canvas, "ortho", on), self.canvas.update()))
        self.canvas.modes_changed.connect(self._sync_modes)
        self.material.currentTextChanged.connect(self._material)
        self._material(self.material.currentText())
        self.canvas.changed.connect(self._schedule)
        self.canvas.selection_changed.connect(self._fill_points)
        self.canvas.prompt_changed.connect(self.prompt.setText)
        self.canvas.cursor_moved.connect(self.coords.setText)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._recompute)
        self.props_result = None
        QtCore.QTimer.singleShot(0, lambda: self.canvas.set_shapes(geom, fit=True))
        if not geom:
            QtCore.QTimer.singleShot(0, lambda: self._set_tool("rect"))
        for seq, fn in (("Ctrl+Z", self.canvas.undo), ("Ctrl+Y", self.canvas.redo), ("Ctrl+Shift+Z", self.canvas.redo),
                        ("Ctrl+A", self.canvas.select_all), ("F8", lambda: self.ortho.toggle()),
                        ("F3", lambda: self.osnap.toggle()), ("F7", lambda: self.show_grid.toggle()),
                        ("F9", lambda: self.grid_snap.toggle()),
                        ("Ctrl+C", lambda: self.canvas.copy_to_clipboard()),
                        ("Ctrl+X", lambda: self.canvas.copy_to_clipboard(cut=True)),
                        ("Ctrl+V", lambda: self._op("paste"))):
            QtGui.QShortcut(QtGui.QKeySequence(seq), self, activated=fn)
        self.canvas.setFocus()

    # -- tools
    def _build_tools(self):
        self.tool_actions: Dict[str, QtGui.QAction] = {}
        group = QtGui.QActionGroup(self)
        for key, tip in (("select", "Select / move / edit points (S)"), ("polygon", "Polygon (P)"),
                         ("rect", "Rectangle (R)"), ("circle", "Circle (CI)"),
                         ("insert", "Insert shape: I, T, channel, angle, pipe ... (I)")):
            a = QtGui.QAction(_tool_icon(key), tip, self)
            a.setCheckable(True)
            a.triggered.connect(lambda _=False, k=key: self._set_tool(k))
            group.addAction(a)
            self.toolbar.addAction(a)
            self.tool_actions[key] = a
        self.tool_actions["select"].setChecked(True)
        self.toolbar.addSeparator()
        self.hole_action = QtGui.QAction(_tool_icon("hole_toggle"), "Draw openings (holes) (H)", self)
        self.hole_action.setCheckable(True)
        self.hole_action.toggled.connect(self._hole_mode)
        self.toolbar.addAction(self.hole_action)
        self.toolbar.addSeparator()
        for key, tip, fn in (("move", "Move: base point -> second point (M)", lambda: self._op("move")),
                             ("copy", "Copy: base point -> second point, repeat (CO)", lambda: self._op("copy")),
                             ("paste", "Paste (Ctrl+V; Ctrl+C / Ctrl+X copy / cut the selection)",
                              lambda: self._op("paste")),
                             ("mirror", "Mirror about a line, keep or erase the source (MI)", lambda: self._op("mirror")),
                             ("rotate", "Rotate about a base point, angle typed or picked (RO)", lambda: self._op("rotate")),
                             ("trim", "Trim: cut along a line, click the parts to remove (TR)", lambda: self._op("trim")),
                             ("delete", "Delete selected (Del)", lambda: self.canvas.delete_selected()),
                             ("hole", "Selected: solid <-> opening", lambda: self.canvas.toggle_hole_selected()),
                             ("undo", "Undo (Ctrl+Z)", lambda: self.canvas.undo()),
                             ("redo", "Redo (Ctrl+Y)", lambda: self.canvas.redo()),
                             ("fit", "Zoom to fit (F)", lambda: self.canvas.fit())):
            a = QtGui.QAction(_tool_icon(key), tip, self)
            a.triggered.connect(fn)
            self.toolbar.addAction(a)
            self.tool_actions[key] = a

    def _op(self, name: str):
        self.canvas.start_op(name)
        self.tool_actions["select"].setChecked(True)
        self.canvas.setFocus()

    def _sync_modes(self):
        cv = self.canvas
        for w, v in ((self.osnap, cv.osnap_on), (self.show_grid, cv.show_grid), (self.grid_snap, cv.snap["grid"]),
                     (self.ortho, cv.ortho)):
            w.blockSignals(True)
            w.setChecked(v)
            w.blockSignals(False)

    def _set_tool(self, key: str):
        if key == "insert":
            self.canvas.request_insert()
            key = self.canvas.tool
        else:
            self.canvas.set_tool(key)
        if key in self.tool_actions:
            self.tool_actions[key].setChecked(True)
        self.canvas.setFocus()

    def _hole_mode(self, on: bool):
        self.canvas.hole_mode = on
        self.canvas._prompt()
        self.canvas.update()

    def _material(self, name):
        mat = self.model.materials.get(name)
        self.canvas.steel = mat is not None and mat.type in STEEL_TYPES
        self.canvas.update()
        self._schedule()

    def _type_into_command(self, text: str):
        self.command.setFocus()
        self.command.insert(text)

    def _command(self):
        t = self.command.text()
        ok = self.canvas.command(t)
        if ok:
            self.command.clear()
            low = t.strip().lower()
            if low in ("h", "hole", "opening"):
                self.hole_action.setChecked(self.canvas.hole_mode)
            if low in ("o", "ortho"):
                self.ortho.setChecked(self.canvas.ortho)
            if self.canvas.tool in self.tool_actions:
                self.tool_actions[self.canvas.tool].setChecked(True)
        else:
            self.prompt.setText(f"Not understood: {t!r} - type x,y  @dx,dy  @length<angle  or a command")
        self.canvas.update()

    # -- properties
    def _schedule(self):
        if hasattr(self, "_timer"):
            self._timer.start()

    def _recompute(self):
        shapes = self.canvas.shapes
        warn = []
        if not shapes:
            self.props.setRowCount(0)
            self.warn.setText("Draw the section: rectangle, polygon or circle (or insert a shape).")
            self.props_result = None
            return
        try:
            from ..model.section_shapes import material_ratios
            ratios = material_ratios(shapes, self.material.currentText(), self.model.materials)
            p, mesh = designer_properties(shapes, torsion=True, ratios=ratios)
        except Exception as exc:  # degenerate drawing while editing
            self.warn.setText(f"Cannot compute the properties: {exc}")
            self.props_result = None
            return
        self.props_result = p
        if p.A <= 0:
            self.props.setRowCount(0)
            self.warn.setText("No material: the openings cover everything.")
            return
        outer = [L for L in mesh.loops if signed_area(L) > 0]
        if len(outer) > 1:
            warn.append(f"{len(outer)} separate parts: they act as one section (rigidly connected).")
        solids, holes = designer_polygons(shapes)
        for H in holes:
            if not any(points_in_polygon(H, S).any() for S in solids):
                warn.append("An opening lies outside the material.")
                break
        tmp = copy.deepcopy(self.sec)
        tmp.material = self.material.currentText()
        tmp.geometry = copy.deepcopy(shapes)
        rows = property_rows(self.model, tmp, p)
        kd, ud = self.k, self.du.label("section_dim")
        tiny = 1e-9 * max(p.width, p.depth)
        cx, cy = (0.0 if abs(p.xc) < tiny else p.xc), (0.0 if abs(p.yc) < tiny else p.yc)
        rows.insert(1, ("Centroid (x, y)", f"{fmt(cx * kd, 5)}, {fmt(cy * kd, 5)}", ud))
        rows.insert(2, ("Size (width x depth)", f"{fmt(p.width * kd, 5)} x {fmt(p.depth * kd, 5)}", ud))
        fill_table(self.props, rows)
        self.warn.setText("\n".join(warn))

    # -- coordinates of the selected shape
    def _fill_points(self):
        self.points.blockSignals(True)
        sel = sorted(self.canvas.selected)
        self.points.setRowCount(0)
        if len(sel) == 1:
            g = self.canvas.shapes[sel[0]]
            if g.get("kind") == "circle":
                self.points.setRowCount(2)
                self.points.setVerticalHeaderLabels(["centre", "radius"])
                c = g["center"]
                for j, v in enumerate(c):
                    self.points.setItem(0, j, QtWidgets.QTableWidgetItem(fmt(v * self.k, 6)))
                self.points.setItem(1, 0, QtWidgets.QTableWidgetItem(fmt(g["radius"] * self.k, 6)))
                it = QtWidgets.QTableWidgetItem("")
                it.setFlags(QtCore.Qt.NoItemFlags)
                self.points.setItem(1, 1, it)
            else:
                pts = g["points"]
                self.points.setRowCount(len(pts))
                self.points.setVerticalHeaderLabels([str(i + 1) for i in range(len(pts))])
                for i, (x, y) in enumerate(pts):
                    self.points.setItem(i, 0, QtWidgets.QTableWidgetItem(fmt(x * self.k, 6)))
                    self.points.setItem(i, 1, QtWidgets.QTableWidgetItem(fmt(y * self.k, 6)))
        self.points.blockSignals(False)
        poly = len(sel) == 1 and self.canvas.shapes[sel[0]].get("kind") != "circle"
        self.btn_add_pt.setEnabled(poly)
        self.btn_del_pt.setEnabled(poly)
        self.shape_mat.setEnabled(bool(sel))
        if sel:
            mats = {self.canvas.shapes[i].get("material") or "" for i in sel}
            idx = self.shape_mat.findData(next(iter(mats))) if len(mats) == 1 else -1
            self.shape_mat.setCurrentIndex(max(idx, 0))

    def _set_shape_material(self, _i=None):
        sel = sorted(self.canvas.selected)
        if not sel:
            return
        name = self.shape_mat.currentData() or ""
        if name == self.material.currentText():
            name = ""
        self.canvas.push_undo()
        for i in sel:
            g = self.canvas.shapes[i]
            if name:
                g["material"] = name
            else:
                g.pop("material", None)
        self.canvas._edited()

    def _point_edited(self, item):
        sel = sorted(self.canvas.selected)
        if len(sel) != 1:
            return
        try:
            v = float(item.text()) / self.k
        except ValueError:
            self._fill_points()
            return
        g = self.canvas.shapes[sel[0]]
        self.canvas.push_undo()
        r, c = item.row(), item.column()
        if g.get("kind") == "circle":
            if r == 0:
                g["center"][c] = v
            elif v > 0:
                g["radius"] = v
        else:
            g["points"][r][c] = v
        self.canvas._edited()

    def _add_point(self):
        sel = sorted(self.canvas.selected)
        if len(sel) != 1:
            return
        g = self.canvas.shapes[sel[0]]
        rows = sorted({i.row() for i in self.points.selectedIndexes()}) or [len(g["points"]) - 1]
        i = rows[-1]
        P = g["points"]
        a, b = np.asarray(P[i]), np.asarray(P[(i + 1) % len(P)])
        self.canvas.push_undo()
        P.insert(i + 1, (0.5 * (a + b)).tolist())
        self.canvas._edited()
        self.points.selectRow(i + 1)

    def _remove_point(self):
        sel = sorted(self.canvas.selected)
        if len(sel) != 1:
            return
        g = self.canvas.shapes[sel[0]]
        rows = sorted({i.row() for i in self.points.selectedIndexes()}, reverse=True)
        if not rows or len(g["points"]) - len(rows) < 3:
            return
        self.canvas.push_undo()
        for r in rows:
            g["points"].pop(r)
        self.canvas._edited()

    def _modifiers(self):
        from ..model.properties import FrameModifiers
        m = edit_modifiers(self, self.sec.modifiers or FrameModifiers())
        if m is not None:
            self.sec.modifiers = m
            self._schedule()

    def _ok(self):
        name = self.name.text().strip()
        if not name:
            self.error.setText("Give the section a name")
            return
        if name in self.existing:
            self.error.setText(f"A section named {name!r} already exists")
            return
        if self.canvas.draft:
            self.canvas.finish_polygon()
        shapes = self.canvas.shapes
        if not shapes:
            self.error.setText("Draw the section first")
            return
        sec = copy.deepcopy(self.sec)
        sec.name, sec.material, sec.shape = name, self.material.currentText(), "Section Designer"
        sec.geometry = copy.deepcopy(shapes)
        if not sec.update_properties(True, self.model.materials):
            self.error.setText("The drawing has no material area")
            return
        if sec.A <= 0:
            self.error.setText("The drawing has no material area")
            return
        self.result_section = sec
        self.accept()
