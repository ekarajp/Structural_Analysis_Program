"""Grid lines drawn and edited in the plan window.

* Draw > Draw Grid Line: two points (snaps, typed lengths).  A line along the axes of a
  Cartesian grid system becomes an X / Y line of it (its ordinate from the points), any
  other line a General line; named on from the last line (A, B ... / 1, 2 ... / L1 ...).
* Draw > Offset Grid Line: click a grid line on the side of the copy (B -> B').
* A right click on a grid line: Edit / Rename / Move / Offset / Hide / Delete.
* Dragging a grid bubble moves the line (Edit > Grid Lines > Lock Grid Lines Against
  Dragging switches it off).  With Move Objects with Grid Lines on, the joints on a moved
  line move with it - also when its ordinate is changed in the grid systems form.

Every change is one undo step.  A grid line is named by (grid system, name): several
towers each have their own A, B, ... lines."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..model import Model
from ..model.base import ModelError
from ..model.grids import CYLINDRICAL, PLAN_DIRECTIONS, system_of, to_global, to_local
from ..model.objects import GridLine, GridSystem
from .document import Document
from .tools import TOOL_PROPERTIES, TOOLS, Preview, SnapResult, Tool, next_step

XY = Tuple[float, float]
PARALLEL_DEG = 0.05  # a drawn line this close to an axis is a line of that axis


# ------------------------------------------------------------------ helpers
def _sys(g: GridLine) -> str:
    return g.system or "G1"


def find_line(model: Model, system: str, name: str) -> GridLine:
    for g in model.grid_lines:
        if _sys(g) == system and g.name == name and g.direction in PLAN_DIRECTIONS:
            return g
    raise ModelError("NO_GRID_LINE", f"Grid line {name} ({system}) not found")


def _tol(model: Model) -> float:
    return max(model.options.merge_tolerance, 1e-9)


def _names_in(model: Model, system: str) -> set:
    return {g.name for g in model.grid_lines if _sys(g) == system}


def _letters(k: int) -> str:
    s = ""
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(65 + r) + s
    return s


def next_name(model: Model, system: str, direction: str) -> str:
    """The name after the last line of that kind in the system: X lines A, B, ... (after
    the highest letter name), Y lines 1, 2, ..., General lines L1, L2, ..."""
    used = _names_in(model, system)
    if direction == "X":
        k = 1
        for g in model.grid_lines:
            if _sys(g) == system and g.direction == "X" and g.name.isalpha() and g.name.isupper():
                n = 0
                for ch in g.name:
                    n = n * 26 + (ord(ch) - 64)
                k = max(k, n + 1)
        while _letters(k) in used:
            k += 1
        return _letters(k)
    if direction == "Y":
        k = 1 + max([int(g.name) for g in model.grid_lines
                     if _sys(g) == system and g.direction == "Y" and g.name.isdigit()] or [0])
        while str(k) in used:
            k += 1
        return str(k)
    k = 1
    while f"L{k}" in used:
        k += 1
    return f"L{k}"


def primed_name(model: Model, system: str, base: str) -> str:
    """B -> B', B'' ... (the first free one in the system)."""
    used = _names_in(model, system)
    name = base + "'"
    while name in used:
        name += "'"
    return name


def drawing_system(model: Model, tower: Optional[str] = None) -> str:
    """The Cartesian grid system new lines go to: the first of the tower's (of the
    model's) systems; a new "G1" when there is none."""
    names = model.grid_systems_of(tower) if model.multi_tower and tower else list(model.grid_systems)
    for n in names:
        if model.grid_systems[n].type != CYLINDRICAL:
            return n
    if names:
        raise ModelError("NO_CARTESIAN_GRID", "Grid lines are drawn in a Cartesian grid system: add one in "
                                              "Edit Stories and Grid Systems")
    return "G1" if tower is None else f"{model._tower(tower)}-G1"


def _sort(model: Model):
    model.grid_lines.sort(key=lambda g: ("XYZGTR".find(g.direction[0]), _sys(g),
                                         g.ordinate if g.direction != "General" else 0.0))


def _exists(model: Model, system: str, direction: str, ordinate: float, skip: Optional[GridLine] = None):
    tol = _tol(model)
    for g in model.grid_lines:
        if g is not skip and _sys(g) == system and g.direction == direction and abs(g.ordinate - ordinate) <= tol:
            return g
    return None


def _general_exists(model: Model, system: str, a: XY, b: XY) -> Optional[GridLine]:
    tol = _tol(model)
    for g in model.grid_lines:
        if _sys(g) != system or g.direction != "General" or not g.start or not g.end:
            continue
        for p, q in ((g.start, g.end), (g.end, g.start)):
            if math.dist(p, a) <= tol and math.dist(q, b) <= tol:
                return g
    return None


# ------------------------------------------------------------------ geometry
def line_ends(model: Model, g: GridLine) -> Optional[Tuple[XY, XY]]:
    """Global plan end points of a straight grid line as it is drawn."""
    from ..model.grids import grid_geometry
    for a, b, name, system in grid_geometry(model).lines:
        if name == g.name and system == _sys(g):
            return a, b
    return None


def _point_segment(p: XY, a: XY, b: XY) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def grid_models_at(model: Model, story: Optional[str], tower: Optional[str] = None) -> List[Model]:
    """The models whose grids a plan of ``story`` shows (several towers: the towers with a
    story at its level - as the window draws them)."""
    if not model.multi_tower:
        return [model]
    if story is not None and story in model.stories:
        return [model.view(model.tower_of_story(st)) for st in model.same_level_stories(story)]
    if tower is not None:
        return [model.view(tower)]
    return [model.view(t) for t in model.tower_names()]


def line_at(model: Model, xy: XY, tol: float, story: Optional[str] = None,
            tower: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """(system, name) of the straight grid line nearest to the plan point (within tol)."""
    from ..model.grids import grid_geometry
    best = None
    for gm in grid_models_at(model, story, tower):
        for a, b, name, system in grid_geometry(gm).lines:
            d = _point_segment(xy, a, b)
            if d <= tol and (best is None or d < best[0]):
                best = (d, system, name)
    return None if best is None else (best[1], best[2])


def bubble_at(model: Model, xy: XY, size, story: Optional[str] = None,
              tower: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """(system, name) of the straight grid line whose bubble holds the plan point:
    ``size(grid model)`` = (radius, distance of the centre past the line's end), as the
    window draws the bubbles of that tower's grids."""
    from ..model.grids import grid_geometry
    for gm in grid_models_at(model, story, tower):
        geo = grid_geometry(gm)
        radius, offset = size(gm)
        for end, d, name in geo.bubbles:
            c = (end[0] + d[0] * offset, end[1] + d[1] * offset)
            if math.dist(c, xy) > radius * 1.15:
                continue
            for a, b, n, system in geo.lines:  # the line this bubble ends
                if n == name and (a == end or b == end):
                    return system, name
    return None


# ------------------------------------------------------------------ commands
def draw_grid_line(doc: Document, a: XY, b: XY, tower: Optional[str] = None,
                   system: Optional[str] = None) -> str:
    """A grid line through two plan points (global x, y); returns its name."""
    m = doc.model
    tol = _tol(m)
    if math.dist(a, b) <= tol:
        raise ModelError("ZERO_LENGTH", "Click two different points")
    system = system or drawing_system(m, tower)
    gs = m.grid_systems.get(system) or GridSystem(name=system, tower=m._tower(tower) if m.multi_tower and tower else None)
    ua, va = to_local(gs, *a)
    ub, vb = to_local(gs, *b)
    ang = math.degrees(math.atan2(vb - va, ub - ua)) % 180.0
    if abs(vb - va) <= tol or min(ang, 180.0 - ang) <= PARALLEL_DEG:
        direction, ordinate = "Y", 0.5 * (va + vb)  # along local x: a line of constant y
    elif abs(ub - ua) <= tol or abs(ang - 90.0) <= PARALLEL_DEG:
        direction, ordinate = "X", 0.5 * (ua + ub)
    else:
        direction, ordinate = "General", 0.0
    if direction == "General":
        if _general_exists(m, system, (ua, va), (ub, vb)):
            raise ModelError("GRID_EXISTS", "That grid line is there already")
    else:
        g0 = _exists(m, system, direction, ordinate)
        if g0 is not None:
            raise ModelError("GRID_EXISTS", f"Grid line {g0.name} is there already")
    name = next_name(m, system, direction)

    def mutate(model: Model):
        if system not in model.grid_systems:
            model.grid_systems[system] = gs
        if direction == "General":
            gl = GridLine(system=system, name=name, direction="General", start=(ua, va), end=(ub, vb))
        else:
            gl = GridLine(system=system, name=name, direction=direction, ordinate=ordinate)
        model.grid_lines.append(gl)
        _sort(model)

    doc.edit(mutate)
    return name


def offset_grid_line(doc: Document, system: str, name: str, distance: float, side: XY) -> str:
    """A copy of the line ``distance`` away on the side of the plan point ``side``
    (named B', B'' ...); returns its name.  Circles: a larger / smaller radius."""
    m = doc.model
    if distance <= 0:
        raise ModelError("NO_OFFSET", "The offset distance must be greater than zero")
    g = find_line(m, system, name)
    gs = system_of(m, g)
    new = primed_name(m, system, name)
    if g.direction in ("X", "Y"):
        u, v = to_local(gs, *side)
        here = u if g.direction == "X" else v
        o = g.ordinate + (distance if here >= g.ordinate else -distance)
        if _exists(m, system, g.direction, o) is not None:
            raise ModelError("GRID_EXISTS", f"A grid line is there already ({o:g})")
        made = dict(direction=g.direction, ordinate=o)
    elif g.direction == "R":
        o = gs.origin or (0.0, 0.0)
        r = g.ordinate + (distance if math.dist(side, o) >= g.ordinate else -distance)
        if r <= 0:
            raise ModelError("NO_OFFSET", "The circle would have no radius - offset outwards or by less")
        made = dict(direction="R", ordinate=r)
    elif g.direction == "General" and g.start and g.end:
        (x0, y0), (x1, y1) = g.start, g.end
        L = math.hypot(x1 - x0, y1 - y0)
        n = (-(y1 - y0) / L, (x1 - x0) / L)
        u, v = to_local(gs, *side)
        s = 1.0 if (u - x0) * n[0] + (v - y0) * n[1] >= 0 else -1.0
        dx, dy = s * distance * n[0], s * distance * n[1]
        made = dict(direction="General", start=(x0 + dx, y0 + dy), end=(x1 + dx, y1 + dy))
    else:
        raise ModelError("NO_OFFSET", "Radial grid lines have no parallel copy - add one in the grid form")

    def mutate(model: Model):
        model.grid_lines.append(GridLine(system=system, name=new, **made))
        _sort(model)

    doc.edit(mutate)
    return new


def rename_grid_line(doc: Document, system: str, name: str, new: str) -> None:
    new = (new or "").strip()
    if not new or new == name:
        return
    if new in _names_in(doc.model, system):
        raise ModelError("NAME_IN_USE", f"Grid system {system} has a line {new} already")
    find_line(doc.model, system, name)

    def mutate(model: Model):
        find_line(model, system, name).name = new

    doc.edit(mutate)


def delete_grid_line(doc: Document, system: str, name: str) -> None:
    g0 = find_line(doc.model, system, name)
    key = (g0.direction, g0.name)

    def mutate(model: Model):
        model.grid_lines = [g for g in model.grid_lines
                            if not (_sys(g) == system and (g.direction, g.name) == key)]

    doc.edit(mutate)


def set_grid_line_visible(doc: Document, system: str, name: str, visible: bool) -> None:
    find_line(doc.model, system, name)

    def mutate(model: Model):
        find_line(model, system, name).visible = bool(visible)

    doc.edit(mutate)


def show_all_grid_lines(doc: Document) -> int:
    hidden = [g for g in doc.model.grid_lines if not getattr(g, "visible", True)]
    if hidden:
        def mutate(model: Model):
            for g in model.grid_lines:
                g.visible = True
        doc.edit(mutate)
    return len(hidden)


# --------------------------------------------------- moving lines (and joints)
def _points_of_system(model: Model, system: str) -> List[str]:
    """The joints a line of the system can carry: with several towers those of its tower."""
    if not model.multi_tower:
        return list(model.points)
    t = model.tower_of_system(system)
    out = []
    for n in model.points:
        try:
            if model.tower_of_point(n) == t:
                out.append(n)
        except Exception:  # noqa: BLE001 - a free joint with no tower
            pass
    return out


def glue_moves(model: Model, moves: Sequence[Tuple[str, str, float, float]]) -> Dict[str, XY]:
    """Global plan displacement of the joints on moved X / Y lines: ``moves`` = (system,
    direction, old ordinate, new ordinate).  A joint on a moved X line and a moved Y line
    moves both ways; the joints are found at their places before the move."""
    tol = _tol(model)
    out: Dict[str, List[float]] = {}
    by_sys: Dict[str, list] = {}
    for system, d, o0, o1 in moves:
        if d in ("X", "Y") and abs(o1 - o0) > 0:
            by_sys.setdefault(system, []).append((d, o0, o1))
    for system, mv in by_sys.items():
        gs = model.grid_systems.get(system) or GridSystem(name=system)
        if gs.type == CYLINDRICAL:
            continue
        c, s = math.cos(math.radians(gs.rotation or 0.0)), math.sin(math.radians(gs.rotation or 0.0))
        for n in _points_of_system(model, system):
            p = model.points[n]
            u, v = to_local(gs, p.x, p.y)
            du = next((o1 - o0 for d, o0, o1 in mv if d == "X" and abs(u - o0) <= tol), 0.0)
            dv = next((o1 - o0 for d, o0, o1 in mv if d == "Y" and abs(v - o0) <= tol), 0.0)
            if du or dv:
                acc = out.setdefault(n, [0.0, 0.0])
                acc[0] += c * du - s * dv
                acc[1] += s * du + c * dv
    return {n: (d[0], d[1]) for n, d in out.items()}


def _general_glue(model: Model, system: str, g: GridLine, dx: float, dy: float) -> Dict[str, XY]:
    """Joints on a General line (between its ends) moved by the local (dx, dy)."""
    gs = system_of(model, g)
    tol = _tol(model)
    a, b = to_global(gs, *g.start), to_global(gs, *g.end)
    c, s = math.cos(math.radians(gs.rotation or 0.0)), math.sin(math.radians(gs.rotation or 0.0))
    d = (c * dx - s * dy, s * dx + c * dy)
    return {n: d for n in _points_of_system(model, system)
            if _point_segment((model.points[n].x, model.points[n].y), a, b) <= tol}


def _apply_moves(model: Model, moves: Dict[str, XY]):
    for n, (dx, dy) in moves.items():
        p = model.points[n]
        p.x += dx
        p.y += dy


def move_grid_line(doc: Document, system: str, name: str, *, to: Optional[float] = None,
                   delta: Optional[float] = None, move_objects: bool = False) -> int:
    """Move a straight grid line: an X / Y line to the ordinate ``to`` (or by ``delta``),
    a General line ``delta`` across itself.  ``move_objects``: the joints on the line move
    with it.  Returns the number of joints moved."""
    m = doc.model
    g = find_line(m, system, name)
    if g.direction in ("X", "Y"):
        o1 = to if to is not None else g.ordinate + (delta or 0.0)
        if abs(o1 - g.ordinate) <= 0:
            return 0
        other = _exists(m, system, g.direction, o1, skip=g)
        if other is not None:
            raise ModelError("GRID_EXISTS", f"Grid line {other.name} is there already")
        moves = glue_moves(m, [(system, g.direction, g.ordinate, o1)]) if move_objects else {}

        def mutate(model: Model):
            find_line(model, system, name).ordinate = o1
            _sort(model)
            _apply_moves(model, moves)
    elif g.direction == "General" and g.start and g.end:
        (x0, y0), (x1, y1) = g.start, g.end
        L = math.hypot(x1 - x0, y1 - y0)
        dist = delta or 0.0
        if not dist:
            return 0
        dx, dy = -(y1 - y0) / L * dist, (x1 - x0) / L * dist
        moves = _general_glue(m, system, g, dx, dy) if move_objects else {}

        def mutate(model: Model):
            gl = find_line(model, system, name)
            gl.start, gl.end = (x0 + dx, y0 + dy), (x1 + dx, y1 + dy)
            _apply_moves(model, moves)
    else:
        raise ModelError("NOT_MOVABLE", "Circles and radial lines are changed in the grid systems form")
    doc.edit(mutate)
    return len(moves)


def line_ordinate(model: Model, system: str, name: str, xy: XY) -> Optional[float]:
    """The ordinate an X / Y line of the system through the plan point has; for a General
    line the signed distance of the point from it."""
    g = find_line(model, system, name)
    gs = system_of(model, g)
    u, v = to_local(gs, *xy)
    if g.direction == "X":
        return u
    if g.direction == "Y":
        return v
    if g.direction == "General" and g.start and g.end:
        (x0, y0), (x1, y1) = g.start, g.end
        L = math.hypot(x1 - x0, y1 - y0)
        return ((u - x0) * -(y1 - y0) + (v - y0) * (x1 - x0)) / L
    return None


def moved_line(model: Model, system: str, name: str, xy: XY) -> Optional[Tuple[XY, XY]]:
    """Plan end points of the line as it would be moved through ``xy`` (drag preview)."""
    g = find_line(model, system, name)
    ends = line_ends(model, g)
    val = line_ordinate(model, system, name, xy)
    if ends is None or val is None:
        return None
    gs = system_of(model, g)
    (ax, ay), (bx, by) = ends
    if g.direction == "General":
        d = val
    else:
        d = val - g.ordinate
    c, s = math.cos(math.radians(gs.rotation or 0.0)), math.sin(math.radians(gs.rotation or 0.0))
    if g.direction == "X":
        vec = (c * d, s * d)
    elif g.direction == "Y":
        vec = (-s * d, c * d)
    else:
        (x0, y0), (x1, y1) = g.start, g.end
        L = math.hypot(x1 - x0, y1 - y0)
        lx, ly = -(y1 - y0) / L * d, (x1 - x0) / L * d
        vec = (c * lx - s * ly, s * lx + c * ly)
    return (ax + vec[0], ay + vec[1]), (bx + vec[0], by + vec[1])


# ------------------------------------------------------------------ tools


class GridLineTool(Tool):
    """Draw > Draw Grid Line: two points in a plan window."""

    name = "grid_line"
    title = "Draw Grid Line"
    prompt = "click the first point of the grid line"
    next_prompt = "click the second point (or type a length + Enter)"
    views = ("plan",)

    def activate(self, state) -> None:
        self.state = state

    def hint(self, state=None) -> str:
        if self.points:
            return next_step(self.title, self.next_prompt, "along an axis = X / Y line, else a General line",
                             "Esc = cancel")
        return next_step(self.title, self.prompt, "snaps work as for beams", "Esc = done")

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self._p3(pt)
        if not self.points:
            self.points = [p]
            self._raw_last = p
            return False
        a = self.points[0]
        self.points = []
        self._raw_last = None
        name = draw_grid_line(self.doc, (a[0], a[1]), (p[0], p[1]), tower=self.tower)
        st = getattr(self, "state", None)
        if st is not None:
            st.status.emit(f"Grid line {name} added")
        return True

    def double_click(self, pt) -> bool:
        return False

    def preview(self, cursor) -> Preview:
        pv = self._line_preview(cursor)
        if self.points and cursor is not None:
            a, b = self.points[0], self.resolve(cursor)
            try:
                gs = self.model.grid_systems.get(drawing_system(self.model, self.tower))
            except ModelError:
                gs = None
            gs = gs or GridSystem()
            ua, va = to_local(gs, a[0], a[1])
            ub, vb = to_local(gs, b[0], b[1])
            tol = _tol(self.model)
            ang = math.degrees(math.atan2(vb - va, ub - ua)) % 180.0
            if abs(vb - va) <= tol or min(ang, 180.0 - ang) <= PARALLEL_DEG:
                pv.text += f"   Y line at {0.5 * (va + vb):g}"
            elif abs(ub - ua) <= tol or abs(ang - 90.0) <= PARALLEL_DEG:
                pv.text += f"   X line at {0.5 * (ua + ub):g}"
            else:
                pv.text += "   General line"
        return pv


class OffsetGridTool(Tool):
    """Draw > Offset Grid Line, as CAD OFFSET: click a grid line, then type the distance +
    Enter (the copy goes to the side of the cursor), or click the point the copy passes
    through.  Enter alone repeats the last distance; Esc drops the line."""

    name = "grid_offset"
    title = "Offset Grid Line"
    prompt = "click the grid line to offset"
    views = ("plan",)
    picks_grids = True  # the viewport passes the grid line under the cursor
    takes_typed_points = False

    def activate(self, state) -> None:
        self.state = state
        self.picked: Optional[Tuple[str, str]] = None

    @property
    def wants_snap(self) -> bool:  # (a through point snaps like any point)
        return getattr(self, "picked", None) is not None

    def hint(self, state=None) -> str:
        if getattr(self, "picked", None) is not None:
            return next_step(f"{self.title} - line {self.picked[1]}",
                             "type the distance + Enter (the copy goes to the cursor's side)",
                             "or click the point the copy passes through",
                             f"Enter = {self.settings.grid_offset_distance:g} again", "Esc = another line")
        return next_step(self.title, self.prompt, "Esc = done")

    def _status(self, text: str):
        st = getattr(self, "state", None)
        if st is not None:
            st.status.emit(text)

    def cancel(self):
        super().cancel()
        self.picked = None

    def _offset(self, distance: float, side: XY) -> str:
        system, name = self.picked
        new = offset_grid_line(self.doc, system, name, distance, side)
        self.settings.grid_offset_distance = distance
        self.picked = None
        self._status(f"Offset grid line {name} by {distance:g} -> {new}")
        return new

    def typed(self, text: str, toward=None):
        v = float(text)
        if v <= 0:
            raise ValueError("The offset distance must be positive")
        if self.picked is None:  # before a line is picked: the distance for the next one
            self.settings.grid_offset_distance = v
            self._status(f"Offset distance {v:g} - click a grid line")
            return None
        if toward is None:
            raise ValueError("Move the cursor to the side of the new line, then type the distance")
        self._offset(v, (toward[0], toward[1]))
        return None

    def key(self, k: str) -> bool:
        if getattr(self, "picked", None) is None:
            return False
        if k == "Escape":
            self.picked = None
            self._status("Offset Grid Line: click the grid line to offset")
            return True
        if k in ("Return", "KP_Enter"):
            cur = getattr(self, "_cursor", None)
            if cur is not None:
                try:
                    self._offset(self.settings.grid_offset_distance, cur)
                except Exception as exc:  # noqa: BLE001
                    st = getattr(self, "state", None)
                    if st is not None:
                        st.report(exc)
            return True
        return False

    def grid_click(self, hit: Optional[Tuple[str, str]], xy: Optional[XY]) -> bool:
        if self.picked is None:
            if hit is None:
                return False
            self.picked = hit
            self._status(f"Grid line {hit[1]}: type the distance + Enter, or click a point the copy passes through")
            return False
        if xy is None:
            return False
        val = line_ordinate(self.model, self.picked[0], self.picked[1], xy)
        if val is None:
            raise ModelError("NO_OFFSET", "Radial grid lines have no parallel copy")
        g = find_line(self.model, *self.picked)
        if g.direction == "R":
            o = system_of(self.model, g).origin or (0.0, 0.0)
            d = abs(math.dist(xy, o) - g.ordinate)
        else:
            d = abs(val if g.direction == "General" else val - g.ordinate)
        if d <= _tol(self.model):
            raise ModelError("NO_OFFSET", "Click away from the line (the copy passes through the point)")
        self._offset(d, xy)
        return True

    def grid_preview(self, hit: Optional[Tuple[str, str]], xy: Optional[XY]) -> Preview:
        self._cursor = xy
        z = self.elevation()
        pv = Preview(marker=None if xy is None else (xy[0], xy[1], z))
        line = getattr(self, "picked", None)
        if line is None:
            if hit is not None:
                pv.text = f"click: offset grid line {hit[1]}"
            return pv
        if xy is None:
            return pv
        m = self.model
        try:
            g = find_line(m, *line)
        except ModelError:
            self.picked = None
            return pv
        if g.direction in ("X", "Y", "General"):
            ln = moved_line(m, line[0], line[1], xy)  # the copy through the cursor
            val = line_ordinate(m, line[0], line[1], xy)
            d = abs(val if g.direction == "General" else val - g.ordinate)
            if ln is not None:
                pv.lines = [((ln[0][0], ln[0][1], z), (ln[1][0], ln[1][1], z))]
        elif g.direction == "R":
            o = system_of(m, g).origin or (0.0, 0.0)
            d = abs(math.dist(xy, o) - g.ordinate)
        else:
            pv.text = "radial lines: no parallel copy - Esc"
            return pv
        pv.text = (f"{primed_name(m, *line)}: {d:.3f} from {line[1]}   "
                   f"(type a distance + Enter, or click;  Enter = {self.settings.grid_offset_distance:g})")
        return pv


TOOLS.update({"grid_line": GridLineTool, "grid_offset": OffsetGridTool})
TOOL_PROPERTIES.update({"grid_offset": [("grid_offset_distance", "Offset distance", "float",
                                          "or type the distance + Enter in the window")]})


# ------------------------------------------------------- right-click menu
def grid_line_menu(win, vp, hit: Tuple[str, str], x: int, y: int) -> None:
    """Right click on a grid line in a plan window: what can be done with it."""
    from PySide6 import QtCore, QtWidgets
    system, name = hit
    doc = win.doc
    m = doc.model
    try:
        g = find_line(m, system, name)
    except ModelError:
        return
    menu = QtWidgets.QMenu(win)
    head = menu.addAction(f"Grid line {name}" + (f"  ({system})" if len(m.grid_systems) > 1 else ""))
    head.setEnabled(False)
    menu.addSeparator()

    def run(fn):
        def go():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(win, f"Grid line {name}", str(exc))
            vp.render()
        return go

    def edit():
        from .dialogs import GridStoriesDialog
        tower = m.tower_of_system(system) if m.multi_tower else None
        GridStoriesDialog(doc, win, system=system, tower=tower, line=name).exec()

    def rename():
        new, ok = QtWidgets.QInputDialog.getText(win, "Rename grid line", "Name", text=name)
        if ok:
            rename_grid_line(doc, system, name, new)

    def move():
        glue = bool(getattr(doc, "glue_grids", False))
        note = "\n(the joints on the line move with it)" if glue else ""
        if g.direction in ("X", "Y"):
            v, ok = QtWidgets.QInputDialog.getDouble(win, f"Move grid line {name}",
                                                     f"{g.direction} ordinate{note}", g.ordinate, -1e9, 1e9, 4)
            if ok:
                n = move_grid_line(doc, system, name, to=v, move_objects=glue)
                win.state.status.emit(f"Grid line {name} moved to {v:g}" + (f" with {n} joint(s)" if n else ""))
        else:
            v, ok = QtWidgets.QInputDialog.getDouble(win, f"Move grid line {name}",
                                                     f"Distance across the line (+ = to its left){note}",
                                                     0.0, -1e9, 1e9, 4)
            if ok:
                n = move_grid_line(doc, system, name, delta=v, move_objects=glue)
                win.state.status.emit(f"Grid line {name} moved" + (f" with {n} joint(s)" if n else ""))

    straight = g.direction in ("X", "Y", "General")
    menu.addAction("Edit in Grid Systems...", run(edit))
    menu.addAction("Rename...", run(rename))
    a = menu.addAction("Move...", run(move))
    a.setEnabled(straight)
    def offset():
        win.state.set_tool("grid_offset")
        win.state.tool.picked = hit  # this line: type the distance + Enter, or click a point
        win.state.update_hint()
    a = menu.addAction("Offset (type the distance + Enter)", run(offset))
    a.setEnabled(g.direction != "T")
    menu.addSeparator()
    menu.addAction("Hide", run(lambda: set_grid_line_visible(doc, system, name, False)))
    menu.addAction("Delete", run(lambda: delete_grid_line(doc, system, name)))
    if any(not getattr(gl, "visible", True) for gl in m.grid_lines):
        menu.addSeparator()
        menu.addAction("Show All Grid Lines", run(lambda: show_all_grid_lines(doc)))
    h = vp.vtk_widget.height()
    menu.exec(vp.vtk_widget.mapToGlobal(QtCore.QPoint(int(x), int(h - y))))
