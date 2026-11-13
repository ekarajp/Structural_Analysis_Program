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
    if story is not None and story in model.stories:  # (the plan's own tower first)
        sts = sorted(model.same_level_stories(story), key=lambda st: st != story)
        return [model.view(model.tower_of_story(st)) for st in sts]
    if tower is not None:
        return [model.view(tower)]
    return [model.view(t) for t in model.tower_names()]


def line_at(model: Model, xy: XY, tol: float, story: Optional[str] = None,
            tower: Optional[str] = None, own_only: bool = False) -> Optional[Tuple[str, str]]:
    """(system, name) of the straight grid line nearest to the plan point (within tol);
    ``own_only``: the lines of the plan's own tower only."""
    from ..model.grids import grid_geometry
    best = None
    gms = grid_models_at(model, story, tower)
    for gm in (gms[:1] if own_only else gms):
        for a, b, name, system in grid_geometry(gm).lines:
            d = _point_segment(xy, a, b)
            if d <= tol and (best is None or d < best[0]):
                best = (d, system, name)
    return None if best is None else (best[1], best[2])


def bubble_at(model: Model, xy: XY, size, story: Optional[str] = None,
              tower: Optional[str] = None, own_only: bool = False) -> Optional[Tuple[str, str]]:
    """(system, name) of the straight grid line whose bubble holds the plan point:
    ``size(grid model)`` = (radius, distance of the centre past the line's end), as the
    window draws the bubbles of that tower's grids."""
    from ..model.grids import grid_geometry
    gms = grid_models_at(model, story, tower)
    for gm in (gms[:1] if own_only else gms):
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


def _local_delta(gs, dx: float, dy: float) -> XY:
    """A global plan displacement in the system's local axes."""
    c, s = math.cos(math.radians(gs.rotation or 0.0)), math.sin(math.radians(gs.rotation or 0.0))
    return c * dx + s * dy, -s * dx + c * dy


def move_grid_lines(doc: Document, keys: Sequence[Tuple[str, str]], dx: float, dy: float,
                    copy: bool = False, move_objects: bool = False) -> List[Tuple[str, str]]:
    """Move (or copy) grid lines by the global plan vector (dx, dy) - an X / Y line by the
    part of it across the line, a General line by all of it.  ``move_objects`` (move
    only): the joints on the lines move with them.  Returns the (system, name) of the lines
    moved / made.  One undo step."""
    m = doc.model
    tol = _tol(m)
    lines = [find_line(m, s_, n_) for s_, n_ in dict.fromkeys(keys)]
    if not lines:
        raise ModelError("NOTHING_SELECTED", "Select the grid lines to move first")
    plan = []  # (line, change)
    xy_moves, gen_moves = [], {}
    names_taken: Dict[str, set] = {}
    for g in lines:
        system = _sys(g)
        gs = system_of(m, g)
        du, dv = _local_delta(gs, dx, dy)
        if g.direction == "X":
            ch = ("ordinate", g.ordinate + du)
        elif g.direction == "Y":
            ch = ("ordinate", g.ordinate + dv)
        elif g.direction == "General" and g.start and g.end:
            ch = ("ends", ((g.start[0] + du, g.start[1] + dv), (g.end[0] + du, g.end[1] + dv)))
        else:
            raise ModelError("NOT_MOVABLE", f"Grid line {g.name}: circles and radial lines are changed in the "
                                            f"grid systems form")
        if ch[0] == "ordinate":
            if abs(ch[1] - g.ordinate) <= tol and not copy:
                continue
            clash = _exists(m, system, g.direction, ch[1], skip=g)
            if clash is not None and (copy or clash not in lines):
                raise ModelError("GRID_EXISTS", f"Grid line {clash.name} is there already ({ch[1]:g})")
            if move_objects and not copy:
                xy_moves.append((system, g.direction, g.ordinate, ch[1]))
        elif move_objects and not copy:
            gen_moves.update(_general_glue(m, system, g, du, dv))
        new = None
        if copy:
            taken = names_taken.setdefault(system, set())
            new = g.name + "'"
            while new in _names_in(m, system) or new in taken:
                new += "'"
            taken.add(new)
        plan.append((system, g.name, g.direction, ch, new))
    moves = glue_moves(m, xy_moves) if xy_moves else {}
    for n_, d_ in gen_moves.items():
        if n_ not in moves:
            moves[n_] = d_
    out = []

    def mutate(model: Model):
        for system, name, direction, ch, new in plan:
            g = find_line(model, system, name)
            if new is not None:
                import dataclasses as _dc
                g = _dc.replace(g, name=new)
                model.grid_lines.append(g)
            if ch[0] == "ordinate":
                g.ordinate = ch[1]
            else:
                g.start, g.end = ch[1]
            out.append((system, g.name))
        _sort(model)
        _apply_moves(model, moves)

    if plan:
        doc.edit(mutate)
    return out


def edit_grid_line(doc: Document, system: str, name: str, *, new_name: Optional[str] = None,
                   ordinate: Optional[float] = None, start: Optional[XY] = None, end: Optional[XY] = None,
                   visible: Optional[bool] = None, bubble: Optional[str] = None,
                   move_objects: bool = False) -> None:
    """Grid line properties (name, ordinate / ends, shown, bubble end) - one undo step."""
    m = doc.model
    g = find_line(m, system, name)
    new_name = (new_name or "").strip() or name
    if new_name != name and new_name in _names_in(m, system):
        raise ModelError("NAME_IN_USE", f"Grid system {system} has a line {new_name} already")
    moves = {}
    if ordinate is not None and g.direction in ("X", "Y", "R", "T") and ordinate != g.ordinate:
        if g.direction == "R" and ordinate <= 0:
            raise ModelError("INVALID_RADIUS", "A circle needs a radius greater than zero")
        clash = _exists(m, system, g.direction, ordinate, skip=g)
        if clash is not None:
            raise ModelError("GRID_EXISTS", f"Grid line {clash.name} is there already")
        if move_objects and g.direction in ("X", "Y"):
            moves = glue_moves(m, [(system, g.direction, g.ordinate, ordinate)])
    if g.direction == "General" and (start is not None or end is not None):
        a, b = start or g.start, end or g.end
        if math.dist(a, b) <= _tol(m):
            raise ModelError("ZERO_LENGTH", "The two ends of the line must differ")

    def mutate(model: Model):
        gl = find_line(model, system, name)
        if ordinate is not None and gl.direction in ("X", "Y", "R", "T"):
            gl.ordinate = float(ordinate)
        if gl.direction == "General":
            if start is not None:
                gl.start = tuple(start)
            if end is not None:
                gl.end = tuple(end)
        if visible is not None:
            gl.visible = bool(visible)
        if bubble in ("Start", "End"):
            gl.bubble_location = bubble
        gl.name = new_name
        _sort(model)
        _apply_moves(model, moves)

    doc.edit(mutate)


def lines_in_box(model: Model, to_screen, box, crossing: bool, story: Optional[str] = None,
                 tower: Optional[str] = None) -> List[Tuple[str, str]]:
    """Grid lines a selection box takes (screen coordinates through ``to_screen(x, y)``):
    a window (left to right) takes the lines it holds whole, a crossing box (right to left)
    the lines it touches."""
    from ..model.grids import grid_geometry
    (x0, y0), (x1, y1) = box
    lo = (min(x0, x1), min(y0, y1))
    hi = (max(x0, x1), max(y0, y1))

    def inside(p):
        return lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]

    def crosses(a, b):
        if inside(a) or inside(b):
            return True
        # the segment against the four sides
        def seg(p, q, r, s_):
            d = (q[0] - p[0]) * (s_[1] - r[1]) - (q[1] - p[1]) * (s_[0] - r[0])
            if abs(d) < 1e-12:
                return False
            t = ((r[0] - p[0]) * (s_[1] - r[1]) - (r[1] - p[1]) * (s_[0] - r[0])) / d
            u = ((r[0] - p[0]) * (q[1] - p[1]) - (r[1] - p[1]) * (q[0] - p[0])) / d
            return 0 <= t <= 1 and 0 <= u <= 1
        c = [lo, (hi[0], lo[1]), hi, (lo[0], hi[1])]
        return any(seg(a, b, c[k], c[(k + 1) % 4]) for k in range(4))

    out = []
    gms = grid_models_at(model, story, tower)[:1]  # (the window's own tower)
    for gm in gms:
        for a, b, name, system in grid_geometry(gm).lines:
            sa, sb = to_screen(*a), to_screen(*b)
            if (inside(sa) and inside(sb)) if not crossing else crosses(sa, sb):
                out.append((system, name))
    return out


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


class GridSelectTool(Tool):
    """Grid Mode: select grid lines (click, Ctrl + click, window / crossing box), drag a
    bubble or a selected line to move it, Delete, double-click = properties."""

    name = "grid_select"
    title = "Select Grid Lines"
    prompt = "click a grid line (Ctrl adds); drag a box; drag a bubble to move its line"
    views = ("plan",)
    grid_selecting = True  # the viewport picks grid lines instead of objects
    takes_typed_points = False

    def hint(self, state=None) -> str:
        n = len(getattr(state, "grid_sel", ()) or ()) if state is not None else 0
        if n:
            return next_step("Grid Mode", f"{n} grid line(s) selected", "Delete = delete", "M, V = move",
                             "C, O = copy", "drag = move", "double-click = properties", "Esc = clear")
        return next_step("Grid Mode", self.prompt, "Esc = leave Grid Mode")


class GridMoveTool(Tool):
    """Grid Mode Move / Copy (CAD): the selected grid lines (or click lines first), then a
    base point and the destination - clicked or typed (a length toward the cursor, dx,dy)."""

    name = "grid_move"
    title = "Move Grid Lines"
    copy = False
    views = ("plan",)

    def activate(self, state) -> None:
        self.state = state
        self._adding = not state.grid_sel  # nothing selected: click the lines first

    def cancel(self):
        if getattr(self, "_adding", False) and getattr(self, "state", None) is not None and self.state.grid_sel:
            self._adding = False  # right click / Esc after picking: the lines are chosen
            return
        super().cancel()

    @property
    def prompt(self) -> str:
        st = getattr(self, "state", None)
        if st is not None and not st.grid_sel:
            return "click the grid lines to move, then Enter (or right-click)"
        return "click the base point" if not self.points else "click the destination (or type a distance + Enter)"

    def hint(self, state=None) -> str:
        return next_step(self.title, self.prompt, "Esc = cancel")

    def _picking(self) -> bool:
        st = getattr(self, "state", None)
        return st is not None and (not st.grid_sel or getattr(self, "_adding", False))

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        st = self.state
        p = self._p3(pt)
        if not st.grid_sel or getattr(self, "_adding", False):
            self._adding = True
            hit = line_at(self.model, (p[0], p[1]), self.tolerance, story=self.story, tower=self.tower,
                          own_only=True)
            if hit is not None:
                st.toggle_grid(hit, add=True)
            return False
        if not self.points:
            self.points = [p]
            self._raw_last = p
            return False
        a = self.points[0]
        self.points = []
        self._raw_last = None
        glue = bool(getattr(self.doc, "glue_grids", False)) and not self.copy
        made = move_grid_lines(self.doc, sorted(st.grid_sel), p[0] - a[0], p[1] - a[1], copy=self.copy,
                               move_objects=glue)
        if made:  # (the copies / the moved lines stay selected)
            st.set_grid_selection(made)
        st.status.emit(("Copied" if self.copy else "Moved") + f" {len(made)} grid line(s)" if made else
                       "Nothing moved: an X / Y line moves only across itself")
        return True

    def finish(self) -> bool:
        if getattr(self, "_adding", False):  # Enter: the lines are chosen - now the base point
            self._adding = False
            return False
        return False

    def double_click(self, pt) -> bool:
        return False

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        st = getattr(self, "state", None)
        if st is None or cursor is None or not self.points or self._picking():
            return pv
        a, b = self.points[0], self._p3(cursor)
        dx, dy = b[0] - a[0], b[1] - a[1]
        m = self.model
        for key in st.grid_sel:
            try:
                g = find_line(m, *key)
            except ModelError:
                continue
            ends = line_ends(m, g)
            if ends is None:
                continue
            gs = system_of(m, g)
            du, dv = _local_delta(gs, dx, dy)
            if g.direction == "X":
                du, dv = du, 0.0
            elif g.direction == "Y":
                du, dv = 0.0, dv
            c, s_ = math.cos(math.radians(gs.rotation or 0.0)), math.sin(math.radians(gs.rotation or 0.0))
            vx, vy = c * du - s_ * dv, s_ * du + c * dv
            pv.lines.append(((ends[0][0] + vx, ends[0][1] + vy, a[2]), (ends[1][0] + vx, ends[1][1] + vy, a[2])))
        pv.guides = [(a, b)]
        pv.text = f"{'copy' if self.copy else 'move'} {math.hypot(dx, dy):.3f}"
        return pv


class GridCopyTool(GridMoveTool):
    name = "grid_copy"
    title = "Copy Grid Lines"
    copy = True


GRID_MODE_TOOLS = ("grid_select", "grid_line", "grid_offset", "grid_move", "grid_copy")
TOOLS.update({"grid_line": GridLineTool, "grid_offset": OffsetGridTool, "grid_select": GridSelectTool,
              "grid_move": GridMoveTool, "grid_copy": GridCopyTool})
TOOL_PROPERTIES.update({"grid_offset": [("grid_offset_distance", "Offset distance", "float",
                                          "or type the distance + Enter in the window")]})


# ------------------------------------------------------- menus / forms
def grid_line_menu(win, vp, hit: Tuple[str, str], x: int, y: int) -> None:
    """Right click on a grid line outside Grid Mode: go to Grid Mode with it selected."""
    from PySide6 import QtCore, QtWidgets
    menu = QtWidgets.QMenu(win)

    def go():
        win.state.set_grid_mode(True)
        win.state.set_grid_selection([hit])
    head = menu.addAction(f"Grid line {hit[1]}")
    head.setEnabled(False)
    menu.addSeparator()
    menu.addAction("Edit Grids on Screen", go)
    h = vp.vtk_widget.height()
    menu.exec(vp.vtk_widget.mapToGlobal(QtCore.QPoint(int(x), int(h - y))))


def grid_mode_menu(win, vp, hit: Optional[Tuple[str, str]], x: int, y: int) -> None:
    """Right click in Grid Mode: on a grid line its commands, elsewhere the mode's own."""
    from PySide6 import QtCore, QtWidgets
    st = win.state
    doc = win.doc
    menu = QtWidgets.QMenu(win)

    def run(fn):
        def go():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(win, "Grid lines", str(exc))
            vp.render()
        return go

    if hit is not None:
        n = len(st.grid_sel)
        head = menu.addAction(f"Grid line {hit[1]}" if n <= 1 else f"{n} grid lines")
        head.setEnabled(False)
        menu.addSeparator()
        menu.addAction("Properties...", run(lambda: grid_line_properties(win, hit)))
        a = menu.addAction("Offset (type the distance + Enter)", run(lambda: _start(st, "grid_offset", hit)))
        a.setEnabled(n <= 1)
        menu.addAction("Move", run(lambda: st.set_tool("grid_move")))
        menu.addAction("Copy", run(lambda: st.set_tool("grid_copy")))
        menu.addSeparator()
        menu.addAction("Hide", run(lambda: _hide(doc, st)))
        menu.addAction("Delete", run(st.delete_grid_selection))
        menu.addSeparator()
    if any(not getattr(g, "visible", True) for g in doc.model.grid_lines):
        menu.addAction("Show All Hidden Grid Lines", run(lambda: show_all_grid_lines(doc)))
    menu.addAction("Edit Stories and Grid Systems (table)...", run(lambda: win.edit_grid_stories()))
    menu.addSeparator()
    menu.addAction("Done (leave Grid Mode)", run(lambda: st.set_grid_mode(False)))
    h = vp.vtk_widget.height()
    menu.exec(vp.vtk_widget.mapToGlobal(QtCore.QPoint(int(x), int(h - y))))


def _start(st, tool: str, hit):
    st.set_tool(tool)
    st.tool.picked = hit
    st.update_hint()


def _hide(doc: Document, st) -> None:
    keys = set(st.grid_sel)
    if not keys:
        return

    def mutate(model: Model):
        for g in model.grid_lines:
            if (_sys(g), g.name) in keys:
                g.visible = False
    doc.edit(mutate)
    st.set_grid_selection(())


def grid_line_properties(win, hit: Tuple[str, str]) -> bool:
    """Grid line properties: name, ordinate (X / Y / circle / radial) or the two ends
    (General), shown, bubble end.  The same data as the grid systems form."""
    from PySide6 import QtWidgets
    doc = win.doc
    m = doc.model
    system, name = hit
    g = find_line(m, system, name)
    d = QtWidgets.QDialog(win)
    d.setWindowTitle(f"Grid Line {name}")
    form = QtWidgets.QFormLayout(d)
    u = m.units.label("length") if hasattr(m.units, "label") else ""
    ed_name = QtWidgets.QLineEdit(name)
    form.addRow("Name", ed_name)
    form.addRow("Grid system", QtWidgets.QLabel(f"{system}  ({system_of(m, g).type})"))
    kind = {"X": "X line (constant X)", "Y": "Y line (constant Y)", "R": "Circle", "T": "Radial line",
            "General": "General (two points)"}.get(g.direction, g.direction)
    form.addRow("Type", QtWidgets.QLabel(kind))
    fields = {}

    def num(v):
        w = QtWidgets.QLineEdit(f"{v:g}")
        return w
    if g.direction == "General":
        for k, v in (("x1", g.start[0]), ("y1", g.start[1]), ("x2", g.end[0]), ("y2", g.end[1])):
            fields[k] = num(v)
            form.addRow(f"{k.upper()} [{u}] (system axes)", fields[k])
    else:
        lab = {"X": f"X ordinate [{u}]", "Y": f"Y ordinate [{u}]", "R": f"Radius [{u}]", "T": "Angle [deg]"}
        fields["o"] = num(g.ordinate)
        form.addRow(lab.get(g.direction, "Ordinate"), fields["o"])
    vis = QtWidgets.QCheckBox("Shown")
    vis.setChecked(getattr(g, "visible", True))
    form.addRow("", vis)
    bub = QtWidgets.QComboBox()
    bub.addItems(["End", "Start"])
    bub.setCurrentText(getattr(g, "bubble_location", "End") or "End")
    form.addRow("Bubble at", bub)
    glue = bool(getattr(doc, "glue_grids", False))
    if g.direction in ("X", "Y"):
        form.addRow("", QtWidgets.QLabel("Moving the line moves the joints on it" if glue else
                                         "Moving the line leaves the objects where they are"))
    bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    form.addRow(bb)

    def ok():
        try:
            kw = {}
            if "o" in fields:
                kw["ordinate"] = float(fields["o"].text())
            else:
                kw["start"] = (float(fields["x1"].text()), float(fields["y1"].text()))
                kw["end"] = (float(fields["x2"].text()), float(fields["y2"].text()))
            new = ed_name.text().strip() or name
            edit_grid_line(doc, system, name, new_name=new, visible=vis.isChecked(),
                           bubble=bub.currentText(), move_objects=glue, **kw)
            st = getattr(win, "state", None)
            if st is not None and getattr(st, "grid_mode", False):
                st.set_grid_selection([(system, new)])
            d.accept()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(d, f"Grid line {name}", str(exc))
    bb.accepted.connect(ok)
    bb.rejected.connect(d.reject)
    return d.exec() == QtWidgets.QDialog.Accepted
