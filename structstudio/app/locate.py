"""Where is an object? Coordinates, story and grid reference for the Model Explorer
("Story2, B/1 to C/1") and the locate / zoom-to commands."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..model import Model

KINDS = ("points", "frames", "areas", "panel_loads")


def object_xyz(model: Model, attr: str, name: str) -> Optional[np.ndarray]:
    """Corner coordinates of a joint / frame / shell / load panel (None if gone)."""
    if attr == "points":
        p = model.points.get(name)
        return None if p is None else np.array([p.xyz], float)
    if attr == "frames":
        f = model.frames.get(name)
        if f is None or f.point_i not in model.points or f.point_j not in model.points:
            return None
        return np.array([model.points[f.point_i].xyz, model.points[f.point_j].xyz], float)
    if attr == "areas":
        a = model.areas.get(name)
        if a is None or any(p not in model.points for p in a.points):
            return None
        return np.array([model.points[p].xyz for p in a.points], float)
    if attr == "panel_loads":
        pl = model.panel_loads.get(name)
        if pl is None:
            return None
        if getattr(pl, "area", None) and pl.area in model.areas:
            return object_xyz(model, "areas", pl.area)
        pts = getattr(pl, "points", None) or getattr(pl, "polygon", None)
        return None if not pts else np.array([tuple(q)[:3] for q in pts], float)
    return None


def object_story(model: Model, attr: str, name: str) -> Optional[str]:
    obj = getattr(model, attr, {}).get(name)
    s = getattr(obj, "story", None)
    if s in model.stories:
        return s
    X = object_xyz(model, attr, name)
    if X is None:
        return None
    st = model.story_at(float(X[:, 2].max()), model.options.merge_tolerance)
    return st.name if st else None


def _axis_label(model: Model, direction: str, value: float, tol: float) -> str:
    from ..model.grids import global_ordinate
    lines = [(g, global_ordinate(model, g)) for g in model.grid_lines if g.direction == direction]
    lines = [(g, v) for g, v in lines if v is not None]
    if not lines:
        return f"{value:g}"
    g, v = min(lines, key=lambda t: abs(t[1] - value))
    d = value - v
    if abs(d) <= tol:
        return g.name
    return f"{g.name}{d:+.2f}"


def grid_ref(model: Model, xyz, tol: Optional[float] = None) -> str:
    """'B/1' on a grid intersection, 'B+1.20/1' off it (X grid / Y grid)."""
    tol = tol if tol is not None else max(model.options.merge_tolerance, 1e-3) * 10
    return f"{_axis_label(model, 'X', xyz[0], tol)}/{_axis_label(model, 'Y', xyz[1], tol)}"


def _grid_span(model: Model, X: np.ndarray) -> str:
    tol = max(model.options.merge_tolerance, 1e-3) * 10

    def span(direction, vals):
        a, b = _axis_label(model, direction, vals.min(), tol), _axis_label(model, direction, vals.max(), tol)
        return a if a == b else f"{a}-{b}"
    return f"{span('X', X[:, 0])} / {span('Y', X[:, 1])}"


def location_text(model: Model, attr: str, name: str) -> str:
    """One line for the status bar / tooltip: what, which story, where on the grid, size."""
    obj = getattr(model, attr, {}).get(name)
    X = object_xyz(model, attr, name)
    if obj is None or X is None:
        return f"{name}: not in the model"
    u = model.units.shown("length")
    story = object_story(model, attr, name)
    at = f"{story}" if story else f"z = {X[:, 2].max():g} {u}"
    if attr == "points":
        return f"Joint {name}  ·  {at}  ·  grid {grid_ref(model, X[0])}  ·  z = {X[0, 2]:g} {u}"
    if attr == "frames":
        L = float(np.linalg.norm(X[1] - X[0]))
        a, b = grid_ref(model, X[0]), grid_ref(model, X[1])
        where = f"at {a}" if a == b else f"{a} to {b}"
        return f"{obj.type} {name}  ·  {obj.section}  ·  {at}  ·  {where}  ·  L = {L:.3f} {u}"
    if attr == "areas":
        P = X - X.mean(axis=0)
        n = np.zeros(3)
        for i in range(len(P)):
            n += np.cross(P[i], P[(i + 1) % len(P)])
        area = 0.5 * float(np.linalg.norm(n))
        sec = obj.section or "no section"
        return (f"{obj.type} {name}  ·  {sec}  ·  {at}  ·  grid {_grid_span(model, X)}  ·  "
                f"area = {area:.2f} {u}²")
    return f"Load panel {name}  ·  {at}  ·  grid {_grid_span(model, X)}"
