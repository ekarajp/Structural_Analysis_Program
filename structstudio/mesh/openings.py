"""Openings cut holes in the shells they are drawn over (ETABS behaviour).

An opening area (type Opening / Null, or no section) lying in the plane of a
floor or wall and overlapping it removes that part of the shell from the
analysis mesh and from the drawing.  Shared by the mesher and the scene so what
is drawn is what is analysed.

* Shell and openings rectilinear in the shell's plane (the usual case: doors,
  windows, shafts, stair openings): the shell is split along the lines through
  all corners and the rectangles inside an opening are dropped - an exact hole
  and a conforming mesh.  An opening may run past the shell edge (a notch).
* Anything else: the shell is meshed as usual and mesh cells whose centre falls
  in an opening are dropped (the hole follows the mesh; a warning says so).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import polygon2d as pg

Hole = Tuple[str, np.ndarray]  # opening name, 2-D polygon in the shell's frame


class OpeningIndex:
    """The model's openings, ready for repeated "which openings cut this shell"
    queries.  ``scale`` converts model lengths (e.g. to SI in the mesher)."""

    def __init__(self, model, scale: float = 1.0):
        self.items: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []  # name, P, bbox min, max
        for a in model.areas.values():
            if not a.is_opening or len(a.points) < 3 or any(p not in model.points for p in a.points):
                continue
            P = np.array([model.points[p].xyz for p in a.points], float) * scale
            self.items.append((a.name, P, P.min(axis=0), P.max(axis=0)))

    def __bool__(self):
        return bool(self.items)

    def holes(self, P: np.ndarray, fr, tol: float, exclude: Sequence[str] = ()) -> List[Hole]:
        """Openings in the plane of polygon ``P`` (frame ``fr``) that overlap it
        with a positive area (touching along an edge does not count)."""
        if not self.items:
            return []
        lo, hi = P.min(axis=0) - tol, P.max(axis=0) + tol
        Q = pg.to_2d(P, fr)
        qlo, qhi = Q.min(axis=0), Q.max(axis=0)
        out: List[Hole] = []
        for name, O, olo, ohi in self.items:
            if name in exclude or np.any(olo > hi) or np.any(ohi < lo):
                continue
            if pg.off_plane(O, fr) > tol:
                continue
            H = pg.to_2d(O, fr)
            hlo, hhi = H.min(axis=0), H.max(axis=0)
            if np.any(np.minimum(qhi, hhi) - np.maximum(qlo, hlo) <= tol):
                continue  # bounding boxes only touch
            if pg.signed_area(H) < 0:
                H = H[::-1]
            out.append((name, H))
        return out


def split(Q: np.ndarray, holes: Sequence[Hole], tol: float):
    """How to cut ``holes`` out of the 2-D shell polygon ``Q``: ``("grid", U, V,
    cells)`` for the exact rectangle split, ``("cells", None, None, None)`` when
    the mesh cells must be filtered, None without holes."""
    if not holes:
        return None
    if pg.axis_aligned(Q, tol) and all(pg.axis_aligned(H, tol) for _, H in holes):
        U, V, cells = pg.grid_cells(Q, [H for _, H in holes], tol)
        return "grid", U, V, cells
    return "cells", None, None, None


def in_hole(uv, holes: Sequence[Hole]) -> bool:
    return any(pg.inside(uv, H) for _, H in holes)


def fill_polygons(P: np.ndarray, holes: Sequence[Hole], tol: float) -> List[np.ndarray]:
    """3-D polygons to draw for a shell: the shell itself, its rectangles with
    the openings left out, or triangles of a non-convex outline."""
    try:
        fr = pg.plane_frame(P)
    except ValueError:
        return [P]
    Q = pg.to_2d(P, fr)
    how = split(Q, holes, tol)
    if how is not None and how[0] == "grid":
        _, U, V, cells = how
        return [pg.to_3d(pg.cell_corners(U, V, k, l), fr) for k, l in cells]
    if how is None and pg.is_convex(Q):
        return [P]
    if how is None and pg.axis_aligned(Q, tol):
        U, V, cells = pg.grid_cells(Q, (), tol)
        return [pg.to_3d(pg.cell_corners(U, V, k, l), fr) for k, l in cells]
    tris = pg.triangulate(Q)
    keep = []
    for t in tris:
        if how is not None and in_hole(Q[list(t)].mean(axis=0), holes):
            continue
        keep.append(P[list(t)])
    return keep or [P]
