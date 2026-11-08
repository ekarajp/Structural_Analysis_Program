"""Extruded view: frame members drawn with their real cross-section and areas
with their thickness (ETABS / SAP2000 "Extrude" display option).

Profiles are in the section's local 2-3 plane (CSI: depth ``t3`` along local 2,
width ``t2`` along local 3), decomposed into convex pieces so every piece
extrudes into a simple prism.  Members follow the local axes of the analysis
(:func:`~structstudio.mesh.geometry.frame_axes`, including the member angle)
and are placed by their cardinal point and joint offsets.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from ..mesh.geometry import frame_axes, polygon_normal
from ..model import Model

CIRCLE_SIDES = 20


def _rect(y2c: float, y3c: float, h2: float, w3: float) -> np.ndarray:
    a, b = h2 / 2.0, w3 / 2.0
    return np.array([[y2c - a, y3c - b], [y2c + a, y3c - b], [y2c + a, y3c + b], [y2c - a, y3c + b]])


def _circle(r: float, n: int = CIRCLE_SIDES) -> np.ndarray:
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _ring(ro: float, ri: float, n: int = CIRCLE_SIDES) -> List[np.ndarray]:
    t = np.linspace(0.0, 2 * math.pi, n + 1)
    out = []
    for k in range(n):
        a, b = t[k], t[k + 1]
        out.append(np.array([[ro * math.cos(a), ro * math.sin(a)], [ro * math.cos(b), ro * math.sin(b)],
                             [ri * math.cos(b), ri * math.sin(b)], [ri * math.cos(a), ri * math.sin(a)]]))
    return out


def section_profile(sec) -> Optional[List[np.ndarray]]:
    """Convex pieces (each an (n, 2) array of (y2, y3)) of a frame section, or
    ``None`` when the section has no usable dimensions."""
    if sec is None:
        return None
    pieces = _profile(sec)
    if pieces is None:
        return None
    out = []
    for pc in pieces:  # counter-clockwise in (y2, y3): the prism faces point outward
        x, y = pc[:, 0], pc[:, 1]
        if 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
            pc = pc[::-1]
        out.append(pc)
    return out


_LEGACY = {"Rectangular", "Circular", "Box", "Pipe"}  # drawn with their own few pieces


def _profile(sec) -> Optional[List[np.ndarray]]:
    import re

    from ..model.section_shapes import convex_pieces, shape_of
    sh = shape_of(sec)
    unequal_i = sh is not None and sh.key == "I" and ((sec.t2b and sec.t2b != sec.t2) or (sec.tfb and sec.tfb != sec.tf))
    if getattr(sec, "geometry", None) or (sh is not None and (sh.key not in _LEGACY and (sh.key != "I" or unequal_i))):
        pieces = convex_pieces(sec)  # Section Designer, T / L beams, channels, angles ... about the centroid
        if pieces:
            return pieces
    shape = (sec.shape or "").lower()
    words = set(re.findall(r"[a-z]+", shape))
    d = float(sec.t3 or 0.0)
    b = float(sec.t2 or 0.0)
    tf = float(sec.tf or 0.0)
    tw = float(sec.tw or 0.0)
    if "pipe" in words and d > 0:
        t = tw if 0 < tw < d / 2 else d / 10
        return _ring(d / 2, d / 2 - t)
    if words & {"circle", "circular", "round", "rod", "circ"} and d > 0:
        return [_circle(d / 2)]
    if d > 0 and b > 0 and tf > 0 and tw > 0:
        if words & {"box", "tube", "hss", "hollow"}:
            tf_, tw_ = min(tf, d / 2), min(tw, b / 2)
            return [_rect(d / 2 - tf_ / 2, 0, tf_, b), _rect(-d / 2 + tf_ / 2, 0, tf_, b),
                    _rect(0, -b / 2 + tw_ / 2, d - 2 * tf_, tw_), _rect(0, b / 2 - tw_ / 2, d - 2 * tf_, tw_)]
        if words & {"tee", "t"}:
            return [_rect(d / 2 - tf / 2, 0, tf, b), _rect(-tf / 2, 0, d - tf, tw)]
        if words & {"angle", "l"}:
            return [_rect(0, -b / 2 + tw / 2, d, tw), _rect(-d / 2 + tf / 2, tw / 2, tf, b - tw)]
        if words & {"channel", "c"}:
            return [_rect(0, -b / 2 + tw / 2, d, tw), _rect(d / 2 - tf / 2, tw / 2, tf, b - tw),
                    _rect(-d / 2 + tf / 2, tw / 2, tf, b - tw)]
        if not words & {"rectangular", "rectangle", "rect"}:  # I / wide flange (also General sections with flange dimensions)
            bb = float(getattr(sec, "t2b", None) or b)
            tfb = float(getattr(sec, "tfb", None) or tf)
            hw = max(d - tf - tfb, 1e-9)
            return [_rect(d / 2 - tf / 2, 0, tf, b), _rect(-d / 2 + tfb / 2, 0, tfb, bb),
                    _rect((tfb - tf) / 2, 0, hw, tw)]
    if d > 0 and b > 0:
        return [_rect(0, 0, d, b)]
    # General section without dimensions: the rectangle with the same A and I33
    A, I33 = float(sec.A or 0.0), float(sec.I33 or 0.0)
    if A > 0 and I33 > 0:
        dd = math.sqrt(12.0 * I33 / A)
        bb = A / dd
        if 0.02 < dd / bb < 50:
            return [_rect(0, 0, dd, bb)]
    if d > 0:
        return [_rect(0, 0, d, d)]
    return None


def cardinal_offset(pieces: Sequence[np.ndarray], cardinal: int) -> np.ndarray:
    """(y2, y3) of the cardinal point (CSI 1..11) of a profile; the member line
    passes through it.  1-3 bottom, 4-6 middle, 7-9 top; left = -3 side."""
    if cardinal in (10, 11, 0) or not pieces:
        return np.zeros(2)
    allp = np.vstack(pieces)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    row, col = divmod(cardinal - 1, 3)
    y2 = (lo[0], 0.5 * (lo[0] + hi[0]), hi[0])[row]
    y3 = (lo[1], 0.5 * (lo[1] + hi[1]), hi[1])[col]
    return np.array([y2, y3])


class _MeshBuilder:
    """Collects prism vertices / polygons into one vtkPolyData."""

    def __init__(self):
        self.pts: List[np.ndarray] = []
        self.cells: List[List[int]] = []
        self.n = 0

    def prism(self, bottom: np.ndarray, top: np.ndarray):
        k = len(bottom)
        base = self.n
        self.pts.append(bottom)
        self.pts.append(top)
        self.n += 2 * k
        self.cells.append(list(range(base, base + k))[::-1])
        self.cells.append(list(range(base + k, base + 2 * k)))
        for i in range(k):
            j = (i + 1) % k
            self.cells.append([base + i, base + j, base + k + j, base + k + i])

    def polydata(self, triangulate: bool = False) -> Optional[vtk.vtkPolyData]:
        if not self.pts:
            return None
        coords = np.ascontiguousarray(np.vstack(self.pts), dtype=float)
        pts = vtk.vtkPoints()
        pts.SetData(numpy_support.numpy_to_vtk(coords, deep=True))
        polys = vtk.vtkCellArray()
        for c in self.cells:
            polys.InsertNextCell(len(c))
            for i in c:
                polys.InsertCellPoint(i)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetPolys(polys)
        if triangulate:
            tf = vtk.vtkTriangleFilter()
            tf.SetInputData(poly)
            tf.Update()
            poly = tf.GetOutput()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.SetFeatureAngle(30.0)
        normals.SplittingOn()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOff()
        normals.Update()
        return normals.GetOutput()


def frame_solids(model: Model, names: Iterable[str]) -> Tuple[Optional[vtk.vtkPolyData], List[str]]:
    """Solid polydata of the frames with a usable section, and the names of the
    frames that have none (drawn as tubes by the caller)."""
    mb = _MeshBuilder()
    missing: List[str] = []
    for n in names:
        f = model.frames[n]
        pieces = section_profile(model.frame_sections.get(f.section))
        a = np.array(model.points[f.point_i].xyz, float) + np.array(f.joint_offset_i, float)
        b = np.array(model.points[f.point_j].xyz, float) + np.array(f.joint_offset_j, float)
        if pieces is None or np.linalg.norm(b - a) <= 1e-12:
            missing.append(n)
            continue
        try:
            ax = frame_axes(a, b, f.angle)
        except ValueError:
            missing.append(n)
            continue
        cp = f.display_cardinal
        if cp is None:
            cp = 8 if (f.type == "Beam" and model.grid_mode == "story") else f.cardinal_point
        off = cardinal_offset(pieces, cp)
        e2, e3 = ax[1], ax[2]
        for pc in pieces:
            q = pc - off
            ring = q[:, 0:1] * e2 + q[:, 1:2] * e3
            mb.prism(a + ring, b + ring)
    return mb.polydata(), missing


def area_solids(model: Model, areas: Iterable) -> Optional[vtk.vtkPolyData]:
    """Areas with their thickness: floors hang below their plane (top of slab at the
    level, as in ETABS), walls are centred on theirs."""
    mb = _MeshBuilder()
    for a in areas:
        sec = model.shell_sections.get(a.section or "")
        t = float(sec.thickness) if sec is not None and sec.thickness else 0.0
        P = np.array([model.points[p].xyz for p in a.points], float)
        if t <= 0 or len(P) < 3:
            continue
        nrm = polygon_normal(P)
        if a.type in ("Floor", "Ramp"):
            if nrm[2] < 0:
                nrm = -nrm
            lo, hi = -t * nrm, np.zeros(3)
        else:
            lo, hi = -0.5 * t * nrm, 0.5 * t * nrm
        mb.prism(P + lo, P + hi)
    return mb.polydata(triangulate=True)
