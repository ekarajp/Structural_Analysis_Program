"""Results drawn on the members themselves, as lines or extruded (real sections
and shell thickness), as in ETABS with the "Extrude" option on:

* :func:`frame_color_actors` - each member coloured along its length by a
  result (P V2 V3 T M2 M3, or the elastic stress ratio with the CSI ratio
  colours), as thick lines or as its real cross-section.
* :func:`thick_shell` - shell elements (contour / deformed) with their thickness.
* :func:`swept_frame` - a deformed member's real section swept along its bent
  axis (used by the deformed shape in the extruded view).

The stress ratio is elastic extreme-fibre stress |P|/A + |M2|/S22 + |M3|/S33
over fy (steel) or fc' (concrete): an indication of how hard a member works,
not a code design check.  When frame design is added, its D/C ratios use the
same colours (:data:`RATIO_BANDS`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from . import vtk_lite as vtk
from vtkmodules.util import numpy_support

from ..mesh.geometry import frame_axes, polygon_normal
from ..model import Model
from . import extrude as ex

FRAME_QUANTITIES = ("P", "V2", "V3", "T", "M2", "M3", "Stress ratio")
_INDEX = {"P": 0, "V2": 1, "V3": 2, "T": 3, "M2": 4, "M3": 5}
# CSI-style ratio colours: (upper limit, colour)
RATIO_BANDS = ((0.5, "#7aa6d8"), (0.7, "#2ec4d6"), (0.9, "#36b34a"), (1.0, "#f2b01e"), (9e9, "#e02424"))
RATIO_MAX = 1.25  # the colour scale ends here (everything above is red)


def _rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# --------------------------------------------------------------------- values
def section_moduli(model: Model, sec) -> Optional[Tuple[float, float, float]]:
    """(A, S22, S33) of a frame section (None without area / dimensions)."""
    if sec is None or not sec.A:
        return None
    pieces = ex.section_profile(sec)
    if pieces is not None:
        P = np.vstack(pieces)
        c2 = float(np.max(np.abs(P[:, 0]))) or None   # extreme fibre in the local-2 direction
        c3 = float(np.max(np.abs(P[:, 1]))) or None
    else:
        c2 = sec.t3 / 2 if sec.t3 else None
        c3 = sec.t2 / 2 if sec.t2 else None
    S33 = sec.I33 / c2 if (sec.I33 and c2) else None
    S22 = sec.I22 / c3 if (sec.I22 and c3) else None
    if S33 is None and S22 is None:
        return None
    return float(sec.A), float(S22 or np.inf), float(S33 or np.inf)


def _strength(model: Model, sec) -> Optional[float]:
    mat = model.materials.get(getattr(sec, "material", None) or "")
    if mat is None:
        return None
    if mat.type == "Steel" or mat.type not in ("Concrete",) and mat.fy:
        return float(mat.fy) if mat.fy else None
    return float(mat.fc) if mat.fc else None


def frame_values(model: Model, results, case: str, quantity: str, frames=None):
    """{frame: (stations, values)} - the largest magnitude at each station
    (both the maximum and minimum of an envelope / combination)."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for f in model.frames.values():
        if frames is not None and f.name not in frames:
            continue
        if f.type in ("Null", "Rigid"):
            continue
        try:
            st = results.frame_forces(f.name, case)
        except Exception:
            continue
        F = np.abs(st.forces)
        if st.forces_min is not None:
            F = np.maximum(F, np.abs(st.forces_min))
        if quantity == "Stress ratio":
            sec = model.frame_sections.get(f.section)
            prop, fs = section_moduli(model, sec), _strength(model, sec)
            if prop is None or not fs:
                continue
            A, S22, S33 = prop
            v = (F[:, 0] / A + F[:, 4] / S22 + F[:, 5] / S33) / fs
        else:
            v = F[:, _INDEX[quantity]]
        out[f.name] = (np.asarray(st.station, float), np.asarray(v, float))
    return out


def quantity_title(model: Model, quantity: str) -> str:
    if quantity == "Stress ratio":
        return "Stress ratio\nσ / fy (fc')"
    from ..display_units import DisplayUnits
    u = DisplayUnits.of(model).label("moment" if quantity in ("T", "M2", "M3") else "force")
    return f"|{quantity}|\n[{u}]"


# ------------------------------------------------------------- lookup tables
def ratio_lut() -> vtk.vtkLookupTable:
    n = 25
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetRange(0.0, RATIO_MAX)
    for i in range(n):
        v = (i + 0.5) * RATIO_MAX / n
        col = next(c for lim, c in RATIO_BANDS if v < lim)
        lut.SetTableValue(i, *_rgb(col), 1.0)
    lut.SetAboveRangeColor(*_rgb(RATIO_BANDS[-1][1]), 1.0)
    lut.UseAboveRangeColorOn()
    for v in (0.0,) + tuple(lim for lim, _c in RATIO_BANDS[:-1]):
        lut.SetAnnotation(v, f"{v:.2f}")
    return lut


def spectrum_lut(vmin: float, vmax: float, n: int = 12) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetHueRange(0.667, 0.0)  # blue (small) -> red (large), like CSI
    lut.SetRange(vmin, vmax if vmax > vmin else vmin + 1e-12)
    lut.Build()
    return lut


def scalar_bar(lut, title: str, labels: int = 7):
    from . import scene
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle(title)
    bar.SetNumberOfLabels(labels)
    bar.SetLabelFormat("%-#6.3g")
    bar.GetTitleTextProperty().SetColor(*scene.COLORS["text"])
    bar.GetLabelTextProperty().SetColor(*scene.COLORS["text"])
    bar.GetAnnotationTextProperty().SetColor(*scene.COLORS["text"])
    scene._place_bar(bar)
    return bar


# ------------------------------------------------------------------ geometry
class ScalarMesh:
    """Polygons + lines with one scalar per point (unshared points per piece:
    every polygon / line has its own consecutive points).  Pieces are added one
    at a time or as arrays of many (same order, same points)."""

    def __init__(self):
        self.pts: List[np.ndarray] = []
        self.vals: List[np.ndarray] = []
        self._polys: List = []  # (first point, point count) - ints or arrays, in order
        self._lines: List = []
        self.n = 0

    @property
    def polys(self) -> List[List[int]]:
        return [list(range(int(b), int(b) + int(k))) for b, k in self._iter(self._polys)]

    @property
    def lines(self) -> List[List[int]]:
        return [list(range(int(b), int(b) + int(k))) for b, k in self._iter(self._lines)]

    @staticmethod
    def _iter(segs):
        for b, k in segs:
            if np.ndim(b):
                yield from zip(b, k)
            else:
                yield b, k

    def _add(self, P, v) -> int:
        P = np.atleast_2d(np.asarray(P, float))
        base = self.n
        self.pts.append(P)
        self.vals.append(np.broadcast_to(np.asarray(v, float), (len(P),)).copy())
        self.n += len(P)
        return base

    def polygon(self, P, v):
        b = self._add(P, v)
        self._polys.append((b, len(self.pts[-1])))

    def polyline(self, P, v):
        b = self._add(P, v)
        self._lines.append((b, len(self.pts[-1])))

    def polygons(self, P: np.ndarray, V: np.ndarray):
        """Many polygons of k points: P (m, k, 3), V (m, k) - as m calls of polygon."""
        P = np.asarray(P, float)
        m, k = P.shape[0], P.shape[1]
        if not m:
            return
        base = self.n
        self.pts.append(P.reshape(-1, 3))
        self.vals.append(np.broadcast_to(np.asarray(V, float), (m, k)).reshape(-1).copy())
        self.n += m * k
        self._polys.append((base + k * np.arange(m, dtype=np.int64), np.full(m, k, dtype=np.int64)))

    def slab(self, P, v, lo: np.ndarray, hi: np.ndarray):
        """A polygon with thickness: faces at P+lo and P+hi and the sides."""
        P = np.asarray(P, float)
        v = np.broadcast_to(np.asarray(v, float), (len(P),))
        k = len(P)
        self.polygon((P + lo)[::-1], v[::-1])
        self.polygon(P + hi, v)
        for i in range(k):
            j = (i + 1) % k
            self.polygon([P[i] + lo, P[j] + lo, P[j] + hi, P[i] + hi], [v[i], v[j], v[j], v[i]])

    @staticmethod
    def slab_faces(P: np.ndarray, V: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        """The faces :meth:`slab` makes, for m quads at once: (m, 6, 4, 3) points and
        (m, 6, 4) values (bottom reversed, top, then the four sides)."""
        m = len(P)
        F = np.empty((m, 6, 4, 3))
        W = np.empty((m, 6, 4))
        lo, hi = lo[:, None, :], hi[:, None, :]
        F[:, 0] = (P + lo)[:, ::-1]
        W[:, 0] = V[:, ::-1]
        F[:, 1] = P + hi
        W[:, 1] = V
        for i in range(4):
            j = (i + 1) % 4
            F[:, 2 + i, 0] = P[:, i] + lo[:, 0]
            F[:, 2 + i, 1] = P[:, j] + lo[:, 0]
            F[:, 2 + i, 2] = P[:, j] + hi[:, 0]
            F[:, 2 + i, 3] = P[:, i] + hi[:, 0]
            W[:, 2 + i] = np.stack([V[:, i], V[:, j], V[:, j], V[:, i]], axis=1)
        return F, W

    def sweep(self, path: np.ndarray, vals: np.ndarray, rings: Sequence[np.ndarray]):
        """Section ``rings`` (each (m, 3) offsets from the axis) swept along
        ``path`` (n, 3), values along the path, with end caps."""
        path = np.asarray(path, float)
        vals = np.asarray(vals, float)
        for ring in rings:
            ring = np.asarray(ring, float)
            m = len(ring)
            if len(path) > 1:
                A = path[:-1, None, :] + ring[None, :, :]  # (n-1, m, 3): a = path[k] + ring
                B = path[1:, None, :] + ring[None, :, :]
                j = (np.arange(m) + 1) % m
                Q = np.stack([A, A[:, j], B[:, j], B], axis=2)  # (n-1, m, 4, 3): a[i], a[j], b[j], b[i]
                va, vb = vals[:-1], vals[1:]
                W = np.broadcast_to(np.stack([va, va, vb, vb], axis=1)[:, None, :], (len(va), m, 4))
                self.polygons(Q.reshape(-1, 4, 3), W.reshape(-1, 4))
            self.polygon((path[0] + ring)[::-1], vals[0])
            self.polygon(path[-1] + ring, vals[-1])

    def polydata(self) -> Optional[vtk.vtkPolyData]:
        if not self.pts:
            return None
        X = np.ascontiguousarray(np.vstack(self.pts), float)
        pts = vtk.vtkPoints()
        pts.SetData(numpy_support.numpy_to_vtk(X, deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        for segs, setter in ((self._polys, poly.SetPolys), (self._lines, poly.SetLines)):
            if segs:
                starts = np.concatenate([np.atleast_1d(np.asarray(b, np.int64)) for b, _ in segs])
                sizes = np.concatenate([np.atleast_1d(np.asarray(k, np.int64)) for _, k in segs])
                offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
                conn = np.repeat(starts, sizes) + (np.arange(offsets[-1], dtype=np.int64)
                                                   - np.repeat(offsets[:-1], sizes))
                ca = vtk.vtkCellArray()
                ca.SetData(numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
                           numpy_support.numpy_to_vtkIdTypeArray(conn, deep=True))
                setter(ca)
        s = numpy_support.numpy_to_vtk(np.ascontiguousarray(np.concatenate(self.vals), float), deep=True)
        s.SetName("value")
        poly.GetPointData().SetScalars(s)
        if self._polys:
            nf = vtk.vtkPolyDataNormals()
            nf.SetInputData(poly)
            nf.SetFeatureAngle(30.0)
            nf.SplittingOn()
            nf.ConsistencyOn()
            nf.AutoOrientNormalsOff()
            nf.Update()
            poly = nf.GetOutput()
        return poly


def mapped_actor(poly, lut, lit: bool = True, line_width: float = 1.0, opacity: float = 1.0) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(*lut.GetRange())
    mapper.SetInterpolateScalarsBeforeMapping(True)
    mapper.SetScalarModeToUsePointData()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    p = actor.GetProperty()
    p.SetOpacity(opacity)
    if lit:
        p.SetAmbient(0.42)
        p.SetDiffuse(0.68)
        p.SetSpecular(0.05)
    else:
        p.SetLighting(False)
    if line_width > 1:
        p.SetLineWidth(line_width)
        p.RenderLinesAsTubesOn()
    return actor


def frame_rings(model: Model, frame, axes: Optional[np.ndarray] = None) -> Optional[List[np.ndarray]]:
    """The member's section pieces as 3-D offsets (placed by the cardinal point)."""
    pieces = ex.section_profile(model.frame_sections.get(frame.section))
    if pieces is None:
        return None
    if axes is None:
        a = np.array(model.points[frame.point_i].xyz, float)
        b = np.array(model.points[frame.point_j].xyz, float)
        try:
            axes = frame_axes(a, b, frame.angle)
        except ValueError:
            return None
    cp = frame.display_cardinal
    if cp is None:
        cp = 8 if (frame.type == "Beam" and model.grid_mode == "story") else frame.cardinal_point
    off = ex.cardinal_offset(pieces, cp)
    e2, e3 = axes[1], axes[2]
    return [(pc - off)[:, 0:1] * e2 + (pc - off)[:, 1:2] * e3 for pc in pieces]


def tube_rings(radius: float, axes: np.ndarray, sides: int = 10) -> List[np.ndarray]:
    t = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return [radius * (np.cos(t)[:, None] * axes[1] + np.sin(t)[:, None] * axes[2])]


def shell_offsets(model: Model, obj_name: str, nrm: np.ndarray):
    """(lo, hi) offsets of a shell's faces from its plane, or None without thickness.
    Floors hang below their level (top of slab at the level, as ETABS draws it)."""
    a = model.areas.get(obj_name)
    sec = model.shell_sections.get(getattr(a, "section", None) or "") if a is not None else None
    t = float(sec.thickness) if sec is not None and sec.thickness else 0.0
    if t <= 0:
        return None
    nrm = np.asarray(nrm, float)
    if a.type in ("Floor", "Ramp"):
        if nrm[2] < 0:
            nrm = -nrm
        return -t * nrm, np.zeros(3)
    return -0.5 * t * nrm, 0.5 * t * nrm


def thick_shell(mesh: ScalarMesh, model: Model, obj_name: str, P: np.ndarray, v) -> None:
    """One shell element as a slab of its thickness (or a flat polygon)."""
    try:
        nrm = polygon_normal(np.asarray(P, float))
    except Exception:
        nrm = np.array([0.0, 0.0, 1.0])
    off = shell_offsets(model, obj_name, nrm)
    if off is None:
        mesh.polygon(P, v)
    else:
        mesh.slab(P, v, *off)


def thick_shells(mesh: ScalarMesh, model: Model, objs: Sequence[str], P: np.ndarray, V: np.ndarray) -> None:
    """Many quad shell elements as slabs (or flat quads without a thickness): the
    same faces, values and order as :func:`thick_shell` element by element.
    ``P`` (m, 4, 3) corners, ``V`` (m, 4) values, ``objs`` their area objects."""
    P = np.asarray(P, float)
    V = np.asarray(V, float)
    m = len(P)
    if not m:
        return
    if P.shape[1] != 4:
        for i in range(m):
            thick_shell(mesh, model, objs[i], P[i], V[i])
        return
    # Newell normal per element (as polygon_normal), (0, 0, 1) for a degenerate one
    n = np.zeros((m, 3))
    for i in range(4):
        p, q = P[:, i], P[:, (i + 1) % 4]
        n[:, 0] += (p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])
        n[:, 1] += (p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])
        n[:, 2] += (p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])
    ln = np.sqrt(np.einsum("ij,ij->i", n, n))
    zero = ln == 0.0
    nrm = np.where(zero[:, None], np.array([0.0, 0.0, 1.0]), n / np.where(zero, 1.0, ln)[:, None])
    # thickness and kind per area object (as shell_offsets)
    info = {}
    for name in dict.fromkeys(objs):
        a = model.areas.get(name)
        sec = model.shell_sections.get(getattr(a, "section", None) or "") if a is not None else None
        t = float(sec.thickness) if sec is not None and sec.thickness else 0.0
        info[name] = (t, a is not None and a.type in ("Floor", "Ramp"))
    t = np.array([info[o][0] for o in objs], float)
    floor = np.array([info[o][1] for o in objs], bool)
    thick = t > 0
    nf = np.where((floor & (nrm[:, 2] < 0))[:, None], -nrm, nrm)
    lo = np.where(floor[:, None], -t[:, None] * nf, -0.5 * t[:, None] * nrm)
    hi = np.where(floor[:, None], np.zeros((m, 3)), 0.5 * t[:, None] * nrm)
    # faces in element order: 6 for a slab, 1 for a flat element (all quads)
    nfaces = np.where(thick, 6, 1)
    start = np.concatenate([[0], np.cumsum(nfaces)[:-1]])
    F = np.empty((int(nfaces.sum()), 4, 3))
    W = np.empty((len(F), 4))
    if np.any(thick):
        Ft, Wt = ScalarMesh.slab_faces(P[thick], V[thick], lo[thick], hi[thick])
        idx = (start[thick][:, None] + np.arange(6)).reshape(-1)
        F[idx] = Ft.reshape(-1, 4, 3)
        W[idx] = Wt.reshape(-1, 4)
    if not np.all(thick):
        F[start[~thick]] = P[~thick]
        W[start[~thick]] = V[~thick]
    mesh.polygons(F, W)


# -------------------------------------------------------------- frame colours
def frame_color_actors(model: Model, results, case: str, quantity: str, frames=None,
                       extruded: bool = False, line_width: float = 7.0) -> List[vtk.vtkProp]:
    """Members coloured along their length by ``quantity`` (+ colour scale)."""
    vals = frame_values(model, results, case, quantity, frames)
    if not vals:
        return []
    from . import scene
    nice = None
    if quantity == "Stress ratio":
        lut = ratio_lut()
    else:
        from ..display_units import DisplayUnits
        k = DisplayUnits.of(model).factor("moment" if quantity in ("T", "M2", "M3") else "force")
        vals = {n: (x, v * k) for n, (x, v) in vals.items()}  # in the display units
        # one colour scale in every window: the range over all members of the model
        every = vals if frames is None else {n: (x, v * k) for n, (x, v) in
                                             frame_values(model, results, case, quantity, None).items()}
        vmax = max(float(v.max()) for _x, v in every.values()) if every else 1.0
        lut, rng, step = scene.result_lut(0.0, vmax or 1e-12)
        nice = (rng[0], rng[1], step)
    solid, lines = ScalarMesh(), ScalarMesh()
    r = scene._tube_radius(model)
    for name, (x, v) in vals.items():
        f = model.frames[name]
        a, b = ex.frame_ends(model, f)
        L = float(np.linalg.norm(b - a))
        if L <= 1e-12 or len(x) < 2:
            continue
        order = np.argsort(x, kind="stable")
        x, v = x[order], v[order]
        path = a + np.clip(x / L, 0, 1)[:, None] * (b - a)
        if extruded:
            rings = frame_rings(model, f)
            if rings is None:
                rings = tube_rings(r, frame_axes(a, b, f.angle))
            solid.sweep(path, v, rings)
        else:
            lines.polyline(path, v)
    out: List[vtk.vtkProp] = []
    if solid.pts:
        out.append(mapped_actor(solid.polydata(), lut))
    if lines.pts:
        out.append(mapped_actor(lines.polydata(), lut, lit=False, line_width=line_width))
    if out:
        if nice is None:
            out.append(scalar_bar(lut, quantity_title(model, quantity), labels=0))
        else:
            out.append(scene.legend_bar(lut, quantity_title(model, quantity), *nice))
    return out


def ratio_summary(model: Model, results, case: str, frames=None) -> Dict[str, float]:
    """Largest stress ratio of every member (for tables / tests)."""
    return {n: float(v.max()) for n, (_x, v) in frame_values(model, results, case, "Stress ratio", frames).items()}
