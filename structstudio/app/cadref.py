"""CAD references (DXF drawings and PDF sheets to draw over) in the application.

* the drawing of a reference: read from the linked file or from the copy embedded in the
  model (:func:`drawing_of`), kept until the file is reloaded;
* its picture in a window (:func:`build_actor`): the visible layers as lines in the view
  plane, grey / in the drawing's colours, over or under the model;
* snapping to it (:class:`RefSnapper`): ends, midpoints, intersections, centres, the
  nearest point and the perpendicular foot on its lines;
* the commands: add, delete, change, move, turn, scale, align - each one undo step that
  keeps the analysis results (a reference is no part of the structure).
"""
from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..io.dxf_read import PT_CENTER, PT_END, PT_INSERT, Drawing, DxfError, read_dxf
from ..io.pdf_read import is_pdf, read_pdf
from ..model import Model, ModelError
from ..model.references import Blob, CadReference, align_two_points

EMBED_LIMIT = 60 * 1024 * 1024  # bytes: a larger drawing stays linked
_CACHE: Dict[tuple, Tuple[Optional[Drawing], str]] = {}
_STAMPS: Dict[tuple, Tuple[int, int]] = {}


def read_drawing(source, page: int = 1) -> Drawing:
    """A drawing file (a path, or its bytes): a DXF file, or page ``page`` of a PDF - told
    by what the file holds, not by its name.  Raises DxfError with a message for the user."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        try:
            with open(source, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise DxfError(f"The file cannot be read: {exc}") from None
    return read_pdf(data, page) if is_pdf(data) else read_dxf(data)


def _file_key(path: str, page: int = 1) -> tuple:
    return ("file", os.path.normcase(os.path.abspath(path)), int(page))


def _remember(path: str, page: int, dwg: Optional[Drawing], why: str = "") -> None:
    key = _file_key(path, page)
    _CACHE[key] = (dwg, why)
    try:
        st = os.stat(path)
        _STAMPS[key] = (st.st_mtime_ns, st.st_size)
    except OSError:
        _STAMPS.pop(key, None)


def forget() -> None:
    """Another model was opened: no drawing read for the last one is kept."""
    _CACHE.clear()
    _STAMPS.clear()


# ----------------------------------------------------------------------- the drawing
def linked_path(ref: CadReference, model_path: Optional[str]) -> Optional[str]:
    """The file of a linked reference: where it was chosen, else beside the model file
    (the model and its drawings moved together).  None when it is not there."""
    if ref.path and os.path.isfile(ref.path):
        return ref.path
    if ref.rel_path and model_path:
        p = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(model_path)), ref.rel_path))
        if os.path.isfile(p):
            return p
    return None


def source_of(ref: CadReference, model: Model, model_path: Optional[str]) -> Tuple[str, Optional[str]]:
    """("file", path) | ("embedded", key) | ("missing", None): where the drawing is read
    from - the linked file first (it may be newer), then the copy in the model."""
    if ref.path or ref.rel_path:
        p = linked_path(ref, model_path)
        if p is not None:
            return "file", p
    blob = model.attachments.get(ref.embedded) if ref.embedded else None
    if blob is not None and blob.stored_size:
        return "embedded", blob.key
    return "missing", None


def drawing_of(ref: CadReference, model: Model, model_path: Optional[str] = None) -> Tuple[Optional[Drawing], str]:
    """(drawing, "") or (None, why it cannot be shown).  A file is read once; see
    :func:`reload`.  A linked file that cannot be read (damaged, being written) gives way
    to the copy in the model, when there is one."""
    kind, where = source_of(ref, model, model_path)
    if kind == "missing":
        return None, f"the file is not found ({ref.path or ref.rel_path or 'no file'})"
    page = int(getattr(ref, "page", 1) or 1)
    if kind == "file":
        key = _file_key(where, page)
        got = _CACHE.get(key)
        if got is None:
            try:
                got = (read_drawing(where, page), "")
            except DxfError as exc:
                got = (None, str(exc))
            _remember(where, page, *got)
        if got[0] is not None:
            return got
        blob = model.attachments.get(ref.embedded) if ref.embedded else None
        if blob is None or not blob.stored_size:
            return got
        where = blob.key  # (the linked file cannot be read: the copy in the model)
    key = ("embedded", where, page)
    got = _CACHE.get(key)
    if got is None:
        try:
            got = (read_drawing(model.attachments[where].data, page), "")
        except DxfError as exc:
            got = (None, str(exc))
        _CACHE[key] = got
    return got


def changed_on_disk(ref: CadReference, model: Model, model_path: Optional[str] = None) -> bool:
    """The linked file was written again since it was read."""
    kind, where = source_of(ref, model, model_path)
    if kind != "file":
        return False
    old = _STAMPS.get(_file_key(where, int(getattr(ref, "page", 1) or 1)))
    if old is None:
        return False
    try:
        st = os.stat(where)
    except OSError:
        return False
    return (st.st_mtime_ns, st.st_size) != old


def reload(ref: Optional[CadReference] = None, model: Optional[Model] = None, model_path: Optional[str] = None) -> None:
    """Read the linked file(s) again at the next use (all files without ``ref``)."""
    if ref is None or model is None:
        for k in [k for k in _CACHE if k[0] == "file"]:
            _CACHE.pop(k, None)
        return
    kind, where = source_of(ref, model, model_path)
    if kind == "file":
        path = os.path.normcase(os.path.abspath(where))
        for k in [k for k in _CACHE if k[0] == "file" and k[1] == path]:
            _CACHE.pop(k, None)


def refresh_embedded(doc, names: Optional[Sequence[str]] = None, model_path: Optional[str] = None) -> List[str]:
    """Reload of references that are linked *and* embedded: the copies in the model
    follow their files (else the model sent alone would show the old drawing).  Returns
    the names whose copy was replaced (one undo step for all)."""
    m = doc.model
    fresh: Dict[str, Blob] = {}
    for name, ref in references_of(m).items():
        if (names is not None and name not in names) or not ref.embedded:
            continue
        kind, where = source_of(ref, m, model_path)
        if kind != "file":
            continue
        try:
            with open(where, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        if len(data) > EMBED_LIMIT:
            continue
        blob = Blob(data, os.path.basename(where))
        if blob.key == ref.embedded:
            continue
        page = int(getattr(ref, "page", 1) or 1)
        drawing_of(ref, m, model_path)
        dwg = (_CACHE.get(_file_key(where, page)) or (None, ""))[0]
        if dwg is None:
            continue  # (the file cannot be read now: the copy that can is kept)
        _CACHE[("embedded", blob.key, page)] = (dwg, "")
        fresh[name] = blob
    if not fresh:
        return []

    def mutate(model: Model):
        for name, blob in fresh.items():
            model.attachments[blob.key] = blob
            model.references[name].embedded = blob.key
        _prune_attachments(model)

    _edit(doc, mutate, "Reload Reference")
    return sorted(fresh)


def visible_layers(ref: CadReference, dwg: Drawing) -> np.ndarray:
    """True per layer of the drawing: shown."""
    off = set(ref.layers_off)
    return np.array([name not in off for name in dwg.layers], bool)


def plane_to_world(view: str, value: float, UV: np.ndarray) -> np.ndarray:
    UV = np.asarray(UV, float).reshape(-1, 2)
    out = np.empty((len(UV), 3))
    if view == "plan":
        out[:, 0], out[:, 1], out[:, 2] = UV[:, 0], UV[:, 1], value
    elif view == "xz":
        out[:, 0], out[:, 1], out[:, 2] = UV[:, 0], value, UV[:, 1]
    else:
        out[:, 0], out[:, 1], out[:, 2] = value, UV[:, 0], UV[:, 1]
    return out


def placed(ref: CadReference, length_unit: str, XY: np.ndarray) -> np.ndarray:
    """File points (N, 2) -> view plane (N, 2)."""
    (a, b, tx), (c, d, ty) = ref.matrix(length_unit)
    XY = np.asarray(XY, float).reshape(-1, 2)
    return np.column_stack([a * XY[:, 0] + b * XY[:, 1] + tx, c * XY[:, 0] + d * XY[:, 1] + ty])


def outline(ref: CadReference, dwg: Drawing, length_unit: str) -> Optional[np.ndarray]:
    """The four corners (view plane) of the box around the shown line work; None: nothing."""
    on = visible_layers(ref, dwg)
    m = on[dwg.seg_layer] if len(dwg.seg) else np.zeros(0, bool)
    if not m.any():
        return None
    s = dwg.seg[m]
    x0, x1 = min(s[:, 0].min(), s[:, 2].min()), max(s[:, 0].max(), s[:, 2].max())
    y0, y1 = min(s[:, 1].min(), s[:, 3].min()), max(s[:, 1].max(), s[:, 3].max())
    return placed(ref, length_unit, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))


# ------------------------------------------------------------------------ the picture
def _lum(rgb) -> float:
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0


MAX_TEXT_SEGMENTS = 1_500_000
_GLYPH_PX = 100.0
_TEXT_PATHS: Dict[str, Tuple[np.ndarray, float]] = {}
_TEXT_FONT: list = []


def _text_path(string: str) -> Tuple[np.ndarray, float]:
    """(the outlines of ``string`` as segments (K, 4), y up, baseline at 0, letters _GLYPH_PX
    high as a font size; its width) - drawn with the program's font, so any script the
    computer has a font for is written (Thai too)."""
    got = _TEXT_PATHS.get(string)
    if got is not None:
        return got
    from PySide6 import QtGui
    if not _TEXT_FONT:
        font = QtGui.QFont()
        font.setFamilies(["Tahoma", "Segoe UI", "Arial", "DejaVu Sans"])
        font.setStyleHint(QtGui.QFont.SansSerif)
        font.setPixelSize(int(_GLYPH_PX))
        fm = QtGui.QFontMetricsF(font)
        _TEXT_FONT.extend([font, fm, float(fm.capHeight()) or 0.72 * _GLYPH_PX])
    font, fm = _TEXT_FONT[0], _TEXT_FONT[1]
    path = QtGui.QPainterPath()
    path.addText(0.0, 0.0, font, string)
    segs = []
    for poly in path.toSubpathPolygons():
        P = np.array([(q.x(), -q.y()) for q in poly], float)
        if len(P) < 2:
            continue
        # (a letter is read from far fewer points than its outline is made of)
        keep = [0]
        for k in range(1, len(P) - 1):
            if abs(P[k, 0] - P[keep[-1], 0]) + abs(P[k, 1] - P[keep[-1], 1]) >= 6.0:
                keep.append(k)
        keep.append(len(P) - 1)
        P = P[keep]
        segs.append(np.hstack([P[:-1], P[1:]]))
    out = (np.vstack(segs) if segs else np.zeros((0, 4)), float(fm.horizontalAdvance(string)))
    if len(_TEXT_PATHS) > 20000:
        _TEXT_PATHS.clear()
    _TEXT_PATHS[string] = out
    return out


def text_segments(dwg: Drawing) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The texts of a drawing as line work to draw (never to snap to): (segments (M, 4) in
    file coordinates, layer index per segment, RGB per segment).  Worked out once per
    drawing; empty without a display (no fonts to draw with)."""
    got = getattr(dwg, "_text_segments", None)
    if got is not None:
        return got
    empty = (np.zeros((0, 4)), np.zeros(0, np.int32), np.zeros((0, 3), np.uint8))
    if not getattr(dwg, "texts", None):
        return empty
    try:
        from PySide6 import QtGui
        if QtGui.QGuiApplication.instance() is None:
            return empty
    except ImportError:
        return empty
    segs, lays, cols = [], [], []
    total = 0
    for (x, y, h, ang, string, lay, rgb, across, up, width, wf) in dwg.texts:
        U, w_px = _text_path(string)
        if not len(U) or not h > 0:
            continue
        cap = _TEXT_FONT[2]
        sy = h / cap
        sx = (width / w_px) if (width > 0 and w_px > 0) else sy * (wf or 1.0)
        dx = -0.5 * across * w_px * sx
        dy = {0: 0.0, 1: 0.22 * h, 2: -0.5 * h, 3: -h}.get(up, 0.0)
        a = math.radians(ang)
        c, sn = math.cos(a), math.sin(a)
        X1, Y1 = U[:, 0] * sx + dx, U[:, 1] * sy + dy
        X2, Y2 = U[:, 2] * sx + dx, U[:, 3] * sy + dy
        segs.append(np.column_stack([x + c * X1 - sn * Y1, y + sn * X1 + c * Y1, x + c * X2 - sn * Y2, y + sn * X2 + c * Y2]))
        lays.append(np.full(len(U), lay, np.int32))
        cols.append(np.broadcast_to(np.array(rgb, np.uint8), (len(U), 3)))
        total += len(U)
        if total > MAX_TEXT_SEGMENTS:
            break
    if not segs:
        return empty
    out = (np.vstack(segs), np.concatenate(lays), np.vstack(cols))
    try:
        dwg._text_segments = out
    except Exception:
        pass
    return out


def shown_lines(ref: Optional[CadReference], dwg: Drawing, layers_off=None, texts: bool = True,
                strings: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """(segments (N, 4), RGB (N, 3)) of what is drawn of a drawing as lines: the line work
    of the layers shown, then - with ``texts`` - the glyphs a PDF has as outlines and - with
    ``strings`` too - its texts as the outlines of the program's font."""
    off = set(ref.layers_off if layers_off is None and ref is not None else (layers_off or ()))
    on = np.array([name not in off for name in dwg.layers], bool)
    m = on[dwg.seg_layer] if len(dwg.seg) else np.zeros(0, bool)
    S, C = dwg.seg[m], dwg.seg_rgb[m]
    if texts and (ref is None or getattr(ref, "texts", True)):
        G = getattr(dwg, "text_seg", None)
        if G is not None and len(G):
            gm = on[dwg.text_seg_layer]
            S, C = np.vstack([S, G[gm]]), np.vstack([C, dwg.text_seg_rgb[gm]])
        if strings:
            T, tl, tc = text_segments(dwg)
            if len(T):
                tm = on[tl]
                S, C = np.vstack([S, T[tm]]), np.vstack([C, tc[tm]])
    return S, C


MIN_TEXT_PX = 3.0  # texts whose capitals are smaller on the screen are not drawn (a smudge, not a word)
MAX_TEXT_TRIANGLES = 3_000_000
_TEXT_FILL: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}


def _point_in_ring(ring: np.ndarray, x: float, y: float) -> bool:
    P, Q = ring, np.roll(ring, -1, axis=0)
    cross = (P[:, 1] <= y) != (Q[:, 1] <= y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = P[:, 0] + (y - P[:, 1]) * (Q[:, 0] - P[:, 0]) / (Q[:, 1] - P[:, 1])
    return bool(np.count_nonzero(cross & (x < xs)) % 2)


def _text_fill(string: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """The letters of ``string`` as filled triangles of the program's font: (points (P, 2), y
    up, baseline at 0, letters _GLYPH_PX high as a font size; triangles (T, 3); its width).
    A filled letter can be read where its outline is a smudge."""
    got = _TEXT_FILL.get(string)
    if got is not None:
        return got
    from PySide6 import QtGui
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator
    _text_path("")  # (the font)
    font, fm = _TEXT_FONT[0], _TEXT_FONT[1]
    path = QtGui.QPainterPath()
    path.addText(0.0, 0.0, font, string)
    rings = []
    for poly in path.toSubpathPolygons():
        P = np.array([(q.x(), -q.y()) for q in poly], float)
        if len(P) > 1 and (P[0] == P[-1]).all():
            P = P[:-1]
        if len(P) > 3:  # (fewer points: a letter is read from far fewer than its outline is made of)
            keep = np.r_[True, np.abs(np.diff(P, axis=0)).sum(axis=1) >= 2.0]
            P = P[keep]
        if len(P) >= 3:
            a = 0.5 * float(P[:, 0] @ np.roll(P[:, 1], -1) - P[:, 1] @ np.roll(P[:, 0], -1))
            if abs(a) > 1e-6:
                rings.append((P, a))
    empty = (np.zeros((0, 2)), np.zeros((0, 3), np.int64), float(fm.horizontalAdvance(string)))
    if not rings:
        _TEXT_FILL[string] = empty
        return empty
    # outer outlines one way round, holes the other (fonts differ: Thai and Latin letters did)
    boxes = [(P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()) for P, _ in rings]
    fixed = []
    for i, (P, a) in enumerate(rings):
        x, y = P[0]
        depth = sum(1 for j, (Q, _) in enumerate(rings) if j != i and boxes[j][0] <= x <= boxes[j][2]
                    and boxes[j][1] <= y <= boxes[j][3] and _point_in_ring(Q, x, y))
        fixed.append(P if (a > 0) == (depth % 2 == 0) else P[::-1])
    X = np.vstack(fixed)
    sizes = np.array([len(P) for P in fixed], np.int64)
    first = np.cumsum(sizes) - sizes
    conn = np.concatenate([np.r_[np.arange(n) + f0, f0] for n, f0 in zip(sizes.tolist(), first.tolist())])
    offsets = np.r_[0, np.cumsum(sizes + 1)]
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.column_stack([X, np.zeros(len(X))]), deep=True))
    lines = vtkCellArray()
    lines.SetData(numpy_support.numpy_to_vtkIdTypeArray(offsets.astype(np.int64), deep=True),
                  numpy_support.numpy_to_vtkIdTypeArray(conn.astype(np.int64), deep=True))
    pd = vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    tris = vtkCellArray()
    vtkContourTriangulator.TriangulateContours(pd, 0, lines.GetNumberOfCells(), tris, (0.0, 0.0, 1.0))
    T = numpy_support.vtk_to_numpy(tris.GetConnectivityArray()).reshape(-1, 3).astype(np.int64) \
        if tris.GetNumberOfCells() else np.zeros((0, 3), np.int64)
    out = (X, T, float(fm.horizontalAdvance(string)))
    if len(_TEXT_FILL) > 20000:
        _TEXT_FILL.clear()
    _TEXT_FILL[string] = out
    return out


def build_text_actors(ref: CadReference, dwg: Drawing, model: Model, view: str, value: float, background, ink):
    """The texts of a reference as filled letters in a window: [(actor, the height of its
    smallest capitals in model length)] - one actor per size (a factor 2 apart), so a window
    shows only the sizes that can be read at its zoom (see MIN_TEXT_PX)."""
    from . import vtk_lite as vtk
    from vtkmodules.util import numpy_support
    if not getattr(ref, "texts", True) or not getattr(dwg, "texts", None):
        return []
    try:
        from PySide6 import QtGui
        if QtGui.QGuiApplication.instance() is None:
            return []
    except ImportError:
        return []
    on = visible_layers(ref, dwg)
    unit = model.units.length
    f = ref.factor(unit)
    bands: Dict[int, list] = {}
    total = 0
    _text_path("")
    cap = _TEXT_FONT[2]
    for (x, y, h, ang, string, lay, rgb, across, up, width, wf) in dwg.texts:
        if not on[lay] or not h > 0:
            continue
        P, T, w_px = _text_fill(string)
        if not len(T):
            continue
        sy = h / cap
        sx = (width / w_px) if (width > 0 and w_px > 0) else sy * (wf or 1.0)
        dx = -0.5 * across * w_px * sx
        dy = {0: 0.0, 1: 0.22 * h, 2: -0.5 * h, 3: -h}.get(up, 0.0)
        a = math.radians(ang)
        c, sn = math.cos(a), math.sin(a)
        U, V = P[:, 0] * sx + dx, P[:, 1] * sy + dy
        F = np.column_stack([x + c * U - sn * V, y + sn * U + c * V])
        band = int(math.floor(math.log2(h * f))) if h * f > 0 else 0
        bands.setdefault(band, []).append((F, T, rgb))
        total += len(T)
        if total > MAX_TEXT_TRIANGLES:
            break
    out = []
    for band, items in sorted(bands.items()):
        Fs, Ts, Cs = [], [], []
        n0 = 0
        for F, T, rgb in items:
            Fs.append(F)
            Ts.append(T + n0)
            Cs.append(np.broadcast_to(np.array(rgb, np.uint8), (len(T), 3)))
            n0 += len(F)
        F = np.vstack(Fs)
        T = np.vstack(Ts)
        X = plane_to_world(view, value, placed(ref, unit, F))
        pts = vtk.vtkPoints()
        pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(X, dtype=float), deep=True))
        cells = vtk.vtkCellArray()
        cells.SetData(numpy_support.numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(T) + 1, 3, dtype=np.int64), deep=True),
                      numpy_support.numpy_to_vtkIdTypeArray(np.ascontiguousarray(T.ravel()), deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetPolys(cells)
        rgb, one = display_colors(ref, dwg, None, background, ink, rgb=np.vstack(Cs))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        if rgb is not None:
            arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(rgb), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            arr.SetName("rgb")
            poly.GetCellData().SetScalars(arr)
            mapper.SetScalarModeToUseCellData()
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0.0, -66000.0 if ref.on_top else 66000.0)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*one)
        prop.LightingOff()
        actor.PickableOff()
        actor.UseBoundsOff()
        out.append((actor, 2.0 ** band))
    return out


def display_colors(ref: CadReference, dwg: Drawing, mask: np.ndarray, background, ink,
                   rgb: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], tuple]:
    """(per-segment RGB uint8 or None, the one colour 0..1): "gray" - the window's ink
    faded towards the background; "#rrggbb" - that colour; "layers" - the drawing's own
    colours, those too close to the background (white lines on a white window) in ink."""
    bg = np.array([float(v) for v in background[:3]])
    fg = np.array([float(v) for v in ink[:3]])
    k = min(max(float(ref.strength), 0.05), 1.0)
    mode = (ref.color or "gray").lower()
    if mode.startswith("#") and len(mode) == 7:
        try:
            c = np.array([int(mode[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
        except ValueError:
            c = fg
        return None, tuple((bg + (c - bg) * k).tolist())
    if mode != "layers":
        return None, tuple((bg + (fg - bg) * (0.25 + 0.6 * k)).tolist())
    rgb = (dwg.seg_rgb[mask] if rgb is None else rgb).astype(float) / 255.0
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    bg_l = float(bg @ np.array([0.299, 0.587, 0.114]))
    low = np.abs(lum - bg_l) < 0.28
    rgb[low] = fg
    if bg_l > 0.5:  # a light window: the bright CAD colours (yellow, cyan) darker
        bright = (~low) & (lum > 0.62)
        rgb[bright] *= 0.62
    rgb = bg + (rgb - bg) * k
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8), (1.0, 1.0, 1.0)


def build_actor(ref: CadReference, dwg: Drawing, model: Model, view: str, value: float, background, ink,
                line_width: float = 1.0):
    """The lines of a reference in a window of ``view`` at plane ``value``; None when
    no layer is shown."""
    from . import vtk_lite as vtk
    from vtkmodules.util import numpy_support
    S, own = shown_lines(ref, dwg, strings=False)  # (the texts: filled letters of their own, build_text_actors)
    mask = None
    n = len(S)
    if not n:
        return None
    unit = model.units.length
    P = np.vstack([placed(ref, unit, S[:, 0:2]), placed(ref, unit, S[:, 2:4])])
    X = plane_to_world(view, value, P)
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(X, dtype=float), deep=True))
    conn = np.empty(2 * n, np.int64)
    conn[0::2] = np.arange(n)
    conn[1::2] = np.arange(n) + n
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_support.numpy_to_vtkIdTypeArray(np.arange(0, 2 * n + 1, 2, dtype=np.int64), deep=True),
                  numpy_support.numpy_to_vtkIdTypeArray(conn, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(cells)
    rgb, one = display_colors(ref, dwg, mask, background, ink, rgb=own)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    if rgb is not None:
        arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(rgb), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        arr.SetName("rgb")
        poly.GetCellData().SetScalars(arr)
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
    # over the model whatever is drawn at its level (extruded sections, filled floors),
    # or under it
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0.0, -66000.0 if ref.on_top else 66000.0)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*one)
    prop.SetLineWidth(line_width)
    prop.LightingOff()
    actor.PickableOff()
    actor.UseBoundsOff()  # (a drawing far away or in the wrong unit must not spoil the view's depth range)
    return actor


# -------------------------------------------------------------------------- snapping
class RefSnapper:
    """Snap points of the references a window shows.  Everything is worked out in the
    view plane: ``hits(uv, radius)`` gives what lies within ``radius`` (model length) of
    plane point ``uv``."""

    NEAR_LIMIT = 48  # lines near the cursor looked at for intersections

    def __init__(self, items: Sequence[Tuple[CadReference, Drawing]], length_unit: str):
        self.items = []
        for ref, dwg in items:
            if not ref.snap or dwg is None:
                continue
            on = visible_layers(ref, dwg)
            m = on[dwg.seg_layer] if len(dwg.seg) else np.zeros(0, bool)
            S = dwg.seg[m]
            pm = on[dwg.pts_layer] if len(dwg.pts) else np.zeros(0, bool)
            self.items.append({
                "ref": ref, "dwg": dwg, "seg": S, "curve": dwg.seg_curve[m], "layer": dwg.seg_layer[m],
                "lo": np.minimum(S[:, 0:2], S[:, 2:4]) if len(S) else np.zeros((0, 2)),
                "hi": np.maximum(S[:, 0:2], S[:, 2:4]) if len(S) else np.zeros((0, 2)),
                "pts": dwg.pts[pm], "pkind": dwg.pts_kind[pm], "player": dwg.pts_layer[pm],
                "factor": ref.factor(length_unit)})
        self.unit = length_unit

    def __bool__(self) -> bool:
        return bool(self.items)

    def hits(self, uv, radius: float, previous=None, kinds: Optional[Sequence[str]] = None,
             midpoints: bool = True, intersections: bool = True, nearest: bool = False,
             perpendicular: bool = True) -> List[Tuple[Tuple[float, float], str, str]]:
        """[(plane point, kind, "REF1: layer")]: kind is end | midpoint | intersection |
        center | nearest | perpendicular.  ``kinds``: only these."""
        out: List[Tuple[Tuple[float, float], str, str]] = []
        want = (lambda k: True) if kinds is None else (lambda k: k in kinds)
        for it in self.items:
            ref, dwg, f = it["ref"], it["dwg"], it["factor"]
            if f <= 0:
                continue
            q = np.array(ref.to_file(uv, self.unit), float)
            r = radius / f
            S = it["seg"]
            near = np.zeros(0, int)
            if len(S):
                box = np.flatnonzero((it["lo"][:, 0] <= q[0] + r) & (it["hi"][:, 0] >= q[0] - r) &
                                     (it["lo"][:, 1] <= q[1] + r) & (it["hi"][:, 1] >= q[1] - r))
                if len(box):
                    A, B = S[box, 0:2], S[box, 2:4]
                    D = B - A
                    L2 = np.maximum(np.einsum("ij,ij->i", D, D), 1e-300)
                    t = np.clip(np.einsum("ij,ij->i", q - A, D) / L2, 0.0, 1.0)
                    F = A + t[:, None] * D
                    d = np.hypot(F[:, 0] - q[0], F[:, 1] - q[1])
                    ok = d <= r
                    order = np.argsort(d[ok], kind="stable")[:self.NEAR_LIMIT]
                    near = box[ok][order]
                    foot = F[ok][order]

            def name(layer_index) -> str:
                return f"{ref.name}: {dwg.layers[int(layer_index)]}"

            def add(xy, kind, layer_index):
                out.append((ref.to_plane((float(xy[0]), float(xy[1])), self.unit), kind, name(layer_index)))

            if len(near):
                A, B = S[near, 0:2], S[near, 2:4]
                straight = ~it["curve"][near]
                lay = it["layer"][near]
                if want("end"):
                    for P in (A, B):
                        m = straight & (np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) <= r)
                        for k in np.flatnonzero(m):
                            add(P[k], "end", lay[k])
                if midpoints and want("midpoint"):
                    M = 0.5 * (A + B)
                    m = straight & (np.hypot(M[:, 0] - q[0], M[:, 1] - q[1]) <= r)
                    for k in np.flatnonzero(m):
                        add(M[k], "midpoint", lay[k])
                if intersections and want("intersection") and len(near) > 1:
                    n = len(near)
                    for i in range(n):
                        d1 = B[i] - A[i]
                        for j in range(i + 1, n):
                            d2 = B[j] - A[j]
                            den = d1[0] * d2[1] - d1[1] * d2[0]
                            if abs(den) < 1e-14 * max(float(np.hypot(*d1) * np.hypot(*d2)), 1e-300):
                                continue
                            w = A[j] - A[i]
                            t = (w[0] * d2[1] - w[1] * d2[0]) / den
                            u = (w[0] * d1[1] - w[1] * d1[0]) / den
                            if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
                                p = A[i] + t * d1
                                if np.hypot(p[0] - q[0], p[1] - q[1]) <= r:
                                    # (two lines that meet end to end - a corner, two chords of
                                    # one curve - do not cross there: it is their end)
                                    meet = min(max(np.hypot(*(p - e1)), np.hypot(*(p - e2)))
                                               for e1 in (A[i], B[i]) for e2 in (A[j], B[j]))
                                    if meet > 1e-9 * max(1.0, float(np.abs(p).max())):
                                        add(p, "intersection", lay[i])
                if perpendicular and want("perpendicular") and previous is not None:
                    pq = np.array(ref.to_file(previous, self.unit), float)
                    for k in np.flatnonzero(straight):
                        d1 = B[k] - A[k]
                        L2 = float(d1 @ d1)
                        if L2 <= 0:
                            continue
                        t = float((pq - A[k]) @ d1) / L2
                        if -1e-9 <= t <= 1 + 1e-9:
                            p = A[k] + t * d1
                            if np.hypot(p[0] - q[0], p[1] - q[1]) <= r and np.hypot(*(p - pq)) > 1e-12:
                                add(p, "perpendicular", lay[k])
                if nearest and want("nearest"):
                    add(foot[0], "nearest", lay[0])
            P = it["pts"]
            if len(P) and (want("end") or want("center")):
                m = np.flatnonzero(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) <= r)
                for k in m:
                    kind = "center" if it["pkind"][k] == PT_CENTER else "end"
                    if want(kind):
                        add(P[k], kind, it["player"][k])
        return out

    def which(self, uv, radius: float) -> Optional[str]:
        """The reference with a line within ``radius`` of ``uv`` (the nearest)."""
        got = self.under(uv, radius)
        return got[0] if got else None

    def under(self, uv, radius: float) -> Optional[Tuple[str, str]]:
        """(reference, layer) of the line nearest ``uv`` within ``radius``."""
        best, best_d = None, None
        for it in self.items:
            ref, f, S = it["ref"], it["factor"], it["seg"]
            if f <= 0 or not len(S):
                continue
            q = np.array(ref.to_file(uv, self.unit), float)
            r = radius / f
            box = np.flatnonzero((it["lo"][:, 0] <= q[0] + r) & (it["hi"][:, 0] >= q[0] - r) &
                                 (it["lo"][:, 1] <= q[1] + r) & (it["hi"][:, 1] >= q[1] - r))
            if not len(box):
                continue
            A, B = S[box, 0:2], S[box, 2:4]
            D = B - A
            L2 = np.maximum(np.einsum("ij,ij->i", D, D), 1e-300)
            t = np.clip(np.einsum("ij,ij->i", q - A, D) / L2, 0.0, 1.0)
            F = A + t[:, None] * D
            dist = np.hypot(F[:, 0] - q[0], F[:, 1] - q[1])
            j = int(np.argmin(dist))
            d = float(dist[j]) * f
            if d <= radius and (best_d is None or d < best_d):
                best, best_d = (ref.name, it["dwg"].layers[int(it["layer"][box[j]])]), d
        return best


# -------------------------------------------------------------------------- commands
def _edit(doc, func, name: Optional[str] = None):
    """One undo step that keeps the analysis results and the lock."""
    prev = getattr(doc, "command_name", None)
    if name and not prev:
        doc.command_name = name
    try:
        doc.edit_design(func, "references")
    finally:
        if name and not prev:
            doc.command_name = prev


def references_of(model: Model) -> Dict[str, CadReference]:
    return getattr(model, "references", {})


def shown_in(model: Model, view: str, value: float, story: Optional[str]) -> List[CadReference]:
    """The references a window of ``view`` at plane ``value`` (plan: of ``story``) shows."""
    tol = max(model.options.merge_tolerance, 1e-6) * 10
    return [r for r in references_of(model).values() if r.shown_on(view, value, story, tol, model.stories)]


def add_reference(doc, path: str, view: str = "plan", stories: Sequence[str] = (), level: float = 0.0,
                  unit: Optional[str] = None, embed: bool = False, link: bool = True,
                  origin: Optional[Tuple[float, float]] = None, base: Optional[Tuple[float, float]] = None,
                  layers_off: Optional[Sequence[str]] = None, name: Optional[str] = None, page: int = 1,
                  scale: Optional[float] = None) -> str:
    """Place the drawing of file ``path`` (a DXF file, or page ``page`` of a PDF) as a
    reference; returns its name.  ``embed``: a copy goes into the model file; ``link``:
    the file is read from where it is (and can be reloaded).  ``scale``: the plot scale
    of a PDF sheet (100 = 1:100; without it the scale the PDF states, else 100).  Raises
    ModelError when the file cannot be read."""
    if not (embed or link):
        link = True
    path = os.path.abspath(path)
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise ModelError("REFERENCE_FILE", f"The file cannot be read: {exc}") from None
    try:
        dwg = read_drawing(data, page)
    except DxfError as exc:
        raise ModelError("REFERENCE_FILE", str(exc)) from None
    page = int(dwg.page)
    if scale is None:
        scale = (dwg.plot_scale or 100.0) if dwg.source == "pdf" else 1.0
    if not (scale > 0 and math.isfinite(scale)):
        raise ModelError("BAD_VALUE", "The scale must be a positive number")
    if embed and len(data) > EMBED_LIMIT:
        raise ModelError("REFERENCE_TOO_LARGE", f"The drawing is {len(data) / 1e6:.0f} MB - too large to keep a copy "
                                                 "in the model file. Link it instead.")
    blob = Blob(data, os.path.basename(path)) if embed else None
    _remember(path, page, dwg)
    if blob is not None:
        _CACHE[("embedded", blob.key, page)] = (dwg, "")
    made = []

    def mutate(model: Model):
        nm = name or model._next_name("REF", model.references)
        if nm in model.references:
            raise ModelError("DUPLICATE_NAME", f"Reference {nm!r} already exists")
        off = list(layers_off) if layers_off is not None else list(dwg.layers_off)
        ref = CadReference(nm, path=path if link else "", embedded=blob.key if blob else None, view=view,
                           stories=[s for s in stories if s in model.stories] if view == "plan" else [],
                           level=float(level), unit="mm" if dwg.source == "pdf" else (unit or dwg.guess_unit()),
                           layers_off=off, page=page, scale=float(scale))
        if view == "plan" and ref.stories:
            ref.level = model.stories[ref.stories[0]].elevation
        # file point ``base`` goes to plane point ``origin`` (both (0, 0): the drawing's own
        # coordinates are the model's)
        ref.base = tuple(float(v) for v in (base or (0.0, 0.0)))
        ref.origin = tuple(float(v) for v in (origin or (0.0, 0.0)))
        ref.pinned = False  # (pinned once it has been put in place)
        if blob is not None:
            model.attachments[blob.key] = blob
        model.references[nm] = ref
        made.append(nm)

    _edit(doc, mutate, "Place Reference")
    return made[0]


def _prune_attachments(model: Model) -> None:
    used = {r.embedded for r in model.references.values() if r.embedded}
    for k in [k for k in model.attachments if k not in used]:
        del model.attachments[k]


def delete_references(doc, names: Sequence[str]) -> int:
    names = [n for n in names if n in references_of(doc.model)]
    if not names:
        return 0

    def mutate(model: Model):
        for n in names:
            model.references.pop(n, None)
        _prune_attachments(model)

    _edit(doc, mutate, "Delete Reference")
    return len(names)


def change_reference(doc, name: str, title: str = "Reference", **changes) -> None:
    """Set fields of a reference (visible, pinned, snap, on_top, color, strength,
    layers_off, stories, level, unit, scale ...)."""
    def mutate(model: Model):
        ref = model.references[name]
        for k, v in changes.items():
            if not hasattr(ref, k):
                raise ModelError("UNKNOWN_FIELD", f"A reference has no {k!r}")
            setattr(ref, k, list(v) if isinstance(v, (list, tuple)) and k in ("stories", "layers_off") else v)
        if "stories" in changes and ref.view == "plan" and ref.stories and ref.stories[0] in model.stories:
            ref.level = model.stories[ref.stories[0]].elevation

    _edit(doc, mutate, title)


def set_storage(doc, name: str, link: bool, embed: bool, model_path: Optional[str] = None) -> None:
    """Linked, embedded or both.  Embedding reads the drawing as it is now; dropping the
    link keeps only the copy; dropping the copy needs the file to be there."""
    m = doc.model
    ref = m.references[name]
    if not (link or embed):
        raise ModelError("REFERENCE_STORAGE", "A reference is linked, embedded or both")
    blob = None
    if embed and not ref.embedded:
        kind, where = source_of(ref, m, model_path)
        if kind != "file":
            raise ModelError("REFERENCE_FILE", "The file is not found: it cannot be embedded")
        with open(where, "rb") as fh:
            data = fh.read()
        if len(data) > EMBED_LIMIT:
            raise ModelError("REFERENCE_TOO_LARGE", f"The drawing is {len(data) / 1e6:.0f} MB - too large to embed")
        blob = Blob(data, os.path.basename(where))
    if not embed and linked_path(ref, model_path) is None:
        raise ModelError("REFERENCE_FILE", "The linked file is not found: the copy in the model is the only one - "
                                           "it is kept (choose the file again first)")
    if link and not (ref.path or ref.rel_path):
        raise ModelError("REFERENCE_FILE", "This reference has no file to link to (use Relink to choose one)")

    def mutate(model: Model):
        r = model.references[name]
        if blob is not None:
            model.attachments[blob.key] = blob
            r.embedded = blob.key
        if not embed:
            r.embedded = None
        if not link:
            r.path, r.rel_path = "", ""
        _prune_attachments(model)

    _edit(doc, mutate, "Reference Storage")


def relink(doc, name: str, path: str, model_path: Optional[str] = None) -> None:
    """Another file for a reference (its place, layers and look stay).  An embedded copy
    becomes a copy of the new file."""
    path = os.path.abspath(path)
    old = references_of(doc.model).get(name)
    if old is None:
        raise ModelError("UNKNOWN_REFERENCE", f"No reference {name!r}")
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise ModelError("REFERENCE_FILE", f"The file cannot be read: {exc}") from None
    page = int(getattr(old, "page", 1) or 1)
    try:
        dwg = read_drawing(data, page)
    except DxfError as exc:
        raise ModelError("REFERENCE_FILE", str(exc)) from None
    page = int(dwg.page)
    blob = None
    if old.embedded:
        if len(data) > EMBED_LIMIT:
            raise ModelError("REFERENCE_TOO_LARGE", f"The drawing is {len(data) / 1e6:.0f} MB - too large to keep a "
                                                     "copy in the model file. Untick Embedded first.")
        blob = Blob(data, os.path.basename(path))
        _CACHE[("embedded", blob.key, page)] = (dwg, "")
    _remember(path, page, dwg)

    was_pdf = old.unit == "mm" and old.source_label().lower().endswith(".pdf")
    kind_changed = (dwg.source == "pdf") != was_pdf

    def mutate(model: Model):
        r = model.references[name]
        r.path, r.rel_path, r.page = path, "", page
        if kind_changed:  # a sheet instead of a drawing in real units (or the other way): its own unit and scale
            if dwg.source == "pdf":
                r.unit, r.scale = "mm", float(dwg.plot_scale or 100.0)
            else:
                r.unit, r.scale = dwg.guess_unit(), 1.0
        if blob is not None:
            model.attachments[blob.key] = blob
            r.embedded = blob.key
            _prune_attachments(model)

    _edit(doc, mutate, "Relink Reference")
    if model_path:
        update_rel_paths(doc.model, model_path)


def set_page(doc, name: str, page: int, model_path: Optional[str] = None) -> None:
    """Another page of the PDF of a reference."""
    ref = references_of(doc.model).get(name)
    if ref is None:
        raise ModelError("UNKNOWN_REFERENCE", f"No reference {name!r}")
    probe = replace(ref, page=int(page))
    dwg, why = drawing_of(probe, doc.model, model_path)
    if dwg is None:
        raise ModelError("REFERENCE_FILE", why)
    if dwg.page != int(page):
        raise ModelError("BAD_VALUE", f"The drawing has {dwg.pages} page(s)")

    def mutate(model: Model):
        model.references[name].page = int(page)

    _edit(doc, mutate, "Reference Page")


def _movable(model: Model, name: str, force: bool = False) -> CadReference:
    """The reference to move: a pinned one is refused, unless ``force`` (the command asked
    the user) - then it is unpinned."""
    ref = references_of(model).get(name)
    if ref is None:
        raise ModelError("UNKNOWN_REFERENCE", f"No reference {name!r}")
    if ref.pinned and not force:
        raise ModelError("REFERENCE_PINNED", f"Reference {name} is pinned in place - unpin it first "
                                             "(References > Pin / Unpin)")
    ref.pinned = False
    return ref


def move_reference(doc, name: str, p_from, p_to, force: bool = False, pin: Optional[bool] = None) -> None:
    """Shift a reference in its view plane: point ``p_from`` of its picture goes to
    ``p_to``.  ``pin``: pinned (or not) once moved."""
    du, dv = p_to[0] - p_from[0], p_to[1] - p_from[1]

    def mutate(model: Model):
        r = _movable(model, name, force)
        r.origin = (r.origin[0] + du, r.origin[1] + dv)
        if pin is not None:
            r.pinned = bool(pin)

    _edit(doc, mutate, "Move Reference")


def rotate_reference(doc, name: str, center, angle_deg: float, force: bool = False, pin: Optional[bool] = None) -> None:
    def mutate(model: Model):
        r = _movable(model, name, force)
        r.rebase(center, model.units.length)
        r.rotation = (r.rotation + float(angle_deg)) % 360.0
        if pin is not None:
            r.pinned = bool(pin)

    _edit(doc, mutate, "Rotate Reference")


def scale_reference(doc, name: str, fixed, factor: float, force: bool = False, pin: Optional[bool] = None) -> None:
    """Resize a reference about plane point ``fixed`` (which stays where it is)."""
    if not (factor > 0 and math.isfinite(factor)):
        raise ModelError("BAD_VALUE", "The scale must be a positive number")

    def mutate(model: Model):
        r = _movable(model, name, force)
        r.rebase(fixed, model.units.length)
        r.scale = float(r.scale) * float(factor)
        if pin is not None:
            r.pinned = bool(pin)

    _edit(doc, mutate, "Scale Reference")


def scale_by_length(doc, name: str, p1, p2, real_length: float, force: bool = False,
                    pin: Optional[bool] = None) -> float:
    """The picture's distance p1 - p2 is really ``real_length``: resize about p1.  Returns
    the factor applied."""
    L = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if L <= 0:
        raise ModelError("BAD_VALUE", "The two points are the same point")
    if not (real_length > 0 and math.isfinite(real_length)):
        raise ModelError("BAD_VALUE", "The length must be a positive number")
    f = float(real_length) / L
    scale_reference(doc, name, p1, f, force, pin)
    return f


def align_reference(doc, name: str, p1, q1, p2, q2, scale: bool = False, force: bool = False,
                    pin: Optional[bool] = None) -> float:
    """Two points of the picture onto two points of the model (see
    :func:`structstudio.model.references.align_two_points`); returns |q1 q2| / |p1 p2|."""
    out = []

    def mutate(model: Model):
        r = _movable(model, name, force)
        out.append(align_two_points(r, model.units.length, p1, q1, p2, q2, scale))
        if pin is not None:
            r.pinned = bool(pin)

    _edit(doc, mutate, "Align Reference")
    return out[0] if out else 1.0


def update_rel_paths(model: Model, model_path: str) -> None:
    """Before the model is saved: where each linked file is from the model file (so the
    model and its drawings can move together)."""
    folder = os.path.dirname(os.path.abspath(model_path))
    for r in references_of(model).values():
        if not r.path:
            continue
        try:
            r.rel_path = os.path.relpath(r.path, folder).replace("\\", "/")
        except ValueError:  # another drive
            r.rel_path = ""


def describe(ref: CadReference, model: Model, model_path: Optional[str] = None) -> str:
    """One line for lists: where it is from and whether it can be shown."""
    kind, where = source_of(ref, model, model_path)
    link = bool(ref.path or ref.rel_path)
    how = "linked + embedded" if link and ref.embedded else "linked" if link else "embedded"
    if kind == "missing":
        return f"{how} - FILE NOT FOUND"
    if link and kind == "embedded":
        return f"{how} - file not found, the embedded copy is shown"
    return how
