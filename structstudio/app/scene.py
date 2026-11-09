"""Pure-VTK scene builders (no Qt) for the desktop viewport.

Everything here takes a :class:`~structstudio.model.Model` (and optionally
results) and returns VTK actors, so it can be rendered off-screen in tests.
Coordinates are the model's own length units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from ..mesh import openings as opn
from ..mesh import polygon2d as pg
from ..model import Model

# Active display colours (RGB 0..1).  Filled from the active theme
# (structstudio.app.theme.apply); the dark theme is the default.
from .theme import DEFAULTS as _THEME_DEFAULTS  # noqa: E402

COLORS = dict(_THEME_DEFAULTS["light"])
# colour objects by "object" type, "section" or "material"
DISPLAY = {"color_by": "object", "theme": "light", "opacity": {}}

# selection highlight: dashed lines so the highlight colour is never mistaken for an
# object's own colour (the colour itself is set with Options > Graphics Colors).
# The OpenGL2 backend ignores line stipple, so the dashes are real geometry: every
# highlighted segment is cut into dashes of DASH_FRACTION x the model extent.
# line weights (pixels) and symbol sizes of the model window, set with Options >
# Graphics Sizes (theme.SIZE_ITEMS): heavy members over light area edges, the CAD
# hierarchy of ETABS / CYPE 3D (columns heaviest, null lines thin)
from .theme import SIZE_DEFAULTS as _SIZE_DEFAULTS  # noqa: E402
_FRAME_WIDTH_KEY = {"Beam": "frame_width", "Brace": "frame_width", "Column": "column_width", "Null": "null_width"}


def gsize(key: str) -> float:
    """Active graphics size (line width in px, text size in pt or a symbol scale)."""
    return float(DISPLAY.get("sizes", {}).get(key, _SIZE_DEFAULTS[key]))


def gopacity(key: str) -> float:
    """Active opacity (0..1) of an item of theme.OPACITY_ITEMS."""
    from .theme import OPACITY_DEFAULTS
    return float(DISPLAY.get("opacity", {}).get(key, OPACITY_DEFAULTS[key]))


def goption(key: str) -> bool:
    """Active on / off drawing option of theme.OPTION_ITEMS."""
    from .theme import OPTION_DEFAULTS
    return bool(DISPLAY.get("options", {}).get(key, OPTION_DEFAULTS[key]))


def gchoice(key: str) -> str:
    """Active pick-one drawing setting of theme.CHOICE_ITEMS."""
    from .theme import CHOICE_DEFAULTS
    return str(DISPLAY.get("choices", {}).get(key, CHOICE_DEFAULTS[key]))


def _text_pt(offset: int = 0) -> int:
    """Font size of object labels / grid text: the user's size + ``offset`` pt."""
    return max(6, int(round(gsize("label_text"))) + offset)

# lines and points drawn exactly on a filled area (beam on a slab edge, area edges)
# are pulled in front of the fill instead of z-fighting with it
vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
try:
    vtk.vtkMapper.SetResolveCoincidentTopologyLineOffsetParameters(0.0, -4.0)
    vtk.vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)
except AttributeError:  # older VTK
    pass

DASH_FRACTION = 0.03
DASH_GAP = 0.6  # gap as a fraction of the dash length


def dashed_segments(coords: np.ndarray, dash: float) -> Tuple[np.ndarray, vtk.vtkCellArray]:
    """Cut the line segments coords[2k] -> coords[2k+1] into dashes (vtk cells)."""
    out = []
    lines = vtk.vtkCellArray()
    period = dash * (1.0 + DASH_GAP)
    for k in range(len(coords) // 2):
        a, b = np.asarray(coords[2 * k], float), np.asarray(coords[2 * k + 1], float)
        L = float(np.linalg.norm(b - a))
        if L <= 0:
            continue
        n = max(1, int(L / period))
        # centre the pattern: whole dashes, the end stubs symmetrical
        start = 0.5 * (L - n * period) + 0.5 * dash * DASH_GAP
        for i in range(n):
            s0 = max(0.0, start + i * period)
            s1 = min(L, s0 + dash)
            if s1 <= s0:
                continue
            lines.InsertNextCell(2)
            lines.InsertCellPoint(len(out))
            out.append(a + (b - a) * s0 / L)
            lines.InsertCellPoint(len(out))
            out.append(a + (b - a) * s1 / L)
        if n == 1 and L < dash:  # very short segment: draw it whole
            lines.InsertNextCell(2)
            lines.InsertCellPoint(len(out))
            out.append(a)
            lines.InsertCellPoint(len(out))
            out.append(b)
    return (np.array(out) if out else np.zeros((0, 3))), lines


def _dashed_lines_actor(model: Model, coords, color, width: float) -> vtk.vtkActor:
    pts, lines = dashed_segments(np.asarray(coords, float), max(model_extent(model) * DASH_FRACTION, 1e-3))
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(pts))
    poly.SetLines(lines)
    return _actor_from_polydata(poly, color, width, lighting=False)


def _group_color(model: Model, obj, kind: str, default_key: str):
    """(group key, colour) of a frame / area under the active colour-by option."""
    from .theme import object_color

    by = DISPLAY.get("color_by", "object")
    if by == "object":
        return default_key, COLORS[default_key]
    table = model.frame_sections if kind == "frame" else model.shell_sections
    sec = table.get(obj.section or "")
    if sec is None:
        return default_key, COLORS[default_key]
    if by == "material":
        mat = model.materials.get(sec.material)
        idx = list(model.materials).index(sec.material) if mat is not None else None
        return "mat:" + sec.material, object_color(sec.material, getattr(mat, "color", None), DISPLAY.get("theme"), idx)
    # frame and shell sections share one hue sequence so a slab never matches a beam
    idx = list(table).index(sec.name) + (len(model.frame_sections) if kind != "frame" else 0)
    return "sec:" + sec.name, object_color(sec.name, sec.color, DISPLAY.get("theme"), idx)


def _actor_from_polydata(poly: vtk.vtkPolyData, color, line_width: float = 1.0, opacity: float = 1.0,
                         lighting: bool = True) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(line_width)
    actor.GetProperty().SetOpacity(opacity)
    if not lighting:
        actor.GetProperty().LightingOff()
    else:  # lit solids (floors, supports, extruded members): no muddy dark sides in either mode
        actor.GetProperty().SetAmbient(0.35)
        actor.GetProperty().SetDiffuse(0.75)
    return actor


def _points(coords: np.ndarray) -> vtk.vtkPoints:
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(coords, dtype=float), deep=True))
    return pts


@dataclass
class SceneActors:
    actors: Dict[str, vtk.vtkProp] = field(default_factory=dict)
    frame_ids: List[str] = field(default_factory=list)  # cell index -> frame name (for picking)
    area_ids: List[str] = field(default_factory=list)
    point_ids: List[str] = field(default_factory=list)

    def all(self) -> List[vtk.vtkProp]:
        return list(self.actors.values())


# ----------------------------------------------------------------------------
# Undeformed model
# ----------------------------------------------------------------------------
def _visible(obj, story: Optional[str]) -> bool:
    return story is None or obj.story == story


def opening_cross(P) -> list:
    """The two diagonals (4 end points) of an opening: corner to opposite corner for a
    quadrilateral, else the diagonals of its bounding rectangle in its plane."""
    P = np.asarray(P, float)
    if len(P) == 4:
        return [P[0], P[2], P[1], P[3]]
    from ..mesh import polygon2d as pg
    try:
        fr = pg.plane_frame(P)
    except ValueError:
        return []
    Q = pg.to_2d(P, fr)
    (u0, v0), (u1, v1) = Q.min(0), Q.max(0)
    C = pg.to_3d([(u0, v0), (u1, v1), (u1, v0), (u0, v1)], fr)
    return [C[0], C[1], C[2], C[3]]


def inset_polygon(P: np.ndarray, model: Model) -> np.ndarray:
    """``P`` moved inward in its plane by a small distance (each edge parallel to
    itself): the dashed highlight of a selected floor / wall then sits inside the
    area, clear of the beams along its edges.  The distance is a few % of the
    area's smaller span, at most ~3 % of a typical member length."""
    from ..mesh import polygon2d as pg
    try:
        fr = pg.plane_frame(P)
    except ValueError:
        return P
    Q = pg.to_2d(P, fr)
    if pg.signed_area(Q) < 0:
        Q, P, flip = Q[::-1], P[::-1], True
    else:
        flip = False
    span = float(min(np.ptp(Q[:, 0]), np.ptp(Q[:, 1])))
    if span <= 0:
        return P
    d = 0.08 * span
    typ = typical_member_length(model)
    if typ > 0:
        d = min(d, 0.07 * typ)  # clear of the beams along the edges, even with many selected
    n = len(Q)
    out = []
    for i in range(n):
        a0, a1 = Q[i - 1], Q[i]
        b0, b1 = Q[i], Q[(i + 1) % n]
        ea, eb = a1 - a0, b1 - b0
        if np.linalg.norm(ea) < 1e-12 or np.linalg.norm(eb) < 1e-12:
            out.append(Q[i])
            continue
        na = np.array([-ea[1], ea[0]]) / np.linalg.norm(ea)  # inward (CCW polygon: left)
        nb = np.array([-eb[1], eb[0]]) / np.linalg.norm(eb)
        p, q = a0 + na * d, b0 + nb * d
        den = ea[0] * eb[1] - ea[1] * eb[0]
        if abs(den) < 1e-12:
            out.append(Q[i] + na * d)
            continue
        t = ((q - p)[0] * eb[1] - (q - p)[1] * eb[0]) / den
        c = p + ea * t
        if np.linalg.norm(c - Q[i]) > 4 * d:  # a sharp corner: do not spike out
            c = Q[i] + (na + nb) * 0.5 * d
        out.append(c)
    R = pg.to_3d(np.array(out), fr)
    return R[::-1] if flip else R


def wall_footprint(model: Model, area) -> Optional[List[Tuple[float, float]]]:
    """Plan footprint of a wall: the rectangle (4 corners x, y) of its horizontal
    extent and its section thickness, centred on the wall plane.  None for a wall
    that is not vertical or has no thickness."""
    sec = model.shell_sections.get(area.section or "")
    t = float(sec.thickness) if sec is not None and sec.thickness else 0.0
    if t <= 0 or len(area.points) < 3:
        return None
    P = np.array([model.points[p].xyz for p in area.points], float)
    xy = P[:, :2]
    c = xy.mean(axis=0)
    # direction of the wall in plan: the longest horizontal spread of its corners
    d = xy - c
    _u, _s, vt = np.linalg.svd(d, full_matrices=False)
    u = vt[0]
    if not np.isfinite(u).all() or np.linalg.norm(u) < 1e-12:
        return None
    s = d @ u
    if float(s.max() - s.min()) <= 1e-9 or float(np.abs(d @ np.array([-u[1], u[0]])).max()) > max(t, 1e-6) * 2:
        return None  # a point in plan, or not a vertical wall
    n = np.array([-u[1], u[0]]) * (0.5 * t)
    a, b = c + u * s.min(), c + u * s.max()
    return [tuple(a - n), tuple(b - n), tuple(b + n), tuple(a + n)]


def visible_sets(model: Model, story: Optional[str] = None, plane: Optional[Tuple[str, float]] = None,
                 tol: Optional[float] = None):
    """Objects shown by a filtered window: ``story`` (plan view of one story)
    or ``plane`` = ("x"|"y", ordinate) (elevation on a grid line).
    Returns (frames, areas, points) name sets, or None when unfiltered."""
    if story is None and plane is None:
        return None
    pts = model.points
    if plane is not None:
        axis, value = plane
        k = {"x": 0, "y": 1, "z": 2}[axis]
        if tol is None:
            from .tools import plane_band
            tol = plane_band(model)  # the same band the snaps use (D05)
        on = {n for n, p in pts.items() if abs(p.xyz[k] - value) <= tol}
        frames = {f.name for f in model.frames.values() if f.point_i in on and f.point_j in on}
        areas = {a.name for a in model.areas.values() if all(p in on for p in a.points)}
        if axis == "z":  # a free plan (SAP2000 X-Y plane at any Z): columns crossing the plane too
            for f in model.frames.values():
                za, zb = pts[f.point_i].z, pts[f.point_j].z
                if min(za, zb) < value - tol < max(za, zb) or (f.point_j in on and za < value - tol) \
                        or (f.point_i in on and zb < value - tol):
                    frames.add(f.name)
        return frames, areas, on
    frames = {f.name for f in model.frames.values() if f.story == story}
    areas = {a.name for a in model.areas.values() if a.story == story}
    points = {p.name for p in pts.values() if p.story == story}
    for f in model.frames.values():
        if f.name in frames:
            points |= {f.point_i, f.point_j} if model.frames[f.name].type != "Column" else {
                f.point_j if pts[f.point_j].z >= pts[f.point_i].z else f.point_i}
    return frames, areas, points


def make_visibility(model: Model, story: Optional[str] = None, plane: Optional[Tuple[str, float]] = None):
    """Predicate obj -> bool (frames and areas) for the objects a window shows."""
    vis = visible_sets(model, story, plane)
    if vis is None:
        return lambda obj: True
    frames, areas, _ = vis
    return lambda obj: obj.name in (frames if hasattr(obj, "point_i") else areas)


def build_elevation_grid(model: Model, plane: Tuple[str, float]) -> List[vtk.vtkProp]:
    """Elevation grid: vertical grid lines in the plane and story level lines, with labels."""
    axis, value = plane
    k = 0 if axis == "x" else 1          # coordinate that is constant in the plane
    along = "Y" if axis == "x" else "X"   # grid lines crossing the plane
    ords = sorted({g.ordinate for g in model.grid_lines if g.direction == along})
    stories = model.stories_bottom_up()
    if not stories:
        return []
    z0, z1 = stories[0].elevation, stories[-1].elevation
    if not ords:
        return []
    o0, o1 = ords[0], ords[-1]
    ext = max(o1 - o0, z1 - z0, 1.0) * 0.06

    def P(o, z):
        return (value, o, z) if axis == "x" else (o, value, z)

    coords, lines = [], vtk.vtkCellArray()

    def seg(a, b):
        n = len(coords)
        coords.extend([a, b])
        lines.InsertNextCell(2)
        lines.InsertCellPoint(n)
        lines.InsertCellPoint(n + 1)

    for o in ords:
        seg(P(o, z0 - ext), P(o, z1 + ext))
    for st in stories:
        seg(P(o0 - ext, st.elevation), P(o1 + ext, st.elevation))
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(np.array(coords, float)))
    poly.SetLines(lines)
    grid = _actor_from_polydata(poly, COLORS["grid"], gsize("grid_width"), opacity=gopacity("grid_lines"),
                                lighting=False)
    pts = vtk.vtkPoints()
    texts = vtk.vtkStringArray()
    texts.SetName("labels")
    r = _bubble_radius(model)
    off = r * (1.0 + 0.6 * gsize("bubble_gap"))
    items = [(P(g.ordinate, z1 + ext + off), g.name, P(g.ordinate, z1 + ext))
             for g in model.grid_lines if g.direction == along]
    e1 = (0, 1, 0) if axis == "x" else (1, 0, 0)
    bubbles = bubble_actor(items, e1, (0, 0, 1), r, COLORS["grid_text"])
    for st in stories:
        pts.InsertNextPoint(*P(o0 - 2.5 * ext, st.elevation))
        texts.InsertNextValue(st.name)
    out = [grid, _text_labels(pts, texts, COLORS["grid_text"], _text_pt(2))]
    if bubbles is not None:
        out.append(bubbles)
    return out


MODEL_LAYERS = ("base", "selection", "labels")
SELECTION_KEYS_END = ("_sel", "_sel_lines", "_sel_tubes")


def model_layer_of(key: str) -> str:
    """The layer an actor key of :func:`build_model_actors` belongs to."""
    if key.endswith(SELECTION_KEYS_END) or key in ("joints", "areas_sel", "points_sel"):
        return "selection"
    if key in ("labels", "joint_labels", "area_labels"):
        return "labels"
    return "base"


def build_model_actors(model: Model, show_labels: bool = False, extruded: bool = False,
                       selection: Optional[set] = None, story: Optional[str] = None,
                       selected_areas: Optional[set] = None, selected_points: Optional[set] = None,
                       plane: Optional[Tuple[str, float]] = None, options=None,
                       hidden: Optional[Tuple[set, set, set]] = None,
                       layers: Sequence[str] = MODEL_LAYERS) -> SceneActors:
    """Actors for the model.  ``story`` limits the display to one story (plan
    view), ``plane`` to one elevation; ``selection``/``selected_areas``/
    ``selected_points`` are highlighted.  ``options`` (:class:`~.display.DisplayOptions`)
    decides what is shown; without it ``show_labels`` / ``extruded`` apply.

    ``layers`` picks the parts to build: "base" (objects, supports, grids ... -
    independent of the selection), "selection" (the highlights and the joint
    marks, which leave out the selected joints) and "labels" (text).  A window
    rebuilds only the layer a change touches."""
    from .display import DisplayOptions

    base_on, sel_on, labels_on = "base" in layers, "selection" in layers, "labels" in layers
    opts = options if options is not None else DisplayOptions(frame_labels=show_labels, extrude=extruded)
    extruded = opts.extrude
    out = SceneActors()
    selection = selection or set()
    selected_areas = selected_areas or set()
    selected_points = selected_points or set()
    pts = model.points
    vis = visible_sets(model, story, plane)
    if hidden is not None and any(hidden):  # View > Make Selected Objects Invisible / Show Selected Only
        base = vis or (set(model.frames), set(model.areas), set(model.points))
        hf, ha, hp = hidden
        frames = set(base[0]) - set(hf)
        areas = set(base[1]) - set(ha)
        vis = (frames, areas, set(base[2]) - set(hp))
    vis_f = vis[0] if vis else None
    vis_a = vis[1] if vis else None
    vis_p = vis[2] if vis else None
    # ---- frames (lines, coloured by type / section / material) --------------
    for ftype, color_key in (("Beam", "beam"), ("Column", "column"), ("Brace", "brace"), ("Null", "null_line")):
        groups: Dict[str, Tuple[tuple, List[str]]] = {}
        if not opts.shows_frame_type(ftype):
            continue
        for f in model.frames.values():
            if f.type == ftype and (vis_f is None or f.name in vis_f):
                gkey, color = _group_color(model, f, "frame", color_key) if ftype != "Null" else (color_key, COLORS[color_key])
                groups.setdefault(gkey, (color, []))[1].append(f.name)
        for gkey, (color, names) in groups.items():
            key = f"frames_{ftype}" if gkey == color_key else f"frames_{ftype}_{gkey}"
            out.frame_ids += names
            sel = [n for n in names if n in selection] if sel_on else []
            if extruded and ftype != "Null":
                # real cross-sections (ETABS "Extrude"); selected members as solids in the selection colour
                from .extrude import frame_solids
                if sel:
                    ssolid, smissing = frame_solids(model, sel)
                    if ssolid is not None:
                        # translucent solid in the highlight colour over the member (drawn in
                        # front of the coincident faces) plus a dashed centre line
                        sa = _actor_from_polydata(ssolid, COLORS["selection"], opacity=0.55)
                        sa.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
                        out.actors[key + "_sel"] = sa
                    if smissing:
                        out.actors[key + "_sel_tubes"] = _tube_actor(model, smissing, COLORS["selection"])
                    out.actors[key + "_sel_lines"] = _frame_lines_actor(model, sel, COLORS["selection"], 3.0, dashed=True)
                if not base_on:
                    continue
                solid, missing = frame_solids(model, names)
                if solid is not None:
                    out.actors[key] = _actor_from_polydata(solid, color)
                if solid is not None and opts.edges:
                    fe = vtk.vtkFeatureEdges()
                    fe.SetInputData(solid)
                    fe.BoundaryEdgesOff()
                    fe.FeatureEdgesOn()
                    fe.SetFeatureAngle(40.0)
                    fe.ManifoldEdgesOff()
                    fe.NonManifoldEdgesOff()
                    fe.Update()
                    out.actors[key + "_edges"] = _actor_from_polydata(fe.GetOutput(), COLORS["area_edge"], 1.0,
                                                                      lighting=False)
                if missing:
                    out.actors[key + "_tubes"] = _tube_actor(model, missing, color)
                continue
            width = gsize(_FRAME_WIDTH_KEY.get(ftype, "frame_width"))
            if not base_on:
                if sel:
                    out.actors[key + "_sel"] = _frame_lines_actor(model, sel, COLORS["selection"], width + 2.5,
                                                                  dashed=True)
                continue
            coords = []
            lines = vtk.vtkCellArray()
            for k, n in enumerate(names):
                f = model.frames[n]
                a, b = pts[f.point_i], pts[f.point_j]
                coords.append(a.xyz)
                coords.append(b.xyz)
                lines.InsertNextCell(2)
                lines.InsertCellPoint(2 * k)
                lines.InsertCellPoint(2 * k + 1)
            poly = vtk.vtkPolyData()
            poly.SetPoints(_points(np.array(coords)))
            poly.SetLines(lines)
            out.actors[key] = _actor_from_polydata(
                poly, color, line_width=width, lighting=False,
                opacity=gopacity("column_lines") if ftype == "Column" else 1.0)
            # selected frames highlighted
            if sel:
                out.actors[key + "_sel"] = _frame_lines_actor(model, sel, COLORS["selection"], width + 2.5,
                                                              dashed=True)
    # ---- areas (polygons) ------------------------------------------------------
    openings = None
    tol = max(float(model.options.merge_tolerance or 1e-3), 1e-6)
    for atype, color_key, opacity in (("Floor", "floor", gopacity("floor_fill")), ("Wall", "wall", gopacity("wall_fill")),
                                      ("Ramp", "floor", gopacity("floor_fill")),
                                      ("Opening", "opening", 0.15), ("Null", "opening", 0.15)):
        groups: Dict[str, Tuple[tuple, list]] = {}
        if not base_on or not opts.shows_area_type(atype):
            continue
        for a in model.areas.values():
            if a.type == atype and (vis_a is None or a.name in vis_a):
                gkey, color = (_group_color(model, a, "area", color_key) if color_key != "opening"
                               else (color_key, COLORS[color_key]))
                groups.setdefault(gkey, (color, []))[1].append(a)
        for gkey, (color, areas) in groups.items():
            coords = []
            polys = vtk.vtkCellArray()
            fcoords = []  # fill: openings cut out, non-convex outlines triangulated
            fpolys = vtk.vtkCellArray()
            for a in areas:
                start = len(coords)
                for p in a.points:
                    coords.append(pts[p].xyz)
                polys.InsertNextCell(len(a.points))
                for k in range(len(a.points)):
                    polys.InsertCellPoint(start + k)
                P = np.array([pts[p].xyz for p in a.points], float)
                if color_key == "opening":
                    pieces = [P]
                else:
                    if openings is None:
                        openings = opn.OpeningIndex(model)
                    try:
                        fr = pg.plane_frame(P)
                        holes = openings.holes(P, fr, tol, exclude=(a.name,))
                    except ValueError:
                        holes = []
                    pieces = opn.fill_polygons(P, holes, tol)
                for Pc in pieces:
                    s0 = len(fcoords)
                    fcoords.extend(Pc)
                    fpolys.InsertNextCell(len(Pc))
                    for k in range(len(Pc)):
                        fpolys.InsertCellPoint(s0 + k)
                    out.area_ids.append(a.name)
            poly = vtk.vtkPolyData()
            poly.SetPoints(_points(np.array(coords)))
            poly.SetPolys(polys)
            fill = vtk.vtkPolyData()
            fill.SetPoints(_points(np.array(fcoords)))
            fill.SetPolys(fpolys)
            key = f"areas_{atype}" if gkey == color_key else f"areas_{atype}_{gkey}"
            if extruded and color_key != "opening":
                from .extrude import area_solids
                solid = area_solids(model, areas)
                if solid is not None:
                    if opts.fill_areas:
                        out.actors[key + "_solid"] = _actor_from_polydata(
                            solid, color, opacity=min(1.0, opacity + 0.25) * (0.6 if story else 1.0))
                    if not (opts.edges or not opts.fill_areas):
                        continue
                    sedges = vtk.vtkFeatureEdges()
                    sedges.SetInputData(solid)
                    sedges.BoundaryEdgesOff()
                    sedges.FeatureEdgesOn()
                    sedges.SetFeatureAngle(30.0)
                    sedges.ManifoldEdgesOff()
                    sedges.NonManifoldEdgesOff()
                    sedges.Update()
                    out.actors[key + "_solid_edges"] = _actor_from_polydata(sedges.GetOutput(), COLORS["area_edge"],
                                                                            1.0, lighting=False)
                    continue
            # an opening is a hole: no fill of its own (the floor / wall around it is cut),
            # only its outline - so it reads as open, extruded or not
            if opts.fill_areas and color_key != "opening":
                out.actors[key] = _actor_from_polydata(fill, color, opacity=opacity * (0.6 if story else 1.0))
            if not (opts.edges or not opts.fill_areas):
                continue
            edges = vtk.vtkFeatureEdges()
            edges.SetInputData(poly)
            edges.BoundaryEdgesOn()
            edges.FeatureEdgesOff()
            edges.ManifoldEdgesOn()
            edges.Update()
            out.actors[key + "_edges"] = _actor_from_polydata(edges.GetOutput(), COLORS["area_edge"],
                                                              gsize("area_edge_width"), lighting=False)
            if color_key == "opening" and (story is not None or plane is not None):
                # the drawing symbol of an opening in plans / elevations: a dashed cross
                seg = []
                for a in areas:
                    seg += opening_cross([pts[p].xyz for p in a.points])
                if seg:
                    out.actors[key + "_cross"] = _dashed_lines_actor(model, seg, COLORS["area_edge"],
                                                                     max(1.0, gsize("area_edge_width") * 0.75))
    # ---- plan-view column markers (section outline at the story level) -------
    if base_on and story is not None and not extruded and opts.columns:
        cols = [f for f in model.frames.values() if f.type == "Column" and f.story == story
                and (vis_f is None or f.name in vis_f)]  # hidden columns (Show only / Make invisible) too
        groups: Dict[str, Tuple[tuple, list]] = {}
        for f in cols:
            gkey, color = _group_color(model, f, "frame", "column")
            groups.setdefault(gkey, (color, []))[1].append(f)
        z = model.stories[story].elevation if story in model.stories else 0.0
        dz = _tube_radius(model) * 0.5
        for gkey, (color, gcols) in groups.items():
            coords, polys = [], vtk.vtkCellArray()
            for f in gcols:
                top = pts[f.point_j] if pts[f.point_j].z >= pts[f.point_i].z else pts[f.point_i]
                sec = model.frame_sections.get(f.section)
                b = (getattr(sec, "t2", None) or 0.0) if sec else 0.0
                d = (getattr(sec, "t3", None) or 0.0) if sec else 0.0
                if b <= 0 or d <= 0:
                    b = d = _tube_radius(model) * 3.0
                ang = math.radians(f.angle or 0.0)
                c, sn = math.cos(ang), math.sin(ang)
                start = len(coords)
                for ex, ey in ((d / 2, b / 2), (-d / 2, b / 2), (-d / 2, -b / 2), (d / 2, -b / 2)):
                    # local 2 = +X for columns (CSI), rotated by the member angle
                    coords.append((top.x + ex * c - ey * sn, top.y + ex * sn + ey * c, z + dz))
                polys.InsertNextCell(4)
                for k in range(4):
                    polys.InsertCellPoint(start + k)
            poly = vtk.vtkPolyData()
            poly.SetPoints(_points(np.array(coords, float)))
            poly.SetPolys(polys)
            key = "column_markers" if gkey == "column" else f"column_markers_{gkey}"
            # a little see-through: the joints, line and area ends under it stay visible
            # (is the beam really snapped to the column?)
            out.actors[key] = _actor_from_polydata(poly, color, opacity=gopacity("column_plan"), lighting=False)
    # ---- plan-view walls with their thickness (like the column outlines) ------
    # a wall is an edge-on line in plan; its footprint rectangle (length x thickness)
    # is drawn at the story level.  Only in plan windows: elevations / 3-D keep the
    # wall as its mid-plane (Extrude shows the real solid everywhere).
    if base_on and story is not None and not extruded and opts.shows_area_type("Wall"):
        walls = [a for a in model.areas.values() if a.type == "Wall" and (vis_a is None or a.name in vis_a)]
        z = model.stories[story].elevation if story in model.stories else 0.0
        dz = _tube_radius(model) * 0.5
        groups = {}
        for a in walls:
            fp = wall_footprint(model, a)
            if fp is None:
                continue
            gkey, color = _group_color(model, a, "area", "wall")
            groups.setdefault(gkey, (color, []))[1].append(fp)
        for gkey, (color, fps) in groups.items():
            coords, polys = [], vtk.vtkCellArray()
            for fp in fps:
                start = len(coords)
                coords += [(x, y, z + dz) for x, y in fp]
                polys.InsertNextCell(4)
                for k in range(4):
                    polys.InsertCellPoint(start + k)
            poly = vtk.vtkPolyData()
            poly.SetPoints(_points(np.array(coords, float)))
            poly.SetPolys(polys)
            key = "wall_plan" if gkey == "wall" else f"wall_plan_{gkey}"
            out.actors[key] = _actor_from_polydata(poly, color, opacity=gopacity("wall_plan"), lighting=False)
            edges = vtk.vtkFeatureEdges()
            edges.SetInputData(poly)
            edges.BoundaryEdgesOn()
            edges.FeatureEdgesOff()
            edges.ManifoldEdgesOff()
            edges.NonManifoldEdgesOff()
            edges.Update()
            out.actors[key + "_edges"] = _actor_from_polydata(edges.GetOutput(), color,
                                                              gsize("area_edge_width"), lighting=False)
    # ---- selected areas / points --------------------------------------------
    sel_areas = [model.areas[n] for n in selected_areas if n in model.areas] if sel_on else []
    if sel_areas:
        seg = []
        for a in sel_areas:
            if story is not None and a.type == "Wall":
                # a wall is edge-on in plan: its footprint (thickness) outlined, dashed, at the story
                fp = wall_footprint(model, a)
                zs = model.stories[story].elevation if story in model.stories else 0.0
                if fp is not None:
                    C = [(x, y, zs + _tube_radius(model) * 0.6) for x, y in fp]
                    for k in range(4):
                        seg += [C[k], C[(k + 1) % 4]]
                    continue
            P = inset_polygon(np.array([pts[p].xyz for p in a.points], float), model)
            n = len(P)
            for k in range(n):
                seg += [P[k], P[(k + 1) % n]]
        out.actors["areas_sel"] = _dashed_lines_actor(model, seg, COLORS["selection"], 4.0)
    sel_pts = [pts[n] for n in selected_points if n in pts] if sel_on else []
    if sel_pts:
        # a large square in the selection colour over everything (lines, extruded solids, slabs),
        # with a contrasting core so it stands out from a selected (same colour) member
        # (a square frame with a dot in it - round marks are the pin symbols)
        out.actors["points_sel"] = _front_marks([p.xyz for p in sel_pts], COLORS["selection"], 15.0,
                                                "square", 2.0, inner=5.0)
    # ---- supports -----------------------------------------------------------
    sup = [p for p in pts.values() if any(p.restraints) and (vis_p is None or p.name in vis_p or (
        story is not None and any(f.story == story and (vis_f is None or f.name in vis_f)
                                  for f in model.frames.values() if p.name in (f.point_i, f.point_j))))
           ] if base_on else []
    if base_on and sup and opts.supports:
        from .supports import support_polydata
        size = _tube_radius(model) * 4 * gsize("support_size")
        append = vtk.vtkAppendPolyData()
        n = 0
        for p in sup:
            poly = support_polydata(p.restraints, p.xyz, size, p.local_axes)
            if poly is not None:
                append.AddInputData(poly)
                n += 1
        if n:
            append.Update()
            out.actors["supports"] = _actor_from_polydata(append.GetOutput(), COLORS["support"])
    # ---- joints, springs, releases, local axes ---------------------------------
    shown_pts = [p for p in pts.values() if vis_p is None or p.name in vis_p]
    # selected: the round mark only; a joint with a support / spring symbol: the symbol
    # marks it (a joint square on the tip of a pin cone only clutters it)
    on_sym = {p.name for p in shown_pts if any(p.restraints)} if opts.supports else set()
    if opts.springs:
        on_sym |= {p.name for p in shown_pts if any(p.springs) or any(p.rotational_springs)}
    plain_pts = [p for p in shown_pts if p.name not in selected_points and p.name not in on_sym]
    if sel_on and opts.joints and plain_pts:
        out.actors["joints"] = _joint_dots(plain_pts, _tube_radius(model) * 1.2 * gsize("joint_size"))
    if base_on and opts.springs:
        sp = [p for p in shown_pts if any(p.springs) or any(p.rotational_springs)]
        if sp:
            out.actors["springs"] = _spring_symbols(model, sp)
    vis_frames = [f for f in model.frames.values() if (vis_f is None or f.name in vis_f)
                  and opts.shows_frame_type(f.type)]
    if base_on and opts.releases:
        a = _release_dots(model, vis_frames)
        if a is not None:
            out.actors["releases"] = a
    if base_on and opts.local_axes and vis_frames:
        for k, a in enumerate(_local_axes_actors(model, vis_frames)):
            out.actors[f"local_axes_{k}"] = a
    # ---- grids ----------------------------------------------------------------
    if not base_on:
        pass
    elif plane is not None and plane[0] == "z":  # free plan: the plan grid at that elevation
        if opts.grid_lines:
            g = build_grid_actor(model, None, levels=[float(plane[1])])
            if g is not None:
                out.actors["grids"] = g
        if opts.grid_bubbles and model.grid_lines and model.stories:
            out.actors["grid_bubbles"] = build_grid_bubbles(model, None, z=float(plane[1]))
    elif plane is not None:
        if opts.grid_lines or opts.grid_bubbles:
            elev = build_elevation_grid(model, plane)
            for k, a in enumerate(elev):
                if (k == 0 and opts.grid_lines) or (k > 0 and opts.grid_bubbles):
                    out.actors[f"elev_grid_{k}"] = a
    else:
        levels = None if story is not None else grid_levels(model, opts.grid_3d)
        if opts.grid_lines:
            if story is None and levels and len(levels) > 1:
                # 3D, every level: the base grid is the reference (strong), the stories above fainter
                base = build_grid_actor(model, None, levels=levels[:1])
                upper = build_grid_actor(model, None, levels=levels[1:])
                if base is not None:
                    base.GetProperty().SetOpacity(0.85)
                    base.GetProperty().SetLineWidth(1.4)
                    out.actors["grids"] = base
                if upper is not None:
                    upper.GetProperty().SetOpacity(0.4)
                    out.actors["grids_upper"] = upper
            else:
                grid_actor = build_grid_actor(model, story, levels=levels)
                if grid_actor is not None:
                    if story is None:  # 3D: the base grid is the reference - draw it stronger
                        grid_actor.GetProperty().SetOpacity(0.85)
                        grid_actor.GetProperty().SetLineWidth(1.4)
                    out.actors["grids"] = grid_actor
            if story is None and opts.grid_vertical:
                vert = build_vertical_grid_actor(model)
                if vert is not None:
                    out.actors["grid_vertical"] = vert
        if opts.grid_bubbles and model.grid_lines and model.stories:
            out.actors["grid_bubbles"] = build_grid_bubbles(model, story, z=levels[0] if levels else None)
    # ---- reference lines (CYPE 3D) --------------------------------------------
    if base_on and getattr(opts, "reference_lines", False):
        from .snap import reference_lines_for
        from .tools import Plane as _Plane
        if plane is not None and plane[0] in ("x", "y", "z"):
            pl = _Plane({"x": "yz", "y": "xz", "z": "plan"}[plane[0]], float(plane[1]))
        elif story is not None:
            pl = _Plane("plan", model.stories[story].elevation if story in model.stories else 0.0, story)
        else:
            pl = _Plane("3d")
        refs = reference_lines_for(model, pl, (opts.reference_x, opts.reference_y, opts.reference_z))
        if refs:
            seg = [q for a, b in refs for q in (a, b)]
            ra = _dashed_lines_actor(model, seg, COLORS["reference"], 1.0)
            ra.GetProperty().SetOpacity(0.7)
            out.actors["reference_lines"] = ra
    # ---- labels ---------------------------------------------------------------
    if labels_on and (opts.frame_labels or opts.section_names):
        out.actors["labels"] = build_label_actor(model, {f.name for f in vis_frames}, opts.frame_labels,
                                                 opts.section_names)
    if labels_on and opts.joint_labels and shown_pts:
        out.actors["joint_labels"] = _text_labels(_points_of([p.xyz for p in shown_pts]),
                                                  _strings([p.label or p.name for p in shown_pts]),
                                                  COLORS["text"], _text_pt(-1))
    if labels_on and opts.area_labels:
        ar = [a for a in model.areas.values() if (vis_a is None or a.name in vis_a) and opts.shows_area_type(a.type)]
        if ar:
            cen = [np.mean([pts[q].xyz for q in a.points], axis=0) for a in ar]
            out.actors["area_labels"] = _text_labels(_points_of(cen), _strings([a.label or a.name for a in ar]),
                                                     COLORS["text"], _text_pt(-1))
    out.point_ids = list(pts)
    return out


def grid_levels(model: Model, mode: str = "base") -> List[float]:
    """Elevations at which a 3D window draws the grid: base, every level or top."""
    zs = sorted({z for _, z in model.levels()}) or sorted({s.elevation for s in model.stories.values()})
    if not zs:
        return [0.0]
    if mode == "all":
        return zs
    return [zs[-1]] if mode == "top" else [zs[0]]


def build_story_plane(model: Model, story: Optional[str]) -> List[vtk.vtkProp]:
    """Faint shaded plane (with outline and name) of ``story`` - shown in 3D windows
    so the user sees which floor the plan window is showing."""
    if not story or story not in model.stories:
        return []
    return build_view_plane(model, "plan", model.stories[story].elevation, story)


def _plan_box(model: Model):
    g = _grid_extent(model)
    if g is not None:
        xs, ys, x0, x1, y0, y1, ext = g
        return x0 - ext * 0.5, x1 + ext * 0.5, y0 - ext * 0.5, y1 + ext * 0.5
    if model.points:
        xyz = np.array([p.xyz for p in model.points.values()])
        (x0, y0), (x1, y1) = xyz[:, :2].min(0), xyz[:, :2].max(0)
        ext = max(x1 - x0, y1 - y0, 1.0) * 0.05
        return x0 - ext, x1 + ext, y0 - ext, y1 + ext
    return None


def build_view_plane(model: Model, kind: str, value: float, label: str = "") -> List[vtk.vtkProp]:
    """The plane a plan / elevation window shows, as a faint shaded rectangle with an
    outline and its name, for the 3D windows: ``kind`` "plan" (z = value), "xz"
    (an elevation at y = value) or "yz" (x = value)."""
    box = _plan_box(model)
    if box is None:
        return []
    x0, x1, y0, y1 = box
    if kind == "plan":
        z = float(value)
        corners = np.array([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], float)
    elif kind in ("xz", "yz"):
        zs = [s.elevation for s in model.stories.values()] or [p.z for p in model.points.values()] or [0.0]
        z0, z1 = min(zs), max(zs)
        dz = max(z1 - z0, 1.0) * 0.06
        z0, z1 = z0 - dz, z1 + dz
        if kind == "xz":  # vertical, along X at y = value
            y = float(value)
            corners = np.array([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], float)
        else:  # along Y at x = value
            x = float(value)
            corners = np.array([(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)], float)
    else:
        return []
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(corners))
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(4)
    for k in range(4):
        cells.InsertCellPoint(k)
    poly.SetPolys(cells)
    color = COLORS.get("story_plane", (0.2, 0.5, 0.9))
    face = _actor_from_polydata(poly, color, opacity=gopacity("view_plane"), lighting=False)
    face.GetProperty().SetBackfaceCulling(False)
    lines = vtk.vtkCellArray()
    for k in range(4):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(k)
        lines.InsertCellPoint((k + 1) % 4)
    outline = vtk.vtkPolyData()
    outline.SetPoints(_points(corners))
    outline.SetLines(lines)
    edge = _actor_from_polydata(outline, color, 2.0, opacity=0.7, lighting=False)
    out = [face, edge]
    if label:
        out.append(_text_labels(_points_of([corners[3] if kind != "plan" else corners[0]]), _strings([label]),
                                color, _text_pt(2)))
    return out


def _points_of(coords) -> vtk.vtkPoints:
    return _points(np.array(coords, float).reshape(-1, 3))


def _strings(texts) -> vtk.vtkStringArray:
    arr = vtk.vtkStringArray()
    arr.SetName("labels")  # the label hierarchy reads the array by this name
    for t in texts:
        arr.InsertNextValue(str(t))
    return arr


def _screen_squares(coords, color, size: float) -> vtk.vtkActor2D:
    """Square marks of a fixed size in pixels at world points, drawn over the
    model (an overlay like the ETABS joint squares): never hidden inside
    extruded members or slabs, whatever the zoom."""
    coords = np.array(coords, float).reshape(-1, 3)
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(coords))
    verts = vtk.vtkCellArray()
    for k in range(len(coords)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(k)
    poly.SetVerts(verts)
    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToWorld()
    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(poly)
    mapper.SetTransformCoordinate(coord)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetPointSize(size)
    prop.SetDisplayLocationToForeground()
    return actor


def _front_dots(coords, color, size: float, rank: int = 1) -> vtk.vtkActor:
    """Round marks of a fixed size in pixels drawn in front of the model (shaded
    balls, pulled to the front of the depth buffer so extruded members and slabs
    do not hide them)."""
    coords = np.array(coords, float).reshape(-1, 3)
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(coords))
    verts = vtk.vtkCellArray()
    for k in range(len(coords)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(k)
    poly.SetVerts(verts)
    actor = _actor_from_polydata(poly, color, lighting=True)
    prop = actor.GetProperty()
    prop.SetPointSize(size)
    prop.SetRenderPointsAsSpheres(True)
    prop.SetAmbient(0.55)
    prop.SetDiffuse(0.6)
    prop.SetSpecular(0.4)
    prop.SetSpecularPower(30)
    m = actor.GetMapper()
    try:  # in front of everything (the core one step more than the ring)
        m.SetResolveCoincidentTopologyToPolygonOffset()
        m.SetRelativeCoincidentTopologyPointOffsetParameter(-60000.0 * rank)
    except AttributeError:
        pass
    return actor


def _front_marks(coords, color, size: float, shape: str = "ring", line_px: float = 1.5,
                 inner: float = 0.0, rank: int = 1) -> vtk.vtkActor:
    """Outline marks of a fixed size in pixels, in front of the model: ``ring`` - a thin
    circle with a see-through middle (the pin symbol), ``square`` - a square frame
    (a selected joint), with an optional filled middle of ``inner`` pixels."""
    coords = np.array(coords, float).reshape(-1, 3)
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(coords))
    verts = vtk.vtkCellArray()
    for k in range(len(coords)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(k)
    poly.SetVerts(verts)
    actor = _actor_from_polydata(poly, color, lighting=False)
    prop = actor.GetProperty()
    prop.SetPointSize(size)
    prop.LightingOff()
    edge = max(0.05, min(0.95, 1.0 - 2.0 * line_px / size))  # inner edge of the outline (0..1)
    core = max(0.0, min(edge, inner / size))
    if shape == "ring":
        dist = "length(gl_PointCoord * 2.0 - 1.0)"
        prop.SetRenderPointsAsSpheres(False)
    else:
        dist = "max(abs(gl_PointCoord.x * 2.0 - 1.0), abs(gl_PointCoord.y * 2.0 - 1.0))"
    code = ("//VTK::Light::Impl\n"
            f"  {{ float rr = {dist};\n"
            f"    if (rr > 1.0 || (rr < {edge:.4f} && rr > {core:.4f})) discard;\n"
            "    gl_FragData[0] = vec4(diffuseColorUniform, 1.0); }\n")
    try:
        actor.GetShaderProperty().AddFragmentShaderReplacement("//VTK::Light::Impl", True, code, False)
    except AttributeError:
        pass
    m = actor.GetMapper()
    try:  # in front of everything
        m.SetResolveCoincidentTopologyToPolygonOffset()
        m.SetRelativeCoincidentTopologyPointOffsetParameter(-60000.0 * rank)
    except AttributeError:
        pass
    return actor


def _joint_dots(points, radius: float = 0.0) -> vtk.vtkActor2D:
    """Joints as small squares (ETABS), always visible."""
    return _screen_squares([p.xyz for p in points], COLORS.get("joint", COLORS["text"]), 6.0 * gsize("joint_size"))


def spring_paths(p, size: float):
    """The spring support of joint ``p`` as (polylines, plates): for every translational
    spring a zig-zag coil along that axis away from the joint ending on a ground plate
    (the same kind of block as a fixed support); for every rotational spring a spiral
    round that axis at the joint.  The joint local axes (an inclined support) turn
    the symbol with them."""
    from .supports import _rot_from_local_axes
    R = _rot_from_local_axes(getattr(p, "local_axes", None)).T  # columns: local 1, 2, 3 in global
    c = np.array(p.xyz, float)
    lines, plates = [], []
    for k in range(3):
        if abs(float(p.springs[k])) <= 0:
            continue
        d = -R[:, k]  # the coil leaves the joint downwards for UZ, towards -1 / -2 for U1 / U2
        side = R[:, (k + 1) % 3] if k != 2 else R[:, 0]
        L = size * 1.4
        lead = 0.16 * L
        pts = [c, c + d * lead]
        n = 7
        for i in range(n):
            t = lead + (L - 2 * lead) * (i + 0.5) / n
            pts.append(c + d * t + side * (0.30 * size if i % 2 == 0 else -0.30 * size))
        pts += [c + d * (L - lead), c + d * L]
        lines.append(np.array(pts))
        plates.append((c + d * (L + 0.11 * size), d, side))
    for k in range(3):
        if abs(float(p.rotational_springs[k])) <= 0:
            continue
        e, u, v = R[:, k], R[:, (k + 1) % 3], R[:, (k + 2) % 3]
        turns, m = 1.75, 40
        pts = []
        for i in range(m + 1):
            th = 2 * math.pi * turns * i / m
            r = size * (0.12 + 0.42 * i / m)
            pts.append(c + (u * math.cos(th) + v * math.sin(th)) * r)
        pts.append(pts[-1] + v * 0.25 * size)
        lines.append(np.array(pts))
    return lines, plates


def spring_segments(p, size: float) -> List[np.ndarray]:
    """The spring symbol as line segments (pairs of end points)."""
    lines, _plates = spring_paths(p, size)
    out = []
    for P in lines:
        for a, b in zip(P[:-1], P[1:]):
            out += [a, b]
    return out


def _spring_symbols(model: Model, points) -> vtk.vtkActor:
    """Spring supports (elastic) drawn like the rigid supports - solid, lit, in the
    support style: a coil (a tube) for each translational spring ending on a ground
    plate, a spiral tube for each rotational spring."""
    size = _tube_radius(model) * 4 * gsize("support_size")
    append = vtk.vtkAppendPolyData()
    n = 0
    for p in points:
        lines, plates = spring_paths(p, size)
        for P in lines:
            poly = vtk.vtkPolyData()
            poly.SetPoints(_points(np.asarray(P, float)))
            cell = vtk.vtkCellArray()
            cell.InsertNextCell(len(P))
            for i in range(len(P)):
                cell.InsertCellPoint(i)
            poly.SetLines(cell)
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(poly)
            tube.SetRadius(size * 0.055 * gsize("spring_width") / 2.0)
            tube.SetNumberOfSides(10)
            tube.CappingOn()
            tube.Update()
            append.AddInputData(tube.GetOutput())
            n += 1
        for centre, normal, side in plates:
            cube = vtk.vtkCubeSource()
            cube.SetXLength(size * 0.95)
            cube.SetYLength(size * 0.95)
            cube.SetZLength(size * 0.22)
            cube.Update()
            third = np.cross(normal, side)
            M = vtk.vtkMatrix4x4()
            for r in range(3):
                M.SetElement(r, 0, side[r])
                M.SetElement(r, 1, third[r])
                M.SetElement(r, 2, normal[r])
                M.SetElement(r, 3, centre[r])
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(cube.GetOutput())
            tf.SetTransform(tr)
            tf.Update()
            append.AddInputData(tf.GetOutput())
            n += 1
    if not n:
        poly = vtk.vtkPolyData()
        return _actor_from_polydata(poly, COLORS.get("spring", COLORS["support"]))
    append.Update()
    return _actor_from_polydata(append.GetOutput(), COLORS.get("spring", COLORS["support"]))


RELEASE_RING_AT = 0.12  # pin symbol: this fraction of the member length in from the released end


def _release_rings(points) -> vtk.vtkPropAssembly:
    ring = vtk.vtkPropAssembly()
    ring.AddPart(_front_marks(points, COLORS.get("release", (0.10, 0.65, 0.25)), 12.0 * gsize("joint_size"),
                              "ring", 3.2))
    return ring


def _release_dots(model: Model, frames) -> Optional[vtk.vtkPropAssembly]:
    """ETABS style pin symbol: a ring near each end with a moment release (a member
    pinned at its start shows one ring by its start joint), drawn over the model."""
    pts = []
    for f in frames:
        a, b = np.array(model.points[f.point_i].xyz), np.array(model.points[f.point_j].xyz)
        L = float(np.linalg.norm(b - a))
        if L <= 0:
            continue
        k = RELEASE_RING_AT  # well in from the joint: clear of the joint square / selected-joint mark
        if any(f.releases_i[3:]) or any(f.releases_i[:3]):
            pts.append(a + k * (b - a))
        if any(f.releases_j[3:]) or any(f.releases_j[:3]):
            pts.append(b + k * (a - b))
    if not pts:
        return None
    # a green circle, see-through in the middle (not a square: squares are joints)
    return _release_rings(pts)


def _local_axes_actors(model: Model, frames) -> List[vtk.vtkActor]:
    """Local axes 1, 2, 3 (Options > Graphics Colors axis_1..3) at member mid-points:
    an arrow per axis and its number placed past the arrow head, so the number
    never sits on the arrow.  Length, line, text size and the number's distance are
    Options > Graphics Sizes (local_axes_*)."""
    from ..mesh.geometry import frame_axes

    L = model_extent(model) * 0.03
    typ = typical_member_length(model)
    if typ > 0:
        L = min(L, typ * 0.15)  # local axes: not metres long on a tall building
    L *= gsize("local_axes_length")
    head = 0.22 * L  # arrow head: length along the axis, half width 0.4 x that
    label_at = L * (1.0 + 0.35 * gsize("local_axes_gap")) + 0.5 * head
    sets = ([], [], [])
    for f in frames:
        a, b = np.array(model.points[f.point_i].xyz, float), np.array(model.points[f.point_j].xyz, float)
        try:
            ax = frame_axes(a, b, f.angle)
        except ValueError:
            continue
        mid = 0.5 * (a + b)
        for k in range(3):
            side = ax[(k + 1) % 3]  # the arrow head opens in a plane through the axis
            sets[k].append((mid, ax[k], side))
    out = []
    width = gsize("local_axes_width")
    for k, segs in enumerate(sets):
        if not segs:
            continue
        color = COLORS.get(f"axis_{k + 1}", ((0.9, 0.2, 0.2), (0.2, 0.75, 0.3), (0.25, 0.45, 0.95))[k])
        coords, lines = [], vtk.vtkCellArray()
        n = 0
        for p, d, side in segs:
            tip = p + d * L
            back = tip - d * head
            for q0, q1 in ((p, tip), (tip, back + side * 0.4 * head), (tip, back - side * 0.4 * head)):
                coords += [q0, q1]
                lines.InsertNextCell(2)
                lines.InsertCellPoint(2 * n)
                lines.InsertCellPoint(2 * n + 1)
                n += 1
        poly = vtk.vtkPolyData()
        poly.SetPoints(_points_of(coords))
        poly.SetLines(lines)
        out.append(_actor_from_polydata(poly, color, width, lighting=False))
        spots = [p + d * label_at for p, d, _ in segs]
        out.append(_text_labels(_points_of(spots), _strings([str(k + 1)] * len(spots)), color,
                                max(6, int(round(gsize("local_axes_text")))), bold=True))
    return out


def _frame_lines_actor(model: Model, names: Sequence[str], color, width: float,
                       dashed: bool = False) -> vtk.vtkActor:
    coords, lines = [], vtk.vtkCellArray()
    for k, n in enumerate(names):
        f = model.frames[n]
        coords.append(model.points[f.point_i].xyz)
        coords.append(model.points[f.point_j].xyz)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2 * k)
        lines.InsertCellPoint(2 * k + 1)
    if dashed:
        return _dashed_lines_actor(model, coords, color, width)
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(np.array(coords)))
    poly.SetLines(lines)
    return _actor_from_polydata(poly, color, width, lighting=False)


def _tube_actor(model: Model, names: Sequence[str], color) -> vtk.vtkActor:
    """Frames without section dimensions in the extruded view: tubes."""
    coords, lines = [], vtk.vtkCellArray()
    for k, n in enumerate(names):
        f = model.frames[n]
        coords += [model.points[f.point_i].xyz, model.points[f.point_j].xyz]
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2 * k)
        lines.InsertCellPoint(2 * k + 1)
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(np.array(coords, float)))
    poly.SetLines(lines)
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly)
    tube.SetNumberOfSides(8)
    tube.SetRadius(_tube_radius(model))
    tube.Update()
    return _actor_from_polydata(tube.GetOutput(), color)


def _tube_radius(model: Model) -> float:
    """Base size of the 3-D symbols (supports, springs, plain tubes): a fraction of
    the model size, but never more than a fraction of a typical member, so a tall
    building does not get supports metres across.  Small models are unchanged."""
    ext = model_extent(model)
    r = ext * 0.004
    typ = typical_member_length(model)
    if typ > 0:
        r = min(r, typ * 0.02)
    return max(r, 1e-3)


def typical_member_length(model: Model) -> float:
    """Median length of the frame objects (0 if none)."""
    if not model.frames:
        return 0.0
    P = model.points
    a = np.array([P[f.point_i].xyz for f in model.frames.values()], float)
    b = np.array([P[f.point_j].xyz for f in model.frames.values()], float)
    L = np.linalg.norm(b - a, axis=1)
    L = L[L > 1e-9]
    return float(np.median(L)) if len(L) else 0.0


def model_extent(model: Model) -> float:
    if not model.points:
        return 10.0
    xyz = np.array([p.xyz for p in model.points.values()])
    return float(np.max(xyz.max(axis=0) - xyz.min(axis=0)) or 1.0)


def _grid_extent(model: Model):
    xs = [g.ordinate for g in model.grid_lines if g.direction == "X"]
    ys = [g.ordinate for g in model.grid_lines if g.direction == "Y"]
    if not xs or not ys:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    ext = max(x1 - x0, y1 - y0, 1.0) * 0.1
    return xs, ys, x0, x1, y0, y1, ext


def _bubble_radius(model: Model) -> float:
    """Radius of the grid bubbles (model units): 2 % of the grid's size x the user's factor."""
    g = _grid_extent(model)
    span = max(g[3] - g[2], g[5] - g[4], g[6] * 10) if g is not None else 10.0
    return 0.02 * span * gsize("bubble_size")


def bubble_actor(items, e1, e2, radius: float, color) -> Optional[vtk.vtkActor]:
    """CAD grid bubbles: a circle with the grid name inside, drawn in the plane
    (e1, e2) - the plan for plans and 3D, the elevation plane for elevations.
    ``items``: (centre xyz, text, leader start xyz or None)."""
    if not items:
        return None
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    app = vtk.vtkAppendPolyData()
    n = 40
    for c, text, lead in items:
        c = np.asarray(c, float)
        ring = vtk.vtkPolyData()
        P = [c + radius * (math.cos(2 * math.pi * k / n) * e1 + math.sin(2 * math.pi * k / n) * e2)
             for k in range(n)]
        ring.SetPoints(_points(np.array(P)))
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(n + 1)
        for k in range(n + 1):
            cells.InsertCellPoint(k % n)
        if lead is not None:  # a short line from the end of the grid line to the bubble
            q = np.asarray(lead, float)
            d = c - q
            L = float(np.linalg.norm(d))
            if L > radius * 1.02:
                ring.GetPoints().InsertNextPoint(*q)
                ring.GetPoints().InsertNextPoint(*(c - d / L * radius))
                cells.InsertNextCell(2)
                cells.InsertCellPoint(n)
                cells.InsertCellPoint(n + 1)
        ring.SetLines(cells)
        app.AddInputData(ring)
        # the name: polygon text, centred, as large as fits in the circle
        vt = vtk.vtkVectorText()
        vt.SetText(str(text))
        vt.Update()
        b = vt.GetOutput().GetBounds()
        w, h = max(b[1] - b[0], 1e-9), max(b[3] - b[2], 1e-9)
        sc = min(1.05 * radius / h, 1.45 * radius / w)
        cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
        out = vtk.vtkPolyData()
        out.DeepCopy(vt.GetOutput())
        pts = out.GetPoints()
        for k in range(pts.GetNumberOfPoints()):
            x, y, _ = pts.GetPoint(k)
            pts.SetPoint(k, *(c + sc * ((x - cx) * e1 + (y - cy) * e2)))
        app.AddInputData(out)
    app.Update()
    actor = _actor_from_polydata(app.GetOutput(), color, 1.6, lighting=False)
    return actor


def build_grid_bubbles(model: Model, story: Optional[str] = None, z: Optional[float] = None) -> vtk.vtkProp:
    """Grid bubbles (A, B, ... / 1, 2, ... in circles) at the line ends of the shown
    story, or at elevation ``z`` (3D windows: the base level by default).  Size and
    distance from the grid: Options > Graphics sizes."""
    g = _grid_extent(model)
    items = []
    r = _bubble_radius(model)
    if g is not None and model.stories:
        xs, ys, x0, x1, y0, y1, ext = g
        if story and story in model.stories:
            z = model.stories[story].elevation
        elif z is None:
            z = min(s.elevation for s in model.stories.values())
        off = r * (1.0 + 0.6 * gsize("bubble_gap"))
        for gl in model.grid_lines:
            if gl.direction == "X":
                end = (gl.ordinate, y1 + ext, z)
                items.append(((gl.ordinate, y1 + ext + off, z), gl.name, end))
            elif gl.direction == "Y":
                end = (x0 - ext, gl.ordinate, z)
                items.append(((x0 - ext - off, gl.ordinate, z), gl.name, end))
    actor = bubble_actor(items, (1, 0, 0), (0, 1, 0), r, COLORS["grid_text"])
    if actor is None:
        actor = vtk.vtkActor2D()
    return actor


def build_grid_actor(model: Model, story: Optional[str] = None,
                     levels: Optional[Sequence[float]] = None) -> Optional[vtk.vtkActor]:
    if not model.grid_lines or not model.stories:
        return None
    g = _grid_extent(model)
    if g is None:
        return None
    xs, ys, x0, x1, y0, y1, ext = g
    coords, lines = [], vtk.vtkCellArray()
    if story and story in model.stories:
        zs = [model.stories[story].elevation]
    elif levels is not None:
        zs = list(levels)
    else:
        zs = [s.elevation for s in model.stories.values()]
    for z in zs:
        for x in xs:
            coords += [(x, y0 - ext, z), (x, y1 + ext, z)]
        for y in ys:
            coords += [(x0 - ext, y, z), (x1 + ext, y, z)]
    for k in range(len(coords) // 2):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2 * k)
        lines.InsertCellPoint(2 * k + 1)
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(np.array(coords, float)))
    poly.SetLines(lines)
    actor = _actor_from_polydata(poly, COLORS["grid"], gsize("grid_width"), opacity=gopacity("grid_lines") * 0.83,
                                 lighting=False)
    actor.GetProperty().SetLineStipplePattern(0xF0F0)
    return actor


def build_vertical_grid_actor(model: Model) -> Optional[vtk.vtkActor]:
    """3D windows: a vertical line at every grid intersection, from the lowest to
    the highest level - the "column lines" that tie the base grid to the stories
    above (like the ETABS 3D view)."""
    g = _grid_extent(model)
    if g is None:
        return None
    zs = sorted({z for _, z in model.levels()}) or sorted({s.elevation for s in model.stories.values()})
    if len(zs) < 2:
        return None
    xs, ys = g[0], g[1]
    z0, z1 = zs[0], zs[-1]
    coords = [(x, y, z) for x in xs for y in ys for z in (z0, z1)]
    lines = vtk.vtkCellArray()
    for k in range(len(coords) // 2):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2 * k)
        lines.InsertCellPoint(2 * k + 1)
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(np.array(coords, float)))
    poly.SetLines(lines)
    return _actor_from_polydata(poly, COLORS["grid"], gsize("grid_width"), opacity=gopacity("grid_lines") * 0.58,
                                lighting=False)


def build_label_actor(model: Model, frames: Optional[set] = None, labels_on: bool = True,
                      sections: bool = False) -> vtk.vtkProp:
    """Frame labels and / or section names at member mid-points (2D text)."""
    pts = vtk.vtkPoints()
    labels = vtk.vtkStringArray()
    labels.SetName("labels")
    for f in model.frames.values():
        if frames is not None and f.name not in frames:
            continue
        a, b = model.points[f.point_i], model.points[f.point_j]
        pts.InsertNextPoint((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)
        parts = ([f.label or f.name] if labels_on else []) + ([f.section] if sections else [])
        labels.InsertNextValue(" / ".join(parts))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.GetPointData().AddArray(labels)
    hier = vtk.vtkPointSetToLabelHierarchy()
    hier.SetInputData(poly)
    hier.SetLabelArrayName("labels")
    hier.GetTextProperty().SetColor(*COLORS["text"])
    hier.GetTextProperty().SetFontSize(_text_pt(1))
    mapper = vtk.vtkLabelPlacementMapper()
    mapper.SetInputConnection(hier.GetOutputPort())
    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    return actor


# ----------------------------------------------------------------------------
# Results overlays
# ----------------------------------------------------------------------------
def build_deformed_actor(model: Model, results, case: str, scale: Optional[float] = None,
                         mode: Optional[int] = None, frames: Optional[set] = None,
                         areas: Optional[set] = None) -> Tuple[vtk.vtkActor, float]:
    """Deformed shape as scaled straight lines (frames) and polygons (areas)."""
    disp = _displacement_field(model, results, case, mode)
    ext = model_extent(model)
    dmax = float(np.max(np.abs(disp[:, :3]))) if len(disp) else 0.0
    if scale is None:
        scale = (0.08 * ext / dmax) if dmax > 0 else 1.0
    names = list(model.points)
    index = {n: k for k, n in enumerate(names)}
    xyz = np.array([model.points[n].xyz for n in names]) + scale * disp[:, :3]
    lines = vtk.vtkCellArray()
    for f in model.frames.values():
        if frames is not None and f.name not in frames:
            continue
        lines.InsertNextCell(2)
        lines.InsertCellPoint(index[f.point_i])
        lines.InsertCellPoint(index[f.point_j])
    polys = vtk.vtkCellArray()
    for a in model.areas.values():
        if areas is not None and a.name not in areas:
            continue
        polys.InsertNextCell(len(a.points))
        for p in a.points:
            polys.InsertCellPoint(index[p])
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(xyz))
    poly.SetLines(lines)
    poly.SetPolys(polys)
    actor = _actor_from_polydata(poly, COLORS["deformed"], gsize("deformed_width"), opacity=0.9, lighting=False)
    actor.GetProperty().SetRepresentationToWireframe()
    return actor, scale


# ---------------------------------------------------------------------------
# Deformed / mode shape: what depends only on the analysis mesh and on the
# objects a window shows (connectivity, frame sample points, Hermite basis) is
# prepared once and reused; a new case / mode only evaluates arrays.
# ---------------------------------------------------------------------------
class _MeshArrays:
    """Read-only arrays of an analysis mesh for drawing (one per mesh)."""

    _serial = 0

    def __init__(self, am):
        _MeshArrays._serial += 1
        self.serial = _MeshArrays._serial  # cache key part (an id() can be reused)
        self.X = am.coords().astype(float) if am.nodes else np.zeros((0, 3))
        sh = am.shells
        self.sh_len = np.array([len(s.nodes) for s in sh], dtype=np.int64)
        self.sh_conn = np.array([n for s in sh for n in s.nodes], dtype=np.int64)
        self.sh_off = np.concatenate([[0], np.cumsum(self.sh_len)]).astype(np.int64)
        self.sh_obj = [s.object for s in sh]
        fr = am.frames
        self.fr_i = np.array([f.node_i for f in fr], dtype=np.int64)
        self.fr_j = np.array([f.node_j for f in fr], dtype=np.int64)
        self.fr_obj = [f.object for f in fr]
        self.fr_hinged = np.array([bool(np.any(f.releases_i[3:]) or np.any(f.releases_j[3:])) for f in fr], bool)
        # every node of the drawn elements: the colour range is the whole model's
        self.every = np.unique(np.concatenate([self.sh_conn, self.fr_i, self.fr_j]).astype(np.int64))
        self.by_obj: Dict[str, List[int]] = {}
        for k, f in enumerate(fr):
            self.by_obj.setdefault(f.object, []).append(k)


_MESH_CACHE: "OrderedDict" = None
_PLAN_CACHE: "OrderedDict" = None
PLAN_CACHE_ITEMS = 12  # deformed-shape plans kept (a few windows x a few view sets) ...
PLAN_CACHE_BYTES = 96 * 2 ** 20  # ... within this many bytes (a whole 21-storey model: ~15 MB)


def _mesh_arrays(am) -> _MeshArrays:
    global _MESH_CACHE
    import weakref
    from collections import OrderedDict
    if _MESH_CACHE is None:
        _MESH_CACHE = OrderedDict()
    hit = _MESH_CACHE.get(id(am))
    if hit is not None and hit[0]() is am:
        return hit[1]
    arrays = _MeshArrays(am)
    _MESH_CACHE[id(am)] = (weakref.ref(am), arrays)
    while len(_MESH_CACHE) > 2:  # the current analysis (and the one before)
        _MESH_CACHE.popitem(last=False)
    return arrays


def _set_key(names):
    return None if names is None else frozenset(names)


class _DeformedPlan:
    """Topology and frame sampling of a deformed-shape view: the shells and the
    frame elements of the objects shown, the used nodes, the sample fractions
    along each frame element and their cubic Hermite (or linear, hinged) basis."""

    def __init__(self, model: Model, am, M: _MeshArrays, Lf: float, frames, areas, shells: bool, releases: bool):
        self.ext = model_extent(model)
        self.X0 = M.X / Lf
        if shells:
            sh = np.array([k for k, o in enumerate(M.sh_obj) if areas is None or o in areas], dtype=np.int64)
        else:
            sh = np.zeros(0, dtype=np.int64)
        fr = np.array([k for k, o in enumerate(M.fr_obj) if frames is None or o in frames], dtype=np.int64)
        self.sh, self.fr = sh, fr
        pick = np.zeros(len(M.sh_len), bool)
        pick[sh] = True
        sh_nodes = M.sh_conn[np.repeat(pick, M.sh_len)]  # in element order (sh ascending)
        self.used = np.unique(np.concatenate([sh_nodes, M.fr_i[fr], M.fr_j[fr]]).astype(np.int64))
        # shells: connectivity in the used-node numbering
        self.sh_conn = np.searchsorted(self.used, sh_nodes).astype(np.int64)
        self.sh_off = np.concatenate([[0], np.cumsum(M.sh_len[sh])]).astype(np.int64)
        # frames: elements of positive length, sampled every ``ext / 80`` (at most 10 parts)
        I, J = M.fr_i[fr], M.fr_j[fr]
        A, B = self.X0[I], self.X0[J]
        L = np.linalg.norm(B - A, axis=1)
        ok = L > 0
        self.fe = fr[ok]  # analysis frame element index of each drawn member piece
        self.I, self.J, A, B, L = I[ok], J[ok], A[ok], B[ok], L[ok]
        self.L = L
        self.E = (B - A) / L[:, None] if len(L) else np.zeros((0, 3))
        self.hinged = M.fr_hinged[self.fe]
        seg = max(self.ext / 80.0, 1e-9)
        n = np.minimum(10, np.maximum(1, np.ceil(L / seg))).astype(np.int64)
        self.n = n
        self.el = np.repeat(np.arange(len(L), dtype=np.int64), n + 1)
        start = np.concatenate([[0], np.cumsum(n + 1)[:-1]]).astype(np.int64)
        self.start = start
        self.x = (np.arange(len(self.el), dtype=np.int64) - start[self.el]) / n[self.el] if len(L) else np.zeros(0)
        self.H = self._basis(self.x, self.hinged[self.el])
        self.base = A[self.el] + self.x[:, None] * (B - A)[self.el]
        # pin symbols: same place along the member as undeformed, riding on the curve
        self.ring_el = np.zeros(0, np.int64)
        self.ring_x = np.zeros(0)
        if releases and len(self.fe):
            pos = {int(e): k for k, e in enumerate(self.fe)}
            shown = set(int(k) for k in fr)
            rel, rx = [], []
            for name in dict.fromkeys(M.fr_obj[k] for k in fr):
                obj = model.frames.get(name)
                if obj is None:
                    continue
                els = [k for k in M.by_obj.get(name, []) if k in shown]
                ends = [t for t, r in ((RELEASE_RING_AT, obj.releases_i), (1.0 - RELEASE_RING_AT, obj.releases_j))
                        if any(r[1:3]) or any(r[3:])]
                for t in ends:
                    k = next((k for k in els if am.frames[k].rel_start - 1e-9 <= t <= am.frames[k].rel_end + 1e-9),
                             None)
                    if k is None or k not in pos:
                        continue
                    f = am.frames[k]
                    span = max(f.rel_end - f.rel_start, 1e-12)
                    rel.append(pos[k])
                    rx.append(min(max((t - f.rel_start) / span, 0.0), 1.0))
            self.ring_el = np.array(rel, dtype=np.int64)
            self.ring_x = np.array(rx, float)
        self.ring_H = self._basis(self.ring_x, self.hinged[self.ring_el])
        self.ring_base = (self.X0[self.I[self.ring_el]] + self.ring_x[:, None]
                          * (self.X0[self.J[self.ring_el]] - self.X0[self.I[self.ring_el]]))
        for a in vars(self).values():
            if isinstance(a, np.ndarray):
                a.setflags(write=False)
        self.nbytes = sum(a.nbytes for a in vars(self).values() if isinstance(a, np.ndarray))

    @staticmethod
    def _basis(x, hinged):
        x2, x3 = x * x, x ** 3
        H = np.stack([1 - 3 * x2 + 2 * x3, x - 2 * x2 + x3, 3 * x2 - 2 * x3, -x2 + x3], axis=1) if len(x) \
            else np.zeros((0, 4))
        if np.any(hinged):
            xh = x[hinged]
            H[hinged] = np.stack([1 - xh, np.zeros_like(xh), xh, np.zeros_like(xh)], axis=1)
        return H

    def deflection(self, D: np.ndarray, el: np.ndarray, x: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Displacement of the member pieces ``el`` at fractions ``x`` (cubic Hermite
        from the end translations and rotations; straight between hinges)."""
        if not len(el):
            return np.zeros((0, 3))
        E, L = self.E, self.L
        di, dj = D[self.I, :3], D[self.J, :3]
        ri = np.cross(D[self.I, 3:6], E) * L[:, None]
        rj = np.cross(D[self.J, 3:6], E) * L[:, None]
        ri[self.hinged] = 0.0
        rj[self.hinged] = 0.0
        ai = np.einsum("ij,ij->i", di, E)
        aj = np.einsum("ij,ij->i", dj, E)
        ti, tj = di - ai[:, None] * E, dj - aj[:, None] * E
        return (((1 - x) * ai[el] + x * aj[el])[:, None] * E[el] + H[:, 0:1] * ti[el] + H[:, 1:2] * ri[el]
                + H[:, 2:3] * tj[el] + H[:, 3:4] * rj[el])


def _deformed_plan(model: Model, results, frames, areas, shells: bool, releases: bool) -> _DeformedPlan:
    global _PLAN_CACHE
    from collections import OrderedDict
    if _PLAN_CACHE is None:
        _PLAN_CACHE = OrderedDict()
    am = results.am
    M = _mesh_arrays(am)
    Lf = results.U.factor("length")
    key = (M.serial, Lf, model_extent(model), _set_key(frames), _set_key(areas), bool(shells), bool(releases))
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        _PLAN_CACHE.move_to_end(key)
        return plan
    plan = _DeformedPlan(model, am, M, Lf, frames, areas, shells, releases)
    _PLAN_CACHE[key] = plan
    while len(_PLAN_CACHE) > 1 and (len(_PLAN_CACHE) > PLAN_CACHE_ITEMS
                                    or sum(p.nbytes for p in _PLAN_CACHE.values()) > PLAN_CACHE_BYTES):
        _PLAN_CACHE.popitem(last=False)  # least recently used first
    return plan


def _cells(offsets: np.ndarray, conn: np.ndarray) -> vtk.vtkCellArray:
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_support.numpy_to_vtkIdTypeArray(np.ascontiguousarray(offsets, dtype=np.int64), deep=True),
                  numpy_support.numpy_to_vtkIdTypeArray(np.ascontiguousarray(conn, dtype=np.int64), deep=True))
    return cells


def _float_scalars(values, name: str):
    arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(np.asarray(values, float), dtype=np.float32), deep=True)
    arr.SetName(name)
    return arr


def build_deformed_mesh_actors(model: Model, results, case: str, mode: Optional[int] = None,
                               frames: Optional[set] = None, areas: Optional[set] = None,
                               shells: bool = True, extruded: bool = False,
                               releases: bool = False) -> Tuple[List[vtk.vtkProp], float]:
    """Deformed shape of the analysis mesh coloured by the displacement (ETABS
    style): shell elements as surfaces, frame elements as lines, with a colour
    scale.  A slab sagging between its beams shows, not only the object corners.
    Same scale as :func:`build_deformed_actor`.  Returns (actors, scale)."""
    am = results.am
    if mode is not None:
        D = np.asarray(results.mode_shape(case, mode, normalize="max"), float)
        title = f"Mode {mode}\ndisplacement"
        fac = 1.0
    else:
        D = np.asarray(results.displacement_envelope(case).max, float)
        fac, unit = display_length(model)
        title = f"Displacement\n[{unit}]"
    if not len(D):
        return [], 1.0
    plan = _deformed_plan(model, results, frames, areas, shells, releases)
    M = _mesh_arrays(am)
    U3 = D[:, :3]
    mag = np.linalg.norm(U3, axis=1)
    ext = plan.ext
    dmax = float(np.max(np.abs(U3)))
    scale = (0.08 * ext / dmax) if dmax > 0 else 1.0
    used = plan.used
    if not len(used):
        return [], scale
    X = plan.X0 + scale * U3
    vals = mag[used] * fac
    # one colour scale in every window (a plan of one story and the 3-D view):
    # the range of the whole model, not of the objects this window shows
    vmax = float(mag[M.every].max()) * fac if len(M.every) else float(vals.max())
    # frames: cubic (Hermite) deflected shape from the end translations and
    # rotations, so members bend and tilt with the joints (straight if released)
    d = plan.deflection(D, plan.el, plan.x, plan.H)
    FP = plan.base + scale * d
    FV = np.linalg.norm(d, axis=1) * fac
    ring_pts = plan.ring_base + scale * plan.deflection(D, plan.ring_el, plan.ring_x, plan.ring_H) \
        if len(plan.ring_el) else np.zeros((0, 3))
    if len(FV):
        vmax = max(vmax, float(FV.max()))
    lut, (vmin, vmax), step = result_lut(0.0, vmax or 1e-12)
    actors: List[vtk.vtkProp] = []
    line_off = np.concatenate([[0], np.cumsum(plan.n + 1)]).astype(np.int64)
    if extruded:  # real sections bending with the joints, shells with their thickness
        from .result_view import ScalarMesh, frame_rings, mapped_actor, thick_shell, thick_shells, tube_rings
        sm = ScalarMesh()
        if len(plan.sh) and np.all(M.sh_len[plan.sh] == 4):
            nodes = M.sh_conn.reshape(-1, 4)[plan.sh] if np.all(M.sh_len == 4) else \
                np.array([am.shells[int(k)].nodes for k in plan.sh], np.int64)
            thick_shells(sm, model, [M.sh_obj[int(k)] for k in plan.sh], X[nodes], mag[nodes] * fac)
        else:
            for k in plan.sh:
                s = am.shells[int(k)]
                thick_shell(sm, model, s.object, X[list(s.nodes)], mag[list(s.nodes)] * fac)
        r = _tube_radius(model) * 0.9
        ring_cache = {}  # the pieces of one member share its section and axes
        for k, e in enumerate(plan.fe):
            f = am.frames[int(e)]
            line = slice(line_off[k], line_off[k + 1])
            axes = np.asarray(f.axes, float)
            key = (f.object, axes.tobytes())
            rings = ring_cache.get(key)
            if rings is None:
                obj = model.frames.get(f.object)
                rings = frame_rings(model, obj, axes=axes) if obj is not None else None
                if rings is None:
                    rings = tube_rings(r, axes)
                ring_cache[key] = rings
            sm.sweep(FP[line], FV[line], rings)
        if sm.pts:
            actors.append(mapped_actor(sm.polydata(), lut))
    for kind in (() if extruded else ("shells", "frames")):
        poly = vtk.vtkPolyData()
        if kind == "shells":
            if not len(plan.sh):
                continue
            poly.SetPoints(_points(X[used]))
            poly.SetPolys(_cells(plan.sh_off, plan.sh_conn))
            poly.GetPointData().SetScalars(_float_scalars(vals, "U"))
        else:
            if not len(plan.fe):
                continue
            # members as coloured screen-space lines (fixed pixel width, as in the
            # undeformed view) - not tubes: a tube radius tied to the model size made
            # every member a thick pipe on a tall building and hundreds of thousands
            # of triangles.  Real section shapes are the extruded view's job.
            poly.SetPoints(_points(FP))
            poly.SetLines(_cells(line_off, np.arange(len(FP), dtype=np.int64)))
            poly.GetPointData().SetScalars(_float_scalars(FV, "U"))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(vmin, vmax)
        mapper.SetInterpolateScalarsBeforeMapping(True)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        if kind == "shells":
            prop.SetOpacity(0.92)
        else:
            prop.SetLineWidth(gsize("deformed_width"))
            prop.LightingOff()
        actors.append(actor)
    if len(ring_pts):
        actors.append(_release_rings(list(ring_pts)))
    actors.append(legend_bar(lut, title, vmin, vmax, step))
    return actors, scale


def build_mesh_actors(model: Model, am, areas: Optional[set] = None, frames: Optional[set] = None,
                      offset: float = 0.0) -> List[vtk.vtkProp]:
    """The analysis mesh (ETABS "show analysis model"): shell element edges, and
    the nodes where frames / joints are tied into the shells as dots - so one can
    see that a column, a loaded joint or a beam really is part of the slab mesh."""
    if am is None:
        return []
    Lf = model.units.factor("length")
    X = np.array([n.xyz for n in am.nodes], float) / Lf
    shells = [s for s in am.shells if areas is None or s.object in areas]
    if not shells:
        return []
    edges = set()
    for s in shells:
        for k in range(4):
            a, b = s.nodes[k], s.nodes[(k + 1) % 4]
            edges.add((min(a, b), max(a, b)))
    used = sorted({n for e in edges for n in e})
    idx = {n: i for i, n in enumerate(used)}
    shift = np.zeros(3)
    if offset:
        shift[2] = offset
    pts = vtk.vtkPoints()
    for n in used:
        pts.InsertNextPoint(*(X[n] + shift))
    cells = vtk.vtkCellArray()
    for a, b in edges:
        cells.InsertNextCell(2)
        cells.InsertCellPoint(idx[a])
        cells.InsertCellPoint(idx[b])
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*COLORS["mesh"])
    prop.SetLineWidth(1.0)
    prop.SetLighting(False)
    out: List[vtk.vtkProp] = [actor]
    # tie points: shell nodes shared with frame elements or joints (columns, loads...)
    shell_nodes = set(used)
    tied = {n for f in am.frames for n in (f.node_i, f.node_j)} | {nd.id for nd in am.nodes if nd.point is not None}
    tie = sorted(shell_nodes & tied)
    if tie:
        tp = vtk.vtkPoints()
        for n in tie:
            tp.InsertNextPoint(*(X[n] + shift))
        tpoly = vtk.vtkPolyData()
        tpoly.SetPoints(tp)
        verts = vtk.vtkCellArray()
        for i in range(len(tie)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        tpoly.SetVerts(verts)
        m2 = vtk.vtkPolyDataMapper()
        m2.SetInputData(tpoly)
        a2 = vtk.vtkActor()
        a2.SetMapper(m2)
        a2.GetProperty().SetColor(0.85, 0.15, 0.15)
        a2.GetProperty().SetPointSize(5)
        a2.GetProperty().RenderPointsAsSpheresOn()
        a2.GetProperty().SetLighting(False)
        out.append(a2)
    return out


def _displacement_field(model: Model, results, case: str, mode: Optional[int]) -> np.ndarray:
    names = list(model.points)
    if mode is not None:
        phi = results.mode_shape(case, mode, normalize="max")
        return np.array([phi[results.am.point_to_node[n]] for n in names])
    env = results.displacement_envelope(case)
    return np.array([env.max[results.am.point_to_node[n]] for n in names])


def _plan_side(nv, v1) -> np.ndarray:
    """The side of a member, in a view with normal ``nv``, a positive value is
    drawn to when the diagram is turned into the view: the same for every member
    (whichever way it was drawn) - below a horizontal one and right of a vertical
    one, as if the drawing were turned so the member reads left to right
    (Options > Plan Diagrams: Positive Below / Right; off: above / left)."""
    nv = np.asarray(nv, float)
    # screen axes of the view: right and up
    up = np.array([0.0, 1.0, 0.0]) if abs(nv[2]) > 0.5 else np.array([0.0, 0.0, 1.0])
    right = np.cross(up, nv)
    right /= np.linalg.norm(right) or 1.0
    up = np.cross(nv, right)
    x, y = float(v1 @ right), float(v1 @ up)
    if abs(x) >= abs(y) - 1e-9:  # read left to right
        if x < 0:
            x, y = -x, -y
    elif y < 0:  # read bottom to top
        x, y = -x, -y
    side = y * right - x * up  # the member direction turned clockwise
    side /= np.linalg.norm(side) or 1.0
    return side if goption("plan_positive_below") else -side


def build_diagram_actors(model: Model, results, case: str, component: str = "M3",
                         scale: Optional[float] = None, frames: Optional[set] = None,
                         view_normal=None, labels: bool = True) -> Tuple[List[vtk.vtkActor], float]:
    """Member force diagrams drawn perpendicular to each member in the plane
    of the component (M3/V2 in the local 1-2 plane, M2/V3 in 1-3, P/T along 2).

    ``view_normal`` (a plan or elevation window): a diagram whose plane is seen
    edge-on (beam moments in a plan) is turned into the view plane, so it shows.
    ``labels``: the values at the member ends and at the largest value, and a
    caption with the component, its unit and the extremes of the window."""
    comp_index = {"P": 0, "V2": 1, "V3": 2, "T": 3, "M2": 4, "M3": 5}[component]
    ext = model_extent(model)
    series = []
    vmax = 0.0
    for f in model.frames.values():
        if frames is not None and f.name not in frames:
            continue
        st = results.frame_forces(f.name, case)
        vals = st.forces[:, comp_index]
        vmax = max(vmax, float(np.max(np.abs(vals))) if len(vals) else 0.0)
        if st.forces_min is not None:
            vmax = max(vmax, float(np.max(np.abs(st.forces_min[:, comp_index]))))
        series.append((f, st))
    if scale is None:
        scale = (0.12 * ext / vmax) if vmax > 0 else 1.0
    pos_blocks, neg_blocks = [], []  # (n, 4, 3) quads of each member, in member order
    lab_pts, lab_txt, lab_pri, ext_vals = [], [], [], []
    from ..display_units import DisplayUnits
    _du = DisplayUnits.of(model)
    _item = "moment" if component in ("T", "M2", "M3") else "force"
    k_disp, u_disp = _du.factor(_item), _du.label(_item)  # labels in the display units
    for f, st in series:
        a = np.array(model.points[f.point_i].xyz)
        b = np.array(model.points[f.point_j].xyz)
        elems = results.am.object_to_frames.get(f.name, [])
        if not elems:
            continue
        axes = results.am.frames[elems[0]].axes
        # plot direction: moments about 3 / shear 2 -> local 2 ; moments about 2 / shear 3 -> local 3
        direction = axes[2] if component in ("M2", "V3") else axes[1]
        if component in ("M2", "M3") and goption("moment_tension_side"):
            # tension side (the usual engineering drawing): positive M3 below a beam,
            # negative above; off = the moment value plotted along the local axis
            direction = -direction
        L = np.linalg.norm(b - a)
        v1 = (b - a) / L
        if view_normal is not None:
            nv = np.asarray(view_normal, float)
            if abs(float(direction @ nv)) > 0.7:  # seen edge-on: turn into the view plane
                direction = _plan_side(nv, v1)
        def quads(xa, xb, va, vb):
            """Quads (n, 4, 3) between stations xa - xb with values va - vb."""
            p0 = a + v1 * xa[:, None]
            p1 = a + v1 * xb[:, None]
            return np.stack([p0, p0 + direction * va[:, None] * scale, p1 + direction * vb[:, None] * scale, p1],
                            axis=1)

        x = np.asarray(st.station, float)
        if st.forces_min is None:
            # one case: coloured by sign (positive / negative), split where the diagram crosses zero
            vals = np.asarray(st.forces[:, comp_index], float)
            if len(x) > 1:
                va, vb = vals[:-1], vals[1:]
                xa_, xb_ = x[:-1], x[1:]
                cross = va * vb < 0
                with np.errstate(invalid="ignore", divide="ignore"):
                    xm = xa_ + (xb_ - xa_) * va / (va - vb)
                # segments in order; a crossing one becomes two (to and from the zero)
                n2 = np.where(cross, 2, 1)
                seg = np.repeat(np.arange(len(va)), n2)
                second = np.zeros(len(seg), bool)
                second[1:] = (seg[1:] == seg[:-1])
                c = cross[seg]
                sxa = np.where(c & second, xm[seg], xa_[seg])
                sxb = np.where(c & ~second, xm[seg], xb_[seg])
                sva = np.where(c & second, 0.0, va[seg])
                svb = np.where(c & ~second, 0.0, vb[seg])
                q = quads(sxa, sxb, sva, svb)
                pos = (sva + svb) >= 0
                if np.any(pos):
                    pos_blocks.append(q[pos])
                if not np.all(pos):
                    neg_blocks.append(q[~pos])
        else:  # envelope: maximum and minimum curves in the two colours
            for arr, blocks in ((st.forces, pos_blocks), (st.forces_min, neg_blocks)):
                vals = np.asarray(arr[:, comp_index], float)
                if len(x) > 1:
                    blocks.append(quads(x[:-1], x[1:], vals[:-1], vals[1:]))
        if labels and len(x):
            # the maximum positive and the maximum negative value of the member are always
            # labelled (an envelope: from its maximum and its minimum curves), plus the
            # end values where they are not zero (a pin shows none)
            hi = np.asarray(st.forces[:, comp_index], float)
            lo = np.asarray(st.forces_min[:, comp_index], float) if st.forces_min is not None else hi
            k_pos = int(np.argmax(hi)) if float(hi.max()) > 0 else None
            k_neg = int(np.argmin(lo)) if float(lo.min()) < 0 else None
            ref = max(abs(float(hi.max())), abs(float(lo.min())), 1e-30)
            kinds = {}
            if k_pos is not None:
                kinds.setdefault(k_pos, []).append(float(hi[k_pos]))
            if k_neg is not None:
                kinds.setdefault(k_neg, []).append(float(lo[k_neg]))
            ends = (0, len(x) - 1)
            for k in ends:
                if st.forces_min is None:
                    vals_k = [float(hi[k])]
                else:
                    vals_k = [float(hi[k]), float(lo[k])]
                for v in vals_k:
                    if abs(v) >= 0.1 * ref and k not in kinds:
                        kinds.setdefault(k, []).append(v)
                    elif abs(v) >= 0.1 * ref and v not in kinds.get(k, []):
                        kinds[k].append(v)
            for k, vs in kinds.items():
                for v in dict.fromkeys(vs):  # distinct values at a station
                    if abs(v) < 1e-9 * max(vmax, 1e-30):
                        continue
                    ext = (k == k_pos and abs(v - float(hi[k])) < 1e-12) or (k == k_neg and abs(v - float(lo[k])) < 1e-12)
                    lab_pts.append(a + v1 * float(x[k]) + direction * v * scale)
                    lab_txt.append(fmt_value(v * k_disp))
                    lab_pri.append((2.0 if ext else 1.0) + abs(v) / max(vmax, 1e-30))
                    ext_vals.append(v)
    actors = []
    for blocks, color in ((pos_blocks, COLORS["moment_pos"]), (neg_blocks, COLORS["moment_neg"])):
        Q = np.concatenate(blocks).reshape(-1, 3) if blocks else np.zeros((0, 3))
        if not len(Q):
            continue
        nq = len(Q) // 4
        poly = vtk.vtkPolyData()
        poly.SetPoints(_points(Q))
        poly.SetPolys(_cells(np.arange(0, 4 * nq + 1, 4, dtype=np.int64), np.arange(4 * nq, dtype=np.int64)))
        actor = _actor_from_polydata(poly, color, 1.0, opacity=gopacity("diagram_fill"), lighting=False)
        actors.append(actor)
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(poly)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        actors.append(_actor_from_polydata(edges.GetOutput(), color, 1.5, lighting=False))
    for act in actors:  # a beam diagram turned into a plan lies on the slab: draw it in front
        act.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -8.0)
    if labels and lab_pts:
        pts = vtk.vtkPoints()
        texts = vtk.vtkStringArray()
        texts.SetName("labels")
        pri = vtk.vtkFloatArray()
        pri.SetName("priority")
        for p, t, w in zip(lab_pts, lab_txt, lab_pri):
            pts.InsertNextPoint(*map(float, p))
            texts.InsertNextValue(t)
            pri.InsertNextValue(float(w))
        lab = _text_labels(pts, texts, COLORS["text"], _text_pt(0), background=True, priority=pri)
        actors.append(lab)
    if labels and series:
        unit = u_disp
        hi = max(ext_vals) * k_disp if ext_vals else 0.0
        lo = min(ext_vals) * k_disp if ext_vals else 0.0
        cap = vtk.vtkTextActor()
        cap.SetInput(f"{component} [{unit}]   max {fmt_value(hi)}   min {fmt_value(lo)}")
        tp = cap.GetTextProperty()
        tp.SetFontSize(13)
        tp.SetBold(True)
        tp.SetColor(*COLORS["text"])
        tp.SetJustificationToRight()
        cap.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        cap.SetPosition(0.97, 0.03)
        actors.append(cap)
    return actors, scale


# ----------------------------------------------------------------------------
# Off-screen rendering helper (tests, reports)
# ----------------------------------------------------------------------------
def render_to_png(actors: Sequence[vtk.vtkProp], path: str, size=(1200, 800), view: str = "3d",
                  background=True) -> str:
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(*size)
    ren = vtk.vtkRenderer()
    if background:
        ren.GradientBackgroundOn()
        ren.SetBackground(*COLORS["background_bottom"])
        ren.SetBackground2(*COLORS["background_top"])
    rw.AddRenderer(ren)
    for a in actors:
        ren.AddActor(a)
    set_view(ren, view)
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()
    wr = vtk.vtkPNGWriter()
    wr.SetFileName(path)
    wr.SetInputConnection(w2i.GetOutputPort())
    wr.Write()
    return path


def set_view(ren: vtk.vtkRenderer, view: str = "3d"):
    """Camera presets: 3d, plan, xz (looking +Y), yz (looking -X)."""
    cam = ren.GetActiveCamera()
    if view == "plan":
        cam.SetPosition(0, 0, 1)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 1, 0)
        cam.ParallelProjectionOn()
    elif view == "xz":
        cam.SetPosition(0, -1, 0)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 0, 1)
        cam.ParallelProjectionOn()
    elif view == "yz":
        cam.SetPosition(1, 0, 0)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 0, 1)
        cam.ParallelProjectionOn()
    else:
        cam.SetPosition(-1.0, -1.6, 0.9)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 0, 1)
        cam.ParallelProjectionOff()
    ren.ResetCamera()
    cam.Zoom(1.15)


# ----------------------------------------------------------------------------
# Shell contours, reactions and load display
# ----------------------------------------------------------------------------
def build_contour_actors(model: Model, results, case: str, component: str = "M11",
                         deformed: bool = False, areas: Optional[set] = None,
                         extruded: bool = False) -> Tuple[List[vtk.vtkProp], Tuple[float, float]]:
    """Filled contour of one shell stress resultant on all shell elements plus a
    scalar bar; ``extruded`` draws each element with its thickness.
    Returns (actors, (vmin, vmax))."""
    vals, nodes = results.shell_node_values(case, component)
    vals = vals * _shell_display(model, component)[1]  # in the display units
    # the slab inside the columns (rigid zones) is left out: what is shown, and the
    # range of the colour scale, are the values up to the column faces
    rigid = results.rigid_zone_shells() if hasattr(results, "rigid_zone_shells") else set()
    am = results.am
    # element ids in the order of ``vals`` (the shells that have results)
    if hasattr(results, "_shell_stack"):
        have = set(results._shell_stack(case)[0].tolist())
    else:
        have = set(results.shell_element_envelope(case))
    elems = [s for s in am.shells if s.id in have]
    if rigid and len(vals):
        keep0 = [i for i, se in enumerate(elems) if se.id not in rigid]
        vals, nodes = vals[keep0], nodes[keep0]
        elems = [elems[i] for i in keep0]
    # one colour scale in every window: the range over the whole model
    gmin, gmax = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 0.0)
    objs = None
    if (areas is not None or extruded) and len(vals):
        keep = [i for i, se in enumerate(elems) if areas is None or se.object in areas]
        vals, nodes = vals[keep], nodes[keep]
        objs = [elems[i].object for i in keep]
    if len(vals) == 0:
        return [], (0.0, 0.0)
    # unshared points per element so each element keeps its own nodal values
    Lf = results.U.factor("length")  # analysis nodes are SI
    if deformed:
        disp, scale = _deformed_positions(model, results, case)
        X = disp
    else:
        X = am.coords() / Lf
    nodes = np.asarray(nodes, int)
    coords = X[nodes.ravel()]
    scalars = numpy_support.numpy_to_vtk(np.ascontiguousarray(np.asarray(vals, float).ravel(), dtype=np.float32),
                                         deep=True)
    scalars.SetName(component)
    n_el = len(vals)
    conn = np.arange(4 * n_el, dtype=np.int64)
    offs = np.arange(0, 4 * n_el + 1, 4, dtype=np.int64)
    polys = vtk.vtkCellArray()
    polys.SetData(numpy_support.numpy_to_vtkIdTypeArray(offs, deep=True),
                  numpy_support.numpy_to_vtkIdTypeArray(conn, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(coords))
    poly.SetPolys(polys)
    poly.GetPointData().SetScalars(scalars)
    if extruded and objs is not None:
        from .result_view import ScalarMesh, thick_shells
        sm = ScalarMesh()
        C = np.array(coords, float).reshape(-1, 4, 3)
        thick_shells(sm, model, objs, C, np.asarray(vals, float).reshape(-1, 4))
        poly = sm.polydata()
    lut, (vmin, vmax), step = result_lut(gmin, gmax)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(vmin, vmax)
    mapper.SetInterpolateScalarsBeforeMapping(True)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.95)
    if extruded:  # true colours on every face of the slab, whatever the light angle
        actor.GetProperty().SetAmbient(0.45)
        actor.GetProperty().SetDiffuse(0.6)
    bar = legend_bar(lut, f"{component}\n[{_shell_unit(model, component)}]", vmin, vmax, step)
    return [actor, bar], (vmin, vmax)


def nice_scale(vmin: float, vmax: float, target: int = 10) -> Tuple[float, float, float, int]:
    """Round colour-scale limits: (lo, hi, step, intervals) with the step 1, 2, 2.5
    or 5 x 10^k, lo/hi multiples of it and about ``target`` (at most 14) intervals,
    so the legend reads 0, 0.25, 0.5 ... instead of 0.000263, 0.000527 ..."""
    vmin, vmax = float(vmin), float(vmax)
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        return 0.0, 1.0, 0.1, 10
    if vmax - vmin <= 1e-12 * max(1.0, abs(vmin), abs(vmax)):
        pad = abs(vmax) * 0.1 or 1.0
        vmin, vmax = vmin - pad, vmax + pad
    raw = (vmax - vmin) / max(1, target)
    p = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = m * p
        lo = math.floor(vmin / step + 1e-9) * step
        hi = math.ceil(vmax / step - 1e-9) * step
        n = int(round((hi - lo) / step))
        if n <= 14:
            break
    return lo, hi, step, max(1, n)


def _decimals(step: float) -> int:
    for d in range(0, 8):
        if abs(round(step, d) - step) <= 1e-9 * max(1.0, abs(step)):
            return d
    return 8


def result_lut(vmin: float, vmax: float, target: int = 10):
    """Banded blue -> red lookup table on round limits.  Returns (lut, (lo, hi), step)."""
    lo, hi, step, n = nice_scale(vmin, vmax, target)
    if n > 10 and n % 2:  # the legend labels every second band edge: make that land on hi
        hi += step
        n += 1
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetHueRange(0.667, 0.0)  # blue (min) -> red (max)
    lut.SetRange(lo, hi)
    lut.Build()
    return lut, (lo, hi), step


def legend_bar(lut, title: str, lo: float, hi: float, step: float):
    """The scalar bar for :func:`result_lut`: a label at every band edge (every
    second one when there are many), with as many decimals as the step needs."""
    n = int(round((hi - lo) / step))
    every = 1 if n <= 10 else 2
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle(title)
    bar.SetNumberOfLabels(n // every + 1)
    bar.SetLabelFormat(f"%-.{_decimals(step * every)}f")
    bar.GetTitleTextProperty().SetColor(*COLORS["text"])
    bar.GetLabelTextProperty().SetColor(*COLORS["text"])
    _place_bar(bar)
    return bar


def display_length(model: Model) -> Tuple[float, str]:
    """Deflections in the "Displacements" display unit (Options > Display Units),
    else mm (or inches) whatever the model length unit: (factor from model length,
    unit label)."""
    from ..display_units import DisplayUnits
    du = DisplayUnits.of(model)
    if "displacement" in du.chosen:
        return du.factor("displacement"), du.label("displacement")
    u = model.units.length
    return {"m": (1000.0, "mm"), "cm": (10.0, "mm"), "mm": (1.0, "mm"),
            "ft": (12.0, "in"), "in": (1.0, "in")}.get(u, (1.0, u))


def _place_bar(bar) -> None:
    """Colour scale at the right, below the ViewCube, title clear of the labels."""
    bar.SetWidth(0.08)
    bar.SetHeight(0.55)
    bar.SetPosition(0.875, 0.1)  # room for the title ("Displacement", "M11 [kN-m/m]")
    bar.SetVerticalTitleSeparation(6)
    bar.SetTextPad(4)
    tp = bar.GetTitleTextProperty()
    tp.SetFontSize(14)
    tp.BoldOn()
    tp.ItalicOff()
    tp.ShadowOff()
    lp = bar.GetLabelTextProperty()
    lp.ItalicOff()
    lp.BoldOff()
    lp.ShadowOff()
    bar.SetUnconstrainedFontSize(True)
    lp.SetFontSize(12)
    # readable over the model if the window is narrow or zoomed in
    bar.DrawBackgroundOn()
    bp = bar.GetBackgroundProperty()
    bp.SetColor(*COLORS.get("background_top", (1.0, 1.0, 1.0)))
    bp.SetOpacity(0.85)


def _shell_unit(model: Model, component: str) -> str:
    return _shell_display(model, component)[0]


def _shell_display(model: Model, component: str) -> Tuple[str, float]:
    """(unit label, factor from model units) of a shell resultant: M11.. per length
    from the moment unit, F11.. and V13.. from the force unit (display units)."""
    from ..display_units import DisplayUnits
    return DisplayUnits.of(model).per_length("moment" if component.startswith("M") else "force")


def _deformed_positions(model: Model, results, case: str):
    """Node positions of the deformed shape (same scaling as build_deformed_actor)."""
    am = results.am
    env = results.displacement_envelope(case).max
    ext = model_extent(model)
    dmax = float(np.max(np.abs(env[:, :3]))) if len(env) else 0.0
    scale = (0.05 * ext / dmax) if dmax > 0 else 1.0
    Lf = results.U.factor("length")
    pos = _mesh_arrays(am).X / Lf + scale * np.asarray(env, float)[:, :3]  # row i = analysis node i
    return pos, scale


def build_reaction_actors(model: Model, results, case: str, labels: bool = True,
                          points: Optional[set] = None) -> List[vtk.vtkProp]:
    """Reaction arrows (forces) at restrained joints with value labels."""
    rows = results.joint_reactions(case)
    if points is not None:
        rows = [r for r in rows if r.get("Point") in points]
    if not rows:
        return []
    ext = model_extent(model)
    fmax = max(max(abs(r.get("FX", 0.0)), abs(r.get("FY", 0.0)), abs(r.get("FZ", 0.0))) for r in rows) or 1.0
    L = 0.12 * ext
    append = vtk.vtkAppendPolyData()
    pts = vtk.vtkPoints()
    texts = vtk.vtkStringArray()
    texts.SetName("labels")

    from ..display_units import DisplayUnits
    du = DisplayUnits.of(model)
    k = du.factor("force")  # labels in the display units

    def fmt(v: float) -> str:  # v in display units
        return "0" if abs(v) < 1e-6 * fmax * k else f"{v:.4g}"

    for r in rows:
        p = model.points.get(r.get("Point") or r.get("Joint") or "")
        if p is None:
            continue
        for comp, d in (("FX", (1, 0, 0)), ("FY", (0, 1, 0)), ("FZ", (0, 0, 1))):
            v = float(r.get(comp, 0.0))
            if abs(v) < 1e-9 * fmax:
                continue
            length = L * abs(v) / fmax
            arrow = vtk.vtkArrowSource()
            arrow.SetTipLength(0.3)
            arrow.SetTipRadius(0.08)
            arrow.SetShaftRadius(0.03)
            tf = vtk.vtkTransform()
            sgn = 1.0 if v > 0 else -1.0
            # arrow points in the reaction direction, tail at the joint
            tf.Translate(p.x, p.y, p.z)
            if comp == "FX":
                if sgn < 0:
                    tf.RotateZ(180)
            elif comp == "FY":
                tf.RotateZ(90 * sgn)
            else:
                tf.RotateY(-90 * sgn)
            tf.Scale(length, length, length)
            tpd = vtk.vtkTransformFilter() if hasattr(vtk, "vtkTransformFilter") else vtk.vtkTransformPolyDataFilter()
            tpd.SetTransform(tf)
            tpd.SetInputConnection(arrow.GetOutputPort())
            tpd.Update()
            append.AddInputData(tpd.GetOutput())
        if labels:
            pts.InsertNextPoint(p.x, p.y, p.z)
            texts.InsertNextValue(f"{fmt(r.get('FX', 0) * k)}, {fmt(r.get('FY', 0) * k)}, "
                                  f"{fmt(r.get('FZ', 0) * k)} {du.label('force')}")
    out: List[vtk.vtkProp] = []
    if append.GetNumberOfInputConnections(0):
        append.Update()
        out.append(_actor_from_polydata(append.GetOutput(), COLORS["support"]))
    if labels and pts.GetNumberOfPoints():
        out.append(_text_labels(pts, texts, COLORS["support"], _text_pt(0), background=True))
    return out


def _text_labels(pts: vtk.vtkPoints, texts: vtk.vtkStringArray, color, size: int = 11,
                 background: bool = False, bold: bool = False, priority=None) -> vtk.vtkProp:
    """Screen-aligned labels that never overlap (a crowded label is dropped
    until there is room).  ``background`` puts each label on a filled
    rounded box in the window colour, so numbers stay readable over lines,
    fills and arrows (the ETABS / CYPE load display)."""
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.GetPointData().AddArray(texts)
    hier = vtk.vtkPointSetToLabelHierarchy()
    hier.SetInputData(poly)
    hier.SetLabelArrayName("labels")
    if priority is not None:  # the most important labels win where they crowd
        poly.GetPointData().AddArray(priority)
        hier.SetPriorityArrayName(priority.GetName())
    tp = hier.GetTextProperty()
    tp.SetColor(*color)
    tp.SetFontSize(size)
    tp.SetBold(bold or background)
    mapper = vtk.vtkLabelPlacementMapper()
    mapper.SetInputConnection(hier.GetOutputPort())
    # every label that fits is placed, whatever the zoom: the default level-of-detail
    # traversal of the label octree dropped most of them once the view was zoomed in
    # (a parallel projection zooms by scale, not by moving the camera closer)
    mapper.SetIteratorType(vtk.vtkLabelHierarchy.FULL_SORT)
    mapper.SetMaximumLabelFraction(1.0)
    if background:
        mapper.SetShapeToRoundedRect()
        mapper.SetStyleToFilled()
        mapper.SetBackgroundColor(*COLORS["background_top"])
        mapper.SetBackgroundOpacity(gopacity("label_box"))
        mapper.SetMargin(3)
    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    return actor


LOAD_KINDS = ("joint", "frame", "area", "panel")
LOAD_KIND_LABELS = {"joint": "Joint loads", "frame": "Frame loads", "area": "Area loads (panels on floors / walls)",
                    "panel": "Free load panels", "applied": "Panel and area loads as applied to the members"}


def build_panel_actors(model: Model, selected: Optional[set] = None) -> List[vtk.vtkProp]:
    """Load panels as objects: a dashed outline, a light fill and the span arrow
    (one-way) or a cross (two-way) at the centre."""
    # area loads (panels made from an area) are drawn as loads, not as objects -
    # only a selected one gets its outline
    panels = [pl for pl in getattr(model, "panel_loads", {}).values()
              if not pl.area or (selected and pl.name in selected)]
    if not panels:
        return []
    color = COLORS["load"]
    seg = []
    fill_coords, polys = [], vtk.vtkCellArray()
    ext = model_extent(model)
    for pl in panels:
        P = np.asarray(pl.points, float)
        n = len(P)
        if n < 3:
            continue
        for k in range(n):
            seg += [P[k], P[(k + 1) % n]]
        start = len(fill_coords)
        fill_coords += [tuple(p) for p in P]
        polys.InsertNextCell(n)
        for k in range(n):
            polys.InsertCellPoint(start + k)
        c = P.mean(axis=0)
        size = min(0.25 * float(np.max(P.max(axis=0) - P.min(axis=0))), 0.08 * ext)
        if pl.distribution == "one_way" and pl.span_dir is not None:
            d = np.asarray(pl.span_dir, float)
            d /= np.linalg.norm(d) or 1.0
            a, b = c - d * size, c + d * size
            seg += [a, b]
            from .supports import _rot_from_local_axes  # noqa: F401 (keeps import local)
            try:
                from ..mesh.panel_loads import polygon_frame
                _, _, _, nrm = polygon_frame(P)
            except ValueError:
                nrm = np.array([0, 0, 1.0])
            perp = np.cross(nrm, d)
            for tip in (a, b):
                dd = (c - tip)
                dd /= np.linalg.norm(dd) or 1.0
                seg += [tip, tip + dd * size * 0.3 + perp * size * 0.15, tip, tip + dd * size * 0.3 - perp * size * 0.15]
        elif pl.distribution == "two_way":
            try:
                from ..mesh.panel_loads import polygon_frame
                _, e1, e2, _ = polygon_frame(P)
            except ValueError:
                e1, e2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
            for d in (e1, e2, (e1 + e2) / math.sqrt(2), (e1 - e2) / math.sqrt(2)):
                seg += [c - d * size, c + d * size]
    out: List[vtk.vtkProp] = []
    if fill_coords:
        poly = vtk.vtkPolyData()
        poly.SetPoints(_points(np.array(fill_coords, float)))
        poly.SetPolys(polys)
        out.append(_actor_from_polydata(poly, color, opacity=0.12))
    if seg:
        out.append(_dashed_lines_actor(model, seg, color, 2.0))
    if selected:
        sel = []
        for pl in panels:
            if pl.name in selected:
                P = np.asarray(pl.points, float)
                for k in range(len(P)):
                    sel += [P[k], P[(k + 1) % len(P)]]
        if sel:
            out.append(_dashed_lines_actor(model, sel, COLORS["selection"], 4.0))
    return out


VIEW_NORMALS = {"plan": (0.0, 0.0, 1.0), "xz": (0.0, 1.0, 0.0), "yz": (1.0, 0.0, 0.0)}


def fmt_value(v: float) -> str:
    """Load value for a label: 3-4 significant digits, no exponent for ordinary
    magnitudes, no trailing zeros ("12.5", "-30", "0.75", "1.25e+04")."""
    v = float(v)
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e5:
        digits = max(0, 3 - int(math.floor(math.log10(a))))
        txt = f"{v:.{digits}f}"
        if "." in txt:
            txt = txt.rstrip("0").rstrip(".")
        return txt
    return f"{v:.3g}"


def _unit_text(model: Model, quantity: str) -> str:
    try:
        u = model.units.shown(quantity)
    except Exception:
        return ""
    return u.replace("^2", "²").replace("^3", "³")


def _panel_in_view(P: np.ndarray, view: str, level: Optional[float], tol: float) -> bool:
    if view not in VIEW_NORMALS or level is None:
        return True
    k = {"plan": 2, "xz": 1, "yz": 0}[view]
    return bool(np.all(np.abs(P[:, k] - level) <= tol))


def build_load_actors(model: Model, pattern: str, frames: Optional[set] = None, areas: Optional[set] = None,
                      points: Optional[set] = None, kinds: Optional[set] = None, view: str = "3d",
                      level: Optional[float] = None) -> List[vtk.vtkProp]:
    """Assigned loads of one pattern, drawn the way ETABS / CYPE 3D do:

    - frame distributed loads: a filled load diagram (trapezoid) with arrows
      pointing at the member and one label "12.5 kN/m" at the diagram's edge;
    - frame point loads and joint loads: a bold arrow with "30 kN";
    - area / panel loads: a few arrows and one label "2.5 kN/m²" at the centre.

    ``view`` ("plan", "xz", "yz" or "3d") is the window the loads are drawn in.
    A load along the line of sight (gravity in a plan) cannot be seen as an
    arrow: frame diagrams are turned into the view plane (as ETABS does in plan),
    and point / joint loads get the CAD symbol for a force into the screen
    (circle with a cross) or out of it (circle with a dot).  ``level`` with a 2D
    ``view`` keeps only the free load panels lying in that plane.

    ``kinds`` (subset of LOAD_KINDS + "applied") selects what is drawn: "panel"
    draws the panel loads on their panels, "applied" draws them as the line
    loads they become on the members (the CYPE 3D display after distribution)."""
    kinds = set(LOAD_KINDS) if kinds is None else set(kinds)
    ext = model_extent(model)
    L = 0.06 * ext * gsize("load_size")
    nview = np.array(VIEW_NORMALS[view], float) if view in VIEW_NORMALS else None
    # values in the display units (Options > Display Units: e.g. kgf/m, tonf, kgf/m^2)
    from ..display_units import DisplayUnits
    du = DisplayUnits.of(model)
    pretty = lambda t: t.replace("^2", "²").replace("^3", "³")  # noqa: E731
    u_line, u_force, u_area, u_moment = (pretty(du.label(i)) for i in ("line_load", "force", "area_load", "moment"))
    kL, kF, kA, kM = (du.factor(i) for i in ("line_load", "force", "area_load", "moment"))
    lines_coords, lines = [], vtk.vtkCellArray()
    head_coords, heads = [], vtk.vtkCellArray()  # solid arrow heads
    fill_coords, fills = [], vtk.vtkCellArray()  # translucent load diagrams
    pts = vtk.vtkPoints()
    texts = vtk.vtkStringArray()
    texts.SetName("labels")

    def seg(a, b):
        k = len(lines_coords)
        lines_coords.extend([tuple(a), tuple(b)])
        lines.InsertNextCell(2)
        lines.InsertCellPoint(k)
        lines.InsertCellPoint(k + 1)

    def poly_into(coords, cells, ring):
        k = len(coords)
        coords.extend(tuple(p) for p in ring)
        cells.InsertNextCell(len(ring))
        for j in range(len(ring)):
            cells.InsertCellPoint(k + j)

    def label(at, text):
        pts.InsertNextPoint(*np.asarray(at, float))
        texts.InsertNextValue(text)

    def side_of(d):
        """A unit vector perpendicular to ``d``, in the view plane when there is one."""
        if nview is not None:
            perp = np.cross(d, nview)
        else:
            perp = np.cross(d, (0, 0, 1))
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(d, (0, 1, 0))
        n = np.linalg.norm(perp)
        return perp / n if n > 1e-9 else np.array([1.0, 0, 0])

    def along_sight(d) -> bool:
        return nview is not None and abs(float(np.dot(d, nview))) > 0.9

    def arrow(tip, direction, length, head: float = 0.22):
        d = np.asarray(direction, float)
        n = np.linalg.norm(d)
        if n < 1e-12 or length <= 0:
            return
        d = d / n
        tip = np.asarray(tip, float)
        h = min(head * L, 0.5 * length)
        seg(tip - d * length, tip - d * h * 0.8)
        perp = side_of(d)
        poly_into(head_coords, heads, [tip, tip - d * h + perp * h * 0.42, tip - d * h - perp * h * 0.42])

    def sight_symbol(at, sign_into: bool, radius: float):
        """CAD symbol of a force along the line of sight: circle + cross (into
        the screen) or circle + dot (towards the viewer)."""
        at = np.asarray(at, float)
        e1 = side_of(nview)
        e2 = np.cross(nview, e1)
        ring = [at + radius * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0, 2 * math.pi, 25)]
        for a, b in zip(ring[:-1], ring[1:]):
            seg(a, b)
        if sign_into:
            r = radius * 0.7
            seg(at - r * (e1 + e2) / math.sqrt(2), at + r * (e1 + e2) / math.sqrt(2))
            seg(at - r * (e1 - e2) / math.sqrt(2), at + r * (e1 - e2) / math.sqrt(2))
        else:
            dot = [at + radius * 0.25 * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0, 2 * math.pi, 9)[:-1]]
            poly_into(head_coords, heads, dot)

    def point_force(at, force, text):
        """A concentrated force at ``at`` (arrow, or the sight symbol) and its label."""
        f = np.asarray(force, float)
        n = np.linalg.norm(f)
        if n < 1e-12:
            return
        d = f / n
        if along_sight(d):
            r = 0.26 * L
            sight_symbol(at, float(np.dot(d, nview)) < 0, r)
            label(np.asarray(at) + side_of(nview) * r * 1.6 + np.cross(nview, side_of(nview)) * r * 1.6, text)
        else:
            arrow(at, d, L)
            label(np.asarray(at) - d * L * 1.1, text)

    def direction_vector(direction: str, axes=None):
        dirs = {"gravity": (0, 0, -1), "x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "-x": (-1, 0, 0),
                "-y": (0, -1, 0)}
        key = direction.lower().replace(" ", "")
        if key.startswith("vec:"):
            v = np.array([float(x) for x in key[4:].split(",")], float)
            return v / (np.linalg.norm(v) or 1.0)
        if key.endswith("projected"):
            key = key[: -len("projected")]
        if key in dirs:
            return np.array(dirs[key], float)
        if axes is not None and key in ("local1", "1"):
            return axes[0]
        if axes is not None and key in ("local2", "2"):
            return axes[1]
        if axes is not None and key in ("local3", "3"):
            return axes[2]
        return np.array((0, 0, -1), float)

    # ---- joint loads --------------------------------------------------------------
    for p in model.points.values():
        if "joint" not in kinds or (points is not None and p.name not in points):
            continue
        for ld in p.loads:
            if ld.pattern != pattern:
                continue
            f = np.asarray(ld.force, float)
            m = np.asarray(getattr(ld, "moment", (0, 0, 0)), float)
            parts = [f"F{ax}={fmt_value(v * kF)}" for ax, v in zip("xyz", f) if v]
            if len(parts) == 1 and np.count_nonzero(f) == 1:
                parts = [f"{fmt_value(abs(float(f[np.nonzero(f)[0][0]])) * kF)} {u_force}"]
            elif parts:
                parts[-1] += f" {u_force}"
            mparts = [f"M{ax}={fmt_value(v * kM)}" for ax, v in zip("xyz", m) if v]
            if mparts:
                mparts[-1] += f" {u_moment}"
            text = "  ".join(parts + mparts)
            if np.linalg.norm(f) > 0:
                point_force((p.x, p.y, p.z), f, text)
            elif mparts:
                label((p.x, p.y, p.z), text)

    # ---- frame loads --------------------------------------------------------------
    from ..mesh.geometry import frame_axes
    derived: Dict[str, list] = {}
    if "applied" in kinds and getattr(model, "panel_loads", None):
        try:
            from ..mesh.panel_loads import panel_loads_to_members
            derived, _, _ = panel_loads_to_members(model)
        except Exception as exc:  # display must not fail on a bad panel
            print("panel load display:", exc)
    for fr in model.frames.values():
        if frames is not None and fr.name not in frames:
            continue
        a = np.array(model.points[fr.point_i].xyz)
        b = np.array(model.points[fr.point_j].xyz)
        length = float(np.linalg.norm(b - a))
        if length < 1e-12:
            continue
        axes = frame_axes(a, b, fr.angle)
        loads = (list(fr.loads) if "frame" in kinds else []) + derived.get(fr.name, [])
        dl = [ld for ld in derived.get(fr.name, []) if ld.pattern == pattern]
        stacked = 0  # several distributed loads on one member: stack the diagrams
        for ld in loads:
            if ld.pattern != pattern or getattr(ld, "load_type", "Force") != "Force":
                continue
            d = direction_vector(ld.direction, axes)
            rel = (lambda x: x / length) if getattr(ld, "absolute", False) else (lambda x: x)
            if ld.kind == "point":
                v = float(ld.value_a)
                if v:
                    x = a + (b - a) * min(max(rel(float(ld.rel_a)), 0.0), 1.0)
                    point_force(x, d * np.sign(v), f"{fmt_value(abs(v) * kF)} {u_force}")
                continue
            if ld.kind != "distributed":
                continue
            ra, rb = rel(float(ld.rel_a)), rel(float(ld.rel_b))
            va, vb = float(ld.value_a), float(ld.value_b)
            vmax = max(abs(va), abs(vb))
            if vmax == 0:
                continue
            # the side the diagram is drawn on: against the load (arrows hit the
            # member); along the line of sight, turned into the view plane
            sgn = 1.0 if (va + vb) >= 0 else -1.0
            if along_sight(d):
                push = -side_of((b - a) / length)  # arrows "push" towards the member
            else:
                push = d * sgn
            off = push * stacked * L * 0.15
            H = L * 0.8
            n = int(max(2, min(10, round(length * abs(rb - ra) / (0.45 * L)))))
            base, top = [], []
            for k in range(n + 1):
                t = ra + (rb - ra) * k / n
                v = va + (vb - va) * k / n
                x = a + (b - a) * t - off
                h = H * abs(v) / vmax
                base.append(x)
                top.append(x - push * h)
                if abs(v) > 1e-12 * vmax:
                    arrow(x, push, h)
            poly_into(fill_coords, fills, base + top[::-1])
            for p0, p1 in zip(top[:-1], top[1:]):
                seg(p0, p1)
            seg(base[0], top[0])
            seg(base[-1], top[-1])
            stacked += 1
            if ld in dl:
                continue  # panel loads on the member are labelled once, below
            txt = (f"{fmt_value(va * kL)} {u_line}" if va == vb
                   else f"{fmt_value(va * kL)} → {fmt_value(vb * kL)} {u_line}")
            mid = 0.5 * (base[0] + base[-1]) - push * H * 1.25
            label(mid, txt)
        if dl:  # one label per member for the panel loads: the peak intensity
            peak = max(max(abs(ld.value_a), abs(ld.value_b)) for ld in dl)
            label(a + (b - a) * 0.5, f"max {fmt_value(peak * kL)} {u_line}")

    # ---- area and panel loads -------------------------------------------------------
    from ..mesh.panel_loads import panel_direction, polygon_frame, point_in_poly2
    tol = max(model.options.merge_tolerance, 1e-6) * 10
    for pl in getattr(model, "panel_loads", {}).values():
        # an area load (panel made from an area) shows with the area loads, a free panel as a panel
        if ("area" if pl.area else "panel") not in kinds:
            continue
        if pl.area and areas is not None and pl.area not in areas:
            continue
        if pl.area and pl.area in model.areas:
            P = np.array([model.points[n].xyz for n in model.areas[pl.area].points], float)
        else:
            P = np.asarray(pl.points, float)
        if len(P) < 3:
            continue
        if not pl.area and not _panel_in_view(P, view, level, tol):
            continue
        try:
            o, e1, e2, _ = polygon_frame(P)
        except ValueError:
            continue
        P2 = np.array([[(p - o) @ e1, (p - o) @ e2] for p in P])
        c2 = P2.mean(axis=0)
        c = o + c2[0] * e1 + c2[1] * e2
        for case in pl.loads:
            if case.pattern != pattern:
                continue
            try:
                d, _ = panel_direction(P, case.direction)
            except ValueError:
                d = np.array([0, 0, -1.0])
            v = case.value if not case.varying else 0.5 * (case.value_lower + case.value_upper)
            if v == 0:
                continue
            dd = d * np.sign(v)
            if not along_sight(dd):
                # a few arrows over the panel (interior points of a 4 x 4 grid)
                lo2, hi2 = P2.min(axis=0), P2.max(axis=0)
                g = 4
                for i in range(1, g):
                    for j in range(1, g):
                        q2 = lo2 + (hi2 - lo2) * (i / g, j / g)
                        if point_in_poly2(q2, P2, 1e-6 * ext):
                            arrow(o + q2[0] * e1 + q2[1] * e2, dd, L * 0.6)
                at = c - dd * L * 0.75
            else:
                at = c
            value = (f"{fmt_value(abs(case.value) * kA)} {u_area}" if not case.varying else
                     f"{fmt_value(case.value_lower * kA)} → {fmt_value(case.value_upper * kA)} {u_area}")
            dist = {"one_way": "1-way", "two_way": "2-way"}.get(pl.distribution, "")
            if pl.area and pl.distribution == "shells":
                label(at, value)  # an ordinary area load: just its value
            else:
                label(at, f"{pl.name}: {value}" + (f" ({dist})" if dist else ""))
    for ar in model.areas.values():  # legacy area loads (files older than schema 2.1)
        if "area" not in kinds or (areas is not None and ar.name not in areas):
            continue
        xyz = np.array([model.points[n].xyz for n in ar.points])
        c = xyz.mean(axis=0)
        for ld in ar.loads:
            if ld.pattern != pattern:
                continue
            d = direction_vector(ld.direction)
            v = float(ld.value)
            if v == 0:
                continue
            dd = d * np.sign(v)
            if not along_sight(dd):
                for n in list(ar.points):
                    q = np.array(model.points[n].xyz)
                    arrow(c + (q - c) * 0.6, dd, L * 0.6)
                arrow(c, dd, L * 0.6)
                label(c - dd * L * 0.75, f"{fmt_value(abs(v) * kA)} {u_area}")
            else:
                label(c, f"{fmt_value(abs(v) * kA)} {u_area}")

    out: List[vtk.vtkProp] = []
    color = COLORS["load"]
    if fill_coords:
        poly = vtk.vtkPolyData()
        poly.SetPoints(_points(np.array(fill_coords, float)))
        poly.SetPolys(fills)
        out.append(_actor_from_polydata(poly, color, opacity=gopacity("load_fill"), lighting=False))
    if lines_coords:
        poly = vtk.vtkPolyData()
        poly.SetPoints(_points(np.array(lines_coords, float)))
        poly.SetLines(lines)
        out.append(_actor_from_polydata(poly, color, 2.0, lighting=False))
    if head_coords:
        poly = vtk.vtkPolyData()
        poly.SetPoints(_points(np.array(head_coords, float)))
        poly.SetPolys(heads)
        out.append(_actor_from_polydata(poly, color, lighting=False))
    if pts.GetNumberOfPoints():
        out.append(_text_labels(pts, texts, color, int(round(gsize("load_text"))), background=True))
    return out
