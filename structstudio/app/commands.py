"""Model editing commands behind the ETABS-style menus (Edit, Select, Assign).

Every function works on a :class:`~.document.Document` (one undoable edit) or a
:class:`~structstudio.model.Model` and has no Qt dependency, so the menu
behaviour is unit-tested without a window.  The names follow the ETABS menu
commands they implement (e.g. *Edit > Merge Joints*, *Edit > Edit Frames >
Join Frames*, *Assign > Copy Assigns*).
"""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..model import Model, ModelError
from ..model.objects import Area, Frame, Group, GridLine, Point
from .document import Document
from ..model.area_panels import copy_area_panels
from ..model.frame_ops import joined_loads, reverse_frame, turn_frame, turn_loads
from ..model.point_ops import (clone_assignments, has_assignments, merge_points, plan_map, resolve_merge,
                               turn_assignments, turns_directions)

XYZ = Tuple[float, float, float]
Selection = Tuple[List[str], List[str], List[str]]  # frames, areas, points
FREE6 = (False,) * 6


# ------------------------------------------------------------------ helpers
def _sel_points(model: Model, frames: Iterable[str], areas: Iterable[str], points: Iterable[str]) -> List[str]:
    """All points used by the selection (selected joints first, in order)."""
    out: List[str] = []
    seen = set()
    for n in list(points):
        if n in model.points and n not in seen:
            out.append(n)
            seen.add(n)
    for n in frames:
        f = model.frames.get(n)
        if f is not None:
            for p in (f.point_i, f.point_j):
                if p not in seen:
                    out.append(p)
                    seen.add(p)
    for n in areas:
        a = model.areas.get(n)
        if a is not None:
            for p in a.points:
                if p not in seen:
                    out.append(p)
                    seen.add(p)
    return out


def _frame_kwargs(f: Frame) -> dict:
    """Everything a copied frame keeps (assignments, not geometry or identity)."""
    kw = {}
    for fl in dataclasses.fields(Frame):
        if fl.name in ("name", "point_i", "point_j", "section", "type", "story", "loads", "label", "group"):
            continue
        kw[fl.name] = copy.deepcopy(getattr(f, fl.name))
    return kw


def _area_kwargs(a: Area) -> dict:
    kw = {}
    for fl in dataclasses.fields(Area):
        if fl.name in ("name", "points", "section", "type", "story", "loads", "label", "group"):
            continue
        kw[fl.name] = copy.deepcopy(getattr(a, fl.name))
    return kw


def _restory(model: Model) -> None:
    tol = model.options.merge_tolerance
    for p in model.points.values():
        s = model.story_at(p.z, tol)
        p.story = s.name if s else p.story
    for f in model.frames.values():
        s = model.story_at(max(model.points[f.point_i].z, model.points[f.point_j].z), tol)
        f.story = s.name if s else f.story
    for a in model.areas.values():
        s = model.story_at(max(model.points[p].z for p in a.points), tol)
        a.story = s.name if s else a.story


def _unused_points(model: Model) -> List[str]:
    used = set()
    for f in model.frames.values():
        used |= {f.point_i, f.point_j}
    for a in model.areas.values():
        used |= set(a.points)
    for s in getattr(model, "line_springs", {}).values():
        used |= {s.point_i, s.point_j}
    return [n for n, p in model.points.items()
            if n not in used and not any(p.restraints) and not p.loads and not any(p.springs)
            and not any(p.mass)]


# ------------------------------------------------------- copy / paste (Edit)
@dataclass
class Clipboard:
    """Geometry and properties of copied objects (ETABS Cut / Copy / Paste:
    objects come with their properties; Paste asks for the offset).  It keeps the
    units it was copied in and the definitions the objects use (sections,
    materials, load patterns, diaphragms), so it pastes into another model or
    after a change of units (function audit FA10)."""

    points: Dict[str, Point] = field(default_factory=dict)
    frames: List[Frame] = field(default_factory=list)
    areas: List[Area] = field(default_factory=list)
    panels: Dict[str, list] = field(default_factory=dict)  # area name -> its area-load panels
    units: object = None  # UnitSystem of the copied values
    definitions: Dict[str, dict] = field(default_factory=dict)  # table -> {name: definition}

    @property
    def empty(self) -> bool:
        return not (self.points or self.frames or self.areas)

    def in_units(self, units) -> "Clipboard":
        """The same objects expressed in ``units``."""
        if self.units is None or units is None or self.units == units:
            return self
        from ..model.base import convert_dataclass
        src = self.units

        def cv(o):
            return convert_dataclass(o, src, units)
        out = Clipboard(points={k: cv(v) for k, v in self.points.items()}, frames=[cv(f) for f in self.frames],
                        areas=[cv(a) for a in self.areas], units=units)
        r = src.factor("length") / units.factor("length")
        for k, lst in self.panels.items():
            new = []
            for pl in lst:
                q = cv(pl)
                q.points = [tuple(v * r for v in pt) for pt in getattr(q, "points", [])]
                new.append(q)
            out.panels[k] = new
        out.definitions = {t: {n: cv(d) for n, d in table.items()} for t, table in self.definitions.items()}
        return out


DEFINITION_TABLES = ("materials", "frame_sections", "shell_sections", "load_patterns", "diaphragms")


def _definitions(model: Model, frames: Sequence[Frame], areas: Sequence[Area], points: Sequence[Point]) -> Dict[str, dict]:
    """Every definition the copied objects refer to."""
    out: Dict[str, dict] = {t: {} for t in DEFINITION_TABLES}

    def take(table: str, name):
        src = getattr(model, table, {})
        if name and name in src and name not in out[table]:
            out[table][name] = copy.deepcopy(src[name])
    for f in frames:
        take("frame_sections", f.section)
        sec = model.frame_sections.get(f.section)
        if sec is not None:
            take("materials", sec.material)
        for ld in f.loads:
            take("load_patterns", ld.pattern)
    for a in areas:
        take("shell_sections", a.section)
        sec = model.shell_sections.get(a.section) if a.section else None
        if sec is not None:
            take("materials", sec.material)
        take("diaphragms", a.diaphragm)
        for ld in a.loads:
            take("load_patterns", ld.pattern)
    for p in points:
        take("diaphragms", p.diaphragm)
        for ld in list(p.loads) + list(p.ground_displacements):
            take("load_patterns", ld.pattern)
    for t in ("frame_sections", "shell_sections"):
        for d in list(out[t].values()):
            take("materials", getattr(d, "material", None))
    return out


def copy_objects(model: Model, frames: Sequence[str], areas: Sequence[str], points: Sequence[str]) -> Clipboard:
    clip = Clipboard(units=model.units)
    for n in _sel_points(model, frames, areas, points):
        clip.points[n] = copy.deepcopy(model.points[n])
    clip.frames = [copy.deepcopy(model.frames[n]) for n in frames if n in model.frames]
    clip.areas = [copy.deepcopy(model.areas[n]) for n in areas if n in model.areas]
    for a in clip.areas:
        clip.panels[a.name] = [copy.deepcopy(p) for p in model.panel_loads.values() if p.area == a.name]
    clip.definitions = _definitions(model, clip.frames, clip.areas, list(clip.points.values()))
    for lst in clip.panels.values():
        for pl in lst:
            for ld in getattr(pl, "loads", None) or []:
                pat = getattr(ld, "pattern", None)
                if pat and pat in model.load_patterns:
                    clip.definitions["load_patterns"].setdefault(pat, copy.deepcopy(model.load_patterns[pat]))
    return clip


def _missing_definitions(model: Model, clip: Clipboard) -> Tuple[Dict[str, List[str]], List[str]]:
    """(definitions the paste must add: table -> names, names that are needed but
    not in the clipboard either)."""
    add: Dict[str, List[str]] = {t: [] for t in DEFINITION_TABLES}
    lost: List[str] = []

    def need(table: str, name, what: str):
        if not name or name in getattr(model, table):
            return
        if name in clip.definitions.get(table, {}):
            if name not in add[table]:
                add[table].append(name)
        else:
            lost.append(f"{what} {name!r}")
    for f in clip.frames:
        if f.type != "Null":
            need("frame_sections", f.section, "frame section")
    for a in clip.areas:
        need("shell_sections", a.section, "shell section")
    for t in ("frame_sections", "shell_sections"):
        for n in add[t]:
            need("materials", getattr(clip.definitions[t][n], "material", None), "material")
    pats = {ld.pattern for f in clip.frames for ld in f.loads} | {ld.pattern for a in clip.areas for ld in a.loads}
    pats |= {ld.pattern for p in clip.points.values() for ld in list(p.loads) + list(p.ground_displacements)}
    for lst in clip.panels.values():
        for pl in lst:
            pats |= {getattr(ld, "pattern", None) for ld in getattr(pl, "loads", None) or []}
    for pat in sorted(x for x in pats if x):
        need("load_patterns", pat, "load pattern")
    dias = {p.diaphragm for p in clip.points.values()} | {a.diaphragm for a in clip.areas}
    for d in sorted(x for x in dias if x):
        need("diaphragms", d, "diaphragm")
    return add, lost


def paste_objects(doc: Document, clip: Clipboard, dx: float, dy: float, dz: float) -> Selection:
    """Paste a clipboard at an offset; returns the new (frames, areas, points).

    Definitions the objects need that the model does not have (a section copied
    from another model) come with them and are reported; a joint that lands on an
    existing joint keeps that joint's own assignments (reported).  Nothing is
    skipped silently: when a definition is missing everywhere the paste stops
    before changing the model."""
    out: Selection = ([], [], [])
    if clip.empty:
        return out
    clip = clip.in_units(doc.model.units)
    add, lost = _missing_definitions(doc.model, clip)
    if lost:
        raise ModelError("MISSING_DEFINITION", "Cannot paste: the copied objects use " + ", ".join(lost) +
                         ", which this model does not have. Define them first.")
    notes: List[str] = []

    def mutate(model: Model):
        for table in DEFINITION_TABLES:
            for n in add[table]:
                getattr(model, table)[n] = copy.deepcopy(clip.definitions[table][n])
        added = [f"{t.replace('_', ' ')[:-1]} {n}" for t in DEFINITION_TABLES for n in add[t]]
        if added:
            notes.append("Paste added the definitions the objects use: " + ", ".join(added))
        new_pts: Dict[str, Point] = {}
        kept = []
        for n, p in clip.points.items():
            existing = model.find_point(p.x + dx, p.y + dy, p.z + dz)
            q = existing or model.add_point(p.x + dx, p.y + dy, p.z + dz)
            if existing is None:
                clone_assignments(p, q)
            elif has_assignments(p):
                kept.append(q.name)
            new_pts[n] = q
            out[2].append(q.name)
        if kept:
            notes.append(f"{len(kept)} pasted joint(s) landed on existing joints ({', '.join(kept[:6])}"
                         f"{'...' if len(kept) > 6 else ''}): those joints keep their own supports, springs and loads")
        for f in clip.frames:
            g = model.add_frame(new_pts[f.point_i], new_pts[f.point_j], f.section, type=f.type, **_frame_kwargs(f))
            g.loads = copy.deepcopy(f.loads)
            out[0].append(g.name)
        for a in clip.areas:
            b = model.add_area([new_pts[p] for p in a.points], a.section, type=a.type, **_area_kwargs(a))
            b.loads = copy.deepcopy(a.loads)
            for panel in clip.panels.get(a.name, []):
                q = model.add_area_panel(b.name, copy.deepcopy(panel.loads), distribution=panel.distribution,
                                         span_dir=panel.span_dir)
                q.receivers = panel.receivers
            out[1].append(b.name)
        _restory_new(model, out)

    doc.edit(mutate)
    for t in notes:
        doc.note(t, "info")
    return out


def _restory_new(model: Model, sel: Selection) -> None:
    tol = model.options.merge_tolerance
    for n in sel[0]:
        f = model.frames[n]
        s = model.story_at(max(model.points[f.point_i].z, model.points[f.point_j].z), tol)
        f.story = s.name if s else f.story
    for n in sel[1]:
        a = model.areas[n]
        s = model.story_at(max(model.points[p].z for p in a.points), tol)
        a.story = s.name if s else a.story


# ------------------------------------------------------------ replicate (Edit)
def _is_translation(T) -> bool:
    o = np.asarray(T(np.zeros(3)), float)
    return all(np.allclose(np.asarray(T(e), float) - o, e, atol=1e-9) for e in np.eye(3))


def replicate_transform(doc: Document, frames: Sequence[str], areas: Sequence[str], points: Sequence[str],
                        transforms: Sequence[Callable[[np.ndarray], np.ndarray]],
                        reverse_areas: bool = False, delete_original: bool = False,
                        extra: Optional[Callable[[Model], None]] = None) -> List[str]:
    """Copy the selection once per transform (xyz -> xyz).  Used by the Linear,
    Radial and Mirror tabs of *Edit > Replicate* and by Copy / Rotate / Mirror
    with the copy option.  New joints get every assignment of their original
    (supports, springs, masses, diaphragm, local axes, loads, ground
    displacements); a copy that lands on an existing joint leaves that joint as it
    is (reported).  ``extra`` runs inside the same undo step."""
    created: List[str] = []
    frames, areas, points = list(frames), list(areas), list(points)
    if not transforms or not (frames or areas or points):
        return created
    notes: List[str] = []

    def mutate(model: Model):
        kept: List[str] = []
        turned = False  # a copy kept directional values as assigned (not a plan rotation / mirror)
        tilted: List[str] = []
        for T in transforms:
            cache: Dict[str, Point] = {}
            rigid = _is_translation(T)
            A = None if rigid else plan_map(T)  # rotation about Z / mirror in a vertical plane

            def moved(name: str) -> Point:
                nonlocal turned
                if name not in cache:
                    p = model.points[name]
                    x, y, z = (float(v) for v in T(np.array([p.x, p.y, p.z], float)))
                    existing = model.find_point(x, y, z)
                    q = existing or model.add_point(x, y, z)
                    if existing is None:
                        clone_assignments(p, q)
                        if A is not None:
                            if not turn_assignments(q, A):
                                tilted.append(q.name)
                        elif not rigid and turns_directions(p):
                            turned = True
                    elif has_assignments(p) and existing.name != name:
                        kept.append(existing.name)
                    cache[name] = q
                return cache[name]

            for n in points:
                created.append(moved(n).name)
            for n in frames:
                f = model.frames[n]
                g = model.add_frame(moved(f.point_i), moved(f.point_j), f.section, type=f.type, **_frame_kwargs(f))
                g.loads = copy.deepcopy(f.loads)
                if A is not None:
                    pi, pj = model.points[g.point_i], model.points[g.point_j]
                    turn_frame(g, A, vertical=math.hypot(pi.x - pj.x, pi.y - pj.y) <= model.options.merge_tolerance)
                created.append(g.name)
            for n in areas:
                a = model.areas[n]
                pts = [moved(p) for p in a.points]
                if reverse_areas:
                    pts = pts[::-1]
                b = model.add_area(pts, a.section, type=a.type, **_area_kwargs(a))
                b.loads = copy.deepcopy(a.loads)
                if A is not None:
                    b.loads = turn_loads(b.loads, A, "area")
                copy_area_panels(model, n, b.name)
                created.append(b.name)
        if tilted:
            notes.append(f"Joints {', '.join(tilted[:6])}: tilted local axes could not be turned with the copy - "
                         "their supports / springs are kept as assigned")
        if kept:
            notes.append(f"{len(set(kept))} copied joint(s) landed on existing joints ({', '.join(sorted(set(kept))[:6])}): "
                         "those joints keep their own supports, springs and loads")
        if turned:
            notes.append("These copies are turned out of the plan (an elevation rotation): their joint loads, "
                         "supports and springs are kept in the same global directions as the originals")
        if extra is not None:
            extra(model)
        if delete_original:
            _delete_originals(model, frames, areas, points, created)
        _restory(model)

    doc.edit(mutate)
    for t in notes:
        doc.note(t, "info")
    return created


def _delete_originals(model: Model, frames, areas, points, created) -> None:
    originals = set(_sel_points(model, frames, areas, points))
    for n in frames:
        model.frames.pop(n, None)
    for n in areas:
        model.areas.pop(n, None)
        for pn in [p.name for p in model.panel_loads.values() if p.area == n]:
            del model.panel_loads[pn]  # the original's area loads go with it
    keep = set(created)
    used = set()
    for f in model.frames.values():
        used |= {f.point_i, f.point_j}
    for a in model.areas.values():
        used |= set(a.points)
    for s in getattr(model, "line_springs", {}).values():
        used |= {s.point_i, s.point_j}
    for lk in getattr(model, "links", {}).values():
        used |= {lk.point_i, lk.point_j}
    explicit = set(points)
    for n in originals:
        if n in keep or n in used:
            continue
        # a joint the user selected goes (its copy has its assignments); a joint that only
        # ended a deleted member goes when nothing else is assigned to it
        if n in explicit or not has_assignments(model.points[n]):
            model.points.pop(n, None)
            for g in model.groups.values():
                if n in g.points:
                    g.points.remove(n)


def linear_transforms(dx: float, dy: float, dz: float, count: int):
    return [lambda p, k=k: p + np.array([dx, dy, dz]) * k for k in range(1, max(count, 0) + 1)]


def radial_transforms(cx: float, cy: float, angle_deg: float, count: int):
    """Rotation about a vertical axis through (cx, cy), ``count`` copies."""
    out = []
    for k in range(1, max(count, 0) + 1):
        a = math.radians(angle_deg * k)
        c, s = math.cos(a), math.sin(a)

        def T(p, c=c, s=s):
            x, y = p[0] - cx, p[1] - cy
            return np.array([cx + c * x - s * y, cy + s * x + c * y, p[2]])

        out.append(T)
    return out


def mirror_transform(axis: str, value: float):
    """Mirror about the vertical plane X = value (``axis`` "x") or Y = value ("y")."""
    k = 0 if axis.lower() == "x" else 1

    def T(p):
        q = p.copy()
        q[k] = 2 * value - p[k]
        return q

    return [T]


# ------------------------------------------------------------- extrude (Edit)
def extrude_points_to_frames(doc: Document, points: Sequence[str], dx: float, dy: float, dz: float,
                             count: int, section: str, frame_type: Optional[str] = None) -> List[str]:
    """*Edit > Extrude > Extrude Joints to Frames* (linear)."""
    created: List[str] = []
    points = list(points)
    if not points or count < 1:
        return created
    if section not in doc.model.frame_sections:
        raise ModelError("UNKNOWN_SECTION", f"Frame section {section!r} is not defined")

    def mutate(model: Model):
        for n in points:
            prev = model.points[n]
            for k in range(1, count + 1):
                nxt = model.add_point(prev.x + dx, prev.y + dy, prev.z + dz)
                created.append(model.add_frame(prev, nxt, section, type=frame_type).name)
                prev = nxt

    doc.edit(mutate)
    return created


def extrude_frames_to_shells(doc: Document, frames: Sequence[str], dx: float, dy: float, dz: float,
                             count: int, section: Optional[str], area_type: Optional[str] = None,
                             delete_frames: bool = False) -> List[str]:
    """*Edit > Extrude > Extrude Frames to Shells* (linear).  All or nothing: a
    frame parallel to the extrusion (it would give a shell of zero area) stops the
    command before anything is made (function audit FA14)."""
    created: List[str] = []
    frames = [n for n in frames if n in doc.model.frames]
    if not frames or count < 1:
        return created
    if section is not None and section not in doc.model.shell_sections:
        raise ModelError("UNKNOWN_SECTION", f"Shell section {section!r} is not defined")
    m = doc.model
    v = np.array([dx, dy, dz], float)
    if np.linalg.norm(v) <= max(m.options.merge_tolerance, 1e-12):
        raise ModelError("ZERO_EXTRUSION", "The extrusion vector is zero")
    bad = []
    for n in frames:
        f = m.frames[n]
        d = np.subtract(m.points[f.point_j].xyz, m.points[f.point_i].xyz)
        area = np.linalg.norm(np.cross(d, v))
        if area <= 1e-9 * max(np.linalg.norm(d) * np.linalg.norm(v), 1e-30):
            bad.append(n)
    if bad:
        raise ModelError("PARALLEL_EXTRUSION",
                         f"{', '.join(bad[:8])}{'...' if len(bad) > 8 else ''}: parallel to the extrusion direction, "
                         "the shell would have no area. Nothing was extruded; deselect them or change the direction.")

    def mutate(model: Model):
        for n in frames:
            f = model.frames[n]
            a, b = model.points[f.point_i], model.points[f.point_j]
            for k in range(1, count + 1):
                a2 = model.add_point(a.x + dx, a.y + dy, a.z + dz)
                b2 = model.add_point(b.x + dx, b.y + dy, b.z + dz)
                created.append(model.add_area([a, b, b2, a2], section, type=area_type).name)
                a, b = a2, b2
            if delete_frames:
                model.frames.pop(n)

    doc.edit(mutate)
    return created


# ---------------------------------------------------------- joints (Edit)
def merge_joints(doc: Document, points: Sequence[str], tolerance: float) -> int:
    """*Edit > Merge Joints*: selected joints within ``tolerance`` of an earlier
    selected joint are merged into it.  Their assignments are combined (supports,
    springs and masses add, loads are kept) and every object that refers to them -
    frames, shells, line springs, links, groups, monitored joints - is re-pointed;
    frames that collapse to zero length are removed.  Two joints that cannot be
    combined (different diaphragms ...) stop the command before any change.
    Returns the number of joints merged."""
    names = sorted(set(points) & set(doc.model.points),
                   key=lambda n: (doc.model.points[n].z, doc.model.points[n].x, doc.model.points[n].y, n))
    if len(names) < 2:
        return 0
    target: Dict[str, str] = {}
    keep: List[str] = []
    m = doc.model
    for n in names:
        p = m.points[n]
        for k in keep:
            q = m.points[k]
            if abs(p.z - q.z) <= tolerance and math.dist((p.x, p.y, p.z), (q.x, q.y, q.z)) <= tolerance:
                target[n] = k
                break
        else:
            keep.append(n)
    if not target:
        return 0
    choices = resolve_merge(m, target)  # different values: the user chooses first (or nothing happens)

    def mutate(model: Model):
        removed = merge_points(model, target, choices)
        if removed["frames"] or removed["areas"]:
            doc.note(f"Merge Joints removed {len(removed['frames'])} frame(s) and {len(removed['areas'])} shell(s) "
                     "that collapsed", "info")

    doc.edit(mutate)
    return len(target)


def align_points(doc: Document, points: Sequence[str], axis: str, value: float) -> int:
    """*Edit > Align Joints/Frames/Edges*: joints to an X, Y or Z ordinate."""
    k = "xyz".index(axis.lower())
    names = [n for n in points if n in doc.model.points]
    if not names:
        return 0

    def mutate(model: Model):
        for n in names:
            setattr(model.points[n], "xyz"[k], float(value))
        _restory(model)

    doc.edit(mutate)
    return len(names)


# ---------------------------------------------------------- frames (Edit)
def reverse_frames(doc: Document, frames: Sequence[str]) -> int:
    """*Edit > Edit Frames > Reverse Frame Connectivity*: swap ends I and J.  The
    member stays the same physical member: loads keep their place (relative or
    absolute distances), end conditions, hinges and local axes follow
    (function audit FA08)."""
    names = [n for n in frames if n in doc.model.frames]

    def mutate(model: Model):
        for n in names:
            f = model.frames[n]
            L = math.dist(model.points[f.point_i].xyz, model.points[f.point_j].xyz)
            reverse_frame(f, L)

    if names:
        doc.edit(mutate)
    return len(names)


def _collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float) -> bool:
    u, v = b - a, c - b
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return n > 0 and np.linalg.norm(np.cross(u, v)) / n < 1e-6 and float(np.dot(u, v)) > 0


def _join_problem(model: Model, f: Frame, g: Frame, mid: str, f_at_mid: str, g_at_mid: str) -> Optional[str]:
    """Why f and g cannot become one member without losing something (None: they can)."""
    pm = model.points[mid]
    if has_assignments(pm):
        return f"joint {mid} between them has supports, springs, mass or loads"
    for h, end in ((f, f_at_mid), (g, g_at_mid)):
        rel = h.releases_i if end == "i" else h.releases_j
        fx = (h.fixity_i_trans, h.fixity_i_rot) if end == "i" else (h.fixity_j_trans, h.fixity_j_rot)
        off = h.end_offset_i if end == "i" else h.end_offset_j
        jo = h.joint_offset_i if end == "i" else h.joint_offset_j
        if any(rel) or any(fx[0]) or any(fx[1]):
            return f"{h.name} has an end release at joint {mid}"
        if off or any(jo):
            return f"{h.name} has an end offset at joint {mid}"
    if any(ld.kind == "temperature" for h in (f, g) for ld in h.loads):
        return "temperature loads cannot be joined"
    if f.hinges or g.hinges:
        return "hinges are assigned (join them, then assign the hinges)"
    for k in ("cardinal_point", "modifiers", "mass_per_length", "rigid_factor", "auto_end_offsets", "pier",
              "spandrel", "tension_limit", "compression_limit", "foundation", "nonlinear", "joint_offset_local"):
        if getattr(f, k) != getattr(g, k):
            return f"{f.name} and {g.name} have a different {k.replace('_', ' ')}"
    return None


def join_frames(doc: Document, frames: Sequence[str]) -> List[str]:
    """*Edit > Edit Frames > Join Frames*: selected collinear frames with the same
    section that meet at a joint used by nothing else are joined into one.  Every
    load of both frames is kept at its place along the joined member, the outer end
    conditions are kept; pairs that would lose something (a release or an end
    offset at the joint between them, a load at that joint ...) are not joined and
    the reason is reported (function audit FA07)."""
    m = doc.model
    names = [n for n in frames if n in m.frames]
    if len(names) < 2:
        return []
    usage: Dict[str, int] = {}
    for f in m.frames.values():
        for p in (f.point_i, f.point_j):
            usage[p] = usage.get(p, 0) + 1
    for a in m.areas.values():
        for p in a.points:
            usage[p] = usage.get(p, 0) + 10
    for s in getattr(m, "line_springs", {}).values():
        for p in (s.point_i, s.point_j):
            usage[p] = usage.get(p, 0) + 10
    for lk in getattr(m, "links", {}).values():
        for p in (lk.point_i, lk.point_j):
            if p:
                usage[p] = usage.get(p, 0) + 10
    joined: List[str] = []
    problems: List[str] = []

    def mutate(model: Model):
        pool = set(names)
        changed = True
        while changed:
            changed = False
            for n in sorted(pool):
                f = model.frames.get(n)
                if f is None:
                    continue
                for o in sorted(pool - {n}):
                    g = model.frames.get(o)
                    if g is None or g.section != f.section or g.type != f.type:
                        continue
                    # (a, mid, b): the joined member runs a -> b; f_end / g_end: their end at mid
                    if f.point_j == g.point_i:
                        a, mid, b, fe, ge = f.point_i, f.point_j, g.point_j, "j", "i"
                    elif f.point_j == g.point_j:
                        a, mid, b, fe, ge = f.point_i, f.point_j, g.point_i, "j", "j"
                    elif f.point_i == g.point_j:
                        a, mid, b, fe, ge = g.point_i, f.point_i, f.point_j, "i", "j"
                    elif f.point_i == g.point_i:
                        a, mid, b, fe, ge = g.point_j, f.point_i, f.point_j, "i", "i"
                    else:
                        continue
                    if usage.get(mid, 0) != 2:
                        continue
                    P = lambda q: np.array(model.points[q].xyz, float)  # noqa: E731
                    if not _collinear(P(a), P(mid), P(b), model.options.merge_tolerance):
                        continue
                    Lf = float(np.linalg.norm(P(f.point_j) - P(f.point_i)))
                    Lg = float(np.linalg.norm(P(g.point_j) - P(g.point_i)))
                    f_first = fe == "j"  # f is the a-side piece; f always runs the joined member's way
                    f_flip = False
                    g_flip = (ge == "j") if f_first else (ge == "i")  # g runs the other way
                    ang_g = -g.angle if g_flip else g.angle
                    if abs(ang_g - f.angle) > 1e-9:
                        problems.append(f"{f.name} + {g.name}: different local axis angles")
                        continue
                    why = _join_problem(model, f, g, mid, fe, ge)
                    if why:
                        problems.append(f"{f.name} + {g.name}: {why}")
                        continue
                    L = Lf + Lg
                    if f_first:
                        parts = [(f, Lf, 0.0, f_flip), (g, Lg, Lf, g_flip)]
                    else:
                        parts = [(g, Lg, 0.0, g_flip), (f, Lf, Lg, f_flip)]
                    loads = joined_loads(parts, L)
                    old_f = copy.deepcopy(f)
                    # outer end conditions: end a from the a-side piece, end b from the b-side piece
                    def end_of(h, at):  # (releases, fixity trans, fixity rot, offset, joint offset) of h at joint `at`
                        if h.point_i == at:
                            return (h.releases_i, h.fixity_i_trans, h.fixity_i_rot, h.end_offset_i, h.joint_offset_i)
                        return (h.releases_j, h.fixity_j_trans, h.fixity_j_rot, h.end_offset_j, h.joint_offset_j)
                    ea = end_of(old_f if f_first else g, a)
                    eb = end_of(g if f_first else old_f, b)
                    f.point_i, f.point_j = a, b
                    f.releases_i, f.fixity_i_trans, f.fixity_i_rot, f.end_offset_i, f.joint_offset_i = ea
                    f.releases_j, f.fixity_j_trans, f.fixity_j_rot, f.end_offset_j, f.joint_offset_j = eb
                    f.loads = loads
                    model.frames.pop(o)
                    model.points.pop(mid, None)
                    for grp in model.groups.values():
                        if o in grp.frames:
                            grp.frames.remove(o)
                            if f.name not in grp.frames:
                                grp.frames.append(f.name)
                        if mid in grp.points:
                            grp.points.remove(mid)
                    pool.discard(o)
                    joined.append(o)
                    changed = True
                    break
                if changed:
                    break

    doc.edit(mutate)
    if problems:
        doc.note("Not joined: " + "; ".join(dict.fromkeys(problems)), "warning")
    return joined


# ---------------------------------------------------------- shells (Edit)
def divide_shells(doc: Document, areas: Sequence[str], n1: int, n2: int) -> List[str]:
    """*Edit > Edit Shells > Divide Shells* ("Divide quadrilaterals / triangles into"
    n1 x n2 objects).  Quadrilaterals only; other shapes are left unchanged."""
    created: List[str] = []
    names = [n for n in areas if n in doc.model.areas and len(doc.model.areas[n].points) == 4]
    if not names or n1 < 1 or n2 < 1 or (n1 == 1 and n2 == 1):
        return created

    def mutate(model: Model):
        for n in names:
            originals = [p for p in model.panel_loads.values() if p.area == n]  # before names get reused
            for p in originals:
                del model.panel_loads[p.name]
            a = model.areas.pop(n)
            c = [np.array(model.points[p].xyz, float) for p in a.points]
            grid = [[None] * (n2 + 1) for _ in range(n1 + 1)]
            for i in range(n1 + 1):
                for j in range(n2 + 1):
                    s, t = i / n1, j / n2
                    xyz = (1 - s) * (1 - t) * c[0] + s * (1 - t) * c[1] + s * t * c[2] + (1 - s) * t * c[3]
                    grid[i][j] = model.add_point(*xyz)
            for i in range(n1):
                for j in range(n2):
                    b = model.add_area([grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]], a.section,
                                       type=a.type, **_area_kwargs(a))
                    b.loads = copy.deepcopy(a.loads)
                    b.label = a.label
                    for p in originals:  # every piece gets the floor's area loads
                        q = model.add_area_panel(b.name, copy.deepcopy(p.loads), distribution=p.distribution,
                                                 span_dir=p.span_dir)
                        q.receivers, q.label = p.receivers, p.label
                    created.append(b.name)

    doc.edit(mutate)
    return created


def reverse_area_axis(doc: Document, areas: Sequence[str]) -> int:
    """*Edit > Edit Shells > Reverse Wall Local 3 Axis* (point order reversed)."""
    names = [n for n in areas if n in doc.model.areas]

    def mutate(model: Model):
        for n in names:
            a = model.areas[n]
            a.points = a.points[::-1]

    if names:
        doc.edit(mutate)
    return len(names)


# ------------------------------------------------------------- grids (Edit)
def _next_grid_name(model: Model, direction: str) -> str:
    used = {g.name for g in model.grid_lines}
    if direction == "X":
        for k in range(1, 10000):
            s, n = "", k
            while n:
                n, r = divmod(n - 1, 26)
                s = chr(65 + r) + s
            if s not in used:
                return s
    k = 1
    while str(k) in used:
        k += 1
    return str(k)


def add_grid_lines_at_points(doc: Document, points: Sequence[str]) -> int:
    """*Edit > Add Grid Lines at Selected Joints*: X and Y grid lines through
    the selected joints where none exists yet."""
    m = doc.model
    tol = max(m.options.merge_tolerance, 1e-9)
    pts = [m.points[n] for n in points if n in m.points]
    if not pts:
        return 0
    new: List[Tuple[str, float]] = []
    for p in pts:
        for d, v in (("X", p.x), ("Y", p.y)):
            if not any(g.direction == d and abs(g.ordinate - v) <= tol for g in m.grid_lines) and \
                    not any(dd == d and abs(vv - v) <= tol for dd, vv in new):
                new.append((d, v))
    if not new:
        return 0

    def mutate(model: Model):
        system = next(iter(model.grid_systems), "G1")
        for d, v in new:
            model.grid_lines.append(GridLine(system=system, name=_next_grid_name(model, d), direction=d, ordinate=v))
        model.grid_lines.sort(key=lambda g: ("XYZG".find(g.direction[0]), g.ordinate))

    doc.edit(mutate)
    return len(new)


# ------------------------------------------------------------ labels (Edit)
LABEL_PREFIX = {"Beam": "B", "Column": "C", "Brace": "D", "Null": "N", "Floor": "F", "Wall": "W",
                "Ramp": "R", "Opening": "O"}


def auto_relabel(doc: Document) -> int:
    """*Edit > Auto Relabel All*: plan labels per story in X-then-Y order
    (B1, B2 ... C1 ... F1 ... W1 ..., joints 1, 2, ...)."""
    def mutate(model: Model):
        pts = model.points

        def key_frame(f):
            a, b = pts[f.point_i], pts[f.point_j]
            return (round((a.y + b.y) / 2, 6), round((a.x + b.x) / 2, 6))

        def key_area(a):
            xs = [pts[p] for p in a.points]
            return (round(sum(p.y for p in xs) / len(xs), 6), round(sum(p.x for p in xs) / len(xs), 6))

        for story in [None] + list(model.stories):
            counters: Dict[str, int] = {}
            for f in sorted((f for f in model.frames.values() if f.story == story), key=key_frame):
                pre = LABEL_PREFIX.get(f.type, "B")
                counters[pre] = counters.get(pre, 0) + 1
                f.label = f"{pre}{counters[pre]}"
            for a in sorted((a for a in model.areas.values() if a.story == story), key=key_area):
                pre = LABEL_PREFIX.get(a.type, "F")
                counters[pre] = counters.get(pre, 0) + 1
                a.label = f"{pre}{counters[pre]}"
            k = 0
            for p in sorted((p for p in pts.values() if p.story == story), key=lambda p: (round(p.y, 6), round(p.x, 6))):
                k += 1
                p.label = str(k)

    doc.edit(mutate)
    return len(doc.model.frames) + len(doc.model.areas) + len(doc.model.points)


# ----------------------------------------------------------- assign helpers
def assign(doc: Document, kind: str, names: Sequence[str], **values) -> int:
    """Set attributes on selected objects (``kind``: point | frame | area)."""
    table = {"point": "points", "frame": "frames", "area": "areas"}[kind]
    names = [n for n in names if n in getattr(doc.model, table)]
    if not names or not values:
        return 0

    def mutate(model: Model):
        objs = getattr(model, table)
        for n in names:
            for k, v in values.items():
                setattr(objs[n], k, copy.deepcopy(v))

    doc.edit(mutate)
    return len(names)


ASSIGN_FIELDS = {
    "point": ("restraints", "springs", "rotational_springs", "mass", "rotational_mass", "diaphragm", "panel_zone"),
    "frame": ("section", "angle", "releases_i", "releases_j", "fixity_i_trans", "fixity_i_rot", "fixity_j_trans",
              "fixity_j_rot", "end_offset_i", "end_offset_j", "rigid_factor", "auto_end_offsets", "cardinal_point",
              "display_cardinal", "joint_offset_i", "joint_offset_j", "modifiers", "mass_per_length",
              "tension_limit", "compression_limit", "auto_mesh", "mesh_at_intersections", "mesh_at_floor",
              "max_station_spacing", "min_stations", "pier", "spandrel"),
    "area": ("section", "diaphragm", "angle", "modifiers", "mass_per_area", "auto_mesh", "mesh_type",
             "max_mesh_size", "mesh_divisions", "edge_releases", "spring", "pier", "spandrel"),
}


@dataclass
class AssignClipboard:
    kind: str
    values: dict
    loads: list


def copy_assigns(model: Model, kind: str, name: str) -> AssignClipboard:
    """*Assign > Copy Assigns* from one object."""
    obj = getattr(model, {"point": "points", "frame": "frames", "area": "areas"}[kind])[name]
    vals = {k: copy.deepcopy(getattr(obj, k)) for k in ASSIGN_FIELDS[kind] if hasattr(obj, k)}
    loads = copy.deepcopy(obj.loads)
    if kind == "area":  # area loads are the panels made from the area
        loads = [copy.deepcopy(p) for p in model.panel_loads.values() if p.area == name]
    return AssignClipboard(kind, vals, loads)


def paste_assigns(doc: Document, clip: AssignClipboard, names: Sequence[str], loads: bool = True) -> int:
    """*Assign > Paste Assigns* to selected objects of the same kind: the
    assignments and (with ``loads``) the loads in one undo step (function audit
    FA12)."""
    table = {"point": "points", "frame": "frames", "area": "areas"}[clip.kind]
    names = [n for n in names if n in getattr(doc.model, table)]
    vals = dict(clip.values)
    if not names or not (vals or loads):
        return 0

    def mutate(m: Model):
        objs = getattr(m, table)
        for x in names:
            for k, v in vals.items():
                setattr(objs[x], k, copy.deepcopy(v))
        if not loads:
            return
        if clip.kind == "area":
            for x in names:
                for pn in [p.name for p in m.panel_loads.values() if p.area == x]:  # paste replaces
                    del m.panel_loads[pn]
                for panel in clip.loads:
                    q = m.add_area_panel(x, copy.deepcopy(panel.loads), distribution=panel.distribution,
                                         span_dir=panel.span_dir)
                    q.receivers = panel.receivers
        else:
            for x in names:
                objs[x].loads = copy.deepcopy(clip.loads)

    doc.edit(mutate)
    return len(names)


def assign_to_group(doc: Document, group: str, frames: Sequence[str], areas: Sequence[str],
                    points: Sequence[str], replace: bool = False) -> int:
    """*Assign > Assign Objects to Group*."""
    if not group:
        raise ModelError("INVALID_NAME", "Group name is empty")

    def mutate(model: Model):
        g = model.groups.get(group)
        if g is None:
            g = model.groups[group] = Group(name=group)
        if replace:
            g.points, g.frames, g.areas = [], [], []
        for lst, names, table in ((g.frames, frames, model.frames), (g.areas, areas, model.areas),
                                  (g.points, points, model.points)):
            for n in names:
                if n in table and n not in lst:
                    lst.append(n)
        for n in frames:
            if n in model.frames:
                model.frames[n].group = group
        for n in areas:
            if n in model.areas:
                model.areas[n].group = group

    doc.edit(mutate)
    return len(frames) + len(areas) + len(points)


# -------------------------------------------------------- select helpers
def objects_with_material(model: Model, material: str) -> Selection:
    fs = [f.name for f in model.frames.values()
          if f.section in model.frame_sections and model.frame_sections[f.section].material == material]
    ar = [a.name for a in model.areas.values()
          if a.section in model.shell_sections and model.shell_sections[a.section].material == material]
    return fs, ar, []


def objects_with_shell_section(model: Model, section: str) -> Selection:
    return [], [a.name for a in model.areas.values() if a.section == section], []


def objects_with_frame_section(model: Model, section: str) -> Selection:
    return [f.name for f in model.frames.values() if f.section == section], [], []


def objects_in_group(model: Model, group: str) -> Selection:
    g = model.groups.get(group)
    if g is None:
        return [], [], []
    fr = set(g.frames) | {f.name for f in model.frames.values() if f.group == group}
    ar = set(g.areas) | {a.name for a in model.areas.values() if a.group == group}
    return sorted(fr & set(model.frames)), sorted(ar & set(model.areas)), [p for p in g.points if p in model.points]


def objects_with_label(model: Model, kind: str, text: str) -> Selection:
    """Objects whose label (or unique name) equals one of the comma separated ``text``."""
    wanted = {t.strip() for t in text.replace(";", ",").split(",") if t.strip()}
    table = {"point": model.points, "frame": model.frames, "area": model.areas}[kind]
    names = [o.name for o in table.values() if (o.label or o.name) in wanted or o.name in wanted]
    return ((names, [], []) if kind == "frame" else ([], names, []) if kind == "area" else ([], [], names))


def objects_with_pier(model: Model, label: str, spandrel: bool = False) -> Selection:
    attr = "spandrel" if spandrel else "pier"
    return ([f.name for f in model.frames.values() if getattr(f, attr) == label],
            [a.name for a in model.areas.values() if getattr(a, attr) == label], [])


# ------------------------------------------------------------- measure
def measure(points: Sequence[XYZ], kind: str) -> str:
    """Status text of *View > Measure* (Line / Area / Angle)."""
    P = [np.array(p, float) for p in points]
    if kind == "line" and len(P) >= 2:
        d = P[1] - P[0]
        return (f"Length = {np.linalg.norm(d):.4g}   (dX {d[0]:.4g}, dY {d[1]:.4g}, dZ {d[2]:.4g})")
    if kind == "angle" and len(P) >= 3:
        u, v = P[0] - P[1], P[2] - P[1]
        c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) or 1.0))
        return f"Angle = {math.degrees(math.acos(max(-1.0, min(1.0, c)))):.4g} deg"
    if kind == "area" and len(P) >= 3:
        n = np.zeros(3)
        for k in range(len(P)):
            n += np.cross(P[k], P[(k + 1) % len(P)])
        per = sum(np.linalg.norm(P[(k + 1) % len(P)] - P[k]) for k in range(len(P)))
        return f"Area = {np.linalg.norm(n) / 2:.4g}   Perimeter = {per:.4g}"
    return ""
