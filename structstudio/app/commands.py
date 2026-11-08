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
    objects come with their properties; Paste asks for the offset)."""

    points: Dict[str, Point] = field(default_factory=dict)
    frames: List[Frame] = field(default_factory=list)
    areas: List[Area] = field(default_factory=list)
    panels: Dict[str, list] = field(default_factory=dict)  # area name -> its area-load panels

    @property
    def empty(self) -> bool:
        return not (self.points or self.frames or self.areas)


def copy_objects(model: Model, frames: Sequence[str], areas: Sequence[str], points: Sequence[str]) -> Clipboard:
    clip = Clipboard()
    for n in _sel_points(model, frames, areas, points):
        clip.points[n] = copy.deepcopy(model.points[n])
    clip.frames = [copy.deepcopy(model.frames[n]) for n in frames if n in model.frames]
    clip.areas = [copy.deepcopy(model.areas[n]) for n in areas if n in model.areas]
    for a in clip.areas:
        clip.panels[a.name] = [copy.deepcopy(p) for p in model.panel_loads.values() if p.area == a.name]
    return clip


def paste_objects(doc: Document, clip: Clipboard, dx: float, dy: float, dz: float) -> Selection:
    """Paste a clipboard at an offset; returns the new (frames, areas, points)."""
    out: Selection = ([], [], [])
    if clip.empty:
        return out

    def mutate(model: Model):
        new_pts: Dict[str, Point] = {}
        for n, p in clip.points.items():
            q = model.add_point(p.x + dx, p.y + dy, p.z + dz)
            if any(p.restraints) and not any(q.restraints):
                q.restraints = p.restraints
            new_pts[n] = q
            out[2].append(q.name)
        for f in clip.frames:
            if f.section not in model.frame_sections:
                continue
            g = model.add_frame(new_pts[f.point_i], new_pts[f.point_j], f.section, type=f.type, **_frame_kwargs(f))
            g.loads = copy.deepcopy(f.loads)
            out[0].append(g.name)
        for a in clip.areas:
            sec = a.section if a.section in model.shell_sections else None
            b = model.add_area([new_pts[p] for p in a.points], sec, type=a.type, **_area_kwargs(a))
            b.loads = copy.deepcopy(a.loads)
            for panel in clip.panels.get(a.name, []):
                q = model.add_area_panel(b.name, copy.deepcopy(panel.loads), distribution=panel.distribution,
                                         span_dir=panel.span_dir)
                q.receivers = panel.receivers
            out[1].append(b.name)

    doc.edit(mutate)
    return out


# ------------------------------------------------------------ replicate (Edit)
def replicate_transform(doc: Document, frames: Sequence[str], areas: Sequence[str], points: Sequence[str],
                        transforms: Sequence[Callable[[np.ndarray], np.ndarray]],
                        reverse_areas: bool = False, delete_original: bool = False) -> List[str]:
    """Copy the selection once per transform (xyz -> xyz).  Used by the Linear,
    Radial and Mirror tabs of *Edit > Replicate*."""
    created: List[str] = []
    frames, areas, points = list(frames), list(areas), list(points)
    if not transforms or not (frames or areas or points):
        return created

    def mutate(model: Model):
        for T in transforms:
            cache: Dict[str, Point] = {}

            def moved(name: str) -> Point:
                if name not in cache:
                    p = model.points[name]
                    x, y, z = (float(v) for v in T(np.array([p.x, p.y, p.z], float)))
                    q = model.add_point(x, y, z)
                    if any(p.restraints) and not any(q.restraints):
                        q.restraints = p.restraints
                    cache[name] = q
                return cache[name]

            for n in points:
                created.append(moved(n).name)
            for n in frames:
                f = model.frames[n]
                g = model.add_frame(moved(f.point_i), moved(f.point_j), f.section, type=f.type, **_frame_kwargs(f))
                g.loads = copy.deepcopy(f.loads)
                created.append(g.name)
            for n in areas:
                a = model.areas[n]
                pts = [moved(p) for p in a.points]
                if reverse_areas:
                    pts = pts[::-1]
                b = model.add_area(pts, a.section, type=a.type, **_area_kwargs(a))
                b.loads = copy.deepcopy(a.loads)
                copy_area_panels(model, n, b.name)
                created.append(b.name)
        if delete_original:
            originals = set(_sel_points(model, frames, areas, points))
            for n in frames:
                model.frames.pop(n, None)
            for n in areas:
                model.areas.pop(n, None)
                for pn in [p.name for p in model.panel_loads.values() if p.area == n]:
                    del model.panel_loads[pn]  # the original's area loads go with it
            keep = set(created)
            for n in _unused_points(model):
                if n in originals and n not in keep:
                    model.points.pop(n, None)

    doc.edit(mutate)
    return created


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
    """*Edit > Extrude > Extrude Frames to Shells* (linear)."""
    created: List[str] = []
    frames = list(frames)
    if not frames or count < 1:
        return created
    if section is not None and section not in doc.model.shell_sections:
        raise ModelError("UNKNOWN_SECTION", f"Shell section {section!r} is not defined")

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
    selected joint are merged into it; objects are reconnected, frames that
    collapse to zero length are removed.  Returns the number of joints merged."""
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

    def mutate(model: Model):
        for n, k in target.items():
            p, q = model.points[n], model.points[k]
            q.restraints = tuple(a or b for a, b in zip(q.restraints, p.restraints))
            q.loads.extend(p.loads)
        for f in list(model.frames.values()):
            f.point_i = target.get(f.point_i, f.point_i)
            f.point_j = target.get(f.point_j, f.point_j)
            if f.point_i == f.point_j:
                model.frames.pop(f.name)
        for a in list(model.areas.values()):
            pts: List[str] = []
            for p in a.points:
                p = target.get(p, p)
                if not pts or pts[-1] != p:
                    pts.append(p)
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts.pop()
            if len(pts) < 3:
                model.areas.pop(a.name)
            else:
                a.points = pts
        for s in getattr(model, "line_springs", {}).values():
            s.point_i = target.get(s.point_i, s.point_i)
            s.point_j = target.get(s.point_j, s.point_j)
        for g in model.groups.values():
            g.points = [p for p in g.points if p not in target]
        for n in target:
            model.points.pop(n, None)

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
    """*Edit > Edit Frames > Reverse Frame Connectivity*: swap ends I and J."""
    names = [n for n in frames if n in doc.model.frames]

    def mutate(model: Model):
        for n in names:
            f = model.frames[n]
            f.point_i, f.point_j = f.point_j, f.point_i
            f.releases_i, f.releases_j = f.releases_j, f.releases_i
            f.fixity_i_trans, f.fixity_j_trans = f.fixity_j_trans, f.fixity_i_trans
            f.fixity_i_rot, f.fixity_j_rot = f.fixity_j_rot, f.fixity_i_rot
            f.end_offset_i, f.end_offset_j = f.end_offset_j, f.end_offset_i
            f.joint_offset_i, f.joint_offset_j = f.joint_offset_j, f.joint_offset_i
            for ld in f.loads:
                if ld.kind == "distributed":
                    ld.rel_a, ld.rel_b = 1.0 - ld.rel_b, 1.0 - ld.rel_a
                    ld.value_a, ld.value_b = ld.value_b, ld.value_a
                else:
                    ld.rel_a = 1.0 - ld.rel_a

    if names:
        doc.edit(mutate)
    return len(names)


def _collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float) -> bool:
    u, v = b - a, c - b
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return n > 0 and np.linalg.norm(np.cross(u, v)) / n < 1e-6 and float(np.dot(u, v)) > 0


def join_frames(doc: Document, frames: Sequence[str]) -> List[str]:
    """*Edit > Edit Frames > Join Frames*: selected collinear frames with the same
    section that meet at a joint used by nothing else are joined into one."""
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
    joined: List[str] = []

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
                    if f.point_j == g.point_i:
                        a, mid, b = f.point_i, f.point_j, g.point_j
                    elif f.point_j == g.point_j:
                        a, mid, b = f.point_i, f.point_j, g.point_i
                    elif f.point_i == g.point_j:
                        a, mid, b = g.point_i, f.point_i, f.point_j
                    elif f.point_i == g.point_i:
                        a, mid, b = g.point_j, f.point_i, f.point_j
                    else:
                        continue
                    pm = model.points[mid]
                    if usage.get(mid, 0) != 2 or any(pm.restraints) or pm.loads:
                        continue
                    P = lambda q: np.array(model.points[q].xyz, float)  # noqa: E731
                    if not _collinear(P(a), P(mid), P(b), model.options.merge_tolerance):
                        continue
                    f.point_i, f.point_j = a, b
                    f.releases_j = g.releases_j if g.point_j == b else g.releases_i
                    f.loads = [ld for ld in f.loads if ld.kind == "distributed" and ld.rel_a == 0 and ld.rel_b == 1]
                    model.frames.pop(o)
                    model.points.pop(mid, None)
                    pool.discard(o)
                    joined.append(o)
                    changed = True
                    break
                if changed:
                    break

    doc.edit(mutate)
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
    """*Assign > Paste Assigns* to selected objects of the same kind."""
    vals = dict(clip.values)
    n = assign(doc, clip.kind, names, **vals)
    if n and loads and clip.kind == "area":
        def mutate(m: Model):
            for x in names:
                if x not in m.areas:
                    continue
                for pn in [p.name for p in m.panel_loads.values() if p.area == x]:  # paste replaces
                    del m.panel_loads[pn]
                for panel in clip.loads:
                    q = m.add_area_panel(x, copy.deepcopy(panel.loads), distribution=panel.distribution,
                                         span_dir=panel.span_dir)
                    q.receivers = panel.receivers
        doc.edit(mutate)
    elif n and loads:
        table = {"point": "points", "frame": "frames", "area": "areas"}[clip.kind]
        doc.edit(lambda m: [setattr(getattr(m, table)[x], "loads", copy.deepcopy(clip.loads))
                            for x in names if x in getattr(m, table)])
    return n


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
