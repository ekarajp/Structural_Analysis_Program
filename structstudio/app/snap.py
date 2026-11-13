"""Object snapping like the ETABS *Snap Options* form and CAD object snaps.

Every snap type can be switched on or off independently (any combination):

=====================  ===================================================================
``joints``             model joints
``ends_midpoints``     ends and midpoints of frames and of shell edges
``grid``               grid-line intersections (and grid x level intersections in
                       elevations, grid x grid x level in 3D)
``lines``              the nearest point on a frame, a shell edge or a grid line
``edges``              the nearest point on a shell edge only
``perpendicular``      from the previous drawn point, the foot of the perpendicular on
                       a frame / shell edge / grid line under the cursor
``intersections``      crossings of frames, shell edges and grid lines in the plane
``extensions``         along the extension of a frame / edge beyond its ends
``parallels``          the next point on a line through the previous point parallel to
                       a frame / edge near the cursor
``fine_grid``          an invisible grid with ``fine_spacing`` (plan views)
``reference``          reference lines through joints (CYPE 3D style)
``ortho``              orthogonal tracking: the next point along the plane axes of the
                       previous point (CAD *Ortho*)
``polar``              polar tracking at multiples of ``polar_angle`` from the previous
                       point
``alignment``          alignment lines (Revit / AutoCAD object snap tracking): the cursor
                       lines up horizontally or vertically with a nearby joint, or with
                       two joints at once (their alignment lines cross)
``length_increments``  snap the segment length to a multiple of one of the increments
``angle_increments``   snap the segment angle to a multiple of ``angle_increment``
=====================  ===================================================================

The engine works in display (screen) coordinates, so one implementation serves
plan, elevation and 3D windows: candidates are projected with the view's
``project`` callback and compared with the cursor in pixels.  The returned
:class:`SnapHit` carries the 3D world point, the snap kind, a description for
the status text and the geometry of any tracking line to draw.

Pressing ``N`` in a window cycles through the candidates inside the tolerance
(``SnapEngine.find`` returns them ordered; the viewport keeps an index).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..model import Model
from .tools import PLANE_BAND, Plane, grid_ordinates, story_levels

XYZ = Tuple[float, float, float]

# (key, label, hint) in the order of the Snap Options form
SNAP_ITEMS: List[Tuple[str, str, str]] = [
    ("joints", "Joints", "Snaps to joints (points) of the model"),
    ("ends_midpoints", "Line ends and midpoints", "Ends and midpoints of frames and of shell edges"),
    ("grid", "Grid intersections", "Intersections of grid lines (with levels in elevations)"),
    ("lines", "Lines, frames and grid lines", "The nearest point on a frame, shell edge or grid line"),
    ("edges", "Edges", "The nearest point on a shell edge"),
    ("perpendicular", "Perpendicular projections",
     "From the previous point, the foot of the perpendicular on the line under the cursor"),
    ("intersections", "Intersections", "Crossings of frames, shell edges and grid lines"),
    ("centers", "Area centers", "The geometric centre of a floor / wall area"),
    ("extensions", "Extensions", "Points on the extension of a frame or edge beyond its ends"),
    ("parallels", "Parallels", "Next point parallel to a frame or edge near the cursor"),
    ("fine_grid", "Fine grid", "An invisible grid with the fine grid spacing (plan views)"),
    ("reference", "Reference lines", "Reference lines through joints"),
    ("ortho", "Orthogonal tracking", "Next point along the plane axes through the previous point"),
    ("polar", "Polar tracking", "Next point at multiples of the polar angle from the previous point"),
    ("alignment", "Alignment lines (object snap tracking)",
     "Line up with nearby joints horizontally / vertically (dashed guide lines)"),
    ("length_increments", "Length increments", "Snap the drawn length to a multiple of an increment"),
    ("angle_increments", "Angle increments", "Snap the drawn angle to a multiple of the angle increment"),
]
SNAP_KEYS = tuple(k for k, _, _ in SNAP_ITEMS)
KIND_LABELS = {
    "joint": "Joint", "end": "End", "midpoint": "Midpoint", "grid": "Grid", "nearest": "Nearest", "edge": "Edge",
    "perpendicular": "Perpendicular", "intersection": "Intersection", "extension": "Extension",
    "parallel": "Parallel", "fine_grid": "Fine grid", "reference": "Reference", "ortho": "Ortho",
    "polar": "Polar", "increment": "Increment", "free": "", "alignment": "Aligned with",
    "alignment2": "Aligned with", "center": "Center",
}


@dataclass
class SnapOptions:
    joints: bool = True
    ends_midpoints: bool = True
    grid: bool = True
    lines: bool = False
    edges: bool = False
    perpendicular: bool = True
    intersections: bool = True
    centers: bool = False
    extensions: bool = True
    parallels: bool = False
    fine_grid: bool = False
    reference: bool = True
    ortho: bool = False
    polar: bool = True
    alignment: bool = True
    length_increments: bool = False
    angle_increments: bool = False
    fine_spacing: float = 0.5  # model length units
    length_steps: List[float] = field(default_factory=lambda: [0.5])  # model length units
    angle_increment: float = 15.0  # degrees
    polar_angle: float = 45.0  # degrees
    tolerance_px: float = 10.0  # screen snap-to tolerance (pixels)
    snap_only: bool = False  # Draw Using Snap Only
    osnap: bool = True  # master switch of the object snaps (status bar SNAP); tracking stays as set

    def enabled(self, key: str) -> bool:
        return bool(getattr(self, key))

    def set(self, key: str, value: bool):
        if key not in SNAP_KEYS:
            raise KeyError(key)
        setattr(self, key, bool(value))

    def select_all(self, on: bool = True):
        for k in SNAP_KEYS:
            setattr(self, k, on)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: Optional[str]) -> "SnapOptions":
        try:
            d = json.loads(text) if text else {}
        except (TypeError, ValueError):
            d = {}
        if not isinstance(d, dict):
            d = {}
        out = cls()
        for f in fields(cls):
            if f.name in d:
                v = d[f.name]
                try:
                    if f.name == "length_steps":
                        out.length_steps = [float(x) for x in v if float(x) > 0]
                    elif isinstance(getattr(out, f.name), bool):
                        setattr(out, f.name, bool(v))
                    else:
                        setattr(out, f.name, float(v))
                except (TypeError, ValueError):
                    pass
        return out

    def copy(self) -> "SnapOptions":
        return SnapOptions.from_json(self.to_json())


@dataclass
class SnapHit:
    xyz: XYZ  # world point
    kind: str  # joint | end | midpoint | grid | nearest | edge | perpendicular | ... | free
    target: Optional[str] = None  # object name
    distance: float = 0.0  # screen distance (pixels)
    track: Optional[Tuple[XYZ, XYZ]] = None  # a tracking line to draw (world)
    text: str = ""  # extra description (length / angle)
    track2: Optional[Tuple[XYZ, XYZ]] = None  # a second tracking line (crossing alignments)
    line: Optional[Tuple[XYZ, XYZ]] = None  # the line the point lies on (nearest / edge / extension ...)

    @property
    def label(self) -> str:
        t = KIND_LABELS.get(self.kind, self.kind)
        if self.target:
            t += f" {self.target}"
        if self.text:
            t += f"  {self.text}"
        return t

    # compatibility with the older SnapResult (xy / kind / target)
    @property
    def xy(self):
        return self.xyz


@dataclass(frozen=True)
class SnapResolution:
    """One resolved pick (immutable): what the marker shows, the preview draws and the
    click commits are the same record.  ``screen_error_px`` is measured by projecting
    ``xyz`` back to the screen (not the distance to a 2D construction)."""

    view: str  # plan | xz | yz | 3d
    cursor: Tuple[float, float]  # display (render) pixels
    xyz: Optional[XYZ]
    kind: str  # joint | ... | free | none
    target: Optional[str]
    screen_error_px: float
    depth: Optional[float] = None
    candidates: int = 0
    index: int = 0
    plane: str = ""  # where a free point lies (plan Z=..., 3D work plane ...)


@dataclass
class _Seg:
    a: np.ndarray
    b: np.ndarray
    kind: str  # frame | edge | grid | reference
    name: Optional[str]


class SnapEngine:
    """Snap targets of one view.  ``project`` maps a world point to display (x, y);
    ``to_plane`` maps display (x, y) to the world point on the drawing plane
    (None in 3D views); ``plane`` is the view's drawing plane."""

    def __init__(self, model: Model, plane: Plane, options: SnapOptions,
                 project: Callable[[Sequence[float]], Tuple[float, float]],
                 to_plane: Callable[[float, float], Optional[XYZ]],
                 previous: Optional[XYZ] = None, world_tol: float = 1e-6,
                 reference_lines: Sequence[Tuple[XYZ, XYZ]] = (),
                 depth: Optional[Callable[[Sequence[float]], float]] = None,
                 objects: Optional[Tuple[Optional[set], Optional[set], Optional[set]]] = None,
                 near: Optional[float] = None, px_scale: float = 1.0, plane_tol: Optional[float] = None,
                 tower: Optional[str] = None, same_level: bool = True):
        self.model = model
        # several towers: the grids and levels snapped to are the tower's (of the plan's
        # story - with the towers that have a story at its level -, else of the window);
        # all towers' when None
        if plane.kind == "plan" and plane.story in model.stories and model.multi_tower:
            ss = model.same_level_stories(plane.story) if same_level else [plane.story]
            tower = [model.tower_of_story(s) for s in ss]
        self.gmodel = model.view(tower)
        self.plane = plane
        self.opt = options
        self.project = project
        self.to_plane = to_plane
        self.previous = np.asarray(previous, float) if previous is not None else None
        self.world_tol = max(world_tol, 1e-9)
        # "in the plane of the window": the same band the window draws (D05)
        self.plane_tol = plane_tol if plane_tol is not None else PLANE_BAND * self.world_tol
        self.reference_lines = reference_lines
        self.depth = depth  # distance from the eye along the view direction (3D views)
        # (frames, areas, points) the window shows - None: all.  Hidden objects are no targets (D03)
        self.objects = objects or (None, None, None)
        # nothing at or behind the eye (or in front of the near clipping plane) is a target (D03)
        self.near = near
        # apertures are in screen (device independent) pixels; display coordinates may be
        # physical pixels (HiDPI): px_scale = physical / logical (D16)
        self.px_scale = max(float(px_scale or 1.0), 1e-6)
        self._segs: Optional[List[_Seg]] = None
        self._disp_cache: Dict[Tuple[float, float, float], np.ndarray] = {}
        self.override: Optional[str] = None  # one-shot snap override (Shift / Ctrl + right-click menu)

    # ------------------------------------------------------------ helpers
    @property
    def tolerance(self) -> float:
        """The snap aperture in display pixels."""
        return float(self.opt.tolerance_px) * self.px_scale

    def _in_plane(self, xyz) -> bool:
        return self.plane.contains(xyz, self.plane_tol)

    def _disp(self, xyz) -> np.ndarray:
        key = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        d = self._disp_cache.get(key)
        if d is None:
            d = self._disp_cache[key] = np.asarray(self.project(key), float)
        return d

    def _in_front(self, xyz) -> bool:
        if self.depth is None or self.near is None:
            return True
        return self.depth(xyz) > self.near

    def _shown(self, kind: str, name: Optional[str]) -> bool:
        fr, ar, pt = self.objects
        which = {"frame": fr, "edge": ar, "point": pt}.get(kind)
        return which is None or name is None or name in which

    def segments(self) -> List[_Seg]:
        """Line segments of the view: frames, shell edges, grid lines, reference lines."""
        if self._segs is not None:
            return self._segs
        m = self.model
        pts = m.points
        out: List[_Seg] = []
        for f in m.frames.values():
            if not self._shown("frame", f.name):
                continue
            a, b = np.asarray(pts[f.point_i].xyz, float), np.asarray(pts[f.point_j].xyz, float)
            if self.plane.kind == "3d" or (self._in_plane(a) and self._in_plane(b)):
                out.append(_Seg(a, b, "frame", f.name))
        for ar in m.areas.values():
            if not self._shown("edge", ar.name):
                continue
            n = len(ar.points)
            for k in range(n):
                a = np.asarray(pts[ar.points[k]].xyz, float)
                b = np.asarray(pts[ar.points[(k + 1) % n]].xyz, float)
                if self.plane.kind == "3d" or (self._in_plane(a) and self._in_plane(b)):
                    out.append(_Seg(a, b, "edge", ar.name))
        out += self._grid_segments()
        for a, b in self.reference_lines:
            out.append(_Seg(np.asarray(a, float), np.asarray(b, float), "reference", None))
        self._segs = out
        return out

    def _grid_segments(self) -> List[_Seg]:
        """Grid lines of the view as snap lines: in a plan / 3D the lines (and the
        arcs of cylindrical systems as short chords) of every system at the plane /
        every level, the vertical lines at the grid points in 3D; in an elevation the
        lines crossing it (Cartesian systems along the axes) and the levels."""
        from ..model.grids import grid_geometry
        m = self.gmodel
        pl = self.plane
        out: List[_Seg] = []
        if pl.kind in ("plan", "3d"):
            geo = grid_geometry(m)
            if not geo.lines and not geo.arcs:
                return []
            polys = geo.polylines(seg_deg=3.0)
            names = [n for _, _, n, _ in geo.lines] + [n for *_, n, _ in geo.arcs]
            if pl.kind == "plan":
                levels = [pl.value]
            else:
                levels = story_levels(m)
                if not levels:
                    return []
            for z in levels:
                for poly, name in zip(polys, names):
                    for a, b in zip(poly, poly[1:]):
                        out.append(_Seg(np.array([a[0], a[1], z]), np.array([b[0], b[1], z]), "grid", name))
            if pl.kind == "3d":
                z0, z1 = min(levels), max(levels)
                seen = set()
                for x, y in geo.points:
                    k = (round(x, 9), round(y, 9))
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(_Seg(np.array([x, y, z0]), np.array([x, y, z1]), "grid", f"{x:g}, {y:g}"))
            return out
        xs, ys = grid_ordinates(m)
        if not xs or not ys:
            return []
        ext = 0.1 * max(xs[-1] - xs[0], ys[-1] - ys[0], 1.0)
        if pl.kind in ("xz", "yz"):
            levels = story_levels(m)
            us = xs if pl.kind == "xz" else ys
            if not levels:
                return []
            z0, z1 = min(levels) - ext, max(levels) + ext
            for u in us:
                a, b = pl.to3((u, z0)), pl.to3((u, z1))
                out.append(_Seg(np.array(a), np.array(b), "grid", f"{'X' if pl.kind == 'xz' else 'Y'}={u:g}"))
            for z in levels:
                a, b = pl.to3((us[0] - ext, z)), pl.to3((us[-1] + ext, z))
                out.append(_Seg(np.array(a), np.array(b), "grid", f"Z={z:g}"))
        return out

    # --------------------------------------------------------- candidates
    def _point_candidates(self, cursor=None, tol: Optional[float] = None) -> List[SnapHit]:
        """Object points (joints, ends, midpoints, grid points, intersections, centres).
        With ``cursor`` the intersections are searched only among the lines passing
        within ``tol`` of it (an intersection near the cursor lies on two of them) (D15)."""
        m = self.model
        o = self.opt
        out: List[SnapHit] = []
        if o.joints:
            for p in m.points.values():
                if not self._shown("point", p.name):
                    continue
                if self.plane.kind == "3d" or self._in_plane(p.xyz):
                    out.append(SnapHit(tuple(p.xyz), "joint", p.name))
        if o.ends_midpoints:
            for s in self.segments():
                if s.kind in ("frame", "edge"):
                    mid = 0.5 * (s.a + s.b)
                    out.append(SnapHit(tuple(mid.tolist()), "midpoint", s.name))
                    if not o.joints:
                        out.append(SnapHit(tuple(s.a.tolist()), "end", s.name))
                        out.append(SnapHit(tuple(s.b.tolist()), "end", s.name))
        if o.grid:
            from .tools import grid_intersections
            m = self.gmodel
            xs, ys = grid_ordinates(m)
            gpts = grid_intersections(m)
            levels = story_levels(m)
            pl = self.plane
            if pl.kind == "plan":
                out += [SnapHit((x, y, pl.value), "grid") for x, y in gpts]
            elif pl.kind == "xz":
                out += [SnapHit((x, pl.value, z), "grid") for x in xs for z in levels]
            elif pl.kind == "yz":
                out += [SnapHit((pl.value, y, z), "grid") for y in ys for z in levels]
            else:
                out += [SnapHit((x, y, z), "grid") for x, y in gpts for z in levels]
        if o.intersections and self.plane.kind != "3d":
            out += self._intersections(cursor, tol)
        if o.centers:
            out += self._centers_of_areas()
        return out

    def _intersections(self, cursor=None, tol: Optional[float] = None) -> List[SnapHit]:
        """Crossings of segments in the plane (frames, edges, grid lines)."""
        segs = [s for s in self.segments() if s.kind in ("frame", "edge", "grid", "reference")]
        if cursor is not None and tol is not None:
            near = []
            for s in segs:
                a, b = self._disp(s.a), self._disp(s.b)
                lo, hi = np.minimum(a, b) - tol, np.maximum(a, b) + tol
                if not (lo[0] <= cursor[0] <= hi[0] and lo[1] <= cursor[1] <= hi[1]):
                    continue
                d = b - a
                L2 = float(d @ d)
                t = 0.0 if L2 <= 1e-12 else min(max(float((cursor - a) @ d) / L2, 0.0), 1.0)
                if float(np.linalg.norm(a + t * d - cursor)) <= tol:
                    near.append(s)
            segs = near
        out = []
        pl = self.plane
        to2 = pl.to2
        for i in range(len(segs)):
            a1, b1 = np.array(to2(segs[i].a)), np.array(to2(segs[i].b))
            for j in range(i + 1, len(segs)):
                if segs[i].kind == "grid" and segs[j].kind == "grid":
                    continue  # grid x grid are the grid intersections
                a2, b2 = np.array(to2(segs[j].a)), np.array(to2(segs[j].b))
                d1, d2 = b1 - a1, b2 - a2
                den = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(den) < 1e-12:
                    continue
                r = a2 - a1
                t = (r[0] * d2[1] - r[1] * d2[0]) / den
                u = (r[0] * d1[1] - r[1] * d1[0]) / den
                if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
                    p = a1 + t * d1
                    # skip shared end points (they are joints already)
                    if min(np.linalg.norm(p - q) for q in (a1, b1, a2, b2)) < self.world_tol:
                        continue
                    name = " x ".join(n for n in (segs[i].name, segs[j].name) if n)
                    out.append(SnapHit(tuple(pl.to3(tuple(p))), "intersection", name or None))
        return out

    # -------------------------------------------------------------- query
    # Snap priority (AutoCAD object snaps): a real object point inside the aperture always
    # wins over a line, a tracking line or an alignment, however close those are - the
    # cursor near a joint lands ON the joint.  Within a tier the nearest wins.
    TIER = {"joint": 0, "end": 0, "intersection": 0, "grid": 0, "midpoint": 0, "center": 0,
            "perpendicular": 1, "extension": 1, "alignment2": 1, "tangent": 1,
            "fine_grid": 2, "parallel": 2, "ortho": 2, "polar": 2, "reference": 2, "alignment": 2,
            "nearest": 3, "edge": 3}
    OVERRIDE_KINDS = {  # Shift / Ctrl + right-click menu: one snap for the next pick
        "endpoint": ("joint", "end"), "midpoint": ("midpoint",), "intersection": ("intersection", "grid"),
        "center": ("center",), "node": ("joint",), "perpendicular": ("perpendicular",),
        "extension": ("extension",), "parallel": ("parallel",), "nearest": ("nearest", "edge"),
        "grid": ("grid",), "none": ()}

    def find(self, x: float, y: float) -> List[SnapHit]:
        """All snap hits within the tolerance of display (x, y): object points first
        (joints, ends, midpoints, intersections, grid points, nearest of them first),
        then perpendicular / extension, then tracking and lines.  With a one-shot
        override (``override``) only that kind of snap is searched."""
        tol = self.tolerance
        cursor = np.array([x, y], float)
        hits: List[SnapHit] = []
        ov = getattr(self, "override", None)
        if ov is not None:
            return [h for h in self._override_hits(ov, cursor, tol) if self._in_front(h.xyz)]
        if self.opt.osnap:
            for h in self._point_candidates(cursor, tol):
                if not self._in_front(h.xyz):
                    continue
                d = float(np.linalg.norm(self._disp(h.xyz) - cursor))
                if d <= tol:
                    h.distance = d
                    hits.append(h)
            hits += self._line_hits(cursor, tol)
        hits += self._tracking_hits(cursor, tol)
        # only points inside the aperture: every distance is the snapped point projected back
        # to the screen.  (A perpendicular foot away from the cursor is offered only by the
        # explicit Perpendicular snap of the Shift / Ctrl + right-click menu.) (D02)
        hits = [h for h in hits if h.distance <= tol and self._in_front(h.xyz)]
        tier = lambda h: self.TIER.get(h.kind, 4)  # noqa: E731
        sub = {"joint": 0, "end": 1, "grid": 2, "intersection": 3, "midpoint": 4, "center": 5}
        # the joints / ends (drawn as squares) are what one clicks; a midpoint or centre
        # that happens to sit near them on the screen (3D: the middle of a beam of another
        # level in front of a base joint) wins only when it is clearly nearer the cursor
        bias = {"grid": 1.0, "intersection": 1.0, "midpoint": 4.0, "center": 4.0}
        rank_d = lambda h: h.distance + (bias.get(h.kind, 0.0) if tier(h) == 0 else 0.0)  # noqa: E731
        hits.sort(key=lambda h: (tier(h), round(rank_d(h), 1), sub.get(h.kind, 9)))
        if self.depth is not None and hits and tier(hits[0]) == 0:
            # 3D: points of different levels can project onto the same pixel - only then
            # (within half a pixel) the one in front is taken; a point nearer the cursor on
            # the screen always wins, however far behind it is (D04).  N cycles through them.
            best = rank_d(hits[0])
            near = [h for h in hits if tier(h) == 0 and rank_d(h) <= best + 0.5 * self.px_scale]
            if len(near) > 1:
                front = min(near, key=lambda h: self.depth(h.xyz))
                hits.remove(front)
                hits.insert(0, front)
        # one hit per location (a joint is also a grid point / an end): keep the best ranked
        out, seen = [], set()
        for h in hits:
            if self.TIER.get(h.kind, 4) == 0:
                key = tuple(round(c / self.world_tol) for c in h.xyz)
                if key in seen:
                    continue
                seen.add(key)
            out.append(h)
        return out

    def _override_hits(self, ov: str, cursor, tol) -> List[SnapHit]:
        kinds = self.OVERRIDE_KINDS.get(ov, ())
        if not kinds:
            return []
        o = self.opt
        saved = {k: getattr(o, k) for k in SNAP_KEYS + ("osnap",)}
        try:  # switch on what the override needs, whatever the settings
            o.select_all(False)
            o.osnap = True
            if "joint" in kinds:
                o.joints = True
            if "end" in kinds or "midpoint" in kinds:
                o.ends_midpoints = True
                if "end" in kinds:
                    o.joints = False  # ends are then listed as "end"
            if "intersection" in kinds:
                o.intersections = True
            if "center" in kinds:
                o.centers = True
            if "grid" in kinds:
                o.grid = True
            if "perpendicular" in kinds:
                o.perpendicular = True
            if "extension" in kinds:
                o.extensions = True
            if "parallel" in kinds:
                o.parallels = True
            if "nearest" in kinds:
                o.lines = True
            self._segs = None
            big = 3.0 * tol  # a chosen snap reaches farther (CAD: the aperture of the override)
            hits = []
            for h in self._point_candidates(cursor, big):
                d = float(np.linalg.norm(self._disp(h.xyz) - cursor))
                if d <= big and h.kind in kinds:
                    h.distance = d
                    hits.append(h)
            hits += [h for h in self._line_hits(cursor, big) if h.kind in kinds]
            hits += [h for h in self._tracking_hits(cursor, big) if h.kind in kinds]
        finally:
            for k, v in saved.items():
                setattr(o, k, v)
            self._segs = None
        hits.sort(key=lambda h: h.distance)
        return hits

    def _centers_of_areas(self) -> List[SnapHit]:
        """Geometric centres of the shell areas (and of closed frame loops are not searched)."""
        out = []
        m = self.model
        for ar in m.areas.values():
            xyz = np.array([m.points[q].xyz for q in ar.points], float)
            if self.plane.kind != "3d" and not all(self._in_plane(p) for p in xyz):
                continue
            out.append(SnapHit(tuple(xyz.mean(axis=0).tolist()), "center", ar.name))
        return out

    def _clip_to_front(self, s: _Seg) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        """The part of the segment in front of the eye: (a, b, t0, t1) with a, b its
        ends and t0, t1 their parameters on the original segment; None if all behind."""
        if self.depth is None or self.near is None:
            return s.a, s.b, 0.0, 1.0
        da, db = self.depth(s.a), self.depth(s.b)
        n = self.near
        if da <= n and db <= n:
            return None
        if da > n and db > n:
            return s.a, s.b, 0.0, 1.0
        tc = (n - da) / (db - da) if db != da else 0.0
        tc = min(max(tc, 0.0), 1.0)
        if da <= n:
            t0, t1 = tc + 1e-9, 1.0
        else:
            t0, t1 = 0.0, tc - 1e-9
        d = s.b - s.a
        return s.a + t0 * d, s.a + t1 * d, t0, t1

    def _nearest_on_segment(self, s: _Seg, cursor, extend: bool = False):
        """(t, world point, screen distance) of the point of the segment's line whose
        projection is nearest the cursor.  Perspective does not keep ratios along a
        line, so the screen parameter is mapped back to the world parameter (a line
        projects through a 1D projective map t -> s, fixed by the two ends and the
        midpoint) and the distance is measured by projecting the world point back (D01)."""
        clip = self._clip_to_front(s)
        if clip is None:
            return None
        ca, cb, t0, t1 = clip
        A, B = self._disp(ca), self._disp(cb)
        D = B - A
        L2 = float(D @ D)
        if L2 <= 1e-12:
            return None
        sp = float((cursor - A) @ D) / L2  # screen parameter on the clipped segment
        M = self._disp(0.5 * (ca + cb))
        sm = float((M - A) @ D) / L2  # where the world midpoint lands on the screen (0.5 if affine)
        k = (1.0 / sm - 1.0) if 1e-9 < sm < 1.0 - 1e-9 else 1.0
        if not extend or t0 > 0.0 or t1 < 1.0:
            sp = min(max(sp, 0.0), 1.0)  # (a line cut at the eye is not extended)
        den = 1.0 - sp + k * sp
        if den <= 1e-12:  # past the vanishing point: no point of the line projects there
            return None
        u = k * sp / den  # world parameter on the clipped segment
        t = t0 + u * (t1 - t0)
        p = s.a + t * (s.b - s.a)
        if not self._in_front(p):
            return None
        return t, p, float(np.linalg.norm(self._disp(p) - cursor))

    def _nearest_on_segment_display(self, s: _Seg, cursor, extend: bool = False):
        """(t, screen distance) - see :meth:`_nearest_on_segment`."""
        res = self._nearest_on_segment(s, cursor, extend)
        return None if res is None else (res[0], res[2])

    def _line_hits(self, cursor, tol) -> List[SnapHit]:
        o = self.opt
        out: List[SnapHit] = []
        if not (o.lines or o.edges or o.perpendicular or o.extensions or o.reference):
            return out
        for s in self.segments():
            if s.kind == "grid" and not o.lines:
                continue
            if s.kind == "reference" and not o.reference:
                continue
            if s.kind in ("frame", "edge") and not (o.lines or (o.edges and s.kind == "edge")
                                                   or o.perpendicular or o.extensions):
                if not (s.kind == "edge" and o.edges):
                    continue
            res = self._nearest_on_segment(s, cursor, extend=o.extensions and s.kind in ("frame", "edge"))
            if res is None:
                continue
            t, p, d = res
            if d > tol:
                continue
            line = (tuple(s.a.tolist()), tuple(s.b.tolist()))
            if o.perpendicular and self.previous is not None and s.kind in ("frame", "edge", "grid", "reference"):
                # foot of the perpendicular from the previous point onto the line
                dv = s.b - s.a
                tp = float((self.previous - s.a) @ dv) / float(dv @ dv)
                foot = s.a + tp * dv
                degenerate = float(np.linalg.norm(self._disp(foot) - self._disp(self.previous))) <= tol
                if (0.0 <= tp <= 1.0 or s.kind == "grid") and not degenerate and self._in_front(foot):
                    out.append(SnapHit(tuple(foot.tolist()), "perpendicular", s.name,
                                       float(np.linalg.norm(self._disp(foot) - cursor)),
                                       track=(tuple(self.previous.tolist()), tuple(foot.tolist())), line=line))
            if t < -1e-9 or t > 1 + 1e-9:
                if o.extensions:
                    out.append(SnapHit(tuple(p.tolist()), "extension", s.name, d,
                                       track=(tuple((s.a if t > 1 else s.b).tolist()), tuple(p.tolist())),
                                       line=line))
                continue
            if s.kind == "reference":
                out.append(SnapHit(tuple(p.tolist()), "reference", None, d, line=line))
            elif s.kind == "edge" and (o.edges or o.lines):
                out.append(SnapHit(tuple(p.tolist()), "edge" if o.edges else "nearest", s.name, d, line=line))
            elif o.lines:
                out.append(SnapHit(tuple(p.tolist()), "nearest", s.name, d, line=line))
        return out

    def _tracking_hits(self, cursor, tol) -> List[SnapHit]:
        """Ortho / polar / parallel tracking and fine grid: need a point on the plane."""
        o = self.opt
        out: List[SnapHit] = []
        if self.plane.kind == "3d":
            return out
        xyz = self.to_plane(float(cursor[0]), float(cursor[1]))
        if xyz is None:
            return out
        pl = self.plane
        u, v = pl.to2(xyz)
        if o.fine_grid and pl.kind == "plan" and o.fine_spacing > 0:
            fs = o.fine_spacing
            p = pl.to3((round(u / fs) * fs, round(v / fs) * fs))
            d = float(np.linalg.norm(self._disp(p) - cursor))
            if d <= tol:
                out.append(SnapHit(tuple(p), "fine_grid", None, d))
        if o.alignment:
            out += self._alignment_hits(cursor, tol, u, v)
        if self.previous is None:
            return [h for h in out if h is not None and h.distance <= tol]
        pu, pv = pl.to2(tuple(self.previous.tolist()))
        angles: List[float] = []
        if o.ortho:
            angles += [0.0, 90.0, 180.0, 270.0]
        if o.polar and o.polar_angle > 0:
            k = 0.0
            while k < 360.0:
                angles.append(k)
                k += o.polar_angle
        if o.parallels:
            for s in self.segments():
                if s.kind not in ("frame", "edge"):
                    continue
                res = self._nearest_on_segment_display(s, cursor)
                if res is None or res[1] > 4 * tol:
                    continue
                a2, b2 = pl.to2(tuple(s.a.tolist())), pl.to2(tuple(s.b.tolist()))
                ang = math.degrees(math.atan2(b2[1] - a2[1], b2[0] - a2[0]))
                out.append(self._track_hit(pu, pv, u, v, ang, "parallel", s.name, cursor))
        for ang in angles:
            out.append(self._track_hit(pu, pv, u, v, ang, "polar" if ang % 90 else "ortho", None, cursor))
        out = [h for h in out if h is not None and h.distance <= tol]
        return out

    ALIGN_RADIUS_PX = 450.0  # joints this close (on screen) give alignment lines
    ALIGN_MAX_ANCHORS = 60

    def _alignment_hits(self, cursor, tol, u, v) -> List[SnapHit]:
        """Alignment lines with nearby joints (and the previous point), CAD object
        snap tracking: same u (vertical on the screen) or same v, or both."""
        pl = self.plane
        anchors = []
        for p in self.model.points.values():
            if not self._in_plane(p.xyz):
                continue
            d = float(np.linalg.norm(self._disp(p.xyz) - cursor))
            if tol < d <= self.ALIGN_RADIUS_PX * self.px_scale:
                anchors.append((d, p.name, pl.to2(p.xyz), p.xyz))
        if self.previous is not None:
            pxyz = tuple(self.previous.tolist())
            d = float(np.linalg.norm(self._disp(pxyz) - cursor))
            if d > tol:
                anchors.append((0.0, "previous point", pl.to2(pxyz), pxyz))  # 0: preferred
        anchors.sort(key=lambda a: a[0])
        anchors = anchors[: self.ALIGN_MAX_ANCHORS]
        best_u = best_v = None  # (screen distance, anchor, point)
        for d, name, (au, av), axyz in anchors:
            for which in ("u", "v"):
                q = (au, v) if which == "u" else (u, av)
                p3 = pl.to3(q)
                dd = float(np.linalg.norm(self._disp(p3) - cursor))
                if dd > tol:
                    continue
                cand = (dd, name, axyz, q)
                if which == "u" and (best_u is None or (dd, d) < (best_u[0], 0)):
                    best_u = cand
                if which == "v" and (best_v is None or (dd, d) < (best_v[0], 0)):
                    best_v = cand
        out: List[SnapHit] = []
        if best_u is not None and best_v is not None and best_u[1] != best_v[1]:
            q = (best_u[3][0], best_v[3][1])
            p3 = tuple(pl.to3(q))
            dd = float(np.linalg.norm(self._disp(p3) - cursor))
            if dd <= tol:
                out.append(SnapHit(p3, "alignment2", f"{best_u[1]} + {best_v[1]}", dd,
                                   track=(tuple(best_u[2]), p3), track2=(tuple(best_v[2]), p3)))
        for b in (best_u, best_v):
            if b is not None:
                p3 = tuple(pl.to3(b[3]))
                out.append(SnapHit(p3, "alignment", b[1], b[0], track=(tuple(b[2]), p3)))
        return out

    def _track_hit(self, pu, pv, u, v, ang_deg, kind, name, cursor) -> Optional[SnapHit]:
        pl = self.plane
        d = np.array([math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))])
        t = (u - pu) * d[0] + (v - pv) * d[1]
        if t <= 0:
            return None
        q = (pu + t * d[0], pv + t * d[1])
        p = pl.to3(q)
        dist = float(np.linalg.norm(self._disp(p) - cursor))
        return SnapHit(tuple(p), kind, name, dist, track=(tuple(pl.to3((pu, pv))), tuple(p)),
                       text=f"{ang_deg:g}°")

    # ------------------------------------------------------------ increments
    LINE_KINDS = ("nearest", "edge", "extension", "reference", "perpendicular")
    TRACK_KINDS = ("ortho", "polar", "parallel")

    def apply_increments(self, hit: SnapHit) -> SnapHit:
        """Length / angle increments relative to the previous point (CAD style) (D10):

        * exact object points (joints, ends, midpoints, grid, intersections, centres)
          are never moved off their object;
        * a point on a line (nearest / edge / extension / reference) stays on that line:
          the length steps to the nearest increment along it (angle comes from the line);
        * a tracking point (ortho / polar / parallel) keeps its direction: length only;
        * a free point (and the fine grid) takes both length and angle steps in the plane.
        The returned hit says what was applied (its text)."""
        o = self.opt
        if self.previous is None or not (o.length_increments or o.angle_increments):
            return hit
        prev = self.previous
        steps = [st for st in (o.length_steps or []) if st > 0] if o.length_increments else []
        if hit.kind in ("perpendicular",):
            return hit  # the foot is an exact point
        if hit.kind in self.LINE_KINDS and hit.line is not None:
            if not steps:
                return hit
            a, b = np.asarray(hit.line[0], float), np.asarray(hit.line[1], float)
            d = b - a
            dd = float(d @ d)
            if dd <= 1e-18:
                return hit
            p = np.asarray(hit.xyz, float)
            L = float(np.linalg.norm(p - prev))
            t_hit = float((p - a) @ d) / dd
            best = None
            for st in steps:
                for Lk in (math.floor(L / st) * st, math.ceil(L / st) * st):
                    if Lk <= 0:
                        continue
                    # |a + t d - prev| = Lk  ->  dd t^2 + 2 (a - prev).d t + |a - prev|^2 - Lk^2 = 0
                    w = a - prev
                    B, C = 2.0 * float(w @ d), float(w @ w) - Lk * Lk
                    disc = B * B - 4.0 * dd * C
                    if disc < 0:
                        continue
                    for t in ((-B - math.sqrt(disc)) / (2 * dd), (-B + math.sqrt(disc)) / (2 * dd)):
                        if hit.kind != "extension" and not (-1e-9 <= t <= 1 + 1e-9):
                            continue
                        q = a + t * d
                        cost = abs(t - t_hit)
                        if best is None or cost < best[0]:
                            best = (cost, q, Lk)
            if best is None:
                return hit
            q = best[1]
            return SnapHit(tuple(q.tolist()), "increment", hit.target, hit.distance,
                           track=(tuple(prev.tolist()), tuple(q.tolist())), line=hit.line,
                           text=f"L={best[2]:g} on {KIND_LABELS.get(hit.kind, hit.kind).lower()}")
        if hit.kind in self.TRACK_KINDS:
            if not steps:
                return hit
            p = np.asarray(hit.xyz, float)
            v = p - prev
            L = float(np.linalg.norm(v))
            if L <= 0:
                return hit
            Lk = min((round(L / st) * st for st in steps), key=lambda x: abs(x - L))
            if Lk <= 0:
                return hit
            q = prev + v / L * Lk
            return SnapHit(tuple(q.tolist()), "increment", hit.target, hit.distance,
                           track=(tuple(prev.tolist()), tuple(q.tolist())), text=f"L={Lk:g}  {hit.text}".strip())
        if hit.kind not in ("free", "fine_grid") or self.plane.kind == "3d":
            return hit  # exact targets win over increments
        pl = self.plane
        pu, pv = pl.to2(tuple(prev.tolist()))
        u, v = pl.to2(hit.xyz)
        du, dv = u - pu, v - pv
        L = math.hypot(du, dv)
        if L <= 0:
            return hit
        ang = math.degrees(math.atan2(dv, du))
        changed = False
        if o.angle_increments and o.angle_increment > 0:
            ang = round(ang / o.angle_increment) * o.angle_increment
            changed = True
        if steps:
            best = min((abs(L - round(L / st) * st), round(L / st) * st) for st in steps)
            if best[1] > 0:
                L = best[1]
                changed = True
        if not changed:
            return hit
        q = (pu + L * math.cos(math.radians(ang)), pv + L * math.sin(math.radians(ang)))
        p = pl.to3(q)
        return SnapHit(tuple(p), "increment", None, hit.distance, track=(tuple(pl.to3((pu, pv))), tuple(p)),
                       text=f"L={L:g}  {ang:g}°")


def reference_lines_for(model: Model, plane: Plane, directions: Tuple[bool, bool, bool] = (True, True, True),
                        only_with_joint: bool = True) -> List[Tuple[XYZ, XYZ]]:
    """CYPE 3D style reference lines: through every joint, parallel to the global
    axes, over the model extent.  In a plan or elevation only the lines in that
    plane are returned."""
    pts = [np.asarray(p.xyz, float) for p in model.points.values()]
    if not pts:
        return []
    xyz = np.array(pts)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ext = 0.15 * max(float(np.max(hi - lo)), 1.0)
    lo, hi = lo - ext, hi + ext
    out: List[Tuple[XYZ, XYZ]] = []
    seen = set()
    tol = max(model.options.merge_tolerance, 1e-6)
    for p in pts:
        if plane.kind != "3d" and not plane.contains(p, tol):
            continue
        for axis in range(3):
            if not directions[axis]:
                continue
            if plane.kind == "plan" and axis == 2:
                continue
            if plane.kind == "xz" and axis == 1:
                continue
            if plane.kind == "yz" and axis == 0:
                continue
            key = (axis,) + tuple(round(float(p[k]) / tol) for k in range(3) if k != axis)
            if key in seen:
                continue
            seen.add(key)
            a, b = p.copy(), p.copy()
            a[axis], b[axis] = lo[axis], hi[axis]
            out.append((tuple(a.tolist()), tuple(b.tolist())))
    return out
