"""The :class:`Model` container: everything an engineer defines.

A model is a plain in-memory object graph.  It knows nothing about finite
elements; :mod:`structstudio.mesh` turns it into an analysis model and
:mod:`structstudio.engine` solves that.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..units import UnitSystem
from .base import ModelError, convert_dataclass, to_dict
from .loads import (ComboItem, LinearBucklingCase, LinearStaticCase, LoadCombination, LoadPattern, MassSource,
                    ModalCase, PDeltaCase, ResponseSpectrumCase, ResponseSpectrumFunction, SpectrumLoad,
                    TimeHistoryFunction)
from .objects import (Area, AreaLoad, Diaphragm, Frame, FrameLoad, GridLine, GridSystem, GroundDisplacement, Group,
                      LineSpring, PanelLoad, PanelLoadCase, Point, PointLoad, Story, Tendon, TendonLoad,
                      TendonPoint, FREE)
from .properties import FrameModifiers, FrameSection, Material, RebarSize, ShellModifiers, ShellSection
from .nonlinear import (NONLINEAR_TYPES, FiberSectionDef, FrameHingeAssignment, FrameNonlinearProperty,
                        HingeProperty, LinkDOF, LinkObject, LinkProperty, NonlinearMaterial, NonlinearStaticCase,
                        TimeHistoryCase, TimeHistoryLoad)

SCHEMA_VERSION = "2.2"  # 2.1: area loads are load panels made from the areas (PanelLoad.area)
# 2.2: the mesh refinement check is off by default (files before 2.2 saved it on by default)

def _version_tuple(text) -> tuple:
    try:
        return tuple(int(x) for x in str(text).split("."))
    except ValueError:
        return (0,)


_CASE_TYPES = {
    "LinearStatic": LinearStaticCase,
    "Modal": ModalCase,
    "ResponseSpectrum": ResponseSpectrumCase,
    "PDelta": PDeltaCase,
    "Buckling": LinearBucklingCase,
    "NonlinearStatic": NonlinearStaticCase,
    "TimeHistory": TimeHistoryCase,
}

_TYPE_REGISTRY = {
    cls.__name__: cls for cls in (
        Material, RebarSize, FrameModifiers, FrameSection, ShellModifiers, ShellSection, Point, Frame, Area, PointLoad,
        GroundDisplacement, Tendon, TendonLoad, TendonPoint, PanelLoad, PanelLoadCase,
        FrameLoad, AreaLoad, Story, GridLine, GridSystem, Diaphragm, Group, LineSpring, LoadPattern, MassSource,
        ResponseSpectrumFunction, TimeHistoryFunction, LinearStaticCase, ModalCase, ResponseSpectrumCase,
        PDeltaCase, LinearBucklingCase, ComboItem, LoadCombination, SpectrumLoad,
    ) + NONLINEAR_TYPES
}


@dataclass
class AnalysisOptions:
    active_dofs: Tuple[bool, bool, bool, bool, bool, bool] = (True,) * 6
    floor_mesh_max_size: Optional[float] = None  # model length units
    wall_mesh_max_size: Optional[float] = None
    merge_tolerance: float = 1e-3  # model length units (points closer than this are the same)
    auto_line_constraints: bool = True
    default_stations: int = 3
    gravity: Optional[float] = None  # model length/s^2; None = standard 9.80665 m/s^2 (ETABS uses 386.4 in/s^2)
    auto_base_restraints: bool = False  # ETABS: mesh-generated nodes at the base level are fixed
    # the slab inside a column's cross-section moves with the column (rigid zone, as
    # SAFE): without it the column-slab joint is one node whose rotational
    # stiffness falls as the mesh is refined, so results never converge
    slab_column_rigid_zone: bool = True
    # frames that do not set their own (Frame.auto_end_offsets / rigid_factor = None):
    # end offsets from the sizes of the connecting members, and the fraction of them
    # taken as rigid - 0.5 for concrete frames, 0 for steel (CSI: steel frames should
    # never use a fully rigid zone unless the panel zone is modelled)
    frame_auto_end_offsets: bool = True
    rigid_zone_factor_concrete: float = 0.5
    rigid_zone_factor_steel: float = 0.0
    rigid_zone_factor_other: float = 0.0
    # mesh refinement check of automatically meshed shells (api.analyze)
    # mesh refinement check (h-convergence): off by default - like ETABS the automatic mesh is
    # used as it is (the check solves the model several times on finer meshes: much slower)
    mesh_refine_check: bool = False
    mesh_refine_tolerance: float = 0.02  # estimated error of the largest shell displacement
    # Analyze > Check Mesh finds the fine mesh once and keeps it here; Analyze > Use Fine Mesh
    # then analyses with it (never coarser than the automatic mesh: factor <= 1)
    fine_mesh_factor: Optional[float] = None  # automatic element size x this (1, 0.5, 0.25 ...)
    fine_mesh_report: str = ""  # what the check found (shown in the Fine mesh help)
    fine_mesh_signature: str = ""  # geometry the check was run on (a later change is reported)
    use_fine_mesh: bool = False
    # Grid system: "story" = X, Y grid lines + stories, "xyz" = X, Y, Z grid
    # lines.  In "xyz" mode the Z grid lines are the levels and the stories mirror them, so
    # diaphragms, story drifts and all story based tools keep working.
    grid_mode: str = "story"


@dataclass
class ProjectInfo:
    name: str = "Untitled"
    engineer: str = ""
    company: str = ""
    notes: str = ""


class Model:
    """Object-based structural model.

    All numeric values are stored in ``self.units`` (default kN-m).  Use the
    ``add_*`` helpers or assign objects directly to the dictionaries.
    """

    def __init__(self, name: str = "Untitled", units: UnitSystem | str = "kN m"):
        self.units: UnitSystem = units if isinstance(units, UnitSystem) else UnitSystem.parse(units)
        self.info = ProjectInfo(name=name)
        self.materials: Dict[str, Material] = {}
        self.rebar_sizes: Dict[str, RebarSize] = {}
        self.frame_sections: Dict[str, FrameSection] = {}
        self.shell_sections: Dict[str, ShellSection] = {}
        self.stories: Dict[str, Story] = {}
        self.grid_systems: Dict[str, GridSystem] = {}
        self.grid_lines: List[GridLine] = []
        self.diaphragms: Dict[str, Diaphragm] = {}
        self.points: Dict[str, Point] = {}
        self.frames: Dict[str, Frame] = {}
        self.areas: Dict[str, Area] = {}
        self.groups: Dict[str, Group] = {}
        self.line_springs: Dict[str, LineSpring] = {}
        self.tendons: Dict[str, Tendon] = {}
        self.panel_loads: Dict[str, PanelLoad] = {}
        self.load_patterns: Dict[str, LoadPattern] = {}
        self.mass_sources: Dict[str, MassSource] = {"MsSrc1": MassSource()}
        self.functions: Dict[str, ResponseSpectrumFunction | TimeHistoryFunction] = {}
        self.load_cases: Dict[str, object] = {}
        self.load_combinations: Dict[str, LoadCombination] = {}
        # nonlinear analysis definitions
        self.nonlinear_materials: Dict[str, NonlinearMaterial] = {}
        self.fiber_sections: Dict[str, FiberSectionDef] = {}
        self.frame_nonlinear: Dict[str, FrameNonlinearProperty] = {}
        self.hinge_properties: Dict[str, HingeProperty] = {}
        self.link_properties: Dict[str, LinkProperty] = {}
        self.links: Dict[str, LinkObject] = {}
        self.options = AnalysisOptions()
        # units each kind of value is shown / typed in (Display Units form, e.g. Thai
        # legacy: section dimensions cm, areas cm^2, deflections mm, strengths ksc);
        # item -> unit label, missing = the consistent units above.  Display only.
        self.display_units: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}

    # ------------------------------------------------------------------ names
    def _next_name(self, prefix: str, table: dict) -> str:
        n = self._counters.get(prefix, 0)
        while True:
            n += 1
            name = f"{prefix}{n}"
            if name not in table:
                self._counters[prefix] = n
                return name

    # ------------------------------------------------------------- properties
    def add_material(self, name: str, E: float, nu: float = 0.2, weight_density: float = 0.0,
                     type: str = "Concrete", **kw) -> Material:
        m = Material(name=name, type=type, E=E, nu=nu, weight_density=weight_density, **kw)
        self.materials[name] = m
        return m

    def add_frame_section(self, section: FrameSection) -> FrameSection:
        self.frame_sections[section.name] = section
        return section

    def rename_frame_section(self, old: str, new: str) -> None:
        """Rename a frame section and every reference to it (frames, nonprismatic
        segments, fiber sections)."""
        if old == new or old not in self.frame_sections:
            return
        if new in self.frame_sections:
            raise ModelError("DUPLICATE_NAME", f"A frame section named {new!r} already exists")
        table = {}
        for k, v in self.frame_sections.items():  # keep the order of the table
            if k == old:
                v.name = new
                table[new] = v
            else:
                table[k] = v
        self.frame_sections = table
        for f in self.frames.values():
            if f.section == old:
                f.section = new
        for sec in self.frame_sections.values():
            for row in sec.nonprismatic or []:
                for key in ("start", "end"):
                    if row.get(key) == old:
                        row[key] = new
        for fs in getattr(self, "fiber_sections", {}).values():
            if getattr(fs, "frame_section", None) == old:
                fs.frame_section = new

    def frame_section_users(self, name: str) -> List[str]:
        """Frames using a frame section, directly or through a nonprismatic one."""
        names = {name} | {k for k, s in self.frame_sections.items()
                          if any(name in (r.get("start"), r.get("end")) for r in s.nonprismatic or [])}
        return [f.name for f in self.frames.values() if f.section in names]

    def add_shell_section(self, section: ShellSection) -> ShellSection:
        self.shell_sections[section.name] = section
        return section

    # ---------------------------------------------------------------- building
    def add_story(self, name: str, elevation: float, height: float = 0.0, **kw) -> Story:
        s = Story(name=name, elevation=elevation, height=height, **kw)
        self.stories[name] = s
        return s

    def set_stories(self, story_heights: Sequence[Tuple[str, float]], base_name: str = "Base",
                    base_elevation: float = 0.0) -> List[Story]:
        """Define stories bottom-up from (name, height) pairs."""
        self.stories.clear()
        self.add_story(base_name, base_elevation, 0.0)
        z = base_elevation
        out = []
        for name, h in story_heights:
            z += h
            out.append(self.add_story(name, z, h))
        return out

    # ------------------------------------------------------- grid system
    GRID_MODES = ("story", "xyz")

    @property
    def grid_mode(self) -> str:
        return self.options.grid_mode if self.options.grid_mode in self.GRID_MODES else "story"

    def z_grid_lines(self) -> List[GridLine]:
        return sorted((g for g in self.grid_lines if g.direction == "Z"), key=lambda g: g.ordinate)

    def levels(self) -> List[Tuple[str, float]]:
        """(name, elevation) of the levels bottom-up: stories (ETABS mode) or Z grid
        lines (SAP2000 mode)."""
        if self.grid_mode == "xyz":
            z = self.z_grid_lines()
            if z:
                return [(g.name, g.ordinate) for g in z]
        return [(s.name, s.elevation) for s in self.stories_bottom_up()]

    def set_grid_mode(self, mode: str) -> None:
        """Switch between the X-Y-Story and X-Y-Z grid systems.
        story -> xyz: one Z grid line per story (same names); xyz -> story: the
        stories are rebuilt from the Z grid lines and the Z lines are removed."""
        if mode not in self.GRID_MODES:
            raise ModelError("INVALID_GRID_MODE", f"Grid mode must be one of {self.GRID_MODES}")
        if mode == self.grid_mode and (mode == "story" or self.z_grid_lines()):
            return
        if mode == "xyz":
            system = next(iter(self.grid_systems), "G1")
            if system not in self.grid_systems:
                self.grid_systems[system] = GridSystem(name=system)
            self.grid_lines = [g for g in self.grid_lines if g.direction != "Z"]
            for s in self.stories_bottom_up():
                self.grid_lines.append(GridLine(system=system, name=s.name, direction="Z", ordinate=s.elevation))
        else:
            z = self.z_grid_lines()
            if z:
                self._stories_from_levels([(g.name, g.ordinate) for g in z])
            self.grid_lines = [g for g in self.grid_lines if g.direction != "Z"]
        self.options.grid_mode = mode

    def set_z_grid(self, levels: Sequence[Tuple[str, float]]) -> None:
        """SAP2000 mode: replace the Z grid lines (name, z) and rebuild the stories."""
        levels = sorted(((str(n), float(z)) for n, z in levels), key=lambda t: t[1])
        names = [n for n, _ in levels]
        if len(set(names)) != len(names):
            raise ModelError("DUPLICATE_NAME", "Z grid line names must be unique")
        system = next(iter(self.grid_systems), "G1")
        if system not in self.grid_systems:
            self.grid_systems[system] = GridSystem(name=system)
        self.grid_lines = [g for g in self.grid_lines if g.direction != "Z"] + [
            GridLine(system=system, name=n, direction="Z", ordinate=z) for n, z in levels]
        self._stories_from_levels(levels)

    def _stories_from_levels(self, levels: Sequence[Tuple[str, float]]) -> None:
        levels = sorted(levels, key=lambda t: t[1])
        old = {s.name: s for s in self.stories.values()}
        self.stories.clear()
        prev = None
        for name, z in levels:
            kw = {}
            if name in old:
                kw = {"is_master": old[name].is_master, "similar_to": old[name].similar_to, "color": old[name].color}
            self.add_story(name, z, 0.0 if prev is None else z - prev, **kw)
            prev = z
        self.reassign_stories()

    def reassign_stories(self) -> None:
        """Re-derive the story of every point / frame / area from its elevation."""
        tol = self.options.merge_tolerance
        for p in self.points.values():
            s = self.story_at(p.z, tol)
            p.story = s.name if s else None
        for f in self.frames.values():
            top = max(self.points[f.point_i].z, self.points[f.point_j].z)
            s = self.story_at(top, tol)
            f.story = s.name if s else f.story
        for a in self.areas.values():
            top = max(self.points[n].z for n in a.points)
            s = self.story_at(top, tol)
            a.story = s.name if s else a.story

    def stories_bottom_up(self) -> List[Story]:
        return sorted(self.stories.values(), key=lambda s: s.elevation)

    def story_below(self, story: str) -> Optional[Story]:
        ordered = self.stories_bottom_up()
        for k, s in enumerate(ordered):
            if s.name == story:
                return ordered[k - 1] if k > 0 else None
        return None

    def story_at(self, z: float, tol: float = 1e-6) -> Optional[Story]:
        for s in self.stories.values():
            if abs(s.elevation - z) <= tol:
                return s
        return None

    def add_diaphragm(self, name: str, rigid: bool = True) -> Diaphragm:
        d = Diaphragm(name, rigid)
        self.diaphragms[name] = d
        return d

    # ----------------------------------------------------------------- objects
    def add_point(self, x: float, y: float, z: float, name: Optional[str] = None, merge: bool = True,
                  **kw) -> Point:
        """Add a point; if ``merge`` an existing point at the same location is returned."""
        if merge:
            existing = self.find_point(x, y, z)
            if existing is not None:
                for k, v in kw.items():
                    setattr(existing, k, v)
                return existing
        name = name or self._next_name("P", self.points)
        if name in self.points:
            raise ModelError("DUPLICATE_NAME", f"Point {name!r} already exists", f"points.{name}")
        p = Point(name=name, x=float(x), y=float(y), z=float(z), **kw)
        if p.story is None:
            s = self.story_at(z, self.options.merge_tolerance)
            p.story = s.name if s else None
        self.points[name] = p
        return p

    def find_point(self, x: float, y: float, z: float, tol: Optional[float] = None) -> Optional[Point]:
        tol = self.options.merge_tolerance if tol is None else tol
        for p in self.points.values():
            if abs(p.x - x) <= tol and abs(p.y - y) <= tol and abs(p.z - z) <= tol:
                return p
        return None

    def add_frame(self, point_i, point_j, section: str, name: Optional[str] = None, type: Optional[str] = None,
                  **kw) -> Frame:
        """Add a frame between two points (names, Point objects or xyz tuples)."""
        if ("end_offset_i" in kw or "end_offset_j" in kw) and "auto_end_offsets" not in kw:
            kw["auto_end_offsets"] = False  # offsets given: they are the offsets
        pi = self._resolve_point(point_i)
        pj = self._resolve_point(point_j)
        if pi.name == pj.name:
            raise ModelError("ZERO_LENGTH", "Frame end points coincide")
        if type == "Null" and section not in self.frame_sections:
            section = section or "None"  # CSI null line: no section needed
        elif section not in self.frame_sections:
            raise ModelError("UNKNOWN_SECTION", f"Frame section {section!r} is not defined")
        if type is None:
            dx, dy, dz = pj.x - pi.x, pj.y - pi.y, pj.z - pi.z
            horiz = (dx * dx + dy * dy) ** 0.5
            type = "Column" if horiz < 1e-9 * max(1.0, abs(dz)) else ("Beam" if abs(dz) < 1e-9 * max(1.0, horiz) else "Brace")
        name = name or self._next_name("B" if type == "Beam" else "C" if type == "Column" else "D", self.frames)
        if name in self.frames:
            raise ModelError("DUPLICATE_NAME", f"Frame {name!r} already exists", f"frames.{name}")
        f = Frame(name=name, point_i=pi.name, point_j=pj.name, section=section, type=type, **kw)
        if f.story is None:
            top = max(pi.z, pj.z)
            s = self.story_at(top, self.options.merge_tolerance)
            f.story = s.name if s else None
        self.frames[name] = f
        return f

    def add_area(self, points: Iterable, section: Optional[str], name: Optional[str] = None,
                 type: Optional[str] = None, **kw) -> Area:
        pts = [self._resolve_point(p).name for p in points]
        if len(pts) < 3:
            raise ModelError("INVALID_AREA", "An area needs at least three points")
        if section is not None and section not in self.shell_sections:
            raise ModelError("UNKNOWN_SECTION", f"Shell section {section!r} is not defined")
        if type is None:
            zs = [self.points[p].z for p in pts]
            if max(zs) - min(zs) <= self.options.merge_tolerance:
                type = "Floor" if section is not None else "Opening"
            else:
                type = "Wall"
        name = name or self._next_name("F" if type == "Floor" else "W", self.areas)
        if name in self.areas:
            raise ModelError("DUPLICATE_NAME", f"Area {name!r} already exists", f"areas.{name}")
        a = Area(name=name, points=pts, section=section, type=type, **kw)
        if a.story is None:
            top = max(self.points[p].z for p in pts)
            s = self.story_at(top, self.options.merge_tolerance)
            a.story = s.name if s else None
        self.areas[name] = a
        return a

    def _resolve_point(self, ref) -> Point:
        if isinstance(ref, Point):
            return ref
        if isinstance(ref, str):
            try:
                return self.points[ref]
            except KeyError:
                raise ModelError("UNKNOWN_POINT", f"Point {ref!r} is not defined")
        x, y, z = ref
        return self.add_point(x, y, z)

    # ------------------------------------------------------------------- loads
    def add_load_pattern(self, name: str, type: str = "Other", self_weight_multiplier: float = 0.0) -> LoadPattern:
        lp = LoadPattern(name, type, self_weight_multiplier)
        self.load_patterns[name] = lp
        return lp

    def add_point_load(self, point, pattern: str, force=(0.0, 0.0, 0.0), moment=(0.0, 0.0, 0.0)) -> PointLoad:
        self._require_pattern(pattern)
        p = self._resolve_point(point)
        load = PointLoad(pattern, tuple(map(float, force)), tuple(map(float, moment)))
        p.loads.append(load)
        return load

    def add_frame_distributed_load(self, frame: str, pattern: str, value: float, value_b: Optional[float] = None,
                                   direction: str = "Gravity", rel_a: float = 0.0, rel_b: float = 1.0,
                                   load_type: str = "Force", absolute: bool = False) -> FrameLoad:
        self._require_pattern(pattern)
        load = FrameLoad(pattern, "distributed", direction, load_type, rel_a, rel_b, absolute,
                         float(value), float(value if value_b is None else value_b))
        self.frames[frame].loads.append(load)
        return load

    def add_frame_point_load(self, frame: str, pattern: str, value: float, rel: float = 0.5,
                             direction: str = "Gravity", load_type: str = "Force", absolute: bool = False) -> FrameLoad:
        self._require_pattern(pattern)
        load = FrameLoad(pattern, "point", direction, load_type, rel, rel, absolute, float(value), float(value))
        self.frames[frame].loads.append(load)
        return load

    def add_panel_load(self, points, distribution: str = "one_way", span_dir=None, name: Optional[str] = None,
                       story: Optional[str] = None, loads=None) -> PanelLoad:
        """A load panel (CYPE 3D style) over the polygon ``points`` (global xyz)."""
        pts = [tuple(float(v) for v in p) for p in points]
        if len(pts) < 3:
            raise ModelError("PANEL_POINTS", "A load panel needs at least three points")
        name = name or self._next_name("PL", self.panel_loads)
        if span_dir is None and distribution == "one_way":
            a, b = pts[0], pts[1]
            d = [b[k] - a[k] for k in range(3)]
            L = sum(v * v for v in d) ** 0.5 or 1.0
            span_dir = tuple(v / L for v in d)
        panel = PanelLoad(name, pts, distribution, None if span_dir is None else tuple(float(v) for v in span_dir),
                          story=story)
        for ld in loads or []:
            self._require_pattern(ld.pattern)
            panel.loads.append(ld)
        self.panel_loads[name] = panel
        return panel

    def add_area_load(self, area: str, pattern: str, value: float, direction: str = "Gravity") -> AreaLoad:
        """Uniform load on an area object.  Area loads are load panels (one form for
        every surface load): this adds the load to the area's panel (distribution
        "shells": the ETABS / SAP area load on that area), making the panel if needed."""
        self._require_pattern(pattern)
        if area not in self.areas:
            raise ModelError("UNKNOWN_AREA", f"Area {area!r} is not defined")
        load = AreaLoad(pattern, float(value), direction)
        case = PanelLoadCase(pattern, float(value), direction)
        panel = self.area_panel(area, pattern)
        if panel is None:
            self.add_area_panel(area, [case])
        else:
            panel.loads.append(case)
        return load

    def area_panel(self, area: str, pattern: Optional[str] = None) -> Optional[PanelLoad]:
        """The "shells" panel of ``area`` (holding only ``pattern`` when given)."""
        for p in self.panel_loads.values():
            if p.area == area and p.distribution == "shells" and \
                    (pattern is None or (p.loads and all(c.pattern == pattern for c in p.loads))):
                return p
        return None

    def add_area_panel(self, area: str, loads=(), distribution: str = "shells", span_dir=None,
                       name: Optional[str] = None) -> PanelLoad:
        """A load panel made from area ``area`` (an *area load*; it follows the area).
        "shells" loads the area itself; "one_way" / "two_way" carry the load to the
        frames around it (also for a load-only Null area)."""
        if area not in self.areas:
            raise ModelError("UNKNOWN_AREA", f"Area {area!r} is not defined")
        a = self.areas[area]
        pts = [self.points[p].xyz for p in a.points]
        panel = self.add_panel_load(pts, distribution, span_dir, name=name, story=a.story, loads=list(loads))
        panel.area = area
        return panel

    def sync_panels(self):
        """After editing objects: area-load panels follow their areas (model/area_panels.py)."""
        from .area_panels import sync
        sync(self)

    def convert_area_loads(self) -> int:
        """Loads stored on area objects (CSI imports, older files, scripts) -> area-load panels."""
        from .area_panels import convert_area_loads
        return convert_area_loads(self)

    def _require_pattern(self, pattern: str):
        if pattern not in self.load_patterns:
            raise ModelError("UNKNOWN_PATTERN", f"Load pattern {pattern!r} is not defined")

    # ------------------------------------------------------------------- cases
    def add_linear_static_case(self, name: str, loads: Optional[Dict[str, float]] = None, **kw) -> LinearStaticCase:
        loads = loads if loads is not None else ({name: 1.0} if name in self.load_patterns else {})
        for p in loads:
            self._require_pattern(p)
        case = LinearStaticCase(name=name, loads=dict(loads), **kw)
        self.load_cases[name] = case
        return case

    def add_modal_case(self, name: str = "Modal", max_modes: int = 12, **kw) -> ModalCase:
        case = ModalCase(name=name, max_modes=max_modes, **kw)
        self.load_cases[name] = case
        return case

    def add_response_spectrum_case(self, name: str, modal_case: str, loads: Sequence[Tuple[str, str, float]],
                                   **kw) -> ResponseSpectrumCase:
        """``loads`` is a sequence of (direction, function name, scale)."""
        if modal_case not in self.load_cases:
            raise ModelError("UNKNOWN_CASE", f"Modal case {modal_case!r} is not defined")
        case = ResponseSpectrumCase(name=name, modal_case=modal_case,
                                    loads=[SpectrumLoad(d, f, s) for d, f, s in loads], **kw)
        self.load_cases[name] = case
        return case

    def add_pdelta_case(self, name: str, loads: Dict[str, float], **kw) -> PDeltaCase:
        case = PDeltaCase(name=name, loads=dict(loads), **kw)
        self.load_cases[name] = case
        return case

    # ------------------------------------------------------- nonlinear analysis
    def add_nonlinear_material(self, mat: NonlinearMaterial) -> NonlinearMaterial:
        self.nonlinear_materials[mat.name] = mat
        return mat

    def add_fiber_section(self, sec: FiberSectionDef) -> FiberSectionDef:
        self.fiber_sections[sec.name] = sec
        return sec

    def add_frame_nonlinear(self, prop: FrameNonlinearProperty, frames: Iterable[str] = ()) -> FrameNonlinearProperty:
        """Define a distributed plasticity property and assign it to ``frames``."""
        self.frame_nonlinear[prop.name] = prop
        for f in frames:
            self.frames[f].nonlinear = prop.name
        return prop

    def add_hinge_property(self, prop: HingeProperty) -> HingeProperty:
        self.hinge_properties[prop.name] = prop
        return prop

    def assign_hinge(self, frame: str, prop: str, relative_distance: float) -> None:
        if prop not in self.hinge_properties:
            raise ModelError("UNKNOWN_HINGE", f"Hinge property {prop!r} is not defined")
        fr = self.frames[frame]
        fr.hinges = list(fr.hinges) + [FrameHingeAssignment(prop, float(relative_distance))]

    def add_link_property(self, prop: LinkProperty) -> LinkProperty:
        self.link_properties[prop.name] = prop
        return prop

    def add_link(self, point_i, point_j, prop: str, name: Optional[str] = None, angle: float = 0.0) -> LinkObject:
        pi = self._resolve_point(point_i).name
        pj = self._resolve_point(point_j).name if point_j is not None else None
        name = name or self._next_name("L", self.links)
        lk = LinkObject(name, pi, pj, prop, angle)
        self.links[name] = lk
        return lk

    def add_nonlinear_static_case(self, name: str, loads: Optional[Dict[str, float]] = None, **kw) -> NonlinearStaticCase:
        for p in (loads or {}):
            self._require_pattern(p)
        case = NonlinearStaticCase(name, dict(loads or {}), **kw)
        self.load_cases[name] = case
        return case

    def add_time_history_function(self, name: str, times: Sequence[float], values: Sequence[float],
                                  kind: str = "acceleration") -> TimeHistoryFunction:
        fn = TimeHistoryFunction(name, [float(t) for t in times], [float(v) for v in values], kind)
        self.functions[name] = fn
        return fn

    def add_time_history_case(self, name: str, loads: Sequence[TimeHistoryLoad | Tuple[str, str, str, float]],
                              **kw) -> TimeHistoryCase:
        """``loads``: TimeHistoryLoad objects or (type, name, function, scale) tuples."""
        items = [ld if isinstance(ld, TimeHistoryLoad) else TimeHistoryLoad(*ld) for ld in loads]
        case = TimeHistoryCase(name, items, **kw)
        self.load_cases[name] = case
        return case

    def add_spectrum_function(self, name: str, periods: Sequence[float], accelerations: Sequence[float],
                              damping: float = 0.05, normalized: bool = True) -> ResponseSpectrumFunction:
        fn = ResponseSpectrumFunction(name, list(map(float, periods)), list(map(float, accelerations)),
                                      damping, normalized)
        self.functions[name] = fn
        return fn

    def add_combination(self, name: str, items: Dict[str, float] | Sequence[Tuple[str, float]],
                        type: str = "LinearAdd") -> LoadCombination:
        combo = LoadCombination(name=name, type=type)
        pairs = items.items() if isinstance(items, dict) else items
        for cname, sf in pairs:
            if cname in self.load_combinations:
                combo.add(cname, sf, is_combo=True)
            elif cname in self.load_cases:
                combo.add(cname, sf)
            else:
                raise ModelError("UNKNOWN_CASE", f"{cname!r} is neither a load case nor a combination")
        self.load_combinations[name] = combo
        return combo

    # --------------------------------------------------------------- validate
    def validate(self) -> List[ModelError]:
        """Return a list of structural errors (empty if the model is consistent)."""
        errors: List[ModelError] = []

        def err(code, msg, path=None):
            errors.append(ModelError(code, msg, path))

        used_frame_sections = {f.section for f in self.frames.values()}
        for name, s in self.frame_sections.items():
            if s.material not in self.materials and name in used_frame_sections:
                err("UNKNOWN_MATERIAL", f"Frame section {name!r} references unknown material {s.material!r}",
                    f"frame_sections.{name}")
            if s.A <= 0 and s.I33 <= 0 and s.I22 <= 0 and name in used_frame_sections:
                err("INVALID_SECTION", f"Frame section {name!r} has no stiffness", f"frame_sections.{name}")
        used_shell_sections = {a.section for a in self.areas.values() if a.section}
        for name, s in self.shell_sections.items():
            if name not in used_shell_sections:
                continue
            if s.material not in self.materials:
                err("UNKNOWN_MATERIAL", f"Shell section {name!r} references unknown material {s.material!r}",
                    f"shell_sections.{name}")
            if s.thickness <= 0:
                err("INVALID_SECTION", f"Shell section {name!r} needs a positive thickness", f"shell_sections.{name}")
        for name, m in self.materials.items():
            if m.E <= 0:
                err("INVALID_MATERIAL", f"Material {name!r} needs E > 0", f"materials.{name}")
        for name, f in self.frames.items():
            for p in (f.point_i, f.point_j):
                if p not in self.points:
                    err("UNKNOWN_POINT", f"Frame {name!r} references unknown point {p!r}", f"frames.{name}")
            if f.section not in self.frame_sections and f.type != "Null":  # null lines carry no section
                err("UNKNOWN_SECTION", f"Frame {name!r} references unknown section {f.section!r}", f"frames.{name}")
            for ld in f.loads:
                if ld.pattern not in self.load_patterns:
                    err("UNKNOWN_PATTERN", f"Frame {name!r} load uses unknown pattern {ld.pattern!r}", f"frames.{name}")
        for name, a in self.areas.items():
            for p in a.points:
                if p not in self.points:
                    err("UNKNOWN_POINT", f"Area {name!r} references unknown point {p!r}", f"areas.{name}")
            if a.section is not None and a.section not in self.shell_sections:
                err("UNKNOWN_SECTION", f"Area {name!r} references unknown section {a.section!r}", f"areas.{name}")
            if a.diaphragm and a.diaphragm not in self.diaphragms:
                err("UNKNOWN_DIAPHRAGM", f"Area {name!r} references unknown diaphragm {a.diaphragm!r}")
        for name, p in self.points.items():
            if p.diaphragm and p.diaphragm not in self.diaphragms:
                err("UNKNOWN_DIAPHRAGM", f"Point {name!r} references unknown diaphragm {p.diaphragm!r}")
            for ld in p.loads:
                if ld.pattern not in self.load_patterns:
                    err("UNKNOWN_PATTERN", f"Point {name!r} load uses unknown pattern {ld.pattern!r}")
        for name, c in self.load_cases.items():
            if isinstance(c, (LinearStaticCase, PDeltaCase, LinearBucklingCase)):
                for p in c.loads:
                    if p not in self.load_patterns:
                        err("UNKNOWN_PATTERN", f"Case {name!r} uses unknown pattern {p!r}", f"load_cases.{name}")
            elif isinstance(c, ResponseSpectrumCase):
                if c.modal_case not in self.load_cases:
                    err("UNKNOWN_CASE", f"Spectrum case {name!r} needs modal case {c.modal_case!r}")
                for ld in c.loads:
                    if ld.function not in self.functions:
                        err("UNKNOWN_FUNCTION", f"Spectrum case {name!r} uses unknown function {ld.function!r}")
            elif isinstance(c, NonlinearStaticCase):
                for p in c.loads:
                    if p not in self.load_patterns:
                        err("UNKNOWN_PATTERN", f"Case {name!r} uses unknown pattern {p!r}", f"load_cases.{name}")
                if c.initial_case and c.initial_case not in self.load_cases:
                    err("UNKNOWN_CASE", f"Case {name!r}: unknown initial case {c.initial_case!r}")
                if c.control.lower().startswith("disp") and c.monitored_point not in self.points:
                    err("MONITORED_POINT", f"Case {name!r}: displacement control needs an existing monitored point")
            elif isinstance(c, TimeHistoryCase):
                for ld in c.loads:
                    if ld.function not in self.functions:
                        err("UNKNOWN_FUNCTION", f"Time history case {name!r} uses unknown function {ld.function!r}")
                    if not ld.type.lower().startswith("acc") and ld.name not in self.load_patterns:
                        err("UNKNOWN_PATTERN", f"Time history case {name!r} uses unknown pattern {ld.name!r}")
                if c.method.lower().startswith("modal") and c.modal_case not in self.load_cases:
                    err("UNKNOWN_CASE", f"Modal time history {name!r} needs modal case {c.modal_case!r}")
                if c.initial_case and c.initial_case not in self.load_cases:
                    err("UNKNOWN_CASE", f"Case {name!r}: unknown initial case {c.initial_case!r}")
        for fname, fr in self.frames.items():
            if fr.nonlinear and fr.nonlinear not in self.frame_nonlinear:
                err("UNKNOWN_FRAME_NONLINEAR", f"Frame {fname!r}: unknown nonlinear property {fr.nonlinear!r}")
            for h in fr.hinges:
                if h.property not in self.hinge_properties:
                    err("UNKNOWN_HINGE", f"Frame {fname!r}: unknown hinge property {h.property!r}")
        for pname, prop in self.frame_nonlinear.items():
            if prop.section not in self.fiber_sections:
                err("UNKNOWN_FIBER_SECTION", f"Nonlinear property {pname!r}: unknown fiber section {prop.section!r}")
        for sname, sec in self.fiber_sections.items():
            for mat in (sec.core_material, sec.cover_material, sec.steel_material):
                if mat and mat not in self.nonlinear_materials:
                    err("UNKNOWN_NL_MATERIAL", f"Fiber section {sname!r}: unknown material {mat!r}")
        for lname, lk in self.links.items():
            if lk.property not in self.link_properties:
                err("UNKNOWN_LINK_PROPERTY", f"Link {lname!r}: unknown property {lk.property!r}")
            for pt in (lk.point_i, lk.point_j):
                if pt is not None and pt not in self.points:
                    err("UNKNOWN_POINT", f"Link {lname!r}: unknown point {pt!r}")
        for name, combo in self.load_combinations.items():
            for it in combo.items:
                if it.is_combo and it.name not in self.load_combinations:
                    err("UNKNOWN_CASE", f"Combination {name!r} uses unknown combination {it.name!r}")
                if not it.is_combo and it.name not in self.load_cases:
                    err("UNKNOWN_CASE", f"Combination {name!r} uses unknown case {it.name!r}")
        return errors

    def require_valid(self):
        errors = self.validate()
        if errors:
            raise ModelError(errors[0].code, "; ".join(e.message for e in errors[:5]), errors[0].path)

    # ------------------------------------------------------------- unit change
    def convert_units(self, new_units: UnitSystem | str) -> "Model":
        """Return a copy of the model expressed in another unit system."""
        new_units = new_units if isinstance(new_units, UnitSystem) else UnitSystem.parse(new_units)
        src = self.units
        out = Model(self.info.name, new_units)
        out.info = dataclasses.replace(self.info)
        for attr in ("materials", "rebar_sizes", "frame_sections", "shell_sections", "stories", "grid_systems", "diaphragms",
                     "points", "frames", "areas", "groups", "line_springs", "tendons", "panel_loads", "load_patterns", "mass_sources",
                     "functions", "load_cases", "load_combinations", "nonlinear_materials", "fiber_sections",
                     "frame_nonlinear", "hinge_properties", "link_properties", "links"):
            table = getattr(self, attr)
            setattr(out, attr, {k: convert_dataclass(v, src, new_units) for k, v in table.items()})
        out.grid_lines = [convert_dataclass(g, src, new_units) for g in self.grid_lines]
        out.options = dataclasses.replace(self.options)
        out.display_units = dict(self.display_units)
        ratio = src.factor("length") / new_units.factor("length")
        for name in ("floor_mesh_max_size", "wall_mesh_max_size"):
            v = getattr(out.options, name)
            if v is not None:
                setattr(out.options, name, v * ratio)
        out.options.merge_tolerance *= ratio
        if out.options.gravity:  # length / s^2
            out.options.gravity *= ratio
        for panel in out.panel_loads.values():  # polygon corners (tuples) are lengths
            panel.points = [tuple(v * ratio for v in p) for p in panel.points]
        return out

    # ------------------------------------------------ frame end offset defaults
    def use_csi_frame_defaults(self) -> None:
        """Model defaults of CSI programs for frames that do not set their own end
        offsets / rigid-zone factor (imported models): no automatic offsets, RZF 0."""
        o = self.options
        o.frame_auto_end_offsets = False
        o.rigid_zone_factor_concrete = o.rigid_zone_factor_steel = o.rigid_zone_factor_other = 0.0

    def frame_auto_end_offsets(self, f) -> bool:
        """End offsets from connectivity for this frame (its own setting or the model default)."""
        return self.options.frame_auto_end_offsets if f.auto_end_offsets is None else bool(f.auto_end_offsets)

    def frame_rigid_factor(self, f) -> float:
        """Rigid-zone factor of this frame: its own, else the model default for its material."""
        if f.rigid_factor is not None:
            return float(f.rigid_factor)
        o = self.options
        sec = self.frame_sections.get(f.section)
        mat = self.materials.get(sec.material) if sec is not None else None
        kind = (getattr(mat, "type", "") or "").lower()
        if kind.startswith("concrete"):
            return float(o.rigid_zone_factor_concrete)
        if kind.startswith("steel"):
            return float(o.rigid_zone_factor_steel)
        return float(o.rigid_zone_factor_other)

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "units": {"force": self.units.force, "length": self.units.length, "temperature": self.units.temperature},
            "display_units": dict(self.display_units),
            "info": to_dict(self.info),
            "options": to_dict(self.options),
            "materials": to_dict(self.materials),
            "rebar_sizes": to_dict(self.rebar_sizes),
            "frame_sections": to_dict(self.frame_sections),
            "shell_sections": to_dict(self.shell_sections),
            "stories": to_dict(self.stories),
            "grid_systems": to_dict(self.grid_systems),
            "grid_lines": to_dict(self.grid_lines),
            "diaphragms": to_dict(self.diaphragms),
            "points": to_dict(self.points),
            "frames": to_dict(self.frames),
            "areas": to_dict(self.areas),
            "groups": to_dict(self.groups),
            "line_springs": to_dict(self.line_springs),
            "tendons": to_dict(self.tendons),
            "panel_loads": to_dict(self.panel_loads),
            "load_patterns": to_dict(self.load_patterns),
            "mass_sources": to_dict(self.mass_sources),
            "functions": to_dict(self.functions),
            "load_cases": to_dict(self.load_cases),
            "load_combinations": to_dict(self.load_combinations),
            "nonlinear_materials": to_dict(self.nonlinear_materials),
            "fiber_sections": to_dict(self.fiber_sections),
            "frame_nonlinear": to_dict(self.frame_nonlinear),
            "hinge_properties": to_dict(self.hinge_properties),
            "link_properties": to_dict(self.link_properties),
            "links": to_dict(self.links),
        }

    def to_json(self, path: Optional[str] = None, indent: int = 1) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        u = data.get("units", {})
        m = cls(data.get("info", {}).get("name", "Untitled"),
                UnitSystem(u.get("force", "kN"), u.get("length", "m"), u.get("temperature", "C")))
        m.info = ProjectInfo(**_strip(data.get("info", {})))
        from ..display_units import force_label
        m.display_units = {str(k): force_label(v) for k, v in (data.get("display_units") or {}).items() if v}
        m.options = AnalysisOptions(**_coerce_tuples(_strip(data.get("options", {}))))
        for attr in ("materials", "rebar_sizes", "frame_sections", "shell_sections", "stories", "grid_systems", "diaphragms",
                     "points", "frames", "areas", "groups", "line_springs", "tendons", "panel_loads", "load_patterns", "mass_sources",
                     "functions", "load_cases", "load_combinations", "nonlinear_materials", "fiber_sections",
                     "frame_nonlinear", "hinge_properties", "link_properties", "links"):
            table = data.get(attr, {})
            setattr(m, attr, {k: _from_dict(v) for k, v in table.items()})
        m.grid_lines = [_from_dict(g) for g in data.get("grid_lines", [])]
        n = max([int(k[2:]) for k in m.panel_loads if k.startswith("PL") and k[2:].isdigit()] or [0])
        if n:
            m._counters["PL"] = max(m._counters.get("PL", 0), n)
        if _version_tuple(data.get("schema_version", "1.0")) < (2, 1):
            m.convert_area_loads()  # older files: area loads stored on the areas -> area-load panels
        if _version_tuple(data.get("schema_version", "1.0")) < (2, 2):
            m.options.mesh_refine_check = False  # was saved on by default: the analysis took minutes
            m.options.mesh_refine_tolerance = max(m.options.mesh_refine_tolerance, 0.02)
        return m

    @classmethod
    def from_json(cls, path_or_text: str) -> "Model":
        if path_or_text.lstrip().startswith("{"):
            return cls.from_dict(json.loads(path_or_text))
        with open(path_or_text, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ----------------------------------------------------------------- summary
    def summary(self) -> dict:
        return {
            "name": self.info.name,
            "units": f"{self.units.force}-{self.units.length}",
            "stories": len(self.stories),
            "points": len(self.points),
            "frames": len(self.frames),
            "areas": len(self.areas),
            "load_patterns": len(self.load_patterns),
            "load_cases": len(self.load_cases),
            "load_combinations": len(self.load_combinations),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"<Model {s['name']!r} [{s['units']}] {s['points']} points, {s['frames']} frames, "
                f"{s['areas']} areas, {s['load_cases']} cases>")


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "_type"}


def _coerce_tuples(d: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}


def _from_dict(d):
    if isinstance(d, dict) and "_type" in d:
        cls = _TYPE_REGISTRY[d["_type"]]
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            kwargs[f.name] = _from_dict(v)
            # dataclass tuples are serialised as lists
            if isinstance(kwargs[f.name], list) and str(f.type).startswith(("Tuple", "SixBool", "Six", "Vec3", "typing.Tuple")):
                kwargs[f.name] = tuple(kwargs[f.name])
        return cls(**kwargs)
    if isinstance(d, dict):
        return {k: _from_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_from_dict(v) for v in d]
    return d
