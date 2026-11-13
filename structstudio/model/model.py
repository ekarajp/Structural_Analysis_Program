"""The :class:`Model` container: everything an engineer defines.

A model is a plain in-memory object graph.  It knows nothing about finite
elements; :mod:`structstudio.mesh` turns it into an analysis model and
:mod:`structstudio.engine` solves that.
"""
from __future__ import annotations

import dataclasses
import math
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
                      TendonPoint, Tower, FREE)
from .properties import FrameModifiers, FrameSection, Material, RebarSize, ShellModifiers, ShellSection
from .design import DesignPreferences, SlabColumnDriftPrefs, SlabDesignModelPrefs
from .nonlinear import (NONLINEAR_TYPES, FiberSectionDef, FrameHingeAssignment, FrameNonlinearProperty,
                        HingeProperty, LinkDOF, LinkObject, LinkProperty, NonlinearMaterial, NonlinearStaticCase,
                        TimeHistoryCase, TimeHistoryLoad)

SCHEMA_VERSION = "2.3"  # 2.1: area loads are load panels made from the areas (PanelLoad.area)
# 2.2: the mesh refinement check is off by default (files before 2.2 saved it on by default)
# 2.3: multiple towers (Model.towers, Story.tower, GridSystem.tower)

def _version_tuple(text) -> tuple:
    try:
        return tuple(int(x) for x in str(text).split("."))
    except ValueError:
        return (0,)


_KNOWN_KEYS = {"schema_version", "units", "display_units", "info", "options", "materials", "rebar_sizes",
               "frame_sections", "shell_sections", "stories", "towers", "grid_systems", "grid_lines", "diaphragms", "points",
               "frames", "areas", "groups", "line_springs", "tendons", "panel_loads", "load_patterns", "mass_sources",
               "functions", "load_cases", "load_combinations", "nonlinear_materials", "fiber_sections",
               "frame_nonlinear", "hinge_properties", "link_properties", "links", "design"}

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
        FrameLoad, AreaLoad, Story, Tower, GridLine, GridSystem, Diaphragm, Group, LineSpring, LoadPattern, MassSource,
        ResponseSpectrumFunction, TimeHistoryFunction, LinearStaticCase, ModalCase, ResponseSpectrumCase,
        PDeltaCase, LinearBucklingCase, ComboItem, LoadCombination, SpectrumLoad,
        DesignPreferences, SlabColumnDriftPrefs, SlabDesignModelPrefs,
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
    fine_mesh_size_min: Optional[float] = None  # element sizes of the fine mesh (model length units)
    fine_mesh_size_max: Optional[float] = None
    use_fine_mesh: bool = False
    # Grid system: "story" = X, Y grid lines + stories, "xyz" = X, Y, Z grid
    # lines.  In "xyz" mode the Z grid lines are the levels and the stories mirror them, so
    # diaphragms, story drifts and all story based tools keep working.
    grid_mode: str = "story"


@dataclass
class ProjectInfo:
    """File > Project Information: the project data for reports and drawing title
    blocks (kept with the model)."""

    name: str = "Untitled"
    engineer: str = ""
    company: str = ""
    notes: str = ""
    number: str = ""  # project number
    location: str = ""
    owner: str = ""
    license_no: str = ""  # engineer's licence / registration number (e.g. COE)
    checker: str = ""
    logo: str = ""  # company logo image file
    revision: str = ""
    date: str = ""
    design_codes: str = ""



def finite_xyz(xyz, what: str = "point"):
    """``xyz`` as three finite floats, or ModelError NONFINITE_COORDINATE (never clamped or zeroed)."""
    try:
        vals = tuple(float(v) for v in xyz)
    except (TypeError, ValueError):
        raise ModelError("NONFINITE_COORDINATE", f"The {what} coordinates {xyz!r} are not numbers") from None
    bad = [a for a, v in zip("xyz", vals) if not math.isfinite(v)]
    if len(vals) != 3 or bad:
        raise ModelError("NONFINITE_COORDINATE",
                         f"The {what} {'/'.join(bad) or 'xyz'} coordinate is not a finite number "
                         f"({', '.join(repr(v) for v in vals)}) - the value is too large or not a number")
    return vals


def nonfinite_values(data, path: str = ""):
    """Paths (a.b.0.c) of every float in a to_dict() tree that is infinite or NaN."""
    if isinstance(data, float):
        if not math.isfinite(data):
            yield path
    elif isinstance(data, dict):
        for k, v in data.items():
            yield from nonfinite_values(v, f"{path}.{k}" if path else str(k))
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            yield from nonfinite_values(v, f"{path}.{i}" if path else str(i))


def nonfinite_points(model) -> List[str]:
    """Names of points with an infinite or NaN coordinate."""
    fin = math.isfinite
    return [n for n, p in model.points.items() if not (fin(p.x) and fin(p.y) and fin(p.z))]

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
        # ETABS "multiple towers": empty (or one entry) = one building; the stories and grid
        # systems name their tower (None = the first)
        self.towers: Dict[str, Tower] = {}
        self.active_tower: Optional[str] = None  # not saved: the tower new objects go to (see story_at)
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
        self.design = DesignPreferences()  # Design menu preferences
        # units each kind of value is shown / typed in (Display Units form, e.g. Thai
        # legacy: section dimensions cm, areas cm^2, deflections mm, strengths ksc);
        # item -> unit label, missing = the consistent units above.  Display only.
        self.display_units: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}
        self._pindex: Optional[dict] = None  # while building: cell -> point names (see point_index)

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

    def z_grid_lines(self, tower: Optional[str] = None) -> List[GridLine]:
        """The Z grid lines (SAP2000 mode) bottom-up; several towers: those of ``tower``'s
        grid systems when it is given."""
        lines = (g for g in self.grid_lines if g.direction == "Z")
        if tower is not None and self.multi_tower:
            mine = set(self.grid_systems_of(tower))
            lines = (g for g in lines if g.system in mine)
        return sorted(lines, key=lambda g: g.ordinate)

    def levels(self) -> List[Tuple[str, float]]:
        """(name, elevation) of the levels bottom-up: stories (ETABS mode) or Z grid
        lines (SAP2000 mode).  Several towers: every tower's (see :meth:`stories_of`)."""
        if self.grid_mode == "xyz" and not self.multi_tower:
            z = self.z_grid_lines()
            if z:
                return [(g.name, g.ordinate) for g in z]
        return [(s.name, s.elevation) for s in self.stories_bottom_up()]

    def _z_system(self, tower: Optional[str] = None) -> str:
        """The grid system that carries the Z lines (of a tower): its first one."""
        names = self.grid_systems_of(tower) if (tower is not None and self.multi_tower) else list(self.grid_systems)
        if names:
            return names[0]
        name = "G1" if tower is None or not self.multi_tower else f"{tower}-G1"
        k = 2
        while name in self.grid_systems:
            name, k = f"{tower or 'G'}-G{k}", k + 1
        self.grid_systems[name] = GridSystem(name=name, tower=tower if self.multi_tower else None)
        return name

    def set_grid_mode(self, mode: str) -> None:
        """Switch between the X-Y-Story and X-Y-Z grid systems.
        story -> xyz: one Z grid line per story (same names); xyz -> story: the
        stories are rebuilt from the Z grid lines and the Z lines are removed.
        Several towers: each tower's Z lines are its stories (they sit in its first
        grid system), so every tower keeps its own levels."""
        if mode not in self.GRID_MODES:
            raise ModelError("INVALID_GRID_MODE", f"Grid mode must be one of {self.GRID_MODES}")
        if mode == self.grid_mode and (mode == "story" or self.z_grid_lines()):
            return
        towers = self.tower_names() if self.multi_tower else [None]
        if mode == "xyz":
            self.grid_lines = [g for g in self.grid_lines if g.direction != "Z"]
            for t in towers:
                system = self._z_system(t)
                for s in (self.stories_of(t) if t is not None else self.stories_bottom_up()):
                    self.grid_lines.append(GridLine(system=system, name=s.name, direction="Z", ordinate=s.elevation))
        else:
            for t in towers:
                z = self.z_grid_lines(t)
                if z:
                    self._levels_to_stories(t, [(g.name, g.ordinate) for g in z])
            self.grid_lines = [g for g in self.grid_lines if g.direction != "Z"]
        self.options.grid_mode = mode

    def sync_z_grid(self) -> None:
        """SAP2000 mode: a tower without Z grid lines (a new or copied tower) gets one per
        story; with one tower the Z lines are the stories again (after towers combined)."""
        if self.grid_mode != "xyz":
            return
        if not self.multi_tower:
            have = {(g.name, round(g.ordinate, 9)) for g in self.z_grid_lines()}
            want = {(st.name, round(st.elevation, 9)) for st in self.stories_bottom_up()}
            if have != want:
                system = self._z_system(None)
                self.grid_lines = [g for g in self.grid_lines if g.direction != "Z"] + [
                    GridLine(system=system, name=st.name, direction="Z", ordinate=st.elevation)
                    for st in self.stories_bottom_up()]
            return
        for t in self.tower_names():
            if not self.z_grid_lines(t):
                system = self._z_system(t)
                self.grid_lines += [GridLine(system=system, name=st.name, direction="Z", ordinate=st.elevation)
                                    for st in self.stories_of(t)]

    def _levels_to_stories(self, tower: Optional[str], levels) -> None:
        if tower is None or not self.multi_tower:
            self._stories_from_levels(levels)
        else:
            self.set_tower_stories(tower, levels)

    def set_z_grid(self, levels: Sequence[Tuple[str, float]], tower: Optional[str] = None) -> None:
        """SAP2000 mode: replace the Z grid lines (name, z) and rebuild the stories;
        several towers: those of ``tower`` (the first by default)."""
        levels = sorted(((str(n), float(z)) for n, z in levels), key=lambda t: t[1])
        names = [n for n, _ in levels]
        if len(set(names)) != len(names):
            raise ModelError("DUPLICATE_NAME", "Z grid line names must be unique")
        t = (self._tower(tower) if self.multi_tower else None)
        system = self._z_system(t)
        mine = set(self.grid_systems_of(t)) if t is not None else None
        self.grid_lines = [g for g in self.grid_lines
                           if g.direction != "Z" or (mine is not None and g.system not in mine)] + [
            GridLine(system=system, name=n, direction="Z", ordinate=z) for n, z in levels]
        self._levels_to_stories(t, levels)

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

    def object_towers(self) -> Dict[Tuple[str, str], str]:
        """(kind, name) -> tower of every point / frame / area that is on a story now
        (several towers only): taken before the stories change, so that every object
        stays in its tower (:meth:`reassign_stories`)."""
        if not self.multi_tower:
            return {}
        out = {}
        for kind, table in (("point", self.points), ("frame", self.frames), ("area", self.areas)):
            for n, o in table.items():
                if o.story in self.stories:
                    out[(kind, n)] = self.tower_of_story(o.story)
        return out

    def reassign_stories(self, towers: Optional[Dict[Tuple[str, str], str]] = None) -> None:
        """Re-derive the story of every point / frame / area from its elevation (each
        object stays in its tower: ``towers`` from :meth:`object_towers` taken before
        the stories changed, else the tower of its present story)."""
        tol = self.options.merge_tolerance
        towers = dict(towers) if towers is not None else self.object_towers()
        for p in self.points.values():
            s = self.story_at(p.z, tol, tower=towers.get(("point", p.name)), xy=(p.x, p.y))
            p.story = s.name if s else None
        for f in self.frames.values():
            pi, pj = self.points[f.point_i], self.points[f.point_j]
            top = max(pi.z, pj.z)
            hint = towers.get(("frame", f.name)) or self._tower_hint((pi, pj))
            s = self.story_at(top, tol, tower=hint, xy=((pi.x + pj.x) / 2, (pi.y + pj.y) / 2))
            f.story = s.name if s else (f.story if f.story in self.stories else None)
        for a in self.areas.values():
            pp = [self.points[n] for n in a.points]
            top = max(q.z for q in pp)
            hint = towers.get(("area", a.name)) or self._tower_hint(pp)
            s = self.story_at(top, tol, tower=hint, xy=(sum(q.x for q in pp) / len(pp), sum(q.y for q in pp) / len(pp)))
            a.story = s.name if s else (a.story if a.story in self.stories else None)

    def set_tower_stories(self, tower: Optional[str], levels: Sequence[Tuple[str, float]]) -> None:
        """Replace the stories of one tower by (name, elevation) levels (the lowest is its
        base); the other towers' stories stay.  Objects keep their tower."""
        levels = sorted(((str(n), float(z)) for n, z in levels), key=lambda t: t[1])
        names = [n for n, _ in levels]
        if len(set(names)) != len(names):
            raise ModelError("DUPLICATE_NAME", "Story names must be unique")
        if not self.multi_tower:
            self._stories_from_levels(levels)
            return
        t = self._tower(tower)
        mine = {st.name for st in self.stories_of(t)}
        clash = [n for n in names if n in self.stories and n not in mine]
        if clash:
            raise ModelError("DUPLICATE_NAME", f"Story name {clash[0]!r} is used by tower "
                                               f"{self.tower_of_story(clash[0])} (story names are unique in the model)")
        keep = self.object_towers()
        old = {n: self.stories.pop(n) for n in mine}
        prev = None
        for name, z in levels:
            kw = {}
            if name in old:
                o = old[name]
                kw = {"is_master": o.is_master, "similar_to": o.similar_to, "color": o.color}
            self.add_story(name, z, 0.0 if prev is None else z - prev, tower=t, **kw)
            prev = z
        self.reassign_stories(keep)

    def add_tower_stories(self, tower: str, n_stories: int, typical: float, bottom: Optional[float] = None,
                          base_elevation: float = 0.0, prefix: Optional[str] = None) -> List[Story]:
        """ETABS "Quick Story Definition for New Tower": a new tower with a base and
        ``n_stories`` stories (the bottom one ``bottom`` high, the others ``typical``)."""
        if n_stories < 1 or typical <= 0 or (bottom is not None and bottom <= 0):
            raise ModelError("INVALID_STORIES", "A tower needs at least one story and story heights > 0")
        prefix = prefix if prefix is not None else f"{tower}-Story"
        base = f"{tower}-Base"
        names = [base] + [f"{prefix}{k}" for k in range(1, n_stories + 1)]
        used = [n for n in names if n in self.stories]
        if used:
            raise ModelError("DUPLICATE_NAME", f"Story {used[0]!r} exists already")
        self.add_tower(tower)
        out = [self.add_story(base, float(base_elevation), 0.0, tower=tower)]
        z = float(base_elevation)
        for k in range(1, n_stories + 1):
            h = float(bottom if (k == 1 and bottom is not None) else typical)
            z += h
            out.append(self.add_story(names[k], z, h, tower=tower))
        return out

    def copy_tower(self, src: str, name: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Tower:
        """A new tower with the stories (renamed ``<name>-<story>``) and the grid systems
        (``<name>-<system>``, moved by dx, dy) of tower ``src`` - its definition only,
        as ETABS "Add Copy of Tower" (no objects)."""
        if src not in self.tower_names():
            raise ModelError("UNKNOWN_TOWER", f"No tower {src!r}")
        src = self._tower(src)
        stories = self.stories_of(src)
        systems = self.grid_systems_of(src)
        self.add_tower(name)
        src = self._tower(src)
        for st in stories:
            n = f"{name}-{st.name}"
            if n in self.stories:
                raise ModelError("DUPLICATE_NAME", f"Story {n!r} exists already")
            self.add_story(n, st.elevation + dz, st.height, tower=name, color=st.color)
        for sname in systems:
            gs = self.grid_systems[sname]
            n = f"{name}-{sname}"
            k = 2
            while n in self.grid_systems:
                n, k = f"{name}-{sname}{k}", k + 1
            o = gs.origin or (0.0, 0.0)
            self.grid_systems[n] = dataclasses.replace(gs, name=n, origin=(o[0] + dx, o[1] + dy), tower=name)
            self.grid_lines += [dataclasses.replace(g, system=n) for g in self.grid_lines
                                if g.system == sname and g.direction != "Z"]
        self.sync_z_grid()
        return self.towers[name]

    def combine_towers(self, keep: str) -> None:
        """ETABS "Do Not Allow Multiple Towers": every tower combined into ``keep`` - a
        story at the same elevation as one of ``keep``'s takes its place (its objects
        move to that story), the others join ``keep``'s stories; grid systems too."""
        if not self.towers:
            return
        keep = self._tower(keep)
        tol = self.options.merge_tolerance
        mine = self.stories_of(keep)
        rename = {}
        for st in list(self.stories.values()):
            if self._tower(st.tower) == keep:
                continue
            same = next((k for k in mine if abs(k.elevation - st.elevation) <= tol), None)
            if same is not None:
                rename[st.name] = same.name
                del self.stories[st.name]
        for table in (self.points, self.frames, self.areas, self.panel_loads):
            for o in table.values():
                if getattr(o, "story", None) in rename:
                    o.story = rename[o.story]
        self.towers.clear()
        for st in self.stories.values():
            st.tower = None
        for gs in self.grid_systems.values():
            gs.tower = None
        prev = None  # heights follow the combined order
        for st in self.stories_bottom_up():
            st.height = 0.0 if prev is None else st.elevation - prev
            prev = st.elevation
        self.sync_z_grid()

    def story_for(self, obj, kind: Optional[str] = None) -> Optional[str]:
        """The story an object belongs to from its present elevation, in its own tower
        (the tower of its present story, else of its joints): a point -> the story at
        its z; a frame / area -> the story at its top.  None off the levels."""
        tol = self.options.merge_tolerance
        kind = kind or ("point" if isinstance(obj, Point) else "frame" if isinstance(obj, Frame) else "area")
        if kind == "point":
            hint = self.tower_of_story(obj.story) if obj.story in self.stories and self.multi_tower else None
            s = self.story_at(obj.z, tol, tower=hint, xy=(obj.x, obj.y))
            return s.name if s else None
        pts = [self.points[obj.point_i], self.points[obj.point_j]] if kind == "frame" else \
            [self.points[n] for n in obj.points]
        hint = None
        if self.multi_tower:
            hint = self.tower_of_story(obj.story) if obj.story in self.stories else self._tower_hint(pts)
        top = max(q.z for q in pts)
        s = self.story_at(top, tol, tower=hint, xy=(sum(q.x for q in pts) / len(pts), sum(q.y for q in pts) / len(pts)))
        return s.name if s else None

    # ------------------------------------------------------------------ towers
    DEFAULT_TOWER = "T1"

    @property
    def multi_tower(self) -> bool:
        """More than one tower (ETABS "Allow multiple towers" with a second tower)."""
        return len(self.towers) > 1

    def tower_names(self) -> List[str]:
        return list(self.towers) or [self.DEFAULT_TOWER]

    def default_tower(self) -> str:
        return next(iter(self.towers), self.DEFAULT_TOWER)

    def _tower(self, name: Optional[str]) -> str:
        """A stored tower name (None or unknown -> the first tower)."""
        return name if name is not None and name in self.towers else self.default_tower()

    def tower_of_story(self, story: Optional[str]) -> str:
        s = self.stories.get(story) if story is not None else None
        return self._tower(s.tower if s is not None else None)

    def tower_of_system(self, system: Optional[str]) -> str:
        gs = self.grid_systems.get(system) if system is not None else None
        return self._tower(gs.tower if gs is not None else None)

    def tower_of_point(self, name: str) -> str:
        """The tower of a point: its story's, else that of a frame / area on it."""
        p = self.points[name]
        if p.story in self.stories:
            return self.tower_of_story(p.story)
        if not self.multi_tower:
            return self.default_tower()
        for f in self.frames.values():
            if (f.point_i == name or f.point_j == name) and f.story in self.stories:
                return self.tower_of_story(f.story)
        for a in self.areas.values():
            if name in a.points and a.story in self.stories:
                return self.tower_of_story(a.story)
        t = self.tower_at(p.x, p.y)
        return t if t is not None else self.default_tower()

    def tower_of_object(self, kind: str, name: str) -> str:
        """Tower of a "point" / "frame" / "area"."""
        if kind == "point":
            return self.tower_of_point(name)
        obj = (self.frames if kind == "frame" else self.areas)[name]
        if obj.story in self.stories:
            return self.tower_of_story(obj.story)
        pts = [obj.point_i, obj.point_j] if kind == "frame" else list(obj.points)
        return self._tower_hint([self.points[n] for n in pts]) or self.default_tower()

    def _draw_tower(self, pts) -> Optional[str]:
        """The tower a new frame / area goes to: the tower drawn in (``active_tower``),
        else that of its joints (a beam from a podium joint drawn in a tower's window
        belongs to that tower)."""
        t = getattr(self, "active_tower", None)
        if t is not None and self.multi_tower and t in self.towers:
            return t
        return self._tower_hint(pts)

    def _tower_hint(self, pts) -> Optional[str]:
        """The tower of the first of ``pts`` that is on a story (None: none is)."""
        if not self.multi_tower:
            return None
        for p in sorted(pts, key=lambda q: -q.z):
            if p.story in self.stories:
                return self.tower_of_story(p.story)
        return None

    def tower_at(self, x: float, y: float) -> Optional[str]:
        """The tower whose grid systems alone cover plan point (x, y); None with one
        tower, when no tower's grid covers it or when several do (a tower on a
        podium: the plan position does not tell)."""
        if not self.multi_tower:
            return None
        found = [t for t, ext in self._tower_extents().items()
                 if ext is not None and ext[0] <= x <= ext[1] and ext[2] <= y <= ext[3]]
        return found[0] if len(found) == 1 else None

    def _tower_extents(self) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
        """Plan extent of the grid of each tower (cached on the grid data)."""
        sig = (tuple((n, gs.type, tuple(gs.origin or (0, 0)), gs.rotation, gs.tower) for n, gs in self.grid_systems.items()),
               tuple((g.system, g.direction, g.ordinate, tuple(g.start or ()), tuple(g.end or ()), g.visible)
                     for g in self.grid_lines), tuple(self.towers))
        cache = getattr(self, "_tower_ext_cache", None)
        if cache is not None and cache[0] == sig:
            return cache[1]
        from .grids import grid_geometry
        out = {t: grid_geometry(_TowerGrids(self, t)).extent for t in self.towers}
        self._tower_ext_cache = (sig, out)
        return out

    def view(self, tower=None):
        """The model seen through ``tower`` - one tower or a list of towers (a
        :class:`TowerView`); the model itself with one tower or when ``tower`` is None
        (all towers)."""
        if tower is None or not self.multi_tower:
            return self
        return TowerView(self, tower)

    def same_level_stories(self, story: Optional[str]) -> List[str]:
        """``story`` and the stories of the other towers at its elevation (a podium and
        the towers on it, worked on as one floor), in tower order."""
        if story is None or story not in self.stories:
            return [story] if story else []
        if not self.multi_tower:
            return [story]
        z = self.stories[story].elevation
        tol = max(self.options.merge_tolerance, 1e-6)
        order = {t: k for k, t in enumerate(self.towers)}
        others = [st.name for st in self.stories.values() if st.name != story and abs(st.elevation - z) <= tol]
        return [story] + sorted(others, key=lambda n: order.get(self.tower_of_story(n), 0))

    def stories_of(self, tower: Optional[str] = None) -> List[Story]:
        """The stories of a tower, bottom-up (all stories when ``tower`` is None)."""
        if tower is None:
            return self.stories_bottom_up()
        t = self._tower(tower)
        return [s for s in self.stories_bottom_up() if self._tower(s.tower) == t]

    def grid_systems_of(self, tower: Optional[str] = None) -> List[str]:
        if tower is None:
            return list(self.grid_systems)
        t = self._tower(tower)
        return [n for n, gs in self.grid_systems.items() if self._tower(gs.tower) == t]

    def allow_towers(self, on: bool, keep: Optional[str] = None) -> None:
        """ETABS Options > Allow Multiple Towers: on - the model's stories and grid systems
        become tower T1; off - every tower combined into ``keep`` (the first by default)."""
        if on:
            if not self.towers:
                self.towers[self.DEFAULT_TOWER] = Tower(self.DEFAULT_TOWER)
                for st in self.stories.values():
                    st.tower = self.DEFAULT_TOWER
                for gs in self.grid_systems.values():
                    gs.tower = self.DEFAULT_TOWER
            return
        self.combine_towers(keep if keep is not None else self.default_tower())

    def add_tower(self, name: str) -> Tower:
        """A new tower (the model's first towers: the existing stories / systems become
        the first tower)."""
        if name in self.towers:
            raise ModelError("DUPLICATE_NAME", f"Tower {name!r} already exists", f"towers.{name}")
        if not self.towers:
            first = self.DEFAULT_TOWER if name != self.DEFAULT_TOWER else "T0"
            self.towers[first] = Tower(first)
            for s in self.stories.values():
                s.tower = s.tower or first
            for gs in self.grid_systems.values():
                gs.tower = gs.tower or first
        t = self.towers[name] = Tower(name)
        return t

    def rename_tower(self, old: str, new: str) -> None:
        if old not in self.towers:
            raise ModelError("UNKNOWN_TOWER", f"No tower {old!r}")
        if new in self.towers and new != old:
            raise ModelError("DUPLICATE_NAME", f"Tower {new!r} already exists")
        self.towers = {(new if k == old else k): (Tower(new) if k == old else v) for k, v in self.towers.items()}
        for s in self.stories.values():
            if s.tower == old:
                s.tower = new
        for gs in self.grid_systems.values():
            if gs.tower == old:
                gs.tower = new

    def delete_tower(self, name: str) -> None:
        """Remove a tower with its stories and grid systems; refused while an object is on
        one of its stories or it is the last tower."""
        if name not in self.towers:
            raise ModelError("UNKNOWN_TOWER", f"No tower {name!r}")
        if len(self.towers) <= 1:
            raise ModelError("LAST_TOWER", "The model keeps at least one tower")
        stories = {s.name for s in self.stories_of(name)}
        used = [n for n, p in self.points.items() if p.story in stories] + \
            [n for n, f in self.frames.items() if f.story in stories] + \
            [n for n, a in self.areas.items() if a.story in stories]
        if used:
            raise ModelError("TOWER_IN_USE", f"Tower {name!r} has {len(used)} object(s) on its stories "
                                             f"(first: {used[0]}); delete them first")
        systems = set(self.grid_systems_of(name))
        first = self.default_tower() == name
        for s in stories:
            del self.stories[s]
        for g in systems:
            del self.grid_systems[g]
        self.grid_lines = [g for g in self.grid_lines if g.system not in systems]
        del self.towers[name]
        if first:  # the stories / systems that meant "the first tower" by None now name it
            for s in self.stories.values():
                s.tower = s.tower or None
        if len(self.towers) == 1:  # back to one building
            self.towers.clear()
            for s in self.stories.values():
                s.tower = None
            for gs in self.grid_systems.values():
                gs.tower = None

    def stories_bottom_up(self) -> List[Story]:
        return sorted(self.stories.values(), key=lambda s: s.elevation)

    def story_below(self, story: str) -> Optional[Story]:
        """The story below ``story`` in its own tower."""
        if story not in self.stories:
            return None
        ordered = self.stories_of(self.tower_of_story(story))
        for k, s in enumerate(ordered):
            if s.name == story:
                return ordered[k - 1] if k > 0 else None
        return None

    def story_at(self, z: float, tol: float = 1e-6, tower: Optional[str] = None,
                 xy: Optional[Tuple[float, float]] = None) -> Optional[Story]:
        """The story at elevation z: of ``tower`` when given; with several towers having a
        story there, the one whose grid covers plan point ``xy`` (else the first tower's)."""
        hits = [s for s in self.stories.values() if abs(s.elevation - z) <= tol]
        if not hits:
            return None
        if not self.multi_tower:
            return hits[0]
        if tower is None:  # the tower of the window being drawn in (set by the program per command)
            tower = getattr(self, "active_tower", None)
            if tower is not None and tower not in self.towers:
                tower = None
        if len(hits) == 1:
            own = self._tower(hits[0].tower)
            if tower is not None:
                return hits[0] if own == self._tower(tower) else None
            t = self.tower_at(*xy) if xy is not None else None
            return hits[0] if t is None or t == own else None  # inside another tower: between its levels
        if tower is not None:
            t = self._tower(tower)
            mine = [s for s in hits if self._tower(s.tower) == t]
            return mine[0] if mine else None
        t = self.tower_at(*xy) if xy is not None else None
        if t is not None:
            mine = [s for s in hits if self._tower(s.tower) == t]
            if mine:
                return mine[0]
        order = {n: k for k, n in enumerate(self.towers)}
        return min(hits, key=lambda s: order.get(self._tower(s.tower), 0))

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
        x, y, z = finite_xyz((x, y, z), "point")
        name = name or self._next_name("P", self.points)
        if name in self.points:
            raise ModelError("DUPLICATE_NAME", f"Point {name!r} already exists", f"points.{name}")
        p = Point(name=name, x=x, y=y, z=z, **kw)
        if p.story is None:
            s = self.story_at(z, self.options.merge_tolerance, xy=(x, y))
            p.story = s.name if s else None
        self.points[name] = p
        if self._pindex is not None:
            self._pindex_add(p)
        return p

    def find_point(self, x: float, y: float, z: float, tol: Optional[float] = None) -> Optional[Point]:
        tol = self.options.merge_tolerance if tol is None else tol
        idx = self._pindex
        if idx is not None and tol <= idx["cell"]:
            return self._pindex_find(x, y, z, tol)
        for p in self.points.values():
            if abs(p.x - x) <= tol and abs(p.y - y) <= tol and abs(p.z - z) <= tol:
                return p
        return None

    # A spatial hash of the points, used while a generator adds thousands of points and
    # members (New Model, its previews): find_point is then O(1) instead of a scan of every
    # point.  Only valid while nothing else moves points, so it is a context manager.
    class _PointIndex:
        def __init__(self, model: "Model"):
            self.model = model

        def __enter__(self):
            m = self.model
            cell = max(m.options.merge_tolerance, 1e-9) * 4
            m._pindex = {"cell": cell, "cells": {}, "order": {}}
            for p in m.points.values():
                m._pindex_add(p)
            return m

        def __exit__(self, *exc):
            self.model._pindex = None
            return False

    def point_index(self) -> "Model._PointIndex":
        """``with model.point_index(): ...`` - fast find_point / add_point(merge=True) while generating."""
        return Model._PointIndex(self)

    def _pindex_key(self, x, y, z):
        c = self._pindex["cell"]
        return (math.floor(x / c), math.floor(y / c), math.floor(z / c))

    def _pindex_add(self, p: Point):
        idx = self._pindex
        idx["order"].setdefault(p.name, len(idx["order"]))
        idx["cells"].setdefault(self._pindex_key(p.x, p.y, p.z), []).append(p.name)

    def _pindex_find(self, x, y, z, tol) -> Optional[Point]:
        idx = self._pindex
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return None
        i, j, k = self._pindex_key(x, y, z)
        best = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for n in idx["cells"].get((i + di, j + dj, k + dk), ()):
                        p = self.points.get(n)
                        if p is None or abs(p.x - x) > tol or abs(p.y - y) > tol or abs(p.z - z) > tol:
                            continue
                        if best is None or idx["order"][n] < idx["order"][best.name]:
                            best = p  # the first point added wins, as in the plain scan
        return best

    def add_frame(self, point_i, point_j, section: str, name: Optional[str] = None, type: Optional[str] = None,
                  **kw) -> Frame:
        """Add a frame between two points (names, Point objects or xyz tuples)."""
        if ("end_offset_i" in kw or "end_offset_j" in kw) and "auto_end_offsets" not in kw:
            kw["auto_end_offsets"] = False  # offsets given: they are the offsets
        pi = self._resolve_point(point_i)
        pj = self._resolve_point(point_j)
        if pi.name == pj.name:
            raise ModelError("ZERO_LENGTH", "Frame end points coincide")
        if type in ("Null", "Rigid") and section not in self.frame_sections:
            section = section or "None"  # null lines and rigid links carry no section
        elif section not in self.frame_sections:
            raise ModelError("UNKNOWN_SECTION", f"Frame section {section!r} is not defined")
        if type is None:
            dx, dy, dz = pj.x - pi.x, pj.y - pi.y, pj.z - pi.z
            horiz = (dx * dx + dy * dy) ** 0.5
            type = "Column" if horiz < 1e-9 * max(1.0, abs(dz)) else ("Beam" if abs(dz) < 1e-9 * max(1.0, horiz) else "Brace")
        name = name or self._next_name({"Beam": "B", "Column": "C", "Rigid": "RL"}.get(type, "D"), self.frames)
        if name in self.frames:
            raise ModelError("DUPLICATE_NAME", f"Frame {name!r} already exists", f"frames.{name}")
        f = Frame(name=name, point_i=pi.name, point_j=pj.name, section=section, type=type, **kw)
        if f.story is None:
            top = max(pi.z, pj.z)
            s = self.story_at(top, self.options.merge_tolerance, tower=self._draw_tower((pi, pj)),
                              xy=((pi.x + pj.x) / 2, (pi.y + pj.y) / 2))
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
            pp = [self.points[p] for p in pts]
            s = self.story_at(top, self.options.merge_tolerance, tower=self._draw_tower(pp),
                              xy=(sum(q.x for q in pp) / len(pp), sum(q.y for q in pp) / len(pp)))
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

        for name in nonfinite_points(self):
            p = self.points[name]
            err("NONFINITE_COORDINATE", f"Point {name!r} has a coordinate that is not a finite number "
                f"({p.x!r}, {p.y!r}, {p.z!r})", f"points.{name}")
        other = [q for q in nonfinite_values(self.to_dict()) if not q.startswith("points.")]
        for q in other[:20]:
            err("NONFINITE_VALUE", f"{q} is not a finite number (infinite or NaN)", q)
        if len(other) > 20:
            err("NONFINITE_VALUE", f"... and {len(other) - 20} more values that are not finite numbers")
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
            if f.section not in self.frame_sections and f.type not in ("Null", "Rigid"):  # no section needed
                err("UNKNOWN_SECTION", f"Frame {name!r} references unknown section {f.section!r}", f"frames.{name}")
            for ld in f.loads:
                if ld.pattern not in self.load_patterns:
                    err("UNKNOWN_PATTERN", f"Frame {name!r} load uses unknown pattern {ld.pattern!r}", f"frames.{name}")
        tol = max(self.options.merge_tolerance, 1e-9)
        for name, f in self.frames.items():
            if f.point_i in self.points and f.point_j in self.points and \
                    math.dist(self.points[f.point_i].xyz, self.points[f.point_j].xyz) <= tol:
                err("ZERO_LENGTH", f"Frame {name!r} has no length (its two ends coincide)", f"frames.{name}")
        from .checks import area_size
        for name, a in self.areas.items():
            for p in a.points:
                if p not in self.points:
                    err("UNKNOWN_POINT", f"Area {name!r} references unknown point {p!r}", f"areas.{name}")
            if all(p in self.points for p in a.points) and \
                    (len(set(a.points)) < 3 or area_size(self, a) <= tol * tol * 10):
                err("DEGENERATE_AREA", f"Area {name!r} has no area (repeated corners or corners on one line)",
                    f"areas.{name}")
            elif all(p in self.points for p in a.points) and len(a.points) > 3:
                from ..mesh import polygon2d as pg
                import numpy as _np
                P = _np.array([self.points[p].xyz for p in a.points], float)
                try:
                    Q = pg.to_2d(P, pg.plane_frame(P))
                    if pg.self_intersects(Q, tol):
                        err("SELF_INTERSECTING_AREA", f"Area {name!r}: its outline crosses itself (corners out of "
                            "order) - its area and loads would be wrong", f"areas.{name}")
                except ValueError:
                    pass
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
        self._validate_values(err)
        self._validate_dependencies(err)
        unk = list((getattr(self, "unknown_data", None) or {}).keys())
        unf = list(getattr(self, "unknown_fields", None) or [])
        if unk or unf:
            names = ", ".join((unk + unf)[:6]) + (" ..." if len(unk) + len(unf) > 6 else "")
            err("UNKNOWN_DATA", f"The file holds data this version does not know ({names}); it is kept and saved "
                "back unchanged, but the model cannot be analysed with it", (unk + unf)[0])
        return errors

    def _validate_values(self, err):
        """Ranges of material, section, support, mass and case values (meaningless values
        are refused before an analysis instead of giving results that look normal)."""
        from .loads import ModalCase, spectrum_case_problems, spectrum_function_problems
        fin = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
        for name, m in self.materials.items():
            if fin(m.nu) and not (-1.0 < m.nu < 0.5):
                err("INVALID_MATERIAL", f"Material {name!r}: Poisson's ratio {m.nu:g} must be in (-1, 0.5)",
                    f"materials.{name}.nu")
            if fin(m.weight_density) and m.weight_density < 0:
                err("INVALID_MATERIAL", f"Material {name!r}: negative weight density", f"materials.{name}.weight_density")
            if m.mass_density is not None and fin(m.mass_density) and m.mass_density < 0:
                err("INVALID_MATERIAL", f"Material {name!r}: negative mass density", f"materials.{name}.mass_density")
        used = {f.section for f in self.frames.values()}
        for name, s in self.frame_sections.items():
            if name not in used:
                continue
            for attr in ("A", "J", "I33", "I22", "As2", "As3"):
                v = getattr(s, attr, 0.0)
                if fin(v) and v < 0:
                    err("INVALID_SECTION", f"Frame section {name!r}: {attr} = {v:g} is negative",
                        f"frame_sections.{name}.{attr}")
        for name, p in self.points.items():
            for lab, vals in (("spring", p.springs), ("rotational spring", p.rotational_springs)):
                if any(fin(v) and v < 0 for v in vals):
                    err("NEGATIVE_SPRING", f"Point {name!r}: a {lab} stiffness is negative "
                        f"({', '.join(f'{v:g}' for v in vals)}) - negative springs are not supported",
                        f"points.{name}")
        for name, ls in getattr(self, "line_springs", {}).items():
            vals = list(ls.stiffness) + [ls.rotational]
            if any(fin(v) and v < 0 for v in vals):
                err("NEGATIVE_SPRING", f"Line spring {name!r}: negative stiffness", f"line_springs.{name}")
        for name, fn in self.functions.items():
            if isinstance(fn, ResponseSpectrumFunction):
                for code, msg in spectrum_function_problems(fn):
                    err(code, msg, f"functions.{name}")
        for name, c in self.load_cases.items():
            if isinstance(c, ResponseSpectrumCase):
                for code, msg in spectrum_case_problems(c):
                    err(code, msg, f"load_cases.{name}")
            elif isinstance(c, ModalCase):
                if not (isinstance(c.max_modes, int) and c.max_modes >= 1):
                    err("INVALID_MODES", f"Modal case {name!r}: the number of modes must be at least 1 "
                        f"(got {c.max_modes!r})", f"load_cases.{name}.max_modes")
                elif c.min_modes > c.max_modes:
                    err("INVALID_MODES", f"Modal case {name!r}: minimum modes {c.min_modes} > maximum {c.max_modes}",
                        f"load_cases.{name}.min_modes")
                if c.mass_source and c.mass_source not in self.mass_sources:
                    err("UNKNOWN_MASS_SOURCE", f"Modal case {name!r} uses unknown mass source {c.mass_source!r}",
                        f"load_cases.{name}.mass_source")
            elif isinstance(c, TimeHistoryCase):
                if not (fin(c.dt) and c.dt > 0) or int(c.n_steps) < 1:
                    err("INVALID_TIME_STEP", f"Time history {name!r}: needs dt > 0 and at least one step",
                        f"load_cases.{name}")
                if not fin(c.damping_ratio) or c.damping_ratio < 0.0:  # > 1 (overdamped) is allowed
                    err("INVALID_DAMPING", f"Time history {name!r}: damping ratio {c.damping_ratio!r} must be >= 0",
                        f"load_cases.{name}.damping_ratio")
        for name, ms in self.mass_sources.items():
            for pat, mult in ms.load_patterns.items():
                if pat not in self.load_patterns:
                    err("UNKNOWN_PATTERN", f"Mass source {name!r} uses unknown pattern {pat!r}", f"mass_sources.{name}")
        # automatic (code) lateral loads are not generated: a pattern that relies on one
        # would be analysed with no load at all
        used_patterns = set()
        for c in self.load_cases.values():
            used_patterns.update(getattr(c, "loads", {}) if isinstance(getattr(c, "loads", None), dict) else {})
            for ld in getattr(c, "loads", []) if isinstance(getattr(c, "loads", None), list) else []:
                if getattr(ld, "name", None):
                    used_patterns.add(ld.name)
        for name, lp in self.load_patterns.items():
            if lp.auto_load and name in used_patterns:
                err("UNSUPPORTED_AUTO_LOAD", f"Load pattern {name!r} has an automatic lateral load definition "
                    f"({', '.join(map(str, list(lp.auto_load)[:3]))}) - automatic loads are not generated by this "
                    "program; assign the loads to the pattern and remove the definition", f"load_patterns.{name}.auto_load")

    def _validate_dependencies(self, err):
        """Load cases that depend on each other in a loop, or on a case of the wrong kind."""
        from .loads import ModalCase

        def deps(c):
            d = [("stiffness", getattr(c, "stiffness_from", None)), ("modal", getattr(c, "modal_case", None)),
                 ("initial", getattr(c, "initial_case", None))]
            d += [("modal", ml[0]) for ml in getattr(c, "mode_loads", [])]
            return [(k, v) for k, v in d if v]

        for name, c in self.load_cases.items():
            for kind, dep in deps(c):
                if dep not in self.load_cases:
                    if kind == "stiffness":
                        err("UNKNOWN_CASE", f"Case {name!r}: unknown stiffness case {dep!r}", f"load_cases.{name}")
                    continue
                d = self.load_cases[dep]
                if kind == "modal" and not isinstance(d, ModalCase):
                    err("CASE_TYPE", f"Case {name!r}: {dep!r} is not a modal case", f"load_cases.{name}")
        state: Dict[str, int] = {}
        reported = set()

        def visit(n, path):
            st = state.get(n, 0)
            if st == 2:
                return
            if st == 1:
                cyc = path[path.index(n):] + [n]
                key = frozenset(cyc)
                if key not in reported:
                    reported.add(key)
                    err("CASE_CYCLE", "Load cases depend on each other in a loop: " + " -> ".join(cyc),
                        f"load_cases.{n}")
                return
            state[n] = 1
            for _, dep in deps(self.load_cases[n]):
                if dep in self.load_cases:
                    visit(dep, path + [n])
            state[n] = 2

        for n in self.load_cases:
            visit(n, [])

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
        for attr in ("materials", "rebar_sizes", "frame_sections", "shell_sections", "stories", "towers", "grid_systems", "diaphragms",
                     "points", "frames", "areas", "groups", "line_springs", "tendons", "panel_loads", "load_patterns", "mass_sources",
                     "functions", "load_cases", "load_combinations", "nonlinear_materials", "fiber_sections",
                     "frame_nonlinear", "hinge_properties", "link_properties", "links"):
            table = getattr(self, attr)
            setattr(out, attr, {k: convert_dataclass(v, src, new_units) for k, v in table.items()})
        # (a cylindrical system's T lines are angles in degrees: not lengths)
        out.grid_lines = [convert_dataclass(g, src, new_units) if g.direction != "T" else dataclasses.replace(g)
                          for g in self.grid_lines]
        out.options = dataclasses.replace(self.options)
        out.design = convert_dataclass(self.design, src, new_units)
        out.display_units = dict(self.display_units)
        ratio = src.factor("length") / new_units.factor("length")
        for name in ("floor_mesh_max_size", "wall_mesh_max_size", "fine_mesh_size_min", "fine_mesh_size_max"):
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
            "towers": to_dict(self.towers),
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
            "design": to_dict(self.design),
            **{k: v for k, v in (getattr(self, "unknown_data", None) or {}).items() if k not in _KNOWN_KEYS},
        }

    def to_json(self, path: Optional[str] = None, indent: int = 1) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        global _UNKNOWN
        ver = data.get("schema_version", "1.0")
        vt = _version_tuple(ver)
        if vt == (0,) and str(ver) not in ("0", "0.0"):
            raise ModelError("SCHEMA_VERSION", f"The file's schema version {ver!r} is not a version number; "
                                               "the file was not opened", "schema_version")
        if vt > _version_tuple(SCHEMA_VERSION):
            raise ModelError("FUTURE_SCHEMA", f"The file was saved by a newer version of the program (schema {ver}); "
                                              f"this version reads schema {SCHEMA_VERSION} and older.  The file was not "
                                              "opened, so none of its data is lost - use the newer version",
                             "schema_version")
        _UNKNOWN = []
        try:
            m = cls._from_dict_checked(data)
            m.unknown_data = {k: v for k, v in data.items() if k not in _KNOWN_KEYS}
            m.unknown_fields = list(_UNKNOWN)
        finally:
            _UNKNOWN = None
        return m

    @classmethod
    def _from_dict_checked(cls, data: dict) -> "Model":
        u = data.get("units", {})
        m = cls(data.get("info", {}).get("name", "Untitled"),
                UnitSystem(u.get("force", "kN"), u.get("length", "m"), u.get("temperature", "C")))
        known = {f.name for f in dataclasses.fields(ProjectInfo)}
        m.info = ProjectInfo(**{k: v for k, v in _strip(data.get("info", {})).items() if k in known})
        from ..display_units import force_label
        m.display_units = {str(k): force_label(v) for k, v in (data.get("display_units") or {}).items() if v}
        m.options = AnalysisOptions(**_coerce_tuples(_strip(data.get("options", {}))))
        if isinstance(data.get("design"), dict):
            m.design = _from_dict(data["design"], "design")
        for attr in ("materials", "rebar_sizes", "frame_sections", "shell_sections", "stories", "towers", "grid_systems", "diaphragms",
                     "points", "frames", "areas", "groups", "line_springs", "tendons", "panel_loads", "load_patterns", "mass_sources",
                     "functions", "load_cases", "load_combinations", "nonlinear_materials", "fiber_sections",
                     "frame_nonlinear", "hinge_properties", "link_properties", "links"):
            table = data.get(attr, {})
            setattr(m, attr, {k: _from_dict(v, f"{attr}.{k}") for k, v in table.items()})
        m.grid_lines = [_from_dict(g, f"grid_lines[{i}]") for i, g in enumerate(data.get("grid_lines", []))]
        n = max([int(k[2:]) for k in m.panel_loads if k.startswith("PL") and k[2:].isdigit()] or [0])
        if n:
            m._counters["PL"] = max(m._counters.get("PL", 0), n)
        if _version_tuple(data.get("schema_version", "1.0")) < (2, 1):
            m.convert_area_loads()  # older files: area loads stored on the areas -> area-load panels
        if _version_tuple(data.get("schema_version", "1.0")) < (2, 2):
            m.options.mesh_refine_check = False  # was saved on by default: the analysis took minutes
            m.options.mesh_refine_tolerance = max(m.options.mesh_refine_tolerance, 0.02)
        bad = nonfinite_points(m)
        if bad:  # a file with Infinity / NaN coordinates cannot be drawn or analysed: refuse it, say where
            raise ModelError("NONFINITE_COORDINATE",
                             f"{len(bad)} point(s) in the file have a coordinate that is not a finite number "
                             f"(first: {bad[0]} at {m.points[bad[0]].xyz!r}); the file was not opened",
                             f"points.{bad[0]}")
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


_UNKNOWN: Optional[List[str]] = None  # paths of unknown fields met by the current from_dict


def _from_dict(d, path: str = ""):
    if isinstance(d, dict) and "_type" in d:
        cls = _TYPE_REGISTRY.get(d["_type"])
        if cls is None:
            raise ModelError("UNKNOWN_TYPE", f"{path or 'the file'}: object type {d['_type']!r} is not known to this "
                                             "version; the file was not opened (its data would be lost)", path)
        kwargs = {}
        names = set()
        for f in dataclasses.fields(cls):
            names.add(f.name)
            if f.name not in d:
                continue
            v = d[f.name]
            kwargs[f.name] = _from_dict(v, f"{path}.{f.name}" if path else f.name)
            # dataclass tuples are serialised as lists
            if isinstance(kwargs[f.name], list) and str(f.type).startswith(("Tuple", "SixBool", "Six", "Vec3", "typing.Tuple")):
                kwargs[f.name] = tuple(kwargs[f.name])
        obj = cls(**kwargs)
        extra = {k: v for k, v in d.items() if k != "_type" and k not in names}
        if extra:  # fields of a newer version: kept (written back as read) and reported
            obj.__dict__["_unknown_fields"] = extra
            if _UNKNOWN is not None:
                _UNKNOWN.extend(f"{path}.{k}" if path else k for k in extra)
        return obj
    if isinstance(d, dict):
        return {k: _from_dict(v, f"{path}.{k}" if path else str(k)) for k, v in d.items()}
    if isinstance(d, list):
        return [_from_dict(v, f"{path}[{i}]") for i, v in enumerate(d)]
    return d


class TowerView:
    """A model seen through one of its towers: that tower's grid systems, grid lines
    and stories; everything else is the model's (read through to it).  What draws or
    snaps to grids and levels takes it in place of the model."""

    def __init__(self, model: "Model", tower):
        """``tower``: a tower, or several (a plan showing the towers that have a story
        at its level: their grids together)."""
        self._model = model
        towers = [tower] if isinstance(tower, str) or tower is None else list(tower)
        towers = list(dict.fromkeys(model._tower(t) for t in towers))
        self.tower = towers[0]
        self.towers = towers
        names = [n for t in towers for n in model.grid_systems_of(t)]
        self.grid_systems = {n: model.grid_systems[n] for n in names}
        keep = set(names)
        self.grid_lines = [g for g in model.grid_lines if g.system in keep]
        self.stories = {st.name: st for st in sorted((st for t in towers for st in model.stories_of(t)),
                                                       key=lambda st: st.elevation)}

    def __getattr__(self, name):
        return getattr(self._model, name)

    multi_tower = False

    def stories_bottom_up(self) -> List[Story]:
        return list(self.stories.values())

    def levels(self) -> List[Tuple[str, float]]:
        return [(st.name, st.elevation) for st in self.stories.values()]

    def story_at(self, z: float, tol: float = 1e-6, tower: Optional[str] = None, xy=None) -> Optional[Story]:
        return self._model.story_at(z, tol, tower=self.tower, xy=xy)

    def view(self, tower=None):
        return self if tower is None or (isinstance(tower, str) and self._model._tower(tower) == self.tower
                                         and len(self.towers) == 1) else self._model.view(tower)


class _TowerGrids:
    """The grid systems and lines of one tower, as ``grid_geometry`` reads a model."""

    def __init__(self, model: "Model", tower: str):
        names = set(model.grid_systems_of(tower))
        self.grid_systems = {n: model.grid_systems[n] for n in names}
        self.grid_lines = [g for g in model.grid_lines if g.system in names]
