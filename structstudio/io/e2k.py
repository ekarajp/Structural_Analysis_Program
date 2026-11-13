"""Import of ETABS text model files (``.e2k`` / ``.$et``).

The ETABS text format is a sequence of ``$ SECTION`` blocks containing one
record per line: ``KEYWORD "name" KEY value KEY "value" ...``.  This reader
covers the subset needed for linear analysis models (stories, grids,
materials, frame/shell sections, points, lines, areas, assignments, loads,
load patterns/cases/combinations, response spectrum functions, mass source,
active DOFs).  Everything it does not understand is reported in
:attr:`E2KImporter.skipped` so nothing is silently lost.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..model import (FIXED, FREE, Area, Diaphragm, Frame, FrameModifiers, FrameSection, LoadPattern, MassSource,
                     Material, Model, ModelError, Point, ShellModifiers, ShellSection, Story)
from ..model.loads import (ComboItem, LinearStaticCase, LoadCombination, ModalCase, ResponseSpectrumCase,
                           ResponseSpectrumFunction, SpectrumLoad)
from ..units import UnitSystem
from .section_db import DEFAULT_SEARCH_PATHS, SectionLibrary

_TOKEN = re.compile(r'"([^"]*)"|(\S+)')


def tokenize(line: str) -> List[Tuple[str, bool]]:
    """Return (token, was_quoted) pairs."""
    out = []
    for m in _TOKEN.finditer(line):
        if m.group(1) is not None:
            out.append((m.group(1), True))
        else:
            out.append((m.group(2), False))
    return out


def _num(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


def _kv(tokens: List[Tuple[str, bool]], start: int) -> Dict[str, object]:
    """Parse KEY value pairs; keys are unquoted uppercase words.  A key may be
    followed by a quoted string, a number, or a bare word (YES/NO).  Keys
    without a value (flags) map to True."""
    d: Dict[str, object] = {}
    i = start
    n = len(tokens)
    while i < n:
        tok, quoted = tokens[i]
        if quoted or _num(tok) is not None:
            i += 1
            continue
        key = tok.upper()
        if i + 1 < n:
            val, vq = tokens[i + 1]
            if vq:
                d[key] = val
                i += 2
                continue
            num = _num(val)
            if num is not None:
                # collect consecutive numbers (e.g. OFFSETS x y z)
                nums = [num]
                j = i + 2
                while j < n and not tokens[j][1] and _num(tokens[j][0]) is not None:
                    nums.append(_num(tokens[j][0]))
                    j += 1
                d[key] = nums[0] if len(nums) == 1 else nums
                i = j
                continue
            if val.upper() in ("YES", "NO", "TRUE", "FALSE"):
                d[key] = val.upper() in ("YES", "TRUE")
                i += 2
                continue
            # bare word value (e.g. TYPE RIGID)
            d[key] = val
            i += 2
            continue
        d[key] = True
        i += 1
    return d


@dataclass
class E2KImporter:
    model: Model = field(default_factory=lambda: Model("imported"))
    skipped: Counter = field(default_factory=Counter)
    warnings: List[str] = field(default_factory=list)
    # ETABS story-based points: label -> (x, y) and generated point names
    _plan_points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    _story_order: List[Tuple[str, Optional[float], Optional[float]]] = field(default_factory=list)
    _rs_functions: Dict[str, dict] = field(default_factory=dict)
    _pending_cases: Dict[str, dict] = field(default_factory=dict)
    _pending_combos: Dict[str, list] = field(default_factory=dict)
    _frame_mods: Dict[str, FrameModifiers] = field(default_factory=dict)
    _area_defs: Dict[str, Tuple[str, List[str], List[int]]] = field(default_factory=dict)
    _line_defs: Dict[str, Tuple[str, str, str, int]] = field(default_factory=dict)
    _section_defaults: Dict[str, dict] = field(default_factory=dict)
    section_library: SectionLibrary = field(default_factory=lambda: SectionLibrary(DEFAULT_SEARCH_PATHS))
    source_dir: Optional[str] = None

    # ------------------------------------------------------------------ api
    @classmethod
    def read(cls, path: str, encoding: str = "utf-8") -> "E2KImporter":
        with open(path, encoding=encoding, errors="replace") as fh:
            text = fh.read()
        imp = cls()
        imp.source_dir = os.path.dirname(os.path.abspath(path))
        imp.section_library.search_paths.insert(0, imp.source_dir)
        imp._parse(text)
        return imp

    @classmethod
    def parse(cls, text: str) -> "E2KImporter":
        imp = cls()
        imp._parse(text)
        return imp

    # ---------------------------------------------------------------- parse
    def _parse(self, text: str):
        self.model.use_csi_frame_defaults()  # frames keep what the file says
        section = ""
        lines = text.splitlines()
        records: List[Tuple[str, List[Tuple[str, bool]]]] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("$"):
                section = line[1:].strip().upper()
                continue
            records.append((section, tokenize(line)))
        # pass 1: definitions that others depend on
        handlers_pass1 = {
            "CONTROLS": self._controls, "STORIES - IN SEQUENCE FROM TOP": self._story,
            "GRIDS": self._grid, "DIAPHRAGM NAMES": self._diaphragm, "MATERIAL PROPERTIES": self._material,
            "FRAME SECTIONS": self._frame_section, "SLAB PROPERTIES": self._shell_prop,
            "WALL PROPERTIES": self._shell_prop, "DECK PROPERTIES": self._shell_prop,
            "WALL/SLAB/DECK PROPERTIES": self._shell_prop, "STATIC LOADS": self._static_load_v9,
            "RESPONSE SPECTRUM FUNCTIONS": self._function,
            "POINT COORDINATES": self._point, "LOAD PATTERNS": self._load_pattern,
            "FUNCTIONS": self._function, "ANALYSIS OPTIONS": self._analysis_options,
            "MASS SOURCE": self._mass_source, "FRAME PROPERTY MODIFIERS": self._frame_mods_def,
        }
        for sec, toks in records:
            h = handlers_pass1.get(sec)
            if h:
                h(toks)
        self._finish_stories()
        # pass 2: objects
        for sec, toks in records:
            if sec == "LINE CONNECTIVITIES":
                self._line(toks)
            elif sec == "AREA CONNECTIVITIES":
                self._area(toks)
        # pass 3: assignments and loads
        handlers_pass3 = {
            "POINT ASSIGNS": self._point_assign, "LINE ASSIGNS": self._line_assign,
            "AREA ASSIGNS": self._area_assign, "POINT OBJECT LOADS": self._point_load,
            "FRAME OBJECT LOADS": self._line_load, "AREA OBJECT LOADS": self._area_load,
            "LOAD CASES": self._load_case, "LOAD COMBINATIONS": self._combo,
            "LOADING COMBINATIONS": self._combo, "RESPONSE SPECTRUM CASES": self._spec_case_v9,
        }
        handled = set(handlers_pass1) | set(handlers_pass3) | {"LINE CONNECTIVITIES", "AREA CONNECTIVITIES",
                                                                "PROGRAM INFORMATION", "LOG", "PROJECT INFORMATION",
                                                                "TABLE SETS", "DATABASE TABLE NAMED SETS"}
        for sec, toks in records:
            h = handlers_pass3.get(sec)
            if h:
                h(toks)
            elif sec not in handled:
                self.skipped[sec] += 1
        self._finish_cases()
        self._finish_combos()
        self.model.convert_area_loads()  # ETABS area loads -> area-load panels

    # ---------------------------------------------------------------- pass 1
    def _controls(self, toks):
        if toks[0][0].upper() == "UNITS":
            force, length = toks[1][0], toks[2][0]
            temp = toks[3][0] if len(toks) > 3 and toks[3] else "C"  # UNITS "KIP" "IN" "F"
            try:
                units = UnitSystem.parse(f"{force} {length} {temp}")
            except ValueError:
                units = UnitSystem.parse("kN m")
                self.warnings.append(f"Unknown units {force} {length}; assuming kN-m")
            self.model.units = units
            self.model.options.auto_base_restraints = True
            # ETABS gravity constant per unit system
            self.model.options.gravity = 386.4 if units.length in ("in",) else (32.2 if units.length == "ft" else None)
        elif toks[0][0].upper() == "TITLE2":
            self.model.info.name = toks[1][0]
        elif toks[0][0].upper() == "PREFERENCE":
            d = _kv(toks, 1)
            if "MERGETOL" in d:
                self.model.options.merge_tolerance = float(d["MERGETOL"])

    def _story(self, toks):
        if toks[0][0].upper() != "STORY":
            return
        name = toks[1][0]
        d = _kv(toks, 2)
        self._story_order.append((name, d.get("HEIGHT"), d.get("ELEV")))

    def _finish_stories(self):
        # list is top -> bottom; compute elevations bottom-up
        order = list(reversed(self._story_order))
        z = 0.0
        for k, (name, height, elev) in enumerate(order):
            if elev is not None:
                z = float(elev)
                h = 0.0
            else:
                h = float(height or 0.0)
                z = z + h
            self.model.stories[name] = Story(name=name, elevation=z, height=h)

    def _grid(self, toks):
        from ..model.objects import GridLine, GridSystem
        kind = toks[0][0].upper()
        if kind == "GRIDSYSTEM":
            d = _kv(toks, 2)
            self.model.grid_systems[toks[1][0]] = GridSystem(name=toks[1][0], type=str(d.get("TYPE", "Cartesian")).title(),
                                                             bubble_size=float(d.get("BUBBLESIZE", 1.0)),
                                                             origin=(float(d.get("UX", 0.0)), float(d.get("UY", 0.0))),
                                                             rotation=float(d.get("ROTATION", d.get("ROTZ", 0.0))))
        elif kind == "GRID":
            d = _kv(toks, 2)
            self.model.grid_lines.append(GridLine(system=toks[1][0], name=str(d.get("LABEL", "")),
                                                  direction=str(d.get("DIR", "X")).upper(),
                                                  ordinate=float(d.get("COORD", 0.0)),
                                                  bubble_location=str(d.get("BUBBLELOC", "End")),
                                                  visible=bool(d.get("VISIBLE", True))))

    def _diaphragm(self, toks):
        d = _kv(toks, 2)
        rigid = str(d.get("TYPE", "RIGID")).upper() == "RIGID"
        self.model.diaphragms[toks[1][0]] = Diaphragm(toks[1][0], rigid)

    def _material(self, toks):
        name = toks[1][0]
        d = _kv(toks, 2)
        mat = self.model.materials.get(name)
        if mat is None:
            mat = Material(name=name, E=0.0)
            self.model.materials[name] = mat
        if "TYPE" in d and str(d["TYPE"]).upper() not in ("ISOTROPIC", "ORTHOTROPIC", "ANISOTROPIC"):
            mat.type = str(d["TYPE"])
        if "DESIGNTYPE" in d:  # ETABS v9
            mat.type = str(d["DESIGNTYPE"]).title()
        if "WEIGHTPERVOLUME" in d or "W" in d:
            mat.weight_density = float(d.get("WEIGHTPERVOLUME", d.get("W")))
        if "MASSPERVOLUME" in d or "M" in d:
            mat.mass_density = float(d.get("MASSPERVOLUME", d.get("M")))
        if "E" in d and not isinstance(d["E"], bool):
            mat.E = float(d["E"])
        if "U" in d and not isinstance(d["U"], bool):
            mat.nu = float(d["U"])
        if "G" in d and not isinstance(d["G"], bool):
            mat.G = float(d["G"])
        if "A" in d and not isinstance(d["A"], bool):
            mat.alpha = float(d["A"])
        for key, attr in (("FC", "fc"), ("FY", "fy"), ("FU", "fu")):
            if key in d and not isinstance(d[key], bool):
                setattr(mat, attr, float(d[key]))

    def _frame_mods_def(self, toks):
        name = toks[1][0]
        d = _kv(toks, 2)
        self._frame_mods[name] = _mods_from(d)

    def _frame_section(self, toks):
        name = toks[1][0]
        d = _kv(toks, 2)
        sec = self.model.frame_sections.get(name)
        material = str(d.get("MATERIAL", sec.material if sec else ""))
        shape = str(d.get("SHAPE", sec.shape if sec else "General"))
        D = d.get("D")
        B = d.get("B")
        shape_l = shape.lower()
        if sec is None and (d.get("FILE") or d.get("SECTINFILE")):
            # the database label is SECTINFILE, else the SHAPE key (e.g. SHAPE "W14x90"
            # for section "W14x90-CP1"), else the section name itself
            hit = None
            for label in (d.get("SECTINFILE"), d.get("SHAPE"), name):
                if label:
                    hit = self.section_library.find(str(label), str(d.get("FILE", "")) or None)
                    if hit is not None:
                        break
            if hit is not None:
                sec = hit.to_frame_section(name, material, self.model.units)
                self.model.frame_sections[name] = sec
            else:
                self.warnings.append(f"Section {name!r}: database {d.get('FILE')!r} not found; "
                                     f"properties computed from dimensions")
        if sec is None:
            if "rectang" in shape_l and D and B:
                sec = FrameSection.rectangular(name, material, float(D), float(B))
            elif "circle" in shape_l and D:
                sec = FrameSection.circular(name, material, float(D))
            elif ("wide flange" in shape_l or shape_l.endswith(" i")) and D and B and d.get("TF") and d.get("TW"):
                sec = FrameSection.i_shape(name, material, float(D), float(B), float(d["TF"]), float(d["TW"]))
            else:
                sec = FrameSection(name=name, material=material, shape="General")
                if D:
                    sec.t3 = float(D)
                if B:
                    sec.t2 = float(B)
            sec.shape = shape
            self.model.frame_sections[name] = sec
        # explicit properties override
        for key, attr in (("AREA", "A"), ("I33", "I33"), ("I22", "I22"), ("J", "J"), ("AS2", "As2"), ("AS3", "As3")):
            if key in d and not isinstance(d[key], bool):
                setattr(sec, attr, float(d[key]))
        if "general" in shape_l:
            # CSI: unspecified general properties are zero
            pass
        mods = _mods_from(d)
        if mods is not None:
            sec.modifiers = mods
        if "DESIGNTYPE" in d:
            sec.design_type = str(d["DESIGNTYPE"])
        if "RIGIDZONE" in d:
            self._section_defaults.setdefault(name, {})["rigid"] = float(d["RIGIDZONE"])

    def _shell_prop(self, toks):
        name = toks[1][0]
        d = _kv(toks, 2)
        ptype = str(d.get("PROPTYPE", "Slab"))
        material = str(d.get("MATERIAL", d.get("CONCMATERIAL", "")))
        thickness = d.get("SLABTHICKNESS", d.get("WALLTHICKNESS", d.get("DECKSLABDEPTH",
                          d.get("MEMBRANE", d.get("THICKNESS", 0.0)))))
        model_type = str(d.get("MODELINGTYPE", d.get("TYPE", "ShellThin")))
        mt = model_type.replace("-", "").replace(" ", "").lower()
        model_type = ("Membrane" if "membrane" in mt else "Plate" if "plate" in mt else
                      "ShellThick" if "thick" in mt else "ShellThin")
        if ptype.lower() == "deck":
            self.warnings.append(f"Deck property {name!r} imported as a slab of the concrete fill depth")
        sec = ShellSection(name=name, material=material, type=ptype.title(), model_type=model_type,
                           thickness=float(thickness or 0.0))
        if d.get("BENDING") not in (None, True, False) and float(d["BENDING"]) != sec.thickness:
            sec.bending_thickness = float(d["BENDING"])
        sm = _shell_mods_from(d)
        if sm is not None:
            sec.modifiers = sm
        self.model.shell_sections[name] = sec

    def _point(self, toks):
        label = toks[1][0]
        nums = [_num(t) for t, q in toks[2:] if not q and _num(t) is not None]
        if len(nums) >= 2:
            self._plan_points[label] = (nums[0], nums[1])
            if len(nums) >= 3:
                self._plan_points[label] = (nums[0], nums[1])

    def _load_pattern(self, toks):
        name = toks[1][0]
        d = _kv(toks, 2)
        typ = str(d.get("TYPE", "Other")).replace(" ", "")
        self.model.load_patterns[name] = LoadPattern(name, typ, float(d.get("SELFWEIGHT", 0.0) or 0.0))

    def _function(self, toks):
        name = toks[1][0]
        d = _kv(toks, 2)
        fn = self._rs_functions.setdefault(name, {"periods": [], "values": [], "damping": 0.05, "type": None})
        if "FUNCTYPE" in d:
            fn["type"] = str(d["FUNCTYPE"]).upper()
        if "DAMPRATIO" in d:
            fn["damping"] = float(d["DAMPRATIO"])
        tv = d.get("TIMEVAL")
        if isinstance(tv, str):
            nums = [float(x) for x in tv.replace(",", " ").split()]
            for k in range(0, len(nums) - 1, 2):
                fn["periods"].append(nums[k])
                fn["values"].append(nums[k + 1])
        else:
            for pk in ("TIMEVAL", "PERIOD", "TIME"):
                if pk in d and "VALUE" in d:
                    fn["periods"].append(float(d[pk]))
                    fn["values"].append(float(d["VALUE"]))
                    break
        if "FUNCTYPE" in d and str(d["FUNCTYPE"]).upper() == "HISTORY":
            self.skipped["FUNCTION time history"] += 1

    def _analysis_options(self, toks):
        kind = toks[0][0].upper()
        if kind == "ACTIVEDOF":
            dofs = toks[1][0].upper().split()
            self.model.options.active_dofs = tuple(n in dofs for n in ("UX", "UY", "UZ", "RX", "RY", "RZ"))
        elif kind in ("DYNAMICANALYSIS", "DYNAMICS") or "NUMMODES" in _kv(toks, 1):
            d = _kv(toks, 1)
            if "NUMMODES" in d:
                self._v9_modes = int(float(d["NUMMODES"]))
        elif kind == "AUTOMESHOPTIONS":
            d = _kv(toks, 1)
            if "FLOORMESHMAXSIZE" in d:
                self.model.options.floor_mesh_max_size = float(d["FLOORMESHMAXSIZE"])
            if "WALLMESHMAXSIZE" in d:
                self.model.options.wall_mesh_max_size = float(d["WALLMESHMAXSIZE"])

    def _mass_source(self, toks):
        name = toks[1][0]
        d = _kv(toks, 2)
        ms = self.model.mass_sources.get(name) or MassSource(name=name)
        ms.include_element_mass = bool(d.get("INCLUDEELEMENTS", ms.include_element_mass))
        ms.include_added_mass = bool(d.get("INCLUDEADDEDMASS", ms.include_added_mass))
        ms.include_lateral = bool(d.get("INCLUDELATERALMASS", ms.include_lateral))
        ms.include_vertical = bool(d.get("INCLUDEVERTICALMASS", ms.include_vertical))
        ms.lump_at_stories = bool(d.get("LUMPATSTORIES", ms.lump_at_stories))
        ms.is_default = bool(d.get("ISDEFAULT", True))
        if "LOADPAT" in d:
            ms.load_patterns[str(d["LOADPAT"])] = float(d.get("MULTIPLIER", 1.0))
        include_loads = d.get("INCLUDELOADS")
        if include_loads is False:
            ms.load_patterns.clear()
        if ms.is_default:
            for other in self.model.mass_sources.values():
                other.is_default = False
        self.model.mass_sources[name] = ms
        if "MsSrc1" in self.model.mass_sources and name != "MsSrc1" and not self.model.mass_sources["MsSrc1"].load_patterns:
            del self.model.mass_sources["MsSrc1"]

    # ---------------------------------------------------------------- pass 2
    def _joint(self, label: str, story: str) -> Point:
        """Point object for an ETABS plan point at a story (created on demand)."""
        name = f"{label}@{story}"
        if name in self.model.points:
            return self.model.points[name]
        if label not in self._plan_points:
            raise ModelError("UNKNOWN_POINT", f"e2k: point {label!r} is not defined")
        if story not in self.model.stories:
            raise ModelError("UNKNOWN_STORY", f"e2k: story {story!r} is not defined")
        x, y = self._plan_points[label]
        z = self.model.stories[story].elevation
        p = Point(name=name, x=x, y=y, z=z, label=label, story=story)
        self.model.points[name] = p
        return p

    def _line(self, toks):
        name = toks[1][0]
        kind = toks[2][0].upper()
        p1, p2 = toks[3][0], toks[4][0]
        span = int(_num(toks[5][0]) or 1) if len(toks) > 5 else 1
        self._line_defs[name] = (kind, p1, p2, span)

    def _area(self, toks):
        name = toks[1][0]
        kind = toks[2][0].upper()
        rest = toks[3:]
        n = None
        if rest and not rest[0][1] and _num(rest[0][0]) is not None:
            n = int(_num(rest[0][0]))
            rest = rest[1:]
        pts = [t for t, q in rest if q]
        nums = [int(_num(t)) for t, q in rest if not q and _num(t) is not None]
        if n is not None:
            pts = pts[:n]
            offsets = (nums + [0] * n)[:n]
        else:
            offsets = [0] * len(pts)
        self._area_defs[name] = (kind, pts, offsets)

    # ---------------------------------------------------------------- pass 3
    def _point_assign(self, toks):
        label, story = toks[1][0], toks[2][0]
        d = _kv(toks, 3)
        p = self._joint(label, story)
        if "RESTRAINT" in d:
            dofs = str(d["RESTRAINT"]).upper().split()
            p.restraints = tuple(n in dofs for n in ("UX", "UY", "UZ", "RX", "RY", "RZ"))
        if "DIAPH" in d:
            p.diaphragm = str(d["DIAPH"])
        mass = list(p.mass)
        rmass = list(p.rotational_mass)
        if "MASSUXUY" in d:
            mass[0] += float(d["MASSUXUY"])
            mass[1] += float(d["MASSUXUY"])
        for key, idx in (("MASSUX", 0), ("MASSUY", 1), ("MASSUZ", 2)):
            if key in d:
                mass[idx] += float(d[key])
        for key, idx in (("MASSRX", 0), ("MASSRY", 1), ("MASSRZ", 2)):
            if key in d:
                rmass[idx] += float(d[key])
        vals = d.get("MASS", d.get("USERJOINTMASS"))
        if isinstance(vals, list):
            vals = [float(v) for v in vals] + [0.0] * 6
            mass = [mass[i] + vals[i] for i in range(3)]
            rmass = [rmass[i] + vals[3 + i] for i in range(3)]
        p.mass = tuple(mass)
        p.rotational_mass = tuple(rmass)
        if "SPRINGPROP" in d:
            self.skipped["POINTASSIGN SPRINGPROP"] += 1
        if "PANELZONE" in d:
            p.panel_zone = str(d["PANELZONE"])

    def _line_assign(self, toks):
        label, story = toks[1][0], toks[2][0]
        d = _kv(toks, 3)
        if label not in self._line_defs:
            self.skipped["LINEASSIGN (unknown line)"] += 1
            return
        kind, p1, p2, span = self._line_defs[label]
        name = f"{label}@{story}"
        if kind == "COLUMN":
            below = self._story_below(story, span)
            pi = self._joint(p1, below)
            pj = self._joint(p2, story)
            ftype = "Column"
        elif kind == "BRACE":
            below = self._story_below(story, span)
            pi = self._joint(p1, below)
            pj = self._joint(p2, story)
            ftype = "Brace"
        else:
            pi = self._joint(p1, story)
            pj = self._joint(p2, story)
            ftype = "Beam" if kind == "BEAM" else "Null"
        section = str(d.get("SECTION", ""))
        if section not in self.model.frame_sections:
            self.skipped[f"LINEASSIGN section {section}"] += 1
            return
        fr = Frame(name=name, point_i=pi.name, point_j=pj.name, section=section, type=ftype, label=label, story=story)
        # CARDINALPT 0 (or absent) in exported files = centroid; explicit 1-11 as given
        fr.cardinal_point = int(d.get("CARDINALPT", 10) or 10)
        fr.display_cardinal = fr.cardinal_point if "CARDINALPT" in d else None
        if "RELEASE" in d:
            fr.releases_i, fr.releases_j = _releases(str(d["RELEASE"]))
        if "ANG" in d:
            fr.angle = float(d["ANG"])
        if "LENGTHOFFI" in d or "LENGTHOFFJ" in d:
            fr.end_offset_i = float(d.get("LENGTHOFFI", 0.0) or 0.0)
            fr.end_offset_j = float(d.get("LENGTHOFFJ", 0.0) or 0.0)
        else:
            fr.auto_end_offsets = True  # ETABS: offsets from connecting member sizes
        if "MASS" in d and not isinstance(d["MASS"], bool):
            fr.mass_per_length = float(d["MASS"])
        rigid = d.get("RIGIDZONE")
        if rigid is None:
            rigid = self._section_defaults.get(section, {}).get("rigid", 0.0)
        fr.rigid_factor = float(rigid or 0.0)
        if "PROPMODS" in d:
            mods = self._frame_mods.get(str(d["PROPMODS"]))
            if mods:
                fr.modifiers = mods
        mods = _mods_from(d)
        if mods is not None:
            fr.modifiers = mods if fr.modifiers is None else fr.modifiers.combined(mods)
        if "MAXSTASPC" in d:
            fr.max_station_spacing = float(d["MAXSTASPC"])
        if "MINNUMSTA" in d:
            fr.min_stations = int(d["MINNUMSTA"])
        if "AUTOMESH" in d:
            fr.auto_mesh = bool(d["AUTOMESH"])
        if "MESHATINTERSECTIONS" in d:
            fr.mesh_at_intersections = bool(d["MESHATINTERSECTIONS"])
        offs = d.get("OFFSETS")
        if isinstance(offs, list) and len(offs) >= 6:
            fr.joint_offset_i = tuple(float(v) for v in offs[:3])
            fr.joint_offset_j = tuple(float(v) for v in offs[3:6])
        if d.get("TRANSFORMSTIFF") is not True:
            # ETABS only transforms the frame stiffness for insertion-point offsets when
            # TRANSFORMSTIFF "Yes" is set; otherwise the cardinal point is graphical and the
            # member acts on the joint line (spandrels of Verification Ex.15c then carry no
            # axial force in a rigid diaphragm, exactly as in ETABS 21).
            fr.cardinal_point = 10
        self.model.frames[name] = fr

    def _story_below(self, story: str, span: int = 1) -> str:
        ordered = self.model.stories_bottom_up()
        names = [s.name for s in ordered]
        k = names.index(story)
        if k - span < 0:
            raise ModelError("INVALID_STORY", f"e2k: no story {span} below {story!r}")
        return names[k - span]

    def _area_assign(self, toks):
        label, story = toks[1][0], toks[2][0]
        d = _kv(toks, 3)
        if label not in self._area_defs:
            self.skipped["AREAASSIGN (unknown area)"] += 1
            return
        kind, pts, offsets = self._area_defs[label]
        name = f"{label}@{story}"
        if kind == "PANEL":  # wall: corner points with per-point story offsets (1 = story below)
            if len(pts) == 2:
                below = self._story_below(story)
                p = [self._joint(pts[0], below), self._joint(pts[1], below), self._joint(pts[1], story), self._joint(pts[0], story)]
            else:
                p = [self._joint(q, self._story_below(story, off) if off else story) for q, off in zip(pts, offsets)]
            atype = "Wall"
        else:
            p = [self._joint(q, self._story_below(story, off) if off else story) for q, off in zip(pts, offsets)]
            atype = "Floor" if kind == "FLOOR" else ("Wall" if kind == "WALL" else "Null")
        # drop consecutive duplicates
        dedup = []
        for q in p:
            if not dedup or dedup[-1].name != q.name:
                dedup.append(q)
        if len(dedup) > 1 and dedup[0].name == dedup[-1].name:
            dedup.pop()
        p = dedup
        if len(p) < 3:
            self.skipped["AREAASSIGN degenerate area"] += 1
            return
        section = d.get("SECTION")
        if section is not None and str(section) not in self.model.shell_sections:
            if str(section).upper() in ("NONE", "OPENING"):
                section = None
            else:
                self.skipped[f"AREAASSIGN section {section}"] += 1
                return
        area = Area(name=name, points=[q.name for q in p], section=None if section is None else str(section),
                    type=atype if section is not None else "Opening", label=label, story=story)
        if "DIAPH" in d:
            area.diaphragm = str(d["DIAPH"])
        if "MESHMAXSIZE" in d:
            area.max_mesh_size = float(d["MESHMAXSIZE"])
        mesh_type = str(d.get("OBJMESHTYPE", "DEFAULT")).upper()
        if atype == "Wall" and mesh_type in ("DEFAULT", "NONE", "NOMESH"):
            # ETABS default for wall objects: one shell element per wall panel
            # (verified against ETABS 21 results of Verification Example 15)
            area.auto_mesh = False
        sm = _shell_mods_from(d)
        if sm is not None:
            area.modifiers = sm
        self.model.areas[name] = area

    def _point_load(self, toks):
        label, story = toks[1][0], toks[2][0]
        d = _kv(toks, 3)
        p = self._joint(label, story)
        pattern = str(d.get("LC", ""))
        if pattern not in self.model.load_patterns:
            self.skipped[f"POINTLOAD pattern {pattern}"] += 1
            return
        from ..model import PointLoad
        force = tuple(float(d.get(k, 0.0) or 0.0) for k in ("FX", "FY", "FZ"))
        moment = tuple(float(d.get(k, 0.0) or 0.0) for k in ("MX", "MY", "MZ"))
        p.loads.append(PointLoad(pattern, force, moment))

    def _line_load(self, toks):
        label, story = toks[1][0], toks[2][0]
        d = _kv(toks, 3)
        name = f"{label}@{story}"
        fr = self.model.frames.get(name)
        if fr is None:
            self.skipped["LINELOAD (unknown frame)"] += 1
            return
        pattern = str(d.get("LC", ""))
        if pattern not in self.model.load_patterns:
            self.skipped[f"LINELOAD pattern {pattern}"] += 1
            return
        ltype = str(d.get("TYPE", "UNIFF")).upper()
        direction = _direction(str(d.get("DIR", "GRAV")))
        load_type = "Moment" if ltype.endswith("M") else "Force"
        from ..model import FrameLoad
        if ltype.startswith("POINT"):
            if "RDIST" in d:
                fr.loads.append(FrameLoad(pattern, "point", direction, load_type, float(d["RDIST"]), float(d["RDIST"]),
                                          False, float(d.get("FVAL", 0.0)), float(d.get("FVAL", 0.0))))
            else:
                fr.loads.append(FrameLoad(pattern, "point", direction, load_type, float(d.get("DIST", 0.0)),
                                          float(d.get("DIST", 0.0)), True, float(d.get("FVAL", 0.0)), float(d.get("FVAL", 0.0))))
        elif ltype.startswith("UNIF"):
            fr.loads.append(FrameLoad(pattern, "distributed", direction, load_type, 0.0, 1.0, False,
                                      float(d.get("FVAL", 0.0)), float(d.get("FVAL", 0.0))))
        elif ltype.startswith("TRAP"):
            if "RDSTART" in d:
                fr.loads.append(FrameLoad(pattern, "distributed", direction, load_type, float(d["RDSTART"]),
                                          float(d["RDEND"]), False, float(d.get("FSTART", 0.0)), float(d.get("FEND", 0.0))))
            else:
                fr.loads.append(FrameLoad(pattern, "distributed", direction, load_type, float(d.get("DSTART", 0.0)),
                                          float(d.get("DEND", 0.0)), True, float(d.get("FSTART", 0.0)), float(d.get("FEND", 0.0))))
        else:
            self.skipped[f"LINELOAD type {ltype}"] += 1

    def _area_load(self, toks):
        label, story = toks[1][0], toks[2][0]
        d = _kv(toks, 3)
        area = self.model.areas.get(f"{label}@{story}")
        if area is None:
            self.skipped["AREALOAD (unknown area)"] += 1
            return
        pattern = str(d.get("LC", ""))
        if pattern not in self.model.load_patterns:
            self.skipped[f"AREALOAD pattern {pattern}"] += 1
            return
        ltype = str(d.get("TYPE", "UNIFF")).upper()
        if ltype != "UNIFF":
            self.skipped[f"AREALOAD type {ltype}"] += 1
            return
        from ..model import AreaLoad
        area.loads.append(AreaLoad(pattern, float(d.get("VALUE", 0.0) or 0.0), _direction(str(d.get("DIR", "GRAV")))))

    def _load_case(self, toks):
        name = toks[1][0]
        d = _kv(toks, 2)
        case = self._pending_cases.setdefault(name, {"type": None, "loads": {}, "rs_loads": [], "opts": {}})
        if "TYPE" in d:
            case["type"] = str(d["TYPE"])
        if "LOADPAT" in d:
            case["loads"][str(d["LOADPAT"])] = float(d.get("SF", 1.0))
        if "ACCEL" in d and ("FUNC" in d or "FUNCTION" in d):
            case["rs_loads"].append((str(d["ACCEL"]), str(d.get("FUNC", d.get("FUNCTION", ""))), float(d.get("SF", 1.0))))
            if "ANGLE" in d:
                case["opts"]["ANGLE"] = float(d["ANGLE"])
        elif "LOADTYPE" in d and str(d["LOADTYPE"]).upper() == "ACCEL":
            case["rs_loads"].append((str(d.get("LOADNAME", "U1")), str(d.get("FUNC", d.get("FUNCTION", ""))),
                                     float(d.get("SF", 1.0))))
        for k, v in d.items():
            if k not in ("TYPE", "LOADPAT", "SF", "LOADTYPE", "LOADNAME", "FUNC", "FUNCTION", "ACCEL"):
                case["opts"][k] = v

    # ------------------------------------------------------------ ETABS v9
    def _static_load_v9(self, toks):
        """ETABS v9 ``$ STATIC LOADS``: LOADCASE "DEAD" TYPE "DEAD" SELFWEIGHT 1 –
        a load pattern with an implied linear static case of the same name."""
        if toks[0][0].upper() not in ("LOADCASE", "LOAD"):
            return
        name = toks[1][0]
        d = _kv(toks, 2)
        typ = str(d.get("TYPE", "Other")).title().replace(" ", "")
        self.model.load_patterns[name] = LoadPattern(name, typ, float(d.get("SELFWEIGHT", 0.0) or 0.0))
        self._pending_cases.setdefault(name, {"type": "Linear Static", "loads": {name: 1.0}, "rs_loads": [], "opts": {}})
        if d.get("AUTOLOAD") or (d.get("TYPE") and str(d["TYPE"]).upper() in ("QUAKE", "WIND") and len(d) > 2):
            self.skipped["STATIC LOADS automatic lateral load"] += 1

    def _spec_case_v9(self, toks):
        """ETABS v9 ``$ RESPONSE SPECTRUM CASES``: SPECCASE "S" DAMP 0.05 MODECOMB "CQC"
        DIRCOMB "SRSS" / SPECCASE "S" U1 "FUNC" SF 9.81 ANG 0 ECCEN 0.05."""
        name = toks[1][0]
        d = _kv(toks, 2)
        case = self._pending_cases.setdefault(name, {"type": "Response Spectrum", "loads": {}, "rs_loads": [],
                                                     "opts": {}})
        for direction in ("U1", "U2", "UZ", "U3"):
            if direction in d:
                case["rs_loads"].append(("U3" if direction == "UZ" else direction, str(d[direction]),
                                         float(d.get("SF", 1.0))))
        for k_old, k_new in (("DAMP", "CONSTDAMP"), ("MODECOMB", "MODECOMBO"), ("DIRCOMB", "DIRCOMBO"),
                             ("ANG", "ANGLE"), ("ECCEN", "ECCEN")):
            if k_old in d:
                case["opts"][k_new] = d[k_old]
        if not any((c["type"] or "").lower().startswith("modal") for c in self._pending_cases.values()):
            self._pending_cases["MODAL"] = {"type": "Modal - Eigen", "loads": {}, "rs_loads": [],
                                            "opts": {"MAXMODES": getattr(self, "_v9_modes", 12)}}

    def _finish_cases(self):
        for name, c in self._pending_cases.items():
            typ = (c["type"] or "").lower()
            o = c["opts"]
            if typ.startswith("linear static"):
                self.model.load_cases[name] = LinearStaticCase(name=name, loads=c["loads"])
            elif typ.startswith("modal"):
                self.model.load_cases[name] = ModalCase(name=name, max_modes=int(o.get("MAXMODES", 12)),
                                                        min_modes=int(o.get("MINMODES", 1)),
                                                        method="Ritz" if "ritz" in typ else "Eigen")
            elif typ.startswith("response spectrum"):
                modal = str(o.get("MODALCASE", "")) or next((n for n, cc in self._pending_cases.items()
                                                             if (cc["type"] or "").lower().startswith("modal")), "")
                self.model.load_cases[name] = ResponseSpectrumCase(
                    name=name, modal_case=modal,
                    loads=[SpectrumLoad(dirn, fn, sf) for dirn, fn, sf in c["rs_loads"]],
                    modal_combination=str(o.get("MODECOMBO", o.get("MODALCOMBO", "CQC"))).upper(),
                    directional_combination=str(o.get("DIRCOMBO", "SRSS")).upper(),
                    damping=float(o.get("CONSTDAMP", o.get("DAMPING", o.get("MODALDAMP", 0.05))) or 0.0),
                    angle=float(o.get("ANGLE", 0.0) or 0.0),
                    eccentricity_ratio=float(o.get("ECCEN", 0.0) or 0.0),
                )
            else:
                self.skipped[f"LOADCASE type {c['type']}"] += 1
        for name, fn in self._rs_functions.items():
            if fn["periods"]:
                self.model.functions[name] = ResponseSpectrumFunction(name, fn["periods"], fn["values"], fn["damping"])

    def _combo(self, toks):
        name = toks[1][0]
        d = _kv(toks, 2)
        c = self._pending_combos.setdefault(name, [])
        if "TYPE" in d:
            c.append(("TYPE", str(d["TYPE"])))
        for key in ("LOADCASE", "LOAD", "SPEC"):  # LOAD / SPEC: ETABS v9
            if key in d:
                c.append(("CASE", str(d[key]), float(d.get("SF", 1.0))))
        if "COMBO" in d:
            c.append(("COMBO", str(d["COMBO"]), float(d.get("SF", 1.0))))

    def _finish_combos(self):
        for name, items in self._pending_combos.items():
            combo = LoadCombination(name=name)
            for it in items:
                if it[0] == "TYPE":
                    combo.type = it[1].replace(" ", "")
                elif it[0] == "CASE":
                    if it[1] in self.model.load_cases:
                        combo.items.append(ComboItem(it[1], it[2], False))
                    else:
                        self.skipped[f"COMBO case {it[1]}"] += 1
                else:
                    combo.items.append(ComboItem(it[1], it[2], True))
            self.model.load_combinations[name] = combo

    # -------------------------------------------------------------- report
    def report(self) -> str:
        lines = [f"Imported {self.model!r}"]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {w}" for w in self.warnings]
        if self.skipped:
            lines.append("Skipped (not supported yet):")
            lines += [f"  - {k}: {v}" for k, v in sorted(self.skipped.items())]
        return "\n".join(lines)


# ----------------------------------------------------------------------------
def _mods_from(d: dict) -> Optional[FrameModifiers]:
    keys = {"AMOD": "A", "A2MOD": "As2", "A3MOD": "As3", "JMOD": "J", "I2MOD": "I22", "I3MOD": "I33",
            "MMOD": "mass", "WMOD": "weight"}
    found = {attr: float(d[k]) for k, attr in keys.items() if k in d and not isinstance(d[k], bool)}
    if not found:
        return None
    m = FrameModifiers()
    for attr, v in found.items():
        setattr(m, attr, v)
    return m


def _shell_mods_from(d: dict) -> Optional[ShellModifiers]:
    keys = {"F11MOD": "f11", "F22MOD": "f22", "F12MOD": "f12", "M11MOD": "m11", "M22MOD": "m22", "M12MOD": "m12",
            "V13MOD": "v13", "V23MOD": "v23", "MMOD": "mass", "WMOD": "weight"}
    found = {attr: float(d[k]) for k, attr in keys.items() if k in d and not isinstance(d[k], bool)}
    if not found:
        return None
    m = ShellModifiers()
    for attr, v in found.items():
        setattr(m, attr, v)
    return m


def _releases(text: str):
    """ETABS release string e.g. "M3J M2I TI" -> (releases_i, releases_j)."""
    ri = [False] * 6
    rj = [False] * 6
    index = {"P": 0, "V2": 1, "V3": 2, "T": 3, "M2": 4, "M3": 5}
    text = text.upper()
    if "PINNED" in text:
        text = text.replace("PINNED", "M2I M3I M2J M3J")
    for tok in text.split():
        end = tok[-1]
        comp = tok[:-1]
        if comp not in index or end not in ("I", "J"):
            continue
        (ri if end == "I" else rj)[index[comp]] = True
    return tuple(ri), tuple(rj)


def _direction(text: str) -> str:
    t = text.upper()
    mapping = {"GRAV": "Gravity", "GRAVPROJ": "GravityProjected", "X": "X", "Y": "Y", "Z": "Z", "XPROJ": "XProjected",
               "YPROJ": "YProjected", "ZPROJ": "ZProjected", "1": "1", "2": "2", "3": "3", "LOCAL1": "1",
               "LOCAL2": "2", "LOCAL3": "3", "-Z": "Gravity"}
    return mapping.get(t, "Gravity")


def read_e2k(path: str) -> Model:
    """Convenience: import an ETABS text file and return the model."""
    return E2KImporter.read(path).model
