"""Build a :class:`~structstudio.model.Model` from CSI database tables
(SAP2000, ETABS, SAFE – current and older versions).

The tables come from :mod:`structstudio.io.csi_tables` (text, Excel or
Access).  Three families of table names are understood:

* **SAP2000** (v8 and later; also CSiBridge): ``JOINT COORDINATES``,
  ``CONNECTIVITY - FRAME`` / ``- AREA``, ``FRAME SECTION ASSIGNMENTS``, ...
  Versions before v14 call load patterns "load cases" and load cases
  "analysis cases"; both are recognised.
* **ETABS 2013+ / SAFE 20+** (object based, story based): ``Point Object
  Connectivity``, ``Beam Object Connectivity``, ``Frame Assignments - Section
  Properties``, ``Story Definitions``, ...
* **SAFE v12 – 2016** (``.f2k``): ``OBJ GEOM - POINT COORDINATES``, ``OBJ GEOM -
  LINES 01 - GENERAL``, ``SLAB PROPERTIES``, ``BEAM / COLUMN PROPERTIES``, ...

Fields are looked up through alias lists, so a field that was renamed between
versions is still found.  Whatever is not understood is listed in
:attr:`CsiImporter.skipped`; approximations are listed in
:attr:`CsiImporter.warnings` – nothing is dropped silently.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..model import (Area, Diaphragm, Frame, FrameModifiers, FrameSection, LoadPattern, MassSource, Material, Model,
                     Point, ShellModifiers, ShellSection)
from ..model.loads import (ComboItem, LinearStaticCase, LoadCombination, ModalCase, PDeltaCase, ResponseSpectrumCase,
                           ResponseSpectrumFunction, SpectrumLoad)
from ..model.objects import AreaLoad, FrameLoad, GridLine, GridSystem, GroundDisplacement, Group, PointLoad, Story
from ..units import UnitSystem
from .csi_tables import TableSet, get, getf, gets, norm, read_tables, to_bool, to_float
from .section_db import DEFAULT_SEARCH_PATHS, SectionLibrary

DOFS = ("UX", "UY", "UZ", "RX", "RY", "RZ")
LOCAL6 = ("U1", "U2", "U3", "R1", "R2", "R3")


STIFF_AREA_FACTOR = 100.0  # SAFE "Stiff" slab property, see _shell_sections


def _area_name(row) -> Optional[str]:
    """The area object's name in a table row.  SAP2000 tables name it "Area";
    ETABS / SAFE object tables name it "UniqueName" and use "Area" for the
    measured area (with "Perimeter") - e.g. SAFE 22 "Floor Object Connectivity":
    ``UniqueName=2 ... Perimeter=240 Area=3600``."""
    if get(row, "Perimeter") is not None or (get(row, "UniqueName") is not None and isinstance(get(row, "Area"), float)):
        return gets(row, "UniqueName", "Name", "Shell", "Area")
    return gets(row, "Area", "UniqueName", "Name", "Shell")


def parse_units(text: str) -> Optional[UnitSystem]:
    """CSI unit labels: "KN, m, C", "Kip-in-F", "kN_m_C", "N mm", "Tonf, m, C"."""
    toks = [t for t in re.split(r"[\s,_\-/]+", str(text or "").strip()) if t]
    if len(toks) < 2:
        return None
    temp = next((t for t in toks[2:] if t.upper().lstrip("°") in ("C", "F")), "C")  # "kip_in_F"
    for f, l in ((toks[0], toks[1]), (toks[1], toks[0])):
        try:
            return UnitSystem.parse(f"{f} {l} {temp}")
        except ValueError:
            continue
    return None


def _direction(text, default: str = "Gravity") -> str:
    t = norm(text or "")
    mapping = {
        "gravity": "Gravity", "grav": "Gravity", "gravityprojected": "GravityProjected", "gravproj": "GravityProjected",
        "x": "X", "y": "Y", "z": "Z", "globalx": "X", "globaly": "Y", "globalz": "Z",
        "xprojected": "XProjected", "yprojected": "YProjected", "zprojected": "ZProjected",
        "xproj": "XProjected", "yproj": "YProjected", "zproj": "ZProjected", "gravityproj": "GravityProjected",
        "1": "1", "2": "2", "3": "3", "local1": "1", "local2": "2", "local3": "3",
        "u1": "1", "u2": "2", "u3": "3",
    }
    return mapping.get(t, default)


def _combo_type(text) -> str:
    t = norm(text or "linear add")
    if "envelope" in t or t == "env":
        return "Envelope"
    if "abs" in t:
        return "Absolute"
    if "srss" in t:
        return "SRSS"
    if "range" in t:
        return "Range"
    return "LinearAdd"


def _model_type(text, default: str = "ShellThin") -> str:
    t = norm(text or "")
    if not t:
        return default
    if "membrane" in t:
        return "Membrane"
    if "plate" in t:
        return "Plate"
    if "thick" in t:
        return "ShellThick"
    if "layer" in t:
        return "ShellThin"
    return "ShellThin" if "shell" in t or "thin" in t else default


@dataclass
class CsiImporter:
    tables: TableSet
    model: Model = field(default_factory=lambda: Model("imported"))
    program: str = ""  # SAP2000 | ETABS | SAFE | CSiBridge
    version: str = ""
    family: str = ""  # sap | etabs | safe_old
    skipped: Counter = field(default_factory=Counter)
    warnings: List[str] = field(default_factory=list)
    section_library: SectionLibrary = field(default_factory=lambda: SectionLibrary(DEFAULT_SEARCH_PATHS))
    source_dir: Optional[str] = None
    _used: set = field(default_factory=set)
    _null_lines: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # ------------------------------------------------------------------ api
    @classmethod
    def read(cls, path: str, **kw) -> "CsiImporter":
        ts = read_tables(path, kw.pop("field_keys", None))
        imp = cls(tables=ts, **kw)
        imp.source_dir = os.path.dirname(os.path.abspath(path))
        imp.section_library.search_paths.insert(0, imp.source_dir)
        imp.model.info.name = os.path.splitext(os.path.basename(path))[0]
        imp.build()
        return imp

    @classmethod
    def from_tables(cls, ts: TableSet, **kw) -> "CsiImporter":
        imp = cls(tables=ts, **kw)
        imp.build()
        return imp

    # ------------------------------------------------------------- helpers
    def _rows(self, *names: str):
        t = self.tables.get(*names)
        if t is None:
            return []
        self._used.add(t.key)
        return t.rows

    def _tables(self, *prefixes: str, contains: Sequence[str] = ()):
        ts = self.tables.matching(*prefixes, contains=contains)
        for t in ts:
            self._used.add(t.key)
        return ts

    def _warn(self, text: str):
        if text not in self.warnings:
            self.warnings.append(text)

    # --------------------------------------------------------------- build
    def build(self) -> Model:
        self.model.use_csi_frame_defaults()  # frames keep what the CSI tables say
        self._detect()
        self._units()
        self._active_dofs()
        self._mesh_options()
        self._materials()
        self._check_number_corruption()
        self._frame_sections()
        self._shell_sections()
        self._stories()
        self._grids()
        self._points()
        self._frames()
        self._areas()
        self._point_assigns()
        self._frame_assigns()
        self._area_assigns()
        self._diaphragms_and_constraints()
        self._safe_supports()
        self._groups()
        self._load_patterns()
        self._functions()
        self._loads()
        self._tendons()
        self._cases()
        self._combos()
        self._mass_source()
        self._drop_auto_points()
        self._levels()
        self._report_unused()
        self.model.convert_area_loads()  # CSI area loads -> area-load panels
        return self.model

    # -------------------------------------------------------------- program
    def _detect(self):
        pc = self.tables.program_control()
        name = gets(pc, "ProgramName", "Program", default="") or ""
        self.version = gets(pc, "Version", "ProgramVersion", default="") or ""
        t = self.tables
        if "point object connectivity" in t or "story definitions" in t:
            self.family = "etabs"
        elif t.get("joint coordinates") is not None or t.get("connectivity - frame") is not None:
            self.family = "sap"
        elif t.matching("obj geom", "object geometry"):
            self.family = "safe_old"
        else:
            self.family = "sap" if "sap" in name.lower() or "bridge" in name.lower() else (
                "etabs" if name.upper().startswith("ETABS") else "safe_old" if name.upper().startswith("SAFE") else "sap")
        if name:
            up = name.upper()
            self.program = ("SAP2000" if "SAP" in up else "ETABS" if "ETABS" in up else "SAFE" if "SAFE" in up else
                            "CSiBridge" if "BRIDGE" in up else name)
        else:
            self.program = {"sap": "SAP2000", "etabs": "ETABS", "safe_old": "SAFE"}[self.family]
        self._used.add(norm("program control"))
        pc = self.tables.program_control()
        if str(gets(pc, "NumbersSuspect", default="No")).lower().startswith("y"):
            self._warn("The converter could not read reliable numbers from the CSI program (whole numbers may have "
                       "lost trailing zeros); point coordinates and materials were repaired, other values may be wrong")
        o = self.model.options
        if self.program == "SAFE":
            # SAFE analyses floors as plates: UZ, RX, RY (in-plane behaviour is not modelled)
            o.active_dofs = (False, False, True, True, True, False)
        if self.family == "etabs":
            o.grid_mode = "story"
            o.auto_base_restraints = self.program != "SAFE"
        else:
            o.grid_mode = "xyz"
            o.auto_base_restraints = False

    def _units(self):
        pc = self.tables.program_control()
        u = parse_units(gets(pc, "CurrUnits", "Units", "PresentUnits", "DatabaseUnits", default=""))
        if u is None:
            # Excel exports carry a units row: length unit of a coordinate, force of a load
            for t in self.tables.tables.values():
                lu = t.units.get("x") or t.units.get("globalx") or t.units.get("xorr") or t.units.get("ordinate")
                if lu:
                    fu = next((v.split("/")[0] for tt in self.tables.tables.values() for k, v in tt.units.items()
                               if k in ("fx", "f1", "force", "unifload", "load", "fovera", "forcea")), "kN")
                    u = parse_units(f"{fu} {lu}")
                    if u is not None:
                        break
        if u is None:
            u = UnitSystem.parse("kN m")
            self._warn("No unit information found; assuming kN-m")
        self.model.units = u
        if self.family == "etabs" and self.program == "ETABS":
            # ETABS uses g = 386.4 in/s^2 (32.2 ft/s^2); SAP2000 / SAFE use 9.80665 m/s^2
            self.model.options.gravity = 386.4 if u.length == "in" else (32.2 if u.length == "ft" else None)
        mt = getf(pc, "MergeTol")
        if mt:
            self.model.options.merge_tolerance = mt

    def _mesh_options(self):
        o = self.model.options
        for r in self._rows("Analysis Options - Automatic Mesh Settings for Floors"):
            v = getf(r, "MaxMeshSize", "MaxSize")
            if v:
                o.floor_mesh_max_size = v
        for r in self._rows("Analysis Options - Automatic Rectangular Mesh Options for Walls"):
            v = getf(r, "MaxMeshSize", "MaxSize")
            if v:
                o.wall_mesh_max_size = v

    def _active_dofs(self):
        rows = self._rows("ACTIVE DEGREES OF FREEDOM", "Analysis Options - Active Degrees of Freedom",
                          "Active Degrees of Freedom")
        if rows:
            r = rows[0]
            self.model.options.active_dofs = tuple(to_bool(get(r, d), True) for d in DOFS)

    # -------------------------------------------------------------- materials
    def _check_number_corruption(self):
        """SAFE 22 (database tables API) drops the trailing zeros of whole numbers:
        E = 3000 ksi arrives as 3, fc = 40 MPa as 4.  A modulus less than 50 x the
        strength is impossible for concrete and steel, so it reveals such a file."""
        bad = []
        for t in self._tables("material properties - concrete data", "material properties 03b - concrete data",
                              "material properties - steel data", "material properties 03a - steel data",
                              "material properties - rebar data", "material properties 03e - rebar data"):
            for row in t.rows:
                mat = self.model.materials.get(gets(row, "Material", "Name") or "")
                f = getf(row, "Fc", "Fy", default=None)
                if mat is None or not f or mat.E <= 0:
                    continue
                E = mat.E * self.model.units.factor("stress")
                fv = f * self.model.units.factor("stress")
                if E < 50 * fv:
                    bad.append(mat.name)
        if bad:
            self._warn("Implausible material data (modulus < 50 x strength) for " + ", ".join(sorted(set(bad))) +
                       ": the file looks corrupted by the CSI table export (lost trailing zeros); re-convert it")
        self._check_spring_values()

    def _check_spring_values(self):
        """SAFE 22 table exports lose the zeros of large whole numbers in both table
        routes: a 1E+10 spring stiffness arrives as 1 (measured on the SAFE verification
        models).  Unless the converter read the springs through the object API
        (SpringsFrom=ObjectAPI), stiffnesses that read exactly 1 are flagged."""
        if self.program != "SAFE":
            return
        pc = self.tables.program_control()
        if norm(gets(pc, "SpringsFrom", default="") or "") == "objectapi" and not to_bool(get(pc, "SpringsSuspect")):
            return
        ones = []
        for t in self._tables("spring property definitions"):
            for row in t.rows:
                for k in ("VertStiff", "RotStiff", "SubModulus", "StiffUX", "StiffUY", "StiffUZ", "StiffRX", "StiffRY",
                          "StiffRZ"):
                    if getf(row, k) == 1.0:
                        ones.append(f"{gets(row, 'Name')}.{k}")
        if ones:
            self._warn("Spring stiffness = 1 in " + ", ".join(sorted(set(ones))[:8]) + (" ..." if len(ones) > 8 else "") +
                       ": SAFE 22 table exports drop the zeros of large whole numbers (1E+10 arrives as 1). "
                       "Check the values in SAFE (Define > Soil / Spring Properties) or re-convert the file "
                       "with the current scripts/Export-SafeSamples.cmd, which reads them from the SAFE API")

    def _materials(self):
        merged: "OrderedDict[str, dict]" = OrderedDict()
        for t in self._tables("material properties", "mat prop", "materials"):
            for row in t.rows:
                name = gets(row, "Material", "Name", "MatName", "Mat")
                if name:
                    merged.setdefault(name, {}).update({k: v for k, v in row.items() if v not in (None, "")})
        for name, d in merged.items():
            mat = Material(name=name)
            typ = gets(d, "Type", "MatType", "DesignType", "DesType", default="")
            if typ:
                tl = typ.lower()
                mat.type = ("Concrete" if "conc" in tl else "Steel" if "steel" in tl or tl in ("s",) else
                            "Rebar" if "rebar" in tl else "Tendon" if "tendon" in tl else
                            "Aluminum" if "alum" in tl else "Masonry" if "mason" in tl else
                            "ColdFormed" if "cold" in tl else "Other")
            mat.E = getf(d, "E1", "E", "Modulus", "EModulus", default=0.0) or 0.0
            nu = getf(d, "U12", "U", "Nu", "Poisson", "PoissonRatio")
            if nu is not None:
                mat.nu = nu
            g = getf(d, "G12", "G")
            if g:
                mat.G = g
            mat.weight_density = getf(d, "UnitWeight", "W", "WeightPerVolume", "WeightPerUnitVolume", default=0.0) or 0.0
            md = getf(d, "UnitMass", "M", "MassPerVolume", "MassPerUnitVolume")
            if md is not None:
                mat.mass_density = md
            a = getf(d, "A1", "A", "Alpha", "CoeffThermExp")
            if a is not None:
                mat.alpha = a
            mat.fc = getf(d, "Fc", "Fpc", "FcPrime", "Fck")
            mat.fy = getf(d, "Fy", "Fye")
            mat.fu = getf(d, "Fu", "Fue")
            damp = getf(d, "ModalRatio", "Damping")
            if damp:
                mat.damping = damp
            if not mat.E:
                self._warn(f"Material {name!r}: no modulus of elasticity in the tables")
            self.model.materials[name] = mat

    def _material_or_default(self, name: Optional[str]) -> str:
        if name and name in self.model.materials:
            return name
        if name and name.lower() not in ("default", "none", ""):
            self._warn(f"Material {name!r} is not defined; using the first material")
        if not self.model.materials:
            self.model.materials["Default"] = Material(name="Default", E=0.0)
            self._warn("No materials defined")
        return next(iter(self.model.materials))

    # ---------------------------------------------------------- frame sections
    def _frame_sections(self):
        merged: "OrderedDict[str, dict]" = OrderedDict()
        shape_of: Dict[str, str] = {}
        for t in self._tables("frame section properties", "frame section property definitions", "frame sec def",
                              "frame prop", "beam properties", "column properties"):
            tname = t.name.lower()
            for row in t.rows:
                name = gets(row, "SectionName", "Name", "Section", "Beam", "Column", "Prop")
                if not name:
                    continue
                merged.setdefault(name, {}).update({k: v for k, v in row.items() if v not in (None, "")})
                # shape from the definition table name (ETABS: "... - Concrete Rectangular")
                if " - " in tname and "summary" not in tname and "general" not in tname:
                    shape_of.setdefault(name, t.name.split(" - ")[-1])
                elif "column properties" in tname:
                    shape_of.setdefault(name, "column")
                elif "beam properties" in tname:
                    shape_of.setdefault(name, "beam")
        for name, d in merged.items():
            sec = self._frame_section_from(name, d, shape_of.get(name, ""))
            if sec is not None:
                self.model.frame_sections[name] = sec
        # nonprismatic segments (one row per segment)
        segs: "OrderedDict[str, list]" = OrderedDict()
        for t in self._tables("frame section properties 05 - nonprismatic", "frame section property definitions - nonprismatic",
                              "frame sec def - nonprismatic"):
            for row in t.rows:
                name = gets(row, "SectionName", "Name")
                if not name:
                    continue
                segs.setdefault(name, []).append(dict(
                    start=gets(row, "StartSect", "StartSection"), end=gets(row, "EndSect", "EndSection"),
                    length=getf(row, "AbsLength", "Length", "VarLength", "RelLength", default=1.0) or 0.0,
                    length_type="Absolute" if norm(gets(row, "LengthType", default="Variable") or "").startswith("abs")
                    else "Variable",
                    ei33=gets(row, "EI33Var", "EI33", default="Linear") or "Linear",
                    ei22=gets(row, "EI22Var", "EI22", default="Linear") or "Linear"))
        for name, rows in segs.items():
            first = self.model.frame_sections.get(rows[0]["start"] or "")
            if first is None or any(r["start"] not in self.model.frame_sections or
                                    r["end"] not in self.model.frame_sections for r in rows):
                self._warn(f"Nonprismatic section {name!r}: undefined segment section")
                continue
            self.model.frame_sections[name] = FrameSection(**{**first.__dict__, "name": name, "shape": "Nonprismatic",
                                                             "nonprismatic": rows})

    def _frame_section_from(self, name: str, d: dict, table_shape: str) -> Optional[FrameSection]:
        material = self._material_or_default(gets(d, "Material", "MatProp", "Mat", "MaterialName"))
        shape = gets(d, "Shape", "SectionType", "Type", default="") or table_shape or "General"
        sl = shape.lower()
        if "nonprismatic" in sl:
            start = gets(d, "StartSect", "StartSection")
            src = self.model.frame_sections.get(start or "")
            if src is None:
                return None
            sec = FrameSection(**{**src.__dict__, "name": name})
            return sec
        t3 = getf(d, "t3", "Depth", "SecDim3", "D", "Diameter", "Dia")
        t2 = getf(d, "t2", "Width", "WidthTop", "SecDim2", "B", "BF")
        tf = getf(d, "tf", "FlangeThick", "TFlange")
        tw = getf(d, "tw", "WebThick", "TWeb")
        A = getf(d, "Area", "A", "XSecArea")
        I33 = getf(d, "I33", "I3", "Inertia33")
        I22 = getf(d, "I22", "I2", "Inertia22")
        J = getf(d, "TorsConst", "J", "Torsion")
        As2 = getf(d, "AS2", "ShearArea2", "As2")
        As3 = getf(d, "AS3", "ShearArea3", "As3")
        circular = any(k in sl for k in ("circ", "round", "rod", "pipe")) or (t3 and not t2 and "rect" not in sl)
        sec: Optional[FrameSection] = None
        explicit = A is not None and (I33 is not None or I22 is not None)
        if explicit and (A or I33 or I22 or "general" in sl or "general" in table_shape.lower()):
            # tabulated properties (SAP2000 "01 - General", ETABS summary / general tables);
            # zero entries are real (e.g. axially rigid / no-area "General" sections)
            sec = FrameSection(name=name, material=material, shape=shape, t3=t3, t2=t2, tf=tf, tw=tw,
                               A=A or 0.0, J=J or 0.0, I33=I33 or 0.0, I22=I22 or 0.0, As2=As2 or 0.0, As3=As3 or 0.0)
            sec.t2b = getf(d, "t2b")
            sec.tfb = getf(d, "tfb")
        else:
            label = gets(d, "SectInFile", "SectionInFile")
            if label or gets(d, "FileName", "File", "FromFile"):
                for lab in (label, name):
                    if not lab:
                        continue
                    hit = self.section_library.find(lab, gets(d, "FileName", "File"))
                    if hit is not None:
                        sec = hit.to_frame_section(name, material, self.model.units)
                        break
            if sec is None:
                if circular and t3 and ("pipe" in sl and tw):
                    ro, ri = t3 / 2, t3 / 2 - tw
                    Ap = math.pi * (ro ** 2 - ri ** 2)
                    Ip = math.pi * (ro ** 4 - ri ** 4) / 4
                    sec = FrameSection.general(name, material, Ap, Ip, Ip, 2 * Ip, 0.5 * Ap, 0.5 * Ap)
                    sec.t3 = t3
                elif circular and t3:
                    sec = FrameSection.circular(name, material, t3)
                elif t3 and t2 and tf and tw and any(k in sl for k in ("i/wide", "wide flange", "isection", " i", "i-"))\
                        or (sl in ("i", "steel i") and t3 and t2 and tf and tw):
                    sec = FrameSection.i_shape(name, material, t3, t2, tf, tw)
                elif t3 and t2 and not tf:
                    sec = FrameSection.rectangular(name, material, t3, t2)
                elif t3 and t2 and tf and tw and ("tube" in sl or "box" in sl):
                    h, b = t3, t2
                    Ab = b * h - (b - 2 * tw) * (h - 2 * tf)
                    Ib33 = (b * h ** 3 - (b - 2 * tw) * (h - 2 * tf) ** 3) / 12
                    Ib22 = (h * b ** 3 - (h - 2 * tf) * (b - 2 * tw) ** 3) / 12
                    Jb = 2 * tw * tf * (b - tw) ** 2 * (h - tf) ** 2 / (b * tw + h * tf - tw ** 2 - tf ** 2)
                    sec = FrameSection.general(name, material, Ab, Ib33, Ib22, Jb, 2 * tw * h, 2 * tf * b)
                    sec.t3, sec.t2, sec.tf, sec.tw = t3, t2, tf, tw
                else:
                    sec = FrameSection(name=name, material=material, shape="General", t3=t3, t2=t2,
                                       A=A or 0.0, J=J or 0.0, I33=I33 or 0.0, I22=I22 or 0.0,
                                       As2=As2 or 0.0, As3=As3 or 0.0)
                    if not A:
                        self._warn(f"Frame section {name!r} ({shape}): no properties or dimensions; "
                                   "properties are zero")
            sec.shape = shape
        mods = {attr: getf(d, *keys) for attr, keys in (
            ("A", ("AMod", "AreaModifier")), ("As2", ("A2Mod", "As2Modifier")), ("As3", ("A3Mod", "As3Modifier")),
            ("J", ("JMod", "JModifier")), ("I22", ("I2Mod", "I22Modifier")), ("I33", ("I3Mod", "I33Modifier")),
            ("mass", ("MMod", "MassModifier")), ("weight", ("WMod", "WeightModifier")))}
        if any(v is not None for v in mods.values()):
            m = FrameModifiers()
            for k, v in mods.items():
                if v is not None:
                    setattr(m, k, v)
            sec.modifiers = m
        color = gets(d, "Color")
        if color and color.startswith("#"):
            sec.color = color
        dt = gets(d, "DesignType")
        if dt:
            sec.design_type = dt
        return sec

    # ---------------------------------------------------------- shell sections
    def _shell_sections(self):
        merged: "OrderedDict[str, dict]" = OrderedDict()
        kind_of: Dict[str, str] = {}
        for t in self._tables("area section properties", "shell section properties", "slab property definitions",
                              "wall property definitions", "deck property definitions", "slab properties",
                              "wall properties", "deck properties", "shell prop"):
            tl = t.name.lower()
            if "layer" in tl or "auto select" in tl:
                if "layer" in tl and t.rows:
                    self.skipped["layered shell definitions (thickness taken from the summary)"] += len(t.rows)
                continue
            for row in t.rows:
                name = gets(row, "Section", "Name", "Slab", "Wall", "Deck", "Prop", "SlabProp")
                if not name:
                    continue
                merged.setdefault(name, {}).update({k: v for k, v in row.items() if v not in (None, "")})
                kind_of.setdefault(name, "Wall" if "wall" in tl else "Deck" if "deck" in tl else "Slab")
        for name, d in merged.items():
            kind = kind_of.get(name, "Slab")
            ptype = gets(d, "PropType", "PropertyType", "AreaType", "SlabType", default="") or ""
            if kind == "Slab" and "wall" in ptype.lower():
                kind = "Wall"
            material = self._material_or_default(gets(d, "Material", "MatProp", "MaterialSlb", "ConcMat", "Mat"))
            thick = getf(d, "Thickness", "SlabThickness", "WallThickness", "Thick", "MembThick", "SlabDepth", "Depth",
                         "OverallDepth", default=0.0) or 0.0
            bend = getf(d, "BendThick", "BendingThickness")
            default_mt = "ShellThick" if self.program == "SAFE" and to_bool(get(d, "ThickPlate"), True) else "ShellThin"
            mt = _model_type(gets(d, "Type", "ModelType", "ModelingType", "ShellType", "ElemType", default=""), default_mt)
            if "plate" in (gets(d, "Type", "ModelType", default="") or "").lower() and "thick" in (gets(d, "Type", "ModelType", default="") or "").lower():
                self._warn(f"Shell section {name!r}: thick plate imported as plate (thin)")
            if kind == "Deck":
                self._warn(f"Deck property {name!r} imported as a slab of the concrete depth")
                kind = "Slab"
            if ptype and any(k in ptype.lower() for k in ("ribbed", "waffle")):
                self._warn(f"Slab property {name!r} ({ptype}) imported as a solid slab of the overall thickness")
            sec = ShellSection(name=name, material=material, type=kind if kind in ("Slab", "Wall") else "Slab",
                               model_type=mt, thickness=thick, bending_thickness=bend if bend and bend != thick else None)
            keys = tuple((a, (a.upper() + "Mod", a, a + "Modifier")) for a in
                         ("f11", "f22", "f12", "m11", "m22", "m12", "v13", "v23")) + (
                ("mass", ("MMod", "MassModifier")), ("weight", ("WMod", "WeightModifier")))
            vals = {a: getf(d, *k) for a, k in keys}
            if any(v is not None for v in vals.values()):
                sm = ShellModifiers()
                for a, v in vals.items():
                    if v is not None:
                        setattr(sm, a, v)
                sec.modifiers = sm
            if ptype.strip().lower() == "stiff":
                # SAFE "Stiff" property (column regions): much stiffer than the slab.  The SAFE
                # manuals give no factor; 100 x membrane and bending stiffness reproduces the
                # SAFE samples with stiff areas (S05d: 1 - 6 %), larger factors change nothing
                sm = sec.modifiers if sec.modifiers is not None else ShellModifiers()
                for a in ("f11", "f22", "f12", "m11", "m22", "m12"):
                    setattr(sm, a, getattr(sm, a) * STIFF_AREA_FACTOR)
                sec.modifiers = sm
                self._warn(f"Slab property {name!r} (Stiff): imported as {STIFF_AREA_FACTOR:g} x the stiffness of a "
                           f"{thick:g} slab")
            if not thick:
                self._warn(f"Shell section {name!r}: no thickness")
            self.model.shell_sections[name] = sec

    # ----------------------------------------------------------- stories
    def _stories(self):
        rows = self._rows("Story Definitions", "STORY DATA", "STORY DEFINITIONS")
        if not rows:
            return
        base_rows = self._rows("Tower and Base Story Definitions", "Base Story Definition")
        base_name = gets(base_rows[0], "BSName", "BaseStory", default="Base") if base_rows else "Base"
        base_elev = getf(base_rows[0], "BSElev", "BaseElev", default=0.0) if base_rows else 0.0
        towers = {gets(r, "Tower", default="T1") for r in rows}
        if len(towers) > 1:
            self._warn("Multiple towers: all towers are imported on one set of stories")
        first_tower = gets(rows[0], "Tower", default="T1")
        story_rows = [r for r in rows if gets(r, "Tower", default="T1") == first_tower]
        # listed top -> bottom
        ordered = list(reversed(story_rows))
        m = self.model
        m.stories.clear()
        m.stories[base_name] = Story(name=base_name, elevation=base_elev or 0.0, height=0.0)
        z = base_elev or 0.0
        for r in ordered:
            name = gets(r, "Story", "Name")
            h = getf(r, "Height", default=0.0) or 0.0
            elev = getf(r, "Elevation", "Elev")
            z = elev if elev is not None else z + h
            m.stories[name] = Story(name=name, elevation=z, height=h,
                                    is_master=to_bool(get(r, "IsMaster", "MasterStory")),
                                    similar_to=gets(r, "SimilarTo"))

    def _grids(self):
        m = self.model
        rows = self._rows("GRID LINES", "Grid Definitions - Grid Lines", "GRID LINES - CARTESIAN")
        systems = self._rows("Grid Definitions - General", "COORDINATE SYSTEMS")
        for r in systems:
            name = gets(r, "Name", "CoordSys")
            if name and name.upper() != "GLOBAL" or (name and not m.grid_systems):
                m.grid_systems.setdefault(name, GridSystem(name=name, origin=(getf(r, "Ux", "X", default=0.0),
                                                                              getf(r, "Uy", "Y", default=0.0)),
                                                           rotation=getf(r, "Rz", "AboutZ", default=0.0) or 0.0))
        for r in rows:
            system = gets(r, "CoordSys", "Name", "GridSys", default="G1")
            axis = (gets(r, "AxisDir", "LineType", "GridLineType", "Dir", default="X") or "X").strip().upper()[:1]
            if axis not in ("X", "Y", "Z"):
                if axis in ("R", "T"):
                    self.skipped["cylindrical grid lines"] += 1
                    continue
                axis = "X"
            gid = gets(r, "GridID", "ID", "Label", default="") or ""
            coord = getf(r, "XRYZCoord", "Ordinate", "Coord", "Coordinate", default=0.0) or 0.0
            if system not in m.grid_systems:
                m.grid_systems[system] = GridSystem(name=system)
            m.grid_lines.append(GridLine(system=system, name=gid, direction=axis, ordinate=coord,
                                         bubble_location=gets(r, "BubbleLoc", "BubbleLocation", default="End") or "End",
                                         visible=to_bool(get(r, "Visible"), True)))

    # -------------------------------------------------------------- points
    def _coord_systems(self) -> Dict[str, Tuple[Tuple[float, float, float], float, float, float]]:
        out = {}
        for r in self._rows("COORDINATE SYSTEMS"):
            name = gets(r, "Name")
            if name:
                out[name.upper()] = ((getf(r, "X", default=0.0), getf(r, "Y", default=0.0), getf(r, "Z", default=0.0)),
                                     getf(r, "AboutZ", default=0.0) or 0.0, getf(r, "AboutY", default=0.0) or 0.0,
                                     getf(r, "AboutX", default=0.0) or 0.0)
        return out

    def _points(self):
        m = self.model
        csys = None
        tables = (self._tables("joint coordinates", "point object connectivity", "obj geom - point coordinates",
                               "object geometry - point coordinates", "point coordinates", "joint object connectivity"))
        for t in tables:
            for row in t.rows:
                name = gets(row, "Joint", "Point", "UniqueName", "Name", "Label")
                if not name:
                    continue
                gx = getf(row, "GlobalX")
                if gx is not None:
                    x, y, z = gx, getf(row, "GlobalY", default=0.0), getf(row, "GlobalZ", default=0.0)
                else:
                    ctype = (gets(row, "CoordType", default="Cartesian") or "Cartesian").lower()
                    a = getf(row, "XorR", "X", "R", default=0.0)
                    b = getf(row, "Y", "T", "Theta", "A", default=0.0)
                    c = getf(row, "Z", "B", default=0.0)
                    if ctype.startswith("cyl"):
                        x, y, z = a * math.cos(math.radians(b)), a * math.sin(math.radians(b)), c
                    elif ctype.startswith("sph"):
                        th, ph = math.radians(b), math.radians(c)
                        x, y, z = a * math.sin(ph) * math.cos(th), a * math.sin(ph) * math.sin(th), a * math.cos(ph)
                    else:
                        x, y, z = a, b, c
                    cs = (gets(row, "CoordSys", default="GLOBAL") or "GLOBAL").upper()
                    if cs != "GLOBAL":
                        csys = csys if csys is not None else self._coord_systems()
                        if cs in csys:
                            (ox, oy, oz), rz, ry, rx = csys[cs]
                            if ry or rx:
                                self._warn(f"Coordinate system {cs}: only the rotation about Z is applied")
                            c_, s_ = math.cos(math.radians(rz)), math.sin(math.radians(rz))
                            x, y, z = ox + c_ * x - s_ * y, oy + s_ * x + c_ * y, oz + z
                        else:
                            self._warn(f"Joint coordinates in unknown coordinate system {cs} taken as global")
                story = gets(row, "Story")
                label = gets(row, "PointBay", "Label", "PointLabel")
                if self.family == "etabs" and story and story in m.stories and getf(row, "Z") is None and gx is None:
                    z = m.stories[story].elevation
                p = Point(name=name, x=x, y=y, z=z, label=label or None, story=story if story in m.stories else None,
                          is_auto=to_bool(get(row, "IsAuto", "IsAutoPoint")))
                m.points[name] = p

    # -------------------------------------------------------------- frames
    def _frames(self):
        m = self.model
        sections = self._frame_section_assignments()
        type_by_table = (("beam object", "Beam"), ("column object", "Column"), ("brace object", "Brace"),
                         ("null line", "Null"))
        tables = self._tables("connectivity - frame", "connectivity frame", "beam object connectivity",
                              "column object connectivity", "brace object connectivity", "null line object connectivity",
                              "line object connectivity", "obj geom - lines", "object geometry - lines",
                              "frame object connectivity")
        no_section = 0
        for t in tables:
            tl = t.name.lower()
            ttype = next((ty for k, ty in type_by_table if k in tl), None)
            for row in t.rows:
                name = gets(row, "Frame", "Line", "UniqueName", "Name")
                pi = gets(row, "JointI", "PointI", "UniquePtI", "PtI", "IJoint", "JtI")
                pj = gets(row, "JointJ", "PointJ", "UniquePtJ", "PtJ", "JJoint", "JtJ")
                if not (name and pi and pj):
                    continue
                if pi not in m.points or pj not in m.points:
                    self.skipped["frames with undefined end points"] += 1
                    continue
                if pi == pj:
                    self.skipped["zero-length frames"] += 1
                    continue
                if to_bool(get(row, "IsCurved")):
                    self._warn("Curved frames imported as straight members")
                ltype = (gets(row, "LineType", "Type", "DesignType", default="") or "").lower()
                ftype = ttype or ("Column" if "col" in ltype else "Brace" if "brace" in ltype else
                                  "Beam" if "beam" in ltype else None)
                if "tendon" in ltype:
                    self.skipped["tendons"] += 1
                    continue
                sec = sections.get(name)
                if ftype == "Null" or (sec is not None and sec.upper() == "NONE"):
                    # SAFE / ETABS null line: no stiffness, but it carries loads (to the slab
                    # along it), line springs and mesh lines
                    self._null_lines[name] = (pi, pj)
                    m.frames[name] = Frame(name=name, point_i=pi, point_j=pj, section="", type="Null",
                                           story=gets(row, "Story"))
                    continue
                if sec is None or sec not in m.frame_sections:
                    no_section += 1
                    continue
                a, b = m.points[pi], m.points[pj]
                if ftype is None:
                    dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
                    horiz = math.hypot(dx, dy)
                    ftype = "Column" if horiz < 1e-9 * max(1.0, abs(dz)) else (
                        "Beam" if abs(dz) < 1e-9 * max(1.0, horiz) else "Brace")
                fr = Frame(name=name, point_i=pi, point_j=pj, section=sec, type=ftype,
                           label=gets(row, "BeamBay", "ColumnBay", "BraceBay", "LineBay", "Label") or None,
                           story=gets(row, "Story"))
                if self.family == "etabs":
                    fr.auto_end_offsets = True
                else:
                    # SAP2000 / SAFE: frame objects are not divided where other objects cross them
                    # unless an automatic mesh is assigned (FRAME AUTO MESH ASSIGNMENTS)
                    fr.auto_mesh = self.family == "safe_old"
                    fr.mesh_at_intersections = self.family == "safe_old"
                m.frames[name] = fr
        if no_section:
            self._warn(f"{no_section} frame object(s) without a defined section were skipped")

    def _frame_section_assignments(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for t in self._tables("frame section assignments", "frame assignments - section properties",
                              "frame assigns - sect prop", "beam property assignments", "column property assignments",
                              "brace property assignments", "line property assignments", "frame property assignments"):
            for row in t.rows:
                name = gets(row, "Frame", "Line", "UniqueName", "Name")
                sec = gets(row, "AnalSect", "SectProp", "SectionProperty", "Section", "BeamProp", "ColProp",
                           "ColumnProp", "BraceProp", "Prop", "DesignSect")
                if name and sec:
                    out[name] = sec
                    mat = gets(row, "MatProp")
                    if mat and mat.lower() not in ("default", "none") and sec in self.model.frame_sections and \
                            self.model.frame_sections[sec].material != mat and mat in self.model.materials:
                        key = f"{sec} ({mat})"
                        if key not in self.model.frame_sections:
                            base = self.model.frame_sections[sec]
                            self.model.frame_sections[key] = FrameSection(**{**base.__dict__, "name": key, "material": mat})
                        out[name] = key
        return out

    # -------------------------------------------------------------- areas
    def _areas(self):
        m = self.model
        sections = self._area_section_assignments()
        type_by_table = (("floor object", "Floor"), ("wall object", "Wall"), ("null area", "Null"),
                         ("ramp object", "Ramp"), ("opening", "Opening"))
        tables = self._tables("connectivity - area", "connectivity area", "floor object connectivity",
                              "wall object connectivity", "null area object connectivity", "area object connectivity",
                              "ramp object connectivity", "obj geom - areas", "object geometry - areas")
        defs: "OrderedDict[str, Tuple[List[str], Optional[str], dict]]" = OrderedDict()
        for t in tables:
            tl = t.name.lower()
            ttype = next((ty for k, ty in type_by_table if k in tl), None)
            for row in t.rows:
                name = _area_name(row)
                if not name:
                    continue
                pts = []
                k = 1
                while True:
                    p = gets(row, f"Joint{k}", f"Point{k}", f"UniquePt{k}", f"Pt{k}")
                    if p is None:
                        if k > 4 and gets(row, f"Joint{k + 1}", f"Point{k + 1}", f"UniquePt{k + 1}") is None:
                            break
                        if k > 64:
                            break
                        k += 1
                        continue
                    pts.append(p)
                    k += 1
                if not pts:
                    p = gets(row, "Joint", "Point", "UniquePt")  # one point per record
                    if p:
                        pts = [p]
                if name in defs:  # continuation records (> 4 points)
                    defs[name][0].extend(pts)
                else:
                    defs[name] = (pts, ttype, row)
        for name, (pts, ttype, row) in defs.items():
            n = int(getf(row, "NumJoints", "NumPoints", "NumOfPts", default=len(pts)) or len(pts))
            pts = pts[:n] if n >= 3 else pts
            dedup = []
            for p in pts:
                if not dedup or dedup[-1] != p:
                    dedup.append(p)
            if len(dedup) > 1 and dedup[0] == dedup[-1]:
                dedup.pop()
            if len(dedup) < 3 or any(p not in m.points for p in dedup):
                self.skipped["areas with fewer than 3 or undefined points"] += 1
                continue
            sec = sections.get(name)
            if sec is not None and sec.upper() in ("NONE", "NULL", "OPENING"):
                sec = None
            elif sec is not None and sec not in m.shell_sections:
                self._warn(f"Area {name}: section {sec!r} not defined; imported as a null area")
                sec = None
            elif sec is None and ttype not in ("Null", "Opening"):
                if self.family == "sap":
                    self.skipped["areas without a section assignment"] += 1
                    continue
            zs = [m.points[p].z for p in dedup]
            if ttype in (None, "Floor") and sec is None and ttype != "Floor":
                atype = "Opening"
            elif ttype is None or ttype in ("Floor", "Ramp", "Wall"):
                vertical = max(zs) - min(zs) > m.options.merge_tolerance
                atype = ttype or ("Wall" if vertical and self._is_vertical(dedup) else "Floor")
            else:
                atype = ttype
            if sec is None and atype in ("Floor", "Wall", "Ramp"):
                atype = "Null"
            area = Area(name=name, points=dedup, section=sec, type=atype,
                        label=gets(row, "FloorBay", "WallBay", "AreaBay", "RampBay", "Label") or None,
                        story=gets(row, "Story"))
            if self.family == "sap":
                area.auto_mesh = False  # SAP2000: areas are elements as drawn unless meshed
            elif self.family == "etabs" and atype == "Wall":
                area.auto_mesh = False  # ETABS default: one element per wall panel (see e2k importer)
            m.areas[name] = area

    def _is_vertical(self, pts: List[str]) -> bool:
        m = self.model
        P = [m.points[p] for p in pts]
        ax, ay, az = P[1].x - P[0].x, P[1].y - P[0].y, P[1].z - P[0].z
        for q in P[2:]:
            bx, by, bz = q.x - P[0].x, q.y - P[0].y, q.z - P[0].z
            nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
            n = math.sqrt(nx * nx + ny * ny + nz * nz)
            if n > 1e-12:
                return abs(nz) / n < 0.5
        return False

    def _area_section_assignments(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for t in self._tables("area section assignments", "area assignments - section properties",
                              "area assigns - sect prop", "slab property assignments", "wall property assignments",
                              "shell section assignments", "area property assignments"):
            for row in t.rows:
                name = _area_name(row)
                sec = gets(row, "Section", "SectProp", "SectionProperty", "SlabProp", "WallProp", "Prop", "Property")
                if name and sec:
                    out[name] = sec
                    mat = gets(row, "MatProp")
                    if mat and mat.lower() not in ("default", "none") and sec in self.model.shell_sections and \
                            mat in self.model.materials and self.model.shell_sections[sec].material != mat:
                        key = f"{sec} ({mat})"
                        if key not in self.model.shell_sections:
                            base = self.model.shell_sections[sec]
                            self.model.shell_sections[key] = ShellSection(**{**base.__dict__, "name": key,
                                                                              "material": mat})
                        out[name] = key
        return out

    # ------------------------------------------------------- point assigns
    def _point_assigns(self):
        m = self.model
        for t in self._tables("joint restraint assignments", "joint assignments - restraints", "joint assigns - restraints",
                              "point restraint assignments", "point assignments - restraints",
                              "jt assigns - restraints"):
            for row in t.rows:
                p = m.points.get(gets(row, "Joint", "Point", "UniqueName", "Name") or "")
                if p is None:
                    continue
                p.restraints = tuple(to_bool(get(row, LOCAL6[k], DOFS[k])) for k in range(6))
        for t in self._tables("joint local axes assignments", "joint assignments - local axes",
                              "point assignments - local axes"):
            for row in t.rows:
                p = m.points.get(gets(row, "Joint", "Point", "UniqueName", "Name") or "")
                if p is None:
                    continue
                ang = tuple(getf(row, f"Angle{c}", c, default=0.0) or 0.0 for c in "ABC")
                if any(ang):
                    p.local_axes = ang
                if gets(row, "Advanced", default="No") not in (None, "No", "no"):
                    self._warn(f"Joint {p.name}: advanced local axes not supported")
        # springs: SAP2000 values per joint, ETABS / SAFE named spring properties
        props: Dict[str, Tuple[float, ...]] = {}
        for t in self._tables("spring property definitions - point", "spring props - point", "point spring properties",
                              "point spring property definitions", "spring properties - point"):
            for row in t.rows:
                n = gets(row, "Name", "Spring", "SpringProp")
                if n:
                    props[n] = tuple(getf(row, f"Stiff{d}", d, f"Stiff{l}", l, default=0.0) or 0.0
                                     for d, l in zip(DOFS, LOCAL6))
        for t in self._tables("joint spring assignments", "joint assignments - springs", "joint assigns - springs",
                              "point spring assignments", "point assignments - springs"):
            for row in t.rows:
                p = m.points.get(gets(row, "Joint", "Point", "UniqueName", "Name") or "")
                if p is None:
                    continue
                prop = gets(row, "SpringProp", "Spring", "Prop")
                if prop:
                    if prop not in props:
                        self.skipped[f"point spring property {prop}"] += 1
                        continue
                    k = props[prop]
                else:
                    k = tuple(getf(row, LOCAL6[i], DOFS[i], default=0.0) or 0.0 for i in range(6))
                    cs = (gets(row, "CoordSys", default="Global") or "Global").lower()
                    if cs not in ("global", "local") and any(k):
                        self._warn(f"Joint springs in coordinate system {cs} taken as global")
                p.springs = tuple(a + b for a, b in zip(p.springs, k[:3]))
                p.rotational_springs = tuple(a + b for a, b in zip(p.rotational_springs, k[3:]))
        for t in self._tables("joint added mass assignments", "joint assignments - additional mass",
                              "jt assigns - additional mass", "point mass assignments", "joint masses"):
            for row in t.rows:
                p = m.points.get(gets(row, "Joint", "Point", "UniqueName", "Name") or "")
                if p is None:
                    continue
                mxy = getf(row, "MassXY")
                if mxy is not None:
                    tr = (mxy, mxy, getf(row, "MassZ", default=0.0) or 0.0)
                else:
                    tr = tuple(getf(row, f"Mass{i}", f"U{i}", default=0.0) or 0.0 for i in (1, 2, 3))
                rot = tuple(getf(row, f"MMI{i}", f"MMI{a}", f"R{i}", default=0.0) or 0.0 for i, a in zip((1, 2, 3), "XYZ"))
                p.mass = tuple(a + b for a, b in zip(p.mass, tr))
                p.rotational_mass = tuple(a + b for a, b in zip(p.rotational_mass, rot))

    # ------------------------------------------------------- frame assigns
    def _frame_rows(self, *prefixes):
        for t in self._tables(*prefixes):
            for row in t.rows:
                f = self.model.frames.get(gets(row, "Frame", "Line", "UniqueName", "Name") or "")
                if f is not None:
                    yield f, row, t

    def _frame_assigns(self):
        comps = ("P", "V2", "V3", "T", "M2", "M3")
        for f, row, t in self._frame_rows("frame release assignments", "frame assignments - releases",
                                          "frame assigns - releases", "line release assignments"):
            if "partial" in t.name.lower() and "releases and" not in t.name.lower():  # SAP2000 "2 - PARTIAL FIXITY"
                si = tuple(getf(row, f"{c}I", default=0.0) or 0.0 for c in comps)
                sj = tuple(getf(row, f"{c}J", default=0.0) or 0.0 for c in comps)
                f.fixity_i_trans, f.fixity_i_rot = si[:3], si[3:]
                f.fixity_j_trans, f.fixity_j_rot = sj[:3], sj[3:]
                continue
            f.releases_i = tuple(to_bool(get(row, f"{c}I")) for c in comps)
            f.releases_j = tuple(to_bool(get(row, f"{c}J")) for c in comps)
            si = tuple(getf(row, f"{c}ISpring", default=0.0) or 0.0 for c in comps)
            sj = tuple(getf(row, f"{c}JSpring", default=0.0) or 0.0 for c in comps)
            if any(si) or any(sj):
                f.fixity_i_trans, f.fixity_i_rot = si[:3], si[3:]
                f.fixity_j_trans, f.fixity_j_rot = sj[:3], sj[3:]
        for f, row, t in self._frame_rows("frame local axes assignments", "frame assignments - local axes",
                                          "frame assigns - local axes", "line local axes"):
            f.angle = getf(row, "Angle", default=0.0) or 0.0
            if to_bool(get(row, "AdvanceAxes")):
                self._warn("Advanced frame local axes are not supported; only the angle is used")
        for f, row, t in self._frame_rows("frame offset along length assignments", "frame assignments - end length offsets",
                                          "frame assigns - end len offsets", "frame end offset"):
            opt = (gets(row, "LengthType", "OffsetOpt", "Type", default="") or "").lower()
            f.rigid_factor = getf(row, "RigidFact", "RigidFactor", default=0.0) or 0.0
            if opt.startswith("auto"):
                f.auto_end_offsets = True
            else:
                f.auto_end_offsets = False
                f.end_offset_i = getf(row, "OffsetI", "LengthI", default=0.0) or 0.0
                f.end_offset_j = getf(row, "OffsetJ", "LengthJ", default=0.0) or 0.0
        for f, row, t in self._frame_rows("frame insertion point assignments", "frame assignments - insertion point",
                                          "frame assigns - insertion point"):
            cp = int(getf(row, "CardinalPt", "CardinalPoint", default=10) or 10)
            transform = to_bool(get(row, "StiffTransform", "TransformStiff"), True)
            if get(row, "NoTransform") is not None:
                transform = not to_bool(get(row, "NoTransform"))
            if self.family == "etabs" and get(row, "StiffTransform", "TransformStiff", "NoTransform") is None:
                transform = False
            f.cardinal_point = cp if transform else 10
            f.display_cardinal = cp
            xi = tuple(getf(row, a, b, default=0.0) or 0.0 for a, b in (("XI", "JtOffsetXI"), ("YI", "JtOffsetYI"), ("ZI", "JtOffsetZI")))
            xj = tuple(getf(row, a, b, default=0.0) or 0.0 for a, b in (("XJ", "JtOffsetXJ"), ("YJ", "JtOffsetYJ"), ("ZJ", "JtOffsetZJ")))
            if any(xi) or any(xj):
                cs = (gets(row, "CoordSys", "OffsetCSys", default="Global") or "Global").lower()
                f.joint_offset_local = cs.startswith("local")
                if cs not in ("global", "glob") and not f.joint_offset_local:
                    self._warn(f"Frame joint offsets in coordinate system {cs} taken as global")
                f.joint_offset_i, f.joint_offset_j = xi, xj
        for f, row, t in self._frame_rows("frame property modifiers", "frame assignments - property modifiers",
                                          "frame assigns - prop modifiers"):
            vals = {"A": getf(row, "AMod", "Area"), "As2": getf(row, "A2Mod", "AS2"), "As3": getf(row, "A3Mod", "AS3"),
                    "J": getf(row, "JMod", "Torsion"), "I22": getf(row, "I2Mod", "I22"), "I33": getf(row, "I3Mod", "I33"),
                    "mass": getf(row, "MMod", "Mass"), "weight": getf(row, "WMod", "Weight")}
            mods = FrameModifiers()
            for k, v in vals.items():
                if v is not None:
                    setattr(mods, k, v)
            f.modifiers = mods
        for f, row, t in self._frame_rows("frame auto mesh assignments", "frame automesh",
                                          "frame assignments - frame auto mesh"):
            f.auto_mesh = to_bool(get(row, "AutoMesh"), f.auto_mesh)
            f.mesh_at_intersections = to_bool(get(row, "AtFrames", "AtIntersections", "AtIntersect"), False) or \
                to_bool(get(row, "AtJoints", "AtIntJoints"), False)
        for f, row, t in self._frame_rows("frame assignments - frame floor meshing"):
            opt = norm(gets(row, "MeshOption", default="") or "")
            f.mesh_at_floor = not opt.startswith(("donot", "no", "exclude"))
            ml = getf(row, "MaxLength", "MaxSegLength")
            if ml:
                f.max_station_spacing = ml
            if (getf(row, "NumSegments", default=0) or 0) > 1 or ml:
                self._warn("Frame auto-mesh by segment count / length is used for output stations only")
        for f, row, t in self._frame_rows("frame output station assignments", "frame assignments - output stations",
                                          "frame assigns - output stations"):
            ms = getf(row, "MaxSpacing", "MaxStaSpcg")
            if ms:
                f.max_station_spacing = ms
            mn = getf(row, "MinStations", "MinNumSta")
            if mn:
                f.min_stations = int(mn)
        for f, row, t in self._frame_rows("frame added mass assignments", "frame assignments - additional mass"):
            f.mass_per_length = getf(row, "MassPerLen", "AddedMass", "Mass", default=0.0) or 0.0
        for f, row, t in self._frame_rows("frame tension and compression limits", "frame assignments - tension/compression"):
            if to_bool(get(row, "TensLimit", "TensionLimit")):
                f.tension_limit = abs(getf(row, "Tension", "TensionValue", default=0.0) or 0.0)
            if to_bool(get(row, "CompLimit", "CompressionLimit")):
                f.compression_limit = abs(getf(row, "Compression", "CompressionValue", default=0.0) or 0.0)
        self._line_springs()

    def _line_springs(self):
        """SAFE / ETABS line springs (property per line object, often on null lines) and
        SAP2000 frame springs (values per frame)."""
        from ..model.objects import LineSpring
        m = self.model
        props: Dict[str, Tuple[float, float]] = {}
        for t in self._tables("spring property definitions - line", "spring props - line", "line spring properties"):
            for row in t.rows:
                n = gets(row, "Name", "Spring")
                if n:
                    props[n] = (getf(row, "VertStiff", "StiffU3", "U3", "Stiffness", default=0.0) or 0.0,
                                getf(row, "RotStiff", "StiffR1", "R1", default=0.0) or 0.0)
                    nl = norm(gets(row, "NonlinOpt", "NonlinOpt3", default="") or "")
                    if nl and not nl.startswith("none") and "linear" not in nl:
                        self._warn(f"Line spring property {n!r}: nonlinear option imported as linear")
        count = 0
        for t in self._tables("frame assignments - line springs", "frame spring assignments", "line spring assignments",
                              "frame assigns - line springs"):
            for row in t.rows:
                name = gets(row, "UniqueName", "Frame", "Line", "Name")
                if name in m.frames and m.frames[name].type != "Null":
                    pi, pj = m.frames[name].point_i, m.frames[name].point_j
                elif name in self._null_lines:
                    pi, pj = self._null_lines[name]
                else:
                    continue
                prop = gets(row, "SpringProp", "Prop")
                if prop:
                    if prop not in props:
                        self.skipped[f"line spring property {prop}"] += 1
                        continue
                    kv, kr = props[prop]
                    k = (0.0, 0.0, kv)
                elif gets(row, "Type") and name in m.frames:  # SAP2000 v14+: Simple spring per direction
                    kk = getf(row, "Stiffness", default=0.0) or 0.0
                    st = norm(gets(row, "SimpleType", default="") or "")
                    if st and "tension and compression" not in st:
                        self._warn(f"Frame {name}: {st} spring imported as linear")
                    d1 = norm(gets(row, "Dir1Type", default="Object Axes") or "")
                    d = str(gets(row, "Dir", default="2") or "2").strip()
                    if "object" in d1 and d.lstrip("+-") in ("1", "2", "3"):
                        fnd = list(m.frames[name].foundation)
                        fnd[int(d.lstrip("+-")) - 1] += kk
                        m.frames[name].foundation = tuple(fnd)
                        count += 1
                        continue
                    self._warn(f"Frame {name}: spring direction {d1} {d} not supported")
                    continue
                else:  # SAP2000: values per unit length in the frame / global directions
                    k = tuple(getf(row, a, b, default=0.0) or 0.0 for a, b in (("UX", "U1"), ("UY", "U2"), ("UZ", "U3")))
                    kr = getf(row, "R1", "RX", default=0.0) or 0.0
                    cs = (gets(row, "CoordSys", "Dir", default="Global") or "Global").lower()
                    if "local" in cs:
                        self._warn("Frame springs in local axes taken as global")
                key = f"LS{len(m.line_springs) + 1}"
                m.line_springs[key] = LineSpring(key, pi, pj, k, kr)
                count += 1


    # -------------------------------------------------------- area assigns
    def _area_rows(self, *prefixes):
        for t in self._tables(*prefixes):
            for row in t.rows:
                a = self.model.areas.get(_area_name(row) or "")
                if a is not None:
                    yield a, row, t

    def _area_assigns(self):
        for a, row, t in self._area_rows("area local axes assignments", "area assignments - local axes",
                                         "area assigns - local axes"):
            a.angle = getf(row, "Angle", default=0.0) or 0.0
        for a, row, t in self._area_rows("area stiffness modifiers", "area assignments - stiffness modifiers",
                                         "area assigns - stiff modifiers", "area property modifiers",
                                         "slab property modifiers"):
            keys = (("f11", ("f11Mod", "f11")), ("f22", ("f22Mod", "f22")), ("f12", ("f12Mod", "f12")),
                    ("m11", ("m11Mod", "m11")), ("m22", ("m22Mod", "m22")), ("m12", ("m12Mod", "m12")),
                    ("v13", ("v13Mod", "v13")), ("v23", ("v23Mod", "v23")), ("mass", ("MMod", "Mass")),
                    ("weight", ("WMod", "Weight")))
            sm = ShellModifiers()
            for attr, ks in keys:
                v = getf(row, *ks)
                if v is not None:
                    setattr(sm, attr, v)
            a.modifiers = sm
        for a, row, t in self._area_rows("area auto mesh assignments", "area assignments - floor auto mesh",
                                         "area assignments - wall auto mesh", "area assigns - floor auto mesh",
                                         "area assigns - wall auto mesh"):
            opt = norm(gets(row, "MeshType", "MeshOption", default="") or "")
            n1, n2 = getf(row, "n1", "N1", "Nh"), getf(row, "n2", "N2", "Nv")
            if "number" in opt or "rectangular" in opt and n1:
                if n1 and n2:
                    a.auto_mesh = True
                    a.mesh_divisions = (int(n1), int(n2))
            elif "maxsize" in opt or "maximumsize" in opt:
                a.auto_mesh = True
                a.max_mesh_size = getf(row, "MaxSize1", "MaxSize", "Size")
            elif "nosubdivision" in opt or opt.startswith("no") or opt == "none":
                a.auto_mesh = False
            elif opt in ("default", "autocookiecut", "autorectangular", "auto", "general"):
                if self.family != "sap":
                    a.auto_mesh = a.type != "Wall" or opt == "autorectangular"
                else:
                    a.auto_mesh = True
        for a, row, t in self._area_rows("area added mass assignments", "area assignments - additional mass"):
            a.mass_per_area = getf(row, "MassPerArea", "Mass", default=0.0) or 0.0
        # area springs: SAP2000 values, ETABS / SAFE spring properties, SAFE soil properties
        props: Dict[str, float] = {}
        for t in self._tables("spring property definitions - area", "spring props - area", "soil properties",
                              "area spring properties", "soil property definitions"):
            for row in t.rows:
                n = gets(row, "Name", "Soil", "SoilProp", "Spring")
                k = getf(row, "StiffU3", "SubModulus", "Subgrade", "SubgradeMod", "U3", "Kz")
                if n and k is not None:
                    props[n] = k
                    nl = (gets(row, "NonlinOpt3", "NonlinOpt", default="") or "").lower()
                    if "compression" in nl or "tension" in nl:
                        self._warn(f"Spring / soil property {n!r}: {nl} behaviour imported as linear")
        for a, row, t in self._area_rows("area spring assignments", "area assignments - springs", "area assigns - springs",
                                         "area assignments - area springs", "soil property assignments",
                                         "area soil assignments"):
            prop = gets(row, "SpringProp", "SoilProp", "Soil", "Prop")
            if prop:
                if prop not in props:
                    self.skipped[f"area spring property {prop}"] += 1
                    continue
                a.spring = props[prop]
            else:
                k = getf(row, "U3", "Stiffness", "Kz")
                if k:
                    a.spring = k
                if getf(row, "U1") or getf(row, "U2"):
                    self.skipped["in-plane area springs"] += 1

    # ------------------------------------------------- diaphragms / constraints
    def _diaphragms_and_constraints(self):
        m = self.model
        for r in self._rows("Diaphragm Definitions", "CONSTRAINT DEFINITIONS - DIAPHRAGM", "DIAPHRAGM DEFINITIONS",
                            "DIAPHRAGM NAMES"):
            name = gets(r, "Name", "Diaphragm")
            if name:
                rig = (gets(r, "Rigidity", "RigidityType", "Type", default="Rigid") or "Rigid").lower()
                m.diaphragms[name] = Diaphragm(name, "semi" not in rig)
                if (gets(r, "Axis", default="Z") or "Z").upper() not in ("Z", "AUTO"):
                    self._warn(f"Diaphragm constraint {name}: only Z-axis diaphragms are supported")
        other = Counter()
        for t in self._tables("joint constraint assignments", "joint assignments - diaphragms", "joint assigns - diaphragms",
                              "point diaphragm assignments"):
            for row in t.rows:
                p = m.points.get(gets(row, "Joint", "Point", "UniqueName", "Name") or "")
                if p is None:
                    continue
                name = gets(row, "Constraint", "Diaphragm")
                ctype = (gets(row, "Type", default="Diaphragm") or "Diaphragm").lower()
                if not name or norm(name) in ("none", "disconnect", "fromshellobject", "fromareaobject"):
                    continue  # "From Shell Object": the joint takes the diaphragm of its floor
                if "diaph" not in ctype:
                    other[ctype] += 1
                    continue
                p.diaphragm = name
                m.diaphragms.setdefault(name, Diaphragm(name, True))
        for ctype, n in other.items():
            self.skipped[f"joint constraints of type {ctype}"] += n
        for a, row, t in self._area_rows("area assignments - diaphragms", "area assigns - diaphragms",
                                         "area diaphragm assignments"):
            name = gets(row, "Diaphragm")
            if name and norm(name) not in ("none", "disconnect", "fromshellobject"):
                a.diaphragm = name
                m.diaphragms.setdefault(name, Diaphragm(name, True))

    def _safe_supports(self):
        """SAFE: the far (free) ends of columns and walls below / above the slab are
        supports – fixed unless a restraint or spring is assigned."""
        m = self.model
        if self.program != "SAFE":
            return
        count = Counter()
        for f in m.frames.values():
            for p in (f.point_i, f.point_j):
                count[p] += 1
        for a in m.areas.values():
            for p in a.points:
                count[p] += 1
        n = 0
        for f in m.frames.values():
            if f.type != "Column":
                continue
            for pn in (f.point_i, f.point_j):
                p = m.points[pn]
                if count[pn] == 1 and not any(p.restraints) and not any(p.springs) and not any(p.rotational_springs):
                    p.restraints = (True,) * 6
                    n += 1
        for a in m.areas.values():
            if a.type != "Wall":
                continue
            zs = [m.points[q].z for q in a.points]
            for pn in a.points:
                p = m.points[pn]
                if abs(p.z - min(zs)) < 1e-9 and count[pn] == 1 and not any(p.restraints) and not any(p.springs):
                    p.restraints = (True,) * 6
                    n += 1
        if n:
            self._warn(f"SAFE: {n} free column / wall end point(s) fixed (SAFE support convention)")

    def _groups(self):
        m = self.model
        for r in self._rows("GROUPS 2 - ASSIGNMENTS", "Group Assignments", "GROUP ASSIGNMENTS"):
            g = gets(r, "GroupName", "Group")
            obj = gets(r, "ObjectLabel", "UniqueName", "Label", "Object")
            typ = (gets(r, "ObjectType", "Type", default="") or "").lower()
            if not (g and obj):
                continue
            grp = m.groups.setdefault(g, Group(g))
            if typ.startswith(("joint", "point")):
                grp.points.append(obj)
            elif typ.startswith(("frame", "line", "beam", "column", "brace")):
                grp.frames.append(obj)
            elif typ.startswith(("area", "shell", "floor", "wall")):
                grp.areas.append(obj)
        for r in self._rows("GROUPS 1 - DEFINITIONS", "Group Definitions"):
            g = gets(r, "GroupName", "Name")
            if g and g.upper() != "ALL":
                m.groups.setdefault(g, Group(g))
        m.groups.pop("ALL", None)
        m.groups.pop("All", None)

    # ------------------------------------------------------------- patterns
    def _load_patterns(self):
        m = self.model
        lp_tables = self._tables("load pattern definitions", "load patterns", "static load cases", "load pattern")
        rows = [r for t in lp_tables if "auto" not in t.name.lower() for r in t.rows]
        if not rows:
            # SAP2000 v8 - v13 / SAFE v12: "LOAD CASE DEFINITIONS" are load patterns
            for t in self._tables("load case definitions", "load cases"):
                if t.has("SelfWtMult", "SelfWeight"):
                    rows += t.rows
        for r in rows:
            name = gets(r, "LoadPat", "LoadPattern", "Name", "LoadCase", "Pattern")
            if not name:
                continue
            typ = gets(r, "DesignType", "Type", "DesType", default="Other") or "Other"
            m.load_patterns[name] = LoadPattern(name, typ.replace(" ", "").title() if typ.isupper() else typ.replace(" ", ""),
                                                getf(r, "SelfWtMult", "SelfWeightMultiplier", "SelfWeight", default=0.0) or 0.0)
            auto = gets(r, "AutoLoad", "AutoLatLoad", default="") or ""
            if auto and auto.lower() not in ("none", ""):
                self.skipped[f"auto lateral loads ({auto})"] += 1

    # ------------------------------------------------------------- functions
    def _functions(self):
        m = self.model
        for t in self._tables("function - response spectrum", "functions - response spectrum", "response spectrum function",
                              "functions - rs"):
            for row in t.rows:
                name = gets(row, "Name", "Function", "FuncName")
                per = getf(row, "Period", "Time", "T")
                val = getf(row, "Accel", "Value", "SA", "Acceleration")
                if not name or per is None or val is None:
                    continue
                fn = m.functions.get(name)
                if not isinstance(fn, ResponseSpectrumFunction):
                    fn = ResponseSpectrumFunction(name)
                    m.functions[name] = fn
                fn.periods.append(per)
                fn.accelerations.append(val)
                d = getf(row, "FuncDamp", "DampRatio", "DampingRatio", "Damping")
                if d is not None:
                    fn.damping = d
            # from-file functions without tabulated values: read the file
            if "file" in t.name.lower():
                for row in t.rows:
                    name = gets(row, "Name")
                    path = gets(row, "File", "FileName", "Filename")
                    if name and path and name not in m.functions:
                        self._function_from_file(name, path, int(getf(row, "HeaderSkip", "HeaderLines", default=0) or 0),
                                                 (gets(row, "ValueType", "PrdFreq", default="Period") or "Period").lower())
        for name, fn in list(m.functions.items()):
            if isinstance(fn, ResponseSpectrumFunction):
                pairs = sorted(zip(fn.periods, fn.accelerations))
                fn.periods = [p for p, _ in pairs]
                fn.accelerations = [a for _, a in pairs]
        th = sum(len(t.rows) for t in self._tables("function - time history", "functions - time history"))
        if th:
            self.skipped["time history functions"] += th

    def _function_from_file(self, name, path, skip, kind):
        cands = [path, os.path.join(self.source_dir or "", os.path.basename(path))]
        for c in cands:
            if c and os.path.isfile(c):
                xs, ys = [], []
                with open(c, encoding="utf-8", errors="replace") as fh:
                    for k, line in enumerate(fh):
                        if k < skip:
                            continue
                        nums = [to_float(v) for v in re.split(r"[\s,;]+", line.strip()) if v]
                        nums = [v for v in nums if v is not None]
                        if len(nums) >= 2:
                            xs.append(1.0 / nums[0] if kind.startswith("freq") and nums[0] else nums[0])
                            ys.append(nums[1])
                if xs:
                    pairs = sorted(zip(xs, ys))
                    self.model.functions[name] = ResponseSpectrumFunction(name, [p for p, _ in pairs],
                                                                          [a for _, a in pairs])
                    return
        self._warn(f"Spectrum function {name!r}: file {path!r} not found")

    # ------------------------------------------------------------- loads
    def _pattern(self, row) -> Optional[str]:
        name = gets(row, "LoadPat", "LoadPattern", "LoadCase", "Pattern", "Case")
        if name and name in self.model.load_patterns:
            return name
        if name:
            self.skipped[f"loads of undefined pattern {name}"] += 1
        return None

    def _loads(self):
        m = self.model
        for t in self._tables("joint loads - force", "joint loads assignments - force", "joint loads - force",
                              "load assignments - point loads", "point loads", "point object loads"):
            for row in t.rows:
                p = m.points.get(gets(row, "Joint", "Point", "UniqueName", "Name") or "")
                pat = self._pattern(row) if p is not None else None
                if p is None or pat is None:
                    continue
                fgrav = getf(row, "Fgrav", "FGravity")
                F = [getf(row, a, b, default=0.0) or 0.0 for a, b in (("F1", "FX"), ("F2", "FY"), ("F3", "FZ"))]
                M = [getf(row, a, b, default=0.0) or 0.0 for a, b in (("M1", "MX"), ("M2", "MY"), ("M3", "MZ"))]
                if fgrav:
                    F[2] -= fgrav
                cs = (gets(row, "CoordSys", default="GLOBAL") or "GLOBAL").upper()
                if cs not in ("GLOBAL", "LOCAL"):  # a named coordinate system: not converted
                    self.skipped[f"joint loads in coordinate system {cs}"] += 1
                    self._warn(f"Joint {p.name}: load in coordinate system {cs} not imported (only GLOBAL and the "
                               "joint LOCAL axes are converted)")
                    continue
                if cs == "LOCAL" and p.local_axes:  # F1 F2 F3 / M1 M2 M3 along the joint local axes
                    from ..mesh.geometry import joint_axes
                    R = joint_axes(*p.local_axes)  # rows = local 1, 2, 3 in global
                    F = list(R.T @ F)
                    M = list(R.T @ M)
                p.loads.append(PointLoad(pat, tuple(float(v) for v in F), tuple(float(v) for v in M)))
        for t in self._tables("joint loads - ground displacement", "joint loads assignments - ground displacement",
                              "joint loads - displacement"):
            for row in t.rows:
                p = m.points.get(gets(row, "Joint", "Point", "UniqueName", "Name") or "")
                pat = self._pattern(row) if p is not None else None
                if p is None or pat is None:
                    continue
                tr = tuple(getf(row, a, b, default=0.0) or 0.0 for a, b in (("U1", "UX"), ("U2", "UY"), ("U3", "UZ")))
                rot = tuple(getf(row, a, b, default=0.0) or 0.0 for a, b in (("R1", "RX"), ("R2", "RY"), ("R3", "RZ")))
                cs = (gets(row, "CoordSys", default="GLOBAL") or "GLOBAL").upper()
                if cs not in ("GLOBAL", "LOCAL"):
                    self.skipped[f"ground displacements in coordinate system {cs}"] += 1
                    self._warn(f"Joint {p.name}: ground displacement in coordinate system {cs} not imported")
                    continue
                p.ground_displacements.append(GroundDisplacement(pat, tr, rot, "Local" if cs == "LOCAL" else "Global"))
        for t in self._tables("frame loads - gravity", "frame loads assignments - gravity"):
            for row in t.rows:
                f = m.frames.get(gets(row, "Frame", "Line", "UniqueName", "Name") or "")
                pat = self._pattern(row) if f is not None else None
                if f is None or pat is None:
                    continue
                for ax in "XYZ":
                    mult = getf(row, f"Multiplier{ax}", f"Mult{ax}", default=0.0) or 0.0
                    if mult:
                        f.loads.append(FrameLoad(pat, "gravity", ax, "Force", 0.0, 1.0, False, mult, mult))
        for t in self._tables("frame loads - distributed", "frame loads assignments - distributed",
                              "load assignments - line loads - distributed", "line loads - distributed",
                              "load assignments - line loads"):
            for row in t.rows:
                f = m.frames.get(gets(row, "Frame", "Line", "UniqueName", "Name") or "")
                pat = self._pattern(row) if f is not None else None
                if f is None or pat is None:
                    continue
                ltype = "Moment" if "moment" in (gets(row, "Type", "LoadType", default="Force") or "").lower() else "Force"
                absolute = (gets(row, "DistType", "DistanceType", default="RelDist") or "").lower().startswith("abs")
                if absolute:
                    a = getf(row, "AbsDistA", "AbsoluteDistanceA", default=0.0)
                    b = getf(row, "AbsDistB", "AbsoluteDistanceB", default=0.0)
                else:
                    a = getf(row, "RelDistA", "RelativeDistanceA", default=0.0)
                    b = getf(row, "RelDistB", "RelativeDistanceB", default=1.0)
                if ltype == "Moment":
                    va, vb = getf(row, "MOverLA", "MomentA", default=0.0), getf(row, "MOverLB", "MomentB", default=0.0)
                else:
                    va = getf(row, "FOverLA", "ForceA", "FA", "LoadA", default=None)
                    vb = getf(row, "FOverLB", "ForceB", "FB", "LoadB", default=None)
                    if va is None:
                        va = vb = getf(row, "FOverL", "Force", "Load", default=0.0)
                    vb = va if vb is None else vb
                f.loads.append(FrameLoad(pat, "distributed", _direction(gets(row, "Dir", "Direction")), ltype,
                                         a or 0.0, 1.0 if b is None else b, absolute, va or 0.0, vb or 0.0))
        for t in self._tables("frame loads - point", "frame loads assignments - point", "load assignments - line loads - point"):
            for row in t.rows:
                f = m.frames.get(gets(row, "Frame", "Line", "UniqueName", "Name") or "")
                pat = self._pattern(row) if f is not None else None
                if f is None or pat is None:
                    continue
                ltype = "Moment" if "moment" in (gets(row, "Type", "LoadType", default="Force") or "").lower() else "Force"
                absolute = (gets(row, "DistType", "DistanceType", default="RelDist") or "").lower().startswith("abs")
                pos = getf(row, *(("AbsDist", "AbsoluteDistance") if absolute else ("RelDist", "RelativeDistance")),
                           default=0.5)
                val = getf(row, "Moment" if ltype == "Moment" else "Force", "Value", default=0.0)
                f.loads.append(FrameLoad(pat, "point", _direction(gets(row, "Dir", "Direction")), ltype,
                                         pos, pos, absolute, val, val))
        for t in self._tables("area loads - uniform", "area load assignments - uniform", "area loads assignments - uniform",
                              "load assignments - surface loads", "surface loads", "shell loads - uniform",
                              "area loads - uniform to frame", "area load assignments - uniform to frame"):
            to_frame = "to frame" in t.name.lower()
            for row in t.rows:
                a = m.areas.get(_area_name(row) or "")
                pat = self._pattern(row) if a is not None else None
                if a is None or pat is None:
                    continue
                if "set" in t.name.lower():
                    continue
                val = getf(row, "UnifLoad", "Load", "Value", "Pressure", default=0.0) or 0.0
                a.loads.append(AreaLoad(pat, val, _direction(gets(row, "Dir", "Direction"))))
                if to_frame or a.section is None:
                    self._warn("Area loads on null / load-transfer areas ('uniform to frame') are applied to the "
                               "area itself; areas without a section carry no load in this version")
        jpat: Dict[Tuple[str, str], float] = {}
        for t in self._tables("joint pattern assignments", "joint patterns"):
            for row in t.rows:
                jn, pn = gets(row, "Joint", "Point"), gets(row, "Pattern")
                if jn and pn:
                    jpat[(pn, jn)] = getf(row, "Value", default=0.0) or 0.0
        for t in self._tables("frame loads - temperature", "frame loads assignments - temperature"):
            for row in t.rows:
                f = m.frames.get(gets(row, "Frame", "Line", "UniqueName", "Name") or "")
                pat = self._pattern(row) if f is not None else None
                if f is None or pat is None:
                    continue
                typ = norm(gets(row, "Type", default="Temperature") or "Temperature")
                if "grad" in typ and "3" in typ:
                    ltype, key = "Gradient3", ("TempGrad3", "Gradient3", "Value")
                elif "grad" in typ:
                    ltype, key = "Gradient2", ("TempGrad2", "Gradient2", "Value")
                else:
                    ltype, key = "Temperature", ("Temp", "Temperature", "Value")
                v = getf(row, *key, default=0.0) or 0.0
                va = vb = v
                jp = gets(row, "JtPattern", "JointPattern")
                if jp and jp.lower() not in ("none", ""):
                    # the joint pattern scales the value, linear between the end joints
                    va = v * jpat.get((jp, f.point_i), 0.0)
                    vb = v * jpat.get((jp, f.point_j), 0.0)
                f.loads.append(FrameLoad(pat, "temperature", "1", ltype, 0.0, 1.0, False, va, vb))
        for t in self._tables("area loads - temperature", "joint loads - temperature", "area load assignments - temperature"):
            if t.rows:
                self.skipped["temperature loads (areas / joints)"] += len(t.rows)
        for t in self._tables("area load assignments - non-uniform", "area loads - surface pressure",
                              "area loads - gravity", "area load assignments - wind"):
            if t.rows:
                self.skipped[f"area loads ({t.name})"] += len(t.rows)

    def _tendons(self):
        """SAP2000 tendon objects (path, section, loads)."""
        from ..model.objects import Tendon, TendonLoad, TendonPoint
        m = self.model
        conn = {}
        for t in self._tables("connectivity - tendon"):
            for row in t.rows:
                n = gets(row, "Tendon", "Name")
                if n:
                    conn[n] = row
        if not conn:
            return
        secs = {}
        for t in self._tables("tendon section definitions"):
            for row in t.rows:
                n = gets(row, "TendonSect", "Name")
                if n:
                    secs[n] = row
        assign = {}
        for t in self._tables("tendon section assignments"):
            for row in t.rows:
                assign[gets(row, "Tendon")] = gets(row, "TendonSect", "Section")
        general = {}
        for t in self._tables("tendon layout data 01"):
            for row in t.rows:
                general[gets(row, "Tendon")] = row
        paths: Dict[str, list] = {}
        for t in self._tables("tendon layout data 02"):
            for row in t.rows:
                n = gets(row, "Tendon")
                st = norm(gets(row, "SegType", default="") or "")
                xyz = tuple(getf(row, f"{c}Global", c, default=0.0) or 0.0 for c in "XYZ")
                kind = ("Start" if st.startswith("start") else "ParabolaIntermediate" if "parabolaintermediate" in st
                        else "ParabolaEnd" if "parabolaend" in st else "Line")
                if kind == "Line" and not st.startswith("line") and st:
                    self._warn(f"Tendon {n}: segment type {gets(row, 'SegType')!r} taken as a straight line")
                paths.setdefault(n, []).append(TendonPoint(kind, xyz))
        for n, row in conn.items():
            path = paths.get(n)
            if not path:
                pi, pj = m.points.get(gets(row, "JointI") or ""), m.points.get(gets(row, "JointJ") or "")
                if pi is None or pj is None:
                    continue
                path = [TendonPoint("Start", tuple(pi.xyz)), TendonPoint("Line", tuple(pj.xyz))]
            sec = secs.get(assign.get(n) or "", {})
            area = getf(sec, "Area", default=None)
            if area is None:
                dia = getf(sec, "Diameter", default=0.0) or 0.0
                area = math.pi * dia * dia / 4
            opt = norm(gets(sec, "ModelOpt", default="Loads") or "Loads")
            g = general.get(n, {})
            m.tendons[n] = Tendon(n, path, area=area, material=self._material_or_default(gets(sec, "Material")),
                                  model_as="Elements" if opt.startswith("elem") else "Loads",
                                  max_discretization=getf(g, "MaxDiscLen", default=None))
        for t in self._tables("tendon loads - tension force or stress", "tendon loads"):
            for row in t.rows:
                td = m.tendons.get(gets(row, "Tendon") or "")
                pat = self._pattern(row) if td is not None else None
                if td is None or pat is None:
                    continue
                if norm(gets(row, "LoadType", default="Force") or "").startswith("stress"):
                    force = (getf(row, "Stress", default=0.0) or 0.0) * td.area
                else:
                    force = getf(row, "Force", default=0.0) or 0.0
                jf = norm(gets(row, "JackFrom", default="I-End") or "")
                if getf(row, "LossAnchor", default=0.0):
                    self._warn(f"Tendon {td.name}: anchorage set loss not supported (ignored)")
                loss = sum(getf(row, k, default=0.0) or 0.0
                           for k in ("LossEShort", "LossCreep", "LossShrink", "LossSRelax"))
                td.loads.append(TendonLoad(pat, force, "Both" if jf.startswith("both") else "J" if jf.startswith("j")
                                           else "I", getf(row, "Curvature", default=0.0) or 0.0,
                                           getf(row, "Wobble", default=0.0) or 0.0, loss))

    # ------------------------------------------------------------- cases
    def _cases(self):
        m = self.model
        types: "OrderedDict[str, str]" = OrderedDict()
        opts: Dict[str, dict] = {}
        loads: Dict[str, Dict[str, float]] = {}
        rs_loads: Dict[str, List[Tuple[str, str, float, float]]] = {}
        has_patterns_table = bool(self.tables.matching("load pattern definitions", "load patterns"))
        sum_tables = self._tables("load case definitions", "analysis case definitions", "load cases 01", "load cases - summary",
                                  "load case definitions - summary")
        done = {t.key for t in sum_tables}
        for t in sum_tables:
            tl = t.name.lower()
            if t.has("SelfWtMult") and not has_patterns_table and "analysis" not in tl:
                continue  # old SAP2000: these are load patterns
            if "damping" in tl:  # ETABS "Load Case Definitions - Damping - Modal": options of RS / TH cases
                for row in t.rows:
                    name = gets(row, "Case", "Name")
                    if name:
                        opts.setdefault(name, {}).update(row)
                continue
            for row in t.rows:
                name = gets(row, "Case", "Name", "LoadCase")
                typ = gets(row, "Type", "CaseType")
                if not name:
                    continue
                if typ and (name not in types or "summary" in tl or "definitions" == tl.split(" - ")[-1]):
                    types[name] = typ
                elif name not in types:
                    # ETABS per-type tables: type from the table name
                    types[name] = t.name.split(" - ", 1)[-1] if " - " in t.name else (typ or "")
                opts.setdefault(name, {}).update(row)
                if get(row, "LoadName", "LoadPat"):
                    self._case_load_row(name, row, loads, rs_loads)
        for t in self._tables("modal case definitions", "case - modal", "load cases 03 - modal"):
            for row in t.rows:
                name = gets(row, "Case", "Name")
                if name:
                    types.setdefault(name, "Modal - " + ("Ritz" if "ritz" in t.name.lower() else "Eigen"))
                    opts.setdefault(name, {}).update(row)
        for t in self._tables("case - static", "load cases 02 - static", "case - nonlinear static",
                              "case - response spectrum", "load case definitions - response spectrum",
                              "load case definitions - linear static", "load case definitions - nonlinear static",
                              "load cases - linear static", "load cases - response spectrum"):
            if t.key in done:
                continue
            done.add(t.key)
            for row in t.rows:
                name = gets(row, "Case", "Name", "LoadCase")
                if not name:
                    continue
                tl = t.name.lower()
                if name not in types:
                    types[name] = ("Response Spectrum" if "spectrum" in tl else "Nonlinear Static" if "nonlinear" in tl
                                   else "Linear Static")
                opts.setdefault(name, {}).update({k: v for k, v in row.items() if k not in ("loadname", "loadsf",
                                                                                             "loadtype", "loadpat", "sf")})
                if get(row, "LoadName", "LoadPat"):
                    self._case_load_row(name, row, loads, rs_loads)
        for name, typ in types.items():
            o = opts.get(name, {})
            tl = norm(typ)
            if tl.startswith("linstatic") or tl in ("linearstatic", "static", "linear") or "linearstatic" in tl:
                pd = norm(gets(o, "GeoNonLin", "GeoNonlin", default="") or "")
                m.load_cases[name] = LinearStaticCase(name=name, loads=loads.get(name, {}))
                if pd and pd != "none":
                    self._warn(f"Case {name}: geometric nonlinearity ignored in a linear case")
            elif tl.startswith("nonstatic") or "nonlinearstatic" in tl:
                pd = norm(gets(o, "GeoNonLin", "GeoNonlin", default="") or "")
                if "pdelta" in pd and "large" not in pd:
                    m.load_cases[name] = PDeltaCase(name=name, loads=loads.get(name, {}))
                else:
                    m.load_cases[name] = LinearStaticCase(name=name, loads=loads.get(name, {}), nonlinear=True)
                    if "large" in pd:
                        self._warn(f"Nonlinear static case {name}: large displacements ignored")
            elif "modal" in tl and "hist" not in tl or tl.startswith("linmodal") or tl.startswith("modal"):
                ritz = "ritz" in tl or norm(gets(o, "ModeType", "Type2", default="") or "") == "ritz"
                m.load_cases[name] = ModalCase(
                    name=name, max_modes=int(getf(o, "MaxNumModes", "MaxModes", default=12) or 12),
                    min_modes=int(getf(o, "MinNumModes", "MinModes", default=1) or 1),
                    method="Ritz" if ritz else "Eigen",
                    shift=getf(o, "EigenShift", "FreqShift", default=0.0) or 0.0,
                    cutoff=getf(o, "EigenCutoff", "CutoffFreq", default=0.0) or 0.0,
                    tolerance=getf(o, "EigenTol", "ConvTol", "ConvergenceTol", default=1e-9) or 1e-9)
            elif "respspec" in tl or "responsespectrum" in tl or tl == "rs":
                m.load_cases[name] = ("RS", o)  # resolved after all modal cases exist
            elif "buck" in tl:
                self.skipped["buckling cases"] += 1
            elif "hist" in tl or "th" == tl[:2]:
                self.skipped[f"time history cases ({typ})"] += 1
            elif "moving" in tl or "steady" in tl or "psd" in tl or "hyper" in tl or "staged" in tl or "multi" in tl:
                self.skipped[f"{typ} cases"] += 1
            else:
                if loads.get(name):
                    m.load_cases[name] = LinearStaticCase(name=name, loads=loads[name])
                else:
                    self.skipped[f"load case type {typ}"] += 1
        for name, val in list(m.load_cases.items()):
            if isinstance(val, tuple):
                o = val[1]
                modal = gets(o, "ModalCase") or next((n for n, c in m.load_cases.items() if isinstance(c, ModalCase)), "")
                if modal not in m.load_cases or not isinstance(m.load_cases[modal], ModalCase):
                    if not modal:
                        modal = "MODAL"
                    m.load_cases.setdefault(modal, ModalCase(name=modal))
                    self._warn(f"Response spectrum case {name}: modal case {modal!r} added")
                rl = rs_loads.get(name, [])
                angle = next((a for *_, a in rl if a), 0.0) or getf(o, "Angle", default=0.0) or 0.0
                combo = gets(o, "ModalCombo", "ModalComboMethod", "ModeComb", default="CQC") or "CQC"
                cn = norm(combo)
                combo = ("CQC" if cn.startswith("cqc") else "SRSS" if cn == "srss" else "ABS" if cn.startswith("abs")
                         else "GMC" if cn.startswith("gmc") else "NRC10" if "nrc" in cn or "10" in cn else
                         "DSC" if "double" in cn else combo.upper())
                dcombo = norm(gets(o, "DirCombo", "DirectionalComboType", "DirComb", default="SRSS") or "SRSS")
                dcombo = "ABS" if dcombo.startswith("abs") else "CQC3" if "cqc3" in dcombo else "SRSS"
                damp = getf(o, "ConstDamp", "ConstantDamping", "Damping", "DampRatio", "DampingRatio", "ModalDamp")
                case = ResponseSpectrumCase(
                    name=name, modal_case=modal, loads=[SpectrumLoad(d, fn, sf) for d, fn, sf, _ in rl],
                    modal_combination=combo, directional_combination=dcombo,
                    damping=0.05 if damp is None else damp, angle=angle,
                    eccentricity_ratio=getf(o, "EccenRatio", "EccentricityRatio", "Eccen", default=0.0) or 0.0,
                    directional_abs_factor=getf(o, "AbsSF", "AbsoluteSF", "DirAbsSF", default=1.0) or 1.0)
                missing = [fn for _, fn, _, _ in rl if fn not in m.functions]
                if missing:
                    self._warn(f"Response spectrum case {name}: function(s) {', '.join(sorted(set(missing)))} "
                               "have no tabulated values; case skipped")
                    del m.load_cases[name]
                else:
                    m.load_cases[name] = case
        if not m.load_cases and m.load_patterns:
            for p in m.load_patterns:
                m.load_cases[p] = LinearStaticCase(name=p, loads={p: 1.0})
            self._warn("No load cases in the tables; one linear static case per load pattern created")

    def _case_load_row(self, name, row, loads, rs_loads):
        ltype = norm(gets(row, "LoadType", default="Load pattern") or "")
        lname = gets(row, "LoadName", "LoadPat") or ""
        sf = getf(row, "LoadSF", "SF", "ScaleFactor", default=1.0)
        fn = gets(row, "Function", "Func")
        if fn or ltype.startswith("accel"):
            if fn:
                rs_loads.setdefault(name, []).append((lname.upper().replace("ACCEL", "").strip() or "U1", fn,
                                                      getf(row, "TransAccSF", "TransAccelSF", "LoadSF", "SF", default=1.0) or 1.0,
                                                      getf(row, "Angle", default=0.0) or 0.0))
            else:
                self.skipped["acceleration loads in static cases"] += 1
            return
        if lname in self.model.load_patterns:
            loads.setdefault(name, {})[lname] = loads.get(name, {}).get(lname, 0.0) + (sf if sf is not None else 1.0)
        else:
            self.skipped[f"case {name}: load {lname} ({ltype or 'pattern'})"] += 1

    # ------------------------------------------------------------- combos
    def _combos(self):
        m = self.model
        items: "OrderedDict[str, list]" = OrderedDict()
        ctype: Dict[str, str] = {}
        for t in self._tables("combination definitions", "load combination definitions", "load combinations",
                              "combinations", "load combos"):
            for row in t.rows:
                name = gets(row, "ComboName", "Combo", "Name")
                if not name:
                    continue
                typ = gets(row, "ComboType", "Type")
                if typ:
                    ctype.setdefault(name, typ)
                item = gets(row, "CaseName", "LoadName", "Load", "Case", "LoadCase")
                items.setdefault(name, [])
                if item:
                    items[name].append((item, getf(row, "ScaleFactor", "SF", default=1.0),
                                        (gets(row, "CaseType", default="") or "").lower()))
        for name, its in items.items():
            combo = LoadCombination(name=name, type=_combo_type(ctype.get(name)))
            for item, sf, ct in its:
                if item in items or "combo" in ct:
                    combo.items.append(ComboItem(item, sf if sf is not None else 1.0, True))
                elif item in m.load_cases:
                    combo.items.append(ComboItem(item, sf if sf is not None else 1.0, False))
                elif item in m.load_patterns:
                    # SAFE v12 / old SAP2000: combinations of load patterns -> their linear case
                    if item not in m.load_cases:
                        m.load_cases[item] = LinearStaticCase(name=item, loads={item: 1.0})
                    combo.items.append(ComboItem(item, sf if sf is not None else 1.0, False))
                else:
                    self.skipped[f"combination {name}: unknown case {item}"] += 1
            if not combo.items:
                self._warn(f"Combination {name}: none of its cases could be imported; combination skipped")
                continue
            m.load_combinations[name] = combo

    # ------------------------------------------------------------- mass
    def _mass_source(self):
        m = self.model
        rows = self._rows("MASS SOURCE", "Mass Source Definition", "MASS SOURCE DEFINITION", "Mass Source")
        sap = self.family != "etabs"
        if not rows:
            if sap:
                m.mass_sources = {"MSSSRC1": MassSource(name="MSSSRC1", include_vertical=True, lump_at_stories=False)}
            return
        sources: "OrderedDict[str, MassSource]" = OrderedDict()
        for r in rows:
            name = gets(r, "MassSource", "Name", default="MSSSRC1") or "MSSSRC1"
            ms = sources.get(name)
            if ms is None:
                ms = MassSource(name=name, include_vertical=sap, lump_at_stories=not sap)
                frm = (gets(r, "MassFrom", default="") or "").lower()
                if frm:  # SAP2000 v14 and earlier
                    ms.include_element_mass = "element" in frm
                    ms.include_added_mass = "element" in frm or "mass" in frm
                    use_loads = "load" in frm
                else:
                    ms.include_element_mass = to_bool(get(r, "Elements", "SourceSelf", "SourceSelfMass",
                                                          "IncludeElements"), True)
                    ms.include_added_mass = to_bool(get(r, "Masses", "SourceAdded", "SourceAddedMass",
                                                        "IncludeAddedMass"), True)
                    use_loads = to_bool(get(r, "Loads", "SourceLoads", "SourceLoadPatterns", "IncludeLoads"), False)
                if get(r, "IncLateral", "IncludeLateralMass") is not None:
                    ms.include_lateral = to_bool(get(r, "IncLateral", "IncludeLateralMass"), True)
                if get(r, "IncVertical", "IncludeVerticalMass") is not None:
                    ms.include_vertical = to_bool(get(r, "IncVertical", "IncludeVerticalMass"), False)
                if get(r, "LumpMass") is not None:
                    ms.lump_at_stories = to_bool(get(r, "LumpMass"), True)
                ms.is_default = to_bool(get(r, "IsDefault"), True)
                ms._use_loads = use_loads  # type: ignore[attr-defined]
                sources[name] = ms
            pat = gets(r, "LoadPat", "LoadPattern", "LoadCase")
            if pat and getattr(ms, "_use_loads", True):
                if pat in m.load_patterns:
                    ms.load_patterns[pat] = ms.load_patterns.get(pat, 0.0) + (getf(r, "Multiplier", "SF", default=1.0) or 0.0)
                else:
                    self.skipped[f"mass source pattern {pat}"] += 1
        for ms in sources.values():
            if hasattr(ms, "_use_loads"):
                del ms._use_loads
        if sources:
            if not any(s.is_default for s in sources.values()):
                next(iter(sources.values())).is_default = True
            m.mass_sources = dict(sources)

    # ------------------------------------------------------------- levels
    def _drop_auto_points(self):
        """ETABS auto points (e.g. diaphragm centres) that no object uses carry
        nothing; kept they would only be free joints in the analysis."""
        m = self.model
        used = {p for f in m.frames.values() for p in (f.point_i, f.point_j)}
        used |= {p for a in m.areas.values() for p in a.points}
        drop = [n for n, p in m.points.items() if p.is_auto and n not in used and not p.loads and not any(p.mass)
                and not any(p.restraints) and not any(p.springs) and not any(p.rotational_springs)]
        for n in drop:
            del m.points[n]
        for g in m.groups.values():
            g.points = [p for p in g.points if p in m.points]

    def _levels(self):
        m = self.model
        if self.family == "etabs":
            if not m.stories and m.points:
                # no story table: stories from the point elevations
                zs = sorted({round(p.z, 6) for p in m.points.values()})
                m._stories_from_levels([("Base" if k == 0 else f"Story{k}", z) for k, z in enumerate(zs)])
            else:
                m.reassign_stories()
            return
        zl = m.z_grid_lines()
        zs = sorted({round(p.z, 9) for p in m.points.values()})
        if zl:
            levels = [(g.name or f"Z{k + 1}", g.ordinate) for k, g in enumerate(zl)]
        elif 0 < len(zs) <= 200:
            levels = [(f"Z{k + 1}", z) for k, z in enumerate(zs)]
        else:
            levels = []
        if levels:
            seen = set()
            uniq = []
            for n, z in levels:
                while n in seen:
                    n = n + "'"
                seen.add(n)
                uniq.append((n, z))
            m.set_z_grid(uniq)
        m.options.grid_mode = "xyz"

    def _report_unused(self):
        ignore = ("program control", "project information", "analysis messages", "table set", "named sets",
                  "preferences", "overwrites", "design", "results", "joint displacements", "joint reactions",
                  "element forces", "element joint forces", "base reactions", "modal", "story drifts", "assembled",
                  "centers of", "groups 3", "options", "material list", "section cut", "auto seismic", "auto wind",
                  "tributary", "concrete", "steel", "rebar", "comp ", "composite", "slab design", "strip", "tendon",
                  "pier", "spandrel", "dimension", "developed", "display", "connectivity - joint",
                  "frame section property definitions - concrete beam reinforcing", "time dependent", "auto wave",
                  "database format", "plot functions", "joint pattern definitions", "solid property definitions",
                  "tables automatically", "cable section definitions", "reinforcing bar", "rebar sizes",
                  "area assignments - insertion point", "area assignments - auto edge", "link property definitions",
                  "frame load transfer", "frame design procedures", "area section property design",
                  "frame assignments - frame floor meshing", "frame assignments - output stations",
                  "analysis modeling options", "tendon section", "strip object", "group definitions")
        for t in self.tables.tables.values():
            if t.key in self._used or not t.rows:
                continue
            tl = t.name.lower()
            if any(k in tl for k in ignore):
                continue
            self.skipped[f"table {t.name}"] += len(t.rows)

    # -------------------------------------------------------------- report
    def report(self) -> str:
        p = f"{self.program} {self.version}".strip()
        lines = [f"Imported {self.model!r} from {p} tables ({self.tables.container})"]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {w}" for w in self.warnings]
        if self.skipped:
            lines.append("Skipped (not supported yet):")
            lines += [f"  - {k}: {v}" for k, v in sorted(self.skipped.items())]
        return "\n".join(lines)


def read_csi_tables(path: str) -> Model:
    return CsiImporter.read(path).model
