"""Legacy SAP2000 v6 / v7 keyword input files (``.s2k``) – *beta*.

Before v8 SAP2000 wrote a keyword format (not database tables)::

    SYSTEM
      DOF=UX,UY,UZ,RX,RY,RZ  LENGTH=FT  FORCE=K
    JOINT
      1  X=0  Y=0  Z=0
    RESTRAINT
      ADD=1  DOF=U1,U2,U3,R1,R2,R3
    MATERIAL
      NAME=STEEL  IDES=S  M=7.324E-04  W=0.2836
        E=29500  U=0.3  A=6.5E-06
    FRAME SECTION
      NAME=FSEC1  MAT=STEEL  SH=R  T=1.5,1
    FRAME
      1  J=1,2  SEC=FSEC1  ANG=0
    LOAD
      NAME=LIVE
        TYPE=FORCE
          ADD=2  UZ=-10

The blocks above plus SHELL SECTION / SHELL, SPRING, MASS, distributed frame
loads, COMBO and MODES are read; everything else is reported as skipped.
Models from these versions are best re-saved with a newer SAP2000 (which
writes the table format) when that is available.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..model import Frame, FrameSection, LoadPattern, Material, Model, Point, ShellSection, Area
from ..model.loads import ComboItem, LinearStaticCase, LoadCombination, ModalCase
from ..model.objects import FrameLoad, PointLoad
from ..units import UnitSystem

_KV = re.compile(r"([A-Za-z][A-Za-z0-9]*)=(\"[^\"]*\"|\S*)")
BLOCKS = ("SYSTEM", "JOINT", "JOINTS", "RESTRAINT", "RESTRAINTS", "CONSTRAINT", "CONSTRAINTS", "SPRING", "SPRINGS",
          "MASS", "MASSES", "MATERIAL", "MATERIALS", "FRAME SECTION", "FRAME SECTIONS", "SHELL SECTION",
          "SHELL SECTIONS", "FRAME", "FRAMES", "SHELL", "SHELLS", "LOAD", "LOADS", "COMBO", "COMBOS", "MODES",
          "MODE", "SPEC", "NLPROP", "NLLINK", "PDELTA", "OUTPUT", "TITLE", "FUNCTION", "HISTORY", "PATTERN",
          "LOCAL", "PRESTRESS", "SOLID", "ASOLID", "PLANE", "SELECT", "END")


def _list(v: str) -> List[str]:
    return [x for x in v.strip('"').split(",") if x != ""]


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _range(spec: str) -> List[str]:
    """ADD=5 or ADD=1,10,1 (first, last, increment)."""
    parts = _list(spec)
    if len(parts) >= 2 and all(re.fullmatch(r"-?\d+", p) for p in parts[:3]):
        a, b = int(parts[0]), int(parts[1])
        inc = int(parts[2]) if len(parts) > 2 else 1
        return [str(k) for k in range(a, b + (1 if inc > 0 else -1), inc or 1)]
    return parts


def _cyclic(a, b, c, d) -> bool:
    """True when a-b-c-d is a proper (non bow-tie) quadrilateral: its diagonals cross."""
    import numpy as np

    P = [np.array([p.x, p.y, p.z], float) for p in (a, b, c, d)]
    n = np.cross(P[1] - P[0], P[2] - P[0]) + np.cross(P[2] - P[0], P[3] - P[0])
    if np.linalg.norm(n) < 1e-14:
        return True
    n = n / np.linalg.norm(n)
    u = P[1] - P[0]
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(n, u)
    q = [(float(np.dot(p - P[0], u)), float(np.dot(p - P[0], v))) for p in P]

    def side(o, p, r):
        return (p[0] - o[0]) * (r[1] - o[1]) - (p[1] - o[1]) * (r[0] - o[0])

    d1, d2 = side(q[0], q[2], q[1]), side(q[0], q[2], q[3])
    d3, d4 = side(q[1], q[3], q[0]), side(q[1], q[3], q[2])
    return d1 * d2 < 0 and d3 * d4 < 0


@dataclass
class LegacyS2KImporter:
    model: Model = field(default_factory=lambda: Model("imported"))
    skipped: Counter = field(default_factory=Counter)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def read(cls, path: str) -> "LegacyS2KImporter":
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        imp = cls()
        imp.model.info.name = os.path.splitext(os.path.basename(path))[0]
        imp.parse(text)
        return imp

    def parse(self, text: str):
        m = self.model
        m.use_csi_frame_defaults()  # frames keep what the file says
        m.options.grid_mode = "xyz"
        block = ""
        current: Dict[str, object] = {}
        load_type = ""
        pattern = None
        for raw in text.splitlines():
            line = raw.split(";")[0].rstrip()
            if not line.strip():
                continue
            head = line.strip().upper()
            if not line[:1].isspace() and (head in BLOCKS or head.split()[0] in BLOCKS and "=" not in head):
                block = head if head in BLOCKS else head.split()[0]
                current = {}
                load_type = ""
                if block == "END":
                    break
                continue
            body = line.strip()
            kv = {k.upper(): v.strip('"') for k, v in _KV.findall(body)}
            label = body.split()[0] if "=" not in body.split()[0] else None
            b = block.rstrip("S") if block not in ("MASS",) else block
            if block.endswith("SECTIONS"):
                b = block[:-1]
            try:
                if b == "SYSTEM":
                    self._system(kv)
                elif b == "JOINT" and label:
                    m.points[label] = Point(name=label, x=_f(kv.get("X")), y=_f(kv.get("Y")), z=_f(kv.get("Z")))
                elif b == "RESTRAINT":
                    self._restraint(kv)
                elif b == "SPRING":
                    self._spring(kv)
                elif b in ("MASS", "MASSE"):
                    self._mass(kv)
                elif b == "MATERIAL":
                    if "NAME" in kv:
                        current = {"name": kv["NAME"]}
                        m.materials[kv["NAME"]] = Material(name=kv["NAME"], type="Steel" if kv.get("IDES", "").upper() == "S"
                                                           else "Concrete" if kv.get("IDES", "").upper() == "C" else "Other")
                    self._material(current.get("name"), kv)
                elif b == "FRAME SECTION":
                    self._frame_section(kv)
                elif b == "SHELL SECTION":
                    self._shell_section(kv)
                elif b == "FRAME" and label:
                    self._frame(label, kv)
                elif b == "SHELL" and label:
                    self._shell(label, kv)
                elif b == "LOAD":
                    if "NAME" in kv:
                        pattern = kv["NAME"]
                        m.load_patterns.setdefault(pattern, LoadPattern(pattern))
                        m.load_cases.setdefault(pattern, LinearStaticCase(name=pattern, loads={pattern: 1.0}))
                    if "TYPE" in kv:
                        load_type = kv["TYPE"].upper()
                    if "ADD" in kv and pattern:
                        self._load(pattern, load_type, body, kv)
                elif b == "COMBO":
                    self._combo(kv, current)
                elif b in ("MODE",):
                    n = int(_f(kv.get("N"), 12))
                    m.load_cases["MODAL"] = ModalCase(name="MODAL", max_modes=n,
                                                      method="Ritz" if kv.get("TYPE", "").upper() == "RITZ" else "Eigen")
                elif b:
                    self.skipped[f"{block} data"] += 1
            except (KeyError, ValueError) as exc:
                self.skipped[f"{block}: {exc}"] += 1
        self._finish()
        self.model.convert_area_loads()  # area loads -> area-load panels

    # ------------------------------------------------------------ blocks
    def _system(self, kv):
        if "LENGTH" in kv or "FORCE" in kv:
            try:
                self.model.units = UnitSystem.parse(
                    f"{kv.get('FORCE', 'kN')} {kv.get('LENGTH', 'm')} {kv.get('TEMP', kv.get('TEMPERATURE', 'C'))}")
            except ValueError:
                self.warnings.append(f"Unknown units {kv.get('FORCE')} {kv.get('LENGTH')}; assuming kN-m")
        if "DOF" in kv:
            dofs = [d.upper() for d in _list(kv["DOF"])]
            self.model.options.active_dofs = tuple(d in dofs for d in ("UX", "UY", "UZ", "RX", "RY", "RZ"))

    def _joints(self, kv):
        return [j for j in _range(kv.get("ADD", "")) if j in self.model.points]

    def _restraint(self, kv):
        dofs = [d.upper() for d in _list(kv.get("DOF", ""))]
        fix = tuple(any(d in dofs for d in pair) for pair in
                    (("U1", "UX"), ("U2", "UY"), ("U3", "UZ"), ("R1", "RX"), ("R2", "RY"), ("R3", "RZ")))
        for j in self._joints(kv):
            self.model.points[j].restraints = fix

    def _spring(self, kv):
        for j in self._joints(kv):
            p = self.model.points[j]
            p.springs = tuple(_f(kv.get(k, kv.get(a))) for k, a in (("U1", "UX"), ("U2", "UY"), ("U3", "UZ")))
            p.rotational_springs = tuple(_f(kv.get(k, kv.get(a))) for k, a in (("R1", "RX"), ("R2", "RY"), ("R3", "RZ")))

    def _mass(self, kv):
        for j in self._joints(kv):
            p = self.model.points[j]
            p.mass = tuple(_f(kv.get(k, kv.get(a))) for k, a in (("U1", "UX"), ("U2", "UY"), ("U3", "UZ")))
            p.rotational_mass = tuple(_f(kv.get(k, kv.get(a))) for k, a in (("R1", "RX"), ("R2", "RY"), ("R3", "RZ")))

    def _material(self, name, kv):
        mat = self.model.materials.get(name or "")
        if mat is None:
            return
        if "E" in kv:
            mat.E = _f(kv["E"])
        if "U" in kv:
            mat.nu = _f(kv["U"], 0.2)
        if "W" in kv:
            mat.weight_density = _f(kv["W"])
        if "M" in kv:
            mat.mass_density = _f(kv["M"])
        if "A" in kv:
            mat.alpha = _f(kv["A"])
        if "FY" in kv:
            mat.fy = _f(kv["FY"])
        if "FC" in kv:
            mat.fc = _f(kv["FC"])

    def _frame_section(self, kv):
        name = kv.get("NAME")
        if not name:
            return
        mat = kv.get("MAT", next(iter(self.model.materials), ""))
        sh = kv.get("SH", "").upper()
        t = [_f(x) for x in _list(kv.get("T", ""))]
        if "A" in kv:
            I = [_f(x) for x in _list(kv.get("I", "0,0"))] + [0.0, 0.0]
            AS = [_f(x) for x in _list(kv.get("AS", "0,0"))] + [0.0, 0.0]
            sec = FrameSection.general(name, mat, _f(kv["A"]), I[0], I[1], _f(kv.get("J")), AS[0], AS[1])
        elif sh in ("R",) and len(t) >= 2:
            sec = FrameSection.rectangular(name, mat, t[0], t[1])
        elif sh in ("C", "CIRCLE") and t:
            sec = FrameSection.circular(name, mat, t[0])
        elif sh == "I" and len(t) >= 4:
            sec = FrameSection.i_shape(name, mat, t[0], t[1], t[2], t[3])
        elif sh == "P" and len(t) >= 2:
            ro, ri = t[0] / 2, t[0] / 2 - t[1]
            A = math.pi * (ro ** 2 - ri ** 2)
            I = math.pi * (ro ** 4 - ri ** 4) / 4
            sec = FrameSection.general(name, mat, A, I, I, 2 * I, A / 2, A / 2)
        else:
            self.warnings.append(f"Frame section {name}: shape {sh or kv.get('FILE', '?')} not supported; properties zero")
            sec = FrameSection(name=name, material=mat)
        self.model.frame_sections[name] = sec

    def _shell_section(self, kv):
        name = kv.get("NAME")
        if not name:
            return
        th = [_f(x) for x in _list(kv.get("TH", "0"))]
        typ = kv.get("TYPE", "Shell").lower()
        mt = "Membrane" if "membrane" in typ else "Plate" if "plate" in typ else (
            "ShellThick" if "thick" in typ else "ShellThin")
        self.model.shell_sections[name] = ShellSection(
            name=name, material=kv.get("MAT", next(iter(self.model.materials), "")), model_type=mt,
            thickness=th[0] if th else 0.0, bending_thickness=th[1] if len(th) > 1 and th[1] != th[0] else None)

    def _frame(self, label, kv):
        j = _list(kv.get("J", ""))
        sec = kv.get("SEC")
        if len(j) < 2 or j[0] not in self.model.points or j[1] not in self.model.points or sec not in self.model.frame_sections:
            self.skipped["FRAME (undefined joints / section)"] += 1
            return
        a, b = self.model.points[j[0]], self.model.points[j[1]]
        horiz = math.hypot(b.x - a.x, b.y - a.y)
        typ = "Column" if horiz < 1e-9 else ("Beam" if abs(b.z - a.z) < 1e-9 else "Brace")
        fr = Frame(name=label, point_i=j[0], point_j=j[1], section=sec, type=typ, angle=_f(kv.get("ANG")))
        fr.auto_mesh = False
        fr.mesh_at_intersections = False
        idx = {"P": 0, "V2": 1, "V3": 2, "T": 3, "M2": 4, "M3": 5, "R2": 4, "R3": 5, "U1": 0, "U2": 1, "U3": 2, "R1": 3}
        for key, attr in (("IREL", "releases_i"), ("JREL", "releases_j")):
            if key in kv:
                rel = [False] * 6
                for c in _list(kv[key]):
                    if c.upper() in idx:
                        rel[idx[c.upper()]] = True
                setattr(fr, attr, tuple(rel))
        self.model.frames[label] = fr

    def _shell(self, label, kv):
        j = _list(kv.get("J", ""))
        sec = kv.get("SEC")
        if len(j) < 3 or any(x not in self.model.points for x in j) or sec not in self.model.shell_sections:
            self.skipped["SHELL (undefined joints / section)"] += 1
            return
        if len(j) == 4 and not _cyclic(*(self.model.points[x] for x in j)):
            j = [j[0], j[1], j[3], j[2]]  # joints given in "Z" order (j1, j4 diagonally opposite)
        zs = [self.model.points[x].z for x in j]
        typ = "Wall" if max(zs) - min(zs) > 1e-9 else "Floor"
        self.model.areas[label] = Area(name=label, points=j, section=sec, type=typ, auto_mesh=False)

    def _load(self, pattern, load_type, body, kv):
        m = self.model
        targets = _range(kv["ADD"])
        if "FORCE" in load_type:
            F = tuple(_f(kv.get(k, kv.get(a))) for k, a in (("UX", "U1"), ("UY", "U2"), ("UZ", "U3")))
            M = tuple(_f(kv.get(k, kv.get(a))) for k, a in (("RX", "R1"), ("RY", "R2"), ("RZ", "R3")))
            for j in targets:
                if j in m.points:
                    m.points[j].loads.append(PointLoad(pattern, F, M))
        elif "DISTRIBUTED" in load_type or "SPAN" in load_type or "UNIFORM" in load_type:
            rd = [_f(x) for x in _list(kv.get("RD", "0,1"))] + [1.0]
            for k, direction in (("UX", "X"), ("UY", "Y"), ("UZ", "Z")):
                if k in kv:
                    vals = [_f(x) for x in _list(kv[k])] or [0.0]
                    va, vb = vals[0], vals[1] if len(vals) > 1 else vals[0]
                    for f in targets:
                        if f in m.frames:
                            m.frames[f].loads.append(FrameLoad(pattern, "distributed", direction, "Force", rd[0], rd[1],
                                                               False, va, vb))
        elif "GRAVITY" in load_type:
            uz = _f(kv.get("UZ"), 0.0)
            if uz:
                m.load_patterns[pattern].self_weight_multiplier = -uz
                if kv["ADD"] != "*":
                    self.warnings.append(f"Load {pattern}: gravity load on selected elements applied to all elements")
        else:
            self.skipped[f"LOAD TYPE={load_type}"] += 1

    def _combo(self, kv, current):
        m = self.model
        if "NAME" in kv:
            current["combo"] = kv["NAME"]
            typ = kv.get("TYPE", "ADD").upper()
            m.load_combinations[kv["NAME"]] = LoadCombination(
                name=kv["NAME"], type={"ENVE": "Envelope", "ABS": "Absolute", "SRSS": "SRSS"}.get(typ, "LinearAdd"))
        name = current.get("combo")
        if name and "LOAD" in kv:
            m.load_combinations[name].items.append(ComboItem(kv["LOAD"], _f(kv.get("SF"), 1.0)))
        if name and "SPEC" in kv:
            self.skipped["combo: response spectrum terms"] += 1

    def _finish(self):
        m = self.model
        for c in m.load_combinations.values():
            c.items = [i for i in c.items if i.name in m.load_cases]
        if m.points:
            zs = sorted({round(p.z, 9) for p in m.points.values()})
            if len(zs) <= 200:
                m.set_z_grid([(f"Z{k + 1}", z) for k, z in enumerate(zs)])
        m.mass_sources["MsSrc1"].include_vertical = True
        m.mass_sources["MsSrc1"].lump_at_stories = False
        self.warnings.append("Legacy SAP2000 v6/v7 file: imported with the beta reader - please check the model")

    def report(self) -> str:
        lines = [f"Imported {self.model!r} (SAP2000 v6/v7 keyword file, beta)"]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {w}" for w in self.warnings]
        if self.skipped:
            lines.append("Skipped (not supported yet):")
            lines += [f"  - {k}: {v}" for k, v in sorted(self.skipped.items())]
        return "\n".join(lines)
