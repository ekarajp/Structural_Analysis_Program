"""Reader for CSI "Frame Properties" XML section databases (AISC15.xml, Euro.xml ...).

Only the geometric analysis properties are read (A, I33, I22, J, As2, As3 and
the outline dimensions).  The catalog data itself (AISC, Eurocode, JIS ...)
is public; this module just understands the container format so that a
model exported from ETABS keeps the exact tabulated properties.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

from ..model.base import ModelError
from ..model.properties import FrameSection
from ..units import UnitSystem

_NS = "{http://www.csiberkeley.com}"
_SHAPE_MAP = {
    "STEEL_I_SECTION": "I", "STEEL_CHANNEL": "Channel", "STEEL_ANGLE": "Angle", "STEEL_DOUBLE_ANGLE": "DoubleAngle",
    "STEEL_TEE": "Tee", "STEEL_BOX": "Box", "STEEL_PIPE": "Pipe", "CONCRETE_RECTANGLE": "Rectangular",
    "CONCRETE_CIRCLE": "Circular", "STEEL_ROD": "Circular", "STEEL_PLATE": "Rectangular",
}
_FIELDS = ("D", "B", "BF", "TF", "TW", "A", "I33", "I22", "J", "AS2", "AS3", "T", "OD", "TDES", "DIS")


@dataclass
class DbSection:
    label: str
    shape: str
    values: Dict[str, float]  # in the database length units
    units: str  # length unit of the file

    def to_frame_section(self, name: str, material: str, model_units: UnitSystem) -> FrameSection:
        """Convert to a FrameSection in the model's units."""
        src = UnitSystem("N", self.units)
        L = src.factor("length") / model_units.factor("length")
        v = self.values
        get = lambda k: v.get(k, 0.0)
        sec = FrameSection(
            name=name, material=material, shape=self.shape,
            t3=get("D") * L if "D" in v else (get("OD") * L if "OD" in v else None),
            t2=(v.get("BF", v.get("B", v.get("OD", 0.0)))) * L if any(k in v for k in ("BF", "B", "OD")) else None,
            tf=get("TF") * L if "TF" in v else None, tw=get("TW") * L if "TW" in v else None,
            A=get("A") * L ** 2, J=get("J") * L ** 4, I33=get("I33") * L ** 4, I22=get("I22") * L ** 4,
            As2=get("AS2") * L ** 2, As3=get("AS3") * L ** 2,
        )
        if self.shape == "DoubleAngle" and sec.t2:  # CSI: B is the overall width of the pair
            gap = get("DIS") * L
            sec.dis = gap
            sec.t2 = max((sec.t2 - gap) / 2.0, 0.0) or sec.t2
        if self.shape == "Pipe" and sec.tw is None and ("TDES" in v or "T" in v):
            sec.tw = (get("TDES") or get("T")) * L
        if self.shape == "Box" and "T" in v and sec.tw is None:
            sec.tw = sec.tf = (get("TDES") or get("T")) * L
        return sec


class SectionDatabase:
    def __init__(self, path: str):
        self.path = path
        self.sections: Dict[str, DbSection] = {}
        self.units = "in"
        self._load()

    def _load(self):
        tree = ET.parse(self.path)
        root = tree.getroot()
        ctrl = root.find(_NS + "CONTROL")
        if ctrl is not None:
            lu = ctrl.find(_NS + "LENGTH_UNITS")
            if lu is not None and lu.text:
                self.units = lu.text.strip().lower()
        for el in root:
            tag = el.tag.replace(_NS, "")
            label_el = el.find(_NS + "LABEL")
            if label_el is None or not label_el.text:
                continue
            values = {}
            for f in _FIELDS:
                node = el.find(_NS + f)
                if node is not None and node.text:
                    try:
                        values[f] = float(node.text)
                    except ValueError:
                        pass
            if "A" not in values:
                continue
            label = label_el.text.strip()
            # first definition wins (files repeat labels for metric/imperial variants)
            self.sections.setdefault(label.upper(), DbSection(label, _SHAPE_MAP.get(tag, tag), values, self.units))

    def get(self, label: str) -> Optional[DbSection]:
        return self.sections.get(label.upper())

    def __len__(self):
        return len(self.sections)


class SectionLibrary:
    """Lazy collection of section databases found on search paths."""

    def __init__(self, search_paths: Optional[List[str]] = None):
        self.search_paths = list(DEFAULT_SEARCH_PATHS if search_paths is None else search_paths)
        self._dbs: Dict[str, SectionDatabase] = {}

    def available(self) -> List[str]:
        """File names of the section databases found on the search paths."""
        out = []
        for p in self.search_paths:
            if os.path.isdir(p):
                out += [fn for fn in sorted(os.listdir(p)) if fn.lower().endswith(".xml")]
        return sorted(set(out))

    def names(self, file_name: str) -> List[str]:
        db = self.database(file_name)
        return sorted(sec.label for sec in db.sections.values()) if db else []

    def frame_section(self, file_name: str, label: str, material: str, model_units: UnitSystem,
                      name: Optional[str] = None) -> FrameSection:
        sec = self.find(label, file_name)
        if sec is None:
            raise ModelError("UNKNOWN_SECTION", f"{label!r} not found in {file_name}")
        out = sec.to_frame_section(name or label, material, model_units)
        out.source = os.path.basename(file_name)
        return out

    def database(self, file_name: str) -> Optional[SectionDatabase]:
        key = os.path.basename(file_name).lower()
        if key in self._dbs:
            return self._dbs[key]
        candidates = [file_name] + [os.path.join(p, os.path.basename(file_name)) for p in self.search_paths]
        for c in candidates:
            if os.path.isfile(c):
                self._dbs[key] = SectionDatabase(c)
                return self._dbs[key]
        return None

    def find(self, label: str, file_name: Optional[str] = None) -> Optional[DbSection]:
        if file_name:
            db = self.database(file_name)
            if db:
                hit = db.get(label)
                if hit:
                    return hit
        for p in self.search_paths:
            if not os.path.isdir(p):
                continue
            for fn in sorted(os.listdir(p)):
                if fn.lower().endswith(".xml"):
                    db = self.database(os.path.join(p, fn))
                    if db:
                        hit = db.get(label)
                        if hit:
                            return hit
        return None


DEFAULT_SEARCH_PATHS = [p for p in os.environ.get("STRUCTSTUDIO_SECTION_PATH", "").split(os.pathsep) if p] + [
    r"E:\REA ETABS\reference\etabs21\Property Libraries",
    r"C:\Program Files\Computers and Structures\ETABS 21\Property Libraries",
]
