"""Built-in catalogs: standard steel sections and reinforcing bar sizes.

The tables are CSV files in ``data/`` (built by ``scripts/build_catalog.py``
from the published standards; sources and licences in ``data/SOURCES.md``).
A user can import a file of the same columns, so a supplier's table or an
office list is read the same way.

Steel sections: dimensions in mm, A in cm2, I in cm4, mass in kg/m.  The
tabulated A, I33, I22 (and J where the table gives it) are used in the
analysis; the dimensions draw the section and give the shear areas, moduli
and - when not tabulated - J (fillets included).
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from ..units import UnitSystem

DATA = os.path.join(os.path.dirname(__file__), "data")
STEEL_FILES = ("steel_tis.csv", "steel_gb.csv", "steel_aisc.csv", "steel_en.csv")
STEEL_COLUMNS = ("standard", "series", "name", "shape", "t3", "t2", "tf", "tw", "r", "r2", "lip", "t2b",
                 "A", "I33", "I22", "J", "mass", "slope")
REBAR_COLUMNS = ("standard", "name", "diameter", "area", "mass")
DIMS = ("t3", "t2", "tf", "tw", "r", "r2", "lip", "t2b")
_MM = UnitSystem("N", "mm")


class CatalogError(ValueError):
    pass


@dataclass
class SteelEntry:
    standard: str
    series: str
    name: str
    shape: str
    dims: Dict[str, Optional[float]]  # mm
    A: float  # cm2
    I33: float  # cm4
    I22: float  # cm4
    J: Optional[float] = None  # cm4 (None: computed from the shape)
    mass: Optional[float] = None  # kg/m
    slope: float = 0.0  # inner flange slope of a tapered I / channel (tan); tf is then the mean thickness

    def frame_section(self, material: str, units: UnitSystem, name: Optional[str] = None):
        """A frame section in the model's units: the shape drawn from the
        dimensions, the tabulated properties used in the analysis."""
        from ..model.properties import FrameSection
        k = _MM.factor("length") / units.factor("length")
        dims = {d: (None if v is None else v * k) for d, v in self.dims.items()}
        if self.slope:
            dims["slope"] = self.slope
        sec = FrameSection.from_shape(self.shape, name or self.name, material, **dims)
        a, i = (k * 10.0) ** 2, (k * 10.0) ** 4  # cm2, cm4 -> model units
        sec.A, sec.I33, sec.I22 = self.A * a, self.I33 * i, self.I22 * i
        if self.J:
            sec.J = self.J * i
        sec.source = self.standard
        return sec


@dataclass
class RebarEntry:
    standard: str
    name: str
    diameter: float  # mm
    area: float  # mm2
    mass: Optional[float] = None  # kg/m

    def rebar_size(self, units: UnitSystem):
        from ..model.properties import RebarSize
        k = _MM.factor("length") / units.factor("length")
        return RebarSize(name=self.name, diameter=self.diameter * k, area=self.area * k * k, standard=self.standard)


def _f(v) -> Optional[float]:
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        raise CatalogError(f"not a number: {v!r}")


def _rows(text: str, required) -> List[dict]:
    rd = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    cols = [c.strip() for c in (rd.fieldnames or [])]
    missing = [c for c in required if c not in cols]
    if missing:
        raise CatalogError("missing column(s): " + ", ".join(missing))
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rd]


def read_steel(text: str, default_standard: str = "Imported") -> List[SteelEntry]:
    """Steel sections from CSV text (the columns of STEEL_COLUMNS; series, r,
    r2, lip, J and mass may be left out)."""
    from ..model.section_shapes import COLD, SHAPES, check_dims
    out = []
    for n, r in enumerate(_rows(text, ("name", "shape", "t3", "A", "I33", "I22")), start=2):
        shape = r.get("shape", "")
        if shape not in SHAPES:
            raise CatalogError(f"line {n}: unknown shape {shape!r} (use one of {', '.join(SHAPES)})")
        dims = {d: _f(r.get(d)) for d in DIMS}
        if shape == "Pipe" or shape == "Circular":
            dims["t2"] = dims["t3"]
        if shape in COLD:
            dims["tf"] = dims["tw"]
        slope = _f(r.get("slope")) or 0.0
        errs = check_dims(SHAPES[shape], dict(dims, slope=slope))
        if errs:
            raise CatalogError(f"line {n} ({r.get('name')}): " + "; ".join(errs))
        out.append(SteelEntry(r.get("standard") or default_standard, r.get("series") or SHAPES[shape].label,
                              r["name"], shape, dims, _f(r["A"]), _f(r["I33"]), _f(r["I22"]), _f(r.get("J")),
                              _f(r.get("mass")), slope))
    return out


def read_rebar(text: str, default_standard: str = "Imported") -> List[RebarEntry]:
    """Rebar sizes from CSV text: name, diameter (mm) and optionally area (mm2,
    default pi d^2 / 4), mass (kg/m) and standard."""
    import math
    out = []
    for n, r in enumerate(_rows(text, ("name", "diameter")), start=2):
        d = _f(r["diameter"])
        if not d or d <= 0:
            raise CatalogError(f"line {n}: diameter must be > 0")
        a = _f(r.get("area")) or math.pi * d * d / 4.0
        out.append(RebarEntry(r.get("standard") or default_standard, r["name"], d, a, _f(r.get("mass"))))
    return out


def _read(fn: str) -> str:
    with open(os.path.join(DATA, fn), encoding="utf-8") as fh:
        return fh.read()


@lru_cache(maxsize=None)
def steel_catalog() -> Dict[str, List[SteelEntry]]:
    """Built-in steel sections by standard (in file order)."""
    out: Dict[str, List[SteelEntry]] = {}
    for fn in STEEL_FILES:
        for e in read_steel(_read(fn)):
            out.setdefault(e.standard, []).append(e)
    return out


@lru_cache(maxsize=None)
def rebar_catalog() -> Dict[str, List[RebarEntry]]:
    """Built-in rebar sizes by standard."""
    out: Dict[str, List[RebarEntry]] = {}
    for e in read_rebar(_read("rebar.csv")):
        out.setdefault(e.standard, []).append(e)
    return out


def find_steel(name: str, standard: Optional[str] = None) -> Optional[SteelEntry]:
    key = name.replace(" ", "").upper()
    for std, entries in steel_catalog().items():
        if standard and std != standard:
            continue
        for e in entries:
            if e.name.replace(" ", "").upper() == key:
                return e
    return None


def steel_csv(entries: List[SteelEntry]) -> str:
    """CSV text of steel entries (the import format)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(STEEL_COLUMNS)
    for e in entries:
        w.writerow([e.standard, e.series, e.name, e.shape] + ["" if e.dims.get(d) is None else f"{e.dims[d]:g}"
                                                              for d in DIMS] +
                   [f"{e.A:g}", f"{e.I33:g}", f"{e.I22:g}", "" if e.J is None else f"{e.J:g}",
                    "" if e.mass is None else f"{e.mass:g}", f"{e.slope:.6g}" if e.slope else ""])
    return buf.getvalue()


def rebar_csv(sizes, units: UnitSystem) -> str:
    """CSV text of rebar sizes (model RebarSize objects) in the import format."""
    k = units.factor("length") / _MM.factor("length")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(REBAR_COLUMNS)
    for s in sizes:
        w.writerow([s.standard, s.name, f"{s.diameter * k:g}", f"{s.area * k * k:g}", f"{s.area * k * k * 7.85e-3:.4g}"])
    return buf.getvalue()
