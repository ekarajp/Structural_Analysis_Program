"""Display units per kind of value (the item-by-item table of the ETABS / SAP2000
"Display Units" form).

The model is stored in one consistent set of units (:class:`~structstudio.units.UnitSystem`).
On top of it every *kind of value* can be shown and typed in its own unit - e.g.
the Thai legacy practice: lengths in m, section dimensions in cm, steel and rebar
areas in cm^2, deflections in mm, strengths and moduli in ksc (kgf/cm^2), and
every force in kgf (kgf, kgf-m, kgf/m, kgf/m^2) or every force in tonf (tonf,
tonf-m, tonf/m, tonf/m^2).  Forces are always labelled kgf / tonf (kilogram-force,
tonne-force), never kg / ton, which are masses.  Nothing in the model changes when they
are changed; only what is shown / typed is converted (through SI).

Unit labels are parsed (``"tonf-m"``, ``"kgf/m^2"``, ``"ksc"``, ``"MPa"``,
``"kip-ft"`` ...), so any consistent combination works.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .units import UnitSystem

# SI value of one unit of each token (forces in N, lengths in m, stresses in Pa)
_TOKENS = {
    "N": 1.0, "kN": 1.0e3, "MN": 1.0e6, "kgf": 9.80665, "kg": 9.80665, "tonf": 9806.65, "ton": 9806.65, "t": 9806.65,
    "lb": 4.4482216152605, "lbf": 4.4482216152605, "kip": 4448.2216152605,
    "m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048,
    "Pa": 1.0, "kPa": 1.0e3, "MPa": 1.0e6, "GPa": 1.0e9, "ksc": 98066.5,
    "psi": 6894.757293168, "ksi": 6894757.293168, "psf": 47.88025898, "ksf": 47880.25898,
    "pcf": 157.0874606, "kcf": 157087.4606, "rad": 1.0, "s": 1.0,
}
_ALIASES = {"N/mm^2": "MPa", "kN/m^2": "kPa"}
# force written as a mass (older files, typed by hand): kg -> kgf, ton / t -> tonf
_FORCE_TOKENS = {"kg": "kgf", "ton": "tonf", "t": "tonf"}


def _plain(label) -> str:
    """Superscript powers (shown labels, m²) back to ``^n``."""
    inv = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+"))
    return re.sub(r"[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: "^" + "".join(inv[c] for c in m.group(0)), str(label))


def _pretty(label: str) -> str:
    """``m^2`` -> ``m²`` (how a unit is printed)."""
    sup = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
    return re.sub(r"\^(-?\d+)", lambda m: m.group(1).translate(sup), str(label))


def force_label(label: str) -> str:
    """``label`` with any force written as a mass made explicit: ``"kg/m^2"`` ->
    ``"kgf/m^2"``, ``"ton-m"`` -> ``"tonf-m"`` (everything else unchanged)."""
    label = _plain(label)
    return re.sub(r"[A-Za-z]+", lambda m: _FORCE_TOKENS.get(m.group(0), m.group(0)), str(label))


def unit_si(label: str) -> float:
    """SI value of one ``label`` (``"tonf-m"`` -> 9806.65, ``"ksc"`` -> 98066.5)."""
    text = re.sub(r"\(.*?\)", "", _plain(label)).strip()
    if text in _TOKENS:
        return _TOKENS[text]
    parts = text.split("/")
    value = 1.0
    for k, part in enumerate(parts):
        for tok in re.split(r"[-·*\s]+", part.strip()):
            if not tok:
                continue
            m = re.fullmatch(r"([A-Za-z]+)\^?(\d*)", tok)
            if m is None or m.group(1) not in _TOKENS:
                raise ValueError(f"Unknown unit {tok!r} in {label!r}")
            v = _TOKENS[m.group(1)] ** int(m.group(2) or 1)
            value = value * v if k == 0 else value / v
    return value


# the items of the form: (key, label, physical quantity)
ITEMS: List[Tuple[str, str, str]] = [
    ("coordinates", "Structure dimensions, coordinates, offsets", "length"),
    ("section_dim", "Section dimensions, slab / wall thickness", "length"),
    ("section_area", "Section and rebar areas", "area"),
    ("section_inertia", "Section moments of inertia, torsion constant", "inertia"),
    ("displacement", "Displacements, deflections", "length"),
    ("force", "Forces, reactions, point loads", "force"),
    ("moment", "Moments", "moment"),
    ("line_load", "Line loads (per length)", "force_per_length"),
    ("area_load", "Area loads, pressures", "force_per_area"),
    ("stress", "Stresses, strengths, E, G (fc', fy)", "stress"),
    ("unit_weight", "Unit weight", "weight_per_volume"),
]
ITEM_QUANTITY = {k: q for k, _, q in ITEMS}

CHOICES: Dict[str, List[str]] = {
    "length": ["m", "cm", "mm", "in", "ft"],
    "area": ["m^2", "cm^2", "mm^2", "in^2", "ft^2"],
    "inertia": ["m^4", "cm^4", "mm^4", "in^4", "ft^4"],
    "force": ["N", "kN", "kgf", "tonf", "lb", "kip"],
    "moment": ["N-m", "N-mm", "kN-m", "kN-mm", "kgf-m", "kgf-cm", "tonf-m", "lb-in", "lb-ft", "kip-in", "kip-ft"],
    "force_per_length": ["N/m", "kN/m", "N/mm", "kgf/m", "tonf/m", "lb/ft", "kip/ft", "kip/in"],
    "force_per_area": ["N/m^2", "kN/m^2", "kgf/m^2", "tonf/m^2", "psf", "ksf"],
    "stress": ["MPa", "kN/m^2", "N/mm^2", "ksc", "tonf/m^2", "psi", "ksi"],
    "weight_per_volume": ["kN/m^3", "kgf/m^3", "tonf/m^3", "pcf", "kcf"],
}

# preset item sets; None = the model's consistent unit for that quantity
PRESETS: Dict[str, Tuple[Optional[Tuple[str, str, str]], Dict[str, str]]] = {
    "Consistent (model units for everything)": (None, {}),
    "Metric SI (kN, m, mm, MPa)": (("kN", "m", "C"), {
        "coordinates": "m", "section_dim": "mm", "section_area": "mm^2", "section_inertia": "mm^4",
        "displacement": "mm", "force": "kN", "moment": "kN-m", "line_load": "kN/m", "area_load": "kN/m^2",
        "stress": "MPa", "unit_weight": "kN/m^3"}),
    # Thai legacy practice, in two sets: every force in kgf, or every force in tonf;
    # lengths m, sections cm / cm^2 / cm^4, deflections mm, strengths ksc in both
    "Thai legacy - kgf (kgf, kgf-m, kgf/m, kgf/m^2, ksc)": (("kgf", "m", "C"), {
        "coordinates": "m", "section_dim": "cm", "section_area": "cm^2", "section_inertia": "cm^4",
        "displacement": "mm", "force": "kgf", "moment": "kgf-m", "line_load": "kgf/m", "area_load": "kgf/m^2",
        "stress": "ksc", "unit_weight": "kgf/m^3"}),
    "Thai legacy - tonf (tonf, tonf-m, tonf/m, tonf/m^2, ksc)": (("tonf", "m", "C"), {
        "coordinates": "m", "section_dim": "cm", "section_area": "cm^2", "section_inertia": "cm^4",
        "displacement": "mm", "force": "tonf", "moment": "tonf-m", "line_load": "tonf/m", "area_load": "tonf/m^2",
        "stress": "ksc", "unit_weight": "tonf/m^3"}),
    "U.S. (kip, ft, in, ksi)": (("kip", "ft", "F"), {
        "coordinates": "ft", "section_dim": "in", "section_area": "in^2", "section_inertia": "in^4",
        "displacement": "in", "force": "kip", "moment": "kip-ft", "line_load": "kip/ft", "area_load": "psf",
        "stress": "ksi", "unit_weight": "pcf"}),
}

# which item a tagged model field belongs to (owner class name, field) -> item;
# otherwise by quantity (QUANTITY_ITEM); None = shown in the model's units
_SECTION_OWNERS = {"FrameSection", "ShellSection", "FiberSectionDef"}
QUANTITY_ITEM = {"length": "coordinates", "area": "section_area", "inertia": "section_inertia", "force": "force",
                 "moment": "moment", "force_per_length": "line_load", "force_per_area": "area_load",
                 "stress": "stress", "weight_per_volume": "unit_weight"}
_NOT_LOADS = {("Frame", "foundation"), ("LineSpring", "stiffness"), ("Area", "spring")}  # springs: model units


def item_of(quantity: Optional[str], owner: Optional[str] = None, field: Optional[str] = None) -> Optional[str]:
    """The display item of a value of ``quantity`` (in field ``owner.field``)."""
    if quantity is None or (owner, field) in _NOT_LOADS:
        return None
    if quantity == "length" and owner in _SECTION_OWNERS:
        return "section_dim"
    return QUANTITY_ITEM.get(quantity)


class DisplayUnits:
    """The units things are shown / typed in: the model's consistent units with
    the per-item choices of ``model.display_units`` on top."""

    def __init__(self, units: UnitSystem, chosen: Optional[Dict[str, str]] = None):
        self.units = units
        self.chosen = {k: force_label(v) for k, v in (chosen or {}).items() if v}

    @classmethod
    def of(cls, model) -> "DisplayUnits":
        return cls(model.units, getattr(model, "display_units", None))

    def label(self, item: Optional[str], quantity: Optional[str] = None) -> str:
        """The unit as it is printed (powers as superscripts: m², cm⁴)."""
        if item in self.chosen:
            return _pretty(self.chosen[item])
        q = ITEM_QUANTITY.get(item, quantity)
        return _pretty(self.units.label(q)) if q else "-"

    def factor(self, item: Optional[str], quantity: Optional[str] = None) -> float:
        """Multiply a value in the model's units by this to get the displayed value."""
        q = ITEM_QUANTITY.get(item, quantity)
        if not q:
            return 1.0
        if item in self.chosen:
            return self.units.factor(q) / unit_si(self.chosen[item])
        return 1.0

    def show(self, value, item, quantity=None):
        return None if value is None else value * self.factor(item, quantity)

    def parse(self, value, item, quantity=None):
        return None if value is None else value / self.factor(item, quantity)

    # results: SI values from the analysis (per kind)
    def si_factor(self, item: str) -> float:
        """Multiply an SI value of ``item`` by this to get the displayed value."""
        q = ITEM_QUANTITY[item]
        return 1.0 / (unit_si(self.chosen[item]) if item in self.chosen else self.units.factor(q))

    def per_length(self, item: str) -> Tuple[str, float]:
        """Shell resultants (F11 per length from ``force``, M11 per length from
        ``moment``): (label, factor from model units)."""
        q = {"force": "force_per_length", "moment": "moment_per_length"}[item]
        if item not in self.chosen and "coordinates" not in self.chosen:
            return self.units.label(q) if item == "force" else f"{self.units.label('moment')}/{self.units.length}", 1.0
        num = self.label(item)
        den = self.label("coordinates")
        f = self.units.factor(q) / (unit_si(num) / unit_si(den))
        return f"{num}/{den}", f

    def summary(self) -> List[Tuple[str, str]]:
        return [(label, self.label(key)) for key, label, _q in ITEMS]
