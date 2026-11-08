"""Unit systems.

Everything inside the engine is stored in SI base units: metre, newton,
kilogram, second (rotations in radians).  A :class:`UnitSystem` describes the
*presentation* units a model was defined in (e.g. ETABS "KIP IN F") and gives
conversion factors for every physical quantity the model uses.

Only force/length/temperature are chosen by the user; derived quantities
(stress, moment, distributed load, density ...) follow from them.  Mass is
derived as force*s^2/length, exactly as in CSI products, so that a model
defined in kip-in has mass in kip-s^2/in.
"""
from __future__ import annotations

from dataclasses import dataclass

# metres per unit length
_LENGTH = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "in": 0.0254,
    "ft": 0.3048,
}

# newtons per unit force
_FORCE = {
    "N": 1.0,
    "kN": 1.0e3,
    "kgf": 9.80665,
    "tonf": 9806.65,
    "lb": 4.4482216152605,
    "kip": 4448.2216152605,
}

# kelvin-sized degrees per unit temperature *difference* (temperature loads are
# changes, never absolute temperatures, so there is no offset)
_TEMPERATURE = {
    "C": 1.0,
    "F": 5.0 / 9.0,
}

# Quantity -> (force exponent, length exponent[, temperature exponent]).
# Seconds never need scaling.
_QUANTITY_EXPONENTS = {
    "length": (0, 1),
    "area": (0, 2),
    "inertia": (0, 4),  # second moment of area (also torsion constant)
    "force": (1, 0),
    "moment": (1, 1),
    "stress": (1, -2),  # also modulus of elasticity, pressure
    "force_per_length": (1, -1),  # distributed line load, line spring
    "force_per_area": (1, -2),  # area load, pressure
    "weight_per_volume": (1, -3),  # weight density
    "mass": (1, -1),  # force*s^2/length
    "mass_per_length": (1, -2),
    "mass_per_area": (1, -3),
    "mass_per_volume": (1, -4),
    "rotational_mass": (1, 1),  # force*s^2*length
    "moment_per_length": (1, 0),
    "translational_stiffness": (1, -1),
    "rotational_stiffness": (1, 1),
    "acceleration": (0, 1),
    "dimensionless": (0, 0),
    "flexural_rigidity": (1, 2),  # EI, GJ (force*length^2)
    "inverse_length": (0, -1),  # tendon wobble coefficient, curvature
    "temperature_change": (0, 0, 1),  # temperature loads
    "temperature_gradient": (0, -1, 1),  # temperature change per length across a section
    "per_temperature": (0, 0, -1),  # coefficient of thermal expansion
}


@dataclass(frozen=True)
class UnitSystem:
    """Force and length units a model is expressed in."""

    force: str = "kN"
    length: str = "m"
    temperature: str = "C"

    def __post_init__(self):
        if self.force not in _FORCE:
            raise ValueError(f"Unknown force unit {self.force!r}; choose from {sorted(_FORCE)}")
        if self.length not in _LENGTH:
            raise ValueError(f"Unknown length unit {self.length!r}; choose from {sorted(_LENGTH)}")
        t = str(self.temperature or "C").strip().upper().lstrip("°")
        t = {"CELSIUS": "C", "FAHRENHEIT": "F", "K": "C", "KELVIN": "C"}.get(t, t)  # kelvin-sized = C
        object.__setattr__(self, "temperature", t)
        if self.temperature not in _TEMPERATURE:
            raise ValueError(f"Unknown temperature unit {self.temperature!r}; choose from {sorted(_TEMPERATURE)}")

    @property
    def force_factor(self) -> float:
        return _FORCE[self.force]

    @property
    def length_factor(self) -> float:
        return _LENGTH[self.length]

    @property
    def temperature_factor(self) -> float:
        return _TEMPERATURE[self.temperature]

    def factor(self, quantity: str) -> float:
        """Multiply a value in this unit system by this to obtain SI."""
        try:
            exps = _QUANTITY_EXPONENTS[quantity]
        except KeyError as exc:
            raise KeyError(f"Unknown quantity {quantity!r}; choose from {sorted(_QUANTITY_EXPONENTS)}") from exc
        fe, le = exps[0], exps[1]
        te = exps[2] if len(exps) > 2 else 0
        return self.force_factor ** fe * self.length_factor ** le * self.temperature_factor ** te

    def to_si(self, value, quantity: str):
        return value * self.factor(quantity)

    def from_si(self, value, quantity: str):
        return value / self.factor(quantity)

    @property
    def gravity(self) -> float:
        """Standard gravity expressed in this unit system (length/s^2)."""
        return 9.80665 / self.length_factor

    def label(self, quantity: str) -> str:
        exps = _QUANTITY_EXPONENTS[quantity]
        fe, le = exps[0], exps[1]
        te = exps[2] if len(exps) > 2 else 0
        if te:  # C, C/m, 1/C
            t = self.temperature
            if quantity == "per_temperature":
                return f"1/{t}"
            return t if not le else f"{t}/{self.length}"
        parts = []
        if fe:
            parts.append(self.force if fe == 1 else f"{self.force}^{fe}")
        if quantity in ("mass", "mass_per_length", "mass_per_area", "mass_per_volume", "rotational_mass"):
            parts.append("s^2")
        if le:
            if le == 1:
                parts.append(self.length)
            elif le > 1:
                parts.append(f"{self.length}^{le}")
            else:
                parts.append(f"/{self.length}" if le == -1 else f"/{self.length}^{-le}")
        if not parts:
            return "-"
        out = parts[0]
        for part in parts[1:]:  # "kN/m", "kN-m", "kN-s^2/m"
            out += part if part.startswith("/") else "-" + part
        return out

    @classmethod
    def parse(cls, text: str) -> "UnitSystem":
        """Parse strings like ``"kN-m"``, ``"kip in"``, ``"KIP,IN,F"``."""
        tokens = [t for t in text.replace(",", " ").replace("-", " ").split() if t]
        if len(tokens) < 2:
            raise ValueError(f"Cannot parse unit system from {text!r}")
        force = _normalise_force(tokens[0])
        length = _normalise_length(tokens[1])
        temp = tokens[2].upper().lstrip("°") if len(tokens) > 2 else "C"
        return cls(force, length, temp)

    def with_(self, force: str = None, length: str = None, temperature: str = None) -> "UnitSystem":
        return UnitSystem(force or self.force, length or self.length, temperature or self.temperature)

    @property
    def text(self) -> str:
        """``"kN, m, C"`` - as the units button of ETABS / SAP2000 shows it."""
        return f"{self.force}, {self.length}, {self.temperature}"


def _normalise_force(token: str) -> str:
    t = token.strip()
    lookup = {k.lower(): k for k in _FORCE}
    lookup.update({"kips": "kip", "lbs": "lb", "kgf": "kgf", "tf": "tonf", "ton": "tonf", "tonf": "tonf"})
    try:
        return lookup[t.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown force unit {token!r}") from exc


def _normalise_length(token: str) -> str:
    t = token.strip()
    lookup = {k.lower(): k for k in _LENGTH}
    lookup.update({"inch": "in", "inches": "in", "feet": "ft", "foot": "ft", "metre": "m", "meter": "m"})
    try:
        return lookup[t.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown length unit {token!r}") from exc


# consistent unit sets offered by the units button / form (as ETABS / SAP2000 list them)
PRESETS = (
    ("N", "mm", "C"), ("N", "m", "C"), ("kN", "mm", "C"), ("kN", "m", "C"),
    ("kgf", "mm", "C"), ("kgf", "cm", "C"), ("kgf", "m", "C"), ("tonf", "mm", "C"), ("tonf", "cm", "C"),
    ("tonf", "m", "C"), ("lb", "in", "F"), ("lb", "ft", "F"), ("kip", "in", "F"), ("kip", "ft", "F"),
)
FORCE_UNITS = tuple(_FORCE)
LENGTH_UNITS = tuple(_LENGTH)
TEMPERATURE_UNITS = tuple(_TEMPERATURE)
QUANTITIES = tuple(_QUANTITY_EXPONENTS)

SI = UnitSystem("N", "m")
KN_M = UnitSystem("kN", "m")
KIP_IN = UnitSystem("kip", "in")
KIP_FT = UnitSystem("kip", "ft")
