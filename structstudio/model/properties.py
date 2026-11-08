"""Material and section property definitions.

Naming follows the CSI database tables ("Material Properties", "Frame Section
Property Definitions", "Slab/Wall Property Definitions") so that models can be
exchanged with ETABS text files without translation tables.

All numeric values are in the owning :class:`~structstudio.model.Model`'s unit
system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .base import q


@dataclass
class Material:
    """Isotropic linear-elastic material.

    ``weight_density`` is force/volume, ``mass_density`` is force*s^2/length^4
    (i.e. mass per volume in the model's unit system).  When ``mass_density``
    is ``None`` it is derived from the weight density using standard gravity.
    """

    name: str
    type: str = "Concrete"  # Concrete | Steel | Rebar | Tendon | Aluminum | Masonry | Other
    E: float = q("stress", 0.0)
    nu: float = 0.2
    G: Optional[float] = q("stress", None)
    weight_density: float = q("weight_per_volume", 0.0)
    mass_density: Optional[float] = q("mass_per_volume", None)
    alpha: float = q("per_temperature", 0.0)  # coefficient of thermal expansion (1/deg)
    fc: Optional[float] = q("stress", None)  # concrete compressive strength
    fy: Optional[float] = q("stress", None)  # steel / rebar yield
    fu: Optional[float] = q("stress", None)
    damping: float = 0.0  # material modal damping ratio
    color: Optional[str] = None
    notes: str = ""

    @property
    def shear_modulus(self) -> float:
        if self.G is not None:
            return self.G
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass
class RebarSize:
    """A reinforcing bar size of the project (Define > Reinforcing Bar Sizes):
    nominal diameter and area, e.g. DB16 (TIS 24) 16 mm, 201.1 mm2."""

    name: str
    diameter: float = q("length", 0.0)
    area: float = q("area", 0.0)
    standard: str = ""  # catalog it came from (TIS 24, ASTM A615 ...) or "" for a user size
    notes: str = ""


@dataclass
class FrameModifiers:
    """Stiffness/mass/weight multipliers of a frame section or object."""

    A: float = 1.0
    As2: float = 1.0
    As3: float = 1.0
    J: float = 1.0
    I22: float = 1.0
    I33: float = 1.0
    mass: float = 1.0
    weight: float = 1.0

    def combined(self, other: "FrameModifiers | None") -> "FrameModifiers":
        if other is None:
            return self
        return FrameModifiers(
            self.A * other.A, self.As2 * other.As2, self.As3 * other.As3, self.J * other.J,
            self.I22 * other.I22, self.I33 * other.I33, self.mass * other.mass, self.weight * other.weight,
        )


@dataclass
class FrameSection:
    """Prismatic frame section.

    Geometric properties follow the CSI local axes: axis 1 along the member,
    ``I33`` is bending in the 1-2 plane (the *depth* ``t3`` contributes mainly
    to it), ``I22`` bending in the 1-3 plane.  ``As2``/``As3`` are effective
    shear areas; a value of ``0`` means shear deformation in that plane is
    neglected (CSI convention).
    """

    name: str
    material: str
    # General | Rectangular | Circular | I | Box | Pipe | Tee | Angle | Channel | DoubleAngle | DoubleChannel |
    # Lipped Channel | Concrete Tee | Concrete L | Concrete Box | Concrete Pipe | Section Designer
    # (see structstudio.model.section_shapes)
    shape: str = "General"
    t3: Optional[float] = q("length", None)  # depth (2-direction)
    t2: Optional[float] = q("length", None)  # width (3-direction)
    tf: Optional[float] = q("length", None)
    tw: Optional[float] = q("length", None)
    t2b: Optional[float] = q("length", None)  # bottom flange width for I
    tfb: Optional[float] = q("length", None)
    dis: Optional[float] = q("length", None)  # gap between the backs of a double angle / channel
    lip: Optional[float] = q("length", None)  # lip of a cold-formed lipped channel
    r: Optional[float] = q("length", None)  # fillet (root) radius / tube outside corner / inside bend radius
    r2: Optional[float] = q("length", None)  # flange tip / angle toe radius
    A: float = q("area", 0.0)
    J: float = q("inertia", 0.0)
    I33: float = q("inertia", 0.0)
    I22: float = q("inertia", 0.0)
    As2: float = q("area", 0.0)
    As3: float = q("area", 0.0)
    modifiers: FrameModifiers = field(default_factory=FrameModifiers)
    mass_per_length: float = q("mass_per_length", 0.0)  # additional
    weight_per_length: float = q("force_per_length", 0.0)  # additional
    design_type: str = ""  # Beam | Column | Brace | "" (auto)
    rebar: dict = field(default_factory=dict)  # reinforcement description (design module)
    color: Optional[str] = None
    notes: str = ""
    # nonprismatic (CSI "Nonprismatic" section): segments from end I, each a dict with
    # start / end (section names), length, length_type ("Absolute" | "Variable"),
    # ei33 / ei22 ("Linear" | "Parabolic" | "Cubic": EI, EI^1/2 or EI^1/3 vary linearly)
    nonprismatic: list = field(default_factory=list)
    params: dict = field(default_factory=dict)  # shape options without units, e.g. {"mirror": True}
    # Section Designer outline (x = local 3, y = local 2, lengths): polygons
    # {"kind": "polygon", "points": [[x, y], ...], "hole": bool} and circles
    # {"kind": "circle", "center": [x, y], "radius": r, "hole": bool}
    geometry: list = field(default_factory=list)
    source: str = ""  # section library the properties come from (e.g. "AISC15.xml")

    def convert_units(self, src, dst) -> "FrameSection":
        """Tagged fields as usual, plus the absolute segment lengths of a nonprismatic
        section and the Section Designer outline."""
        import dataclasses as _dc
        from .base import convert_tagged
        out = convert_tagged(self, src, dst)
        if self.geometry:
            r = src.factor("length") / dst.factor("length")
            geom = []
            for g in self.geometry:
                g = dict(g)
                if "points" in g:
                    g["points"] = [[float(x) * r, float(y) * r] for x, y in g["points"]]
                if "center" in g:
                    g["center"] = [float(c) * r for c in g["center"]]
                if g.get("radius") is not None:
                    g["radius"] = float(g["radius"]) * r
                if isinstance(g.get("dims"), dict):  # the dimensions a generated outline came from
                    g["dims"] = {k: (None if x is None else float(x) * r) for k, x in g["dims"].items()}
                geom.append(g)
            out = _dc.replace(out, geometry=geom)
        if self.nonprismatic:
            r = src.factor("length") / dst.factor("length")
            rows = []
            for row in self.nonprismatic:
                row = dict(row)
                if row.get("length_type") == "Absolute" and row.get("length") is not None:
                    row["length"] = row["length"] * r
                rows.append(row)
            out = _dc.replace(out, nonprismatic=rows)
        return out

    # ---- constructors for common shapes -----------------------------------
    @classmethod
    def rectangular(cls, name: str, material: str, depth: float, width: float, **kw) -> "FrameSection":
        d, b = float(depth), float(width)
        a, bb = max(d, b), min(d, b)
        # Roark/Young: torsion constant of a solid rectangle
        J = a * bb ** 3 * (1.0 / 3.0 - 0.21 * (bb / a) * (1.0 - bb ** 4 / (12.0 * a ** 4)))
        return cls(
            name=name, material=material, shape="Rectangular", t3=d, t2=b,
            A=b * d, J=J, I33=b * d ** 3 / 12.0, I22=d * b ** 3 / 12.0,
            As2=5.0 / 6.0 * b * d, As3=5.0 / 6.0 * b * d, **kw,
        )

    @classmethod
    def circular(cls, name: str, material: str, diameter: float, **kw) -> "FrameSection":
        r = float(diameter) / 2.0
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        return cls(name=name, material=material, shape="Circular", t3=diameter, t2=diameter,
                   A=A, J=2.0 * I, I33=I, I22=I, As2=0.9 * A, As3=0.9 * A, **kw)

    @classmethod
    def i_shape(cls, name: str, material: str, depth: float, flange_width: float,
                flange_thickness: float, web_thickness: float, **kw) -> "FrameSection":
        d, bf, tf, tw = map(float, (depth, flange_width, flange_thickness, web_thickness))
        hw = d - 2.0 * tf
        A = 2.0 * bf * tf + hw * tw
        I33 = bf * d ** 3 / 12.0 - (bf - tw) * hw ** 3 / 12.0
        I22 = 2.0 * tf * bf ** 3 / 12.0 + hw * tw ** 3 / 12.0
        J = (2.0 * bf * tf ** 3 + hw * tw ** 3) / 3.0
        return cls(name=name, material=material, shape="I", t3=d, t2=bf, tf=tf, tw=tw,
                   A=A, J=J, I33=I33, I22=I22, As2=tw * d, As3=5.0 / 3.0 * tf * bf, **kw)

    @classmethod
    def from_shape(cls, shape: str, name: str, material: str, mirror: bool = False, **dims) -> "FrameSection":
        """A parametric section of the catalogue (``section_shapes.SHAPES``) with
        its properties computed from the dimensions (t3, t2, tf, tw, t2b, tfb,
        dis, lip, the corner radii r, r2 and - I / channels - the inner flange
        ``slope``, tan of the taper angle, kept in ``params``)."""
        from .base import ModelError
        from .section_shapes import COLD, SHAPES, check_dims
        sh = SHAPES.get(shape)
        if sh is None:
            raise ModelError("UNKNOWN_SHAPE", f"Unknown section shape {shape!r}")
        errs = check_dims(sh, dims)
        if errs:
            raise ModelError("INVALID_SECTION", f"{name}: " + "; ".join(errs))
        kw = {k: dims.get(k) for k in ("t3", "t2", "tf", "tw", "t2b", "tfb", "dis", "lip", "r", "r2")}
        params = {"mirror": True} if mirror and sh.mirror else {}
        if dims.get("slope"):  # inner flange slope of a tapered I / channel (tan; unitless)
            params["slope"] = float(dims["slope"])
        if shape == "Circular" or shape.endswith("Pipe") or shape == "Pipe":
            kw["t2"] = kw["t3"]
        if shape in COLD:
            kw["tf"] = kw["tw"]
        sec = cls(name=name, material=material, shape=shape, params=params, **kw)
        sec.update_properties()
        return sec

    @classmethod
    def from_designer(cls, name: str, material: str, geometry: list, **kw) -> "FrameSection":
        """A Section Designer section: any polygons / circles (some of them
        openings), properties computed (J by finite elements)."""
        sec = cls(name=name, material=material, shape="Section Designer", geometry=[dict(g) for g in geometry], **kw)
        sec.update_properties()
        return sec

    def update_properties(self, torsion: bool = True, materials: Optional[dict] = None) -> bool:
        """Recompute A, I33, I22, J, As2, As3 (and the outline size t3 / t2) from
        the shape dimensions or the Section Designer outline.  False when the
        section has neither (General / library sections keep their values).
        A composite outline (shapes of other materials) gives transformed
        properties in this section's material; pass the model's materials (the
        modular ratios are kept in ``params["n"]``)."""
        from .section_shapes import section_properties, section_ratios
        ratios = section_ratios(self, materials)
        if ratios:
            self.params = dict(self.params or {}, n={k: list(v) for k, v in ratios.items()})
        p = section_properties(self, torsion, materials)
        if p is None:
            return False
        self.A, self.I33, self.I22, self.J, self.As2, self.As3 = p.A, p.I33, p.I22, p.J, p.As2, p.As3
        if self.geometry:
            self.t3, self.t2 = p.depth, p.width
        return True

    @classmethod
    def general(cls, name: str, material: str, A: float, I33: float, I22: float = 0.0, J: float = 0.0,
                As2: float = 0.0, As3: float = 0.0, **kw) -> "FrameSection":
        return cls(name=name, material=material, shape="General", A=A, J=J, I33=I33, I22=I22,
                   As2=As2, As3=As3, **kw)


@dataclass
class ShellModifiers:
    f11: float = 1.0
    f22: float = 1.0
    f12: float = 1.0
    m11: float = 1.0
    m22: float = 1.0
    m12: float = 1.0
    v13: float = 1.0
    v23: float = 1.0
    mass: float = 1.0
    weight: float = 1.0

    def combined(self, other: "ShellModifiers | None") -> "ShellModifiers":
        if other is None:
            return self
        return ShellModifiers(*(getattr(self, n) * getattr(other, n) for n in
                                ("f11", "f22", "f12", "m11", "m22", "m12", "v13", "v23", "mass", "weight")))


@dataclass
class ShellSection:
    """Slab, wall or deck property assigned to area objects."""

    name: str
    material: str
    type: str = "Slab"  # Slab | Wall | Deck
    model_type: str = "ShellThick"  # ShellThick (DKMQ, thin -> DKQ) | ShellThin (DKQ) | Membrane | Plate
    thickness: float = q("length", 0.0)  # membrane thickness
    bending_thickness: Optional[float] = q("length", None)  # defaults to thickness
    modifiers: ShellModifiers = field(default_factory=ShellModifiers)
    mass_per_area: float = q("mass_per_area", 0.0)  # additional
    weight_per_area: float = q("force_per_area", 0.0)  # additional
    color: Optional[str] = None
    notes: str = ""

    @property
    def bending_thickness_value(self) -> float:
        return self.thickness if self.bending_thickness is None else self.bending_thickness
