"""Model templates for the New Model wizard (grid + stories + default properties)."""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..model import FIXED, FrameSection, Model, ShellSection
from ..model.objects import PINNED
from ..model.objects import GridLine, GridSystem


def _letters(n: int):
    out = []
    for i in range(n):
        s, k = "", i
        while True:
            s = chr(ord("A") + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        out.append(s)
    return out


def default_properties(model: Model) -> None:
    """Concrete material and a few RC sections, in the model's units: metric sizes
    (C28, 30x60 cm ...) for metric lengths, U.S. sizes (4000 psi, 12x24 in ...) for
    inch / foot models, defined once and converted to any force / length unit."""
    from ..units import UnitSystem
    units = model.units
    us = units.length in ("in", "ft")
    base = Model("defaults", UnitSystem("kip", "in", units.temperature) if us else UnitSystem("kN", "m", units.temperature))
    if not us:
        base.add_material("C28", E=25.0e6, nu=0.2, weight_density=24.0, type="Concrete", fc=28.0e3,
                          alpha=9.9e-6 if units.temperature == "C" else 5.5e-6)
        for name, d, b in (("B30x60", 0.6, 0.3), ("B25x50", 0.5, 0.25), ("C40x40", 0.4, 0.4), ("C50x50", 0.5, 0.5)):
            base.add_frame_section(FrameSection.rectangular(name, "C28", depth=d, width=b))
        for name, kind, t in (("S15", "Slab", 0.15), ("S20", "Slab", 0.20), ("W20", "Wall", 0.20), ("W25", "Wall", 0.25)):
            base.add_shell_section(ShellSection(name, "C28", type=kind, thickness=t))
    else:
        # 3600 ksi, 150 pcf = 0.150 kip / 1728 in^3, f'c 4 ksi
        base.add_material("4000Psi", E=3600.0, nu=0.2, weight_density=0.150 / 1728.0, type="Concrete", fc=4.0,
                          alpha=5.5e-6 if units.temperature == "F" else 9.9e-6)
        base.add_frame_section(FrameSection.rectangular("B12x24", "4000Psi", depth=24.0, width=12.0))
        base.add_frame_section(FrameSection.rectangular("C18x18", "4000Psi", depth=18.0, width=18.0))
        base.add_shell_section(ShellSection("S8", "4000Psi", type="Slab", thickness=8.0))
        base.add_shell_section(ShellSection("W10", "4000Psi", type="Wall", thickness=10.0))
    conv = base.convert_units(units)
    for k, v in conv.materials.items():
        model.materials[k] = v
    for k, v in conv.frame_sections.items():
        model.frame_sections[k] = v
    for k, v in conv.shell_sections.items():
        model.shell_sections[k] = v
    model.add_load_pattern("Dead", "Dead", self_weight_multiplier=1.0)
    model.add_load_pattern("Live", "Live")
    model.add_linear_static_case("Dead", {"Dead": 1.0})
    model.add_linear_static_case("Live", {"Live": 1.0})
    model.add_modal_case("Modal", max_modes=12)
    model.add_combination("1.2D+1.6L", {"Dead": 1.2, "Live": 1.6})
    model.add_diaphragm("D1")


def new_building(name: str = "Untitled", units: str = "kN m", x_spacings: Sequence[float] = (6.0, 6.0, 6.0),
                 y_spacings: Sequence[float] = (5.0, 5.0), story_heights: Sequence[float] = (3.5, 3.5, 3.5),
                 base_name: str = "Base", story_prefix: str = "Story", properties: bool = True,
                 restrain_base: bool = True, grid_only: bool = True, frame_section: Optional[str] = None,
                 slab_section: Optional[str] = None, grid_mode: str = "story") -> Model:
    """Grid-and-stories model like ETABS' "New Model > Grid Only" (or a full
    frame + slab building when ``grid_only`` is False)."""
    m = Model(name, units)
    m.grid_systems["G1"] = GridSystem(name="G1")
    xs, ys = [0.0], [0.0]
    for s in x_spacings:
        xs.append(xs[-1] + float(s))
    for s in y_spacings:
        ys.append(ys[-1] + float(s))
    for lab, x in zip(_letters(len(xs)), xs):
        m.grid_lines.append(GridLine(system="G1", name=lab, direction="X", ordinate=x))
    for k, y in enumerate(ys, 1):
        m.grid_lines.append(GridLine(system="G1", name=str(k), direction="Y", ordinate=y))
    if grid_mode == "xyz":  # SAP2000 style: Z1 (base), Z2, ... grid lines
        base_name, story_prefix = "Z1", "Z"
        m.set_stories([(f"Z{k + 1}", float(h)) for k, h in enumerate(story_heights, 1)], base_name=base_name)
    else:
        m.set_stories([(f"{story_prefix}{k}", float(h)) for k, h in enumerate(story_heights, 1)], base_name=base_name)
    if properties:
        default_properties(m)
    if grid_mode == "xyz":
        m.set_grid_mode("xyz")
    if not grid_only:
        fs = frame_section or next(n for n, s in m.frame_sections.items() if n.startswith("C"))
        bs = frame_section or next(n for n, s in m.frame_sections.items() if n.startswith("B"))
        ss = slab_section or next(iter(m.shell_sections))
        stories = m.stories_bottom_up()
        for st in stories[1:]:
            below = m.story_below(st.name)
            z0, z1 = below.elevation, st.elevation
            for x in xs:
                for y in ys:
                    m.add_frame((x, y, z0), (x, y, z1), fs, type="Column", story=st.name)
            for y in ys:
                for x0, x1 in zip(xs, xs[1:]):
                    m.add_frame((x0, y, z1), (x1, y, z1), bs, type="Beam", story=st.name)
            for x in xs:
                for y0, y1 in zip(ys, ys[1:]):
                    m.add_frame((x, y0, z1), (x, y1, z1), bs, type="Beam", story=st.name)
            for x0, x1 in zip(xs, xs[1:]):
                for y0, y1 in zip(ys, ys[1:]):
                    m.add_area([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], ss, type="Floor",
                               story=st.name, diaphragm="D1" if "D1" in m.diaphragms else None)
        if restrain_base:
            for p in m.points.values():
                if abs(p.z - stories[0].elevation) < 1e-9:
                    p.restraints = FIXED
    return m


# =============================================================================
# New Model wizard: region settings and starting structures
# =============================================================================
# Materials are written in kN, m (stress in kN/m2 = MPa x 1000) and converted to
# the model's units.  E of the Thai concrete: 15,100 sqrt(f'c) ksc (EIT 1008).
_MPA = 1000.0


def _mat(name, type, E, fy=None, fu=None, fc=None, gamma=0.0, nu=None, alpha=None):
    return dict(name=name, type=type, E=E * _MPA, fy=None if fy is None else fy * _MPA,
                fu=None if fu is None else fu * _MPA, fc=None if fc is None else fc * _MPA,
                weight_density=gamma, nu=(0.2 if type == "Concrete" else 0.3) if nu is None else nu,
                alpha=(9.9e-6 if type == "Concrete" else 1.17e-5) if alpha is None else alpha)


def _ksc(v):  # kgf/cm2 -> MPa
    return v * 0.0980665


_IN = 0.0254
REGIONS = {
    "thailand": dict(
        label="Thailand (TIS / EIT)",
        materials=[_mat("C240", "Concrete", 15100 * 240 ** 0.5 * 0.0980665, fc=_ksc(240), gamma=24.0),
                   _mat("C320", "Concrete", 15100 * 320 ** 0.5 * 0.0980665, fc=_ksc(320), gamma=24.0),
                   _mat("SD40", "Rebar", 200000, fy=_ksc(4000), fu=_ksc(5700), gamma=77.0),
                   _mat("SR24", "Rebar", 200000, fy=_ksc(2400), fu=_ksc(3900), gamma=77.0),
                   _mat("SS400", "Steel", 200000, fy=245, fu=400, gamma=77.0),
                   _mat("SM520", "Steel", 200000, fy=355, fu=520, gamma=77.0)],
        concrete="C240", concrete_col="C320", steel="SS400",
        frames=[("B30x60", "C240", 0.60, 0.30), ("B25x50", "C240", 0.50, 0.25), ("B40x80", "C240", 0.80, 0.40),
                ("C40x40", "C320", 0.40, 0.40), ("C50x50", "C320", 0.50, 0.50), ("C60x60", "C320", 0.60, 0.60)],
        shells=[("S12", "Slab", 0.12), ("S15", "Slab", 0.15), ("S20", "Slab", 0.20), ("S25", "Slab", 0.25),
                ("W20", "Wall", 0.20), ("W25", "Wall", 0.25), ("W30", "Wall", 0.30)],
        steel_sections=("TIS 1227 / JIS G3192", ["H300x300x10x15", "H250x250x9x14", "H400x200x8x13",
                                                  "H350x175x7x11", "H300x150x6.5x9"]),
        rebar=("TIS 20 (round, SR24)", "TIS 24 (deformed)"),
        units="kN m C",
        steel_db="TIS 1227 / JIS hot-rolled", rebar_label="Thai – RB6 … RB34, DB6 … DB40",
        concrete_code="ACI 318-19 / EIT 1008-38", steel_code="AISC 360-22 / EIT 1020"),
    "us": dict(
        label="United States (ACI / AISC)",
        materials=[_mat("4000Psi", "Concrete", 24855.6, fc=27.58, gamma=23.56),
                   _mat("5000Psi", "Concrete", 27789.0, fc=34.47, gamma=23.56),
                   _mat("A615Gr60", "Rebar", 199948, fy=413.7, fu=620.5, gamma=76.97),
                   _mat("A992Fy50", "Steel", 199948, fy=344.7, fu=448.2, gamma=76.97)],
        concrete="4000Psi", concrete_col="5000Psi", steel="A992Fy50",
        frames=[("B12x24", "4000Psi", 24 * _IN, 12 * _IN), ("B16x30", "4000Psi", 30 * _IN, 16 * _IN),
                ("C18x18", "5000Psi", 18 * _IN, 18 * _IN), ("C24x24", "5000Psi", 24 * _IN, 24 * _IN)],
        shells=[("S6", "Slab", 6 * _IN), ("S8", "Slab", 8 * _IN), ("S10", "Slab", 10 * _IN),
                ("W10", "Wall", 10 * _IN), ("W12", "Wall", 12 * _IN)],
        steel_sections=("AISC", ["W12X65", "W10X33", "W18X35", "W16X31", "W16X31"]),
        rebar=("ASTM A615",),
        units="kip ft F",
        steel_db="AISC 16 – W, HSS, C, L", rebar_label="ASTM A615 – #3 … #18",
        concrete_code="ACI 318-19", steel_code="AISC 360-22"),
    "europe": dict(
        label="Europe (Eurocode)",
        materials=[_mat("C30/37", "Concrete", 33000, fc=30, gamma=25.0),
                   _mat("C40/50", "Concrete", 35000, fc=40, gamma=25.0),
                   _mat("B500B", "Rebar", 200000, fy=500, fu=540, gamma=78.5),
                   _mat("S355", "Steel", 210000, fy=355, fu=490, gamma=78.5)],
        concrete="C30/37", concrete_col="C40/50", steel="S355",
        frames=[("B30x60", "C30/37", 0.60, 0.30), ("B25x50", "C30/37", 0.50, 0.25),
                ("C40x40", "C40/50", 0.40, 0.40), ("C50x50", "C40/50", 0.50, 0.50)],
        shells=[("S15", "Slab", 0.15), ("S20", "Slab", 0.20), ("W20", "Wall", 0.20), ("W25", "Wall", 0.25)],
        steel_sections=("EN", ["HE 300 B", "HE 240 B", "IPE 400", "IPE 360", "IPE 300"]),
        rebar=("EN 10080",),
        units="kN m C",
        steel_db="EN 10365 – IPE, HE, UPE", rebar_label="EN 10080 – Ø6 … Ø50",
        concrete_code="EN 1992-1-1", steel_code="EN 1993-1-1"),
    "china": dict(
        label="China (GB)",
        materials=[_mat("C30", "Concrete", 30000, fc=20.1, gamma=25.0),
                   _mat("C40", "Concrete", 32500, fc=26.8, gamma=25.0),
                   _mat("HRB400", "Rebar", 200000, fy=400, fu=540, gamma=78.5),
                   _mat("Q355", "Steel", 206000, fy=355, fu=470, gamma=78.5)],
        concrete="C30", concrete_col="C40", steel="Q355",
        frames=[("B30x60", "C30", 0.60, 0.30), ("B25x50", "C30", 0.50, 0.25),
                ("C40x40", "C40", 0.40, 0.40), ("C50x50", "C40", 0.50, 0.50)],
        shells=[("S15", "Slab", 0.15), ("S20", "Slab", 0.20), ("W20", "Wall", 0.20), ("W25", "Wall", 0.25)],
        steel_sections=("GB/T 11263", ["HW300x300x10x15", "HW250x250x9x14", "HN400x200x8x13",
                                        "HN350x175x7x11", "HN300x150x6.5x9"]),
        rebar=("GB 1499 (HPB / HRB)",),
        units="kN m C",
        steel_db="GB/T 11263 – HW, HM, HN", rebar_label="GB 1499 – Φ6 … Φ50",
        concrete_code="GB 50010", steel_code="GB 50017"),
}
REGION_ORDER = ("thailand", "us", "europe", "china")

# starting structures (key, label, description)
STRUCTURES = (
    ("blank", "Blank", "Empty model: stories only, no grid lines and no members."),
    ("grid", "Grid only", "Grid lines and stories, no members."),
    ("rc_frame", "RC frame + two-way slab", "Columns at grid points, beams on all grid lines, slab panels."),
    ("flat_slab", "Flat slab", "Columns and slabs, no beams."),
    ("flat_slab_edge", "Flat slab + edge beams", "Beams only around the outside."),
    ("core_frame", "Core walls + frame", "Shear-wall box in the middle bay, frame around it."),
    ("steel_deck", "Steel frame + deck", "Steel columns and girders, infill beams at mid-bay."),
    ("frame_only", "Frame only (no slabs)", "Columns and beams, for 2-D or industrial frames."),
)
STRUCTURE_KEYS = tuple(k for k, _, _ in STRUCTURES)
SUPPORTS = ("Fixed", "Pinned", "None")


_STEEL_SECTIONS: Dict[tuple, Optional[FrameSection]] = {}


def _steel_section(name: str, std: str, material: str, temperature: str) -> Optional[FrameSection]:
    """A catalog steel section in kN m, worked out once per session: its properties come
    from meshing the shape (about 0.2 s each), which made every New Model slow to open."""
    from ..catalog import find_steel
    from ..units import UnitSystem
    key = (name, std, material, temperature)
    if key not in _STEEL_SECTIONS:
        e = find_steel(name, std)
        _STEEL_SECTIONS[key] = None if e is None else e.frame_section(material, UnitSystem("kN", "m", temperature))
    return _STEEL_SECTIONS[key]


def warm_region_properties(region: str = "thailand", steel_db: Optional[str] = None) -> None:
    """Works out the region's catalog steel sections ahead (the program does it in the
    background after it opens, so New Model opens at once)."""
    std, names = REGIONS[steel_db or region]["steel_sections"]
    for n in dict.fromkeys(names):
        _steel_section(n, std, REGIONS[region]["steel"], "C")


def region_properties(model: Model, region: str = "thailand", steel_db: Optional[str] = None,
                      rebar_db: Optional[str] = None) -> None:
    """Materials, sections, rebar sizes of a region plus the default load
    patterns and cases, in the model's units.  ``steel_db`` / ``rebar_db``: the
    region whose steel section table / rebar sizes to use (default: ``region``),
    e.g. Thai materials with AISC W shapes."""
    from ..units import UnitSystem
    from ..catalog import find_steel, rebar_catalog
    reg = REGIONS[region]
    base = Model("defaults", UnitSystem("kN", "m", model.units.temperature))
    for m in reg["materials"]:
        d = dict(m)
        if model.units.temperature == "F":
            d["alpha"] = d["alpha"] / 1.8
        base.add_material(**d)
    for name, mat, depth, width in reg["frames"]:
        base.add_frame_section(FrameSection.rectangular(name, mat, depth=depth, width=width))
    for name, kind, t in reg["shells"]:
        mat = reg["concrete_col"] if kind == "Wall" else reg["concrete"]
        base.add_shell_section(ShellSection(name, mat, type=kind, thickness=t))
    std, names = REGIONS[steel_db or region]["steel_sections"]
    for n in dict.fromkeys(names):
        sec = _steel_section(n, std, reg["steel"], model.units.temperature)
        if sec is not None:
            base.add_frame_section(copy.deepcopy(sec))
    conv = base.convert_units(model.units)
    model.materials.update(conv.materials)
    model.frame_sections.update(conv.frame_sections)
    model.shell_sections.update(conv.shell_sections)
    cat = rebar_catalog()
    for s in REGIONS[rebar_db or region]["rebar"]:
        for e in cat.get(s, []):
            model.rebar_sizes[e.name] = e.rebar_size(model.units)
    _default_loads(model)


def _default_loads(model: Model) -> None:
    if "Dead" not in model.load_patterns:
        model.add_load_pattern("Dead", "Dead", self_weight_multiplier=1.0)
        model.add_linear_static_case("Dead", {"Dead": 1.0})
    if "Live" not in model.load_patterns:
        model.add_load_pattern("Live", "Live")
        model.add_linear_static_case("Live", {"Live": 1.0})
    if "Modal" not in getattr(model, "load_cases", {}):
        model.add_modal_case("Modal", max_modes=12)
    model.add_combination("1.2D+1.6L", {"Dead": 1.2, "Live": 1.6})
    if not model.diaphragms:
        model.add_diaphragm("D1")


def copy_settings(model: Model, source: Model) -> None:
    """Settings from a saved model: units are taken over by the caller; here the
    materials, sections, rebar sizes and the load definitions (no geometry)."""
    src = source.convert_units(model.units) if source.units != model.units else source
    for attr in ("materials", "rebar_sizes", "frame_sections", "shell_sections", "load_patterns",
                 "load_cases", "combinations", "diaphragms"):
        a, b = getattr(model, attr, None), getattr(src, attr, None)
        if isinstance(a, dict) and isinstance(b, dict):
            a.update(b)


def first_section(model: Model, prefix: str = "", kind: Optional[str] = None, steel: Optional[bool] = None):
    """Name of the first frame (``kind`` None) or shell (``kind`` Slab / Wall) section."""
    if kind:
        return next((n for n, s in model.shell_sections.items() if s.type == kind), None)
    for n, s in model.frame_sections.items():
        mat = model.materials.get(s.material)
        is_steel = mat is not None and mat.type == "Steel"
        if (steel is None or steel == is_steel) and n.startswith(prefix):
            return n
    return None


def build_model(name: str = "Untitled", units: str = "kN m C", region: str = "thailand",
                x_ordinates: Sequence[float] = (0.0, 6.0, 12.0, 18.0), y_ordinates: Sequence[float] = (0.0, 5.0, 10.0),
                x_labels: Optional[Sequence[str]] = None, y_labels: Optional[Sequence[str]] = None,
                stories: Sequence[Tuple[str, float]] = (("Story1", 3.5), ("Story2", 3.5)), base_name: str = "Base",
                similar: Optional[dict] = None, grid_mode: str = "story", structure: str = "grid",
                column: Optional[str] = None, beam: Optional[str] = None, slab: Optional[str] = None,
                wall: Optional[str] = None, infill: Optional[str] = None, support: str = "Fixed",
                settings_from: Optional[Model] = None, properties: bool = True,
                steel_db: Optional[str] = None, rebar_db: Optional[str] = None) -> Model:
    """The model of the New Model wizard: settings (a region, or a saved
    model's), grid lines at the given ordinates, stories (bottom up), and one
    of the STRUCTURES on that grid."""
    if structure not in STRUCTURE_KEYS:
        raise ValueError(f"unknown structure {structure!r}")
    xs, ys = sorted(float(v) for v in x_ordinates), sorted(float(v) for v in y_ordinates)
    if structure != "blank" and (len(xs) < 2 or len(ys) < 2):
        raise ValueError("At least two grid lines are needed in X and in Y")
    if any(b - a <= 1e-9 for a, b in zip(xs, xs[1:])) or any(b - a <= 1e-9 for a, b in zip(ys, ys[1:])):
        raise ValueError("Two grid lines have the same ordinate")
    if any(float(h) <= 0 for _, h in stories):
        raise ValueError("Story heights must be greater than zero")
    names = [n for n, _ in stories] + [base_name]
    if len(set(names)) != len(names):
        raise ValueError("Story names must be different")
    m = Model(name, units)
    if settings_from is not None:
        copy_settings(m, settings_from)
        _default_loads(m)
    elif properties:
        region_properties(m, region, steel_db, rebar_db)
    if structure != "blank":
        m.grid_systems["G1"] = GridSystem(name="G1")
        xl = list(x_labels) if x_labels else _letters(len(xs))
        yl = list(y_labels) if y_labels else [str(k) for k in range(1, len(ys) + 1)]
        for lab, x in zip(xl, xs):
            m.grid_lines.append(GridLine(system="G1", name=str(lab), direction="X", ordinate=x))
        for lab, y in zip(yl, ys):
            m.grid_lines.append(GridLine(system="G1", name=str(lab), direction="Y", ordinate=y))
    m.set_stories([(n, float(h)) for n, h in stories], base_name=base_name)
    for st, ref in (similar or {}).items():  # {story: the master it is similar to, or None for a master}
        if st in m.stories:
            m.stories[st].similar_to = ref or None
            m.stories[st].is_master = not ref
    if grid_mode == "xyz":
        m.set_grid_mode("xyz")
    if structure in ("blank", "grid"):
        return m
    with m.point_index():  # thousands of members: find the shared joints by a hash, not by a scan
        _place_structure(m, structure, xs, ys, column, beam, slab, wall, infill, support)
    return m


def _place_structure(m: Model, kind, xs, ys, column, beam, slab, wall, infill, support):
    steel = kind == "steel_deck"
    column = column or first_section(m, "", steel=True) if steel else column or first_section(m, "C", steel=False)
    if steel:
        steels = [n for n, s in m.frame_sections.items() if m.materials.get(s.material) and m.materials[s.material].type == "Steel"]
        column = column if column in m.frame_sections else (steels[0] if steels else None)
        beam = beam if beam in m.frame_sections else (steels[2] if len(steels) > 2 else column)
        infill = infill if infill in m.frame_sections else (steels[-1] if steels else beam)
    else:
        beam = beam if beam in m.frame_sections else first_section(m, "B", steel=False)
    slab = slab if slab in m.shell_sections else first_section(m, kind="Slab")
    wall = wall if wall in m.shell_sections else first_section(m, kind="Wall")
    need = [column] + ([beam] if kind in ("rc_frame", "flat_slab_edge", "core_frame", "steel_deck", "frame_only") else []) \
        + ([slab] if kind not in ("frame_only",) else []) + ([wall] if kind == "core_frame" else [])
    if any(n is None for n in need):
        raise ValueError("The settings have no section for this structure (define sections first)")
    nx, ny = len(xs) - 1, len(ys) - 1
    core = (max(0, (nx - 1) // 2), max(0, (ny - 1) // 2)) if kind == "core_frame" else None
    core_x = (xs[core[0]], xs[core[0] + 1]) if core else None
    core_y = (ys[core[1]], ys[core[1] + 1]) if core else None

    def on_core_corner(x, y):
        return core is not None and x in core_x and y in core_y

    def on_core_edge(a, b):  # a beam along a core wall
        if core is None:
            return False
        (x0, y0), (x1, y1) = a, b
        if x0 == x1 and x0 in core_x:
            return min(y0, y1) >= core_y[0] - 1e-9 and max(y0, y1) <= core_y[1] + 1e-9
        if y0 == y1 and y0 in core_y:
            return min(x0, x1) >= core_x[0] - 1e-9 and max(x0, x1) <= core_x[1] + 1e-9
        return False

    beams = kind in ("rc_frame", "core_frame", "steel_deck", "frame_only")
    edge_only = kind == "flat_slab_edge"
    slabs = kind != "frame_only"
    dia = "D1" if "D1" in m.diaphragms else None
    stories = m.stories_bottom_up()
    for st in stories[1:]:
        z0, z1 = m.story_below(st.name).elevation, st.elevation
        for x in xs:
            for y in ys:
                if not on_core_corner(x, y):
                    m.add_frame((x, y, z0), (x, y, z1), column, type="Column", story=st.name)
        if beams or edge_only:
            for j, y in enumerate(ys):
                if edge_only and j not in (0, ny):
                    continue
                for x0, x1 in zip(xs, xs[1:]):
                    if not on_core_edge((x0, y), (x1, y)):
                        m.add_frame((x0, y, z1), (x1, y, z1), beam, type="Beam", story=st.name)
            for i, x in enumerate(xs):
                if edge_only and i not in (0, nx):
                    continue
                for y0, y1 in zip(ys, ys[1:]):
                    if on_core_edge((x, y0), (x, y1)):
                        continue
                    if kind == "steel_deck":  # girders split at mid-bay where the infill beam frames in
                        ym = 0.5 * (y0 + y1)
                        m.add_frame((x, y0, z1), (x, ym, z1), beam, type="Beam", story=st.name)
                        m.add_frame((x, ym, z1), (x, y1, z1), beam, type="Beam", story=st.name)
                    else:
                        m.add_frame((x, y0, z1), (x, y1, z1), beam, type="Beam", story=st.name)
        if kind == "steel_deck":  # simply supported infill beams parallel to X at mid-bay
            for y0, y1 in zip(ys, ys[1:]):
                ym = 0.5 * (y0 + y1)
                for x0, x1 in zip(xs, xs[1:]):
                    f = m.add_frame((x0, ym, z1), (x1, ym, z1), infill, type="Beam", story=st.name)
                    f.releases_i = f.releases_j = (False, False, False, False, True, True)
        if core:
            (cx0, cx1), (cy0, cy1) = core_x, core_y
            for a, b in (((cx0, cy0), (cx1, cy0)), ((cx1, cy0), (cx1, cy1)), ((cx1, cy1), (cx0, cy1)), ((cx0, cy1), (cx0, cy0))):
                m.add_area([(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)], wall,
                           type="Wall", story=st.name)
        if slabs:
            for i, (x0, x1) in enumerate(zip(xs, xs[1:])):
                for j, (y0, y1) in enumerate(zip(ys, ys[1:])):
                    if core and (i, j) == core:
                        continue  # the core is open (stairs, lifts)
                    if kind == "steel_deck":
                        ym = 0.5 * (y0 + y1)
                        parts = [((x0, y0), (x1, y0), (x1, ym), (x0, ym)), ((x0, ym), (x1, ym), (x1, y1), (x0, y1))]
                    else:
                        parts = [((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
                    for p in parts:
                        m.add_area([(x, y, z1) for x, y in p], slab, type="Floor", story=st.name, diaphragm=dia)
    restr = {"Fixed": FIXED, "Pinned": PINNED}.get(support)
    if restr is not None:
        zb = stories[0].elevation
        for p in m.points.values():
            if abs(p.z - zb) < 1e-9:
                p.restraints = restr


def structure_counts(kind: str, nx_lines: int, ny_lines: int, n_stories: int) -> dict:
    """What build_model will create (for the wizard summary)."""
    nx, ny = nx_lines - 1, ny_lines - 1
    out = dict(columns=0, beams=0, slabs=0, walls=0, supports=0)
    if kind in ("blank", "grid") or nx < 1 or ny < 1 or n_stories < 1:
        return out
    core = kind == "core_frame"
    out["columns"] = (nx_lines * ny_lines - (4 if core else 0)) * n_stories
    if kind in ("rc_frame", "core_frame", "frame_only"):
        out["beams"] = (ny_lines * nx + nx_lines * ny - (4 if core else 0)) * n_stories
    elif kind == "flat_slab_edge":
        out["beams"] = (2 * nx + 2 * ny) * n_stories
    elif kind == "steel_deck":
        out["beams"] = (ny_lines * nx + nx_lines * ny * 2 + nx * ny) * n_stories
    if kind != "frame_only":
        out["slabs"] = (nx * ny - (1 if core else 0)) * (2 if kind == "steel_deck" else 1) * n_stories
    out["walls"] = 4 * n_stories if core else 0
    out["supports"] = nx_lines * ny_lines
    return out


# ------------------------------------------------------- Edit > Add to Model From Template
FRAME_2D = (
    ("moment", "Moment frame", "Columns and beams, rigid joints."),
    ("x_brace", "X-braced frame", "Two diagonals crossing in each braced bay."),
    ("diagonal", "Diagonal-braced frame", "One diagonal per story in each braced bay, alternating."),
    ("chevron", "Chevron (inverted V)", "Braces from the lower corners to the middle of the beam above."),
    ("v_brace", "V-braced frame", "Braces from the upper corners to the middle of the beam below."),
    ("eccentric", "Eccentrically braced (EBF)", "Chevron braces meeting the beam a link length apart."),
)
FRAME_2D_KEYS = tuple(k for k, _, _ in FRAME_2D)
STRUCTURE_3D_KEYS = tuple(k for k in STRUCTURE_KEYS if k not in ("blank", "grid"))


def parse_spacings(text: str) -> List[float]:
    """Bay widths from text: "6, 6, 5", "3@6" (three bays of 6), "2@6 4.5" ..."""
    out: List[float] = []
    for tok in str(text).replace(";", ",").replace(",", " ").split():
        if "@" in tok:
            n, v = tok.split("@", 1)
            k = int(float(n))
            if k < 1 or float(n) != k:
                raise ValueError(f"{tok!r}: the count before @ must be a whole number >= 1")
            out += [float(v)] * k
        else:
            out.append(float(tok))
    if not out:
        raise ValueError("Give at least one bay width")
    if any(not (v > 0) for v in out):
        raise ValueError("Bay widths must be greater than zero")
    return out


def _levels(model: Model, bottom: str, top: str) -> List[Tuple[str, float]]:
    st = model.stories_bottom_up()
    names = [s.name for s in st]
    if bottom not in names or top not in names:
        raise ValueError("Unknown story")
    i, j = names.index(bottom), names.index(top)
    if j <= i:
        raise ValueError("The top story must be above the bottom story")
    return [(s.name, s.elevation) for s in st[i:j + 1]]


def _placer(origin: Tuple[float, float], angle: float, z_shift: float = 0.0):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    ox, oy = float(origin[0]), float(origin[1])

    def T(x, y, z):
        return (ox + c * x - s * y, oy + s * x + c * y, z + z_shift)

    return T


class Added(list):
    """The frames and areas a template added; ``skipped``: members the model had already."""
    skipped = 0


def _copy_into(model: Model, tmp: Model, T, angle: float, support: str, base_z: float) -> "Added":
    """The members of ``tmp`` (local coordinates) into ``model`` through T; shared joints
    merge with the model's.  Returns the new frames and areas."""
    made = Added()
    tol = model.options.merge_tolerance

    def dia_for(story):
        for a in model.areas.values():
            if a.type == "Floor" and a.diaphragm and a.story == story:
                return a.diaphragm
        return "D1" if "D1" in model.diaphragms else None

    have_f = {frozenset((f.point_i, f.point_j)) for f in model.frames.values()}
    have_a = {frozenset(a.points) for a in model.areas.values()}
    skipped = 0

    def existing(xyz):
        q = model.find_point(*xyz)
        return q.name if q is not None else None

    with model.point_index():
        for f in tmp.frames.values():
            a, b = tmp.points[f.point_i], tmp.points[f.point_j]
            pa, pb = T(a.x, a.y, a.z), T(b.x, b.y, b.z)
            na, nb_ = existing(pa), existing(pb)
            if na and nb_ and frozenset((na, nb_)) in have_f:
                skipped += 1  # the model has this member already
                continue
            nf = model.add_frame(pa, pb, f.section, type=f.type, story=f.story)
            nf.releases_i, nf.releases_j = f.releases_i, f.releases_j
            if f.type == "Column":
                nf.angle = (f.angle + angle) % 360.0
            have_f.add(frozenset((nf.point_i, nf.point_j)))
            made.append(nf)
        for ar in tmp.areas.values():
            pts = [T(tmp.points[p].x, tmp.points[p].y, tmp.points[p].z) for p in ar.points]
            names = [existing(p) for p in pts]
            if all(names) and frozenset(names) in have_a:
                skipped += 1
                continue
            dia = dia_for(ar.story) if (ar.type == "Floor" and ar.diaphragm) else None
            na = model.add_area(pts, ar.section, type=ar.type, story=ar.story, diaphragm=dia)
            have_a.add(frozenset(na.points))
            made.append(na)
    made.skipped = skipped
    restr = {"Fixed": FIXED, "Pinned": PINNED}.get(support)
    if restr is not None:
        for p in tmp.points.values():
            if abs(p.z - base_z) <= tol:
                q = model.find_point(*T(p.x, p.y, p.z))
                if q is not None:
                    q.restraints = restr
    return made


def _scratch(model: Model, levels) -> Model:
    t = Model("template", model.units)
    t.options = model.options
    t.materials, t.frame_sections, t.shell_sections = model.materials, model.frame_sections, model.shell_sections
    t.diaphragms = model.diaphragms
    t.stories.clear()
    prev = None
    for name, z in levels:
        t.add_story(name, z, 0.0 if prev is None else z - prev)
        prev = z
    return t


def add_structure(model: Model, kind: str, x_spacings: Sequence[float], y_spacings: Sequence[float],
                  bottom: str, top: str, origin: Tuple[float, float] = (0.0, 0.0), angle: float = 0.0,
                  column: Optional[str] = None, beam: Optional[str] = None, slab: Optional[str] = None,
                  wall: Optional[str] = None, infill: Optional[str] = None, support: str = "None") -> "Added":
    """Edit > Add to Model From Template > 3D Structures: one of the New Model
    structures, on its own bays, between two stories of the model, at ``origin``
    turned ``angle`` degrees about Z.  Joints on existing ones are shared, members the
    model has already are not added again (``Added.skipped``)."""
    t, levels = build_structure(model, kind, x_spacings, y_spacings, bottom, top, column, beam, slab, wall, infill)
    return _copy_into(model, t, _placer(origin, angle), angle, support, levels[0][1])


def build_structure(model: Model, kind: str, x_spacings, y_spacings, bottom: str, top: str, column=None, beam=None,
                    slab=None, wall=None, infill=None):
    """The 3D template in local coordinates (a scratch model) and its levels."""
    if kind not in STRUCTURE_3D_KEYS:
        raise ValueError(f"unknown structure {kind!r}")
    levels = _levels(model, bottom, top)
    xs = [0.0]
    for v in x_spacings:
        xs.append(xs[-1] + float(v))
    ys = [0.0]
    for v in y_spacings:
        ys.append(ys[-1] + float(v))
    if len(xs) < 2 or len(ys) < 2 or any(float(v) <= 0 for v in list(x_spacings) + list(y_spacings)):
        raise ValueError("Give bay widths greater than zero in X and in Y")
    t = _scratch(model, levels)
    _place_structure(t, kind, xs, ys, column, beam, slab, wall, infill, "None")
    return t, levels


def add_plane_frame(model: Model, kind: str, spacings: Sequence[float], bottom: str, top: str,
                    origin: Tuple[float, float] = (0.0, 0.0), angle: float = 0.0,
                    column: Optional[str] = None, beam: Optional[str] = None, brace: Optional[str] = None,
                    braced_bays: Optional[Sequence[int]] = None, link: float = 0.0, support: str = "None") -> "Added":
    """Edit > Add to Model From Template > 2D Frames: a plane frame along the
    direction ``angle`` (degrees from X) from ``origin``, between two stories.
    ``braced_bays``: 1-based bays with braces (None = all); braces are pinned at
    both ends (M2, M3 released).  ``link``: the EBF link length (eccentric)."""
    t, levels = build_plane_frame(model, kind, spacings, bottom, top, column, beam, brace, braced_bays, link)
    return _copy_into(model, t, _placer(origin, angle), angle, support, levels[0][1])


def build_plane_frame(model: Model, kind: str, spacings, bottom: str, top: str, column=None, beam=None, brace=None,
                      braced_bays=None, link: float = 0.0):
    """The 2D template in local coordinates (x along the frame, y = 0) and its levels."""
    if kind not in FRAME_2D_KEYS:
        raise ValueError(f"unknown frame {kind!r}")
    levels = _levels(model, bottom, top)
    xs = [0.0]
    for v in spacings:
        if float(v) <= 0:
            raise ValueError("Bay widths must be greater than zero")
        xs.append(xs[-1] + float(v))
    nb = len(xs) - 1
    if nb < 1:
        raise ValueError("Give at least one bay")
    column = column if column in model.frame_sections else first_section(model, "C", steel=None)
    beam = beam if beam in model.frame_sections else first_section(model, "B", steel=None)
    braced = sorted(set(range(1, nb + 1) if not braced_bays else [int(b) for b in braced_bays]))
    if kind != "moment":
        if any(b < 1 or b > nb for b in braced):
            raise ValueError(f"Braced bays must be between 1 and {nb}")
        brace = brace if brace in model.frame_sections else column
    if column is None or beam is None:
        raise ValueError("Define frame sections first")
    if kind == "eccentric":
        widths = [xs[b] - xs[b - 1] for b in braced]
        if not (link > 0) or any(link >= 0.9 * w for w in widths):
            raise ValueError("The link length must be greater than zero and shorter than the braced bays")
    t = _scratch(model, levels)
    pin = (False, False, False, False, True, True)
    with t.point_index():
        for k, ((st, z1), (_, z0)) in enumerate(zip(levels[1:], levels[:-1])):
            for x in xs:
                t.add_frame((x, 0, z0), (x, 0, z1), column, type="Column", story=st)
            for b in range(1, nb + 1):
                x0, x1 = xs[b - 1], xs[b]
                xm = 0.5 * (x0 + x1)
                br = kind != "moment" and b in braced
                split = []
                if br and kind == "chevron":
                    split = [xm]
                elif br and kind == "eccentric":
                    split = [xm - 0.5 * link, xm + 0.5 * link]
                # the beam above (split where braces meet it)
                xx = [x0] + split + [x1]
                for a, c in zip(xx, xx[1:]):
                    t.add_frame((a, 0, z1), (c, 0, z1), beam, type="Beam", story=st)
                if not br:
                    continue
                braces = []
                if kind == "x_brace":
                    braces = [((x0, z0), (x1, z1)), ((x1, z0), (x0, z1))]
                elif kind == "diagonal":
                    braces = [((x0, z0), (x1, z1))] if k % 2 == 0 else [((x1, z0), (x0, z1))]
                elif kind == "chevron":
                    braces = [((x0, z0), (xm, z1)), ((x1, z0), (xm, z1))]
                elif kind == "eccentric":
                    braces = [((x0, z0), (xm - 0.5 * link, z1)), ((x1, z0), (xm + 0.5 * link, z1))]
                elif kind == "v_brace":
                    braces = [((x0, z1), (xm, z0)), ((x1, z1), (xm, z0))]
                for (ax, az), (bx, bz) in braces:
                    f = t.add_frame((ax, 0, az), (bx, 0, bz), brace, type="Brace", story=st)
                    f.releases_i = f.releases_j = pin
        if kind == "v_brace":  # the beams below the V tips (above the bottom level) split at mid-bay
            for (st, z) in levels[1:-1]:
                for b in braced:
                    x0, x1 = xs[b - 1], xs[b]
                    xm = 0.5 * (x0 + x1)
                    old = [f for f in t.frames.values() if f.type == "Beam"
                           and {round(t.points[f.point_i].x, 9), round(t.points[f.point_j].x, 9)} == {round(x0, 9), round(x1, 9)}
                           and abs(t.points[f.point_i].z - z) < 1e-9]
                    for f in old:
                        del t.frames[f.name]
                        t.add_frame((x0, 0, z), (xm, 0, z), beam, type="Beam", story=st)
                        t.add_frame((xm, 0, z), (x1, 0, z), beam, type="Beam", story=st)
    return t, levels
