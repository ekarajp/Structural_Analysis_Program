"""Model templates for the New Model wizard (grid + stories + default properties)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

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


def region_properties(model: Model, region: str = "thailand") -> None:
    """Materials, sections, rebar sizes of a region plus the default load
    patterns and cases, in the model's units."""
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
    std, names = reg["steel_sections"]
    for n in dict.fromkeys(names):
        e = find_steel(n, std)
        if e is not None:
            base.add_frame_section(e.frame_section(reg["steel"], base.units))
    conv = base.convert_units(model.units)
    model.materials.update(conv.materials)
    model.frame_sections.update(conv.frame_sections)
    model.shell_sections.update(conv.shell_sections)
    cat = rebar_catalog()
    for s in reg["rebar"]:
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
                settings_from: Optional[Model] = None, properties: bool = True) -> Model:
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
        region_properties(m, region)
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
