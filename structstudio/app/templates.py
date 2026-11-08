"""Model templates for the New Model wizard (grid + stories + default properties)."""
from __future__ import annotations

from typing import Optional, Sequence

from ..model import FIXED, FrameSection, Model, ShellSection
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
