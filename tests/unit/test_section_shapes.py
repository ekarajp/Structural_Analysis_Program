"""Frame section shapes and the Section Designer geometry: exact A / I, the
closed-form J and shear areas of the parametric shapes, the finite-element
torsion constant, unions / openings, units, saving, renaming."""
import math

import numpy as np
import pytest

from structstudio.model import FrameSection, Model
from structstudio.model.section_shapes import (SHAPES, check_dims, convex_pieces, designer_properties, mesh_region,
                                               rect, section_loops, shape_properties, signed_area, torsion_constant)
from structstudio.units import UnitSystem


def j_rect_exact(a, b):
    """Saint-Venant series for a solid rectangle."""
    a, b = max(a, b), min(a, b)
    s = sum(math.tanh(n * math.pi * a / (2 * b)) / n ** 5 for n in range(1, 400, 2))
    return a * b ** 3 / 3 * (1 - 192 / math.pi ** 5 * b / a * s)


def poly(P, hole=False):
    return {"kind": "polygon", "points": np.asarray(P, float).tolist(), "hole": hole}


@pytest.mark.parametrize("a,b", [(0.6, 0.3), (1.0, 1.0), (1.0, 0.1)])
def test_designer_rectangle_exact_area_inertia_and_fe_torsion(a, b):
    p, mesh = designer_properties([poly(rect(0, 0, b, a))])
    assert p.A == pytest.approx(a * b)
    assert p.I33 == pytest.approx(b * a ** 3 / 12) and p.I22 == pytest.approx(a * b ** 3 / 12)
    assert p.xc == pytest.approx(b / 2) and p.yc == pytest.approx(a / 2)
    assert p.J == pytest.approx(j_rect_exact(a, b), rel=1e-3)  # quadratic elements
    assert p.J >= j_rect_exact(a, b) * (1 - 1e-9)  # displacement FE: an upper bound
    assert p.As2 == pytest.approx(5 / 6 * a * b, rel=1e-4)  # shear-flow integral of a rectangle = 5/6 A
    assert p.Z33 == pytest.approx(b * a * a / 4, rel=1e-3)


def test_designer_circle_and_hollow_circle():
    r = 0.25
    p, _ = designer_properties([{"kind": "circle", "center": [0, 0], "radius": r}])
    assert p.A == pytest.approx(math.pi * r * r, rel=2e-3)  # 96-gon
    assert p.J == pytest.approx(math.pi * r ** 4 / 2, rel=3e-3)
    p, _ = designer_properties([{"kind": "circle", "center": [0, 0], "radius": r},
                                {"kind": "circle", "center": [0, 0], "radius": 0.8 * r, "hole": True}])
    J = math.pi * (r ** 4 - (0.8 * r) ** 4) / 2
    assert p.J == pytest.approx(J, rel=3e-3)  # multiply connected: the warping formulation handles the hole


def test_t_drawn_as_two_rectangles_is_united():
    """A flange and a web drawn as touching rectangles = the parametric T-beam."""
    geom = [poly(rect(-0.6, 0.48, 1.2, 0.12)), poly(rect(-0.15, 0.0, 0.3, 0.48))]
    p, mesh = designer_properties(geom)
    q = shape_properties(SHAPES["Concrete Tee"], {"t3": 0.6, "t2": 1.2, "tf": 0.12, "tw": 0.3})
    for k in ("A", "I33", "I22"):
        assert getattr(p, k) == pytest.approx(getattr(q, k), rel=1e-10)
    assert len(mesh.loops) == 1 and len(mesh.loops[0]) == 8  # one outline, the shared edge gone
    assert p.As2 == pytest.approx(q.As2, rel=1e-6)  # both from the shear-flow integral
    # overlapping instead of touching (web drawn up into the flange): the same union
    p2, _ = designer_properties([poly(rect(-0.6, 0.48, 1.2, 0.12)), poly(rect(-0.15, 0.0, 0.3, 0.55))])
    assert p2.A == pytest.approx(p.A) and p2.I33 == pytest.approx(p.I33)
    # the junction adds torsional stiffness beyond the sum of the two rectangles
    assert p.J > q.J


def test_opening_and_separate_parts():
    box = [poly(rect(0, 0, 0.2, 0.2)), poly(rect(0.01, 0.01, 0.18, 0.18), hole=True)]
    p, mesh = designer_properties(box)
    q = shape_properties(SHAPES["Box"], {"t3": 0.2, "t2": 0.2, "tf": 0.01, "tw": 0.01})
    assert p.A == pytest.approx(q.A) and p.I33 == pytest.approx(q.I33)
    assert sorted(np.sign([signed_area(L) for L in mesh.loops])) == [-1, 1]
    # thin closed tube: Bredt (centre-line) J
    Bredt = 4 * (0.19 * 0.19) ** 2 * 0.01 / (4 * 0.19)
    assert p.J == pytest.approx(Bredt, rel=0.03)
    two, mesh = designer_properties([poly(rect(0, 0, 0.1, 0.3)), poly(rect(0.5, 0, 0.1, 0.3))])
    assert two.A == pytest.approx(0.06) and two.xc == pytest.approx(0.3)
    assert two.J == pytest.approx(2 * j_rect_exact(0.3, 0.1), rel=2e-3)  # parts: J adds


def test_conforming_triangulation_of_a_non_convex_outline():
    L = np.array([[0, 0], [1, 0], [1, 0.1], [0.1, 0.1], [0.1, 1], [0, 1]], float)
    mesh = mesh_region([L], [], None)
    assert mesh.conforming
    area = sum(abs(signed_area(mesh.P[t])) for t in mesh.tris)
    assert area == pytest.approx(0.19)
    fine = mesh_region([L], [], 0.02)
    assert torsion_constant(fine) == pytest.approx(2 * j_rect_exact(1, 0.1) * 1.03, rel=0.04)


@pytest.mark.parametrize("key", list(SHAPES))
def test_every_catalogue_shape(key):
    sh = SHAPES[key]
    v = {d.attr: d.default for d in sh.dims}
    if key == "Lipped Channel":
        v["tf"] = v["tw"]
    assert check_dims(sh, v) == []
    p = shape_properties(sh, v)
    assert p.A > 0 and p.I33 > 0 and p.I22 > 0 and p.J > 0 and p.As2 > 0 and p.As3 > 0
    sec = FrameSection.from_shape(key, key, "M", **v)
    assert sec.A == pytest.approx(p.A) and sec.J == pytest.approx(p.J)
    solids, holes = section_loops(sec)
    from structstudio.model.section_shapes import loop_moments
    m = sum(loop_moments(P) for P in solids + holes)
    assert m[1] == pytest.approx(0, abs=1e-12) and m[2] == pytest.approx(0, abs=1e-12)  # about the centroid
    pieces = convex_pieces(sec)
    assert sum(abs(signed_area(P)) for P in pieces) == pytest.approx(p.A, rel=1e-9 if key not in ("Circular", "Pipe", "Concrete Pipe") else 3e-3)


def test_closed_forms_against_hand_values():
    # H300x150x6.5x9 without fillets
    p = shape_properties(SHAPES["I"], {"t3": 0.3, "t2": 0.15, "tf": 0.009, "tw": 0.0065})
    A = 2 * 0.15 * 0.009 + 0.282 * 0.0065
    I33 = 0.15 * 0.3 ** 3 / 12 - (0.15 - 0.0065) * 0.282 ** 3 / 12
    assert p.A == pytest.approx(A) and p.I33 == pytest.approx(I33)
    assert p.As2 == pytest.approx(0.3 * 0.0065) and p.As3 == pytest.approx(5 / 3 * 0.15 * 0.009)
    assert p.J == pytest.approx((2 * 0.15 * 0.009 ** 3 + 0.282 * 0.0065 ** 3) / 3)
    assert p.Z33 == pytest.approx(0.15 * 0.009 * 0.291 + 0.0065 * 0.141 ** 2, rel=2e-3)
    # unequal flanges: centroid moves toward the bigger one
    q = shape_properties(SHAPES["I"], {"t3": 0.3, "t2": 0.15, "tf": 0.009, "tw": 0.0065, "t2b": 0.25, "tfb": 0.012})
    assert q.A > p.A
    # angle: product of inertia, principal axes at 45 deg for an equal angle
    a = shape_properties(SHAPES["Angle"], {"t3": 0.1, "t2": 0.1, "tf": 0.01, "tw": 0.01})
    assert a.I23 != 0 and abs(abs(math.degrees(a.angle)) - 45) < 1e-6
    m = shape_properties(SHAPES["Angle"], {"t3": 0.1, "t2": 0.1, "tf": 0.01, "tw": 0.01}, mirror=True)
    assert m.I23 == pytest.approx(-a.I23) and m.I33 == pytest.approx(a.I33)
    # pipe and box closed forms
    pp = shape_properties(SHAPES["Pipe"], {"t3": 0.1143, "tw": 0.0045})
    Di = 0.1143 - 0.009
    assert pp.A == pytest.approx(math.pi / 4 * (0.1143 ** 2 - Di ** 2))
    assert pp.J == pytest.approx(math.pi / 32 * (0.1143 ** 4 - Di ** 4))
    bx = shape_properties(SHAPES["Box"], {"t3": 0.2, "t2": 0.1, "tf": 0.008, "tw": 0.006})
    assert bx.As2 == pytest.approx(2 * 0.006 * 0.2) and bx.As3 == pytest.approx(2 * 0.008 * 0.1)


def test_dimension_checks():
    assert check_dims(SHAPES["Concrete Tee"], {"t3": 0.6, "t2": 1.2, "tf": 0.7, "tw": 0.3})
    assert check_dims(SHAPES["Box"], {"t3": 0.2, "t2": 0.2, "tf": 0.11, "tw": 0.01})
    assert check_dims(SHAPES["Rectangular"], {"t3": 0.6, "t2": -0.3})
    assert check_dims(SHAPES["I"], {"t3": 0.3, "t2": 0.15, "tf": 0.009, "tw": 0.0065}) == []
    from structstudio.model.base import ModelError
    with pytest.raises(ModelError):
        FrameSection.from_shape("Concrete Tee", "bad", "M", t3=0.6, t2=1.2, tf=0.7, tw=0.3)


def test_units_saving_and_renaming():
    m = Model("s", UnitSystem("kN", "m", "C"))
    m.add_material("C", E=25e6, weight_density=24)
    geom = [poly(rect(-0.15, 0, 0.3, 0.6)), {"kind": "circle", "center": [0, 0.3], "radius": 0.05, "hole": True}]
    sd = m.add_frame_section(FrameSection.from_designer("SD1", "C", geom))
    tee = m.add_frame_section(FrameSection.from_shape("Concrete L", "L1", "C", mirror=True, t3=0.6, t2=0.8, tf=0.12, tw=0.3))
    assert tee.params == {"mirror": True} and sd.t3 == pytest.approx(0.6) and sd.t2 == pytest.approx(0.3)
    a, b = m.add_point(0, 0, 0), m.add_point(5, 0, 0)
    f = m.add_frame(a, b, "SD1")
    # units: the drawing scales with the lengths, the properties stay consistent
    cm = m.convert_units(UnitSystem("kN", "cm", "C"))
    s2 = cm.frame_sections["SD1"]
    assert s2.geometry[0]["points"][1] == pytest.approx([15.0, 0.0])
    assert s2.geometry[1]["radius"] == pytest.approx(5.0)
    A, J = s2.A, s2.J
    s2.update_properties()
    assert s2.A == pytest.approx(A, rel=1e-9) and s2.J == pytest.approx(J, rel=1e-6)
    # saving
    back = Model.from_dict(m.to_dict())
    assert back.frame_sections["SD1"].geometry == m.frame_sections["SD1"].geometry
    assert back.frame_sections["L1"].params == {"mirror": True}
    # renaming follows the members
    m.rename_frame_section("SD1", "PIER")
    assert m.frames[f.name].section == "PIER" and list(m.frame_sections)[0] == "PIER"
    assert m.frame_section_users("PIER") == [f.name]
    with pytest.raises(Exception):
        m.rename_frame_section("PIER", "L1")


def test_analysis_uses_the_designer_properties():
    from structstudio.api import analyze
    from structstudio.model import FIXED
    m = Model("cant", UnitSystem("kN", "m", "C"))
    m.add_material("C", E=25e6, nu=0.2, weight_density=0.0)
    geom = [poly(rect(-0.6, 0.48, 1.2, 0.12)), poly(rect(-0.15, 0.0, 0.3, 0.48))]
    sec = m.add_frame_section(FrameSection.from_designer("T", "C", geom))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(4, 0, 0)
    m.add_frame(a, b, "T", name="F")
    m.add_load_pattern("P", "Other")
    m.add_point_load(b.name, "P", force=(0, 0, -10))
    m.add_linear_static_case("P")
    res = analyze(m)
    E, G = 25e6, 25e6 / 2.4
    u = -10 * 4 ** 3 / (3 * E * sec.I33) - 10 * 4 / (G * sec.As2)  # bending + shear (Timoshenko)
    assert res.joint_displacement(b.name, "P")[2] == pytest.approx(u, rel=1e-6)


@pytest.mark.parametrize("key,v,table", [
    # JIS G3192 H300x150x6.5x9 r13: A 46.78 cm2, Ix 7210 cm4, Iy 508 cm4, Zx (elastic) 481 cm3
    ("I", {"t3": 0.3, "t2": 0.15, "tf": 0.009, "tw": 0.0065, "r": 0.013},
     {"A": 46.78e-4, "I33": 7210e-8, "I22": 508e-8, "S33": 481e-6}),
    # JIS L100x100x10 r1 10, r2 7: A 19.00 cm2, Ix 175 cm4, Iu 278 cm4, Iv 72.0 cm4
    ("Angle", {"t3": 0.1, "t2": 0.1, "tf": 0.01, "tw": 0.01, "r": 0.010, "r2": 0.007},
     {"A": 19.00e-4, "I33": 175e-8, "Imax": 278e-8, "Imin": 72.0e-8}),
    # JIS G3350 C150x65x20x2.3 (inside radius = t): A 7.012 cm2, Ix 248 cm4, Iy 41.1 cm4
    ("Lipped Channel", {"t3": 0.15, "t2": 0.065, "lip": 0.02, "tw": 0.0023, "tf": 0.0023, "r": 0.0023},
     {"A": 7.012e-4, "I33": 248e-8, "I22": 41.1e-8}),
])
def test_fillets_reproduce_the_steel_tables(key, v, table):
    p = shape_properties(SHAPES[key], v)
    for k, x in table.items():
        assert getattr(p, k) == pytest.approx(x, rel=0.006), k


def test_fillets_torsion_by_finite_elements_matches_aisc():
    """AISC W14x90 (in): d 14.0, bf 14.5, tf 0.710, tw 0.440, kdes 1.31 -> r = 0.60:
    A 26.5, Ix 999, Iy 362, J 4.06 (AISC tables, El Darwish & Johnston)."""
    v = {"t3": 14.0, "t2": 14.5, "tf": 0.71, "tw": 0.44, "r": 0.60}
    p = shape_properties(SHAPES["I"], v)
    assert p.J_method.startswith("finite elements")
    assert p.J == pytest.approx(4.06, rel=0.01)
    assert p.A == pytest.approx(26.5, rel=0.01) and p.I33 == pytest.approx(999, rel=0.01)
    sharp = shape_properties(SHAPES["I"], dict(v, r=0))
    assert sharp.J_method.startswith("closed form") and sharp.J < p.J  # the fillets stiffen the junctions
    # rounded tube corners: Bredt on the rounded centre line
    b = shape_properties(SHAPES["Box"], {"t3": 0.2, "t2": 0.2, "tf": 0.008, "tw": 0.008, "r": 0.016})
    bs = shape_properties(SHAPES["Box"], {"t3": 0.2, "t2": 0.2, "tf": 0.008, "tw": 0.008})
    assert b.A < bs.A
    fe = torsion_constant(mesh_region(*__import__("structstudio.model.section_shapes", fromlist=["x"]).shape_outline(
        SHAPES["Box"], {"t3": 0.2, "t2": 0.2, "tf": 0.008, "tw": 0.008, "r": 0.016}), 0.002))
    assert b.J == pytest.approx(fe, rel=0.02)


def test_radii_must_fit():
    I = SHAPES["I"]
    assert check_dims(I, {"t3": 0.3, "t2": 0.15, "tf": 0.009, "tw": 0.0065, "r": 0.08})
    assert check_dims(I, {"t3": 0.3, "t2": 0.15, "tf": 0.009, "tw": 0.0065, "r": -0.001})
    assert check_dims(I, {"t3": 0.3, "t2": 0.15, "tf": 0.009, "tw": 0.0065, "r": 0.0}) == []
    assert check_dims(SHAPES["Angle"], {"t3": 0.1, "t2": 0.1, "tf": 0.01, "tw": 0.01, "r": 0.05, "r2": 0.05})
    sec = FrameSection.from_shape("I", "H", "S", t3=0.3, t2=0.15, tf=0.009, tw=0.0065, r=0.013)
    cm = sec.convert_units(UnitSystem("kN", "m", "C"), UnitSystem("kN", "cm", "C"))
    assert cm.r == pytest.approx(1.3) and cm.A == pytest.approx(sec.A * 1e4)


def _src_model():
    from structstudio.model import FIXED
    from structstudio.model.section_shapes import shape_outline
    m = Model("src", UnitSystem("kN", "m", "C"))
    m.add_material("C", E=25e6, nu=0.2, weight_density=24.0)
    m.add_material("S", E=200e6, nu=0.3, weight_density=77.0, type="Steel")
    I, _ = shape_outline(SHAPES["I"], {"t3": 0.3, "t2": 0.3, "tf": 0.015, "tw": 0.01, "r": 0.018})
    geom = [poly(rect(-0.3, -0.3, 0.6, 0.6)), {"kind": "polygon", "points": I[0].tolist(), "material": "S"}]
    sec = m.add_frame_section(FrameSection(name="SRC", material="C", shape="Composite", geometry=geom))
    sec.update_properties(True, m.materials)
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(0, 0, 3.5)
    m.add_frame(a, b, "SRC", name="COL")
    return m, sec, geom


def test_composite_src_column_transformed_properties_and_weight():
    from structstudio.model.section_shapes import material_areas, shape_properties as sp
    m, sec, geom = _src_model()
    n = 200e6 / 25e6
    st = sp(SHAPES["I"], {"t3": 0.3, "t2": 0.3, "tf": 0.015, "tw": 0.01, "r": 0.018})
    areas = material_areas(geom)
    assert areas["S"] == pytest.approx(st.A, rel=1e-9) and areas[None] == pytest.approx(0.36 - st.A, rel=1e-9)
    assert sec.A == pytest.approx(0.36 - st.A + n * st.A, rel=1e-9)
    assert sec.I33 == pytest.approx(0.6 ** 4 / 12 + (n - 1) * st.I33, rel=1e-9)
    assert sec.I22 == pytest.approx(0.6 ** 4 / 12 + (n - 1) * st.I22, rel=1e-9)
    assert sec.params["n"]["S"][0] == pytest.approx(n)
    # J: between the concrete alone and concrete + n_G x steel (the steel is stiffer in torsion)
    assert j_rect_exact(0.6, 0.6) < sec.J < j_rect_exact(0.6, 0.6) * 1.3
    # the analysis model: stiffness transformed, weight from each material's own area
    from structstudio.mesh.mesher import build_analysis_model
    am = build_analysis_model(m)
    el = next(e for e in am.frames if e.object == "COL")
    assert el.A == pytest.approx(sec.A) and el.E == pytest.approx(25e6 * 1e3)  # SI: kN -> N
    w = (areas[None] * 24.0 + areas["S"] * 77.0) * 1e3  # N/m
    assert el.weight_per_length == pytest.approx(w, rel=1e-9)
    # a stiffer concrete changes n: the analysis follows the materials
    m.materials["C"].E = 32e6
    el2 = next(e for e in build_analysis_model(m).frames if e.object == "COL")
    assert el2.A == pytest.approx(0.36 - st.A + 200 / 32 * st.A, rel=1e-9)
    # units: the drawing and its dimensions scale
    cm = m.convert_units(UnitSystem("kN", "cm", "C"))
    assert cm.frame_sections["SRC"].geometry[0]["points"][1] == pytest.approx([30.0, -30.0])


def test_lipped_z_has_inclined_principal_axes():
    v = {"t3": 0.15, "t2": 0.065, "lip": 0.02, "tw": 0.0023, "tf": 0.0023, "r": 0.0023}
    z = shape_properties(SHAPES["Lipped Z"], v)
    c = shape_properties(SHAPES["Lipped Channel"], v)
    assert z.A == pytest.approx(c.A, rel=1e-9) and z.I33 == pytest.approx(c.I33, rel=1e-9)  # same plates
    assert abs(z.I23) > 0.1 * z.I22 and z.Imin < z.I22 < z.Imax
    assert z.xc == pytest.approx(0, abs=1e-12) and z.yc == pytest.approx(0, abs=1e-12)  # point symmetric


@pytest.mark.parametrize("shape, dims, slope, A, I33, I22", [
    # TIS 1227 / JIS G3192 channel C150x75x6.5x10 r1 10 r2 5, 5 deg taper: 23.71 cm2, 861, 117 cm4
    ("Channel", dict(t3=150, t2=75, tf=10, tw=6.5, r=10, r2=5), math.tan(math.radians(5)), 2371, 861e4, 117e4),
    # GB/T 706 channel [10: 100 x 48 x 5.3, t 8.5, r 8.5, r1 4.2, slope 1:10: 12.74 cm2, 198, 25.6
    ("Channel", dict(t3=100, t2=48, tf=8.5, tw=5.3, r=8.5, r2=4.25), 0.1, 1274, 198e4, 25.6e4),
    # GB/T 706 I20a: 200 x 100 x 7, t 11.4, r 9, r1 4.5, slope 1:6: 35.55 cm2, 2370, 158
    ("I", dict(t3=200, t2=100, tf=11.4, tw=7, r=9, r2=4.5), 1 / 6, 3555, 2370e4, 158e4),
])
def test_tapered_flanges_match_the_standards(shape, dims, slope, A, I33, I22):
    p = shape_properties(SHAPES[shape], dict(dims, slope=slope), False, False)
    assert p.A == pytest.approx(A, rel=0.002)
    assert p.I33 == pytest.approx(I33, rel=0.005) and p.I22 == pytest.approx(I22, rel=0.01)
    flat = shape_properties(SHAPES[shape], dims, False, False)
    assert flat.A == pytest.approx(p.A, rel=0.01) and abs(flat.I22 / p.I22 - 1) > 0.02  # the taper matters for I22


def test_flange_slope_checks_saving_and_units():
    v = dict(t3=150, t2=75, tf=10, tw=6.5, r=10, r2=5)
    assert check_dims(SHAPES["Channel"], dict(v, slope=0.6))  # out of range
    assert check_dims(SHAPES["Channel"], dict(v, tf=2, r2=0, slope=0.1))  # no flange left at the tip
    assert check_dims(SHAPES["Angle"], dict(t3=100, t2=100, tf=10, tw=10, slope=0.1))  # only I / channels
    m = Model("t", "kN m")
    sec = FrameSection.from_shape("Channel", "C150", "S", t3=0.15, t2=0.075, tf=0.01, tw=0.0065, r=0.01, r2=0.005,
                                  slope=math.tan(math.radians(5)))
    assert sec.params["slope"] == pytest.approx(0.0875, abs=1e-4) and sec.A == pytest.approx(23.71e-4, rel=0.002)
    m.frame_sections[sec.name] = sec
    back = Model.from_json(m.to_json()).frame_sections["C150"]
    assert back.params["slope"] == sec.params["slope"]
    cm = m.convert_units("kgf cm").frame_sections["C150"]
    assert cm.params["slope"] == sec.params["slope"] and cm.A == pytest.approx(23.71, rel=0.002)
    cm.update_properties()
    assert cm.I22 == pytest.approx(117, rel=0.01)


@pytest.mark.parametrize("key, dims, A, I33, I22", [
    # TIS 1228-2561 annex A (= JIS G3350), inside bend radius = t; mm, cm2, cm4
    ("Cold Angle", dict(t3=60, t2=60, tw=3.2, r=3.2), 3.672, 13.1, 13.1),
    ("Cold Channel", dict(t3=450, t2=75, tw=6, r=6), 34.82, 8400, 122),
    ("Cold Z", dict(t3=75, t2=40, t2b=30, tw=2.3, r=2.3), 3.161, 26.8, 6.15),
    ("Hat", dict(t3=60, t2=30, lip=25, tw=2.3, r=2.3), 4.358, 20.9, 14.7),
])
def test_cold_formed_shapes_match_tis_1228(key, dims, A, I33, I22):
    from structstudio.model.section_shapes import COLD
    assert key in COLD
    dims = dict(dims, tf=dims["tw"])
    assert not check_dims(SHAPES[key], dims)
    p = shape_properties(SHAPES[key], dims, False, False)
    assert p.A / 100 == pytest.approx(A, rel=0.002)
    assert p.I33 / 1e4 == pytest.approx(I33, rel=0.01) and p.I22 / 1e4 == pytest.approx(I22, rel=0.01)
    assert p.J > 0
    assert check_dims(SHAPES[key], dict(dims, tw=dims["t3"]))  # too thick
