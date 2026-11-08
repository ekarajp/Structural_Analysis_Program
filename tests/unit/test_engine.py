"""Physical unit tests of the finite element engine (closed-form checks)."""
import math

import numpy as np
import pytest

from structstudio import Model, analyze
from structstudio.model import FIXED, PINNED, FrameSection, ShellSection
from structstudio.units import UnitSystem


def steel_beam_model(units="kN m", shear_area=False, weight_density=77.0):
    m = Model("beam", units)
    m.add_material("S", E=200e6, nu=0.3, weight_density=weight_density, type="Steel")
    As = 0.005 if shear_area else 0.0
    m.add_frame_section(FrameSection.general("B", "S", A=0.01, I33=1e-4, I22=5e-5, J=1e-6, As2=As, As3=As))
    m.add_load_pattern("P", "Other")
    return m


# ---------------------------------------------------------------- statics
def test_cantilever_tip_load():
    m = steel_beam_model()
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(3, 0, 0)
    m.add_frame(a, b, "B", name="F")
    m.add_point_load(b, "P", force=(0, 0, -10))
    m.add_linear_static_case("P")
    res = analyze(m)
    assert res.joint_displacement(b.name, "P")[2] == pytest.approx(-10 * 27 / (3 * 200e6 * 1e-4), rel=1e-9)
    assert res.joint_reaction(a.name, "P")[2] == pytest.approx(10.0, rel=1e-9)
    assert res.joint_reaction(a.name, "P")[4] == pytest.approx(-30.0, rel=1e-9)
    end = res.frame_end_forces("F", "P")
    assert end[0, 1] == pytest.approx(-10.0)  # V2 at end I
    assert end[0, 5] == pytest.approx(-30.0)  # hogging M3 at the support


def test_fixed_fixed_uniform_load_and_partial_load():
    m = steel_beam_model()
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(6, 0, 0, restraints=FIXED)
    m.add_frame(a, b, "B", name="F", min_stations=13)
    m.add_frame_distributed_load("F", "P", -5.0, direction="Z")  # 5 kN/m downward
    m.add_linear_static_case("P")
    res = analyze(m)
    st = res.frame_forces("F", "P")
    L, w = 6.0, 5.0
    assert st.forces[0, 5] == pytest.approx(-w * L ** 2 / 12, rel=1e-9)
    mid = int(np.argmin(np.abs(st.station - 3.0)))
    assert st.forces[mid, 5] == pytest.approx(w * L ** 2 / 24, rel=1e-9)
    # partial uniform load on the left half: reactions by superposition (textbook)
    m2 = steel_beam_model()
    a = m2.add_point(0, 0, 0, restraints=FIXED)
    b = m2.add_point(6, 0, 0, restraints=FIXED)
    m2.add_frame(a, b, "B", name="F")
    m2.add_frame_distributed_load("F", "P", w, rel_a=0.0, rel_b=0.5)  # Gravity: positive = downward
    m2.add_linear_static_case("P")
    r2 = analyze(m2)
    # fixed-end moments for partial UDL from 0 to a: M_A = w a^2 (6L^2 - 8aL + 3a^2)/(12 L^2)
    aa = 3.0
    MA = w * aa ** 2 * (6 * L ** 2 - 8 * aa * L + 3 * aa ** 2) / (12 * L ** 2)
    MB = w * aa ** 3 * (4 * L - 3 * aa) / (12 * L ** 2)
    ra = r2.joint_reaction(a.name, "P")
    rb = r2.joint_reaction(b.name, "P")
    assert ra[4] == pytest.approx(-MA, rel=1e-9)
    assert rb[4] == pytest.approx(MB, rel=1e-9)
    assert ra[2] + rb[2] == pytest.approx(w * aa, rel=1e-9)


def test_timoshenko_cantilever():
    m = steel_beam_model(shear_area=True)
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(2, 0, 0)
    m.add_frame(a, b, "B", name="F")
    m.add_point_load(b, "P", force=(0, 0, -10))
    m.add_linear_static_case("P")
    res = analyze(m)
    E, I, G, As, L, P = 200e6, 1e-4, 200e6 / 2.6, 0.005, 2.0, 10.0
    exact = -(P * L ** 3 / (3 * E * I) + P * L / (G * As))
    assert res.joint_displacement(b.name, "P")[2] == pytest.approx(exact, rel=1e-9)


def test_pinned_release_and_spring_support():
    m = steel_beam_model()
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(4, 0, 0, springs=(0, 0, 1000.0))
    m.add_frame(a, b, "B", name="F", releases_j=(False, False, False, False, False, True))
    m.add_point_load(b, "P", force=(0, 0, -10))
    m.add_linear_static_case("P")
    res = analyze(m)
    # cantilever stiffness in series with a spring: tip force shared
    kc = 3 * 200e6 * 1e-4 / 64
    k = 1000.0
    u = -10 / (kc + k)
    assert res.joint_displacement(b.name, "P")[2] == pytest.approx(u, rel=1e-9)
    assert res.joint_reaction(b.name, "P")[2] == pytest.approx(-k * u, rel=1e-9)


def test_rigid_end_zone_reduces_flexible_length():
    """Column with a fully rigid end offset: lateral stiffness 12EI/Lf^3."""
    m = steel_beam_model()
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(0, 0, 3, restraints=(False, True, True, True, True, True))  # only UX free
    m.add_frame(a, b, "B", name="C", end_offset_j=0.5, rigid_factor=1.0)
    m.add_point_load(b, "P", force=(10, 0, 0))
    m.add_linear_static_case("P")
    res = analyze(m)
    EI = 200e6 * 5e-5  # bending about local 3 for a column: local 2 = +X -> I33? column local 2 = X, so I33 governs
    EI = 200e6 * 1e-4
    Lf = 2.5
    assert res.joint_displacement(b.name, "P")[0] == pytest.approx(10 * Lf ** 3 / (12 * EI), rel=1e-9)


def test_insertion_point_offset_couples_axial_and_bending():
    """A beam inserted at its top (cardinal 8) hanging from fixed joints
    behaves like a beam with rigid links; equilibrium must hold."""
    m = Model("ins", "kN m")
    m.add_material("C", E=30e6, nu=0.2)
    m.add_frame_section(FrameSection.rectangular("B", "C", 0.6, 0.3))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(5, 0, 0, restraints=FIXED)
    m.add_frame(a, b, "B", name="F", cardinal_point=8)
    m.add_load_pattern("P", "Other")
    m.add_frame_distributed_load("F", "P", 10.0)  # Gravity: positive = downward
    m.add_linear_static_case("P")
    res = analyze(m)
    ra = res.joint_reaction(a.name, "P")
    rb = res.joint_reaction(b.name, "P")
    assert ra[2] + rb[2] == pytest.approx(50.0, rel=1e-9)
    assert ra[2] == pytest.approx(25.0, rel=1e-9)


def test_unit_system_invariance():
    """Same beam in kN-m and kip-in gives the same physical answer."""
    def build(units):
        m = Model("u", units)
        U = UnitSystem.parse(units)
        E = 200e6 * UnitSystem("kN", "m").factor("stress") / U.factor("stress")
        I = 1e-4 * UnitSystem("kN", "m").factor("inertia") / U.factor("inertia")
        A = 0.01 * UnitSystem("kN", "m").factor("area") / U.factor("area")
        L = 3.0 / U.factor("length")
        P = 10.0 * 1e3 / U.factor("force")
        m.add_material("S", E=E, nu=0.3)
        m.add_frame_section(FrameSection.general("B", "S", A=A, I33=I, I22=I, J=I))
        a = m.add_point(0, 0, 0, restraints=FIXED)
        b = m.add_point(L, 0, 0)
        m.add_frame(a, b, "B", name="F")
        m.add_load_pattern("P", "Other")
        m.add_point_load(b, "P", force=(0, 0, -P))
        m.add_linear_static_case("P")
        r = analyze(m)
        return r.joint_displacement(b.name, "P")[2] * U.factor("length"), r.frame_end_forces("F", "P")[0, 5] * U.factor("moment")
    u1, m1 = build("kN m")
    u2, m2 = build("kip in")
    assert u1 == pytest.approx(u2, rel=1e-9)
    assert m1 == pytest.approx(m2, rel=1e-9)


# ---------------------------------------------------------------- dynamics
def test_sdof_period_and_response_spectrum():
    m = steel_beam_model(weight_density=0.0)  # no element self mass
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(0, 0, 3, mass=(2.0, 2.0, 0.0), restraints=(False, True, True, True, True, True))
    m.add_frame(a, b, "B", name="C")
    m.add_modal_case("Modal", max_modes=1)
    m.add_spectrum_function("FLAT", [0.0, 10.0], [0.25, 0.25], damping=0.05)  # 0.25 g constant
    m.add_response_spectrum_case("RS", "Modal", [("U1", "FLAT", 9.80665)])
    res = analyze(m)
    k = 12 * 200e6 * 1e-4 / 27  # fixed-fixed lateral stiffness (rotation restrained at top)
    T = 2 * math.pi * math.sqrt(2.0 / k)
    assert res.modal_periods("Modal")[0]["Period"] == pytest.approx(T, rel=1e-9)
    # SDOF spectral displacement Sd = Sa / w^2
    Sa = 0.25 * 9.80665
    assert res.joint_displacement(b.name, "RS")[0] == pytest.approx(Sa / (k / 2.0), rel=1e-9)
    assert res.base_reactions("RS")[0] == pytest.approx(2.0 * Sa, rel=1e-9)
    ratios = res.modal_participating_mass_ratios("Modal")[0]
    assert ratios["UX"] == pytest.approx(1.0, rel=1e-9)


def test_cqc_two_close_modes_vs_hand():
    """Two-mass shear building: CQC combination of a quantity vs. hand calc."""
    from structstudio.engine.solvers import _correlation, _combine_modes
    omega = np.array([10.0, 11.0])
    rho = _correlation(omega, 0.05, "CQC")
    r = 1.1
    z = 0.05
    expected = 8 * z ** 2 * (1 + r) * r ** 1.5 / ((1 - r ** 2) ** 2 + 4 * z ** 2 * r * (1 + r) ** 2)
    assert rho[0, 1] == pytest.approx(expected)
    vals = np.array([[1.0], [-0.5]])
    assert _combine_modes(vals, rho)[0] == pytest.approx(math.sqrt(1 + 0.25 + 2 * rho[0, 1] * -0.5))


def test_pdelta_beam_column_closed_form():
    """Cantilever column with axial compression: tip deflection amplification."""
    m = steel_beam_model(weight_density=0.0)
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(0, 0, 3)
    m.add_frame(a, b, "B", name="C")
    m.add_load_pattern("G", "Dead")
    m.add_point_load(b, "G", force=(0, 0, -500.0))
    m.add_point_load(b, "P", force=(10, 0, 0))
    m.add_pdelta_case("PD", {"G": 1.0})
    m.add_linear_static_case("LAT", {"P": 1.0}, stiffness_from="PD")
    res = analyze(m)
    EI = 200e6 * 1e-4
    L, Pax, H = 3.0, 500.0, 10.0
    u0 = H * L ** 3 / (3 * EI)
    k = math.sqrt(Pax / EI)
    exact = H / Pax * (math.tan(k * L) / k - L)
    assert res.joint_displacement(b.name, "LAT")[0] == pytest.approx(exact, rel=2e-3)
    assert res.joint_displacement(b.name, "LAT")[0] > u0


# ------------------------------------------------------------------ shells
def test_simply_supported_plate_navier():
    """Square simply supported plate under uniform load (Timoshenko & W-K)."""
    m = Model("plate", "kN m")
    m.add_material("C", E=30e6, nu=0.3)
    t = 0.1
    m.add_shell_section(ShellSection("S", "C", type="Slab", model_type="ShellThin", thickness=t))
    m.set_stories([("L1", 0.0)])
    a = 4.0
    pts = []
    for x, y in ((0, 0), (a, 0), (a, a), (0, a)):
        pts.append(m.add_point(x, y, 0).name)
    m.add_area(pts, "S", name="F", mesh_divisions=(12, 12))
    m.add_load_pattern("Q", "Other")
    m.add_area_load("F", "Q", 10.0)  # 10 kN/m2 gravity
    m.add_linear_static_case("Q")
    am = None
    from structstudio.mesh import build_analysis_model
    am = build_analysis_model(m)
    # simply supported edges: restrain UZ on all boundary nodes (+ in-plane to avoid rigid body)
    for n in am.nodes:
        x, y = n.xyz[0], n.xyz[1]
        if abs(x) < 1e-9 or abs(x - a) < 1e-9 or abs(y) < 1e-9 or abs(y - a) < 1e-9:
            n.restraints[2] = True
        n.restraints[0] = n.restraints[1] = n.restraints[5] = True
    from structstudio.engine import Engine
    eng = Engine(m, am)
    r = eng.run()["Q"]
    centre = min(am.nodes, key=lambda n: (n.xyz[0] - a / 2) ** 2 + (n.xyz[1] - a / 2) ** 2)
    w = -r.displacements[centre.id, 2]
    D = 30e6 * t ** 3 / (12 * (1 - 0.09))
    exact = 0.00406 * 10.0 * a ** 4 / D
    assert w == pytest.approx(exact, rel=2e-2)


def test_shell_cantilever_membrane_tip_deflection():
    """Deep cantilever modelled with shells (membrane action) vs beam theory."""
    m = Model("wall", "kN m")
    m.add_material("C", E=30e6, nu=0.0)
    m.add_shell_section(ShellSection("W", "C", type="Wall", model_type="ShellThin", thickness=0.2))
    L, h = 8.0, 1.0
    p = [m.add_point(0, 0, 0).name, m.add_point(L, 0, 0).name, m.add_point(L, 0, h).name, m.add_point(0, 0, h).name]
    m.add_area(p, "W", name="A", mesh_divisions=(32, 4))
    m.add_load_pattern("P", "Other")
    m.add_linear_static_case("P")
    from structstudio.mesh import build_analysis_model
    from structstudio.engine import Engine
    am = build_analysis_model(m)
    tip = []
    for n in am.nodes:
        if abs(n.xyz[0]) < 1e-9:
            n.restraints[:] = True
        if abs(n.xyz[0] - L) < 1e-9:
            tip.append(n.id)
        n.restraints[1] = n.restraints[3] = n.restraints[5] = True  # keep in-plane problem
    P = 10.0
    for nid in tip:
        am.add_nodal_load("P", nid, np.array([0, 0, -P * 1e3 / len(tip), 0, 0, 0]))  # SI newtons
    r = Engine(m, am).run()["P"]
    w = -np.mean([r.displacements[n, 2] for n in tip])
    E, I, A = 30e6, 0.2 * h ** 3 / 12, 0.2 * h
    G = E / 2.0
    exact = P * L ** 3 / (3 * E * I) + P * L / (G * A * 5 / 6)
    assert w == pytest.approx(exact, rel=3e-2)


def test_thin_shell_dkq_plate_and_beam_continuity():
    """ShellThin = discrete Kirchhoff quadrilateral: Navier plate on a coarse mesh, and a
    beam under a meshed slab keeps a continuous moment diagram (no jumps at the slab
    mesh nodes, the rotations of slab and beam agree: right-hand rule)."""
    import structstudio.engine.elements.shell as sh
    assert sh.THIN_FORMULATION == "DKQ"
    from structstudio.app.templates import new_building
    m = new_building(grid_only=False)
    res = analyze(m)
    case = next(iter(res.cases))
    b = next(f for f in m.frames.values() if f.type == "Beam" and f.story == "Story1")
    st = res.frame_forces(b.name, case)
    M = st.forces[:, 5]
    L = float(st.station[-1])
    # inside the columns (rigid zones of the slab, <= 0.3 m from the ends) the column
    # takes the moment out of the beam/slab: only the span has to be continuous
    jumps = [abs(M[k + 1] - M[k]) for k in range(len(M) - 1)
             if abs(st.station[k + 1] - st.station[k]) < 1e-6 and 0.3 < st.station[k] < L - 0.3]
    assert jumps and max(jumps) < 0.05 * float(np.max(np.abs(M)))


def test_thin_shell_dkq_rotation_sign_with_a_frame():
    """A cantilever beam welded along a cantilever plate strip: both bend together
    (with a wrong rotation sign the beam and the plate would fight and lock)."""
    m = Model("strip", "kN m")
    m.add_material("C", E=30e6, nu=0.0)
    m.add_shell_section(ShellSection("S", "C", type="Slab", model_type="ShellThin", thickness=0.2))
    m.add_frame_section(FrameSection.rectangular("B", "C", depth=0.2, width=0.2))
    m.set_stories([("L1", 0.0)])
    L, w = 4.0, 1.0
    p = [m.add_point(0, 0, 0), m.add_point(L, 0, 0), m.add_point(L, w, 0), m.add_point(0, w, 0)]
    m.add_area([q.name for q in p], "S", name="A", mesh_divisions=(4, 1))
    m.add_frame(p[0], p[1], "B")
    m.add_load_pattern("P", "Other")
    m.add_point_load(p[1].name, "P", (0, 0, -10.0))
    m.add_point_load(p[2].name, "P", (0, 0, -10.0))
    m.add_linear_static_case("P")
    for q in (p[0], p[3]):
        q.restraints = FIXED
    res = analyze(m)
    tip = -res.joint_displacement(p[1].name, "P")[2]
    # beam + plate strip as one cantilever (plate: E w t^3 / 12, beam: E b h^3 / 12)
    EI = 30e6 * (w * 0.2 ** 3 / 12 + 0.2 * 0.2 ** 3 / 12)
    exact = 20.0 * L ** 3 / (3 * EI)
    assert tip == pytest.approx(exact, rel=0.05)


def test_pinned_beam_with_end_offsets_is_a_simple_span_between_the_faces():
    """A pinned end with an end offset: the hinge is at the face (end of the rigid zone),
    so a pin-pin beam spans the clear length as a simple span - zero moment at both
    faces, only sagging moment (w Lc^2 / 8), shear w Lc / 2; the load on the rigid
    zones goes straight into the joints."""
    m = steel_beam_model()
    a = m.add_point(0, 0, 0, restraints=FIXED)
    c = m.add_point(6, 0, 0, restraints=FIXED)
    ta = m.add_point(0, 0, 3.5)
    tc = m.add_point(6, 0, 3.5)
    m.add_frame(a, ta, "B", name="CA")
    m.add_frame(c, tc, "B", name="CC")
    pin = (False, False, False, False, True, True)
    m.add_frame(ta, tc, "B", name="BM", releases_i=pin, releases_j=pin, end_offset_i=0.2, end_offset_j=0.2)
    m.add_frame_distributed_load("BM", "P", 10, 10)  # gravity (down)
    m.add_linear_static_case("P")
    res = analyze(m)
    st = res.frame_forces("BM", "P")
    M = st.forces[:, 5]
    assert st.station[0] == pytest.approx(0.2) and st.station[-1] == pytest.approx(5.8)  # the faces
    assert M[0] == pytest.approx(0.0, abs=1e-9) and M[-1] == pytest.approx(0.0, abs=1e-9)
    assert M.min() >= -1e-9  # no hogging anywhere
    assert M.max() == pytest.approx(10 * 5.6 ** 2 / 8, rel=1e-3)
    assert abs(st.forces[0, 1]) == pytest.approx(28.0, rel=1e-6)  # w x clear span / 2


def test_pinned_beams_in_a_slab_are_hinged_at_the_column_face():
    """With a slab, the slab inside each column's cross-section moves with the column; the
    hinge of a pinned beam must sit at the column face (first beam piece outside that zone),
    or the beam would act fixed-ended.  Moment zero at both faces, sagging between."""
    from structstudio.app.templates import new_building
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5,), grid_only=False)
    pin = (False, False, False, False, True, True)
    for f in m.frames.values():
        if f.type == "Beam":
            f.releases_i, f.releases_j = pin, pin
            m.add_frame_distributed_load(f.name, "Dead", 10, 10)
    m.add_linear_static_case("Dead")
    res = analyze(m)
    st = res.frame_forces("B1", "Dead")
    M = st.forces[:, 5]
    assert M[0] == pytest.approx(0.0, abs=1e-6) and M[-1] == pytest.approx(0.0, abs=1e-6)
    assert M.max() > 40.0 and M.min() > -0.1 * M.max()
