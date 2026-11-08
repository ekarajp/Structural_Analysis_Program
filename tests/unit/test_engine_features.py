"""Closed-form checks of the CSI features added for the SAP2000 verification set:
ground displacements, joint local axes, temperature loads, elastic foundations,
tension / compression limits, nonprismatic members and tendons."""
import math

import numpy as np
import pytest

from structstudio import Model, analyze
from structstudio.model import FIXED, PINNED, FrameSection
from structstudio.model.loads import LinearStaticCase
from structstudio.model.objects import FrameLoad, GroundDisplacement, Tendon, TendonLoad, TendonPoint

E, A, I = 200e6, 0.01, 1e-4


def beam_model(alpha=0.0):
    m = Model("t", "kN m")
    m.add_material("S", E=E, nu=0.3, weight_density=0.0, type="Steel")
    m.materials["S"].alpha = alpha
    m.add_frame_section(FrameSection.general("B", "S", A=A, I33=I, I22=I, J=1e-6))
    m.add_load_pattern("P", "Other")
    return m


def test_support_settlement_of_a_propped_cantilever():
    m = beam_model()
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(4, 0, 0, restraints=(False, False, True, False, False, False))
    m.add_frame(a, b, "B", name="F")
    b.ground_displacements.append(GroundDisplacement("P", (0.0, 0.0, -0.01)))
    m.add_linear_static_case("P")
    res = analyze(m)
    assert res.joint_displacement(b.name, "P")[2] == pytest.approx(-0.01, abs=1e-12)
    # reaction at the settled support: 3 E I delta / L^3 (upward pull on the beam = downward reaction)
    assert res.joint_reaction(b.name, "P")[2] == pytest.approx(-3 * E * I * 0.01 / 64, rel=1e-9)


def test_settlement_through_a_spring():
    m = beam_model()
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(4, 0, 0, springs=(0, 0, 1000.0))
    m.add_frame(a, b, "B", name="F")
    b.ground_displacements.append(GroundDisplacement("P", (0.0, 0.0, -0.02)))
    m.add_linear_static_case("P")
    res = analyze(m)
    kb = 3 * E * I / 64
    u = -0.02 * 1000.0 / (1000.0 + kb)
    assert res.joint_displacement(b.name, "P")[2] == pytest.approx(u, rel=1e-9)
    assert res.joint_reaction(b.name, "P")[2] == pytest.approx(-1000.0 * (u + 0.02), rel=1e-9)


def test_inclined_roller_with_joint_local_axes():
    # cantilever tip on a roller whose normal is at 45 deg in the X-Z plane: the tip can
    # only move along the incline, so an axial-only load gives an incline-direction motion
    m = beam_model()
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(3, 0, 0, restraints=(False, False, True, False, False, False))
    b.local_axes = (0.0, -45.0, 0.0)
    m.add_frame(a, b, "B", name="F")
    m.add_point_load(b, "P", force=(-100.0, 0, 0))
    m.add_linear_static_case("P")
    res = analyze(m)
    u = res.joint_displacement(b.name, "P")
    n = res.am.nodes[res.am.point_to_node[b.name]].local_axes[2]
    assert float(n @ u[:3]) == pytest.approx(0.0, abs=1e-12)
    r = res.joint_reaction(b.name, "P")[:3]
    assert np.cross(r, n) == pytest.approx(np.zeros(3), abs=1e-9)  # reaction along the local 3 axis


def test_temperature_change_and_gradient_in_a_fixed_beam():
    m = beam_model(alpha=1.2e-5)
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(5, 0, 0, restraints=FIXED)
    m.add_frame(a, b, "B", name="F")
    m.frames["F"].loads.append(FrameLoad("P", "temperature", "1", "Temperature", 0, 1, False, 30.0, 30.0))
    m.add_load_pattern("G", "Other")
    m.frames["F"].loads.append(FrameLoad("G", "temperature", "1", "Gradient2", 0, 1, False, 50.0, 50.0))
    m.add_linear_static_case("P")
    m.add_linear_static_case("G")
    res = analyze(m)
    st = res.frame_forces("F", "P")
    assert st.forces[:, 0] == pytest.approx(-E * A * 1.2e-5 * 30.0, rel=1e-9)  # compression
    g = res.frame_forces("F", "G")
    assert np.abs(g.forces[:, 5]) == pytest.approx(E * I * 1.2e-5 * 50.0, rel=1e-9)  # uniform moment
    assert g.forces[:, 1] == pytest.approx(0.0, abs=1e-6)
    # free beam: no forces, free elongation
    m2 = beam_model(alpha=1.2e-5)
    a = m2.add_point(0, 0, 0, restraints=FIXED)
    b = m2.add_point(5, 0, 0)
    m2.add_frame(a, b, "B", name="F")
    m2.frames["F"].loads.append(FrameLoad("P", "temperature", "1", "Temperature", 0, 1, False, 30.0, 30.0))
    m2.add_linear_static_case("P")
    r2 = analyze(m2)
    assert r2.joint_displacement(b.name, "P")[0] == pytest.approx(1.2e-5 * 30 * 5, rel=1e-9)


def test_beam_on_elastic_foundation_long_beam_point_load():
    # Hetenyi: infinite beam, P at the middle: w0 = P beta / (2 k), beta = (k / 4EI)^1/4
    m = beam_model()
    k = 5000.0
    L = 60.0
    n = 240
    pts = [m.add_point(L * i / n - L / 2, 0, 0, restraints=(False, True, False, True, False, False))
           for i in range(n + 1)]
    pts[0].restraints = (True, True, False, True, False, False)
    for i in range(n):
        m.add_frame(pts[i], pts[i + 1], "B", name=f"F{i}")
        m.frames[f"F{i}"].foundation = (0.0, k, 0.0)
    mid = pts[n // 2]
    m.add_point_load(mid, "P", force=(0, 0, -100.0))
    m.add_linear_static_case("P")
    res = analyze(m)
    beta = (k / (4 * E * I)) ** 0.25
    w0 = 100.0 * beta / (2 * k)
    assert res.joint_displacement(mid.name, "P")[2] == pytest.approx(-w0, rel=2e-5)
    assert res.base_reactions("P")[2] == pytest.approx(100.0, rel=1e-9)


def test_tension_only_brace():
    m = beam_model()
    m.add_frame_section(FrameSection.general("R", "S", A=1.0, I33=1.0, I22=1.0, J=1.0))
    p1 = m.add_point(0, 0, 0, restraints=PINNED)
    p2 = m.add_point(0, 0, 3)
    p3 = m.add_point(4, 0, 0, restraints=PINNED)
    p4 = m.add_point(4, 0, 3)
    for n, (a, b, s) in enumerate([(p1, p2, "R"), (p3, p4, "R"), (p2, p4, "R"), (p1, p4, "B"), (p3, p2, "B")]):
        m.add_frame(a, b, s, name=f"F{n}", releases_i=(False,) * 4 + (True, True),
                    releases_j=(False,) * 4 + (True, True))
    m.frames["F4"].compression_limit = 0.0
    m.frames["F3"].compression_limit = 0.0
    for f in m.frames.values():  # the braces cross without a joint
        f.mesh_at_intersections = False
    m.add_point_load(p2, "P", force=(100.0, 0, 0))
    m.load_cases["P"] = LinearStaticCase("P", {"P": 1.0}, nonlinear=True)
    res = analyze(m)
    p_tension = res.frame_forces("F3", "P").forces[0, 0]
    p_off = res.frame_forces("F4", "P").forces[0, 0]
    assert p_tension == pytest.approx(100.0 * 5 / 4, rel=1e-6)
    assert p_off == pytest.approx(0.0, abs=1e-6)


def test_nonprismatic_cantilever_linear_inertia():
    m = beam_model()
    m.add_frame_section(FrameSection.general("S1", "S", A=A, I33=2 * I, I22=2 * I, J=1e-6))
    m.add_frame_section(FrameSection.general("S2", "S", A=A, I33=I, I22=I, J=1e-6))
    sec = FrameSection.general("NP", "S", A=A, I33=2 * I, I22=2 * I, J=1e-6)
    sec.nonprismatic = [dict(start="S1", end="S2", length=1.0, length_type="Variable", ei33="Linear", ei22="Linear")]
    m.add_frame_section(sec)
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(4, 0, 0)
    m.add_frame(a, b, "NP", name="F")
    m.add_point_load(b, "P", force=(0, 0, -10.0))
    m.add_linear_static_case("P")
    res = analyze(m)
    # tip deflection: integral of M m / EI with I(x) = I (2 - x / L)
    L = 4.0
    xs = np.linspace(0, L, 200001)
    integrand = (10 * (L - xs)) * (L - xs) / (E * I * (2 - xs / L))
    exact = np.trapezoid(integrand, xs)
    assert -res.joint_displacement(b.name, "P")[2] == pytest.approx(exact, rel=1e-8)


def test_straight_eccentric_tendon_gives_uniform_moment():
    m = beam_model()
    a = m.add_point(0, 0, 0, restraints=PINNED)
    b = m.add_point(10, 0, 0, restraints=(False, True, True, True, False, False))
    m.add_frame(a, b, "B", name="F", min_stations=5)
    e = -0.2
    m.tendons["T"] = Tendon("T", [TendonPoint("Start", (0, 0, e)), TendonPoint("Line", (10, 0, e))], area=1e-3,
                            material="S", loads=[TendonLoad("P", force=500.0)])
    m.add_linear_static_case("P")
    res = analyze(m)
    st = res.frame_forces("F", "P")
    assert st.forces[:, 0] == pytest.approx(-500.0, rel=1e-9)
    assert st.forces[:, 5] == pytest.approx(500.0 * -e * -1, rel=1e-6) or \
        st.forces[:, 5] == pytest.approx(500.0 * e * -1, rel=1e-6)
    assert np.abs(st.forces[:, 5]) == pytest.approx(100.0, rel=1e-6)
    assert res.base_reactions("P") == pytest.approx(np.zeros(3), abs=1e-6)


def test_parabolic_tendon_balances_its_load():
    # parabolic drape: equivalent uplift 8 P f / L^2, self-equilibrated, no reactions on a
    # simply supported beam except from the anchors' eccentricity (zero here)
    m = beam_model()
    a = m.add_point(0, 0, 0, restraints=PINNED)
    b = m.add_point(10, 0, 0, restraints=(False, True, True, True, False, False))
    m.add_frame(a, b, "B", name="F", min_stations=11)
    f = 0.3
    m.tendons["T"] = Tendon("T", [TendonPoint("Start", (0, 0, 0)), TendonPoint("ParabolaIntermediate", (5, 0, -f)),
                                  TendonPoint("ParabolaEnd", (10, 0, 0))], area=1e-3, material="S",
                            loads=[TendonLoad("P", force=1000.0)])
    m.add_linear_static_case("P")
    res = analyze(m)
    st = res.frame_forces("F", "P")
    mid = int(np.argmin(np.abs(st.station - 5.0)))
    # mid-span moment = P f (the tendon force times the drape), hogging for an upward load
    assert abs(st.forces[mid, 5]) == pytest.approx(1000.0 * f, rel=2e-3)
    assert res.base_reactions("P") == pytest.approx(np.zeros(3), abs=1e-6)
