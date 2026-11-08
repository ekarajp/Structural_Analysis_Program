"""Closed-form and consistency checks of the nonlinear engine and seismic tools."""
import math

import numpy as np
import pytest

from structstudio import Model, analyze
from structstudio.engine.nonlinear import materials as M
from structstudio.engine.nonlinear import sections as S
from structstudio.engine.nonlinear.hinges import moment_curvature
from structstudio.model import FIXED, FrameSection
from structstudio.model.nonlinear import (FiberSectionDef, FrameNonlinearProperty, HingeProperty, LinkDOF,
                                          LinkProperty, NonlinearMaterial, TimeHistoryLoad)
from structstudio.seismic import (GroundMotionRecord, asce41_target_displacement, bilinear_asce41,
                                  fit_fragility_mle, incremental_dynamic_analysis, n2_target_displacement,
                                  response_spectrum)


# ------------------------------------------------------------------ sections
def test_fiber_section_plastic_moment_of_rectangle():
    """EPP rectangle b x h: Mp = fy b h^2 / 4, My = fy b h^2 / 6."""
    fy, E, b, h = 250e6, 200e9, 0.2, 0.5
    mat = M.ElasticPP(E, fy / E)
    sec = S.FiberSection3D([S.patch_rect(mat, 200, 1, -h / 2, -b / 2, h / 2, b / 2)], GJ=1.0)
    mc = moment_curvature(sec, 0.0, "z", kappa_max=60 * fy / E / (h / 2), n=300)
    assert np.abs(mc["moment"]).max() == pytest.approx(fy * b * h * h / 4, rel=2e-3)
    k_y = fy / E / (h / 2)
    i = int(np.argmin(np.abs(mc["curvature"] - 0.5 * k_y)))
    assert abs(mc["moment"][i]) == pytest.approx(E * b * h ** 3 / 12 * mc["curvature"][i], rel=1e-4)


def test_integration_rules_are_exact_for_polynomials():
    for bi, deg in ((S.lobatto(5), 7), (S.legendre(3), 5), (S.radau(3), 4)):
        for p in range(deg + 1):
            assert float(bi.w @ bi.xi ** p) == pytest.approx(1.0 / (p + 1), rel=1e-12)
    hr = S.hinge_radau(0.3, 0.4, 5.0)
    assert hr.w.sum() == pytest.approx(1.0)
    assert hr.w[0] * 5.0 == pytest.approx(0.3) and hr.w[-1] * 5.0 == pytest.approx(0.4)


# -------------------------------------------------------------------- hinges
def _cantilever(hinge_points, hysteresis="Kinematic"):
    m = Model("cant", "kN m")
    m.add_material("S", E=200e6, nu=0.3)
    m.add_frame_section(FrameSection.general("B", "S", A=0.01, I33=2e-4, I22=1e-4, J=1e-5))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(0, 0, 3.0)
    m.add_frame(a, b, "B", name="COL")
    m.add_hinge_property(HingeProperty("H", "M3", points=hinge_points, hysteresis=hysteresis, drop_after_e=False))
    m.assign_hinge("COL", "H", 0.0)
    m.add_load_pattern("L")
    m.add_point_load(b, "L", force=(1.0, 0, 0))
    return m, b


def test_elastic_perfectly_plastic_hinge_cantilever_plateau():
    My = 120.0  # kN m
    m, top = _cantilever([(0.0, My), (0.2, My)])
    m.add_nonlinear_static_case("PUSH", {"L": 1.0}, control="Displacement", monitored_point=top.name,
                                monitored_dof="UX", target_displacement=0.2, n_steps=40,
                                geometric_nonlinearity="None")
    res = analyze(m, ["PUSH"])
    pc = res.pushover_curve("PUSH")
    assert pc[-1]["BaseForce"] == pytest.approx(My / 3.0, rel=1e-6)
    E, I, L = 200e6, 2e-4, 3.0
    k_el = 3 * E * I / L ** 3
    first = pc[1]
    assert first["BaseForce"] / first["Displacement"] == pytest.approx(k_el, rel=2e-3)  # rigid-plastic hinge
    h = res.hinge_results("PUSH")[0]
    # plastic rotation = (top displacement - elastic part) / L
    assert h["PlasticDeformation"] == pytest.approx((0.2 - (My / 3) / k_el) / L, rel=1e-3)


@pytest.mark.parametrize("hysteresis", ["Kinematic", "Takeda", "Elastic"])
def test_vertical_hinge_drop_is_released_at_fixed_displacement(hysteresis):
    """CSI-style C-D drop: the capacity curve steps down vertically to the D strength,
    the regularised option gives a sloped drop instead."""
    My = 120.0
    pts = [(0.0, My), (0.02, 1.1 * My), (0.02, 0.3 * My), (0.06, 0.3 * My)]
    curves = {}
    for mode in ("Redistribute", "Regularised"):
        m, top = _cantilever(pts, hysteresis)
        m.add_nonlinear_static_case("PUSH", {"L": 1.0}, control="Displacement", monitored_point=top.name,
                                    monitored_dof="UX", target_displacement=0.15, n_steps=17,
                                    geometric_nonlinearity="None", hinge_unloading=mode)
        res = analyze(m, ["PUSH"])
        curves[mode] = (res.pushover_curve("PUSH"), res._case("PUSH").history)
    pc, h = curves["Redistribute"]
    d = np.array([r["Displacement"] for r in pc])
    V = np.array([r["BaseForce"] for r in pc])
    assert h["failed"] is None and h["drop_substeps"] >= 2
    same = np.flatnonzero(np.abs(np.diff(d)) < 1e-12)  # vertical segment
    assert len(same) == h["drop_substeps"]
    i0, i1 = same[0], same[-1] + 1
    assert V[i0] == pytest.approx(1.1 * My / 3.0, rel=1e-6)  # held at C until released
    assert V[i1] == pytest.approx(0.3 * My / 3.0, rel=1e-6)  # D strength at the same roof displacement
    assert np.allclose(V[i1:], 0.3 * My / 3.0, rtol=1e-6)
    pr, hr = curves["Regularised"]
    dr = np.array([r["Displacement"] for r in pr])
    assert hr["drop_substeps"] == 0 and np.all(np.diff(dr) > 0)  # no vertical segment
    Vr = np.array([r["BaseForce"] for r in pr])
    assert Vr[-1] == pytest.approx(0.3 * My / 3.0, rel=1e-6) and Vr.max() <= 1.1 * My / 3.0 * (1 + 1e-9)


def test_pdelta_reduces_plateau_by_P_over_H():
    My, P = 120.0, 400.0
    m, top = _cantilever([(0.0, My), (0.3, My)])
    m.add_load_pattern("G")
    m.add_point_load(top, "G", force=(0, 0, -P))
    m.add_nonlinear_static_case("GRAV", {"G": 1.0}, n_steps=2)
    m.add_nonlinear_static_case("PUSH", {"L": 1.0}, initial_case="GRAV", control="Displacement",
                                monitored_point=top.name, monitored_dof="UX", target_displacement=0.15, n_steps=30)
    res = analyze(m, ["GRAV", "PUSH"])
    last = res.pushover_curve("PUSH")[-1]
    # leaning-column P-Delta: H = (My - P * delta) / L
    assert last["BaseForce"] == pytest.approx((My - P * last["Displacement"]) / 3.0, rel=2e-3)


# ---------------------------------------------------------------- dynamics
def _sdof_model(k=4e3, mass=10.0):
    m = Model("sdof", "kN m")
    m.add_material("S", E=200e6, nu=0.3)
    L = 3.0
    I = k * L ** 3 / (3 * 200e6)
    m.add_frame_section(FrameSection.general("B", "S", A=1.0, I33=I, I22=I, J=I))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(0, 0, L, mass=(mass, mass, 0))
    m.add_frame(a, b, "B", name="COL")
    return m, b


def test_linear_direct_and_modal_time_history_agree_with_exact_sdof():
    m, top = _sdof_model()
    t = np.arange(0, 6, 0.005)
    ag = np.sin(2 * np.pi * 1.3 * t) * np.exp(-0.3 * t) * 3.0
    m.add_time_history_function("F", t, ag)
    m.add_modal_case("Modal", max_modes=2)
    common = dict(n_steps=len(t) - 1, dt=0.005, damping="Modal", damping_ratio=0.05, modal_case="Modal",
                  monitor_points=[top.name])
    m.add_time_history_case("MOD", [TimeHistoryLoad("Acceleration", "UX", "F", 1.0)], method="Modal",
                            nonlinear=False, **common)
    m.add_time_history_case("DIR", [TimeHistoryLoad("Acceleration", "UX", "F", 1.0)], nonlinear=False, **common)
    res = analyze(m, ["Modal", "MOD", "DIR"])
    _, u_mod = res.time_history("MOD", top.name, "UX")
    _, u_dir = res.time_history("DIR", top.name, "UX")
    rec = GroundMotionRecord(0.005, ag / 9.80665)
    w = math.sqrt(4e3 / 10.0)
    sd = response_spectrum(rec, [2 * math.pi / w], 0.05)["Sd"][0]
    assert np.abs(u_mod).max() == pytest.approx(sd, rel=1e-3)
    assert np.abs(u_dir - u_mod).max() <= 5e-3 * np.abs(u_mod).max()


def test_response_spectrum_matches_fine_newmark():
    rng = np.random.default_rng(3)
    dt = 0.01
    a = rng.standard_normal(1500) * np.hanning(1500) * 0.2
    rec = GroundMotionRecord(dt, a)
    T = np.array([0.1, 0.5, 1.0, 2.0])
    sp = response_spectrum(rec, T, 0.05)
    for Ti, sd in zip(T, sp["Sd"]):
        w = 2 * math.pi / Ti
        n = 20
        h = dt / n
        ag = np.interp(np.arange(len(a) * n) * h, rec.times, a * 9.80665)
        u = v = 0.0
        acc = -ag[0]
        umax = 0.0
        beta, gam = 0.25, 0.5
        k_eff = w * w + gam / (beta * h) * 2 * 0.05 * w + 1 / (beta * h * h)
        for k in range(1, len(ag)):
            p = -ag[k] + (u / (beta * h * h) + v / (beta * h) + (0.5 / beta - 1) * acc) \
                + 2 * 0.05 * w * (gam / (beta * h) * u + (gam / beta - 1) * v + h * (gam / (2 * beta) - 1) * acc)
            un = p / k_eff
            an = (un - u) / (beta * h * h) - v / (beta * h) - (0.5 / beta - 1) * acc
            v = v + h * ((1 - gam) * acc + gam * an)
            u, acc = un, an
            umax = max(umax, abs(u))
        assert sd == pytest.approx(umax, rel=2e-3)


def test_link_wen_isolator_runs_and_dissipates():
    m = Model("iso", "kN m")
    p = m.add_point(0, 0, 0, restraints=(False, True, True, True, True, True), mass=(200.0, 0, 0))
    m.add_link_property(LinkProperty("ISO", [LinkDOF("U1", "Wen", k=8e3, yield_force=150.0, post_yield_ratio=0.1,
                                                     exponent=2.0)]))
    m.add_link(p, None, "ISO")
    t = np.arange(0, 8, 0.01)
    m.add_time_history_function("F", t, 0.4 * np.sin(2 * np.pi * 0.7 * t) * np.minimum(1, t / 2))
    m.add_time_history_case("TH", [TimeHistoryLoad("Acceleration", "UX", "F", 9.80665)], n_steps=len(t) - 1,
                            dt=0.01, damping="None", monitor_points=[p.name])
    res = analyze(m, ["TH"])
    _, br = res.base_reaction_history("TH")
    assert np.abs(br[:, 0]).max() < 150.0 * 1.2 + 0.1 * 8e3 * 0.5  # bounded by the Wen yield surface


@pytest.mark.parametrize("law", ["Wen", "ElasticPP"])
def test_fna_matches_direct_integration_on_isolated_stick(law):
    """Base-isolated 6-storey stick: FNA with 2 eigenvectors + Ritz vectors against
    Newmark direct integration of the full model (mass-proportional damping)."""
    m = Model("iso", "kN m")
    m.add_material("S", E=200e6, nu=0.3)
    m.add_frame_section(FrameSection.general("C", "S", A=0.5, I33=0.05, I22=0.05, J=1e-2))
    base = m.add_point(0, 0, 0, restraints=(False, True, True, True, True, True), mass=(80.0, 0, 0))
    pts, prev = [base], base
    for s in range(1, 7):
        p = m.add_point(0, 0, 3.0 * s, restraints=(False, True, False, True, False, True), mass=(60.0, 0, 0))
        m.add_frame(prev, p, "C", name=f"C{s}")
        prev = p
        pts.append(p)
    kw = dict(k=6e3, yield_force=60.0) if law == "ElasticPP" else dict(k=6e3, yield_force=60.0,
                                                                        post_yield_ratio=0.1, exponent=2.0)
    m.add_link_property(LinkProperty("ISO", [LinkDOF("U2", law, **kw)]))  # local 2 = global X
    m.add_link(base, None, "ISO")
    t = np.arange(0, 8, 0.005)
    m.add_time_history_function("F", t, 0.4 * np.sin(2 * np.pi * 0.6 * t) * np.minimum(1, t / 2)
                                + 0.2 * np.sin(2 * np.pi * 2.3 * t))
    m.add_modal_case("Modal", max_modes=2)
    common = dict(n_steps=len(t) - 1, dt=0.005, damping="Rayleigh", mass_coefficient=0.3, stiffness_coefficient=0.0,
                  monitor_points=[pts[0].name, pts[-1].name], geometric_nonlinearity="None")
    load = [TimeHistoryLoad("Acceleration", "UX", "F", 9.80665)]
    m.add_time_history_case("DI", load, **common)
    m.add_time_history_case("FNA", load, method="FNA", modal_case="Modal", **common)
    res = analyze(m, ["Modal", "DI", "FNA"])
    h = res._case("FNA").history
    assert h["failed"] is None and h["fna"]["n_eigen"] == 2 and h["fna"]["n_vectors"] > 2
    for p in (pts[0], pts[-1]):
        _, ua = res.time_history("DI", p.name, "UX")
        _, ub = res.time_history("FNA", p.name, "UX")
        assert np.abs(ua - ub).max() < 3e-3 * np.abs(ua).max()
    assert h["links"]["force"].shape[0] == len(t)


# -------------------------------------------------------------- persistence
def test_nonlinear_definitions_survive_json_and_unit_conversion():
    m, top = _cantilever([(0.0, 120.0), (0.2, 150.0)])
    m.add_nonlinear_material(NonlinearMaterial.rebar("R", 460e3, 200e6))
    m.add_nonlinear_material(NonlinearMaterial.unconfined_concrete("C", 30e3))
    m.add_fiber_section(FiberSectionDef("F", "RCRectangle", depth=0.5, width=0.3, core_material="C",
                                        steel_material="R", n_top=3, n_bottom=3, bar_area=4.9e-4))
    m.add_nonlinear_static_case("PUSH", {"L": 1.0}, control="Displacement", monitored_point=top.name,
                                target_displacement=0.1, n_steps=10, geometric_nonlinearity="None")
    m2 = Model.from_json(m.to_json())
    assert m2.to_json() == m.to_json()
    m3 = m.convert_units("N mm")
    assert m3.hinge_properties["H"].points[1][1] == pytest.approx(150.0 * 1e6)
    assert m3.nonlinear_materials["R"].params["Fy"] == pytest.approx(460.0)
    assert m3.fiber_sections["F"].bar_area == pytest.approx(490.0)
    r1 = analyze(m, ["PUSH"]).pushover_curve("PUSH")[-1]
    r3 = analyze(m3, ["PUSH"]).pushover_curve("PUSH")[-1]
    assert r3["BaseForce"] == pytest.approx(r1["BaseForce"] * 1e3, rel=1e-9)


# ------------------------------------------------------------ seismic tools
def test_bilinear_recovers_a_bilinear_curve():
    d = np.linspace(0, 0.3, 301)
    Ke, Vy, a = 1000.0, 50.0, 0.05
    dy = Vy / Ke
    v = np.where(d <= dy, Ke * d, Vy + a * Ke * (d - dy))
    bl = bilinear_asce41(d, v, 0.3)
    assert bl["Vy"] == pytest.approx(Vy, rel=1e-6)
    assert bl["Ke"] == pytest.approx(Ke, rel=1e-6)
    assert bl["alpha1"] == pytest.approx(a, rel=1e-5)


def test_asce41_elastic_long_period_equals_spectral_displacement():
    d = np.linspace(0, 0.5, 501)
    v = 2000.0 * d  # elastic, stiff enough to remain elastic
    r = asce41_target_displacement(d, v, weight=10000.0, Sa_of_T=lambda T: 0.3, Ti=1.5, n_stories=1)
    assert r["C1"] == 1.0 and r["C2"] == 1.0
    assert r["target_displacement"] == pytest.approx(0.3 * 1.5 ** 2 / (4 * math.pi ** 2) * 9.80665, rel=1e-9)


def test_n2_equal_displacement_rule_for_long_periods():
    d = np.linspace(0, 1.0, 1001)
    F = np.minimum(500.0 * d, 50.0)  # EPP SDOF, Gamma = 1
    r = n2_target_displacement(d, F, masses=[10.0], phi=[1.0], Se_of_T=lambda T: 0.5, Tc=0.5)
    T = 2 * math.pi * math.sqrt(10.0 / 500.0)
    assert r["T_star"] == pytest.approx(T, rel=1e-3)
    assert r["target_displacement"] == pytest.approx(0.5 * 9.80665 * (T / (2 * math.pi)) ** 2, rel=2e-3)


def test_fragility_mle_recovers_parameters():
    rng = np.random.default_rng(0)
    theta, beta = 0.8, 0.45
    im = np.linspace(0.1, 2.5, 15)
    n = np.full(len(im), 400)
    from scipy import stats
    z = rng.binomial(n, stats.norm.cdf(np.log(im / theta) / beta))
    f = fit_fragility_mle(im, z, n)
    assert f["theta"] == pytest.approx(theta, rel=0.03)
    assert f["beta"] == pytest.approx(beta, rel=0.08)


def test_ida_runs_and_flags_collapse():
    m, top = _sdof_model(k=2e3, mass=20.0)
    m.add_hinge_property(HingeProperty("H", "M3", points=[(0.0, 30.0), (0.05, 30.0), (0.08, 5.0)],
                                       hysteresis="Takeda"))
    m.assign_hinge("COL", "H", 0.0)
    rng = np.random.default_rng(4)
    recs = []
    for s in range(2):
        a = rng.standard_normal(800) * np.hanning(800)
        recs.append(GroundMotionRecord(0.01, 0.3 * a / np.abs(a).max(), name=f"R{s}"))
    rows = incremental_dynamic_analysis(m, recs, [0.05, 0.3, 1.5], im="PGA",
                                        case_template=dict(damping="Rayleigh", damping_ratio=0.05,
                                                           damping_periods=(0.6, 0.1), monitor_points=[top.name]),
                                        edp=lambda r, c: float(np.abs(r.time_history(c, top.name, "UX")[1]).max()) / 3.0,
                                        collapse_edp=0.05)
    assert len(rows) == 6
    for rec in ("R0", "R1"):
        e = [r["EDP"] for r in rows if r["record"] == rec]
        assert e[0] < e[1] or not np.isfinite(e[1])


def test_corotational_cantilever_matches_elastica():
    """Tip-loaded cantilever, P L^2 / EI = 1 and 2 (Bisshopp & Drucker 1945)."""
    E, I, L, n = 200e9, 1e-5, 2.0, 20
    m = Model("elastica", "N m")
    m.add_material("S", E=E, nu=0.3)
    m.add_frame_section(FrameSection.general("B", "S", A=1.0, I33=I, I22=I, J=2 * I))
    pts = [m.add_point(L * k / n, 0, 0, restraints=FIXED if k == 0 else None) for k in range(n + 1)]
    for k in range(n):
        m.add_frame(pts[k], pts[k + 1], "B", name=f"F{k}")
    m.add_load_pattern("P")
    m.add_point_load(pts[-1], "P", force=(0, 0, -E * I / L ** 2))
    m.add_nonlinear_static_case("NL", {"P": 2.0}, n_steps=20, geometric_nonlinearity="Corotational")
    res = analyze(m, ["NL"])
    h = res._case("NL").history
    tip = res.am.point_to_node[pts[-1].name]
    for step, (u, w) in ((9, (0.05643, 0.30172)), (19, (0.16064, 0.49346))):
        d = h["displacements"][step, tip]
        assert -d[0] / L == pytest.approx(u, rel=3e-3)
        assert -d[2] / L == pytest.approx(w, rel=1e-3)


def _rc_column(P_kN: float, hinge: str):
    m = Model("pmm", "kN m")
    m.add_material("C", E=27e6, nu=0.2, fc=30e3)
    m.add_frame_section(FrameSection.rectangular("C50", "C", depth=0.5, width=0.5))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(0, 0, 3.0)
    m.add_frame(a, b, "C50", name="COL")
    m.add_nonlinear_material(NonlinearMaterial.unconfined_concrete("Conc", 30e3, ft_ratio=0.0))
    m.add_nonlinear_material(NonlinearMaterial.rebar("Bar", 460e3, 200e6, 0.005))
    m.add_fiber_section(FiberSectionDef("F", "RCRectangle", frame_section="C50", cover=0.05, core_material="Conc",
                                        steel_material="Bar", n_top=4, n_bottom=4, n_side=2, bar_area=4.91e-4,
                                        mesh=(12, 6)))
    if hinge == "pmm":
        m.add_hinge_property(HingeProperty("PMM", "PMM", fiber_section="F", hinge_length=0.25))
        m.assign_hinge("COL", "PMM", 0.0)
    else:
        m.add_frame_nonlinear(FrameNonlinearProperty("HR", "F", "ForceBased", "HingeRadau", hinge_length_i=0.25),
                              ["COL"])
    m.add_load_pattern("G")
    m.add_load_pattern("L")
    m.add_point_load(b, "G", force=(0, 0, -P_kN))
    m.add_point_load(b, "L", force=(1.0, 0, 0))
    m.add_nonlinear_static_case("GRAV", {"G": 1.0}, n_steps=2, geometric_nonlinearity="None")
    m.add_nonlinear_static_case("PUSH", {"L": 1.0}, initial_case="GRAV", control="Displacement",
                                monitored_point=b.name, target_displacement=0.06, n_steps=30,
                                geometric_nonlinearity="None")
    return m


def test_pmm_fiber_hinge_equals_plastic_hinge_integration_and_feels_axial_load():
    r_h = analyze(_rc_column(500.0, "pmm"), ["GRAV", "PUSH"])
    r_e = analyze(_rc_column(500.0, "explicit"), ["GRAV", "PUSH"])
    vh = np.array([r["BaseForce"] for r in r_h.pushover_curve("PUSH")])
    ve = np.array([r["BaseForce"] for r in r_e.pushover_curve("PUSH")])
    assert np.abs(vh - ve).max() <= 1e-9 * np.abs(ve).max()
    hr = r_h.hinge_results("PUSH")[0]
    assert hr["DOF"] == "PMM" and hr["P"] == pytest.approx(-500.0, rel=1e-6)
    assert hr["M3"] == pytest.approx(vh[-1] * 3.0, rel=1e-6)  # base moment = V H (no P-Delta)
    assert hr["PlasticDeformation"] > 0.01
    # more compression (below the balance point) -> larger peak moment, faster degradation
    r_hi = analyze(_rc_column(2500.0, "pmm"), ["GRAV", "PUSH"])
    v_hi = np.array([r["BaseForce"] for r in r_hi.pushover_curve("PUSH")])
    assert v_hi.max() > 1.2 * vh.max()
    assert v_hi[-1] < v_hi.max()
