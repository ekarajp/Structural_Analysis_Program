"""Nonlinear cross-checks against OpenSees (skipped without openseespy).

Every comparison runs the same model in both programs and compares the
response step by step; agreement is expected to the convergence tolerance.
"""
import numpy as np
import pytest

ops = pytest.importorskip("openseespy.opensees")

from structstudio import Model, analyze  # noqa: E402
from structstudio.engine.nonlinear import materials as M  # noqa: E402
from structstudio.model import FIXED, FrameSection  # noqa: E402
from structstudio.model.nonlinear import (FiberSectionDef, FrameNonlinearProperty, HingeProperty,  # noqa: E402
                                          NonlinearMaterial, TimeHistoryLoad)
from tests.xcheck.nonlinear_opensees import OpenSeesTwin  # noqa: E402


def _history(amp, n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    return amp * (np.sin(2 * np.pi * 3 * t) * t * 3 + 0.3 * rng.standard_normal(n).cumsum() / np.sqrt(n))


MATERIALS = [
    (["Steel01", 1, 420e6, 200e9, 0.02, 0.04, 1.0, 0.04, 1.0], M.Steel01(420e6, 200e9, 0.02, 0.04, 1, 0.04, 1), 0.01),
    (["Steel02", 1, 420e6, 200e9, 0.01, 18, 0.925, 0.15, 0.02, 1, 0.02, 1],
     M.Steel02(420e6, 200e9, 0.01, 18, 0.925, 0.15, 0.02, 1, 0.02, 1), 0.012),
    (["Concrete01", 1, -30e6, -0.002, -6e6, -0.006], M.Concrete01(-30e6, -0.002, -6e6, -0.006), -0.004),
    (["Concrete02", 1, -30e6, -0.002, -6e6, -0.006, 0.1, 3e6, 1.5e9],
     M.Concrete02(-30e6, -0.002, -6e6, -0.006, 0.1, 3e6, 1.5e9), -0.004),
    (["ElasticPP", 1, 200e9, 0.002], M.ElasticPP(200e9, 0.002), 0.006),
    (["Hysteretic", 1, 100., 0.01, 120., 0.03, 80., 0.08, -100., -0.01, -120., -0.03, -80., -0.08, 0.5, 0.4, 0.01,
      0.02, 0.3], M.Hysteretic(100, 0.01, 120, 0.03, 80, 0.08, -100, -0.01, -120, -0.03, -80, -0.08, 0.5, 0.4, 0.01,
                               0.02, 0.3), 0.05),
    (["BoucWen", 1, 0.05, 1000., 2., 0.5, 0.5, 1.0, 0.0, 0.0, 0.0], M.BoucWen(0.05, 1000, 2, 0.5, 0.5, 1.0), 0.05),
]


@pytest.mark.parametrize("cmd,mat,amp", MATERIALS, ids=[m[0][0] for m in MATERIALS])
def test_material_strain_history_matches_opensees(cmd, mat, amp):
    for seed in range(3):
        eps = _history(abs(amp), seed=seed) + (amp if amp < 0 else 0.0) * 0.8
        ops.wipe()
        ops.uniaxialMaterial(*cmd)
        ops.testUniaxialMaterial(cmd[1])
        m = mat.clone()
        for e in eps:
            ops.setStrain(float(e))
            m.set_trial(np.array([e]))
            m.commit()
            s_ref = ops.getStress()
            assert m.stress[0] == pytest.approx(s_ref, rel=1e-9, abs=1e-9 * abs(cmd[2]))
            assert m.tangent[0] == pytest.approx(ops.getTangent(), rel=1e-9, abs=1e-9 * abs(cmd[3]) + 1e-6)


# ----------------------------------------------------------------------------
def _building(beams: str):
    """3D two-storey frame (N, m): fiber columns, rigid diaphragms, joint loads and masses."""
    m = Model("xbld", "N m")
    m.use_csi_frame_defaults()  # OpenSees: members join at the joints
    m.add_material("C", E=30e9, nu=0.2)
    m.add_frame_section(FrameSection.rectangular("C45", "C", depth=0.45, width=0.45))
    m.add_frame_section(FrameSection.general("B", "C", A=0.18, I33=0.0054, I22=0.00135, J=0.003))
    m.set_stories([("L1", 3.5), ("L2", 3.2)])
    m.add_diaphragm("D1")
    xs, ys = (0.0, 6.0, 12.0), (0.0, 5.0)
    for x in xs:
        for y in ys:
            m.add_point(x, y, 0, restraints=FIXED)
    m.add_load_pattern("G")
    m.add_load_pattern("L")
    for z, st, h, fl in ((3.5, "L1", 3.5, 1.0), (6.7, "L2", 3.2, 2.0)):
        for x in xs:
            for y in ys:
                p = m.add_point(x, y, z, diaphragm="D1", mass=(30e3, 30e3, 0))
                m.add_frame((x, y, z - h), p, "C45", name=f"C{x:g}{y:g}{st}")
                m.add_point_load(p, "G", force=(0, 0, -400e3))
                m.add_point_load(p, "L", force=(fl * 1e3, 0, 0))
        for y in ys:
            for x0, x1 in zip(xs[:-1], xs[1:]):
                m.add_frame((x0, y, z), (x1, y, z), "B", name=f"BX{x0:g}{y:g}{st}")
        for x in xs:
            m.add_frame((x, 0, z), (x, 5, z), "B", name=f"BY{x:g}{st}")
    m.add_nonlinear_material(NonlinearMaterial("Core", "Concrete02", {"fpc": -38e6, "epsc0": -0.004, "fpcu": -26e6,
                                                                      "epscu": -0.016, "lam": 0.1, "ft": 3e6,
                                                                      "Ets": 2e9}))
    m.add_nonlinear_material(NonlinearMaterial("Cover", "Concrete02", {"fpc": -30e6, "epsc0": -0.002, "fpcu": -6e6,
                                                                       "epscu": -0.006, "lam": 0.1, "ft": 3e6,
                                                                       "Ets": 2e9}))
    m.add_nonlinear_material(NonlinearMaterial.rebar("Bar", 460e6, 200e9, 0.01))
    m.add_fiber_section(FiberSectionDef("CF", "RCRectangle", frame_section="C45", cover=0.05, core_material="Core",
                                        cover_material="Cover", steel_material="Bar", n_top=3, n_bottom=3, n_side=1,
                                        bar_area=4.91e-4, mesh=(8, 4), GJ=1e10))
    cols = [f for f in m.frames if f.startswith("C")]
    m.add_frame_nonlinear(FrameNonlinearProperty("COL", "CF", "ForceBased", "Lobatto", 5), cols)
    if beams == "hinges":
        k = 6 * 30e9 * 0.0054 / 6.0
        # moderately stiff springs: OpenSees' plain Newton stalls with near-rigid ones
        m.add_hinge_property(HingeProperty("BH", "M3", points=[(0.0, 120e3), (0.05, 120e3 + 0.02 * 0.05 * k)],
                                           hysteresis="Kinematic", drop_after_e=False, rigid_factor=20.0))
        for f in m.frames:
            if f.startswith("BX"):
                m.assign_hinge(f, "BH", 0.0)
                m.assign_hinge(f, "BH", 1.0)
    return m


def _twin_static(res, case_g, geometry="PDelta"):
    eng = res.engine
    dom = eng.nonlinear.states[case_g].dom
    tw = OpenSeesTwin(dom, geometry)
    ops.timeSeries("Linear", 1)
    tw.nodal_pattern(1, 1, eng.asm.load_vector_full("G"))
    tw.solver()
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    assert ops.analyze(10) == 0
    ops.loadConst("-time", 0.0)
    return tw


@pytest.mark.parametrize("beams", ["elastic", "hinges"])
def test_building_pushover_matches_opensees(beams):
    m = _building(beams)
    roof = m.find_point(12, 5, 6.7).name
    m.add_nonlinear_static_case("GRAV", {"G": 1.0}, n_steps=10, tolerance=1e-10, max_iterations=50)
    m.add_nonlinear_static_case("PUSH", {"L": 1.0}, initial_case="GRAV", control="Displacement",
                                monitored_point=roof, monitored_dof="UX", target_displacement=0.12, n_steps=40,
                                tolerance=1e-10, max_iterations=50)
    res = analyze(m, ["GRAV", "PUSH"])
    ours = np.array([r["LoadFactor"] for r in res.pushover_curve("PUSH")[1:]])
    tw = _twin_static(res, "GRAV")
    ops.timeSeries("Linear", 2)
    tw.nodal_pattern(2, 2, res.engine.asm.load_vector_full("L"))
    # OpenSees cannot control a constrained (diaphragm slave) DOF: use the roof master
    # (the load is symmetric, so the diaphragm does not rotate)
    node = [c.master for c in res.am.constraints if c.story == "L2"][0] + 1
    ops.integrator("DisplacementControl", node, 1, 0.12 / 40)
    ops.analysis("Static")
    ref = []
    for _ in range(40):
        if ops.analyze(1) != 0:  # compare where OpenSees converges with plain Newton
            break
        ref.append(ops.getLoadFactor(2))
    ref = np.array(ref)
    assert len(ref) >= 20
    assert np.abs(ours[:len(ref)] - ref).max() <= 1e-6 * np.abs(ref).max()


def test_building_nonlinear_time_history_matches_opensees():
    m = _building("elastic")
    roof = m.find_point(12, 5, 6.7).name
    dt, n = 0.01, 400
    t = np.arange(n + 1) * dt
    rng = np.random.default_rng(7)
    ag = np.zeros_like(t)
    for f, a, ph in zip(rng.uniform(0.5, 6, 20), rng.uniform(0.2, 1, 20), rng.uniform(0, 6, 20)):
        ag += a * np.sin(2 * np.pi * f * t + ph)
    ag *= np.minimum(1, t / 1.0) * 0.3 * 9.81 / np.abs(ag).max()
    m.add_time_history_function("GM", t, ag)
    m.add_nonlinear_static_case("GRAV", {"G": 1.0}, n_steps=10, tolerance=1e-10, max_iterations=50)
    a0, a1 = 0.5, 0.002
    m.add_time_history_case("TH", [TimeHistoryLoad("Acceleration", "UX", "GM", 1.0)], n_steps=n, dt=dt,
                            initial_case="GRAV", mass_coefficient=a0, stiffness_coefficient=a1,
                            rayleigh_stiffness="current", tolerance=1e-11, max_iterations=60, monitor_points=[roof])
    res = analyze(m, ["GRAV", "TH"])
    _, ours = res.time_history("TH", roof, "UX")
    tw = _twin_static(res, "GRAV")
    ops.timeSeries("Path", 3, "-dt", dt, "-values", *ag.tolist())
    ops.pattern("UniformExcitation", 3, 1, "-accel", 3)
    ops.rayleigh(a0, a1, 0.0, 0.0)
    ops.wipeAnalysis()
    tw.solver(1e-10)
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    node = res.am.point_to_node[roof] + 1
    ref = [ops.nodeDisp(node, 1)]
    for _ in range(n):
        if ops.analyze(1, dt) != 0:  # OpenSees' fallbacks would loosen the comparison
            break
        ref.append(ops.nodeDisp(node, 1))
    ref = np.array(ref)
    assert len(ref) >= 150
    assert np.abs(ours[:len(ref)] - ref).max() <= 1e-6 * np.abs(ref).max()


# ----------------------------------------------------------------------------
def _l_frame(kind: str, n: int = 6):
    m = Model("lframe", "N m")
    m.use_csi_frame_defaults()
    m.add_material("S", E=200e9, nu=0.3)
    m.add_frame_section(FrameSection.general("B", "S", A=4e-3, I33=2e-5, I22=8e-6, J=1e-6))
    L = 3.0
    pts = [m.add_point(0, 0, L * k / n, restraints=FIXED if k == 0 else None) for k in range(n + 1)]
    pts += [m.add_point(L * k / n, 0, L) for k in range(1, n + 1)]
    for k in range(2 * n):
        m.add_frame(pts[k], pts[k + 1], "B", name=f"F{k}")
    if kind == "fiber":
        m.add_nonlinear_material(NonlinearMaterial("St", "Steel02", {"Fy": 355e6, "E0": 200e9, "b": 0.01}))
        m.add_fiber_section(FiberSectionDef("FS", "SteelBox", depth=0.2, width=0.12, tf=0.01, tw=0.008,
                                            steel_material="St", GJ=77e9 * 1e-6))
        m.add_frame_nonlinear(FrameNonlinearProperty("NL", "FS", "ForceBased", "Lobatto", 4), list(m.frames))
    m.add_load_pattern("P")
    m.add_point_load(pts[-1], "P", force=(0, 4e3, -6e3), moment=(1e3, 0, 2e3))
    m.add_nonlinear_static_case("NL", {"P": 1.0}, n_steps=20, geometric_nonlinearity="Corotational",
                                tolerance=1e-10, max_iterations=60)
    return m, pts[-1]


@pytest.mark.parametrize("kind", ["elastic", "fiber"])
def test_corotational_3d_large_rotation_matches_opensees(kind):
    """L-shaped cantilever, tip forces and moments, ~1.5 m tip displacement, 0.5 rad rotation."""
    m, tip_pt = _l_frame(kind)
    res = analyze(m, ["NL"])
    tip = res.am.point_to_node[tip_pt.name]
    ours = res._case("NL").history["displacements"][:, tip, :3]
    dom = res.engine.nonlinear.states["NL"].dom
    tw = OpenSeesTwin(dom, "Corotational")
    ops.timeSeries("Linear", 1)
    tw.nodal_pattern(1, 1, res.engine.asm.load_vector_full("P"))
    tw.solver(1e-12, 100)
    ops.integrator("LoadControl", 1 / 20)
    ops.analysis("Static")
    ref = []
    for _ in range(20):
        assert ops.analyze(1) == 0
        ref.append(ops.nodeDisp(tip + 1)[:3])
    ref = np.array(ref)
    assert np.abs(ref).max() > 1.0
    assert np.abs(ours - ref).max() <= 1e-6 * np.abs(ref).max()
