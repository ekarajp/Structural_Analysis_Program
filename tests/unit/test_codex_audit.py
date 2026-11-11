"""Regressions for the Codex comprehensive audit 2026-11-10 (CA01-CA26) and the
in-house audit items (O1-O14).  Every expected value comes from an independent
closed form (quoted in the test), not from the program; the models are copies
of the audit's input files (tests/data/codex, unchanged)."""
import copy
import math
import os

import numpy as np
import pytest

from structstudio import Model, analyze
from structstudio.engine.kernel import SciPyKernel
from structstudio.model import ModelError

D = os.path.join(os.path.dirname(__file__), "..", "data", "codex")


def load(name):
    return Model.from_json(os.path.join(D, name + ".ssj.json"))


def run(m, **kw):
    return analyze(m, kernel=SciPyKernel(), **kw)


def code_of(fn):
    with pytest.raises(ModelError) as e:
        fn()
    return e.value.code


# ------------------------------------------------------------------ CA01 / CA22
def test_ca01_pdelta_beyond_buckling_is_refused():
    """P = 1.1 Pcr: K_e + K_g has a negative eigenvalue - no results (both kernels)."""
    assert code_of(lambda: run(load("PD_1.1"))) == "NEGATIVE_STIFFNESS"
    r = run(load("PD_0.8"))  # below buckling: runs
    assert r.cases


def test_ca22_thermal_axial_force_enters_the_geometric_stiffness():
    """Restrained bar heated 100 C: N = -E A alpha dT = -4.8e6 N; cantilever column 3 m,
    k0 = 12 E I / L^3, K_g = 36 N / (30 L) (consistent geometric stiffness of one element)."""
    r = run(load("thermal_fixed"))
    k0 = 12 * 200e9 * 8e-5 / 27
    P = -200e9 * 0.02 * 1.2e-5 * 100
    keff = k0 + 36 * P / (30 * 3)
    top = [p for p in r.model.points.values() if p.z > 1][0].name
    assert r.joint_displacement(top, "PD")[0] == pytest.approx(1000 / keff, rel=1e-6)
    r = run(load("thermal_free"))  # free expansion: no axial force, P-Delta = linear
    top = [p for p in r.model.points.values() if p.z > 1][0].name
    assert r.joint_displacement(top, "PD")[0] == pytest.approx(1000 / k0, rel=1e-9)


# ------------------------------------------------------------------ CA02
def test_ca02_story_drift_from_signed_responses():
    r = run(load("N02"))  # constant spectrum: drift of each mode, then combined
    d = {(x["Story"], x["Direction"]): x["Drift"] for x in r.story_drifts("RS")}
    assert d[("L2", "X")] == pytest.approx(5.54632480e-5, rel=1e-6)
    m = load("envelope-drift")  # two static cases, 1 mm per story at different times
    r = run(m)
    env = [c for c in m.load_combinations][0]
    d = {(x["Story"], x["Direction"]): x["Drift"] for x in r.story_drifts(env)}
    assert d[("L2", "X")] == pytest.approx(1 / 3000, rel=1e-9)
    r = run(load("RS_equal_envelopes"))  # joint peaks equal: drift is NOT zero
    d = {(x["Story"], x["Direction"]): x["Drift"] for x in r.story_drifts("RS")}
    assert d[("L2", "X")] == pytest.approx(0.01490711985, rel=1e-8)


# ------------------------------------------------------------------ CA03 / CA06 / CA07
def test_ca03_modal_case_uses_its_mass_source():
    """T = 2 pi sqrt(m / k): 1,000 kg vs 4,000 kg -> ratio 2."""
    r = run(load("N01"))
    assert r.cases["LIGHT"].modal.periods[0] == pytest.approx(0.07450941199, rel=1e-9)
    assert r.cases["HEAVY"].modal.periods[0] == pytest.approx(0.14901882399, rel=1e-9)


def test_ca06_negative_net_mass_is_refused():
    assert code_of(lambda: run(load("I_negative_mass"))) == "NEGATIVE_MASS"
    assert code_of(lambda: run(load("I_negative_density"))) == "INVALID_MATERIAL"


def test_ca06_negative_added_mass_with_positive_net_mass_runs():
    """Source "heavy": 4,000 kg from the loads plus -1,000 kg of added mass = 3,000 kg net:
    T = T_light sqrt(3) (allowed - only a negative net mass is refused)."""
    m = load("N01")
    m.mass_sources["heavy"].include_added_mass = True
    m.points["P2"].mass = (-1000.0, -1000.0, -1000.0)
    r = run(m, cases=["HEAVY"])
    assert r.cases["HEAVY"].modal.periods[0] == pytest.approx(0.07450941199 * math.sqrt(3.0), rel=1e-9)


def test_ca07_lump_lateral_mass_at_story_levels():
    """A 1,000 kg joint at z = 1.5 m between levels 0 and 6 m (no joint at 3 m): 3/4 to the
    base (restrained, inactive), 1/4 to the top - total and first moment kept."""
    m = load("N02")
    m.points["P2"].z = 1.5
    m.points["P2"].story = None
    out = {}
    for lump in (False, True):
        mm = copy.deepcopy(m)
        mm.mass_sources["MsSrc1"].lump_at_stories = lump
        r = run(mm)
        out[lump] = r.engine.asm.mass_full().reshape(-1, 6)[:, 0]
    assert np.allclose(out[False], [0, 1000, 1000])
    assert np.allclose(out[True], [750, 0, 1250])


# ------------------------------------------------------------------ CA04 / CA05
def test_ca04_combination_of_one_spectrum_keeps_its_base_shear():
    r = run(load("N03"))
    assert r.base_reactions("RS")[0] == pytest.approx(math.hypot(1000, 2000), rel=1e-9)
    combo = list(r.model.load_combinations)[0]
    assert r.base_reactions(combo)[0] == pytest.approx(math.hypot(1000, 2000), rel=1e-9)


def test_ca05_story_shear_includes_walls():
    r = run(load("N06"))  # cantilever wall, 1,000 N at the top
    rows = r.story_shears("P")
    assert rows[0]["VX"] == pytest.approx(1000.0, rel=1e-9)


# ------------------------------------------------------------------ CA08 / CA09
def test_ca08_spectrum_accidental_torsion():
    """Diaphragm 4 m wide, excitation U2, e = 0.15 x 4 = 0.6 m: torsion = sum m a e =
    2000 N x 0.6 m = 1200 N m added to the base MZ (4000 -> 5200)."""
    m = load("RS_eccentricity")
    vals = {}
    for e in (0.0, 0.15):
        mm = copy.deepcopy(m)
        mm.load_cases["RS"].eccentricity_ratio = e
        vals[e] = run(mm).cases["RS"].spectrum_base
    assert vals[0.15][5] - vals[0.0][5] == pytest.approx(1200.0, rel=1e-9)
    assert vals[0.15][1] == pytest.approx(vals[0.0][1], rel=1e-12)  # no extra shear


def test_ca09_unknown_or_unimplemented_spectrum_options_are_refused():
    m = load("RS_eccentricity")
    for value, code in (("TYPO", "INVALID_COMBINATION"), ("GMC", "INVALID_RIGID_RESPONSE")):
        mm = copy.deepcopy(m)
        mm.load_cases["RS"].modal_combination = value
        assert code in [e.code for e in mm.validate()]
        assert code_of(lambda: run(mm)) == code
    mm = copy.deepcopy(m)
    mm.load_cases["RS"].directional_combination = "CQC3"
    assert code_of(lambda: run(mm)) == "NOT_IMPLEMENTED"


def test_ca09_rigid_response_follows_gupta():
    """alpha = ln(f/f1)/ln(f2/f1): with f1, f2 below every frequency all modes are rigid
    and the response is the algebraic sum of the modal responses."""
    from structstudio.engine.solvers import _rigid_fractions
    m = load("RS_eccentricity")
    m.load_cases["RS"].eccentricity_ratio = 0.0
    r0 = run(m)
    f = r0.cases["MOD"].modal.omega / (2 * np.pi)
    m.load_cases["RS"].rigid_response = (float(f.min()) / 4, float(f.min()) / 2)
    r = run(m)
    md = r.cases["RS"].modal_displacements[0]  # (modes, nodes, 6) of the one direction
    assert np.allclose(_rigid_fractions(r.cases["MOD"].modal.omega, m.load_cases["RS"]), 1.0)
    assert np.allclose(r.cases["RS"].displacements, np.abs(md.sum(axis=0)), rtol=1e-9, atol=1e-15)


# ------------------------------------------------------------------ CA10 - CA13
@pytest.mark.parametrize("name,code", [
    ("I_unknown_auto_wind", "UNSUPPORTED_AUTO_LOAD"), ("I_invalid_nu", "INVALID_MATERIAL"),
    ("I_negative_I", "INVALID_SECTION"), ("I_negative_spring", "NEGATIVE_SPRING"),
    ("I_unsorted_spectrum", "INVALID_FUNCTION"), ("I_zero_modes", "INVALID_MODES"),
    ("I_invalid_damping", "INVALID_DAMPING"), ("I_case_cycle", "CASE_CYCLE")])
def test_ca10_ca11_ca12_meaningless_input_is_refused_before_solving(name, code):
    m = load(name)
    assert code in [e.code for e in m.validate()]
    assert code_of(lambda: run(m)) == code


def test_ca11_valid_controls_still_run():
    for name in ("N01", "N02", "N03", "N06"):
        m = load(name)
        assert m.validate() == []


def test_ca13_future_schema_refused_unknown_data_kept_and_blocked():
    d = load("N01").to_dict()
    d["schema_version"] = "99.0"
    d["cables"] = {"Cable1": {"axial_stiffness": 1e8}}
    assert code_of(lambda: Model.from_dict(d)) == "FUTURE_SCHEMA"
    d["schema_version"] = "2.2"
    m = Model.from_dict(d)
    assert "cables" in m.to_dict()  # kept as read
    assert "UNKNOWN_DATA" in [e.code for e in m.validate()]


# ------------------------------------------------------------------ CA14
def test_ca14_the_window_frame_fits_the_screen():
    """fit_to_screen sizes the client area so that client + native frame fit (frame 8 px
    borders, 31 px title bar as on Windows 11)."""
    from PySide6 import QtCore
    from structstudio.app import fit

    class Stub:
        def __init__(self):
            self.g = QtCore.QRect(2560 - 50, 1400 - 50, 2960, 1800)

        def frameGeometry(self):
            return QtCore.QRect(self.g.x() - 8, self.g.y() - 31, self.g.width() + 16, self.g.height() + 39)

        def geometry(self):
            return QtCore.QRect(self.g)

        def width(self):
            return self.g.width()

        def height(self):
            return self.g.height()

        def sizeHint(self):
            return QtCore.QSize(400, 300)

        def resize(self, w, h):
            self.g.setSize(QtCore.QSize(w, h))

        def move(self, x, y):  # frame position
            self.g.moveTo(x + 8, y + 31)

    avail = QtCore.QRect(0, 0, 2560, 1400)
    old = fit.available_rect
    fit.available_rect = lambda w: avail.adjusted(fit.MARGIN, fit.MARGIN, -fit.MARGIN, -fit.MARGIN)
    try:
        s = Stub()
        fit.fit_to_screen(s)
        assert avail.contains(s.frameGeometry()), s.frameGeometry()
    finally:
        fit.available_rect = old


# ------------------------------------------------------------------ CA15 - CA17
def test_ca15_time_history_envelope_in_any_units():
    m = load("TH_envelope_units")
    ref = None
    for unit, lf in (("N m", 1.0), ("N mm", 1000.0), ("kN m", 1.0)):
        mm = m.convert_units(unit)
        r = run(mm)
        th = [c for c, cr in r.cases.items() if isinstance(cr.history, dict) and "envelope" in cr.history][0]
        peak = float(np.max(np.abs(r.time_history_envelope(th)["displacements"].max[:, 0]))) / lf
        ref = peak if ref is None else ref
        assert peak == pytest.approx(ref, rel=1e-9), unit


def test_ca16_linear_links_in_linear_cases():
    """Guided column k_b = 12EI/L^3 parallel to a grounded link k = 1e7 N/m:
    u = F / (k_b + k); T = 2 pi sqrt(m / (k_b + k))."""
    m = load("linear_link")
    r = run(m)
    top = max(m.points.values(), key=lambda p: p.z).name
    assert r.joint_displacement(top, "LIN")[0] == pytest.approx(5.844155844e-5, rel=1e-8)
    assert r.joint_displacement(top, "NL")[0] == pytest.approx(5.844155844e-5, rel=1e-8)
    assert r.cases["MOD"].modal.periods[0] == pytest.approx(0.04803311618, rel=1e-8)
    for p in m.link_properties.values():
        p.mass = 9000.0  # link mass: 10,000 kg in all at the joint
    T = run(m).cases["MOD"].modal.periods[0]
    assert T == pytest.approx(0.04803311618 * math.sqrt(10.0), rel=1e-8)


def test_ca17_distributed_supports_in_every_base_reaction():
    r = run(load("foundation_reactions"))
    for case in ("LIN", "NL", "RS"):
        assert abs(r.base_reactions(case)[2]) == pytest.approx(2000.0, rel=1e-9), case


# ------------------------------------------------------------------ CA18 - CA21, CA25, CA26
def test_ca18_ground_records_in_one_direction_add_up():
    r = run(load("ground_sum"))
    h1, h2 = r.cases["ONE"].history, r.cases["TWO"].history
    k = int(np.argmin(np.abs(np.asarray(h1["time"]) - 0.042)))
    assert h2["ground_acceleration"][k][0] == pytest.approx(2 * h1["ground_acceleration"][k][0], rel=1e-12)


def test_ca19_modal_time_history_keeps_fixed_end_actions():
    """Fixed-fixed beam 4 m, 1,000 N/m: end shears 2,000 N, end moments wL^2/12."""
    r = run(load("fully_fixed_member_TH"))
    for case in ("DirectIntegration", "Modal", "FNA"):
        assert r.cases[case].history["base_reaction"][-1][2] == pytest.approx(4000.0, rel=1e-9), case
    beam = [f for f in r.model.frames.values() if abs(r.model.points[f.point_i].z - r.model.points[f.point_j].z) < 1e-9]
    ff = r.frame_forces(beam[0].name, "Modal")
    assert abs(ff.forces[0, 5]) == pytest.approx(1000 * 16 / 12, rel=1e-6)


def test_ca20_overdamped_modal_time_history_is_exact():
    """zeta = 2, step load: closed form of the overdamped oscillator at t = 0.05 s."""
    r = run(load("overdamped_modal_TH"))
    top = max(r.model.points.values(), key=lambda p: p.z).name
    assert r.cases["TH"].history["modal_damping"][0] == pytest.approx(2.0)
    nid = r.am.point_to_node[top]
    assert r.cases["TH"].displacements[nid, 0] == pytest.approx(9.167358289e-5, rel=1e-8)


def test_ca21_nonlinear_case_with_support_displacement_is_refused():
    assert code_of(lambda: run(load("nonlinear_settlement"))) == "NOT_IMPLEMENTED"


def test_ca25_end_of_the_time_function_is_not_lost_to_round_off():
    r = run(load("time_function_end_roundoff"))
    th = [c for c, cr in r.cases.items() if isinstance(cr.history, dict) and "envelope" in cr.history][0]
    assert r.cases[th].history["base_reaction"][-1][2] == pytest.approx(4000.0, rel=1e-9)


def test_ca26_direct_integration_starts_in_equilibrium():
    """SDOF 1,000 kg, step 1,000 N: a(0) = 1 m/s^2; first displacement ~ 0.5 a dt^2."""
    r = run(load("nonzero_initial_acceleration"))
    h = r.cases["DirectIntegration"].history
    top = max(r.model.points.values(), key=lambda p: p.z).name
    i = h["monitor_nodes"].index(r.am.point_to_node[top])
    assert h["accelerations"][0][i][0] == pytest.approx(1.0, rel=1e-9)
    assert h["displacements"][1][i][0] == pytest.approx(4.997037739e-7, rel=2e-3)


# ------------------------------------------------------------------ CA23 / CA24
def test_ca23_local_joint_load_is_turned_to_global():
    from structstudio.io.importers import import_model
    m = import_model(os.path.join(D, "local_joint_force.s2k")).model
    f = m.points["2"].loads[0].force
    assert np.allclose(f, (0.0, 1000.0, 0.0), atol=1e-9)


def test_ca24_native_conversion_ignores_stale_files(tmp_path, monkeypatch):
    import subprocess
    from structstudio.io import native
    src = tmp_path / "in"
    src.mkdir()
    p = src / "Model.sdb"
    p.write_text("placeholder")
    cache = tmp_path / "cache"
    cache.mkdir()
    stale = cache / "Model.e2k"
    stale.write_text("old ETABS model")
    os.utime(stale, (1000, 1000))
    old_tables = cache / "Model.SAP2000.tables.txt"
    old_tables.write_text("old tables")
    os.utime(old_tables, (1000, 1000))

    def conversion(*a, **k):
        (cache / "Model.SAP2000.tables.txt").write_text("fresh tables from this run")
        return subprocess.CompletedProcess(a[0], 0, stdout="OK", stderr="")

    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setattr(native.subprocess, "run", conversion)
    out = native.convert(str(p), out_dir=str(cache))
    assert out == [str(cache / "Model.SAP2000.tables.txt")]
    # nothing written by this run: no stale file is taken as its output
    monkeypatch.setattr(native.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="", stderr=""))
    with pytest.raises(native.ConversionError):
        native.convert(str(p), out_dir=str(cache))
    # two models of the same name in other folders never share a default folder
    q = tmp_path / "other" / "Model.sdb"
    q.parent.mkdir()
    q.write_text("x")
    assert native.default_out_dir(str(p)) != native.default_out_dir(str(q))


# ------------------------------------------------------------------ in-house items
def _building():
    from structstudio.app.templates import new_building
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    for p in m.load_patterns.values():
        p.self_weight_multiplier = 0.0
    return m


def test_o2_shells_panel_over_part_of_a_floor_loads_that_part_only():
    from structstudio.model.objects import PanelLoadCase
    m = _building()
    pat = next(iter(m.load_patterns))
    pl = m.add_panel_load([(0, 0, 3), (3, 0, 3), (3, 5, 3), (0, 5, 3)], distribution="shells")
    pl.loads.append(PanelLoadCase(pat, 2.0, "Gravity"))
    r = run(m)
    case = [c for c, v in m.load_cases.items() if getattr(v, "loads", None) and pat in v.loads][0]
    e = r.base_reaction_envelope(case).max
    assert e[2] == pytest.approx(2.0 * 3 * 5, rel=1e-9)  # half of the 6 x 5 floor
    assert -e[4] / e[2] == pytest.approx(1.5, rel=1e-6)  # load centroid at x = 1.5


def test_o3_output_station_at_a_point_load():
    m = _building()
    pat = next(iter(m.load_patterns))
    beam = [f for f in m.frames.values() if f.type == "Beam"][0]
    L = math.dist(m.points[beam.point_i].xyz, m.points[beam.point_j].xyz)
    m.add_frame_point_load(beam.name, pat, 10.0, rel=0.37)
    r = run(m)
    case = [c for c, v in m.load_cases.items() if getattr(v, "loads", None) and pat in v.loads][0]
    st = np.asarray(r.frame_forces(beam.name, case).station)
    assert np.min(np.abs(st - 0.37 * L)) < 1e-6


def test_o5_gravity_projected_area_load_on_a_slope():
    """q per unit plan area on a roof sloping 3 in 4: total = q x plan area."""
    from structstudio.model import PINNED
    from structstudio.model.properties import ShellSection
    m = Model("roof", "kN m")
    m.add_material("C", E=30e6)
    m.add_shell_section(ShellSection("S", "C", thickness=0.2))
    pts = [m.add_point(x, y, z, restraints=PINNED).name for x, y, z in ((0, 0, 0), (4, 0, 3), (4, 5, 3), (0, 5, 0))]
    a = m.add_area(pts, "S", type="Floor")
    m.add_load_pattern("Q")
    m.add_linear_static_case("Q")
    m.add_area_load(a.name, "Q", 2.0, direction="GravityProjected")
    assert run(m).base_reactions("Q")[2] == pytest.approx(2.0 * 4 * 5, rel=1e-9)


def test_o7_joint_restrained_vertically_stays_in_its_diaphragm():
    from structstudio.mesh import build_analysis_model
    from structstudio.engine.assembly import Assembler
    m = _building()
    floor = [a for a in m.areas.values() if a.type == "Floor"][0]
    p = m.points[floor.points[0]]
    p.restraints = (False, False, True, False, False, False)
    am = build_analysis_model(m)
    Assembler(am)
    nid = am.point_to_node[p.name]
    assert any(nid in c.nodes for c in am.constraints)


def test_o8_drawn_outline_problems():
    from structstudio.app.tools import _outline_problem
    assert _outline_problem([(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)], 1e-6) is None
    assert "one line" in _outline_problem([(0, 0, 0), (1, 0, 0), (2, 0, 0)], 1e-6)
    assert "crosses" in _outline_problem([(0, 0, 0), (4, 4, 0), (4, 0, 0), (0, 4, 0)], 1e-6)
    assert "same place" in _outline_problem([(0, 0, 0), (4, 0, 0), (0, 0, 0), (0, 4, 0)], 1e-6)


def test_o10_declining_to_unlock_is_not_an_error():
    from structstudio.app.menus import guarded_call

    def edit():
        raise ModelError("MODEL_LOCKED", "locked")

    assert guarded_call(edit) is None
    with pytest.raises(ModelError):
        guarded_call(lambda: (_ for _ in ()).throw(ModelError("OTHER", "x")))


def test_o12_mechanism_message_names_the_loose_joint():
    from structstudio.model import FIXED
    from structstudio.model.properties import FrameSection
    m = Model("mech", "kN m")
    m.add_material("S", E=200e6)
    m.add_frame_section(FrameSection.general("B", "S", A=0.01, I33=1e-4, I22=1e-4, J=1e-4))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(3, 0, 0)
    c = m.add_point(6, 0, 0, name="LOOSE")
    m.add_frame(a, b, "B")
    rel = (False, False, False, True, True, True)
    m.add_frame(b, c, "B", releases_i=rel, releases_j=(False, False, False, False, True, True))
    m.add_load_pattern("P")
    m.add_point_load(b, "P", force=(0, 0, -1))
    m.add_linear_static_case("P")
    try:
        r = run(m)
        text = " ".join(i.message for i in r.issues)
    except ModelError as exc:
        text = str(exc)
    assert "LOOSE" in text


def test_o14_empty_model_says_so():
    assert code_of(lambda: run(Model("e", "kN m"))) == "EMPTY_MODEL"


def test_offset_beam_is_shown_on_its_true_axis_with_rigid_links():
    """Show Analysis Mesh draws the analysis axis of a beam moved by its joint offset and
    the rigid links to its joints (the drawn object stays between its joints)."""
    pytest.importorskip("vtk")
    from structstudio.app import scene
    m = _building()
    b = [f for f in m.frames.values() if f.type == "Beam"
         and abs(m.points[f.point_i].y) < 1e-9 and abs(m.points[f.point_j].y) < 1e-9][0]
    b.joint_offset_i = b.joint_offset_j = (0.0, 0.1, 0.0)  # inside the 0.4 columns
    r = run(m)
    X = r.am.coords()
    ys = {round(float(X[n][1]), 9) for e in r.am.object_to_frames[b.name] for n in (r.am.frames[e].node_i, r.am.frames[e].node_j)}
    assert ys == {0.1}  # analysis nodes on the true axis
    assert len([lk for lk in r.am.rigid_links if lk.object == b.name]) == 2
    assert len(scene.build_offset_axis_actors(m, r.am)) == 3  # axis, rigid links, link ends


def test_move_beam_to_real_axis_with_rigid_links():
    """Edit > Move Beams to Real Axis: the beam object itself moves (its line is the real
    axis), rigid links join it to its old joints inside the columns' joint zones (0.1 in
    a 0.4 column), its horizontal offsets become zero, and the analysis equals the beam
    analysed on its offset axis; one undo restores it."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    m = _building()
    b = [f for f in m.frames.values() if f.type == "Beam"
         and abs(m.points[f.point_i].y) < 1e-9 and abs(m.points[f.point_j].y) < 1e-9][0]
    b.joint_offset_i = b.joint_offset_j = (0.0, 0.1, 0.0)
    ref = copy.deepcopy(m)
    doc = Document(m)
    old = (b.point_i, b.point_j)
    moved, links, notes = C.move_beams_to_axis(doc, [b.name])  # by its own joint offsets
    m = doc.model
    f = m.frames[b.name]
    assert moved == [b.name] and len(links) == 2 and not notes
    assert m.points[f.point_i].y == pytest.approx(0.1) and m.points[f.point_j].y == pytest.approx(0.1)
    assert f.joint_offset_i == (0.0, 0.0, 0.0) and f.joint_offset_j == (0.0, 0.0, 0.0)
    assert {frozenset((m.frames[n].point_i, m.frames[n].point_j)) for n in links} == \
        {frozenset((old[0], f.point_i)), frozenset((old[1], f.point_j))}
    assert all(m.frames[n].type == "Rigid" for n in links)
    from structstudio.model.checks import duplicates_of
    assert not duplicates_of(m, list(m.frames))  # links inside the columns: the joint zone, not duplicates
    case = _floor_load(m)
    _floor_load(ref)
    r1, r2 = run(m), run(ref)
    assert r1.base_reactions(case) == pytest.approx(r2.base_reactions(case), rel=1e-9, abs=1e-9)
    m1 = np.abs(r1.frame_forces(b.name, case).forces[:, 5]).max()
    m2 = np.abs(r2.frame_forces(b.name, case).forces[:, 5]).max()
    assert m1 == pytest.approx(m2, rel=1e-9)
    doc.undo()
    assert (doc.model.frames[b.name].point_i, doc.model.frames[b.name].point_j) == old
    assert not any(g.type == "Rigid" for g in doc.model.frames.values())


def test_moved_beam_outside_the_column_frames_into_the_side_beam():
    """Moved 0.5 inward (outside the 0.4 columns), each end lands on a side beam: it
    frames into that beam there - no rigid link along the side beam (that made part of
    it rigid and was reported as a duplicate member) - and the beam analysed on its
    offset axis is joined the same way: the same results."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    from structstudio.mesh.mesher import build_analysis_model
    from structstudio.model.checks import duplicates_of
    m = _building()
    b = _edge_beams(m, y=0.0)[0]
    sides = _edge_beams(m, x=0.0) + _edge_beams(m, x=6.0)
    m.frames[b].joint_offset_i = m.frames[b].joint_offset_j = (0.0, 0.5, 0.0)
    ref = copy.deepcopy(m)
    doc = Document(m)
    moved, links, notes = C.move_beams_to_axis(doc, [b])
    m = doc.model
    assert moved == [b] and not links
    assert sorted(n.split(" frames into ")[1].split(" ")[0] for n in notes) == sorted(sides)
    assert not duplicates_of(m, list(m.frames))
    am_a = build_analysis_model(ref)
    assert not am_a.rigid_links and sum("frames into" in i for i in am_a.info) == 2
    case = _floor_load(m)
    _floor_load(ref)
    r1, r2 = run(m), run(ref)
    assert r1.base_reactions(case) == pytest.approx(r2.base_reactions(case), rel=1e-9, abs=1e-9)
    for n in [b] + sides:
        assert np.abs(r1.frame_forces(n, case).forces).max(axis=0) == \
            pytest.approx(np.abs(r2.frame_forces(n, case).forces).max(axis=0), rel=1e-9, abs=1e-9)
    # the side beam carries the edge beam's end shear: more than with the beam on the column line
    m0 = _building()
    c0 = _floor_load(m0)
    v0 = np.abs(run(m0).frame_forces(sides[0], c0).forces[:, 1]).max()
    assert np.abs(r1.frame_forces(sides[0], case).forces[:, 1]).max() > v0


def test_beam_moved_down_frames_into_the_columns_partway():
    """Insertion point dy = -1.0 (local 2: down) with Move: the beam is moved down 1.0;
    the columns go on past it, so it frames into them there (no rigid link) - the same
    structure as that beam drawn there by hand; it gets the column's rigid end zone."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    from structstudio.app.templates import new_building
    from structstudio.mesh.mesher import build_analysis_model
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0, 3.0), grid_only=False)
    for p in m.load_patterns.values():
        p.self_weight_multiplier = 0.0
    for a in [a for a in m.areas if m.areas[a].type == "Floor"]:
        del m.areas[a]
    b = [n for n in _edge_beams(m, y=0.0) if abs(m.points[m.frames[n].point_i].z - 6.0) < 1e-9][0]
    pat = next(iter(m.load_patterns))
    case = [c for c, v in m.load_cases.items() if getattr(v, "loads", None) and pat in v.loads][0]
    m.add_frame_distributed_load(b, pat, 12.0, 12.0)
    hand = copy.deepcopy(m)
    f0 = hand.frames[b]
    pi, pj = hand.points[f0.point_i], hand.points[f0.point_j]
    f0.point_i = hand.add_point(pi.x, pi.y, pi.z - 1.0).name
    f0.point_j = hand.add_point(pj.x, pj.y, pj.z - 1.0).name
    doc = Document(m)
    moved, links, notes = C.move_beams_to_axis(
        doc, [b], assign={"joint_offset_i": (0.0, -1.0, 0.0), "joint_offset_j": (0.0, -1.0, 0.0),
                          "joint_offset_local": True})
    m = doc.model
    f = m.frames[b]
    assert moved == [b] and not links and m.points[f.point_i].z == pytest.approx(5.0)
    assert f.joint_offset_i == (0.0, 0.0, 0.0) and sum("frames into column" in t for t in notes) == 2
    assert any("out of the floor plane" in t for t in notes)
    am = build_analysis_model(m)
    els = am.object_to_frames[b]
    assert am.frames[els[0]].end_offset_i == pytest.approx(0.2)  # half the 0.4 column
    r1, r2 = run(m), run(hand)
    assert r1.base_reactions(case) == pytest.approx(r2.base_reactions(case), rel=1e-9, abs=1e-9)
    assert r1.base_reactions(case)[2] == pytest.approx(12.0 * 6.0)
    assert np.abs(r1.frame_forces(b, case).forces).max(axis=0) == \
        pytest.approx(np.abs(r2.frame_forces(b, case).forces).max(axis=0), rel=1e-9, abs=1e-9)


def test_beam_moved_across_and_down_hangs_on_the_side_beam():
    """0.3 across (outside the 0.4 column) and 0.2 down: the beam end is within the side
    beam's depth - a joint on the side beam and a 0.2 rigid link down from it, exactly
    what the offset way does (axis on the side beam, a 0.2 rigid arm): same statics."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    m = _building()
    for a in [a for a in m.areas if m.areas[a].type == "Floor"]:
        del m.areas[a]
    b = _edge_beams(m, y=0.0)[0]
    pat = next(iter(m.load_patterns))
    case = [c for c, v in m.load_cases.items() if getattr(v, "loads", None) and pat in v.loads][0]
    m.add_frame_distributed_load(b, pat, 12.0, 12.0)
    m.frames[b].joint_offset_i = m.frames[b].joint_offset_j = (0.0, 0.3, -0.2)
    ref = copy.deepcopy(m)
    doc = Document(m)
    moved, links, notes = C.move_beams_to_axis(doc, [b])
    m = doc.model
    assert moved == [b] and len(links) == 2
    for n in links:
        lk = m.frames[n]
        top, bot = m.points[lk.point_i], m.points[lk.point_j]
        assert (top.y, top.z, bot.y, bot.z) == pytest.approx((0.3, 3.0, 0.3, 2.8))
    r1, r2 = run(m), run(ref)
    assert r1.base_reactions(case) == pytest.approx(r2.base_reactions(case), rel=1e-9, abs=1e-9)
    assert np.abs(r1.frame_forces(b, case).forces).max(axis=0) == \
        pytest.approx(np.abs(r2.frame_forces(b, case).forces).max(axis=0), rel=1e-9, abs=1e-9)

def _edge_beams(m, y=None, x=None):
    out = []
    for f in m.frames.values():
        if f.type != "Beam":
            continue
        a, b = m.points[f.point_i], m.points[f.point_j]
        if y is not None and abs(a.y - y) < 1e-9 and abs(b.y - y) < 1e-9:
            out.append(f.name)
        if x is not None and abs(a.x - x) < 1e-9 and abs(b.x - x) < 1e-9:
            out.append(f.name)
    return sorted(out)


def _column_end_offsets(am, m):
    """Top end offsets of the column elements (the depth of the deepest beam framing in)."""
    return [e.end_offset_j for e in am.frames
            if m.frames[e.object].type == "Column" and e.end_offset_j > 0]


def _floor_load(m, q=2.0):
    pat = next(iter(m.load_patterns))
    case = [c for c, v in m.load_cases.items() if getattr(v, "loads", None) and pat in v.loads][0]
    for a in m.areas.values():
        if a.type == "Floor":
            m.add_area_load(a.name, pat, q)
    return case


def test_insertion_point_dialog_moves_beams_in_one_undo_step():
    """Assign > Insertion Point with 'Move Beams to Real Axis' ticked: the offsets are
    assigned and the beam moved in ONE edit (one undo goes back to the original)."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    m = _building()
    b = _edge_beams(m, y=0.0)[0]
    old = (m.frames[b].point_i, m.frames[b].point_j)
    doc = Document(m)
    moved, links, notes = C.move_beams_to_axis(
        doc, [b], assign={"joint_offset_i": (0.0, 0.1, 0.0), "joint_offset_j": (0.0, 0.1, 0.0),
                          "joint_offset_local": False})
    f = doc.model.frames[b]
    assert moved == [b] and len(links) == 2 and not notes
    assert doc.model.points[f.point_i].y == pytest.approx(0.1)
    assert f.joint_offset_i == (0.0, 0.0, 0.0)
    doc.undo()
    g = doc.model.frames[b]
    assert (g.point_i, g.point_j) == old and g.joint_offset_i == (0.0, 0.0, 0.0)
    assert not any(h.type == "Rigid" for h in doc.model.frames.values())


def test_two_corner_beams_moved_equal_the_offset_beams():
    """Both edge beams at a corner moved 0.25 (outside the 0.4 column): at the shared
    corner one rigid link per beam (no link to the other beam's new end; the other beam
    moved away, so nothing to frame into); at their far ends they frame into the side
    beams; the corner column keeps its top end offset through the rigid links, and the
    results equal the two beams analysed on their offset axes."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    from structstudio.mesh.mesher import build_analysis_model
    m = _building()
    bs = _edge_beams(m, y=0.0) + _edge_beams(m, x=0.0)
    m.frames[bs[0]].joint_offset_i = m.frames[bs[0]].joint_offset_j = (0.0, 0.25, 0.0)
    m.frames[bs[1]].joint_offset_i = m.frames[bs[1]].joint_offset_j = (0.25, 0.0, 0.0)
    ref = copy.deepcopy(m)
    doc = Document(m)
    moved, links, notes = C.move_beams_to_axis(doc, bs)
    m = doc.model
    assert sorted(moved) == sorted(bs) and len(links) == 2
    assert {m.frames[n].point_i for n in links} == {ref.frames[bs[0]].point_i}  # the shared corner
    assert sum("frames into" in t for t in notes) == 2
    am_a, am_b = build_analysis_model(ref), build_analysis_model(m)
    assert sorted(_column_end_offsets(am_b, m)) == pytest.approx(sorted(_column_end_offsets(am_a, ref)))
    assert max(_column_end_offsets(am_b, m)) == pytest.approx(0.6)  # deepest beam B30x60
    case_a, case_b = _floor_load(ref), _floor_load(m)
    ra, rb = run(ref), run(m)
    assert rb.base_reactions(case_b) == pytest.approx(ra.base_reactions(case_a), rel=1e-9, abs=1e-9)
    for b in bs:
        fa = ra.frame_forces(b, case_a).forces
        fb = rb.frame_forces(b, case_b).forces
        assert np.abs(fb).max() == pytest.approx(np.abs(fa).max(), rel=1e-9)


def test_length_inside_column_is_exact_off_the_axis():
    """Automatic end offset of a beam end at a 0.4 x 0.6 column: half the column on its
    axis (ETABS), the exact rectangle exit off it, zero outside."""
    from structstudio.mesh.mesher import _length_inside_column as L
    axes = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])  # local 2 = X (t3), 3 = Y (t2)
    x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    t3, t2 = 0.6, 0.4
    assert L(np.zeros(3), x, axes, t3, t2, 1e-9) == pytest.approx(0.3)
    assert L(np.zeros(3), y, axes, t3, t2, 1e-9) == pytest.approx(0.2)
    assert L(np.array([0, 0.15, 0]), x, axes, t3, t2, 1e-9) == pytest.approx(0.3)  # along X, off in Y
    assert L(np.array([0.1, 0, 0]), x, axes, t3, t2, 1e-9) == pytest.approx(0.2)   # 0.3 - 0.1
    assert L(np.array([0.1, 0, 0]), -x, axes, t3, t2, 1e-9) == pytest.approx(0.4)  # 0.3 + 0.1
    assert L(np.array([0, 0.3, 0]), x, axes, t3, t2, 1e-9) == 0.0                   # outside


@pytest.mark.parametrize("how", ["offset", "moved"])
def test_beam_outside_its_floor_is_warned(how):
    """An edge beam whose real axis is outside the floor edge (offset outward, or moved
    there) is not connected to the slab: a warning names it; inward gives none."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    from structstudio.mesh.mesher import build_analysis_model
    for dy, want in ((0.3, False), (-0.3, True)):
        m = _building()
        b = _edge_beams(m, y=0.0)[0]
        m.frames[b].joint_offset_i = m.frames[b].joint_offset_j = (0.0, dy, 0.0)
        if how == "moved":
            doc = Document(m)
            assert C.move_beams_to_axis(doc, [b])[0] == [b]
            m = doc.model
        hits = [w for w in build_analysis_model(m).warnings if "outside floor" in w]
        assert bool(hits) == want, (dy, hits)
        if want:
            assert hits[0].startswith(f"Beam {b}:") and "0.300" in hits[0]


def test_offset_zone_issues_sideways_and_up_down():
    """An offset in the analysis beyond the rigid zone of the joint - sideways outside the
    column, or up / down so that the joint's level is outside the beam's depth - is
    reported (the Insertion Point form then offers Move Beams to Real Axis)."""
    from structstudio.app import commands as C
    m = _building()  # columns C40x40, beam B30x60
    b = _edge_beams(m, y=0.0)[0]

    def issues(off, cp=8, local=False, include=True):
        return C.offset_zone_issues(m, [b], {"joint_offset_i": off, "joint_offset_j": off, "cardinal_point": cp,
                                             "joint_offset_local": local, "offsets_in_analysis": include})
    assert issues((0, 0.1, 0)) == []  # inside the column
    assert len(issues((0, 0.5, 0))) == 2  # both ends outside it
    assert issues((0, 0.5, 0), include=False) == []  # drawn only: not in the analysis
    assert issues((0, 0.2, 0), local=True) == []  # 0.2 up: the joint still within the 0.6 depth
    got = issues((0, 0, -0.2))  # 0.2 down from the top: the slab no longer touches it
    assert len(got) == 1 and "0.200 below" in got[0]
    assert issues((0, 0.3, 0), cp=10, local=True) == []  # centroid 0.3 up = half the depth: the edge


def test_offset_beyond_depth_is_warned_in_the_analysis():
    from structstudio.mesh.mesher import build_analysis_model
    m = _building()
    b = _edge_beams(m, y=0.0)[0]
    f = m.frames[b]
    f.cardinal_point, f.offsets_in_analysis = 8, True  # the line is the beam top, in the analysis
    f.joint_offset_i = f.joint_offset_j = (0.0, 0.0, -0.2)
    w = [x for x in build_analysis_model(m).warnings if "outside its depth" in x]
    assert len(w) == 1 and w[0].startswith(f"Beam {b}: 0.200 below") and "Move Beams to Real Axis" in w[0]


def test_secondary_beam_end_follows_the_moved_beam():
    """A secondary beam framing into the moved beam's old line: its end goes to the beam's
    new line (no stub past the beam moved inward, no rigid link to reach it moved
    outward) - exactly the beams drawn there by hand."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    from structstudio.model.checks import duplicates_of
    for dy in (0.3, -0.3):
        m = _building()
        b = _edge_beams(m, y=0.0)[0]
        mid, midt = m.add_point(3.0, 0.0, 3.0), m.add_point(3.0, 5.0, 3.0)
        s2 = m.add_frame(mid.name, midt.name, m.frames[b].section).name
        case = _floor_load(m)
        hand = copy.deepcopy(m)
        f = hand.frames[b]
        ji, jj = f.point_i, f.point_j
        f.point_i, f.point_j = hand.add_point(0.0, dy, 3.0).name, hand.add_point(6.0, dy, 3.0).name
        hand.frames[s2].point_i = hand.add_point(3.0, dy, 3.0).name
        if dy < 0:  # outside the columns, on no member: rigid links from the column joints
            hand.add_frame(ji, f.point_i, "None", type="Rigid")
            hand.add_frame(jj, f.point_j, "None", type="Rigid")
        m.frames[b].joint_offset_i = m.frames[b].joint_offset_j = (0.0, dy, 0.0)
        doc = Document(m)
        moved, links, notes = C.move_beams_to_axis(doc, [b])
        mm = doc.model
        assert mm.points[mm.frames[s2].point_i].y == pytest.approx(dy) and len(links) == (0 if dy > 0 else 2)
        assert any("its end moved" in t for t in notes) and not duplicates_of(mm, list(mm.frames))
        if dy < 0:  # 0.3 out: past the 0.4 column's face, the 0.3 wide beam still bears on the column
            assert sum("still bears on it" in t for t in notes) == 2
        r1, r2 = run(mm), run(hand)
        assert r1.base_reactions(case) == pytest.approx(r2.base_reactions(case), rel=1e-9, abs=1e-9)
        for n in (b, s2):
            assert np.abs(r1.frame_forces(n, case).forces).max(axis=0) == \
                pytest.approx(np.abs(r2.frame_forces(n, case).forces).max(axis=0), rel=1e-9, abs=1e-9)


def test_rigid_links_over_each_other_are_not_duplicates():
    """Two beams meeting at a joint, moved across onto the side beam and down by different
    depths within its depth: each gets its own short rigid link down from the side beam.
    The links lie over each other - one rigid body, not reported as duplicate members -
    and each beam can be put back on its own."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    from structstudio.app.templates import new_building
    from structstudio.model.checks import duplicates_of
    m = new_building(x_spacings=(6,), y_spacings=(5, 4), story_heights=(3.5,), grid_only=False)
    a, b = sorted(_edge_beams(m, x=6.0), key=lambda n: m.points[m.frames[n].point_i].y)
    for n, dz in ((a, -0.4), (b, -0.2)):  # 0.6 inward: onto the beams along x, clear of the columns
        m.frames[n].joint_offset_i = m.frames[n].joint_offset_j = (-0.6, 0.0, dz)
    doc = Document(m)
    moved, links, notes = C.move_beams_to_axis(doc, [a, b])
    mm = doc.model
    assert sorted(moved) == sorted([a, b]) and len(links) == 4 and not duplicates_of(mm, list(mm.frames))
    C.set_insertion_point(doc, [a], {"joint_offset_i": (0.0, 0.0, 0.0), "joint_offset_j": (0.0, 0.0, 0.0)}, move=False)
    mm = doc.model
    assert sum(1 for g in mm.frames.values() if g.type == "Rigid") == 2 and C.moved_beam_state(mm, b) is not None
    assert mm.points[mm.frames[a].point_i].x == pytest.approx(6.0) and mm.points[mm.frames[a].point_i].z == pytest.approx(3.5)


def test_offset_beam_outside_a_membrane_floor_takes_no_load_from_it():
    """A beam analysed on its real axis outside a membrane floor (warned) takes no load
    from that floor - the same as the beam moved there (before: it took the floor's edge
    load as if under it, unlike a shell floor)."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    m = _building()
    for s in m.shell_sections.values():
        s.model_type = "Membrane"
    b = _edge_beams(m, y=0.0)[0]
    m.frames[b].joint_offset_i = m.frames[b].joint_offset_j = (0.0, -0.3, 0.0)
    case = _floor_load(m)
    doc = Document(copy.deepcopy(m))
    C.move_beams_to_axis(doc, [b])
    rA, rB = run(m), run(doc.model)
    assert any(f"Beam {b}: its real axis is 0.300 outside floor" in w for w in rA.am.warnings)
    shear = lambda r: np.abs(r.frame_forces(b, case).forces[:, 1]).max()
    assert shear(rA) == pytest.approx(shear(rB), abs=1e-9) and shear(rA) < 1e-6
    assert rA.base_reactions(case) == pytest.approx(rB.base_reactions(case), rel=1e-9)


def test_moved_beam_warned_only_against_its_own_floor():
    """A moved beam outside its floor is warned for that floor only - not for a floor that
    merely has a corner at one of its joints."""
    from structstudio.app.document import Document
    from structstudio.app import commands as C
    from structstudio.app.templates import new_building
    from structstudio.mesh.mesher import build_analysis_model
    m = new_building(x_spacings=(5, 4, 8), y_spacings=(7,), story_heights=(3.5,), grid_only=False)
    b = [n for n in _edge_beams(m, y=0.0) if m.points[m.frames[n].point_i].x == pytest.approx(5.0)
         or m.points[m.frames[n].point_j].x == pytest.approx(5.0)]
    b = [n for n in b if min(m.points[m.frames[n].point_i].x, m.points[m.frames[n].point_j].x) == pytest.approx(5.0)][0]
    m.frames[b].joint_offset_i = m.frames[b].joint_offset_j = (0.0, -0.3, 0.0)
    doc = Document(m)
    C.move_beams_to_axis(doc, [b])
    w = [x for x in build_analysis_model(doc.model).warnings if "outside floor" in x]
    assert len(w) == 1 and "0.300 outside floor" in w[0]


def test_offset_beam_local_axes_from_its_joint_line():
    """A beam on its real axis with unequal end offsets (its axis turned in plan) keeps the
    local axes of the CSI rule from its joint-to-joint line (v3 = v1~ x v2): the section
    is not turned with the axis (found by the OpenSees cross-check, seed 50027)."""
    from structstudio.mesh.mesher import build_analysis_model
    from structstudio.mesh.geometry import frame_axes
    m = _building()
    b = _edge_beams(m, y=0.0)[0]
    f = m.frames[b]
    f.angle = 90.0
    f.joint_offset_i, f.joint_offset_j = (0.25, -0.13, -0.13), (-0.2, 0.22, -0.26)
    am = build_analysis_model(m)
    el = am.frames[am.object_to_frames[b][0]]
    a = np.array(m.points[f.point_i].xyz, float)
    c = np.array(m.points[f.point_j].xyz, float)
    nominal = frame_axes(a, c, 90.0)
    v1 = (c + np.array(f.joint_offset_j)) - (a + np.array(f.joint_offset_i))
    v1 /= np.linalg.norm(v1)
    v3 = np.cross(v1, nominal[1])
    v3 /= np.linalg.norm(v3)
    assert el.axes[0] == pytest.approx(v1, abs=1e-9) and el.axes[2] == pytest.approx(v3, abs=1e-9)
