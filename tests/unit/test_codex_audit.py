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
