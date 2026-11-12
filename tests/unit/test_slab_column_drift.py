"""Slab-column connections under the design drift (ACI 318-19 18.14.5 / 318-14 18.14.5.1).

Every number is checked against an independent hand calculation: vc from Table 22.6.5.2 /
22.6.5.5 (ACI 318M-19 and 318M-14 as printed), the critical section, Vu from the column
axial forces of each case, the design drift Cd/Ie x the drift of the column's own ends."""
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from structstudio.app.templates import new_building
from structstudio.design import check_slab_column_drift, earthquake_cases
from structstudio.design.slab_column_drift import _critical_section, lambda_s, two_way_vc
from structstudio.model import Model
from structstudio.model.design import SlabColumnDriftPrefs

sys.path.insert(0, str(Path(__file__).parent))
from test_stiffness_sets import _flat_plate, _run  # noqa: E402


# ------------------------------------------------------------ formulas
def test_vc_aci318_19_nonprestressed():
    fc, d, bo = 28.0, 400.0, 4000.0
    ls = math.sqrt(2 / (1 + 0.004 * 400))  # 22.5.5.1.3: 0.877 (< 1 for d > 250 mm)
    assert lambda_s(400) == pytest.approx(ls) and lambda_s(200) == 1.0 and ls < 1
    a = 0.33 * ls * math.sqrt(fc)
    b = (0.17 + 0.33 / 2.5) * ls * math.sqrt(fc)
    c = 0.083 * (2 + 40 * d / bo) * ls * math.sqrt(fc)
    vc, clause, l = two_way_vc("ACI 318-19", fc, d, bo, 2.5, 40)
    assert clause == "22.6.5.2" and l == pytest.approx(ls)
    assert vc == pytest.approx(min(a, b, c))
    # √f'c is limited to 8.3 MPa (22.6.3.1)
    assert two_way_vc("ACI 318-19", 100.0, 150, 2000, 1.0, 40)[0] == pytest.approx(0.33 * 8.3)


def test_vc_aci318_14_nonprestressed():
    fc, d, bo = 30.0, 250.0, 3000.0
    a = 0.33 * math.sqrt(fc)
    b = 0.17 * (1 + 2 / 3.0) * math.sqrt(fc)  # 318-14: 0.17(1 + 2/β), no size effect
    c = 0.083 * (2 + 30 * d / bo) * math.sqrt(fc)
    vc, clause, l = two_way_vc("ACI 318-14", fc, d, bo, 3.0, 30)
    assert l == 1.0 and vc == pytest.approx(min(a, b, c))


def test_vc_prestressed_22_6_5_5():
    fc, d, bo, fpc = 40.0, 180.0, 2400.0, 5.0
    sq = 5.8  # √f'c <= 5.8 MPa, fpc <= 3.5 MPa, Vp = 0
    va = 0.29 * sq + 0.3 * 3.5
    vb = 0.083 * (1.5 + 40 * d / bo) * sq + 0.3 * 3.5
    vc, clause, _ = two_way_vc("ACI 318-19", fc, d, bo, 1.0, 40, fpc=fpc)
    assert clause == "22.6.5.5" and vc == pytest.approx(min(va, vb))


def test_critical_section():
    c, d = 0.4, 0.12
    bo, n, closed, _ = _critical_section(c, c, d, [None] * 4)
    assert bo == pytest.approx(4 * (c + d)) and n == 4
    # face flush with the edge: three sides, two of them running to the edge
    bo, n, closed, _ = _critical_section(c, c, d, [None, 0.0, None, None])
    assert n == 3 and bo == pytest.approx((c + d) + 2 * (c + d / 2))
    # 0.3 m overhang: the open section (to the edge) is shorter than the closed one
    bo, n, _, _ = _critical_section(c, c, d, [0.3, None, None, None])
    assert n == 3 and bo == pytest.approx((c + d) + 2 * (c / 2 + d / 2 + c / 2 + 0.3))
    assert bo < 4 * (c + d)
    # a long overhang: closed
    bo, n, _, _ = _critical_section(c, c, d, [2.0, None, None, None])
    assert n == 4 and bo == pytest.approx(4 * (c + d))
    # an edge nearer than d/2 cannot be closed; corner
    bo, n, _, _ = _critical_section(0.5, 0.3, d, [0.02, None, 0.0, None])
    assert n == 2 and bo == pytest.approx((0.5 / 2 + d / 2 + 0.25 + 0.02) + (0.3 / 2 + d / 2 + 0.15))


# ---------------------------------------------------------- the model
@pytest.fixture(scope="module")
def plate():
    m, floors = _flat_plate()
    m.load_cases["EQX"].cd = 5.5
    m.load_cases["RSX"].cd = 5.0
    m.load_cases["RSX"].ie = 1.25
    m.add_combination("G+RS", {"Dead": 1.2, "Live": 0.5, "RSX": 1.0})
    return m, _run(m)


def _shear(r, below, above, pt, case):
    """P above - P below at the joint (model units) for one case."""
    m = r.model

    def end(fr):
        f = m.frames[fr]
        return r.frame_forces(fr, case).forces[0 if f.point_i == pt else -1][0]

    return (end(above) if above else 0.0) - end(below)


def test_connections_and_hand_values(plate):
    m, r = plate
    assert set(earthquake_cases(m)) == {"EQX", "RSX"}
    rep = check_slab_column_drift(r)
    assert not rep.errors and set(rep.combos) == {"G+E", "G+RS"}
    assert len(rep.checks) == 18  # 9 columns x 2 floors
    kinds = {}
    for c in rep.checks:
        kinds[c.location] = kinds.get(c.location, 0) + 1
    assert kinds == {"Interior": 2, "Edge": 8, "Corner": 8}
    sec = m.shell_sections[m.areas[next(a for a in m.areas if m.areas[a].type == "Floor")].section]
    fc = m.materials[sec.material].fc / 1000.0  # kN/m2 -> MPa
    for c in rep.checks:
        h = sec.thickness
        d = h - 0.030
        assert c.h == pytest.approx(h) and c.d == pytest.approx(d)
        col = m.frame_sections[m.frames[c.column].section]
        a = col.t3
        bo = {"Interior": 4 * (a + d), "Edge": (a + d) + 2 * (a + d / 2), "Corner": 2 * (a + d / 2)}[c.location]
        assert c.bo == pytest.approx(bo)
        als = {"Interior": 40, "Edge": 30, "Corner": 20}[c.location]
        ls = min(1.0, math.sqrt(2 / (1 + 0.004 * d * 1e3)))
        sq = math.sqrt(fc)
        vc = min(0.33 * ls * sq, (0.17 + 0.33) * ls * sq, 0.083 * (2 + als * d / bo) * ls * sq)
        assert c.phi_vc == pytest.approx(0.75 * vc)
        # Vu: the combinations formed from the cases' column forces
        V = {n: _shear(r, c.column, c.column_above, c.point, n) for n in ("Dead", "Live", "EQX")}
        ge = 1.2 * V["Dead"] + 1.0 * V["Live"] + V["EQX"]
        g_rs = 1.2 * V["Dead"] + 0.5 * V["Live"]
        assert c.Vu / 1e3 >= ge - 1e-6
        if c.combo == "G+E":
            assert c.Vu / 1e3 == pytest.approx(ge, rel=1e-9)
        else:
            assert c.Vu / 1e3 > g_rs  # the spectrum adds
        assert c.vu == pytest.approx(c.Vu / (c.bo * c.d) / 1e6)
        assert c.ratio == pytest.approx(c.vu / c.phi_vc)
        # the criterion of 18.14.5.1(a) / 18.14.5.2(a)
        assert c.limit == pytest.approx(0.035 - 0.05 * c.ratio) and c.exempt == 0.005
        assert c.required == (c.drift > 0.005 and c.drift >= c.limit)
        if c.required:
            assert c.vs_min == pytest.approx(0.29 * math.sqrt(fc)) and c.extent == pytest.approx(4 * h)


def test_spectrum_shear_is_formed_mode_by_mode(plate):
    """Vu of the spectrum alone: between the difference and the sum of the column
    forces' spectrum values (never the difference of two envelopes)."""
    m, r = plate
    rep = check_slab_column_drift(r, SlabColumnDriftPrefs(combos=["RSX"]))
    for c in rep.checks:
        pa = abs(_shear(r, c.column_above, None, c.point, "RSX")) if c.column_above else 0.0
        pb = abs(_shear(r, c.column, None, c.point, "RSX"))
        v = c.Vu / 1e3
        assert abs(pb - pa) - 1e-6 <= v <= pa + pb + 1e-6


def test_design_drift(plate):
    m, r = plate
    stories = [s.name for s in m.stories_bottom_up()]
    # a static case: Cd/Ie x the drift of the column's ends, X or Y, greater of below / above
    rep = check_slab_column_drift(r, SlabColumnDriftPrefs(seismic_cases=["EQX"]))
    pts = {(round(p.x, 6), round(p.y, 6), round(p.z, 6)): p.name for p in m.points.values()}

    def drift(top, bot, h):
        ut, ub = r.joint_displacement(top, "EQX"), r.joint_displacement(bot, "EQX")
        return 5.5 * max(abs(ut[0] - ub[0]), abs(ut[1] - ub[1])) / h

    for c in rep.checks:
        k = stories.index(c.story)
        zb = m.stories[stories[k - 1]].elevation
        below = pts[(round(c.x, 6), round(c.y, 6), round(zb, 6))]
        assert c.drift_below == pytest.approx(drift(c.point, below, c.z - zb), rel=1e-9)
        if k + 1 < len(stories):
            za = m.stories[stories[k + 1]].elevation
            above = pts[(round(c.x, 6), round(c.y, 6), round(za, 6))]
            assert c.drift_above == pytest.approx(drift(above, c.point, za - c.z), rel=1e-9)
        else:
            assert c.drift_above == 0.0
        assert c.drift == max(c.drift_below, c.drift_above) and c.drift_case == "EQX"
    # a spectrum: formed mode by mode like the story drifts (Cd/Ie = 5 / 1.25 = 4)
    rep = check_slab_column_drift(r, SlabColumnDriftPrefs(seismic_cases=["RSX"]))
    rows = r.story_drifts("RSX")
    for st in stories[1:]:
        best = max((row for row in rows if row["Story"] == st), key=lambda row: row["Drift"])
        mine = [c for c in rep.checks if c.story == st]
        assert max(c.drift_below for c in mine) <= 4.0 * best["Drift"] * (1 + 1e-9)
        lab = [c for c in mine if c.point == best["Label"]]
        if lab:
            assert lab[0].drift_below == pytest.approx(4.0 * best["Drift"], rel=1e-9)


def test_aci318_14_gravity_shear(plate):
    m, r = plate
    m.design.slab_column_drift.code = "ACI 318-14"
    try:
        rep = check_slab_column_drift(r, model=m)
    finally:
        m.design.slab_column_drift.code = "ACI 318-19"
    assert not rep.errors
    for c in rep.checks:
        V = {n: _shear(r, c.column, c.column_above, c.point, n) for n in ("Dead", "Live")}
        assert c.Vu / 1e3 == pytest.approx(1.2 * V["Dead"] + 1.0 * V["Live"], rel=1e-9)  # G+E less E
        assert c.lambda_s == 1.0 and c.exempt == 0.005 and c.limit == pytest.approx(0.035 - c.ratio / 20)


def test_post_tensioned(plate):
    m, r = plate
    sec = m.shell_sections[m.areas[next(a for a in m.areas if m.areas[a].type == "Floor")].section]
    old = dataclasses.replace(sec)
    sec.post_tensioned, sec.fpc1, sec.fpc2, sec.rebar_cover = True, 1400.0, 1000.0, 0.040  # kN/m2
    try:
        rep = check_slab_column_drift(r, model=m)
        rep14 = check_slab_column_drift(r, SlabColumnDriftPrefs(code="ACI 318-14"), model=m)
        sec.fpc2 = 800.0  # < 0.9 MPa: designed as nonprestressed (8.6.2.1)
        rep_np = check_slab_column_drift(r, model=m)
    finally:
        m.shell_sections[sec.name] = old
    h = sec.thickness
    fc = m.materials[sec.material].fc / 1000.0
    for c in rep.checks:
        assert c.d == pytest.approx(max(h - 0.040, 0.8 * h))  # 22.6.2.2
        assert c.prestressed and c.exempt == 0.01 and c.limit == pytest.approx(0.040 - 0.05 * c.ratio)
        if c.location == "Interior":
            sq = math.sqrt(fc)
            vc = min(0.29 * sq + 0.3 * 1.2, 0.083 * (1.5 + 40 * c.d / c.bo) * sq + 0.3 * 1.2)
            assert c.vc_clause == "22.6.5.5" and c.vc == pytest.approx(vc)
        else:  # within 4h of an edge: 22.6.5.4(b) not met
            assert c.vc_clause == "22.6.5.2"
    for c in rep14.checks:  # 318-14 has a single (nonprestressed) criterion
        assert not c.prestressed and c.exempt == 0.005
    for c in rep_np.checks:
        assert not c.prestressed and c.vc_clause == "22.6.5.2" and c.limit == pytest.approx(0.035 - 0.05 * c.ratio)


def test_cd_is_required(plate):
    m, r = plate
    m.load_cases["EQX"].cd = None
    try:
        rep = check_slab_column_drift(r, model=m)
    finally:
        m.load_cases["EQX"].cd = 5.5
    assert rep.errors and "Cd" in rep.errors[0] and not rep.checks
    rep = check_slab_column_drift(r, SlabColumnDriftPrefs(combos=["nope"]), model=m)
    assert rep.errors


def test_beams_both_ways_are_not_slab_column_connections():
    m = new_building(x_spacings=(6, 6), y_spacings=(6,), story_heights=(3.0,), grid_only=False)
    m.add_load_pattern("EQX", "Seismic")
    for p in m.points.values():
        if p.z > 0:
            m.add_point_load(p.name, "EQX", force=(10.0, 0, 0))
    m.add_linear_static_case("EQX", {"EQX": 1.0})
    m.load_cases["EQX"].cd = 4.0
    m.add_combination("G+E", {"Dead": 1.2, "Live": 0.5, "EQX": 1.0})
    r = _run(m)
    rep = check_slab_column_drift(r)
    assert rep.checks and all(c.status == "Not checked" for c in rep.checks)
    assert all("two directions" in c.notes[0] for c in rep.checks)


def test_saved_with_the_model():
    m, _ = _flat_plate()
    m.load_cases["EQX"].cd, m.load_cases["EQX"].ie = 5.5, 1.5
    m.load_cases["RSX"].cd = 4.0
    sec = next(s for s in m.shell_sections.values() if s.type == "Slab")
    sec.post_tensioned, sec.fpc1, sec.fpc2, sec.rebar_cover = True, 1200.0, 1100.0, 0.035
    m.design.slab_column_drift = SlabColumnDriftPrefs(code="ACI 318-14", phi=0.7, combos=["G+E"])
    m2 = Model.from_dict(m.to_dict())
    assert (m2.load_cases["EQX"].cd, m2.load_cases["EQX"].ie, m2.load_cases["RSX"].cd) == (5.5, 1.5, 4.0)
    s2 = m2.shell_sections[sec.name]
    assert (s2.post_tensioned, s2.fpc1, s2.fpc2, s2.rebar_cover) == (True, 1200.0, 1100.0, 0.035)
    assert m2.design.slab_column_drift == m.design.slab_column_drift
    m3 = m2.convert_units("N mm C")
    s3 = m3.shell_sections[sec.name]
    assert s3.fpc1 == pytest.approx(1.2) and s3.rebar_cover == pytest.approx(35.0)
    assert m3.load_cases["EQX"].cd == 5.5


# ------------------------------------------------------------- the window
import os  # noqa: E402
import subprocess  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"


@pytest.mark.skipif(CHILD, reason="parent only")
def test_drift_window_in_subprocess():
    pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtk")
    import shutil
    env = dict(os.environ, STRUCTSTUDIO_GUI_CHILD="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "gui_", __file__]
    if not env.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24"] + cmd
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower(), \
        (proc.stdout + proc.stderr)[-4000:]


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_drift_dialog(tmp_path, monkeypatch):
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    from structstudio.app.document import Document
    from structstudio.app.mainwindow import MainWindow
    from structstudio.app.slab_drift_dialog import SlabColumnDriftDialog
    m, _ = _flat_plate()
    w = MainWindow(Document(m), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.show()
    app.processEvents()
    assert "design.slab.drift_check" in w.menu_commands()
    res = _run(w.doc.model)
    w.doc.set_results(res)
    assert w.doc.locked
    d = SlabColumnDriftDialog(w)
    d.run()  # Cd missing: an error, nothing checked
    assert d.report is not None and d.report.errors and "Cd" in d.report.errors[0]
    rows = {d.cases.item(i, 1).text(): i for i in range(d.cases.rowCount())}
    assert set(rows) == {"EQX", "RSX"}
    d.cases.item(rows["EQX"], 2).setText("5.5")
    d.cases.item(rows["RSX"], 2).setText("5")
    d.cases.item(rows["RSX"], 3).setText("1.25")
    d.run()
    # design data stored, the results kept, the model still locked
    assert w.doc.results is res and w.doc.locked
    assert w.doc.model.load_cases["RSX"].cd == 5.0 and w.doc.model.load_cases["RSX"].ie == 1.25
    assert not d.report.errors and len(d.report.checks) == 18
    assert d.table.rowCount() == 18
    ref = check_slab_column_drift(res, model=w.doc.model)
    assert [c.status for c in d.report.checks] == [c.status for c in ref.checks]
    d.tabs.setCurrentWidget(d.plot)
    d.plot.repaint()
    app.processEvents()
    d.select_required()
    assert w.state.frames == {c.column for c in ref.required}
    # slab data: PT
    floor = next(a.section for a in w.doc.model.areas.values() if a.type == "Floor")
    i = next(i for i in range(d.slabs.rowCount()) if d.slabs.item(i, 0).text() == floor)
    d.slabs.item(i, 3).setCheckState(QtCore.Qt.Checked)
    k = d.k_str
    d.slabs.item(i, 4).setText(f"{1200.0 * k:g}")
    d.slabs.item(i, 5).setText(f"{1100.0 * k:g}")
    d.run()
    assert w.doc.model.shell_sections[floor].post_tensioned and d.report.checks[0].prestressed
    assert w.doc.results is res
    d.copy_table()
    assert "Status" in QtWidgets.QApplication.clipboard().text().splitlines()[0]
    d.close()
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    for _ in range(3):
        app.processEvents()
