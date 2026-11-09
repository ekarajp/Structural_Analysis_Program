"""New Model wizard and start page: region settings, the starting structures
(counts, sections, supports, a balanced self-weight analysis), the wizard's
steps (uniform and custom grids, similar stories, settings from a saved
model, saved defaults) and the start page (recent models with pictures,
missing files, show-at-startup).  The widget tests run in a child
interpreter like the other desktop tests."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from structstudio.app.templates import (REGION_ORDER, REGIONS, STRUCTURE_KEYS, build_model, region_properties,
                                        structure_counts)
from structstudio.model import Model

QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"
gui = pytest.mark.skipif(not CHILD, reason="runs in the child interpreter (test_wizard_in_subprocess)")
GRID = dict(x_ordinates=[0, 8, 16, 24], y_ordinates=[0, 7.5, 15, 22.5])
STORIES = [("Story1", 4.5), ("Story2", 3.5), ("Story3", 3.5)]


def _counts(m):
    return dict(columns=sum(f.type == "Column" for f in m.frames.values()),
                beams=sum(f.type == "Beam" for f in m.frames.values()),
                slabs=sum(a.type == "Floor" for a in m.areas.values()),
                walls=sum(a.type == "Wall" for a in m.areas.values()),
                supports=sum(any(p.restraints) for p in m.points.values()))


# ----------------------------------------------------------------- backend
@pytest.mark.parametrize("region", REGION_ORDER)
def test_region_settings_are_complete(region):
    m = Model("t", REGIONS[region]["units"])
    region_properties(m, region)
    types = {mat.type for mat in m.materials.values()}
    assert {"Concrete", "Rebar", "Steel"} <= types
    assert any(s.type == "Slab" for s in m.shell_sections.values())
    assert any(s.type == "Wall" for s in m.shell_sections.values())
    steel = [n for n, s in m.frame_sections.items() if m.materials[s.material].type == "Steel"]
    assert len(steel) >= 3, "the region's steel sections come from the catalog"
    assert len(m.rebar_sizes) >= 10
    assert {"Dead", "Live"} <= set(m.load_patterns)


def test_thai_concrete_follows_eit_modulus():
    m = Model("t", "kgf cm C")
    region_properties(m, "thailand")
    c = m.materials["C240"]
    assert c.fc == pytest.approx(240, rel=1e-3)  # ksc
    assert c.E == pytest.approx(15100 * 240 ** 0.5, rel=1e-3)
    assert m.materials["SD40"].fy == pytest.approx(4000, rel=1e-3)


@pytest.mark.parametrize("kind", STRUCTURE_KEYS)
def test_structure_counts_match_the_model(kind):
    m = build_model(structure=kind, stories=STORIES, **GRID)
    assert _counts(m) == structure_counts(kind, 4, 4, 3)
    assert not m.validate()
    if kind == "blank":
        assert not m.grid_lines and len(m.stories) == 4
    else:
        assert len(m.grid_lines) == 8


def test_core_walls_close_the_middle_bay():
    m = build_model(structure="core_frame", stories=STORIES, **GRID)
    walls = [a for a in m.areas.values() if a.type == "Wall"]
    xy = {(round(m.points[p].x, 6), round(m.points[p].y, 6)) for a in walls for p in a.points}
    assert xy == {(8, 7.5), (16, 7.5), (16, 15), (8, 15)}
    cols = {(round(m.points[f.point_i].x, 6), round(m.points[f.point_i].y, 6)) for f in m.frames.values()
            if f.type == "Column"}
    assert not cols & xy, "no columns inside the walls"
    floors = [a for a in m.areas.values() if a.type == "Floor"]
    centres = {(round(np.mean([m.points[p].x for p in a.points]), 3), round(np.mean([m.points[p].y for p in a.points]), 3))
               for a in floors}
    assert (12.0, 11.25) not in centres, "the core is open"


def test_steel_deck_infill_beams_are_pinned_and_frame_into_girders():
    m = build_model(structure="steel_deck", stories=STORIES[:1], **GRID)
    infill = [f for f in m.frames.values() if f.releases_i[5]]
    assert len(infill) == 9
    for f in infill:
        assert f.releases_j[4] and f.releases_j[5]
        pi = m.points[f.point_i]
        assert any(g.point_i == f.point_i or g.point_j == f.point_i for g in m.frames.values()
                   if g is not f and g.type == "Beam"), "the girder is split where the infill beam lands"
        assert m.materials[m.frame_sections[f.section].material].type == "Steel"


@pytest.mark.parametrize("kind", ["rc_frame", "core_frame", "steel_deck", "flat_slab"])
def test_self_weight_reaches_the_supports(kind):
    from structstudio.api import analyze
    m = build_model(structure=kind, stories=STORIES[:2], **GRID)
    res = analyze(m, cases=["Dead"])
    w = 0.0
    for f in m.frames.values():
        s = m.frame_sections[f.section]
        a, b = m.points[f.point_i], m.points[f.point_j]
        w += s.A * np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]) * m.materials[s.material].weight_density
    for ar in m.areas.values():
        s = m.shell_sections[ar.section]
        P = np.array([[m.points[p].x, m.points[p].y, m.points[p].z] for p in ar.points])
        n = np.cross(P[1] - P[0], P[2] - P[0]) + np.cross(P[2] - P[0], P[3] - P[0])
        w += np.linalg.norm(n) / 2 * s.thickness * m.materials[s.material].weight_density
    assert res.base_reactions("Dead")[2] == pytest.approx(w, rel=1e-6)


def test_supports_choice_and_settings_from_a_saved_model():
    src = build_model(units="kgf cm C", structure="rc_frame", stories=STORIES[:1], **GRID)
    src.materials["C240"].notes = "office mix"
    m = build_model(units="kgf cm C", structure="flat_slab", settings_from=src, support="Pinned",
                    x_ordinates=[0, 600, 1200], y_ordinates=[0, 500], stories=[("S1", 350)])
    assert m.materials["C240"].notes == "office mix"
    assert not m.grid_lines[0].ordinate and m.units.length == "cm"
    sup = [p for p in m.points.values() if any(p.restraints)]
    assert len(sup) == 6 and all(p.restraints == (True, True, True, False, False, False) for p in sup)


def test_bad_grids_are_refused():
    with pytest.raises(ValueError):
        build_model(structure="rc_frame", x_ordinates=[0, 6, 6], y_ordinates=[0, 5], stories=STORIES)
    with pytest.raises(ValueError):
        build_model(structure="grid", stories=[("A", 3.0), ("A", 3.0)], **GRID)
    with pytest.raises(ValueError):
        build_model(structure="grid", stories=[("A", 0.0)], **GRID)


def test_sketch_layers():
    from structstudio.app.sketch import model_sketch, sketch_items
    sk = model_sketch(build_model(structure="core_frame", stories=STORIES[:1], **GRID))
    assert set(sketch_items(sk)) == {"grid", "bubble", "floor", "wall", "beam", "column", "support"}
    assert json.loads(json.dumps(sk)) == sk


# ------------------------------------------------------------------- widgets
@pytest.mark.skipif(CHILD, reason="parent only")
def test_wizard_in_subprocess():
    import shutil
    env = dict(os.environ, STRUCTSTUDIO_GUI_CHILD="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "gui_", __file__]
    if not env.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1600x1000x24"] + cmd
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower(), \
        (proc.stdout + proc.stderr)[-4000:]


@pytest.fixture(scope="module")
def app():
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    for w in app.topLevelWidgets():
        w.close()
        w.deleteLater()
    app.processEvents()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    return QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat)


def _wizard(settings):
    from structstudio.app.new_model import NewModelWizard
    w = NewModelWizard(None, settings)
    w.show()
    return w


@gui
def test_gui_wizard_uniform_rc_frame(app, settings):
    w = _wizard(settings)
    assert w.btn_next.text().startswith("Next") and w.btn_alt.text() == "Blank model"
    w.u_nx.setValue(3)
    w.u_xs.setValue(6.0)
    w.u_ny.setValue(2)
    w.u_ns.setValue(2)
    w._next()
    assert w.pages.currentIndex() == 1
    assert w.plan_preview.xs == [0.0, 6.0, 12.0] and w.plan_preview.xl == ["A", "B", "C"]
    assert [n for n, _ in w.elev_preview.levels] == ["Base", "Story1", "Story2"]
    w._next()
    assert w.pages.currentIndex() == 2 and w.cards["rc_frame"].isChecked()
    assert all(w.cards[k].sketch for k in ("grid", "rc_frame", "core_frame", "steel_deck"))
    assert "12 columns" in w.summary.text()
    w._next()
    assert w.result() == QtWidgets.QDialog.Accepted
    m = w.model()
    assert _counts(m)["columns"] == 12 and m.stories["Story1"].height == pytest.approx(4.5)
    assert m.frame_sections[next(f.section for f in m.frames.values() if f.type == "Column")].name.startswith("C")


@gui
def test_gui_units_change_rescales_the_grid(app, settings):
    w = _wizard(settings)
    w.units.setCurrentText("kN mm C")
    assert w.u_xs.value() == pytest.approx(8000.0)
    assert w.u_h1.value() == pytest.approx(4500.0)
    assert "mm" in w.u_h_label.text()
    w.structure = "grid"
    m = w.build()
    assert m.units.length == "mm" and max(g.ordinate for g in m.grid_lines if g.direction == "X") == pytest.approx(24000)


@gui
def test_gui_custom_tables(app, settings):
    w = _wizard(settings)
    w.go(1)
    w.mode_custom.setChecked(True)
    assert w.t_x.rowCount() == 4 and w.t_st.rowCount() == 9  # 8 stories + base
    # a spacing typed moves this line and the ones after it
    w.t_x.item(1, 2).setText("6")
    assert [r[1] for r in w._axis_rows(w.t_x)] == [0, 6, 14, 22]
    # ordinates typed: the table sorts itself
    w.enter_by.setCurrentIndex(1)
    w.t_x.setCurrentCell(3, 1)
    w._table_action(w.t_x, "add")
    assert [r[1] for r in w._axis_rows(w.t_x)] == [0, 6, 14, 22, 30]
    assert w._axis_rows(w.t_x)[-1][0] == "E"
    w.t_x.setCurrentCell(1, 1)
    w._table_action(w.t_x, "delete")
    assert [r[0] for r in w._axis_rows(w.t_x)] == ["A", "C", "D", "E"]
    w._table_action(w.t_x, "relabel")
    assert [r[0] for r in w._axis_rows(w.t_x)] == ["A", "B", "C", "D"]
    w.t_x.cellWidget(0, 3).setCurrentText("Start")
    w.t_x.item(3, 4).setCheckState(QtCore.Qt.Unchecked)
    # stories: heights, a master and a similar story
    w.t_st.item(7, 1).setText("5.0")  # Story1 (rows top to bottom; base last)
    assert w.t_st.item(7, 2).text() == "5" and w.t_st.item(0, 2).text() == "29.5"
    w.t_st.item(7, 3).setCheckState(QtCore.Qt.Checked)
    box = w.t_st.cellWidget(6, 4)
    assert box.findText("Story1") >= 0
    box.setCurrentText("Story1")
    w.t_st.setCurrentCell(0, 0)
    w._table_action(w.t_st, "above")
    assert w.t_st.rowCount() == 10 and w.t_st.item(0, 0).text() == "Story9"
    w.structure = "grid"
    m = w.build()
    xs = sorted((g.ordinate, g.name, g.bubble_location, g.visible) for g in m.grid_lines if g.direction == "X")
    assert xs == [(0, "A", "Start", True), (14, "B", "End", True), (22, "C", "End", True), (30, "D", "End", False)]
    assert m.stories["Story1"].is_master and m.stories["Story2"].similar_to == "Story1"
    assert len(m.stories) == 10 and m.stories["Story9"].elevation == pytest.approx(33.0)


@gui
def test_gui_grid_errors_stop_next(app, settings, monkeypatch):
    w = _wizard(settings)
    w.go(1)
    w.mode_custom.setChecked(True)
    w.t_st.item(0, 1).setText("0")
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a: shown.append(a[2])))
    w._next()
    assert w.pages.currentIndex() == 1 and "greater than zero" in shown[-1]


@gui
def test_gui_defaults_and_saved_model(app, settings, tmp_path):
    w = _wizard(settings)
    assert not w.src_defaults.isEnabled()
    w.units.setCurrentText("tonf m C")
    w.region.setCurrentIndex(w.region.findData("europe"))
    w.save_defaults.setChecked(True)
    w._alt()  # Blank model
    assert w.model().units.force == "tonf" and "C30/37" in w.model().materials
    assert json.loads(settings.value("new_model/defaults")) == {"units": "tonf m C", "region": "europe"}
    w2 = _wizard(settings)
    assert w2.src_defaults.isEnabled()
    w2.src_defaults.setChecked(True)
    assert w2.units.currentText() == "tonf m C" and w2._region_key() == "europe"
    # settings from a saved model: its units and materials, no geometry
    src = build_model(units="kgf cm C", structure="rc_frame", stories=STORIES[:1], **GRID)
    path = str(tmp_path / "src.ssj.json")
    src.to_json(path)
    w3 = _wizard(settings)
    w3.src_model.setChecked(True)
    assert not w3.btn_next.isEnabled()
    w3.src_path.setText(path)
    w3._load_source()
    assert w3.btn_next.isEnabled() and "cm" in w3.u_h_label.text()
    w3.structure = "grid"
    m = w3.build()
    assert m.units.length == "cm" and "C240" in m.materials and not m.frames


@gui
def test_gui_start_page_recent_models(app, settings, tmp_path, monkeypatch):
    from structstudio.app.start_page import KEY_SHOW, RecentModels, StartDialog
    rec = RecentModels(settings)
    paths = []
    for name, kind in (("a", "rc_frame"), ("b", "grid")):
        m = build_model(structure=kind, stories=STORIES[:1], **GRID)
        p = str(tmp_path / f"{name}.ssj.json")
        m.to_json(p)
        rec.add(p, m)
        paths.append(p)
    rec.add(paths[0])  # opened again: first, keeps its summary and picture
    e = rec.entries()
    assert [x["name"] for x in e] == ["a", "b"] and "1 story" in e[0]["info"] and rec.sketch(paths[0])
    os.remove(paths[1])
    d = StartDialog(settings, None, rec)
    d.show()
    app.processEvents()
    assert d.list.count() == 2 and d.btn_go.isEnabled()
    d.search.setText("b.ssj")
    assert d.list.count() == 1
    d.list.setCurrentRow(0)
    assert not d.btn_go.isEnabled(), "a missing file cannot be opened"
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a: QtWidgets.QMessageBox.Yes))
    d._open_item(d.list.currentItem())
    assert len(rec.entries()) == 1
    d.search.clear()
    d.list.setCurrentRow(0)
    d._open_item(d.list.currentItem())
    assert d.choice == ("recent", os.path.abspath(paths[0]))
    d.show_check.setChecked(False)
    assert StartDialog.show_at_startup(settings) is False
    d2 = StartDialog(settings, None, rec)
    d2.btn_new.click()
    assert d2.choice == ("new",)
    d2._set_view(False)
    assert settings.value("start/view") == "list"


@gui
def test_gui_main_window_start_page_and_recent_menu(app, settings, tmp_path, monkeypatch):
    pytest.importorskip("vtk")
    from structstudio.app.document import Document
    from structstudio.app.mainwindow import MainWindow
    from structstudio.app.new_model import NewModelWizard
    from structstudio.app.start_page import StartDialog
    w = MainWindow(Document(), settings)
    w.show()
    app.processEvents()
    path = str(tmp_path / "m.ssj.json")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (path, "")))
    w.save_model_as()
    assert w.recent.entries()[0]["path"] == os.path.abspath(path)
    titles = [a.text() for a in w.menus["File"].actions()]
    assert "Recent Models" in titles and "Start Page..." in titles
    rec_menu = next(a.menu() for a in w.menus["File"].actions() if a.text() == "Recent Models")
    w._fill_recent_menu(rec_menu)
    assert rec_menu.actions()[0].text().endswith("m")
    # start page -> New Model wizard -> model in the window
    picked = []

    def fake_start_exec(self):
        self.choice = ("new",)
        return 1

    def fake_wizard_exec(self):
        picked.append(self)
        self.cards["flat_slab"].setChecked(True)
        self.u_ns.setValue(1)
        self._finish()
        return 1
    monkeypatch.setattr(StartDialog, "exec", fake_start_exec)
    monkeypatch.setattr(NewModelWizard, "exec", fake_wizard_exec)
    assert w.show_start_page() == ("new",)
    assert picked and _counts(w.doc.model)["slabs"] == 9
    assert w.act_start_page.isChecked()
    w.act_start_page.trigger()
    assert StartDialog.show_at_startup(settings) is False
    w.close()
