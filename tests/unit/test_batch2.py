"""Second round of user requests (2026-11-10): model windows (drop-down place, redraw on
resize, fit of a new window, no per-window display button, moving windows without tab
groups), start page, New Model databases and default, fine mesh in Floor Auto Mesh, the
left palette / Loads toolbar, Assign command-first, the hint line, the 3D active-plane
marker, Load > Show Loads, VTK messages on exit.  Window tests run in a child interpreter.
"""
import math
import os
import subprocess
import sys
import time
import types

import pytest

QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from structstudio.app.document import Document  # noqa: E402
from structstudio.app.templates import build_model, new_building  # noqa: E402
from structstudio.model.loads import LoadCombination  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"


# ================================================================ no window
def test_new_model_steel_and_rebar_tables_of_any_region():
    m = build_model(region="thailand", steel_db="us", rebar_db="europe", structure="grid")
    assert "SS400" in m.materials  # materials of the region
    assert any(n.startswith("W") for n in m.frame_sections)  # AISC shapes
    assert any(n.startswith("Ø") for n in m.rebar_sizes) and not any(n.startswith("DB") for n in m.rebar_sizes)
    plain = build_model(region="thailand", structure="grid")
    assert any(n.startswith("DB") for n in plain.rebar_sizes)


def test_fine_mesh_size_text_follows_the_units():
    from structstudio.app.mesh_dialog import fine_mesh_size_text
    from structstudio.units import UnitSystem
    m = new_building(grid_only=False)
    assert fine_mesh_size_text(m) == ""
    o = m.options
    o.fine_mesh_factor, o.fine_mesh_size_min, o.fine_mesh_size_max = 0.5, 0.25, 0.5
    assert fine_mesh_size_text(m) == "0.25-0.5 m"
    mm = m.convert_units(UnitSystem("kN", "mm", "C"))
    assert fine_mesh_size_text(mm) == "250-500 mm"


def test_combination_expands_to_pattern_factors():
    from structstudio.app.load_display import combo_pattern_factors
    m = new_building(grid_only=False)
    c = LoadCombination("U1").add("Dead", 1.2).add("Live", 1.6)
    m.load_combinations["U1"] = c
    m.load_combinations["U2"] = LoadCombination("U2").add("U1", 0.5, is_combo=True).add("Dead", 0.4)
    assert combo_pattern_factors(m, "U1") == {"Dead": 1.2, "Live": 1.6}
    assert combo_pattern_factors(m, "U2") == pytest.approx({"Dead": 1.0, "Live": 0.8})
    m.load_combinations["E"] = LoadCombination("E", type="Envelope").add("Dead")
    with pytest.raises(ValueError, match="Envelope"):
        combo_pattern_factors(m, "E")


def test_load_labels_carry_factor_and_pattern(monkeypatch):
    pytest.importorskip("vtk")
    from structstudio.app import scene
    m = new_building(grid_only=False)
    f = next(iter(m.frames))
    m.add_frame_distributed_load(f, "Dead", 5.0)
    m.add_frame_distributed_load(f, "Live", 2.0)

    seen = []
    real = scene._text_labels

    def capture(pts, texts, *a, **k):
        seen.append([texts.GetValue(i) for i in range(texts.GetNumberOfValues())])
        return real(pts, texts, *a, **k)
    monkeypatch.setattr(scene, "_text_labels", capture)

    def labels(actors):
        return seen.pop() if seen else []
    one = labels(scene.build_load_actors(m, "Dead"))
    assert any(t.startswith("5 ") for t in one) and not any("Dead:" in t for t in one)
    combo = labels(scene.build_load_actors(m, {"Dead": 1.2, "Live": 1.6}))
    assert any(t.startswith("1.2×Dead: 6 ") for t in combo) and any(t.startswith("1.6×Live: 3.2 ") for t in combo)


def test_vtk_messages_go_to_a_file(tmp_path, monkeypatch):
    vtk = pytest.importorskip("vtk")
    from structstudio.app import main
    monkeypatch.setattr(main, "log_path", lambda: str(tmp_path / "structstudio.log"))
    main._vtk_messages_to_file()
    vtk.vtkOutputWindow.GetInstance().DisplayErrorText("context error")
    assert "context error" in (tmp_path / "vtk-messages.log").read_text()


# ================================================================ window (child)
@pytest.mark.skipif(CHILD, reason="parent only")
def test_batch2_window_cases_in_subprocess():
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


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    return app


def _settle(app, n=6, ms=0):
    for _ in range(n):
        if ms:
            QtCore.QThread.msleep(ms)
        app.processEvents()


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    pytest.importorskip("vtk")
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    from structstudio.app.mainwindow import MainWindow
    w = MainWindow(Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3, 3), grid_only=False)),
                   QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.show()
    w.resize(1600, 950)
    _settle(qapp)
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    _settle(qapp)


gui = pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")


@gui
def test_gui_window_header_has_no_display_button_and_lists_open_under_the_box(window, qapp):
    V = window.views
    V.set_layout("3")
    _settle(qapp)
    for vp in V.visible_viewports():
        assert not hasattr(vp, "display_button")
    vp = V.visible_viewports()[2]  # the third window
    vp.kind_box.showPopup()
    _settle(qapp)
    pop = vp.kind_box.view().window()
    top = vp.kind_box.window()
    want = top.mapToGlobal(vp.kind_box.mapTo(top, QtCore.QPoint(0, vp.kind_box.height())))
    assert abs(pop.x() - want.x()) <= 2 and abs(pop.y() - want.y()) <= 2
    vp.kind_box.hidePopup()


@gui
def test_gui_windows_move_left_right_above_below_without_tabs(window, qapp):
    V = window.views
    assert not (V.host.dockOptions() & QtWidgets.QMainWindow.AllowTabbedDocks)
    V.set_layout("3")
    _settle(qapp)
    a, _b, c = V.visible_viewports()

    def geo(v):
        return V.docks[V.viewports.index(v)].geometry()
    for where in ("left", "right", "above", "below"):
        assert V.move_window(a, c, where)
        _settle(qapp)
        ga, gc = geo(a), geo(c)
        ok = {"left": ga.right() < gc.left(), "right": gc.right() < ga.left(),
              "above": ga.bottom() < gc.top(), "below": gc.bottom() < ga.top()}[where]
        assert ok, (where, ga, gc)
        assert not V.host.tabifiedDockWidgets(V.docks[V.viewports.index(a)])
    menu = V.title_menu(a)
    assert [x.text() for x in menu.actions()][:4] == ["To the Left of", "To the Right of", "Above", "Below"]


@gui
def test_gui_new_window_is_fitted_once_it_has_its_size(window, qapp):
    vp = window.views.add_window()
    _settle(qapp, 10, 20)
    assert vp.isVisible() and getattr(vp, "_shown_once", False) and not getattr(vp, "_fit_on_show", True)


@gui
def test_gui_window_redraws_after_a_resize(window, qapp, monkeypatch):
    vp = window.views.visible_viewports()[0]
    _settle(qapp, 10, 20)
    calls = []
    monkeypatch.setattr(vp, "render", lambda: calls.append(1))
    window.resize(1300, 800)
    _settle(qapp, 10, 20)
    assert calls  # a native VTK window may get no paint event: it is drawn again anyway


@gui
def test_gui_3d_window_active_hides_the_blue_plane(window, qapp):
    V = window.views
    plan = next(v for v in V.visible_viewports() if v.config.kind == "plan")
    v3d = next(v for v in V.visible_viewports() if v.config.kind == "3d")
    V.set_active(plan)
    assert V.highlight_target() is not None
    V.set_active(v3d)
    assert V.highlight_target() is None


@gui
def test_gui_hint_line_is_plain_text(window):
    ss = window.hint_label.styleSheet()
    assert "background: transparent" in ss and "fff4c2" not in ss


@gui
def test_gui_compact_flyout_face_shows_the_last_tool(window, qapp):
    window.set_palette_mode("compact")
    b = window.palette_buttons["Draw Beams"]
    b.open_flyout()
    _settle(qapp)
    target = b.row.buttons[0]
    target.click()  # a plain click on a tool of the row
    _settle(qapp)
    assert b.defaultAction() is target.defaultAction()
    window.tool_actions["secondary_beams"].trigger()  # chosen elsewhere (menu / shortcut)
    assert b.defaultAction() is window.tool_actions["secondary_beams"]


@gui
def test_gui_loads_on_top_toolbar_not_in_left_palette(window):
    from structstudio.app.mainwindow import DRAW_PALETTE, LOAD_PALETTE
    loads = window.findChild(QtWidgets.QToolBar, "loads")
    assert all(window.tool_actions[k] in loads.actions() for k in LOAD_PALETTE)
    palette = {k for _t, keys in DRAW_PALETTE for k in keys}
    assert not palette & set(LOAD_PALETTE)


@gui
def test_gui_assign_command_first_then_select_then_enter(window, qapp):
    calls = []

    def assign_test_stub(self):
        if not self._need("frame"):
            return
        calls.append(sorted(self.state.frames))
    window.assign_test_stub = types.MethodType(assign_test_stub, window)
    window.state.clear_selection()
    window.assign_test_stub()
    assert window._pending_assign is not None and not calls
    assert "then press Enter or right-click" in window.hint_label.text()
    window.run_pending_assign()  # nothing selected yet: it keeps waiting
    assert window._pending_assign is not None and not calls
    f = sorted(window.doc.model.frames)[:2]
    window.state.set_selection(frames=f)
    vp = window.views.active
    ev = QtGui_key_event(QtCore.Qt.Key_Return)
    window.eventFilter(vp.vtk_widget, ev)
    _settle(qapp)
    assert calls == [f] and window._pending_assign is None
    # Esc cancels a waiting command
    window.state.clear_selection()
    window.assign_test_stub()
    window.eventFilter(vp.vtk_widget, QtGui_key_event(QtCore.Qt.Key_Escape))
    assert window._pending_assign is None
    # right-click runs it
    window.assign_test_stub()
    window.state.set_selection(frames=f[:1])
    window.show_background_menu(vp, 10, 10)
    assert calls[-1] == f[:1]


def QtGui_key_event(key):
    from PySide6 import QtGui
    return QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier)


@gui
def test_gui_show_loads_dialog_and_toolbar_box(window, qapp):
    from structstudio.app.load_display import ShowLoadsDialog
    m = window.doc.model
    m.load_combinations["U1"] = LoadCombination("U1").add("Dead", 1.2).add("Live", 1.6)
    d = ShowLoadsDialog(m, {"joint", "frame"}, {"Dead": 1.0}, "", window)
    assert d.kinds() == {"joint", "frame"} and not d.generated.isEnabled()
    d.by_combo.setChecked(True)
    d.combo_box.setCurrentText("U1")
    assert d.selection() == ({"Dead": 1.2, "Live": 1.6}, "Combination U1")
    d.by_pattern.setChecked(True)
    for i in range(d.patterns.count()):
        d.patterns.item(i).setCheckState(QtCore.Qt.Checked)
    assert set(d.selection()[0]) == set(m.load_patterns)
    d.deleteLater()
    window._set_load_box({"Dead": 1.2, "Live": 1.6}, "Combination U1")
    assert window.load_box.currentText() == "Combination U1"
    assert window.state.load_factors == {"Dead": 1.2, "Live": 1.6} and window.state.load_pattern is None
    window.load_box.setCurrentText("Live")
    assert window.state.load_factors is None and window.state.load_pattern == "Live"
    load_menu = [a.text() for a in window.menuBar().actions()]
    assert any("Load" in t for t in load_menu)
    assert "load.show" in window.menu_actions
    assert "display.loads.point" not in window.menu_actions


@gui
def test_gui_fine_mesh_lives_in_floor_auto_mesh(window, qapp):
    from structstudio.app.mesh_dialog import AreaMeshDialog
    assert not hasattr(window, "fine_mesh_box")  # not on the toolbar
    asked = []
    m = window.doc.model
    d = AreaMeshDialog("Floor", m, True, None, lambda: asked.append(1), window)
    assert not d.use_fine.isEnabled() and "Not found yet" in d.found.text()
    d.find.click()
    assert asked == [1]
    o = m.options
    o.fine_mesh_factor, o.fine_mesh_size_min, o.fine_mesh_size_max = 0.5, 0.25, 0.25
    d.refresh()
    assert d.use_fine.isEnabled() and d.use_fine.text() == "Use fine mesh (0.25 m)"
    d.none.setChecked(True)
    d.use_fine.setChecked(True)
    assert d.values() == {"auto": False, "size": None, "use_fine": True}
    assert not hasattr(d, "type")  # no mesh type choice
    d.deleteLater()
    window._sync_fine_mesh()
    assert window.act_use_fine.text() == "Use Fine Mesh (0.25 m)"


@gui
def test_gui_start_page_can_be_turned_on_again(window, qapp, monkeypatch):
    from structstudio.app.start_page import KEY_SHOW, StartDialog
    st = window.settings_store()

    def exec_unticked(self):  # the user unticks "Show this window at startup" and closes it
        self.show_check.setChecked(False)
        assert self.show_note.isVisibleTo(self) and "Options" in self.show_note.text()
        return 0
    monkeypatch.setattr(StartDialog, "exec", exec_unticked)
    window.show_start_page()
    assert not StartDialog.show_at_startup(st) and not window.act_start_page.isChecked()
    window.act_start_page.trigger()  # Options > Show Start Page at Startup
    assert StartDialog.show_at_startup(st) and st.value(KEY_SHOW) in (True, "true")


@gui
def test_gui_new_model_defaults_and_databases(qapp, tmp_path):
    from structstudio.app.new_model import NewModelWizard
    w = NewModelWizard(None, QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    try:
        assert w.structure == "grid" and w.cards["grid"].isChecked()
        assert w.steel_db.count() == 4 and w.rebar_db.count() == 4
        w.region.setCurrentIndex(w.region.findData("europe"))
        assert w.steel_db.currentData() == "europe" and w.rebar_db.currentData() == "europe"
        w.steel_db.setCurrentIndex(w.steel_db.findData("china"))
        m = w.build()
        assert any(n.startswith("HW") for n in m.frame_sections)
    finally:
        w.deleteLater()
        _settle(qapp)


@gui
def test_gui_nothing_is_drawn_after_finalize(window, qapp):
    vp = window.views.visible_viewports()[0]
    vp.finalize()
    vp.render()  # a late timer: no drawing into a released context
    window.views.render_all()
    assert vp._finalized
