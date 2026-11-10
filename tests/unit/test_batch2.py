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
    """Side moves never make tabs (tabs only by the title bar or "As a Tab with")."""
    V = window.views
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
    """Floor Auto Mesh Options: one of auto (read-only default max) / manual maximum /
    fine (only once found) / no mesh, and the allowed error of Find Fine Mesh."""
    from structstudio.app.mesh_dialog import AreaMeshDialog
    assert not hasattr(window, "fine_mesh_box")  # not on the toolbar
    asked = []
    m = window.doc.model
    d = AreaMeshDialog("Floor", m, "auto", None, lambda: asked.append(1), window)
    assert d.auto.isChecked() and d.default.isReadOnly() and d.default.text() == "Default = max 1.2 m"
    assert not d.fine.isEnabled() and "not found yet" in d.found.text()
    assert not d.size.isEnabled()
    d.manual.setChecked(True)
    assert d.size.isEnabled() and not d.auto.isChecked()  # exclusive
    d.size.setText("0.5")
    assert d.values()["mode"] == "manual" and d.values()["size"] == 0.5
    d.size.setText("-1")
    with pytest.raises(ValueError):
        d.values()
    d.tol.setValue(1.0)
    d.find.click()  # Find uses the allowed error shown
    assert asked == [1] and m.options.mesh_refine_tolerance == 0.01
    o = m.options
    o.fine_mesh_factor, o.fine_mesh_size_min, o.fine_mesh_size_max = 0.5, 0.25, 0.25
    d.refresh()
    assert d.fine.isEnabled() and d.fine.text() == "Fine mesh (0.25 m)"
    d.fine.setChecked(True)
    v = d.values()
    assert v["mode"] == "fine" and v["fine"] and v["auto"] and v["size"] is None
    d.none.setChecked(True)
    assert d.values()["mode"] == "none" and not d.values()["auto"]
    assert not hasattr(d, "type")  # no mesh type choice
    d.deleteLater()


@gui
def test_gui_assign_area_mesh_sets_the_mode(window, qapp):
    """Floor Auto Mesh Options in the box at the lower left: the mode of the floor
    selected before; Apply keeps the box, OK (Enter) ends."""
    from structstudio.app.templates import new_building
    window.doc.model = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m = window.doc.model
    floor = next(a for a in m.areas.values() if a.type == "Floor")
    m.options.fine_mesh_factor = 0.5
    st = window.state
    for mode, check in (("manual", lambda a: a.max_mesh_size == 0.4 and not a.fine_mesh),
                        ("fine", lambda a: a.fine_mesh and a.max_mesh_size is None),
                        ("none", lambda a: not a.auto_mesh and not a.fine_mesh),
                        ("auto", lambda a: a.auto_mesh and not a.fine_mesh and a.max_mesh_size is None)):
        st.set_tool("select")
        st.set_selection(areas=[floor.name])
        window.assign_area_mesh("Floor")
        assert st.tool.name.startswith("assign:") and window.assign_window.isVisible() and \
            window.assign_window.windowTitle() == "Assign - Floor Auto Mesh Options"
        assert window.dock_properties_of_object.windowTitle() == "Properties of Object"
        d = window.assign_panel._form
        getattr(d, mode).setChecked(True)
        if mode == "manual":
            d.size.setText("0.4")
        window.assign_panel.apply_button.click()  # Apply: assigned, the box stays
        assert check(window.doc.model.areas[floor.name]), mode
        assert st.tool.name.startswith("assign:") and window.assign_panel._form is d
    st.set_selection(areas=[floor.name])
    window.assign_panel.ok_button.click()  # OK: assign and finish
    assert st.tool.name == "select"


@gui
def test_gui_assign_forms_in_the_box(window, qapp):
    """Command first: the box opens at once; pick, Apply (stay), Enter (finish); objects
    first: the box shows their values; Cancel leaves."""
    from structstudio.app.templates import new_building
    window.doc.model = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m = window.doc.model
    st = window.state
    beams = [n for n, f in m.frames.items() if f.type == "Beam"][:2]
    st.set_tool("select")
    window.assign_frame_mesh()
    p = window.assign_panel
    assert st.tool.name == "assign:frame.mesh" and p._form is not None and p.title.text().startswith("Assign -")
    assert window.assign_window.isVisible() and not p.apply_button.isEnabled() and p.ok_button.isEnabled()
    assert window.object_props._form is None  # nothing at the lower left
    p._form.set_value("auto", False)
    st.toggle("frame", beams[0], True)
    p.apply_button.click()
    assert window.doc.model.frames[beams[0]].auto_mesh is False and st.tool.name == "assign:frame.mesh"
    st.toggle("frame", beams[1], True)
    vp = window.views.visible_viewports()[0]
    vp.enter()  # Enter: assign and finish
    assert window.doc.model.frames[beams[1]].auto_mesh is False and st.tool.name == "select"
    # objects first: the values of the selected frame
    st.set_selection([beams[0]])
    window.assign_frame_mesh()
    assert p._form.values()["auto"] is False
    p.cancel_button.click()
    assert st.tool.name == "select" and not window.assign_window.isVisible()
    assert window.assign_frame_mesh() is None and window.assign_window.isVisible()
    window.assign_window.reject()  # X / Esc on the window: cancel
    assert st.tool.name == "select" and not window.assign_window.isVisible()
    # Assign > Slab / Wall: split menus, a wall command changes walls only
    from structstudio.app.menus import ETABS_MENUS
    assign = dict((t, e) for t, e in ETABS_MENUS)["&Assign"]
    subs = [t for t, *_ in assign if isinstance(t, str)]
    assert "Slab" in subs and "Wall" in subs and "Shell" not in subs
    floor = next(n for n, a in m.areas.items() if a.type == "Floor")
    window.menu_actions["assign.wall_mass"].trigger()
    st.toggle("area", floor, True)
    p._form.set_value("v", 1.0)
    p.apply_button.click()  # a floor picked in a wall command: nothing changes
    assert window.doc.model.areas[floor].mass_per_area == 0.0
    p.cancel_button.click()


@gui
def test_gui_restraints_in_the_box(window, qapp):
    from structstudio.app.templates import new_building
    window.doc.model = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m = window.doc.model
    j = next(n for n, pt in m.points.items() if pt.z > 0)
    st = window.state
    st.set_selection(points=[j])
    window.assign_restraints()
    p = window.assign_panel
    assert window.assign_window.isVisible() and window.assign_window.windowTitle() == "Assign - Restraints"
    w = p._form
    for b in w.boxes:
        b.setChecked(True)
    window.views.visible_viewports()[0].enter()
    assert all(window.doc.model.points[j].restraints) and st.tool.name == "select"
    assert not window.assign_window.isVisible()


@gui
def test_gui_analyze_menu_has_no_fine_mesh_items_and_loads_define_first(window, qapp):
    from structstudio.app.menus import ETABS_MENUS as MENUS
    labels = {}

    def walk(items, path):
        for it in items:
            if isinstance(it, tuple):
                if isinstance(it[1], list):
                    walk(it[1], path + (it[0],))
                else:
                    labels.setdefault(path, []).append(it[1])
    walk(MENUS, ())
    ana = labels[("A&nalyze",)]
    assert "analyze.check_mesh" not in ana and "analyze.use_fine_mesh" not in ana
    load = labels[("&Load",)]
    assert load.index("define.load_patterns") < load.index("tool.area_load")
    tb = window.findChild(QtWidgets.QToolBar, "loads")
    acts = [a for a in tb.actions() if not a.isSeparator()]
    first_tool = min(i for i, a in enumerate(acts) if a in window.tool_actions.values())
    assert all(a not in window.tool_actions.values() for a in acts[:first_tool]) and first_tool >= 3


@gui
def test_gui_window_toolbars_menu_shows_and_hides(window, qapp):
    names = [tb.objectName() for tb in window.top_toolbars()]
    assert {"main", "edit", "define", "loads", "analyze", "navigation", "display"} <= set(names)
    menu = QtWidgets.QMenu()
    window.fill_toolbars_menu(menu)
    a = next(x for x in menu.actions() if x.objectName() == "toolbar:loads")
    assert a.isChecked()
    a.setChecked(False)
    tb = window.findChild(QtWidgets.QToolBar, "loads")
    assert tb.isHidden() and "loads" in str(window.qsettings.value("ui/hidden_toolbars"))
    window._restore_toolbars()
    assert tb.isHidden()
    window.show_all_toolbars()
    assert not tb.isHidden() and "loads" not in str(window.qsettings.value("ui/hidden_toolbars"))


@gui
def test_gui_show_loads_applied_has_help(window, qapp):
    from structstudio.app.load_display import ShowLoadsDialog
    d = ShowLoadsDialog(window.doc.model, ["joint"], parent=window)
    assert d.applied_help.text() == "?"
    d.deleteLater()


@gui
def test_gui_move_window_as_a_tab_and_by_title(window, qapp):
    V = window.views
    V.set_layout("3")
    _settle(qapp)
    a, b, _c = V.visible_viewports()
    menu = QtWidgets.QMenu()
    V.fill_move_menu(menu, a)
    tab = next(x for x in menu.actions() if x.text() == "As a Tab with").menu()
    tab.actions()[0].trigger()
    _settle(qapp)
    assert V.docks[V.viewports.index(b)] in V.host.tabifiedDockWidgets(V.docks[V.viewports.index(a)])


@gui
def test_gui_right_click_runs_a_waiting_assign(window, qapp, monkeypatch):
    """Command first: a right click on an object runs the waiting command, not the
    object's properties."""
    from structstudio.app.templates import new_building
    window.doc.model = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m = window.doc.model
    floor = next(a for a in m.areas.values() if a.type == "Floor")
    window.state.clear_selection()
    asked = []
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem",
                        lambda *a, **k: (asked.append(1), ("", False))[1])
    window.assign_group()  # nothing selected: the command waits
    assert window._pending_assign is not None and not asked
    assert window._pending_right_click(("area", floor.name)) is True
    assert asked == [1] and window._pending_assign is None and window._have("area")
    # no command waiting: the right click is the normal one
    assert window._pending_right_click(("area", floor.name)) is False


@gui
def test_gui_drawing_preview_is_shared_and_esc_clears_every_window(window, qapp):
    from structstudio.app.tools import Preview
    V = window.views
    V.set_layout("2")
    _settle(qapp)
    a, b = V.visible_viewports()[:2]
    pv = Preview(lines=[((0, 0, 0), (3, 0, 0))])
    window.state.preview_shown.emit(a, pv)
    assert b._mirror_actors and not a._mirror_actors
    b._preview_actors.append(b._lines_actor([((0, 0, 0), (1, 0, 0))], (1, 0, 0), 2.0))  # stale own preview
    window.state.preview_shown.emit(a, pv)
    assert not b._preview_actors and b._mirror_actors
    a.escape()  # Esc in the first window: the second one drops the drawing too
    assert not b._mirror_actors and not b._preview_actors


@gui
def test_gui_no_qt_window_menu_on_right_click_and_header_click_activates(window, qapp):
    V = window.views
    V.set_layout("2")
    _settle(qapp)
    a, b = V.visible_viewports()[:2]
    assert V.host.contextMenuPolicy() == QtCore.Qt.PreventContextMenu
    assert a.vtk_widget.contextMenuPolicy() == QtCore.Qt.PreventContextMenu
    V.set_active(a)
    from PySide6.QtTest import QTest
    QTest.mouseClick(b.header, QtCore.Qt.LeftButton, pos=QtCore.QPoint(5, 5))
    assert V.active is b
    t = V.docks[V.viewports.index(a)].titleBarWidget()
    QTest.mousePress(t, QtCore.Qt.LeftButton, pos=QtCore.QPoint(5, 5))
    QTest.mouseRelease(t, QtCore.Qt.LeftButton, pos=QtCore.QPoint(5, 5))
    assert V.active is a


@gui
def test_gui_popups_from_a_model_window_belong_to_the_program_window(window, qapp):
    """Menus / forms opened from inside the (native) model windows area get the program
    window as their owner (else on Windows a form can open behind it: looks hung)."""
    vp = window.views.visible_viewports()[0]
    window.show()
    _settle(qapp)
    d = QtWidgets.QDialog(vp.header)
    d.show()
    _settle(qapp)
    assert d.windowHandle().transientParent() is window.windowHandle()
    d.close()
    d.deleteLater()


@gui
def test_gui_hang_watchdog_writes_the_stacks(qapp, tmp_path, monkeypatch):
    import time
    from structstudio.app import main as app_main
    app = qapp
    monkeypatch.setattr(app_main, "log_path", lambda: str(tmp_path / "structstudio.log"))
    monkeypatch.setattr(app_main, "HANG_SECONDS", 1.5)
    stop = app_main._start_hang_watchdog(app)
    try:
        time.sleep(3.5)  # the event loop does not run: a "hang"
    finally:
        stop.set()
    files = list(tmp_path.glob("hang-*.txt"))
    assert files and "Thread" in files[0].read_text(encoding="utf-8")


@gui
def test_gui_explorer_assign_section_opens_the_command(window, qapp, monkeypatch):
    from structstudio.app.templates import new_building
    window.doc.model = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m = window.doc.model
    window.tree.refresh() if hasattr(window.tree, "refresh") else None
    beam = next(n for n, f in m.frames.items() if f.type == "Beam")
    window._tree_object_action("assign_section", "frames", beam)
    assert window.state.tool.name == "frame_section" and window.state.frames == {beam}
    window.state.set_tool("select")
    floor = next(n for n, a in m.areas.items() if a.type == "Floor")
    window._tree_object_action("assign_section", "areas", floor)
    assert window.state.tool.name == "slab_section" and window.state.areas == {floor}
    p = window.assign_panel
    assert p.section_list.count() >= 1 and "Define sections" in p.define_button.text()
    # a new section made with Define sections... is the one chosen
    from structstudio.model.properties import ShellSection
    mat = next(iter(m.materials))
    monkeypatch.setattr(window, "define_shell_sections",
                        lambda: window.doc.edit(lambda mm: mm.shell_sections.__setitem__(
                            "S300", ShellSection("S300", mat, "Slab", thickness=0.3))))
    p.define_button.click()
    assert window.state.settings.shell_section == "S300"
    window.state.apply_load_tool()
    assert m.areas[floor].section == "S300" if window.doc.model is m else window.doc.model.areas[floor].section == "S300"


@gui
def test_gui_project_information(window, qapp, monkeypatch, tmp_path):
    from structstudio.app import project_info as pi
    from structstudio.model import Model

    def fake_exec(self):
        self._widgets["name"].setText("Hospital A")
        self._widgets["license_no"].setText("ส-12345")
        self._widgets["revision"].setText("R1")
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(pi.ProjectInfoDialog, "exec", fake_exec)
    window.menu_actions["file.project_info"].trigger()
    info = window.doc.model.info
    assert info.name == "Hospital A" and info.license_no == "ส-12345" and info.revision == "R1"
    back = Model.from_dict(window.doc.model.to_dict())
    assert back.info.license_no == "ส-12345"
    # an older file (four fields) still opens
    d = window.doc.model.to_dict()
    d["info"] = {"name": "Old", "engineer": "", "company": "", "notes": ""}
    assert Model.from_dict(d).info.name == "Old"


@gui
def test_gui_load_panel_commands(window, qapp):
    """Load Panel: a small window of commands; panels are picked only in them; shape /
    load / move / copy / delete; Display Options > Load panels hides them."""
    from structstudio.app.mainwindow import LOAD_PALETTE
    from structstudio.app.templates import new_building
    assert LOAD_PALETTE == ["area_load", "frame_load", "joint_load", "panel_load"]
    window.doc.model = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m = window.doc.model
    pat = next(iter(m.load_patterns))
    from structstudio.model.objects import PanelLoadCase
    z = m.stories["Story1"].elevation
    pl = m.add_panel_load([(1, 1, z), (3, 1, z), (3, 3, z), (1, 3, z)], "two_way",
                          loads=[PanelLoadCase(pat, 2.0, "Gravity")])
    window.doc.edit(lambda mm: None)
    st = window.state
    window.tool_actions["panel_load"].trigger()
    d = window._panel_commands
    assert d.isVisible() and set(d.buttons) == {"draw", "shape", "load", "move", "copy", "delete"}
    assert st.tool.name == "select" and window.tool_actions["select"].isChecked()
    d.buttons["draw"].click()
    assert st.tool.name == "panel_load" and not d.isVisible()
    # delete: only panels can be picked, Apply deletes
    window.panel_command("delete")
    assert st.tool.picks == ("panel",) and window.assign_window.isVisible()
    st.toggle("panel", pl.name, True)
    window.assign_panel.ok_button.click()
    assert pl.name not in window.doc.model.panel_loads and st.tool.name == "select"
    window.doc.undo()
    assert pl.name in window.doc.model.panel_loads
    # move: OK starts Move with the panel selected
    window.panel_command("move")
    assert not window.assign_panel.apply_button.isVisible()
    st.toggle("panel", pl.name, True)
    window.assign_panel.ok_button.click()
    assert st.tool.name == "move" and st.panels == {pl.name} and not st.tool.selecting
    st.set_tool("select")
    # shape: OK, then the new outline replaces the old one (loads kept)
    st.clear_selection()
    window.panel_command("shape")
    st.toggle("panel", pl.name, True)
    window.assign_panel.ok_button.click()
    assert st.tool.name == "panel_load" and st.tool.reshape_of == pl.name
    window.add_panel_load([(0, 0, z), (4, 0, z), (4, 4, z)])
    new = window.doc.model.panel_loads[pl.name]
    assert len(new.points) == 3 and new.loads[0].value == 2.0 and st.tool.name == "select"
    # Display Options > Load panels off: hidden, but shown while a panel command runs
    vp = window.views.visible_viewports()[0]
    st.clear_selection()  # (a selected panel is always drawn)
    o = vp.display_options
    o.load_panels = False
    vp.set_display_options(o)
    vp.rebuild() if hasattr(vp, "rebuild") else None
    vp._build_panels()
    assert not vp._panel_actors
    window.panel_command("load")
    assert vp._panel_actors
    window.assign_panel.cancel_button.click()
    assert not vp._panel_actors


@gui
def test_gui_capture_pictures_have_the_drawings(window, qapp, tmp_path, monkeypatch):
    """Capture Picture: the model views are not black (native OpenGL windows painted in);
    Selected Region crops; no Entire Screen / Create Video / Print Graphics in File."""
    from structstudio.app import capture
    from structstudio.app.menus import ETABS_MENUS
    file_menu = dict(ETABS_MENUS)["&File"]
    labels = [e[0] for e in file_menu if isinstance(e, tuple)]
    assert "Create Video" not in labels and "Print Graphics..." not in labels
    cap = dict((e[0], e[1]) for e in file_menu if isinstance(e, tuple))["Capture Picture"]
    assert [e[0] for e in cap] == ["Main Window", "Current Window", "Selected Region (drag a rectangle)..."]
    window.resize(1400, 900)
    _settle(qapp, 10, 20)
    vp = window.views.visible_viewports()[0]
    pm = capture.compose(window, window.views.viewports)
    img = pm.toImage()
    c = vp.vtk_widget.mapTo(window, vp.vtk_widget.rect().center())
    k = pm.devicePixelRatio()
    col = img.pixelColor(int(c.x() * k), int(c.y() * k))
    bg = [int(255 * v) for v in __import__("structstudio.app.scene", fromlist=["COLORS"]).COLORS["background_top"]]
    assert (col.red(), col.green(), col.blue()) != (0, 0, 0) and max(col.red(), col.green(), col.blue()) > 40
    saved = []
    monkeypatch.setattr(capture, "save_picture", lambda parent, p, title, name: saved.append((title, p.size())) or "x")

    def fake_exec(self):
        self.rect_logical = QtCore.QRect(10, 10, 200, 100)
        return 1
    monkeypatch.setattr(capture.RegionPicker, "exec", fake_exec)
    window.menu_actions["file.capture_region"].trigger()
    window.menu_actions["file.capture_window"].trigger()
    assert saved[0][0] == "Capture Selected Region" and saved[0][1].width() >= 200
    assert saved[1][0] == "Capture Current Window"
    del bg


@gui
def test_gui_insertion_point(window, qapp, monkeypatch):
    """Cardinal point: a named drop-down + ... (picture of the section); the offset goes
    into the analysis or only into the drawing; Assign > Frame > Insertion Point."""
    from structstudio.app import insertion_dialog as ins
    from structstudio.app.tools import _cardinal_fields
    from structstudio.app.templates import new_building
    from structstudio.app.menus import ETABS_MENUS
    labels = [e[0] for e in dict(ETABS_MENUS)["&File"] if isinstance(e, tuple)]
    assert "Comments and Log..." not in labels and "Show Input/Output Text Files..." not in labels
    window.doc.model = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m = window.doc.model
    st = window.state
    st.set_tool("beam")
    ed = window.object_props.editors["cardinal_point"]
    assert ed.box.currentText().startswith("Auto") and ed.more.text() == "..."
    # Auto: beams at the top (8), columns at the centroid; analysed at the centroid
    assert _cardinal_fields(st.settings, "Beam") == {"cardinal_point": 8, "display_cardinal": 8,
                                                     "offsets_in_analysis": False}
    assert _cardinal_fields(st.settings, "Column") == {"cardinal_point": 10, "display_cardinal": 10}
    assert st.settings.cardinal_transform is False

    def fake_exec(self):
        v = self.form.values()
        assert v["cp"] == 8 and v["transform"] is False and v["shift"] == (0.0, 0.0)  # (Auto for a beam: 8)
        assert not self.form.mode.isEnabled()  # drawing: dx, dy only
        self.form.picture.picked.emit(2)  # a click on point 2
        self.form.transform.setChecked(True)
        return 1
    monkeypatch.setattr(ins.InsertionPointDialog, "exec", fake_exec)
    ed.more.click()
    assert st.settings.cardinal_point == 2 and st.settings.cardinal_transform is True
    assert _cardinal_fields(st.settings, "Beam") == {"cardinal_point": 2, "display_cardinal": 2,
                                                     "offsets_in_analysis": True}
    st.settings.cardinal_transform = False
    st.settings.cardinal_point = 10  # chosen 10: drawn at the centroid (no more drawn like 8)
    assert _cardinal_fields(st.settings, "Beam") == {"cardinal_point": 10, "display_cardinal": 10}
    st.settings.cardinal_point = None
    st.set_tool("select")
    # the picture follows the choice
    w = ins.InsertionPointWidget(m, next(iter(m.frame_sections)), 3, True)
    assert w.picture.cardinal == 3
    w.cardinal.setCurrentIndex(6)
    assert w.picture.cardinal == 7
    # Assign > Frame > Insertion Point (in the Assign window)
    beam = next(n for n, f in m.frames.items() if f.type == "Beam")
    st.set_selection([beam])
    window.assign_insertion_point()
    f = window.assign_panel._form
    assert isinstance(f, ins.InsertionPointWidget) and f.values()["cp"] == 8  # beams are drawn at the top
    assert f.values()["transform"] is False  # ... without the eccentric stiffness (as before)
    f.cardinal.setCurrentIndex(1)
    f.transform.setChecked(True)
    f.mode.setCurrentIndex(1)  # each end, global X Y Z
    f.offset_boxes[("End-I", 2)].setValue(0.1)
    window.assign_panel.ok_button.click()
    fr = window.doc.model.frames[beam]
    assert fr.cardinal_point == 2 and fr.display_cardinal == 2 and fr.joint_offset_i[2] == pytest.approx(0.1)


@gui
def test_gui_insertion_shift_and_window_size(window, qapp):
    """Insertion Point window opens big enough; dx / dy from the cardinal point are joint
    offsets along the section (local 3 / 2, both ends), drawn and in the analysis."""
    from structstudio.app import extrude as ex
    from structstudio.app.templates import new_building
    window.doc.model = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m = window.doc.model
    st = window.state
    beam = next(n for n, f in m.frames.items() if f.type == "Beam")
    st.set_selection([beam])
    window.assign_insertion_point()
    _settle(qapp)
    f = window.assign_panel._form
    win = window.assign_window
    assert win.width() >= f.sizeHint().width() and win.height() >= f.sizeHint().height()
    assert f.mode.currentIndex() == 0 and f.local_box.isVisibleTo(f) and not f.global_box.isVisibleTo(f)
    f.dy.setValue(-0.2)
    assert f.picture.offset == (-0.2, 0.0) and not f.shift_warn.text()
    f.dx.setValue(5.0)
    assert "rigid arm" in f.shift_warn.text()
    f.dx.setValue(0.1)
    window.assign_panel.ok_button.click()
    fr = window.doc.model.frames[beam]
    assert fr.joint_offset_local and fr.joint_offset_i == pytest.approx((0.0, -0.2, 0.1)) \
        and fr.joint_offset_j == pytest.approx((0.0, -0.2, 0.1))
    a, b = ex.frame_ends(window.doc.model, fr)  # drawn along the local axes
    pi = window.doc.model.points[fr.point_i].xyz
    assert abs((a - pi)[2] - (-0.2)) < 1e-9  # a level beam: local 2 = +Z
    # opened again: the same dx / dy
    st.set_selection([beam])
    window.assign_insertion_point()
    f = window.assign_panel._form
    assert f.mode.currentIndex() == 0 and f.dy.value() == pytest.approx(-0.2) and f.dx.value() == pytest.approx(0.1)
    assert f.dx.suffix().strip() == "m" and f.offset_boxes[("End-I", 0)].suffix().strip() == "m"  # units shown
    f.mode.setCurrentIndex(1)  # each end, global: dx / dy converted (a level beam along X: local 2 = Z, 3 = -Y)
    vi = [f.offset_boxes[("End-I", c)].value() for c in range(3)]
    vj = [f.offset_boxes[("End-J", c)].value() for c in range(3)]
    assert vi == pytest.approx(vj) and abs(vi[2] - (-0.2)) < 1e-9 and f.picture_j.isVisibleTo(f)
    assert f.picture.offset == pytest.approx((-0.2, 0.1)) and f.picture_j.offset == pytest.approx((-0.2, 0.1))
    f.offset_boxes[("End-J", 2)].setValue(0.5)

    def scale(pic):
        to_px = pic._frame()[2]
        return to_px(0.0, 1.0).x() - to_px(0.0, 0.0).x()
    assert scale(f.picture) == pytest.approx(scale(f.picture_j))  # one beam: the same section size at both ends
    f.mode.setCurrentIndex(0)  # back: End-I's dx / dy, and a note that End-J differed
    assert f.dy.value() == pytest.approx(-0.2) and f.dx.value() == pytest.approx(0.1) and "End-J" in f.mode_note.text()
    f.mode.setCurrentIndex(1)
    f.offset_boxes[("End-J", 0)].setValue(0.3)
    f.transform.setChecked(True)  # (at the bottom: cardinal point and joint offsets in the analysis)
    window.assign_panel.ok_button.click()
    fr = window.doc.model.frames[beam]
    assert not fr.joint_offset_local and fr.joint_offset_j[0] == pytest.approx(0.3) and fr.offsets_in_analysis


@gui
def test_gui_draw_rigid_link(window, qapp):
    """Draw > Draw Rigid Links: a Rigid line object (no section), drawn in its own
    colour, listed in the explorer, picked by Select > Rigid links; analysed as an
    exact rigid link."""
    from structstudio.app.menus import ETABS_MENUS
    from structstudio.app.tools import select_by_type
    labels = [e[0] for e in dict(ETABS_MENUS)["D&raw"] if isinstance(e, tuple)]
    assert "Draw Rigid Links" in labels
    st = window.state
    st.set_tool("rigid_link")
    t = st.tool
    t.set_story("Story1")
    t.click((0, 0, 3))
    t.click((-1.2, 0, 3))
    _settle(qapp)
    f = window.doc.model.frames[t.created[-1]]
    assert f.type == "Rigid" and f.name.startswith("RL")
    names, _, _ = select_by_type(window.doc.model, "rigid links")
    assert names == [f.name]
    vp = next(v for v in window.views.viewports if v.config.kind == "3d")
    vp._build_scene()
    assert any(k.startswith("frames_Rigid") for k in vp._scene_keys), vp._scene_keys
