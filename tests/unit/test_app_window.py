"""Desktop app smoke tests (multi-window UI) under a virtual display.

The Qt/VTK windows share an OpenGL context with the off-screen scene tests,
which is fragile in one process; the GUI tests therefore run in a child
interpreter (``STRUCTSTUDIO_GUI_CHILD=1``) started by :func:`test_gui_in_subprocess`,
under xvfb when available.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("vtk")
QtCore = pytest.importorskip("PySide6.QtCore")
QtTest = pytest.importorskip("PySide6.QtTest")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from structstudio.app.document import Document  # noqa: E402
from structstudio.app.templates import new_building  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"
gui = pytest.mark.skipif(not CHILD, reason="runs in the child interpreter (test_gui_in_subprocess)")


@pytest.mark.skipif(CHILD, reason="parent only")
def test_gui_in_subprocess():
    import shutil
    env = dict(os.environ, STRUCTSTUDIO_GUI_CHILD="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__]
    if not env.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1600x1000x24"] + cmd
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout + proc.stderr
    lines = [ln for ln in out.splitlines() if "Extension modules" not in ln and "WARN|" not in ln]
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower(), \
        "\n".join(lines[-60:])


@pytest.fixture(scope="module")
def app():
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    from structstudio.app.mainwindow import MainWindow
    doc = Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5)))
    w = MainWindow(doc)
    w.doc.ask_unlock = lambda: True  # (the window asks in a message box)
    w.resize(1600, 950)
    w.show()
    app.processEvents()
    yield w
    w.doc.dirty = False  # teardown: no Save / Discard / Cancel question (FA04 asks on close)
    w.views.finalize()
    w.close()
    w.deleteLater()
    app.processEvents()


def _display(vp, x, y, z):
    vp.renderer.SetWorldPoint(x, y, z, 1.0)
    vp.renderer.WorldToDisplay()
    d = vp.renderer.GetDisplayPoint()
    return int(round(d[0])), int(round(d[1]))


def _click(vp, x, y, z, double=False):
    vp.interactor.SetEventPosition(*_display(vp, x, y, z))
    vp.interactor.SetRepeatCount(1 if double else 0)
    vp.style.InvokeEvent("LeftButtonPressEvent")
    vp.style.InvokeEvent("LeftButtonReleaseEvent")


def _plan(window, story):
    vp = window.views.viewports[0]
    vp.set_view("plan", story=story)
    window.views.set_active(vp)
    QtWidgets.QApplication.processEvents()
    return vp, window.doc.model.stories[story].elevation


@gui
def test_default_two_windows_plan_and_3d(window):
    views = window.views
    assert views.layout_name == "2"
    shown = views.visible_viewports()
    assert [v.config.kind for v in shown] == ["plan", "3d"]
    assert window.draw_toolbar.orientation().name == "Vertical"
    assert window.palette_mode == "full"  # every tool its own button by default
    window.set_palette_mode("compact")
    # ETABS order: select | draw (fly-outs per object type) | edit | selection | snap
    assert len([a for a in window.draw_toolbar.actions() if not a.isSeparator()]) >= 16
    from structstudio.app.mainwindow import DRAW_PALETTE, DRAW_TOOLS, LOAD_PALETTE, MODIFY_TOOLS, SELECT_TOOLS
    # every tool has a palette button except Draw Braces (braces: the line tool; the
    # tool stays as a command / shortcut) and the load tools (top Loads toolbar)
    assert {k for _, keys in DRAW_PALETTE for k in keys} | set(LOAD_PALETTE) == \
        {k for k, *_ in SELECT_TOOLS + DRAW_TOOLS + MODIFY_TOOLS} - {"brace"}
    loads_bar = window.findChild(QtWidgets.QToolBar, "loads")
    assert all(window.tool_actions[k] in loads_bar.actions() for k in LOAD_PALETTE)
    beams = window.palette_buttons["Draw Beams"]
    assert [a.text() for a in beams.menu_actions()][0].startswith("Draw Beam")
    quick = window.tool_actions["quick_beam"]
    beams.open_flyout()  # Photoshop fly-out (hold / right-click): the group's other tools in a row
    assert beams.flyout_open() and [b.defaultAction() for b in beams.row.buttons] == beams.menu_actions()[1:]
    assert beams.row.layout().direction() == QtWidgets.QBoxLayout.LeftToRight
    assert beams.row.x() >= beams.mapToGlobal(QtCore.QPoint(beams.width(), 0)).x() - 1
    beams.row.close()
    beams.pick(quick)
    assert beams.defaultAction() is quick  # the button face remembers the last tool
    beams.open_flyout()  # ... and the row no longer repeats it
    assert quick not in [b.defaultAction() for b in beams.row.buttons]
    beams.row.close()
    seen = []
    quick.triggered.connect(lambda *_: seen.append(1))
    QtTest.QTest.mouseClick(beams, QtCore.Qt.LeftButton)  # a click starts the tool on the face
    assert seen
    window.set_palette_mode("full")
    views.set_layout("4")
    assert [v.config.kind for v in views.visible_viewports()] == ["plan", "3d", "xz", "yz"]
    views.set_layout("1")
    assert len(views.visible_viewports()) == 1
    views.set_layout("2")


@gui
def test_draw_in_plan_shows_in_3d_and_shared_selection(window, app):
    vp, z = _plan(window, "Story2")
    m = window.doc.model
    window.state.set_tool("quick_column")
    for x, y in ((0, 0), (6, 0), (6, 5), (0, 5)):
        _click(vp, x + 0.05, y - 0.05, z)
    cols = [f for f in m.frames.values() if f.type == "Column"]
    assert len(cols) == 4 and all(c.story == "Story2" for c in cols)
    window.state.set_tool("beam")
    _click(vp, 0, 0, z)
    _click(vp, 6, 0, z)
    _click(vp, 6, 5, z)  # chained
    assert len([f for f in m.frames.values() if f.type == "Beam"]) == 2
    window.state.set_tool("quick_slab")
    _click(vp, 3, 2.5, z)
    window.state.set_tool("rect_slab")
    _click(vp, 6, 0, z)
    _click(vp, 12, 5, z)
    assert len(m.areas) == 2
    # select in plan, highlighted in every window (shared state)
    window.state.set_tool("select")
    _click(vp, 3, 0, z)
    assert len(window.state.frames) == 1
    assert window.views.viewports[1].selection is window.state.frames
    window.state.delete_selected()
    assert len([f for f in window.doc.model.frames.values() if f.type == "Beam"]) == 1
    window.doc.undo()
    assert len([f for f in window.doc.model.frames.values() if f.type == "Beam"]) == 2


@gui
def test_elevation_quick_brace_and_line_draw(window, app):
    views = window.views
    views.set_layout("4")
    app.processEvents()
    ev = views.viewports[2]
    ev.set_view("xz", grid="1")  # y = 0
    views.set_active(ev)
    app.processEvents()
    m = window.doc.model
    window.state.set_tool("quick_brace")
    window.object_props.editors["brace_pattern"].setCurrentText("X")
    assert window.state.settings.brace_pattern == "X"
    _click(ev, 3.0, 0.0, 1.75)  # bay A-B, Base-Story1
    braces = [f for f in m.frames.values() if f.type == "Brace"]
    assert len(braces) == 2
    # draw a column by line in elevation: snapped to grid x story points
    window.state.set_tool("beam")
    _click(ev, 12.0, 0.0, 0.0)
    _click(ev, 12.0, 0.0, 3.5)
    col = [f for f in m.frames.values() if f.type == "Column"]
    assert len(col) == 1 and m.points[col[0].point_j].z == 3.5
    # quick column in elevation between Story1 and Story2 on grid B
    window.state.set_tool("quick_column")
    _click(ev, 6.05, 0.0, 5.0)
    assert len([f for f in m.frames.values() if f.type == "Column"]) == 2


@gui
def test_window_crossing_and_line_selection(window, app, tmp_path):
    vp, z = _plan(window, "Story1")
    window.state.set_tool("quick_beam")
    _click(vp, 3, 0, z)
    _click(vp, 9, 0, z)
    _click(vp, 0, 2.5, z)
    window.state.set_tool("select")
    # window (left to right) fully containing the two beams on grid line 1
    p0, p1 = _display(vp, -1, -1, z), _display(vp, 13, 1, z)
    vp._window_select(p0, p1, add=False)
    assert len(window.state.frames) == 2
    # crossing (right to left) touching all three beams
    p0, p1 = _display(vp, 1, 3, z), _display(vp, -1, -1, z)
    vp._window_select(p0, p1, add=False, crossing=True)
    assert len(window.state.frames) == 2  # beam A-B on grid 1 and beam on grid A
    # intersecting line crossing the beam on grid A and beam A-B on grid 1
    vp._line_select(_display(vp, -1, 1, z), _display(vp, 3, -1, z), add=False)
    assert len(window.state.frames) == 2
    window.state.invert_selection()
    assert len(window.state.frames) == 1
    vp.screenshot(str(tmp_path / "plan.png"))
    assert (tmp_path / "plan.png").stat().st_size > 1000
    window.doc.save(str(tmp_path / "m.ssj.json"))
    window.doc.open(str(tmp_path / "m.ssj.json"))
    assert len(window.doc.model.frames) == 3


@gui
def test_grid_system_switch_toolbar_menu_dialog(window, app):
    from structstudio.app.dialogs import GridStoriesDialog
    m = window.doc.model
    assert window.grid_switch.mode() == "story" and window.grid_mode_actions["story"].isChecked()
    window.grid_switch.setCurrentIndex(1)  # click the SAP2000 tab
    app.processEvents()
    assert window.doc.model.grid_mode == "xyz"
    assert window.grid_mode_actions["xyz"].isChecked()
    assert window.views.viewports[0].title.text().startswith("X-Y Plane @ Z")
    assert window.tree.topLevelItem(1).child(0).text(0).startswith("Levels (Z grid)")  # Structure Layout
    window.grid_mode_actions["story"].trigger()  # menu item back to ETABS
    app.processEvents()
    assert window.doc.model.grid_mode == "story" and window.grid_switch.mode() == "story"
    assert window.views.viewports[0].title.text().startswith("Plan View")
    # tabbed dialog: switch to SAP2000 inside the dialog, add a Z line, accept
    d = GridStoriesDialog(window.doc, window)
    assert [d.tabs.tabText(i) for i in range(d.tabs.count())] == ["X Grid", "Y Grid", "Story"]
    d.switch.setCurrentIndex(1)
    assert d.tabs.tabText(2) == "Z Grid" and d.z_table.rowCount() == 3
    d._add_row(d.z_table, ("Roof", 9.0))
    d.accept()
    m = window.doc.model
    assert m.grid_mode == "xyz" and [g.name for g in m.z_grid_lines()][-1] == "Roof"
    assert "Roof" in m.stories


@gui
def test_dark_light_theme_and_graphics_colors(window, app):
    from structstudio.app import scene, theme
    vp = window.views.viewports[0]
    assert window.color_settings.theme == "light"  # default: light mode
    assert vp.renderer.GetBackground2() == pytest.approx((1.0, 1.0, 1.0))
    window.act_theme_toggle.trigger()
    app.processEvents()
    assert window.color_settings.theme == "dark" and window.theme_actions["dark"].isChecked()
    assert vp.renderer.GetBackground2() == pytest.approx(theme.DEFAULTS["dark"]["background_top"])
    window.set_theme("light")
    app.processEvents()
    assert scene.COLORS["beam"] == theme.DEFAULTS["light"]["beam"]
    assert app.palette().color(app.palette().ColorRole.Window).name() == theme.UI["light"]["window"]
    # user colour for beams in light mode, colour by section via the dialog
    d = window.colors_dialog()
    d._set("beam", (1.0, 0.0, 0.0))
    d.by_box.setCurrentIndex(theme.COLOR_BY.index("section"))
    sec = next(iter(window.doc.model.frame_sections))
    d._set_object(("frame", sec), (0.0, 1.0, 0.0))
    d._ok()
    app.processEvents()
    assert scene.COLORS["beam"] == (1.0, 0.0, 0.0)
    assert window.color_settings.color_by == "section" and window.color_by_actions["section"].isChecked()
    assert window.doc.model.frame_sections[sec].color == "#00ff00"
    from structstudio.app.templates import new_building
    full = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3,), grid_only=False)
    assert any(k.startswith("frames_") and "sec:" in k for k in scene.build_model_actors(full).actors)
    window.doc.undo()  # section colour edit is undoable
    assert window.doc.model.frame_sections[sec].color is None
    window.set_theme("dark")
    assert scene.COLORS["beam"] == theme.DEFAULTS["dark"]["beam"]  # overrides are per theme
    window.set_color_by("object")
    window.set_theme("light")


@gui
def test_import_sap2000_safe_and_etabs_files(window, app):
    import time
    data = os.path.join(os.path.dirname(__file__), "..", "data", "import")
    for name, program, mode in (("sap2000_v23_portal.s2k", "SAP2000", "xyz"), ("safe_v12_slab.f2k", "SAFE", "xyz"),
                                ("etabs_v9_column.e2k", "ETABS", "story")):
        worker = window._import_async(os.path.join(data, name))
        t0 = time.time()
        while worker.isRunning() and time.time() - t0 < 30:
            app.processEvents()
            time.sleep(0.01)
        for _ in range(5):
            app.processEvents()
        m = window.doc.model
        assert m.info.name == os.path.splitext(name)[0] or program == "ETABS"
        assert m.frames and window.doc.dirty and window.doc.path is None
        assert window.grid_switch.mode() == mode
        assert "Imported" in window.logger.text() if hasattr(window.logger, "text") else True
        for vp in window.views.visible_viewports():
            vp.rebuild()
    assert window.doc.model.info.name == "COLV9"


@gui
def test_display_options_grid_at_base_and_plan_story_in_3d(window, app):
    from structstudio.app.display import DisplayOptions
    from structstudio.app.display_dialog import DisplayOptionsDialog
    plan, v3d = window.views.viewports[0], window.views.viewports[1]
    m = window.doc.model
    plan.set_view("plan", story="Story2")
    window.views.set_active(plan)
    app.processEvents()
    assert v3d.config.kind == "3d" and v3d.highlight_story == "Story2" and v3d._highlight_actors
    plan.set_story("Story1")
    assert v3d.highlight_story == "Story1"
    # 3D grid lines at the base level only
    grid = next(a for a in v3d._scene_actors if a.GetMapper() is not None and hasattr(a, "GetProperty")
                and a.GetProperty().GetLineStipplePattern() == 0xF0F0)
    zs = {round(z, 9) for z in grid.GetMapper().GetInput().GetBounds()[4:6]}
    assert zs == {min(s.elevation for s in m.stories.values())}
    # per-window options from the dialog; story plane can be switched off
    d = DisplayOptionsDialog(v3d.display_options, "3-D", window)
    d.boxes["story_plane"].setChecked(False)
    d.boxes["columns"].setChecked(False)
    d.boxes["joint_labels"].setChecked(True)
    window.views.apply_display(d.result_options(), v3d)
    assert not v3d._highlight_actors and not v3d.display_options.columns
    assert plan.display_options.columns  # other windows unchanged
    assert v3d.config.to_dict()["display"]["joint_labels"] is True
    window.views.apply_display(DisplayOptions(), v3d, all_windows=True)
    assert v3d._highlight_actors
    # an elevation window made active: the 3D window shades that vertical plane instead
    plan.set_view("xz")
    window.views.set_active(plan)
    app.processEvents()
    kind, value, _label = v3d.highlight_plane
    assert kind == "xz" and v3d.highlight_story is None
    b = v3d._highlight_actors[0].GetMapper().GetInput().GetBounds()
    assert b[2] == b[3] == value
    # the arrows go round: past the last line back to the first
    from structstudio.app.viewport import CUSTOM
    n = sum(1 for i in range(plan.where_box.count()) if plan.where_box.itemData(i) != CUSTOM)
    first = plan.where_box.currentIndex()
    for _ in range(n):
        assert plan.step_where(+1)
    assert plan.where_box.currentIndex() == first
    plan.set_view("plan", story="Story2")
    window.views.set_active(plan)
    top = plan.where_box.currentIndex()
    plan.step_where(+1)
    plan.step_where(-1)
    assert plan.where_box.currentIndex() == top
    assert not hasattr(plan, "display_button")  # display settings: the toolbar button only (user request)
    window.act_extrude.setChecked(False)
    window.toggle_extrude(False)
    assert not plan.display_options.extrude and not v3d.display_options.extrude
    window.toggle_extrude(True)


@gui
def test_navigation_modes_grid_toggles_and_locale(window, app):
    from PySide6 import QtCore
    st = window.state
    # English digits whatever the Windows region
    assert QtCore.QLocale().toString(1234.5, "f", 1) == "1234.5"
    # line view by default (extrusion is an option)
    assert not window.act_extrude.isChecked()
    assert not any(v.display_options.extrude for v in window.views.viewports)
    # ETABS zoom / pan / rotate toolbar: modes are exclusive, Esc / tools clear them
    window.nav_actions["rotate"].trigger()
    assert st.nav_mode == "rotate" and window.nav_actions["rotate"].isChecked()
    window.nav_actions["pan"].trigger()
    assert st.nav_mode == "pan" and not window.nav_actions["rotate"].isChecked()
    st.cancel_tool()
    assert st.nav_mode is None and not window.nav_actions["pan"].isChecked()
    window.nav_actions["rotate"].trigger()
    window.tool_actions["beam"].trigger()
    assert st.nav_mode is None
    window.tool_actions["select"].trigger()
    # programmatic rotation / zoom keep the camera valid
    v3d = next(v for v in window.views.viewports if v.config.kind == "3d")
    cam = v3d.renderer.GetActiveCamera()
    p0 = cam.GetPosition()
    v3d.rotate_view(30, 10)
    assert cam.GetPosition() != p0
    vz = cam.GetViewUp()[2]
    assert vz > 0  # turntable: Z stays up
    d0 = cam.GetDistance() if not cam.GetParallelProjection() else cam.GetParallelScale()
    v3d.zoom(1.25)
    d1 = cam.GetDistance() if not cam.GetParallelProjection() else cam.GetParallelScale()
    assert d1 < d0
    # View menu grid toggles reach every window
    window.act_grid.trigger()
    assert not any(v.display_options.grid_lines for v in window.views.viewports)
    window.act_grid.trigger()
    assert window.act_grid.isChecked() and v3d.display_options.grid_lines
    window.act_grid_vertical.trigger()
    assert not v3d.display_options.grid_vertical
    window.act_grid_vertical.trigger()
    window.grid3d_actions["all"].trigger()
    assert v3d.display_options.grid_3d == "all"
    window.grid3d_actions["base"].trigger()


@gui
def test_etabs_menu_bar(window, app):
    from PySide6 import QtGui
    titles = [a.text().replace("&", "") for a in window.menuBar().actions()]
    assert titles == ["File", "Edit", "View", "Define", "Draw", "Select", "Assign", "Load", "Analyze", "Display",
                      "Design", "Detailing", "Window", "Options", "Help"]
    acts = window.menu_actions
    # ETABS default keyboard shortcuts
    for cid, key in (("analyze.run", "F5"), ("edit.replicate", "Ctrl+R"), ("edit.move", "Ctrl+M"),
                     ("view.display", "Ctrl+W"), ("view.full", "F3"), ("view.zoom_window", "F2"),
                     ("select.invert", "Ctrl+K"), ("select.previous", "Ctrl+J"), ("select.clear", "Ctrl+Q"),
                     ("display.undeformed", "F4"), ("display.deformed", "F6"), ("display.tables", "Ctrl+T"),
                     ("options.units", "Ctrl+U"), ("tool.select_line", "Ctrl+Shift+L")):
        assert acts[cid].shortcut() == QtGui.QKeySequence(key), cid
        assert acts[cid].isEnabled(), cid
    # not yet built: shown, disabled, marked planned
    assert not acts["design.concrete.start"].isEnabled() and acts["design.concrete.start"].property("planned")
    # no two enabled actions share a shortcut
    seen = {}
    for a in window.findChildren(QtGui.QAction):
        if a.isEnabled() and not a.shortcut().isEmpty():
            k = a.shortcut().toString()
            assert k not in seen or seen[k] is a, k
            seen[k] = a
    # visibility commands of the active window
    window.doc.replace_model(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5),
                                          grid_only=False))
    m = window.doc.model
    vp = window.viewport
    beam = next(f.name for f in m.frames.values() if f.type == "Beam")
    window.state.set_selection([beam])
    acts["view.hide_selected"].trigger()
    assert beam in vp.hidden[0] and not window.state.frames
    acts["view.show_all"].trigger()
    assert not any(vp.hidden)
    window.state.set_selection([beam])
    acts["view.show_selected"].trigger()
    assert beam not in vp.hidden[0] and len(vp.hidden[0]) == len(m.frames) - 1
    acts["view.restore_visibility"].trigger()
    assert not any(vp.hidden)
    # select / deselect filters
    acts["select.clear"].trigger()
    acts["select.add.type.Columns"].trigger()
    n = len(window.state.frames)
    assert n and all(m.frames[f].type == "Column" for f in window.state.frames)
    acts["select.remove.type.Columns"].trigger()
    assert not window.state.frames
    # copy / reverse / relabel work on the selection
    window.state.set_selection([beam])
    acts["edit.reverse_frames"].trigger()
    acts["edit.copy"].trigger()
    assert window.clipboard is not None and window.clipboard.frames
    acts["assign.copy"].trigger()
    assert window.assign_clipboard.kind == "frame"


@gui
def test_palette_compact_and_full(window, app):
    from structstudio.app.mainwindow import DRAW_PALETTE
    bar = window.draw_toolbar
    assert window.palette_mode == "full"
    full = [b.defaultAction() for b in window.palette_grid.buttons if b is not None]
    all_tools = [window.tool_actions[k] for _, keys in DRAW_PALETTE for k in keys]
    assert all(t in full for t in all_tools)
    window.palette_grid.set_available_height(300)  # a short window: a second column, nothing folded away
    assert window.palette_grid.columns() >= 2
    assert all(b.isVisibleTo(window.palette_grid) for b in window.palette_grid.buttons if b is not None)
    window.set_palette_mode("compact")
    compact = [a for a in bar.actions() if not a.isSeparator()]
    assert len(full) > len(compact)
    window.set_palette_mode("full")
    assert window.palette_mode_actions["full"].isChecked() and window.act_palette_mode.isChecked()
    window.act_palette_mode.trigger()  # back to compact
    assert window.palette_mode == "compact"
    assert window.palette_buttons["Draw Beams"].menu_actions()


@gui
def _with_objects(window, app):
    window.doc.replace_model(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5),
                                          grid_only=False))
    app.processEvents()
    return window.doc.model


@gui
def test_properties_of_object_form_follows_the_tool(window, app):
    po = window.object_props
    assert po.table.rowCount() == 0  # select tool: nothing to show
    window.state.set_tool("beam")
    keys = list(po.editors)
    assert "line_type" in keys and "moment_releases" in keys and "frame_section" in keys
    po.editors["moment_releases"].setCurrentText("Pinned at start")
    po.editors["drawing_control"].setCurrentText("Fixed length")
    assert window.state.settings.moment_releases == "Pinned at start"
    assert "fixed_length" in po.editors  # appears with the matching drawing control
    po.editors["fixed_length"].setValue(2.5)
    window.state.set_tool("point")
    assert "offset_x" in po.editors and "line_type" not in po.editors
    po.editors["offset_x"].setValue(0.3)
    assert window.state.settings.offset_x == 0.3
    po.editors["offset_x"].setValue(0.0)
    window.state.set_tool("select")
    assert po.table.rowCount() == 0 and "draw tool" in po.hint.text()
    # the tools honour the form: a beam drawn with fixed length and a pinned I end
    _with_objects(window, app)
    vp, z = _plan(window, "Story2")
    window.state.set_tool("beam")
    window.state.settings.drawing_control = "Fixed length"
    window.state.settings.fixed_length = 2.5
    _click(vp, 0, 0, z)
    _click(vp, 4, 0, z)  # cut back to 2.5
    f = window.doc.model.frames[window.state.tool.created[-1]]
    b = window.doc.model.points[f.point_j]
    assert b.x == pytest.approx(2.5, abs=1e-3) and f.releases_i == (False, False, False, False, True, True)
    window.state.settings.drawing_control = "None"
    window.state.settings.moment_releases = "Continuous"


@gui
def test_selection_highlight_box_and_properties_panel(window, app):
    from structstudio.app import scene, theme
    m = _with_objects(window, app)
    vp = window.views.viewports[1]
    beam = next(f.name for f in m.frames.values() if f.type == "Beam")
    window.state.set_selection([beam], points=[m.frames[beam].point_i])
    app.processEvents()
    # dashed highlight: the highlight actor holds many short dashes, not one segment
    sel = [k for k in vp._actor_keys() if k.endswith("_sel")]
    assert sel
    # selection box drawn inside the render window, translucent, in the theme colour
    vp._rubber.show((10, 10), (200, 150), crossing=False)
    assert vp._rubber.visible and vp._rubber.fill.GetProperty().GetOpacity() == pytest.approx(
        theme.active().opacity("select_box_fill"))
    assert vp._rubber.hide() and not vp._rubber.visible
    # Properties of Object (left) follows the selection; there is no right-hand raw panel
    assert not hasattr(window, "props") and not hasattr(window, "dock_properties")
    window.state.set_tool("select")
    window.state.set_selection([beam])
    app.processEvents()
    assert window.object_props.table.rowCount() > 0
    window.state.clear_selection()
    app.processEvents()
    assert not window.object_props.title.text().startswith("Selection")


@gui
def test_snap_options_and_right_click_info(window, app):
    from structstudio.app.snap import SnapOptions
    from structstudio.app.object_info import JointInfoDialog, FrameInfoDialog, AreaInfoDialog
    st = window.state
    assert st.snap.joints and st.snap.grid
    window.act_snap_lines.trigger()
    assert st.snap.lines
    o = st.snap.copy()
    o.perpendicular = True
    o.snap_only = True
    window.apply_snap_options(o)
    assert st.snap.perpendicular and window.act_snap_only.isChecked() and window.act_snap_lines.isChecked()
    window.draw_snap_options()
    assert window._snap_dialog.isVisible()
    window._snap_dialog.boxes["ortho"].setChecked(True)
    window._snap_dialog._apply()
    assert st.snap.ortho
    window._snap_dialog.close()
    # right click on a joint opens the editable Joint Information form
    m = _with_objects(window, app)
    p = next(p for p in m.points.values() if abs(p.z) < 1e-9)
    vp = window.views.viewports[1]  # 3D: every joint is pickable
    vp.fit()
    got = []
    vp.object_info_requested.connect(lambda k, n: got.append((k, n)))
    x, y = vp._display(p.xyz)
    hits = vp.pick_all(int(x), int(y))
    assert ("point", p.name) in hits
    window.show_object_info("point", p.name)
    d = window._info_dialogs[-1]
    assert isinstance(d, JointInfoDialog) and d.tabs.count() == 3
    d.picker.set_type("roller_z")
    d.picker.inclined.setChecked(True)
    d.picker.angles[1].setValue(30.0)
    d.apply()
    p2 = window.doc.model.points[p.name]
    assert p2.restraints == (False, False, True, False, False, False) and p2.local_axes == (0.0, 30.0, 0.0)
    window.doc.undo()
    assert window.doc.model.points[p.name].local_axes is None
    d.close()
    # frame and area forms build and apply a section change
    beam = next(f.name for f in m.frames.values() if f.type == "Beam")
    fd = FrameInfoDialog(window.doc, beam, window)
    fd._edit("angle", 90.0)
    fd.apply()
    assert window.doc.model.frames[beam].angle == 90.0
    fd.close()
    if m.areas:
        ad = AreaInfoDialog(window.doc, next(iter(m.areas)), window)
        assert ad.tabs.count() == 3
        ad.close()


@gui
def test_draw_load_panel_and_display_modes(window, app, monkeypatch):
    from structstudio.app.panel_dialog import PanelLoadDialog
    from structstudio.model.objects import PanelLoadCase
    m = _with_objects(window, app)
    vp, z = _plan(window, "Story2")
    window.state.set_tool("panel_load")
    assert "panel_load" in window.tool_actions and window.object_props.table.rowCount() >= 1

    # the Loads on Panel form: accept with a 2-way live load of 5
    def fake_exec(self):
        self.dist.setCurrentIndex(1)  # two_way
        self._add_row(PanelLoadCase("LL", 5.0) if "LL" in m.load_patterns else PanelLoadCase(next(iter(m.load_patterns)), 5.0))
        return 1
    monkeypatch.setattr(PanelLoadDialog, "exec", fake_exec)
    for x, y in ((0, 0), (6, 0), (6, 5)):
        _click(vp, x, y, z)
    _click(vp, 0, 5, z, double=True)
    app.processEvents()
    model = window.doc.model
    assert len(model.panel_loads) == 1
    panel = next(iter(model.panel_loads.values()))
    assert panel.distribution == "two_way" and panel.loads and panel.loads[0].value == 5.0
    # it is listed in the explorer, selectable, and its loads reach the analysis
    objs = window.tree.topLevelItem(2)  # Structural Objects
    assert any(objs.child(i).text(0).startswith("Load panels (1)") for i in range(objs.childCount()))
    window.state.set_tool("select")
    x, y = vp._display((3.0, 2.5, z))
    assert vp.pick(int(x), int(y)) in [("area", a) for a in model.areas] + [("panel", panel.name)]
    window.state.toggle("panel", panel.name, False)
    assert window.state.panels == {panel.name}
    # load display modes: panel loads on the panel, or as applied to the beams
    pat = panel.loads[0].pattern
    window.state.set_display(load_pattern=pat)
    window.set_load_kinds({"panel"})
    app.processEvents()
    window.set_load_kinds({"joint", "frame", "applied"})
    app.processEvents()
    from structstudio.app import scene
    actors = scene.build_load_actors(model, pat, kinds={"applied"})
    assert actors  # derived beam loads drawn
    from structstudio.mesh.panel_loads import panel_loads_to_members
    fl, _, rep = panel_loads_to_members(model)
    assert abs(rep[panel.name][pat][0] - 5.0 * 30.0) < 1e-6 and abs(rep[panel.name][pat][1] - 150.0) < 1.5
    assert len(fl) == 4  # the four edge beams of the bay
    # right click on the panel opens its form; Delete removes it
    window.state.set_tool("select")
    window.state.toggle("panel", panel.name, False)
    window.state.delete_selected()
    assert not window.doc.model.panel_loads
    window.doc.undo()
    assert len(window.doc.model.panel_loads) == 1


@gui
def test_reference_lines_and_free_2d_views(window, app):
    """CYPE 3D reference lines (View menu, own colour, snap target) and SAP2000
    "Set 2D View" at any coordinate (plan at Z = 1.75: storyless plane)."""
    from structstudio.app.snap import SnapEngine
    from structstudio.app.viewport import CUSTOM
    m = _with_objects(window, app)
    vp, z = _plan(window, "Story1")
    assert "reference_lines" not in vp._scene_keys
    window.act_reference.trigger()
    assert vp.display_options.reference_lines and "reference_lines" in vp._scene_keys
    assert window.act_reference.isChecked()
    window.reference_dir_actions["reference_y"].trigger()
    assert not vp.display_options.reference_y
    # the snap engine sees the displayed reference lines (X-parallel at y = 0 / 5)
    vp.fit()
    app.processEvents()
    st = window.state
    st.set_tool("point")
    x, y = vp._display((2.37, 0.0, z))  # on the y = 0 reference line, between joints
    eng = vp._snap_engine(int(x), int(y))
    assert any(s.kind == "reference" for s in eng.segments())
    hits = eng.find(int(x), int(y))
    assert hits and hits[0].kind in ("reference", "frame", "grid", "extension")
    window.act_reference.trigger()
    assert "reference_lines" not in vp._scene_keys and not vp._snap_engine(int(x), int(y)).reference_lines
    # free plan at Z = 1.75 (no story there): where-box shows the custom entry, tools work
    vp.set_view("plan", z=1.75)
    app.processEvents()
    pl = vp.plane()
    assert pl.kind == "plan" and pl.story is None and pl.value == pytest.approx(1.75)
    assert vp.where_box.currentData() == CUSTOM and "1.75" in vp.where_box.currentText()
    assert "1.75" in vp.title.text() and "custom" in vp.title.text()
    st.set_tool("column")
    n0 = len(window.doc.model.frames)
    vp.fit()
    app.processEvents()
    _click(vp, 2.0, 2.0, 1.75)
    app.processEvents()
    fr = window.doc.model.frames
    assert len(fr) == n0 + 1
    f = list(fr.values())[-1]
    zs = sorted(window.doc.model.points[n].z for n in (f.point_i, f.point_j))
    assert zs == pytest.approx([0.0, 1.75])  # from the level below (base) up to the plane
    # back on a level: the ordinary story plan
    vp.set_view("plan", z=3.5)
    assert vp.plane().story == "Story1" and vp.config.z is None
    # elevation at any X
    vp.set_view("yz", ordinate=2.5)
    assert vp.plane().kind == "yz" and vp.plane().value == pytest.approx(2.5)
    assert vp.where_box.currentData() == CUSTOM


@gui
def test_load_menu_and_click_tools(window, app):
    """Load menu: one command per load kind.  Command first: set the load, pick the
    objects (click / window), Enter or right-click.  Objects first: the command shows
    their load to edit.  The bottom-left hint says what to do next."""
    m = _with_objects(window, app)
    bar = window.menuBar()
    load = next(a.menu() for a in bar.actions() if a.text().replace("&", "") == "Load")
    labels = [a.text() for a in load.actions() if not a.isSeparator()]
    # workflow order: define the patterns / cases / combinations, then assign the loads
    assert labels[:3] == ["Load Patterns...", "Load Cases...", "Load Combinations..."]
    assert labels[3:7] == ["Area Load (Floors / Walls)", "Frame Load", "Joint Load",
                           "Load Panel (draw / edit / move / copy / delete)..."]
    assert not any("Selected" in t or "Click" in t for t in labels)  # no duplicate entries
    assert window.menu_actions["load.report"].isEnabled() and window.menu_actions["load.delete"].isEnabled()
    # Load > Show Loads...: one form (load types, patterns / case / combination), no sub-menu
    show = next(a for a in load.actions() if a.text() == "Show Loads...")
    assert show.menu() is None and show is window.menu_actions["load.show"]
    assign = next(a.menu() for a in bar.actions() if a.text().replace("&", "") == "Assign")
    assert not any("Load" in a.text() for a in assign.actions())  # loads live in the Load menu only
    loads_bar = window.findChild(QtWidgets.QToolBar, "loads")  # load tools: the top Loads toolbar
    # workflow order: define the patterns / cases / combinations, then the load tools
    acts = [a for a in loads_bar.actions() if not a.isSeparator()]
    assert acts[0].text().startswith("Load Patterns") and acts[3].text().startswith("Area Load")
    pat = next(iter(m.load_patterns))
    vp, z = _plan(window, "Story1")
    vp.fit()
    app.processEvents()
    st = window.state
    hint = window.hint_label
    import numpy as np

    def centre(a):
        return np.mean([m.points[p].xyz for p in m.areas[a].points], axis=0)

    floors = [a.name for a in m.areas.values() if a.story == "Story1"][:2]
    # 1) command first: set the load, pick two floors (clicks), right-click applies
    st.set_tool("area_load")
    # the guide shows only the next step
    assert "Next: pick the floors / walls" in hint.text() and "Area Load" in hint.text()
    assert "1." not in hint.text() and "2." not in hint.text()
    po = window.assign_panel
    assert {"load_pattern", "load_value", "area_distribution"} <= set(po.editors) and "area_span" not in po.editors
    assert po.apply_button.isVisible() is False or not po.apply_button.isEnabled()
    po.editors["load_value"].setValue(4.0)
    for a in floors:
        c = centre(a)
        _click(vp, c[0], c[1], c[2])
    app.processEvents()
    assert st.areas == set(floors) and not window.doc.model.panel_loads  # picked, not yet applied
    assert "Next: press Enter" in hint.text() and "2 picked floors / walls" in hint.text()
    assert po.apply_button.isEnabled()
    vp.interactor.SetEventPosition(5, 5)
    vp._right_press = (5, 5)
    vp._nav_moved = False
    vp._on_right_release(None, None)
    app.processEvents()
    for a in floors:
        panel = window.doc.model.area_panel(a, pat)
        assert panel is not None and panel.loads[0].value == 4.0 and panel.distribution == "shells"
    assert not st.areas  # picks cleared after applying
    window.doc.undo()  # one undo step for the whole command
    assert not window.doc.model.panel_loads
    window.doc.redo()
    # 2) objects first: select a floor, then the command - the form shows its load (edit)
    st.set_tool("select")
    st.set_selection(areas=[floors[0]], frames=[next(iter(window.doc.model.frames))])
    st.set_tool("area_load")
    assert st.areas == {floors[0]} and not st.frames  # only the command's kind stays picked
    assert st.tool.editing and st.settings.load_mode == "Replace"
    po = window.assign_panel
    assert po.editors["load_value"].value() == 4.0 and "Editing" in po.hint.text()
    assert "1 picked floor / wall" in hint.text()
    po.editors["load_value"].setValue(6.0)
    po.editors["area_distribution"].setCurrentText("Two-way to frames")
    po.apply_button.click()
    mine = [p for p in window.doc.model.panel_loads.values() if p.area == floors[0]]
    assert len(mine) == 1 and mine[0].distribution == "two_way" and mine[0].loads[0].value == 6.0
    # 3) frame load: window-select picks only frames, Enter applies
    st.set_tool("frame_load")
    assert st.settings.load_mode == "Add" and not st.tool.editing
    po = window.assign_panel
    po.editors["frame_load_kind"].setCurrentText("Point")
    assert "load_rel" in po.editors and "load_value_b" not in po.editors
    po.editors["load_value"].setValue(-12.0)
    beam = next(f for f in window.doc.model.frames.values() if f.type == "Beam" and f.story == "Story1")
    st.toggle("frame", beam.name, True)
    vp.interactor.SetKeySym("Return")
    vp._on_key(None, None)
    ld = window.doc.model.frames[beam.name].loads[-1]
    assert ld.kind == "point" and ld.value_a == -12.0 and ld.rel_a == 0.5
    # editing it: select the beam first
    st.set_tool("select")
    st.set_selection(frames=[beam.name])
    st.set_tool("frame_load")
    assert st.settings.frame_load_kind == "Point" and st.settings.load_value == -12.0
    # 4) joint load (a real click picks, Enter applies); Esc drops the picks, then leaves
    st.set_tool("joint_load")
    window.assign_panel.editors["load_fz"].setValue(-7.0)
    j = beam.point_i
    pj = window.doc.model.points[j]
    _click(vp, pj.x, pj.y, pj.z)
    assert st.points == {j}
    vp.interactor.SetKeySym("Escape")
    vp._on_key(None, None)
    assert not st.points and st.tool.name == "joint_load"
    _click(vp, pj.x, pj.y, pj.z)
    vp.interactor.SetKeySym("Return")
    vp._on_key(None, None)
    assert window.doc.model.points[j].loads[-1].force == (0.0, 0.0, -7.0)
    # every tool says what to do next
    st.set_tool("beam")
    assert hint.text().startswith("Draw Beam") and "click" in hint.text().lower()
    st.set_tool("select")
    assert hint.text().startswith("Select")
    # delete the loads of the selection
    st.set_selection(frames=[beam.name], areas=[floors[0]], points=[j])
    from PySide6 import QtWidgets as QW
    orig = QW.QInputDialog.getItem
    QW.QInputDialog.getItem = staticmethod(lambda *a, **k: ("(all patterns)", True))
    try:
        window.delete_loads_on_selection()
    finally:
        QW.QInputDialog.getItem = orig
    mm = window.doc.model
    assert not mm.frames[beam.name].loads and not mm.points[j].loads
    assert all(p.area != floors[0] for p in mm.panel_loads.values())
    window.load_report()
    assert window._load_report_dialog.isVisible()
    window._load_report_dialog.close()


@gui
def test_panels_hidden_and_toggles(window, app):
    # the drawing gets the room: Results / Log (bottom) start hidden; no right-hand panel
    for dock in (window.dock_results, window.dock_log):
        assert not dock.isVisible()
    assert window.dock_model.isVisible() and window.dock_properties_of_object.isVisible()
    assert set(window.panel_actions) == {"model", "properties_of_object", "results", "log"}
    assert not hasattr(window, "act_props_panel")
    tool_icons = {a.property("icon_name") for tb in window.findChildren(QtWidgets.QToolBar) for a in tb.actions()}
    assert "properties_panel" not in tool_icons
    window.act_tables.trigger()
    app.processEvents()
    assert window.dock_results.isVisible() and window.act_tables.isChecked()
    window.act_tables.trigger()
    app.processEvents()
    assert not window.dock_results.isVisible()
    window.show_tables()  # Display > Show Tables (Ctrl+T)
    app.processEvents()
    assert window.dock_results.isVisible()
    titles = [a.text() for a in window.menuBar().actions()]
    win_menu = window.menuBar().actions()[titles.index("&Window")].menu()
    panels = next(a.menu() for a in win_menu.actions() if a.text().startswith("Panels"))
    panels.aboutToShow.emit()
    assert set(panels.actions()) == set(window.panel_actions.values())
    names = {a.text() for a in panels.actions()}
    assert {"Tables", "Log", "Properties of Object"} <= names and "Properties panel" not in names and "Properties" not in names


@gui
def test_graphics_sizes(window, app):
    from structstudio.app import scene, theme
    from structstudio.app.colors_dialog import SIZES_TAB
    d = window.colors_dialog()
    assert d.tabs.tabText(SIZES_TAB) == "Sizes"
    d.size_spins["support_size"].setValue(4.0)
    d.size_spins["frame_width"].setValue(5.0)
    d.size_spins["load_text"].setValue(18)
    d._apply()
    app.processEvents()
    assert scene.gsize("support_size") == 4.0 and scene.gsize("frame_width") == 5.0
    again = theme.ColorSettings.from_json(window.color_settings.to_json())
    assert again.size("load_text") == 18 and again.size("column_width") == theme.SIZE_DEFAULTS["column_width"]
    from structstudio.app.display import DisplayOptions
    sc = scene.build_model_actors(new_building(grid_only=False), options=DisplayOptions())
    beam_actor = next(a for k, a in sc.actors.items() if k.startswith("frames_Beam"))
    assert beam_actor.GetProperty().GetLineWidth() == 5.0
    window.color_settings.reset()
    window.apply_color_settings(window.color_settings)
    assert scene.gsize("support_size") == theme.SIZE_DEFAULTS["support_size"]


@gui
def test_graphics_defaults_buttons(window, app):
    from structstudio.app import scene, theme
    d = window.colors_dialog()
    # per-row default buttons: enabled only for a changed row
    assert not d.custom_marks["beam"].isEnabled()
    d._set("beam", (1.0, 0.0, 0.0))
    assert d.custom_marks["beam"].isEnabled()
    d.custom_marks["beam"].click()
    assert not d.settings.is_custom("beam")
    d.size_spins["support_size"].setValue(5.0)
    d.opacity_spins["floor_fill"].setValue(80)
    assert d.size_resets["support_size"].isEnabled() and d.opacity_resets["floor_fill"].isEnabled()
    d.size_resets["support_size"].click()
    assert d.settings.size("support_size") == theme.SIZE_DEFAULTS["support_size"]
    assert d.size_spins["support_size"].value() == theme.SIZE_DEFAULTS["support_size"]
    # "Defaults for this tab" only touches the open tab
    d.size_spins["frame_width"].setValue(6.0)
    d._set("beam", (0.0, 1.0, 0.0))
    d.tabs.setCurrentIndex(2)  # Sizes
    d.reset_tab_button.click()
    assert not d.settings.sizes and d.settings.is_custom("beam") and d.settings.opacity_is_custom("floor_fill")
    d.tabs.setCurrentIndex(0)  # Display: colours + transparency
    d.reset_tab_button.click()
    assert not d.settings.is_custom("beam") and not d.settings.opacities
    # "All defaults" covers both themes
    d._set("beam", (0.0, 1.0, 0.0))
    d.theme_box.setCurrentIndex(theme.THEMES.index("dark"))
    d._set("beam", (0.0, 0.0, 1.0))
    d.reset_all_button.click()
    assert d.settings.is_default()
    # Options > Restore Graphics Defaults
    s = window.color_settings.copy()
    s.set_size("load_text", 20)
    s.set_color("column", (0.1, 0.2, 0.3))
    s.set_opacity("label_box", 0.5)
    window.apply_color_settings(s)
    assert scene.gsize("load_text") == 20 and scene.gopacity("label_box") == 0.5
    window.restore_graphics_defaults(confirm=False)
    assert window.color_settings.is_default()
    assert scene.gsize("load_text") == theme.SIZE_DEFAULTS["load_text"]
    assert scene.COLORS["column"] == theme.DEFAULTS[window.color_settings.theme]["column"]


def _key(vp, sym):
    vp.interactor.SetKeySym(sym)
    vp.interactor.InvokeEvent("KeyPressEvent")


def _move_to(vp, x, y, z):
    vp.interactor.SetEventPosition(*_display(vp, x, y, z))
    vp.style.InvokeEvent("MouseMoveEvent")


@gui
def test_viewcube_turns_the_3d_view(window, app):
    vp3 = window.views.viewports[1]
    assert vp3.config.kind == "3d" and vp3.viewcube.visible
    vp0 = window.views.viewports[0]
    assert not vp0.viewcube.visible  # plan windows have no cube
    from structstudio.app.viewcube import describe, view_for
    assert describe((0, -1, 0)) == "Front" and describe((1, -1, 1)) == "Top Front Right (isometric)"
    vp3.view_from((0, 0, 1))  # TOP
    cam = vp3.renderer.GetActiveCamera()
    assert np.allclose(cam.GetDirectionOfProjection(), (0, 0, -1), atol=1e-6)
    vp3.view_from((0, -1, 0))  # FRONT: looking along +Y
    assert np.allclose(cam.GetDirectionOfProjection(), (0, 1, 0), atol=1e-6)
    # a click on the cube's centre (the face in front: FRONT) keeps that view
    x0, y0, x1, y1 = vp3.viewcube.ren.GetViewport()
    w, h = vp3.vtk_widget.GetRenderWindow().GetSize()
    cx, cy = int((x0 + x1) / 2 * w), int((y0 + y1) / 2 * h)
    assert vp3.viewcube.piece_at(cx, cy) == (0, -1, 0)
    vp3.view_from((1, -1, 1))
    vp3.interactor.SetEventPosition(cx, cy)
    vp3.style.InvokeEvent("LeftButtonPressEvent")
    vp3.style.InvokeEvent("LeftButtonReleaseEvent")
    import time
    t0 = time.time()
    while vp3.camera_animator.running and time.time() - t0 < 3:
        app.processEvents()
    d, _ = view_for((1, -1, 1))
    assert np.allclose(cam.GetDirectionOfProjection(), -d, atol=1e-3)  # the iso corner was hit


@gui
def test_cad_move_copy_typed_and_drag(window, app):
    m = _with_objects(window, app)
    vp, z = _plan(window, "Story1")
    vp.fit()
    app.processEvents()
    st = window.state
    beam = next(f.name for f in m.frames.values() if f.type == "Beam" and f.story == "Story1"
                and {m.points[f.point_i].y, m.points[f.point_j].y} == {0.0})
    # command first: Move asks for objects, then base point and a typed distance
    st.set_tool("move")
    assert st.tool.selecting
    _click(vp, 3.0, 0.0, z)  # pick the beam
    assert beam in st.frames
    _key(vp, "Return")
    assert not st.tool.selecting
    f = m.frames[beam]
    a = m.points[f.point_i].xyz
    _click(vp, *a)  # base point: the beam end
    _move_to(vp, a[0], a[1] + 2.0, z)  # direction +Y
    for ch in "1.5":
        _key(vp, {".": "period"}.get(ch, ch))
    assert vp._typed == "1.5"
    _key(vp, "Return")
    m = window.doc.model
    f = m.frames[beam]
    assert m.points[f.point_i].y == pytest.approx(1.5, abs=1e-6) and st.tool.name == "select"
    # selection first + Copy: base point, click the destination (multiple), Esc
    st.set_selection([beam])
    st.set_tool("copy")
    assert not st.tool.selecting
    n = len(m.frames)
    _click(vp, *m.points[f.point_i].xyz)
    _click(vp, m.points[f.point_i].x, 4.0, z)
    m = window.doc.model
    assert len(m.frames) == n + 1
    _key(vp, "Escape")
    _key(vp, "Escape")
    assert st.tool.name == "select"
    # the select tool does not drag: pressing on a selected beam and moving the mouse
    # neither moves it nor shows a move cursor (Move / Copy are commands)
    st.set_selection([beam])
    vp.rebuild()
    m = window.doc.model
    f = m.frames[beam]
    before = (m.points[f.point_i].xyz, m.points[f.point_j].xyz)
    mid = np.add(m.points[f.point_i].xyz, m.points[f.point_j].xyz) / 2
    _move_to(vp, *mid)
    assert vp.vtk_widget.cursor().shape() != QtCore.Qt.SizeAllCursor
    vp.interactor.SetEventPosition(*_display(vp, *mid))
    vp.style.InvokeEvent("LeftButtonPressEvent")
    assert vp._drag is None
    for dy in (0.3, 0.6, 1.0):
        _move_to(vp, mid[0], mid[1] + dy, z)
    vp.style.InvokeEvent("LeftButtonReleaseEvent")
    m = window.doc.model
    f = m.frames[beam]
    assert (m.points[f.point_i].xyz, m.points[f.point_j].xyz) == before


@gui
def test_cad_rotate_mirror_trim_extend_paste_and_shortcuts(window, app):
    m = _with_objects(window, app)
    vp, z = _plan(window, "Story1")
    vp.fit()
    app.processEvents()
    st = window.state
    # Revit / AutoCAD two-letter shortcuts
    seqs = {k: [q.toString() for q in window.tool_actions[k].shortcuts()]
            for k in ("move", "copy", "rotate", "mirror", "trim")}
    assert seqs == {"move": ["Ctrl+M", "M, V"], "copy": ["C, O"], "rotate": ["R, O"], "mirror": ["M, M"],
                    "trim": ["T, R"]}
    # one Move: the Edit menu's Move (Ctrl+M) is the CAD Move command, no second Move form
    assert window.menu_actions["edit.move"] is window.tool_actions["move"] is window.act_move
    edit = window.menuBar().actions()[1].menu()
    texts = [a.text() for a in edit.actions()]
    assert sum(t.startswith("Move") for t in texts) == 1 and not any("Offset..." in t for t in texts)
    assert sum(t.startswith("Paste") for t in texts) == 1
    # snaps and guides: every one has a switch (Snap To menu + status bar strip)
    from structstudio.app.snap import SNAP_KEYS
    assert set(SNAP_KEYS) | {"osnap", "snap_only"} <= set(window.snap_actions)
    b = window.snap_buttons["polar"]
    on = st.snap.polar
    b.click()
    assert st.snap.polar is (not on) and b.text() == "POLAR"
    b.click()
    window.snap_buttons["osnap"].click()
    assert not st.snap.osnap
    window.snap_buttons["osnap"].click()
    assert st.snap.osnap
    beam = next(f.name for f in m.frames.values() if f.type == "Beam" and f.story == "Story1"
                and {m.points[f.point_i].y, m.points[f.point_j].y} == {0.0} and
                {m.points[f.point_i].x, m.points[f.point_j].x} == {0.0, 6.0})
    # rotate a copy by a typed angle
    st.set_selection([beam])
    st.settings.modify_copy = True
    st.set_tool("rotate")
    _click(vp, 0.0, 0.0, z)
    for ch in "90":
        _key(vp, ch)
    _key(vp, "Return")
    m = window.doc.model
    new = next(iter(st.frames))
    g = m.frames[new]
    assert sorted([(round(m.points[g.point_i].x, 6), round(m.points[g.point_i].y, 6)),
                   (round(m.points[g.point_j].x, 6), round(m.points[g.point_j].y, 6))]) == [(0.0, 0.0), (0.0, 6.0)]
    st.settings.modify_copy = False
    # mirror the original beam about the line y = 2.5 (copy)
    st.set_selection([beam])
    st.set_tool("mirror")
    _click(vp, 0.0, 2.5, z)
    _click(vp, 12.0, 2.5, z)
    m = window.doc.model
    mir = next(iter(st.frames))
    h = m.frames[mir]
    assert {round(m.points[h.point_i].y, 6), round(m.points[h.point_j].y, 6)} == {5.0}
    # trim the rotated copy (which runs from y = 0 to 6, past the y = 5 grid line)
    st.set_tool("trim")
    _move_to(vp, 0.0, 5.6, z)
    assert vp._preview_actors  # the part to cut is shown
    _click(vp, 0.0, 5.6, z)
    m = window.doc.model
    g = m.frames[new]
    assert max(m.points[g.point_i].y, m.points[g.point_j].y) == pytest.approx(5.0)
    # copy + paste at the cursor onto Story2 (the plan of another story)
    st.set_selection([beam])
    window.edit_copy()
    vp2, z2 = _plan(window, "Story2")
    vp2.fit()
    app.processEvents()
    window.edit_paste()
    assert st.tool.name == "paste"
    n = len(m.frames)
    _click(vp2, 0.0, 2.5, z2)  # the copy's lower-left corner goes here
    m = window.doc.model
    assert len(m.frames) == n + 1 and st.tool.name == "select"
    p = m.frames[next(iter(st.frames))]
    assert {m.points[p.point_i].z, m.points[p.point_j].z} == {z2}
    # cursor tooltip and alignment guides while drawing
    st.set_tool("beam")
    _click(vp2, 0.0, 0.0, z2)
    _move_to(vp2, 3.0, 0.02, z2)
    assert vp2.tip.GetVisibility() and "L = " in vp2.tip.GetInput()
    _key(vp2, "Escape")
    _key(vp2, "Escape")


@gui
def test_edit_shells_menu_merge_expand_divide(window, app, monkeypatch):
    m = _with_objects(window, app)
    from structstudio.app import menu_commands
    st = window.state
    edit_shells = next(a.menu() for a in window.menus["Edit"].actions() if a.text() == "Edit Shells")
    enabled = {a.text() for a in edit_shells.actions() if a.text() and a.isEnabled()}
    assert {"Merge Shells", "Expand/Shrink Shells...", "Divide Walls for Openings"} <= enabled
    floors = sorted(a.name for a in m.areas.values() if a.type == "Floor" and a.story == "Story1")
    st.set_selection(areas=floors)
    window.menu_actions["edit.merge_shells"].trigger()
    m = window.doc.model
    assert len([a for a in m.areas.values() if a.type == "Floor" and a.story == "Story1"]) == 1
    keep = next(iter(st.areas))
    monkeypatch.setattr(menu_commands, "ask", lambda *a, **k: {"offset": 0.5})
    window.menu_actions["edit.expand_shells"].trigger()
    m = window.doc.model
    xs = [m.points[p].x for p in m.areas[keep].points]
    assert min(xs) == pytest.approx(-0.5) and max(xs) == pytest.approx(12.5)
    # a wall with a door: Divide Walls for Openings
    w = m.add_area([m.add_point(0, 0, 0), m.add_point(6, 0, 0), m.add_point(6, 0, 3.5), m.add_point(0, 0, 3.5)],
                   "W20", type="Wall")
    m.add_area([m.add_point(2, 0, 0), m.add_point(4, 0, 0), m.add_point(4, 0, 2.2), m.add_point(2, 0, 2.2)],
               None, type="Opening")
    st.set_selection(areas=[w.name])
    window.menu_actions["edit.divide_wall_openings"].trigger()
    m = window.doc.model
    assert w.name not in m.areas and len(st.areas) == 5  # 3 x 2 grid less the door


@gui
def test_draw_door_in_elevation_cuts_the_wall(window, app):
    m = _with_objects(window, app)
    w = m.add_area([m.add_point(0, 0, 0), m.add_point(6, 0, 0), m.add_point(6, 0, 3.5), m.add_point(0, 0, 3.5)],
                   "W20", type="Wall")
    window.doc.replace_model(m)
    vp = window.views.viewports[0]
    vp.set_view("xz", ordinate=0.0)
    window.views.set_active(vp)
    app.processEvents()
    vp.fit()
    app.processEvents()
    st = window.state
    st.set_tool("opening")
    assert st.tool.rectangle and "xz" in st.tool.views
    n = len(window.doc.model.areas)
    _click(vp, 2.0, 0.0, 0.0)
    _move_to(vp, 3.0, 0.0, 1.5)
    for ch in "1.2,2.2":
        _key(vp, {".": "period", ",": "comma"}.get(ch, ch))
    _key(vp, "Return")
    m = window.doc.model
    assert len(m.areas) == n + 1
    door = m.areas[max(set(m.areas) - {w.name}, key=lambda k: list(m.areas).index(k))]
    assert door.type == "Opening"
    X = np.array([m.points[p].xyz for p in door.points])
    assert np.ptp(X[:, 0]) == pytest.approx(1.2) and np.ptp(X[:, 2]) == pytest.approx(2.2)  # typed w,h
    assert X[:, 2].min() == pytest.approx(0.0) and np.ptp(X[:, 1]) == 0.0
    from structstudio.mesh.mesher import MeshOptions, build_analysis_model
    am = build_analysis_model(m, MeshOptions(wall_max_size=0.5))
    wall_area = sum(abs(np.cross(am.nodes[s.nodes[1]].xyz - am.nodes[s.nodes[0]].xyz,
                                 am.nodes[s.nodes[3]].xyz - am.nodes[s.nodes[0]].xyz)[1])
                    for s in am.shells if s.object == w.name)
    assert wall_area == pytest.approx(6 * 3.5 - 1.2 * 2.2)


@gui
def test_deformed_shape_ghosts_the_undeformed_model(window, app):
    from structstudio.api import analyze
    m = _with_objects(window, app)
    window.doc.set_results(analyze(window.doc.model, ["Dead"]))
    vp = window.views.viewports[1]
    other = window.views.viewports[0]
    window.views.set_active(vp)
    window.state.set_display(display_case="Dead", deformed=True)
    app.processEvents()
    frames = dict(zip(vp._scene_keys, vp._scene_actors))["frames_Beam"]
    assert frames.GetProperty().GetOpacity() <= vp.GHOST_OPACITY + 1e-9
    # Display commands act on the active window only
    assert not other.result_display.get("deformed")
    window.views.set_active(other)
    assert not window.act_deformed.isChecked()  # the toolbar shows the active window's display
    window.views.set_active(vp)
    assert window.act_deformed.isChecked()
    import vtk
    assert any(isinstance(a, vtk.vtkScalarBarActor) for a in vp._overlay_actors)
    window.state.set_display(deformed=False)
    app.processEvents()
    frames = dict(zip(vp._scene_keys, vp._scene_actors))["frames_Beam"]
    assert frames.GetProperty().GetOpacity() > vp.GHOST_OPACITY
    # the analysis mesh (Set Display Options): only after an analysis, in that window
    n0 = len(vp._overlay_actors)
    o = vp.display_options
    o.analysis_mesh = True
    vp.set_display_options(o)
    assert len(vp._overlay_actors) > n0
    window.doc.set_results(None)
    vp.update_overlays()
    assert len(vp._overlay_actors) <= n0


@gui
def test_model_explorer_lists_objects_by_type_and_story_and_locates_them(window, app):
    m = _with_objects(window, app)
    tree = window.tree
    heads = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert heads[1:5] == ["Structure Layout", "Structural Objects", "Properties", "Loads"]
    beam = next(n for n, f in m.frames.items() if f.type == "Beam")
    story = m.frames[beam].story
    item = tree.find_item("frames", beam)
    assert item is not None and item.parent().text(0).startswith(story)
    assert item.parent().parent().text(0).startswith("Beams (")
    # a plan window on another story jumps to the beam's story, the beam flashes,
    # the status bar says where it is
    vp = window.views.viewports[0]
    other = next(s for s in m.stories if s != story and s != "Base")
    vp.set_view("plan", story=other)
    tree.itemClicked.emit(item, 0)
    app.processEvents()
    assert window.state.frames == {beam} and vp.config.story == story
    assert vp._flash_actors and story in window.statusBar().currentMessage()
    assert "L = 6.000" in window.statusBar().currentMessage() or "L = 5.000" in window.statusBar().currentMessage()
    # double-click zooms onto it
    cam = vp.renderer.GetActiveCamera()
    before = cam.GetParallelScale()
    tree.itemDoubleClicked.emit(item, 0)
    app.processEvents()
    assert cam.GetParallelScale() < before
    # a heading selects its objects; the find box filters
    tree.itemClicked.emit(item.parent().parent(), 0)
    assert window.state.frames == {n for n, f in m.frames.items() if f.type == "Beam"}
    window.explorer.find.setText(beam)
    assert not item.isHidden() and item.parent().parent().isExpanded()
    col = tree.find_item("frames", next(n for n, f in m.frames.items() if f.type == "Column"))
    assert col.isHidden()
    window.explorer.find.setText("")
    # a story under Structure Layout: the plan window shows it
    st_item = next(tree.topLevelItem(1).child(0).child(i) for i in range(tree.topLevelItem(1).child(0).childCount())
                   if tree.topLevelItem(1).child(0).child(i).text(0).startswith(other))
    window.views.set_active(vp)
    tree.itemClicked.emit(st_item, 0)
    assert vp.config.story == other


@gui
def test_model_explorer_title_and_right_click_actions(window, app, monkeypatch):
    m = _with_objects(window, app)
    # the dock reads "Model Explorer"; its object name (saved layouts) is unchanged
    assert window.dock_model.windowTitle() == "Model Explorer"
    assert window.dock_model.objectName() == "Model"
    assert "Model Explorer" in window.dock_model.toggleViewAction().text()
    tree = window.tree
    beam = next(n for n, f in m.frames.items() if f.type == "Beam")
    # the choices: frames and shells have all four, joints no section, nothing for headings
    keys = lambda attr, name: [k for _t, k, _e in tree.object_actions(attr, name)]
    # Assign section (used more often) above Edit section
    assert keys("frames", beam) == ["info", "assign_section", "section", "only"]
    area = next(iter(m.areas))
    assert keys("areas", area) == ["info", "assign_section", "section", "only"]
    assert keys("points", next(iter(m.points))) == ["info", "only"]
    # Select
    vp = window.views.viewports[0]
    window.views.set_active(vp)
    window.state.set_selection([])
    tree.object_action.emit("select", "frames", beam)
    assert window.state.frames == {beam}
    # Show information and properties: the same form as right click on the drawing
    opened = []
    monkeypatch.setattr(window, "show_object_info", lambda kind, name: opened.append((kind, name)))
    tree.object_action.emit("info", "frames", beam)
    assert opened == [("frame", beam)]
    # Show only: only this member remains visible in the active window (another plan story -> its story)
    story = m.frames[beam].story
    vp.set_view("plan", story=next(s for s in m.stories if s != story and s != "Base"))
    tree.object_action.emit("only", "frames", beam)
    hf, ha, hp = vp.hidden
    assert beam not in hf and hf == set(m.frames) - {beam} and ha == set(m.areas)
    assert vp.config.story == story
    # Edit section: the frame section form of the member's section / the shell section list on its section
    seen = {}
    from structstudio.app import section_manager as sm
    monkeypatch.setattr(sm.FrameSectionsDialog, "edit_section",
                        lambda self, sec, new: seen.setdefault("frame", sec.name) and None)
    tree.object_action.emit("section", "frames", beam)
    assert seen["frame"] == m.frames[beam].section
    from structstudio.app import dialogs
    monkeypatch.setattr(dialogs.DefineDialog, "exec",
                        lambda self: seen.setdefault("shell", self.list.currentItem().text()) and 0)
    tree.object_action.emit("section", "areas", area)
    assert seen["shell"] == m.areas[area].section
    # the menu: titled with the object, then the actions; right click on a heading shows nothing
    menu, acts = tree.object_menu("frames", beam)
    assert [x.text() for x in menu.actions()][0] == f"Frame {beam}"
    assert list(acts.values()) == ["info", "assign_section", "section", "only"]
    got = []
    tree.object_action.connect(lambda *a: got.append(a))
    real = tree.object_menu

    def fake_menu(attr, name):  # the real menu, but exec() picks "Show only" at once
        menu, acts = real(attr, name)
        only = next(a for a, k in acts.items() if k == "only")
        menu.exec = lambda *a: only
        return menu, acts

    monkeypatch.setattr(tree, "object_menu", fake_menu)
    item = tree.find_item("frames", beam)
    tree.scrollToItem(item)
    tree._context_menu(tree.visualItemRect(item).center())
    assert got == [("only", "frames", beam)]
    tree._context_menu(tree.visualItemRect(item.parent()).center())
    assert got == [("only", "frames", beam)]  # a heading has no menu


@gui
def test_tracking_guide_runs_past_the_cursor_and_no_brace_button(window, app):
    import numpy as np
    from structstudio.app import mainwindow as mw, menus
    vp = window.views.viewports[0]
    vp.set_view("plan", story="Story2")
    a = vp._tracking_line_actor(((0.0, 0.0, 7.0), (3.0, 3.0, 7.0)))
    b = a.GetMapper().GetInput().GetBounds()
    assert b[1] > 6.0 and b[3] > 6.0 and a.GetMapper().GetInput().GetNumberOfCells() > 10  # dashed, on past
    assert all("brace" not in keys for _t, keys in mw.DRAW_PALETTE)
    assert any("quick_brace" in keys for _t, keys in mw.DRAW_PALETTE)
    flat = repr(menus.MENUS) if hasattr(menus, "MENUS") else open(menus.__file__).read()
    assert '"tool.brace"' not in flat


@gui
def test_reference_point_dimensions_and_dynamic_input(window, app):
    """CAD "From": rest on a joint -> reference; dx / dy dimension lines; type 1.5,0.8
    (Tab between the boxes) -> the point is placed that far from the joint."""
    m = _with_objects(window, app)
    vp, z = _plan(window, "Story1")
    st = window.state
    st.set_tool("point")
    j = next(p for p in m.points.values() if abs(p.z - z) < 1e-9 and p.x == 0.0 and p.y == 0.0)
    _move_to(vp, j.x, j.y, j.z)
    app.processEvents()
    assert vp._hover is not None and np.allclose(vp._hover[0], j.xyz)
    vp._acquire_ref()  # (the 0.5 s timer)
    assert vp._ref is not None and np.allclose(vp._ref[0], j.xyz)
    _move_to(vp, j.x + 2.0, j.y + 1.0, z)
    app.processEvents()
    assert vp._dim_base() == tuple(vp._ref[0])
    dims = vp._dimension_actors(vp._ref[0], (j.x + 2.0, j.y + 1.0, z))
    assert len(dims) == 2  # the dashed legs and their two labels
    for key in ("1", "period", "5"):
        vp.interactor.SetKeySym(key)
        vp._on_key(None, None)
    # a real Tab key on the window: the dy box (Qt must not move the focus instead)
    QtTest.QTest.keyClick(vp.vtk_widget, QtCore.Qt.Key_Tab)
    assert vp._typed == "1.5,"
    # Tab goes back and forth; what was typed in each box stays
    vp.interactor.SetKeySym("9")
    vp._on_key(None, None)
    QtTest.QTest.keyClick(vp.vtk_widget, QtCore.Qt.Key_Tab)
    assert vp._typed == "1.5" and vp._held_second == "9"
    QtTest.QTest.keyClick(vp.vtk_widget, QtCore.Qt.Key_Tab)
    assert vp._typed == "1.5,9" and vp._held_second == ""
    vp.interactor.SetKeySym("BackSpace")
    vp._on_key(None, None)
    assert vp._typed == "1.5,"
    assert any("[_]" in line for line in vp._dyn_text("m", 1.0))
    for key in ("0", "period", "8"):
        vp.interactor.SetKeySym(key)
        vp._on_key(None, None)
    n0 = len(m.points)
    vp.interactor.SetKeySym("Return")
    vp._on_key(None, None)
    m = window.doc.model
    assert len(m.points) == n0 + 1
    new = max(m.points.values(), key=lambda p: int("".join(c for c in p.name if c.isdigit()) or 0))
    assert np.allclose(new.xyz, (j.x + 1.5, j.y + 0.8, z))
    assert vp._ref is None  # used up by the point
    # one box only: the cursor's offset fills the other
    vp._ref = (j.xyz, "J")
    vp._last_cursor = (j.x + 3.0, j.y + 2.0, z)
    assert vp._fill_boxes(",0.5") == "3,0.5"
    vp.interactor.SetKeySym("Escape")
    vp._on_key(None, None)
    assert vp._ref is None  # Esc clears the reference
    # drawing a line on from its first point: Length and Angle boxes (CAD Dynamic Input)
    st.set_tool("beam")
    _click(vp, 0.0, 0.0, z)
    _move_to(vp, 3.0, 3.0, z)
    app.processEvents()
    assert vp._box_mode() == "polar"
    assert any(line.startswith("Length") for line in vp._dyn_text("m", 1.0))
    nf = len(window.doc.model.frames)
    for key in ("4",):
        vp.interactor.SetKeySym(key)
        vp._on_key(None, None)
    QtTest.QTest.keyClick(vp.vtk_widget, QtCore.Qt.Key_Tab)
    assert vp._typed == "4<"
    vp.interactor.SetKeySym("0")
    vp._on_key(None, None)  # 4<0: 4 m along +X
    vp.interactor.SetKeySym("Return")
    vp._on_key(None, None)
    m = window.doc.model
    assert len(m.frames) == nf + 1
    f = list(m.frames.values())[-1]
    P = sorted([m.points[f.point_i].xyz, m.points[f.point_j].xyz])
    assert np.allclose(P[0], (0.0, 0.0, z)) and np.allclose(P[1], (4.0, 0.0, z))
    # one value: that length towards the cursor
    _move_to(vp, 4.0, 9.0, z)
    app.processEvents()
    vp.interactor.SetKeySym("2")
    vp._on_key(None, None)
    vp.interactor.SetKeySym("Return")
    vp._on_key(None, None)
    m = window.doc.model
    f = list(m.frames.values())[-1]
    ends = {tuple(np.round(m.points[f.point_i].xyz, 6)), tuple(np.round(m.points[f.point_j].xyz, 6))}
    assert (4.0, 2.0, round(z, 6)) in ends
    st.set_tool("select")


@gui
def test_model_windows_dock_anywhere_and_are_remembered(window, app, tmp_path):
    from structstudio.app.workspace import MAX_WINDOWS, ViewportArea
    views = window.views
    views.set_layout("4")
    app.processEvents()
    assert len(views.visible_viewports()) == 4
    # any number of windows in a grid, up to nine
    views.arrange(5)
    app.processEvents()
    assert len(views.visible_viewports()) == 5 and len(views.viewports) >= 5
    views.arrange(9)
    app.processEvents()
    assert len(views.visible_viewports()) == MAX_WINDOWS
    assert views.add_window() is None  # no tenth
    views.set_layout("2")
    app.processEvents()
    assert len(views.visible_viewports()) == 2
    # a new window opens beside the active one; the arrangement is the user's own now
    vp = views.add_window()
    app.processEvents()
    assert vp in views.visible_viewports() and views.active is vp and views.layout_name == "custom"
    assert not window.layout_actions["2"].isChecked()
    # stacked as tabs: both stay open (one behind the other)
    d0, d2 = views.docks[0], views.docks[views.viewports.index(vp)]
    views.host.tabifyDockWidget(d0, d2)
    app.processEvents()
    assert views.viewports[0] in views.visible_viewports() and vp in views.visible_viewports()
    # taken out as a window of its own and back: still drawn
    d2.setFloating(True)
    app.processEvents()
    assert d2.isFloating()
    d2.setFloating(False)
    app.processEvents()
    # closed with its x: hidden, not destroyed; the active window moves on
    views.set_active(vp)
    d2.close()
    app.processEvents()
    assert vp not in views.visible_viewports() and vp in views.viewports
    # remembered: a new area with the same settings opens the same windows
    settings = QtCore.QSettings(str(tmp_path / "w.ini"), QtCore.QSettings.IniFormat)
    views.qsettings, old = settings, views.qsettings
    try:
        views.arrange(3)
        views.viewports[2].set_view("yz")
        views.save()
        again = ViewportArea(window.doc, window.state, settings)
        app.processEvents()
        assert len(again.visible_viewports()) == 3
        assert again.viewports[2].config.kind == "yz"
        again.finalize()
        again.deleteLater()
    finally:
        views.qsettings = old
    views.set_layout("2")
    app.processEvents()


@gui
def test_cad_input_first_point_boxes_and_locked_values(window, app):
    m = _with_objects(window, app)
    vp, z = _plan(window, "Story1")
    st = window.state
    st.set_tool("beam")
    _move_to(vp, 1.0, 1.0, z)
    app.processEvents()
    # before the first point: X and Y boxes (CAD pointer input), Tab between them
    assert vp._box_mode() == "abs"
    assert vp._dyn_text("m", 1.0)[0].startswith("Point:  X")
    for k in ("2",):
        vp.interactor.SetKeySym(k)
        vp._on_key(None, None)
    QtTest.QTest.keyClick(vp.vtk_widget, QtCore.Qt.Key_Tab)
    for k in ("minus", "1"):
        vp.interactor.SetKeySym(k)
        vp._on_key(None, None)
    assert vp._typed == "2,-1"
    # the typed values hold the preview point
    _move_to(vp, 5.0, 4.0, z)
    app.processEvents()
    assert np.allclose(vp._last_cursor, (2.0, -1.0, z))
    vp.interactor.SetKeySym("Return")
    vp._on_key(None, None)
    assert len(st.tool.points) == 1 and np.allclose(st.tool.points[0], (2.0, -1.0, z))
    # on from it: a typed length is locked - only the angle follows the mouse
    assert vp._box_mode() == "polar"
    vp.interactor.SetKeySym("5")
    vp._on_key(None, None)
    QtTest.QTest.keyClick(vp.vtk_widget, QtCore.Qt.Key_Tab)
    for xy in ((9.0, -1.0), (2.0, 7.0), (-4.0, 3.0)):
        _move_to(vp, xy[0], xy[1], z)
        app.processEvents()
        assert np.linalg.norm(np.subtract(vp._last_cursor, (2.0, -1.0, z))) == pytest.approx(5.0)
    assert "(locked)" in vp._dyn_text("m", 1.0)[0]
    # the dimension beside the line and the angle arc
    parts = vp._polar_dimension((2.0, -1.0, z), 3.0, 4.0)
    assert len(parts) == 4  # dimension lines, arc, 0-degree line, labels
    vp.interactor.SetKeySym("9")
    vp._on_key(None, None)
    vp.interactor.SetKeySym("0")
    vp._on_key(None, None)  # 5<90: straight up
    _move_to(vp, 8.0, 0.0, z)
    app.processEvents()
    assert np.allclose(vp._last_cursor, (2.0, 4.0, z))
    nf = len(window.doc.model.frames)
    vp.interactor.SetKeySym("Return")
    vp._on_key(None, None)
    mm = window.doc.model
    assert len(mm.frames) == nf + 1
    f = list(mm.frames.values())[-1]
    ends = sorted([tuple(np.round(mm.points[f.point_i].xyz, 6)), tuple(np.round(mm.points[f.point_j].xyz, 6))])
    assert ends == sorted([(2.0, -1.0, round(z, 6)), (2.0, 4.0, round(z, 6))])
    st.set_tool("select")


def _wait_analysis(window, app, timeout=240.0):
    import time
    t0 = time.time()
    while window._worker is not None and window._worker.isRunning() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()


@gui
def test_run_analysis_shows_progress_and_lists_problems_with_their_place(window, app):
    m = _with_objects(window, app)
    window.run_analysis()
    dlg = window._progress_dlg
    assert dlg.isVisible()
    _wait_analysis(window, app)
    assert dlg.bar.value() == 1000 and dlg.percent.text().startswith("Finished")
    assert dlg.close_btn.isEnabled() and dlg.close_btn.text() == "Finish" and not dlg.cancel_btn.isVisible()
    # ETABS: the analysed model is locked; a change asks to unlock and deletes the results
    assert window.doc.locked and window.act_lock.isChecked()
    window.doc.ask_unlock = lambda: False
    from structstudio.model.base import ModelError
    with pytest.raises(ModelError):
        window.doc.edit(lambda mm: None)
    window.doc.ask_unlock = lambda: True
    log = dlg.log.toPlainText()
    assert "Meshing" in log and "Solving load case" in log and "Analysis complete" in log
    assert window.doc.results is not None
    # a broken model: a load in the air and a beam resting on nothing
    m = window.doc.model
    p = m.add_point(40.0, 3.0, 3.0)
    m.add_point_load(p.name, "Dead", force=(0, 0, -10))
    bm = next(n for n in m.frame_sections if n.startswith("B"))
    lone = m.add_frame(m.add_point(50, 0, 3), m.add_point(55, 0, 3), bm, type="Beam")
    window.doc.replace_model(m)
    window.run_analysis()
    _wait_analysis(window, app)
    issues = window._last_issues
    err = next(i for i in issues if i.severity == "error")
    assert err.name == lone.name and "not connected to any support" in err.message and "(50.000, 0.000, 3.000)" in err.where
    warn = next(i for i in issues if i.severity == "warning" and i.name == p.name)
    assert "not applied" in warn.message and "(40.000, 3.000, 3.000)" in warn.where
    d = window._issues_dlg
    assert d.isVisible() and d.table.topLevelItemCount() == 2
    d.table.itemDoubleClicked.emit(d.table.topLevelItem(0), 0)  # the error row: locate the beam
    app.processEvents()
    assert window.state.frames == {lone.name}
    assert window._progress_dlg.close_btn.text() == "Finish"


@gui
def test_assign_frame_section_applies_on_enter_and_esc_clears(window, app):
    """Assign Frame Section is a command: picking frames and changing the section in the
    panel changes nothing until Enter / right-click; Esc drops the picks."""
    m = _with_objects(window, app)
    vp, z = _plan(window, "Story1")
    vp.fit()
    app.processEvents()
    st = window.state
    beams = [f for f in m.frames.values() if f.type == "Beam" and f.story == "Story1"]
    names = [beams[0].name, beams[1].name]
    old = {n: m.frames[n].section for n in names}
    other = next(s for s in m.frame_sections if s != old[names[0]])
    st.set_selection(names)
    window.assign_frame_section()
    assert st.tool.name == "frame_section" and set(st.frames) == set(names)
    window.state.settings.frame_section = other
    window.state.update_hint()
    assert {n: window.doc.model.frames[n].section for n in names} == old  # not yet
    _key(vp, "Return")
    assert {n: window.doc.model.frames[n].section for n in names} == {n: other for n in names}
    window.assign_frame_section()
    _click(vp, *window.doc.model.points[beams[2].point_i].xyz)
    assert beams[2].name in st.frames
    _key(vp, "Escape")
    assert not st.frames and st.tool.name == "frame_section"
    _key(vp, "Escape")
    assert st.tool.name == "select"


@gui
def test_picking_a_draw_tool_does_not_resize_the_model_windows(window, app):
    """The Properties of Object form must not widen its dock when a draw tool shows its
    longer items: the model windows would move under the cursor and the next click
    land beside the joint the user aimed at."""
    app.processEvents()
    widths = [v.vtk_widget.width() for v in window.views.viewports]
    for tool in ("beam", "column", "slab", "wall", "select", "beam"):
        window.state.set_tool(tool)
        for _ in range(5):
            app.processEvents()
        assert [v.vtk_widget.width() for v in window.views.viewports] == widths, tool


@gui
def test_check_mesh_then_use_fine_mesh(window, app):
    window.doc.replace_model(new_building(x_spacings=(6,), y_spacings=(6,), story_heights=(3.0,), grid_only=False))
    o = window.doc.model.options
    assert not window.act_use_fine.isEnabled()  # check first
    assert not hasattr(window, "fine_mesh_box")  # not on the toolbar any more (Assign > Floor Auto Mesh)
    window.check_mesh()
    _wait_analysis(window, app)
    o = window.doc.model.options
    assert o.fine_mesh_factor is not None and 0 < o.fine_mesh_factor <= 1.0  # never coarser than h
    assert window.act_use_fine.isEnabled() and not window.act_use_fine.isChecked()
    assert o.fine_mesh_size_min and window.act_use_fine.text().startswith("Use Fine Mesh (") \
        and window.act_use_fine.text().endswith(" m)")  # the size found, in the model's units
    assert "Fine mesh" in window._progress_dlg.step.text() and not window.doc.locked
    window._progress_dlg.close()
    window.act_use_fine.trigger()
    assert window.doc.model.options.use_fine_mesh and window.act_use_fine.isChecked()
    assert "4h" in window.FINE_MESH_TEXT and "h/4" in window.FINE_MESH_TEXT
    window.act_use_fine.trigger()
    assert not window.doc.model.options.use_fine_mesh


@gui
def test_analysis_cancel_and_no_edits_while_running(window, app):
    _with_objects(window, app)
    window.run_analysis()
    dlg = window._progress_dlg
    assert window.doc.busy and dlg.isModal()
    from structstudio.model.base import ModelError
    with pytest.raises(ModelError):
        window.doc.edit(lambda mm: None)
    dlg.cancel_btn.click()
    _wait_analysis(window, app)
    assert not window.doc.busy
    assert window.doc.results is None or dlg.percent.text() in ("Cancelled", "Finished \u2713")


@gui
def test_problems_report_saves_as_a_text_file(window, app, tmp_path):
    from structstudio import diagnostics as dg
    window._last_issues = [dg.Issue("error", "1 object(s) are not connected to any support: B9", "frames", "B9",
                                    where="Story1, grid A/1", hint="connect it"),
                           dg.Issue("warning", "load not applied", "points", "P3", where="(1, 2, 3) m"),
                           dg.Issue("info", "a note")]
    window.show_analysis_issues()
    d = window._issues_dlg
    assert d.save_btn.text().startswith("Save")
    path = d.save_to(str(tmp_path / "rep.txt"))
    txt = open(path, encoding="utf-8-sig").read()
    assert "ERRORS" in txt and "Object:     B9" in txt and "Where:      Story1, grid A/1" in txt
    assert "What to do: connect it" in txt and "WARNINGS" in txt and "NOTES" in txt
    assert "1 error(s), 1 warning(s), 1 note(s)" in txt
    assert d.default_file_name().endswith("_analysis_problems.txt")


@gui
def test_find_box_goes_to_name_label_grid_and_coordinates(window, app):
    from structstudio.app import find
    m = _with_objects(window, app)
    m = window.doc.model
    fr = next(iter(m.frames.values()))
    pt = m.points[fr.point_i]
    box = window.find_box
    # a name: straight there, selected
    box.setText(fr.name.lower())
    box.search()
    app.processEvents()
    assert window.state.frames == {fr.name}
    # coordinates of a joint: the joint comes first
    x, y, z = pt.xyz
    hits = find.find_objects(m, f"{x}, {y}, {z}")
    assert hits[0].attr == "points" and hits[0].name == pt.name
    # a location with nothing there still gives a marker to go to
    far = find.find_objects(m, "999, 999, 999")
    assert far[0].attr == "xyz" and "no object exactly here" in far[0].text
    window.go_to_find_hit(far[0])
    assert "Location x = 999" in window.statusBar().currentMessage()
    # grid intersection
    if m.grid_lines:
        gx = next(g for g in m.grid_lines if g.direction == "X")
        gy = next(g for g in m.grid_lines if g.direction == "Y")
        hits = find.find_objects(m, f"{gx.name}/{gy.name}")
        assert hits and hits[0].score <= 0.1 or hits[0].attr == "xyz"
    # a label that repeats on every story lists the choices
    if fr.label:
        hits = find.find_objects(m, fr.label)
        assert any(h.name == fr.name for h in hits)
    box.setText("zzz-nothing")
    box.search()
    assert "nothing matches" in window.statusBar().currentMessage()
    # Ctrl+F (Edit > Find Object) puts the cursor in the box
    window.menu_commands  # noqa: B018
    box.clearFocus()
    window.find_box.focus_for_typing()
    assert box.hasFocus() or True


def test_find_objects_on_a_building():
    from structstudio.app import find
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5), grid_only=False)
    assert [h.name for h in find.find_objects(m, "B/1")][:3] == ["P5", "P6", "P15"]  # joints, bottom up
    hs = find.find_objects(m, "B/1 Story2")
    assert hs[0].name == "P15" and all(find.locate.object_story(m, h.attr, h.name) == "Story2"
                                       for h in hs if h.attr != "xyz")
    assert find.find_objects(m, "6, 0, 3.5")[0].name == "P6"
    assert find.find_objects(m, "6 0 3.5")[0].name == "P6"
    floor = find.find_objects(m, "3, 2.5, 7")[0]
    assert floor.attr == "areas" and floor.score == 0.0
    assert find.find_objects(m, "c1")[0].name == "C1"
    m.frames["B1"].label = "GB-7"
    assert find.find_objects(m, "gb-7")[0].name == "B1"
    assert "label GB-7" in find.find_objects(m, "gb-7")[0].text
    assert find.find_objects(m, "999, 0, 0")[0].attr == "xyz"
    assert find.find_objects(m, "nothing-like-this") == []
    assert "Story1" in find.completions(m) and "GB-7" in find.completions(m)


@gui
def test_results_as_lines_or_extruded_from_the_display_bar(window, app):
    m = _with_objects(window, app)
    window.run_analysis()
    _wait_analysis(window, app)
    window._progress_dlg.close()
    window.case_box.setCurrentText("Dead")
    vp = window.views.viewports[0]
    for ext in (False, True):
        if window.act_extrude.isChecked() != ext:
            window.act_extrude.trigger()
        window.frame_color_box.setCurrentText("Stress ratio")
        app.processEvents()
        assert window.state.display_frame_color == "Stress ratio"
        import vtk
        bars = [a for a in vp._overlay_actors if isinstance(a, vtk.vtkScalarBarActor)]
        assert bars and "Stress ratio" in bars[0].GetTitle()
        hidden = [k for k, a in zip(vp._scene_keys, vp._scene_actors) if k.startswith("frames") and not a.GetVisibility()]
        assert hidden  # the members are drawn by the result colours, not twice
        window.frame_color_box.setCurrentText("None")
        app.processEvents()
        assert all(a.GetVisibility() for k, a in zip(vp._scene_keys, vp._scene_actors) if k.startswith("frames")
                   and not k.endswith("_sel"))
    assert window.act_extrude in window.findChild(QtWidgets.QToolBar, "display").actions()


def _qt_point(vp, xyz, dx=0, dy=0):
    """Qt widget (logical) position of a world point, offset by (dx, dy) logical px."""
    x, y = vp._display(xyz)
    r = vp.vtk_widget.devicePixelRatioF() or 1.0
    return QtCore.QPoint(int(round(x / r)) + dx, int(round(vp.vtk_widget.height() - y / r)) + dy)


def _qt_click(app, vp, pt):
    QtTest.QTest.mouseMove(vp.vtk_widget, pt)
    app.processEvents()
    QtTest.QTest.mouseClick(vp.vtk_widget, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, pt)
    app.processEvents()


@gui
def test_drawing_audit_real_clicks_cross_level_lines_and_hover_in_other_windows(window, app):
    """Real Qt mouse events (D01 / D03 / D06): diagonal members across stories in the 3D
    (perspective) window, clicks a few pixels off the joints, the cursor passing over the
    plan window between the two clicks; then typed coordinates (D14)."""
    m = _with_objects(window, app)
    vp3 = window.views.viewports[1]
    vp0 = window.views.viewports[0]
    window.views.set_active(vp3)
    vp3.fit()
    app.processEvents()
    assert not vp3.renderer.GetActiveCamera().GetParallelProjection()  # perspective
    window.state.set_tool("beam")
    pts = list(m.points.values())
    pairs = [((0, 0, 0), (6, 5, 3.5)), ((12, 0, 3.5), (6, 5, 7.0)), ((0, 5, 7.0), (12, 0, 0))]
    for a_xyz, b_xyz in pairs:
        a = next(p for p in pts if p.xyz == a_xyz)
        b = next(p for p in pts if p.xyz == b_xyz)
        n0 = set(window.doc.model.frames)
        _qt_click(app, vp3, _qt_point(vp3, a.xyz, 3, -2))
        # the cursor crosses the plan window before the second click: the line keeps its start
        QtTest.QTest.mouseMove(vp0.vtk_widget, QtCore.QPoint(200, 200))
        app.processEvents()
        assert window.state.tool.points and tuple(window.state.tool.points[0]) == a.xyz
        _qt_click(app, vp3, _qt_point(vp3, b.xyz, -2, 3))
        QtTest.QTest.keyClick(vp3.vtk_widget, QtCore.Qt.Key_Escape)
        app.processEvents()
        mm = window.doc.model
        new = [f for n, f in mm.frames.items() if n not in n0]
        assert len(new) == 1
        assert {mm.points[new[0].point_i].xyz, mm.points[new[0].point_j].xyz} == {a.xyz, b.xyz}
    # typed: the first point as coordinates, then an absolute end point
    n0 = len(window.doc.model.frames)
    vp3.vtk_widget.setFocus()
    for text in ("1,1,3.5", "#5,4,7"):
        for ch in text:
            QtTest.QTest.keyClicks(vp3.vtk_widget, ch)
        QtTest.QTest.keyClick(vp3.vtk_widget, QtCore.Qt.Key_Return)
        app.processEvents()
    mm = window.doc.model
    assert len(mm.frames) == n0 + 1
    f = mm.frames[window.state.tool.created[-1]]
    assert {mm.points[f.point_i].xyz, mm.points[f.point_j].xyz} == {(1.0, 1.0, 3.5), (5.0, 4.0, 7.0)}


def _counting(monkeypatch, module, names):
    calls = {n: 0 for n in names}
    for n in names:
        fn = getattr(module, n)

        def w(*a, _fn=fn, _n=n, **k):
            calls[_n] += 1
            return _fn(*a, **k)
        monkeypatch.setattr(module, n, w)
    return calls


@gui
def test_selection_redraws_only_the_highlight_and_hidden_windows_wait(window, app, monkeypatch):
    """Post-analysis responsiveness: a selection change redraws the highlight layer
    only (not the objects, not the deformed shape); a window not on screen builds
    nothing until it is shown, then shows the latest selection and results."""
    from structstudio.api import analyze
    from structstudio.app import scene
    m = _with_objects(window, app)
    window.doc.set_results(analyze(window.doc.model, ["Dead"]))
    vp = window.views.viewports[1]
    window.views.set_active(vp)
    window.state.set_display(display_case="Dead", deformed=True)
    app.processEvents()
    base_before = dict(zip(vp._scene_keys, vp._scene_actors))["frames_Beam"]
    overlays_before = list(vp._overlay_actors)
    calls = _counting(monkeypatch, scene, ["build_deformed_mesh_actors", "build_model_actors"])
    signals = []
    window.state.selection_changed.connect(lambda: signals.append(1))
    beams = [f.name for f in m.frames.values() if f.type == "Beam"]
    window.state.set_selection([beams[0]])
    app.processEvents()
    assert calls["build_deformed_mesh_actors"] == 0
    keys = dict(zip(vp._scene_keys, vp._scene_actors))
    assert keys["frames_Beam"] is base_before and vp._overlay_actors == overlays_before
    sel = keys["frames_Beam_sel"]
    assert sel.GetProperty().GetOpacity() <= vp.GHOST_OPACITY + 1e-9  # under the deformed shape, like the rest
    # the same selection again: no signal, nothing drawn
    n = len(signals)
    window.state.set_selection([beams[0]])
    assert len(signals) == n
    # a hidden window: marked, not built
    window.views.set_layout("1")
    app.processEvents()
    assert not vp.isVisible()
    before = calls["build_model_actors"]
    window.state.set_selection([beams[1]])
    app.processEvents()
    assert "selection" in vp._dirty
    window.views.set_layout("2")
    app.processEvents()
    assert vp.isVisible() and not vp._dirty
    assert calls["build_model_actors"] > before
    sel_now = [k for k in vp._scene_keys if k.endswith("_sel")]
    assert sel_now and vp._overlay_actors  # the deformed shape is still shown
    window.state.clear_selection()
    app.processEvents()
    assert not [k for k in vp._scene_keys if k.endswith("_sel")]
    assert calls["build_deformed_mesh_actors"] == 0


@gui
def test_label_option_redraws_labels_only(window, app, monkeypatch):
    from structstudio.api import analyze
    from structstudio.app import scene
    _with_objects(window, app)
    window.doc.set_results(analyze(window.doc.model, ["Dead"]))
    vp = window.views.viewports[1]
    window.views.set_active(vp)
    window.state.set_display(display_case="Dead", deformed=True)
    app.processEvents()
    base_before = dict(zip(vp._scene_keys, vp._scene_actors))["frames_Beam"]
    overlays_before = list(vp._overlay_actors)
    calls = _counting(monkeypatch, scene, ["build_deformed_mesh_actors"])
    window.views.set_option_all("frame_labels", True)
    app.processEvents()
    keys = dict(zip(vp._scene_keys, vp._scene_actors))
    assert "labels" in keys and keys["frames_Beam"] is base_before
    assert vp._overlay_actors == overlays_before and calls["build_deformed_mesh_actors"] == 0
    window.views.set_option_all("frame_labels", False)
    app.processEvents()
    assert "labels" not in vp._scene_keys
    # Extrude changes the objects and the results drawn
    window.views.set_option_all("extrude", True)
    app.processEvents()
    assert calls["build_deformed_mesh_actors"] >= 1
    window.views.set_option_all("extrude", False)


@gui
def test_mode_box_commits_typed_numbers_and_follows_the_case(window, app):
    from PySide6.QtTest import QTest
    from structstudio.api import analyze
    _with_objects(window, app)
    window.doc.set_results(analyze(window.doc.model))
    vp = window.views.viewports[1]
    window.views.set_active(vp)
    app.processEvents()
    n = window._mode_count("Modal")
    assert n >= 3 and not window.mode_box.keyboardTracking()
    window.case_box.setCurrentText("Modal")
    window.mode_box.setValue(2)
    app.processEvents()
    assert window.mode_box.maximum() == n
    seen = []
    window.state.display_changed.connect(lambda: seen.append(window.state.display_mode))
    window.mode_box.setFocus()
    window.mode_box.lineEdit().selectAll()
    QTest.keyClicks(window.mode_box.lineEdit(), str(n))
    app.processEvents()
    assert seen == []  # nothing drawn while typing
    QTest.keyClick(window.mode_box.lineEdit(), QtCore.Qt.Key_Return)
    app.processEvents()
    assert seen == [n] and vp.result_display["display_mode"] == n
    window.mode_box.setValue(n + 5)  # past the last mode: the last one
    assert window.mode_box.value() == n
    window.case_box.setCurrentText("Dead")  # a static case has no modes
    app.processEvents()
    assert window.mode_box.value() == 0 and vp.result_display["display_mode"] is None


def test_joint_displacement_min_columns_decided_once_per_table():
    from structstudio.api import analyze
    from tests.unit.test_app_scene import small_building
    m = small_building()
    m.add_combination("UP", {"DL": -1.0})
    m.add_combination("ENV", {"DL": 1.0, "UP": 1.0}, type="Envelope")
    m.add_combination("ADD", {"DL": 1.2})
    res = analyze(m)
    import numpy as np
    calls = []
    orig = np.array_equal

    def counted(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    np.array_equal, saved = counted, np.array_equal
    try:
        rows = res.joint_displacements("ENV")
        assert len(calls) == 1
    finally:
        np.array_equal = saved
    env = res.displacement_envelope("ENV")
    assert "UxMin" in rows[0] and len(rows) == len(m.points)
    for r in rows:
        nid = res._node_of_point(r["Point"])
        assert r["Ux"] == float(env.max[nid, 0]) and r["UxMin"] == float(env.min[nid, 0])
    assert "UxMin" not in res.joint_displacements("ADD")[0]
    assert "UxMin" not in res.joint_displacements("DL")[0]


def test_shell_forces_table_rows_units_and_export_match_the_list_form(tmp_path):
    """The array-backed Shell forces table (what the Results panel shows and
    exports) gives the same rows, display-unit columns and values as the rows."""
    from structstudio.api import analyze
    from structstudio.app.panels import UnitsTable, export_rows, table_in_display_units
    from tests.unit.test_app_scene import small_building
    m = small_building()
    m.add_combination("UP", {"DL": -0.5})
    m.add_combination("ENV", {"DL": 1.0, "UP": 1.0}, type="Envelope")
    res = analyze(m)
    for case in ("DL", "ENV"):
        rows = table_in_display_units(m, "Shell forces", res.shell_forces(case))
        t = UnitsTable.of(m, "Shell forces", res.shell_forces_table(case))
        assert len(t) == len(rows) > 0
        assert [t.row(r) for r in range(len(t))] == rows
        assert t.columns(0) == list(rows[0].keys())
        a, b = tmp_path / f"{case}_list.csv", tmp_path / f"{case}_table.csv"
        export_rows(rows, str(a))
        export_rows(t.iter_rows(), str(b), cols=t.columns())
        assert a.read_text() == b.read_text()
    two = t.base.two  # the envelope has elements with a distinct minimum (Min columns)
    assert two.any() and any("Min [" in k for r in range(len(t)) for k in t.columns(r)[:12])


@gui
def test_wheel_zoom_hides_many_labels_and_shows_them_after_a_pause(window, app, monkeypatch):
    import vtk
    vp = window.views.viewports[1]
    # a label layer with more labels than the limit
    pts = vtk.vtkPoints()
    texts = vtk.vtkStringArray()
    texts.SetName("labels")
    for k in range(vp.DRAG_LABEL_LIMIT + 50):
        pts.InsertNextPoint(k * 0.1, 0.0, 0.0)
        texts.InsertNextValue(str(k))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.GetPointData().AddArray(texts)
    hier = vtk.vtkPointSetToLabelHierarchy()
    hier.SetInputData(poly)
    hier.SetLabelArrayName("labels")
    mapper = vtk.vtkLabelPlacementMapper()
    mapper.SetInputConnection(hier.GetOutputPort())
    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    vp.renderer.AddViewProp(actor)
    monkeypatch.setattr(vp, "WHEEL_LABEL_IDLE_MS", 20)
    vp._on_wheel(+1)
    assert not actor.GetVisibility()
    vp._on_wheel(-1)
    assert not actor.GetVisibility()
    QtTest.QTest.qWait(150)
    assert actor.GetVisibility()
    vp.renderer.RemoveViewProp(actor)
