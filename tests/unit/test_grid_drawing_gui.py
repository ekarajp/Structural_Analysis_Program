"""Grid Mode (Edit > Edit Grids on Screen): grid lines selected, drawn, offset, moved,
copied and deleted in a plan window like CAD, the model faded and untouched; the same
lines as the grid table (General lines too); one place per command in the menus.  GUI:
runs in a child interpreter like test_app_window."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("vtk")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from structstudio.app.document import Document  # noqa: E402
from structstudio.app.templates import build_model  # noqa: E402

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
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower(), \
        "\n".join(out.splitlines()[-60:])


@pytest.fixture(scope="module")
def app():
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, tmp_path):
    from PySide6 import QtCore
    from structstudio.app.mainwindow import MainWindow
    m = build_model(structure="rc_frame", stories=[("Story1", 3.5)], x_ordinates=[0, 6, 12], y_ordinates=[0, 5])
    w = MainWindow(Document(m), settings=QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.doc.ask_unlock = lambda: True
    w.resize(1600, 950)
    w.show()
    app.processEvents()
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    app.processEvents()


def _display(vp, x, y, z):
    vp.renderer.SetWorldPoint(x, y, z, 1.0)
    vp.renderer.WorldToDisplay()
    d = vp.renderer.GetDisplayPoint()
    return int(round(d[0])), int(round(d[1]))


def _plan(window):
    vp = window.views.viewports[0]
    vp.set_view("plan", story="Story1")
    window.views.set_active(vp)
    QtWidgets.QApplication.processEvents()
    vp.render()
    return vp


def _click(vp, x, y, z):
    vp.interactor.SetEventPosition(*_display(vp, x, y, z))
    vp.interactor.SetRepeatCount(0)
    vp.style.InvokeEvent("LeftButtonPressEvent")
    vp.style.InvokeEvent("LeftButtonReleaseEvent")


def _line(m, name, system="G1"):
    return next(g for g in m.grid_lines if (g.system or "G1") == system and g.name == name)


def _menu_texts(window, top):
    bar = window.menuBar()
    menu = next(a.menu() for a in bar.actions() if a.text().replace("&", "") == top)
    out = {}

    def walk(mn, path):
        for a in mn.actions():
            if a.menu() is not None:
                walk(a.menu(), path + [a.text()])
            elif not a.isSeparator():
                out[" > ".join(path + [a.text()])] = a
    walk(menu, [])
    return out


def _press_move_release(vp, a, b, z=3.5, ctrl=False):
    vp.interactor.SetEventPosition(*_display(vp, a[0], a[1], z))
    vp.interactor.SetRepeatCount(0)
    vp.interactor.SetControlKey(1 if ctrl else 0)
    vp.style.InvokeEvent("LeftButtonPressEvent")
    for k in range(1, 6):
        vp.interactor.SetEventPosition(*_display(vp, a[0] + (b[0] - a[0]) * k / 5, a[1] + (b[1] - a[1]) * k / 5, z))
        vp.style.InvokeEvent("MouseMoveEvent")
    vp.style.InvokeEvent("LeftButtonReleaseEvent")
    vp.interactor.SetControlKey(0)


def _bubble(vp, m, name):
    from structstudio.model.grids import grid_geometry
    from structstudio.app import scene
    end, d, _ = next(b for b in grid_geometry(m).bubbles if b[2] == name)
    r = scene._bubble_radius(m)
    off = r * (1.0 + 0.6 * scene.gsize("bubble_gap"))
    return end[0] + d[0] * off, end[1] + d[1] * off


@gui
def test_menus_one_place_per_grid_command(window):
    draw = _menu_texts(window, "Draw")
    assert not any("Grid" in k for k in draw)  # (drawn in Grid Mode)
    edit = _menu_texts(window, "Edit")
    grid_items = [k for k in edit if "Grid" in k]
    assert set(grid_items) >= {"Edit Stories and Grid Systems...", "Edit Grids on Screen",
                               "Add Grid Lines at Selected Joints", "Move Objects with Grid Lines"}
    assert not any("Lock" in k or "Grid Line Editing" in k for k in edit)
    assert not edit["Move Objects with Grid Lines"].isChecked()  # off by default
    edit["Edit Grids on Screen"].trigger()
    assert window.state.grid_mode and window.state.tool.name == "grid_select"
    assert window.grid_bar.isVisible() and not window.draw_toolbar.isVisible()
    assert "GRID MODE" in _plan(window).title.text()
    window.state.set_grid_mode(False)
    assert window.draw_toolbar.isVisible() and not window.grid_bar.isVisible()


@gui
def test_grid_mode_select_delete_and_model_untouched(window):
    vp = _plan(window)
    m0 = window.doc.model
    n_frames, n_points = len(m0.frames), len(m0.points)
    window.state.set_grid_mode(True)
    _click(vp, 6.0, 2.5, 3.5)  # B (a beam lies on it: the grid line is picked, not the beam)
    assert window.state.grid_sel == {("G1", "B")} and not window.state.frames
    vp.interactor.SetControlKey(1)
    _click(vp, 12.0, 2.5, 3.5)
    vp.interactor.SetControlKey(0)
    assert window.state.grid_sel == {("G1", "B"), ("G1", "C")}
    window.state.delete_selected()
    m = window.doc.model
    assert {g.name for g in m.grid_lines} == {"A", "1", "2"}
    assert (len(m.frames), len(m.points)) == (n_frames, n_points)  # the model stays
    window.doc.undo()
    # crossing box (right to left) over line 2 and the A..C span: takes A, B, C and 2
    _press_move_release(vp, (13.0, 6.0), (-1.0, 4.0))
    assert {k[1] for k in window.state.grid_sel} >= {"A", "B", "C", "2"}
    # Esc: clear, then leave
    vp.escape()
    assert not window.state.grid_sel and window.state.grid_mode
    vp.escape()
    assert not window.state.grid_mode


@gui
def test_grid_mode_drag_a_bubble_moves_only_the_line(window):
    vp = _plan(window)
    m = window.doc.model
    on_b = [n for n, p in m.points.items() if abs(p.x - 6.0) < 1e-9]
    c = _bubble(vp, m, "B")
    window.state.set_tool("select")
    _press_move_release(vp, c, (9.0, c[1]))  # outside Grid Mode: no drag
    assert _line(window.doc.model, "B").ordinate == 6.0
    window.state.set_grid_mode(True)
    _press_move_release(vp, c, (9.0, c[1]))
    x_new = _line(window.doc.model, "B").ordinate
    assert abs(x_new - 6.0) > 1.0
    assert all(abs(window.doc.model.points[n].x - 6.0) < 1e-9 for n in on_b)  # objects stay
    window.doc.undo()
    window.set_glue_grids(True)  # Move Objects with Grid Lines on: the joints follow
    _press_move_release(vp, c, (9.0, c[1]))
    x_new = _line(window.doc.model, "B").ordinate
    assert all(abs(window.doc.model.points[n].x - x_new) < 1e-9 for n in on_b)


@gui
def test_grid_mode_move_and_copy_with_typed_distance(window):
    vp = _plan(window)
    window.state.set_grid_mode(True)
    _click(vp, 12.0, 2.5, 3.5)  # C
    window.state.set_tool("grid_copy")
    _click(vp, 12.0, 0.0, 3.5)  # base point (grid point)
    vp.interactor.SetEventPosition(*_display(vp, 15.0, 0.0, 3.5))
    vp.style.InvokeEvent("MouseMoveEvent")
    vp._typed = "3"
    vp._apply_typed()  # 3 toward the cursor (+X)
    assert _line(window.doc.model, "C'").ordinate == pytest.approx(15.0)
    assert window.state.grid_sel == {("G1", "C'")}
    window.state.set_tool("grid_move")
    _click(vp, 15.0, 0.0, 3.5)
    vp.interactor.SetEventPosition(*_display(vp, 15.0, -3.0, 3.5))
    vp.style.InvokeEvent("MouseMoveEvent")
    vp._typed = "2"
    vp._apply_typed()  # an X line moved along Y: unchanged ordinate -> nothing moves
    assert _line(window.doc.model, "C'").ordinate == pytest.approx(15.0)
    _click(vp, 15.0, 0.0, 3.5)
    _click(vp, 16.5, 0.0, 3.5)
    assert _line(window.doc.model, "C'").ordinate == pytest.approx(16.5, abs=0.05)


@gui
def test_grid_mode_draw_and_offset_like_cad(window):
    vp = _plan(window)
    window.state.set_grid_mode(True)
    window._grid_tool("grid_line")
    _click(vp, 12.0, 0.0, 3.5)
    _click(vp, 6.0, 5.0, 3.5)
    g = _line(window.doc.model, "L1")
    assert g.direction == "General"
    window._grid_tool("grid_offset")
    tool = window.state.tool
    _click(vp, 6.0, 2.5, 3.5)  # picks B
    assert tool.picked == ("G1", "B")
    vp.interactor.SetEventPosition(*_display(vp, 8.0, 2.5, 3.5))
    vp.style.InvokeEvent("MouseMoveEvent")
    vp._typed = "1.5"
    vp._apply_typed()
    assert _line(window.doc.model, "B'").ordinate == pytest.approx(7.5)
    assert window.state.grid_mode  # the grid tools stay in Grid Mode


@gui
def test_grid_mode_right_click_menu_and_outside(window, monkeypatch):
    vp = _plan(window)
    from structstudio.app import grid_tools
    seen = {}
    monkeypatch.setattr(grid_tools, "grid_line_menu", lambda win, v, hit, x, y: seen.setdefault("out", hit))
    window.show_background_menu(vp, *_display(vp, 3.0, 5.0, 3.5))  # outside Grid Mode, on line 2
    assert seen.get("out") == ("G1", "2")
    monkeypatch.setattr(grid_tools, "grid_mode_menu", lambda win, v, hit, x, y: seen.setdefault("in", hit))
    window.state.set_grid_mode(True)
    vp.interactor.SetEventPosition(*_display(vp, 3.0, 5.0, 3.5))
    vp._right_press = vp.interactor.GetEventPosition()
    vp._nav_moved = False
    vp._on_right_release(None, None)
    assert seen.get("in") == ("G1", "2") and window.state.grid_sel == {("G1", "2")}


@gui
def test_table_and_screen_are_the_same_grid_lines(window):
    """A General line drawn on screen is in the table's General tab; one typed there is
    drawn; a changed ordinate moves only the line unless Move Objects with Grid Lines."""
    from structstudio.app.dialogs import GridStoriesDialog
    from structstudio.app.grid_tools import draw_grid_line
    draw_grid_line(window.doc, (0.0, 0.0), (6.0, 5.0))
    d = GridStoriesDialog(window.doc, window, system="G1", line="L1")
    assert d.tabs.currentWidget() is d.g_page
    t = d.g_table
    assert [t.item(0, c).text() for c in range(5)] == ["L1", "0", "0", "6", "5"]
    t.item(0, 3).setText("12")
    d._add_row(t, ("L2", 0.0, 5.0, 12.0, 0.0))
    d.accept()
    m = window.doc.model
    assert tuple(_line(m, "L1").end) == (12.0, 5.0) and _line(m, "L2").direction == "General"
    on_c = [n for n, p in m.points.items() if abs(p.x - 12.0) < 1e-9]
    for glue, x in ((False, 12.0), (True, 13.0)):
        window.doc.glue_grids = glue
        d = GridStoriesDialog(window.doc, window, system="G1", line="C")
        t = d.x_table
        t.item(t.currentRow(), 1).setText("13")
        d.accept()
        assert d.glued == (len(on_c) if glue else 0)
        assert all(abs(window.doc.model.points[n].x - x) < 1e-9 for n in on_c)
        window.doc.undo()
