"""Grid lines in a plan window: Draw Grid Line (two clicks), Offset Grid Line (click the
line on the side of the copy), dragging a bubble (with / without the joints, locked), the
right-click menu and the menus.  GUI: runs in a child interpreter like test_app_window."""
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


@gui
def test_menus_draw_and_edit_grid_lines(window):
    draw = _menu_texts(window, "Draw")
    assert draw["Draw Grid Lines (Plan)"].isEnabled() and draw["Offset Grid Line (Plan)"].isEnabled()
    assert not any("Draw Grids" in k for k in draw)
    edit = _menu_texts(window, "Edit")
    for k in ("Grid Lines > Move Objects with Grid Lines", "Grid Lines > Lock Grid Lines Against Dragging",
              "Grid Lines > Add Grid Lines at Selected Joints", "Grid Lines > Show All Hidden Grid Lines"):
        assert edit[k].isEnabled(), k
    assert edit["Grid Lines > Move Objects with Grid Lines"].isChecked()  # on by default
    edit["Grid Lines > Lock Grid Lines Against Dragging"].trigger()
    assert window.doc.lock_grids
    edit["Grid Lines > Lock Grid Lines Against Dragging"].trigger()
    assert not window.doc.lock_grids


@gui
def test_draw_a_grid_line_with_two_clicks_on_grid_points(window):
    vp = _plan(window)
    window.state.set_tool("grid_line")
    z = 3.5
    _click(vp, 12.0, 0.0, z)  # (grid points: snapped)
    _click(vp, 12.0 - 6.0, 5.0, z)  # a diagonal: a General line
    m = window.doc.model
    g = _line(m, "L1")
    assert g.direction == "General"
    assert g.start == pytest.approx((12.0, 0.0), abs=1e-6) and g.end == pytest.approx((6.0, 5.0), abs=1e-6)
    window.doc.undo()
    assert not any(gl.name == "L1" for gl in window.doc.model.grid_lines)


@gui
def test_offset_grid_line_as_cad_offset(window):
    """Click the line, then type the distance + Enter (to the cursor's side); or click the
    point the copy passes through; Enter alone repeats the last distance; Esc drops the line."""
    vp = _plan(window)
    window.state.set_tool("grid_offset")
    tool = window.state.tool
    _click(vp, 6.0, 2.5, 3.5)  # picks B
    assert tool.picked == ("G1", "B")
    vp.interactor.SetEventPosition(*_display(vp, 8.0, 2.5, 3.5))  # cursor right of B
    vp.style.InvokeEvent("MouseMoveEvent")
    for ch in "1.5":
        vp._typed += ch
    vp._apply_typed()
    m = window.doc.model
    assert _line(m, "B'").ordinate == pytest.approx(7.5) and tool.picked is None
    _click(vp, 6.0, 2.5, 3.5)  # B again; the cursor left of it; Enter = 1.5 again
    vp.interactor.SetEventPosition(*_display(vp, 4.0, 2.5, 3.5))
    vp.style.InvokeEvent("MouseMoveEvent")
    assert tool.key("Return")
    assert _line(window.doc.model, "B''").ordinate == pytest.approx(4.5)
    _click(vp, 0.0, 2.5, 3.5)  # A, then the point the copy passes through
    _click(vp, -2.0, 2.5, 3.5)
    assert _line(window.doc.model, "A'").ordinate == pytest.approx(-2.0, abs=0.05)
    _click(vp, 12.0, 2.5, 3.5)  # C picked, Esc drops it (the tool stays)
    assert tool.picked is not None and tool.key("Escape") and tool.picked is None
    assert window.state.tool is tool


@gui
def test_drag_a_bubble_moves_the_line_and_its_joints(window):
    from structstudio.model.grids import grid_geometry
    from structstudio.app import scene
    vp = _plan(window)
    window.state.set_tool("select")
    m = window.doc.model
    on_b = [n for n, p in m.points.items() if abs(p.x - 6.0) < 1e-9]
    end, d, _ = next(b for b in grid_geometry(m).bubbles if b[2] == "B")
    r = scene._bubble_radius(m)
    off = r * (1.0 + 0.6 * scene.gsize("bubble_gap"))
    c = (end[0] + d[0] * off, end[1] + d[1] * off)
    assert vp.grid_bubble_under(*_display(vp, c[0], c[1], 3.5)) == ("G1", "B")

    def drag(to_x):
        vp.interactor.SetEventPosition(*_display(vp, c[0], c[1], 3.5))
        vp.style.InvokeEvent("LeftButtonPressEvent")
        for k in range(1, 6):
            vp.interactor.SetEventPosition(*_display(vp, c[0] + (to_x - c[0]) * k / 5, c[1], 3.5))
            vp.style.InvokeEvent("MouseMoveEvent")
        vp.style.InvokeEvent("LeftButtonReleaseEvent")

    drag(9.0)
    x_new = _line(window.doc.model, "B").ordinate
    assert abs(x_new - 6.0) > 1.0  # moved (to the cursor, or the snap near it)
    assert all(abs(window.doc.model.points[n].x - x_new) < 1e-9 for n in on_b)
    window.doc.undo()
    assert _line(window.doc.model, "B").ordinate == 6.0
    # locked: the bubble does not drag (the press is a plain select click)
    window.set_lock_grids(True)
    drag(9.0)
    assert _line(window.doc.model, "B").ordinate == 6.0
    window.set_lock_grids(False)
    # objects stay: only the line moves
    window.set_glue_grids(False)
    drag(9.0)
    assert _line(window.doc.model, "B").ordinate != 6.0
    assert all(abs(window.doc.model.points[n].x - 6.0) < 1e-9 for n in on_b)


@gui
def test_right_click_on_a_grid_line_opens_its_menu(window, monkeypatch):
    vp = _plan(window)
    seen = {}
    from structstudio.app import grid_tools
    monkeypatch.setattr(grid_tools, "grid_line_menu", lambda win, v, hit, x, y: seen.setdefault("hit", hit))
    window.show_background_menu(vp, *_display(vp, 3.0, 5.0, 3.5))  # on line 2 (y = 5), between columns
    assert seen.get("hit") == ("G1", "2")


@gui
def test_grid_form_moves_the_joints_with_a_changed_ordinate(window):
    from structstudio.app.dialogs import GridStoriesDialog
    m = window.doc.model
    on_c = [n for n, p in m.points.items() if abs(p.x - 12.0) < 1e-9]
    d = GridStoriesDialog(window.doc, window, system="G1", line="C")
    t = d.x_table
    r = t.currentRow()
    assert t.item(r, 0).text() == "C"
    t.item(r, 1).setText("13")
    d.accept()
    assert d.glued == len(on_c) > 0
    assert all(abs(window.doc.model.points[n].x - 13.0) < 1e-9 for n in on_c)
