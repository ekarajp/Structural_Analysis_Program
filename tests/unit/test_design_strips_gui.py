"""Design strips in the program window: the menu commands, the strips drawn in a plan,
picked / selected / deleted, reshaped by their grips, drawn with Draw Design Strips -
all without unlocking an analysed model.  Runs in a child interpreter (xvfb when
available), as the other window tests."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("vtk")
QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

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
    from structstudio.app.document import Document
    from structstudio.app.mainwindow import MainWindow
    from structstudio.app.templates import build_model
    doc = Document(build_model(structure="flat_slab", x_ordinates=(0, 8, 16, 24), y_ordinates=(0, 6, 12)))
    w = MainWindow(doc)
    w.doc.ask_unlock = lambda: pytest.fail("a strip command must not ask to unlock the model")
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


def _click(vp, x, y, z, double=False):
    vp.interactor.SetEventPosition(*_display(vp, x, y, z))
    vp.interactor.SetRepeatCount(1 if double else 0)
    vp.style.InvokeEvent("LeftButtonPressEvent")
    vp.style.InvokeEvent("LeftButtonReleaseEvent")


def _plan(w, story="Story1"):
    vp = w.views.viewports[0]
    vp.set_view("plan", story=story)
    w.views.set_active(vp)
    QtWidgets.QApplication.processEvents()
    return vp, w.doc.model.stories[story].elevation


@gui
def test_menus_and_palette(window):
    from structstudio.app.menus import iter_commands
    ids = {cid for _, _, cid, _ in iter_commands()}
    assert {"edit.strips_auto", "edit.strip_widths", "tool.strip"} <= ids
    assert not ({"edit.slab_panels", "edit.support_lines", "edit.strips_grids", "edit.strips_support"} & ids)
    for cid in ("edit.strips_auto", "edit.strip_widths", "tool.strip"):
        assert window.menu_actions[cid].isEnabled(), cid
    assert window.tool_actions["strip"].shortcut().toString() == "D, S"


@gui
def test_auto_strips_shown_picked_deleted_kept_results(window, app):
    from structstudio.app import strip_commands as SC
    from structstudio.api import analyze
    vp, z = _plan(window)
    cases = [n for n in window.doc.model.load_cases if n in window.doc.model.load_patterns][:1]
    window.doc.set_results(analyze(window.doc.model, cases))  # (locks the model)
    sentinel = window.doc.results
    assert window.doc.locked and sentinel is not None
    SC.auto_strips(window.doc, ["Story1"])
    window._show_strips()
    app.processEvents()
    assert vp.strips_shown() and len(vp.visible_strips()) == 12 and vp._strip_actors
    # a plan of the other story shows its own strips (none)
    vp2 = window.views.viewports[1]
    vp2.set_view("plan", story="Story2")
    app.processEvents()
    assert vp2.visible_strips() == []
    # a click on a middle strip's centre line picks it (before the slab under it)
    x, y = _display(vp, 4, 3.0, z)
    assert vp.pick(x, y)[0] == "strip"
    _click(vp, 4, 3.0, z)
    st = window.state
    assert len(st.strips) == 1
    name = next(iter(st.strips))
    st.delete_selected()
    assert name not in window.doc.model.design_strips and window.doc.results is sentinel
    window.doc.undo()
    assert name in window.doc.model.design_strips and window.doc.results is sentinel and window.doc.locked
    # layer B hidden: only layer A strips in the window
    o = vp.display_options
    o.strips_layer_b = False
    vp.set_display_options(o)
    assert {ds.layer for ds in vp.visible_strips()} == {"A"}


@gui
def test_draw_strip_tool_and_reshape(window, app):
    vp, z = _plan(window)
    st = window.state
    st.set_tool("strip")
    app.processEvents()
    assert vp.strips_shown()  # the strips show while drawing them
    s = st.settings
    s.strip_layer, s.strip_kind, s.strip_widths = "B", "Column", "Specified"
    s.strip_width_left = s.strip_width_right = 1.25
    _click(vp, 8, 0, z)
    _click(vp, 8, 6, z)
    _click(vp, 8, 12, z, double=True)
    app.processEvents()
    made = [ds for ds in window.doc.model.design_strips.values() if ds.layer == "B"]
    assert len(made) == 1 and made[0].widths == [(1.25, 1.25, 1.25, 1.25)] * 2
    assert made[0].story == "Story1" and made[0].kind == "Column"
    # right click finishes a strip too
    _click(vp, 0, 3, z)
    vp.interactor.SetEventPosition(*_display(vp, 24, 3, z))
    vp.style.InvokeEvent("LeftButtonPressEvent")
    vp.style.InvokeEvent("LeftButtonReleaseEvent")
    vp.style.InvokeEvent("RightButtonPressEvent")
    vp.style.InvokeEvent("RightButtonReleaseEvent")
    app.processEvents()
    assert len(window.doc.model.design_strips) == 2
    # Reshape: the strip's grips; a drag of its middle point moves it
    name = made[0].name
    st.set_tool("select")
    st.toggle("strip", name, False)
    st.set_tool("reshape")
    app.processEvents()
    keys = [k for k, _ in st.tool.grips() if k[0] == "svert" and k[1] == name]
    assert len(keys) == 3
    tool = st.tool
    tool.set_plane(vp.plane())
    assert tool.pick_grip(("svert", name, 1))
    tool.click((9.0, 6.0))
    assert window.doc.model.design_strips[name].points[1][:2] == pytest.approx((9.0, 6.0))
    st.set_tool("select")


@gui
def test_strip_properties_form(window, app):
    from structstudio.app import strip_commands as SC
    from structstudio.app.strip_dialog import AutoStripsDialog, AutoWidthsDialog, StripDialog
    vp, z = _plan(window)
    d = AutoStripsDialog(window.doc.model, "Story1", window)
    assert d.chosen_stories() == ["Story1"] and d.options().source == "auto"
    d.reject()
    SC.auto_strips(window.doc, ["Story1"])
    name = next(n for n, ds in window.doc.model.design_strips.items() if ds.kind == "Column")
    f = StripDialog(window.doc, name, window)
    f.auto.setChecked(False)
    f.widths.item(0, 0).setText("2")
    f.ed_name.setText("CS-X")
    f._ok()
    ds = window.doc.model.design_strips["CS-X"]
    assert ds.widths[0][0] == pytest.approx(2.0) and not ds.auto_width and not ds.auto
    w = AutoWidthsDialog(window.doc.model, "Story1", ["CS-X"], window)
    assert w.names() == ["CS-X"]
    w.reject()
