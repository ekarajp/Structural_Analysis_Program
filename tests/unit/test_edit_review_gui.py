"""Edit menu review E01 - E13 through the program window: the menu actions (QAction)
of each group - Trim, Replicate (units, a full ring, several stories), Merge Shells,
Cut / Copy / Paste (drawn panels, Esc, a section of the same name from another
model), Delete (a line spring's joint), Undo / Redo back to the saved file.  Runs in a
child interpreter, as the other window tests."""
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
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if "WARN|" not in ln]
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower(), \
        "\n".join(lines[-60:])


@pytest.fixture(scope="module")
def app():
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _model(depth=0.30):
    from structstudio import Model
    from structstudio.model import FrameSection, ShellSection
    m = Model("Audit", "kN m")
    m.add_material("C", E=25e6, nu=0.2, weight_density=24)
    m.add_frame_section(FrameSection.rectangular("B", "C", depth, 0.20))
    m.add_shell_section(ShellSection("Slab", "C", thickness=0.15))
    m.add_load_pattern("Q", "Live")
    m.set_stories([("L1", 3), ("L2", 3), ("L3", 3)])
    return m


def _panel(m, x=0.0, name="S", z=3.0):
    return m.add_area([(x, 0, z), (x + 2, 0, z), (x + 2, 2, z), (x, 2, z)], "Slab", name=name)


@pytest.fixture
def win(app, tmp_path):
    from structstudio.app.document import Document
    from structstudio.app.mainwindow import MainWindow
    w = MainWindow(Document(_model()), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    errors = []
    old = sys.excepthook
    sys.excepthook = lambda t, e, b: errors.append(e)
    w._test_errors = errors

    def load(m):
        w.state.set_tool("select")
        w.state.clear_selection()
        w.doc.set_imported(m, "")
        app.processEvents()
        return w.doc.model
    w.load = load
    yield w
    sys.excepthook = old
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    app.processEvents()


def _act(w, app, key):
    w.menu_actions[key].trigger()
    app.processEvents()


@gui
def test_e01_trim_from_the_tool_keeps_the_point_load_once(win, app):
    from structstudio.model.frame_ops import resultant
    m = _model()
    f = m.add_frame((0, 0, 3), (10, 0, 3), "B", name="Loaded")
    for x in (4, 6):
        m.add_frame((x, -1, 3), (x, 1, 3), "B")
    m.add_frame_point_load(f.name, "Q", 10, rel=0.2)
    m = win.load(m)
    _act(win, app, "tool.trim")
    t = win.state.tool
    assert t.name == "trim"
    t.set_story("L1")
    assert t.frame_click("Loaded", (5.0, 0.0))
    m = win.doc.model
    loads = [(f.name, ld) for f in m.frames.values() for ld in f.loads]
    assert len(loads) == 1
    f = m.frames[loads[0][0]]
    L = abs(m.points[f.point_j].x - m.points[f.point_i].x)
    F, Mi = resultant(f.loads, L, "Gravity")
    assert F == pytest.approx(10.0) and m.points[f.point_i].x + Mi / F == pytest.approx(2.0)
    _act(win, app, "edit.undo")
    assert len(win.doc.model.frames) == 3 and not win._test_errors


@gui
def test_e04_replicate_offsets_in_the_coordinate_display_units_and_a_full_ring(win, app, monkeypatch):
    from structstudio.app import forms
    m = _model()
    m.display_units = {"coordinates": "mm"}
    f = m.add_frame((0, 0, 3), (2, 0, 3), "B", name="Source")
    m = win.load(m)
    win.state.set_selection([f.name])
    seen = {}

    def linear(dlg):
        page = dlg.pages[0]
        seen["label"] = page.layout().itemAt(0).layout().labelForField(page._widgets["dx"]).text()
        page._widgets["dx"].setText("100")
        dlg.accept()
        return dlg.result()
    monkeypatch.setattr(forms.TabbedFormDialog, "exec", linear)
    _act(win, app, "edit.replicate")
    m = win.doc.model
    assert seen["label"] == "dx [mm]"
    assert sorted(round(m.points[x.point_i].x, 6) for x in m.frames.values()) == [0.0, 0.1]
    # radial 90 deg x 4: the 4th copy would lie on the original - asked before anything is made
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QtWidgets.QMessageBox.No)

    def radial(dlg):
        dlg.tabs.setCurrentIndex(1)
        page = dlg.pages[1]
        page._widgets["angle"].setText("90")
        page._widgets["n"].setValue(4)
        dlg.accept()  # refused: the form stays open
        assert dlg.result() != QtWidgets.QDialog.Accepted and asked and "Copy 4 of 4" in asked[0]
        page._widgets["n"].setValue(3)
        dlg.accept()
        return dlg.result()
    win.state.set_selection(["Source"])
    n0 = len(win.doc.model.frames)
    monkeypatch.setattr(forms.TabbedFormDialog, "exec", radial)
    _act(win, app, "edit.replicate")
    assert len(win.doc.model.frames) == n0 + 3 and len(asked) == 1
    assert not [i for i in win.doc.model.validate() if "DUPLICATE" in str(i)]


@gui
def test_e06_replicate_several_stories_asks_what_the_selection_means(win, app, monkeypatch):
    from structstudio.app import forms, menu_commands
    m = _model()
    _panel(m, 0, "A", 3)
    _panel(m, 4, "B", 6)
    m = win.load(m)
    win.state.set_selection([], ["A", "B"])

    def pick(dlg):
        dlg.tabs.setCurrentIndex(2)
        dlg.pages[2].set_value("stories", ["L3"])
        dlg.accept()
        return dlg.result()
    monkeypatch.setattr(forms.TabbedFormDialog, "exec", pick)
    asked = []

    def how(parent, title, fields, note=None):
        asked.append(fields[0][3]["choices"])
        return {"how": fields[0][3]["choices"][0]}  # each object onto the ticked story
    monkeypatch.setattr(menu_commands, "ask", how)
    _act(win, app, "edit.replicate")
    m = win.doc.model
    assert asked and len(asked[0]) == 3
    assert sorted((a.story, m.points[a.points[0]].z) for a in m.areas.values() if a.name not in ("A", "B")) == \
        [("L3", 9.0), ("L3", 9.0)]


@gui
def test_e03_e11_merge_shells_from_the_menu_reports_and_never_raises(win, app, monkeypatch):
    said = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: said.append(a[2]))
    m = _model()
    a = _panel(m)
    m.add_area(list(a.points), "Slab", name="Overlap")
    m = win.load(m)
    win.state.set_selection([], ["S", "Overlap"])
    _act(win, app, "edit.merge_shells")
    assert not win._test_errors and len(win.doc.model.areas) == 2 and "lie on each other" in said[-1]
    m = _model()
    a = _panel(m, 0, "A")
    _panel(m, 2, "B")
    a.spring = 25000
    m.add_area_load("A", "Q", 5)
    m = win.load(m)
    k = len(win.doc._undo)
    win.state.set_selection([], ["B", "A"])
    _act(win, app, "edit.merge_shells")
    assert len(win.doc.model.areas) == 2 and len(win.doc._undo) == k and "area spring" in said[-1]


@gui
def test_e08_e12_cut_a_drawn_panel_paste_it_and_esc_before_placing(win, app):
    from PySide6 import QtTest
    from structstudio.model.objects import PanelLoadCase
    m = _model()
    f = m.add_frame((0, 0, 3), (4, 0, 3), "B")
    pl = m.add_panel_load([(0, 0, 3), (4, 0, 3), (4, 2, 3), (0, 2, 3)], span_dir=(1, 0, 0),
                          loads=[PanelLoadCase("Q", 2.0)])
    m = win.load(m)
    win.state.set_selection([f.name])
    win.state.panels = {pl.name}
    _act(win, app, "edit.cut")
    clip = win.state.clipboard
    assert len(clip.frames) == 1 and len(clip.free_panels) == 1
    assert not win.doc.model.frames and not win.doc.model.panel_loads
    _act(win, app, "edit.paste")
    assert win.state.tool.name == "paste"
    QtTest.QTest.keyClick(win.views.active.vtk_widget, QtCore.Qt.Key_Escape)  # before the click: nothing
    app.processEvents()
    assert win.state.tool.name == "select" and not win.doc.model.frames
    _act(win, app, "edit.paste")
    win.state.tool.click((10.0, 0.0, 3.0))
    m = win.doc.model
    assert len(m.frames) == 1 and len(m.panel_loads) == 1
    q = next(iter(m.panel_loads.values()))
    assert q.points[0] == pytest.approx((10, 0, 3)) and q.loads[0].value == 2.0 and q.span_dir == (1, 0, 0)
    _act(win, app, "edit.undo")
    assert not win.doc.model.panel_loads
    _act(win, app, "edit.redo")
    assert len(win.doc.model.panel_loads) == 1


@gui
def test_e07_paste_from_another_model_asks_for_a_section_of_the_same_name(win, app, monkeypatch):
    src = _model(0.30)
    f = src.add_frame((0, 0, 3), (2, 0, 3), "B")
    win.load(src)
    win.state.set_selection([f.name])
    _act(win, app, "edit.copy")
    dest = win.load(_model(0.60))
    choice = {"text": None}

    def answer(box):
        b = next(x for x in box.buttons() if x.text() == choice["text"]) if choice["text"] else \
            box.button(QtWidgets.QMessageBox.Cancel)
        b.click()
        return 0
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", answer)
    _act(win, app, "edit.paste")  # Cancel: no paste at all
    assert win.state.tool.name != "paste" and not win.doc.model.frames
    choice["text"] = "Bring in under new names"
    _act(win, app, "edit.paste")
    win.state.tool.click((5.0, 0.0, 3.0))
    m = win.doc.model
    g = next(iter(m.frames.values()))
    assert g.section == "B (2)" and m.frame_sections["B (2)"].I33 == pytest.approx(src.frame_sections["B"].I33)
    assert m.frame_sections["B"].I33 == pytest.approx(dest.frame_sections["B"].I33)


@gui
def test_e10_delete_a_line_spring_joint_from_the_window(win, app, monkeypatch):
    from structstudio.model.objects import LineSpring
    m = _model()
    a, b = m.add_point(0, 0, 0), m.add_point(4, 0, 0)
    m.line_springs["LS"] = LineSpring("LS", a.name, b.name, (0, 0, 200))
    m = win.load(m)
    answers = [QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes]
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: answers.pop(0))
    win.state.set_selection(points=[a.name])
    win.act_delete.trigger()
    assert a.name in win.doc.model.points and "LS" in win.doc.model.line_springs
    win.state.set_selection(points=[a.name])
    win.act_delete.trigger()
    m = win.doc.model
    assert a.name not in m.points and "LS" not in m.line_springs and not m.validate()


@gui
def test_e13_undo_back_to_the_saved_file_clears_the_star(win, app, tmp_path):
    m = win.load(_model())
    path = str(tmp_path / "audit.json")
    win.doc.save(path)
    app.processEvents()
    win.doc.edit(lambda mm: mm.add_point(4, 2, 3))
    app.processEvents()
    assert win.windowTitle().count("*") == 1 or win.doc.title.endswith("*")
    _act(win, app, "edit.undo")
    assert not win.doc.dirty and "*" not in win.doc.title
    _act(win, app, "edit.redo")
    assert win.doc.dirty
    del m
