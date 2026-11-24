"""Edit menu audit fixes in the program window: Edit Stories moves the objects with
their stories (default, a setting turns it off), Replicate refuses copies on the
originals and lists the stories to tick, strips go through Move / Copy / Ctrl+C,
Undo / Redo name the command.  Runs in a child interpreter, as the other window tests."""
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


def _window(app, tmp_path, model):
    from structstudio.app.document import Document
    from structstudio.app.mainwindow import MainWindow
    w = MainWindow(Document(model), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.resize(1500, 900)
    w.show()
    app.processEvents()
    return w


def _close(w, app):
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    app.processEvents()


@gui
def test_edit_stories_moves_the_objects(app, tmp_path):
    from structstudio.app.dialogs import GridStoriesDialog
    from structstudio.app.templates import new_building
    w = _window(app, tmp_path, new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.0, 3.0, 3.0),
                                            grid_only=False))
    try:
        assert w.doc.stories_carry and w.act_carry_stories.isChecked()
        assert "edit.carry_stories" in w.menu_actions
        d = GridStoriesDialog(w.doc, w)
        assert d.carry_box.isChecked()
        d.story_table.item(1, 1).setText("4")
        d.accept()
        zs = sorted({round(p.z, 6) for p in w.doc.model.points.values()})
        assert zs == [0.0, 4.0, 7.0, 10.0] and d.carried > 0
        assert w.doc.undo_name() == "Edit"  # (opened outside a menu here)
        w.doc.undo()
        # the setting off: only the levels change
        w.act_carry_stories.trigger()
        assert not w.doc.stories_carry
        assert w.settings_store().value("stories/move_objects") in (False, "false")
        d = GridStoriesDialog(w.doc, w)
        assert not d.carry_box.isChecked()
        d.story_table.item(1, 1).setText("4")
        d.accept()
        assert sorted({round(p.z, 6) for p in w.doc.model.points.values()}) == [0.0, 3.0, 6.0, 9.0]
        w.act_carry_stories.trigger()
    finally:
        _close(w, app)


@gui
def test_replicate_checks_and_story_list(app, tmp_path, monkeypatch):
    from structstudio.app import forms
    from structstudio.app.templates import new_building
    w = _window(app, tmp_path, new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0, 3.0, 3.0),
                                            grid_only=False))
    try:
        m = w.doc.model
        beam = next(n for n, f in m.frames.items() if f.type == "Beam" and f.story == "Story1")
        w.state.set_selection([beam])
        said = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: said.append(a[2]))
        runs = {"n": 0}

        def fake_exec(dlg):
            runs["n"] += 1
            if runs["n"] == 1:  # zero offset: refused, the form comes back
                dlg.tabs.setCurrentIndex(0)
                return 1
            dlg.tabs.setCurrentIndex(2)
            page = dlg.pages[2]
            lst = page._widgets["stories"].list
            names = [lst.item(i).text() for i in range(lst.count())]
            assert names == ["Story3", "Story2"]  # no Base, not its own story
            assert all(lst.item(i).checkState() == QtCore.Qt.Checked for i in range(lst.count()))
            return 1
        monkeypatch.setattr(forms.TabbedFormDialog, "exec", fake_exec)
        n0 = len(m.frames)
        w.edit_replicate_tabs()
        assert runs["n"] == 2 and said and "offset" in said[0]
        assert len(w.doc.model.frames) == n0 + 2
    finally:
        _close(w, app)


@gui
def test_strips_copy_paste_and_undo_names(app, tmp_path):
    from structstudio.app import strip_commands as SC
    from structstudio.app.templates import build_model
    w = _window(app, tmp_path, build_model(structure="flat_slab", x_ordinates=(0, 8, 16), y_ordinates=(0, 6)))
    try:
        SC.auto_strips(w.doc, ["Story1"])
        name = next(iter(w.doc.model.design_strips))
        w.state.set_selection()
        w.state.add_strips([name], add=False)
        assert w._have("any")
        w.menu_actions["edit.copy"].trigger()
        assert w.state.clipboard is not None and not w.state.clipboard.empty
        n = len(w.doc.model.design_strips)
        from structstudio.app import commands as C
        C.paste_objects(w.doc, w.state.clipboard, 0.0, 0.0, 3.5)
        assert len(w.doc.model.design_strips) == n + 1
        w.state.set_selection()
        w.state.add_strips([name], add=False)
        w.act_delete.trigger()
        assert name not in w.doc.model.design_strips
        assert w.act_undo.text() == "Undo Delete", w.act_undo.text()
        w.act_undo.trigger()
        assert name in w.doc.model.design_strips and w.act_redo.text() == "Redo Delete"
        assert w.menu_actions["tool.copy"].text().startswith("Copy by Base Point")
    finally:
        _close(w, app)
