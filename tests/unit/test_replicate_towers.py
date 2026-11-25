"""Edit > Replicate > Story in a model of several towers (test_towers._three_towers:
T1 a podium 0 / 3.5 / 7, T2 a tower on its roof 7 / 10 / 13 over the same plan, T3 a
separate building at x = 40 with T1's elevations).  A story takes the objects of its own
tower: each object goes only to the ticked stories of its tower, its copies and their
joints get that tower's stories (also where a story of another tower is at the same
level), a block cannot cross towers, and an object with no ticked story of its tower
stays as it is (also with "Delete original objects").  The window part runs in a child
interpreter, as the other window tests."""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_towers import _three_towers  # noqa: E402

from structstudio.app.tools import replicate_to_stories  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"
gui = pytest.mark.skipif(not CHILD, reason="runs in the child interpreter (test_gui_in_subprocess)")


def _floor(m, x0, z, name):
    sec = next(n for n, s in m.shell_sections.items() if s.type != "Wall")
    return m.add_area([(x0, 0, z), (x0 + 6, 0, z), (x0 + 6, 6, z), (x0, 6, z)], sec, name=name)


def _doc():
    """P1 on the podium (Story1, T1), T2F in the tower over it (T2-Story1), T3F in the
    separate building (T3-Story1)."""
    d = _three_towers()
    m = d.model
    _floor(m, 0, 3.5, "P1")
    _floor(m, 0, 10.0, "T2F")
    _floor(m, 40, 3.5, "T3F")
    assert [m.areas[n].story for n in ("P1", "T2F", "T3F")] == ["Story1", "T2-Story1", "T3-Story1"]
    d._mark_saved()
    return d


def _where(d, names):
    m = d.model
    return sorted([(m.areas[n].story, m.tower_of_object("area", n), m.points[m.areas[n].points[0]].x,
                   m.points[m.areas[n].points[0]].z, tuple(sorted({m.points[p].story for p in m.areas[n].points})))
                  for n in names], key=str)


def _notes(d):
    return " | ".join(t for _, t in d.notes)


# ----------------------------------------------------------------- core, not the window
@pytest.mark.skipif(CHILD, reason="parent only")
def test_each_tower_to_its_own_story():
    d = _doc()
    assert _where(d, replicate_to_stories(d, [], ["T3F"], [], ["T3-Story2"])) == \
        [("T3-Story2", "T3", 40.0, 7.0, ("T3-Story2",))]
    d = _doc()
    assert _where(d, replicate_to_stories(d, [], ["T2F"], [], ["T2-Story2"])) == \
        [("T2-Story2", "T2", 0.0, 13.0, ("T2-Story2",))]


@pytest.mark.skipif(CHILD, reason="parent only")
def test_a_tower_story_at_the_level_of_a_podium_story_gets_the_towers_story():
    # T2-Base (7) is at the level of the podium's Story2: the copy and its joints are T2's
    d = _doc()
    made = replicate_to_stories(d, [], ["T2F"], [], ["T2-Base"])
    assert _where(d, made) == [("T2-Base", "T2", 0.0, 7.0, ("T2-Base",))]


@pytest.mark.skipif(CHILD, reason="parent only")
def test_a_story_of_another_tower_copies_nothing_and_says_why():
    for src, target in (("T3F", "Story2"), ("T3F", "T2-Story1"), ("P1", "T2-Story1"), ("T2F", "Story2")):
        d = _doc()
        before = (set(d.model.areas), set(d.model.points))
        assert replicate_to_stories(d, [], [src], [], [target]) == [], (src, target)
        assert (set(d.model.areas), set(d.model.points)) == before
        assert not d.can_undo and not d.dirty  # nothing changed: no undo step
        assert ("warning", next(t for _, t in d.notes if target in t)) in d.notes
        assert "own tower" in _notes(d)


@pytest.mark.skipif(CHILD, reason="parent only")
def test_a_selection_in_two_towers_goes_object_by_object():
    # the T2 floor does not come down onto the podium roof (z 7) as "Story2"
    d = _doc()
    made = replicate_to_stories(d, [], ["P1", "T2F"], [], ["Story2"])
    assert _where(d, made) == [("Story2", "T1", 0.0, 7.0, ("Story2",))]
    assert "tower T2 not copied" in _notes(d)
    d = _doc()
    made = replicate_to_stories(d, [], ["P1", "T2F", "T3F"], [], ["Story2", "T2-Story2", "T3-Story2"])
    assert _where(d, made) == [("Story2", "T1", 0.0, 7.0, ("Story2",)),
                               ("T2-Story2", "T2", 0.0, 13.0, ("T2-Story2",)),
                               ("T3-Story2", "T3", 40.0, 7.0, ("T3-Story2",))]
    assert d.undo()
    assert set(d.model.areas) == {"P1", "T2F", "T3F"} and not d.dirty


@pytest.mark.skipif(CHILD, reason="parent only")
def test_frames_and_joints_of_a_tower():
    d = _doc()
    m = d.model
    sec = next(n for n in m.frame_sections if n.startswith("C"))
    with_t2 = m.add_frame((6, 6, 7.0), (6, 6, 10.0), sec, name="CT2")  # T2 column on the podium roof
    m.points[with_t2.point_i].story = "T2-Base"
    assert with_t2.story == "T2-Story1"
    m.add_frame((46, 6, 3.5), (46, 6, 7.0), sec, name="CT3")
    assert m.frames["CT3"].story == "T3-Story2"
    made = replicate_to_stories(d, ["CT2", "CT3"], [], [], ["T2-Story2", "T3-Story2", "Story2"])
    new = [n for n in made if n in m.frames]
    got = sorted((m.frames[n].story, m.points[m.frames[n].point_i].z, m.points[m.frames[n].point_j].z,
                  m.points[m.frames[n].point_i].story, m.points[m.frames[n].point_j].story) for n in new)
    # CT3 is on its top story already (T3-Story2 is its own): it is not copied
    assert got == [("T2-Story2", 10.0, 13.0, "T2-Story1", "T2-Story2")]
    assert "Story2 (tower T1): no selected object" in _notes(d)


@pytest.mark.skipif(CHILD, reason="parent only")
def test_delete_original_keeps_what_was_not_copied():
    d = _doc()
    assert replicate_to_stories(d, [], ["T3F"], [], ["Story2"], delete_original=True) == []
    assert "T3F" in d.model.areas and not d.dirty
    d = _doc()
    made = replicate_to_stories(d, [], ["P1", "T3F"], [], ["Story2"], delete_original=True)
    assert _where(d, made) == [("Story2", "T1", 0.0, 7.0, ("Story2",))]
    assert "P1" not in d.model.areas and "T3F" in d.model.areas


@pytest.mark.skipif(CHILD, reason="parent only")
def test_a_block_stays_in_the_tower_of_its_base_story():
    d = _doc()
    assert replicate_to_stories(d, [], ["T2F"], [], ["Story2"], base_story="T2-Story1") == []
    assert "a block moves within its own tower" in _notes(d)
    d = _doc()
    made = replicate_to_stories(d, [], ["T2F"], [], ["T2-Story2", "Story2"], base_story="T2-Story1")
    assert _where(d, made) == [("T2-Story2", "T2", 0.0, 13.0, ("T2-Story2",))]
    d = _doc()
    made = replicate_to_stories(d, [], ["P1", "T3F"], [], ["Story2"], base_story="Story1")
    assert _where(d, made) == [("Story2", "T1", 0.0, 7.0, ("Story2",))]
    assert "tower(s) T3 not copied" in _notes(d)


@pytest.mark.skipif(CHILD, reason="parent only")
def test_a_block_that_lands_between_levels_is_on_no_story():
    # one block from Story1: the Story2 floor raised by 3.5 m lands at 10.5 - no T1 level there
    d = _doc()
    m = d.model
    _floor(m, 0, 7.0, "P2")
    assert m.areas["P2"].story == "Story2"
    made = replicate_to_stories(d, [], ["P1", "P2"], [], ["Story2"], base_story="Story1")
    assert _where(d, made) == [("Story2", "T1", 0.0, 7.0, ("Story2",)), (None, "T1", 0.0, 10.5, (None,))]
    assert "between story levels" in _notes(d)


# ------------------------------------------------------------------------ the window
@pytest.mark.skipif(CHILD, reason="parent only")
def test_gui_in_subprocess():
    import shutil
    pytest.importorskip("vtk")
    pytest.importorskip("PySide6.QtWidgets")
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
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app, tmp_path):
    from PySide6 import QtCore
    from structstudio.app.mainwindow import MainWindow
    d = _doc()
    w = MainWindow(d, QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    app.processEvents()


def _replicate(win, app, monkeypatch, areas, tick):
    from structstudio.app import forms, menu_commands
    seen = {}

    def pick(dlg):
        dlg.tabs.setCurrentIndex(2)
        lst = dlg.pages[2]._widgets["stories"].list
        seen["items"] = [(lst.item(i).text(), forms.FormDialog._check_name(lst.item(i))) for i in range(lst.count())]
        seen["ticked"] = dlg.pages[2].values()["stories"]
        dlg.pages[2].set_value("stories", tick)
        dlg.accept()
        return dlg.result()
    monkeypatch.setattr(forms.TabbedFormDialog, "exec", pick)
    asked = []
    monkeypatch.setattr(menu_commands, "ask", lambda *a, **k: asked.append(a) or None)
    monkeypatch.setattr(menu_commands.QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(menu_commands.QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    win.state.set_selection([], areas)
    before = set(win.doc.model.areas)
    win.menu_actions["edit.replicate"].trigger()
    app.processEvents()
    seen["asked"] = asked
    seen["made"] = sorted(set(win.doc.model.areas) - before)
    return seen


@gui
def test_the_story_list_of_a_selection_in_two_towers(win, app, monkeypatch):
    seen = _replicate(win, app, monkeypatch, ["P1", "T3F"], ["Story2", "T3-Story2"])
    # the stories of the towers of the selection (no Base, not T2's), each named with its tower
    assert seen["items"] == [("T3-Story2  (T3)", "T3-Story2"), ("Story2  (T1)", "Story2")]
    assert sorted(seen["ticked"]) == ["Story2", "T3-Story2"]
    assert not seen["asked"]  # no "one block" question: a block cannot cross towers
    assert _where(win.doc, seen["made"]) == [("Story2", "T1", 0.0, 7.0, ("Story2",)),
                                             ("T3-Story2", "T3", 40.0, 7.0, ("T3-Story2",))]


@gui
def test_the_story_list_of_a_tower_on_a_podium(win, app, monkeypatch):
    seen = _replicate(win, app, monkeypatch, ["T2F"], ["T2-Story2"])
    assert seen["items"] == [("T2-Story2", "T2-Story2")]  # one tower: plain names
    assert _where(win.doc, seen["made"]) == [("T2-Story2", "T2", 0.0, 13.0, ("T2-Story2",))]
    assert win.doc.undo() and set(win.doc.model.areas) == {"P1", "T2F", "T3F"}
