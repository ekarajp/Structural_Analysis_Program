"""Edit commands in a large model (review 20261127: 15 x 15 bays x 40 stories, 49 000
objects).  What made them slow is checked by the work done, never by a stopwatch:

* the story of every object after a command: asked once per level, not once per object,
  with the same answer as ``Model.story_for`` (one tower and several);
* the checks after a command (members on each other, a missed snap, floors overlapping):
  those of the objects the command made are the same as the ones of a check of the whole
  model that involve them;
* Replicate / Paste look the joints of the copies up in the spatial index;
* the Model Explorer lists the objects of a story when its heading is opened.

The window part runs in a child interpreter, as the other window tests."""
import copy
import os
import random
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_towers import _three_towers  # noqa: E402

from structstudio.app import commands as C  # noqa: E402
from structstudio.app.document import Document  # noqa: E402
from structstudio.app.templates import new_building  # noqa: E402
from structstudio.app.tools import delete_objects, replicate_linear, replicate_to_stories  # noqa: E402
from structstudio.model import Model, checks  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"
gui = pytest.mark.skipif(not CHILD, reason="runs in the child interpreter (test_gui_in_subprocess)")
core = pytest.mark.skipif(CHILD, reason="parent only")


def _building(nx=3, ny=3, ns=4):
    return new_building(x_spacings=(6,) * nx, y_spacings=(5,) * ny, story_heights=(3.5,) * ns, grid_only=False)


def _restory_by_object(model):
    """What the program did before: ``story_for`` for every object."""
    for p in model.points.values():
        p.story = model.story_for(p, "point") or p.story
    for f in model.frames.values():
        f.story = model.story_for(f, "frame") or f.story
    for a in model.areas.values():
        a.story = model.story_for(a, "area") or a.story


def _stories(model):
    return ({n: p.story for n, p in model.points.items()}, {n: f.story for n, f in model.frames.items()},
            {n: a.story for n, a in model.areas.items()})


def _stir(model, rng):
    """Objects off their stories: wrong and missing stories, joints between levels, members
    across levels and beside the grids."""
    names = list(model.stories)
    for p in rng.sample(list(model.points.values()), min(12, len(model.points))):
        p.story = rng.choice(names + [None])
    for f in rng.sample(list(model.frames.values()), min(12, len(model.frames))):
        f.story = rng.choice(names + [None])
    for a in rng.sample(list(model.areas.values()), min(6, len(model.areas))):
        a.story = rng.choice(names + [None])
    for p in rng.sample(list(model.points.values()), min(6, len(model.points))):
        p.z += rng.choice([0.4, -0.7, 3.5, 3.0])
    sec = next(n for n in model.frame_sections if n.startswith("B"))
    slab = next(iter(model.shell_sections))
    for _ in range(8):
        x, y = rng.choice([0, 6, 40, 46, 100]), rng.choice([0, 5, 6])
        z = rng.choice([0.0, 3.5, 7.0, 10.0, 13.0, 5.2])
        try:
            f = model.add_frame((x, y, z), (x + rng.choice([3, 6]), y, rng.choice([z, z + 3.0])), sec)
            f.story = rng.choice([f.story, None])
            a = model.add_area([(x, y, z), (x + 6, y, z), (x + 6, y + 5, z), (x, y + 5, z)], slab)
            a.story = rng.choice([a.story, None])
        except Exception:
            pass


# ------------------------------------------------------------------------- stories
@core
@pytest.mark.parametrize("towers", [False, True])
def test_restory_gives_every_object_the_story_story_for_gives(towers):
    for seed in range(12):
        rng = random.Random(seed)
        m = _three_towers().model if towers else _building(2, 2, 3)
        if towers:
            slab = next(iter(m.shell_sections))
            for x, z in ((0, 3.5), (0, 10.0), (40, 3.5), (0, 7.0)):
                m.add_area([(x, 0, z), (x + 6, 0, z), (x + 6, 6, z), (x, 6, z)], slab)
        _stir(m, rng)
        for active in ([None, "T2", "T3", "no such tower"] if towers else [None]):
            a, b = copy.deepcopy(m), copy.deepcopy(m)
            a.active_tower = b.active_tower = active
            _restory_by_object(a)
            C._restory(b)
            assert _stories(a) == _stories(b), (seed, active)


@core
def test_restory_asks_once_per_level_not_once_per_object():
    m = _building(4, 4, 6)
    asked = []
    real = m.story_at
    m.story_at = lambda *a, **k: asked.append(a) or real(*a, **k)
    C._restory(m)
    del m.story_at
    assert len(m.points) + len(m.frames) + len(m.areas) > 500
    assert len(asked) <= 2 * len(m.stories), len(asked)


# --------------------------------------------------- the checks after a command
def _mess(m, rng, n=30):
    """Members on each other, ends beside joints, floors over floors (and walls)."""
    bs = next(k for k in m.frame_sections if k.startswith("B"))
    ss = next(iter(m.shell_sections))
    xs, ys, zs = [0, 3, 6, 9, 12, 6.05, 5.95, 2.5], [0, 2.5, 5, 10, 5.08, 4.9], [3.5, 7.0, 3.5, 3.52]
    for _ in range(n):
        z = rng.choice(zs)
        a, b = (rng.choice(xs), rng.choice(ys), z), (rng.choice(xs), rng.choice(ys), rng.choice([z, z, rng.choice(zs)]))
        if a != b:
            m.add_frame(a, b, bs)
    for _ in range(n // 2):
        z, x0, y0 = rng.choice([3.5, 7.0]), rng.choice([0, 3, 6, 1.5]), rng.choice([0, 2.5, 5, 1.0])
        w, h = rng.choice([3, 6, 4.5]), rng.choice([2.5, 5])
        m.add_area([(x0, y0, z), (x0 + w, y0, z), (x0 + w, y0 + h, z), (x0, y0 + h, z)], ss)
    for _ in range(n // 6):
        x0 = rng.choice([0, 3, 6])
        m.add_area([(x0, 0, 0), (x0 + rng.choice([3, 6]), 0, 0), (x0 + 6, 0, 3.5), (x0, 0, 3.5)], ss)


def _issue(i):
    return (i.code, tuple(sorted(i.frames or ())), tuple(sorted(getattr(i, "areas", ()) or ())))


@core
def test_checks_of_some_objects_are_those_of_the_whole_model_that_involve_them():
    seen = set()
    for seed in range(25):
        rng = random.Random(seed)
        m = _three_towers().model if seed % 3 == 2 else _building(2, 2, 2) if seed % 2 else \
            new_building(x_spacings=(6, 6), y_spacings=(5, 5), story_heights=(3.5, 3.5))
        _mess(m, rng)
        whole = [_issue(i) for i in checks.drawing_issues(m, list(m.frames), list(m.areas))]
        seen |= {c for c, _, _ in whole}
        for _ in range(4):
            fr = set(rng.sample(list(m.frames), rng.randint(1, min(8, len(m.frames)))))
            ar = set(rng.sample(list(m.areas), rng.randint(0, min(5, len(m.areas)))))
            part = [_issue(i) for i in checks.drawing_issues(m, list(fr), list(ar))]
            # an end of a member beside a joint: that member's; the others: any of the two
            want = [w for w in whole if (set(w[1]) & fr and w[0] != "NEAR_MISS") or (set(w[2]) & ar) or
                    (w[0] == "NEAR_MISS" and set(w[1]) <= fr)]
            assert sorted(part) == sorted(want), seed
    assert {"DUPLICATE_FRAME", "OVERLAPPING_FRAMES", "NEAR_MISS", "AREAS_OVERLAP"} <= seen  # (all were met)


@core
def test_overlap_of_floors_is_found_among_many_floors_and_far_ones_are_not_looked_at(monkeypatch):
    m = _building(6, 6, 5)
    slab = next(iter(m.shell_sections))
    over = m.add_area([(3, 2, 7.0), (9, 2, 7.0), (9, 4, 7.0), (3, 4, 7.0)], slab).name  # over two floors of Story2
    looked = []
    real = checks._plane_key
    monkeypatch.setattr(checks, "_plane_key", lambda model, a, tol: looked.append(a.name) or real(model, a, tol))
    found = [i for i in checks.drawing_issues(m, [], [over]) if i.code == "AREAS_OVERLAP"]
    assert sorted(sorted(i.areas) for i in found) == sorted(
        sorted([over, a.name]) for a in m.areas.values()
        if a.name != over and a.story == "Story2" and m.points[a.points[0]].x in (0.0, 6.0) and
        m.points[a.points[0]].y == 0.0)
    assert len(looked) < 12 < len(m.areas)  # the floors around it, not all 180


# -------------------------------------------------- Replicate / Paste: the joints
@core
def test_copies_look_their_joints_up_and_join_the_joints_already_there(monkeypatch):
    doc = Document(_building(3, 3, 4))
    m = doc.model
    scans = []
    real = Model.find_point
    monkeypatch.setattr(Model, "find_point", lambda self, *a, **k: scans.append(self._pindex is None) or real(self, *a, **k))
    top = [n for n, f in m.frames.items() if f.story == "Story4" and f.type == "Beam"]
    slabs = [n for n, a in m.areas.items() if a.story == "Story4"]
    delete_objects(doc, top, slabs)  # the columns stay: their top joints too
    joints = len(m.points)
    src = ([n for n, f in m.frames.items() if f.story == "Story2" and f.type == "Beam"],
           [n for n, a in m.areas.items() if a.story == "Story2"])
    del scans[:]
    made = replicate_to_stories(doc, src[0], src[1], [], ["Story4"])
    assert scans and not any(scans)  # every lookup through the index
    assert len(doc.model.points) == joints  # the copies stand on the column tops that were there
    assert len(made) == len(src[0]) + len(src[1])
    assert {doc.model.frames[n].story for n in made if n in doc.model.frames} == {"Story4"}
    assert doc.model._pindex is None  # (the index is the command's: gone after it)
    del scans[:]
    made = replicate_linear(doc, src[0], src[1], [], 18.0, 0.0, 0.0, 1)  # the next plan to the right: shares a column line
    assert scans and not any(scans)
    # the copies share the joints of the line x = 18 with the originals: one new line of joints less
    assert len(doc.model.points) == joints + 3 * 4
    del scans[:]
    clip = C.copy_objects(doc.model, src[0], src[1], [])
    C.paste_objects(doc, clip, 0.0, 15.0, 0.0)
    assert scans and not any(scans)
    assert len(doc.model.points) == joints + 3 * 4 + 3 * 4  # (the plan beside it, sharing the line y = 15)
    assert doc.model._pindex is None
    # a command that fails leaves no index behind
    with pytest.raises(Exception):
        replicate_to_stories(doc, src[0], src[1], [], ["Story4"], base_story="no such story")
    assert doc.model._pindex is None


# ------------------------------------------------------------------------ the window
@core
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
    w = MainWindow(Document(_building(3, 3, 4)), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    app.processEvents()


def _items(tree):
    from PySide6 import QtWidgets
    out, it = [], QtWidgets.QTreeWidgetItemIterator(tree)
    while it.value():
        out.append(it.value())
        it += 1
    return out


def _heading(tree, path):
    return next(i for i in _items(tree) if tree._key(i) == path)


@gui
def test_explorer_lists_the_objects_of_a_story_when_its_heading_is_opened(win, app):
    from PySide6 import QtCore
    tree, m = win.tree, win.doc.model
    n_objects = len(m.points) + len(m.frames) + len(m.areas)
    assert len(_items(tree)) < n_objects / 4  # headings, not an item per object
    h = _heading(tree, "objects/frames/Beams/Story2")
    beams = [n for n, f in m.frames.items() if f.type == "Beam" and f.story == "Story2"]
    assert h.text(0) == f"Story2 ({len(beams)})" and h.childCount() == 0
    assert h.childIndicatorPolicy() == h.ChildIndicatorPolicy.ShowIndicator  # (it can be opened)
    assert h.data(0, QtCore.Qt.UserRole) == ("set", {"frames": beams})  # a click still selects them all
    h.setExpanded(True)
    app.processEvents()
    assert [h.child(i).text(0) for i in range(h.childCount())] == [f"{n}    {m.frames[n].section}" for n in beams]
    assert [h.child(i).data(0, QtCore.Qt.UserRole) for i in range(h.childCount())] == [("frames", n) for n in beams]
    # an edit: the open heading stays open with the new member, the others stay closed
    sec = m.frames[beams[0]].section
    win.doc.edit(lambda mm: mm.add_frame((1.0, 2.0, 7.0), (5.0, 2.0, 7.0), sec, name="NEWBEAM"))
    app.processEvents()
    h = _heading(tree, "objects/frames/Beams/Story2")
    assert h.isExpanded() and h.childCount() == len(beams) + 1
    assert any(h.child(i).data(0, QtCore.Qt.UserRole) == ("frames", "NEWBEAM") for i in range(h.childCount()))
    assert _heading(tree, "objects/frames/Beams/Story3").childCount() == 0
    assert len(_items(tree)) < n_objects / 2


@gui
def test_explorer_finds_an_object_under_a_closed_heading_and_the_find_box_searches_all(win, app):
    from PySide6 import QtCore
    tree, m = win.tree, win.doc.model
    col = next(n for n, f in m.frames.items() if f.type == "Column" and f.story == "Story3")
    assert _heading(tree, "objects/frames/Columns/Story3").childCount() == 0
    item = tree.find_item("frames", col)
    assert item is not None and item.text(0).startswith(col) and item.parent().text(0).startswith("Story3")
    assert _heading(tree, "objects/frames/Columns/Story4").childCount() == 0  # only its own story was listed
    joint = next(iter(m.points))
    assert tree.find_item("points", joint).data(0, QtCore.Qt.UserRole) == ("points", joint)
    assert tree.find_item("frames", "no such member") is None
    # the find box: every object is searched (also under headings never opened)
    slab = next(n for n, a in m.areas.items() if a.story == "Story1")
    win.explorer.find.setText(slab)
    app.processEvents()
    shown = [i for i in _items(tree) if not i.isHidden() and i.data(0, QtCore.Qt.UserRole) == ("areas", slab)]
    assert len(shown) == 1
    win.explorer.find.setText("")
    app.processEvents()
    assert not any(i.isHidden() for i in _items(tree))
