"""Edit > Add to Model From Template > 2D Frames / 3D Structures."""
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from structstudio.app.templates import (FRAME_2D_KEYS, add_plane_frame, add_structure, new_building,
                                        parse_spacings)
from structstudio.model.checks import geometry_issues


def _bldg():
    return new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.0, 3.0), grid_only=False)


def _xyz(m, name):
    p = m.points[name]
    return np.array([p.x, p.y, p.z])


def test_parse_spacings():
    assert parse_spacings("6, 6 5") == [6, 6, 5]
    assert parse_spacings("3@6 4.5") == [6, 6, 6, 4.5]
    assert parse_spacings("2@7.5;1") == [7.5, 7.5, 1]
    for bad in ("", "0", "-3", "1.5@6", "x"):
        with pytest.raises(ValueError):
            parse_spacings(bad)


def test_3d_next_to_the_building_shares_joints():
    m = _bldg()
    n_pts, n_fr, n_ar = len(m.points), len(m.frames), len(m.areas)
    made = add_structure(m, "rc_frame", [6], [5], "Base", "Story3", origin=(12, 0), support="Fixed")
    # one bay x one bay x 3 stories: 12 columns, 12 beams, 3 slabs - the 6 columns and 3
    # beams on x = 12 are the building's already
    assert made.skipped == 9
    assert len(m.frames) - n_fr == 15 and len(m.areas) - n_ar == 3
    assert len(m.points) - n_pts == 8  # the new column line x = 18
    assert not [i for i in geometry_issues(m) if "duplicate" in i.message.lower()]
    base = [p for p in m.points.values() if p.z == 0 and p.x == 18]
    assert len(base) == 2 and all(all(p.restraints) for p in base)
    # the new slabs join the diaphragm of the floors of their story
    for a in made:
        if getattr(a, "type", "") == "Floor":
            other = next(b for b in m.areas.values() if b.story == a.story and b.name != a.name and b.type == "Floor")
            assert a.diaphragm == other.diaphragm


def test_3d_rotated_and_from_an_upper_story():
    m = _bldg()
    made = add_structure(m, "flat_slab", [4], [3], "Story1", "Story3", origin=(30, 0), angle=90.0)
    cols = [f for f in made if f.type == "Column"]
    assert len(cols) == 8 and all(f.angle == pytest.approx(90.0) for f in cols)
    xy = {(round(m.points[f.point_i].x, 6), round(m.points[f.point_i].y, 6)) for f in cols}
    assert xy == {(30, 0), (30, 4), (27, 0), (27, 4)}  # local x -> global +Y, local y -> -X
    zs = {round(_xyz(m, f.point_i)[2], 6) for f in cols} | {round(_xyz(m, f.point_j)[2], 6) for f in cols}
    assert zs == {3.5, 6.5, 9.5}
    assert not any(any(m.points[f.point_i].restraints) for f in cols)  # support None above the base


@pytest.mark.parametrize("kind", FRAME_2D_KEYS)
def test_2d_frames(kind):
    m = _bldg()
    n_fr = len(m.frames)
    made = add_plane_frame(m, kind, [6, 6, 6], "Base", "Story3", origin=(0, 20), angle=0.0,
                           column="C40x40" if "C40x40" in m.frame_sections else None,
                           braced_bays=[2], link=1.0, support="Fixed")
    assert made.skipped == 0 and len(m.frames) - n_fr == len(made)
    cols = [f for f in made if f.type == "Column"]
    braces = [f for f in made if f.type == "Brace"]
    beams = [f for f in made if f.type == "Beam"]
    assert len(cols) == 4 * 3
    n_br = {"moment": 0, "x_brace": 6, "diagonal": 3, "chevron": 6, "v_brace": 6, "eccentric": 6}[kind]
    assert len(braces) == n_br
    assert all(f.releases_i[4] and f.releases_i[5] and f.releases_j[4] and f.releases_j[5] for f in braces)
    assert all(abs(m.points[q].y - 20) < 1e-9 for f in made for q in (f.point_i, f.point_j))
    # the braces are in bay 2 (x 6..12)
    for f in braces:
        for q in (f.point_i, f.point_j):
            assert 6 - 1e-9 <= m.points[q].x <= 12 + 1e-9
    # beams split where braces meet them
    extra = {"moment": 0, "x_brace": 0, "diagonal": 0, "chevron": 3, "v_brace": 2, "eccentric": 6}[kind]
    assert len(beams) == 9 + extra
    if kind == "eccentric":
        L = sorted(round(np.linalg.norm(_xyz(m, f.point_j) - _xyz(m, f.point_i)), 6) for f in beams)
        assert L.count(1.0) == 3  # the links
    if kind == "diagonal":  # alternating
        dirs = sorted(round(np.sign(_xyz(m, f.point_j)[0] - _xyz(m, f.point_i)[0]), 0) for f in braces)
        assert dirs == [-1, 1, 1]
    base = [p for p in m.points.values() if p.y == 20 and p.z == 0]
    assert base and all(all(p.restraints) for p in base)
    assert not [i for i in geometry_issues(m) if "duplicate" in i.message.lower()]


def test_2d_frame_turned_and_analysed():
    sys.path.insert(0, os.path.dirname(__file__))
    from test_stiffness_sets import _run
    m = _bldg()
    made = add_plane_frame(m, "x_brace", [5], "Base", "Story3", origin=(0, 5), angle=-90.0, braced_bays=[1],
                           support="Fixed")
    # along -Y from (0, 5): onto the building's column line x = 0 -> columns and beams exist
    assert made.skipped == 9 and len(made) == 6 and all(f.type == "Brace" for f in made)
    m.add_load_pattern("EQY", "Seismic")
    top = next(p for p in m.points.values() if p.x == 0 and p.y == 0 and p.z > 9)
    m.add_point_load(top.name, "EQY", force=(0, 100.0, 0))
    m.add_linear_static_case("EQY", {"EQY": 1.0})
    r = _run(m)
    assert r.base_reactions("EQY")[1] == pytest.approx(-100.0, rel=1e-6)


def test_errors():
    m = _bldg()
    with pytest.raises(ValueError):
        add_structure(m, "rc_frame", [6], [5], "Story3", "Story1")
    with pytest.raises(ValueError):
        add_structure(m, "grid", [6], [5], "Base", "Story1")
    with pytest.raises(ValueError):
        add_plane_frame(m, "chevron", [6, 6], "Base", "Story1", braced_bays=[3])
    with pytest.raises(ValueError):
        add_plane_frame(m, "eccentric", [6], "Base", "Story1", link=6.0)


# ------------------------------------------------------------- the window
CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"


@pytest.mark.skipif(CHILD, reason="parent only")
def test_template_window_in_subprocess():
    pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtk")
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


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_template_dialogs(tmp_path, monkeypatch):
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    from structstudio.app.document import Document
    from structstudio.app.mainwindow import MainWindow
    from structstudio.app.template_dialog import TemplateDialog
    m = _bldg()
    w = MainWindow(Document(m), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.show()
    app.processEvents()
    cmds = w.menu_commands()
    assert "edit.template_2d" in cmds and "edit.template_3d" in cmds
    act = w.menu_actions.get("edit.template_2d") if isinstance(getattr(w, "menu_actions", None), dict) else None
    if act is not None:
        assert act.isEnabled()
    n = len(w.doc.model.frames)
    d = TemplateDialog(w, plane=True)
    d.kind.setCurrentIndex(d.kind.findData("chevron"))
    d.xsp.setText("2@6")
    d.oy.setText("20")
    d.sketch.repaint()
    assert d.sketch.model is not None and not d.sketch.error
    d.xsp.setText("0")
    assert d.sketch.error
    d.xsp.setText("2@6")
    assert d.support.currentText() == "Fixed"
    d.add()
    assert d.added and len(w.doc.model.frames) == n + len(d.added)
    assert set(w.state.frames) == {o.name for o in d.added}
    w.doc.undo()
    assert len(w.doc.model.frames) == n
    d3 = TemplateDialog(w, plane=False)
    d3.kind.setCurrentIndex(d3.kind.findData("core_frame"))
    d3.xsp.setText("3@6")
    d3.ysp.setText("3@6")
    d3.ox.setText("40")
    d3.bottom.setCurrentIndex(1)
    assert d3.support.currentText() == "None"
    d3.sketch.repaint()
    assert d3.sketch.model is not None and d3.sketch.model.areas
    d3.add()
    assert d3.added and any(getattr(o, "type", "") == "Wall" for o in d3.added)
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    for _ in range(3):
        app.processEvents()
