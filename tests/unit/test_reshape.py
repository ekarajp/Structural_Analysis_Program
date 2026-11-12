"""Reshape (Edit > Reshape): which joints a grip drag moves, which members follow,
what is split off, merged, kept and reported.  One bay 6 x 5 x 3 with its floor."""
import json

import numpy as np
import pytest

from structstudio.app import reshape as R
from structstudio.app.document import Document
from structstudio.app.templates import new_building
from structstudio.model import ModelError

Z = 3.0


def _doc(floor=True):
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(Z,), grid_only=False)
    if not floor:
        for a in [a for a in m.areas if m.areas[a].type == "Floor"]:
            del m.areas[a]
    return Document(m)


def _pt(m, x, y, z):
    p = m.find_point(x, y, z)
    assert p is not None, (x, y, z)
    return p.name


def _beam(m, a, b):
    return next(f.name for f in m.frames.values() if {f.point_i, f.point_j} == {a, b})


def _len(m, f):
    f = m.frames[f]
    return float(np.linalg.norm(np.subtract(m.points[f.point_j].xyz, m.points[f.point_i].xyz)))


def _members_at(m, n):
    return sorted(f.name for f in m.frames.values() if n in (f.point_i, f.point_j)) + \
        sorted(a.name for a in m.areas.values() if n in a.points)


def test_one_joint_all_members_follow():
    d = _doc()
    m = d.model
    j = _pt(m, 6, 0, Z)
    users = _members_at(m, j)
    rs = R.reshape_set(m, [], [], [j])
    assert rs.mode == "joints"
    info = R.reshape(d, rs, {j: (7, 0, Z)})
    m = d.model
    assert m.points[j].xyz == (7, 0, Z)
    assert _members_at(m, j) == users  # every member followed
    assert info["stayed"] == []


def test_several_selected_joints_a_grip_moves_only_its_joint():
    d = _doc(floor=False)
    m = d.model
    right = [_pt(m, 6, 0, Z), _pt(m, 6, 5, Z)]
    rs = R.reshape_set(m, [], [], right)
    keys = [k for k, _ in R.grips(m, rs)]
    assert keys == [("joint", n) for n in right]
    assert R.grip_joints(m, rs, ("joint", right[0])) == [right[0]]  # not both: that is Move
    bx = _beam(m, _pt(m, 0, 0, Z), right[0])
    by = _beam(m, right[0], right[1])
    R.reshape(d, rs, {right[0]: (7, 0, Z)})
    m = d.model
    assert _len(m, bx) == pytest.approx(7.0)
    assert m.points[right[1]].xyz == (6, 5, Z)  # the other selected joint stayed
    assert _len(m, by) == pytest.approx(np.hypot(1, 5))


def test_selected_beam_only_detaches_from_column_and_others():
    d = _doc(floor=False)
    m = d.model
    a, b = _pt(m, 0, 0, Z), _pt(m, 6, 0, Z)
    beam = _beam(m, a, b)
    stay_before = [x for x in _members_at(m, b) if x != beam]
    rs = R.reshape_set(m, [beam], [], [])
    assert rs.mode == "elements"
    assert sorted(R.detached(m, rs, [b])[0]) == sorted(stay_before)
    info = R.reshape(d, rs, {b: (6.5, 0, Z)})
    m = d.model
    f = m.frames[beam]
    assert m.points[f.point_j if f.point_i == a else f.point_i].xyz == (6.5, 0, Z)
    assert _len(m, beam) == pytest.approx(6.5)
    assert sorted(info["stayed"]) == sorted(stay_before)
    q = m.find_point(6, 0, Z)
    assert q is not None and sorted(_members_at(m, q.name)) == sorted(stay_before)  # they kept a joint there


def test_mixed_joint_and_member_moves_only_selected():
    d = _doc(floor=False)
    m = d.model
    a, b = _pt(m, 0, 0, Z), _pt(m, 6, 0, Z)
    beam = _beam(m, a, b)
    rs = R.reshape_set(m, [beam], [], [b])
    assert rs.mode == "mixed"
    info = R.reshape(d, rs, {b: (6, -1, Z)})
    assert beam in _members_at(d.model, b)
    assert info["stayed"] and beam not in info["stayed"]


def test_floor_edge_grip_moves_square_to_edge():
    d = _doc()
    m = d.model
    floor = next(a.name for a in m.areas.values() if a.type == "Floor")
    rs = R.reshape_set(m, [], [floor], [])
    edges = [k for k, _ in R.grips(m, rs) if k[0] == "edge"]
    assert len(edges) == 4
    # the edge at x = 6
    a = m.areas[floor]
    k = next(k for k in edges if all(abs(m.points[a.points[i % len(a.points)]].x - 6) < 1e-9
                                     for i in (k[2], k[2] + 1)))
    dv = R.constrain(m, rs, k, (1.0, 0.7, 0.4), "plan")
    assert np.allclose(dv, (1.0, 0, 0))
    moves = {n: tuple(np.add(m.points[n].xyz, dv)) for n in R.grip_joints(m, rs, k)}
    R.reshape(d, rs, moves)
    m = d.model
    xs = sorted(m.points[p].x for p in m.areas[floor].points)
    assert xs == pytest.approx([0, 0, 7, 7])  # still a rectangle
    # the beams and columns at x = 6 stay (only the floor was selected)
    assert m.find_point(6, 0, Z) is not None


def test_plane_and_shift_lock():
    d = _doc(floor=False)
    m = d.model
    b = _pt(m, 6, 0, Z)
    rs = R.reshape_set(m, [], [], [b])
    assert np.allclose(R.constrain(m, rs, ("joint", b), (1, 2, 3), "plan"), (1, 2, 0))
    assert np.allclose(R.constrain(m, rs, ("joint", b), (1, 2, 3), "3d"), (1, 2, 0))
    assert np.allclose(R.constrain(m, rs, ("joint", b), (1, 2, 3), "xz"), (1, 0, 3))
    assert np.allclose(R.constrain(m, rs, ("joint", b), (1, 0.2, 0), "plan", lock=True), (1, 0, 0))
    beam = _beam(m, _pt(m, 0, 0, Z), b)
    rs2 = R.reshape_set(m, [beam], [], [])
    v = R.constrain(m, rs2, ("joint", b), (0.9, 0.3, 0), "plan", lock=True)
    assert np.allclose(v, (0.9, 0, 0))  # along the beam (= X here)


def test_drop_on_joint_merges_and_zero_length_refused():
    d = _doc(floor=False)
    m = d.model
    a, b, c = _pt(m, 0, 0, Z), _pt(m, 6, 0, Z), _pt(m, 6, 5, Z)
    beam = _beam(m, b, c)
    before = json.dumps(m.to_dict(), sort_keys=True)
    with pytest.raises(ModelError):  # its end on its other end
        R.reshape(d, R.reshape_set(m, [beam], [], []), {b: (6, 5, Z)})
    assert json.dumps(d.model.to_dict(), sort_keys=True) == before
    # a free beam end dropped on a joint joins it
    m = d.model
    bx = _beam(m, a, b)
    info = R.reshape(d, R.reshape_set(m, [bx], [], []), {b: (6, 5, Z)})
    m = d.model
    f = m.frames[bx]
    assert c in (f.point_i, f.point_j)
    assert info["merged"]


def test_drop_on_member_is_reported_connected():
    d = _doc(floor=False)
    m = d.model
    a, b = _pt(m, 0, 0, Z), _pt(m, 6, 0, Z)
    by = _beam(m, a, _pt(m, 0, 5, Z))
    bx = _beam(m, a, b)
    info = R.reshape(d, R.reshape_set(m, [bx], [], []), {b: (0, 3, Z)})
    assert info["on"] and set(info["on"].values()) == {by} or info["merged"]


def test_loads_and_real_axis_and_undo():
    d = _doc(floor=False)
    m = d.model
    a, b = _pt(m, 0, 0, Z), _pt(m, 6, 0, Z)
    bx = _beam(m, a, b)
    pat = next(iter(m.load_patterns))

    def add(model):
        model.add_frame_point_load(bx, pat, 10.0, rel=0.5)
        ld = model.frames[bx].loads[-1]
        ld.absolute, ld.rel_a, ld.rel_b = True, 3.0, 3.0
        model.frames[bx].real_axis = {"joints": [a, b], "ends": [a, b]}
        model.points[b].loads.append(type(model.points[b].loads[0])(pat) if model.points[b].loads else None) \
            if False else None
    d.edit(add)
    before = json.dumps(d.model.to_dict(), sort_keys=True)
    info = R.reshape(d, R.reshape_set(d.model, [], [], [b]), {b: (9, 0, Z)})
    f = d.model.frames[bx]
    assert f.loads[-1].rel_a == pytest.approx(4.5)  # the middle stays the middle
    assert f.real_axis is None and info["cleared"] == [bx]
    d.undo()
    assert json.dumps(d.model.to_dict(), sort_keys=True) == before


def test_opening_pushed_out_of_floor_warns():
    d = _doc()
    m = d.model
    floor = next(a.name for a in m.areas.values() if a.type == "Floor")

    def add(model):
        ps = [model.add_point(x, y, Z).name for x, y in ((2, 2), (3, 2), (3, 3), (2, 3))]
        model.add_area(ps, None, type="Opening")
    d.edit(add)
    m = d.model
    op = next(a.name for a in m.areas.values() if a.is_opening)
    assert R.openings_outside(m) == []
    rs = R.reshape_set(m, [], [op], [])
    moves = {n: tuple(np.add(m.points[n].xyz, (10, 0, 0))) for n in m.areas[op].points}
    info = R.reshape(d, rs, moves)
    assert info["openings"] == [op]
    assert any(lv == "warning" and op in t for lv, t in d.notes)
    del floor


def test_story_change_reported_in_elevation():
    d = _doc(floor=False)
    m = d.model
    b = _pt(m, 6, 0, Z)
    info = R.reshape(d, R.reshape_set(m, [], [], [b]), {b: (6, 1.0, 0.0)})
    assert info["stories"] and info["stories"][0][0] == b


def test_locked_model_asks():
    d = _doc(floor=False)
    d.locked = True
    asked = []
    d.ask_unlock = lambda: asked.append(1) or False
    b = _pt(d.model, 6, 0, Z)
    with pytest.raises(ModelError):
        R.reshape(d, R.reshape_set(d.model, [], [], [b]), {b: (7, 0, Z)})
    assert asked == [1]


# ================================================================ window (child)
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"
gui = pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")


@pytest.mark.skipif(CHILD, reason="parent only")
def test_reshape_window_cases_in_subprocess():
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


@pytest.fixture
def window(tmp_path, monkeypatch):
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtk")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    from structstudio.app.mainwindow import MainWindow
    w = MainWindow(_doc(floor=False), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.show()
    w.resize(1600, 950)
    for _ in range(6):
        app.processEvents()
    w._app = app
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    for _ in range(4):
        app.processEvents()


def _plan(w):
    vp = next(v for v in w.views.visible_viewports() if v.config.kind == "plan")
    if abs(vp.plane().value - Z) > 1e-9:
        vp.set_story("Story1") if hasattr(vp, "set_story") else None
    return vp


def _mouse(vp, xy, shift=False):
    vp.interactor = SimpleNamespace(GetEventPosition=lambda: xy, GetShiftKey=lambda: int(shift),
                                    GetControlKey=lambda: 0, GetRepeatCount=lambda: 0,
                                    GetKeySym=lambda: "")


def _px(vp, xyz):
    return tuple(int(round(c)) for c in vp._display(xyz))


@gui
def test_gui_edit_menu_and_drag_a_selected_beam_end(window):
    from structstudio.app.viewport import ModelViewport
    assert "tool.reshape" in window.menu_actions or window.tool_actions.get("reshape") is not None
    vp = _plan(window)
    st = window.state
    m = window.doc.model
    a, b = _pt(m, 0, 0, Z), _pt(m, 6, 0, Z)
    beam = _beam(m, a, b)
    st.set_selection([beam])
    st.set_tool("reshape")
    assert not st.tool.selecting
    keys = [k for k, _ in vp._reshape_grips()]
    assert ("joint", b) in keys
    real = vp.interactor
    try:
        vp.activate = lambda: None
        vp._cube_event = lambda *a: False
        start = _px(vp, (6, 0, Z))
        end = _px(vp, (7.5, 0, Z))
        _mouse(vp, start)
        ModelViewport._on_left_press(vp, None, None)
        assert st.tool.hot == ("joint", b)
        _mouse(vp, end)
        ModelViewport._on_mouse_move(vp, None, None)
        pv = st.tool.preview(vp._last_cursor)
        assert pv.detach  # the column and the other beam stay: dashed
        ModelViewport._on_left_release(vp, None, None)
    finally:
        vp.interactor = real
    m = window.doc.model
    assert _len(m, beam) == pytest.approx(7.5, abs=0.05)
    assert m.find_point(6, 0, Z) is not None  # the column top stayed
    assert st.tool.name == "reshape" and st.tool.hot is None  # on for the next drag
    window.doc.undo()
    assert _len(window.doc.model, beam) == pytest.approx(6.0)
    vp.escape()
    assert st.tool.name == "select"


@gui
def test_gui_typed_distance_while_a_grip_is_hot(window):
    from structstudio.app.viewport import ModelViewport
    vp = _plan(window)
    st = window.state
    m = window.doc.model
    b = _pt(m, 6, 0, Z)
    st.set_selection(points=[b])
    st.set_tool("reshape")
    real = vp.interactor
    try:
        vp.activate = lambda: None
        vp._cube_event = lambda *a: False
        _mouse(vp, _px(vp, (6, 0, Z)))
        ModelViewport._on_left_press(vp, None, None)
        ModelViewport._on_left_release(vp, None, None)  # a click: the grip stays hot
        assert st.tool.hot == ("joint", b)
        _mouse(vp, _px(vp, (8, 0, Z)))
        ModelViewport._on_mouse_move(vp, None, None)
        vp._typed = "1.25"
        vp._apply_typed()
    finally:
        vp.interactor = real
    assert window.doc.model.points[b].xyz == pytest.approx((7.25, 0, Z))
