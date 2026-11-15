"""Navigation frame scheduling (performance audit P01/P02, G Tower 20261115):
one pending frame per window, every camera step applied in order, labels hidden
for every kind of navigation and restored with a full frame at the end.

Runs in a child interpreter like test_app_window (Qt/VTK and the off-screen scene
tests do not share one process well)."""
import os
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("vtk")
QtCore = pytest.importorskip("PySide6.QtCore")
QtTest = pytest.importorskip("PySide6.QtTest")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from structstudio.app import navigation  # noqa: E402
from structstudio.app.document import Document  # noqa: E402
from structstudio.app.templates import new_building  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"
gui = pytest.mark.skipif(not CHILD, reason="runs in the child interpreter (test_nav_render_in_subprocess)")


@pytest.mark.skipif(CHILD, reason="parent only")
def test_nav_render_in_subprocess():
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


@pytest.fixture
def window(app):
    from structstudio.app.mainwindow import MainWindow
    doc = Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5)))
    w = MainWindow(doc)
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    app.processEvents()


def _v3d(window):
    return next(v for v in window.views.viewports if v.config.kind == "3d" and v.isVisible())


def _count_renders(vp):
    n = [0]
    vp.vtk_widget.GetRenderWindow().AddObserver("StartEvent", lambda *a: n.__setitem__(0, n[0] + 1))
    return n


def _camera(vp):
    c = vp.renderer.GetActiveCamera()
    return np.r_[c.GetPosition(), c.GetFocalPoint(), c.GetViewUp(), c.GetParallelScale(), c.GetClippingRange()]


def _set_camera(vp, state):
    c = vp.renderer.GetActiveCamera()
    c.SetPosition(*state[0:3])
    c.SetFocalPoint(*state[3:6])
    c.SetViewUp(*state[6:9])
    c.SetParallelScale(state[9])
    vp.renderer.ResetCameraClippingRange()
    vp.render()


def _many_labels(vp):
    import vtk
    pts, texts = vtk.vtkPoints(), vtk.vtkStringArray()
    texts.SetName("labels")
    for k in range(vp.DRAG_LABEL_LIMIT + 50):
        pts.InsertNextPoint(k * 0.01, 0.0, 0.0)
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
    return actor


@gui
def test_burst_draws_once_with_the_same_final_camera(window, app):
    vp = _v3d(window)
    start = _camera(vp)
    n = _count_renders(vp)
    # reference: every step drawn (the old behaviour)
    sync = vp.request_render
    vp.request_render = vp.render
    try:
        vp._start_nav("orbit", 300, 300)
        for i in range(120):
            vp._drag_nav(300 + i, 300 + i * .2)
        vp._end_nav()
    finally:
        del vp.request_render
    assert vp.request_render == sync
    ref, ref_frames = _camera(vp), n[0]
    assert ref_frames >= 120
    _set_camera(vp, start)
    n[0] = 0
    vp._start_nav("orbit", 300, 300)
    t = time.perf_counter()
    for i in range(120):
        vp._drag_nav(300 + i, 300 + i * .2)
    app.processEvents()
    drained = (time.perf_counter() - t) * 1000
    during = n[0]
    vp._end_nav()
    assert during <= 2, during  # not one frame per event
    assert not vp.frame_pending
    assert np.max(np.abs(_camera(vp) - ref)) <= 1e-9  # every camera step applied, in order
    print(f"burst: {ref_frames} frames synchronous, {n[0]} coalesced, drain {drained:.1f} ms")


@gui
def test_continuous_input_keeps_drawing(window, app):
    """10 s of a drag is too long for a unit test: 1.5 s of moves every 2 ms must keep
    frames coming (no wait for the end of the drag), and a flooded loop still draws."""
    vp = _v3d(window)
    n = _count_renders(vp)
    vp._start_nav("orbit", 300, 300)
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < 1.5:
        i += 1
        vp._drag_nav(300 + i % 200, 300)
        app.processEvents()
        time.sleep(0.002)
    assert n[0] >= 1.5 * 1000 / (3 * vp.FRAME_BUDGET_MS), n[0]  # at least a frame every ~3 budgets
    # no event processing at all for 0.3 s: the starvation guard draws anyway
    n[0] = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3:
        i += 1
        vp._drag_nav(300 + i % 200, 310)
    assert n[0] >= 3, n[0]
    vp._end_nav()
    assert not vp.frame_pending and not vp.interacting


@gui
def test_viewcube_drag_hides_labels_and_restores_them(window, app):
    vp = _v3d(window)
    actor = _many_labels(vp)
    vp.render()
    x0, y0, x1, y1 = vp.viewcube.ren.GetViewport()
    w, h = vp.vtk_widget.GetRenderWindow().GetSize()
    cx, cy = int((x0 + x1) / 2 * w), int((y0 + y1) / 2 * h)
    assert vp._cube_event("press", cx, cy)
    assert vp._cube_event("move", cx + 20, cy + 5)
    assert not actor.GetVisibility() and vp.interacting
    for k in range(10):
        vp._cube_event("move", cx + 25 + k, cy + 5)
    assert vp._cube_event("release", cx + 35, cy + 5)
    assert actor.GetVisibility() and not vp.interacting and not vp.frame_pending
    vp.renderer.RemoveViewProp(actor)


@gui
def test_camera_animation_hides_labels_and_lands_exactly(window, app):
    from structstudio.app.viewcube import view_for
    vp = _v3d(window)
    actor = _many_labels(vp)
    vp.view_from((0, -1, 0), animate=True)
    assert not actor.GetVisibility()
    t0 = time.time()
    while vp.camera_animator.running and time.time() - t0 < 3:
        app.processEvents()
    assert actor.GetVisibility() and not vp.interacting and not vp.frame_pending
    d, _ = view_for((0, -1, 0))
    assert np.allclose(vp.renderer.GetActiveCamera().GetDirectionOfProjection(), -np.asarray(d), atol=1e-9)
    vp.renderer.RemoveViewProp(actor)


@gui
def test_keys_and_wheel_restore_after_a_pause(window, app, monkeypatch):
    vp = _v3d(window)
    actor = _many_labels(vp)
    monkeypatch.setattr(vp, "WHEEL_LABEL_IDLE_MS", 20)
    vp.rotate_view(15, 0)
    vp.zoom(1.25)
    assert not actor.GetVisibility()
    QtTest.QTest.qWait(150)
    assert actor.GetVisibility() and not vp.interacting and not vp.frame_pending
    vp.renderer.RemoveViewProp(actor)


@gui
def test_escape_and_picture_give_the_full_view(window, app, tmp_path):
    from structstudio.app.capture import vtk_image
    vp = _v3d(window)
    actor = _many_labels(vp)
    vp._start_nav("orbit", 300, 300)
    vp._drag_nav(320, 300)
    assert not actor.GetVisibility()
    img = vtk_image(vp)
    assert img is not None and actor.GetVisibility() and not vp.frame_pending
    vp._drag_nav(330, 300)  # still dragging: labels stay as they are now (shown)
    vp._end_nav()
    vp.pulse_interaction("wheel")
    assert not actor.GetVisibility()
    vp.screenshot(str(tmp_path / "v.png"))
    assert actor.GetVisibility() and (tmp_path / "v.png").stat().st_size > 0
    vp.renderer.RemoveViewProp(actor)


@gui
def test_pick_after_navigation_matches_the_camera(window, app):
    """World-to-screen after a coalesced drag is the camera's (the projection does
    not wait for the frame)."""
    vp = _v3d(window)
    vp._start_nav("orbit", 300, 300)
    for i in range(30):
        vp._drag_nav(300 + 3 * i, 300)
    ren = vp.renderer
    ren.SetWorldPoint(6.0, 5.0, 7.0, 1.0)
    ren.WorldToDisplay()
    before_flush = ren.GetDisplayPoint()
    vp._end_nav()
    ren.SetWorldPoint(6.0, 5.0, 7.0, 1.0)
    ren.WorldToDisplay()
    assert np.allclose(before_flush, ren.GetDisplayPoint(), atol=1e-9)


@gui
def test_close_with_a_frame_pending(app):
    from structstudio.app.mainwindow import MainWindow
    doc = Document(new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.5,)))
    w = MainWindow(doc)
    w.show()
    app.processEvents()
    vp = _v3d(w)
    vp.pulse_interaction("wheel")
    vp.request_render()
    vp.view_from((1, -1, 1), animate=True)
    assert vp.frame_pending
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    QtTest.QTest.qWait(100)  # the timers fire (or not) after the GL context is gone: no crash
    app.processEvents()
    assert not vp.frame_pending


def test_animator_without_qt_runs_hooks():
    """CameraAnimator without a Qt application: one step, hooks still called."""
    import vtk
    from structstudio.app.viewcube import CameraAnimator
    ren = vtk.vtkRenderer()
    calls = []
    a = CameraAnimator(ren, lambda: calls.append("frame"), on_start=lambda: calls.append("start"),
                       on_finish=lambda: calls.append("finish"))
    if QtCore.QCoreApplication.instance() is not None:
        pytest.skip("a Qt application exists in this process")
    a.go((1, 0, 0), (0, 0, 1), animate=True)
    assert calls == ["start", "frame", "finish"] and not a.running
    assert navigation is not None


@gui
def test_hidden_window_builds_its_scene_when_first_shown(app, tmp_path):
    """Performance audit P03: a window kept in the layout but not on screen builds no
    scene at start-up; when opened it shows the model as it is then."""
    from structstudio.app.mainwindow import MainWindow
    from structstudio.app import scene
    settings = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat)
    doc = Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5)))
    calls = []
    real = scene.build_model_actors

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    scene.build_model_actors = spy
    try:
        w = MainWindow(doc, settings)
        w.resize(1400, 900)
        w.show()
        app.processEvents()
        hidden = [(i, v) for i, v in enumerate(w.views.viewports) if not v.isVisible()]
        assert hidden, "the default layout keeps hidden windows"
        assert all(v._dirty >= {"scene"} and not v._scene_actors for _, v in hidden)
        doc.edit(lambda m: m.add_point(20.0, 20.0, 0.0))  # while hidden: marked only
        app.processEvents()
        assert all(not v._scene_actors for _, v in hidden)
        n_before_show = len(calls)
        i, v = hidden[0]
        w.views.docks[i].show()
        QtTest.QTest.qWait(200)
        assert v.isVisible() and v._scene_actors and not v._dirty
        assert len(calls) > n_before_show  # built from the model as it is now (after the edit)
        assert all(not u._scene_actors for _, u in hidden[1:] if not u.isVisible())
    finally:
        scene.build_model_actors = real
        w.doc.dirty = False
        w.views.finalize()
        w.close()
        w.deleteLater()
        app.processEvents()


@pytest.fixture
def bwindow(app):
    """A window on a building with frames, slabs and core walls."""
    from structstudio.app.mainwindow import MainWindow
    from structstudio.app.templates import build_model
    doc = Document(build_model(structure="core_frame", x_ordinates=(0, 6, 12, 19), y_ordinates=(0, 5, 10.5, 15),
                               stories=[(f"S{k}", 3.5) for k in range(1, 4)]))
    w = MainWindow(doc)
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    for v in w.views.viewports:
        if v.isVisible():
            v.fit()
    app.processEvents()
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    app.processEvents()


@gui
def test_snap_broad_phase_finds_the_same_hits(bwindow, app):
    window = bwindow
    """G Tower audit D1: with the window's cache and many-points projection the snap engine
    tests only the targets near the cursor; every list of hits must be the one of the
    plain engine (same hits, same order, same distances)."""
    import dataclasses
    st = window.state
    window.tool_actions["beam"].trigger()
    app.processEvents()
    snap0 = dataclasses.replace(st.snap) if dataclasses.is_dataclass(st.snap) else None
    checked = with_hits = 0
    try:
        for variant in ("default", "all", "previous"):
            if variant != "default":
                st.snap.select_all(True)
                st.snap.osnap = True
            for vp in [v for v in window.views.viewports if v.isVisible()]:
                w, h = vp.vtk_widget.GetRenderWindow().GetSize()
                if variant == "previous":
                    p = next(iter(window.doc.model.points.values()))
                    st.tool.points = [tuple(p.xyz)]
                for i in range(9):
                    for j in range(9):
                        x, y = int(w * (0.1 + 0.1 * i)) + (i * 7) % 5, int(h * (0.1 + 0.1 * j)) + (j * 3) % 4
                        fast = vp._snap_engine(x, y)
                        assert fast.project_many is not None and fast.cache is not None
                        plain = vp._snap_engine(x, y)
                        plain.project_many, plain.cache = None, None
                        a, b = fast.find(x, y), plain.find(x, y)
                        assert a == b, (variant, vp.config.kind, x, y)
                        checked += 1
                        with_hits += bool(a)
                st.tool.points = []
    finally:
        st.tool.points = []
        if snap0 is not None:
            for f in dataclasses.fields(snap0):
                setattr(st.snap, f.name, getattr(snap0, f.name))
        st.cancel_tool()
    assert checked >= 400 and with_hits >= 100, (checked, with_hits)


@gui
def test_snap_cache_follows_the_model(bwindow, app):
    window = bwindow
    vp = next(v for v in window.views.viewports if v.config.kind == "plan" and v.isVisible())
    window.tool_actions["beam"].trigger()
    app.processEvents()
    z = vp.plane().value
    e1 = vp._snap_engine(10, 10)
    n1 = len(e1._static_points("before"))
    e1.find(10, 10)
    assert vp._snap_engine(10, 10).cache is e1.cache  # nothing changed: the same lists
    window.doc.edit(lambda m: m.add_point(2.345, 1.234, z))
    e2 = vp._snap_engine(10, 10)
    assert e2.cache is not e1.cache
    assert len(e2._static_points("before")) == n1 + 1
    x, y = vp._display((2.345, 1.234, z))
    hits = e2.find(x, y)
    assert hits and hits[0].kind == "joint" and np.allclose(hits[0].xyz, (2.345, 1.234, z))
    window.state.cancel_tool()
