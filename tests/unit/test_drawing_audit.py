"""Regression tests for the drawing / snap audit of 2026-11-08 (D01 ... D17).

Each test names the finding it guards.  The view adapter calls the real, unmodified
ModelViewport geometry and snap methods on an off-screen VTK renderer (no window).
GUI click workflows (QTest mouse events in a real MainWindow) are in
test_app_window.py (test_drawing_audit_* tests)."""
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from structstudio.app.document import Document  # noqa: E402
from structstudio.app.snap import SnapEngine, SnapOptions  # noqa: E402
from structstudio.app.state import ViewState  # noqa: E402
from structstudio.app.templates import new_building  # noqa: E402
from structstudio.app.tools import (TOOLS, BeamTool, BraceTool, ColumnTool, DrawSettings, Plane,  # noqa: E402
                                    PointTool, QuickBraceTool, QuickSlabTool, RectSlabTool, SlabTool,
                                    WallTool, arc_polyline, arc_problem)
from structstudio.app.viewport import ModelViewport, ViewConfig  # noqa: E402
from structstudio.model import ModelError  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    import os
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def base():
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3, 3))
    m.points.clear()
    m.frames.clear()
    m.areas.clear()
    m.grid_lines.clear()
    return m


def opts(**kw):
    o = SnapOptions()
    o.select_all(False)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


class View:
    """The real viewport geometry / snap methods on an off-screen renderer."""
    PICK_PIXELS = ModelViewport.PICK_PIXELS
    for _n in ("_sync_aspect", "_display", "_ray", "_plane_point", "_snap_engine", "_snapped", "_world_tolerance",
               "_scene_plane", "visible", "_update_preview"):
        locals()[_n] = getattr(ModelViewport, _n)
    del _n

    def __init__(self, m, kind="3d", state=None, perspective=True, dpr=1.0):
        self.doc = Document(m) if state is None else state.doc
        self.state = state or ViewState(self.doc)
        self.config = ViewConfig(kind=kind)
        self.renderer = vtk.vtkRenderer()
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetSize(1000, 800)
        self.window.AddRenderer(self.renderer)
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(12, -24, 16)
        cam.SetFocalPoint(4, 8, 4)
        cam.SetViewUp(0, 0, 1)
        cam.SetViewAngle(40)
        cam.SetClippingRange(.1, 200)
        cam.SetParallelProjection(not perspective)
        cam.SetParallelScale(16)
        self.vtk_widget = SimpleNamespace(height=lambda: 800 / dpr, devicePixelRatioF=lambda: dpr)
        self._snap_index = 0
        self._snap_hits = []
        self._last_cursor = None
        self.hidden = (set(), set(), set())
        self.display_options = SimpleNamespace(reference_lines=False)
        self.drawn = None

    def plane(self):
        if self.config.kind == "3d":
            cam = self.renderer.GetActiveCamera()
            return Plane("3d", normal=tuple(np.subtract(cam.GetPosition(), cam.GetFocalPoint())))
        return Plane(self.config.kind, 0.0)

    def _clear_preview(self):
        self.drawn = None

    def render(self):
        pass

    def _draw_preview(self, pv, snap, x, y):
        self.drawn = (pv, snap)


def on_ray(v, x, y, depth):
    a, d = v._ray(x, y)
    cam = v.renderer.GetActiveCamera()
    eye = np.array(cam.GetPosition())
    dop = np.array(cam.GetDirectionOfProjection())
    return a + (depth - np.dot(a - eye, dop)) / np.dot(d, dop) * d


def err_px(v, xyz, cursor):
    return float(np.linalg.norm(np.subtract(v._display(xyz), cursor)))


# ----------------------------------------------------------------------- D01
@pytest.mark.parametrize("perspective", [True, False])
def test_d01_line_snaps_land_under_the_cursor_in_perspective(perspective):
    m = base()
    a, b = np.array([0., 0, 0]), np.array([8., 18, 9])
    m.add_frame(m.add_point(*a), m.add_point(*b), "B30x60")
    v = View(m, perspective=perspective)
    for t in (0.1, 0.25, 0.5, 0.73, 0.9):
        target = a + t * (b - a)
        cursor = v._display(target)
        eng = SnapEngine(m, Plane("3d"), opts(lines=True), v._display, lambda x, y: None)
        h = eng.find(*cursor)[0]
        p = np.asarray(h.xyz)
        # on the 3D line, and projecting back onto the cursor
        assert np.linalg.norm(np.cross(p - a, b - a)) / np.linalg.norm(b - a) < 1e-9
        assert err_px(v, p, cursor) < 1e-6
        assert np.linalg.norm(p - target) < 1e-6
        assert abs(h.distance - err_px(v, p, cursor)) < 1e-9  # the reported distance is the real one
    # extension beyond the end (on by default)
    target = a + 1.4 * (b - a)
    eng = SnapEngine(m, Plane("3d"), SnapOptions(), v._display, lambda x, y: None)
    ext = next(h for h in eng.find(*v._display(target)) if h.kind == "extension")
    assert np.linalg.norm(np.subtract(ext.xyz, target)) < 1e-6


def test_d01_click_handler_commits_the_point_under_the_cursor():
    """P17 path: the logical left-press handler of the viewport."""
    m = base()
    m.add_frame(m.add_point(0, 0, 0), m.add_point(8, 18, 9), "B30x60")
    v = View(m)
    v.state.set_tool("beam")
    v.state.snap.select_all(False)
    v.state.snap.lines = True
    target = np.array((4., 9., 4.5))
    cursor = tuple(int(round(c)) for c in v._display(target))
    v.interactor = SimpleNamespace(GetEventPosition=lambda: cursor, GetShiftKey=lambda: 0, GetControlKey=lambda: 0,
                                   GetRepeatCount=lambda: 0)
    v.activate = lambda: None
    v._cube_event = lambda *a: False
    v._check_pending_panel = lambda: None
    v.DIRECT_EDIT = False
    st = v.state
    st.tool.set_plane(v.plane())
    st.tool.click((1, 2, 0))
    ModelViewport._on_left_press(v, None, None)
    end = m.points[m.frames[st.tool.created[-1]].point_j].xyz
    assert err_px(v, end, cursor) < 1.0  # integer cursor pixel: within one pixel
    assert np.linalg.norm(np.subtract(end, target)) < 0.01


# ----------------------------------------------------------------------- D02
@pytest.mark.parametrize("kind", ["xz", "3d"])
def test_d02_far_perpendicular_foot_is_not_taken_silently(kind):
    m = base()
    m.add_frame(m.add_point(10, 0, 0), m.add_point(20, 0, 20), "B30x60")
    m.add_point(0, 0, 15)
    proj = lambda p: (p[0] * 10, p[2] * 10)  # noqa: E731
    eng = SnapEngine(m, Plane(kind), SnapOptions(), proj, lambda x, y: (x / 10, 0, y / 10), previous=(0, 0, 15))
    hits = eng.find(165., 130.)
    assert all(h.distance <= eng.tolerance for h in hits)
    assert not any(h.kind == "perpendicular" for h in hits)  # the foot (14,0,8) is 56 px away
    # the explicit Perpendicular snap (Shift + right-click menu) still finds it, with its guide line
    eng.override = "perpendicular"
    perp = eng.find(165., 130.)[0]
    assert perp.kind == "perpendicular" and perp.xyz == pytest.approx((14, 0, 8)) and perp.track is not None
    # a foot inside the aperture is still a normal snap
    eng.override = None
    near = [h for h in eng.find(*proj((14.2, 0, 8.4))) if h.kind == "perpendicular"]
    assert near and near[0].xyz == pytest.approx((14, 0, 8))


# ----------------------------------------------------------------------- D03
def test_d03_hidden_objects_and_points_behind_the_eye_are_no_targets():
    m = base()
    f = m.add_frame(m.add_point(0, 0, 0), m.add_point(5, 0, 3), "B30x60")
    v = View(m)
    v.state.snap = opts(joints=True, lines=True)
    cursor = v._display(m.points[f.point_i].xyz)
    assert v._snapped(*cursor)[1].target == f.point_i  # control: visible
    v.hidden = ({f.name}, set(), {f.point_i, f.point_j})
    xyz, h = v._snapped(*cursor)
    assert h is None or h.target not in (f.name, f.point_i, f.point_j)
    # behind the eye: mirrored onto the screen by the perspective, never a target
    m2 = base()
    v2 = View(m2)
    v2.state.snap = opts(joints=True)
    back = m2.add_point(*on_ray(v2, 500, 400, 20), name="BACK")
    m2.add_point(*on_ray(v2, 500, 400, -5), name="BEHIND")
    xyz, h = v2._snapped(500, 400)
    assert h.target == back.name and all(x.target != "BEHIND" for x in v2._snap_hits)


def test_d03_object_types_switched_off_in_the_window_are_no_targets():
    from structstudio.app.display import DisplayOptions
    m = base()
    f = m.add_frame(m.add_point(0, 0, 3), m.add_point(6, 0, 3), "B30x60", type="Beam")
    v = View(m)
    v.state.snap = opts(lines=True)
    cursor = v._display((3, 0, 3))
    assert v._snapped(*cursor)[1].target == f.name
    o = DisplayOptions()
    o.beams = False
    o.reference_lines = False
    v.display_options = o
    xyz, h = v._snapped(*cursor)
    assert h is None or h.target != f.name


# ----------------------------------------------------------------------- D04
def test_d04_exact_point_beats_one_in_front_and_n_choice_is_kept():
    m = base()
    v = View(m)
    v.state.snap = opts(joints=True)
    back = m.add_point(*on_ray(v, 500, 400, 20), name="BACK")
    m.add_point(*on_ray(v, 500.8, 400, 5), name="FRONT")
    assert v._snapped(500, 400)[1].target == back.name  # nearest on the screen wins
    # truly on the same pixel (< 0.5 px): the one in front
    m.add_point(*on_ray(v, 500.2, 400, 3), name="FRONT2")
    assert v._snapped(500, 400)[1].target == "FRONT2"
    # N chooses another candidate: it stays chosen while it is under the cursor
    v.state.set_tool("beam")
    v._snap_index = 1
    v._update_preview(500, 400, keep_index=True)
    chosen = v._snap_hits[v._snap_index].target
    for x in (500, 501, 499):
        v._update_preview(x, 400)
        assert v._snap_hits[v._snap_index].target == chosen
    v._update_preview(800, 100)  # left the aperture: the choice is forgotten
    v._update_preview(500, 400)
    assert v._snap_index == 0


# ----------------------------------------------------------------------- D05
def test_d05_a_point_shown_in_the_elevation_is_snapped_at_its_own_xyz():
    from structstudio.app import scene
    m = base()
    m.options.merge_tolerance = .001
    p = m.add_point(3, .005, 2, name="OFF5MM")
    assert p.name in scene.visible_sets(m, plane=("y", 0.0))[2]
    eng = SnapEngine(m, Plane("xz", 0), opts(joints=True), lambda q: (q[0] * 100, q[2] * 100),
                     lambda x, y: (x / 100, 0, y / 100), world_tol=.001)
    h = eng.find(300, 200)[0]
    assert h.target == p.name and h.xyz == pytest.approx((3, .005, 2))  # connected, not a new joint
    far = m.add_point(4, .05, 2, name="OFF50MM")  # outside the band: neither shown nor snapped
    assert far.name not in scene.visible_sets(m, plane=("y", 0.0))[2]
    assert not [h for h in eng.find(400, 200) if h.target == far.name]


# ----------------------------------------------------------------------- D06
def test_d06_hovering_another_window_keeps_the_first_point():
    m = base()
    doc = Document(m)
    state = ViewState(doc)
    state.set_tool("beam")
    state.snap = opts()
    tool = state.tool
    tool.set_plane(Plane("3d"))
    tool.click((0, 0, 0))
    View(m, "xz", state)._snapped(500, 400)
    assert tool.points == [(0.0, 0.0, 0.0)] and tool.plane.kind == "3d"
    # clicking in the other window continues the straight chain (its points are 3D)
    tool.set_plane(Plane("xz", 0.0))
    assert tool.points == [(0.0, 0.0, 0.0)]
    # an arc (planar construction) starts again when the plane changes
    state.settings.line_shape = "Arc: 3 points"
    tool.cancel()
    tool.set_plane(Plane("3d"))
    tool.click((0, 0, 0))
    tool.set_plane(Plane("xz", 0.0))
    assert tool.points == []


# ----------------------------------------------------------------------- D07
@pytest.mark.parametrize("ctl", ["None", "Parallel to X"])
def test_d07_preview_is_what_is_committed_with_offsets(ctl):
    doc = Document(base())
    t = BeamTool(doc, DrawSettings(frame_section="B30x60", offset_z=1, drawing_control=ctl))
    t.set_story("Story1")
    t.click((0, 0, 3))
    pv = t.preview((5, 2, 3))
    t.click((5, 2, 3))
    f = doc.model.frames[t.created[-1]]
    a, b = doc.model.points[f.point_i].xyz, doc.model.points[f.point_j].xyz
    assert pv.lines[-1][1] == pytest.approx(b) and pv.lines[-1][0] == pytest.approx(a)
    assert a[2] == pytest.approx(4.0) and b[2] == pytest.approx(4.0)  # offset once, not twice
    if ctl == "Parallel to X":
        assert b[1] - a[1] == pytest.approx(0) and b[2] - a[2] == pytest.approx(0)


def test_d07_offsets_of_another_tool_do_not_move_a_beam():
    doc = Document(base())
    s = DrawSettings(frame_section="B30x60", offset_x=1, offset_y=.5)  # set in the Draw Joints form
    t = BeamTool(doc, s)  # the beam form shows no plan offsets
    t.set_story("Story1")
    t.click((0, 0, 3))
    pv = t.preview((5, 2, 3))
    t.click((5, 2, 3))
    f = doc.model.frames[t.created[-1]]
    assert doc.model.points[f.point_j].xyz == pytest.approx((5, 2, 3)) == pv.lines[-1][1]
    p = PointTool(doc, s)  # the joint form shows them: they apply there
    p.set_story("Story1")
    p.click((0, 0, 3))
    assert doc.model.points[p.created[-1]].xyz == pytest.approx((1, .5, 3))


# ----------------------------------------------------------------------- D08
def test_d08_wall_preview_is_the_wall_and_cross_level_walls_span_the_stories():
    doc = Document(base())
    t = WallTool(doc, DrawSettings(wall_section="W20", drawing_control="Parallel to X"))  # left over from beams
    t.set_story("Story1")
    t.click((0, 0, 3))
    pv = t.preview((5, 2, 3))
    t.click((5, 2, 3))
    corners = [doc.model.points[p].xyz for p in doc.model.areas[t.created[-1]].points]
    drawn = {tuple(map(float, q)) for seg in pv.lines for q in seg}
    assert {tuple(c) for c in corners} <= drawn  # every corner of the wall was in the preview
    assert any(c[1] == 2 for c in corners)  # no drawing control on walls (their form has none)
    doc2 = Document(base())
    t2 = WallTool(doc2, DrawSettings(wall_section="W20"))
    t2.set_plane(Plane("3d"))
    t2.click((0, 0, 0))
    t2.click((4, 0, 6))
    zs = sorted({doc2.model.points[p].z for a in doc2.model.areas.values() for p in a.points})
    assert zs == [0.0, 3.0, 6.0] and len(doc2.model.areas) == 2
    t2.cancel()
    t2.click((0, 0, 0))
    with pytest.raises(ModelError):
        t2.click((4, 0, 4.5))  # not a story level


# ----------------------------------------------------------------------- D09
def test_d09_a_typed_length_never_uses_an_old_cursor():
    m = base()
    m.add_point(0, 0, 0)
    v = View(m)
    v.state.set_tool("beam")
    v.state.snap = opts(joints=True)
    t = v.state.tool
    t.set_plane(Plane("3d"))
    t.click((0, 0, 0))
    v._last_cursor = (4, 0, 0)
    v._update_preview(999, 799)  # nothing under the cursor (3D, snap only)
    assert v._last_cursor is None
    with pytest.raises(ValueError):
        t.typed("5", v._last_cursor)
    # with a work plane the cursor gives a direction - the current one
    t.settings.work_plane_3d = "XY through the last point"
    x, y = v._display((0, 6, 0))
    v._update_preview(x, y)
    assert v._last_cursor is not None
    assert t.typed("5", v._last_cursor) == pytest.approx((0, 5, 0), abs=1e-6)


# ----------------------------------------------------------------------- D10
def test_d10_increments_keep_line_snaps_on_their_line():
    m = base()
    m.add_frame(m.add_point(0, .2, 0), m.add_point(6, .2, 0), "B30x60")
    o = opts(lines=True, joints=True, length_increments=True, angle_increments=True)
    o.length_steps = [.5]
    eng = SnapEngine(m, Plane("plan", 0), o, lambda p: (p[0] * 100, p[1] * 100), lambda x, y: (x / 100, y / 100, 0),
                     previous=(0, 0, 0))
    h = eng.apply_increments(eng.find(290, 20)[0])
    assert h.kind == "increment" and h.xyz[1] == pytest.approx(.2)  # still on the line
    assert math.hypot(h.xyz[0], h.xyz[1]) == pytest.approx(3.0)  # a length step
    j = eng.find(600, 20)[0]  # the joint at the end: exact, never moved
    assert j.kind == "joint" and eng.apply_increments(j).xyz == j.xyz
    free = eng.apply_increments(type(j)(xyz=(2.9, 0.8, 0), kind="free"))
    assert math.hypot(*free.xyz[:2]) == pytest.approx(3.0)
    ang = math.degrees(math.atan2(free.xyz[1], free.xyz[0])) % 15
    assert min(ang, 15 - ang) == pytest.approx(0, abs=1e-6)


# ----------------------------------------------------------------------- D11
def test_d11_joint_on_a_free_plan_is_made():
    doc = Document(base())
    t = PointTool(doc, DrawSettings())
    t.set_plane(Plane("plan", 4.5, None))
    assert t.click((2, 2, 4.5))
    assert [p.xyz for p in doc.model.points.values()] == [(2.0, 2.0, 4.5)]


# ----------------------------------------------------------------------- D12
def test_d12_form_values_reach_the_model():
    m = base()
    doc = Document(m)
    b = BraceTool(doc, DrawSettings(brace_section="B30x60", moment_releases="Pinned"))
    b.set_plane(Plane("3d"))
    b.click((0, 0, 0))
    b.click((6, 0, 3))
    f = m.frames[b.created[-1]]
    assert f.releases_i[4] and f.releases_i[5] and f.releases_j[4] and f.releases_j[5]
    m2 = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3, 3))
    q = QuickBraceTool(Document(m2), DrawSettings(brace_section="B30x60", moment_releases="Pinned"))
    q.set_plane(Plane("xz", 0))
    q.click((3, 0, 1.5))
    assert all(m2.frames[n].releases_i[5] and m2.frames[n].releases_j[5] for n in q.created)
    c = ColumnTool(Document(m2), DrawSettings(frame_section="C40x40", cardinal_point=5))
    c.set_story("Story1")
    c.click((2, 2, 3))
    assert m2.frames[c.created[-1]].cardinal_point == 5
    s = SlabTool(doc, DrawSettings(shell_section="S15", area_type="Opening", offset_z=1))
    s.set_story("Story1")
    for p in ((10, 0), (14, 0), (14, 3), (10, 3)):
        s.click(p)
    s.finish()
    ar = m.areas[s.created[-1]]
    assert ar.type == "Opening" and {m.points[p].z for p in ar.points} == {4.0}
    bm = base()
    p = [bm.add_point(*a) for a in ((0, 0, 3), (4, 0, 3), (4, 3, 3), (0, 3, 3))]
    for k in range(4):
        bm.add_frame(p[k], p[(k + 1) % 4], "B30x60")
    qs = QuickSlabTool(Document(bm), DrawSettings(shell_section="S15", quick_floor_edge="Beams"))
    qs.set_story("Story1")
    assert qs.click((2, 1, 3)) and len(bm.areas) == 1
    with pytest.raises(ModelError):
        qs.click((9, 9, 3))  # no loop there: said, not ignored


# ----------------------------------------------------------------------- D13
def test_d13_floor_wall_ramp_polygons_in_elevation_and_3d():
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3, 3))
    doc = Document(m)
    s = SlabTool(doc, DrawSettings(shell_section="S15", wall_section="W20"))
    s.set_plane(Plane("xz", 0.0))
    for p in ((0, 0, 0), (6, 0, 0), (6, 0, 3), (3, 0, 4), (0, 0, 3)):  # a gable wall
        s.click(p)
    assert s.finish() and m.areas[s.created[-1]].type == "Wall" and len(m.areas[s.created[-1]].points) == 5
    s.set_plane(Plane("3d"))
    for p in ((0, 0, 0), (6, 0, 0), (6, 5, 3), (0, 5, 3)):  # sloped: a ramp
        s.click(p)
    assert s.finish() and m.areas[s.created[-1]].type == "Ramp"
    for p in ((0, 0, 0), (6, 0, 0), (6, 5, 3), (0, 5, 0)):  # not flat: refused, nothing made
        s.click(p)
    n = len(m.areas)
    with pytest.raises(ModelError):
        s.finish()
    assert len(m.areas) == n
    r = RectSlabTool(doc, DrawSettings(shell_section="S15", wall_section="W20"))
    r.set_plane(Plane("yz", 0.0))
    r.click((0, 0, 0))
    r.click((0, 5, 3))
    assert m.areas[r.created[-1]].type == "Wall"


def test_d13_menu_labels_match_the_views_of_the_tools():
    """Capability matrix: a menu item saying (Plan, Elev, 3D) works in those windows,
    and a tool that works in a kind of window is not labelled as if it did not."""
    import re
    from structstudio.app import menus
    want = {"Plan": {"plan"}, "Elev": {"xz", "yz"}, "3D": {"3d"}}
    text = open(menus.__file__, encoding="utf-8").read()
    items = re.findall(r'\("([^"]*\(([^)]*)\)[^"]*)", "tool\.(\w+)"\)', text)
    assert len(items) >= 5
    for label, paren, cmd in items:
        words = {w.strip() for w in paren.split(",")} & set(want)
        if not words:
            continue
        views = set(TOOLS[cmd].views)
        for w in words:
            assert want[w] & views, f"{label}: the tool {cmd} does not work in {w} windows"
        for w, kinds in want.items():
            if w not in words:
                assert not (kinds & views), f"{label}: the tool {cmd} also works in {w} windows"


# ----------------------------------------------------------------------- D14
def test_d14_typed_absolute_points_and_3d_work_plane():
    doc = Document(base())
    t = BeamTool(doc, DrawSettings(frame_section="B30x60"))
    t.set_plane(Plane("3d"))
    p = t.typed("1,2,3")  # the first point as coordinates
    assert p == (1.0, 2.0, 3.0)
    t.click(p)
    assert t.typed("1,0,0") == pytest.approx((2, 2, 3))  # relative after it
    assert t.typed("#5,5,5") == (5.0, 5.0, 5.0)  # absolute after it
    fresh = BeamTool(doc, DrawSettings())
    fresh.set_plane(Plane("3d"))
    with pytest.raises(ValueError):
        fresh.typed("2,3")  # 3D: all three coordinates
    m = base()
    v = View(m)
    v.state.set_tool("beam")
    v.state.snap = opts(joints=True)
    tool = v.state.tool
    tool.set_plane(v.plane())
    tool.click((0, 0, 3))
    target = (4, 7, 3)
    assert v._snapped(*v._display(target))[0] is None  # snap only (default): no guessed depth
    tool.settings.work_plane_3d = "XY through the last point"
    xyz, h = v._snapped(*v._display(target))
    assert h.kind == "free" and xyz == pytest.approx(target)
    tool.settings.work_plane_3d = "XZ through the last point"
    xyz, h = v._snapped(*v._display((4, 0, 6)))
    assert xyz == pytest.approx((4, 0, 6))


# ----------------------------------------------------------------------- D15
def test_d15_intersections_are_searched_near_the_cursor_only():
    m = base()
    for k in range(600):
        m.add_frame(m.add_point(0, k * .07, 0), m.add_point(10, k * .07, 0), "B30x60")
    m.add_frame(m.add_point(5, -1, 0), m.add_point(5, 50, 0), "B30x60")
    eng = SnapEngine(m, Plane("plan", 0), opts(joints=True, intersections=True), lambda p: (p[0] * 20, p[1] * 20),
                     lambda x, y: (x / 20, y / 20, 0))
    t0 = time.perf_counter()
    assert eng.find(-200, -200) == []
    assert time.perf_counter() - t0 < 0.2  # was ~0.6 s scanning all 180,000 pairs
    hit = eng.find(5 * 20 + 1, 0.7 * 20 + 0.5)[0]  # crossing of the vertical with the 11th line
    assert hit.kind == "intersection" and hit.xyz == pytest.approx((5, 0.7, 0))


# ----------------------------------------------------------------------- D16
def test_d16_apertures_are_in_logical_pixels_at_every_scaling():
    m = base()
    rows = []
    for dpr in (1.0, 1.25, 1.5, 2.0):
        v = View(m, "plan", dpr=dpr)
        cam = v.renderer.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetParallelScale(10)
        eng = v._snap_engine(0, 0)
        rows.append((dpr, eng.tolerance, v._world_tolerance()))
    for dpr, tol, wt in rows:
        assert tol == pytest.approx(SnapOptions().tolerance_px * dpr)  # physical px = logical x dpr
        # the world size of PICK_PIXELS logical px: (2 x scale / logical height) x PICK
        assert wt == pytest.approx(2 * 10 / (800 / dpr) * ModelViewport.PICK_PIXELS)
        # both describe the same on-screen size: 10 logical px of aperture
        assert tol / dpr == pytest.approx(10.0)


# ----------------------------------------------------------------------- D17
def test_d17_an_impossible_arc_radius_is_reported_not_changed():
    pl = Plane("plan", 0)
    assert arc_polyline("Arc: start, end, radius", [(0, 0, 0), (4, 0, 0)], 8, 1, pl) == []
    msg = arc_problem("Arc: start, end, radius", [(0, 0, 0), (4, 0, 0)], 1, pl)
    assert "1" in msg and "2" in msg
    doc = Document(base())
    t = BeamTool(doc, DrawSettings(frame_section="B30x60", line_shape="Arc: start, end, radius", arc_radius=1))
    t.set_story("Story1")
    t.click((0, 0, 3))
    pv = t.preview((4, 0, 3))
    assert not pv.lines and "smaller than half the chord" in pv.text
    with pytest.raises(ModelError):
        t.click((4, 0, 3))
    assert not doc.model.frames
    t.settings.arc_radius = 3
    t.click((0, 0, 3))
    pv = t.preview((4, 0, 3))
    assert "R = 3.000" in pv.text
    t.click((4, 0, 3))
    assert len(doc.model.frames) == 8
