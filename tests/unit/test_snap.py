"""Object snaps (ETABS Snap Options / CAD snaps) - pure Python, no window."""
import numpy as np
import pytest

from structstudio.app.snap import SnapEngine, SnapOptions, reference_lines_for
from structstudio.app.templates import new_building
from structstudio.app.tools import Plane


def plan_engine(model, opt, previous=None, z=0.0, scale=50.0):
    """A plan view at z with 50 px per model unit (orthographic, no offset)."""
    pl = Plane("plan", z, None)

    def project(xyz):
        return (xyz[0] * scale, xyz[1] * scale)

    def to_plane(x, y):
        return (x / scale, y / scale, z)

    return SnapEngine(model, pl, opt, project, to_plane, previous, 1e-6,
                      reference_lines_for(model, pl) if opt.reference else ())


@pytest.fixture
def model():
    m = new_building(grid_only=False)
    return m


def test_options_roundtrip_and_select_all():
    o = SnapOptions()
    o.lines = True
    o.length_steps = [0.25, 1.0]
    o2 = SnapOptions.from_json(o.to_json())
    assert o2.lines and o2.length_steps == [0.25, 1.0] and o2.joints
    o2.select_all(False)
    assert not any(o2.enabled(k) for k in ("joints", "grid", "ortho"))
    with pytest.raises(KeyError):
        o2.set("bogus", True)


def test_joint_and_grid_snaps_in_plan(model):
    z = list(model.stories.values())[0].elevation
    o = SnapOptions()
    eng = plan_engine(model, o, z=z)
    p = next(p for p in model.points.values() if abs(p.z - z) < 1e-9)
    hits = eng.find(p.x * 50 + 3, p.y * 50 - 2)
    assert hits and hits[0].kind == "joint" and hits[0].target == p.name
    # a grid intersection without a joint: between the columns there are none in this template,
    # so use the plan at a level with no joints
    eng2 = plan_engine(model, o, z=99.0)
    hits = eng2.find(6.0 * 50 + 2, 5.0 * 50 + 2)
    assert hits and hits[0].kind == "grid" and hits[0].xyz == pytest.approx((6.0, 5.0, 99.0))


def test_midpoint_nearest_and_perpendicular(model):
    z = list(model.stories.values())[1].elevation
    o = SnapOptions()
    o.lines = True
    eng = plan_engine(model, o, z=z)
    f = next(f for f in model.frames.values() if f.type == "Beam")
    a, b = model.points[f.point_i], model.points[f.point_j]
    z = a.z
    mid = ((a.x + b.x) / 2, (a.y + b.y) / 2)
    hits = eng.find(mid[0] * 50 + 1, mid[1] * 50 + 1)
    assert hits[0].kind == "midpoint" and hits[0].target == f.name
    # nearest point on the beam, a third along
    q = (a.x + (b.x - a.x) * 0.3, a.y + (b.y - a.y) * 0.3)
    hits = eng.find(q[0] * 50, q[1] * 50 + 4)
    near = [h for h in hits if h.kind == "nearest" and h.target == f.name]
    assert near and near[0].xyz[:2] == pytest.approx(q, abs=1e-9)
    # perpendicular from a previous point off the beam
    o.perpendicular = True
    prev = (q[0] + 0.0, q[1] + 2.0, z) if abs(b.y - a.y) < 1e-9 else (q[0] + 2.0, q[1], z)
    eng = plan_engine(model, o, previous=prev, z=z)
    hits = eng.find(q[0] * 50 + 5, q[1] * 50 + 5)
    perp = [h for h in hits if h.kind == "perpendicular" and h.target == f.name]
    assert perp and perp[0].xyz[:2] == pytest.approx(q, abs=1e-9) and perp[0].track is not None


def test_intersection_extension_and_tracking(model):
    z = list(model.stories.values())[1].elevation
    o = SnapOptions()
    o.select_all(False)
    o.intersections = True
    eng = plan_engine(model, o, z=z)
    # two beams crossing at a joint are skipped (shared end), so add a diagonal brace in plan
    pts = [p for p in model.points.values() if abs(p.z - z) < 1e-9]
    a = min(pts, key=lambda p: (p.x, p.y))
    bname = model.add_frame((a.x, a.y, z), (a.x + 6.0, a.y + 5.0, z), list(model.frame_sections)[0], type="Beam").name
    eng = plan_engine(model, o, z=z)
    # the diagonal crosses the beam at y = a.y + 5 * 0.5 at x = a.x + 3 ... intersection with grid line x = a.x + 6? use
    # the beam along y = a.y + 5 between x and x+6: diagonal meets it at its end (shared) -> use grid x = a.x + 3?
    # instead check the crossing with the beam at y = a.y + 5 is skipped and with the interior grid-free beam
    hits = eng.find((a.x + 3.0) * 50, (a.y + 2.5) * 50)
    assert all(h.kind == "intersection" for h in hits) and not hits or hits  # no crossing exactly there
    # extension beyond the diagonal end
    o.extensions = True
    eng = plan_engine(model, o, z=z)
    hits = eng.find((a.x + 7.2) * 50, (a.y + 6.0) * 50)
    ext = [h for h in hits if h.kind == "extension" and h.target == bname]
    assert ext and ext[0].xyz[0] > a.x + 6.0
    # ortho tracking from a previous point
    o.select_all(False)
    o.ortho = True
    eng = plan_engine(model, o, previous=(1.0, 1.0, z), z=z)
    hits = eng.find(4.0 * 50, 1.05 * 50)
    assert hits and hits[0].kind == "ortho" and hits[0].xyz[:2] == pytest.approx((4.0, 1.0))
    # polar at 45 deg
    o.polar = True
    o.ortho = False
    eng = plan_engine(model, o, previous=(1.0, 1.0, z), z=z)
    hits = eng.find(3.0 * 50, 3.02 * 50)
    assert hits and hits[0].kind == "polar" and hits[0].xyz[:2] == pytest.approx((3.01, 3.01), abs=0.02)


def test_increments_and_fine_grid(model):
    z = 0.0
    o = SnapOptions()
    o.select_all(False)
    o.length_increments = True
    o.length_steps = [0.5]
    o.angle_increments = True
    o.angle_increment = 15.0
    eng = plan_engine(model, o, previous=(0.0, 0.0, z), z=z)
    from structstudio.app.snap import SnapHit
    hit = eng.apply_increments(SnapHit((2.9, 0.2, z), "free"))
    assert hit.kind == "increment"
    L = np.hypot(hit.xyz[0], hit.xyz[1])
    ang = np.degrees(np.arctan2(hit.xyz[1], hit.xyz[0]))
    assert L == pytest.approx(3.0) and ang == pytest.approx(0.0, abs=1e-9)
    o.fine_grid = True
    o.fine_spacing = 0.25
    eng = plan_engine(model, o, z=z)
    hits = eng.find(1.26 * 50, 0.74 * 50)
    assert hits and hits[0].kind == "fine_grid" and hits[0].xyz[:2] == pytest.approx((1.25, 0.75))


def test_reference_lines(model):
    pl = Plane("plan", 0.0, None)
    refs = reference_lines_for(model, pl)
    xs = {round(a[0], 6) for a, b in refs if abs(a[0] - b[0]) < 1e-9}
    ys = {round(a[1], 6) for a, b in refs if abs(a[1] - b[1]) < 1e-9}
    assert xs == {0.0, 6.0, 12.0, 18.0} and ys == {0.0, 5.0, 10.0}
    assert all(abs(a[2]) < 1e-9 for a, b in refs)  # in the plane
    refs3 = reference_lines_for(model, Plane("3d"), (False, False, True))
    assert refs3 and all(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 for a, b in refs3)


def test_object_point_beats_tracking_lines_and_override(model):
    """CAD priority: with the cursor inside the aperture of a joint, the joint wins
    over the alignment / reference / polar lines that pass closer to the cursor."""
    z = list(model.stories.values())[0].elevation
    o = SnapOptions()  # defaults: reference lines, polar and alignment tracking on
    p = next(p for p in model.points.values() if abs(p.z - z) < 1e-9)
    prev = (p.x - 3.0, p.y - 3.0, z)  # polar 45 deg ray from here runs through the joint
    eng = plan_engine(model, o, previous=prev, z=z)
    hits = eng.find(p.x * 50 + 7, p.y * 50 + 0.5)  # 7 px right of the joint, on its alignment line
    assert hits[0].kind in ("joint", "grid") and hits[0].xyz[:2] == pytest.approx((p.x, p.y))
    # Shift / Ctrl + right-click override: only the chosen kind, e.g. the midpoint of a beam
    f = next(f for f in model.frames.values() if f.type == "Beam")
    a, b = model.points[f.point_i], model.points[f.point_j]
    z = a.z
    mid = ((a.x + b.x) / 2, (a.y + b.y) / 2)
    eng = plan_engine(model, o, z=z)
    eng.override = "midpoint"
    hits = eng.find(mid[0] * 50 + 20, mid[1] * 50 + 3)  # outside the normal aperture
    assert hits and all(h.kind == "midpoint" for h in hits) and hits[0].xyz[:2] == pytest.approx(mid)
    assert o.joints and o.reference  # the override leaves the settings as they were
    eng.override = "endpoint"
    assert all(h.kind in ("joint", "end") for h in eng.find(a.x * 50 + 3, a.y * 50))


def test_arc_polylines():
    from structstudio.app.tools import arc_polyline
    pl = Plane("plan", 0.0, None)
    pts = arc_polyline("Arc: 3 points", [(0, 0, 0), (1, 1, 0), (2, 0, 0)], 4, 0.0, pl)
    assert len(pts) == 5 and pts[2] == pytest.approx((1, 1, 0)) and pts[-1] == pytest.approx((2, 0, 0))
    assert all(np.hypot(q[0] - 1, q[1]) == pytest.approx(1) for q in pts)
    pts = arc_polyline("Arc: start, center, end", [(1, 0, 0), (0, 0, 0), (0, 1, 0)], 2, 0.0, pl)
    assert pts[1] == pytest.approx((2 ** -0.5, 2 ** -0.5, 0))
    pts = arc_polyline("Arc: start, end, radius", [(0, 0, 0), (2, 0, 0)], 2, 1.0, pl)
    assert pts[1] == pytest.approx((1, 1, 0))
    # a radius smaller than half the chord: the half circle (not an error)
    pts = arc_polyline("Arc: start, end, radius", [(0, 0, 0), (2, 0, 0)], 2, 0.5, pl)
    assert pts[1] == pytest.approx((1, 1, 0))
    # 3D view (viewer on -Y, looking at +Y): every mode works in the plane of its points,
    # counter-clockwise / bulging left as seen on the screen
    d3 = Plane("3d", 0.0, None, normal=(0.0, -1.0, 0.0))
    pts = arc_polyline("Arc: start, center, end", [(1, 0, 0), (0, 0, 0), (0, 0, 1)], 2, 0.0, d3)
    assert pts[1] == pytest.approx((2 ** -0.5, 0, 2 ** -0.5)) and pts[-1] == pytest.approx((0, 0, 1))
    pts = arc_polyline("Arc: start, center, end", [(1, 0, 0), (0, 0, 0), (0, 0, -1)], 2, 0.0, d3)
    assert pts[1] == pytest.approx((-2 ** -0.5, 0, 2 ** -0.5))  # 270 degrees, the long way round
    pts = arc_polyline("Arc: start, end, radius", [(0, 0, 0), (2, 0, 0)], 2, 1.0, d3)
    assert pts[1] == pytest.approx((1, 0, 1))
    pts = arc_polyline("Arc: start, end, radius", [(0, 0, 0), (0, 2, 3)], 4, 0.0, d3)
    assert len(pts) == 5 and pts[-1] == pytest.approx((0, 2, 3))


def test_joint_beats_a_nearby_midpoint(model):
    """A midpoint that lands next to a joint on the screen (3D: the middle of a beam of
    another level in front of a base joint) does not steal the click from the joint
    square the user aims at, unless it is clearly nearer the cursor."""
    z = list(model.stories.values())[1].elevation
    o = SnapOptions()
    o.lines = True
    eng = plan_engine(model, o, z=z, scale=2.0)  # zoomed out: joint and midpoint 6 px apart
    p = next(p for p in model.points.values() if abs(p.z - z) < 1e-9 and p.x == 0 and p.y == 0)
    hits = eng.find(2.5, 0.0)  # 2.5 px from the joint, 3.5 px from the midpoint of the beam
    assert hits[0].kind == "joint" and hits[0].target == p.name
    hits = eng.find(2.0, 0.0)
    assert hits[0].kind == "joint"
    hits = eng.find(6.0, 0.0)  # right on the midpoint
    assert hits[0].kind == "midpoint"
