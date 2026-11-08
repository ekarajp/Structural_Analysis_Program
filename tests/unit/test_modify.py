"""CAD modify commands: move / copy / rotate / mirror / trim / extend / offset, typed input."""
import math

import numpy as np
import pytest

from structstudio.app import modify as M
from structstudio.app.document import Document
from structstudio.app.templates import new_building
from structstudio.app.tools import Plane
from structstudio.model import ModelError


def _doc():
    m = new_building(x_spacings=(6.0, 6.0), y_spacings=(5.0,), story_heights=(3.0,), grid_only=False)
    return Document(m)


def _beam(m, x0, y0, x1, y1, z=3.0):
    for f in m.frames.values():
        a, b = m.points[f.point_i].xyz, m.points[f.point_j].xyz
        if {(round(a[0], 6), round(a[1], 6), round(a[2], 6)), (round(b[0], 6), round(b[1], 6), round(b[2], 6))} == \
                {(x0, y0, z), (x1, y1, z)}:
            return f.name
    raise KeyError((x0, y0, x1, y1))


def test_transforms():
    T = M.rotation((1, 1, 0), (0, 0, 1), 90)
    assert np.allclose(T(np.array([2.0, 1.0, 5.0])), [1.0, 2.0, 5.0])
    R = M.mirror((0, 0, 0), (0, 1, 0), (0, 0, 1))  # mirror about the Y axis in plan
    assert np.allclose(R(np.array([3.0, 2.0, 1.0])), [-3.0, 2.0, 1.0])
    with pytest.raises(ModelError):
        M.mirror((1, 1, 0), (1, 1, 0), (0, 0, 1))


def test_move_detaches_and_merges():
    doc = _doc()
    m = doc.model
    b = _beam(m, 0, 0, 6, 0)
    n_pts = len(m.points)
    # moving a beam does not drag its columns (CAD behaviour): the beam gets its own joints
    M.transform_in_place(doc, [b], [], [], M.translation((0, 2.0, 0)))
    f = m.frames[b]
    assert m.points[f.point_i].xyz == (0.0, 2.0, 3.0) and m.points[f.point_j].xyz == (6.0, 2.0, 3.0)
    cols = [c for c in m.frames.values() if c.type == "Column"]
    assert all(m.points[c.point_j].z == 3.0 and m.points[c.point_j].y in (0.0, 5.0) for c in cols)
    assert len(m.points) == n_pts + 2
    # moved back onto the column tops: merged again (connected, no duplicate joints)
    doc.undo()
    m = doc.model  # undo restores a snapshot of the model
    M.transform_in_place(doc, [b], [], [], M.translation((0, 5.0, 0)))
    assert len(m.points) == n_pts  # the beam now sits on the y = 5 joints
    assert {m.frames[b].point_i, m.frames[b].point_j} <= set(m.points)
    # stretch: shared joints move with the beam (ETABS Move)
    doc.undo()
    m = doc.model
    M.transform_in_place(doc, [b], [], [], M.translation((0, 0, 0.5)), stretch=True)
    assert m.points[m.frames[b].point_i].z == 3.5


def test_copy_rotate_mirror():
    doc = _doc()
    m = doc.model
    b = _beam(m, 0, 0, 6, 0)
    fr, ar, pts = M.copy_transformed(doc, [b], [], [], M.translation((0, 2.5, 0)))
    assert len(fr) == 1 and m.frames[fr[0]].section == m.frames[b].section
    g = m.frames[fr[0]]
    assert m.points[g.point_i].y == 2.5
    fr2, _, _ = M.copy_transformed(doc, [b], [], [], M.rotation((0, 0, 3), (0, 0, 1), 90))
    h = m.frames[fr2[0]]
    ends = sorted([m.points[h.point_i].xyz, m.points[h.point_j].xyz])
    assert np.allclose(ends, [(0, 0, 3), (0, 6, 3)])
    area = next(a for a in m.areas.values() if a.type == "Floor")
    _, ar3, _ = M.copy_transformed(doc, [], [area.name], [], M.mirror((0, 0, 0), (0, 1, 0), (0, 0, 1)),
                                   reverse_areas=True)
    assert ar3 and all(m.points[p].x <= 0 for p in m.areas[ar3[0]].points)


def test_trim_extend_offset():
    doc = _doc()
    m = doc.model
    pl = Plane("plan", 3.0, "Story1")
    b = _beam(m, 0, 0, 6, 0)
    # a beam across the bay: y = 2.5 from x = 0 to 12 crosses the x = 6 grid / beam
    nb = m.add_frame(m.add_point(0, 2.5, 3), m.add_point(12, 2.5, 3), m.frames[b].section).name
    span = M.trim_plan(m, nb, pl, (9.0, 2.5, 3.0))
    assert span is not None and abs(span[0] - 0.5) < 1e-9 and span[1] == 1.0
    M.trim_frame(doc, nb, pl, (9.0, 2.5, 3.0))
    f = m.frames[nb]
    assert sorted([m.points[f.point_i].x, m.points[f.point_j].x]) == [0.0, 6.0]
    # extend it again to the x = 12 grid line
    M.extend_frame(doc, nb, pl, (5.5, 2.5, 3.0))
    f = m.frames[nb]
    assert sorted([m.points[f.point_i].x, m.points[f.point_j].x]) == [0.0, 12.0]
    # trim in the middle splits it in two (cut at x = 6 and nothing else: end part removed)
    n_before = len(m.frames)
    M.trim_frame(doc, nb, pl, (2.0, 2.5, 3.0))
    assert len(m.frames) == n_before
    # offset: a parallel copy 1.0 towards +y
    new = M.offset_frame(doc, b, pl, 1.0, (3.0, 4.0, 3.0))
    g = m.frames[new]
    assert m.points[g.point_i].y == pytest.approx(1.0) and m.points[g.point_j].y == pytest.approx(1.0)
    with pytest.raises(ModelError):
        M.offset_frame(doc, b, pl, 0.0, (3.0, 4.0, 3.0))


def test_parse_typed():
    pl = Plane("plan", 3.0, "Story1")
    assert np.allclose(M.parse_typed("6", (0, 0, 3), (1, 1, 3), pl), (6 / math.sqrt(2), 6 / math.sqrt(2), 3))
    assert np.allclose(M.parse_typed("2<90", (1, 1, 3), None, pl), (1, 3, 3))
    assert np.allclose(M.parse_typed("@3,4", (1, 1, 3), None, pl), (4, 5, 3))
    assert np.allclose(M.parse_typed("1,2,3", (0, 0, 0), None, pl), (1, 2, 3))
    el = Plane("xz", 0.0)
    assert np.allclose(M.parse_typed("3,4", (0, 0, 0), None, el), (3, 0, 4))
    with pytest.raises(ValueError):
        M.parse_typed("abc", (0, 0, 0), None, pl)


def test_move_joint_merges_onto_existing():
    doc = _doc()
    m = doc.model
    b = _beam(m, 0, 0, 6, 0)
    nb = m.add_frame(m.add_point(0, 2.5, 3), m.add_point(3, 2.5, 3), m.frames[b].section).name
    end = m.frames[nb].point_j
    target = m.frames[b].point_j  # (6, 0, 3)
    name = M.move_joint(doc, end, m.points[target].xyz)
    assert name == target and m.frames[nb].point_j == target and end not in m.points
