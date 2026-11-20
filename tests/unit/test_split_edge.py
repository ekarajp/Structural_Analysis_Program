"""Edit > Edit Shells > Split Shell Edge / Remove Joints From Shells."""
import pytest

from structstudio import Model
from structstudio.app import shell_edit as SE
from structstudio.app.document import Document
from structstudio.model import FrameSection


def _doc():
    m = Model("se", "kN m")
    m.add_material("C", E=25e6, nu=.2)
    m.add_frame_section(FrameSection.general("B", "C", A=.1, I33=1e-3, I22=1e-3, J=1e-3))
    return Document(m)


def _slab(doc, xy, z=3.0):
    m = doc.model
    return m.add_area([m.add_point(x, y, z) for x, y in xy], None, type="Floor").name


def _xy(doc, n):
    m = doc.model
    return [(round(m.points[p].x, 6), round(m.points[p].y, 6)) for p in m.areas[n].points]


def test_split_at_joints_on_the_edges():
    doc = _doc()
    m = doc.model
    s = _slab(doc, [(0, 0), (8, 0), (8, 6), (0, 6)])
    b = m.add_frame(m.add_point(3, 0, 3), m.add_point(3, -4, 3), "B")  # a beam ending mid-edge
    m.add_point(5, 3, 3)  # inside, not on an edge
    done, _ = SE.split_edges(doc, [s], "joints")
    assert done == [s] and _xy(doc, s) == [(0, 0), (3, 0), (8, 0), (8, 6), (0, 6)]
    assert m.frames[b.name].point_i in m.areas[s].points
    # only the selected joints
    t = _slab(doc, [(20, 0), (28, 0), (28, 6), (20, 6)])
    p1, p2 = m.add_point(24, 0, 3), m.add_point(28, 2, 3)
    SE.split_edges(doc, [t], "joints", joints=[p2.name])
    assert _xy(doc, t) == [(20, 0), (28, 0), (28, 2), (28, 6), (20, 6)]
    doc.undo()
    assert len(doc.model.areas[t].points) == 4


def test_split_into_equal_parts_and_by_length():
    doc = _doc()
    s = _slab(doc, [(0, 0), (8, 0), (8, 6), (0, 6)])
    SE.split_edges(doc, [s], "equal", 2)
    assert _xy(doc, s) == [(0, 0), (4, 0), (8, 0), (8, 3), (8, 6), (4, 6), (0, 6), (0, 3)]
    t = _slab(doc, [(20, 0), (28, 0), (28, 6), (20, 6)])
    SE.split_edges(doc, [t], "max_length", 2.5)  # 8 m -> 4 parts, 6 m -> 3 parts
    assert len(doc.model.areas[t].points) == 4 + 2 * 3 + 2 * 2


def test_remove_joints_from_shells():
    doc = _doc()
    m = doc.model
    s = _slab(doc, [(0, 0), (4, 0), (8, 0), (8, 6), (0, 6)])
    mid = m.areas[s].points[1]
    done, _ = SE.remove_joints(doc, [], [mid])  # no shell selected: every shell using it
    assert done == [s] and _xy(doc, s) == [(0, 0), (8, 0), (8, 6), (0, 6)]
    assert mid not in m.points  # not used by anything any more
    # a corner taken out cuts it off; fewer than three corners: left alone
    tri = _slab(doc, [(20, 0), (24, 0), (20, 4)])
    done, notes = SE.remove_joints(doc, [tri], [m.areas[tri].points[1]])
    assert done == [] and notes and len(m.areas[tri].points) == 3
