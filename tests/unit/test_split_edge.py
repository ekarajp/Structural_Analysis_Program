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


def test_split_edge_meshes_like_a_joint_on_the_edge():
    """A corner added in the middle of a straight edge is meshed like a joint lying on the
    edge: a column standing there carries the same load with or without the split (it
    was a hanging node of one element: 93 kN instead of 83 kN at a 0.25 m mesh)."""
    from structstudio.api import analyze
    from structstudio.model import FIXED, ShellSection

    def run(split):
        doc = _doc()
        m = doc.model
        m.add_shell_section(ShellSection("S", "C", thickness=0.2))
        m.options.floor_mesh_max_size = 0.5
        a = m.add_area([m.add_point(x, y, 3) for x, y in ((0, 0), (8, 0), (8, 6), (0, 6))], "S", type="Floor")
        for x, y in ((0, 0), (8, 0), (8, 6), (0, 6), (4, 0)):
            m.add_frame(m.add_point(x, y, 0, restraints=FIXED), m.add_point(x, y, 3), "B", type="Column")
        m.add_load_pattern("L")
        m.add_area_load(a.name, "L", 5.0)
        m.add_linear_static_case("L", {"L": 1.0})
        if split:
            SE.split_edges(doc, [a.name], "joints")
            assert len(doc.model.areas[a.name].points) == 5
        r = analyze(doc.model)
        base = next(n for n, q in doc.model.points.items() if tuple(q.xyz) == (4.0, 0.0, 0.0))
        top = next(n for n, q in doc.model.points.items() if tuple(q.xyz) == (4.0, 0.0, 3.0))
        nid = r.am.point_to_node[top]
        return float(r.joint_reaction(base, "L")[2]), sum(1 for s in r.am.shells if nid in s.nodes)
    plain, split = run(False), run(True)
    assert split[0] == pytest.approx(plain[0], rel=1e-9)
    assert split[1] == plain[1] == 2  # the elements on both sides share the node
