"""Edit > Divide Frames: equal / max length / distances / at joints, crossing frames,
grid lines, reference lines, story levels and shell edges."""
import math

import pytest

from structstudio import Model
from structstudio.app import frame_divide as FD
from structstudio.app.document import Document
from structstudio.app.tools import divide_frames, divide_frames_at
from structstudio.model import FrameSection


def _doc():
    m = Model("div", "kN m")
    m.add_material("C", E=25e6, nu=.2)
    m.add_frame_section(FrameSection.general("B", "C", A=.1, I33=1e-3, I22=1e-3, J=1e-3))
    return Document(m)


def _beam(doc, a, b, sec="B"):
    m = doc.model
    return m.add_frame(m.add_point(*a), m.add_point(*b), sec).name


def _xs(doc, names):
    m = doc.model
    out = set()
    for n in names:
        f = m.frames[n]
        out |= {round(m.points[f.point_i].x, 6), round(m.points[f.point_j].x, 6)}
    return sorted(out)


def test_equal_unchanged_and_general_stations():
    doc = _doc()
    f = _beam(doc, (0, 0, 3), (6, 0, 3))
    made = divide_frames(doc, [f], 3)
    assert len(made) == 3 and _xs(doc, made) == [0, 2, 4, 6]
    g = _beam(doc, (0, 5, 3), (10, 5, 3))
    made = divide_frames_at(doc, {g: [0.25, 0.7, 0.7, 1e-9, 1.0]})  # duplicates / ends dropped
    assert len(made) == 3


def test_max_length_and_distances():
    doc = _doc()
    m = doc.model
    f = _beam(doc, (0, 0, 0), (10, 0, 0))
    assert FD.max_length(m, [f], 3.0)[f] == pytest.approx([.25, .5, .75])  # 4 pieces of 2.5
    assert FD.distances(m, [f], [1.5, 4], "I")[f] == pytest.approx([.15, .4])
    assert FD.distances(m, [f], [1.5], "J")[f] == pytest.approx([.85])
    assert FD.distances(m, [f], [.3], relative=True)[f] == pytest.approx([.3])
    assert FD.parse_numbers("1.5, 3") == [1.5, 3.0]
    assert FD.parse_numbers("3@2") == [2.0, 4.0, 6.0]


def test_at_joints_on_the_frame():
    doc = _doc()
    m = doc.model
    f = _beam(doc, (0, 0, 0), (8, 0, 0))
    m.add_point(3, 0, 0)
    m.add_point(5, 0.2, 0)  # off the frame
    st = FD.at_joints(m, [f])
    assert st[f] == pytest.approx([3 / 8])


def test_crossing_frames_share_a_joint():
    doc = _doc()
    m = doc.model
    a = _beam(doc, (0, 2, 3), (8, 2, 3))
    b = _beam(doc, (4, 0, 3), (4, 6, 3))
    st = FD.at_crossing_frames(m, [a, b], cut_others=False)
    assert st[a] == pytest.approx([.5]) and st[b] == pytest.approx([2 / 6])
    n0 = len(m.points)
    made = divide_frames_at(doc, st)
    assert len(made) == 4 and len(doc.model.points) == n0 + 1  # one shared new joint
    # only one selected, crossing frame cut too
    doc = _doc()
    a = _beam(doc, (0, 2, 3), (8, 2, 3))
    b = _beam(doc, (4, 0, 3), (4, 6, 3))
    st = FD.at_crossing_frames(doc.model, [a], cut_others=True)
    assert set(st) == {a, b}
    # skew lines that miss each other in 3-D are not cut
    c = _beam(doc, (0, 4, 0), (8, 4, 2))
    assert c not in FD.at_crossing_frames(doc.model, [c], [b])


def test_grid_lines_in_plan():
    doc = _doc()
    m = doc.model
    from structstudio.model.objects import GridLine
    sysname = next(iter(m.grid_systems), "G1")
    for nm, d, o in (("A", "X", 0.0), ("B", "X", 3.0), ("C", "X", 7.0), ("1", "Y", 0.0), ("2", "Y", 6.0)):
        m.grid_lines.append(GridLine(sysname, nm, d, o))
    f = _beam(doc, (0, 1, 3), (10, 1, 3))
    assert FD.at_grid_lines(m, [f])[f] == pytest.approx([.3, .7])
    col = _beam(doc, (3, 0, 0), (3, 0, 3))
    assert col not in FD.at_grid_lines(m, [col])


def test_reference_lines_and_stories_and_edges():
    doc = _doc()
    m = doc.model
    f = _beam(doc, (0, 0, 3), (10, 0, 3))
    m.add_point(4, 5, 3)  # Y reference line through it crosses the beam at x = 4
    m.add_point(6, 5, 0)  # other level: no crossing of a plan line at z = 3
    st = FD.at_reference_lines(m, [f], "XY")
    assert st[f] == pytest.approx([.4])
    m.add_story("S1", 3.0, 3.0) if hasattr(m, "add_story") else None
    col = _beam(doc, (20, 0, 0), (20, 0, 9))
    elev = [s.elevation for s in m.stories_bottom_up()]
    ss = FD.at_stories(m, [col])
    assert ss.get(col, []) == pytest.approx(sorted(z / 9 for z in elev if 0 < z < 9))
    pts = [m.add_point(x, y, 3) for x, y in ((2, -1), (5, -1), (5, 1), (2, 1))]
    m.add_area(pts, None)
    assert FD.at_shell_edges(m, [f])[f] == pytest.approx([.2, .5])


def test_loads_and_hinges_follow_general_stations():
    doc = _doc()
    m = doc.model
    m.add_load_pattern("D")
    f = _beam(doc, (0, 0, 0), (10, 0, 0))
    m.add_frame_distributed_load(f, "D", -2.0)
    made = divide_frames_at(doc, {f: [.3]})
    tot = 0.0
    for n in made:
        fr = doc.model.frames[n]
        Ln = math.dist(doc.model.points[fr.point_i].xyz, doc.model.points[fr.point_j].xyz)
        for ld in fr.loads:
            assert ld.value_a == ld.value_b == -2.0
            tot += ld.value_a * (ld.rel_b - ld.rel_a) * Ln
    assert tot == pytest.approx(-20.0)
