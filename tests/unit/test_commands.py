"""ETABS-style Edit / Select / Assign menu commands (GUI-free)."""
import math

import pytest

from structstudio.app import commands as C
from structstudio.app.document import Document
from structstudio.app.templates import new_building


@pytest.fixture
def doc():
    return Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3, 3), grid_only=False))


def _beams(m, story):
    return [f.name for f in m.frames.values() if f.type == "Beam" and f.story == story]


def test_copy_paste_keeps_properties_and_undo(doc):
    m = doc.model
    b = _beams(m, "Story1")[0]
    m.frames[b].angle = 30.0
    clip = C.copy_objects(m, [b], [], [])
    n0 = len(m.frames)
    fr, ar, pts = C.paste_objects(doc, clip, 0, 0, 10)
    assert len(fr) == 1 and len(doc.model.frames) == n0 + 1
    new = doc.model.frames[fr[0]]
    assert new.angle == 30.0 and new.section == doc.model.frames[b].section
    assert doc.model.points[new.point_i].z == pytest.approx(13.0)
    doc.undo()
    assert len(doc.model.frames) == n0


def test_replicate_radial_and_mirror(doc):
    m = doc.model
    col = next(f for f in m.frames.values() if f.type == "Column")
    p = m.points[col.point_i]
    made = C.replicate_transform(doc, [col.name], [], [], C.radial_transforms(0, 0, 90, 2))
    assert len(made) == 2
    q = doc.model.points[doc.model.frames[made[1]].point_i]
    assert (q.x, q.y) == pytest.approx((-p.x, -p.y), abs=1e-9)
    slab = next(a for a in doc.model.areas.values() if a.type == "Floor")
    made = C.replicate_transform(doc, [], [slab.name], [], C.mirror_transform("x", -1.0), reverse_areas=True)
    xs = [doc.model.points[n].x for n in doc.model.areas[made[0]].points]
    assert max(xs) <= -2.0 + 1e-9


def test_extrude_joints_and_frames(doc):
    m = doc.model
    p = next(iter(m.points))
    sec = next(iter(m.frame_sections))
    fr = C.extrude_points_to_frames(doc, [p], 0, 0, 2.0, 3, sec)
    assert len(fr) == 3 and all(doc.model.frames[n].type == "Column" for n in fr)
    wall = next(s for s in doc.model.shell_sections.values() if s.type == "Wall").name
    beam = _beams(doc.model, "Story1")[0]
    ar = C.extrude_frames_to_shells(doc, [beam], 0, 0, -3.0, 1, wall)
    assert doc.model.areas[ar[0]].type == "Wall"


def test_merge_align_reverse_join(doc):
    m = doc.model
    a = m.add_point(100, 0, 0)
    b = m.add_point(100.0004, 0, 0, merge=False)
    sec = next(iter(m.frame_sections))
    c = m.add_point(103, 0, 0)
    f1 = m.add_frame(b, c, sec)
    assert C.merge_joints(doc, [a.name, b.name], 0.001) == 1
    f1 = doc.model.frames[f1.name]
    assert {f1.point_i, f1.point_j} == {a.name, c.name} or b.name not in doc.model.points
    assert C.align_points(doc, [c.name], "z", 0.5) == 1 and doc.model.points[c.name].z == 0.5
    doc.model.points[c.name].z = 0.0
    # join two collinear frames meeting at an otherwise unused joint
    d = doc.model.add_point(106, 0, 0)
    f2 = doc.model.add_frame(c.name, d, sec)
    f1n = [f.name for f in doc.model.frames.values() if c.name in (f.point_i, f.point_j) and f.name != f2.name][0]
    joined = C.join_frames(doc, [f1n, f2.name])
    assert len(joined) == 1 and c.name not in doc.model.points
    keep = doc.model.frames[f1n]
    L = math.dist(doc.model.points[keep.point_i].xyz, doc.model.points[keep.point_j].xyz)
    assert L == pytest.approx(6.0, abs=1e-3)
    pi = keep.point_i
    assert C.reverse_frames(doc, [f1n]) == 1 and doc.model.frames[f1n].point_j == pi


def test_divide_shells_and_reverse_axis(doc):
    slab = next(a for a in doc.model.areas.values() if a.type == "Floor" and len(a.points) == 4)
    n0 = len(doc.model.areas)
    made = C.divide_shells(doc, [slab.name], 2, 3)
    assert len(made) == 6 and len(doc.model.areas) == n0 + 5
    a = doc.model.areas[made[0]]
    pts = list(a.points)
    C.reverse_area_axis(doc, [a.name])
    assert doc.model.areas[a.name].points == pts[::-1]


def test_grid_lines_at_points_and_relabel(doc):
    p = doc.model.add_point(3.0, 2.5, 0.0)
    n0 = len(doc.model.grid_lines)
    assert C.add_grid_lines_at_points(doc, [p.name]) == 2
    assert len(doc.model.grid_lines) == n0 + 2
    assert C.add_grid_lines_at_points(doc, [p.name]) == 0
    C.auto_relabel(doc)
    labels = {f.label for f in doc.model.frames.values() if f.story == "Story1" and f.type == "Column"}
    assert "C1" in labels


def test_copy_paste_assigns_groups_and_selection_by_property(doc):
    m = doc.model
    b1, b2 = _beams(m, "Story1")[:2]
    C.assign(doc, "frame", [b1], angle=15.0, mass_per_length=0.2)
    clip = C.copy_assigns(doc.model, "frame", b1)
    C.paste_assigns(doc, clip, [b2])
    assert doc.model.frames[b2].angle == 15.0 and doc.model.frames[b2].mass_per_length == 0.2
    C.assign_to_group(doc, "G-beams", [b1, b2], [], [])
    assert set(C.objects_in_group(doc.model, "G-beams")[0]) == {b1, b2}
    mat = doc.model.frame_sections[doc.model.frames[b1].section].material
    assert b1 in C.objects_with_material(doc.model, mat)[0]
    sec = doc.model.frames[b1].section
    assert b1 in C.objects_with_frame_section(doc.model, sec)[0]


def test_measure_text():
    assert C.measure([(0, 0, 0), (3, 4, 0)], "line").startswith("Length = 5")
    assert C.measure([(1, 0, 0), (0, 0, 0), (0, 1, 0)], "angle").startswith("Angle = 90")
    assert C.measure([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], "area").startswith("Area = 4")
