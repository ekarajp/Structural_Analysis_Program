"""Edit > Edit Frames > Join Frames through joints that other objects use."""
import numpy as np
import pytest

from structstudio import Model, analyze
from structstudio.app import commands as C
from structstudio.app.document import Document
from structstudio.engine.kernel import SciPyKernel
from structstudio.model import FIXED, FrameSection


def _model():
    m = Model("join", "kN m")
    m.use_csi_frame_defaults()
    m.add_material("C", E=25e6, nu=.2)
    m.add_frame_section(FrameSection.general("B", "C", A=.12, I33=1.6e-3, I22=4e-4, J=1e-3))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    mid = m.add_point(5, 0, 0)
    b = m.add_point(10, 0, 0, restraints=FIXED)
    f = m.add_frame(a, mid, "B", type="Beam").name
    g = m.add_frame(mid, b, "B", type="Beam").name
    # a cross beam framing in at the joint between the two pieces
    s = m.add_point(5, 4, 0, restraints=FIXED)
    h = m.add_frame(mid, s, "B", type="Beam").name
    m.add_load_pattern("P")
    m.add_frame_distributed_load(f, "P", -10.0)
    m.add_frame_distributed_load(g, "P", -10.0)
    m.add_point_load(mid, "P", force=(0, 0, -20))
    m.add_linear_static_case("P")
    return m, f, g, h, mid.name


def test_join_through_a_joint_used_by_another_beam_keeps_it_and_the_results():
    m, f, g, h, mid = _model()
    ref = analyze(m, kernel=SciPyKernel()).joint_displacement(mid, "P")
    doc = Document(m)
    assert C.join_frames(doc, [f, g]) == [g]
    m2 = doc.model
    assert mid in m2.points and len(m2.frames) == 2  # the joint stays (cross beam, load)
    got = analyze(m2, kernel=SciPyKernel()).joint_displacement(mid, "P")
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_join_refused_across_a_column_with_the_reason():
    m, f, g, h, mid = _model()
    top = m.add_point(5, 0, 3)
    m.add_frame(m.points[mid], top, "B", type="Column")
    doc = Document(m)
    assert C.join_frames(doc, [f, g]) == []
    assert any("column" in p for p in C.join_frames.last_problems)


def test_join_reports_different_sections_and_no_pair():
    m, f, g, h, mid = _model()
    m.add_frame_section(FrameSection.general("B2", "C", A=.2, I33=2e-3, I22=4e-4, J=1e-3))
    m.frames[g].section = "B2"
    doc = Document(m)
    assert C.join_frames(doc, [f, g]) == []
    assert any("different section" in p for p in C.join_frames.last_problems)
    m, f, g, h, mid = _model()
    doc = Document(m)
    assert C.join_frames(doc, [f, h]) == []  # at a right angle: not a pair
    assert C.join_frames.last_pairs == 0


def test_join_beams_along_a_slab_edge_same_results():
    from structstudio.model import ShellSection
    def build():
        m, f, g, h, mid = _model()
        m.frames.pop(h)
        m.add_shell_section(ShellSection("S", "C", thickness=.2))
        q1, q2 = m.add_point(0, 3, 0, restraints=FIXED), m.add_point(5, 3, 0)
        m.add_area([m.points[m.frames[f].point_i], m.points[mid], q2, q1], "S")
        return m, f, g, mid
    m, f, g, mid = build()
    ref = analyze(m, kernel=SciPyKernel()).joint_displacement(mid, "P")
    m, f, g, mid = build()
    doc = Document(m)
    assert C.join_frames(doc, [f, g]) == [g] and mid in doc.model.points
    got = analyze(doc.model, kernel=SciPyKernel()).joint_displacement(mid, "P")
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-12)


def test_edit_frames_menu_has_no_beam_axis_or_frame_type():
    """Move Beams to Real Axis is in Assign > Frame > Insertion Point and the frame type
    in Object Properties: Edit > Edit Frames does not repeat them."""
    from structstudio.app.menus import iter_commands
    cmds = {c: (path, label) for path, label, c, _ in iter_commands()}
    assert "edit.beam_axis" not in cmds and "edit.frame_type" not in cmds
    assert cmds["assign.insertion_point"][0] == ("Assign", "Frame")
    edit_frames = [label for path, label, c, _ in iter_commands() if path == ("Edit", "Edit Frames")]
    assert edit_frames == ["Divide Frames...", "Join Frames", "Reverse Frame Connectivity"]


def _doc():
    m = Model("join2", "kN m")
    m.add_material("C", E=25e6, nu=.2)
    m.add_frame_section(FrameSection.general("B", "C", A=.12, I33=1.6e-3, I22=4e-4, J=1e-3))
    return Document(m)


def _building(kind="rc_frame"):
    from structstudio.app import templates as T
    return Document(T.build_model(x_ordinates=(0, 6, 12), y_ordinates=(0, 5, 10), structure=kind))


def test_join_never_through_a_floor_joint_of_columns():
    """Joining everything selected joined the columns through the floors (and then the
    beams across them): the rigid end zones at the column ends were lost and the periods
    changed by several per cent.  Columns are not joined through a joint that beams or
    slabs use, and a beam never across a column - selected or not."""
    doc = _building()
    n_frames = len(doc.model.frames)
    assert C.join_frames(doc, list(doc.model.frames)) == []
    assert len(doc.model.frames) == n_frames
    why = " ".join(C.join_frames.last_problems)
    assert "column ends" in why and "is on a column" in why
    cols = [n for n, f in doc.model.frames.items() if f.type == "Column"]
    assert C.join_frames(doc, cols) == []


def test_join_columns_through_a_free_joint():
    doc = _doc()
    m = doc.model
    a, b, c = m.add_point(0, 0, 0), m.add_point(0, 0, 2), m.add_point(0, 0, 4)
    m.add_frame(a, b, "B", type="Column")
    m.add_frame(b, c, "B", type="Column")
    assert len(C.join_frames(doc, list(doc.model.frames))) == 1
    assert len(doc.model.frames) == 1 and b.name not in doc.model.points


def test_join_beams_not_across_a_column_running_through():
    """A column drawn through several stories has no end at the floor: the beams are not
    joined across it either."""
    doc = _doc()
    m = doc.model
    m.add_frame(m.add_point(4, 0, 0), m.add_point(4, 0, 6), "B", type="Column")
    p = [m.add_point(x, 0, 3) for x in (0, 4, 8)]
    m.add_frame(p[0], p[1], "B")
    m.add_frame(p[1], p[2], "B")
    assert C.join_frames(doc, [n for n, f in m.frames.items() if f.type == "Beam"]) == []
    assert "is on a column" in " ".join(C.join_frames.last_problems)


def test_check_model_warns_of_a_column_through_a_floor():
    """Check Model: a column drawn through a floor where beams end has no rigid end zone
    there; dividing it at the joints gives the story-by-story model back."""
    from structstudio.app import frame_divide as FD
    from structstudio.app import tools as TL
    from structstudio.model.checks import geometry_issues
    doc = _building()
    assert not [i for i in geometry_issues(doc.model) if i.code == "COLUMN_THROUGH_FLOOR"]

    def through(model):  # every column in one piece from the base to the roof
        cols = [f for f in model.frames.values() if f.type == "Column"]
        low = {f.point_j: f for f in cols}
        for f in cols:
            if f.point_i in low:
                low[f.point_i].point_j = f.point_j
                model.frames.pop(f.name)
    doc.edit(through)
    found = [i for i in geometry_issues(doc.model) if i.code == "COLUMN_THROUGH_FLOOR"]
    cols = [n for n, f in doc.model.frames.items() if f.type == "Column"]
    assert len(found) == 1 and found[0].severity == "warning" and found[0].points  # one line for all
    assert sorted(found[0].frames) == sorted(cols) and len(cols) == 9
    TL.divide_frames_at(doc, FD.at_joints(doc.model, cols))
    assert not [i for i in geometry_issues(doc.model) if i.code == "COLUMN_THROUGH_FLOOR"]
