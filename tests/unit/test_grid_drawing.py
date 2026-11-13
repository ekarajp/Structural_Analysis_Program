"""Grid lines drawn and edited in the plan window (structstudio.app.grid_tools): drawn by
two points, offset, renamed, hidden, deleted and moved - with the joints on a moved line
moving along (Move Objects with Grid Lines) - each one undo step."""
import math

import pytest

from structstudio.app.document import Document
from structstudio.app.grid_tools import (bubble_at, delete_grid_line, draw_grid_line, glue_moves, line_at,
                                         move_grid_line, offset_grid_line, rename_grid_line,
                                         set_grid_line_visible, show_all_grid_lines)
from structstudio.app.templates import new_building
from structstudio.model.base import ModelError
from structstudio.model.grids import grid_geometry
from structstudio.model.objects import GridLine, GridSystem


def _doc(frames=False):
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.0, 3.0), grid_only=not frames)
    return Document(m)


def _line(m, name, system="G1"):
    return next(g for g in m.grid_lines if (g.system or "G1") == system and g.name == name)


def test_drawn_line_along_an_axis_is_an_x_or_y_line_named_on():
    doc = _doc()
    assert draw_grid_line(doc, (15.0, -2.0), (15.0, 8.0)) == "D"  # constant x: an X line after A, B, C
    g = _line(doc.model, "D")
    assert (g.direction, g.ordinate) == ("X", 15.0)
    assert draw_grid_line(doc, (-1.0, 9.0), (14.0, 9.0)) == "3"  # constant y: a Y line after 1, 2
    assert _line(doc.model, "3").ordinate == 9.0
    with pytest.raises(ModelError):  # the same line again
        draw_grid_line(doc, (15.0, 0.0), (15.0, 1.0))
    doc.undo()
    assert "3" not in {g.name for g in doc.model.grid_lines}


def test_drawn_line_at_an_angle_is_a_general_line_of_the_system():
    doc = _doc()
    m = doc.model
    m.grid_systems["G1"].origin = (1.0, 2.0)
    name = draw_grid_line(doc, (0.0, 0.0), (6.0, 5.0))
    g = _line(doc.model, name)
    assert name == "L1" and g.direction == "General"
    assert g.start == pytest.approx((-1.0, -2.0)) and g.end == pytest.approx((5.0, 3.0))  # local (origin moved)
    a, b = next((a, b) for a, b, n, _ in grid_geometry(doc.model).lines if n == "L1")
    assert a == pytest.approx((0.0, 0.0)) and b == pytest.approx((6.0, 5.0))


def test_drawn_line_in_a_turned_system_along_its_axes():
    doc = _doc()
    m = doc.model
    gs = m.grid_systems["G1"]
    gs.rotation = 30.0
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    # along the system's local y at u = 4: an X line of ordinate 4
    name = draw_grid_line(doc, (4 * c - 0 * s, 4 * s), (4 * c - 7 * s, 4 * s + 7 * c))
    g = _line(doc.model, name)
    assert (g.direction, g.ordinate) == ("X", pytest.approx(4.0))


def test_offset_on_the_side_clicked_named_primed():
    doc = _doc()
    assert offset_grid_line(doc, "G1", "B", 1.5, (7.0, 2.0)) == "B'"
    assert _line(doc.model, "B'").ordinate == 7.5
    assert offset_grid_line(doc, "G1", "B", 1.5, (5.0, 2.0)) == "B''"
    assert _line(doc.model, "B''").ordinate == 4.5
    assert offset_grid_line(doc, "G1", "1", 2.0, (3.0, -1.0)) == "1'"
    assert _line(doc.model, "1'").ordinate == -2.0
    with pytest.raises(ModelError):
        offset_grid_line(doc, "G1", "B", 0.0, (7.0, 2.0))
    name = draw_grid_line(doc, (0.0, 0.0), (4.0, 4.0))
    new = offset_grid_line(doc, "G1", name, math.sqrt(2.0), (0.0, 4.0))  # to the upper left
    g = _line(doc.model, new)
    assert g.start == pytest.approx((-1.0, 1.0)) and g.end == pytest.approx((3.0, 5.0))


def test_offset_of_a_circle_and_not_of_a_radial_line():
    doc = _doc()
    m = doc.model
    m.grid_systems["C1"] = GridSystem(name="C1", type="Cylindrical", origin=(30.0, 0.0))
    m.grid_lines += [GridLine(system="C1", name="R1", direction="R", ordinate=4.0),
                     GridLine(system="C1", name="T1", direction="T", ordinate=0.0)]
    assert offset_grid_line(doc, "C1", "R1", 1.0, (30.0, 6.0)) == "R1'"
    assert _line(doc.model, "R1'", "C1").ordinate == 5.0
    with pytest.raises(ModelError):
        offset_grid_line(doc, "C1", "T1", 1.0, (33.0, 1.0))


def test_rename_hide_show_delete():
    doc = _doc()
    rename_grid_line(doc, "G1", "B", "B1")
    assert _line(doc.model, "B1").ordinate == 6.0
    with pytest.raises(ModelError):
        rename_grid_line(doc, "G1", "A", "C")
    set_grid_line_visible(doc, "G1", "C", False)
    assert "C" not in {n for _, _, n, _ in grid_geometry(doc.model).lines}
    assert show_all_grid_lines(doc) == 1 and _line(doc.model, "C").visible
    delete_grid_line(doc, "G1", "C")
    assert "C" not in {g.name for g in doc.model.grid_lines}
    doc.undo()
    assert "C" in {g.name for g in doc.model.grid_lines}


def test_move_with_and_without_the_joints():
    from structstudio.app.templates import build_model
    m = build_model(structure="rc_frame", stories=[("Story1", 3.0)], x_ordinates=[0, 6, 12], y_ordinates=[0, 5])
    doc = Document(m)
    on_b = sorted(n for n, p in m.points.items() if abs(p.x - 6.0) < 1e-9)
    assert on_b
    assert move_grid_line(doc, "G1", "B", to=6.5, move_objects=False) == 0
    assert _line(doc.model, "B").ordinate == 6.5
    assert all(abs(doc.model.points[n].x - 6.0) < 1e-9 for n in on_b)
    doc.undo()
    n = move_grid_line(doc, "G1", "B", to=6.5, move_objects=True)
    assert n == len(on_b)
    assert all(abs(doc.model.points[k].x - 6.5) < 1e-9 for k in on_b)
    assert all(abs(doc.model.points[k].x - 6.5) > 1e-6 for k in doc.model.points if k not in on_b)
    with pytest.raises(ModelError):  # onto another line
        move_grid_line(doc, "G1", "B", to=12.0)
    doc.undo()  # one step: the line and the joints together
    assert _line(doc.model, "B").ordinate == 6.0
    assert all(abs(doc.model.points[k].x - 6.0) < 1e-9 for k in on_b)


def test_glue_moves_of_two_crossing_lines_and_a_turned_system():
    doc = _doc()
    m = doc.model
    p = m.add_point(6.0, 5.0, 3.0)
    q = m.add_point(6.0, 2.0, 3.0)
    mv = glue_moves(m, [("G1", "X", 6.0, 7.0), ("G1", "Y", 5.0, 4.0)])
    assert mv[p.name] == pytest.approx((1.0, -1.0)) and mv[q.name] == pytest.approx((1.0, 0.0))
    gs = m.grid_systems["G1"]
    gs.rotation = 90.0  # local x = global y
    r = m.add_point(-3.0, 2.0, 3.0)  # local (2, 3)
    mv = glue_moves(m, [("G1", "X", 2.0, 2.5)])
    assert mv[r.name] == pytest.approx((0.0, 0.5))


def test_pick_a_line_and_a_bubble_in_plan():
    doc = _doc()
    m = doc.model
    assert line_at(m, (6.05, 2.0), 0.1) == ("G1", "B")
    assert line_at(m, (3.0, 2.0), 0.1) is None
    geo = grid_geometry(m)
    end, d, name = next(b for b in geo.bubbles if b[2] == "B")
    c = (end[0] + d[0] * 1.0, end[1] + d[1] * 1.0)
    assert bubble_at(m, c, lambda gm: (0.5, 1.0)) == ("G1", "B")
    assert bubble_at(m, (c[0] + 2.0, c[1]), lambda gm: (0.5, 1.0)) is None


def test_towers_draw_into_the_tower_of_the_plan():
    """Several towers: the line goes to the tower's own system (a new T2-G1 when it has
    none), named within it (T2 starts at A again)."""
    doc = _doc()
    m = doc.model
    m.allow_towers(True)
    m.add_tower("T2")
    assert draw_grid_line(doc, (40.0, 0.0), (40.0, 5.0), tower="T2") == "A"
    g = _line(doc.model, "A", "T2-G1")
    assert g.direction == "X" and g.ordinate == 40.0
    assert doc.model.tower_of_system("T2-G1") == "T2"
    assert draw_grid_line(doc, (18.0, 0.0), (18.0, 5.0), tower="T1") == "D"  # T1: after A, B, C


def test_move_and_copy_several_lines_by_a_vector():
    from structstudio.app.grid_tools import move_grid_lines
    doc = _doc()
    name = draw_grid_line(doc, (0.0, 0.0), (6.0, 5.0))
    made = move_grid_lines(doc, [("G1", "B"), ("G1", "2"), ("G1", name)], 1.0, -2.0, copy=True)
    m = doc.model
    assert sorted(n for _, n in made) == sorted(["B'", "2'", name + "'"])
    assert _line(m, "B'").ordinate == 7.0 and _line(m, "2'").ordinate == 3.0
    assert tuple(_line(m, name + "'").start) == pytest.approx((1.0, -2.0))
    assert _line(m, "B").ordinate == 6.0  # (copies: the originals stay)
    move_grid_lines(doc, [("G1", "B"), ("G1", "C")], 6.0, 0.0)  # B onto C while C moves on: allowed
    assert (_line(doc.model, "B").ordinate, _line(doc.model, "C").ordinate) == (12.0, 18.0)
    with pytest.raises(ModelError):
        move_grid_lines(doc, [("G1", "A")], 7.0, 0.0)  # onto B' (7)
    doc.undo()
    assert (_line(doc.model, "B").ordinate, _line(doc.model, "C").ordinate) == (6.0, 12.0)


def test_grid_line_properties_edit():
    from structstudio.app.grid_tools import edit_grid_line
    doc = _doc()
    edit_grid_line(doc, "G1", "B", new_name="B2", ordinate=7.0, visible=False, bubble="Start")
    g = _line(doc.model, "B2")
    assert (g.ordinate, g.visible, g.bubble_location) == (7.0, False, "Start")
    with pytest.raises(ModelError):
        edit_grid_line(doc, "G1", "B2", ordinate=12.0)  # C is there
