"""Drawing tools and templates (no window needed)."""
import pytest

from structstudio import analyze
from structstudio.app.document import Document
from structstudio.app.templates import new_building
from structstudio.app.tools import (BeamTool, ColumnTool, DrawSettings, QuickBeamTool, QuickColumnTool,
                                    QuickSlabTool, SlabTool, Snapper, WallTool, delete_objects, grid_intersections,
                                    objects_in_window)


@pytest.fixture
def doc():
    return Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5)))


def test_new_building_template_grid_only(doc):
    m = doc.model
    assert [s.name for s in m.stories_bottom_up()] == ["Base", "Story1", "Story2"]
    assert len(grid_intersections(m)) == 6
    assert "C40x40" in m.frame_sections and "S15" in m.shell_sections
    assert not m.frames


def test_new_building_full_frame_analyses():
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5), grid_only=False)
    assert len([f for f in m.frames.values() if f.type == "Column"]) == 12
    assert len(m.areas) == 4
    res = analyze(m, cases=["Dead", "Modal"])
    assert res.base_reactions("Dead")[2] > 0
    assert res.modal_periods("Modal")[0]["Period"] > 0


def test_snapper_prefers_points_then_grid(doc):
    m = doc.model
    p = m.add_point(6.0, 5.0, 3.5)
    s = Snapper(m, "Story1", tolerance=0.3)
    assert s.snap((6.1, 4.9)).kind == "point" and s.snap((6.1, 4.9)).target == p.name
    assert s.snap((0.1, 0.1)).kind == "grid" and s.snap((0.1, 0.1)).xy == (0.0, 0.0)
    r = s.snap((3.0, 0.1))  # orthogonal snap onto grid line y=0
    assert r.xy == (3.0, 0.0)
    assert s.snap((3.0, 2.0)).kind == "free"


def test_beam_tool_chains_and_undo(doc):
    t = BeamTool(doc, DrawSettings(frame_section="B30x60"))
    t.set_story("Story1")
    assert t.click((0, 0)) is False
    assert t.click((6, 0)) is True
    assert t.click((12, 0)) is True  # chained from the last point
    beams = [f for f in doc.model.frames.values() if f.type == "Beam"]
    assert len(beams) == 2 and all(f.story == "Story1" for f in beams)
    assert all(doc.model.points[f.point_i].z == 3.5 for f in beams)
    doc.undo()
    assert len(doc.model.frames) == 1


def test_column_tool_all_stories(doc):
    t = ColumnTool(doc, DrawSettings(frame_section="C40x40", all_stories=True))
    t.set_story("Story2")
    assert t.click((6, 5))
    cols = list(doc.model.frames.values())
    assert len(cols) == 2 and {c.story for c in cols} == {"Story1", "Story2"}
    assert all(c.type == "Column" for c in cols)


def test_slab_and_wall_tools(doc):
    s = SlabTool(doc, DrawSettings(shell_section="S15", diaphragm="D1"))
    s.set_story("Story1")
    for xy in ((0, 0), (6, 0), (6, 5)):
        assert s.click(xy) is False
    assert s.click((0, 5)) is False
    assert s.finish() is True
    a = next(iter(doc.model.areas.values()))
    assert a.type == "Floor" and len(a.points) == 4 and a.diaphragm == "D1"
    w = WallTool(doc, DrawSettings(wall_section="W20"))
    w.set_story("Story1")
    w.click((0, 0))
    assert w.click((6, 0))
    wall = [x for x in doc.model.areas.values() if x.type == "Wall"][0]
    zs = sorted({doc.model.points[p].z for p in wall.points})
    assert zs == [0.0, 3.5] and wall.section == "W20"


def test_quick_tools(doc):
    m = doc.model
    qb = QuickBeamTool(doc, DrawSettings(frame_section="B30x60"))
    qb.tolerance = 0.5
    qb.set_story("Story1")
    assert qb.click((3.0, 0.2))  # on grid line 1 between A and B
    assert qb.click((3.0, 0.2)) is True and len(m.frames) == 1  # no duplicate
    f = next(iter(m.frames.values()))
    assert {m.points[f.point_i].x, m.points[f.point_j].x} == {0.0, 6.0}
    qc = QuickColumnTool(doc, DrawSettings(frame_section="C40x40"))
    qc.tolerance = 0.5
    qc.set_story("Story1")
    assert qc.click((6.2, 5.1))
    assert [c for c in m.frames.values() if c.type == "Column"][0].point_j == m.find_point(6, 5, 3.5).name
    qs = QuickSlabTool(doc, DrawSettings(shell_section="S15"))
    qs.set_story("Story2")
    assert qs.click((2.0, 2.0))
    assert qs.click((20.0, 2.0)) is False  # outside the grid
    a = next(iter(m.areas.values()))
    assert sorted((m.points[p].x, m.points[p].y) for p in a.points) == [(0, 0), (0, 5), (6, 0), (6, 5)]


def test_window_selection_and_delete(doc):
    m = doc.model
    t = BeamTool(doc, DrawSettings(frame_section="B30x60"))
    t.set_story("Story1")
    t.click((0, 0)); t.click((6, 0)); t.click((12, 0))
    frames, areas, points = objects_in_window(m, lambda p: p.x <= 6.5, story="Story1")
    assert len(frames) == 1 and len(points) == 2
    assert delete_objects(doc, frames=frames, points=points) == 1
    assert len(m.frames) == 1 and len(m.points) == 2  # shared point kept
    doc.undo()
    assert len(doc.model.frames) == 2


def test_edit_commands(doc):
    from structstudio.app.tools import divide_frames, move_objects, replicate_linear, replicate_to_stories
    m = doc.model
    t = BeamTool(doc, DrawSettings(frame_section="B30x60"))
    t.set_story("Story1")
    t.click((0, 0)); t.click((6, 0))
    beam = next(iter(doc.model.frames))
    doc.model.add_frame_distributed_load(beam, "Dead", -10.0)
    doc.model.points[doc.model.frames[beam].point_i].restraints = (True,) * 6
    # replicate along Y, twice
    new = replicate_linear(doc, [beam], [], [], 0, 5, 0, count=2)
    assert len(new) == 2 and len(doc.model.frames) == 3
    f2 = doc.model.frames[new[1]]
    assert doc.model.points[f2.point_i].y == 10 and f2.loads[0].value_a == -10.0
    assert all(doc.model.points[f2.point_i].restraints)  # restraint copied
    # replicate to Story2
    new = replicate_to_stories(doc, [beam], [], [], ["Story2"])
    assert len(new) == 1 and doc.model.frames[new[0]].story == "Story2"
    assert doc.model.points[doc.model.frames[new[0]].point_j].z == 7.0
    # move the first beam up by 1 (shared point moves too)
    n = move_objects(doc, [beam], [], [], 0, 0, 1.0)
    assert n == 2 and doc.model.points[doc.model.frames[beam].point_j].z == 4.5
    # divide into 3
    pieces = divide_frames(doc, [beam], 3)
    assert len(pieces) == 3 and beam not in doc.model.frames
    xs = sorted(doc.model.points[doc.model.frames[p].point_i].x for p in pieces)
    assert xs == [0.0, 2.0, 4.0]
    assert all(doc.model.frames[p].loads[0].value_a == -10.0 for p in pieces)
    doc.undo()
    assert beam in doc.model.frames


def test_grid_modes_etabs_and_sap2000():
    from structstudio import analyze
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.0), grid_only=False)
    assert m.grid_mode == "story" and not m.z_grid_lines()
    ref = analyze(m, cases=["Dead"]).base_reactions("Dead")
    m.set_grid_mode("xyz")
    assert [(g.name, g.ordinate) for g in m.z_grid_lines()] == [("Base", 0.0), ("Story1", 3.5), ("Story2", 6.5)]
    assert m.levels() == [("Base", 0.0), ("Story1", 3.5), ("Story2", 6.5)]
    # analysis is unchanged by the switch
    assert abs(analyze(m, cases=["Dead"]).base_reactions("Dead") - ref).max() < 1e-9
    # SAP2000 style edit: add a mezzanine Z line -> a level (story) appears, objects re-assigned
    m.set_z_grid([("Base", 0), ("Story1", 3.5), ("Mezz", 5.0), ("Story2", 6.5)])
    assert [s.name for s in m.stories_bottom_up()] == ["Base", "Story1", "Mezz", "Story2"]
    assert m.stories["Story2"].height == 1.5
    # round trip through the project file keeps the mode
    from structstudio import Model
    m2 = Model.from_json(m.to_json())
    assert m2.grid_mode == "xyz" and len(m2.z_grid_lines()) == 4
    # back to ETABS: stories kept, Z lines removed
    m.set_grid_mode("story")
    assert m.grid_mode == "story" and not m.z_grid_lines()
    assert [s.name for s in m.stories_bottom_up()] == ["Base", "Story1", "Mezz", "Story2"]
    # SAP2000 template
    s = new_building(grid_mode="xyz", story_heights=(3, 3))
    assert s.levels() == [("Z1", 0.0), ("Z2", 3.0), ("Z3", 6.0)]


def test_grid_mode_switch_is_undoable(doc):
    from structstudio.app.dialogs import apply_grid_mode
    assert apply_grid_mode(doc, "xyz") and doc.model.grid_mode == "xyz"
    assert not apply_grid_mode(doc, "xyz")
    doc.undo()
    assert doc.model.grid_mode == "story" and not doc.model.z_grid_lines()
