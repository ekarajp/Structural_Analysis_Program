"""Display options of a model window (ETABS "Set Display Options")."""
import pytest

pytest.importorskip("vtk")

from structstudio.app import scene  # noqa: E402
from structstudio.app.display import DisplayOptions  # noqa: E402
from structstudio.app.templates import new_building  # noqa: E402


def _model():
    return new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3, 3), grid_only=False)


def test_options_roundtrip_and_defaults():
    o = DisplayOptions(columns=False, grid_3d="all")
    assert DisplayOptions.from_dict(o.to_dict()) == o
    assert DisplayOptions.from_dict({"grid_3d": "bogus", "nonsense": 1}).grid_3d == "all"
    # windows saved before v3 (grid at the base only) open with the grid at every story
    assert DisplayOptions.from_dict({"version": 2, "grid_3d": "base"}).grid_3d == "all"
    assert DisplayOptions.from_dict(DisplayOptions(grid_3d="base").to_dict()).grid_3d == "base"
    assert DisplayOptions().story_plane and not DisplayOptions().extrude
    assert DisplayOptions().joints and DisplayOptions().releases  # joint squares and pin rings shown (v4)
    assert DisplayOptions.from_dict({"version": 3, "joints": False}).joints  # older windows: joints shown too
    assert DisplayOptions().grid_vertical
    # windows saved before v2 (extrusion on by default) open as line views
    assert not DisplayOptions.from_dict({"extrude": True, "columns": False}).extrude
    assert DisplayOptions.from_dict(DisplayOptions(extrude=True).to_dict()).extrude


def test_vertical_grid_lines_in_3d_only():
    m = _model()
    keys = set(scene.build_model_actors(m, options=DisplayOptions()).actors)
    assert "grid_vertical" in keys
    vert = scene.build_model_actors(m, options=DisplayOptions()).actors["grid_vertical"]
    b = vert.GetMapper().GetInput().GetBounds()
    assert (b[4], b[5]) == (0.0, 6.0)
    assert vert.GetMapper().GetInput().GetNumberOfLines() == 3 * 2  # 3 x-lines * 2 y-lines
    story = next(iter(m.stories))
    assert "grid_vertical" not in scene.build_model_actors(m, story=story, options=DisplayOptions()).actors
    assert "grid_vertical" not in scene.build_model_actors(m, options=DisplayOptions(grid_vertical=False)).actors
    assert "grid_vertical" not in scene.build_model_actors(m, options=DisplayOptions(grid_lines=False)).actors


def test_object_visibility_labels_and_assignments():
    m = _model()
    keys = set(scene.build_model_actors(m, options=DisplayOptions()).actors)
    assert {"frames_Column", "frames_Beam", "supports", "grids", "grid_bubbles"} <= keys
    o = DisplayOptions(columns=False, floors=False, supports=False, joints=True, joint_labels=True,
                       frame_labels=True, area_labels=True, local_axes=True, grid_bubbles=False)
    keys = set(scene.build_model_actors(m, options=o).actors)
    assert not any(k.startswith("frames_Column") for k in keys)
    assert not any(k.startswith("areas_Floor") for k in keys)
    assert "supports" not in keys and "grid_bubbles" not in keys
    assert {"joints", "joint_labels", "labels", "local_axes_0"} <= keys
    assert "area_labels" not in keys  # floors hidden: no floor labels either
    assert "area_labels" in scene.build_model_actors(m, options=DisplayOptions(area_labels=True)).actors
    f = next(iter(m.frames.values()))
    f.releases_j = (False, False, False, False, False, True)
    assert "releases" in scene.build_model_actors(m, options=DisplayOptions(releases=True)).actors
    no_fill = scene.build_model_actors(m, options=DisplayOptions(fill_areas=False, extrude=False)).actors
    assert "areas_Floor" not in no_fill and "areas_Floor_edges" in no_fill


def test_3d_grid_level_and_story_plane():
    m = _model()
    assert scene.grid_levels(m, "base") == [0.0]
    assert scene.grid_levels(m, "top") == [6.0]
    assert scene.grid_levels(m, "all") == [0.0, 3.0, 6.0]
    acts = scene.build_model_actors(m, options=DisplayOptions()).actors  # default: every story
    assert acts["grids"].GetMapper().GetInput().GetBounds()[4:6] == (0.0, 0.0)  # base, strong
    assert acts["grids_upper"].GetMapper().GetInput().GetBounds()[4:6] == (3.0, 6.0)  # stories above
    assert acts["grids_upper"].GetProperty().GetOpacity() < acts["grids"].GetProperty().GetOpacity()
    acts = scene.build_model_actors(m, options=DisplayOptions(grid_3d="base")).actors
    assert "grids_upper" not in acts
    assert acts["grids"].GetMapper().GetInput().GetBounds()[4:6] == (0.0, 0.0)
    story = [s.name for s in m.stories_bottom_up()][1]
    plane = scene.build_story_plane(m, story)
    assert len(plane) == 3
    assert plane[0].GetMapper().GetInput().GetBounds()[4:6] == (3.0, 3.0)
    assert scene.build_story_plane(m, None) == [] and scene.build_story_plane(m, "nope") == []


def test_reference_lines_through_joints():
    """CYPE 3D reference lines: through every joint, own colour, dashed, per direction."""
    COLORS = scene.COLORS
    m = _model()
    assert "reference_lines" not in scene.build_model_actors(m, options=DisplayOptions()).actors
    acts = scene.build_model_actors(m, options=DisplayOptions(reference_lines=True)).actors
    ra = acts["reference_lines"]
    assert tuple(round(c, 3) for c in ra.GetProperty().GetColor()) == tuple(round(c, 3) for c in COLORS["reference"])
    assert ra.GetProperty().GetColor() != acts["grids"].GetProperty().GetColor()
    b = ra.GetMapper().GetInput().GetBounds()
    assert b[4] < 0.0 < 6.0 < b[5]  # vertical lines run past the model extent
    # plan: no vertical lines, only X / Y lines at the story elevation
    story = [s.name for s in m.stories_bottom_up()][1]
    acts = scene.build_model_actors(m, story=story, options=DisplayOptions(reference_lines=True, reference_y=False)).actors
    b = acts["reference_lines"].GetMapper().GetInput().GetBounds()
    assert b[4] == b[5] == 3.0  # in the story plane
    assert b[1] - b[0] > 12.0 and b[3] - b[2] == 5.0  # X-parallel lines at y = 0 and y = 5 only
    acts = scene.build_model_actors(m, story=story, options=DisplayOptions(reference_lines=True, reference_x=False,
                                                                           reference_y=False)).actors
    assert "reference_lines" not in acts  # Z lines never appear in a plan


def test_joint_and_area_labels_and_grid_bubbles_build():
    """The joint / area labels have their text array (they showed nothing before), the
    grid names sit in circles."""
    import vtk
    from structstudio.app import scene
    m = _model()
    o = DisplayOptions(joint_labels=True, area_labels=True)
    sc = scene.build_model_actors(m, options=o)
    for key in ("joint_labels", "area_labels"):
        mapper = sc.actors[key].GetMapper()
        mapper.Update() if hasattr(mapper, "Update") else None
        hier = mapper.GetInputAlgorithm()
        assert hier.GetLabelArrayName() == "labels"
        assert hier.GetInput().GetPointData().GetAbstractArray("labels") is not None
    bub = sc.actors.get("grid_bubbles")
    assert isinstance(bub, vtk.vtkActor) and bub.GetMapper().GetInput().GetNumberOfCells() > 0
