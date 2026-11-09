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


def test_local_axes_numbers_sit_past_the_arrow_heads():
    import numpy as np
    from structstudio.app import theme as th
    m = _model()
    beam = next(f for f in m.frames.values() if f.type == "Beam")
    acts = scene._local_axes_actors(m, [beam])
    assert len(acts) == 6  # per axis: the arrow (shaft + head) and its number
    for k in range(3):
        arrow, label = acts[2 * k], acts[2 * k + 1]
        assert arrow.GetProperty().GetColor() == pytest.approx(scene.COLORS[f"axis_{k + 1}"])
        pts = arrow.GetMapper().GetInput().GetPoints()
        P = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
        mid, tip = P[0], P[1]
        spot = np.array(label.GetMapper().GetInputConnection(0, 0).GetProducer().GetInput().GetPoint(0))
        # the number is beyond the tip, on the axis, by the "distance past the arrow tip" setting
        assert np.linalg.norm(spot - mid) > np.linalg.norm(tip - mid) * 1.3
    # bigger with the length setting
    s = th.ColorSettings()
    s.set_size("local_axes_length", 3.0)
    th.apply(s)
    try:
        big = scene._local_axes_actors(m, [beam])[0].GetMapper().GetInput().GetPoints()
        assert np.linalg.norm(np.subtract(big.GetPoint(1), big.GetPoint(0))) == pytest.approx(
            2.0 * np.linalg.norm(np.subtract(pts.GetPoint(1), pts.GetPoint(0))))
    finally:
        th.apply(th.ColorSettings())


def test_hidden_columns_leave_the_plan_window():
    """Show only this object: a hidden column's plan marker and support go too."""
    m = _model()
    story = next(s for s in m.stories if s != "Base")
    beam = next(f.name for f in m.frames.values() if f.type == "Beam" and f.story == story)
    plain = scene.build_model_actors(m, story=story, options=DisplayOptions()).actors
    assert "column_markers" in plain
    hidden = (set(m.frames) - {beam}, set(m.areas), set(m.points) - {m.frames[beam].point_i, m.frames[beam].point_j})
    acts = scene.build_model_actors(m, story=story, options=DisplayOptions(), hidden=hidden).actors
    assert not any(k.startswith("column_markers") for k in acts) and "supports" not in acts
    assert "frames_Beam" in acts


def _with_wall_and_opening():
    import numpy as np
    m = _model()
    st = sorted(m.stories.values(), key=lambda s: s.elevation)
    z0, z1 = st[0].elevation, st[1].elevation
    wall = m.add_area([m.add_point(0, 0, z0), m.add_point(6, 0, z0), m.add_point(6, 0, z1), m.add_point(0, 0, z1)],
                      "W20", type="Wall")
    fl = next(a for a in m.areas.values() if a.type == "Floor")
    P = np.array([m.points[p].xyz for p in fl.points])
    c, z = P.mean(0), P[0, 2]
    m.add_area([m.add_point(c[0] - 1, c[1] - 1, z), m.add_point(c[0] + 1, c[1] - 1, z),
                m.add_point(c[0] + 1, c[1] + 1, z), m.add_point(c[0] - 1, c[1] + 1, z)], None, type="Opening")
    return m, wall, fl, st[1].name


def test_walls_show_their_thickness_in_plan_only():
    m, wall, _fl, story = _with_wall_and_opening()
    fp = scene.wall_footprint(m, m.areas[wall.name])
    xs, ys = sorted({round(x, 9) for x, _ in fp}), sorted({round(y, 9) for _, y in fp})
    t = m.shell_sections["W20"].thickness
    assert xs == [0.0, 6.0] and ys == [round(-t / 2, 9), round(t / 2, 9)]
    plan = scene.build_model_actors(m, story=story, options=DisplayOptions()).actors
    assert "wall_plan" in plan and "wall_plan_edges" in plan
    assert not any(k.startswith("wall_plan") for k in scene.build_model_actors(m, options=DisplayOptions()).actors)
    assert not any(k.startswith("wall_plan") for k in
                   scene.build_model_actors(m, plane=("y", 0.0), options=DisplayOptions()).actors)


def test_openings_go_through_floors_extruded_or_not():
    import vtk
    from structstudio.app.extrude import area_solids
    m, _wall, fl, _story = _with_wall_and_opening()
    acts = scene.build_model_actors(m, options=DisplayOptions()).actors
    assert "areas_Opening" not in acts and "areas_Opening_edges" in acts  # no grey fill: a hole
    solid = area_solids(m, [m.areas[fl.name]])
    mp = vtk.vtkMassProperties()
    mp.SetInputData(solid)
    mp.Update()
    import numpy as np
    P = np.array([m.points[p].xyz for p in m.areas[fl.name].points])
    t = m.shell_sections[fl.section].thickness
    full = np.ptp(P[:, 0]) * np.ptp(P[:, 1]) * t
    assert mp.GetVolume() == pytest.approx(full - 4.0 * t, rel=1e-9)  # the 2 x 2 hole is cut out


def test_view_plane_of_plan_and_elevation():
    m = _model()
    plan = scene.build_view_plane(m, "plan", 3.0, "Story1")
    b = plan[0].GetMapper().GetInput().GetBounds()
    assert b[4] == b[5] == 3.0
    elev = scene.build_view_plane(m, "xz", 5.0, "2")
    b = elev[0].GetMapper().GetInput().GetBounds()
    assert b[2] == b[3] == 5.0 and b[5] - b[4] > 6.0  # vertical, over the full height
    assert scene.build_view_plane(m, "yz", 6.0, "B")[0].GetMapper().GetInput().GetBounds()[0] == 6.0


def test_selected_area_highlight_sits_inside_the_area():
    import numpy as np
    m = _model()
    fl = next(a for a in m.areas.values() if a.type == "Floor")
    P = np.array([m.points[p].xyz for p in fl.points])
    R = scene.inset_polygon(P, m)
    assert R.shape == P.shape
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    assert np.all(R[:, :2] > lo + 1e-6) and np.all(R[:, :2] < hi - 1e-6)  # strictly inside
    assert np.allclose(R[:, 2], P[:, 2])  # in the floor's plane
    assert scene.build_model_actors(m, selected_areas={fl.name}, options=DisplayOptions()).actors["areas_sel"]
