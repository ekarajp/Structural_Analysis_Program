"""Headless rendering tests of the desktop scene builders (skipped without VTK)."""
import os

import pytest

vtk = pytest.importorskip("vtk")

from structstudio import analyze  # noqa: E402
from structstudio.model import FIXED, FrameSection, ShellSection  # noqa: E402
from structstudio import Model  # noqa: E402


def small_building():
    m = Model("t", "kN m")
    m.add_material("C30", E=27e6, nu=0.2, weight_density=24)
    m.add_frame_section(FrameSection.rectangular("B", "C30", 0.5, 0.3))
    m.add_shell_section(ShellSection("S", "C30", thickness=0.15))
    m.set_stories([("L1", 3.0)])
    m.add_diaphragm("D1")
    base = [m.add_point(x, y, 0, restraints=FIXED).name for x, y in ((0, 0), (5, 0), (5, 4), (0, 4))]
    top = [m.add_point(x, y, 3).name for x, y in ((0, 0), (5, 0), (5, 4), (0, 4))]
    for b, t in zip(base, top):
        m.add_frame(b, t, "B")
    for k in range(4):
        m.add_frame(top[k], top[(k + 1) % 4], "B")
    m.add_area(top, "S", diaphragm="D1", name="F1")
    m.add_load_pattern("DL", "Dead", 1.0)
    m.add_linear_static_case("DL")
    m.add_modal_case("Modal", max_modes=3)
    return m


def test_scene_renders(tmp_path):
    from structstudio.app import scene
    m = small_building()
    res = analyze(m)
    sc = scene.build_model_actors(m, show_labels=True)
    assert sc.actors and len(sc.frame_ids) == 8
    deformed, scale = scene.build_deformed_actor(m, res, "DL")
    assert scale > 0
    diagrams, _ = scene.build_diagram_actors(m, res, "DL", "M3")
    mode_actor, _ = scene.build_deformed_actor(m, res, "Modal", mode=1)
    out = scene.render_to_png(sc.all() + [deformed] + diagrams, str(tmp_path / "scene.png"))
    assert os.path.getsize(out) > 1000
    # deflection view: the analysis mesh, members bent by their end rotations, a colour scale
    actors, sc2 = scene.build_deformed_mesh_actors(m, res, "DL")
    import vtk
    assert sc2 > 0 and any(isinstance(a, vtk.vtkScalarBarActor) for a in actors)
    tubes = [a for a in actors if isinstance(a, vtk.vtkActor)][-1].GetMapper().GetInput()
    assert tubes.GetNumberOfPoints() > 0
    mode_actors, _ = scene.build_deformed_mesh_actors(m, res, "Modal", mode=1)
    assert mode_actors
    out = scene.render_to_png(sc.all() + actors, str(tmp_path / "deflection.png"))
    assert os.path.getsize(out) > 1000


def test_shell_contours_reactions_loads(tmp_path):
    from structstudio.app import scene
    from structstudio.app.panels import export_rows
    m = small_building()
    m.add_load_pattern("LL", "Live")
    m.add_area_load("F1", "LL", 2.5)
    beam = next(f.name for f in m.frames.values() if f.type == "Beam")
    m.add_frame_distributed_load(beam, "LL", -5.0, -8.0)
    m.add_point_load(m.frames[beam].point_i, "LL", (10.0, 0.0, 0.0))
    m.add_linear_static_case("LL")
    m.add_combination("U", {"DL": 1.2, "LL": 1.6})
    res = analyze(m, cases=["DL", "LL"])
    rows = res.shell_forces("U")
    assert rows and {"F11", "M11", "V13"} <= set(rows[0])
    vals, nodes = res.shell_node_values("U", "M11")
    assert vals.shape[1] == 4 and nodes.shape == vals.shape
    # linear-add combination = 1.2 DL + 1.6 LL at every element node
    v_dl, _ = res.shell_node_values("DL", "M11")
    v_ll, _ = res.shell_node_values("LL", "M11")
    assert abs(vals - (1.2 * v_dl + 1.6 * v_ll)).max() < 1e-9
    actors, (vmin, vmax) = scene.build_contour_actors(m, res, "U", "M11", deformed=True)
    assert len(actors) == 2 and vmax > vmin
    assert scene.build_reaction_actors(m, res, "DL")
    # load diagram fill, lines (shafts), solid arrow heads and the labels (with units)
    assert len(scene.build_load_actors(m, "LL")) == 4
    export_rows(rows, str(tmp_path / "shell.csv"))
    assert (tmp_path / "shell.csv").read_text().startswith("Area,Element,Node,Case,F11")



def _label_texts(actor):
    hier = actor.GetMapper().GetInputConnection(0, 0).GetProducer()
    arr = hier.GetInput().GetPointData().GetAbstractArray("labels")
    return [arr.GetValue(k) for k in range(arr.GetNumberOfValues())]


def test_load_labels_and_plan_symbols():
    from structstudio.app import scene
    assert [scene.fmt_value(v) for v in (12.5, -30.0, 0.75, 1 / 3, 2.0)] == ["12.5", "-30", "0.75", "0.3333", "2"]
    assert Model("u").units.label("force_per_length") == "kN/m"
    assert Model("u").units.label("moment") == "kN-m"
    m = small_building()
    beam = next(f.name for f in m.frames.values() if f.type == "Beam")
    m.add_frame_distributed_load(beam, "DL", 12.5)
    m.add_point_load(m.frames[beam].point_i, "DL", (0.0, 0.0, -50.0))
    m.add_area_load("F1", "DL", 2.5)
    for view in ("3d", "plan"):
        actors = scene.build_load_actors(m, "DL", view=view, level=3.0)
        texts = _label_texts(actors[-1])
        assert "12.5 kN/m" in texts and "50 kN" in texts and "2.5 kN/m\u00b2" in texts, texts
        # load diagram fill, lines, solid heads (arrows in 3D, the force-into-screen
        # symbol and the diagram turned into the plan), labels
        assert len(actors) == 4
    # a free panel of another level is not drawn in this plan
    from structstudio.model.objects import PanelLoadCase
    m.add_panel_load([(0, 0, 9.0), (5, 0, 9.0), (5, 4, 9.0)], "two_way", loads=[PanelLoadCase("DL", 7.0)])
    assert "PL2: 7 kN/m\u00b2 (2-way)" in _label_texts(scene.build_load_actors(m, "DL", view="3d")[-1])
    assert not any("PL2" in t for t in _label_texts(scene.build_load_actors(m, "DL", view="plan", level=3.0)[-1]))
