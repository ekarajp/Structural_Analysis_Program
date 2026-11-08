"""Edit > Edit Shells: Merge Shells, Expand/Shrink Shells, Divide Walls for Openings,
and meshing of the non-convex (L-shaped) shells that Merge Shells produces."""
import numpy as np
import pytest

from structstudio.api import analyze
from structstudio.app import shell_edit as SE
from structstudio.app.document import Document
from structstudio.app.templates import new_building
from structstudio.mesh.mesher import MeshOptions, Mesher, build_analysis_model
from structstudio.mesh import polygon2d as pg


def _doc():
    return Document(new_building(x_spacings=(6, 6), y_spacings=(5, 5), story_heights=(3.5,), grid_only=False))


def _floor_at(m, x, y):
    for a in m.areas.values():
        c = np.array([m.points[p].xyz for p in a.points]).mean(axis=0)
        if a.type == "Floor" and abs(c[0] - x) < 1e-6 and abs(c[1] - y) < 1e-6:
            return a.name


def _xy(m, a):
    return [tuple(m.points[p].xyz[:2]) for p in m.areas[a].points]


def _l_shape_model():
    """Three floors forming an L, interior beams removed so the slab spans."""
    d = _doc()
    m = d.model
    for f in list(m.frames.values()):
        a, b = m.points[f.point_i], m.points[f.point_j]
        if f.type == "Beam" and (a.x == b.x == 6 or a.y == b.y == 5):
            del m.frames[f.name]
    del m.areas[_floor_at(m, 9, 7.5)]
    for n, a in m.areas.items():
        if a.type == "Floor":
            m.add_area_load(n, "Live", 3.0)
    return d


def test_merge_two_floors_keeps_first_name_and_shared_edge_disappears():
    d = _doc()
    m = d.model
    f1, f2 = _floor_at(m, 3, 2.5), _floor_at(m, 3, 7.5)
    done, notes = SE.merge_shells(d, [f1, f2])
    m = d.model
    assert done == [f1] and not notes and f2 not in m.areas
    xs = sorted(set(_xy(m, f1)))
    # the joints at y = 5 stay: the beams of grid line 2 use them
    assert xs == [(0.0, 0.0), (0.0, 5.0), (0.0, 10.0), (6.0, 0.0), (6.0, 5.0), (6.0, 10.0)]
    assert d.undo() and f2 in d.model.areas


def test_merge_removes_unused_mid_edge_joints_and_handles_t_junctions():
    d = _doc()
    m = d.model
    for f in list(m.frames.values()):
        if f.type == "Beam":
            del m.frames[f.name]
    for f in list(m.frames.values()):  # columns at y = 5 gone too: nothing holds those joints
        if abs(m.points[f.point_i].y - 5) < 1e-9:
            del m.frames[f.name]
    a, b = _floor_at(m, 3, 2.5), _floor_at(m, 3, 7.5)
    big = _floor_at(m, 9, 2.5)
    # T-junction: the right floor spans y 0..10 in one piece, its left edge has no joint at y = 5
    top = _floor_at(m, 9, 7.5)
    pts = [m.add_point(6, 0, 3.5).name, m.add_point(12, 0, 3.5).name,
           m.add_point(12, 10, 3.5).name, m.add_point(6, 10, 3.5).name]
    del m.areas[big], m.areas[top]
    r = m.add_area(pts, m.areas[a].section, type="Floor")
    done, notes = SE.merge_shells(d, [a, b, r.name])
    m = d.model
    assert done == [a] and not notes
    # (6, 0) and (6, 10) hold columns, the y = 5 joints hold nothing any more
    assert sorted(_xy(m, a)) == [(0.0, 0.0), (0.0, 10.0), (6.0, 0.0), (6.0, 10.0), (12.0, 0.0), (12.0, 10.0)]


def test_merge_refuses_holes_corners_and_different_sections():
    d = _doc()
    m = d.model
    f1, f4 = _floor_at(m, 3, 2.5), _floor_at(m, 9, 7.5)
    done, notes = SE.merge_shells(d, [f1, f4])  # only touch at the corner (6, 5)
    assert not done and any("corner" in n or "no edge" in n for n in notes)
    f2 = _floor_at(m, 3, 7.5)
    m.areas[f2].section = next(s for s in m.shell_sections if s != m.areas[f1].section)
    done, notes = SE.merge_shells(d, [f1, f2])
    assert not done


def test_merged_l_floor_deflects_like_its_pieces():
    mo = MeshOptions(floor_max_size=0.5)

    def max_uz(model):
        r = analyze(model, ["Live"], mesh_options=mo)
        return float(np.abs(r.displacement_envelope("Live").max[:, 2]).max())

    pieces = max_uz(_l_shape_model().model)
    d = _l_shape_model()
    floors = [a.name for a in d.model.areas.values() if a.type == "Floor"]
    done, _ = SE.merge_shells(d, floors)
    assert len(done) == 1 and len(d.model.areas[done[0]].points) > 4  # an L, one load panel
    assert sum(p.area == done[0] for p in d.model.panel_loads.values()) == 1
    assert max_uz(d.model) == pytest.approx(pieces, rel=1e-6)


def test_l_shaped_polygon_is_split_into_grid_rectangles():
    L = np.array([(0, 0, 0), (12, 0, 0), (12, 5, 0), (6, 5, 0), (6, 10, 0), (0, 10, 0)], float)
    quads, spans = Mesher._grid_patches(L, 1e-6)
    assert len(quads) == 3 and sorted(spans) == [(6, 5), (6, 5), (6, 5)]
    assert all(pg.signed_area(pg.to_2d(q, pg.plane_frame(L))) > 0 for q in quads)  # same orientation
    rect6 = np.array([(0, 0, 0), (6, 0, 0), (12, 0, 0), (12, 5, 0), (6, 5, 0), (0, 5, 0)], float)
    assert Mesher._grid_patches(rect6, 1e-6) is None  # convex: unchanged meshing


def test_expand_and_shrink_move_every_edge():
    d = _doc()
    f = _floor_at(d.model, 3, 2.5)
    done, _ = SE.expand_shells(d, [f], 0.5)
    assert done == [f]
    assert sorted(_xy(d.model, f)) == [(-0.5, -0.5), (-0.5, 5.5), (6.5, -0.5), (6.5, 5.5)]
    d.undo()
    SE.expand_shells(d, [f], -0.25)
    assert sorted(_xy(d.model, f)) == [(0.25, 0.25), (0.25, 4.75), (5.75, 0.25), (5.75, 4.75)]
    d.undo()
    done, notes = SE.expand_shells(d, [f], -3.0)  # 5 m wide: cannot shrink 3 m from each side
    assert not done and "inside out" in notes[0]


def _wall_with_window():
    d = Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5,), grid_only=False))
    m = d.model
    w = m.add_area([m.add_point(0, 0, 0), m.add_point(6, 0, 0), m.add_point(6, 0, 3.5), m.add_point(0, 0, 3.5)],
                   "W20", type="Wall")
    w.pier = "P1"
    o = m.add_area([m.add_point(2, 0, 1), m.add_point(4, 0, 1), m.add_point(4, 0, 2.5), m.add_point(2, 0, 2.5)],
                   None, type="Opening")
    return d, w.name, o.name


def test_divide_wall_for_openings_leaves_a_hole_and_meshes_compatibly():
    d, w, o = _wall_with_window()
    made, notes = SE.divide_for_openings(d, [o])  # selecting the opening finds its wall
    m = d.model
    assert len(made) == 8 and not notes and w not in m.areas and o in m.areas
    assert {m.areas[n].pier for n in made} == {"P1"} and {m.areas[n].type for n in made} == {"Wall"}
    area = 0.0
    for n in made:
        P = np.array([m.points[p].xyz for p in m.areas[n].points])
        area += abs(pg.signed_area(pg.to_2d(P, pg.plane_frame(P))))
    assert area == pytest.approx(6 * 3.5 - 2 * 1.5)
    am = build_analysis_model(m, MeshOptions(wall_max_size=0.5))
    count = {}
    for s in am.shells:
        if s.object in made:
            for i in range(4):
                k = frozenset((s.nodes[i], s.nodes[(i + 1) % 4]))
                count[k] = count.get(k, 0) + 1
    free = sum(np.linalg.norm(am.nodes[a].xyz - am.nodes[b].xyz)
               for k, c in count.items() if c == 1 for a, b in [tuple(k)])
    assert free == pytest.approx(2 * (6 + 3.5) + 2 * (2 + 1.5))  # outline + opening: no hanging nodes
    assert d.undo() and w in d.model.areas


def test_divide_for_openings_reports_shells_without_openings():
    d, w, o = _wall_with_window()
    f = next(a.name for a in d.model.areas.values() if a.type == "Floor")
    made, notes = SE.divide_for_openings(d, [f])
    assert not made and "no openings" in notes[0]


# ------------------------------------------------ openings cut holes automatically
def _floor_mesh_area(m, am, story):
    tot = 0.0
    for s in am.shells:
        a = m.areas[s.object]
        if a.type == "Floor" and a.story == story:
            P = np.array([am.nodes[n].xyz for n in s.nodes])
            tot += abs(pg.signed_area(P[:, :2]))
    return tot


def _two_story():
    return new_building(x_spacings=(6, 6), y_spacings=(5, 5), story_heights=(3.5, 3.5), grid_only=False)


def _opening(m, xy, z):
    return m.add_area([m.add_point(x, y, z) for x, y in xy], None, type="Opening")


def test_opening_over_a_wall_is_a_hole_without_dividing():
    def top_ux(divide):
        d, w, o = _wall_with_window()
        if divide:
            SE.divide_for_openings(d, [w])
        m = d.model
        p = m.find_point(0, 0, 3.5).name
        m.add_point_load(p, "Live", force=(100, 0, 0))
        mo = MeshOptions(wall_max_size=0.25, floor_max_size=1.0)  # wall top nodes on floor nodes
        am = build_analysis_model(m, mo)
        assert not am.warnings
        return analyze(m, ["Live"], mesh_options=mo).joint_displacement(p, "Live")[0], len(am.shells)
    auto, divided = top_ux(False), top_ux(True)
    assert auto[0] == pytest.approx(divided[0], rel=1e-9) and auto[1] == divided[1]


def test_floor_openings_exact_notch_and_whole_bay():
    mo = MeshOptions(floor_max_size=0.5)
    m = _two_story()
    _opening(m, [(1, 1), (3, 1), (3, 4), (1, 4)], 3.5)  # stair opening inside a bay
    _opening(m, [(10, -1), (13, -1), (13, 2), (10, 2)], 3.5)  # runs past the slab edge
    _opening(m, [(6, 5), (12, 5), (12, 10), (6, 10)], 7.0)  # a whole bay of the roof
    am = build_analysis_model(m, mo)
    assert _floor_mesh_area(m, am, "Story1") == pytest.approx(120 - 6 - 4)
    assert _floor_mesh_area(m, am, "Story2") == pytest.approx(90)
    assert any("covered by openings" in w for w in am.warnings)
    used = {n for s in am.shells for n in s.nodes} | {n for f in am.frames for n in f.nodes}
    loose = [nd.id for nd in am.nodes if nd.id not in used and not nd.is_master and nd.point is None]
    assert not loose  # no mesh nodes left inside the holes
    r = analyze(m, ["Dead", "Modal"], mesh_options=mo)  # fine mesh: > 3000 DOFs, rigid diaphragms
    assert r._case("Modal").modal.periods[0] > 0


def test_non_rectangular_opening_is_cut_exactly():
    m = _two_story()
    oc = [(3 + np.cos(t), 2.5 + np.sin(t)) for t in np.linspace(0, 2 * np.pi, 9)[:-1]]
    _opening(m, oc, 3.5)
    am = build_analysis_model(m, MeshOptions(floor_max_size=0.25))
    exact = 120 - 2 * np.sqrt(2)  # the octagon is cut out exactly (freeform mesh around it)
    assert _floor_mesh_area(m, am, "Story1") == pytest.approx(exact, rel=1e-9)
    assert not am.warnings


def test_diaphragm_inertia_leaves_the_opening_out():
    from structstudio.mesh.geometry import polygon_polar_inertia
    m = _two_story()
    _opening(m, [(1, 1), (3, 1), (3, 4), (1, 4)], 3.5)
    mesher = Mesher(m, MeshOptions(floor_max_size=1.0))
    mesher.build()
    pieces = [p for k, v in mesher._diaphragm_areas.items() if k[1] == "Story1" for p, *_ in v]
    c = np.array([6.0, 5.0, 3.5])
    j = sum(polygon_polar_inertia(q, c) for ps in pieces for q in ps)
    full = 12 * 10 * (12 ** 2 + 10 ** 2) / 12
    hole = 2 * 3 * (2 ** 2 + 3 ** 2) / 12 + 6 * ((2 - 6) ** 2 + (2.5 - 5) ** 2)
    assert j == pytest.approx(full - hole)


def test_opening_tool_rectangle_in_plan_and_elevation():
    from structstudio.app.tools import DrawSettings, OpeningTool, Plane
    d = Document(_two_story())
    t = OpeningTool(d, DrawSettings())
    t.set_story("Story1")
    assert t.click((1, 1)) is False and t.click((3, 4)) is True
    o = d.model.areas[t.created[-1]]
    assert o.type == "Opening" and sorted(d.model.points[p].xyz for p in o.points)[0] == (1.0, 1.0, 3.5)
    t.set_plane(Plane("xz", 0.0, None))  # elevation of grid line 1: a door
    t.click((2, 0))
    p = t.typed("1.2,2.2")
    assert np.allclose(p, (3.2, 0.0, 2.2))
    assert t.click(p) is True
    door = d.model.areas[t.created[-1]]
    assert sorted(d.model.points[q].xyz for q in door.points) == [(2.0, 0.0, 0.0), (2.0, 0.0, 2.2),
                                                                  (3.2, 0.0, 0.0), (3.2, 0.0, 2.2)]
    t.settings.opening_shape = "Polygon"
    for q in ((7, 1), (9, 1), (8, 3)):
        t.click(q)
    assert t.finish() is True and len(d.model.areas[t.created[-1]].points) == 3


def test_drawing_cuts_the_opening_out_of_the_fill():
    from structstudio.mesh import openings as opn
    P = np.array([(0, 0, 0), (6, 0, 0), (6, 0, 3.5), (0, 0, 3.5)], float)
    m = _two_story()
    idx = opn.OpeningIndex(m)
    fr = pg.plane_frame(P)
    door = np.array([(2, 0, 0), (3.2, 0, 0), (3.2, 0, 2.2), (2, 0, 2.2)], float)
    holes = [("D", pg.to_2d(door, fr))]
    pieces = opn.fill_polygons(P, holes, 1e-6)
    area = sum(abs(pg.signed_area(pg.to_2d(q, fr))) for q in pieces)
    assert area == pytest.approx(6 * 3.5 - 1.2 * 2.2)
    L = np.array([(0, 0, 0), (12, 0, 0), (12, 5, 0), (6, 5, 0), (6, 10, 0), (0, 10, 0)], float)
    assert sum(abs(pg.signed_area(q[:, :2])) for q in opn.fill_polygons(L, [], 1e-6)) == pytest.approx(90)
    assert not idx.holes(P, fr, 1e-6)  # nothing drawn in that wall plane


# ------------------------------------------------ result conventions / display
def _one_bay_slab(model_type):
    m = new_building(x_spacings=(6,), y_spacings=(6,), story_heights=(3.0,), grid_only=False)
    for s in m.shell_sections.values():
        s.model_type = model_type
    f = next(a for a in m.areas.values() if a.type == "Floor")
    m.add_area_load(f.name, "Live", 10.0)
    return m


@pytest.mark.parametrize("model_type", ["ShellThin", "ShellThick"])
def test_slab_moment_sign_is_csi_sagging_positive(model_type):
    """CSI: S11 = F11/t - 12 M11 x3 / t^3 - positive M11 compresses the top (+3)
    face, so a gravity slab has positive M11 at mid-span, negative at the beams."""
    m = _one_bay_slab(model_type)
    r = analyze(m, ["Live"], mesh_options=MeshOptions(floor_max_size=0.5))
    vals, nodes = r.shell_node_values("Live", "M11")
    c = np.array([[r.am.nodes[n].xyz for n in row] for row in nodes]).mean(axis=1)
    mid = np.argmin(np.linalg.norm(c[:, :2] - (3, 3), axis=1))
    edge = np.argmin(np.abs(c[:, 0] - 0.25) + np.abs(c[:, 1] - 3))
    assert vals[mid].mean() > 0 > vals[edge].mean()


def test_combination_shell_results_are_built_once():
    import time
    m = _two_story()
    for a in m.areas.values():
        if a.type == "Floor":
            m.add_area_load(a.name, "Live", 3.0)
    m.add_combination("C", {"Dead": 1.2, "Live": 1.6})
    r = analyze(m, ["Dead", "Live"], mesh_options=MeshOptions(floor_max_size=0.5))
    t = time.time()
    env = r.shell_element_envelope("C")
    assert time.time() - t < 5.0 and len(env) == len(r.shell_element_envelope("Dead"))
    e = next(iter(env))
    assert np.allclose(env[e].max, 1.2 * r.shell_element_envelope("Dead")[e].max
                       + 1.6 * r.shell_element_envelope("Live")[e].max)
