"""Floor / wall meshing around what stands on them (ETABS "mesh at beams and
meshing lines", SAFE meshing at columns, point and line loads): joints, planted
columns, loaded joints, beams, null lines with line loads, walls - and the mesh
staying conforming between slab panels and between slabs and walls."""
import numpy as np
import pytest

from structstudio.api import analyze
from structstudio.app.templates import new_building
from structstudio.mesh.mesher import MeshOptions, build_analysis_model
from structstudio.model import Model
from structstudio.model.objects import LineSpring
from structstudio.model.properties import ShellSection

Z1, Z2 = 3.5, 7.0


def building():
    m = new_building(x_spacings=(8, 8), y_spacings=(7, 7), story_heights=(3.5, 3.5), grid_only=False)
    col = next(n for n in m.frame_sections if n.startswith("C"))
    bm = next(n for n in m.frame_sections if n.startswith("B"))
    m.add_frame(m.add_point(4.3, 3.1, Z1), m.add_point(4.3, 3.1, Z2), col, type="Column")  # planted column
    for xy in [(11.7, 2.2), (13.1, 10.9)]:
        p = m.add_point(*xy, Z1)
        m.add_point_load(p.name, "Live", force=(0, 0, -50))
    f = m.add_frame(m.add_point(8.5, 10.3, Z1), m.add_point(15.0, 10.3, Z1), None, type="Null")
    m.add_frame_distributed_load(f.name, "Live", 8.0)
    f = m.add_frame(m.add_point(1.0, 8.0, Z1), m.add_point(6.5, 13.0, Z1), None, type="Null")
    m.add_frame_distributed_load(f.name, "Live", 5.0)
    m.add_frame(m.add_point(8, 3.5, Z2), m.add_point(16, 3.5, Z2), bm, type="Beam")  # secondary beam
    m.add_frame(m.add_point(0, 7, Z2), m.add_point(8, 14, Z2), bm, type="Beam")  # skew beam
    w = next(n for n in m.shell_sections if n.startswith("W"))
    m.add_area([m.add_point(9.5, 6, Z1), m.add_point(13.5, 6, Z1), m.add_point(13.5, 6, Z2),
                m.add_point(9.5, 6, Z2)], w, type="Wall")  # wall standing on the slab
    return m


APPLIED = 2 * 50 + 8 * 6.5 + 5 * np.hypot(5.5, 5.0)


def _hanging_nodes(am):
    X = np.array([n.xyz for n in am.nodes])
    edges = {}
    for s in am.shells:
        for k in range(4):
            e = frozenset((s.nodes[k], s.nodes[(k + 1) % 4]))
            edges[e] = edges.get(e, 0) + 1
    Y = X[sorted({n for s in am.shells for n in s.nodes})]
    hang = 0
    for e, c in edges.items():
        if c != 1:
            continue
        a, b = tuple(e)
        A, d = X[a], X[b] - X[a]
        t = ((Y - A) @ d) / (d @ d)
        dist = np.linalg.norm(Y - (A + np.outer(t, d)), axis=1)
        hang += int(np.sum((t > 1e-6) & (t < 1 - 1e-6) & (dist < 1e-4)))
    return hang


@pytest.fixture(scope="module")
def solved():
    m = building()
    return m, analyze(m, ["Live", "Dead"], mesh_options=MeshOptions(floor_max_size=1.0, wall_max_size=1.0))


def _point(m, x, y, z):
    return next(n for n, p in m.points.items() if np.allclose(p.xyz, (x, y, z)))


def test_point_and_line_loads_on_slabs_all_reach_the_supports(solved):
    m, r = solved
    assert r.base_reactions("Live")[2] == pytest.approx(APPLIED, rel=1e-9)
    assert not r.am.warnings
    for xy in [(11.7, 2.2), (13.1, 10.9)]:  # loaded joints are slab nodes, and deflect
        nid = r.am.point_to_node[_point(m, *xy, Z1)]
        assert any(nid in s.nodes for s in r.am.shells)
        assert r.joint_displacement(_point(m, *xy, Z1), "Live")[2] < -1e-4


def test_planted_column_sits_on_the_slab_mesh(solved):
    m, r = solved
    base = r.am.point_to_node[_point(m, 4.3, 3.1, Z1)]
    assert any(base in s.nodes for s in r.am.shells)
    assert r.joint_displacement(_point(m, 4.3, 3.1, Z1), "Dead")[2] < 0


def test_beams_inside_a_panel_are_tied_in_along_their_length(solved):
    m, r = solved
    shell_nodes = {n for s in r.am.shells for n in s.nodes}
    for (x0, y0, x1, y1) in [(8, 3.5, 16, 3.5), (0, 7, 8, 14)]:
        f = next(n for n, f in m.frames.items() if f.type == "Beam"
                 and np.allclose(m.points[f.point_i].xyz, (x0, y0, Z2)) and np.allclose(m.points[f.point_j].xyz, (x1, y1, Z2)))
        els = [r.am.frames[i] for i in r.am.object_to_frames[f]]
        assert len(els) >= 8  # cut at the slab nodes along it
        assert all(e.node_i in shell_nodes and e.node_j in shell_nodes for e in els)


def test_mesh_is_conforming_and_well_shaped(solved):
    m, r = solved
    am = r.am
    assert _hanging_nodes(am) == 0  # between panels, and between the slabs and the wall
    X = np.array([n.xyz for n in am.nodes])
    for s in am.shells:
        P = X[list(s.nodes)]
        for k in range(4):
            u, v = P[k - 1] - P[k], P[(k + 1) % 4] - P[k]
            ang = np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
            assert 25 <= ang <= 155


def test_bare_joint_in_a_slab_changes_nothing():
    m = new_building(x_spacings=(6,), y_spacings=(6,), story_heights=(3.0,), grid_only=False)
    n0 = len(build_analysis_model(m, MeshOptions(floor_max_size=1.0)).shells)
    m.add_point(2.3, 3.7, 3.0)  # no load, no member, no support: SAFE does not mesh at it
    assert len(build_analysis_model(m, MeshOptions(floor_max_size=1.0)).shells) == n0


def test_off_grid_point_load_on_a_plate_converges_to_navier():
    a = b = 6.0
    t, E, nu, P, x0, y0 = 0.2, 25e6, 0.2, 100.0, 2.3, 3.7
    D = E * t ** 3 / (12 * (1 - nu ** 2))
    k = np.arange(1, 200)
    mm, nn = np.meshgrid(k, k, indexing="ij")
    ref = 4 * P / (np.pi ** 4 * D * a * b) * np.sum(
        (np.sin(mm * np.pi * x0 / a) * np.sin(nn * np.pi * y0 / b)) ** 2 / ((mm / a) ** 2 + (nn / b) ** 2) ** 2)
    m = Model("plate", "kN m")
    m.add_material("C", E=E, nu=nu)
    m.add_shell_section(ShellSection(name="S", material="C", thickness=t, model_type="ShellThin"))
    c = [m.add_point(0, 0, 0), m.add_point(a, 0, 0), m.add_point(a, b, 0), m.add_point(0, b, 0)]
    m.add_area(c, "S", type="Floor")
    for i in range(4):  # simply supported edges
        m.line_springs[f"L{i}"] = LineSpring(f"L{i}", c[i].name, c[(i + 1) % 4].name, stiffness=(0, 0, 1e9))
    c[0].restraints = (True, True, False, False, False, True)
    c[1].restraints = (False, True, False, False, False, False)
    p = m.add_point(x0, y0, 0)
    m.add_load_pattern("P")
    m.add_point_load(p.name, "P", force=(0, 0, -P))
    m.add_linear_static_case("P")
    r = analyze(m, ["P"], mesh_options=MeshOptions(floor_max_size=0.25))
    w = -r.joint_displacement(p.name, "P")[2]
    assert w == pytest.approx(ref, rel=0.01)  # 0.42 % at 0.25 m (3.9 % at 1 m, 0.13 % at 0.125 m)
