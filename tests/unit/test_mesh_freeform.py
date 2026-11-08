"""Freeform floors (non-convex, curved, openings), columns / loads / line loads
anywhere, slabs that share an edge with non-matching meshes (auto edge
constraints), ramps, skew walls standing on slabs and members passing through
slabs: equilibrium exact, no floating parts, valid elements, and the plate
solution independent of the plate's orientation (Navier)."""
import numpy as np
import pytest

from structstudio.api import analyze
from structstudio.app.templates import new_building
from structstudio.mesh import polygon2d as pg
from structstudio.mesh.mesher import MeshOptions
from structstudio.model import Model
from structstudio.model.objects import LineSpring
from structstudio.model.properties import ShellSection


def _secs(m):
    return (next(n for n in m.frame_sections if n.startswith("C")), next(n for n in m.frame_sections if n.startswith("B")),
            next(n for n in m.shell_sections if n.startswith("S")), next(n for n in m.shell_sections if n.startswith("W")))


def _column(m, col, x, y, z):
    b = m.add_point(x, y, 0.0)
    b.restraints = (True,) * 6
    m.add_frame(b, m.add_point(x, y, z), col, type="Column")


def test_freeform_slab_with_odd_columns_loads_and_a_curved_line_load():
    m = new_building(x_spacings=(8,), y_spacings=(8,), story_heights=(4.0,), grid_only=True)
    col, bm, slab, _ = _secs(m)
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    R = np.array([9, 7, 10, 6, 9.5, 8, 11, 7, 8.5, 6.5, 10, 7.5])
    poly = [(R[i] * np.cos(ang[i]) + 0.3 * i, 0.8 * R[i] * np.sin(ang[i])) for i in range(12)]
    f = m.add_area([m.add_point(x, y, 4.0) for x, y in poly], slab, type="Floor")
    for xy in [(0.3, 0.2), (4.7, -2.1), (-3.3, -4.4), poly[0], poly[3], poly[6], poly[9]]:
        _column(m, col, *xy, 4.0)
    t = m.add_point(2.7, -1.3, 7.0)  # planted column with a load on top
    m.add_frame(m.add_point(2.7, -1.3, 4.0), t, col, type="Column")
    t.restraints = (True, True, False, False, False, False)
    m.add_point_load(t.name, "Live", force=(0, 0, -200))
    pts = [(-6 + 1.1 * k, 2.5 * np.sin(0.6 * k) - 1) for k in range(11)]
    L = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = m.add_frame(m.add_point(x0, y0, 4.0), m.add_point(x1, y1, 4.0), None, type="Null")
        m.add_frame_distributed_load(n.name, "Live", 6.0)
        L += np.hypot(x1 - x0, y1 - y0)
    for xy in [(1.37, -1.84), (3.13, 3.30), (1.07, 1.84)]:
        p = m.add_point(*xy, 4.0)
        m.add_point_load(p.name, "Live", force=(0, 0, -30))
    m.add_area_load(f.name, "Live", 2.0)
    r = analyze(m, ["Live"], mesh_options=MeshOptions(floor_max_size=0.5))
    area = abs(pg.signed_area(np.array(poly)))
    assert r.base_reactions("Live")[2] == pytest.approx(200 + 6 * L + 90 + 2 * area, rel=1e-9)
    assert not r.am.warnings
    X = np.array([n.xyz for n in r.am.nodes])
    good = 0
    for s in r.am.shells:
        P = X[list(s.nodes)]
        assert pg.signed_area(P[:, :2]) > 0  # valid, counter-clockwise
        ang_ = [np.degrees(np.arccos(np.clip((P[k - 1] - P[k]) @ (P[(k + 1) % 4] - P[k]) /
                np.linalg.norm(P[k - 1] - P[k]) / np.linalg.norm(P[(k + 1) % 4] - P[k]), -1, 1))) for k in range(4)]
        good += 25 <= min(ang_) and max(ang_) <= 155
    assert good >= 0.99 * len(r.am.shells)


def test_slabs_with_non_matching_meshes_are_tied_and_an_opening_is_exact():
    m = new_building(x_spacings=(8,), y_spacings=(8,), story_heights=(4.0,), grid_only=True)
    col, bm, slab, _ = _secs(m)
    A = [(5, 9)] + [(5 - 5.5 * np.sin(t), 4.5 + 4.5 * np.cos(t)) for t in np.linspace(0.25, np.pi - 0.25, 9)] + [(5, 0)]
    B = [(5, 0), (9, -0.8), (12.5, 1.5), (13.2, 5.0), (11, 8.6), (7.5, 9.8), (5, 9)]
    for poly in (A, B):
        a = m.add_area([m.add_point(x, y, 4.0) for x, y in poly], slab, type="Floor")
        m.add_area_load(a.name, "Live", 3.0)
    oc = [(9.5 + 1.2 * np.cos(t), 4.5 + 0.9 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 9)[:-1]]
    m.add_area([m.add_point(x, y, 4.0) for x, y in oc], None, type="Opening")
    for xy in [(5, 4.2), oc[2], (2.0, 4.5), (10.5, 8.0), (3.0, 1.5), (12.0, 3.0), (1.0, 6.0), (8, 0.5)]:
        _column(m, col, *xy, 4.0)
    pts = [(1.5 + k, 7.0 - 0.07 * k * k) for k in range(10)]  # crosses from one slab into the other
    L = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = m.add_frame(m.add_point(x0, y0, 4.0), m.add_point(x1, y1, 4.0), None, type="Null")
        m.add_frame_distributed_load(n.name, "Live", 5.0)
        L += np.hypot(x1 - x0, y1 - y0)
    r = analyze(m, ["Live"], mesh_options=MeshOptions(floor_max_size=0.5))
    area = abs(pg.signed_area(np.array(A))) + abs(pg.signed_area(np.array(B))) - abs(pg.signed_area(np.array(oc)))
    assert r.base_reactions("Live")[2] == pytest.approx(5 * L + 3 * area, rel=1e-8)
    assert r.am.ties and not r.am.warnings


def _inclined_building():
    m = new_building(x_spacings=(8, 8), y_spacings=(8, 8), story_heights=(3.5, 3.5), grid_only=False)
    col, bm, slab, wall = _secs(m)
    z1, z2 = 3.5, 7.0
    m.add_area([m.add_point(2, 2, z1), m.add_point(6.5, 5.3, z1), m.add_point(6.5, 5.3, z2), m.add_point(2, 2, z2)],
               wall, type="Wall")  # skew wall standing on the slab
    ramp = m.add_area([m.add_point(0, 16, z1), m.add_point(4, 16, z1), m.add_point(4, 22, z2), m.add_point(0, 22, z2)],
                      slab, type="Ramp")
    for x in (0, 4):
        _column(m, col, x, 22, z2)
    m.add_frame(m.add_point(0, 22, z2), m.add_point(4, 22, z2), bm, type="Beam")
    zr = lambda y: z1 + (z2 - z1) * (y - 16) / 6
    _column(m, col, 2.7, 19.3, zr(19.3))  # column under the ramp, mid-ramp
    p = m.add_point(1.1, 17.4, zr(17.4))
    m.add_point_load(p.name, "Live", force=(0, 0, -25))
    n = m.add_frame(m.add_point(0.3, 18, zr(18)), m.add_point(3.6, 20.5, zr(20.5)), None, type="Null")
    m.add_frame_distributed_load(n.name, "Live", 4.0)
    m.add_frame(m.add_point(0, 16, z1), m.add_point(4, 22, z2), bm, type="Brace")  # beam lying on the ramp
    a = m.add_point(10, 3, 0)
    a.restraints = (True,) * 6
    m.add_frame(a, m.add_point(14, 6, z2), bm, type="Brace")  # through the slab mid-panel
    for ar in list(m.areas.values()):
        if ar.type in ("Floor", "Ramp"):
            m.add_area_load(ar.name, "Live", 2.0)
    return m, np.hypot(3.3, np.hypot(2.5, zr(20.5) - zr(18)))


def test_ramp_skew_wall_and_a_brace_through_the_slab():
    m, L = _inclined_building()
    area = 0.0
    for ar in m.areas.values():
        if ar.type in ("Floor", "Ramp"):
            P = np.array([m.points[p].xyz for p in ar.points])
            area += 0.5 * np.linalg.norm(sum(np.cross(P[i], P[(i + 1) % len(P)]) for i in range(len(P))))
    r = analyze(m, ["Live"], mesh_options=MeshOptions(floor_max_size=0.5, wall_max_size=0.5))
    assert r.base_reactions("Live")[2] == pytest.approx(25 + 4 * L + 2 * area, rel=1e-9)
    assert not r.am.warnings
    brace = next(n for n, f in m.frames.items() if f.type == "Brace" and np.allclose(m.points[f.point_i].xyz, (10, 3, 0)))
    assert len(r.am.object_to_frames[brace]) == 2  # cut and joined where it passes the slab
    assert any("pass through" in i for i in r.am.info)


@pytest.mark.parametrize("kind", ["tilt", "skew"])
def test_tilted_and_skewed_plates_match_navier(kind):
    a = b = 6.0
    t, E, nu, P, x0, y0 = 0.2, 25e6, 0.2, 100.0, 2.3, 3.7
    D = E * t ** 3 / (12 * (1 - nu ** 2))
    k = np.arange(1, 300)
    mm, nn = np.meshgrid(k, k, indexing="ij")
    if kind == "tilt":
        th = np.radians(30)
        R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        Pn = P * np.cos(th)
    else:
        ps = np.radians(23)
        R = np.array([[np.cos(ps), -np.sin(ps), 0], [np.sin(ps), np.cos(ps), 0], [0, 0, 1]])
        Pn = P
    ref = 4 * Pn / (np.pi ** 4 * D * a * b) * np.sum(
        (np.sin(mm * np.pi * x0 / a) * np.sin(nn * np.pi * y0 / b)) ** 2 / ((mm / a) ** 2 + (nn / b) ** 2) ** 2)
    m = Model("plate", "kN m")
    m.add_material("C", E=E, nu=nu)
    m.add_shell_section(ShellSection(name="S", material="C", thickness=t, model_type="ShellThin"))
    X = lambda u, v: tuple(R @ np.array([u, v, 0.0]) + np.array([0, 0, 10.0]))
    c = [m.add_point(*X(0, 0)), m.add_point(*X(a, 0)), m.add_point(*X(a, b)), m.add_point(*X(0, b))]
    m.add_area(c, "S", type="Ramp" if kind == "tilt" else "Floor")
    for i in range(4):
        m.line_springs[f"L{i}"] = LineSpring(f"L{i}", c[i].name, c[(i + 1) % 4].name, stiffness=(1e9, 1e9, 1e9))
    p = m.add_point(*X(x0, y0))
    m.add_load_pattern("P")
    m.add_point_load(p.name, "P", force=(0, 0, -P))
    m.add_linear_static_case("P")
    r = analyze(m, ["P"], mesh_options=MeshOptions(floor_max_size=0.25))
    w = -np.array(r.joint_displacement(p.name, "P")[:3]) @ (R @ np.array([0, 0, 1.0]))
    assert w == pytest.approx(ref, rel=0.01)
