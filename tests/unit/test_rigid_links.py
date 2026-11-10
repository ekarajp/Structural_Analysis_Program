"""Rigid links, beams meshed on their true (offset) axis, membrane floors carried
to their frames, rigid zones split over several elements: every load is applied
in full and the structure stays in equilibrium (forces and moments)."""
import copy

import numpy as np

from structstudio.api import analyze
from structstudio.app.templates import new_building
from structstudio.mesh import build_analysis_model
from structstudio.mesh.panel_loads import panel_loads_to_members
from structstudio.model.objects import PanelLoadCase
from structstudio.model.properties import FrameSection


def _bay(ext=0.5, off=0.35, model_type="ShellThick", xs=(6,), types=None):
    """One story, 1 m square columns, floors out to the column faces, edge beams
    flush with the outer face (offset ``off`` from the grid, in the analysis)."""
    m = new_building(x_spacings=xs, y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m.add_frame_section(FrameSection.rectangular("C100", "C28", 1.0, 1.0))
    for f in m.frames.values():
        if f.type == "Column":
            f.section = "C100"
    X = float(sum(xs))
    for k, name in enumerate(sorted(m.areas)):
        ar = m.areas[name]
        P = [m.points[p] for p in ar.points]
        x0, x1 = min(p.x for p in P), max(p.x for p in P)
        x0 -= ext if x0 < 1e-9 else 0.0
        x1 += ext if x1 > X - 1e-9 else 0.0
        ar.points = [m.add_point(x, y, 3.0).name for x, y in ((x0, -ext), (x1, -ext), (x1, 5 + ext), (x0, 5 + ext))]
        sec = copy.deepcopy(m.shell_sections[ar.section])
        sec.name = f"S{k}"
        sec.model_type = (types or (model_type,))[k % len(types or (model_type,))]
        m.add_shell_section(sec)
        ar.section = sec.name
    for f in m.frames.values():
        if f.type != "Beam":
            continue
        a, b = m.points[f.point_i], m.points[f.point_j]
        my, mx = (a.y + b.y) / 2, (a.x + b.x) / 2
        if abs(a.y - b.y) < 1e-9:
            o = (0.0, -off if my < 1e-9 else off if my > 5 - 1e-9 else 0.0)
        else:
            o = (-off if mx < 1e-9 else off if mx > X - 1e-9 else 0.0, 0.0)
        f.joint_offset_i = f.joint_offset_j = (o[0], o[1], 0.0)
        f.joint_offset_local = False
        f.offsets_in_analysis = True
    return m


def _area(m, name):
    P = np.array([m.points[p].xyz for p in m.areas[name].points])[:, :2]
    return 0.5 * abs(P[:, 0] @ np.roll(P[:, 1], -1) - P[:, 1] @ np.roll(P[:, 0], -1))


def _self_weight(m):
    w = sum(24.0 * m.shell_sections[a.section].thickness * _area(m, n) for n, a in m.areas.items())
    for f in m.frames.values():
        if f.type in ("Null", "Rigid"):
            continue
        L = np.linalg.norm(np.subtract(m.points[f.point_j].xyz, m.points[f.point_i].xyz))
        w += 24.0 * m.frame_sections[f.section].A * L
    return w


def _base_moment(m, r, case):
    tot = np.zeros(3)
    for row in r.joint_reactions(case):
        p = np.array(m.points[row["Point"]].xyz)
        tot += np.cross(p, [row["FX"], row["FY"], row["FZ"]]) + np.array([row["MX"], row["MY"], row["MZ"]])
    return tot


def test_beam_on_true_axis_equals_explicit_rigid_links():
    """Offsets in the analysis = the beam drawn on its axis + rigid links to the joints."""
    m1 = _bay()
    m2 = _bay(off=0.0)
    for m in (m1, m2):
        for f in m.frames.values():
            f.auto_end_offsets = False
    for f in list(m2.frames.values()):
        if f.type != "Beam":
            continue
        v = np.array(m1.frames[f.name].joint_offset_i)
        pi, pj = m2.points[f.point_i], m2.points[f.point_j]
        a = m2.add_point(pi.x + v[0], pi.y + v[1], pi.z)
        b = m2.add_point(pj.x + v[0], pj.y + v[1], pj.z)
        for jp, q in ((pi, a), (pj, b)):
            if not any(x.type == "Rigid" and {x.point_i, x.point_j} == {jp.name, q.name} for x in m2.frames.values()):
                m2.add_frame(jp.name, q.name, "", type="Rigid")
        f.point_i, f.point_j = a.name, b.name
    for m in (m1, m2):
        m.add_area_load("F1", "Live", 3.0)
    r1, r2 = analyze(m1), analyze(m2)
    assert len(r1.am.rigid_links) == 8 and len(r2.am.rigid_links) == 8
    for case in ("Dead", "Live"):
        assert np.allclose(r1.base_reactions(case), r2.base_reactions(case), atol=1e-8)
        assert np.allclose(r1.frame_forces("B1", case).forces, r2.frame_forces("B1", case).forces, atol=1e-6)
    # the floor shares the beam's nodes on its true axis
    am = r1.am
    beam_nodes = {n for e in am.frames if e.object == "B1" for n in (e.node_i, e.node_j)}
    ys = {round(float(am.nodes[n].xyz[1]), 6) for n in beam_nodes}
    assert ys == {-0.35}
    shell_nodes = {n for s in am.shells for n in s.nodes}
    assert len(beam_nodes & shell_nodes) >= len(beam_nodes) - 2  # all but the two inside the columns at most


def test_offset_beam_without_floor_same_as_element_offsets():
    """Without a floor the true-axis mesh gives exactly the old rigid-arm elements."""
    import structstudio.mesh.mesher as MS
    m = _bay()
    m.areas.clear()
    m.panel_loads.clear()
    for f in m.frames.values():
        if f.type == "Beam":
            m.add_frame_distributed_load(f.name, "Live", 10.0)
    r_new = analyze(m)
    orig = MS.beam_axis_shift
    MS.beam_axis_shift = lambda *a, **k: None
    try:
        r_old = analyze(copy.deepcopy(m))
    finally:
        MS.beam_axis_shift = orig
    for p in m.points:
        assert np.allclose(r_new.joint_displacement(p, "Live"), r_old.joint_displacement(p, "Live"), atol=1e-12)
    assert np.allclose(r_new.frame_forces("B1", "Live").forces, r_old.frame_forces("B1", "Live").forces, atol=1e-8)


def test_membrane_floor_loads_and_self_weight_reach_the_frames():
    for mt in ("ShellThick", "Membrane"):
        m = _bay(model_type=mt)
        m.add_area_load("F1", "Live", 3.0)
        r = analyze(m)
        assert abs(r.base_reactions("Live")[2] - 3.0 * _area(m, "F1")) < 1e-6
        assert abs(r.base_reactions("Dead")[2] - _self_weight(m)) < 1e-6
        assert np.allclose(_base_moment(m, r, "Live") + np.cross([3.0, 2.5, 3.0], [0, 0, -3.0 * _area(m, "F1")]),
                           0.0, atol=1e-6)
        if mt == "Membrane":
            assert not any("Null DOF" in w for w in r.am.warnings)
            assert any("Membrane floors" in i for i in r.am.info)


def test_membrane_two_way_share_per_beam():
    """Each beam takes the floor nearest to its true axis (independent integration)."""
    m = _bay(model_type="Membrane")
    m.add_area_load("F1", "Live", 3.0)
    fl, _, _ = panel_loads_to_members(m)
    beams = [f for f in m.frames.values() if f.type == "Beam"]
    P = np.array([m.points[p].xyz[:2] for p in m.areas["F1"].points])
    lo, hi = P.min(0), P.max(0)
    g = 500
    xs = lo[0] + (np.arange(g) + 0.5) * (hi[0] - lo[0]) / g
    ys = lo[1] + (np.arange(g) + 0.5) * (hi[1] - lo[1]) / g
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    D = []
    for f in beams:
        a = np.array(m.points[f.point_i].xyz[:2]) + np.array(f.joint_offset_i[:2])
        b = np.array(m.points[f.point_j].xyz[:2]) + np.array(f.joint_offset_j[:2])
        d = b - a
        t = np.clip((pts - a) @ d / (d @ d), 0, 1)
        D.append(np.linalg.norm(pts - (a + t[:, None] * d), axis=1))
    k = np.argmin(np.array(D), axis=0)
    dA = (hi[0] - lo[0]) * (hi[1] - lo[1]) / g ** 2
    for i, f in enumerate(beams):
        L = np.linalg.norm(np.subtract(m.points[f.point_j].xyz, m.points[f.point_i].xyz))
        got = sum((ld.rel_b - ld.rel_a) * L * (ld.value_a + ld.value_b) / 2 for ld in fl[f.name] if ld.pattern == "Live")
        assert abs(got - 3.0 * dA * np.sum(k == i)) < 0.01 * got


def test_shell_and_membrane_floors_in_one_model_and_drawn_panels():
    m = _bay(xs=(6, 6), types=("ShellThick", "Membrane"))
    m.add_area_load("F1", "Live", 3.0)
    m.add_area_load("F2", "Live", 3.0)
    r = analyze(m)
    assert abs(r.base_reactions("Live")[2] - 3.0 * (_area(m, "F1") + _area(m, "F2"))) < 1e-6
    assert abs(r.base_reactions("Dead")[2] - _self_weight(m)) < 1e-6
    for mt in ("ShellThick", "Membrane"):
        for dist in ("shells", "two_way", "one_way"):
            m = _bay(model_type=mt)
            P = [m.points[p].xyz for p in m.areas["F1"].points]
            m.add_panel_load(P, dist, span_dir=(1, 0, 0) if dist == "one_way" else None,
                             loads=[PanelLoadCase("Live", 2.0)])
            r = analyze(m)
            assert abs(r.base_reactions("Live")[2] - 2.0 * _area(m, "F1")) < 1e-6, (mt, dist)


def test_membrane_keeps_its_mass():
    from structstudio.engine.assembly import Assembler
    ms = [Assembler(build_analysis_model(_bay(model_type=mt))).mass_full()[0::6].sum()
          for mt in ("ShellThick", "Membrane")]
    assert abs(ms[0] - ms[1]) < 1e-6 * ms[0]


def test_rigid_link_bracket_is_rigid_and_in_equilibrium():
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    top = next(p for p in m.points.values() if abs(p.z - 3) < 1e-9 and abs(p.x) < 1e-9 and abs(p.y) < 1e-9)
    tip = m.add_point(-1.2, 0.0, 3.0)
    lk = m.add_frame(top.name, tip.name, "", type="Rigid")
    assert lk.name.startswith("RL")
    m.add_point_load(tip.name, "Live", force=(0, 0, -10.0))
    r = analyze(m)
    assert not any("not connected" in i.message or "not applied" in i.message for i in r.issues)
    assert abs(r.base_reactions("Live")[2] - 10.0) < 1e-9
    assert np.allclose(_base_moment(m, r, "Live") + np.cross([-1.2, 0, 3], [0, 0, -10.0]), 0.0, atol=1e-8)
    dt, dp = r.joint_displacement(top.name, "Live"), r.joint_displacement(tip.name, "Live")
    assert np.allclose(dp[:3], dt[:3] + np.cross(dt[3:], [-1.2, 0, 0]), atol=1e-12)
    assert np.allclose(dp[3:], dt[3:], atol=1e-12)


def test_rigid_zone_cut_by_the_slab_mesh_keeps_equilibrium():
    """Rigid zones and the slab mesh: a joint moment on a column top is balanced
    exactly, and the columns get their moments however fine the mesh."""
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    top = next(p for p in m.points.values() if abs(p.z - 3) < 1e-9 and abs(p.x) < 1e-9 and abs(p.y) < 1e-9)
    m.add_point_load(top.name, "Live", force=(0, 0, -10.0), moment=(0, 12.0, 0))
    r = analyze(m)
    applied = np.cross([0, 0, 3], [0, 0, -10.0]) + np.array([0, 12.0, 0])
    assert np.allclose(_base_moment(m, r, "Live") + applied, 0.0, atol=1e-8)
    am = r.am
    pieces = [e for e in am.frames if e.object == "B1"]
    L = sum(np.linalg.norm(am.nodes[e.node_j].xyz - am.nodes[e.node_i].xyz) for e in pieces)
    rig = sum(e.rigid_i + e.rigid_j for e in pieces)
    # RZF 0.5 x half of the 0.4 column at both ends; the slab mesh nodes inside the zones
    # do not cut the beam (no wholly rigid piece), so the zones are exact
    assert abs(rig - 0.2) < 1e-9
    for e in pieces:
        Le = np.linalg.norm(am.nodes[e.node_j].xyz - am.nodes[e.node_i].xyz)
        assert e.rigid_i + e.rigid_j < 0.99 * Le


def test_beam_axis_outside_the_column_warns():
    m = _bay(off=0.7)
    m.add_area_load("F1", "Live", 3.0)
    r = analyze(m)
    assert any("outside column" in w for w in r.am.warnings)
    assert abs(r.base_reactions("Live")[2] - 3.0 * _area(m, "F1")) < 1e-6


def test_column_moments_do_not_depend_on_the_mesh_inside_the_rigid_zones():
    out = []
    for refine in (False, True):
        m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5), grid_only=False)
        for a in m.areas.values():
            m.add_area_load(a.name, "Live", 90.0)
        m.options.mesh_refine_check = refine
        r = analyze(m)
        out.append(np.abs(r.frame_forces("C1", "Live").forces[:, 4:6]).max())
    assert out[0] > 100.0 and abs(out[1] - out[0]) < 0.05 * out[0]


def test_offset_beam_nodes_inside_a_column_rigid_zone_without_a_diaphragm():
    """The beam's axis nodes lie in the slab rigid zone of the column they are linked
    to: the zone and the link must not define each other (was CONSTRAINT_LOOP when no
    diaphragm made the joint the master)."""
    m = _bay()
    for a in m.areas.values():
        a.diaphragm = None
    for p in m.points.values():
        p.diaphragm = None
    assert m.options.slab_column_rigid_zone
    m.add_area_load("F1", "Live", 3.0)
    r = analyze(m)
    assert r.am.bodies and r.am.rigid_links
    assert abs(r.base_reactions("Live")[2] - 3.0 * _area(m, "F1")) < 1e-6
    assert np.allclose(_base_moment(m, r, "Live") + np.cross([3.0, 2.5, 3.0], [0, 0, -3.0 * _area(m, "F1")]),
                       0.0, atol=1e-5)


def test_membrane_floor_with_an_opening_carries_no_load_on_the_opening():
    for mt in ("ShellThick", "Membrane"):
        m = _bay(model_type=mt, ext=0.0, off=0.0)
        op = [m.add_point(x, y, 3.0).name for x, y in ((1, 1), (3, 1), (3, 2.5), (1, 2.5))]
        m.add_area(op, None, type="Opening", story="Story1")
        m.add_area_load("F1", "Live", 3.0)
        r = analyze(m)
        net = _area(m, "F1") - 3.0
        assert abs(r.base_reactions("Live")[2] - 3.0 * net) < 1e-3 * 3.0 * net, mt
        frames = sum(24.0 * m.frame_sections[f.section].A *
                     np.linalg.norm(np.subtract(m.points[f.point_j].xyz, m.points[f.point_i].xyz))
                     for f in m.frames.values())
        assert abs(r.base_reactions("Dead")[2] - (frames + 24.0 * 0.15 * net)) < 1e-3 * 24.0 * 0.15 * net, mt


def test_reactions_include_the_constrained_nodes_hanging_on_a_support():
    """The reaction of a support is the residual of its reduced equation: what tied
    nodes, a slab rigid zone or a rigid link pass to it was left out (the reported
    reactions did not balance the loads)."""
    from structstudio import Model
    from structstudio.model import FIXED, PINNED, ShellSection
    from structstudio.mesh import MeshOptions
    # 1. a finer floor tied along the edge of a coarser one, at point supports
    m = Model("p", "kN m")
    m.add_material("C", E=30e6, nu=0.2)
    m.add_shell_section(ShellSection("S", "C", type="Slab", model_type="ShellThick", thickness=0.25))
    m.set_stories([("L1", 3.0)])
    P = {(x, y): m.add_point(x, y, 3.0, restraints=PINNED).name for x in (0, 6, 12) for y in (0, 6)}
    m.add_area([P[(0, 0)], P[(6, 0)], P[(6, 6)], P[(0, 6)]], "S", name="F1", max_mesh_size=0.75)
    m.add_area([P[(6, 0)], P[(12, 0)], P[(12, 6)], P[(6, 6)]], "S", name="F2", max_mesh_size=0.5)
    m.add_load_pattern("Q")
    m.add_linear_static_case("Q")
    m.add_area_load("F1", "Q", 10.0)
    m.add_area_load("F2", "Q", 10.0)
    r = analyze(m, mesh_options=MeshOptions(refine_check=False))
    assert r.am.ties
    assert abs(r.base_reactions("Q")[2] - 720.0) < 1e-6
    assert np.allclose(_base_moment(m, r, "Q") + np.cross([6.0, 3.0, 3.0], [0, 0, -720.0]), 0.0, atol=1e-6)
    # 2. a member hanging on a rigid link to a support
    m = Model("b", "kN m")
    m.add_material("S", E=200e6, nu=0.3, type="Steel")
    m.add_frame_section(FrameSection.general("B", "S", A=0.01, I33=1e-4, I22=5e-5, J=1e-6))
    m.add_load_pattern("P")
    m.add_linear_static_case("P")
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(1.5, 0, 0)
    c = m.add_point(1.5, 0, 3)
    m.add_frame(a, b, "", type="Rigid")
    m.add_frame(b, c, "B", name="C")
    m.add_point_load(c, "P", force=(10, 0, -50))
    r = analyze(m)
    assert np.allclose(r.joint_reaction(a.name, "P"), [-10, 0, 50, 0, -105, 0], atol=1e-8)
    # 3. a slab at the level of the restrained column bases (their slab rigid zones)
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    base = {(p.x, p.y): p.name for p in m.points.values() if abs(p.z) < 1e-9}
    m.add_area([base[(0, 0)], base[(6, 0)], base[(6, 5)], base[(0, 5)]], "S20", name="G", type="Floor", story="Base")
    m.add_area_load("G", "Live", 10.0)
    r = analyze(m)
    assert r.am.bodies
    assert abs(r.base_reactions("Live")[2] - 300.0) < 1e-6


def test_membrane_floor_beside_a_shell_floor_with_the_beam_offset_between_them():
    """The beam between a membrane floor and a shell floor is meshed on its offset
    axis (inside the shell floor): the membrane's edge nodes then have nothing out
    of their plane, and the shell floor's edge nodes tied into the membrane must not
    hang on them vertically (was 'Unstable ...' naming unrelated joints)."""
    import copy as _copy
    from structstudio.mesh import MeshOptions
    m = new_building(x_spacings=(6,), y_spacings=(3, 3), story_heights=(3.0,), grid_only=False)
    floors = sorted(m.areas.values(), key=lambda a: np.mean([m.points[p].y for p in a.points]))
    sec = _copy.deepcopy(m.shell_sections[floors[0].section])
    sec.name, sec.model_type = "MEM", "Membrane"
    m.add_shell_section(sec)
    floors[0].section = "MEM"
    floors[0].max_mesh_size = 0.8
    mid = next(f for f in m.frames.values() if f.type == "Beam"
               and abs(m.points[f.point_i].y - 3) < 1e-9 and abs(m.points[f.point_j].y - 3) < 1e-9)
    mid.joint_offset_i = mid.joint_offset_j = (0.0, 0.15, 0.0)
    mid.offsets_in_analysis = True
    for a in floors:
        m.add_area_load(a.name, "Live", 4.0)
    r = analyze(m, mesh_options=MeshOptions(refine_check=False))
    assert any(t.dofs == (0, 1, 5) for t in r.am.ties) or not r.am.ties
    assert abs(r.base_reactions("Live")[2] - 4.0 * 36.0) < 1e-6
    assert abs(r.base_reactions("Dead")[2] - _self_weight(m)) < 1e-6
