"""Multiple towers (ETABS "Allow multiple towers"): every tower has its own stories and
grid systems.  Two towers that do not touch must give, in one model, exactly the
results of each tower analysed alone - joint displacements, frame forces, story drifts,
story shears, the base reactions, modal periods - also when the towers have a story at
the same elevation (generated mesh joints must join their own tower's diaphragm, the
lateral mass of the joints between levels must go to their own tower's levels)."""
import copy
import dataclasses

import numpy as np
import pytest

from structstudio import analyze
from structstudio.app.templates import new_building
from structstudio.engine.kernel import SciPyKernel
from structstudio.model import Model, ModelError
from structstudio.model.objects import GridLine, GridSystem


def _tower(heights, xs=(6.0, 6.0), ys=(5.0,), wall=True, lump=False):
    m = new_building(x_spacings=xs, y_spacings=ys, story_heights=heights, grid_only=False)
    for a in m.areas.values():  # meshed shell floors: generated joints at the story levels
        if a.type == "Floor":
            m.shell_sections[a.section].model_type = "ShellThin"
    if wall:  # a wall on line A: generated wall joints at and between the levels
        wsec = "W20" if "W20" in m.shell_sections else next(n for n, s in m.shell_sections.items() if s.type == "Wall")
        zs = [s.elevation for s in m.stories_bottom_up()]
        for z0, z1 in zip(zs, zs[1:]):
            P = [m.add_point(*c).name for c in ((0, 0, z0), (0, ys[0], z0), (0, ys[0], z1), (0, 0, z1))]
            m.add_area(P, wsec, type="Wall")
    if "Dead" not in m.load_patterns:
        m.add_load_pattern("Dead", "Dead", 1.0)
    m.add_load_pattern("EQX", "Seismic")
    top = max(s.elevation for s in m.stories.values())
    for p in m.points.values():
        if p.z > 1e-9 and abs(p.y) < 1e-9 and abs(p.x - sum(xs)) < 1e-9:
            m.add_point_load(p.name, "EQX", force=(20.0 * p.z / top, 5.0 * p.z / top, 0))
    if "Dead" not in m.load_cases:
        m.add_linear_static_case("Dead", {"Dead": 1.0})
    m.add_linear_static_case("EQX", {"EQX": 1.0})
    m.add_combination("G+E", {"Dead": 1.2, "EQX": 1.0})
    if "Modal" not in m.load_cases:
        m.add_modal_case("Modal", max_modes=6)
    m.load_cases["Modal"].max_modes = 6
    for ms in m.mass_sources.values():
        ms.lump_at_stories = lump
    return m


def merge(A: Model, B: Model, dx: float, prefix: str = "B-") -> Model:
    """A and B (moved by dx) as towers T1 and T2 of one model."""
    C = Model.from_dict(A.to_dict())
    C.add_tower("T2")
    smap = {}
    for s in B.stories_bottom_up():
        smap[s.name] = prefix + s.name
        C.add_story(prefix + s.name, s.elevation, s.height, tower="T2")
    for n, gs in B.grid_systems.items():
        C.grid_systems[prefix + n] = dataclasses.replace(gs, name=prefix + n, origin=(gs.origin[0] + dx, gs.origin[1]),
                                                         tower="T2")
    C.grid_lines += [dataclasses.replace(g, system=prefix + g.system) for g in B.grid_lines]
    for n, d in B.diaphragms.items():
        C.diaphragms.setdefault(n, copy.deepcopy(d))  # the same diaphragm name in both towers
    for n, p in B.points.items():
        q = copy.deepcopy(p)
        q.name, q.x, q.story = prefix + n, p.x + dx, smap.get(p.story)
        C.points[q.name] = q
    for n, f in B.frames.items():
        g = copy.deepcopy(f)
        g.name, g.point_i, g.point_j, g.story = prefix + n, prefix + f.point_i, prefix + f.point_j, smap.get(f.story)
        C.frames[g.name] = g
    for n, a in B.areas.items():
        b = copy.deepcopy(a)
        b.name, b.points, b.story = prefix + n, [prefix + q for q in a.points], smap.get(a.story)
        C.areas[b.name] = b
    assert not B.panel_loads and not A.panel_loads
    C.load_cases["Modal"].max_modes = 12
    C._pindex = None
    return C


def _run(m):
    return analyze(m, kernel=SciPyKernel())


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "lumped"])
def runs(request):
    lump = request.param
    A = _tower((3.5, 3.0), lump=lump)
    B = _tower((3.5, 3.5, 3.0), xs=(7.0,), ys=(6.0,), lump=lump)  # same 3.5 level, then others
    C = merge(A, B, 40.0)
    C.require_valid()
    return A, B, C, _run(A), _run(B), _run(C)


def _close(a, b, scale=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = scale if scale is not None else max(1.0, float(np.abs(b).max()))
    assert np.allclose(a, b, rtol=1e-7, atol=1e-8 * s), (a, b)


def test_tower_api():
    A = _tower((3.5, 3.0), wall=False)
    assert not A.multi_tower and A.tower_names() == ["T1"]
    C = merge(A, _tower((3.5, 3.5, 3.0), wall=False), 40.0)
    assert C.multi_tower and C.tower_names() == ["T1", "T2"]
    assert [s.name for s in C.stories_of("T2")] == ["B-Base", "B-Story1", "B-Story2", "B-Story3"]
    assert C.story_below("B-Story1").name == "B-Base" and C.story_below("Story1").name == "Base"
    # the same elevation in both towers: the tower decides
    assert C.story_at(3.5, tower="T2").name == "B-Story1"
    assert C.story_at(3.5, xy=(43.0, 2.0)).name == "B-Story1"  # inside T2's grid
    assert C.story_at(3.5, xy=(3.0, 2.0)).name == "Story1"
    assert C.story_at(7.0, xy=(3.0, 2.0)) is None  # T2's level, but inside T1: between T1's levels
    C.active_tower = "T2"
    assert C.story_at(3.5).name == "B-Story1"
    C.active_tower = None
    p = C.add_point(41.0, 1.0, 3.5)
    assert p.story == "B-Story1" and C.tower_of_point(p.name) == "T2"
    f = C.add_frame((41.0, 1.0, 0.0), (41.0, 1.0, 3.5), next(iter(C.frame_sections)))
    assert f.story == "B-Story1"
    assert C.tower_of_object("frame", "B-" + next(iter(_tower((3.5,), wall=False).frames))) == "T2"
    # saved and read back; units converted
    D = Model.from_json(C.to_json())
    assert D.tower_names() == ["T1", "T2"] and D.stories["B-Story2"].tower == "T2"
    assert D.grid_systems["B-G1"].tower == "T2"
    E = C.convert_units("kN mm")
    assert E.stories["B-Story2"].tower == "T2" and E.stories["B-Story2"].elevation == pytest.approx(7000.0)
    # a tower with objects cannot be deleted; renaming keeps its stories and systems
    with pytest.raises(ModelError):
        C.delete_tower("T2")
    C.rename_tower("T2", "North")
    assert C.stories["B-Story2"].tower == "North" and C.grid_systems["B-G1"].tower == "North"
    assert C.tower_names() == ["T1", "North"]


def test_reassign_stories_keeps_towers():
    C = merge(_tower((3.5, 3.0), wall=False), _tower((3.5, 3.5, 3.0), wall=False), 40.0)
    before = {n: p.story for n, p in C.points.items()}
    C.reassign_stories()
    assert {n: p.story for n, p in C.points.items()} == before


def test_empty_tower_delete_and_back_to_one():
    m = _tower((3.0,), wall=False)
    m.add_tower("T2")
    m.add_story("T2-Base", 0.0, 0.0, tower="T2")
    m.add_story("T2-S1", 4.0, 4.0, tower="T2")
    m.grid_systems["G2"] = GridSystem("G2", origin=(30.0, 0.0), tower="T2")
    m.grid_lines.append(GridLine(system="G2", name="Z", direction="X", ordinate=0.0))
    m.delete_tower("T2")
    assert not m.towers and "T2-S1" not in m.stories and "G2" not in m.grid_systems
    assert all(s.tower is None for s in m.stories.values())


def test_two_towers_equal_each_tower_alone(runs):
    A, B, C, ra, rb, rc = runs
    for case in ("Dead", "EQX", "G+E"):
        for src, r, pre in ((A, ra, ""), (B, rb, "B-")):
            dc = rc.displacement_envelope(case).max
            dr = r.displacement_envelope(case).max
            for n in src.points:
                _close(dc[rc._node_of_point(pre + n)], dr[r._node_of_point(n)], scale=float(np.abs(dr).max()))
            for n in src.frames:
                _close(rc.frame_forces(pre + n, case).forces, r.frame_forces(n, case).forces)
        _close(rc.base_reactions(case), np.asarray(ra.base_reactions(case)) + np.asarray(rb.base_reactions(case)))


def test_two_towers_story_tables(runs):
    A, B, C, ra, rb, rc = runs
    for case in ("EQX", "G+E"):
        dc = {(d["Story"], d["Direction"]): d for d in rc.story_drifts(case)}
        assert {d["Tower"] for d in dc.values()} == {"T1", "T2"}
        for r, pre, tw in ((ra, "", "T1"), (rb, "B-", "T2")):
            for d in r.story_drifts(case):
                e = dc[(pre + d["Story"], d["Direction"])]
                assert e["Tower"] == tw
                assert e["Drift"] == pytest.approx(d["Drift"], rel=1e-7, abs=1e-12)
                assert e["Label"] == pre + d["Label"] if d["Label"] else True
        sc = {d["Story"]: d for d in rc.story_shears(case)}
        assert len(sc) == len(rc.story_shears(case))
        for r, pre, tw in ((ra, "", "T1"), (rb, "B-", "T2")):
            for d in r.story_shears(case):
                e = sc[pre + d["Story"]]
                assert e["Tower"] == tw
                for k in ("VX", "VY", "P"):
                    assert e[k] == pytest.approx(d[k], rel=1e-7, abs=1e-6), (case, d["Story"], k)
        # the T2 shear at its first story is its whole lateral load (not T1's as well)
        assert sc["B-Story1"]["VX"] == pytest.approx(-rb.base_reactions(case)[0], rel=1e-6, abs=1e-6)


def test_two_towers_modes(runs):
    A, B, C, ra, rb, rc = runs
    pa = [d["Period"] for d in ra.modal_periods("Modal")]
    pb = [d["Period"] for d in rb.modal_periods("Modal")]
    pc = [d["Period"] for d in rc.modal_periods("Modal")]
    union = sorted(pa + pb, reverse=True)
    # the first modes of the two-tower model are the modes of the towers alone
    n = sum(1 for t in union if t >= min(max(pa[-1], pb[-1]), pc[-1]) - 1e-12)
    assert pc[:n] == pytest.approx(union[:n], rel=1e-6)


def test_generated_joints_join_their_own_diaphragm(runs):
    A, B, C, ra, rb, rc = runs
    am = rc.am
    for con in am.constraints:
        tw = C.tower_of_story(con.story)
        assert all(am.nodes[n].tower == tw for n in con.nodes), con.story
    # T1 and T2 each have a diaphragm at the 3.5 level
    at35 = {C.tower_of_story(c.story) for c in am.constraints if abs(am.nodes[c.master].xyz[2] - 3.5) < 1e-9}
    assert at35 == {"T1", "T2"}


# ------------------------------------------------------------------ podium
def _podium(towers: bool, lump: bool):
    """A two-story podium (T1) with a three-story tower on part of its roof: as two
    towers (T2 drawn in its own window: active_tower) or as one tower with every level."""
    m = _tower((3.5, 3.5), xs=(6.0, 6.0, 6.0), ys=(6.0, 6.0), wall=False, lump=lump)
    sec_c = next(n for n in m.frame_sections if n.startswith("C"))
    sec_b = next(n for n in m.frame_sections if n.startswith("B"))
    slab = next(a.section for a in m.areas.values() if a.type == "Floor")
    m.add_diaphragm("D2")
    levels = [("S3", 10.2), ("S4", 13.4)]
    if towers:
        m.add_tower("T2")
        m.add_story("T2-Base", 7.0, 0.0, tower="T2")
        prev = 7.0
        for n, z in levels:
            m.add_story("T2-" + n, z, z - prev, tower="T2")
            prev = z
        m.grid_systems["GT"] = GridSystem("GT", tower="T2")
        m.grid_lines += [GridLine("GT", f"T{k}", "X", float(x)) for k, x in enumerate((0, 6, 12))]
        m.grid_lines += [GridLine("GT", f"t{k}", "Y", float(y)) for k, y in enumerate((0, 6, 12))]
        m.active_tower = "T2"
    else:
        prev = 7.0
        for n, z in levels:
            m.add_story(n, z, z - prev)
            prev = z
    zs = [7.0] + [z for _, z in levels]
    for z0, z1 in zip(zs, zs[1:]):
        for x in (0.0, 6.0, 12.0):
            for y in (0.0, 6.0, 12.0):
                m.add_frame((x, y, z0), (x, y, z1), sec_c, type="Column")
        for y in (0.0, 6.0, 12.0):
            for x0, x1 in ((0.0, 6.0), (6.0, 12.0)):
                m.add_frame((x0, y, z1), (x1, y, z1), sec_b, type="Beam")
        for x in (0.0, 6.0, 12.0):
            for y0, y1 in ((0.0, 6.0), (6.0, 12.0)):
                m.add_frame((x, y0, z1), (x, y1, z1), sec_b, type="Beam")
        for x0, x1 in ((0.0, 6.0), (6.0, 12.0)):
            for y0, y1 in ((0.0, 6.0), (6.0, 12.0)):
                m.add_area([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], slab, type="Floor",
                           diaphragm="D2")
        top = [p for p in m.points.values() if abs(p.z - z1) < 1e-9 and p.x == 12.0 and p.y == 0.0][0]
        m.add_point_load(top.name, "EQX", force=(15.0 * z1 / 13.4, 4.0, 0))
    m.active_tower = None
    return m


@pytest.mark.parametrize("lump", [False, True], ids=["plain", "lumped"])
def test_tower_on_a_podium_equals_one_tower_with_every_level(lump):
    T = _podium(True, lump)
    S = _podium(False, lump)
    assert set(T.points) == set(S.points) and set(T.frames) == set(S.frames)
    # the tower's objects are in T2, the podium's (and its roof joints) in T1
    t2 = {n for n in T.frames if T.tower_of_object("frame", n) == "T2"}
    assert t2 == {n for n, f in S.frames.items() if f.story in ("S3", "S4")}
    roof = [p for p in T.points.values() if abs(p.z - 7.0) < 1e-9]
    assert roof and all(p.story == "Story2" and T.tower_of_point(p.name) == "T1" for p in roof)
    rt, rs = _run(T), _run(S)
    for case in ("Dead", "EQX", "G+E"):
        dt, ds = rt.displacement_envelope(case).max, rs.displacement_envelope(case).max
        for n in S.points:
            _close(dt[rt._node_of_point(n)], ds[rs._node_of_point(n)], scale=float(np.abs(ds).max()))
        for n in S.frames:
            _close(rt.frame_forces(n, case).forces, rs.frame_forces(n, case).forces)
        names = {"T2-S3": "S3", "T2-S4": "S4"}
        dr_s = {(d["Story"], d["Direction"]): d["Drift"] for d in rs.story_drifts(case)}
        for d in rt.story_drifts(case):
            assert d["Drift"] == pytest.approx(dr_s[(names.get(d["Story"], d["Story"]), d["Direction"])],
                                               rel=1e-7, abs=1e-12)
        sh_s = {d["Story"]: d for d in rs.story_shears(case)}
        rows = rt.story_shears(case)
        assert [d["Story"] for d in rows] == ["Story1", "Story2", "T2-S3", "T2-S4"]
        for d in rows:
            e = sh_s[names.get(d["Story"], d["Story"])]
            for k in ("VX", "VY", "P"):
                assert d[k] == pytest.approx(e[k], rel=1e-7, abs=1e-6), (case, d["Story"], k)
    pt = [d["Period"] for d in rt.modal_periods("Modal")]
    ps = [d["Period"] for d in rs.modal_periods("Modal")]
    assert pt == pytest.approx(ps, rel=1e-8)


# ------------------------------------------------------------------ drawing / views
def _three_towers():
    """T1: a podium (Base 0, Story1 3.5, Story2 7); T2: a tower on its roof (7, 10, 13)
    whose grid overlaps T1's; T3: a separate building at x = 40 with T1's elevations."""
    from structstudio.app.document import Document
    m = new_building(x_spacings=(6, 6), y_spacings=(6, 6), story_heights=(3.5, 3.5))
    m.add_tower_stories("T2", 2, 3.0, base_elevation=7.0)
    m.grid_systems["T2-G1"] = GridSystem("T2-G1", tower="T2")
    m.grid_lines += [GridLine("T2-G1", n, "X", x) for n, x in (("a", 0.0), ("b", 6.0))]
    m.grid_lines += [GridLine("T2-G1", n, "Y", y) for n, y in (("i", 0.0), ("ii", 6.0))]
    m.add_tower_stories("T3", 2, 3.5, base_elevation=0.0)
    m.grid_systems["T3-G1"] = GridSystem("T3-G1", origin=(40.0, 0.0), tower="T3")
    m.grid_lines += [GridLine("T3-G1", n, "X", x) for n, x in (("A", 0.0), ("B", 8.0))]
    m.grid_lines += [GridLine("T3-G1", n, "Y", y) for n, y in (("1", 0.0), ("2", 8.0))]
    return Document(m)


def test_tools_draw_in_the_tower_of_the_plan():
    from structstudio.app.tools import ColumnTool, DrawSettings, Snapper, story_levels  # noqa: F401
    doc = _three_towers()
    m = doc.model
    assert [s.name for s in m.stories_of("T2")] == ["T2-Base", "T2-Story1", "T2-Story2"]
    sec = next(n for n in m.frame_sections if n.startswith("C"))
    t = ColumnTool(doc, DrawSettings(frame_section=sec, all_stories=True))
    t.set_story("T2-Story2")
    assert t.story_names_to_draw() == ["T2-Story1", "T2-Story2"]
    assert t.click((6.0, 6.0))
    cols = [f for f in m.frames.values()]
    assert sorted(c.story for c in cols) == ["T2-Story1", "T2-Story2"]
    zs = sorted((m.points[c.point_i].z, m.points[c.point_j].z) for c in cols)
    assert zs == [(7.0, 10.0), (10.0, 13.0)]
    # the bottom joint on the podium roof: the T2 base (the tower drawn in), its top a T2 story
    assert {m.tower_of_object("frame", c.name) for c in cols} == {"T2"}
    # T3 (same elevations as T1): its columns on its own stories
    t3 = ColumnTool(doc, DrawSettings(frame_section=sec, all_stories=True))
    t3.set_story("T3-Story1")
    assert t3.story_names_to_draw() == ["T3-Story1", "T3-Story2"]
    assert t3.click((48.0, 8.0))
    new = [f for f in m.frames.values() if f not in cols]
    assert sorted(f.story for f in new) == ["T3-Story1", "T3-Story2"]
    assert all(m.points[p].story.startswith("T3-") for f in new for p in (f.point_i, f.point_j))
    # T1's plan: T1's stories only
    t1 = ColumnTool(doc, DrawSettings(frame_section=sec, all_stories=True))
    t1.set_story("Story2")
    assert t1.story_names_to_draw() == ["Story1", "Story2"]


def test_views_and_snaps_per_tower():
    from structstudio.app import scene
    from structstudio.app.snap import SnapEngine, SnapOptions
    from structstudio.app.tools import ColumnTool, DrawSettings, Plane
    doc = _three_towers()
    m = doc.model
    sec = next(n for n in m.frame_sections if n.startswith("C"))
    for story, xy in (("Story2", (0.0, 0.0)), ("T2-Story2", (6.0, 6.0)), ("T3-Story2", (40.0, 0.0))):
        t = ColumnTool(doc, DrawSettings(frame_section=sec, all_stories=True))
        t.set_story(story)
        t.click(xy)
    fr = {t: {n for n in m.frames if m.tower_of_object("frame", n) == t} for t in ("T1", "T2", "T3")}
    assert all(fr.values())
    # an elevation on y = 0 shows the columns of the window's tower only
    vis = scene.visible_sets(m, None, ("y", 0.0), tower="T3")
    assert vis[0] == {n for n in fr["T3"] if m.points[m.frames[n].point_i].y == 0.0}
    vis_all = scene.visible_sets(m, None, ("y", 0.0))
    assert vis[0] < vis_all[0]
    assert scene.visible_sets(m, None, None, tower="T2")[0] == fr["T2"]
    # snapping in T3's plan: T3's grid points only, at its level
    pl = Plane("plan", 3.5, "T3-Story1")
    eng = SnapEngine(m, pl, SnapOptions(), lambda p: (p[0], p[1]), lambda x, y: (x, y, 3.5))
    gp = {h.xyz[:2] for h in eng._point_candidates() if h.kind == "grid"}
    assert gp == {(40.0, 0.0), (48.0, 0.0), (40.0, 8.0), (48.0, 8.0)}
    eng1 = SnapEngine(m, Plane("plan", 3.5, "Story1"), SnapOptions(), lambda p: (p[0], p[1]),
                      lambda x, y: (x, y, 3.5))
    assert {h.xyz[:2] for h in eng1._point_candidates() if h.kind == "grid"} == \
        {(x, y) for x in (0.0, 6.0, 12.0) for y in (0.0, 6.0, 12.0)}
    # an elevation window of T2: its levels
    eng2 = SnapEngine(m, Plane("xz", 0.0), SnapOptions(), lambda p: (p[0], p[2]), lambda x, y: (x, 0.0, y),
                      tower="T2")
    assert sorted({h.xyz[2] for h in eng2._point_candidates() if h.kind == "grid"}) == [7.0, 10.0, 13.0]
    # the grid drawn in a plan is the story's tower's
    out = scene.build_model_actors(m, story="T3-Story1")
    assert "grids" in out.actors and not any(k.startswith("grids@") for k in out.actors)
    out3d = scene.build_model_actors(m)
    assert {"grids", "grids@1", "grids@2"} <= set(out3d.actors)


def test_tower_dialog_values_make_the_tower():
    from structstudio.app.grid_manager import add_tower_from
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,))
    add_tower_from(m, {"name": "T2", "n": 3, "typical": 3.2, "bottom": 4.5, "base": 0.0, "prefix": None,
                       "copy": True, "src": "T1", "dx": 30.0, "dy": 0.0})
    assert [(s.name, s.elevation) for s in m.stories_of("T2")] == [
        ("T2-Base", 0.0), ("T2-Story1", 4.5), ("T2-Story2", pytest.approx(7.7)), ("T2-Story3", pytest.approx(10.9))]
    assert m.grid_systems_of("T2") == ["T2-G1"] and m.grid_systems["T2-G1"].origin == (30.0, 0.0)
    assert len([g for g in m.grid_lines if g.system == "T2-G1"]) == 4
    add_tower_from(m, {"name": "T3", "n": 1, "typical": 3.0, "bottom": 3.0, "base": 2.0, "prefix": "N",
                       "copy": False, "xs": [5.0, 5.0], "ys": [4.0], "ox": -20.0, "oy": 0.0})
    assert [s.name for s in m.stories_of("T3")] == ["T3-Base", "N1"]
    assert sorted(g.ordinate for g in m.grid_lines if g.system == "T3-G1" and g.direction == "X") == [0.0, 5.0, 10.0]
    m.copy_tower("T2", "T4", dx=0.0, dy=40.0)
    assert [s.name for s in m.stories_of("T4")] == ["T4-T2-Base", "T4-T2-Story1", "T4-T2-Story2", "T4-T2-Story3"]
    assert m.grid_systems["T4-T2-G1"].origin == (30.0, 40.0)


def test_set_tower_stories_and_combine():
    doc = _three_towers()
    m = doc.model
    sec = next(n for n in m.frame_sections if n.startswith("C"))
    from structstudio.app.tools import ColumnTool, DrawSettings
    for story, xy in (("Story2", (0.0, 0.0)), ("T3-Story1", (40.0, 0.0))):
        t = ColumnTool(doc, DrawSettings(frame_section=sec, all_stories=False))
        t.set_story(story)
        t.click(xy)
    t3col = next(n for n, f in m.frames.items() if f.story == "T3-Story1")
    # rename T3's stories (same elevations as T1): T3's objects stay in T3
    m.set_tower_stories("T3", [("T3-B", 0.0), ("T3-L1", 3.5), ("T3-L2", 7.0)])
    assert m.frames[t3col].story == "T3-L1"
    assert m.frames[next(n for n, f in m.frames.items() if f.story != "T3-L1")].story == "Story2"
    with pytest.raises(ModelError):
        m.set_tower_stories("T3", [("T3-B", 0.0), ("Story1", 3.5)])  # T1's name
    # back to one tower: T3's levels at T1's elevations merge into T1's stories
    m.combine_towers("T1")
    assert not m.towers and m.frames[t3col].story == "Story1"
    assert "T3-L1" not in m.stories and "T2-Story2" in m.stories  # (10 and 13: no T1 story there)
    assert [s.name for s in m.stories_bottom_up()][:3] == ["Base", "Story1", "Story2"]


# ------------------------------------------------------------------ the windows (child process)
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"


@pytest.mark.skipif(CHILD, reason="parent only")
def test_tower_windows_in_subprocess():
    pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtk")
    import shutil
    env = dict(os.environ, STRUCTSTUDIO_GUI_CHILD="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "gui_", __file__]
    if not env.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24"] + cmd
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower() \
        and proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_towers(tmp_path, monkeypatch):
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    from structstudio.app import dialogs
    from structstudio.app.grid_manager import GridSystemsManager, add_tower_from
    from structstudio.app.mainwindow import MainWindow
    doc = _three_towers()
    w = MainWindow(doc, QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.show()
    app.processEvents()
    m = w.doc.model
    plan = next(v for v in w.views.viewports if v.config.kind == "plan")
    assert plan.tower_box.isVisibleTo(plan)
    assert [plan.tower_box.itemText(i) for i in range(plan.tower_box.count())] == ["T1", "T2", "T3"]
    plan.tower_box.setCurrentIndex(1)  # T2
    app.processEvents()
    assert plan.config.story in {s.name for s in m.stories_of("T2")}
    items = [plan.where_box.itemData(i) for i in range(plan.where_box.count())]
    assert [x for x in items if x and not str(x).startswith("__")] == ["T2-Story2", "T2-Story1", "T2-Base"]
    assert "T2 /" in plan.title.text()
    plan.activate()
    assert w.doc.active_tower == "T2"
    # an elevation window: All Towers or one tower; grid names of two systems told apart
    plan.kind_box.setCurrentIndex([plan.kind_box.itemData(i) for i in range(plan.kind_box.count())].index("xz"))
    app.processEvents()
    assert plan.tower_box.itemText(0) == "All Towers"
    names = [n for n, _, _ in plan._elevation_lines()]
    assert len(names) == len(set(names))
    plan.tower_box.setCurrentIndex(plan.tower_box.findData("T3"))
    app.processEvents()
    assert plan.config.tower == "T3" and [n for n, _, _ in plan._elevation_lines()] == ["1", "2"]
    assert w.doc.active_tower == "T3"
    for vp in w.views.viewports:
        vp.render()
    # the manager: towers, the selected tower's systems and stories
    d = GridSystemsManager(w.doc, w)
    d.show()
    app.processEvents()
    assert d.allow.isChecked() and d.towers.rowCount() == 3
    d.towers.selectRow(2)  # T3
    app.processEvents()
    assert d.system_names() == ["T3-G1"] and d.stories.item(0, 0).text() == "T3-Story2"
    seen = {}

    def fake_exec(self):
        seen["tower"] = self.tower
        seen["systems"] = list(self._systems)
        self._add_row(self.story_table, ("T3-Story3", 4.0))
        self.accept()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(dialogs.GridStoriesDialog, "exec", fake_exec)
    before = {n: (s.elevation, s.tower) for n, s in m.stories.items() if s.tower != "T3"}
    systems_before = {n: gs.tower for n, gs in m.grid_systems.items()}
    d.modify_stories()
    m = w.doc.model
    assert seen == {"tower": "T3", "systems": ["T3-G1"]}
    assert m.stories["T3-Story3"].elevation == pytest.approx(11.0) and m.stories["T3-Story3"].tower == "T3"
    assert {n: (s.elevation, s.tower) for n, s in m.stories.items() if s.tower != "T3"} == before
    assert {n: gs.tower for n, gs in m.grid_systems.items()} == systems_before
    assert d.stories.item(0, 0).text() == "T3-Story3"
    w.doc.edit(lambda mm: add_tower_from(mm, {"name": "T4", "n": 2, "typical": 3.0, "bottom": 3.0, "base": 0.0,
                                               "prefix": None, "copy": False, "xs": [5.0], "ys": [5.0],
                                               "ox": 80.0, "oy": 0.0}))
    d.refresh(tower="T4")
    assert d.towers.rowCount() == 4 and d.selected_tower() == "T4" and d.system_names() == ["T4-G1"]
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes)
    d.delete_tower()
    assert "T4" not in w.doc.model.towers
    d.close()
    app.processEvents()
    w.views.finalize()
    w.doc.dirty = False
    w.close()
    for _ in range(3):
        app.processEvents()


def test_time_history_story_drifts_per_tower():
    from structstudio.model.nonlinear import TimeHistoryLoad
    t = np.arange(0, 1.0, 0.01)
    ag = 2.0 * np.sin(2 * np.pi * 2.0 * t)

    def th(m):
        m.add_time_history_function("F", t, ag)
        m.add_time_history_case("TH", [TimeHistoryLoad("Acceleration", "UX", "F", 1.0)], n_steps=len(t) - 1,
                                dt=0.01, damping="None", nonlinear=False)
        return m
    A = th(_tower((3.5, 3.0), wall=False))
    B = th(_tower((3.5, 3.5, 3.0), xs=(7.0,), ys=(6.0,), wall=False))
    C = merge(A, B, 40.0)
    ra, rb, rc = (analyze(x, ["TH"], kernel=SciPyKernel()) for x in (A, B, C))
    hc = {(d["Story"], d["Direction"]): d for d in rc.story_drift_history("TH")}
    assert hc and {d["Tower"] for d in hc.values()} == {"T1", "T2"}
    for r, pre, tw in ((ra, "", "T1"), (rb, "B-", "T2")):
        rows = r.story_drift_history("TH")
        assert rows
        for d in rows:
            e = hc[(pre + d["Story"], d["Direction"])]
            assert e["Tower"] == tw and e["MaxDrift"] == pytest.approx(d["MaxDrift"], rel=1e-7, abs=1e-12)


def _raised(m: Model, dz: float) -> Model:
    for p in m.points.values():
        p.z += dz
    for st in m.stories.values():
        st.elevation += dz
    m._pindex = None
    return m


def test_a_tower_on_higher_ground_has_its_wall_base_restrained():
    """B stands on the ground 2 m higher (a sloping site): its wall's generated base
    joints are restrained as when it is analysed alone; a tower on a podium is not."""
    A = _tower((3.5, 3.0))
    B = _raised(_tower((3.5, 3.5, 3.0), xs=(7.0,), ys=(6.0,)), 2.0)
    for x in (A, B):
        x.options.auto_base_restraints = True  # (ETABS: generated joints at the base restrained)
    C = merge(A, B, 40.0)
    ra, rb, rc = _run(A), _run(B), _run(C)
    for case in ("Dead", "EQX"):
        dc, dr = rc.displacement_envelope(case).max, rb.displacement_envelope(case).max
        for n in B.points:
            _close(dc[rc._node_of_point("B-" + n)], dr[rb._node_of_point(n)], scale=float(np.abs(dr).max()))
        _close(rc.base_reactions(case), np.asarray(ra.base_reactions(case)) + np.asarray(rb.base_reactions(case)))
    am = rc.am
    gen = [n for n in am.nodes if n.point is None and n.tower == "T2" and abs(n.xyz[2] - 2.0) < 1e-9]
    assert gen and all(n.restraints.all() for n in gen)
    # podium: the tower's generated joints at the podium roof stay free
    T = _podium(True, False)
    T.options.auto_base_restraints = True
    rt = _run(T)
    roof = [n for n in rt.am.nodes if n.point is None and abs(n.xyz[2] - 7.0) < 1e-9]
    assert not any(n.restraints.any() for n in roof)


def test_slab_design_model_takes_its_own_tower():
    from structstudio.design.slab_model import build_slab_model, membrane_stories

    def mem(m):
        for a in m.areas.values():
            if a.type == "Floor":
                m.shell_sections[a.section].model_type = "Membrane"
        return m
    A = mem(_tower((3.5, 3.0), wall=False))
    B = mem(_tower((3.5, 3.5, 3.0), xs=(7.0,), ys=(6.0,), wall=False))
    C = merge(A, B, 40.0)
    assert membrane_stories(C) == ["Story1", "B-Story1", "Story2", "B-Story2", "B-Story3"] or \
        sorted(membrane_stories(C)) == sorted(["Story1", "Story2", "B-Story1", "B-Story2", "B-Story3"])
    for story, src, pre in (("Story1", A, ""), ("B-Story1", B, "B-")):
        sm = build_slab_model(C, story).model
        alone = build_slab_model(src, story[len(pre):]).model
        assert sorted(sm.areas) == sorted(pre + n for n in alone.areas)
        assert sorted(sm.frames) == sorted(pre + n.split("~")[0] + ("~" + n.split("~")[1] if "~" in n else "")
                                           for n in alone.frames)


def test_several_towers_keep_stories_mode():
    doc = _three_towers()
    with pytest.raises(ModelError):
        doc.model.set_grid_mode("xyz")
    assert doc.model.grid_mode == "story"
