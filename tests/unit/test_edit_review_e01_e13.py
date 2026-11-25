"""Edit menu review E01 - E13 (25 Nov 2026): the commands that changed loads, supports,
stiffness or coordinates without the user knowing.  Each test builds a small model,
checks the forces the analysis gets (reactions, load positions and moments, the
transfer to the receiving frames, the stiffness in the analysis mesh) - not only the
shape on screen - and Undo / Redo and Save / Open where the command changes the model.
On version 9133333 every test here fails with the symptom of its code."""
import json
import math

import pytest

from structstudio import Model, analyze
from structstudio.app import commands as C
from structstudio.app import modify as M
from structstudio.app import shell_edit as S
from structstudio.app.document import Document
from structstudio.app.tools import Plane, delete_objects, deleted_joint_users, replicate_to_stories
from structstudio.engine.kernel import SciPyKernel
from structstudio.model import FIXED, FrameSection, ShellSection
from structstudio.model.frame_ops import resultant
from structstudio.model.objects import LineSpring, PanelLoadCase


def model(depth=0.30):
    m = Model("Audit", "kN m")
    m.add_material("C", E=25e6, nu=0.2, weight_density=24)
    m.add_frame_section(FrameSection.rectangular("B", "C", depth, 0.20))
    m.add_shell_section(ShellSection("Slab", "C", thickness=0.15))
    m.add_load_pattern("Q", "Live")
    m.set_stories([("L1", 3), ("L2", 3), ("L3", 3)])
    return m


def panel(m, x=0.0, name="S", z=3.0, y=0.0):
    return m.add_area([(x, y, z), (x + 2, y, z), (x + 2, y + 2, z), (x, y + 2, z)], "Slab", name=name)


def reactions(m):
    if "Q" not in m.load_cases:
        m.add_linear_static_case("Q")
    r = analyze(m, cases=["Q"], kernel=SciPyKernel())
    return [round(float(v), 6) for v in r.base_reactions("Q")]


def global_moment_y(m):
    """Sum over the frames of (gravity force x global x of the force) - the moment of the
    frame loads about the line x = 0 (frames along global X)."""
    tot_f = tot_m = 0.0
    for f in m.frames.values():
        a, b = m.points[f.point_i], m.points[f.point_j]
        L = math.dist(a.xyz, b.xyz)
        F, Mi = resultant(f.loads, L, "Gravity")
        ux = (b.x - a.x) / L if L else 0.0
        tot_f += F
        tot_m += F * a.x + Mi * ux
    return tot_f, tot_m


# --------------------------------------------------------------------------- E01
def _trim_model(load):
    m = model()
    d = Document(m)
    f = m.add_frame((0, 0, 3), (10, 0, 3), "B", name="Loaded")
    for x in (4, 6):
        m.add_frame((x, -1, 3), (x, 1, 3), "B")
    load(m, f.name)
    for p in m.points.values():
        p.restraints = FIXED
    return m, d


def test_e01_trim_keeps_a_point_load_once_where_it_acts():
    m, d = _trim_model(lambda m, f: m.add_frame_point_load(f, "Q", 10, rel=0.2))
    before = reactions(m)
    F0, M0 = global_moment_y(m)
    assert M.trim_frame(d, "Loaded", Plane("plan", 3), (5, 0, 3)) == "Trimmed (frame split in two)"
    m = d.model
    F1, M1 = global_moment_y(m)
    assert (F1, M1) == pytest.approx((F0, M0)) and F1 == pytest.approx(10.0) and M1 == pytest.approx(20.0)
    pts = [(m.points[f.point_i].x + ld.rel_a * math.dist(m.points[f.point_i].xyz, m.points[f.point_j].xyz))
           for f in m.frames.values() for ld in f.loads if ld.kind == "point"]
    assert pts == pytest.approx([2.0])
    for p in m.points.values():
        p.restraints = FIXED
    assert reactions(m) == pytest.approx(before)
    d.undo()
    assert len(d.model.frames) == 3 and global_moment_y(d.model) == pytest.approx((F0, M0))
    d.redo()
    assert global_moment_y(d.model) == pytest.approx((F0, M0))


def test_e01_trim_cuts_a_triangular_load_where_the_frame_is_cut():
    m, d = _trim_model(lambda m, f: m.add_frame_distributed_load(f, "Q", 0, value_b=10))
    M.trim_frame(d, "Loaded", Plane("plan", 3), (5, 0, 3))
    m = d.model
    rows = sorted((m.points[f.point_i].x, m.points[f.point_j].x, ld.value_a, ld.value_b)
                  for f in m.frames.values() for ld in f.loads)
    assert rows == [pytest.approx((0, 4, 0, 4)), pytest.approx((6, 10, 6, 10))]
    F, Mx = global_moment_y(m)
    # 0..4: 8 kN at 8/3; 6..10: 32 kN at 6 + 4 * (6 + 2 * 10) / (3 * 16)
    assert F == pytest.approx(40.0)
    assert Mx == pytest.approx(8 * 8 / 3 + 32 * (6 + 4 * 26 / 48))
    for p in m.points.values():
        p.restraints = FIXED
    assert reactions(m)[2] == pytest.approx(40.0)


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("click", [1.0, 9.0])
def test_e01_trim_an_end_absolute_or_relative_and_releases(absolute, click):
    def load(m, f):
        m.add_frame_point_load(f, "Q", 10, rel=2.0 if absolute else 0.2, absolute=absolute)
        m.add_frame_point_load(f, "Q", 5, rel=0.8)
        m.frames[f].releases_i = (False, False, False, False, False, True)
        m.frames[f].releases_j = (False, False, False, False, True, True)
    m, d = _trim_model(load)
    F0, M0 = global_moment_y(m)
    M.trim_frame(d, "Loaded", Plane("plan", 3), (click, 0, 3))
    m = d.model
    f = m.frames["Loaded"]
    xi, xj = m.points[f.point_i].x, m.points[f.point_j].x
    F, Mx = global_moment_y(m)
    if click < 4:  # end I cut away (x 0..4): the 10 kN at x = 2 goes, the 5 kN at x = 8 stays
        assert (xi, xj) == pytest.approx((4, 10)) and (F, Mx) == pytest.approx((5.0, 40.0))
        assert not any(f.releases_i) and f.releases_j == (False, False, False, False, True, True)
    else:  # end J cut away (x 6..10)
        assert (xi, xj) == pytest.approx((0, 6)) and (F, Mx) == pytest.approx((10.0, 20.0))
        assert f.releases_i == (False, False, False, False, False, True) and not any(f.releases_j)
    assert all(ld.absolute == (absolute and ld.value_a == 10) for ld in f.loads)
    d.undo()
    assert global_moment_y(d.model) == pytest.approx((F0, M0))


def test_e01_extend_keeps_loads_where_they_act():
    m = model()
    d = Document(m)
    f = m.add_frame((0, 0, 3), (4, 0, 3), "B")
    m.add_frame((6, -1, 3), (6, 1, 3), "B")
    m.add_frame_point_load(f.name, "Q", 10, rel=0.5)
    m.add_frame_distributed_load(f.name, "Q", 2.0)
    F0, M0 = global_moment_y(m)
    M.extend_frame(d, f.name, Plane("plan", 3), (3.9, 0, 3))
    assert d.model.points[d.model.frames[f.name].point_j].x == pytest.approx(6.0)
    assert global_moment_y(d.model) == pytest.approx((F0, M0))


# --------------------------------------------------------------------------- E02
def _ring(m, x0=0.0, name="S"):
    s = panel(m, x0, name)
    frames = [m.add_frame(s.points[i], s.points[(i + 1) % 4], "B").name for i in range(4)]
    return s, frames


def test_e02_radial_copy_uses_its_own_receivers_and_turned_span():
    from structstudio.mesh.panel_loads import panel_loads_to_members
    m = model()
    d = Document(m)
    s, frames = _ring(m)
    p = m.add_area_panel(s.name, [PanelLoadCase("Q", 2.5, direction="Gravity")], distribution="one_way",
                         span_dir=(1, 0, 0))
    p.receivers = list(frames)
    for q in m.points.values():
        q.restraints = FIXED
    assert reactions(m)[2] == pytest.approx(10.0)
    made = C.replicate_transform(d, frames, [s.name], [], C.radial_transforms(-3, 0, 90, 1))
    m = d.model
    cp = next(q for q in m.panel_loads.values() if q.area != s.name)
    assert sorted(cp.receivers) == sorted(n for n in made if n in m.frames)
    assert cp.span_dir == pytest.approx((0, 1, 0), abs=1e-12) and cp.loads[0].direction == "Gravity"
    for q in m.points.values():
        q.restraints = FIXED
    per_panel = panel_loads_to_members(m)[2]
    assert all(v["Q"] == pytest.approx([10.0, 10.0]) for v in per_panel.values()) and len(per_panel) == 2
    assert reactions(m)[2] == pytest.approx(20.0)
    d.undo()
    assert len(d.model.panel_loads) == 1
    d.redo()
    assert len(d.model.panel_loads) == 2


def test_e02_copy_of_the_slab_only_takes_the_frames_where_it_lands_and_x_loads_turn():
    m = model()
    d = Document(m)
    s, frames = _ring(m)
    _ring(m, 10.0, "S2")  # frames already at the copy's place
    p = m.add_area_panel(s.name, [PanelLoadCase("Q", 4, direction="X")], distribution="one_way",
                         span_dir=(1, 0, 0))
    p.receivers = list(frames)
    C.replicate_transform(d, [], [s.name], [], C.linear_transforms(10, 0, 0, 1))
    m = d.model
    cp = next(q for q in m.panel_loads.values() if q.area not in (s.name, "S2"))
    xs = sorted(round(min(m.points[m.frames[n].point_i].x, m.points[m.frames[n].point_j].x), 6)
                for n in cp.receivers)
    assert len(cp.receivers) == 4 and min(xs) >= 10.0
    # a 90 deg copy: global X load becomes Y, gravity stays down
    C.replicate_transform(d, frames, [s.name], [], C.radial_transforms(-3, 0, 90, 1))
    turned = [q for q in d.model.panel_loads.values() if q.loads[0].direction == "Y"]
    assert len(turned) == 1 and turned[0].loads[0].value == pytest.approx(4.0)


def test_e02_paste_maps_receivers_to_the_pasted_frames():
    m = model()
    d = Document(m)
    s, frames = _ring(m)
    p = m.add_area_panel(s.name, [PanelLoadCase("Q", 2.5)], distribution="one_way", span_dir=(1, 0, 0))
    p.receivers = list(frames)
    clip = C.copy_objects(m, frames, [s.name], [])
    fr, ar, pts = C.paste_objects(d, clip, 0, 5, 0)
    cp = next(q for q in d.model.panel_loads.values() if q.area == ar[0])
    assert sorted(cp.receivers) == sorted(fr)


# --------------------------------------------------------------------------- E03, E11
@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
def test_e03_merge_refuses_shells_with_other_loads_or_springs(order):
    m = model()
    d = Document(m)
    a, b = panel(m, 0, "A"), panel(m, 2, "B")
    a.spring, b.spring = 25000, 0
    m.add_area_load("A", "Q", 5)
    m.add_area_load("B", "Q", 1)
    for p in m.points.values():
        p.restraints = FIXED
    before = reactions(m)
    k = len(d._undo)
    names, notes = S.merge_shells(d, list(order))
    assert names == [] and len(d._undo) == k and len(d.model.areas) == 2
    assert "A+B" in notes[0] and "area spring (25000 / 0)" in notes[0] and "area loads" in notes[0]
    assert reactions(d.model) == pytest.approx(before) and before[2] == pytest.approx(24.0)


def test_e03_merge_of_alike_shells_keeps_the_load_and_spring():
    m = model()
    d = Document(m)
    a, b = panel(m, 0, "A"), panel(m, 2, "B")
    a.spring = b.spring = 25000
    m.add_area_load("A", "Q", 3)
    m.add_area_load("B", "Q", 3)
    for p in m.points.values():
        p.restraints = FIXED
    before = reactions(m)
    names, notes = S.merge_shells(d, ["B", "A"])
    assert names == ["A"] and not notes and len(d.model.areas) == 1
    assert d.model.areas["A"].spring == 25000
    assert reactions(d.model) == pytest.approx(before) and before[2] == pytest.approx(24.0)
    d.undo()
    assert set(d.model.areas) == {"A", "B"}


def test_e03_merge_refuses_a_spring_difference_alone():
    m = model()
    d = Document(m)
    panel(m, 0, "A")
    b = panel(m, 2, "B")
    b.spring = 25000
    names, notes = S.merge_shells(d, ["A", "B"])
    assert names == [] and "area spring" in notes[0]


def test_e11_merge_of_shells_lying_on_each_other_reports():
    m = model()
    d = Document(m)
    s = panel(m)
    m.add_area(list(s.points), "Slab", name="Overlapping")
    names, notes = S.merge_shells(d, [s.name, "Overlapping"])
    assert names == [] and "lie on each other" in notes[0] and len(d.model.areas) == 2


# --------------------------------------------------------------------------- E05, E06
def test_e05_replicate_to_stories_keeps_every_shell_and_frame_property(tmp_path):
    m = model()
    d = Document(m)
    s = panel(m)
    s.spring, s.max_mesh_size, s.mesh_divisions, s.pier, s.fine_mesh = 25000, 0.25, (3, 2), "PW", True
    f = m.add_frame((0, 0, 3), (2, 0, 3), "B")
    f.releases_j = (False, False, False, False, True, True)
    f.end_offset_i = 0.1
    made = replicate_to_stories(d, [f.name], [s.name], [], ["L2"])
    m = d.model
    b = next(m.areas[n] for n in made if n in m.areas)
    g = next(m.frames[n] for n in made if n in m.frames)
    assert b.story == "L2" and m.points[b.points[0]].z == pytest.approx(6.0)
    assert (b.spring, b.max_mesh_size, b.mesh_divisions, b.pier, b.fine_mesh) == (25000, 0.25, (3, 2), "PW", True)
    assert g.releases_j == f.releases_j and g.end_offset_i == 0.1
    path = str(tmp_path / "m.json")
    d.save(path)
    d2 = Document()
    d2.open(path)
    b2 = d2.model.areas[b.name]
    assert (b2.spring, b2.max_mesh_size, tuple(b2.mesh_divisions), b2.pier, b2.fine_mesh) == \
        (25000, 0.25, (3, 2), "PW", True)


@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
def test_e06_several_stories_each_object_onto_the_target_story(order):
    m = model()
    panel(m, 0, "A", 3)
    panel(m, 4, "B", 6)
    d = Document(m)
    names = replicate_to_stories(d, [], list(order), [], ["L3"])
    got = sorted((d.model.areas[n].story, d.model.points[d.model.areas[n].points[0]].z) for n in names)
    assert got == [("L3", 9.0), ("L3", 9.0)]


def test_e06_several_stories_as_one_block_from_a_chosen_story():
    m = model()
    panel(m, 0, "A", 3)
    panel(m, 4, "B", 6)
    d = Document(m)
    names = replicate_to_stories(d, [], ["B", "A"], [], ["L2"], base_story="L1")
    got = sorted((d.model.areas[n].story, d.model.points[d.model.areas[n].points[0]].z) for n in names)
    assert got == [("L2", 6.0), ("L3", 9.0)]  # each copy named after the level it is on


# --------------------------------------------------------------------------- E07
def test_e07_paste_across_models_asks_about_a_section_of_the_same_name():
    src, dst = model(0.30), model(0.60)
    f = src.add_frame((0, 0, 3), (4, 0, 3), "B")
    clip = C.copy_objects(src, [f.name], [], [])
    d = Document(dst)
    assert C.definition_conflicts(dst, clip) == [("frame_sections", "B")]
    with pytest.raises(Exception) as e:
        C.paste_objects(d, clip, 0, 0, 0)
    assert "same name" in str(e.value)
    assert not d.model.frames  # nothing changed
    fr = C.paste_objects(d, clip, 0, 0, 0, on_conflict="rename")[0]
    sec = d.model.frames[fr[0]].section
    assert sec == "B (2)" and d.model.frame_sections[sec].I33 == pytest.approx(src.frame_sections["B"].I33)
    assert d.model.frame_sections["B"].I33 == pytest.approx(0.2 * 0.6 ** 3 / 12)
    d.undo()
    fr = C.paste_objects(d, clip, 0, 0, 0, on_conflict="existing")[0]
    assert d.model.frames[fr[0]].section == "B" and "B (2)" not in d.model.frame_sections


def test_e07_the_same_section_in_other_units_is_no_conflict():
    from structstudio.units import UnitSystem
    src = model(0.30)
    f = src.add_frame((0, 0, 3), (4, 0, 3), "B")
    clip = C.copy_objects(src, [f.name], [], [])
    dst = model(0.30).convert_units(UnitSystem.parse("kN mm"))
    assert C.definition_conflicts(dst, clip) == []


# --------------------------------------------------------------------------- E08
def test_e08_cut_keeps_drawn_panels_on_the_clipboard():
    m = model()
    d = Document(m)
    f = m.add_frame((0, 0, 3), (4, 0, 3), "B")
    pl = m.add_panel_load([(0, 0, 3), (4, 0, 3), (4, 2, 3), (0, 2, 3)], distribution="one_way",
                          span_dir=(1, 0, 0), loads=[PanelLoadCase("Q", 2.0)])
    clip = C.copy_objects(m, [f.name], [], [], panels=[pl.name])
    assert len(clip.free_panels) == 1 and not clip.empty
    delete_objects(d, frames=[f.name], panels=[pl.name])
    assert pl.name not in d.model.panel_loads
    fr, ar, pts = C.paste_objects(d, clip, 10, 0, 0)
    new = [q for q in d.model.panel_loads.values() if not q.area]
    assert len(new) == 1 and new[0].points[0] == pytest.approx((10, 0, 3)) and new[0].span_dir == (1, 0, 0)
    assert new[0].loads[0].pattern == "Q" and new[0].loads[0].value == 2.0
    d.undo()
    d.redo()
    assert len([q for q in d.model.panel_loads.values() if not q.area]) == 1


# --------------------------------------------------------------------------- E09
def test_e09_trim_and_extend_leave_joints_elsewhere():
    m, d = _trim_model(lambda m, f: None)
    m.add_point(33, 4, 3, name="Survey")
    m.add_point(32, 4, 3, name="RotSpring", rotational_springs=(100, 0, 0))
    M.trim_frame(d, "Loaded", Plane("plan", 3), (5, 0, 3))
    assert {"Survey", "RotSpring"} <= set(d.model.points)
    m = model()
    d = Document(m)
    f = m.add_frame((0, 0, 3), (4, 0, 3), "B")
    m.add_frame((6, -1, 3), (6, 1, 3), "B")
    m.add_point(31, 0, 3, name="Remote", rotational_springs=(150, 0, 0))
    m.add_point(30, 0, 3, name="Free")
    M.extend_frame(d, f.name, Plane("plan", 3), (3.9, 0, 3))
    assert {"Remote", "Free"} <= set(d.model.points)


# --------------------------------------------------------------------------- E10
def _spring_model():
    m = model()
    a, b = m.add_point(0, 0, 0), m.add_point(4, 0, 0)
    m.line_springs["LS"] = LineSpring("LS", a.name, b.name, (0, 0, 200))
    return m, a.name


def _mesh_kz(m):
    from structstudio.mesh.mesher import build_analysis_model
    return round(sum(float(n.springs[2]) for n in build_analysis_model(m).nodes) / 1000.0, 6)  # (mesh in N: kN/m)


def test_e10_delete_a_line_spring_joint_takes_the_spring_or_keeps_the_joint():
    m, p = _spring_model()
    assert _mesh_kz(m) == pytest.approx(800.0)
    assert deleted_joint_users(m, points=[p]) == {"line_springs": ["LS"], "links": []}
    d = Document(m)
    n = delete_objects(d, points=[p], with_users=False)  # keep: nothing is left dangling
    assert n == 0 and p in d.model.points and "LS" in d.model.line_springs
    assert _mesh_kz(d.model) == pytest.approx(800.0)
    n = delete_objects(d, points=[p])  # with the spring
    assert n == 1 and delete_objects.last_joints == 1
    assert "LS" not in d.model.line_springs and p not in d.model.points and not d.model.validate()
    d.undo()
    assert "LS" in d.model.line_springs and _mesh_kz(d.model) == pytest.approx(800.0)


def test_e10_validate_names_a_line_spring_without_its_joint():
    m, p = _spring_model()
    del m.points[p]
    errs = [str(e) for e in m.validate()]
    assert any("Line spring 'LS'" in e and p in e for e in errs)


# --------------------------------------------------------------------------- E13
def test_e13_undo_redo_back_to_the_saved_state_clears_the_star(tmp_path):
    path = str(tmp_path / "audit.json")
    d = Document(model())
    d.save(path)
    same = lambda: json.loads(open(path, encoding="utf-8-sig").read()) == json.loads(d.model.to_json())  # noqa: E731
    d.edit(lambda m: m.add_point(4, 2, 3))
    assert d.dirty and d.title.endswith("*")
    d.undo()
    assert same() and not d.dirty and d.title == "audit.json"
    d.redo()
    assert d.dirty
    d.save(path)
    d.undo()
    assert d.dirty
    d.redo()
    assert same() and not d.dirty
    d.edit(lambda m: m.add_point(5, 2, 3))  # a new step after the saved one: unsaved
    d.undo()
    assert not d.dirty
    d.edit(lambda m: m.add_point(6, 2, 3))
    assert d.dirty
    # an edit that changes nothing keeps the saved state
    d.save(path)
    d.edit(lambda m: None)
    assert not d.dirty
