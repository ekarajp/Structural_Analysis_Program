"""Regression tests for the function audit of 2026-11-09 (FA01-FA15) and the
drawing review (duplicate members, Check Model, story snap in elevations, 3D
typing messages, undo snapshots, edit rollback).

Every test here fails on the audited baseline (commit 3d8e746 / 0072afc) and
passes on the candidate.  The window tests (FA04 and the duplicate warning) run
in a child interpreter like the other desktop tests.
"""
import dataclasses
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from structstudio.app import commands as C  # noqa: E402
from structstudio.app import modify as M  # noqa: E402
from structstudio.app import tools as T  # noqa: E402
from structstudio.app.document import Document  # noqa: E402
from structstudio.app.templates import new_building  # noqa: E402
from structstudio.model.base import ModelError  # noqa: E402
from structstudio.model.nonlinear import LinkProperty  # noqa: E402
from structstudio.model.objects import Frame, FrameLoad, GroundDisplacement, Group, LineSpring, PointLoad  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"
gui = pytest.mark.skipif(not CHILD, reason="runs in the child interpreter (test_window_cases_in_subprocess)")
ASSIGN = ("restraints", "springs", "rotational_springs", "mass", "rotational_mass", "diaphragm", "local_axes",
          "loads", "ground_displacements")


def doc(units="kN m"):
    d = Document(new_building(x_spacings=(10,), y_spacings=(5,), story_heights=(3, 3)))
    if units != "kN m":
        d.change_units(units)
        d._undo.clear()
    return d


def seed(p):
    p.restraints = (True, False, True, False, False, False)
    p.springs, p.rotational_springs = (100.0, 200.0, 300.0), (4.0, 5.0, 6.0)
    p.mass, p.rotational_mass = (7.0, 8.0, 9.0), (1.0, 2.0, 3.0)
    p.diaphragm, p.local_axes = "D1", (10.0, 20.0, 30.0)
    p.loads = [PointLoad("Live", (11.0, 12.0, 13.0), (1.0, 2.0, 3.0))]
    p.ground_displacements = [GroundDisplacement("Live", (0.001, 0.0, 0.0))]
    return p


def assigns(p):
    v = dataclasses.asdict(p)
    return {k: v[k] for k in ASSIGN}


def resultant(loads, L):
    """Independent oracle: (total force, moment about end I) of the gravity force
    loads, integrated numerically (not the code under test)."""
    xs = np.linspace(0.0, L, 20001)
    F = Mo = 0.0
    for ld in loads:
        if ld.direction != "Gravity" or ld.load_type != "Force":
            continue
        a, b = (ld.rel_a, ld.rel_b) if ld.absolute else (ld.rel_a * L, ld.rel_b * L)
        if ld.kind == "point":
            F += ld.value_a
            Mo += ld.value_a * a
        elif ld.kind == "distributed" and b > a:
            x = np.linspace(a, b, 4001)
            w = ld.value_a + (ld.value_b - ld.value_a) * (x - a) / (b - a)
            F += np.trapezoid(w, x)
            Mo += np.trapezoid(w * x, x)
    del xs
    return F, Mo


def length(m, f):
    return math.dist(m.points[f.point_i].xyz, m.points[f.point_j].xyz)


def global_resultant(m, frames):
    """(sum Fz, sum Fz * x) of gravity force loads on ``frames``, in global X."""
    F = Mx = 0.0
    for n in frames:
        f = m.frames[n]
        L = length(m, f)
        Fi, Mi = resultant(f.loads, L)
        xi, xj = m.points[f.point_i].x, m.points[f.point_j].x
        ux = (xj - xi) / L
        F += Fi
        Mx += Fi * xi + Mi * ux
    return F, Mx


# ---------------------------------------------------------------- FA05
@pytest.mark.parametrize("op", ["paste", "replicate", "story", "copy_tool", "replicate_linear"])
def test_fa05_copies_keep_every_joint_assignment(op):
    d = doc()
    p = seed(d.model.add_point(0, 0, 3))
    want = assigns(p)
    if op == "paste":
        C.paste_objects(d, C.copy_objects(d.model, [], [], [p.name]), 20, 0, 0)
        q = d.model.find_point(20, 0, 3)
    elif op == "replicate":
        C.replicate_transform(d, [], [], [p.name], C.linear_transforms(20, 0, 0, 2))
        q = d.model.find_point(40, 0, 3)
    elif op == "story":
        T.replicate_to_stories(d, [], [], [p.name], ["Story2"])
        q = d.model.find_point(0, 0, 6)
    elif op == "copy_tool":
        M.copy_transformed(d, [], [], [p.name], M.translation((20, 0, 0)))
        q = d.model.find_point(20, 0, 3)
    else:
        T.replicate_linear(d, [], [], [p.name], 20, 0, 0, 1)
        q = d.model.find_point(20, 0, 3)
    assert q is not None and assigns(q) == want
    assert len(d._undo) == 1


def test_fa05_copy_onto_an_existing_joint_keeps_that_joint_and_says_so():
    d = doc()
    p = seed(d.model.add_point(0, 0, 3))
    other = d.model.add_point(20, 0, 3)
    other.mass = (1.0, 1.0, 1.0)
    C.paste_objects(d, C.copy_objects(d.model, [], [], [p.name]), 20, 0, 0)
    q = d.model.points[other.name]
    assert q.mass == (1.0, 1.0, 1.0) and not q.loads
    assert any("existing joints" in t for _, t in d.notes)


# ---------------------------------------------------------------- FA06
def _two_joints():
    d = doc()
    m = d.model
    a = m.add_point(0, 0, 3)
    b = m.add_point(0.0001, 0, 3, merge=False)
    return d, m, a, b


def test_fa06_merge_takes_assignments_and_retargets_every_reference():
    d, m, a, b = _two_joints()
    b.mass, b.springs, b.diaphragm = (7, 8, 9), (100, 200, 300), "D1"
    b.loads = [PointLoad("Live", (0, 0, -5))]
    m.groups["G"] = Group("G", points=[b.name, a.name])
    m.add_link_property(LinkProperty("LP"))
    lk = m.add_link(b, None, "LP")
    c = m.add_point(5, 0, 3)
    m.line_springs["LS"] = LineSpring("LS", b.name, c.name, (1, 1, 1))
    f = m.add_frame(b, c, "B30x60")
    assert C.merge_joints(d, [a.name, b.name], 0.001) == 1
    m = d.model
    q = m.points[a.name]
    assert (q.mass, q.springs, q.diaphragm) == ((7, 8, 9), (100, 200, 300), "D1")
    assert len(q.loads) == 1 and b.name not in m.points
    assert m.groups["G"].points == [a.name]
    assert m.links[lk.name].point_i == a.name
    assert m.line_springs["LS"].point_i == a.name and m.frames[f.name].point_i == a.name
    d.undo()
    assert b.name in d.model.points and d.model.points[b.name].mass == (7, 8, 9)


def test_fa06_identical_duplicates_merge_once_and_conflicts_are_refused_untouched():
    d, m, a, b = _two_joints()
    for p in (a, b):
        p.mass, p.loads = (7, 8, 9), [PointLoad("Live", (0, 0, -5))]
    C.merge_joints(d, [a.name, b.name], 0.001)
    q = d.model.points[a.name]
    assert q.mass == (7, 8, 9) and len(q.loads) == 1  # duplicates are not doubled
    d, m, a, b = _two_joints()
    a.diaphragm, b.diaphragm = "D1", "D2"
    m.add_diaphragm("D2")
    before = m.to_json()
    with pytest.raises(ModelError, match="diaphragm"):
        C.merge_joints(d, [a.name, b.name], 0.001)
    assert d.model.to_json() == before and not d._undo


def test_fa06_grip_drag_onto_a_joint_merges_like_merge_joints():
    d = doc()
    m = d.model
    a = m.add_point(0, 0, 3)
    b = m.add_point(2, 0, 3)
    b.mass, b.springs = (7, 8, 9), (100, 200, 300)
    m.groups["G"] = Group("G", points=[b.name])
    M.move_joint(d, b.name, a.xyz)
    q = d.model.points[a.name]
    assert q.mass == (7, 8, 9) and q.springs == (100, 200, 300)
    assert d.model.groups["G"].points == [a.name]


def test_fa06_failed_grip_drag_leaves_the_model_and_undo_as_they_were():
    d = doc()
    m = d.model
    m.add_diaphragm("D2")
    a = m.add_point(0, 0, 3)
    b = m.add_point(2, 0, 3)
    a.diaphragm, b.diaphragm = "D1", "D2"
    before = m.to_json()
    with pytest.raises(ModelError):
        M.move_joint(d, b.name, a.xyz)
    assert d.model.to_json() == before and not d._undo


# ---------------------------------------------------------------- FA07
def _chain(d, loads_f, loads_g, flip_g=False):
    m = d.model
    a, b, c = m.add_point(0, 0, 3), m.add_point(3, 0, 3), m.add_point(10, 0, 3)
    f = m.add_frame(a, b, "B30x60")
    g = m.add_frame(c, b, "B30x60") if flip_g else m.add_frame(b, c, "B30x60")
    f.loads, g.loads = loads_f, loads_g
    return f, g


@pytest.mark.parametrize("flip", [False, True])
def test_fa07_join_keeps_force_and_moment_of_every_load(flip):
    d = doc()
    f, g = _chain(d, [FrameLoad("Live", value_a=2, value_b=2), FrameLoad("Live", kind="point", value_a=5, rel_a=.5)],
                  [FrameLoad("Live", value_a=8, value_b=4, rel_a=.2, rel_b=.9),
                   FrameLoad("Live", kind="point", value_a=7, rel_a=1.5, absolute=True)], flip_g=flip)
    before = global_resultant(d.model, [f.name, g.name])
    assert before[0] == pytest.approx(2 * 3 + 6 * 0.7 * 7 + 5 + 7)
    joined = C.join_frames(d, [f.name, g.name])
    assert joined == [g.name] and len(d.model.frames) == 1
    after = global_resultant(d.model, [f.name])
    assert after == pytest.approx(before)
    assert length(d.model, d.model.frames[f.name]) == pytest.approx(10)


def test_fa07_join_refuses_pairs_that_would_lose_an_end_condition():
    d = doc()
    f, g = _chain(d, [], [])
    f.releases_j = (False, False, False, False, False, True)
    before = d.model.to_json()
    assert C.join_frames(d, [f.name, g.name]) == []
    assert any("release" in t for lvl, t in d.notes if lvl == "warning")
    assert json.loads(d.model.to_json())["frames"] == json.loads(before)["frames"]


def test_fa07_join_keeps_the_outer_end_offsets():
    d = doc()
    f, g = _chain(d, [], [])
    f.end_offset_i, g.end_offset_j = 0.2, 0.3
    C.join_frames(d, [f.name, g.name])
    h = d.model.frames[f.name]
    assert (h.end_offset_i, h.end_offset_j) == (0.2, 0.3)


# ---------------------------------------------------------------- FA08
@pytest.mark.parametrize("units,k", [("kN m", 1.0), ("kN mm", 1000.0), ("kip ft", 1 / 0.3048)])
def test_fa08_reverse_keeps_absolute_and_relative_load_positions(units, k):
    d = doc(units)
    m = d.model
    f = m.add_frame((0, 0, 3 * k), (10 * k, 0, 3 * k), next(iter(m.frame_sections)))
    f.loads = [FrameLoad("Live", kind="point", value_a=5, rel_a=2 * k, absolute=True),
               FrameLoad("Live", value_a=4, value_b=6, rel_a=2 * k, rel_b=6 * k, absolute=True),
               FrameLoad("Live", value_a=1, value_b=3, rel_a=0.1, rel_b=0.4),
               FrameLoad("Live", direction="3", value_a=2, value_b=2)]
    L = 10 * k
    F0, M0 = resultant(f.loads, L)
    C.reverse_frames(d, [f.name])
    g = d.model.frames[f.name]
    assert [(ld.rel_a, ld.rel_b) for ld in g.loads[:2]] == [pytest.approx((8 * k, 8 * k)), pytest.approx((4 * k, 8 * k))]
    assert (g.loads[1].value_a, g.loads[1].value_b) == (6, 4)
    assert g.loads[3].value_a == -2  # local 3 turns round with the member
    F1, M1 = resultant(g.loads, L)
    assert F1 == pytest.approx(F0) and M1 == pytest.approx(F0 * L - M0)  # same moment about the old end I
    C.reverse_frames(d, [f.name])  # twice: back where it started
    h = d.model.frames[f.name]
    for a, b in zip(h.loads, f.loads):
        assert (a.rel_a, a.rel_b, a.value_a, a.value_b) == pytest.approx((b.rel_a, b.rel_b, b.value_a, b.value_b))


# ---------------------------------------------------------------- FA09
@pytest.mark.parametrize("n", [2, 3, 5])
def test_fa09_divide_keeps_every_load_where_it_acts(n):
    d = doc()
    m = d.model
    f = m.add_frame((0, 0, 3), (10, 0, 3), "B30x60")
    f.loads = [FrameLoad("Live", kind="point", value_a=5, rel_a=.2), FrameLoad("Live", kind="point", value_a=3, rel_a=.5),
               FrameLoad("Live", value_a=2, value_b=6, rel_a=.2, rel_b=.8),
               FrameLoad("Live", value_a=1, value_b=1, rel_a=1, rel_b=4, absolute=True)]
    before = global_resultant(m, [f.name])
    made = T.divide_frames(d, [f.name], n)
    assert len(made) == n
    assert global_resultant(d.model, made) == pytest.approx(before)
    assert sum(ld.kind == "point" for x in made for ld in d.model.frames[x].loads) == 2


def test_fa09_divide_keeps_offsets_and_releases_at_the_outer_ends_only():
    d = doc()
    m = d.model
    f = m.add_frame((0, 0, 3), (10, 0, 3), "B30x60", end_offset_i=.2, end_offset_j=.3, auto_end_offsets=False)
    f.releases_i = f.releases_j = (False, False, False, False, True, True)
    f.fixity_j_rot = (0.0, 50.0, 50.0)
    a, b, c = T.divide_frames(d, [f.name], 3)
    fa, fb, fc = (d.model.frames[x] for x in (a, b, c))
    assert (fa.end_offset_i, fa.end_offset_j, fb.end_offset_i, fb.end_offset_j, fc.end_offset_i, fc.end_offset_j) == \
        (.2, 0, 0, 0, 0, .3)
    assert any(fa.releases_i) and not any(fa.releases_j) and not any(fb.releases_i + fb.releases_j)
    assert any(fc.releases_j) and fc.fixity_j_rot == (0.0, 50.0, 50.0) and fa.fixity_j_rot == (0.0, 0.0, 0.0)


def test_fa09_divide_then_join_gives_the_same_loads():
    d = doc()
    m = d.model
    f = m.add_frame((0, 0, 3), (10, 0, 3), "B30x60")
    f.loads = [FrameLoad("Live", value_a=2, value_b=6, rel_a=.2, rel_b=.8), FrameLoad("Live", kind="point", value_a=5, rel_a=.3)]
    before = global_resultant(m, [f.name])
    made = T.divide_frames(d, [f.name], 2)
    C.join_frames(d, made)
    (left,) = list(d.model.frames)
    assert global_resultant(d.model, [left]) == pytest.approx(before)


# ---------------------------------------------------------------- FA10 / FA11
def test_fa10_paste_into_another_model_brings_the_definitions():
    d = doc()
    m = d.model
    f = m.add_frame((0, 0, 3), (10, 0, 3), "B30x60")
    f.loads = [FrameLoad("Live", value_a=2, value_b=2)]
    clip = C.copy_objects(m, [f.name], [], [])
    d.new()
    fr, _, pts = C.paste_objects(d, clip, 0, 0, 0)
    m2 = d.model
    assert len(fr) == 1 and "B30x60" in m2.frame_sections and m2.frame_sections["B30x60"].material in m2.materials
    assert "Live" in m2.load_patterns and not m2.validate()
    assert any("added" in t for _, t in d.notes)


def test_fa10_paste_refuses_before_changing_anything_when_a_definition_is_nowhere():
    d = doc()
    f = d.model.add_frame((0, 0, 3), (10, 0, 3), "B30x60")
    clip = C.copy_objects(d.model, [f.name], [], [])
    clip.definitions["frame_sections"].clear()
    d.new()
    before = d.model.to_json()
    with pytest.raises(ModelError, match="B30x60"):
        C.paste_objects(d, clip, 0, 0, 0)
    assert d.model.to_json() == before and not d.model.points


def test_fa10_clipboard_follows_a_change_of_units():
    d = doc()
    f = d.model.add_frame((0, 0, 3), (10, 0, 3), "B30x60", end_offset_i=0.25, auto_end_offsets=False)
    clip = C.copy_objects(d.model, [f.name], [], [])
    d.change_units("kN mm")
    fr, _, _ = C.paste_objects(d, clip, 0, 5000, 0)
    g = d.model.frames[fr[0]]
    assert length(d.model, g) == pytest.approx(10000) and g.end_offset_i == pytest.approx(250)


def test_fa11_null_lines_paste_and_replicate():
    d = doc()
    m = d.model
    f = m.add_frame((0, 0, 3), (10, 0, 3), "None", type="Null")
    fr, _, _ = C.paste_objects(d, C.copy_objects(m, [f.name], [], []), 0, 10, 0)
    assert len(fr) == 1 and d.model.frames[fr[0]].type == "Null"
    made = C.replicate_transform(d, [f.name], [], [], C.linear_transforms(0, 20, 0, 1))
    assert sum(d.model.frames[n].type == "Null" for n in made if n in d.model.frames) == 1
    assert not d.model.validate()


# ---------------------------------------------------------------- FA12 / FA13
@pytest.mark.parametrize("kind", ["point", "frame", "area"])
def test_fa12_paste_assigns_is_one_undo_step(kind):
    d = doc()
    m = d.model
    if kind == "point":
        a, b = m.add_point(0, 0, 3), m.add_point(5, 0, 3)
        a.mass, a.loads = (1, 2, 3), [PointLoad("Live", (0, 0, -1))]
    elif kind == "frame":
        a, b = m.add_frame((0, 0, 3), (5, 0, 3), "B30x60"), m.add_frame((0, 5, 3), (5, 5, 3), "B30x60")
        a.angle, a.loads = 30, [FrameLoad("Live", value_a=2, value_b=2)]
    else:
        a = m.add_area([(0, 0, 3), (5, 0, 3), (5, 5, 3)], next(iter(m.shell_sections)))
        b = m.add_area([(10, 0, 3), (15, 0, 3), (15, 5, 3)], next(iter(m.shell_sections)))
        a.angle = 30
        m.add_area_panel(a.name, [], distribution="two_way")
    before = d.model.to_json()
    C.paste_assigns(d, C.copy_assigns(d.model, kind, a.name), [b.name])
    assert len(d._undo) == 1
    d.undo()
    assert d.model.to_json() == before and not d._undo


def test_fa13_mesh_cache_follows_every_edit_past_the_undo_limit():
    d = doc()
    f = d.model.add_frame((0, 0, 3), (10, 0, 3), "B30x60")
    p = f.point_j
    for i in range(Document.MAX_UNDO + 5):
        C.align_points(d, [p], "x", 10 + i * .01)
    am = d.analysis_mesh()
    r0 = d.revision
    C.align_points(d, [p], "x", 20)
    assert d.revision > r0
    am2 = d.analysis_mesh()
    assert am2 is not am
    assert next(n.xyz[0] for n in am2.nodes if n.point == p) == pytest.approx(20)
    d.undo()
    assert next(n.xyz[0] for n in d.analysis_mesh().nodes if n.point == p) == pytest.approx(10 + (Document.MAX_UNDO + 4) * .01)


# ---------------------------------------------------------------- FA14 / FA15
@pytest.mark.parametrize("vec", [(0, 0, 3), (0, 0, 0), (1e-14, 0, 3)])
def test_fa14_extrude_is_all_or_nothing(vec):
    d = doc()
    m = d.model
    f = m.add_frame((0, 0, 3), (10, 0, 3), "B30x60")
    g = m.add_frame((20, 0, 3), (20, 0, 6), "C40x40")
    before = m.to_json()
    with pytest.raises(ModelError):
        C.extrude_frames_to_shells(d, [f.name, g.name], *vec, 1, "W20")
    assert d.model.to_json() == before and not d._undo
    assert C.extrude_frames_to_shells(d, [f.name], 0, 0, 3, 1, "W20")


def test_fa14_validate_reports_shells_and_frames_without_size():
    d = doc()
    m = d.model
    m.add_area([(0, 0, 3), (5, 0, 3), (10, 0, 3)], next(iter(m.shell_sections)))
    p = m.add_point(1, 1, 3)
    m.frames["Z"] = Frame(name="Z", point_i=p.name, point_j=p.name, section="B30x60")
    codes = {e.code for e in m.validate()}
    assert {"DEGENERATE_AREA", "ZERO_LENGTH"} <= codes


@pytest.mark.parametrize("delete", [True, False])
def test_fa15_replicate_to_story_delete_original(delete):
    d = Document(new_building(story_heights=(3, 3), grid_only=True))
    f = d.model.add_frame((0, 0, 3), (6, 0, 3), "B30x60")
    seed(d.model.points[f.point_i])
    T.replicate_to_stories(d, [f.name], [], [], ["Story2"], delete_original=delete)
    assert (f.name in d.model.frames) is not delete
    assert len(d.model.frames) == (1 if delete else 2)
    q = d.model.find_point(0, 0, 6)
    assert q.mass == (7.0, 8.0, 9.0)
    assert len(d._undo) == 1
    d.undo()
    assert f.name in d.model.frames and len(d.model.frames) == 1


# ------------------------------------------------- state: FA01 / FA02 / FA03
@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _state():
    from structstudio.app.state import ViewState
    d = Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3, 3)))
    return d, ViewState(d)


@pytest.mark.parametrize("units,k", [("kN mm", 1000.0), ("kip ft", 1 / 0.3048), ("kN cm", 100.0)])
def test_fa01_every_tool_keeps_its_offsets_physical_through_unit_changes(qapp, units, k):
    d, st = _state()
    for tool, vals in (("point", (1.0, .5, .25)), ("beam", (.3, 0, 0)), ("wall", (0, .2, 0)), ("slab", (0, 0, .1))):
        st.set_tool(tool)
        st.settings.offset_x, st.settings.offset_y, st.settings.offset_z = vals
    st.set_tool("beam")
    st.settings.drawing_control, st.settings.fixed_length = "Fixed length", 2.5
    st.set_tool("select")
    d.change_units(units)
    for tool, vals in (("point", (1.0, .5, .25)), ("beam", (.3, 0, 0)), ("wall", (0, .2, 0)), ("slab", (0, 0, .1))):
        st.set_tool(tool)
        got = (st.settings.offset_x, st.settings.offset_y, st.settings.offset_z)
        assert got == pytest.approx(tuple(v * k for v in vals)), tool
        if tool == "beam":
            assert st.settings.fixed_length == pytest.approx(2.5 * k)
    d.undo()  # back to metres: the values follow again
    st.set_tool("point")
    assert (st.settings.offset_x, st.settings.offset_y) == pytest.approx((1.0, .5))


@pytest.mark.parametrize("tool", ["beam", "slab", "move"])
def test_fa02_points_of_a_command_in_progress_follow_a_change_of_units(qapp, tool):
    d, st = _state()
    st.set_tool(tool)
    st.tool.set_story("Story1")
    st.tool.points = [(2.0, 2.0, 3.0), (4.0, 2.0, 3.0)][: 1 if tool != "slab" else 2]
    d.change_units("kN mm")
    assert st.tool.points[0] == pytest.approx((2000.0, 2000.0, 3000.0))
    d.undo()
    assert st.tool.points[0] == pytest.approx((2.0, 2.0, 3.0))


@pytest.mark.parametrize("how", ["new", "open", "import"])
def test_fa03_another_document_cancels_the_command_in_progress(qapp, tmp_path, how):
    d, st = _state()
    st.set_tool("beam")
    st.tool.set_story("Story1")
    st.tool.click((99.0, 98.0, 3.0))
    st.set_selection(list(d.model.frames)[:1])
    g = d.generation
    path = str(tmp_path / "next.ssj.json")
    new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3,)).to_json(path)
    if how == "new":
        d.new()
    elif how == "open":
        d.open(path)
    else:
        d.set_imported(new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3,)))
    assert d.generation == g + 1 and st.tool.points == [] and st.count == 0
    st.tool.set_story("Story1")
    assert st.tool.click((6.0, 5.0, 3.0)) is False  # starts a new line, nothing from the old file
    assert not d.model.frames or all(99.0 not in d.model.points[f.point_i].xyz for f in d.model.frames.values())


def test_fa03_an_ordinary_edit_keeps_the_command_in_progress(qapp):
    d, st = _state()
    st.set_tool("beam")
    st.tool.set_story("Story1")
    st.tool.click((0.0, 0.0, 3.0))
    d.edit(lambda m: m.add_point(50, 50, 3))
    assert st.tool.points == [(0.0, 0.0, 3.0)]


# ------------------------------------------------------- drawing review
def test_duplicate_members_made_by_a_command_are_reported():
    d = doc()
    m = d.model
    f = m.add_frame((0, 0, 3), (10, 0, 3), "B30x60")
    d.edit(lambda mm: mm.add_frame((0, 0, 3), (10, 0, 3), "B30x60"))
    assert [i.code for i in d.duplicates] == ["DUPLICATE_FRAME"] and f.name in d.duplicates[0].frames
    d.undo()
    d.edit(lambda mm: mm.add_frame((5, 0, 3), (15, 0, 3), "B30x60"))
    assert [i.code for i in d.duplicates] == ["OVERLAPPING_FRAMES"]
    d.edit(lambda mm: mm.add_frame((0, 2, 3), (10, 2, 3), "B30x60"))
    assert d.duplicates == []


def test_copy_and_move_onto_a_member_are_reported():
    d = doc()
    m = d.model
    c1 = m.add_frame((0, 0, 0), (0, 0, 3), "C40x40", type="Column")
    m.add_frame((6, 0, 0), (6, 0, 3), "C40x40", type="Column")
    M.copy_transformed(d, [c1.name], [], [], M.translation((6, 0, 0)))
    assert [i.code for i in d.duplicates] == ["DUPLICATE_FRAME"]
    d.undo()
    M.transform_in_place(d, [c1.name], [], [], M.translation((6, 0, 0)))
    assert [i.code for i in d.duplicates] == ["DUPLICATE_FRAME"]
    d.undo()
    M.copy_transformed(d, [c1.name], [], [], M.mirror((3, -1, 0), (3, 1, 0), (0, 0, 1)))
    assert [i.code for i in d.duplicates] == ["DUPLICATE_FRAME"]


def test_check_model_lists_geometry_problems():
    from structstudio.model.checks import geometry_issues
    d = doc()
    m = d.model
    m.add_frame((0, 0, 3), (10, 0, 3), "B30x60")
    m.add_frame((0, 0, 3), (10, 0, 3), "B30x60")
    m.add_frame((10, 0, 3), (20, 0, 3), "B30x60")
    m.add_frame((15, 0, 3), (25, 0, 3), "B30x60")
    m.add_point(40, 0, 3)
    m.add_point(40.0, 0.0, 3.0 + 1e-6, merge=False)
    codes = sorted(i.code for i in geometry_issues(m))
    assert codes.count("DUPLICATE_FRAME") == 1 and codes.count("OVERLAPPING_FRAMES") == 1
    assert "COINCIDENT_JOINTS" in codes and "UNUSED_JOINTS" in codes


@pytest.mark.parametrize("z", [5.97, 6.04, 5.2])
def test_columns_and_walls_in_an_elevation_snap_to_the_nearest_story(qapp, z):
    from structstudio.app.tools import Plane
    d = Document(new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3, 3), grid_only=True))
    for name, pts in (("column", [(0.0, z)]), ("wall", [(0.0, z), (6.0, z)])):
        cls = T.TOOLS[name]
        t = cls(d, T.DrawSettings())
        t.set_plane(Plane("xz", 0.0, None))
        for u, v in pts:
            t.click((u, v))
        made = [d.model.frames.get(n) or d.model.areas.get(n) for n in t.created]
        assert made, name
        tops = {max(d.model.points[p].z for p in ((o.point_i, o.point_j) if hasattr(o, "point_i") else o.points))
                for o in made}
        assert tops == {6.0}, name


def test_3d_typing_messages_say_what_to_do():
    from structstudio.app.modify import parse_absolute, parse_typed
    from structstudio.app.tools import Plane
    with pytest.raises(ValueError, match="point the cursor"):
        parse_typed("2", (0, 0, 0), None, Plane("3d", 0.0, None))
    with pytest.raises(ValueError, match="x,y,z"):
        parse_absolute("2,3", Plane("3d", 0.0, None))


def test_a_failed_edit_leaves_no_trace():
    d = doc()
    d.model.add_point(1, 1, 3)
    d.edit(lambda m: m.add_point(2, 2, 3))
    d.undo()
    before = d.model.to_json()

    def bad(m):
        m.add_point(5, 5, 3)
        raise ModelError("X", "half way")
    with pytest.raises(ModelError):
        d.edit(bad)
    assert d.model.to_json() == before and not d._undo and d.can_redo


def test_undo_snapshots_restore_the_same_model():
    d = Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3, 3), grid_only=False))
    before = d.model.to_json()
    d.edit(lambda m: m.add_point(9, 9, 9))
    assert isinstance(d._undo[-1], bytes)
    d.undo()
    assert d.model.to_json() == before
    d.redo()
    assert d.model.find_point(9, 9, 9) is not None


# ------------------------------------------------------------- window (child)
@pytest.mark.skipif(CHILD, reason="parent only")
def test_window_cases_in_subprocess():
    import shutil
    env = dict(os.environ, STRUCTSTUDIO_GUI_CHILD="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "gui_", __file__]
    if not env.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1600x1000x24"] + cmd
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower(), \
        (proc.stdout + proc.stderr)[-4000:]


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    pytest.importorskip("vtk")
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    from structstudio.app.mainwindow import MainWindow
    w = MainWindow(Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3, 3))),
                   QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.show()
    qapp.processEvents()
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    qapp.processEvents()


def _answer(monkeypatch, button):
    asked = []
    B = QtWidgets.QMessageBox
    monkeypatch.setattr(B, "question", staticmethod(lambda *a, **k: (asked.append(a[2]), getattr(B, button))[1]))
    return asked


@gui
@pytest.mark.parametrize("command", ["open", "new", "import", "close"])
def test_gui_fa04_cancel_keeps_the_unsaved_model(window, monkeypatch, tmp_path, command):
    path = str(tmp_path / "other.ssj.json")
    new_building(name="OTHER").to_json(path)
    window.doc.edit(lambda m: m.add_point(99, 99, 3))
    before = window.doc.model.to_json()
    asked = _answer(monkeypatch, "Cancel")
    opened = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (opened.append(1), (path, ""))[1]))
    {"open": window.open_model, "new": window.new_model, "import": window.import_csi,
     "close": window.close_model}[command]()
    assert asked and window.doc.model.to_json() == before and window.doc.dirty and not opened


@gui
def test_gui_fa04_discard_and_save(window, monkeypatch, tmp_path):
    path = str(tmp_path / "other.ssj.json")
    new_building(name="OTHER").to_json(path)
    window.doc.edit(lambda m: m.add_point(99, 99, 3))
    _answer(monkeypatch, "Discard")
    window._open(path)
    assert window.doc.model.info.name == "OTHER"
    # Save: a file name is asked; cancelling it keeps the model
    window.doc.edit(lambda m: m.add_point(98, 98, 3))
    before = window.doc.model.to_json()
    _answer(monkeypatch, "Save")
    window.doc.path = None
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    window._open(path)
    assert window.doc.model.to_json() == before
    # Save to a file that cannot be written: the model stays, with a message
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a: warned.append(a[2])))
    bad = str(tmp_path / "no" / "such" / "dir" / "m.ssj.json")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (bad, "")))
    window._open(path)
    assert window.doc.model.to_json() == before and warned
    # Save works: then the file opens
    good = str(tmp_path / "saved.ssj.json")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (good, "")))
    window._open(path)
    assert os.path.exists(good) and window.doc.model.info.name == "OTHER"


@gui
def test_gui_fa04_exit_asks_and_cancel_keeps_the_window(window, monkeypatch, qapp):
    window.doc.edit(lambda m: m.add_point(99, 99, 3))
    asked = _answer(monkeypatch, "Cancel")
    ev = QtCore.QEvent(QtCore.QEvent.Close)
    from PySide6 import QtGui
    ev = QtGui.QCloseEvent()
    window.closeEvent(ev)
    assert asked and not ev.isAccepted()


@gui
def test_gui_duplicate_member_warning_offers_undo(window, qapp):
    m = window.doc.model
    f = next(iter(m.frames.values())) if m.frames else None
    if f is None:
        window.doc.edit(lambda mm: mm.add_frame((0, 0, 3), (6, 0, 3), "B30x60"))
        f = next(iter(window.doc.model.frames.values()))
    a, b = window.doc.model.points[f.point_i].xyz, window.doc.model.points[f.point_j].xyz
    window.doc.edit(lambda mm: mm.add_frame(a, b, f.section))
    n = len(window.doc.model.frames)
    qapp.processEvents()
    box = getattr(window, "_duplicate_box", None)
    assert box is not None and box.isVisible() and "DUPLICATE" not in box.text()
    undo = next(btn for btn in box.buttons() if btn.text() == "Undo")
    undo.click()
    qapp.processEvents()
    assert len(window.doc.model.frames) == n - 1


@gui
def test_gui_check_model_button_and_pre_analysis_question(window, monkeypatch, qapp):
    titles = [a.toolTip() for tb in window.findChildren(QtWidgets.QToolBar) for a in tb.actions()]
    assert any(t.startswith("Check Model") for t in titles)
    window.doc.dirty = False
    window.doc.check_duplicates = False
    window.doc.edit(lambda mm: [mm.add_frame((0, 0, 3), (6, 0, 3), "B30x60") for _ in range(2)])
    shown = []
    monkeypatch.setattr(type(window), "show_analysis_issues", lambda self, title="": shown.append(title))
    asked = _answer(monkeypatch, "No")
    window.run_analysis()
    assert asked and "warning" in shown[-1] and window._worker is None
    issues = window.validate_model()
    assert any("DUPLICATE_FRAME" in i.message for i in issues)


# ------------------------------------------- merge choices (user's request, 2026-11-10)
def _conflicting_pair():
    d = doc()
    m = d.model
    m.add_diaphragm("D2")
    a = m.add_point(0, 0, 3)
    b = m.add_point(0.0001, 0, 3, merge=False)
    a.springs, b.springs = (1.0, 1.0, 1.0), (5.0, 5.0, 5.0)
    a.diaphragm, b.diaphragm = "D1", "D2"
    a.mass, b.mass = (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)  # identical: not asked
    return d, m, a, b


@pytest.fixture
def resolver():
    from structstudio.model import point_ops
    old = point_ops.CONFLICT_RESOLVER
    yield point_ops
    point_ops.CONFLICT_RESOLVER = old


def test_merge_asks_which_value_to_keep_and_uses_the_answer(resolver):
    d, m, a, b = _conflicting_pair()
    asked = []

    def answer(items):
        asked.extend(items)
        return {(i.keep, i.gone, i.group): (i.gone if i.group == "springs" else i.keep) for i in items}
    resolver.CONFLICT_RESOLVER = answer
    C.merge_joints(d, [a.name, b.name], 0.001)
    assert sorted(i.group for i in asked) == ["diaphragm", "springs"]
    q = d.model.points[a.name]
    assert q.springs == (5.0, 5.0, 5.0) and q.diaphragm == "D1" and q.mass == (1.0, 1.0, 1.0)
    assert b.name not in d.model.points and len(d._undo) == 1


def test_merge_cancelled_or_unanswered_changes_nothing(resolver):
    d, m, a, b = _conflicting_pair()
    before = m.to_json()
    resolver.CONFLICT_RESOLVER = lambda items: None
    with pytest.raises(ModelError, match="cancelled"):
        C.merge_joints(d, [a.name, b.name], 0.001)
    resolver.CONFLICT_RESOLVER = None
    with pytest.raises(resolver.MergeConflict) as e:
        C.merge_joints(d, [a.name, b.name], 0.001)
    assert {i.group for i in e.value.items} == {"springs", "diaphragm"}
    assert d.model.to_json() == before and not d._undo


def test_merge_choice_for_different_local_axes_takes_the_whole_set(resolver):
    d, m, a, b = _two_joints()
    a.local_axes, a.restraints, a.springs = (30.0, 0.0, 0.0), (True, False, False, False, False, False), (0, 9, 0)
    b.restraints = (False, True, False, False, False, False)
    resolver.CONFLICT_RESOLVER = lambda items: {(i.keep, i.gone, i.group): i.gone for i in items}
    C.merge_joints(d, [a.name, b.name], 0.001)
    q = d.model.points[a.name]
    assert q.local_axes is None and q.restraints == (False, True, False, False, False, False) and q.springs == (0, 0, 0)


def test_grip_drag_onto_a_joint_with_different_values_asks_too(resolver):
    d = doc()
    m = d.model
    a, b = m.add_point(0, 0, 3), m.add_point(2, 0, 3)
    a.mass, b.mass = (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)
    resolver.CONFLICT_RESOLVER = lambda items: {(i.keep, i.gone, i.group): i.gone for i in items}
    M.move_joint(d, b.name, a.xyz)
    assert d.model.points[a.name].mass == (2.0, 2.0, 2.0) and len(d.model.points) == 1


# -------------------------------- copies turn with Replicate Radial / Mirror (2026-11-10)
def test_radial_copy_turns_joint_loads_supports_springs_and_mass():
    d = doc()
    m = d.model
    p = m.add_point(5, 0, 3)
    p.restraints, p.springs = (True, False, False, False, False, False), (100.0, 0.0, 0.0)
    p.loads = [PointLoad("Live", (10.0, 0.0, -5.0), (0.0, 2.0, 0.0))]
    p.mass = (4.0, 1.0, 1.0)
    C.replicate_transform(d, [], [], [p.name], C.radial_transforms(0, 0, 90, 1))
    q = d.model.find_point(0, 5, 3)
    assert q.loads[0].force == pytest.approx((0.0, 10.0, -5.0)) and q.loads[0].moment == pytest.approx((-2.0, 0.0, 0.0))
    assert q.local_axes == pytest.approx((90.0, 0.0, 0.0))  # the X support / spring now acts along Y
    assert q.restraints == p.restraints and q.springs == p.springs and q.mass == pytest.approx((1.0, 4.0, 1.0))


def test_mirror_copy_mirrors_joint_loads_and_moments():
    d = doc()
    m = d.model
    p = m.add_point(15, 2, 3)
    p.loads = [PointLoad("Live", (10.0, 3.0, -5.0), (1.0, 2.0, 4.0))]
    M.copy_transformed(d, [], [], [p.name], M.mirror((20, 0, 0), (20, 1, 0), (0, 0, 1)))
    q = d.model.find_point(25, 2, 3)
    assert q.loads[0].force == pytest.approx((-10.0, 3.0, -5.0))
    assert q.loads[0].moment == pytest.approx((1.0, -2.0, -4.0))  # a moment is an axial vector


@pytest.mark.parametrize("angle", [90.0, 45.0, 30.0])
def test_radial_copy_turns_global_frame_loads_and_keeps_their_resultant(angle):
    d = doc()
    m = d.model
    f = m.add_frame((5, 0, 3), (8, 0, 3), "B30x60")
    f.loads = [FrameLoad("Live", direction="X", value_a=3, value_b=3), FrameLoad("Live", value_a=2, value_b=2)]
    made = C.replicate_transform(d, [f.name], [], [], C.radial_transforms(0, 0, angle, 1))
    g = d.model.frames[made[0]]
    fx = sum(ld.value_a for ld in g.loads if ld.direction == "X")
    fy = sum(ld.value_a for ld in g.loads if ld.direction == "Y")
    a = math.radians(angle)
    assert (fx, fy) == pytest.approx((3 * math.cos(a), 3 * math.sin(a)), abs=1e-9)
    assert [ld.value_a for ld in g.loads if ld.direction == "Gravity"] == [2]


def test_radial_copy_turns_column_axes_and_rotate_in_place_turns_loads():
    d = doc()
    m = d.model
    c = m.add_frame((5, 0, 0), (5, 0, 3), "C40x40", type="Column", angle=10)
    made = C.replicate_transform(d, [c.name], [], [], C.radial_transforms(0, 0, 90, 1))
    assert d.model.frames[made[0]].angle == pytest.approx(100.0)
    p = d.model.add_point(20, 0, 3)
    p.loads = [PointLoad("Live", (10.0, 0.0, 0.0))]
    M.transform_in_place(d, [], [], [p.name], M.rotation((0, 0, 0), (0, 0, 1), 90))
    q = d.model.points[p.name]
    assert (q.x, q.y) == pytest.approx((0.0, 20.0)) and q.loads[0].force == pytest.approx((0.0, 10.0, 0.0))


def test_a_copy_turned_out_of_plan_keeps_loads_as_assigned():
    d = doc()
    p = d.model.add_point(5, 0, 3)
    p.loads = [PointLoad("Live", (0.0, 0.0, -10.0))]
    M.copy_transformed(d, [], [], [p.name], M.rotation((0, 0, 0), (0, 1, 0), 90))
    q = d.model.find_point(3, 0, -5)
    assert q.loads[0].force == (0.0, 0.0, -10.0)  # gravity stays gravity


@gui
def test_gui_merge_choice_dialog(qapp):
    from structstudio.app.merge_dialog import MergeChoiceDialog
    from structstudio.model.point_ops import merge_conflicts
    d, m, a, b = _conflicting_pair()
    items = merge_conflicts(a, b)
    dlg = MergeChoiceDialog(items)
    assert dlg.table.rowCount() == 2
    assert dlg.choices() == {(a.name, b.name, i.group): a.name for i in items}
    dlg.choose(0, True)
    got = dlg.choices()
    assert got[(a.name, b.name, items[0].group)] == b.name
    dlg.close()
    dlg.deleteLater()  # a parentless Qt dialog must go before the application does
    qapp.processEvents()


@gui
def test_gui_window_asks_merge_choices_in_a_dialog(window, monkeypatch, qapp):
    from structstudio.app import merge_dialog
    from structstudio.model import point_ops
    shown = []

    def fake_exec(self):
        shown.append(self.table.rowCount())
        self.choose(0, True)
        return 1
    monkeypatch.setattr(merge_dialog.MergeChoiceDialog, "exec", fake_exec)
    m = window.doc.model
    a, b = m.add_point(30, 0, 3), m.add_point(30.0001, 0, 3, merge=False)
    a.mass, b.mass = (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)
    C.merge_joints(window.doc, [a.name, b.name], 0.001)
    assert shown == [1] and window.doc.model.points[a.name].mass == (2.0, 2.0, 2.0)
    assert point_ops.CONFLICT_RESOLVER is not None
