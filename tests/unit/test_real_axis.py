"""Move Beams to Real Axis: when a rigid link, when just the move.

The rule (commands.move_beams_to_axis): a rigid link only inside the rigid zone of the
joint the beam leaves - its axis inside the column, or the beam's body still bearing on
the column, or (up / down) within the depth of what it is cast into.  Beyond it the beam
is as drawn: it frames into what it lands on, is trimmed / extended to the member it
meets, or is reported as not supported.  Every case is also put back (the Insertion Point
form with the move un-ticked) and must give the model it started from.

One bay 6 x 5 x 3, columns C40x40 (axis zone 0.20), beams B30x60 (0.30 wide: they bear on
the column up to 0.20 + 0.15 = 0.35).
"""
import copy
import json

import numpy as np
import pytest

from structstudio import analyze
from structstudio.app import commands as C
from structstudio.app.document import Document
from structstudio.app.templates import new_building
from structstudio.diagnostics import AnalysisError
from structstudio.engine.kernel import SciPyKernel
from structstudio.model.checks import duplicates_of

W = 12.0  # load on the moved beam, per length


def _frame(hs=(3.0,), ys=(5,), xs=(6,)):
    m = new_building(x_spacings=xs, y_spacings=ys, story_heights=hs, grid_only=False)
    for p in m.load_patterns.values():
        p.self_weight_multiplier = 0.0
    for a in [a for a in m.areas if m.areas[a].type == "Floor"]:
        del m.areas[a]
    return m


def _beams(m, z=None, **k):
    P = m.points
    return sorted(f.name for f in m.frames.values() if f.type == "Beam"
                  and all(abs(getattr(P[q], a) - v) < 1e-9 for a, v in k.items() for q in (f.point_i, f.point_j))
                  and (z is None or abs(P[f.point_i].z - z) < 1e-9))


def _case(m):
    pat = next(iter(m.load_patterns))
    return pat, [c for c, v in m.load_cases.items() if getattr(v, "loads", None) and pat in v.loads][0]


def _dump(m):
    return json.dumps(m.to_dict(), sort_keys=True)


def _length(m, b):
    f = m.frames[b]
    return float(np.linalg.norm(np.array(m.points[f.point_j].xyz) - np.array(m.points[f.point_i].xyz)))


def _run_case(build, offsets, links, has, supported, check=None):
    """Assigns ``offsets`` {beam: (dx, dy, dz) global} with the move, checks the number of
    rigid links, the notes (``has``: {text: count}), that the analysis runs with every
    load reaching the supports - or stops with "not connected to any support" when an end
    is on nothing (``supported`` False) - and that un-ticking the move gives the model back."""
    m, beams = build()
    pat, case = _case(m)
    for b in beams:
        m.add_frame_distributed_load(b, pat, W, W)
    before = _dump(m)
    doc = Document(m)
    moved, made, notes = [], [], []
    for b, off in offsets(beams).items():
        off = tuple(float(x) for x in off)
        mv, lk, nt, _ = C.set_insertion_point(
            doc, [b], {"joint_offset_i": off, "joint_offset_j": off, "joint_offset_local": False}, move=True) \
            if len(offsets(beams)) == 1 else ([], [], [], [])
        moved += mv
        made += lk
        notes += nt
    if len(offsets(beams)) > 1:  # several beams: one command (they are decided together)
        def mutate(mm):
            for b, off in offsets(beams).items():
                mm.frames[b].joint_offset_i = mm.frames[b].joint_offset_j = tuple(float(x) for x in off)
        doc.edit(mutate)
        moved, made, notes = C.move_beams_to_axis(doc, list(offsets(beams)))
    mm = doc.model
    assert sorted(moved) == sorted(offsets(beams)), notes
    assert len(made) == links, (made, notes)
    for text, count in has.items():
        assert sum(text in t for t in notes) == count, (text, notes)
    assert not duplicates_of(mm, list(mm.frames))
    assert all(C.moved_beam_state(mm, b) is not None for b in moved)
    if supported:
        r = analyze(mm, kernel=SciPyKernel())
        total = W * sum(_length(mm, b) for b in beams)
        assert r.base_reactions(case)[2] == pytest.approx(total, rel=1e-9)
        if check:
            check(mm, r, case, beams)
    else:
        assert sum("NOT SUPPORTED" in t for t in notes) >= 1
        with pytest.raises(AnalysisError, match="not connected to any support"):
            analyze(mm, kernel=SciPyKernel())
    # back: the move un-ticked, the offsets set to zero - the model as it was
    zero = {"joint_offset_i": (0.0, 0.0, 0.0), "joint_offset_j": (0.0, 0.0, 0.0), "joint_offset_local": False}
    _, _, _, restored = C.set_insertion_point(doc, list(moved), zero, move=False)
    assert sorted(restored) == sorted(moved)
    assert _dump(doc.model) == before
    return mm, notes


def _south():
    m = _frame()
    return m, _beams(m, y=0.0)


one = lambda off: (lambda beams: {beams[0]: off})

CASES = {
    # ---- in the floor plane, column and side beam at each end ----
    "0.10 in: axis inside the column - rigid links": (_south, one((0, 0.10, 0)), 2, {}, True),
    "0.30 in: past the column face, on the side beam - frames into it": (_south, one((0, 0.30, 0)), 0, {"frames into": 2}, True),
    "0.50 in: on the side beam - frames into it": (_south, one((0, 0.50, 0)), 0, {"frames into": 2}, True),
    # ---- outward: nothing there but the column ----
    "0.10 out: axis inside the column - rigid links": (_south, one((0, -0.10, 0)), 2, {}, True),
    "0.30 out: the beam still bears on the column - rigid links": (_south, one((0, -0.30, 0)), 2, {"still bears on it": 2}, True),
    "0.50 out: clear of the column, on nothing - not supported": (_south, one((0, -0.50, 0)), 0, {"NOT SUPPORTED": 2}, False),
    # ---- up / down ----
    "0.20 up at the roof (upturned, joint within its depth) - rigid links": (_south, one((0, 0, 0.20)), 2, {}, True),
    "1.00 up at the roof: above the column tops - not supported": (_south, one((0, 0, 1.00)), 0, {"NOT SUPPORTED": 2}, False),
    "1.00 down: frames into the columns partway": (_south, one((0, 0, -1.00)), 0, {"frames into column": 2}, True),
    "0.10 across (in the column), 1.00 down: link from the column axis there": (_south, one((0, 0.10, -1.00)), 2, {}, True),
    "0.30 across, 0.20 down: within the side beam's depth - link down from it": (_south, one((0, 0.30, -0.20)), 2, {"by a rigid link 0.200 down": 2}, True),
    "0.50 across, 1.00 down: below the side beam, on nothing - not supported": (_south, one((0, 0.50, -1.00)), 0, {"NOT SUPPORTED": 2}, False),
}


@pytest.mark.parametrize("name", list(CASES))
def test_rigid_link_or_move(name):
    build, offsets, links, has, supported = CASES[name]
    _run_case(build, offsets, links, has, supported)


def test_out_of_the_frame_onto_cantilevers_drawn_first():
    """0.50 outward is not supported - unless what carries the beam is drawn: the side beams
    carried on as cantilevers.  The moved beam then frames into them, no rigid link."""
    def build():
        m = _frame()
        b = _beams(m, y=0.0)
        sec = m.frames[b[0]].section
        for x in (0.0, 6.0):
            j = next(p.name for p in m.points.values() if p.xyz == (x, 0.0, 3.0))
            m.add_frame(j, m.add_point(x, -0.5, 3.0).name, sec)
        return m, b
    _run_case(build, one((0, -0.50, 0)), 0, {"frames into": 2}, True)


def test_two_corner_beams_meet_where_their_axes_cross():
    """Both edge beams at a corner moved 0.50 inward (clear of the column): each would end
    0.50 short of nothing - they are trimmed to the point where their axes cross and end
    on one joint there; their far ends frame into the side beams.  No rigid link."""
    def build():
        m = _frame()
        return m, [_beams(m, y=0.0)[0], _beams(m, x=0.0)[0]]

    def check(mm, r, case, beams):
        a, b = (mm.frames[n] for n in beams)
        shared = {a.point_i, a.point_j} & {b.point_i, b.point_j}
        assert len(shared) == 1 and mm.points[next(iter(shared))].xyz == pytest.approx((0.5, 0.5, 3.0))
    _run_case(build, lambda beams: {beams[0]: (0, 0.5, 0), beams[1]: (0.5, 0, 0)}, 0,
              {"trimmed": 2, "frames into": 2}, True, check)


def test_two_corner_beams_still_on_the_column():
    """Both moved 0.25 inward: past the column's face, both still bear on it - a rigid link
    each to the column's joint; they cross (and are joined) just inside."""
    def build():
        m = _frame()
        return m, [_beams(m, y=0.0)[0], _beams(m, x=0.0)[0]]
    _run_case(build, lambda beams: {beams[0]: (0, 0.25, 0), beams[1]: (0.25, 0, 0)}, 2,
              {"still bears on it": 2, "frames into": 2}, True)


def test_beam_between_girders_slides_along_them():
    """A secondary beam framing into two girders (no columns at its ends) moved sideways: it
    frames into the girders at its new place - never a rigid link along a girder."""
    def build():
        m = _frame()
        sec = m.frames[_beams(m, y=0.0)[0]].section
        s = m.add_frame(m.add_point(3.0, 0.0, 3.0).name, m.add_point(3.0, 5.0, 3.0).name, sec).name
        return m, [s]
    _run_case(build, one((0.5, 0, 0)), 0, {"frames into": 2}, True)


def test_end_on_a_skew_girder_is_trimmed_to_it():
    """A beam ending on a skew girder (no column there): moved sideways, its end is where its
    new axis meets the girder - trimmed 0.6 here - and it frames into it."""
    def build():
        m = _frame()
        sec = m.frames[_beams(m, y=0.0)[0]].section
        j = lambda x, y: next((p.name for p in m.points.values() if p.xyz == (x, y, 3.0)), None) or m.add_point(x, y, 3.0).name
        m.add_frame(j(0.0, 0.0), j(6.0, 5.0), sec)  # the skew girder (through (3, 2.5))
        s = m.add_frame(j(3.0, 2.5), j(6.0, 2.5), sec).name
        return m, [s]

    def check(mm, r, case, beams):
        f = mm.frames[beams[0]]
        assert mm.points[f.point_i].xyz == pytest.approx((3.6, 3.0, 3.0)) and _length(mm, beams[0]) == pytest.approx(2.4)
    _run_case(build, one((0, 0.5, 0)), 0, {"trimmed 0.600": 1, "frames into": 1}, True, check)


def test_continuous_beam_over_a_column():
    """One beam object over a middle column: 0.10 - a rigid link at each of the three
    columns; 0.50 - none (it frames into the three cross beams)."""
    def build():
        m = _frame(xs=(6, 6))
        a, b = sorted(_beams(m, y=0.0), key=lambda n: m.points[m.frames[n].point_i].x)
        m.frames[a].point_j = m.frames[b].point_j
        del m.frames[b]
        return m, [a]
    _run_case(build, one((0, 0.10, 0)), 3, {}, True)
    _run_case(build, one((0, 0.50, 0)), 0, {"frames into": 2}, True)


def test_the_form_remembers_the_move_and_changes_it():
    """Insertion Point on a moved beam: the values it was moved with; the same values again
    change nothing; other values move it afresh from the joints it was drawn on (not from
    where it is now); one undo each."""
    m, beams = _south()
    b = beams[0]
    before = _dump(m)
    doc = Document(m)
    f1 = {"joint_offset_i": (0.0, 0.1, 0.0), "joint_offset_j": (0.0, 0.1, 0.0), "joint_offset_local": False}
    C.set_insertion_point(doc, [b], f1, move=True)
    st = C.moved_beam_state(doc.model, b)
    assert st["offset_i"] == pytest.approx((0.0, 0.1, 0.0)) and st["local"] is False
    assert C.offset_zone_issues(doc.model, [b], f1, in_analysis_only=False) == []
    snap = _dump(doc.model)
    assert C.set_insertion_point(doc, [b], f1, move=True) == ([], [], [], []) and _dump(doc.model) == snap
    f2 = {"joint_offset_i": (0.0, 0.5, 0.0), "joint_offset_j": (0.0, 0.5, 0.0), "joint_offset_local": False}
    assert len(C.offset_zone_issues(doc.model, [b], f2, in_analysis_only=False)) == 2  # judged from the old joints
    moved, links, notes, restored = C.set_insertion_point(doc, [b], f2, move=True)
    mm = doc.model
    assert moved == [b] and restored == [b] and not links
    assert mm.points[mm.frames[b].point_i].y == pytest.approx(0.5)  # 0.5 from the joint, not 0.1 + 0.5
    assert not any(g.type == "Rigid" for g in mm.frames.values())  # the links of the first move are gone
    doc.undo()
    assert _dump(doc.model) == snap
    doc.undo()
    assert _dump(doc.model) == before


def test_move_unticked_keeps_the_values_as_offsets():
    """Un-ticking the move with the values left in the form: the beam is back on its joints
    and the values are its joint offsets (an offset in the rigid zone - no rigid link
    object); a moved beam survives save / load and a change of units."""
    from structstudio.model import Model
    from structstudio.units import UnitSystem
    m, beams = _south()
    b = beams[0]
    doc = Document(m)
    f1 = {"joint_offset_i": (0.0, 0.1, 0.0), "joint_offset_j": (0.0, 0.1, 0.0), "joint_offset_local": False}
    C.set_insertion_point(doc, [b], f1, move=True)
    m2 = Model.from_dict(json.loads(json.dumps(doc.model.to_dict())))
    assert C.moved_beam_state(m2, b) == C.moved_beam_state(doc.model, b)
    mm = m2.convert_units(UnitSystem("kN", "mm"))
    assert C.moved_beam_state(mm, b)["offset_i"] == pytest.approx((0.0, 100.0, 0.0))
    C.set_insertion_point(doc, [b], f1, move=False)
    f = doc.model.frames[b]
    assert doc.model.points[f.point_i].y == 0.0 and f.joint_offset_i == pytest.approx((0.0, 0.1, 0.0))
    assert f.real_axis is None and not any(g.type == "Rigid" for g in doc.model.frames.values())


def test_a_moved_beam_changed_by_hand_is_no_longer_remembered():
    """The beam re-connected by another command since: the record no longer describes it -
    the form shows it as it is and nothing is 'put back'."""
    m, beams = _south()
    b = beams[0]
    doc = Document(m)
    C.set_insertion_point(doc, [b], {"joint_offset_i": (0.0, 0.5, 0.0), "joint_offset_j": (0.0, 0.5, 0.0)}, move=True)

    def by_hand(mm):
        mm.frames[b].point_i = mm.add_point(0.0, 1.0, 3.0).name
    doc.edit(by_hand)
    assert C.moved_beam_state(doc.model, b) is None
    _, _, notes, restored = C.set_insertion_point(doc, [b], {"joint_offset_i": (0.0, 0.0, 0.0)}, move=False)
    assert not restored and any("changed since it was moved" in t for t in notes)
    assert doc.model.points[doc.model.frames[b].point_i].y == 1.0


def test_offsets_in_the_rigid_zone_equal_the_moved_beam():
    """Inside the rigid zone the offset way (offsets in the analysis) and the moved beam are
    the same structure: the same results - 0.10 (axis in the column), 0.30 outward (bearing
    on the column) and 0.30 inward (on the side beam)."""
    for dy in (0.10, -0.30, 0.30):
        m, beams = _south()
        b = beams[0]
        pat, case = _case(m)
        m.add_frame_distributed_load(b, pat, W, W)
        m.frames[b].joint_offset_i = m.frames[b].joint_offset_j = (0.0, dy, 0.0)
        m.frames[b].offsets_in_analysis = True
        assert C.offset_zone_issues(m, [b]) == []
        ref = copy.deepcopy(m)
        doc = Document(m)
        C.move_beams_to_axis(doc, [b])
        ra, rb = analyze(ref, kernel=SciPyKernel()), analyze(doc.model, kernel=SciPyKernel())
        assert rb.base_reactions(case) == pytest.approx(ra.base_reactions(case), rel=1e-9, abs=1e-9)
        for n in [x for x in ref.frames if ref.frames[x].type != "Rigid"]:
            assert np.abs(rb.frame_forces(n, case).forces).max(axis=0) == \
                pytest.approx(np.abs(ra.frame_forces(n, case).forces).max(axis=0), rel=1e-9, abs=1e-9), (dy, n)
    m, beams = _south()
    f = {"joint_offset_i": (0.0, -0.5, 0.0), "joint_offset_j": (0.0, -0.5, 0.0), "offsets_in_analysis": True}
    assert len(C.offset_zone_issues(m, beams[:1], f)) == 2  # clear of the column: not as an offset
