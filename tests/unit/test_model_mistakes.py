"""Check Model (and the check before Analyze) catches the mistakes a new user makes,
and the check right after drawing says so at once; correct models stay quiet."""
import numpy as np
import pytest

from structstudio.app.document import Document
from structstudio.app.templates import new_building
from structstudio.model.checks import drawing_issues, geometry_issues

NEW = {"NEAR_MISS", "AREAS_OVERLAP", "BASE_NOT_SUPPORTED", "FAR_FROM_GRID", "INVALID_SECTION", "UNUSUAL_E",
       "SELF_WEIGHT_TWICE", "NO_SELF_WEIGHT", "UPWARD_GRAVITY_LOAD"}


def _m(floors=True, hs=(3.0,)):
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=hs, grid_only=False)
    if not floors:
        for a in [a for a in m.areas if m.areas[a].type == "Floor"]:
            del m.areas[a]
    return m


def _codes(m):
    return {i.code for i in geometry_issues(m)}


def _beam(m, **k):
    P = m.points
    return sorted(f.name for f in m.frames.values() if f.type == "Beam"
                  and all(abs(getattr(P[q], a) - v) < 1e-9 for a, v in k.items() for q in (f.point_i, f.point_j)))[0]


def test_correct_models_stay_quiet():
    for kw in ({}, {"grid_only": True}):
        m = new_building(x_spacings=(6, 6), y_spacings=(5, 5), story_heights=(3, 3), **kw)
        assert not (_codes(m) & NEW)


@pytest.mark.parametrize("gap", [0.002, 0.02, 0.10])
def test_beam_end_just_short_of_the_column_joint(gap):
    m = _m(floors=False)
    f = m.frames[_beam(m, y=0.0)]
    p = m.points[f.point_j]
    f.point_j = m.add_point(p.x - gap, p.y, p.z, merge=False).name
    iss = [i for i in geometry_issues(m) if i.code == "NEAR_MISS"]
    assert len(iss) == 1 and f"{gap:.3f}" in iss[0].message and iss[0].frames == [f.name]


def test_free_end_beside_a_member_and_a_real_cantilever():
    m = _m(floors=False)
    a = m.add_point(3.0, 2.0, 3.0)
    b = m.add_point(3.0, 0.03, 3.0)  # 30 mm short of the edge beam
    m.add_frame(a.name, b.name, "B30x60")
    assert "NEAR_MISS" in _codes(m)
    m2 = _m(floors=False)  # a cantilever 1.5 out: its free end is near nothing - not a mistake
    j = next(p.name for p in m2.points.values() if p.xyz == (0.0, 0.0, 3.0))
    m2.add_frame(j, m2.add_point(0.0, -1.5, 3.0).name, "B30x60")
    assert "NEAR_MISS" not in _codes(m2)


def test_floors_and_walls_overlapping():
    m = _m()
    fl = next(a for a in m.areas.values() if a.type == "Floor")
    P = [m.points[p].xyz for p in fl.points]
    m.add_area([m.add_point(x + 1.0, y + 1.0, z).name for x, y, z in P], fl.section, type="Floor")
    assert "AREAS_OVERLAP" in _codes(m)
    m = _m()  # floors side by side, sharing an edge: fine
    assert "AREAS_OVERLAP" not in _codes(m)
    m = _m(floors=False)
    sec = next(iter(m.shell_sections))
    j = lambda x, z: m.add_point(x, 0.0, z).name
    m.add_area([j(0, 0), j(12, 0), j(12, 3), j(0, 3)], sec, type="Wall")
    m.add_area([j(3, 0), j(9, 0), j(9, 3), j(3, 3)], sec, type="Wall")
    assert "AREAS_OVERLAP" in _codes(m)


def test_column_base_without_support():
    m = _m()
    c = next(f for f in m.frames.values() if f.type == "Column")
    lo = min((c.point_i, c.point_j), key=lambda q: m.points[q].z)
    m.points[lo].restraints = (False,) * 6
    iss = [i for i in geometry_issues(m) if i.code == "BASE_NOT_SUPPORTED"]
    assert len(iss) == 1 and lo in iss[0].points


def test_far_from_the_grid():
    m = _m(floors=False)
    j = next(p.name for p in m.points.values() if p.xyz == (0.0, 0.0, 3.0))
    m.add_frame(j, m.add_point(100000.0, 0.0, 3.0).name, "B30x60")
    assert "FAR_FROM_GRID" in _codes(m)


def test_section_without_size_and_unusual_E():
    m = _m()
    m.frame_sections["B30x60"].t3 = 0.0
    assert "INVALID_SECTION" in _codes(m)
    m = _m()
    mat = next(x for x in m.materials.values() if x.type == "Concrete")
    mat.E = mat.E / 1000.0  # MPa typed where kN/m2 is asked
    assert "UNUSUAL_E" in _codes(m)


def test_self_weight_twice_or_nowhere_and_upward_loads():
    m = _m()
    m.load_patterns["Live"].self_weight_multiplier = 1.0
    assert "SELF_WEIGHT_TWICE" in _codes(m)
    m = _m()
    for p in m.load_patterns.values():
        p.self_weight_multiplier = 0.0
    assert [i.severity for i in geometry_issues(m) if i.code == "NO_SELF_WEIGHT"] == ["info"]
    m = _m()
    m.add_area_load(next(a.name for a in m.areas.values() if a.type == "Floor"), "Live", -3.0)
    assert "UPWARD_GRAVITY_LOAD" in _codes(m)


def test_said_right_after_drawing():
    """The check run after every edit (the "Check what was drawn" box): a beam end that
    missed the joint, a floor drawn over another."""
    doc = Document(_m(floors=False))

    def short(mm):
        mm.add_frame(mm.add_point(3.0, 2.5, 3.0).name, mm.add_point(5.98, 2.5, 3.0).name, "B30x60")  # 20 mm short
    doc.edit(short)
    assert [i.code for i in doc.duplicates] == ["NEAR_MISS"]
    doc = Document(_m())
    fl = next(a for a in doc.model.areas.values() if a.type == "Floor")

    def over(mm):
        P = [mm.points[p].xyz for p in fl.points]
        mm.add_area([mm.add_point(x + 1.0, y, z).name for x, y, z in P], fl.section, type="Floor")
    doc.edit(over)
    assert "AREAS_OVERLAP" in {i.code for i in doc.duplicates}
    assert drawing_issues(doc.model, [], []) == []
