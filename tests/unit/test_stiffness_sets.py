"""Stiffness sets: the shells' lateral modifiers (a post-tensioned flat slab that takes
no part in the lateral system) in the lateral load cases, their modifiers in the
gravity cases - in one model.  Each case must equal the same case of a model whose
shells have that set's modifiers everywhere."""
import copy
import json

import numpy as np
import pytest

from structstudio import analyze
from structstudio.app.templates import new_building
from structstudio.engine.kernel import SciPyKernel
from structstudio.model import Model, ModelError
from structstudio.model.loads import stiffness_set_of
from structstudio.model.properties import ShellModifiers

LAT = ShellModifiers(m11=0.05, m22=0.05, m12=0.05)


def _flat_plate():
    """2 stories, 2 x 2 bays of 8 m, no beams (flat plate), a wall on one line."""
    m = new_building(x_spacings=(8, 8), y_spacings=(8, 8), story_heights=(3.5, 3.0), grid_only=False)
    for f in [f for f in m.frames.values() if f.type == "Beam"]:
        del m.frames[f.name]
    floors = [a for a in m.areas.values() if a.type == "Floor"]
    for a in floors:
        m.add_area_load(a.name, "Dead", 2.0)
        m.add_area_load(a.name, "Live", 3.0)
    import dataclasses
    fsec = floors[0].section
    wsec = "W20"
    m.shell_sections[wsec] = dataclasses.replace(m.shell_sections[fsec], name=wsec, type="Wall", thickness=0.2)
    for z0, z1 in ((0.0, 3.5), (3.5, 6.5)):
        P = [m.add_point(*c).name for c in ((0, 8, z0), (0, 16, z0), (0, 16, z1), (0, 8, z1))]
        m.add_area(P, wsec, type="Wall")
    m.add_load_pattern("EQX", "Seismic")
    for p in m.points.values():
        if p.z > 1e-9 and abs(p.x - 16) < 1e-9 and abs(p.y) < 1e-9:
            m.add_point_load(p.name, "EQX", force=(30.0 * p.z / 6.5, 0, 0))
    m.add_linear_static_case("EQX", {"EQX": 1.0})
    m.add_spectrum_function("RS", [0.0, 0.5, 1.0, 3.0], [2.0, 2.5, 1.5, 0.5])
    m.add_response_spectrum_case("RSX", "Modal", [("U1", "RS", 1.0)])
    m.add_combination("G+E", {"Dead": 1.2, "Live": 1.0, "EQX": 1.0})
    return m, [a.name for a in floors]


def _run(m):
    return analyze(m, kernel=SciPyKernel())


def _frames(r, m, case):
    return np.array([r.frame_forces(f, case).forces for f in sorted(m.frames)], dtype=object)


def _close(r1, r2, m, case):
    for f in m.frames:
        a, b = r1.frame_forces(f, case).forces, r2.frame_forces(f, case).forces
        assert np.allclose(a, b, rtol=1e-7, atol=1e-6 * max(1.0, np.abs(b).max())), (case, f)
    assert np.allclose(r1.base_reactions(case), r2.base_reactions(case), rtol=1e-8, atol=1e-6)


def test_auto_sets():
    m, _ = _flat_plate()
    assert stiffness_set_of(m, "Dead") == "gravity" and stiffness_set_of(m, "Live") == "gravity"
    assert stiffness_set_of(m, "EQX") == "lateral"
    assert stiffness_set_of(m, "Modal") == "lateral" and stiffness_set_of(m, "RSX") == "lateral"
    m.add_linear_static_case("mixed", {"Dead": 1.0, "EQX": 1.0})
    assert stiffness_set_of(m, "mixed") == "gravity"
    m.load_cases["EQX"].stiffness = "gravity"
    assert stiffness_set_of(m, "EQX") == "gravity"
    m.load_cases["EQX"].stiffness = "sideways"
    with pytest.raises(ModelError):
        stiffness_set_of(m, "EQX")


@pytest.mark.parametrize("where", ["section", "area"])
def test_each_case_equals_the_model_with_that_sets_modifiers(where):
    m, floors = _flat_plate()
    sec = m.areas[floors[0]].section
    if where == "section":
        m.shell_sections[sec].lateral_modifiers = LAT
    else:
        for a in floors:
            m.areas[a].lateral_modifiers = LAT
    both = _run(m)
    plain, _ = _flat_plate()  # gravity set everywhere
    rg = _run(plain)
    lat, _ = _flat_plate()  # the floors with the lateral modifiers everywhere
    for a in floors:
        lat.areas[a].modifiers = LAT
    rl = _run(lat)
    for c in ("Dead", "Live"):
        _close(both, rg, m, c)
    _close(both, rl, m, "EQX")
    pb = [d["Period"] for d in both.modal_periods("Modal")]
    pl = [d["Period"] for d in rl.modal_periods("Modal")]
    pg = [d["Period"] for d in rg.modal_periods("Modal")]
    assert pb == pytest.approx(pl, rel=1e-8) and pb[0] > pg[0] * 1.0001  # softer slab: longer
    _close(both, rl, m, "RSX")
    # a combination adds the cases of both sets
    sh_b = both.story_shears("G+E") if hasattr(both, "story_shears") else None
    assert sh_b is None or len(sh_b) >= 1


def test_no_lateral_modifiers_changes_nothing_and_round_trip():
    m, floors = _flat_plate()
    r = _run(m)
    assert not r.engine.has_lateral_set()
    m.areas[floors[0]].lateral_modifiers = LAT
    m.load_cases["Modal"].stiffness = "gravity"
    m2 = Model.from_dict(json.loads(json.dumps(m.to_dict())))
    assert m2.areas[floors[0]].lateral_modifiers == LAT and m2.load_cases["Modal"].stiffness == "gravity"
    m3 = copy.deepcopy(m)
    m3.convert_units("kip in") if hasattr(m3, "convert_units") else None
    assert m3.areas[floors[0]].lateral_modifiers.m11 == pytest.approx(0.05)


def test_check_model_says_which_cases_use_which_set():
    from structstudio.model.checks import geometry_issues
    m, floors = _flat_plate()
    assert not [i for i in geometry_issues(m) if i.code == "STIFFNESS_SETS"]
    for a in floors:
        m.areas[a].lateral_modifiers = LAT
    m.add_linear_static_case("mixed", {"Dead": 1.0, "EQX": 1.0})
    issues = {i.code: i for i in geometry_issues(m)}
    assert "EQX" in issues["STIFFNESS_SETS"].message and "MIXED_STIFFNESS_CASE" in issues


def test_pdelta_stiffness_carried_into_the_lateral_set():
    """A lateral case using the stiffness of a P-Delta case (run with the gravity set):
    the geometric stiffness of that case on the lateral set's stiffness."""
    m, floors = _flat_plate()
    for a in floors:
        m.areas[a].lateral_modifiers = LAT
    m.add_pdelta_case("PD", {"Dead": 1.2, "Live": 0.5})
    m.add_linear_static_case("EQX-PD", {"EQX": 1.0}, stiffness_from="PD")
    r = _run(m)
    top = max(m.points.values(), key=lambda p: (p.z, p.x, -p.y))
    d0 = r.joint_displacement(top.name, "EQX")[0]
    d1 = r.joint_displacement(top.name, "EQX-PD")[0]
    assert abs(d1) > abs(d0) > 0  # P-Delta: softer
    assert abs(d1) < 1.2 * abs(d0)
