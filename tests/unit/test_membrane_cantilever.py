"""A membrane floor that cantilevers past a beam: the load goes to the beam (two-way)
and the moment at the root is no longer lost - balanced by the floor behind the beam
(a couple on that beam and the next support behind), or torsion in the beam when there
is no floor behind it.  The converted loads must be statically equivalent to the floor
load (same force, same moment about any axis)."""
import math

import numpy as np
import pytest

from structstudio import analyze
from structstudio.app.templates import new_building
from structstudio.engine.kernel import SciPyKernel
from structstudio.mesh.panel_loads import panel_loads_to_members
from structstudio.model.checks import geometry_issues

Z = 3.0
Q = 2.0


def _model(interior=True, one_area=False, opening=False):
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(Z,), grid_only=False)
    for p in m.load_patterns.values():
        p.self_weight_multiplier = 0.0
    fl = next(a for a in m.areas.values() if a.type == "Floor")
    sec = fl.section
    if one_area or not interior:
        del m.areas[fl.name]
    if one_area:
        P = [m.add_point(*c).name for c in ((0, 0, Z), (6, 0, Z), (6, 6.5, Z), (0, 6.5, Z))]
    else:
        P = [m.add_point(*c).name for c in ((0, 5, Z), (6, 5, Z), (6, 6.5, Z), (0, 6.5, Z))]
    a = m.add_area(P, sec, type="Floor")
    m.add_area_load(a.name, "Dead", Q)
    if opening:
        H = [m.add_point(*c).name for c in ((2, 5.5, Z), (4, 5.5, Z), (4, 6.0, Z), (2, 6.0, Z))]
        m.add_area(H, None, type="Opening")
    m.shell_sections[sec].model_type = "Membrane"
    return m, a.name


def _resultants(m, fr):
    """Total force (along -Z) and moments about the X and Y axes (global, through the
    origin) of the converted frame loads."""
    F = Mx = My = 0.0
    for f, loads in fr.items():
        fo = m.frames[f]
        a = np.array(m.points[fo.point_i].xyz)
        b = np.array(m.points[fo.point_j].xyz)
        L = float(np.linalg.norm(b - a))
        for ld in loads:
            n = 200
            for k in range(n):
                r = ld.rel_a + (ld.rel_b - ld.rel_a) * (k + 0.5) / n
                w = ld.value_a + (ld.value_b - ld.value_a) * (k + 0.5) / n
                dl = (ld.rel_b - ld.rel_a) * L / n
                p = a + r * (b - a)
                if ld.load_type == "Force":
                    assert ld.direction == "Gravity"
                    F += w * dl
                    Mx += -p[1] * w * dl  # (r x (0, 0, -w))_x
                    My += p[0] * w * dl
                else:
                    axis = {"X": (1, 0, 0), "-X": (-1, 0, 0), "Y": (0, 1, 0), "-Y": (0, -1, 0)}[ld.direction]
                    Mx += axis[0] * w * dl
                    My += axis[1] * w * dl
    return F, Mx, My


@pytest.mark.parametrize("case", ["behind", "balcony", "one_area", "opening"])
def test_converted_loads_are_statically_equivalent(case):
    m, name = _model(interior=(case != "balcony"), one_area=(case == "one_area"), opening=(case == "opening"))
    notes = []
    fr, _, _ = panel_loads_to_members(m, notes=notes)
    F, Mx, My = _resultants(m, fr)
    if case == "one_area":
        area, yc, xc = 6 * 6.5, 6.5 / 2, 3.0
    elif case == "opening":
        area, yc, xc = 6 * 1.5 - 2 * 0.5, (6 * 1.5 * 5.75 - 2 * 0.5 * 5.75) / (6 * 1.5 - 1.0), 3.0
    else:
        area, yc, xc = 6 * 1.5, 5.75, 3.0
    assert F == pytest.approx(Q * area, rel=2e-3)
    assert Mx == pytest.approx(-Q * area * yc, rel=3e-3)  # the root moment is not lost
    assert My == pytest.approx(Q * area * xc, rel=3e-3)
    modes = {n[2] for n in notes}
    assert modes == ({"torsion"} if case == "balcony" else {"balanced"})


def test_balanced_couple_on_the_beam_behind_and_torsion_without_floor():
    m, _ = _model(interior=True)
    fr, _, _ = panel_loads_to_members(m)
    beam_edge = next(f.name for f in m.frames.values() if f.type == "Beam" and
                     all(abs(m.points[p].y - 5) < 1e-9 for p in (f.point_i, f.point_j)))
    beam_far = next(f.name for f in m.frames.values() if f.type == "Beam" and
                    all(abs(m.points[p].y) < 1e-9 for p in (f.point_i, f.point_j)))
    tot = {f: sum((ld.value_a + ld.value_b) / 2 * (ld.rel_b - ld.rel_a) * 6 for ld in fr[f]) for f in fr}
    S = Q * 6 * 1.5 * 0.75
    assert tot[beam_edge] == pytest.approx(Q * 9 + S / 5, rel=2e-3)
    assert tot[beam_far] == pytest.approx(-S / 5, rel=2e-3)
    m, _ = _model(interior=False)
    fr, _, _ = panel_loads_to_members(m)
    torque = [ld for ld in fr[beam_edge] if ld.load_type == "Moment"]
    assert torque and sum((ld.value_a + ld.value_b) / 2 * (ld.rel_b - ld.rel_a) * 6 for ld in torque) \
        == pytest.approx(-S, rel=2e-3)


def test_floor_supported_all_round_has_no_overhang():
    m = new_building(x_spacings=(6, 6), y_spacings=(5, 5), story_heights=(Z,), grid_only=False)
    for a in m.areas.values():
        if a.type == "Floor":
            m.add_area_load(a.name, "Dead", Q)
            m.shell_sections[a.section].model_type = "Membrane"
    notes = []
    fr, _, _ = panel_loads_to_members(m, notes=notes)
    assert notes == []
    assert not any(ld.load_type == "Moment" for ls in fr.values() for ld in ls)


def test_check_model_says_where_the_root_moment_goes_and_analysis_balances():
    m, name = _model(interior=False)
    codes = {i.code for i in geometry_issues(m)}
    assert "MEMBRANE_CANTILEVER_TORSION" in codes
    r = analyze(m, kernel=SciPyKernel())
    assert r.base_reactions("Dead")[2] == pytest.approx(Q * 9, rel=1e-6)
    m, name = _model(interior=True)
    assert "MEMBRANE_CANTILEVER" in {i.code for i in geometry_issues(m)}
