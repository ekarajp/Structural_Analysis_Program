"""Load panels: one-way strips, two-way tributary areas, height-varying loads,
receivers and the bridge into the analysis (hand solutions)."""
import numpy as np
import pytest

from structstudio import Model, analyze
from structstudio.mesh.panel_loads import distribute, panel_loads_to_members, polygon_frame
from structstudio.model import FIXED, FrameSection
from structstudio.model.objects import PanelLoadCase

RECT = [(0, 0, 0), (6, 0, 0), (6, 5, 0), (0, 5, 0)]
EDGES = {"B1": (np.array([0, 0, 0.]), np.array([6, 0, 0.])), "B2": (np.array([6, 0, 0.]), np.array([6, 5, 0.])),
         "B3": (np.array([0, 5, 0.]), np.array([6, 5, 0.])), "B4": (np.array([0, 0, 0.]), np.array([0, 5, 0.]))}


def totals(res):
    return {b.frame: round(b.total, 6) for b in res.bar_loads}


def test_one_way_rectangle_goes_to_the_two_supporting_edges():
    r = distribute(RECT, "one_way", (1, 0, 0), {"value": 10.0}, (0, 0, -1), EDGES)
    assert totals(r) == {"B2": 150.0, "B4": 150.0}
    assert r.applied_total == pytest.approx(300.0) and r.carried_total == pytest.approx(300.0)
    # the bars parallel to the span receive nothing (CYPE rule); the line load is uniform
    b = next(b for b in r.bar_loads if b.frame == "B2")
    assert np.allclose(b.values, 30.0)  # 10 kN/m2 x 3 m tributary
    r = distribute(RECT, "one_way", (0, 1, 0), {"value": 10.0}, (0, 0, -1), EDGES)
    assert totals(r) == {"B1": 150.0, "B3": 150.0}


def test_one_way_continuous_over_an_interior_beam():
    fe = dict(EDGES)
    fe["B5"] = (np.array([3, 0, 0.]), np.array([3, 5, 0.]))
    r = distribute(RECT, "one_way", (1, 0, 0), {"value": 10.0}, (0, 0, -1), fe)
    assert totals(r) == {"B2": 75.0, "B4": 75.0, "B5": 150.0}  # simply supported strips: half to each support
    # receivers restricted: only the edges carry, the interior beam is disabled (CYPE "Assign")
    r = distribute(RECT, "one_way", (1, 0, 0), {"value": 10.0}, (0, 0, -1), fe, receivers_allowed={"B2", "B4"})
    assert totals(r) == {"B2": 150.0, "B4": 150.0}


def test_two_way_rectangle_45_degree_split():
    r = distribute(RECT, "two_way", None, {"value": 10.0}, (0, 0, -1), EDGES)
    t = totals(r)
    # short beams: triangles q L^2 / 4 = 62.5, long beams: trapezoids 87.5
    assert t["B2"] == pytest.approx(62.5, rel=0.01) and t["B4"] == pytest.approx(62.5, rel=0.01)
    assert t["B1"] == pytest.approx(87.5, rel=0.01) and t["B3"] == pytest.approx(87.5, rel=0.01)
    assert r.carried_total == pytest.approx(300.0, rel=0.01)
    # the long beam carries a trapezoid: zero at the ends, flat in the middle
    b = next(b for b in r.bar_loads if b.frame == "B1")
    assert b.values[0] < b.values[len(b.values) // 2] and abs(b.values[len(b.values) // 2] - 25.0) < 2.0


def test_sloped_panel_with_height_varying_wind_load_normal_to_it():
    poly = [(0, 0, 0), (6, 0, 0), (6, 4, 3), (0, 4, 3)]
    _, _, _, n = polygon_frame(np.array(poly, float))
    rafters = {"R1": (np.array([0, 0, 0.]), np.array([0, 4, 3.])), "R2": (np.array([6, 0, 0.]), np.array([6, 4, 3.])),
               "R3": (np.array([3, 0, 0.]), np.array([3, 4, 3.]))}
    load = {"varying": True, "value_lower": 1.0, "value_upper": 3.0, "z0": 0.0, "z1": 3.0}
    r = distribute(poly, "one_way", (1, 0, 0), load, tuple(n), rafters)
    # area 6 x 5 = 30, mean pressure 2 -> 60; the middle rafter carries half
    assert r.applied_total == pytest.approx(60.0, rel=1e-3) and r.carried_total == pytest.approx(60.0, rel=1e-3)
    t = totals(r)
    assert t["R3"] == pytest.approx(30.0, rel=1e-3) and t["R1"] == pytest.approx(15.0, rel=1e-3)
    b = next(b for b in r.bar_loads if b.frame == "R3")
    assert b.values[-1] > b.values[0]  # pressure grows with height
    assert np.allclose(b.direction, n)


def test_shell_distribution_and_projected_direction():
    areas = {"A1": np.array(RECT, float)}
    r = distribute(RECT, "shells", None, {"value": 4.0}, (0, 0, -1), {}, areas=areas)
    assert r.shell_pressures == {"A1": 4.0} and r.carried_total == pytest.approx(120.0)
    # projected gravity on a 45 deg panel = half the load of the plain gravity load
    m = Model("t", "kN m")
    m.add_load_pattern("W", "Wind")
    poly = [(0, 0, 0), (6, 0, 0), (6, 4, 4), (0, 4, 4)]
    m.add_panel_load(poly, "one_way", (1, 0, 0), loads=[PanelLoadCase("W", 10.0, "GravityProjected")])
    m.add_point(0, 0, 0)
    m.add_point(0, 4, 4)
    m.add_point(6, 0, 0)
    m.add_point(6, 4, 4)
    m.add_material("S", E=2e8, nu=0.3, weight_density=0.0, type="Steel")
    m.add_frame_section(FrameSection.general("B", "S", A=0.01, I33=1e-4, I22=1e-4, J=1e-6))
    m.add_frame((0, 0, 0), (0, 4, 4), "B", name="R1")
    m.add_frame((6, 0, 0), (6, 4, 4), "B", name="R2")
    fl, _, rep = panel_loads_to_members(m)
    applied, carried = rep["PL1"]["W"]
    assert applied == pytest.approx(10.0 * 6 * 4, rel=1e-6)  # plan area x value
    assert carried == pytest.approx(applied, rel=1e-6)
    assert set(fl) == {"R1", "R2"}


def test_panel_loads_reach_the_analysis_and_roundtrip():
    m = Model("t", "kN m")
    m.add_material("S", E=2e8, nu=0.3, weight_density=0.0, type="Steel")
    m.add_frame_section(FrameSection.general("B", "S", A=0.01, I33=1e-4, I22=1e-4, J=1e-6))
    base = [m.add_point(x, y, 0, restraints=FIXED) for x, y in ((0, 0), (6, 0), (6, 5), (0, 5))]
    top = [m.add_point(x, y, 3) for x, y in ((0, 0), (6, 0), (6, 5), (0, 5))]
    for a, b in zip(base, top):
        m.add_frame(a, b, "B", type="Column")
    beams = [m.add_frame(top[i], top[(i + 1) % 4], "B", type="Beam", name=f"BM{i}") for i in range(4)]
    m.add_load_pattern("LL", "Live")
    m.add_panel_load([(0, 0, 3), (6, 0, 3), (6, 5, 3), (0, 5, 3)], "one_way", (0, 1, 0),
                     loads=[PanelLoadCase("LL", 10.0)])
    m.add_linear_static_case("LL")
    res = analyze(m)
    assert res.base_reactions("LL")[2] == pytest.approx(300.0, rel=1e-9)
    assert res.am.panel_report["PL1"]["LL"][1] == pytest.approx(300.0, rel=1e-9)
    # one-way along Y: the beams along X (BM0, BM2) carry 150 each as a uniform 25 kN/m
    f0 = res.frame_forces("BM0", "LL")
    assert abs(f0.forces[0, 1] - f0.forces[-1, 1]) == pytest.approx(150.0, rel=1e-6)  # shear jump = total load
    f1 = res.frame_forces("BM1", "LL")
    assert abs(f1.forces[:, 1]).max() < 1e-6
    # the panel survives the file format and the model copy
    m2 = Model.from_dict(m.to_dict())
    assert m2.panel_loads["PL1"].span_dir == pytest.approx((0.0, 1.0, 0.0))
    assert m2.panel_loads["PL1"].loads[0].pattern == "LL"
