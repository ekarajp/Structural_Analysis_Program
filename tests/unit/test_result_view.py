"""Results on the members: colours along members (line / extruded), shell
contours and deformed shapes with real sections and thickness."""
import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

from structstudio import api  # noqa: E402
from structstudio.app import result_view as rv, scene  # noqa: E402
from structstudio.app.templates import new_building  # noqa: E402


@pytest.fixture(scope="module")
def solved():
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5), grid_only=False)
    for a in m.areas.values():
        m.add_area_load(a.name, "Live", 90.0)  # every colour band (RZF 0.5, column rigid zones)
    m.options.mesh_refine_check = True  # the converged mesh this load was chosen with
    return m, api.analyze(m)


def _bounds(actor):
    return np.array(actor.GetMapper().GetInput().GetBounds())


def test_stress_ratio_is_the_elastic_extreme_fibre_stress_over_fc(solved):
    m, res = solved
    st = res.frame_forces("C3", "Live")
    F = np.abs(st.forces)
    S = 0.4 ** 3 / 6
    hand = float(((F[:, 0] / 0.16 + F[:, 4] / S + F[:, 5] / S) / 28000).max())
    assert rv.ratio_summary(m, res, "Live")["C3"] == pytest.approx(hand, rel=1e-9)
    r = rv.ratio_summary(m, res, "Live")
    assert max(r.values()) > 1.0 and min(r.values()) < 0.5   # this load shows every colour band


def test_ratio_colours_follow_the_csi_bands():
    lut = rv.ratio_lut()
    def col(v):
        c = [0.0, 0.0, 0.0]
        lut.GetColor(v, c)
        return tuple(round(x, 3) for x in c)
    rgb = lambda h: tuple(round(c, 3) for c in rv._rgb(h))
    assert col(0.3) == rgb("#7aa6d8") and col(0.6) == rgb("#2ec4d6") and col(0.8) == rgb("#36b34a")
    assert col(0.95) == rgb("#f2b01e") and col(1.1) == rgb("#e02424")


@pytest.mark.parametrize("q", ["P", "M3", "Stress ratio"])
def test_frame_colours_as_lines_and_as_real_sections(solved, q):
    m, res = solved
    line = rv.frame_color_actors(m, res, "Live", q, extruded=False)
    solid = rv.frame_color_actors(m, res, "Live", q, extruded=True)
    assert isinstance(line[-1], vtk.vtkScalarBarActor) and isinstance(solid[-1], vtk.vtkScalarBarActor)
    lp, sp = line[0].GetMapper().GetInput(), solid[0].GetMapper().GetInput()
    assert lp.GetNumberOfLines() > 0 and lp.GetNumberOfPolys() == 0
    assert sp.GetNumberOfPolys() > 0
    # the solids have the section size: beams 0.6 deep hang below the floor
    zb = _bounds(solid[0])
    assert zb[4] == pytest.approx(-0.0, abs=1e-9) and zb[5] == pytest.approx(7.0, abs=1e-9)
    assert _bounds(line[0])[4] == pytest.approx(0.0)
    # colours vary along a member: more than two values on a beam (M3) / a column (P)
    vals = rv.frame_values(m, res, "Live", q)["C1" if q == "P" else "B1"][1]
    assert vals.max() > 0 and (q == "P" or len(np.unique(np.round(vals, 6))) > 2)


def test_contour_extruded_has_the_slab_thickness(solved):
    m, res = solved
    flat, rng = scene.build_contour_actors(m, res, "Live", "M11")
    thick, rng2 = scene.build_contour_actors(m, res, "Live", "M11", extruded=True)
    assert rng == rng2
    zf = _bounds(flat[0])
    zt = _bounds(thick[0])
    t = next(iter(m.shell_sections.values())).thickness
    slab_t = max(s.thickness for s in m.shell_sections.values() if s.name in {a.section for a in m.areas.values()})
    assert zf[5] - zf[4] == pytest.approx(3.5)            # floors at 3.5 and 7
    assert zt[4] == pytest.approx(3.5 - slab_t)            # top of slab at the level, slab below
    assert thick[0].GetMapper().GetInput().GetNumberOfPolys() > 5 * flat[0].GetMapper().GetInput().GetNumberOfPolys()
    assert t > 0


def test_deformed_shape_extruded_sweeps_the_sections(solved):
    m, res = solved
    lines, s1 = scene.build_deformed_mesh_actors(m, res, "Live")
    solid, s2 = scene.build_deformed_mesh_actors(m, res, "Live", extruded=True)
    assert s1 == s2
    assert isinstance(solid[-1], vtk.vtkScalarBarActor)
    poly = solid[0].GetMapper().GetInput()
    assert poly.GetNumberOfPolys() > 1000
    rng = poly.GetPointData().GetScalars().GetRange()
    assert rng[0] == pytest.approx(0.0, abs=1e-9) and rng[1] > 0
