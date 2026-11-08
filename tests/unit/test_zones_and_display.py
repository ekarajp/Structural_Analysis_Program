"""Column cross-sections followed exactly by the slab mesh (rigid zones of the real
size at every mesh size), results starting at the column face, and the result
display helpers (round colour scales, plan diagram sides, results summary)."""
import numpy as np
import pytest

from structstudio.api import analyze
from structstudio.app.templates import new_building
from structstudio.mesh import MeshOptions, build_analysis_model


def _frame():
    return new_building(x_spacings=(6,), y_spacings=(6,), story_heights=(3.0,), grid_only=False)


@pytest.mark.parametrize("h", [1.0, 0.5, 0.25])
def test_rigid_zone_is_the_column_section_at_every_mesh_size(h):
    m = _frame()
    am = build_analysis_model(m, MeshOptions(floor_max_size=h, refine_check=False))
    assert len(am.bodies) == 4  # four corner columns, cut to the slab
    for b in am.bodies:
        c = am.nodes[b.master].xyz
        X = np.array([am.nodes[n].xyz for n in b.nodes])
        ext = X[:, :2].max(axis=0) - X[:, :2].min(axis=0)
        assert ext == pytest.approx([0.2, 0.2], abs=1e-9)  # a quarter of the 0.4 m column, exactly
        assert np.all(np.abs(X[:, :2] - c[:2]) <= 0.2 + 1e-9)


def test_beam_results_start_at_the_column_face():
    m = _frame()
    r = analyze(m, mesh_options=MeshOptions(floor_max_size=0.5, refine_check=False))
    beam = next(n for n, f in m.frames.items() if f.type == "Beam")
    case = next(iter(r.cases))
    st = r.frame_forces(beam, case)
    assert st.station[0] >= 0.2 - 1e-9 and st.station[-1] <= 6.0 - 0.2 + 1e-9
    assert abs(st.forces[0, 5]) > 0.2 * np.max(np.abs(st.forces[:, 5]))  # a real end moment, not 0
    assert not (r.rigid_zone_shells() & {s.id for s in r.am.shells if s.object not in m.areas})


def test_round_colour_scale_and_display_units():
    from structstudio.app import scene
    lo, hi, step, n = scene.nice_scale(0.0, 0.00158)
    assert (lo, step) == (0.0, 0.0002) and hi == pytest.approx(0.0016) and n == 8
    lo, hi, step, n = scene.nice_scale(-5.62, 6.52)
    assert (lo, hi, step) == (-6.0, 7.0, 1.0)
    assert scene.display_length(_frame()) == (1000.0, "mm")


def test_plan_diagrams_put_positive_on_one_side():
    from structstudio.app import scene
    z = (0.0, 0.0, 1.0)
    for v in ((1, 0, 0), (-1, 0, 0)):  # a beam drawn either way: positive below
        assert scene._plan_side(z, np.array(v, float)) == pytest.approx([0, -1, 0], abs=1e-12)
    for v in ((0, 1, 0), (0, -1, 0)):  # vertical in the plan: positive to the right
        assert scene._plan_side(z, np.array(v, float)) == pytest.approx([1, 0, 0], abs=1e-12)


def test_results_summary_rows():
    from structstudio.app.object_props import results_summary
    m = _frame()
    r = analyze(m, mesh_options=MeshOptions(floor_max_size=0.5, refine_check=False))
    rows = dict(results_summary(m, r, next(iter(r.cases))))
    assert "mm at" in rows["Largest vertical deflection"]
    assert rows["Analysis model"].endswith("shell elements")
