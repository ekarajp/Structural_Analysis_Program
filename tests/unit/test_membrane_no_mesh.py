"""Membrane floors on auto mesh: no element in a rigid diaphragm (the joints under the
floor join the diaphragm and take its mass), meshed as before otherwise.  The
results must be those of the finely meshed membrane (MeshOptions(membrane_coarse=False)),
with and without the slab-column rigid zones."""
import numpy as np
import pytest

from structstudio import analyze
from structstudio.app.templates import new_building
from structstudio.engine.kernel import SciPyKernel
from structstudio.mesh.mesher import MeshOptions


def _model(dia="rigid", balcony=False, zones=False):
    m = new_building(x_spacings=(6, 6), y_spacings=(5, 5), story_heights=(3.5, 3.0), grid_only=False)
    m.options.slab_column_rigid_zone = zones
    floors = [a for a in m.areas.values() if a.type == "Floor"]
    for s in m.shell_sections.values():
        if s.name.startswith("S"):
            s.model_type = "Membrane"
    for a in floors:
        m.add_area_load(a.name, "Dead", 1.5)
    if balcony:
        P = [m.add_point(*c).name for c in ((0, 10, 3.5), (6, 10, 3.5), (6, 11.5, 3.5), (0, 11.5, 3.5))]
        b = m.add_area(P, floors[0].section, type="Floor")
        b.diaphragm = floors[0].diaphragm
        m.add_area_load(b.name, "Dead", 3.0)
    if dia == "semi":
        for d in m.diaphragms.values():
            d.rigid = False
    m.add_load_pattern("EQX")
    for p in m.points.values():
        if p.z > 1e-9 and abs(p.x) < 1e-9 and abs(p.y) < 1e-9:
            m.add_point_load(p.name, "EQX", force=(20.0 * p.z / 6.5, 0, 0))
    m.add_linear_static_case("EQX")
    ms = next(iter(m.mass_sources.values()))
    ms.load_patterns = {"Dead": 1.0}
    return m


def _run(m, coarse):
    return analyze(m, kernel=SciPyKernel(), mesh_options=MeshOptions(membrane_coarse=coarse))


def _compare(rn, rr, m, tol_force, tol_period):
    for c in ("Dead", "EQX"):
        assert np.allclose(rn.base_reactions(c), rr.base_reactions(c), rtol=1e-6, atol=1e-6)
        for comp in (0, 1, 3, 5):
            a = np.array([np.abs(rn.frame_forces(f, c).forces[:, comp]).max() for f in m.frames])
            b = np.array([np.abs(rr.frame_forces(f, c).forces[:, comp]).max() for f in m.frames])
            assert np.max(np.abs(a - b)) <= tol_force * max(b.max(), 1e-9), (c, comp)
    pn = [d["Period"] for d in rn.modal_periods("Modal")][:6]
    pr = [d["Period"] for d in rr.modal_periods("Modal")][:6]
    assert pn == pytest.approx(pr, rel=tol_period)
    assert rn.engine.asm.mass_full()[0::6].sum() == pytest.approx(rr.engine.asm.mass_full()[0::6].sum(), rel=1e-9)


@pytest.mark.parametrize("balcony,zones", [(False, False), (True, False), (False, True), (True, True)])
def test_rigid_diaphragm_membrane_is_no_element_same_results(balcony, zones):
    """With the slab-column rigid zones (the default) the beams in the floor keep a node on
    the column face held by the zone, as with the meshed floor."""
    m = _model(balcony=balcony, zones=zones)
    rn, rr = _run(m, True), _run(m, False)
    assert len(rn.am.shells) == 0 and len(rr.am.shells) > 100
    _compare(rn, rr, m, tol_force=5e-3, tol_period=2e-3)
    assert not any("Null DOF" in w for w in rn.am.warnings)


@pytest.mark.parametrize("zones", [False, True])
def test_semi_rigid_membrane_is_meshed_as_before(zones):
    """Semi-rigid: meshed as before (the beams' chord / collector forces need the mesh:
    1 m elements were 11 % off at 0.25 m), the same results."""
    m = _model(dia="semi", zones=zones)
    rn, rr = _run(m, True), _run(m, False)
    assert len(rn.am.shells) == len(rr.am.shells) > 0
    _compare(rn, rr, m, tol_force=1e-9, tol_period=1e-9)


def test_manual_mesh_size_meshes_a_membrane_like_a_slab():
    m = _model()
    for a in m.areas.values():
        if a.type == "Floor":
            a.max_mesh_size = 1.0
    r = _run(m, True)
    assert len(r.am.shells) > 0
