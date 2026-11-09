"""ScalarMesh built from arrays (slabs of many shells, swept members) against
the one-polygon-at-a-time construction it replaced: same points, values and
cells, in the same order."""
import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

from vtkmodules.util import numpy_support as nps  # noqa: E402

from structstudio.app.result_view import ScalarMesh, thick_shell, thick_shells  # noqa: E402
from tests.unit.test_app_scene import small_building  # noqa: E402


def _ref_sweep(mesh, path, vals, rings):
    """The former per-quad sweep."""
    for ring in rings:
        m = len(ring)
        for k in range(len(path) - 1):
            a, b = path[k] + ring, path[k + 1] + ring
            for i in range(m):
                j = (i + 1) % m
                mesh.polygon([a[i], a[j], b[j], b[i]], [vals[k], vals[k], vals[k + 1], vals[k + 1]])
        mesh.polygon((path[0] + ring)[::-1], vals[0])
        mesh.polygon(path[-1] + ring, vals[-1])


def _arrays(mesh):
    X = np.vstack(mesh.pts)
    V = np.concatenate(mesh.vals)
    return X, V, mesh.polys


def test_swept_member_matches_the_per_quad_construction():
    rng = np.random.default_rng(3)
    path = np.cumsum(rng.normal(size=(7, 3)), axis=0)
    vals = rng.normal(size=7)
    rings = [rng.normal(size=(5, 3)), rng.normal(size=(3, 3))]
    a, b = ScalarMesh(), ScalarMesh()
    _ref_sweep(a, path, vals, rings)
    b.sweep(path, vals, rings)
    Xa, Va, Pa = _arrays(a)
    Xb, Vb, Pb = _arrays(b)
    assert np.array_equal(Xa, Xb) and np.array_equal(Va, Vb) and Pa == Pb
    pa, pb = a.polydata(), b.polydata()
    assert pa.GetNumberOfPolys() == pb.GetNumberOfPolys()
    assert np.array_equal(nps.vtk_to_numpy(pa.GetPoints().GetData()), nps.vtk_to_numpy(pb.GetPoints().GetData()))


def test_thick_shells_match_the_element_by_element_slabs():
    m = small_building()
    m.add_area([p for p in list(m.points)[:4]], "S", name="WALL")  # base points: a vertical-ish plane
    m.areas["WALL"].type = "Wall"
    rng = np.random.default_rng(5)
    P = rng.normal(size=(9, 4, 3))
    P[3] = 0.0  # a degenerate element: normal (0, 0, 1)
    V = rng.normal(size=(9, 4))
    objs = ["F1", "WALL", "F1", "F1", "NONE", "WALL", "F1", "NONE", "F1"]  # NONE: no thickness -> flat
    a, b = ScalarMesh(), ScalarMesh()
    for i in range(len(P)):
        thick_shell(a, m, objs[i], P[i], V[i])
    thick_shells(b, m, objs, P, V)
    Xa, Va, Pa = _arrays(a)
    Xb, Vb, Pb = _arrays(b)
    assert Pa == Pb and np.array_equal(Va, Vb)
    assert np.allclose(Xa, Xb, rtol=1e-14, atol=1e-14)
