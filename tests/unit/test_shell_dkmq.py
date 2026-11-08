"""ShellThick = DKMQ (Katili 1993): the Mindlin solution of a thick plate,
the DKQ (ShellThin) when the plate is thin, any quad shape."""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from structstudio.engine.elements import shell as sh
from structstudio.mesh.analysis_model import ShellElement


def _plate(n, a, t, kind, distort=0.0, E=1e7, nu=0.3, q=1.0):
    """Hard simply supported square plate (w and the tangential rotation fixed)
    under uniform load: centre deflection."""
    G = E / (2 * (1 + nu))
    xs = np.linspace(0, a, n + 1)
    X = np.array([[x, y] for y in xs for x in xs])
    rng = np.random.default_rng(1)
    for k, (x, y) in enumerate(X):
        if distort and 0 < x < a and 0 < y < a and k != (n // 2) * (n + 1) + n // 2:
            X[k] += distort * a / n * (rng.random(2) - 0.5)
    nn = len(X)
    K = sp.lil_matrix((6 * nn, 6 * nn))
    F = np.zeros(6 * nn)
    for j in range(n):
        for i in range(n):
            ids = [j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i]
            el = ShellElement(0, "A", tuple(ids), E, nu, G, t, t, kind, 0.0, 0.0, np.ones(8))
            f = sh.ShellFormulation(el, np.c_[X[ids], np.zeros(4)])
            dofs = np.concatenate([np.arange(6 * d, 6 * d + 6) for d in ids])
            K[np.ix_(dofs, dofs)] += f.stiffness_global()
            F[2 + 6 * np.array(ids)] += -q * f.tributary
    fixed = set()
    for k, (x, y) in enumerate(X):
        fixed.update([6 * k, 6 * k + 1, 6 * k + 5])
        if abs(x) < 1e-9 or abs(x - a) < 1e-9:
            fixed.update([6 * k + 2, 6 * k + 3])
        if abs(y) < 1e-9 or abs(y - a) < 1e-9:
            fixed.update([6 * k + 2, 6 * k + 4])
    free = np.array(sorted(set(range(6 * nn)) - fixed))
    K = K.tocsr()
    u = np.zeros(6 * nn)
    u[free] = spl.spsolve(K[free][:, free].tocsc(), F[free])
    return -u[6 * ((n // 2) * (n + 1) + n // 2) + 2]


def _navier(a, t, E=1e7, nu=0.3, q=1.0, mindlin=True):
    D = E * t ** 3 / (12 * (1 - nu ** 2))
    G = E / (2 * (1 + nu))
    w = 0.0
    for m in range(1, 300, 2):
        for n in range(1, 300, 2):
            k2 = (m / a) ** 2 + (n / a) ** 2
            qmn = 16 * q / (np.pi ** 2 * m * n)
            w += qmn / (D * np.pi ** 4 * k2 ** 2) * (1 + (D * np.pi ** 2 * k2 / (5 / 6 * G * t) if mindlin else 0)) \
                * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
    return w


@pytest.mark.parametrize("t", [2.0, 1.0])
def test_thick_plate_converges_to_mindlin(t):
    assert sh.THICK_FORMULATION == "DKMQ"
    ref = _navier(10.0, t)
    assert ref / _navier(10.0, t, mindlin=False) > 1.04          # shear deformation matters here
    assert _plate(16, 10.0, t, "ShellThick") == pytest.approx(ref, rel=2e-3)
    assert _plate(16, 10.0, t, "ShellThick", distort=0.6) == pytest.approx(ref, rel=5e-3)


def test_thin_limit_is_the_dkq():
    thick = _plate(8, 10.0, 0.01, "ShellThick")
    thin = _plate(8, 10.0, 0.01, "ShellThin")
    assert thick == pytest.approx(thin, rel=1e-4)
    assert thin == pytest.approx(_navier(10.0, 0.01, mindlin=False), rel=2e-3)
