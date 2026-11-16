"""When the double-precision PARDISO factor does not fit in the free memory, the
stiffness is factored in single precision (unit-diagonal scaling) and every solve is
refined by double-precision CG to the same equilibrium check: same results."""
import numpy as np
import pytest
from scipy import sparse

kp = pytest.importorskip("structstudio.engine.kernel_pardiso")
try:
    KERNEL = kp.PardisoKernel()
except Exception:  # pragma: no cover - no MKL
    pytest.skip("PARDISO not available", allow_module_level=True)

from structstudio import Model, analyze  # noqa: E402
from structstudio.model import FIXED, FrameSection  # noqa: E402


def _no_memory_for_double(monkeypatch):
    real = kp.available_memory
    calls = []

    def fake():
        calls.append(1)
        return 1.0 if len(calls) == 1 else real()  # the first (double) check finds nothing free
    monkeypatch.setattr(kp, "available_memory", fake)


def _ill_scaled_spd(n=400, seed=1):
    rng = np.random.default_rng(seed)
    A = sparse.random(n, n, density=0.02, random_state=rng)
    K = (A @ A.T + sparse.eye(n)).tocsr()
    s = 10.0 ** rng.uniform(-1, 3, n)  # entries over 8 decades  # translations / rotations: very different scales
    return (sparse.diags(s) @ K @ sparse.diags(s)).tocsr()


def test_single_precision_factor_solves_to_double_accuracy(monkeypatch):
    K = _ill_scaled_spd()
    b = np.random.default_rng(2).normal(size=(K.shape[0], 3))
    x_ref = KERNEL.factor(K).solve(b)
    _no_memory_for_double(monkeypatch)
    fac = KERNEL.factor([K.copy()])
    assert fac.precision == "single"
    x = fac.solve(b)
    assert np.linalg.norm(K @ x - b) / np.linalg.norm(b) <= 1e-10
    assert np.max(np.abs(x - x_ref)) <= 1e-8 * np.max(np.abs(x_ref))
    assert fac.solve(b[:, 0]).shape == (K.shape[0],)


def test_holder_hands_the_matrix_over_and_fallback_rebuilds_it():
    K = _ill_scaled_spd(50)
    holder = [K.copy()]
    fac = KERNEL.factor(holder)
    assert holder == []
    b = np.ones(50)
    assert np.allclose(K @ fac.solve(b), b, rtol=0, atol=1e-6 * np.linalg.norm(b))
    # a mechanism falls back to SuperLU, which names it: the full matrix is rebuilt from the triangle
    M = sparse.csr_matrix(np.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 1.]]))
    with pytest.raises(Exception) as ei:
        f = KERNEL.factor([M])
        f.solve(np.array([1., 0., 0.]))
    assert "UNSTABLE" in str(getattr(ei.value, "code", "")) + str(ei.value) or KERNEL.last_fallback


def test_frame_results_identical_with_single_precision(monkeypatch):
    def model():
        m = Model("portal", "N m")
        m.add_material("S", E=200e9, nu=.3)
        m.add_frame_section(FrameSection.general("B", "S", A=.02, I33=8e-5, I22=4e-5, J=1e-5))
        a, b = m.add_point(0, 0, 0, restraints=FIXED), m.add_point(6, 0, 0, restraints=FIXED)
        c, d = m.add_point(0, 0, 4), m.add_point(6, 0, 4)
        for i, j in ((a, c), (b, d), (c, d)):
            m.add_frame(i, j, "B")
        m.add_load_pattern("P")
        m.add_point_load(c, "P", force=(1e4, 0, -5e4))
        m.add_linear_static_case("P")
        return m, c
    m, c = model()
    ref = analyze(m, kernel=KERNEL).joint_displacement(c.name, "P")
    _no_memory_for_double(monkeypatch)
    m, c = model()
    res = analyze(m, kernel=kp.PardisoKernel())
    got = res.joint_displacement(c.name, "P")
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-14)
    assert any("single precision" in i.message for i in res.issues)


def test_memory_retry_releases_the_assembled_stiffness(monkeypatch):
    """A MemoryError at the factor is retried once with the full stiffness released;
    the results are those of the normal run."""
    def model():
        m = Model("cant", "N m")
        m.add_material("S", E=200e9, nu=.3)
        m.add_frame_section(FrameSection.general("B", "S", A=.02, I33=8e-5, I22=4e-5, J=1e-5))
        a, b = m.add_point(0, 0, 0, restraints=FIXED), m.add_point(3, 0, 0)
        m.add_frame(a, b, "B")
        m.add_load_pattern("P")
        m.add_point_load(b, "P", force=(0, 0, -1e4))
        m.add_linear_static_case("P")
        return m, b
    m, b = model()
    ref = analyze(m, kernel=KERNEL).joint_displacement(b.name, "P")
    k = kp.PardisoKernel()
    real, n = k.factor, []

    def flaky(K):
        n.append(1)
        if len(n) == 1:
            raise MemoryError("test")
        return real(K)
    k.factor = flaky
    m, b = model()
    res = analyze(m, kernel=k)
    assert len(n) == 2
    assert np.allclose(res.joint_displacement(b.name, "P"), ref, rtol=1e-12, atol=0)
    assert any("released" in i.message for i in res.issues)
