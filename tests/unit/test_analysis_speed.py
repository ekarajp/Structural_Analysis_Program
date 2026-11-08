"""Analysis speed-ups keep the results: PARDISO kernel, unloaded cases, modal
reuse of the static factor, result caches, mesher broad phases."""
import numpy as np
import pytest
from scipy import sparse

from structstudio.api import analyze
from structstudio.app.templates import new_building
from structstudio.engine import kernel as K
from structstudio.engine.solvers import Engine
from structstudio.mesh import build_analysis_model
from structstudio.model import LinearStaticCase


def _building(stories=2):
    return new_building(x_spacings=(6, 5), y_spacings=(5,), story_heights=(3.0,) * stories, grid_only=False)


def _spd(n=400, seed=0):
    rng = np.random.default_rng(seed)
    A = sparse.random(n, n, density=0.01, random_state=seed) + sparse.eye(n) * 0.1
    return (A @ A.T + sparse.eye(n) * 5).tocsr(), rng.normal(size=n)


# ------------------------------------------------------------------ kernels
def test_scipy_and_default_kernel_solve_the_same():
    A, b = _spd()
    x0 = K.SciPyKernel().factor(A).solve(b)
    x1 = K.default_kernel().factor(A).solve(b)
    assert np.allclose(x0, x1, rtol=1e-9, atol=1e-12)
    assert "SuperLU" in K.SciPyKernel().describe()


def test_pardiso_falls_back_to_superlu_for_a_mechanism():
    kp = pytest.importorskip("structstudio.engine.kernel_pardiso")
    try:
        k = kp.PardisoKernel()
    except Exception:
        pytest.skip("MKL PARDISO not available")
    A, b = _spd(50)
    f = k.factor(A)
    assert isinstance(f, kp._PardisoFactor) and k.last_fallback is None
    assert np.allclose(f.solve(b), K.SciPyKernel().factor(A).solve(b), rtol=1e-9)
    # a mechanism: two DOFs joined by a spring, nothing else
    M = sparse.csr_matrix(np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 5.0]]))
    try:
        g = k.factor(M)
    except Exception:
        g = None
    assert k.last_fallback  # handed to SuperLU, which names the DOF or refuses
    if g is not None:
        assert isinstance(g, K._SciPyFactor)
        assert g.unstable == K.SciPyKernel().factor(M).unstable
    assert "SuperLU used" in K.describe(k)


def test_scipy_can_be_forced(monkeypatch):
    monkeypatch.setenv("STRUCTSTUDIO_KERNEL", "scipy")
    assert K.default_kernel().name == "scipy" and "STRUCTSTUDIO_KERNEL" in K.KERNEL_NOTE


def test_analysis_reports_the_equation_solver():
    r = analyze(_building(1))
    assert any("Equation solver:" in i.text() for i in r.issues)


# ------------------------------------------------------------ unloaded cases
def test_unloaded_case_is_exactly_zero_and_like_a_solved_one():
    m = _building()
    m.load_patterns  # a pattern with no load at all
    from structstudio.model.loads import LoadPattern
    m.load_patterns["EMPTY"] = LoadPattern("EMPTY")  # a pattern with no load at all
    m.load_cases["EMPTY"] = LinearStaticCase("EMPTY", {"EMPTY": 1.0})
    am = build_analysis_model(m)
    eng = Engine(m, am)
    fast = eng.run(["EMPTY"])["EMPTY"]
    assert fast.diagnostics.get("unloaded")
    assert not np.any(fast.displacements) and not np.any(fast.frame_end_forces)
    # the general solver gives the same (all zero) result, in the same layout
    eng2 = Engine(m, build_analysis_model(m))
    eng2._unloaded = lambda case, F: False
    slow = eng2.run(["EMPTY"])["EMPTY"]
    assert set(fast.reactions) == set(slow.reactions)
    assert all(np.allclose(fast.reactions[k], slow.reactions[k]) for k in slow.reactions)
    assert set(fast.shell_resultants) == set(slow.shell_resultants)
    for eid, r in slow.shell_resultants.items():
        for key in ("F_nodes", "M_nodes", "V_nodes", "F", "M", "V"):
            assert np.allclose(fast.shell_resultants[eid][key], r[key], atol=1e-12)
    assert fast.frame_loads == {k: [] for k in slow.frame_loads}


def test_a_loaded_case_is_not_taken_as_unloaded():
    m = _building(1)
    r = analyze(m)
    for name, c in m.load_cases.items():
        if type(c).__name__ == "LinearStaticCase" and r.cases[name].diagnostics.get("unloaded"):
            assert not np.any(r.cases[name].displacements)


# ------------------------------------------------------------------ modal
def test_modal_condensation_with_the_static_factor_equals_the_direct_one():
    from structstudio.mesh import MeshOptions
    m = new_building(x_spacings=(6, 5, 6), y_spacings=(5, 5), story_heights=(3.0,) * 2, grid_only=False)
    am = build_analysis_model(m, MeshOptions(floor_max_size=0.5))  # big enough for the sparse path
    eng = Engine(m, am)
    modal = next(n for n, c in m.load_cases.items() if type(c).__name__ == "ModalCase")
    Kf = eng._free_matrix(eng.K_reduced())
    Mf = eng._free_matrix(eng.asm.mass_reduced())
    if Kf.shape[0] <= 3000:
        pytest.skip("small model: the dense path is used")
    w_a, phi_a = eng.kernel.eigen(Kf, Mf, 6)
    w_b, phi_b = eng.kernel.eigen(Kf, Mf, 6, factor=eng.kernel.factor(Kf))
    assert np.allclose(w_a, w_b, rtol=1e-9)
    for j in range(phi_a.shape[1]):
        s = np.sign(phi_a[:, j] @ phi_b[:, j])
        assert np.allclose(phi_a[:, j], s * phi_b[:, j], rtol=1e-6, atol=1e-9 * np.abs(phi_a[:, j]).max())
    assert modal


# ------------------------------------------------------------- result caches
def test_shell_results_are_cached_read_only_and_combinations_match():
    m = _building(1)
    r = analyze(m)
    static = [n for n, c in m.load_cases.items() if type(c).__name__ == "LinearStaticCase"]
    case = static[0]
    v1, n1 = r.shell_node_values(case, "M11")
    v2, _ = r.shell_node_values(case, "M11")
    assert np.array_equal(v1, v2) and v1 is not v2
    v1[:] = 0.0  # a caller may change its copy
    assert np.array_equal(r.shell_node_values(case, "M11")[0], v2)
    env = r.shell_element_envelope(case)
    e = next(iter(env.values()))
    with pytest.raises(ValueError):
        e.max[0, 0] = 1.0  # the shared arrays cannot be changed in place
    for combo in m.load_combinations:
        ce = r.shell_element_envelope(combo)
        c = m.load_combinations[combo]
        if c.type.replace(" ", "").lower() in ("linearadd", "additive", "add") and all(
                it.name in r.cases for it in c.items):
            eid = next(iter(ce))
            expect = sum(it.scale * r.shell_element_envelope(it.name)[eid].max for it in c.items)
            assert np.allclose(ce[eid].max, expect)
    rows = r.shell_forces(case)
    assert rows and all("M11Min" not in row for row in rows)  # a single case: no Min columns


def test_frame_cut_broad_phase_finds_what_the_full_search_finds():
    from structstudio.mesh.mesher import Mesher
    from structstudio.mesh.geometry import point_on_segment
    m = _building(2)
    me = Mesher(m)
    am = me.build()
    rng = np.random.default_rng(4)
    X = am.coords()
    probes = np.vstack([X[rng.integers(0, len(X), 200)], X[rng.integers(0, len(X), 50)] + rng.normal(0, 1e-4, (50, 3))])
    for x in probes:
        want = []
        for f in m.frames.values():
            if f.auto_mesh and f.mesh_at_floor:
                a, b = me._frame_geom[f.name]
                t = point_on_segment(x, a, b, me.tol)
                if t is not None:
                    want.append((f.name, t))
        before = {k: list(v) for k, v in me._frame_cuts.items()}
        me._cut_frames_at_node(x)
        got = [(k, t) for k, v in me._frame_cuts.items() for t in v[len(before.get(k, [])):]]
        assert sorted(got) == sorted(want)
