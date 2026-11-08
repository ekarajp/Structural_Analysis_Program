"""The batched shell kernel equals the per-element ShellFormulation (round-off)."""
import numpy as np
import pytest

from structstudio.engine.elements import shell as S
from structstudio.engine.elements import shell_batch as SB
from structstudio.mesh.analysis_model import ShellElement

TYPES = ["ShellThin", "ShellThick", "Membrane", "Plate"]


def _elements(n=60, seed=3):
    rng = np.random.default_rng(seed)
    out, coords = [], []
    for k in range(n):
        a, b = rng.uniform(0.3, 2.5, 2)
        P = np.array([[0, 0], [a, 0], [a, b], [0, b]], float) + rng.uniform(-0.12, 0.12, (4, 2)) * min(a, b)
        # random plane: rotate the square into 3D (floors, walls, inclined)
        th, ph, ps = rng.uniform(-np.pi, np.pi, 3)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
        X = (Rz @ Rx @ np.column_stack([P, np.zeros(4)]).T).T + rng.uniform(-5, 5, 3)
        if k % 5 == 0:
            X = np.column_stack([P, np.full(4, 3.2)])  # flat floor
        mods = np.ones(8) if k % 3 else rng.uniform(0.2, 1.0, 8)
        t = rng.uniform(0.12, 0.6)
        axes = None
        if k % 4 == 1:  # assigned local axes
            n_ = np.cross(X[2] - X[0], X[3] - X[1])
            n_ /= np.linalg.norm(n_)
            v = np.cross(n_, [0.3, 0.9, 0.1])
            axes = np.vstack([v / np.linalg.norm(v), np.cross(n_, v / np.linalg.norm(v)), n_])
        el = ShellElement(k, "A", (4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3), 2.5e10, 0.2, 2.5e10 / 2.4, t,
                          t * rng.uniform(0.7, 1.0), TYPES[k % 4], 240.0 * t, 2400.0 * t, mods, axes)
        out.append(S.ShellFormulation(el, X))
        coords.append(X)
    return out


def test_supported_with_default_switches():
    assert SB.supported()


def test_batched_stiffness_equals_element_stiffness():
    shells = _elements()
    ref = [se.stiffness_global() for se in _elements()]
    got = []
    for d, K in SB.global_stiffness(shells):
        got.extend(K)
    for se, a, b in zip(shells, ref, got):
        scale = np.max(np.abs(a))
        assert np.max(np.abs(a - b)) <= 1e-11 * scale, (se.el.model_type, se.el.id)
    fresh = _elements()
    for se, f in zip(shells, fresh):
        f.stiffness_local()
        assert se.area == pytest.approx(f.area, rel=1e-13)
        assert np.allclose(se.tributary, f.tributary, rtol=1e-13, atol=0)


def test_batched_resultants_equal_element_resultants():
    shells = _elements()
    rng = np.random.default_rng(9)
    u = rng.normal(size=6 * 4 * len(shells)) * 1e-3
    got = SB.resultants(shells, u)
    for se in shells:
        d = SB.element_dofs([se])[0]
        r = se.resultants(u[d])
        g = got[se.el.id]
        for key in ("F", "M", "V", "F_nodes", "M_nodes", "V_nodes"):
            scale = max(np.max(np.abs(r[key])), 1e-12)
            assert np.max(np.abs(r[key] - g[key])) <= 1e-9 * scale, (key, se.el.model_type)
        assert [tuple(map(float, p)) for p in r["gauss"]] == list(g["gauss"])


def test_loads_and_mass_need_no_element_stiffness():
    shells = _elements(12)
    SB.prepare(shells)
    for se in shells:
        assert se._K is None and se._prepared
        se.lumped_mass()
        assert se._K is None


def test_non_positive_jacobian_is_reported_like_the_element():
    X = np.array([[0, 0, 0], [1, 0, 0], [0.2, 1, 0], [0.4, 0.3, 0]], float)  # re-entrant corner
    el = ShellElement(7, "A", (0, 1, 2, 3), 2e10, 0.2, 8e9, 0.2, 0.2, "ShellThin", 1, 1, np.ones(8))
    with pytest.raises(ValueError, match="shell element 7: non-positive Jacobian"):
        S.ShellFormulation(el, X).stiffness_local()
    with pytest.raises(ValueError, match="shell element 7: non-positive Jacobian"):
        list(SB.global_stiffness([S.ShellFormulation(el, X)]))
