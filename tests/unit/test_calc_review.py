"""Regression tests for the OpenSees calculation review (opensees-study/20261115-calculation-review).

Each test reproduces the fixture of one finding (CORE01/02, NL01-NL04, DYN01-DYN06) and checks
the acceptance stated for it.  Reference numbers marked "OpenSees 3.8.0" were produced by the
study with openseespy 3.8.0 on the same inputs (reviews/*/probe-results.json); analytic
references are computed here."""
import math

import numpy as np
import pytest
from scipy import sparse

from structstudio import Model, analyze
from structstudio.engine.kernel import SciPyKernel
from structstudio.engine.nonlinear import materials as M
from structstudio.engine.nonlinear.analysis import ConvergenceTest, TimeLoad, TransientAnalysis
from structstudio.model import FIXED, FrameSection
from structstudio.model.base import ModelError
from structstudio.model.nonlinear import TimeHistoryLoad, integrator_contract


# ----------------------------------------------------------------- NL01 / NL03 BoucWen
BW = dict(alpha=.05, ko=1000., nexp=2., gamma=.5, beta=.5, Ao=1., deltaA=.02, deltaNu=.01, deltaEta=.01,
          tol=1e-12, maxIter=100)
HIST = (0., .03, .07, .1)


def _bw(**kw):
    mat = M.BoucWen(**{**BW, **kw})
    for e in HIST:
        mat.response(e)
        mat.commit()
    return mat


def _fd(mat, e, h):
    return (mat.copy().response(e + h)[0] - mat.copy().response(e - h)[0]) / (2 * h)


def test_nl01_degraded_boucwen_tangent_is_the_derivative_of_its_stress():
    mat = _bw()
    stress, tangent = mat.copy().response(.12)
    assert stress == pytest.approx(106.10491291736552, rel=1e-14)  # unchanged stress update
    fd = _fd(mat, .12, 1e-6)
    assert abs(tangent - fd) / abs(fd) <= 1e-6
    # OpenSees 3.8.0 BoucWen, same committed history and trial strain
    assert tangent == pytest.approx(722.8746480327683, rel=1e-12)


@pytest.mark.parametrize("e", [.12, .05, -.02, .2])
def test_nl01_tangent_matches_finite_difference_along_a_history(e):
    mat = _bw()
    _, k = mat.copy().response(e)
    fd = _fd(mat, e, 1e-6)
    assert abs(k - fd) <= 1e-6 * max(abs(fd), 1.0)


def test_nl01_nondegraded_control_unchanged():
    mat = _bw(deltaA=0., deltaNu=0., deltaEta=0.)
    s, k = mat.copy().response(.12)
    assert s == pytest.approx(119.23737048522098, rel=1e-13)
    assert k == pytest.approx(982.0584672911698, rel=1e-12)


def test_nl03_local_nonconvergence_is_a_structured_failure_and_keeps_the_committed_state():
    mat = M.BoucWen(.05, 1000, 2, .5, .5, tol=1e-12, maxIter=1)
    with pytest.raises(M.MaterialNonConvergence) as ei:
        mat.response(2.)
    exc = ei.value
    assert exc.iterations == 1 and exc.residual > 1.0 and exc.strain == 2.0
    # an element non-convergence: the global solvers revert and subdivide on it
    from structstudio.engine.nonlinear.elements import ElementNonConvergence
    assert isinstance(exc, ElementNonConvergence)
    pt = mat._pts[0]
    assert (pt.Cstrain, pt.Cz, pt.Ce, pt.Cstress) == (0.0, 0.0, 0.0, 0.0)


def test_nl03_normal_budget_recovers_the_analytic_root():
    mat = M.BoucWen(.05, 1000, 2, .5, .5, tol=1e-12, maxIter=100)
    s, _ = mat.response(2.)
    z = (-1 + math.sqrt(17.)) / 4.  # root of z = (1 - z^2) * 2
    assert s == pytest.approx(.05 * 1000 * 2 + .95 * 1000 * z, rel=1e-12)


# ----------------------------------------------------------------- NL02 transient algorithms
class _CubicDomain:
    """One DOF, R(u) = u + 100 u^3, unit mass: full Newton from rest overshoots."""

    def __init__(self, fail_above=None):
        self.n = 1
        self.u = np.zeros(1)
        self.u_c = self.u.copy()
        self.M = sparse.csr_matrix([[1.]])
        self.M_data = np.array([1., 0.])
        self.F = np.zeros(1)
        self.load_norm = 10.
        self.path = []
        self.fail_above = fail_above

    def initial_stiffness(self):
        return sparse.csr_matrix([[1.]])

    def set_factors(self, f):
        self.F = np.array([10 * f.get('P', 0)])

    def resisting(self, u, v=None, tangent=True):
        x = float(u[0])
        self.path.append(x)
        if self.fail_above is not None and abs(x - float(self.u_c[0])) > self.fail_above:
            raise M.MaterialNonConvergence("Test", x, float(self.u_c[0]), 1.0, 1, 1e-12)
        return np.array([x + 100 * x ** 3]), (sparse.csr_matrix([[1 + 300 * x * x]]) if tangent else None), None

    def commit(self):
        self.u_c = self.u.copy()

    def revert(self):
        self.u = self.u_c.copy()

    def influence(self, d):
        return np.ones(1)


def _ta(dom, alg, fallback=(), **kw):
    return TransientAnalysis(dom, {}, time_loads=[TimeLoad('P', np.array([0., 1.]), np.array([0., 1.]))],
                             algorithm=alg, fallback=fallback, test=ConvergenceTest(tol=1e-8, max_iter=25), **kw)


def _equilibrium(ta):
    x = float(ta.u[0])
    return abs(10 - (x + 100 * x ** 3) - float(ta.a[0]))


def test_nl02_newton_line_search_really_searches():
    d0, d1 = _CubicDomain(), _CubicDomain()
    n0 = _ta(d0, "Newton")._step(1., "Newton")
    ta = _ta(d1, "NewtonLineSearch")
    n1 = ta._step(1., "NewtonLineSearch")
    assert d0.path != d1.path  # no longer the Newton path under another name
    assert ta.stats["line_search_evaluations"] > 0 and ta.stats["line_search_scaled"] > 0
    assert all(0.1 <= e <= 10 and e != 1.0 for e in ta.stats["line_search_eta"])
    assert n1 <= n0
    assert _equilibrium(ta) <= 1e-9


def test_nl02_accelerated_newton_applies_anderson_corrections():
    d = _CubicDomain()
    ta = _ta(d, "AcceleratedNewton")
    ta._step(1., "AcceleratedNewton")
    assert ta.stats["anderson_corrections"] > 0 and ta.stats["anderson_max_change"] > 0
    assert _equilibrium(ta) <= 1e-8
    dm = _CubicDomain()
    with pytest.raises(Exception):  # modified Newton alone diverges on this fixture
        _ta(dm, "ModifiedNewton")._step(1., "ModifiedNewton")
    assert dm.path != d.path


def test_nl02_unknown_algorithm_is_rejected():
    with pytest.raises(ModelError) as ei:
        _ta(_CubicDomain(), "NewtonLinesearch")
    assert ei.value.code == "INVALID_ALGORITHM"


def test_nl03_local_failure_subdivides_from_the_committed_state():
    # the "material" refuses a trial more than 0.2 from the committed state: the step must
    # revert and subdivide, never accept the invalid trial
    d = _CubicDomain(fail_above=0.2)
    ta = _ta(d, "Newton")
    ta.step(1.)
    assert ta.t == pytest.approx(1.0)
    assert any("local iteration did not converge" in line for line in ta.log)
    assert float(d.u_c[0]) == pytest.approx(float(ta.u[0]))
    assert _equilibrium(ta) <= 1e-7


# ----------------------------------------------------------------- DYN01 integrator contract
def _two_oscillators(n_modes=2):
    m = Model('Two uncoupled unit mass oscillators', 'N m')
    m.use_csi_frame_defaults()
    tops = []
    for i, k in enumerate([1000., 4000.]):
        m.add_material('S' + str(i), E=k, nu=.3, type='Steel')
        m.add_frame_section(FrameSection.general('B' + str(i), 'S' + str(i), A=1., I33=.1, I22=.1, J=.1))
        a = m.add_point(0, 3 * i, 0, restraints=FIXED)
        b = m.add_point(1, 3 * i, 0, restraints=(False, True, True, True, True, True), mass=(1, 0, 0))
        m.add_frame(a, b, 'B' + str(i))
        tops.append(b)
    m.add_modal_case('M', max_modes=n_modes)
    return m, tops


def _th(settings, dt=.01, n=100):
    m, tops = _two_oscillators()
    t = np.arange(n + 1) * dt
    m.add_time_history_function('GM', t, np.sin(2 * np.pi * t))
    m.add_time_history_case('TH', [TimeHistoryLoad('Acceleration', 'UX', 'GM', 1.)], nonlinear=False, n_steps=n,
                            dt=dt, damping='None', monitor_points=[p.name for p in tops], **settings)
    return m, tops


def test_dyn01_baseline_newmark_unchanged():
    m, tops = _th({})
    res = analyze(m, ['TH'], kernel=SciPyKernel())
    _, u = res.time_history('TH', tops[0].name, 'UX')
    assert float(u[-1]) == pytest.approx(-1.0918750712903414e-05, rel=1e-12)  # = OpenSees Newmark(.5,.25)
    h = res._case('TH').history
    assert h["integrator"]["effective"] == "Newmark"


@pytest.mark.parametrize("settings, code", [
    ({'integrator': 'BogusIntegrator'}, "INVALID_INTEGRATOR"),
    ({'beta': 0.}, "INVALID_INTEGRATOR_PARAMETER"),
    ({'integrator': 'HHT', 'alpha': .1}, "INVALID_INTEGRATOR_PARAMETER"),
])
def test_dyn01_invalid_integrator_rejected_before_solve(settings, code):
    m, _ = _th(settings)
    assert any(e.code == code for e in m.validate())
    with pytest.raises(Exception) as ei:
        analyze(m, ['TH'], kernel=SciPyKernel())
    assert code in str(ei.value)


def test_dyn01_conditional_settings_warn_but_run():
    m, _ = _th({'gamma': .1, 'beta': .01})
    res = analyze(m, ['TH'], kernel=SciPyKernel())
    assert any("negative numerical damping" in i.message for i in res.issues)
    _, _, w = integrator_contract("Newmark", .5, 0., 1.)
    _, errs, _ = integrator_contract("Newmark", .5, 0., 1.)
    assert errs and errs[0][0] == "INVALID_INTEGRATOR_PARAMETER"
    eff, errs, w = integrator_contract("Newmark", .5, 1 / 6, 1.)  # linear acceleration: valid, conditional
    assert not errs and eff[0] == "Newmark" and any("conditionally stable" in x for x in w)
    eff, errs, w = integrator_contract("hht", .5, .25, .9)
    assert not errs and eff == ("HHT", pytest.approx(.6), pytest.approx(1.1 ** 2 / 4), .9)


def test_dyn01_supported_hht_still_runs():
    m, tops = _th({'integrator': 'HHT', 'alpha': .9})
    res = analyze(m, ['TH'], kernel=SciPyKernel())
    _, u = res.time_history('TH', tops[0].name, 'UX')
    assert np.all(np.isfinite(u))


# ----------------------------------------------------------------- DYN02 / DYN03 eigen kernel
def test_dyn02_negative_eigenvalue_not_hidden_as_rigid_mode():
    k = SciPyKernel()
    w2, phi = k.eigen(sparse.diags([-1., 1e12]).tocsr(), sparse.eye(2).tocsr(), 2)
    assert w2[0] == pytest.approx(-1.0) and w2[1] == pytest.approx(1e12)
    r = np.linalg.norm(sparse.diags([-1., 1e12]) @ phi[:, 0] - w2[0] * phi[:, 0])
    assert r <= 1e-9


def test_dyn02_rigid_body_round_off_still_zero():
    # free-free two masses on a stiff spring: one exact zero eigenvalue
    K = sparse.csr_matrix(np.array([[1., -1.], [-1., 1.]]) * 1e9)
    k = SciPyKernel()
    w2, phi = k.eigen(K, sparse.eye(2).tocsr(), 2)
    assert w2[0] >= 0 and abs(w2[0]) <= 1e-6
    assert w2[1] == pytest.approx(2e9)
    assert k.last_eigen_check["mass_orthonormality_error"] <= 1e-12


def test_dyn02_nonlinear_modes_reject_negative_eigenvalues():
    from structstudio.engine.nonlinear.runner import NonlinearRunner

    class _Dom:
        M = sparse.eye(2).tocsr()

        def initial_stiffness(self):
            return sparse.diags([-1., 1e12]).tocsr()

    class _Eng:
        kernel = SciPyKernel()

    r = NonlinearRunner.__new__(NonlinearRunner)
    r.engine = _Eng()
    with pytest.raises(ModelError) as ei:
        r._modes(_Dom(), 2)
    assert ei.value.code == "NEGATIVE_EIGENVALUE"


def test_dyn03_semidefinite_mass_gives_one_mode_one_shape():
    K = sparse.diags([2., 3.]).tocsr()
    Mm = sparse.csr_matrix(np.ones((2, 2)))
    w2, phi = SciPyKernel().eigen(K, Mm, 2)
    assert w2.shape == (1,) and phi.shape == (2, 1)
    assert w2[0] == pytest.approx(1.2)
    assert float(phi[:, 0] @ (Mm @ phi[:, 0])) == pytest.approx(1.0)


# ----------------------------------------------------------------- DYN04 modal mass coverage
def _rs(n_modes):
    m, tops = _two_oscillators(n_modes)
    m.add_spectrum_function('S', [.001, 10.], [1., 1.])
    m.add_response_spectrum_case('R', 'M', [('U1', 'S', 1.)], modal_combination='SRSS')
    return m, tops


def test_dyn04_incomplete_modal_mass_is_reported():
    m, tops = _rs(1)
    res = analyze(m, kernel=SciPyKernel())
    msgs = [i.message for i in res.issues]
    assert any("Modal case M" in x and "50.0%" in x for x in msgs)
    assert any("Spectrum R" in x and "50.0%" in x for x in msgs)
    assert res._case('R').mass_coverage['U1'] == pytest.approx(.5)
    assert res._case('M').modal.solver["mass_coverage"]["UX"] == pytest.approx(.5)
    assert res._case('M').modal.solver["mass_coverage"]["UY"] is None  # no mass along UY: N/A


def test_dyn04_full_basis_recovers_the_analytic_response_without_warning():
    m, tops = _rs(2)
    res = analyze(m, kernel=SciPyKernel())
    assert not any("target" in i.message for i in res.issues)
    assert float(res.joint_displacement(tops[0].name, 'R')[0]) == pytest.approx(.001, rel=1e-9)
    assert float(res.joint_displacement(tops[1].name, 'R')[0]) == pytest.approx(.00025, rel=1e-9)
    assert float(res.base_reactions('R')[0]) == pytest.approx(math.sqrt(2), rel=1e-9)


# ----------------------------------------------------------------- DYN05 time-step accuracy
def test_dyn05_dt_check_reports_time_step_error_separately_from_convergence():
    rels = []
    for dt in (.02, .005):
        n = round(1 / dt)
        m, tops = _th({'dt_check': True}, dt=dt, n=n)
        res = analyze(m, ['TH'], kernel=SciPyKernel())
        h = res._case('TH').history
        assert h["equilibrium"] == "converged at every step"
        chk = h["temporal_check"]
        rels.append(chk["relative_difference"])
        coarse = any("dt/2 differ" in i.message for i in res.issues)
        assert coarse == (chk["relative_difference"] > .01)
    assert rels[0] > .01 > rels[1]
    assert rels[0] / rels[1] == pytest.approx(16, rel=.25)  # second order: 4x dt -> 16x


def test_dyn05_default_marks_time_step_accuracy_unchecked():
    m, _ = _th({})
    h = analyze(m, ['TH'], kernel=SciPyKernel())._case('TH').history
    assert h["temporal_check"]["status"].startswith("not checked")


# ----------------------------------------------------------------- DYN06 modal options
def test_dyn06_ritz_request_reports_eigen_as_effective_method():
    m, _ = _two_oscillators()
    m.load_cases['M'].method = 'Ritz'
    res = analyze(m, kernel=SciPyKernel())
    modal = res._case('M').modal
    assert modal.method == "Eigen" and modal.requested_method == "Ritz"
    assert any("Ritz vectors are not implemented" in i.message for i in res.issues)


def test_dyn06_frequency_shift_is_a_centre_in_cyc_per_s_in_both_paths():
    f2 = math.sqrt(4000.) / (2 * math.pi)  # 10.07 Hz
    K = sparse.diags([1000., 4000., 9000.]).tocsr()
    Mm = sparse.eye(3).tocsr()
    for method in ("dense", "sparse"):
        w2, _ = SciPyKernel().eigen(K, Mm, 1, shift=0.9 * f2 + 0.1 * 15.0, method=method)
        assert w2[0] == pytest.approx(4000., rel=1e-9), method
    m, _ = _two_oscillators(1)
    m.load_cases['M'].shift = 10.0
    modal = analyze(m, kernel=SciPyKernel())._case('M').modal
    assert modal.periods[0] == pytest.approx(2 * math.pi / math.sqrt(4000.), rel=1e-9)
    assert modal.solver["shift_cyc_s"] == 10.0


def test_dyn06_invalid_modal_options_rejected():
    m, _ = _two_oscillators()
    m.load_cases['M'].method = 'Lanczos'
    m.load_cases['M'].shift = -1.0
    codes = {e.code for e in m.validate()}
    assert {"INVALID_MODAL_METHOD", "INVALID_MODAL_SHIFT"} <= codes


# ----------------------------------------------------------------- CORE01 / CORE02 (fixed before)
def test_core01_nonfinite_point_force_rejected():
    m = Model('nf', 'N m')
    m.add_material('S', E=200e9, nu=.3)
    m.add_frame_section(FrameSection.general('B', 'S', A=.02, I33=8e-5, I22=8e-5, J=1e-4))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(3, 0, 0)
    m.add_frame(a, b, 'B')
    m.add_load_pattern('P')
    m.add_point_load(b, 'P', force=(float('nan'), 0, -1000))
    m.add_linear_static_case('P')
    with pytest.raises(Exception) as ei:
        analyze(m, kernel=SciPyKernel())
    assert "NONFINITE" in str(getattr(ei.value, "code", "")) + str(ei.value)
