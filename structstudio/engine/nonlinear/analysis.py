"""Static and transient nonlinear solution algorithms.

* Equilibrium iterations: ``Newton`` (tangent every iteration),
  ``ModifiedNewton`` (tangent of the step start), ``InitialNewton`` (initial
  stiffness), ``NewtonLineSearch`` (interpolated line search, ratio 0.8) -
  the OpenSees algorithms of the same names.
* Convergence tests: ``NormDispIncr``, ``NormUnbalance``, ``EnergyIncr``
  (absolute, OpenSees semantics) and ``Relative`` (unbalance relative to the
  applied load plus displacement increment relative to the step), the
  default.
* Static control: load control and displacement control (Batoz & Dhatt
  1979 / OpenSees ``DisplacementControl``).
* Transient: Newmark (gamma, beta) and HHT-alpha (OpenSees convention,
  alpha in [2/3, 1], alpha = 1 is Newmark), relative-displacement
  formulation with uniform base excitation; Rayleigh damping on the initial,
  committed or current tangent stiffness and constant modal damping
  (low-rank, solved with the Woodbury identity).
* Robustness: when a step fails the next algorithm in ``fallback`` is tried,
  then the step is subdivided (up to ``max_subdivisions`` halvings).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ...model.base import ModelError
from .domain import NonlinearDomain
from .elements import ElementNonConvergence


class StepFailed(Exception):
    pass


# ============================================================================
@dataclass
class ConvergenceTest:
    kind: str = "Relative"  # Relative | NormDispIncr | NormUnbalance | EnergyIncr
    tol: float = 1e-6
    max_iter: int = 25
    disp_tol: float = 1e-8  # Relative: |du| / |u step| (and absolute floor)

    def check(self, du: np.ndarray, r: np.ndarray, ref_force: float, ref_disp: float) -> Tuple[bool, float]:
        k = self.kind.lower()
        if k == "normdispincr":
            e = float(np.linalg.norm(du))
            return e <= self.tol, e
        if k == "normunbalance":
            e = float(np.linalg.norm(r))
            return e <= self.tol, e
        if k == "energyincr":
            e = 0.5 * abs(float(du @ r))
            return e <= self.tol, e
        # relative: unbalance vs reference force and displacement increment vs step size
        ef = float(np.linalg.norm(r)) / max(ref_force, 1e-300)
        ed = float(np.linalg.norm(du)) / max(ref_disp, 1e-300)
        return (ef <= self.tol and ed <= max(self.disp_tol, 1e-12)) or ef <= 1e-3 * self.tol, max(ef, ed)


class _Factor:
    """LU of a sparse matrix with an optional low-rank symmetric update
    K + U diag(d) U^T (Woodbury)."""

    def __init__(self, K: sparse.csr_matrix, U: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None):
        n = K.shape[0]
        if n == 0:
            self.lu = None
            return
        try:
            self.lu = splu(K.tocsc(), diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        except RuntimeError as exc:
            raise StepFailed(f"singular tangent: {exc}") from exc
        self.U, self.d = U, d
        if U is not None and U.shape[1]:
            KiU = self.lu.solve(U)
            S = np.diag(1.0 / d) + U.T @ KiU
            self.KiU = KiU
            self.S = np.linalg.inv(S)

    def solve(self, b: np.ndarray) -> np.ndarray:
        if self.lu is None:
            return np.zeros_like(b)
        x = self.lu.solve(b)
        if self.U is not None and self.U.shape[1]:
            x = x - self.KiU @ (self.S @ (self.U.T @ x))
        if not np.all(np.isfinite(x)):
            raise StepFailed("non-finite solution (singular tangent)")
        return x


ALGORITHMS = ("Newton", "AcceleratedNewton", "NewtonLineSearch", "ModifiedNewton", "InitialNewton")


class _Anderson:
    """Anderson acceleration (depth m) of a fixed-point correction x <- x + g(x).
    With g = K0^-1 r this is an accelerated modified Newton method, akin to
    OpenSees' KrylovNewton (Scott & Fenves 2010); it does not cycle at the
    stiffness jumps of piecewise linear hinge laws."""

    def __init__(self, m: int = 6):
        self.m = m
        self.X: List[np.ndarray] = []
        self.G: List[np.ndarray] = []

    def step(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.X.append(x.copy())
        self.G.append(g.copy())
        if len(self.X) > self.m + 1:
            self.X.pop(0)
            self.G.pop(0)
        if len(self.X) < 2:
            return g
        dX = np.array([self.X[k + 1] - self.X[k] for k in range(len(self.X) - 1)]).T
        dG = np.array([self.G[k + 1] - self.G[k] for k in range(len(self.G) - 1)]).T
        try:
            gam, *_ = np.linalg.lstsq(dG, g, rcond=None)
        except np.linalg.LinAlgError:
            return g
        return g - (dX + dG) @ gam


# ============================================================================
class StaticAnalysis:
    """Incremental-iterative static analysis of a :class:`NonlinearDomain`.

    ``const``: pattern factors held constant (e.g. gravity from a previous
    case), ``ref``: reference pattern factors scaled by the load factor."""

    def __init__(self, dom: NonlinearDomain, const: Dict[str, float], ref: Dict[str, float],
                 test: Optional[ConvergenceTest] = None, algorithm: str = "Newton",
                 fallback: Sequence[str] = ("AcceleratedNewton", "NewtonLineSearch", "ModifiedNewton", "InitialNewton"),
                 max_subdivisions: int = 8, ref_vector: Optional[np.ndarray] = None,
                 const_vector: Optional[np.ndarray] = None):
        self.dom = dom
        self.const_vector = const_vector  # constant extra load (free), e.g. from a previous pushover
        self.const = dict(const)
        self.ref = dict(ref)
        self.test = test or ConvergenceTest()
        self.algorithms = [algorithm] + [a for a in fallback if a != algorithm]
        self.max_sub = max_subdivisions
        self.lam = 0.0
        self.ref_vector = ref_vector  # extra reference load (free coords), e.g. modal / acceleration pushover
        self.iterations: List[int] = []
        self.log: List[str] = []

    def _factors(self, lam: float) -> Dict[str, float]:
        f = dict(self.const)
        for p, s in self.ref.items():
            f[p] = f.get(p, 0.0) + lam * s
        return f

    def _set_lambda(self, lam: float):
        self.dom.set_factors(self._factors(lam))
        F = self.dom.F.copy()
        if self.ref_vector is not None:
            F += lam * self.ref_vector
        if self.const_vector is not None:
            F += self.const_vector
        self._F = F

    def ref_load(self) -> np.ndarray:
        """Reference load vector P_ref (free) for a unit load factor."""
        self.dom.set_factors(self._factors(1.0))
        F1 = self.dom.F.copy()
        self.dom.set_factors(self._factors(0.0))
        F0 = self.dom.F.copy()
        P = F1 - F0
        if self.ref_vector is not None:
            P = P + self.ref_vector
        self._set_lambda(self.lam)
        return P

    # --------------------------------------------------------- load control
    def load_step(self, dlam: float) -> int:
        """Advance the load factor by ``dlam`` with fallbacks and subdivision."""
        return self._with_subdivision(lambda frac, alg: self._load_increment(dlam * frac, alg))

    def _tangent(self, K, algorithm):
        return self.dom.initial_stiffness() if algorithm == "InitialNewton" else K

    def _max_iter(self, algorithm):
        return self.test.max_iter if algorithm in ("Newton", "NewtonLineSearch") else 6 * self.test.max_iter

    def _load_increment(self, dlam: float, algorithm: str) -> int:
        dom = self.dom
        lam = self.lam + dlam
        self._set_lambda(lam)
        u = dom.u.copy()
        u0 = u.copy()
        fac0 = None
        du = None
        acc = _Anderson() if algorithm == "AcceleratedNewton" else None
        ref_f = max(float(np.linalg.norm(self._F)), dom.load_norm) + 1e-300
        for it in range(self._max_iter(algorithm) + 1):
            need_k = algorithm in ("Newton", "NewtonLineSearch") or fac0 is None
            R, K, _ = dom.resisting(u, tangent=need_k)
            r = self._F - R
            if du is not None:
                ok, _ = self.test.check(du, r, ref_f + float(np.linalg.norm(R)), float(np.linalg.norm(u - u0)) + 1e-300)
                if ok:
                    dom.u = u
                    self.lam = lam
                    return it
            if need_k:
                fac = _Factor(self._tangent(K, algorithm))
                fac0 = fac0 or fac
            else:
                fac = fac0
            du = fac.solve(r)
            if algorithm == "NewtonLineSearch":
                du = self._line_search(u, du, r)
            elif acc is not None:
                du = acc.step(u, du)
            u = u + du
        raise StepFailed(f"load step did not converge with {algorithm}")

    def _line_search(self, u, du, r0, ratio: float = 0.8, max_iter: int = 10):
        s0 = float(du @ r0)
        if s0 == 0:
            return du
        R, _, _ = self.dom.resisting(u + du, tangent=False)
        s = float(du @ (self._F - R))
        if abs(s) <= ratio * abs(s0):
            return du
        eta, eta_prev, s_prev = 1.0, 0.0, s0
        for _ in range(max_iter):
            if s == s_prev:
                break
            eta_new = eta - s * (eta - eta_prev) / (s - s_prev)
            eta_new = min(max(eta_new, 0.1), 10.0)
            eta_prev, s_prev = eta, s
            eta = eta_new
            R, _, _ = self.dom.resisting(u + eta * du, tangent=False)
            s = float(du @ (self._F - R))
            if abs(s) <= ratio * abs(s0):
                break
        return eta * du

    # ------------------------------------------------- displacement control
    def disp_step(self, c: np.ndarray, du_target: float) -> int:
        """Advance so that c . u increases by ``du_target`` (displacement control).

        When displacement control fails (typically a snap-back after a hinge
        loses strength), the path is followed with arc-length steps until the
        control displacement reaches the target again; every committed arc step
        is reported through ``on_substep``."""
        start = float(c @ self.dom.u)
        target = start + du_target
        u_before = self.dom.u.copy()
        try:
            inc = lambda frac, alg: self._disp_increment(c, du_target * frac, alg)
            if self.arc_length:
                n = self._with_subdivision(inc, max_sub=3, algorithms=("Newton", "AcceleratedNewton"))
            else:
                n = self._with_subdivision(inc)
            self._last_du = self.dom.u - u_before
            return n
        except ModelError as exc:
            if exc.code != "NONCONVERGENCE" or not self.arc_length:
                raise
        return self._arc_until(c, target, u_before)

    arc_length = True
    on_substep = None

    def _arc_until(self, c, target, u_before, max_steps: int = 400) -> int:
        dom = self.dom
        sign = 1.0 if target >= float(c @ u_before) else -1.0
        last = getattr(self, "_last_du", None)
        dl = float(np.linalg.norm(last)) if last is not None and np.any(last) else 1e-3
        dl0 = dl
        prev_dir = last.copy() if last is not None else None
        n_it = 0
        for _ in range(max_steps):
            if sign * (float(c @ dom.u) - target) >= 0.0:
                self.log.append("arc-length: control displacement recovered")
                return n_it
            done = False
            for shrink in range(10):
                lam0 = self.lam
                u0 = dom.u.copy()
                if shrink == 3:
                    self._freeze(True)
                    self.log.append("arc-length: softening hinges held on their loading branch")
                for alg in ("Newton", "AcceleratedNewton"):
                    try:
                        k, du = self._arc_increment(dl, prev_dir, alg)
                        n_it += k
                        done = True
                        break
                    except (StepFailed, ElementNonConvergence, np.linalg.LinAlgError) as exc:
                        dom.revert()
                        self.lam = lam0
                        self._set_lambda(lam0)
                if done:
                    break
                dl *= 0.5
            frozen = self._freeze(False)
            if not done:
                raise ModelError("NONCONVERGENCE", "Arc-length continuation failed after a strength drop")
            dom.commit()
            prev_dir = du
            self._last_du = du
            if self.on_substep is not None:
                self.on_substep()
            dl = min(dl * 1.5, 4.0 * dl0)
        raise ModelError("NONCONVERGENCE", "Arc-length continuation did not recover the control displacement")

    def _freeze(self, on: bool) -> bool:
        any_ = False
        for e in self.dom.nl_frames.values():
            for hg in getattr(e, "hinges", []):
                if getattr(hg.material, "freeze_loading", False) != on:
                    hg.material.freeze_loading = on
                    any_ = True
        return any_

    def _arc_increment(self, dl: float, prev_dir, algorithm: str):
        """Cylindrical arc-length increment |du| = dl (Crisfield 1981)."""
        dom = self.dom
        P = self.ref_load()
        u0 = dom.u.copy()
        R, K, _ = dom.resisting(u0)
        fac = _Factor(K)
        uh = fac.solve(P)
        s = 1.0
        if prev_dir is not None and float(prev_dir @ uh) < 0.0:
            s = -1.0
        dlam = s * dl / max(float(np.linalg.norm(uh)), 1e-300)
        Du = dlam * uh
        lam = self.lam + dlam
        self._set_lambda(lam)
        fac0 = fac
        acc = _Anderson() if algorithm == "AcceleratedNewton" else None
        for it in range(1, self._max_iter(algorithm) + 1):
            u = u0 + Du
            need_k = algorithm == "Newton"
            R, K, _ = dom.resisting(u, tangent=need_k)
            r = self._F - R
            ref_f = max(float(np.linalg.norm(self._F)), dom.load_norm) + float(np.linalg.norm(R)) + 1e-300
            if float(np.linalg.norm(r)) / ref_f <= self.test.tol:
                dom.u = u
                self.lam = lam
                return it, Du
            fac = _Factor(K) if need_k else fac0
            ur = fac.solve(r)
            uh = fac.solve(P)
            a = float(uh @ uh)
            w = Du + ur
            b = 2.0 * float(uh @ w)
            cq = float(w @ w) - dl * dl
            disc = b * b - 4 * a * cq
            if disc < 0.0:
                raise StepFailed("arc-length: no real root")
            r1 = (-b + np.sqrt(disc)) / (2 * a)
            r2 = (-b - np.sqrt(disc)) / (2 * a)
            c1 = w + r1 * uh
            c2 = w + r2 * uh
            dlm = r1 if float(c1 @ Du) >= float(c2 @ Du) else r2
            step_u = ur + dlm * uh
            if acc is not None:
                x = np.concatenate([Du, [lam]])
                st = acc.step(x, np.concatenate([step_u, [dlm]]))
                step_u, dlm = st[:-1], float(st[-1])
            Du = Du + step_u
            lam += dlm
            self._set_lambda(lam)
        raise StepFailed("arc-length increment did not converge")

    def _disp_increment(self, c: np.ndarray, du_target: float, algorithm: str) -> int:
        dom = self.dom
        P = self.ref_load()
        u = dom.u.copy()
        u0 = u.copy()
        lam = self.lam
        fac0 = None
        du = None
        acc = _Anderson() if algorithm == "AcceleratedNewton" else None
        for it in range(self._max_iter(algorithm) + 1):
            need_k = algorithm in ("Newton", "NewtonLineSearch") or fac0 is None
            R, K, _ = dom.resisting(u, tangent=need_k)
            r = self._F - R
            if du is not None:
                ref_f = max(float(np.linalg.norm(self._F)), dom.load_norm) + float(np.linalg.norm(R)) + 1e-300
                ok, _ = self.test.check(du, r, ref_f, float(np.linalg.norm(u - u0)) + 1e-300)
                if ok:
                    dom.u = u
                    self.lam = lam
                    return it
            if need_k:
                fac = _Factor(self._tangent(K, algorithm))
                fac0 = fac0 or fac
            else:
                fac = fac0
            uh = fac.solve(P)
            ur = fac.solve(r)
            cu = float(c @ uh)
            if abs(cu) < 1e-6 * float(np.abs(uh).max()) or abs(cu) < 1e-300:
                raise ModelError("CONTROL_DOF", "Displacement control: the pushover load hardly moves the monitored "
                                                "DOF (check the monitored point / direction against the load pattern "
                                                "or mode)")
            if du is None:
                dl = (du_target - float(c @ ur)) / cu
            else:
                dl = -float(c @ ur) / cu
            du = ur + dl * uh
            if acc is not None:
                # accelerate the bordered correction on (u, lambda); the constraint
                # c . u = target stays exact because every correction satisfies it
                x = np.concatenate([u, [lam]])
                step = acc.step(x, np.concatenate([du, [dl]]))
                du, dl = step[:-1], float(step[-1])
            u = u + du
            lam += dl
            self._set_lambda(lam)
        raise StepFailed(f"displacement step did not converge with {algorithm}")

    # ------------------------------------------------------------ robustness
    def _with_subdivision(self, increment, max_sub: Optional[int] = None,
                          algorithms: Optional[Sequence[str]] = None) -> int:
        """Try the algorithms on the full increment, then on halves, quarters ..."""
        dom = self.dom
        max_sub = self.max_sub if max_sub is None else max_sub
        algorithms = self.algorithms if algorithms is None else list(algorithms)
        frac_left = 1.0
        n_iter = 0
        level = 0
        frac = 1.0
        while frac_left > 1e-12:
            frac = min(frac, frac_left)
            ok = False
            for alg in algorithms:
                lam0 = self.lam
                try:
                    n_iter += increment(frac, alg)
                    ok = True
                    break
                except (StepFailed, ElementNonConvergence, np.linalg.LinAlgError, FloatingPointError) as exc:
                    self.log.append(f"{alg} failed ({exc}); ")
                    dom.revert()
                    self.lam = lam0
                    self._set_lambda(self.lam)
            if ok:
                dom.commit()
                frac_left -= frac
                if level > 0 and frac_left > 1e-12:
                    frac = min(frac * 2.0, frac_left)
                    level -= 1
            else:
                level += 1
                if level > max_sub:
                    raise ModelError("NONCONVERGENCE", "Nonlinear static step did not converge after "
                                                       f"{max_sub} subdivisions: " + "".join(self.log[-4:]))
                frac = frac / 2.0
        self.iterations.append(n_iter)
        return n_iter


# ============================================================================
def _function_value(times: np.ndarray, values: np.ndarray, t: float) -> float:
    """Linear interpolation of a time function, zero outside it.  A time within round-off
    of an end of the function (a time accumulated from many steps: 100 x 0.001 s gives
    0.10000000000000007) is that end: the end value is never lost to round-off."""
    t0, t1 = float(times[0]), float(times[-1])
    tol = 1e-9 * max(abs(t0), abs(t1), t1 - t0, 1e-12)
    if t < t0 - tol or t > t1 + tol:
        return 0.0
    return float(np.interp(min(max(t, t0), t1), times, values))


@dataclass
class GroundMotion:
    """Uniform base excitation: acceleration history (m/s^2) along a global
    direction (UX, UY, UZ), linearly interpolated, zero outside the record."""

    direction: str
    times: np.ndarray
    values: np.ndarray
    scale: float = 1.0

    def __call__(self, t: float) -> float:
        return _function_value(self.times, self.values, t) * self.scale


@dataclass
class TimeLoad:
    """Load pattern multiplied by a time function."""

    pattern: str
    times: np.ndarray
    values: np.ndarray
    scale: float = 1.0

    def __call__(self, t: float) -> float:
        return _function_value(self.times, self.values, t) * self.scale


@dataclass
class Damping:
    """Rayleigh (alpha M + beta K) and/or constant modal damping."""

    alpha_m: float = 0.0
    beta_k: float = 0.0
    stiffness: str = "initial"  # initial | committed | current
    modal_ratio: Optional[float] = None  # constant modal damping ratio (applied to modal_shapes)
    modal_shapes: Optional[np.ndarray] = None  # (n_free, m) mass normalised
    modal_omegas: Optional[np.ndarray] = None
    modal_ratios: Optional[np.ndarray] = None  # per mode (overrides modal_ratio)

    @classmethod
    def rayleigh_from_periods(cls, zeta1: float, T1: float, zeta2: float, T2: float, stiffness: str = "initial"):
        w1, w2 = 2 * math.pi / T1, 2 * math.pi / T2
        A = np.array([[1 / (2 * w1), w1 / 2], [1 / (2 * w2), w2 / 2]])
        a, b = np.linalg.solve(A, [zeta1, zeta2])
        return cls(float(a), float(b), stiffness)


class TransientAnalysis:
    """Direct integration (Newmark / HHT) of M a + C v + R(u) = F(t) - M r a_g(t)."""

    def __init__(self, dom: NonlinearDomain, const: Dict[str, float], ground: Sequence[GroundMotion] = (),
                 time_loads: Sequence[TimeLoad] = (), damping: Optional[Damping] = None,
                 integrator: str = "Newmark", gamma: float = 0.5, beta: float = 0.25, alpha: float = 1.0,
                 test: Optional[ConvergenceTest] = None, algorithm: str = "Newton",
                 fallback: Sequence[str] = ("AcceleratedNewton", "NewtonLineSearch", "ModifiedNewton", "InitialNewton"),
                 max_subdivisions: int = 8, const_vector: Optional[np.ndarray] = None):
        self.dom = dom
        self.const = dict(const)
        self.const_vector = const_vector
        self.ground = list(ground)
        self.time_loads = list(time_loads)
        self.damp = damping or Damping()
        integ = integrator.lower()
        if integ == "hht":
            self.alpha = alpha
            self.gamma = 1.5 - alpha
            self.beta = (2.0 - alpha) ** 2 / 4.0
        else:
            self.alpha, self.gamma, self.beta = 1.0, gamma, beta
        self.test = test or ConvergenceTest()
        self.algorithms = [algorithm] + [a for a in fallback if a != algorithm]
        self.max_sub = max_subdivisions
        n = dom.n
        self.u = dom.u.copy()
        self.v = np.zeros(n)
        self.a = np.zeros(n)
        self.t = 0.0
        self._r = {g.direction: dom.influence(g.direction) for g in self.ground}
        self._ag_prev = 0.0
        self.M = dom.M
        self._Kc = None  # committed tangent (for "committed" Rayleigh)
        self._K0 = dom.initial_stiffness()
        self._modal = None
        if self.damp.modal_shapes is not None:
            Phi = self.damp.modal_shapes
            om = self.damp.modal_omegas
            z = self.damp.modal_ratios if self.damp.modal_ratios is not None else np.full(len(om), self.damp.modal_ratio)
            MPhi = self.M @ Phi
            self._modal = (MPhi, 2.0 * z * om)
        self.log: List[str] = []
        self.iterations: List[int] = []
        self._F_prev = self._external(0.0)
        self.a = self._initial_acceleration(self._F_prev)

    def _initial_acceleration(self, F0: np.ndarray) -> np.ndarray:
        """a(0) from the equation of motion at t = 0: M a0 = F(0) - C v0 - R(u0) on the DOFs
        with mass; the massless DOFs follow the massive ones through the stiffness
        (a_z = -K_zz^-1 K_zm a_m).  Zero when the start is in equilibrium."""
        n = self.dom.n
        try:
            R, K, _ = self.dom.resisting(self.u, self.v, tangent=True)
        except Exception:  # an element that cannot give its state at rest: start from rest
            return np.zeros(n)
        fd, _ = self._damping_force_and_matrix(self.v, K if K is not None else self._K0)
        r = F0 - R - fd
        if self._modal is not None:
            MPhi, cd = self._modal
            r = r - MPhi @ (cd * (MPhi.T @ self.v))
        scale = max(float(np.linalg.norm(F0)), float(np.linalg.norm(R)), 1e-300)
        if float(np.linalg.norm(r)) <= 1e-12 * scale:
            return np.zeros(n)
        M = sparse.csr_matrix(self.M)
        row_mass = np.asarray(abs(M).sum(axis=1)).ravel()
        mm = np.where(row_mass > 0)[0]
        zz = np.where(row_mass <= 0)[0]
        a = np.zeros(n)
        if mm.size == 0:
            return a
        from scipy.sparse.linalg import spsolve, splu
        Mmm = M[mm][:, mm].tocsc()
        a[mm] = np.atleast_1d(spsolve(Mmm, r[mm]))
        if zz.size:
            Kt = sparse.csc_matrix(K if K is not None else self._K0)
            try:
                lu = splu(Kt[zz][:, zz].tocsc())
                a[zz] = -lu.solve(np.asarray(Kt[zz][:, mm] @ a[mm]).ravel())
            except RuntimeError:  # singular massless block: those DOFs start at rest
                a[zz] = 0.0
        if not np.all(np.isfinite(a)):
            return np.zeros(n)
        return a

    # ----------------------------------------------------------------- loads
    def _factors(self, t: float) -> Dict[str, float]:
        f = dict(self.const)
        for tl in self.time_loads:
            f[tl.pattern] = f.get(tl.pattern, 0.0) + tl(t)
        return f

    def _external(self, t: float) -> np.ndarray:
        self.dom.set_factors(self._factors(t))
        F = self.dom.F.copy()
        for g in self.ground:
            F -= self.M @ (self._r[g.direction] * g(t))
        if self.const_vector is not None:
            F += self.const_vector
        return F

    def ground_accel(self, t: float) -> Dict[str, float]:
        """Total ground acceleration per global direction (every record along it added)."""
        out: Dict[str, float] = {}
        for g in self.ground:
            out[g.direction] = out.get(g.direction, 0.0) + g(t)
        return out

    def _damping_force_and_matrix(self, v, K_tan):
        d = self.damp
        C = None
        f = np.zeros_like(v)
        if d.alpha_m:
            f += d.alpha_m * (self.M @ v)
            C = d.alpha_m * self.M
        if d.beta_k:
            Kd = {"initial": self._K0, "committed": self._Kc if self._Kc is not None else self._K0,
                  "current": K_tan}[d.stiffness]
            f += d.beta_k * (Kd @ v)
            C = d.beta_k * Kd if C is None else C + d.beta_k * Kd
        return f, C

    def _keff(self, K, al, c1, c2, Cel):
        """al K + c1 M + al c2 (C_rayleigh + C_elements) on the domain's fixed pattern."""
        dom = self.dom
        npat = len(dom.M_data)
        d = self.damp
        if K.nnz == npat and self._K0.nnz == npat:
            data = al * K.data + c1 * dom.M_data
            cd = al * c2
            if d.alpha_m:
                data = data + cd * d.alpha_m * dom.M_data
            if d.beta_k:
                Kd = {"initial": self._K0, "committed": self._Kc if self._Kc is not None else self._K0,
                      "current": K}[d.stiffness]
                data = data + cd * d.beta_k * Kd.data
            if Cel is not None:
                data = data + cd * Cel.data
            return dom._csr(data)
        Keff = al * K + c1 * self.M
        _, C = self._damping_force_and_matrix(np.zeros(K.shape[0]), K)
        if C is not None:
            Keff = Keff + al * c2 * C
        if Cel is not None:
            Keff = Keff + al * c2 * Cel
        return Keff

    # ------------------------------------------------------------------ step
    def step(self, dt: float, t_end: Optional[float] = None) -> int:
        """Advance by dt.  ``t_end``: the exact time of the end of the step on the case's
        time grid (k dt), so that times do not drift by round-off over many steps."""
        target = (self.t + dt) if t_end is None else float(t_end)
        frac_left = 1.0
        frac = 1.0
        level = 0
        n_it = 0
        while frac_left > 1e-12:
            frac = min(frac, frac_left)
            ok = False
            for alg in self.algorithms:
                try:
                    n_it += self._step(dt * frac, alg, target)
                    ok = True
                    break
                except (StepFailed, ElementNonConvergence, np.linalg.LinAlgError, FloatingPointError) as exc:
                    self.log.append(f"t={self.t:.4f} {alg}: {exc}")
                    self.dom.revert()
            if ok:
                frac_left -= frac
                if level > 0:
                    frac = min(2 * frac, frac_left) if frac_left > 0 else frac
                    level -= 1
            else:
                level += 1
                if level > self.max_sub:
                    raise ModelError("NONCONVERGENCE", f"Time step at t={self.t:.5f} s did not converge after "
                                                       f"{self.max_sub} subdivisions")
                frac /= 2.0
        self.iterations.append(n_it)
        return n_it

    def _step(self, dt: float, algorithm: str, target: Optional[float] = None) -> int:
        dom = self.dom
        g, b, al = self.gamma, self.beta, self.alpha
        un, vn, an = self.u, self.v, self.a
        t1 = self.t + dt
        if target is not None and abs(t1 - target) <= 1e-9 * max(abs(dt), 1e-12):
            t1 = target  # the end of the step on the time grid
        F1 = self._external(t1)
        Fn = self._F_prev if hasattr(self, "_F_prev") else self._external(self.t)
        if al != 1.0:
            F = al * F1 + (1 - al) * Fn
            self._external(t1)  # element loads at t1 (constant patterns in practice)
        else:
            F = F1
        c1 = 1.0 / (b * dt * dt)
        c2 = g / (b * dt)
        u = un + dt * vn + dt * dt * (0.5 - b) * an  # predictor consistent with a = 0 correction
        u = un.copy()
        fac0 = None
        ref_f = max(float(np.linalg.norm(F)), dom.load_norm) + float(np.linalg.norm(self.M @ an)) + 1e-300
        for it in range(1, self.test.max_iter + 1):
            a1 = c1 * (u - un) - vn / (b * dt) - (0.5 / b - 1.0) * an
            v1 = vn + dt * ((1 - g) * an + g * a1)
            ua = un + al * (u - un)
            va = vn + al * (v1 - vn)
            need_k = algorithm in ("Newton", "NewtonLineSearch") or fac0 is None
            R, K, Cel = dom.resisting(ua, va, tangent=need_k)
            fd, C = self._damping_force_and_matrix(va, K if K is not None else self._K0)
            r = F - self.M @ a1 - fd - R
            if need_k:
                if algorithm == "InitialNewton":
                    K = self._K0
                Keff = self._keff(K, al, c1, c2, Cel)
                U = d = None
                if self._modal is not None:
                    U, d = self._modal[0], al * c2 * self._modal[1]
                fac = _Factor(Keff.tocsr(), U, d)
                if fac0 is None:
                    fac0 = fac
            else:
                fac = fac0
            if self._modal is not None:
                MPhi, cd = self._modal
                r = r - MPhi @ (cd * (MPhi.T @ va))
            if it > 1:
                ok, _ = self.test.check(du, r, ref_f + float(np.linalg.norm(R)), float(np.linalg.norm(u - un)) + 1e-300)
                if ok:
                    self._accept(u, v1, a1, t1, F1, K)
                    return it
            du = fac.solve(r)
            u = u + du
        raise StepFailed(f"time step did not converge with {algorithm}")

    def _accept(self, u, v, a, t, F1, K):
        dom = self.dom
        # make sure the element state corresponds to u_{n+1}
        if self.alpha != 1.0:
            dom.resisting(u, v, tangent=False)
        dom.u = u
        dom.commit()
        self.u, self.v, self.a, self.t = u, v, a, t
        self._F_prev = F1
        if self.damp.stiffness == "committed" and self.damp.beta_k:
            _, Kc, _ = dom.resisting(u, v)
            self._Kc = Kc
