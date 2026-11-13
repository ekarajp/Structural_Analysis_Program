"""Run nonlinear static and time history load cases of a :class:`Model`."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from ...model.base import ModelError
from ..results import CaseResult
from ..solvers import _scale_load
from .analysis import ConvergenceTest, Damping, GroundMotion, StaticAnalysis, StepFailed, TimeLoad, TransientAnalysis
from .elements import ElementNonConvergence
from .builder import NonlinearBuilder
from .domain import NonlinearDomain

_DOF = {"UX": 0, "UY": 1, "UZ": 2, "RX": 3, "RY": 4, "RZ": 5, "U1": 0, "U2": 1, "U3": 2, "R1": 3, "R2": 4, "R3": 5}


@dataclass
class NLState:
    """End state of a nonlinear case, used by cases that continue from it."""

    dom: NonlinearDomain
    factors: Dict[str, float]  # pattern multipliers applied at the end of the case
    ref_vector: Optional[np.ndarray] = None  # extra constant load (free) carried over


def _is_corot(g) -> bool:
    return (g or "").lower().replace("-", "").replace(" ", "").replace("_", "") in (
        "corotational", "corot", "largedisplacement", "largedisplacements", "pdeltalargedisplacements")


class NonlinearRunner:
    def __init__(self, engine):
        self.engine = engine
        self.model = engine.model
        self.am = engine.am
        self.asm = engine.asm
        self.states: Dict[str, NLState] = {}
        self.builder = NonlinearBuilder(self.model, self.am, self.asm)

    # ---------------------------------------------------------------- domain
    def _new_domain(self, geometry: str) -> NonlinearDomain:
        b = NonlinearBuilder(self.model, self.am, self.asm)
        frames = b.frames()
        links = b.links()
        if _is_corot(geometry):
            # large displacements: every frame becomes an element of the domain
            from .elements import ElasticHingeFrame
            for fe in self.asm.frames:
                if fe.el.id not in frames:
                    frames[fe.el.id] = ElasticHingeFrame(fe, [])
            if self.asm.shells:
                b.warnings.append("Corotational analysis: shell elements stay small-displacement (linear geometry)")
        dom = NonlinearDomain(self.asm, frames, links, geometry=geometry)
        for w in b.warnings:
            if w not in self.am.warnings:
                self.am.warnings.append(w)
        return dom

    def _start(self, case, geometry: str) -> Tuple[NonlinearDomain, Dict[str, float], Optional[np.ndarray]]:
        if case.initial_case:
            if case.initial_case not in self.states:
                raise ModelError("CASE_ORDER", f"Case {case.name!r}: initial case {case.initial_case!r} has not been "
                                               "run or is not a nonlinear case")
            st = self.states[case.initial_case]
            dom = st.dom.snapshot()
            const = dict(st.factors)
            ref0 = None if st.ref_vector is None else st.ref_vector.copy()
        else:
            dom = self._new_domain(geometry)
            const, ref0 = {}, None
        dom.set_geometry(geometry)
        return dom, const, ref0

    # ------------------------------------------------------------ recording
    def _base_reaction(self, reactions: Dict[int, np.ndarray], u_full: Optional[np.ndarray] = None) -> np.ndarray:
        tot = np.zeros(6)
        for nid, r in reactions.items():
            x = np.asarray(self.am.nodes[nid].xyz, float)
            tot[:3] += r[:3]
            tot[3:] += r[3:] + np.cross(x, r[:3])
        if u_full is not None:
            dist = self._distributed_reaction(u_full)
            if dist is not None:
                tot += dist
        return tot

    def _distributed_reaction(self, u_full: np.ndarray) -> Optional[np.ndarray]:
        """Resultant of the distributed supports (frame foundations), global, about the origin."""
        dist = None
        for fe in self.asm.frames:
            if getattr(fe.el, "foundation", None) is None:
                continue
            fr = fe.foundation_reaction_global(u_full[self.asm._element_dofs(fe.el.nodes)])
            if fr is not None:
                dist = fr.copy() if dist is None else dist + fr
        return dist

    def _state(self, dom: NonlinearDomain):
        u = dom.u
        uf = dom.full(u)
        R = dom.resisting_full(u)
        react = dom.reactions(uf, R, dom.F_full)
        return uf, react

    def _final_result(self, case, kind: str, dom: NonlinearDomain, history: dict) -> CaseResult:
        uf, react = self._state(dom)
        ef = dom.frame_end_forces(uf)
        res = CaseResult(case.name, kind, uf.reshape(-1, 6), react, ef, dom.frame_loads_scaled(),
                         dom.shell_resultants(uf))
        res.distributed_reaction = self._distributed_reaction(uf)
        res.diagnostics["nonlinear"] = True
        res.diagnostics["end_forces_exact"] = True
        res.history = history
        return res

    # ===================================================== nonlinear static
    def nonlinear_static(self, case) -> CaseResult:
        U = self.model.units
        self._reject_prescribed(case, case.loads)
        dom, const, ref0 = self._start(case, case.geometric_nonlinearity)
        unload = (getattr(case, "hinge_unloading", None) or "Redistribute").lower()
        redistribute = unload.startswith("redis")
        dom.set_drop_mode("redistribute" if redistribute else "regularise")
        test = ConvergenceTest("Relative", case.tolerance, case.max_iterations)
        # extra reference loads (acceleration / mode)
        ref_vec = np.zeros(dom.n)
        for d, sc in case.acceleration_loads.items():
            ref_vec += -(dom.M @ dom.influence(d.upper())) * sc * self.am.gravity
        for mc, mode, sc in case.mode_loads:
            r = self.engine.results.get(mc)
            if r is None or r.modal is None:
                raise ModelError("CASE_ORDER", f"Case {case.name!r}: modal case {mc!r} must be run first")
            phi = r.modal.shapes[:, int(mode) - 1].copy()
            tr = phi.reshape(-1, 6)[:, :3]
            phi /= max(np.abs(tr).max(), 1e-300)
            ref_vec += sc * (dom.AfreeT @ (dom.M_full * phi))
        sa = StaticAnalysis(dom, const, case.loads, test=test, max_subdivisions=case.max_subdivisions,
                            ref_vector=ref_vec if np.any(ref_vec) else None, const_vector=ref0)
        sa._set_lambda(0.0)
        steps = []
        disps = []
        ctrl_c = None
        if case.monitored_point:
            nid = self.am.point_to_node.get(case.monitored_point)
            if nid is None:
                raise ModelError("UNKNOWN_POINT", f"Case {case.name!r}: monitored point {case.monitored_point!r}")
            ctrl_c = dom.free_row(nid, _DOF[case.monitored_dof.upper()])
        failed = None
        n = max(int(case.n_steps), 1)

        def record():
            uf, react = self._state(dom)
            st = {"step": len(steps) + 1, "lambda": sa.lam,
                  "control": float(ctrl_c @ dom.u) if ctrl_c is not None else None,
                  "base_reaction": self._base_reaction(react, uf), "hinges": dom.hinge_results()}
            steps.append(st)
            if case.save_steps.lower().startswith("mult"):
                disps.append(uf.reshape(-1, 6).copy())

        n_released = [0]

        def release_drops(disp_control: bool):
            """CSI-style vertical strength drops: hinges that passed a drop in the last
            step keep their strength; release it now in sub-steps (25 % to start,
            halved on failure) with the control displacement (or the load) held,
            so the structure redistributes / sheds the load."""
            if not redistribute:
                return
            dl = 0.25
            for _ in range(2000):
                pend = dom.pending_drops()
                if not pend:
                    return
                lam0 = sa.lam
                for m, pt, key, lev in pend:
                    m.set_level(pt, key, min(1.0, lev + dl))
                ok = False
                for alg in ("Newton", "AcceleratedNewton"):
                    try:
                        if disp_control:
                            sa._disp_increment(ctrl_c, 0.0, alg)
                        else:
                            sa._load_increment(0.0, alg)
                        ok = True
                        break
                    except (StepFailed, ElementNonConvergence, np.linalg.LinAlgError, FloatingPointError):
                        dom.revert()
                        sa.lam = lam0
                        sa._set_lambda(lam0)
                if ok:
                    dom.commit()
                    n_released[0] += 1
                    record()
                    dl = min(2.0 * dl, 1.0)
                else:
                    for m, pt, key, lev in pend:
                        m.set_level(pt, key, lev)
                    dl *= 0.5
                    if dl < 1.0 / 512:
                        raise ModelError("NONCONVERGENCE", "the structure cannot redistribute the load released by "
                                                           "a hinge strength drop (mechanism)")
            raise ModelError("NONCONVERGENCE", "hinge strength drops: too many redistribution sub-steps")

        try:
            if case.control.lower().startswith("disp"):
                if ctrl_c is None:
                    raise ModelError("MONITORED_POINT", f"Case {case.name!r}: displacement control needs a monitored point")
                target = case.target_displacement * U.factor("length" if case.monitored_dof.upper().startswith("U")
                                                             else "dimensionless")
                start = float(ctrl_c @ dom.u)
                du = (target - start) / n
                sa.on_substep = record  # arc-length steps through strength drops are recorded too
                for _ in range(n):
                    before = len(steps)
                    sa.disp_step(ctrl_c, du)
                    if len(steps) == before or abs(float(ctrl_c @ dom.u) - steps[-1]["control"]) > 1e-12:
                        record()
                    release_drops(True)
            else:
                for _ in range(n):
                    sa.load_step(1.0 / n)
                    record()
                    release_drops(False)
        except ModelError as exc:
            if case.stop_on_failure or not steps:
                raise
            failed = exc.message
        history = {"kind": "NonlinearStatic", "steps": steps, "control_dof": case.monitored_dof,
                   "monitored_point": case.monitored_point, "failed": failed, "log": sa.log[-20:],
                   "iterations": sa.iterations, "hinge_unloading": "Redistribute" if redistribute else "Regularised",
                   "drop_substeps": n_released[0]}
        if disps:
            history["displacements"] = np.array(disps)
        res = self._final_result(case, "NonlinearStatic", dom, history)
        if failed:
            res.warnings.append(f"Case {case.name}: stopped before the end ({failed})")
        res.diagnostics["iterations"] = int(sum(sa.iterations))
        # state for dependent cases: all loads now constant at their final value
        final = dict(const)
        for p, s in case.loads.items():
            final[p] = final.get(p, 0.0) + sa.lam * s
        extra = None
        if np.any(ref_vec) or ref0 is not None:
            extra = (ref0 if ref0 is not None else 0.0) + sa.lam * ref_vec
        self.states[case.name] = NLState(dom, final, extra)
        return res

    # ======================================================= time history
    def _reject_prescribed(self, case, patterns) -> None:
        """Ground (support) displacements are applied by the linear static solver only:
        a nonlinear / time-history case of such a pattern is refused, not run without them."""
        pats = {p for p in patterns if p}
        for node in self.am.nodes:
            for pat, dg in (node.ground_disp or {}).items():
                if pat in pats and dg is not None and np.any(np.asarray(dg) != 0):
                    raise ModelError("NOT_IMPLEMENTED",
                                     f"Case {case.name!r}: load pattern {pat!r} has a ground (support) displacement "
                                     f"at {node.point or f'node {node.id}'}; prescribed displacements are applied "
                                     "by linear static cases only - this case type cannot use them yet")

    def _ground_and_loads(self, case):
        U = self.model.units
        ground, tloads = [], []
        self._reject_prescribed(case, [ld.name for ld in case.loads if not ld.type.lower().startswith("acc")])
        for ld in case.loads:
            fn = self.model.functions.get(ld.function)
            if fn is None or not hasattr(fn, "times"):
                raise ModelError("UNKNOWN_FUNCTION", f"Case {case.name!r}: time function {ld.function!r} not found")
            t = np.asarray(fn.times, float)
            v = np.asarray(fn.values, float)
            if ld.type.lower().startswith("acc"):
                a = math.radians(ld.angle)
                s = ld.scale * U.factor("acceleration")
                name = ld.name.upper()
                if name in ("U1", "UX"):
                    comps = {"UX": math.cos(a) if name == "U1" else 1.0, "UY": math.sin(a) if name == "U1" else 0.0}
                elif name in ("U2", "UY"):
                    comps = {"UX": -math.sin(a) if name == "U2" else 0.0, "UY": math.cos(a) if name == "U2" else 1.0}
                elif name in ("U3", "UZ"):
                    comps = {"UZ": 1.0}
                else:
                    raise ModelError("TH_DIRECTION", f"Unknown acceleration direction {ld.name!r}")
                for d, c in comps.items():
                    if abs(c) > 1e-14:
                        ground.append(GroundMotion(d, t, v, s * c))
            else:
                if ld.name not in self.model.load_patterns:
                    raise ModelError("UNKNOWN_PATTERN", f"Case {case.name!r}: load pattern {ld.name!r}")
                tloads.append(TimeLoad(ld.name, t, v, ld.scale))
        return ground, tloads

    def _modes(self, dom: NonlinearDomain, n_modes: int):
        """Eigenpairs of the domain (initial stiffness), free coordinates, mass normalised."""
        K = dom.initial_stiffness()
        w2, phi = self.engine.kernel.eigen(K, dom.M, n_modes)
        return np.sqrt(np.maximum(w2, 0.0)), phi

    def _damping(self, case, dom: NonlinearDomain) -> Damping:
        kind = (case.damping or "None").lower()
        if kind == "none":
            return Damping()
        z = case.damping_ratio
        if kind == "modal":
            nm = 12
            if case.modal_case and case.modal_case in self.model.load_cases:
                nm = getattr(self.model.load_cases[case.modal_case], "max_modes", 12)
            om, phi = self._modes(dom, nm)
            keep = om > 0
            return Damping(modal_ratio=z, modal_shapes=phi[:, keep], modal_omegas=om[keep])
        if case.mass_coefficient is not None or case.stiffness_coefficient is not None:
            return Damping(case.mass_coefficient or 0.0, case.stiffness_coefficient or 0.0, case.rayleigh_stiffness)
        if case.damping_periods:
            T1, T2 = case.damping_periods
        else:
            om, phi = self._modes(dom, 12)
            om = om[om > 0]
            if len(om) < 2:
                T1 = 2 * math.pi / om[0]
                T2 = T1 / 5.0
            else:
                T = 2 * math.pi / om
                # second period: the mode where 90 % of the lateral mass participates
                rx, ry = dom.influence("UX"), dom.influence("UY")
                mx = float(rx @ (dom.M @ rx)) or 1.0
                my = float(ry @ (dom.M @ ry)) or 1.0
                cx = np.cumsum((phi.T @ (dom.M @ rx)) ** 2) / mx
                cy = np.cumsum((phi.T @ (dom.M @ ry)) ** 2) / my
                c = np.minimum(cx, cy) if my > 1.0 and mx > 1.0 else np.maximum(cx, cy)
                k = int(np.argmax(c >= 0.9)) if np.any(c >= 0.9) else len(T) - 1
                T1, T2 = T[0], T[max(k, 1)]
        return Damping.rayleigh_from_periods(z, T1, z, T2, case.rayleigh_stiffness)

    def time_history(self, case) -> CaseResult:
        meth = case.method.lower().replace(" ", "").replace("-", "")
        if meth in ("fna", "fastnonlinear", "fastnonlinearanalysis"):
            return self.fna_time_history(case)
        if meth.startswith("modal"):
            return self.modal_time_history(case)
        geom = case.geometric_nonlinearity if case.nonlinear else "None"
        dom, const, ref0 = self._start(case, geom)
        dom.set_drop_mode("regularise")  # time cannot be held: drops keep a finite slope
        if not case.nonlinear:
            # linear direct integration: elastic members, zero initial conditions (CSI linear cases)
            if case.initial_case:
                self.am.warnings.append(f"Case {case.name}: a linear time history starts from zero initial "
                                        "conditions; initial case ignored")
            dom = NonlinearDomain(self.asm, {}, NonlinearBuilder(self.model, self.am, self.asm).linear_links(),
                                  geometry="None")
            const, ref0 = {}, None
        ground, tloads = self._ground_and_loads(case)
        damp = self._damping(case, dom)
        test = ConvergenceTest("Relative", case.tolerance, case.max_iterations)
        ta = TransientAnalysis(dom, const, ground, tloads, damp, integrator=case.integrator, gamma=case.gamma,
                               beta=case.beta, alpha=case.alpha, test=test, max_subdivisions=case.max_subdivisions,
                               const_vector=ref0)
        # monitors
        mon = []
        for p in case.monitor_points:
            nid = self.am.point_to_node.get(p)
            if nid is None:
                raise ModelError("UNKNOWN_POINT", f"Case {case.name!r}: monitor point {p!r}")
            mon.append(nid)
        if not mon:
            mon = [c.master for c in self.am.constraints] if self.am.constraints else []
            if self.am.n_nodes <= 400:
                mon = sorted(set(mon) | set(range(self.am.n_nodes)))
        mon = np.array(mon, dtype=int)
        n_full_nodes = self.am.n_nodes
        times, D, V, A, BR, AG = [], [], [], [], [], []
        dmax = np.full((n_full_nodes, 6), -np.inf)
        dmin = np.full((n_full_nodes, 6), np.inf)
        fmax = fmin = None
        rmax: Dict[int, np.ndarray] = {}
        rmin: Dict[int, np.ndarray] = {}
        dt = float(case.dt)
        every = max(int(case.save_every), 1)
        failed = None

        def record(k):
            uf, react = self._state(dom)
            u6 = uf.reshape(-1, 6)
            nonlocal fmax, fmin
            np.maximum(dmax, u6, out=dmax)
            np.minimum(dmin, u6, out=dmin)
            ef = dom.frame_end_forces(uf)
            fmax = ef.copy() if fmax is None else np.maximum(fmax, ef)
            fmin = ef.copy() if fmin is None else np.minimum(fmin, ef)
            for nid, r in react.items():
                rmax[nid] = np.maximum(rmax.get(nid, r), r)
                rmin[nid] = np.minimum(rmin.get(nid, r), r)
            if k % every == 0:
                times.append(ta.t)
                D.append(u6[mon].copy())
                V.append(dom.full(ta.v).reshape(-1, 6)[mon].copy())
                A.append(dom.full(ta.a).reshape(-1, 6)[mon].copy())
                BR.append(self._base_reaction(react, uf))
                AG.append([ta.ground_accel(ta.t).get(d, 0.0) for d in ("UX", "UY", "UZ")])

        record(0)
        for k in range(1, int(case.n_steps) + 1):
            try:
                ta.step(dt, k * dt)
            except ModelError as exc:
                failed = exc.message
                break
            record(k)
        hist = {"kind": "TimeHistory", "time": np.array(times), "monitor_nodes": mon.tolist(),
                "displacements": np.array(D), "velocities": np.array(V), "accelerations": np.array(A),
                "ground_acceleration": np.array(AG), "base_reaction": np.array(BR),
                "envelope": {"disp_max": dmax, "disp_min": dmin, "frame_max": fmax, "frame_min": fmin,
                             "react_max": rmax, "react_min": rmin},
                "hinges": dom.hinge_results(), "failed": failed, "log": ta.log[-20:],
                "iterations": ta.iterations,
                "damping": {"alpha_m": damp.alpha_m, "beta_k": damp.beta_k, "stiffness": damp.stiffness,
                            "modal_ratio": damp.modal_ratio}}
        kind = "NonlinearTimeHistory" if case.nonlinear else "LinearTimeHistory"
        res = self._final_result(case, kind, dom, hist)
        if failed:
            res.warnings.append(f"Case {case.name}: stopped at t = {ta.t:.4f} s ({failed})")
        final = dict(const)
        for tl in tloads:
            final[tl.pattern] = final.get(tl.pattern, 0.0) + tl(ta.t)
        self.states[case.name] = NLState(dom, final, ref0)
        return res

    # ------------------------------------------------- FNA (Wilson 2002)
    def _fna_basis(self, dom: NonlinearDomain, K0, n_modes: int, ground):
        """Load-dependent basis: eigenvectors of the initial stiffness plus the static
        response to unit deformations of every nonlinear link DOF and to the ground
        inertia loads, orthonormalised by a Rayleigh-Ritz solution (CSI recommends
        Ritz vectors started from the link deformation loads)."""
        from scipy.sparse.linalg import splu
        om, phi = self._modes(dom, n_modes)
        cols = [phi]
        lu = None
        try:
            lu = splu(sparse.csc_matrix(K0))
        except RuntimeError:
            lu = None
        if lu is not None:
            loads = []
            for e in dom.links:
                for i in e.dofs_used:
                    f = np.zeros(dom.n)
                    np.add.at(f, e._cols, e._E.T @ e.Tg[i])
                    if np.any(f):
                        loads.append(f)
            for g in ground:
                loads.append(dom.M @ dom.influence(g.direction))
            if loads:
                X = lu.solve(np.array(loads).T)
                X = X if X.ndim == 2 else X[:, None]
                # Gram-Schmidt in the mass metric (twice): M-orthonormal to the eigenvectors and to
                # each other; vectors without mass of their own (inside the span) are dropped
                basis = phi
                Mb = dom.M @ basis
                for j in range(X.shape[1]):
                    x = X[:, j]
                    x0 = float(np.sqrt(max(x @ (dom.M @ x), 0.0)))
                    if x0 <= 0.0:
                        continue
                    for _ in range(2):
                        x = x - basis @ (Mb.T @ x)
                    nx = float(np.sqrt(max(x @ (dom.M @ x), 0.0)))
                    if nx > 1e-6 * x0:
                        x = x / nx
                        basis = np.column_stack([basis, x])
                        Mb = np.column_stack([Mb, dom.M @ x])
                if basis.shape[1] > phi.shape[1]:
                    cols.append(basis[:, phi.shape[1]:])
        Psi = np.hstack(cols)  # M-orthonormal
        Kr = Psi.T @ (K0 @ Psi)
        w2, Y = np.linalg.eigh(0.5 * (Kr + Kr.T))
        Phi = Psi @ Y
        return np.sqrt(np.maximum(w2, 0.0)), Phi, int(phi.shape[1])

    def fna_time_history(self, case) -> CaseResult:
        """Fast nonlinear analysis (Wilson; CSI "FNA"): the structure is linear
        except for the links; the equations are projected on a load-dependent
        basis, each modal equation is integrated exactly for piecewise linear
        loads, and the link forces are found at every step by Newton iteration
        on the modal link forces (Jacobian I + Phi^T (Kt - K0) Phi D)."""
        from .analysis import Damping, TransientAnalysis
        if case.initial_case:
            st = self.states.get(case.initial_case)
            if st is None or not getattr(st, "fna", False):
                raise ModelError("CASE_ORDER", f"Case {case.name!r}: an FNA case can only continue from another FNA "
                                               "case (apply gravity as a ramped load in the FNA case)")
        b = NonlinearBuilder(self.model, self.am, self.asm)
        links = b.links()
        if case.initial_case:
            dom = self.states[case.initial_case].dom.snapshot()
        else:
            dom = NonlinearDomain(self.asm, {}, links, geometry="None")
        if any(getattr(f, "hinges", None) or getattr(f, "nonlinear", None) for f in self.model.frames.values()):
            self.am.warnings.append(f"Case {case.name}: FNA treats frame hinges and fiber members as linear "
                                    "(only links are nonlinear)")
        if not dom.links:
            self.am.warnings.append(f"Case {case.name}: FNA without nonlinear links is a linear modal time history")
        ground, tloads = self._ground_and_loads(case)
        const = dict(self.states[case.initial_case].factors) if case.initial_case else {}
        ta = TransientAnalysis(dom, const, ground, tloads, Damping())
        K0 = dom.initial_stiffness()
        nm = 12
        if case.modal_case and case.modal_case in self.model.load_cases:
            nm = int(getattr(self.model.load_cases[case.modal_case], "max_modes", 12))
        nm = max(1, min(nm, dom.n - 1))
        om, Phi, n_eig = self._fna_basis(dom, K0, nm, ground)
        m = Phi.shape[1]
        kind = (case.damping or "None").lower()
        if kind == "none":
            z = np.zeros(m)
        elif kind == "modal":
            z = np.full(m, case.damping_ratio)
        else:
            if case.mass_coefficient is not None or case.stiffness_coefficient is not None:
                a0, a1 = case.mass_coefficient or 0.0, case.stiffness_coefficient or 0.0
            else:
                pos = om[om > 1e-9]
                T = 2 * np.pi / pos if len(pos) else np.array([1.0])
                T1, T2 = case.damping_periods or (T[0], T[min(2, len(T) - 1)])
                d = Damping.rayleigh_from_periods(case.damping_ratio, T1, case.damping_ratio, T2)
                a0, a1 = d.alpha_m, d.beta_k
            z = a0 / (2 * np.maximum(om, 1e-12)) + a1 * om / 2
        _check_modal_damping(case.name, z)
        dt = float(case.dt)
        # exact for under-, critically and over-damped modes (no cap on the damping)
        cA, cB, cC, cD, cAp, cBp, cCp, cDp = _modal_step_coefficients(om, z, dt)
        rigid = om <= 1e-9
        tol, max_it = float(case.tolerance), max(int(case.max_iterations), 4) * 2

        def link_force(u, v, tangent=True):
            R, K, C = dom.resisting(u, v, tangent=tangent)
            return R - K0 @ u, K, C

        u = dom.u.copy()
        q = Phi.T @ (dom.M @ u)
        qv = Phi.T @ (dom.M @ (getattr(dom, "v_fna", np.zeros(dom.n))))
        rN, _, _ = link_force(u, Phi @ qv, tangent=False)
        pN = Phi.T @ rN
        F0 = Phi.T @ ta._external(0.0)
        ta.t = 0.0
        ta.v = Phi @ qv
        ta.a = Phi @ (F0 - pN - 2 * z * om * qv - om * om * q)
        dom.u = u
        dom.commit()
        mon = []
        for pnt in case.monitor_points:
            nid = self.am.point_to_node.get(pnt)
            if nid is None:
                raise ModelError("UNKNOWN_POINT", f"Case {case.name!r}: monitor point {pnt!r}")
            mon.append(nid)
        if not mon:
            mon = [c.master for c in self.am.constraints] if self.am.constraints else []
            if self.am.n_nodes <= 400:
                mon = sorted(set(mon) | set(range(self.am.n_nodes)))
        mon = np.array(mon, dtype=int)
        times, D, V, A, BR, AG, LK = [], [], [], [], [], [], []
        dmax = np.full((self.am.n_nodes, 6), -np.inf)
        dmin = np.full((self.am.n_nodes, 6), np.inf)
        env = {"frame_max": None, "frame_min": None}
        rmax: Dict[int, np.ndarray] = {}
        rmin: Dict[int, np.ndarray] = {}
        every = max(int(case.save_every), 1)
        iters = []
        failed = None

        def record(k):
            uf, react = self._state(dom)
            u6 = uf.reshape(-1, 6)
            np.maximum(dmax, u6, out=dmax)
            np.minimum(dmin, u6, out=dmin)
            ef = dom.frame_end_forces(uf)
            env["frame_max"] = ef.copy() if env["frame_max"] is None else np.maximum(env["frame_max"], ef)
            env["frame_min"] = ef.copy() if env["frame_min"] is None else np.minimum(env["frame_min"], ef)
            for nid, r in react.items():
                rmax[nid] = np.maximum(rmax.get(nid, r), r)
                rmin[nid] = np.minimum(rmin.get(nid, r), r)
            if k % every == 0:
                times.append(ta.t)
                D.append(u6[mon].copy())
                V.append(dom.full(ta.v).reshape(-1, 6)[mon].copy())
                A.append(dom.full(ta.a).reshape(-1, 6)[mon].copy())
                BR.append(self._base_reaction(react, uf))
                AG.append([ta.ground_accel(ta.t).get(d, 0.0) for d in ("UX", "UY", "UZ")])
                LK.append([(lk.d.copy(), lk.f.copy()) for lk in dom.links])

        record(0)
        Pk = F0 - pN
        for k in range(1, int(case.n_steps) + 1):
            t1 = k * dt
            F1 = Phi.T @ ta._external(t1)
            q_r = cA * q + cB * qv + cC * Pk + cD * F1
            qv_r = cAp * q + cBp * qv + cCp * Pk + cDp * F1
            p_hat = pN.copy()
            ok = False
            scale = float(np.abs(F1).max() + np.abs(pN).max()) + 1e-300
            for it in range(1, max_it + 1):
                qn = np.where(rigid, 0.0, q_r - cD * p_hat)
                qvn = np.where(rigid, 0.0, qv_r - cDp * p_hat)
                un, vn = Phi @ qn, Phi @ qvn
                rN, Kt, Ct = link_force(un, vn)
                g = p_hat - Phi.T @ rN
                if np.abs(g).max() <= tol * max(scale, float(np.abs(p_hat).max())) and it > 1:
                    ok = True
                    break
                dK = Kt - K0
                J = np.eye(m) + (Phi.T @ (dK @ Phi)) * cD[None, :]
                if Ct is not None:
                    J += (Phi.T @ (Ct @ Phi)) * cDp[None, :]
                try:
                    p_hat = p_hat - np.linalg.solve(J, g)
                except np.linalg.LinAlgError:
                    p_hat = Phi.T @ rN
            iters.append(it)
            if not ok:
                failed = f"FNA iteration did not converge at t = {t1:.4f} s"
                break
            q, qv, pN = qn, qvn, Phi.T @ rN
            Pk = F1 - pN
            dom.u = un
            dom.v_fna = vn
            dom.commit()
            ta.t = t1
            ta.v = vn
            ta.a = Phi @ np.where(rigid, 0.0, Pk - 2 * z * om * qv - om * om * q)
            record(k)
        hist = {"kind": "TimeHistory", "time": np.array(times), "monitor_nodes": mon.tolist(),
                "displacements": np.array(D), "velocities": np.array(V), "accelerations": np.array(A),
                "ground_acceleration": np.array(AG), "base_reaction": np.array(BR),
                "envelope": {"disp_max": dmax, "disp_min": dmin, "frame_max": env["frame_max"],
                             "frame_min": env["frame_min"], "react_max": rmax, "react_min": rmin},
                "hinges": [], "failed": failed, "log": [], "iterations": iters,
                "fna": {"n_vectors": m, "n_eigen": n_eig, "periods": [float(2 * np.pi / x) if x > 1e-9 else None
                                                                       for x in om], "damping_ratios": z.tolist()},
                "links": {"names": [lk.name for lk in dom.links],
                          "deformation": np.array([[d for d, _ in row] for row in LK]) if dom.links else None,
                          "force": np.array([[f for _, f in row] for row in LK]) if dom.links else None},
                "damping": {"modal_ratios": z.tolist()}}
        res = self._final_result(case, "NonlinearTimeHistory", dom, hist)
        res.diagnostics["method"] = "FNA"
        if failed:
            res.warnings.append(f"Case {case.name}: stopped ({failed})")
        final = dict(const)
        for tl in tloads:
            final[tl.pattern] = final.get(tl.pattern, 0.0) + tl(ta.t)
        st = NLState(dom, final, None)
        st.fna = True
        self.states[case.name] = st
        return res

    # ------------------------------------------------------- modal (linear)
    def modal_time_history(self, case) -> CaseResult:
        eng = self.engine
        if not case.modal_case or case.modal_case not in eng.results:
            raise ModelError("CASE_ORDER", f"Case {case.name!r}: modal case {case.modal_case!r} must be run first")
        modal = eng.results[case.modal_case].modal
        phi = modal.shapes  # (n_full, m) mass normalised
        om = modal.omega
        m = phi.shape[1]
        asm = self.asm
        M = eng._case_mass(case.modal_case)
        ground, tloads = self._ground_and_loads(case)
        n = int(case.n_steps)
        dt = float(case.dt)
        t = np.arange(n + 1) * dt
        # modal loads p_n(t) = phi_n^T F(t)
        P = np.zeros((n + 1, m))
        for g in ground:
            r = asm.influence_vector_full(g.direction)
            r[asm.dofs.restrained_full] = 0.0
            gam = phi.T @ (M * r)
            ag = np.array([g(tt) for tt in t])
            P -= np.outer(ag, gam)
        for tl in tloads:
            f = phi.T @ asm.load_vector_full(tl.pattern)
            P += np.outer(np.array([tl(tt) for tt in t]), f)
        # damping per mode
        kind = (case.damping or "None").lower()
        if kind == "modal" or kind == "none":
            z = np.full(m, case.damping_ratio if kind == "modal" else 0.0)
        else:
            if case.mass_coefficient is not None or case.stiffness_coefficient is not None:
                a0, a1 = case.mass_coefficient or 0.0, case.stiffness_coefficient or 0.0
            else:
                T1, T2 = case.damping_periods or (modal.periods[0], modal.periods[min(2, m - 1)])
                d = Damping.rayleigh_from_periods(case.damping_ratio, T1, case.damping_ratio, T2)
                a0, a1 = d.alpha_m, d.beta_k
            z = a0 / (2 * np.maximum(om, 1e-12)) + a1 * om / 2
        _check_modal_damping(case.name, z)
        Q = _piecewise_exact(om, z, P, dt)  # exact for any damping ratio >= 0
        # outputs
        U = Q @ phi.T  # (n+1, n_full)
        n_nodes = self.am.n_nodes
        mon = [self.am.point_to_node[p] for p in case.monitor_points if p in self.am.point_to_node]
        if not mon:
            mon = sorted(set([c.master for c in self.am.constraints]) | (set(range(n_nodes)) if n_nodes <= 400 else set()))
        mon = np.array(mon, dtype=int)
        U6 = U.reshape(n + 1, n_nodes, 6)
        # per-mode frame end forces and reactions, superposed
        K_full = asm.stiffness_full()
        ef_m = np.zeros((m, len(self.am.frames), 12))
        rm: Dict[int, np.ndarray] = {}
        for k in range(m):
            react, ef, _, _ = eng._recover(phi[:, k], {}, np.zeros(asm.dofs.n_full), K_full)
            ef_m[k] = ef
            for nid, r in react.items():
                rm.setdefault(nid, np.zeros((m, 6)))[k] = r
        EF = np.einsum("tm,mfk->tfk", Q, ef_m)
        RE = {nid: Q @ arr for nid, arr in rm.items()}
        # the loads of the patterns: fixed-end actions of the loaded members and the static
        # response of the DOFs without mass (which the modes follow only through the
        # stiffness: u_z = K_zz^-1 (F_z - K_zm u_m)); both at the instantaneous load factor
        final_loads: Dict[int, list] = {}
        if tloads:
            stiff = getattr(self.model.load_cases.get(case.modal_case), "stiffness_from", None)
            K_red = eng._pdelta_K[stiff] if stiff else eng.K_reduced()
            dofs = asm.dofs
            Kf = eng._free_matrix(K_red)
            T = dofs.T
            Mf = eng._free_matrix((T.T @ sparse.diags(M) @ T).tocsr())
            zz = np.where(np.asarray(abs(Mf).sum(axis=1)).ravel() <= 0)[0]
            lu_zz = None
            if zz.size:
                from scipy.sparse.linalg import splu
                try:
                    lu_zz = splu(Kf[zz][:, zz].tocsc())
                except RuntimeError:
                    lu_zz = None
            for tl in tloads:
                F_full = asm.load_vector_full(tl.pattern)
                u_c = np.zeros(dofs.n_full)
                if lu_zz is not None:
                    F_free = (T.T @ F_full)[dofs.free]
                    u_red = np.zeros(T.shape[1])
                    sol = np.zeros(len(dofs.free))
                    sol[zz] = lu_zz.solve(F_free[zz])
                    u_red[dofs.free] = sol
                    u_c = T @ u_red
                react_p, ef_p, el_p, _ = eng._recover(u_c, {tl.pattern: 1.0}, F_full, K_full)
                fac = np.array([tl(tt) for tt in t])
                for eid, lds in el_p.items():  # span loads of the final step (station forces)
                    if lds and fac[-1]:
                        final_loads.setdefault(eid, []).extend(_scale_load(ld, float(fac[-1])) for ld in lds)
                if np.any(u_c):
                    U = U + np.outer(fac, u_c)
                EF = EF + fac[:, None, None] * ef_p[None]
                for nid, rv in react_p.items():
                    RE[nid] = RE.get(nid, np.zeros((n + 1, 6))) + np.outer(fac, rv)
            U6 = U.reshape(n + 1, n_nodes, 6)
        BR = np.zeros((n + 1, 6))
        for nid, r in RE.items():
            x = np.asarray(self.am.nodes[nid].xyz, float)
            BR[:, :3] += r[:, :3]
            BR[:, 3:] += r[:, 3:] + np.cross(x, r[:, :3])
        hist = {"kind": "TimeHistory", "time": t, "monitor_nodes": mon.tolist(), "displacements": U6[:, mon],
                "base_reaction": BR, "modal_coordinates": Q,
                "ground_acceleration": np.array([[sum(g(tt) for g in ground if g.direction == d) for d in ("UX", "UY", "UZ")]
                                                 for tt in t]),
                "envelope": {"disp_max": U6.max(0), "disp_min": U6.min(0), "frame_max": EF.max(0),
                             "frame_min": EF.min(0), "react_max": {k: v.max(0) for k, v in RE.items()},
                             "react_min": {k: v.min(0) for k, v in RE.items()}},
                "modal_damping": z}
        react_end = {nid: r[-1] for nid, r in RE.items()}
        res = CaseResult(case.name, "LinearModalTimeHistory", U6[-1], react_end, EF[-1], final_loads, {})
        res.diagnostics["end_forces_exact"] = True
        res.history = hist
        return res


def _check_modal_damping(name: str, z: np.ndarray) -> None:
    if np.any(~np.isfinite(z)) or np.any(z < 0):
        raise ModelError("INVALID_DAMPING", f"Case {name!r}: modal damping ratios must be finite and >= 0 "
                                            f"(got min {float(np.nanmin(z)):.4g})")


def _modal_step_coefficients(omega: np.ndarray, zeta: np.ndarray, dt: float):
    """Exact one-step coefficients of unit-mass oscillators q'' + 2 zeta w q' + w^2 q = p(t)
    for a load linear over the step, any zeta >= 0 (under-, critically and over-damped):
    matrix exponential of the augmented state [q, q', p0, s].  Returns
    (A, B, C, D, A', B', C', D'): q1 = A q0 + B v0 + C p0 + D p1, v1 = A' q0 + B' v0 + C' p0 + D' p1."""
    from scipy.linalg import expm
    m = len(omega)
    out = np.zeros((8, m))
    for k in range(m):
        w = float(max(omega[k], 0.0))
        z = float(zeta[k])
        Mx = np.zeros((4, 4))
        Mx[0, 1] = 1.0
        Mx[1, 0] = -w * w
        Mx[1, 1] = -2.0 * z * w
        Mx[1, 2] = 1.0
        Mx[2, 3] = 1.0
        E = expm(Mx * dt)
        G1, G2 = E[:2, 2], E[:2, 3] / dt
        c0, c1 = G1 - G2, G2
        out[:, k] = [E[0, 0], E[0, 1], c0[0], c1[0], E[1, 0], E[1, 1], c0[1], c1[1]]
    return tuple(out)


def _piecewise_exact(omega: np.ndarray, zeta: np.ndarray, P: np.ndarray, dt: float) -> np.ndarray:
    """Exact response of unit-mass modal oscillators to piecewise linear loads
    (Nigam & Jennings 1969; Chopra, Dynamics of Structures, Table 5.2.1)."""
    nt, m = P.shape
    Q = np.zeros((nt, m))
    qv = np.zeros(m)
    q = np.zeros(m)
    A, B, C, D, Ap, Bp, Cp, Dp = _modal_step_coefficients(omega, zeta, dt)
    rigid = omega <= 1e-9
    for i in range(nt - 1):
        q_new = A * q + B * qv + C * P[i] + D * P[i + 1]
        qv = Ap * q + Bp * qv + Cp * P[i] + Dp * P[i + 1]
        q = np.where(rigid, 0.0, q_new)
        qv = np.where(rigid, 0.0, qv)
        Q[i + 1] = q
    return Q
