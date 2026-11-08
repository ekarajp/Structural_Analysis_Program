"""Load case solvers: linear static, modal, response spectrum, P-Delta."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from ..mesh.analysis_model import AnalysisModel, ElementLoad
from ..model import Model, ModelError
from ..model.loads import (LinearStaticCase, ModalCase, PDeltaCase, ResponseSpectrumCase,
                           ResponseSpectrumFunction)
from ..model.nonlinear import NonlinearStaticCase, TimeHistoryCase
from .assembly import Assembler, node_transform
from .kernel import Kernel, default_kernel
from .results import CaseResult, ModalData


class Engine:
    """Runs load cases of a :class:`Model` on its :class:`AnalysisModel`."""

    def __init__(self, model: Model, am: AnalysisModel, kernel: Optional[Kernel] = None):
        self.model = model
        self.am = am
        self.kernel = kernel or default_kernel()
        self.asm = Assembler(am)
        self._K_red: Optional[sparse.csr_matrix] = None
        self._factor = None
        self._pdelta_K: Dict[str, sparse.csr_matrix] = {}
        self.results: Dict[str, CaseResult] = {}

    @property
    def nonlinear(self):
        """Runner of nonlinear static / time history cases (created on first use)."""
        if getattr(self, "_nl_runner", None) is None:
            from .nonlinear.runner import NonlinearRunner
            self._nl_runner = NonlinearRunner(self)
        return self._nl_runner

    # ---------------------------------------------------------------- matrices
    def K_reduced(self) -> sparse.csr_matrix:
        if self._K_red is None:
            self._K_red = self.asm.stiffness_reduced()
        return self._K_red

    def _free_matrix(self, K: sparse.csr_matrix) -> sparse.csr_matrix:
        f = self.asm.dofs.free
        return K[f][:, f].tocsr()

    def _factor_for(self, stiffness_from: Optional[str]):
        if stiffness_from:
            if stiffness_from not in self._pdelta_K:
                raise ModelError("CASE_ORDER", f"Stiffness case {stiffness_from!r} has not been run yet")
            return self.kernel.factor(self._free_matrix(self._pdelta_K[stiffness_from]))
        if self._factor is None:
            self._factor = self.kernel.factor(self._free_matrix(self.K_reduced()))
            self._report_instability(self._factor)
        return self._factor

    def _report_instability(self, fac):
        """Warn (ETABS / SAP2000 style) about equations that are numerically singular."""
        bad = getattr(fac, "unstable", None)
        if not bad:
            return
        dofs = self.asm.dofs
        keep = np.where(dofs.reduced_of_full >= 0)[0]
        labels = []
        for j in bad[:12]:
            red = int(dofs.free[j])
            full = int(keep[red]) if red < len(keep) else -1
            node, comp = divmod(full, 6)
            name = self.am.nodes[node].point if 0 <= node < len(self.am.nodes) else None
            labels.append(f"{name or f'node {node}'} {('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ')[comp]}")
        more = f" (+{len(bad) - 12} more)" if len(bad) > 12 else ""
        msg = ("The structure is unstable or ill-conditioned: no stiffness for " + ", ".join(labels) + more
               + ". Check supports, releases and connectivity.")
        if msg not in self.am.warnings:
            self.am.warnings.append(msg)

    # ------------------------------------------------------------------ run
    def run(self, cases: Optional[List[str]] = None, progress=None) -> Dict[str, CaseResult]:
        """Run the cases; ``progress(done, total, case name)`` is called before each."""
        names = cases or list(self.model.load_cases)
        ordered = self._order_cases(names)
        for k, name in enumerate(ordered):
            if progress is not None:
                progress(k, len(ordered), name)
            case = self.model.load_cases[name]
            if isinstance(case, LinearStaticCase):
                self.results[name] = self.linear_static(case)
            elif isinstance(case, ModalCase):
                self.results[name] = self.modal(case)
            elif isinstance(case, ResponseSpectrumCase):
                self.results[name] = self.response_spectrum(case)
            elif isinstance(case, PDeltaCase):
                self.results[name] = self.pdelta(case)
            elif isinstance(case, NonlinearStaticCase):
                self.results[name] = self.nonlinear.nonlinear_static(case)
            elif isinstance(case, TimeHistoryCase):
                self.results[name] = self.nonlinear.time_history(case)
            else:
                raise ModelError("UNSUPPORTED_CASE", f"Load case type {type(case).__name__} is not supported yet")
        if progress is not None:
            progress(len(ordered), len(ordered), "")
        return self.results

    def _order_cases(self, names: List[str]) -> List[str]:
        """Dependencies first (modal before spectrum, P-Delta before dependents)."""
        out: List[str] = []
        seen = set()

        def visit(n):
            if n in seen:
                return
            c = self.model.load_cases[n]
            deps = [getattr(c, "modal_case", None), getattr(c, "stiffness_from", None), getattr(c, "initial_case", None)]
            deps += [ml[0] for ml in getattr(c, "mode_loads", [])]
            for dep in deps:
                if dep and dep in self.model.load_cases and dep != n:
                    visit(dep)
            seen.add(n)
            out.append(n)

        for n in names:
            visit(n)
        return out

    # ------------------------------------------------------------- static
    def _solve_static(self, F_full: np.ndarray, stiffness_from: Optional[str] = None,
                      loads: Optional[Dict[str, float]] = None, K_red=None, fac=None) -> np.ndarray:
        """Solve K u = F on the free DOFs.  Ground displacements of restrained DOFs
        (CSI "ground displacement" load) enter as K_ff u_f = F_f - K_fs u_s."""
        dofs = self.asm.dofs
        F_red = dofs.T.T @ F_full
        fac = fac or self._factor_for(stiffness_from)
        u_red = np.zeros(dofs.T.shape[1])
        us = self.asm.prescribed_reduced(loads) if loads else None
        rhs = F_red[dofs.free]
        if us is not None:
            if K_red is None:
                K_red = self._pdelta_K[stiffness_from] if stiffness_from else self.K_reduced()
            rhs = rhs - K_red[dofs.free] @ us
            u_red[:] = us
        u_red[dofs.free] = fac.solve(rhs)
        if us is not None:  # one step of iterative refinement: imposed motions are often near rigid-body
            r = F_red[dofs.free] - K_red[dofs.free] @ u_red
            u_red[dofs.free] += fac.solve(r)
        return dofs.T @ u_red  # full

    def _recover(self, u_full: np.ndarray, pattern_loads: Dict[str, float], F_full: np.ndarray,
                 K_full: Optional[sparse.csr_matrix] = None):
        """End forces, reactions and element loads for a displacement state."""
        am = self.am
        K_full = self.asm.stiffness_full() if K_full is None else K_full
        residual = K_full @ u_full - F_full
        reactions: Dict[int, np.ndarray] = {}
        for node in am.nodes:
            if np.any(node.restraints) or np.any(node.springs):
                Tn = node_transform(node)
                sl = slice(6 * node.id, 6 * node.id + 6)
                res_l = Tn @ residual[sl]  # joint local axes
                u_l = Tn @ u_full[sl]
                dg = np.zeros(6)
                for pat, sf in pattern_loads.items():
                    g = node.ground_disp.get(pat)
                    if g is not None:
                        dg += sf * g
                r = np.zeros(6)
                sel = node.restraints | (node.springs > 0)
                r[sel] = res_l[sel]
                # spring reaction = -k (u - u_ground) (force the ground applies)
                for d in range(6):
                    if node.springs[d] > 0 and not node.restraints[d]:
                        r[d] = -node.springs[d] * (u_l[d] - dg[d])
                reactions[node.id] = Tn.T @ r  # global
        end_forces = np.zeros((len(am.frames), 12))
        elem_loads = {}
        dist = None
        for fe in self.asm.frames:
            loads = []
            for pat, sf in pattern_loads.items():
                for ld in fe.element_loads(pat, am.self_weight.get(pat, 0.0), am.gravity):
                    loads.append(_scale_load(ld, sf))
            d = self.asm._element_dofs(fe.el.nodes)
            end_forces[fe.el.id] = fe.end_forces_local(u_full[d], loads)
            elem_loads[fe.el.id] = loads
            fr = fe.foundation_reaction_global(u_full[d])
            if fr is not None:
                dist = np.zeros(6) if dist is None else dist
                dist[:3] += fr
        self._last_distributed = dist
        shell_res = {}
        if not getattr(self, "displacements_only", False):  # a mesh check needs the displacements only
            shell_res = self.asm.shell_resultants(u_full)
        return reactions, end_forces, elem_loads, shell_res

    def _combined_load_vector(self, loads: Dict[str, float], accel: Optional[Dict[str, float]] = None) -> np.ndarray:
        F = np.zeros(self.asm.dofs.n_full)
        for pat, sf in loads.items():
            F += sf * self.asm.load_vector_full(pat)
        if accel:
            M = self.asm.mass_full()
            for d, factor in accel.items():
                r = self.asm.influence_vector_full(d)
                F += -M * r * factor * self.am.gravity
        return F

    def _limited_frames(self):
        out = []
        for fe in self.asm.frames:
            el = fe.el
            if (el.tension_limit is None and el.compression_limit is None) or el.releases_i[0] or el.releases_j[0]:
                continue
            out.append(fe)
        return out

    def _tc_limits(self, case: LinearStaticCase, F: np.ndarray):
        """Tension / compression limits (CSI nonlinear static): a member whose elastic
        axial force passes a limit carries the limit force with no axial stiffness.
        Iterates on the set of capped members; the converged state is stored as an
        imposed axial strain per capped member so that recovery with the elastic
        stiffness reproduces the capped force exactly."""
        frames = self._limited_frames()
        dofs = self.asm.dofs
        K0 = self.asm.stiffness_full()
        capped: Dict[int, float] = {}
        u = self._solve_static(F, loads=case.loads)
        for it in range(200):
            changed = False
            for fe in frames:
                el = fe.el
                loads = []
                for pat, sf in case.loads.items():
                    for ld in fe.element_loads(pat, self.am.self_weight.get(pat, 0.0), self.am.gravity):
                        loads.append(_scale_load(ld, sf))
                d = self.asm._element_dofs(el.nodes)
                n_el = float(fe.internal_forces(u[d], loads, np.array([0.5 * fe.kin.L]))[0, 0])
                tl = el.tension_limit
                cl = el.compression_limit
                cap = None
                if tl is not None and n_el > tl * (1 + 1e-9) + 1e-12:
                    cap = tl
                elif cl is not None and n_el < -cl * (1 + 1e-9) - 1e-12:
                    cap = -cl
                old = capped.get(el.id)
                if cap is None and old is not None:
                    del capped[el.id]
                    changed = True
                elif cap is not None and old != cap:
                    capped[el.id] = cap
                    changed = True
            if not changed:
                break
            # stiffness without the axial stiffness of capped members, plus their constant forces
            Kmod = K0.tolil(copy=True) if capped else K0
            Fmod = F.copy()
            for fe in self.asm.frames:
                c = capped.get(fe.el.id)
                if c is None:
                    continue
                b = fe.elongation_row()
                d = self.asm._element_dofs(fe.el.nodes)
                ka = fe.el.E * fe.el.A / fe.kin.L
                Kmod[np.ix_(d, d)] = Kmod[np.ix_(d, d)].toarray() - ka * np.outer(b, b)
                Fmod[d] -= c * b
            Kmod = sparse.csr_matrix(Kmod)
            K_red = (dofs.T.T @ Kmod @ dofs.T).tocsr()
            fac = self.kernel.factor(self._free_matrix(K_red))
            u = self._solve_static(Fmod, loads=case.loads, K_red=K_red, fac=fac)
        else:
            raise ModelError("NONCONVERGENCE", f"Case {case.name!r}: tension / compression limits did not converge")
        strains = {}
        for fe in self.asm.frames:
            c = capped.get(fe.el.id)
            if c is None:
                continue
            d = self.asm._element_dofs(fe.el.nodes)
            # N = EA (delta / L - eps0) = c  ->  eps0
            eps0 = float(fe.elongation_row() @ u[d]) / fe.kin.L - c / (fe.el.E * fe.el.A)
            strains[fe.el.id] = eps0
        return u, strains, it + 1

    def linear_static(self, case: LinearStaticCase) -> CaseResult:
        F = self._combined_load_vector(case.loads, case.acceleration_loads)
        if getattr(case, "nonlinear", False) and not case.stiffness_from and self._limited_frames():
            u, strains, iters = self._tc_limits(case, F)
            key = f"__tc_{case.name}"
            for fe in self.asm.frames:
                fe.el.loads = [ld for ld in fe.el.loads if ld.pattern != key]
                if fe.el.id in strains:
                    e = np.array([strains[fe.el.id], 0.0, 0.0])
                    fe.el.loads.append(ElementLoad(key, "temperature", "Strain", 0.0, fe.kin.L, e, e.copy()))
                    F[self.asm._element_dofs(fe.el.nodes)] += fe.load_vector_global(fe.element_loads(key))[0]
            loads = dict(case.loads)
            loads[key] = 1.0
            reactions, end_forces, elem_loads, shell_res = self._recover(u, loads, F)
            res = CaseResult(case.name, "NonlinearStatic", u.reshape(-1, 6), reactions, end_forces, elem_loads,
                             shell_res)
            res.distributed_reaction = self._last_distributed
            res.diagnostics["iterations"] = iters
            res.diagnostics["capped_members"] = len(strains)
            return res
        if self._unloaded(case, F):
            # nothing loads this case (no element / joint load, ground motion or
            # acceleration): the solution is exactly zero.  The factorisation and its
            # stability check still run, as for any case.
            self._factor_for(case.stiffness_from)
            return self._zero_static(case)
        u = self._solve_static(F, case.stiffness_from, case.loads)
        K = self._pdelta_K.get(case.stiffness_from) if case.stiffness_from else None
        K_full = None
        if K is not None:
            K_full = self._pdelta_K_full[case.stiffness_from]
        reactions, end_forces, elem_loads, shell_res = self._recover(u, case.loads, F, K_full)
        res = CaseResult(case.name, "LinearStatic", u.reshape(-1, 6), reactions, end_forces, elem_loads, shell_res)
        res.distributed_reaction = self._last_distributed
        res.diagnostics["pivot_ratio"] = getattr(self._factor, "pivot_ratio", None)
        return res

    def _unloaded(self, case: LinearStaticCase, F: np.ndarray) -> bool:
        """True when the case has no load of any kind (its exact solution is zero)."""
        if np.any(F) or case.acceleration_loads or getattr(case, "nonlinear", False):
            return False
        am = self.am
        pats = [p for p, sf in case.loads.items() if sf]
        for p in pats:
            if am.self_weight.get(p, 0.0):
                return False
            if any(np.any(v) for v in am.nodal_loads.get(p, {}).values()):
                return False
        if any(nd.ground_disp.get(p) is not None and np.any(nd.ground_disp[p]) for nd in am.nodes for p in pats):
            return False
        for fe in self.asm.frames:
            if any(ld.pattern in pats for ld in fe.el.loads):
                return False
        for se in self.asm.shells:
            if any(ld.pattern in pats for ld in se.el.loads):
                return False
        return True

    def _zero_static(self, case: LinearStaticCase) -> CaseResult:
        """The result of an unloaded linear static case: every quantity zero, in
        the same layout as a solved case."""
        am = self.am
        n = am.n_nodes
        reactions = {node.id: np.zeros(6) for node in am.nodes if np.any(node.restraints) or np.any(node.springs)}
        end_forces = np.zeros((len(am.frames), 12))
        elem_loads = {fe.el.id: [] for fe in self.asm.frames}
        shell_res = {}
        if not getattr(self, "displacements_only", False):
            from .elements.shell_batch import zero_resultants
            shell_res = zero_resultants(self.asm.shells)
        found = any(getattr(fe.el, "foundation", None) is not None and np.any(fe.el.foundation)
                    for fe in self.asm.frames)
        self._last_distributed = np.zeros(6) if found else None
        res = CaseResult(case.name, "LinearStatic", np.zeros((n, 6)), reactions, end_forces, elem_loads, shell_res)
        res.distributed_reaction = self._last_distributed
        res.diagnostics["pivot_ratio"] = getattr(self._factor, "pivot_ratio", None)
        res.diagnostics["unloaded"] = True
        return res

    # -------------------------------------------------------------- P-Delta
    def pdelta(self, case: PDeltaCase) -> CaseResult:
        F = self._combined_load_vector(case.loads)
        dofs = self.asm.dofs
        K0 = self.asm.stiffness_full()
        u = self._solve_static(F, loads=case.loads)
        K_full = K0
        converged = False
        for it in range(case.max_iterations):
            Kg = self._geometric_stiffness_full(u)
            K_full = (K0 + Kg).tocsr()
            K_red = (dofs.T.T @ K_full @ dofs.T).tocsr()
            fac = self.kernel.factor(self._free_matrix(K_red))
            u_new = self._solve_static(F, loads=case.loads, K_red=K_red, fac=fac)
            change = np.linalg.norm(u_new - u) / max(np.linalg.norm(u_new), 1e-30)
            u = u_new
            if change < case.tolerance:
                converged = True
                break
        if not converged:
            raise ModelError("NONCONVERGENCE", f"P-Delta case {case.name!r} did not converge")
        self._pdelta_K[case.name] = K_red
        if not hasattr(self, "_pdelta_K_full"):
            self._pdelta_K_full = {}
        self._pdelta_K_full[case.name] = K_full
        reactions, end_forces, elem_loads, shell_res = self._recover(u, case.loads, F, K_full)
        res = CaseResult(case.name, "PDelta", u.reshape(-1, 6), reactions, end_forces, elem_loads, shell_res)
        res.distributed_reaction = self._last_distributed
        res.diagnostics["iterations"] = it + 1
        return res

    def _geometric_stiffness_full(self, u_full: np.ndarray) -> sparse.csr_matrix:
        n = self.asm.dofs.n_full
        rows, cols, vals = [], [], []
        for fe in self.asm.frames:
            d = self.asm._element_dofs(fe.el.nodes)
            f = fe.end_forces_local(u_full[d], [])
            P = -f[0]  # tension positive (force on element at I along -1 means tension... f[0] is node->element)
            Kg = fe.geometric_stiffness_global(P)
            r, c = np.meshgrid(d, d, indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(Kg.ravel())
        for se in self.asm.shells:
            d = self.asm._element_dofs(se.el.nodes)
            Kg = se.geometric_stiffness_global(u_full[d])
            r, c = np.meshgrid(d, d, indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(Kg.ravel())
        if not rows:
            return sparse.csr_matrix((n, n))
        return sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(n, n)).tocsr()

    # ---------------------------------------------------------------- modal
    def modal(self, case: ModalCase) -> CaseResult:
        dofs = self.asm.dofs
        K_red = self._pdelta_K[case.stiffness_from] if case.stiffness_from else self.K_reduced()
        M_red = self.asm.mass_reduced()
        Kf = self._free_matrix(K_red)
        Mf = self._free_matrix(M_red)
        if case.method.lower() == "ritz":
            self.am.warnings.append("Ritz vectors not implemented; eigenvectors used instead")
        # the static factor is of this same matrix: the modal condensation reuses it
        fac = None if case.stiffness_from else self._modal_factor(Kf)
        try:
            w2, phi_free = self.kernel.eigen(Kf, Mf, case.max_modes, shift=case.shift, factor=fac)
        except TypeError:  # a kernel without the factor argument
            w2, phi_free = self.kernel.eigen(Kf, Mf, case.max_modes, shift=case.shift)
        if case.cutoff > 0:
            keep = np.sqrt(np.maximum(w2, 0)) / (2 * np.pi) <= case.cutoff
            keep[:max(case.min_modes, 1)] = True
            w2, phi_free = w2[keep], phi_free[:, keep]
        omega = np.sqrt(np.maximum(w2, 1e-300))
        periods = 2 * np.pi / omega
        # expand to the full DOF set
        n_modes = phi_free.shape[1]
        phi_red = np.zeros((dofs.T.shape[1], n_modes))
        phi_red[dofs.free] = phi_free
        phi_full = dofs.T @ phi_red
        # participation
        M_full = self.asm.mass_full()
        com = self._center_of_mass(M_full)
        part = np.zeros((n_modes, 6))
        total = np.zeros(6)
        for k, d in enumerate(("UX", "UY", "UZ", "RX", "RY", "RZ")):
            r = self.asm.influence_vector_full(d, com)
            # only unrestrained & active mass participates
            r_act = r.copy()
            r_act[dofs.restrained_full] = 0.0
            inactive = ~np.tile(self.am.active_dofs, self.am.n_nodes)
            r_act[inactive] = 0.0
            Mr = M_full * r_act
            part[:, k] = phi_full.T @ Mr
            total[k] = float(r_act @ Mr)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(total > 0, part ** 2 / total, 0.0)
        modal = ModalData(periods, omega, w2, phi_full, part, ratios, total, com, case.method)
        res = CaseResult(case.name, "Modal", np.zeros((self.am.n_nodes, 6)), {},
                         np.zeros((len(self.am.frames), 12)), modal=modal)
        res.modal_displacements = np.transpose(phi_full.reshape(self.am.n_nodes, 6, n_modes), (2, 0, 1))
        return res

    def _modal_factor(self, Kf):
        """The static factor of K, when it exists or can be made cleanly (a stiffness
        that is singular on its own - isolators, a mechanism carried by mass - is
        left to the eigen solver's own condensation, as before)."""
        if self._factor is not None:
            return self._factor
        if Kf.shape[0] <= 3000:
            return None  # the dense path does not use it
        try:
            fac = self.kernel.factor(Kf)
        except (ModelError, ArithmeticError, RuntimeError, ValueError):
            return None
        if getattr(fac, "unstable", None):
            return None
        self._factor = fac  # the static cases reuse it
        return fac

    def _center_of_mass(self, M_full: np.ndarray) -> np.ndarray:
        coords = self.am.coords()
        mx = M_full[0::6]
        tot = mx.sum()
        if tot <= 0:
            return coords.mean(axis=0)
        return (mx[:, None] * coords).sum(axis=0) / tot

    # ---------------------------------------------------- response spectrum
    def response_spectrum(self, case: ResponseSpectrumCase) -> CaseResult:
        if case.modal_case not in self.results:
            raise ModelError("CASE_ORDER", f"Run modal case {case.modal_case!r} before {case.name!r}")
        modal = self.results[case.modal_case].modal
        n_modes = modal.n_modes
        omega = modal.omega
        phi = modal.shapes  # (n_full, n_modes)
        M_full = self.asm.mass_full()
        ang = math.radians(case.angle)
        dir_vectors = {
            "U1": np.array([math.cos(ang), math.sin(ang), 0.0]),
            "U2": np.array([-math.sin(ang), math.cos(ang), 0.0]),
            "U3": np.array([0.0, 0.0, 1.0]),
        }
        unit_r = {d: self.asm.influence_vector_full(d) for d in ("UX", "UY", "UZ")}
        dofs = self.asm.dofs
        K_full = self.asm.stiffness_full()
        warnings: List[str] = []

        # per-direction, per-mode signed responses
        dir_results = []
        for ld in case.loads:
            fn = self.model.functions.get(ld.function)
            if not isinstance(fn, ResponseSpectrumFunction):
                raise ModelError("UNKNOWN_FUNCTION", f"Spectrum function {ld.function!r} not found")
            if abs(fn.damping - case.damping) > 1e-9:
                warnings.append(f"Spectrum {fn.name}: function damping {fn.damping} differs from case damping "
                                f"{case.damping}; no damping scaling applied")
            v = dir_vectors[ld.direction]
            r = v[0] * unit_r["UX"] + v[1] * unit_r["UY"] + v[2] * unit_r["UZ"]
            r[dofs.restrained_full] = 0.0
            gamma = phi.T @ (M_full * r)  # (n_modes,)
            scale = ld.scale * self.model.units.factor("acceleration")  # -> m/s^2 multiplier of the function
            u_modes = np.zeros((n_modes, dofs.n_full))
            for n in range(n_modes):
                Sa = fn.value(modal.periods[n]) * scale
                Sd = Sa / (omega[n] ** 2)
                u_modes[n] = gamma[n] * Sd * phi[:, n]
            dir_results.append(u_modes)

        # modal combination per direction
        rho = _correlation(omega, case.damping, case.modal_combination)
        per_dir = []
        base_dir = []
        for u_modes in dir_results:
            disp_m = u_modes  # (n_modes, n_full)
            ef_m = np.zeros((n_modes, len(self.am.frames), 12))
            react_m: Dict[int, np.ndarray] = {}
            base_m = np.zeros((n_modes, 6))  # total base reaction of each mode, about the global origin
            for n in range(n_modes):
                reactions, ef, _, _ = self._recover(u_modes[n], {}, np.zeros(dofs.n_full), K_full)
                ef_m[n] = ef
                for nid, rv in reactions.items():
                    react_m.setdefault(nid, np.zeros((n_modes, 6)))[n] = rv
                    xyz = np.asarray(self.am.nodes[nid].xyz, float)
                    base_m[n, :3] += rv[:3]
                    base_m[n, 3:] += rv[3:] + np.cross(xyz, rv[:3])
            per_dir.append((
                _combine_modes(disp_m, rho), _combine_modes(ef_m.reshape(n_modes, -1), rho).reshape(len(self.am.frames), 12),
                {nid: _combine_modes(arr, rho) for nid, arr in react_m.items()},
            ))
            # like ETABS, the base reaction is combined from the modal totals, not summed
            # from the (positive) envelopes of the individual supports
            base_dir.append(_combine_modes(base_m, rho))
        # directional combination
        disp = _combine_directions([p[0] for p in per_dir], case)
        ef = _combine_directions([p[1] for p in per_dir], case)
        react_nodes = set().union(*[p[2].keys() for p in per_dir]) if per_dir else set()
        reactions = {nid: _combine_directions([p[2].get(nid, np.zeros(6)) for p in per_dir], case) for nid in react_nodes}
        res = CaseResult(case.name, "ResponseSpectrum", disp.reshape(-1, 6), reactions, ef, {}, {}, is_envelope=True,
                         modal=modal)
        res.spectrum_base = _combine_directions(base_dir, case) if base_dir else np.zeros(6)
        res.modal_displacements = np.stack([d.reshape(n_modes, self.am.n_nodes, 6) for d in dir_results]) if dir_results else None
        res.warnings = warnings
        return res


# ----------------------------------------------------------------------------
def _scale_load(ld, sf: float):
    if sf == 1.0:
        return ld
    from ..mesh.analysis_model import ElementLoad
    return ElementLoad(ld.pattern, ld.kind, ld.load_type, ld.x_a, ld.x_b, ld.value_a * sf, ld.value_b * sf)


class AbsoluteCorrelation(np.ndarray):
    """Correlation matrix applied to |R_i R_j| (NRC 10 % / double-sum rules)."""


def _correlation(omega: np.ndarray, zeta: float, method: str) -> np.ndarray:
    n = len(omega)
    method = method.upper().replace(" ", "").replace("%", "PCT")
    if method in ("SRSS",):
        return np.eye(n)
    if method in ("ABS", "ABSSUM"):
        return np.ones((n, n))
    if method in ("NRC10", "10PCT", "NRC10PCT", "TENPERCENT"):
        # US NRC Reg. Guide 1.92 ten-percent method: modes i, j are closely spaced when
        # their frequencies differ by at most 10 % of the lower one; their responses are
        # added as 2|R_i R_j| to the SRSS sum.
        rho = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    lo, hi = sorted((omega[i], omega[j]))
                    if lo > 0 and (hi - lo) <= 0.1 * lo:
                        rho[i, j] = 1.0
        return rho.view(AbsoluteCorrelation)
    # CQC (Der Kiureghian) ; GMC treated as CQC
    rho = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = omega[j] / omega[i] if omega[i] > 0 else 1.0
            num = 8 * zeta ** 2 * (1 + r) * r ** 1.5
            den = (1 - r * r) ** 2 + 4 * zeta ** 2 * r * (1 + r) ** 2
            rho[i, j] = num / den if den > 0 else 1.0
    return rho


def _combine_modes(values: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """values: (n_modes, ...) signed modal responses -> combined positive."""
    shape = values.shape[1:]
    v = values.reshape(values.shape[0], -1)
    if isinstance(rho, AbsoluteCorrelation):
        a = np.abs(v)
        out = np.sqrt(np.einsum("ik,ij,jk->k", a, np.asarray(rho), a))
        return out.reshape(shape)
    if np.array_equal(rho, np.eye(len(rho))):
        out = np.sqrt(np.sum(v * v, axis=0))
    elif np.all(rho == 1.0):
        out = np.sum(np.abs(v), axis=0)
    else:
        out = np.sqrt(np.maximum(np.einsum("ik,ij,jk->k", v, rho, v), 0.0))
    return out.reshape(shape)


def _combine_directions(items: List[np.ndarray], case: ResponseSpectrumCase) -> np.ndarray:
    if not items:
        return np.zeros(0)
    method = case.directional_combination.upper()
    arr = np.stack(items)
    if method in ("SRSS", "CQC3"):
        return np.sqrt(np.sum(arr * arr, axis=0))
    if method in ("ABS", "ABSSUM"):
        return case.directional_abs_factor * np.sum(np.abs(arr), axis=0)
    raise ModelError("INVALID_COMBINATION", f"Unknown directional combination {case.directional_combination!r}")
