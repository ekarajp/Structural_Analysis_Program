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
        links = []
        if getattr(model, "links", None):
            from .nonlinear.builder import NonlinearBuilder
            links = NonlinearBuilder(model, am, None).linear_links()
        self.asm = Assembler(am, links)
        self._links = links
        self._K_red: Optional[sparse.csr_matrix] = None
        self._factor = None
        self._pdelta_K: Dict[str, sparse.csr_matrix] = {}
        self._pdelta_K_full: Dict[str, sparse.csr_matrix] = {}
        self.results: Dict[str, CaseResult] = {}
        # stiffness sets (gravity / lateral shell modifiers): the matrices of the other set
        self.stiffness_set = "gravity"
        self._sets: Dict[str, dict] = {}

    # ------------------------------------------------------------ stiffness sets
    _SET_STATE = ("asm", "_K_red", "_factor", "_pdelta_K", "_pdelta_K_full", "_mass_cache")

    def has_lateral_set(self) -> bool:
        """Some shells have other modifiers in the lateral load cases."""
        return any(sh.modifiers_lateral is not None for sh in self.am.shells)

    def use_stiffness_set(self, name: str):
        """Solve the next cases with the matrices of the stiffness set ``name``."""
        if name == self.stiffness_set or (name == "lateral" and not self.has_lateral_set()):
            return
        self._sets[self.stiffness_set] = {k: self.__dict__.get(k) for k in self._SET_STATE}
        state = self._sets.get(name)
        if state is None:
            import copy
            import dataclasses
            am = copy.copy(self.am)  # the same mesh, nodes, constraints: the shells' lateral modifiers
            am.shells = [dataclasses.replace(sh, modifiers=sh.modifiers_lateral) if sh.modifiers_lateral is not None
                         else sh for sh in self.am.shells]
            state = {"asm": Assembler(am, self._links), "_K_red": None, "_factor": None, "_pdelta_K": {},
                     "_pdelta_K_full": {}, "_mass_cache": None}
        for k, v in state.items():
            if v is None and k == "_mass_cache":
                self.__dict__.pop(k, None)
            else:
                self.__dict__[k] = v
        self.stiffness_set = name

    def asm_for_case(self, case: Optional[str]) -> Assembler:
        """The assembler (shell stiffness) a case was solved with."""
        if not case or case not in self.model.load_cases or not self.has_lateral_set():
            return self.asm
        from ..model.loads import stiffness_set_of
        c = self.model.load_cases[case]
        name = "gravity" if isinstance(c, (NonlinearStaticCase, TimeHistoryCase)) else stiffness_set_of(self.model,
                                                                                                         case)
        if name == self.stiffness_set:
            return self.asm
        if name not in self._sets:
            prev = self.stiffness_set
            self.use_stiffness_set(name)
            self.use_stiffness_set(prev)
        return self._sets[name]["asm"]

    def _pdelta_into_set(self, pd_case: str):
        """The stiffness of P-Delta case ``pd_case`` (run with the gravity set) for the
        current set: its geometric stiffness added to this set's elastic stiffness."""
        if pd_case in self._pdelta_K or self.stiffness_set == "gravity":
            return
        grav = self._sets.get("gravity") or {}
        K_full_g = (grav.get("_pdelta_K_full") or {}).get(pd_case)
        if K_full_g is None:
            return
        Kg = (K_full_g - grav["asm"].stiffness_full()).tocsr()
        K_full = (self.asm.stiffness_full() + Kg).tocsr()
        dofs = self.asm.dofs
        self._pdelta_K[pd_case] = (dofs.T.T @ K_full @ dofs.T).tocsr()
        self._pdelta_K_full[pd_case] = K_full

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

    _MASS_FLAGS = (("mass_include_elements", "include_element_mass"), ("mass_include_added", "include_added_mass"),
                   ("mass_lateral", "include_lateral"), ("mass_vertical", "include_vertical"),
                   ("mass_lump_at_stories", "lump_at_stories"))

    def mass_full_for(self, source: Optional[str]) -> np.ndarray:
        """Lumped mass (full DOF set) of the named mass source; the model's default
        source (the one the analysis model was built with) when None."""
        if not source:
            return self.asm.mass_full()
        ms = self.model.mass_sources.get(source)
        if ms is None:
            raise ModelError("UNKNOWN_MASS_SOURCE", f"Mass source {source!r} is not defined")
        am = self.am
        want = {a: bool(getattr(ms, b, False)) for a, b in self._MASS_FLAGS}
        want_loads = {k: float(v) for k, v in ms.load_patterns.items()}
        if all(getattr(am, a) == v for a, v in want.items()) and am.mass_from_loads == want_loads:
            return self.asm.mass_full()
        cache = self.__dict__.setdefault("_mass_cache", {})
        if source in cache:
            return cache[source]
        for pat in want_loads:
            if pat not in self.model.load_patterns:
                raise ModelError("UNKNOWN_PATTERN", f"Mass source {source!r}: load pattern {pat!r} is not defined")
        saved = {a: getattr(am, a) for a in want}
        saved_loads, saved_M = am.mass_from_loads, self.asm._M_full
        saved_lump = getattr(self.asm, "_lump_map", None)
        try:
            for a, v in want.items():
                setattr(am, a, v)
            am.mass_from_loads = want_loads
            self.asm._M_full = None
            self.asm._lump_map = None
            M = self.asm.mass_full().copy()
        finally:
            for a, v in saved.items():
                setattr(am, a, v)
            am.mass_from_loads = saved_loads
            self.asm._M_full = saved_M
            self.asm._lump_map = saved_lump
        cache[source] = M
        return M

    def _case_mass(self, modal_case: Optional[str]) -> np.ndarray:
        """Mass of a case that depends on a modal case (RS, modal TH): that case's source."""
        res = self.results.get(modal_case) if modal_case else None
        M = getattr(res, "mass_full", None) if res is not None else None
        return M if M is not None else self.asm.mass_full()

    def _free_matrix(self, K: sparse.csr_matrix) -> sparse.csr_matrix:
        f = self.asm.dofs.free
        return K[f][:, f].tocsr()

    def _factor_for(self, stiffness_from: Optional[str]):
        if stiffness_from:
            if stiffness_from not in self._pdelta_K:
                raise ModelError("CASE_ORDER", f"Stiffness case {stiffness_from!r} has not been run yet")
            fac = self.kernel.factor(self._free_matrix(self._pdelta_K[stiffness_from]))
            self._check_definite(fac, f"Stiffness from {stiffness_from!r}")
            return fac
        if self._factor is None:
            fac = self.kernel.factor(self._free_matrix(self.K_reduced()))
            self._check_definite(fac, "Linear stiffness")
            self._factor = fac
            self._report_instability(self._factor)
        return self._factor

    def _dof_labels(self, free_indices, limit: int = 8):
        dofs = self.asm.dofs
        keep = np.where(dofs.reduced_of_full >= 0)[0]
        labels = []
        for j in list(free_indices)[:limit]:
            red = int(dofs.free[j])
            full = int(keep[red]) if red < len(keep) else -1
            node, comp = divmod(full, 6)
            name = self.am.nodes[node].point if 0 <= node < len(self.am.nodes) else None
            labels.append(f"{name or f'node {node}'} {('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ')[comp]}")
        more = f" (+{len(free_indices) - limit} more)" if len(free_indices) > limit else ""
        return ", ".join(labels) + more

    def _check_definite(self, fac, what: str):
        """A stiffness with negative eigenvalues (negative pivots of its LDL^T) is no
        stable equilibrium: the axial compression of a P-Delta case beyond buckling,
        negative springs.  The solution of such a system is not a result: refused."""
        neg = getattr(fac, "negative", None)
        if neg:
            raise ModelError(
                "NEGATIVE_STIFFNESS",
                f"{what}: the stiffness is not positive definite ({len(neg)} negative eigenvalue(s), at "
                f"{self._dof_labels(neg)}) - the structure is unstable under these loads (axial compression "
                f"beyond the buckling load, or negative stiffness); no results are given")

    def _report_instability(self, fac):
        """Warn (ETABS / SAP2000 style) about equations that are numerically singular."""
        bad = getattr(fac, "unstable", None)
        if not bad:
            return
        dofs = self.asm.dofs
        keep = np.where(dofs.reduced_of_full >= 0)[0]
        # the pivot where the factorisation broke down is not necessarily where the
        # mechanism is (the elimination order moves it): the DOFs that move in the null
        # vector of K are.  One inverse-iteration step with the (nearly singular) factor:
        # K x = b grows without bound along the mechanism.
        try:
            rng = np.random.default_rng(12345)
            b = rng.standard_normal(len(dofs.free))
            x = np.asarray(fac.solve(b), float).ravel()
            if np.all(np.isfinite(x)) and np.max(np.abs(x)) > 0:
                ax = np.abs(x)
                order = np.argsort(-ax)
                moving = [int(j) for j in order if ax[j] >= 0.01 * ax[order[0]]]
                if moving:
                    bad = moving
        except Exception:  # keep the pivot positions
            pass
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
        from ..model.loads import stiffness_set_of
        try:
            self._run_ordered(ordered, progress, stiffness_set_of)
        finally:
            self.use_stiffness_set("gravity")  # results read the gravity set's matrices (frames: the same)
            for lst in (self.am.warnings, self.am.info):
                lst[:] = list(dict.fromkeys(lst))  # (a message of both sets once)
        return self.results

    def _run_ordered(self, ordered, progress, stiffness_set_of):
        for k, name in enumerate(ordered):
            if progress is not None:
                progress(k, len(ordered), name)
            case = self.model.load_cases[name]
            if isinstance(case, (NonlinearStaticCase, TimeHistoryCase)):
                self.use_stiffness_set("gravity")
            else:
                self.use_stiffness_set(stiffness_set_of(self.model, name))
                sf = getattr(case, "stiffness_from", None)
                if sf:
                    self._pdelta_into_set(sf)
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
        stack: List[str] = []

        def visit(n):
            if n in seen:
                return
            if n in stack:  # a case that depends (through others) on itself
                cyc = stack[stack.index(n):] + [n]
                raise ModelError("CASE_CYCLE", "Load cases depend on each other in a loop: " + " -> ".join(cyc),
                                 f"load_cases.{n}")
            stack.append(n)
            c = self.model.load_cases[n]
            deps = [getattr(c, "modal_case", None), getattr(c, "stiffness_from", None), getattr(c, "initial_case", None)]
            deps += [ml[0] for ml in getattr(c, "mode_loads", [])]
            for dep in deps:
                if dep and dep in self.model.load_cases:
                    visit(dep)
            stack.pop()
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
        support_res = self.asm.support_residual(residual)
        reactions: Dict[int, np.ndarray] = {}
        for node in am.nodes:
            if np.any(node.restraints) or np.any(node.springs):
                Tn = node_transform(node)
                sl = slice(6 * node.id, 6 * node.id + 6)
                res_l = support_res(node, Tn)  # joint local axes, with the constrained nodes on it
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
        for lk in self.asm.links:  # a grounded link is a support: its force is a reaction
            if len(lk.nodes) == 1:
                nid = lk.nodes[0]
                d = self.asm._element_dofs(lk.nodes)
                f = lk.initial_stiffness_global() @ u_full[d]
                reactions[nid] = reactions.get(nid, np.zeros(6)) - f
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
                dist += fr
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
            Kg = self._geometric_stiffness_full(u, case.loads)
            K_full = (K0 + Kg).tocsr()
            K_red = (dofs.T.T @ K_full @ dofs.T).tocsr()
            fac = self.kernel.factor(self._free_matrix(K_red))
            self._check_definite(fac, f"P-Delta case {case.name!r}")
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

    def _geometric_stiffness_full(self, u_full: np.ndarray, pattern_loads: Optional[Dict[str, float]] = None
                                  ) -> sparse.csr_matrix:
        """Initial-stress stiffness from the actual axial forces: the end forces of
        the displacements *and* of the element loads of the case (fixed-end actions,
        temperature eigenstrains) - a free thermal expansion has no axial force, a
        restrained one does (CA22)."""
        n = self.asm.dofs.n_full
        rows, cols, vals = [], [], []
        am = self.am
        for fe in self.asm.frames:
            d = self.asm._element_dofs(fe.el.nodes)
            loads = []
            for pat, sf in (pattern_loads or {}).items():
                for ld in fe.element_loads(pat, am.self_weight.get(pat, 0.0), am.gravity):
                    loads.append(_scale_load(ld, sf))
            f = fe.end_forces_local(u_full[d], loads)
            # axial force (tension positive), the mean of the two ends (a distributed
            # axial load makes it vary along the member)
            P = 0.5 * (-f[0] + f[6])
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
        M_full = self.mass_full_for(getattr(case, "mass_source", None))
        T = dofs.T
        M_red = (T.T @ sparse.diags(M_full) @ T).tocsr()
        Kf = self._free_matrix(K_red)
        Mf = self._free_matrix(M_red)
        self._check_mass(Mf, case.name)
        if case.method.lower() == "ritz":
            self.am.warnings.append("Ritz vectors not implemented; eigenvectors used instead")
        # the static factor is of this same matrix: the modal condensation reuses it
        fac = None if case.stiffness_from else self._modal_factor(Kf)
        try:
            w2, phi_free = self.kernel.eigen(Kf, Mf, case.max_modes, shift=case.shift, factor=fac)
        except TypeError:  # a kernel without the factor argument
            w2, phi_free = self.kernel.eigen(Kf, Mf, case.max_modes, shift=case.shift)
        bad = ~np.isfinite(w2) | (w2 < 0)
        if np.any(bad):
            raise ModelError("NEGATIVE_EIGENVALUE",
                             f"Modal case {case.name!r}: {int(bad.sum())} eigenvalue(s) are negative or not finite "
                             f"(lowest {float(np.min(w2)):.4g} rad^2/s^2) - the mass or the stiffness is not positive "
                             "(negative mass / density, or an unstable stiffness); no modal results are given")
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
        res.mass_full = M_full
        return res

    def _accidental_torsion(self, case, acc_full: np.ndarray, M_full: np.ndarray, warnings: List[str]):
        """Static response to the accidental torsion of a spectrum case (CSI SAP2000 /
        ETABS 2013 procedure): at every joint of a rigid diaphragm the combined X and Y
        accelerations times the joint's mass times the eccentricity (ratio x diaphragm
        extent across the motion) give two torsions; the larger is applied about the
        joint.  Returns (u_full, reactions, end forces, base resultant) or None."""
        am = self.am
        xyz = am.coords()
        F = np.zeros(self.asm.dofs.n_full)
        n_diaph = 0
        skipped = 0
        for c in am.constraints:
            if not c.rigid:
                skipped += 1
                continue
            ids = list(c.nodes)
            if not ids:
                continue
            ax = [a for a in range(3) if a != c.normal]  # the two in-plane axes
            ext = xyz[ids][:, ax].max(axis=0) - xyz[ids][:, ax].min(axis=0)
            e0, e1 = case.eccentricity_ratio * ext[1], case.eccentricity_ratio * ext[0]
            t = 0.0
            for n in ids:
                m0, m1 = M_full[6 * n + ax[0]], M_full[6 * n + ax[1]]
                a0, a1 = acc_full[6 * n + ax[0]], acc_full[6 * n + ax[1]]
                t += max(abs(m0 * a0 * e0), abs(m1 * a1 * e1))
            if t:
                F[6 * c.master + 3 + c.normal] += t
                n_diaph += 1
        if skipped:
            warnings.append(f"Spectrum {case.name}: accidental eccentricity is applied to rigid diaphragms only; "
                            f"{skipped} semi-rigid diaphragm(s) get no accidental torsion")
        if not n_diaph:
            warnings.append(f"Spectrum {case.name}: eccentricity ratio {case.eccentricity_ratio:g} has no effect - "
                            "the model has no rigid diaphragm with lateral mass (accidental torsion "
                            "is applied at diaphragms only)")
            return None
        stiff = getattr(self.model.load_cases.get(case.modal_case), "stiffness_from", None)
        u = self._solve_static(F, stiff)
        K_full = self._pdelta_K_full[stiff] if stiff else None
        reactions, ef, _, _ = self._recover(u, {}, F, K_full)
        base = np.zeros(6)
        for nid, rv in reactions.items():
            base[:3] += rv[:3]
            base[3:] += rv[3:] + np.cross(xyz[nid], rv[:3])
        if self._last_distributed is not None:
            base += self._last_distributed
        return u, reactions, ef, base

    def _check_mass(self, Mf, what: str):
        """The assembled mass of the free DOFs must not be negative (negative added mass is
        allowed while the net mass of every DOF stays >= 0)."""
        d = np.asarray(Mf.diagonal()).ravel()
        if d.size == 0:
            return
        tol = 1e-9 * max(float(np.max(np.abs(d))), 1e-300)
        neg = np.where(d < -tol)[0]
        if neg.size:
            raise ModelError("NEGATIVE_MASS",
                             f"Modal case {what!r}: the net mass is negative at {len(neg)} degree(s) of freedom "
                             f"({self._dof_labels(neg)}; "
                             f"lowest {float(d[neg].min()):.4g}) - check negative added mass, mass from loads "
                             "and material densities; no modal results are given")

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
        check_spectrum_case(case)
        modal = self.results[case.modal_case].modal
        n_modes = modal.n_modes
        omega = modal.omega
        phi = modal.shapes  # (n_full, n_modes)
        M_full = self._case_mass(case.modal_case)
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
        alpha = _rigid_fractions(omega, case)

        def comb(arr):
            """Modal combination; with rigid frequencies (f1, f2) the response of each mode
            is split into a periodic part (combined by the rule) and a rigid part (added
            algebraically over the modes), Gupta / CSI: R = sqrt(Rp^2 + Rr^2)."""
            if alpha is None:
                return _combine_modes(arr, rho)
            a = alpha.reshape((-1,) + (1,) * (arr.ndim - 1))
            per = _combine_modes(arr * np.sqrt(1.0 - a * a), rho)
            rig = np.abs(np.sum(arr * a, axis=0))
            return np.sqrt(per * per + rig * rig)

        per_dir = []
        base_dir = []
        acc_dir = []
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
                if self._last_distributed is not None:  # distributed supports (frame foundations)
                    base_m[n] += self._last_distributed
            per_dir.append((
                comb(disp_m), comb(ef_m.reshape(n_modes, -1)).reshape(len(self.am.frames), 12),
                {nid: comb(arr) for nid, arr in react_m.items()},
            ))
            # like ETABS, the base reaction is combined from the modal totals, not summed
            # from the (positive) envelopes of the individual supports
            base_dir.append(comb(base_m))
            if case.eccentricity_ratio:
                acc_dir.append(comb(disp_m * (omega ** 2)[:, None]))  # joint (pseudo) accelerations
        # directional combination
        disp = _combine_directions([p[0] for p in per_dir], case)
        ef = _combine_directions([p[1] for p in per_dir], case)
        react_nodes = set().union(*[p[2].keys() for p in per_dir]) if per_dir else set()
        reactions = {nid: _combine_directions([p[2].get(nid, np.zeros(6)) for p in per_dir], case) for nid in react_nodes}
        spectrum_base = _combine_directions(base_dir, case) if base_dir else np.zeros(6)
        ecc = None
        if case.eccentricity_ratio and acc_dir:
            ecc = self._accidental_torsion(case, _combine_directions(acc_dir, case), M_full, warnings)
            if ecc is not None:  # its static response is added to the spectrum envelope (+/- eccentricity)
                u_t, react_t, ef_t, base_t = ecc
                disp = disp + np.abs(u_t)
                ef = ef + np.abs(ef_t)
                for nid, rv in react_t.items():
                    reactions[nid] = reactions.get(nid, np.zeros(6)) + np.abs(rv)
                spectrum_base = spectrum_base + np.abs(base_t)
        res = CaseResult(case.name, "ResponseSpectrum", disp.reshape(-1, 6), reactions, ef, {}, {}, is_envelope=True,
                         modal=modal)
        res.spectrum_base = spectrum_base
        res.static_addition = None if ecc is None else ecc[0].reshape(-1, 6)  # accidental torsion response
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
    if method not in ("CQC", "GMC"):
        raise ModelError("INVALID_COMBINATION", f"Unknown modal combination {method!r}")
    # CQC (Der Kiureghian); GMC: the same correlation of the periodic parts (Gupta), the
    # rigid parts are added algebraically (see _rigid_fractions)
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


MODAL_COMBINATIONS = ("CQC", "SRSS", "ABS", "GMC", "NRC10%")
DIRECTIONAL_COMBINATIONS = ("SRSS", "ABS")


def check_spectrum_case(case) -> None:
    """Refuse spectrum options that are unknown or not implemented (never substituted)."""
    from ..model.loads import spectrum_case_problems
    probs = spectrum_case_problems(case)
    if probs:
        raise ModelError(probs[0][0], "; ".join(m for _, m in probs))


def _rigid_fractions(omega: np.ndarray, case) -> Optional[np.ndarray]:
    """Rigid-response coefficient of each mode (Gupta): alpha = ln(f/f1) / ln(f2/f1)
    clipped to [0, 1]; None when the case gives no rigid frequencies."""
    if case.rigid_response is None:
        return None
    f1, f2 = float(case.rigid_response[0]), float(case.rigid_response[1])
    f = np.asarray(omega, float) / (2 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log(np.where(f > 0, f, f1) / f1) / math.log(f2 / f1)
    return np.clip(a, 0.0, 1.0)


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
    if method == "SRSS":
        return np.sqrt(np.sum(arr * arr, axis=0))
    if method in ("ABS", "ABSSUM"):
        return case.directional_abs_factor * np.sum(np.abs(arr), axis=0)
    raise ModelError("INVALID_COMBINATION", f"Unknown directional combination {case.directional_combination!r}")
