"""Nonlinear finite element domain over the linear engine's DOF numbering.

The free equations are the same as in the linear engine: rigid diaphragms
are master-slave constraints, restrained / inactive DOFs are removed
(``u_full = A u_free``).  Shells, elastic frames and joint springs keep their
linear stiffness in one constant sparse matrix; only nonlinear frames and
links are evaluated element by element.  Elastic frames get the P-Delta
string stiffness of their current axial force when the analysis is
geometrically nonlinear, exactly like the nonlinear frames.
"""
from __future__ import annotations

import copy as _copy
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import sparse

from ...mesh.analysis_model import ElementLoad
from ..assembly import Assembler, node_transform
from .elements import ElementNonConvergence, Link, NLFrame


def _scale_load(ld: ElementLoad, sf: float) -> ElementLoad:
    return ElementLoad(ld.pattern, ld.kind, ld.load_type, ld.x_a, ld.x_b, ld.value_a * sf, ld.value_b * sf)


class NonlinearDomain:
    def __init__(self, asm: Assembler, nl_frames: Dict[int, NLFrame], links: Sequence[Link] = (),
                 geometry: str = "None", patterns: Iterable[str] = ()):
        self.asm = asm
        self.am = asm.am
        self.dofs = asm.dofs
        self.nl_frames = dict(nl_frames)
        self.links = list(links)
        self.elements = list(self.nl_frames.values()) + self.links
        self.geometry = geometry.lower().replace("-", "")
        d = self.dofs
        T = d.T.tocsc()
        self.Afree = T[:, d.free].tocsr()  # full x free
        self.AfreeT = self.Afree.T.tocsr()
        self.n = d.n_free
        self.n_full = d.n_full
        # element DOF lists (full numbering)
        for e in self.elements:
            e.dofs = np.concatenate([np.arange(6 * n, 6 * n + 6) for n in e.nodes])
        self._build_linear()
        M = asm.mass_full()
        self.M_full = M
        self.M = (self.AfreeT @ sparse.diags(M) @ self.Afree).tocsr()
        self._build_pattern()
        self._pattern_cache: Dict[str, np.ndarray] = {}
        self._elem_loads_cache: Dict[str, Dict[int, List[ElementLoad]]] = {}
        self.factors: Dict[str, float] = {}
        self.load_norm = 0.0
        # state
        self.u = np.zeros(self.n)
        self.u_c = np.zeros(self.n)
        self.R_full = np.zeros(self.n_full)
        self._K_init = None

    # ------------------------------------------------------------ linear part
    def _build_linear(self):
        asm = self.asm
        n_full = self.n_full
        rows, cols, vals = [], [], []
        self.linear_frames = [fe for fe in asm.frames if fe.el.id not in self.nl_frames]
        for fe in self.linear_frames:
            K = fe.stiffness_global()
            dd = asm._element_dofs(fe.el.nodes)
            r, c = np.meshgrid(dd, dd, indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(K.ravel())
        for se in asm.shells:
            K = se.stiffness_global()
            dd = asm._element_dofs(se.el.nodes)
            r, c = np.meshgrid(dd, dd, indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(K.ravel())
        for node in self.am.nodes:
            if not np.any(node.springs):
                continue
            Tn = node_transform(node)
            Kn = Tn.T @ np.diag(node.springs) @ Tn
            r, c = np.nonzero(Kn)
            rows.append(6 * node.id + r)
            cols.append(6 * node.id + c)
            vals.append(Kn[r, c])
        if rows:
            K = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(n_full, n_full)).tocsr()
        else:
            K = sparse.csr_matrix((n_full, n_full))
        self.K_lin_full = K
        self.K_lin = (self.AfreeT @ K @ self.Afree).tocsr()
        # P-Delta strings of the elastic frames: N = Nrow . u + N0(loads); Kg = sum N_e G_e
        self._pd_avail = bool(self.linear_frames)
        self._pd = self.geometry in ("pdelta",) and self._pd_avail
        if self._pd_avail:
            nr, nc, nv = [], [], []
            gr, gc, gv, ge = [], [], [], []
            for k, fe in enumerate(self.linear_frames):
                kin = fe.kin
                Kc, _, _ = fe.condense(fe.local_stiffness_unreleased())
                Ag = kin.R12 @ kin.T_off
                row = -(Kc[0] @ Ag)  # N (tension +) = -f[0]
                dd = asm._element_dofs(fe.el.nodes)
                nz = np.nonzero(row)[0]
                nr += [k] * len(nz)
                nc += list(dd[nz])
                nv += list(row[nz])
                S = np.zeros((12, 12))
                g = 1.0 / kin.Lf
                for a, b in ((1, 7), (2, 8)):
                    S[a, a] += g
                    S[b, b] += g
                    S[a, b] -= g
                    S[b, a] -= g
                A = kin.T_rigid @ kin.R12 @ kin.T_off
                G = A.T @ S @ A
                r, c = np.meshgrid(dd, dd, indexing="ij")
                gr.append(r.ravel())
                gc.append(c.ravel())
                gv.append(G.ravel())
                ge.append(np.full(144, k))
            self._N_mat = sparse.csr_matrix((nv, (nr, nc)), shape=(len(self.linear_frames), n_full))
            self._G_rows = np.concatenate(gr)
            self._G_cols = np.concatenate(gc)
            self._G_vals = np.concatenate(gv)
            self._G_elem = np.concatenate(ge)
            self._N0 = np.zeros(len(self.linear_frames))

    def _build_pattern(self):
        """Fixed sparsity pattern of the free tangent and the scatter maps of every
        element (element DOFs -> free equations through the constraint matrix)."""
        Af = self.Afree.tocsr()
        n = self.n
        keys = []
        K = self.K_lin.tocoo()
        keys.append(K.row.astype(np.int64) * n + K.col)
        Mc = self.M.tocoo()
        keys.append(Mc.row.astype(np.int64) * n + Mc.col)
        for e in self.elements:
            sub = Af[e.dofs]
            cols = np.unique(sub.indices)
            e._cols = cols
            e._E = sub[:, cols].toarray()  # (12, m)
            r, c = np.meshgrid(cols, cols, indexing="ij")
            keys.append(r.ravel().astype(np.int64) * n + c.ravel())
        if self._pd_avail:
            Gf = []
            for k, fe in enumerate(self.linear_frames):
                d = self.asm._element_dofs(fe.el.nodes)
                sub = Af[d]
                cols = np.unique(sub.indices)
                E = sub[:, cols].toarray()
                G = self._G_vals[self._G_elem == k].reshape(12, 12)
                Gf.append((cols, E.T @ G @ E))
                r, c = np.meshgrid(cols, cols, indexing="ij")
                keys.append(r.ravel().astype(np.int64) * n + c.ravel())
        allk = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
        rows, cols = allk // n if n else allk, allk % n if n else allk
        self._P_keys = allk
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        self._P_indptr = np.cumsum(indptr)
        self._P_indices = cols.astype(np.int64)
        self._K_lin_data = np.zeros(len(allk))
        self._K_lin_data[np.searchsorted(allk, K.row.astype(np.int64) * n + K.col)] += 0.0
        np.add.at(self._K_lin_data, np.searchsorted(allk, K.row.astype(np.int64) * n + K.col), K.data)
        self.M_data = np.zeros(len(allk))
        np.add.at(self.M_data, np.searchsorted(allk, Mc.row.astype(np.int64) * n + Mc.col), Mc.data)
        for e in self.elements:
            r, c = np.meshgrid(e._cols, e._cols, indexing="ij")
            e._pos = np.searchsorted(allk, r.ravel().astype(np.int64) * n + c.ravel())
        if self._pd_avail:
            gpos, gval, gel = [], [], []
            for k, (cols_k, Gk) in enumerate(Gf):
                r, c = np.meshgrid(cols_k, cols_k, indexing="ij")
                gpos.append(np.searchsorted(allk, r.ravel().astype(np.int64) * n + c.ravel()))
                gval.append(Gk.ravel())
                gel.append(np.full(Gk.size, k))
            self._Gf_pos = np.concatenate(gpos)
            self._Gf_val = np.concatenate(gval)
            self._Gf_el = np.concatenate(gel)
            self._N_free = (self._N_mat @ self.Afree).tocsr()
            self._Kg_free_pattern = None

    def set_geometry(self, geometry: str):
        """Geometric nonlinearity of the next analysis: "None", "PDelta" or
        "Corotational" (large displacements; every frame must then be an element
        of the domain, see :class:`NonlinearRunner`)."""
        g = (geometry or "None").lower().replace("-", "").replace(" ", "").replace("_", "")
        if g in ("largedisplacement", "largedisplacements", "pdeltalargedisplacements", "corot"):
            g = "corotational"
        if g not in ("none", "pdelta", "linear", "corotational"):
            raise ValueError(f"Geometric nonlinearity {geometry!r} is not supported (None, PDelta, Corotational)")
        if g == "corotational" and self.linear_frames:
            raise ValueError("Corotational analysis needs every frame as a nonlinear element; start the chain of "
                             "cases with geometric nonlinearity 'Corotational'")
        self.geometry = {"pdelta": "pdelta", "corotational": "corotational"}.get(g, "none")
        self._pd = self.geometry == "pdelta" and self._pd_avail
        for e in self.nl_frames.values():
            e.set_geometry({"pdelta": "pdelta", "corotational": "corotational"}.get(self.geometry, "linear"))
        self._K_init = None

    def _csr(self, data):
        return sparse.csr_matrix((data, self._P_indices, self._P_indptr), shape=(self.n, self.n))

    # ------------------------------------------------------------------ loads
    def _pattern_vector(self, pattern: str) -> np.ndarray:
        """Full load vector of a pattern without the span loads of nonlinear frames
        (those are carried inside the elements)."""
        if pattern not in self._pattern_cache:
            F = self.asm.load_vector_full(pattern).copy()
            self._total_cache = getattr(self, "_total_cache", {})
            self._total_cache[pattern] = self.AfreeT @ F  # every load as equivalent joint load
            sw = self.am.self_weight.get(pattern, 0.0)
            per_elem = {}
            for eid, e in self.nl_frames.items():
                loads = e.fe.element_loads(pattern, sw, self.am.gravity)
                if loads:
                    f, _ = e.fe.load_vector_global(loads)
                    F[e.dofs] -= f
                    per_elem[eid] = loads
            self._pattern_cache[pattern] = F
            self._elem_loads_cache[pattern] = per_elem
            if self._pd_avail:
                n0 = np.zeros(len(self.linear_frames))
                for k, fe in enumerate(self.linear_frames):
                    loads = fe.element_loads(pattern, sw, self.am.gravity)
                    if loads:
                        _, fc, _ = fe.condense(fe.local_stiffness_unreleased(), fe.fixed_end_load_vector(loads))
                        n0[k] = fc[0]
                self._n0_cache = getattr(self, "_n0_cache", {})
                self._n0_cache[pattern] = n0
        return self._pattern_cache[pattern]

    def set_factors(self, factors: Dict[str, float]):
        """Current pattern multipliers (gravity held constant, lateral scaled ...)."""
        self.factors = {p: f for p, f in factors.items() if f != 0.0}
        F = np.zeros(self.n_full)
        for p, f in self.factors.items():
            F += f * self._pattern_vector(p)
        self.F_full = F
        self.F = self.AfreeT @ F
        Ft = np.zeros(self.n)
        for p, f in self.factors.items():
            Ft += f * self._total_cache[p]
        # reference force for relative convergence tests (span loads of nonlinear members included)
        self.load_norm = float(np.linalg.norm(Ft))
        loads: Dict[int, List[ElementLoad]] = {}
        for p, f in self.factors.items():
            for eid, lds in self._elem_loads_cache[p].items():
                loads.setdefault(eid, []).extend(_scale_load(ld, f) for ld in lds)
        for eid, e in self.nl_frames.items():
            e.set_loads(loads.get(eid, []))
        self.elem_loads = loads
        if self._pd_avail:
            n0 = np.zeros(len(self.linear_frames))
            for p, f in self.factors.items():
                n0 += f * self._n0_cache[p]
            self._N0 = n0

    def frame_loads_scaled(self) -> Dict[int, list]:
        """Span loads of every frame element for the current factors (for station forces)."""
        out: Dict[int, list] = {}
        for p, f in self.factors.items():
            sw = self.am.self_weight.get(p, 0.0)
            for fe in self.asm.frames:
                for ld in fe.element_loads(p, sw, self.am.gravity):
                    out.setdefault(fe.el.id, []).append(_scale_load(ld, f))
        return out

    # --------------------------------------------------------------- response
    def full(self, u_free: np.ndarray) -> np.ndarray:
        return self.Afree @ u_free

    def resisting(self, u: np.ndarray, v: Optional[np.ndarray] = None, tangent: bool = True):
        """Internal force R(u) (free) and tangent K (sparse, free).  Also returns the
        element damping tangent (rate dependent links) or None."""
        R = self.K_lin @ u
        data = self._K_lin_data.copy() if tangent else None
        cdata = None
        for e in self.elements:
            E = e._E
            ue = E @ u[e._cols]
            ve = None if v is None else E @ v[e._cols]
            e.set_trial(ue, ve)
            R[e._cols] += E.T @ e.p
            if tangent:
                data[e._pos] += (E.T @ e.k @ E).ravel()
                c = getattr(e, "c", None)
                if c is not None:
                    if cdata is None:
                        cdata = np.zeros_like(data)
                    cdata[e._pos] += (E.T @ c @ E).ravel()
        if self._pd:
            N = self._N_free @ u + self._N0
            self._N_lin = N
            gdata = np.bincount(self._Gf_pos, self._Gf_val * N[self._Gf_el], minlength=len(self._P_keys))
            R += self._csr(gdata) @ u
            if tangent:
                data += gdata
        self._u_last = u
        if not tangent:
            return R, None, None
        return R, self._csr(data), (self._csr(cdata) if cdata is not None else None)

    def resisting_full(self, u: np.ndarray) -> np.ndarray:
        """Internal force over the full DOF set (for reactions); uses the element
        states of the last :meth:`resisting` call at ``u``."""
        uf = self.Afree @ u
        R = self.K_lin_full @ uf
        for e in self.elements:
            R[e.dofs] += e.p
        if self._pd:
            N = self._N_free @ u + self._N0
            gv = self._G_vals * N[self._G_elem]
            Kg = sparse.coo_matrix((gv, (self._G_rows, self._G_cols)), shape=(self.n_full,) * 2).tocsr()
            R += Kg @ uf
        return R

    def initial_stiffness(self) -> sparse.csr_matrix:
        if self._K_init is None:
            data = self._K_lin_data.copy()
            for e in self.elements:
                E = e._E
                data[e._pos] += (E.T @ e.initial_stiffness_global() @ E).ravel()
            self._K_init = self._csr(data)
        return self._K_init

    # ----------------------------------------------------------------- state
    def commit(self):
        for e in self.elements:
            e.commit()
            if hasattr(e, "commit_geometry"):
                e.commit_geometry()
        self.u_c = self.u.copy()

    def revert(self):
        for e in self.elements:
            e.revert()
        self.u = self.u_c.copy()

    def snapshot(self) -> "NonlinearDomain":
        return _copy.deepcopy(self)

    # -------------------------------------------------------------- recovery
    def reactions(self, u_full: np.ndarray, R_full: np.ndarray, F_full: np.ndarray,
                  inertia_full: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
        """Support reactions (global, force the support applies to the structure) for
        restrained joints, joint springs and grounded links."""
        am = self.am
        res = R_full - F_full
        if inertia_full is not None:
            res = res + inertia_full
        out: Dict[int, np.ndarray] = {}
        for node in am.nodes:
            if np.any(node.restraints) or np.any(node.springs):
                Tn = node_transform(node)
                sl = slice(6 * node.id, 6 * node.id + 6)
                r_l = Tn @ res[sl]
                u_l = Tn @ u_full[sl]
                r = np.zeros(6)
                sel = node.restraints.astype(bool)
                r[sel] = r_l[sel]
                for k in range(6):
                    if node.springs[k] > 0 and not node.restraints[k]:
                        r[k] = -node.springs[k] * u_l[k]
                out[node.id] = Tn.T @ r
        for lk in self.links:
            if len(lk.nodes) == 1:
                nid = lk.nodes[0]
                out[nid] = out.get(nid, np.zeros(6)) - lk.p
        return out

    def frame_end_forces(self, u_full: np.ndarray) -> np.ndarray:
        """Local end forces (n_frames, 12) of every frame element at the current state."""
        out = np.zeros((len(self.am.frames), 12))
        loads = self.frame_loads_scaled()
        for fe in self.asm.frames:
            eid = fe.el.id
            if eid in self.nl_frames:
                out[eid] = self.nl_frames[eid].end_forces_local()
            else:
                d = self.asm._element_dofs(fe.el.nodes)
                out[eid] = fe.end_forces_local(u_full[d], loads.get(eid, []))
        return out

    def shell_resultants(self, u_full: np.ndarray) -> dict:
        out = {}
        for se in self.asm.shells:
            d = self.asm._element_dofs(se.el.nodes)
            out[se.el.id] = se.resultants(u_full[d])
        return out

    # ---- vertical hinge drops ----------------------------------------------
    def hinge_materials(self):
        out = []
        for e in self.nl_frames.values():
            for hg in getattr(e, "hinges", []):
                if getattr(hg.material, "vertical", None) is not None:
                    out.append(hg.material)
        return out

    def set_drop_mode(self, mode: str) -> int:
        """``redistribute`` (vertical drops released by the static solver) or
        ``regularise``; returns the number of hinges with vertical drops."""
        mats = self.hinge_materials()
        for m in mats:
            m.drop_mode = mode
        return len(mats)

    def pending_drops(self):
        """[(material, point, key, level)] for drops passed and not yet released."""
        return [(m, pt, key, lev) for m in self.hinge_materials() for pt, key, lev in m.pending_drops()]

    def hinge_results(self) -> List[dict]:
        out = []
        for e in self.nl_frames.values():
            if hasattr(e, "hinge_results"):
                out.extend(e.hinge_results())
        return out

    # ---------------------------------------------------------------- helpers
    def free_row(self, node: int, dof: int) -> np.ndarray:
        """Vector c with c . u_free = global displacement component ``dof`` of ``node``."""
        r = self.Afree.getrow(6 * node + dof).toarray().ravel()
        return r

    def influence(self, direction: str) -> np.ndarray:
        """Rigid body ground motion pattern in free coordinates (UX, UY, UZ)."""
        r_red = self.asm.influence_vector_reduced(direction)
        return r_red[self.dofs.free]


__all__ = ["NonlinearDomain", "ElementNonConvergence"]
