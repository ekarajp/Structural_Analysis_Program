"""Global assembly: DOF numbering, constraint transformation, stiffness,
mass and load vectors.

DOF layout: every node (including virtual constraint masters) owns 6
consecutive *full* DOFs ``6*node + d``.  Rigid diaphragm constraints are
applied with a master-slave transformation ``u_full = T u_red`` where the
reduced vector omits the constrained slave DOFs.  Inactive and restrained
DOFs are removed afterwards to obtain the *free* equation set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

from ..mesh.analysis_model import AnalysisModel
from ..model.base import ModelError
from .elements.frame import FrameFormulation
from .elements import shell_batch
from .elements.shell import ShellFormulation


def sum_csr(parts, n: int) -> sparse.csr_matrix:
    """Sum of CSR matrices, pairwise (bounded memory, O(nnz log k))."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return sparse.csr_matrix((n, n))
    while len(parts) > 1:
        nxt = []
        for i in range(0, len(parts), 2):
            nxt.append(parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i])
        parts = nxt
    K = parts[0].tocsr()
    K.sum_duplicates()
    return K


def node_transform(node) -> np.ndarray:
    """6x6 global -> joint-local transformation of a node (identity without local axes)."""
    if node.local_axes is None:
        return np.eye(6)
    R = np.asarray(node.local_axes, float)
    T = np.zeros((6, 6))
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


@dataclass
class DofMap:
    n_nodes: int
    n_full: int
    T: sparse.csr_matrix  # full x reduced
    reduced_of_full: np.ndarray  # full dof -> reduced index (-1 for slaves)
    free_of_reduced: np.ndarray  # reduced dof -> free equation index (-1 if fixed/inactive)
    free: np.ndarray  # reduced indices that are free
    fixed: np.ndarray  # reduced indices that are restrained (active)
    restrained_full: np.ndarray  # full dof indices that carry restraints

    @property
    def n_free(self) -> int:
        return len(self.free)


class Assembler:
    """Builds and caches global matrices for an :class:`AnalysisModel`."""

    def __init__(self, am: AnalysisModel, links=None):
        self.am = am
        # links of the linear analyses (linear stiffness + mass); see NonlinearBuilder.linear_links
        self.links = list(links or [])
        coords = am.coords()
        self.frames = [FrameFormulation(e, coords[e.node_i], coords[e.node_j]) for e in am.frames]
        self.shells = [ShellFormulation(e, coords[list(e.nodes)]) for e in am.shells]
        shell_batch.prepare(self.shells)  # areas / tributary areas, vectorised (no element matrices kept)
        self._K_full: Optional[sparse.csr_matrix] = None
        self._M_full: Optional[np.ndarray] = None
        self._place_masters_at_center_of_mass()
        self.dofs = self._number_dofs()

    def _place_masters_at_center_of_mass(self):
        """Move each rigid-diaphragm master to the centre of lateral mass of
        its constrained nodes (CSI automatic master joint).  With the master
        at the centre of mass the reduced mass matrix has no translation-
        rotation coupling, which keeps lumped-mass eigensolutions exact."""
        am = self.am
        if not am.constraints:
            return
        # node masses without the diaphragm corrections (they depend on the master position only
        # through the inertia term, which is invariant to the master location when applied at the CM)
        M = self._node_lateral_mass()
        for c in am.constraints:
            if not c.rigid:
                continue
            ids = list(c.nodes)
            masses = np.array([M[n] for n in ids])
            if masses.sum() <= 0:
                continue
            xy = np.array([am.nodes[n].xyz for n in ids])
            cm = (masses[:, None] * xy).sum(axis=0) / masses.sum()
            master = am.nodes[c.master]
            new_xyz = master.xyz.copy()
            for a in range(3):
                if a != c.normal:
                    new_xyz[a] = cm[a]
            master.xyz = new_xyz

    def _node_lateral_mass(self) -> np.ndarray:
        am = self.am
        M = np.zeros(am.n_nodes)
        if am.mass_include_elements:
            for fe in self.frames:
                mi, mj = fe.lumped_mass()
                M[fe.el.node_i] += mi
                M[fe.el.node_j] += mj
            for se in self.shells:
                for nid, m in zip(se.el.nodes, se.lumped_mass()):
                    M[nid] += m
            for nid, m in getattr(am, "area_node_mass", {}).items():
                M[nid] += m
            for lk in self.links:
                for nid in lk.nodes:
                    M[nid] += getattr(lk, "mass", 0.0) / len(lk.nodes)
        if am.mass_include_added:
            for node in am.nodes:
                M[node.id] += node.mass[0]
        if am.mass_from_loads:
            for pattern, mult in am.mass_from_loads.items():
                F = self._raw_load_vector(pattern)
                M += -F[2::6] * mult / am.gravity
        return self._lump_to_stories(M, stride=1)

    def _story_lumping(self):
        """[(node, [(target node, weight), ...])] for the joints between two story levels
        when the mass source lumps lateral mass at the stories (CSI "Lump lateral mass at
        story levels"): the mass is split to the levels below and above in inverse
        proportion to the distance (total mass and its first moment in Z are kept) and
        put at the joint of that level nearest in plan."""
        am = self.am
        if not am.mass_lump_at_stories or len(am.story_elevations) < 2:
            return []
        if getattr(self, "_lump_map", None) is not None:
            return self._lump_map
        xyz = am.coords()
        tol = 1e-4
        # levels that carry joints (a story with no joint has nothing to take the mass)
        levels = sorted(z for z in set(float(z) for z in am.story_elevations.values())
                        if np.any(np.abs(xyz[:, 2] - z) <= tol))
        if len(levels) < 2:
            self._lump_map = []
            return []
        at_level = {}
        for k, z in enumerate(levels):
            at_level[k] = np.where(np.abs(xyz[:, 2] - z) <= tol)[0]
        out = []
        for nid in range(am.n_nodes):
            z = xyz[nid, 2]
            if z <= levels[0] + tol or z >= levels[-1] - tol:
                continue
            k = int(np.searchsorted(levels, z))
            if abs(levels[k] - z) <= tol or abs(levels[k - 1] - z) <= tol:
                continue
            zb, za = levels[k - 1], levels[k]
            targets = []
            for lev, w in ((k - 1, (za - z) / (za - zb)), (k, (z - zb) / (za - zb))):
                cand = at_level[lev]
                if len(cand) == 0:
                    targets = []
                    break
                d = np.hypot(xyz[cand, 0] - xyz[nid, 0], xyz[cand, 1] - xyz[nid, 1])
                targets.append((int(cand[int(np.argmin(d))]), float(w)))
            if targets:
                out.append((nid, targets))
        self._lump_map = out
        return out

    def _lump_to_stories(self, M: np.ndarray, stride: int) -> np.ndarray:
        """Move the lateral (UX, UY) mass of joints between story levels to the levels."""
        lump = self._story_lumping()
        if not lump:
            return M
        comps = (0,) if stride == 1 else (0, 1)
        for nid, targets in lump:
            for c in comps:
                m = M[stride * nid + c]
                if m == 0.0:
                    continue
                M[stride * nid + c] = 0.0
                for t, w in targets:
                    M[stride * t + c] += w * m
        return M

    def _raw_load_vector(self, pattern: str) -> np.ndarray:
        """Load vector without constraint transformation (used before DOF numbering)."""
        am = self.am
        F = np.zeros(6 * am.n_nodes)
        sw = am.self_weight.get(pattern, 0.0)
        for fe in self.frames:
            loads = fe.element_loads(pattern, sw, am.gravity)
            if loads:
                f, _ = fe.load_vector_global(loads)
                F[self._element_dofs(fe.el.nodes)] += f
        for se in self.shells:
            f = se.load_vector_global(pattern, sw)
            if np.any(f):
                F[self._element_dofs(se.el.nodes)] += f
        for nid, vec in am.nodal_loads.get(pattern, {}).items():
            F[6 * nid:6 * nid + 6] += vec
        return F

    # -------------------------------------------------------------- numbering
    def support_residual(self, residual: np.ndarray):
        """``f(node, Tn)`` -> the residual force (6,) on a node in its joint local
        axes, with what the constrained nodes hanging on it pass to it (nodes tied
        into a shell at a support, the slab rigid zone of a supported column, a
        rigid link to a support): the reaction of a supported DOF is the residual of
        its *reduced* equation, T^T (K u - F), not of the node's own row alone."""
        dofs = self.dofs
        rho = dofs.T.T @ residual
        red = dofs.reduced_of_full

        def at(node, Tn):
            sl = slice(6 * node.id, 6 * node.id + 6)
            out = Tn @ residual[sl]
            idx = red[sl]
            keep = idx >= 0
            out[keep] = rho[idx[keep]]
            return out
        return at

    def _rigid_link_defs(self, defs: Dict[int, List], restrained_nodes) -> None:
        """Rigid links: every group of nodes joined by links moves as one rigid body
        (exact, no stiffness).  One master per group - a supported node, else a
        diaphragm node, else the first - and the others follow it in all six DOFs
        (a DOF already constrained by a diaphragm / tie / rigid zone, or supported,
        keeps that)."""
        am = self.am
        links = getattr(am, "rigid_links", [])
        if not links:
            return
        parent: Dict[int, int] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for lk in links:
            if lk.node_i == lk.node_j:
                continue
            ra, rb = find(lk.node_i), find(lk.node_j)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        groups: Dict[int, List[int]] = {}
        for x in list(parent):
            groups.setdefault(find(x), []).append(x)
        any_restraint = {node.id for node in am.nodes if np.any(node.restraints)}
        for members in groups.values():
            members = sorted(members)
            gset = set(members)
            # a constraint between two members of the group (the slab rigid zone of the
            # column a linked node lies in, a tie into an element of linked nodes) says
            # nothing the link does not: dropped, or the two would define each other
            for nid in members:
                for k in range(6):
                    terms = defs.get(6 * nid + k)
                    if terms and all(md // 6 in gset for md, _ in terms):
                        del defs[6 * nid + k]

            def refers_to_group(n):
                return any(md // 6 in gset for k in range(6) for md, _ in defs.get(6 * n + k, ()))

            def constrained(n):
                return any(6 * n + k in defs for k in range(6))
            # master: a supported node, else one whose own constraints (a diaphragm) do not
            # lead back into the group, else any - its constraints into the group then go
            master = min(members, key=lambda n: (n not in any_restraint, refers_to_group(n), not constrained(n), n))
            if refers_to_group(master):
                for k in range(6):
                    terms = defs.get(6 * master + k)
                    if terms and any(md // 6 in gset for md, _ in terms):
                        del defs[6 * master + k]
            xm = am.nodes[master].xyz
            m6 = 6 * master
            for nid in members:
                if nid == master:
                    continue
                node = am.nodes[nid]
                if node.local_axes is not None or am.nodes[master].local_axes is not None:
                    am.warnings.append(f"Rigid link at node {node.point or nid}: joint local axes - not linked")
                    continue
                dx, dy, dz = node.xyz - xm
                terms = (
                    [(m6 + 0, 1.0), (m6 + 4, dz), (m6 + 5, -dy)],
                    [(m6 + 1, 1.0), (m6 + 5, dx), (m6 + 3, -dz)],
                    [(m6 + 2, 1.0), (m6 + 3, dy), (m6 + 4, -dx)],
                    [(m6 + 3, 1.0)], [(m6 + 4, 1.0)], [(m6 + 5, 1.0)])
                skipped = False
                for k in range(6):
                    dof = 6 * nid + k
                    if dof in defs:
                        continue
                    if node.restraints[k]:
                        skipped = True
                        continue
                    defs[dof] = [(d, w) for d, w in terms[k] if w != 0.0]
                if skipped:
                    am.warnings.append(f"Rigid link: supported node {node.point or nid} keeps its support "
                                       f"(the link is not rigid in the supported DOFs)")

    def _number_dofs(self) -> DofMap:
        am = self.am
        n = am.n_nodes
        n_full = 6 * n
        active = np.tile(am.active_dofs, n)
        # virtual master nodes only own the constrained DOFs of their plane
        for c in am.constraints:
            if c.rigid:
                for a in range(6):
                    if a not in (0, 1, 2, 3, 4, 5) or a == c.normal or (a >= 3 and a != 3 + c.normal):
                        active[6 * c.master + a] = False
        # slave DOFs: rigid diaphragms and ties (auto edge constraints); each slave
        # full DOF is defined as a combination of other full DOFs, resolved below
        slave = np.zeros(n_full, dtype=bool)
        defs: Dict[int, List] = {}
        restrained_nodes = {node.id for node in am.nodes if np.any(node.restraints[:3])}
        for c in am.constraints:
            if not c.rigid:
                continue
            axes = [a for a in range(3) if a != c.normal]
            # a joint restrained in a DOF the diaphragm constrains (in-plane translation or the
            # rotation about the normal) cannot also follow it: dropped (CSI gives restraints
            # precedence).  A restraint the diaphragm does not touch (UZ, RX, RY of a floor)
            # leaves the joint in it.
            cdofs = (axes[0], axes[1], 3 + c.normal)
            dropped = [n for n in c.nodes if any(am.nodes[n].restraints[k] for k in cdofs)]
            if dropped:
                c.nodes = [n for n in c.nodes if n not in set(dropped)]
                am.warnings.append(f"Diaphragm {c.name}: {len(dropped)} joint(s) restrained in the diaphragm's plane "
                                   "removed from the constraint")
            master = am.nodes[c.master]
            xm = master.xyz
            a1, a2 = axes  # e.g. X, Y for a horizontal diaphragm
            rn = c.normal  # rotation about the normal
            m6 = 6 * c.master
            for nid in c.nodes:
                d = am.nodes[nid].xyz - xm
                # u_a1 = u_a1,m - r_n,m * d_a2 ; u_a2 = u_a2,m + r_n,m * d_a1 ; r_n = r_n,m
                defs[6 * nid + a1] = [(m6 + a1, 1.0), (m6 + 3 + rn, -d[a2])]
                defs[6 * nid + a2] = [(m6 + a2, 1.0), (m6 + 3 + rn, d[a1])]
                defs[6 * nid + 3 + rn] = [(m6 + 3 + rn, 1.0)]
        for tie in getattr(am, "ties", []):
            node = am.nodes[tie.node]
            if node.local_axes is not None or any(am.nodes[m].local_axes is not None for m, _ in tie.masters):
                am.warnings.append(f"Tie of node {node.point or node.id}: joint local axes - not tied")
                continue
            for k in (getattr(tie, "dofs", None) or range(6)):
                dof = 6 * tie.node + k
                if dof in defs or node.restraints[k]:
                    continue  # already constrained (diaphragm) / a support wins
                defs[dof] = [(6 * m + k, float(w)) for m, w in tie.masters]
        for body in getattr(am, "bodies", []):
            xm = am.nodes[body.master].xyz
            m6 = 6 * body.master
            for nid in body.nodes:
                node = am.nodes[nid]
                d = node.xyz - xm
                for k, terms in ((2, [(m6 + 2, 1.0), (m6 + 3, d[1]), (m6 + 4, -d[0])]),
                                 (3, [(m6 + 3, 1.0)]), (4, [(m6 + 4, 1.0)])):
                    dof = 6 * nid + k
                    if dof in defs or node.restraints[k]:
                        continue
                    defs[dof] = terms
        self._rigid_link_defs(defs, restrained_nodes)
        memo: Dict[int, Dict[int, float]] = {}

        def expand(dof: int, depth: int = 0) -> Dict[int, float]:
            if dof not in defs:
                return {dof: 1.0}
            if dof in memo:
                return memo[dof]
            if depth > 50:
                raise ModelError("CONSTRAINT_LOOP", f"Constraint loop at node {am.nodes[dof // 6].point or dof // 6}")
            out: Dict[int, float] = {}
            for md, w in defs[dof]:
                for k, v in expand(md, depth + 1).items():
                    out[k] = out.get(k, 0.0) + w * v
            memo[dof] = out
            return out
        rows, cols, vals = [], [], []
        for dof in defs:
            slave[dof] = True
        for dof in defs:
            for k, v in expand(dof).items():
                if v != 0.0:
                    rows.append(dof)
                    cols.append(k)
                    vals.append(v)
        reduced_of_full = -np.ones(n_full, dtype=int)
        keep = np.where(~slave)[0]
        reduced_of_full[keep] = np.arange(len(keep))
        # joint local axes: the reduced DOFs of such a node are its local components,
        # u_global = R^T u_local (restraints and springs then act in the local axes)
        local_nodes = set()
        for node in am.nodes:
            if node.local_axes is None:
                continue
            if slave[6 * node.id:6 * node.id + 6].any():
                am.warnings.append(f"Joint {node.point or node.id}: local axes ignored on a diaphragm joint")
                node.local_axes = None
                continue
            local_nodes.add(node.id)
            Rt = np.asarray(node.local_axes, float).T
            for blk in (0, 3):
                for i in range(3):
                    for j in range(3):
                        if Rt[i, j] != 0.0:
                            rows.append(6 * node.id + blk + i)
                            cols.append(6 * node.id + blk + j)
                            vals.append(float(Rt[i, j]))
        plain = [int(k) for k in keep if (k // 6) not in local_nodes]
        rows += plain
        cols += plain
        vals += [1.0] * len(plain)
        # map master columns (full index) to reduced index
        cols_red = reduced_of_full[np.array(cols, dtype=int)]
        if np.any(cols_red < 0):
            raise ModelError("CONSTRAINT", "A constraint refers to a constrained DOF")
        T = sparse.csr_matrix((vals, (rows, cols_red)), shape=(n_full, len(keep)))

        # restraints & activity on the reduced set
        restrained = np.zeros(n_full, dtype=bool)
        for node in am.nodes:
            restrained[6 * node.id:6 * node.id + 6] = node.restraints
        # a restrained slave DOF: allowed only if the DOF is inactive, otherwise error
        bad = np.where(restrained & slave & active)[0]
        if bad.size:
            nid = int(bad[0] // 6)
            raise ModelError("RESTRAINED_CONSTRAINED_DOF",
                             f"Node {am.nodes[nid].point or nid} has a restraint on a DOF constrained by a "
                             f"rigid diaphragm; remove the restraint or the diaphragm assignment")
        # masters: inactive DOFs of masters follow the global active flags; masters are never restrained
        red_restrained = restrained[keep]
        red_active = active[keep]
        free_mask = red_active & ~red_restrained
        free = np.where(free_mask)[0]
        fixed = np.where(red_active & red_restrained)[0]
        free_of_reduced = -np.ones(len(keep), dtype=int)
        free_of_reduced[free] = np.arange(len(free))
        return DofMap(n, n_full, T, reduced_of_full, free_of_reduced, free, fixed, np.where(restrained)[0])

    # --------------------------------------------------------------- matrices
    def _element_dofs(self, nodes) -> np.ndarray:
        return np.concatenate([np.arange(6 * n, 6 * n + 6) for n in nodes])

    def stiffness_full(self) -> sparse.csr_matrix:
        if self._K_full is not None:
            return self._K_full
        am = self.am
        n_full = self.dofs.n_full
        rows, cols, vals = [], [], []
        for fe in self.frames:
            K = fe.stiffness_global()
            d = self._element_dofs(fe.el.nodes)
            r, c = np.meshgrid(d, d, indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(K.ravel())
        # nodal springs (in the joint local axes when the joint has them)
        sp_rows, sp_cols, sp_vals = [], [], []
        for node in am.nodes:
            if not np.any(node.springs):
                continue
            Tn = node_transform(node)
            Kn = Tn.T @ np.diag(node.springs) @ Tn
            r, c = np.nonzero(Kn)
            sp_rows += (6 * node.id + r).tolist()
            sp_cols += (6 * node.id + c).tolist()
            sp_vals += Kn[r, c].tolist()
        for lk in self.links:  # linear (effective) link stiffness
            K = lk.initial_stiffness_global()
            d = self._element_dofs(lk.nodes)
            r, c = np.meshgrid(d, d, indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(K.ravel())
        if sp_rows:
            rows.append(np.array(sp_rows))
            cols.append(np.array(sp_cols))
            vals.append(np.array(sp_vals))
        parts = []
        if rows:
            parts.append(sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                           shape=(n_full, n_full)).tocsr())
        parts += list(self.shell_stiffness_parts())
        K = sum_csr(parts, n_full)
        self._K_full = K
        return K

    def shell_stiffness_parts(self, shells=None):
        """Global stiffness of the shells as CSR matrices, one per chunk of
        elements (vectorised; the element matrices are not kept)."""
        n_full = self.dofs.n_full
        shells = self.shells if shells is None else shells
        idx = np.int32 if n_full < 2 ** 31 - 1 else np.int64
        for d, Kg in shell_batch.global_stiffness(shells):
            m = len(d)
            r = np.repeat(d, 24, axis=1).astype(idx, copy=False).ravel()
            c = np.tile(d, (1, 24)).astype(idx, copy=False).ravel()
            yield sparse.coo_matrix((Kg.ravel(), (r, c)), shape=(n_full, n_full)).tocsr()
            del r, c, Kg

    def shell_resultants(self, u_full: np.ndarray, shells=None) -> dict:
        """Element id -> stress resultants of every shell (vectorised)."""
        return shell_batch.resultants(self.shells if shells is None else shells, u_full)

    def stiffness_reduced(self) -> sparse.csr_matrix:
        T = self.dofs.T
        K = (T.T @ self.stiffness_full() @ T).tocsr()
        self._restrain_null_dofs(K)
        return K

    def _restrain_null_dofs(self, K_red: sparse.csr_matrix):
        """Free DOFs with no stiffness at all (e.g. the rotation of a joint
        where every connected member is released) would make the system
        singular.  Like SAP2000/ETABS they are automatically restrained and
        reported as warnings."""
        d = np.asarray(K_red.diagonal()).ravel()
        # "no stiffness": exactly zero, or round-off left by the transformations (a membrane's
        # out-of-plane / drilling terms ~1e-22 beside ~1e+8: numerically a mechanism).  Measured
        # against the median diagonal term (not the largest: a penalty / rigid term ~1e+26
        # would take real stiffnesses with it)
        free_d = np.abs(d[self.dofs.free]) if len(self.dofs.free) else np.zeros(0)
        floor = 1e-18 * float(np.median(free_d[free_d > 0])) if np.any(free_d > 0) else 0.0
        null = [i for i in self.dofs.free if abs(d[i]) <= floor]
        if not null:
            return
        dofs = self.dofs
        keep = np.where(dofs.reduced_of_full >= 0)[0]
        # the out-of-plane DOFs inside a membrane floor have no stiffness by definition
        # (its loads went to the frames): restrained too, reported once, not per node
        membrane, other = set(), set()
        for se in self.shells:
            (membrane if se.el.model_type == "Membrane" else other).update(se.el.nodes)
        for fe in self.frames:
            other.update((fe.el.node_i, fe.el.node_j))
        quiet = 0
        loose_corners = set(getattr(self.am, "quiet_nodes", ()) or ())
        quiet_corners = set()
        for i in null:
            full = int(keep[i])
            node, comp = divmod(full, 6)
            n = self.am.nodes[node]
            if comp in (2, 3, 4) and node in membrane and node not in other and n.point is None:
                quiet += 1
                continue
            if node in loose_corners:
                quiet_corners.add(node)
                continue
            label = n.point or f"node {node}"
            self.am.warnings.append(f"Null DOF {('UX','UY','UZ','RX','RY','RZ')[comp]} at {label} has no stiffness; "
                                    f"automatically restrained")
        if quiet:
            self.am.info.append(f"Membrane floors: {quiet} out-of-plane DOF(s) of their inner mesh nodes held "
                                f"(a membrane has no bending stiffness; its loads go to the supporting frames)")
        if quiet_corners:
            self.am.info.append(f"Membrane floors without elements (rigid diaphragm): {len(quiet_corners)} corner "
                                f"joint(s) that nothing else uses held (the floor's loads and mass are on the frames)")
        null_set = set(null)
        dofs.free = np.array([i for i in dofs.free if i not in null_set], dtype=int)
        dofs.free_of_reduced[:] = -1
        dofs.free_of_reduced[dofs.free] = np.arange(len(dofs.free))

    def mass_full(self) -> np.ndarray:
        """Diagonal lumped mass over the full DOF set (kg, kg·m²)."""
        if self._M_full is not None:
            return self._M_full
        am = self.am
        M = np.zeros(self.dofs.n_full)
        if am.mass_include_elements:
            for fe in self.frames:
                mi, mj = fe.lumped_mass()
                for nid, m in ((fe.el.node_i, mi), (fe.el.node_j, mj)):
                    M[6 * nid:6 * nid + 3] += m
            for se in self.shells:
                ms = se.lumped_mass()
                for nid, m in zip(se.el.nodes, ms):
                    M[6 * nid:6 * nid + 3] += m
            for nid, m in getattr(am, "area_node_mass", {}).items():
                M[6 * nid:6 * nid + 3] += m
            for lk in self.links:  # link mass: equal shares at its joints (CSI)
                for nid in lk.nodes:
                    M[6 * nid:6 * nid + 3] += getattr(lk, "mass", 0.0) / len(lk.nodes)
        if am.mass_include_added:
            for node in am.nodes:
                M[6 * node.id:6 * node.id + 6] += node.mass
        if am.mass_from_loads:
            # gravity loads (global -Z) of the given patterns / g -> translational mass
            for pattern, mult in am.mass_from_loads.items():
                F = self.load_vector_full(pattern)
                fz = F[2::6]
                m_add = -fz * mult / am.gravity
                for nid in range(am.n_nodes):
                    M[6 * nid:6 * nid + 3] += m_add[nid]
        # analytical inertia corrections for area objects lumped into rigid diaphragms
        for c in am.constraints:
            if not c.rigid:
                continue
            extra = c.inertia_correction_self if am.mass_include_elements else 0.0
            for pattern, mult in am.mass_from_loads.items():
                extra += mult * c.inertia_correction_loads.get(pattern, 0.0)
                sw = am.self_weight.get(pattern, 0.0)
                if sw:
                    extra += mult * sw * c.inertia_correction_loads.get("__self_weight__", 0.0)
            if am.mass_lateral:
                M[6 * c.master + 3 + c.normal] += extra
        self._lump_to_stories(M, stride=6)
        if not am.mass_lateral:
            M[0::6] = 0.0
            M[1::6] = 0.0
        if not am.mass_vertical:
            M[2::6] = 0.0
        self._M_full = M
        return M

    def mass_reduced(self) -> sparse.csr_matrix:
        T = self.dofs.T
        Mf = sparse.diags(self.mass_full())
        return (T.T @ Mf @ T).tocsr()

    # ------------------------------------------------------------------ loads
    def load_vector_full(self, pattern: str) -> np.ndarray:
        am = self.am
        F = np.zeros(self.dofs.n_full)
        sw = am.self_weight.get(pattern, 0.0)
        for fe in self.frames:
            loads = fe.element_loads(pattern, sw, am.gravity)
            if not loads:
                continue
            f, _ = fe.load_vector_global(loads)
            F[self._element_dofs(fe.el.nodes)] += f
        for se in self.shells:
            f = se.load_vector_global(pattern, sw)
            if np.any(f):
                F[self._element_dofs(se.el.nodes)] += f
        for nid, vec in am.nodal_loads.get(pattern, {}).items():
            F[6 * nid:6 * nid + 6] += vec
        # ground displacement through springs: the ground end of a spring moves by dg,
        # which loads the joint with k * dg (local axes)
        for node in am.nodes:
            dg = node.ground_disp.get(pattern)
            if dg is not None and np.any(node.springs):
                F[6 * node.id:6 * node.id + 6] += node_transform(node).T @ (node.springs * dg)
        return F

    def prescribed_reduced(self, loads: Dict[str, float]) -> Optional[np.ndarray]:
        """Reduced displacement vector holding the ground displacements of the
        restrained DOFs for a combination of load patterns (None when there are none)."""
        dofs = self.dofs
        u = np.zeros(dofs.T.shape[1])
        any_ = False
        for node in self.am.nodes:
            if not node.ground_disp or not np.any(node.restraints):
                continue
            for pat, sf in loads.items():
                dg = node.ground_disp.get(pat)
                if dg is None:
                    continue
                for d in range(6):
                    if node.restraints[d] and dg[d]:
                        r = dofs.reduced_of_full[6 * node.id + d]
                        if r >= 0:
                            u[r] += sf * dg[d]
                            any_ = True
        return u if any_ else None

    def load_vector_reduced(self, pattern: str) -> np.ndarray:
        return self.dofs.T.T @ self.load_vector_full(pattern)

    # -------------------------------------------------------------- influence
    def influence_vector_full(self, direction: str, center: Optional[np.ndarray] = None) -> np.ndarray:
        """Rigid-body displacement pattern for unit ground motion.

        ``direction``: "UX", "UY", "UZ" or "RX", "RY", "RZ" (rotation about
        the global axis through ``center``)."""
        am = self.am
        r = np.zeros(self.dofs.n_full)
        coords = am.coords()
        if direction in ("UX", "UY", "UZ"):
            d = ("UX", "UY", "UZ").index(direction)
            r[d::6] = 1.0
            return r
        axis = ("RX", "RY", "RZ").index(direction)
        c = np.zeros(3) if center is None else center
        e = np.zeros(3)
        e[axis] = 1.0
        # all nodes at once: np.cross row by row gives the same numbers
        disp = np.cross(e[None, :], coords - c)
        r[0::6] = disp[:, 0]
        r[1::6] = disp[:, 1]
        r[2::6] = disp[:, 2]
        r[3 + axis::6] = 1.0
        return r

    def influence_vector_reduced(self, direction: str, center: Optional[np.ndarray] = None) -> np.ndarray:
        """Influence vector expressed in reduced coordinates (slaves follow
        masters automatically)."""
        r_full = self.influence_vector_full(direction, center)
        keep = np.where(self.dofs.reduced_of_full >= 0)[0]
        return r_full[keep]
