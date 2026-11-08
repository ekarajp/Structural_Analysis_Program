"""Batched (vectorised) evaluation of :class:`ShellFormulation` for many shells.

The same element as ``shell.py`` - membrane with QM6 incompatible modes and a
drilling penalty, DKQ / DKMQ / MITC4 plate - evaluated for a whole block of
elements at once with NumPy arrays instead of one Python call per element and
Gauss point.  Every formula mirrors the per-element code line by line; the
results agree with it to round-off (tests/unit/test_shell_batch.py).

Only the default formulation switches of ``shell.py`` are vectorised (no
Allman drilling interpolation, QM6 modes on, drilling penalty at the 2x2
points); :func:`supported` is False otherwise and callers use the element
code.  Work is done in chunks of ``CHUNK`` elements so that temporary arrays
stay small whatever the size of the model.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from . import shell as S

CHUNK = 4096
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_GP = [(xi, eta) for xi in S._G2 for eta in S._G2]  # same order as the element loops
_GAUSS_LIST = [(float(a), float(b)) for a, b in _GP]


def _T(a: np.ndarray) -> np.ndarray:
    return np.swapaxes(a, -1, -2)


def supported() -> bool:
    """True when the formulation switches are the ones vectorised here."""
    return (not S.ALLMAN and S.INCOMPATIBLE and not S.DRILL_ONE_POINT and not S.DRILLING_FACTOR_1PT
            and S.DRILL_INCOMPATIBLE)


def chunks(n: int, size: int = CHUNK) -> Iterator[slice]:
    for a in range(0, n, size):
        yield slice(a, min(n, a + size))


# ----------------------------------------------------------------- constants
def _shape_const(xi, eta):
    N, dN = S._shape(xi, eta)
    return N, dN


_SHAPES = {p: _shape_const(*p) for p in _GP + [(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)]}
_SER = {p: S._serendipity_derivatives(*p) for p in _GP}


class Block:
    """Geometry and section data of a list of shell formulations (one chunk)."""

    def __init__(self, shells: Sequence["S.ShellFormulation"]):
        self.shells = list(shells)
        n = len(self.shells)
        self.n = n
        self.xy = np.empty((n, 4, 2))
        self.axes = np.empty((n, 3, 3))
        E = np.empty(n)
        nu = np.empty(n)
        G = np.empty(n)
        t = np.empty(n)
        tb = np.empty(n)
        mods = np.empty((n, 8))
        mtype = []
        for k, se in enumerate(self.shells):
            el = se.el
            self.xy[k] = se.xy
            self.axes[k] = se.axes
            E[k], nu[k], G[k] = el.E, el.nu, el.G
            t[k], tb[k] = el.thickness, el.bending_thickness
            mods[k] = el.modifiers
            mtype.append(el.model_type)
        self.mtype = np.array(mtype, dtype=object)
        self.G, self.t = G, t
        self.dkq = np.array([se.dkq for se in self.shells], dtype=bool)
        self.dkmq = np.array([se.dkmq for se in self.shells], dtype=bool)
        self.plate = self.mtype == "Plate"
        self.membrane = self.mtype == "Membrane"
        self.Dm, self.Db, self.Ds = self._constitutive(E, nu, G, t, tb, mods)

    # ------------------------------------------------------------ constitutive
    def _constitutive(self, E, nu, G, t, tb, mods):
        n = self.n
        f11, f22, f12, m11, m22, m12, v13, v23 = mods.T
        c = E / (1 - nu * nu)
        plane = np.zeros((n, 3, 3))
        plane[:, 0, 0] = c
        plane[:, 0, 1] = c * nu
        plane[:, 1, 0] = c * nu
        plane[:, 1, 1] = c
        plane[:, 2, 2] = c * ((1 - nu) / 2)
        Fm = np.zeros((n, 3, 3))
        Fm[:, 0, 0] = f11
        Fm[:, 0, 1] = Fm[:, 1, 0] = np.sqrt(f11 * f22)
        Fm[:, 1, 1] = f22
        Fm[:, 2, 2] = f12
        Fb = np.zeros((n, 3, 3))
        Fb[:, 0, 0] = m11
        Fb[:, 0, 1] = Fb[:, 1, 0] = np.sqrt(m11 * m22)
        Fb[:, 1, 1] = m22
        Fb[:, 2, 2] = m12
        Dm = plane * t[:, None, None] * Fm
        Db = plane * (tb ** 3 / 12.0)[:, None, None] * Fb
        Gs = G * (5.0 / 6.0) * tb
        Gs = np.where(self.mtype == "ShellThin", Gs * S.THIN_SHEAR_FACTOR, Gs)
        Ds = np.zeros((n, 2, 2))
        Ds[:, 0, 0] = Gs * v13
        Ds[:, 1, 1] = Gs * v23
        mem = self.mtype == "Membrane"
        Db[mem] = 0.0
        Ds[mem] = 0.0
        Dm[self.mtype == "Plate"] = 0.0
        return Dm, Db, Ds

    # ---------------------------------------------------------------- geometry
    def jacobian(self, xi, eta, check=True):
        N, dN = _SHAPES[(xi, eta)] if (xi, eta) in _SHAPES else S._shape(xi, eta)
        J = np.einsum("ij,njk->nik", dN, self.xy)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        if check and np.any(det <= 0.0):
            k = int(np.nonzero(det <= 0.0)[0][0])
            raise ValueError(f"shell element {self.shells[k].el.id}: non-positive Jacobian (check node ordering)")
        dNdx = np.linalg.solve(J, np.broadcast_to(dN, (self.n, 2, 4)))
        return N, dNdx, J, det

    def dkq_nodal(self):
        """(n, 8, 2, 24) slopes of the DKQ / DKMQ and (n, 4, 24) edge shears."""
        if getattr(self, "_T", None) is not None:
            return self._T, self._g
        n = self.n
        # phi of the DKMQ (zero for the DKQ)
        phi = np.zeros((n, 4))
        D = 0.5 * (self.Db[:, 0, 0] + self.Db[:, 1, 1])
        Sv = 0.5 * (self.Ds[:, 0, 0] + self.Ds[:, 1, 1])
        L = np.empty((n, 4))
        for k, (i, j) in enumerate(_EDGES):
            L[:, k] = np.linalg.norm(self.xy[:, j] - self.xy[:, i], axis=1)
        use = self.dkmq & (Sv > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ph = 12.0 * D[:, None] / (Sv[:, None] * L * L)
        phi[use] = ph[use]
        T = np.zeros((n, 8, 2, 24))
        for i in range(4):
            T[:, i, 0, 6 * i + 4] = -1.0
            T[:, i, 1, 6 * i + 3] = 1.0
        g = np.zeros((n, 4, 24))
        for k, (i, j) in enumerate(_EDGES):
            d = self.xy[:, j] - self.xy[:, i]
            Lk = np.linalg.norm(d, axis=1)
            sv = d / Lk[:, None]
            nv = np.stack([-sv[:, 1], sv[:, 0]], axis=1)
            bs_i = np.einsum("nc,ncd->nd", sv, T[:, i])
            bs_j = np.einsum("nc,ncd->nd", sv, T[:, j])
            bn_i = np.einsum("nc,ncd->nd", nv, T[:, i])
            bn_j = np.einsum("nc,ncd->nd", nv, T[:, j])
            ws = np.zeros((n, 24))
            ws[:, 6 * i + 2] = -1.5 / Lk
            ws[:, 6 * j + 2] = 1.5 / Lk
            f = phi[:, k][:, None]
            bs_mid = (ws + (0.5 * f - 0.25) * (bs_i + bs_j)) / (1.0 + f)
            g[:, k] = -(f / 3.0) * (bs_i + bs_j - 2.0 * bs_mid)
            bn_mid = 0.5 * (bn_i + bn_j)
            T[:, 4 + k] = sv[:, :, None] * bs_mid[:, None, :] + nv[:, :, None] * bn_mid[:, None, :]
        self._T, self._g, self._L = T, g, L
        return T, g

    def matrices(self, xi, eta):
        """N, Bm, Bb, Bs, Bd, det at one point for every element of the block."""
        n = self.n
        N, dNdx, J, det = self.jacobian(xi, eta)
        Bm = np.zeros((n, 3, 24))
        Bb = np.zeros((n, 3, 24))
        Bd = np.zeros((n, 1, 24))
        for i in range(4):
            dx, dy = dNdx[:, 0, i], dNdx[:, 1, i]
            j = 6 * i
            Bm[:, 0, j] = dx
            Bm[:, 1, j + 1] = dy
            Bm[:, 2, j] = dy
            Bm[:, 2, j + 1] = dx
            Bb[:, 0, j + 4] = dx
            Bb[:, 1, j + 3] = -dy
            Bb[:, 2, j + 4] = dy
            Bb[:, 2, j + 3] = -dx
            Bd[:, 0, j + 1] = 0.5 * dx
            Bd[:, 0, j] = -0.5 * dy
            Bd[:, 0, j + 5] = -N[i]

        def covariant(a, b, axis):
            Nt, dNt = _SHAPES[(a, b)]
            Jt = np.einsum("ij,njk->nik", dNt, self.xy)
            B = np.zeros((n, 24))
            for i in range(4):
                j = 6 * i
                B[:, j + 2] = dNt[axis, i]
                B[:, j + 4] = Nt[i] * Jt[:, axis, 0]
                B[:, j + 3] = -Nt[i] * Jt[:, axis, 1]
            return B

        cov = np.stack([
            0.5 * ((1 - eta) * covariant(0.0, -1.0, 0) + (1 + eta) * covariant(0.0, 1.0, 0)),
            0.5 * ((1 - xi) * covariant(-1.0, 0.0, 1) + (1 + xi) * covariant(1.0, 0.0, 1)),
        ], axis=1)
        Bs = np.linalg.solve(J, cov)
        kirch = self.dkq | self.dkmq
        if np.any(kirch):
            T, g = self.dkq_nodal()
            dS = np.linalg.solve(J, np.broadcast_to(_SER[(xi, eta)], (n, 2, 8)))
            bx = T[:, :, 0, :]
            by = T[:, :, 1, :]
            Bk = np.empty((n, 3, 24))
            sx = dS @ bx  # (n, 2, 24): rows d/dx, d/dy of bx
            sy = dS @ by
            Bk[:, 0] = -sx[:, 0]
            Bk[:, 1] = -sy[:, 1]
            Bk[:, 2] = -(sx[:, 1] + sy[:, 0])
            Bb[kirch] = Bk[kirch]
            Bs[self.dkq] = 0.0
            if np.any(self.dkmq):
                L = self._L
                g_xi = (0.5 * (1 - eta) * g[:, 0] * (L[:, 0] / 2)[:, None]
                        - 0.5 * (1 + eta) * g[:, 2] * (L[:, 2] / 2)[:, None])
                g_eta = (0.5 * (1 + xi) * g[:, 1] * (L[:, 1] / 2)[:, None]
                         - 0.5 * (1 - xi) * g[:, 3] * (L[:, 3] / 2)[:, None])
                Bsk = np.linalg.solve(J, np.stack([g_xi, g_eta], axis=1))
                Bs[self.dkmq] = Bsk[self.dkmq]
        return N, Bm, Bb, Bs, Bd, det

    def _incompatible(self, xi, eta, det, J0inv, det0):
        dNT = np.array([[-2.0 * xi, 0.0], [0.0, -2.0 * eta]])
        d = np.einsum("nij,jk->nik", J0inv, dNT) * (det0 / det)[:, None, None]
        Ba = np.zeros((self.n, 3, 4))
        Bda = np.zeros((self.n, 1, 4))
        for a in range(2):
            dx, dy = d[:, 0, a], d[:, 1, a]
            Ba[:, 0, 2 * a] = dx
            Ba[:, 1, 2 * a + 1] = dy
            Ba[:, 2, 2 * a] = dy
            Ba[:, 2, 2 * a + 1] = dx
            Bda[:, 0, 2 * a] = -0.5 * dy
            Bda[:, 0, 2 * a + 1] = 0.5 * dx
        return Ba, Bda

    # --------------------------------------------------------------- stiffness
    def stiffness_local(self):
        """(n, 24, 24) local stiffness, area (n,), tributary (n, 4)."""
        n = self.n
        Dm, Db, Ds = self.Dm, self.Db, self.Ds
        K = np.zeros((n, 24, 24))
        Kaa = np.zeros((n, 4, 4))
        Kua = np.zeros((n, 24, 4))
        area = np.zeros(n)
        trib = np.zeros((n, 4))
        drill = S.DRILLING_FACTOR * self.G * self.t
        inc = ~self.plate
        _, _, J0, det0 = self.jacobian(0.0, 0.0)
        J0inv = np.linalg.inv(J0)
        for xi, eta in _GP:
            N, Bm, Bb, Bs, Bd, det = self.matrices(xi, eta)
            w = det[:, None, None]
            BmT, DmBm = _T(Bm), Dm @ Bm
            K += (BmT @ DmBm + _T(Bb) @ (Db @ Bb) + _T(Bs) @ (Ds @ Bs)) * w
            wd = (drill * det)[:, None, None]
            BdT = _T(Bd)
            K += (BdT @ Bd) * wd
            Ba, Bda = self._incompatible(xi, eta, det, J0inv, det0)
            ka = (_T(Ba) @ (Dm @ Ba)) * w
            ku = (BmT @ (Dm @ Ba)) * w
            dz = (drill != 0)[:, None, None]
            ka = ka + np.where(dz, (_T(Bda) @ Bda) * wd, 0.0)
            ku = ku + np.where(dz, (BdT @ Bda) * wd, 0.0)
            Kaa[inc] += ka[inc]
            Kua[inc] += ku[inc]
            area += det
            trib += N[None, :] * det[:, None]
        cond = inc & (np.max(np.abs(Kaa), axis=(1, 2)) > 0)
        if np.any(cond):
            X = np.linalg.solve(Kaa[cond], np.transpose(Kua[cond], (0, 2, 1)))
            K[cond] -= Kua[cond] @ X
        for kind, dofs in (("Membrane", (3, 4)), ("Plate", (5,))):
            sel = self.mtype == kind
            if np.any(sel):
                diag = np.abs(np.diagonal(K[sel], axis1=1, axis2=2)).max(axis=1)
                scale = 1e-7 * np.maximum(diag, 1.0)
                for i in range(4):
                    for d in dofs:
                        K[sel, 6 * i + d, 6 * i + d] += scale
        K = 0.5 * (K + np.transpose(K, (0, 2, 1)))
        return K, area, trib

    def to_global(self, K: np.ndarray) -> np.ndarray:
        """R^T K R for every element (R = blockdiag of the 3x3 axes, 8 times)."""
        n = self.n
        A = self.axes
        tmp = K.reshape(n, 8, 3, 8, 3) @ A[:, None, None, :, :]  # sum_l K[a k b l] A[l j]
        tmp = tmp.reshape(n, 8, 3, 24)
        return (_T(A)[:, None, :, :] @ tmp).reshape(n, 24, 24)  # sum_k A[k i] tmp[a k .]

    # ---------------------------------------------------------------- recovery
    def resultants(self, u_global: np.ndarray) -> List[dict]:
        """Resultants of every element; ``u_global`` (n, 24)."""
        n = self.n
        u = np.einsum("nkj,naj->nak", self.axes, u_global.reshape(n, 8, 3)).reshape(n, 24)
        F = np.empty((n, 4, 3))
        M = np.empty((n, 4, 3))
        V = np.empty((n, 4, 2))
        for g, (xi, eta) in enumerate(_GP):
            _, Bm, Bb, Bs, _, _ = self.matrices(xi, eta)
            uc = u[:, :, None]
            F[:, g] = (self.Dm @ (Bm @ uc))[:, :, 0]
            M[:, g] = (self.Db @ (Bb @ uc))[:, :, 0]
            V[:, g] = (self.Ds @ (Bs @ uc))[:, :, 0]
        s = np.sqrt(3.0)
        ext = np.array([[S._shape(s * a, s * b)[0][k] for k in range(4)]
                        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        order = [0, 2, 3, 1]
        if np.any(self.dkq):
            # equilibrium shears of the thin plate: linear fit of the Gauss moments
            P = np.stack([np.einsum("i,nij->nj", _SHAPES[p][0], self.xy) for p in _GP], axis=1)  # (n, 4, 2)
            A = np.concatenate([np.ones((n, 4, 1)), P], axis=2)
            coef = np.linalg.pinv(A[self.dkq]) @ M[self.dkq]
            v13 = coef[:, 1, 0] + coef[:, 2, 2]
            v23 = coef[:, 1, 2] + coef[:, 2, 1]
            V[self.dkq] = np.stack([v13, v23], axis=1)[:, None, :] * S.SHEAR_SIGN
        M = -M
        Fn = np.einsum("ij,njk->nik", ext, F[:, order])
        Mn = np.einsum("ij,njk->nik", ext, M[:, order])
        Vn = np.einsum("ij,njk->nik", ext, V[:, order])
        return [{"gauss": _GAUSS_LIST, "F": F[k], "M": M[k], "V": V[k],
                 "F_nodes": Fn[k], "M_nodes": Mn[k], "V_nodes": Vn[k]} for k in range(n)]


# ---------------------------------------------------------------- front end
def element_dofs(shells: Sequence["S.ShellFormulation"]) -> np.ndarray:
    nodes = np.array([se.el.nodes for se in shells], dtype=np.int64).reshape(-1, 4)
    return (6 * nodes[:, :, None] + np.arange(6)[None, None, :]).reshape(-1, 24)


def prepare(shells: Sequence["S.ShellFormulation"]) -> None:
    """Fill area / tributary of every shell (all that mass and loads need)
    without keeping the element stiffness matrices."""
    if not supported():
        for se in shells:
            se.stiffness_local()
        return
    for sl in chunks(len(shells)):
        part = shells[sl]
        blk = Block(part)
        area = np.zeros(blk.n)
        trib = np.zeros((blk.n, 4))
        for xi, eta in _GP:
            N, _, _, det = blk.jacobian(xi, eta)
            area += det
            trib += N[None, :] * det[:, None]
        for k, se in enumerate(part):
            se.area = float(area[k])
            se.tributary = trib[k]
            se._prepared = True


def global_stiffness(shells: Sequence["S.ShellFormulation"]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """(dofs (m, 24), K_global (m, 24, 24)) chunk by chunk."""
    if not supported():
        for sl in chunks(len(shells)):
            part = shells[sl]
            yield element_dofs(part), np.stack([se.stiffness_global() for se in part])
        return
    for sl in chunks(len(shells)):
        part = shells[sl]
        blk = Block(part)
        K, area, trib = blk.stiffness_local()
        for k, se in enumerate(part):
            se.area = float(area[k])
            se.tributary = trib[k]
            se._prepared = True
        yield element_dofs(part), blk.to_global(K)


def resultants(shells: Sequence["S.ShellFormulation"], u_full: np.ndarray) -> Dict[int, dict]:
    """Element id -> resultants dict (as ShellFormulation.resultants)."""
    out: Dict[int, dict] = {}
    if not supported():
        for se in shells:
            d = element_dofs([se])[0]
            out[se.el.id] = se.resultants(u_full[d])
        return out
    for sl in chunks(len(shells)):
        part = shells[sl]
        d = element_dofs(part)
        blk = Block(part)
        for se, r in zip(part, blk.resultants(u_full[d])):
            out[se.el.id] = r
    return out


def zero_resultants(shells: Sequence["S.ShellFormulation"]) -> Dict[int, dict]:
    """Resultants of an unloaded case (all zero), without evaluating the elements."""
    n = len(shells)
    G = np.zeros((n, 4, 3))
    out: Dict[int, dict] = {}
    Z3 = [np.zeros((n, 4, 3)) for _ in range(5)]
    Z2 = [np.zeros((n, 4, 2)) for _ in range(2)]
    for k, se in enumerate(shells):
        out[se.el.id] = {"gauss": _GAUSS_LIST, "F": Z3[0][k], "M": Z3[1][k], "V": Z2[0][k],
                         "F_nodes": Z3[2][k], "M_nodes": Z3[3][k], "V_nodes": Z2[1][k]}
    del G
    return out

