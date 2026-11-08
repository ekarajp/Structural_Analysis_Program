"""Numerical kernel boundary.

The rest of the engine only talks to a :class:`Kernel` through
``factor``/``solve`` and ``eigen``.  The default implementation uses SciPy
(SuperLU, ARPACK, LAPACK).  Replace it with PARDISO/MKL (``pypardiso``) or a
compiled Rust/C++ kernel without touching assembly, elements or results.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, Protocol, Tuple

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh, splu

from ..model.base import ModelError


class Factorization(Protocol):
    def solve(self, b: np.ndarray) -> np.ndarray: ...


class Kernel(Protocol):
    name: str

    def factor(self, K: sparse.csr_matrix) -> Factorization: ...

    def eigen(self, K: sparse.csr_matrix, M: sparse.csr_matrix, n_modes: int, shift: float = 0.0,
              method: str = "auto") -> Tuple[np.ndarray, np.ndarray]: ...


UNSTABLE_PIVOT = 1e-12  # pivot / largest pivot below this: the equation is (nearly) singular


class _SciPyFactor:
    def __init__(self, K: sparse.csr_matrix):
        n = K.shape[0]
        if n == 0:
            self.lu = None
            return
        Kc = K.tocsc()
        try:
            # COLAMD kept on purpose: the pivot-based instability report (which joints /
            # DOFs) depends on the ordering, and must not change behind the user's back
            self.lu = splu(Kc, diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        except RuntimeError as exc:
            raise ModelError("SINGULAR_STIFFNESS",
                             "Stiffness matrix is singular: the structure is unstable (mechanism, missing "
                             "supports or zero-stiffness members)") from exc
        # pivot check: a mechanism shows up as a (relatively) vanishing pivot
        d = np.abs(self.lu.U.diagonal())
        self.pivot_ratio = float(np.min(d) / np.max(d)) if d.size else 1.0
        # equations that lost (nearly) all significant digits: a mechanism, like the
        # SAPFire "structure is unstable or ill-conditioned" check (free DOF indices)
        if d.size:
            dmax = float(np.max(d))
            bad = np.where(d <= UNSTABLE_PIVOT * dmax)[0]
            self.unstable = sorted(int(self.lu.perm_c[i]) for i in bad)
        else:
            self.unstable = []
        self.K = Kc

    def solve(self, b: np.ndarray) -> np.ndarray:
        if self.lu is None:
            return np.zeros_like(b)
        x = self.lu.solve(b)
        # a mechanism or severe ill-conditioning shows up as a large residual
        nb = float(np.linalg.norm(b))
        if nb > 0:
            res = float(np.linalg.norm(self.K @ x - b)) / nb
            if not np.isfinite(res) or res > 1e-6:
                raise ModelError("UNSTABLE_STRUCTURE",
                                 f"Equilibrium could not be satisfied (relative residual {res:.2e}): the "
                                 "structure is unstable (mechanism) or severely ill-conditioned")
        return x


class SciPyKernel:
    name = "scipy"

    def factor(self, K: sparse.csr_matrix) -> Factorization:
        return _SciPyFactor(K)

    def describe(self) -> str:
        return "SciPy SuperLU (single thread)"

    def eigen(self, K: sparse.csr_matrix, M: sparse.csr_matrix, n_modes: int, shift: float = 0.0,
              method: str = "auto", factor: Optional[Factorization] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Lowest ``n_modes`` generalized eigenpairs K phi = w^2 M phi.

        ``M`` is a (possibly singular) diagonal lumped mass matrix.  Massless
        DOFs are condensed statically.  Returns (omega2, phi) with phi mass
        normalised, columns = modes, over the *same* DOF set as K.

        ``factor``: a factorisation of this same K (the static cases'), reused
        for the condensation onto the massive DOFs instead of factoring the
        massless block again: with Y = K^-1 E_m (E_m the unit vectors of the
        massive DOFs), the condensed stiffness is Y_mm^-1 and the massless
        part of a mode -K_zz^-1 K_zm v = Y_zm Y_mm^-1 v (block inverse).
        """
        n = K.shape[0]
        M = sparse.csr_matrix(M)
        # DOFs with a non-zero row in M carry mass (M may have off-diagonal
        # coupling terms, e.g. eccentric mass under a rigid diaphragm)
        row_mass = np.asarray(abs(M).sum(axis=1)).ravel()
        m = np.asarray(M.diagonal()).ravel()
        massive = np.where(row_mass > 0)[0]
        massless = np.where(row_mass <= 0)[0]
        n_dyn = len(massive)
        if n_dyn == 0:
            raise ModelError("NO_MASS", "The model has no mass for dynamic analysis")
        n_modes = min(n_modes, n_dyn)
        if method == "auto":
            # few massive DOFs (rigid diaphragms lump the floor mass on 3 DOFs per
            # story): condense onto them with a sparse factorisation - ARPACK cannot
            # work with a mass matrix of such low rank
            method = "dense" if (n <= 3000 or n_dyn <= 1500) else "sparse"
        if method == "dense":
            sparse_k = n > 3000
            Kd = None if sparse_k else K.toarray()
            Kmm = K[massive][:, massive].toarray() if sparse_k else Kd[np.ix_(massive, massive)]
            Y = None
            if massless.size and sparse_k and factor is not None:
                E = np.zeros((n, n_dyn))
                E[massive, np.arange(n_dyn)] = 1.0
                Y = np.asarray(factor.solve(E)).reshape(n, n_dyn)
                if not np.all(np.isfinite(Y)):
                    raise ModelError("SINGULAR_STIFFNESS", "Massless DOFs are unstable (mechanism)")
                Ymm = 0.5 * (Y[massive] + Y[massive].T)
                try:
                    Kc = linalg.inv(Ymm)
                except linalg.LinAlgError as exc:
                    raise ModelError("SINGULAR_STIFFNESS", "Massless DOFs are unstable (mechanism)") from exc
                X = None
            elif massless.size:
                if sparse_k:
                    Kcsc = sparse.csc_matrix(K)
                    Kzz = Kcsc[massless][:, massless].tocsc()
                    Kzm = Kcsc[massless][:, massive].toarray()
                    try:
                        X = splu(Kzz).solve(Kzm)
                    except RuntimeError as exc:
                        raise ModelError("SINGULAR_STIFFNESS", "Massless DOFs are unstable (mechanism)") from exc
                    if not np.all(np.isfinite(X)):
                        raise ModelError("SINGULAR_STIFFNESS", "Massless DOFs are unstable (mechanism)")
                else:
                    Kzz = Kd[np.ix_(massless, massless)]
                    Kzm = Kd[np.ix_(massless, massive)]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", linalg.LinAlgWarning)
                            X = linalg.solve(Kzz, Kzm, assume_a="sym")
                    except linalg.LinAlgError as exc:
                        raise ModelError("SINGULAR_STIFFNESS", "Massless DOFs are unstable (mechanism)") from exc
                Kc = Kmm - Kzm.T @ X
            else:
                X = None
                Kc = Kmm
            Kc = 0.5 * (Kc + Kc.T)
            Mc = M[massive][:, massive].toarray()
            Mc = 0.5 * (Mc + Mc.T)
            try:
                w2, v = linalg.eigh(Kc, Mc)
            except linalg.LinAlgError:
                # M only positive semi-definite: fall back to the reversed problem
                w2, v = _eigh_semidefinite(Kc, Mc)
            order = np.argsort(w2)[:n_modes]
            w2 = w2[order]
            vm = v[:, order]
            phi = np.zeros((n, n_modes))
            phi[massive] = vm
            if massless.size:
                phi[massless] = (Y[massless] @ (Kc @ vm)) if Y is not None else -X @ vm
        else:
            sigma = -float(shift) if shift else -1e-3 * float(np.max(np.abs(K.diagonal())) / max(np.max(m), 1e-300)) * 1e-6
            Ms = M
            k = min(n_modes, n_dyn - 1) if n_dyn > 1 else 1
            if k < 1:
                raise ModelError("TOO_FEW_DOF", "Not enough dynamic degrees of freedom for sparse eigensolution")
            try:
                w2, phi = eigsh(K.tocsc(), k=k, M=Ms.tocsc(), sigma=sigma, which="LM", mode="normal")
            except Exception as exc:  # ARPACK failure
                raise ModelError("EIGEN_FAILED", f"Sparse eigensolver failed: {exc}") from exc
            order = np.argsort(w2)
            w2 = w2[order]
            phi = phi[:, order]
        # mass normalise
        for j in range(phi.shape[1]):
            mj = float(phi[:, j] @ (M @ phi[:, j]))
            if mj > 0:
                phi[:, j] /= np.sqrt(mj)
        w2 = np.where(w2 < 0, 0.0, w2)
        return w2, phi


def _eigh_semidefinite(K: np.ndarray, M: np.ndarray):
    """Generalised eigenproblem with a semi-definite M: solve M v = mu K v
    (K positive definite) and invert the eigenvalues."""
    mu, v = linalg.eigh(M, K)
    with np.errstate(divide="ignore"):
        w2 = np.where(mu > 1e-300, 1.0 / mu, np.inf)
    order = np.argsort(w2)
    return w2[order], v[:, order]


KERNEL_NOTE = ""  # why the default kernel is what it is (shown with the analysis messages)


def default_kernel() -> Kernel:
    """Intel MKL PARDISO when ``pypardiso`` is installed, else SciPy.
    ``STRUCTSTUDIO_KERNEL=scipy`` forces SciPy."""
    global KERNEL_NOTE
    if os.environ.get("STRUCTSTUDIO_KERNEL", "").lower() == "scipy":
        KERNEL_NOTE = "SciPy selected (STRUCTSTUDIO_KERNEL=scipy)"
        return SciPyKernel()
    try:
        from .kernel_pardiso import PardisoKernel  # optional accelerated kernel

        k = PardisoKernel()
        KERNEL_NOTE = ""
        return k
    except Exception as exc:
        KERNEL_NOTE = f"PARDISO not available ({type(exc).__name__}: {exc}) - install pypardiso for a faster solver"
        return SciPyKernel()


def describe(kernel) -> str:
    d = getattr(kernel, "describe", None)
    text = d() if callable(d) else getattr(kernel, "name", "kernel")
    fb = getattr(kernel, "last_fallback", None)
    if fb:
        text += f"; SuperLU used for a matrix PARDISO could not take ({fb})"
    return text
