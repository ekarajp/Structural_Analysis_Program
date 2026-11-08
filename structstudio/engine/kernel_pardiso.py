"""Intel MKL PARDISO kernel (multi-threaded sparse Cholesky) - optional.

Used when the ``pypardiso`` package (which brings Intel MKL) is installed;
otherwise :func:`structstudio.engine.kernel.default_kernel` stays with SciPy.

The stiffness of a stable structure is symmetric positive definite, so the
factorisation is a Cholesky (PARDISO matrix type 2) of the upper triangle with
the METIS nested-dissection ordering, on all CPU cores.  The stability checks of
the SciPy kernel are kept:

* PARDISO reports a matrix that is not positive definite (error -4: a
  mechanism) or a pivot that has lost (nearly) all significant digits; the
  factorisation is then redone with SciPy SuperLU, whose diagnostics name the
  unstable joints and DOFs exactly as before;
* every solve checks the equilibrium residual (relative 1e-6), like SciPy.
"""
from __future__ import annotations

import ctypes
import os
from typing import Tuple

import numpy as np
from scipy import sparse

from ..model.base import ModelError
from .kernel import UNSTABLE_PIVOT, SciPyKernel, _SciPyFactor

_I = ctypes.c_int32
_P = ctypes.POINTER
# a pivot this small relative to the largest one: let SuperLU find and name the DOF
FALLBACK_PIVOT = 1e3 * UNSTABLE_PIVOT


_LIB = None


def _library():
    """mkl_rt, found as pypardiso finds it (Windows / Linux); loaded once."""
    global _LIB
    if _LIB is None:
        import pypardiso
        _LIB = pypardiso.PyPardisoSolver().libmkl
    return _LIB


class _PardisoFactor:
    """Factor of a symmetric positive definite matrix (upper triangle, CSR)."""

    def __init__(self, lib, K: sparse.csr_matrix):
        self.lib = lib
        self.n = n = K.shape[0]
        self.K = K.tocsr()
        self.pt = np.zeros(64, np.int64)
        self.iparm = np.zeros(64, np.int32)
        self._alive = False
        self.unstable = []
        self.pivot_ratio = 1.0
        if n == 0:
            return
        A = sparse.triu(self.K, format="csr")
        A.sum_duplicates()
        A.sort_indices()
        # every row of the upper triangle needs its diagonal entry (PARDISO requirement)
        d = A.diagonal()
        if np.any(d == 0.0):
            raise _Fallback("zero diagonal")
        self.ia = A.indptr.astype(np.int32)
        self.ja = A.indices.astype(np.int32)
        self.a = np.ascontiguousarray(A.data, dtype=np.float64)
        self.mtype = 2
        lib.pardisoinit(self.pt.ctypes.data_as(ctypes.c_void_p), ctypes.byref(_I(self.mtype)),
                        self.iparm.ctypes.data_as(_P(_I)))
        self.iparm[34] = 1  # zero-based ia / ja
        self.iparm[55] = 1  # keep the pivots (pardiso_getdiag)
        err = self._call(12, np.zeros((n, 1)))[1]
        self._alive = True
        if err != 0:
            self.free()
            raise _Fallback(f"PARDISO error {err}")
        df = np.zeros(n)
        da = np.zeros(n)
        e2 = _I(0)
        lib.pardiso_getdiag(self.pt.ctypes.data_as(ctypes.c_void_p), df.ctypes.data_as(ctypes.c_void_p),
                            da.ctypes.data_as(ctypes.c_void_p), ctypes.byref(_I(1)), ctypes.byref(e2))
        if e2.value == 0:
            p = np.abs(df)
            if p.size and (not np.all(np.isfinite(p)) or np.max(p) <= 0):
                self.free()
                raise _Fallback("invalid pivots")
            if p.size:
                self.pivot_ratio = float(np.min(p) / np.max(p))
                if self.pivot_ratio <= FALLBACK_PIVOT:
                    self.free()
                    raise _Fallback("near-singular pivot")

    def _call(self, phase: int, b: np.ndarray):
        n = self.n
        b = np.asfortranarray(b, dtype=np.float64)
        nrhs = b.shape[1] if b.ndim == 2 else 1
        x = np.zeros_like(b)
        err = _I(0)
        perm = np.zeros(0, np.int32)
        self.lib.pardiso(self.pt.ctypes.data_as(_P(ctypes.c_int64)), ctypes.byref(_I(1)), ctypes.byref(_I(1)),
                         ctypes.byref(_I(self.mtype)), ctypes.byref(_I(phase)), ctypes.byref(_I(n)),
                         self.a.ctypes.data_as(ctypes.c_void_p), self.ia.ctypes.data_as(_P(_I)),
                         self.ja.ctypes.data_as(_P(_I)), perm.ctypes.data_as(_P(_I)), ctypes.byref(_I(nrhs)),
                         self.iparm.ctypes.data_as(_P(_I)), ctypes.byref(_I(0)),
                         b.ctypes.data_as(ctypes.c_void_p), x.ctypes.data_as(ctypes.c_void_p), ctypes.byref(err))
        return x, err.value

    def solve(self, b: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.zeros_like(b)
        b = np.asarray(b, float)
        x, err = self._call(33, b if b.ndim == 2 else b[:, None])
        if err != 0:
            raise ModelError("UNSTABLE_STRUCTURE", f"The sparse solver failed (PARDISO error {err})")
        x = x if b.ndim == 2 else x[:, 0]
        nb = float(np.linalg.norm(b))
        if nb > 0:
            res = float(np.linalg.norm(self.K @ x - b)) / nb
            if not np.isfinite(res) or res > 1e-6:
                raise ModelError("UNSTABLE_STRUCTURE",
                                 f"Equilibrium could not be satisfied (relative residual {res:.2e}): the "
                                 "structure is unstable (mechanism) or severely ill-conditioned")
        return x

    def free(self):
        if self._alive:
            try:
                self._call(-1, np.zeros((self.n, 1)))
            except Exception:
                pass
            self._alive = False

    def __del__(self):
        self.free()


class _Fallback(Exception):
    pass


class PardisoKernel(SciPyKernel):
    """PARDISO for the stiffness factorisation; eigen problems as SciPy (they
    reuse the static factor when the stiffness is the same)."""

    name = "pardiso"

    def __init__(self):
        self.lib = _library()
        self.last_fallback = None

    def factor(self, K: sparse.csr_matrix):
        try:
            fac = _PardisoFactor(self.lib, K)
            self.last_fallback = None
            return fac
        except _Fallback as why:
            # not positive definite / nearly singular: SuperLU names the DOFs (or raises)
            self.last_fallback = str(why)
            return _SciPyFactor(K)

    def describe(self) -> str:
        threads = os.environ.get("MKL_NUM_THREADS") or os.cpu_count()
        return f"Intel MKL PARDISO (sparse Cholesky, {threads} threads)"
