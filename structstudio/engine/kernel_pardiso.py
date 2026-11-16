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


def available_memory():
    """Physical memory that can be used now (bytes), or None when it cannot be told."""
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:
        pass
    if os.name == "nt":
        try:
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return float(ms.ullAvailPhys)
        except Exception:
            return None
        return None
    try:
        avail = None
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail = float(line.split()[1]) * 1024.0
        # a container's limit (cgroup v2) can be lower than the machine's memory
        for path in ("/sys/fs/cgroup/memory.max",):
            try:
                with open(path) as fh:
                    lim = fh.read().strip()
                with open("/sys/fs/cgroup/memory.current") as fh:
                    used = float(fh.read().strip())
                if lim != "max":
                    left = float(lim) - used
                    avail = left if avail is None else min(avail, left)
            except OSError:
                pass
        return avail
    except Exception:
        return None


class _PardisoFactor:
    """Factor of a symmetric positive definite matrix (upper triangle, CSR)."""

    # relative residual the refinement of a single-precision factor aims at (the solve
    # still accepts up to 1e-6, as for the double factor)
    REFINE_TOL = 1e-12
    REFINE_MAX_IT = 300

    def __init__(self, lib, K, allow_single: bool = True):
        holder = K if isinstance(K, list) else None
        if holder is not None:  # the caller hands the matrix over: released once triangulated
            K = holder.pop()
        self.lib = lib
        self.precision = "double"
        self.refine_iterations = 0
        self.n = n = K.shape[0]
        self.K = None  # (the full matrix is not kept: the residual uses the upper triangle)
        self.pt = np.zeros(64, np.int64)
        self.iparm = np.zeros(64, np.int32)
        self._alive = False
        self.unstable = []
        self.negative = []  # a Cholesky that succeeded: positive definite
        self.pivot_ratio = 1.0
        if n == 0:
            return
        A = sparse.triu(K.tocsr(), format="csr")
        del K  # (with a holder, the full matrix is freed here: only the upper triangle stays)
        A.sum_duplicates()
        A.sort_indices()
        self.U = A  # upper triangle: the equilibrium check of every solve (half the memory of K)
        # every row of the upper triangle needs its diagonal entry (PARDISO requirement)
        d = A.diagonal()
        if np.any(d == 0.0):
            raise _Fallback("zero diagonal")
        self.ia = A.indptr.astype(np.int32, copy=False)  # (shared with U when already int32)
        self.ja = A.indices.astype(np.int32, copy=False)
        self.a = np.ascontiguousarray(A.data, dtype=np.float64)
        self.mtype = 2
        lib.pardisoinit(self.pt.ctypes.data_as(ctypes.c_void_p), ctypes.byref(_I(self.mtype)),
                        self.iparm.ctypes.data_as(_P(_I)))
        self.iparm[34] = 1  # zero-based ia / ja
        self.iparm[55] = 1  # keep the pivots (pardiso_getdiag)
        # ordering first (phase 11): it tells how much memory the factor needs, so a model too
        # large for this computer is reported before the factorisation (phase 22) starts,
        # instead of the program being stopped by the system part way
        err = self._call(11, np.zeros((n, 1)))[1]
        self._alive = True
        if err == 0:
            need = (int(self.iparm[15]) + int(self.iparm[16])) * 1024.0  # bytes (permanent + factor)
            self.memory_needed = need
            avail = available_memory()
            if avail is not None and need > avail:
                self.free()
                if not allow_single:
                    raise MemoryError(f"the factorisation of {n:,} equations needs about {need / 2 ** 30:.1f} GB; "
                                      f"{avail / 2 ** 30:.1f} GB of memory is free")
                err = self._single_precision(need, avail)
            else:
                err = self._call(22, np.zeros((n, 1)))[1]
        if err != 0:
            self.free()
            if err == -2:  # not enough memory: SuperLU would need more still - say so instead
                raise MemoryError(f"PARDISO: not enough memory to factor {n} equations")
            if self.precision == "single":  # SuperLU (double, more memory) cannot take over here
                raise MemoryError(f"the factorisation of {n:,} equations does not fit in double precision and the "
                                  f"single-precision factor failed (PARDISO error {err}: the stiffness may be unstable)")
            raise _Fallback(f"PARDISO error {err}")
        dt = np.float32 if self.precision == "single" else np.float64
        df = np.zeros(n, dt)
        da = np.zeros(n, dt)
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
                if self.precision == "single":
                    # pivots of the diagonally scaled matrix: a stable structure has them of order 1;
                    # one below the single-precision resolution is a (near) mechanism that cannot be
                    # told apart here and that the double-precision diagnosis has no memory for
                    if self.pivot_ratio <= 1e3 * float(np.finfo(np.float32).eps) * 1e-3:
                        self.free()
                        raise MemoryError(f"the factorisation of {n:,} equations does not fit in double precision, "
                                          f"and in single precision a pivot is {self.pivot_ratio:.1e} of the largest "
                                          "(a near mechanism that needs the double-precision diagnosis)")
                elif self.pivot_ratio <= FALLBACK_PIVOT:
                    self.free()
                    raise _Fallback("near-singular pivot")

    def _single_precision(self, need_double: float, avail: float) -> int:
        """Factor in single precision when the double-precision factor does not fit.

        The matrix is scaled to unit diagonal (D^-1/2 K D^-1/2, so translations and
        rotations keep their digits in float32) and factored by PARDISO in single
        precision; every solve then runs preconditioned conjugate gradients in double
        precision on the original matrix, to the same equilibrium residual test as the
        double factor.  The factor takes about half the memory."""
        n = self.n
        d = np.asarray(self.U.diagonal(), float)
        if np.any(d <= 0) or not np.all(np.isfinite(d)):
            raise _Fallback("non-positive diagonal")
        self.scale = 1.0 / np.sqrt(d)
        U = self.U
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(U.indptr))
        a32 = (U.data * self.scale[rows] * self.scale[U.indices]).astype(np.float32)
        del rows
        self.pt = np.zeros(64, np.int64)
        self.iparm = np.zeros(64, np.int32)
        self.lib.pardisoinit(self.pt.ctypes.data_as(ctypes.c_void_p), ctypes.byref(_I(self.mtype)),
                             self.iparm.ctypes.data_as(_P(_I)))
        self.iparm[34] = 1
        self.iparm[55] = 1
        self.iparm[27] = 1  # single precision: a, b, x and the factor in float32
        self.iparm[7] = 0   # no internal refinement: the refinement is the double-precision CG below
        self.precision = "single"
        self.a = a32
        err = self._call(11, np.zeros((n, 1)))[1]
        self._alive = True
        if err != 0:
            return err
        need = (int(self.iparm[15]) + int(self.iparm[16])) * 1024.0
        self.memory_needed = need
        self.memory_needed_double = need_double
        avail2 = available_memory()
        avail2 = avail if avail2 is None else avail2
        if need > avail2:
            self.free()
            raise MemoryError(f"the factorisation of {n:,} equations needs about {need_double / 2 ** 30:.1f} GB "
                              f"(double precision) or {need / 2 ** 30:.1f} GB (single precision with refinement); "
                              f"{avail2 / 2 ** 30:.1f} GB of memory is free")
        return self._call(22, np.zeros((n, 1)))[1]

    def _precondition(self, R: np.ndarray) -> np.ndarray:
        """(single-precision factor)^-1 R, for a block of right-hand sides (n, k)."""
        s = self.scale[:, None]
        x, err = self._call(33, (R * s).astype(np.float32))
        if err != 0:
            raise ModelError("UNSTABLE_STRUCTURE", f"The sparse solver failed (PARDISO error {err})")
        return np.asarray(x, np.float64) * s

    def _refine(self, B: np.ndarray) -> np.ndarray:
        """Preconditioned conjugate gradients in double precision, column by column
        (blocked); converges in a few steps when the single factor is good."""
        nb = np.linalg.norm(B, axis=0)
        X = np.zeros_like(B)
        act = np.where(nb > 0)[0]
        if act.size == 0:
            return X
        R = B[:, act].copy()
        Z = self._precondition(R)
        P = Z.copy()
        rz = np.einsum("ij,ij->j", R, Z)
        it = 0
        for it in range(1, self.REFINE_MAX_IT + 1):
            AP = self._matvec(P)
            pap = np.einsum("ij,ij->j", P, AP)
            ok = (pap > 0) & np.isfinite(pap)
            alpha = np.where(ok, rz / np.where(ok, pap, 1.0), 0.0)
            X[:, act] += P * alpha
            R -= AP * alpha
            res = np.linalg.norm(R, axis=0) / nb[act]
            keep = (res > self.REFINE_TOL) & ok
            if not np.any(keep):
                break
            act, R, P, rz = act[keep], R[:, keep], P[:, keep], rz[keep]
            Z = self._precondition(R)
            rz_new = np.einsum("ij,ij->j", R, Z)
            P = Z + P * (rz_new / np.where(rz != 0, rz, 1.0))
            rz = rz_new
        self.refine_iterations = max(self.refine_iterations, it)
        return X

    def _call(self, phase: int, b: np.ndarray):
        n = self.n
        b = np.asfortranarray(b, dtype=np.float32 if self.precision == "single" else np.float64)
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
        if self.precision == "single":
            B = b if b.ndim == 2 else b[:, None]
            # a few right-hand sides at a time: the CG keeps several n x k work arrays
            x = np.concatenate([self._refine(B[:, j:j + 8]) for j in range(0, B.shape[1], 8)], axis=1) \
                if B.shape[1] else np.zeros_like(B)
        else:
            x, err = self._call(33, b if b.ndim == 2 else b[:, None])
            if err != 0:
                raise ModelError("UNSTABLE_STRUCTURE", f"The sparse solver failed (PARDISO error {err})")
        x = x if b.ndim == 2 else x[:, 0]
        nb = float(np.linalg.norm(b))
        if nb > 0:
            res = float(np.linalg.norm(self._matvec(x) - b)) / nb
            if not np.isfinite(res) or res > 1e-6:
                raise ModelError("UNSTABLE_STRUCTURE",
                                 f"Equilibrium could not be satisfied (relative residual {res:.2e}): the "
                                 "structure is unstable (mechanism) or severely ill-conditioned")
        return x

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """K @ x from the upper triangle (K symmetric): U x + U^T x - diag(U) x."""
        U = self.U
        d = U.diagonal()
        if x.ndim == 2:
            return U @ x + U.T @ x - d[:, None] * x
        return U @ x + U.T @ x - d * x

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

    accepts_holder = True  # factor([K]) takes the matrix over and frees it once triangulated

    def factor(self, K):
        holder = K if isinstance(K, list) else None
        fac = _PardisoFactor.__new__(_PardisoFactor)
        try:
            fac.__init__(self.lib, K)
            self.last_fallback = None
            return fac
        except _Fallback as why:
            # not positive definite / nearly singular: SuperLU names the DOFs (or raises)
            self.last_fallback = str(why)
            if holder is not None:  # the full matrix was released: rebuilt from the triangle
                U = fac.U
                K = (U + U.T - sparse.diags(U.diagonal())).tocsr()
            return _SciPyFactor(K)

    def describe(self) -> str:
        threads = os.environ.get("MKL_NUM_THREADS") or os.cpu_count()
        return f"Intel MKL PARDISO (sparse Cholesky, {threads} threads)"
