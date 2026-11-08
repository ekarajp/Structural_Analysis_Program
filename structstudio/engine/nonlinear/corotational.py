"""Corotational transformation for 3D frame elements (large displacements,
small strains).

Kinematics (Crisfield 1990; Battini & Pacoste 2002):

* nodal triads are updated multiplicatively from the committed state,
  ``R = exp(Δθ) R_committed`` with Δθ the trial rotation increment;
* the corotated frame has axis 1 along the deformed chord, axis 2 from the
  mean of the nodal axes 2, axis 3 = 1 x 2;
* the local nodal rotations are the rotation vectors (log map) of the nodal
  triads seen from the corotated frame;
* basic deformations: v = [Ln - L0, thz_I, thz_J, thy_I, thy_J, thx_J - thx_I].

The basic-to-global matrix B = dv/du and the geometric stiffness
sum_k q_k d²v_k/du² are obtained by central differences of v(u) evaluated
for all perturbations in one vectorised call, so that the internal force and
the tangent are consistent with one definition of the kinematics.  Span
loads keep their initial direction (gravity-like, not follower loads).
"""
from __future__ import annotations

import numpy as np


def _skew(w):
    z = np.zeros(w.shape[:-1])
    return np.stack([np.stack([z, -w[..., 2], w[..., 1]], -1),
                     np.stack([w[..., 2], z, -w[..., 0]], -1),
                     np.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def expmap(w: np.ndarray) -> np.ndarray:
    """Rodrigues formula, batched (..., 3) -> (..., 3, 3).  Analytic in w (series
    near zero) so that it can be differentiated by complex steps."""
    t2 = _dot(w, w)
    small = np.real(t2) < 1e-6
    t2s = np.where(small, 1.0, t2)
    t = np.sqrt(t2s)
    a = np.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0, np.sin(t) / t)
    b = np.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0, (1.0 - np.cos(t)) / t2s)
    K = _skew(w)
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + a[..., None, None] * K + b[..., None, None] * (K @ K)


def logmap(R: np.ndarray) -> np.ndarray:
    """Rotation vector of rotation matrices, batched (..., 3, 3) -> (..., 3);
    analytic (complex-step safe), valid for rotations below pi."""
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0  # cos(theta)
    s2 = _dot(v, v) / 4.0  # sin(theta)^2
    small = np.real(s2) < 1e-4
    # f = theta / (2 sin theta) as an even series in s = sin(theta) (theta = asin s)
    ser = 0.5 * (1.0 + s2 / 6.0 + 3.0 * s2 ** 2 / 40.0 + 5.0 * s2 ** 3 / 112.0 + 35.0 * s2 ** 4 / 1152.0)
    ss = np.sqrt(np.where(small, 1.0, s2))
    th = np.where(np.real(c) >= 0.0, np.arcsin(ss), np.pi - np.arcsin(ss))
    f = np.where(small, ser, th / (2.0 * ss))
    return f[..., None] * v


def left_jacobian(w: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian: spatial spin = J(w) dw for R = exp(w) R0."""
    t2 = float(w @ w)
    K = _skew(w)
    if t2 < 1e-12:
        return np.eye(3) + 0.5 * K + (K @ K) / 6.0
    t = np.sqrt(t2)
    return np.eye(3) + (1 - np.cos(t)) / t2 * K + (t - np.sin(t)) / (t2 * t) * (K @ K)


class CorotTransform3D:
    def __init__(self, xI: np.ndarray, xJ: np.ndarray, axes: np.ndarray):
        self.xI = np.asarray(xI, float)
        self.xJ = np.asarray(xJ, float)
        self.L0 = float(np.linalg.norm(self.xJ - self.xI))
        self.E0T = np.asarray(axes, float).T  # columns = initial local axes
        self.reset()

    def reset(self):
        self.RIc = np.eye(3)
        self.RJc = np.eye(3)
        self.thIc = np.zeros(3)
        self.thJc = np.zeros(3)
        self._u = np.zeros(12)

    def commit(self):
        u = self._u
        self.RIc = expmap(u[3:6] - self.thIc) @ self.RIc
        self.RJc = expmap(u[9:12] - self.thJc) @ self.RJc
        self.thIc = u[3:6].copy()
        self.thJc = u[9:12].copy()

    def revert(self):
        pass

    # ------------------------------------------------------------ kinematics
    def basic(self, U: np.ndarray) -> np.ndarray:
        """Basic deformations for a batch of global element displacement vectors (N, 12)."""
        with np.errstate(all="ignore"):  # diverging trial states give nan -> the step is subdivided
            return self._basic(np.atleast_2d(U))

    def _basic(self, U: np.ndarray) -> np.ndarray:
        RI = expmap(U[:, 3:6] - self.thIc) @ self.RIc
        RJ = expmap(U[:, 9:12] - self.thJc) @ self.RJc
        d0 = self.xJ - self.xI
        dd = U[:, 6:9] - U[:, 0:3]
        d = d0 + dd
        Ln = np.sqrt(_dot(d, d))
        # elongation without cancellation: (Ln^2 - L0^2) / (Ln + L0)
        elong = (2.0 * _dot(dd, d0) + _dot(dd, dd)) / (Ln + self.L0)
        e1 = d / Ln[:, None]
        tI = RI @ self.E0T
        tJ = RJ @ self.E0T
        qv = 0.5 * (tI[:, :, 1] + tJ[:, :, 1])
        e3 = np.cross(e1, qv)
        e3 = e3 / np.sqrt(_dot(e3, e3))[:, None]
        e2 = np.cross(e3, e1)
        E = np.stack([e1, e2, e3], axis=2)  # columns
        Et = np.transpose(E, (0, 2, 1))
        wI = self._local_rotation(Et @ tI)
        wJ = self._local_rotation(Et @ tJ)
        return np.stack([elong, wI[:, 2], wJ[:, 2], wI[:, 1], wJ[:, 1], wJ[:, 0] - wI[:, 0]], axis=1)

    local_rotation = "asin"  # "asin" (OpenSees CorotCrdTransf3d) or "log" (exact rotation vector)

    def _local_rotation(self, R):
        if self.local_rotation == "log":
            return logmap(R)
        v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
        return np.arcsin(0.5 * v)

    def steps(self, rel: float) -> np.ndarray:
        h = np.empty(12)
        h[[0, 1, 2, 6, 7, 8]] = rel * max(self.L0, 1e-9)
        h[[3, 4, 5, 9, 10, 11]] = rel
        return h

    def update(self, u: np.ndarray, q_needed: bool = True):
        """v(u), B = dv/du (6 x 12, complex-step exact) and the second derivatives
        (central differences of the exact B) for the geometric stiffness."""
        u = np.asarray(u, float)
        self._u = u.copy()
        hc = 1e-30
        I12 = np.eye(12)
        g = self.steps(1e-5)
        base = np.vstack([u, u + g[:, None] * I12, u - g[:, None] * I12])  # (25, 12)
        # complex perturbation of every DOF for every base point: (25, 12, 12)
        Z = base[:, None, :] + 1j * hc * I12[None, :, :]
        V = self.basic(Z.reshape(-1, 12)).reshape(25, 12, 6)
        Bs = np.imag(V) / hc  # (25, 12 dofs, 6)
        v0 = np.real(V[0, 0])
        B = Bs[0].T
        Bp, Bm = Bs[1:13], Bs[13:25]  # (12 j, 12 i, 6)
        # H[i, j, k] = d^2 v_k / du_i du_j
        self._H = np.transpose((Bp - Bm) / (2.0 * g[:, None, None]), (1, 0, 2))
        return v0, B

    def spin_map(self) -> np.ndarray:
        """12x12 block matrix T^-T mapping forces conjugate to the additive rotation
        DOFs to forces conjugate to spatial spins (the sense in which joint moments
        act, as in OpenSees / CSI); identity on translations."""
        u = self._u
        T = np.eye(12)
        T[3:6, 3:6] = np.linalg.inv(left_jacobian(u[3:6] - self.thIc)).T
        T[9:12, 9:12] = np.linalg.inv(left_jacobian(u[9:12] - self.thJc)).T
        return T

    def geometric_stiffness(self, q: np.ndarray) -> np.ndarray:
        # exact second derivative w.r.t. the (additive) rotation DOFs: not symmetric in
        # general because finite rotations do not add; keep it as is for quadratic convergence
        return np.einsum("ijk,k->ij", self._H, q)
