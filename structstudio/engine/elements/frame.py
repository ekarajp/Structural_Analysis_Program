"""3D two-node frame (beam-column) element.

* Euler-Bernoulli or Timoshenko (explicit shear areas) bending;
* end releases with optional partial fixity springs (static condensation);
* rigid end zones (bending/shear only, CSI convention) and rigid insertion
  offsets (joint -> centroid links);
* equivalent nodal loads for partial trapezoidal / point loads and moments,
  consistent with the element interpolation;
* internal force recovery at any station;
* consistent geometric stiffness for P-Delta.

Local axes: 1 along the member (I -> J), 2 and 3 transverse.  DOF order per
node: U1, U2, U3, R1, R2, R3.  Bending in the 1-2 plane uses I33/As2 and
rotation R3; bending in the 1-3 plane uses I22/As3 and rotation R2
(theta2 = -dw/dx).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ...mesh.analysis_model import ElementLoad, FrameElement
from ...mesh.geometry import skew

# 6-point Gauss-Legendre on [0, 1]
_GX, _GW = np.polynomial.legendre.leggauss(6)
_GX = 0.5 * (_GX + 1.0)
_GW = 0.5 * _GW


# ----------------------------------------------------------------------------
# Shape functions (Timoshenko, Friedman & Kosmatka form)
# ----------------------------------------------------------------------------
def _bending_shapes(xi: np.ndarray, L: float, phi: float):
    """Transverse displacement (N) and rotation (M) interpolation for the
    DOFs (v1, t1, v2, t2) where t = rotation (dv/dx for Euler)."""
    a = 1.0 / (1.0 + phi)
    xi2, xi3 = xi * xi, xi * xi * xi
    N = np.stack([
        a * (1 - 3 * xi2 + 2 * xi3 + phi * (1 - xi)),
        a * L * (xi - 2 * xi2 + xi3 + 0.5 * phi * (xi - xi2)),
        a * (3 * xi2 - 2 * xi3 + phi * xi),
        a * L * (-xi2 + xi3 - 0.5 * phi * (xi - xi2)),
    ], axis=-1)
    M = np.stack([
        a * 6.0 / L * (-xi + xi2),
        a * (1 - 4 * xi + 3 * xi2 + phi * (1 - xi)),
        a * 6.0 / L * (xi - xi2),
        a * (-2 * xi + 3 * xi2 + phi * xi),
    ], axis=-1)
    return N, M


def _bending_stiffness(EI: float, GAs: float, L: float) -> Tuple[np.ndarray, float]:
    """4x4 bending stiffness for (v1, t1, v2, t2)."""
    phi = 0.0 if GAs <= 0 else 12.0 * EI / (GAs * L * L)
    c = EI / ((1.0 + phi) * L ** 3)
    K = c * np.array([
        [12.0, 6 * L, -12.0, 6 * L],
        [6 * L, (4 + phi) * L * L, -6 * L, (2 - phi) * L * L],
        [-12.0, -6 * L, 12.0, -6 * L],
        [6 * L, (2 - phi) * L * L, -6 * L, (4 + phi) * L * L],
    ])
    return K, phi


def _flex_bending_stiffness(moments, L: float) -> np.ndarray:
    """4x4 bending stiffness for (v1, t1, v2, t2) of a member with variable EI from
    m_k = int_0^L x^k / EI dx (k = 0, 1, 2): basic flexibility of the simply supported
    member inverted and transformed with q = (t1 - psi, t2 - psi), psi = (v2 - v1) / L."""
    m0, m1, m2 = moments
    f11 = m0 - 2 * m1 / L + m2 / L ** 2
    f12 = -(m1 / L - m2 / L ** 2)
    f22 = m2 / L ** 2
    kb = np.linalg.inv(np.array([[f11, f12], [f12, f22]]))
    G = np.array([[1 / L, 1.0, -1 / L, 0.0], [1 / L, 0.0, -1 / L, 1.0]])
    return G.T @ kb @ G


# index maps: DOFs in the 12-vector for each bending plane
_PLANE12 = [1, 5, 7, 11]  # v along 2: U2_i, R3_i, U2_j, R3_j
_PLANE13 = [2, 4, 8, 10]  # w along 3: U3_i, R2_i, U3_j, R2_j  (R2 = -dw/dx)
_SIGN13 = np.array([1.0, -1.0, 1.0, -1.0])


@dataclass
class FrameKinematics:
    """Pre-computed geometric data for an element."""

    L: float  # node-to-node (centroidal axis) length
    Lf: float  # flexible length for bending (L - rigid_i - rigid_j)
    axes: np.ndarray  # 3x3 rows = local axes
    R12: np.ndarray  # 12x12 rotation global->local
    T_off: np.ndarray  # 12x12 rigid-link transform node DOFs (global) -> element end DOFs (global)
    T_rigid: np.ndarray  # 12x12 local transform: element ends -> flexible span faces (bending only use)
    phi2: float
    phi3: float


class FrameFormulation:
    """All element level operations for one :class:`FrameElement`."""

    def __init__(self, el: FrameElement, xi: np.ndarray, xj: np.ndarray):
        self.el = el
        self.xi = np.asarray(xi, float)
        self.xj = np.asarray(xj, float)
        self.kin = self._kinematics()
        self._k_local_full: Optional[np.ndarray] = None
        self._release_data = None

    # ---------------------------------------------------------------- setup
    def _kinematics(self) -> FrameKinematics:
        el = self.el
        pa = self.xi + el.offset_i
        pb = self.xj + el.offset_j
        L = float(np.linalg.norm(pb - pa))
        if L <= 0.0:
            raise ValueError(f"frame element {el.id} has zero length")
        axes = el.axes
        R = np.zeros((12, 12))
        for b in range(4):
            R[3 * b:3 * b + 3, 3 * b:3 * b + 3] = axes
        # rigid links from node to element end (global): u_end = u_node + theta x r
        T = np.eye(12)
        for end, r in ((0, el.offset_i), (1, el.offset_j)):
            if np.any(r):
                T[6 * end:6 * end + 3, 6 * end + 3:6 * end + 6] = -skew(r)  # (theta x r) = -skew(r) theta
        Lf = L - el.rigid_i - el.rigid_j
        if Lf <= 0.01 * L:
            Lf = 0.01 * L
        Tr = np.eye(12)
        for end, a in ((0, el.rigid_i), (1, -el.rigid_j)):
            if a:
                r = np.array([a, 0.0, 0.0])
                Tr[6 * end:6 * end + 3, 6 * end + 3:6 * end + 6] = -skew(r)
        EI33 = el.E * el.I33
        EI22 = el.E * el.I22
        phi2 = 0.0 if el.As2 <= 0 or EI33 <= 0 else 12.0 * EI33 / (el.G * el.As2 * Lf * Lf)
        phi3 = 0.0 if el.As3 <= 0 or EI22 <= 0 else 12.0 * EI22 / (el.G * el.As3 * Lf * Lf)
        return FrameKinematics(L, Lf, axes, R, T, Tr, phi2, phi3)

    # -------------------------------------------------------------- stiffness
    def local_stiffness_unreleased(self) -> np.ndarray:
        """12x12 stiffness in local axes at the element end DOFs (before
        releases), including rigid end zones."""
        if self._k_local_full is not None:
            return self._k_local_full
        el, k = self.el, self.kin
        L, Lf = k.L, k.Lf
        K = np.zeros((12, 12))
        # axial & torsion: full length (CSI: rigid zones do not affect them)
        if el.A > 0:
            ka = el.E * el.A / L
            K[np.ix_([0, 6], [0, 6])] += ka * np.array([[1, -1], [-1, 1]])
        if el.J > 0:
            kt = el.G * el.J / L
            K[np.ix_([3, 9], [3, 9])] += kt * np.array([[1, -1], [-1, 1]])
        Kb = np.zeros((12, 12))
        flex = getattr(el, "bending_flex", None) or {}
        if el.I33 > 0:
            if "33" in flex:
                K12 = _flex_bending_stiffness(flex["33"], Lf)
            else:
                K12, _ = _bending_stiffness(el.E * el.I33, el.G * el.As2, Lf)
            Kb[np.ix_(_PLANE12, _PLANE12)] += K12
        if el.I22 > 0:
            if "22" in flex:
                K13 = _flex_bending_stiffness(flex["22"], Lf)
            else:
                K13, _ = _bending_stiffness(el.E * el.I22, el.G * el.As3, Lf)
            S = np.diag(_SIGN13)
            Kb[np.ix_(_PLANE13, _PLANE13)] += S @ K13 @ S
        # elastic foundation (Winkler, CSI frame springs): consistent stiffness on the
        # flexible span, k * integral(N^T N) with the element's own shape functions
        kf = getattr(el, "foundation", None)
        if kf is not None and np.any(kf):
            xg, wg = np.polynomial.legendre.leggauss(8)
            xi = 0.5 * (xg + 1.0)
            wts = 0.5 * wg * Lf
            if kf[0]:
                Na = np.stack([1 - xi, xi], axis=-1)
                K[np.ix_([0, 6], [0, 6])] += kf[0] * (Na.T * wts) @ Na * (L / Lf)
            for plane, kk, phi, sign in ((_PLANE12, kf[1], k.phi2, None), (_PLANE13, kf[2], k.phi3, _SIGN13)):
                if kk:
                    N, _ = _bending_shapes(xi, Lf, phi)
                    Kw = kk * (N.T * wts) @ N
                    if sign is not None:
                        S = np.diag(sign)
                        Kw = S @ Kw @ S
                    Kb[np.ix_(plane, plane)] += Kw
        if el.rigid_i or el.rigid_j:
            Kb = k.T_rigid.T @ Kb @ k.T_rigid
        self._k_local_full = K + Kb
        return self._k_local_full

    def _releases(self) -> Tuple[np.ndarray, np.ndarray]:
        """Indices of released DOFs (in the 12-vector) and their partial
        fixity spring stiffness (0 = fully released)."""
        el = self.el
        idx, springs = [], []
        for end, rel, fix in ((0, el.releases_i, el.fixity_i), (1, el.releases_j, el.fixity_j)):
            for d in range(6):
                if rel[d]:
                    idx.append(6 * end + d)
                    springs.append(float(fix[d]))
        return np.array(idx, dtype=int), np.array(springs)

    def condense(self, K: np.ndarray, f: Optional[np.ndarray] = None):
        """Static condensation of the released DOFs, applied where the flexible span
        begins: with rigid end zones a pin is at the end of the rigid zone (the face
        of the support for a full rigid zone), not at the joint inside the column -
        a pinned beam then spans the clear length as a simple span and the moment is
        zero at the end of the rigid zone.  Done in the DOFs of the flexible-span
        faces (u_face = T_rigid u_end) and transformed back to the element ends."""
        k = self.kin
        if not (self.el.rigid_i or self.el.rigid_j) or self._releases()[0].size == 0:
            return self._condense_ends(K, f)
        T = k.T_rigid
        Ti = np.linalg.inv(T)
        # loads on the rigid zones belong to the joint side of the hinge: they go to
        # the node directly; only the flexible-span loads pass the condensation
        f_rig = np.zeros(12)
        part = getattr(self, "_rigid_part", None)
        if f is not None and part is not None and part[0].shape == f.shape and np.array_equal(part[0], f):
            f_rig = part[1]
        f_flex = None if f is None else f - f_rig
        Kc, fc, rec = self._condense_ends(Ti.T @ K @ Ti, None if f_flex is None else Ti.T @ f_flex)
        return T.T @ Kc @ T, T.T @ fc + f_rig, rec

    def _condense_ends(self, K: np.ndarray, f: Optional[np.ndarray] = None):
        """Static condensation of released element DOFs.

        Released DOFs become internal; the node DOF is connected to the
        internal DOF through a spring of stiffness ``fix`` (0 = free).
        Returns (K_cond, f_cond, recovery) where ``recovery`` maps node
        displacements (and loads) to the internal DOF values.
        """
        rel, springs = self._releases()
        n = 12
        if rel.size == 0:
            return K, (np.zeros(n) if f is None else f), None
        keep = np.array([i for i in range(n) if i not in set(rel.tolist())])
        # unknowns: node DOFs (12) + internal DOFs (len(rel))
        m = len(rel)
        Kbig = np.zeros((n + m, n + m))
        fbig = np.zeros(n + m)
        # element stiffness lives on element DOFs: kept = node DOF, released = internal DOF
        emap = np.arange(n)
        emap[rel] = n + np.arange(m)
        Kbig[np.ix_(emap, emap)] += K
        if f is not None:
            fbig[emap] += f
        for a, (i, ks) in enumerate(zip(rel, springs)):
            if ks > 0:
                ii, jj = int(i), n + a
                Kbig[ii, ii] += ks
                Kbig[jj, jj] += ks
                Kbig[ii, jj] -= ks
                Kbig[jj, ii] -= ks
        Knn = Kbig[:n, :n]
        Kni = Kbig[:n, n:]
        Kii = Kbig[n:, n:]
        # Released DOFs that carry no stiffness at all (e.g. rotations of a member with
        # I22 = I33 = 0 and pinned ends, a truss) are simply disconnected.  A singular
        # coupling among stiff released DOFs is a real mechanism (unstable releases).
        diag = np.abs(np.diag(Kii))
        scale = max(np.abs(K).max(), 1e-300)
        live = diag > 1e-12 * scale
        Kii_inv = np.zeros_like(Kii)
        if live.any():
            idx = np.where(live)[0]
            sub = Kii[np.ix_(idx, idx)]
            try:
                inv = np.linalg.inv(sub)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"frame element {self.el.id}: unstable end releases") from exc
            if not np.all(np.isfinite(inv)) or np.linalg.cond(sub) > 1e14:
                raise ValueError(f"frame element {self.el.id}: unstable end releases")
            Kii_inv[np.ix_(idx, idx)] = inv
        if np.any(np.abs(fbig[n:][~live]) > 1e-9 * max(np.abs(fbig).max(), 1.0)):
            raise ValueError(f"frame element {self.el.id}: span load on a released DOF without stiffness")
        Kc = Knn - Kni @ Kii_inv @ Kni.T
        fc = fbig[:n] - Kni @ Kii_inv @ fbig[n:]
        recovery = (Kii_inv, Kni, fbig[n:], emap, rel)
        return Kc, fc, recovery

    def stiffness_global(self) -> np.ndarray:
        """12x12 condensed stiffness at the node DOFs in global axes."""
        Kl = self.local_stiffness_unreleased()
        Kc, _, _ = self.condense(Kl)
        k = self.kin
        return k.T_off.T @ k.R12.T @ Kc @ k.R12 @ k.T_off

    # -------------------------------------------------------------------- mass
    def lumped_mass(self) -> np.ndarray:
        """Lumped translational mass (kg) at ends I and J (half each)."""
        m = self.el.mass_per_length * self.kin.L
        return np.array([0.5 * m, 0.5 * m])

    # ------------------------------------------------------------------- loads
    def element_loads(self, pattern: str, self_weight: float = 0.0, gravity: float = 9.80665) -> List[ElementLoad]:
        """Loads of ``pattern`` on this element in local axes, including self
        weight when ``self_weight`` (pattern multiplier) is non-zero."""
        loads = [ld for ld in self.el.loads if ld.pattern == pattern]
        if self_weight and self.el.weight_per_length:
            w = -self.el.weight_per_length * self_weight  # global -Z
            vec = self.kin.axes @ np.array([0.0, 0.0, w])
            loads.append(ElementLoad(pattern, "distributed", "Force", 0.0, self.kin.L, vec, vec.copy()))
        return loads

    def fixed_end_load_vector(self, loads: List[ElementLoad]) -> np.ndarray:
        """Equivalent nodal load vector (local, 12) at the *element end* DOFs
        (before release condensation).  Positive = load applied to the
        element in the positive local directions."""
        el, k = self.el, self.kin
        L, Lf = k.L, k.Lf
        f = np.zeros(12)
        a_i = el.rigid_i  # flexible span starts at x = a_i
        self._rigid_acc = np.zeros(12)  # the part of f from loads on the rigid zones
        for ld in loads:
            if ld.kind == "point":
                f += self._point_load_vector(ld.x_a, ld.value_a, ld.load_type)
            elif ld.kind == "temperature":
                f += self._thermal_load_vector(ld)
            else:
                f += self._distributed_load_vector(ld.x_a, ld.x_b, ld.value_a, ld.value_b, ld.load_type)
        self._rigid_part = (f.copy(), self._rigid_acc)
        self._rigid_acc = None
        return f

    def _thermal_load_vector(self, ld: ElementLoad) -> np.ndarray:
        """Equivalent nodal loads of a temperature change (axial strain alpha*T) or a
        temperature gradient across local 2 / 3 (free curvature -alpha*g, hot side
        expands).  Linear variation along the element: the average acts."""
        el = self.el
        f = np.zeros(12)
        t = 0.5 * (float(ld.value_a[0]) + float(ld.value_b[0]))
        if ld.load_type == "Strain":  # imposed axial strain (used for tension / compression limits)
            n = el.E * el.A * t
            f[0] -= n
            f[6] += n
            return f
        if not el.alpha or not t:
            return f
        if ld.load_type == "Temperature":
            n = el.E * el.A * el.alpha * t
            f[0] -= n
            f[6] += n
        elif ld.load_type == "Gradient2":
            mz = el.E * el.I33 * el.alpha * t
            f[5] += mz
            f[11] -= mz
        elif ld.load_type == "Gradient3":
            my = el.E * el.I22 * el.alpha * t  # (w, theta) form as plane 1-2, R2 = -theta
            f[4] -= my
            f[10] += my
        return f

    def _flex_shapes(self, x: np.ndarray):
        """Shape functions on the flexible span for a global station x
        (0..L).  Points inside rigid zones are clipped to the faces."""
        k = self.kin
        xf = np.clip(x - self.el.rigid_i, 0.0, k.Lf)
        xi = xf / k.Lf
        N2, M2 = _bending_shapes(xi, k.Lf, k.phi2)  # plane 1-2
        N3, M3 = _bending_shapes(xi, k.Lf, k.phi3)  # plane 1-3 (with sign map)
        Na = np.stack([1 - x / k.L, x / k.L], axis=-1)  # axial / torsion: full length, linear
        return N2, M2, N3, M3, Na

    def _rigid_zone_statics(self, x: float, F: np.ndarray, Mo: np.ndarray) -> np.ndarray:
        """Transfer a force F and moment Mo at station x inside a rigid zone
        to the element end DOFs (local)."""
        f = np.zeros(12)
        if x <= self.el.rigid_i:
            r = np.array([x, 0.0, 0.0])  # from end I to the load
            f[0:3] += F
            f[3:6] += Mo + np.cross(r, F)
        else:
            r = np.array([x - self.kin.L, 0.0, 0.0])
            f[6:9] += F
            f[9:12] += Mo + np.cross(r, F)
        acc = getattr(self, "_rigid_acc", None)
        if acc is not None:
            acc += f
        return f

    def _assemble_from_shapes(self, N2, M2, N3, M3, Na, q, m, weight) -> np.ndarray:
        """Nodal load contribution for force per length q (3,) and moment per
        length m (3,) at one or more stations with quadrature weights."""
        f = np.zeros(12)
        # axial
        f[[0, 6]] += np.sum(weight[:, None] * Na * q[:, 0:1], axis=0)
        # torsion
        f[[3, 9]] += np.sum(weight[:, None] * Na * m[:, 0:1], axis=0)
        # plane 1-2: transverse q2 and distributed moment m3
        f[_PLANE12] += np.sum(weight[:, None] * (N2 * q[:, 1:2] + M2 * m[:, 2:3]), axis=0)
        # plane 1-3: transverse q3 and distributed moment m2.  The shapes are those of
        # the 1-2 plane for (w, theta = dw/dx); R2 = -dw/dx, so a moment about +2 does
        # work on -theta: in (w, theta) form the load is (q3, -m2), then mapped to
        # (U3, R2) with _SIGN13.
        f13 = np.sum(weight[:, None] * (N3 * q[:, 2:3] - M3 * m[:, 1:2]), axis=0)
        f[_PLANE13] += _SIGN13 * f13
        return f

    def _point_load_vector(self, x: float, value: np.ndarray, load_type: str) -> np.ndarray:
        el = self.el
        F = value if load_type == "Force" else np.zeros(3)
        Mo = value if load_type == "Moment" else np.zeros(3)
        if x < el.rigid_i - 1e-12 or x > self.kin.L - el.rigid_j + 1e-12:
            return self._rigid_zone_statics(x, F, Mo)
        xs = np.array([x])
        N2, M2, N3, M3, Na = self._flex_shapes(xs)
        f = self._assemble_from_shapes(N2, M2, N3, M3, Na, F[None, :], Mo[None, :], np.array([1.0]))
        return self._rigid_to_ends(f)

    def _distributed_load_vector(self, xa: float, xb: float, va: np.ndarray, vb: np.ndarray, load_type: str) -> np.ndarray:
        """Integrate a trapezoidal load between xa and xb."""
        el, k = self.el, self.kin
        f = np.zeros(12)
        if xb <= xa:
            return f
        # split into rigid I / flexible / rigid J parts
        pieces = []
        bounds = [xa, max(xa, min(xb, el.rigid_i)), max(xa, min(xb, k.L - el.rigid_j)), xb]
        for p in range(3):
            lo, hi = bounds[p], bounds[p + 1]
            if hi - lo > 1e-14:
                pieces.append((lo, hi, p))
        f_flex = np.zeros(12)
        for lo, hi, part in pieces:
            xs = lo + (hi - lo) * _GX
            w = (hi - lo) * _GW
            t = (xs - xa) / (xb - xa)
            vals = va[None, :] * (1 - t)[:, None] + vb[None, :] * t[:, None]
            q = vals if load_type == "Force" else np.zeros_like(vals)
            m = vals if load_type == "Moment" else np.zeros_like(vals)
            if part == 1:
                N2, M2, N3, M3, Na = self._flex_shapes(xs)
                f_flex += self._assemble_from_shapes(N2, M2, N3, M3, Na, q, m, w)
            else:
                for s, ws, qq, mm in zip(xs, w, q, m):
                    f += self._rigid_zone_statics(float(s), qq * ws, mm * ws)
        return f + self._rigid_to_ends(f_flex)

    def _rigid_to_ends(self, f_faces: np.ndarray) -> np.ndarray:
        """Flexible-span face loads -> element end loads through the rigid
        zone links (f_end = T_rigid^T f_face).  Axial and torsion rows pass
        through unchanged because the links lie along the axis."""
        el = self.el
        if not (el.rigid_i or el.rigid_j):
            return f_faces
        return self.kin.T_rigid.T @ f_faces

    def load_vector_global(self, loads: List[ElementLoad]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (F_global_12 at node DOFs, f_local_end_unreleased)."""
        f_local = self.fixed_end_load_vector(loads)
        Kl = self.local_stiffness_unreleased()
        _, fc, _ = self.condense(Kl, f_local)
        k = self.kin
        return k.T_off.T @ k.R12.T @ fc, f_local

    # ------------------------------------------------------------ recovery
    def end_forces_local(self, u_global: np.ndarray, loads: List[ElementLoad]) -> np.ndarray:
        """Forces the nodes apply to the element at its ends, in local axes
        (12,).  Includes the fixed-end actions of the element loads."""
        k = self.kin
        u_end = k.R12 @ k.T_off @ u_global  # local displacements at the element ends (node side)
        Kl = self.local_stiffness_unreleased()
        f_local = self.fixed_end_load_vector(loads)
        Kc, fc, recovery = self.condense(Kl, f_local)
        # forces from the condensed system: f_nodes = Kc u - fc
        return Kc @ u_end - fc

    def internal_forces(self, u_global: np.ndarray, loads: List[ElementLoad], x: np.ndarray,
                        end_forces: Optional[np.ndarray] = None) -> np.ndarray:
        """Section forces [P, V2, V3, T, M2, M3] at stations x (0..L from
        end I along the centroidal axis), CSI sign convention.  ``end_forces``:
        local end forces from a nonlinear analysis (equilibrium is then used
        instead of the elastic stiffness)."""
        fI = (self.end_forces_local(u_global, loads) if end_forces is None else np.asarray(end_forces))[:6]
        x = np.atleast_1d(np.asarray(x, float))
        out = np.zeros((len(x), 6))
        L = self.kin.L
        span = [ld for ld in loads if ld.kind != "temperature"]
        kf = getattr(self.el, "foundation", None)
        found = kf is not None and np.any(kf)
        if found:
            u_end = self.kin.R12 @ self.kin.T_off @ u_global
        for n, xs in enumerate(x):
            Fsum = fI[:3].copy()
            Msum = fI[3:6].copy() + np.cross(np.array([-xs, 0.0, 0.0]), fI[:3])
            for ld in span:
                Fl, Ml = _load_resultant_left(ld, xs, L)
                Fsum += Fl
                Msum += Ml
            if found and xs > 0:  # foundation reaction -k u(s) on [0, x]
                Fl, Ml = self._foundation_resultant_left(u_end, xs)
                Fsum += Fl
                Msum += Ml
            S = -Fsum
            Ms = -Msum
            # CSI sign convention: M2 is positive for the 1-3 plane "sagging" that
            # a positive V3 produces, i.e. opposite to the right-hand rule about +2
            out[n] = [S[0], S[1], S[2], Ms[0], -Ms[1], Ms[2]]
        return out

    def elongation_row(self) -> np.ndarray:
        """Row b (12,) with axial elongation = b @ u_global (node DOFs)."""
        A = self.kin.R12 @ self.kin.T_off
        return A[6] - A[0]

    def foundation_reaction_global(self, u_global: np.ndarray) -> Optional[np.ndarray]:
        """Total force (global) the elastic foundation applies to the element."""
        kf = getattr(self.el, "foundation", None)
        if kf is None or not np.any(kf):
            return None
        u_end = self.kin.R12 @ self.kin.T_off @ u_global
        F, _ = self._foundation_resultant_left(u_end, self.kin.L)
        return self.kin.axes.T @ F

    def _foundation_resultant_left(self, u_end: np.ndarray, x: float) -> Tuple[np.ndarray, np.ndarray]:
        """Resultant (about station x) of the foundation reaction between end I and x."""
        k = self.kin
        kf = self.el.foundation
        xg, wg = np.polynomial.legendre.leggauss(8)
        s = 0.5 * x * (xg + 1.0)
        w = 0.5 * x * wg
        N2, _, N3, _, Na = self._flex_shapes(s)
        ua = Na @ u_end[[0, 6]]
        v = N2 @ u_end[_PLANE12]
        ww = N3 @ (_SIGN13 * u_end[_PLANE13])
        q = -np.stack([kf[0] * ua, kf[1] * v, kf[2] * ww], axis=-1)  # (n, 3) local force per length
        F = (q * w[:, None]).sum(axis=0)
        r = np.stack([s - x, np.zeros_like(s), np.zeros_like(s)], axis=-1)
        M = (np.cross(r, q) * w[:, None]).sum(axis=0)
        return F, M

    # ------------------------------------------------------------ geometric
    def geometric_stiffness_global(self, axial_force: float) -> np.ndarray:
        """Consistent geometric stiffness (12x12 global) for axial force P
        (tension positive), CSI-style initial stress formulation."""
        k = self.kin
        L = k.Lf
        P = axial_force
        Kg = np.zeros((12, 12))
        if P == 0.0:
            return Kg
        c = P / (30.0 * L)
        G4 = c * np.array([
            [36.0, 3 * L, -36.0, 3 * L],
            [3 * L, 4 * L * L, -3 * L, -L * L],
            [-36.0, -3 * L, 36.0, -3 * L],
            [3 * L, -L * L, -3 * L, 4 * L * L],
        ])
        Kg[np.ix_(_PLANE12, _PLANE12)] += G4
        S = np.diag(_SIGN13)
        Kg[np.ix_(_PLANE13, _PLANE13)] += S @ G4 @ S
        if self.el.rigid_i or self.el.rigid_j:
            Kg = k.T_rigid.T @ Kg @ k.T_rigid
        Kc, _, _ = self.condense(Kg)
        return k.T_off.T @ k.R12.T @ Kc @ k.R12 @ k.T_off


def _load_resultant_left(ld: ElementLoad, x: float, L: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Resultant force and moment (about the station x) of the part of a
    load that lies between end I and station x.  A point load exactly at a
    station counts as already applied (value just after it), except at end J,
    where the section force is the element end force (value just before it,
    CSI convention)."""
    F = np.zeros(3)
    M = np.zeros(3)
    if ld.kind == "point":
        tol = 1e-9 * max(L or 1.0, 1.0)
        at_end_j = L is not None and abs(ld.x_a - L) <= tol and abs(x - L) <= tol
        if ld.x_a <= x + 1e-12 and not at_end_j:
            if ld.load_type == "Force":
                F += ld.value_a
                M += np.cross(np.array([ld.x_a - x, 0.0, 0.0]), ld.value_a)
            else:
                M += ld.value_a
        return F, M
    lo, hi = ld.x_a, min(ld.x_b, x)
    if hi <= lo:
        return F, M
    span = ld.x_b - ld.x_a
    ta = (lo - ld.x_a) / span if span > 0 else 0.0
    tb = (hi - ld.x_a) / span if span > 0 else 0.0
    qa = ld.value_a + (ld.value_b - ld.value_a) * ta
    qb = ld.value_a + (ld.value_b - ld.value_a) * tb
    seg = hi - lo
    total = 0.5 * (qa + qb) * seg
    # centroid of trapezoid measured from lo
    denom = qa + qb
    with np.errstate(invalid="ignore", divide="ignore"):
        xc = np.where(np.abs(denom) > 1e-300, seg * (qa + 2 * qb) / (3.0 * denom), seg / 2.0)
    if ld.load_type == "Force":
        F += total
        # moment of each component about x: r x q with r = (lo + xc - x) e1
        for comp in range(3):
            r = np.array([lo + float(xc[comp]) - x, 0.0, 0.0])
            fc = np.zeros(3)
            fc[comp] = total[comp]
            M += np.cross(r, fc)
    else:
        M += total
    return F, M
