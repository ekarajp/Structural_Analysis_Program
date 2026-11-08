"""Four-node planar shell element (membrane + Mindlin plate, MITC4 shear).

* membrane: bilinear plane stress with a drilling-rotation penalty;
* bending: Mindlin-Reissner plate with MITC4 covariant edge tying for the
  transverse shear (free of shear locking);
* "ShellThin" sections use the Discrete Kirchhoff Quadrilateral (DKQ, Batoz &
  Tahar 1982): no transverse shear deformation, Kirchhoff constraints along
  the edges with cubic deflection - the CSI "thin-plate" behaviour, and
  compatible with Euler-Bernoulli beams along slab edges (no spurious moment
  exchange at the mesh nodes); ``THIN_FORMULATION = "MITC"`` restores the older
  penalised-shear MITC4 version.  "ShellThick" uses the DKMQ (Katili 1993): the
  DKQ with the constant edge shear of a Timoshenko strip (phi = 12 D / (Ds L^2)),
  tied covariantly - it converges to the Mindlin solution and becomes the DKQ as
  the plate gets thin; ``THICK_FORMULATION = "MITC"`` gives MITC4 with the actual
  shear stiffness.  "Membrane" and "Plate" switch the corresponding part off;
* CSI style stiffness modifiers f11 f22 f12 m11 m22 m12 v13 v23;
* lumped mass and equivalent nodal loads (pressure, uniform global force,
  self weight) using the consistent tributary areas.

Local DOF order per node: U1 U2 U3 R1 R2 R3 (24 per element).  Internal
force resultants follow the CSI convention: F11 F22 F12 (force/length),
M11 M22 M12 (moment/length), V13 V23 (shear force/length).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import linalg

from ...mesh.analysis_model import ShellElement, ShellLoad

_G2 = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
THIN_SHEAR_FACTOR = 1.0e2  # penalty multiplier for "ShellThin" transverse shear stiffness (MITC variant)
THIN_FORMULATION = "DKQ"  # "DKQ" (discrete Kirchhoff) | "MITC" (penalised MITC4 shear)
# "ShellThick": "DKMQ" (discrete Kirchhoff-Mindlin quadrilateral, Katili 1993: the DKQ
# with the edge shear of a Timoshenko strip - becomes the DKQ as the plate gets thin)
# | "MITC" (MITC4 with the actual shear stiffness)
THICK_FORMULATION = "DKMQ"
SHEAR_SIGN = 1.0  # sign of the equilibrium shears of the DKQ relative to the MITC4 resultants
# Membrane: bilinear displacements + two condensed incompatible bending modes
# (QM6) + drilling rotations tied to the continuum rotation by a penalty
# gamma * G * t (Hughes-Brezzi), fully integrated.  The incompatible modes are
# included in the continuum rotation as well, so pure in-plane bending satisfies
# the drilling constraint and the penalty does not lock (cf. Ibrahimbegovic &
# Wilson 1991).  gamma = 0.7 reproduces every joint displacement of the ETABS 21
# wall examples (Verification Example 15a-f, one element per wall panel) within
# 2.6 %, including frame-wall coupling (15b, 15c); see docs/CONVENTIONS.md.
# The Allman-type variant is kept for research but bends a wall on two columns
# under its own weight, which ETABS does not.
DRILLING_FACTOR = 0.7
ALLMAN = False           # Allman drilling interpolation of the membrane displacements
INCOMPATIBLE = True      # QM6 incompatible bending modes (condensed)
DRILL_ONE_POINT = False  # integrate the drilling penalty with one point
DRILLING_FACTOR_1PT = 0.0  # extra one-point (centroid) penalty gamma (x G t), added to the above
DRILL_INCOMPATIBLE = True  # incompatible modes also enter the continuum rotation of the penalty  # integrate the drilling penalty with one point


def _midside_values(xi: float, eta: float) -> np.ndarray:
    """Serendipity mid-side functions N5..N8 (edges 12, 23, 34, 41)."""
    return np.array([0.5 * (1 - xi * xi) * (1 - eta), 0.5 * (1 + xi) * (1 - eta * eta),
                     0.5 * (1 - xi * xi) * (1 + eta), 0.5 * (1 - xi) * (1 - eta * eta)])


def _midside_derivatives(xi: float, eta: float) -> np.ndarray:
    """Natural derivatives (d/dxi, d/deta) of the serendipity mid-side
    functions N5..N8 (edges 12, 23, 34, 41)."""
    return np.array([
        [-xi * (1 - eta), 0.5 * (1 - eta * eta), -xi * (1 + eta), -0.5 * (1 - eta * eta)],
        [-0.5 * (1 - xi * xi), -eta * (1 + xi), 0.5 * (1 - xi * xi), -eta * (1 - xi)],
    ])


def _serendipity_derivatives(xi: float, eta: float) -> np.ndarray:
    """Natural derivatives (2, 8) of the 8-node serendipity functions: corners
    1-4 then mid-sides 5-8 (edges 12, 23, 34, 41)."""
    xs = np.array([-1.0, 1.0, 1.0, -1.0])
    es = np.array([-1.0, -1.0, 1.0, 1.0])
    d = np.zeros((2, 8))
    for i in range(4):
        a, b = xs[i], es[i]
        # N = 1/4 (1 + a xi)(1 + b eta)(a xi + b eta - 1)
        d[0, i] = 0.25 * a * (1 + b * eta) * (2 * a * xi + b * eta)
        d[1, i] = 0.25 * b * (1 + a * xi) * (a * xi + 2 * b * eta)
    d[:, 4:] = _midside_derivatives(xi, eta)
    return d


def _shape(xi: float, eta: float):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)],
                          [-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]])
    return N, dN


class ShellFormulation:
    def __init__(self, el: ShellElement, coords: np.ndarray):
        """``coords``: (4, 3) global coordinates of the element nodes."""
        self.el = el
        self.xyz = np.asarray(coords, float)
        self.axes = self._axes()
        self.R = linalg.block_diag(*([self.axes] * 8))  # 24x24 global -> local
        local = (self.axes @ (self.xyz - self.xyz[0]).T).T
        scale = float(np.max(np.linalg.norm(self.xyz - self.xyz[0], axis=1)))
        if np.max(np.abs(local[:, 2])) > max(scale, 1.0) * 1e-6:
            raise ValueError(f"shell element {el.id} is warped; subdivide into planar quads")
        self.xy = local[:, :2]
        self.dkq = el.model_type == "ShellThin" and THIN_FORMULATION == "DKQ"
        self.dkmq = el.model_type == "ShellThick" and THICK_FORMULATION == "DKMQ"
        self._K = None
        self.area = 0.0
        self.tributary = np.zeros(4)

    # ------------------------------------------------------------- geometry
    def _axes(self) -> np.ndarray:
        p = self.xyz
        normal = np.cross(p[2] - p[0], p[3] - p[1])
        n = np.linalg.norm(normal)
        if n < 1e-14:
            raise ValueError(f"shell element {self.el.id} is degenerate")
        normal /= n
        if self.el.axes is not None:
            v1 = self.el.axes[0] - (self.el.axes[0] @ normal) * normal
            if np.linalg.norm(v1) < 1e-9:
                v1 = self.el.axes[1] - (self.el.axes[1] @ normal) * normal
            v1 /= np.linalg.norm(v1)
        else:
            v1 = p[1] - p[0]
            v1 = v1 - (v1 @ normal) * normal
            v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        return np.vstack([v1, v2, normal])

    def _jacobian(self, xi, eta):
        N, dN = _shape(xi, eta)
        J = dN @ self.xy
        det = float(np.linalg.det(J))
        if det <= 0.0:
            raise ValueError(f"shell element {self.el.id}: non-positive Jacobian (check node ordering)")
        return N, np.linalg.solve(J, dN), J, det

    def _matrices(self, xi, eta):
        N, dNdx, J, det = self._jacobian(xi, eta)
        Bm = np.zeros((3, 24))
        Bb = np.zeros((3, 24))
        Bd = np.zeros((1, 24))
        # membrane displacement gradient operators (2 x 24): rows d/dx, d/dy
        dU = np.zeros((2, 24))
        dV = np.zeros((2, 24))
        for i in range(4):
            dx, dy = dNdx[:, i]
            j = 6 * i
            dU[0, j] = dx
            dU[1, j] = dy
            dV[0, j + 1] = dx
            dV[1, j + 1] = dy
            # curvatures with rotations theta1 (about 1) and theta2 (about 2):
            # w,x = theta2, w,y = -theta1
            Bb[0, j + 4] = dx
            Bb[1, j + 3] = -dy
            Bb[2, j + 4] = dy
            Bb[2, j + 3] = -dx
        if ALLMAN and self.el.model_type != "Plate":
            # Allman-type drilling rotations: hierarchical quadratic edge bumps
            # u += Nk (yj - yi)(tj - ti)/8 ;  v += Nk (xi - xj)(tj - ti)/8
            dNm = _midside_derivatives(xi, eta)  # (2, 4) natural derivatives of N5..N8
            Jinv = np.linalg.inv(J)
            dNm_x = Jinv @ dNm  # (2, 4): d/dx, d/dy of each mid-side function
            for k, (i, jn) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
                ax = (self.xy[jn, 1] - self.xy[i, 1]) / 8.0
                ay = (self.xy[i, 0] - self.xy[jn, 0]) / 8.0
                mx, my = dNm_x[0, k], dNm_x[1, k]
                for node, sign in ((jn, 1.0), (i, -1.0)):
                    c = 6 * node + 5
                    dU[0, c] += sign * mx * ax
                    dU[1, c] += sign * my * ax
                    dV[0, c] += sign * mx * ay
                    dV[1, c] += sign * my * ay
        Bm[0] = dU[0]
        Bm[1] = dV[1]
        Bm[2] = dU[1] + dV[0]
        Bd[0] = 0.5 * (dV[0] - dU[1])
        for i in range(4):
            Bd[0, 6 * i + 5] -= N[i]

        def covariant(a, b, axis):
            Nt, dNt = _shape(a, b)
            Jt = dNt @ self.xy
            B = np.zeros(24)
            for i in range(4):
                j = 6 * i
                B[j + 2] = dNt[axis, i]
                B[j + 4] = Nt[i] * Jt[axis, 0]
                B[j + 3] = -Nt[i] * Jt[axis, 1]
            return B

        cov = np.array([
            0.5 * ((1 - eta) * covariant(0, -1, 0) + (1 + eta) * covariant(0, 1, 0)),
            0.5 * ((1 - xi) * covariant(-1, 0, 1) + (1 + xi) * covariant(1, 0, 1)),
        ])
        Bs = np.linalg.solve(J, cov)
        if self.dkq:
            Bb = self._dkq_Bb(xi, eta, J)
            Bs = np.zeros((2, 24))
        elif self.dkmq:
            Bb = self._dkq_Bb(xi, eta, J)
            Bs = self._dkmq_Bs(xi, eta, J)
        return N, Bm, Bb, Bs, Bd, det

    # ------------------------------------------------------------------ DKQ
    def _dkq_nodal(self) -> np.ndarray:
        """(8, 2, 24): slopes (w,x , w,y) at the 4 corners and the 4 mid-sides in
        terms of the element DOFs.  Corners (right-hand rotations, as the MITC4
        shear tying and the frame elements): w,x = -R2, w,y = R1.  Mid-sides:
        cubic deflection along the edge (Kirchhoff) for the tangential slope,
        linear normal slope (Batoz & Tahar 1982)."""
        if getattr(self, "_dkq_T", None) is not None:
            return self._dkq_T
        phi = self._dkmq_phi() if self.dkmq else np.zeros(4)
        self._edge_shear = np.zeros((4, 24))
        T = np.zeros((8, 2, 24))
        for i in range(4):
            T[i, 0, 6 * i + 4] = -1.0  # w,x = -theta2
            T[i, 1, 6 * i + 3] = 1.0   # w,y = theta1
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
            d = self.xy[j] - self.xy[i]
            L = float(np.linalg.norm(d))
            sv = d / L
            nv = np.array([-sv[1], sv[0]])
            # slope components of the corner slopes along s and n
            bs_i, bs_j = sv @ T[i], sv @ T[j]
            bn_i, bn_j = nv @ T[i], nv @ T[j]
            ws = np.zeros(24)
            ws[6 * i + 2] = -1.5 / L
            ws[6 * j + 2] = 1.5 / L
            # tangential rotation at mid-edge: a Timoshenko strip along the edge (quadratic
            # rotation, constant shear gamma = -(phi/3)(b_i + b_j - 2 b_m)); phi = 0: Kirchhoff
            f = phi[k]
            bs_mid = (ws + (0.5 * f - 0.25) * (bs_i + bs_j)) / (1.0 + f)
            self._edge_shear[k] = -(f / 3.0) * (bs_i + bs_j - 2.0 * bs_mid)
            bn_mid = 0.5 * (bn_i + bn_j)
            T[4 + k] = np.outer(sv, bs_mid) + np.outer(nv, bn_mid)
        self._dkq_T = T
        return T

    def _dkmq_phi(self) -> np.ndarray:
        """phi_k = 12 D / (Ds L_k^2) of the four edges (Katili: 2/(k(1-nu)) (t/L)^2)."""
        _, Db, Ds = self.constitutive()
        D = 0.5 * (Db[0, 0] + Db[1, 1])
        S = 0.5 * (Ds[0, 0] + Ds[1, 1])
        out = np.zeros(4)
        if S <= 0:
            return out
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
            L = float(np.linalg.norm(self.xy[j] - self.xy[i]))
            out[k] = 12.0 * D / (S * L * L)
        return out

    def _dkmq_Bs(self, xi, eta, J) -> np.ndarray:
        """Transverse shear (gamma_xz, gamma_yz) of the DKMQ: the constant edge
        shears tied covariantly (as MITC4) and mapped to x, y."""
        self._dkq_nodal()
        g = self._edge_shear
        Lk = [float(np.linalg.norm(self.xy[j] - self.xy[i])) for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))]
        # covariant components: edge 1 (+xi, eta=-1), 2 (+eta, xi=+1), 3 (-xi, eta=+1), 4 (-eta, xi=-1)
        g_xi = 0.5 * (1 - eta) * g[0] * Lk[0] / 2 - 0.5 * (1 + eta) * g[2] * Lk[2] / 2
        g_eta = 0.5 * (1 + xi) * g[1] * Lk[1] / 2 - 0.5 * (1 - xi) * g[3] * Lk[3] / 2
        return np.linalg.solve(J, np.array([g_xi, g_eta]))

    def _dkq_Bb(self, xi, eta, J) -> np.ndarray:
        """Curvature matrix (3 x 24) of the DKQ: kappa = (bx,x , by,y , bx,y + by,x)
        with the slopes b = grad w interpolated by the 8-node serendipity functions."""
        T = self._dkq_nodal()
        dN = np.linalg.solve(J, _serendipity_derivatives(xi, eta))  # (2, 8): d/dx, d/dy
        bx = T[:, 0, :]  # (8, 24)
        by = T[:, 1, :]
        Bb = np.zeros((3, 24))
        # same sign as the MITC4 curvatures (theta2,x , -theta1,y , ...) = -(w,xx , w,yy , 2 w,xy)
        Bb[0] = -(dN[0] @ bx)
        Bb[1] = -(dN[1] @ by)
        Bb[2] = -(dN[1] @ bx + dN[0] @ by)
        return Bb

    # --------------------------------------------------------- constitutive
    def constitutive(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        el = self.el
        E, nu = el.E, el.nu
        t, tb = el.thickness, el.bending_thickness
        f11, f22, f12, m11, m22, m12, v13, v23 = el.modifiers
        plane = E / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        Fm = np.array([[f11, np.sqrt(f11 * f22), 0], [np.sqrt(f11 * f22), f22, 0], [0, 0, f12]])
        Fb = np.array([[m11, np.sqrt(m11 * m22), 0], [np.sqrt(m11 * m22), m22, 0], [0, 0, m12]])
        Dm = plane * t * Fm
        Db = plane * tb ** 3 / 12.0 * Fb
        Gs = el.G * (5.0 / 6.0) * tb
        if el.model_type == "ShellThin":
            Gs *= THIN_SHEAR_FACTOR
        Ds = np.diag([Gs * v13, Gs * v23])
        if el.model_type == "Membrane":
            Db = Db * 0.0
            Ds = Ds * 0.0
        elif el.model_type == "Plate":
            Dm = Dm * 0.0
        return Dm, Db, Ds

    # ------------------------------------------------------------ stiffness
    def _incompatible_B(self, xi, eta):
        """Strain matrix (3 x 4) of the two incompatible bending modes
        (1 - xi^2, 1 - eta^2) in u and v, QM6 style: the Jacobian of the
        incompatible part is evaluated at the element centre so the element
        passes the constant strain patch test."""
        _, _, J0, det0 = self._jacobian(0.0, 0.0)
        Jinv0 = np.linalg.inv(J0)
        _, _, J, det = self._jacobian(xi, eta)
        # natural derivatives of the incompatible modes
        dN = np.array([[-2.0 * xi, 0.0], [0.0, -2.0 * eta]])  # rows: modes, cols: d/dxi, d/deta
        d = (Jinv0 @ dN.T) * (det0 / det)  # Taylor et al. correction
        B = np.zeros((3, 4))
        for a in range(2):
            dx, dy = d[0, a], d[1, a]
            B[0, 2 * a] = dx
            B[1, 2 * a + 1] = dy
            B[2, 2 * a] = dy
            B[2, 2 * a + 1] = dx
        return B

    def _incompatible_Bd(self, xi, eta):
        """Continuum rotation 1/2 (dv/dx - du/dy) of the incompatible modes (1 x 4)."""
        _, _, J0, det0 = self._jacobian(0.0, 0.0)
        _, _, J, det = self._jacobian(xi, eta)
        dN = np.array([[-2.0 * xi, 0.0], [0.0, -2.0 * eta]])
        d = (np.linalg.inv(J0) @ dN.T) * (det0 / det)
        Bd = np.zeros((1, 4))
        for a in range(2):
            dx, dy = d[0, a], d[1, a]
            Bd[0, 2 * a] = -0.5 * dy      # u_a contributes -1/2 du/dy
            Bd[0, 2 * a + 1] = 0.5 * dx   # v_a contributes +1/2 dv/dx
        return Bd

    def stiffness_local(self) -> np.ndarray:
        if self._K is not None:
            return self._K
        Dm, Db, Ds = self.constitutive()
        K = np.zeros((24, 24))
        Kaa = np.zeros((4, 4))
        Kua = np.zeros((24, 4))
        area = 0.0
        trib = np.zeros(4)
        drill = DRILLING_FACTOR * self.el.G * self.el.thickness
        for xi in _G2:
            for eta in _G2:
                N, Bm, Bb, Bs, Bd, det = self._matrices(xi, eta)
                K += (Bm.T @ Dm @ Bm + Bb.T @ Db @ Bb + Bs.T @ Ds @ Bs) * det
                if not DRILL_ONE_POINT:
                    K += Bd.T @ Bd * drill * det
                if INCOMPATIBLE and self.el.model_type != "Plate":
                    Ba = self._incompatible_B(xi, eta)
                    Kaa += Ba.T @ Dm @ Ba * det
                    Kua += Bm.T @ Dm @ Ba * det
                    if DRILL_INCOMPATIBLE and not DRILL_ONE_POINT and drill:
                        Bda = self._incompatible_Bd(xi, eta)
                        Kaa += Bda.T @ Bda * drill * det
                        Kua += Bd.T @ Bda * drill * det
                area += det
                trib += N * det
        if DRILL_ONE_POINT or DRILLING_FACTOR_1PT:
            N, Bm, Bb, Bs, Bd, det = self._matrices(0.0, 0.0)
            g1 = (drill if DRILL_ONE_POINT else 0.0) + DRILLING_FACTOR_1PT * self.el.G * self.el.thickness
            K += Bd.T @ Bd * g1 * det * 4.0
        if INCOMPATIBLE and self.el.model_type != "Plate" and np.max(np.abs(Kaa)) > 0:
            # static condensation of the incompatible modes
            K -= Kua @ np.linalg.solve(Kaa, Kua.T)
        # small stabilising stiffness on the rotations a membrane / plate does not resist.
        # Never on translations: for a wall the out-of-plane translation is a global
        # horizontal DOF and a spring there would leak load to the ground (truly free
        # translations are caught and reported by the null-DOF check of the assembler).
        if self.el.model_type in ("Membrane", "Plate"):
            scale = 1e-7 * max(np.abs(np.diag(K)).max(), 1.0)
            for i in range(4):
                for d in (3, 4) if self.el.model_type == "Membrane" else (5,):
                    K[6 * i + d, 6 * i + d] += scale
        self._K = 0.5 * (K + K.T)
        self.area = area
        self.tributary = trib
        return self._K

    def stiffness_global(self) -> np.ndarray:
        K = self.stiffness_local()
        return self.R.T @ K @ self.R

    def lumped_mass(self) -> np.ndarray:
        """Translational mass at the four nodes (kg)."""
        self.stiffness_local()
        return self.el.mass_per_area * self.tributary

    # ---------------------------------------------------------------- loads
    def load_vector_global(self, pattern: str, self_weight: float = 0.0) -> np.ndarray:
        self.stiffness_local()
        f_local = np.zeros(24)
        f_global = np.zeros(24)
        for ld in self.el.loads:
            if ld.pattern != pattern:
                continue
            if ld.kind == "pressure":
                f_local[2::6] += float(ld.value[0]) * self.tributary
            elif ld.kind == "force":
                for i in range(4):
                    f_global[6 * i:6 * i + 3] += np.asarray(ld.value, float) * self.tributary[i]
        if self_weight and self.el.weight_per_area:
            b = np.array([0.0, 0.0, -self.el.weight_per_area * self_weight])
            f_local += self._body_force_local(self.axes @ b)
        return self.R.T @ f_local + f_global

    def _body_force_local(self, b_local: np.ndarray) -> np.ndarray:
        """Consistent nodal loads (local DOFs) of a uniform force per area
        ``b_local`` = (b1, b2, b3).  In-plane parts load u, v and, through the
        Allman edge modes, the drilling rotations; the normal part is lumped."""
        f = np.zeros(24)
        f[2::6] += b_local[2] * self.tributary
        if abs(b_local[0]) + abs(b_local[1]) == 0.0:
            return f
        if not (ALLMAN and self.el.model_type != "Plate"):
            f[0::6] += b_local[0] * self.tributary
            f[1::6] += b_local[1] * self.tributary
            return f
        for xi in _G2:
            for eta in _G2:
                N, _, _, det = self._jacobian(xi, eta)
                Nm = _midside_values(xi, eta)
                for i in range(4):
                    f[6 * i] += b_local[0] * N[i] * det
                    f[6 * i + 1] += b_local[1] * N[i] * det
                for k, (i, jn) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
                    ax = (self.xy[jn, 1] - self.xy[i, 1]) / 8.0
                    ay = (self.xy[i, 0] - self.xy[jn, 0]) / 8.0
                    w = Nm[k] * (b_local[0] * ax + b_local[1] * ay) * det
                    f[6 * jn + 5] += w
                    f[6 * i + 5] -= w
        return f

    # ------------------------------------------------------------- recovery
    def resultants(self, u_global: np.ndarray) -> dict:
        """Stress resultants at the 2x2 Gauss points and extrapolated to the
        nodes.  Keys: F (membrane F11,F22,F12), M (M11,M22,M12), V (V13,V23)."""
        u = self.R @ u_global
        Dm, Db, Ds = self.constitutive()
        gp = []
        F, M, V = [], [], []
        for xi in _G2:
            for eta in _G2:
                _, Bm, Bb, Bs, _, _ = self._matrices(xi, eta)
                F.append(Dm @ Bm @ u)
                M.append(Db @ Bb @ u)
                V.append(Ds @ Bs @ u)
                gp.append((xi, eta))
        F, M, V = np.array(F), np.array(M), np.array(V)
        # bilinear extrapolation Gauss -> nodes
        s = np.sqrt(3.0)
        ext = np.array([[_shape(s * xi_n, s * eta_n)[0][k] for k in range(4)]
                        for xi_n, eta_n in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        # gauss ordering above is (xi,eta) = (-,-),(-,+),(+,-),(+,+): reorder to CCW (-,-),(+,-),(+,+),(-,+)
        order = [0, 2, 3, 1]
        if self.dkq:
            V = self._dkq_shear(M, gp)
        # CSI sign (S11 = F11/t - 12 M11 x3 / t^3): a positive moment compresses the
        # +3 face - a gravity slab sags with positive M11 / M22 at mid-span
        M = -M
        Fn = ext @ F[order]
        Mn = ext @ M[order]
        Vn = ext @ V[order]
        return {"gauss": gp, "F": F, "M": M, "V": V, "F_nodes": Fn, "M_nodes": Mn, "V_nodes": Vn}

    def _dkq_shear(self, M: np.ndarray, gp) -> np.ndarray:
        """Transverse shears of the thin plate from moment equilibrium
        (V13 = M11,1 + M12,2 ; V23 = M12,1 + M22,2) with a linear fit of the
        Gauss-point moments - constant over the element."""
        pts = []
        for xi, eta in gp:
            N, _ = _shape(xi, eta)
            pts.append(N @ self.xy)
        A = np.column_stack([np.ones(len(pts)), np.array(pts)])
        coef, *_ = np.linalg.lstsq(A, M, rcond=None)  # rows: const, d/dx, d/dy ; cols M11 M22 M12
        v13 = coef[1, 0] + coef[2, 2]
        v23 = coef[1, 2] + coef[2, 1]
        return np.tile([v13, v23], (len(gp), 1)) * SHEAR_SIGN

    def geometric_stiffness_global(self, u_global: np.ndarray) -> np.ndarray:
        """Initial membrane-stress geometric stiffness (transverse w only)."""
        u = self.R @ u_global
        Dm, _, _ = self.constitutive()
        Kg = np.zeros((24, 24))
        idx = np.arange(2, 24, 6)
        for xi in _G2:
            for eta in _G2:
                _, Bm, _, _, _, det = self._matrices(xi, eta)
                _, dNdx, _, _ = self._jacobian(xi, eta)
                Nx, Ny, Nxy = Dm @ Bm @ u
                stress = np.array([[Nx, Nxy], [Nxy, Ny]])
                Kg[np.ix_(idx, idx)] += dNdx.T @ stress @ dNdx * det
        return self.R.T @ Kg @ self.R
