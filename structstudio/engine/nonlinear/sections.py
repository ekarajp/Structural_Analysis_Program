"""Cross sections for nonlinear frame elements and beam integration rules.

Section deformation / force order (OpenSees 3D order):

    e = [eps_a, kappa_z, kappa_y, twist]     s = [P, Mz, My, T]

with fiber strain ``eps = eps_a - y kappa_z + z kappa_y``.  Local ``y`` is
the CSI axis 2 and ``z`` the CSI axis 3, so ``Mz`` is CSI ``M3`` and ``My``
is ``-M2`` (CSI's M2 sign is opposite to the right-hand rule).  Torsion is
elastic (``GJ``) and uncoupled, as in an OpenSees fiber section with
``-GJ``.

Section builders (``rc_rectangle``, ``rc_circle``, ``steel_i``, ``steel_box``,
``steel_pipe``, ``patch_*``, ``layer_*``) follow the OpenSees patch / layer
discretisation so that models can be cross-checked fiber by fiber.
"""
from __future__ import annotations

import copy as _copy
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .materials import UniaxialMaterial


# ============================================================================
# Fiber geometry helpers
# ============================================================================
@dataclass
class FiberSet:
    material: UniaxialMaterial  # template (state is not used)
    y: np.ndarray
    z: np.ndarray
    A: np.ndarray


def patch_rect(material, ny: int, nz: int, yI: float, zI: float, yJ: float, zJ: float) -> FiberSet:
    """Rectangular patch from corner (yI, zI) to (yJ, zJ), ny x nz fibers."""
    return patch_quad(material, ny, nz, (yI, zI), (yJ, zI), (yJ, zJ), (yI, zJ))


def patch_quad(material, nIJ: int, nJK: int, I, J, K, L) -> FiberSet:
    """Quadrilateral patch I-J-K-L (counter clockwise), bilinear subdivision
    (OpenSees ``patch quad``)."""
    P = np.array([I, J, K, L], float)
    ys, zs, As = [], [], []
    for a in range(nIJ):
        for b in range(nJK):
            s0, s1 = a / nIJ, (a + 1) / nIJ
            t0, t1 = b / nJK, (b + 1) / nJK
            corners = []
            for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1)):
                N = np.array([(1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t])
                corners.append(N @ P)
            c = np.array(corners)
            area, cen = _polygon_area_centroid(c)
            ys.append(cen[0])
            zs.append(cen[1])
            As.append(area)
    return FiberSet(material, np.array(ys), np.array(zs), np.array(As))


def patch_circ(material, n_circ: int, n_rad: int, yc: float, zc: float, r_in: float, r_out: float,
               a0: float = 0.0, a1: float = 360.0) -> FiberSet:
    """Circular (annular) patch, angles in degrees (OpenSees ``patch circ``)."""
    ys, zs, As = [], [], []
    da = math.radians(a1 - a0) / n_circ
    dr = (r_out - r_in) / n_rad
    for i in range(n_rad):
        ri, ro = r_in + i * dr, r_in + (i + 1) * dr
        for j in range(n_circ):
            t0 = math.radians(a0) + j * da
            t1 = t0 + da
            area = 0.5 * da * (ro * ro - ri * ri)
            # centroid of an annular sector
            rc = (2.0 / 3.0) * (ro ** 3 - ri ** 3) / (ro ** 2 - ri ** 2) * math.sin(da / 2) / (da / 2)
            tm = 0.5 * (t0 + t1)
            ys.append(yc + rc * math.cos(tm))
            zs.append(zc + rc * math.sin(tm))
            As.append(area)
    return FiberSet(material, np.array(ys), np.array(zs), np.array(As))


def layer_straight(material, n: int, area: float, yI: float, zI: float, yJ: float, zJ: float) -> FiberSet:
    """``n`` bars of ``area`` evenly spaced from (yI, zI) to (yJ, zJ) inclusive."""
    if n == 1:
        t = np.array([0.5])
    else:
        t = np.linspace(0.0, 1.0, n)
    return FiberSet(material, yI + t * (yJ - yI), zI + t * (zJ - zI), np.full(n, float(area)))


def layer_circ(material, n: int, area: float, yc: float, zc: float, r: float, a0: float = 0.0,
               a1: Optional[float] = None) -> FiberSet:
    if a1 is None:
        a1 = 360.0 - 360.0 / n
    t = np.radians(np.linspace(a0, a1, n)) if n > 1 else np.radians([a0])
    return FiberSet(material, yc + r * np.cos(t), zc + r * np.sin(t), np.full(n, float(area)))


def _polygon_area_centroid(P: np.ndarray):
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = 0.5 * cr.sum()
    if abs(A) < 1e-300:
        return 0.0, P.mean(axis=0)
    cx = ((x + x1) * cr).sum() / (6 * A)
    cy = ((y + y1) * cr).sum() / (6 * A)
    return abs(A), np.array([cx, cy])


# ============================================================================
# Sections
# ============================================================================
class Section3D:
    order = ("P", "Mz", "My", "T")

    def set_trial(self, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def commit(self): ...

    def revert(self): ...

    def reset(self): ...

    def initial_stiffness(self) -> np.ndarray:
        raise NotImplementedError

    def copy(self) -> "Section3D":
        return _copy.deepcopy(self)


class ElasticSection3D(Section3D):
    def __init__(self, EA: float, EIz: float, EIy: float, GJ: float):
        self.k = np.diag([EA, EIz, EIy, GJ]).astype(float)
        self.e = np.zeros(4)
        self.s = np.zeros(4)

    def set_trial(self, e):
        self.e = np.asarray(e, float).copy()
        self.s = self.k @ self.e
        return self.s, self.k

    def initial_stiffness(self):
        return self.k.copy()

    @property
    def tangent(self):
        return self.k


class FiberSection3D(Section3D):
    """Fiber section: fibers grouped by material for vectorised state determination."""

    def __init__(self, fiber_sets: Sequence[FiberSet], GJ: float, name: str = ""):
        self.name = name
        self.GJ = float(GJ)
        # merge sets that share the same material template object
        groups = {}
        for fs in fiber_sets:
            key = id(fs.material)
            if key not in groups:
                groups[key] = [fs.material, [], [], []]
            g = groups[key]
            g[1].append(fs.y)
            g[2].append(fs.z)
            g[3].append(fs.A)
        self.groups = []
        for mat, ys, zs, As in groups.values():
            y, z, A = np.concatenate(ys), np.concatenate(zs), np.concatenate(As)
            self.groups.append((mat.clone(len(y)), y, z, A))
        self.e = np.zeros(4)
        self.s = np.zeros(4)
        self.ks = self.initial_stiffness()

    # ---------------------------------------------------------------- geometry
    @property
    def area(self) -> float:
        return float(sum(A.sum() for _, _, _, A in self.groups))

    def centroid(self, weighted: bool = True) -> Tuple[float, float]:
        num_y = num_z = den = 0.0
        for mat, y, z, A in self.groups:
            w = A * (mat.initial_tangent if weighted else 1.0)
            num_y += float(w @ y)
            num_z += float(w @ z)
            den += float(w.sum())
        return (num_y / den, num_z / den) if den else (0.0, 0.0)

    def fibers(self):
        """(material type, y, z, A) of every fiber (for plotting / export)."""
        for mat, y, z, A in self.groups:
            for i in range(len(y)):
                yield mat.type_name, float(y[i]), float(z[i]), float(A[i])

    # ------------------------------------------------------------------- state
    @staticmethod
    def _k_from(Et, y, z, A):
        EA = Et * A
        k = np.zeros((4, 4))
        k[0, 0] = EA.sum()
        k[0, 1] = k[1, 0] = -(EA * y).sum()
        k[0, 2] = k[2, 0] = (EA * z).sum()
        k[1, 1] = (EA * y * y).sum()
        k[1, 2] = k[2, 1] = -(EA * y * z).sum()
        k[2, 2] = (EA * z * z).sum()
        return k

    def initial_stiffness(self) -> np.ndarray:
        k = np.zeros((4, 4))
        for mat, y, z, A in self.groups:
            k += self._k_from(mat.initial_tangent, y, z, A)
        k[3, 3] = self.GJ
        return k

    def set_trial(self, e):
        e = np.asarray(e, float)
        self.e = e.copy()
        s = np.zeros(4)
        k = np.zeros((4, 4))
        for mat, y, z, A in self.groups:
            eps = e[0] - y * e[1] + z * e[2]
            mat.set_trial(eps)
            sA = mat.stress * A
            s[0] += sA.sum()
            s[1] -= (sA * y).sum()
            s[2] += (sA * z).sum()
            k += self._k_from(mat.tangent, y, z, A)
        s[3] = self.GJ * e[3]
        k[3, 3] = self.GJ
        self.s, self.ks = s, k
        return s, k

    @property
    def tangent(self):
        return self.ks

    def commit(self):
        for mat, *_ in self.groups:
            mat.commit()
        self._ce = self.e.copy()

    def revert(self):
        for mat, *_ in self.groups:
            mat.revert()
        self.e = getattr(self, "_ce", np.zeros(4)).copy()
        # recompute forces at the committed state
        s = np.zeros(4)
        k = np.zeros((4, 4))
        for mat, y, z, A in self.groups:
            sA = mat.stress * A
            s[0] += sA.sum()
            s[1] -= (sA * y).sum()
            s[2] += (sA * z).sum()
            k += self._k_from(mat.tangent, y, z, A)
        s[3] = self.GJ * self.e[3]
        k[3, 3] = self.GJ
        self.s, self.ks = s, k

    def reset(self):
        for mat, *_ in self.groups:
            mat.reset()
        self.e = np.zeros(4)
        self._ce = np.zeros(4)
        self.s = np.zeros(4)
        self.ks = self.initial_stiffness()

    def fiber_states(self):
        """List of (material type, y, z, strain, stress) arrays per material group."""
        return [(mat.type_name, y, z, mat.strain.copy(), mat.stress.copy()) for mat, y, z, A in self.groups]


# ============================================================================
# Section builders
# ============================================================================
def rc_rectangle(depth: float, width: float, cover: float, core, cover_mat, steel, n_top: int, n_bot: int,
                 bar_area: float, n_side: int = 0, side_bar_area: Optional[float] = None, GJ: float = 0.0,
                 ny: int = 12, nz: int = 6, name: str = "") -> FiberSection3D:
    """Rectangular RC section.  ``depth`` along local 2 (y, CSI t3), ``width``
    along local 3 (z, CSI t2); ``cover`` to the bar centres.  Bars: ``n_top``
    at +y, ``n_bot`` at -y, ``n_side`` intermediate bars on each side face.
    Core concrete inside the bar centre-lines, cover concrete around it."""
    hy, hz = depth / 2.0, width / 2.0
    cy, cz = hy - cover, hz - cover
    ncy = max(1, int(round(ny * cy / hy)))
    ncz = max(1, int(round(nz * cz / hz)))
    ncov = max(1, int(round(ny * cover / depth)))
    sets = [patch_rect(core, 2 * ncy, 2 * ncz, -cy, -cz, cy, cz)]
    # cover: top/bottom strips (full width) and side strips (between)
    sets.append(patch_rect(cover_mat, ncov, 2 * nz, cy, -hz, hy, hz))
    sets.append(patch_rect(cover_mat, ncov, 2 * nz, -hy, -hz, -cy, hz))
    sets.append(patch_rect(cover_mat, 2 * ncy, ncov, -cy, -hz, cy, -cz))
    sets.append(patch_rect(cover_mat, 2 * ncy, ncov, -cy, cz, cy, hz))
    sets.append(layer_straight(steel, n_top, bar_area, cy, -cz, cy, cz))
    sets.append(layer_straight(steel, n_bot, bar_area, -cy, -cz, -cy, cz))
    if n_side:
        a = side_bar_area if side_bar_area is not None else bar_area
        sp = 2 * cy / (n_side + 1)
        sets.append(layer_straight(steel, n_side, a, -cy + sp, -cz, cy - sp, -cz))
        sets.append(layer_straight(steel, n_side, a, -cy + sp, cz, cy - sp, cz))
    return FiberSection3D(sets, GJ, name)


def rc_circle(diameter: float, cover: float, core, cover_mat, steel, n_bars: int, bar_area: float, GJ: float = 0.0,
              n_circ: int = 24, n_rad_core: int = 8, n_rad_cover: int = 2, name: str = "") -> FiberSection3D:
    R = diameter / 2.0
    rc = R - cover
    sets = [patch_circ(core, n_circ, n_rad_core, 0.0, 0.0, 0.0, rc),
            patch_circ(cover_mat, n_circ, n_rad_cover, 0.0, 0.0, rc, R),
            layer_circ(steel, n_bars, bar_area, 0.0, 0.0, rc)]
    return FiberSection3D(sets, GJ, name)


def steel_i(d: float, bf: float, tf: float, tw: float, material, GJ: float = 0.0, n_flange: int = 4,
            n_web: int = 12, nf_width: int = 8, bf_bot: Optional[float] = None, tf_bot: Optional[float] = None,
            strong_axis: str = "z", name: str = "") -> FiberSection3D:
    """I / W shape.  ``strong_axis='z'``: web along y (bending M3 = strong axis, CSI default
    for I sections: t3 = depth along local 2)."""
    bfb = bf if bf_bot is None else bf_bot
    tfb = tf if tf_bot is None else tf_bot
    hw = d - tf - tfb
    y0 = -d / 2.0
    sets = [patch_rect(material, n_flange, nf_width, y0, -bfb / 2, y0 + tfb, bfb / 2),
            patch_rect(material, n_web, 1, y0 + tfb, -tw / 2, y0 + tfb + hw, tw / 2),
            patch_rect(material, n_flange, nf_width, d / 2 - tf, -bf / 2, d / 2, bf / 2)]
    sec = FiberSection3D(sets, GJ, name)
    if strong_axis == "y":
        sec.groups = [(m, z.copy(), y.copy(), A) for m, y, z, A in sec.groups]
        sec.ks = sec.initial_stiffness()
    # shift to the elastic centroid (unequal flanges)
    yc, zc = sec.centroid(weighted=False)
    if abs(yc) > 1e-12 or abs(zc) > 1e-12:
        sec.groups = [(m, y - yc, z - zc, A) for m, y, z, A in sec.groups]
        sec.ks = sec.initial_stiffness()
    return sec


def steel_box(d: float, b: float, tf: float, tw: float, material, GJ: float = 0.0, n: int = 4,
              name: str = "") -> FiberSection3D:
    hy, hz = d / 2.0, b / 2.0
    sets = [patch_rect(material, n, 8, hy - tf, -hz, hy, hz),
            patch_rect(material, n, 8, -hy, -hz, -hy + tf, hz),
            patch_rect(material, 12, n, -hy + tf, -hz, hy - tf, -hz + tw),
            patch_rect(material, 12, n, -hy + tf, hz - tw, hy - tf, hz)]
    return FiberSection3D(sets, GJ, name)


def steel_pipe(D: float, t: float, material, GJ: float = 0.0, n_circ: int = 24, n_rad: int = 2,
               name: str = "") -> FiberSection3D:
    return FiberSection3D([patch_circ(material, n_circ, n_rad, 0.0, 0.0, D / 2 - t, D / 2)], GJ, name)


# ============================================================================
# Beam integration
# ============================================================================
class BeamIntegration:
    """Integration points ``xi`` (0..1 along the flexible length) and weights
    (summing to 1); ``section_kind[i]`` is "hinge_i", "hinge_j", "interior" or
    "all" (every point uses the element's section)."""

    name = "Generic"

    def __init__(self, xi, w, kinds=None):
        self.xi = np.asarray(xi, float)
        self.w = np.asarray(w, float)
        self.kinds = list(kinds) if kinds is not None else ["all"] * len(self.xi)

    @property
    def n(self) -> int:
        return len(self.xi)


def lobatto(n: int = 5) -> BeamIntegration:
    """Gauss-Lobatto (OpenSees ``Lobatto``; includes both element ends)."""
    if n < 2:
        raise ValueError("Lobatto integration needs at least 2 points")
    if n == 2:
        x, w = np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    else:
        Pn1 = np.polynomial.legendre.Legendre.basis(n - 1)
        inner = np.sort(Pn1.deriv().roots().real)
        x = np.concatenate([[-1.0], inner, [1.0]])
        w = 2.0 / (n * (n - 1) * Pn1(x) ** 2)
    bi = BeamIntegration(0.5 * (x + 1.0), 0.5 * w)
    bi.name = f"Lobatto({n})"
    return bi


def legendre(n: int = 3) -> BeamIntegration:
    x, w = np.polynomial.legendre.leggauss(n)
    bi = BeamIntegration(0.5 * (x + 1.0), 0.5 * w)
    bi.name = f"Legendre({n})"
    return bi


def radau(n: int = 3) -> BeamIntegration:
    """Gauss-Radau with the end-I point (OpenSees ``Radau``)."""
    if n == 1:
        bi = BeamIntegration([0.0], [1.0])
    else:
        # roots of (P_{n-1} + P_n) / (1 + x)
        c = np.zeros(n + 1)
        c[n - 1] = 1.0
        c[n] = 1.0
        roots = np.sort(np.polynomial.legendre.Legendre(c).roots().real)
        roots[0] = -1.0
        Pn1 = np.polynomial.legendre.Legendre.basis(n - 1)
        w = (1 - roots) / (n * n * Pn1(roots) ** 2)
        w[0] = 2.0 / n ** 2
        bi = BeamIntegration(0.5 * (roots + 1.0), 0.5 * w)
    bi.name = f"Radau({n})"
    return bi


def hinge_radau(lpI: float, lpJ: float, L: float) -> BeamIntegration:
    """Modified Gauss-Radau plastic hinge integration (Scott & Fenves 2006,
    OpenSees ``HingeRadau``): two Radau points over 4 lp at each end with the
    hinge section, two Gauss points for the elastic interior."""
    a = 1.0 / L
    alpha = 0.5 - 2.0 * (lpI + lpJ) * a
    beta = 0.5 + 2.0 * (lpI - lpJ) * a
    g = 1.0 / math.sqrt(3.0)
    xi = [0.0, 8.0 / 3.0 * lpI * a, -alpha * g + beta, alpha * g + beta, 1.0 - 8.0 / 3.0 * lpJ * a, 1.0]
    w = [lpI * a, 3.0 * lpI * a, alpha, alpha, 3.0 * lpJ * a, lpJ * a]
    bi = BeamIntegration(xi, w, ["hinge_i", "hinge_i", "interior", "interior", "hinge_j", "hinge_j"])
    bi.name = "HingeRadau"
    return bi


def make_integration(kind: str, n: int = 5, lpI: float = 0.0, lpJ: float = 0.0, L: float = 1.0) -> BeamIntegration:
    k = kind.lower()
    if k == "lobatto":
        return lobatto(n)
    if k == "legendre":
        return legendre(n)
    if k == "radau":
        return radau(n)
    if k == "hingeradau":
        return hinge_radau(lpI, lpJ, L)
    raise ValueError(f"Unknown beam integration {kind!r}")


# ============================================================================
# Stacked sections: all integration points of an element in one vector block
# ============================================================================
class StackedSections:
    """``n_pts`` copies of one section evaluated together.

    Fiber sections: every material group holds the fibers of all points
    (fiber-major per point), so one material call serves the whole element.
    Elastic sections are evaluated in closed form."""

    def __init__(self, template: Section3D, n_pts: int):
        self.n = int(n_pts)
        self.template = template
        if isinstance(template, FiberSection3D):
            self.fiber = True
            self.GJ = template.GJ
            self.groups = []
            for mat, y, z, A in template.groups:
                nf = len(y)
                self.groups.append((mat.clone(nf * self.n), np.tile(y, self.n), np.tile(z, self.n),
                                    np.tile(A, self.n), nf))
            k0 = template.initial_stiffness()
        else:
            self.fiber = False
            k0 = template.initial_stiffness()
        self.k0 = np.repeat(k0[None], self.n, axis=0)
        self.e = np.zeros((self.n, 4))
        self.s = np.zeros((self.n, 4))
        self.ks = self.k0.copy()
        self._ce = self.e.copy()

    def initial_stiffness(self) -> np.ndarray:
        return self.k0.copy()

    def _forces(self, stress_groups):
        n = self.n
        s = np.zeros((n, 4))
        k = np.zeros((n, 4, 4))
        for (mat, y, z, A, nf), (sig, Et) in zip(self.groups, stress_groups):
            sA = (sig * A).reshape(n, nf)
            yy = y.reshape(n, nf)
            zz = z.reshape(n, nf)
            EA = (Et * A).reshape(n, nf)
            s[:, 0] += sA.sum(1)
            s[:, 1] -= (sA * yy).sum(1)
            s[:, 2] += (sA * zz).sum(1)
            EAy = EA * yy
            EAz = EA * zz
            k[:, 0, 0] += EA.sum(1)
            k[:, 0, 1] -= EAy.sum(1)
            k[:, 0, 2] += EAz.sum(1)
            k[:, 1, 1] += (EAy * yy).sum(1)
            k[:, 1, 2] -= (EAy * zz).sum(1)
            k[:, 2, 2] += (EAz * zz).sum(1)
        k[:, 1, 0] = k[:, 0, 1]
        k[:, 2, 0] = k[:, 0, 2]
        k[:, 2, 1] = k[:, 1, 2]
        return s, k

    def set_trial(self, e: np.ndarray):
        """e: (n, 4) -> s (n, 4), ks (n, 4, 4)."""
        e = np.asarray(e, float)
        self.e = e.copy()
        if not self.fiber:
            k = self.k0
            s = np.einsum("nij,nj->ni", k, e)
            self.s, self.ks = s, k
            return s, k
        out = []
        for mat, y, z, A, nf in self.groups:
            e0 = np.repeat(e[:, 0], nf)
            kz = np.repeat(e[:, 1], nf)
            ky = np.repeat(e[:, 2], nf)
            mat.set_trial(e0 - y * kz + z * ky)
            out.append((mat.stress, mat.tangent))
        s, k = self._forces(out)
        s[:, 3] = self.GJ * e[:, 3]
        k[:, 3, 3] = self.GJ
        self.s, self.ks = s, k
        return s, k

    def commit(self):
        if self.fiber:
            for g in self.groups:
                g[0].commit()
        self._ce = self.e.copy()

    def revert(self):
        self.e = self._ce.copy()
        if self.fiber:
            for g in self.groups:
                g[0].revert()
            s, k = self._forces([(g[0].stress, g[0].tangent) for g in self.groups])
            s[:, 3] = self.GJ * self.e[:, 3]
            k[:, 3, 3] = self.GJ
            self.s, self.ks = s, k
        else:
            self.s = np.einsum("nij,nj->ni", self.k0, self.e)

    def reset(self):
        if self.fiber:
            for g in self.groups:
                g[0].reset()
        self.e = np.zeros((self.n, 4))
        self._ce = self.e.copy()
        self.s = np.zeros((self.n, 4))
        self.ks = self.k0.copy()

    def fiber_states(self, point: int):
        """(material type, y, z, strain, stress) per group at one integration point."""
        out = []
        if not self.fiber:
            return out
        for mat, y, z, A, nf in self.groups:
            sl = slice(point * nf, (point + 1) * nf)
            out.append((mat.type_name, y[sl], z[sl], mat.strain[sl].copy(), mat.stress[sl].copy()))
        return out


class ElementSections:
    """The sections of one frame element, grouped by template: ``templates[i]``
    is the section at integration point i; points that share the *same*
    template object are evaluated as one stacked block."""

    def __init__(self, templates: Sequence[Section3D]):
        self.n = len(templates)
        order = {}
        for i, t in enumerate(templates):
            order.setdefault(id(t), (t, []))[1].append(i)
        self.blocks = [(StackedSections(t, len(idx)), np.array(idx)) for t, idx in order.values()]

    def initial_stiffness(self) -> np.ndarray:
        k = np.zeros((self.n, 4, 4))
        for blk, idx in self.blocks:
            k[idx] = blk.initial_stiffness()
        return k

    def set_trial(self, e: np.ndarray):
        s = np.zeros((self.n, 4))
        k = np.zeros((self.n, 4, 4))
        for blk, idx in self.blocks:
            si, ki = blk.set_trial(e[idx])
            s[idx] = si
            k[idx] = ki
        return s, k

    def commit(self):
        for blk, _ in self.blocks:
            blk.commit()

    def revert(self):
        for blk, _ in self.blocks:
            blk.revert()

    def reset(self):
        for blk, _ in self.blocks:
            blk.reset()

    def state(self):
        s = np.zeros((self.n, 4))
        e = np.zeros((self.n, 4))
        for blk, idx in self.blocks:
            s[idx] = blk.s
            e[idx] = blk.e
        return e, s

    def fiber_states(self, point: int):
        for blk, idx in self.blocks:
            hit = np.where(idx == point)[0]
            if hit.size:
                return blk.fiber_states(int(hit[0]))
        return []
