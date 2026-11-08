"""Nonlinear elements.

Frame elements work in the *basic system* of a simply supported member
(OpenSees convention, Filippou & Fenves 2004):

    v = [elongation, theta_zI, theta_zJ, theta_yI, theta_yJ, twist]
    q = [N,          Mz_I,     Mz_J,     My_I,     My_J,     T    ]

Local axes are the CSI axes (1 along the member, y = 2, z = 3).  The basic
deformations are taken between the faces of the flexible span: joint
offsets and rigid end zones are rigid (the 12x12 ``A`` matrix of the linear
frame formulation is reused), so a nonlinear member sees exactly the same
geometry as the linear one.

Element types
-------------
* :class:`ForceBeamColumn` - force (flexibility) based, exact equilibrium
  along the member, iterative element state determination with
  subdivision (Spacone, Filippou & Taucer 1996; Neuenhofer & Filippou 1997).
* :class:`DispBeamColumn`  - displacement based, cubic transverse / linear
  axial interpolation.
* :class:`ElasticHingeFrame` - elastic (Timoshenko) member with CSI-style
  concentrated hinges (M3, M2, P) in series at its ends: the ETABS / SAP2000
  "frame hinge" model.
* :class:`Link` - zero-length two-joint or grounded link with one uniaxial
  material per DOF (CSI link / OpenSees zeroLength).

Geometric transformations: ``Linear`` and ``PDelta`` (leaning column P-Delta
string stiffness, OpenSees ``PDelta``).  Element span loads (uniform,
trapezoidal, point, self weight) are carried exactly: the simply supported
reactions go straight to the joints and, for the force based element, the
particular section forces enter every integration point.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...mesh.analysis_model import ElementLoad
from ..elements.frame import FrameFormulation, _load_resultant_left
from .materials import UniaxialMaterial
from .sections import BeamIntegration, ElasticSection3D, Section3D

EPS = np.finfo(float).eps


class ElementNonConvergence(RuntimeError):
    """Raised when an element state determination fails (the global solver subdivides the step)."""


def basic_matrix(L: float) -> np.ndarray:
    """6x12 linear map local face displacements -> basic deformations."""
    B = np.zeros((6, 12))
    B[0, 0], B[0, 6] = -1.0, 1.0
    B[1, 1], B[1, 7], B[1, 5] = 1.0 / L, -1.0 / L, 1.0
    B[2, 1], B[2, 7], B[2, 11] = 1.0 / L, -1.0 / L, 1.0
    B[3, 2], B[3, 8], B[3, 4] = -1.0 / L, 1.0 / L, 1.0
    B[4, 2], B[4, 8], B[4, 10] = -1.0 / L, 1.0 / L, 1.0
    B[5, 3], B[5, 9] = -1.0, 1.0
    return B


def _face_loads(loads: Sequence[ElementLoad], L: float, ri: float, rj: float) -> List[ElementLoad]:
    """Loads re-expressed on the flexible span (x from face I).  Parts lying in
    rigid zones are moved to the faces as point forces."""
    if not ri and not rj:
        return [ld for ld in loads if ld.kind != "temperature"]
    out = []
    a, b = ri, L - rj
    for ld in loads:
        if ld.kind == "temperature":
            continue
        if ld.kind == "point":
            x = min(max(ld.x_a, a), b) - a
            out.append(dataclasses.replace(ld, x_a=x, x_b=x))
            continue
        span = ld.x_b - ld.x_a

        def val(x):
            t = (x - ld.x_a) / span if span > 0 else 0.0
            return ld.value_a + (ld.value_b - ld.value_a) * t

        lo, hi = max(ld.x_a, a), min(ld.x_b, b)
        if hi > lo:
            out.append(ElementLoad(ld.pattern, "distributed", ld.load_type, lo - a, hi - a, val(lo), val(hi)))
        for z0, z1, xf in ((ld.x_a, min(ld.x_b, a), 0.0), (max(ld.x_a, b), ld.x_b, b - a)):
            if z1 > z0:
                tot = 0.5 * (val(z0) + val(z1)) * (z1 - z0)
                out.append(ElementLoad(ld.pattern, "point", ld.load_type, xf, xf, tot, tot.copy()))
    return out


def ss_reactions(loads: Sequence[ElementLoad], L: float) -> np.ndarray:
    """Reactions (12, local, force the supports apply to the member) of the
    simply supported basic system: pin + torsion support at I, roller at J."""
    F = np.zeros(3)
    M = np.zeros(3)
    for ld in loads:
        f, m = _load_resultant_left(ld, L, None)
        F += f
        M += m
    R = np.zeros(12)
    R[0] = -F[0]
    R[3] = -M[0]
    R[1] = M[2] / L
    R[2] = -M[1] / L
    R[7] = -F[1] - R[1]
    R[8] = -F[2] - R[2]
    return R


def ss_section_forces(loads: Sequence[ElementLoad], L: float, R: np.ndarray, x: float) -> np.ndarray:
    """Particular section forces [P, Mz, My, T] of the simply supported member at x."""
    Fs = R[:3].copy()
    Ms = R[3:6].copy() + np.cross(np.array([-x, 0.0, 0.0]), R[:3])
    for ld in loads:
        f, m = _load_resultant_left(ld, x, L)
        Fs += f
        Ms += m
    S, Mm = -Fs, -Ms
    return np.array([S[0], Mm[2], Mm[1], Mm[0]])


# ============================================================================
class NLFrame:
    """Common machinery of the nonlinear frame elements."""

    kind = "Frame"
    uses_flexible_length_axial = True

    def __init__(self, fe: FrameFormulation, geom: str = "Linear"):
        self.fe = fe
        self.el = fe.el
        self.id = fe.el.id
        self.nodes = (fe.el.node_i, fe.el.node_j)
        k = fe.kin
        self.L = k.Lf
        self.A = k.T_rigid @ k.R12 @ k.T_off
        self.B = basic_matrix(self.L)
        self.crd = None
        self.set_geometry(geom)
        # Fiber members deform over the flexible span only (rigid zones are rigid in every
        # direction).  CSI's linear frame lets axial and torsion act over the full length;
        # the hinged elastic member follows CSI, fiber members follow the physics.
        self.D = np.ones(6)
        self._loads_key = None
        self.loads: List[ElementLoad] = []
        self.R_ss = np.zeros(12)
        self.q0 = np.zeros(6)
        self.p = np.zeros(12)
        self.k = None
        self.pl = np.zeros(12)
        self.q = np.zeros(6)
        self.v = np.zeros(6)
        self.c = None  # no element damping tangent

    # ------------------------------------------------------------ geometry
    def set_geometry(self, geom: str):
        g = geom.lower().replace("-", "").replace("_", "").replace(" ", "")
        if g in ("none",):
            g = "linear"
        if g in ("largedisplacement", "pdeltalargedisplacements", "corot"):
            g = "corotational"
        if g not in ("linear", "pdelta", "corotational"):
            raise ValueError(f"Geometric transformation {geom!r} not supported for nonlinear frames "
                             "(use 'Linear', 'PDelta' or 'Corotational')")
        if g == "corotational" and self.crd is None:
            el = self.el
            if np.any(el.offset_i) or np.any(el.offset_j) or el.rigid_i or el.rigid_j:
                raise ValueError(f"frame element {el.id}: the corotational transformation does not support joint "
                                 "offsets or rigid end zones")
            from .corotational import CorotTransform3D
            self.crd = CorotTransform3D(self.fe.xi, self.fe.xj, self.fe.kin.axes)
        self.geom = g

    def commit_geometry(self):
        if self.geom == "corotational" and self.crd is not None:
            self.crd.commit()

    def reset_geometry(self):
        if self.crd is not None:
            self.crd.reset()

    # --------------------------------------------------------------- loads
    def set_loads(self, loads: Sequence[ElementLoad]):
        key = tuple((ld.kind, ld.load_type, ld.x_a, ld.x_b, tuple(np.round(ld.value_a, 12)),
                     tuple(np.round(ld.value_b, 12))) for ld in loads)
        if key == self._loads_key:
            return False
        self._loads_key = key
        el = self.el
        self.loads = _face_loads(loads, self.fe.kin.L, el.rigid_i, el.rigid_j)
        self.R_ss = ss_reactions(self.loads, self.L) if self.loads else np.zeros(12)
        self._loads_changed()
        return True

    def _loads_changed(self):
        pass

    def _clamped_basic_forces(self) -> np.ndarray:
        """Basic forces of the clamped elastic member under the span loads (q0)."""
        if not self.loads:
            return np.zeros(6)
        el = self.el
        face = dataclasses.replace(el, offset_i=np.zeros(3), offset_j=np.zeros(3), rigid_i=0.0, rigid_j=0.0,
                                   releases_i=np.zeros(6, bool), releases_j=np.zeros(6, bool), loads=[],
                                   axes=np.eye(3), foundation=np.zeros(3), bending_flex=None,
                                   tension_limit=None, compression_limit=None)
        ff = FrameFormulation(face, np.zeros(3), np.array([self.L, 0.0, 0.0]))
        fef = ff.fixed_end_load_vector(self.loads)  # load applied to the element (equivalent nodal load)
        rhs = -fef - self.R_ss
        q0, *_ = np.linalg.lstsq(self.B.T, rhs, rcond=None)
        return q0

    # ------------------------------------------------------------- response
    def basic_state(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def basic_initial_stiffness(self) -> np.ndarray:
        raise NotImplementedError

    def set_trial(self, ug: np.ndarray, vg: Optional[np.ndarray] = None):
        if self.geom == "corotational":
            v, B = self.crd.update(ug)
            q, kb = self.basic_state(v)
            self.v, self.q = v, q
            S = self.crd.spin_map()
            p = S @ (B.T @ q) + self.A.T @ self.R_ss  # span loads keep their initial direction
            self.p = p
            self.k = S @ (B.T @ kb @ B + self.crd.geometric_stiffness(q))
            self.pl = self.A @ p  # end forces in the initial local axes
            return
        ul = self.A @ ug
        v = self.B @ ul
        q, kb = self.basic_state(v)
        self.v, self.q = v, q
        pl = self.B.T @ q + self.R_ss
        kl = self.B.T @ kb @ self.B
        if self.geom == "pdelta":
            P = q[0]
            g = P / self.L
            for a, b in ((1, 7), (2, 8)):
                d = ul[a] - ul[b]
                pl[a] += g * d
                pl[b] -= g * d
                kl[a, a] += g
                kl[b, b] += g
                kl[a, b] -= g
                kl[b, a] -= g
        self.pl = pl
        self.p = self.A.T @ pl
        self.k = self.A.T @ kl @ self.A

    def initial_stiffness_global(self) -> np.ndarray:
        kb = self.basic_initial_stiffness()
        return self.A.T @ (self.B.T @ kb @ self.B) @ self.A

    def end_forces_local(self) -> np.ndarray:
        """Forces the joints apply to the member at its element ends (local, 12),
        the convention of :meth:`FrameFormulation.end_forces_local`."""
        return self.fe.kin.T_rigid.T @ self.pl

    # ----------------------------------------------------------------- state
    def commit(self):
        raise NotImplementedError

    def revert(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def response(self) -> dict:
        return {"basic_forces": self.q.copy(), "basic_deformations": self.v.copy()}


def _b_matrix(xi: float) -> np.ndarray:
    b = np.zeros((4, 6))
    b[0, 0] = 1.0
    b[1, 1], b[1, 2] = xi - 1.0, xi
    b[2, 3], b[2, 4] = xi - 1.0, xi
    b[3, 5] = 1.0
    return b


def _inv(k: np.ndarray) -> np.ndarray:
    try:
        f = np.linalg.inv(k)
        if np.all(np.isfinite(f)):
            return f
    except np.linalg.LinAlgError:
        pass
    d = np.abs(np.diag(k)).max()
    return np.linalg.inv(k + np.eye(len(k)) * 1e-10 * max(d, 1.0))


def _binv(k: np.ndarray) -> np.ndarray:
    """Batched inverse of (n, 4, 4) section stiffnesses (regularised if singular)."""
    try:
        f = np.linalg.inv(k)
        if np.all(np.isfinite(f)):
            return f
    except np.linalg.LinAlgError:
        pass
    out = np.empty_like(k)
    for i in range(len(k)):
        out[i] = _inv(k[i])
    return out


def _as_sections(sections, n: int):
    from .sections import ElementSections
    if isinstance(sections, ElementSections):
        return sections
    if isinstance(sections, Section3D):
        return ElementSections([sections] * n)
    sections = list(sections)
    if len(sections) != n:
        raise ValueError("one section per integration point is required")
    return ElementSections(sections)


# ============================================================================
class ForceBeamColumn(NLFrame):
    """Force-based beam-column (OpenSees ``forceBeamColumn``).

    ``sections``: one section for every integration point (points sharing the
    same section object are evaluated as one vector block) or a single section."""

    kind = "ForceBeamColumn"

    def __init__(self, fe: FrameFormulation, sections, integration: BeamIntegration,
                 geom: str = "Linear", max_iters: int = 10, tol: float = 1e-12, max_subdivisions: int = 10):
        super().__init__(fe, geom)
        self.secs = _as_sections(sections, integration.n)
        self.bi = integration
        self.max_iters, self.tol, self.max_sub = max_iters, tol, max_subdivisions
        self.b = np.array([_b_matrix(x) for x in integration.xi])  # (n, 4, 6)
        self.wL = integration.w * self.L
        self._fs0 = _binv(self.secs.initial_stiffness())
        self.K0 = _inv(self._flex(self._fs0))
        n = integration.n
        self.sp = np.zeros((n, 4))
        self._sp_used = np.zeros((n, 4))
        self._init_state()

    def _flex(self, fs):
        return np.einsum("n,nki,nkl,nlj->ij", self.wL, self.b, fs, self.b)

    def _init_state(self):
        n = self.bi.n
        self.T = dict(Se=np.zeros(6), kv=self.K0.copy(), e=np.zeros((n, 4)), Sr=np.zeros((n, 4)),
                      fs=self._fs0.copy(), v=np.zeros(6), sp=np.zeros((n, 4)))
        self.C = {k: v.copy() for k, v in self.T.items()}

    def _loads_changed(self):
        self.sp = np.array([ss_section_forces(self.loads, self.L, self.R_ss, x * self.L) for x in self.bi.xi])

    def basic_initial_stiffness(self):
        return self.K0.copy()

    def basic_state(self, v):
        T = self.T
        b, wL = self.b, self.wL
        dv = v - T["v"]
        vin = T["v"].copy()
        dv_todo = dv.copy()
        dv_trial = dv.copy()
        n_sub = 1
        converged = np.linalg.norm(dv) <= EPS and np.array_equal(self.sp, T["sp"])
        while not converged and n_sub <= self.max_sub:
            done = False
            for l in range(3):
                eT = T["e"].copy()
                SrT = T["Sr"].copy()
                fsT = T["fs"].copy()
                kvT = T["kv"] if l == 0 else self.K0
                SeT = T["Se"] + kvT @ dv_trial
                # tangent iterations can cycle at material kinks; the initial-flexibility
                # passes converge monotonically and get many more iterations
                n_it = 4 * self.max_iters if l == 0 else 10 * self.max_iters
                omega, dW_prev = 1.0, None
                for j in range(n_it):
                    Ss = np.einsum("nij,j->ni", b, SeT) + self.sp
                    fuse = fsT if l == 0 else self._fs0
                    eT = eT + np.einsum("nij,nj->ni", fuse, Ss - SrT)
                    SrT, ks = self.secs.set_trial(eT)
                    fsT = _binv(ks)
                    dvs = np.einsum("nij,nj->ni", fsT, Ss - SrT)
                    kvT = _inv(self._flex(fsT))
                    vr = np.einsum("n,nki,nk->i", wL, b, eT + dvs)
                    dvr = vin + dv_trial - vr
                    dSe = kvT @ dvr
                    dW = float(dvr @ dSe)
                    # under-relax when the energy norm stops decreasing (cycling at kinks)
                    if dW_prev is not None and abs(dW) > 0.9 * abs(dW_prev):
                        omega = max(0.5 * omega, 0.05)
                    elif omega < 1.0:
                        omega = min(1.0, 1.5 * omega)
                    dW_prev = dW
                    SeT = SeT + omega * dSe
                    scale = 1e-15 * (np.abs(SeT) @ np.abs(vin + dv_trial) + 1.0)
                    if abs(dW) < max(self.tol, scale):
                        done = True
                        break
                if done:
                    break
            if done:
                dv_todo = dv_todo - dv_trial
                vin = vin + dv_trial
                T["e"], T["Sr"], T["fs"], T["kv"], T["Se"] = eT, SrT, fsT, kvT, SeT
                T["v"] = vin.copy()
                T["sp"] = self.sp.copy()
                if np.linalg.norm(dv_todo) <= EPS * max(1.0, np.linalg.norm(v)):
                    converged = True
                else:
                    dv_trial = dv_todo.copy()
                    n_sub = 1
            else:
                dv_trial = dv_trial / 10.0
                n_sub += 1
        if not converged:
            raise ElementNonConvergence(f"force-based element {self.id}: state determination did not converge")
        T["v"] = v.copy()
        return T["Se"].copy(), T["kv"].copy()

    def commit(self):
        self.secs.commit()
        self.C = {k: v.copy() for k, v in self.T.items()}
        for h in getattr(self, "fiber_hinges", None) or []:
            h["max"] = max(h["max"], self._fiber_hinge_state(h)["deformation"])

    def _fiber_hinge_state(self, h):
        k = 0 if h["end"] == "I" else self.bi.n - 1
        e = self.T["e"][k]
        sr = self.T["Sr"][k]
        lp = h["lp"]
        thz, thy = e[1] * lp, e[2] * lp
        pz = (e[1] - sr[1] / self.EIz) * lp if self.EIz else 0.0
        py = (e[2] - sr[2] / self.EIy) * lp if self.EIy else 0.0
        return {"deformation": float(np.hypot(thz, thy)), "plastic": float(np.hypot(pz, py)),
                "M3": float(sr[1]), "M2": float(-sr[2]), "P": float(sr[0]), "R3": float(thz), "R2": float(-thy)}

    def hinge_results(self):
        """Fiber P-M2-M3 hinges: rotation = curvature x hinge length at the end section."""
        out = []
        for h in getattr(self, "fiber_hinges", None) or []:
            st = self._fiber_hinge_state(h)
            perf = ""
            if h["acceptance"]:
                io, ls, cp = h["acceptance"]
                pl = st["plastic"]
                perf = "<IO" if pl <= io else "IO to LS" if pl <= ls else "LS to CP" if pl <= cp else ">CP"
            out.append({"element": self.id, "dof": "PMM", "end": h["end"], "name": h["name"],
                        "deformation": st["deformation"], "force": float(np.hypot(st["M3"], st["M2"])),
                        "plastic_deformation": st["plastic"], "state": "B to C" if st["plastic"] > 1e-5 else "A to B",
                        "performance": perf, "max_deformation": max(h["max"], st["deformation"]), "min_deformation": 0.0,
                        "P": st["P"], "M2": st["M2"], "M3": st["M3"], "R2": st["R2"], "R3": st["R3"]})
        return out

    def revert(self):
        self.secs.revert()
        self.T = {k: v.copy() for k, v in self.C.items()}

    def reset(self):
        self.secs.reset()
        self._init_state()

    def response(self):
        out = super().response()
        out["section_forces"] = self.T["Sr"].copy()
        out["section_deformations"] = self.T["e"].copy()
        out["xi"] = self.bi.xi.copy()
        return out


# ============================================================================
class DispBeamColumn(NLFrame):
    """Displacement-based beam-column (OpenSees ``dispBeamColumn``)."""

    kind = "DispBeamColumn"

    def __init__(self, fe: FrameFormulation, sections, integration: BeamIntegration, geom: str = "Linear"):
        super().__init__(fe, geom)
        self.secs = _as_sections(sections, integration.n)
        self.bi = integration
        L = self.L
        Bs = []
        for x in integration.xi:
            Bm = np.zeros((4, 6))
            Bm[0, 0] = 1.0 / L
            Bm[1, 1], Bm[1, 2] = (6 * x - 4) / L, (6 * x - 2) / L
            Bm[2, 3], Bm[2, 4] = (6 * x - 4) / L, (6 * x - 2) / L
            Bm[3, 5] = 1.0 / L
            Bs.append(Bm)
        self.Bs = np.array(Bs)
        self.wL = integration.w * L

    def _loads_changed(self):
        self.q0 = self._clamped_basic_forces()

    def basic_initial_stiffness(self):
        return np.einsum("n,nki,nkl,nlj->ij", self.wL, self.Bs, self.secs.initial_stiffness(), self.Bs)

    def basic_state(self, v):
        e = np.einsum("nij,j->ni", self.Bs, v)
        s, ks = self.secs.set_trial(e)
        q = np.einsum("n,nki,nk->i", self.wL, self.Bs, s)
        kb = np.einsum("n,nki,nkl,nlj->ij", self.wL, self.Bs, ks, self.Bs)
        return q + self.q0, kb

    def commit(self):
        self.secs.commit()

    def revert(self):
        self.secs.revert()

    def reset(self):
        self.secs.reset()

    def response(self):
        out = super().response()
        e, s = self.secs.state()
        out["section_forces"], out["section_deformations"], out["xi"] = s, e, self.bi.xi.copy()
        return out


# ============================================================================
class FrameHinge:
    """A concentrated hinge in an :class:`ElasticHingeFrame`.

    ``dof``: "M3", "M2" or "P"; ``end``: "I" or "J".  The material works in the
    CSI sign convention of the member force at that end (moment M3/M2 and its
    rotation, or axial force P / elongation).  ``acceptance``: optional (IO, LS,
    CP) plastic deformations; ``yield_deformation``: deformation at point B of
    the backbone (to report plastic deformations)."""

    _INDEX = {("M3", "I"): (1, -1.0), ("M3", "J"): (2, 1.0), ("M2", "I"): (3, 1.0), ("M2", "J"): (4, -1.0),
              ("P", "I"): (0, 1.0), ("P", "J"): (0, 1.0)}

    def __init__(self, dof: str, end: str, material: UniaxialMaterial, name: str = "",
                 acceptance: Optional[Tuple[float, float, float]] = None, yield_deformation: float = 0.0,
                 backbone_points: Optional[Sequence[Tuple[float, float]]] = None):
        self.dof, self.end = dof.upper(), end.upper()
        if (self.dof, self.end) not in self._INDEX:
            raise ValueError(f"Unsupported hinge {dof}/{end}")
        self.index, self.sign = self._INDEX[(self.dof, self.end)]
        self.material = material
        self.name = name
        self.acceptance = acceptance
        self.yield_deformation = yield_deformation
        self.backbone_points = list(backbone_points) if backbone_points else None
        self.max_def = 0.0
        self.min_def = 0.0
        self._c_ext = (0.0, 0.0)

    def state_label(self, d: float) -> str:
        """CSI hinge state (A-B, B-C, C-D, D-E, >E) from the backbone points."""
        if not self.backbone_points:
            return ""
        pts = self.backbone_points  # positive side (B, C, D, E) deformations
        a = abs(d)
        labels = ["A to B", "B to C", "C to D", "D to E"]
        for lab, (dd, _) in zip(labels, pts):
            if a <= dd:
                return lab
        return ">E"

    def performance(self, d: float) -> str:
        if not self.acceptance:
            return ""
        pl = max(0.0, abs(d) - self.yield_deformation)
        io, ls, cp = self.acceptance
        if pl <= io:
            return "<IO"
        if pl <= ls:
            return "IO to LS"
        if pl <= cp:
            return "LS to CP"
        return ">CP"


class ElasticHingeFrame(NLFrame):
    """Elastic (Timoshenko) member with hinges in series at the ends of its flexible span."""

    kind = "ElasticHingeFrame"
    uses_flexible_length_axial = False  # axial / torsion stiffness already over the full length

    def __init__(self, fe: FrameFormulation, hinges: Sequence[FrameHinge], geom: str = "Linear",
                 max_iters: int = 50, tol: float = 1e-12):
        super().__init__(fe, geom)
        el = fe.el
        L, Lfull = self.L, fe.kin.L
        kel = np.zeros((6, 6))
        kel[0, 0] = el.E * el.A / Lfull
        kel[5, 5] = el.G * el.J / Lfull
        for (a, b), EI, As in (((1, 2), el.E * el.I33, el.As2), ((3, 4), el.E * el.I22, el.As3)):
            phi = 0.0 if As <= 0 or EI <= 0 else 12.0 * EI / (el.G * As * L * L)
            c = EI / ((1.0 + phi) * L)
            kel[a, a] = kel[b, b] = c * (4.0 + phi)
            kel[a, b] = kel[b, a] = c * (2.0 - phi)
        self.kel = kel
        # one hinge per basic DOF (an axial hinge at either end acts on the elongation)
        seen = {}
        for h in hinges:
            if h.index in seen:
                raise ValueError(f"frame element {self.id}: two hinges act on the same DOF ({h.dof} {h.end})")
            seen[h.index] = h
        self.hinges = list(seen.values())
        self.idx = np.array([h.index for h in self.hinges], dtype=int)
        self.sgn = np.array([h.sign for h in self.hinges])
        self.max_iters, self.tol = max_iters, tol
        m = len(self.hinges)
        self.Th = np.zeros(m)  # trial hinge deformations (basic sign)
        self.Ch = np.zeros(m)
        self.Cv = np.zeros(6)
        self._hk = np.zeros(m)
        self._hf = np.zeros(m)

    def _loads_changed(self):
        self.q0 = self._clamped_basic_forces()

    def basic_initial_stiffness(self):
        if not len(self.hinges):
            return self.kel.copy()
        kh = np.array([float(h.material.initial_tangent[0]) for h in self.hinges])
        return self._condensed(kh)

    def _condensed(self, kh):
        K = self.kel
        i = self.idx
        J = np.diag(kh) + K[np.ix_(i, i)]
        X = np.linalg.solve(J, K[i, :])
        return K - K[:, i] @ X

    def _hinge_eval(self, h):
        f = np.zeros(len(h))
        k = np.zeros(len(h))
        for n, (hg, s) in enumerate(zip(self.hinges, self.sgn)):
            hg.material.set_trial(np.array([s * h[n]]))
            f[n] = s * float(hg.material.stress[0])
            k[n] = float(hg.material.tangent[0])
        return f, k

    def _residual(self, v, h):
        K, i = self.kel, self.idx
        f, kh = self._hinge_eval(h)
        vel = v.copy()
        vel[i] -= h
        qel = K @ vel + self.q0
        return f - qel[i], f, kh, qel

    def _solve_hinges(self, v, h0):
        """Hinge deformations h with M_hinge(h) = q(v - h).

        1. plain Newton (fast path);
        2. if it cycles (piecewise linear laws with stiffness jumps of several
           orders of magnitude: rigid-plastic hinges, Takeda reloading), every
           hinge equation is solved exactly by a bracketed scalar root finder,
           Gauss-Seidel over the hinges of the member.  Each equation is monotone
           in its own deformation because the hinge drops are regularised to at
           most half the member stiffness, so the bracket always exists."""
        K, i = self.kel, self.idx
        h = h0.copy()
        for _ in range(15):
            g, f, kh, qel = self._residual(v, h)
            scale = np.abs(qel[i]).max() + np.abs(f).max() + 1.0
            if np.abs(g).max() <= 1e-11 * scale:
                return h, f, kh
            J = np.diag(kh) + K[np.ix_(i, i)]
            try:
                h = h - np.linalg.solve(J, g)
            except np.linalg.LinAlgError:
                break
        h = h0.copy()
        for sweep in range(200):
            h_old = h.copy()
            for k in range(len(h)):
                h[k] = self._scalar_root(v, h, k)
            g, f, kh, qel = self._residual(v, h)
            scale = np.abs(qel[i]).max() + np.abs(f).max() + 1.0
            if np.abs(g).max() <= 1e-10 * scale:
                # Newton polish (quadratic once the branches are settled)
                for _ in range(3):
                    J = np.diag(kh) + K[np.ix_(i, i)]
                    hn = h - np.linalg.solve(J, g)
                    gn, fn, khn, qn = self._residual(v, hn)
                    if np.abs(gn).max() < np.abs(g).max():
                        h, g, f, kh, qel = hn, gn, fn, khn, qn
                    else:
                        break
                return h, f, kh
            if np.abs(h - h_old).max() <= 1e-15 * (np.abs(h).max() + 1e-12):
                break
        g, f, kh, qel = self._residual(v, h)
        scale = np.abs(qel[i]).max() + np.abs(f).max() + 1.0
        if np.abs(g).max() <= 1e-8 * scale:
            return h, f, kh
        return None

    def _scalar_root(self, v, h, k):
        """Solve g_k(h_k) = 0 with the other hinge deformations fixed (Illinois method
        on an expanding bracket)."""
        def gk(x):
            hh = h.copy()
            hh[k] = x
            return self._residual(v, hh)[0][k]

        x0 = h[k]
        g0 = gk(x0)
        if g0 == 0.0:
            return x0
        Kkk = self.kel[self.idx[k], self.idx[k]]
        step = abs(g0) / Kkk + 1e-12
        # g is increasing in x: move against the sign of g to bracket the root
        a, ga = x0, g0
        b = x0 - np.sign(g0) * step
        gb = gk(b)
        n = 0
        while np.sign(gb) == np.sign(ga) and n < 60:
            a, ga = b, gb
            step *= 2.0
            b = a - np.sign(ga) * step
            gb = gk(b)
            n += 1
        if np.sign(gb) == np.sign(ga):
            return b
        side = 0
        for _ in range(100):
            c = b - gb * (b - a) / (gb - ga)
            gc = gk(c)
            if gc == 0.0 or abs(b - a) <= 1e-15 * (abs(c) + 1e-12):
                return c
            if np.sign(gc) == np.sign(gb):
                b, gb = c, gc
                if side == -1:
                    ga *= 0.5
                side = -1
            else:
                a, ga = b, gb
                b, gb = c, gc
                side = 1
        return c

    def basic_state(self, v):
        if not len(self.hinges):
            return self.kel @ v + self.q0, self.kel.copy()
        sol = self._solve_hinges(v, self.Th)
        if sol is None:
            # march from the committed state in sub-increments
            h = self.Ch.copy()
            for t in np.linspace(0.0, 1.0, 21)[1:]:
                sol = self._solve_hinges(self.Cv + t * (v - self.Cv), h)
                if sol is None:
                    raise ElementNonConvergence(f"hinged frame element {self.id}: hinge state did not converge")
                h = sol[0]
        h, f, kh = sol
        self.Th = h
        self._hf, self._hk = f, kh
        vel = v.copy()
        vel[self.idx] -= h
        q = self.kel @ vel + self.q0
        return q, self._condensed(kh)

    def commit(self):
        for hg, s, h in zip(self.hinges, self.sgn, self.Th):
            hg.material.commit()
            d = s * h
            hg.max_def = max(hg.max_def, d)
            hg.min_def = min(hg.min_def, d)
        self.Ch = self.Th.copy()
        self.Cv = self.v.copy()
        self._commit_ext = [(hg.max_def, hg.min_def) for hg in self.hinges]

    def revert(self):
        for hg in self.hinges:
            hg.material.revert()
        self.Th = self.Ch.copy()

    def reset(self):
        for hg in self.hinges:
            hg.material.reset()
            hg.max_def = hg.min_def = 0.0
        self.Th = np.zeros(len(self.hinges))
        self.Ch = np.zeros(len(self.hinges))
        self.Cv = np.zeros(6)

    def hinge_results(self) -> List[dict]:
        out = []
        for n, (hg, s) in enumerate(zip(self.hinges, self.sgn)):
            d = float(s * self.Th[n])
            F = float(hg.material.stress[0]) if hg.material.n else 0.0
            out.append({"element": self.id, "dof": hg.dof, "end": hg.end, "name": hg.name, "deformation": d,
                        "force": F, "plastic_deformation": max(0.0, abs(d) - hg.yield_deformation) * np.sign(d),
                        "state": hg.state_label(d), "performance": hg.performance(d),
                        "max_deformation": hg.max_def, "min_deformation": hg.min_def})
        return out

    def response(self):
        out = super().response()
        out["hinges"] = self.hinge_results()
        return out


# ============================================================================
class Link:
    """Zero-length link (CSI link with zero length / OpenSees zeroLength).

    ``axes``: rows = link local 1, 2, 3 in global.  ``materials``: dict DOF
    index (0..5 = U1 U2 U3 R1 R2 R3) -> uniaxial material.  Deformation =
    local motion of joint J relative to joint I (or of the single joint of a
    grounded link)."""

    kind = "Link"

    def __init__(self, link_id: int, node_i: int, node_j: Optional[int], axes: np.ndarray,
                 materials: Dict[int, UniaxialMaterial], name: str = ""):
        self.id = link_id
        self.name = name
        self.nodes = (node_i,) if node_j is None else (node_i, node_j)
        R = np.asarray(axes, float)
        T = np.zeros((6, 6))
        T[:3, :3] = R
        T[3:, 3:] = R
        if node_j is None:
            self.Tg = T  # d = T u_i
        else:
            self.Tg = np.hstack([-T, T])  # d = T (u_j - u_i)
        self.mats = dict(materials)
        self.dofs_used = np.array(sorted(self.mats), dtype=int)
        self.rate_dependent = any(m.rate_dependent for m in self.mats.values())
        self.p = np.zeros(self.Tg.shape[1])
        self.k = np.zeros((self.Tg.shape[1],) * 2)
        self.c = None
        self.d = np.zeros(6)
        self.f = np.zeros(6)

    def set_loads(self, loads):
        return False

    def set_trial(self, ug, vg=None):
        d = self.Tg @ ug
        r = self.Tg @ vg if vg is not None else np.zeros(6)
        f = np.zeros(6)
        kd = np.zeros(6)
        cd = np.zeros(6)
        for i, m in self.mats.items():
            m.set_trial(np.array([d[i]]), np.array([r[i]]))
            f[i] = float(m.stress[0])
            kd[i] = float(m.tangent[0])
            cd[i] = float(m.damping_tangent[0])
        self.d, self.f = d, f
        self.p = self.Tg.T @ f
        self.k = self.Tg.T @ np.diag(kd) @ self.Tg
        self.c = self.Tg.T @ np.diag(cd) @ self.Tg if self.rate_dependent else None

    def initial_stiffness_global(self):
        kd = np.zeros(6)
        for i, m in self.mats.items():
            kd[i] = float(m.initial_tangent[0])
        return self.Tg.T @ np.diag(kd) @ self.Tg

    def commit(self):
        for m in self.mats.values():
            m.commit()

    def revert(self):
        for m in self.mats.values():
            m.revert()

    def reset(self):
        for m in self.mats.values():
            m.reset()

    def response(self):
        return {"deformation": self.d.copy(), "force": self.f.copy()}
