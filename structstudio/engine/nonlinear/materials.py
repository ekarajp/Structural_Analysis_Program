"""Uniaxial materials for nonlinear analysis.

Every material is a *group*: it holds the state of ``n`` identical material
points (the fibers of a section that share one material, or a single hinge).
Strains, stresses and tangents are numpy arrays of length ``n``, so a fiber
section with hundreds of fibers is evaluated with a handful of vector
operations instead of a Python loop per fiber.

State handling follows OpenSees: ``set_trial`` computes a trial state from
the last *committed* state; ``commit`` accepts it, ``revert`` discards it.

The algorithms of ``Steel01``, ``Steel02``, ``Concrete01``, ``Concrete02``,
``ElasticPP``, ``Hysteretic`` and ``BoucWen`` are re-implemented from the
published material models and checked strain-history by strain-history
against OpenSees 3.8 (``tests/xcheck/test_nonlinear_xcheck.py``):

* Steel01  - bilinear kinematic hardening with optional isotropic hardening
  (Filippou et al.).
* Steel02  - Giuffre-Menegotto-Pinto with isotropic hardening (Filippou,
  Popov & Bertero 1983, EERC 83/19).
* Concrete01 - Kent-Scott-Park envelope, Karsan-Jirsa degraded linear
  unloading/reloading, no tension.
* Concrete02 - Kent-Park envelope with linear tension softening (Mohd Yassin
  1994).
* Hysteretic - tri-linear pinching material with damage (OpenSees).
* BoucWen  - smooth hysteresis with degradation (Haukaas & Der Kiureghian).

Hinge laws used by the frame hinges (``HingeMaterial``) are the CSI-style
backbone A-B-C-D-E with kinematic or Takeda (peak-oriented) hysteresis.
Strain and stress are generic: for a hinge they are rotation and moment.
"""
from __future__ import annotations

import copy as _copy
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS = np.finfo(float).eps


def _arr(x, n):
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return np.full(n, float(a))
    return a.astype(float).copy()


class UniaxialMaterial:
    """Base class: a group of ``n`` identical material points."""

    type_name = "Uniaxial"
    rate_dependent = False

    def __init__(self, n: int = 1):
        self.n = int(n)
        self.strain = np.zeros(self.n)
        self.stress = np.zeros(self.n)
        self.tangent = np.zeros(self.n)
        self.damping_tangent = np.zeros(self.n)  # d stress / d strain-rate (rate dependent materials)

    # --- parameters -------------------------------------------------------
    def params(self) -> dict:
        raise NotImplementedError

    def clone(self, n: Optional[int] = None) -> "UniaxialMaterial":
        """Fresh (virgin) group with the same parameters."""
        return type(self)(**self.params(), n=self.n if n is None else n)

    @property
    def initial_tangent(self) -> np.ndarray:
        raise NotImplementedError

    # --- state --------------------------------------------------------------
    def set_trial(self, strain, rate=None) -> None:
        raise NotImplementedError

    def commit(self) -> None:
        raise NotImplementedError

    def revert(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        fresh = self.clone()
        self.__dict__.update(fresh.__dict__)

    def copy(self) -> "UniaxialMaterial":
        return _copy.deepcopy(self)

    # convenience for scalar use
    def response(self, strain: float, rate: float = 0.0) -> Tuple[float, float]:
        self.set_trial(np.full(self.n, strain), np.full(self.n, rate))
        return float(self.stress[0]), float(self.tangent[0])


# ============================================================================
class Elastic(UniaxialMaterial):
    type_name = "Elastic"

    def __init__(self, E: float, eta: float = 0.0, Eneg: Optional[float] = None, n: int = 1):
        super().__init__(n)
        self.E = float(E)
        self.eta = float(eta)
        self.Eneg = float(E if Eneg is None else Eneg)
        self.rate_dependent = eta != 0.0
        self.tangent[:] = self.E
        self._c = (self.strain.copy(), np.zeros(self.n))

    def params(self):
        return {"E": self.E, "eta": self.eta, "Eneg": self.Eneg}

    @property
    def initial_tangent(self):
        return np.full(self.n, self.E)

    def set_trial(self, strain, rate=None):
        e = _arr(strain, self.n)
        r = np.zeros(self.n) if rate is None else _arr(rate, self.n)
        self.strain = e
        k = np.where(e >= 0.0, self.E, self.Eneg)
        self.stress = k * e + self.eta * r
        self.tangent = k
        self.damping_tangent = np.full(self.n, self.eta)

    def commit(self):
        self._c = (self.strain.copy(), self.stress.copy())

    def revert(self):
        self.strain = self._c[0].copy()
        self.set_trial(self.strain)


# ============================================================================
class ElasticPP(UniaxialMaterial):
    """Elastic-perfectly plastic (OpenSees ElasticPP): E, eps_yp, eps_yn, eps0."""

    type_name = "ElasticPP"

    def __init__(self, E: float, epsyP: float, epsyN: Optional[float] = None, eps0: float = 0.0, n: int = 1):
        super().__init__(n)
        self.E = float(E)
        self.epsyP = float(epsyP)
        self.epsyN = float(-epsyP if epsyN is None else epsyN)
        self.eps0 = float(eps0)
        self.fyp = self.E * self.epsyP
        self.fyn = self.E * self.epsyN
        self.ep = np.zeros(n)  # committed plastic strain
        self.tangent[:] = self.E
        self._cs = np.zeros(n)

    def params(self):
        return {"E": self.E, "epsyP": self.epsyP, "epsyN": self.epsyN, "eps0": self.eps0}

    @property
    def initial_tangent(self):
        return np.full(self.n, self.E)

    def set_trial(self, strain, rate=None):
        e = _arr(strain, self.n)
        self.strain = e
        sig = self.E * (e - self.eps0 - self.ep)
        f = np.where(sig >= 0.0, sig - self.fyp, -sig + self.fyn)
        elastic = f <= -self.E * EPS
        self.stress = np.where(elastic, sig, np.where(sig > 0.0, self.fyp, self.fyn))
        self.tangent = np.where(elastic, self.E, 0.0)

    def commit(self):
        sig = self.E * (self.strain - self.eps0 - self.ep)
        self.ep = np.where(sig > self.fyp, self.ep + (sig - self.fyp) / self.E,
                           np.where(sig < self.fyn, self.ep + (sig - self.fyn) / self.E, self.ep))
        self._cs = self.strain.copy()

    def revert(self):
        self.set_trial(self._cs)


# ============================================================================
class Steel01(UniaxialMaterial):
    """Bilinear steel with kinematic (and optional isotropic) hardening."""

    type_name = "Steel01"

    def __init__(self, Fy: float, E0: float, b: float, a1: float = 0.0, a2: float = 1.0, a3: float = 0.0,
                 a4: float = 1.0, n: int = 1):
        super().__init__(n)
        self.Fy, self.E0, self.b = float(Fy), float(E0), float(b)
        self.a1, self.a2, self.a3, self.a4 = float(a1), float(a2), float(a3), float(a4)
        z = np.zeros(n)
        # committed
        self.CminStrain, self.CmaxStrain = z.copy(), z.copy()
        self.CshiftP, self.CshiftN = np.ones(n), np.ones(n)
        self.Cloading = np.zeros(n, dtype=int)
        self.Cstrain, self.Cstress, self.Ctangent = z.copy(), z.copy(), np.full(n, self.E0)
        self._trial_from_commit()

    def params(self):
        return {"Fy": self.Fy, "E0": self.E0, "b": self.b, "a1": self.a1, "a2": self.a2, "a3": self.a3, "a4": self.a4}

    @property
    def initial_tangent(self):
        return np.full(self.n, self.E0)

    def _trial_from_commit(self):
        self.TminStrain, self.TmaxStrain = self.CminStrain.copy(), self.CmaxStrain.copy()
        self.TshiftP, self.TshiftN = self.CshiftP.copy(), self.CshiftN.copy()
        self.Tloading = self.Cloading.copy()
        self.strain, self.stress, self.tangent = self.Cstrain.copy(), self.Cstress.copy(), self.Ctangent.copy()

    def set_trial(self, strain, rate=None):
        self._trial_from_commit()
        e = _arr(strain, self.n)
        d = e - self.Cstrain
        self.strain = e
        act = np.abs(d) > EPS
        if not act.any():
            return
        fy1b = self.Fy * (1.0 - self.b)
        Esh = self.b * self.E0
        c1 = Esh * e
        c2 = self.TshiftN * fy1b
        c3 = self.TshiftP * fy1b
        c = self.Cstress + self.E0 * d
        s = np.minimum(c1 + c3, c)
        s = np.maximum(c1 - c2, s)
        k = np.where(np.abs(s - c) < EPS, self.E0, Esh)
        self.stress = np.where(act, s, self.stress)
        self.tangent = np.where(act, k, self.tangent)
        # load reversal
        epsy = self.Fy / self.E0
        L = self.Tloading
        init = act & (L == 0)
        L = np.where(init & (d > 0), 1, np.where(init & (d < 0), -1, L))
        to_unload = act & (L == 1) & (d < 0.0)
        tmax = np.where(to_unload & (self.Cstrain > self.TmaxStrain), self.Cstrain, self.TmaxStrain)
        shN = np.where(to_unload, 1.0 + self.a1 * np.power(np.abs(tmax - self.TminStrain) / (2.0 * self.a2 * epsy), 0.8),
                       self.TshiftN)
        L = np.where(to_unload, -1, L)
        to_load = act & (L == -1) & (d > 0.0)
        tmin = np.where(to_load & (self.Cstrain < self.TminStrain), self.Cstrain, self.TminStrain)
        shP = np.where(to_load, 1.0 + self.a3 * np.power(np.abs(tmax - tmin) / (2.0 * self.a4 * epsy), 0.8), self.TshiftP)
        L = np.where(to_load, 1, L)
        self.TmaxStrain, self.TminStrain, self.TshiftN, self.TshiftP, self.Tloading = tmax, tmin, shN, shP, L

    def commit(self):
        self.CminStrain, self.CmaxStrain = self.TminStrain.copy(), self.TmaxStrain.copy()
        self.CshiftP, self.CshiftN = self.TshiftP.copy(), self.TshiftN.copy()
        self.Cloading = self.Tloading.copy()
        self.Cstrain, self.Cstress, self.Ctangent = self.strain.copy(), self.stress.copy(), self.tangent.copy()

    def revert(self):
        self._trial_from_commit()


# ============================================================================
class Steel02(UniaxialMaterial):
    """Giuffre-Menegotto-Pinto steel with isotropic strain hardening."""

    type_name = "Steel02"

    def __init__(self, Fy: float, E0: float, b: float, R0: float = 15.0, cR1: float = 0.925, cR2: float = 0.15,
                 a1: float = 0.0, a2: float = 1.0, a3: float = 0.0, a4: float = 1.0, sigini: float = 0.0, n: int = 1):
        super().__init__(n)
        self.Fy, self.E0, self.b = float(Fy), float(E0), float(b)
        self.R0, self.cR1, self.cR2 = float(R0), float(cR1), float(cR2)
        self.a1, self.a2, self.a3, self.a4 = float(a1), float(a2), float(a3), float(a4)
        self.sigini = float(sigini)
        z = np.zeros(n)
        self.P = dict(epsmin=z.copy(), epsmax=z.copy(), epspl=z.copy(), epss0=z.copy(), sigs0=z.copy(),
                      epsr=z.copy(), sigr=z.copy(), kon=np.zeros(n, dtype=int), eps=z.copy(),
                      sig=z.copy(), e=np.full(n, self.E0))
        if self.sigini:
            self.P["eps"] = np.full(n, self.sigini / self.E0)
            self.P["sig"] = np.full(n, self.sigini)
        self._load_trial()

    def params(self):
        return {"Fy": self.Fy, "E0": self.E0, "b": self.b, "R0": self.R0, "cR1": self.cR1, "cR2": self.cR2,
                "a1": self.a1, "a2": self.a2, "a3": self.a3, "a4": self.a4, "sigini": self.sigini}

    @property
    def initial_tangent(self):
        return np.full(self.n, self.E0)

    def _load_trial(self):
        self.T = {k: v.copy() for k, v in self.P.items()}
        self.strain = self.T["eps"] - self.sigini / self.E0
        self.stress = self.T["sig"].copy()
        self.tangent = self.T["e"].copy()

    def set_trial(self, strain, rate=None):
        P = self.P
        T = {k: v.copy() for k, v in P.items()}
        Esh = self.b * self.E0
        epsy = self.Fy / self.E0
        eps = _arr(strain, self.n) + self.sigini / self.E0
        deps = eps - P["eps"]
        kon = T["kon"]
        first = (kon == 0) | (kon == 3)
        small = first & (np.abs(deps) < 10.0 * EPS)
        start = first & ~small
        neg = start & (deps < 0.0)
        pos = start & (deps >= 0.0)
        T["epsmax"] = np.where(start, epsy, T["epsmax"])
        T["epsmin"] = np.where(start, -epsy, T["epsmin"])
        kon = np.where(neg, 2, np.where(pos, 1, kon))
        T["epss0"] = np.where(neg, -epsy, np.where(pos, epsy, T["epss0"]))
        T["sigs0"] = np.where(neg, -self.Fy, np.where(pos, self.Fy, T["sigs0"]))
        T["epspl"] = np.where(neg, -epsy, np.where(pos, epsy, T["epspl"]))
        # reversal negative -> positive
        r1 = ~small & (kon == 2) & (deps > 0.0)
        if r1.any():
            T["epsr"] = np.where(r1, P["eps"], T["epsr"])
            T["sigr"] = np.where(r1, P["sig"], T["sigr"])
            T["epsmin"] = np.where(r1 & (P["eps"] < T["epsmin"]), P["eps"], T["epsmin"])
            d1 = (T["epsmax"] - T["epsmin"]) / (2.0 * (self.a4 * epsy))
            shft = 1.0 + self.a3 * np.power(np.abs(d1), 0.8)
            e0 = (self.Fy * shft - Esh * epsy * shft - T["sigr"] + self.E0 * T["epsr"]) / (self.E0 - Esh)
            T["epss0"] = np.where(r1, e0, T["epss0"])
            T["sigs0"] = np.where(r1, self.Fy * shft + Esh * (e0 - epsy * shft), T["sigs0"])
            T["epspl"] = np.where(r1, T["epsmax"], T["epspl"])
            kon = np.where(r1, 1, kon)
        r2 = ~small & ~r1 & (kon == 1) & (deps < 0.0)
        if r2.any():
            T["epsr"] = np.where(r2, P["eps"], T["epsr"])
            T["sigr"] = np.where(r2, P["sig"], T["sigr"])
            T["epsmax"] = np.where(r2 & (P["eps"] > T["epsmax"]), P["eps"], T["epsmax"])
            d1 = (T["epsmax"] - T["epsmin"]) / (2.0 * (self.a2 * epsy))
            shft = 1.0 + self.a1 * np.power(np.abs(d1), 0.8)
            e0 = (-self.Fy * shft + Esh * epsy * shft - T["sigr"] + self.E0 * T["epsr"]) / (self.E0 - Esh)
            T["epss0"] = np.where(r2, e0, T["epss0"])
            T["sigs0"] = np.where(r2, -self.Fy * shft + Esh * (e0 + epsy * shft), T["sigs0"])
            T["epspl"] = np.where(r2, T["epsmin"], T["epspl"])
            kon = np.where(r2, 2, kon)
        kon = np.where(small, 3, kon)
        T["kon"] = kon
        xi = np.abs((T["epspl"] - T["epss0"]) / epsy)
        R = self.R0 * (1.0 - (self.cR1 * xi) / (self.cR2 + xi))
        den = T["epss0"] - T["epsr"]
        den = np.where(den == 0.0, EPS, den)
        epsrat = (eps - T["epsr"]) / den
        dum1 = 1.0 + np.power(np.abs(epsrat), R)
        dum2 = np.power(dum1, 1.0 / R)
        sig = self.b * epsrat + (1.0 - self.b) * epsrat / dum2
        sig = sig * (T["sigs0"] - T["sigr"]) + T["sigr"]
        e = self.b + (1.0 - self.b) / (dum1 * dum2)
        e = e * (T["sigs0"] - T["sigr"]) / den
        T["sig"] = np.where(small, P["sig"] if self.sigini == 0 else self.sigini, sig)
        T["sig"] = np.where(small & (P["kon"] == 0), self.sigini, T["sig"])
        T["e"] = np.where(small, self.E0, e)
        T["eps"] = eps
        self.T = T
        self.strain = eps - self.sigini / self.E0
        self.stress = T["sig"]
        self.tangent = T["e"]

    def commit(self):
        self.P = {k: v.copy() for k, v in self.T.items()}

    def revert(self):
        self._load_trial()


# ============================================================================
class Concrete01(UniaxialMaterial):
    """Kent-Scott-Park concrete, degraded linear unloading, no tensile strength.
    fpc, epsc0, fpcu, epscu are compressive (negative) values."""

    type_name = "Concrete01"

    def __init__(self, fpc: float, epsc0: float, fpcu: float, epscu: float, n: int = 1):
        super().__init__(n)
        self.fpc, self.epsc0 = -abs(float(fpc)), -abs(float(epsc0))
        self.fpcu, self.epscu = -abs(float(fpcu)), -abs(float(epscu))
        self.Ec0 = 2.0 * self.fpc / self.epsc0
        z = np.zeros(n)
        self.CminStrain, self.CendStrain = z.copy(), z.copy()
        self.CunloadSlope = np.full(n, self.Ec0)
        self.Cstrain, self.Cstress, self.Ctangent = z.copy(), z.copy(), np.full(n, self.Ec0)
        self._from_commit()

    def params(self):
        return {"fpc": self.fpc, "epsc0": self.epsc0, "fpcu": self.fpcu, "epscu": self.epscu}

    @property
    def initial_tangent(self):
        return np.full(self.n, self.Ec0)

    def _from_commit(self):
        self.TminStrain, self.TendStrain = self.CminStrain.copy(), self.CendStrain.copy()
        self.TunloadSlope = self.CunloadSlope.copy()
        self.strain, self.stress, self.tangent = self.Cstrain.copy(), self.Cstress.copy(), self.Ctangent.copy()

    def _envelope(self, e):
        eta = e / self.epsc0
        s1 = self.fpc * (2 * eta - eta * eta)
        k1 = self.Ec0 * (1.0 - eta)
        kd = (self.fpc - self.fpcu) / (self.epsc0 - self.epscu)
        s2 = self.fpc + kd * (e - self.epsc0)
        s = np.where(e > self.epsc0, s1, np.where(e > self.epscu, s2, self.fpcu))
        k = np.where(e > self.epsc0, k1, np.where(e > self.epscu, kd, 0.0))
        return s, k

    def _unload(self, minS, s):
        t = np.where(minS < self.epscu, self.epscu, minS)
        eta = t / self.epsc0
        ratio = np.where(eta < 2.0, 0.145 * eta * eta + 0.13 * eta, 0.707 * (eta - 2.0) + 0.834)
        endS = ratio * self.epsc0
        temp1 = minS - endS
        temp2 = s / self.Ec0
        c1 = temp1 > -EPS
        c2 = ~c1 & (temp1 <= temp2)
        safe1 = np.where(temp1 == 0.0, -1.0, temp1)
        slope = np.where(c1, self.Ec0, np.where(c2, s / safe1, self.Ec0))
        endS = np.where(c1, endS, np.where(c2, minS - temp1, minS - temp2))
        return endS, slope

    def set_trial(self, strain, rate=None):
        self._from_commit()
        e = _arr(strain, self.n)
        d = e - self.Cstrain
        act = np.abs(d) >= EPS
        if not act.any():
            return
        self.strain = np.where(act, e, self.strain)
        tens = act & (e > 0.0)
        comp = act & ~tens
        temp = self.Cstress + self.CunloadSlope * (e - self.Cstrain)
        A = comp & (e <= self.Cstrain)
        R1 = A & (e <= self.CminStrain)
        R2 = A & ~R1 & (e <= self.CendStrain)
        s_env, k_env = self._envelope(e)
        endN, slopeN = self._unload(e, s_env)
        s = np.where(R1, s_env, np.where(R2, self.CunloadSlope * (e - self.CendStrain), 0.0))
        k = np.where(R1, k_env, np.where(R2, self.CunloadSlope, 0.0))
        self.TminStrain = np.where(R1, e, self.TminStrain)
        self.TendStrain = np.where(R1, endN, self.TendStrain)
        self.TunloadSlope = np.where(R1, slopeN, self.TunloadSlope)
        up = A & (temp > s)
        s = np.where(up, temp, s)
        k = np.where(up, self.TunloadSlope, k)
        B = comp & ~A & (temp <= 0.0)
        s = np.where(B, temp, s)
        k = np.where(B, self.CunloadSlope, k)
        self.stress = np.where(tens, 0.0, np.where(comp, s, self.stress))
        self.tangent = np.where(tens, 0.0, np.where(comp, k, self.tangent))

    def commit(self):
        self.CminStrain, self.CendStrain = self.TminStrain.copy(), self.TendStrain.copy()
        self.CunloadSlope = self.TunloadSlope.copy()
        self.Cstrain, self.Cstress, self.Ctangent = self.strain.copy(), self.stress.copy(), self.tangent.copy()

    def revert(self):
        self._from_commit()


# ============================================================================
class Concrete02(UniaxialMaterial):
    """Concrete with linear tension softening (OpenSees Concrete02).
    fpc, epsc0, fpcu, epscu negative; ft >= 0, Ets > 0; lam = unloading ratio."""

    type_name = "Concrete02"

    def __init__(self, fpc: float, epsc0: float, fpcu: float, epscu: float, lam: float = 0.1, ft: float = 0.0,
                 Ets: float = 0.0, n: int = 1):
        super().__init__(n)
        self.fc, self.epsc0 = -abs(float(fpc)), -abs(float(epsc0))
        self.fcu, self.epscu = -abs(float(fpcu)), -abs(float(epscu))
        self.rat, self.ft = float(lam), float(ft)
        self.Ec0 = 2.0 * self.fc / self.epsc0
        self.Ets = float(Ets) if Ets else 0.1 * self.Ec0
        z = np.zeros(n)
        self.P = dict(ecmin=z.copy(), dept=z.copy(), eps=z.copy(), sig=z.copy(), e=np.full(n, self.Ec0))
        self._load()
        # Mohd Yassin reloading focus point R; inside the compression range the unloading
        # rule becomes discontinuous (stress jumps to zero for small unloading strains)
        self.epsr = (self.fcu - self.rat * self.Ec0 * self.epscu) / (self.Ec0 * (1.0 - self.rat))
        self.warning = None
        if self.epsr < 0.0:
            lam_min = self.fcu / (self.Ec0 * self.epscu)
            self.warning = (f"Concrete02: the unloading focus point lies in compression (eps_r = {self.epsr:.3g}); "
                            f"use lambda >= {lam_min:.3g} for a continuous unloading branch")

    def params(self):
        return {"fpc": self.fc, "epsc0": self.epsc0, "fpcu": self.fcu, "epscu": self.epscu, "lam": self.rat,
                "ft": self.ft, "Ets": self.Ets}

    @property
    def initial_tangent(self):
        return np.full(self.n, self.Ec0)

    def _load(self):
        self.T = {k: v.copy() for k, v in self.P.items()}
        self.strain, self.stress, self.tangent = self.T["eps"].copy(), self.T["sig"].copy(), self.T["e"].copy()

    def _compr(self, e):
        r = e / self.epsc0
        s1 = self.fc * r * (2.0 - r)
        k1 = self.Ec0 * (1.0 - r)
        kd = (self.fcu - self.fc) / (self.epscu - self.epsc0)
        s2 = (self.fcu - self.fc) * (e - self.epsc0) / (self.epscu - self.epsc0) + self.fc
        s = np.where(e >= self.epsc0, s1, np.where(e > self.epscu, s2, self.fcu))
        k = np.where(e >= self.epsc0, k1, np.where(e > self.epscu, kd, 1.0e-10))
        return s, k

    def _tens(self, e):
        Ec0 = self.Ec0
        eps0 = self.ft / Ec0
        epsu = self.ft * (1.0 / self.Ets + 1.0 / Ec0)
        s = np.where(e <= eps0, e * Ec0, np.where(e <= epsu, self.ft - self.Ets * (e - eps0), 0.0))
        k = np.where(e <= eps0, Ec0, np.where(e <= epsu, -self.Ets, 1.0e-10))
        return s, k

    def set_trial(self, strain, rate=None):
        P = self.P
        ec0 = self.Ec0
        eps = _arr(strain, self.n)
        deps = eps - P["eps"]
        same = np.abs(deps) < EPS  # OpenSees: no strain change -> committed state
        ecmin = P["ecmin"].copy()
        dept = P["dept"].copy()
        sig = np.zeros(self.n)
        e = np.zeros(self.n)
        env = eps < ecmin
        s_env, k_env = self._compr(eps)
        sig = np.where(env, s_env, sig)
        e = np.where(env, k_env, e)
        epsr = (self.fcu - self.rat * ec0 * self.epscu) / (ec0 * (1.0 - self.rat))
        sigmr = ec0 * epsr
        sigmm, _ = self._compr(ecmin)
        den = ecmin - epsr
        den = np.where(den == 0.0, -EPS, den)
        er = (sigmm - sigmr) / den
        ept = ecmin - sigmm / er
        ur = ~env & (eps <= ept)
        sigmin = sigmm + er * (eps - ecmin)
        sigmax = er * 0.5 * (eps - ept)
        s = P["sig"] + ec0 * deps
        k = np.full(self.n, ec0)
        lo = s <= sigmin
        s = np.where(lo, sigmin, s)
        k = np.where(lo, er, k)
        hi = s >= sigmax
        s = np.where(hi, sigmax, s)
        k = np.where(hi, 0.5 * er, k)
        sig = np.where(ur, s, sig)
        e = np.where(ur, k, e)
        tz = ~env & ~ur
        epn = ept + dept
        rel = tz & (eps <= epn)
        sicn, _ = self._tens(dept)
        kk = np.where(dept != 0.0, sicn / np.where(dept == 0.0, 1.0, dept), ec0)
        sig = np.where(rel, kk * (eps - ept), sig)
        e = np.where(rel, kk, e)
        te = tz & ~rel
        st, kt = self._tens(eps - ept)
        sig = np.where(te, st, sig)
        e = np.where(te, kt, e)
        dept = np.where(te, eps - ept, dept)
        ecmin = np.where(env, eps, ecmin)
        if same.any():
            sig = np.where(same, P["sig"], sig)
            e = np.where(same, P["e"], e)
            ecmin = np.where(same, P["ecmin"], ecmin)
            dept = np.where(same, P["dept"], dept)
        self.T = dict(ecmin=ecmin, dept=dept, eps=eps, sig=sig, e=e)
        self.strain, self.stress, self.tangent = eps, sig, e

    def commit(self):
        self.P = {k: v.copy() for k, v in self.T.items()}

    def revert(self):
        self._load()


# ============================================================================
class _ScalarGroup(UniaxialMaterial):
    """Vector wrapper around a scalar state machine (few material points)."""

    def __init__(self, n: int = 1):
        super().__init__(n)
        self._pts = [self._new_point() for _ in range(self.n)]
        self._sync()

    def _new_point(self):
        raise NotImplementedError

    def _sync(self):
        self.strain = np.array([p.Tstrain for p in self._pts])
        self.stress = np.array([p.Tstress for p in self._pts])
        self.tangent = np.array([p.Ttangent for p in self._pts])
        self.damping_tangent = np.array([getattr(p, "Tdamp", 0.0) for p in self._pts])

    def set_trial(self, strain, rate=None):
        e = _arr(strain, self.n)
        r = np.zeros(self.n) if rate is None else _arr(rate, self.n)
        for p, ei, ri in zip(self._pts, e, r):
            p.set_trial(float(ei), float(ri))
        self._sync()

    def commit(self):
        for p in self._pts:
            p.commit()

    def revert(self):
        for p in self._pts:
            p.revert()
        self._sync()


class _HystereticPoint:
    POS_INF, NEG_INF = 1.0e16, -1.0e16

    def __init__(self, m):
        self.m = m
        self.CrotMax = self.CrotMin = 0.0
        self.CrotPu = self.CrotNu = 0.0
        self.CenergyD = 0.0
        self.CloadIndicator = 0
        self.Cstrain = self.Cstress = 0.0
        self.Ctangent = m.E1p
        self.revert()

    def revert(self):
        self.TrotMax, self.TrotMin = self.CrotMax, self.CrotMin
        self.TrotPu, self.TrotNu = self.CrotPu, self.CrotNu
        self.TenergyD, self.TloadIndicator = self.CenergyD, self.CloadIndicator
        self.Tstrain, self.Tstress, self.Ttangent = self.Cstrain, self.Cstress, self.Ctangent

    def commit(self):
        self.CrotMax, self.CrotMin = self.TrotMax, self.TrotMin
        self.CrotPu, self.CrotNu = self.TrotPu, self.TrotNu
        self.CenergyD, self.CloadIndicator = self.TenergyD, self.TloadIndicator
        self.Cstrain, self.Cstress, self.Ctangent = self.Tstrain, self.Tstress, self.Ttangent

    def set_trial(self, strain, rate=0.0):
        m = self.m
        self.TrotMax, self.TrotMin = self.CrotMax, self.CrotMin
        self.TenergyD = self.CenergyD
        self.TrotPu, self.TrotNu = self.CrotPu, self.CrotNu
        self.Tstrain = strain
        d = strain - self.Cstrain
        self.TloadIndicator = self.CloadIndicator
        if self.TloadIndicator == 0:
            self.TloadIndicator = 2 if d < 0.0 else 1
        if strain >= self.CrotMax:
            self.TrotMax = strain
            self.Ttangent = m.pos_tangent(strain)
            self.Tstress = m.pos_stress(strain)
            self.TloadIndicator = 1
        elif strain <= self.CrotMin:
            self.TrotMin = strain
            self.Ttangent = m.neg_tangent(strain)
            self.Tstress = m.neg_stress(strain)
            self.TloadIndicator = 2
        else:
            if d < 0.0:
                self._neg_inc(d)
            elif d > 0.0:
                self._pos_inc(d)
            else:
                self.Tstress, self.Ttangent = self.Cstress, self.Ctangent
        self.TenergyD = self.CenergyD + 0.5 * (self.Cstress + self.Tstress) * d

    def _pos_inc(self, d):
        m = self.m
        kn = (self.CrotMin / m.rot1n) ** m.beta
        kn = 1.0 if kn < 1.0 else 1.0 / kn
        kp = (self.CrotMax / m.rot1p) ** m.beta
        kp = 1.0 if kp < 1.0 else 1.0 / kp
        if self.TloadIndicator == 2:
            self.TloadIndicator = 1
            if self.Cstress <= 0.0:
                self.TrotNu = self.Cstrain - self.Cstress / (m.E1n * kn)
                energy = self.CenergyD - 0.5 * self.Cstress / (m.E1n * kn) * self.Cstress
                damfc = 0.0
                if self.CrotMin < m.rot1n:
                    damfc = m.damfc2 * energy / m.energyA
                    damfc += m.damfc1 * (self.CrotMin - m.rot1n) / m.rot1n
                self.TrotMax = self.CrotMax * (1.0 + damfc)
        self.TloadIndicator = 1
        self.TrotMax = max(self.TrotMax, m.rot1p)
        maxmom = m.pos_stress(self.TrotMax)
        rotlim = m.neg_rotlim(self.CrotMin)
        rotrel = max(rotlim, self.TrotNu)
        rotmp2 = self.TrotMax - (1.0 - m.pinchY) * maxmom / (m.E1p * kp)
        rotch = rotrel + (rotmp2 - rotrel) * m.pinchX
        s = self.Tstrain
        if s < self.TrotNu:
            self.Ttangent = m.E1n * kn
            self.Tstress = self.Cstress + self.Ttangent * d
            if self.Tstress >= 0.0:
                self.Tstress = 0.0
                self.Ttangent = m.E1n * 1.0e-9
        elif s < rotch:
            if s <= rotrel:
                self.Tstress = 0.0
                self.Ttangent = m.E1p * 1.0e-9
            else:
                self.Ttangent = maxmom * m.pinchY / (rotch - rotrel)
                t1 = self.Cstress + m.E1p * kp * d
                t2 = (s - rotrel) * self.Ttangent
                if t1 < t2:
                    self.Tstress, self.Ttangent = t1, m.E1p * kp
                else:
                    self.Tstress = t2
        else:
            self.Ttangent = (1.0 - m.pinchY) * maxmom / (self.TrotMax - rotch)
            t1 = self.Cstress + m.E1p * kp * d
            t2 = m.pinchY * maxmom + (s - rotch) * self.Ttangent
            if t1 < t2:
                self.Tstress, self.Ttangent = t1, m.E1p * kp
            else:
                self.Tstress = t2

    def _neg_inc(self, d):
        m = self.m
        kn = (self.CrotMin / m.rot1n) ** m.beta
        kn = 1.0 if kn < 1.0 else 1.0 / kn
        kp = (self.CrotMax / m.rot1p) ** m.beta
        kp = 1.0 if kp < 1.0 else 1.0 / kp
        if self.TloadIndicator == 1:
            self.TloadIndicator = 2
            if self.Cstress >= 0.0:
                self.TrotPu = self.Cstrain - self.Cstress / (m.E1p * kp)
                energy = self.CenergyD - 0.5 * self.Cstress / (m.E1p * kp) * self.Cstress
                damfc = 0.0
                if self.CrotMax > m.rot1p:
                    damfc = m.damfc2 * energy / m.energyA
                    damfc += m.damfc1 * (self.CrotMax - m.rot1p) / m.rot1p
                self.TrotMin = self.CrotMin * (1.0 + damfc)
        self.TloadIndicator = 2
        self.TrotMin = min(self.TrotMin, m.rot1n)
        minmom = m.neg_stress(self.TrotMin)
        rotlim = m.pos_rotlim(self.CrotMax)
        rotrel = min(rotlim, self.TrotPu)
        rotmp2 = self.TrotMin - (1.0 - m.pinchY) * minmom / (m.E1n * kn)
        rotch = rotrel + (rotmp2 - rotrel) * m.pinchX
        s = self.Tstrain
        if s > self.TrotPu:
            self.Ttangent = m.E1p * kp
            self.Tstress = self.Cstress + self.Ttangent * d
            if self.Tstress <= 0.0:
                self.Tstress = 0.0
                self.Ttangent = m.E1p * 1.0e-9
        elif s > rotch:
            if s >= rotrel:
                self.Tstress = 0.0
                self.Ttangent = m.E1n * 1.0e-9
            else:
                self.Ttangent = minmom * m.pinchY / (rotch - rotrel)
                t1 = self.Cstress + m.E1n * kn * d
                t2 = (s - rotrel) * self.Ttangent
                if t1 > t2:
                    self.Tstress, self.Ttangent = t1, m.E1n * kn
                else:
                    self.Tstress = t2
        else:
            self.Ttangent = (1.0 - m.pinchY) * minmom / (self.TrotMin - rotch)
            t1 = self.Cstress + m.E1n * kn * d
            t2 = m.pinchY * minmom + (s - rotch) * self.Ttangent
            if t1 > t2:
                self.Tstress, self.Ttangent = t1, m.E1n * kn
            else:
                self.Tstress = t2


class Hysteretic(_ScalarGroup):
    """OpenSees Hysteretic material: tri-linear envelope (s1p e1p s2p e2p s3p e3p,
    negative values for the negative side), pinching pinchX/pinchY, damage
    damage1 (ductility) / damage2 (energy), unloading degradation beta."""

    type_name = "Hysteretic"

    def __init__(self, s1p, e1p, s2p, e2p, s3p=None, e3p=None, s1n=None, e1n=None, s2n=None, e2n=None, s3n=None,
                 e3n=None, pinchX=1.0, pinchY=1.0, damage1=0.0, damage2=0.0, beta=0.0, n: int = 1):
        if s3p is None:  # two-point envelope -> third point on the extension of segment 2
            s3p, e3p = s2p + (s2p - s1p) / (e2p - e1p) * e2p, 2.0 * e2p
        if s1n is None:
            s1n, e1n, s2n, e2n, s3n, e3n = -s1p, -e1p, -s2p, -e2p, -s3p, -e3p
        elif s3n is None:
            s3n, e3n = s2n + (s2n - s1n) / (e2n - e1n) * e2n, 2.0 * e2n
        self.mom1p, self.rot1p, self.mom2p, self.rot2p, self.mom3p, self.rot3p = map(float, (s1p, e1p, s2p, e2p, s3p, e3p))
        self.mom1n, self.rot1n, self.mom2n, self.rot2n, self.mom3n, self.rot3n = map(float, (s1n, e1n, s2n, e2n, s3n, e3n))
        self.pinchX, self.pinchY = float(pinchX), float(pinchY)
        self.damfc1, self.damfc2, self.beta = float(damage1), float(damage2), float(beta)
        self.E1p = self.mom1p / self.rot1p
        self.E2p = (self.mom2p - self.mom1p) / (self.rot2p - self.rot1p)
        self.E3p = (self.mom3p - self.mom2p) / (self.rot3p - self.rot2p)
        self.E1n = self.mom1n / self.rot1n
        self.E2n = (self.mom2n - self.mom1n) / (self.rot2n - self.rot1n)
        self.E3n = (self.mom3n - self.mom2n) / (self.rot3n - self.rot2n)
        self.energyA = 0.5 * (self.rot1p * self.mom1p + (self.rot2p - self.rot1p) * (self.mom2p + self.mom1p)
                              + (self.rot3p - self.rot2p) * (self.mom3p + self.mom2p) + self.rot1n * self.mom1n
                              + (self.rot2n - self.rot1n) * (self.mom2n + self.mom1n)
                              + (self.rot3n - self.rot2n) * (self.mom3n + self.mom2n))
        super().__init__(n)

    def params(self):
        return dict(s1p=self.mom1p, e1p=self.rot1p, s2p=self.mom2p, e2p=self.rot2p, s3p=self.mom3p, e3p=self.rot3p,
                    s1n=self.mom1n, e1n=self.rot1n, s2n=self.mom2n, e2n=self.rot2n, s3n=self.mom3n, e3n=self.rot3n,
                    pinchX=self.pinchX, pinchY=self.pinchY, damage1=self.damfc1, damage2=self.damfc2, beta=self.beta)

    def _new_point(self):
        return _HystereticPoint(self)

    @property
    def initial_tangent(self):
        return np.full(self.n, self.E1p)

    def pos_stress(self, s):
        if s <= 0.0:
            return 0.0
        if s <= self.rot1p:
            return self.E1p * s
        if s <= self.rot2p:
            return self.mom1p + self.E2p * (s - self.rot1p)
        if s <= self.rot3p or self.E3p > 0.0:
            return self.mom2p + self.E3p * (s - self.rot2p)
        return self.mom3p

    def neg_stress(self, s):
        if s >= 0.0:
            return 0.0
        if s >= self.rot1n:
            return self.E1n * s
        if s >= self.rot2n:
            return self.mom1n + self.E2n * (s - self.rot1n)
        if s >= self.rot3n or self.E3n > 0.0:
            return self.mom2n + self.E3n * (s - self.rot2n)
        return self.mom3n

    def pos_tangent(self, s):
        if s <= 0.0:
            return self.E1p * 1.0e-9
        if s <= self.rot1p:
            return self.E1p
        if s <= self.rot2p:
            return self.E2p
        if s <= self.rot3p or self.E3p > 0.0:
            return self.E3p
        return self.E1p * 1.0e-9

    def neg_tangent(self, s):
        if s >= 0.0:
            return self.E1n * 1.0e-9
        if s >= self.rot1n:
            return self.E1n
        if s >= self.rot2n:
            return self.E2n
        if s >= self.rot3n or self.E3n > 0.0:
            return self.E3n
        return self.E1n * 1.0e-9

    def pos_rotlim(self, s):
        if s < self.rot1p:
            return _HystereticPoint.POS_INF
        if s < self.rot2p and self.E2p < 0.0:
            return self.rot1p - self.mom1p / self.E2p
        if s < self.rot3p and self.E3p < 0.0:
            return self.rot2p - self.mom2p / self.E3p
        return _HystereticPoint.POS_INF

    def neg_rotlim(self, s):
        if s > self.rot1n:
            return _HystereticPoint.NEG_INF
        if s > self.rot2n and self.E2n < 0.0:
            return self.rot1n - self.mom1n / self.E2n
        if s > self.rot3n and self.E3n < 0.0:
            return self.rot2n - self.mom2n / self.E3n
        return _HystereticPoint.NEG_INF


# ============================================================================
class _BoucWenPoint:
    def __init__(self, m):
        self.m = m
        self.Cstrain = self.Cz = self.Ce = 0.0
        self.Cstress = 0.0
        self.Ctangent = m.ko * (m.alpha + (1 - m.alpha) * m.Ao)
        self.revert()

    def revert(self):
        self.Tstrain, self.Tz, self.Te = self.Cstrain, self.Cz, self.Ce
        self.Tstress, self.Ttangent = self.Cstress, self.Ctangent

    def commit(self):
        self.Cstrain, self.Cz, self.Ce = self.Tstrain, self.Tz, self.Te
        self.Cstress, self.Ctangent = self.Tstress, self.Ttangent

    def set_trial(self, strain, rate=0.0):
        m = self.m
        self.Tstrain = strain
        d = strain - self.Cstrain
        if abs(d) < 1e-300:
            self.Tz, self.Te, self.Tstress, self.Ttangent = self.Cz, self.Ce, self.Cstress, self.Ctangent
            return
        z = 0.01
        z_old, z_new = 0.01, 1.0
        it = 0
        c = (1.0 - m.alpha) * m.ko
        while abs(z_old - z_new) > m.tol and it < m.maxIter:
            Te = self.Ce + c * d * z
            TA = m.Ao - m.deltaA * Te
            Tnu = 1.0 + m.deltaNu * Te
            Teta = 1.0 + m.deltaEta * Te
            sign = math.copysign(1.0, d * z) if d * z != 0 else 0.0
            Psi = m.gamma + m.beta * sign
            az = abs(z)
            Phi = TA - (az ** m.nexp) * Psi * Tnu
            f = z - self.Cz - Phi / Teta * d
            Te_ = c * d
            TA_ = -m.deltaA * Te_
            Tnu_ = m.deltaNu * Te_
            Teta_ = m.deltaEta * Te_
            sz = math.copysign(1.0, z) if z != 0 else 0.0
            pw = az ** (m.nexp - 1.0) if az > 0 else 0.0
            Phi_ = TA_ - m.nexp * pw * sz * Psi * Tnu - (az ** m.nexp) * Psi * Tnu_
            f_ = 1.0 - (Phi_ * Teta - Phi * Teta_) / (Teta * Teta) * d
            if f_ == 0.0:
                break
            z_new = z - f / f_
            z_old = z
            z = z_new
            it += 1
        self.Tz = z
        self.Tstress = m.alpha * m.ko * strain + c * z
        Te = self.Ce + c * d * z
        TA = m.Ao - m.deltaA * Te
        Tnu = 1.0 + m.deltaNu * Te
        Teta = 1.0 + m.deltaEta * Te
        self.Te = Te
        sign = math.copysign(1.0, d * z) if d * z != 0 else 0.0
        Psi = m.gamma + m.beta * sign
        az = abs(z)
        Phi = TA - (az ** m.nexp) * Psi * Tnu
        # dz/deps from the implicit equation (energy dependence through d is neglected, as OpenSees)
        sz = math.copysign(1.0, z) if z != 0 else 0.0
        pw = az ** (m.nexp - 1.0) if az > 0 else 0.0
        f_z = 1.0 + m.nexp * pw * sz * Psi * Tnu / Teta * d
        dzde = (Phi / Teta) / f_z if f_z != 0 else 0.0
        self.Ttangent = m.alpha * m.ko + c * dzde


class BoucWen(_ScalarGroup):
    """Bouc-Wen smooth hysteresis (OpenSees BoucWen): stress = alpha*ko*eps + (1-alpha)*ko*z,
    dz = (A - |z|^n (gamma + beta sgn(de z)) nu) / eta * de with energy degradation."""

    type_name = "BoucWen"

    def __init__(self, alpha, ko, nexp, gamma, beta, Ao=1.0, deltaA=0.0, deltaNu=0.0, deltaEta=0.0, tol=1e-8,
                 maxIter=20, n: int = 1):
        self.alpha, self.ko, self.nexp = float(alpha), float(ko), float(nexp)
        self.gamma, self.beta, self.Ao = float(gamma), float(beta), float(Ao)
        self.deltaA, self.deltaNu, self.deltaEta = float(deltaA), float(deltaNu), float(deltaEta)
        self.tol, self.maxIter = float(tol), int(maxIter)
        super().__init__(n)

    @classmethod
    def csi_wen(cls, k: float, yield_force: float, ratio: float, exp: float = 2.0, n: int = 1) -> "BoucWen":
        """CSI "Plastic (Wen)" link property (k, yield, post-yield ratio, yield exponent)."""
        g = 0.5 * (k / yield_force) ** exp
        return cls(ratio, k, exp, g, g, 1.0, n=n)

    def params(self):
        return dict(alpha=self.alpha, ko=self.ko, nexp=self.nexp, gamma=self.gamma, beta=self.beta, Ao=self.Ao,
                    deltaA=self.deltaA, deltaNu=self.deltaNu, deltaEta=self.deltaEta, tol=self.tol, maxIter=self.maxIter)

    def _new_point(self):
        return _BoucWenPoint(self)

    @property
    def initial_tangent(self):
        return np.full(self.n, self.ko * (self.alpha + (1 - self.alpha) * self.Ao))


# ============================================================================
class Viscous(UniaxialMaterial):
    """Rate dependent damper: stress = C |rate|^alpha sgn(rate) (OpenSees Viscous)."""

    type_name = "Viscous"
    rate_dependent = True

    def __init__(self, C: float, alpha: float = 1.0, min_rate: float = 1e-11, n: int = 1):
        super().__init__(n)
        self.C, self.alpha, self.min_rate = float(C), float(alpha), float(min_rate)
        self._c = np.zeros(n)

    def params(self):
        return {"C": self.C, "alpha": self.alpha, "min_rate": self.min_rate}

    @property
    def initial_tangent(self):
        return np.zeros(self.n)

    def set_trial(self, strain, rate=None):
        self.strain = _arr(strain, self.n)
        r = np.zeros(self.n) if rate is None else _arr(rate, self.n)
        ar = np.maximum(np.abs(r), self.min_rate)
        self.stress = self.C * np.power(np.abs(r), self.alpha) * np.sign(r)
        self.tangent = np.zeros(self.n)
        self.damping_tangent = self.C * self.alpha * np.power(ar, self.alpha - 1.0)

    def commit(self):
        self._c = self.strain.copy()

    def revert(self):
        self.set_trial(self._c)


# ============================================================================
# CSI-style hinge law: backbone A-B-C-D-E with kinematic or Takeda hysteresis
# ============================================================================
class Backbone:
    """Piecewise linear skeleton through (0,0) on each side.  ``pos``/``neg``:
    lists of (deformation, force) points after A (B, C, D, E ...), deformations
    increasing in magnitude.  ``drop``: after the last point the force drops to
    ``residual`` (CSI "drops load after point E"); otherwise the last segment is
    extrapolated."""

    def __init__(self, pos: Sequence[Tuple[float, float]], neg: Optional[Sequence[Tuple[float, float]]] = None,
                 drop: bool = True, residual: float = 0.0):
        pos = [(float(d), float(f)) for d, f in pos]
        if neg is None:
            neg = [(-d, -f) for d, f in pos]
        neg = [(-abs(float(d)), -abs(float(f))) for d, f in neg]
        self.pd = np.array([0.0] + [p[0] for p in pos])
        self.pf = np.array([0.0] + [p[1] for p in pos])
        self.nd = np.array([0.0] + [p[0] for p in neg])
        self.nf = np.array([0.0] + [p[1] for p in neg])
        if np.any(np.diff(self.pd) <= 0) or np.any(np.diff(self.nd) >= 0):
            raise ValueError("Backbone deformations must increase in magnitude")
        self.drop = drop
        self.residual = residual
        self.k0p = self.pf[1] / self.pd[1]
        self.k0n = self.nf[1] / self.nd[1]

    def _eval(self, d, D, F, sign):
        a = abs(d)
        Da = np.abs(D)
        if a <= Da[-1]:
            i = int(np.searchsorted(Da, a, side="right")) - 1
            i = min(max(i, 0), len(D) - 2)
            k = (F[i + 1] - F[i]) / (D[i + 1] - D[i])
            return F[i] + k * (d - D[i]), k
        if self.drop:
            # CSI: beyond the last point the force drops to the residual
            return sign * self.residual * abs(F[-1]), 1e-9 * abs(F[1] / D[1])
        k = (F[-1] - F[-2]) / (D[-1] - D[-2])
        return F[-1] + k * (d - D[-1]), k

    def force(self, d: float) -> Tuple[float, float]:
        if d >= 0.0:
            return self._eval(d, self.pd, self.pf, 1.0)
        return self._eval(d, self.nd, self.nf, -1.0)

    def pos_points(self):
        return list(zip(self.pd[1:], self.pf[1:]))

    def neg_points(self):
        return list(zip(self.nd[1:], self.nf[1:]))


class _PointEnvelope:
    """Backbone seen by one hinge point.

    ``regularise`` mode (time history, or hinges without vertical drops): the
    regularised backbone.  ``redistribute`` mode (static cases): the backbone
    with *vertical* drops (CSI); a drop whose start has been passed is held at
    the strength before the drop (level 0) until the static solver releases
    it (level 0 -> 1) at a fixed control displacement, as CSI's unloading of
    the structure after a strength loss.  Released drops stay released in
    either mode."""

    def __init__(self, m, pt):
        self.m = m
        self.pt = pt

    # elastic data of the (regularised) backbone; explicit so that copy/deepcopy work
    pd = property(lambda self: self.m.backbone.pd)
    nd = property(lambda self: self.m.backbone.nd)
    pf = property(lambda self: self.m.backbone.pf)
    nf = property(lambda self: self.m.backbone.nf)
    k0p = property(lambda self: self.m.backbone.k0p)
    k0n = property(lambda self: self.m.backbone.k0n)

    def force(self, d):
        m, pt = self.m, self.pt
        if m.vertical is None:
            return m.backbone.force(d)
        released = any(v > 0.0 for v in pt.levels.values())
        if m.drop_mode != "redistribute" and not released:
            return m.backbone.force(d)
        sgn = 1.0 if d >= 0.0 else -1.0
        drops = m.vertical["drops_pos"] if d >= 0.0 else m.vertical["drops_neg"]
        a = abs(d)
        for di, dF, eps in drops:  # inside a vertical sliver: use the branch just after it
            if di <= a < di + eps:
                a = di + eps
        f, k = m.vbackbone.force(sgn * a)
        for i, (di, dF, eps) in enumerate(drops):
            if abs(d) >= di:
                f += sgn * (1.0 - pt.levels.get((sgn, i), 0.0)) * dF
        return f, k


def _frozen_softening(pt, bb, d) -> bool:
    """Event-to-event style "loading" assumption: a hinge that is on a descending
    branch of its backbone at the start of the increment keeps following that
    branch (in either direction) while ``freeze_loading`` is set on its material.
    Used by the static solver to pass the softening bifurcation (continued
    softening vs. elastic unloading) after a strength drop."""
    if not getattr(pt.m, "freeze_loading", False):
        return False
    c = pt.Cstrain
    on_env = (c > 0 and c >= pt.CdMax - 1e-15) or (c < 0 and c <= pt.CdMin + 1e-15)
    if not on_env:
        return False
    f0, k0 = bb.force(c)
    if k0 >= 0.0:
        return False
    pt.Tstress, pt.Ttangent = bb.force(d) if abs(d) > abs(c) and d * c > 0 else (f0 + k0 * (d - c), k0)
    if d > pt.CdMax:
        pt.TdMax = d
    if d < pt.CdMin:
        pt.TdMin = d
    return True


class _TakedaPoint:
    """Peak-oriented (Takeda / Clough) rule on a softening backbone.

    Unloading follows the initial stiffness; once the force changes sign the
    path aims at the maximum previous excursion on the opposite backbone; a
    reversal on the way unloads again with the initial stiffness.  This is
    the "Takeda" hysteresis of the CSI hinge definition in its simplified
    form (Takeda, Sozen & Nielsen 1970; CSI Analysis Reference, hinge
    hysteresis types)."""

    def __init__(self, m):
        self.m = m
        self.levels = {}
        self.bb = _PointEnvelope(m, self)
        bb = m.backbone
        self.CdMax, self.CdMin = bb.pd[1], bb.nd[1]
        self.Cstrain = self.Cstress = 0.0
        self.Ctangent = bb.k0p
        self.revert()

    def revert(self):
        self.TdMax, self.TdMin = self.CdMax, self.CdMin
        self.Tstrain, self.Tstress, self.Ttangent = self.Cstrain, self.Cstress, self.Ctangent

    def commit(self):
        self.CdMax, self.CdMin = self.TdMax, self.TdMin
        self.Cstrain, self.Cstress, self.Ctangent = self.Tstrain, self.Tstress, self.Ttangent
        self._dirty = False

    def set_trial(self, d, rate=0.0):
        bb = self.bb
        self.TdMax, self.TdMin = self.CdMax, self.CdMin
        self.Tstrain = d
        dd = d - self.Cstrain
        if dd == 0.0 and not getattr(self, "_dirty", False):
            self.Tstress, self.Ttangent = self.Cstress, self.Ctangent
            return
        if _frozen_softening(self, bb, d):
            return
        if d >= self.CdMax:
            self.TdMax = d
            self.Tstress, self.Ttangent = bb.force(d)
            return
        if d <= self.CdMin:
            self.TdMin = d
            self.Tstress, self.Ttangent = bb.force(d)
            return
        k0 = bb.k0p if dd > 0 else bb.k0n
        s_el = self.Cstress + k0 * dd
        if dd > 0.0:
            tgt_d = self.CdMax
            tgt_f, _ = bb.force(tgt_d)
            # zero crossing of the elastic path from the committed point
            d0 = self.Cstrain - self.Cstress / k0 if self.Cstress < 0.0 else None
            if self.Cstress < 0.0 and s_el <= 0.0:
                self.Tstress, self.Ttangent = s_el, k0
                return
            if d0 is None:  # already positive: reloading line through the last zero point
                d0 = self.Cstrain - self.Cstress / self._line_slope(self.Cstrain, self.Cstress, tgt_d, tgt_f)
            ks = tgt_f / (tgt_d - d0) if tgt_d > d0 else k0
            s_line = ks * (d - d0)
            if s_el < s_line and self.Cstress >= 0.0:
                self.Tstress, self.Ttangent = s_el, k0
            else:
                self.Tstress, self.Ttangent = s_line, ks
        else:
            tgt_d = self.CdMin
            tgt_f, _ = bb.force(tgt_d)
            if self.Cstress > 0.0 and s_el >= 0.0:
                self.Tstress, self.Ttangent = s_el, k0
                return
            if self.Cstress > 0.0:
                d0 = self.Cstrain - self.Cstress / k0
            else:
                d0 = self.Cstrain - self.Cstress / self._line_slope(self.Cstrain, self.Cstress, tgt_d, tgt_f)
            ks = tgt_f / (tgt_d - d0) if tgt_d < d0 else k0
            s_line = ks * (d - d0)
            if s_el > s_line and self.Cstress <= 0.0:
                self.Tstress, self.Ttangent = s_el, k0
            else:
                self.Tstress, self.Ttangent = s_line, ks

    @staticmethod
    def _line_slope(dc, fc, dt, ft):
        if dt == dc:
            return 1.0e30
        k = (ft - fc) / (dt - dc)
        return k if k != 0.0 else 1.0e30


class _KinematicPoint:
    """CSI "Kinematic" hysteresis on the hardening part of the backbone as an
    Iwan (parallel elasto-plastic springs, Masing) model: unloading and
    reloading segments are the backbone segments with doubled length.  Beyond
    the peak (negative slope) the force is limited by the backbone at the
    largest excursion in that direction (strength memory)."""

    def __init__(self, m):
        self.m = m
        self.levels = {}
        self.bb = _PointEnvelope(m, self)
        self.Cep = np.zeros(len(m.iwan_k))  # plastic deformation of each Iwan spring
        self.Cstrain = self.Cstress = 0.0
        self.Ctangent = float(np.sum(m.iwan_k))
        self.CdMax, self.CdMin = 0.0, 0.0
        self.revert()

    def revert(self):
        self.Tep = self.Cep.copy()
        self.TdMax, self.TdMin = self.CdMax, self.CdMin
        self.Tstrain, self.Tstress, self.Ttangent = self.Cstrain, self.Cstress, self.Ctangent

    def commit(self):
        self.Cep = self.Tep.copy()
        self.CdMax, self.CdMin = self.TdMax, self.TdMin
        self.Cstrain, self.Cstress, self.Ctangent = self.Tstrain, self.Tstress, self.Ttangent

    def set_trial(self, d, rate=0.0):
        m = self.m
        self.Tstrain = d
        if _frozen_softening(self, self.bb, d):
            self.Tep = self.Cep.copy()
            return
        k, fy = m.iwan_k, m.iwan_fy
        f = k * (d - self.Cep)
        over = np.abs(f) > fy
        fs = np.where(over, np.sign(f) * fy, f)
        self.Tep = np.where(over, d - fs / k, self.Cep)
        s = float(fs.sum())
        kt = float(np.sum(np.where(over, 0.0, k))) + m.k_last
        s += m.k_last * d
        self.TdMax, self.TdMin = max(self.CdMax, d), min(self.CdMin, d)
        bb = self.bb
        # strength limit once the peak has been passed
        if self.TdMax > m.d_peak_p and s > 0:
            cap, kc = bb.force(self.TdMax)
            if s > cap:
                s, kt = cap, (kc if d >= self.CdMax else 1e-9 * bb.k0p)
        if self.TdMin < m.d_peak_n and s < 0:
            cap, kc = bb.force(self.TdMin)
            if s < cap:
                s, kt = cap, (kc if d <= self.CdMin else 1e-9 * bb.k0n)
        self.Tstress, self.Ttangent = s, kt


class HingeMaterial(_ScalarGroup):
    """Deformation (rotation or displacement) - force law of a CSI-style
    plastic hinge: backbone A-B-C-D-E with ``hysteresis`` in {"Kinematic",
    "Takeda", "Elastic"}.  The tangent never drops below ``1e-6`` times the
    elastic stiffness, so a perfectly plastic mechanism keeps a (tiny) stiffness
    for the solvers; the stress is not affected."""

    type_name = "Hinge"
    TANGENT_FLOOR = 1e-6

    def _sync(self):
        super()._sync()
        floor = self.TANGENT_FLOOR * self.backbone.k0p
        small = np.abs(self.tangent) < floor
        if small.any():
            self.tangent = np.where(small, floor, self.tangent)

    def __init__(self, pos, neg=None, hysteresis: str = "Kinematic", drop: bool = True, residual: float = 0.0,
                 vertical: Optional[dict] = None, n: int = 1):
        self.pos, self.neg = [tuple(p) for p in pos], (None if neg is None else [tuple(p) for p in neg])
        self.hysteresis = hysteresis
        self.drop, self.residual = drop, residual
        self.backbone = Backbone(self.pos, self.neg, drop, residual)
        # vertical drops (CSI): {"pos", "neg": backbone points with vertical slivers,
        # "drops_pos", "drops_neg": [(deformation at the drop, force loss, sliver width)]}
        self.vertical = vertical
        self.vbackbone = Backbone(vertical["pos"], vertical.get("neg"), False, 0.0) if vertical else None
        self.drop_mode = "regularise"
        if hysteresis.lower() == "kinematic":
            self._build_iwan()
        super().__init__(n)

    def _build_iwan(self):
        bb = self.backbone
        # use the hardening (non-negative slope) part of the positive backbone
        d, f = [0.0], [0.0]
        for dd, ff in bb.pos_points():
            if ff < f[-1]:
                break
            d.append(dd)
            f.append(ff)
        d, f = np.array(d), np.array(f)
        slopes = np.diff(f) / np.diff(d)
        self.d_peak_p = d[-1]
        dn = [0.0]
        for dd, ff in bb.neg_points():
            if abs(ff) < abs(bb.nf[len(dn) - 1]):
                break
            dn.append(dd)
        self.d_peak_n = dn[-1]
        # Iwan springs: k_i = s_{i-1} - s_i, yield at d_i ; last slope is a linear spring
        ks, fys = [], []
        for i in range(1, len(slopes) + 1):
            s_prev = slopes[i - 1]
            s_next = slopes[i] if i < len(slopes) else None
            if s_next is None:
                break
            ki = s_prev - s_next
            if ki > 0:
                ks.append(ki)
                fys.append(ki * d[i])
        self.k_last = float(slopes[-1]) if len(slopes) > 1 else 0.0
        if len(slopes) == 1:  # elastic - perfectly plastic beyond B handled by the cap
            ks, fys = [float(slopes[0])], [float(slopes[0] * d[1])]
            self.k_last = 0.0
        self.iwan_k = np.array(ks if ks else [float(slopes[0])])
        self.iwan_fy = np.array(fys if fys else [float(slopes[0] * d[1])])

    def params(self):
        return dict(pos=self.pos, neg=self.neg, hysteresis=self.hysteresis, drop=self.drop, residual=self.residual,
                    vertical=self.vertical)

    # ---- vertical drops (static cases) ------------------------------------
    def pending_drops(self):
        """Drops passed by the committed state and not fully released: [(point, key, level)]."""
        out = []
        if self.vertical is None or self.drop_mode != "redistribute":
            return out
        for pt in self._pts:
            for sgn, key in ((1.0, "drops_pos"), (-1.0, "drops_neg")):
                ext = pt.CdMax if sgn > 0 else -pt.CdMin
                for i, (di, dF, eps) in enumerate(self.vertical[key]):
                    lev = pt.levels.get((sgn, i), 0.0)
                    if ext >= di and lev < 1.0:
                        out.append((pt, (sgn, i), lev))
        return out

    def set_level(self, pt, key, level):
        pt.levels[key] = float(level)
        pt._dirty = True

    def _new_point(self):
        h = self.hysteresis.lower()
        if h == "kinematic":
            return _KinematicPoint(self)
        if h == "takeda":
            return _TakedaPoint(self)
        if h == "elastic":
            return _ElasticNLPoint(self)
        raise ValueError(f"Unknown hinge hysteresis {self.hysteresis!r}")

    @property
    def initial_tangent(self):
        return np.full(self.n, self.backbone.k0p)


class _ElasticNLPoint:
    """Nonlinear elastic: loads and unloads along the backbone (no energy dissipation)."""

    def __init__(self, m):
        self.m = m
        self.levels = {}
        self.bb = _PointEnvelope(m, self)
        self.Cstrain = self.Cstress = 0.0
        self.Ctangent = m.backbone.k0p
        self.revert()

    def revert(self):
        self.Tstrain, self.Tstress, self.Ttangent = self.Cstrain, self.Cstress, self.Ctangent

    def commit(self):
        self.Cstrain, self.Cstress, self.Ctangent = self.Tstrain, self.Tstress, self.Ttangent

    def set_trial(self, d, rate=0.0):
        self.Tstrain = d
        self.Tstress, self.Ttangent = self.bb.force(d)

    @property
    def CdMax(self):
        return max(self.Cstrain, 0.0)

    @property
    def CdMin(self):
        return min(self.Cstrain, 0.0)


# ============================================================================
class Parallel(UniaxialMaterial):
    """Materials acting in parallel (same strain, stresses add)."""

    type_name = "Parallel"

    def __init__(self, materials, n: int = 1):
        super().__init__(n)
        self.mats = [m.clone(n) for m in materials]
        self.rate_dependent = any(m.rate_dependent for m in self.mats)
        self.tangent = sum(m.initial_tangent for m in self.mats)

    def params(self):
        return {"materials": self.mats}

    @property
    def initial_tangent(self):
        return sum(m.initial_tangent for m in self.mats)

    def set_trial(self, strain, rate=None):
        for m in self.mats:
            m.set_trial(strain, rate)
        self.strain = self.mats[0].strain
        self.stress = sum(m.stress for m in self.mats)
        self.tangent = sum(m.tangent for m in self.mats)
        self.damping_tangent = sum(m.damping_tangent for m in self.mats)

    def commit(self):
        for m in self.mats:
            m.commit()

    def revert(self):
        for m in self.mats:
            m.revert()
        self.stress = sum(m.stress for m in self.mats)
        self.tangent = sum(m.tangent for m in self.mats)


# ============================================================================
MATERIAL_TYPES = {cls.type_name: cls for cls in (Elastic, ElasticPP, Steel01, Steel02, Concrete01, Concrete02,
                                                  Hysteretic, BoucWen, Viscous, HingeMaterial)}


def make_material(type_name: str, params: Dict[str, float], n: int = 1) -> UniaxialMaterial:
    try:
        cls = MATERIAL_TYPES[type_name]
    except KeyError as exc:
        raise ValueError(f"Unknown nonlinear material type {type_name!r}; choose from {sorted(MATERIAL_TYPES)}") from exc
    return cls(**params, n=n)
