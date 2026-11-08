"""Target displacements from a capacity curve.

* :func:`bilinear_asce41` - ASCE/SEI 41 §7.4.3.2.4 idealised force-displacement
  curve (effective stiffness through 0.6 Vy, equal area up to the target).
* :func:`asce41_target_displacement` - coefficient method (ASCE 41 Eq. 7-28).
* :func:`n2_target_displacement` - Eurocode 8-1 Annex B (N2 method).

Coefficients follow the cited documents; the functions report every
intermediate value so that the calculation can be checked by hand.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

G = 9.80665


def _area(d, v):
    return float(np.trapezoid(v, d)) if hasattr(np, "trapezoid") else float(np.trapz(v, d))


def bilinear_asce41(disp: Sequence[float], force: Sequence[float], d_target: Optional[float] = None,
                    iterations: int = 50) -> Dict[str, float]:
    """Bilinear (tri-linear when softening) idealisation per ASCE 41 Fig. 7-3.

    Effective stiffness Ke passes through the curve at 0.6 Vy; the post-yield
    line passes through the curve at ``d_target`` (default: displacement of the
    maximum force); areas under both curves are equal up to that point."""
    d = np.abs(np.asarray(disp, float))
    v = np.abs(np.asarray(force, float))
    order = np.argsort(d)
    d, v = d[order], v[order]
    if d_target is None:
        d_target = float(d[int(np.argmax(v))])
    d_target = min(d_target, float(d[-1]))
    m = d <= d_target + 1e-15
    dd, vv = d[m], v[m]
    v_t = float(np.interp(d_target, d, v))
    A = _area(dd, vv)

    def disp_at(force):
        k = int(np.argmax(vv >= force))
        if k == 0:
            return float(dd[0]) if vv[0] > 0 else 0.0
        return float(dd[k - 1] + (force - vv[k - 1]) * (dd[k] - dd[k - 1]) / (vv[k] - vv[k - 1]))

    Vy = v_t
    Ke = v_t / max(d_target, 1e-300)
    if abs(2.0 * A - v_t * d_target) <= 1e-9 * v_t * d_target:  # linear (elastic) curve
        return {"Ke": Ke, "Vy": v_t, "dy": d_target, "alpha1": 0.0, "d_target": d_target, "V_target": v_t}
    for _ in range(iterations):
        Ke = 0.6 * Vy / max(disp_at(0.6 * Vy), 1e-300)
        # equal areas: Vy (d_t - V_t / Ke) = 2 A - V_t d_t
        Vy_new = (2.0 * A - v_t * d_target) / (d_target - v_t / Ke)
        if not np.isfinite(Vy_new) or Vy_new <= 0:
            break
        done = abs(Vy_new - Vy) <= 1e-12 * Vy
        Vy = Vy_new
        if done:
            break
    dy = Vy / Ke
    alpha = (v_t - Vy) / (d_target - dy) / Ke if d_target > dy else 0.0
    return {"Ke": Ke, "Vy": Vy, "dy": dy, "alpha1": alpha, "d_target": d_target, "V_target": v_t}


# ASCE 41-17 Table 7-5 (shear buildings, triangular load pattern)
_C0_TABLE = {1: 1.0, 2: 1.2, 3: 1.2, 5: 1.3, 10: 1.4}


def _c0_table(n_stories: int) -> float:
    ks = sorted(_C0_TABLE)
    if n_stories >= ks[-1]:
        return _C0_TABLE[ks[-1]]
    return float(np.interp(n_stories, ks, [_C0_TABLE[k] for k in ks]))


def asce41_target_displacement(disp, force, weight: float, Sa_of_T, Ti: float, n_stories: int = 1,
                               C0: Optional[float] = None, Cm: float = 1.0, site_a: float = 60.0,
                               iterations: int = 30) -> Dict[str, float]:
    """ASCE 41 coefficient method: dt = C0 C1 C2 Sa Te^2 / (4 pi^2) g.

    ``disp``/``force``: capacity curve (roof displacement in m, base shear in the
    units of ``weight``); ``Sa_of_T``: callable returning the spectral
    acceleration (g) at a period; ``Ti``: elastic fundamental period (s);
    ``site_a``: 130 (site class A/B), 90 (C), 60 (D, E, F) per ASCE 41 Eq. 7-29;
    ``C0``: from the roof modal participation (Gamma * phi_roof) when given,
    otherwise ASCE 41 Table 7-5."""
    C0 = _c0_table(n_stories) if C0 is None else C0
    d = np.abs(np.asarray(disp, float))
    v = np.abs(np.asarray(force, float))
    dt = None
    res = {}
    for _ in range(iterations):
        bl = bilinear_asce41(d, v, dt)
        Ke, Vy = bl["Ke"], bl["Vy"]
        K_i = float(v[1] / d[1]) if len(d) > 1 and d[1] > 0 else Ke
        Te = Ti * math.sqrt(K_i / Ke)
        Sa = float(Sa_of_T(Te))
        mu = Sa / (Vy / weight) * Cm
        C1 = 1.0 + (mu - 1.0) / (site_a * Te * Te)
        if Te < 0.2:
            C1 = 1.0 + (mu - 1.0) / (site_a * 0.04)
        if Te > 1.0:
            C1 = 1.0
        C2 = 1.0 + ((mu - 1.0) / Te) ** 2 / 800.0
        if Te > 0.7:
            C2 = 1.0
        dt_new = C0 * C1 * C2 * Sa * Te * Te / (4 * math.pi ** 2) * G
        res = {"C0": C0, "C1": C1, "C2": C2, "Te": Te, "Sa": Sa, "mu_strength": mu, "Ke": Ke, "Vy": Vy,
               "alpha1": bl["alpha1"], "target_displacement": dt_new}
        if dt is not None and abs(dt_new - dt) <= 1e-6 * dt_new:
            break
        dt = min(dt_new, float(d[-1]))
    res["mu_max_check"] = None
    return res


def n2_target_displacement(disp, force, masses: Sequence[float], phi: Sequence[float], Se_of_T, Tc: float,
                           iterate: bool = False, iterations: int = 30) -> Dict[str, float]:
    """Eurocode 8-1 Annex B (N2): equivalent SDOF from the floor masses and the
    normalised displacement shape (phi = 1 at the control node), elastic-perfectly
    plastic idealisation by equal energy up to the plastic mechanism (optionally
    iterated so that the idealisation ends at the target displacement, EC8-1
    B.5), target displacement with the Tc rule.  ``Se_of_T`` in g."""
    m = np.asarray(masses, float)
    ph = np.asarray(phi, float)
    m_star = float(m @ ph)
    gamma = m_star / float(m @ (ph * ph))
    d = np.abs(np.asarray(disp, float)) / gamma
    F = np.abs(np.asarray(force, float)) / gamma
    dm = float(d[int(np.argmax(F))])
    res = {}
    for _ in range(iterations):
        mask = d <= dm + 1e-15
        Fy = float(np.interp(dm, d, F)) if dm < d[-1] else float(F[mask].max())
        Fy = float(F[mask].max())
        Em = _area(d[mask], F[mask])
        dy = 2.0 * (dm - Em / Fy)
        dy = max(dy, 1e-12)
        T = 2 * math.pi * math.sqrt(m_star * dy / Fy)
        Se = float(Se_of_T(T)) * G
        det = Se * (T / (2 * math.pi)) ** 2
        if T < Tc:
            if Fy / m_star >= Se:
                dt = det
            else:
                qu = Se * m_star / Fy
                dt = det / qu * (1 + (qu - 1) * Tc / T)
                dt = max(dt, det)
        else:
            dt = det
        res = {"Gamma": gamma, "m_star": m_star, "Fy_star": Fy, "dy_star": dy, "T_star": T, "Se": Se / G,
               "d_et_star": det, "d_t_star": dt, "target_displacement": gamma * dt}
        if not iterate or abs(dt - dm) <= 1e-4 * dt or dt > d[-1]:
            break
        dm = min(dt, float(d[-1]))
    return res
