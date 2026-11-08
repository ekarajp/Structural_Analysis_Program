"""Elastic response spectra."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..engine.nonlinear.runner import _piecewise_exact
from .records import G, GroundMotionRecord


def response_spectrum(record: GroundMotionRecord, periods: Sequence[float], damping: float = 0.05,
                      substeps: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Elastic SDOF spectra of a record (exact for piecewise linear excitation,
    Nigam & Jennings 1969).  Returns periods, Sd (m), Sv (m/s, relative),
    PSv (m/s), Sa (g, absolute acceleration) and PSa (g)."""
    T = np.asarray(periods, float)
    w = 2 * np.pi / np.maximum(T, 1e-6)
    a = record.acc_g * G
    dt = record.dt
    n = substeps or max(1, int(np.ceil(10 * dt / max(T.min(), 1e-3))))
    if n > 1:  # the exact solution only needs dt << T for the peak sampling
        t = record.times
        tt = np.linspace(0, t[-1], (len(t) - 1) * n + 1)
        a = np.interp(tt, t, a)
        dt = dt / n
    P = -np.outer(a, np.ones(len(T)))
    z = np.full(len(T), damping)
    Q = _piecewise_exact(w, z, P, dt)
    V = np.gradient(Q, dt, axis=0)
    acc_abs = -(2 * z * w * V + w * w * Q)  # absolute acceleration = -(c v + k u)/m
    Sd = np.abs(Q).max(0)
    Sv = np.abs(V).max(0)
    Sa = np.abs(acc_abs).max(0) / G
    return {"T": T, "Sd": Sd, "Sv": Sv, "PSv": w * Sd, "Sa": Sa, "PSa": w * w * Sd / G}


def scale_to_target(record: GroundMotionRecord, target_T: Sequence[float], target_Sa: Sequence[float],
                    T_range: Optional[tuple] = None, damping: float = 0.05) -> float:
    """Amplitude scale factor minimising the log misfit to a target spectrum
    over ``T_range`` (ASCE 7-16 §16.2 style period range)."""
    T = np.asarray(target_T, float)
    Sa_t = np.asarray(target_Sa, float)
    if T_range:
        m = (T >= T_range[0]) & (T <= T_range[1])
        T, Sa_t = T[m], Sa_t[m]
    sp = response_spectrum(record, T, damping)
    return float(np.exp(np.mean(np.log(Sa_t / sp["PSa"]))))
