"""Incremental dynamic analysis (Vamvatsikos & Cornell 2002) and fragility fitting.

:func:`incremental_dynamic_analysis` scales each record to a list of
intensity levels, runs a nonlinear time history case of a model for each, and
collects an engineering demand parameter (peak story drift by default).
Collapse is flagged when the analysis stops (dynamic instability) or the EDP
exceeds ``collapse_edp``.

Fragility functions:

* :func:`fit_fragility_ida` - lognormal fit to the collapse (or limit-state)
  intensities of each record (method of moments in log space, Baker 2015 Eq. 3).
* :func:`fit_fragility_mle` - multiple stripe analysis: maximum likelihood fit
  of a lognormal CDF to the fraction of exceedances per intensity level
  (Baker 2015 Eq. 11).
"""
from __future__ import annotations

import copy
import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import optimize, stats

from .. import analyze
from ..model.nonlinear import TimeHistoryLoad
from .records import G, GroundMotionRecord


def peak_story_drift(res, case: str) -> float:
    rows = res.story_drift_history(case)
    return max((r["MaxDrift"] for r in rows), default=0.0)


def incremental_dynamic_analysis(model, records: Sequence[GroundMotionRecord], intensities: Sequence[float],
                                 im: str = "PGA", direction: str = "U1", gravity_case: Optional[str] = None,
                                 case_template: Optional[dict] = None, edp: Callable = peak_story_drift,
                                 collapse_edp: float = 0.10, im_function: Optional[Callable] = None,
                                 stop_after_collapse: bool = True, progress: Optional[Callable] = None) -> List[dict]:
    """Run an IDA.  ``im``: "PGA" (g) or "Sa" with ``im_function(record) -> Sa(T1)`` of
    the unscaled record; ``intensities`` in the same units.  ``case_template``:
    extra :class:`~structstudio.model.nonlinear.TimeHistoryCase` fields
    (damping, integrator, tolerance ...).  Returns one row per run."""
    U = model.units
    out = []
    for k, rec in enumerate(records):
        if im.upper() == "PGA":
            im0 = float(np.abs(rec.acc_g).max())
        else:
            if im_function is None:
                raise ValueError("im='Sa' needs im_function(record) -> Sa of the unscaled record")
            im0 = float(im_function(rec))
        collapsed = False
        for level in intensities:
            if collapsed and stop_after_collapse:
                out.append({"record": rec.name or k, "IM": level, "EDP": np.inf, "collapse": True, "scale": None})
                continue
            sf = level / im0
            m = copy.deepcopy(model)
            fn = f"__IDA_{k}"
            m.add_time_history_function(fn, rec.times, rec.acc_g)
            name = f"__IDA_{k}_{level:g}"
            kw = dict(n_steps=len(rec.acc_g) - 1, dt=rec.dt, initial_case=gravity_case)
            kw.update(case_template or {})
            m.add_time_history_case(name, [TimeHistoryLoad("Acceleration", direction, fn,
                                                           sf * G / U.factor("acceleration"))], **kw)
            cases = ([gravity_case] if gravity_case else []) + [name]
            try:
                res = analyze(m, cases)
                stopped = bool(res._case(name).history.get("failed"))
                e = float(edp(res, name))
            except Exception as exc:  # dynamic instability / divergence
                stopped, e = True, np.inf
            col = stopped or e >= collapse_edp
            collapsed = collapsed or col
            row = {"record": rec.name or k, "IM": float(level), "EDP": e if not stopped else np.inf,
                   "collapse": col, "scale": sf}
            out.append(row)
            if progress:
                progress(row)
    return out


def collapse_intensities(rows: List[dict]) -> Dict[str, float]:
    """Lowest intensity at which each record collapsed (IDA flatline)."""
    out: Dict[str, float] = {}
    for r in rows:
        if r["collapse"]:
            out[r["record"]] = min(out.get(r["record"], math.inf), r["IM"])
    return out


def fit_fragility_ida(im_values: Sequence[float]) -> Dict[str, float]:
    """Lognormal fragility from limit-state intensities of individual records."""
    x = np.log(np.asarray([v for v in im_values if np.isfinite(v) and v > 0], float))
    theta = float(np.exp(x.mean()))
    beta = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {"theta": theta, "beta": beta, "n": len(x)}


def fit_fragility_mle(im_levels: Sequence[float], n_exceed: Sequence[int], n_total: Sequence[int]) -> Dict[str, float]:
    """Maximum likelihood lognormal fragility for multiple stripe analysis."""
    im = np.asarray(im_levels, float)
    z = np.asarray(n_exceed, float)
    n = np.asarray(n_total, float)

    def nll(p):
        th, b = math.exp(p[0]), math.exp(p[1])
        P = stats.norm.cdf(np.log(im / th) / b)
        P = np.clip(P, 1e-12, 1 - 1e-12)
        return -float(np.sum(z * np.log(P) + (n - z) * np.log(1 - P)))

    frac = np.where(n > 0, z / np.maximum(n, 1), 0)
    th0 = float(np.interp(0.5, np.clip(frac, 0, 1), im)) if np.any(frac >= 0.5) else float(im.max())
    r = optimize.minimize(nll, [math.log(max(th0, 1e-6)), math.log(0.4)], method="Nelder-Mead",
                          options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
    return {"theta": math.exp(r.x[0]), "beta": math.exp(r.x[1]), "neg_log_likelihood": float(r.fun)}


def fragility_curve(theta: float, beta: float, im: Sequence[float]) -> np.ndarray:
    return stats.norm.cdf(np.log(np.asarray(im, float) / theta) / beta)
