"""Ground motion records."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

G = 9.80665


@dataclass
class GroundMotionRecord:
    """Acceleration history in units of g with a constant time step."""

    dt: float
    acc_g: np.ndarray
    name: str = ""
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.acc_g)) * self.dt

    @property
    def duration(self) -> float:
        return (len(self.acc_g) - 1) * self.dt

    def scaled(self, factor: float) -> "GroundMotionRecord":
        return GroundMotionRecord(self.dt, self.acc_g * factor, self.name, dict(self.meta))

    # ---------------------------------------------------------- processing
    def baseline_corrected(self, order: int = 2) -> "GroundMotionRecord":
        """Remove a polynomial trend from the displacement (least squares fit of
        order ``order`` applied to the acceleration, Boore 2005 style)."""
        t = self.times
        a = self.acc_g * G
        v = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * self.dt)])
        d = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * self.dt)])
        # fit d(t) = sum c_k t^(k+2), k = 0..order-2 (keeps zero initial conditions)
        A = np.stack([t ** (k + 2) for k in range(max(order - 1, 1))], axis=1)
        c, *_ = np.linalg.lstsq(A, d, rcond=None)
        corr = sum(c[k] * (k + 2) * (k + 1) * t ** k for k in range(len(c)))
        return GroundMotionRecord(self.dt, (a - corr) / G, self.name, dict(self.meta))

    # ---------------------------------------------------------- measures
    def intensity_measures(self) -> Dict[str, float]:
        """PGA (g), PGV (m/s), PGD (m), Arias intensity (m/s), significant duration
        D5-95 (s) and cumulative absolute velocity CAV (m/s)."""
        a = self.acc_g * G
        dt = self.dt
        v = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * dt)])
        d = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
        ia_c = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] ** 2 + a[:-1] ** 2) * dt)]) * math.pi / (2 * G)
        ia = float(ia_c[-1])
        if ia > 0:
            t5 = float(np.interp(0.05 * ia, ia_c, self.times))
            t95 = float(np.interp(0.95 * ia, ia_c, self.times))
        else:
            t5 = t95 = 0.0
        return {"PGA": float(np.abs(self.acc_g).max()), "PGV": float(np.abs(v).max()), "PGD": float(np.abs(d).max()),
                "AriasIntensity": ia, "D5_95": t95 - t5, "CAV": float(np.sum(np.abs(a)) * dt)}


def read_peer_at2(path: str) -> GroundMotionRecord:
    """PEER NGA ``.AT2`` file (acceleration in g, NPTS / DT header line)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    header = lines[:4]
    npts, dt = None, None
    for ln in header:
        mn = re.search(r"NPTS\s*=\s*([0-9]+)", ln, re.I)
        md = re.search(r"DT\s*=\s*([0-9.Ee+-]+)", ln, re.I)
        if mn and md:
            npts, dt = int(mn.group(1)), float(md.group(1))
    if npts is None:  # old format: "NPTS DT" numbers on line 4
        nums = re.findall(r"[0-9.Ee+-]+", lines[3])
        npts, dt = int(float(nums[0])), float(nums[1])
    vals = []
    for ln in lines[4:]:
        vals.extend(float(x) for x in ln.replace(",", " ").split())
    acc = np.array(vals[:npts])
    return GroundMotionRecord(dt, acc, name=lines[1].strip() if len(lines) > 1 else path,
                              meta={"source": path, "header": "\n".join(header)})


def read_record(path: str, dt: Optional[float] = None, units: str = "g", skip: int = 0) -> GroundMotionRecord:
    """Plain text record: two columns (time, acceleration) or one column with ``dt``.
    ``units``: g | m/s2 | cm/s2 | gal."""
    if path.lower().endswith(".at2"):
        return read_peer_at2(path)
    data = np.loadtxt(path, skiprows=skip, ndmin=2)
    if data.shape[1] >= 2 and dt is None:
        t, a = data[:, 0], data[:, 1]
        dts = np.diff(t)
        dt = float(np.median(dts))
        if np.max(np.abs(dts - dt)) > 1e-6 * dt:  # resample to a constant step
            tt = np.arange(t[0], t[-1] + 0.5 * dt, dt)
            a = np.interp(tt, t, a)
    else:
        if dt is None:
            raise ValueError("single column record: give dt")
        a = data.ravel()
    f = {"g": 1.0, "m/s2": 1 / G, "cm/s2": 0.01 / G, "gal": 0.01 / G}[units.lower()]
    return GroundMotionRecord(float(dt), a * f, name=path, meta={"source": path})
