"""Hinge backbones: user (CSI style) and automatic ASCE 41 hinges, and
moment-curvature analysis of fiber sections.

The automatic hinges follow the modelling parameters a, b, c and the
acceptance criteria tabulated in ASCE/SEI 41 (Table 10-7 for RC beams,
condition i, flexure controlled; Table 9-7.1 for steel beams in flexure).
The tables are reproduced for the common cases only and interpolated
linearly, as ASCE 41 permits; the values should be checked against the
edition the project is designed to before relying on them.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...model.base import ModelError
from .sections import FiberSection3D

# ASCE 41-13/-17 Table 10-7, RC beams, condition i (flexure):
# ((rho - rho')/rho_bal, conforming, V/(bw d sqrt(f'c [psi]))) -> (a, b, c, IO, LS, CP)
_RC_BEAM = {
    (0.0, True, 3.0): (0.025, 0.05, 0.2, 0.010, 0.025, 0.05),
    (0.0, True, 6.0): (0.020, 0.04, 0.2, 0.005, 0.020, 0.04),
    (0.5, True, 3.0): (0.020, 0.03, 0.2, 0.005, 0.020, 0.03),
    (0.5, True, 6.0): (0.015, 0.02, 0.2, 0.005, 0.015, 0.02),
    (0.0, False, 3.0): (0.020, 0.03, 0.2, 0.005, 0.020, 0.03),
    (0.0, False, 6.0): (0.010, 0.015, 0.2, 0.0015, 0.010, 0.015),
    (0.5, False, 3.0): (0.010, 0.015, 0.2, 0.005, 0.010, 0.015),
    (0.5, False, 6.0): (0.005, 0.010, 0.2, 0.0015, 0.005, 0.010),
}


def asce41_rc_beam(rho_ratio: float, conforming: bool, shear_index: float) -> Tuple[float, ...]:
    """(a, b, c, IO, LS, CP) for an RC beam; ``rho_ratio`` = (rho - rho')/rho_bal,
    ``shear_index`` = V / (bw d sqrt(f'c)) with f'c in psi (ASCE 41 units)."""
    r = min(max(rho_ratio, 0.0), 0.5)
    v = min(max(shear_index, 3.0), 6.0)
    tr = (r - 0.0) / 0.5
    tv = (v - 3.0) / 3.0
    vals = []
    for k in range(6):
        p00 = _RC_BEAM[(0.0, conforming, 3.0)][k]
        p01 = _RC_BEAM[(0.0, conforming, 6.0)][k]
        p10 = _RC_BEAM[(0.5, conforming, 3.0)][k]
        p11 = _RC_BEAM[(0.5, conforming, 6.0)][k]
        vals.append((1 - tr) * ((1 - tv) * p00 + tv * p01) + tr * ((1 - tv) * p10 + tv * p11))
    return tuple(vals)


def asce41_steel_beam(theta_y: float, compactness: float = 0.0) -> Tuple[float, ...]:
    """(a, b, c, IO, LS, CP) for a steel beam in flexure (ASCE 41 Table 9-7.1).
    ``compactness`` 0 = seismically compact (a = 9 theta_y ...), 1 = non-compact
    (a = 4 theta_y ...), interpolated in between."""
    t = min(max(compactness, 0.0), 1.0)
    comp = (9.0, 11.0, 0.6, 1.0, 9.0, 11.0)
    nonc = (4.0, 6.0, 0.2, 0.25, 3.0, 4.0)
    v = [(1 - t) * x + t * y for x, y in zip(comp, nonc)]
    return (v[0] * theta_y, v[1] * theta_y, v[2], v[3] * theta_y, v[4] * theta_y, v[5] * theta_y)


def i_plastic_modulus(d: float, bf: float, tf: float, tw: float) -> float:
    return bf * tf * (d - tf) + tw * (d - 2 * tf) ** 2 / 4.0


def csi_points_from_abc(My: float, a: float, b: float, c: float, hardening: float, theta_y: float = 0.0):
    """CSI backbone B C D E (plastic deformation, force) from ASCE 41 a, b, c.
    ``hardening``: force increase B -> C as a fraction of My."""
    return [(0.0, My), (a, My * (1.0 + hardening)), (a, c * My), (b, c * My)]


# ----------------------------------------------------------------------------
def moment_curvature(section: FiberSection3D, axial: float = 0.0, axis: str = "z", kappa_max: float = 0.1,
                     n: int = 200, eps_cu: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Moment-curvature of a fiber section under constant axial force (tension +).

    ``axis`` "z": bending Mz (CSI M3), "y": My (CSI -M2).  Stops at
    ``kappa_max`` or when the extreme concrete fiber strain reaches ``eps_cu``
    (compression, positive number).  Returns curvature, moment, axial strain."""
    sec = section.copy()
    sec.reset()
    j = 1 if axis == "z" else 2
    e = np.zeros(4)
    ks, M, E0 = [0.0], [0.0], [0.0]
    for kap in np.linspace(0.0, kappa_max, n + 1)[1:]:
        e[j] = kap
        for _ in range(50):
            s, k = sec.set_trial(e)
            r = axial - s[0]
            if abs(r) <= 1e-9 * (abs(axial) + abs(s[0]) + 1.0) + 1e-6:
                break
            e[0] += r / k[0, 0] if k[0, 0] else 0.0
        sec.commit()
        ks.append(kap)
        M.append(s[j])
        E0.append(e[0])
        if eps_cu is not None:
            emin = min(float(np.min(g[3])) for g in sec.fiber_states() if g[0].startswith("Concrete")) \
                if any(g[0].startswith("Concrete") for g in sec.fiber_states()) else 0.0
            if emin <= -abs(eps_cu):
                break
    return {"curvature": np.array(ks), "moment": np.array(M), "axial_strain": np.array(E0)}


def yield_and_ultimate(mc: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Bilinear idealisation (equal area up to the peak) of a moment-curvature curve."""
    k, m = mc["curvature"], np.abs(mc["moment"])
    ip = int(np.argmax(m))
    Mu, ku = float(m[ip]), float(k[ip])
    # equal energy bilinear: initial slope from the first points
    K0 = m[1] / k[1] if k[1] > 0 else 0.0
    area = float(np.trapezoid(m[: ip + 1], k[: ip + 1])) if hasattr(np, "trapezoid") else float(np.trapz(m[: ip + 1], k[: ip + 1]))
    # area under the bilinear curve: My ku - My^2 / (2 K0) = area -> smaller root
    disc = ku * ku - 2.0 * area / K0 if K0 > 0 else -1.0
    My = K0 * (ku - math.sqrt(disc)) if disc >= 0 else Mu
    if My <= 0 or My > Mu or not np.isfinite(My):
        My = Mu
    return {"My": My, "phi_y": My / K0 if K0 else 0.0, "Mu": Mu, "phi_u": ku, "EI_eff": K0}
