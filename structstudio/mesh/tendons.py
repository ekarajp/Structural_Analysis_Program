"""Prestressing tendons: path geometry, friction losses and the action of a
tendon on the frames it is bonded to (SAP2000 "Tendon" objects).

Two models, as in SAP2000:

* ``Loads``: the tendon is replaced by the loads it applies to the concrete.  With
  the tendon force P(s) and unit tangent t(s), the tendon pushes on the member with
  +P t at the start anchor, -P t at the end anchor and d(P t)/ds along its length
  (curvature and friction).  Applied at the tendon (eccentric to the member axis),
  so each force also carries the moment (r x F) about the axis.
* ``Elements``: the tendon is an axial member, bonded to the frame at its
  discretisation points by rigid links, prestressed by an imposed shortening
  strain so that it carries P(s) when fully restrained; elastic shortening then
  follows from the analysis.

Friction: P(s) = P_jack exp(-(mu alpha(s) + K s)) from the jacking end(s);
uniform stress losses (elastic shortening, creep, shrinkage, relaxation) are
subtracted as stress x area.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

_PARABOLA = ("parabolaintermediate", "parabolaend")


def path_pieces(points: Sequence[Tuple[str, np.ndarray]]) -> List[Tuple[str, List[np.ndarray]]]:
    """Group the path points into pieces [("line", [a, b]) | ("parabola", [a, m, b])]."""
    out = []
    cur = None
    k = 0
    while k < len(points):
        kind, x = points[k]
        kind = kind.lower().replace(" ", "")
        if cur is None or kind.startswith("start"):
            cur = x
            k += 1
            continue
        if kind.startswith("parabolaintermediate") and k + 1 < len(points):
            end = points[k + 1][1]
            out.append(("parabola", [cur, x, end]))
            cur = end
            k += 2
            continue
        out.append(("line", [cur, x]))
        cur = x
        k += 1
    return out


def sample_path(points, max_chord: float, min_per_piece: int = 1) -> np.ndarray:
    """Points along the path with chords no longer than ``max_chord``."""
    pts = [np.asarray(points[0][1] if points else np.zeros(3), float)]
    for kind, p in path_pieces([(k, np.asarray(x, float)) for k, x in points]):
        if kind == "line":
            a, b = p
            n = max(min_per_piece, int(math.ceil(np.linalg.norm(b - a) / max_chord)))
            for i in range(1, n + 1):
                pts.append(a + (b - a) * i / n)
        else:
            a, m, b = p
            chord = b - a
            c2 = float(chord @ chord)
            tm = float((m - a) @ chord) / c2 if c2 > 0 else 0.5
            tm = min(max(tm, 1e-6), 1 - 1e-6)
            # quadratic through (0, a), (tm, m), (1, b): x(t) = a + c1 t + c2 t^2
            # with c1 + c2 = b - a and c1 tm + c2 tm^2 = m - a
            q2 = ((m - a) - (b - a) * tm) / (tm * tm - tm)
            q1 = (b - a) - q2
            length = sum(np.linalg.norm(q1 + 2 * q2 * t) for t in np.linspace(0, 1, 33)) / 33
            n = max(min_per_piece, int(math.ceil(length / max_chord)))
            for i in range(1, n + 1):
                t = i / n
                pts.append(a + q1 * t + q2 * t * t)
    return np.array(pts)


def path_length(points) -> float:
    X = sample_path(points, 1e9, 200)
    return float(np.sum(np.linalg.norm(np.diff(X, axis=0), axis=1)))


def chord_forces(X: np.ndarray, force: float, jack_from: str, mu: float, wobble: float,
                 loss: float) -> np.ndarray:
    """Tendon force in each chord (between X[k] and X[k+1]) after friction and
    uniform losses (``loss`` = force)."""
    d = np.diff(X, axis=0)
    ln = np.linalg.norm(d, axis=1)
    t = d / ln[:, None]
    # cumulative length / angle at chord mid points, measured from end I
    turn = np.zeros(len(t))
    for k in range(1, len(t)):
        c = float(np.clip(t[k - 1] @ t[k], -1.0, 1.0))
        turn[k] = math.acos(c)
    s_mid = np.cumsum(ln) - 0.5 * ln
    a_mid = np.cumsum(turn)  # change of angle up to the start of each chord (applied at the kink)
    s_tot, a_tot = float(ln.sum()), float(turn.sum())
    pi = force * np.exp(-(mu * a_mid + wobble * s_mid))
    a_rev = a_tot - a_mid + turn  # angle change from end J back to the chord start kink
    pj = force * np.exp(-(mu * (a_rev - turn) + wobble * (s_tot - s_mid)))
    j = jack_from.strip().lower()
    if j.startswith("both"):
        p = np.maximum(pi, pj)
    elif j.startswith("j"):
        p = pj
    else:
        p = pi
    return np.maximum(p - loss, 0.0)


def node_forces(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Forces the tendon applies to the concrete at the path points (sum = 0)."""
    d = np.diff(X, axis=0)
    t = d / np.linalg.norm(d, axis=1)[:, None]
    Pt = P[:, None] * t
    F = np.zeros_like(X)
    F[0] += Pt[0]
    F[-1] -= Pt[-1]
    F[1:-1] += Pt[1:] - Pt[:-1]
    return F
