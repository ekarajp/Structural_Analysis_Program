"""Planar polygon helpers shared by the mesher and the Edit > Edit Shells commands.

A planar 3-D polygon is mapped to 2-D (u, v) coordinates in its own plane
(:func:`plane_frame`), worked on there, and mapped back (:func:`to_3d`).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

Frame = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # origin, e1, e2, normal


def newell_normal(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    if len(P) < 3:
        raise ValueError("degenerate polygon")
    Q = np.roll(P, -1, axis=0)  # one vectorised Newell sum (called for every shell of a scene)
    n = np.array([np.sum((P[:, 1] - Q[:, 1]) * (P[:, 2] + Q[:, 2])),
                  np.sum((P[:, 2] - Q[:, 2]) * (P[:, 0] + Q[:, 0])),
                  np.sum((P[:, 0] - Q[:, 0]) * (P[:, 1] + Q[:, 1]))])
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        raise ValueError("degenerate polygon")
    return n / ln


def plane_frame(P: np.ndarray, normal: Optional[np.ndarray] = None,
                e1_hint: Optional[np.ndarray] = None) -> Frame:
    """(origin, e1, e2, n) of a planar polygon; e1 along ``e1_hint`` or the first
    non-zero edge, n by the point order (Newell) unless ``normal`` is given."""
    P = np.asarray(P, float)
    n = newell_normal(P) if normal is None else np.asarray(normal, float) / np.linalg.norm(normal)
    e1 = None
    if e1_hint is not None:
        c = np.asarray(e1_hint, float)
        c = c - n * (c @ n)
        if np.linalg.norm(c) > 1e-9:
            e1 = c / np.linalg.norm(c)
    if e1 is None:  # the first edge that is not along the normal (all edges in one go)
        E = np.roll(P, -1, axis=0) - P
        E = E - np.outer(E @ n, n)
        L = np.linalg.norm(E, axis=1)
        k = np.flatnonzero(L > 1e-9)
        if len(k):
            e1 = E[k[0]] / L[k[0]]
    if e1 is None:
        raise ValueError("degenerate polygon")
    return P[0].copy(), e1, np.cross(n, e1), n


def to_2d(P: np.ndarray, fr: Frame) -> np.ndarray:
    o, e1, e2, _ = fr
    D = np.asarray(P, float) - o
    return np.column_stack([D @ e1, D @ e2])


def to_3d(uv, fr: Frame) -> np.ndarray:
    o, e1, e2, _ = fr
    uv = np.atleast_2d(np.asarray(uv, float))
    return o + np.outer(uv[:, 0], e1) + np.outer(uv[:, 1], e2)


def off_plane(P: np.ndarray, fr: Frame) -> float:
    """Largest distance of the points from the frame's plane."""
    o, _, _, n = fr
    return float(np.max(np.abs((np.asarray(P, float) - o) @ n))) if len(P) else 0.0


def signed_area(Q: np.ndarray) -> float:
    x, y = Q[:, 0], Q[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_convex(Q: np.ndarray, tol: float = 1e-9) -> bool:
    """True for a convex polygon (collinear vertices allowed)."""
    n = len(Q)
    sign = 0
    for i in range(n):
        a, b, c = Q[i - 1], Q[i], Q[(i + 1) % n]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        scale = max(np.linalg.norm(b - a) * np.linalg.norm(c - b), 1e-30)
        if abs(cr) <= tol * scale:
            continue
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def axis_aligned(Q: np.ndarray, tol: float) -> bool:
    """Every edge parallel to u or v (an orthogonal / rectilinear polygon)."""
    for i in range(len(Q)):
        d = Q[(i + 1) % len(Q)] - Q[i]
        if abs(d[0]) > tol and abs(d[1]) > tol:
            return False
    return True


def inside(p, Q: np.ndarray, edge_tol: float = 0.0) -> bool:
    """Point in polygon; within ``edge_tol`` of an edge counts as inside when > 0."""
    x, y = float(p[0]), float(p[1])
    n = len(Q)
    if edge_tol > 0:
        for i in range(n):
            a, b = Q[i], Q[(i + 1) % n]
            e = b - a
            L2 = float(e @ e)
            t = 0.0 if L2 <= 0 else min(max(((x - a[0]) * e[0] + (y - a[1]) * e[1]) / L2, 0.0), 1.0)
            if math.hypot(a[0] + t * e[0] - x, a[1] + t * e[1] - y) <= edge_tol:
                return True
    res = False
    for i in range(n):
        x1, y1 = Q[i]
        x2, y2 = Q[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                res = not res
    return res


def _levels(values: Sequence[float], tol: float) -> List[float]:
    out: List[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(float(v))
    return out


def grid_cells(outer: np.ndarray, holes: Sequence[np.ndarray] = (), tol: float = 1e-6):
    """Split a rectilinear polygon (minus rectilinear holes) into the rectangles of
    the grid through all their vertices.

    Returns ``(U, V, cells)``: the grid levels and ``(k, l)`` index pairs of the
    kept cells (cell k, l spans U[k]..U[k+1] x V[l]..V[l+1]).  Neighbouring cells
    share whole edges, so meshing each cell with per-interval divisions gives a
    conforming mesh.
    """
    pts = [outer] + list(holes)
    U = _levels([q[0] for P in pts for q in P], tol)
    V = _levels([q[1] for P in pts for q in P], tol)
    cells = []
    for k in range(len(U) - 1):
        for l in range(len(V) - 1):
            c = ((U[k] + U[k + 1]) / 2, (V[l] + V[l + 1]) / 2)
            if inside(c, outer) and not any(inside(c, h) for h in holes):
                cells.append((k, l))
    return U, V, cells


def cell_corners(U, V, k: int, l: int) -> np.ndarray:
    """Counter-clockwise corners of grid cell (k, l)."""
    return np.array([(U[k], V[l]), (U[k + 1], V[l]), (U[k + 1], V[l + 1]), (U[k], V[l + 1])], float)


def segments_cross(a, b, c, d, tol: float) -> bool:
    """Proper crossing of segments a-b and c-d (touching ends do not count)."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    s = tol * max(np.linalg.norm(np.subtract(b, a)), np.linalg.norm(np.subtract(d, c)), 1e-30)
    return ((o1 > s and o2 < -s) or (o1 < -s and o2 > s)) and ((o3 > s and o4 < -s) or (o3 < -s and o4 > s))


def self_intersects(Q: np.ndarray, tol: float) -> bool:
    n = len(Q)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if segments_cross(Q[i], Q[(i + 1) % n], Q[j], Q[(j + 1) % n], tol):
                return True
    return False


def offset_polygon(Q: np.ndarray, d: float) -> np.ndarray:
    """Move every edge of a counter-clockwise polygon outward by ``d`` (< 0: inward);
    each vertex goes to the intersection of its two offset edges."""
    n = len(Q)
    dirs = [Q[(i + 1) % n] - Q[i] for i in range(n)]
    dirs = [v / np.linalg.norm(v) for v in dirs]
    normals = [np.array([v[1], -v[0]]) for v in dirs]  # outward for CCW
    out = []
    for i in range(n):
        d0, d1 = dirs[i - 1], dirs[i]
        p0 = Q[i] + normals[i - 1] * d
        p1 = Q[i] + normals[i] * d
        cr = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(cr) < 1e-9:  # collinear edges: straight out
            out.append(p1)
            continue
        t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / cr
        out.append(p0 + t * d0)
    return np.array(out)


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: polygon ``subject`` (any simple polygon) cut to the
    convex polygon ``clip`` (both 2D; ``clip`` either orientation).  Returns the
    piece of ``subject`` inside ``clip`` (counter-clockwise, possibly empty)."""
    C = np.asarray(clip, float)
    if signed_area(C) < 0:
        C = C[::-1]
    out = [np.asarray(p, float) for p in subject]
    n = len(C)
    for k in range(n):
        a, b = C[k], C[(k + 1) % n]

        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        src, out = out, []
        for i in range(len(src)):
            P, R = src[i - 1], src[i]
            sp, sr = side(P), side(R)
            if sr >= -1e-12:
                if sp < -1e-12:
                    out.append(P + (R - P) * (sp / (sp - sr)))
                out.append(R)
            elif sp >= -1e-12:
                out.append(P + (R - P) * (sp / (sp - sr)))
        if not out:
            return np.zeros((0, 2))
    res = []
    for p in out:  # drop repeated vertices
        if not res or np.linalg.norm(p - res[-1]) > 1e-12:
            res.append(p)
    if len(res) > 1 and np.linalg.norm(res[0] - res[-1]) <= 1e-12:
        res.pop()
    R = np.array(res, float).reshape(-1, 2)
    if len(R) >= 3 and signed_area(R) < 0:
        R = R[::-1]
    return R


def triangulate(Q: np.ndarray) -> List[Tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon (any orientation)."""
    n = len(Q)
    if n < 3:
        return []
    idx = list(range(n)) if signed_area(Q) > 0 else list(range(n))[::-1]
    tris: List[Tuple[int, int, int]] = []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    guard = 0
    while len(idx) > 3 and guard < 10 * n * n:
        guard += 1
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = Q[i0], Q[i1], Q[i2]
            if cross(a, b, c) <= 1e-12:
                continue  # reflex or flat corner
            if any(j not in (i0, i1, i2) and cross(a, b, Q[j]) >= 0 and cross(b, c, Q[j]) >= 0
                   and cross(c, a, Q[j]) >= 0 for j in idx):
                continue  # another vertex inside the ear
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break  # degenerate input: stop with what we have
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris
