"""Small geometry helpers used by the mesher and the engine."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

EZ = np.array([0.0, 0.0, 1.0])
EX = np.array([1.0, 0.0, 0.0])
VERTICAL_SINE = 1.0e-3  # CSI: element is vertical if sin(angle to Z) < 1e-3


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("zero-length vector")
    return v / n


def frame_axes(pi: np.ndarray, pj: np.ndarray, angle_deg: float = 0.0,
               reference: Optional[np.ndarray] = None) -> np.ndarray:
    """Local axes of a line element following the CSI default orientation.

    * axis 1 from I to J;
    * the 1-2 plane is vertical with axis 2 pointing upward (+Z sense);
    * for vertical members axis 2 is +X (and 3 = +Y for a member pointing up);
    * ``angle_deg`` rotates axes 2 and 3 about axis 1 (counter-clockwise when
      +1 points toward the viewer).

    Returns a (3, 3) array whose rows are the unit vectors of axes 1, 2, 3.
    """
    v1 = unit(np.asarray(pj, float) - np.asarray(pi, float))
    if reference is not None:
        ref = unit(np.asarray(reference, float))
        v3 = np.cross(v1, ref)
        if np.linalg.norm(v3) < 1e-9:
            raise ValueError("reference vector parallel to the element")
        v3 = unit(v3)
        v2 = np.cross(v3, v1)
    else:
        sin_to_z = np.linalg.norm(np.cross(v1, EZ))
        if sin_to_z < VERTICAL_SINE:
            v2 = EX.copy()
            v3 = unit(np.cross(v1, v2))
            v2 = np.cross(v3, v1)
        else:
            v3 = unit(np.cross(v1, EZ))
            v2 = np.cross(v3, v1)
    if angle_deg:
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        v2, v3 = c * v2 + s * v3, -s * v2 + c * v3
    return np.vstack([v1, v2, v3])


def transformed_axes(nominal: np.ndarray, new_v1: np.ndarray) -> np.ndarray:
    """Re-derive local axes when the neutral axis direction changes due to
    insertion points (CSI: v3~ = v1~ x v2, v2~ = v3~ x v1~)."""
    v1 = unit(new_v1)
    if np.allclose(v1, nominal[0], atol=1e-12):
        return nominal
    v3 = np.cross(v1, nominal[1])
    if np.linalg.norm(v3) < 1e-9:
        v3 = np.cross(v1, nominal[2])
        v3 = unit(v3)
        v2 = np.cross(v3, v1)
        return np.vstack([v1, v2, v3])
    v3 = unit(v3)
    v2 = np.cross(v3, v1)
    return np.vstack([v1, v2, v3])



def cross3(a, b) -> np.ndarray:
    """Cross product of two 3-vectors: the same arithmetic as ``np.cross``
    (bit-identical results) without its per-call overhead."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])

def point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> Optional[float]:
    """Return the parameter t (0..1) if ``p`` lies on segment a-b (strictly
    inside, away from the ends), otherwise ``None``."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0.0:
        return None
    t = float((p - a) @ ab) / L2
    L = math.sqrt(L2)
    if t * L <= tol or (1.0 - t) * L <= tol:
        return None
    closest = a + t * ab
    if np.linalg.norm(p - closest) <= tol:
        return t
    return None


def segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
                         tol: float) -> Optional[Tuple[float, float]]:
    """Parameters (t, s) where segments a-b and c-d intersect in 3D (within
    ``tol``), strictly inside both segments; ``None`` otherwise."""
    u = b - a
    v = d - c
    w = a - c
    uu, uv, vv = u @ u, u @ v, v @ v
    uw, vw = u @ w, v @ w
    den = uu * vv - uv * uv
    if abs(den) <= 1e-12 * max(uu * vv, 1e-30):
        return None  # parallel
    t = (uv * vw - vv * uw) / den
    s = (uu * vw - uv * uw) / den
    Lu, Lv = math.sqrt(uu), math.sqrt(vv)
    if t * Lu <= tol or (1 - t) * Lu <= tol or s * Lv <= tol or (1 - s) * Lv <= tol:
        return None
    gap = np.linalg.norm((a + t * u) - (c + s * v))
    if gap > tol:
        return None
    return t, s


def polygon_normal(pts: np.ndarray) -> np.ndarray:
    """Newell's method normal of a planar polygon (rows = points)."""
    n = np.zeros(3)
    k = len(pts)
    for i in range(k):
        p, q = pts[i], pts[(i + 1) % k]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return unit(n)


def polygon_area(pts: np.ndarray) -> float:
    n = np.zeros(3)
    k = len(pts)
    for i in range(k):
        n += cross3(pts[i], pts[(i + 1) % k])
    return 0.5 * float(np.linalg.norm(n))


def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Area centroid of a planar polygon."""
    normal = polygon_normal(pts)
    # build in-plane 2D coordinates
    e1 = unit(pts[1] - pts[0])
    e2 = cross3(normal, e1)
    origin = pts[0]
    uv = np.array([[(p - origin) @ e1, (p - origin) @ e2] for p in pts])
    a = 0.0
    cx = cy = 0.0
    k = len(uv)
    for i in range(k):
        x0, y0 = uv[i]
        x1, y1 = uv[(i + 1) % k]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-30:
        return pts.mean(axis=0)
    a *= 0.5
    cx /= 6.0 * a
    cy /= 6.0 * a
    return origin + cx * e1 + cy * e2


def shell_axes(pts: np.ndarray, angle_deg: float = 0.0) -> np.ndarray:
    """Default local axes for an area element (CSI convention).

    Axis 3 is the outward normal from the node ordering.  Axis 1 is the
    projection of global +X onto the element plane unless the element is
    vertical... CSI rule: local 1 is horizontal (in the X-Y plane) and the 1-2
    plane contains the normal; for horizontal elements local 1 = +X.
    """
    v3 = polygon_normal(pts)
    if abs(v3 @ EZ) > 1.0 - 1e-6:  # horizontal element
        v1 = EX.copy()
        v2 = cross3(v3, v1)
    else:
        v2 = unit(cross3(v3, cross3(EZ, v3)))  # in plane, projection of Z
        if v2 @ EZ < 0:
            v2 = -v2
        v1 = cross3(v2, v3)
    if angle_deg:
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        v1, v2 = c * v1 + s * v2, -s * v1 + c * v2
    return np.vstack([v1, v2, v3])


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def polygon_polar_inertia(pts: np.ndarray, about: np.ndarray) -> float:
    """Polar second moment of area (per unit thickness) of a planar polygon
    about the axis normal to its plane through point ``about``."""
    normal = polygon_normal(pts)
    e1 = unit(pts[1] - pts[0])
    e2 = cross3(normal, e1)
    uv = np.array([[(p - about) @ e1, (p - about) @ e2] for p in pts])
    k = len(uv)
    ixx = iyy = 0.0
    for i in range(k):
        x0, y0 = uv[i]
        x1, y1 = uv[(i + 1) % k]
        cross = x0 * y1 - x1 * y0
        ixx += (y0 * y0 + y0 * y1 + y1 * y1) * cross
        iyy += (x0 * x0 + x0 * x1 + x1 * x1) * cross
    return abs(ixx + iyy) / 12.0


def quad_tributary_areas(quad: np.ndarray) -> np.ndarray:
    """Consistent (shape-function integrated) tributary areas of the four
    nodes of a planar quadrilateral."""
    normal = polygon_normal(quad)
    e1 = unit(quad[1] - quad[0])
    e2 = cross3(normal, e1)
    xy = np.array([[(p - quad[0]) @ e1, (p - quad[0]) @ e2] for p in quad])
    g = 1.0 / math.sqrt(3.0)
    trib = np.zeros(4)
    for xi in (-g, g):
        for eta in (-g, g):
            N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
            dN = 0.25 * np.array([[-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)], [-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]])
            det = abs(np.linalg.det(dN @ xy))
            trib += N * det
    return trib


def joint_axes(a_deg: float = 0.0, b_deg: float = 0.0, c_deg: float = 0.0) -> np.ndarray:
    """Joint local axes from the CSI joint angles (rows = local 1, 2, 3 in global):
    starting from the global axes, rotate by ``a`` about +3, then by ``b`` about the
    resulting +2, then by ``c`` about the resulting +1 (right-hand rule)."""
    e1, e2, e3 = np.eye(3)
    a, b, c = (math.radians(v or 0.0) for v in (a_deg, b_deg, c_deg))
    if a:
        e1, e2 = math.cos(a) * e1 + math.sin(a) * e2, -math.sin(a) * e1 + math.cos(a) * e2
    if b:
        e3, e1 = math.cos(b) * e3 + math.sin(b) * e1, -math.sin(b) * e3 + math.cos(b) * e1
    if c:
        e2, e3 = math.cos(c) * e2 + math.sin(c) * e3, -math.sin(c) * e2 + math.cos(c) * e3
    return np.vstack([e1, e2, e3])


# cardinal point -> (factor of the depth t3 along local 2, factor of the width t2 along local 3)
CARDINAL_FACTORS = {
    1: (0.5, 0.5), 2: (0.5, 0.0), 3: (0.5, -0.5),
    4: (0.0, 0.5), 5: (0.0, 0.0), 6: (0.0, -0.5),
    7: (-0.5, 0.5), 8: (-0.5, 0.0), 9: (-0.5, -0.5),
    10: (0.0, 0.0), 11: (0.0, 0.0),
}


def frame_insertion_offsets(model, f, t3: Optional[float] = None, t2: Optional[float] = None):
    """Global vectors (model units) from the joints I, J of frame ``f`` to its
    element axis: the cardinal point plus the joint offsets (local or global)."""
    pts = model.points
    a = np.array(pts[f.point_i].xyz, float)
    b = np.array(pts[f.point_j].xyz, float)
    nominal = frame_axes(a, b, f.angle)
    if t3 is None or t2 is None:
        sec = model.frame_sections.get(f.section)
        t3 = float(getattr(sec, "t3", 0.0) or 0.0) if t3 is None else t3
        t2 = float(getattr(sec, "t2", 0.0) or 0.0) if t2 is None else t2
    c2, c3 = CARDINAL_FACTORS.get(f.cardinal_point, (0.0, 0.0))
    cardinal = c2 * t3 * nominal[1] + c3 * t2 * nominal[2]
    oi = np.array(f.joint_offset_i, float)
    oj = np.array(f.joint_offset_j, float)
    if getattr(f, "joint_offset_local", False):
        oi, oj = nominal.T @ oi, nominal.T @ oj
    return oi + cardinal, oj + cardinal


def beam_axis_shift(model, f, tol: float = 1e-9):
    """Horizontal shift (model units, global, Z = 0) of the analysis axis of a beam
    whose insertion offsets are part of the analysis: the beam is meshed (and loaded
    by load panels) on its true axis, joined to its joints by rigid links; the
    vertical part of the offset stays a rigid arm of the elements.  None when the
    beam is not shifted."""
    if f.type != "Beam" or not getattr(f, "offsets_in_analysis", True):
        return None
    try:
        oi, oj = frame_insertion_offsets(model, f)
    except (ValueError, KeyError):
        return None
    hi = np.array([oi[0], oi[1], 0.0])
    hj = np.array([oj[0], oj[1], 0.0])
    if max(np.linalg.norm(hi), np.linalg.norm(hj)) <= tol:
        return None
    return hi, hj


def column_containing(model, point: str, xyz, tol: float, scale: float = 1.0, xyz_of=None, grow: float = 0.0):
    """Name of a column with an end at joint ``point`` whose section rectangle, in
    plan, contains ``xyz`` at the joint's level (the joint zone of that column), or
    None.  ``grow``: the rectangle made that much larger each way (half the width of a
    beam: its axis there, the beam's body still overlaps the column - it bears on it).  ``xyz`` and ``tol`` are in model units times ``scale`` (the mesher passes
    SI with the length factor); ``xyz_of(p)`` gives a point's coordinates in those units."""
    xyz_of = xyz_of or (lambda p: np.array(p.xyz, float) * scale)
    q = np.asarray(xyz, float)
    j = xyz_of(model.points[point])
    h = q - j
    if abs(h[2]) > tol:
        return None
    for c in model.frames.values():
        if c.type != "Column" or point not in (c.point_i, c.point_j):
            continue
        sec = model.frame_sections.get(c.section)
        if sec is None:
            continue
        try:
            ax = frame_axes(xyz_of(model.points[c.point_i]), xyz_of(model.points[c.point_j]), c.angle)
        except ValueError:
            continue
        t3, t2 = (sec.t3 or 0.0) * scale, (sec.t2 or 0.0) * scale
        if abs(float(h @ ax[1])) <= 0.5 * t3 + grow + tol and abs(float(h @ ax[2])) <= 0.5 * t2 + grow + tol:
            return c.name
    return None


def member_through(model, point: str, xyz, tol: float, exclude=(), geom=None, scale: float = 1.0, xyz_of=None):
    """Name of a member (not a rigid link / null line) with an end at joint ``point``
    that passes through ``xyz`` (strictly inside it, on its line) and is meshed at the
    joints on it (so a member ending at ``xyz`` connects to it there), or None.
    ``geom(name)`` gives a member's analysis line (default: joint to joint)."""
    xyz_of = xyz_of or (lambda p: np.array(p.xyz, float) * scale)
    q = np.asarray(xyz, float)
    for g in model.frames.values():
        if g.name in exclude or g.type in ("Rigid", "Null") or point not in (g.point_i, g.point_j):
            continue
        if not (g.auto_mesh and g.mesh_at_intersections):
            continue
        a, b = geom(g.name) if geom is not None else (xyz_of(model.points[g.point_i]), xyz_of(model.points[g.point_j]))
        t = point_on_segment(q, np.asarray(a, float), np.asarray(b, float), tol)
        if t is not None:
            return g.name
    return None
