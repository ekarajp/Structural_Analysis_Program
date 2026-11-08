"""Frame section shapes: outlines, section properties and the Section Designer.

Everything here works in the plane of the cross-section with **x = local axis 3**
(horizontal in the drawings, the section *width*) and **y = local axis 2**
(vertical, the section *depth*), the usual view of a beam section.  So

* ``I33 = integral of y^2 dA`` (bending in the 1-2 plane, about the 3 axis),
* ``I22 = integral of x^2 dA``, ``I23 = integral of x y dA``,
* ``As2`` is the shear area for shear along y (the depth), ``As3`` along x,

all about the centroid, where the frame line passes (cardinal point 10).

Two kinds of section come from here:

* **parametric shapes** (:data:`SHAPES`): concrete rectangle / circle / T-beam /
  L-beam / hollow box / hollow circle, steel I / channel / tee / angle / double
  angle / double channel / box / pipe / cold-formed lipped channel.  The outline
  follows from a few dimensions; A and I are exact (Green's theorem), J and the
  shear areas use the usual closed forms (thin-walled for steel), so the values
  are the ones a designer checks by hand.
* **Section Designer** sections: any combination of polygons and circles drawn
  by the user, some of them openings.  Overlapping or touching pieces are united
  (e.g. a T drawn as two rectangles); the region is triangulated (a conforming
  Delaunay triangulation of the outline), A and I are exact, the torsion constant
  J comes from the Saint-Venant warping function solved by finite elements and
  the shear areas from the shear-flow (Jourawski) energy integral.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

CIRCLE_SEGMENTS = 96


# ---------------------------------------------------------------- polygons
def rect(x0: float, y0: float, w: float, h: float) -> np.ndarray:
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], float)


def circle(cx: float, cy: float, r: float, n: int = CIRCLE_SEGMENTS) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, int(n), endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def signed_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ccw(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    return P if signed_area(P) >= 0 else P[::-1].copy()


def cw(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    return P if signed_area(P) <= 0 else P[::-1].copy()


def loop_moments(P: np.ndarray) -> np.ndarray:
    """(A, Sx, Sy, Ixx, Iyy, Ixy) of a closed loop about the origin, signed by its
    orientation (counter-clockwise positive): Sx = int x dA, Sy = int y dA,
    Ixx = int y^2 dA, Iyy = int x^2 dA, Ixy = int x y dA."""
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    c = x0 * y1 - x1 * y0
    A = c.sum() / 2.0
    Sx = ((x0 + x1) * c).sum() / 6.0
    Sy = ((y0 + y1) * c).sum() / 6.0
    Ixx = ((y0 * y0 + y0 * y1 + y1 * y1) * c).sum() / 12.0
    Iyy = ((x0 * x0 + x0 * x1 + x1 * x1) * c).sum() / 12.0
    Ixy = ((x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * c).sum() / 24.0
    return np.array([A, Sx, Sy, Ixx, Iyy, Ixy])


def points_in_polygon(pts: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test of many points (vectorised)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    xa, ya = P[:, 0][None, :], P[:, 1][None, :]
    xb, yb = np.roll(P[:, 0], -1)[None, :], np.roll(P[:, 1], -1)[None, :]
    straddle = (ya > y) != (yb > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = xa + (y - ya) * (xb - xa) / (yb - ya)
    hits = straddle & (x < xc)
    return (hits.sum(axis=1) % 2) == 1


def _seg_dist(pts: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance of each point to the nearest of the segments A[k]-B[k]."""
    if len(A) == 0:
        return np.full(len(pts), np.inf)
    out = np.full(len(pts), np.inf)
    d = B - A
    L2 = np.maximum((d * d).sum(axis=1), 1e-300)
    for s in range(0, len(pts), 2000):
        p = pts[s:s + 2000][:, None, :]
        t = np.clip(((p - A[None]) * d[None]).sum(axis=2) / L2[None], 0.0, 1.0)
        q = A[None] + t[..., None] * d[None]
        out[s:s + 2000] = np.sqrt(((p - q) ** 2).sum(axis=2)).min(axis=1)
    return out


def _chords(loops: Sequence[np.ndarray], levels: np.ndarray, axis: int) -> np.ndarray:
    """Width of the region cut by the lines (y = level for axis 1, x = level for
    axis 0): solids add, holes (clockwise loops) subtract."""
    out = np.zeros(len(levels))
    a, o = (1, 0) if axis == 1 else (0, 1)
    lv = levels[:, None]
    for P in loops:
        s = 1.0 if signed_area(P) >= 0 else -1.0
        ya, yb = P[:, a][None, :], np.roll(P[:, a], -1)[None, :]
        xa, xb = P[:, o][None, :], np.roll(P[:, o], -1)[None, :]
        straddle = (ya > lv) != (yb > lv)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = np.where(straddle, xa + (lv - ya) * (xb - xa) / (yb - ya), np.nan)
        xs = np.sort(xc, axis=1)
        if xs.shape[1] % 2:
            xs = np.concatenate([xs, np.full((len(levels), 1), np.nan)], axis=1)
        w = np.nan_to_num(xs[:, 1::2] - xs[:, 0::2], nan=0.0).sum(axis=1)
        out += s * w
    return np.maximum(out, 0.0)


# ---------------------------------------------------------------- properties
@dataclass
class SectionProps:
    """Section properties about the centroid (x = local 3, y = local 2)."""
    A: float = 0.0
    xc: float = 0.0  # centroid in the coordinates of the outline it was computed from
    yc: float = 0.0
    I33: float = 0.0
    I22: float = 0.0
    I23: float = 0.0
    J: float = 0.0
    As2: float = 0.0
    As3: float = 0.0
    S33: float = 0.0  # elastic moduli I / c (c = furthest fibre)
    S22: float = 0.0
    Z33: float = 0.0  # plastic moduli
    Z22: float = 0.0
    r33: float = 0.0
    r22: float = 0.0
    Imax: float = 0.0  # principal moments, angle of the major axis from the 3 axis (rad)
    Imin: float = 0.0
    angle: float = 0.0
    width: float = 0.0  # bounding box (t2 x t3)
    depth: float = 0.0
    perimeter: float = 0.0
    J_method: str = ""
    as_method: str = ""

    def as_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def region_properties(loops: Sequence[np.ndarray], strips: int = 1600) -> SectionProps:
    """Exact A, centroid, I33, I22, I23 of the oriented loops (solids
    counter-clockwise, holes clockwise; loops do not overlap), the moduli, and the
    shear areas from the shear-flow energy integral As = I^2 / int(Q^2 / b)."""
    loops = [np.asarray(P, float) for P in loops if len(P) >= 3]
    p = SectionProps()
    if not loops:
        return p
    m = sum(loop_moments(P) for P in loops)
    A = float(m[0])
    if A <= 0:
        return p
    xc, yc = m[1] / A, m[2] / A
    I33 = m[3] - A * yc * yc
    I22 = m[4] - A * xc * xc
    I23 = m[5] - A * xc * yc
    allp = np.vstack(loops)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    p.A, p.xc, p.yc = A, float(xc), float(yc)
    if abs(I23) < 1e-7 * (abs(I33) + abs(I22)):  # symmetric (round-off of the arcs)
        I23 = 0.0
    p.I33, p.I22, p.I23 = float(I33), float(I22), float(I23)
    p.width, p.depth = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    p.perimeter = float(sum(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).sum() for P in loops))
    cy = max(hi[1] - yc, yc - lo[1])
    cx = max(hi[0] - xc, xc - lo[0])
    p.S33 = p.I33 / cy if cy > 0 else 0.0
    p.S22 = p.I22 / cx if cx > 0 else 0.0
    p.r33 = math.sqrt(max(p.I33, 0.0) / A)
    p.r22 = math.sqrt(max(p.I22, 0.0) / A)
    c, d = 0.5 * (p.I33 + p.I22), math.hypot(0.5 * (p.I33 - p.I22), p.I23)
    p.Imax, p.Imin = c + d, c - d
    p.angle = 0.5 * math.atan2(-2.0 * p.I23, p.I33 - p.I22) if (abs(p.I23) > 1e-12 * max(c, 1e-300)) else 0.0
    for axis in (1, 0):  # 1: along y (As2, Z33), 0: along x (As3, Z22)
        a0, a1 = float(lo[axis]), float(hi[axis])
        e = np.linspace(a0, a1, strips + 1)
        mid = 0.5 * (e[1:] + e[:-1])
        h = (a1 - a0) / strips
        b = _chords(loops, mid, axis)
        dA = b * h
        cen = yc if axis == 1 else xc
        s = mid - cen
        mm = s * dA
        above = np.concatenate([np.cumsum(mm[::-1])[::-1][1:], [0.0]])
        Q = above + 0.5 * mm
        ok = b > 1e-12 * max(p.width, p.depth)
        integral = float((Q[ok] ** 2 / b[ok]).sum() * h)
        I = p.I33 if axis == 1 else p.I22
        As = I * I / integral if integral > 0 else 0.0
        cum = np.cumsum(dA)
        k = int(np.searchsorted(cum, 0.5 * cum[-1]))
        k = min(max(k, 0), len(mid) - 1)
        prev = cum[k - 1] if k > 0 else 0.0
        frac = (0.5 * cum[-1] - prev) / dA[k] if dA[k] > 0 else 0.5
        pna = e[k] + frac * h
        Z = float((np.abs(mid - pna) * dA).sum())
        if axis == 1:
            p.As2, p.Z33 = As, Z
        else:
            p.As3, p.Z22 = As, Z
    p.as_method = "shear-flow integral"
    return p


# ------------------------------------------------------ closed-form torsion
def j_rectangle(a: float, b: float) -> float:
    """Saint-Venant torsion constant of a solid rectangle (Roark)."""
    a, b = max(a, b), min(a, b)
    if b <= 0:
        return 0.0
    return a * b ** 3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - b ** 4 / (12.0 * a ** 4)))


def j_box(d: float, b: float, tf: float, tw: float) -> float:
    """Closed thin-walled rectangle (Bredt), the CSI box formula."""
    den = b * tw + d * tf - tw * tw - tf * tf
    return 2.0 * tf * tw * (b - tw) ** 2 * (d - tf) ** 2 / den if den > 0 else 0.0


# ---------------------------------------------------------------- catalogue
@dataclass
class Dim:
    attr: str  # FrameSection field (t3, t2, tf, tw, t2b, tfb, dis, lip)
    label: str
    default: Optional[float]  # metres (None: optional, blank = same as another)
    optional: bool = False
    placeholder: str = ""  # shown in an empty optional field


@dataclass
class Shape:
    key: str  # FrameSection.shape
    label: str
    group: str  # Concrete | Steel
    dims: List[Dim]
    outline: Callable[[dict], Tuple[List[np.ndarray], List[np.ndarray]]]  # (solids, holes)
    torsion: Callable[[dict], float]
    shear: Optional[Callable[[dict, SectionProps], Tuple[float, float]]] = None  # None: shear-flow integral
    exact: Optional[Callable[[dict], Dict[str, float]]] = None  # exact A / I for curved outlines
    mirror: bool = False  # the outline can be mirrored (flange / legs to the other side)
    material: str = "Concrete"
    note: str = ""


def _g(v: dict, k: str, default: float = 0.0) -> float:
    x = v.get(k)
    return float(default if x is None else x)


def rounded(P, radii: Sequence[float], per_quarter: int = 24) -> np.ndarray:
    """Polygon with its corners rounded: ``radii[i]`` at vertex i (0 = sharp).
    The arc is tangent to both edges, so a convex corner loses material (a
    rolled flange tip, a tube corner) and a re-entrant corner gains it (the
    fillet between web and flange)."""
    P = np.asarray(P, float)
    n = len(P)
    out = []
    for i in range(n):
        r = float(radii[i] or 0.0) if i < len(radii) else 0.0
        v = P[i]
        if r <= 0:
            out.append(v)
            continue
        a, b = P[i - 1], P[(i + 1) % n]
        u1, u2 = a - v, b - v
        u1, u2 = u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)
        th = math.acos(min(max(float(u1 @ u2), -1.0), 1.0))
        if th < 1e-6 or abs(math.pi - th) < 1e-6:
            out.append(v)
            continue
        t = r / math.tan(th / 2)
        t1 = v + u1 * t
        bis = (u1 + u2) / np.linalg.norm(u1 + u2)
        c = v + bis * r / math.sin(th / 2)
        a1 = math.atan2(*(t1 - c)[::-1])
        t2 = v + u2 * t
        a2 = math.atan2(*(t2 - c)[::-1])
        d = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        k = max(2, int(math.ceil(per_quarter * abs(d) / (math.pi / 2) - 1e-6)))
        for j in range(k + 1):
            ang = a1 + d * j / k
            out.append(c + r * np.array([math.cos(ang), math.sin(ang)]))
    return np.array(out)


def _r(v, k="r") -> float:
    x = v.get(k)
    return float(x) if x else 0.0


def _i_dims(v):
    d, bf, tf, tw = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw")
    bb = _g(v, "t2b", bf) or bf
    tfb = _g(v, "tfb", tf) or tf
    return d, bf, tf, tw, bb, tfb


def _out_rect(v):
    d, b = _g(v, "t3"), _g(v, "t2")
    return [rect(-b / 2, -d / 2, b, d)], []


def _out_circle(v):
    return [circle(0, 0, _g(v, "t3") / 2)], []


def _out_tee(v):
    d, bf, tf, tw, r = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw"), _r(v)
    P = [[-tw / 2, 0], [tw / 2, 0], [tw / 2, d - tf], [bf / 2, d - tf], [bf / 2, d],
         [-bf / 2, d], [-bf / 2, d - tf], [-tw / 2, d - tf]]
    return [rounded(P, [0, 0, r, 0, 0, 0, 0, r])], []


def _out_lbeam(v):
    d, bf, tf, tw = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw")
    return [np.array([[0, 0], [tw, 0], [tw, d - tf], [bf, d - tf], [bf, d], [0, d]], float)], []


def _out_box(v):
    d, b, tf, tw, ro = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw"), _r(v)
    ri = max(ro - max(tf, tw), 0.0)  # inside corner radius of a formed / rolled tube
    return [rounded(rect(-b / 2, -d / 2, b, d), [ro] * 4)], \
        [rounded(rect(-b / 2 + tw, -d / 2 + tf, b - 2 * tw, d - 2 * tf), [ri] * 4)]


def _out_pipe(v):
    D, t = _g(v, "t3"), _g(v, "tw")
    return [circle(0, 0, D / 2)], [circle(0, 0, D / 2 - t)]


# Inner flange slope of tapered rolled shapes (tan of the angle): 0 = parallel
# flanges.  The flange thickness given (tf) is then the mean thickness, at the
# middle of the outstand (JIS / TIS, GB and AISC define it so).
FLANGE_SLOPES = [("Parallel flanges", 0.0),
                 ("Tapered 5\u00b0 - JIS / TIS channel", math.tan(math.radians(5.0))),
                 ("Tapered 8\u00b0 - JIS / TIS I-beam", math.tan(math.radians(8.0))),
                 ("Tapered 1:10 - GB channel", 0.1),
                 ("Tapered 1:6 - GB I-beam, AISC C / S (16.7%)", 1.0 / 6.0),
                 ("Tapered 8% - EN UPN (h \u2264 300)", 0.08),
                 ("Tapered 5% - EN UPN (h > 300)", 0.05),
                 ("Tapered 14% - EN IPN", 0.14)]
SLOPED = {"I", "Channel", "DoubleChannel"}


def _slope(v) -> float:
    x = v.get("slope")
    return float(x) if x else 0.0


def flange_thickness(tf: float, outstand: float, s: float) -> Tuple[float, float]:
    """(at the web, at the tip) of a flange of mean thickness ``tf``."""
    return tf + s * outstand / 2.0, tf - s * outstand / 2.0


def _out_i(v):
    d, bf, tf, tw, bb, tfb = _i_dims(v)
    r, r2, s = _r(v), _r(v, "r2"), _slope(v)
    tw_b, tt_b = flange_thickness(tfb, (bb - tw) / 2.0, s)
    tw_t, tt_t = flange_thickness(tf, (bf - tw) / 2.0, s)
    P = [[-bb / 2, 0], [bb / 2, 0], [bb / 2, tt_b], [tw / 2, tw_b], [tw / 2, d - tw_t], [bf / 2, d - tt_t],
         [bf / 2, d], [-bf / 2, d], [-bf / 2, d - tt_t], [-tw / 2, d - tw_t], [-tw / 2, tw_b], [-bb / 2, tt_b]]
    return [rounded(P, [0, 0, r2, r, r, r2, 0, 0, r2, r, r, r2])], []


def _place(P, x0, sign):
    P = np.array([[x0 + sign * x, y] for x, y in P], float)
    return P if sign > 0 else P[::-1].copy()


def _channel(d, bf, tf, tw, x0=0.0, sign=1.0, r1=0.0, r2=0.0, s=0.0):
    tfw, tft = flange_thickness(tf, bf - tw, s)
    pts = [[0, 0], [bf, 0], [bf, tft], [tw, tfw], [tw, d - tfw], [bf, d - tft], [bf, d], [0, d]]
    return _place(rounded(pts, [0, 0, r2, r1, r1, r2, 0, 0]), x0, sign)


def _out_channel(v):
    return [_channel(_g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw"), 0.0, 1.0, _r(v), _r(v, "r2"), _slope(v))], []


def _angle(d, b, tf, tw, x0=0.0, sign=1.0, r1=0.0, r2=0.0):
    pts = [[0, 0], [b, 0], [b, tf], [tw, tf], [tw, d], [0, d]]
    return _place(rounded(pts, [0, 0, r2, r1, r2, 0]), x0, sign)


def _out_angle(v):
    return [_angle(_g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw"), 0.0, 1.0, _r(v), _r(v, "r2"))], []


def _out_2angle(v):
    d, b, tf, tw, s = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw"), _g(v, "dis")
    r1, r2 = _r(v), _r(v, "r2")
    return [_angle(d, b, tf, tw, s / 2, 1.0, r1, r2), _angle(d, b, tf, tw, -s / 2, -1.0, r1, r2)], []


def _out_2channel(v):
    d, b, tf, tw, s = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw"), _g(v, "dis")
    r1, r2 = _r(v), _r(v, "r2")
    k = _slope(v)
    return [_channel(d, b, tf, tw, s / 2, 1.0, r1, r2, k), _channel(d, b, tf, tw, -s / 2, -1.0, r1, r2, k)], []


def _out_lipped(v):
    d, b, t, c = _g(v, "t3"), _g(v, "t2"), _g(v, "tw"), _g(v, "lip")
    ri = _r(v)
    ro = ri + t if ri > 0 else 0.0
    pts = [[0, 0], [b, 0], [b, c], [b - t, c], [b - t, t], [t, t], [t, d - t], [b - t, d - t], [b - t, d - c],
           [b, d - c], [b, d], [0, d]]
    return [rounded(pts, [ro, ro, 0, 0, ri, ri, ri, ri, 0, 0, ro, ro])], []


def _out_lipped_z(v):
    """Cold-formed lipped Z: web centred on the axis, top flange and lip to the
    right (lip down), bottom flange and lip to the left (lip up)."""
    d, b, t, c = _g(v, "t3"), _g(v, "t2"), _g(v, "tw"), _g(v, "lip")
    ri = _r(v)
    ro = ri + t if ri > 0 else 0.0
    B = b - t / 2
    pts = [[-B, 0], [t / 2, 0], [t / 2, d - t], [B - t, d - t], [B - t, d - c], [B, d - c], [B, d], [-t / 2, d],
           [-t / 2, t], [-B + t, t], [-B + t, c], [-B, c]]
    return [rounded(pts, [ro, ro, ri, ri, 0, 0, ro, ro, ri, ri, 0, 0])], []



COLD = {"Lipped Channel", "Lipped Z", "Cold Channel", "Cold Z", "Cold Angle", "Hat"}  # one thickness tw (tf = tw)


def _cold(v):
    t, ri = _g(v, "tw"), _r(v)
    return t, ri, (ri + t if ri > 0 else 0.0)


def _out_cold_channel(v):
    """Cold-formed plain channel (JIS G3350 / TIS 1228 light channel)."""
    d, b = _g(v, "t3"), _g(v, "t2")
    t, ri, ro = _cold(v)
    pts = [[0, 0], [b, 0], [b, t], [t, t], [t, d - t], [b, d - t], [b, d], [0, d]]
    return [rounded(pts, [ro, 0, 0, ri, ri, 0, 0, ro])], []


def _out_cold_z(v):
    """Cold-formed plain Z: top flange to the right, bottom flange (t2b, blank =
    t2) to the left; widths measured over the web."""
    d, b = _g(v, "t3"), _g(v, "t2")
    bb = _g(v, "t2b") or b
    t, ri, ro = _cold(v)
    h = t / 2
    pts = [[-(bb - h), 0], [h, 0], [h, d - t], [b - h, d - t], [b - h, d], [-h, d], [-h, t], [-(bb - h), t]]
    return [rounded(pts, [0, ro, ri, 0, 0, ro, ri, 0])], []


def _out_cold_angle(v):
    """Cold-formed angle: legs t3 (vertical) and t2, one bend."""
    d, b = _g(v, "t3"), _g(v, "t2")
    t, ri, ro = _cold(v)
    pts = [[0, 0], [b, 0], [b, t], [t, t], [t, d], [0, d]]
    return [rounded(pts, [ro, 0, 0, ri, 0, 0])], []


def _out_hat(v):
    """Cold-formed hat: height t3, crown width t2 (over the webs), brims of
    length lip each side at the bottom (JIS G3350 / TIS 1228 H x A x C)."""
    d, a, c = _g(v, "t3"), _g(v, "t2"), _g(v, "lip")
    t, ri, ro = _cold(v)
    x0, x1 = a / 2, a / 2 + c  # web outer face, brim tip
    pts = [[-x1, 0], [-(x0 - t), 0], [-(x0 - t), d - t], [x0 - t, d - t], [x0 - t, 0], [x1, 0], [x1, t],
           [x0, t], [x0, d], [-x0, d], [-x0, t], [-x1, t]]
    return [rounded(pts, [0, ro, ri, ri, ro, 0, 0, ri, ro, ro, ri, 0])], []


def _j_cold(v, length):
    t, ri, _ro = _cold(v)
    return t ** 3 * length / 3.0


def _j_cold_channel(v):
    t, ri, _ = _cold(v)
    rm = ri + t / 2 if ri > 0 else 0.0
    return _j_cold(v, _g(v, "t3") + 2.0 * _g(v, "t2") - 2.0 * t - 2.0 * (2.0 - math.pi / 2) * rm)


def _j_cold_z(v):
    t, ri, _ = _cold(v)
    rm = ri + t / 2 if ri > 0 else 0.0
    b = _g(v, "t2")
    return _j_cold(v, _g(v, "t3") + b + (_g(v, "t2b") or b) - 2.0 * t - 2.0 * (2.0 - math.pi / 2) * rm)


def _j_cold_angle(v):
    t, ri, _ = _cold(v)
    rm = ri + t / 2 if ri > 0 else 0.0
    return _j_cold(v, _g(v, "t3") + _g(v, "t2") - t - (2.0 - math.pi / 2) * rm)


def _j_hat(v):
    t, ri, _ = _cold(v)
    rm = ri + t / 2 if ri > 0 else 0.0
    return _j_cold(v, 2.0 * _g(v, "t3") + _g(v, "t2") + 2.0 * _g(v, "lip") - 3.0 * t
                   - 4.0 * (2.0 - math.pi / 2) * rm)


def _sh_cold_channel(v, p):
    t = _g(v, "tw")
    return t * _g(v, "t3"), 5.0 / 3.0 * t * _g(v, "t2")


def _sh_cold_angle(v, p):
    t = _g(v, "tw")
    return 5.0 / 6.0 * t * _g(v, "t3"), 5.0 / 6.0 * t * _g(v, "t2")


def _sh_hat(v, p):
    t = _g(v, "tw")
    return 2.0 * t * _g(v, "t3"), 5.0 / 6.0 * t * (_g(v, "t2") + 2.0 * _g(v, "lip"))


def _j_tee(v):
    d, bf, tf, tw = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw")
    return (bf * tf ** 3 + (d - tf) * tw ** 3) / 3.0


def _j_concrete_tee(v):
    d, bf, tf, tw = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw")
    return j_rectangle(bf, tf) + j_rectangle(d - tf, tw)


def _j_lbeam(v):
    d, bf, tf, tw = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw")
    return j_rectangle(bf, tf) + j_rectangle(d - tf, tw)


def _j_i(v):
    d, bf, tf, tw, bb, tfb = _i_dims(v)
    return (bf * tf ** 3 + bb * tfb ** 3 + (d - tf - tfb) * tw ** 3) / 3.0


def _j_channel(v):
    d, bf, tf, tw = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw")
    return (2.0 * bf * tf ** 3 + (d - 2.0 * tf) * tw ** 3) / 3.0


def _j_angle(v):
    d, b, tf, tw = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw")
    return (b * tf ** 3 + (d - tf) * tw ** 3) / 3.0


def _j_lipped(v):
    """Thin-walled open section: (centre-line length) t^3 / 3; rounded bends
    shorten the centre line by (2 - pi/2) r_m each."""
    d, b, t, c = _g(v, "t3"), _g(v, "t2"), _g(v, "tw"), _g(v, "lip")
    ri = _r(v)
    rm = ri + t / 2 if ri > 0 else 0.0
    return t ** 3 * (d + 2.0 * (b - t) + 2.0 * (c - t) - 4.0 * (2.0 - math.pi / 2) * rm) / 3.0


def _j_box(v):
    """Bredt: 4 Am^2 / sum(s / t) about the wall centre line; rounded corners
    (outside radius r) reduce the enclosed area and the centre-line length."""
    d, b, tf, tw, ro = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw"), _r(v)
    if ro <= 0:
        return j_box(d, b, tf, tw)
    bm, dm = b - tw, d - tf
    rm = max(ro - 0.25 * (tf + tw), 0.0)
    Am = bm * dm - (4.0 - math.pi) * rm * rm
    s_t = 2 * (bm - 2 * rm) / tf + 2 * (dm - 2 * rm) / tw + 2 * math.pi * rm / (0.5 * (tf + tw))
    return 4.0 * Am * Am / s_t if s_t > 0 else 0.0


def _j_pipe(v):
    D, t = _g(v, "t3"), _g(v, "tw")
    return math.pi * (D ** 4 - (D - 2 * t) ** 4) / 32.0


def _exact_circle(v):
    D = _g(v, "t3")
    A, I = math.pi * D * D / 4.0, math.pi * D ** 4 / 64.0
    return {"A": A, "I33": I, "I22": I, "I23": 0.0, "J": 2 * I, "As2": 0.9 * A, "As3": 0.9 * A,
            "S33": I / (D / 2), "S22": I / (D / 2), "Z33": D ** 3 / 6.0, "Z22": D ** 3 / 6.0}


def _exact_pipe(v):
    D, t = _g(v, "t3"), _g(v, "tw")
    Di = D - 2 * t
    A, I = math.pi * (D * D - Di * Di) / 4.0, math.pi * (D ** 4 - Di ** 4) / 64.0
    Z = (D ** 3 - Di ** 3) / 6.0
    return {"A": A, "I33": I, "I22": I, "I23": 0.0, "J": 2 * I, "As2": 0.5 * A, "As3": 0.5 * A,
            "S33": I / (D / 2), "S22": I / (D / 2), "Z33": Z, "Z22": Z}


def _sh_rect(v, p):
    return 5.0 / 6.0 * p.A, 5.0 / 6.0 * p.A


def _sh_i(v, p):
    d, bf, tf, tw, bb, tfb = _i_dims(v)
    return tw * d, 5.0 / 6.0 * (bf * tf + bb * tfb)


def _sh_channel(v, p):
    return _g(v, "tw") * _g(v, "t3"), 5.0 / 3.0 * _g(v, "t2") * _g(v, "tf")


def _sh_tee(v, p):
    return _g(v, "tw") * _g(v, "t3"), 5.0 / 6.0 * _g(v, "t2") * _g(v, "tf")


def _sh_angle(v, p):
    return 5.0 / 6.0 * _g(v, "tw") * _g(v, "t3"), 5.0 / 6.0 * _g(v, "t2") * _g(v, "tf")


def _sh_2angle(v, p):
    return 5.0 / 3.0 * _g(v, "tw") * _g(v, "t3"), 5.0 / 3.0 * _g(v, "t2") * _g(v, "tf")


def _sh_2channel(v, p):
    return 2.0 * _g(v, "tw") * _g(v, "t3"), 10.0 / 3.0 * _g(v, "t2") * _g(v, "tf")


def _sh_box(v, p):
    return 2.0 * _g(v, "tw") * _g(v, "t3"), 2.0 * _g(v, "tf") * _g(v, "t2")


def _sh_lipped(v, p):
    t = _g(v, "tw")
    return t * _g(v, "t3"), 5.0 / 3.0 * t * _g(v, "t2")


_D, _B = Dim("t3", "Depth (t3)", 0.6), Dim("t2", "Width (t2)", 0.3)
SHAPES: Dict[str, Shape] = {s.key: s for s in [
    # ---- concrete
    Shape("Rectangular", "Rectangular", "Concrete", [_D, _B], _out_rect,
          lambda v: j_rectangle(_g(v, "t3"), _g(v, "t2")), _sh_rect,
          note="Beam or column.  Also a steel plate / flat bar with a steel material."),
    Shape("Circular", "Circular", "Concrete", [Dim("t3", "Diameter", 0.5)], _out_circle,
          lambda v: math.pi * _g(v, "t3") ** 4 / 32.0, None, _exact_circle,
          note="Round column.  Also a steel rod with a steel material."),
    Shape("Concrete Tee", "T-beam", "Concrete",
          [Dim("t3", "Total depth (t3)", 0.6), Dim("t2", "Flange width (bf)", 1.2),
           Dim("tf", "Flange (slab) thickness", 0.12), Dim("tw", "Web width (bw)", 0.3)],
          _out_tee, _j_concrete_tee,
          note="Beam acting with the slab on both sides (effective flange width bf)."),
    Shape("Concrete L", "L-beam (edge beam)", "Concrete",
          [Dim("t3", "Total depth (t3)", 0.6), Dim("t2", "Flange width incl. web (bf)", 0.8),
           Dim("tf", "Flange (slab) thickness", 0.12), Dim("tw", "Web width (bw)", 0.3)],
          _out_lbeam, _j_lbeam, mirror=True,
          note="Spandrel / edge beam with the slab on one side."),
    Shape("Concrete Box", "Hollow rectangle", "Concrete",
          [Dim("t3", "Depth (t3)", 1.0), Dim("t2", "Width (t2)", 0.8), Dim("tf", "Top / bottom wall", 0.15),
           Dim("tw", "Side wall", 0.15)], _out_box,
          lambda v: j_box(_g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw")),
          note="Box girder, hollow pier or core column."),
    Shape("Concrete Pipe", "Hollow circle", "Concrete",
          [Dim("t3", "Outside diameter", 1.0), Dim("tw", "Wall thickness", 0.15)], _out_pipe, _j_pipe,
          None, _exact_pipe, note="Hollow round pier or spun pile."),
    # ---- steel
    Shape("I", "I / Wide flange (H)", "Steel",
          [Dim("t3", "Depth (t3)", 0.3), Dim("t2", "Top flange width", 0.15), Dim("tf", "Top flange thickness", 0.009),
           Dim("tw", "Web thickness", 0.0065), Dim("t2b", "Bottom flange width", None, True, "same as top"),
           Dim("tfb", "Bottom flange thickness", None, True, "same as top"),
           Dim("r", "Fillet radius r (web-flange)", 0.013, True, "0 = sharp corner"),
           Dim("r2", "Flange tip radius r2", None, True, "0 = sharp corner")],
          _out_i, _j_i, _sh_i, material="Steel",
          note="H / W / I beam or column (rolled: e.g. H300x150x6.5x9 r13); bottom flange blank = same as the top; "
               "fillets included in A, I, S, Z and J."),
    Shape("Channel", "Channel [", "Steel",
          [Dim("t3", "Depth (t3)", 0.2), Dim("t2", "Flange width", 0.08), Dim("tf", "Flange thickness", 0.011),
           Dim("tw", "Web thickness", 0.0075), Dim("r", "Fillet radius r1 (web-flange)", 0.012, True, "0 = sharp corner"),
           Dim("r2", "Flange tip radius r2", 0.006, True, "0 = sharp corner")],
          _out_channel, _j_channel, _sh_channel, mirror=True, material="Steel",
          note="Rolled channel (r1 at the web, r2 at the flange tips); parallel or tapered flanges "
               "(JIS / TIS 5\u00b0, GB 1:10, AISC 1:6, EN UPN 8% / 5%) - the flange thickness is then the "
               "mean thickness, at the middle of the flange."),
    Shape("Tee", "Tee (T)", "Steel",
          [Dim("t3", "Depth (t3)", 0.15), Dim("t2", "Flange width", 0.15), Dim("tf", "Flange thickness", 0.01),
           Dim("tw", "Stem thickness", 0.007), Dim("r", "Fillet radius r (stem-flange)", 0.011, True, "0 = sharp corner")],
          _out_tee, _j_tee, _sh_tee, material="Steel", note="Rolled or cut tee (half an H)."),
    Shape("Angle", "Angle L", "Steel",
          [Dim("t3", "Vertical leg (t3)", 0.1), Dim("t2", "Horizontal leg (t2)", 0.1),
           Dim("tf", "Horizontal leg thickness", 0.01), Dim("tw", "Vertical leg thickness", 0.01),
           Dim("r", "Root radius r1", 0.010, True, "0 = sharp corner"),
           Dim("r2", "Toe radius r2", 0.007, True, "0 = sharp corner")],
          _out_angle, _j_angle, _sh_angle, mirror=True, material="Steel",
          note="Rolled angle, e.g. L100x100x10 r1 10, r2 7."),
    Shape("DoubleAngle", "Double angle 2L", "Steel",
          [Dim("t3", "Vertical leg (t3)", 0.1), Dim("t2", "Horizontal leg (each)", 0.1),
           Dim("tf", "Horizontal leg thickness", 0.01), Dim("tw", "Vertical leg thickness", 0.01),
           Dim("dis", "Gap between backs", 0.01), Dim("r", "Root radius r1", 0.010, True, "0 = sharp corner"),
           Dim("r2", "Toe radius r2", 0.007, True, "0 = sharp corner")],
          _out_2angle, lambda v: 2.0 * _j_angle(v), _sh_2angle, material="Steel",
          note="Two angles back to back (truss chords, bracing)."),
    Shape("DoubleChannel", "Double channel 2C", "Steel",
          [Dim("t3", "Depth (t3)", 0.2), Dim("t2", "Flange width (each)", 0.08), Dim("tf", "Flange thickness", 0.011),
           Dim("tw", "Web thickness", 0.0075), Dim("dis", "Gap between backs", 0.0),
           Dim("r", "Fillet radius r1 (web-flange)", 0.012, True, "0 = sharp corner"),
           Dim("r2", "Flange tip radius r2", 0.006, True, "0 = sharp corner")],
          _out_2channel, lambda v: 2.0 * _j_channel(v), _sh_2channel, material="Steel",
          note="Two channels back to back."),
    Shape("Box", "Box / Tube (RHS, SHS)", "Steel",
          [Dim("t3", "Depth (t3)", 0.2), Dim("t2", "Width (t2)", 0.2), Dim("tf", "Flange (top / bottom) thickness", 0.008),
           Dim("tw", "Web (side) thickness", 0.008),
           Dim("r", "Outside corner radius", 0.016, True, "0 = sharp corner")], _out_box, _j_box, _sh_box,
          material="Steel", note="Square / rectangular hollow section; inside corner radius = outside - thickness."),
    Shape("Pipe", "Pipe (CHS)", "Steel",
          [Dim("t3", "Outside diameter", 0.1143), Dim("tw", "Wall thickness", 0.0045)], _out_pipe, _j_pipe,
          None, _exact_pipe, material="Steel"),
    Shape("Lipped Channel", "Cold-formed lipped C", "Steel",
          [Dim("t3", "Depth (t3)", 0.15), Dim("t2", "Flange width", 0.065), Dim("lip", "Lip length", 0.02),
           Dim("tw", "Thickness", 0.0023), Dim("r", "Inside bend radius", 0.0023, True, "0 = sharp corner")],
          _out_lipped, _j_lipped, _sh_lipped, mirror=True, material="Steel",
          note="Light-gauge purlin / girt / roof truss member (e.g. C150x65x20x2.3, inside bend radius = t)."),
    Shape("Lipped Z", "Cold-formed lipped Z", "Steel",
          [Dim("t3", "Depth (t3)", 0.15), Dim("t2", "Flange width", 0.065), Dim("lip", "Lip length", 0.02),
           Dim("tw", "Thickness", 0.0023), Dim("r", "Inside bend radius", 0.0023, True, "0 = sharp corner")],
          _out_lipped_z, _j_lipped, _sh_lipped, mirror=True, material="Steel",
          note="Light-gauge Z purlin (lapped over supports); its principal axes are inclined - I23 and the "
               "principal moments are shown."),
    Shape("Cold Channel", "Cold-formed channel C", "Steel",
          [Dim("t3", "Depth (t3)", 0.1), Dim("t2", "Flange width", 0.05), Dim("tw", "Thickness", 0.0032),
           Dim("r", "Inside bend radius", 0.0032, True, "0 = sharp corner")],
          _out_cold_channel, _j_cold_channel, _sh_cold_channel, mirror=True, material="Steel",
          note="Light-gauge channel without lips (TIS 1228 / JIS G3350, e.g. C100x50x3.2; inside bend radius = t)."),
    Shape("Cold Z", "Cold-formed Z", "Steel",
          [Dim("t3", "Depth (t3)", 0.1), Dim("t2", "Top flange width", 0.05),
           Dim("t2b", "Bottom flange width", None, True, "same as top"), Dim("tw", "Thickness", 0.0032),
           Dim("r", "Inside bend radius", 0.0032, True, "0 = sharp corner")],
          _out_cold_z, _j_cold_z, None, mirror=True, material="Steel",
          note="Light-gauge Z without lips; its principal axes are inclined - I23 and the principal moments are shown."),
    Shape("Cold Angle", "Cold-formed angle L", "Steel",
          [Dim("t3", "Vertical leg (t3)", 0.05), Dim("t2", "Horizontal leg (t2)", 0.05), Dim("tw", "Thickness", 0.0032),
           Dim("r", "Inside bend radius", 0.0032, True, "0 = sharp corner")],
          _out_cold_angle, _j_cold_angle, _sh_cold_angle, mirror=True, material="Steel",
          note="Bent angle (TIS 1228 / JIS G3350, e.g. L50x50x3.2; inside bend radius = t)."),
    Shape("Hat", "Cold-formed hat", "Steel",
          [Dim("t3", "Height (t3)", 0.06), Dim("t2", "Crown width (over the webs)", 0.03),
           Dim("lip", "Brim length (each side)", 0.025), Dim("tw", "Thickness", 0.0023),
           Dim("r", "Inside bend radius", 0.0023, True, "0 = sharp corner")],
          _out_hat, _j_hat, _sh_hat, material="Steel",
          note="Hat / omega section (TIS 1228 / JIS G3350 H x A x C, e.g. 60x30x25x2.3): crown on top, brims below."),
]}

SHAPE_ALIASES = {"rectangle": "Rectangular", "concrete rectangular": "Rectangular", "circle": "Circular",
                 "concrete circle": "Circular", "steel i/wide flange": "I", "i/wide flange": "I", "wide flange": "I",
                 "steel channel": "Channel", "steel tee": "Tee", "steel angle": "Angle",
                 "steel double angle": "DoubleAngle", "double angle": "DoubleAngle", "steel tube": "Box",
                 "tube": "Box", "steel pipe": "Pipe", "concrete tee": "Concrete Tee", "concrete l": "Concrete L",
                 "steel box": "Box", "double channel": "DoubleChannel", "lipped channel": "Lipped Channel", "lipped z": "Lipped Z", "z": "Lipped Z",
                 "cold channel": "Cold Channel", "cold z": "Cold Z", "cold angle": "Cold Angle", "hat": "Hat",
                 "cold formed c": "Lipped Channel"}

DESIGNER = "Section Designer"
GENERAL = "General"


def shape_of(sec) -> Optional[Shape]:
    """The catalogue entry of a frame section (by its shape name), or None."""
    name = str(getattr(sec, "shape", "") or "")
    if name in SHAPES:
        return SHAPES[name]
    return SHAPES.get(SHAPE_ALIASES.get(name.lower().strip(), ""))


def dims_of(sec) -> dict:
    keys = ("t3", "t2", "tf", "tw", "t2b", "tfb", "dis", "lip", "r", "r2")
    out = {k: getattr(sec, k, None) for k in keys}
    slope = (getattr(sec, "params", None) or {}).get("slope")
    if slope and getattr(sec, "shape", "") in SLOPED:
        out["slope"] = float(slope)
    if shape_of(sec) is not None and shape_of(sec).key in COLD:
        out["tf"] = out.get("tw")
    return out


def check_dims(shape: Shape, v: dict) -> List[str]:
    """Problems with the dimensions of a parametric shape (empty: fine)."""
    errs = []
    for dm in shape.dims:
        x = v.get(dm.attr)
        if x is None:
            if not dm.optional:
                errs.append(f"{dm.label} is required")
            continue
        if dm.attr in ("dis", "r", "r2"):
            if x < 0:
                errs.append(f"{dm.label} cannot be negative")
        elif x <= 0:
            errs.append(f"{dm.label} must be positive")
    if errs:
        return errs
    d, b, tf, tw = _g(v, "t3"), _g(v, "t2"), _g(v, "tf"), _g(v, "tw")
    k = shape.key
    if k in ("Concrete Tee", "Tee", "Concrete L"):
        if tf >= d:
            errs.append("The flange must be thinner than the depth")
        if tw > b:
            errs.append("The web is wider than the flange")
    elif k in ("Concrete Box", "Box"):
        if 2 * tf >= d or 2 * tw >= b:
            errs.append("The walls fill the whole section - make them thinner")
    elif k in ("Concrete Pipe", "Pipe"):
        if 2 * tw >= d:
            errs.append("The wall is thicker than the radius")
    elif k == "I":
        _d, bf, tf, tw, bb, tfb = _i_dims(v)
        if tf + tfb >= d:
            errs.append("The flanges are thicker than the depth")
        if tw > min(bf, bb):
            errs.append("The web is wider than a flange")
    elif k in ("Channel", "DoubleChannel"):
        if 2 * tf >= d:
            errs.append("The flanges are thicker than the depth")
        if tw > b:
            errs.append("The web is thicker than the flange width")
    elif k in ("Angle", "DoubleAngle"):
        if tf >= d or tw >= b:
            errs.append("A leg is thicker than the other leg is long")
    elif k in ("Lipped Channel", "Lipped Z"):
        c = _g(v, "lip")
        if 2 * tw >= d or tw >= b or c > d / 2:
            errs.append("Thickness / lip too large for the depth and flange width")
        if c < tw:
            errs.append("The lip must be at least as long as the thickness")
    elif k in ("Cold Channel", "Cold Z", "Cold Angle", "Hat"):
        bb = _g(v, "t2b") or b
        if k == "Hat":
            if 2 * tw >= d or 2 * tw >= b or _g(v, "lip") <= tw:
                errs.append("Thickness too large for the height / crown, or the brims are too short")
        elif (2 if k != "Cold Angle" else 1) * tw >= d or tw >= min(b, bb):
            errs.append("The thickness is too large for the depth / flange width")
    if errs:
        return errs
    sl = _slope(v)
    if sl:
        if k not in SLOPED:
            errs.append("Only I and channel shapes have tapered flanges")
        elif not 0 < sl < 0.5:
            errs.append("The flange slope must be between 0 and 0.5 (tan of the angle)")
        else:
            if k == "I":
                _d, bf, tf, tw, bb, tfb = _i_dims(v)
                outs = [(tf, (bf - tw) / 2.0), (tfb, (bb - tw) / 2.0)]
            else:
                outs = [(tf, b - tw)]
            for t_, o in outs:
                web, tip = flange_thickness(t_, o, sl)
                if tip <= 0:
                    errs.append("The taper leaves no flange at the tip - check the slope / thickness")
                elif k != "I" and 2 * web >= d or k == "I" and web >= d / 2:
                    errs.append("The tapered flanges are thicker than the depth")
        if errs:
            return errs
    r, r2 = _r(v), _r(v, "r2")
    if r or r2:  # the arcs must fit on the straight parts they replace
        if k == "I":
            _d, bf, tf, tw, bb, tfb = _i_dims(v)
            webt = flange_thickness(tf, (bf - tw) / 2.0, sl)[0] + flange_thickness(tfb, (bb - tw) / 2.0, sl)[0]
            tip = min(flange_thickness(tf, (bf - tw) / 2.0, sl)[1], flange_thickness(tfb, (bb - tw) / 2.0, sl)[1])
            if r + r2 > 0.5 * (min(bf, bb) - tw) or 2 * r > d - webt or r2 > tip:
                errs.append("The radii are too large for the flanges / web")
        elif k == "Tee":
            if r > 0.5 * (b - tw) or r > d - tf:
                errs.append("The fillet radius is too large for the flange / stem")
        elif k in ("Channel", "DoubleChannel"):
            web, tip = flange_thickness(tf, b - tw, sl)
            if r + r2 > b - tw or 2 * r > d - 2 * web or r2 > tip:
                errs.append("The radii are too large for the flange / web")
        elif k in ("Angle", "DoubleAngle"):
            if r + r2 > min(b - tw, d - tf) or r2 > min(tf, tw):
                errs.append("The radii are too large for the legs")
        elif k == "Box":
            if r > 0.5 * min(b, d):
                errs.append("The corner radius is larger than half the width")
        elif k in ("Lipped Channel", "Lipped Z"):
            c = _g(v, "lip")
            if r + tw > c or 2 * (r + tw) > b or 2 * (r + tw) > d / 2:
                errs.append("The bend radius is too large for the lip / flange")
        elif k in ("Cold Channel", "Cold Z", "Cold Angle"):
            if r + tw > min(b, _g(v, "t2b") or b) or r + tw > d / 2:
                errs.append("The bend radius is too large for the flange / web")
        elif k == "Hat":
            if 2 * (r + tw) > b or r + tw > _g(v, "lip") or 2 * (r + tw) > d:
                errs.append("The bend radius is too large for the crown / brims")
    return errs


def mirrored(loops: List[np.ndarray]) -> List[np.ndarray]:
    return [np.column_stack([-P[:, 0], P[:, 1]])[::-1].copy() for P in loops]


def shape_outline(shape: Shape, v: dict, mirror: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """(solids, holes) of a parametric shape about its centroid; solids counter-
    clockwise, holes clockwise."""
    solids, holes = shape.outline(v)
    solids, holes = [ccw(P) for P in solids], [cw(P) for P in holes]
    if mirror and shape.mirror:
        solids, holes = mirrored(solids), mirrored(holes)
        solids, holes = [ccw(P) for P in solids], [cw(P) for P in holes]
    m = sum(loop_moments(P) for P in solids + holes)
    if m[0] > 0:
        c = np.array([m[1] / m[0], m[2] / m[0]])
        solids, holes = [P - c for P in solids], [P - c for P in holes]
    return solids, holes


FILLET_FE = {"I", "Channel", "Tee", "Angle", "DoubleAngle", "DoubleChannel"}


def shape_properties(shape: Shape, v: dict, mirror: bool = False, torsion: bool = True) -> SectionProps:
    """Properties of a parametric shape.  With fillets / rounded corners the
    outline carries the arcs (A, I, S, Z exact for it); J of rolled open shapes
    with fillets comes from finite elements (the fillets stiffen the junctions
    noticeably), of tubes and cold-formed sections from the thin-walled
    formulas with the rounded centre line."""
    solids, holes = shape_outline(shape, v, mirror)
    p = region_properties(solids + holes)
    if torsion and shape.key in FILLET_FE and (_r(v) > 0 or _r(v, "r2") > 0):
        p.J = torsion_constant(mesh_region(solids, holes, fe_size(solids + holes)))
        p.J_method = "finite elements (with fillets)"
    else:
        p.J = float(shape.torsion(v))
        p.J_method = "closed form" + (" (thin-walled)" if shape.group == "Steel" else "")
    if shape.shear is not None:
        p.As2, p.As3 = map(float, shape.shear(v, p))
        p.as_method = "closed form"
    if shape.exact is not None:
        for k, x in shape.exact(v).items():
            setattr(p, k, float(x))
        p.r33, p.r22 = math.sqrt(p.I33 / p.A), math.sqrt(p.I22 / p.A)
        p.Imax = p.Imin = p.I33
        p.as_method = "closed form"
    if mirror and shape.mirror:
        p.angle = -p.angle
    return p


# ---------------------------------------------------------- Section Designer
def designer_polygons(geometry: Sequence[dict]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """(solids, holes) of the drawn shapes of a Section Designer section.
    Shapes: ``{"kind": "polygon", "points": [[x, y], ...], "hole": bool}`` and
    ``{"kind": "circle", "center": [x, y], "radius": r, "hole": bool}``."""
    solids, holes = [], []
    for g in geometry or []:
        kind = g.get("kind", "polygon")
        if kind == "circle":
            r = float(g.get("radius") or 0.0)
            if r <= 0:
                continue
            c = g.get("center") or (0.0, 0.0)
            P = circle(float(c[0]), float(c[1]), r, int(g.get("segments") or CIRCLE_SEGMENTS))
        else:
            P = np.asarray(g.get("points") or [], float).reshape(-1, 2)
            if len(P) >= 2 and np.allclose(P[0], P[-1]):
                P = P[:-1]
            if len(P) < 3 or abs(signed_area(P)) <= 0:
                continue
        (holes if g.get("hole") else solids).append(P)
    return solids, holes


@dataclass
class RegionMesh:
    """A triangulated region: nodes, triangles (counter-clockwise) and the
    boundary loops (solids counter-clockwise, holes clockwise)."""
    P: np.ndarray
    tris: np.ndarray
    loops: List[np.ndarray] = field(default_factory=list)
    conforming: bool = True


def _split_edges(polys: Sequence[np.ndarray], tol: float):
    """All polygon edges split at their mutual intersections / touching points."""
    A = np.vstack([P for P in polys])
    B = np.vstack([np.roll(P, -1, axis=0) for P in polys])
    src = np.concatenate([np.full(len(P), k) for k, P in enumerate(polys)])
    D = B - A
    out = []
    for i in range(len(A)):
        a, d = A[i], D[i]
        L = math.hypot(*d)
        if L <= tol:
            continue
        ts = [0.0, 1.0]
        # proper / touching intersections with every other edge
        den = d[0] * D[:, 1] - d[1] * D[:, 0]
        w = A - a
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[:, 0] * D[:, 1] - w[:, 1] * D[:, 0]) / den
            u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / den
        ok = (np.abs(den) > 1e-14 * L * np.linalg.norm(D, axis=1)) & (u >= -tol / np.maximum(np.linalg.norm(D, axis=1), 1e-300)) \
            & (u <= 1 + tol / np.maximum(np.linalg.norm(D, axis=1), 1e-300))
        ts += [float(x) for x in t[ok] if tol / L < x < 1 - tol / L]
        # end points of other edges lying on this one (colinear overlaps, T-joints)
        for Q in (A, B):
            r = Q - a
            tt = (r @ d) / (L * L)
            dist = np.abs(r[:, 0] * d[1] - r[:, 1] * d[0]) / L
            sel = (dist <= tol) & (tt > tol / L) & (tt < 1 - tol / L)
            ts += [float(x) for x in tt[sel]]
        ts = sorted(set(round(x, 12) for x in ts))
        for t0, t1 in zip(ts[:-1], ts[1:]):
            if (t1 - t0) * L > tol:
                out.append((a + t0 * d, a + t1 * d, int(src[i])))
    return out


def _inside(pts: np.ndarray, solids, holes) -> np.ndarray:
    pts = np.atleast_2d(pts)
    ins = np.zeros(len(pts), bool)
    for P in solids:
        ins |= points_in_polygon(pts, P)
    for H in holes:
        ins &= ~points_in_polygon(pts, H)
    return ins


def _boundary_segments(solids, holes, tol: float, scale: float, interfaces=()):
    """The pieces of the drawn edges that bound the region (union of the solids
    minus the holes), oriented with the region on their left; plus the pieces of
    the ``interfaces`` outlines (material boundaries) lying inside the region."""
    n_sh = len(solids) + len(holes)
    segs = _split_edges(list(solids) + list(holes) + list(interfaces), tol)
    if not segs:
        return np.zeros((0, 2)), np.zeros((0, 2))
    a = np.array([s[0] for s in segs])
    b = np.array([s[1] for s in segs])
    src = np.array([s[2] for s in segs])
    d = b - a
    n = np.column_stack([-d[:, 1], d[:, 0]]) / np.linalg.norm(d, axis=1)[:, None]
    m = 0.5 * (a + b)
    eps = max(1e-7 * scale, 10 * tol)
    left = _inside(m + eps * n, solids, holes)
    right = _inside(m - eps * n, solids, holes)
    keep = (left != right) | ((src >= n_sh) & left & right)
    a, b, left = a[keep], b[keep], left[keep]
    A = np.where(left[:, None], a, b)
    B = np.where(left[:, None], b, a)
    # drop duplicates (coincident edges of two shapes)
    key = {}
    for i in range(len(A)):
        k = (round(A[i, 0] / tol), round(A[i, 1] / tol), round(B[i, 0] / tol), round(B[i, 1] / tol))
        key.setdefault(k, i)
    idx = sorted(key.values())
    return A[idx], B[idx]


def _loops_from_tris(P: np.ndarray, tris: np.ndarray) -> List[np.ndarray]:
    edges: Dict[Tuple[int, int], int] = {}
    for t in tris:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = 1
    nxt: Dict[int, List[int]] = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
    loops = []
    used = set()
    for start in list(nxt):
        for first in list(nxt[start]):
            if (start, first) in used:
                continue
            loop = [start]
            a, b = start, first
            guard = 0
            while guard < len(edges) + 2:
                guard += 1
                used.add((a, b))
                if b == start:
                    break
                loop.append(b)
                cand = [c for c in nxt.get(b, []) if (b, c) not in used]
                if not cand:
                    break
                a, b = b, cand[0]
            if len(loop) >= 3:
                loops.append(_simplify(P[loop]))
    return [L for L in loops if len(L) >= 3]


def _simplify(L: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop vertices lying on the straight line through their neighbours."""
    L = np.asarray(L, float)
    if len(L) < 4:
        return L
    scale = float(np.ptp(L, axis=0).max()) or 1.0
    keep = []
    n = len(L)
    for i in range(n):
        a, b, c = L[i - 1], L[i], L[(i + 1) % n]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) > tol * scale * scale:
            keep.append(i)
    return L[keep] if len(keep) >= 3 else L


def mesh_region(solids: Sequence[np.ndarray], holes: Sequence[np.ndarray], h: Optional[float] = None,
                max_iter: int = 40, interfaces: Sequence[np.ndarray] = ()) -> RegionMesh:
    """Conforming Delaunay triangulation of the union of ``solids`` minus
    ``holes``: the outline (all intersections included) as triangle edges, an
    interior lattice of spacing ``h`` (None: the outline points only).  The
    ``interfaces`` outlines (e.g. a steel shape inside concrete) are triangle
    edges too, so every triangle lies in one material."""
    from scipy.spatial import Delaunay

    solids = [np.asarray(P, float) for P in solids if len(P) >= 3]
    holes = [np.asarray(P, float) for P in holes if len(P) >= 3]
    if not solids:
        return RegionMesh(np.zeros((0, 2)), np.zeros((0, 3), int), [])
    allp = np.vstack(solids)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    scale = float(max(hi - lo)) or 1.0
    tol = 1e-9 * scale
    A, B = _boundary_segments(solids, holes, tol, scale, [np.asarray(P, float) for P in interfaces if len(P) >= 3])
    if len(A) == 0:
        return RegionMesh(np.zeros((0, 2)), np.zeros((0, 3), int), [])
    # boundary points (each segment sampled at spacing h)
    pts = []
    chain = []  # consecutive boundary point pairs that must be triangle edges
    for a, b in zip(A, B):
        L = float(np.linalg.norm(b - a))
        k = 1 if not h or h <= 0 else max(1, int(math.ceil(L / h - 1e-9)))
        seg = [a + (b - a) * j / k for j in range(k + 1)]
        chain += list(zip(seg[:-1], seg[1:]))
        pts += seg
    if h and h > 0:  # interior triangular lattice, kept clear of the outline
        dy = h * math.sqrt(3.0) / 2.0
        ys = np.arange(lo[1] + 0.5 * dy, hi[1], dy)
        lat = []
        for j, y in enumerate(ys):
            xs = np.arange(lo[0] + (0.5 if j % 2 else 0.0) * h + 0.25 * h, hi[0], h)
            lat.append(np.column_stack([xs, np.full(len(xs), y)]))
        lat = np.vstack(lat) if lat else np.zeros((0, 2))
        if len(lat):
            lat = lat[_inside(lat, solids, holes)]
            if len(lat):
                lat = lat[_seg_dist(lat, A, B) > 0.45 * h]
        pts += list(lat)
    P = np.array(pts, float)
    P, inv = _unique_points(P, tol)
    need = []
    for a, b in chain:
        need.append((a, b))
    conforming = False
    tri = None
    for _ in range(max_iter):
        tri = Delaunay(P, qhull_options="Qbb Qc Qz")
        T = tri.simplices
        es = set()
        for t in T:
            for i in range(3):
                x, y = int(t[i]), int(t[(i + 1) % 3])
                es.add((min(x, y), max(x, y)))
        index = {(round(p[0] / tol), round(p[1] / tol)): i for i, p in enumerate(P)}
        missing, keep = [], []
        for a, b in need:
            ia = index.get((round(a[0] / tol), round(a[1] / tol)))
            ib = index.get((round(b[0] / tol), round(b[1] / tol)))
            if ia is None or ib is None or (min(ia, ib), max(ia, ib)) not in es:
                missing.append((a, b))
            else:
                keep.append((a, b))
        if not missing:
            conforming = True
            break
        new = []
        need = keep
        for a, b in missing:
            m = 0.5 * (a + b)
            new.append(m)
            need += [(a, m), (m, b)]
        P = np.vstack([P, np.array(new)])
        P, _ = _unique_points(P, tol)
    T = tri.simplices
    c = P[T].mean(axis=1)
    T = T[_inside(c, solids, holes)]
    # counter-clockwise, drop slivers of zero area
    v = P[T]
    ar = 0.5 * ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))
    T = np.where((ar < 0)[:, None], T[:, [0, 2, 1]], T)
    T = T[np.abs(ar) > 1e-14 * scale * scale]
    used = np.unique(T)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, T = P[used], remap[T]
    return RegionMesh(P, T, _loops_from_tris(P, T), conforming)


def _unique_points(P: np.ndarray, tol: float):
    key = np.round(P / max(tol, 1e-300)).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    order = np.sort(idx)
    return P[order], inv


def torsion_constant(mesh: RegionMesh, weights: Optional[np.ndarray] = None) -> float:
    """Saint-Venant torsion constant J of a triangulated region: the warping
    function w from quadratic (6-node) finite elements, J = Ip - f . w.  The
    displacement-type solution bounds J from above and converges with h^4."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    P, T = mesh.P, mesh.tris
    if len(T) == 0:
        return 0.0
    v = P[T]
    ar = 0.5 * ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))
    A = ar.sum()
    c = (ar[:, None] * v.mean(axis=1)).sum(axis=0) / A
    X = P - c
    v = X[T]
    # mid-side nodes
    n = len(P)
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.ravel()
    m = len(T)
    mid = n + inv.reshape(3, m).T  # (m, 3): edges 0-1, 1-2, 2-0
    X6 = np.vstack([X, 0.5 * (X[ue[:, 0]] + X[ue[:, 1]])])
    T6 = np.column_stack([T, mid])
    x1, y1 = v[:, 0, 0], v[:, 0, 1]
    x2, y2 = v[:, 1, 0], v[:, 1, 1]
    x3, y3 = v[:, 2, 0], v[:, 2, 1]
    gLx = np.column_stack([y2 - y3, y3 - y1, y1 - y2]) / (2 * ar)[:, None]  # grad of the area coordinates
    gLy = np.column_stack([x3 - x2, x1 - x3, x2 - x1]) / (2 * ar)[:, None]
    gauss = [(2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)]
    Ke = np.zeros((m, 6, 6))
    fe = np.zeros((m, 6))
    wt = np.ones(m) if weights is None else np.asarray(weights, float)  # G_i / G_base per triangle
    for L1, L2, L3 in gauss:
        L = (L1, L2, L3)
        # dN/dL (6 x 3): corners L_i(2 L_i - 1), mid-sides 4 L_i L_j
        dN = np.zeros((6, 3))
        for i in range(3):
            dN[i, i] = 4 * L[i] - 1
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            dN[3 + k, i] = 4 * L[j]
            dN[3 + k, j] = 4 * L[i]
        Nx = gLx @ dN.T  # (m, 6)
        Ny = gLy @ dN.T
        w = wt * ar / 3.0
        Ke += w[:, None, None] * (Nx[:, :, None] * Nx[:, None, :] + Ny[:, :, None] * Ny[:, None, :])
        xg = L1 * x1 + L2 * x2 + L3 * x3
        yg = L1 * y1 + L2 * y2 + L3 * y3
        fe += w[:, None] * (yg[:, None] * Nx - xg[:, None] * Ny)
    N6 = len(X6)
    rows = np.repeat(T6, 6, axis=1).ravel()
    cols = np.tile(T6, (1, 6)).ravel()
    K = coo_matrix((Ke.ravel(), (rows, cols)), shape=(N6, N6)).tocsr()
    f = np.bincount(T6.ravel(), weights=fe.ravel(), minlength=N6)
    Ip = (wt * ar / 6.0 * (x1 * x1 + x2 * x2 + x3 * x3 + x1 * x2 + x2 * x3 + x3 * x1
                           + y1 * y1 + y2 * y2 + y3 * y3 + y1 * y2 + y2 * y3 + y3 * y1)).sum()
    # the warping function is defined up to a constant: hold one node of each part
    ncomp, lab = connected_components(K, directed=False)
    fixed = np.array([int(np.argmax(lab == k)) for k in range(ncomp)])
    free = np.setdiff1d(np.arange(N6), fixed)
    w = np.zeros(N6)
    w[free] = spsolve(K[free][:, free].tocsc(), f[free])
    return float(Ip - f @ w)


def fe_size(loops: Sequence[np.ndarray], max_nodes: int = 5000) -> float:
    """Element size for the torsion solution: about a sixth of the typical wall
    thickness (2 A / perimeter) - quadratic elements, so a few across the wall
    are enough - limited so the mesh stays below ``max_nodes``."""
    m = sum(loop_moments(P) for P in loops)
    A = abs(float(m[0]))
    per = float(sum(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).sum() for P in loops))
    if A <= 0 or per <= 0:
        return 0.0
    t = 2.0 * A / per
    h = t / 4.0
    return max(h, math.sqrt(A / (0.866 * max_nodes)))


def designer_properties(geometry: Sequence[dict], torsion: bool = True,
                        ratios: Optional[dict] = None) -> Tuple[SectionProps, RegionMesh]:
    """Properties of a Section Designer section and its (coarse) triangulation,
    with the outline relative to the drawing origin (the centroid is in
    ``xc, yc``).  Shapes of other materials: transformed properties with the
    modular ratios ``ratios`` ({material: (E_i/E, G_i/G)})."""
    if is_composite(geometry):
        p, union, _ = composite_properties(geometry, ratios or {}, torsion)
        return p, union
    solids, holes = designer_polygons(geometry)
    mesh = mesh_region(solids, holes, None)
    p = region_properties(mesh.loops)
    if torsion and p.A > 0:
        h = fe_size(mesh.loops)
        fine = mesh_region(solids, holes, h)
        p.J = torsion_constant(fine)
        p.J_method = "finite elements"
    return p, mesh


# ------------------------------------------------- several materials (composite)
def is_composite(geometry) -> bool:
    """A Section Designer outline with shapes of another material than the
    section's (e.g. a steel I in a concrete column)."""
    return any(g.get("material") and not g.get("hole") for g in geometry or [])


def designer_parts(geometry) -> Dict[Optional[str], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """(solids, holes) per material (None = the section's material).  A shape of
    another material displaces the section material where they overlap (the
    steel inside the concrete); among other materials a later one wins."""
    order: List[Optional[str]] = [None]
    by: Dict[Optional[str], List[dict]] = {None: []}
    holes = []
    for g in geometry or []:
        if g.get("hole"):
            holes.append(g)
            continue
        k = g.get("material") or None
        if k not in by:
            by[k] = []
            order.append(k)
        by[k].append(g)
    hole_polys = designer_polygons(holes)[0] if holes else []
    out = {}
    for i, k in enumerate(order):
        sol = designer_polygons(by[k])[0]
        if not sol:
            continue
        above = [P for kk in order[i + 1:] for P in designer_polygons(by[kk])[0]]
        out[k] = (sol, list(hole_polys) + above)
    return out


def _ratio(ratios, k, j):
    if k is None or not ratios:
        return 1.0
    r = ratios.get(k)
    if r is None:
        return 1.0
    return float(r[j] if isinstance(r, (list, tuple)) else r)


def material_ratios(geometry, base: str, materials) -> Dict[str, Tuple[float, float]]:
    """(E_i / E_base, G_i / G_base) of the other materials of a composite outline."""
    out = {}
    mb = materials.get(base) if materials else None
    for g in geometry or []:
        k = g.get("material")
        if not k or k in out or mb is None or k not in materials:
            continue
        mi = materials[k]
        out[k] = (float(mi.E / mb.E) if mb.E else 1.0, float(mi.shear_modulus / mb.shear_modulus)
                  if mb.shear_modulus else 1.0)
    return out


def composite_properties(geometry, ratios, torsion: bool = True, strips: int = 1600):
    """Transformed properties (in the section material) of a multi-material
    outline: A = sum n_i A_i, I about the transformed centroid, shear areas from
    the shear-flow integral over the transformed widths, J by finite elements
    with each triangle's G.  Returns (props, union mesh, {material: loops})."""
    parts = designer_parts(geometry)
    loops_by = {}
    mt = np.zeros(6)
    for k, (sol, hol) in parts.items():
        mesh = mesh_region(sol, hol, None)
        loops_by[k] = mesh.loops
        if mesh.loops:
            mt = mt + _ratio(ratios, k, 0) * sum(loop_moments(L) for L in mesh.loops)
    solids, holes = designer_polygons(geometry)
    union = mesh_region(solids, holes, None)
    p = SectionProps()
    if mt[0] <= 0 or not union.loops:
        return p, union, loops_by
    A = float(mt[0])
    xc, yc = mt[1] / A, mt[2] / A
    p.A, p.xc, p.yc = A, float(xc), float(yc)
    p.I33 = float(mt[3] - A * yc * yc)
    p.I22 = float(mt[4] - A * xc * xc)
    I23 = float(mt[5] - A * xc * yc)
    p.I23 = 0.0 if abs(I23) < 1e-7 * (abs(p.I33) + abs(p.I22)) else I23
    allp = np.vstack(union.loops)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    p.width, p.depth = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    p.perimeter = float(sum(np.linalg.norm(np.roll(L, -1, axis=0) - L, axis=1).sum() for L in union.loops))
    p.r33, p.r22 = math.sqrt(max(p.I33, 0) / A), math.sqrt(max(p.I22, 0) / A)
    c, d = 0.5 * (p.I33 + p.I22), math.hypot(0.5 * (p.I33 - p.I22), p.I23)
    p.Imax, p.Imin = c + d, c - d
    p.angle = 0.5 * math.atan2(-2.0 * p.I23, p.I33 - p.I22) if p.I23 else 0.0
    for axis in (1, 0):
        a0, a1 = float(lo[axis]), float(hi[axis])
        e = np.linspace(a0, a1, strips + 1)
        mid = 0.5 * (e[1:] + e[:-1])
        h = (a1 - a0) / strips
        b = sum(_ratio(ratios, k, 0) * _chords(L, mid, axis) for k, L in loops_by.items() if L)
        dA = b * h
        sft = mid - (yc if axis == 1 else xc)
        mm = sft * dA
        Q = np.concatenate([np.cumsum(mm[::-1])[::-1][1:], [0.0]]) + 0.5 * mm
        ok = b > 1e-12 * max(p.width, p.depth)
        integral = float((Q[ok] ** 2 / b[ok]).sum() * h)
        I = p.I33 if axis == 1 else p.I22
        As = I * I / integral if integral > 0 else 0.0
        if axis == 1:
            p.As2 = As
        else:
            p.As3 = As
    p.as_method = "shear-flow integral (transformed)"
    if torsion:
        others = [k for k in parts if k is not None]
        inter = [P for k in others for P in parts[k][0]]
        fine = mesh_region(solids, holes, fe_size(union.loops), interfaces=inter)
        cen = fine.P[fine.tris].mean(axis=1)
        wt = np.ones(len(fine.tris))
        for k in others:  # later materials win, as in designer_parts
            sol, hol = parts[k]
            inside = _inside(cen, sol, hol)
            wt[inside] = _ratio(ratios, k, 1)
        p.J = torsion_constant(fine, wt)
        p.J_method = "finite elements (transformed, G_i / G)"
    return p, union, loops_by


def material_areas(geometry) -> Dict[Optional[str], float]:
    """Area of each material of a (composite) outline (None = section material)."""
    key = "areas" + __import__("json").dumps(geometry, sort_keys=True, default=str)
    if key not in _PIECES:
        out = {}
        for k, (sol, hol) in designer_parts(geometry).items():
            mesh = mesh_region(sol, hol, None)
            out[k] = float(sum(loop_moments(L)[0] for L in mesh.loops))
        _PIECES[key] = out
    return dict(_PIECES[key])


def section_parts(sec):
    """[(material or None, solids, holes)] of a composite frame section about the
    same origin as :func:`section_loops`, for drawing each material."""
    geom = getattr(sec, "geometry", None)
    if not geom or not is_composite(geom):
        return None
    lo = section_loops(sec)
    if lo is None:
        return None
    solids, holes = designer_polygons(geom)
    union = mesh_region(solids, holes, None)
    m = sum(loop_moments(L) for L in union.loops)
    c = np.array([m[1] / m[0], m[2] / m[0]])
    out = []
    for k, (sol, hol) in designer_parts(geom).items():
        mesh = mesh_region(sol, hol, None)
        out.append((k, [L - c for L in mesh.loops if signed_area(L) > 0], [L - c for L in mesh.loops if signed_area(L) < 0]))
    return out


# -------------------------------------------------------- frame sections
def section_loops(sec) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """(solids, holes) of a frame section about its centroid (x = local 3,
    y = local 2), or None when the shape / dimensions are unknown."""
    if sec is None:
        return None
    geom = getattr(sec, "geometry", None)
    if geom:
        solids, holes = designer_polygons(geom)
        mesh = mesh_region(solids, holes, None)
        if not mesh.loops:
            return None
        m = sum(loop_moments(P) for P in mesh.loops)
        if m[0] <= 0:
            return None
        c = np.array([m[1] / m[0], m[2] / m[0]])
        sol = [L - c for L in mesh.loops if signed_area(L) > 0]
        hol = [L - c for L in mesh.loops if signed_area(L) < 0]
        return sol, hol
    shape = shape_of(sec)
    if shape is None:
        return None
    v = dims_of(sec)
    if check_dims(shape, v):
        return None
    return shape_outline(shape, v, bool((getattr(sec, "params", None) or {}).get("mirror")))


_PIECES: Dict[str, Optional[List[np.ndarray]]] = {}


def _key(sec) -> str:
    import json
    return json.dumps([getattr(sec, "shape", ""), dims_of(sec), getattr(sec, "params", None) or {},
                       getattr(sec, "geometry", None) or []], sort_keys=True, default=str)


def convex_pieces(sec) -> Optional[List[np.ndarray]]:
    """Triangles of a frame section for the 3-D extrusion, in (y2, y3) = (y, x)
    order, or None (cached by shape, dimensions and outline)."""
    key = _key(sec)
    if key in _PIECES:
        return _PIECES[key]
    lo = section_loops(sec)
    out = None
    if lo is not None:
        solids, holes = lo
        mesh = mesh_region(solids, holes, None)
        if len(mesh.tris):
            out = [mesh.P[t][:, ::-1].copy() for t in mesh.tris]
    if len(_PIECES) > 500:
        _PIECES.clear()
    _PIECES[key] = out
    return out


def section_outline_cached(sec):
    """:func:`section_loops`, cached (previews, plan footprints)."""
    key = "L" + _key(sec)
    if key not in _PIECES:
        if len(_PIECES) > 500:
            _PIECES.clear()
        _PIECES[key] = section_loops(sec)
    return _PIECES[key]


_PROPS: Dict[str, Optional[SectionProps]] = {}


def section_ratios(sec, materials=None) -> dict:
    """Modular ratios of a composite section: from the materials when given, else
    those stored when its properties were last computed."""
    geom = getattr(sec, "geometry", None)
    if not geom or not is_composite(geom):
        return {}
    if materials:
        r = material_ratios(geom, sec.material, materials)
        if r:
            return r
    return {k: tuple(v) for k, v in ((getattr(sec, "params", None) or {}).get("n") or {}).items()}


def section_properties(sec, torsion: bool = True, materials=None) -> Optional[SectionProps]:
    """Computed properties of a parametric or Section Designer frame section
    (cached by shape, dimensions and outline: J may need finite elements).
    Composite sections need the materials (or the modular ratios stored)."""
    import copy as _copy
    ratios = section_ratios(sec, materials)
    key = f"{torsion}{sorted(ratios.items())}" + _key(sec)
    if key not in _PROPS:
        if len(_PROPS) > 300:
            _PROPS.clear()
        _PROPS[key] = _section_properties(sec, torsion, ratios)
    p = _PROPS[key]
    return None if p is None else _copy.copy(p)


def _section_properties(sec, torsion: bool, ratios=None) -> Optional[SectionProps]:
    if getattr(sec, "geometry", None):
        p, _ = designer_properties(sec.geometry, torsion, ratios)
        return p if p.A > 0 else None
    shape = shape_of(sec)
    if shape is None:
        return None
    v = dims_of(sec)
    if check_dims(shape, v):
        return None
    return shape_properties(shape, v, bool((getattr(sec, "params", None) or {}).get("mirror")), torsion)
