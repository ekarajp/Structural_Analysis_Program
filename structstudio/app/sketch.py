"""Small isometric pictures of a model (start page thumbnails, New Model cards).

:func:`model_sketch` projects the model once into a light, JSON-safe drawing
(grid lines, floors, walls, beams, columns, braces, supports); :func:`paint_sketch`
draws it into any rectangle in the colours of the current theme, so a thumbnail
follows the light / dark mode like the model windows do.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui

COS30 = math.cos(math.radians(30.0))
MAX_ITEMS = 6000  # per layer; a huge model is thinned evenly


def _proj(x: float, y: float, z: float):
    return ((x - y) * COS30, (x + y) * 0.5 - z)


def _thin(items: list) -> list:
    if len(items) <= MAX_ITEMS:
        return items
    step = len(items) / MAX_ITEMS
    return [items[int(i * step)] for i in range(MAX_ITEMS)]


def _r(v: float) -> float:
    return round(v, 3)


def model_sketch(model) -> Dict[str, list]:
    """The model drawn in an isometric view: lists of projected segments (x1, y1,
    x2, y2), polygons [x1, y1, x2, y2, ...] and points (x, y) by layer."""
    pts = model.points
    out: Dict[str, list] = {k: [] for k in ("grid", "bubble", "floor", "wall", "beam", "column", "brace", "support")}
    zs = [s.elevation for s in model.stories.values()] or [min((p.z for p in pts.values()), default=0.0)]
    zb = min(zs)
    xs = sorted({g.ordinate for g in model.grid_lines if g.direction == "X" and g.visible})
    ys = sorted({g.ordinate for g in model.grid_lines if g.direction == "Y" and g.visible})
    if xs and ys:
        span = max(xs[-1] - xs[0], ys[-1] - ys[0], 1e-9)
        ext, gap = 0.08 * span, 0.05 * span
        for x in xs:
            a, b = _proj(x, ys[0] - ext, zb), _proj(x, ys[-1] + ext, zb)
            out["grid"].append([_r(a[0]), _r(a[1]), _r(b[0]), _r(b[1])])
            c = _proj(x, ys[-1] + ext + gap, zb)
            out["bubble"].append([_r(c[0]), _r(c[1])])
        for y in ys:
            a, b = _proj(xs[0] - ext, y, zb), _proj(xs[-1] + ext, y, zb)
            out["grid"].append([_r(a[0]), _r(a[1]), _r(b[0]), _r(b[1])])
            c = _proj(xs[-1] + ext + gap, y, zb)
            out["bubble"].append([_r(c[0]), _r(c[1])])
    areas = sorted(model.areas.values(), key=lambda a: min(pts[n].z for n in a.points if n in pts) if a.points else 0.0)
    for a in areas:
        if any(n not in pts for n in a.points):
            continue
        poly = []
        for n in a.points:
            p = pts[n]
            u, v = _proj(p.x, p.y, p.z)
            poly += [_r(u), _r(v)]
        kind = (a.type or "").lower()
        out["wall" if kind == "wall" else "floor"].append(poly)
    for f in model.frames.values():
        pi, pj = pts.get(f.point_i), pts.get(f.point_j)
        if pi is None or pj is None:
            continue
        kind = (f.type or "").lower()
        if kind not in ("beam", "column", "brace"):
            dz, dl = abs(pj.z - pi.z), math.hypot(pj.x - pi.x, pj.y - pi.y)
            kind = "column" if dl < 1e-6 else ("beam" if dz < 1e-6 else "brace")
        a, b = _proj(pi.x, pi.y, pi.z), _proj(pj.x, pj.y, pj.z)
        out[kind].append([_r(a[0]), _r(a[1]), _r(b[0]), _r(b[1])])
    for p in pts.values():
        if any(p.restraints or ()) or any(p.springs or ()) or any(p.rotational_springs or ()):
            u, v = _proj(p.x, p.y, p.z)
            out["support"].append([_r(u), _r(v)])
    return {k: _thin(v) for k, v in out.items()}


def _bounds(sk: Dict[str, list]):
    xs, ys = [], []
    for k, items in sk.items():
        for it in items:
            xs += it[0::2]
            ys += it[1::2]
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _qc(rgb, alpha: float = 1.0) -> QtGui.QColor:
    c = QtGui.QColor.fromRgbF(*rgb)
    c.setAlphaF(alpha)
    return c


def paint_sketch(p: QtGui.QPainter, rect: QtCore.QRectF, sk: Optional[Dict[str, list]], colors: Dict[str, tuple],
                 background: bool = True, pad: float = 10.0) -> None:
    """Draw a :func:`model_sketch` fitted into ``rect`` with the theme ``colors``."""
    p.save()
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    if background:
        p.fillRect(rect, _qc(colors.get("background_top", (1, 1, 1))))
    b = _bounds(sk or {})
    if b is None:
        p.restore()
        return
    x0, y0, x1, y1 = b
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    k = min((rect.width() - 2 * pad) / w, (rect.height() - 2 * pad) / h)
    ox = rect.left() + (rect.width() - w * k) / 2 - x0 * k
    oy = rect.top() + (rect.height() - h * k) / 2 - y0 * k

    def P(u, v):
        return QtCore.QPointF(ox + u * k, oy + v * k)

    def lines(key, color, width, style=QtCore.Qt.SolidLine, alpha=1.0):
        items = sk.get(key) or []
        if not items:
            return
        pen = QtGui.QPen(_qc(color, alpha), width)
        pen.setStyle(style)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.drawLines([QtCore.QLineF(P(a, b), P(c, d)) for a, b, c, d in items])

    def polys(key, fill, edge, fill_alpha, width):
        items = sk.get(key) or []
        if not items:
            return
        p.setPen(QtGui.QPen(_qc(edge, 0.9), width))
        p.setBrush(_qc(fill, fill_alpha))
        for it in items:
            p.drawPolygon(QtGui.QPolygonF([P(it[i], it[i + 1]) for i in range(0, len(it) - 1, 2)]))

    many = sum(len(sk.get(n) or []) for n in ("beam", "column", "brace"))
    thin = 0.6 if many > 1500 else 1.0
    grid = colors.get("grid", (0.66, 0.69, 0.74))
    lines("grid", grid, 0.8, QtCore.Qt.DashLine)
    bub = sk.get("bubble") or []
    if bub:
        r = max(2.5, min(5.0, 0.018 * min(rect.width(), rect.height())))
        p.setPen(QtGui.QPen(_qc(colors.get("grid_text", grid)), 0.8))
        p.setBrush(_qc(colors.get("background_top", (1, 1, 1))))
        for u, v in bub:
            p.drawEllipse(P(u, v), r, r)
    polys("floor", colors.get("floor", (0.66, 0.76, 0.90)), colors.get("area_edge", (0.22, 0.28, 0.40)), 0.35,
          0.6 * thin)
    lines("beam", colors.get("beam", (0.02, 0.30, 0.80)), 1.5 * thin)
    lines("brace", colors.get("brace", (0.5, 0.2, 0.78)), 1.5 * thin)
    lines("column", colors.get("column", (0.88, 0.30, 0.0)), 2.2 * thin)
    sup = sk.get("support") or []
    if sup:
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(_qc(colors.get("support", (0.0, 0.4, 0.13))))
        s = max(1.4, min(4.0, 0.025 * min(rect.width(), rect.height())))
        if len(sup) > 200:
            s *= 0.65
        for u, v in sup:
            c = P(u, v)
            p.drawPolygon(QtGui.QPolygonF([c, c + QtCore.QPointF(-s, 1.5 * s), c + QtCore.QPointF(s, 1.5 * s)]))
    wall = colors.get("wall", (0.62, 0.32, 0.34))
    polys("wall", wall, tuple(0.65 * c for c in wall), 0.88, 0.7 * thin)
    p.restore()


def sketch_pixmap(sk: Optional[Dict[str, list]], size: QtCore.QSize, colors: Dict[str, tuple],
                  ratio: float = 1.0) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(int(size.width() * ratio), int(size.height() * ratio))
    pm.setDevicePixelRatio(ratio)
    pm.fill(_qc(colors.get("background_top", (1, 1, 1))))
    p = QtGui.QPainter(pm)
    paint_sketch(p, QtCore.QRectF(0, 0, size.width(), size.height()), sk, colors)
    p.end()
    return pm


def sketch_items(sk: Optional[Dict[str, list]]) -> List[str]:
    """Layers that have something in them (for tests)."""
    return [k for k, v in (sk or {}).items() if v]
