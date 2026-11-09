"""Toolbar icons drawn with QPainter (no image assets).

Colours follow the viewport palette: beams blue, columns orange, braces
green, slabs grey-blue, walls tan, selection yellow.
"""
from __future__ import annotations

from functools import lru_cache

from PySide6 import QtCore, QtGui

S = 32  # logical icon size
INK = QtGui.QColor(40, 44, 52)
BEAM = QtGui.QColor(60, 130, 210)
COLUMN = QtGui.QColor(230, 145, 40)
BRACE = QtGui.QColor(90, 170, 80)
SLAB = QtGui.QColor(150, 170, 195)
WALL = QtGui.QColor(200, 170, 130)
GRID = QtGui.QColor(150, 155, 165)
ACCENT = QtGui.QColor(250, 200, 40)
RED = QtGui.QColor(215, 70, 60)


def _pen(color, w=2.0, style=QtCore.Qt.SolidLine):
    p = QtGui.QPen(color, w, style)
    p.setCapStyle(QtCore.Qt.RoundCap)
    p.setJoinStyle(QtCore.Qt.RoundJoin)
    return p


def _grid(p: QtGui.QPainter):
    p.setPen(_pen(GRID, 1.0, QtCore.Qt.DashLine))
    for v in (6, 26):
        p.drawLine(QtCore.QPointF(v, 2), QtCore.QPointF(v, 30))
        p.drawLine(QtCore.QPointF(2, v), QtCore.QPointF(30, v))


def _dot(p, x, y, color=INK, r=2.2):
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(color)
    p.drawEllipse(QtCore.QPointF(x, y), r, r)


def _wf(p: QtGui.QPainter, cx: float, cy: float, w: float, h: float, t: float, outline: bool = True):
    """Wide-flange (I / H) cross-section centred on (cx, cy): flanges along X."""
    q = QtCore.QPointF
    hw, hh, ht = w / 2.0, h / 2.0, t / 2.0
    poly = QtGui.QPolygonF([q(cx - hw, cy - hh), q(cx + hw, cy - hh), q(cx + hw, cy - hh + t), q(cx + ht, cy - hh + t),
                            q(cx + ht, cy + hh - t), q(cx + hw, cy + hh - t), q(cx + hw, cy + hh), q(cx - hw, cy + hh),
                            q(cx - hw, cy + hh - t), q(cx - ht, cy + hh - t), q(cx - ht, cy - hh + t),
                            q(cx - hw, cy - hh + t)])
    p.setPen(_pen(COLUMN.darker(135), 1.2) if outline else QtCore.Qt.NoPen)
    p.setBrush(COLUMN)
    p.drawPolygon(poly)


def _bolt(p):
    """Small lightning mark for 'quick' tools."""
    p.setPen(_pen(ACCENT, 1.0))
    p.setBrush(ACCENT)
    p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*xy) for xy in ((24, 1), (19, 9), (23, 9), (20, 16), (28, 6), (24, 6), (27, 1))]))


def _arrow(p):
    path = QtGui.QPainterPath()
    pts = [(7, 4), (7, 25), (12, 20), (16, 28), (19, 27), (15, 19), (22, 19)]
    path.moveTo(*pts[0])
    for xy in pts[1:]:
        path.lineTo(*xy)
    path.closeSubpath()
    p.setPen(_pen(INK, 1.5))
    p.setBrush(QtGui.QColor(255, 255, 255))
    p.drawPath(path)


def _draw(name: str, p: QtGui.QPainter):
    q = QtCore.QPointF
    if name == "select":
        _arrow(p)
    elif name == "check_model":
        # a small frame with a check mark: Analyze > Check Model
        p.setPen(_pen(INK, 1.6))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawLine(q(5, 27), q(5, 9))
        p.drawLine(q(19, 27), q(19, 9))
        p.drawLine(q(3, 9), q(21, 9))
        p.drawLine(q(3, 27), q(21, 27))
        p.setPen(_pen(QtGui.QColor(40, 150, 80), 3.0))
        p.drawLine(q(15, 19), q(20, 25))
        p.drawLine(q(20, 25), q(29, 12))
    elif name == "check_mesh":
        # a slab mesh getting finer (coarse on the left, fine on the right) under a magnifier
        p.setPen(_pen(SLAB.darker(140), 1.0))
        p.setBrush(SLAB.lighter(115))
        p.drawRect(QtCore.QRectF(3, 6, 22, 20))
        for x in (14,):
            p.drawLine(q(x, 6), q(x, 26))
        p.drawLine(q(3, 16), q(14, 16))
        for x in (19.5,):
            p.drawLine(q(x, 6), q(x, 26))
        for y in (11, 16, 21):
            p.drawLine(q(14, y), q(25, y))
        p.setPen(_pen(INK, 2.2))
        p.setBrush(QtGui.QColor(255, 255, 255, 120))
        p.drawEllipse(q(22, 13), 6, 6)
        p.drawLine(q(26.5, 17.5), q(30, 21))
    elif name in ("lock", "unlock"):
        # padlock (ETABS lock button): closed = analysed model locked, open = editable
        p.setPen(_pen(INK, 2.4))
        p.setBrush(QtCore.Qt.NoBrush)
        path = QtGui.QPainterPath()
        if name == "lock":
            path.moveTo(10, 15)
            path.lineTo(10, 10)
            path.arcTo(QtCore.QRectF(10, 4, 12, 12), 180, -180)
            path.lineTo(22, 15)
        else:
            path.moveTo(10, 15)
            path.lineTo(10, 8)
            path.arcTo(QtCore.QRectF(10, 2, 12, 12), 180, -180)
            path.lineTo(22, 9)
        p.drawPath(path)
        p.setPen(_pen(INK, 1.2))
        p.setBrush(QtGui.QColor(230, 160, 40) if name == "lock" else QtGui.QColor(120, 190, 120))
        p.drawRoundedRect(QtCore.QRectF(6, 14, 20, 15), 3, 3)
        _dot(p, 16, 21, INK, 2.0)
    elif name == "theme":
        # half sun / half moon: dark <-> light display
        p.setPen(_pen(INK, 1.4))
        p.setBrush(QtGui.QColor(250, 200, 40))
        p.drawEllipse(q(16, 16), 9, 9)
        path = QtGui.QPainterPath()
        path.moveTo(16, 7)
        path.arcTo(QtCore.QRectF(7, 7, 18, 18), 90, 180)
        path.closeSubpath()
        p.setBrush(QtGui.QColor(45, 50, 62))
        p.drawPath(path)
        for ang in range(0, 360, 45):
            import math
            c, sn = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            if c > 0.1 or ang in (90, 270):
                p.drawLine(q(16 + 11.5 * c, 16 + 11.5 * sn), q(16 + 14 * c, 16 + 14 * sn))
    elif name == "display":
        # an eye: set display options
        path = QtGui.QPainterPath()
        path.moveTo(3, 16)
        path.quadTo(16, 4, 29, 16)
        path.quadTo(16, 28, 3, 16)
        p.setPen(_pen(INK, 1.6))
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawPath(path)
        p.setBrush(BEAM)
        p.drawEllipse(q(16, 16), 5, 5)
        p.setBrush(INK)
        p.drawEllipse(q(16, 16), 2, 2)
    elif name == "extrude":
        # an I-beam in perspective: section at the front, extruded to the back
        p.setPen(_pen(INK, 1.2))
        p.setBrush(BEAM.lighter(130))
        back = [(14, 4), (28, 4), (28, 8), (23, 8), (23, 17), (28, 17), (28, 21), (14, 21), (14, 17), (19, 17),
                (19, 8), (14, 8)]
        front = [(x - 10, y + 8) for x, y in back]
        for (x0, y0), (x1, y1) in zip(front, back):
            p.drawLine(q(x0, y0), q(x1, y1))
        p.setBrush(BEAM)
        p.drawPolygon(QtGui.QPolygonF([q(x, y) for x, y in front]))
    elif name == "colors":
        for k, col in enumerate((BEAM, COLUMN, BRACE, WALL)):
            p.setPen(_pen(INK, 1.0))
            p.setBrush(col)
            p.drawRect(QtCore.QRectF(4 + (k % 2) * 13, 4 + (k // 2) * 13, 11, 11))
    elif name == "select_window":
        p.setPen(_pen(INK, 1.4, QtCore.Qt.DashLine))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(3, 3, 20, 16))
        p.save()
        p.translate(8, 6)
        p.scale(0.75, 0.75)
        _arrow(p)
        p.restore()
    elif name == "select_line":
        p.setPen(_pen(BEAM, 3))
        p.drawLine(q(4, 10), q(28, 10))
        p.drawLine(q(4, 22), q(28, 22))
        p.setPen(_pen(RED, 1.6, QtCore.Qt.DashLine))
        p.drawLine(q(10, 3), q(22, 29))
    elif name == "point":
        _grid(p)
        _dot(p, 16, 16, RED, 4)
    elif name == "beam":
        _grid(p)
        p.setPen(_pen(BEAM, 3.5))
        p.drawLine(q(6, 26), q(26, 6))
        _dot(p, 6, 26)
        _dot(p, 26, 6)
    elif name == "quick_beam":
        _grid(p)
        p.setPen(_pen(BEAM, 4))
        p.drawLine(q(6, 26), q(26, 26))
        _bolt(p)
    elif name == "secondary_beams":
        p.setPen(_pen(BEAM, 3))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(5, 6, 22, 20))
        p.setPen(_pen(BEAM, 1.8))
        for y in (12.7, 19.3):
            p.drawLine(q(5, y), q(27, y))
    elif name == "column":
        # wide-flange section seen from above: the column is drawn in plan
        _wf(p, 16, 16, 20, 20, 3.6)
        p.setPen(_pen(INK, 1.0, QtCore.Qt.DashLine))
        p.drawLine(q(3, 16), q(29, 16))
        p.drawLine(q(16, 3), q(16, 29))
    elif name == "quick_column":
        _grid(p)
        for cx, cy in ((6, 6), (26, 6), (6, 26)):
            _wf(p, cx, cy, 8, 8, 1.7, outline=False)
        _bolt(p)
    elif name == "brace":
        p.setPen(_pen(INK, 1.5))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(5, 5, 22, 22))
        p.setPen(_pen(BRACE, 3))
        p.drawLine(q(5, 27), q(27, 5))
    elif name == "quick_brace":
        p.setPen(_pen(INK, 1.5))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(4, 8, 22, 20))
        p.setPen(_pen(BRACE, 2.5))
        p.drawLine(q(4, 28), q(26, 8))
        p.drawLine(q(4, 8), q(26, 28))
        _bolt(p)
    elif name == "wall":
        p.setPen(_pen(INK, 1.2))
        p.setBrush(WALL)
        p.drawPolygon(QtGui.QPolygonF([q(5, 9), q(21, 4), q(21, 24), q(5, 29)]))
        p.setBrush(WALL.darker(120))
        p.drawPolygon(QtGui.QPolygonF([q(21, 4), q(27, 7), q(27, 27), q(21, 24)]))
    elif name == "quick_wall":
        p.setPen(_pen(INK, 1.2))
        p.setBrush(WALL)
        p.drawRect(QtCore.QRectF(4, 10, 20, 18))
        _bolt(p)
    elif name == "slab":
        p.setPen(_pen(INK, 1.4))
        p.setBrush(SLAB)
        p.drawPolygon(QtGui.QPolygonF([q(4, 12), q(14, 5), q(28, 10), q(24, 26), q(8, 27)]))
        for xy in ((4, 12), (14, 5), (28, 10), (24, 26), (8, 27)):
            _dot(p, *xy, r=1.8)
    elif name == "rect_slab":
        p.setPen(_pen(INK, 1.4))
        p.setBrush(SLAB)
        p.drawRect(QtCore.QRectF(5, 8, 22, 16))
        _dot(p, 5, 24, r=2)
        _dot(p, 27, 8, r=2)
    elif name == "quick_slab":
        _grid(p)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(SLAB)
        p.drawRect(QtCore.QRectF(6, 6, 20, 20))
        _bolt(p)
    elif name == "opening":
        p.setPen(_pen(INK, 1.4))
        p.setBrush(SLAB)
        p.drawRect(QtCore.QRectF(4, 4, 24, 24))
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.setPen(_pen(RED, 1.4, QtCore.Qt.DashLine))
        p.drawRect(QtCore.QRectF(11, 11, 10, 10))
    elif name == "move":
        p.setPen(_pen(INK, 2))
        p.drawLine(q(16, 4), q(16, 28))
        p.drawLine(q(4, 16), q(28, 16))
        for a, b, c in (((16, 3), (12, 8), (20, 8)), ((16, 29), (12, 24), (20, 24)),
                        ((3, 16), (8, 12), (8, 20)), ((29, 16), (24, 12), (24, 20))):
            p.setBrush(INK)
            p.drawPolygon(QtGui.QPolygonF([q(*a), q(*b), q(*c)]))
    elif name == "copy":
        p.setPen(_pen(BEAM, 3))
        p.drawLine(q(4, 22), q(16, 10))
        p.setPen(_pen(BEAM, 3, QtCore.Qt.DotLine))
        p.drawLine(q(14, 26), q(26, 14))
        p.setPen(_pen(INK, 1.6))
        p.drawLine(q(11, 17), q(20, 21))
        p.setBrush(INK)
        p.drawPolygon(QtGui.QPolygonF([q(22, 22), q(17, 23.5), q(19, 18.5)]))
        p.setPen(_pen(INK, 1.8))
        p.drawLine(q(25, 3), q(25, 9))
        p.drawLine(q(22, 6), q(28, 6))
    elif name == "rotate":
        p.setPen(_pen(BEAM, 3))
        p.drawLine(q(8, 24), q(27, 24))
        p.setPen(_pen(BEAM, 3, QtCore.Qt.DotLine))
        p.drawLine(q(8, 24), q(20, 7))
        p.setPen(_pen(INK, 1.6))
        path = QtGui.QPainterPath()
        path.moveTo(22, 23)
        path.arcTo(QtCore.QRectF(-6, 10, 28, 28), 0, 52)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(path)
        _dot(p, 8, 24, RED, 2.6)
    elif name == "mirror":
        p.setPen(_pen(INK, 1.5, QtCore.Qt.DashDotLine))
        p.drawLine(q(16, 2), q(16, 30))
        p.setPen(_pen(BEAM, 3))
        p.drawPolyline(QtGui.QPolygonF([q(13, 6), q(4, 16), q(13, 26)]))
        p.setPen(_pen(BEAM, 3, QtCore.Qt.DotLine))
        p.drawPolyline(QtGui.QPolygonF([q(19, 6), q(28, 16), q(19, 26)]))
    elif name == "offset":
        p.setPen(_pen(BEAM, 3))
        p.drawLine(q(4, 10), q(28, 10))
        p.setPen(_pen(BEAM, 3, QtCore.Qt.DotLine))
        p.drawLine(q(4, 24), q(28, 24))
        p.setPen(_pen(INK, 1.4))
        p.drawLine(q(16, 12), q(16, 22))
        p.setBrush(INK)
        p.drawPolygon(QtGui.QPolygonF([q(16, 12), q(13.5, 16), q(18.5, 16)]))
        p.drawPolygon(QtGui.QPolygonF([q(16, 22), q(13.5, 18), q(18.5, 18)]))
    elif name == "trim":
        p.setPen(_pen(INK, 2))
        p.drawLine(q(20, 3), q(20, 29))
        p.setPen(_pen(BEAM, 3))
        p.drawLine(q(3, 16), q(20, 16))
        p.setPen(_pen(RED, 2.5, QtCore.Qt.DotLine))
        p.drawLine(q(21, 16), q(30, 16))
        p.setPen(_pen(RED, 2))
        p.drawLine(q(23, 21), q(28, 26))
        p.drawLine(q(28, 21), q(23, 26))
    elif name == "extend":
        p.setPen(_pen(INK, 2))
        p.drawLine(q(27, 3), q(27, 29))
        p.setPen(_pen(BEAM, 3))
        p.drawLine(q(3, 16), q(15, 16))
        p.setPen(_pen(BEAM, 3, QtCore.Qt.DotLine))
        p.drawLine(q(15, 16), q(26, 16))
        p.setBrush(INK)
        p.setPen(_pen(INK, 1))
        p.drawPolygon(QtGui.QPolygonF([q(26, 16), q(21, 12.5), q(21, 19.5)]))
    elif name == "replicate":
        p.setPen(_pen(BEAM, 3))
        p.drawLine(q(4, 10), q(18, 10))
        p.setPen(_pen(BEAM, 3, QtCore.Qt.DotLine))
        p.drawLine(q(12, 22), q(26, 22))
        p.setPen(_pen(INK, 1.5))
        p.drawLine(q(18, 12), q(22, 18))
    elif name == "divide":
        p.setPen(_pen(BEAM, 3))
        p.drawLine(q(3, 16), q(29, 16))
        for x in (11.7, 20.3):
            _dot(p, x, 16, RED, 3)
    elif name == "delete":
        p.setPen(_pen(RED, 3.5))
        p.drawLine(q(8, 8), q(24, 24))
        p.drawLine(q(24, 8), q(8, 24))
    elif name == "undo" or name == "redo":
        p.setPen(_pen(INK, 2.5))
        path = QtGui.QPainterPath()
        if name == "undo":
            path.moveTo(9, 13)
            path.cubicTo(14, 6, 27, 8, 26, 20)
            p.drawPath(path)
            p.setBrush(INK)
            p.drawPolygon(QtGui.QPolygonF([q(4, 12), q(12, 7), q(12, 17)]))
        else:
            path.moveTo(23, 13)
            path.cubicTo(18, 6, 5, 8, 6, 20)
            p.drawPath(path)
            p.setBrush(INK)
            p.drawPolygon(QtGui.QPolygonF([q(28, 12), q(20, 7), q(20, 17)]))
    elif name == "run":
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(60, 170, 90))
        p.drawPolygon(QtGui.QPolygonF([q(9, 5), q(27, 16), q(9, 27)]))
    elif name.startswith("view_"):
        kind = name[5:]
        p.setPen(_pen(INK, 1.4))
        p.setBrush(QtCore.Qt.NoBrush)
        if kind == "3d":
            p.drawPolygon(QtGui.QPolygonF([q(6, 11), q(18, 6), q(27, 10), q(15, 15)]))
            p.drawLine(q(6, 11), q(6, 24))
            p.drawLine(q(15, 15), q(15, 28))
            p.drawLine(q(27, 10), q(27, 23))
            p.drawLine(q(6, 24), q(15, 28))
            p.drawLine(q(15, 28), q(27, 23))
        elif kind == "plan":
            _grid(p)
            p.setPen(_pen(BEAM, 2.5))
            p.drawRect(QtCore.QRectF(6, 6, 20, 20))
        else:
            p.setPen(_pen(COLUMN, 2.5))
            p.drawLine(q(7, 28), q(7, 6))
            p.drawLine(q(25, 28), q(25, 6))
            p.setPen(_pen(BEAM, 2.5))
            p.drawLine(q(7, 6), q(25, 6))
            p.drawLine(q(7, 17), q(25, 17))
            p.setPen(_pen(INK, 1.2))
            p.drawText(QtCore.QRectF(0, 18, 32, 12), QtCore.Qt.AlignCenter, kind.upper())
    elif name.startswith("layout_"):
        n = name[7:]
        p.setPen(_pen(INK, 1.5))
        p.setBrush(QtGui.QColor(225, 230, 238))
        rects = {
            "1": [(3, 5, 26, 22)],
            "2": [(3, 5, 12.5, 22), (16.5, 5, 12.5, 22)],
            "2v": [(3, 5, 26, 10.5), (3, 16.5, 26, 10.5)],
            "3": [(3, 5, 12.5, 22), (16.5, 5, 12.5, 10.5), (16.5, 16.5, 12.5, 10.5)],
            "4": [(3, 5, 12.5, 10.5), (16.5, 5, 12.5, 10.5), (3, 16.5, 12.5, 10.5), (16.5, 16.5, 12.5, 10.5)],
        }.get(n)
        if rects is None and n.isdigit():  # n windows in a grid (6, 9 ...)
            import math as _m
            k = int(n)
            cols = max(1, _m.ceil(_m.sqrt(k)))
            rows = max(1, _m.ceil(k / cols))
            w, h = (26 - (cols - 1) * 2) / cols, (22 - (rows - 1) * 2) / rows
            rects = [(3 + c * (w + 2), 5 + r * (h + 2), w, h) for r in range(rows) for c in range(cols)
                     if r * cols + c < k]
        rects = rects or [(3, 5, 26, 22)]
        for r in rects:
            p.drawRect(QtCore.QRectF(*r))
    elif name.startswith("snap_"):
        kind = name[5:]
        p.setPen(_pen(INK, 1.4))
        p.setBrush(QtCore.Qt.NoBrush)
        if kind == "grid":
            _grid(p)
            p.setPen(_pen(ACCENT.darker(130), 2))
            p.drawRect(QtCore.QRectF(21, 21, 10, 10))
        elif kind == "point":
            p.setPen(_pen(ACCENT.darker(130), 2))
            p.drawRect(QtCore.QRectF(10, 10, 12, 12))
            _dot(p, 16, 16)
        elif kind == "mid":
            p.setPen(_pen(BEAM, 3))
            p.drawLine(q(3, 22), q(29, 22))
            p.setPen(_pen(ACCENT.darker(130), 2))
            p.drawPolygon(QtGui.QPolygonF([q(16, 14), q(22, 24), q(10, 24)]))
        elif kind == "line":  # nearest point on a line: hourglass on the line
            p.setPen(_pen(BEAM, 3))
            p.drawLine(q(3, 10), q(29, 24))
            p.setPen(_pen(ACCENT.darker(130), 2))
            p.drawPolygon(QtGui.QPolygonF([q(11, 11), q(21, 11), q(11, 23), q(21, 23)]))
        elif kind == "perp":  # perpendicular projection: right-angle mark
            p.setPen(_pen(BEAM, 3))
            p.drawLine(q(3, 26), q(29, 26))
            p.setPen(_pen(ACCENT.darker(130), 2))
            p.drawLine(q(16, 4), q(16, 26))
            p.drawPolyline(QtGui.QPolygonF([q(16, 18), q(24, 18), q(24, 26)]))
        elif kind == "int":  # intersection: crossing lines + X mark
            p.setPen(_pen(BEAM, 3))
            p.drawLine(q(4, 6), q(28, 26))
            p.drawLine(q(4, 26), q(28, 6))
            p.setPen(_pen(ACCENT.darker(130), 2))
            p.drawLine(q(11, 11), q(21, 21))
            p.drawLine(q(11, 21), q(21, 11))
        elif kind == "ortho":  # orthogonal tracking: L of dashed axes
            p.setPen(_pen(ACCENT.darker(130), 2, QtCore.Qt.DashLine))
            p.drawLine(q(6, 26), q(28, 26))
            p.drawLine(q(6, 26), q(6, 4))
            _dot(p, 6, 26)
        else:  # options: a small form
            p.setPen(_pen(INK, 1.3))
            p.setBrush(QtGui.QColor(255, 255, 255))
            p.drawRect(QtCore.QRectF(4, 4, 24, 24))
            for k, y in enumerate((10, 16, 22)):
                p.drawRect(QtCore.QRectF(7, y - 2, 4, 4))
                if k != 1:
                    p.drawLine(q(8, y), q(10, y))
                p.drawLine(q(14, y), q(25, y))
    elif name == "panel_load":
        # a dashed panel with a field of small load arrows
        p.setPen(_pen(ACCENT.darker(120), 1.4, QtCore.Qt.DashLine))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPolygon(QtGui.QPolygonF([q(4, 24), q(14, 8), q(29, 8), q(19, 24)]))
        p.setPen(_pen(INK, 1.3))
        for x in (10, 16, 22):
            p.drawLine(q(x, 12), q(x, 20))
            p.drawLine(q(x, 20), q(x - 2, 17))
            p.drawLine(q(x, 20), q(x + 2, 17))
    elif name in ("joint_load", "frame_load", "area_load"):
        LOADC = QtGui.QColor(200, 60, 50)

        def arrow(x, y0, y1):
            p.setPen(_pen(LOADC, 1.6))
            p.drawLine(q(x, y0), q(x, y1))
            p.drawLine(q(x, y1), q(x - 2.5, y1 - 3.5))
            p.drawLine(q(x, y1), q(x + 2.5, y1 - 3.5))

        if name == "joint_load":  # a joint with a concentrated force on it
            p.setPen(_pen(BEAM, 3))
            p.drawLine(q(3, 26), q(29, 26))
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(INK)
            p.drawEllipse(q(16, 26), 2.6, 2.6)
            arrow(16, 3, 22)
        elif name == "frame_load":  # a beam under a distributed load
            p.setPen(_pen(BEAM, 3.5))
            p.drawLine(q(3, 26), q(29, 26))
            p.setPen(_pen(LOADC, 1.2))
            p.drawLine(q(5, 8), q(27, 8))
            for x in (6, 12, 18, 24):
                arrow(x + 1, 8, 22)
        else:  # a floor under a surface load (becomes a panel on the area)
            p.setPen(_pen(INK, 1.2))
            p.setBrush(SLAB)
            p.drawPolygon(QtGui.QPolygonF([q(3, 28), q(11, 18), q(29, 18), q(21, 28)]))
            for x in (10, 16, 22):
                arrow(x, 3, 20)
    elif name == "palette":
        # a column of tool buttons, one with a drop-down arrow: compact <-> full palette
        p.setPen(_pen(INK, 1.3))
        p.setBrush(QtGui.QColor(255, 255, 255))
        for y in (3, 12, 21):
            p.drawRect(QtCore.QRectF(4, y, 9, 8))
        p.setBrush(BEAM)
        p.drawRect(QtCore.QRectF(17, 3, 11, 26))
        p.setBrush(INK)
        p.drawPolygon(QtGui.QPolygonF([q(19, 14), q(26, 14), q(22.5, 19)]))
    elif name == "rotate3d":
        # orbit arrows around a small box
        p.setPen(_pen(INK, 1.4))
        p.setBrush(SLAB)
        p.drawPolygon(QtGui.QPolygonF([q(11, 13), q(18, 10), q(23, 13), q(16, 16)]))
        p.setBrush(SLAB.darker(115))
        p.drawPolygon(QtGui.QPolygonF([q(11, 13), q(16, 16), q(16, 23), q(11, 20)]))
        p.drawPolygon(QtGui.QPolygonF([q(16, 16), q(23, 13), q(23, 20), q(16, 23)]))
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(_pen(BEAM, 2))
        p.drawArc(QtCore.QRectF(3, 5, 26, 22), 200 * 16, 140 * 16)
        p.setBrush(BEAM)
        p.drawPolygon(QtGui.QPolygonF([q(27, 21), q(29, 14), q(23, 17)]))
    elif name == "pan":
        p.setPen(_pen(INK, 1.6))
        p.drawLine(q(16, 3), q(16, 29))
        p.drawLine(q(3, 16), q(29, 16))
        p.setBrush(INK)
        for tri in (((16, 2), (12, 7), (20, 7)), ((16, 30), (12, 25), (20, 25)),
                    ((2, 16), (7, 12), (7, 20)), ((30, 16), (25, 12), (25, 20))):
            p.drawPolygon(QtGui.QPolygonF([q(*xy) for xy in tri]))
    elif name in ("zoom_in", "zoom_out", "zoom_window", "zoom_full"):
        p.setPen(_pen(INK, 1.8))
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawEllipse(q(13, 13), 9, 9)
        p.setPen(_pen(INK, 3.2))
        p.drawLine(q(20, 20), q(28, 28))
        p.setPen(_pen(BEAM, 2))
        if name in ("zoom_in", "zoom_out"):
            p.drawLine(q(9, 13), q(17, 13))
            if name == "zoom_in":
                p.drawLine(q(13, 9), q(13, 17))
        elif name == "zoom_window":
            p.setPen(_pen(BEAM, 1.4, QtCore.Qt.DashLine))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(QtCore.QRectF(8.5, 9.5, 9, 7))
        else:
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(QtCore.QRectF(9, 9, 8, 8))
            p.setPen(_pen(INK, 1.2))
            for a, b in (((9, 9), (6, 6)), ((17, 9), (20, 6)), ((9, 17), (6, 20))):
                p.drawLine(q(*a), q(*b))
    elif name == "grid":
        p.setPen(_pen(GRID.darker(120), 1.4, QtCore.Qt.DashLine))
        for v in (8, 16, 24):
            p.drawLine(q(v, 6), q(v, 30))
            p.drawLine(q(2, v), q(26, v))
        p.setPen(_pen(INK, 1.2))
        p.setBrush(QtGui.QColor(255, 255, 255))
        for v in (8, 16, 24):
            p.drawEllipse(q(v, 4), 3, 3)
            p.drawEllipse(q(28, v), 3, 3)
    elif name in ("select_all", "select_previous", "select_clear", "palette"):
        p.setPen(_pen(ACCENT.darker(140), 1.6, QtCore.Qt.DashLine))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(4, 4, 24, 24))
        p.setPen(_pen(BEAM, 2.5))
        p.drawLine(q(8, 22), q(24, 22))
        p.setPen(_pen(COLUMN, 2.5))
        p.drawLine(q(10, 22), q(10, 9))
        p.drawLine(q(22, 22), q(22, 9))
        if name == "select_all":
            p.setPen(_pen(ACCENT.darker(120), 2.5))
            p.drawLine(q(8, 9), q(24, 9))
        elif name == "select_previous":
            p.setPen(_pen(INK, 1.8))
            p.drawArc(QtCore.QRectF(10, 6, 12, 10), 0, 200 * 16)
            p.setBrush(INK)
            p.drawPolygon(QtGui.QPolygonF([q(8, 9), q(13, 11), q(10, 15)]))
        else:
            p.setPen(_pen(RED, 2.4))
            p.drawLine(q(18, 18), q(29, 29))
            p.drawLine(q(29, 18), q(18, 29))
    elif name == "tables":
        # a results table: header row + grid
        p.setPen(_pen(INK, 1.4))
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawRect(QtCore.QRectF(4, 6, 24, 20))
        p.setBrush(ACCENT)
        p.drawRect(QtCore.QRectF(4, 6, 24, 5))
        for y in (16, 21):
            p.drawLine(q(4, y), q(28, y))
        for x in (12, 20):
            p.drawLine(q(x, 6), q(x, 26))
    elif name == "properties_panel":
        # a window with its right-hand panel highlighted
        p.setPen(_pen(INK, 1.4))
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawRect(QtCore.QRectF(3, 6, 26, 20))
        p.setBrush(ACCENT)
        p.drawRect(QtCore.QRectF(20, 6, 9, 20))
        p.setPen(_pen(INK, 1.1))
        for y in (11, 15, 19):
            p.drawLine(q(22, y), q(27, y))
    elif name in ("file_new", "file_open", "file_save"):
        if name == "file_open":  # a folder
            p.setPen(_pen(INK, 1.4))
            p.setBrush(ACCENT)
            p.drawPolygon(QtGui.QPolygonF([q(3, 9), q(12, 9), q(14, 12), q(29, 12), q(29, 26), q(3, 26)]))
            p.setBrush(QtGui.QColor(255, 236, 150))
            p.drawPolygon(QtGui.QPolygonF([q(6, 15), q(31, 15), q(28, 26), q(3, 26)]))
        elif name == "file_save":  # a disk
            p.setPen(_pen(INK, 1.4))
            p.setBrush(BEAM)
            p.drawRoundedRect(QtCore.QRectF(5, 5, 22, 22), 2, 2)
            p.setBrush(QtGui.QColor(255, 255, 255))
            p.drawRect(QtCore.QRectF(10, 5, 12, 8))
            p.drawRect(QtCore.QRectF(9, 18, 14, 9))
        else:  # a sheet with a folded corner
            p.setPen(_pen(INK, 1.4))
            p.setBrush(QtGui.QColor(255, 255, 255))
            p.drawPolygon(QtGui.QPolygonF([q(8, 4), q(20, 4), q(26, 10), q(26, 28), q(8, 28)]))
            p.drawPolyline(QtGui.QPolygonF([q(20, 4), q(20, 10), q(26, 10)]))
            p.setPen(_pen(QtGui.QColor(60, 170, 90), 2.4))
            p.drawLine(q(17, 15), q(17, 23))
            p.drawLine(q(13, 19), q(21, 19))
    elif name == "materials":  # a stress-strain curve over a block of material
        p.setPen(_pen(INK, 1.3))
        p.setBrush(SLAB)
        p.drawRect(QtCore.QRectF(4, 18, 24, 10))
        p.setPen(_pen(RED, 2.2))
        path = QtGui.QPainterPath()
        path.moveTo(5, 16)
        path.cubicTo(9, 4, 14, 3, 27, 4)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(path)
        p.setPen(_pen(INK, 1.2))
        p.drawLine(q(5, 3), q(5, 16))
    elif name == "frame_sections":  # a rectangular and an I section
        p.setPen(_pen(BEAM.darker(130), 1.2))
        p.setBrush(BEAM)
        p.drawRect(QtCore.QRectF(3, 7, 10, 18))
        _wf(p, 22, 16, 13, 18, 3.2)
    elif name in ("slab_sections", "wall_sections"):  # a plate with its thickness
        col = SLAB if name == "slab_sections" else WALL
        p.setPen(_pen(col.darker(140), 1.2))
        p.setBrush(col)
        if name == "slab_sections":
            p.drawPolygon(QtGui.QPolygonF([q(3, 14), q(19, 8), q(29, 13), q(13, 20)]))
            p.setBrush(col.darker(120))
            p.drawPolygon(QtGui.QPolygonF([q(3, 14), q(13, 20), q(13, 24), q(3, 18)]))
            p.drawPolygon(QtGui.QPolygonF([q(13, 20), q(29, 13), q(29, 17), q(13, 24)]))
        else:
            p.drawPolygon(QtGui.QPolygonF([q(8, 5), q(22, 9), q(22, 28), q(8, 24)]))
            p.setBrush(col.darker(120))
            p.drawPolygon(QtGui.QPolygonF([q(22, 9), q(26, 7), q(26, 26), q(22, 28)]))
            p.drawPolygon(QtGui.QPolygonF([q(8, 5), q(12, 3), q(26, 7), q(22, 9)]))
    elif name in ("load_patterns", "load_cases", "combos"):
        p.setPen(_pen(INK, 1.3))
        p.setBrush(QtGui.QColor(255, 255, 255))
        if name == "load_patterns":  # arrows down on a line
            p.setPen(_pen(RED, 1.8))
            for x in (8, 16, 24):
                p.drawLine(q(x, 5), q(x, 19))
                p.setBrush(RED)
                p.drawPolygon(QtGui.QPolygonF([q(x - 3, 16), q(x + 3, 16), q(x, 21)]))
            p.setPen(_pen(BEAM, 3.0))
            p.drawLine(q(4, 25), q(28, 25))
        elif name == "load_cases":  # a list of cases
            p.drawRect(QtCore.QRectF(5, 5, 22, 22))
            for y in (11, 16, 21):
                _dot(p, 9, y, RED, 1.6)
                p.setPen(_pen(INK, 1.3))
                p.drawLine(q(12, y), q(24, y))
        else:  # combinations: 1.2 D + 1.6 L
            p.setPen(_pen(INK, 1.4))
            p.drawRect(QtCore.QRectF(3, 8, 11, 16))
            p.drawRect(QtCore.QRectF(18, 8, 11, 16))
            p.setPen(_pen(QtGui.QColor(60, 170, 90), 2.4))
            p.drawLine(q(16, 12), q(16, 20))
            p.drawLine(q(13.5, 16), q(18.5, 16))
            p.setPen(_pen(RED, 1.8))
            p.drawLine(q(8.5, 11), q(8.5, 19))
            p.drawLine(q(23.5, 11), q(23.5, 19))
    elif name in ("undeformed", "deformed"):
        p.setPen(_pen(GRID, 1.4, QtCore.Qt.DashLine) if name == "deformed" else _pen(BEAM, 2.6))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPolyline(QtGui.QPolygonF([q(6, 28), q(6, 8), q(26, 8), q(26, 28)]))
        if name == "deformed":
            p.setPen(_pen(RED, 2.6))
            path = QtGui.QPainterPath()
            path.moveTo(6, 28)
            path.cubicTo(6, 18, 8, 10, 11, 10)
            path.cubicTo(16, 13, 19, 13, 31, 10)
            path.cubicTo(29, 14, 27, 22, 26, 28)
            p.drawPath(path)
        p.setPen(_pen(INK, 1.4))
        for x in (6, 26):
            p.drawLine(q(x - 4, 29), q(x + 4, 29))
    elif name == "reactions":
        p.setPen(_pen(INK, 1.4))
        p.setBrush(SLAB)
        p.drawPolygon(QtGui.QPolygonF([q(16, 6), q(9, 16), q(23, 16)]))
        p.setPen(_pen(RED, 2.4))
        p.drawLine(q(16, 30), q(16, 20))
        p.setBrush(RED)
        p.drawPolygon(QtGui.QPolygonF([q(12, 22), q(20, 22), q(16, 16.5)]))
    elif name == "frame_forces":  # a moment diagram on a beam
        p.setPen(_pen(BEAM, 2.6))
        p.drawLine(q(3, 12), q(29, 12))
        p.setPen(_pen(QtGui.QColor(40, 150, 120), 1.4))
        p.setBrush(QtGui.QColor(120, 200, 170))
        path = QtGui.QPainterPath()
        path.moveTo(3, 12)
        path.quadTo(16, 38, 29, 12)
        path.closeSubpath()
        p.drawPath(path)
    else:
        p.setPen(_pen(INK, 1.5))
        p.drawText(QtCore.QRectF(0, 0, 32, 32), QtCore.Qt.AlignCenter, name[:2].upper())


def set_ink(color: str) -> None:
    """Line colour of the icons (dark ink on light chrome, light ink on dark).
    Mutated in place so drawing helpers with ``INK`` default arguments follow."""
    c = QtGui.QColor(color)
    INK.setRgb(c.red(), c.green(), c.blue())
    icon.cache_clear()


@lru_cache(maxsize=None)
def icon(name: str) -> QtGui.QIcon:
    ic = QtGui.QIcon()
    for scale in (1, 2):
        pm = QtGui.QPixmap(S * scale, S * scale)
        pm.setDevicePixelRatio(scale)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        _draw(name, p)
        p.end()
        ic.addPixmap(pm)
    return ic


ICON_NAMES = ("joint_load", "frame_load", "area_load", "select", "select_window", "select_line", "point", "beam", "quick_beam", "secondary_beams", "column",
              "quick_column", "brace", "quick_brace", "wall", "quick_wall", "slab", "rect_slab", "quick_slab",
              "opening", "panel_load", "move", "replicate", "divide", "delete", "undo", "redo", "run", "view_3d", "view_plan",
              "view_xz", "view_yz", "layout_1", "layout_2", "layout_2v", "layout_3", "layout_4", "snap_grid",
              "snap_point", "snap_mid", "snap_line", "snap_perp", "snap_int", "snap_ortho", "snap_options", "theme", "colors", "extrude", "display", "rotate3d", "pan", "zoom_in",
              "zoom_out", "zoom_window", "zoom_full", "grid", "select_all", "select_previous", "select_clear", "palette",
              "tables", "properties_panel", "copy", "rotate", "mirror", "offset", "trim", "extend",
              "file_new", "file_open", "file_save", "materials", "frame_sections", "slab_sections", "wall_sections",
              "load_patterns", "load_cases", "combos", "undeformed", "deformed", "reactions", "frame_forces",
              "lock", "unlock", "check_mesh", "check_model")
