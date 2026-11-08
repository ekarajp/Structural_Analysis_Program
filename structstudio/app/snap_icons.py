"""AutoCAD object-snap markers: one shape per snap kind, used for the marker
drawn at the cursor and for the icons of the Snap Options form and of the
Shift / Ctrl + right-click snap menu.

Shapes are lists of strokes (polylines) in a unit box [-1, 1] x [-1, 1], y up.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Stroke = List[Tuple[float, float]]


def _circle(r: float = 1.0, n: int = 20, cx: float = 0.0, cy: float = 0.0) -> Stroke:
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]


SQUARE = [(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]
SHAPES: Dict[str, List[Stroke]] = {
    # object snaps (AutoCAD markers)
    "end": [SQUARE],  # Endpoint: square
    "joint": [SQUARE],  # a joint is the end point of its members
    "midpoint": [[(0, 1.1), (-1.1, -0.8), (1.1, -0.8), (0, 1.1)]],  # Midpoint: triangle
    "center": [_circle()],  # Center: circle
    "node": [_circle(), [(-0.7, -0.7), (0.7, 0.7)], [(-0.7, 0.7), (0.7, -0.7)]],  # Node: circle with X
    "intersection": [[(-1, -1), (1, 1)], [(-1, 1), (1, -1)]],  # Intersection: X
    "extension": [[(-1.0, 0), (-0.6, 0)], [(-0.2, 0), (0.2, 0)], [(0.6, 0), (1.0, 0)]],  # Extension: dashes
    "perpendicular": [[(-1, 1), (-1, -1), (1, -1)], [(-1, 0), (0, 0), (0, -1)]],  # Perpendicular: right angle
    "parallel": [[(-1, -0.5), (0.2, 1)], [(-0.2, -1), (1, 0.5)]],  # Parallel: //
    "nearest": [[(-1, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]],  # Nearest: hourglass
    "edge": [[(-1, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]],
    "grid": [[(1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)], [(-0.45, 0), (0.45, 0)], [(0, -0.45), (0, 0.45)]],
    "tangent": [_circle(0.75, 18, 0, -0.25), [(-1, 0.75), (1, 0.75)]],
    "none": [_circle(), [(-0.7, -0.7), (0.7, 0.7)]],
    # tracking and helpers
    "fine_grid": [[(x, y), (x + 0.01, y)] for x in (-0.8, 0, 0.8) for y in (-0.8, 0, 0.8)],
    "reference": [[(-1, 0), (-0.4, 0)], [(0.0, 0), (0.6, 0)], [(0, -1), (0, -0.4)], [(0, 0.2), (0, 0.8)]],
    "ortho": [[(-1, 1), (-1, -1), (1, -1)]],
    "alignment": [[(-1, 0), (-0.5, 0)], [(-0.2, 0), (0.3, 0)], [(-0.35, -0.8), (-0.35, 0.8)], [(0.6, 0), (1, 0)]],
    "alignment2": [[(-1, 0), (1, 0)], [(0, -1), (0, 1)]],
    "length_increments": [[(-1, -0.3), (1, -0.3), (1, 0.3), (-1, 0.3), (-1, -0.3)],
                          [(-0.5, 0.3), (-0.5, 0)], [(0, 0.3), (0, -0.1)], [(0.5, 0.3), (0.5, 0)]],
    "angle_increments": [[(1, -1), (-1, -1), (0.6, 0.8)], [(0.2, -1), (0.15, -0.6)], [(-0.2, -0.35), (0.1, -0.25)]],
}
# polar tracking: two rays and the angle arc
SHAPES["polar"] = [[(-1, -1), (1, -1)], [(-1, -1), (0.8, 0.6)],
                   [(-1 + 1.2 * math.cos(a), -1 + 1.2 * math.sin(a)) for a in [k * math.radians(41.6) / 6
                                                                               for k in range(7)]]]
# a point with no snap (where the next joint goes): a diagonal cross - not a circle,
# circles are the pin (release) symbols
SHAPES["free"] = [[(-0.7, -0.7), (0.7, 0.7)], [(-0.7, 0.7), (0.7, -0.7)]]
for _k in ("increment", "fine_grid"):
    SHAPES.setdefault(_k, [_circle(0.6)])

# Snap Options form item -> marker
OPTION_ICON = {
    "joints": "joint", "ends_midpoints": "midpoint", "grid": "grid", "lines": "nearest", "edges": "edge",
    "perpendicular": "perpendicular", "intersections": "intersection", "centers": "center",
    "extensions": "extension", "parallels": "parallel", "fine_grid": "fine_grid", "reference": "reference",
    "ortho": "ortho", "polar": "polar", "alignment": "alignment", "length_increments": "length_increments",
    "angle_increments": "angle_increments",
}

# Shift / Ctrl + right-click menu (AutoCAD "Object Snap" shortcut menu): (override key, label, marker)
OVERRIDE_MENU: Sequence[Tuple[str, str, str]] = (
    ("endpoint", "Endpoint", "end"),
    ("midpoint", "Midpoint", "midpoint"),
    ("intersection", "Intersection", "intersection"),
    ("extension", "Extension", "extension"),
    ("center", "Center (area)", "center"),
    ("node", "Node (joint)", "node"),
    ("perpendicular", "Perpendicular", "perpendicular"),
    ("parallel", "Parallel", "parallel"),
    ("grid", "Grid intersection", "grid"),
    ("nearest", "Nearest", "nearest"),
    ("none", "None", "none"),
)


def strokes(kind: str) -> List[Stroke]:
    return SHAPES.get(kind) or [_circle(0.8)]


def qicon(kind: str, size: int = 18, color: str = "#d48a00"):
    """A QIcon of the marker (AutoCAD draws its markers in a gold / yellow colour)."""
    from PySide6 import QtCore, QtGui
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(color), max(1.4, size / 11.0))
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(pen)
    s = (size - 5) / 2.0
    c = size / 2.0
    for st in strokes(kind):
        if len(st) == 2 and abs(st[0][0] - st[1][0]) < 0.05 and abs(st[0][1] - st[1][1]) < 0.05:
            p.setBrush(QtGui.QColor(color))
            p.drawEllipse(QtCore.QPointF(c + s * st[0][0], c - s * st[0][1]), 1.3, 1.3)
            p.setBrush(QtCore.Qt.NoBrush)
            continue
        path = QtGui.QPainterPath(QtCore.QPointF(c + s * st[0][0], c - s * st[0][1]))
        for x, y in st[1:]:
            path.lineTo(c + s * x, c - s * y)
        p.drawPath(path)
    p.end()
    return QtGui.QIcon(pm)
