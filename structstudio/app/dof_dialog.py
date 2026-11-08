"""Analyze > Set Active Degrees of Freedom: the six DOFs with a picture (ETABS /
SAP2000 form) - the active ones drawn as arrows on the global axes."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

DOFS = ("UX", "UY", "UZ", "RX", "RY", "RZ")
DOF_TEXT = {"UX": "Translation along X", "UY": "Translation along Y", "UZ": "Translation along Z",
            "RX": "Rotation about X", "RY": "Rotation about Y", "RZ": "Rotation about Z"}
PRESETS: Sequence[Tuple[str, Tuple[bool, ...]]] = (
    ("Space frame (all)", (True, True, True, True, True, True)),
    ("Plane frame X-Z", (True, False, True, False, True, False)),
    ("Plane frame Y-Z", (False, True, True, True, False, False)),
    ("Plane grid / slab (UZ, RX, RY)", (False, False, True, True, True, False)),
    ("Plane truss X-Z", (True, False, True, False, False, False)),
    ("Space truss", (True, True, True, False, False, False)),
)
AXIS_COLOR = {"X": QtGui.QColor(215, 60, 60), "Y": QtGui.QColor(50, 160, 70), "Z": QtGui.QColor(50, 100, 220)}


class DofPicture(QtWidgets.QWidget):
    """Isometric global axes; active translations as straight arrows, active
    rotations as curved arrows around their axis (right-hand rule)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active: Dict[str, bool] = {d: True for d in DOFS}
        self.setMinimumSize(260, 240)

    def set_active(self, active: Dict[str, bool]):
        self.active = dict(active)
        self.update()

    # isometric projection of a global direction
    @staticmethod
    def _dir(axis: str) -> QtCore.QPointF:
        c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
        return {"X": QtCore.QPointF(c30, s30), "Y": QtCore.QPointF(-c30, s30), "Z": QtCore.QPointF(0, -1)}[axis]

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        o = QtCore.QPointF(w * 0.5, h * 0.58)
        L = min(w, h) * 0.36
        pal = self.palette()
        ink = pal.color(QtGui.QPalette.WindowText)
        faint = QtGui.QColor(ink)
        faint.setAlpha(60)
        # the joint
        p.setPen(QtGui.QPen(ink, 1.2))
        p.setBrush(ink)
        p.drawRect(QtCore.QRectF(o.x() - 4, o.y() - 4, 8, 8))
        f = p.font()
        f.setBold(True)
        p.setFont(f)
        for k, axis in enumerate("XYZ"):
            d = self._dir(axis)
            col = AXIS_COLOR[axis]
            t_on = self.active.get("U" + axis, False)
            r_on = self.active.get("R" + axis, False)
            # axis line (faint) and translation arrow
            p.setPen(QtGui.QPen(faint, 1.0, QtCore.Qt.DashLine))
            p.drawLine(o, o + d * L * 1.15)
            tip = o + d * L
            pen = QtGui.QPen(col if t_on else faint, 3.0 if t_on else 1.5)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(o + d * 8, tip)
            self._head(p, tip, d, col if t_on else faint)
            p.setPen(col if t_on else faint)
            p.drawText(tip + d * 14 + QtCore.QPointF(-8, 5), ("U" if t_on else "") + axis)
            # rotation: a curved arrow around the axis, two thirds out
            c = o + d * L * 0.62
            self._rotation(p, c, d, col if r_on else faint, r_on)
            if r_on:
                p.setPen(col)
                p.drawText(c + QtCore.QPointF(14, -14 if axis != "Z" else 4), "R" + axis)
        p.end()

    @staticmethod
    def _head(p, tip, d, color, size=10.0):
        n = QtCore.QPointF(-d.y(), d.x())
        poly = QtGui.QPolygonF([tip + d * 2, tip - d * size + n * size * 0.45, tip - d * size - n * size * 0.45])
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(color)
        p.drawPolygon(poly)

    def _rotation(self, p, c, d, color, on):
        # an ellipse across the axis (the circle about it, seen in the isometric view)
        n = QtCore.QPointF(-d.y(), d.x())
        r1, r2 = 13.0, 6.0
        pts = []
        for k in range(0, 300, 12):
            a = math.radians(k)
            pts.append(c + n * (r1 * math.cos(a)) + d * (r2 * math.sin(a)))
        pen = QtGui.QPen(color, 2.4 if on else 1.2)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPolyline(QtGui.QPolygonF(pts))
        if on:
            a, b = pts[-1], pts[-2]
            v = a - b
            ln = math.hypot(v.x(), v.y()) or 1.0
            self._head(p, a + v / ln * 4, v / ln, color, 7.0)


class DofDialog(QtWidgets.QDialog):
    def __init__(self, active: Sequence[bool], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Active Degrees of Freedom")
        lay = QtWidgets.QHBoxLayout(self)
        self.pic = DofPicture(self)
        lay.addWidget(self.pic, 1)
        right = QtWidgets.QVBoxLayout()
        grp = QtWidgets.QGroupBox("Active degrees of freedom")
        g = QtWidgets.QGridLayout(grp)
        self.boxes: Dict[str, QtWidgets.QCheckBox] = {}
        for k, d in enumerate(DOFS):
            b = QtWidgets.QCheckBox(d)
            b.setToolTip(DOF_TEXT[d])
            b.setChecked(bool(active[k]))
            b.toggled.connect(self._changed)
            g.addWidget(b, k % 3, k // 3)
            self.boxes[d] = b
        right.addWidget(grp)
        pre = QtWidgets.QGroupBox("Fast DOFs")
        pv = QtWidgets.QVBoxLayout(pre)
        for label, flags in PRESETS:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda _=False, fl=flags: self._set(fl))
            pv.addWidget(btn)
        right.addWidget(pre)
        note = QtWidgets.QLabel("Inactive DOFs are fixed at every joint: a plane frame in X-Z needs only "
                                "UX, UZ and RY; a slab (SAFE) UZ, RX and RY.")
        note.setWordWrap(True)
        right.addWidget(note)
        right.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        right.addWidget(bb)
        lay.addLayout(right)
        self._changed()

    def _set(self, flags):
        for d, on in zip(DOFS, flags):
            self.boxes[d].setChecked(on)

    def _changed(self, *_):
        self.pic.set_active({d: b.isChecked() for d, b in self.boxes.items()})

    def value(self) -> Tuple[bool, ...]:
        return tuple(self.boxes[d].isChecked() for d in DOFS)
