"""Frame Insertion Point: the cardinal point (CSI 1-11) shown on a picture of the
section - click a point or choose it in the list, the picture follows - whether the
offset from the centroid goes into the analysis (stiffness transformed), and the joint
offsets of the two ends.

Cardinal point = *which point of the section* lies on the line drawn (1-9 on the
bounding box: 1-3 bottom, 4-6 middle, 7-9 top, left = -3 side; 10 centroid, 11 shear
centre).  Insertion point = the whole placing of the member relative to its joints: the
cardinal point plus the joint offsets."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

CARDINAL_NAMES = ["1 Bottom left", "2 Bottom center", "3 Bottom right", "4 Middle left", "5 Middle center",
                  "6 Middle right", "7 Top left", "8 Top center", "9 Top right", "10 Centroid",
                  "11 Shear center"]
TRANSFORM_TEXT = "Include the offset from the centroid in the analysis (stiffness transformed - eccentric member)"
TRANSFORM_TIP = ("On: the member's centroid axis lies off the drawn line by the cardinal point; the analysis uses "
                 "that eccentricity (the stiffness is transformed to the joints).\n"
                 "Off: the member is only drawn at that position; the analysis takes it on the drawn line "
                 "(centroid at the joints), like ETABS 'Do not transform frame stiffness for offsets from centroid'.")


def section_pieces(model, section: Optional[str]):
    from .extrude import section_profile
    sec = model.frame_sections.get(section) if section else None
    try:
        pieces = section_profile(sec) if sec is not None else None
    except Exception:
        pieces = None
    if not pieces:  # no dimensions: a plain rectangle (depth 2 x width 1)
        pieces = [np.array([[-0.5, -0.25], [0.5, -0.25], [0.5, 0.25], [-0.5, 0.25]], float)]
    return pieces


class SectionPicture(QtWidgets.QWidget):
    """The section (local 2 up, local 3 to the right) with its 11 cardinal points; the
    chosen one large and red.  A click picks the nearest point."""

    picked = QtCore.Signal(int)

    def __init__(self, pieces: Sequence[np.ndarray], parent=None):
        super().__init__(parent)
        self.pieces = [np.asarray(p, float) for p in pieces]
        self.cardinal = 10
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setToolTip("Click a point to choose it")

    def set_cardinal(self, cp: int):
        self.cardinal = int(cp)
        self.update()

    def _frame(self):
        allp = np.vstack(self.pieces)
        lo, hi = allp.min(axis=0), allp.max(axis=0)  # (y2, y3)
        w3, h2 = max(hi[1] - lo[1], 1e-9), max(hi[0] - lo[0], 1e-9)
        m = 34
        s = min((self.width() - 2 * m) / w3, (self.height() - 2 * m) / h2)
        cx, cy = self.width() / 2, self.height() / 2
        mid3, mid2 = 0.5 * (lo[1] + hi[1]), 0.5 * (lo[0] + hi[0])

        def to_px(y2, y3):
            return QtCore.QPointF(cx + (y3 - mid3) * s, cy - (y2 - mid2) * s)
        return lo, hi, to_px

    def points(self):
        lo, hi, to_px = self._frame()
        out = {}
        for cp in range(1, 10):
            row, col = divmod(cp - 1, 3)
            y2 = (lo[0], 0.5 * (lo[0] + hi[0]), hi[0])[row]
            y3 = (lo[1], 0.5 * (lo[1] + hi[1]), hi[1])[col]
            out[cp] = to_px(y2, y3)
        # 5 is the middle of the box; 10 the centroid (the profile is about it) and 11 the
        # shear centre - the same point for symmetric sections: drawn a little apart
        out[10] = to_px(0.0, 0.0) + QtCore.QPointF(-16, 16)
        out[11] = to_px(0.0, 0.0) + QtCore.QPointF(16, 16)
        return out

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.color(QtGui.QPalette.Base))
        lo, hi, to_px = self._frame()
        p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Text), 1.5))
        p.setBrush(QtGui.QColor(150, 170, 200, 110))
        for pc in self.pieces:
            p.drawPolygon(QtGui.QPolygonF([to_px(y2, y3) for y2, y3 in pc]))
        # bounding box (where 1-9 lie)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Mid), 1, QtCore.Qt.DashLine))
        p.drawRect(QtCore.QRectF(to_px(hi[0], lo[1]), to_px(lo[0], hi[1])))
        # local axes (lower left corner)
        c = QtCore.QPointF(10, self.height() - 34)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 150, 0), 1.4))
        p.drawLine(c, c + QtCore.QPointF(0, -22))
        p.drawText(c + QtCore.QPointF(3, -24), "2")
        p.setPen(QtGui.QPen(QtGui.QColor(0, 90, 200), 1.4))
        p.drawLine(c, c + QtCore.QPointF(22, 0))
        p.drawText(c + QtCore.QPointF(24, 4), "3")
        # centroid / shear centre marks at the real point
        cc = to_px(0.0, 0.0)
        p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Text), 1.0))
        p.drawLine(cc + QtCore.QPointF(-5, 0), cc + QtCore.QPointF(5, 0))
        p.drawLine(cc + QtCore.QPointF(0, -5), cc + QtCore.QPointF(0, 5))
        f = p.font()
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        p.setFont(f)
        for cp, pt in self.points().items():
            sel = cp == self.cardinal
            r = 6.5 if sel else 4.0
            p.setPen(QtGui.QPen(QtGui.QColor(200, 30, 30) if sel else pal.color(QtGui.QPalette.Text), 1.2))
            p.setBrush(QtGui.QColor(220, 40, 40) if sel else pal.color(QtGui.QPalette.Base))
            p.drawEllipse(pt, r, r)
            p.setPen(QtGui.QColor(200, 30, 30) if sel else pal.color(QtGui.QPalette.Text))
            label = {10: "10 C", 11: "11 S"}.get(cp, str(cp))
            off = QtCore.QPointF(-34, 14) if cp == 10 else QtCore.QPointF(7, 14) if cp == 11 else QtCore.QPointF(7, -6)
            p.drawText(pt + off, label)
        p.setPen(pal.color(QtGui.QPalette.Text))
        p.drawText(self.rect().adjusted(40, 4, -4, -4), QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight,
                   "seen from End-I; red = on the line")
        p.end()

    def mousePressEvent(self, ev):
        pos = ev.position()
        best = min(self.points().items(), key=lambda kv: (kv[1] - pos).manhattanLength())
        if (best[1] - pos).manhattanLength() < 40:
            self.set_cardinal(best[0])
            self.picked.emit(best[0])


class InsertionPointWidget(QtWidgets.QWidget):
    """The form: picture + cardinal point list + 'include in the analysis' + (optional)
    joint offsets End-I / End-J (global X Y Z, model length units)."""

    def __init__(self, model, section: Optional[str], cardinal: int = 10, transform: bool = False,
                 offset_i=(0.0, 0.0, 0.0), offset_j=(0.0, 0.0, 0.0), offsets: bool = True, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.picture = SectionPicture(section_pieces(model, section))
        lay.addWidget(self.picture, 1)
        right = QtWidgets.QVBoxLayout()
        lay.addLayout(right, 1)
        right.addWidget(QtWidgets.QLabel(f"Section: {section or '(none)'}"))
        self.cardinal = QtWidgets.QComboBox()
        self.cardinal.addItems(CARDINAL_NAMES)
        form = QtWidgets.QFormLayout()
        form.addRow("Cardinal point", self.cardinal)
        right.addLayout(form)
        self.transform = QtWidgets.QCheckBox(TRANSFORM_TEXT)
        self.transform.setToolTip(TRANSFORM_TIP)
        self.transform.setStyleSheet("QCheckBox { spacing: 6px; }")
        lab = QtWidgets.QLabel(TRANSFORM_TEXT)
        lab.setWordWrap(True)
        self.transform.setText("")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.transform, 0, QtCore.Qt.AlignTop)
        row.addWidget(lab, 1)
        lab.mousePressEvent = lambda _e: self.transform.toggle()
        right.addLayout(row)
        self.transform.setChecked(bool(transform))
        self.offset_boxes = {}
        if offsets:
            u = model.units.length
            g = QtWidgets.QGroupBox(f"Joint offsets (global, {u})")
            gl = QtWidgets.QGridLayout(g)
            for c, axis in enumerate("XYZ"):
                gl.addWidget(QtWidgets.QLabel(axis), 0, c + 1, QtCore.Qt.AlignHCenter)
            for r, (end, vals) in enumerate((("End-I", offset_i), ("End-J", offset_j))):
                gl.addWidget(QtWidgets.QLabel(end), r + 1, 0)
                for c in range(3):
                    sp = QtWidgets.QDoubleSpinBox()
                    sp.setRange(-1e6, 1e6)
                    sp.setDecimals(4)
                    sp.setValue(float(vals[c]))
                    gl.addWidget(sp, r + 1, c + 1)
                    self.offset_boxes[(end, c)] = sp
            note = QtWidgets.QLabel("Moves each end of the member away from its joint by this vector "
                                    "(a rigid link joint - member end).")
            note.setWordWrap(True)
            note.setStyleSheet("color: gray")
            gl.addWidget(note, 3, 0, 1, 4)
            right.addWidget(g)
        right.addStretch(1)
        self.cardinal.currentIndexChanged.connect(lambda i: self.picture.set_cardinal(i + 1))
        self.picture.picked.connect(lambda cp: self.cardinal.setCurrentIndex(cp - 1))
        self.cardinal.setCurrentIndex(max(1, min(11, int(cardinal or 10))) - 1)
        self.picture.set_cardinal(self.cardinal.currentIndex() + 1)

    def values(self) -> dict:
        out = {"cp": self.cardinal.currentIndex() + 1, "transform": self.transform.isChecked()}
        if self.offset_boxes:
            out["ioff"] = tuple(self.offset_boxes[("End-I", c)].value() for c in range(3))
            out["joff"] = tuple(self.offset_boxes[("End-J", c)].value() for c in range(3))
        return out


def frame_fields(cp: int, transform: bool) -> dict:
    """The Frame attributes for a cardinal point: in the analysis (``cardinal_point``)
    or only where it is drawn (``display_cardinal``)."""
    cp = int(cp or 10)
    return {"cardinal_point": cp if transform else 10, "display_cardinal": cp}


def frame_state(frame, model) -> tuple:
    """(cardinal point shown, offset included in the analysis) of a frame."""
    cp = frame.display_cardinal
    if cp is None:
        cp = 8 if (frame.type == "Beam" and model.grid_mode == "story") else frame.cardinal_point
    transform = int(frame.cardinal_point or 10) != 10 and int(frame.cardinal_point) == int(cp)
    return int(cp), bool(transform)


AUTO_TEXT = "Auto (beams 8 Top center, columns / braces 10)"


class InsertionPointDialog(QtWidgets.QDialog):
    def __init__(self, model, section, cardinal=10, transform=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insertion Point")
        lay = QtWidgets.QVBoxLayout(self)
        self.form = InsertionPointWidget(model, section, cardinal, transform, offsets=False)
        lay.addWidget(self.form)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(640, 360)
