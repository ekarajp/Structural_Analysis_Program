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
SHIFT_HELP = (
    "<b>Joint offsets</b><br><br>"
    "<b>dx, dy from the cardinal point</b>: both ends of the member move by dx along its local 3 axis "
    "and dy along local 2 - a free shift for positions the 11 cardinal points do not give (a beam set "
    "down from the slab, a member flush with an edge that is not its own).  "
    "<b>Each end - global X, Y, Z</b>: every end its own vector (ETABS / SAP2000 joint offsets).<br><br>"
    "A joint offset is always in the analysis: the member's axis lies that far from its joints and "
    "the analysis joins them by rigid arms.  Moderate offsets are fine.  A shift larger than "
    "the section itself makes long rigid arms: an axial force then gives moments at the joints "
    "(N x e), the frame gets stiffer than the real connection may be, loads can take paths that do "
    "not exist, and very long arms against flexible members can make the stiffness matrix "
    "ill-conditioned.  The forces and the design are those of the member's own axis, not of the "
    "drawn line.  For a member that really is somewhere else, draw it there (or use a link / "
    "joint offsets) instead.<br><br>"
    "The cardinal point tick ('Include the offset from the centroid ...') is separate: it decides "
    "only whether the cardinal point's own offset is in the analysis.")
TRANSFORM_TEXT = ("Include these offsets (cardinal point and joint offsets) in the analysis - stiffness "
                  "transformed, eccentric member")
TRANSFORM_TIP = ("On: the member's axis lies off its joints by the cardinal point and the joint offsets; the "
                 "analysis joins them by rigid arms (stiffness transformed to the joints).\n"
                 "Off: the member is only drawn at that position; the analysis takes it on its joints, like ETABS "
                 "'Do not transform frame stiffness for offsets from centroid'.")


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

    def __init__(self, pieces: Sequence[np.ndarray], parent=None, title: str = ""):
        super().__init__(parent)
        self.title = title
        self.pieces = [np.asarray(p, float) for p in pieces]
        self.cardinal = 10
        self.offset = (0.0, 0.0)  # (d2, d3): the section moved from the cardinal point
        self.setMinimumSize(190, 230)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setToolTip("Click a point to choose it")

    def set_cardinal(self, cp: int):
        self.cardinal = int(cp)
        self.update()

    def set_offset(self, d2: float, d3: float):
        self.offset = (float(d2), float(d3))
        self.update()

    def _frame(self):
        allp = np.vstack(self.pieces)
        lo, hi = allp.min(axis=0), allp.max(axis=0)  # (y2, y3)
        d = np.asarray(self.offset, float)
        lo_all, hi_all = np.minimum(lo, lo - d), np.maximum(hi, hi - d)  # room for the line moved by d
        w3, h2 = max(hi_all[1] - lo_all[1], 1e-9), max(hi_all[0] - lo_all[0], 1e-9)
        m = 34
        s = min((self.width() - 2 * m) / w3, (self.height() - 2 * m) / h2)
        cx, cy = self.width() / 2, self.height() / 2
        mid3, mid2 = 0.5 * (lo_all[1] + hi_all[1]), 0.5 * (lo_all[0] + hi_all[0])

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
        if self.offset != (0.0, 0.0):  # the drawn line: the cardinal point moved back by (d2, d3)
            pts = self.points()
            base = pts.get(self.cardinal)
            lo, hi, to_px = self._frame()
            if base is not None and self.cardinal <= 10:
                ref = to_px(0.0, 0.0)
                moved = to_px(-self.offset[0], -self.offset[1])
                line = base + (moved - ref)
                p.setPen(QtGui.QPen(QtGui.QColor(230, 0, 160), 2))
                p.drawLine(line + QtCore.QPointF(-8, -8), line + QtCore.QPointF(8, 8))
                p.drawLine(line + QtCore.QPointF(-8, 8), line + QtCore.QPointF(8, -8))
                p.setPen(QtGui.QPen(QtGui.QColor(230, 0, 160), 1, QtCore.Qt.DashLine))
                p.drawLine(line, base)
                p.drawText(line + QtCore.QPointF(10, 14), "line")
        p.setPen(pal.color(QtGui.QPalette.Text))
        if self.title:
            f2 = p.font()
            f2.setBold(True)
            p.setFont(f2)
            p.drawText(self.rect().adjusted(6, 4, -4, -4), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter, self.title)
            f2.setBold(False)
            p.setFont(f2)
        p.drawText(self.rect().adjusted(40, 4, -4, -4), QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight,
                   "seen from End-I; red = cardinal point" if self.width() >= 250 else "red = cardinal pt")
        p.end()

    def mousePressEvent(self, ev):
        pos = ev.position()
        best = min(self.points().items(), key=lambda kv: (kv[1] - pos).manhattanLength())
        if (best[1] - pos).manhattanLength() < 40:
            self.set_cardinal(best[0])
            self.picked.emit(best[0])


class InsertionPointWidget(QtWidgets.QWidget):
    size_changed = QtCore.Signal()  # (one / two pictures: the window grows with it)

    """The form: picture(s) + cardinal point + joint offsets (dx, dy from the cardinal
    point along the section, or each end global X Y Z - converted when switching) + at
    the bottom: include these offsets in the analysis.  ``axes``: the member's local axes
    (rows 1, 2, 3) for the global <-> local conversion and the pictures of both ends."""

    def __init__(self, model, section: Optional[str], cardinal: int = 10, transform: bool = False,
                 offset_i=(0.0, 0.0, 0.0), offset_j=(0.0, 0.0, 0.0), offsets: bool = True, parent=None,
                 shift=(0.0, 0.0), local: bool = True, axes=None):
        super().__init__(parent)
        self.axes = None if axes is None else np.asarray(axes, float)
        u = self.unit = model.units.length
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        pieces = section_pieces(model, section)
        pics = QtWidgets.QHBoxLayout()
        self.picture = SectionPicture(pieces)  # dx, dy mode (both ends) / End-I
        self.picture_j = SectionPicture(pieces, title="End-J")
        pics.addWidget(self.picture, 1)
        pics.addWidget(self.picture_j, 1)
        lay.addLayout(pics, 1)
        right = QtWidgets.QVBoxLayout()
        lay.addLayout(right, 1)
        right.addWidget(QtWidgets.QLabel(f"Section: {section or '(none)'}"))
        self.cardinal = QtWidgets.QComboBox()
        self.cardinal.addItems(CARDINAL_NAMES)
        form = QtWidgets.QFormLayout()
        form.addRow("Cardinal point", self.cardinal)
        right.addLayout(form)
        from .mesh_dialog import help_button

        def spin(v, tip=""):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-1e6, 1e6)
            sp.setDecimals(4)
            sp.setSuffix(f" {u}")  # the unit in every box
            sp.setValue(float(v))
            if tip:
                sp.setToolTip(tip)
            return sp
        g = QtWidgets.QGroupBox(f"Joint offsets [{u}]")
        gl = QtWidgets.QVBoxLayout(g)
        mrow = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["dx, dy from the cardinal point (local 3, 2; both ends)",
                            "Each end - global X, Y, Z"])
        mrow.addWidget(self.mode, 1)
        mrow.addWidget(help_button(SHIFT_HELP, "Joint offsets", self))
        gl.addLayout(mrow)
        self.local_box = QtWidgets.QWidget()
        sl = QtWidgets.QFormLayout(self.local_box)
        sl.setContentsMargins(0, 0, 0, 0)
        self.dx = spin(shift[1], "along local 3 (to the right in the picture)")
        self.dy = spin(shift[0], "along local 2 (up in the picture)")
        sl.addRow(f"dx (local 3) [{u}]", self.dx)
        sl.addRow(f"dy (local 2) [{u}]", self.dy)
        gl.addWidget(self.local_box)
        self.offset_boxes = {}
        self.global_box = QtWidgets.QWidget()
        gg = QtWidgets.QGridLayout(self.global_box)
        gg.setContentsMargins(0, 0, 0, 0)
        for c, axis in enumerate("XYZ"):
            gg.addWidget(QtWidgets.QLabel(f"{axis} [{u}]"), 0, c + 1, QtCore.Qt.AlignHCenter)
        for r, (end, vals) in enumerate((("End-I", offset_i), ("End-J", offset_j))):
            gg.addWidget(QtWidgets.QLabel(end), r + 1, 0)
            for c in range(3):
                sp = spin(vals[c])
                gg.addWidget(sp, r + 1, c + 1)
                self.offset_boxes[(end, c)] = sp
        gl.addWidget(self.global_box)
        self.mode_note = QtWidgets.QLabel()
        self.mode_note.setWordWrap(True)
        self.mode_note.setStyleSheet("color: gray")
        gl.addWidget(self.mode_note)
        self.shift_warn = QtWidgets.QLabel()
        self.shift_warn.setWordWrap(True)
        self.shift_warn.setStyleSheet("color: #b05000")
        gl.addWidget(self.shift_warn)
        right.addWidget(g)
        # at the bottom: it is about the cardinal point and the joint offsets together
        self.transform = QtWidgets.QCheckBox()
        self.transform.setToolTip(TRANSFORM_TIP)
        lab = QtWidgets.QLabel(TRANSFORM_TEXT)
        lab.setWordWrap(True)
        lab.setToolTip(TRANSFORM_TIP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.transform, 0, QtCore.Qt.AlignTop)
        row.addWidget(lab, 1)
        lab.mousePressEvent = lambda _e: self.transform.toggle()
        right.addLayout(row)
        self.transform.setChecked(bool(transform))
        right.addStretch(1)
        if not offsets:  # (drawing: no member yet - dx, dy only)
            self.mode.setEnabled(False)
        self.mode.setCurrentIndex(0 if (local or not offsets) else 1)
        if self.axes is None:
            self.mode.model().item(1).setEnabled(offsets)
        allp = np.vstack(pieces)
        self._size = float(np.max(allp.max(axis=0) - allp.min(axis=0)))
        for w in (self.dx, self.dy, *self.offset_boxes.values()):
            w.valueChanged.connect(self._shift_changed)
        self._last_mode = self.mode.currentIndex()
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.cardinal.currentIndexChanged.connect(self._cardinal_changed)
        self.picture.picked.connect(lambda cp: self.cardinal.setCurrentIndex(cp - 1))
        self.picture_j.picked.connect(lambda cp: self.cardinal.setCurrentIndex(cp - 1))
        self.cardinal.setCurrentIndex(max(1, min(11, int(cardinal or 10))) - 1)
        self._cardinal_changed()
        self._show_mode()

    # ---- the two ways of giving the offsets, converted into each other
    def _to_local(self, v) -> np.ndarray:
        return self.axes @ np.asarray(v, float) if self.axes is not None else np.asarray(v, float)

    def _to_global(self, v) -> np.ndarray:
        return self.axes.T @ np.asarray(v, float) if self.axes is not None else np.asarray(v, float)

    def _set_boxes(self, end: str, vec):
        for c in range(3):
            b = self.offset_boxes[(end, c)]
            b.blockSignals(True)
            b.setValue(float(vec[c]))
            b.blockSignals(False)

    def _mode_changed(self, idx: int):
        note = ""
        if idx == 1 and self._last_mode == 0:  # dx, dy -> the same global vector at both ends
            g = self._to_global((0.0, self.dy.value(), self.dx.value()))
            self._set_boxes("End-I", g)
            self._set_boxes("End-J", g)
        elif idx == 0 and self._last_mode == 1:  # global -> dx, dy of End-I
            oi = [self.offset_boxes[("End-I", c)].value() for c in range(3)]
            oj = [self.offset_boxes[("End-J", c)].value() for c in range(3)]
            loc = self._to_local(oi)
            for sp, v in ((self.dy, loc[1]), (self.dx, loc[2])):
                sp.blockSignals(True)
                sp.setValue(float(v))
                sp.blockSignals(False)
            if not np.allclose(oi, oj, atol=1e-9):
                note = "End-J had other offsets: dx, dy are those of End-I (both ends now)."
            if abs(loc[0]) > 1e-9:
                note += (" " if note else "") + "The part along the member (local 1) is dropped."
        self._last_mode = idx
        self.mode_note.setText(note)
        self._show_mode()

    def _show_mode(self):
        local = self.mode.currentIndex() == 0
        self.local_box.setVisible(local)
        self.global_box.setVisible(not local)
        self.picture_j.setVisible(not local)
        self.picture.title = "" if local else "End-I"
        self._shift_changed()
        self.updateGeometry()
        self.size_changed.emit()

    def _cardinal_changed(self, *_):
        cp = self.cardinal.currentIndex() + 1
        self.picture.set_cardinal(cp)
        self.picture_j.set_cardinal(cp)

    def _shift_changed(self, *_):
        if self.mode.currentIndex() == 0:
            d2, d3 = self.dy.value(), self.dx.value()
            self.picture.set_offset(d2, d3)
            far = max(abs(d2), abs(d3)) > self._size
        else:
            far = False
            for end, pic in (("End-I", self.picture), ("End-J", self.picture_j)):
                loc = self._to_local([self.offset_boxes[(end, c)].value() for c in range(3)])
                pic.set_offset(float(loc[1]), float(loc[2]))  # (the part along the member: not drawn)
                far = far or max(abs(loc[1]), abs(loc[2])) > self._size
        self.shift_warn.setText("Larger than the section itself: in the analysis a long rigid arm "
                                "(see the ?)." if far else "")

    def values(self) -> dict:
        local = self.mode.currentIndex() == 0
        out = {"cp": self.cardinal.currentIndex() + 1, "transform": self.transform.isChecked(), "local": local,
               "shift": (self.dy.value(), self.dx.value())}
        if local:
            d = (0.0, self.dy.value(), self.dx.value())
            out["ioff"] = out["joff"] = d
        else:
            out["ioff"] = tuple(self.offset_boxes[("End-I", c)].value() for c in range(3))
            out["joff"] = tuple(self.offset_boxes[("End-J", c)].value() for c in range(3))
        return out


def frame_fields(cp: int, include: bool) -> dict:
    """The Frame attributes of the cardinal point; ``include``: the cardinal point and the
    joint offsets in the analysis (``offsets_in_analysis``), else drawn there only."""
    cp = int(cp or 10)
    return {"cardinal_point": cp, "display_cardinal": cp, "offsets_in_analysis": bool(include)}


def offset_fields(v: dict) -> dict:
    """Joint offsets of the form: local (dx, dy at both ends) or global per end."""
    return {"joint_offset_i": tuple(v["ioff"]), "joint_offset_j": tuple(v["joff"]),
            "joint_offset_local": bool(v.get("local"))}


def offset_state(frame):
    """(local, (dy, dx), offset_i, offset_j) of a frame's joint offsets for the form."""
    oi, oj = tuple(frame.joint_offset_i), tuple(frame.joint_offset_j)
    if not any(oi) and not any(oj):
        return True, (0.0, 0.0), oi, oj
    if getattr(frame, "joint_offset_local", False) and oi == oj and abs(oi[0]) < 1e-12:
        return True, (oi[1], oi[2]), oi, oj
    return False, (0.0, 0.0), oi, oj


def frame_state(frame, model) -> tuple:
    """(cardinal point shown, offsets included in the analysis) of a frame."""
    cp = frame.display_cardinal
    if cp is None:
        cp = 8 if (frame.type == "Beam" and model.grid_mode == "story") else frame.cardinal_point
    include = bool(getattr(frame, "offsets_in_analysis", True))
    if int(frame.cardinal_point or 10) != int(cp):  # (drawn at cp, analysed at another point: not included)
        include = False
    elif int(cp) == 10 and not any(frame.joint_offset_i) and not any(frame.joint_offset_j):
        include = False  # nothing to include: the default stays off
    return int(cp), include


AUTO_TEXT = "Auto (beams 8 Top center, columns / braces 10)"


class InsertionPointDialog(QtWidgets.QDialog):
    """Drawing: cardinal point, include it in the analysis, dx / dy (both ends)."""

    def __init__(self, model, section, cardinal=10, transform=False, parent=None, shift=(0.0, 0.0)):
        super().__init__(parent)
        self.setWindowTitle("Insertion Point")
        lay = QtWidgets.QVBoxLayout(self)
        self.form = InsertionPointWidget(model, section, cardinal, transform, offsets=False, shift=shift)
        lay.addWidget(self.form)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(640, 360)
