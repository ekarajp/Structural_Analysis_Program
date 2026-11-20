"""Compact dialogs of Edit > Edit Frames > Divide Frames and Edit > Edit Shells > Divide Shells.

One "Divide by" choice; only the few fields of that choice are shown (the dialog
shrinks to them).  ``result()`` returns (kind, values) with lengths in model units,
the same keys as the station finders in :mod:`structstudio.app.frame_divide` use.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from PySide6 import QtCore, QtWidgets

from .forms import display_units


class GridPick(QtWidgets.QWidget):
    """Which grid lines to divide at: the grids shown, every grid (hidden ones too) or
    chosen grids (a checklist of the grid lines of the model)."""

    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self._model = model
        self.chosen: list = []  # (system, name) when "chosen grids"
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["shown grids", "all grids (hidden too)", "chosen grids..."])
        self.pick = QtWidgets.QToolButton()
        self.pick.setText("...")
        self.pick.setToolTip("Choose the grid lines")
        self.pick.setVisible(False)
        h.addWidget(self.combo)
        h.addWidget(self.pick)
        self.combo.activated.connect(self._activated)
        self.pick.clicked.connect(self.choose)

    def _activated(self, i: int):
        self.pick.setVisible(i == 2)
        if i == 2 and not self.choose() and not self.chosen:
            self.combo.setCurrentIndex(0)
            self.pick.setVisible(False)

    def choose(self) -> bool:
        from .frame_divide import grid_lines_list
        rows = grid_lines_list(self._model) if self._model is not None else []
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Grid lines to divide at")
        lay = QtWidgets.QVBoxLayout(d)
        lst = QtWidgets.QListWidget()
        systems = {r[0] for r in rows}
        for sy, nm, dr, shown in rows:
            text = (f"{sy}: " if len(systems) > 1 else "") + f"{nm}  ({dr})" + ("" if shown else "  - hidden")
            it = QtWidgets.QListWidgetItem(text)
            it.setData(QtCore.Qt.UserRole, (sy, nm))
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if (sy, nm) in self.chosen else QtCore.Qt.Unchecked)
            lst.addItem(it)
        lst.setMinimumHeight(min(360, 22 * max(4, len(rows)) + 8))
        lay.addWidget(lst)
        allb = QtWidgets.QPushButton("All")
        noneb = QtWidgets.QPushButton("None")

        def set_all(state):
            for k in range(lst.count()):
                lst.item(k).setCheckState(state)
        allb.clicked.connect(lambda: set_all(QtCore.Qt.Checked))
        noneb.clicked.connect(lambda: set_all(QtCore.Qt.Unchecked))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.addButton(allb, QtWidgets.QDialogButtonBox.ResetRole)
        bb.addButton(noneb, QtWidgets.QDialogButtonBox.ResetRole)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        lay.addWidget(bb)
        if not d.exec():
            return False
        self.chosen = [lst.item(k).data(QtCore.Qt.UserRole) for k in range(lst.count())
                       if lst.item(k).checkState() == QtCore.Qt.Checked]
        self.pick.setToolTip(", ".join(nm for _, nm in self.chosen) or "no grid line chosen")
        return True

    def value(self):
        i = self.combo.currentIndex()
        return "shown" if i == 0 else "all" if i == 1 else list(self.chosen)

    def set_value(self, v):
        if v in (None, "shown"):
            self.combo.setCurrentIndex(0)
        elif v == "all":
            self.combo.setCurrentIndex(1)
        else:
            self.chosen = [tuple(x) for x in v]
            self.combo.setCurrentIndex(2)
        self.pick.setVisible(self.combo.currentIndex() == 2)

KINDS = (("equal", "Equal pieces"), ("length", "Maximum length"), ("dist", "Distances"),
         ("cross", "Intersections"))


class DivideFramesDialog(QtWidgets.QDialog):
    def __init__(self, tol0: float, parent=None, model=None):
        super().__init__(parent)
        self.setWindowTitle("Divide Frames")
        du = display_units()
        self._k = du.factor("coordinates") if du is not None else 1.0
        unit = du.label("coordinates") if du is not None else ""

        def length(value: float) -> QtWidgets.QDoubleSpinBox:
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(0.0, 1e9)
            w.setValue(value * self._k)
            if unit:
                w.setSuffix(f" {unit}")
            return w

        def row(*widgets) -> QtWidgets.QWidget:
            box = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            for w in widgets:
                if isinstance(w, str):
                    w = QtWidgets.QLabel(w)
                h.addWidget(w)
            h.addStretch(1)
            return box

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 8)
        lay.setSpacing(6)
        self.kind = QtWidgets.QComboBox()
        for _, label in KINDS:
            self.kind.addItem(label)
        lay.addWidget(row("Divide by", self.kind))

        self.stack = QtWidgets.QStackedWidget()
        # equal pieces
        self.n = QtWidgets.QSpinBox()
        self.n.setRange(2, 1000)
        self.n.setValue(2)
        self.stack.addWidget(row("Pieces", self.n))
        # maximum length
        self.lmax = length(1.0)
        self.stack.addWidget(row("Longest piece", self.lmax))
        # distances
        self.values = QtWidgets.QLineEdit()
        self.values.setPlaceholderText("1.5, 3.2  or  4@1.2")
        self.values.setMinimumWidth(150)
        self.end = QtWidgets.QComboBox()
        self.end.addItems(["from I", "from J"])
        self.rel = QtWidgets.QCheckBox("ratios 0-1")
        self.stack.addWidget(row(self.values, self.end, self.rel))
        # intersections
        page = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(page)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(3)
        self.joints = QtWidgets.QCheckBox("Joints on the frame")
        self.frames = QtWidgets.QCheckBox("Crossing frames")
        self.grids = QtWidgets.QCheckBox("Grid lines")
        self.grid_pick = GridPick(model)
        self.ref = QtWidgets.QCheckBox("Reference lines")
        self.stories = QtWidgets.QCheckBox("Story levels")
        self.edges = QtWidgets.QCheckBox("Floor / wall edges")
        self.joints.setChecked(True)
        self.frames.setChecked(True)
        self.scope = QtWidgets.QComboBox()
        self.scope.addItems(["all frames", "selected only"])
        self.cut_others = QtWidgets.QCheckBox("cut them too")
        self.ref_axes = QtWidgets.QComboBox()
        self.ref_axes.addItems(["X, Y", "X, Y, Z", "X", "Y", "Z"])
        self.tol = length(tol0)
        g.addWidget(self.joints, 0, 0)
        g.addWidget(self.stories, 0, 1)
        g.addWidget(self.frames, 1, 0)
        g.addWidget(row(self.scope, self.cut_others), 1, 1)
        g.addWidget(self.grids, 2, 0)
        g.addWidget(self.grid_pick, 2, 1, QtCore.Qt.AlignLeft)
        g.addWidget(self.ref, 3, 0)
        g.addWidget(self.ref_axes, 3, 1, QtCore.Qt.AlignLeft)
        g.addWidget(self.edges, 4, 0)
        g.addWidget(row("Tolerance", self.tol), 5, 0, 1, 2)
        self.frames.toggled.connect(self.scope.setEnabled)
        self.frames.toggled.connect(self.cut_others.setEnabled)
        self.ref.toggled.connect(self.ref_axes.setEnabled)
        self.ref_axes.setEnabled(False)
        self.grids.toggled.connect(self.grid_pick.setEnabled)
        self.grid_pick.setEnabled(False)
        self.stack.addWidget(page)
        lay.addWidget(self.stack)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.kind.currentIndexChanged.connect(self._show)
        self._show(0)

    def _show(self, i: int):
        self.stack.setCurrentIndex(i)
        for k in range(self.stack.count()):  # only the shown page takes room
            pol = QtWidgets.QSizePolicy.Preferred if k == i else QtWidgets.QSizePolicy.Ignored
            self.stack.widget(k).setSizePolicy(pol, pol)
        self.stack.adjustSize()
        self.adjustSize()

    def set_kind(self, key: str):
        self.kind.setCurrentIndex([k for k, _ in KINDS].index(key))

    def choice(self) -> Tuple[str, Dict[str, Any]]:
        key = KINDS[self.kind.currentIndex()][0]
        k = self._k or 1.0
        v: Dict[str, Any] = {
            "n": self.n.value(), "lmax": self.lmax.value() / k,
            "values": self.values.text(), "end": "End J" if self.end.currentIndex() == 1 else "End I",
            "rel": self.rel.isChecked(),
            "joints": self.joints.isChecked(), "frames": self.frames.isChecked(),
            "frames_scope": "Selected frames only" if self.scope.currentIndex() == 1 else "All frames",
            "cut_others": self.cut_others.isChecked(), "grids": self.grids.isChecked(),
            "grid_which": self.grid_pick.value(),
            "ref": self.ref.isChecked(),
            "ref_axes": {0: "X and Y", 1: "X, Y and Z"}.get(self.ref_axes.currentIndex(), self.ref_axes.currentText()),
            "stories": self.stories.isChecked(), "edges": self.edges.isChecked(), "tol": self.tol.value() / k,
        }
        return key, v


class DividePreview(QtWidgets.QWidget):
    """A picture of the selected shells as they will be divided: the pieces filled,
    the cuts drawn, the frames lying on the shells dashed and the outline bold."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 190)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.outlines, self.pieces, self.frames = [], [], []
        self.message = ""

    def show_result(self, outlines, pieces, frames, message: str = ""):
        self.outlines, self.pieces, self.frames, self.message = outlines, pieces, frames, message
        self.update()

    def paintEvent(self, ev):
        from PySide6 import QtGui
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.color(QtGui.QPalette.Base))
        p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Mid), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        allpts = [q for arr in self.outlines for q in arr]
        if not allpts:
            p.setPen(pal.color(QtGui.QPalette.Text))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, self.message or "No shell selected")
            return
        xs, ys = [q[0] for q in allpts], [q[1] for q in allpts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        top = 18  # room for the count
        W, H = self.width() - 16, self.height() - 16 - top
        k = min(W / max(x1 - x0, 1e-9), H / max(y1 - y0, 1e-9))
        ox = 8 + (W - k * (x1 - x0)) / 2
        oy = 8 + top + (H - k * (y1 - y0)) / 2

        def poly(arr):
            return QtGui.QPolygonF([QtCore.QPointF(ox + k * (u - x0), oy + k * (y1 - v)) for u, v in arr])
        fills = [QtGui.QColor(120, 170, 230, 90), QtGui.QColor(150, 210, 150, 90)]
        cut_pen = QtGui.QPen(QtGui.QColor(220, 60, 40), 1.4)
        for i, arr in enumerate(self.pieces):
            p.setBrush(fills[i % 2])
            p.setPen(cut_pen)
            p.drawPolygon(poly(arr))
        dash = QtGui.QPen(QtGui.QColor(90, 90, 90), 1.2, QtCore.Qt.DashLine)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(dash)
        for arr in self.frames:
            p.drawPolyline(poly(arr))
        p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Text), 2.0))
        for arr in self.outlines:
            p.drawPolygon(poly(arr))
        p.setPen(pal.color(QtGui.QPalette.Text))
        p.drawText(QtCore.QRect(8, 2, self.width() - 16, top), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   self.message)
        p.end()


SHELL_KINDS = (("equal", "Equal pieces"), ("size", "Maximum size"), ("cross", "Intersections"))


class DivideShellsDialog(QtWidgets.QDialog):
    """Compact dialog of Edit > Edit Shells > Divide Shells, like Divide Frames: equal
    pieces n1 x n2, a largest piece size, or along frames drawn on the shell / grid
    lines / story levels."""

    def __init__(self, tol0: float, parent=None, model=None):
        super().__init__(parent)
        self.setWindowTitle("Divide Shells")
        du = display_units()
        self._k = du.factor("coordinates") if du is not None else 1.0
        unit = du.label("coordinates") if du is not None else ""

        def length(value: float) -> QtWidgets.QDoubleSpinBox:
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(0.0, 1e9)
            w.setValue(value * self._k)
            if unit:
                w.setSuffix(f" {unit}")
            return w

        def row(*widgets) -> QtWidgets.QWidget:
            box = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            for w in widgets:
                if isinstance(w, str):
                    w = QtWidgets.QLabel(w)
                h.addWidget(w)
            h.addStretch(1)
            return box

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 8)
        lay.setSpacing(6)
        self.kind = QtWidgets.QComboBox()
        for _, label in SHELL_KINDS:
            self.kind.addItem(label)
        lay.addWidget(row("Divide by", self.kind))

        self.stack = QtWidgets.QStackedWidget()
        # equal pieces
        self.n1, self.n2 = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for w in (self.n1, self.n2):
            w.setRange(1, 1000)
            w.setValue(2)
        self.stack.addWidget(row("Along edge 1-2 (axis 1)", self.n1, "x  along 1-4 (axis 2)", self.n2))
        # maximum size
        self.size = length(1.0)
        self.stack.addWidget(row("Largest piece side", self.size))
        # intersections
        page = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(page)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(3)
        self.frames = QtWidgets.QCheckBox("Frames drawn on the shell")
        self.scope = QtWidgets.QComboBox()
        self.scope.addItems(["all frames", "selected frames only"])
        self.grids = QtWidgets.QCheckBox("Grid lines")
        self.grid_pick = GridPick(model)
        self.stories = QtWidgets.QCheckBox("Story levels")
        self.frames.setChecked(True)
        self.tol = length(tol0)
        g.addWidget(self.frames, 0, 0)
        g.addWidget(self.scope, 0, 1, QtCore.Qt.AlignLeft)
        g.addWidget(self.grids, 1, 0)
        g.addWidget(self.grid_pick, 1, 1, QtCore.Qt.AlignLeft)
        g.addWidget(self.stories, 2, 0)
        g.addWidget(row("Tolerance", self.tol), 3, 0, 1, 2)
        self.frames.toggled.connect(self.scope.setEnabled)
        self.grids.toggled.connect(self.grid_pick.setEnabled)
        self.grid_pick.setEnabled(False)
        self.stack.addWidget(page)
        lay.addWidget(self.stack)
        self.preview = DividePreview()
        self.preview.setVisible(False)
        lay.addWidget(self.preview, 1)
        self._preview_fn = None
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._refresh)
        for w in (self.n1, self.n2):
            w.valueChanged.connect(self._later)
        for w in (self.size, self.tol):
            w.valueChanged.connect(self._later)
        for w in (self.frames, self.grids, self.stories):
            w.toggled.connect(self._later)
        self.scope.currentIndexChanged.connect(self._later)
        self.grid_pick.combo.currentIndexChanged.connect(self._later)
        self.grid_pick.pick.clicked.connect(self._later)
        note = QtWidgets.QLabel("A frame cuts the shell where it runs right across it; an arc grid cuts a "
                                "floor in short straight pieces. A quadrilateral is divided along its own "
                                "edges, other shapes along their local axes.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.kind.currentIndexChanged.connect(self._show)
        self._show(0)

    def _show(self, i: int):
        self.stack.setCurrentIndex(i)
        for k in range(self.stack.count()):  # only the shown page takes room
            pol = QtWidgets.QSizePolicy.Preferred if k == i else QtWidgets.QSizePolicy.Ignored
            self.stack.widget(k).setSizePolicy(pol, pol)
        self.stack.adjustSize()
        self.adjustSize()
        if getattr(self, "_preview_fn", None) is not None:
            self._later()

    def set_kind(self, key: str):
        self.kind.setCurrentIndex([k for k, _ in SHELL_KINDS].index(key))

    def set_preview(self, fn):
        """``fn(kind, values) -> (outlines, pieces, frames)``: the picture of the result,
        redrawn as the choices change."""
        self._preview_fn = fn
        self.preview.setVisible(fn is not None)
        self._refresh()
        self.adjustSize()

    def _later(self, *_):
        if self._preview_fn is not None:
            self._timer.start()

    def _refresh(self):
        if self._preview_fn is None:
            return
        kind, v = self.choice()
        try:
            outlines, pieces, frames = self._preview_fn(kind, v)
        except Exception as e:  # a picture must never stop the edit
            self.preview.show_result([], [], [], f"No preview ({e})")
            return
        n_sh = len(outlines)
        msg = (f"{n_sh} shell(s) -> {len(pieces)} piece(s)" if len(pieces) > n_sh
               else f"{n_sh} shell(s): nothing to divide with these choices")
        self.preview.show_result(outlines, pieces, frames, msg)

    def choice(self) -> Tuple[str, Dict[str, Any]]:
        key = SHELL_KINDS[self.kind.currentIndex()][0]
        k = self._k or 1.0
        return key, {"n1": self.n1.value(), "n2": self.n2.value(), "size": self.size.value() / k,
                     "frames": self.frames.isChecked(), "selected_frames": self.scope.currentIndex() == 1,
                     "grids": self.grids.isChecked(), "grid_which": self.grid_pick.value(),
                     "stories": self.stories.isChecked(),
                     "tol": self.tol.value() / k}
