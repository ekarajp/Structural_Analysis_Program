"""Compact dialogs of Edit > Edit Frames > Divide Frames and Edit > Edit Shells > Divide Shells.

One "Divide by" choice; only the few fields of that choice are shown (the dialog
shrinks to them).  ``result()`` returns (kind, values) with lengths in model units,
the same keys as the station finders in :mod:`structstudio.app.frame_divide` use.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from PySide6 import QtCore, QtWidgets

from .forms import display_units

KINDS = (("equal", "Equal pieces"), ("length", "Maximum length"), ("dist", "Distances"),
         ("cross", "Intersections"))


class DivideFramesDialog(QtWidgets.QDialog):
    def __init__(self, tol0: float, parent=None):
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
        g.addWidget(self.grids, 0, 1)
        g.addWidget(self.frames, 1, 0)
        g.addWidget(row(self.scope, self.cut_others), 1, 1)
        g.addWidget(self.ref, 2, 0)
        g.addWidget(self.ref_axes, 2, 1, QtCore.Qt.AlignLeft)
        g.addWidget(self.stories, 3, 0)
        g.addWidget(self.edges, 3, 1)
        g.addWidget(row("Tolerance", self.tol), 4, 0, 1, 2)
        self.frames.toggled.connect(self.scope.setEnabled)
        self.frames.toggled.connect(self.cut_others.setEnabled)
        self.ref.toggled.connect(self.ref_axes.setEnabled)
        self.ref_axes.setEnabled(False)
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
            "ref": self.ref.isChecked(),
            "ref_axes": {0: "X and Y", 1: "X, Y and Z"}.get(self.ref_axes.currentIndex(), self.ref_axes.currentText()),
            "stories": self.stories.isChecked(), "edges": self.edges.isChecked(), "tol": self.tol.value() / k,
        }
        return key, v


SHELL_KINDS = (("equal", "Equal pieces"), ("size", "Maximum size"), ("cross", "Intersections"))


class DivideShellsDialog(QtWidgets.QDialog):
    """Compact dialog of Edit > Edit Shells > Divide Shells, like Divide Frames: equal
    pieces n1 x n2, a largest piece size, or along frames drawn on the shell / grid
    lines / story levels."""

    def __init__(self, tol0: float, parent=None):
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
        self.stories = QtWidgets.QCheckBox("Story levels")
        self.frames.setChecked(True)
        self.tol = length(tol0)
        g.addWidget(self.frames, 0, 0)
        g.addWidget(self.scope, 0, 1, QtCore.Qt.AlignLeft)
        g.addWidget(self.grids, 1, 0)
        g.addWidget(self.stories, 1, 1)
        g.addWidget(row("Tolerance", self.tol), 2, 0, 1, 2)
        self.frames.toggled.connect(self.scope.setEnabled)
        self.stack.addWidget(page)
        lay.addWidget(self.stack)
        note = QtWidgets.QLabel("A frame cuts the shell where it runs right across it; a quadrilateral is "
                                "divided along its own edges, other shapes along their local axes.")
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

    def set_kind(self, key: str):
        self.kind.setCurrentIndex([k for k, _ in SHELL_KINDS].index(key))

    def choice(self) -> Tuple[str, Dict[str, Any]]:
        key = SHELL_KINDS[self.kind.currentIndex()][0]
        k = self._k or 1.0
        return key, {"n1": self.n1.value(), "n2": self.n2.value(), "size": self.size.value() / k,
                     "frames": self.frames.isChecked(), "selected_frames": self.scope.currentIndex() == 1,
                     "grids": self.grids.isChecked(), "stories": self.stories.isChecked(),
                     "tol": self.tol.value() / k}
