"""Draw > Snap Options: the ETABS Snap Options form with the CAD object snaps."""
from __future__ import annotations

from typing import Dict

from PySide6 import QtCore, QtWidgets

from .snap import SNAP_ITEMS, SnapOptions
from .snap_icons import OPTION_ICON, qicon


class SnapOptionsDialog(QtWidgets.QDialog):
    """Non-modal (stays open, Apply applies) like the ETABS form.  ``applied``
    carries a copy of the options."""

    applied = QtCore.Signal(object)

    def __init__(self, options: SnapOptions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Snap Options")
        self.options = options.copy()
        lay = QtWidgets.QVBoxLayout(self)
        self.boxes: Dict[str, QtWidgets.QCheckBox] = {}
        grp = QtWidgets.QGroupBox("Snap To")
        grid = QtWidgets.QGridLayout(grp)
        half = (len(SNAP_ITEMS) + 1) // 2
        for k, (key, label, hint) in enumerate(SNAP_ITEMS):
            b = QtWidgets.QCheckBox(label)
            b.setIcon(qicon(OPTION_ICON.get(key, key)))  # the AutoCAD marker of the snap
            b.setIconSize(QtCore.QSize(18, 18))
            b.setToolTip(hint)
            b.setChecked(self.options.enabled(key))
            b.toggled.connect(lambda on, kk=key: self.options.set(kk, on))
            grid.addWidget(b, k % half, k // half)
            self.boxes[key] = b
        lay.addWidget(grp)

        sel = QtWidgets.QHBoxLayout()
        all_b = QtWidgets.QPushButton("Select All")
        none_b = QtWidgets.QPushButton("Deselect All")
        all_b.clicked.connect(lambda: self._set_all(True))
        none_b.clicked.connect(lambda: self._set_all(False))
        sel.addWidget(all_b)
        sel.addWidget(none_b)
        sel.addStretch(1)
        self.only = QtWidgets.QCheckBox("Draw using snap only")
        self.only.setToolTip("Points can be placed only where the cursor snaps to something (Draw menu toggle)")
        self.only.setChecked(self.options.snap_only)
        self.only.toggled.connect(lambda on: setattr(self.options, "snap_only", bool(on)))
        sel.addWidget(self.only)
        lay.addLayout(sel)

        inc = QtWidgets.QGroupBox("Snap Increments and Settings")
        form = QtWidgets.QFormLayout(inc)
        self.lengths = QtWidgets.QLineEdit("; ".join(f"{v:g}" for v in self.options.length_steps))
        self.lengths.setToolTip("Length increments in model units, separated by ';' (e.g. 0.5; 0.25)")
        self.lengths.editingFinished.connect(self._lengths_changed)
        form.addRow("Snap at length increments of", self.lengths)
        self.angle = QtWidgets.QDoubleSpinBox()
        self.angle.setRange(0.1, 180.0)
        self.angle.setSuffix(" °")
        self.angle.setValue(self.options.angle_increment)
        self.angle.valueChanged.connect(lambda v: setattr(self.options, "angle_increment", float(v)))
        form.addRow("Snap at angle increments of", self.angle)
        self.polar = QtWidgets.QDoubleSpinBox()
        self.polar.setRange(1.0, 180.0)
        self.polar.setSuffix(" °")
        self.polar.setValue(self.options.polar_angle)
        self.polar.valueChanged.connect(lambda v: setattr(self.options, "polar_angle", float(v)))
        form.addRow("Polar tracking angle", self.polar)
        self.fine = QtWidgets.QDoubleSpinBox()
        self.fine.setRange(0.001, 1e6)
        self.fine.setDecimals(3)
        self.fine.setValue(self.options.fine_spacing)
        self.fine.valueChanged.connect(lambda v: setattr(self.options, "fine_spacing", float(v)))
        form.addRow("Plan fine grid spacing", self.fine)
        self.tol = QtWidgets.QSpinBox()
        self.tol.setRange(2, 60)
        self.tol.setSuffix(" px")
        self.tol.setValue(int(self.options.tolerance_px))
        self.tol.valueChanged.connect(lambda v: setattr(self.options, "tolerance_px", float(v)))
        form.addRow("Screen snap-to tolerance", self.tol)
        lay.addWidget(inc)
        note = QtWidgets.QLabel("Several snaps within the tolerance: press N while drawing to cycle to the next one. "
                                "Shift or Ctrl + right-click while drawing / moving / copying: choose one snap "
                                "(Endpoint, Midpoint, Intersection, ...) for the next point. "
                                "Perpendicular, extension, parallel, orthogonal and polar snaps need a previous point.")
        note.setWordWrap(True)
        lay.addWidget(note)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Close
                                        | QtWidgets.QDialogButtonBox.Apply)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self._apply)
        lay.addWidget(bb)

    def _set_all(self, on: bool):
        self.options.select_all(on)
        for b in self.boxes.values():
            b.setChecked(on)

    def _lengths_changed(self):
        vals = []
        for part in self.lengths.text().replace(",", ";").split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                v = float(part)
            except ValueError:
                continue
            if v > 0:
                vals.append(v)
        self.options.length_steps = vals or [0.5]
        self.lengths.setText("; ".join(f"{v:g}" for v in self.options.length_steps))

    def _apply(self):
        self._lengths_changed()
        self.applied.emit(self.options.copy())

    def _ok(self):
        self._apply()
        self.accept()
