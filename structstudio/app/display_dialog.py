"""View > Set Display Options (Ctrl+W) - what a model window shows."""
from __future__ import annotations

from typing import Dict

from PySide6 import QtWidgets

from .display import GRID_3D, GRID_3D_LABELS, GROUPS, DisplayOptions


class DisplayOptionsDialog(QtWidgets.QDialog):
    def __init__(self, options: DisplayOptions, window_title: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Display Options" + (f" - {window_title}" if window_title else ""))
        self.options = options.copy()
        self.boxes: Dict[str, QtWidgets.QCheckBox] = {}
        lay = QtWidgets.QVBoxLayout(self)
        body = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(body)
        grid.setContentsMargins(0, 0, 0, 0)
        for k, (title, items) in enumerate(GROUPS):
            gb = QtWidgets.QGroupBox(title)
            v = QtWidgets.QVBoxLayout(gb)
            for field, label in items:
                cb = QtWidgets.QCheckBox(label)
                cb.setChecked(bool(getattr(self.options, field)))
                v.addWidget(cb)
                self.boxes[field] = cb
            if title == "Other":
                row = QtWidgets.QHBoxLayout()
                row.addWidget(QtWidgets.QLabel("Grid in 3D windows"))
                self.grid_3d = QtWidgets.QComboBox()
                for key in GRID_3D:
                    self.grid_3d.addItem(GRID_3D_LABELS[key], key)
                self.grid_3d.setCurrentIndex(GRID_3D.index(self.options.grid_3d))
                row.addWidget(self.grid_3d, 1)
                v.addLayout(row)
            v.addStretch(1)
            grid.addWidget(gb, k // 2, k % 2)
        from .fit import install, scroll_page
        lay.addWidget(scroll_page(body), 1)  # the check boxes scroll on a short screen; the buttons below stay
        quick = QtWidgets.QHBoxLayout()
        for text, slot in (("Show all objects", self._all_objects), ("Defaults", self._defaults)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            quick.addWidget(b)
        quick.addStretch(1)
        self.apply_all = QtWidgets.QCheckBox("Apply to all windows")
        quick.addWidget(self.apply_all)
        lay.addLayout(quick)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        install(self)

    def _all_objects(self):
        for f in ("beams", "columns", "braces", "null_lines", "rigid_links", "floors", "walls", "ramps", "openings", "load_panels"):
            self.boxes[f].setChecked(True)

    def _defaults(self):
        d = DisplayOptions()
        for f, cb in self.boxes.items():
            cb.setChecked(bool(getattr(d, f)))
        self.grid_3d.setCurrentIndex(GRID_3D.index(d.grid_3d))

    def result_options(self) -> DisplayOptions:
        o = self.options.copy()
        for f, cb in self.boxes.items():
            setattr(o, f, cb.isChecked())
        o.grid_3d = self.grid_3d.currentData()
        return o
