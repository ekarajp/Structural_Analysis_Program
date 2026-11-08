"""Loads on Panel form (CYPE 3D "Loads on panel" + "Distribution direction",
with the two-way option)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..model import Model
from ..model.objects import PanelLoad, PanelLoadCase

DISTRIBUTIONS = [("one_way", "One-way (strips parallel to the span direction, simply supported on each bar)"),
                 ("two_way", "Two-way (tributary areas: nearest bar, 45° split on four beams)"),
                 ("shells", "To shells (pressure on the area objects inside the panel)")]
DIRECTIONS = ["Gravity", "Normal", "X", "Y", "Z", "-X", "-Y", "GravityProjected"]


def edge_direction(points: Sequence, k: int, perpendicular: bool = False) -> tuple:
    """Unit vector parallel (or perpendicular, in the panel plane) to edge k."""
    P = np.asarray(points, float)
    n = len(P)
    d = P[(k + 1) % n] - P[k]
    d = d / (np.linalg.norm(d) or 1.0)
    if perpendicular:
        from ..mesh.panel_loads import polygon_frame
        _, _, _, nrm = polygon_frame(P)
        d = np.cross(nrm, d)
        d = d / (np.linalg.norm(d) or 1.0)
    return tuple(float(v) for v in d)


class PanelLoadDialog(QtWidgets.QDialog):
    """Edit a PanelLoad (geometry fixed): distribution, span direction, receivers
    and the loads per pattern.  ``value()`` returns a new PanelLoad."""

    def __init__(self, model: Model, panel: PanelLoad, parent=None):
        super().__init__(parent)
        self.model = model
        self.panel = panel
        self.setWindowTitle(f"Loads on Panel - {panel.name}")
        self.resize(640, 520)
        lay = QtWidgets.QVBoxLayout(self)
        P = np.asarray(panel.points, float)
        zs = P[:, 2]
        self.horizontal = abs(zs.max() - zs.min()) < 1e-6
        info = QtWidgets.QLabel(f"{len(P)} corners, {'horizontal' if self.horizontal else 'inclined / vertical'} panel, "
                                f"Z from {zs.min():g} to {zs.max():g}")
        lay.addWidget(info)

        grp = QtWidgets.QGroupBox("Distribution")
        f = QtWidgets.QFormLayout(grp)
        self.dist = QtWidgets.QComboBox()
        for key, label in DISTRIBUTIONS:
            self.dist.addItem(label, key)
        self.dist.setCurrentIndex([k for k, _ in DISTRIBUTIONS].index(panel.distribution)
                                  if panel.distribution in dict(DISTRIBUTIONS) else 0)
        f.addRow("Load distribution", self.dist)
        self.edge = QtWidgets.QComboBox()
        n = len(P)
        for k in range(n):
            a, b = P[k], P[(k + 1) % n]
            self.edge.addItem(f"Edge {k + 1}: ({a[0]:g}, {a[1]:g}, {a[2]:g}) → ({b[0]:g}, {b[1]:g}, {b[2]:g})", k)
        self.edge_mode = QtWidgets.QComboBox()
        self.edge_mode.addItems(["Parallel to the selected edge", "Perpendicular to the selected edge"])
        # preselect the edge closest to the current span direction
        if panel.span_dir is not None:
            best, bmode = 0, 0
            score = -1
            for k in range(n):
                for mode in (0, 1):
                    d = edge_direction(P, k, mode == 1)
                    sc = abs(float(np.dot(d, panel.span_dir)))
                    if sc > score:
                        score, best, bmode = sc, k, mode
            self.edge.setCurrentIndex(best)
            self.edge_mode.setCurrentIndex(bmode)
        f.addRow("Span direction", self.edge)
        f.addRow("", self.edge_mode)
        self.receivers = QtWidgets.QLineEdit("" if panel.receivers is None else ", ".join(panel.receivers))
        self.receivers.setPlaceholderText("all frames inside the panel")
        self.receivers.setToolTip("Frames that may receive load from this panel (CYPE 'Assign'): names separated by "
                                  "commas; empty = every frame in the panel plane")
        f.addRow("Receiving frames", self.receivers)
        lay.addWidget(grp)
        self.dist.currentIndexChanged.connect(self._dist_changed)
        self._dist_changed()

        grp2 = QtWidgets.QGroupBox("Loads")
        v = QtWidgets.QVBoxLayout(grp2)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Load pattern", "Direction", "Uniform value", "Value at lower level",
                                              "Value at upper level"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        v.addWidget(self.table, 1)
        from ..display_units import DisplayUnits
        du = DisplayUnits.of(model)  # values in the display unit of area loads
        unit, self._k = du.label("area_load"), du.factor("area_load")
        v.addWidget(QtWidgets.QLabel(f"Values in {unit}.  Leave the lower / upper values empty for a uniform load; "
                                     f"fill both for a load varying with height (inclined / vertical panels)."))
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add")
        add.clicked.connect(lambda: self._add_row())
        cp = QtWidgets.QPushButton("Copy")
        cp.clicked.connect(self._copy_row)
        rm = QtWidgets.QPushButton("Delete")
        rm.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()) if self.table.currentRow() >= 0 else None)
        row.addWidget(add)
        row.addWidget(cp)
        row.addWidget(rm)
        row.addStretch(1)
        v.addLayout(row)
        lay.addWidget(grp2, 1)
        for ld in panel.loads:
            self._add_row(ld)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _dist_changed(self, *_):
        one = self.dist.currentData() == "one_way"
        self.edge.setEnabled(one)
        self.edge_mode.setEnabled(one)
        self.receivers.setEnabled(self.dist.currentData() != "shells")

    def _add_row(self, ld: Optional[PanelLoadCase] = None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        pat = QtWidgets.QComboBox()
        pat.addItems(list(self.model.load_patterns))
        d = QtWidgets.QComboBox()
        d.addItems(DIRECTIONS)
        if ld is not None:
            if ld.pattern in self.model.load_patterns:
                pat.setCurrentText(ld.pattern)
            if ld.direction in DIRECTIONS:
                d.setCurrentText(ld.direction)
        else:
            # wind-like patterns act normal to the panel (CYPE default)
            d.setCurrentText("Gravity")
        self.table.setCellWidget(r, 0, pat)
        self.table.setCellWidget(r, 1, d)
        vals = ("", "", "")
        if ld is not None:
            k = getattr(self, "_k", 1.0)
            vals = (("" if ld.varying else f"{ld.value * k:.10g}"), (f"{ld.value_lower * k:.10g}" if ld.varying else ""),
                    (f"{ld.value_upper * k:.10g}" if ld.varying else ""))
        for c, t in enumerate(vals):
            it = QtWidgets.QTableWidgetItem(t)
            if c > 0 and self.horizontal:
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                it.setToolTip("Only for inclined or vertical panels")
            self.table.setItem(r, 2 + c, it)

    def _copy_row(self):
        r = self.table.currentRow()
        if r < 0:
            return
        ld = self._row(r)
        if ld is not None:
            self._add_row(ld)

    def _row(self, r) -> Optional[PanelLoadCase]:
        pat = self.table.cellWidget(r, 0).currentText()
        d = self.table.cellWidget(r, 1).currentText()
        if not pat:
            return None

        def num(c):
            it = self.table.item(r, c)
            t = (it.text() if it else "").strip()
            return float(t) / getattr(self, "_k", 1.0) if t else None
        try:
            u, lo, hi = num(2), num(3), num(4)
        except ValueError:
            return None
        if lo is not None and hi is not None and not self.horizontal:
            return PanelLoadCase(pat, 0.5 * (lo + hi), d, True, lo, hi)
        if u is None:
            return None
        return PanelLoadCase(pat, u, d)

    def value(self) -> PanelLoad:
        p = self.panel
        dist = self.dist.currentData()
        span = p.span_dir
        if dist == "one_way":
            span = edge_direction(p.points, self.edge.currentData(), self.edge_mode.currentIndex() == 1)
        rec_text = self.receivers.text().strip()
        receivers = None if not rec_text else [t.strip() for t in rec_text.replace(";", ",").split(",") if t.strip()]
        loads: List[PanelLoadCase] = []
        for r in range(self.table.rowCount()):
            ld = self._row(r)
            if ld is not None:
                loads.append(ld)
        return PanelLoad(p.name, list(p.points), dist, span, loads, receivers, p.story, p.label, p.group,
                         getattr(p, "area", None))
