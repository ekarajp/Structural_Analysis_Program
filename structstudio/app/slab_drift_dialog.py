"""Design > Concrete Slab Design > Slab-Column Drift Check (ACI 318 18.14.5).

One window: the preferences (code edition, φ, λ, the combinations of the shear stress,
the earthquake cases of the drift with their Cd / Ie), the slab sections' design data
(cover to the reinforcement, unbonded post-tensioning, fpc), then the table of the
connections, the drift - shear stress plot of the code and a selection of the columns
that need slab shear reinforcement.  Changing these does not need a new analysis."""
from __future__ import annotations

import dataclasses
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ..design import check_slab_column_drift, earthquake_cases
from ..design.slab_column_drift import SlabColumnReport, combinations_with_e
from ..model.base import ModelError
from ..model.design import SLAB_COLUMN_CODES

NOTE = ("ACI 318-19 18.14.5 / ACI 318-14 18.14.5.1 (มยผ. 1301/1302-61 ข้อ 2.11.5: ACI 318-14 18.14) - "
        "for buildings of SDC D, E, F (ประเภท ง).  Slab-column connections of two-way slabs without beams "
        "need slab shear reinforcement (vs >= 0.29 √f'c over 4h from the support face) when the design story "
        "drift ratio is large for the gravity shear stress on the critical section.  The design drift is "
        "Cd/Ie x the elastic drift of the column; vu is formed from the column forces case by case.")

COLUMNS = ["Story", "Point", "Column", "Location", "Slab", "h [mm]", "d [mm]", "bo [mm]", "αs", "f'c [MPa]", "PT",
           "vc eq", "φvc [MPa]", "Vu [kN]", "vu [MPa]", "Combo", "vu/φvc", "Δ/h below", "Δ/h above", "Δ/h",
           "Drift case", "Limit", "Exempt ≤", "Status", "vs ≥ [MPa]", "Extent ≥ [mm]", "Notes"]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "Yes" if v else ""
    if isinstance(v, float):
        if abs(v) < 1 and v != 0:
            return f"{v:.4f}"
        return f"{v:.3g}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


class DriftPlot(QtWidgets.QWidget):
    """Design story drift ratio against vu/φvc with the lines of 18.14.5.1 (a) and (b) and
    the limits of 18.14.5.2 (as Fig. R18.14.5.1)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.report: Optional[SlabColumnReport] = None
        self.setMinimumSize(420, 300)
        self.highlight: Optional[int] = None

    def set_report(self, rep: Optional[SlabColumnReport]):
        self.report = rep
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.palette()
        fg = pal.color(QtGui.QPalette.WindowText)
        W, H = self.width(), self.height()
        L, R, T, B = 78, 20, 20, 44
        xmax = 1.0
        ymax = 0.045
        rep = self.report
        pts = [c for c in (rep.checks if rep else []) if c.status != "Not checked"]
        if pts:
            xmax = max(xmax, max(c.ratio for c in pts) * 1.05)
            ymax = max(ymax, max(c.drift for c in pts) * 1.1)

        def X(v):
            return L + (W - L - R) * v / xmax

        def Y(v):
            return H - B - (H - T - B) * v / ymax

        p.setPen(QtGui.QPen(fg, 1))
        p.drawLine(L, H - B, W - R, H - B)
        p.drawLine(L, T, L, H - B)
        f = p.font()
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        p.setFont(f)
        k = 0.0
        while k <= xmax + 1e-9:
            p.drawLine(int(X(k)), H - B, int(X(k)), H - B + 4)
            p.drawText(int(X(k)) - 12, H - B + 16, f"{k:.1f}")
            k += 0.1 if xmax <= 1.5 else 0.5
        y = 0.0
        while y <= ymax + 1e-9:
            p.drawLine(L - 4, int(Y(y)), L, int(Y(y)))
            p.drawText(L - 46, int(Y(y)) + 4, f"{y:.3f}")
            y += 0.005 if ymax <= 0.06 else 0.01
        p.drawText(L + 4, H - 6, "vu / φvc")
        p.save()
        p.translate(14, T + 120)
        p.rotate(-90)
        p.drawText(0, 0, "Δx / hsx")
        p.restore()
        # the code lines
        for top, ex, color, label in ((0.035, 0.005, QtGui.QColor("#1f6fb2"), "nonprestressed"),
                                      (0.040, 0.010, QtGui.QColor("#b2621f"), "unbonded PT")):
            pen = QtGui.QPen(color, 2)
            p.setPen(pen)
            x_ex = (top - ex) / 0.05  # where the line meets the exemption
            p.drawLine(QtCore.QPointF(X(0), Y(top)), QtCore.QPointF(X(min(x_ex, xmax)), Y(top - 0.05 * min(x_ex, xmax))))
            if x_ex < xmax:
                p.drawLine(QtCore.QPointF(X(x_ex), Y(ex)), QtCore.QPointF(X(xmax), Y(ex)))
            p.drawText(QtCore.QPointF(X(0) + 6, Y(top) - 4), label)
        p.setPen(QtGui.QPen(fg, 1, QtCore.Qt.DashLine))
        # the connections
        for i, c in enumerate(pts):
            col = QtGui.QColor("#c62828") if c.required else QtGui.QColor("#2e7d32")
            p.setPen(QtGui.QPen(col, 1))
            p.setBrush(col if c.prestressed else QtCore.Qt.NoBrush)
            r = 6 if self.highlight == i else 4
            p.drawEllipse(QtCore.QPointF(X(c.ratio), Y(c.drift)), r, r)
        p.setPen(QtGui.QPen(fg, 1))
        p.drawText(W - R - 260, T + 12, "● filled: PT (b)   red: shear reinforcement required")
        p.end()


class SlabColumnDriftDialog(QtWidgets.QDialog):
    def __init__(self, win, parent=None):
        super().__init__(parent or win)
        self.win = win
        self.doc = win.doc
        self.setWindowTitle("Slab-Column Drift Check (ACI 318 18.14.5)")
        self.resize(1100, 680)
        self.report: Optional[SlabColumnReport] = None
        lay = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(NOTE)
        note.setWordWrap(True)
        lay.addWidget(note)
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.tabs.addTab(self._prefs_tab(), "Preferences")
        self.tabs.addTab(self._slabs_tab(), "Slab data")
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemSelectionChanged.connect(self._row_changed)
        self.tabs.addTab(self.table, "Results")
        self.plot = DriftPlot()
        self.tabs.addTab(self.plot, "Plot")
        self.msg = QtWidgets.QPlainTextEdit()
        self.msg.setReadOnly(True)
        self.msg.setMaximumHeight(110)
        lay.addWidget(self.msg)
        row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run Check")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.run)
        self.sel_btn = QtWidgets.QPushButton("Select Columns Needing Shear Reinforcement")
        self.sel_btn.clicked.connect(self.select_required)
        self.copy_btn = QtWidgets.QPushButton("Copy Table")
        self.copy_btn.clicked.connect(self.copy_table)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        for b in (self.run_btn, self.sel_btn, self.copy_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)
        if self.doc.results is None:
            self.msg.setPlainText("Run the analysis first (the check uses its results; the settings can be "
                                  "entered now).")

    # ------------------------------------------------------------ preferences
    def _prefs_tab(self) -> QtWidgets.QWidget:
        m = self.doc.model
        p = m.design.slab_column_drift
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)
        form = QtWidgets.QFormLayout()
        self.code = QtWidgets.QComboBox()
        self.code.addItems(list(SLAB_COLUMN_CODES))
        self.code.setCurrentText(p.code)
        self.code.setToolTip("ACI 318-19: vuv from the combinations with E, criterion (b) for unbonded PT.\n"
                             "ACI 318-14 (referred to by มยผ. 1301/1302-61): vug from the gravity loads, one "
                             "criterion, no size effect λs")
        self.phi = QtWidgets.QLineEdit(f"{p.phi:g}")
        self.lam = QtWidgets.QLineEdit(f"{p.lam:g}")
        form.addRow("Code", self.code)
        form.addRow("φ (shear)", self.phi)
        form.addRow("λ (lightweight concrete)", self.lam)
        g.addLayout(form, 0, 0, 1, 2)
        # combinations
        g.addWidget(QtWidgets.QLabel("<b>Combinations for vu</b> (none ticked = the default: 318-19 every "
                                     "combination with E; 318-14 their gravity part)"), 1, 0)
        self.combos = QtWidgets.QListWidget()
        with_e = set(combinations_with_e(m))
        for n in m.load_combinations:
            it = QtWidgets.QListWidgetItem(n + ("   (E)" if n in with_e else ""))
            it.setData(QtCore.Qt.UserRole, n)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if n in p.combos else QtCore.Qt.Unchecked)
            self.combos.addItem(it)
        g.addWidget(self.combos, 2, 0)
        # earthquake cases
        g.addWidget(QtWidgets.QLabel("<b>Earthquake cases for the design drift</b> (none ticked = all); "
                                     "Cd and Ie of each case"), 1, 1)
        self.cases = QtWidgets.QTableWidget(0, 4)
        self.cases.setHorizontalHeaderLabels(["Use", "Case", "Cd", "Ie"])
        self.cases.horizontalHeader().setStretchLastSection(True)
        eq = earthquake_cases(m)
        self.cases.setRowCount(len(eq))
        for i, n in enumerate(eq):
            c = m.load_cases[n]
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            use.setCheckState(QtCore.Qt.Checked if n in p.seismic_cases else QtCore.Qt.Unchecked)
            self.cases.setItem(i, 0, use)
            name = QtWidgets.QTableWidgetItem(n)
            name.setFlags(QtCore.Qt.ItemIsEnabled)
            self.cases.setItem(i, 1, name)
            self.cases.setItem(i, 2, QtWidgets.QTableWidgetItem("" if c.cd is None else f"{c.cd:g}"))
            self.cases.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{getattr(c, 'ie', 1.0):g}"))
        if not eq:
            g.addWidget(QtWidgets.QLabel("No earthquake case: a response spectrum case, or a linear static case "
                                         "loading a pattern of type Seismic"), 3, 1)
        g.addWidget(self.cases, 2, 1)
        return w

    def _slabs_tab(self) -> QtWidgets.QWidget:
        from .forms import display_units
        m = self.doc.model
        du = display_units()
        self.k_len = du.factor("section_dim") if du is not None else 1.0
        self.k_str = du.factor("stress") if du is not None else 1.0
        ul = du.label("section_dim") if du is not None else m.units.label("length")
        us = du.label("stress") if du is not None else m.units.label("stress")
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        lab = QtWidgets.QLabel(f"Cover = slab face to the centroid of the reinforcement (blank = 30 mm); d = h - "
                               f"cover, for PT not less than 0.8h (22.6.2.2).  Unbonded PT: fpc in each direction "
                               f"(8.6.2.1 needs >= 0.9 MPa; vc per 22.6.5.5 at interior columns, Vp = 0).")
        lab.setWordWrap(True)
        v.addWidget(lab)
        slabs = [s for s in m.shell_sections.values() if s.type == "Slab"]
        self.slabs = QtWidgets.QTableWidget(len(slabs), 6)
        self.slabs.setHorizontalHeaderLabels(["Section", "Thickness", f"Cover [{ul}]", "Unbonded PT",
                                              f"fpc1 [{us}]", f"fpc2 [{us}]"])
        self.slabs.horizontalHeader().setStretchLastSection(True)
        for i, s in enumerate(slabs):
            it = QtWidgets.QTableWidgetItem(s.name)
            it.setFlags(QtCore.Qt.ItemIsEnabled)
            self.slabs.setItem(i, 0, it)
            th = QtWidgets.QTableWidgetItem(f"{s.thickness * self.k_len:g} {ul}")
            th.setFlags(QtCore.Qt.ItemIsEnabled)
            self.slabs.setItem(i, 1, th)
            self.slabs.setItem(i, 2, QtWidgets.QTableWidgetItem(
                "" if s.rebar_cover is None else f"{s.rebar_cover * self.k_len:g}"))
            pt = QtWidgets.QTableWidgetItem()
            pt.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            pt.setCheckState(QtCore.Qt.Checked if s.post_tensioned else QtCore.Qt.Unchecked)
            self.slabs.setItem(i, 3, pt)
            for j, val in ((4, s.fpc1), (5, s.fpc2)):
                self.slabs.setItem(i, j, QtWidgets.QTableWidgetItem("" if val is None else f"{val * self.k_str:g}"))
        v.addWidget(self.slabs, 1)
        return w

    @staticmethod
    def _num(text: str, what: str, optional: bool = True) -> Optional[float]:
        t = (text or "").strip()
        if not t:
            if optional:
                return None
            raise ValueError(f"{what}: a number is needed")
        try:
            return float(t)
        except ValueError:
            raise ValueError(f"{what}: {t!r} is not a number")

    def _store(self) -> bool:
        """Write the settings into the model (design data: the results are kept)."""
        try:
            phi = self._num(self.phi.text(), "φ", False)
            lam = self._num(self.lam.text(), "λ", False)
            combos = [self.combos.item(i).data(QtCore.Qt.UserRole) for i in range(self.combos.count())
                      if self.combos.item(i).checkState() == QtCore.Qt.Checked]
            cases, factors = [], {}
            for i in range(self.cases.rowCount()):
                n = self.cases.item(i, 1).text()
                if self.cases.item(i, 0).checkState() == QtCore.Qt.Checked:
                    cases.append(n)
                cd = self._num(self.cases.item(i, 2).text(), f"{n} Cd")
                ie = self._num(self.cases.item(i, 3).text(), f"{n} Ie", False)
                if (cd is not None and cd <= 0) or ie <= 0:
                    raise ValueError(f"{n}: Cd and Ie must be > 0")
                factors[n] = (cd, ie)
            slabs = {}
            for i in range(self.slabs.rowCount()):
                n = self.slabs.item(i, 0).text()
                cov = self._num(self.slabs.item(i, 2).text(), f"{n} cover")
                f1 = self._num(self.slabs.item(i, 4).text(), f"{n} fpc1")
                f2 = self._num(self.slabs.item(i, 5).text(), f"{n} fpc2")
                if cov is not None and cov < 0:
                    raise ValueError(f"{n}: the cover must be >= 0")
                slabs[n] = (None if cov is None else cov / self.k_len,
                            self.slabs.item(i, 3).checkState() == QtCore.Qt.Checked,
                            None if f1 is None else f1 / self.k_str, None if f2 is None else f2 / self.k_str)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Slab-Column Drift Check", str(exc))
            return False
        code = self.code.currentText()

        def apply(model):
            p = model.design.slab_column_drift
            model.design.slab_column_drift = dataclasses.replace(p, code=code, phi=phi, lam=lam, combos=combos,
                                                                 seismic_cases=cases)
            for n, (cd, ie) in factors.items():
                if n in model.load_cases:
                    model.load_cases[n].cd, model.load_cases[n].ie = cd, ie
            for n, (cov, pt, f1, f2) in slabs.items():
                s = model.shell_sections.get(n)
                if s is not None:
                    s.rebar_cover, s.post_tensioned, s.fpc1, s.fpc2 = cov, pt, f1, f2

        try:
            self.doc.edit_design(apply)
        except ModelError as exc:
            QtWidgets.QMessageBox.warning(self, "Slab-Column Drift Check", exc.message)
            return False
        return True

    # ------------------------------------------------------------------ run
    def run(self):
        if not self._store():
            return
        res = self.doc.results
        if res is None:
            self.msg.setPlainText("Run the analysis first: the settings are kept with the model.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            rep = check_slab_column_drift(res, model=self.doc.model)
        except ModelError as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.msg.setPlainText(exc.message)
            return
        finally:
            if QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()
        self.report = rep
        self._fill(rep)

    def _fill(self, rep: SlabColumnReport):
        lines = [f"ERROR: {e}" for e in rep.errors] + [f"Warning: {w}" for w in rep.warnings] + rep.info
        if rep.checks:
            n = len(rep.required)
            lines.insert(0, f"{len(rep.checks)} connection(s); slab shear reinforcement required at {n}"
                         + (f", not checked: {sum(c.status == 'Not checked' for c in rep.checks)}"
                            if any(c.status == 'Not checked' for c in rep.checks) else ""))
        self.msg.setPlainText("\n".join(lines))
        rows = rep.rows()
        self.table.clear()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setRowCount(len(rows))
        red = QtGui.QColor(255, 205, 205)
        grey = QtGui.QColor(225, 225, 225)
        for i, (r, c) in enumerate(zip(rows, rep.checks)):
            for j, k in enumerate(COLUMNS):
                it = QtWidgets.QTableWidgetItem(_fmt(r.get(k)))
                if c.required:
                    it.setBackground(red)
                    it.setForeground(QtGui.QColor("black"))
                elif c.status == "Not checked":
                    it.setBackground(grey)
                    it.setForeground(QtGui.QColor("black"))
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        self.plot.set_report(rep)
        if rep.checks:
            self.tabs.setCurrentWidget(self.table)

    def _row_changed(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows or self.report is None:
            return
        pts = [c for c in self.report.checks if c.status != "Not checked"]
        c = self.report.checks[rows[0]]
        self.plot.highlight = pts.index(c) if c in pts else None
        self.plot.update()

    def select_required(self):
        if self.report is None:
            return
        cols = [c.column for c in self.report.required if c.column in self.doc.model.frames]
        if not cols:
            self.msg.appendPlainText("No connection needs slab shear reinforcement.")
            return
        try:
            self.win.state.set_selection(cols, (), [c.point for c in self.report.required], add=False)
        except TypeError:
            self.win.state.set_selection(cols, (), [c.point for c in self.report.required])
        self.msg.appendPlainText(f"Selected {len(cols)} column(s) below the connections that need slab shear "
                                 "reinforcement.")

    def copy_table(self):
        if self.report is None:
            return
        lines = ["\t".join(COLUMNS)]
        for r in self.report.rows():
            lines.append("\t".join(_fmt(r.get(k)) for k in COLUMNS))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
