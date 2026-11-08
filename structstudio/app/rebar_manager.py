"""Define > Reinforcing Bar Sizes: the bar sizes of the project.

The list is the project's (saved in the model, edited in place: name,
diameter, area).  Catalog... adds sizes from the built-in standards
(TIS 20 / TIS 24, ASTM A615, JIS G3112, EN 10080); Import... reads a CSV
file (name, diameter in mm, optionally area in mm2 and standard); Export...
writes the list in the same format.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from ..display_units import DisplayUnits
from ..model.properties import RebarSize

COLS = ("Name", "Standard", "Diameter", "Area", "Mass")


def _fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}g}"


def _mass_kg_m(size: RebarSize, units) -> float:
    return size.area * units.factor("area") * 7850.0


class RebarSizesDialog(QtWidgets.QDialog):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.model = doc.model
        self.du = DisplayUnits.of(self.model)
        self.sizes: Dict[str, RebarSize] = copy.deepcopy(self.model.rebar_sizes)
        self.setWindowTitle("Reinforcing Bar Sizes")
        self.resize(640, 460)
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        dl, al = self.du.label("section_dim"), self.du.label("section_area")
        self.table.setHorizontalHeaderLabels(["Name", "Standard", f"Diameter [{dl}]", f"Area [{al}]", "Mass [kg/m]"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._edited)
        left.addWidget(self.table, 1)
        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet("color: #c0392b")
        self.error.setWordWrap(True)
        left.addWidget(self.error)
        lay.addLayout(left, 1)
        side = QtWidgets.QVBoxLayout()
        self.btn_catalog = QtWidgets.QPushButton("Catalog...")
        self.btn_catalog.setToolTip("Add sizes from the standards: TIS 20 / 24, ASTM A615, JIS G3112, EN 10080")
        self.btn_add = QtWidgets.QPushButton("Add Size")
        self.btn_add.setToolTip("A size of your own (type its name and diameter; area = pi d^2 / 4)")
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_import = QtWidgets.QPushButton("Import...")
        self.btn_import.setToolTip("CSV: name, diameter (mm) [, area (mm2), standard]")
        self.btn_export = QtWidgets.QPushButton("Export...")
        for b, f in ((self.btn_catalog, self.catalog), (self.btn_add, self.add), (self.btn_delete, self.delete),
                     (self.btn_import, self.import_file), (self.btn_export, self.export_file)):
            b.clicked.connect(f)
            side.addWidget(b)
        side.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        side.addWidget(bb)
        lay.addLayout(side)
        self.refresh()

    # -- table
    def refresh(self, select: Optional[List[str]] = None):
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(self.sizes))
        for i, s in enumerate(self.sizes.values()):
            vals = (s.name, s.standard or "(user)", _fmt(s.diameter * self.du.factor("section_dim")),
                    _fmt(s.area * self.du.factor("section_area")), _fmt(_mass_kg_m(s, self.model.units), 3))
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setData(QtCore.Qt.UserRole, s.name)
                if j in (1, 4):
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                if j >= 2:
                    it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                t.setItem(i, j, it)
        t.resizeColumnsToContents()
        t.blockSignals(False)
        if select:
            t.clearSelection()
            for i in range(t.rowCount()):
                if t.item(i, 0).text() in select:
                    t.selectRow(i)
        self.btn_delete.setEnabled(bool(self.sizes))
        self.btn_export.setEnabled(bool(self.sizes))

    def _edited(self, item):
        old = item.data(QtCore.Qt.UserRole)
        s = self.sizes.get(old)
        if s is None:
            return
        txt = item.text().strip()
        self.error.setText("")
        try:
            if item.column() == 0:
                if not txt:
                    raise ValueError("a size needs a name")
                if txt != old and txt in self.sizes:
                    raise ValueError(f"{txt!r} already exists")
                s.name = txt
                self.sizes = {(txt if k == old else k): v for k, v in self.sizes.items()}
            elif item.column() == 2:
                d = float(txt) / self.du.factor("section_dim")
                if d <= 0:
                    raise ValueError("diameter must be > 0")
                s.diameter = d
                s.area = math.pi * d * d / 4.0  # a new diameter: nominal area of a round bar
                s.standard = ""
            elif item.column() == 3:
                a = float(txt) / self.du.factor("section_area")
                if a <= 0:
                    raise ValueError("area must be > 0")
                s.area = a
                s.standard = ""
        except ValueError as exc:
            self.error.setText(str(exc))
        self.refresh(select=[s.name])

    def selected(self) -> List[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).text() for r in rows]

    # -- actions
    def add(self):
        k = 1
        while f"Bar{k}" in self.sizes:
            k += 1
        name = f"Bar{k}"
        d = 0.012 / self.model.units.factor("length")
        self.sizes[name] = RebarSize(name=name, diameter=d, area=math.pi * d * d / 4.0)
        self.refresh(select=[name])
        row = list(self.sizes).index(name)
        self.table.editItem(self.table.item(row, 0))

    def delete(self):
        names = self.selected()
        for n in names:
            self.sizes.pop(n, None)
        self.refresh()

    def add_entries(self, entries) -> List[str]:
        added = []
        for e in entries:
            s = e.rebar_size(self.model.units)
            if s.name in self.sizes and self.sizes[s.name].standard != s.standard:
                s.name = f"{s.name} ({e.standard.split()[0]})"
            self.sizes[s.name] = s
            added.append(s.name)
        self.refresh(select=added)
        return added

    def catalog(self):
        d = RebarCatalogDialog(set(self.sizes), self)
        if d.exec() == QtWidgets.QDialog.Accepted and d.chosen:
            self.add_entries(d.chosen)

    def import_file(self):
        import os
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import rebar sizes", "", "CSV (*.csv)")
        if not path:
            return
        from ..catalog import read_rebar
        try:
            with open(path, encoding="utf-8-sig") as fh:
                entries = read_rebar(fh.read(), default_standard=os.path.splitext(os.path.basename(path))[0])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import rebar sizes", f"{os.path.basename(path)}:\n{exc}")
            return
        self.add_entries(entries)

    def export_file(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export rebar sizes", "rebar_sizes.csv", "CSV (*.csv)")
        if not path:
            return
        from ..catalog import rebar_csv
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(rebar_csv(list(self.sizes.values()), self.model.units))

    def _ok(self):
        sizes = copy.deepcopy(self.sizes)
        if sizes != self.model.rebar_sizes:
            self.doc.edit(lambda m: setattr(m, "rebar_sizes", sizes))
        self.accept()


class RebarCatalogDialog(QtWidgets.QDialog):
    """Pick bar sizes of a standard: tick the sizes, Add to project."""

    def __init__(self, existing=(), parent=None):
        super().__init__(parent)
        from ..catalog import rebar_catalog
        self.cat = rebar_catalog()
        self.existing = set(existing)
        self.chosen = []
        self.setWindowTitle("Rebar Catalog")
        self.resize(420, 480)
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.standard = QtWidgets.QComboBox()
        self.standard.addItems(list(self.cat))
        form.addRow("Standard", self.standard)
        lay.addLayout(form)
        self.list = QtWidgets.QListWidget()
        lay.addWidget(self.list, 1)
        row = QtWidgets.QHBoxLayout()
        allb = QtWidgets.QPushButton("Select all")
        allb.clicked.connect(lambda: self._tick_all(True))
        noneb = QtWidgets.QPushButton("Clear")
        noneb.clicked.connect(lambda: self._tick_all(False))
        row.addWidget(allb)
        row.addWidget(noneb)
        row.addStretch(1)
        lay.addLayout(row)
        note = QtWidgets.QLabel("Nominal diameter, area and mass per metre as given in the standard.")
        note.setStyleSheet("color: gray")
        note.setWordWrap(True)
        lay.addWidget(note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Add to project")
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.standard.currentTextChanged.connect(self._fill)
        self._fill()

    def _fill(self, *_):
        self.list.clear()
        for e in self.cat.get(self.standard.currentText(), []):
            mass = "" if e.mass is None else f",  {e.mass:g} kg/m"
            it = QtWidgets.QListWidgetItem(f"{e.name}   Ø {e.diameter:g} mm,  {e.area:g} mm²{mass}")
            it.setData(QtCore.Qt.UserRole, e.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if e.name in self.existing else QtCore.Qt.Unchecked)
            self.list.addItem(it)

    def _tick_all(self, on: bool):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)

    def _ok(self):
        names = {self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())
                 if self.list.item(i).checkState() == QtCore.Qt.Checked}
        self.chosen = [e for e in self.cat.get(self.standard.currentText(), []) if e.name in names]
        self.accept()
