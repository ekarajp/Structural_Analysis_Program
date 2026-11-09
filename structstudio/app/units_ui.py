"""Units: the units button in the status bar and the Display Units form (as the
"Units" button in the lower right corner of ETABS / SAP2000 and Options >
Display Units).

The model is always expressed in one consistent set of units (force, length,
temperature).  Changing them converts every value of the model - nothing in the
structure changes, the analysis results are kept and shown in the new units,
and the change can be undone.  Values typed in the forms are in these units.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from ..units import FORCE_UNITS, LENGTH_UNITS, PRESETS, TEMPERATURE_UNITS, UnitSystem

# items listed with their units (the hover text of the units button, the form's table)
ITEMS: List[Tuple[str, str]] = [
    ("Coordinates, dimensions, displacements", "length"),
    ("Section areas", "area"),
    ("Moments of inertia, torsion constant", "inertia"),
    ("Forces, reactions", "force"),
    ("Moments", "moment"),
    ("Line loads, line springs (per length)", "force_per_length"),
    ("Area loads, pressures", "force_per_area"),
    ("Stresses, E, G, fc', fy", "stress"),
    ("Unit weight", "weight_per_volume"),
    ("Mass", "mass"),
    ("Mass density", "mass_per_volume"),
    ("Joint springs (translation)", "translational_stiffness"),
    ("Joint springs (rotation)", "rotational_stiffness"),
    ("Area springs (subgrade modulus)", "weight_per_volume"),
    ("Accelerations, spectrum scale factors", "acceleration"),
    ("Temperature changes", "temperature_change"),
    ("Thermal expansion coefficient", "per_temperature"),
]

# quick sets (the "Set the unit sets quickly" choices of the ETABS form)
QUICK = [("Metric SI (kN, m, C)", ("kN", "m", "C")), ("Metric SI (N, mm, C)", ("N", "mm", "C")),
         ("Metric MKS (tonf, m, C)", ("tonf", "m", "C")), ("Metric MKS (kgf, cm, C)", ("kgf", "cm", "C")),
         ("U.S. (kip, in, F)", ("kip", "in", "F")), ("U.S. (kip, ft, F)", ("kip", "ft", "F"))]


def units_summary(u: UnitSystem, display: Optional[dict] = None) -> str:
    """Hover text of the units button: every item with its units."""
    from ..display_units import DisplayUnits
    rows = [f"Model units: {u.text}", ""]
    if display:
        rows += ["Shown in:"] + [f"  {label}: {unit}" for label, unit in DisplayUnits(u, display).summary()] + [""]
    rows += [f"{label}: {u.label(q)}" for label, q in ITEMS]
    rows += ["", "Click to change: every value is converted (undoable), results are kept."]
    return "\n".join(rows)


class UnitsDialog(QtWidgets.QDialog):
    """Options > Display Units: the model's consistent units (force, length,
    temperature) and, item by item, the units things are shown and typed in
    (presets: consistent, Metric SI, Thai legacy, U.S.)."""

    def __init__(self, units: UnitSystem, parent=None, display: Optional[dict] = None):
        super().__init__(parent)
        from ..display_units import CHOICES, ITEMS as DITEMS, PRESETS as DPRESETS
        self.setWindowTitle("Display Units")
        self.original = units
        self.display_original = dict(display or {})
        lay = QtWidgets.QVBoxLayout(self)
        # ---- quick sets for both tabs
        top_q = QtWidgets.QHBoxLayout()
        top_q.addWidget(QtWidgets.QLabel("Unit set:"))
        self.preset = QtWidgets.QComboBox()
        self.preset.addItem("(choose a set...)")
        self.preset.addItems(list(DPRESETS))
        self.preset.activated.connect(self._apply_preset)
        top_q.addWidget(self.preset, 1)
        lay.addLayout(top_q)
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs, 1)
        # ---- tab 1: consistent units of the model
        t1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(t1)
        top = QtWidgets.QHBoxLayout()
        self.force = QtWidgets.QComboBox()
        self.force.addItems(FORCE_UNITS)
        self.length = QtWidgets.QComboBox()
        self.length.addItems(LENGTH_UNITS)
        self.temperature = QtWidgets.QComboBox()
        self.temperature.addItems(TEMPERATURE_UNITS)
        for label, box in (("Force", self.force), ("Length", self.length), ("Temperature", self.temperature)):
            top.addWidget(QtWidgets.QLabel(label))
            top.addWidget(box)
            box.currentTextChanged.connect(self._update)
        top.addStretch(1)
        l1.addLayout(top)
        quick = QtWidgets.QHBoxLayout()
        quick.addWidget(QtWidgets.QLabel("Consistent sets:"))
        self.quick_buttons = []
        for text, (f, l, t) in QUICK:
            b = QtWidgets.QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, v=(f, l, t): self.set_units(UnitSystem(*v)))
            quick.addWidget(b)
            self.quick_buttons.append(b)
        quick.addStretch(1)
        l1.addLayout(quick)
        self.table = QtWidgets.QTableWidget(len(ITEMS), 3)
        self.table.setHorizontalHeaderLabels(["Item", "Now", "New"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for r, (label, q) in enumerate(ITEMS):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(units.shown(q)))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(""))
        l1.addWidget(self.table, 1)
        n1 = QtWidgets.QLabel(
            "The model is stored in these units: changing them converts every value - the structure does "
            "not change, the analysis results are kept, Edit > Undo goes back.  Forms without a unit next to "
            "a value use these units.")
        n1.setWordWrap(True)
        n1.setStyleSheet("color: gray")
        l1.addWidget(n1)
        tabs.addTab(t1, "Model (consistent) units")
        # ---- tab 2: display units item by item
        t2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(t2)
        grid = QtWidgets.QFormLayout()
        self.item_boxes = {}
        for key, label, q in DITEMS:
            box = QtWidgets.QComboBox()
            box.addItem("")  # = the model unit
            box.addItems(CHOICES.get(q, []))
            box.setEditable(True)  # any other unit, e.g. "kgf-cm"
            box.setCurrentText(self.display_original.get(key, ""))
            box.currentTextChanged.connect(self._update)
            self.item_boxes[key] = box
            grid.addRow(label, box)
        l2.addLayout(grid)
        n2 = QtWidgets.QLabel(
            "What each kind of value is shown and typed in (results, diagrams, legends, tables, the property "
            "and load forms).  Blank: the model units.  Display only - the model is not changed.")
        n2.setWordWrap(True)
        n2.setStyleSheet("color: gray")
        l2.addWidget(n2)
        l2.addStretch(1)
        tabs.addTab(t2, "Display units per item")
        self.tabs = tabs
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(760, 620)
        self.set_units(units)

    def _apply_preset(self, index: int):
        from ..display_units import PRESETS as DPRESETS
        if index <= 0:
            return
        consistent, items = DPRESETS[self.preset.itemText(index)]
        if consistent is not None:
            self.set_units(UnitSystem(*consistent))
        for key, box in self.item_boxes.items():
            box.setCurrentText(items.get(key, ""))
        self._update()

    def set_units(self, u: UnitSystem):
        for box, v in ((self.force, u.force), (self.length, u.length), (self.temperature, u.temperature)):
            box.blockSignals(True)
            box.setCurrentText(v)
            box.blockSignals(False)
        self._update()

    def units(self) -> UnitSystem:
        return UnitSystem(self.force.currentText(), self.length.currentText(), self.temperature.currentText())

    def display(self) -> dict:
        return {k: b.currentText().strip() for k, b in self.item_boxes.items() if b.currentText().strip()}

    def _accept(self):
        from ..display_units import unit_si
        for key, unit in self.display().items():
            try:
                unit_si(unit)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Display Units", str(exc))
                return
        self.accept()

    def _update(self, *_):
        u = self.units()
        for r, (_label, q) in enumerate(ITEMS):
            it = self.table.item(r, 2)
            it.setText(u.label(q))
            f = it.font()
            f.setBold(u.label(q) != self.original.label(q))
            it.setFont(f)
        for key, box in getattr(self, "item_boxes", {}).items():
            from ..display_units import ITEM_QUANTITY
            box.lineEdit().setPlaceholderText(f"model: {u.label(ITEM_QUANTITY[key])}")


class UnitsButton(QtWidgets.QToolButton):
    """The units of the model in the status bar ("kN, m, C"): click for the
    consistent unit sets and the Display Units form; hover lists every item."""

    def __init__(self, get_units: Callable[[], UnitSystem], change: Callable[[UnitSystem], None],
                 show_form: Callable[[], None], parent=None, get_display: Optional[Callable[[], dict]] = None,
                 set_display: Optional[Callable[[dict, Optional[UnitSystem]], None]] = None):
        super().__init__(parent)
        self._get, self._change, self._form = get_units, change, show_form
        self._get_display, self._set_display = get_display, set_display
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setAutoRaise(True)
        self.setObjectName("units_button")
        menu = QtWidgets.QMenu(self)
        menu.aboutToShow.connect(lambda: self._fill(menu))
        self.setMenu(menu)
        self.refresh()

    def _fill(self, menu: QtWidgets.QMenu):
        menu.clear()
        cur = self._get()
        for f, l, t in PRESETS:
            u = UnitSystem(f, l, t)
            a = menu.addAction(u.text)
            a.setCheckable(True)
            a.setChecked(u == cur)
            a.triggered.connect(lambda _=False, uu=u: self._change(uu))
        menu.addSeparator()
        other = cur.with_(temperature="F" if cur.temperature == "C" else "C")
        a = menu.addAction(f"Temperature in {other.temperature} ({other.text})")
        a.triggered.connect(lambda _=False, uu=other: self._change(uu))
        if self._set_display is not None:
            from ..display_units import PRESETS as DPRESETS
            menu.addSeparator()
            sub = menu.addMenu("Display unit sets")
            for name, (consistent, items) in DPRESETS.items():
                a = sub.addAction(name)
                a.triggered.connect(lambda _=False, c=consistent, it=items: self._set_display(
                    dict(it), UnitSystem(*c) if c else None))
        menu.addSeparator()
        menu.addAction("Show Units Form...", self._form)

    def refresh(self):
        u = self._get()
        disp = self._get_display() if self._get_display else {}
        name = ""
        if disp:
            from ..display_units import PRESETS as DPRESETS
            name = next((n.split(" (")[0] for n, (_c, items) in DPRESETS.items() if items and items == disp),
                        "custom display units")
        self.setText(u.text + (f"  ·  {name}" if name else ""))
        self.setToolTip(units_summary(u, disp))
