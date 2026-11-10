"""Load > Show Loads: which assigned loads the model windows draw.

* **Load types** - joint, frame, area (on floors / walls), free load panels, the panel
  loads as they reach the members, and generated lateral loads (seismic / wind applied at
  the diaphragms) - any number at once.
* **What** - one or more load patterns (Dead, SDL, Live, Wind, Seismic ...), a load case
  (its patterns times their scale factors) or a load combination (expanded down to the
  patterns; linear-add combinations only, as an envelope has no single set of loads).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Set

from PySide6 import QtCore, QtWidgets

from ..model import Model

KIND_ROWS = (
    ("joint", "Joint loads"),
    ("frame", "Frame loads (beams, columns, braces)"),
    ("area", "Floor / wall loads (area loads on shells)"),
    ("panel", "Load panels (drawn load areas)"),
    ("applied", "Panel and area loads as they reach the members"),
)
APPLIED_HELP = (
    "<b>Panel and area loads as they reach the members</b><br><br>"
    "A load panel or a floor / wall area load is drawn as you assigned it: a load per area over "
    "the panel.  The analysis does not use it like that - it hands the load on to the structure "
    "that carries it:<br>"
    "&bull; a load panel goes to the beams, walls and columns around it as <b>line loads and point "
    "loads</b> (one-way or two-way distribution by tributary areas), or - distribution 'shells' - "
    "as a pressure on the shells under it;<br>"
    "&bull; an area load on a floor / wall goes to its shell elements and from them to the mesh "
    "joints.<br><br>"
    "Tick this to see those loads on the members - the loads the analysis actually applies - "
    "instead of the panels themselves (both at once would show the same load twice).  Use it to "
    "check which beams a panel loads and how much.  Load &gt; Load Panel / Area Load Report lists "
    "the same values.")
GENERATED_LABEL = "Generated lateral loads (seismic / wind at the diaphragm centres)"


def case_pattern_factors(model: Model, case: str) -> Dict[str, float]:
    """Patterns x scale factors of a linear static load case."""
    c = getattr(model, "load_cases", {}).get(case)
    loads = getattr(c, "loads", None)
    if not isinstance(loads, dict):
        raise ValueError(f"Load case {case!r} has no load patterns to show (it is not a static case)")
    return {str(p): float(f) for p, f in loads.items()}


def combo_pattern_factors(model: Model, name: str, _seen: Optional[Set[str]] = None) -> Dict[str, float]:
    """Patterns x factors of a linear-add load combination (nested combinations and load
    cases expanded).  Envelope / SRSS / absolute combinations have no single set of loads."""
    seen = set(_seen or ())
    if name in seen:
        raise ValueError(f"Combination {name!r} refers to itself")
    seen.add(name)
    combo = model.load_combinations[name]
    if combo.type != "LinearAdd":
        raise ValueError(f"{name} is a {combo.type} combination: it has no single set of loads to draw "
                         "(only linear-add combinations can be shown)")
    out: Dict[str, float] = {}
    for it in combo.items:
        if it.is_combo or it.name in model.load_combinations and it.name not in getattr(model, "load_cases", {}):
            part = combo_pattern_factors(model, it.name, seen)
        else:
            part = case_pattern_factors(model, it.name)
        for p, f in part.items():
            out[p] = out.get(p, 0.0) + float(it.scale) * f
    return {p: f for p, f in out.items() if abs(f) > 1e-12}


class ShowLoadsDialog(QtWidgets.QDialog):
    def __init__(self, model: Model, kinds: Iterable[str], factors: Optional[dict] = None,
                 title: str = "", parent=None):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Show Loads")
        lay = QtWidgets.QVBoxLayout(self)

        g = QtWidgets.QGroupBox("Load types")
        gl = QtWidgets.QVBoxLayout(g)
        self.kind_boxes: Dict[str, QtWidgets.QCheckBox] = {}
        kinds = set(kinds)
        from .mesh_dialog import help_button
        for key, label in KIND_ROWS:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(key in kinds)
            if key == "applied":  # "?" - what it is
                hl = QtWidgets.QHBoxLayout()
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(cb)
                self.applied_help = help_button(APPLIED_HELP, "Loads as they reach the members", self)
                hl.addWidget(self.applied_help)
                hl.addStretch(1)
                gl.addLayout(hl)
            else:
                gl.addWidget(cb)
            self.kind_boxes[key] = cb
        self.kind_boxes["applied"].toggled.connect(self._applied_toggled)
        for k in ("area", "panel"):
            self.kind_boxes[k].toggled.connect(lambda on: on and self.kind_boxes["applied"].setChecked(False))
        self.generated = QtWidgets.QCheckBox(GENERATED_LABEL)
        has_auto = any(getattr(p, "auto_load", None) for p in model.load_patterns.values())
        self.generated.setEnabled(has_auto)
        if not has_auto:
            self.generated.setToolTip("No load pattern of this model generates lateral loads yet "
                                      "(automatic seismic / wind loads are planned)")
            self.generated.setText(GENERATED_LABEL + "  - none defined")
        gl.addWidget(self.generated)
        row = QtWidgets.QHBoxLayout()
        for text, keys in (("All", {"joint", "frame", "area", "panel"}), ("None", set())):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, kk=keys: [cb.setChecked(k in kk) for k, cb in self.kind_boxes.items()])
            row.addWidget(b)
        row.addStretch(1)
        gl.addLayout(row)
        lay.addWidget(g)

        w = QtWidgets.QGroupBox("Loads of")
        wl = QtWidgets.QGridLayout(w)
        self.by_pattern = QtWidgets.QRadioButton("Load patterns (tick one or more)")
        self.by_case = QtWidgets.QRadioButton("Load case")
        self.by_combo = QtWidgets.QRadioButton("Load combination")
        self.patterns = QtWidgets.QListWidget()
        self.patterns.setMaximumHeight(160)
        sel = dict(factors or {})
        for name, lp in model.load_patterns.items():
            it = QtWidgets.QListWidgetItem(f"{name}   ({lp.type})")
            it.setData(QtCore.Qt.UserRole, name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if name in sel else QtCore.Qt.Unchecked)
            self.patterns.addItem(it)
        self.case_box = QtWidgets.QComboBox()
        self.case_box.addItems([n for n, c in getattr(model, "load_cases", {}).items()
                                if isinstance(getattr(c, "loads", None), dict) and c.loads])
        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(list(model.load_combinations))
        wl.addWidget(self.by_pattern, 0, 0, 1, 2)
        wl.addWidget(self.patterns, 1, 0, 1, 2)
        pr = QtWidgets.QHBoxLayout()
        for text, state in (("All patterns", QtCore.Qt.Checked), ("No pattern", QtCore.Qt.Unchecked)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, s=state: [self.patterns.item(i).setCheckState(s)
                                                        for i in range(self.patterns.count())])
            pr.addWidget(b)
        pr.addStretch(1)
        wl.addLayout(pr, 2, 0, 1, 2)
        wl.addWidget(self.by_case, 3, 0)
        wl.addWidget(self.case_box, 3, 1)
        wl.addWidget(self.by_combo, 4, 0)
        wl.addWidget(self.combo_box, 4, 1)
        self.by_case.setEnabled(self.case_box.count() > 0)
        self.by_combo.setEnabled(self.combo_box.count() > 0)
        if title.startswith("Case ") and title[5:] in [self.case_box.itemText(i) for i in range(self.case_box.count())]:
            self.by_case.setChecked(True)
            self.case_box.setCurrentText(title[5:])
        elif title.startswith("Combination ") and title[12:] in model.load_combinations:
            self.by_combo.setChecked(True)
            self.combo_box.setCurrentText(title[12:])
        else:
            self.by_pattern.setChecked(True)
        for rb, wdg in ((self.by_pattern, self.patterns), (self.by_case, self.case_box),
                        (self.by_combo, self.combo_box)):
            rb.toggled.connect(lambda on, wdg=wdg: wdg.setEnabled(on))
            wdg.setEnabled(rb.isChecked())
        lay.addWidget(w)
        note = QtWidgets.QLabel("A case or combination shows its patterns' loads times their factors "
                                "(each load is drawn where it is assigned; loads are not added up).")
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(placeholder-text);")
        lay.addWidget(note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.hide_button = bb.addButton("Hide Loads", QtWidgets.QDialogButtonBox.ResetRole)
        self.hide_button.clicked.connect(self._hide_all)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.hidden = False
        from .fit import install
        install(self)

    def _applied_toggled(self, on: bool):
        if on:  # loads as they reach the members: not also the panels / floor loads themselves
            for k in ("area", "panel"):
                self.kind_boxes[k].setChecked(False)

    def _hide_all(self):
        self.hidden = True
        self.accept()

    def kinds(self) -> Set[str]:
        return {k for k, cb in self.kind_boxes.items() if cb.isChecked()}

    def selection(self):
        """(factors {pattern: factor}, title) of the choice; ValueError if it has no loads."""
        m = self.model
        if self.by_case.isChecked():
            name = self.case_box.currentText()
            return case_pattern_factors(m, name), f"Case {name}"
        if self.by_combo.isChecked():
            name = self.combo_box.currentText()
            return combo_pattern_factors(m, name), f"Combination {name}"
        names = [self.patterns.item(i).data(QtCore.Qt.UserRole) for i in range(self.patterns.count())
                 if self.patterns.item(i).checkState() == QtCore.Qt.Checked]
        if not names:
            raise ValueError("Tick at least one load pattern")
        return {n: 1.0 for n in names}, " + ".join(names)

    def accept(self):
        if not self.hidden:
            try:
                self.selection()
            except (ValueError, KeyError) as exc:
                QtWidgets.QMessageBox.warning(self, "Show Loads", str(exc))
                return
        super().accept()
