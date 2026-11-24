"""Edit > Change Labels: new labels for the selected objects - a numbering (prefix,
next number, increment, suffix) or typed one by one in the table, checked for labels
two objects would share on a story before anything changes."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from . import commands as C

KIND_WORDS = {"frame": "Frames (beams, columns, braces)", "area": "Shells (floors, walls)", "point": "Joints",
              "strip": "Design strips"}


class ChangeLabelsDialog(QtWidgets.QDialog):
    def __init__(self, doc, selection: Dict[str, Sequence[str]], parent=None):
        """``selection``: kind -> names of the selected objects of that kind."""
        super().__init__(parent)
        self.setWindowTitle("Change Labels")
        self.doc = doc
        self.selection = {k: [n for n in v if n in getattr(doc.model, C.LABEL_KINDS[k])]
                          for k, v in selection.items()}
        self.selection = {k: v for k, v in self.selection.items() if v}
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QFormLayout()
        self.kind = QtWidgets.QComboBox()
        for k in ("frame", "area", "point", "strip"):
            if k in self.selection:
                self.kind.addItem(f"{KIND_WORDS[k]} - {len(self.selection[k])} selected", k)
        top.addRow("Objects", self.kind)
        row = QtWidgets.QHBoxLayout()
        self.prefix = QtWidgets.QLineEdit()
        self.prefix.setPlaceholderText("e.g. 2B")
        self.start = QtWidgets.QSpinBox()
        self.start.setRange(0, 10 ** 7)
        self.start.setValue(1)
        self.step = QtWidgets.QSpinBox()
        self.step.setRange(1, 10 ** 6)
        self.step.setValue(1)
        self.suffix = QtWidgets.QLineEdit()
        self.suffix.setPlaceholderText("e.g. -A")
        self.width = QtWidgets.QSpinBox()
        self.width.setRange(0, 6)
        self.width.setToolTip("Digits (zero padded): 2 gives 01, 02 ...; 0 = as the number is")
        for lab, w in (("Prefix", self.prefix), ("Next number", self.start), ("Increment", self.step),
                       ("Suffix", self.suffix), ("Digits", self.width)):
            row.addWidget(QtWidgets.QLabel(lab))
            row.addWidget(w)
        top.addRow("Numbering", row)
        opts = QtWidgets.QHBoxLayout()
        self.order = QtWidgets.QComboBox()
        self.order.addItem("by position in plan (Y, then X)", "position")
        self.order.addItem("in the order of the present labels", "label")
        self.per_story = QtWidgets.QCheckBox("Each story starts at the next number again")
        self.per_story.setChecked(True)
        self.per_story.setToolTip("As ETABS: the same label (C1) on every story; off: one count over the stories")
        number = QtWidgets.QPushButton("Number the list")
        number.setToolTip("Fill the New label column with the numbering above")
        number.clicked.connect(self.renumber)
        opts.addWidget(QtWidgets.QLabel("Order"))
        opts.addWidget(self.order)
        opts.addWidget(self.per_story)
        opts.addStretch(1)
        opts.addWidget(number)
        top.addRow("", opts)
        lay.addLayout(top)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Object", "Story", "Present label", "New label"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(lambda *_: self._check())
        lay.addWidget(self.table, 1)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.kind.currentIndexChanged.connect(lambda *_: self._fill())
        for w in (self.prefix, self.suffix):
            w.textEdited.connect(lambda *_: self.renumber())
        for w in (self.start, self.step, self.width):
            w.valueChanged.connect(lambda *_: self.renumber())
        self.order.currentIndexChanged.connect(lambda *_: self.renumber())
        self.per_story.toggled.connect(lambda *_: self.renumber())
        self._fill()
        self.resize(760, 520)
        from .fit import install
        install(self)

    # ------------------------------------------------------------------ table
    def current_kind(self) -> str:
        return self.kind.currentData()

    def _fill(self):
        """The selected objects of the kind, with their present labels (new = present)."""
        k = self.current_kind()
        if k is None:
            return
        m = self.doc.model
        table = getattr(m, C.LABEL_KINDS[k])
        rows = C.number_labels(m, k, self.selection[k], order="position")  # (display order)
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for r, (n, _) in enumerate(rows):
            o = table[n]
            for c, text in enumerate((n, o.story or "-", C.label_of(o), C.label_of(o))):
                it = QtWidgets.QTableWidgetItem(text)
                if c < 3:
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(r, c, it)
        self.table.blockSignals(False)
        self._check()

    def renumber(self):
        k = self.current_kind()
        if k is None:
            return
        names = [self.table.item(r, 0).text() for r in range(self.table.rowCount())]
        new = dict(C.number_labels(self.doc.model, k, names, self.prefix.text().strip(), self.start.value(),
                                   self.step.value(), self.suffix.text().strip(), self.order.currentData(),
                                   self.per_story.isChecked(), self.width.value()))
        self.table.blockSignals(True)
        for r, n in enumerate(names):
            self.table.item(r, 3).setText(new.get(n, ""))
        self.table.blockSignals(False)
        self._check()

    def new_labels(self) -> Dict[str, str]:
        return {self.table.item(r, 0).text(): self.table.item(r, 3).text().strip()
                for r in range(self.table.rowCount())}

    def _check(self) -> List[Tuple[str, str]]:
        """Labels two objects would share on a story: shown in red, said below."""
        k = self.current_kind()
        if k is None:
            return []
        new = {n: (lab or n) for n, lab in self.new_labels().items()}
        clashes = C.label_clashes(self.doc.model, k, new)
        bad = set(clashes)
        self.table.blockSignals(True)
        red = QtGui.QBrush(QtGui.QColor(200, 30, 30))
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 3)
            key = ((it.text().strip() or self.table.item(r, 0).text()),
                   None if self.table.item(r, 1).text() == "-" else self.table.item(r, 1).text())
            it.setForeground(red if key in bad else QtGui.QBrush())
        self.table.blockSignals(False)
        if clashes:
            shown = ", ".join(f"{lab} ({st or 'no story'})" for lab, st in clashes[:6])
            self.note.setText(f"Two objects would have the same label on one story: {shown}"
                              f"{' ...' if len(clashes) > 6 else ''}.  Labels are names only (the analysis "
                              "does not use them), but the same label twice makes the tables and drawings "
                              "ambiguous.")
        else:
            self.note.setText("An empty New label gives the object back its name.  Labels are names only: the "
                              "analysis results are kept; Undo brings the old labels back.")
        return clashes

    def _ok(self):
        k = self.current_kind()
        if k is None:
            self.reject()
            return
        clashes = self._check()
        if clashes and QtWidgets.QMessageBox.question(
                self, self.windowTitle(), "Some objects would share a label on one story (shown in red).  "
                                          "Change the labels anyway?") != QtWidgets.QMessageBox.Yes:
            return
        try:
            n = C.change_labels(self.doc, k, self.new_labels())
        except C.ModelError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), exc.message)
            return
        self.changed = n
        self.accept()
