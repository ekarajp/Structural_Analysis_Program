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
        self.digits_box = QtWidgets.QSpinBox()
        self.digits_box.setRange(0, 6)
        self.digits_box.setToolTip("Digits (zero padded): 2 gives 01, 02 ...; 0 = as the number is")
        for lab, w in (("Prefix", self.prefix), ("Next number", self.start), ("Increment", self.step),
                       ("Suffix", self.suffix), ("Digits", self.digits_box)):
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
        for w in (self.start, self.step, self.digits_box):
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
                                   self.per_story.isChecked(), self.digits_box.value()))
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


TYPE_WORDS = {"Beam": "Beams", "Column": "Columns", "Brace": "Braces", "Null": "Null lines", "Rigid": "Rigid links",
              "Floor": "Floors", "Wall": "Walls", "Ramp": "Ramps", "Opening": "Openings", "Joint": "Joints"}
CORNER_WORDS = {"bottom-left": "bottom left", "top-left": "top left", "bottom-right": "bottom right",
                "top-right": "top right"}


class AutoRelabelDialog(QtWidgets.QDialog):
    """Edit > Auto Relabel All: the label rules (kept with the model) and a preview."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        import dataclasses
        from ..model.objects import DEFAULT_LABEL_PREFIXES, LABEL_TYPES, LabelRules
        self.setWindowTitle("Auto Relabel All")
        self.doc = doc
        self._defaults = LabelRules()
        r = dataclasses.replace(doc.model.label_rules, prefixes=dict(doc.model.label_rules.prefixes))
        lay = QtWidgets.QVBoxLayout(self)
        body = QtWidgets.QHBoxLayout()
        # prefixes per type
        box = QtWidgets.QGroupBox("Prefix of each type")
        g = QtWidgets.QGridLayout(box)
        self.prefix: Dict[str, QtWidgets.QLineEdit] = {}
        for k, t in enumerate(LABEL_TYPES):
            e = QtWidgets.QLineEdit(str(r.prefixes.get(t, DEFAULT_LABEL_PREFIXES.get(t, ""))))
            e.setMaximumWidth(80)
            e.textEdited.connect(lambda *_: self._preview())
            self.prefix[t] = e
            g.addWidget(QtWidgets.QLabel(TYPE_WORDS[t]), k, 0)
            g.addWidget(e, k, 1)
        body.addWidget(box)
        # rules
        rb = QtWidgets.QGroupBox("Numbering")
        f = QtWidgets.QFormLayout(rb)
        self.corner = QtWidgets.QComboBox()
        for c in C.CORNERS:
            self.corner.addItem(f"from the {CORNER_WORDS[c]} of the plan", c)
        self.corner.setCurrentIndex(max(0, self.corner.findData(r.corner)))
        f.addRow("Start", self.corner)
        self.order = QtWidgets.QComboBox()
        self.order.addItem("row by row (along X, then the next row in Y)", "YX")
        self.order.addItem("column by column (along Y, then the next column in X)", "XY")
        self.order.setCurrentIndex(max(0, self.order.findData(r.order)))
        f.addRow("Order", self.order)
        self.start = QtWidgets.QSpinBox()
        self.start.setRange(0, 10 ** 6)
        self.start.setValue(int(r.start))
        f.addRow("First number", self.start)
        self.step = QtWidgets.QSpinBox()
        self.step.setRange(1, 10 ** 5)
        self.step.setValue(int(r.step))
        f.addRow("Increment", self.step)
        self.digits = QtWidgets.QSpinBox()
        self.digits.setRange(0, 6)
        self.digits.setValue(int(r.digits))
        self.digits.setToolTip("Zero padded: 2 gives B01, B02 ...; 0 = as the number is")
        f.addRow("Digits", self.digits)
        self.per_story = QtWidgets.QCheckBox("Each story starts again (the same label on every story, ETABS)")
        self.per_story.setChecked(bool(r.per_story))
        f.addRow("", self.per_story)
        self.by_type = QtWidgets.QCheckBox("Each type its own count (B1, B2 ... C1, C2 ...)")
        self.by_type.setChecked(bool(r.by_type))
        self.by_type.setToolTip("Off: one count over beams, columns, floors ... (B1, C2, F3 ...); joints count "
                                "on their own")
        f.addRow("", self.by_type)
        self.story_tag = QtWidgets.QComboBox()
        for key, text in (("none", "no"), ("number", "story number before (2-B1)"),
                          ("name", "story name before (Story2-B1)")):
            self.story_tag.addItem(text, key)
        self.story_tag.setCurrentIndex(max(0, self.story_tag.findData(r.story_tag)))
        f.addRow("Story in the label", self.story_tag)
        self.sep = QtWidgets.QLineEdit(r.story_sep)
        self.sep.setMaximumWidth(60)
        f.addRow("Between story and label", self.sep)
        kinds = QtWidgets.QHBoxLayout()
        self.k_frames = QtWidgets.QCheckBox("Frames")
        self.k_shells = QtWidgets.QCheckBox("Shells")
        self.k_joints = QtWidgets.QCheckBox("Joints")
        for cb, on in ((self.k_frames, r.frames), (self.k_shells, r.shells), (self.k_joints, r.joints)):
            cb.setChecked(bool(on))
            kinds.addWidget(cb)
        kinds.addStretch(1)
        f.addRow("Relabel", kinds)
        body.addWidget(rb, 1)
        lay.addLayout(body)
        self.preview = QtWidgets.QLabel()
        self.preview.setWordWrap(True)
        lay.addWidget(self.preview)
        for w in (self.corner, self.order, self.story_tag):
            w.currentIndexChanged.connect(lambda *_: self._preview())
        for w in (self.start, self.step, self.digits):
            w.valueChanged.connect(lambda *_: self._preview())
        for w in (self.per_story, self.by_type, self.k_frames, self.k_shells, self.k_joints):
            w.toggled.connect(lambda *_: self._preview())
        self.sep.textEdited.connect(lambda *_: self._preview())
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        dflt = bb.addButton("Defaults", QtWidgets.QDialogButtonBox.ResetRole)
        dflt.clicked.connect(self.defaults)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._preview()
        from .fit import install
        install(self)

    def rules(self):
        from ..model.objects import LabelRules
        return LabelRules(prefixes={t: e.text().strip() for t, e in self.prefix.items()},
                          order=self.order.currentData(), corner=self.corner.currentData(),
                          per_story=self.per_story.isChecked(), by_type=self.by_type.isChecked(),
                          start=self.start.value(), step=self.step.value(), digits=self.digits.value(),
                          story_tag=self.story_tag.currentData(), story_sep=self.sep.text(),
                          joints=self.k_joints.isChecked(), frames=self.k_frames.isChecked(),
                          shells=self.k_shells.isChecked())

    def defaults(self):
        d = self._defaults
        for t, e in self.prefix.items():
            e.setText(d.prefixes.get(t, ""))
        self.corner.setCurrentIndex(self.corner.findData(d.corner))
        self.order.setCurrentIndex(self.order.findData(d.order))
        self.start.setValue(d.start)
        self.step.setValue(d.step)
        self.digits.setValue(d.digits)
        self.per_story.setChecked(d.per_story)
        self.by_type.setChecked(d.by_type)
        self.story_tag.setCurrentIndex(self.story_tag.findData(d.story_tag))
        self.sep.setText(d.story_sep)
        for cb in (self.k_frames, self.k_shells, self.k_joints):
            cb.setChecked(True)
        self._preview()

    def _preview(self):
        m = self.doc.model
        plan = C.relabel_plan(m, self.rules())
        sample = []
        tables = {"point": m.points, "frame": m.frames, "area": m.areas}
        seen_types = set()
        for (kind, name), lab in plan.items():
            o = tables[kind][name]
            t = getattr(o, "type", "Joint") if kind != "point" else "Joint"
            if t not in seen_types:
                seen_types.add(t)
                sample.append(f"{TYPE_WORDS.get(t, t)}: {lab}")
        custom = sum(1 for (kind, name) in plan if tables[kind][name].label)
        text = ("First labels: " + ", ".join(sample[:8]) + (" ..." if len(sample) > 8 else "")) if sample else \
            "Nothing to relabel (tick Frames / Shells / Joints)."
        if custom:
            text += f"\n{custom} label(s) the objects have now - also the ones typed by hand - are replaced."
        text += "\nThe rules are kept with the model.  The analysis results stay (labels are names only); " \
                "Undo brings the old labels back."
        self.preview.setText(text)
        return plan

    def _ok(self):
        if not (self.k_frames.isChecked() or self.k_shells.isChecked() or self.k_joints.isChecked()):
            QtWidgets.QMessageBox.information(self, self.windowTitle(), "Tick Frames, Shells or Joints")
            return
        try:
            self.count = C.auto_relabel(self.doc, self.rules())
        except C.ModelError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), exc.message)
            return
        self.accept()
