"""Forms of the design strips: Edit > Add/Edit Design Strips > Auto Design Strips /
Auto Strip Widths, and a strip's properties (right-click on a strip)."""
from __future__ import annotations

from typing import List, Optional

from PySide6 import QtCore, QtWidgets

from ..design.strips import StripOptions, floor_of
from .forms import display_units


def _du():
    du = display_units()
    k = du.factor("coordinates") if du is not None else 1.0
    unit = du.label("coordinates") if du is not None else ""
    return k, unit


def slab_stories(model) -> List[str]:
    """Stories (bottom up) with floor slabs at their level."""
    out = []
    for st in model.stories_bottom_up():
        try:
            floor, _ = floor_of(model, st.name)
        except Exception:
            continue
        if floor.polys:
            out.append(st.name)
    return out


class StoryList(QtWidgets.QListWidget):
    """Stories to work on, ticked."""

    def __init__(self, stories: List[str], ticked: List[str], parent=None):
        super().__init__(parent)
        for s in reversed(stories):  # top story first, as the story lists of the program
            it = QtWidgets.QListWidgetItem(s)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if s in ticked else QtCore.Qt.Unchecked)
            self.addItem(it)
        self.setMinimumHeight(min(220, 22 * max(3, len(stories)) + 8))

    def ticked(self) -> List[str]:
        return [self.item(k).text() for k in range(self.count()) if self.item(k).checkState() == QtCore.Qt.Checked][::-1]

    def set_all(self, on: bool):
        for k in range(self.count()):
            self.item(k).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)


def _story_box(stories, ticked, title="Stories"):
    box = QtWidgets.QGroupBox(title)
    v = QtWidgets.QVBoxLayout(box)
    lst = StoryList(stories, ticked)
    v.addWidget(lst)
    h = QtWidgets.QHBoxLayout()
    for text, on in (("All", True), ("None", False)):
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(lambda _=False, on=on: lst.set_all(on))
        h.addWidget(b)
    h.addStretch(1)
    v.addLayout(h)
    return box, lst


SOURCES = [("auto", "Supports and grid lines (recommended)"),
           ("supports", "Supports only (columns, walls, piles, stiff beams)"),
           ("grids", "Grid lines only (every grid line over the slab)")]


class AutoStripsDialog(QtWidgets.QDialog):
    """Auto Design Strips: where the strips come from and on which stories."""

    def __init__(self, model, current: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Design Strips")
        k, unit = _du()
        self._k = k
        lay = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Column strips along the lines of supports (columns, piles, walls, beams stiff enough to support "
            "the slab), widths 0.25 min(l1, l2) each side (ACI 318 8.4.1.5), middle strips between them.  "
            "Strips are design data: the analysis results are kept.")
        note.setWordWrap(True)
        lay.addWidget(note)
        stories = slab_stories(model)
        ticked = [current] if current in stories else stories
        box, self.stories = _story_box(stories, ticked)
        lay.addWidget(box)
        form = QtWidgets.QFormLayout()
        self.source = QtWidgets.QComboBox()
        for key, label in SOURCES:
            self.source.addItem(label, key)
        form.addRow("Lay the strips out from", self.source)
        self.layer_a = QtWidgets.QCheckBox("Layer A")
        self.layer_b = QtWidgets.QCheckBox("Layer B")
        self.layer_a.setChecked(True)
        self.layer_b.setChecked(True)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.layer_a)
        h.addWidget(self.layer_b)
        h.addStretch(1)
        w = QtWidgets.QWidget()
        w.setLayout(h)
        form.addRow("Layers", w)
        self.middle = QtWidgets.QCheckBox("Make the middle strips")
        self.middle.setChecked(True)
        form.addRow("", self.middle)
        self.keep = QtWidgets.QCheckBox("Keep the strips drawn by hand (the ones Auto made are replaced)")
        self.keep.setChecked(True)
        form.addRow("", self.keep)
        d = StripOptions()
        self.alpha = QtWidgets.QDoubleSpinBox()
        self.alpha.setRange(0.0, 100.0)
        self.alpha.setDecimals(2)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(d.beam_alpha)
        self.alpha.setToolTip("A beam is a line of supports of the slab when alpha_f l2 / l1 is at least this "
                              "(ACI 318 8.10.2.7: 1.0); a shallower beam is part of the slab (flat slab there)")
        form.addRow("Beam as a support: αf l2/l1 ≥", self.alpha)
        self.angle = QtWidgets.QDoubleSpinBox()
        self.angle.setRange(0.0, 45.0)
        self.angle.setDecimals(1)
        self.angle.setSuffix(" °")
        self.angle.setValue(d.align_angle)
        self.angle.setToolTip("Columns a strip joins may be offset: the strip turns at most this much "
                              "between one support and the next")
        form.addRow("Column strip turns at most", self.angle)
        self.tol = QtWidgets.QDoubleSpinBox()
        self.tol.setRange(0.0, 1e6)
        self.tol.setDecimals(3)
        if unit:
            self.tol.setSuffix(f" {unit}")
        self._m_per_unit = model.units.factor("length")
        self.tol.setValue(d.on_line_tol / self._m_per_unit * k)
        self.tol.setToolTip("A support this close to a grid line counts as on it")
        form.addRow("Support on a grid line within", self.tol)
        lay.addLayout(form)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if not stories:
            note.setText(note.text() + "\n\nThe model has no floor slab: draw the floors first.")
            bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        from .fit import install
        install(self)

    def _ok(self):
        if not self.stories.ticked():
            QtWidgets.QMessageBox.information(self, self.windowTitle(), "Tick the stories to lay the strips out on")
            return
        if not (self.layer_a.isChecked() or self.layer_b.isChecked()):
            QtWidgets.QMessageBox.information(self, self.windowTitle(), "Tick layer A, layer B or both")
            return
        self.accept()

    def options(self) -> StripOptions:
        layers = tuple(L for L, cb in (("A", self.layer_a), ("B", self.layer_b)) if cb.isChecked())
        return StripOptions(source=self.source.currentData(), beam_alpha=self.alpha.value(),
                            align_angle=self.angle.value(),
                            on_line_tol=self.tol.value() / self._k * self._m_per_unit,
                            middle=self.middle.isChecked(), keep_user=self.keep.isChecked(), layers=layers)

    def chosen_stories(self) -> List[str]:
        return self.stories.ticked()


class AutoWidthsDialog(QtWidgets.QDialog):
    """Auto Strip Widths: the column strips' widths again, the middle strips again."""

    def __init__(self, model, current: Optional[str] = None, selected=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Strip Widths")
        lay = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "The widths of the column strips are worked out again from the supports and the strips beside "
            "them (0.25 min(l1, l2) each side, ACI 318 8.4.1.5) - after a strip was moved or reshaped, or "
            "the supports changed.  A strip drawn with specified widths keeps them unless it is selected.")
        note.setWordWrap(True)
        lay.addWidget(note)
        stories = sorted({ds.story for ds in model.design_strips.values() if ds.story in model.stories},
                         key=lambda s: model.stories[s].elevation)
        ticked = [current] if current in stories else stories
        box, self.stories = _story_box(stories, ticked)
        lay.addWidget(box)
        self.which = QtWidgets.QComboBox()
        self.which.addItem("Column strips with automatic widths", "auto")
        self.selected = [n for n in selected if n in model.design_strips]
        if self.selected:
            self.which.addItem(f"The {len(self.selected)} selected strip(s)", "selected")
            self.which.setCurrentIndex(1)
        form = QtWidgets.QFormLayout()
        form.addRow("Work out the widths of", self.which)
        self.middle = QtWidgets.QCheckBox("Make the middle strips again (the ones drawn by hand stay)")
        self.middle.setChecked(True)
        form.addRow("", self.middle)
        lay.addLayout(form)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if not stories:
            note.setText(note.text() + "\n\nThere are no design strips yet: Auto Design Strips or Draw Design "
                                       "Strips makes them.")
            bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        from .fit import install
        install(self)

    def names(self):
        return list(self.selected) if self.which.currentData() == "selected" else None

    def chosen_stories(self) -> List[str]:
        return self.stories.ticked()


class StripDialog(QtWidgets.QDialog):
    """A design strip's properties: name, layer, type, widths (automatic or per
    segment) and the points of its centre line."""

    def __init__(self, doc, name: str, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.name = name
        ds = doc.model.design_strips[name]
        self.setWindowTitle(f"Design Strip {name}")
        k, unit = _du()
        self._k = k
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.ed_name = QtWidgets.QLineEdit(name)
        form.addRow("Name", self.ed_name)
        self.ed_label = QtWidgets.QLineEdit(ds.label or "")
        self.ed_label.setPlaceholderText("(the name)")
        form.addRow("Label", self.ed_label)
        form.addRow("Story", QtWidgets.QLabel(ds.story or "-"))
        self.layer = QtWidgets.QComboBox()
        self.layer.addItems(["A", "B", "Other"])
        self.layer.setCurrentText(ds.layer)
        form.addRow("Layer", self.layer)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Column", "Middle"])
        self.kind.setCurrentText(ds.kind)
        form.addRow("Strip type", self.kind)
        self.auto = QtWidgets.QCheckBox("Automatic widths (worked out again when the strip changes)")
        self.auto.setChecked(ds.auto_width)
        form.addRow("", self.auto)
        lay.addLayout(form)
        u = f" [{unit}]" if unit else ""
        lay.addWidget(QtWidgets.QLabel("Widths of each segment (left / right looking from point k to k + 1)"))
        self.widths = QtWidgets.QTableWidget(len(ds.widths), 4)
        self.widths.setHorizontalHeaderLabels([f"Start left{u}", f"Start right{u}", f"End left{u}", f"End right{u}"])
        self.widths.setVerticalHeaderLabels([f"{j + 1}-{j + 2}" for j in range(len(ds.widths))])
        for r, w in enumerate(ds.widths):
            for c, v in enumerate(w):
                self.widths.setItem(r, c, QtWidgets.QTableWidgetItem(f"{v * k:.4g}"))
        self.widths.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self.widths)
        lay.addWidget(QtWidgets.QLabel("Centre line points"))
        self.pts = QtWidgets.QTableWidget(len(ds.points), 2)
        self.pts.setHorizontalHeaderLabels([f"X{u}", f"Y{u}"])
        for r, p in enumerate(ds.points):
            for c in range(2):
                self.pts.setItem(r, c, QtWidgets.QTableWidgetItem(f"{p[c] * k:.6g}"))
        self.pts.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self.pts)
        self.auto.toggled.connect(self._sync)
        self._sync(self.auto.isChecked())
        self._orig = (self._read_widths(), self._read_points())
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        from .fit import install
        install(self)

    def _sync(self, auto: bool):
        self.widths.setEnabled(not auto)

    def _cell(self, table, r, c) -> float:
        it = table.item(r, c)
        try:
            return float((it.text() if it is not None else "0").replace(",", "."))
        except ValueError:
            raise ValueError(f"Row {r + 1}, column {c + 1}: not a number")

    def _read_widths(self):
        return [tuple(self._cell(self.widths, r, c) / self._k for c in range(4)) for r in range(self.widths.rowCount())]

    def _read_points(self):
        return [tuple(self._cell(self.pts, r, c) / self._k for c in range(2)) for r in range(self.pts.rowCount())]

    def _ok(self):
        from ..model import ModelError
        from .strip_commands import set_strip
        try:
            ws, pts = self._read_widths(), self._read_points()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        ds = self.doc.model.design_strips.get(self.name)
        if ds is None:
            self.reject()
            return
        z = float(ds.points[0][2])
        changed_w = ws != self._orig[0]
        changed_p = pts != self._orig[1]
        try:
            set_strip(self.doc, self.name, new_name=self.ed_name.text(), layer=self.layer.currentText(),
                      kind=self.kind.currentText(), label=self.ed_label.text().strip(),
                      auto_width=self.auto.isChecked(),
                      widths=ws if (changed_w and not self.auto.isChecked()) else None,
                      points=[(x, y, z) for x, y in pts] if changed_p else None)
        except ModelError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), exc.message)
            return
        self.accept()
