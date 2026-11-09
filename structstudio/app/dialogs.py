"""Define / Assign dialogs for the desktop app."""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence

from PySide6 import QtCore, QtWidgets

from ..model import FIXED, FrameSection, Model, ModelError, PINNED, ShellSection
from ..model.loads import LinearStaticCase, LoadCombination, LoadPattern, ModalCase
from ..model.objects import GridLine, Story
from .document import Document
from .panels import _parse
from .templates import new_building

FREE6 = (False, False, False, False, False, False)


# ------------------------------------------------------------ generic form
class DataclassForm(QtWidgets.QWidget):
    """Editable form for a dataclass instance (simple fields only)."""

    def __init__(self, obj=None, skip: Sequence[str] = (), parent=None):
        super().__init__(parent)
        self.form = QtWidgets.QFormLayout(self)
        self.editors: Dict[str, QtWidgets.QWidget] = {}
        self.obj = None
        self.skip = set(skip)
        if obj is not None:
            self.set_object(obj)

    def set_object(self, obj):
        from .panels import PropertyPanel
        while self.form.rowCount():
            self.form.removeRow(0)
        self.editors.clear()
        self.obj = obj
        from .panels import field_unit, scaled
        self.scales = {}
        for f in dataclasses.fields(obj):
            if f.name in self.skip:
                continue
            value = getattr(obj, f.name)
            k, unit = field_unit(obj, f)  # shown in the display units (Options > Display Units)
            editor = PropertyPanel._make_editor(None, scaled(value, k))
            if editor is None or isinstance(editor, QtWidgets.QLabel):
                continue
            self.editors[f.name] = editor
            self.scales[f.name] = (k, editor.text() if isinstance(editor, QtWidgets.QLineEdit) else None)
            from .symbols import rich
            lab = QtWidgets.QLabel(rich(f"{f.name} [{unit}]" if unit else f.name))
            lab.setTextFormat(QtCore.Qt.RichText)
            self.form.addRow(lab, editor)

    def values(self) -> dict:
        from .panels import scaled
        out = {}
        for name, ed in self.editors.items():
            old = getattr(self.obj, name)
            k, shown = self.scales.get(name, (1.0, None))
            if shown is not None and isinstance(ed, QtWidgets.QLineEdit) and ed.text() == shown:
                out[name] = old  # untouched
                continue
            out[name] = scaled(_parse(ed, scaled(old, k)), 1.0 / k)
        return out

    def apply(self, obj=None):
        obj = obj or self.obj
        for k, v in self.values().items():
            setattr(obj, k, v)
        return obj


# ------------------------------------------------------------ define dialog
class DefineDialog(QtWidgets.QDialog):
    """List + form editor for one of the model's definition tables
    (materials, sections, load patterns, cases, combinations ...)."""

    def __init__(self, doc: Document, attr: str, title: str, factory: Callable[[str, Model], object],
                 skip: Sequence[str] = ("name",), parent=None):
        super().__init__(parent)
        self.doc, self.attr, self.factory = doc, attr, factory
        self.setWindowTitle(title)
        self.resize(640, 440)
        self.table: Dict[str, object] = {k: dataclasses.replace(v) for k, v in getattr(doc.model, attr).items()}
        self.deleted: set = set()
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.addItems(list(self.table))
        self.list.currentTextChanged.connect(self._show)
        left.addWidget(self.list, 1)
        btns = QtWidgets.QHBoxLayout()
        for text, slot in (("Add", self.add), ("Copy", self.copy), ("Delete", self.delete)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            btns.addWidget(b)
        left.addLayout(btns)
        lay.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        self.form = DataclassForm(skip=skip)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.form)
        right.addWidget(scroll, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        right.addWidget(bb)
        lay.addLayout(right, 2)
        self._current: Optional[str] = None
        if self.table:
            self.list.setCurrentRow(0)

    def _commit_form(self):
        if self._current and self._current in self.table and self.form.obj is not None:
            try:
                self.form.apply(self.table[self._current])
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))

    def _show(self, name: str):
        self._commit_form()
        self._current = name or None
        if name and name in self.table:
            self.form.set_object(self.table[name])

    def _new_name(self, base: str) -> str:
        n = 1
        while f"{base}{n}" in self.table:
            n += 1
        return f"{base}{n}"

    def add(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New", "Name:", text=self._new_name("New"))
        if not ok or not name or name in self.table:
            return
        try:
            self.table[name] = self.factory(name, self.doc.model)
        except ModelError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot add", exc.message)
            return
        self.list.addItem(name)
        self.list.setCurrentRow(self.list.count() - 1)

    def copy(self):
        if not self._current:
            return
        self._commit_form()
        name = self._new_name(self._current + "-")
        obj = dataclasses.replace(self.table[self._current])
        if hasattr(obj, "name"):
            obj.name = name
        self.table[name] = obj
        self.list.addItem(name)
        self.list.setCurrentRow(self.list.count() - 1)

    def delete(self):
        if not self._current:
            return
        name = self._current
        self.table.pop(name, None)
        self.deleted.add(name)
        self._current = None
        self.list.takeItem(self.list.currentRow())

    def accept(self):
        self._commit_form()
        table, attr = self.table, self.attr

        def mutate(model: Model):
            target = getattr(model, attr)
            target.clear()
            target.update(table)

        self.doc.edit(mutate)
        super().accept()


# ----------------------------------------------------- factories for defines
def material_factory(name: str, model: Model):
    from ..model.properties import Material
    return Material(name=name, type="Concrete", E=25e6, nu=0.2, weight_density=24.0)


def shell_section_factory(name: str, model: Model):
    mat = next(iter(model.materials), None)
    if mat is None:
        raise ModelError("NO_MATERIAL", "Define a material first")
    return ShellSection(name, mat, type="Slab", thickness=0.15)


def load_pattern_factory(name: str, model: Model):
    return LoadPattern(name, "Other", 0.0)


def static_case_factory(name: str, model: Model):
    pat = next(iter(model.load_patterns), None)
    return LinearStaticCase(name, {pat: 1.0} if pat else {})


def modal_case_factory(name: str, model: Model):
    return ModalCase(name, max_modes=12)


def combo_factory(name: str, model: Model):
    c = LoadCombination(name)
    for case in list(model.load_cases)[:1]:
        c.add(case, 1.0)
    return c


# ------------------------------------------------------------- new model
class NewModelDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Model")
        f = QtWidgets.QFormLayout(self)
        from ..units import PRESETS
        self.units = QtWidgets.QComboBox()  # the consistent unit sets of the units button
        self.units.addItems([" ".join(p) for p in PRESETS])
        self.units.setCurrentText("kN m C")
        self.units.setToolTip("Force, length, temperature - can be changed any time (units button, lower right)")
        self.xs = QtWidgets.QLineEdit("6, 6, 6")
        self.ys = QtWidgets.QLineEdit("5, 5")
        self.hs = QtWidgets.QLineEdit("3.5, 3.5, 3.5")
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Grid only", "Frame + slabs (fixed base)"])
        self.props = QtWidgets.QCheckBox("Add default concrete properties and load cases")
        self.props.setChecked(True)
        self.grid_mode = GridModeSwitch()
        f.addRow("Units", self.units)
        f.addRow("X bay widths", self.xs)
        f.addRow("Y bay widths", self.ys)
        f.addRow("Grid system", self.grid_mode)
        f.addRow("Story / level heights (bottom up)", self.hs)
        f.addRow("Template", self.kind)
        f.addRow(self.props)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    @staticmethod
    def _floats(text: str) -> List[float]:
        return [float(t) for t in text.replace(";", ",").split(",") if t.strip()]

    def build(self) -> Model:
        return new_building(units=self.units.currentText(), x_spacings=self._floats(self.xs.text()),
                            y_spacings=self._floats(self.ys.text()), story_heights=self._floats(self.hs.text()),
                            properties=self.props.isChecked(), grid_only=self.kind.currentIndex() == 0,
                            grid_mode=self.grid_mode.mode())


# ------------------------------------------------------ grids and stories
GRID_MODE_LABELS = {"story": "X · Y · Story", "xyz": "X · Y · Z"}


def apply_grid_mode(doc: Document, mode: str) -> bool:
    """Switch the document's grid system (undoable).  Returns True when it changed."""
    if doc.model.grid_mode == mode:
        return False
    doc.edit(lambda m: m.set_grid_mode(mode))
    return True


class GridModeSwitch(QtWidgets.QTabBar):
    """Two-tab switch between the X-Y-Story and X-Y-Z grid systems."""

    mode_changed = QtCore.Signal(str)
    MODES = ("story", "xyz")

    def __init__(self, parent=None):
        super().__init__(parent)
        for mode in self.MODES:
            self.addTab(GRID_MODE_LABELS[mode])
        self.setToolTip("Grid system: X, Y grid lines + stories, or X, Y, Z grid lines")
        self.setDrawBase(False)
        self.setExpanding(False)
        self.currentChanged.connect(lambda i: self.mode_changed.emit(self.MODES[i]))

    def set_mode(self, mode: str):
        self.blockSignals(True)
        self.setCurrentIndex(self.MODES.index(mode if mode in self.MODES else "story"))
        self.blockSignals(False)

    def mode(self) -> str:
        return self.MODES[self.currentIndex()]


class GridStoriesDialog(QtWidgets.QDialog):
    """Edit Grid Lines: X / Y grid tabs plus a Story tab (X-Y-Story mode) or a Z grid
    tab (X-Y-Z mode).  The switch at the top changes the grid system."""

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setWindowTitle("Grid Lines and Levels")
        self.resize(620, 520)
        m = doc.model
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Grid system"))
        self.switch = GridModeSwitch()
        self.switch.set_mode(m.grid_mode)
        self.switch.mode_changed.connect(self._mode_changed)
        top.addWidget(self.switch)
        top.addStretch(1)
        lay.addLayout(top)
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.x_table = self._table(["Name", "X ordinate"], [(g.name, g.ordinate) for g in m.grid_lines if g.direction == "X"])
        self.y_table = self._table(["Name", "Y ordinate"], [(g.name, g.ordinate) for g in m.grid_lines if g.direction == "Y"])
        stories = m.stories_bottom_up()
        # Story tab: name, height (bottom-up; the first row is the base level with its elevation)
        rows = [(s.name, s.elevation if k == 0 else s.height) for k, s in enumerate(stories)]
        self.story_table = self._table(["Story (bottom up)", "Height (base: elevation)"], rows)
        self.z_table = self._table(["Name", "Z ordinate"], m.levels())
        self.tabs.addTab(self._with_buttons(self.x_table, ("", 0.0)), "X Grid")
        self.tabs.addTab(self._with_buttons(self.y_table, ("", 0.0)), "Y Grid")
        self.story_page = self._with_buttons(self.story_table, ("", 3.0))
        self.z_page = self._with_buttons(self.z_table, ("", 0.0))
        self._level_tab = None
        self._show_level_tab(m.grid_mode)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    # ------------------------------------------------------------ widgets
    @staticmethod
    def _table(headers, rows) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setStretchLastSection(True)
        for row in rows:
            GridStoriesDialog._add_row(t, row)
        return t

    @staticmethod
    def _add_row(table: QtWidgets.QTableWidget, values):
        r = table.rowCount()
        table.insertRow(r)
        for c, v in enumerate(values):
            table.setItem(r, c, QtWidgets.QTableWidgetItem(f"{v:g}" if isinstance(v, float) else str(v)))

    def _with_buttons(self, table, blank) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(table, 1)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add")
        add.clicked.connect(lambda _=False: self._add_row(table, blank))
        rm = QtWidgets.QPushButton("Remove")
        rm.clicked.connect(lambda _=False: table.removeRow(table.currentRow()))
        row.addWidget(add)
        row.addWidget(rm)
        row.addStretch(1)
        v.addLayout(row)
        return w

    def _show_level_tab(self, mode: str):
        if self._level_tab is not None:
            self.tabs.removeTab(2)
        page, title = (self.z_page, "Z Grid") if mode == "xyz" else (self.story_page, "Story")
        self.tabs.addTab(page, title)
        self._level_tab = page

    def _mode_changed(self, mode: str):
        # carry the levels across so switching tabs inside the dialog loses nothing
        try:
            levels = self._levels_from(self.story_table if mode == "xyz" else self.z_table,
                                       "story" if mode == "xyz" else "xyz")
        except ValueError:
            levels = None
        if levels:
            target = self.z_table if mode == "xyz" else self.story_table
            target.setRowCount(0)
            if mode == "xyz":
                for name, z in levels:
                    self._add_row(target, (name, z))
            else:
                prev = None
                for name, z in levels:
                    self._add_row(target, (name, z if prev is None else z - prev))
                    prev = z
        self._show_level_tab(mode)
        self.tabs.setCurrentIndex(2)

    # ------------------------------------------------------------- values
    @staticmethod
    def _rows(table):
        out = []
        for r in range(table.rowCount()):
            a, b = table.item(r, 0), table.item(r, 1)
            name = a.text().strip() if a else ""
            if not name:
                continue
            out.append((name, float(b.text()) if b and b.text().strip() else 0.0))
        return out

    def _levels_from(self, table, mode) -> list:
        rows = self._rows(table)
        if mode == "xyz":
            return sorted(rows, key=lambda t: t[1])
        levels, z = [], 0.0
        for k, (name, h) in enumerate(rows):
            z = h if k == 0 else z + h
            levels.append((name, z))
        return levels

    def accept(self):
        mode = self.switch.mode()
        try:
            xs, ys = self._rows(self.x_table), self._rows(self.y_table)
            levels = self._levels_from(self.z_table if mode == "xyz" else self.story_table, mode)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
            return
        if not levels:
            QtWidgets.QMessageBox.warning(self, "Levels", "Define at least one level")
            return
        names = [n for n, _ in levels]
        if len(set(names)) != len(names):
            QtWidgets.QMessageBox.warning(self, "Levels", "Level names must be unique")
            return

        def mutate(model: Model):
            system = next(iter(model.grid_systems), "G1")
            if system not in model.grid_systems:
                from ..model.objects import GridSystem
                model.grid_systems[system] = GridSystem(name=system)
            model.grid_lines = ([GridLine(system=system, name=n, direction="X", ordinate=o) for n, o in xs]
                                + [GridLine(system=system, name=n, direction="Y", ordinate=o) for n, o in ys])
            if mode == "xyz":
                model.options.grid_mode = "xyz"
                model.set_z_grid(levels)
            else:
                model.options.grid_mode = "story"
                model._stories_from_levels(levels)

        self.doc.edit(mutate)
        super().accept()


class RestraintDialog(QtWidgets.QDialog):
    def __init__(self, current=FREE6, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Joint Restraints")
        lay = QtWidgets.QVBoxLayout(self)
        self.boxes = []
        grid = QtWidgets.QGridLayout()
        for k, lab in enumerate(("UX", "UY", "UZ", "RX", "RY", "RZ")):
            cb = QtWidgets.QCheckBox(lab)
            cb.setChecked(bool(current[k]))
            self.boxes.append(cb)
            grid.addWidget(cb, k // 3, k % 3)
        lay.addLayout(grid)
        quick = QtWidgets.QHBoxLayout()
        for text, val in (("Fixed", FIXED), ("Pinned", PINNED), ("Free", FREE6)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, v=val: [cb.setChecked(x) for cb, x in zip(self.boxes, v)])
            quick.addWidget(b)
        lay.addLayout(quick)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def value(self):
        return tuple(cb.isChecked() for cb in self.boxes)


class LoadDialog(QtWidgets.QDialog):
    """Point / frame / area load assignment."""

    def __init__(self, model: Model, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setWindowTitle({"point": "Assign Joint Loads", "frame": "Assign Frame Distributed Loads",
                             "area": "Assign Uniform Area Loads"}[kind])
        f = QtWidgets.QFormLayout(self)
        self.pattern = QtWidgets.QComboBox()
        self.pattern.addItems(list(model.load_patterns))
        f.addRow("Load pattern", self.pattern)
        self.fields: Dict[str, QtWidgets.QLineEdit] = {}
        if kind == "point":
            for lab in ("FX", "FY", "FZ", "MX", "MY", "MZ"):
                self.fields[lab] = QtWidgets.QLineEdit("0")
                f.addRow(lab, self.fields[lab])
        else:
            self.fields["value"] = QtWidgets.QLineEdit("0")
            f.addRow("Value (per length)" if kind == "frame" else "Value (per area)", self.fields["value"])
            self.direction = QtWidgets.QComboBox()
            self.direction.addItems(["Gravity", "X", "Y", "Z"] + (["Local 2", "Local 3"] if kind == "frame" else []))
            f.addRow("Direction", self.direction)
            if kind == "frame":
                self.fields["value_b"] = QtWidgets.QLineEdit("")
                f.addRow("Value at end (trapezoid, optional)", self.fields["value_b"])
            if kind == "area":  # an area load is a load panel made from the area
                self.distribution = QtWidgets.QComboBox()
                self.distribution.addItem("On the area (shells)", "shells")
                self.distribution.addItem("One-way to the frames around it", "one_way")
                self.distribution.addItem("Two-way to the frames around it", "two_way")
                f.addRow("Distribution", self.distribution)
                self.span = QtWidgets.QComboBox()
                self.span.addItems(["X", "Y"])
                f.addRow("One-way span direction", self.span)
        self.replace = QtWidgets.QCheckBox("Replace existing loads in this pattern")
        self.replace.setChecked(True)
        f.addRow(self.replace)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def apply(self, doc: Document, frames=(), areas=(), points=()):
        pat = self.pattern.currentText()
        if not pat:
            raise ModelError("NO_PATTERN", "Define a load pattern first")
        vals = {k: (float(e.text()) if e.text().strip() else None) for k, e in self.fields.items()}
        replace = self.replace.isChecked()
        kind = self.kind
        direction = getattr(self, "direction", None)
        direction = direction.currentText() if direction else "Gravity"

        def mutate(model: Model):
            if kind == "point":
                for n in points:
                    p = model.points[n]
                    if replace:
                        p.loads = [ld for ld in p.loads if ld.pattern != pat]
                    model.add_point_load(n, pat, (vals["FX"] or 0, vals["FY"] or 0, vals["FZ"] or 0),
                                         (vals["MX"] or 0, vals["MY"] or 0, vals["MZ"] or 0))
            elif kind == "frame":
                for n in frames:
                    fr = model.frames[n]
                    if replace:
                        fr.loads = [ld for ld in fr.loads if ld.pattern != pat]
                    model.add_frame_distributed_load(n, pat, vals["value"] or 0.0, vals.get("value_b"),
                                                     direction=direction)
            else:
                from ..model.objects import PanelLoadCase
                dist = self.distribution.currentData()
                span = (1.0, 0.0, 0.0) if self.span.currentText() == "X" else (0.0, 1.0, 0.0)
                for n in areas:
                    a = model.areas[n]
                    if replace:  # this pattern leaves the area's panels
                        a.loads = [ld for ld in a.loads if ld.pattern != pat]
                        for p in [p for p in model.panel_loads.values() if p.area == n]:
                            p.loads = [c for c in p.loads if c.pattern != pat]
                            if not p.loads:
                                del model.panel_loads[p.name]
                    if dist == "shells":
                        model.add_area_load(n, pat, vals["value"] or 0.0, direction=direction)
                    else:
                        model.add_area_panel(n, [PanelLoadCase(pat, vals["value"] or 0.0, direction)],
                                             distribution=dist, span_dir=span if dist == "one_way" else None)

        doc.edit(mutate)


class ChoiceDialog(QtWidgets.QDialog):
    def __init__(self, title: str, label: str, options: Sequence[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        f = QtWidgets.QFormLayout(self)
        self.box = QtWidgets.QComboBox()
        self.box.addItems(list(options))
        f.addRow(label, self.box)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def value(self) -> str:
        return self.box.currentText()


class VectorDialog(QtWidgets.QDialog):
    """A few numeric fields (dX, dY, dZ ...)."""

    def __init__(self, title: str, labels: Sequence[str], defaults: Optional[Sequence[float]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        f = QtWidgets.QFormLayout(self)
        self.edits = []
        defaults = defaults or [0.0] * len(labels)
        for lab, val in zip(labels, defaults):
            e = QtWidgets.QLineEdit(f"{val:g}")
            self.edits.append(e)
            f.addRow(lab, e)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def values(self) -> List[float]:
        return [float(e.text() or 0.0) for e in self.edits]


class MultiChoiceDialog(QtWidgets.QDialog):
    def __init__(self, title: str, options: Sequence[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list.addItems(list(options))
        lay.addWidget(self.list)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def values(self) -> List[str]:
        return [i.text() for i in self.list.selectedItems()]
