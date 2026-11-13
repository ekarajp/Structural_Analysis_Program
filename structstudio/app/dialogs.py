"""Define / Assign dialogs for the desktop app."""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

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

    def __init__(self, doc: Document, parent=None, system: Optional[str] = None, add: Optional[str] = None,
                 copy_of: Optional[str] = None, show_levels: bool = False, tower: Optional[str] = None,
                 line: Optional[str] = None):
        """``system``: the grid system shown first; ``add``: start with a new system of
        that type ("Cartesian" / "Cylindrical"); ``copy_of``: start with a copy of that
        system; ``show_levels``: open on the Story / Z Grid tab; ``tower`` (several
        towers): the tower whose grid systems and stories are edited (the others stay)."""
        super().__init__(parent)
        self.doc = doc
        m = doc.model
        self.tower = m._tower(tower if tower is not None else (system and m.tower_of_system(system))) \
            if m.multi_tower else None
        self.setWindowTitle("Grid Systems and Levels" if self.tower is None
                            else f"Tower {self.tower} - Grid Systems and Stories")
        self.resize(620, 520)
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Levels"))
        self.switch = GridModeSwitch()
        self.switch.set_mode(m.grid_mode)
        self.switch.mode_changed.connect(self._mode_changed)
        if self.tower is not None:
            self.switch.setToolTip("Stories or Z grid lines - for every tower (each keeps its own levels)")
        top.addWidget(self.switch)
        top.addStretch(1)
        lay.addLayout(top)
        # grid systems (ETABS: several Cartesian and cylindrical systems): the X / Y tabs
        # show the lines of the system chosen here
        self._systems = self._systems_from(m, self.tower)
        self._extras = {(g.system, g.direction, g.name): g for g in m.grid_lines}
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Grid system"))
        self.system_box = QtWidgets.QComboBox()
        self.system_box.setMinimumWidth(110)
        srow.addWidget(self.system_box)
        self.type_label = QtWidgets.QLabel()
        srow.addWidget(self.type_label)
        for text, fn in (("Add Cartesian", lambda: self._add_system("Cartesian")),
                         ("Add Cylindrical", lambda: self._add_system("Cylindrical")),
                         ("Rename...", self._rename_system), ("Delete", self._delete_system)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, fn=fn: fn())
            srow.addWidget(b)
        srow.addStretch(1)
        lay.addLayout(srow)
        orow = QtWidgets.QHBoxLayout()
        u = m.units.label("length") if hasattr(m.units, "label") else ""
        self.ox, self.oy, self.rot = QtWidgets.QLineEdit(), QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for lab, w in ((f"Origin X [{u}]", self.ox), (f"Origin Y [{u}]", self.oy), ("Rotation [deg]", self.rot)):
            orow.addWidget(QtWidgets.QLabel(lab))
            w.setMaximumWidth(90)
            orow.addWidget(w)
        orow.addStretch(1)
        lay.addLayout(orow)
        # one page: the X and Y lines and the levels side by side, the General lines under
        # them, the plan of the system (with its spacings) on the right - live
        self.x_table = self._table(["Name", "X", "Spacing"], [])
        self.y_table = self._table(["Name", "Y", "Spacing"], [])
        self.g_table = self._table(["Name", "X1", "Y1", "X2", "Y2"], [])
        for t in (self.x_table, self.y_table):
            t.horizontalHeaderItem(2).setToolTip("Distance to the next line: changing it moves every line after it")
            t.itemChanged.connect(lambda it, t=t: self._lines_edited(t, it))
        self.g_table.itemChanged.connect(lambda _it: self._refresh_preview())
        self._current = None
        for name in self._systems:
            self.system_box.addItem(name)
        stories = m.stories_of(self.tower) if self.tower is not None else m.stories_bottom_up()
        # Story table: name, height (bottom-up; the first row is the base level with its elevation)
        rows = [(s.name, s.elevation if k == 0 else s.elevation - stories[k - 1].elevation)
                for k, s in enumerate(stories)]
        self.story_table = self._table(["Story", "Height"], rows)
        self.story_table.horizontalHeaderItem(0).setToolTip("Bottom up: the first row is the base")
        self.story_table.horizontalHeaderItem(1).setToolTip("Story height; the first row (base): its elevation")
        zl = [(g.name, g.ordinate) for g in m.z_grid_lines(self.tower)] if m.grid_mode == "xyz" else []
        self.z_table = self._table(["Name", "Z"], zl or [(s.name, s.elevation) for s in stories])
        self.box_a = self._box("X Grid", self.x_table, ("", 0.0))
        self.box_b = self._box("Y Grid", self.y_table, ("", 0.0))
        self.story_page = self._with_buttons(self.story_table, ("", 3.0))
        self.z_page = self._with_buttons(self.z_table, ("", 0.0))
        self.levels = QtWidgets.QStackedWidget()
        self.levels.addWidget(self.story_page)
        self.levels.addWidget(self.z_page)
        self.box_levels = QtWidgets.QGroupBox("Story")
        QtWidgets.QVBoxLayout(self.box_levels).addWidget(self.levels)
        self.box_g = self._box("General lines (any angle: two points in the system's axes)", self.g_table,
                               ("", 0.0, 0.0, 1.0, 0.0))
        self.g_page = self.box_g
        self.box_g.setMaximumHeight(150)
        for t in (self.x_table, self.y_table, self.g_table, self.story_table, self.z_table):
            hh = t.horizontalHeader()
            hh.setStretchLastSection(False)
            hh.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            hh.setMinimumSectionSize(36)
            t.verticalHeader().setDefaultSectionSize(22)
            t.verticalHeader().setMinimumWidth(18)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        cols = QtWidgets.QHBoxLayout()
        for b in (self.box_a, self.box_b, self.box_levels):
            cols.addWidget(b, 1)
        lv.addLayout(cols, 1)
        lv.addWidget(self.box_g)
        from .grid_manager import GridPreview
        self.preview = GridPreview(self)
        self.preview.setMinimumSize(300, 300)
        pbox = QtWidgets.QGroupBox("Plan")
        pv = QtWidgets.QVBoxLayout(pbox)
        pv.addWidget(self.preview, 1)
        self.dims_box = QtWidgets.QCheckBox("Dimensions")
        self.dims_box.setChecked(True)
        self.dims_box.toggled.connect(self._refresh_preview)
        pv.addWidget(self.dims_box)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(pbox)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)
        self.resize(1180, 580)
        for w in (self.ox, self.oy, self.rot):
            w.editingFinished.connect(self._refresh_preview)
        self._show_level_tab(m.grid_mode)
        self.system_box.currentTextChanged.connect(self._switch_system)
        self._switch_system(self.system_box.currentText())
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if system and system in self._systems:
            self.system_box.setCurrentText(system)
        if copy_of and copy_of in self._systems:
            self._copy_system(copy_of)
        elif add in ("Cartesian", "Cylindrical"):
            self._add_system(add)
        if show_levels:
            self.focus_levels()
        if line and system and system in self._systems:
            self.select_line(line)

    def _box(self, title: str, table, blank) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        QtWidgets.QVBoxLayout(box).addWidget(self._with_buttons(table, blank))
        return box

    def level_table(self) -> QtWidgets.QTableWidget:
        """The levels shown: the story table (X-Y-Story) or the Z lines (X-Y-Z)."""
        return self.z_table if self.levels.currentWidget() is self.z_page else self.story_table

    def focus_levels(self):
        self.on_levels = True  # (opened on the levels: Stories box / Modify Stories)
        t = self.level_table()
        t.setFocus()
        if t.rowCount():
            t.setCurrentCell(0, 0)

    def select_line(self, name: str) -> bool:
        """Show the grid line ``name`` of the current system: its row selected."""
        for t in (self.x_table, self.y_table, self.g_table):
            for r in range(t.rowCount()):
                it = t.item(r, 0)
                if it is not None and it.text().strip() == name:
                    t.setFocus()
                    t.setCurrentCell(r, 1)
                    t.selectRow(r)
                    t.scrollToItem(it)
                    return True
        return False

    # ------------------------------------------------- spacings, the live plan
    def _lines_edited(self, table, item):
        """A changed spacing moves every line after it (ETABS spacing input); a changed
        ordinate updates the spacings.  The plan follows."""
        if getattr(self, "_filling", False):
            return
        if item is not None and item.column() == 2:
            r = item.row()
            try:
                new = float(item.text())
                o0 = float(table.item(r, 1).text())
                o1 = float(table.item(r + 1, 1).text()) if r + 1 < table.rowCount() else None
            except (ValueError, AttributeError):
                new, o1 = None, None
            if new is not None and o1 is not None:
                delta = (o0 + new) - o1
                self._filling = True
                try:
                    for k in range(r + 1, table.rowCount()):
                        it = table.item(k, 1)
                        try:
                            it.setText(f"{float(it.text()) + delta:g}")
                        except (ValueError, AttributeError):
                            pass
                finally:
                    self._filling = False
        self._spacings(table)
        self._refresh_preview()

    def _spacings(self, table):
        self._filling = True
        try:
            for r in range(table.rowCount()):
                try:
                    a = float(table.item(r, 1).text())
                    b = float(table.item(r + 1, 1).text()) if r + 1 < table.rowCount() else None
                    text = "" if b is None else f"{b - a:g}"
                except (ValueError, AttributeError):
                    text = ""
                it = table.item(r, 2)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    table.setItem(r, 2, it)
                it.setText(text)
                it.setForeground(QtGui.QBrush(QtGui.QColor(110, 110, 110)))
        finally:
            self._filling = False

    def _refresh_preview(self, *_):
        """The plan of the system as the tables have it now (the others faintly)."""
        prev = getattr(self, "preview", None)
        if prev is None or self._current is None:
            return
        from types import SimpleNamespace
        from ..model.objects import GridSystem
        try:
            self._save_system()
        except ValueError:
            return
        systems, lines = {}, []
        for sname, d in self._systems.items():
            gs = GridSystem(name=sname, type=d["type"], origin=tuple(d["origin"]), rotation=float(d["rotation"]))
            systems[sname] = gs
            da, db = ("T", "R") if d["type"] == "Cylindrical" else ("X", "Y")
            for dirn, rows in ((da, d["a"]), (db, d["b"])):
                for n, o in rows:
                    if dirn != "R" or o > 0:
                        lines.append(GridLine(system=sname, name=n, direction=dirn, ordinate=o))
            for n, x1, y1, x2, y2 in (d.get("g", []) if d["type"] == "Cartesian" else []):
                lines.append(GridLine(system=sname, name=n, direction="General", start=(x1, y1), end=(x2, y2)))
        model = SimpleNamespace(grid_systems=systems, grid_lines=lines, units=self.doc.model.units)
        prev.dims = self.dims_box.isChecked()
        prev.set_system(model, self._current)

    # ------------------------------------------------------------ grid systems
    @staticmethod
    def _systems_from(m, tower: Optional[str] = None) -> dict:
        """name -> {type, origin, rotation, a: rows of X / radial lines, b: rows of Y / circles}
        (several towers: the systems of ``tower``)."""
        out = {}
        mine = set(m.grid_systems_of(tower)) if tower is not None else None
        for name, gs in m.grid_systems.items():
            if mine is not None and name not in mine:
                continue
            out[name] = {"type": gs.type if gs.type in ("Cartesian", "Cylindrical") else "Cartesian",
                         "origin": tuple(gs.origin or (0.0, 0.0)), "rotation": float(gs.rotation or 0.0),
                         "a": [], "b": []}
        for g in m.grid_lines:
            if g.direction not in ("X", "Y", "R", "T"):
                continue
            if mine is not None and g.system not in mine:
                continue
            sysd = out.setdefault(g.system or "G1", {"type": "Cylindrical" if g.direction in ("R", "T") else "Cartesian",
                                                     "origin": (0.0, 0.0), "rotation": 0.0, "a": [], "b": []})
            # Cartesian: a = X lines, b = Y lines; cylindrical: a = radial lines (T, angle),
            # b = circles (R, radius) - the ETABS directions
            sysd["a" if g.direction in ("X", "T") else "b"].append((g.name, g.ordinate))
        for g in m.grid_lines:  # General lines (two points, in the system's axes)
            if g.direction != "General" or not g.start or not g.end:
                continue
            if mine is not None and g.system not in mine:
                continue
            sysd = out.get(g.system or "G1")
            if sysd is not None:
                sysd.setdefault("g", []).append((g.name, float(g.start[0]), float(g.start[1]),
                                                 float(g.end[0]), float(g.end[1])))
        if not out:
            name = "G1" if tower is None else f"{tower}-G1"
            k = 2
            while name in m.grid_systems:
                name, k = f"{tower}-G{k}", k + 1
            out[name] = {"type": "Cartesian", "origin": (0.0, 0.0), "rotation": 0.0, "a": [], "b": []}
        return out

    def _tab_titles(self, kind: str):
        return ("Radial lines", "Circles") if kind == "Cylindrical" else ("X Grid", "Y Grid")

    def _save_system(self):
        name = self._current
        if name is None or name not in self._systems:
            return
        d = self._systems[name]
        d["a"], d["b"] = self._rows(self.x_table), self._rows(self.y_table)
        if d["type"] == "Cartesian":
            d["g"] = self._grows(self.g_table)
        d["origin"] = (float(self.ox.text() or 0), float(self.oy.text() or 0))
        d["rotation"] = float(self.rot.text() or 0)

    def _switch_system(self, name: str):
        if not name or name not in self._systems:
            return
        if self._current is not None and self._current != name:
            try:
                self._save_system()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
                self.system_box.blockSignals(True)
                self.system_box.setCurrentText(self._current)
                self.system_box.blockSignals(False)
                return
        d = self._systems[name]
        self._current = name
        kind = d["type"]
        ta, tb = self._tab_titles(kind)
        self.box_a.setTitle(ta)
        self.box_b.setTitle(tb)
        self.x_table.setHorizontalHeaderLabels(["Name", "Angle [deg]" if kind == "Cylindrical" else "X", "Spacing"])
        self.y_table.setHorizontalHeaderLabels(["Name", "Radius" if kind == "Cylindrical" else "Y", "Spacing"])
        self._filling = True
        try:
            for t, rows in ((self.x_table, d["a"]), (self.y_table, d["b"]), (self.g_table, d.get("g", []))):
                t.setRowCount(0)
                for row in rows:
                    self._add_row(t, row)
        finally:
            self._filling = False
        self._spacings(self.x_table)
        self._spacings(self.y_table)
        self.box_g.setEnabled(kind == "Cartesian")
        self._refresh_preview()
        self.type_label.setText(kind)
        self.ox.setText(f"{d['origin'][0]:g}")
        self.oy.setText(f"{d['origin'][1]:g}")
        self.rot.setText(f"{d['rotation']:g}")

    def _new_name(self, base: str) -> str:
        k = 1
        taken = set(self._systems) | set(self.doc.model.grid_systems)  # (the other towers' too)
        while f"{base}{k}" in taken:
            k += 1
        return f"{base}{k}"

    def _add_system(self, kind: str):
        try:
            self._save_system()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
            return
        name = self._new_name("G" if kind == "Cartesian" else "C")
        if kind == "Cylindrical":
            rows_a = [(f"{name}-R{k + 1}", float(a)) for k, a in enumerate(range(0, 360, 45))]
            rows_b = [(f"{name}-T{k + 1}", float(r)) for k, r in enumerate((5.0, 10.0))]
        else:
            rows_a = [(f"{name}-{k + 1}", float(x)) for k, x in enumerate((0, 6, 12))]
            rows_b = [(f"{name}-{chr(65 + k)}", float(y)) for k, y in enumerate((0, 6))]
        self._systems[name] = {"type": kind, "origin": (0.0, 0.0), "rotation": 0.0, "a": rows_a, "b": rows_b}
        self.system_box.addItem(name)
        self.system_box.setCurrentText(name)

    def _copy_system(self, src: str):
        """A new system with the lines, origin and rotation of ``src`` (named src_copy)."""
        try:
            self._save_system()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
            return
        import copy as _copy
        name, k = f"{src}_copy", 2
        while name in self._systems or name in self.doc.model.grid_systems:
            name, k = f"{src}_copy{k}", k + 1
        self._systems[name] = _copy.deepcopy(self._systems[src])
        import dataclasses as _dc
        for (s_, d_, n_), g in list(self._extras.items()):
            if s_ == src:  # bubble / visibility settings and general lines come along
                self._extras[(name, d_, n_)] = _dc.replace(g, system=name)
        self.system_box.addItem(name)
        self.system_box.setCurrentText(name)

    def _rename_system(self):
        old = self._current
        if old is None:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename grid system", "Name", text=old)
        new = (new or "").strip()
        if not ok or not new or new == old:
            return
        if new in self._systems or (new in self.doc.model.grid_systems and new not in self._systems):
            QtWidgets.QMessageBox.warning(self, "Grid systems", f"A grid system {new!r} exists already")
            return
        self._save_system()
        self._systems = {(new if k == old else k): v for k, v in self._systems.items()}
        self._extras = {((new if s == old else s), d, n): g for (s, d, n), g in self._extras.items()}
        i = self.system_box.findText(old)
        self._current = new
        self.system_box.blockSignals(True)
        self.system_box.setItemText(i, new)
        self.system_box.blockSignals(False)

    def _delete_system(self):
        if len(self._systems) <= 1 or self._current is None:
            QtWidgets.QMessageBox.information(self, "Grid systems", "The model keeps at least one grid system")
            return
        name = self._current
        del self._systems[name]
        self._current = None
        i = self.system_box.findText(name)
        self.system_box.removeItem(i)
        self._switch_system(self.system_box.currentText())

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
        page, title = (self.z_page, "Z Grid") if mode == "xyz" else (self.story_page, "Story")
        self.levels.setCurrentWidget(page)
        self.box_levels.setTitle(title)

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
        self.focus_levels()

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

    @staticmethod
    def _grows(table):
        """General lines: (name, x1, y1, x2, y2) rows; a line needs two different points."""
        out = []
        for r in range(table.rowCount()):
            it = table.item(r, 0)
            name = it.text().strip() if it else ""
            if not name:
                continue
            v = []
            for c in range(1, 5):
                cell = table.item(r, c)
                v.append(float(cell.text()) if cell and cell.text().strip() else 0.0)
            if abs(v[0] - v[2]) + abs(v[1] - v[3]) <= 0:
                raise ValueError(f"General line {name}: its two points are the same")
            out.append((name, *v))
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
            self._save_system()
            systems = {k: dict(v) for k, v in self._systems.items()}
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

        for sname, d in systems.items():
            if d["type"] == "Cylindrical" and any(r <= 0 for _, r in d["b"]):
                QtWidgets.QMessageBox.warning(self, "Grid systems", f"{sname}: circle radii must be greater than zero")
                return
        extras = self._extras
        tower = self.tower
        if tower is not None:
            m0 = self.doc.model
            mine = {st.name for st in m0.stories_of(tower)}
            clash = [n for n in names if n in m0.stories and n not in mine]
            if clash:
                QtWidgets.QMessageBox.warning(self, "Levels", f"Story name {clash[0]!r} is used by tower "
                                              f"{m0.tower_of_story(clash[0])} - story names are unique in the model")
                return
            clash = [n for n in systems if n in m0.grid_systems and m0.tower_of_system(n) != tower]
            if clash:
                QtWidgets.QMessageBox.warning(self, "Grid systems", f"Grid system {clash[0]!r} belongs to tower "
                                              f"{m0.tower_of_system(clash[0])}")
                return

        glue = bool(getattr(self.doc, "glue_grids", False))
        self.glued = 0

        def mutate(model: Model):
            import dataclasses as _dc
            from ..model.objects import GridSystem
            from .grid_tools import glue_moves
            old = model.grid_systems
            # Move Objects with Grid Lines: the joints on an X / Y line whose ordinate changed
            # (its system not moved or turned) move with it - found before anything changes
            moves = []
            for sname, d in (systems.items() if glue else ()):
                gs0 = old.get(sname)
                if gs0 is None or d["type"] != "Cartesian" or gs0.type == "Cylindrical" or \
                        tuple(gs0.origin or (0.0, 0.0)) != tuple(d["origin"]) or \
                        float(gs0.rotation or 0.0) != float(d["rotation"]):
                    continue
                for dirn, rows in (("X", d["a"]), ("Y", d["b"])):
                    for n, o in rows:
                        prev = extras.get((sname, dirn, n))
                        if prev is not None and prev.ordinate != o:
                            moves.append((sname, dirn, prev.ordinate, o))
            shifted = glue_moves(model, moves) if moves else {}
            # several towers: the other towers' systems and lines stay as they are
            edited = set(model.grid_systems_of(tower)) if tower is not None else set(old)
            model.grid_systems = {n: gs for n, gs in old.items() if n not in edited}
            # (the other towers' lines - their Z lines too - stay; this tower's Z lines are
            # made again from its levels below)
            lines = [g for g in model.grid_lines if g.system not in edited] if tower is not None else []
            for sname, d in systems.items():
                gs0 = old.get(sname)
                gs = _dc.replace(gs0, name=sname) if gs0 is not None else GridSystem(name=sname)
                gs.type, gs.origin, gs.rotation = d["type"], tuple(d["origin"]), float(d["rotation"])
                gs.tower = tower if tower is not None else gs.tower
                model.grid_systems[sname] = gs
                da, db = ("T", "R") if d["type"] == "Cylindrical" else ("X", "Y")
                for dirn, rows in ((da, d["a"]), (db, d["b"])):
                    for n, o in rows:
                        prev = extras.get((sname, dirn, n))
                        gl = _dc.replace(prev, ordinate=o) if prev is not None else \
                            GridLine(system=sname, name=n, direction=dirn, ordinate=o)
                        lines.append(gl)
                # General lines (the General tab)
                for n, x1, y1, x2, y2 in (d.get("g", []) if d["type"] == "Cartesian" else []):
                    prev = extras.get((sname, "General", n))
                    lines.append(_dc.replace(prev, start=(x1, y1), end=(x2, y2)) if prev is not None else
                                 GridLine(system=sname, name=n, direction="General", start=(x1, y1),
                                          end=(x2, y2)))
            model.grid_lines = lines
            for pn, (dx, dy) in shifted.items():
                model.points[pn].x += dx
                model.points[pn].y += dy
            self.glued = len(shifted)
            if tower is not None:
                if mode == "xyz" and model.grid_mode == "xyz":
                    model.set_z_grid(levels, tower)
                elif mode == "xyz":  # every tower to Z grids: this tower's levels first
                    model.set_tower_stories(tower, levels)
                    model.set_grid_mode("xyz")
                else:
                    model.set_grid_mode("story")  # (the other towers' Z lines -> their stories)
                    model.set_tower_stories(tower, levels)
            elif mode == "xyz":
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
