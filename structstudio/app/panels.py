"""Dock panels: model tree, property editor, results tables, log."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ..model import Model
from .document import Document

_OBJECT_LIMIT = 3000  # items listed per group (very large models stay responsive)

_FRAME_GROUPS = (("Columns", "Column"), ("Beams", "Beam"), ("Braces", "Brace"), ("Null frames", "Null"), ("Rigid links", "Rigid"))
_SHELL_GROUPS = (("Floors", "Floor"), ("Walls", "Wall"), ("Ramps", "Ramp"), ("Openings", "Opening"))


class ModelTree(QtWidgets.QTreeWidget):
    """ETABS-style Model Explorer: Structure Layout (stories, grids), Structural
    Objects (joints, frames by type, shells by type, load panels - each by story),
    Groups, Properties and Loads.

    Click an object: it is selected and located (status bar: story, grid, size;
    the plan windows go to its story, it flashes).  Double-click: zoom to it.
    Right-click an object: Show information and properties / Edit section /
    Show only this object (in the active window).
    Click a group / story heading of objects: select all of them.  Click a story
    under Structure Layout: the active plan window shows it.  Type in the box
    above the tree to find objects by name or section."""

    item_selected = QtCore.Signal(str, str)  # collection attr, object name (any model item)
    object_activated = QtCore.Signal(str, str)  # double-click on an object: zoom to it
    set_clicked = QtCore.Signal(object)  # {"frames": [...], "areas": [...], "points": [...]}
    story_clicked = QtCore.Signal(str)
    object_action = QtCore.Signal(str, str, str)  # right-click menu: action, collection attr, name

    #: right-click > what (menu text, action key); the same choices as right-click on the drawing
    #: (clicking the object in the tree already selects it, so there is no Select here)
    OBJECT_ACTIONS = (("Show Information and Properties...", "info"),
                      ("Assign section...", "assign_section"),  # (used more often than Edit section)
                      ("Edit section...", "section"),
                      ("Show only this object (active window)", "only"))

    def __init__(self, document: Document, parent=None):
        super().__init__(parent)
        self.doc = document
        self.setHeaderHidden(True)
        self.setUniformRowHeights(True)
        self.itemClicked.connect(self._clicked)
        self.itemDoubleClicked.connect(self._double_clicked)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self.doc.on_change(lambda what: self.refresh() if what == "model" else None)
        self._filter = ""
        self.refresh()

    # ------------------------------------------------------------ building
    @staticmethod
    def _key(item) -> str:
        parts = []
        while item is not None:
            parts.append(item.data(0, QtCore.Qt.UserRole + 1) or item.text(0))
            item = item.parent()
        return "/".join(reversed(parts))

    def _expanded_keys(self) -> set:
        out = set()
        it = QtWidgets.QTreeWidgetItemIterator(self)
        while it.value():
            if it.value().isExpanded():
                out.add(self._key(it.value()))
            it += 1
        return out

    def _node(self, parent, title: str, count: Optional[int] = None, data=None, key: Optional[str] = None,
              tip: Optional[str] = None):
        item = QtWidgets.QTreeWidgetItem([title if count is None else f"{title} ({count})"])
        item.setData(0, QtCore.Qt.UserRole, data)
        item.setData(0, QtCore.Qt.UserRole + 1, key or title)
        if tip:
            item.setToolTip(0, tip)
        if parent is None:
            self.addTopLevelItem(item)
        else:
            parent.addChild(item)
        return item

    def _story_order(self) -> List[str]:
        m = self.doc.model
        return [s.name for s in reversed(m.stories_bottom_up())]

    def _by_story(self, parent, attr: str, names: List[str], text) -> None:
        """Objects under per-story sub-headings (top story first), like ETABS."""
        m = self.doc.model
        table = getattr(m, attr)
        groups: Dict[str, List[str]] = {}
        for n in names:
            groups.setdefault(getattr(table[n], "story", None) or "(no story)", []).append(n)
        order = [s for s in self._story_order() if s in groups] + [s for s in groups if s not in m.stories]
        for st in order:
            members = groups[st]
            sn = self._node(parent, st, len(members), data=("set", {attr: list(members)}), key=st)
            for n in members[:_OBJECT_LIMIT]:
                self._node(sn, text(n), data=(attr, n), key=n)
            if len(members) > _OBJECT_LIMIT:
                self._node(sn, f"... {len(members) - _OBJECT_LIMIT} more (use Select / tables)")

    def refresh(self):
        expanded = self._expanded_keys() if self.topLevelItemCount() else None
        self.setUpdatesEnabled(False)
        self.clear()
        m = self.doc.model
        root = self._node(None, f"{m.info.name}  [{m.units.force}-{m.units.length}]", data=("model", ""), key="model")
        # --- Structure Layout
        lay = self._node(None, "Structure Layout", key="layout")
        u = m.units.shown("length")
        title = "Levels (Z grid)" if m.grid_mode == "xyz" else "Stories"
        sto = self._node(lay, title, len(m.stories), data=("stories", ""), key="stories")
        for name in self._story_order():
            s = m.stories[name]
            self._node(sto, f"{name}   (z = {s.elevation:g} {u})", data=("story", name), key=name,
                       tip="Click: show this story in the active plan window")
        grids = [g for g in m.grid_lines if g.direction in ("X", "Y")]
        gn = self._node(lay, "Grid Lines", len(grids), key="grids")
        for g in grids:
            self._node(gn, f"{g.name}   ({g.direction} = {g.ordinate:g} {u})", key=f"{g.direction}{g.name}")
        # --- Structural Objects
        obj = self._node(None, "Structural Objects", key="objects")
        pts = self._node(obj, "Joints", len(m.points), data=("set", {"points": list(m.points)}), key="joints")
        self._by_story(pts, "points", list(m.points), lambda n: n)
        fr = self._node(obj, "Frames", len(m.frames), data=("set", {"frames": list(m.frames)}), key="frames")
        for title, t in _FRAME_GROUPS:
            names = [n for n, f in m.frames.items() if f.type == t]
            if names:
                g = self._node(fr, title, len(names), data=("set", {"frames": names}), key=title)
                self._by_story(g, "frames", names, lambda n: f"{n}    {m.frames[n].section}")
        sh = self._node(obj, "Shells", len(m.areas), data=("set", {"areas": list(m.areas)}), key="shells")
        for title, t in _SHELL_GROUPS:
            names = [n for n, a in m.areas.items() if a.type == t]
            if names:
                g = self._node(sh, title, len(names), data=("set", {"areas": names}), key=title)
                self._by_story(g, "areas", names, lambda n: f"{n}    {m.areas[n].section or ''}".rstrip())
        if m.panel_loads:
            pl = self._node(obj, "Load panels", len(m.panel_loads), key="panels")
            self._by_story(pl, "panel_loads", list(m.panel_loads), lambda n: n)
        # --- Groups
        if m.groups:
            gr = self._node(None, "Groups", len(m.groups), key="groups")
            for g in m.groups.values():
                self._node(gr, g.name, len(g.points) + len(g.frames) + len(g.areas),
                           data=("set", {"points": list(g.points), "frames": list(g.frames), "areas": list(g.areas)}),
                           key=g.name)
        # --- Properties / Loads (definitions: click to edit in the property panel)
        for head, cols in (("Properties", (("Materials", "materials"), ("Frame Sections", "frame_sections"),
                                            ("Shell Sections", "shell_sections"), ("Diaphragms", "diaphragms"),
                                            ("Functions", "functions"))),
                           ("Loads", (("Load Patterns", "load_patterns"), ("Load Cases", "load_cases"),
                                      ("Load Combinations", "load_combinations")))):
            h = self._node(None, head, key=head)
            for title, attr in cols:
                table = getattr(m, attr)
                c = self._node(h, title, len(table), data=(attr, ""), key=attr)
                for name in list(table)[:_OBJECT_LIMIT]:
                    self._node(c, name, data=(attr, name), key=name)
        # expansion: as before, or the ETABS default (headings open, object lists closed)
        it = QtWidgets.QTreeWidgetItemIterator(self)
        while it.value():
            item = it.value()
            if expanded is not None:
                item.setExpanded(self._key(item) in expanded)
            else:
                k = self._key(item)
                item.setExpanded(item.parent() is None or k in ("objects/frames", "objects/shells", "layout/stories"))
            it += 1
        self.setUpdatesEnabled(True)
        if self._filter:
            self.set_filter(self._filter)

    # ------------------------------------------------------------- filter
    def set_filter(self, text: str):
        """Show only the objects whose text contains ``text`` (and their headings)."""
        self._filter = text.strip().lower()

        def walk(item) -> bool:
            own = not self._filter or self._filter in item.text(0).lower()
            kids = [walk(item.child(i)) for i in range(item.childCount())]
            show = own or any(kids)
            item.setHidden(not show)
            if self._filter and any(kids):
                item.setExpanded(True)
            return show
        for i in range(self.topLevelItemCount()):
            walk(self.topLevelItem(i))

    # ------------------------------------------------------------- clicks
    def _clicked(self, item, col):
        data = item.data(0, QtCore.Qt.UserRole)
        if not data:
            return
        kind, value = data
        if kind == "set":
            self.set_clicked.emit(value)
        elif kind == "story":
            self.story_clicked.emit(value)
        elif value:
            self.item_selected.emit(kind, value)

    def _double_clicked(self, item, col):
        data = item.data(0, QtCore.Qt.UserRole)
        if data and data[0] in ("points", "frames", "areas", "panel_loads") and data[1]:
            self.object_activated.emit(data[0], data[1])

    def object_actions(self, attr: str, name: str) -> List[tuple]:
        """The right-click choices for one object: (menu text, action key, enabled)."""
        m = self.doc.model
        has_section = bool(getattr(getattr(m, attr, {}).get(name), "section", None)) and attr in ("frames", "areas")
        out = []
        for text, key in self.OBJECT_ACTIONS:
            if key in ("section", "assign_section") and attr not in ("frames", "areas"):
                continue  # joints and load panels have no section
            if key == "only" and attr == "panel_loads":
                continue  # a load panel is not something a window can isolate
            out.append((text, key, has_section if key == "section" else True))
        return out

    def object_menu(self, attr: str, name: str):
        """The right-click menu of one object and its actions ({QAction: key})."""
        menu = QtWidgets.QMenu(self)
        what = {"points": "Joint", "frames": "Frame", "areas": "Shell", "panel_loads": "Load panel"}[attr]
        menu.addAction(f"{what} {name}").setEnabled(False)
        menu.addSeparator()
        actions = {}
        for text, key, enabled in self.object_actions(attr, name):
            a = menu.addAction(text)
            a.setEnabled(enabled)
            actions[a] = key
        return menu, actions

    def _context_menu(self, pos):
        item = self.itemAt(pos)
        data = item.data(0, QtCore.Qt.UserRole) if item is not None else None
        if not data or data[0] not in ("points", "frames", "areas", "panel_loads") or not data[1]:
            return
        attr, name = data
        menu, actions = self.object_menu(attr, name)
        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        if chosen in actions:
            self.object_action.emit(actions[chosen], attr, name)

    def find_item(self, attr: str, name: str):
        it = QtWidgets.QTreeWidgetItemIterator(self)
        while it.value():
            if it.value().data(0, QtCore.Qt.UserRole) == (attr, name):
                return it.value()
            it += 1
        return None


class ModelExplorer(QtWidgets.QWidget):
    """The Model Explorer dock: a find box above the :class:`ModelTree`."""

    def __init__(self, document: Document, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.find = QtWidgets.QLineEdit()
        self.find.setPlaceholderText("Find object (name or section)...")
        self.find.setClearButtonEnabled(True)
        lay.addWidget(self.find)
        self.tree = ModelTree(document)
        lay.addWidget(self.tree, 1)
        self.find.textChanged.connect(self.tree.set_filter)

    def __getattr__(self, name):  # behave like the tree (topLevelItem, signals, ...)
        if name == "tree":
            raise AttributeError(name)
        return getattr(self.tree, name)


class PropertyPanel(QtWidgets.QWidget):
    """Generic editor for one dataclass object of the model."""

    def __init__(self, document: Document, parent=None):
        super().__init__(parent)
        self.doc = document
        self.attr: Optional[str] = None
        self.name: Optional[str] = None
        layout = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel("Select an object")
        self.title.setStyleSheet("font-weight: bold")
        layout.addWidget(self.title)
        self.form_host = QtWidgets.QScrollArea()
        self.form_host.setWidgetResizable(True)
        layout.addWidget(self.form_host, 1)
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply)
        layout.addWidget(self.apply_btn)
        self._editors: Dict[str, QtWidgets.QWidget] = {}
        self._obj = None
        self.apply_btn.setEnabled(False)
        self.doc.on_change(self._on_doc_change)

    def _on_doc_change(self, what: str):
        if what != "model" or self.attr is None:
            return
        table = getattr(self.doc.model, self.attr, {})
        if self.name in table:
            self.show_object(self.attr, self.name)
        else:
            self.clear()

    def clear(self, text: str = "Select an object"):
        """Nothing selected: empty form, Apply disabled."""
        self.attr = self.name = None
        self._obj = None
        self._editors.clear()
        self.title.setText(text)
        self.form_host.setWidget(QtWidgets.QWidget())
        self.apply_btn.setEnabled(False)

    def show_object(self, attr: str, name: str):
        self.attr, self.name = attr, name
        table = getattr(self.doc.model, attr)
        obj = table.get(name)
        self._obj = obj
        self.apply_btn.setEnabled(obj is not None)
        self.title.setText(f"{attr}: {name}")
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        self._editors.clear()
        if obj is None:
            self.form_host.setWidget(w)
            return
        self._scales = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            k, unit = field_unit(obj, f)
            editor = self._make_editor(scaled(value, k))
            if editor is None:
                continue
            self._editors[f.name] = editor
            self._scales[f.name] = (k, editor.text() if isinstance(editor, QtWidgets.QLineEdit) else None)
            form.addRow(f"{f.name} [{unit}]" if unit else f.name, editor)
        self.form_host.setWidget(w)

    def _make_editor(self, value) -> Optional[QtWidgets.QWidget]:
        if isinstance(value, bool):
            cb = QtWidgets.QCheckBox()
            cb.setChecked(value)
            return cb
        if isinstance(value, (int, float)) or value is None:
            e = QtWidgets.QLineEdit("" if value is None else (f"{value:.12g}" if isinstance(value, float) else repr(value)))
            return e
        if isinstance(value, str):
            return QtWidgets.QLineEdit(value)
        if isinstance(value, (tuple, list)) and all(isinstance(v, (int, float, bool)) or v is None for v in value):
            return QtWidgets.QLineEdit(", ".join("" if v is None else str(v) for v in value))
        if isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
            return QtWidgets.QLineEdit(", ".join(f"{k}={v}" for k, v in value.items()))
        if dataclasses.is_dataclass(value):
            lbl = QtWidgets.QLabel(", ".join(f"{f.name}={getattr(value, f.name)}" for f in dataclasses.fields(value)))
            lbl.setWordWrap(True)
            return lbl
        if isinstance(value, list):
            lbl = QtWidgets.QLabel(f"{len(value)} item(s)")
            return lbl
        return None

    def apply(self):
        if self._obj is None:
            return
        obj = self._obj
        changes = {}
        for f in dataclasses.fields(obj):
            ed = self._editors.get(f.name)
            if ed is None or isinstance(ed, QtWidgets.QLabel):
                continue
            old = getattr(obj, f.name)
            k, shown = getattr(self, "_scales", {}).get(f.name, (1.0, None))
            if shown is not None and isinstance(ed, QtWidgets.QLineEdit) and ed.text() == shown:
                continue  # untouched (no round trip through the display unit)
            try:
                new = scaled(_parse(ed, scaled(old, k)), 1.0 / k)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid value", f"{f.name}: {exc}")
                return
            if new != old:
                changes[f.name] = new
        if not changes:
            return

        def mutate(model: Model):
            target = getattr(model, self.attr)[self.name]
            for k, v in changes.items():
                setattr(target, k, v)

        self.doc.edit(mutate)
        self.show_object(self.attr, self.name)


def field_unit(obj, f):
    """(factor model -> shown, unit label or None) of a tagged dataclass field in
    the display units of the open model (Options > Display Units)."""
    from ..display_units import item_of
    from .forms import display_units
    q = f.metadata.get("quantity")
    du = display_units()
    if not q or du is None:
        return 1.0, None
    item = item_of(q, type(obj).__name__, f.name)
    return du.factor(item, q), du.label(item, q)


def scaled(value, k: float):
    """A field value (number, numeric tuple / list / dict) times ``k``."""
    if k == 1.0 or value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value * k
    if isinstance(value, (tuple, list)) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                or v is None for v in value):
        return type(value)(None if v is None else v * k for v in value)
    if isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
        return {kk: v * k for kk, v in value.items()}
    return value


def _parse(editor, old):
    if isinstance(editor, QtWidgets.QCheckBox):
        return editor.isChecked()
    text = editor.text().strip()
    if isinstance(old, bool):
        return text.lower() in ("1", "true", "yes")
    if old is None or isinstance(old, (int, float)):
        if text == "":
            return None
        v = float(text)
        return int(v) if isinstance(old, int) and v == int(v) else v
    if isinstance(old, str):
        return text
    if isinstance(old, (tuple, list)):
        parts = [p.strip() for p in text.split(",")] if text else []
        if old and isinstance(old[0], bool):
            vals = [p.lower() in ("1", "true", "yes") for p in parts]
        else:
            vals = [None if p == "" else float(p) for p in parts]
        return type(old)(vals)
    if isinstance(old, dict):
        out = {}
        for part in text.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = float(v)
        return out
    raise ValueError("unsupported field")


class ResultsPanel(QtWidgets.QWidget):
    """Tables of results for the selected case / frame."""

    TABLES = ["Joint displacements", "Joint reactions", "Frame forces (selected frame)", "Shell forces",
              "Modal periods", "Modal mass ratios", "Story drifts", "Story shears", "Base reactions"]

    def __init__(self, document: Document, parent=None):
        super().__init__(parent)
        self.doc = document
        self.selected_frame: Optional[str] = None
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.case_box = QtWidgets.QComboBox()
        self.table_box = QtWidgets.QComboBox()
        self.table_box.addItems(self.TABLES)
        top.addWidget(QtWidgets.QLabel("Case"))
        top.addWidget(self.case_box, 1)
        top.addWidget(QtWidgets.QLabel("Table"))
        top.addWidget(self.table_box, 1)
        self.export_btn = QtWidgets.QPushButton("Export...")
        self.export_btn.clicked.connect(self.export_table)
        top.addWidget(self.export_btn)
        layout.addLayout(top)
        self._rows: List[dict] = []
        # a view on the rows (cell text made when a cell is drawn): a shell-forces
        # table of a large model has hundreds of thousands of rows
        self.table = QtWidgets.QTableView()
        self.model_rows = _RowsModel(self)
        self.table.setModel(self.model_rows)
        self.table.verticalHeader().setDefaultSectionSize(20)
        layout.addWidget(self.table, 1)
        self._dirty = False
        self.case_box.currentTextChanged.connect(lambda _: self.refresh())
        self.table_box.currentTextChanged.connect(lambda _: self.refresh())
        self.doc.on_change(self._on_change)

    def _on_change(self, what):
        if what in ("results", "model"):
            self.case_box.blockSignals(True)
            current = self.case_box.currentText()
            self.case_box.clear()
            if self.doc.results is not None:
                self.case_box.addItems(list(self.doc.results.cases) + list(self.doc.model.load_combinations))
                if current:
                    self.case_box.setCurrentText(current)
            self.case_box.blockSignals(False)
            self.refresh()

    def set_frame(self, name: str):
        if name == self.selected_frame:
            return  # the same frame again (Model Explorer and the selection both say so)
        self.selected_frame = name
        if self.table_box.currentText().startswith("Frame forces"):
            self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def refresh(self):
        if not self.isVisible():
            self._dirty = True  # filled when the panel is shown
            return
        self._dirty = False
        res = self.doc.results
        case = self.case_box.currentText()
        self.model_rows.set_rows([])
        self._rows = []
        if res is None or not case:
            return
        kind = self.table_box.currentText()
        try:
            if kind == "Joint displacements":
                rows = res.joint_displacements(case)
            elif kind == "Joint reactions":
                rows = res.joint_reactions(case)
            elif kind.startswith("Frame forces"):
                rows = res.frame_forces(self.selected_frame, case).table() if self.selected_frame else []
            elif kind == "Shell forces":
                # rows as arrays: cell text (in the display units) made when a cell is drawn
                table = UnitsTable.of(self.doc.model, kind, res.shell_forces_table(case))
                self._rows = table
                if len(table):
                    self.model_rows.set_rows(table)
                    self._fit_columns()
                return
            elif kind == "Modal periods":
                rows = res.modal_periods(case)
            elif kind == "Modal mass ratios":
                rows = res.modal_participating_mass_ratios(case)
            elif kind == "Story drifts":
                rows = res.story_drifts(case)
            elif kind == "Story shears":
                rows = res.story_shears(case)
            else:
                br = res.base_reactions(case)
                rows = [{"FX": br[0], "FY": br[1], "FZ": br[2]}]
        except Exception as exc:
            rows = [{"error": str(exc)}]
        rows = table_in_display_units(self.doc.model, kind, rows)
        self._rows = rows
        if not rows:
            return
        self.model_rows.set_rows(rows)
        self._fit_columns()

    BIG_TABLE = 1000

    def _fit_columns(self):
        """Column widths from the header and the first rows (Qt reads 1000 rows by
        default: a cell at a time through Python for a table of 491 400 rows)."""
        h = self.table.horizontalHeader()
        h.setResizeContentsPrecision(100 if self.model_rows.rowCount() > self.BIG_TABLE else 1000)
        self.table.resizeColumnsToContents()


    def export_table(self):
        if self._dirty:
            self.refresh()
        if not self._rows:
            return
        name = self.table_box.currentText().split(" (")[0].replace(" ", "_").lower()
        path, flt = QtWidgets.QFileDialog.getSaveFileName(self, "Export table", f"{name}.csv",
                                                          "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        try:
            if isinstance(self._rows, UnitsTable):  # every row, written as it is made
                export_rows(self._rows.iter_rows(), path, cols=self._rows.columns())
            else:
                export_rows(self._rows, path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", str(exc))


class UnitsTable:
    """An array-backed result table (``len``, ``columns(r)``, ``value(r, col)``,
    e.g. :class:`~structstudio.results.ShellForcesTable`) in the display units:
    the unit in the column name and the value converted when a cell is read -
    the same names and values as :func:`table_in_display_units` on the rows."""

    def __init__(self, base, conv):
        self.base, self.conv = base, conv  # base column -> (shown name, factor)
        self._names = {}

    @classmethod
    def of(cls, model, kind: str, base):
        from ..display_units import DisplayUnits
        key = next((k for k in _COLUMN_ITEMS if kind.startswith(k)), None)
        conv = {}
        if key is not None:
            du = DisplayUnits.of(model)
            for col, item in _COLUMN_ITEMS[key].items():
                if item.endswith("/len"):
                    label, k = du.per_length(item[:-4])
                else:
                    label, k = du.label(item), du.factor(item)
                conv[col] = (f"{col} [{label}]", k)
        return cls(base, conv)

    def __len__(self):
        return len(self.base)

    def _shown(self, col: str):
        hit = self._names.get(col)
        if hit is None:
            base = col[:-3] if col.endswith("Min") and col[:-3] in self.conv else col
            if base in self.conv:
                name, k = self.conv[base]
                if base != col:
                    name = name.replace(base, col, 1)
            else:
                name, k = col, None
            hit = self._names[col] = (name, k)
        return hit

    def columns(self, r: int = 0) -> List[str]:
        return [self._shown(c)[0] for c in self.base.columns(r)]

    def _base_columns(self, r: int = 0) -> List[str]:
        return self.base.columns(r)

    def value(self, r: int, base_col: str):
        v = self.base.value(r, base_col)
        k = self._shown(base_col)[1]
        if k is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
            v = v * k
        return v

    def row(self, r: int) -> dict:
        return {self._shown(c)[0]: self.value(r, c) for c in self.base.columns(r)}

    def iter_rows(self):
        for r in range(len(self)):
            yield self.row(r)


class _RowsModel(QtCore.QAbstractTableModel):
    """Read-only table of result rows: a list of dicts (columns of the first row)
    or a :class:`UnitsTable` (values read from its arrays when a cell is drawn)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.cols: List[str] = []
        self._base_cols: List[str] = []

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        if isinstance(rows, UnitsTable):
            self.cols = rows.columns(0) if len(rows) else []
            self._base_cols = rows._base_columns(0) if len(rows) else []
        else:
            self.cols = list(rows[0].keys()) if rows else []
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.cols)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or not index.isValid():
            return None
        if isinstance(self.rows, UnitsTable):
            v = self.rows.value(index.row(), self._base_cols[index.column()])
        else:
            v = self.rows[index.row()].get(self.cols[index.column()], "")
        return f"{v:.6g}" if isinstance(v, float) else str(v)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            from .symbols import pretty
            return pretty(self.cols[section]) if section < len(self.cols) else None
        return str(section + 1)


# result table columns and their display item (Options > Display Units); shell
# resultants are per length ("force/len", "moment/len")
_COLUMN_ITEMS = {
    "Joint displacements": {"Ux": "displacement", "Uy": "displacement", "Uz": "displacement"},
    "Joint reactions": {"FX": "force", "FY": "force", "FZ": "force", "MX": "moment", "MY": "moment", "MZ": "moment"},
    "Frame forces": {"Station": "coordinates", "P": "force", "V2": "force", "V3": "force", "T": "moment",
                     "M2": "moment", "M3": "moment"},
    "Shell forces": {"F11": "force/len", "F22": "force/len", "F12": "force/len", "V13": "force/len",
                     "V23": "force/len", "M11": "moment/len", "M22": "moment/len", "M12": "moment/len"},
    "Story shears": {"VX": "force", "VY": "force", "P": "force"},
    "Base reactions": {"FX": "force", "FY": "force", "FZ": "force"},
}


def table_in_display_units(model, kind: str, rows: List[dict]) -> List[dict]:
    """Result rows with the values in the display units and the unit in the column
    name ("Uz [mm]", "M3 [tonf-m]")."""
    from ..display_units import DisplayUnits
    if not rows or "error" in rows[0]:
        return rows
    key = next((k for k in _COLUMN_ITEMS if kind.startswith(k)), None)
    if key is None:
        return rows
    du = DisplayUnits.of(model)
    conv = {}
    for col, item in _COLUMN_ITEMS[key].items():
        if item.endswith("/len"):
            label, k = du.per_length(item[:-4])
        else:
            label, k = du.label(item), du.factor(item)
        conv[col] = (f"{col} [{label}]", k)
    out = []
    for row in rows:
        new = {}
        for col, v in row.items():
            base = col[:-3] if col.endswith("Min") and col[:-3] in conv else col  # envelope minimum
            if base in conv:
                name, k = conv[base]
                if base != col:
                    name = name.replace(base, col, 1)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    v = v * k
            else:
                name = col
            new[name] = v
        out.append(new)
    return out


def export_rows(rows, path: str, cols: Optional[List[str]] = None):
    """Write table rows (dicts; a list or an iterator, written as they come) to
    .csv or .xlsx (openpyxl when available).  ``cols``: the columns (default:
    those of the first row)."""
    if cols is None:
        rows = list(rows)
        cols = list(rows[0].keys())
    if path.lower().endswith(".xlsx"):
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.append(cols)
        for r in rows:
            ws.append([r.get(c, "") for c in cols])
        wb.save(path)
    else:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})


class LogPanel(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)

    def log(self, text: str):
        self.appendPlainText(text)
