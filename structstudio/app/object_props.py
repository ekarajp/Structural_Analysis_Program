"""ETABS / SAP2000 *Properties of Object* form.

Shown below the Model Explorer while a draw tool is active.  A two-column
property grid (item / value) that lists only the items the active tool uses
(:data:`structstudio.app.tools.TOOL_PROPERTIES`), each edited in place in the
right-hand column: section drop-downs, moment releases, plan offsets, drawing
control, local axis angle, cardinal point, "All stories", and so on.  With the
select tool the form is empty.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from .document import Document
from .forms import display_units
from .symbols import pretty
from .state import ViewState
from .tools import TOOL_PROPERTIES

TOOL_TITLES = {
    "point": "Joint", "beam": "Beam / Column / Brace", "brace": "Brace", "column": "Column",
    "quick_column": "Quick Column", "quick_beam": "Quick Beam", "secondary_beams": "Secondary Beams",
    "quick_brace": "Quick Brace", "wall": "Wall", "quick_wall": "Quick Wall", "slab": "Floor / Wall",
    "rect_slab": "Rectangular Floor / Wall", "quick_slab": "Quick Floor", "opening": "Opening",
    "joint_load": "Joint Load", "frame_load": "Frame Load", "area_load": "Area Load (floors / walls)",
    "panel_load": "Load Panel (drawn load area)",
    "frame_section": "Frame Section",
    "slab_section": "Slab Section", "deck_section": "Deck Section", "wall_section": "Wall Section",
}
SECTION_COMMANDS = ("frame_section", "slab_section", "deck_section", "wall_section")


class ObjectPropertiesPanel(QtWidgets.QWidget):
    """The form; ``state.settings`` (a DrawSettings) holds the values.  Assign and load
    commands show their form here too ("Assign - Restraints"): Apply assigns and keeps
    the box for the next set, OK / Enter assigns and ends, Cancel leaves."""

    title_changed = QtCore.Signal(str)  # the box's (dock's) title

    def __init__(self, state: ViewState, document: Document, parent=None, role: str = "dock"):
        super().__init__(parent)
        # "dock": Properties of Object at the lower left (draw tools, the selection);
        # "assign": the form of an Assign / load command, in its own window (AssignWindow)
        self.role = role
        self.state = state
        self.doc = document
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.title = QtWidgets.QLabel("")
        self.title.setStyleSheet("font-weight: 600")
        lay.addWidget(self.title)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Item", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        lay.addWidget(self.table, 1)
        # the section picker of the Assign Frame Section command (built per refresh)
        self.picker_box = QtWidgets.QWidget()
        self.picker_lay = QtWidgets.QVBoxLayout(self.picker_box)
        self.picker_lay.setContentsMargins(0, 0, 0, 0)
        self.picker_box.setVisible(False)
        lay.addWidget(self.picker_box)
        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray")
        lay.addWidget(self.hint)
        # assign / load commands (their own window): Apply (assign, keep the window for the next
        # set) | OK (= Enter: assign if anything is picked, close) | Cancel (close, change nothing)
        self.picked_label = QtWidgets.QLabel("")
        self.picked_label.setStyleSheet("font-weight: 600")
        self.picked_label.setVisible(False)
        lay.addWidget(self.picked_label)
        self.button_row = QtWidgets.QWidget()
        br = QtWidgets.QHBoxLayout(self.button_row)
        br.setContentsMargins(0, 4, 0, 0)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setToolTip("Assign to the picked objects; the window stays for the next set "
                                     "(right-click in the model does the same)")
        self.apply_button.clicked.connect(self._apply_load)
        self.ok_button = QtWidgets.QPushButton("OK")
        self.ok_button.setToolTip("Assign to the picked objects (if any) and close (Enter)")
        self.ok_button.clicked.connect(self._ok)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setToolTip("Close without assigning (Esc)")
        self.cancel_button.clicked.connect(lambda: self.state.set_tool("select"))
        br.addStretch(1)
        for b in (self.apply_button, self.ok_button, self.cancel_button):
            br.addWidget(b)
        self.button_row.setVisible(False)
        lay.addWidget(self.button_row)
        self._form = None  # the Assign form shown (FormDialog / own widget)
        self._form_tool = None
        self.state.selection_changed.connect(self._update_apply)
        self.state.selection_changed.connect(self._selection_changed)
        self.editors: Dict[str, QtWidgets.QWidget] = {}
        self.tool_name = ""
        self.state.tool_changed.connect(self._tool_changed)
        self.doc.on_change(lambda what: self.refresh() if what in ("model", "results") else None)
        self.state.display_changed.connect(self._display_changed)
        self._tool_changed(self.state.tool.name)

    # A fixed width hint: the form must never push the dock wider when a draw
    # tool shows its longer items (Section / Snap / Plane ...).  A dock growing
    # by 40 px moved and resized the model windows the moment a tool was picked
    # or a line started - the joint under the cursor was then somewhere else,
    # and the click took the wrong point.  Long items scroll inside the table.
    def minimumSizeHint(self):  # noqa: N802 (Qt name)
        return QtCore.QSize(120, super().minimumSizeHint().height())

    def sizeHint(self):  # noqa: N802
        return QtCore.QSize(240, super().sizeHint().height())

    def _display_changed(self):
        pass  # the panel shows the tool's settings or the selection, not the results

    def _selection_changed(self):
        if self.tool_name in ("select", "select_line") and self.state.tool.name in ("select", "select_line"):
            self.refresh()

    # ------------------------------------------------------- selection summary
    def _selection_summary(self) -> bool:
        """With the select tool: the main properties of the selected object(s), in
        short (the full list: right-click > Show Information and Properties).  False if nothing selected."""
        st = self.state
        try:
            title, rows = selection_summary(self.doc.model, sorted(st.frames), sorted(st.areas), sorted(st.points))
        except Exception as exc:  # a summary must never break the panel
            title, rows = "Selection", [("Properties", f"not available ({exc})")]
        if not rows:
            return False
        self.title.setText(title)
        self.hint.setText("Change the values in place (drop-downs / numbers); right-click an object > "
                          "Show Information and Properties for every property.")
        edits = self._selection_edits(sorted(st.frames), sorted(st.areas), sorted(st.points))
        labels = [lb for lb, _ in rows]
        for lb in edits:  # editable items the summary leaves out (no local angle set ...)
            if lb not in labels:
                rows.insert(1 if rows else 0, (lb, ""))
                labels.insert(1 if len(labels) else 0, lb)
        for label, value in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QtWidgets.QTableWidgetItem(pretty(label))  # m², I₃₃, f′c, φ ...
            self.table.setItem(r, 0, it)
            if label in edits:
                self.table.setCellWidget(r, 1, edits[label])
                self.editors["sel:" + label] = edits[label]
                continue
            v = QtWidgets.QTableWidgetItem(pretty(value))
            v.setToolTip(pretty(value))
            if label.startswith("  "):
                it.setForeground(QtCore.Qt.gray)
            if value.startswith("!"):
                v.setText(value[1:])
                v.setForeground(QtCore.Qt.darkRed)
            self.table.setItem(r, 1, v)
        self.table.resizeRowsToContents()

    # ------------------------------------------------- editing the selection in place
    RELEASES = ("Continuous", "Pinned", "Pinned at start", "Pinned at end")
    SUPPORTS = {"Free": (False,) * 6, "Pinned": (True, True, True, False, False, False),
                "Fixed": (True,) * 6, "Roller (UZ)": (False, False, True, False, False, False)}

    def _apply(self, mutate):
        try:
            self.doc.edit(mutate)
        except Exception as exc:
            self.state.report(exc)
            self.refresh()

    def _combo(self, choices, current, apply):
        box = QtWidgets.QComboBox()
        box.addItems(list(choices))
        if current not in choices:
            box.insertItem(0, current or "(mixed)")
        box.setCurrentText(current or "(mixed)")
        box.currentTextChanged.connect(lambda t: apply(t) if t in choices else None)
        return box

    def _number(self, value, suffix, apply, lo=-1e9, hi=1e9, decimals=3):
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setValue(value)
        if suffix:
            sp.setSuffix(" " + suffix)
        sp.setKeyboardTracking(False)
        sp.valueChanged.connect(apply)
        return sp

    def _selection_edits(self, frames, areas, points) -> Dict[str, QtWidgets.QWidget]:
        """The editable items of the selection (all selected objects of the kind change)."""
        from .tools import FREE6, PIN_ENDS
        m = self.doc.model
        out: Dict[str, QtWidgets.QWidget] = {}
        frames = [f for f in frames if f in m.frames]
        areas = [a for a in areas if a in m.areas]
        points = [p for p in points if p in m.points]
        n = len(frames) + len(areas) + len(points)
        if frames:
            label = "Section" if n == 1 else ("  Frame section" if len({m.frames[f].section for f in frames}) == 1
                                              else "  Frame sections")
            secs = {m.frames[f].section for f in frames}
            cur = next(iter(secs)) if len(secs) == 1 else ""

            def set_sec(t, names=frames):
                self._apply(lambda mm: [setattr(mm.frames[f], "section", t) for f in names])
            out[label] = self._combo(sorted(m.frame_sections), cur, set_sec)

            def rel_name(f):
                fr = m.frames[f]
                pi, pj = all(fr.releases_i[4:6]), all(fr.releases_j[4:6])
                if not any(fr.releases_i) and not any(fr.releases_j):
                    return "Continuous"
                if fr.releases_i == PIN_ENDS and fr.releases_j == PIN_ENDS:
                    return "Pinned"
                if pi and not any(fr.releases_j):
                    return "Pinned at start"
                if pj and not any(fr.releases_i):
                    return "Pinned at end"
                return "(other releases)"
            rels = {rel_name(f) for f in frames}

            def set_rel(t, names=frames):
                ri, rj = {"Continuous": (FREE6, FREE6), "Pinned": (PIN_ENDS, PIN_ENDS),
                          "Pinned at start": (PIN_ENDS, FREE6), "Pinned at end": (FREE6, PIN_ENDS)}[t]

                def mut(mm):
                    for f in names:
                        mm.frames[f].releases_i, mm.frames[f].releases_j = tuple(ri), tuple(rj)
                self._apply(mut)
            out["Releases" if n == 1 else "  Releases"] = self._combo(self.RELEASES,
                                                                      next(iter(rels)) if len(rels) == 1 else "",
                                                                      set_rel)
            if n == 1:
                fr = m.frames[frames[0]]
                out["Local axis 2 angle"] = self._number(
                    float(fr.angle), "deg",
                    lambda v, f=fr.name: self._apply(lambda mm: setattr(mm.frames[f], "angle", float(v))), -360, 360, 2)

                def set_type(t, f=fr.name):
                    self._apply(lambda mm: setattr(mm.frames[f], "type", t))
                out["Type"] = self._combo(("Beam", "Column", "Brace", "Null"), fr.type, set_type)
        if areas and not frames:
            secs = {m.areas[a].section for a in areas}
            cur = next(iter(secs)) if len(secs) == 1 else ""

            def set_ssec(t, names=areas):
                self._apply(lambda mm: [setattr(mm.areas[a], "section", t) for a in names])
            if all(not m.areas[a].is_opening for a in areas):
                out["Section" if n == 1 else "  Shell section"] = self._combo(sorted(m.shell_sections), cur or "",
                                                                             set_ssec)
        if points and not frames and not areas:
            sup = {next((k for k, v in self.SUPPORTS.items() if tuple(m.points[p].restraints) == v), "(other)")
                   for p in points}

            def set_sup(t, names=points):
                r = self.SUPPORTS[t]
                self._apply(lambda mm: [setattr(mm.points[p], "restraints", tuple(r)) for p in names])
            out["Support"] = self._combo(tuple(self.SUPPORTS), next(iter(sup)) if len(sup) == 1 else "", set_sup)
        return out

    # ------------------------------------------------------------- section picker
    def _clear_picker(self):
        while self.picker_lay.count():
            w = self.picker_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.picker_box.setVisible(False)

    def _build_section_picker(self):
        """Assign Frame / Slab / Deck / Wall Section: a find box, the sections as a list
        (click to choose), a line about the chosen one and Define sections... (make a new
        one here).  Only the setting changes here; the objects are assigned by Enter /
        right-click (the command's apply)."""
        m, s = self.doc.model, self.state.settings
        frame = self.tool_name == "frame_section"
        tool = self.state.tool
        attr = "frame_section" if frame else getattr(tool, "attr", "shell_section")
        kind = "frame" if frame else getattr(tool, "kind", "Slab").lower()

        def names_all():
            if frame:
                return list(m.frame_sections)
            return tool.sections() if hasattr(tool, "sections") else list(m.shell_sections)

        find = QtWidgets.QLineEdit()
        find.setPlaceholderText("Find a section ...")
        find.setClearButtonEnabled(True)
        lst = QtWidgets.QListWidget()
        lst.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        lst.setMinimumHeight(140)
        info = QtWidgets.QLabel()
        info.setWordWrap(True)
        info.setStyleSheet("color: gray")
        define = QtWidgets.QPushButton("Define sections...  (add a new one)")
        define.setToolTip("Make a new section (or change one) without leaving the command")
        busy = [False]

        def fill(text: str = ""):
            busy[0] = True
            lst.clear()
            every = names_all()
            names = [n for n in every if text.strip().lower() in n.lower()]
            lst.addItems(names)
            if getattr(s, attr) in names:
                lst.setCurrentRow(names.index(getattr(s, attr)))
            info.setText(f"{len(names)} of {len(every)} sections"
                         if every else f"No {kind} section yet - use Define sections.")
            busy[0] = False

        def chosen(row: int):
            if busy[0] or row < 0:
                return
            name = lst.item(row).text()
            setattr(s, attr, name)
            if frame:
                sec = m.frame_sections.get(name)
                if sec is not None:
                    dims = [f"{k} = {getattr(sec, k)}" for k in ("t3", "t2") if getattr(sec, k, None)]
                    info.setText(f"{name}: {sec.shape}, {sec.material}" + (f"  ({', '.join(dims)})" if dims else ""))
            else:
                sec = m.shell_sections.get(name)
                if sec is not None:
                    info.setText(f"{name}: {sec.type}, {sec.material}, thickness {sec.thickness:g} "
                                 f"{m.units.length}")
            self.state.update_hint()
            self._update_apply()

        def open_define():
            before = set(m.frame_sections if frame else m.shell_sections)
            if frame:
                from .section_manager import FrameSectionsDialog
                FrameSectionsDialog(self.doc, self.window()).exec()
            else:
                w = self.window()
                if not hasattr(w, "define_shell_sections") and w.parentWidget() is not None:
                    w = w.parentWidget().window()  # (the Assign window: its program window)
                if hasattr(w, "define_shell_sections"):
                    w.define_shell_sections()
            mm = self.doc.model
            new = [n for n in (mm.frame_sections if frame else mm.shell_sections) if n not in before]
            if new:
                setattr(s, attr, new[-1])  # a section just made is the one to assign
            fill(find.text())
            self.refresh()

        find.textChanged.connect(fill)
        lst.currentRowChanged.connect(chosen)
        lst.itemActivated.connect(lambda _item: self._apply_load())  # Enter in the list applies
        find.returnPressed.connect(self._apply_load)
        define.clicked.connect(open_define)
        self.picker_lay.addWidget(find)
        self.picker_lay.addWidget(lst)
        self.picker_lay.addWidget(info)
        self.picker_lay.addWidget(define)
        self.picker_box.setVisible(True)
        self.section_list, self.define_button = lst, define
        fill()
        if getattr(s, attr) and lst.currentRow() >= 0:
            chosen(lst.currentRow())

    def _build_assign_form(self, tool, spec):
        """An Assign command's own form, in the box: the values of the objects picked
        before the command (or the defaults); Apply / OK / Cancel below."""
        from .forms import FormDialog
        self._set_title(f"Assign - {tool.title.replace('Assign ', '')}")
        n = len(tool.picked(self.state))
        self.hint.setText((f"Showing the values of the {n} picked {tool.plural(n)}. " if n else "") +
                          f"Pick {tool.plural()} in the model (click / window). Apply = assign and pick the "
                          "next set, OK / Enter = assign and close, Cancel / Esc = close.")
        self.table.setVisible(False)
        if spec.widget is not None:
            w, values = spec.widget(tool.first)
        else:
            w = FormDialog(tool.title, tool.fields, self, spec.note)
            bb = w.findChild(QtWidgets.QDialogButtonBox)
            if bb is not None:
                bb.hide()
            values = w.values
        w.setWindowFlags(QtCore.Qt.Widget)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(w)
        self.picker_lay.addWidget(scroll, 1)
        self.picker_box.setVisible(True)
        tool.read_values = values
        self._form, self._form_tool = w, tool

    # ------------------------------------------------------------- build
    def _tool_changed(self, name: str):
        self.tool_name = name
        self.refresh()

    def _is_load_command(self) -> bool:
        return bool(getattr(self.state.tool, "picks", None))

    def _apply_load(self):
        self.state.apply_load_tool()

    def _ok(self):
        """OK: assign to the picked objects (if any), then close the command."""
        tool = self.state.tool
        if hasattr(tool, "picked") and tool.picked(self.state):
            if not self.state.finish_load_tool():
                return  # (an error was shown: the window stays)
        else:
            self.state.set_tool("select")

    def _set_title(self, text: str):
        self.title.setText(text)
        tool = self.state.tool
        if self.role == "assign" and getattr(tool, "picks", None):
            box = f"Assign - {getattr(tool, 'title', '').replace('Assign ', '')}"
        else:
            box = "Properties of Object"
        self.title_changed.emit(box)

    def _update_apply(self):
        on = self._is_load_command() and self.role == "assign"
        self.button_row.setVisible(on)
        self.picked_label.setVisible(on)
        if on:
            tool = self.state.tool
            n = len(tool.picked(self.state))
            spec = getattr(tool, "spec", None)
            self.apply_button.setVisible(spec is None or spec.apply is not None)
            self.apply_button.setEnabled(n > 0)
            what = f"{n} {tool.plural(n)} picked" if n else f"Pick {tool.plural()} in the model (click / window)"
            if n and not getattr(tool, "apply_label", None) and not getattr(tool, "spec", None):
                what += f"  ({self.state.settings.load_mode})"
            self.picked_label.setText(what)

    def refresh(self):
        tool = self.state.tool
        picks = bool(getattr(tool, "picks", None))
        if picks != (self.role == "assign"):
            # a pick command's form is in its own window; draw tools / the selection here
            self._form, self._form_tool = None, None
            self.editors.clear()
            self.table.setRowCount(0)
            self._clear_picker()
            self._update_apply()
            if self.role == "dock":
                self.table.setVisible(True)
                self._set_title("")
                self.hint.setText(f"{tool.title}: its form is open in its own window (Apply / OK / Cancel).")
            return
        spec = getattr(tool, "spec", None)
        if spec is not None and self._form_tool is tool and self._form is not None:
            self._update_apply()  # the same command: keep the form and the values typed in it
            self._set_title(self.title.text())
            return
        self._form, self._form_tool = None, None
        if spec is not None:
            self.editors.clear()
            self.table.setRowCount(0)
            self._clear_picker()
            self._build_assign_form(tool, spec)
            self._update_apply()
            self.state.update_hint()
            return
        self.table.setVisible(True)
        items = TOOL_PROPERTIES.get(self.tool_name)
        self.editors.clear()
        self.table.setRowCount(0)
        self._clear_picker()
        self._update_apply()
        if not items:
            self._set_title("")
            self.hint.setText("Select a draw tool: its drawing properties appear here."
                              if self.tool_name in ("select", "select_line") else "")
            if self.tool_name in ("select", "select_line") and self._selection_summary():
                pass
            return
        what = TOOL_TITLES.get(self.tool_name, self.tool_name)
        self._set_title(f"Assign - {what.replace('Assign ', '')}" if self._is_load_command()
                        else f"Properties of Object - {what}")
        if self.tool_name in SECTION_COMMANDS:
            what = {"frame_section": "frames", "wall_section": "walls"}.get(self.tool_name, "floors")
            self.hint.setText(f"Pick the section here, then pick {what} in the model and press Enter / "
                              "right-click. Nothing changes until then.")
            self.table.setVisible(False)
            self._build_section_picker()
            self.state.update_hint()
            return
        self.table.setVisible(True)
        if self._is_load_command():
            tool = self.state.tool
            self.hint.setText(
                f"Editing the {self.state.settings.load_pattern} load of the selected {tool.plural()}: change the "
                f"values, then Enter / right-click / the button." if tool.editing else
                f"Set the load, pick {tool.plural()} in the model (click or drag a window), then Enter / "
                f"right-click / the button.  With {tool.plural()} selected before the command, their load "
                f"shows here to edit.")
        else:
            self.hint.setText("Click a value to change it; the settings apply to every object drawn next.")
        s = self.state.settings
        m = self.doc.model
        for key, label, kind, extra in items:
            if key in ("fixed_length", "fixed_angle") and s.drawing_control not in ("Fixed length", "Fixed angle"):
                continue  # only with the matching drawing control
            if kind == "section_list":
                continue  # shown by the picker below the table
            if key == "fixed_length" and s.drawing_control != "Fixed length":
                continue
            if key == "fixed_angle" and s.drawing_control != "Fixed angle":
                continue
            if key == "arc_segments" and not str(getattr(s, "line_shape", "Straight")).startswith("Arc"):
                continue  # only for arcs
            if key == "arc_radius" and getattr(s, "line_shape", "") != "Arc: start, end, radius":
                continue
            if key == "load_value_b" and s.frame_load_kind != "Trapezoidal":
                continue  # only for the matching load kind
            if key == "load_rel" and s.frame_load_kind != "Point":
                continue
            if key == "area_span" and s.area_distribution != "One-way to frames":
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            item = self._setting_item(key) if kind == "float" else None
            du = display_units()
            if item and du is not None:
                label = f"{label} [{du.label(item)}]"
            it = QtWidgets.QTableWidgetItem(pretty(label))
            it.setToolTip(pretty(str(extra)) if isinstance(extra, str) else "")
            self.table.setItem(r, 0, it)
            w = self._editor(key, kind, extra, s, m)
            self.table.setCellWidget(r, 1, w)
            self.editors[key] = w
        self.table.resizeRowsToContents()

    def _cardinal_editor(self, key, s, m) -> QtWidgets.QWidget:
        """Cardinal point: a drop-down with the names (Auto: beams 8, columns 10), and ... =
        the Insertion Point form (picture of the section, include the offset in the
        analysis or not)."""
        from .insertion_dialog import AUTO_TEXT, CARDINAL_NAMES, InsertionPointDialog
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(2)
        box = QtWidgets.QComboBox()
        box.addItems([AUTO_TEXT] + CARDINAL_NAMES)
        cur = getattr(s, key)
        box.setCurrentIndex(0 if cur is None else max(1, min(11, int(cur))))
        box.setToolTip("The point of the section on the drawn line.  Auto: beams 8 (top flush with the slab), "
                       "columns and braces 10 (centroid)")
        box.currentIndexChanged.connect(lambda i: setattr(s, key, None if i == 0 else i))
        more = QtWidgets.QToolButton()
        more.setText("...")
        more.setToolTip("Insertion Point: the section with its points; include the offset in the analysis or not")

        def open_form():
            cp = getattr(s, key)
            if cp is None:
                cp = 10 if self.tool_name in ("column", "quick_column", "brace", "quick_brace") else 8
            d = InsertionPointDialog(m, s.frame_section, int(cp), bool(getattr(s, "cardinal_transform", False)),
                                     self.window())
            if d.exec():
                v = d.form.values()
                s.cardinal_transform = v["transform"]
                box.setCurrentIndex(v["cp"])
        more.clicked.connect(open_form)
        h.addWidget(box, 1)
        h.addWidget(more)
        w.box, w.more = box, more
        return w

    def _editor(self, key, kind, extra, s, m) -> QtWidgets.QWidget:
        if kind == "cardinal":
            return self._cardinal_editor(key, s, m)
        val = getattr(s, key)
        if kind in ("section_frame", "section_shell", "diaphragm", "choice", "pattern"):
            box = QtWidgets.QComboBox()
            if kind == "pattern":
                items = list(m.load_patterns)
                if val not in items and items:
                    val = items[0]
                    setattr(s, key, val)
            elif kind == "section_frame":
                items = list(m.frame_sections)
            elif kind == "section_shell":
                want = "Wall" if key == "wall_section" else None
                items = [n for n, x in m.shell_sections.items() if want is None or x.type == want] or list(m.shell_sections)
            elif kind == "diaphragm":
                items = ["(none)"] + list(m.diaphragms)
            else:
                items = list(extra)
            box.addItems(items)
            cur = "(none)" if (kind == "diaphragm" and not val) else (val if val in items else None)
            if cur is None and items and kind in ("section_frame", "section_shell"):
                cur = items[0]
                setattr(s, key, cur)
            if cur is not None:
                box.setCurrentText(cur)

            def changed(text, k=key, kd=kind):
                setattr(s, k, None if (kd == "diaphragm" and text == "(none)") else text)
                tool = self.state.tool
                if k == "load_pattern" and getattr(tool, "editing", False):
                    tool.prefill(tool.picked(self.state))  # show the picked objects' load of that pattern
                    self.refresh()
                    self.state.update_hint()
                elif k in ("drawing_control", "frame_load_kind", "area_distribution", "line_shape"):
                    self.refresh()
                    if k == "line_shape":
                        self.state.tool.cancel()
                        self.state.update_hint()
                elif k == "load_mode":
                    self._update_apply()
                    self.state.update_hint()
            box.currentTextChanged.connect(changed)
            return box
        if kind == "bool":
            box = QtWidgets.QComboBox()
            yes, no = ("All Stories", "One Story") if key == "all_stories" else ("Yes", "No")
            if key == "all_stories" and m.grid_mode == "xyz":
                yes, no = "All Levels", "One Level"
            box.addItems([no, yes])
            box.setCurrentIndex(1 if val else 0)
            box.currentIndexChanged.connect(lambda i, k=key: setattr(s, k, i == 1))
            return box
        if kind == "int":
            sp = QtWidgets.QSpinBox()
            sp.setRange(1, 999)
            sp.setValue(int(val or 1))
            sp.valueChanged.connect(lambda v, k=key: setattr(s, k, int(v)))
            return sp
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(-1e9, 1e9)
        sp.setDecimals(4)
        item = self._setting_item(key)
        du = display_units()
        f = du.factor(item) if (item and du is not None) else 1.0  # shown in the display unit
        sp.setValue(float(val or 0.0) * f)
        sp.valueChanged.connect(lambda v, k=key, ff=f: setattr(s, k, float(v) / ff))
        if key in ("offset_x", "offset_y", "offset_z"):  # a non-zero offset moves what is drawn: show it
            def mark(v, w=sp):
                w.setStyleSheet("" if abs(v) < 1e-12 else
                                "QDoubleSpinBox { color: #e07000; font-weight: bold; }")
                w.setToolTip("" if abs(v) < 1e-12 else
                             "Non-zero offset: every picked point of this tool is moved by it")
            sp.valueChanged.connect(mark)
            mark(sp.value())
        return sp

    def _setting_item(self, key: str):
        """Display item of a drawing / load setting (Options > Display Units)."""
        if key in ("offset_x", "offset_y", "offset_z", "fixed_length", "offset_distance", "arc_radius"):
            return "coordinates"
        if key in ("load_fx", "load_fy", "load_fz"):
            return "force"
        if key in ("load_mx", "load_my", "load_mz"):
            return "moment"
        if key in ("load_value", "load_value_b"):
            if self.tool_name == "area_load":
                return "area_load"
            return "force" if self.state.settings.frame_load_kind == "Point" else "line_load"
        return None


class AssignWindow(QtWidgets.QDialog):
    """The window of an Assign / load command: its own window (not modal - pick objects in
    the model while it is open), with Apply / OK / Cancel.  Shown while such a command
    runs, closed when it ends; closing it (X / Esc) cancels the command."""

    def __init__(self, state: ViewState, document: Document, parent=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.Tool, True)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setModal(False)
        self.state = state
        self.setWindowTitle("Assign")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.panel = ObjectPropertiesPanel(state, document, self, role="assign")
        lay.addWidget(self.panel)
        self.panel.title_changed.connect(self.setWindowTitle)
        state.tool_changed.connect(self._tool_changed)
        self._placed = False
        self.resize(360, 460)

    def _tool_changed(self, _name: str):
        if getattr(self.state.tool, "picks", None):
            if not self.isVisible():
                self._place()
                self.show()
            self.raise_()
        elif self.isVisible():
            self.hide()

    def _place(self):
        """First time: near the top left of the program window, over the drawing."""
        if self._placed or self.parentWidget() is None:
            return
        self._placed = True
        top = self.parentWidget().window()
        g = top.geometry()
        self.move(g.x() + 260, g.y() + 120)

    def reject(self):  # Esc / X: cancel the command (nothing assigned)
        if getattr(self.state.tool, "picks", None):
            self.state.set_tool("select")
        super().reject()

    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and not isinstance(
                QtWidgets.QApplication.focusWidget(), (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
            self.panel._ok()  # Enter = OK
            return
        super().keyPressEvent(ev)


def _sec_text(model, name: Optional[str], du) -> str:
    sec = model.frame_sections.get(name) if name else None
    if sec is None:
        return str(name or "-")
    from .section_manager import dims_text, type_label
    return f"{sec.name}  ({dims_text(sec, du)}, {type_label(sec)})"


def fmt(v) -> str:
    from .scene import fmt_value
    return fmt_value(float(v))


def _material_text(model, mat_name: Optional[str], du) -> str:
    mat = model.materials.get(mat_name) if mat_name else None
    if mat is None:
        return str(mat_name or "-")
    ks, us = du.factor("stress"), du.label("stress")
    parts = [mat.name]
    if mat.fc:
        parts.append(f"fc' {fmt(mat.fc * ks)} {us}")
    if mat.fy:
        parts.append(f"fy {fmt(mat.fy * ks)} {us}")
    return ", ".join(parts)


def selection_summary(model, frames, areas, points):
    """(title, rows) of the selected objects in short, in the display units."""
    from ..display_units import DisplayUnits
    from .supports import describe
    du = DisplayUnits.of(model)
    kc, uc = du.factor("coordinates"), du.label("coordinates")
    n = len(frames) + len(areas) + len(points)
    if n == 0:
        return "", []
    rows: List[tuple] = []
    if n > 1:  # several: what is selected, and what they have in common
        title = f"Selection - {n} objects"
        for kind, names in (("Frames", frames), ("Shells", areas), ("Joints", points)):
            if names:
                rows.append((kind, f"{len(names)}: " + ", ".join(names[:6]) + (" ..." if len(names) > 6 else "")))
        secs = {model.frames[f].section for f in frames if f in model.frames}
        if len(secs) == 1:
            rows.append(("  Frame section", _sec_text(model, next(iter(secs)), du)))
        elif secs:
            rows.append(("  Frame sections", ", ".join(sorted(map(str, secs)))))
        ssecs = {model.areas[a].section for a in areas if a in model.areas}
        if ssecs:
            rows.append(("  Shell section" + ("s" if len(ssecs) > 1 else ""), ", ".join(sorted(map(str, ssecs)))))
        L = sum(math.dist(model.points[model.frames[f].point_i].xyz, model.points[model.frames[f].point_j].xyz)
                for f in frames if f in model.frames)
        if frames:
            rows.append(("  Total frame length", f"{fmt(L * kc)} {uc}"))
        return title, rows
    if frames:
        fr = model.frames[frames[0]]
        a, b = model.points[fr.point_i], model.points[fr.point_j]
        title = f"{fr.type} {fr.label or fr.name}" + (f"  ({fr.name}, {fr.story})" if fr.story else f"  ({fr.name})")
        sec = model.frame_sections.get(fr.section)
        rows.append(("Section", _sec_text(model, fr.section, du)))
        if sec is not None:
            rows.append(("Material", _material_text(model, sec.material, du)))
        rows.append(("Length", f"{fmt(math.dist(a.xyz, b.xyz) * kc)} {uc}"))
        rows.append(("Joints (start - end)", f"{fr.point_i} - {fr.point_j}"))
        rel = []
        for end, r in (("Start", fr.releases_i), ("End", fr.releases_j)):
            if any(r[3:]):
                rel.append(f"{end}: " + " ".join(lab for lab, on in zip(("T", "M2", "M3"), r[3:]) if on))
        rows.append(("Releases", "; ".join(rel) if rel else "none (continuous)"))
        if fr.end_offset_i or fr.end_offset_j:
            rows.append(("End offsets", f"{fmt(fr.end_offset_i * kc)} / {fmt(fr.end_offset_j * kc)} {uc}"))
        if abs(fr.angle) > 1e-9:
            rows.append(("Local axis 2 angle", f"{fmt(fr.angle)} deg"))
        loads = []
        for ld in fr.loads:
            if ld.kind == "distributed":
                item = "moment" if ld.load_type == "Moment" else "line_load"
                k, u = du.factor(item), du.label(item)
                v = fmt(ld.value_a * k) if ld.value_a == ld.value_b else f"{fmt(ld.value_a * k)}-{fmt(ld.value_b * k)}"
                loads.append(f"{ld.pattern} {v} {u}" + ("/" + uc if ld.load_type == "Moment" else ""))
            elif ld.kind == "point":
                item = "moment" if ld.load_type == "Moment" else "force"
                loads.append(f"{ld.pattern} {fmt(ld.value_a * du.factor(item))} {du.label(item)} (point)")
            else:
                loads.append(f"{ld.pattern} ({ld.kind})")
        rows.append(("Loads", "; ".join(loads) if loads else "none"))
        return title, rows
    if areas:
        ar = model.areas[areas[0]]
        title = f"{ar.type} {ar.label or ar.name}" + (f"  ({ar.name}, {ar.story})" if ar.story else f"  ({ar.name})")
        ss = model.shell_sections.get(ar.section) if ar.section else None
        if ss is not None:
            rows.append(("Section", f"{ss.name}  ({fmt(ss.thickness * du.factor('section_dim'))} "
                                    f"{du.label('section_dim')}, {ss.model_type})"))
            rows.append(("Material", _material_text(model, ss.material, du)))
        else:
            rows.append(("Section", "opening" if ar.is_opening else "-"))
        try:
            from ..mesh import polygon2d as pg
            import numpy as np
            P = np.array([model.points[p].xyz for p in ar.points])
            A = abs(pg.signed_area(pg.to_2d(P, pg.plane_frame(P))))
            rows.append(("Area", f"{fmt(A * kc * kc)} {uc}²"))
        except Exception:
            pass
        rows.append(("Corners", str(len(ar.points))))
        ka, ua = du.factor("area_load"), du.label("area_load")
        loads = [f"{c.pattern} {fmt(c.value * ka)} {ua}" for p in model.panel_loads.values() if p.area == ar.name
                 for c in p.loads] + [f"{ld.pattern} {fmt(ld.value * ka)} {ua}" for ld in ar.loads]
        rows.append(("Loads", "; ".join(loads) if loads else "none"))
        if ar.diaphragm:
            rows.append(("Diaphragm", ar.diaphragm))
        return title, rows
    p = model.points[points[0]]
    title = f"Joint {p.label or p.name}" + (f"  ({p.name}, {p.story})" if p.story else f"  ({p.name})")
    rows.append(("Coordinates", f"{fmt(p.x * kc)}, {fmt(p.y * kc)}, {fmt(p.z * kc)} {uc}"))
    rows.append(("Support", describe(p.restraints, p.local_axes) if any(p.restraints) else "free"))
    kf, uf = du.factor("force"), du.label("force")
    km, um = du.factor("moment"), du.label("moment")
    loads = []
    for ld in p.loads:
        parts = [f"F{ax} {fmt(v * kf)}" for ax, v in zip("xyz", ld.force) if v]
        parts += [f"M{ax} {fmt(v * km)}" for ax, v in zip("xyz", ld.moment) if v]
        if parts:
            loads.append(f"{ld.pattern}: " + ", ".join(parts))
    rows.append(("Loads", ("; ".join(loads) + f"  ({uf}, {um})") if loads else "none"))
    conn = [f.name for f in model.frames.values() if p.name in (f.point_i, f.point_j)]
    rows.append(("Connected frames", ", ".join(conn) if conn else "none"))
    return title, rows


def results_summary(model, results, case=None) -> List[tuple]:
    """(item, value) rows: the summary shown in the Properties panel."""
    import numpy as np
    from .scene import display_length, fmt_value
    am = results.am
    cases = list(results.cases) if hasattr(results, "cases") else []
    if case is None or (case not in cases and case not in model.load_combinations):
        case = cases[0] if cases else None
    rows: List[tuple] = []
    s = am.summary()
    rows.append(("Analysis model", f"{s['nodes']} nodes, {s['frame_elements']} frame, {s['shell_elements']} shell elements"))
    sizes = sorted(set(getattr(am, "auto_sizes", {}).values()))
    if sizes:
        from ..display_units import DisplayUnits
        dq = DisplayUnits.of(model)
        kc = dq.si_factor("coordinates")  # the sizes are SI
        lo, hi = fmt_value(sizes[0] * kc), fmt_value(sizes[-1] * kc)
        rows.append(("  Shell mesh size", lo + (f" - {hi}" if len(sizes) > 1 else "") +
                     f" {dq.label('coordinates')} (automatic)"))
    mc = getattr(am, "mesh_check", None)
    if mc:
        e = 100 * float(mc.get("estimated_error", 0.0))
        tol = 100 * float(mc.get("tolerance", 0.01))
        txt = f"{e:.2f} % estimated (limit {tol:g} %)"
        rows.append(("  Mesh check", txt if mc.get("converged") else "!" + txt + " - refine the mesh"))
    if case is None:
        return rows
    rows.append(("Load case", case))
    fac, unit = display_length(model)
    try:
        D = np.asarray(results.displacement_envelope(case).max, float)
    except Exception:
        D = np.zeros((0, 6))
    if len(D):
        Lf = results.U.factor("length")
        k = int(np.argmax(np.abs(D[:, 2])))
        x, y, z = (np.asarray(am.nodes[k].xyz, float) / Lf)
        where = am.nodes[k].point or f"({x:.2f}, {y:.2f}, {z:.2f})"
        rows.append(("Largest vertical deflection", f"{fmt_value(D[k, 2] * fac)} {unit} at {where}"))
        k2 = int(np.argmax(np.linalg.norm(D[:, :2], axis=1)))
        if np.linalg.norm(D[k2, :2]) > 1e-12:
            x, y, z = (np.asarray(am.nodes[k2].xyz, float) / Lf)
            where = am.nodes[k2].point or f"({x:.2f}, {y:.2f}, {z:.2f})"
            rows.append(("Largest horizontal displacement",
                         f"{fmt_value(float(np.linalg.norm(D[k2, :2])) * fac)} {unit} at {where}"))
    try:
        from ..display_units import DisplayUnits
        dq = DisplayUnits.of(model)
        kf, fu = dq.factor("force"), dq.label("force")
        base = np.asarray(results.base_reactions(case), float) * kf
        big = max(float(np.max(np.abs(base[:3]))), 1e-30)
        rows.append(("Base reaction FX / FY / FZ",
                     " / ".join(fmt_value(v if abs(v) > 1e-9 * big else 0.0) for v in base[:3]) + f" {fu}"))
        reac = results.joint_reactions(case)
        if reac:
            r = max(reac, key=lambda q: abs(q.get("FZ", 0.0)))
            rows.append(("  Largest support FZ", f"{fmt_value(r['FZ'] * kf)} {fu} at {r.get('Point') or '?'}"))
    except Exception:
        pass
    issues = [i for i in getattr(results, "issues", []) if getattr(i, "severity", "") in ("error", "warning")]
    if issues:
        rows.append(("Warnings", f"!{len(issues)} - Analyze > Last Analysis Problems"))
    return rows
