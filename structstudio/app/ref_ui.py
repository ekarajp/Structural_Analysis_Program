"""The References menu: CAD drawings (DXF) placed in plan and elevation windows to draw
over.  The forms (place a drawing, its layers, the list of references) and the menu
commands of the main window (:class:`ReferenceCommandsMixin`)."""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from . import cadref
from . import ref_tools  # noqa: F401  (registers the tools)
from ..io.dxf_read import UNIT_NAMES, UNIT_TO_M, Drawing, DxfError, read_dxf
from ..model import ModelError

UNIT_LABELS = {"mm": "millimetres (mm)", "cm": "centimetres (cm)", "m": "metres (m)", "in": "inches (in)",
               "ft": "feet (ft)", "dm": "decimetres (dm)", "km": "kilometres (km)", "yd": "yards (yd)",
               "mi": "miles (mi)"}
FILE_FILTER = "DXF drawings (*.dxf);;All files (*)"


def _du(model) -> Tuple[float, str]:
    """(factor model length -> shown, its label)."""
    try:
        from ..display_units import DisplayUnits
        du = DisplayUnits.of(model)
        return du.factor("coordinates"), du.label("coordinates")
    except Exception:
        return 1.0, model.units.length


def size_text(dwg: Drawing, unit: str) -> str:
    """The size of the drawing in metres with ``unit`` as its drawing unit - what tells a
    wrong unit at a glance (a floor plan 42 km or 4 cm wide)."""
    w, h = dwg.size()
    f = UNIT_TO_M.get(unit, 1.0)
    w, h = w * f, h * f

    def fmt(v):
        if v >= 1000:
            return f"{v / 1000:,.2f} km"
        if v < 1:
            return f"{v * 100:,.1f} cm"
        return f"{v:,.2f} m"
    return f"{fmt(w)} x {fmt(h)}"


def drawing_text(dwg: Drawing) -> str:
    n = len(dwg.seg)
    t = f"{n:,} line{'s' if n != 1 else ''} on {sum(1 for v in dwg.layer_counts().values() if v)} layer(s)"
    if dwg.skipped:
        words = {"TEXT": ("text", "texts"), "MTEXT": ("text", "texts"), "HATCH": ("hatch", "hatches"),
                 "IMAGE": ("image", "images"), "ATTRIB": ("text", "texts"), "ATTDEF": ("text", "texts")}
        other = ("other object", "other objects")
        cnt: Dict[tuple, int] = {}
        for k, v in dwg.skipped.items():
            cnt[words.get(k, other)] = cnt.get(words.get(k, other), 0) + v
        t += "   (not drawn: " + ", ".join(f"{v:,} {k[0] if v == 1 else k[1]}" for k, v in cnt.items()) + ")"
    return t


class LayerList(QtWidgets.QWidget):
    """The layers of a drawing with a tick each (the number of lines beside the name)."""

    changed = QtCore.Signal()

    def __init__(self, dwg: Drawing, off, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.list = QtWidgets.QListWidget()
        counts = dwg.layer_counts()
        off = set(off)
        for name in sorted(dwg.layers, key=str.lower):
            if not counts.get(name):
                continue  # (a layer with nothing to draw)
            it = QtWidgets.QListWidgetItem(f"{name}    ({counts[name]:,})")
            it.setData(QtCore.Qt.UserRole, name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked if name in off else QtCore.Qt.Checked)
            self.list.addItem(it)
        self._hidden = [n for n in off if not counts.get(n)]
        self.list.itemChanged.connect(lambda _: self.changed.emit())
        v.addWidget(self.list)
        h = QtWidgets.QHBoxLayout()
        for text, on in (("All", True), ("None", False)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, on=on: self.set_all(on))
            h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)

    def set_all(self, on: bool):
        self.list.blockSignals(True)
        for k in range(self.list.count()):
            self.list.item(k).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
        self.list.blockSignals(False)
        self.changed.emit()

    def off(self) -> List[str]:
        out = [self.list.item(k).data(QtCore.Qt.UserRole) for k in range(self.list.count())
               if self.list.item(k).checkState() != QtCore.Qt.Checked]
        return sorted(out + self._hidden)

    def set_off(self, off):
        off = set(off)
        self.list.blockSignals(True)
        for k in range(self.list.count()):
            it = self.list.item(k)
            it.setCheckState(QtCore.Qt.Unchecked if it.data(QtCore.Qt.UserRole) in off else QtCore.Qt.Checked)
        self.list.blockSignals(False)


class PlaceReferenceDialog(QtWidgets.QDialog):
    """References > Place DXF Drawing: the drawing unit (with the size it gives), where it
    goes (plans of stories / an elevation), how it is positioned and kept, its layers."""

    def __init__(self, model, path: str, dwg: Drawing, view: str = "plan", story: Optional[str] = None,
                 ordinate: float = 0.0, near_model: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Place DXF Drawing")
        self.model, self.dwg, self.path = model, dwg, path
        k, unit = _du(model)
        self._k = k
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        name = QtWidgets.QLabel(os.path.basename(path))
        name.setToolTip(path)
        form.addRow("File", name)
        form.addRow("Drawing", QtWidgets.QLabel(drawing_text(dwg)))
        self.unit = QtWidgets.QComboBox()
        for u in UNIT_NAMES:
            self.unit.addItem(UNIT_LABELS[u], u)
        self.unit.setCurrentIndex(UNIT_NAMES.index(dwg.guess_unit()))
        self.unit_note = QtWidgets.QLabel("(from the file)" if dwg.unit else "(the file does not say - check the size)")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.unit)
        row.addWidget(self.unit_note, 1)
        form.addRow("Drawing unit", row)
        self.size = QtWidgets.QLabel()
        self.size.setStyleSheet("font-weight: 600;")
        form.addRow("Size of the drawing", self.size)
        self.unit.currentIndexChanged.connect(self._size)
        lay.addLayout(form)

        # ---- where
        where = QtWidgets.QGroupBox("Place on")
        g = QtWidgets.QGridLayout(where)
        self.on_plan = QtWidgets.QRadioButton("Plan of the stories ticked")
        self.on_xz = QtWidgets.QRadioButton(f"Elevation X-Z at Y = ")
        self.on_yz = QtWidgets.QRadioButton(f"Elevation Y-Z at X = ")
        self.stories = QtWidgets.QListWidget()
        names = [s.name for s in model.stories_bottom_up()]
        for s in reversed(names):
            it = QtWidgets.QListWidgetItem(s)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if s == story else QtCore.Qt.Unchecked)
            self.stories.addItem(it)
        self.stories.setMaximumHeight(120)
        self.y_at = QtWidgets.QLineEdit(f"{ordinate * k:g}" if view == "xz" else "0")
        self.x_at = QtWidgets.QLineEdit(f"{ordinate * k:g}" if view == "yz" else "0")
        for w in (self.y_at, self.x_at):
            w.setMaximumWidth(90)
        g.addWidget(self.on_plan, 0, 0, 1, 3)
        g.addWidget(self.stories, 1, 0, 1, 3)
        g.addWidget(self.on_xz, 2, 0)
        g.addWidget(self.y_at, 2, 1)
        g.addWidget(QtWidgets.QLabel(unit), 2, 2)
        g.addWidget(self.on_yz, 3, 0)
        g.addWidget(self.x_at, 3, 1)
        g.addWidget(QtWidgets.QLabel(unit), 3, 2)
        g.setColumnStretch(2, 1)
        {"plan": self.on_plan, "xz": self.on_xz, "yz": self.on_yz}.get(view, self.on_plan).setChecked(True)
        if not names:  # (no stories: a plan at a level)
            self.stories.hide()
            self.on_plan.setText("Plan at the level of the window")
        lay.addWidget(where)

        # ---- position
        pos = QtWidgets.QGroupBox("Position")
        v = QtWidgets.QVBoxLayout(pos)
        self.pos_own = QtWidgets.QRadioButton("The drawing's own coordinates (its 0, 0 at the model's 0, 0)")
        self.pos_mid = QtWidgets.QRadioButton("In the middle of the window - then put it in place with two points")
        (self.pos_own if near_model else self.pos_mid).setChecked(True)
        self.align = QtWidgets.QCheckBox("Start Align (2 Points) when it is placed")
        self.align.setChecked(True)
        for w in (self.pos_own, self.pos_mid, self.align):
            v.addWidget(w)
        lay.addWidget(pos)

        # ---- how it is kept
        keep = QtWidgets.QGroupBox("Keep the drawing")
        v = QtWidgets.QVBoxLayout(keep)
        self.link = QtWidgets.QCheckBox("Linked: read from the file (Reload brings in its changes)")
        self.embed = QtWidgets.QCheckBox("Embedded: a copy inside the model file (the model can be sent alone)")
        self.link.setChecked(True)
        v.addWidget(self.link)
        v.addWidget(self.embed)
        lay.addWidget(keep)

        box = QtWidgets.QGroupBox("Layers shown")
        v = QtWidgets.QVBoxLayout(box)
        self.layers = LayerList(dwg, dwg.layers_off)
        self.layers.list.setMaximumHeight(140)
        v.addWidget(self.layers)
        lay.addWidget(box)

        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._size()

    def _size(self):
        self.size.setText(size_text(self.dwg, self.unit.currentData()))

    def accept(self):
        try:
            self.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        super().accept()

    def values(self) -> dict:
        view = "plan" if self.on_plan.isChecked() else "xz" if self.on_xz.isChecked() else "yz"
        stories = [self.stories.item(i).text() for i in range(self.stories.count())
                   if self.stories.item(i).checkState() == QtCore.Qt.Checked][::-1]
        level = 0.0
        if view == "plan":
            if self.stories.count() and not stories:
                raise ValueError("Tick the story (or stories) whose plan shows the drawing.")
        else:
            try:
                level = float((self.y_at if view == "xz" else self.x_at).text().replace(",", "")) / self._k
            except ValueError:
                raise ValueError("The position of the elevation is not a number.") from None
        if not (self.link.isChecked() or self.embed.isChecked()):
            raise ValueError("Tick Linked, Embedded, or both.")
        if not len(self.dwg.seg):
            raise ValueError("The drawing has no lines to show.")
        return {"view": view, "stories": stories, "level": level, "unit": self.unit.currentData(),
                "own": self.pos_own.isChecked(), "align": self.align.isChecked(), "link": self.link.isChecked(),
                "embed": self.embed.isChecked(), "layers_off": self.layers.off()}


class ReferenceLayersDialog(QtWidgets.QDialog):
    """Layers of a reference: ticked = shown (the window follows every tick)."""

    def __init__(self, ref, dwg: Drawing, apply: Callable[[List[str]], None], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Layers - {ref.name} ({ref.source_label()})")
        self._apply = apply
        self._was = list(ref.layers_off)
        lay = QtWidgets.QVBoxLayout(self)
        self.layers = LayerList(dwg, ref.layers_off)
        self.layers.changed.connect(lambda: apply(self.layers.off()))
        lay.addWidget(self.layers)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(340, 420)

    def reject(self):
        if self.layers.off() != sorted(self._was):
            self._apply(self._was)
        super().reject()


COLOR_CHOICES = [("gray", "Grey (faded)"), ("layers", "The drawing's colours"), ("custom", "One colour...")]


class ManageReferencesDialog(QtWidgets.QDialog):
    """References > Manage References: the list, and for the one selected how it is
    shown, snapped to and kept.  Every change is applied at once (and can be undone)."""

    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.setWindowTitle("References")
        self.window_ = window
        self.doc = window.doc
        self._fill = False
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "File", "Shown on", "Kept"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._show)
        left.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        self.b_add = QtWidgets.QPushButton("Place DXF Drawing...")
        self.b_del = QtWidgets.QPushButton("Delete")
        self.b_reload = QtWidgets.QPushButton("Reload")
        self.b_relink = QtWidgets.QPushButton("Choose File...")
        self.b_zoom = QtWidgets.QPushButton("Zoom To")
        for b, fn in ((self.b_add, self._add), (self.b_del, self._delete), (self.b_reload, self._reload),
                      (self.b_relink, self._relink), (self.b_zoom, self._zoom)):
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        left.addLayout(row)
        lay.addLayout(left, 3)

        self.props = QtWidgets.QGroupBox("Selected reference")
        f = QtWidgets.QFormLayout(self.props)
        self.c_visible = QtWidgets.QCheckBox("Shown")
        self.c_snap = QtWidgets.QCheckBox("Snap to it")
        self.c_pinned = QtWidgets.QCheckBox("Pinned in place")
        self.c_top = QtWidgets.QCheckBox("Drawn over the model (not hidden by extruded sections)")
        self.color = QtWidgets.QComboBox()
        for key, label in COLOR_CHOICES:
            self.color.addItem(label, key)
        self.strength = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.strength.setRange(10, 100)
        self.strength.setSingleStep(5)
        self.unit = QtWidgets.QComboBox()
        for u in UNIT_NAMES:
            self.unit.addItem(UNIT_LABELS[u], u)
        self.scale = QtWidgets.QLineEdit()
        self.scale.setMaximumWidth(110)
        self.size = QtWidgets.QLabel()
        self.c_link = QtWidgets.QCheckBox("Linked to the file")
        self.c_embed = QtWidgets.QCheckBox("Embedded (a copy in the model file)")
        self.b_layers = QtWidgets.QPushButton("Layers...")
        self.b_stories = QtWidgets.QPushButton("Stories...")
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        for w in (self.c_visible, self.c_snap, self.c_pinned, self.c_top):
            f.addRow(w)
        f.addRow("Colour", self.color)
        f.addRow("Strength", self.strength)
        f.addRow("Drawing unit", self.unit)
        f.addRow("Extra scale", self.scale)
        f.addRow("Size", self.size)
        f.addRow(self.c_link)
        f.addRow(self.c_embed)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.b_layers)
        h.addWidget(self.b_stories)
        h.addStretch(1)
        f.addRow(h)
        f.addRow(self.status)
        lay.addWidget(self.props, 2)

        self.c_visible.clicked.connect(lambda on: self._set(visible=bool(on)))
        self.c_snap.clicked.connect(lambda on: self._set(snap=bool(on)))
        self.c_pinned.clicked.connect(lambda on: self._set(pinned=bool(on)))
        self.c_top.clicked.connect(lambda on: self._set(on_top=bool(on)))
        self.color.activated.connect(self._color)
        self.strength.sliderReleased.connect(lambda: self._set(strength=self.strength.value() / 100.0))
        self.strength.actionTriggered.connect(lambda _a: QtCore.QTimer.singleShot(0, self, self._strength_key))
        self.unit.activated.connect(lambda _i: self._set(unit=self.unit.currentData()))
        self.scale.editingFinished.connect(self._scale)
        self.c_link.clicked.connect(self._storage)
        self.c_embed.clicked.connect(self._storage)
        self.b_layers.clicked.connect(self._layers)
        self.b_stories.clicked.connect(self._stories)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        close.accepted.connect(self.accept)
        left.addWidget(close)
        self.resize(900, 460)
        self.refresh()

    # ---- the list
    def current(self) -> Optional[str]:
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        return it.text() if it is not None else None

    def refresh(self, select: Optional[str] = None):
        m = self.doc.model
        keep = select or self.current()
        self._fill = True
        self.table.setRowCount(0)
        for ref in cadref.references_of(m).values():
            r = self.table.rowCount()
            self.table.insertRow(r)
            if ref.view == "plan":
                on = "Plan: " + (", ".join(ref.stories) if ref.stories else f"Z = {ref.level * _du(m)[0]:g}")
            else:
                on = f"Elevation {'X-Z at Y' if ref.view == 'xz' else 'Y-Z at X'} = {ref.level * _du(m)[0]:g}"
            for c, text in enumerate((ref.name, ref.source_label(), on,
                                      cadref.describe(ref, m, getattr(self.doc, "path", None)))):
                it = QtWidgets.QTableWidgetItem(text)
                if c == 1:
                    it.setToolTip(ref.path or "(embedded)")
                self.table.setItem(r, c, it)
            if ref.name == keep:
                self.table.selectRow(r)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        if self.table.rowCount() and self.table.currentRow() < 0:
            self.table.selectRow(0)
        self._fill = False
        self._show()

    def _ref(self):
        return cadref.references_of(self.doc.model).get(self.current() or "")

    def _show(self):
        if self._fill:
            return
        ref = self._ref()
        on = ref is not None
        self.props.setEnabled(on)
        for b in (self.b_del, self.b_reload, self.b_relink, self.b_zoom):
            b.setEnabled(on)
        if not on:
            self.status.setText("")
            self.size.setText("")
            return
        m = self.doc.model
        self.c_visible.setChecked(ref.visible)
        self.c_snap.setChecked(ref.snap)
        self.c_pinned.setChecked(ref.pinned)
        self.c_top.setChecked(ref.on_top)
        mode = ref.color if ref.color in ("gray", "layers") else "custom"
        self.color.setCurrentIndex([k for k, _ in COLOR_CHOICES].index(mode))
        self.strength.setValue(int(round(ref.strength * 100)))
        self.unit.setCurrentIndex(UNIT_NAMES.index(ref.unit) if ref.unit in UNIT_NAMES else 2)
        self.scale.setText(f"{ref.scale:.6g}")
        self.c_link.setChecked(bool(ref.path or ref.rel_path))
        self.c_embed.setChecked(bool(ref.embedded))
        self.b_stories.setEnabled(ref.view == "plan" and bool(m.stories))
        dwg, why = cadref.drawing_of(ref, m, getattr(self.doc, "path", None))
        self.b_layers.setEnabled(dwg is not None)
        if dwg is None:
            self.size.setText("-")
            self.status.setText(f"<b>Not shown:</b> {why}")
        else:
            w, h = dwg.size()
            f = ref.factor("m")
            self.size.setText(f"{w * f:,.2f} x {h * f:,.2f} m")
            note = "The file was changed since it was read - Reload." if cadref.changed_on_disk(
                ref, m, getattr(self.doc, "path", None)) else ""
            self.status.setText(note)

    # ---- changes
    def _try(self, fn) -> bool:
        try:
            fn()
            return True
        except ModelError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), exc.message if hasattr(exc, "message") else str(exc))
        except (OSError, DxfError) as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
        return False

    def _set(self, **changes):
        name = self.current()
        if name:
            self._try(lambda: cadref.change_reference(self.doc, name, "Reference", **changes))
            self.refresh(name)

    def _strength_key(self):
        if not self.strength.isSliderDown():
            ref = self._ref()
            if ref is not None and abs(ref.strength - self.strength.value() / 100.0) > 1e-9:
                self._set(strength=self.strength.value() / 100.0)

    def _color(self, _i=None):
        key = self.color.currentData()
        ref = self._ref()
        if ref is None:
            return
        if key == "custom":
            start = QtGui.QColor(ref.color) if ref.color.startswith("#") else QtGui.QColor("#d9534f")
            c = QtWidgets.QColorDialog.getColor(start, self, "Colour of the reference")
            if not c.isValid():
                self._show()
                return
            key = c.name()
        self._set(color=key)

    def _scale(self):
        ref = self._ref()
        if ref is None:
            return
        try:
            v = float(self.scale.text().replace(",", ""))
            if not v > 0:
                raise ValueError
        except ValueError:
            self.scale.setText(f"{ref.scale:.6g}")
            return
        if abs(v - ref.scale) > 1e-12 * max(1.0, abs(v)):
            self._set(scale=v)

    def _storage(self):
        name = self.current()
        if name:
            self._try(lambda: cadref.set_storage(self.doc, name, self.c_link.isChecked(), self.c_embed.isChecked(),
                                                 getattr(self.doc, "path", None)))
            self.refresh(name)

    def _layers(self):
        self.window_.reference_layers(self.current(), parent=self)
        self.refresh()

    def _stories(self):
        ref = self._ref()
        if ref is None:
            return
        from .strip_dialog import _story_box
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Stories - {ref.name}")
        v = QtWidgets.QVBoxLayout(dlg)
        box, lst = _story_box([s.name for s in self.doc.model.stories_bottom_up()], list(ref.stories),
                              "Plans that show the drawing")
        v.addWidget(box)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() == QtWidgets.QDialog.Accepted and lst.ticked():
            self._set(stories=lst.ticked())

    def _add(self):
        name = self.window_.reference_place(parent=self, start_align=False)
        self.refresh(name)

    def _delete(self):
        name = self.current()
        if not name:
            return
        box = QtWidgets.QMessageBox
        if box.question(self, self.windowTitle(), f"Delete reference {name}?\n\n(The drawing file itself is not "
                        "touched.)", box.Yes | box.No, box.Yes) == box.Yes:
            cadref.delete_references(self.doc, [name])
            self.refresh()

    def _reload(self):
        ref = self._ref()
        if ref is not None:
            cadref.reload(ref, self.doc.model, getattr(self.doc, "path", None))
            self.window_.references_redraw()
            self.refresh()

    def _relink(self):
        name = self.current()
        ref = self._ref()
        if ref is None:
            return
        start = os.path.dirname(ref.path) if ref.path else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose the Drawing File", start, FILE_FILTER)
        if path and self._try(lambda: cadref.relink(self.doc, name, path)):
            self.refresh(name)

    def _zoom(self):
        self.window_.reference_zoom(self.current())


class ReferenceCommandsMixin:
    """The References menu of the main window."""

    def reference_commands(self) -> Dict[str, Callable]:
        return {
            "ref.place": self.reference_place,
            "ref.manage": self.reference_manage,
            "ref.align": lambda: self.reference_tool("ref_align"),
            "ref.move": lambda: self.reference_tool("ref_move"),
            "ref.rotate": lambda: self.reference_tool("ref_rotate"),
            "ref.scale": lambda: self.reference_tool("ref_scale"),
            "ref.pin": self.reference_pin,
            "ref.layers": self.reference_layers,
            "ref.zoom": self.reference_zoom,
            "ref.reload": self.reference_reload,
        }

    def reference_actions(self) -> Dict[str, QtGui.QAction]:
        """The two switches (made with the other actions, before the menus): the window's
        show / hide, and the snap - the "CAD references" snap of the snap list."""
        self.act_ref_show = self._make("Show in This Window", self.reference_show_window, "Ctrl+Alt+R", checkable=True,
                                       checked=True, tip="Show / hide the references in the active window (Ctrl+Alt+R)")
        return {"ref.show_window": self.act_ref_show, "ref.snap": self.snap_actions["cad"]}

    # ---- helpers
    def _ref_window(self):
        return self.viewport

    def _refs_here(self) -> List[str]:
        vp = self.viewport
        return [r.name for r, _, _ in vp.shown_references()] if hasattr(vp, "shown_references") else []

    def _pick_reference(self, title: str, parent=None) -> Optional[str]:
        """The reference a command works on: the only one, the only one in the active
        window, else asked."""
        refs = list(cadref.references_of(self.doc.model))
        if not refs:
            self._info("No reference yet - References > Place DXF Drawing")
            return None
        if len(refs) == 1:
            return refs[0]
        here = self._refs_here()
        if len(here) == 1:
            return here[0]
        m = self.doc.model
        labels = [f"{n}  -  {m.references[n].source_label()}" for n in refs]
        text, ok = QtWidgets.QInputDialog.getItem(parent or self, title, "Reference", labels, 0, False)
        return refs[labels.index(text)] if ok and text in labels else None

    def references_redraw(self):
        for vp in self.views.viewports:
            vp._ref_sig = None
            vp.invalidate("references")

    def sync_reference_actions(self):
        if hasattr(self, "act_ref_show"):
            self.act_ref_show.setChecked(not getattr(self.viewport, "references_hidden", False))

    # ---- commands
    def reference_place(self, parent=None, start_align: bool = True) -> Optional[str]:
        settings = getattr(self, "qsettings", None)
        start = settings.value("references/folder", "") if settings is not None else ""
        if not start and getattr(self.doc, "path", None):
            start = os.path.dirname(self.doc.path)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(parent or self, "Place DXF Drawing", start or "", FILE_FILTER)
        if not path:
            return None
        if settings is not None:
            settings.setValue("references/folder", os.path.dirname(path))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            dwg = read_dxf(path)
        except DxfError as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(parent or self, "Place DXF Drawing", str(exc))
            return None
        finally:
            if QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()
        if not len(dwg.seg):
            QtWidgets.QMessageBox.warning(parent or self, "Place DXF Drawing",
                                          "The drawing has no lines in its model space to show.")
            return None
        return self._place_drawing(path, dwg, parent, start_align)

    def _place_drawing(self, path: str, dwg: Drawing, parent=None, start_align: bool = True) -> Optional[str]:
        m = self.doc.model
        vp = self.viewport
        kind = vp.config.kind
        pl = vp.plane() if kind != "3d" else None
        story = pl.story if kind == "plan" else None
        if kind == "3d" or (kind == "plan" and story is None and m.stories):
            names = [s.name for s in m.stories_bottom_up()]
            story = names[1] if len(names) > 1 else (names[0] if names else None)
        view = kind if kind in ("xz", "yz") else "plan"
        dlg = PlaceReferenceDialog(m, path, dwg, view, story, pl.value if pl is not None and view != "plan" else 0.0,
                                   self._drawing_near_model(dwg), parent or self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        v = dlg.values()
        level = v["level"]
        if v["view"] == "plan" and not v["stories"]:
            level = pl.value if (pl is not None and kind == "plan") else 0.0
        # a window that shows it: the active one when it is a plan / elevation, else the first plan
        target = vp if kind == v["view"] else next((w for w in self.views.viewports if w.config.kind == v["view"]), None)
        if target is not None:
            if v["view"] == "plan":
                if v["stories"] and target.config.story not in v["stories"]:
                    target.set_view("plan", story=v["stories"][0])
            else:
                target.set_view(v["view"], ordinate=level)
            if getattr(target, "references_hidden", False):
                target.set_references_hidden(False)
        base, origin = (0.0, 0.0), (0.0, 0.0)
        if not v["own"]:
            e = dwg.extents
            base = (0.5 * (e[0] + e[2]), 0.5 * (e[1] + e[3]))
            origin = self._window_centre(target) if target is not None else (0.0, 0.0)
        try:
            name = cadref.add_reference(self.doc, path, v["view"], v["stories"], level, v["unit"], v["embed"],
                                        v["link"], origin, base, v["layers_off"])
        except ModelError as exc:
            QtWidgets.QMessageBox.warning(parent or self, "Place DXF Drawing",
                                          exc.message if hasattr(exc, "message") else str(exc))
            return None
        if target is not None:
            self.views.set_active(target)
            self.reference_zoom(name, window=target, only_if_out=v["own"])
        self._info(f"Reference {name}: {os.path.basename(path)} placed ({size_text(dwg, v['unit'])})"
                   + ("" if target is not None else " - open a plan / elevation window to see it"))
        if v["align"] and start_align and target is not None:
            self.reference_tool("ref_align", name)
        elif not v["align"]:
            cadref.change_reference(self.doc, name, "Pin Reference", pinned=True)
        return name

    def _drawing_near_model(self, dwg: Drawing) -> bool:
        """Do the drawing's own coordinates (in its guessed unit) fall on the model's
        grids / objects?  Then "own coordinates" is the position offered."""
        m = self.doc.model
        e = dwg.extents
        if e is None:
            return True
        f = UNIT_TO_M.get(dwg.guess_unit(), 1.0) / UNIT_TO_M.get(m.units.length, 1.0)
        xs = [p.x for p in m.points.values()]
        ys = [p.y for p in m.points.values()]
        try:
            from .tools import grid_ordinates
            gx, gy = grid_ordinates(m)
            xs += list(gx)
            ys += list(gy)
        except Exception:
            pass
        if not xs or not ys:
            return max(abs(v) * f for v in e) < 1.0e4  # an empty model: near its origin
        w = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        return not (e[2] * f < min(xs) - w or e[0] * f > max(xs) + w or e[3] * f < min(ys) - w or e[1] * f > max(ys) + w)

    def _window_centre(self, vp) -> Tuple[float, float]:
        cam = vp.renderer.GetActiveCamera()
        return tuple(float(c) for c in vp.plane().to2(tuple(cam.GetFocalPoint())))

    def reference_zoom(self, name: Optional[str] = None, window=None, only_if_out: bool = False):
        name = name or self._pick_reference("Zoom to Reference")
        m = self.doc.model
        ref = cadref.references_of(m).get(name or "")
        if ref is None:
            return
        dwg, why = cadref.drawing_of(ref, m, getattr(self.doc, "path", None))
        if dwg is None:
            self._info(f"Reference {name} cannot be shown: {why}")
            return
        vp = window
        if vp is None:
            vp = next((w for w in [self.viewport] + list(self.views.viewports)
                       if any(r.name == name for r, _, _ in w.shown_references())), None)
        if vp is None:  # no window shows it: the active one goes to its plane
            vp = self.viewport
            if ref.view == "plan":
                vp.set_view("plan", story=ref.stories[0]) if ref.stories else vp.set_view("plan", z=ref.level)
            else:
                vp.set_view(ref.view, ordinate=ref.level)
        box = cadref.outline(ref, dwg, m.units.length)
        if box is None:
            return
        pl = vp.plane()
        X = cadref.plane_to_world(pl.kind, pl.value, box)
        if only_if_out:
            # (placed on its own coordinates: only when none of it is in sight)
            w, h = vp.vtk_widget.width() * vp._dpr(), vp.vtk_widget.height() * vp._dpr()
            D = [vp._display(tuple(x)) for x in X]
            if not (max(d[0] for d in D) < 0 or min(d[0] for d in D) > w or max(d[1] for d in D) < 0
                    or min(d[1] for d in D) > h):
                return
        vp.zoom_to(X, margin=0.55)
        vp.render()

    def reference_tool(self, tool: str, name: Optional[str] = None):
        if not cadref.references_of(self.doc.model):
            self._info("No reference yet - References > Place DXF Drawing")
            return
        if self.viewport.config.kind == "3d":
            self._info("References are put in place in plan and elevation windows - click in the window that shows it")
        self.state.reference_target = name if name else None
        self.state.set_tool(tool)

    def reference_pin(self):
        name = self._pick_reference("Pin / Unpin Reference")
        if not name:
            return
        ref = self.doc.model.references[name]
        cadref.change_reference(self.doc, name, "Pin Reference" if not ref.pinned else "Unpin Reference",
                                pinned=not ref.pinned)
        self._info(f"Reference {name} " + ("pinned in place" if not ref.pinned else "unpinned - it can be moved"))

    def reference_layers(self, name: Optional[str] = None, parent=None):
        name = name or self._pick_reference("Layers of a Reference", parent)
        m = self.doc.model
        ref = cadref.references_of(m).get(name or "")
        if ref is None:
            return
        dwg, why = cadref.drawing_of(ref, m, getattr(self.doc, "path", None))
        if dwg is None:
            self._info(f"Reference {name} cannot be shown: {why}")
            return
        was = list(ref.layers_off)

        def live(off):  # (while the form is open the windows follow; one undo step at OK)
            m.references[name].layers_off = list(off)
            self.references_redraw()

        dlg = ReferenceLayersDialog(ref, dwg, live, parent or self)
        ok = dlg.exec() == QtWidgets.QDialog.Accepted
        new = list(m.references[name].layers_off)
        m.references[name].layers_off = was
        if ok and sorted(new) != sorted(was):
            cadref.change_reference(self.doc, name, "Reference Layers", layers_off=new)
        else:
            self.references_redraw()

    def reference_manage(self):
        dlg = ManageReferencesDialog(self)
        dlg.exec()
        self.sync_reference_actions()

    def reference_reload(self):
        m = self.doc.model
        if not cadref.references_of(m):
            self._info("No reference yet - References > Place DXF Drawing")
            return
        cadref.reload()
        self.references_redraw()
        missing = [r.name for r in m.references.values()
                   if cadref.source_of(r, m, getattr(self.doc, "path", None))[0] == "missing"]
        self._info("Linked drawings read again" + (f" - not found: {', '.join(missing)}" if missing else ""))

    def reference_show_window(self, on: bool = True):
        self.viewport.set_references_hidden(not on)
        n = len(cadref.references_of(self.doc.model))
        self._info(("References shown" if on else "References hidden") + " in this window" +
                   ("" if n else " (there is none yet)"))
