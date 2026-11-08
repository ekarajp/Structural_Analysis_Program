"""Options > Graphics Colors: theme, colour-by option and every display colour
(per theme), plus the display colours of sections and materials."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..model import Model
from . import theme as th
from .theme import (COLOR_BY, COLOR_BY_LABELS, COLOR_ITEMS, OPACITY_ITEMS, OPTION_ITEMS, SIZE_ITEMS, THEME_LABELS,
                    THEMES, ColorSettings)

SIZES_TAB = 2  # index of the "Sizes" tab (Options > Graphics Sizes...)


def _qcolor(rgb) -> QtGui.QColor:
    return QtGui.QColor.fromRgbF(*rgb)


class ColorButton(QtWidgets.QToolButton):
    """A colour swatch that opens QColorDialog."""

    color_changed = QtCore.Signal(tuple)

    def __init__(self, rgb=(0.5, 0.5, 0.5), title: str = "Colour", parent=None):
        super().__init__(parent)
        self._rgb = tuple(rgb)
        self._title = title
        self._auto = False
        self.setFixedSize(54, 20)
        self.clicked.connect(self._pick)
        self._paint()

    def rgb(self):
        return self._rgb

    def set_rgb(self, rgb, auto: bool = False):
        self._rgb = tuple(rgb)
        self._auto = auto
        self._paint()

    def _paint(self):
        hexc = th.to_hex(self._rgb)
        border = "dashed" if self._auto else "solid"
        self.setStyleSheet(f"QToolButton {{ background: {hexc}; border: 1px {border} #808080; border-radius: 3px; }}")
        self.setToolTip(hexc + ("  (automatic)" if self._auto else ""))

    def _pick(self):
        c = QtWidgets.QColorDialog.getColor(_qcolor(self._rgb), self, self._title)
        if c.isValid():
            self.set_rgb((c.redF(), c.greenF(), c.blueF()))
            self.color_changed.emit(self._rgb)


def _default_button(tip: str) -> QtWidgets.QToolButton:
    """Small per-row "back to the default" button (enabled only when the row is changed)."""
    b = QtWidgets.QToolButton()
    b.setText("\u21ba Default")
    b.setToolTip(tip)
    b.setAutoRaise(True)
    return b


class GraphicsColorsDialog(QtWidgets.QDialog):
    """Edits a copy of the colour settings; ``applied`` fires on Apply / OK."""

    applied = QtCore.Signal(object)  # ColorSettings

    def __init__(self, settings: ColorSettings, model: Optional[Model] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Graphics Colors and Sizes")
        self.resize(560, 640)
        self.settings = settings.copy()
        self.model = model
        # (kind, name) -> hex or None (automatic); kind in frame | shell | material
        self.object_colors: Dict[Tuple[str, str], Optional[str]] = {}
        lay = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QFormLayout()
        self.theme_box = QtWidgets.QComboBox()
        for t in THEMES:
            self.theme_box.addItem(THEME_LABELS[t], t)
        self.theme_box.setCurrentIndex(THEMES.index(self.settings.theme))
        self.theme_box.currentIndexChanged.connect(self._theme_changed)
        self.by_box = QtWidgets.QComboBox()
        for b in COLOR_BY:
            self.by_box.addItem(COLOR_BY_LABELS[b], b)
        self.by_box.setCurrentIndex(COLOR_BY.index(self.settings.color_by))
        self.by_box.currentIndexChanged.connect(self._by_changed)
        top.addRow("Theme", self.theme_box)
        top.addRow("Color objects by", self.by_box)
        lay.addLayout(top)

        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.buttons: Dict[str, ColorButton] = {}
        self.custom_marks: Dict[str, QtWidgets.QLabel] = {}
        self.option_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.option_resets: Dict[str, QtWidgets.QToolButton] = {}
        for tab_name, groups in (("Display", ("Window", "Objects", "Interaction")), ("Output", ("Output",))):
            page = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(page)
            row = 0
            for group in groups:
                head = QtWidgets.QLabel(group)
                head.setStyleSheet("font-weight: 600; padding-top: 6px")
                grid.addWidget(head, row, 0, 1, 3)
                row += 1
                for key, label, g in COLOR_ITEMS:
                    if g != group:
                        continue
                    grid.addWidget(QtWidgets.QLabel("   " + label), row, 0)
                    b = ColorButton(title=label)
                    b.color_changed.connect(lambda rgb, k=key: self._set(k, rgb))
                    grid.addWidget(b, row, 1)
                    mark = _default_button("Back to the default colour of this theme")
                    mark.clicked.connect(lambda _=False, k=key: self._reset_color(k))
                    grid.addWidget(mark, row, 2)
                    self.buttons[key] = b
                    self.custom_marks[key] = mark
                    row += 1
            if tab_name == "Display":
                head = QtWidgets.QLabel("Transparency")
                head.setStyleSheet("font-weight: 600; padding-top: 6px")
                grid.addWidget(head, row, 0, 1, 3)
                row += 1
                self.opacity_spins: Dict[str, QtWidgets.QSpinBox] = {}
                self.opacity_resets: Dict[str, QtWidgets.QToolButton] = {}
                for key, label, _default in OPACITY_ITEMS:
                    grid.addWidget(QtWidgets.QLabel("   " + label), row, 0)
                    sp = QtWidgets.QSpinBox()
                    sp.setRange(0, 100)
                    sp.setSuffix(" %")
                    sp.setToolTip("0 % = invisible, 100 % = opaque")
                    sp.valueChanged.connect(lambda v, k=key: self._set_opacity(k, v))
                    grid.addWidget(sp, row, 1)
                    rb = _default_button("Back to the default transparency")
                    rb.clicked.connect(lambda _=False, k=key: self._reset_opacity(k))
                    grid.addWidget(rb, row, 2)
                    self.opacity_spins[key] = sp
                    self.opacity_resets[key] = rb
                    row += 1
            if tab_name == "Output":
                head = QtWidgets.QLabel("Diagrams")
                head.setStyleSheet("font-weight: 600; padding-top: 6px")
                grid.addWidget(head, row, 0, 1, 3)
                row += 1
                for key, label, _default in OPTION_ITEMS:
                    cb = QtWidgets.QCheckBox(label)
                    cb.toggled.connect(lambda on, k=key: self._set_option(k, on))
                    grid.addWidget(cb, row, 0, 1, 2)
                    rb = _default_button("Back to the default")
                    rb.clicked.connect(lambda _=False, k=key: self._reset_option(k))
                    grid.addWidget(rb, row, 2)
                    self.option_checks[key] = cb
                    self.option_resets[key] = rb
                    row += 1
            grid.setRowStretch(row, 1)
            grid.setColumnStretch(0, 1)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, tab_name)
        self.tabs.addTab(self._sizes_page(), "Sizes")
        if model is not None:
            self.tabs.addTab(self._objects_page(model), "Sections && Materials")

        row = QtWidgets.QHBoxLayout()
        self.reset_tab_button = QtWidgets.QPushButton("Defaults for this tab")
        self.reset_tab_button.setToolTip("Every item of the open tab back to its default "
                                         "(colours: of the theme chosen above)")
        self.reset_tab_button.clicked.connect(self._reset_tab)
        row.addWidget(self.reset_tab_button)
        self.reset_all_button = QtWidgets.QPushButton("All defaults (light + dark)")
        self.reset_all_button.setToolTip("All colours of both themes, transparencies and sizes back to the defaults")
        self.reset_all_button.clicked.connect(self._reset_all)
        row.addWidget(self.reset_all_button)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                                        | QtWidgets.QDialogButtonBox.Apply)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self._apply)
        lay.addWidget(bb)
        self._fill()

    # ------------------------------------------------------------ sizes
    def _sizes_page(self) -> QtWidgets.QWidget:
        """Symbol sizes, line weights and text sizes of the model windows."""
        page = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(page)
        note = QtWidgets.QLabel("Symbol sizes (x = times the standard size), line widths and text sizes "
                                "of the model windows. Apply to see the change.")
        note.setWordWrap(True)
        grid.addWidget(note, 0, 0, 1, 2)
        self.size_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.size_resets: Dict[str, QtWidgets.QToolButton] = {}
        for row, (key, label, _default, lo, hi, step, suffix) in enumerate(SIZE_ITEMS, start=1):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(2 if step < 1 else 0)
            sp.setSuffix(suffix)
            sp.valueChanged.connect(lambda v, k=key: self._set_size(k, v))
            grid.addWidget(sp, row, 1)
            rb = _default_button(f"Back to the default ({_default:g}{suffix.strip() and ' ' + suffix.strip()})")
            rb.clicked.connect(lambda _=False, k=key: self._reset_size(k))
            grid.addWidget(rb, row, 2)
            self.size_spins[key] = sp
            self.size_resets[key] = rb
        grid.setRowStretch(len(SIZE_ITEMS) + 1, 1)
        grid.setColumnStretch(0, 1)
        return page

    # ------------------------------------------------------------ objects
    def _objects_page(self, model: Model) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.addWidget(QtWidgets.QLabel("Used when objects are coloured by section or material. "
                                     "Dashed swatches are automatic colours."))
        self.obj_table = QtWidgets.QTableWidget(0, 4)
        self.obj_table.setHorizontalHeaderLabels(["Kind", "Name", "Colour", ""])
        self.obj_table.verticalHeader().setVisible(False)
        self.obj_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self._obj_buttons: Dict[Tuple[str, str], ColorButton] = {}
        rows = ([("frame", n, s.color) for n, s in model.frame_sections.items()]
                + [("shell", n, s.color) for n, s in model.shell_sections.items()]
                + [("material", n, m.color) for n, m in model.materials.items()])
        labels = {"frame": "Frame section", "shell": "Shell section", "material": "Material"}
        for k, (kind, name, explicit) in enumerate(rows):
            self.obj_table.insertRow(k)
            self.obj_table.setItem(k, 0, QtWidgets.QTableWidgetItem(labels[kind]))
            self.obj_table.setItem(k, 1, QtWidgets.QTableWidgetItem(name))
            b = ColorButton(title=name)
            b.color_changed.connect(lambda rgb, key=(kind, name): self._set_object(key, rgb))
            self.obj_table.setCellWidget(k, 2, b)
            auto = QtWidgets.QToolButton()
            auto.setText("Auto")
            auto.clicked.connect(lambda _=False, key=(kind, name): self._set_object(key, None))
            self.obj_table.setCellWidget(k, 3, auto)
            self._obj_buttons[(kind, name)] = b
            self._paint_object((kind, name), explicit)
        v.addWidget(self.obj_table, 1)
        return page

    def _paint_object(self, key, explicit: Optional[str]):
        b = self._obj_buttons[key]
        kind, name = key
        m = self.model
        if kind == "material":
            idx = list(m.materials).index(name)
        elif kind == "frame":
            idx = list(m.frame_sections).index(name)
        else:
            idx = list(m.shell_sections).index(name) + len(m.frame_sections)
        b.set_rgb(th.object_color(name, explicit, self.settings.theme, idx), auto=not explicit)

    def _explicit(self, key) -> Optional[str]:
        if key in self.object_colors:
            return self.object_colors[key]
        kind, name = key
        m = self.model
        obj = (m.frame_sections if kind == "frame" else m.shell_sections if kind == "shell" else m.materials).get(name)
        return getattr(obj, "color", None)

    def _set_object(self, key, rgb):
        self.object_colors[key] = None if rgb is None else th.to_hex(rgb)
        self._paint_object(key, self.object_colors[key])

    # ------------------------------------------------------------ events
    def _fill(self):
        cols = self.settings.colors()
        for key, b in self.buttons.items():
            b.set_rgb(cols[key])
            self.custom_marks[key].setEnabled(self.settings.is_custom(key))
        for key, sp in getattr(self, "opacity_spins", {}).items():
            sp.blockSignals(True)
            sp.setValue(int(round(self.settings.opacity(key) * 100)))
            sp.blockSignals(False)
            self.opacity_resets[key].setEnabled(self.settings.opacity_is_custom(key))
        for key, cb in self.option_checks.items():
            cb.blockSignals(True)
            cb.setChecked(self.settings.option(key))
            cb.blockSignals(False)
            self.option_resets[key].setEnabled(key in self.settings.options)
        for key, sp in getattr(self, "size_spins", {}).items():
            sp.blockSignals(True)
            sp.setValue(self.settings.size(key))
            sp.blockSignals(False)
            self.size_resets[key].setEnabled(self.settings.size_is_custom(key))
        if self.model is not None:
            for key in self._obj_buttons:
                self._paint_object(key, self._explicit(key))

    def _set(self, key, rgb):
        self.settings.set_color(key, rgb)
        self.custom_marks[key].setEnabled(self.settings.is_custom(key))

    def _set_opacity(self, key, percent):
        self.settings.set_opacity(key, percent / 100.0)
        self.opacity_resets[key].setEnabled(self.settings.opacity_is_custom(key))

    def _set_size(self, key, value):
        self.settings.set_size(key, value)
        self.size_resets[key].setEnabled(self.settings.size_is_custom(key))

    def _set_option(self, key, on):
        self.settings.set_option(key, on)
        self.option_resets[key].setEnabled(key in self.settings.options)

    def _reset_option(self, key):
        self.settings.options.pop(key, None)
        self._fill()

    def _reset_color(self, key):
        self.settings.reset_color(key)
        self._fill()

    def _reset_opacity(self, key):
        self.settings.reset_opacity(key)
        self._fill()

    def _reset_size(self, key):
        self.settings.reset_size(key)
        self._fill()

    def _reset_tab(self):
        """Defaults for the open tab only."""
        name = self.tabs.tabText(self.tabs.currentIndex()).replace("&&", "&")
        if name == "Display":
            self.settings.reset_colors([k for k, _, g in COLOR_ITEMS if g != "Output"])
            self.settings.opacities = {}
        elif name == "Output":
            self.settings.reset_colors([k for k, _, g in COLOR_ITEMS if g == "Output"])
            self.settings.options = {}
        elif name == "Sizes":
            self.settings.sizes = {}
        elif self.model is not None:  # sections & materials: automatic colours
            for key in self._obj_buttons:
                self.object_colors[key] = None
        self._fill()

    def _reset_all(self):
        self.settings.reset_all()
        if self.model is not None:
            for key in self._obj_buttons:
                self.object_colors[key] = None
        self._fill()

    def _theme_changed(self, idx):
        self.settings.theme = THEMES[idx]
        self._fill()

    def _by_changed(self, idx):
        self.settings.color_by = COLOR_BY[idx]

    def _reset(self):  # old name, kept for callers: everything of the active theme
        self.settings.reset()
        self._fill()

    def _apply(self):
        self.applied.emit(self.settings.copy())

    def _ok(self):
        self._apply()
        self.accept()

    # ------------------------------------------------------------ model
    def apply_object_colors(self, model: Model):
        """Write the edited section / material colours into ``model``."""
        for (kind, name), hexc in self.object_colors.items():
            table = model.frame_sections if kind == "frame" else model.shell_sections if kind == "shell" else model.materials
            if name in table:
                table[name].color = hexc
