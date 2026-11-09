"""Model window area: 1-4 tiled model windows (ETABS "Window > Tile").

The four window slots sit in a fixed splitter grid (left column: slots 0, 2;
right column: slots 1, 3) and a layout just shows a subset, so VTK widgets
are never re-parented.  Default: two windows, plan on the left and 3D on the
right.  The layout and what each window shows are kept in QSettings.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from .document import Document
from .state import ViewState
from .display import DisplayOptions
from .viewport import ModelViewport, ViewConfig

LAYOUTS: Dict[str, List[int]] = {
    "1": [0],
    "2": [0, 1],     # side by side
    "2v": [0, 2],    # stacked
    "3": [0, 1, 3],  # one left, two right
    "4": [0, 1, 2, 3],
}
LAYOUT_LABELS = {"1": "One window", "2": "Two windows (side by side)", "2v": "Two windows (stacked)",
                 "3": "Three windows", "4": "Four windows"}
DEFAULT_CONFIGS = [ViewConfig("plan"), ViewConfig("3d"), ViewConfig("xz"), ViewConfig("yz")]
DEFAULT_LAYOUT = "2"


class ViewportArea(QtWidgets.QWidget):
    active_changed = QtCore.Signal(object)
    layout_changed = QtCore.Signal(str)
    display_applied = QtCore.Signal()  # display options of some window changed

    def __init__(self, document: Document, state: ViewState, settings: Optional[QtCore.QSettings] = None,
                 parent=None):
        super().__init__(parent)
        self.doc = document
        self.state = state
        self.qsettings = settings
        layout_name, configs = self._load()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.outer = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.columns = [QtWidgets.QSplitter(QtCore.Qt.Vertical), QtWidgets.QSplitter(QtCore.Qt.Vertical)]
        for c in self.columns:
            c.setChildrenCollapsible(False)
            self.outer.addWidget(c)
        self.outer.setChildrenCollapsible(False)
        lay.addWidget(self.outer)
        self.viewports: List[ModelViewport] = []
        for slot in range(4):
            vp = ModelViewport(document, state, configs[slot])
            vp.activated.connect(self.set_active)
            vp.view_changed.connect(lambda _vp: self._update_highlight())
            vp.display_requested.connect(self.edit_display)
            self.columns[slot % 2].addWidget(vp)  # slot 0,2 left; 1,3 right
            self.viewports.append(vp)
        self.layout_name = ""
        self.active: ModelViewport = self.viewports[0]
        self.set_layout(layout_name)
        self.set_active(self.viewports[0])

    # ----------------------------------------------------------- settings
    def _load(self):
        layout_name, configs = DEFAULT_LAYOUT, [ViewConfig(**c.to_dict()) for c in DEFAULT_CONFIGS]
        if self.qsettings is not None:
            name = self.qsettings.value("windows/layout", DEFAULT_LAYOUT)
            if name in LAYOUTS:
                layout_name = name
            raw = self.qsettings.value("windows/configs", "")
            try:
                data = json.loads(raw) if raw else []
                for k, d in enumerate(data[:4]):
                    configs[k] = ViewConfig.from_dict(d)
            except (ValueError, TypeError):
                pass
        return layout_name, configs

    def save(self):
        if self.qsettings is None:
            return
        self.qsettings.setValue("windows/layout", self.layout_name)
        self.qsettings.setValue("windows/configs", json.dumps([vp.config.to_dict() for vp in self.viewports]))

    # ------------------------------------------------------------- layout
    def visible_viewports(self) -> List[ModelViewport]:
        return [self.viewports[i] for i in LAYOUTS[self.layout_name]]

    def set_layout(self, name: str):
        if name not in LAYOUTS:
            raise ValueError(f"unknown layout {name!r}")
        self.layout_name = name
        shown = set(LAYOUTS[name])
        for slot, vp in enumerate(self.viewports):
            vp.setVisible(slot in shown)
        for col, slots in ((0, (0, 2)), (1, (1, 3))):
            self.columns[col].setVisible(any(s in shown for s in slots))
        # equal sizes
        self.outer.setSizes([1000, 1000])
        for c in self.columns:
            c.setSizes([1000, 1000])
        if self.active not in self.visible_viewports():
            self.set_active(self.visible_viewports()[0])
        for vp in self.visible_viewports():
            vp.render()
        self.save()
        self.layout_changed.emit(name)

    def set_active(self, vp: ModelViewport):
        self.active = vp
        for v in self.viewports:
            v.set_active(v is vp)
        if vp.config.kind != "3d":
            self.state.tool.set_plane(vp.plane())
            self._last_2d = vp
        self.active_changed.emit(vp)
        self._update_highlight()

    # ------------------------------------------------------------ display
    def plan_story(self) -> Optional[str]:
        """Story shown by the plan window the user works in: the active window if it
        is a plan, else the first visible plan window."""
        cands = [self.active] + [v for v in self.visible_viewports() if v is not self.active]
        for v in cands:
            if v.isVisible() or v is self.active:
                if v.config.kind == "plan" and v.config.story:
                    return v.config.story
        return None

    def highlight_target(self):
        """(kind, value, label) of the plane the 3D windows shade: the active plan /
        elevation window, else the 2-D window used last, else the first plan shown."""
        cands = [self.active, getattr(self, "_last_2d", None)]
        cands += [v for v in self.visible_viewports() if v.config.kind == "plan"]
        for v in cands:
            if v is None or v not in self.viewports or v.config.kind == "3d":
                continue
            p = v.plane()
            if v.config.kind == "plan":
                label = p.story or f"Z = {p.value:g}"
            else:
                axis = "Y" if v.config.kind == "xz" else "X"
                label = f"{v.config.grid}  ({axis} = {p.value:g})" if v.config.grid else f"{axis} = {p.value:g}"
            return (v.config.kind, float(p.value), label)
        return None

    def _update_highlight(self):
        spec = self.highlight_target()
        for v in self.viewports:
            if v.config.kind == "3d":
                v.set_highlight_plane(spec)
        self.save()

    def edit_display(self, vp: Optional[ModelViewport] = None):
        """Set Display Options of ``vp`` (default: the active window)."""
        from .display_dialog import DisplayOptionsDialog
        vp = vp or self.active
        dlg = DisplayOptionsDialog(vp.display_options, vp.title.text(), self)
        if dlg.exec():
            self.apply_display(dlg.result_options(), vp, dlg.apply_all.isChecked())

    def apply_display(self, options: DisplayOptions, vp: Optional[ModelViewport] = None, all_windows: bool = False):
        for v in (self.viewports if all_windows else [vp or self.active]):
            v.set_display_options(options)
        self._update_highlight()
        self.save()
        self.display_applied.emit()

    def set_option_all(self, name: str, value):
        """One display option (e.g. ``extrude``) in every window."""
        for v in self.viewports:
            o = v.display_options
            setattr(o, name, value)
            v.set_display_options(o)
        self._update_highlight()
        self.save()
        self.display_applied.emit()

    # ----------------------------------------------------------- commands
    def configure(self, slot: int, config: ViewConfig):
        self.viewports[slot].set_config(config)
        self.save()

    def render_all(self):
        for vp in self.visible_viewports():
            vp.render()

    def finalize(self):
        self.save()
        for vp in self.viewports:
            vp.finalize()


class WindowsDialog(QtWidgets.QDialog):
    """Window > Arrange: number of windows and what each one shows."""

    def __init__(self, area: ViewportArea, parent=None):
        super().__init__(parent)
        self.area = area
        self.setWindowTitle("Model Windows")
        f = QtWidgets.QFormLayout(self)
        self.layout_box = QtWidgets.QComboBox()
        for key, label in LAYOUT_LABELS.items():
            self.layout_box.addItem(label, key)
        self.layout_box.setCurrentIndex(list(LAYOUT_LABELS).index(area.layout_name))
        f.addRow("Layout", self.layout_box)
        self.rows = []
        m = area.doc.model
        stories = [s.name for s in m.stories_bottom_up()][::-1]
        for slot, vp in enumerate(area.viewports):
            kind = QtWidgets.QComboBox()
            for key, label in (("plan", "Plan"), ("3d", "3D"), ("xz", "Elevation X-Z"), ("yz", "Elevation Y-Z")):
                kind.addItem(label, key)
            kind.setCurrentIndex(["plan", "3d", "xz", "yz"].index(vp.config.kind))
            where = QtWidgets.QComboBox()
            results = QtWidgets.QCheckBox("results")
            results.setChecked(vp.config.results)

            def fill(_=None, kind=kind, where=where, vp=vp):
                where.clear()
                k = kind.currentData()
                if k == "plan":
                    where.addItems(stories)
                    if vp.config.story in stories:
                        where.setCurrentText(vp.config.story)
                elif k == "xz":
                    where.addItems([g.name for g in m.grid_lines if g.direction == "Y"])
                elif k == "yz":
                    where.addItems([g.name for g in m.grid_lines if g.direction == "X"])
                if k in ("xz", "yz") and vp.config.grid:
                    where.setCurrentText(vp.config.grid)
                where.setEnabled(k != "3d")

            kind.currentIndexChanged.connect(fill)
            fill()
            row = QtWidgets.QHBoxLayout()
            row.addWidget(kind)
            row.addWidget(where, 1)
            row.addWidget(results)
            f.addRow(f"Window {slot + 1}", row)
            self.rows.append((kind, where, results))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def accept(self):
        for slot, (kind, where, results) in enumerate(self.rows):
            k = kind.currentData()
            cfg = ViewConfig(k, results=results.isChecked())
            if k == "plan":
                cfg.story = where.currentText() or None
            elif k in ("xz", "yz"):
                cfg.grid = where.currentText() or None
            self.area.configure(slot, cfg)
        self.area.set_layout(self.layout_box.currentData())
        super().accept()
