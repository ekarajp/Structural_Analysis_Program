"""Model window area: 1-9 model windows the user arranges freely (ETABS "Window > Tile").

Every model window is a dock in an inner dock area: drag its title bar onto the
side of another window to split there, onto its middle to stack them as tabs,
drag a divider to resize, double-click the title (or its float button) to take it
out as a window of its own (another screen), close it with its x.  The quick
layouts (1, 2 side by side, 2 stacked, 3, 4 ... 9) arrange the first windows in a
grid.  The arrangement, the number of windows and what each one shows are kept in
QSettings.  A window is never destroyed while the program runs (VTK widgets are
not re-created); a closed window is only hidden.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from .document import Document
from .state import ViewState
from .display import DisplayOptions
from .viewport import ModelViewport, ViewConfig

MAX_WINDOWS = 9
# quick layouts: number of windows (and "2v": two stacked)
LAYOUTS: Dict[str, int] = {"1": 1, "2": 2, "2v": 2, "3": 3, "4": 4, "6": 6, "9": 9}
LAYOUT_LABELS = {"1": "One window", "2": "Two windows (side by side)", "2v": "Two windows (stacked)",
                 "3": "Three windows", "4": "Four windows", "6": "Six windows", "9": "Nine windows"}
DEFAULT_CONFIGS = [ViewConfig("plan"), ViewConfig("3d"), ViewConfig("xz"), ViewConfig("yz")]
DEFAULT_LAYOUT = "2"
STATE_VERSION = 1


def _default_config(k: int) -> ViewConfig:
    return ViewConfig.from_dict(DEFAULT_CONFIGS[k % len(DEFAULT_CONFIGS)].to_dict())


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
        layout_name, configs, dock_state = self._load()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        # the inner dock area: a QMainWindow used as a widget, docks only (no centre)
        self.host = QtWidgets.QMainWindow()
        self.host.setWindowFlags(QtCore.Qt.Widget)
        self.host.setDockNestingEnabled(True)
        self.host.setDockOptions(QtWidgets.QMainWindow.AllowNestedDocks | QtWidgets.QMainWindow.AllowTabbedDocks
                                 | QtWidgets.QMainWindow.AnimatedDocks | QtWidgets.QMainWindow.GroupedDragging)
        self.host.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)
        self.host.setStyleSheet(
            "QMainWindow::separator { width: 6px; height: 6px; background: palette(mid); }"
            "QMainWindow::separator:hover { background: palette(highlight); }"
            "QDockWidget { font-weight: 600; }"
            "QDockWidget::title { padding: 3px 6px; background: palette(button); }")
        lay.addWidget(self.host)
        self.viewports: List[ModelViewport] = []
        self.docks: List[QtWidgets.QDockWidget] = []
        self.layout_name = ""
        self.active: Optional[ModelViewport] = None
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self.save)
        for cfg in configs:
            self._new_window(cfg)
        self.active = self.viewports[0]
        restored = False
        if dock_state is not None:
            open_n = dock_state.get("open")
            try:
                restored = bool(self.host.restoreState(QtCore.QByteArray.fromBase64(
                    dock_state["state"].encode("ascii")), STATE_VERSION))
            except Exception:
                restored = False
            if restored and isinstance(open_n, list):
                for k, d in enumerate(self.docks):
                    d.setVisible(k in open_n)
                if not any(d.isVisible() for d in self.docks):
                    restored = False
            if restored:
                self.layout_name = layout_name if layout_name in LAYOUTS else "custom"
        if not restored:
            self.set_layout(layout_name if layout_name in LAYOUTS else DEFAULT_LAYOUT)
        self.set_active(self.visible_viewports()[0] if self.visible_viewports() else self.viewports[0])

    # ----------------------------------------------------------- windows
    def _new_window(self, config: ViewConfig) -> ModelViewport:
        k = len(self.viewports)
        vp = ModelViewport(self.doc, self.state, config)
        vp.activated.connect(self.set_active)
        vp.view_changed.connect(lambda _vp: self._update_highlight())
        vp.view_changed.connect(lambda _vp, k=k: self._retitle(k))
        vp.display_requested.connect(self.edit_display)
        dock = QtWidgets.QDockWidget(self.host)
        dock.setObjectName(f"model_window_{k + 1}")
        dock.setWidget(vp)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable | QtWidgets.QDockWidget.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        dock.topLevelChanged.connect(lambda floating, vp=vp: self._window_moved(vp, floating))
        dock.dockLocationChanged.connect(lambda _a: self._moved_by_hand())
        dock.visibilityChanged.connect(lambda shown, vp=vp: self._visibility(vp, shown))
        self.viewports.append(vp)
        self.docks.append(dock)
        self.host.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        dock.hide()
        self._retitle(k)
        return vp

    def _retitle(self, k: int):
        if k >= len(self.docks):
            return
        c = self.viewports[k].config
        what = {"plan": f"Plan {c.story or ''}".strip(), "3d": "3-D View",
                "xz": f"Elevation {c.grid or 'X-Z'}", "yz": f"Elevation {c.grid or 'Y-Z'}"}.get(c.kind, c.kind)
        self.docks[k].setWindowTitle(f"Window {k + 1}  -  {what}")

    def _window_moved(self, vp: ModelViewport, floating: bool):
        """Taken out / put back: the picture is drawn again in its new place (the
        native window of the drawing may have changed)."""
        def again():
            try:
                w = vp.vtk_widget
                rw = w.GetRenderWindow()
                wid = str(int(w.winId()))
                if getattr(vp, "_window_id", None) not in (None, wid):
                    rw.SetWindowInfo(wid)
                vp._window_id = wid
            except Exception:
                pass
            vp.render()
        QtCore.QTimer.singleShot(0, again)
        self._custom()
        self._changed()

    def _moved_by_hand(self):
        if not getattr(self, "_arranging", False) and getattr(self, "active", None) is not None:
            self._custom()
        self._changed()

    def _custom(self):
        """The windows were arranged by hand: no quick layout is in use."""
        if self.layout_name != "custom":
            self.layout_name = "custom"
            self.layout_changed.emit("custom")

    def _visibility(self, vp: ModelViewport, shown: bool):
        if getattr(self, "active", None) is None or getattr(self, "_closing", False):
            return  # still being built / being closed
        try:
            self._visibility_changed(vp, shown)
        except RuntimeError:  # the windows are being deleted (the program closes)
            self._closing = True

    def _visibility_changed(self, vp: ModelViewport, shown: bool):
        if shown:
            vp.render()
        elif vp is self.active:
            rest = [v for v in self.visible_viewports() if v is not vp]
            if rest:
                QtCore.QTimer.singleShot(0, lambda: self.set_active(rest[0]) if rest[0] in self.visible_viewports()
                                         else None)
        self._changed()

    def _changed(self):
        self._save_timer.start()

    def add_window(self, config: Optional[ViewConfig] = None) -> Optional[ModelViewport]:
        """Window > New Model Window: a closed window back, else a new one (up to
        MAX_WINDOWS), placed beside the active window."""
        closed = [k for k, d in enumerate(self.docks) if d.isHidden()]
        if closed:
            k = closed[0]
        elif len(self.viewports) < MAX_WINDOWS:
            self._new_window(config or _default_config(len(self.viewports)))
            k = len(self.viewports) - 1
        else:
            return None
        if config is not None:
            self.viewports[k].set_config(config)
        dock = self.docks[k]
        anchor = self.docks[self.viewports.index(self.active)] if self.active in self.viewports else None
        if anchor is not None and not anchor.isHidden() and not anchor.isFloating() and anchor is not dock:
            self.host.splitDockWidget(anchor, dock, QtCore.Qt.Horizontal)
        else:
            self.host.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        dock.setFloating(False)
        dock.show()
        self._custom()
        self.set_active(self.viewports[k])
        self._changed()
        return self.viewports[k]

    # ----------------------------------------------------------- settings
    def _load(self):
        layout_name = DEFAULT_LAYOUT
        configs = [ViewConfig.from_dict(c.to_dict()) for c in DEFAULT_CONFIGS]
        dock_state = None
        if self.qsettings is not None:
            name = self.qsettings.value("windows/layout", DEFAULT_LAYOUT)
            layout_name = name if (name in LAYOUTS or name == "custom") else DEFAULT_LAYOUT
            raw = self.qsettings.value("windows/configs", "")
            try:
                data = json.loads(raw) if raw else []
                for k, d in enumerate(data[:MAX_WINDOWS]):
                    if k < len(configs):
                        configs[k] = ViewConfig.from_dict(d)
                    else:
                        configs.append(ViewConfig.from_dict(d))
            except (ValueError, TypeError):
                pass
            raw = self.qsettings.value("windows/docks", "")
            try:
                d = json.loads(raw) if raw else None
                if isinstance(d, dict) and d.get("version") == STATE_VERSION and len(configs) >= d.get("count", 0):
                    dock_state = d
            except (ValueError, TypeError):
                dock_state = None
        return layout_name, configs, dock_state

    def save(self):
        if self.qsettings is None or not getattr(self, "docks", None):
            return
        self.qsettings.setValue("windows/layout", self.layout_name)
        self.qsettings.setValue("windows/configs", json.dumps([vp.config.to_dict() for vp in self.viewports]))
        state = bytes(self.host.saveState(STATE_VERSION).toBase64()).decode("ascii")
        self.qsettings.setValue("windows/docks", json.dumps({
            "version": STATE_VERSION, "count": len(self.docks), "state": state,
            "open": [k for k, d in enumerate(self.docks) if not d.isHidden()]}))

    def arrange(self, n: int, stacked: bool = False):
        """The first ``n`` windows in an equal grid (any number 1 - MAX_WINDOWS)."""
        n = max(1, min(MAX_WINDOWS, int(n)))
        if stacked and n == 2:
            self.set_layout("2v")
            return
        LAYOUTS.setdefault(str(n), n)
        self.set_layout(str(n))

    def show_windows(self, n: int):
        """Keep the arrangement: open / close windows to have ``n`` open."""
        n = max(1, min(MAX_WINDOWS, int(n)))
        while len(self.visible_viewports()) < n and self.add_window() is not None:
            pass
        for vp in list(reversed(self.visible_viewports()))[:max(0, len(self.visible_viewports()) - n)]:
            self.docks[self.viewports.index(vp)].close()
        self._custom()
        self._changed()

    # ------------------------------------------------------------- layout
    def visible_viewports(self) -> List[ModelViewport]:
        """The open windows (also one behind another in a tab group)."""
        return [vp for vp, d in zip(self.viewports, self.docks) if not d.isHidden()]

    def set_layout(self, name: str):
        """A quick layout: the first windows in a grid ("2v": two stacked)."""
        if name not in LAYOUTS:
            raise ValueError(f"unknown layout {name!r}")
        n = LAYOUTS[name]
        self._arranging = True
        try:
            self._grid(name, n)
        finally:
            self._arranging = False
        if self.active not in self.visible_viewports():
            self.set_active(self.visible_viewports()[0])
        for vp in self.visible_viewports():
            vp.render()
        self.save()
        self.layout_changed.emit(name)

    def _grid(self, name: str, n: int):
        while len(self.viewports) < n:
            self._new_window(_default_config(len(self.viewports)))
        self.layout_name = name
        for d in self.docks:
            d.setFloating(False)
            self.host.removeDockWidget(d)
        docks = self.docks[:n]
        if name == "2v":
            cols, rows = 1, 2
        else:
            cols = max(1, math.ceil(math.sqrt(n)))
            rows = max(1, math.ceil(n / cols))
        grid = [docks[r * cols:(r + 1) * cols] for r in range(rows)]
        # columns first (the top row split left to right), then each column split downwards
        top = grid[0]
        self.host.addDockWidget(QtCore.Qt.LeftDockWidgetArea, top[0])
        top[0].show()
        for a, b in zip(top[:-1], top[1:]):
            self.host.splitDockWidget(a, b, QtCore.Qt.Horizontal)
            b.show()
        for c in range(cols):
            col = [grid[r][c] for r in range(rows) if c < len(grid[r])]
            for a, b in zip(col[:-1], col[1:]):
                self.host.splitDockWidget(a, b, QtCore.Qt.Vertical)
                b.show()
        for d in self.docks[n:]:
            d.hide()
        # equal sizes
        W = max(self.host.width(), 400)
        H = max(self.host.height(), 300)
        if len(top) > 1:
            self.host.resizeDocks(top, [W // len(top)] * len(top), QtCore.Qt.Horizontal)
        for c in range(cols):
            col = [grid[r][c] for r in range(rows) if c < len(grid[r])]
            if len(col) > 1:
                self.host.resizeDocks(col, [H // len(col)] * len(col), QtCore.Qt.Vertical)

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
        self._closing = True
        for vp in self.viewports:
            vp.finalize()


class WindowsDialog(QtWidgets.QDialog):
    """Window > Arrange: how many windows, a quick layout (or keep the arrangement
    made by dragging), and what each window shows."""

    def __init__(self, area: ViewportArea, parent=None):
        super().__init__(parent)
        self.area = area
        self.setWindowTitle("Model Windows")
        f = QtWidgets.QFormLayout(self)
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(1, MAX_WINDOWS)
        self.count.setValue(max(1, len(area.visible_viewports())))
        f.addRow("Number of windows", self.count)
        self.layout_box = QtWidgets.QComboBox()
        self.layout_box.addItem("Grid (equal sizes)", "grid")
        self.layout_box.addItem("Two windows stacked", "2v")
        self.layout_box.addItem("Keep my arrangement (drag the window titles)", "custom")
        self.layout_box.setCurrentIndex(2 if area.layout_name == "custom" else (1 if area.layout_name == "2v" else 0))
        f.addRow("Arrangement", self.layout_box)
        note = QtWidgets.QLabel("Drag a window title onto the side of another window to put it there, onto its "
                                "middle to stack them as tabs; double-click a title to take the window out "
                                "(e.g. to a second screen).")
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid)")
        f.addRow(note)
        self.rows = []
        self._row_widgets = []
        m = area.doc.model
        stories = [s.name for s in m.stories_bottom_up()][::-1]
        for slot in range(MAX_WINDOWS):
            cfg = area.viewports[slot].config if slot < len(area.viewports) else _default_config(slot)
            kind = QtWidgets.QComboBox()
            for key, label in (("plan", "Plan"), ("3d", "3D"), ("xz", "Elevation X-Z"), ("yz", "Elevation Y-Z")):
                kind.addItem(label, key)
            kind.setCurrentIndex(["plan", "3d", "xz", "yz"].index(cfg.kind))
            where = QtWidgets.QComboBox()
            results = QtWidgets.QCheckBox("results")
            results.setChecked(cfg.results)

            def fill(_=None, kind=kind, where=where, cfg=cfg):
                where.clear()
                k = kind.currentData()
                if k == "plan":
                    where.addItems(stories)
                    if cfg.story in stories:
                        where.setCurrentText(cfg.story)
                elif k == "xz":
                    where.addItems([g.name for g in m.grid_lines if g.direction == "Y"])
                elif k == "yz":
                    where.addItems([g.name for g in m.grid_lines if g.direction == "X"])
                if k in ("xz", "yz") and cfg.grid:
                    where.setCurrentText(cfg.grid)
                where.setEnabled(k != "3d")

            kind.currentIndexChanged.connect(fill)
            fill()
            roww = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(roww)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(kind)
            row.addWidget(where, 1)
            row.addWidget(results)
            label = QtWidgets.QLabel(f"Window {slot + 1}")
            f.addRow(label, roww)
            self.rows.append((kind, where, results))
            self._row_widgets.append((label, roww))
        self.count.valueChanged.connect(self._show_rows)
        self._show_rows(self.count.value())
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _show_rows(self, n: int):
        for k, (label, roww) in enumerate(self._row_widgets):
            label.setVisible(k < n)
            roww.setVisible(k < n)
        if n != 2 and self.layout_box.currentData() == "2v":
            self.layout_box.setCurrentIndex(0)

    def accept(self):
        n = self.count.value()
        for slot, (kind, where, results) in enumerate(self.rows[:n]):
            k = kind.currentData()
            cfg = ViewConfig(k, results=results.isChecked())
            if k == "plan":
                cfg.story = where.currentText() or None
            elif k in ("xz", "yz"):
                cfg.grid = where.currentText() or None
            while len(self.area.viewports) <= slot:
                self.area._new_window(_default_config(len(self.area.viewports)))
            self.area.configure(slot, cfg)
        how = self.layout_box.currentData()
        if how == "custom":
            self.area.show_windows(n)
        else:
            self.area.arrange(n, stacked=(how == "2v"))
        super().accept()
