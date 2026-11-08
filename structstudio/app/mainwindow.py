"""Main window: menus, toolbars (draw palette on the left), docks, tiled
model windows and the analysis runner."""
from __future__ import annotations

import json
import os
import traceback
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__, analyze
from ..model import ModelError
from . import dialogs, icons, theme
from .colors_dialog import GraphicsColorsDialog
from .document import Document
from .icons import icon
from .menu_commands import MenuCommandsMixin
from .panels import LogPanel, ModelExplorer, PropertyPanel, ResultsPanel
from .state import ViewState
from .viewport import ModelViewport
from .workspace import LAYOUT_LABELS, ViewportArea, WindowsDialog

DIAGRAMS = ["None", "P", "V2", "V3", "T", "M2", "M3", "F11", "F22", "F12", "M11", "M22", "M12", "V13", "V23"]
FRAME_COLORS = ["None", "P", "V2", "V3", "T", "M2", "M3", "Stress ratio"]



def use_english_numbers() -> None:
    """Numbers in every widget with Arabic digits 0-9, '.' decimals and no group
    separators - independent of the Windows region (a Thai region may otherwise
    show Thai digits in spin boxes and tables)."""
    loc = QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates)
    loc.setNumberOptions(QtCore.QLocale.OmitGroupSeparator | QtCore.QLocale.RejectGroupSeparator)
    QtCore.QLocale.setDefault(loc)


# (key, label, shortcut, icon) grouped like the ETABS / SAP2000 draw toolbars
SELECT_TOOLS = [
    ("select", "Select Object (pointer / window / crossing)", "S", "select"),
    ("select_line", "Select by Intersecting Line", "Ctrl+L", "select_line"),
]
DRAW_TOOLS = [
    ("point", "Draw Joint Objects", "J", "point"),
    ("beam", "Draw Beam / Column / Brace (line)", "B", "beam"),
    ("quick_beam", "Quick Draw Beams (click grid segment)", "Shift+B", "quick_beam"),
    ("secondary_beams", "Quick Draw Secondary Beams", "Shift+E", "secondary_beams"),
    ("column", "Draw Columns (plan click)", "C, L", "column"),
    ("quick_column", "Quick Draw Columns (grid intersections)", "Shift+C", "quick_column"),
    ("brace", "Draw Braces", "Shift+R", "brace"),
    ("quick_brace", "Quick Draw Braces (click a bay in elevation)", "Shift+X", "quick_brace"),
    ("wall", "Draw Walls", "W", "wall"),
    ("quick_wall", "Quick Draw Walls", "Shift+W", "quick_wall"),
    ("slab", "Draw Floor / Slab (polygon)", "A", "slab"),
    ("rect_slab", "Draw Rectangular Floor", "Shift+R+A", "rect_slab"),
    ("quick_slab", "Quick Draw Floor (click grid cell)", "Shift+A", "quick_slab"),
    ("opening", "Draw Openings", "O, P", "opening"),
    ("panel_load", "Load Panel (drawn load area)", "Shift+P", "panel_load"),
    ("area_load", "Area Load (floors / walls)", "Ctrl+Alt+A", "area_load"),
    ("frame_load", "Frame Load", "Ctrl+Alt+F", "frame_load"),
    ("joint_load", "Joint Load", "Ctrl+Alt+J", "joint_load"),
]
# CAD modify commands (Revit / AutoCAD shortcuts: type the two letters in a window)
MODIFY_TOOLS = [
    ("move", "Move (base point, destination)", "M, V", "move"),
    ("copy", "Copy (base point, destinations)", "C, O", "copy"),
    ("rotate", "Rotate (centre, start, end / typed angle)", "R, O", "rotate"),
    ("mirror", "Mirror (draw the axis)", "M, M", "mirror"),
    ("offset", "Offset (parallel copy of a frame)", "O, F", "offset"),
    ("trim", "Trim (cut a frame at crossing frames / grids)", "T, R", "trim"),
    ("extend", "Extend (to the next frame / grid)", "E, X", "extend"),
]
TOOL_ITEMS = [(k, l, s) for k, l, s, _ in SELECT_TOOLS + DRAW_TOOLS + MODIFY_TOOLS]

# left palette, in ETABS workflow order: each entry is one button; a group of
# several tools becomes a fly-out button that remembers the last tool used
DRAW_PALETTE = [
    ("Select", ["select", "select_line"]),
    ("Draw Joints", ["point"]),
    ("Draw Beams", ["beam", "quick_beam", "secondary_beams"]),
    ("Draw Columns", ["column", "quick_column"]),
    ("Draw Braces", ["brace", "quick_brace"]),
    ("Draw Floors", ["slab", "rect_slab", "quick_slab"]),
    ("Draw Walls", ["wall", "quick_wall"]),
    ("Draw Openings", ["opening"]),
    ("Loads", ["area_load", "panel_load", "frame_load", "joint_load"]),
    ("Modify", ["move", "copy", "rotate", "mirror", "offset", "trim", "extend"]),
]


class AnalysisWorker(QtCore.QThread):
    finished_ok = QtCore.Signal(object)
    failed = QtCore.Signal(str, object)  # message, issues (diagnostics.Issue list)
    progress = QtCore.Signal(float, str)

    def __init__(self, model, mode: str = "analysis"):
        super().__init__()
        self.model = model
        self.mode = mode  # analysis | check_mesh

    cancelled = QtCore.Signal()

    class Cancelled(Exception):
        pass

    def cancel(self):
        """Stop at the next progress step (between load cases / solver stages)."""
        self._cancel = True

    def _progress(self, p, msg):
        if getattr(self, "_cancel", False):
            raise AnalysisWorker.Cancelled()
        self.progress.emit(p, msg)

    def run(self):
        from .. import diagnostics as dg
        self._cancel = False
        try:
            if self.mode == "check_mesh":
                from ..api import check_mesh
                res = check_mesh(self.model, progress=self._progress)
            else:
                res = analyze(self.model, progress=self._progress)
            if self._cancel:
                raise AnalysisWorker.Cancelled()
            self.finished_ok.emit(res)
        except AnalysisWorker.Cancelled:
            self.cancelled.emit()
        except dg.AnalysisError as exc:
            self.failed.emit(f"{exc.code}: {exc.message}", exc.issues)
        except ModelError as exc:
            self.failed.emit(f"{exc.code}: {exc.message}", [dg.Issue("error", f"{exc.code}: {exc.message}")])
        except Exception:
            tb = traceback.format_exc()
            self.failed.emit(tb, [dg.Issue("error", "Internal error - please report it: "
                                           + tb.strip().splitlines()[-1])])


class ImportWorker(QtCore.QThread):
    """Imports a CSI file off the GUI thread (native files are converted by the
    installed program, which takes a while)."""

    finished_ok = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        from ..io.importers import import_model
        try:
            self.finished_ok.emit(import_model(self.path))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(MenuCommandsMixin, QtWidgets.QMainWindow):
    def __init__(self, document: Optional[Document] = None, settings: Optional[QtCore.QSettings] = None):
        super().__init__()
        use_english_numbers()
        self.doc = document or Document()
        self.qsettings = settings
        self.setWindowTitle(f"Structural Studio {__version__}")
        self.resize(1600, 950)
        self.state = ViewState(self.doc, self)
        # display theme + graphics colours (before any window is drawn)
        raw = settings.value("graphics/colors", "") if settings is not None else ""
        self.color_settings = theme.ColorSettings.from_json(raw if isinstance(raw, str) else "")
        self._apply_theme_globals(self.color_settings)
        self.views = ViewportArea(self.doc, self.state, settings)
        self.setCentralWidget(self.views)
        for vp in self.views.viewports:
            vp.object_info_requested.connect(self.show_object_info)
            vp.background_menu_requested.connect(self.show_background_menu)
            vp.panel_requested.connect(self.add_panel_load)
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbars()
        self.doc.on_change(self._on_doc_change)
        self.doc.ask_unlock = self._ask_unlock
        self.state.selection_changed.connect(self._selection_changed)
        self.state.status.connect(self._status)
        self.state.cursor_moved.connect(self.coord_label.setText)
        self.state.error.connect(self._error)
        self.state.tool_changed.connect(self._tool_changed)
        self.views.active_changed.connect(self._active_changed)
        self.views.display_applied.connect(self._sync_display_actions)
        self._build_hint_bar()
        self._worker: Optional[AnalysisWorker] = None
        self._on_doc_change("model")
        self._sync_display_actions()

    @property
    def viewport(self) -> ModelViewport:
        """The active model window."""
        return self.views.active

    # ------------------------------------------------------------ layout
    def _build_docks(self):
        self.explorer = ModelExplorer(self.doc)
        self.tree = self.explorer.tree
        self.props = PropertyPanel(self.doc)
        self.results = ResultsPanel(self.doc)
        self.logger = LogPanel()
        from .object_props import ObjectPropertiesPanel
        self.object_props = ObjectPropertiesPanel(self.state, self.doc)
        for title, widget, area in (
            ("Model", self.explorer, QtCore.Qt.LeftDockWidgetArea),
            ("Properties of Object", self.object_props, QtCore.Qt.LeftDockWidgetArea),
            ("Properties", self.props, QtCore.Qt.RightDockWidgetArea),
            ("Results", self.results, QtCore.Qt.BottomDockWidgetArea),
            ("Log", self.logger, QtCore.Qt.BottomDockWidgetArea),
        ):
            dock = QtWidgets.QDockWidget(title, self)
            dock.setWidget(widget)
            dock.setObjectName(title)
            self.addDockWidget(area, dock)
            setattr(self, "dock_" + title.lower().replace(" ", "_"), dock)
        self.tabifyDockWidget(self.dock_results, self.dock_log)
        # ETABS: the Properties of Object form sits below the Model Explorer
        self.splitDockWidget(self.dock_model, self.dock_properties_of_object, QtCore.Qt.Vertical)
        self.dock_results.raise_()
        # the drawing gets the room: Results / Log (bottom) and Properties (right) start
        # hidden; the toolbar buttons and Options > Panels bring them back
        for dock in (self.dock_results, self.dock_log, self.dock_properties):
            dock.hide()
        self.panel_actions = {}
        for key, dock, tip in (
                ("model", self.dock_model, "Model explorer (left)"),
                ("properties_of_object", self.dock_properties_of_object, "Properties of Object (left)"),
                ("properties", self.dock_properties, "Properties of the selection (right)"),
                ("results", self.dock_results, "Results tables (bottom)"),
                ("log", self.dock_log, "Messages / log (bottom)")):
            a = dock.toggleViewAction()
            a.setToolTip(f"Show / hide: {tip}")
            self.panel_actions[key] = a
        self.tree.item_selected.connect(self._tree_selected)
        self.tree.object_activated.connect(lambda attr, name: self.locate_object(attr, name, zoom=True))
        self.tree.set_clicked.connect(self._tree_set_clicked)
        self.tree.story_clicked.connect(self._tree_story_clicked)

    def _make(self, text, slot, shortcut=None, checkable=False, checked=False, icon_name=None, tip=None):
        a = QtGui.QAction(text, self)
        if icon_name:
            a.setIcon(icon(icon_name))
            a.setProperty("icon_name", icon_name)
        if shortcut:
            a.setShortcut(shortcut)
        a.setCheckable(checkable)
        a.setChecked(checked)
        a.setToolTip(tip or (f"{text} ({shortcut})" if shortcut else text))
        a.triggered.connect(slot)
        return a

    def _menu_tool(self, cid: str, icon_name: str, tip: Optional[str] = None):
        """The menu's action for a command, with an icon, for a toolbar (None if the
        command does not exist)."""
        a = self.menu_actions.get(cid)
        if a is None or a.property("planned"):
            return None
        a.setIcon(icon(icon_name))
        a.setProperty("icon_name", icon_name)
        sc = a.shortcut().toString()
        if tip:
            a.setToolTip(tip if (not sc or sc in tip) else f"{tip} ({sc})")
        return a

    def _build_actions(self):
        st = self.state
        self.tool_actions = {}
        self.tool_group = QtGui.QActionGroup(self)
        for key, label, shortcut, ic in SELECT_TOOLS + DRAW_TOOLS + MODIFY_TOOLS:
            sc = shortcut if "+" not in shortcut or shortcut.count("+") == 1 else None
            a = self._make(label, lambda _=False, k=key: st.set_tool(k), sc, checkable=True,
                           checked=(key == "select"), icon_name=ic)
            self.tool_group.addAction(a)
            self.tool_actions[key] = a
        # Move = the CAD Move command (Ctrl+M as in ETABS, or M V); exact distances are typed
        # in the window (dx,dy,dz), which replaces the old ETABS Move form
        self.act_move = self.tool_actions["move"]
        self.act_move.setShortcuts([QtGui.QKeySequence("Ctrl+M"), QtGui.QKeySequence("M, V")])
        self.act_move.setToolTip("Move (Ctrl+M or M V): base point, destination - or type dx,dy,dz")
        self.act_replicate = self._make("Array / Replicate to Stories...", self.edit_replicate_tabs, "Ctrl+R",
                                        icon_name="replicate")
        self.act_divide = self._make("Divide Frames...", self.edit_divide, icon_name="divide")
        self.act_delete = self._make("Delete", lambda: st.delete_selected(), "Del", icon_name="delete")
        self.act_undo = self._make("Undo", lambda: self.doc.undo(), "Ctrl+Z", icon_name="undo")
        self.act_redo = self._make("Redo", lambda: self.doc.redo(), "Ctrl+Y", icon_name="redo")
        self.act_run = self._make("Run Analysis", self.run_analysis, "F5", icon_name="run")
        self.act_check_mesh = self._make("Check Mesh", self.check_mesh, icon_name="check_mesh",
                                         tip="Check Mesh: find the fine mesh of the floors and walls (solves the "
                                             "model on several meshes; slower) - then tick Use fine mesh")
        self.act_use_fine = self._make("Use Fine Mesh", self.set_fine_mesh, checkable=True,
                                       tip="Analyse with the fine mesh found by Check Mesh")
        self.act_lock = self._make("Lock Model", self.toggle_lock, checkable=True, icon_name="unlock",
                                   tip="Lock / Unlock Model: after an analysis the model is locked (ETABS); "
                                       "unlocking deletes the analysis results")
        self.act_tension_side = self._make(
            "Moment Diagrams on Tension Side", self.set_moment_tension_side, checkable=True,
            checked=self.color_settings.option("moment_tension_side"),
            tip="Positive moment drawn below a beam and negative above (off: plotted along the local axis)")
        self.act_plan_below = self._make(
            "Plan Diagrams: Positive Below / Right", self.set_plan_positive_below, checkable=True,
            checked=self.color_settings.option("plan_positive_below"),
            tip="In a plan, every diagram turned into the plan has its positive side the same way: "
                "below horizontal members, right of vertical ones (off: above / left)")
        # snapping: toolbar toggles for the common snaps, the Snap Options form for all
        raw = self.qsettings.value("draw/snap", "") if self.qsettings is not None else ""
        from .snap import SnapOptions
        st.snap = SnapOptions.from_json(raw if isinstance(raw, str) else "")

        def snap_toggle(key):
            def f(c):
                st.snap.set(key, c)
                self.apply_snap_options(st.snap)
            return f

        self.snap_actions = {}
        self.act_snap_grid = self._make("Snap to grid intersections", snap_toggle("grid"),
                                        checkable=True, checked=st.snap.grid, icon_name="snap_grid")
        self.act_snap_pts = self._make("Snap to joints", snap_toggle("joints"),
                                       checkable=True, checked=st.snap.joints, icon_name="snap_point")
        self.act_snap_mid = self._make("Snap to line ends and midpoints", snap_toggle("ends_midpoints"),
                                       checkable=True, checked=st.snap.ends_midpoints, icon_name="snap_mid")
        self.act_snap_lines = self._make("Snap to lines and edges (nearest)", snap_toggle("lines"),
                                         checkable=True, checked=st.snap.lines, icon_name="snap_line")
        self.act_snap_perp = self._make("Snap to perpendicular projections", snap_toggle("perpendicular"),
                                        checkable=True, checked=st.snap.perpendicular, icon_name="snap_perp")
        self.act_snap_int = self._make("Snap to intersections", snap_toggle("intersections"),
                                       checkable=True, checked=st.snap.intersections, icon_name="snap_int")
        self.act_snap_ortho = self._make("Orthogonal tracking", snap_toggle("ortho"),
                                         checkable=True, checked=st.snap.ortho, icon_name="snap_ortho")
        self.act_snap_only = self._make("Draw Using Snap Only", self.toggle_snap_only, checkable=True,
                                        checked=st.snap.snap_only)
        self.snap_actions = {"grid": self.act_snap_grid, "joints": self.act_snap_pts,
                             "ends_midpoints": self.act_snap_mid, "lines": self.act_snap_lines,
                             "perpendicular": self.act_snap_perp, "intersections": self.act_snap_int,
                             "ortho": self.act_snap_ortho, "snap_only": self.act_snap_only}
        # every other snap / guide of the Snap Options form gets its toggle too (Snap To menu,
        # status bar), so each can be switched on and off in one click
        from .snap import SNAP_ITEMS
        for key, label, hint in SNAP_ITEMS:
            if key not in self.snap_actions:
                a = self._make(label, snap_toggle(key), checkable=True, checked=st.snap.enabled(key), tip=hint)
                self.snap_actions[key] = a

        def osnap_toggle(c):
            st.snap.osnap = bool(c)
            self.apply_snap_options(st.snap)
        self.act_osnap = self._make("Object Snaps On (master switch)", osnap_toggle, checkable=True,
                                    checked=st.snap.osnap,
                                    tip="All object snaps on / off at once (joints, ends, grid, lines ...); "
                                        "tracking (ortho, polar, alignment) keeps its own switch")
        self.snap_actions["osnap"] = self.act_osnap
        self.act_snap_options = self._make("Snap Options...", self.draw_snap_options, icon_name="snap_options")
        self.act_select_all = self._make("Select All (active window)", lambda: self.viewport.select_all(),
                                         "Ctrl+A", icon_name="select_all")
        self.act_select_prev = self._make("Get Previous Selection", st.previous_selection, "Ctrl+P",
                                          icon_name="select_previous")
        self.act_select_clear = self._make("Clear Selection", st.clear_selection, "Ctrl+Shift+A",
                                           icon_name="select_clear")
        self.act_theme_toggle = self._make("Dark / Light Mode", self.toggle_theme, icon_name="theme",
                                           tip="Switch between dark and light display (Window menu)")
        self.layout_actions = {}
        group = QtGui.QActionGroup(self)
        for key, label in LAYOUT_LABELS.items():
            a = self._make(label, lambda _=False, k=key: self.views.set_layout(k), checkable=True,
                           checked=(key == self.views.layout_name), icon_name="layout_" + key)
            group.addAction(a)
            self.layout_actions[key] = a
        self.grid_mode_actions = {}
        gm_group = QtGui.QActionGroup(self)
        for mode, label in dialogs.GRID_MODE_LABELS.items():
            a = self._make(label.replace("  ", " "), lambda _=False, k=mode: self.set_grid_mode(k), checkable=True,
                           tip="X, Y grid lines + stories" if mode == "story" else "X, Y, Z grid lines")
            gm_group.addAction(a)
            self.grid_mode_actions[mode] = a
        # ---- navigation (ETABS Zoom / Pan / Rotate toolbar)
        self.nav_actions = {}
        for key, label, sc, ic, tip in (
                ("rotate", "Rotate 3D View", "Ctrl+Shift+R", "rotate3d",
                 "Left-drag rotates the 3D view (also: right-drag, Shift + middle-drag, arrow keys)"),
                ("pan", "Pan", "Ctrl+Shift+P", "pan", "Left-drag pans the view (also: middle-drag, Ctrl + arrows)"),
                ("zoom_window", "Rubber Band Zoom", "Ctrl+Shift+Z", "zoom_window",
                 "Drag a rectangle around the region to enlarge")):
            a = self._make(label, lambda c, k=key: st.set_nav_mode(k if c else None), sc, checkable=True,
                           icon_name=ic, tip=f"{label} ({sc}) - {tip}. Esc returns to Select.")
            self.nav_actions[key] = a
        st.nav_changed.connect(self._sync_nav)
        self.act_zoom_in = self._make("Zoom In One Step", lambda: self.viewport.zoom(1.25), None,
                                      icon_name="zoom_in", tip="Zoom in (also: mouse wheel at the cursor, +)")
        self.act_zoom_out = self._make("Zoom Out One Step", lambda: self.viewport.zoom(0.8), None,
                                       icon_name="zoom_out", tip="Zoom out (also: mouse wheel, -)")
        self.act_full_view = self._make("Restore Full View", lambda: self.viewport.fit(), "F",
                                        icon_name="zoom_full",
                                        tip="Fit the whole model in the active window (F, double-click the wheel)")
        self.act_default_3d = self._make("Default 3D View", lambda: self.viewport.default_3d(), None,
                                         icon_name="view_3d",
                                         tip="Active window to the standard 3D view direction (Home in a 3D window "
                                             "restores it too)")
        self.view_actions = {}
        for key, label in (("plan", "Plan view"), ("3d", "3D view"), ("xz", "Elevation X-Z"), ("yz", "Elevation Y-Z")):
            self.view_actions[key] = self._make(label, lambda _=False, k=key: self.viewport.set_view(k),
                                                icon_name="view_" + key, tip=f"{label} in the active window")

    def _build_menus(self):
        """Menu bar in the ETABS layout (see :mod:`.menus`)."""
        st = self.state
        # ---- checkable / shared actions used by menus and toolbars
        self.act_labels = self._make("Object Labels", self.toggle_labels, "L", checkable=True)
        self.act_extrude = self._make("Extruded View (real sections)", self.toggle_extrude, "Ctrl+E", checkable=True,
                                      checked=False, icon_name="extrude",
                                      tip="Draw members with their real cross-sections and areas with their "
                                          "thickness (Ctrl+E)")
        self.act_display = self._make("Set Display Options...", lambda: self.views.edit_display(), "Ctrl+W",
                                      icon_name="display",
                                      tip="What the active window shows: objects, assignments, labels, grid, "
                                          "extrusion (Ctrl+W)")
        self.act_grid = self._make("Show Grid Lines", lambda c: self.views.set_option_all("grid_lines", c),
                                   "Ctrl+D", checkable=True, checked=True, icon_name="grid",
                                   tip="Show / hide the grid lines in every window (Ctrl+D)")
        self.act_bubbles = self._make("Show Grid Bubbles (A, B, 1, 2)",
                                      lambda c: self.views.set_option_all("grid_bubbles", c), checkable=True,
                                      checked=True)
        self.act_grid_vertical = self._make("Show Vertical Grid Lines in 3D",
                                            lambda c: self.views.set_option_all("grid_vertical", c),
                                            checkable=True, checked=True,
                                            tip="Column lines at the grid intersections in 3D windows")
        self.act_story_plane = self._make("Show Bounding Plane (plan story in 3D)",
                                          lambda c: self.views.set_option_all("story_plane", c), checkable=True,
                                          checked=True)
        self.act_reference = self._make("Show Reference Lines (through joints)",
                                        lambda c: self.views.set_option_all("reference_lines", c), checkable=True,
                                        checked=False, tip="CYPE 3D style reference lines through every joint, "
                                                           "parallel to the global axes (own colour)")
        self.reference_menu = QtWidgets.QMenu("Reference Lines", self)
        self.reference_menu.addAction(self.act_reference)
        self.reference_menu.addSeparator()
        self.reference_dir_actions = {}
        for key, label in (("reference_x", "Parallel to X"), ("reference_y", "Parallel to Y"),
                           ("reference_z", "Parallel to Z")):
            a = self._make(label, lambda c, k=key: self.views.set_option_all(k, c), checkable=True, checked=True)
            self.reference_menu.addAction(a)
            self.reference_dir_actions[key] = a
        self.grid3d_menu = QtWidgets.QMenu("Grid in 3D Windows", self)
        self.grid3d_actions = {}
        g3 = QtGui.QActionGroup(self)
        from .display import GRID_3D, GRID_3D_LABELS
        for k in GRID_3D:
            a = self._make(GRID_3D_LABELS[k], lambda _=False, k=k: self.views.set_option_all("grid_3d", k),
                           checkable=True, checked=(k == "all"))
            g3.addAction(a)
            self.grid3d_menu.addAction(a)
            self.grid3d_actions[k] = a
        self.act_axes = self._make("Show Axes", self.toggle_axes, checkable=True, checked=True,
                                   tip="Global axes marker in the corner of every window")
        self.act_deformed = self._make("Deformed shape", self.toggle_deformed, checkable=True)
        self.act_mesh = self._make("Show Analysis Mesh", self.toggle_mesh, checkable=True,
                                   tip="Show the analysis mesh of the floors and walls (red dots: where columns, "
                                       "joints and beams are tied into it)")
        self.act_reactions = self._make("Reactions", self.toggle_deformed, checkable=True)
        # theme / colour choices (Options menu)
        self.theme_group = QtGui.QActionGroup(self)
        self.theme_actions = {}
        for t in theme.THEMES:
            a = self._make(theme.THEME_LABELS[t] + " mode", lambda _=False, t=t: self.set_theme(t), checkable=True,
                           checked=(t == self.color_settings.theme))
            self.theme_group.addAction(a)
            self.theme_actions[t] = a
        self.color_by_group = QtGui.QActionGroup(self)
        self.color_by_actions = {}
        for b in theme.COLOR_BY:
            a = self._make(theme.COLOR_BY_LABELS[b], lambda _=False, b=b: self.set_color_by(b), checkable=True,
                           checked=(b == self.color_settings.color_by))
            self.color_by_group.addAction(a)
            self.color_by_actions[b] = a
        # measure tools (View > Measure) are tools like the draw commands
        for key, label in (("measure_line", "Measure Line"), ("measure_area", "Measure Area"),
                           ("measure_angle", "Measure Angle")):
            a = self._make(label, lambda _=False, k=key: st.set_tool(k), checkable=True)
            self.tool_group.addAction(a)
            self.tool_actions[key] = a

        # ---- command id -> existing action
        acts = {f"tool.{k}": a for k, a in self.tool_actions.items()}
        acts.update({
            "edit.undo": self.act_undo, "edit.redo": self.act_redo, "edit.delete": self.act_delete,
            "edit.move": self.act_move, "edit.divide_frames": self.act_divide, "edit.replicate": self.act_replicate,
            "analyze.run": self.act_run, "analyze.lock": self.act_lock,
            "analyze.check_mesh": self.act_check_mesh, "analyze.use_fine_mesh": self.act_use_fine,
            "view.display": self.act_display, "view.zoom_window": self.nav_actions["zoom_window"],
            "view.full": self.act_full_view, "view.zoom_in": self.act_zoom_in, "view.zoom_out": self.act_zoom_out,
            "view.pan": self.nav_actions["pan"], "view.rotate": self.nav_actions["rotate"],
            "view.default_3d": self.act_default_3d, "view.3d": self.view_actions["3d"],
            "view.labels": self.act_labels, "view.extrude": self.act_extrude, "view.axes": self.act_axes,
            "select.add.all": self.act_select_all, "select.previous": self.act_select_prev,
            "select.clear": self.act_select_clear,
            "options.bounding_plane": self.act_story_plane,
            "draw.snap_only": self.act_snap_only, "draw.snap_options": self.act_snap_options,
            "options.model_explorer": self.dock_model.toggleViewAction(),
            "options.moment_tension": self.act_tension_side,
            "options.plan_positive_below": self.act_plan_below,
            "display.mesh": self.act_mesh,
            "window.theme": self.act_theme_toggle,
        })
        cmds_extra = {"analyze.problems": self.show_analysis_issues,
                      "edit.find": lambda: self.find_box.focus_for_typing()}
        cmds = self.menu_commands()
        cmds.update(cmds_extra)
        cmds.update({
            "file.new": self.new_model, "file.open": self.open_model, "file.save": self.save_model,
            "file.save_as": self.save_model_as, "file.import_csi": self.import_csi,
        })
        for k in acts:
            cmds.pop(k, None)
        from .menus import MenuBuilder
        self.menu_builder = MenuBuilder(self, acts, cmds, self.menu_fillers(), self._make)
        self.menus = self.menu_builder.build(self.menuBar())
        self.menu_actions = self.menu_builder.actions

    def _build_toolbars(self):
        """Top toolbars, one per category, in the order of the work (as the ETABS /
        SAP2000 / SAFE toolbars):

        row 1 - the model:  File | Edit | Define (materials, sections) | Loads (patterns,
                 cases, combinations) | Analyze | Find ......... grid system (right end)
        row 2 - looking at it:  View (directions, zoom, pan, rotate) | Display (options,
                 grid, extruded sections, load assigns) | Results (case, mode, shapes,
                 reactions, forces, diagram, colours, tables)

        Window layouts, panels and the theme are in the Window menu."""
        def bar(title, name):
            tb = self.addToolBar(title)
            tb.setObjectName(name)
            tb.setIconSize(QtCore.QSize(22, 22))
            return tb

        def add_cmds(tb, items):
            for cid, ic, tip in items:
                a = self._menu_tool(cid, ic, tip)
                if a is not None:
                    tb.addAction(a)

        # =============================================================== row 1
        top = bar("File", "main")
        add_cmds(top, (("file.new", "file_new", "New Model"), ("file.open", "file_open", "Open"),
                       ("file.save", "file_save", "Save")))
        edit = bar("Edit", "edit")
        for a in (self.act_undo, self.act_redo):
            edit.addAction(a)
        dfn = bar("Define", "define")
        self.define_toolbar = dfn
        add_cmds(dfn, (("define.materials", "materials", "Material Properties"),
                       ("define.frame_sections", "frame_sections", "Frame Sections"),
                       ("define.slab_sections", "slab_sections", "Slab Sections"),
                       ("define.wall_sections", "wall_sections", "Wall Sections")))
        loads = bar("Loads", "loads")
        add_cmds(loads, (("define.load_patterns", "load_patterns", "Load Patterns"),
                         ("define.load_cases", "load_cases", "Load Cases"),
                         ("define.combos", "combos", "Load Combinations")))
        ana = bar("Analyze", "analyze")
        ana.addAction(self.act_run)
        # Fine mesh: Check Mesh finds it once (kept with the model); "Use fine mesh" analyses with it
        ana.addAction(self.act_check_mesh)
        self.fine_mesh_box = QtWidgets.QCheckBox("Use fine mesh")
        self.fine_mesh_box.toggled.connect(self.set_fine_mesh)
        ana.addWidget(self.fine_mesh_box)
        why = QtWidgets.QToolButton()
        why.setText("!")
        why.setObjectName("fine_mesh_info")
        why.setToolTip("What do Check Mesh and Use fine mesh do?")
        why.setStyleSheet("QToolButton#fine_mesh_info { font-weight: 700; color: white; background: #e0a020;"
                          " border-radius: 8px; min-width: 16px; max-width: 16px; min-height: 16px;"
                          " max-height: 16px; padding: 0px; }")
        why.clicked.connect(self.explain_fine_mesh)
        ana.addWidget(why)
        self._sync_fine_mesh()
        ana.addAction(self.act_lock)
        find = bar("Find", "find")
        from .find_ui import FindBox
        self.find_box = FindBox(lambda: self.doc.model, self.go_to_find_hit,
                                lambda t: self.statusBar().showMessage(t, 10000), self)
        find.addWidget(self.find_box)
        # grid-system "persona" switch (like Affinity's Photo / Designer / Publisher
        # personas): X·Y·Story <-> X·Y·Z, any time (undoable).  Last on the row and
        # pushed to its right end: a mode of the whole model, set apart from commands
        persona = self.addToolBar("Grid System")
        persona.setObjectName("grid_system")
        persona.setMovable(False)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        persona.addWidget(spacer)
        self.grid_switch = dialogs.GridModeSwitch()
        self.persona_bar = persona
        self._style_grid_switch()
        self.grid_switch.mode_changed.connect(self.set_grid_mode)
        persona.addWidget(self.grid_switch)

        # =============================================================== row 2
        self.addToolBarBreak()
        # ---- view: directions, then zoom / pan / rotate (ETABS "View" / "Zoom" toolbars)
        nav = bar("View", "navigation")
        for a in self.view_actions.values():
            nav.addAction(a)
        nav.addSeparator()
        for a in (self.nav_actions["zoom_window"], self.act_full_view, self.act_zoom_in, self.act_zoom_out,
                  self.nav_actions["pan"], self.nav_actions["rotate"], self.act_default_3d):
            nav.addAction(a)
        self.nav_toolbar = nav
        # ---- display: how the model is drawn, and its load assignments
        disp = bar("Display", "display")
        for a in (self.act_display, self.act_grid, self.act_extrude):
            disp.addAction(a)
        disp.addSeparator()
        disp.addWidget(QtWidgets.QLabel(" Loads "))
        self.load_box = QtWidgets.QComboBox()
        self.load_box.setMinimumWidth(100)
        self.load_box.setToolTip("Show the loads of a load pattern (Display > Load Assigns)")
        self.load_box.currentTextChanged.connect(self._display_changed)
        disp.addWidget(self.load_box)
        # which loads: joint / frame / area / panel, or the panel loads as applied to members
        self.load_kind_button = QtWidgets.QToolButton()
        self.load_kind_button.setText("Show")
        self.load_kind_button.setToolTip("Which loads are displayed")
        self.load_kind_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        lk_menu = QtWidgets.QMenu(self.load_kind_button)
        self.load_kind_actions = {}
        from .scene import LOAD_KIND_LABELS
        for key in ("joint", "frame", "area", "panel", "applied"):
            a = QtGui.QAction(LOAD_KIND_LABELS[key], self)
            a.setCheckable(True)
            a.setChecked(key in self.state.load_kinds)
            a.toggled.connect(lambda on, k=key: self._load_kind_toggled(k, on))
            lk_menu.addAction(a)
            self.load_kind_actions[key] = a
        lk_menu.addSeparator()
        for text, kinds in (("All loads", {"joint", "frame", "area", "panel"}),
                            ("Only loads on members (panels applied)", {"joint", "frame", "applied"}),
                            ("Only panel loads", {"panel"})):
            a = QtGui.QAction(text, self)
            a.triggered.connect(lambda _=False, kk=kinds: self.set_load_kinds(kk))
            lk_menu.addAction(a)
        self.load_kind_button.setMenu(lk_menu)
        disp.addWidget(self.load_kind_button)
        # ---- results: which case, which shape / force, tables (ETABS Display toolbar F4-F8)
        res = bar("Results", "results")
        res.addWidget(QtWidgets.QLabel(" Case "))
        self.case_box = QtWidgets.QComboBox()
        self.case_box.setMinimumWidth(130)
        self.case_box.currentTextChanged.connect(self._display_changed)
        res.addWidget(self.case_box)
        res.addWidget(QtWidgets.QLabel(" Mode "))
        self.mode_box = QtWidgets.QSpinBox()
        self.mode_box.setRange(0, 999)
        self.mode_box.setSpecialValueText("-")
        self.mode_box.valueChanged.connect(self._display_changed)
        res.addWidget(self.mode_box)
        res.addSeparator()
        add_cmds(res, (("display.undeformed", "undeformed", "Undeformed Shape (F4)"),
                       ("display.deformed", "deformed", "Deformed Shape (F6)"),
                       ("display.reactions", "reactions", "Support Reactions (F7)"),
                       ("display.frame_forces", "frame_forces", "Frame Forces (F8)")))
        res.addWidget(QtWidgets.QLabel(" Diagram "))
        self.diagram_box = QtWidgets.QComboBox()
        self.diagram_box.addItems(DIAGRAMS)
        self.diagram_box.setToolTip("Member force diagram (P V2 V3 T M2 M3) or shell contour (F11 ... V23)")
        self.diagram_box.currentTextChanged.connect(self._display_changed)
        res.addWidget(self.diagram_box)
        res.addWidget(QtWidgets.QLabel(" Colour "))
        self.frame_color_box = QtWidgets.QComboBox()
        self.frame_color_box.addItems(FRAME_COLORS)
        self.frame_color_box.setToolTip(
            "Colour the members along their length by a result (blue small - red large).\n"
            "Stress ratio = elastic |P|/A + |M2|/S22 + |M3|/S33 over fy (steel) or fc' (concrete),\n"
            "shown in ratio colours (0.5 / 0.7 / 0.9 / 1.0) - an indication, not a design check.\n"
            "Lines or real sections: the Extrude button (Display toolbar, Ctrl+E).")
        self.frame_color_box.currentTextChanged.connect(self._display_changed)
        res.addWidget(self.frame_color_box)
        res.addSeparator()
        # panel toggles live in the Window menu; Tables also here, next to the results
        self.act_tables = self.panel_actions["results"]
        self.act_tables.setText("Tables")
        self.act_tables.setIcon(icon("tables"))
        self.act_tables.setProperty("icon_name", "tables")
        self.act_tables.setToolTip("Show / hide the Results tables (Ctrl+T) - use it after an analysis")
        res.addAction(self.act_tables)
        self.results_toolbar = res
        self.act_props_panel = self.panel_actions["properties"]
        self.act_props_panel.setText("Properties panel")
        self.act_props_panel.setIcon(icon("properties_panel"))
        self.act_props_panel.setProperty("icon_name", "properties_panel")
        # ---- left: select / draw / edit / snap palette (vertical, like ETABS)
        left = QtWidgets.QToolBar("Draw and Select")
        left.setObjectName("draw")
        left.setOrientation(QtCore.Qt.Vertical)
        left.setIconSize(QtCore.QSize(26, 26))
        self.draw_toolbar = left
        self.palette_buttons = {}
        # every tool its own button by default ("ui/palette" replaces the former key that
        # defaulted to the compact palette)
        raw = self.qsettings.value("ui/palette", "full") if self.qsettings is not None else "full"
        self.palette_mode = raw if raw in ("compact", "full") else "full"
        self.act_palette_mode = self._make("Show every tool (full palette)", self.toggle_palette_mode,
                                           checkable=True, checked=self.palette_mode == "full",
                                           icon_name="palette",
                                           tip="Left palette: compact (tools of a group in a drop-down) or "
                                               "full (every tool as its own button)")
        self._fill_palette()
        for k, a in getattr(self, "palette_mode_actions", {}).items():
            a.setChecked(k == self.palette_mode)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, left)
        self._build_snap_bar()
        self.coord_label = QtWidgets.QLabel("")
        self.lock_label = QtWidgets.QToolButton()
        self.lock_label.setDefaultAction(self.act_lock)
        self.lock_label.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.lock_label.setAutoRaise(True)
        self.lock_label.setVisible(False)
        self.statusBar().addPermanentWidget(self.lock_label)
        self.statusBar().addPermanentWidget(self.coord_label)
        # the units of the model, lower right corner (as ETABS / SAP2000): click to change
        from .units_ui import UnitsButton
        self.units_button = UnitsButton(lambda: self.doc.model.units, self.change_units, self.options_units, self,
                                        get_display=lambda: self.doc.model.display_units,
                                        set_display=self.set_display_units)
        self.statusBar().addPermanentWidget(self.units_button)

    def _fill_palette(self):
        """(Re)build the left palette in the current mode.  Compact: one button per
        group with a drop-down (ETABS style); full: every tool its own button."""
        left = self.draw_toolbar
        for btn in self.palette_buttons.values():
            if btn is not None:
                btn.deleteLater()
        self.palette_buttons = {}
        left.clear()
        full = self.palette_mode == "full"
        left.setIconSize(QtCore.QSize(22, 22) if full else QtCore.QSize(26, 26))
        edit = [self.act_replicate, self.act_divide, self.act_delete]
        sel = [self.act_select_all, self.act_select_prev, self.act_select_clear]
        if full:
            # every tool its own button; a window too short for one column gets a second one
            from .flyout import PaletteGrid
            items = []
            for title, keys in DRAW_PALETTE:
                if items and title in ("Draw Joints", "Loads", "Modify"):
                    items.append(None)  # a thin line between categories: select | draw | loads | modify
                items += [self.tool_actions[k] for k in keys]
                self.palette_buttons[title] = None
            items += [None] + edit + [None] + sel + [None, self.act_palette_mode]
            self.palette_grid = PaletteGrid(items, left.iconSize(), left)
            left.addWidget(self.palette_grid)
            self._fit_palette()
            return
        self.palette_grid = None
        for title, keys in DRAW_PALETTE:
            if title in ("Draw Joints",):
                left.addSeparator()  # select | draw groups
            self._palette_button(left, title, keys)
        left.addSeparator()  # draw | edit
        # (Move / Copy / Rotate ... are in the Modify fly-out above; snaps live in the status bar)
        for a in edit:
            left.addAction(a)
        left.addSeparator()  # edit | selection
        for a in sel:
            left.addAction(a)
        left.addSeparator()
        left.addAction(self.act_palette_mode)

    def _fit_palette(self):
        """Full palette: flow into more columns when the window is not tall enough."""
        grid = getattr(self, "palette_grid", None)
        if grid is None:
            return
        cw = self.centralWidget()
        h = (cw.height() if cw is not None and cw.height() > 50 else self.height() - 120) - 12
        grid.set_available_height(h)
        self.draw_toolbar.updateGeometry()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, "palette_grid", None) is not None:
            QtCore.QTimer.singleShot(0, self._fit_palette)

    # status bar snap strip (AutoCAD style): one click per snap / guide
    SNAP_BAR = [("osnap", "SNAP"), ("grid", "GRID"), ("ortho", "ORTHO"), ("polar", "POLAR"),
                ("alignment", "TRACK"), ("extensions", "EXT"), ("perpendicular", "PERP"),
                ("length_increments", "INC")]

    def _build_snap_bar(self):
        bar = QtWidgets.QWidget()
        bar.setObjectName("snap_bar")
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(1)
        self.snap_buttons = {}
        for key, text in self.SNAP_BAR:
            b = QtWidgets.QToolButton()
            self.snap_actions[key].setIconText(text)  # the button's label (menus show the full text)
            b.setDefaultAction(self.snap_actions[key])
            b.setAutoRaise(True)
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            b.setToolTip(self.snap_actions[key].toolTip())
            self.snap_buttons[key] = b
            lay.addWidget(b)
        more = QtWidgets.QToolButton()
        more.setText("\u25be")
        more.setToolTip("All snaps and guides, Snap Options...")
        more.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(more)
        self._fill_snap_menu(menu)
        more.setMenu(menu)
        more.setAutoRaise(True)
        lay.addWidget(more)
        self.snap_bar = bar
        self._style_snap_bar()
        self.statusBar().addPermanentWidget(bar)

    def _style_snap_bar(self):
        bar = getattr(self, "snap_bar", None)
        if bar is None:
            return
        from .theme import UI
        u = UI.get(getattr(self, "color_settings", None).theme if getattr(self, "color_settings", None) else "light",
                   UI["light"])
        bar.setStyleSheet(
            "QToolButton { font-size: 11px; font-weight: 600; padding: 1px 5px; border-radius: 3px;"
            f" color: {u['text']}; }}"
            f"QToolButton:checked {{ background: {u['accent']}; color: white; }}"
            "QToolButton:!checked { color: gray; }")

    def _fill_snap_menu(self, menu):
        """Draw > Snap To and the status bar menu: every snap and guide, one place."""
        from .snap import SNAP_ITEMS
        menu.addAction(self.act_osnap)
        menu.addSeparator()
        guides = ("ortho", "polar", "alignment", "length_increments", "angle_increments")
        for key, _, _ in SNAP_ITEMS:
            if key not in guides:
                menu.addAction(self.snap_actions[key])
        menu.addSeparator()
        for key in guides:
            menu.addAction(self.snap_actions[key])
        menu.addSeparator()
        menu.addAction(self.act_snap_only)
        menu.addAction(self.act_snap_options)

    def set_palette_mode(self, mode: str):
        self.palette_mode = "full" if mode == "full" else "compact"
        self.act_palette_mode.blockSignals(True)
        self.act_palette_mode.setChecked(self.palette_mode == "full")
        self.act_palette_mode.blockSignals(False)
        for k, a in getattr(self, "palette_mode_actions", {}).items():
            a.setChecked(k == self.palette_mode)
        if self.qsettings is not None:
            self.qsettings.setValue("ui/palette", self.palette_mode)
        self._fill_palette()

    def toggle_palette_mode(self, checked: bool):
        self.set_palette_mode("full" if checked else "compact")

    def _palette_button(self, bar: QtWidgets.QToolBar, title: str, keys) -> None:
        """One palette button; several tools make a CAD-style fly-out: pressing the
        face opens the group's tools in a row beside the button at once (no arrow,
        no vertical menu), and the face shows the tool picked last."""
        from .flyout import FlyoutButton
        acts = [self.tool_actions[k] for k in keys]
        if len(acts) == 1:
            bar.addAction(acts[0])
            self.palette_buttons[title] = None
            return
        btn = FlyoutButton(title, acts, bar, side="right" if bar.orientation() == QtCore.Qt.Vertical else "down")
        btn.setIconSize(bar.iconSize())
        bar.addWidget(btn)
        self.palette_buttons[title] = btn

    def _sync_nav(self, mode: str):
        for k, a in self.nav_actions.items():
            a.blockSignals(True)
            a.setChecked(k == mode)
            a.blockSignals(False)

    # ----------------------------------------------------------- handlers
    def _on_doc_change(self, what: str):
        if what in ("lock", "results", "model"):
            self._sync_lock()
            self._sync_fine_mesh()
        self.setWindowTitle(f"Structural Studio {__version__} - {self.doc.title}")
        self.act_undo.setEnabled(self.doc.can_undo)
        self.act_redo.setEnabled(self.doc.can_redo)
        if what == "model":
            self._sync_grid_mode()
            self._follow_model_units()
            if getattr(self, "units_button", None) is not None:
                self.units_button.refresh()
            self.load_box.blockSignals(True)
            cur = self.load_box.currentText()
            self.load_box.clear()
            self.load_box.addItems(["(none)"] + list(self.doc.model.load_patterns))
            if cur:
                self.load_box.setCurrentText(cur)
            self.load_box.blockSignals(False)
        if what in ("results", "model"):
            self.case_box.blockSignals(True)
            cur = self.case_box.currentText()
            self.case_box.clear()
            if self.doc.results is not None:
                self.case_box.addItems(list(self.doc.results.cases) + list(self.doc.model.load_combinations))
                if cur:
                    self.case_box.setCurrentText(cur)
            self.case_box.blockSignals(False)
            self._display_changed()

    def _follow_model_units(self):
        """The values typed in the Properties of Object form and the snap settings
        are in the model's units: when the units change (Display Units, undo, a model
        opened or created in other units) they are converted with it.  Forms show
        values in the model's display units (Options > Display Units)."""
        from ..display_units import DisplayUnits
        from . import forms
        forms.DISPLAY["units"] = DisplayUnits.of(self.doc.model)
        u = self.doc.model.units
        prev = getattr(self, "_session_units", None)
        if prev is not None and prev != u:
            self._convert_session_values(prev, u)
        self._session_units = u

    def _frame_color(self):
        box = getattr(self, "frame_color_box", None)
        t = box.currentText() if box is not None else ""
        return None if t in ("", "None") else t

    def _display_changed(self, *_):
        mode = self.mode_box.value()
        d = self.diagram_box.currentText()
        lp = self.load_box.currentText()
        self.state.set_display(display_case=self.case_box.currentText() or None,
                               display_mode=mode if mode > 0 else None,
                               display_diagram=None if d == "None" else d,
                               display_frame_color=self._frame_color(),
                               deformed=self.act_deformed.isChecked(),
                               show_reactions=self.act_reactions.isChecked(),
                               load_pattern=None if lp in ("", "(none)") else lp)

    # ------------------------------------------------------- load display
    def _load_kind_toggled(self, key: str, on: bool):
        kinds = set(self.state.load_kinds)
        if on:
            kinds.add(key)
            if key == "applied":  # loads as they reach the members: no panels / floor loads
                kinds.discard("panel")
                kinds.discard("area")
            elif key in ("panel", "area"):
                kinds.discard("applied")
        else:
            kinds.discard(key)
        self.set_load_kinds(kinds)

    def set_load_kinds(self, kinds):
        self.state.load_kinds = set(kinds)
        for k, a in self.load_kind_actions.items():
            a.blockSignals(True)
            a.setChecked(k in self.state.load_kinds)
            a.blockSignals(False)
        self.state.display_changed.emit()

    # ---------------------------------------------------- right-click forms
    def add_panel_load(self, points):
        """A load panel polygon was drawn: ask for its loads, then add it."""
        from .panel_dialog import PanelLoadDialog
        from ..model.objects import PanelLoad
        m = self.doc.model
        if not m.load_patterns:
            QtWidgets.QMessageBox.information(self, "Load panel", "Define a load pattern first (Define menu).")
            return
        story = self.state.tool.story if self.state.tool.plane.kind == "plan" else None
        draft = PanelLoad(m._next_name("PL", m.panel_loads) if hasattr(m, "_next_name") else "PL1",
                          [tuple(p) for p in points], story=story)
        if draft.span_dir is None:
            from .panel_dialog import edge_direction
            draft.span_dir = edge_direction(draft.points, 0)
        d = PanelLoadDialog(m, draft, self)
        if not d.exec():
            return
        panel = d.value()
        stories = self.state.tool.story_names_to_draw() if story else []
        all_st = self.state.settings.all_stories and len(stories) > 1

        def mutate(model):
            if all_st:
                z0 = model.stories[story].elevation
                for st in stories:
                    dz = model.stories[st].elevation - z0
                    pts = [(x, y, z + dz) for x, y, z in panel.points]
                    model.add_panel_load(pts, panel.distribution, panel.span_dir, story=st, loads=list(panel.loads))
            else:
                model.add_panel_load(panel.points, panel.distribution, panel.span_dir, panel.name, story,
                                     list(panel.loads)).receivers = panel.receivers
        self.doc.edit(mutate)
        self.state.update_hint()

    def _edit_selected_panels(self, names):
        self.state.set_tool("select")
        for n in names:
            self.edit_panel_load(n)

    def edit_panel_load(self, name: str):
        from .panel_dialog import PanelLoadDialog
        m = self.doc.model
        panel = m.panel_loads.get(name)
        if panel is None:
            return
        d = PanelLoadDialog(m, panel, self)
        if d.exec():
            new = d.value()
            self.doc.edit(lambda model: model.panel_loads.__setitem__(name, new))

    def show_object_info(self, kind: str, name: str):
        """Right click on an object: its Information form (ETABS / SAP2000)."""
        if kind == "panel":
            self.edit_panel_load(name)
            return
        from .object_info import open_object_info
        self._info_dialogs = [d for d in getattr(self, "_info_dialogs", []) if d.isVisible()]
        self._info_dialogs.append(open_object_info(self.doc, kind, name, self))

    def show_background_menu(self, vp, x: int, y: int):
        """Right click on the background of a model window (ETABS list)."""
        menu = QtWidgets.QMenu(self)
        ma = self.menu_actions
        for cid in ("draw.grids", "-", "view.show_selected", "view.invert_visibility", "view.hide_selected",
                    "view.restore_visibility", "view.show_all", "-", "draw.snap_options", "options.colors_display"):
            if cid == "-":
                menu.addSeparator()
            elif cid in ma:
                menu.addAction(ma[cid])
        h = vp.vtk_widget.height()
        menu.exec(vp.vtk_widget.mapToGlobal(QtCore.QPoint(int(x), int(h - y))))

    def _tree_selected(self, attr: str, name: str):
        self.props.show_object(attr, name)
        if attr == "frames":
            self.state.set_selection([name])
            self.results.set_frame(name)
        elif attr == "areas":
            self.state.set_selection(areas=[name])
        elif attr == "points":
            self.state.set_selection(points=[name])
        elif attr == "panel_loads":
            self.state.toggle("panel", name, False)
        if attr in ("frames", "areas", "points", "panel_loads"):
            self.locate_object(attr, name)

    def locate_object(self, attr: str, name: str, zoom: bool = False):
        """Show where an object is: plan windows go to its story, it flashes in
        every window that shows it, the status bar says story / grid / size;
        ``zoom`` (double-click in the Model Explorer) also zooms onto it."""
        from . import locate
        m = self.doc.model
        X = locate.object_xyz(m, attr, name)
        if X is None:
            return
        story = locate.object_story(m, attr, name)
        shown = 0
        for vp in self.views.visible_viewports():
            if vp.config.kind == "plan" and story and vp.config.story != story and not vp.shows_points(X):
                vp.set_story(story)
            if vp.shows_points(X):
                vp.flash(X, closed=attr in ("areas", "panel_loads"))
                if zoom:
                    vp.zoom_to(X)
                shown += 1
        text = locate.location_text(m, attr, name)
        if not shown:
            text += "   (not in the open windows - open a 3-D view or its elevation)"
        self.statusBar().showMessage(text, 15000)

    def _locate_issue(self, attr: str, name: str, zoom: bool = True):
        """From the problems list: select the object (so it can be fixed right
        away) and show where it is."""
        m = self.doc.model
        if attr in ("frames", "areas", "points") and name in getattr(m, attr):
            self.state.set_selection([name] if attr == "frames" else [],
                                     areas=[name] if attr == "areas" else [],
                                     points=[name] if attr == "points" else [])
        self.locate_object(attr, name, zoom=zoom)

    def go_to_find_hit(self, hit):
        """A result of the Find box: select + show the object, or mark a location."""
        if hit is None:
            return
        if hit.attr != "xyz":
            self._locate_issue(hit.attr, hit.name, True)
            return
        import numpy as np
        from . import locate
        m = self.doc.model
        X = np.array([hit.xyz], float)
        st = m.story_at(float(X[0, 2]), max(m.options.merge_tolerance, 1e-3) * 10)
        shown = 0
        for vp in self.views.visible_viewports():
            if vp.config.kind == "plan" and st and vp.config.story != st.name and not vp.shows_points(X):
                vp.set_story(st.name)
            if vp.shows_points(X):
                vp.flash(X)
                vp.zoom_to(np.vstack([X - 2.0, X + 2.0]))
                shown += 1
        u = m.units.label("length")
        x, y, z = hit.xyz
        text = (f"Location x = {x:g}, y = {y:g}, z = {z:g} {u}  ·  grid {locate.grid_ref(m, X[0])}"
                + (f"  ·  {st.name}" if st else "") + ("" if shown else "   (not in the open windows)"))
        self.statusBar().showMessage(text, 15000)

    def _tree_set_clicked(self, sets: dict):
        frames = [n for n in sets.get("frames", []) if n in self.doc.model.frames]
        areas = [n for n in sets.get("areas", []) if n in self.doc.model.areas]
        points = [n for n in sets.get("points", []) if n in self.doc.model.points]
        self.state.set_selection(frames, areas=areas, points=points)
        n = len(frames) + len(areas) + len(points)
        self.statusBar().showMessage(f"{n} object(s) selected - Esc clears", 8000)

    def _tree_story_clicked(self, story: str):
        vps = [v for v in self.views.visible_viewports() if v.config.kind == "plan"]
        act = self.views.active
        vp = act if act is not None and act.config.kind == "plan" else (vps[0] if vps else None)
        if vp is None:
            vp = act or self.views.viewports[0]
            vp.set_view("plan", story=story)
        else:
            vp.set_story(story)
        self.statusBar().showMessage(f"Plan view: {story}", 5000)

    def _selection_changed(self):
        st = self.state
        n = st.count
        if len(st.frames) == 1 and not st.areas and not st.points:
            name = next(iter(st.frames))
            self.props.show_object("frames", name)
            self.results.set_frame(name)
        elif len(st.areas) == 1 and not st.frames:
            self.props.show_object("areas", next(iter(st.areas)))
        elif len(st.points) == 1 and not st.frames and not st.areas:
            self.props.show_object("points", next(iter(st.points)))
        elif len(getattr(st, "panels", ())) == 1 and not st.frames and not st.areas and not st.points:
            self.props.show_object("panel_loads", next(iter(st.panels)))
        else:
            # nothing (or several objects) selected: the panel must not keep showing a
            # stale object
            self.props.clear(f"{n} objects selected" if n else "Select an object")

    def _status(self, text: str):
        """A passing message; the guidance of the command comes back after it."""
        self.statusBar().showMessage(text, 5000)

    def _build_hint_bar(self):
        """Bottom left: what to do next in the current command (every tool)."""
        self.hint_label = QtWidgets.QLabel("")
        self.hint_label.setObjectName("commandHint")
        self.hint_label.setStyleSheet("QLabel#commandHint { padding: 0 6px; font-weight: 600; }")
        self.hint_label.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.statusBar().addWidget(self.hint_label, 1)
        self.state.hint_changed.connect(self.hint_label.setText)
        self.state.update_hint()
        # Esc / Enter reach the command even when the keyboard focus is on a panel
        # (Model tree, the section list of Properties of Object ...), not the model window
        QtWidgets.QApplication.instance().installEventFilter(self)

    _TEXT_INPUTS = (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QTextEdit,
                    QtWidgets.QPlainTextEdit)

    def eventFilter(self, obj, ev):  # noqa: N802 (Qt name)
        """Esc / Enter pressed outside the model windows (focus on a panel of this window):
        the active model window handles them, as if it had the focus.  Text fields, open
        drop-down lists, menus and dialogs keep their own Esc / Enter."""
        try:
            if ev.type() != QtCore.QEvent.KeyPress or ev.key() not in (
                    QtCore.Qt.Key_Escape, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                return False
            if not isinstance(obj, QtWidgets.QWidget) or obj.window() is not self:
                return False
            app = QtWidgets.QApplication.instance()
            if app.activePopupWidget() is not None or app.activeModalWidget() is not None:
                return False
            if isinstance(obj, self._TEXT_INPUTS) or isinstance(obj.parent(), QtWidgets.QAbstractSpinBox):
                return False
            if isinstance(obj, QtWidgets.QComboBox) and obj.isEditable():
                return False
            if isinstance(obj, QtWidgets.QAbstractItemView) and obj.state() == QtWidgets.QAbstractItemView.EditingState:
                return False
            vp = self.views.active if hasattr(self, "views") else None
            if vp is None or obj is vp.vtk_widget or vp.vtk_widget.isAncestorOf(obj):
                return False  # the model window has the focus: its own key handler
            if any(obj is v.vtk_widget for v in self.views.viewports):
                return False
            if ev.key() == QtCore.Qt.Key_Escape:
                vp.escape()
            else:
                if not getattr(self.state.tool, "picks", None):
                    return False  # Enter on a panel means the panel's Enter, except in pick commands
                vp.enter()
            return True
        except Exception:
            return False

    def _error(self, text: str):
        self.logger.log("ERROR " + text)
        QtWidgets.QMessageBox.warning(self, "Cannot draw", text)

    def _tool_changed(self, key: str):
        a = self.tool_actions.get(key)
        if a and not a.isChecked():
            a.setChecked(True)
        if key == "panel_load" and self.state.panels:
            # Load Panel with panels selected first: edit them (the command's form)
            names = sorted(self.state.panels)
            QtCore.QTimer.singleShot(0, lambda: self._edit_selected_panels(names))
        if key not in ("select", "select_line"):
            self.dock_properties_of_object.raise_()

    def _active_changed(self, vp):
        self.state.update_hint()  # the window title already names the view
        self._sync_display_actions()

    def _sync_display_actions(self):
        """View-menu check marks follow the display options of the active window."""
        if not hasattr(self, "act_grid"):
            return
        o = self.viewport.display_options
        for a, v in ((self.act_grid, o.grid_lines), (self.act_bubbles, o.grid_bubbles),
                     (self.act_grid_vertical, o.grid_vertical), (self.act_story_plane, o.story_plane),
                     (self.act_extrude, o.extrude), (self.act_labels, o.frame_labels),
                     (self.act_reference, o.reference_lines),
                     *((a, getattr(o, k)) for k, a in self.reference_dir_actions.items()),
                     *((a3, k == o.grid_3d) for k, a3 in self.grid3d_actions.items())):
            a.blockSignals(True)
            a.setChecked(bool(v))
            a.blockSignals(False)

    # ------------------------------------------------------- grid system
    # ------------------------------------------------------------ theme
    def _apply_theme_globals(self, settings: "theme.ColorSettings"):
        theme.apply(settings)
        theme.apply_qt(QtWidgets.QApplication.instance(), settings)
        icons.set_ink(settings.ui()["ink"])

    def _style_grid_switch(self):
        u = self.color_settings.ui()
        self.grid_switch.setStyleSheet(
            f"QTabBar {{ background: {u['window']}; }}"
            f"QTabBar::tab {{ padding: 4px 14px; margin-right: 2px; border: 1px solid {u['mid']};"
            f" border-radius: 4px; background: {u['tab']}; color: {u['tab_text']}; font-weight: 600; }}"
            f"QTabBar::tab:selected {{ background: {u['accent']}; color: {u['accent_text']};"
            f" border-color: {u['accent']}; }}")
        self._style_snap_bar()
        bar = getattr(self, "persona_bar", None)
        if bar is not None:  # no light frame around the switch in dark mode
            bar.setStyleSheet(f"QToolBar#grid_system {{ background: {u['window']}; border: none; }}")

    def apply_color_settings(self, settings: "theme.ColorSettings"):
        """Make ``settings`` active everywhere: scene colours, widget chrome, icons."""
        self.color_settings = settings
        self._apply_theme_globals(settings)
        for a in self.findChildren(QtGui.QAction):
            name = a.property("icon_name")
            if name:
                a.setIcon(icon(name))
        self._style_grid_switch()
        for t, a in self.theme_actions.items():
            a.setChecked(t == settings.theme)
        for b, a in self.color_by_actions.items():
            a.setChecked(b == settings.color_by)
        if hasattr(self, "act_tension_side"):
            self.act_tension_side.blockSignals(True)
            self.act_tension_side.setChecked(settings.option("moment_tension_side"))
            self.act_tension_side.blockSignals(False)
        if hasattr(self, "act_plan_below"):
            self.act_plan_below.blockSignals(True)
            self.act_plan_below.setChecked(settings.option("plan_positive_below"))
            self.act_plan_below.blockSignals(False)
        for vp in self.views.viewports:
            vp.apply_theme()
        self.views.set_active(self.views.active)
        if self.qsettings is not None:
            self.qsettings.setValue("graphics/colors", settings.to_json())

    def set_theme(self, name: str):
        s = self.color_settings.copy()
        s.theme = name
        self.apply_color_settings(s)
        self._status(f"{theme.THEME_LABELS[name]} mode")

    def toggle_theme(self):
        self.set_theme("light" if self.color_settings.theme == "dark" else "dark")

    def set_color_by(self, by: str):
        s = self.color_settings.copy()
        s.color_by = by
        self.apply_color_settings(s)

    def edit_colors(self):
        self.colors_dialog().exec()

    def colors_dialog(self) -> GraphicsColorsDialog:
        """Options > Graphics Colors, wired to apply its changes (Apply / OK)."""
        dlg = GraphicsColorsDialog(self.color_settings, self.doc.model, self)

        def applied(settings):
            changes = {k: v for k, v in dlg.object_colors.items() if self._object_color(k) != v}
            if changes:
                self.doc.edit(dlg.apply_object_colors)
            self.apply_color_settings(settings)

        dlg.applied.connect(applied)
        return dlg

    def set_moment_tension_side(self, on: bool):
        """Options > Moment Diagrams on Tension Side (saved with the graphics settings)."""
        s = self.color_settings.copy()
        s.set_option("moment_tension_side", on)
        self.apply_color_settings(s)

    def set_plan_positive_below(self, on: bool):
        """Options > Plan Diagrams: Positive Below / Right (saved with the graphics settings)."""
        s = self.color_settings.copy()
        s.set_option("plan_positive_below", on)
        self.apply_color_settings(s)

    def restore_graphics_defaults(self, confirm: bool = True):
        """Options > Restore Graphics Defaults: colours of both themes, transparencies and
        sizes back to the defaults (the theme, colour-by mode and model colours stay)."""
        if confirm and QtWidgets.QMessageBox.question(
                self, "Restore Graphics Defaults",
                "Put all colours (light and dark mode), transparencies and sizes back to the defaults?"
        ) != QtWidgets.QMessageBox.Yes:
            return
        settings = self.color_settings.copy()
        settings.reset_all()
        self.apply_color_settings(settings)
        self.statusBar().showMessage("Graphics colours and sizes restored to the defaults", 5000)

    def _object_color(self, key):
        kind, name = key
        m = self.doc.model
        table = m.frame_sections if kind == "frame" else m.shell_sections if kind == "shell" else m.materials
        return getattr(table.get(name), "color", None)

    def set_grid_mode(self, mode: str):
        if dialogs.apply_grid_mode(self.doc, mode):
            label = dialogs.GRID_MODE_LABELS[mode].replace("  ", " ")
            self.logger.log(f"Grid system: {label}")
            self.statusBar().showMessage(f"Grid system: {label}", 5000)
        self._sync_grid_mode()

    def _sync_grid_mode(self):
        mode = self.doc.model.grid_mode
        if hasattr(self, "grid_switch"):
            self.grid_switch.set_mode(mode)
        for k, a in getattr(self, "grid_mode_actions", {}).items():
            a.blockSignals(True)
            a.setChecked(k == mode)
            a.blockSignals(False)

    def arrange_windows(self):
        if WindowsDialog(self.views, self).exec():
            self.layout_actions[self.views.layout_name].setChecked(True)

    def toggle_labels(self, checked: bool):
        self.views.set_option_all("frame_labels", checked)

    def toggle_extrude(self, checked: bool):
        self.views.set_option_all("extrude", checked)

    def toggle_axes(self, checked: bool):
        for vp in self.views.viewports:
            vp.orientation.SetEnabled(bool(checked))
            vp.render()

    def toggle_deformed(self, checked: bool):
        self._display_changed()

    def toggle_mesh(self, checked: bool):
        self.state.set_display(show_mesh=bool(checked))

    def closeEvent(self, event):
        self.views.save()
        super().closeEvent(event)

    # ------------------------------------------------------------- define
    def _define(self, attr: str, title: str, factory):
        dialogs.DefineDialog(self.doc, attr, title, factory, parent=self).exec()

    def define_diaphragms(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Diaphragm", "New rigid diaphragm name:", text="D1")
        if ok and name:
            self.doc.edit(lambda m: m.add_diaphragm(name))

    def add_modal_case(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Modal case", "Name:", text="Modal")
        if ok and name:
            self.doc.edit(lambda m: m.add_modal_case(name, max_modes=12))

    def edit_grid_stories(self):
        dialogs.GridStoriesDialog(self.doc, self).exec()

    def _sel(self):
        return self.state.selected()

    def _has_selection(self) -> bool:
        if not any(self._sel()) and not self.state.panels:
            QtWidgets.QMessageBox.information(self, "Nothing selected", "Select objects first.")
            return False
        return True

    def edit_divide(self):
        if not self.state.frames:
            return self._need("frame")
        n, ok = QtWidgets.QInputDialog.getInt(self, "Divide frames", "Number of pieces:", 2, 2, 100)
        if ok:
            from .tools import divide_frames
            created = divide_frames(self.doc, list(self.state.frames), n)
            self.state.set_selection(created)

    def edit_point(self):
        pts = list(self.state.points)
        if len(pts) != 1:
            QtWidgets.QMessageBox.information(self, "Edit point", "Select exactly one point.")
            return
        p = self.doc.model.points[pts[0]]
        d = dialogs.VectorDialog(f"Point {p.name}", ("X", "Y", "Z"), defaults=(p.x, p.y, p.z), parent=self)
        if d.exec():
            from .tools import set_point_coordinates
            set_point_coordinates(self.doc, p.name, *d.values())

    # ------------------------------------------------------------- select
    def select_by_story(self):
        d = dialogs.ChoiceDialog("Select by story", "Story", [s.name for s in self.doc.model.stories_bottom_up()][::-1], self)
        if d.exec():
            st = d.value()
            m = self.doc.model
            self.state.set_selection([f.name for f in m.frames.values() if f.story == st],
                                        [a.name for a in m.areas.values() if a.story == st],
                                        [p.name for p in m.points.values() if p.story == st])

    def select_by_section(self):
        d = dialogs.ChoiceDialog("Select by section", "Frame section", list(self.doc.model.frame_sections), self)
        if d.exec():
            sec = d.value()
            self.state.set_selection([f.name for f in self.doc.model.frames.values() if f.section == sec])

    # ------------------------------------------------------------- assign
    def _need(self, kind: str) -> bool:
        have = {"point": self.state.points, "frame": self.state.frames, "area": self.state.areas}[kind]
        if not have:
            QtWidgets.QMessageBox.information(self, "Nothing selected", f"Select one or more {kind} objects first.")
            return False
        return True

    def assign_restraints(self):
        if not self._need("point"):
            return
        pts = list(self.state.points)
        from .object_info import RestraintDialog
        p0 = self.doc.model.points[pts[0]]
        d = RestraintDialog(p0.restraints, p0.local_axes, self)
        if d.exec():
            val, la = d.value(), d.local_axes()
            self.doc.edit(lambda m: [(setattr(m.points[p], "restraints", val), setattr(m.points[p], "local_axes", la))
                                     for p in pts])

    def assign_diaphragm(self):
        if not self.state.points and not self.state.areas:
            return self._need("point")
        d = dialogs.ChoiceDialog("Assign diaphragm", "Diaphragm", ["(none)"] + list(self.doc.model.diaphragms), self)
        if d.exec():
            val = None if d.value() == "(none)" else d.value()
            pts, areas = list(self.state.points), list(self.state.areas)
            self.doc.edit(lambda m: [setattr(m.points[p], "diaphragm", val) for p in pts]
                          + [setattr(m.areas[a], "diaphragm", val) for a in areas])

    def assign_frame_section(self):
        """Assign > Frame > Section Property: a command (choose here, pick, Enter / right-click)."""
        self.load_command("frame_section")

    def assign_shell_section(self):
        if not self._need("area"):
            return
        d = dialogs.ChoiceDialog("Assign shell section", "Section", list(self.doc.model.shell_sections), self)
        if d.exec():
            sec, names = d.value(), list(self.state.areas)
            self.doc.edit(lambda m: [setattr(m.areas[n], "section", sec) for n in names])

    def assign_releases(self):
        if not self._need("frame"):
            return
        d = dialogs.ChoiceDialog("Frame releases", "Ends", ["Fixed (no releases)", "Pinned both ends (M2, M3)",
                                                              "Pinned at start", "Pinned at end"], self)
        if d.exec():
            choice, names = d.box.currentIndex(), list(self.state.frames)
            pin = (False, False, False, False, True, True)
            free = (False,) * 6

            def mutate(m):
                for n in names:
                    f = m.frames[n]
                    f.releases_i = pin if choice in (1, 2) else free
                    f.releases_j = pin if choice in (1, 3) else free

            self.doc.edit(mutate)

    def _assign_load(self, kind: str):
        if not self._need(kind):
            return
        if not self.doc.model.load_patterns:
            QtWidgets.QMessageBox.information(self, "No load pattern", "Define a load pattern first (Define > Load Patterns).")
            return
        d = dialogs.LoadDialog(self.doc.model, kind, self)
        if d.exec():
            try:
                d.apply(self.doc, frames=list(self.state.frames), areas=list(self.state.areas), points=list(self.state.points))
            except (ModelError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(self, "Load", str(exc))

    # --------------------------------------------------------------- files
    def new_model(self):
        d = dialogs.NewModelDialog(self)
        if d.exec():
            try:
                model = d.build()
            except (ValueError, ModelError) as exc:
                QtWidgets.QMessageBox.warning(self, "New model", str(exc))
                return
            self.doc.new()
            self.doc.model = model
            self.doc._emit("model")
            for vp in self.views.viewports:
                vp.apply_camera()
            self.logger.log(f"New model: {len(model.grid_lines)} grid lines, {len(model.stories)} stories")

    def open_model(self):
        from ..io.importers import FILE_FILTER
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open model", "", "Structural Studio (*.ssj.json *.json);;" + FILE_FILTER)
        if path:
            self._open(path)

    def _open(self, path: str):
        """Open a Structural Studio model; ETABS / SAP2000 / SAFE files go through Import."""
        if not path.lower().endswith(".json"):
            return self.import_csi(path)
        try:
            self.doc.open(path)
        except Exception as exc:  # bad or foreign JSON: tell the user, keep the current model
            QtWidgets.QMessageBox.critical(self, "Open failed", f"{os.path.basename(path)}:\n{exc}")
            return None
        for vp in self.views.viewports:
            vp.apply_camera()
        self._sync_grid_mode()
        self.logger.log(f"Opened {path}")
        return path

    def import_e2k(self):  # kept for scripts / older shortcuts
        self.import_csi()

    def import_csi(self, path: Optional[str] = None):
        """File > Import: any ETABS / SAP2000 / SAFE model or table export."""
        from ..io.importers import FILE_FILTER, detect, program_of
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import ETABS / SAP2000 / SAFE model", "", FILE_FILTER)
        if not path:
            return
        if detect(path) == "native":
            prog = program_of(path) or "CSI program"
            ok = QtWidgets.QMessageBox.question(
                self, "Import native file",
                f"{os.path.basename(path)} is a native {prog} file.\n\nThe {prog} installed on this computer will "
                "open a copy of it and write it as text (your file is not changed). This can take a minute.\n\n"
                "Continue?")
            if ok != QtWidgets.QMessageBox.Yes:
                return
        self._import_async(path)

    def _import_async(self, path: str):
        worker = ImportWorker(path)
        dlg = QtWidgets.QProgressDialog(f"Importing {os.path.basename(path)} ...", None, 0, 0, self)
        dlg.setWindowTitle("Import")
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(300)
        self._import_worker = worker

        def done(res):
            dlg.close()
            self.doc.set_imported(res.model, res.report)
            self._after_import(path, res)

        def failed(msg):
            dlg.close()
            QtWidgets.QMessageBox.critical(self, "Import failed", msg)

        worker.finished_ok.connect(done)
        worker.failed.connect(failed)
        worker.start()
        return worker

    def _after_import(self, path, res):
        self.logger.log(f"Imported {path}")
        self.logger.log(res.report)
        for vp in self.views.viewports:
            vp.apply_camera()
        self._sync_grid_mode()
        if res.warnings or res.skipped:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Information)
            box.setWindowTitle("Import report")
            src = f"{res.program} {res.version}".strip() or "CSI"
            box.setText(f"Imported {os.path.basename(path)} ({src}).\n{len(res.warnings)} warning(s), "
                        f"{sum(res.skipped.values())} item(s) not supported yet - see details.")
            box.setDetailedText(res.report)
            box.open()
            self._import_box = box

    def save_model(self):
        if not self.doc.path:
            return self.save_model_as()
        self.doc.save()
        self.logger.log(f"Saved {self.doc.path}")

    def save_model_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save model", "model.ssj.json", "Structural Studio (*.ssj.json)")
        if path:
            self.doc.save(path)
            self.logger.log(f"Saved {path}")

    def export_results(self):
        if self.doc.results is None:
            QtWidgets.QMessageBox.information(self, "No results", "Run the analysis first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", "results.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.doc.results.to_dict(), fh, indent=1)
            self.logger.log(f"Exported {path}")

    def screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", "view.png", "PNG (*.png)")
        if path:
            self.views.active.screenshot(path)

    # ------------------------------------------------------------ analysis
    def validate_model(self):
        errors = self.doc.model.validate()
        if errors:
            from .. import diagnostics as dg
            from ..api import _located
            issues = []
            for e in errors:
                self.logger.log(f"ERROR {e.code}: {e.message}")
                attr = name = None
                if e.path and "." in e.path:
                    a, n = e.path.split(".", 1)
                    if a in ("points", "frames", "areas") and n in getattr(self.doc.model, a):
                        attr, name = a, n
                issues.append(dg.Issue("error", f"{e.code}: {e.message}", attr, name))
            self._last_issues = _located(self.doc.model, issues)
            self.show_analysis_issues(title=f"Check model: {len(errors)} problem(s)")
        else:
            self.logger.log("Model is valid")
            self.statusBar().showMessage("Model is valid", 5000)

    def run_analysis(self):
        if self._worker is not None and self._worker.isRunning():
            return
        errors = self.doc.model.validate()
        if errors:
            self.validate_model()
            return
        self.statusBar().showMessage("Running analysis...")
        self.logger.log("Running analysis...")
        from .analysis_ui import AnalysisProgressDialog
        if getattr(self, "_progress_dlg", None) is not None:
            self._progress_dlg.close()
        dlg = self._progress_dlg = AnalysisProgressDialog(self)
        dlg.show_issues.connect(self.show_analysis_issues)
        self._last_issues = []
        self.doc.busy = True  # no edits while the analysis runs (the dialog is modal too)
        self._worker = AnalysisWorker(self.doc.model)
        self._worker.progress.connect(self._analysis_progress)
        self._worker.finished_ok.connect(self._analysis_done)
        self._worker.failed.connect(self._analysis_failed)
        self._worker.cancelled.connect(self._analysis_cancelled)
        dlg.cancel_requested.connect(self._worker.cancel)
        dlg.show()
        self._worker.start()

    def _analysis_cancelled(self):
        self.doc.busy = False
        self.logger.log("Analysis cancelled")
        self.statusBar().showMessage("Analysis cancelled - the model is unchanged", 8000)
        dlg = getattr(self, "_progress_dlg", None)
        if dlg is not None:
            dlg.finish(False, "Analysis cancelled", 0, cancelled=True)

    FINE_MESH_TEXT = (
        "<b>Check Mesh and Use fine mesh</b><br><br>"
        "<b>Default (box not ticked):</b> floors and walls are meshed once with the automatic element size "
        "<b>h</b>, from the geometry of each slab / wall: at least 8 elements across it and 6 between "
        "neighbouring columns, 0.1 to 1.0 m, rounded down to 1, 0.5, 0.25 ... m - quick.<br><br>"
        "<b>Check Mesh</b> (Analyze menu / toolbar) looks for the mesh at which the results stop "
        "changing (h-convergence): it solves the linear static load cases on meshes of size 4h, 2h and h, "
        "and estimates from how the largest slab / wall deflection changes how far the result at h still "
        "is from the converged value.  While the estimate is above the allowed error (2 % by default, "
        "Analyze &gt; Analysis Model Defaults) the size is halved - h/2, then h/4 (each step about four "
        "times the elements), at most 12,000 shell elements.  The size found is kept with the model.<br><br>"
        "<b>Use fine mesh</b> (can be ticked after Check Mesh): every analysis then uses that size.  "
        "It is never coarser than the default h.  The check is run on the geometry of that moment: after "
        "changing slabs, walls, frames or supports run Check Mesh again (the analysis warns).<br><br>"
        "Slab results near columns and point loads converge slowly, so the fine mesh matters most for "
        "punching, deflections and slab reinforcement.")

    def explain_fine_mesh(self, *_):
        o = self.doc.model.options
        from ..api import mesh_signature
        if o.fine_mesh_factor:
            cur = o.fine_mesh_report or f"element size h x {o.fine_mesh_factor:g}"
            if o.fine_mesh_signature and o.fine_mesh_signature != mesh_signature(self.doc.model):
                cur += "<br><b style='color:#b26a00'>The model has changed since this check - run Check Mesh " \
                       "again.</b>"
        else:
            cur = "Not checked yet: run Check Mesh first (Use fine mesh is off until then)."
        QtWidgets.QMessageBox.information(self, "Fine mesh", self.FINE_MESH_TEXT + "<br><br><b>This model:</b> " + cur)

    def set_fine_mesh(self, on: bool):
        """An analysis setting, not a change of the structure: kept with the model, no unlock."""
        o = self.doc.model.options
        on = bool(on) and bool(o.fine_mesh_factor)
        if bool(o.use_fine_mesh) != on:
            o.use_fine_mesh = on
            self.doc.dirty = True
            self.statusBar().showMessage(
                f"Use fine mesh: the next analysis uses element size h x {o.fine_mesh_factor:g}" if on
                else "Fine mesh off: the automatic mesh h is used", 6000)
        self._sync_fine_mesh()

    def _sync_fine_mesh(self):
        o = self.doc.model.options
        have = bool(o.fine_mesh_factor)
        tip = ("Analyse with the fine mesh found by Check Mesh (" + (o.fine_mesh_report or "") + ")" if have
               else "Run Check Mesh first - it finds the fine mesh this option uses")
        for w in (getattr(self, "fine_mesh_box", None), getattr(self, "act_use_fine", None)):
            if w is None:
                continue
            w.blockSignals(True)
            w.setEnabled(have)
            w.setChecked(have and bool(o.use_fine_mesh))
            w.setToolTip(tip)
            w.blockSignals(False)

    def check_mesh(self, *_):
        """Analyze > Check Mesh: find the fine mesh once (in the background, with progress)."""
        if self._worker is not None and self._worker.isRunning():
            return
        errors = self.doc.model.validate()
        if errors:
            self.validate_model()
            return
        from .analysis_ui import AnalysisProgressDialog
        if getattr(self, "_progress_dlg", None) is not None:
            self._progress_dlg.close()
        dlg = self._progress_dlg = AnalysisProgressDialog(self)
        dlg.setWindowTitle("Check Mesh")
        self.doc.busy = True
        self._worker = AnalysisWorker(self.doc.model, mode="check_mesh")
        self._worker.progress.connect(self._analysis_progress)
        self._worker.finished_ok.connect(self._mesh_checked)
        self._worker.failed.connect(self._analysis_failed)
        self._worker.cancelled.connect(self._analysis_cancelled)
        dlg.cancel_requested.connect(self._worker.cancel)
        dlg.show()
        self._worker.start()

    def _mesh_checked(self, info):
        self.doc.busy = False
        o = self.doc.model.options
        o.fine_mesh_factor = float(info["factor"])
        o.fine_mesh_report = info["text"]
        o.fine_mesh_signature = info.get("signature", "")
        self.doc.dirty = True
        self.logger.log("Check Mesh: " + info["text"])
        self._sync_fine_mesh()
        dlg = getattr(self, "_progress_dlg", None)
        if dlg is not None:
            dlg.finish(True, info["text"] + " - tick Use fine mesh to analyse with it")
            dlg.setWindowTitle("Check Mesh - Finished")
        self.statusBar().showMessage("Check Mesh finished: " + info["text"], 12000)

    def _ask_unlock(self) -> bool:
        """ETABS: the analysed model is locked; changing it deletes the results."""
        r = QtWidgets.QMessageBox.question(
            self, "Model is locked",
            "The model is locked because it has analysis results.\n\n"
            "Unlock it to make this change? The analysis results will be deleted.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return r == QtWidgets.QMessageBox.Yes

    def toggle_lock(self, *_):
        """Analyze > Lock / Unlock Model."""
        if self.doc.locked:
            if self._ask_unlock():
                self.doc.unlock()
        else:
            self.doc.lock()
        self._sync_lock()

    def _sync_lock(self):
        act = getattr(self, "act_lock", None)
        if act is not None:
            act.blockSignals(True)
            act.setChecked(self.doc.locked)
            act.setIcon(icon("lock" if self.doc.locked else "unlock"))
            act.setProperty("icon_name", "lock" if self.doc.locked else "unlock")
            act.setText("Unlock Model" if self.doc.locked else "Lock Model")
            act.blockSignals(False)
        lbl = getattr(self, "lock_label", None)
        if lbl is not None:
            lbl.setVisible(self.doc.locked)

    def _analysis_progress(self, pct: float, message: str):
        dlg = getattr(self, "_progress_dlg", None)
        if dlg is not None:
            dlg.set_progress(pct, message)
        self.statusBar().showMessage(f"Analysis {pct:.0f} % - {message}")

    def _analysis_done(self, results):
        self.doc.busy = False
        self.doc.set_results(results)  # ... and locks the model (ETABS)
        s = results.am.summary()
        self.logger.log(f"Analysis complete: {s['nodes']} nodes, {s['frame_elements']} frame elements, "
                        f"{s['shell_elements']} shell elements, {len(results.cases)} cases")
        issues = list(getattr(results, "issues", []))
        self._last_issues = issues
        for i in issues:
            self.logger.log(i.text())
        problems = [i for i in issues if i.severity in ("error", "warning")]
        dlg = getattr(self, "_progress_dlg", None)
        if dlg is not None:
            for i in problems:
                dlg.add_log(i.text())
            dlg.finish(True, f"Analysis complete: {s['nodes']} nodes, {s['shell_elements']} shells, "
                             f"{len(results.cases)} cases", len(problems))
        self.statusBar().showMessage(
            "Analysis complete" + (f" - {len(problems)} warning(s): see Analyze > Last Analysis Problems"
                                   if problems else " - press Tables (Ctrl+T) for the result tables"), 10000)
        if problems:
            self.show_analysis_issues()
        if self.case_box.count() and not self.case_box.currentText():
            self.case_box.setCurrentIndex(0)

    def _analysis_failed(self, message: str, issues=None):
        self.doc.busy = False
        self.logger.log(f"Analysis failed: {message}")
        issues = list(issues or [])
        self._last_issues = issues
        for i in issues:
            self.logger.log(i.text())
        self.statusBar().showMessage("Analysis failed - see the problems list", 8000)
        dlg = getattr(self, "_progress_dlg", None)
        if dlg is not None:
            for i in issues:
                dlg.add_log(i.text())
            dlg.finish(False, "Analysis failed: " + (issues[0].message if issues else message), len(issues))
        if issues:
            self.show_analysis_issues(title="Analysis failed")
        else:
            QtWidgets.QMessageBox.critical(self, "Analysis failed", message)

    def show_analysis_issues(self, title: str = "Analysis problems"):
        """The problems of the last analysis; double-click locates the object."""
        from .analysis_ui import IssuesDialog
        issues = getattr(self, "_last_issues", [])
        if not issues:
            self.statusBar().showMessage("No problems from the last analysis", 5000)
            return
        if getattr(self, "_issues_dlg", None) is not None:
            self._issues_dlg.close()
        self._issues_dlg = IssuesDialog(issues, self._locate_issue, self, title,
                                        model_path=self.doc.path, model_name=self.doc.model.info.name)
        self._issues_dlg.show()
