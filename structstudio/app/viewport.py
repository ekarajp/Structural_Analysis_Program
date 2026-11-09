"""One model window: a VTK view (plan / 3D / elevation) with its own header.

Several windows share one :class:`~structstudio.app.state.ViewState` (tool,
selection, display options) like the model windows of ETABS / SAP2000.
"""
from __future__ import annotations

import os

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
import vtk
from PySide6 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from . import navigation, scene, theme
from .display import DisplayOptions
from .document import Document
from .icons import icon
from .state import ViewState
from .tools import Plane, Preview, Tool, grid_ordinates, objects_crossing, objects_in_window, objects_on_line, point_in_polygon


SHELL_COMPONENTS = ("F11", "F22", "F12", "M11", "M22", "M12", "V13", "V23")
VIEW_KINDS = [("plan", "Plan"), ("3d", "3D"), ("xz", "Elevation X-Z"), ("yz", "Elevation Y-Z")]
CUSTOM = "__custom__"  # where-box entry: a plan / elevation at a typed coordinate



class _SharpVTKWidget(QVTKRenderWindowInteractor):
    """The VTK view rendered at the real pixel density of the screen the window is on.

    The stock widget takes the scale of the screen under the *mouse*, so with
    two monitors (or Windows display scaling 125 % / 150 %) the picture could be
    drawn at the wrong resolution and stretched - jagged lines and blurred text."""

    def _getPixelRatio(self):  # noqa: N802 (VTK's name)
        try:
            return float(self.devicePixelRatioF())
        except Exception:
            return 1.0

    def event(self, ev):  # moved to a screen of another scale: re-render at its density
        handled = super().event(ev)
        if ev.type() in _DPR_EVENTS:
            self.resizeEvent(QtGui.QResizeEvent(self.size(), self.size()))
        return handled


_DPR_EVENTS = tuple(getattr(QtCore.QEvent.Type, n) for n in ("DevicePixelRatioChange", "ScreenChangeInternal")
                    if hasattr(QtCore.QEvent.Type, n))

def view_dpr(view) -> float:
    """Physical / logical pixels of the window (VTK events and projections are in
    physical pixels; apertures are set in logical, device-independent pixels)."""
    w = getattr(view, "vtk_widget", None)
    try:
        return max(float(w.devicePixelRatioF()), 1.0) if w is not None else 1.0
    except Exception:
        return 1.0

def view_snap_objects(view):
    """(frames, areas, points) a snap may target in this window: what it shows -
    its plan / elevation, minus objects made invisible and object types switched
    off in its display options (None: no limit of that kind)."""
    m = view.doc.model
    vis = None
    try:
        plane_of = getattr(view, "_scene_plane", None)
        vis = scene.visible_sets(m, view.config.story if view.config.kind == "plan" else None,
                                 plane_of() if plane_of else None)
    except Exception:
        vis = None
    hf, ha, hp = getattr(view, "hidden", (set(), set(), set()))
    fr, ar, pt = (set(vis[0]), set(vis[1]), set(vis[2])) if vis else (None, None, None)
    o = getattr(view, "display_options", None)
    ft = o is not None and hasattr(o, "shows_frame_type")
    at = o is not None and hasattr(o, "shows_area_type")
    if hf or (ft and not all(o.shows_frame_type(t) for t in ("Beam", "Column", "Brace", "Null"))):
        fr = {n for n in (fr if fr is not None else m.frames)
              if n not in hf and (not ft or o.shows_frame_type(m.frames[n].type))}
    if ha or (at and not all(o.shows_area_type(t) for t in ("Floor", "Wall", "Ramp", "Opening"))):
        ar = {n for n in (ar if ar is not None else m.areas)
              if n not in ha and (not at or o.shows_area_type(m.areas[n].type))}
    if hp:
        pt = {n for n in (pt if pt is not None else m.points) if n not in hp}
    return fr, ar, pt


def view_work_plane_point(view, x: int, y: int):
    """3D: where a click with nothing snapped lands - the work plane of the form
    (through the last picked point: horizontal XY, or vertical XZ / YZ).  With no
    picked point, or "Snap only", there is no free point in 3D (a depth can not be
    guessed from the screen) (D14)."""
    tool = view.state.tool
    mode = getattr(tool.settings, "work_plane_3d", "Snap only")
    if not getattr(tool, "points", None) or mode.startswith("Snap"):
        return None, ""
    base = np.asarray(tool.points[-1], float)
    k = {"XY": 2, "XZ": 1, "YZ": 0}.get(mode[:2], 2)
    p0, d = view._ray(x, y)
    if abs(d[k]) < 1e-12:
        return None, ""
    t = (base[k] - p0[k]) / d[k]
    if t < 0:
        return None, ""
    p = p0 + t * d
    p[k] = base[k]
    return (float(p[0]), float(p[1]), float(p[2])), f"{mode[:2]} plane {'XYZ'[k]} = {base[k]:g}"


def _hit_key(h):
    """Identity of a snap candidate (kind, object, location) - for keeping an N choice."""
    return (h.kind, h.target, tuple(round(float(c), 9) for c in h.xyz))


class SelectionBox:
    """The drag box of window / crossing selection and Zoom Window, drawn inside the
    render window as a translucent filled rectangle with a border (the selection-box
    colour and opacities of Options > Graphics Colors).  A window selection (drag to
    the right) has a solid border, a crossing selection (drag to the left) a dashed
    one, like CAD programs."""

    def __init__(self, renderer: vtk.vtkRenderer):
        self.renderer = renderer
        self.points = vtk.vtkPoints()
        self.points.SetNumberOfPoints(4)
        quad = vtk.vtkCellArray()
        quad.InsertNextCell(4)
        for k in range(4):
            quad.InsertCellPoint(k)
        fill = vtk.vtkPolyData()
        fill.SetPoints(self.points)
        fill.SetPolys(quad)
        border = vtk.vtkPolyData()
        self.border_poly = border
        self.fill = vtk.vtkActor2D()
        m = vtk.vtkPolyDataMapper2D()
        m.SetInputData(fill)
        self.fill.SetMapper(m)
        self.border = vtk.vtkActor2D()
        m2 = vtk.vtkPolyDataMapper2D()
        m2.SetInputData(border)
        self.border.SetMapper(m2)
        self.border.GetProperty().SetLineWidth(1.5)
        self.visible = False

    def show(self, p0, p1, crossing: bool = False):
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        for k, (x, y) in enumerate(((x0, y0), (x1, y0), (x1, y1), (x0, y1))):
            self.points.SetPoint(k, x, y, 0.0)
        self.points.Modified()
        op = scene.DISPLAY.get("opacity", {})
        color = scene.COLORS.get("select_box", (0.0, 0.47, 0.84))
        self.fill.GetProperty().SetColor(*color)
        self.fill.GetProperty().SetOpacity(op.get("select_box_fill", 0.18))
        self.border.GetProperty().SetColor(*color)
        self.border.GetProperty().SetOpacity(op.get("select_box_border", 0.85))
        # crossing box: dashed border (real dash geometry, stipple is ignored by OpenGL2)
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        seg = []
        for k in range(4):
            seg += [(*corners[k], 0.0), (*corners[(k + 1) % 4], 0.0)]
        if crossing:
            pts, lines = scene.dashed_segments(np.array(seg, float), 8.0)
        else:
            pts = np.array(seg, float)
            lines = vtk.vtkCellArray()
            for k in range(4):
                lines.InsertNextCell(2)
                lines.InsertCellPoint(2 * k)
                lines.InsertCellPoint(2 * k + 1)
        vp = vtk.vtkPoints()
        for p in pts:
            vp.InsertNextPoint(*p)
        self.border_poly.SetPoints(vp)
        self.border_poly.SetLines(lines)
        self.border_poly.Modified()
        if not self.visible:
            self.renderer.AddActor(self.fill)
            self.renderer.AddActor(self.border)
            self.visible = True

    def hide(self) -> bool:
        if not self.visible:
            return False
        self.renderer.RemoveActor(self.fill)
        self.renderer.RemoveActor(self.border)
        self.visible = False
        return True


@dataclass
class ViewConfig:
    """What a window shows: ``kind`` plan | 3d | xz | yz; ``story`` for plans;
    ``grid`` (grid line name) for elevations."""

    kind: str = "3d"
    story: Optional[str] = None
    grid: Optional[str] = None
    results: bool = True  # draw result overlays in this window
    display: dict = field(default_factory=dict)  # DisplayOptions of this window
    # plan at a free elevation (SAP2000 "Set 2D View" at any Z; no story): used when not None
    z: Optional[float] = None
    # elevation at a free ordinate (x for Y-Z views, y for X-Z views)
    ordinate: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ViewConfig":
        return cls(**{k: d[k] for k in ("kind", "story", "grid", "results", "display", "z", "ordinate") if k in d})


class ModelViewport(QtWidgets.QWidget):
    """A model window."""

    activated = QtCore.Signal(object)
    view_changed = QtCore.Signal(object)       # kind / story / grid line changed
    display_requested = QtCore.Signal(object)  # the window's "Display" button
    object_info_requested = QtCore.Signal(str, str)  # right click on an object: (kind, name)
    panel_requested = QtCore.Signal(object)  # a load panel polygon was drawn: list of xyz
    background_menu_requested = QtCore.Signal(object, int, int)  # right click on the background

    PICK_PIXELS = 8.0

    def __init__(self, document: Document, state: Optional[ViewState] = None, config: Optional[ViewConfig] = None,
                 parent=None):
        super().__init__(parent)
        self.doc = document
        self.state = state or ViewState(document)
        self.config = config or ViewConfig()
        self._scene_actors: List[vtk.vtkProp] = []
        self._scene_keys: List[str] = []
        self._overlay_actors: List[vtk.vtkProp] = []
        self._panel_actors: List[vtk.vtkProp] = []
        # parts to redo when this window is shown again ("scene", "selection",
        # "labels", "overlays"): a window that is not on screen builds nothing
        self._dirty: Set[str] = set()
        self._highlight_actors: List[vtk.vtkProp] = []
        self.highlight_story: Optional[str] = None  # story of the plan window, shaded in 3D
        # View > Show Selected Objects Only / Make Selected Objects Invisible (per window)
        self.hidden: Tuple[Set[str], Set[str], Set[str]] = (set(), set(), set())
        self._hidden_history: List[Tuple[Set[str], Set[str], Set[str]]] = []
        self._preview_actors: List[vtk.vtkProp] = []
        self._press: Optional[Tuple[int, int]] = None
        self._snap_hits = []
        self._snap_index = 0  # N key cycles through the snap hits under the cursor
        self._dragging = False
        self._nav: Optional[str] = None  # "orbit" | "pan" | "zoom_window" while a navigation drag runs
        self._nav_last: Optional[Tuple[int, int]] = None
        self._nav_start: Optional[Tuple[int, int]] = None
        self._nav_moved = False
        self._right_press: Optional[Tuple[int, int]] = None
        self._active = False
        self._camera_set = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = QtWidgets.QFrame()
        self.header.setObjectName("viewHeader")
        hl = QtWidgets.QHBoxLayout(self.header)
        hl.setContentsMargins(4, 1, 4, 1)
        hl.setSpacing(4)
        self.kind_box = QtWidgets.QComboBox()
        for key, label in VIEW_KINDS:
            self.kind_box.addItem(icon("view_" + key), label, key)
        self.where_box = QtWidgets.QComboBox()
        self.where_box.setMinimumWidth(110)
        self.where_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.title = QtWidgets.QLabel("")
        self.results_box = QtWidgets.QCheckBox("Results")
        self.results_box.setChecked(self.config.results)
        fit = QtWidgets.QToolButton()
        fit.setText("Fit")
        fit.clicked.connect(self.fit)
        self.display_button = QtWidgets.QToolButton()
        self.display_button.setIcon(icon("display"))
        self.display_button.setToolTip("Set Display Options of this window (Ctrl+W)")
        self.display_button.clicked.connect(lambda: self.display_requested.emit(self))
        # one story / grid line up or down (plans: next story above / below; elevations: the
        # next grid line) - also Page Up / Page Down in the window
        # drawn as light chips with dark arrows: they stay visible on the coloured
        # header of the active window (Qt's arrow buttons took the header colour)
        self.step_up = QtWidgets.QToolButton()
        self.step_up.setObjectName("stepButton")
        self.step_up.setText("\u25B2")
        self.step_up.clicked.connect(lambda: self.step_where(+1))
        self.step_down = QtWidgets.QToolButton()
        self.step_down.setObjectName("stepButton")
        self.step_down.setText("\u25BC")
        self.step_down.clicked.connect(lambda: self.step_where(-1))
        for b in (self.step_up, self.step_down):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setFixedSize(22, 20)
        hl.addWidget(self.kind_box)
        hl.addWidget(self.where_box)
        hl.addWidget(self.step_up)
        hl.addWidget(self.step_down)
        hl.addWidget(self.title, 1)
        hl.addWidget(self.results_box)
        hl.addWidget(self.display_button)
        hl.addWidget(fit)
        layout.addWidget(self.header)

        self.vtk_widget = _SharpVTKWidget(self)
        # anti-aliased lines and edges (multisampling); set before the first render
        self.vtk_widget.GetRenderWindow().SetMultiSamples(8)
        layout.addWidget(self.vtk_widget, 1)
        self.renderer = vtk.vtkRenderer()
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground(*scene.COLORS["background_bottom"])
        self.renderer.SetBackground2(*scene.COLORS["background_top"])
        # optional FXAA post-pass (STRUCTSTUDIO_FXAA=1); the window already asks for
        # multisampled anti-aliasing, and FXAA blanks the view on some software GL drivers
        try:
            if os.environ.get("STRUCTSTUDIO_FXAA", "0") == "1":
                self.renderer.UseFXAAOn()
        except AttributeError:
            pass
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        # CAD ViewCube (3D windows): click a face / edge / corner, drag to orbit
        from .viewcube import CameraAnimator, ViewCube
        self.viewcube = ViewCube(self.vtk_widget.GetRenderWindow(), self.renderer, self._viewcube_pick)
        self.camera_animator = CameraAnimator(self.renderer, self.render)
        _bg = scene.COLORS["background_top"]
        self.viewcube.apply_colors(dark=(0.2126 * _bg[0] + 0.7152 * _bg[1] + 0.0722 * _bg[2]) < 0.5)
        self._cube_press: Optional[Tuple[int, int]] = None
        self._cube_moved = False
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.style)
        # observers on the *style* replace its default handling of these events
        self.style.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self.style.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        self.style.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self.style.AddObserver("RightButtonPressEvent", self._on_right_press)
        self.style.AddObserver("RightButtonReleaseEvent", self._on_right_release)
        self.style.AddObserver("MiddleButtonPressEvent", self._on_middle_press)
        self.style.AddObserver("MiddleButtonReleaseEvent", self._on_middle_release)
        self.style.AddObserver("MouseWheelForwardEvent", lambda o, e: self._on_wheel(+1))
        self.style.AddObserver("MouseWheelBackwardEvent", lambda o, e: self._on_wheel(-1))
        # swallow VTK's built-in keys ('e'/'q' exit, 'w' wireframe, 'r' reset, 'f' fly, ...)
        self.style.AddObserver("CharEvent", lambda o, e: None)
        self.style.AddObserver("KeyPressEvent", lambda o, e: None)
        self.interactor.AddObserver("KeyPressEvent", self._on_key, 1.0)
        self.axes = vtk.vtkAxesActor()
        self.orientation = vtk.vtkOrientationMarkerWidget()
        self.orientation.SetOrientationMarker(self.axes)
        self.orientation.SetInteractor(self.interactor)
        self.orientation.SetViewport(0.0, 0.0, 0.13, 0.18)
        self.orientation.EnabledOn()
        self.orientation.InteractiveOff()
        self.text = vtk.vtkTextActor()
        self.text.GetTextProperty().SetColor(*scene.COLORS["preview"])
        self.text.GetTextProperty().SetFontSize(14)
        self.text.SetPosition(12, 12)
        self.renderer.AddActor(self.text)
        # CAD cursor tooltip: snap, length / angle, typed value
        self.tip = vtk.vtkTextActor()
        tp = self.tip.GetTextProperty()
        tp.SetFontSize(13)
        tp.SetBold(True)
        tp.SetFrame(1)
        tp.SetBackgroundOpacity(0.92)
        self.tip.SetVisibility(0)
        self.renderer.AddActor(self.tip)
        self._style_tip()
        self._typed = ""  # CAD dynamic input being typed (distance, d<angle, dx,dy)
        self._last_cursor = None  # last snapped cursor point (typed distances follow its direction)
        self._last_xy = (0, 0)
        self._drag = None  # select tool: {"mode": "move" | "grip", ...} while dragging objects / a grip
        self._grip_actor = None
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self._rubber = SelectionBox(self.renderer)
        self._line_preview: Optional[vtk.vtkActor2D] = None

        self.kind_box.currentIndexChanged.connect(self._header_changed)
        self.where_box.currentIndexChanged.connect(self._header_changed)
        self.results_box.toggled.connect(self._results_toggled)
        self.doc.on_change(self._on_doc_change)
        self.state.selection_changed.connect(self._on_selection_changed)
        # this window's own result / load display (Display commands act on the active window)
        self.result_display = self.state.window_display() if hasattr(self.state, "window_display") else {}
        self.state.display_changed.connect(self._on_display_changed)
        if hasattr(self.state, "nav_changed"):
            self.state.nav_changed.connect(self._on_nav_changed)
        self._fill_header()
        self.set_active(False)
        self.rebuild()
        self.apply_camera()

    # --------------------------------------------------------- properties
    @property
    def selection(self):
        return self.state.frames

    @property
    def selected_areas(self):
        return self.state.areas

    @property
    def selected_points(self):
        return self.state.points

    @property
    def tool(self):
        return self.state.tool

    @property
    def view_name(self) -> str:
        return self.config.kind

    @property
    def current_story(self) -> Optional[str]:
        return self.config.story

    # ------------------------------------------------------------- header
    def _fill_header(self):
        m = self.doc.model
        c = self.config
        self.kind_box.blockSignals(True)
        self.kind_box.setCurrentIndex(max(0, [k for k, _ in VIEW_KINDS].index(c.kind)))
        self.kind_box.blockSignals(False)
        self.where_box.blockSignals(True)
        self.where_box.clear()
        if c.kind == "plan":
            levels = m.levels()[::-1]
            names = [n for n, _ in levels]
            for n, z in levels:
                self.where_box.addItem(n if m.grid_mode == "story" else f"{n}  (Z={z:g})", n)
            self.where_box.addItem(f"Custom Z = {c.z:g}..." if c.z is not None else "Custom Z...", CUSTOM)
            if c.z is not None:
                self.where_box.setCurrentIndex(self.where_box.count() - 1)
            else:
                if c.story not in names:
                    c.story = names[0] if names else None
                if c.story:
                    self.where_box.setCurrentIndex(names.index(c.story))
        elif c.kind in ("xz", "yz"):
            lines = self._elevation_lines()
            self.where_box.addItems([f"{n}  ({lab})" for n, lab, _ in lines])
            names = [n for n, _, _ in lines]
            axis = "Y" if c.kind == "xz" else "X"
            self.where_box.addItem(f"Custom {axis} = {c.ordinate:g}..." if c.ordinate is not None
                                   else f"Custom {axis}...", CUSTOM)
            if c.ordinate is not None:
                self.where_box.setCurrentIndex(self.where_box.count() - 1)
            else:
                if c.grid not in names:
                    c.grid = names[0] if names else None
                if c.grid:
                    self.where_box.setCurrentIndex(names.index(c.grid))
        self.where_box.setVisible(c.kind != "3d")
        self.where_box.blockSignals(False)
        for b in (self.step_up, self.step_down):
            b.setVisible(c.kind != "3d")
        if c.kind == "plan":
            self.step_up.setToolTip("Story above (Page Up)")
            self.step_down.setToolTip("Story below (Page Down)")
        else:
            self.step_up.setToolTip("Next grid line (Page Up)")
            self.step_down.setToolTip("Previous grid line (Page Down)")
        self._update_title()

    def step_where(self, direction: int) -> bool:
        """Go to the story above (+1) / below (-1) in a plan, or to the next / previous
        grid line in an elevation."""
        c = self.config
        if c.kind == "3d":
            return False
        n = sum(1 for i in range(self.where_box.count()) if self.where_box.itemData(i) != CUSTOM)
        i = self.where_box.currentIndex()
        if self.where_box.itemData(i) == CUSTOM or i < 0:
            i = 0 if (c.kind == "plan") == (direction > 0) else n - 1
        else:
            # plans list the levels top down; elevations the lines in increasing order
            i += -direction if c.kind == "plan" else direction
        if not 0 <= i < n:
            return False
        self.where_box.setCurrentIndex(i)
        return True

    def _elevation_lines(self) -> List[Tuple[str, str, float]]:
        """(grid name, label, ordinate) of the grid lines defining this kind of elevation.
        X-Z elevations lie on Y-direction lines (constant y); Y-Z on X-direction lines."""
        m = self.doc.model
        if self.config.kind == "xz":
            return [(g.name, f"Y={g.ordinate:g}", g.ordinate) for g in m.grid_lines if g.direction == "Y"]
        if self.config.kind == "yz":
            return [(g.name, f"X={g.ordinate:g}", g.ordinate) for g in m.grid_lines if g.direction == "X"]
        return []

    def _update_title(self):
        c = self.config
        m = self.doc.model
        sap = m.grid_mode == "xyz"
        if c.kind == "plan":
            pl = self.plane()
            z = pl.value
            where = pl.story or ("custom" if c.z is not None else "-")
            self.title.setText(f"X-Y Plane @ Z = {z:g}  ({where})" if sap
                               else f"Plan View - {where} - Z = {z:g}")
        elif c.kind in ("xz", "yz"):
            pl = self.plane()
            axis = "Y" if c.kind == "xz" else "X"
            where = c.grid or ("custom" if c.ordinate is not None else "-")
            self.title.setText(f"{c.kind.upper()[0]}-Z Plane @ {axis} = {pl.value:g}  ({where})" if sap
                               else f"Elevation View - {where} @ {axis} = {pl.value:g}")
        else:
            self.title.setText("3-D View")

    def _header_changed(self, *_):
        kind = self.kind_box.currentData()
        c = self.config
        changed_kind = kind != c.kind
        c.kind = kind
        if not changed_kind:
            data = self.where_box.currentData()
            if data == CUSTOM:
                if not self._ask_custom_coordinate():
                    self._fill_header()
                    return
            elif kind == "plan":
                c.story = data or self.where_box.currentText() or None
                c.z = None
            elif kind in ("xz", "yz"):
                lines = self._elevation_lines()
                i = self.where_box.currentIndex()
                c.grid = lines[i][0] if 0 <= i < len(lines) else None
                c.ordinate = None
        self._fill_header()
        self.state.tool.cancel()
        self.invalidate("scene", "overlays")
        self.apply_camera()
        self.activate()
        self.view_changed.emit(self)

    def _ask_custom_coordinate(self) -> bool:
        """SAP2000 "Set 2D View": a plan at any Z, an elevation at any X / Y."""
        c = self.config
        if c.kind == "plan":
            cur = c.z if c.z is not None else self.plane().value
            label = "Plan view at Z ="
        else:
            cur = c.ordinate if c.ordinate is not None else self.plane().value
            label = ("Elevation at Y =" if c.kind == "xz" else "Elevation at X =")
        v, ok = QtWidgets.QInputDialog.getDouble(self, "Set 2D View", label, float(cur), -1e9, 1e9, 4)
        if not ok:
            return False
        self.set_custom_coordinate(v)
        return True

    def set_custom_coordinate(self, value: float):
        c = self.config
        if c.kind == "plan":
            c.z = float(value)
            st = self.doc.model.story_at(c.z, max(self.doc.model.options.merge_tolerance, 1e-6))
            c.story = st.name if st else None
            if st is not None:
                c.z = None  # on a level: the ordinary story plan
        elif c.kind in ("xz", "yz"):
            c.ordinate = float(value)
            c.grid = None

    def _results_toggled(self, on: bool):
        self.config.results = on
        self.update_overlays()

    def set_config(self, config: ViewConfig):
        self.config = config
        self.results_box.blockSignals(True)
        self.results_box.setChecked(config.results)
        self.results_box.blockSignals(False)
        self._fill_header()
        self.invalidate("scene", "overlays")
        self.apply_camera()
        self.view_changed.emit(self)

    # ------------------------------------------------------------ display
    @property
    def display_options(self) -> DisplayOptions:
        return DisplayOptions.from_dict(self.config.display)

    # options whose change only touches the text labels / also the results drawn
    LABEL_OPTIONS = frozenset({"frame_labels", "joint_labels", "area_labels", "section_names"})
    OVERLAY_OPTIONS = frozenset({"extrude", "releases", "analysis_mesh"})

    def set_display_options(self, options: DisplayOptions):
        old = DisplayOptions.from_dict(self.config.display).to_dict()  # same keys on both sides
        new = options.to_dict()
        self.config.display = new
        changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
        if not changed:
            return
        if changed <= self.LABEL_OPTIONS:
            self.invalidate("labels")  # labels only: the objects and results stay
            return
        parts = ["scene"]
        if changed & self.OVERLAY_OPTIONS:
            parts.append("overlays")
        self.invalidate(*parts)

    def set_highlight_story(self, story: Optional[str]):
        """3D windows: shade the plane of ``story`` (the story of the plan window)."""
        if story == self.highlight_story and self._highlight_actors:
            return
        self.highlight_story = story
        self._draw_highlight()
        self.render()

    def _draw_highlight(self):
        for a in self._highlight_actors:
            self.renderer.RemoveActor(a)
        self._highlight_actors = []
        if self.config.kind == "3d" and self.display_options.story_plane and self.highlight_story:
            self._highlight_actors = scene.build_story_plane(self.doc.model, self.highlight_story)
            for a in self._highlight_actors:
                self.renderer.AddActor(a)

    def set_view(self, kind: str, story: Optional[str] = None, grid: Optional[str] = None,
                 z: Optional[float] = None, ordinate: Optional[float] = None):
        self.config.kind = kind
        if story:
            self.config.story = story
            self.config.z = None
        if grid:
            self.config.grid = grid
            self.config.ordinate = None
        if z is not None:
            self.set_custom_coordinate(z)
        if ordinate is not None and kind in ("xz", "yz"):
            self.config.ordinate = float(ordinate)
            self.config.grid = None
        self._fill_header()
        self.invalidate("scene", "overlays")
        self.apply_camera()
        self.view_changed.emit(self)

    def set_story(self, story: Optional[str]):
        self.config.story = story
        self._fill_header()
        self.invalidate("scene", "overlays")
        if self.config.kind == "plan":
            self.apply_camera()
        self.view_changed.emit(self)

    # ------------------------------------------------------------- active
    def set_active(self, on: bool):
        self._active = on
        ui = theme.active().ui()
        color = ui["accent"] if on else ui["header_off"]
        fg = ui["accent_text"] if on else ui["header_off_text"]
        self.header.setStyleSheet(
            f"#viewHeader {{ background: {color}; }} QLabel, QCheckBox {{ color: {fg}; }}"
            "QToolButton#stepButton { background: #f7f9fb; color: #1f2a36; border: 1px solid #8a96a3;"
            " border-radius: 3px; font-size: 9pt; font-weight: 700; padding: 0px; }"
            "QToolButton#stepButton:hover { background: #ffffff; border-color: #33475b; }"
            "QToolButton#stepButton:pressed { background: #d8e2ec; }"
            "QToolButton#stepButton:disabled { color: #a9b2bb; background: #e9edf1; }")

    def apply_theme(self):
        """Re-read the active colours (theme switch / colour edit) and redraw."""
        self.renderer.SetBackground(*scene.COLORS["background_bottom"])
        self.renderer.SetBackground2(*scene.COLORS["background_top"])
        self.text.GetTextProperty().SetColor(*scene.COLORS["preview"])
        bg = scene.COLORS["background_top"]
        self.viewcube.apply_colors(dark=(0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]) < 0.5)
        self._style_tip()
        for i in range(self.kind_box.count()):
            self.kind_box.setItemIcon(i, icon("view_" + self.kind_box.itemData(i)))
        self.set_active(self._active)
        self._clear_preview()
        self.invalidate("scene", "overlays")

    def _style_tip(self):
        tp = self.tip.GetTextProperty()
        bg = scene.COLORS["background_top"]
        dark = (0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]) < 0.5
        tp.SetColor(*((0.95, 0.96, 0.98) if dark else (0.10, 0.12, 0.16)))
        tp.SetBackgroundColor(*((0.20, 0.22, 0.26) if dark else (1.0, 1.0, 0.94)))
        tp.SetFrameColor(*scene.COLORS["snap"])

    def activate(self, focus: bool = True):
        self.activated.emit(self)
        if focus and not self.vtk_widget.hasFocus():
            self.vtk_widget.setFocus(QtCore.Qt.MouseFocusReason)  # Esc / Enter / typed lengths come here

    # ------------------------------------------------------------- planes
    def plane(self) -> Plane:
        c = self.config
        m = self.doc.model
        if c.kind == "plan":
            if c.z is not None:  # free plan: a story only when a level sits exactly there
                st = m.story_at(c.z, max(m.options.merge_tolerance, 1e-6))
                return Plane("plan", float(c.z), st.name if st else None)
            z = m.stories[c.story].elevation if c.story in m.stories else 0.0
            return Plane("plan", z, c.story)
        if c.kind in ("xz", "yz"):
            if c.ordinate is not None:
                return Plane(c.kind, float(c.ordinate))
            for name, _, ordinate in self._elevation_lines():
                if name == c.grid:
                    return Plane(c.kind, ordinate)
            return Plane(c.kind, 0.0)
        cam = self.renderer.GetActiveCamera()
        to_eye = tuple(float(v) for v in np.subtract(cam.GetPosition(), cam.GetFocalPoint()))
        return Plane("3d", 0.0, normal=to_eye)

    def _scene_plane(self):
        """Scene filter plane: ("x", x0) = plane x = x0 (Y-Z view); ("y", y0) for X-Z views."""
        p = self.plane()
        if p.kind == "xz":
            return ("y", p.value)
        if p.kind == "yz":
            return ("x", p.value)
        if p.kind == "plan" and p.story is None:  # free plan at a custom Z
            return ("z", p.value)
        return None

    def visible(self):
        base = scene.make_visibility(self.doc.model, self.config.story if self.config.kind == "plan" else None,
                                     self._scene_plane())
        hf, ha, _ = self.hidden
        if not (hf or ha):
            return base
        return lambda obj: base(obj) and obj.name not in (hf if hasattr(obj, "point_i") else ha)

    # ------------------------------------------------- object visibility (View menu)
    def _set_hidden(self, frames, areas, points):
        self._hidden_history.append(tuple(set(x) for x in self.hidden))
        self.hidden = (set(frames), set(areas), set(points))
        self.invalidate("scene", "overlays")

    def make_selected_invisible(self):
        st = self.state
        hf, ha, hp = self.hidden
        self._set_hidden(hf | st.frames, ha | st.areas, hp | st.points)
        st.clear_selection()

    def show_selected_only(self):
        st, m = self.state, self.doc.model
        if not (st.frames or st.areas or st.points):
            return
        keep_pts = set(st.points)
        for n in st.frames:
            f = m.frames[n]
            keep_pts |= {f.point_i, f.point_j}
        for n in st.areas:
            keep_pts |= set(m.areas[n].points)
        self._set_hidden(set(m.frames) - st.frames, set(m.areas) - st.areas, set(m.points) - keep_pts)

    def _keep_sets(self, frames, areas, points):
        m = self.doc.model
        keep_pts = set(points)
        for n in frames:
            f = m.frames[n]
            keep_pts |= {f.point_i, f.point_j}
        for n in areas:
            keep_pts |= set(m.areas[n].points)
        return keep_pts

    def show_only(self, kind: str, name: str):
        """Right-click > Show only this object (the rest hidden in this window)."""
        m = self.doc.model
        fr = {name} if kind == "frame" else set()
        ar = {name} if kind == "area" else set()
        pt = {name} if kind == "point" else set()
        keep = self._keep_sets(fr, ar, pt)
        self._set_hidden(set(m.frames) - fr, set(m.areas) - ar, set(m.points) - keep)

    def hide_object(self, kind: str, name: str):
        hf, ha, hp = self.hidden
        self._set_hidden(hf | ({name} if kind == "frame" else set()), ha | ({name} if kind == "area" else set()),
                         hp | ({name} if kind == "point" else set()))

    def object_menu(self, kind: str, name: str, x: int, y: int):
        """Right-click on an object: Information (ETABS), and what to show."""
        st = self.state
        what = {"frame": "Frame", "area": "Shell", "point": "Joint", "panel": "Load panel"}.get(kind, kind)
        menu = QtWidgets.QMenu(self.vtk_widget)
        info = menu.addAction(f"{what} {name} - Information...")
        f = menu.font()
        f.setBold(True)
        info.setFont(f)
        menu.addSeparator()
        sel = menu.addAction("Select it")
        only = menu.addAction("Show only this object")
        sel_only = menu.addAction("Show only the selected objects") if (st.frames or st.areas or st.points) else None
        hide = menu.addAction("Hide this object")
        show_all = menu.addAction("Show all objects") if any(self.hidden) else None
        menu.addSeparator()
        axes = menu.addAction("Show local axes (1 red, 2 green, 3 blue)")
        axes.setCheckable(True)
        axes.setChecked(self.display_options.local_axes)
        h = self.vtk_widget.height()
        r = self.vtk_widget.devicePixelRatioF() or 1.0
        a = menu.exec(self.vtk_widget.mapToGlobal(QtCore.QPoint(int(x / r), int(h - y / r))))
        if a is None:
            return
        if a is info:
            self.object_info_requested.emit(kind, name)
        elif a is sel and kind in ("frame", "area", "point"):
            st.toggle(kind, name, True)
        elif a is only and kind in ("frame", "area", "point"):
            self.show_only(kind, name)
        elif sel_only is not None and a is sel_only:
            self.show_selected_only()
        elif a is hide and kind in ("frame", "area", "point"):
            self.hide_object(kind, name)
        elif show_all is not None and a is show_all:
            self.show_all_objects()
        elif a is axes:
            o = self.display_options
            o.local_axes = not o.local_axes
            self.set_display_options(o)

    def invert_visibility(self):
        m = self.doc.model
        hf, ha, hp = self.hidden
        self._set_hidden(set(m.frames) - hf, set(m.areas) - ha, set(m.points) - hp)

    def restore_previous_visibility(self):
        if self._hidden_history:
            self.hidden = self._hidden_history.pop()
            self.invalidate("scene", "overlays")

    def show_all_objects(self):
        if any(self.hidden):
            self._set_hidden((), (), ())

    # ------------------------------------------------------------ building
    def _on_doc_change(self, what: str):
        if what == "units":  # same view of the structure: the coordinates were scaled
            r = float(getattr(self.doc, "units_ratio", 1.0))
            cam = self.renderer.GetActiveCamera()
            cam.SetPosition(*(np.array(cam.GetPosition()) * r))
            cam.SetFocalPoint(*(np.array(cam.GetFocalPoint()) * r))
            cam.SetParallelScale(cam.GetParallelScale() * r)
            self.renderer.ResetCameraClippingRange()
            return
        if what == "model":
            self._fill_header()
            self.invalidate("scene", "overlays")
            if not self._camera_set:
                self.apply_camera()
            else:  # new / resized objects (e.g. after a change of units): nothing clipped away
                self.renderer.ResetCameraClippingRange()
                self.render()
        elif what == "results":
            self.invalidate("overlays")

    def _on_display_changed(self):
        """A Display command (the toolbar's case / diagram / deformed ...): the active
        window takes it; the other windows keep showing what they show."""
        if getattr(self, "_active", False) and hasattr(self.state, "window_display"):
            self.result_display = self.state.window_display()
        elif self.result_display is not None:
            return  # this window keeps its own display: nothing it shows has changed
        # a Display command changes results / loads only - the model actors stay; a
        # window that is not on screen catches up when it is shown again
        self.invalidate("overlays")

    # ------------------------------------------------------- what to redo
    def invalidate(self, *parts: str):
        """Redo ``parts`` of the picture now, or when the window is shown again if
        it is not on screen: "scene" (objects - with selection and labels),
        "selection", "labels", "overlays" (results / loads)."""
        self._dirty.update(parts)
        if self.isVisible():
            self.flush()

    def flush(self):
        """Redo what is marked (once, whatever number of changes marked it)."""
        parts, self._dirty = self._dirty, set()
        if not parts:
            return
        if "scene" in parts:
            self._build_scene()
        else:
            if "selection" in parts:
                self._build_layer("selection")
            if "labels" in parts:
                self._build_layer("labels")
        if "overlays" in parts:
            self.update_overlays()
        else:
            self._style_scene()
            if "selection" in parts or "scene" in parts:
                self._build_panels()
            self.render()

    def refresh(self):
        """View > Refresh: everything again (when shown)."""
        self.invalidate("scene", "overlays")

    def _on_selection_changed(self):
        self.invalidate("selection")

    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty:
            self.flush()

    GHOST_OPACITY = 0.18

    def _ghost_scene(self, on, prefixes=None):
        """Deformed shape / mode shape shown: the undeformed model stays as a faint
        shadow (grid lines and labels unchanged).  ``prefixes`` limits it to the
        objects a result is drawn on (``areas`` under a shell contour, ``frames``
        under frame colours)."""
        store = getattr(self, "_ghost_orig", None)
        if store is None:
            store = self._ghost_orig = {}
        for key, a in zip(getattr(self, "_scene_keys", []), self._scene_actors):
            if not isinstance(a, vtk.vtkActor) or key.startswith("grid"):
                continue
            prop = a.GetProperty()
            k = id(a)
            if on and (prefixes is None or key.startswith(tuple(prefixes))):
                if k not in store:
                    store[k] = prop.GetOpacity()
                    prop.SetOpacity(min(store[k], self.GHOST_OPACITY))
            elif k in store:
                prop.SetOpacity(store.pop(k))

    def _hide_scene(self, prefixes):
        store = getattr(self, "_hidden_orig", None)
        if store is None:
            store = self._hidden_orig = {}
        for key, a in zip(getattr(self, "_scene_keys", []), self._scene_actors):
            k = id(a)
            if prefixes and key.startswith(tuple(prefixes)) and not key.endswith(("_sel", "_sel_lines", "_sel_tubes")):
                if k not in store:
                    store[k] = a.GetVisibility()
                    a.SetVisibility(False)
            elif k in store:
                a.SetVisibility(store.pop(k))

    def _actor_keys(self):
        return list(getattr(self, "_scene_keys", []))

    def rebuild(self):
        """The whole picture again, now (objects, selection, labels, results)."""
        self._dirty = set()
        self._build_scene()
        self.update_overlays()

    def _model_actors(self, layers):
        st = self.state
        story = self.config.story if self.config.kind == "plan" else None
        return scene.build_model_actors(self.doc.model, st.show_labels, st.extruded, st.frames, story=story,
                                        selected_areas=st.areas, selected_points=st.points,
                                        plane=self._scene_plane(), options=self.display_options,
                                        hidden=self.hidden, layers=layers)

    def _build_scene(self):
        for a in self._scene_actors:
            self.renderer.RemoveActor(a)
        self._scene_actors = []
        self._scene_keys = []
        self._ghost_orig = {}
        self._hidden_orig = {}
        sc = self._model_actors(scene.MODEL_LAYERS)
        for key, actor in sc.actors.items():
            self.renderer.AddActor(actor)
            self._scene_keys.append(key)
            self._scene_actors.append(actor)
        self._draw_highlight()
        self._draw_grips()

    def _build_layer(self, layer: str):
        """Only the selection highlights / only the labels again: the objects (and
        the results) stay as they are."""
        keep_k, keep_a = [], []
        for key, a in zip(self._scene_keys, self._scene_actors):
            if scene.model_layer_of(key) == layer:
                self.renderer.RemoveActor(a)
                self._ghost_orig.pop(id(a), None)
                self._hidden_orig.pop(id(a), None)
            else:
                keep_k.append(key)
                keep_a.append(a)
        self._scene_keys, self._scene_actors = keep_k, keep_a
        sc = self._model_actors((layer,))
        for key, actor in sc.actors.items():
            self.renderer.AddActor(actor)
            self._scene_keys.append(key)
            self._scene_actors.append(actor)
        if layer == "selection":
            self._draw_grips()

    # ------------------------------------------------------------- grips
    # Dragging the selection / its joints with the select tool (AutoCAD grips) is off:
    # a click on a selected member only selects, the cursor stays an arrow, and objects
    # move or copy only through Edit > Move / Copy (ETABS).
    DIRECT_EDIT = False
    GRIP_LIMIT = 400  # grips are shown for selections up to this many joints
    GRIP_PIXELS = 7.0

    def _grip_points(self) -> List[str]:
        """Joints of the selected frames / shells (and selected joints) that can be dragged."""
        st = self.state
        if not self.DIRECT_EDIT or st.tool.name != "select" or not (st.frames or st.areas or st.points):
            return []
        m = self.doc.model
        names = []
        for n in st.frames:
            f = m.frames.get(n)
            if f is not None:
                names += [f.point_i, f.point_j]
        for n in st.areas:
            a = m.areas.get(n)
            if a is not None:
                names += list(a.points)
        names += [n for n in st.points if n in m.points]
        names = list(dict.fromkeys(names))
        if len(names) > self.GRIP_LIMIT:
            return []
        if self.config.kind != "3d":
            pl = self.plane()
            tol = max(m.options.merge_tolerance, 1e-6) * 10
            names = [n for n in names if pl.contains(m.points[n].xyz, tol)]
        return names

    def _draw_grips(self):
        if self._grip_actor is not None:
            self.renderer.RemoveActor(self._grip_actor)
            self._grip_actor = None
        names = self._grip_points()
        if not names:
            return
        m = self.doc.model
        pts = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        for k, n in enumerate(names):
            pts.InsertNextPoint(*m.points[n].xyz)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(k)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetVerts(verts)
        actor = scene._actor_from_polydata(poly, scene.COLORS["select_box"], lighting=False)
        actor.GetProperty().SetPointSize(9)
        actor.GetProperty().SetRenderPointsAsSpheres(False)
        self._grip_actor = actor
        self.renderer.AddActor(actor)

    def _grip_at(self, x: int, y: int) -> Optional[str]:
        m = self.doc.model
        best, best_d = None, self.GRIP_PIXELS
        for n in self._grip_points():
            dx, dy = self._display(m.points[n].xyz)
            d = math.hypot(dx - x, dy - y)
            if d < best_d:
                best, best_d = n, d
        return best

    def _on_selection(self, x: int, y: int) -> bool:
        st = self.state
        for kind, name in self.pick_all(x, y):
            if name in {"frame": st.frames, "area": st.areas, "point": st.points, "panel": st.panels}.get(kind, ()):
                return True
        return False

    def _frame_under(self, x: int, y: int) -> Optional[str]:
        """The frame nearest to display (x, y) within the pick tolerance (joints ignored)."""
        m = self.doc.model
        vis = self.visible()
        best, best_d = None, self.PICK_PIXELS * 1.5 * view_dpr(self)
        for f in m.frames.values():
            if not vis(f):
                continue
            ax, ay = self._display(m.points[f.point_i].xyz)
            bx, by = self._display(m.points[f.point_j].xyz)
            L2 = (bx - ax) ** 2 + (by - ay) ** 2
            if L2 <= 0:
                continue
            t = max(0.0, min(1.0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / L2))
            d = math.hypot(ax + t * (bx - ax) - x, ay + t * (by - ay) - y)
            if d < best_d:
                best, best_d = f.name, d
        return best

    # ------------------------------------------------ drag objects / grips (select tool)
    def _drag_update(self, x: int, y: int):
        """Ghost of the dragged selection (or of the members at a dragged grip)."""
        from .modify import ghost_lines, translation
        dr = self._drag
        st = self.state
        st.tool.points = [dr["base"]]  # polar / alignment tracking from the base point
        try:
            pt, snap = self._snapped(x, y)
        finally:
            st.tool.points = []
        self._clear_preview()
        if pt is None:
            self.render()
            return
        dr["to"] = pt
        d = np.subtract(pt, dr["base"])
        m = self.doc.model
        if dr["mode"] == "move":
            lines = ghost_lines(m, st.frames, st.areas, st.points, st.panels, translation(d))
            copy = bool(self.interactor.GetControlKey())
            text = ("Copy" if copy else "Move") + f"  L = {float(np.linalg.norm(d)):.3f}"
        else:
            g = dr["grip"]
            lines = []
            for f in m.frames.values():
                if g in (f.point_i, f.point_j):
                    other = f.point_j if f.point_i == g else f.point_i
                    lines.append((tuple(pt), m.points[other].xyz))
            for a in m.areas.values():
                if g in a.points:
                    q = [tuple(pt) if n == g else m.points[n].xyz for n in a.points]
                    lines += [(q[k], q[(k + 1) % len(q)]) for k in range(len(q))]
            text = f"Joint {g}  L = {float(np.linalg.norm(d)):.3f}"
        from .tools import Preview
        pv = Preview(lines=lines, marker=pt, text=text, guides=[(dr["base"], tuple(pt))])
        self._draw_preview(pv, snap, x, y)

    def _drag_finish(self, x: int, y: int):
        from .modify import copy_transformed, move_joint, transform_in_place, translation
        dr, self._drag = self._drag, None
        st = self.state
        self._clear_preview()
        to = dr.get("to")
        if not dr.get("moved") or to is None:
            return False
        d = np.subtract(to, dr["base"])
        if float(np.linalg.norm(d)) < 1e-12:
            self.render()
            return True
        try:
            if dr["mode"] == "grip":
                name = move_joint(self.doc, dr["grip"], tuple(to))
                st.status.emit(f"Joint {dr['grip']} moved" + (f" onto {name}" if name != dr["grip"] else ""))
            elif self.interactor.GetControlKey():
                fr, ar, pts = copy_transformed(self.doc, list(st.frames), list(st.areas), list(st.points),
                                               translation(d), panels=list(st.panels))
                st.set_selection(fr, ar, () if (fr or ar) else pts)
                st.status.emit(f"Copied {len(fr)} frame(s), {len(ar)} shell(s)")
            else:
                transform_in_place(self.doc, list(st.frames), list(st.areas), list(st.points), translation(d),
                                   panels=list(st.panels), stretch=st.settings.modify_stretch)
                st.status.emit("Moved (Ctrl while dragging = copy)")
        except Exception as exc:
            st.report(exc)
        self.render()
        return True

    def _result_display(self):
        st = self.state
        from types import SimpleNamespace
        rd = SimpleNamespace(**{k: getattr(st, k, None) for k in getattr(st, "WINDOW_DISPLAY", ())})
        for k, v in (self.result_display or {}).items():
            setattr(rd, k, v)
        return rd

    def _result_flags(self, rd):
        res = self.doc.results
        shown = bool(res is not None and rd.display_case and self.config.results)
        fcol = getattr(rd, "display_frame_color", None) if shown and rd.display_mode is None else None
        contour_on = shown and rd.display_diagram in SHELL_COMPONENTS and rd.display_mode is None
        return shown, fcol, contour_on

    def _style_scene(self, rd=None):
        """The objects under a result: a faint shadow under a deformed shape, hidden
        under a contour / member colours (also for actors made after the result)."""
        rd = rd or self._result_display()
        shown, fcol, contour_on = self._result_flags(rd)
        if shown and (rd.deformed or rd.display_mode is not None):
            self._ghost_scene(False)
            self._ghost_scene(True)
        else:
            self._ghost_scene(False)
        # objects a result is drawn on are hidden (a faint copy in the same place
        # would wash the colours out): shells under a contour, members under colours
        hide = ((("areas",) if contour_on and not rd.deformed else ()) + (("frames", "column_markers") if fcol and not rd.deformed else ()))
        if shown and (rd.deformed or rd.display_mode is not None):
            hide += ("releases",)  # the pin symbols are drawn on the deflected members instead
        self._hide_scene(hide)

    def _build_panels(self, rd=None):
        """Load panels (a selected one with its outline): shown with the plain
        model - they would veil a contour / member colours."""
        for a in self._panel_actors:
            self.renderer.RemoveActor(a)
        self._panel_actors = []
        rd = rd or self._result_display()
        _shown, fcol, contour_on = self._result_flags(rd)
        st = self.state
        try:
            if not (contour_on or fcol) or getattr(st, "panels", None):
                self._panel_actors = scene.build_panel_actors(self.doc.model, getattr(st, "panels", None))
        except Exception as exc:
            print("panel display error:", exc)
        for a in self._panel_actors:
            self.renderer.AddActor(a)

    def update_overlays(self):
        for a in self._overlay_actors:
            self.renderer.RemoveActor(a)
        self._overlay_actors = []
        self._dirty.discard("overlays")
        st = self.state
        rd = self._result_display()
        res = self.doc.results
        extr = bool(getattr(self.display_options, "extrude", False))
        shown, fcol, contour_on = self._result_flags(rd)
        self._style_scene(rd)
        if getattr(self.display_options, "analysis_mesh", False) and res is not None:
            # Set Display Options > Analysis mesh: the mesh the analysis used (nothing
            # before an analysis)
            try:
                _, ar, _ = self._view_sets()
                self._overlay_actors += scene.build_mesh_actors(self.doc.model, res.am, areas=ar)
            except Exception as exc:
                print("mesh display error:", exc)
        if res is not None and rd.display_case and self.config.results:
            try:
                # a plan / elevation window shows the results of its own plan / frame line only
                fr, ar, pt = self._view_sets()
                if rd.deformed or rd.display_mode is not None:
                    # the shell contour (if shown) is already drawn deformed: then only the frames
                    contour = rd.display_diagram in SHELL_COMPONENTS and rd.display_mode is None
                    actors, _ = scene.build_deformed_mesh_actors(self.doc.model, res, rd.display_case,
                                                                 mode=rd.display_mode, frames=fr, areas=ar,
                                                                 shells=not contour, extruded=extr,
                                                                 releases=bool(getattr(self.display_options,
                                                                                       "releases", True)))
                    if contour:
                        actors = [a for a in actors if not isinstance(a, vtk.vtkScalarBarActor)]
                    self._overlay_actors += actors
                if rd.display_diagram and rd.display_mode is None:
                    if rd.display_diagram in SHELL_COMPONENTS:
                        actors, _ = scene.build_contour_actors(self.doc.model, res, rd.display_case,
                                                               rd.display_diagram, deformed=rd.deformed, areas=ar,
                                                               extruded=extr)
                    else:
                        dfr = fr
                        if self.config.kind == "plan" and fr is not None:  # columns are seen end-on in a plan
                            dfr = {n for n in fr if self.doc.model.frames[n].type != "Column"}
                        actors, _ = scene.build_diagram_actors(
                            self.doc.model, res, rd.display_case, rd.display_diagram, frames=dfr,
                            view_normal=scene.VIEW_NORMALS.get(self.config.kind))
                    self._overlay_actors += actors
                if fcol and not rd.deformed:
                    from .result_view import frame_color_actors
                    actors = frame_color_actors(self.doc.model, res, rd.display_case, fcol, frames=fr,
                                                extruded=extr)
                    if rd.display_diagram in SHELL_COMPONENTS:  # one colour scale on screen: the contour's
                        actors = [a for a in actors if not isinstance(a, vtk.vtkScalarBarActor)]
                    self._overlay_actors += actors
                if rd.show_reactions and rd.display_mode is None:
                    self._overlay_actors += scene.build_reaction_actors(self.doc.model, res, rd.display_case,
                                                                        points=pt)
            except Exception as exc:  # display must never crash the app
                print("overlay error:", exc)
        self._build_panels(rd)
        if rd.load_pattern and rd.load_pattern in self.doc.model.load_patterns:
            try:
                kind = self.config.kind
                fr, ar, pt = self._view_sets()
                self._overlay_actors += scene.build_load_actors(
                    self.doc.model, rd.load_pattern, frames=fr, areas=ar, points=pt,
                    kinds=getattr(rd, "load_kinds", None), view=kind if kind in scene.VIEW_NORMALS else "3d",
                    level=self.plane().value if kind in scene.VIEW_NORMALS else None)
            except Exception as exc:
                print("load display error:", exc)
        for a in self._overlay_actors:
            self.renderer.AddActor(a)
        self._make_legend_room(any(isinstance(a, vtk.vtkScalarBarActor) for a in self._overlay_actors))
        self.render()

    LEGEND_ROOM = 0.07  # the model moves left by this fraction of the width when a colour scale is shown

    def _make_legend_room(self, on: bool, force: bool = False):
        """Pan the model left while a colour scale is on the right edge (and back
        when it goes), so the scale does not cover the model."""
        if on == getattr(self, "_legend_room", False) and not force:
            return
        self._legend_room = on
        w = max(self.vtk_widget.width(), 1)
        h = max(self.vtk_widget.height(), 1)
        dx = self.LEGEND_ROOM * w * (1 if on else -1)
        try:
            navigation.pan(self.renderer, w / 2, h / 2, w / 2 - dx, h / 2)
        except Exception:
            pass

    def _view_sets(self):
        """(frames, areas, points) this window shows - its story plan or frame line,
        without the objects made invisible - or (None, None, None) for everything."""
        kind = self.config.kind
        vis = scene.visible_sets(self.doc.model, self.config.story if kind == "plan" else None, self._scene_plane())
        hf, ha, hp = self.hidden
        if vis is None:
            if not (hf or ha or hp):
                return None, None, None
            m = self.doc.model
            vis = (set(m.frames), set(m.areas), set(m.points))
        fr, ar, pt = vis
        return fr - hf, ar - ha, pt - hp

    def render(self):
        if self.isVisible() or not self._camera_set:
            cube = getattr(self, "viewcube", None)
            if cube is not None:
                show = self.config.kind == "3d"
                if show != cube.visible:
                    cube.set_visible(show)
                cube.sync()
            self.vtk_widget.GetRenderWindow().Render()

    # ----------------------------------------------------------- viewcube
    def _viewcube_pick(self, key):
        from .viewcube import describe, view_for
        d, up = view_for(key)
        self.camera_animator.go(d, up, animate=self.isVisible())
        self.state.status.emit(f"View: {describe(key)}")

    def view_from(self, key, animate: bool = False):
        """Turn the 3D camera to a ViewCube piece (e.g. (0, -1, 0) = Front, (1, -1, 1) = iso)."""
        from .viewcube import view_for
        d, up = view_for(tuple(int(v) for v in key))
        self.camera_animator.go(d, up, animate=animate)

    def _cube_event(self, kind: str, x: int, y: int) -> bool:
        """Mouse events over the ViewCube (3D windows); True when the cube took the event."""
        cube = self.viewcube
        if self.config.kind != "3d" or not cube.visible:
            return False
        if kind == "press":
            if not cube.contains(x, y):
                return False
            self._cube_press, self._cube_moved = (x, y), False
            return True
        if kind == "move":
            if self._cube_press is not None:
                lx, ly = self._cube_press
                if self._cube_moved or abs(x - lx) > 3 or abs(y - ly) > 3:
                    self._cube_moved = True
                    navigation.orbit(self.renderer, (x - lx) * navigation.ROTATE_DEG_PER_PIXEL * 1.5,
                                     -(y - ly) * navigation.ROTATE_DEG_PER_PIXEL * 1.5)
                    self._cube_press = (x, y)
                    self.render()
                return True
            changed = cube.hover(x, y)
            if changed:
                self.render()
            if cube.contains(x, y):
                self.vtk_widget.setToolTip("ViewCube: click a face, edge or corner for that view; drag to rotate")
                return True
            self.vtk_widget.setToolTip("")
            return False
        if kind == "release":
            if self._cube_press is None:
                return False
            moved, self._cube_press = self._cube_moved, None
            if not moved:
                cube.click(x, y)
            return True
        return False

    # ------------------------------------------------------------- camera
    def apply_camera(self):
        kind = self.config.kind
        scene.set_view(self.renderer, {"plan": "plan", "xz": "xz", "yz": "yz"}.get(kind, "3d"))
        cam = self.renderer.GetActiveCamera()
        p = self.plane()
        fx, fy, fz = cam.GetFocalPoint()
        if kind == "plan":
            cam.SetFocalPoint(fx, fy, p.value)
            cam.SetPosition(fx, fy, p.value + 1000.0 * max(scene.model_extent(self.doc.model), 1.0))
        elif kind == "xz":
            cam.SetFocalPoint(fx, p.value, fz)
            cam.SetPosition(fx, p.value - 1000.0 * max(scene.model_extent(self.doc.model), 1.0), fz)
        elif kind == "yz":
            cam.SetFocalPoint(p.value, fy, fz)
            cam.SetPosition(p.value + 1000.0 * max(scene.model_extent(self.doc.model), 1.0), fy, fz)
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        if kind != "3d":
            cam.Zoom(1.1)
        self._camera_set = True
        if getattr(self, "_legend_room", False):  # ResetCamera centred the model again
            self._legend_room = False
            self._make_legend_room(True)
        self.render()

    def fit(self):
        self.renderer.ResetCamera()
        if getattr(self, "_legend_room", False):
            self._legend_room = False
            self._make_legend_room(True)
        self.render()

    # ------------------------------------------------------- locate an object
    FLASH_COUNT = 6  # on/off toggles
    FLASH_MS = 260
    FLASH_DOT_PX = 14  # end dots of the flash, constant on screen
    FLASH_RGB = (1.0, 0.1, 0.65)  # magenta: unlike the selection blue and every object colour

    def shows_points(self, X) -> bool:
        """Would these coordinates be drawn in this window (its plan story /
        elevation plane, or always in 3-D)?"""
        kind = self.config.kind
        if kind == "3d":
            return True
        m = self.doc.model
        tol = max(m.options.merge_tolerance, 1e-6) * 10
        X = np.atleast_2d(np.asarray(X, float))
        if kind == "plan":  # on the level, or a column / wall of the story below it
            zs = float(self.plane().value)
            story = m.stories.get(self.config.story) if self.config.story else None
            below = m.story_below(story.name) if story is not None else None
            z0 = below.elevation if below is not None else zs
            z = X[:, 2]
            return bool(np.any(np.abs(z - zs) <= tol) and np.all((z <= zs + tol) & (z >= z0 - tol)))
        pl = self.plane()
        return all(pl.contains(tuple(x), tol) for x in X)

    def zoom_to(self, X, margin: float = 0.6):
        """Zoom this window onto the coordinates (keeps the view direction)."""
        X = np.atleast_2d(np.asarray(X, float))
        cam = self.renderer.GetActiveCamera()
        c = (X.min(axis=0) + X.max(axis=0)) / 2
        up = np.array(cam.GetViewUp(), float)
        dop = np.array(cam.GetDirectionOfProjection(), float)
        right = np.cross(dop, up)
        right /= np.linalg.norm(right) or 1.0
        up = np.cross(right, dop)
        D = X - c
        w = float(np.ptp(D @ right)) if len(X) > 1 else 0.0
        h = float(np.ptp(D @ up)) if len(X) > 1 else 0.0
        size = max(w, h, scene.model_extent(self.doc.model) * 0.08) * (1 + 2 * margin)
        width_px, height_px = max(self.vtk_widget.width(), 1), max(self.vtk_widget.height(), 1)
        aspect = width_px / height_px
        half = max(h * (1 + 2 * margin), size / aspect if w >= h else 0.0, size * 0.5) / 2
        half = max(half, (w * (1 + 2 * margin)) / aspect / 2)
        dist = float(np.linalg.norm(np.subtract(cam.GetPosition(), cam.GetFocalPoint()))) or 1.0
        cam.SetFocalPoint(*c)
        if cam.GetParallelProjection():
            cam.SetPosition(*(c - dop * dist))
            cam.SetParallelScale(half)
        else:  # perspective: back off so the object fills about half the window
            d = half / math.tan(math.radians(cam.GetViewAngle()) / 2)
            cam.SetPosition(*(c - dop * d))
        self.renderer.ResetCameraClippingRange()
        self.render()

    def flash(self, X, closed: bool = False):
        """Blink a bright outline on the object so it is easy to find."""
        self._stop_flash()
        X = np.atleast_2d(np.asarray(X, float))
        rgb = self.FLASH_RGB
        actors = []
        if len(X) >= 2:
            pts = vtk.vtkPoints()
            for x in X:
                pts.InsertNextPoint(*x)
            line = vtk.vtkPolyLine()
            idx = list(range(len(X))) + ([0] if closed and len(X) > 2 else [])
            line.GetPointIds().SetNumberOfIds(len(idx))
            for k, i in enumerate(idx):
                line.GetPointIds().SetId(k, i)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(line)
            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetLines(cells)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            a = vtk.vtkActor()
            a.SetMapper(mapper)
            a.GetProperty().SetColor(*rgb)
            a.GetProperty().SetLineWidth(7)
            a.GetProperty().SetLighting(False)
            a.GetProperty().RenderLinesAsTubesOn()
            actors.append(a)
        # end / corner dots of a fixed size on the screen: a radius tied to the model
        # size became a ball metres across on a tall building, filling the window
        # once the camera zoomed onto a short member
        pts = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        for k, x in enumerate(X):
            pts.InsertNextPoint(*x)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(k)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetVerts(verts)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        a = vtk.vtkActor()
        a.SetMapper(mapper)
        a.GetProperty().SetColor(*rgb)
        a.GetProperty().SetPointSize(self.FLASH_DOT_PX)
        a.GetProperty().RenderPointsAsSpheresOn()
        actors.append(a)
        for a in actors:
            self.renderer.AddActor(a)
        self._flash_actors = actors
        self._flash_left = self.FLASH_COUNT
        timer = QtCore.QTimer(self)
        timer.setInterval(self.FLASH_MS)
        timer.timeout.connect(self._flash_tick)
        self._flash_timer = timer
        timer.start()
        self.render()

    def _flash_tick(self):
        self._flash_left -= 1
        if self._flash_left <= 0:
            self._stop_flash()
            return
        on = self._flash_left % 2 == 0
        for a in self._flash_actors:
            a.SetVisibility(on)
        self.render()

    def _stop_flash(self):
        t = getattr(self, "_flash_timer", None)
        if t is not None:
            t.stop()
            self._flash_timer = None
        for a in getattr(self, "_flash_actors", []):
            self.renderer.RemoveActor(a)
        had = bool(getattr(self, "_flash_actors", []))
        self._flash_actors = []
        if had:
            self.render()

    # ---------------------------------------------------------- geometry
    def _world_tolerance(self) -> float:
        """World size of PICK_PIXELS (logical, device-independent pixels) at the focal
        plane: the logical height of the widget, so it is the same on every display
        scaling - like the snap aperture (D16)."""
        cam = self.renderer.GetActiveCamera()
        h = self.vtk_widget.height() or 1
        if cam.GetParallelProjection():
            return 2.0 * cam.GetParallelScale() / h * self.PICK_PIXELS
        d = float(np.linalg.norm(np.subtract(cam.GetPosition(), cam.GetFocalPoint())))
        return 2.0 * d * math.tan(math.radians(cam.GetViewAngle()) / 2) / h * self.PICK_PIXELS

    def _sync_aspect(self):
        """World <-> screen conversions use the renderer's aspect ratio from its last
        render; after a resize without a render (a panel shown, the palette flowing into
        two columns, the splitter moved) it is stale, and every snap, pick and marker would
        be computed for the old shape of the window: the click landing beside the joint
        the cursor is on.  Recompute it from the window's current size first."""
        try:
            self.renderer.ComputeAspect()
        except Exception:
            pass

    def _display(self, xyz) -> Tuple[float, float]:
        self._sync_aspect()
        self.renderer.SetWorldPoint(xyz[0], xyz[1], xyz[2], 1.0)
        self.renderer.WorldToDisplay()
        d = self.renderer.GetDisplayPoint()
        return d[0], d[1]

    def _ray(self, x: int, y: int):
        ren = self.renderer
        self._sync_aspect()
        ren.SetDisplayPoint(x, y, 0.0)
        ren.DisplayToWorld()
        w0 = ren.GetWorldPoint()
        p0 = np.array(w0[:3], float) / (w0[3] or 1.0)
        ren.SetDisplayPoint(x, y, 1.0)
        ren.DisplayToWorld()
        w1 = ren.GetWorldPoint()
        p1 = np.array(w1[:3], float) / (w1[3] or 1.0)
        return p0, p1 - p0

    def _plane_point(self, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        """Intersect the view ray through display (x, y) with the drawing plane."""
        pl = self.plane()
        if pl.kind == "3d":
            return None
        p0, d = self._ray(x, y)
        k = {"plan": 2, "xz": 1, "yz": 0}[pl.kind]
        if abs(d[k]) < 1e-12:
            return None
        t = (pl.value - p0[k]) / d[k]
        p = p0 + t * d
        p[k] = pl.value
        return float(p[0]), float(p[1]), float(p[2])

    def _dpr(self) -> float:
        return view_dpr(self)

    def _snap_objects(self):
        return view_snap_objects(self)

    def _snap_engine(self, x: int, y: int):
        from .snap import SnapEngine, reference_lines_for
        from .tools import plane_band
        st = self.state
        pl = self.plane()
        prev = st.tool.points[-1] if getattr(st.tool, "points", None) else None
        refs = ()
        o = self.display_options
        if st.snap.reference and getattr(o, "reference_lines", False):
            refs = reference_lines_for(self.doc.model, pl, (o.reference_x, o.reference_y, o.reference_z))
        depth = near = None
        cam = self.renderer.GetActiveCamera()
        if pl.kind == "3d":
            eye = np.asarray(cam.GetPosition(), float)
            dop = np.asarray(cam.GetDirectionOfProjection(), float)
            depth = lambda xyz: float(np.dot(np.asarray(xyz, float) - eye, dop))  # noqa: E731
            # behind the eye / in front of the near clipping plane: not a target (perspective
            # would mirror it onto the screen)
            near = max(float(cam.GetClippingRange()[0]), 0.0) if not cam.GetParallelProjection() else None
        m = self.doc.model
        eng = SnapEngine(m, pl, st.snap, self._display, self._plane_point, prev,
                         max(m.options.merge_tolerance, 1e-6), refs, depth=depth,
                         objects=view_snap_objects(self), near=near, px_scale=view_dpr(self),
                         plane_tol=plane_band(m))
        eng.override = getattr(st, "snap_override", None)
        return eng

    def _snapped(self, x: int, y: int):
        """(point, SnapHit) on the drawing plane for display (x, y).

        All enabled snap types compete; the N key cycles through the hits inside
        the tolerance (``_snap_index``).  Without a hit the free cursor point on
        the plane is used (none in 3D views, or with *Draw Using Snap Only*)."""
        from .snap import SnapHit, SnapResolution
        st = self.state
        pl = self.plane()
        tool = st.tool
        # hovering is read-only: a command with picked points keeps its plane (and its
        # points) when the cursor passes over another window; the plane changes when a
        # point is clicked there (D06)
        if not getattr(tool, "points", None):
            tool.set_plane(pl)
            tool.tolerance = self._world_tolerance() * 2.0
        eng = self._snap_engine(x, y)
        hits = eng.find(x, y)
        self._snap_hits = hits
        hit, where = None, ""
        if hits:
            # the candidate chosen with N stays chosen while it is under the cursor (D04)
            idx = self._snap_index % len(hits)
            pick = getattr(self, "_snap_pick", None)
            if pick is not None:
                j = next((k for k, h in enumerate(hits) if _hit_key(h) == pick), None)
                if j is None:
                    self._snap_pick, idx = None, 0
                else:
                    idx = j
            self._snap_index = idx
            if idx or pick is not None:
                self._snap_pick = _hit_key(hits[idx])
            hit = eng.apply_increments(hits[idx])  # increments on lines / tracking too (D10)
        else:
            self._snap_pick, self._snap_index = None, 0  # left the candidates: the N choice ends
        if not hits and not st.snap.snap_only:
            if pl.kind == "3d":
                xyz, where = view_work_plane_point(self, x, y)
            else:
                xyz = self._plane_point(x, y)
                if xyz is not None:
                    where = {"plan": f"plane Z = {xyz[2]:g}", "xz": f"plane Y = {xyz[1]:g}",
                             "yz": f"plane X = {xyz[0]:g}"}[pl.kind]
            if xyz is not None:
                hit = eng.apply_increments(SnapHit(tuple(xyz), "free"))
        if hit is None:
            self._resolution = SnapResolution(pl.kind, (float(x), float(y)), None, "none", None, float("inf"),
                                              candidates=len(hits))
            return None, None
        err = float(np.linalg.norm(np.subtract(self._display(hit.xyz), (x, y))))
        self._resolution = SnapResolution(pl.kind, (float(x), float(y)), tuple(hit.xyz), hit.kind, hit.target, err,
                                          depth=eng.depth(hit.xyz) if eng.depth else None,
                                          candidates=len(hits), index=self._snap_index if hits else 0,
                                          plane=where)
        return hit.xyz, hit

    def _work_plane_point(self, x: int, y: int):
        return view_work_plane_point(self, x, y)

    # ------------------------------------------------------------- events
    def _on_left_press(self, obj, event):
        self.activate()
        x, y = self.interactor.GetEventPosition()
        if self._cube_event("press", x, y):
            return
        mode = getattr(self.state, "nav_mode", None)
        if self.interactor.GetShiftKey() or mode in ("rotate", "pan", "zoom_window"):
            # Shift + left drag rotates the 3D view in every tool; toolbar navigation modes
            kind = {"rotate": "orbit", "pan": "pan", "zoom_window": "zoom_window"}.get(mode, "orbit")
            self._start_nav(kind, x, y)
            return
        st = self.state
        if self.DIRECT_EDIT and st.tool.name == "select" and not self.interactor.GetShiftKey():
            # CAD direct editing: drag a grip (joint of the selection) or the selection itself
            grip = self._grip_at(x, y)
            if grip is not None or (st.count and self._on_selection(x, y)):
                if grip is not None:
                    base = self.doc.model.points[grip].xyz
                else:
                    base, _ = self._snapped(x, y)
                    if base is None:
                        base = self._plane_point(x, y)
                if base is not None:
                    self._drag = {"mode": "grip" if grip else "move", "grip": grip, "base": tuple(base),
                                  "start": (x, y), "moved": False}
                    self._press = (x, y)
                    self._dragging = False
                    return
        if st.tool.name in ("select", "select_line") or getattr(st.tool, "picks", None) \
                or getattr(st.tool, "selecting", False):
            # select tools, load commands picking their objects, and modify commands asking
            # for their objects (click or window)
            self._press = (x, y)
            self._dragging = False
            return
        if getattr(st.tool, "picks_frames", False):  # Trim / Extend / Offset: click a frame
            if self.config.kind not in st.tool.views:
                st.status.emit(f"{st.tool.title} works in plan and elevation windows")
                return
            st.tool.set_plane(self.plane())
            cursor = self._plane_point(x, y)
            try:
                st.tool.frame_click(self._frame_under(x, y), cursor)
            except Exception as exc:
                st.report(exc)
            st.update_hint()
            self._update_preview(x, y)
            return
        if self.config.kind not in st.tool.views:
            st.status.emit(f"{st.tool.name.replace('_', ' ').title()} works in "
                           f"{', '.join(v.upper() if v != 'plan' else 'plan' for v in st.tool.views)} views")
            return
        # the window clicked in becomes the command's window: its plane is used from now
        # on (a straight chain goes on across windows; polygons / arcs start again)
        st.tool.set_plane(self.plane())
        st.tool.tolerance = self._world_tolerance() * 2.0
        pt, snap = self._snapped(x, y)
        if pt is None:
            st.status.emit("Nothing to snap to here: in 3D pick a joint, line or grid point, or set "
                           "'3D: point off the snaps' to a work plane (Properties of Object), or type x,y,z")
            return
        try:
            if self.interactor.GetRepeatCount() > 0:
                done = st.tool.double_click(pt)
            else:
                done = st.tool.click(pt, snap)
            st.snap_override = None  # a one-shot snap is used up by the point
            self._snap_pick, self._snap_index = None, 0
        except Exception as exc:
            st.report(exc)
            st.tool.cancel()
            self._clear_preview()
            self.render()
            return
        st.update_hint()
        self._update_preview(x, y)
        self._check_pending_panel()

    def _check_pending_panel(self):
        tool = self.state.tool
        pending = getattr(tool, "pending", None)
        if pending:
            tool.pending = None
            self._clear_preview()
            self.render()
            self.panel_requested.emit(list(pending))

    def _on_left_release(self, obj, event):
        x0, y0 = self.interactor.GetEventPosition()
        if self._cube_event("release", x0, y0):
            return
        if self._nav is not None:
            self._end_nav()
            return
        st = self.state
        if self._press is None:
            return
        x, y = self.interactor.GetEventPosition()
        ctrl = bool(self.interactor.GetControlKey())
        p0 = self._press
        self._press = None
        if self._drag is not None:
            if self._drag_finish(x, y):
                self._dragging = False
                return
            self._dragging = False  # a click on the selection without dragging: plain select
            self._click_select(x, y, ctrl)
            return
        if getattr(st.tool, "selecting", False):  # modify command asking for its objects
            if self._rubber.hide():
                self.render()
            if self._dragging:
                self._window_select(p0, (x, y), True, crossing=x < p0[0])
            else:
                hit = self.pick(x, y)
                if hit is not None:
                    st.toggle(hit[0], hit[1], True)
            self._dragging = False
            st.update_hint()
            return
        if self._rubber.hide():
            self.render()
        self._hide_line()
        if st.tool.name == "select_line":
            if self._dragging:
                self._line_select(p0, (x, y), ctrl)
            self._dragging = False
            return
        picks = getattr(st.tool, "picks", None)
        if picks:  # load command: picking adds / removes objects of its kind
            if self._dragging:
                self._window_select(p0, (x, y), True, crossing=x < p0[0])
                st.keep_only(picks[0])
            else:
                hit = next((h for h in self.pick_all(x, y) if h[0] in picks), None)
                if hit is not None:
                    st.toggle(hit[0], hit[1], True)
            self._dragging = False
            return
        if self._dragging:
            self._window_select(p0, (x, y), ctrl, crossing=x < p0[0])
        else:
            self._click_select(x, y, ctrl)
        self._dragging = False

    def _on_mouse_move(self, obj, event):
        x, y = self.interactor.GetEventPosition()
        st = self.state
        if self._nav is None and self._press is None and self._cube_event("move", x, y):
            return
        if self._nav is not None:
            self._drag_nav(x, y)
            return
        self._last_xy = (x, y)
        if self._press is not None and self._drag is not None:
            dx, dy = x - self._press[0], y - self._press[1]
            if self._drag["moved"] or abs(dx) > 4 or abs(dy) > 4:
                self._drag["moved"] = True
                self._drag_update(x, y)
            return
        if self._press is not None:
            dx, dy = x - self._press[0], y - self._press[1]
            if self._dragging or abs(dx) > 4 or abs(dy) > 4:
                self._dragging = True
                if st.tool.name == "select_line":
                    self._show_line(self._press, (x, y))
                else:
                    # window (left to right, solid) or crossing (right to left, dashed) box
                    self._rubber.show(self._press, (x, y), crossing=x < self._press[0])
                    self.render()
            return
        if st.tool.name not in ("select", "select_line") and not getattr(st.tool, "picks", None) \
                and not getattr(st.tool, "selecting", False):
            self._update_preview(x, y)
        elif self.DIRECT_EDIT and st.tool.name == "select" and (st.frames or st.areas or st.points):
            grip = self._grip_at(x, y)
            want = QtCore.Qt.SizeAllCursor if (grip or self._on_selection(x, y)) else None
            if want is not None:
                self.vtk_widget.setCursor(want)
            elif getattr(self.state, "nav_mode", None) is None:
                self.vtk_widget.unsetCursor()
        xyz = self._plane_point(x, y)
        if xyz is not None:
            pl = self.plane()
            where = self.config.story if pl.kind == "plan" else (self.config.grid or "3D")
            st.cursor_moved.emit(f"X {xyz[0]:.3f}  Y {xyz[1]:.3f}  Z {xyz[2]:.3f}  [{where}]")

    def _on_right_press(self, obj, event):
        self._right_press = self.interactor.GetEventPosition()
        self._start_nav("orbit", *self._right_press)  # 3D: rotate, plan / elevation: pan

    def _on_right_release(self, obj, event):
        moved = self._nav_moved
        self._end_nav()
        if self._right_press is None:
            return
        x, y = self.interactor.GetEventPosition()
        if not moved and abs(x - self._right_press[0]) <= 3 and abs(y - self._right_press[1]) <= 3:
            st = self.state
            if (self.interactor.GetShiftKey() or self.interactor.GetControlKey()) and self._takes_points():
                self._right_press = None
                self.snap_menu(x, y)  # CAD: Shift / Ctrl + right-click = object snap menu
                return
            if st.tool.name in ("select", "select_line") and not getattr(st.tool, "points", None):
                # ETABS: right click on an object opens its Information form, on the
                # background the window context menu; Shift lists the objects under the cursor
                if self.interactor.GetShiftKey():
                    hits = self.pick_all(x, y)
                else:
                    hit = self.pick(x, y)
                    hits = [hit] if hit else []
                if hits:
                    from .object_info import choose_object
                    chosen = choose_object(hits, self) if len(hits) > 1 else hits[0]
                    if chosen:
                        self.object_menu(chosen[0], chosen[1], x, y)
                else:
                    self.background_menu_requested.emit(self, x, y)
            elif getattr(st.tool, "picks", None):
                st.apply_load_tool()  # load command: right click applies to the picked objects
                self.render()
            elif getattr(st.tool, "selecting", False):
                st.tool.done_selecting(st)  # modify command: right click = objects chosen
                self.render()
            elif getattr(st.tool, "closes_polygon", False) and len(getattr(st.tool, "points", []) or []) >= 3:
                # floors / walls / panels: right click closes the outline back to the first corner
                try:
                    st.tool.finish()
                except Exception as exc:
                    st.report(exc)
                st.update_hint()
                self._clear_preview()
                self.render()
                self._check_pending_panel()
            else:
                # right click without drag ends the current chain (ETABS behaviour)
                st.tool.cancel()
                self._clear_preview()
                self.render()
        self._right_press = None

    def _takes_points(self) -> bool:
        """The command is waiting for a point (draw, move / copy base or target ...)."""
        t = self.state.tool
        if t.name in ("select", "select_line"):
            return False
        return not (getattr(t, "picks", None) or getattr(t, "selecting", False) or getattr(t, "picks_frames", False))

    def snap_menu(self, x: int, y: int):
        """AutoCAD Object Snap shortcut menu: the chosen snap is used for the next point only."""
        from .snap_icons import OVERRIDE_MENU, qicon
        st = self.state
        menu = QtWidgets.QMenu(self.vtk_widget)
        menu.setToolTipsVisible(True)
        title = menu.addAction("Snap for the next point")
        title.setEnabled(False)
        menu.addSeparator()
        chosen = {}
        for key, label, marker in OVERRIDE_MENU:
            if key == "none":
                menu.addSeparator()
            a = menu.addAction(qicon(marker), label)
            a.setCheckable(True)
            a.setChecked(st.snap_override == key)
            chosen[a] = key
        menu.addSeparator()
        opts = menu.addAction(qicon("grid"), "Snap Options...")
        h = self.vtk_widget.height()
        r = self.vtk_widget.devicePixelRatioF() or 1.0
        a = menu.exec(self.vtk_widget.mapToGlobal(QtCore.QPoint(int(x / r), int(h - y / r))))
        if a is None:
            return
        if a is opts:
            act = getattr(self.window(), "menu_actions", {}).get("draw.snap_options")
            if act is not None:
                act.trigger()
            return
        key = chosen.get(a)
        st.snap_override = None if key == st.snap_override else key
        if st.snap_override:
            label = next(lb for k, lb, _ in OVERRIDE_MENU if k == key)
            st.status.emit(f"Snap override: {label} - for the next point only")
        lx, ly = getattr(self, "_last_xy", (x, y))
        self._update_preview(lx, ly)

    # --------------------------------------------------------- navigation
    def _start_nav(self, kind: str, x: int, y: int):
        if kind == "orbit" and self.config.kind != "3d":
            kind = "pan"  # plan and elevation windows stay 2D
        self._nav, self._nav_last, self._nav_start, self._nav_moved = kind, (x, y), (x, y), False

    # placing many labels is the slowest part of a frame: while the view is dragged
    # (rotate / pan) windows with more than this many labels hide them, and show
    # them again the moment the button is released
    DRAG_LABEL_LIMIT = 400

    def _drag_labels(self, hide: bool):
        if hide:
            if getattr(self, "_drag_hidden", None):
                return
            acts, n = [], 0
            col = self.renderer.GetActors2D()
            col.InitTraversal()
            for _ in range(col.GetNumberOfItems()):
                a = col.GetNextActor2D()
                mp = a.GetMapper() if a is not None else None
                if isinstance(mp, vtk.vtkLabelPlacementMapper) and a.GetVisibility():
                    src = mp.GetInputAlgorithm()
                    data = src.GetInputDataObject(0, 0) if src is not None else None
                    n += data.GetNumberOfPoints() if data is not None and hasattr(data, "GetNumberOfPoints") else 0
                    acts.append(a)
            if n > self.DRAG_LABEL_LIMIT:
                for a in acts:
                    a.SetVisibility(False)
                self._drag_hidden = acts
        else:
            acts = getattr(self, "_drag_hidden", None)
            self._drag_hidden = None
            if acts:
                for a in acts:
                    a.SetVisibility(True)
                self.render()

    def _drag_nav(self, x: int, y: int):
        lx, ly = self._nav_last
        if abs(x - self._nav_start[0]) > 2 or abs(y - self._nav_start[1]) > 2:
            self._nav_moved = True
            if self._nav in ("orbit", "pan"):
                self._drag_labels(True)
        if self._nav == "orbit":
            navigation.orbit(self.renderer, (x - lx) * navigation.ROTATE_DEG_PER_PIXEL,
                             -(y - ly) * navigation.ROTATE_DEG_PER_PIXEL)
        elif self._nav == "pan":
            navigation.pan(self.renderer, lx, ly, x, y)
        elif self._nav == "zoom_window":
            self._rubber.show(self._nav_start, (x, y), crossing=False)
            self._nav_last = (x, y)
            self.render()
            return
        self._nav_last = (x, y)
        self.render()

    def _end_nav(self):
        if self._nav == "zoom_window" and self._nav_start and self._nav_last:
            self._rubber.hide()
            sz = self.vtk_widget.GetRenderWindow().GetSize()
            navigation.zoom_window(self.renderer, *self._nav_start, *self._nav_last, sz[0], sz[1])
            self.render()
            if getattr(self.state, "nav_mode", None) == "zoom_window":
                self.state.set_nav_mode(None)  # one-shot, like ETABS Zoom Window
        self._nav = None
        self._drag_labels(False)
        self._nav_last = self._nav_start = None

    def _on_nav_changed(self, mode: str):
        cursors = {"rotate": QtCore.Qt.SizeAllCursor, "pan": QtCore.Qt.OpenHandCursor,
                   "zoom_window": QtCore.Qt.CrossCursor}
        if mode in cursors:
            self.vtk_widget.setCursor(cursors[mode])
        else:
            self.vtk_widget.unsetCursor()

    def _on_middle_press(self, obj, event):
        self.activate()
        x, y = self.interactor.GetEventPosition()
        if self.interactor.GetRepeatCount() > 0:  # double click on the wheel: full view
            self.fit()
            return
        self._start_nav("orbit" if self.interactor.GetShiftKey() else "pan", x, y)

    def _on_middle_release(self, obj, event):
        self._end_nav()

    WHEEL_LABEL_IDLE_MS = 180  # labels hidden while the wheel turns come back after this pause

    def _on_wheel(self, direction: int):
        x, y = self.interactor.GetEventPosition()
        f = navigation.WHEEL_FACTOR if direction > 0 else 1.0 / navigation.WHEEL_FACTOR
        # many labels (force diagrams of a large model): hidden while zooming, like a drag
        self._drag_labels(True)
        if getattr(self, "_drag_hidden", None):
            t = getattr(self, "_wheel_timer", None)
            if t is None:
                t = self._wheel_timer = QtCore.QTimer(self)
                t.setSingleShot(True)
                t.setInterval(self.WHEEL_LABEL_IDLE_MS)
                t.timeout.connect(self._wheel_idle)
            t.start()
        navigation.zoom_at(self.renderer, x, y, f)
        self.render()

    def _wheel_idle(self):
        if getattr(self, "_nav", None) is None:  # not in the middle of a drag (it shows them on release)
            self._drag_labels(False)

    def rotate_view(self, d_azimuth: float, d_elevation: float):
        if self.config.kind == "3d":
            navigation.orbit(self.renderer, d_azimuth, d_elevation)
            self.render()

    def zoom(self, factor: float):
        sz = self.vtk_widget.GetRenderWindow().GetSize()
        navigation.zoom_at(self.renderer, sz[0] / 2, sz[1] / 2, factor)
        self.render()

    def pan_view(self, fx: float, fy: float):
        """Pan by a fraction of the window size."""
        sz = self.vtk_widget.GetRenderWindow().GetSize()
        cx, cy = sz[0] / 2, sz[1] / 2
        navigation.pan(self.renderer, cx, cy, cx + fx * sz[0], cy + fy * sz[1])
        self.render()

    def default_3d(self):
        self.config.kind = "3d"
        self._fill_header()
        self.invalidate("scene", "overlays")
        self.apply_camera()

    _TYPED_KEYS = {"period": ".", "KP_Decimal": ".", "comma": ",", "minus": "-", "KP_Subtract": "-",
                   "less": "<", "at": "@", "plus": "+", "KP_Add": "+", "e": "e", "E": "e", "numbersign": "#"}

    def _accepts_typing(self) -> bool:
        """CAD dynamic input: a point tool with a first point (or a tool reading a
        typed value of its own: rotate angle, offset distance)."""
        tool = self.state.tool
        if tool.name in ("select", "select_line") or getattr(tool, "picks", None) or getattr(tool, "selecting", False):
            return False
        if self.config.kind not in getattr(tool, "views", ()):
            return False
        # a first point can be typed as coordinates too (x,y,z) (D14)
        return bool(tool.points) or type(tool).typed is not Tool.typed or \
            (getattr(tool, "takes_typed_points", False) and not getattr(tool, "picks_frames", False))

    def _typed_char(self, key: str) -> Optional[str]:
        if len(key) == 1 and key.isdigit():
            return key
        if key.startswith("KP_") and key[3:].isdigit():
            return key[3:]
        ch = self._TYPED_KEYS.get(key)
        if ch == "e" and not self._typed:
            return None  # letters only inside a number (1e3)
        if ch == "#" and self._typed:
            return None  # '#' (absolute point) only at the start
        if ch in ("+",) and not self._typed:
            return None  # +/- zoom when nothing is typed
        return ch

    def _length_unit(self):
        """(label, factor model -> shown) of the lengths typed / shown while drawing:
        the "Structure dimensions, coordinates" item of the display units."""
        from ..display_units import DisplayUnits
        try:
            du = DisplayUnits.of(self.doc.model)
            return du.label("coordinates"), du.factor("coordinates")
        except Exception:
            return self.doc.model.units.length, 1.0

    def _typed_to_model(self, text: str) -> str:
        """Typed distances are in the display length unit: convert them to model units
        (an angle after '<' stays in degrees)."""
        _, f = self._length_unit()
        if abs(f - 1.0) < 1e-12:
            return text
        import re
        num = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        t = text.strip()
        head, sep, ang = t.partition("<")
        conv = re.sub(num, lambda m: f"{float(m.group(0)) / f:.12g}", head)
        return conv + (sep + ang if sep else "")

    def _apply_typed(self):
        st = self.state
        text, self._typed = self._typed, ""
        text = self._typed_to_model(text)
        try:
            pt = st.tool.typed(text, self._last_cursor)
            if pt is not None:
                st.tool.click(pt, None)
        except Exception as exc:
            st.report(exc) if not isinstance(exc, ValueError) else st.error.emit(str(exc))
        st.update_hint()
        x, y = self._last_xy
        self._update_preview(x, y)
        self._check_pending_panel()

    def _on_key(self, obj, event):
        key = self.interactor.GetKeySym()
        st = self.state
        ctrl = bool(self.interactor.GetControlKey())
        if not ctrl and self._accepts_typing():
            ch = self._typed_char(key or "")
            if ch is not None and not (ch == "-" and not self._typed and not st.tool.points):
                self._typed += ch
                self._update_preview(*self._last_xy, keep_index=True)
                return
            if key == "BackSpace" and self._typed:
                self._typed = self._typed[:-1]
                self._update_preview(*self._last_xy, keep_index=True)
                return
            if key in ("Return", "KP_Enter") and self._typed:
                self._apply_typed()
                return
            if key == "Escape" and self._typed:
                self._typed = ""
                self._update_preview(*self._last_xy, keep_index=True)
                return
        arrows = {"Left": (-1, 0), "Right": (1, 0), "Up": (0, 1), "Down": (0, -1)}
        if key in arrows:
            dx, dy = arrows[key]
            if self.config.kind == "3d" and not ctrl:
                self.rotate_view(15 * dx, 10 * dy)
            else:
                self.pan_view(-0.1 * dx, -0.1 * dy)
            return
        if key in ("plus", "equal", "KP_Add"):
            self.zoom(navigation.WHEEL_FACTOR)
            return
        if key in ("minus", "KP_Subtract"):
            self.zoom(1.0 / navigation.WHEEL_FACTOR)
            return
        if key == "Home":
            self.apply_camera()
            return
        if key in ("Prior", "Next", "Page_Up", "Page_Down"):
            self.step_where(+1 if key in ("Prior", "Page_Up") else -1)
            return
        if key in ("n", "N") and st.tool.name not in ("select", "select_line"):
            if len(self._snap_hits) > 1:
                self._snap_pick = None  # the next candidate becomes the chosen one
                self._snap_index = (self._snap_index + 1) % len(self._snap_hits)
                x, y = self.interactor.GetEventPosition()
                self._update_preview(x, y, keep_index=True)
            return
        if key == "Escape":
            self.escape()
        elif key in ("Return", "KP_Enter"):
            self.enter()
        elif key == "Delete":
            st.delete_selected()

    def escape(self):
        """Esc: end the chain / drop the picks / clear the selection / back to Select."""
        st = self.state
        if True:  # (kept as a block: the body was the key handler's)
            st.snap_override = None
            if self._drag is not None:
                self._drag = None
                self._press = None
                self._clear_preview()
                self.render()
                return
            st.cancel_tool()
            self._clear_preview()
            self.render()

    def enter(self):
        """Enter: apply the command (picks), end the selection step, or finish the tool."""
        st = self.state
        if True:
            if getattr(st.tool, "picks", None):
                st.apply_load_tool()
                self.render()
                return
            if getattr(st.tool, "selecting", False):
                st.tool.done_selecting(st)
                self.render()
                return
            try:
                st.tool.finish()
                st.update_hint()
            except Exception as exc:
                st.report(exc)
            self._clear_preview()
            self.render()
            self._check_pending_panel()

    # ----------------------------------------------------------- selection
    def pick(self, x: int, y: int):
        """Screen-space pick: ('point'|'frame'|'area', name) or None."""
        m = self.doc.model
        vis = self.visible()
        tol = self.PICK_PIXELS * view_dpr(self)  # logical pixels -> display (physical) pixels (D16)
        frames = [f for f in m.frames.values() if vis(f)]
        areas = [a for a in m.areas.values() if vis(a)]
        shown_pts = {n for f in frames for n in (f.point_i, f.point_j)} | {n for a in areas for n in a.points}
        if self.config.kind == "3d":
            shown_pts = set(m.points)
        else:
            pl = self.plane()
            tolw = max(m.options.merge_tolerance, 1e-6)
            shown_pts |= {p.name for p in m.points.values() if pl.contains(p.xyz, tolw)}
        disp = {n: self._display(m.points[n].xyz) for n in shown_pts | {n for f in frames for n in (f.point_i, f.point_j)}
                | {n for a in areas for n in a.points}}
        best, best_d = None, tol
        for n in shown_pts:
            d = math.hypot(disp[n][0] - x, disp[n][1] - y)
            if d < best_d:
                best, best_d = ("point", n), d
        if best:
            return best
        for f in frames:
            (ax, ay), (bx, by) = disp[f.point_i], disp[f.point_j]
            L2 = (bx - ax) ** 2 + (by - ay) ** 2
            if L2 <= 0:
                continue
            t = max(0.0, min(1.0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / L2))
            d = math.hypot(ax + t * (bx - ax) - x, ay + t * (by - ay) - y)
            if d < best_d:
                best, best_d = ("frame", f.name), d
        if best:
            return best
        for a in areas:
            if point_in_polygon((x, y), [disp[p] for p in a.points]):
                return ("area", a.name)
        for pl in getattr(m, "panel_loads", {}).values():
            if len(pl.points) >= 3 and point_in_polygon((x, y), [self._display(p) for p in pl.points]):
                return ("panel", pl.name)
        return None

    def pick_all(self, x: int, y: int):
        """Every object within the pick tolerance of display (x, y): points, then
        frames, then areas containing the cursor."""
        m = self.doc.model
        vis = self.visible()
        tol = self.PICK_PIXELS * view_dpr(self)  # logical pixels -> display (physical) pixels (D16)
        out = []
        frames = [f for f in m.frames.values() if vis(f)]
        areas = [a for a in m.areas.values() if vis(a)]
        pv = self._point_visible()
        for p in m.points.values():
            if not pv(p):
                continue
            dx, dy = self._display(p.xyz)
            if math.hypot(dx - x, dy - y) < tol:
                out.append(("point", p.name))
        disp = {}
        for f in frames:
            for n in (f.point_i, f.point_j):
                if n not in disp:
                    disp[n] = self._display(m.points[n].xyz)
            (ax, ay), (bx, by) = disp[f.point_i], disp[f.point_j]
            L2 = (bx - ax) ** 2 + (by - ay) ** 2
            if L2 <= 0:
                continue
            t = max(0.0, min(1.0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / L2))
            if math.hypot(ax + t * (bx - ax) - x, ay + t * (by - ay) - y) < tol:
                out.append(("frame", f.name))
        for a in areas:
            if point_in_polygon((x, y), [self._display(m.points[p].xyz) for p in a.points]):
                out.append(("area", a.name))
        for pl in getattr(m, "panel_loads", {}).values():
            if len(pl.points) >= 3 and point_in_polygon((x, y), [self._display(p) for p in pl.points]):
                out.append(("panel", pl.name))
        return out

    def _click_select(self, x: int, y: int, add: bool):
        hit = self.pick(x, y)
        st = self.state
        if hit is None:
            if not add:
                st.clear_selection()
            return
        st.toggle(hit[0], hit[1], add)

    def _window_select(self, p0, p1, add: bool, crossing: bool = False):
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        m = self.doc.model
        vis = self.visible()
        if crossing:
            frames, areas, points = objects_crossing(m, self._display, (x0, y0, x1, y1), vis,
                                                     self._point_visible())
        else:
            def inside(p):
                dx, dy = self._display(p.xyz)
                return x0 <= dx <= x1 and y0 <= dy <= y1
            frames, areas, points = objects_in_window(m, inside, None)
            frames = [f for f in frames if vis(m.frames[f])]
            areas = [a for a in areas if vis(m.areas[a])]
            pv = self._point_visible()
            points = [p for p in points if pv(m.points[p])]
        self.state.set_selection(frames, areas, points, add=add)

    def _line_select(self, p0, p1, add: bool):
        frames, areas, points = objects_on_line(self.doc.model, self._display, p0, p1, self.visible())
        self.state.set_selection(frames, areas, points, add=add)

    def _point_visible(self):
        if self.config.kind == "3d":
            return lambda p: True
        pl = self.plane()
        tol = max(self.doc.model.options.merge_tolerance, 1e-6)
        return lambda p: pl.contains(p.xyz, tol)

    def _show_line(self, a, b):
        self._hide_line()
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(a[0], a[1], 0)
        pts.InsertNextPoint(b[0], b[1], 0)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*scene.COLORS["selection"])
        actor.GetProperty().SetLineWidth(2)
        self.renderer.AddActor(actor)
        self._line_preview = actor
        self.render()

    def _hide_line(self):
        if self._line_preview is not None:
            self.renderer.RemoveActor(self._line_preview)
            self._line_preview = None
            self.render()

    # ------------------------------------------------------------- preview
    def _clear_preview(self):
        for a in self._preview_actors:
            self.renderer.RemoveActor(a)
        self._preview_actors = []
        self.text.SetInput("")
        self.tip.SetVisibility(0)

    def _lines_actor(self, segs, color, width):
        coords, lines = [], vtk.vtkCellArray()
        for k, (a, b) in enumerate(segs):
            coords += [tuple(a), tuple(b)]
            lines.InsertNextCell(2)
            lines.InsertCellPoint(2 * k)
            lines.InsertCellPoint(2 * k + 1)
        poly = vtk.vtkPolyData()
        poly.SetPoints(scene._points(np.array(coords, float)))
        poly.SetLines(lines)
        return scene._actor_from_polydata(poly, color, width, lighting=False)

    def _update_preview(self, x: int, y: int, keep_index: bool = False):
        st = self.state
        self._last_xy = (x, y)
        # (the candidate chosen with N is kept by _snapped while it stays under the cursor)
        self._clear_preview()
        if self.config.kind not in st.tool.views:
            self.render()
            return
        if getattr(st.tool, "picks_frames", False):  # Trim / Extend / Offset: the frame under the cursor
            st.tool.set_plane(self.plane())
            cursor = self._plane_point(x, y)
            frame = self._frame_under(x, y)
            try:
                pv = st.tool.frame_preview(frame, cursor)
            except Exception:
                self.render()
                return
            if frame is not None:
                m = self.doc.model
                f = m.frames[frame]
                self._preview_actors.append(self._lines_actor([(m.points[f.point_i].xyz, m.points[f.point_j].xyz)],
                                                              scene.COLORS["select_box"], 5.0))
            self._draw_preview(pv, None, x, y)
            return
        pt, snap = self._snapped(x, y)
        if pt is None:
            self._last_cursor = None  # no direction for a typed distance from here (D09)
            self.render()
            return
        self._last_cursor = tuple(pt)
        try:
            pv: Preview = st.tool.preview(pt)
        except Exception:
            self.render()
            return
        self._draw_preview(pv, snap, x, y)

    def _draw_preview(self, pv: Preview, snap, x: int, y: int):
        """Draw a tool preview: rubber lines, ghost, guides, the snap glyph and its
        tracking lines, the text (bottom left) and the CAD tooltip at the cursor."""
        color = scene.COLORS["preview"]
        if pv.lines:
            self._preview_actors.append(self._lines_actor(pv.lines, color, 3.0))
        if getattr(pv, "remove", None):
            self._preview_actors.append(self._lines_actor(pv.remove, (0.92, 0.15, 0.15), 6.0))
        if getattr(pv, "guides", None):
            self._preview_actors.append(scene._dashed_lines_actor(
                self.doc.model, [p for seg in pv.guides for p in seg], scene.COLORS["grid_text"], 1.2))
        if len(pv.polygon) >= 2:
            n = len(pv.polygon)
            coords = [tuple(p) for p in pv.polygon]
            segs = [(coords[k], coords[(k + 1) % n]) for k in range(n if n > 2 else 1)]
            self._preview_actors.append(self._lines_actor(segs, color, 3.0))
            if n >= 3:
                polys = vtk.vtkCellArray()
                polys.InsertNextCell(n)
                for k in range(n):
                    polys.InsertCellPoint(k)
                fill = vtk.vtkPolyData()
                fill.SetPoints(scene._points(np.array(coords, float)))
                fill.SetPolys(polys)
                self._preview_actors.append(scene._actor_from_polydata(fill, color, opacity=0.25))
        snapped = snap is not None and snap.kind != "free"
        if pv.marker is not None or snapped:
            self._preview_actors.append(self._snap_marker(pv.marker if pv.marker is not None else snap.xyz,
                                                          snap.kind if snap else "free", snapped))
        for track in (getattr(snap, "track", None), getattr(snap, "track2", None)):
            if track:
                self._preview_actors.append(scene._dashed_lines_actor(self.doc.model, [track[0], track[1]],
                                                                      scene.COLORS["snap"], 1.5))
        txt = pv.text
        tip = []
        if snapped:
            label = getattr(snap, "label", snap.kind)
            n = len(self._snap_hits)
            tip.append(label + (f"   [{self._snap_index % n + 1}/{n}  N]" if n > 1 else ""))
            txt = (txt + "   " if txt else "") + f"snap: {label}" + (f"  [{self._snap_index % n + 1}/{n}, N = next]"
                                                                        if n > 1 else "")
        elif snap is not None and getattr(snap, "text", ""):
            txt = (txt + "   " if txt else "") + snap.text
        unit, uf = self._length_unit()
        res = getattr(self, "_resolution", None)
        if snap is not None and res is not None and res.xyz is not None:
            # where the point is, in full: two candidates can overlap on the screen (D04 / D14)
            p = res.xyz
            st_ = self.doc.model.story_at(p[2], max(self.doc.model.options.merge_tolerance, 1e-6))
            where = f"X {p[0] * uf:.3f}  Y {p[1] * uf:.3f}  Z {p[2] * uf:.3f} {unit}"
            where += f"  [{st_.name}]" if st_ is not None else ""
            if not snapped and res.plane:
                where += f"  (free point on {res.plane})"
            tip.append(where)
        if pv.text:
            import re
            # lengths of the tools are in model units: show them in the display unit, with the unit
            pv.text = re.sub(r"\b(L =|dX|dY|dZ|trim)\s+([-+]?\d+(?:\.\d+)?)",
                             lambda m: f"{m.group(1)} {float(m.group(2)) * uf:.4g} {unit}", pv.text)
            tip.append(pv.text)
        if self._typed:
            tip.append(f"> {self._typed}_ {unit}   (Enter)")
        elif self._accepts_typing():
            if getattr(self.state.tool, "points", None):
                tip.append(f"type a length in {unit}: 6 or 6<30 (angle °) or 3,4  (#x,y,z = absolute)")
            else:
                tip.append(f"or type the point: x,y,z in {unit}")
        self.text.SetInput(txt)
        if tip and (x or y):
            self.tip.SetInput("\n".join(tip))
            self.tip.SetPosition(x + 18, max(4, y - 22 - 16 * len(tip)))
            self.tip.SetVisibility(1)
        for a in self._preview_actors:
            self.renderer.AddActor(a)
        self.render()

    # ETABS draws a dot and a text at the snap location; here each snap kind has its
    # own marker shape so it can be told at a glance (CAD style)
    def _snap_marker(self, xyz, kind: str, snapped: bool) -> vtk.vtkActor:
        r = self._world_tolerance() * 0.7
        c = np.asarray(xyz, float)
        cam = self.renderer.GetActiveCamera()
        vu = np.asarray(cam.GetViewUp(), float)
        vd = np.asarray(cam.GetDirectionOfProjection(), float)
        vr = np.cross(vd, vu)
        from .snap_icons import strokes
        coords = []
        for st in strokes(kind):
            for k in range(len(st) - 1):
                coords += [c + r * (st[k][0] * vr + st[k][1] * vu), c + r * (st[k + 1][0] * vr + st[k + 1][1] * vu)]
        lines = vtk.vtkCellArray()
        for k in range(len(coords) // 2):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(2 * k)
            lines.InsertCellPoint(2 * k + 1)
        poly = vtk.vtkPolyData()
        poly.SetPoints(scene._points(np.array(coords, float)))
        poly.SetLines(lines)
        # no snap: a green cross (circles are the pin symbols); the line drawn is pink
        color = scene.COLORS["snap"] if snapped else scene.COLORS.get("free_point", (0.00, 0.62, 0.32))
        return scene._actor_from_polydata(poly, color, 2.5, lighting=False)

    # ------------------------------------------------------------ capture
    def screenshot(self, path: str):
        self.vtk_widget.GetRenderWindow().Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.vtk_widget.GetRenderWindow())
        w2i.ReadFrontBufferOff()
        w2i.Update()
        wr = vtk.vtkPNGWriter()
        wr.SetFileName(path)
        wr.SetInputConnection(w2i.GetOutputPort())
        wr.Write()

    # ---------------------------------------------------- compatibility API
    def set_tool(self, name: str):
        self.state.set_tool(name)

    def delete_selected(self):
        return self.state.delete_selected()

    def set_selection(self, frames=(), areas=(), points=()):
        self.state.set_selection(frames, areas, points)

    def select_all(self):
        self.state.select_all(None if self.config.kind == "3d" else self.visible())

    def clear_selection(self):
        self.state.clear_selection()

    def finalize(self):
        try:
            self.vtk_widget.GetRenderWindow().Finalize()
        except Exception:
            pass
