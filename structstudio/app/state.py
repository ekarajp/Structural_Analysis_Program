"""State shared by all model windows: active tool, selection, snapping and
results display options (ETABS keeps these per session, not per window)."""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from PySide6 import QtCore

from ..model import ModelError
from .document import Document
from .tools import TOOLS, DrawSettings, SelectTool, Tool, delete_objects, select_by_type
from . import modify as _modify  # noqa: F401  (registers Move / Copy / Rotate / Mirror / Trim ... tools)


class ViewState(QtCore.QObject):
    selection_changed = QtCore.Signal()
    display_changed = QtCore.Signal()
    tool_changed = QtCore.Signal(str)
    nav_changed = QtCore.Signal(str)  # "" | "rotate" | "pan" | "zoom_window"
    status = QtCore.Signal(str)
    error = QtCore.Signal(str)
    cursor_moved = QtCore.Signal(str)
    snap_options_changed = QtCore.Signal()
    hint_changed = QtCore.Signal(str)  # what to do next (status bar, bottom left)

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.settings = DrawSettings()
        self.tool: Tool = SelectTool(doc, self.settings)
        # snapping (ETABS Snap Options / CAD object snaps), shared by every window
        from .snap import SnapOptions
        self.snap = SnapOptions()
        self.snap_override: Optional[str] = None  # one-shot snap (Shift / Ctrl + right-click menu)
        # display
        self.show_labels = False
        self.extruded = False  # line view by default; extrusion is a display option
        self.nav_mode: Optional[str] = None  # toolbar navigation mode for left drag
        self.display_case: Optional[str] = None
        self.display_mode: Optional[int] = None
        self.display_diagram: Optional[str] = None
        self.deformed = False
        self.display_frame_color: Optional[str] = None  # colour members by P..M3 / Stress ratio
        self.show_mesh = False  # Display > Show Analysis Mesh
        self.show_reactions = False
        self.load_pattern: Optional[str] = None
        # which loads the Loads display shows: joint / frame / area / panel, or "applied"
        # (panel loads as the line loads they put on the members)
        self.load_kinds: Set[str] = {"joint", "frame", "area", "panel"}
        self.panels: Set[str] = set()  # selected load panels
        self.clipboard = None  # Edit > Copy (commands.Clipboard); Ctrl+V pastes it at the cursor
        # selection (shared by every window)
        self.frames: Set[str] = set()
        self.areas: Set[str] = set()
        self.points: Set[str] = set()
        self._previous: Tuple[Set[str], Set[str], Set[str]] = (set(), set(), set())
        doc.on_change(self._on_doc_change)
        self.selection_changed.connect(self.update_hint)

    # ----------------------------------------------------------- snapping
    # the three classic toggles stay available as properties (toolbar / tests)
    @property
    def snap_grid(self) -> bool:
        return self.snap.grid

    @snap_grid.setter
    def snap_grid(self, on: bool):
        self.snap.grid = bool(on)
        self.snap_options_changed.emit()

    @property
    def snap_points(self) -> bool:
        return self.snap.joints

    @snap_points.setter
    def snap_points(self, on: bool):
        self.snap.joints = bool(on)
        self.snap_options_changed.emit()

    @property
    def snap_midpoints(self) -> bool:
        return self.snap.ends_midpoints

    @snap_midpoints.setter
    def snap_midpoints(self, on: bool):
        self.snap.ends_midpoints = bool(on)
        self.snap_options_changed.emit()

    def set_snap_options(self, options):
        self.snap = options
        self.snap_options_changed.emit()

    # ----------------------------------------------------------- document
    def _on_doc_change(self, what: str):
        if what == "model":
            m = self.doc.model
            self.frames &= set(m.frames)
            self.areas &= set(m.areas)
            self.points &= set(m.points)
            self.panels &= set(getattr(m, "panel_loads", {}))
            # the tool may hold a stale model reference after undo/open
            self.tool.doc = self.doc

    # -------------------------------------------------------------- tools
    NAV_PROMPTS = {
        "rotate": "Rotate 3D View: drag with the left button (Esc to stop)",
        "pan": "Pan: drag with the left button (Esc to stop)",
        "zoom_window": "Zoom Window: drag a rectangle around the region to enlarge",
    }

    def set_nav_mode(self, mode: Optional[str]):
        """Left-drag navigation mode from the toolbar (ETABS Rotate 3D View / Pan /
        Zoom Window); ``None`` returns the left button to the current tool."""
        self.nav_mode = mode or None
        self.nav_changed.emit(self.nav_mode or "")
        self.status.emit(self.NAV_PROMPTS.get(self.nav_mode, self.tool.prompt))

    def set_tool(self, name: str):
        if self.nav_mode:
            self.set_nav_mode(None)
        cls = TOOLS.get(name, SelectTool)
        old = self.tool
        self.tool = cls(self.doc, self.settings)
        self.tool.set_plane(old.plane)
        picks = getattr(self.tool, "picks", ())
        if picks:  # load commands: the selection of their kind is what they act on
            self.keep_only(picks[0])
        self.tool.activate(self)
        self.tool_changed.emit(self.tool.name)
        self.update_hint()

    def update_hint(self, *_):
        try:
            text = self.tool.hint(self)
        except Exception:  # guidance must never break the app
            text = self.tool.prompt
        self.hint_changed.emit(text)

    def keep_only(self, kind: str):
        """Drop every selected object that is not of ``kind`` (point | frame | area)."""
        keep = {"point": "points", "frame": "frames", "area": "areas"}[kind]
        changed = False
        for attr in ("points", "frames", "areas", "panels"):
            if attr != keep and getattr(self, attr):
                setattr(self, attr, set())
                changed = True
        if changed:
            self.selection_changed.emit()

    def apply_load_tool(self) -> int:
        """Enter / right-click / Apply with a load command: apply the form to the
        picked objects (one undo step), then clear the picks."""
        tool = self.tool
        if not getattr(tool, "picks", None):
            return 0
        names = tool.picked(self)
        if not names:
            self.status.emit(f"{tool.title}: pick {tool.plural()} first (click or drag a window)")
            return 0
        try:
            msg = tool.apply_many(names)
        except Exception as exc:
            self.report(exc)
            return 0
        tool.editing = False
        self.clear_selection()
        self.status.emit(msg + " - pick more, or Esc")
        self.update_hint()
        return len(names)

    def cancel_tool(self):
        if self.nav_mode:
            self.set_nav_mode(None)
        elif self.tool.points:
            self.tool.cancel()  # first Esc ends the current chain
        elif getattr(self.tool, "picks", None) and self.count:
            self.clear_selection()  # load command: first Esc drops the picks
            self.tool.editing = False
        elif self.tool.name != "select":
            self.set_tool("select")
        elif self.count:
            self.clear_selection()  # select tool: Esc drops the selection

    # ---------------------------------------------------------- selection
    @property
    def selection(self) -> Set[str]:  # frames (backwards compatible name)
        return self.frames

    @property
    def count(self) -> int:
        return len(self.frames) + len(self.areas) + len(self.points) + len(self.panels)

    def set_selection(self, frames=(), areas=(), points=(), add: bool = False):
        if self.count:
            self._previous = (set(self.frames), set(self.areas), set(self.points))
        if add:
            self.frames |= set(frames)
            self.areas |= set(areas)
            self.points |= set(points)
        else:
            self.frames, self.areas, self.points = set(frames), set(areas), set(points)
            self.panels = set()
        self.selection_changed.emit()

    def toggle(self, kind: str, name: str, add: bool):
        target = {"frame": self.frames, "area": self.areas, "point": self.points, "panel": self.panels}[kind]
        if not add:
            if kind == "panel":
                self.set_selection()
                self.panels = {name}
                self.selection_changed.emit()
                return
            self.set_selection(**{kind + "s": [name]})
            return
        if name in target:
            target.discard(name)
        else:
            target.add(name)
        self.selection_changed.emit()

    def deselect(self, frames=(), areas=(), points=()):
        """Select > Deselect: remove objects from the selection."""
        self.set_selection(self.frames - set(frames), self.areas - set(areas), self.points - set(points))

    def clear_selection(self):
        self.set_selection()

    def select_all(self, visible=None):
        m = self.doc.model
        vis = visible or (lambda o: True)
        self.set_selection([f.name for f in m.frames.values() if vis(f)],
                           [a.name for a in m.areas.values() if vis(a)],
                           list(m.points) if visible is None else
                           [p for f in m.frames.values() if vis(f) for p in (f.point_i, f.point_j)])

    def invert_selection(self):
        m = self.doc.model
        self.set_selection(set(m.frames) - self.frames, set(m.areas) - self.areas, set(m.points) - self.points)

    def previous_selection(self):
        f, a, p = self._previous
        m = self.doc.model
        self.set_selection(f & set(m.frames), a & set(m.areas), p & set(m.points))

    def select_type(self, kind: str, add: bool = False):
        f, a, p = select_by_type(self.doc.model, kind)
        self.set_selection(f, a, p, add=add)

    def delete_selected(self) -> int:
        n = delete_objects(self.doc, self.frames, self.areas, self.points)
        if self.panels:
            names = set(self.panels)
            self.doc.edit(lambda m: [m.panel_loads.pop(p, None) for p in names])
            n += len(names)
        self.frames, self.areas, self.points, self.panels = set(), set(), set(), set()
        self.status.emit(f"Deleted {n} object(s)")
        self.selection_changed.emit()
        return n

    def selected(self) -> Tuple[List[str], List[str], List[str]]:
        return list(self.frames), list(self.areas), list(self.points)

    # ------------------------------------------------------------ display
    def set_display(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.display_changed.emit()

    def report(self, exc: Exception):
        if isinstance(exc, ModelError):
            self.error.emit(f"{exc.code}: {exc.message}")
        else:
            self.error.emit(str(exc))
