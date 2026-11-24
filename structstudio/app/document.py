"""Document: the model being edited, its file, undo/redo and results."""
from __future__ import annotations

import json
import pickle
from typing import Callable, List, Optional, Tuple

from ..model import Model
from ..model.model import nonfinite_points
from ..model.base import ModelError
from ..results import Results


class Document:
    MAX_UNDO = 100

    def __init__(self, model: Optional[Model] = None):
        self.model: Model = model or Model("Untitled", "kN m")
        self.path: Optional[str] = None
        self.dirty = False
        self.results: Optional[Results] = None
        self._undo: List[str] = []
        self._redo: List[str] = []
        self._listeners: List[Callable[[str], None]] = []
        # ETABS: after an analysis the model is locked (the results belong to it); an edit asks
        # to unlock, which deletes the results.  While the analysis runs nothing can be edited.
        self.locked = False
        self.busy = False
        self.ask_unlock: Optional[Callable[[], bool]] = None  # the window's "unlock?" question
        # revision: +1 on every change of the model object or its content (edit, undo / redo,
        # units, new / open / import) - caches key on it, never on the undo stack length.
        # generation: +1 when another document replaces this one (new / open / import):
        # anything pending from the old one (a half-drawn line, picks) must be dropped.
        self.revision = 0
        self.generation = 0
        self.notes: List[Tuple[str, str]] = []  # (level, text) said by the last command
        # duplicate / overlapping members made by the last command (drawing, copy, move,
        # mirror ...): the window warns and offers to undo it
        self.check_duplicates = True
        self.duplicates: list = []
        # Edit > Move Objects with Grid Lines (off: moving a grid line moves only the line -
        # on screen in Grid Mode or a new ordinate in the grid form)
        self.glue_grids = False
        self.active_tower = None  # several towers: the tower of the active window (new objects go there)
        self._geometry_before = None

    # snapshots for undo / redo: a pickle of the model is ~10x faster to make than
    # its JSON (the snapshot is taken before every edit) and gives the same model back
    @staticmethod
    def _snapshot(model: Model) -> bytes:
        return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _restore(snap) -> Model:
        if isinstance(snap, (bytes, bytearray)):
            return pickle.loads(snap)
        return Model.from_json(snap)  # older snapshots (JSON text)

    # ------------------------------------------------------------- events
    def on_change(self, callback: Callable[[str], None]):
        self._listeners.append(callback)

    def _emit(self, what: str):
        if what in ("model", "units", "replaced"):
            self.revision += 1
        for cb in list(self._listeners):
            cb(what)

    def note(self, text: str, level: str = "info"):
        """A command tells the user something about what it did (shown by the window
        after the edit: ``info`` in the log / status bar, ``warning`` in a message)."""
        self.notes.append((level, text))

    def _replaced(self):
        """Another document took this one's place (new / open / import)."""
        self.generation += 1
        self._emit("replaced")

    # --------------------------------------------------------------- edits
    def check_editable(self):
        """Raise ModelError when the model cannot be changed now (analysis running, or
        locked after the analysis and the user keeps the lock)."""
        if self.busy:
            raise ModelError("ANALYSIS_RUNNING", "The analysis is running - wait for it to finish or press Cancel")
        if self.locked:
            if self.ask_unlock is not None and self.ask_unlock():
                self.unlock()
                return
            raise ModelError("MODEL_LOCKED", "The model is locked after the analysis: Analyze > Unlock Model "
                                             "(the analysis results are deleted) to change it")

    def lock(self):
        if not self.locked:
            self.locked = True
            self._emit("lock")

    def unlock(self):
        """Unlock the model; its analysis results are deleted (they no longer match)."""
        if self.locked or self.results is not None:
            self.locked = False
            self.results = None
            self._emit("lock")
            self._emit("results")

    def begin_edit(self, check: bool = True):
        """Snapshot before a change (call before mutating the model)."""
        if check:
            self.check_editable()
        self.notes = []
        self.duplicates = []
        self._geometry_before = self._geometry() if self.check_duplicates else None
        self._undo.append(self._snapshot(self.model))
        if len(self._undo) > self.MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def _geometry(self):
        m = self.model
        P = {n: p.xyz for n, p in m.points.items()}
        frames = {n: (f.point_i, f.point_j, P.get(f.point_i), P.get(f.point_j)) for n, f in m.frames.items()}
        areas = {n: (tuple(a.points), tuple(P.get(q) for q in a.points)) for n, a in m.areas.items()}
        return frames, areas

    def _find_duplicates(self):
        """Duplicate / overlapping members among the frames and shells this edit made
        or moved."""
        before = self._geometry_before
        self._geometry_before = None
        if before is None:
            return
        try:
            from ..model.checks import drawing_issues
            fb, ab = before
            fa, aa = self._geometry()
            frames = [n for n, v in fa.items() if fb.get(n) != v]
            areas = [n for n, v in aa.items() if ab.get(n) != v]
            if frames or areas:
                self.duplicates = drawing_issues(self.model, frames, areas)
        except Exception:  # a check must never break an edit
            self.duplicates = []

    def end_edit(self, what: str = "model"):
        self._find_duplicates()
        self.model.sync_panels()  # area-load panels follow the edited areas
        self.model.prune_groups()  # groups keep no object that is gone
        self.dirty = True
        self.results = None  # model edits invalidate results
        if what not in ("model", "units", "replaced"):
            self.revision += 1  # an edit is an edit whatever the windows are told
        self._emit(what)

    def edit(self, func: Callable[[Model], None], what: str = "model"):
        """One command = one undo step.  If ``func`` fails half way the model is
        put back exactly as it was (no partial edit, no extra undo step)."""
        undo, redo = list(self._undo), list(self._redo)  # full history (begin_edit may drop the oldest)
        # a locked model (analysis results on screen): the command is worked out first and
        # the "unlock? the results are deleted" question comes only when it changes something
        # (OK in a form that was only looked at asked it, and threw the results away)
        locked = self.locked and not self.busy
        self.begin_edit(check=not locked)
        try:
            # several towers: new objects go to the tower of the window drawn in
            self.model.active_tower = getattr(self, "active_tower", None)
            try:
                func(self.model)
            finally:
                self.model.active_tower = None
            bad = nonfinite_points(self.model)
            if bad:  # never commit a model with inf / NaN coordinates (refused, never clamped)
                p = self.model.points[bad[0]]
                raise ModelError("NONFINITE_COORDINATE",
                                 f"The command would put point {bad[0]} at ({p.x!r}, {p.y!r}, {p.z!r}) - a coordinate "
                                 f"is not a finite number (or is beyond 1e9); nothing was changed")
            if self._unchanged(self._undo[-1], thorough=locked):
                # nothing changed: no undo step, the redo history, the results and the
                # "saved" state stay as they were
                self._undo, self._redo = undo, redo
                self._geometry_before = None
                return
            if locked:
                self.check_editable()  # asks; refused: MODEL_LOCKED, the model is put back
        except BaseException:
            self._rollback(undo, redo)
            raise
        self.end_edit(what)

    def _unchanged(self, before, thorough: bool = False) -> bool:
        """The model is as it was at the snapshot ``before``.  The pickles of an untouched
        model are equal byte for byte; ``thorough`` also sees a model whose objects were
        made again with the same values (a form that rebuilds what it shows)."""
        if not isinstance(before, (bytes, bytearray)):
            return False
        try:
            if self._snapshot(self.model) == bytes(before):
                return True
            return thorough and self._restore(before).to_dict() == self.model.to_dict()
        except Exception:  # (a check must never break an edit)
            return False

    def _rollback(self, undo: List[str], redo: List[str]):
        """Put the model back as it was before a failed command, with the undo / redo
        history exactly as it was (also when the history was full and begin_edit had
        dropped its oldest step - audit G05)."""
        # put back INTO the same model object: whatever holds it (the analysis results of a
        # locked model, a form that is still open) sees the model as it was, not the
        # half-changed one
        restored = self._restore(self._undo[-1])
        self.model.__dict__.clear()
        self.model.__dict__.update(restored.__dict__)
        self._undo, self._redo = undo, redo

    def edit_design(self, func: Callable[[Model], None], what: str = "design"):
        """A change of design data only (design preferences, Cd / Ie of the load cases,
        slab design data): undoable, but the model stays locked and the analysis results
        are kept - they do not depend on it (as ETABS keeps them for design overwrites)."""
        if self.busy:
            raise ModelError("ANALYSIS_RUNNING", "The analysis is running - wait for it to finish or press Cancel")
        res = self.results
        undo, redo = list(self._undo), list(self._redo)  # full history (begin_edit may drop the oldest)
        self.begin_edit(check=False)
        try:
            func(self.model)
            bad = nonfinite_points(self.model)
            if bad:
                raise ModelError("NONFINITE_COORDINATE", "A coordinate is not a finite number; nothing was changed")
            if self._unchanged(self._undo[-1], thorough=True):
                self._undo, self._redo = undo, redo  # nothing changed: no undo step
                self._geometry_before = None
                return
        except BaseException:
            self._rollback(undo, redo)
            raise
        self._geometry_before = None
        self.dirty = True
        self.results = res
        self._emit(what)

    # what the analysis does not depend on: design preferences, design strips, the
    # project information - undoing / redoing a change of these keeps the results
    DESIGN_KEYS = ("design", "design_strips", "info")

    def _analysis_part(self, model) -> dict:
        """The model as the analysis sees it: no design data, and no labels of joints /
        frames / shells (display names)."""
        d = {k: v for k, v in model.to_dict().items() if k not in self.DESIGN_KEYS}
        for table in ("points", "frames", "areas"):
            d[table] = {n: {f: v for f, v in o.items() if f != "label"} if isinstance(o, dict) else o
                        for n, o in (d.get(table) or {}).items()}
        return d

    def _design_only(self, other) -> bool:
        """``other`` (a model) differs from the current model in design data only."""
        if self.results is None and not self.locked:
            return False
        try:
            return self._analysis_part(other) == self._analysis_part(self.model)
        except Exception:  # (a check must never break undo)
            return False

    def replace_model(self, model: Model, what: str = "model"):
        """Swap in a new model (e.g. converted units) as one undoable edit."""
        self.begin_edit()
        self.model = model
        self.end_edit(what)

    def change_units(self, units, display: Optional[dict] = None) -> None:
        """Express the model in other units (Options > Display Units, the units
        button): every value is converted, nothing in the structure changes, the
        analysis results are kept (shown in the new units).  Undoable.  Emits
        "units" (windows rescale their views) and then "model"."""
        from ..units import UnitSystem
        units = units if isinstance(units, UnitSystem) else UnitSystem.parse(units)
        if units == self.model.units and (display is None or dict(display) == self.model.display_units):
            return
        old = self.model.units
        new = self.model.convert_units(units) if units != old else Model.from_dict(self.model.to_dict())
        if display is not None:  # what each kind of value is shown in (display only)
            from ..display_units import force_label
            new.display_units = {k: force_label(v) for k, v in dict(display).items() if v}
        res = self.results
        if self.busy:
            raise ModelError("ANALYSIS_RUNNING", "The analysis is running - wait for it to finish or press Cancel")
        self.begin_edit(check=False)  # units: the same structure (the results are kept, converted)
        self.model = new
        self.model.sync_panels()
        self.dirty = True
        self.results = res.in_units(new) if res is not None else None
        self.units_ratio = old.factor("length") / units.factor("length")  # new coordinates = old x this
        self.units_change = (old, units)
        self._emit("units")
        self._emit("model")

    def undo(self) -> bool:
        if not self._undo:
            return False
        return self._step(self._undo, self._redo)

    def redo(self) -> bool:
        if not self._redo:
            return False
        return self._step(self._redo, self._undo)

    def _step(self, source: list, target: list) -> bool:
        """Undo (source = the undo stack) or redo: a step that changed design data only
        (design strips, design preferences) keeps the analysis results and the lock."""
        if self.busy:
            raise ModelError("ANALYSIS_RUNNING", "The analysis is running - wait for it to finish or press Cancel")
        other = self._restore(source[-1])
        keep = self._design_only(other)
        if not keep:
            self.check_editable()
        res = self.results
        target.append(self._snapshot(self.model))
        source.pop()
        self.model = other
        self.results = res if keep else None
        if keep and res is not None:
            try:
                res.model = other  # (the results read the model they belong to)
            except Exception:
                pass
        self.dirty = True
        self._emit("model")
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # ---------------------------------------------------------------- files
    def new(self, units: str = "kN m"):
        self.model = Model("Untitled", units)
        self.locked = False
        self.path = None
        self.dirty = False
        self.results = None
        self._undo.clear()
        self._redo.clear()
        self._replaced()
        self._emit("model")

    def open(self, path: str):
        """Open a Structural Studio model, or import an ETABS / SAP2000 / SAFE file
        (text, tables, or native files converted by the installed CSI program)."""
        low = path.lower()
        if low.endswith(".json"):
            self.model = Model.from_json(path)
            self.locked = False
            self.path = path
            self.dirty = False
            report = ""
            self.results = None
            self._undo.clear()
            self._redo.clear()
            self._replaced()
            self._emit("model")
            return report
        from ..io.importers import import_model
        res = import_model(path)
        return self.set_imported(res.model, res.report)

    def set_imported(self, model: Model, report: str = "") -> str:
        """Replace the model by an imported one (unsaved, no undo history)."""
        model.convert_area_loads()
        self.model = model
        self.locked = False
        self.path = None
        self.dirty = True
        self.results = None
        self._undo.clear()
        self._redo.clear()
        self._replaced()
        self._emit("model")
        return report

    def save(self, path: Optional[str] = None):
        path = path or self.path
        if not path:
            raise ValueError("no file name")
        self.model.to_json(path)
        self.path = path
        self.dirty = False
        self._emit("file")

    # ------------------------------------------------------------- results
    def set_results(self, results: Results, lock: bool = True):
        self.results = results
        self._emit("results")
        if lock and results is not None:
            self.lock()

    def analysis_mesh(self):
        """The analysis model (mesh) of the current model: from the results when
        there are any, otherwise built now and kept until the model changes.
        None if the model cannot be meshed yet (incomplete)."""
        if self.results is not None:
            return self.results.am
        key = id(self.model), self.revision
        cache = getattr(self, "_mesh_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        from ..mesh.mesher import build_analysis_model
        try:
            am = build_analysis_model(self.model)
        except Exception:
            am = None
        self._mesh_cache = (key, am)
        return am

    @property
    def title(self) -> str:
        name = self.path.split("/")[-1].split("\\")[-1] if self.path else "Untitled"
        return name + (" *" if self.dirty else "")
