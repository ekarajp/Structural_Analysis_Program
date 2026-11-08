"""Document: the model being edited, its file, undo/redo and results."""
from __future__ import annotations

import json
from typing import Callable, List, Optional

from ..model import Model
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

    # ------------------------------------------------------------- events
    def on_change(self, callback: Callable[[str], None]):
        self._listeners.append(callback)

    def _emit(self, what: str):
        for cb in list(self._listeners):
            cb(what)

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
        self._undo.append(self.model.to_json(indent=0))
        if len(self._undo) > self.MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def end_edit(self, what: str = "model"):
        self.model.sync_panels()  # area-load panels follow the edited areas
        self.dirty = True
        self.results = None  # model edits invalidate results
        self._emit(what)

    def edit(self, func: Callable[[Model], None], what: str = "model"):
        self.begin_edit()
        func(self.model)
        self.end_edit(what)

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
        self._emit("units")
        self._emit("model")

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.check_editable()
        self._redo.append(self.model.to_json(indent=0))
        self.model = Model.from_json(self._undo.pop())
        self.results = None
        self.dirty = True
        self._emit("model")
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self.check_editable()
        self._undo.append(self.model.to_json(indent=0))
        self.model = Model.from_json(self._redo.pop())
        self.results = None
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
        key = id(self.model), len(self._undo), len(self._redo), self.dirty
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
