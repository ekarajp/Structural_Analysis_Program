"""Assign > Floor / Wall Auto Mesh Options.

One choice for the selected floors / walls:

* **Auto mesh** - the program's element size h (read-only "Default = max 1.2 m": the
  largest element of the automatic mesh, or the size set in Analyze > Automatic Mesh
  Settings);
* **Manual maximum mesh size** - the user's own largest element size for these areas;
* **Fine mesh** - the size found by **Find Fine Mesh** (h-convergence, Analyze > Check
  Mesh); possible only once it has been found.  Its allowed error (the deflection error
  limit, ``model.options.mesh_refine_tolerance``) is set here;
* **No mesh** - each area is one element; joints / lines inside are not connected.

"?" buttons explain auto mesh and fine mesh.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtCore, QtWidgets

from ..model import Model

AUTO_MESH_HELP = (
    "<b>Auto mesh</b><br><br>"
    "Floors and walls are divided into shell elements for the analysis.  With auto mesh the "
    "program chooses the element size <b>h</b> from the geometry of each floor / wall: at least 8 "
    ""
    "elements: each span between columns, beams and walls is split into equal elements no larger "
    "than the default maximum (1.2 m, or the size of Analyze &gt; Automatic Mesh Settings).  Columns, beams, walls, point loads and openings inside the area are joined to the "
    "mesh (their joints become mesh nodes).  Quick, and accurate enough for most models.<br><br>"
    "<b>Manual maximum mesh size</b>: your own largest element size for these areas instead of the "
    "default.<br><br>"
    "<b>No mesh</b>: each area is one element as drawn.  Joints, columns or lines inside it are "
    "<i>not</i> connected to it - use it only for areas that are already divided as you want them.")

FINE_MESH_HELP = (
    "<b>Fine mesh</b><br><br>"
    "<b>Find Fine Mesh</b> looks for the mesh at which the results stop changing (h-convergence): "
    "it solves the linear static load cases on meshes of size 4h, 2h and h and, while the estimated "
    "error of the largest slab / wall deflection is above the allowed error (2 % by default, set "
    "in this dialog), halves the size - h/2, h/4 ... (at most 12,000 shell "
    "elements).  It takes longer than one analysis.  The size found is kept with the model and "
    "shown here.<br><br>"
    "<b>Fine mesh</b> (possible once it has been found): the selected floors / walls are meshed "
    "with that size (the auto size h x the factor found).  It is never coarser than the automatic "
    "mesh.  Areas in one plane share their mesh lines, so they all get the finest size among them.  After changing floors, walls, frames or supports, find it "
    "again (the analysis warns when the model changed since).<br><br>"
    "Slab results near columns and point loads converge slowly: the fine mesh matters most for "
    "punching, deflections and slab reinforcement.")


def fine_mesh_size_text(model: Model) -> str:
    """'0.25 m' / '0.125-0.25 m' in the model's length units; '' if not found yet."""
    o = model.options
    if not o.fine_mesh_factor:
        return ""
    lo, hi = o.fine_mesh_size_min, o.fine_mesh_size_max
    u = model.units.length
    if lo is None:
        return f"h x {o.fine_mesh_factor:g}"
    if hi is None or abs(hi - lo) <= 1e-9 * max(abs(hi), 1.0):
        return f"{lo:.3g} {u}"
    return f"{lo:.3g}-{hi:.3g} {u}"


def help_button(text: str, title: str, parent: QtWidgets.QWidget) -> QtWidgets.QToolButton:
    b = QtWidgets.QToolButton(parent)
    b.setText("?")
    b.setObjectName("helpButton")
    b.setToolTip(f"What is {title.lower()}?")
    b.setStyleSheet("QToolButton#helpButton { font-weight: 700; border: 1px solid palette(mid); border-radius: 9px;"
                    " min-width: 16px; max-width: 16px; min-height: 16px; max-height: 16px; padding: 0px; }")
    b.clicked.connect(lambda: QtWidgets.QMessageBox.information(parent, title, text))
    return b


MODES = ("auto", "manual", "fine", "none")


def default_max_size(model: Model, kind: str) -> float:
    """Largest element of the automatic mesh, in the model's length units: the size of
    Analyze > Automatic Mesh Settings for Floors / Walls if set, else 1.2 m."""
    o = model.options
    v = o.wall_mesh_max_size if kind == "Wall" else o.floor_mesh_max_size
    if v:
        return float(v)
    from ..mesh.area_mesh import DEFAULT_MAX_SIZE
    return DEFAULT_MAX_SIZE / model.units.factor("length")


def area_mode(area) -> str:
    if not area.auto_mesh:
        return "none"
    if getattr(area, "fine_mesh", False):
        return "fine"
    if area.max_mesh_size:
        return "manual"
    return "auto"


class AreaMeshDialog(QtWidgets.QDialog):
    """``kind``: "Floor" or "Wall"; ``mode`` one of :data:`MODES`.  ``find_fine_mesh()``
    starts Analyze > Check Mesh (in the background); call :meth:`refresh` when it has
    finished."""

    def __init__(self, kind: str, model: Model, mode: str, max_size: Optional[float],
                 find_fine_mesh: Callable[[], None], parent=None):
        super().__init__(parent)
        self.kind = kind
        self.model = model
        self.setWindowTitle(f"{kind} Auto Mesh Options")
        lay = QtWidgets.QVBoxLayout(self)
        u = model.units.length

        g = QtWidgets.QGroupBox(f"Mesh of the selected {kind.lower()}s (choose one)")
        gl = QtWidgets.QGridLayout(g)
        self.group = QtWidgets.QButtonGroup(self)
        self.auto = QtWidgets.QRadioButton("Auto mesh")
        self.manual = QtWidgets.QRadioButton("Manual maximum mesh size")
        self.fine = QtWidgets.QRadioButton("Fine mesh")
        self.none = QtWidgets.QRadioButton("No mesh (each area is one element)")
        for i, b in enumerate((self.auto, self.manual, self.fine, self.none)):
            self.group.addButton(b, i)
        # Auto mesh  [Default = max 1.2 m]  ?
        gl.addWidget(self.auto, 0, 0)
        self.default = QtWidgets.QLineEdit(f"Default = max {default_max_size(model, kind):.4g} {u}")
        self.default.setReadOnly(True)
        self.default.setFocusPolicy(QtCore.Qt.NoFocus)
        self.default.setToolTip("Largest element of the automatic mesh (Analyze > Automatic Mesh Settings)")
        self.default.setStyleSheet("QLineEdit { background: palette(window); color: palette(mid); }")
        gl.addWidget(self.default, 0, 1, 1, 2)
        gl.addWidget(help_button(AUTO_MESH_HELP, "Auto mesh", self), 0, 3)
        # Manual maximum mesh size  [ 0.5 ] m
        gl.addWidget(self.manual, 1, 0)
        self.size = QtWidgets.QLineEdit("" if max_size is None else f"{max_size:g}")
        self.size.setPlaceholderText("maximum size")
        gl.addWidget(self.size, 1, 1)
        gl.addWidget(QtWidgets.QLabel(u), 1, 2)
        # Fine mesh  [Find Fine Mesh...]  ?
        gl.addWidget(self.fine, 2, 0)
        self.find = QtWidgets.QPushButton("Find Fine Mesh...")
        self.find.setToolTip("Solves the model on finer and finer meshes (slower than one analysis) "
                             "and keeps the size found")
        self.find.clicked.connect(self._find)
        self._find_fn = find_fine_mesh
        gl.addWidget(self.find, 2, 1, 1, 2)
        gl.addWidget(help_button(FINE_MESH_HELP, "Fine mesh", self), 2, 3)
        self.found = QtWidgets.QLabel()
        self.found.setWordWrap(True)
        self.found.setTextFormat(QtCore.Qt.RichText)
        gl.addWidget(self.found, 3, 0, 1, 4)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Fine mesh: allowed deflection error <"))
        self.tol = QtWidgets.QDoubleSpinBox()
        self.tol.setRange(0.1, 50.0)
        self.tol.setDecimals(1)
        self.tol.setSingleStep(0.5)
        self.tol.setSuffix(" %")
        self.tol.setValue(100.0 * float(getattr(model.options, "mesh_refine_tolerance", 0.02) or 0.02))
        self.tol.setToolTip("Find Fine Mesh refines until the estimated error of the largest slab / wall "
                            "deflection is below this")
        row.addWidget(self.tol)
        row.addStretch(1)
        gl.addLayout(row, 4, 0, 1, 4)
        gl.addWidget(self.none, 5, 0, 1, 4)
        gl.setColumnStretch(1, 1)
        lay.addWidget(g)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        {"auto": self.auto, "manual": self.manual, "fine": self.fine}.get(mode, self.none).setChecked(True)
        self.group.idToggled.connect(lambda *_: self._enable())
        self.refresh()

    def _find(self):
        self.apply_tolerance()  # Find uses the allowed error shown
        self.find.setEnabled(False)
        self._find_fn()

    def apply_tolerance(self):
        o = self.model.options
        t = round(self.tol.value() / 100.0, 6)
        if abs(float(getattr(o, "mesh_refine_tolerance", 0.02) or 0.02) - t) > 1e-9:
            o.mesh_refine_tolerance = t
            return True
        return False

    def _enable(self):
        self.size.setEnabled(self.manual.isChecked())
        if self.manual.isChecked() and not self.size.text().strip():
            self.size.setFocus()

    def refresh(self, *_):
        """After Find Fine Mesh: the size found; Fine mesh can be chosen."""
        o = self.model.options
        have = bool(o.fine_mesh_factor)
        self.find.setEnabled(True)
        self.find.setText("Find Again..." if have else "Find Fine Mesh...")
        self.fine.setEnabled(have)
        if not have:
            if self.fine.isChecked():
                self.auto.setChecked(True)
            self.fine.setText("Fine mesh")
            self.found.setText("Fine mesh: not found yet - press Find Fine Mesh first.")
        else:
            size = fine_mesh_size_text(self.model)
            self.fine.setText(f"Fine mesh ({size})")
            from ..api import mesh_signature
            stale = bool(o.fine_mesh_signature) and o.fine_mesh_signature != mesh_signature(self.model)
            self.found.setText(f"Fine mesh found: element size {size}." + (
                "  <b>The model has changed since - find it again to be sure.</b>" if stale else ""))
        self._enable()

    def mode(self) -> str:
        return MODES[max(self.group.checkedId(), 0)]

    def values(self) -> dict:
        mode = self.mode()
        size = None
        if mode == "manual":
            t = self.size.text().strip()
            import math
            try:
                size = float(t)
            except ValueError:
                raise ValueError(f"Manual maximum mesh size: enter a positive number, not {t!r}") from None
            if not math.isfinite(size) or size <= 0:
                raise ValueError(f"Manual maximum mesh size must be a positive number, not {t!r}")
        return {"mode": mode, "auto": mode != "none", "size": size, "fine": mode == "fine",
                "tolerance": round(self.tol.value() / 100.0, 6)}

    def accept(self):
        try:
            self.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        super().accept()
