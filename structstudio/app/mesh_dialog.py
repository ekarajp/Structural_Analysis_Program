"""Assign > Floor / Wall Auto Mesh Options.

* Mesh of the selected floors / walls: **Auto mesh** (the program's element size h)
  or **No mesh** (each area is one element; joints / lines inside are not connected).
* Fine mesh (the whole model, all automatically meshed floors and walls): found once by
  **Find Fine Mesh** (Analyze > Check Mesh), shown with its element size, then **Use fine
  mesh** can be ticked any time without finding it again.
* "?" buttons explain auto mesh and fine mesh.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtCore, QtWidgets

from ..model import Model

AUTO_MESH_HELP = (
    "<b>Auto mesh</b><br><br>"
    "Floors and walls are divided into shell elements for the analysis.  With auto mesh the "
    "program chooses the element size <b>h</b> from the geometry of each floor / wall: at least 8 "
    "elements across it and 6 between neighbouring columns, 0.1 to 1.0 m, rounded down to 1, 0.5, "
    "0.25 ... m.  Columns, beams, walls, point loads and openings inside the area are joined to the "
    "mesh (their joints become mesh nodes).  Quick, and accurate enough for most models.<br><br>"
    "<b>Maximum element size</b>: an upper limit for h in these areas (blank: the program's own).<br><br>"
    "<b>No mesh</b>: each area is one element as drawn.  Joints, columns or lines inside it are "
    "<i>not</i> connected to it - use it only for areas that are already divided as you want them.")

FINE_MESH_HELP = (
    "<b>Fine mesh</b><br><br>"
    "<b>Find Fine Mesh</b> looks for the mesh at which the results stop changing (h-convergence): "
    "it solves the linear static load cases on meshes of size 4h, 2h and h and, while the estimated "
    "error of the largest slab / wall deflection is above the allowed error (2 % by default, "
    "Analyze &gt; Analysis Model Defaults), halves the size - h/2, h/4 ... (at most 12,000 shell "
    "elements).  It takes longer than one analysis.  The size found is kept with the model and "
    "shown here.<br><br>"
    "<b>Use fine mesh</b> (possible once it has been found): every analysis uses that size, for all "
    "automatically meshed floors and walls - it is a setting of the whole model.  It is never "
    "coarser than the automatic mesh.  After changing floors, walls, frames or supports, find it "
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


class AreaMeshDialog(QtWidgets.QDialog):
    """``kind``: "Floor" or "Wall".  ``find_fine_mesh()`` starts Analyze > Check Mesh (in the
    background); call :meth:`refresh` when it has finished."""

    def __init__(self, kind: str, model: Model, auto_mesh: bool, max_size: Optional[float],
                 find_fine_mesh: Callable[[], None], parent=None):
        super().__init__(parent)
        self.kind = kind
        self.model = model
        self.setWindowTitle(f"{kind} Auto Mesh Options")
        lay = QtWidgets.QVBoxLayout(self)

        g = QtWidgets.QGroupBox(f"Mesh of the selected {kind.lower()}s")
        gl = QtWidgets.QGridLayout(g)
        self.auto = QtWidgets.QRadioButton("Auto mesh")
        self.none = QtWidgets.QRadioButton("No mesh (each area is one element)")
        (self.auto if auto_mesh else self.none).setChecked(True)
        gl.addWidget(self.auto, 0, 0)
        gl.addWidget(help_button(AUTO_MESH_HELP, "Auto mesh", self), 0, 1)
        gl.addWidget(self.none, 1, 0, 1, 2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Maximum element size"))
        self.size = QtWidgets.QLineEdit("" if max_size is None else f"{max_size:g}")
        self.size.setPlaceholderText("program default")
        row.addWidget(self.size, 1)
        row.addWidget(QtWidgets.QLabel(model.units.length))
        gl.addLayout(row, 2, 0, 1, 2)
        self.auto.toggled.connect(self.size.setEnabled)
        self.size.setEnabled(self.auto.isChecked())
        gl.setColumnStretch(0, 1)
        lay.addWidget(g)

        f = QtWidgets.QGroupBox("Fine mesh (whole model: all auto-meshed floors and walls)")
        fl = QtWidgets.QGridLayout(f)
        self.use_fine = QtWidgets.QCheckBox("Use fine mesh")
        self.find = QtWidgets.QPushButton("Find Fine Mesh...")
        self.find.setToolTip("Analyze > Check Mesh: solves the model on finer and finer meshes (slower than "
                             "one analysis) and keeps the size found")
        self.find.clicked.connect(lambda: (self.find.setEnabled(False), find_fine_mesh()))
        self.found = QtWidgets.QLabel()
        self.found.setWordWrap(True)
        fl.addWidget(self.use_fine, 0, 0)
        fl.addWidget(self.find, 0, 1)
        fl.addWidget(help_button(FINE_MESH_HELP, "Fine mesh", self), 0, 2)
        fl.addWidget(self.found, 1, 0, 1, 3)
        fl.setColumnStretch(0, 1)
        lay.addWidget(f)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.use_fine.setChecked(bool(model.options.use_fine_mesh and model.options.fine_mesh_factor))
        self.refresh()

    def refresh(self, *_):
        """After Find Fine Mesh: the size found; Use fine mesh can be ticked."""
        o = self.model.options
        have = bool(o.fine_mesh_factor)
        self.find.setEnabled(True)
        self.find.setText("Find Again..." if have else "Find Fine Mesh...")
        self.use_fine.setEnabled(have)
        if not have:
            self.use_fine.setChecked(False)
            self.use_fine.setText("Use fine mesh")
            self.found.setText("Not found yet: press Find Fine Mesh first.")
        else:
            size = fine_mesh_size_text(self.model)
            self.use_fine.setText(f"Use fine mesh ({size})")
            from ..api import mesh_signature
            stale = bool(o.fine_mesh_signature) and o.fine_mesh_signature != mesh_signature(self.model)
            self.found.setText(f"Found: element size {size}." + (
                "  <b>The model has changed since - find it again to be sure.</b>" if stale else ""))
            self.found.setTextFormat(QtCore.Qt.RichText)

    def values(self) -> dict:
        t = self.size.text().strip()
        size = None
        if t:
            size = float(t)
            import math
            if not math.isfinite(size) or size <= 0:
                raise ValueError(f"Maximum element size must be a positive number, not {t!r}")
        return {"auto": self.auto.isChecked(), "size": size, "use_fine": self.use_fine.isChecked()}

    def accept(self):
        try:
            self.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        super().accept()
