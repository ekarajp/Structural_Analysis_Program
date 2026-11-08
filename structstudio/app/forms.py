"""Small generic forms used by the ETABS-style menu commands.

``FormDialog(title, fields)`` builds a dialog from field specs::

    ("angle", "Rotation angle (deg)", 0.0)                 # float
    ("n", "Number", 2, {"min": 1, "max": 100})             # int spin box
    ("pin", "Pinned", False)                               # check box
    ("dir", "Direction", "X", {"choices": ["X", "Y"]})     # combo box
    ("size", "Max size", None, {"optional": True})          # float or blank = None
    ("name", "Label", "P1")                                # text

``values()`` returns {key: value}.  Groups of fields can be separated with
``("-", "Group title")``.

A float field with ``{"item": "line_load"}`` (a display item of
:mod:`structstudio.display_units`) is shown and typed in that item's display
unit (its label gets ``[kgf/m]``) and returned in the model's units.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from PySide6 import QtCore, QtWidgets

# the display units of the open model (set by the main window when the model changes)
DISPLAY = {"units": None}


def display_units():
    return DISPLAY.get("units")


class FormDialog(QtWidgets.QDialog):
    def __init__(self, title: str, fields: Sequence[tuple], parent=None, note: Optional[str] = None,
                 apply_button: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        self._kinds: Dict[str, str] = {}
        self._factors: Dict[str, float] = {}  # model value x factor = shown value
        lay = QtWidgets.QVBoxLayout(self)
        if note:
            lab = QtWidgets.QLabel(note)
            lab.setWordWrap(True)
            lay.addWidget(lab)
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)
        for spec in fields:
            key, label = spec[0], spec[1]
            if key == "-":
                box = QtWidgets.QLabel(f"<b>{label}</b>")
                form.addRow(box)
                continue
            default = spec[2] if len(spec) > 2 else None
            opts: Dict[str, Any] = spec[3] if len(spec) > 3 else {}
            du = display_units()
            if opts.get("item") and du is not None:
                k = du.factor(opts["item"])
                self._factors[key] = k
                if isinstance(default, (int, float)) and not isinstance(default, bool):
                    default = float(default) * k
                    opts = dict(opts, float=True)
                label = f"{label} [{du.label(opts['item'])}]"
            w, kind = self._editor(default, opts)
            self._widgets[key] = w
            self._kinds[key] = kind
            from .symbols import pretty, rich
            if kind == "bool":
                w.setText(pretty(label))
                form.addRow(w)
            else:
                lab = QtWidgets.QLabel(rich(label))  # m², f′c, I33 with subscripts, φ ...
                lab.setTextFormat(QtCore.Qt.RichText)
                form.addRow(lab, w)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @staticmethod
    def _editor(default, opts):
        if "choices" in opts:
            w = QtWidgets.QComboBox()
            w.addItems([str(c) for c in opts["choices"]])
            if default is not None and str(default) in [str(c) for c in opts["choices"]]:
                w.setCurrentText(str(default))
            return w, "choice"
        if isinstance(default, bool):
            w = QtWidgets.QCheckBox()
            w.setChecked(default)
            return w, "bool"
        if isinstance(default, int) and not opts.get("float"):
            w = QtWidgets.QSpinBox()
            w.setRange(int(opts.get("min", -10 ** 6)), int(opts.get("max", 10 ** 6)))
            w.setValue(default)
            return w, "int"
        if isinstance(default, float) or opts.get("optional") or opts.get("float"):
            w = QtWidgets.QLineEdit("" if default is None else f"{float(default):.10g}")
            if opts.get("optional"):
                w.setPlaceholderText("(none)")
            return w, "float?" if opts.get("optional") else "float"
        w = QtWidgets.QLineEdit("" if default is None else str(default))
        return w, "text"

    def values(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, w in self._widgets.items():
            kind = self._kinds[key]
            if kind == "bool":
                out[key] = w.isChecked()
            elif kind == "int":
                out[key] = w.value()
            elif kind == "choice":
                out[key] = w.currentText()
            elif kind in ("float", "float?"):
                t = w.text().strip()
                if not t:
                    out[key] = None if kind == "float?" else 0.0
                else:
                    out[key] = float(t) / self._factors.get(key, 1.0)
            else:
                out[key] = w.text()
        return out

    def set_value(self, key: str, value):
        w, kind = self._widgets[key], self._kinds[key]
        if kind == "bool":
            w.setChecked(bool(value))
        elif kind == "int":
            w.setValue(int(value))
        elif kind == "choice":
            w.setCurrentText(str(value))
        else:
            if isinstance(value, float):
                value = value * self._factors.get(key, 1.0)
            w.setText("" if value is None else (f"{value:g}" if isinstance(value, float) else str(value)))


def ask(parent, title: str, fields: Sequence[tuple], note: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Show a :class:`FormDialog`; the values, or None when cancelled."""
    d = FormDialog(title, fields, parent, note)
    if d.exec() != QtWidgets.QDialog.Accepted:
        return None
    try:
        return d.values()
    except ValueError as exc:
        QtWidgets.QMessageBox.warning(parent, title, f"Invalid number: {exc}")
        return None


class TabbedFormDialog(QtWidgets.QDialog):
    """Several :class:`FormDialog`-style pages in tabs (e.g. Edit > Replicate)."""

    def __init__(self, title: str, pages: Sequence[tuple], parent=None, extra: Sequence[tuple] = ()):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.pages: List[FormDialog] = []
        self.keys: List[str] = []
        for key, label, fields in pages:
            page = FormDialog(label, fields, self)
            page.layout().itemAt(page.layout().count() - 1).widget().hide()  # its own OK / Cancel
            page.setWindowFlags(QtCore.Qt.Widget)
            self.tabs.addTab(page, label)
            self.pages.append(page)
            self.keys.append(key)
        lay.addWidget(self.tabs)
        self.extra = FormDialog("", extra, self) if extra else None
        if self.extra is not None:
            self.extra.layout().itemAt(self.extra.layout().count() - 1).widget().hide()
            self.extra.setWindowFlags(QtCore.Qt.Widget)
            lay.addWidget(self.extra)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result_values(self):
        i = self.tabs.currentIndex()
        vals = self.pages[i].values()
        if self.extra is not None:
            vals.update(self.extra.values())
        return self.keys[i], vals
