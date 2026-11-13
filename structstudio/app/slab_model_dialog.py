"""Design > Concrete Slab Design > Slab Design Model...

Takes the membrane floors of the chosen stories out of the building (see
``structstudio.design.slab_model``), opens each in its own window and analyses it -
the slab moments to design the slabs for, shown with the usual Display tools."""
from __future__ import annotations

import dataclasses
from typing import List

from PySide6 import QtCore, QtWidgets

from ..design.slab_model import build_slab_model, membrane_stories
from ..model.base import ModelError

HELP = ("A membrane floor only hands its loads to the beams and columns of the building model, so it has no "
        "slab moments of its own.  The Slab Design Model takes the floor out of the building (a separate "
        "slab model): its slabs as plates, the beams of the level, the columns and walls one story up and down "
        "fixed at their far ends, the gravity loads and combinations.  Each story opens in its own window and is "
        "analysed; the building model is not changed.  Shell floors are designed with the forces of the "
        "building model.")


class SlabModelDialog(QtWidgets.QDialog):
    def __init__(self, win, parent=None):
        super().__init__(parent or win)
        self.win = win
        self.doc = win.doc
        self.setWindowTitle("Slab Design Model")
        self.resize(560, 520)
        p = self.doc.model.design.slab_model
        lay = QtWidgets.QVBoxLayout(self)
        lab = QtWidgets.QLabel(HELP)
        lab.setWordWrap(True)
        lay.addWidget(lab)
        lay.addWidget(QtWidgets.QLabel("<b>Stories with membrane floors</b>"))
        self.stories = QtWidgets.QListWidget()
        names = membrane_stories(self.doc.model)
        for n in reversed(names):  # top down, as the story list
            it = QtWidgets.QListWidgetItem(n)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if n in p.stories else QtCore.Qt.Unchecked)
            self.stories.addItem(it)
        if names and not any(n in p.stories for n in names):
            self.stories.item(0).setCheckState(QtCore.Qt.Checked)
        lay.addWidget(self.stories, 1)
        if not names:
            lay.addWidget(QtWidgets.QLabel("No story has a membrane floor slab - shell floors are designed with the "
                                           "forces of the building model."))
        form = QtWidgets.QFormLayout()
        self.beams = QtWidgets.QComboBox()
        self.beams.addItem("Flexible - the beams' own stiffness", "flexible")
        self.beams.addItem("Rigid - unyielding line supports (coefficient method)", "rigid")
        self.beams.setCurrentIndex(0 if p.beams == "flexible" else 1)
        self.columns = QtWidgets.QComboBox()
        self.columns.addItem("Fixed at the far end (one story up and down)", "fixed")
        self.columns.addItem("Pinned at the far end", "pinned")
        self.columns.setCurrentIndex(0 if p.columns == "fixed" else 1)
        self.shell = QtWidgets.QCheckBox("Keep the shell slabs of the same level (the slab is continuous)")
        self.shell.setChecked(p.include_shell)
        form.addRow("Beams", self.beams)
        form.addRow("Columns / walls", self.columns)
        form.addRow("", self.shell)
        lay.addLayout(form)
        self.run_box = QtWidgets.QCheckBox("Run the analysis of each new model")
        self.run_box.setChecked(True)
        lay.addWidget(self.run_box)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Create")
        bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(bool(names))
        bb.accepted.connect(self.create)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.windows: List = []

    def chosen(self) -> List[str]:
        return [self.stories.item(i).text() for i in range(self.stories.count())
                if self.stories.item(i).checkState() == QtCore.Qt.Checked]

    def create(self):
        stories = self.chosen()
        if not stories:
            QtWidgets.QMessageBox.information(self, "Slab Design Model", "Tick at least one story.")
            return
        beams, columns, shell = self.beams.currentData(), self.columns.currentData(), self.shell.isChecked()
        prefs = dataclasses.replace(self.doc.model.design.slab_model, beams=beams, columns=columns,
                                    include_shell=shell, stories=stories)
        if prefs != self.doc.model.design.slab_model:
            def apply(model):
                model.design.slab_model = prefs
            try:
                self.doc.edit_design(apply)
            except ModelError as exc:
                QtWidgets.QMessageBox.warning(self, "Slab Design Model", exc.message)
                return
        made = []
        for st in stories:
            try:
                res = build_slab_model(self.doc.model, st, beams=beams, columns=columns, include_shell=shell)
            except ModelError as exc:
                QtWidgets.QMessageBox.warning(self, "Slab Design Model", f"{st}: {exc.message}")
                continue
            made.append(res)
        for res in made:
            w = open_slab_model_window(self.win, res, run=self.run_box.isChecked())
            self.windows.append(w)
        if made:
            self.accept()


def open_slab_model_window(win, res, run: bool = True):
    """A new main window holding the slab design model ``res`` (SlabModelResult)."""
    from .document import Document
    from .mainwindow import MainWindow
    doc = Document(res.model)
    doc.dirty = False
    w = MainWindow(doc, getattr(win, "qsettings", None))
    w.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    keep = getattr(win, "_slab_model_windows", None)
    if keep is None:
        keep = win._slab_model_windows = []
    keep.append(w)
    w.destroyed.connect(lambda *_: keep.remove(w) if w in keep else None)
    w.show()
    for n in res.notes:
        try:
            w.logger.log(f"Slab Design Model: {n}")
        except Exception:
            pass
    w.statusBar().showMessage(f"Slab Design Model of {res.story}: " + res.notes[0], 15000)
    if run:
        QtCore.QTimer.singleShot(0, w.run_analysis)
    return w
