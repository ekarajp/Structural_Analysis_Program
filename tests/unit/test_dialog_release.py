"""Forms run with exec() are freed when nothing holds them any more (every opening of
Define > Frame Sections kept about 4 MB until the program closed)."""
import gc
import weakref

import shiboken6
from PySide6 import QtCore, QtWidgets

from structstudio.app import dialog_release


class _Form(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.button = QtWidgets.QPushButton("x", self)
        self.button.clicked.connect(lambda: self.setWindowTitle("clicked"))  # a callback over the form
        self.edit = QtWidgets.QLineEdit(self)
        self.edit.textChanged.connect(self.setWindowTitle)  # a bound method of the form


def _run(form):
    QtCore.QTimer.singleShot(0, form.reject)
    form.exec()


def _settle(app):
    for _ in range(3):
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    gc.collect()


def test_a_form_nobody_holds_is_deleted_and_a_held_one_is_kept():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog_release.install()
    parent = QtWidgets.QWidget()

    def open_and_forget():
        f = _Form(parent)
        _run(f)
        return weakref.ref(f)
    ref = open_and_forget()
    _settle(app)
    assert ref() is None or not shiboken6.isValid(ref())
    assert not parent.findChildren(_Form)

    kept = _Form(parent)  # a form a command still holds (an attribute, a variable)
    _run(kept)
    _settle(app)
    assert shiboken6.isValid(kept) and kept.button.text() == "x"

    def open_and_keep_a_callback(holder):
        f = _Form(parent)
        holder.append((lambda form: (lambda: form.windowTitle()))(f))  # kept by something else
        _run(f)
        return weakref.ref(f)
    holder = []
    ref2 = open_and_keep_a_callback(holder)
    _settle(app)
    assert shiboken6.isValid(ref2()) and holder[0]() == ""
    parent.deleteLater()


class _Helper:
    def __init__(self, form):
        self.form = form


def test_a_form_whose_own_parts_point_back_at_it_is_deleted_too():
    """The fit-to-screen filter, a helper object and a child widget each keep a
    reference to their form; with a callback over the form on top, Python's collector
    cannot free it - it is deleted like any other form nobody holds."""
    from structstudio.app import fit
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog_release.install()
    parent = QtWidgets.QWidget()

    def open_and_forget():
        f = _Form(parent)
        fit.install(f)
        f.helper = _Helper(f)
        f.button.owner = f
        _run(f)
        return weakref.ref(f)
    ref = open_and_forget()
    _settle(app)
    assert ref() is None or not shiboken6.isValid(ref())
    assert not parent.findChildren(_Form)
    parent.deleteLater()
