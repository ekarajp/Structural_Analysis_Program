"""fit.install: the event filter that keeps a dialog on its screen outlives the Python
reference to it (a bare wrapper raised "'FitToScreen' object has no attribute 'dialog'",
e.g. opening View > Set Display Options to show frame local axes)."""
import gc
import sys

from PySide6 import QtWidgets


def test_filter_survives_garbage_collection():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from structstudio.app.fit import install
    errors = []
    old = sys.excepthook
    sys.excepthook = lambda *a: errors.append(a[1])
    try:
        d = QtWidgets.QDialog()
        install(d)  # the result is not kept, as the dialogs do
        gc.collect()
        d.show()
        d.move(d.x() + 5, d.y() + 5)
        for _ in range(5):
            app.processEvents()
        d.close()
        app.processEvents()
    finally:
        sys.excepthook = old
    assert not errors, errors
