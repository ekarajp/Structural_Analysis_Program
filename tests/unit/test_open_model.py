"""File > Open and ``python -m structstudio.app model.ssj.json`` go through MainWindow._open."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "office.ssj.json")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_open_native_json(app, tmp_path):
    from structstudio.app.document import Document
    from structstudio.app.mainwindow import MainWindow
    win = MainWindow(Document(), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    assert win._open(EXAMPLE) == EXAMPLE
    assert win.doc.path == EXAMPLE
    assert len(win.doc.model.frames) == 16
    win.close()
