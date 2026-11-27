"""Test session setup.

The desktop tests set QT_QPA_PLATFORM=offscreen when nothing is set (Linux CI: no
display, xvfb for the VTK windows).  On Windows the offscreen platform gives VTK no
OpenGL context: creating a 3-D view there ends in an access violation.  A Windows
session always has a desktop, so the native platform is used (the test windows open
and close by themselves)."""
import os
import sys

if sys.platform == "win32" and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "windows"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _delete_closed_windows():
    """A window test closes its window with ``deleteLater()``; ``processEvents()`` does
    not run those deletions (Qt keeps them for the event loop that posted them), so the
    closed windows of every earlier test - thousands of widgets - stayed alive and were
    all deleted inside the first ``QTest.qWait`` of a later test, using up its wait
    (the wheel-zoom label test: 4 s of deletions in a 150 ms wait).  Each test now
    deletes what it closed."""
    yield
    QtCore = sys.modules.get("PySide6.QtCore")
    if QtCore is None or QtCore.QCoreApplication.instance() is None:
        return
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
