"""Entry point: ``python -m structstudio.app [model.ssj.json | model.e2k]``.

Windows: double-click ``Start Structural Studio.cmd`` (installs the UI libraries on the
first run).  Errors are shown in a message box and written to
``%LOCALAPPDATA%\\StructuralStudio\\structstudio.log`` (``~/.structstudio/`` elsewhere).
"""
from __future__ import annotations

import datetime
import os
import sys
import traceback


def log_path() -> str:
    base = os.environ.get("LOCALAPPDATA")
    folder = os.path.join(base, "StructuralStudio") if base else os.path.join(os.path.expanduser("~"), ".structstudio")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "structstudio.log")


def _log(text: str) -> None:
    try:
        with open(log_path(), "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")
    except OSError:
        pass


def _install_excepthook():
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        _log(text)
        try:
            from PySide6 import QtWidgets
            if QtWidgets.QApplication.instance() is not None:
                QtWidgets.QMessageBox.critical(None, "Structural Studio - error",
                                               f"{exc_type.__name__}: {exc}\n\nDetails: {log_path()}")
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook


def main(argv=None) -> int:
    _install_excepthook()
    from PySide6 import QtCore, QtWidgets

    from .document import Document
    from .mainwindow import MainWindow

    argv = list(sys.argv if argv is None else argv)
    from .mainwindow import use_english_numbers
    use_english_numbers()  # 0-9 digits and '.' decimals whatever the Windows region (e.g. Thai)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("Structural Studio")
    app.setStyle("Fusion")
    _log(f"start {' '.join(argv)}")
    doc = Document()
    settings = QtCore.QSettings("StructuralStudio", "StructuralStudio")
    win = MainWindow(doc, settings)
    win.show()
    if len(argv) > 1:
        win._open(argv[1])
    elif "--no-wizard" not in argv:
        # the start page (New Model / Browse / Import / recent models), unless switched off
        from .start_page import StartDialog
        if StartDialog.show_at_startup(settings):
            QtCore.QTimer.singleShot(0, win.show_start_page)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
