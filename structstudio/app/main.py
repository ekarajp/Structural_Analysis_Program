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


def _vtk_messages_to_file() -> None:
    """VTK warnings / errors go to a log file next to structstudio.log, never to the
    "vtkOutputWindow" pop-up (on Windows it opens on the first message and can fill up)."""
    try:
        from . import vtk_lite as vtk
        out = vtk.vtkFileOutputWindow()
        out.SetFileName(os.path.join(os.path.dirname(log_path()), "vtk-messages.log"))
        out.SetFlush(True)
        out.SetAppend(False)
        vtk.vtkOutputWindow.SetInstance(out)
    except Exception:
        pass


HANG_SECONDS = 15.0


def _start_hang_watchdog(app):
    """When the program stops answering (the event loop has not run for HANG_SECONDS),
    write where every thread is - the Python stacks - to ``hang-<time>.txt`` next to
    structstudio.log and to the error output (the launcher's app-stderr.log), once per
    hang.  Long analyses run in their own thread and do not count; dialogs keep the loop
    running."""
    import faulthandler
    import threading
    import time
    from PySide6 import QtCore

    beat = [time.monotonic()]
    timer = QtCore.QTimer(app)
    timer.timeout.connect(lambda: beat.__setitem__(0, time.monotonic()))
    timer.start(500)
    app._hang_timer = timer  # (kept alive)

    def dump(stale: float):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        head = f"[{stamp}] Structural Studio has not answered for {stale:.0f} s - thread stacks:\n"
        targets = []
        try:
            targets.append(open(os.path.join(os.path.dirname(log_path()), f"hang-{stamp}.txt"), "w",
                                encoding="utf-8"))
        except OSError:
            pass
        err = sys.__stderr__
        try:
            if err is not None and err.fileno() >= 0:
                targets.append(err)
        except (OSError, ValueError, AttributeError):
            pass
        for fh in targets:
            try:
                fh.write(head)
                fh.flush()
                faulthandler.dump_traceback(fh, all_threads=True)
                fh.flush()
            except Exception:
                pass
        for fh in targets[:1]:
            try:
                fh.close()
            except Exception:
                pass
        _log(f"not answering for {stale:.0f} s: stacks in hang-{stamp}.txt")

    stop = threading.Event()

    def watch():
        dumped = False
        while not stop.wait(1.0):
            stale = time.monotonic() - beat[0]
            if stale >= HANG_SECONDS and not dumped:
                dump(stale)
                dumped = True
            elif stale < 2.0:
                dumped = False

    threading.Thread(target=watch, name="hang-watchdog", daemon=True).start()
    return stop  # .set() ends the watchdog


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


def _warm_new_model(settings) -> None:
    """In the background, right after the window opens: the region's catalog steel
    sections (their properties take a moment to work out), so New Model opens at once."""
    import json
    import threading
    try:
        d = json.loads(settings.value("new_model/defaults", "") or "{}")
    except (TypeError, ValueError):
        d = {}

    def work():
        try:
            from .templates import warm_region_properties
            warm_region_properties(d.get("region") or "thailand", d.get("steel_db"))
        except Exception as exc:  # only a head start: New Model works them out itself otherwise
            _log(f"New Model warm-up skipped: {exc!r}")
    threading.Thread(target=work, name="new-model-warm-up", daemon=True).start()


def main(argv=None) -> int:
    _install_excepthook()
    _vtk_messages_to_file()
    from PySide6 import QtCore, QtWidgets

    from .document import Document
    from .mainwindow import MainWindow

    argv = list(sys.argv if argv is None else argv)
    from .mainwindow import use_english_numbers
    use_english_numbers()  # 0-9 digits and '.' decimals whatever the Windows region (e.g. Thai)
    # the VTK views are native windows; keep their neighbours (window headers, combos)
    # ordinary Qt widgets, so their drop-down lists open where they are
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_DontCreateNativeWidgetSiblings, True)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("Structural Studio")
    app.setStyle("Fusion")
    _log(f"start {' '.join(argv)}")
    _start_hang_watchdog(app)
    doc = Document()
    settings = QtCore.QSettings("StructuralStudio", "StructuralStudio")
    win = MainWindow(doc, settings)
    win.show()
    _warm_new_model(settings)
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
