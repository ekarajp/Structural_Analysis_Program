"""Analysis progress window and the analysis problems list.

* :class:`AnalysisProgressDialog` - percent, a green bar with a moving stripe
  (so it visibly works even on a long step), the current step, the elapsed
  time and a live log of what the analysis is doing.
* :class:`IssuesDialog` - every error / warning / note with the object and
  where it is (story, grid, coordinates) and a hint; double-click (or Locate)
  selects the object, takes the plan windows to its story and flashes it.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme

GREEN = "#2fb344"
GREEN_DARK = "#1f8a32"


class AnalysisProgressDialog(QtWidgets.QDialog):
    show_issues = QtCore.Signal()
    cancel_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Analysis")
        # modal: the model cannot be edited while it is being analysed (ETABS)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.resize(640, 420)
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.step = QtWidgets.QLabel("Starting...")
        self.step.setStyleSheet("font-weight: 600;")
        self.percent = QtWidgets.QLabel("0 %")
        self.percent.setStyleSheet("font-size: 18px; font-weight: 700;")
        top.addWidget(self.step, 1)
        top.addWidget(self.percent)
        lay.addLayout(top)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(22)
        lay.addWidget(self.bar)
        info = QtWidgets.QHBoxLayout()
        self.elapsed = QtWidgets.QLabel("0.0 s")
        self.summary = QtWidgets.QLabel("")
        info.addWidget(self.elapsed)
        info.addStretch(1)
        info.addWidget(self.summary)
        lay.addLayout(info)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.log.setFont(f)
        lay.addWidget(self.log, 1)
        btns = QtWidgets.QHBoxLayout()
        self.issues_btn = QtWidgets.QPushButton("Show problems")
        self.issues_btn.setVisible(False)
        self.issues_btn.clicked.connect(self.show_issues.emit)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop the analysis (at the next step); the model stays as it is")
        self.cancel_btn.clicked.connect(self._cancel)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.setEnabled(False)  # until the analysis has finished
        self.close_btn.clicked.connect(self.accept)
        btns.addWidget(self.issues_btn)
        btns.addStretch(1)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.close_btn)
        lay.addLayout(btns)
        self._t0 = time.time()
        self._phase = 0.0
        self._state = "running"
        self._last_msg = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._style()

    # ----------------------------------------------------------- look
    def _style(self):
        """Green chunk with a light stripe sliding along it (moving while running)."""
        if self._state == "failed":
            a, b = "#d64545", "#b02a2a"
        else:
            a, b = GREEN, GREEN_DARK
        ui = theme.active().ui()
        base = ui.get("panel", "#f0f0f0") if isinstance(ui, dict) else "#f0f0f0"
        if self._state == "running":
            p = self._phase
            s0, s1, s2 = max(p - 0.12, 0.0), p, min(p + 0.12, 1.0)
            grad = (f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {a}, stop:{s0:.3f} {a}, "
                    f"stop:{s1:.3f} #8fe39d, stop:{s2:.3f} {a}, stop:1 {b})")
        else:
            grad = a
        self.bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid #9aa4ae; border-radius: 4px; background: {base}; }}"
            f"QProgressBar::chunk {{ border-radius: 3px; background: {grad}; }}")

    def _now(self) -> float:
        """Seconds since the start; frozen at the moment the run ended."""
        end = getattr(self, "_t_end", None)
        return (end if end is not None else time.time()) - self._t0

    def _tick(self):
        self.elapsed.setText(f"{self._now():.1f} s")
        if self._state == "running":
            self._phase = (self._phase + 0.035) % 1.0
            self._style()

    # ----------------------------------------------------------- updates
    def set_progress(self, pct: float, message: str):
        self.bar.setValue(int(round(pct * 10)))
        self.percent.setText(f"{pct:.0f} %")
        if message and message != self._last_msg:
            self._last_msg = message
            self.step.setText(message)
            self.log.appendPlainText(f"[{self._now():7.2f} s] {message}")

    def add_log(self, text: str):
        self.log.appendPlainText(f"[{self._now():7.2f} s] {text}")

    def _cancel(self):
        if self._state == "running":
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")
            self.step.setText("Cancelling - stops at the next step...")
            self.cancel_requested.emit()

    def closeEvent(self, ev):
        if self._state == "running":  # the window's X while running = Cancel
            self._cancel()
            ev.ignore()
            return
        super().closeEvent(ev)

    def reject(self):  # Esc
        if self._state == "running":
            self._cancel()
            return
        super().reject()

    def finish(self, ok: bool, text: str, n_problems: int = 0, cancelled: bool = False):
        self._t_end = time.time()  # the clock stops: the time shown is the run time
        self._state = "done" if ok else "failed"
        self.cancel_btn.setVisible(False)
        self.close_btn.setEnabled(True)
        self.close_btn.setDefault(True)
        self.close_btn.setFocus()
        if ok:
            self.set_progress(100.0, text)
            self.percent.setText("Finished \u2713")
            self.add_log("Finished")
            self.setWindowTitle("Run Analysis - Finished")
        else:
            self.step.setText(text)
            self.add_log(text)
            self.percent.setText("Cancelled" if cancelled else "Failed")
            self.setWindowTitle("Run Analysis - " + ("Cancelled" if cancelled else "Failed"))
        self.step.setStyleSheet(f"font-weight: 600; color: {GREEN_DARK if ok else '#c62828'};")
        self.summary.setText(f"{n_problems} problem(s) to look at" if n_problems else ("no problems" if ok else ""))
        self.issues_btn.setVisible(n_problems > 0)
        self.close_btn.setText("Close")
        self._style()
        self._tick()
        self._timer.stop()
        self.elapsed.setStyleSheet("font-weight: 600;")


_SEV = {"error": ("Error", "#c62828"), "warning": ("Warning", "#b26a00"), "info": ("Note", "#5f6b7a")}


class IssuesDialog(QtWidgets.QDialog):
    """Analysis problems: what, the object, where, how to fix; Locate jumps to it."""

    def __init__(self, issues, locate: Callable[[str, str, bool], None], parent=None, title="Analysis problems",
                 model_path: Optional[str] = None, model_name: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._title, self._model_path, self._model_name = title, model_path, model_name
        self.setModal(False)
        self.resize(980, 460)
        self._locate = locate
        lay = QtWidgets.QVBoxLayout(self)
        n_err = sum(i.severity == "error" for i in issues)
        n_w = sum(i.severity == "warning" for i in issues)
        head = QtWidgets.QLabel(
            (f"<b style='color:#c62828'>{n_err} error(s)</b> - the analysis cannot run until they are fixed. "
             if n_err else "") + (f"<b style='color:#b26a00'>{n_w} warning(s)</b>. " if n_w else "")
            + "Double-click a row to show the object in the model.")
        head.setWordWrap(True)
        lay.addWidget(head)
        self.show_notes = QtWidgets.QCheckBox("Show notes")
        self.show_notes.toggled.connect(self._fill)
        lay.addWidget(self.show_notes)
        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["", "Problem", "Object", "Where", "What to do"])
        self.table.setRootIsDecorated(False)
        self.table.setWordWrap(True)
        self.table.setUniformRowHeights(False)
        self.table.itemDoubleClicked.connect(lambda it, c: self._go(it))
        self.table.currentItemChanged.connect(lambda it, _old: self._detail(it))
        lay.addWidget(self.table, 1)
        self.details = QtWidgets.QTextBrowser()
        self.details.setMaximumHeight(110)
        lay.addWidget(self.details)
        btns = QtWidgets.QHBoxLayout()
        b_loc = QtWidgets.QPushButton("Locate")
        b_loc.clicked.connect(lambda: self._go(self.table.currentItem()))
        b_copy = QtWidgets.QPushButton("Copy all")
        b_copy.clicked.connect(self._copy)
        self.save_btn = QtWidgets.QPushButton("Save as text file...")
        self.save_btn.setToolTip("Save the whole report (errors, warnings and notes) as a .txt file")
        self.save_btn.clicked.connect(self._save)
        b_close = QtWidgets.QPushButton("Close")
        b_close.clicked.connect(self.close)
        btns.addWidget(b_loc)
        btns.addWidget(b_copy)
        btns.addWidget(self.save_btn)
        btns.addStretch(1)
        btns.addWidget(b_close)
        lay.addLayout(btns)
        self.issues = list(issues)
        self._fill()
        hdr = self.table.header()
        hdr.resizeSection(0, 70)
        hdr.resizeSection(1, 430)
        hdr.resizeSection(2, 80)
        hdr.resizeSection(3, 230)

    def _fill(self, *_):
        self.table.clear()
        order = {"error": 0, "warning": 1, "info": 2}
        for i in sorted(self.issues, key=lambda i: order.get(i.severity, 3)):
            if i.severity == "info" and not self.show_notes.isChecked():
                continue
            label, color = _SEV.get(i.severity, (i.severity, "#000"))
            it = QtWidgets.QTreeWidgetItem([label, i.message, i.name or "", i.where, i.hint])
            it.setForeground(0, QtGui.QBrush(QtGui.QColor(color)))
            f = it.font(0)
            f.setBold(True)
            it.setFont(0, f)
            for c in range(5):
                it.setToolTip(c, i.text())
            it.setData(0, QtCore.Qt.UserRole, (i.attr, i.name))
            it.setData(1, QtCore.Qt.UserRole, i)
            self.table.addTopLevelItem(it)
        if self.table.topLevelItemCount():
            self.table.setCurrentItem(self.table.topLevelItem(0))

    def _detail(self, item):
        if item is None:
            self.details.clear()
            return
        i = item.data(1, QtCore.Qt.UserRole)
        label, color = _SEV.get(i.severity, (i.severity, "#000"))
        html = f"<b style='color:{color}'>{label}</b>: {i.message}"
        if i.name:
            html += f"<br><b>Object:</b> {i.name}"
        if i.where:
            html += f"<br><b>Where:</b> {i.where}"
        if i.hint:
            html += f"<br><b>What to do:</b> {i.hint}"
        self.details.setHtml(html)

    def _go(self, item):
        if item is None:
            return
        attr, name = item.data(0, QtCore.Qt.UserRole) or (None, None)
        if attr in ("points", "frames", "areas") and name:
            self._locate(attr, name, True)

    def _copy(self):
        QtWidgets.QApplication.clipboard().setText(self.report_text())

    def report_text(self) -> str:
        """The whole report, readable in any text editor (errors first, notes last)."""
        import datetime
        order = [("error", "ERRORS - the analysis cannot run until these are fixed"),
                 ("warning", "WARNINGS - check these (the results may be affected)"),
                 ("info", "NOTES")]
        lines = [f"Structural Studio - {self._title}",
                 f"Model:   {self._model_name or '-'}" + (f"   ({self._model_path})" if self._model_path else ""),
                 f"Written: {datetime.datetime.now():%Y-%m-%d %H:%M}",
                 "Summary: " + ", ".join(f"{sum(i.severity == k for i in self.issues)} {w}(s)"
                                         for k, w in (("error", "error"), ("warning", "warning"),
                                                      ("info", "note"))),
                 ""]
        for sev, head in order:
            group = [i for i in self.issues if i.severity == sev]
            if not group:
                continue
            lines += [head, "=" * len(head)]
            for k, i in enumerate(group, 1):
                lines.append(f"{k:3d}. {i.message}")
                if i.name:
                    lines.append(f"     Object:     {i.name}")
                if i.where:
                    lines.append(f"     Where:      {i.where}")
                if i.hint:
                    lines.append(f"     What to do: {i.hint}")
            lines.append("")
        return "\n".join(lines)

    def default_file_name(self) -> str:
        import os
        base = os.path.splitext(self._model_path)[0] if self._model_path else (self._model_name or "model")
        return f"{base}_analysis_problems.txt"

    def save_to(self, path: str) -> str:
        with open(path, "w", encoding="utf-8-sig", newline="\r\n") as fh:  # opens in Notepad
            fh.write(self.report_text() + "\n")
        return path

    def _save(self):
        path, _f = QtWidgets.QFileDialog.getSaveFileName(self, "Save the problems report", self.default_file_name(),
                                                         "Text file (*.txt);;All files (*)")
        if not path:
            return
        try:
            self.save_to(path)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Save report", f"Could not write {path}:\n{e}")
            return
        par = self.parent()
        if par is not None and hasattr(par, "statusBar"):
            par.statusBar().showMessage(f"Problems report saved: {path}", 10000)
