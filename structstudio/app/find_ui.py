"""The Find box on the main toolbar (Ctrl+F): type a name / id, a label, a grid
intersection or coordinates, press Enter and the program goes there."""
from __future__ import annotations

from typing import Callable, List

from PySide6 import QtCore, QtGui, QtWidgets

from . import find

PLACEHOLDER = "Find: name, label, B/3, or x,y,z  (Ctrl+F)"
HELP = ("Type what you are looking for and press Enter:\n"
        "  B23, P104          - the object with that name (id)\n"
        "  C5   or  C5 Story3 - a label (add the story when the label repeats on every story)\n"
        "  B/3  or  B/3 Story2 - a grid intersection\n"
        "  12.5, 6, 9         - coordinates x, y, z (x, y alone = nearest in plan)\n"
        "One match goes straight there; several show a list (arrows + Enter).")


class FindBox(QtWidgets.QLineEdit):
    """``model()`` gives the current model, ``go(hit)`` shows a result."""

    def __init__(self, model: Callable, go: Callable, status: Callable = None, parent=None):
        super().__init__(parent)
        self._model, self._go, self._status = model, go, status or (lambda t: None)
        self.setPlaceholderText(PLACEHOLDER)
        self.setToolTip(HELP)
        self.setClearButtonEnabled(True)
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)
        self.addAction(QtGui.QIcon.fromTheme("edit-find"), QtWidgets.QLineEdit.LeadingPosition)
        self._words = QtCore.QStringListModel(self)
        comp = QtWidgets.QCompleter(self._words, self)
        comp.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        comp.setFilterMode(QtCore.Qt.MatchStartsWith)
        comp.setMaxVisibleItems(12)
        self.setCompleter(comp)
        self.results = QtWidgets.QListWidget()
        self.results.setWindowFlags(QtCore.Qt.Popup)
        self.results.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.results.setMinimumWidth(560)
        self.results.itemActivated.connect(self._chosen)
        self.results.itemClicked.connect(self._chosen)
        self.results.installEventFilter(self)
        self.hits: List[find.Hit] = []
        self.returnPressed.connect(self.search)

    def refresh_words(self):
        try:
            self._words.setStringList(find.completions(self._model()))
        except Exception:  # pragma: no cover - a half-built model while loading
            pass

    def focusInEvent(self, e):
        self.refresh_words()
        super().focusInEvent(e)

    def search(self):
        text = self.text().strip()
        if not text:
            return
        if self.completer().popup().isVisible():
            self.completer().popup().hide()
        hits = find.find_objects(self._model(), text)
        self.hits = hits
        if not hits:
            self._status(f"Find: nothing matches '{text}' - try the name, label, B/3 or x,y,z")
            self._flash_bad()
            return
        best = hits[0]
        exact = [h for h in hits if h.score == best.score]
        joint_here = best.query == "location" and best.attr == "points" and \
            sum(h.attr == "points" and h.score == best.score for h in hits) == 1
        if len(hits) == 1 or joint_here or (best.attr != "xyz" and best.score <= 1.0 and len(exact) == 1):
            self._go(best)
            return
        self._show_list(hits)

    def _show_list(self, hits):
        self.results.clear()
        for h in hits:
            it = QtWidgets.QListWidgetItem(h.text)
            it.setData(QtCore.Qt.UserRole, h)
            it.setToolTip(h.text)
            self.results.addItem(it)
        self.results.setCurrentRow(0)
        rows = min(len(hits), 12)
        self.results.resize(max(self.width(), 560), rows * (self.results.sizeHintForRow(0) + 2) + 8)
        self.results.move(self.mapToGlobal(QtCore.QPoint(0, self.height())))
        self.results.show()
        self.results.setFocus()
        self._status(f"Find: {len(hits)} matches - pick one (arrows + Enter)")

    def _chosen(self, item):
        self.results.hide()
        if item is not None:
            self._go(item.data(QtCore.Qt.UserRole))

    def eventFilter(self, obj, e):
        if obj is self.results and e.type() == QtCore.QEvent.KeyPress and e.key() == QtCore.Qt.Key_Escape:
            self.results.hide()
            self.setFocus()
            return True
        return super().eventFilter(obj, e)

    def _flash_bad(self):
        self.setStyleSheet("QLineEdit { border: 2px solid #d03030; }")
        QtCore.QTimer.singleShot(900, lambda: self.setStyleSheet(""))

    def focus_for_typing(self):
        self.setFocus(QtCore.Qt.ShortcutFocusReason)
        self.selectAll()
