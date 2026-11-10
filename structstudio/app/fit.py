"""Dialogs that fit the screen they are on.

A dialog's minimum size is the sum of the minimums of everything in it - every
tab, every page of a stacked widget, even the ones not shown.  On a small or
scaled screen (1366x768, 1920x1080 at 150 %) that can be taller or wider than
the screen, and OK / Cancel end up off screen.

* :func:`scroll_page` puts a page's content in a scroll area: the page can get
  smaller than its content (it scrolls) while the buttons, kept outside it,
  stay visible.
* :func:`fit_to_screen` makes a dialog no larger than the available area of the
  screen it is on (taskbar excluded) and moves it fully onto that screen.
"""
from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

MARGIN = 16  # kept free around a dialog fitted to the screen (window frame, shadow)


class _PageScroll(QtWidgets.QScrollArea):
    """Asks for the whole content (Qt caps a scroll area's wish at a fraction of the screen,
    which would make a dialog scroll with room to spare); the minimum stays small."""

    def sizeHint(self):
        w = self.widget()
        if w is None:
            return super().sizeHint()
        h = w.sizeHint().expandedTo(w.minimumSizeHint())
        m = self.contentsMargins()
        bar = self.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent, None, self)
        # room for the vertical scroll bar, so a page that only scrolls up/down never also scrolls sideways
        return QtCore.QSize(h.width() + m.left() + m.right() + bar, h.height() + m.top() + m.bottom())

    def minimumSizeHint(self):
        return QtCore.QSize(120, 80)


def scroll_page(content: QtWidgets.QWidget, horizontal: bool = True) -> QtWidgets.QScrollArea:
    """``content`` in a frameless scroll area that takes the size it is given."""
    area = _PageScroll()
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded if horizontal else QtCore.Qt.ScrollBarAlwaysOff)
    area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    area.setWidget(content)
    return area


def screen_of(widget: QtWidgets.QWidget) -> Optional[QtGui.QScreen]:
    parent = widget.parentWidget()
    for w in (widget, parent.window() if parent is not None else None):
        if w is None:
            continue
        s = w.screen() if hasattr(w, "screen") else None
        if s is not None:
            return s
    return QtGui.QGuiApplication.primaryScreen()


def available_rect(widget: QtWidgets.QWidget) -> QtCore.QRect:
    s = screen_of(widget)
    r = s.availableGeometry() if s is not None else QtCore.QRect(0, 0, 1280, 720)
    return r.adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN)


def fit_to_screen(dialog: QtWidgets.QWidget, grow: bool = False) -> None:
    """No larger than the screen's free area, and fully on it.  ``grow`` (first show): as large
    as its content wants, within that area (Qt opens a window at most 2/3 of the screen,
    which would make it scroll with room to spare)."""
    avail = available_rect(dialog)
    # the native window frame (title bar, borders) around the client area: known once the
    # window has been shown (the deferred second fit); the frame, not only the client
    # area, has to fit the screen
    fg, cg = dialog.frameGeometry(), dialog.geometry()
    extra_w = max(fg.width() - cg.width(), 0)
    extra_h = max(fg.height() - cg.height(), 0)
    w, h = dialog.width(), dialog.height()
    if grow:
        hint = dialog.sizeHint()
        w, h = max(w, hint.width()), max(h, hint.height())
    w = min(w, avail.width() - extra_w)
    h = min(h, avail.height() - extra_h)
    if (w, h) != (dialog.width(), dialog.height()):
        dialog.resize(w, h)
    # the frame after the resize (frameGeometry may not be updated before the next event)
    fw = max(dialog.width(), w) + extra_w
    fh = max(dialog.height(), h) + extra_h
    gx, gy = fg.x(), fg.y()
    x = min(max(gx, avail.left()), avail.right() - fw + 1)
    y = min(max(gy, avail.top()), avail.bottom() - fh + 1)
    if (x, y) != (gx, gy):
        dialog.move(max(x, avail.left()), max(y, avail.top()))


class FitToScreen(QtCore.QObject):
    """Event filter: fit ``dialog`` to its screen when it is shown or moved to another screen."""

    def __init__(self, dialog: QtWidgets.QWidget):
        super().__init__(dialog)
        self.dialog = dialog
        dialog.installEventFilter(self)
        self._screen = None
        self._shown = False

    def eventFilter(self, obj, ev):
        if obj is self.dialog:
            t = ev.type()
            if t == QtCore.QEvent.Show:
                self._fit()  # before it is drawn ...
                QtCore.QTimer.singleShot(0, self._fit)  # ... and again once the frame is known
            elif t == QtCore.QEvent.Move:
                s = screen_of(self.dialog)
                if s is not self._screen and self._screen is not None:
                    QtCore.QTimer.singleShot(0, self._fit)
        return False

    def _fit(self):
        try:
            self._screen = screen_of(self.dialog)
            fit_to_screen(self.dialog, grow=not self._shown)
            self._shown = True
        except RuntimeError:  # the dialog was deleted meanwhile
            pass


def install(dialog: QtWidgets.QWidget) -> FitToScreen:
    return FitToScreen(dialog)
