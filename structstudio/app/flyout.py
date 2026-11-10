"""Photoshop / Affinity style fly-out tool button.

One button on the palette stands for a group of tools (Draw Beams: beam, quick
beam, secondary beams).  A click on the face starts the tool shown on it;
press-and-hold (or a right-click) opens a row with the group's other tools
beside the button.  The tool picked there takes the face, so the next click
reuses it.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets


class FlyoutRow(QtWidgets.QFrame):
    """The pop-up row of tool buttons (``Qt.Popup``: closes on a click elsewhere)."""

    picked = QtCore.Signal(object)  # the QAction chosen

    def __init__(self, actions: List[QtGui.QAction], icon_size: QtCore.QSize, parent=None):
        super().__init__(parent, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.buttons: List[QtWidgets.QToolButton] = []
        for a in actions:
            b = QtWidgets.QToolButton(self)
            b.setDefaultAction(a)
            b.setIconSize(icon_size)
            b.setAutoRaise(True)
            b.setToolTip(a.toolTip() or a.text())
            b.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
            b.setMouseTracking(True)
            # a plain click on a tool of the row picks it too (not only press-drag-release):
            # the face of the palette button then shows it
            b.clicked.connect(lambda _=False, a=a: self._clicked(a))
            lay.addWidget(b)
            self.buttons.append(b)
        self.setMouseTracking(True)

    def _clicked(self, action: QtGui.QAction):
        self.picked.emit(action)
        self.close()

    def button_at(self, global_pos: QtCore.QPoint) -> Optional[QtWidgets.QToolButton]:
        for b in self.buttons:
            if b.rect().contains(b.mapFromGlobal(global_pos)):
                return b
        return None

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        # press on the face, drag onto a tool, release: one gesture
        b = self.button_at(ev.globalPosition().toPoint())
        if b is not None:
            b.defaultAction().trigger()
            self.picked.emit(b.defaultAction())
            self.close()
            return
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        b = self.button_at(ev.globalPosition().toPoint())
        for other in self.buttons:
            other.setDown(other is b)
        super().mouseMoveEvent(ev)


class FlyoutButton(QtWidgets.QToolButton):
    """Palette button with a sideways fly-out of its tools.

    ``actions``: the group's tools; the face shows ``actions[0]`` until a tool
    is picked.  ``side``: "right" (vertical palettes) or "down" (horizontal)."""

    tool_picked = QtCore.Signal(object)

    def __init__(self, title: str, actions: List[QtGui.QAction], parent=None, side: str = "right"):
        super().__init__(parent)
        self.title = title
        self.actions_ = list(actions)
        self.side = side
        self.row: Optional[FlyoutRow] = None
        self._pressed_at: Optional[QtCore.QPoint] = None
        self.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.setDefaultAction(self.actions_[0])
        self._update_tip()
        self.setMouseTracking(True)
        # Photoshop: the face shows the tool used last, however it was chosen (this fly-out,
        # the Draw menu, a shortcut, the command line)
        for a in self.actions_:
            a.triggered.connect(lambda _=False, a=a: self._follow(a))

    def _follow(self, action: QtGui.QAction):
        try:
            if action is not self.defaultAction() and action in self.actions_:
                self.setDefaultAction(action)
                self._update_tip()
        except RuntimeError:  # the palette was rebuilt (compact <-> full): this button is gone
            pass

    # ---- helpers
    def _update_tip(self):
        a = self.defaultAction()
        self.setToolTip(f"{self.title}: {a.text()}  (press: more tools)")

    def menu_actions(self) -> List[QtGui.QAction]:
        """The group's tools (compatibility with the former drop-down ``menu()``)."""
        return list(self.actions_)

    def pick(self, action: QtGui.QAction):
        """A tool of the group was chosen: the face remembers it."""
        if action in self.actions_:
            self.setDefaultAction(action)
            self._update_tip()
            self.tool_picked.emit(action)

    HOLD_MS = 350  # press and hold this long: the fly-out opens (Photoshop / Affinity)

    def open_flyout(self):
        """Show the row of the group's other tools beside the button (the tool on the
        face is not repeated in the row)."""
        if self.row is not None:
            self.row.close()
            self.row.deleteLater()
        others = [a for a in self.actions_ if a is not self.defaultAction()]
        self.row = FlyoutRow(others, self.iconSize(), self)
        self.row.picked.connect(self.pick)
        self.row.adjustSize()
        if self.side == "down":
            pos = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        else:
            pos = self.mapToGlobal(QtCore.QPoint(self.width(), 0))
        scr = QtWidgets.QApplication.screenAt(pos) or QtWidgets.QApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            pos.setX(min(pos.x(), g.right() - self.row.width()))
            pos.setY(min(pos.y(), g.bottom() - self.row.height()))
        self.row.move(pos)
        self.row.show()

    def flyout_open(self) -> bool:
        return self.row is not None and self.row.isVisible()

    # ---- mouse: a click uses the tool on the face; press-and-hold / right-click opens the row
    def _hold_elapsed(self):
        if self._pressed_at is not None:
            self.setDown(False)
            self.open_flyout()
            self._pressed_at = None  # the release goes to the row (a Qt popup)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton:
            self._pressed_at = ev.globalPosition().toPoint()
            self.setDown(True)
            QtCore.QTimer.singleShot(self.HOLD_MS, self._hold_elapsed)
            ev.accept()
            return
        if ev.button() == QtCore.Qt.RightButton:
            self.open_flyout()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton and self._pressed_at is not None:
            self._pressed_at = None
            self.setDown(False)
            if self.rect().contains(ev.position().toPoint()):
                self.defaultAction().trigger()  # plain click: the tool shown on the face
                self.tool_picked.emit(self.defaultAction())
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def contextMenuEvent(self, ev):
        ev.accept()

    def paintEvent(self, ev: QtGui.QPaintEvent):
        super().paintEvent(ev)
        # small corner triangle: "more tools inside" (CAD convention)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = self.rect()
        tri = QtGui.QPolygonF([QtCore.QPointF(r.right() - 1, r.bottom() - 6), QtCore.QPointF(r.right() - 1, r.bottom() - 1),
                               QtCore.QPointF(r.right() - 6, r.bottom() - 1)])
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(self.palette().color(QtGui.QPalette.WindowText))
        p.drawPolygon(tri)
        p.end()


class PaletteGrid(QtWidgets.QWidget):
    """The full palette: every tool its own button, flowing into a second (third ...)
    column when the window is not tall enough - nothing is folded away behind a
    toolbar extension arrow.  ``items``: QActions, ``None`` = a group gap."""

    GAP = 9
    COLGAP = 7  # between columns: a thin vertical line in it

    def __init__(self, items, icon_size: QtCore.QSize, parent=None):
        super().__init__(parent)
        self.items = list(items)
        self.icon_size = icon_size
        self.buttons: List[Optional[QtWidgets.QToolButton]] = []
        for a in self.items:
            if a is None:
                self.buttons.append(None)
                continue
            b = QtWidgets.QToolButton(self)
            b.setDefaultAction(a)
            b.setIconSize(icon_size)
            b.setAutoRaise(True)
            b.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
            self.buttons.append(b)
        bs = [b for b in self.buttons if b is not None]
        hint = bs[0].sizeHint() if bs else QtCore.QSize(30, 30)
        self.bw, self.bh = hint.width(), hint.height()
        self._avail = 10 ** 6
        self._cols = 1
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)

    def minimumSizeHint(self):
        if not hasattr(self, "_size"):
            self._relayout()
        return QtCore.QSize(self._size.width(), self.bh * 3)

    def set_available_height(self, h: int):
        """Height the palette may use (the window's left dock area)."""
        h = max(int(h), self.bh * 3)
        if h != self._avail:
            self._avail = h
            self._relayout()

    def _positions(self, avail: int):
        x = y = 0
        cols = 1
        out = []
        for b in self.buttons:
            if b is None:
                if y > 0:
                    y += self.GAP
                out.append(None)
                continue
            if y + self.bh > avail and y > 0:
                x += self.bw + self.COLGAP
                y = 0
                cols += 1
            out.append((x, y))
            y += self.bh
        return out, cols

    def _relayout(self):
        pos, cols = self._positions(self._avail)
        # a thin line across the column in each gap between groups (select | draw | edit | ...)
        self._lines = []
        for k, (b, p) in enumerate(zip(self.buttons, pos)):
            if b is None:
                nxt = next((q for bb, q in zip(self.buttons[k + 1:], pos[k + 1:]) if bb is not None), None)
                prv = next((q for bb, q in zip(self.buttons[:k][::-1], pos[:k][::-1]) if bb is not None), None)
                if nxt is not None and prv is not None and nxt[0] == prv[0]:
                    self._lines.append((nxt[0], nxt[1] - self.GAP / 2.0))
        self.update()
        maxy = 0
        for b, p in zip(self.buttons, pos):
            if b is not None and p is not None:
                b.setGeometry(p[0], p[1], self.bw, self.bh)
                maxy = max(maxy, p[1] + self.bh)
        self._cols = cols
        self._size = QtCore.QSize(cols * self.bw + (cols - 1) * self.COLGAP, maxy)
        self._vlines = [(k * (self.bw + self.COLGAP) - self.COLGAP / 2.0, maxy) for k in range(1, cols)]
        # The width follows the columns; the height is only *preferred*.  A fixed height would
        # become the window's minimum height, so the window could not get shorter and the palette
        # would never reflow into more columns (a feedback loop).  The minimum is three rows: the
        # window decides the height and the palette reflows into it (MainWindow._fit_palette).
        self.setFixedWidth(self._size.width())
        self.setMinimumHeight(self.bh * 3)
        self.setMaximumHeight(max(maxy, self.bh * 3))
        self.resize(self._size)
        self.updateGeometry()

    def columns(self) -> int:
        return self._cols

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QtGui.QPainter(self)
        c = self.palette().color(QtGui.QPalette.Mid)
        p.setPen(QtGui.QPen(c, 1.0))
        for x, y in getattr(self, "_lines", []):
            p.drawLine(QtCore.QPointF(x + 4, y), QtCore.QPointF(x + self.bw - 4, y))
        for x, h in getattr(self, "_vlines", []):  # column 1 | column 2
            p.drawLine(QtCore.QPointF(x, 2), QtCore.QPointF(x, h - 2))
        p.end()

    def sizeHint(self):
        if not hasattr(self, "_size"):
            self._relayout()
        return self._size
