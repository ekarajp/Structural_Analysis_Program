"""New Model wizard: 1 settings, 2 grid lines and stories, 3 starting structure.

Step 1 takes the settings from the built-in regions (materials, sections,
rebar sizes), from the user's saved defaults or from a saved model.  Step 2
sets the grid and the stories - uniform (counts and spacings) or custom
(tables) - with a plan and an elevation drawn as they are typed.  Step 3
chooses what the model starts with; the cards are drawn from the real model
the wizard would build on that grid.
"""
from __future__ import annotations

import time

import json
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..model import Model
from ..units import PRESETS, UnitSystem
from . import theme
from .sketch import model_sketch, paint_sketch
from .templates import (REGIONS, REGION_ORDER, STRUCTURES, SUPPORTS, _letters, build_model, region_properties,
                        structure_counts)

KEY_DEFAULTS = "new_model/defaults"
STEP_TITLES = (("Model settings", "Units, materials and section lists for the new model. All can be changed later."),
               ("Grid lines and stories", "Start simple with uniform spacing, or switch to tables for any spacing."),
               ("Starting structure", "Members are placed on your grid; all can be edited afterwards. "
                                      "Double-click a picture to create the model."))
PREVIEW_STORIES = 3  # the structure cards draw the lowest stories only (a tall model is unreadable that small)
PREVIEW_LINES = 12  # ... and the first grid lines each way: a 30 x 30 grid is unreadable at card size and slow to build
PREVIEW_BUDGET = 0.15  # s of card drawing done at once; the rest is drawn in later event-loop turns


def _muted(w: QtWidgets.QWidget) -> QtWidgets.QWidget:
    w.setForegroundRole(QtGui.QPalette.PlaceholderText)
    return w


def _bold(w: QtWidgets.QWidget, scale: float = 1.0) -> QtWidgets.QWidget:
    f = w.font()
    f.setBold(True)
    if scale != 1.0:
        f.setPointSizeF(f.pointSizeF() * scale)
    w.setFont(f)
    return w


def _group(title: str) -> Tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout]:
    g = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(g)
    return g, lay


def _num(v: float) -> str:
    """Engineering values without exponents: 23,536 or 0.2354."""
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.4g}"


def _utext(u: UnitSystem) -> str:
    return f"{u.force} {u.length} {u.temperature}"


def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v - round(v, 3)) < 1e-12 else f"{v:g}"


# =============================================================================
# step header
# =============================================================================
class StepBar(QtWidgets.QWidget):
    """1 Settings - 2 Grid and stories - 3 Starting structure."""

    LABELS = ("Settings", "Grid and stories", "Starting structure")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.step = 0
        self.setMinimumHeight(26)
        self.setMinimumWidth(3 * 30 + 8)  # narrow: numbers only, the current step keeps its label

    def _width(self, labels) -> int:
        fm = self.fontMetrics()
        return sum((22 + fm.horizontalAdvance(lab) + 22 if lab else 30) + 4 for lab in labels)

    def _labels(self):
        """Every step's label if they fit, else the current one's only, else none."""
        full = list(self.LABELS)
        if self._width(full) <= self.width():
            return full
        cur = [lab if i == self.step else "" for i, lab in enumerate(full)]
        return cur if self._width(cur) <= self.width() else ["" for _ in full]

    def sizeHint(self):
        return QtCore.QSize(self._width(self.LABELS), 26)

    def set_step(self, i: int):
        self.step = i
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pal = self.palette()
        fm = self.fontMetrics()
        x = self.width()
        items = []
        for i, lab in reversed(list(enumerate(self._labels()))):
            w = 22 + fm.horizontalAdvance(lab) + 22 if lab else 30
            x -= w + 4
            items.append((i, lab, QtCore.QRectF(x, (self.height() - 24) / 2, w, 24)))
        for i, lab, r in items:
            now, done = i == self.step, i < self.step
            if now:
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(pal.color(QtGui.QPalette.Highlight))
                p.drawRoundedRect(r, 3, 3)
            c = QtCore.QRectF(r.left() + 6, r.center().y() - 9, 18, 18)
            p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.HighlightedText) if now else
                                (pal.color(QtGui.QPalette.Highlight) if done else pal.color(QtGui.QPalette.Mid)), 1))
            p.setBrush(pal.color(QtGui.QPalette.HighlightedText) if now else
                       (pal.color(QtGui.QPalette.Highlight) if done else pal.color(QtGui.QPalette.Button)))
            p.drawEllipse(c)
            p.setPen(pal.color(QtGui.QPalette.Highlight) if now else
                     (pal.color(QtGui.QPalette.HighlightedText) if done else pal.color(QtGui.QPalette.PlaceholderText)))
            p.drawText(c, QtCore.Qt.AlignCenter, "✓" if done else str(i + 1))
            p.setPen(pal.color(QtGui.QPalette.HighlightedText) if now else
                     pal.color(QtGui.QPalette.WindowText if done else QtGui.QPalette.PlaceholderText))
            p.drawText(QtCore.QRectF(c.right() + 6, r.top(), r.right() - c.right() - 6, r.height()),
                       QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, lab)
        p.end()


class CardGrid(QtWidgets.QWidget):
    """The structure cards in 4 columns, or 3 / 2 when the window is narrow (a small or
    scaled screen) - the cards reflow instead of widening the window."""

    MAX_COLS, MIN_COLS, SPACING = 4, 2, 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(self.SPACING)
        self.items: List[QtWidgets.QWidget] = []
        self.cols = self.MAX_COLS

    def add(self, w: QtWidgets.QWidget):
        self.items.append(w)
        self._place()

    def _card_min(self) -> int:
        return max((w.minimumSizeHint().expandedTo(w.minimumSize()).width() for w in self.items), default=180)

    def _cols_for(self, width: int) -> int:
        c, sp = self._card_min(), self.SPACING
        return max(self.MIN_COLS, min(self.MAX_COLS, (width + sp) // (c + sp)))

    def _place(self):
        for w in self.items:
            self.grid.removeWidget(w)
        for i, w in enumerate(self.items):
            self.grid.addWidget(w, i // self.cols, i % self.cols)
        for k in range(self.MAX_COLS):
            self.grid.setColumnStretch(k, 1 if k < self.cols else 0)

    def minimumSizeHint(self):
        c = self._card_min()
        rows = -(-len(self.items) // self.MIN_COLS)
        h = max((w.minimumSizeHint().height() for w in self.items), default=100)
        return QtCore.QSize(self.MIN_COLS * c + (self.MIN_COLS - 1) * self.SPACING, rows * h + (rows - 1) * self.SPACING)

    def resizeEvent(self, ev):
        cols = self._cols_for(ev.size().width())
        if cols != self.cols:
            self.cols = cols
            self._place()
        super().resizeEvent(ev)


# =============================================================================
# grid previews
# =============================================================================
class GridPreview(QtWidgets.QWidget):
    """Plan (``kind`` "plan") or elevation ("elev") of the grid being set."""

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.xl: List[str] = []
        self.yl: List[str] = []
        self.levels: List[Tuple[str, float]] = []  # bottom up (name, elevation)
        self.current = None  # ("x" | "y" | "story", index) highlighted
        self.setMinimumSize(220, 200)

    def set_data(self, xs, ys, xl, yl, levels, current=None):
        self.xs, self.ys, self.xl, self.yl, self.levels, self.current = list(xs), list(ys), list(xl), list(yl), \
            list(levels), current
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        colors = theme.active().colors()
        bg = QtGui.QColor.fromRgbF(*colors["background_top"])
        p.fillRect(self.rect(), bg)
        grid = QtGui.QColor.fromRgbF(*colors["grid"])
        text = QtGui.QColor.fromRgbF(*colors["grid_text"])
        hi = QtGui.QColor.fromRgbF(*colors["selection"])
        dimc = self.palette().color(QtGui.QPalette.Highlight)
        dash = QtGui.QPen(grid, 1.0, QtCore.Qt.DashLine)
        if self.kind == "plan":
            self._plan(p, dash, text, hi, dimc, bg)
        else:
            self._elev(p, dash, text, hi, colors)
        p.end()

    def _plan(self, p, dash, text, hi, dimc, bg):
        xs, ys = self.xs, self.ys
        if len(xs) < 1 or len(ys) < 1:
            return
        W, H = self.width(), self.height()
        sx = max(xs[-1] - xs[0], 1e-9)
        sy = max(ys[-1] - ys[0], 1e-9)
        k = min((W - 110) / sx, (H - 110) / sy)
        ox = 50 + (W - 110 - sx * k) / 2
        oy = 50 + (H - 110 - sy * k) / 2

        # Y up on the screen (plan as in the model windows)
        def X(x):
            return ox + (x - xs[0]) * k

        def Y(y):
            return oy + sy * k - (y - ys[0]) * k
        x0, x1, y0, y1 = X(xs[0]), X(xs[-1]), Y(ys[0]), Y(ys[-1])
        r = 9.0
        for i, x in enumerate(xs):
            cur = self.current == ("x", i)
            p.setPen(QtGui.QPen(hi, 1.8) if cur else dash)
            p.drawLine(QtCore.QPointF(X(x), y1 - 12), QtCore.QPointF(X(x), y0 + 12))
            self._bubble(p, QtCore.QPointF(X(x), y1 - 12 - r), r, self.xl[i] if i < len(self.xl) else "", text, bg)
        for j, y in enumerate(ys):
            cur = self.current == ("y", j)
            p.setPen(QtGui.QPen(hi, 1.8) if cur else dash)
            p.drawLine(QtCore.QPointF(x0 - 12, Y(y)), QtCore.QPointF(x1 + 12, Y(y)))
            self._bubble(p, QtCore.QPointF(x1 + 12 + r, Y(y)), r, self.yl[j] if j < len(self.yl) else "", text, bg)
        # dimensions
        p.setPen(QtGui.QPen(dimc, 0.9))
        f = p.font()
        f.setPointSizeF(max(6.5, f.pointSizeF() * 0.85))
        p.setFont(f)
        dy = y0 + 26
        if len(xs) > 1:
            p.drawLine(QtCore.QPointF(x0, dy), QtCore.QPointF(x1, dy))
            for i, x in enumerate(xs):
                p.drawLine(QtCore.QPointF(X(x), dy - 4), QtCore.QPointF(X(x), dy + 4))
                if i:
                    mid = (X(xs[i - 1]) + X(x)) / 2
                    p.drawText(QtCore.QRectF(mid - 40, dy - 16, 80, 14), QtCore.Qt.AlignCenter, _fmt(x - xs[i - 1]))
        dx = x0 - 26
        if len(ys) > 1:
            p.drawLine(QtCore.QPointF(dx, y0), QtCore.QPointF(dx, y1))
            for j, y in enumerate(ys):
                p.drawLine(QtCore.QPointF(dx - 4, Y(y)), QtCore.QPointF(dx + 4, Y(y)))
                if j:
                    mid = (Y(ys[j - 1]) + Y(y)) / 2
                    p.save()
                    p.translate(dx - 4, mid)
                    p.rotate(-90)
                    p.drawText(QtCore.QRectF(-40, -14, 80, 14), QtCore.Qt.AlignCenter, _fmt(y - ys[j - 1]))
                    p.restore()

    @staticmethod
    def _bubble(p, c, r, label, text, bg):
        p.setPen(QtGui.QPen(text, 0.9))
        p.setBrush(bg)
        p.drawEllipse(c, r, r)
        p.drawText(QtCore.QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), QtCore.Qt.AlignCenter, label)

    def _elev(self, p, dash, text, hi, colors):
        lv, xs = self.levels, self.xs
        if not lv:
            return
        W, H = self.width(), self.height()
        zt = max(lv[-1][1] - lv[0][1], 1e-9)
        k = (H - 40) / zt
        left, right = 24, W - 74

        def Z(z):
            return H - 20 - (z - lv[0][1]) * k
        if xs:
            sx = max(xs[-1] - xs[0], 1e-9)
            p.setPen(dash)
            for x in xs:
                u = left + 10 + (x - xs[0]) / sx * (right - left - 20)
                p.drawLine(QtCore.QPointF(u, Z(lv[-1][1]) - 8), QtCore.QPointF(u, Z(lv[0][1]) + 8))
        plane = QtGui.QColor.fromRgbF(*colors["story_plane"])
        f = p.font()
        f.setPointSizeF(max(6.5, f.pointSizeF() * 0.85))
        p.setFont(f)
        step = max(1, int(len(lv) / max(1, (H - 40) / 14)))
        for i, (name, z) in enumerate(lv):
            cur = self.current == ("story", i)
            p.setPen(QtGui.QPen(hi, 1.8) if cur else QtGui.QPen(plane, 1.0))
            p.drawLine(QtCore.QPointF(left, Z(z)), QtCore.QPointF(right, Z(z)))
            if cur or i % step == 0 or i == len(lv) - 1:
                p.setPen(hi if cur else text)
                p.drawText(QtCore.QRectF(right + 4, Z(z) - 8, W - right - 4, 16), QtCore.Qt.AlignVCenter, name)


# =============================================================================
# structure cards
# =============================================================================
class StructureCard(QtWidgets.QAbstractButton):
    """One starting structure; click = choose it, double-click = choose it and create the model."""
    activated = QtCore.Signal(str)

    def __init__(self, key: str, title: str, desc: str, parent=None):
        super().__init__(parent)
        self.key, self.title, self.desc = key, title, desc
        self.sketch: Optional[Dict[str, list]] = None
        self.empty_text = ""
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setToolTip(desc)
        self.setMinimumSize(180, 200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAccessibleName(title)

    def set_sketch(self, sk, empty_text=""):
        self.sketch, self.empty_text = sk, empty_text
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pal = self.palette()
        r = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)
        on = self.isChecked()
        hl = pal.color(QtGui.QPalette.Highlight)
        p.setPen(QtGui.QPen(hl if on or self.underMouse() else pal.color(QtGui.QPalette.Mid), 2.0 if on else 1.0))
        base = pal.color(QtGui.QPalette.Base)
        if on:
            base = QtGui.QColor(hl)
            base.setAlphaF(0.10)
            base = QtGui.QColor(
                *[int(a * 0.1 + b * 0.9) for a, b in zip(hl.getRgb()[:3], pal.color(QtGui.QPalette.Base).getRgb()[:3])])
        p.setBrush(base)
        p.drawRoundedRect(r, 3, 3)
        fm = self.fontMetrics()
        text_h = fm.height() * 3 + 10
        pic = QtCore.QRectF(r.left() + 1, r.top() + 1, r.width() - 2, r.height() - text_h - 2)
        colors = theme.active().colors()
        p.save()
        p.setClipRect(pic)
        paint_sketch(p, pic, self.sketch, colors, pad=10)
        if self.empty_text:
            p.setPen(pal.color(QtGui.QPalette.PlaceholderText))
            p.drawText(pic, QtCore.Qt.AlignCenter, self.empty_text)
        p.restore()
        p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Midlight), 1.0))
        p.drawLine(QtCore.QPointF(r.left() + 1, pic.bottom()), QtCore.QPointF(r.right() - 1, pic.bottom()))
        f = QtGui.QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        p.setPen(pal.color(QtGui.QPalette.Text))
        tr = QtCore.QRectF(r.left() + 8, pic.bottom() + 4, r.width() - 16, fm.height() + 2)
        p.drawText(tr, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   QtGui.QFontMetrics(f).elidedText(self.title, QtCore.Qt.ElideRight, int(tr.width())))
        p.setFont(self.font())
        p.setPen(pal.color(QtGui.QPalette.PlaceholderText))
        p.drawText(QtCore.QRectF(r.left() + 8, tr.bottom(), r.width() - 16, r.bottom() - tr.bottom() - 2),
                   QtCore.Qt.TextWordWrap | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, self.desc)
        p.end()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.setChecked(True)
            self.activated.emit(self.key)
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    def enterEvent(self, ev):
        self.update()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.update()
        super().leaveEvent(ev)


class Legend(QtWidgets.QWidget):
    ITEMS = (("Column", "column", "line3"), ("Beam", "beam", "line2"), ("Slab", "floor", "fill"),
             ("Wall", "wall", "fill"), ("Support", "support", "tri"), ("Grid line", "grid", "dash"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(20)

    def sizeHint(self):
        return QtCore.QSize(520, 20)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        colors = theme.active().colors()
        x, cy = 0.0, self.height() / 2
        fm = self.fontMetrics()
        for label, key, kind in self.ITEMS:
            c = QtGui.QColor.fromRgbF(*colors[key])
            if kind.startswith("line"):
                p.setPen(QtGui.QPen(c, 3.0 if kind == "line3" else 2.0))
                p.drawLine(QtCore.QPointF(x, cy), QtCore.QPointF(x + 16, cy))
            elif kind == "dash":
                p.setPen(QtGui.QPen(c, 1.2, QtCore.Qt.DashLine))
                p.drawLine(QtCore.QPointF(x, cy), QtCore.QPointF(x + 16, cy))
            elif kind == "tri":
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(c)
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x + 8, cy - 5), QtCore.QPointF(x + 2, cy + 5),
                                               QtCore.QPointF(x + 14, cy + 5)]))
            else:
                edge = QtGui.QColor.fromRgbF(*(colors["area_edge"] if key == "floor" else tuple(0.65 * v for v in colors[key])))
                p.setPen(QtGui.QPen(edge, 1.0))
                fill = QtGui.QColor(c)
                fill.setAlphaF(0.55 if key == "floor" else 0.9)
                p.setBrush(fill)
                p.drawRect(QtCore.QRectF(x + 1, cy - 5, 14, 10))
            p.setPen(self.palette().color(QtGui.QPalette.PlaceholderText))
            p.drawText(QtCore.QPointF(x + 21, cy + fm.ascent() / 2 - 1), label)
            x += 21 + fm.horizontalAdvance(label) + 16
        p.end()


# =============================================================================
# the wizard
# =============================================================================
class NewModelWizard(QtWidgets.QDialog):
    def __init__(self, parent=None, settings: Optional[QtCore.QSettings] = None):
        super().__init__(parent)
        self.settings = settings or QtCore.QSettings("StructuralStudio", "StructuralStudio")
        self.setWindowTitle("New Model")
        self.setObjectName("newModelWizard")
        self.resize(1100, 720)
        self._source_model: Optional[Model] = None
        self._settings_cache: Dict[tuple, Model] = {}
        self._units_text = "kN m C"
        self._building = False

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QtWidgets.QFrame()
        head.setAutoFillBackground(True)
        hp = head.palette()
        hp.setColor(QtGui.QPalette.Window, hp.color(QtGui.QPalette.Base))
        head.setPalette(hp)
        hl = QtWidgets.QHBoxLayout(head)
        hl.setContentsMargins(18, 10, 18, 10)
        tv = QtWidgets.QVBoxLayout()
        tv.setSpacing(2)
        self.title = _bold(QtWidgets.QLabel(), 1.3)
        self.subtitle = _muted(QtWidgets.QLabel())
        tv.addWidget(self.title)
        tv.addWidget(self.subtitle)
        hl.addLayout(tv, 1)
        self.steps = StepBar()
        hl.addWidget(self.steps)
        lay.addWidget(head)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Plain)
        sep.setForegroundRole(QtGui.QPalette.Mid)
        lay.addWidget(sep)

        self.pages = QtWidgets.QStackedWidget()
        # each page scrolls when the window is smaller than it (a small or scaled screen);
        # the step bar and the Back / Next / Create buttons stay outside, always visible
        from .fit import scroll_page
        for make in (self._page_settings, self._page_grid, self._page_structure):
            self.pages.addWidget(scroll_page(make()))
        lay.addWidget(self.pages, 1)

        foot = QtWidgets.QHBoxLayout()
        foot.setContentsMargins(14, 8, 14, 10)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_back = QtWidgets.QPushButton("‹ Back")
        self.btn_back.clicked.connect(lambda: self.go(self.pages.currentIndex() - 1))
        self.btn_alt = QtWidgets.QPushButton()
        self.btn_alt.clicked.connect(self._alt)
        self.btn_next = QtWidgets.QPushButton()
        self.btn_next.setDefault(True)
        self.btn_next.clicked.connect(self._next)
        foot.addWidget(self.btn_cancel)
        foot.addWidget(self.btn_back)
        foot.addStretch(1)
        foot.addWidget(self.btn_alt)
        foot.addWidget(self.btn_next)
        lay.addLayout(foot)
        self.structure = "grid"  # the starting structure: Grid only until another is picked
        self._update_unit_labels()
        self._load_defaults_flag()
        self._source_changed()
        self.go(0)
        from .fit import install
        install(self)  # no larger than the screen it opens on

    # ================================================================ step 1
    def _page_settings(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setContentsMargins(18, 14, 18, 8)
        h.setSpacing(12)
        g, gl = _group("Start from")
        g.setMinimumWidth(230)
        g.setMaximumWidth(330)
        self.src_buttons = QtWidgets.QButtonGroup(self)
        self.src_builtin = self._source_option(gl, "Built-in settings",
                                               "Choose units and a region. Materials and steel sections follow the region.")
        self.src_defaults = self._source_option(gl, "My default settings", "Units and region saved earlier.")
        self.src_model = self._source_option(gl, "Settings from a saved model",
                                             "Copy units, materials, sections and load definitions from another "
                                             "model. No geometry is copied.")
        row = QtWidgets.QHBoxLayout()
        self.src_path = QtWidgets.QLineEdit()
        self.src_path.setPlaceholderText("Model file (.ssj.json)")
        self.src_path.editingFinished.connect(self._load_source)
        self.src_browse = QtWidgets.QPushButton("Browse...")
        self.src_browse.clicked.connect(self._browse_source)
        row.addWidget(self.src_path, 1)
        row.addWidget(self.src_browse)
        gl.addLayout(row)
        self.src_info = _muted(QtWidgets.QLabel())
        self.src_info.setWordWrap(True)
        gl.addWidget(self.src_info)
        gl.addStretch(1)
        # the project data (reports, title blocks): here or later in File > Project Information
        self.project_values = {}
        self.btn_project = QtWidgets.QPushButton("Project Information...")
        self.btn_project.setToolTip("Project name, number, engineer, company ... for reports and drawings "
                                    "(also File > Project Information later)")
        self.btn_project.clicked.connect(self._edit_project)
        gl.addWidget(self.btn_project)
        self.src_builtin.setChecked(True)
        self.src_buttons.buttonToggled.connect(lambda *_: self._source_changed())
        h.addWidget(g)

        g2, gl2 = _group("Built-in settings")
        self.settings_group = g2
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setHorizontalSpacing(14)
        self.units = QtWidgets.QComboBox()
        self.units.addItems([" ".join(p) for p in PRESETS])
        self.units.setCurrentText("kN m C")
        self.units.setToolTip("Force, length, temperature - can be changed any time (units button, lower right)")
        self.units.currentTextChanged.connect(self._units_changed)
        self.region = QtWidgets.QComboBox()
        for key in REGION_ORDER:
            self.region.addItem(REGIONS[key]["label"], key)
        self.region.currentIndexChanged.connect(self._region_changed)
        # steel sections and rebar sizes: any region's table (Thai materials with AISC
        # shapes, ...); they follow the region until one is picked by hand
        self.steel_db = QtWidgets.QComboBox()
        self.rebar_db = QtWidgets.QComboBox()
        for key in REGION_ORDER:
            self.steel_db.addItem(REGIONS[key]["steel_db"], key)
            self.rebar_db.addItem(REGIONS[key]["rebar_label"], key)
        self.steel_db.currentIndexChanged.connect(self._db_changed)
        self.rebar_db.currentIndexChanged.connect(self._db_changed)
        self.code_concrete = QtWidgets.QComboBox()
        self.code_steel = QtWidgets.QComboBox()
        for w in (self.code_concrete, self.code_steel):
            w.setEnabled(False)
            w.setToolTip("Design is planned: the code chosen here will be used when design is available")
        for w in (self.units, self.region, self.steel_db, self.rebar_db, self.code_concrete, self.code_steel):
            w.setMinimumWidth(180)
        form.addRow("Display units", self.units)
        form.addRow("Region (materials)", self.region)
        form.addRow("Steel section database", self.steel_db)
        form.addRow("Rebar sizes", self.rebar_db)
        form.addRow("Concrete design code", self._with_note(self.code_concrete, "planned"))
        form.addRow("Steel design code", self._with_note(self.code_steel, "planned"))
        gl2.addLayout(form)
        gl2.addSpacing(6)
        gl2.addWidget(_bold(QtWidgets.QLabel("Materials added to the model")))
        self.mat_table = QtWidgets.QTableWidget(0, 4)
        self.mat_table.setHorizontalHeaderLabels(["Name", "Type", "Strength", "E"])
        self.mat_table.verticalHeader().setVisible(False)
        self.mat_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.mat_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.mat_table.horizontalHeader().setStretchLastSection(True)
        self.mat_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        gl2.addWidget(self.mat_table, 1)
        self.save_defaults = QtWidgets.QCheckBox("Save as my default settings")
        gl2.addWidget(self.save_defaults)
        h.addWidget(g2, 1)
        self._region_changed()
        return page

    def _source_option(self, lay, title, desc) -> QtWidgets.QRadioButton:
        b = QtWidgets.QRadioButton(title)
        _bold(b)
        self.src_buttons.addButton(b)
        lay.addWidget(b)
        d = _muted(QtWidgets.QLabel(desc))
        d.setWordWrap(True)
        d.setContentsMargins(22, 0, 0, 6)
        lay.addWidget(d)
        return b

    @staticmethod
    def _with_note(w, note) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w)
        h.addWidget(_muted(QtWidgets.QLabel(note)))
        h.addStretch(1)
        return box

    def _region_key(self) -> str:
        return self.region.currentData() or "thailand"

    def _region_changed(self, *_):
        reg = REGIONS[self._region_key()]
        for w in (self.steel_db, self.rebar_db):  # follow the region
            w.blockSignals(True)
            w.setCurrentIndex(max(0, w.findData(self._region_key())))
            w.blockSignals(False)
        for w, text in ((self.code_concrete, reg["concrete_code"]), (self.code_steel, reg["steel_code"])):
            w.clear()
            w.addItem(text)
        self._fill_materials()
        self._refresh_cards()

    def _db_changed(self, *_):
        self._fill_materials()
        self._refresh_cards()

    def _steel_key(self) -> str:
        return self.steel_db.currentData() or self._region_key()

    def _rebar_key(self) -> str:
        return self.rebar_db.currentData() or self._region_key()

    def _units_changed(self, text: str):
        if not text:
            return
        try:
            old, new = UnitSystem.parse(self._units_text), UnitSystem.parse(text)
        except Exception:
            return
        k = old.factor("length") / new.factor("length")
        self._units_text = text
        if abs(k - 1.0) > 1e-12 and hasattr(self, "u_xs"):
            self._rescale(k)
        self._fill_materials()
        self._update_unit_labels()
        self._refresh_cards()

    def unit_system(self) -> UnitSystem:
        if self.src_model.isChecked() and self._source_model is not None:
            return self._source_model.units
        return UnitSystem.parse(self.units.currentText())

    def settings_model(self) -> Model:
        """A model with the chosen settings and no geometry (sections for the
        option boxes and the cards)."""
        if self.src_model.isChecked() and self._source_model is not None:
            key = ("model", id(self._source_model))
        else:
            key = ("region", self._region_key(), self.units.currentText(), self._steel_key(), self._rebar_key())
        m = self._settings_cache.get(key)
        if m is None:
            m = Model("settings", self.unit_system())
            if key[0] == "model":
                from .templates import copy_settings
                copy_settings(m, self._source_model)
            else:
                region_properties(m, self._region_key(), self._steel_key(), self._rebar_key())
            self._settings_cache[key] = m
        return m

    def _fill_materials(self):
        m = self.settings_model()
        u = m.units
        stress = u.shown("stress")
        rows = []
        for mat in m.materials.values():
            st = mat.fc if mat.type == "Concrete" else mat.fy
            label = "f'c" if mat.type == "Concrete" else "fy"
            rows.append((mat.name, mat.type, f"{label} = {_num(st)} {stress}" if st else "", f"{_num(mat.E)} {stress}"))
        self.mat_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                self.mat_table.setItem(i, j, QtWidgets.QTableWidgetItem(v))

    def _source_changed(self):
        builtin = self.src_builtin.isChecked()
        from_model = self.src_model.isChecked()
        self.src_path.setEnabled(from_model)
        self.src_browse.setEnabled(from_model)
        if self.src_defaults.isChecked():
            d = self._defaults()
            if d:
                self.units.setCurrentText(d.get("units", self.units.currentText()))
                i = self.region.findData(d.get("region"))
                if i >= 0:
                    self.region.setCurrentIndex(i)
                for w, k in ((self.steel_db, "steel_db"), (self.rebar_db, "rebar_db")):
                    j = w.findData(d.get(k))
                    if j >= 0:
                        w.setCurrentIndex(j)
        for w in (self.units, self.region, self.steel_db, self.rebar_db):
            w.setEnabled(builtin)
        self.settings_group.setTitle("Settings of the model" if from_model else
                                     ("My default settings" if self.src_defaults.isChecked() else "Built-in settings"))
        self.save_defaults.setEnabled(builtin)
        if from_model and self._source_model is not None:
            self.units.blockSignals(True)
            self.units.setCurrentText(_utext(self._source_model.units))
            self.units.blockSignals(False)
            self._units_changed(self.units.currentText())
        self._fill_materials()
        self._refresh_cards()
        self._update_buttons()

    def _defaults(self) -> Optional[dict]:
        raw = self.settings.value(KEY_DEFAULTS, "")
        try:
            d = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            d = None
        return d if isinstance(d, dict) else None

    def _load_defaults_flag(self):
        has = self._defaults() is not None
        self.src_defaults.setEnabled(has)
        if not has:
            self.src_defaults.setToolTip("Nothing saved yet: tick 'Save as my default settings' once")

    def _browse_source(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Settings from a saved model", "",
                                                        "Structural Studio (*.ssj.json *.json)")
        if path:
            self.src_path.setText(path)
            self._load_source()

    def _load_source(self):
        path = self.src_path.text().strip()
        if not path:
            return
        try:
            self._source_model = Model.from_json(path)
        except Exception as exc:
            self._source_model = None
            self.src_info.setText(f"Cannot read this file: {exc}")
        else:
            m = self._source_model
            self.src_info.setText(f"{len(m.materials)} materials, {len(m.frame_sections)} frame and "
                                  f"{len(m.shell_sections)} shell sections, units {m.units.text}")
        self._source_changed()

    # ================================================================ step 2
    def _page_grid(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setContentsMargins(18, 12, 18, 8)
        h.setSpacing(12)
        left = QtWidgets.QVBoxLayout()
        top = QtWidgets.QHBoxLayout()
        self.mode_uniform = QtWidgets.QToolButton()
        self.mode_custom = QtWidgets.QToolButton()
        for b, t in ((self.mode_uniform, "Uniform"), (self.mode_custom, "Custom (tables)")):
            b.setText(t)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setMinimumWidth(120)
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            top.addWidget(b)
        self.mode_uniform.setChecked(True)
        self.mode_uniform.toggled.connect(self._grid_mode_changed)
        from .dialogs import GridModeSwitch
        self.grid_system = GridModeSwitch()
        top.addSpacing(12)
        top.addWidget(self.grid_system)
        top.addStretch(1)
        left.addLayout(top)
        self.grid_stack = QtWidgets.QStackedWidget()
        self.grid_stack.addWidget(self._uniform_page())
        self.grid_stack.addWidget(self._custom_page())
        left.addWidget(self.grid_stack, 1)
        self.total_label = _muted(QtWidgets.QLabel())
        self.total_label.setWordWrap(True)
        left.addWidget(self.total_label)
        h.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        self.plan_preview = GridPreview("plan")
        self.elev_preview = GridPreview("elev")
        for title, w in (("Plan", self.plan_preview), ("Elevation", self.elev_preview)):
            g, gl = _group(title)
            gl.setContentsMargins(4, 4, 4, 4)
            gl.addWidget(w)
            right.addWidget(g, 1)
        self.elev_group = right.itemAt(1).widget()
        h.addLayout(right)
        self.plan_preview.setMinimumWidth(340)
        self._enter_by_changed()
        self._grid_mode_changed()
        return page

    def _uniform_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        g, gl = _group("Grid lines")
        f = QtWidgets.QGridLayout()
        f.setHorizontalSpacing(10)
        self.u_unit_labels = []
        hdr2 = _muted(QtWidgets.QLabel())
        self.u_unit_labels.append(hdr2)
        f.addWidget(_muted(QtWidgets.QLabel("Number of lines")), 0, 1)
        f.addWidget(hdr2, 0, 2)
        self.u_nx, self.u_ny = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        self.u_xs, self.u_ys = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for r, (lab, n, s, nv, sv) in enumerate((("X direction", self.u_nx, self.u_xs, 4, 8.0),
                                                 ("Y direction", self.u_ny, self.u_ys, 4, 7.5)), 1):
            n.setMaximumWidth(140)
            s.setMaximumWidth(140)
            n.setRange(1, 200)
            n.setValue(nv)
            s.setRange(0.001, 1e6)
            s.setDecimals(3)
            s.setValue(sv)
            f.addWidget(QtWidgets.QLabel(lab), r, 0)
            f.addWidget(n, r, 1)
            f.addWidget(s, r, 2)
        self.u_labels = QtWidgets.QComboBox()
        self.u_labels.addItems(["X: A, B, C …    Y: 1, 2, 3 …", "X: 1, 2, 3 …    Y: A, B, C …"])
        f.addWidget(QtWidgets.QLabel("Labels"), 3, 0)
        f.addWidget(self.u_labels, 3, 1, 1, 2)
        f.setColumnStretch(3, 1)
        gl.addLayout(f)
        v.addWidget(g)
        g2, gl2 = _group("Stories")
        f2 = QtWidgets.QGridLayout()
        f2.setHorizontalSpacing(10)
        self.u_ns = QtWidgets.QSpinBox()
        self.u_ns.setRange(1, 300)
        self.u_ns.setValue(8)
        self.u_h, self.u_h1 = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        self.u_ns.setMaximumWidth(140)
        for s, val in ((self.u_h, 3.5), (self.u_h1, 4.5)):
            s.setMaximumWidth(140)
            s.setRange(0.001, 1e6)
            s.setDecimals(3)
            s.setValue(val)
        self.u_h_label = QtWidgets.QLabel()
        self.u_h1_label = QtWidgets.QLabel()
        f2.addWidget(QtWidgets.QLabel("Number of stories"), 0, 0)
        f2.addWidget(self.u_ns, 0, 1)
        f2.addWidget(self.u_h_label, 1, 0)
        f2.addWidget(self.u_h, 1, 1)
        f2.addWidget(self.u_h1_label, 2, 0)
        f2.addWidget(self.u_h1, 2, 1)
        f2.setColumnStretch(2, 1)
        gl2.addLayout(f2)
        v.addWidget(g2)
        v.addStretch(1)
        for w_ in (self.u_nx, self.u_ny, self.u_ns):
            w_.valueChanged.connect(self._grid_changed)
        for w_ in (self.u_xs, self.u_ys, self.u_h, self.u_h1):
            w_.valueChanged.connect(self._grid_changed)
        self.u_labels.currentIndexChanged.connect(self._grid_changed)
        return w

    def _custom_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        opts = QtWidgets.QHBoxLayout()
        self.enter_by_label = QtWidgets.QLabel("Type grid lines by")
        self.enter_by = QtWidgets.QComboBox()
        self.enter_by.addItems(["Spacing", "Ordinate"])
        self.enter_by.currentIndexChanged.connect(self._enter_by_changed)
        opts.addWidget(self.enter_by_label)
        opts.addWidget(self.enter_by)
        opts.addStretch(1)
        self.dxf = QtWidgets.QPushButton("Import from DXF...")
        self.dxf.setEnabled(False)
        self.dxf.setToolTip("Planned: read grid lines from a DXF layer")
        opts.addWidget(self.dxf)
        v.addLayout(opts)
        row = QtWidgets.QHBoxLayout()
        self.t_x = self._axis_table()
        self.t_y = self._axis_table()
        for title, t in (("X grid lines", self.t_x), ("Y grid lines", self.t_y)):
            g, gl = _group(title)
            gl.setContentsMargins(4, 4, 4, 4)
            gl.addWidget(t, 1)
            gl.addLayout(self._table_buttons(t, (("Add", "add"), ("Insert", "insert"), ("Delete", "delete"),
                                                 ("Relabel", "relabel"))))
            row.addWidget(g)
        v.addLayout(row, 1)
        g, gl = _group("Stories (top to bottom)")
        gl.setContentsMargins(4, 4, 4, 4)
        self.t_st = QtWidgets.QTableWidget(0, 5)
        self.t_st.verticalHeader().setVisible(False)
        sh = self.t_st.horizontalHeader()
        for c, mode in enumerate((QtWidgets.QHeaderView.Stretch, QtWidgets.QHeaderView.Stretch,
                                  QtWidgets.QHeaderView.Stretch, QtWidgets.QHeaderView.ResizeToContents,
                                  QtWidgets.QHeaderView.Stretch)):
            sh.setSectionResizeMode(c, mode)
        self.t_st.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.t_st.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.t_st.itemChanged.connect(self._story_item_changed)
        self.t_st.currentCellChanged.connect(lambda *_: self._update_previews())
        gl.addWidget(self.t_st, 1)
        gl.addLayout(self._table_buttons(self.t_st, (("Add above", "above"), ("Add below", "below"),
                                                     ("Delete", "delete"))))
        v.addWidget(g, 1)
        return w

    def _axis_table(self) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, 5)
        t.verticalHeader().setVisible(False)
        hh = t.horizontalHeader()
        hh.setStretchLastSection(False)
        for c, mode in enumerate((QtWidgets.QHeaderView.ResizeToContents, QtWidgets.QHeaderView.Stretch,
                                  QtWidgets.QHeaderView.Stretch, QtWidgets.QHeaderView.ResizeToContents,
                                  QtWidgets.QHeaderView.ResizeToContents)):
            hh.setSectionResizeMode(c, mode)
        t.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        t.itemChanged.connect(lambda it, t=t: self._axis_item_changed(t, it))
        t.currentCellChanged.connect(lambda *_: self._update_previews())
        return t

    def _table_buttons(self, table, items) -> QtWidgets.QHBoxLayout:
        h = QtWidgets.QHBoxLayout()
        for text, act in items:
            b = QtWidgets.QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, t=table, a=act: self._table_action(t, a))
            h.addWidget(b)
            if act == "delete":
                h.addStretch(1)
        return h

    # ---------------------------------------------------------- grid data
    def _length_unit(self) -> str:
        return self.unit_system().length

    def _update_unit_labels(self):
        if not hasattr(self, "u_h_label"):
            return
        u = self._length_unit()
        for lab in self.u_unit_labels:
            lab.setText(f"Spacing ({u})")
        self.u_h_label.setText(f"Typical story height ({u})")
        self.u_h1_label.setText(f"Bottom story height ({u})")
        self._axis_headers()
        self.t_st.setHorizontalHeaderLabels(["Story", f"Height ({u})", f"Elevation ({u})", "Master", "Similar to"])
        self._grid_changed()

    def _axis_headers(self):
        u = self._length_unit()
        for t in (self.t_x, self.t_y):
            t.setHorizontalHeaderLabels(["Label", f"Ordinate ({u})", f"Spacing ({u})", "Bubble", "Show"])

    def _rescale(self, k: float):
        for s in (self.u_xs, self.u_ys, self.u_h, self.u_h1):
            s.blockSignals(True)
            s.setValue(s.value() * k)
            s.blockSignals(False)
        if self._custom_filled:
            xs, ys, st = self._custom_data()
            self._fill_custom([(l, o * k, b, v) for l, o, b, v in xs], [(l, o * k, b, v) for l, o, b, v in ys],
                              [(n, h * k, m, s) for n, h, m, s in st])

    def is_custom(self) -> bool:
        return self.mode_custom.isChecked()

    def _uniform_data(self):
        xs = [i * self.u_xs.value() for i in range(self.u_nx.value())]
        ys = [j * self.u_ys.value() for j in range(self.u_ny.value())]
        letters_x = self.u_labels.currentIndex() == 0
        xl = _letters(len(xs)) if letters_x else [str(i + 1) for i in range(len(xs))]
        yl = [str(j + 1) for j in range(len(ys))] if letters_x else _letters(len(ys))
        n = self.u_ns.value()
        stories = [(f"Story{i}", self.u_h1.value() if i == 1 else self.u_h.value()) for i in range(1, n + 1)]
        return ([(l, o, "End", True) for l, o in zip(xl, xs)], [(l, o, "End", True) for l, o in zip(yl, ys)],
                [(n_, h, False, "") for n_, h in stories])

    _custom_filled = False

    def _fill_custom(self, xs, ys, stories):
        """Tables from (label, ordinate, bubble, visible) rows and stories bottom up
        (name, height, master, similar_to)."""
        self._building = True
        for t, rows in ((self.t_x, xs), (self.t_y, ys)):
            t.setRowCount(0)
            for r in rows:
                self._axis_row(t, t.rowCount(), *r)
            self._axis_spacings(t)
        self.t_st.setRowCount(0)
        for name, h, master, sim in reversed(stories):
            self._story_row(self.t_st.rowCount(), name, h, master, sim)
        self._story_row(self.t_st.rowCount(), "Base", None, False, "")
        self._story_elevations()
        self._building = False
        self._custom_filled = True

    def _axis_row(self, t, r, label, ordinate, bubble="End", visible=True):
        t.insertRow(r)
        t.setItem(r, 0, QtWidgets.QTableWidgetItem(str(label)))
        t.setItem(r, 1, QtWidgets.QTableWidgetItem(_fmt(ordinate)))
        t.setItem(r, 2, QtWidgets.QTableWidgetItem(""))
        b = QtWidgets.QComboBox()
        b.addItems(["End", "Start"])
        b.setCurrentText(bubble)
        b.setFrame(False)
        t.setCellWidget(r, 3, b)
        show = QtWidgets.QTableWidgetItem()
        show.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)
        show.setCheckState(QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked)
        t.setItem(r, 4, show)

    def _story_row(self, r, name, h, master, sim):
        t = self.t_st
        t.insertRow(r)
        t.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
        hi = QtWidgets.QTableWidgetItem("" if h is None else _fmt(h))
        el = QtWidgets.QTableWidgetItem("")
        el.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        el.setForeground(self.palette().color(QtGui.QPalette.PlaceholderText))
        t.setItem(r, 1, hi)
        t.setItem(r, 2, el)
        m = QtWidgets.QTableWidgetItem()
        if h is None:  # the base level
            hi.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            hi.setText("–")
            m.setFlags(QtCore.Qt.ItemIsEnabled)
            t.setItem(r, 3, m)
            t.setItem(r, 4, QtWidgets.QTableWidgetItem(""))
            t.item(r, 4).setFlags(QtCore.Qt.ItemIsEnabled)
            return
        m.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)
        m.setCheckState(QtCore.Qt.Checked if master else QtCore.Qt.Unchecked)
        t.setItem(r, 3, m)
        sim_box = QtWidgets.QComboBox()
        sim_box.setFrame(False)
        sim_box.setProperty("value", sim or "")
        sim_box.currentTextChanged.connect(lambda _t: None)
        t.setCellWidget(r, 4, sim_box)

    def _story_names(self) -> List[str]:
        return [self.t_st.item(r, 0).text().strip() for r in range(self.t_st.rowCount() - 1)]

    def _refresh_similar(self):
        names = self._story_names()
        masters = [self.t_st.item(r, 0).text().strip() for r in range(self.t_st.rowCount() - 1)
                   if self.t_st.item(r, 3).checkState() == QtCore.Qt.Checked]
        for r in range(self.t_st.rowCount() - 1):
            box = self.t_st.cellWidget(r, 4)
            if box is None:
                continue
            cur = box.currentText() or box.property("value") or ""
            box.blockSignals(True)
            box.clear()
            is_master = self.t_st.item(r, 3).checkState() == QtCore.Qt.Checked
            box.addItem("–" if is_master else "None")
            if not is_master:
                box.addItems([m for m in masters if m != names[r]])
            i = box.findText(cur)
            box.setCurrentIndex(i if i >= 0 else 0)
            box.setEnabled(not is_master)
            box.blockSignals(False)

    def _axis_spacings(self, t):
        self._building, was = True, self._building
        prev = None
        for r in range(t.rowCount()):
            try:
                o = float(t.item(r, 1).text())
            except (TypeError, ValueError, AttributeError):
                o = None
            it = t.item(r, 2)
            it.setText("–" if r == 0 or prev is None or o is None else _fmt(o - prev))
            if r == 0:
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            prev = o
        self._building = was

    def _story_elevations(self):
        self._building, was = True, self._building
        n = self.t_st.rowCount()
        z = 0.0
        for r in range(n - 1, -1, -1):
            if r < n - 1:
                try:
                    z += float(self.t_st.item(r, 1).text())
                except (TypeError, ValueError):
                    pass
            self.t_st.item(r, 2).setText(_fmt(z))
        self._building = was
        self._refresh_similar()

    def _enter_by_changed(self, *_):
        was, self._building = self._building, True  # flag / colour changes emit itemChanged too
        try:
            self._set_editable()
        finally:
            self._building = was

    def _set_editable(self):
        spacing = self.enter_by.currentIndex() == 0
        for t in (getattr(self, "t_x", None), getattr(self, "t_y", None)):
            if t is None:
                continue
            for r in range(t.rowCount()):
                for c, editable in ((1, not spacing), (2, spacing and r > 0)):
                    it = t.item(r, c)
                    if it is None:
                        continue
                    flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                    if editable:
                        flags |= QtCore.Qt.ItemIsEditable
                    it.setFlags(flags)
                    it.setForeground(self.palette().color(QtGui.QPalette.Text if editable else QtGui.QPalette.PlaceholderText))

    def _axis_item_changed(self, t, it):
        if self._building:
            return
        if it.column() == 2 and it.row() > 0:  # spacing typed: move this and the next lines
            try:
                sp = float(it.text())
                o_prev = float(t.item(it.row() - 1, 1).text())
                old = float(t.item(it.row(), 1).text())
            except (TypeError, ValueError):
                return
            shift = o_prev + sp - old
            self._building = True
            for r in range(it.row(), t.rowCount()):
                try:
                    t.item(r, 1).setText(_fmt(float(t.item(r, 1).text()) + shift))
                except (TypeError, ValueError):
                    pass
            self._building = False
        self._axis_spacings(t)
        self._enter_by_changed()
        self._update_previews()

    def _story_item_changed(self, it):
        if self._building:
            return
        if it.column() == 3:
            self._refresh_similar()
        self._story_elevations()
        self._update_previews()

    def _table_action(self, t, act):
        r = t.currentRow()
        if t is self.t_st:
            n = t.rowCount()
            if act in ("above", "below"):
                at = (r if r >= 0 else 0) if act == "above" else (min(r + 1, n - 1) if r >= 0 else n - 1)
                names = set(self._story_names())
                k = len(names) + 1
                while f"Story{k}" in names:
                    k += 1
                try:
                    h = float(t.item(min(at, n - 2), 1).text()) if n > 1 else 3.0
                except (TypeError, ValueError):
                    h = 3.0
                self._building = True
                self._story_row(at, f"Story{k}", h, False, "")
                self._building = False
                t.setCurrentCell(at, 0)
            elif act == "delete" and 0 <= r < n - 1 and n > 2:
                t.removeRow(r)
            self._story_elevations()
        else:
            n = t.rowCount()
            if act in ("add", "insert"):
                rows = self._axis_rows(t)
                if act == "add" or r < 0:
                    sp = rows[-1][1] - rows[-2][1] if len(rows) > 1 else 6.0
                    at, o = n, (rows[-1][1] + sp if rows else 0.0)
                else:
                    at = r
                    o = (rows[r - 1][1] + rows[r][1]) / 2 if r > 0 else rows[0][1] - 1.0
                labels = [x[0] for x in rows]
                lab = next(c for c in (_letters(n + 26) if t is self._letter_table() else
                                       [str(i) for i in range(1, n + 30)]) if c not in labels)
                self._building = True
                self._axis_row(t, at, lab, o)
                self._building = False
                t.setCurrentCell(at, 1)
            elif act == "delete" and r >= 0 and n > 1:
                t.removeRow(r)
            elif act == "relabel":
                self._building = True
                labels = _letters(n) if t is self._letter_table() else [str(i) for i in range(1, n + 1)]
                order = sorted(range(n), key=lambda i: float(t.item(i, 1).text() or 0))
                for k_, i in enumerate(order):
                    t.item(i, 0).setText(labels[k_])
                self._building = False
            self._sort_axis(t)
            self._axis_spacings(t)
            self._enter_by_changed()
        self._update_previews()

    def _letter_table(self):
        return self.t_x if self.u_labels.currentIndex() == 0 else self.t_y

    def _sort_axis(self, t):
        rows = self._axis_rows(t)
        if [r[1] for r in rows] == sorted(r[1] for r in rows):
            return
        cur = t.currentRow()
        key = rows[cur][0] if 0 <= cur < len(rows) else None
        self._building = True
        t.setRowCount(0)
        for r in sorted(rows, key=lambda x: x[1]):
            self._axis_row(t, t.rowCount(), *r)
        self._building = False
        if key is not None:
            for i in range(t.rowCount()):
                if t.item(i, 0).text() == key:
                    t.setCurrentCell(i, 1)

    def _axis_rows(self, t):
        out = []
        for r in range(t.rowCount()):
            try:
                o = float(t.item(r, 1).text())
            except (TypeError, ValueError, AttributeError):
                continue
            b = t.cellWidget(r, 3)
            out.append((t.item(r, 0).text().strip(), o, b.currentText() if b else "End",
                        t.item(r, 4).checkState() == QtCore.Qt.Checked))
        return out

    def _custom_data(self):
        xs, ys = self._axis_rows(self.t_x), self._axis_rows(self.t_y)
        st = []
        for r in range(self.t_st.rowCount() - 2, -1, -1):  # bottom up, without the base
            try:
                h = float(self.t_st.item(r, 1).text())
            except (TypeError, ValueError):
                h = 0.0
            box = self.t_st.cellWidget(r, 4)
            sim = box.currentText() if box is not None else ""
            st.append((self.t_st.item(r, 0).text().strip(), h, self.t_st.item(r, 3).checkState() == QtCore.Qt.Checked,
                       "" if sim in ("None", "–") else sim))
        return xs, ys, st

    def grid_data(self):
        """((label, ordinate, bubble, visible) X rows, Y rows, stories bottom up
        (name, height, master, similar_to))."""
        return self._custom_data() if self.is_custom() and self._custom_filled else self._uniform_data()

    def _grid_mode_changed(self, *_):
        custom = self.is_custom()
        if custom and not self._custom_filled:
            self._fill_custom(*self._uniform_data())
            self._enter_by_changed()
        self.grid_stack.setCurrentIndex(1 if custom else 0)
        self._grid_changed()

    def _grid_changed(self, *_):
        self._update_previews()

    def _update_previews(self):
        if not hasattr(self, "plan_preview"):
            return
        xs, ys, st = self.grid_data()
        levels, z = [("Base", 0.0)], 0.0
        for name, h, _, _ in st:
            z += h
            levels.append((name, z))
        cur = None
        if self.is_custom():
            for key, t in (("x", self.t_x), ("y", self.t_y)):
                if t.hasFocus() or t.currentRow() >= 0 and self.focusWidget() is t:
                    cur = (key, t.currentRow())
            if self.t_st.currentRow() >= 0 and (self.focusWidget() is self.t_st or cur is None):
                cur = ("story", self.t_st.rowCount() - 1 - self.t_st.currentRow())
        self.plan_preview.set_data([r[1] for r in xs], [r[1] for r in ys], [r[0] for r in xs], [r[0] for r in ys],
                                   levels, cur)
        self.elev_preview.set_data([r[1] for r in xs], [r[1] for r in ys], [r[0] for r in xs], [r[0] for r in ys],
                                   levels, cur)
        if ys:
            self.elev_group.setTitle(f"Elevation · grid {ys[0][0]}")
        u = self._length_unit()
        self.total_label.setText(
            f"{len(xs)} × {len(ys)} grid lines, {len(st)} stor{'y' if len(st) == 1 else 'ies'}, total height "
            f"{_fmt(z)} {u}." + ("" if self.is_custom() else
                                 "  Use Custom for uneven spacing, story names or similar stories."))

    # ================================================================ step 3
    def _page_structure(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setContentsMargins(18, 10, 18, 8)
        h.setSpacing(12)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(Legend())
        self.preview_note = _muted(QtWidgets.QLabel())
        left.addWidget(self.preview_note)
        self.card_grid = CardGrid()
        self.cards: Dict[str, StructureCard] = {}
        for i, (key, title, desc) in enumerate(STRUCTURES):
            c = StructureCard(key, title, desc)
            c.toggled.connect(lambda on, k=key: on and self._structure_changed(k))
            c.activated.connect(self._card_activated)
            self.card_grid.add(c)
            self.cards[key] = c
        left.addWidget(self.card_grid, 1)
        h.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        self.opt_group, gl = _group("")
        self.opt_group.setFixedWidth(260)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.opt_column, self.opt_beam, self.opt_infill = QtWidgets.QComboBox(), QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.opt_slab, self.opt_wall, self.opt_support = QtWidgets.QComboBox(), QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.opt_support.addItems(list(SUPPORTS))
        for w in (self.opt_column, self.opt_beam, self.opt_infill, self.opt_slab, self.opt_wall, self.opt_support):
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._opt_rows = {}
        for key, label, w in (("column", "Columns", self.opt_column), ("beam", "Beams", self.opt_beam),
                              ("infill", "Infill beams", self.opt_infill), ("slab", "Slabs", self.opt_slab),
                              ("wall", "Walls", self.opt_wall), ("support", "Supports at Base", self.opt_support)):
            lab = QtWidgets.QLabel(label)
            form.addRow(lab, w)
            self._opt_rows[key] = (lab, w)
        gl.addLayout(form)
        right.addWidget(self.opt_group)
        g2, gl2 = _group("Will be created")
        g2.setFixedWidth(260)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        gl2.addWidget(self.summary)
        right.addWidget(g2)
        right.addStretch(1)
        h.addLayout(right)
        self.cards["grid"].setChecked(True)
        return page

    def _card_activated(self, key: str):
        """Double-click on a card: that structure, and Create model."""
        if self.pages.currentIndex() == 2 and self.btn_next.isEnabled():
            self._structure_changed(key)
            self._finish()

    def _structure_changed(self, key: str):
        self.structure = key
        title = next(t for k, t, _ in STRUCTURES if k == key)
        self.opt_group.setTitle(title)
        self._fill_options()
        self._update_summary()

    def _fill_options(self):
        m = self.settings_model()
        key = self.structure
        steel = key == "steel_deck"

        def frames(prefix=None, is_steel=False):
            out = []
            for n, s in m.frame_sections.items():
                mat = m.materials.get(s.material)
                st = mat is not None and mat.type == "Steel"
                if st == is_steel and (prefix is None or n.upper().startswith(prefix)):
                    out.append(n)
            return out or [n for n in m.frame_sections]

        def fill(box, names, prefer=None):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(names)
            i = box.findText(cur if cur in names else (prefer or (names[0] if names else "")))
            box.setCurrentIndex(max(0, i))
            box.blockSignals(False)
        steels = frames(None, True)
        if steel:
            fill(self.opt_column, steels, steels[0] if steels else None)
            fill(self.opt_beam, steels, steels[2] if len(steels) > 2 else None)
            fill(self.opt_infill, steels, steels[-1] if steels else None)
        else:
            fill(self.opt_column, frames("C"))
            fill(self.opt_beam, frames("B"))
        fill(self.opt_slab, [n for n, s in m.shell_sections.items() if s.type == "Slab"])
        fill(self.opt_wall, [n for n, s in m.shell_sections.items() if s.type == "Wall"])
        show = {
            "column": key not in ("blank", "grid"),
            "beam": key in ("rc_frame", "flat_slab_edge", "core_frame", "steel_deck", "frame_only"),
            "infill": steel, "slab": key not in ("blank", "grid", "frame_only"),
            "wall": key == "core_frame", "support": key not in ("blank", "grid")}
        for k, (lab, w) in self._opt_rows.items():
            lab.setVisible(show[k])
            w.setVisible(show[k])
        self.opt_group.setVisible(any(show.values()))

    def _update_summary(self):
        xs, ys, st = self.grid_data()
        c = structure_counts(self.structure, len(xs), len(ys), len(st))
        if self.structure == "blank":
            text = f"{len(st)} stories, no grid lines, no members"
        elif self.structure == "grid":
            text = f"{len(xs)} × {len(ys)} grid lines, {len(st)} stories, no members"
        else:
            parts = [f"{c['columns']:,} columns", f"{c['beams']:,} beams" if c["beams"] else "",
                     f"{c['slabs']:,} slab panels" if c["slabs"] else "", f"{c['walls']:,} wall panels" if c["walls"] else ""]
            sup = self.opt_support.currentText()
            text = " · ".join(p for p in parts if p) + (f"\n{c['supports']:,} {sup.lower()} supports at Base"
                                                        if sup != "None" else "\nno supports")
        self.summary.setText(text)

    def _refresh_cards(self):
        """Draw every card from the model the wizard would build, bounded to the lowest
        stories and the first grid lines (said in the note above the cards).  Cards are
        cached; what does not fit in a short time budget is drawn in later event-loop turns
        (a generation number drops work for a grid that has changed since), so the page
        opens and the window answers at once whatever the grid size."""
        if not hasattr(self, "cards"):
            return
        xs, ys, st = self.grid_data()
        low = st[:PREVIEW_STORIES]
        px, py = xs[:PREVIEW_LINES], ys[:PREVIEW_LINES]
        n = len(st)
        cut = len(px) < len(xs) or len(py) < len(ys)
        self.preview_note.setText(
            f"Previews follow your grid: {len(xs)} × {len(ys)} lines." +
            (f"  The first {len(px)} × {len(py)} lines" if cut else "") +
            ((" and the lowest" if cut else "  The lowest") + f" {len(low)} of {n} stories" if n > len(low) else "") +
            (" are drawn." if cut or n > len(low) else "") +
            ("  The model gets them all." if cut or n > len(low) else ""))
        base = self.settings_model()
        self._card_gen = getattr(self, "_card_gen", 0) + 1
        gen = self._card_gen
        sig = (_utext(base.units), id(base), tuple(r[1] for r in px), tuple(r[1] for r in py),
               tuple((s[0], s[1]) for s in low))
        cache = self.__dict__.setdefault("_card_cache", {})
        todo = []
        for key, card in self.cards.items():
            hit = cache.get((key,) + sig)
            if hit is not None:
                card.set_sketch(*hit)
            else:
                card.set_sketch(None, "Drawing preview...")
                todo.append(key)

        def build(key):
            try:
                m = build_model(units=sig[0], x_ordinates=list(sig[2]), y_ordinates=list(sig[3]),
                                stories=list(sig[4]), structure=key, settings_from=base, properties=False)
                out = (model_sketch(m), "Empty model" if key == "blank" else "")
            except Exception:
                out = (None, "No preview")
            if len(cache) > 64:
                cache.clear()
            cache[(key,) + sig] = out
            self.cards[key].set_sketch(*out)

        def work(budget=PREVIEW_BUDGET):
            if gen != self._card_gen:
                return  # the grid / settings changed: a newer refresh owns the cards
            t0 = time.perf_counter()
            while todo and (time.perf_counter() - t0 < budget):
                build(todo.pop(0))
            if todo:
                QtCore.QTimer.singleShot(0, work)

        work()
        self._fill_options()
        self._update_summary()

    def previews_pending(self) -> bool:
        return any(c.sketch is None and c.empty_text == "Drawing preview..." for c in self.cards.values())

    # ================================================================ flow
    def go(self, i: int):
        i = max(0, min(2, i))
        if i == 2:
            self._refresh_cards()
        self.pages.setCurrentIndex(i)
        self.steps.set_step(i)
        self.title.setText(STEP_TITLES[i][0])
        self.subtitle.setText(STEP_TITLES[i][1])
        self._update_buttons()

    def _update_buttons(self):
        i = self.pages.currentIndex()
        self.btn_back.setVisible(i > 0)
        self.btn_alt.setText(("Blank model", "Finish with grid only", "")[i])
        self.btn_alt.setVisible(i < 2)
        self.btn_next.setText("Create model" if i == 2 else "Next ›")
        ok = not (self.src_model.isChecked() and self._source_model is None)
        self.btn_next.setEnabled(ok)
        self.btn_alt.setEnabled(ok)

    def _next(self):
        i = self.pages.currentIndex()
        if i == 1:
            err = self._grid_error()
            if err:
                QtWidgets.QMessageBox.warning(self, "Grid lines and stories", err)
                return
        if i < 2:
            self.go(i + 1)
        else:
            self._finish()

    def _alt(self):
        i = self.pages.currentIndex()
        if i == 1:
            err = self._grid_error()
            if err:
                QtWidgets.QMessageBox.warning(self, "Grid lines and stories", err)
                return
        self.structure = "blank" if i == 0 else "grid"
        self._finish()

    def _grid_error(self) -> str:
        xs, ys, st = self.grid_data()
        for name, rows in (("X", xs), ("Y", ys)):
            o = [r[1] for r in rows]
            if len(o) < 1:
                return f"Give at least one {name} grid line."
            if len(set(round(v, 9) for v in o)) != len(o):
                return f"Two {name} grid lines have the same ordinate."
            labels = [r[0] for r in rows]
            if any(not lab for lab in labels) or len(set(labels)) != len(labels):
                return f"Every {name} grid line needs its own label."
        if not st:
            return "Give at least one story."
        if any(h <= 0 for _, h, _, _ in st):
            return "Story heights must be greater than zero."
        names = [s[0] for s in st] + ["Base"]
        if any(not n for n in names) or len(set(names)) != len(names):
            return "Every story needs its own name."
        return ""

    def _finish(self):
        try:
            self._model = self.build()
        except Exception as exc:  # a message, not a crash
            QtWidgets.QMessageBox.warning(self, "New Model", str(exc))
            return
        if self.save_defaults.isChecked() and self.src_builtin.isChecked():
            self.settings.setValue(KEY_DEFAULTS, json.dumps({"units": self.units.currentText(),
                                                             "region": self._region_key(),
                                                             "steel_db": self._steel_key(),
                                                             "rebar_db": self._rebar_key()}))
        self.accept()

    def _edit_project(self):
        from .project_info import edit_project_info
        v = edit_project_info(dict(self.project_values), self)
        if v is not None:
            self.project_values = v
            self.btn_project.setText(f"Project Information: {v.get('name') or 'Untitled'}")

    def build(self) -> Model:
        xs, ys, st = self.grid_data()
        structure = self.structure
        if structure != "blank" and (len(xs) < 2 or len(ys) < 2) and structure != "grid":
            raise ValueError("This structure needs at least two grid lines in X and in Y.")
        from_model = self.src_model.isChecked() and self._source_model is not None
        m = build_model(units=_utext(self.unit_system()), region=self._region_key(),
                        steel_db=self._steel_key(), rebar_db=self._rebar_key(),
                        x_ordinates=[r[1] for r in xs], y_ordinates=[r[1] for r in ys],
                        x_labels=[r[0] for r in xs], y_labels=[r[0] for r in ys],
                        stories=[(s[0], s[1]) for s in st],
                        grid_mode=self.grid_system.mode(), structure=structure,
                        column=self.opt_column.currentText() or None, beam=self.opt_beam.currentText() or None,
                        slab=self.opt_slab.currentText() or None, wall=self.opt_wall.currentText() or None,
                        infill=self.opt_infill.currentText() or None, support=self.opt_support.currentText(),
                        settings_from=self._source_model if from_model else None)
        if self.is_custom():
            for name, _, master, sim in st:
                story = m.stories.get(name)
                if story is not None:
                    story.is_master = bool(master)
                    story.similar_to = None if master or not sim else sim
        extra = {("X", r[0]): r for r in xs}
        extra.update({("Y", r[0]): r for r in ys})
        for g in m.grid_lines:
            r = extra.get((g.direction, g.name))
            if r is not None:
                g.bubble_location, g.visible = r[2], bool(r[3])
        for k, v in getattr(self, "project_values", {}).items():
            if hasattr(m.info, k):
                setattr(m.info, k, v)
        return m

    def model(self) -> Model:
        return getattr(self, "_model", None) or self.build()
