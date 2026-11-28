"""The picture of a drawing in the forms of the References menu.

Before a drawing is placed the user sees it: its lines in their colours, the layers
ticked (the picture follows every tick), the sheet of a PDF page, and how large it is
with the drawing unit / plot scale chosen - the overall width and height are written on
the picture.  Two clicks on the picture measure a length (the clicks snap to the ends of
lines), which is how a wrong unit or scale is seen at once and put right ("this bay is
really 6 m").

The wheel zooms about the cursor, dragging (any button) moves the picture, a double
click fits it again; a click without dragging picks a point to measure from.  The texts
of the drawing are drawn too (so a dimension can be read and measured against).

The lines are drawn into an image with numpy (:func:`raster_lines`) - a drawing of some
hundred thousand lines is redrawn in a few hundredths of a second, where one painter
call per line would take seconds.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..io.dxf_read import Drawing

MAX_SAMPLES = 6_000_000  # pixels written for one picture


def raster_lines(P: np.ndarray, rgb: np.ndarray, width: int, height: int, background=(255, 255, 255)) -> np.ndarray:
    """Draw segments ``P`` (N, 4: x1, y1, x2, y2 in pixels, y down) with colours ``rgb``
    (N, 3 uint8, or one colour) into an image (height, width, 3) uint8.  One pixel wide,
    clipped to the image."""
    img = np.empty((max(int(height), 1), max(int(width), 1), 3), np.uint8)
    img[:] = np.asarray(background, np.uint8)
    P = np.asarray(P, float).reshape(-1, 4)
    if not len(P):
        return img
    one = np.ndim(rgb) == 1
    w, h = img.shape[1], img.shape[0]
    x1, y1, x2, y2 = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    keep = np.isfinite(P).all(axis=1) & (np.maximum(x1, x2) >= 0) & (np.minimum(x1, x2) <= w - 1) & \
        (np.maximum(y1, y2) >= 0) & (np.minimum(y1, y2) <= h - 1)
    if not keep.any():
        return img
    P = P[keep]
    col = None if one else np.asarray(rgb, np.uint8)[keep]
    x1, y1 = P[:, 0], P[:, 1]
    dx, dy = P[:, 2] - x1, P[:, 3] - y1
    # the part of each segment inside the image (Liang - Barsky): a long line seen from
    # close up is sampled only where it shows
    t0 = np.zeros(len(P))
    t1 = np.ones(len(P))
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-dx, x1), (dx, (w - 1) - x1), (-dy, y1), (dy, (h - 1) - y1)):
            r = q / p
            neg, pos = p < 0, p > 0
            t0 = np.where(neg, np.maximum(t0, r), t0)
            t1 = np.where(pos, np.minimum(t1, r), t1)
            t1 = np.where((p == 0) & (q < 0), -1.0, t1)
    ok = t1 >= t0
    if not ok.any():
        return img
    x1, y1, dx, dy, t0, t1 = x1[ok], y1[ok], dx[ok], dy[ok], t0[ok], t1[ok]
    if col is not None:
        col = col[ok]
    n = np.ceil(np.maximum(np.abs(dx), np.abs(dy)) * (t1 - t0)).astype(np.int64) + 1
    total = int(n.sum())
    if total > MAX_SAMPLES:  # (too much for one picture: every line thinner, none left out)
        n = np.maximum((n * (MAX_SAMPLES / total)).astype(np.int64), 1)
        total = int(n.sum())
    idx = np.repeat(np.arange(len(n)), n)
    first = np.cumsum(n) - n
    k = np.arange(total) - first[idx]
    span = np.maximum(n - 1, 1)[idx]
    t = t0[idx] + (t1 - t0)[idx] * (k / span)
    xs = np.rint(x1[idx] + t * dx[idx]).astype(np.int64)
    ys = np.rint(y1[idx] + t * dy[idx]).astype(np.int64)
    good = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if one:
        img[ys[good], xs[good]] = np.asarray(rgb, np.uint8)
    else:
        img[ys[good], xs[good]] = col[idx[good]]
    return img


def _ink_colours(rgb: np.ndarray, background, ink) -> np.ndarray:
    """The drawing's colours as they can be seen on ``background``: those too near it
    (white lines on a white form, black on a dark one) in ``ink``."""
    c = rgb.astype(float)
    lum = c @ np.array([0.299, 0.587, 0.114])
    bg = float(np.asarray(background, float) @ np.array([0.299, 0.587, 0.114]))
    out = rgb.copy()
    out[np.abs(lum - bg) < 70.0] = np.asarray(ink, np.uint8)
    if bg > 128:  # a light form: the bright CAD colours (yellow, cyan) darker
        bright = (np.abs(lum - bg) >= 70.0) & (lum > 160)
        out[bright] = (c[bright] * 0.62).astype(np.uint8)
    return out


def _length_text(metres: float) -> str:
    v = abs(metres)
    if v >= 1000:
        return f"{metres / 1000:,.3f} km"
    if v >= 1:
        return f"{metres:,.3f} m"
    if v >= 0.01:
        return f"{metres * 100:,.2f} cm"
    return f"{metres * 1000:,.2f} mm"


class DrawingPreview(QtWidgets.QWidget):
    """The picture of a :class:`Drawing` (see the module text)."""

    measured = QtCore.Signal(float)  # the distance of the two points clicked, in file units
    SNAP_PX = 9.0

    def __init__(self, parent=None, measuring: bool = True):
        super().__init__(parent)
        self.dwg: Optional[Drawing] = None
        self._off: set = set()
        self._metres = 1.0  # metres per file unit (the drawing unit x scale) - for the texts
        self._measuring = bool(measuring)
        self._image: Optional[QtGui.QImage] = None
        self._buf = None
        self._scale = 1.0  # pixels per file unit
        self._centre = (0.0, 0.0)  # the file point in the middle of the widget
        self._fitted = True
        self._drag = None
        self._press = None  # a left press that may become a click (a pick) or a drag (a move)
        self._pick: list = []  # file points clicked (0, 1 or 2)
        self._hover: Optional[Tuple[float, float]] = None
        self._shown = 0
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setCursor(QtCore.Qt.CrossCursor if measuring else QtCore.Qt.OpenHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    # ------------------------------------------------------------- what is shown
    def set_drawing(self, dwg: Optional[Drawing], layers_off: Sequence[str] = (), metres_per_unit: float = 1.0):
        self.dwg = dwg
        self._off = set(layers_off)
        self._metres = float(metres_per_unit) if metres_per_unit and metres_per_unit > 0 else 1.0
        self._pick, self._hover = [], None
        self._mask = None
        self.fit()

    def set_layers_off(self, layers_off: Sequence[str]):
        if set(layers_off) != self._off:
            self._off = set(layers_off)
            self._mask = None
            self._redraw()

    def set_metres_per_unit(self, metres: float):
        metres = float(metres) if metres and metres > 0 else 1.0
        if metres != self._metres:
            self._metres = metres
            self.update()

    def measured_length(self) -> Optional[float]:
        """The distance of the two points clicked (file units); None: not two yet."""
        if len(self._pick) < 2:
            return None
        return math.dist(self._pick[0], self._pick[1])

    def clear_measure(self):
        self._pick = []
        self.update()

    def shown_segments(self) -> int:
        return self._shown

    # ------------------------------------------------------------- the view
    def _visible(self) -> np.ndarray:
        d = self.dwg
        m = getattr(self, "_mask", None)
        if m is None:
            on = np.array([n not in self._off for n in d.layers], bool)
            m = on[d.seg_layer] if len(d.seg) else np.zeros(0, bool)
            self._mask = m
        return m

    def _box(self) -> Optional[Tuple[float, float, float, float]]:
        """What Fit shows: the sheet of a PDF page, else the lines shown (all of them when
        every layer is off)."""
        d = self.dwg
        if d is None:
            return None
        if d.sheet:
            return (0.0, 0.0, float(d.sheet[0]), float(d.sheet[1]))
        if not len(d.seg):
            return None
        s = d.seg[self._visible()]
        if not len(s):
            s = d.seg
        return (float(min(s[:, 0].min(), s[:, 2].min())), float(min(s[:, 1].min(), s[:, 3].min())),
                float(max(s[:, 0].max(), s[:, 2].max())), float(max(s[:, 1].max(), s[:, 3].max())))

    def fit(self):
        b = self._box()
        self._fitted = True
        if b is not None:
            w, h = max(self.width() - 56, 20), max(self.height() - 56, 20)
            bw, bh = max(b[2] - b[0], 1e-12), max(b[3] - b[1], 1e-12)
            self._scale = min(w / bw, h / bh)
            self._centre = (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))
        self._redraw()

    def to_widget(self, xy) -> Tuple[float, float]:
        return (self.width() * 0.5 + (xy[0] - self._centre[0]) * self._scale,
                self.height() * 0.5 - (xy[1] - self._centre[1]) * self._scale)

    def to_file(self, px, py) -> Tuple[float, float]:
        return (self._centre[0] + (px - self.width() * 0.5) / self._scale,
                self._centre[1] - (py - self.height() * 0.5) / self._scale)

    def _colours(self):
        pal = self.palette()
        bg = pal.color(QtGui.QPalette.Base)
        ink = pal.color(QtGui.QPalette.Text)
        return (bg.red(), bg.green(), bg.blue()), (ink.red(), ink.green(), ink.blue())

    def _redraw(self):
        d = self.dwg
        w, h = max(self.width(), 1), max(self.height(), 1)
        bg, ink = self._colours()
        if d is None or not len(d.seg):
            self._image, self._shown = None, 0
            self.update()
            return
        from . import cadref
        self._shown = int(self._visible().sum())
        s, colours = cadref.shown_lines(None, d, self._off, texts=False)  # (the texts: written in paintEvent)
        cx, cy = self._centre
        k = self._scale
        P = np.empty((len(s), 4))
        P[:, 0] = w * 0.5 + (s[:, 0] - cx) * k
        P[:, 1] = h * 0.5 - (s[:, 1] - cy) * k
        P[:, 2] = w * 0.5 + (s[:, 2] - cx) * k
        P[:, 3] = h * 0.5 - (s[:, 3] - cy) * k
        img = raster_lines(P, _ink_colours(colours, bg, ink) if len(s) else np.zeros((0, 3), np.uint8), w, h, bg)
        self._buf = np.ascontiguousarray(img)
        self._image = QtGui.QImage(self._buf.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.update()

    # ------------------------------------------------------------- snapping in the picture
    def _snap(self, px: float, py: float) -> Tuple[float, float]:
        """The end of a shown line within a few pixels of the cursor, else the point under it."""
        d = self.dwg
        q = self.to_file(px, py)
        if d is None or not len(d.seg):
            return q
        s = d.seg[self._visible()]
        if not len(s):
            return q
        r = self.SNAP_PX / self._scale
        best, bd = None, r
        for a in (s[:, 0:2], s[:, 2:4]):
            near = np.flatnonzero((np.abs(a[:, 0] - q[0]) <= r) & (np.abs(a[:, 1] - q[1]) <= r))
            if len(near):
                dist = np.hypot(a[near, 0] - q[0], a[near, 1] - q[1])
                j = int(np.argmin(dist))
                if dist[j] <= bd:
                    best, bd = (float(a[near[j], 0]), float(a[near[j], 1])), float(dist[j])
        return best if best is not None else q

    # ------------------------------------------------------------- events
    def resizeEvent(self, ev):
        if self._fitted:
            self.fit()
        else:
            self._redraw()

    def wheelEvent(self, ev):
        if self.dwg is None:
            return
        steps = ev.angleDelta().y() / 120.0
        if not steps:
            return
        pos = ev.position()
        before = self.to_file(pos.x(), pos.y())
        self._scale = min(max(self._scale * (1.25 ** steps), 1e-12), 1e12)
        after = self.to_file(pos.x(), pos.y())
        self._centre = (self._centre[0] + before[0] - after[0], self._centre[1] + before[1] - after[1])
        self._fitted = False
        self._redraw()
        ev.accept()

    def mousePressEvent(self, ev):
        pos = ev.position()
        pan = ev.button() in (QtCore.Qt.MiddleButton, QtCore.Qt.RightButton) or \
            (ev.button() == QtCore.Qt.LeftButton and not self._measuring)
        if pan:
            self._drag = (pos.x(), pos.y(), self._centre)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        elif ev.button() == QtCore.Qt.LeftButton:
            self._press = (pos.x(), pos.y())

    def _pick_at(self, px: float, py: float):
        if self.dwg is None:
            return
        p = self._snap(px, py)
        if len(self._pick) >= 2:
            self._pick = []
        self._pick.append(p)
        if len(self._pick) == 2:
            if math.dist(*self._pick) <= 0:
                self._pick.pop()
            else:
                self.measured.emit(math.dist(*self._pick))
        self.update()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._press is not None and self._drag is None and \
                abs(pos.x() - self._press[0]) + abs(pos.y() - self._press[1]) > 5:
            # the left button dragged: the picture moves (a click without dragging picks)
            self._drag = (self._press[0], self._press[1], self._centre)
            self._press = None
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        if self._drag is not None:
            x0, y0, c = self._drag
            self._centre = (c[0] - (pos.x() - x0) / self._scale, c[1] + (pos.y() - y0) / self._scale)
            self._fitted = False
            self._redraw()
        elif self._measuring and self.dwg is not None:
            self._hover = self._snap(pos.x(), pos.y())
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._press is not None and ev.button() == QtCore.Qt.LeftButton:
            x, y = self._press
            self._press = None
            self._pick_at(x, y)
            return
        if self._drag is not None:
            self._drag = None
            self.setCursor(QtCore.Qt.CrossCursor if self._measuring else QtCore.Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, ev):
        self._pick, self._press = [], None
        self.fit()

    def leaveEvent(self, ev):
        self._hover = None
        self.update()

    # ------------------------------------------------------------- painting
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        pal = self.palette()
        bg, ink = pal.color(QtGui.QPalette.Base), pal.color(QtGui.QPalette.Text)
        if self._image is not None:
            p.drawImage(0, 0, self._image)
        else:
            p.fillRect(self.rect(), bg)
        d = self.dwg
        if d is None:
            p.setPen(ink)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "No drawing")
            p.end()
            return
        self._paint_texts(p, bg, ink)
        soft = QtGui.QColor(ink)
        soft.setAlpha(110)
        accent = pal.color(QtGui.QPalette.Highlight)
        f = p.font()
        f.setPointSizeF(max(f.pointSizeF() - 0.5, 7.0))
        p.setFont(f)
        if d.sheet:  # the sheet of a PDF page
            a, b = self.to_widget((0.0, 0.0)), self.to_widget(d.sheet)
            p.setPen(QtGui.QPen(soft, 1, QtCore.Qt.DashLine))
            p.drawRect(QtCore.QRectF(QtCore.QPointF(a[0], b[1]), QtCore.QPointF(b[0], a[1])))
        e = self._extent()
        if e is not None:  # the overall size, written under and beside the lines
            (x0, y0), (x1, y1) = self.to_widget((e[0], e[1])), self.to_widget((e[2], e[3]))
            p.setPen(QtGui.QPen(accent, 1))
            yb = min(y0 + 14, self.height() - 16)
            xl = max(x0 - 14, 16)
            p.drawLine(QtCore.QPointF(x0, yb), QtCore.QPointF(x1, yb))
            p.drawLine(QtCore.QPointF(xl, y0), QtCore.QPointF(xl, y1))
            for x in (x0, x1):
                p.drawLine(QtCore.QPointF(x, yb - 4), QtCore.QPointF(x, yb + 4))
            for y in (y0, y1):
                p.drawLine(QtCore.QPointF(xl - 4, y), QtCore.QPointF(xl + 4, y))
            self._label(p, (0.5 * (x0 + x1), yb + 10), _length_text((e[2] - e[0]) * self._metres), accent, bg)
            self._label(p, (xl, 0.5 * (y0 + y1)), _length_text((e[3] - e[1]) * self._metres), accent, bg, upright=True)
        # the length being measured
        pts = list(self._pick)
        if len(pts) == 1 and self._hover is not None:
            pts.append(self._hover)
        red = QtGui.QColor(214, 40, 40)
        if self._measuring and self._hover is not None and len(self._pick) != 1:
            hx, hy = self.to_widget(self._hover)
            p.setPen(QtGui.QPen(red, 1))
            p.drawRect(QtCore.QRectF(hx - 4, hy - 4, 8, 8))
        if pts:
            p.setPen(QtGui.QPen(red, 2))
            w0 = self.to_widget(pts[0])
            p.drawEllipse(QtCore.QPointF(*w0), 3.5, 3.5)
            if len(pts) == 2:
                w1 = self.to_widget(pts[1])
                p.drawLine(QtCore.QPointF(*w0), QtCore.QPointF(*w1))
                p.drawEllipse(QtCore.QPointF(*w1), 3.5, 3.5)
                self._label(p, (0.5 * (w0[0] + w1[0]), 0.5 * (w0[1] + w1[1]) - 12),
                            _length_text(math.dist(pts[0], pts[1]) * self._metres), red, bg)
        p.setPen(soft)
        hint = "wheel: zoom   drag: move   double click: fit"
        if self._measuring:
            hint = "click two points to measure   " + hint
        p.drawText(QtCore.QRectF(6, self.height() - 18, self.width() - 12, 16),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, hint)
        p.end()

    MIN_TEXT_PX = 5.0  # smaller letters cannot be read: not written
    MAX_TEXTS_DRAWN = 4000

    def _paint_texts(self, p: QtGui.QPainter, bg, ink):
        """The texts of the layers shown, written as text (readable at any zoom, where the
        letters are large enough to read)."""
        d = self.dwg
        texts = getattr(d, "texts", None)
        if not texts:
            return
        k = self._scale
        w, h = self.width(), self.height()
        on = [n not in self._off for n in d.layers]
        back = (bg.red(), bg.green(), bg.blue())
        dark = (ink.red(), ink.green(), ink.blue())
        font = QtGui.QFont(p.font())
        font.setFamilies(["Tahoma", "Segoe UI", "Arial", "DejaVu Sans"])
        drawn = 0
        p.save()
        p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        for (x, y, height, ang, string, lay, rgb, across, up, width, wf) in texts:
            px = height * k / 0.72  # the size of the font for capitals ``height`` high
            if px < self.MIN_TEXT_PX / 0.72 or not on[lay]:
                continue
            wx, wy = w * 0.5 + (x - self._centre[0]) * k, h * 0.5 - (y - self._centre[1]) * k
            reach = px * (len(string) + 2)
            if wx < -reach or wx > w + reach or wy < -reach or wy > h + reach:
                continue
            font.setPixelSize(max(int(round(px)), 1))
            fm = QtGui.QFontMetricsF(font)
            adv = fm.horizontalAdvance(string) or 1.0
            sx = (width * k / adv) if width > 0 else (wf or 1.0)
            c = _ink_colours(np.array([rgb], np.uint8), back, dark)[0]
            p.save()
            p.translate(wx, wy)
            p.rotate(-ang)
            p.scale(sx, 1.0)
            p.setFont(font)
            p.setPen(QtGui.QColor(int(c[0]), int(c[1]), int(c[2])))
            cap = height * k
            dy = {0: 0.0, 1: -0.22 * cap, 2: 0.5 * cap, 3: cap}.get(up, 0.0)
            p.drawText(QtCore.QPointF(-0.5 * across * adv, dy), string)
            p.restore()
            drawn += 1
            if drawn >= self.MAX_TEXTS_DRAWN:
                break
        p.restore()
        self._texts_drawn = drawn

    def _extent(self) -> Optional[Tuple[float, float, float, float]]:
        d = self.dwg
        if d is None or not len(d.seg):
            return None
        s = d.seg[self._visible()]
        if not len(s):
            return None
        return (float(min(s[:, 0].min(), s[:, 2].min())), float(min(s[:, 1].min(), s[:, 3].min())),
                float(max(s[:, 0].max(), s[:, 2].max())), float(max(s[:, 1].max(), s[:, 3].max())))

    @staticmethod
    def _label(p: QtGui.QPainter, at, text: str, colour, background, upright: bool = False):
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 8, fm.height() + 2
        p.save()
        p.translate(at[0], at[1])
        if upright:
            p.rotate(-90)
        r = QtCore.QRectF(-w * 0.5, -h * 0.5, w, h)
        back = QtGui.QColor(background)
        back.setAlpha(225)
        p.fillRect(r, back)
        p.setPen(colour)
        p.drawText(r, QtCore.Qt.AlignCenter, text)
        p.restore()


def render(dwg: Drawing, width: int = 1400, height: int = 1000, layers_off: Sequence[str] = (),
           background=(255, 255, 255), ink=(20, 20, 20)) -> np.ndarray:
    """The picture of a drawing as an image array (for checks and tests; no widget)."""
    from . import cadref
    s, colours = cadref.shown_lines(None, dwg, layers_off)
    if dwg.sheet:
        b = (0.0, 0.0, float(dwg.sheet[0]), float(dwg.sheet[1]))
    elif len(s):
        b = (s[:, (0, 2)].min(), s[:, (1, 3)].min(), s[:, (0, 2)].max(), s[:, (1, 3)].max())
    else:
        return raster_lines(np.zeros((0, 4)), np.zeros((0, 3), np.uint8), width, height, background)
    k = min((width - 20) / max(b[2] - b[0], 1e-12), (height - 20) / max(b[3] - b[1], 1e-12))
    cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
    P = np.column_stack([width * 0.5 + (s[:, 0] - cx) * k, height * 0.5 - (s[:, 1] - cy) * k,
                         width * 0.5 + (s[:, 2] - cx) * k, height * 0.5 - (s[:, 3] - cy) * k])
    return raster_lines(P, _ink_colours(colours, background, ink), width, height, background)
