"""File > Capture Picture: Main Window / Current Window / Selected Region.

The model windows are native OpenGL windows: ``QWidget.grab()`` gives black there.  The
picture is the Qt grab of the widgets with each visible drawing painted over it from its
own render window (vtkWindowToImageFilter, back buffer), at its place."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


def vtk_image(vp) -> Optional[QtGui.QImage]:
    """The drawing of a model window as a QImage (physical pixels), or None."""
    try:
        from . import vtk_lite as vtk
        from vtkmodules.util.numpy_support import vtk_to_numpy
        rw = vp.vtk_widget.GetRenderWindow()
        rw.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()
        img = w2i.GetOutput()
        w, h, _ = img.GetDimensions()
        arr = vtk_to_numpy(img.GetPointData().GetScalars())
        if arr is None or w <= 0 or h <= 0:
            return None
        arr = np.ascontiguousarray(arr.reshape(h, w, -1)[::-1, :, :3])  # VTK rows go up
        q = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        return q.copy()  # (own the pixels: arr goes away)
    except Exception:
        return None


def compose(widget: QtWidgets.QWidget, viewports: Iterable) -> QtGui.QPixmap:
    """``widget`` as a picture with the drawings of ``viewports`` in it."""
    pm = widget.grab()
    p = QtGui.QPainter(pm)
    try:
        for vp in viewports:
            vw = vp.vtk_widget
            if not vw.isVisible() or not (widget is vw or widget.isAncestorOf(vw)) or vw.window() is not widget.window():
                continue
            img = vtk_image(vp)
            if img is None:
                continue
            top_left = vw.mapTo(widget, QtCore.QPoint(0, 0))
            p.drawImage(QtCore.QRect(top_left, vw.size()), img)
    finally:
        p.end()
    return pm


class RegionPicker(QtWidgets.QDialog):
    """Selected Region: the picture of the program window, over it; drag a rectangle
    (Esc cancels).  ``region()``: the chosen part, in the picture's own pixels."""

    def __init__(self, picture: QtGui.QPixmap, geometry: QtCore.QRect, parent=None):
        super().__init__(parent, QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        self.picture = picture
        self.setGeometry(geometry)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setModal(True)
        self._a: Optional[QtCore.QPoint] = None
        self._b: Optional[QtCore.QPoint] = None
        self.rect_logical: Optional[QtCore.QRect] = None

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.drawPixmap(self.rect(), self.picture)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 90))
        r = self._rect()
        if r is not None:
            src = self._to_picture(r)
            p.drawPixmap(r, self.picture, src)
            p.setPen(QtGui.QPen(QtGui.QColor(0, 120, 215), 2))
            p.drawRect(r.adjusted(0, 0, -1, -1))
        p.setPen(QtGui.QColor(255, 255, 255))
        p.drawText(self.rect().adjusted(12, 8, -12, -8), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter,
                   "Drag a rectangle round the part to save - Esc = cancel")
        p.end()

    def _rect(self) -> Optional[QtCore.QRect]:
        if self._a is None or self._b is None:
            return None
        return QtCore.QRect(self._a, self._b).normalized()

    def _to_picture(self, r: QtCore.QRect) -> QtCore.QRect:
        sx = self.picture.width() / max(1, self.width())
        sy = self.picture.height() / max(1, self.height())
        return QtCore.QRect(int(r.x() * sx), int(r.y() * sy), max(1, int(r.width() * sx)), max(1, int(r.height() * sy)))

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._a = self._b = ev.position().toPoint()
            self.update()

    def mouseMoveEvent(self, ev):
        if self._a is not None:
            self._b = ev.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self._a is not None:
            self._b = ev.position().toPoint()
            r = self._rect()
            if r is not None and r.width() > 3 and r.height() > 3:
                self.rect_logical = r
                self.accept()
            else:
                self._a = self._b = None
                self.update()

    def region(self) -> Optional[QtGui.QPixmap]:
        if self.rect_logical is None:
            return None
        return self.picture.copy(self._to_picture(self.rect_logical))


def save_picture(parent, pm: QtGui.QPixmap, title: str, name: str) -> Optional[str]:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, title, name, "PNG (*.png);;JPEG (*.jpg)")
    if path:
        pm.save(path)
        if hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(f"Picture saved: {path}", 6000)
    return path or None
