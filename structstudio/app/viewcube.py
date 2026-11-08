"""CAD-style ViewCube for the 3D model windows (AutoCAD / Revit / ArchiCAD).

A small cube drawn in the top-right corner of a 3D window, turning with the
camera.  Its 26 pieces are clickable:

* the 6 faces (TOP, BOTTOM, FRONT, BACK, LEFT, RIGHT) give the orthographic
  directions - FRONT looks along +Y (the X-Z elevation), RIGHT along -X;
* the 12 edges give the 45-degree views between two faces;
* the 8 corners give the isometric views.

The piece under the mouse is highlighted; a click turns the camera smoothly
(about the focal point, same zoom) to that direction; dragging on the cube
orbits the view.  Pure VTK: the cube is a second renderer on layer 1 of the
window's render window, so it never mixes with the model's depth buffer.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import vtk

Key = Tuple[int, int, int]

FACE_LABELS = {(0, 0, 1): "TOP", (0, 0, -1): "BOTTOM", (0, -1, 0): "FRONT", (0, 1, 0): "BACK",
               (1, 0, 0): "RIGHT", (-1, 0, 0): "LEFT"}
SIZE_PX = 118  # cube area (pixels, square) at the top-right corner
_INNER = 0.62  # half size of a face piece; edges / corners take the rest of the [-1, 1] cube


def view_for(key: Key) -> Tuple[np.ndarray, np.ndarray]:
    """(direction from the focal point to the camera, view-up) of a cube piece."""
    d = np.array(key, float)
    d /= np.linalg.norm(d)
    if key[0] == 0 and key[1] == 0:  # top / bottom: north (+Y) up on the screen
        up = np.array([0.0, 1.0, 0.0]) if key[2] > 0 else np.array([0.0, -1.0, 0.0])
    else:
        up = np.array([0.0, 0.0, 1.0])
        up = up - d * float(np.dot(up, d))
        up /= np.linalg.norm(up) or 1.0
    return d, up


def describe(key: Key) -> str:
    if key in FACE_LABELS:
        return FACE_LABELS[key].title()
    names = []
    for axis, (neg, pos) in enumerate((("Left", "Right"), ("Front", "Back"), ("Bottom", "Top"))):
        if key[axis]:
            names.append(pos if key[axis] > 0 else neg)
    order = {"Top": 0, "Bottom": 0, "Front": 1, "Back": 1, "Left": 2, "Right": 2}
    names.sort(key=lambda n: order[n])
    return " ".join(names) + (" (isometric)" if len(names) == 3 else "")


def _range(v: int) -> Tuple[float, float]:
    return (-1.0, -_INNER) if v < 0 else (_INNER, 1.0) if v > 0 else (-_INNER, _INNER)


class ViewCube:
    """The cube of one model window.  ``on_pick(key)`` is called after a click."""

    def __init__(self, render_window: vtk.vtkRenderWindow, main: vtk.vtkRenderer,
                 on_pick: Optional[Callable[[Key], None]] = None):
        self.rw = render_window
        self.main = main
        self.on_pick = on_pick
        self.ren = vtk.vtkRenderer()
        self.ren.SetLayer(1)
        self.ren.InteractiveOff()
        if self.rw.GetNumberOfLayers() < 2:
            self.rw.SetNumberOfLayers(2)
        self.rw.AddRenderer(self.ren)
        cam = self.ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetParallelScale(2.15)
        self.pieces: Dict[Key, vtk.vtkActor] = {}
        self.labels = []
        self._hover: Optional[Key] = None
        self.colors = {"face": (0.86, 0.89, 0.93), "edge": (0.78, 0.82, 0.87), "hover": (0.30, 0.62, 1.00),
                       "text": (0.18, 0.22, 0.28), "outline": (0.40, 0.45, 0.52)}
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if (i, j, k) == (0, 0, 0):
                        continue
                    self._add_piece((i, j, k))
        for key, text in FACE_LABELS.items():
            self._add_label(key, text)
        outline = vtk.vtkOutlineSource()
        outline.SetBounds(-1, 1, -1, 1, -1, 1)
        om = vtk.vtkPolyDataMapper()
        om.SetInputConnection(outline.GetOutputPort())
        self.outline = vtk.vtkActor()
        self.outline.SetMapper(om)
        self.outline.GetProperty().SetLineWidth(1.5)
        self.outline.GetProperty().LightingOff()
        self.outline.PickableOff()
        self.ren.AddActor(self.outline)
        self.picker = vtk.vtkPropPicker()
        self.visible = True
        self.apply_colors()

    # ------------------------------------------------------------ build
    def _add_piece(self, key: Key):
        (x0, x1), (y0, y1), (z0, z1) = (_range(v) for v in key)
        src = vtk.vtkCubeSource()
        src.SetBounds(x0, x1, y0, y1, z0, z1)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(src.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetAmbient(0.55)
        a.GetProperty().SetDiffuse(0.55)
        a.GetProperty().SetSpecular(0.0)
        self.ren.AddActor(a)
        self.pieces[key] = a

    def _add_label(self, key: Key, text: str):
        vt = vtk.vtkVectorText()
        vt.SetText(text)
        vt.Update()
        b = vt.GetOutput().GetBounds()
        w, h = b[1] - b[0], b[3] - b[2]
        s = min(1.05 / max(w, 1e-6), 0.42 / max(h, 1e-6))
        tf = vtk.vtkTransform()
        n = np.array(key, float)
        # as seen from outside the face: screen-up of that view, right = view dir x up
        _, upv = view_for(key)
        right = np.cross(-n, upv)
        normal = np.cross(right, upv)
        mat = vtk.vtkMatrix4x4()
        origin = n * 1.004
        for r in range(3):
            mat.SetElement(r, 0, right[r] * s)
            mat.SetElement(r, 1, upv[r] * s)
            mat.SetElement(r, 2, normal[r] * s)
            mat.SetElement(r, 3, origin[r] - (right[r] * (b[0] + w / 2) + upv[r] * (b[2] + h / 2)) * s)
        tf.SetMatrix(mat)
        tp = vtk.vtkTransformPolyDataFilter()
        tp.SetTransform(tf)
        tp.SetInputConnection(vt.GetOutputPort())
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(tp.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().LightingOff()
        a.PickableOff()
        self.ren.AddActor(a)
        self.labels.append(a)

    def apply_colors(self, dark: Optional[bool] = None):
        if dark is not None:
            if dark:
                self.colors.update(face=(0.30, 0.33, 0.38), edge=(0.25, 0.28, 0.33), text=(0.92, 0.94, 0.96),
                                   outline=(0.62, 0.66, 0.72), hover=(0.30, 0.62, 1.00))
            else:
                self.colors.update(face=(0.86, 0.89, 0.93), edge=(0.78, 0.82, 0.87), text=(0.18, 0.22, 0.28),
                                   outline=(0.40, 0.45, 0.52), hover=(0.25, 0.55, 0.95))
        for key, a in self.pieces.items():
            kind = "face" if sum(1 for v in key if v) == 1 else "edge"
            a.GetProperty().SetColor(*self.colors["hover" if key == self._hover else kind])
        for a in self.labels:
            a.GetProperty().SetColor(*self.colors["text"])
        self.outline.GetProperty().SetColor(*self.colors["outline"])

    # ----------------------------------------------------------- layout
    def layout(self):
        w, h = self.rw.GetSize()
        if w <= 0 or h <= 0:
            return
        s = min(SIZE_PX * self._dpr(), 0.4 * min(w, h))
        self.ren.SetViewport(1.0 - s / w, 1.0 - s / h, 1.0, 1.0)

    def _dpr(self) -> float:
        return 1.0

    def set_visible(self, on: bool):
        self.visible = bool(on)
        self.ren.SetDraw(1 if on else 0)
        for a in list(self.pieces.values()) + self.labels + [self.outline]:
            a.SetVisibility(1 if on else 0)

    def sync(self):
        """Turn the cube like the model camera (call before every render)."""
        if not self.visible:
            return
        self.layout()
        mc = self.main.GetActiveCamera()
        dop = np.array(mc.GetDirectionOfProjection(), float)
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(*(-dop * 10.0))
        cam.SetViewUp(*mc.GetViewUp())
        self.ren.ResetCameraClippingRange()

    # ---------------------------------------------------------- picking
    def contains(self, x: float, y: float) -> bool:
        if not self.visible:
            return False
        w, h = self.rw.GetSize()
        vx0, vy0, vx1, vy1 = self.ren.GetViewport()
        return vx0 * w <= x <= vx1 * w and vy0 * h <= y <= vy1 * h

    def piece_at(self, x: float, y: float) -> Optional[Key]:
        if not self.contains(x, y):
            return None
        self.sync()
        if not self.picker.Pick(x, y, 0, self.ren):
            return None
        actor = self.picker.GetActor()
        for key, a in self.pieces.items():
            if a is actor:
                return key
        return None

    def hover(self, x: float, y: float) -> bool:
        """Highlight the piece under (x, y); True when the highlight changed."""
        key = self.piece_at(x, y)
        if key == self._hover:
            return False
        self._hover = key
        self.apply_colors()
        return True

    def click(self, x: float, y: float) -> Optional[Key]:
        key = self.piece_at(x, y)
        if key is not None and self.on_pick is not None:
            self.on_pick(key)
        return key


class CameraAnimator:
    """Smoothly turns a renderer's camera about its focal point to a new
    direction / view-up (a few frames on a Qt timer; immediate without Qt)."""

    def __init__(self, renderer: vtk.vtkRenderer, render: Callable[[], None], steps: int = 14,
                 interval_ms: int = 15):
        self.ren = renderer
        self.render = render
        self.steps = steps
        self.interval = interval_ms
        self._timer = None
        self._frames = []

    def go(self, direction, up, animate: bool = True):
        cam = self.ren.GetActiveCamera()
        foc = np.array(cam.GetFocalPoint(), float)
        dist = float(np.linalg.norm(np.array(cam.GetPosition(), float) - foc)) or 1.0
        d0 = np.array(cam.GetPosition(), float) - foc
        d0 /= np.linalg.norm(d0) or 1.0
        u0 = np.array(cam.GetViewUp(), float)
        d1 = np.asarray(direction, float) / (np.linalg.norm(direction) or 1.0)
        u1 = np.asarray(up, float) / (np.linalg.norm(up) or 1.0)
        n = self.steps if animate else 1
        frames = []
        for k in range(1, n + 1):
            t = k / n
            t = t * t * (3 - 2 * t)  # ease in / out
            d = _slerp(d0, d1, t)
            u = u0 * (1 - t) + u1 * t
            u = u - d * float(np.dot(u, d))
            if np.linalg.norm(u) < 1e-6:
                u = u1
            frames.append((foc + d * dist, u / (np.linalg.norm(u) or 1.0)))
        self._frames = frames
        try:
            from PySide6 import QtCore
        except ImportError:  # pragma: no cover
            QtCore = None
        if not animate or QtCore is None or QtCore.QCoreApplication.instance() is None:
            self._apply(frames[-1], foc)
            return
        if self._timer is None:
            self._timer = QtCore.QTimer()
            self._timer.timeout.connect(self._tick)
        self._foc = foc
        self._timer.start(self.interval)

    def _apply(self, frame, foc):
        cam = self.ren.GetActiveCamera()
        pos, up = frame
        cam.SetFocalPoint(*foc)
        cam.SetPosition(*pos)
        cam.SetViewUp(*up)
        cam.OrthogonalizeViewUp()
        self.ren.ResetCameraClippingRange()
        self.render()

    def _tick(self):
        if not self._frames:
            self._timer.stop()
            return
        self._apply(self._frames.pop(0), self._foc)
        if not self._frames:
            self._timer.stop()

    @property
    def running(self) -> bool:
        return bool(self._frames)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = max(-1.0, min(1.0, float(np.dot(a, b))))
    if dot > 0.9995:
        v = a + (b - a) * t
        return v / (np.linalg.norm(v) or 1.0)
    if dot < -0.9995:  # opposite: turn about any perpendicular axis
        axis = np.cross(a, [0.0, 0.0, 1.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        ang = math.pi * t
        return a * math.cos(ang) + np.cross(axis, a) * math.sin(ang)
    om = math.acos(dot)
    so = math.sin(om)
    return (math.sin((1 - t) * om) / so) * a + (math.sin(t * om) / so) * b
