"""Camera navigation for the model windows, following ETABS / SAP2000:

* mouse wheel: zoom in / out at the cursor;
* middle button drag: pan;  Shift + middle (or left) drag: rotate the 3D view;
* right button drag: rotate (3D windows) / pan (plan and elevation windows);
* toolbar modes *Rotate 3D View*, *Pan*, *Zoom Window* (left drag);
* keys: arrows rotate (3D) or pan (2D), Ctrl+arrows pan, +/- zoom, Home = full view.

Rotation is a *turntable* about the global Z axis and a horizontal axis, so the
model never rolls over (Z stays up), unlike a free trackball.  Pure VTK, no Qt.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import vtk

ROTATE_DEG_PER_PIXEL = 0.4
WHEEL_FACTOR = 1.2
MIN_POLAR_DEG = 1.0  # keep the view direction away from straight up / down


def _vec(t) -> np.ndarray:
    return np.array(t, float)


def _rotate(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues rotation of ``v`` about unit ``axis`` by ``ang`` (radians)."""
    axis = axis / (np.linalg.norm(axis) or 1.0)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def orbit(ren: vtk.vtkRenderer, d_azimuth: float, d_elevation: float, pivot: Optional[Sequence[float]] = None):
    """Turntable rotation (degrees): azimuth about global Z, elevation about the
    horizontal screen axis; the camera keeps +Z up.  ``pivot`` defaults to the focal point."""
    cam = ren.GetActiveCamera()
    pos, foc = _vec(cam.GetPosition()), _vec(cam.GetFocalPoint())
    pv = foc if pivot is None else _vec(pivot)
    ez = np.array([0.0, 0.0, 1.0])
    a = math.radians(-d_azimuth)
    pos = pv + _rotate(pos - pv, ez, a)
    foc = pv + _rotate(foc - pv, ez, a)
    view = foc - pos
    right = np.cross(view, ez)
    if np.linalg.norm(right) < 1e-12:  # looking straight down: use the current view-up
        right = np.cross(view, _vec(cam.GetViewUp()))
    right /= np.linalg.norm(right) or 1.0
    # clamp the polar angle of the viewing direction
    polar = math.degrees(math.acos(max(-1.0, min(1.0, -view[2] / (np.linalg.norm(view) or 1.0)))))
    d_el = d_elevation
    new_polar = polar - d_el
    if new_polar < MIN_POLAR_DEG:
        d_el = polar - MIN_POLAR_DEG
    elif new_polar > 180.0 - MIN_POLAR_DEG:
        d_el = polar - (180.0 - MIN_POLAR_DEG)
    e = math.radians(-d_el)
    pos = pv + _rotate(pos - pv, right, e)
    foc = pv + _rotate(foc - pv, right, e)
    cam.SetPosition(*pos)
    cam.SetFocalPoint(*foc)
    cam.SetViewUp(0.0, 0.0, 1.0)
    cam.OrthogonalizeViewUp()
    ren.ResetCameraClippingRange()


def _focal_depth(ren: vtk.vtkRenderer) -> float:
    cam = ren.GetActiveCamera()
    ren.SetWorldPoint(*cam.GetFocalPoint(), 1.0)
    ren.WorldToDisplay()
    return ren.GetDisplayPoint()[2]


def display_to_world(ren: vtk.vtkRenderer, x: float, y: float, depth: Optional[float] = None) -> np.ndarray:
    """World point under display (x, y) at the focal-plane depth (or ``depth``)."""
    z = _focal_depth(ren) if depth is None else depth
    ren.SetDisplayPoint(x, y, z)
    ren.DisplayToWorld()
    w = ren.GetWorldPoint()
    return np.array(w[:3], float) / (w[3] or 1.0)


def pan(ren: vtk.vtkRenderer, x0: float, y0: float, x1: float, y1: float):
    """Move the view so the point under (x0, y0) follows the cursor to (x1, y1)."""
    cam = ren.GetActiveCamera()
    z = _focal_depth(ren)
    d = display_to_world(ren, x0, y0, z) - display_to_world(ren, x1, y1, z)
    cam.SetPosition(*(_vec(cam.GetPosition()) + d))
    cam.SetFocalPoint(*(_vec(cam.GetFocalPoint()) + d))
    ren.ResetCameraClippingRange()


def zoom_at(ren: vtk.vtkRenderer, x: float, y: float, factor: float):
    """Zoom by ``factor`` (>1 = in) keeping the point under the cursor fixed."""
    cam = ren.GetActiveCamera()
    w = display_to_world(ren, x, y)
    pos, foc = _vec(cam.GetPosition()), _vec(cam.GetFocalPoint())
    if cam.GetParallelProjection():
        cam.SetParallelScale(cam.GetParallelScale() / factor)
        shift = (w - foc) * (1.0 - 1.0 / factor)
        cam.SetFocalPoint(*(foc + shift))
        cam.SetPosition(*(pos + shift))
    else:
        cam.SetFocalPoint(*(w + (foc - w) / factor))
        cam.SetPosition(*(w + (pos - w) / factor))
    ren.ResetCameraClippingRange()


def zoom_window(ren: vtk.vtkRenderer, x0: float, y0: float, x1: float, y1: float, width: int, height: int):
    """Rubber-band zoom: the rectangle (display pixels) fills the window."""
    w, h = abs(x1 - x0), abs(y1 - y0)
    if w < 4 or h < 4:
        return
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    cam = ren.GetActiveCamera()
    centre = display_to_world(ren, cx, cy)
    foc, pos = _vec(cam.GetFocalPoint()), _vec(cam.GetPosition())
    shift = centre - foc
    cam.SetFocalPoint(*(foc + shift))
    cam.SetPosition(*(pos + shift))
    factor = min(width / w, height / h)
    if cam.GetParallelProjection():
        cam.SetParallelScale(cam.GetParallelScale() / factor)
    else:
        cam.Dolly(factor)
    ren.ResetCameraClippingRange()


def default_3d(ren: vtk.vtkRenderer):
    """ETABS "Set Default 3D View": looking from the -X, -Y side, slightly above."""
    from . import scene

    scene.set_view(ren, "3d")
