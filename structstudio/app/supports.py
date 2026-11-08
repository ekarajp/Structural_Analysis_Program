"""Joint restraint types with pictures and 3D symbols (ETABS / SAP2000 / CYPE 3D).

A restraint is six booleans (UX UY UZ RX RY RZ) in the joint axes.  The common
cases get a name, a picture (for the *Assign Joint Restraints* form) and a
distinct 3D symbol in the model window so a roller can be told from a pin or
a fixed support at a glance:

===========  ===================================  =============================
type         restraints                           symbol
===========  ===================================  =============================
fixed        all six                              flat box (plate) under the joint
pinned       UX UY UZ                             cone (pin)
roller_z     UZ only (rolls in the plane)         cone on two rollers
roller_x     UX only                              cone on rollers, normal along X
roller_y     UY only                              cone on rollers, normal along Y
slider_xy    UX UY (vertical sliding)             cone sideways on rollers
guided       UZ RX RY (fixed against rotation, slides horizontally)  box on rollers
free         none                                 no symbol
custom       anything else                        cone with the fixed DOF ticks
===========  ===================================  =============================

An *inclined* support (CYPE 3D "roller on an inclined plane") is a roller or
pin given joint local axes: the restrained direction is the joint's local 3
(or 1 / 2) axis, so the whole symbol is rotated by the joint axes.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SixBool = Tuple[bool, bool, bool, bool, bool, bool]

SUPPORT_TYPES: List[Tuple[str, str, SixBool, str]] = [
    ("fixed", "Fixed", (True,) * 6, "All translations and rotations"),
    ("pinned", "Pinned", (True, True, True, False, False, False), "Translations only (hinge)"),
    ("roller_z", "Roller (vertical)", (False, False, True, False, False, False), "UZ only: rolls in the plane"),
    ("roller_x", "Roller (normal X)", (True, False, False, False, False, False), "UX only"),
    ("roller_y", "Roller (normal Y)", (False, True, False, False, False, False), "UY only"),
    ("slider_xy", "Slider (horizontal)", (True, True, False, False, False, False), "UX and UY: slides vertically"),
    ("guided", "Guided (fixed, slides)", (False, False, True, True, True, False), "UZ RX RY: a fixed support on rollers"),
    ("free", "Free", (False,) * 6, "No restraint"),
]
TYPE_BY_KEY = {k: r for k, _, r, _ in SUPPORT_TYPES}
LABELS = {k: lab for k, lab, _, _ in SUPPORT_TYPES}


def support_type(restraints: Sequence[bool]) -> str:
    r = tuple(bool(x) for x in restraints)
    for key, _, pat, _ in SUPPORT_TYPES:
        if r == pat:
            return key
    return "custom"


def describe(restraints: Sequence[bool], local_axes=None) -> str:
    key = support_type(restraints)
    dofs = ("UX", "UY", "UZ", "RX", "RY", "RZ")
    if local_axes is not None and any(local_axes):
        dofs = ("U1", "U2", "U3", "R1", "R2", "R3")
    fixed = [d for d, r in zip(dofs, restraints) if r]
    text = LABELS.get(key, "Custom")
    if key in ("custom",) or (local_axes is not None and any(local_axes)):
        text += " (" + (", ".join(fixed) if fixed else "none") + ")"
    if local_axes is not None and any(local_axes):
        a, b, c = local_axes
        text += f"  inclined a={a:g}° b={b:g}° c={c:g}°"
    return text


# ------------------------------------------------------------------ pictures
def paint_support(painter, key: str, size: int = 56, ink=None, accent=None):
    """Draw the 2D picture of a support type (as in textbooks / CYPE) into a
    QPainter whose device is ``size`` x ``size`` pixels."""
    from PySide6 import QtCore, QtGui

    ink = ink or QtGui.QColor(40, 44, 52)
    accent = accent or QtGui.QColor(200, 30, 40)
    p = painter
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    s = size / 56.0
    pen = QtGui.QPen(ink, 2.0 * s)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.NoBrush)
    cx, ground = 28 * s, 40 * s

    def q(x, y):
        return QtCore.QPointF(x * s, y * s)

    def hatch(y, x0=8, x1=48, step=6):
        p.setPen(QtGui.QPen(ink, 1.3 * s))
        p.drawLine(q(x0, y), q(x1, y))
        x = x0
        while x < x1:
            p.drawLine(q(x, y), q(x - 4, y + 5))
            x += step
        p.setPen(pen)

    def member():
        p.setPen(QtGui.QPen(accent, 3.0 * s))
        p.drawLine(QtCore.QPointF(cx, 6 * s), QtCore.QPointF(cx, 22 * s))
        p.setPen(pen)

    def cone(top_y=22, base_y=40, half=12):
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(cx, top_y * s), q(28 - half, base_y), q(28 + half, base_y)]))
        p.setBrush(QtCore.Qt.NoBrush)
        p.setBrush(ink)
        p.drawEllipse(QtCore.QPointF(cx, top_y * s), 2.6 * s, 2.6 * s)
        p.setBrush(QtCore.Qt.NoBrush)

    def rollers(y, xs=(20, 28, 36), r=3.2):
        p.setBrush(QtGui.QColor(255, 255, 255))
        for x in xs:
            p.drawEllipse(q(x, y), r * s, r * s)
        p.setBrush(QtCore.Qt.NoBrush)

    if key == "fixed":
        member()
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawRect(QtCore.QRectF(12 * s, 22 * s, 32 * s, 8 * s))
        p.setBrush(QtCore.Qt.NoBrush)
        hatch(30)
    elif key == "pinned":
        member()
        cone()
        hatch(ground)
    elif key == "roller_z":
        member()
        cone(base_y=34)
        rollers(37.5)
        hatch(41)
    elif key in ("roller_x", "roller_y"):
        # roller against a vertical wall: the cone lies sideways, rollers between it and the wall
        p.setPen(QtGui.QPen(accent, 3.0 * s))
        p.drawLine(q(6, 28), q(22, 28))
        p.setPen(pen)
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawPolygon(QtGui.QPolygonF([q(22, 28), q(36, 16), q(36, 40)]))
        p.setBrush(ink)
        p.drawEllipse(q(22, 28), 2.6 * s, 2.6 * s)
        p.setBrush(QtGui.QColor(255, 255, 255))
        for y in (20, 28, 36):
            p.drawEllipse(q(40, y), 3.2 * s, 3.2 * s)
        p.setBrush(QtCore.Qt.NoBrush)
        # wall with hatching
        p.drawLine(q(44, 10), q(44, 46))
        y = 12
        while y < 46:
            p.drawLine(q(44, y), q(49, y - 4))
            y += 6
        p.setPen(QtGui.QPen(ink, 1.0 * s))
        p.drawText(QtCore.QRectF(4 * s, 40 * s, 20 * s, 14 * s), QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom,
                   "X" if key == "roller_x" else "Y")
        p.setPen(pen)
    elif key == "slider_xy":
        member()
        # a collar sliding on a vertical guide: horizontal translations fixed
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawRect(QtCore.QRectF(20 * s, 22 * s, 16 * s, 10 * s))
        p.setBrush(QtCore.Qt.NoBrush)
        rollers(26, xs=(16, 40), r=3.0)
        p.drawLine(q(11, 12), q(11, 44))
        p.drawLine(q(45, 12), q(45, 44))
        for y in (14, 20, 26, 32, 38):
            p.drawLine(q(11, y), q(7, y + 4))
            p.drawLine(q(45, y), q(49, y + 4))
    elif key == "guided":
        member()
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawRect(QtCore.QRectF(12 * s, 22 * s, 32 * s, 8 * s))
        p.setBrush(QtCore.Qt.NoBrush)
        rollers(33.5, xs=(18, 28, 38))
        hatch(37.5)
    elif key == "free":
        member()
        p.setPen(QtGui.QPen(ink, 1.2 * s, QtCore.Qt.DashLine))
        p.drawEllipse(QtCore.QPointF(cx, 22 * s), 5 * s, 5 * s)
        p.setPen(pen)
    else:  # custom: cone with "?"
        member()
        cone()
        hatch(ground)
        p.setPen(QtGui.QPen(accent, 1.5 * s))
        p.drawText(QtCore.QRectF(38 * s, 6 * s, 16 * s, 16 * s), QtCore.Qt.AlignCenter, "*")
        p.setPen(pen)


def support_pixmap(key: str, size: int = 56, ink=None, accent=None):
    from PySide6 import QtCore, QtGui

    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    paint_support(p, key, size, ink, accent)
    p.end()
    return pm


# ---------------------------------------------------------------- 3D symbols
def _rot_from_local_axes(local_axes) -> np.ndarray:
    """Rows = joint local axes in global coordinates (identity when None)."""
    if local_axes is None or not any(local_axes):
        return np.eye(3)
    from ..mesh.geometry import joint_axes
    return np.asarray(joint_axes(*local_axes), float)


def support_polydata(restraints: Sequence[bool], xyz, size: float, local_axes=None):
    """vtkPolyData of the support symbol at ``xyz`` (joint local axes applied)."""
    import vtk

    key = support_type(restraints)
    R = _rot_from_local_axes(local_axes)  # local -> global: x_g = R.T @ x_l
    parts: List[vtk.vtkPolyData] = []

    def add(src):
        src.Update()
        parts.append(src.GetOutput())

    # symbols are built in joint local axes with +3 up (the restrained normal of a
    # roller), then rotated; a horizontal roller / slider uses the other axes
    if key == "fixed":
        c = vtk.vtkCubeSource()
        c.SetXLength(size)
        c.SetYLength(size)
        c.SetZLength(size * 0.4)
        c.SetCenter(0, 0, -size * 0.2)
        add(c)
    elif key in ("pinned", "custom"):
        cone = vtk.vtkConeSource()
        cone.SetRadius(size * 0.55)
        cone.SetHeight(size)
        cone.SetDirection(0, 0, 1)
        cone.SetCenter(0, 0, -size * 0.5)
        cone.SetResolution(12)
        add(cone)
    elif key in ("roller_z", "roller_x", "roller_y", "slider_xy", "guided"):
        if key == "guided":
            c = vtk.vtkCubeSource()
            c.SetXLength(size)
            c.SetYLength(size)
            c.SetZLength(size * 0.35)
            c.SetCenter(0, 0, -size * 0.175)
            add(c)
            top = -size * 0.35
        elif key == "slider_xy":
            # collar: a short tube around the joint, rollers at its sides
            cyl = vtk.vtkCylinderSource()
            cyl.SetRadius(size * 0.45)
            cyl.SetHeight(size * 0.5)
            cyl.SetResolution(16)
            cyl.SetCenter(0, 0, 0)
            tf = vtk.vtkTransform()
            tf.RotateX(90)
            f = vtk.vtkTransformPolyDataFilter()
            f.SetTransform(tf)
            f.SetInputConnection(cyl.GetOutputPort())
            add(f)
            top = None
        else:
            cone = vtk.vtkConeSource()
            cone.SetRadius(size * 0.55)
            cone.SetHeight(size * 0.8)
            cone.SetDirection(0, 0, 1)
            cone.SetCenter(0, 0, -size * 0.4)
            cone.SetResolution(12)
            add(cone)
            top = -size * 0.8
        if top is not None:
            for dx in (-size * 0.3, size * 0.3):
                r = vtk.vtkSphereSource()
                r.SetRadius(size * 0.16)
                r.SetThetaResolution(10)
                r.SetPhiResolution(10)
                r.SetCenter(dx, 0, top - size * 0.16)
                add(r)
        else:
            for dx in (-size * 0.62, size * 0.62):
                r = vtk.vtkSphereSource()
                r.SetRadius(size * 0.16)
                r.SetThetaResolution(10)
                r.SetPhiResolution(10)
                r.SetCenter(dx, 0, 0)
                add(r)
    else:
        return None
    append = vtk.vtkAppendPolyData()
    for p in parts:
        append.AddInputData(p)
    append.Update()
    # orientation: the symbol's +3 is the restrained normal
    M = np.eye(4)
    Rl = np.eye(3)  # symbol -> joint local
    if key == "roller_x":
        Rl = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)  # symbol z -> local x
    elif key == "roller_y":
        Rl = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # symbol z -> local y
    Rg = R.T @ Rl.T  # symbol -> global
    M[:3, :3] = Rg
    M[:3, 3] = np.asarray(xyz, float)
    tf = vtk.vtkTransform()
    tf.SetMatrix(M.flatten().tolist())
    f = vtk.vtkTransformPolyDataFilter()
    f.SetTransform(tf)
    f.SetInputData(append.GetOutput())
    f.Update()
    return f.GetOutput()
