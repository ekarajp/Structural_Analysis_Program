"""Putting a CAD reference in place on screen (References menu), CAD style:

* **Align (2 Points)** - click a point of the drawing, then where it belongs in the model
  (the drawing moves there); then a second point of the drawing and its place: the
  drawing turns about the first.  When the two distances differ the command says by how
  much and asks whether to resize the drawing too.  Enter after the first pair = moved only.
* **Move** - a base point and where it goes (clicked, or typed: a distance toward the
  cursor, ``dx,dy``).
* **Rotate** - a centre, a start direction and an end direction (or the angle typed).
* **Scale by a Known Length** - two points of the drawing whose real distance is known,
  then that distance typed: the drawing is resized about the first point.

Each tool works in the plan / elevation window that shows the reference.  With several
references in the window the first click chooses the one whose line is clicked.  A
pinned reference asks before it is moved, and is pinned again when it is in place.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from . import cadref
from ..model import ModelError
from .tools import TOOLS, Preview, SnapResult, Tool, next_step

XYZ = Tuple[float, float, float]


class RefTool(Tool):
    """Base of the reference tools: which reference, the pinned question, the outline."""

    views = ("plan", "xz", "yz")
    takes_typed_points = True
    target: Optional[str] = None  # the reference worked on (None: chosen by the first click)

    def activate(self, state) -> None:
        self.state = state
        self.ref: Optional[str] = None
        self._confirmed = False
        want = getattr(state, "reference_target", None)
        if want in cadref.references_of(self.model):
            self.ref = want
        state.reference_target = None

    # ---- which reference
    def _shown(self) -> List[str]:
        pl = self.plane
        if pl.kind == "3d":
            return []
        return [r.name for r in cadref.shown_in(self.model, pl.kind, pl.value, pl.story)]

    def _choose(self, p: XYZ) -> bool:
        """Make sure ``self.ref`` is a reference of this window; False: not yet (told why)."""
        shown = self._shown()
        if self.ref in shown:
            return True
        if self.ref is not None and self.ref in cadref.references_of(self.model):
            raise ModelError("REFERENCE_NOT_HERE", f"Reference {self.ref} is not shown in this window - click in a "
                                                   "window of its story / elevation")
        if not shown:
            raise ModelError("NO_REFERENCE", "This window shows no reference (References > Place DXF Drawing, or "
                                             "switch to the story / elevation it is on)")
        if len(shown) == 1:
            self.ref = shown[0]
            return True
        m = self.model
        items = []
        for n in shown:
            dwg, _ = cadref.drawing_of(m.references[n], m, getattr(self.doc, "path", None))
            if dwg is not None:
                items.append((m.references[n], dwg))
        sn = cadref.RefSnapper([(cadref.replace(r, snap=True), d) for r, d in items], m.units.length)
        hit = sn.which(self.plane.to2(p), float(min(self.tolerance, 1e6)))
        if hit is None:
            self.state.status.emit("Several references here: click a line of the one to work on")
            return False
        self.ref = hit
        return True

    def _unpin(self) -> bool:
        """A pinned reference: ask once.  True: go on (it is unpinned by the command)."""
        ref = cadref.references_of(self.model).get(self.ref)
        if ref is None or not ref.pinned or self._confirmed:
            return True
        if self.confirm_unpin(self.ref):
            self._confirmed = True
            return True
        return False

    def confirm_unpin(self, name: str) -> bool:
        from PySide6 import QtWidgets
        box = QtWidgets.QMessageBox
        return box.question(None, self.title, f"Reference {name} is pinned in place.\n\nUnpin it and go on?",
                            box.Yes | box.No, box.Yes) == box.Yes

    def _uv(self, p) -> Tuple[float, float]:
        return self.plane.to2(p)

    def _outline(self, shift=(0.0, 0.0), turn: float = 0.0, about=None, factor: float = 1.0):
        """The box of the reference moved / turned / resized, as preview lines."""
        m = self.model
        ref = cadref.references_of(m).get(self.ref)
        if ref is None:
            return []
        dwg, _ = cadref.drawing_of(ref, m, getattr(self.doc, "path", None))
        if dwg is None:
            return []
        box = cadref.outline(ref, dwg, m.units.length)
        if box is None:
            return []
        c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
        ax, ay = about if about is not None else (0.0, 0.0)
        pts = []
        for u, v in box.tolist():
            du, dv = (u - ax) * factor, (v - ay) * factor
            pts.append(self.plane.to3((ax + c * du - s * dv + shift[0], ay + s * du + c * dv + shift[1])))
        return [(pts[k], pts[(k + 1) % 4]) for k in range(4)]

    def hint(self, state=None) -> str:
        return next_step(self.title, self.step(), "Esc = cancel")

    def step(self) -> str:
        return self.prompt

    def double_click(self, pt) -> bool:
        return False

    def _done(self, text: str) -> bool:
        self.points = []
        self._raw_last = None
        self.state.status.emit(text)
        return True


class MoveRefTool(RefTool):
    name = "ref_move"
    title = "Move Reference"

    def step(self) -> str:
        if not self.points:
            return "click a point of the drawing (the base point)"
        return "click where that point goes (or type a distance / dx,dy + Enter)"

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self._p3(pt)
        if not self.points:
            if not self._choose(p) or not self._unpin():
                return False
            self.points = [p]
            self._raw_last = p
            return False
        a = self.points[0]
        cadref.move_reference(self.doc, self.ref, self._uv(a), self._uv(p), force=True, pin=True)
        return self._done(f"Reference {self.ref} moved (pinned in place)")

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is None or not self.points:
            return pv
        a, b = self.points[0], self._p3(cursor)
        (u0, v0), (u1, v1) = self._uv(a), self._uv(b)
        pv.guides = [(a, b)]
        pv.lines = self._outline((u1 - u0, v1 - v0))
        pv.text = f"move {math.hypot(u1 - u0, v1 - v0):.3f}"
        return pv


class AlignRefTool(RefTool):
    name = "ref_align"
    title = "Align Reference"
    SAME = 0.005  # the two distances agree within this (0.5 %): no question about the scale

    def step(self) -> str:
        n = len(self.points)
        if n == 0:
            return "click a point of the drawing (for example a grid crossing of the drawing)"
        if n == 1:
            return "click where that point is in the model (the same grid crossing)"
        if n == 2:
            return "click a second point of the drawing, far from the first  [Enter / right-click = moved only]"
        return "click where the second point is in the model"

    def cancel(self):
        if len(getattr(self, "points", ())) >= 2 and getattr(self, "ref", None):
            self._pin()  # the first pair is done: it is in place
        super().cancel()

    def finish(self) -> bool:
        if len(self.points) >= 2:
            self._pin()
            return self._done(f"Reference {self.ref} moved (pinned in place)")
        return False

    def _pin(self):
        ref = cadref.references_of(self.model).get(self.ref)
        if ref is not None and not ref.pinned:
            cadref.change_reference(self.doc, self.ref, "Pin Reference", pinned=True)

    def ask_scale(self, drawn: float, real: float) -> Optional[bool]:
        """The two distances differ: True = resize the drawing, False = turn it only,
        None = cancel."""
        from PySide6 import QtWidgets
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, self.title,
                                    f"The two points are {drawn:.4g} apart on the drawing but {real:.4g} apart in the "
                                    f"model ({(real / drawn - 1) * 100:+.2f} %).\n\nThe drawing unit may be wrong, or "
                                    "the drawing is not to scale.")
        yes = box.addButton(f"Resize the drawing (x {real / drawn:.5g})", QtWidgets.QMessageBox.AcceptRole)
        no = box.addButton("Turn only (keep its size)", QtWidgets.QMessageBox.NoRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        return True if box.clickedButton() is yes else False if box.clickedButton() is no else None

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self._p3(pt)
        n = len(self.points)
        if n == 0:
            if not self._choose(p) or not self._unpin():
                return False
            self.points = [p]
            return False
        if n == 1:
            cadref.move_reference(self.doc, self.ref, self._uv(self.points[0]), self._uv(p), force=True)
            self.points = [p, p]  # (the drawing's first point is now at p)
            return False
        if n == 2:
            if math.dist(self._uv(p), self._uv(self.points[1])) <= 1e-9:
                raise ModelError("BAD_VALUE", "The second point is the first point - click another point of the drawing")
            self.points.append(p)
            return False
        q1, p2, q2 = self._uv(self.points[1]), self._uv(self.points[2]), self._uv(p)
        drawn, real = math.dist(q1, p2), math.dist(q1, q2)
        if real <= 1e-9:
            raise ModelError("BAD_VALUE", "The second point is at the first point - click its place in the model")
        scale = False
        if abs(real / drawn - 1.0) > self.SAME:
            answer = self.ask_scale(drawn, real)
            if answer is None:
                self.points = self.points[:3]
                return False
            scale = bool(answer)
        cadref.align_reference(self.doc, self.ref, q1, q1, p2, q2, scale=scale, force=True, pin=True)
        return self._done(f"Reference {self.ref} aligned" + (f", resized x {real / drawn:.5g}" if scale else "") +
                          " (pinned in place)")

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is None or not self.points:
            return pv
        c = self._p3(cursor)
        n = len(self.points)
        if n == 1:
            (u0, v0), (u1, v1) = self._uv(self.points[0]), self._uv(c)
            pv.guides = [(self.points[0], c)]
            pv.lines = self._outline((u1 - u0, v1 - v0))
        elif n == 3:
            q1, p2, q2 = self._uv(self.points[1]), self._uv(self.points[2]), self._uv(c)
            pv.guides = [(self.points[1], self.points[2]), (self.points[1], c)]
            if math.dist(q1, q2) > 1e-9 and math.dist(q1, p2) > 1e-9:
                ang = math.degrees(math.atan2(q2[1] - q1[1], q2[0] - q1[0]) - math.atan2(p2[1] - q1[1], p2[0] - q1[0]))
                pv.lines = self._outline(turn=ang, about=q1)
                pv.text = f"turn {((ang + 180) % 360) - 180:.2f}°   drawing {math.dist(q1, p2):.3f} / model {math.dist(q1, q2):.3f}"
        return pv


class RotateRefTool(RefTool):
    name = "ref_rotate"
    title = "Rotate Reference"

    @property
    def typed_is_angle(self) -> bool:
        return bool(self.points)

    def step(self) -> str:
        n = len(self.points)
        if n == 0:
            return "click the centre of rotation"
        if n == 1:
            return "click the start direction (or type the angle + Enter)"
        return "click the end direction (or type the angle + Enter)"

    def _angle(self, p) -> float:
        c, q = self._uv(self.points[0]), self._uv(p)
        return math.degrees(math.atan2(q[1] - c[1], q[0] - c[0]))

    def _turn(self, ang: float) -> bool:
        cadref.rotate_reference(self.doc, self.ref, self._uv(self.points[0]), ang, force=True, pin=True)
        return self._done(f"Reference {self.ref} turned by {((ang + 180) % 360) - 180:.2f}° (pinned in place)")

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self._p3(pt)
        if not self.points:
            if not self._choose(p) or not self._unpin():
                return False
            self.points = [p]
            return False
        if len(self.points) == 1:
            self.points.append(p)
            return False
        return self._turn(self._angle(p) - self._angle(self.points[1]))

    def typed(self, text: str, toward=None):
        if not self.points:
            raise ValueError("Click the centre of rotation first")
        self._turn(float(text.replace("°", "").strip()))
        return None

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is None or not self.points:
            return pv
        c = self._p3(cursor)
        pv.guides = [(self.points[0], c)]
        if len(self.points) == 2:
            ang = self._angle(c) - self._angle(self.points[1])
            pv.guides.append((self.points[0], self.points[1]))
            pv.lines = self._outline(turn=ang, about=self._uv(self.points[0]))
            pv.text = f"angle {((ang + 180) % 360) - 180:.2f}°"
        return pv


class ScaleRefTool(RefTool):
    name = "ref_scale"
    title = "Scale Reference"

    def step(self) -> str:
        n = len(self.points)
        if n == 0:
            return "click the first end of a line of the drawing whose real length you know"
        if n == 1:
            return "click the other end of that line"
        L = math.dist(self._uv(self.points[0]), self._uv(self.points[1]))
        return f"that line is {self._shown_length(L)} long now - type its real length + Enter"

    def _shown_length(self, L: float) -> str:
        try:
            from ..display_units import DisplayUnits
            du = DisplayUnits.of(self.model)
            return f"{L * du.factor('coordinates'):.4g} {du.label('coordinates')}"
        except Exception:
            return f"{L:.4g} {self.model.units.length}"

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self._p3(pt)
        if not self.points:
            if not self._choose(p) or not self._unpin():
                return False
            self.points = [p]
            return False
        if len(self.points) == 1:
            if math.dist(self._uv(p), self._uv(self.points[0])) <= 1e-9:
                raise ModelError("BAD_VALUE", "That is the same point - click the other end of the line")
            self.points.append(p)
            return False
        # a third click: the real length shown by a point (from the first point)
        return self._scale(math.dist(self._uv(self.points[0]), self._uv(p)))

    def _scale(self, real: float) -> bool:
        if len(self.points) < 2:
            raise ValueError("Click the two ends of a line of the drawing first")
        f = cadref.scale_by_length(self.doc, self.ref, self._uv(self.points[0]), self._uv(self.points[1]), real,
                                   force=True, pin=True)
        return self._done(f"Reference {self.ref} resized x {f:.6g} (pinned in place)")

    def typed(self, text: str, toward=None):
        if len(self.points) < 2:
            return super().typed(text, toward)
        self._scale(float(text.strip()))
        return None

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        if cursor is None or not self.points:
            return pv
        c = self._p3(cursor)
        if len(self.points) == 1:
            pv.lines = [(self.points[0], c)]
            pv.text = f"L = {self._shown_length(math.dist(self._uv(self.points[0]), self._uv(c)))}"
        else:
            pv.lines = [(self.points[0], self.points[1])]
            L0 = math.dist(self._uv(self.points[0]), self._uv(self.points[1]))
            L1 = math.dist(self._uv(self.points[0]), self._uv(c))
            pv.guides = [(self.points[0], c)]
            if L0 > 0:
                pv.text = f"type the real length + Enter   (or click: x {L1 / L0:.4g})"
        return pv


TOOLS.update({"ref_move": MoveRefTool, "ref_align": AlignRefTool, "ref_rotate": RotateRefTool,
              "ref_scale": ScaleRefTool})
REF_TOOLS = ("ref_align", "ref_move", "ref_rotate", "ref_scale")
