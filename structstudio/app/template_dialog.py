"""Edit > Add to Model From Template > 2D Frames / 3D Structures.

A frame or structure built from a few numbers (bays, stories, sections) and added to
the open model at a point, turned about Z; joints on existing ones are shared and
members the model has already are not added twice.  A sketch shows what will be added."""
from __future__ import annotations

from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ..model.base import ModelError
from . import templates as T
from .tools import Preview, Tool, next_step


class _Sketch(QtWidgets.QWidget):
    """Elevation (2D) or plan (3D) of the template in its local coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.plan = False
        self.error = ""
        self.ref = (0.0, 0.0)  # the reference point (local u, v of the sketch)
        self.on_pick = None
        self._map = None
        self.setMinimumSize(320, 260)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip("Click a joint: the reference point of the new structure (the point that goes where "
                        "you click / type the origin)")

    def show_model(self, model, plan: bool, error: str = ""):
        self.model, self.plan, self.error = model, plan, error
        self.update()

    def candidates(self):
        """The joints a reference point can be: (u, v) of the bottom level (2D: along the
        frame; 3D: in plan)."""
        m = self.model
        if m is None:
            return []
        if self.plan:
            return sorted({(round(p.x, 9), round(p.y, 9)) for p in m.points.values()})
        z0 = min(p.z for p in m.points.values())
        return sorted({(round(p.x, 9), 0.0) for p in m.points.values() if abs(p.z - z0) < 1e-9})

    def mousePressEvent(self, ev):
        if self._map is None or self.model is None:
            return
        X, pts = self._map
        pos = ev.position() if hasattr(ev, "position") else ev.pos()
        best, bd = None, 1e9
        for c in self.candidates():
            q = X(c[0], c[1] if self.plan else self._v0)
            d = (q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2
            if d < bd:
                best, bd = c, d
        if best is not None and bd < 30 ** 2:
            self.ref = best
            if self.on_pick:
                self.on_pick(best)
            self.update()

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        fg = self.palette().color(QtGui.QPalette.WindowText)
        if self.error or self.model is None:
            p.setPen(QtGui.QColor("#c62828"))
            p.drawText(self.rect().adjusted(8, 8, -8, -8), QtCore.Qt.TextWordWrap, self.error or "")
            return
        m = self.model

        def uv(pt):
            P = m.points[pt]
            return (P.x, P.y) if self.plan else (P.x, P.z)

        pts = [uv(n) for n in m.points]
        if not pts:
            return
        u0, u1 = min(a for a, _ in pts), max(a for a, _ in pts)
        v0, v1 = min(b for _, b in pts), max(b for _, b in pts)
        W, H, pad = self.width(), self.height(), 18
        s = min((W - 2 * pad) / max(u1 - u0, 1e-9), (H - 2 * pad) / max(v1 - v0, 1e-9))

        def X(u, v):
            return QtCore.QPointF(pad + (u - u0) * s, H - pad - (v - v0) * s)

        self._v0 = v0
        self._map = (X, pts)

        if self.plan:
            p.setPen(QtCore.Qt.NoPen)
            for a in m.areas.values():
                if a.type == "Floor":
                    p.setBrush(QtGui.QColor(120, 160, 220, 60))
                    p.drawPolygon(QtGui.QPolygonF([X(*uv(q)) for q in a.points]))
            for a in m.areas.values():
                if a.type == "Wall":
                    p.setPen(QtGui.QPen(QtGui.QColor("#8d6e63"), 4))
                    q = [uv(n) for n in a.points]
                    p.drawLine(X(*q[0]), X(*q[1]))
        colors = {"Column": QtGui.QColor("#1565c0"), "Beam": fg, "Brace": QtGui.QColor("#2e7d32")}
        for f in m.frames.values():
            if self.plan and f.type == "Column":
                continue
            p.setPen(QtGui.QPen(colors.get(f.type, fg), 2))
            p.drawLine(X(*uv(f.point_i)), X(*uv(f.point_j)))
        if self.plan:
            p.setPen(QtGui.QPen(colors["Column"], 1))
            p.setBrush(colors["Column"])
            done = set()
            for f in m.frames.values():
                if f.type == "Column":
                    k = (round(m.points[f.point_i].x, 6), round(m.points[f.point_i].y, 6))
                    if k not in done:
                        done.add(k)
                        p.drawRect(QtCore.QRectF(X(*k) - QtCore.QPointF(3, 3), QtCore.QSizeF(6, 6)))
        # the reference point
        r = X(self.ref[0], self.ref[1] if self.plan else v0)
        p.setPen(QtGui.QPen(QtGui.QColor("#d32f2f"), 2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(r, 7, 7)
        p.drawLine(r - QtCore.QPointF(11, 0), r + QtCore.QPointF(11, 0))
        p.drawLine(r - QtCore.QPointF(0, 11), r + QtCore.QPointF(0, 11))
        p.setPen(fg)
        p.drawText(4, 12, ("Plan (one story)" if self.plan else "Elevation") + " - click a joint: reference point")
        p.end()


class TemplateDialog(QtWidgets.QDialog):
    def __init__(self, win, plane: bool, parent=None):
        super().__init__(parent or win)
        self.win, self.doc, self.plane = win, win.doc, plane
        m = self.doc.model
        self.setWindowTitle("Add 2D Frame From Template" if plane else "Add 3D Structure From Template")
        from .forms import display_units
        du = display_units()
        self.k = du.factor("coordinates") if du is not None else 1.0
        ul = du.label("coordinates") if du is not None else m.units.label("length")
        outer = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        outer.addLayout(left, 1)
        form = QtWidgets.QFormLayout()
        left.addLayout(form)
        self.kind = QtWidgets.QComboBox()
        items = T.FRAME_2D if plane else [s for s in T.STRUCTURES if s[0] in T.STRUCTURE_3D_KEYS]
        for key, label, tip in items:
            self.kind.addItem(label, key)
            self.kind.setItemData(self.kind.count() - 1, tip, QtCore.Qt.ToolTipRole)
        form.addRow("Template", self.kind)
        self.desc = QtWidgets.QLabel()
        self.desc.setWordWrap(True)
        form.addRow("", self.desc)
        sp = "Bay widths [%s]  (6 6 5  or  3@6)" % ul
        self.xsp = QtWidgets.QLineEdit("3@6" if plane else "2@6")
        form.addRow(("Bays along the frame " if plane else "Bays in local x ") + f"[{ul}]", self.xsp)
        self.xsp.setToolTip(sp)
        self.ysp = QtWidgets.QLineEdit("2@5")
        self.ysp.setToolTip(sp)
        if not plane:
            form.addRow(f"Bays in local y [{ul}]", self.ysp)
        stories = [s.name for s in m.stories_bottom_up()]
        self.bottom = QtWidgets.QComboBox()
        self.bottom.addItems(stories)
        self.top = QtWidgets.QComboBox()
        self.top.addItems(stories)
        if stories:
            self.top.setCurrentIndex(len(stories) - 1)
        form.addRow("From story (bottom)", self.bottom)
        form.addRow("To story (top)", self.top)
        # where: a selected joint, else the origin
        x0 = y0 = 0.0
        sel = list(getattr(win.state, "points", []) or [])
        if len(sel) == 1 and sel[0] in m.points:
            x0, y0 = m.points[sel[0]].x, m.points[sel[0]].y
        self.ox = QtWidgets.QLineEdit(f"{x0 * self.k:g}")
        self.oy = QtWidgets.QLineEdit(f"{y0 * self.k:g}")
        self.ang = QtWidgets.QLineEdit("0")
        form.addRow(f"Origin X [{ul}]", self.ox)
        form.addRow(f"Origin Y [{ul}]", self.oy)
        form.addRow("Direction / rotation (deg from X)" if plane else "Rotation about Z (deg)", self.ang)
        frames = list(m.frame_sections)
        slabs = [n for n, s in m.shell_sections.items() if s.type == "Slab"]
        walls = [n for n, s in m.shell_sections.items() if s.type == "Wall"]

        def combo(names, prefer=""):
            c = QtWidgets.QComboBox()
            c.addItems(names)
            pick = next((n for n in names if prefer and n.upper().startswith(prefer)), None)
            if pick:
                c.setCurrentText(pick)
            return c

        self.column = combo(frames, "C")
        self.beam = combo(frames, "B")
        form.addRow("Column section", self.column)
        form.addRow("Beam section", self.beam)
        if plane:
            self.brace = combo(frames, "BR")
            self.bays = QtWidgets.QLineEdit("all")
            self.bays.setToolTip("Bays with braces: all, or numbers from the origin, e.g. 1 3")
            self.link = QtWidgets.QLineEdit(f"{1.0 * self.k / max(self.doc.model.units.factor('length'), 1e-12):g}")
            form.addRow("Brace section", self.brace)
            form.addRow("Braced bays", self.bays)
            form.addRow(f"EBF link length [{ul}]", self.link)
        else:
            self.slab = combo(slabs)
            self.wall = combo(walls)
            self.infill = combo(frames)
            form.addRow("Slab section", self.slab)
            form.addRow("Wall section (core)", self.wall)
            form.addRow("Infill beam (steel deck)", self.infill)
        self.from_click = QtWidgets.QCheckBox("Place in View: the bottom story is the level clicked")
        self.from_click.setToolTip("Off (default): the stories are those above, wherever you click (only X, Y come "
                                   "from the click).\nOn: the structure starts at the story of the point clicked "
                                   "(in a plan: the story below it) with the same number of stories.")
        form.addRow("", self.from_click)
        self.support = QtWidgets.QComboBox()
        self.support.addItems(list(T.SUPPORTS))
        form.addRow("Supports at the bottom", self.support)
        self.ref_label = QtWidgets.QLabel()
        form.addRow("Reference point", self.ref_label)
        self.note = QtWidgets.QLabel("The reference point goes to the origin (or where you click with Place in "
                                     "View).  Joints on existing joints are shared; members the model has already "
                                     "are not added again.")
        self.note.setWordWrap(True)
        left.addWidget(self.note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Add")
        self.place_btn = bb.addButton("Place in View...", QtWidgets.QDialogButtonBox.ActionRole)
        self.place_btn.setToolTip("Hide this form and place the structure with the mouse: it follows the cursor "
                                  "and snaps to joints and grid crossings; the story comes from the point clicked")
        self.place_btn.clicked.connect(self.place_in_view)
        self.pick = False
        bb.accepted.connect(self.add)
        bb.rejected.connect(self.reject)
        left.addWidget(bb)
        self.sketch = _Sketch()
        self.sketch.on_pick = lambda _r: self._show_ref()
        outer.addWidget(self.sketch, 1)
        for w in (self.kind, self.bottom, self.top, self.column, self.beam):
            w.currentIndexChanged.connect(self.refresh)
        for w in [self.xsp, self.ysp] + ([self.bays, self.link] if plane else []):
            w.textChanged.connect(self.refresh)
        self.bottom.currentIndexChanged.connect(self._default_support)
        self._default_support()
        self.added: Optional[list] = None
        self.refresh()
        self._show_ref()

    def _default_support(self):
        lowest = self.bottom.currentIndex() == 0
        self.support.setCurrentText("Fixed" if lowest else "None")

    def _show_ref(self):
        u, v = self.sketch.ref
        k = self.k
        self.ref_label.setText(f"local ({u * k:g}, {v * k:g})" if not self.plane else f"{u * k:g} along the frame")

    def cycle_ref(self):
        """B in the view: the next corner / the middle of the template."""
        c = self.sketch.candidates()
        if not c:
            return
        us, vs = [a for a, _ in c], [b for _, b in c]
        if self.plane:
            ring = [(min(us), 0.0), (max(us), 0.0)]
            mid = min(c, key=lambda q: abs(q[0] - 0.5 * (min(us) + max(us))))
            ring.append(mid)
        else:
            ring = [(min(us), min(vs)), (max(us), min(vs)), (max(us), max(vs)), (min(us), max(vs))]
            cx, cy = 0.5 * (min(us) + max(us)), 0.5 * (min(vs) + max(vs))
            ring.append(min(c, key=lambda q: (q[0] - cx) ** 2 + (q[1] - cy) ** 2))
        cur = self.sketch.ref
        k = next((i for i, q in enumerate(ring) if abs(q[0] - cur[0]) < 1e-9 and abs(q[1] - cur[1]) < 1e-9), -1)
        self.sketch.ref = ring[(k + 1) % len(ring)]
        self._show_ref()
        self.sketch.update()

    def span(self) -> int:
        """Number of stories between the chosen bottom and top."""
        return self.top.currentIndex() - self.bottom.currentIndex()

    def stories_from(self, bottom: str):
        """(bottom, top) when the structure starts at ``bottom``: the same number of stories."""
        names = [s.name for s in self.doc.model.stories_bottom_up()]
        if bottom not in names:
            raise ValueError("Click a point at a story level")
        i = names.index(bottom)
        n = self.span()
        if n < 1:
            raise ValueError("The top story must be above the bottom story")
        if i + n >= len(names):
            raise ValueError(f"{n} stories from {bottom} go above the top of the model")
        return names[i], names[i + n]

    def local_model(self, bottom: str):
        """The template in local coordinates with its levels starting at ``bottom``."""
        b, t = self.stories_from(bottom)
        p = dict(self.params(), bottom=b, top=t)
        fn = T.build_plane_frame if self.plane else T.build_structure
        return fn(self.doc.model, **p)

    def origin_for(self, point, angle: float):
        """The origin putting the reference point on ``point`` (x, y) turned ``angle``."""
        import math
        u, v = self.sketch.ref
        c, s_ = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        return point[0] - (c * u - s_ * v), point[1] - (s_ * u + c * v)

    def place_in_view(self):
        try:
            self.params()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.pick = True
        self.accept()

    def place(self, point, bottom: str, angle: float) -> list:
        """Add the template with its reference point at ``point`` (xyz), its bottom on the
        story ``bottom``.  Returns the new members (one undo step)."""
        b, t = self.stories_from(bottom)
        ox, oy = self.origin_for(point, angle)
        p = dict(self.params(), bottom=b, top=t)
        sup = self.support.currentText()
        if self.bottom.currentIndex() == 0 and b != self.bottom.currentText() and sup != "None":
            sup = "None"  # placed above the base: no supports there
        made: List = []
        skipped = [0]

        def run(model):
            fn = T.add_plane_frame if self.plane else T.add_structure
            res = fn(model, origin=(ox, oy), angle=angle, support=sup, **p)
            made.extend(res)
            skipped[0] = res.skipped

        self.doc.edit(run)
        self.added = made
        self._report(made, skipped[0])
        return made

    def _report(self, made, skipped):
        try:
            from ..model.objects import Area, Frame
            self.win.state.set_selection([o.name for o in made if isinstance(o, Frame)],
                                         [o.name for o in made if isinstance(o, Area)])
        except Exception:
            pass
        msg = f"Added {len(made)} member(s)" + (f"; {skipped} already in the model, not added again"
                                                  if skipped else "")
        try:
            self.win.statusBar().showMessage(msg, 10000)
            self.win.logger.log(msg)
        except Exception:
            pass

    # --------------------------------------------------------------- values
    def _len(self, w, what) -> float:
        try:
            return float(w.text()) / self.k
        except ValueError:
            raise ValueError(f"{what}: {w.text()!r} is not a number")

    def params(self) -> dict:
        kind = self.kind.currentData()
        sp = [v / self.k for v in T.parse_spacings(self.xsp.text())]
        out = dict(kind=kind, bottom=self.bottom.currentText(), top=self.top.currentText(),
                   column=self.column.currentText() or None, beam=self.beam.currentText() or None)
        if self.plane:
            txt = self.bays.text().strip().lower()
            bays = None if txt in ("", "all") else [int(v) for v in txt.replace(",", " ").split()]
            out.update(spacings=sp, brace=self.brace.currentText() or None, braced_bays=bays,
                       link=self._len(self.link, "Link length") if kind == "eccentric" else 0.0)
        else:
            out.update(x_spacings=sp, y_spacings=[v / self.k for v in T.parse_spacings(self.ysp.text())],
                       slab=self.slab.currentText() or None, wall=self.wall.currentText() or None,
                       infill=self.infill.currentText() or None)
        return out

    def refresh(self, *_):
        key = self.kind.currentData()
        items = T.FRAME_2D if self.plane else T.STRUCTURES
        self.desc.setText(next((d for k, _, d in items if k == key), ""))
        if self.plane:
            self.link.setEnabled(key == "eccentric")
            self.bays.setEnabled(key != "moment")
            self.brace.setEnabled(key != "moment")
        else:
            self.wall.setEnabled(key == "core_frame")
            self.infill.setEnabled(key == "steel_deck")
            self.slab.setEnabled(key != "frame_only")
            self.beam.setEnabled(key not in ("flat_slab",))
        try:
            p = self.params()
            m = self.doc.model
            if self.plane:
                t, _ = T.build_plane_frame(m, **p)
            else:
                t, _ = T.build_structure(m, **p)
                lv = t.stories_bottom_up()  # plan: the top story only
                ztop = lv[-1].elevation
                keep = [f for f in t.frames.values()
                        if f.type != "Column" and abs(t.points[f.point_i].z - ztop) < 1e-9]
                cols = [f for f in t.frames.values() if f.type == "Column" and
                        abs(max(t.points[f.point_i].z, t.points[f.point_j].z) - ztop) < 1e-9]
                t.frames = {f.name: f for f in keep + cols}
                t.areas = {a.name: a for a in t.areas.values()
                           if all(abs(t.points[q].z - ztop) < 1e-9 for q in a.points) or
                           (a.type == "Wall" and any(abs(t.points[q].z - ztop) < 1e-9 for q in a.points))}
            self.sketch.show_model(t, plan=not self.plane)
            c = self.sketch.candidates()
            if c and not any(abs(q[0] - self.sketch.ref[0]) < 1e-9 and abs(q[1] - self.sketch.ref[1]) < 1e-9 for q in c):
                self.sketch.ref = (0.0, 0.0)
                self._show_ref()
        except (ValueError, ModelError) as exc:
            self.sketch.show_model(None, plan=not self.plane, error=str(exc))

    def add(self):
        try:
            p = self.params()
            ox, oy = self._len(self.ox, "Origin X"), self._len(self.oy, "Origin Y")
            ang = float(self.ang.text())
            ox, oy = self.origin_for((ox, oy), ang)  # the reference point goes to the origin typed
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        sup = self.support.currentText()
        made: List = []
        skipped = [0]

        def run(model):
            fn = T.add_plane_frame if self.plane else T.add_structure
            res = fn(model, origin=(ox, oy), angle=ang, support=sup, **p)
            made.extend(res)
            skipped[0] = res.skipped

        try:
            self.doc.edit(run)
        except (ModelError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.added = made
        self._report(made, skipped[0])
        self.accept()


class TemplatePlaceTool(Tool):
    """Place in View: the template follows the cursor by its reference point and snaps
    with the drawing snaps; the story of the point gives its bottom.  3D: one click
    (R turns 90 degrees).  2D: the first click places the reference point, the second
    the direction (Enter keeps the angle).  B: the next corner / the middle as the
    reference point.  Esc: back to the form."""

    name = "template_place"
    title = "Add From Template"
    views = ("plan", "xz", "yz", "3d")

    def __init__(self, doc, settings, dialog: "TemplateDialog"):
        super().__init__(doc, settings)
        self.dialog = dialog
        self.angle = float(dialog.ang.text() or 0.0) if dialog.ang.text().strip() else 0.0
        self.done = False
        self._cache = {}
        self.error = ""

    def activate(self, state) -> None:
        self.state = state

    @property
    def prompt(self) -> str:
        if self.dialog.plane and self.points:
            return "click the direction of the frame (or Enter: keep the angle)"
        return "click where the reference point goes"

    def hint(self, state=None) -> str:
        keys = ["B = reference point", "Esc = back to the form"]
        if not self.dialog.plane or not self.points:
            keys.insert(0, "R = turn 90°")
        return next_step(self.title, self.prompt, *keys)

    def key(self, k: str) -> bool:
        if k in ("r", "R"):
            if self.dialog.plane and self.points:
                return False
            self.angle = (self.angle + (90.0 if k == "r" else -90.0)) % 360.0
            return True
        if k in ("b", "B"):
            self.dialog.cycle_ref()
            return True
        return False

    def _bottom_at(self, P) -> Optional[str]:
        """The bottom story of the structure: the form's - or, with "the bottom story is the
        level clicked", the point's story level (in a plan the story below it, as columns
        drawn in a plan: the plan shows the tops of the first story)."""
        if not self.dialog.from_click.isChecked():
            return self.dialog.bottom.currentText()
        st = self._story_at(P[2])
        if st is None or self.plane.kind != "plan":
            return st
        below = self.model.story_below(st) if hasattr(self.model, "story_below") else None
        return below.name if below is not None else st

    def _local(self, bottom: str):
        if bottom not in self._cache:
            self._cache[bottom] = self.dialog.local_model(bottom)
        return self._cache[bottom]

    def _angle_to(self, a, b) -> float:
        import math
        dx, dy = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dy) < 1e-9:
            return self.angle
        return math.degrees(math.atan2(dy, dx))

    def _ghost(self, P, angle):
        """Lines of the template with its reference point at P, or (None, error)."""
        import math
        bottom = self._bottom_at(P)
        if bottom is None:
            return None, "not at a story level"
        try:
            t, levels = self._local(bottom)
        except (ValueError, ModelError) as exc:
            return None, str(exc)
        ox, oy = self.dialog.origin_for(P, angle)
        c, s_ = math.cos(math.radians(angle)), math.sin(math.radians(angle))

        flat = self.plane.kind  # in a plan / elevation the ghost is drawn on the window's plane
        val = self.plane.value

        def X(q):
            pt = t.points[q]
            x, y, z = ox + c * pt.x - s_ * pt.y, oy + s_ * pt.x + c * pt.y, pt.z
            if flat == "plan":
                z = val
            elif flat == "xz":
                y = val
            elif flat == "yz":
                x = val
            return (x, y, z)

        out = []
        for f in list(t.frames.values())[:4000]:
            a, b = X(f.point_i), X(f.point_j)
            if a != b:
                out.append((a, b))
        for a in list(t.areas.values())[:4000]:
            q = [X(n) for n in a.points]
            out += [(q[k], q[(k + 1) % len(q)]) for k in range(len(q))]
        return out, f"{bottom} - {levels[-1][0]}"

    def preview(self, cursor) -> Preview:
        if cursor is None:
            return Preview()
        P = self._p3(cursor)
        pv = Preview(marker=P)
        if self.dialog.plane and self.points:
            base = self.points[0]
            ang = self._angle_to(base, P)
            lines, info = self._ghost(base, ang)
            pv.guides = [(base, P)]
            pv.text = f"∠{ang:.1f}°"
        else:
            ang = self.angle
            lines, info = self._ghost(P, ang)
            pv.text = f"∠{ang:.0f}°"
        if lines is None:
            pv.text += f"  (cannot place here: {info})"
            return pv
        pv.lines = lines
        pv.text += f"  {info}"
        return pv

    def click(self, pt, snap=None) -> bool:
        P = self._p3(pt)
        if self.dialog.plane and not self.points:
            if self._story_at(P[2]) is None:
                self.state.status.emit("Add From Template: click a point at a story level")
                return False
            self.points = [P]
            self.state.update_hint()
            return False
        if self.dialog.plane:
            return self._place(self.points[0], self._angle_to(self.points[0], P))
        return self._place(P, self.angle)

    def finish(self) -> bool:
        if self.dialog.plane and self.points:
            return self._place(self.points[0], self.angle)
        return False

    def _place(self, P, angle) -> bool:
        bottom = self._bottom_at(P)
        try:
            if bottom is None:
                raise ValueError("Click a point at a story level (or switch off 'the bottom story is the level "
                                 "clicked')")
            self.dialog.place(P, bottom, angle)
        except (ValueError, ModelError) as exc:
            self.state.status.emit(f"Add From Template: {exc}")
            self.points = []
            return False
        self.done = True
        self.points = []
        self.state.set_tool("select")
        if self.dialog.added:
            from ..model.objects import Area, Frame
            self.state.set_selection([o.name for o in self.dialog.added if isinstance(o, Frame)],
                                     [o.name for o in self.dialog.added if isinstance(o, Area)])
        return True

    def deactivate(self):
        """Left without placing (Esc / another command): back to the form."""
        if not self.done:
            QtCore.QTimer.singleShot(0, lambda: reopen(self.dialog))


def reopen(dialog: "TemplateDialog"):
    """Show the form again (after Place in View); Add / Place in View / Cancel as before."""
    dialog.pick = False
    dialog.refresh()
    dialog.exec()
    if dialog.pick:
        start_placing(dialog)


def start_placing(dialog: "TemplateDialog"):
    win = dialog.win
    st = win.state
    st.start_tool(lambda: TemplatePlaceTool(win.doc, st.settings, dialog), "template_place")
