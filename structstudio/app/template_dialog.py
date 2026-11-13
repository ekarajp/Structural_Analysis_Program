"""Edit > Add to Model From Template > 2D Frames / 3D Structures.

A frame or structure built from a few numbers (bays, stories, sections) and added to
the open model at a point, turned about Z; joints on existing ones are shared and
members the model has already are not added twice.  A sketch shows what will be added."""
from __future__ import annotations

from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ..model.base import ModelError
from . import templates as T


class _Sketch(QtWidgets.QWidget):
    """Elevation (2D) or plan (3D) of the template in its local coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.plan = False
        self.error = ""
        self.setMinimumSize(320, 260)

    def show_model(self, model, plan: bool, error: str = ""):
        self.model, self.plan, self.error = model, plan, error
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
        p.setPen(fg)
        p.drawText(4, 12, "Plan (one story), origin at bottom left" if self.plan else "Elevation, origin at bottom left")
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
        self.support = QtWidgets.QComboBox()
        self.support.addItems(list(T.SUPPORTS))
        form.addRow("Supports at the bottom", self.support)
        self.note = QtWidgets.QLabel("Joints on existing joints are shared; members the model has already are "
                                     "not added again.")
        self.note.setWordWrap(True)
        left.addWidget(self.note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Add")
        bb.accepted.connect(self.add)
        bb.rejected.connect(self.reject)
        left.addWidget(bb)
        self.sketch = _Sketch()
        outer.addWidget(self.sketch, 1)
        for w in (self.kind, self.bottom, self.top, self.column, self.beam):
            w.currentIndexChanged.connect(self.refresh)
        for w in [self.xsp, self.ysp] + ([self.bays, self.link] if plane else []):
            w.textChanged.connect(self.refresh)
        self.bottom.currentIndexChanged.connect(self._default_support)
        self._default_support()
        self.added: Optional[list] = None
        self.refresh()

    def _default_support(self):
        lowest = self.bottom.currentIndex() == 0
        self.support.setCurrentText("Fixed" if lowest else "None")

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
        except (ValueError, ModelError) as exc:
            self.sketch.show_model(None, plan=not self.plane, error=str(exc))

    def add(self):
        try:
            p = self.params()
            ox, oy = self._len(self.ox, "Origin X"), self._len(self.oy, "Origin Y")
            ang = float(self.ang.text())
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
        try:
            from ..model.objects import Area, Frame
            self.win.state.set_selection([o.name for o in made if isinstance(o, Frame)],
                                         [o.name for o in made if isinstance(o, Area)])
        except Exception:
            pass
        msg = f"Added {len(made)} member(s)" + (f"; {skipped[0]} already in the model, not added again"
                                                  if skipped[0] else "")
        try:
            self.win.statusBar().showMessage(msg, 10000)
            self.win.logger.log(msg)
        except Exception:
            pass
        self.accept()
