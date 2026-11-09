"""Object Information forms: right click on a joint, frame or area object.

Like the ETABS *{Joint, Beam/Column/Brace, Slab/Deck/Wall} Information* forms
and the SAP2000 *Object Model - {Point, Line, Area} Information* forms: a tabbed
window with *Geometry*, *Assignments* and *Loads*, every assignment editable in
place (changes are applied through the document, so they are undoable).
Shift + right click with several objects under the cursor shows a chooser.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..model import FIXED, PINNED
from ..model.objects import Area, Frame, FrameLoad, Point, PointLoad
from .document import Document
from .symbols import pretty, rich
from . import supports

FREE6 = (False,) * 6
DOFS = ("UX", "UY", "UZ", "RX", "RY", "RZ")


def _num(v, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}g}" if abs(float(v)) < 1e6 else f"{float(v):.4e}"
    except (TypeError, ValueError):
        return str(v)


def _du(model):
    from ..display_units import DisplayUnits
    return DisplayUnits.of(model)


class _Grid(QtWidgets.QTableWidget):
    """Two-column (item / value) read-mostly grid, like the ETABS forms."""

    def __init__(self, rows: Sequence[Tuple[str, str]], parent=None):
        super().__init__(len(rows), 2, parent)
        self.setHorizontalHeaderLabels(["Item", "Value"])
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        for r, (k, v) in enumerate(rows):
            self.setItem(r, 0, QtWidgets.QTableWidgetItem(pretty(k)))  # m², I₃₃, f′c, φ ...
            self.setItem(r, 1, QtWidgets.QTableWidgetItem(pretty(v)))
        self.resizeRowsToContents()


def _rich_label(text: str) -> QtWidgets.QLabel:
    """A form label with the engineering symbols typeset (m², f′<sub>c</sub>, I<sub>33</sub>, φ)."""
    lab = QtWidgets.QLabel(rich(text))
    lab.setTextFormat(QtCore.Qt.RichText)
    return lab


# ------------------------------------------------------------- restraints
class RestraintPicker(QtWidgets.QWidget):
    """Assign Joint Restraints with pictures: a row of support-type buttons, the six
    DOF boxes and the inclined-support angles (joint local axes a, b, c)."""

    changed = QtCore.Signal()

    def __init__(self, restraints=FREE6, local_axes=None, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        # --- support type: one size of button, 4 a row, the picture over its name
        types = QtWidgets.QGroupBox("Support type")
        row = QtWidgets.QGridLayout(types)
        row.setSpacing(6)
        self.type_buttons: Dict[str, QtWidgets.QToolButton] = {}
        pal = self.palette()
        ink = pal.color(QtGui.QPalette.WindowText)
        for k, (key, label, pat, hint) in enumerate(supports.SUPPORT_TYPES):
            b = QtWidgets.QToolButton()
            b.setIcon(QtGui.QIcon(supports.support_pixmap(key, 52, ink)))
            b.setIconSize(QtCore.QSize(52, 52))
            b.setText(label.replace(" (", "\n("))  # the qualifier on its own line
            b.setToolTip(hint + "\n" + ", ".join(d for d, r in zip(DOFS, pat) if r))
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            b.setFixedSize(122, 104)
            b.setCheckable(True)
            b.setAutoExclusive(False)
            b.clicked.connect(lambda _=False, kk=key: self.set_type(kk))
            row.addWidget(b, k // 4, k % 4)
            self.type_buttons[key] = b
        lay.addWidget(types)
        # --- the six restraints: translations and rotations in two aligned rows
        rgrp = QtWidgets.QGroupBox("Restrained directions")
        dof = QtWidgets.QGridLayout(rgrp)
        dof.setHorizontalSpacing(24)
        for c, axis in enumerate(("X", "Y", "Z")):
            h = QtWidgets.QLabel(axis)
            h.setAlignment(QtCore.Qt.AlignCenter)
            h.setStyleSheet("font-weight: 600")
            dof.addWidget(h, 0, c + 1)
        dof.addWidget(QtWidgets.QLabel("Translation"), 1, 0)
        dof.addWidget(QtWidgets.QLabel("Rotation"), 2, 0)
        self.boxes: List[QtWidgets.QCheckBox] = []
        for k, lab in enumerate(DOFS):
            cb = QtWidgets.QCheckBox(lab)
            cb.toggled.connect(self._boxes_changed)
            dof.addWidget(cb, 1 + k // 3, 1 + k % 3, alignment=QtCore.Qt.AlignCenter)
            self.boxes.append(cb)
        dof.setColumnStretch(4, 1)
        lay.addWidget(rgrp)
        inc = QtWidgets.QGroupBox("Inclined support (joint local axes)")
        inc.setCheckable(True)
        self.inclined = inc
        outer = QtWidgets.QVBoxLayout(inc)
        body = QtWidgets.QWidget()  # shown only when ticked: the form stays short
        outer.addWidget(body)
        f = QtWidgets.QFormLayout(body)
        f.setContentsMargins(0, 0, 0, 0)
        self.angles: List[QtWidgets.QDoubleSpinBox] = []
        for lab, hint in (("a: about +3 (plan rotation)", "Rotate the joint axes about global Z"),
                          ("b: about +2 (tilt)", "Then tilt about the new local 2 (a roller on a slope)"),
                          ("c: about +1", "Then rotate about the new local 1")):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-360.0, 360.0)
            sp.setSuffix(" °")
            sp.setDecimals(3)
            sp.setToolTip(hint)
            sp.valueChanged.connect(lambda _v: self.changed.emit())
            f.addRow(lab, sp)
            self.angles.append(sp)
        note = QtWidgets.QLabel("Restraints act along the joint local axes: a roller on a 30° slope is "
                                "'Roller (vertical)' with b = 30° (UZ becomes U3, normal to the slope).")
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid)")
        f.addRow(note)
        inc.toggled.connect(body.setVisible)
        self._inclined_body = body
        inc.toggled.connect(lambda _on: self.changed.emit())
        lay.addWidget(inc)
        self.set_value(restraints, local_axes)

    # ---- value
    def set_value(self, restraints, local_axes=None):
        for cb, r in zip(self.boxes, restraints):
            cb.blockSignals(True)
            cb.setChecked(bool(r))
            cb.blockSignals(False)
        la = local_axes if local_axes is not None and any(local_axes) else None
        self.inclined.setChecked(la is not None)
        self._inclined_body.setVisible(la is not None)
        for sp, v in zip(self.angles, la or (0.0, 0.0, 0.0)):
            sp.setValue(float(v))
        self._sync_type()

    def restraints(self) -> Tuple[bool, ...]:
        return tuple(cb.isChecked() for cb in self.boxes)

    def local_axes(self) -> Optional[Tuple[float, float, float]]:
        if not self.inclined.isChecked():
            return None
        v = tuple(sp.value() for sp in self.angles)
        return v if any(v) else None

    def set_type(self, key: str):
        pat = supports.TYPE_BY_KEY.get(key)
        if pat is None:
            return
        for cb, r in zip(self.boxes, pat):
            cb.blockSignals(True)
            cb.setChecked(r)
            cb.blockSignals(False)
        self._sync_type()
        self.changed.emit()

    def _boxes_changed(self, _on):
        self._sync_type()
        self.changed.emit()

    def _sync_type(self):
        key = supports.support_type(self.restraints())
        for k, b in self.type_buttons.items():
            b.setChecked(k == key)


class RestraintDialog(QtWidgets.QDialog):
    """Assign > Joint > Restraints (with pictures and inclined supports)."""

    def __init__(self, current=FREE6, local_axes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Joint Restraints")
        lay = QtWidgets.QVBoxLayout(self)
        self.picker = RestraintPicker(current, local_axes)
        lay.addWidget(self.picker)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        # compatibility with the old dialog
        self.boxes = self.picker.boxes

    def value(self):
        return self.picker.restraints()

    def local_axes(self):
        return self.picker.local_axes()


# --------------------------------------------------------- base info form
class ObjectInfoDialog(QtWidgets.QDialog):
    """Common frame of the three information forms."""

    kind = "object"

    def __init__(self, doc: Document, name: str, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.name = name
        self.setWindowFlag(QtCore.Qt.Tool, True)
        self.resize(560, 620)
        lay = QtWidgets.QVBoxLayout(self)
        self.head = QtWidgets.QLabel()
        self.head.setStyleSheet("font-weight: 600")
        lay.addWidget(self.head)
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray")
        lay.addWidget(self.hint)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Apply
                                        | QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Reset)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        bb.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(self.reload)
        lay.addWidget(bb)
        self._changes: Dict[str, object] = {}
        self.reload()

    # ---- helpers
    def obj(self):
        raise NotImplementedError

    def reload(self):
        self._changes = {}
        while self.tabs.count():
            self.tabs.removeTab(0)
        self.build()

    def build(self):
        raise NotImplementedError

    def mutate(self, model):
        raise NotImplementedError

    def apply(self):
        if not self._changes:
            return
        changes = dict(self._changes)

        def fn(model):
            self.mutate_with(model, changes)

        self.doc.edit(fn)
        self.reload()

    def _ok(self):
        self.apply()
        self.accept()

    def _edit(self, key: str, value):
        self._changes[key] = value

    def _row_edit(self, form: QtWidgets.QFormLayout, label: str, key: str, value, kind: str = "float",
                  choices: Optional[Sequence[str]] = None, hint: str = "", item: Optional[str] = None):
        k = 1.0
        if item is not None:  # shown / typed in the display unit of ``item``
            du = _du(self.doc.model)
            k = du.factor(item)
            label = f"{label} ({du.label(item)})"
            if kind == "vec":
                value = tuple(v * k for v in value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                value = value * k
        if kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda on, k=key: self._edit(k, bool(on)))
        elif kind == "choice":
            w = QtWidgets.QComboBox()
            w.addItems(list(choices or []))
            if value in (choices or []):
                w.setCurrentText(str(value))
            w.currentTextChanged.connect(lambda t, k=key: self._edit(k, t))
        elif kind == "text":
            w = QtWidgets.QLineEdit("" if value is None else str(value))
            w.editingFinished.connect(lambda k=key, ww=w: self._edit(k, ww.text().strip() or None))
        elif kind == "vec":
            w = QtWidgets.QLineEdit(", ".join(_num(v) for v in value))
            w.setToolTip("three values separated by commas")

            def vec_done(kv=key, ww=w, kk=k):
                try:
                    vals = tuple(float(x) / kk for x in ww.text().replace(";", ",").split(","))
                    if len(vals) == 3:
                        self._edit(kv, vals)
                except ValueError:
                    pass
            w.editingFinished.connect(vec_done)
        else:
            w = QtWidgets.QLineEdit("" if value is None else _num(value, 6))

            def num_done(kv=key, ww=w, optional=value is None, kk=k):
                t = ww.text().strip()
                if not t:
                    self._edit(kv, None)
                    return
                try:
                    self._edit(kv, float(t) / kk)
                except ValueError:
                    pass
            w.editingFinished.connect(num_done)
        if hint:
            w.setToolTip(hint)
        form.addRow(_rich_label(label), w)
        return w


# ------------------------------------------------------------------ joint
class JointInfoDialog(ObjectInfoDialog):
    kind = "joint"

    def obj(self) -> Optional[Point]:
        return self.doc.model.points.get(self.name)

    def build(self):
        m = self.doc.model
        p = self.obj()
        if p is None:
            self.head.setText(f"Joint {self.name} no longer exists")
            return
        self.setWindowTitle(f"Joint Object Information - {p.name}")
        label = p.label or p.name
        self.head.setText(f"Object ID: Story {p.story or '-'}   Label {label}   Unique Name {p.name}")
        # Geometry
        g = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(g)
        self._row_edit(f, "Global X", "x", p.x, item="coordinates")
        self._row_edit(f, "Global Y", "y", p.y, item="coordinates")
        self._row_edit(f, "Global Z", "z", p.z, item="coordinates")
        frames = [fr.name for fr in m.frames.values() if p.name in (fr.point_i, fr.point_j)]
        areas = [a.name for a in m.areas.values() if p.name in a.points]
        f.addRow("Special joint", QtWidgets.QLabel("Yes" if not frames and not areas else "No"))
        conn = QtWidgets.QLabel(("Frames: " + ", ".join(frames) if frames else "Frames: none") + "\n"
                                + ("Areas: " + ", ".join(areas) if areas else "Areas: none"))
        conn.setWordWrap(True)
        f.addRow("Connectivity", conn)
        self.tabs.addTab(g, "Geometry")
        # Assignments
        a = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(a)
        # a joint is held by restraints (rigid) or by springs (elastic): two kinds of
        # support - the springs only appear on a joint that has them
        has_springs = any(abs(float(k)) > 0 for k in tuple(p.springs) + tuple(p.rotational_springs))
        if any(p.restraints) or not has_springs:
            grp = QtWidgets.QGroupBox("Restraints: " + supports.describe(p.restraints, p.local_axes))
            gl = QtWidgets.QVBoxLayout(grp)
            self.picker = RestraintPicker(p.restraints, p.local_axes)
            self.picker.changed.connect(self._restraints_changed)
            gl.addWidget(self.picker)
            al.addWidget(grp)
        if has_springs:
            sf = QtWidgets.QFormLayout()
            ku, kr = m.units.shown("translational_stiffness"), m.units.shown("rotational_stiffness")
            self._row_edit(sf, f"Springs UX, UY, UZ  [{ku}]", "springs", p.springs, "vec")
            self._row_edit(sf, f"Springs RX, RY, RZ  [{kr}]", "rotational_springs", p.rotational_springs, "vec")
            sg = QtWidgets.QGroupBox("Spring support (elastic)")
            sg.setLayout(sf)
            al.addWidget(sg)
        af = QtWidgets.QFormLayout()
        self._row_edit(af, "Diaphragm", "diaphragm", p.diaphragm, "choice", ["(none)"] + list(m.diaphragms))
        self._row_edit(af, f"Additional mass  [{m.units.shown('mass')}]", "mass", p.mass, "vec")
        self._row_edit(af, "Panel zone", "panel_zone", p.panel_zone, "text")
        grp2 = QtWidgets.QGroupBox("Diaphragm, mass")
        grp2.setLayout(af)
        al.addWidget(grp2)
        al.addStretch(1)
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(a)
        self.tabs.addTab(sc, "Assignments")
        # Loads
        rows = []
        for ld in p.loads:
            rows.append((f"Force {ld.pattern}", f"F = ({', '.join(_num(v) for v in ld.force)})  "
                                               f"M = ({', '.join(_num(v) for v in ld.moment)})"))
        for gd in p.ground_displacements:
            rows.append((f"Ground displacement {gd.pattern}",
                         f"U = ({', '.join(_num(v) for v in gd.translations)})  "
                         f"R = ({', '.join(_num(v) for v in gd.rotations)})  [{gd.coord_sys}]"))
        self.tabs.addTab(_Grid(rows or [("Loads", "none")]), "Loads")
        self.hint.setText("Click a support picture or tick the DOFs; the inclined box rotates the joint axes "
                          "(CYPE 3D style roller on a slope).  Changes are applied with Apply / OK and can be undone.")

    def _restraints_changed(self):
        self._edit("restraints", self.picker.restraints())
        self._edit("local_axes", self.picker.local_axes())

    def mutate_with(self, model, changes):
        p = model.points.get(self.name)
        if p is None:
            return
        for k, v in changes.items():
            if k == "diaphragm" and v == "(none)":
                v = None
            if k in ("springs", "rotational_springs", "mass") and v is not None:
                v = tuple(float(x) for x in v)
            setattr(p, k, v)


# ------------------------------------------------------------------ frame
class FrameInfoDialog(ObjectInfoDialog):
    kind = "frame"

    def obj(self) -> Optional[Frame]:
        return self.doc.model.frames.get(self.name)

    def build(self):
        m = self.doc.model
        fr = self.obj()
        if fr is None:
            self.head.setText(f"Frame {self.name} no longer exists")
            return
        self.setWindowTitle(f"{fr.type} Object Information - {fr.name}")
        self.head.setText(f"Object ID: Story {fr.story or '-'}   Label {fr.label or fr.name}   Unique Name {fr.name}")
        a, b = m.points[fr.point_i], m.points[fr.point_j]
        L = math.dist(a.xyz, b.xyz)
        du = _du(m)
        u, kc = du.label("coordinates"), du.factor("coordinates")
        rows = [("Start joint", f"{fr.point_i}  ({_num(a.x * kc)}, {_num(a.y * kc)}, {_num(a.z * kc)}) {u}"),
                ("End joint", f"{fr.point_j}  ({_num(b.x * kc)}, {_num(b.y * kc)}, {_num(b.z * kc)}) {u}"),
                (f"Length ({u})", _num(L * kc, 6)),
                ("Local 1 direction", f"({', '.join(_num(v) for v in ((b.x - a.x) / L, (b.y - a.y) / L, (b.z - a.z) / L))})"
                 if L > 0 else "-")]
        self.tabs.addTab(_Grid(rows), "Geometry")
        # Assignments: only what applies to this kind of member; the rarely used items in "Advanced"
        t = fr.type
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self._row_edit(f, "Object type", "type", fr.type, "choice", ["Beam", "Column", "Brace", "Null"])
        if t != "Null":
            self._row_edit(f, "Section property", "section", fr.section, "choice", list(m.frame_sections))
            self._row_edit(f, "Local axis 2 angle (deg)", "angle", fr.angle)
            self._rel_row(f, "Releases at start (U1 U2 U3 R1 R2 R3)", "releases_i", fr.releases_i)
            self._rel_row(f, "Releases at end", "releases_j", fr.releases_j)
            self._row_edit(f, "Cardinal point (1-11)", "cardinal_point", fr.cardinal_point)
            if t in ("Beam", "Column"):
                self._row_edit(f, "End length offset at start", "end_offset_i", fr.end_offset_i, item="coordinates")
                self._row_edit(f, "End length offset at end", "end_offset_j", fr.end_offset_j, item="coordinates")
                self._row_edit(f, "Rigid zone factor" + (" (model default)" if fr.rigid_factor is None else ""),
                               "rigid_factor", m.frame_rigid_factor(fr))
            if t == "Brace":
                self._row_edit(f, "Tension limit", "tension_limit", fr.tension_limit, item="force")
                self._row_edit(f, "Compression limit", "compression_limit", fr.compression_limit, item="force")
            if t == "Beam":
                self._row_edit(f, f"Frame springs k1, k2, k3 ({m.units.shown('force_per_area')})", "foundation",
                               fr.foundation, "vec")
                self._row_edit(f, "Spandrel", "spandrel", fr.spandrel, "text")
            if t == "Column":
                self._row_edit(f, "Pier", "pier", fr.pier, "text")
            mods = fr.modifiers
            f.addRow("Property modifiers", QtWidgets.QLabel("none" if mods is None else str(mods)))
        self._row_edit(f, "Group", "group", fr.group, "text")
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        self.tabs.addTab(sc, "Assignments")
        if t != "Null":
            adv = QtWidgets.QWidget()
            g = QtWidgets.QFormLayout(adv)
            if t in ("Beam", "Column"):
                self._row_edit(g, "Automatic end offsets" + (" (model default)" if fr.auto_end_offsets is None
                                                               else ""),
                               "auto_end_offsets", m.frame_auto_end_offsets(fr), "bool")
            self._row_edit(g, "Joint offset at start", "joint_offset_i", fr.joint_offset_i, "vec", item="coordinates")
            self._row_edit(g, "Joint offset at end", "joint_offset_j", fr.joint_offset_j, "vec", item="coordinates")
            self._row_edit(g, "Joint offsets in local axes", "joint_offset_local", fr.joint_offset_local, "bool")
            self._row_edit(g, "Max station spacing", "max_station_spacing", fr.max_station_spacing,
                           item="coordinates")
            self._row_edit(g, "Min number of stations", "min_stations", fr.min_stations)
            self._row_edit(g, f"Additional mass / length ({m.units.shown('mass_per_length')})", "mass_per_length",
                           fr.mass_per_length)
            self._row_edit(g, "Auto mesh", "auto_mesh", fr.auto_mesh, "bool")
            self._row_edit(g, "Mesh at intersections", "mesh_at_intersections", fr.mesh_at_intersections, "bool")
            sa = QtWidgets.QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(adv)
            self.tabs.addTab(sa, "Advanced")
        rows = []
        for ld in fr.loads:
            if ld.kind == "distributed":
                rows.append((f"Distributed {ld.load_type} {ld.pattern}",
                             f"{ld.direction}: {_num(ld.value_a)} -> {_num(ld.value_b)} "
                             f"from {_num(ld.rel_a)} to {_num(ld.rel_b)}{' (abs)' if ld.absolute else ''}"))
            elif ld.kind == "point":
                rows.append((f"Point {ld.load_type} {ld.pattern}",
                             f"{ld.direction}: {_num(ld.value_a)} at {_num(ld.rel_a)}{' (abs)' if ld.absolute else ''}"))
            elif ld.kind == "temperature":
                rows.append((f"Temperature {ld.load_type} {ld.pattern}", f"{_num(ld.value_a)} / {_num(ld.value_b)}"))
            else:
                rows.append((f"{ld.kind} {ld.pattern}", f"{ld.direction} x {_num(ld.value_a)}"))
        self.tabs.addTab(_Grid(rows or [("Loads", "none")]), "Loads")
        sec = m.frame_sections.get(fr.section)
        self.hint.setText("Design: " + (f"{sec.shape} section, material {sec.material}" if sec else "no section"))

    def _rel_row(self, form, label, key, value):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        boxes = []
        for k, lab in enumerate(("U1", "U2", "U3", "R1", "R2", "R3")):
            cb = QtWidgets.QCheckBox(lab)
            cb.setChecked(bool(value[k]))
            cb.toggled.connect(lambda _on, kk=key, bb=boxes: self._edit(kk, tuple(b.isChecked() for b in bb)))
            h.addWidget(cb)
            boxes.append(cb)
        form.addRow(_rich_label(label), w)

    def mutate_with(self, model, changes):
        fr = model.frames.get(self.name)
        if fr is None:
            return
        for k, v in changes.items():
            if k in ("min_stations", "cardinal_point") and v is not None:
                v = int(v)
            setattr(fr, k, v)


# ------------------------------------------------------------------- area
class AreaInfoDialog(ObjectInfoDialog):
    kind = "area"

    def obj(self) -> Optional[Area]:
        return self.doc.model.areas.get(self.name)

    def build(self):
        m = self.doc.model
        ar = self.obj()
        if ar is None:
            self.head.setText(f"Area {self.name} no longer exists")
            return
        self.setWindowTitle(f"{ar.type} Object Information - {ar.name}")
        self.head.setText(f"Object ID: Story {ar.story or '-'}   Label {ar.label or ar.name}   Unique Name {ar.name}")
        pts = [m.points[n] for n in ar.points]
        rows = [("Number of joints / edges", str(len(pts)))]
        for k, p in enumerate(pts):
            rows.append((f"Joint {k + 1}", f"{p.name}  ({_num(p.x)}, {_num(p.y)}, {_num(p.z)})"))
        try:
            import numpy as np
            P = np.array([p.xyz for p in pts])
            c = P.mean(axis=0)
            area = 0.5 * float(np.linalg.norm(sum(np.cross(P[i] - c, P[(i + 1) % len(P)] - c) for i in range(len(P)))))
            rows.append((f"Total area ({m.units.shown('area')})", _num(area, 6)))
        except Exception:
            pass
        self.tabs.addTab(_Grid(rows), "Geometry")
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self._row_edit(f, "Object type", "type", ar.type, "choice", ["Floor", "Wall", "Ramp", "Opening", "Null"])
        self._row_edit(f, "Section property", "section", ar.section or "(none)", "choice",
                       ["(none)"] + list(m.shell_sections))
        self._row_edit(f, "Diaphragm", "diaphragm", ar.diaphragm, "choice", ["(none)"] + list(m.diaphragms))
        self._row_edit(f, "Local axis 2 angle (deg)", "angle", ar.angle)
        self._row_edit(f, f"Additional mass / area ({m.units.shown('mass_per_area')})", "mass_per_area",
                       ar.mass_per_area)
        self._row_edit(f, "Auto mesh", "auto_mesh", ar.auto_mesh, "bool")
        self._row_edit(f, "Mesh type", "mesh_type", ar.mesh_type, "choice", ["Rectangular", "General", "None"])
        self._row_edit(f, "Max mesh size", "max_mesh_size", ar.max_mesh_size, item="coordinates")
        self._row_edit(f, f"Area spring ({m.units.shown('weight_per_volume')})", "spring", ar.spring)
        self._row_edit(f, "Pier", "pier", ar.pier, "text")
        self._row_edit(f, "Spandrel", "spandrel", ar.spandrel, "text")
        f.addRow("Property modifiers", QtWidgets.QLabel("none" if ar.modifiers is None else str(ar.modifiers)))
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        self.tabs.addTab(sc, "Assignments")
        # area loads are the load panels made from this area (Load menu)
        du = _du(m)
        u, ka = du.label("area_load"), du.factor("area_load")
        dist = {"shells": "on the area", "one_way": "one-way to frames", "two_way": "two-way to frames"}
        rows = [(f"Uniform {ld.pattern}", f"{ld.direction}: {_num(ld.value * ka)} {u}") for ld in ar.loads]
        panels = [p for p in m.panel_loads.values() if p.area == ar.name]
        for p in panels:
            for c in p.loads:
                v = _num(c.value * ka) if not c.varying else \
                    f"{_num(c.value_lower * ka)}..{_num(c.value_upper * ka)} (height)"
                rows.append((f"{p.name}  {c.pattern}", f"{c.direction}: {v} {u}  [{dist.get(p.distribution, p.distribution)}]"))
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_Grid(rows or [("Loads", "none")]))
        self.panel_names = [p.name for p in panels]
        self.edit_loads = QtWidgets.QPushButton("Edit Area Loads..." if panels else "Add Area Load...")
        self.edit_loads.clicked.connect(self._edit_area_loads)
        lay.addWidget(self.edit_loads)
        self.tabs.addTab(page, "Loads")

    def _edit_area_loads(self):
        """Edit (or add) the area-load panel of this area in the Loads on Panel form."""
        from .panel_dialog import PanelLoadDialog
        m = self.doc.model
        panel = m.panel_loads.get(self.panel_names[0]) if self.panel_names else None
        tmp = panel
        if tmp is None:
            from ..model.objects import PanelLoad
            ar = m.areas[self.name]
            tmp = PanelLoad("(new)", [m.points[p].xyz for p in ar.points], "shells", area=self.name)
        d = PanelLoadDialog(m, tmp, self)
        if not d.exec():
            return
        v = d.value()
        name = self.name

        def mutate(model):
            if panel is None:
                q = model.add_area_panel(name, [], distribution=v.distribution, span_dir=v.span_dir)
            else:
                q = model.panel_loads[panel.name]
            q.distribution, q.span_dir, q.receivers = v.distribution, v.span_dir, v.receivers
            q.loads = list(v.loads)

        self.doc.edit(mutate)
        self.close()

    def mutate_with(self, model, changes):
        ar = model.areas.get(self.name)
        if ar is None:
            return
        for k, v in changes.items():
            if k in ("section", "diaphragm") and v == "(none)":
                v = None
            setattr(ar, k, v)


DIALOGS = {"point": JointInfoDialog, "frame": FrameInfoDialog, "area": AreaInfoDialog}


def open_object_info(doc: Document, kind: str, name: str, parent=None) -> ObjectInfoDialog:
    d = DIALOGS[kind](doc, name, parent)
    d.show()
    d.raise_()
    return d


def choose_object(hits: Sequence[Tuple[str, str]], parent=None) -> Optional[Tuple[str, str]]:
    """Shift + right click: several objects under the cursor."""
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    items = [f"{k.title()} {n}" for k, n in hits]
    text, ok = QtWidgets.QInputDialog.getItem(parent, "Select object", "Objects at this location:", items, 0, False)
    if not ok:
        return None
    return hits[items.index(text)]
