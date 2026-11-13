"""Edit > Edit Stories and Grid Systems: the grid systems of the model and its stories
(levels) at a glance, as ETABS and the Frame Sections window do.

The list shows every grid system with its type, lines, origin and rotation; the
selected one is drawn on the right (the other systems faintly, for where it sits).
Add New / Add Copy / Modify / Delete act on the model at once (each one undoable),
and Modify / double-click opens the grid editor (``GridStoriesDialog``) on that
system.  The Stories box lists the levels and opens the same editor on its Story
(or Z Grid) tab.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..model.base import ModelError
from ..model.grids import CYLINDRICAL, GridGeometry, grid_geometry, to_global
from .document import Document

ACCENT = QtGui.QColor(61, 139, 253)


def system_geometry(model, name: str) -> GridGeometry:
    """The plan geometry of one grid system of ``model``."""
    gs = model.grid_systems.get(name)
    lines = [g for g in model.grid_lines if (g.system or "G1") == name]
    systems = {name: gs} if gs is not None else {}
    return grid_geometry(SimpleNamespace(grid_systems=systems, grid_lines=lines))


def lines_summary(model, name: str) -> str:
    gs = model.grid_systems.get(name)
    lines = [g for g in model.grid_lines if (g.system or "G1") == name]
    count = {d: sum(1 for g in lines if g.direction == d) for d in ("X", "Y", "T", "R", "General")}
    if gs is not None and gs.type == CYLINDRICAL:
        text = f"{count['T']} radial, {count['R']} circles"
    else:
        text = f"{count['X']} X, {count['Y']} Y"
    if count["General"]:
        text += f", {count['General']} general"
    return text


class GridPreview(QtWidgets.QWidget):
    """Plan drawing of a grid system: its lines and arcs with bubbles, the grid points,
    the spacings of a Cartesian system and its origin and local axes; the other
    systems of the model faintly behind it."""

    def __init__(self, parent=None, bubbles: bool = True):
        super().__init__(parent)
        self.setMinimumSize(320, 280)
        self.geo: Optional[GridGeometry] = None
        self.others: Optional[GridGeometry] = None
        self.gs = None
        self.lines = []
        self.bubbles = bubbles
        self.show_others = True
        self.message = "No grid system selected"
        self.units = ""

    def set_system(self, model, name: Optional[str]):
        if name is None or model is None:
            self.geo, self.others, self.gs, self.lines = None, None, None, []
        else:
            self.geo = system_geometry(model, name)
            full = grid_geometry(model)
            self.others = GridGeometry()
            self.others.lines = [l for l in full.lines if l[3] != name]
            self.others.arcs = [a for a in full.arcs if a[5] != name]
            self.gs = model.grid_systems.get(name)
            self.lines = [g for g in model.grid_lines if (g.system or "G1") == name]
            try:
                self.units = model.units.label("length")
            except Exception:  # noqa: BLE001 - units without labels
                self.units = ""
            self.message = "" if self.geo.lines or self.geo.arcs else "This grid system has no lines"
        self.update()

    # -- painting
    def _fit(self, rect: QtCore.QRectF) -> Optional[Tuple[float, float, float]]:
        geo = self.geo
        if geo is None or geo.extent is None:
            return None
        x0, x1, y0, y1 = geo.extent
        w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
        pad = 34.0 if self.bubbles else 3.0
        s = min((rect.width() - 2 * pad) / w, (rect.height() - 2 * pad) / h)
        if not math.isfinite(s) or s <= 0:
            return None
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        return s, rect.center().x() - s * cx, rect.center().y() + s * cy  # y up

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.color(QtGui.QPalette.Base))
        paint_grid(p, QtCore.QRectF(self.rect()), self, pal)
        p.end()


def paint_grid(p: QtGui.QPainter, rect: QtCore.QRectF, view: GridPreview, pal: QtGui.QPalette):
    """Draw ``view``'s system into ``rect`` (also used for the small icons)."""
    text_col = pal.color(QtGui.QPalette.Text)
    fit = view._fit(rect)
    if fit is None:
        if view.bubbles:
            p.setPen(pal.color(QtGui.QPalette.PlaceholderText))
            p.drawText(rect, QtCore.Qt.AlignCenter, view.message or "No grid lines")
        return
    s, tx, ty = fit

    def P(xy) -> QtCore.QPointF:
        return QtCore.QPointF(tx + s * xy[0], ty - s * xy[1])

    p.save()
    p.setClipRect(rect)
    # the other systems, faint and dashed
    if view.others is not None and view.show_others and view.bubbles:
        faint = QtGui.QColor(text_col)
        faint.setAlpha(55)
        pen = QtGui.QPen(faint, 1.0, QtCore.Qt.DashLine)
        p.setPen(pen)
        for poly in view.others.polylines(8.0):
            p.drawPolyline(QtGui.QPolygonF([P(q) for q in poly]))
    geo = view.geo
    width = 1.4 if view.bubbles else 1.0
    pen = QtGui.QPen(ACCENT, width)
    pen.setCosmetic(True)
    if view.bubbles:
        pen.setStyle(QtCore.Qt.DashDotLine)
    p.setPen(pen)
    for poly in geo.polylines(4.0):
        p.drawPolyline(QtGui.QPolygonF([P(q) for q in poly]))
    if not view.bubbles:
        p.restore()
        return
    # grid points
    dot = QtGui.QColor(ACCENT).darker(120)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(dot)
    for q in geo.points:
        p.drawEllipse(P(q), 2.4, 2.4)
    # spacings of a Cartesian system, along its first X / Y line
    gs = view.gs
    if gs is not None and gs.type != CYLINDRICAL:
        _spacings(p, view, P, text_col)
    # origin and local axes
    if gs is not None:
        o = P(tuple(gs.origin or (0.0, 0.0)))
        L = 26.0
        for ang, col, lab in ((0.0, QtGui.QColor(220, 70, 60), "x"), (90.0, QtGui.QColor(60, 170, 80), "y")):
            t = math.radians((gs.rotation or 0.0) + ang)
            e = QtCore.QPointF(o.x() + L * math.cos(t), o.y() - L * math.sin(t))
            p.setPen(QtGui.QPen(col, 1.8))
            p.drawLine(o, e)
            p.drawText(QtCore.QRectF(e.x() - 8 + 6 * math.cos(t), e.y() - 8 - 6 * math.sin(t), 16, 16),
                       QtCore.Qt.AlignCenter, lab)
        p.setPen(QtGui.QPen(text_col, 1.0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(o, 3.0, 3.0)
    # bubbles
    f = p.font()
    f.setPointSizeF(max(7.0, f.pointSizeF() * 0.85))
    p.setFont(f)
    fm = QtGui.QFontMetricsF(f)
    bub_fill = pal.color(QtGui.QPalette.Base)
    for end, d, name in geo.bubbles:
        r = max(10.0, 0.5 * fm.horizontalAdvance(name) + 4.0)
        c = P(end)
        c = QtCore.QPointF(c.x() + d[0] * r, c.y() - d[1] * r)
        p.setPen(QtGui.QPen(ACCENT, 1.2))
        p.setBrush(bub_fill)
        p.drawEllipse(c, r, r)
        p.setPen(text_col)
        p.drawText(QtCore.QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), QtCore.Qt.AlignCenter, name)
    p.restore()


def _spacings(p, view, P, text_col):
    gs, lines, geo = view.gs, view.lines, view.geo
    xs = sorted({g.ordinate for g in lines if g.direction == "X"})
    ys = sorted({g.ordinate for g in lines if g.direction == "Y"})
    if not xs or not ys:
        return
    col = QtGui.QColor(text_col)
    col.setAlpha(170)
    p.setPen(col)
    f = p.font()
    fm = QtGui.QFontMetricsF(f)
    off = 0.45 * geo.ext
    for vals, along in ((xs, "x"), (ys, "y")):
        for a, b in zip(vals, vals[1:]):
            mid = 0.5 * (a + b)
            q = to_global(gs, mid, ys[0] - off) if along == "x" else to_global(gs, xs[0] - off, mid)
            c = P(q)
            t = f"{b - a:g}"
            w = fm.horizontalAdvance(t) + 6
            p.drawText(QtCore.QRectF(c.x() - w / 2, c.y() - 9, w, 18), QtCore.Qt.AlignCenter, t)


def grid_icon(model, name: str, size: int = 22) -> QtGui.QIcon:
    view = GridPreview(bubbles=False)
    view.set_system(model, name)
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    paint_grid(p, QtCore.QRectF(1, 1, size - 2, size - 2), view, view.palette())
    p.end()
    return QtGui.QIcon(pm)


class TowerDialog(QtWidgets.QDialog):
    """ETABS "Quick Story Definition for New Tower": the tower's name, its stories (number,
    typical and bottom story height, base elevation) and its grid - a copy of another
    tower's grid systems moved by dX / dY, or a new Cartesian grid."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("New Tower")
        u = model.units.label("length")
        lay = QtWidgets.QVBoxLayout(self)
        f = QtWidgets.QFormLayout()
        k = len(model.tower_names()) + 1
        while f"T{k}" in model.towers:
            k += 1
        self.name = QtWidgets.QLineEdit(f"T{k}")
        self.n = QtWidgets.QSpinBox()
        self.n.setRange(1, 300)
        self.n.setValue(4)
        self.typical = QtWidgets.QLineEdit("3")
        self.bottom = QtWidgets.QLineEdit("3")
        self.base = QtWidgets.QLineEdit("0")
        self.prefix = QtWidgets.QLineEdit("")
        self.prefix.setPlaceholderText("default: <tower>-Story")
        f.addRow("Tower name", self.name)
        f.addRow("Number of stories", self.n)
        f.addRow(f"Typical story height [{u}]", self.typical)
        f.addRow(f"Bottom story height [{u}]", self.bottom)
        f.addRow(f"Base elevation [{u}]", self.base)
        f.addRow("Story names", self.prefix)
        lay.addLayout(f)
        g = QtWidgets.QGroupBox("Grid")
        gl = QtWidgets.QGridLayout(g)
        self.copy_grid = QtWidgets.QRadioButton("Copy the grid systems of tower")
        self.src = QtWidgets.QComboBox()
        self.src.addItems(model.tower_names())
        self.dx, self.dy = QtWidgets.QLineEdit("0"), QtWidgets.QLineEdit("0")
        self.new_grid = QtWidgets.QRadioButton("New Cartesian grid")
        self.xs, self.ys = QtWidgets.QLineEdit("6, 6, 6"), QtWidgets.QLineEdit("6, 6")
        self.ox, self.oy = QtWidgets.QLineEdit("0"), QtWidgets.QLineEdit("0")
        self.new_grid.setChecked(True)
        gl.addWidget(self.copy_grid, 0, 0, 1, 2)
        gl.addWidget(self.src, 0, 2)
        gl.addWidget(QtWidgets.QLabel(f"moved by dX / dY [{u}]"), 1, 0)
        gl.addWidget(self.dx, 1, 1)
        gl.addWidget(self.dy, 1, 2)
        gl.addWidget(self.new_grid, 2, 0, 1, 3)
        gl.addWidget(QtWidgets.QLabel(f"X spacings [{u}]"), 3, 0)
        gl.addWidget(self.xs, 3, 1, 1, 2)
        gl.addWidget(QtWidgets.QLabel(f"Y spacings [{u}]"), 4, 0)
        gl.addWidget(self.ys, 4, 1, 1, 2)
        gl.addWidget(QtWidgets.QLabel(f"Origin X / Y [{u}]"), 5, 0)
        gl.addWidget(self.ox, 5, 1)
        gl.addWidget(self.oy, 5, 2)
        lay.addWidget(g)
        note = QtWidgets.QLabel("Story names are unique in the model: the stories of the new tower are "
                                "named after it.  Every object belongs to the tower of its story; use a "
                                "different diaphragm for each tower's floors.")
        note.setWordWrap(True)
        lay.addWidget(note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @staticmethod
    def _spacings(text: str) -> List[float]:
        from .templates import parse_spacings
        return [float(v) for v in parse_spacings(text)]

    def values(self) -> dict:
        """The tower data (raises ValueError on bad input)."""
        name = self.name.text().strip()
        if not name:
            raise ValueError("Give the tower a name")
        if name in self.model.towers or (not self.model.towers and name == self.model.DEFAULT_TOWER):
            raise ValueError(f"Tower {name!r} exists already")
        d = {"name": name, "n": self.n.value(), "typical": float(self.typical.text()),
             "bottom": float(self.bottom.text()), "base": float(self.base.text() or 0),
             "prefix": self.prefix.text().strip() or None, "copy": self.copy_grid.isChecked(),
             "src": self.src.currentText(), "dx": float(self.dx.text() or 0), "dy": float(self.dy.text() or 0)}
        if not d["copy"]:
            d["xs"], d["ys"] = self._spacings(self.xs.text()), self._spacings(self.ys.text())
            d["ox"], d["oy"] = float(self.ox.text() or 0), float(self.oy.text() or 0)
        if d["typical"] <= 0 or d["bottom"] <= 0:
            raise ValueError("Story heights must be greater than zero")
        return d

    def accept(self):
        try:
            self.values()
        except (ValueError, ModelError) as exc:
            QtWidgets.QMessageBox.warning(self, "New tower", str(exc))
            return
        super().accept()


def add_tower_from(model, d: dict) -> None:
    """Make the tower of :class:`TowerDialog` values ``d`` in ``model``."""
    from ..model.objects import GridLine, GridSystem
    name = d["name"]
    model.add_tower_stories(name, d["n"], d["typical"], d["bottom"], d["base"], d["prefix"])
    if d["copy"]:
        src = d["src"]
        for sname in model.grid_systems_of(src):
            gs = model.grid_systems[sname]
            n, k = f"{name}-{sname}", 2
            while n in model.grid_systems:
                n, k = f"{name}-{sname}{k}", k + 1
            o = gs.origin or (0.0, 0.0)
            import dataclasses as _dc
            model.grid_systems[n] = _dc.replace(gs, name=n, origin=(o[0] + d["dx"], o[1] + d["dy"]), tower=name)
            model.grid_lines += [_dc.replace(g, system=n) for g in model.grid_lines
                                 if g.system == sname and g.direction != "Z"]
    else:
        n, k = f"{name}-G1", 2
        while n in model.grid_systems:
            n, k = f"{name}-G{k}", k + 1
        model.grid_systems[n] = GridSystem(name=n, origin=(d["ox"], d["oy"]), tower=name)
        x = 0.0
        letters = [chr(65 + i) if i < 26 else f"A{chr(65 + i - 26)}" for i in range(len(d["xs"]) + 1)]
        for i, lab in enumerate(letters):
            model.grid_lines.append(GridLine(system=n, name=f"{name}-{lab}", direction="X", ordinate=x))
            if i < len(d["xs"]):
                x += d["xs"][i]
        y = 0.0
        for i in range(len(d["ys"]) + 1):
            model.grid_lines.append(GridLine(system=n, name=f"{name}-{i + 1}", direction="Y", ordinate=y))
            if i < len(d["ys"]):
                y += d["ys"][i]
    model.sync_z_grid()  # SAP2000 mode: the new tower's Z grid lines


class GridSystemsManager(QtWidgets.QDialog):
    """Towers, Stories and Grid Systems: the towers (ETABS "Allow Multiple Towers"), the
    grid systems and stories of the selected tower, the selected system drawn; Add /
    Copy / Modify / Delete for towers and grid systems."""

    COLUMNS = ("Name", "Type", "Lines", "Origin", "Rotation")
    TOWER_COLUMNS = ("Tower", "Stories", "Base", "Top", "Grid systems")

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setWindowTitle("Towers, Stories and Grid Systems")
        self.resize(1160, 700)
        lay = QtWidgets.QHBoxLayout(self)
        # --- lists
        left = QtWidgets.QVBoxLayout()
        self.allow = QtWidgets.QCheckBox("Allow multiple towers")
        self.allow.setToolTip("Several buildings in one model, each with its own stories and grid systems "
                              "(ETABS Options > Allow Multiple Towers)")
        self.allow.toggled.connect(self._allow_toggled)
        left.addWidget(self.allow)
        self.tbox = QtWidgets.QGroupBox("Towers")
        tl = QtWidgets.QVBoxLayout(self.tbox)
        self.towers = QtWidgets.QTableWidget(0, len(self.TOWER_COLUMNS))
        self.towers.setHorizontalHeaderLabels(self.TOWER_COLUMNS)
        self.towers.verticalHeader().setVisible(False)
        self.towers.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.towers.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.towers.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.towers.setAlternatingRowColors(True)
        self.towers.horizontalHeader().setStretchLastSection(True)
        self.towers.itemSelectionChanged.connect(self._tower_selected)
        self.towers.itemDoubleClicked.connect(lambda *_: self.modify_stories())
        tl.addWidget(self.towers)
        left.addWidget(self.tbox, 2)
        gbox = QtWidgets.QGroupBox("Grid Systems")
        self.gbox = gbox
        gl = QtWidgets.QVBoxLayout(gbox)
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setIconSize(QtCore.QSize(22, 22))
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        for j in range(len(self.COLUMNS)):
            hh.setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch if j == 2 else QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.itemDoubleClicked.connect(lambda *_: self.modify())
        gl.addWidget(self.table, 1)
        left.addWidget(gbox, 3)
        sbox = QtWidgets.QGroupBox("Stories")
        self.sbox = sbox
        sl = QtWidgets.QVBoxLayout(sbox)
        self.levels_label = QtWidgets.QLabel("")
        sl.addWidget(self.levels_label)
        self.stories = QtWidgets.QTableWidget(0, 3)
        self.stories.verticalHeader().setVisible(False)
        self.stories.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.stories.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.stories.setAlternatingRowColors(True)
        self.stories.horizontalHeader().setStretchLastSection(True)
        self.stories.itemDoubleClicked.connect(lambda *_: self.modify_stories())
        sl.addWidget(self.stories, 1)
        left.addWidget(sbox, 3)
        lay.addLayout(left, 5)
        # --- buttons
        mid = QtWidgets.QVBoxLayout()
        self.tbtn = QtWidgets.QGroupBox("Towers")
        tb = QtWidgets.QVBoxLayout(self.tbtn)
        self.btn_tower_add = QtWidgets.QPushButton("Add New Tower...")
        self.btn_tower_copy = QtWidgets.QPushButton("Add Copy of Tower...")
        self.btn_tower_rename = QtWidgets.QPushButton("Rename Tower...")
        self.btn_tower_delete = QtWidgets.QPushButton("Delete Tower")
        for b, fn, tip in ((self.btn_tower_add, self.add_tower, "A new tower with its stories and grid"),
                           (self.btn_tower_copy, self.copy_tower, "A new tower with the stories and grid systems "
                                                                  "of the selected one (no objects)"),
                           (self.btn_tower_rename, self.rename_tower, "Rename the selected tower"),
                           (self.btn_tower_delete, self.delete_tower, "Delete the selected tower with its stories "
                                                                      "and grid systems (it must have no objects)")):
            b.clicked.connect(lambda _=False, fn=fn: fn())
            b.setToolTip(tip)
            tb.addWidget(b)
        mid.addWidget(self.tbtn)
        box = QtWidgets.QGroupBox("Grid systems")
        bl = QtWidgets.QVBoxLayout(box)
        self.btn_add = QtWidgets.QPushButton("Add New Grid System...")
        menu = QtWidgets.QMenu(self.btn_add)
        menu.addAction("Cartesian (X / Y lines)", lambda: self.add("Cartesian"))
        menu.addAction("Cylindrical (radial lines and circles)", lambda: self.add("Cylindrical"))
        self.btn_add.setMenu(menu)
        self.btn_copy = QtWidgets.QPushButton("Add Copy of Grid System...")
        self.btn_modify = QtWidgets.QPushButton("Modify / Show Grid System...")
        self.btn_delete = QtWidgets.QPushButton("Delete Grid System")
        bl.addWidget(self.btn_add)
        for b, fn, tip in ((self.btn_copy, self.copy_system, "A new system starting from the selected one"),
                           (self.btn_modify, self.modify, "Edit the lines, origin and rotation (double-click a row)"),
                           (self.btn_delete, self.delete, "Delete the selected system and its lines")):
            b.clicked.connect(lambda _=False, fn=fn: fn())
            b.setToolTip(tip)
            bl.addWidget(b)
        mid.addWidget(box)
        box2 = QtWidgets.QGroupBox("Stories")
        bl2 = QtWidgets.QVBoxLayout(box2)
        self.btn_stories = QtWidgets.QPushButton("Modify / Show Stories...")
        self.btn_stories.clicked.connect(lambda _=False: self.modify_stories())
        bl2.addWidget(self.btn_stories)
        mid.addWidget(box2)
        mid.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.accept)
        mid.addWidget(bb)
        lay.addLayout(mid, 0)
        # --- drawing
        right = QtWidgets.QVBoxLayout()
        head = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("")
        f = self.title.font()
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 1.15)
        self.title.setFont(f)
        head.addWidget(self.title, 1)
        self.others = QtWidgets.QCheckBox("Show other systems")
        self.others.setChecked(True)
        self.others.toggled.connect(self._toggle_others)
        head.addWidget(self.others)
        right.addLayout(head)
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        fl = QtWidgets.QVBoxLayout(frame)
        fl.setContentsMargins(1, 1, 1, 1)
        self.preview = GridPreview()
        fl.addWidget(self.preview)
        right.addWidget(frame, 1)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        right.addWidget(self.info)
        lay.addLayout(right, 5)
        QtGui.QShortcut(QtGui.QKeySequence.Delete, self.table, activated=self.delete)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, activated=self.copy_system)
        self.refresh()

    @property
    def model(self):
        return self.doc.model

    def done(self, r):
        # the lists are not rebuilt while the window is torn down (selection signals of
        # tables being deleted would reach a half-deleted window)
        for sig in (self.towers.itemSelectionChanged, self.table.itemSelectionChanged):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        super().done(r)

    def _edit(self, fn) -> bool:
        try:
            self.doc.edit(fn)
            return True
        except (ModelError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Towers, Stories and Grid Systems", str(exc))
            return False

    # -- towers
    def selected_tower(self) -> Optional[str]:
        """The tower whose systems and stories are listed (None: one tower)."""
        m = self.model
        if not m.multi_tower:
            return None
        rows = {i.row() for i in self.towers.selectedIndexes()}
        if len(rows) == 1:
            it = self.towers.item(rows.pop(), 0)
            if it is not None and it.text() in m.towers:
                return it.text()
        return m.default_tower()

    def _refresh_towers(self, select: Optional[str] = None):
        m = self.model
        allowed = bool(m.towers)
        self.allow.blockSignals(True)
        self.allow.setChecked(allowed)
        self.allow.blockSignals(False)
        self.tbox.setVisible(allowed)
        self.tbtn.setVisible(allowed)
        keep = select if select is not None else (self.selected_tower() if self.towers.rowCount() else None)
        self.towers.blockSignals(True)
        self.towers.setRowCount(0)
        names = m.tower_names() if allowed else []
        self.towers.setRowCount(len(names))
        for i, t in enumerate(names):
            ss = m.stories_of(t) if m.multi_tower else m.stories_bottom_up()
            gs = m.grid_systems_of(t) if m.multi_tower else list(m.grid_systems)
            row = (t, str(max(len(ss) - 1, 0)), f"{ss[0].elevation:g}" if ss else "-",
                   f"{ss[-1].elevation:g}" if ss else "-", ", ".join(gs) or "(none)")
            for j, txt in enumerate(row):
                it = QtWidgets.QTableWidgetItem(txt)
                if j in (1, 2, 3):
                    it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.towers.setItem(i, j, it)
        self.towers.resizeColumnsToContents()
        self.towers.horizontalHeader().setStretchLastSection(True)
        if names:
            self.towers.selectRow(names.index(keep) if keep in names else 0)
        self.towers.blockSignals(False)
        self.btn_tower_delete.setEnabled(m.multi_tower)
        self.btn_tower_rename.setEnabled(allowed)
        self.btn_tower_copy.setEnabled(allowed)
        tw = self.selected_tower()
        self.gbox.setTitle("Grid Systems" + (f" - tower {tw}" if tw else ""))
        self.sbox.setTitle("Stories" + (f" - tower {tw}" if tw else ""))

    def _tower_selected(self):
        tw = self.selected_tower()
        self.gbox.setTitle("Grid Systems" + (f" - tower {tw}" if tw else ""))
        self.sbox.setTitle("Stories" + (f" - tower {tw}" if tw else ""))
        self.refresh(towers=False)

    def _allow_toggled(self, on: bool):
        m = self.model
        if on:
            self._edit(lambda mm: mm.allow_towers(True))
        else:
            keep = m.default_tower()
            if m.multi_tower:
                names = m.tower_names()
                keep, ok = QtWidgets.QInputDialog.getItem(
                    self, "Do not allow multiple towers",
                    "Combine every tower into tower (stories at the same elevation are merged):",
                    names, 0, False)
                if not ok:
                    self._refresh_towers()
                    return
            self._edit(lambda mm: mm.allow_towers(False, keep))
        self.refresh()

    def add_tower(self):
        d = TowerDialog(self.model, self)
        if d.exec() != QtWidgets.QDialog.Accepted:
            return
        vals = d.values()
        if self._edit(lambda mm: add_tower_from(mm, vals)):
            self.refresh(tower=vals["name"])

    def copy_tower(self):
        src = self.selected_tower() or self.model.default_tower()
        name, ok = QtWidgets.QInputDialog.getText(self, "Add copy of tower", f"New tower (a copy of {src}'s stories "
                                                  "and grid systems):", text=f"{src}_copy")
        name = (name or "").strip()
        if not ok or not name:
            return
        u = self.model.units.label("length")
        text, ok = QtWidgets.QInputDialog.getText(self, "Add copy of tower", f"Move its grid by dX, dY [{u}]:",
                                                  text="0, 0")
        if not ok:
            return
        try:
            dx, dy = (float(v) for v in text.replace(";", ",").split(",")[:2])
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Add copy of tower", "Give dX, dY as two numbers")
            return
        if self._edit(lambda mm: mm.copy_tower(src, name, dx, dy)):
            self.refresh(tower=name)

    def rename_tower(self):
        old = self.selected_tower() or self.model.default_tower()
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename tower", "Name", text=old)
        new = (new or "").strip()
        if ok and new and new != old and self._edit(lambda mm: mm.rename_tower(old, new)):
            self.refresh(tower=new)

    def delete_tower(self):
        t = self.selected_tower()
        if t is None:
            return
        if QtWidgets.QMessageBox.question(self, "Delete tower", f"Delete tower {t} with its stories and grid "
                                          "systems?") != QtWidgets.QMessageBox.Yes:
            return
        if self._edit(lambda mm: mm.delete_tower(t)):
            self.refresh()

    # -- lists
    def system_names(self) -> List[str]:
        m = self.model
        tw = self.selected_tower()
        if tw is not None:
            return m.grid_systems_of(tw)
        names = list(m.grid_systems)
        for g in m.grid_lines:
            if g.direction != "Z" and (g.system or "G1") not in names:
                names.append(g.system or "G1")
        return names

    def refresh(self, select: Optional[str] = None, tower: Optional[str] = None, towers: bool = True):
        if towers:
            self._refresh_towers(tower)
        m = self.model
        keep = select if select is not None else self.selected_name()
        self.table.setRowCount(0)
        names = self.system_names()
        self.table.setRowCount(len(names))
        for i, name in enumerate(names):
            gs = m.grid_systems.get(name)
            kind = gs.type if gs is not None and gs.type == CYLINDRICAL else "Cartesian"
            o = tuple(gs.origin or (0.0, 0.0)) if gs is not None else (0.0, 0.0)
            rot = float(gs.rotation or 0.0) if gs is not None else 0.0
            texts = (name, kind, lines_summary(m, name), f"({o[0]:g}, {o[1]:g})", f"{rot:g}°")
            for j, t in enumerate(texts):
                it = QtWidgets.QTableWidgetItem(t)
                if j == 0:
                    it.setIcon(grid_icon(m, name))
                self.table.setItem(i, j, it)
        if keep in names:
            self.table.selectRow(names.index(keep))
        elif names:
            self.table.selectRow(0)
        self._refresh_stories()
        self._selected()

    def _refresh_stories(self):
        m = self.model
        xyz = m.grid_mode == "xyz"
        try:
            u = m.units.label("length")
        except Exception:  # noqa: BLE001
            u = ""
        tw = self.selected_tower()
        if xyz:
            self.levels_label.setText("Levels from the Z grid lines (SAP2000 mode)")
            levels = list(reversed(m.levels()))
            self.stories.setColumnCount(2)
            self.stories.setHorizontalHeaderLabels(["Z line", f"Z [{u}]"])
            rows = [(n, f"{z:g}") for n, z in levels]
        else:
            self.levels_label.setText("Stories (ETABS mode), top down" + (f" - tower {tw}" if tw else ""))
            ss = list(reversed(m.stories_of(tw) if tw else m.stories_bottom_up()))
            self.stories.setColumnCount(3)
            self.stories.setHorizontalHeaderLabels(["Story", f"Height [{u}]", f"Elevation [{u}]"])
            rows = [(s.name, "" if k == len(ss) - 1 else f"{s.elevation - ss[k + 1].elevation:g}", f"{s.elevation:g}")
                    for k, s in enumerate(ss)]
        self.stories.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, t in enumerate(row):
                it = QtWidgets.QTableWidgetItem(t)
                if j:
                    it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.stories.setItem(i, j, it)
        self.stories.resizeColumnsToContents()
        self.stories.horizontalHeader().setStretchLastSection(True)

    def selected_name(self) -> Optional[str]:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None
        it = self.table.item(rows.pop(), 0)
        return it.text() if it is not None else None

    def _toggle_others(self, on: bool):
        self.preview.show_others = on
        self.preview.update()

    def _selected(self):
        name = self.selected_name()
        for b in (self.btn_copy, self.btn_modify, self.btn_delete):
            b.setEnabled(name is not None)
        self.preview.set_system(self.model if name else None, name)
        if name is None:
            self.title.setText("")
            self.info.setText("")
            return
        m = self.model
        gs = m.grid_systems.get(name)
        kind = gs.type if gs is not None and gs.type == CYLINDRICAL else "Cartesian"
        tw = f"  (tower {m.tower_of_system(name)})" if m.multi_tower else ""
        self.title.setText(f"{name}  -  {kind}{tw}")
        lines = [g for g in m.grid_lines if (g.system or "G1") == name]
        if kind == CYLINDRICAL:
            angs = sorted(g.ordinate for g in lines if g.direction == "T")
            rads = sorted(g.ordinate for g in lines if g.direction == "R")
            text = (f"Radial lines at {', '.join(f'{a:g}°' for a in angs) or '-'};  "
                    f"circles R = {', '.join(f'{r:g}' for r in rads) or '-'}")
        else:
            xs = sorted(g.ordinate for g in lines if g.direction == "X")
            ys = sorted(g.ordinate for g in lines if g.direction == "Y")
            text = (f"X lines at {', '.join(f'{x:g}' for x in xs) or '-'};  "
                    f"Y lines at {', '.join(f'{y:g}' for y in ys) or '-'}")
        self.info.setText(text + f"  [{self.preview.units}]" if self.preview.units else text)

    # -- actions
    def _editor(self, **kw):
        from .dialogs import GridStoriesDialog
        before = set(self.system_names())
        tw = self.selected_tower()
        d = GridStoriesDialog(self.doc, self, tower=tw, **kw)
        ok = d.exec() == QtWidgets.QDialog.Accepted
        current = getattr(d, "_current", None)
        after = self.system_names()
        new = [n for n in after if n not in before]
        self.refresh(select=(new[0] if new else current) if ok else None, tower=tw)
        return ok

    def add(self, kind: str):
        self._editor(add=kind)

    def copy_system(self):
        name = self.selected_name()
        if name:
            self._editor(copy_of=name)

    def modify(self):
        name = self.selected_name()
        if name:
            self._editor(system=name)

    def modify_stories(self):
        self._editor(system=self.selected_name(), show_levels=True)

    def delete(self):
        name = self.selected_name()
        if name is None:
            return
        if len(self.system_names()) <= 1:
            QtWidgets.QMessageBox.information(self, "Grid systems", "The model keeps at least one grid system"
                                              + (" in each tower" if self.model.multi_tower else ""))
            return
        n = sum(1 for g in self.model.grid_lines if (g.system or "G1") == name)
        if QtWidgets.QMessageBox.question(self, "Delete grid system",
                                          f"Delete grid system {name} and its {n} line(s)?") != QtWidgets.QMessageBox.Yes:
            return

        def edit(m):
            m.grid_systems.pop(name, None)
            m.grid_lines = [g for g in m.grid_lines if g.direction == "Z" or (g.system or "G1") != name]

        self._edit(edit)
        self.refresh()
