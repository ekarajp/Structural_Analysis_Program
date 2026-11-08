"""Define > Frame Sections: the section manager (list, Add / Copy / Modify /
Delete, a drawing of the selected section with its dimensions and properties),
the section-type chooser, the parametric shape form, General / library sections.
The Section Designer (any shape drawn like in CAD) is in :mod:`.section_designer`.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..display_units import DisplayUnits
from ..model import FrameSection, ModelError
from ..model.properties import FrameModifiers
from ..model.section_shapes import (COLD, DESIGNER, FLANGE_SLOPES, GENERAL, SHAPES, SLOPED, SectionProps, Shape, check_dims, dims_of,
                                    is_composite, material_areas, section_outline_cached, section_parts,
                                    section_properties, section_ratios, shape_of, shape_outline, shape_properties)
from .document import Document

COMPOSITE_RECT = "Composite Rectangular"
COMPOSITE_CIRCLE = "Composite Circular"

# the "Add" chooser: (label, shape key, overrides of the default dimensions, material type)
CHOICES: List[Tuple[str, List[Tuple[str, str, dict, str]]]] = [
    ("Concrete", [
        ("Rectangular", "Rectangular", {}, "Concrete"),
        ("Circular", "Circular", {}, "Concrete"),
        ("T-beam", "Concrete Tee", {}, "Concrete"),
    ]),
    ("Steel - hot-rolled", [
        ("I / H", "I", {}, "Steel"),
        ("Channel", "Channel", {}, "Steel"),
        ("Tee", "Tee", {}, "Steel"),
        ("Angle", "Angle", {}, "Steel"),
        ("Double angle", "DoubleAngle", {}, "Steel"),
        ("Double channel", "DoubleChannel", {}, "Steel"),
        ("Box / Tube", "Box", {}, "Steel"),
        ("Pipe", "Pipe", {}, "Steel"),
        ("Plate / flat bar", "Rectangular", {"t3": 0.2, "t2": 0.012}, "Steel"),
        ("Round bar", "Circular", {"t3": 0.025}, "Steel"),
    ]),
    ("Steel - cold-formed", [
        ("Lipped C", "Lipped Channel", {}, "Steel"),
        ("Lipped Z", "Lipped Z", {}, "Steel"),
        ("Channel C", "Cold Channel", {}, "Steel"),
        ("Z", "Cold Z", {}, "Steel"),
        ("Angle L", "Cold Angle", {}, "Steel"),
        ("Hat", "Hat", {}, "Steel"),
    ]),
    ("Composite (SRC)", [
        ("SRC rectangular", COMPOSITE_RECT, {}, "Concrete"),
        ("SRC circular", COMPOSITE_CIRCLE, {}, "Concrete"),
    ]),
    ("Other", [
        ("From catalog", "library", {}, "Steel"),
        ("General (A, I)", GENERAL, {}, "Concrete"),
        ("Section Designer", DESIGNER, {}, "Concrete"),
    ]),
]
# shapes kept for older models / import but not offered in this version
HIDDEN_SHAPES = {"Concrete L", "Concrete Box", "Concrete Pipe"}

STEEL_TYPES = {"Steel", "ColdFormed", "Cold-formed", "Aluminum"}


def calc(text: str) -> Optional[float]:
    """A number or a simple arithmetic expression ("30", "60-12", "2*15")."""
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError(text)
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return ev(ast.parse(text.replace(",", "."), mode="eval").body)
    except Exception:
        return None


def fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "-"
    v = float(v)
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e9:
        d = max(0, digits - 1 - int(math.floor(math.log10(a))))
        t = f"{v:.{d}f}"
        return t.rstrip("0").rstrip(".") if "." in t else t
    return f"{v:.{digits - 1}e}"


def section_group(sec, model) -> str:
    """Concrete | Steel | Composite | Other (by shape, then material)."""
    if is_composite(getattr(sec, "geometry", None)):
        return "Composite"
    if sec.shape == DESIGNER or getattr(sec, "geometry", None):
        return "Other"
    mat = model.materials.get(sec.material)
    if mat is not None and mat.type in STEEL_TYPES:
        return "Steel"
    sh = shape_of(sec)
    if sh is not None:
        return sh.group if mat is None else ("Concrete" if mat.type == "Concrete" else sh.group)
    return "Other"


def type_label(sec) -> str:
    if sec.nonprismatic:
        return "Nonprismatic"
    kind = (sec.params or {}).get("composite")
    if kind and getattr(sec, "geometry", None):
        return "Composite (WF in " + ("circular" if kind == "circle" else "rectangular") + " concrete)"
    if getattr(sec, "geometry", None) or sec.shape == DESIGNER:
        return "Section Designer"
    sh = shape_of(sec)
    lab = sh.label if sh else (sec.shape or "General")
    if getattr(sec, "source", ""):
        lab += f" ({sec.source})"
    return lab


def dims_text(sec, du: DisplayUnits, steel_mm: bool = False) -> str:
    """Short size text in the section-dimension display unit, e.g. "30 x 60",
    "bf 120, d 60", "D 50"."""
    k, u = du.factor("section_dim"), du.label("section_dim")
    sh = shape_of(sec)
    f = lambda x: fmt(x * k, 4) if x else "-"
    comp = composite_dims(sec)
    if comp is not None:
        o, st = comp
        txt = f"D {f(o['t3'])}" if o["kind"] == "circle" else f"{f(o['t2'])} x {f(o['t3'])}"
        return f"{txt} {u} + I {f(st.get('t3'))} x {f(st.get('t2'))} {u}"
    if getattr(sec, "geometry", None):
        return f"{f(sec.t2)} x {f(sec.t3)} {u} (bounding box)"
    if sh is None:
        if sec.t3 and sec.t2:
            return f"{f(sec.t2)} x {f(sec.t3)} {u}"
        return f"A {fmt(sec.A * du.factor('section_area'))} {du.label('section_area')}"
    key = sh.key
    if key == "Rectangular":
        return f"{f(sec.t2)} x {f(sec.t3)} {u}"
    if key == "Circular":
        return f"D {f(sec.t3)} {u}"
    if key in ("Pipe", "Concrete Pipe"):
        return f"D {f(sec.t3)} x t {f(sec.tw)} {u}"
    if key in ("Concrete Tee", "Concrete L"):
        return f"bw {f(sec.tw)} x h {f(sec.t3)}, bf {f(sec.t2)} x hf {f(sec.tf)} {u}"
    if key in ("Lipped Channel", "Lipped Z", "Hat"):
        return f"{f(sec.t3)} x {f(sec.t2)} x {f(sec.lip)} x {f(sec.tw)} {u}" + (f", r {f(sec.r)}" if sec.r else "")
    parts = [f(sec.t3), f(sec.t2)]
    if sec.tw:
        parts.append(f(sec.tw))
    if sec.tf and key != "Box":
        parts.append(f(sec.tf))
    txt = " x ".join(parts) + f" {u}"
    if getattr(sec, "r", None):
        txt += f", r {f(sec.r)}" + (f" / {f(sec.r2)}" if getattr(sec, "r2", None) else "")
    return txt


def suggest_name(key: str, v: dict, existing: Sequence[str]) -> str:
    """Thai / common practice names: concrete in cm (B30x60, T30x60, L30x60),
    steel in mm (H300x150x6.5x9, C150x65x20x2.3, [200x80, L100x100x10 ...)."""
    cm = lambda x: fmt((x or 0) * 100, 3)
    mm = lambda x: fmt((x or 0) * 1000, 3)
    d, b, tf, tw = v.get("t3"), v.get("t2"), v.get("tf"), v.get("tw")
    names = {
        "Rectangular": f"B{cm(b)}x{cm(d)}", "Circular": f"D{cm(d)}",
        "Concrete Tee": f"T{cm(tw)}x{cm(d)}", "Concrete L": f"L{cm(tw)}x{cm(d)}",
        "Concrete Box": f"BOX{cm(b)}x{cm(d)}", "Concrete Pipe": f"HC{cm(d)}",
        "I": f"H{mm(d)}x{mm(b)}x{mm(tw)}x{mm(tf)}", "Channel": f"[{mm(d)}x{mm(b)}x{mm(tw)}x{mm(tf)}",
        "Tee": f"T{mm(d)}x{mm(b)}x{mm(tw)}x{mm(tf)}", "Angle": f"L{mm(d)}x{mm(b)}x{mm(tw)}",
        "DoubleAngle": f"2L{mm(d)}x{mm(b)}x{mm(tw)}", "DoubleChannel": f"2[{mm(d)}x{mm(b)}",
        "Box": f"BOX{mm(d)}x{mm(b)}x{mm(tw)}", "Pipe": f"PIPE{mm(d)}x{mm(tw)}",
        "Lipped Channel": f"C{mm(d)}x{mm(b)}x{mm(v.get('lip'))}x{mm(tw)}",
        "Lipped Z": f"Z{mm(d)}x{mm(b)}x{mm(v.get('lip'))}x{mm(tw)}",
        "Cold Channel": f"C{mm(d)}x{mm(b)}x{mm(tw)}", "Cold Z": f"Z{mm(d)}x{mm(b)}x{mm(tw)}",
        "Cold Angle": f"L{mm(d)}x{mm(b)}x{mm(tw)}", "Hat": f"HAT{mm(d)}x{mm(b)}x{mm(v.get('lip'))}x{mm(tw)}",
    }
    base = names.get(key, key)
    return unique_name(base, existing)


def unique_name(base: str, existing: Sequence[str]) -> str:
    existing = set(existing)
    if base not in existing:
        return base
    k = 2
    while f"{base}-{k}" in existing:
        k += 1
    return f"{base}-{k}"


def si_dims(key: str, overrides: dict, units) -> dict:
    """Default dimensions of a catalogue shape in the model's length unit."""
    L = 1.0 / units.factor("length")  # metres -> model length
    out = {}
    for dm in SHAPES[key].dims:
        x = overrides.get(dm.attr, dm.default)
        out[dm.attr] = None if x is None else x * L
    return out


def steel_material(units):
    """A structural steel (SS400, TIS 1227 / JIS G3101: Fy 245 MPa, Fu 400 MPa)
    in the model's units, for the first steel section of a model without steel."""
    from ..model.properties import Material
    st = lambda v: v / units.factor("stress")
    return Material(name="SS400", type="Steel", E=st(200e9), nu=0.3, weight_density=76972.86 / units.factor("weight_per_volume"),
                    alpha=1.17e-5 / units.factor("per_temperature"), fy=st(245e6), fu=st(400e6), damping=0.02,
                    notes="Structural steel SS400 (TIS 1227 / JIS G3101)")


def material_for(model, kind: str) -> str:
    mats = list(model.materials.values())
    for m in mats:
        if (kind == "Steel" and m.type in STEEL_TYPES) or (kind != "Steel" and m.type == kind):
            return m.name
    return mats[0].name if mats else ""


# ------------------------------------------------------------- composite
def composite_geometry(kind: str, outline: dict, steel: dict, steel_mat: str, rotate: bool = False) -> List[dict]:
    """Outline of a steel-reinforced concrete (SRC) column: the concrete
    rectangle / circle (the section's material) and the steel I (``steel_mat``)
    at its centre, strong axis about 3 unless ``rotate``."""
    from ..model.section_shapes import rect as _rect, shape_outline as _so
    if kind == "circle":
        conc = {"kind": "circle", "center": [0.0, 0.0], "radius": float(outline["t3"]) / 2, "hole": False,
                "role": "concrete", "dims": {"t3": float(outline["t3"])}}
    else:
        d, b = float(outline["t3"]), float(outline["t2"])
        conc = {"kind": "polygon", "points": _rect(-b / 2, -d / 2, b, d).tolist(), "hole": False, "role": "concrete",
                "dims": {"t3": d, "t2": b}}
    sol, _ = _so(SHAPES["I"], steel, False)
    P = sol[0]
    if rotate:
        P = np.column_stack([-P[:, 1], P[:, 0]])
    st = {"kind": "polygon", "points": P.tolist(), "hole": False, "material": steel_mat, "role": "steel",
          "dims": {k: float(v) for k, v in steel.items() if v is not None}, "rotate": bool(rotate)}
    return [conc, st]


def composite_dims(sec):
    """((outline kind / dims), steel I dims) of a composite section made by the
    composite form, or None."""
    kind = (getattr(sec, "params", None) or {}).get("composite")
    geom = getattr(sec, "geometry", None) or []
    if not kind or not geom:
        return None
    conc = next((g for g in geom if g.get("role") == "concrete"), None)
    st = next((g for g in geom if g.get("role") == "steel"), None)
    if conc is None or st is None:
        return None
    o = dict(conc.get("dims") or {})
    o["kind"] = kind
    return o, dict(st.get("dims") or {})


# ---------------------------------------------------------------- drawing
def _theme_colors(widget, steel: bool, hole_bg=None):
    pal = widget.palette()
    base = pal.color(QtGui.QPalette.Base)
    dark = base.lightness() < 128
    fill = QtGui.QColor("#7fa3c7" if steel else "#b9b4a8") if not dark else QtGui.QColor("#4f7396" if steel else "#7b766c")
    line = pal.color(QtGui.QPalette.Text)
    dim = QtGui.QColor("#1f6fd1") if not dark else QtGui.QColor("#6fb1ff")
    return base, fill, line, dim


def outline_path(solids, holes, tf) -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    path.setFillRule(QtCore.Qt.OddEvenFill)
    for P in list(solids) + list(holes):
        pts = [tf(x, y) for x, y in P]
        path.addPolygon(QtGui.QPolygonF(pts + pts[:1]))
    return path


class SectionPreview(QtWidgets.QWidget):
    """Drawing of a frame section: filled outline, local axes 2 / 3 through the
    centroid and dimension lines (overall width and depth, flange / web
    thickness) in the section-dimension display unit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 200)
        self.solids: List[np.ndarray] = []
        self.holes: List[np.ndarray] = []
        self.steel = False
        self.du: Optional[DisplayUnits] = None
        self.message = ""
        self.marks = True  # dimension lines
        self.parts = None  # [(steel?, solids, holes)] of a composite section

    def set_section(self, sec, model):
        self.du = DisplayUnits.of(model) if model is not None else None
        self.solids, self.holes, self.message, self.parts = [], [], "", None
        if sec is None:
            self.message = "No section selected"
        else:
            mat = model.materials.get(sec.material) if model is not None else None
            self.steel = mat is not None and mat.type in STEEL_TYPES
            lo = section_outline_cached(sec)
            if lo is None:
                from .extrude import section_profile  # General sections: the equivalent rectangle
                pieces = section_profile(sec)
                if pieces:
                    allp = np.vstack(pieces)
                    lo2, hi2 = allp.min(axis=0), allp.max(axis=0)
                    d, b = hi2[0] - lo2[0], hi2[1] - lo2[1]
                    from ..model.section_shapes import rect
                    self.solids = [rect(-b / 2, -d / 2, b, d)]
                    self.message = "General section: rectangle of the same A and I33"
                else:
                    self.message = "No outline (General section)"
            else:
                self.solids, self.holes = lo
                parts = section_parts(sec)
                if parts:
                    mats = model.materials if model is not None else {}
                    self.parts = [((mats.get(k).type in STEEL_TYPES) if k and k in mats else
                                   (k is not None) if not mats else self.steel, so, ho) for k, so, ho in parts]
        self.update()

    def set_outline(self, solids, holes, steel: bool, du: Optional[DisplayUnits], message: str = "", parts=None):
        self.solids, self.holes, self.steel, self.du, self.message = list(solids), list(holes), steel, du, message
        self.parts = parts
        self.update()

    # -- painting
    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        base, fill, line, dimc = _theme_colors(self, self.steel)
        p.fillRect(self.rect(), base)
        if not self.solids:
            p.setPen(line)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, self.message or "")
            p.end()
            return
        allp = np.vstack(self.solids)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        w, h = max(hi[0] - lo[0], 1e-12), max(hi[1] - lo[1], 1e-12)
        mx, my = 58, 40  # room for dimensions and axis labels
        W, H = max(self.width() - 2 * mx, 20), max(self.height() - 2 * my, 20)
        s = min(W / w, H / h)
        if w / h > 30 or h / w > 30:  # very slender (a plate): keep it visible
            s = min(W / w, H / h)
        cx, cy = self.width() / 2, self.height() / 2
        ox, oy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        tf = lambda x, y: QtCore.QPointF(cx + (x - ox) * s, cy - (y - oy) * s)
        path = outline_path(self.solids, self.holes, tf)
        p.setPen(QtGui.QPen(line, 1.4))
        p.setBrush(fill)
        p.drawPath(path)
        if self.parts:  # each material in its own colour (steel inside the concrete)
            for steel, so, ho in self.parts:
                p.setBrush(_theme_colors(self, steel)[1])
                p.setPen(QtGui.QPen(line, 1.0))
                p.drawPath(outline_path(so, ho, tf))
        # local axes through the centroid (origin of the outline)
        o = tf(0.0, 0.0)
        L = 0.28 * min(W, H)
        axc = QtGui.QColor("#d43f3a")
        p.setPen(QtGui.QPen(axc, 1.2))
        self._arrow(p, o, QtCore.QPointF(o.x() + L, o.y()), "3", axc)
        self._arrow(p, o, QtCore.QPointF(o.x(), o.y() - L), "2", axc)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(o, 3.5, 3.5)
        if self.marks:
            self._dimensions(p, lo, hi, tf, dimc, s)
        if self.message:
            p.setPen(line)
            p.drawText(QtCore.QRectF(4, self.height() - 18, self.width() - 8, 16), QtCore.Qt.AlignLeft, self.message)
        p.end()

    def _arrow(self, p, a, b, label, color):
        p.drawLine(a, b)
        d = b - a
        n = math.hypot(d.x(), d.y()) or 1.0
        ux, uy = d.x() / n, d.y() / n
        hd = QtGui.QPolygonF([b, QtCore.QPointF(b.x() - 8 * ux - 4 * uy, b.y() - 8 * uy + 4 * ux),
                              QtCore.QPointF(b.x() - 8 * ux + 4 * uy, b.y() - 8 * uy - 4 * ux)])
        p.setBrush(color)
        p.drawPolygon(hd)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawText(QtCore.QPointF(b.x() + 4 * ux + (4 if ux else -4), b.y() + 4 * uy - (0 if ux else 4)), label)

    def _text(self, v: float) -> str:
        if self.du is None:
            return fmt(v, 3)
        return f"{fmt(v * self.du.factor('section_dim'), 4)}"

    def _dim(self, p, a, b, off, label, color, vertical):
        """Dimension line between screen points a and b, offset by ``off`` px."""
        pen = QtGui.QPen(color, 1.0)
        p.setPen(pen)
        if vertical:
            x = a.x() + off
            p.drawLine(QtCore.QPointF(a.x(), a.y()), QtCore.QPointF(x, a.y()))
            p.drawLine(QtCore.QPointF(b.x(), b.y()), QtCore.QPointF(x, b.y()))
            p.drawLine(QtCore.QPointF(x, a.y()), QtCore.QPointF(x, b.y()))
            for y in (a.y(), b.y()):
                p.drawLine(QtCore.QPointF(x - 3, y + 3), QtCore.QPointF(x + 3, y - 3))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label)
            ym = 0.5 * (a.y() + b.y())
            r = QtCore.QRectF(x - tw - 6 if off < 0 else x + 4, ym - 8, tw + 2, 16)
            p.drawText(r, QtCore.Qt.AlignCenter, label)
        else:
            y = a.y() + off
            p.drawLine(QtCore.QPointF(a.x(), a.y()), QtCore.QPointF(a.x(), y))
            p.drawLine(QtCore.QPointF(b.x(), b.y()), QtCore.QPointF(b.x(), y))
            p.drawLine(QtCore.QPointF(a.x(), y), QtCore.QPointF(b.x(), y))
            for x in (a.x(), b.x()):
                p.drawLine(QtCore.QPointF(x - 3, y + 3), QtCore.QPointF(x + 3, y - 3))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label)
            xm = 0.5 * (a.x() + b.x())
            if abs(b.x() - a.x()) < tw + 6 and off == 0:  # thin web: label beside it with a leader
                x1 = max(a.x(), b.x())
                p.drawLine(QtCore.QPointF(x1, y), QtCore.QPointF(x1 + 14, y))
                p.drawText(QtCore.QRectF(x1 + 16, y - 8, tw + 4, 16), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, label)
                return
            r = QtCore.QRectF(xm - tw / 2 - 1, y + 2 if off > 0 else y - 17, tw + 2, 15)
            p.drawText(r, QtCore.Qt.AlignCenter, label)

    def _dimensions(self, p, lo, hi, tf, color, s):
        loops = list(self.solids) + list(self.holes)
        # overall width (below) and depth (left)
        self._dim(p, tf(lo[0], lo[1]), tf(hi[0], lo[1]), 16, self._text(hi[0] - lo[0]), color, False)
        self._dim(p, tf(lo[0], lo[1]), tf(lo[0], hi[1]), -16, self._text(hi[1] - lo[1]), color, True)
        # web / wall thickness: the first material piece cut at mid height of the web
        top_t = _vertical_top_piece(loops, lo, hi)
        if top_t is not None and top_t[1] - top_t[0] < 0.9 * (hi[1] - lo[1]):
            y0, y1, xc = top_t
            # at the right end of the top piece (a hat's crown stops short of the brims)
            row = [sg for sg in _segments_at(loops, 0.5 * (y0 + y1), 1) if sg[0] - 1e-9 <= xc <= sg[1] + 1e-9]
            xd = row[0][1] if row else hi[0]
            self._dim(p, tf(xd, y0), tf(xd, y1), 16, self._text(y1 - y0), color, True)
        web = _web_piece(loops, lo, hi, top_t)
        if web is not None and web[1] - web[0] < 0.9 * (hi[0] - lo[0]):
            x0, x1, y = web
            self._dim(p, tf(x0, y), tf(x1, y), 0, self._text(x1 - x0), color, False)


def _segments_at(loops, level: float, axis: int) -> List[Tuple[float, float]]:
    """Material pieces cut by the line (axis 1: y = level, axis 0: x = level)."""
    xs = []
    a, o = (1, 0) if axis == 1 else (0, 1)
    for P in loops:
        ya, yb = P[:, a], np.roll(P[:, a], -1)
        xa, xb = P[:, o], np.roll(P[:, o], -1)
        m = (ya > level) != (yb > level)
        xs += list(xa[m] + (level - ya[m]) * (xb[m] - xa[m]) / (yb[m] - ya[m]))
    xs.sort()
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def _vertical_top_piece(loops, lo, hi):
    """The top flange / wall thickness: the top-most piece cut by a vertical line
    near the right edge (or through the middle for closed sections)."""
    w = hi[0] - lo[0]
    xs = [hi[0] - 0.15 * w, 0.5 * (lo[0] + hi[0]) + 0.013 * w]
    mid = _segments_at(loops, 0.5 * (lo[1] + hi[1]) + 1e-9, 1)
    if len(mid) == 1 and mid[0][1] < hi[0] - 0.2 * w:  # I / channel: the middle of the flange right of the web
        xs.insert(0, 0.5 * (mid[0][1] + hi[0]))  # (tapered flanges: their mean thickness is there)
    for x in xs:
        segs = _segments_at(loops, x, 0)
        if len(segs) >= 1:
            y0, y1 = segs[-1]
            if abs(y1 - hi[1]) < 1e-6 * (hi[1] - lo[1] + w) and (len(segs) > 1 or y1 - y0 < 0.9 * (hi[1] - lo[1])):
                return (y0, y1, x)
    return None


def _web_piece(loops, lo, hi, top):
    """The web / side wall thickness: the left-most piece cut half way between
    the bottom and the underside of the top flange."""
    ytop = top[0] if top is not None else hi[1]
    bottom = _segments_at(loops, lo[0] + 0.5 * (hi[0] - lo[0]) + 1e-9, 0)
    yb = bottom[0][1] if bottom and bottom[0][0] <= lo[1] + 1e-9 and bottom[0][1] < ytop - 1e-12 and len(bottom) > 1 else lo[1]
    y = 0.5 * (yb + ytop)
    segs = _segments_at(loops, y, 1)
    if not segs:
        return None
    x0, x1 = segs[0]
    return (x0, x1, y)


def paint_icon(solids, holes, steel: bool, size: int = 56) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    if not solids:
        return pm
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    allp = np.vstack(solids)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    w, h = max(hi[0] - lo[0], 1e-12), max(hi[1] - lo[1], 1e-12)
    s = (size - 8) / max(w, h)
    ox, oy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    tf = lambda x, y: QtCore.QPointF(size / 2 + (x - ox) * s, size / 2 - (y - oy) * s)
    p.setPen(QtGui.QPen(QtGui.QColor("#30343a"), 1.2))
    p.setBrush(QtGui.QColor("#7fa3c7" if steel else "#b9b4a8"))
    p.drawPath(outline_path(solids, holes, tf))
    p.end()
    return pm


def designer_icon(size: int = 56) -> QtGui.QPixmap:
    """A free polygon with a pencil."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    poly = QtGui.QPolygonF([QtCore.QPointF(x * size, y * size) for x, y in
                            ((0.12, 0.85), (0.62, 0.88), (0.55, 0.55), (0.85, 0.45), (0.6, 0.15), (0.2, 0.3))])
    p.setPen(QtGui.QPen(QtGui.QColor("#30343a"), 1.2))
    p.setBrush(QtGui.QColor("#c9b98f"))
    p.drawPolygon(poly)
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(QtGui.QPen(QtGui.QColor("#1f6fd1"), 2.2))
    p.drawLine(QtCore.QPointF(0.62 * size, 0.88 * size), QtCore.QPointF(0.95 * size, 0.6 * size))
    for x, y in ((0.12, 0.85), (0.62, 0.88), (0.55, 0.55), (0.85, 0.45), (0.6, 0.15), (0.2, 0.3)):
        p.fillRect(QtCore.QRectF(x * size - 2.5, y * size - 2.5, 5, 5), QtGui.QColor("#1f6fd1"))
    p.end()
    return pm


def composite_icon(circle: bool, size: int = 56) -> QtGui.QPixmap:
    from ..model.section_shapes import circle as _circ, rect as _rect
    outer = _circ(0, 0, 0.36) if circle else _rect(-0.3, -0.3, 0.6, 0.6)
    sol, _ = shape_outline(SHAPES["I"], {"t3": 0.3, "t2": 0.3, "tf": 0.03, "tw": 0.025}, False)
    pm = paint_icon([outer], [], False, size)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    s = (size - 8) / 0.72 if circle else (size - 8) / 0.6
    tf = lambda x, y: QtCore.QPointF(size / 2 + x * s, size / 2 - y * s)
    p.setPen(QtGui.QPen(QtGui.QColor("#30343a"), 1.0))
    p.setBrush(QtGui.QColor("#7fa3c7"))
    p.drawPath(outline_path(sol, [], tf))
    p.end()
    return pm


def text_icon(text: str, size: int = 56) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor("#30343a"), 1.2))
    p.setBrush(QtGui.QColor("#e6e2d6"))
    p.drawRoundedRect(QtCore.QRectF(6, 10, size - 12, size - 20), 4, 4)
    f = p.font()
    f.setBold(True)
    f.setPointSizeF(9)
    p.setFont(f)
    p.drawText(QtCore.QRectF(6, 10, size - 12, size - 20), QtCore.Qt.AlignCenter, text)
    p.end()
    return pm


def choice_icon(key: str, overrides: dict, kind: str, size: int = 56) -> QtGui.QPixmap:
    if key == DESIGNER:
        return designer_icon(size)
    if key == GENERAL:
        return text_icon("A, I", size)
    if key == "library":
        return text_icon("xml", size)
    if key in (COMPOSITE_RECT, COMPOSITE_CIRCLE):
        return composite_icon(key == COMPOSITE_CIRCLE, size)
    sh = SHAPES[key]
    v = {dm.attr: overrides.get(dm.attr, dm.default) for dm in sh.dims}
    if key == "Rectangular" and kind == "Steel":
        v = {"t3": 0.2, "t2": 0.05}
    if key == "I":
        v.update(tf=0.02, tw=0.012)  # bolder for the icon
    if key in ("Channel", "Tee", "Angle", "DoubleAngle", "DoubleChannel", "Box"):
        v = {k: (x * 2.2 if k in ("tf", "tw") and x else x) for k, x in v.items()}
    if key == "Pipe":
        v["tw"] = 0.012
    if key in COLD:
        v["tw"] = 0.006
    try:
        solids, holes = shape_outline(sh, v, False)
    except Exception:
        return text_icon(key[:4], size)
    return paint_icon(solids, holes, kind == "Steel", size)


# ------------------------------------------------------------ properties
def property_rows(model, sec, props: Optional[SectionProps]) -> List[Tuple[str, str, str]]:
    """(name, value, unit) rows of a section in the display units."""
    du = DisplayUnits.of(model)
    ka, ua = du.factor("section_area"), du.label("section_area")
    ki, ui = du.factor("section_inertia"), du.label("section_inertia")
    kd, ud = du.factor("section_dim"), du.label("section_dim")
    rows = []
    if props is not None:
        A, I33, I22, J, As2, As3 = props.A, props.I33, props.I22, props.J, props.As2, props.As3
    else:
        A, I33, I22, J, As2, As3 = sec.A, sec.I33, sec.I22, sec.J, sec.As2, sec.As3
    rows += [("Area A", fmt(A * ka), ua), ("I33 (major, about 3)", fmt(I33 * ki), ui),
             ("I22 (minor, about 2)", fmt(I22 * ki), ui)]
    if props is not None and abs(props.I23) > 1e-6 * max(props.I33, props.I22, 1e-300):
        rows += [("I23 (product)", fmt(props.I23 * ki), ui),
                 ("Imax (principal, u)", fmt(props.Imax * ki), ui), ("Imin (principal, v)", fmt(props.Imin * ki), ui),
                 ("Principal axis angle", fmt(math.degrees(props.angle), 3), "deg")]
    rows += [("Torsion J", fmt(J * ki), ui)]
    if props is not None and props.J_method:
        rows.append(("  J from", props.J_method, ""))
    rows += [("Shear area As2", fmt(As2 * ka), ua), ("Shear area As3", fmt(As3 * ka), ua)]
    if props is not None and props.S33 > 0:
        k3 = kd ** 3
        rows += [("S33 (elastic modulus)", fmt(props.S33 * k3), f"{ud}^3"), ("S22", fmt(props.S22 * k3), f"{ud}^3"),
                 ("Z33 (plastic modulus)", fmt(props.Z33 * k3), f"{ud}^3"), ("Z22", fmt(props.Z22 * k3), f"{ud}^3"),
                 ("r33 (radius of gyration)", fmt(props.r33 * kd), ud), ("r22", fmt(props.r22 * kd), ud)]
    elif props is not None and props.r33 > 0:
        rows += [("r33 (radius of gyration)", fmt(props.r33 * kd), ud), ("r22", fmt(props.r22 * kd), ud)]
    mat = model.materials.get(sec.material)
    comp = sec.geometry and is_composite(sec.geometry)
    if comp:
        areas = material_areas(sec.geometry)
        ratios = section_ratios(sec, model.materials)
        tot = sum(areas.values())
        rows.insert(0, ("(transformed to " + sec.material + ")", "", ""))
        for k, a in areas.items():
            name = k or sec.material
            rows.append((f"Area of {name}", fmt(a * ka), ua))
            if k:
                rows.append((f"n = E({name}) / E({sec.material})", fmt(ratios.get(k, (1, 1))[0], 4), ""))
                rows.append((f"Ratio {name} / gross", fmt(100 * a / tot, 3) if tot else "-", "%"))
        w = sum(a * ((model.materials.get(k) if k else mat) or mat).weight_density for k, a in areas.items()) \
            if mat is not None else 0.0
        if w:
            rows.append(("Weight per length", fmt(w * du.factor("line_load")), du.label("line_load")))
    elif mat is not None and mat.weight_density:
        w = A * mat.weight_density  # force / length (model units)
        rows.append(("Weight per length", fmt(w * du.factor("line_load")), du.label("line_load")))
    if mat is not None:
        ks, us = du.factor("stress"), du.label("stress")
        txt = mat.name
        if mat.fc:
            txt += f", fc' {fmt(mat.fc * ks, 3)} {us}"
        if mat.fy:
            txt += f", fy {fmt(mat.fy * ks, 3)} {us}"
        rows.append(("Material", txt, ""))
    m = sec.modifiers
    if m is not None and any(abs(getattr(m, f) - 1.0) > 1e-12 for f in ("A", "As2", "As3", "J", "I22", "I33", "mass", "weight")):
        rows.append(("Property modifiers", ", ".join(f"{f} {fmt(getattr(m, f), 3)}" for f in
                                                     ("A", "As2", "As3", "J", "I22", "I33", "mass", "weight")
                                                     if abs(getattr(m, f) - 1.0) > 1e-12), ""))
    return rows


def fill_table(table: QtWidgets.QTableWidget, rows):
    from .symbols import pretty
    table.setRowCount(len(rows))
    for i, (a, b, c) in enumerate(rows):
        for j, t in enumerate((a, b, c)):
            it = QtWidgets.QTableWidgetItem(pretty(t) if j != 1 else t)  # I₃₃, cm⁴, f′c ...
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            if j == 1:
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            table.setItem(i, j, it)
    table.resizeColumnsToContents()


def props_table() -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, 3)
    t.setHorizontalHeaderLabels(["Property", "Value", "Unit"])
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setStretchLastSection(True)
    t.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    t.setAlternatingRowColors(True)
    return t


def edit_modifiers(parent, mods: FrameModifiers) -> Optional[FrameModifiers]:
    from .forms import FormDialog
    fields = [(f, lab, float(getattr(mods, f))) for f, lab in
              (("A", "Cross-section (axial) area"), ("As2", "Shear area in 2 direction"),
               ("As3", "Shear area in 3 direction"), ("J", "Torsional constant"),
               ("I22", "Moment of inertia about 2 axis"), ("I33", "Moment of inertia about 3 axis"),
               ("mass", "Mass"), ("weight", "Weight"))]
    d = FormDialog("Property / Stiffness Modifiers", fields, parent,
                   note="Multipliers on the section properties (e.g. 0.35 on I33 for cracked beams, "
                        "0.70 for columns).")
    if d.exec() != QtWidgets.QDialog.Accepted:
        return None
    try:
        return FrameModifiers(**{k: float(v) for k, v in d.values().items()})
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- chooser
class SectionTypeDialog(QtWidgets.QDialog):
    """Add New Property: pick the kind of section (drawn icons by group)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Frame Section - Select Type")
        self.choice: Optional[Tuple[str, dict, str]] = None
        lay = QtWidgets.QVBoxLayout(self)
        self.buttons: Dict[str, QtWidgets.QToolButton] = {}
        for group, items in CHOICES:
            box = QtWidgets.QGroupBox(group)
            grid = QtWidgets.QGridLayout(box)
            grid.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            for i, (label, key, over, kind) in enumerate(items):
                b = QtWidgets.QToolButton()
                b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
                b.setIcon(QtGui.QIcon(choice_icon(key, over, kind)))
                b.setIconSize(QtCore.QSize(56, 56))
                b.setText(label)
                b.setFixedSize(112, 92)
                b.setAutoRaise(True)
                tip = SHAPES[key].note if key in SHAPES else {
                    DESIGNER: "Draw any outline: polygons, rectangles, circles, openings - like in CAD",
                    GENERAL: "Type A, I33, I22, J, As2, As3 directly",
                    "library": "Standard steel sections: TIS / JIS, AISC, EN (built in), or a CSV / xml file"}.get(key, "")
                if key in (COMPOSITE_RECT, COMPOSITE_CIRCLE):
                    tip = "Steel-reinforced concrete column: a WF / H shape encased in " + \
                          ("circular" if key == COMPOSITE_CIRCLE else "rectangular") + " concrete"
                b.setToolTip(tip or label)
                b.clicked.connect(lambda _=False, k=key, o=over, t=kind: self._pick(k, o, t))
                grid.addWidget(b, i // 6, i % 6)
                self.buttons[label] = b
            lay.addWidget(box)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick(self, key, over, kind):
        self.choice = (key, over, kind)
        self.accept()


# ---------------------------------------------------------- shape editor
class ShapeSectionDialog(QtWidgets.QDialog):
    """Parametric section: name, material, dimensions (display units), live
    drawing and properties, modifiers."""

    def __init__(self, model, sec: FrameSection, parent=None, existing: Sequence[str] = (), new: bool = True):
        super().__init__(parent)
        self.model = model
        self.orig = sec
        self.sec = copy.deepcopy(sec)
        self.shape: Shape = shape_of(sec)
        self.existing = [n for n in existing if n != sec.name]
        self.new = new
        self.du = DisplayUnits.of(model)
        self.k = self.du.factor("section_dim")
        self._auto_name = new
        self.setWindowTitle(f"Frame Section - {self.shape.label}")
        self.resize(860, 560)
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(sec.name)
        self.name.textEdited.connect(self._name_typed)
        form.addRow("Section name", self.name)
        self.material = QtWidgets.QComboBox()
        self.material.addItems(list(model.materials))
        self.material.setCurrentText(sec.material)
        self.material.currentTextChanged.connect(self._changed)
        form.addRow("Material", self.material)
        left.addLayout(form)
        box = QtWidgets.QGroupBox(f"Dimensions [{self.du.label('section_dim')}]")
        dform = QtWidgets.QFormLayout(box)
        self.edits: Dict[str, QtWidgets.QLineEdit] = {}
        for dm in self.shape.dims:
            val = getattr(sec, dm.attr, None)
            e = QtWidgets.QLineEdit("" if val is None else fmt(val * self.k, 6))
            if dm.optional:
                e.setPlaceholderText(dm.placeholder or "optional")
            e.textChanged.connect(self._changed)
            self.edits[dm.attr] = e
            dform.addRow(dm.label, e)
        self.mirror = None
        if self.shape.mirror:
            self.mirror = QtWidgets.QCheckBox("Mirror (flange / legs to the other side)")
            self.mirror.setChecked(bool((sec.params or {}).get("mirror")))
            self.mirror.toggled.connect(self._changed)
            dform.addRow(self.mirror)
        self.slope = self.slope_custom = None
        if self.shape.key in SLOPED:
            self.slope = QtWidgets.QComboBox()
            for label, val in FLANGE_SLOPES:
                self.slope.addItem(label, val)
            self.slope.addItem("Custom slope (%)", -1.0)
            self.slope.setToolTip("Inner face of the flanges: parallel, or tapered as the rolled shapes of a "
                                  "standard; the flange thickness is then the mean thickness (middle of the flange)")
            self.slope_custom = QtWidgets.QLineEdit()
            self.slope_custom.setPlaceholderText("%")
            self.slope_custom.setToolTip("Custom slope in % (tan of the angle x 100)")
            self.slope_custom.setMaximumWidth(64)
            cur = float((sec.params or {}).get("slope") or 0.0)
            k = next((i for i, (_l, val) in enumerate(FLANGE_SLOPES) if abs(val - cur) < 1e-6), None)
            if k is None:
                self.slope.setCurrentIndex(self.slope.count() - 1)
                self.slope_custom.setText(fmt(cur * 100.0, 5))
            else:
                self.slope.setCurrentIndex(k)
            self.slope_custom.setEnabled(k is None)
            self.slope.currentIndexChanged.connect(self._slope_kind)
            self.slope_custom.textChanged.connect(self._changed)
            srow = QtWidgets.QHBoxLayout()
            srow.addWidget(self.slope, 1)
            srow.addWidget(self.slope_custom)
            dform.addRow("Flanges", srow)
        left.addWidget(box)
        if self.shape.note:
            note = QtWidgets.QLabel(self.shape.note)
            note.setWordWrap(True)
            note.setStyleSheet("color: gray")
            left.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.mod_btn = QtWidgets.QPushButton("Property Modifiers...")
        self.mod_btn.clicked.connect(self._modifiers)
        row.addWidget(self.mod_btn)
        self.design = QtWidgets.QComboBox()
        self.design.addItems(["(auto)", "Beam", "Column", "Brace"])
        self.design.setCurrentText(sec.design_type or "(auto)")
        self.design.setToolTip("Design type (concrete design: beam or column)")
        row.addWidget(QtWidgets.QLabel("Design as"))
        row.addWidget(self.design)
        left.addLayout(row)
        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet("color: #c0392b")
        self.error.setWordWrap(True)
        left.addWidget(self.error)
        left.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        self.ok_button = bb.button(QtWidgets.QDialogButtonBox.Ok)
        left.addWidget(bb)
        lay.addLayout(left, 2)
        right = QtWidgets.QVBoxLayout()
        self.preview = SectionPreview()
        self.preview.setMinimumSize(320, 260)
        right.addWidget(self.preview, 3)
        self.table = props_table()
        right.addWidget(self.table, 2)
        lay.addLayout(right, 3)
        self._changed()

    def _name_typed(self, _t):
        self._auto_name = False

    def _slope_kind(self, *_):
        custom = self.slope.currentData() is not None and self.slope.currentData() < 0
        self.slope_custom.setEnabled(custom)
        self._changed()

    def slope_value(self) -> Optional[float]:
        """tan of the inner flange slope (0 = parallel); None when the custom
        value is not a number."""
        if self.slope is None:
            return 0.0
        val = self.slope.currentData()
        if val is not None and val >= 0:
            return float(val)
        x = calc(self.slope_custom.text().strip() or "0")
        return None if x is None else x / 100.0

    def values(self) -> Optional[dict]:
        v = {}
        for attr, e in self.edits.items():
            t = e.text().strip()
            if not t:
                v[attr] = None
                continue
            x = calc(t)
            if x is None:
                return None
            v[attr] = x / self.k
        if self.shape.key in COLD:
            v["tf"] = v.get("tw")
        if self.slope is not None:
            sl = self.slope_value()
            if sl is None:
                return None
            if sl:
                v["slope"] = sl
        return v

    def _changed(self, *_):
        v = self.values()
        if v is None:
            self._bad(["Type numbers for the dimensions"])
            return
        errs = check_dims(self.shape, v)
        if errs:
            self._bad(errs)
            return
        mirror = bool(self.mirror and self.mirror.isChecked())
        props = shape_properties(self.shape, v, mirror)
        solids, holes = shape_outline(self.shape, v, mirror)
        mat = self.model.materials.get(self.material.currentText())
        steel = mat is not None and mat.type in STEEL_TYPES
        self.preview.set_outline(solids, holes, steel, self.du)
        tmp = copy.deepcopy(self.sec)
        tmp.material = self.material.currentText()
        fill_table(self.table, property_rows(self.model, tmp, props))
        self.error.setText("")
        self.ok_button.setEnabled(True)
        if self._auto_name:
            self.name.setText(suggest_name(self.shape.key, {k: (x or 0) * self.model.units.factor("length")
                                                            for k, x in v.items() if k != "slope"}, self.existing))

    def _bad(self, errs):
        self.error.setText("\n".join(errs))
        self.ok_button.setEnabled(False)
        self.preview.set_outline([], [], False, self.du, "Check the dimensions")
        self.table.setRowCount(0)

    def _modifiers(self):
        m = edit_modifiers(self, self.sec.modifiers or FrameModifiers())
        if m is not None:
            self.sec.modifiers = m
            self._changed()

    def _ok(self):
        name = self.name.text().strip()
        if not name:
            self.error.setText("Give the section a name")
            return
        if name in self.existing:
            self.error.setText(f"A section named {name!r} already exists")
            return
        v = self.values()
        mirror = bool(self.mirror and self.mirror.isChecked())
        try:
            sec = FrameSection.from_shape(self.shape.key, name, self.material.currentText(), mirror=mirror, **v)
        except ModelError as exc:
            self.error.setText(exc.message)
            return
        # keep everything else of the section being edited
        for f in ("modifiers", "mass_per_length", "weight_per_length", "rebar", "color", "notes"):
            setattr(sec, f, copy.deepcopy(getattr(self.sec, f)))
        sec.design_type = "" if self.design.currentText() == "(auto)" else self.design.currentText()
        if not self.new and self.orig.shape == sec.shape and _same_dims(self.orig, v, mirror) and not self.orig.geometry:
            for f in ("A", "I33", "I22", "J", "As2", "As3", "source"):  # untouched: tabulated values stay
                setattr(sec, f, getattr(self.orig, f))
        self.result_section = sec
        self.accept()


def _same_dims(sec, v, mirror) -> bool:
    d = dims_of(sec)
    for k, x in v.items():
        y = d.get(k)
        if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-9 * max(abs(y), 1e-30)):
            return False
    return bool((sec.params or {}).get("mirror")) == mirror


# ---------------------------------------------------------- composite form
class CompositeSectionDialog(QtWidgets.QDialog):
    """Steel-reinforced concrete (SRC) column: a WF / H shape encased in a
    rectangular or circular concrete section.  Properties are transformed to
    the concrete (n = Es / Ec), weight from each material."""

    STEEL = (("t3", "Depth d"), ("t2", "Flange width bf"), ("tf", "Flange thickness tf"),
             ("tw", "Web thickness tw"), ("r", "Fillet radius r"))

    def __init__(self, model, sec: FrameSection, parent=None, existing: Sequence[str] = (), new: bool = True):
        super().__init__(parent)
        self.model = model
        self.sec = copy.deepcopy(sec)
        self.existing = [n for n in existing if n != sec.name]
        self.du = DisplayUnits.of(model)
        self.k = self.du.factor("section_dim")
        self._auto_name = new
        comp = composite_dims(sec)
        outline, steel = comp
        self.kind = outline["kind"]
        st_shape = next((g for g in sec.geometry if g.get("role") == "steel"), {})
        self.setWindowTitle("Composite Column - WF in " + ("circular" if self.kind == "circle" else "rectangular")
                            + " concrete")
        self.resize(920, 640)
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(sec.name)
        self.name.textEdited.connect(lambda *_: setattr(self, "_auto_name", False))
        form.addRow("Section name", self.name)
        self.conc = QtWidgets.QComboBox()
        self.conc.addItems(list(model.materials))
        self.conc.setCurrentText(sec.material)
        form.addRow("Concrete material", self.conc)
        self.steel_mat = QtWidgets.QComboBox()
        self.steel_mat.addItems(list(model.materials))
        self.steel_mat.setCurrentText(st_shape.get("material") or material_for(model, "Steel"))
        form.addRow("Steel material", self.steel_mat)
        left.addLayout(form)
        u = self.du.label("section_dim")
        box = QtWidgets.QGroupBox(f"Concrete [{u}]")
        cf = QtWidgets.QFormLayout(box)
        self.edits: Dict[str, QtWidgets.QLineEdit] = {}
        conc_fields = (("t3", "Diameter"),) if self.kind == "circle" else (("t3", "Depth (t3)"), ("t2", "Width (t2)"))
        for key, lab in conc_fields:
            e = QtWidgets.QLineEdit(fmt(outline[key] * self.k, 6))
            e.textChanged.connect(self._changed)
            self.edits["c_" + key] = e
            cf.addRow(lab, e)
        left.addWidget(box)
        box = QtWidgets.QGroupBox(f"Steel WF / H inside [{u}]")
        sf = QtWidgets.QFormLayout(box)
        for key, lab in self.STEEL:
            val = steel.get(key)
            e = QtWidgets.QLineEdit("" if val is None else fmt(val * self.k, 6))
            if key == "r":
                e.setPlaceholderText("0 = sharp corner")
            e.textChanged.connect(self._changed)
            self.edits["s_" + key] = e
            sf.addRow(lab, e)
        self.rotate = QtWidgets.QCheckBox("Rotate the steel 90 deg (its strong axis about the 2 axis)")
        self.rotate.setChecked(bool(st_shape.get("rotate")))
        self.rotate.toggled.connect(self._changed)
        sf.addRow(self.rotate)
        left.addWidget(box)
        self.conc.currentTextChanged.connect(self._changed)
        self.steel_mat.currentTextChanged.connect(self._changed)
        note = QtWidgets.QLabel("Properties transformed to the concrete (n = Es / Ec) for the analysis; the weight "
                                "comes from each material's own area.  Reinforcing bars are not included.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        left.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        mb = QtWidgets.QPushButton("Property Modifiers...")
        mb.clicked.connect(self._modifiers)
        row.addWidget(mb)
        self.to_designer = QtWidgets.QPushButton("Open in Section Designer...")
        self.to_designer.setToolTip("Continue drawing on this outline (add plates, openings ...)")
        self.to_designer.clicked.connect(self._designer)
        row.addWidget(self.to_designer)
        left.addLayout(row)
        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet("color: #c0392b")
        self.error.setWordWrap(True)
        left.addWidget(self.error)
        left.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        self.ok_button = bb.button(QtWidgets.QDialogButtonBox.Ok)
        left.addWidget(bb)
        lay.addLayout(left, 2)
        right = QtWidgets.QVBoxLayout()
        self.preview = SectionPreview()
        self.preview.setMinimumSize(340, 300)
        right.addWidget(self.preview, 3)
        self.table = props_table()
        right.addWidget(self.table, 3)
        lay.addLayout(right, 3)
        self._changed()

    def _read(self):
        v = {}
        for key, e in self.edits.items():
            t = e.text().strip()
            if not t:
                v[key] = None
                continue
            x = calc(t)
            if x is None:
                return None
            v[key] = x / self.k
        outline = {"t3": v.get("c_t3"), "t2": v.get("c_t2")}
        steel = {k: v.get("s_" + k) for k, _ in self.STEEL}
        return outline, steel

    def _check(self, outline, steel) -> List[str]:
        errs = check_dims(SHAPES["I"], steel)
        if outline["t3"] is None or outline["t3"] <= 0 or (self.kind != "circle" and (outline["t2"] or 0) <= 0):
            errs.append("The concrete needs positive dimensions")
        if errs:
            return errs
        d, b = steel["t3"], steel["t2"]
        if self.rotate.isChecked():
            d, b = b, d
        if self.kind == "circle":
            if math.hypot(d / 2, b / 2) >= outline["t3"] / 2:
                errs.append("The steel does not fit inside the concrete circle")
        elif d >= outline["t3"] or b >= outline["t2"]:
            errs.append("The steel does not fit inside the concrete")
        if self.steel_mat.currentText() == self.conc.currentText():
            errs.append("Choose a steel material different from the concrete")
        return errs

    def build(self) -> Optional[FrameSection]:
        r = self._read()
        if r is None:
            self.error.setText("Type numbers for the dimensions")
            return None
        outline, steel = r
        errs = self._check(outline, steel)
        if errs:
            self.error.setText("\n".join(errs))
            return None
        sec = copy.deepcopy(self.sec)
        sec.material = self.conc.currentText()
        sec.shape = "Composite"
        sec.geometry = composite_geometry(self.kind, outline, steel, self.steel_mat.currentText(),
                                          self.rotate.isChecked())
        sec.params = dict(sec.params or {}, composite=self.kind)
        sec.update_properties(True, self.model.materials)
        return sec

    def _changed(self, *_):
        sec = self.build()
        if sec is None:
            self.ok_button.setEnabled(False)
            self.preview.set_outline([], [], False, self.du, "Check the dimensions")
            self.table.setRowCount(0)
            return
        self.error.setText("")
        self.ok_button.setEnabled(True)
        self.preview.set_section(sec, self.model)
        props = section_properties(sec, True, self.model.materials)
        fill_table(self.table, property_rows(self.model, sec, props))
        if self._auto_name:
            o, st = composite_dims(sec)
            L = self.model.units.factor("length")
            cm = lambda x: fmt((x or 0) * L * 100, 3)
            mm = lambda x: fmt((x or 0) * L * 1000, 3)
            base = (f"SRC-D{cm(o['t3'])}" if self.kind == "circle" else f"SRC{cm(o['t2'])}x{cm(o['t3'])}") + \
                f"-H{mm(st['t3'])}x{mm(st['t2'])}"
            self.name.setText(unique_name(base, self.existing))

    def _modifiers(self):
        m = edit_modifiers(self, self.sec.modifiers or FrameModifiers())
        if m is not None:
            self.sec.modifiers = m
            self._changed()

    def _designer(self):
        sec = self.build()
        if sec is None:
            return
        sec.name = self.name.text().strip() or sec.name
        from .section_designer import SectionDesignerDialog
        d = SectionDesignerDialog(self.model, sec, self, existing=self.existing)
        if d.exec() == QtWidgets.QDialog.Accepted and getattr(d, "result_section", None) is not None:
            out = d.result_section
            out.params = {k: v for k, v in (out.params or {}).items() if k != "composite"}  # now a free drawing
            self.result_section = out
            self.accept()

    def _ok(self):
        name = self.name.text().strip()
        if not name:
            self.error.setText("Give the section a name")
            return
        if name in self.existing:
            self.error.setText(f"A section named {name!r} already exists")
            return
        sec = self.build()
        if sec is None:
            return
        sec.name = name
        self.result_section = sec
        self.accept()


# --------------------------------------------------- general / library
class GeneralSectionDialog(QtWidgets.QDialog):
    """A section given by its properties (General), or a library section
    (tabulated properties, shown and editable)."""

    FIELDS = (("A", "Cross-section area A", "section_area"), ("I33", "Moment of inertia I33", "section_inertia"),
              ("I22", "Moment of inertia I22", "section_inertia"), ("J", "Torsional constant J", "section_inertia"),
              ("As2", "Shear area As2 (0 = no shear deformation)", "section_area"),
              ("As3", "Shear area As3 (0 = no shear deformation)", "section_area"),
              ("t3", "Depth t3 (for drawing)", "section_dim"), ("t2", "Width t2 (for drawing)", "section_dim"))

    def __init__(self, model, sec: FrameSection, parent=None, existing: Sequence[str] = ()):
        super().__init__(parent)
        self.model = model
        self.sec = copy.deepcopy(sec)
        self.existing = [n for n in existing if n != sec.name]
        self.du = DisplayUnits.of(model)
        self.setWindowTitle("Frame Section - " + (f"{type_label(sec)}" if sec.shape != GENERAL else "General"))
        self.resize(760, 480)
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(sec.name)
        form.addRow("Section name", self.name)
        self.material = QtWidgets.QComboBox()
        self.material.addItems(list(model.materials))
        self.material.setCurrentText(sec.material)
        form.addRow("Material", self.material)
        self.edits = {}
        for f, lab, item in self.FIELDS:
            val = getattr(sec, f) or 0.0
            e = QtWidgets.QLineEdit(fmt(val * self.du.factor(item), 6))
            e.textChanged.connect(self._changed)
            self.edits[f] = (e, item)
            form.addRow(f"{lab} [{self.du.label(item)}]", e)
        left.addLayout(form)
        if sec.source:
            lab = QtWidgets.QLabel(f"Properties from the section library {sec.source} (dimensions in the drawing "
                                   f"are for display; the tabulated values are used in the analysis).")
            lab.setWordWrap(True)
            lab.setStyleSheet("color: gray")
            left.addWidget(lab)
        row = QtWidgets.QHBoxLayout()
        mb = QtWidgets.QPushButton("Property Modifiers...")
        mb.clicked.connect(self._modifiers)
        row.addWidget(mb)
        row.addStretch(1)
        left.addLayout(row)
        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet("color: #c0392b")
        left.addWidget(self.error)
        left.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        left.addWidget(bb)
        lay.addLayout(left, 2)
        right = QtWidgets.QVBoxLayout()
        self.preview = SectionPreview()
        right.addWidget(self.preview, 3)
        self.table = props_table()
        right.addWidget(self.table, 2)
        lay.addLayout(right, 3)
        self._changed()

    def _values(self):
        out = {}
        for f, (e, item) in self.edits.items():
            try:
                out[f] = float(e.text() or 0) / self.du.factor(item)
            except ValueError:
                return None
        return out

    def _changed(self, *_):
        v = self._values()
        if v is None:
            self.error.setText("Type numbers")
            return
        self.error.setText("")
        tmp = copy.deepcopy(self.sec)
        for k, x in v.items():
            setattr(tmp, k, x if k not in ("t3", "t2") or x > 0 else None)
        self.preview.set_section(tmp, self.model)
        fill_table(self.table, property_rows(self.model, tmp, None))

    def _modifiers(self):
        m = edit_modifiers(self, self.sec.modifiers or FrameModifiers())
        if m is not None:
            self.sec.modifiers = m
            self._changed()

    def _ok(self):
        v = self._values()
        name = self.name.text().strip()
        if v is None or not name:
            self.error.setText("Give a name and numbers")
            return
        if name in self.existing:
            self.error.setText(f"A section named {name!r} already exists")
            return
        if v["A"] <= 0 and v["I33"] <= 0 and v["I22"] <= 0:
            self.error.setText("The section needs an area or a moment of inertia")
            return
        sec = self.sec
        sec.name, sec.material = name, self.material.currentText()
        for k, x in v.items():
            setattr(sec, k, x if k not in ("t3", "t2") or x > 0 else None)
        self.result_section = sec
        self.accept()


class LibraryDialog(QtWidgets.QDialog):
    """Steel Section Catalog: pick sections (several at once) from the built-in
    tables (TIS / JIS, AISC, EN), a CSV file of the same columns or a CSI
    section database (xml), into the project."""

    BUILTIN = {"TIS 1227 / JIS G3192": "Thailand / Japan - hot-rolled (TIS 1227, JIS G3192)",
               "TIS 1228 / JIS G3350": "Thailand / Japan - cold-formed (TIS 1228, JIS G3350)",
               "TIS 107-2566": "Thailand - pipes and tubes (TIS 107: JIS G3444 / G3466, ASTM A500, EN 10219)",
               "GB/T 11263": "China - H sections (GB/T 11263)",
               "GB/T 706": "China - I-beams, channels, angles (GB/T 706)",
               "AISC": "USA - AISC shapes", "EN": "Europe - EN (IPE, HE, IPN, UPN, UPE, L)"}

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Steel Section Catalog")
        self.resize(860, 560)
        self.sections: List[FrameSection] = []
        self.sources: Dict[str, object] = {}  # label -> list of SteelEntry | xml file name
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.file = QtWidgets.QComboBox()
        self.browse = QtWidgets.QPushButton("Import...")
        self.browse.setToolTip("A catalog file: CSV (standard, series, name, shape, t3, t2, tf, tw, r, r2, lip, "
                               "A, I33, I22, J, mass in mm / cm2 / cm4 / kg/m) or a CSI section database (xml)")
        self.browse.clicked.connect(self._browse)
        fr = QtWidgets.QHBoxLayout()
        fr.addWidget(self.file, 1)
        fr.addWidget(self.browse)
        form.addRow("Catalog", fr)
        self.series = QtWidgets.QComboBox()
        form.addRow("Series", self.series)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("e.g. H300  or  W14X  or  C150")
        form.addRow("Filter", self.filter)
        self.material = QtWidgets.QComboBox()
        self.material.addItems(list(model.materials))
        self.material.setCurrentText(material_for(model, "Steel"))
        form.addRow("Material", self.material)
        left.addLayout(form)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        left.addWidget(self.list, 1)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray")
        left.addWidget(self.info)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.ok_button = bb.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok_button.setText("Add to project")
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        left.addWidget(bb)
        lay.addLayout(left, 3)
        right = QtWidgets.QVBoxLayout()
        self.preview = SectionPreview()
        right.addWidget(self.preview, 3)
        self.table = props_table()
        right.addWidget(self.table, 2)
        lay.addLayout(right, 3)
        try:
            from ..catalog import steel_catalog
            for std, entries in steel_catalog().items():
                self._add_source(self.BUILTIN.get(std, std), entries)
        except Exception as exc:  # a broken install must not stop the form
            self.info.setText(f"Built-in catalog not available: {exc}")
        try:
            from ..io.section_db import SectionLibrary
            self.library = SectionLibrary()
            for name in self.library.available():
                self._add_source(name, name)
        except Exception:
            self.library = None
        self.file.currentTextChanged.connect(self._fill_series)
        self.series.currentTextChanged.connect(self._fill)
        self.filter.textChanged.connect(self._fill)
        self.list.currentTextChanged.connect(self._show)
        self.list.itemSelectionChanged.connect(self._count)
        self._fill_series()

    def _add_source(self, label, src):
        self.sources[label] = src
        if self.file.findText(label) < 0:
            self.file.addItem(label)

    def _browse(self):
        import os
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Section catalog", "",
                                                        "Section catalogs (*.csv *.xml);;CSV (*.csv);;CSI xml (*.xml)")
        if not path:
            return
        label = os.path.basename(path)
        try:
            if path.lower().endswith(".csv"):
                from ..catalog import read_steel
                with open(path, encoding="utf-8-sig") as fh:
                    entries = read_steel(fh.read(), default_standard=os.path.splitext(label)[0])
                if not entries:
                    raise ValueError("no sections in the file")
                self._add_source(label, entries)
            else:
                if self.library is None:
                    raise ValueError("the xml reader is not available")
                self.library.search_paths.insert(0, os.path.dirname(path))
                self._add_source(label, label)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import catalog", f"{label}:\n{exc}")
            return
        self.file.setCurrentText(label)

    def _entries(self):
        src = self.sources.get(self.file.currentText())
        return src if isinstance(src, list) else None

    def _fill_series(self, *_):
        self.series.blockSignals(True)
        self.series.clear()
        self.series.addItem("All")
        ents = self._entries()
        if ents:
            seen = []
            for e in ents:
                if e.series not in seen:
                    seen.append(e.series)
            self.series.addItems(seen)
        self.series.blockSignals(False)
        self._fill()

    def _fill(self, *_):
        self.list.clear()
        flt = self.filter.text().upper().replace(" ", "")
        ents = self._entries()
        if ents is not None:
            ser = self.series.currentText()
            names = [e.name for e in ents if ser in ("", "All") or e.series == ser]
        else:
            if self.library is None or not self.file.currentText():
                names = []
            else:
                try:
                    names = self.library.names(self.file.currentText())
                except Exception as exc:
                    self.info.setText(str(exc))
                    return
        self.list.addItems([n for n in names if flt in n.upper().replace(" ", "")][:3000])
        if self.file.count() == 0:
            self.info.setText("No catalog - Import... a CSV or xml section table.")
        self._count()

    def _count(self):
        n = len(self.list.selectedItems())
        self.ok_button.setEnabled(n > 0)
        ents = self._entries()
        src = ""
        if ents:
            src = "Tabulated A and I (and J where given) are used in the analysis; the dimensions draw the section."
        self.info.setText((f"{n} selected.  " if n else "Select one or more sections (Ctrl / Shift for several).  ")
                          + src)

    def _make(self, label) -> Optional[FrameSection]:
        mat = self.material.currentText()
        try:
            ents = self._entries()
            if ents is not None:
                e = next(x for x in ents if x.name == label)
                sec = e.frame_section(mat, self.model.units)
            else:
                sec = self.library.frame_section(self.file.currentText(), label, mat, self.model.units)
        except Exception as exc:
            self.info.setText(str(exc))
            return None
        return sec

    def _show(self, label):
        if not label:
            return
        sec = self._make(label)
        if sec is None:
            return
        self.preview.set_section(sec, self.model)
        props = section_properties(sec, True, self.model.materials) if shape_of(sec) else None
        if props is not None:  # tabulated values win
            props.A, props.I33, props.I22, props.J, props.As2, props.As3 = sec.A, sec.I33, sec.I22, sec.J, sec.As2, sec.As3
            props.J_method = f"tabulated ({sec.source})" if sec.J and self._tab_j(label) else props.J_method
        fill_table(self.table, property_rows(self.model, sec, props))

    def _tab_j(self, label) -> bool:
        ents = self._entries() or []
        return any(x.name == label and x.J for x in ents)

    def _ok(self):
        names = [it.text() for it in self.list.selectedItems()]
        out = []
        taken = list(self.model.frame_sections)
        for n in names:
            sec = self._make(n)
            if sec is not None:
                if sec.name in taken:
                    sec.name = unique_name(sec.name, taken)
                taken.append(sec.name)
                out.append(sec)
        self.sections = out
        self.accept()


# ---------------------------------------------------------------- manager
class FrameSectionsDialog(QtWidgets.QDialog):
    """Define > Frame Sections: every frame section with its type, material,
    size and use; Add New / Add Copy / Modify / Delete; the selected section
    drawn with its dimensions, and its properties."""

    COLUMNS = ("Name", "Type", "Material", "Size", "Used by")

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setWindowTitle("Frame Sections")
        self.resize(1180, 640)
        lay = QtWidgets.QHBoxLayout(self)
        # --- list
        left = QtWidgets.QVBoxLayout()
        frow = QtWidgets.QHBoxLayout()
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Find section...")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self.refresh)
        self.group = QtWidgets.QComboBox()
        self.group.addItems(["All types", "Concrete", "Steel", "Composite", "Other"])
        self.group.currentTextChanged.connect(self.refresh)
        frow.addWidget(self.filter, 1)
        frow.addWidget(self.group)
        left.addLayout(frow)
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(False)
        for j in range(len(self.COLUMNS)):
            hh.setSectionResizeMode(j, QtWidgets.QHeaderView.Stretch if j == 3 else QtWidgets.QHeaderView.ResizeToContents)
        hh.setSortIndicator(-1, QtCore.Qt.AscendingOrder)  # definition order until a header is clicked
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.itemDoubleClicked.connect(lambda *_: self.modify())
        left.addWidget(self.table, 1)
        self.count = QtWidgets.QLabel("")
        left.addWidget(self.count)
        lay.addLayout(left, 5)
        # --- buttons
        mid = QtWidgets.QVBoxLayout()
        box = QtWidgets.QGroupBox("Click to:")
        bl = QtWidgets.QVBoxLayout(box)
        self.btn_add = QtWidgets.QPushButton("Add New Section...")
        self.btn_copy = QtWidgets.QPushButton("Add Copy of Section...")
        self.btn_modify = QtWidgets.QPushButton("Modify / Show Section...")
        self.btn_delete = QtWidgets.QPushButton("Delete Section")
        self.btn_import = QtWidgets.QPushButton("Catalog...")
        for b, f, tip in ((self.btn_add, self.add, "Choose a type: concrete, steel, library, General, Section Designer"),
                          (self.btn_copy, self.copy_section, "A new section starting from the selected one"),
                          (self.btn_modify, self.modify, "Change the selected section (double-click a row)"),
                          (self.btn_delete, self.delete, "Delete the selected sections (not if a member uses them)"),
                          (self.btn_import, self.import_library, "Standard steel sections: TIS / JIS, AISC, EN (built in), or a CSV / xml file")):
            b.clicked.connect(f)
            b.setToolTip(tip)
            bl.addWidget(b)
        mid.addWidget(box)
        mid.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.accept)
        mid.addWidget(bb)
        lay.addLayout(mid, 0)
        # --- drawing and properties
        right = QtWidgets.QVBoxLayout()
        self.title = QtWidgets.QLabel("")
        f = self.title.font()
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 1.15)
        self.title.setFont(f)
        right.addWidget(self.title)
        self.preview = SectionPreview()
        self.preview.setMinimumSize(340, 260)
        right.addWidget(self.preview, 3)
        self.props = props_table()
        right.addWidget(self.props, 2)
        lay.addLayout(right, 4)
        QtGui.QShortcut(QtGui.QKeySequence.Delete, self.table, activated=self.delete)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, activated=self.copy_section)
        QtGui.QShortcut(QtGui.QKeySequence.New, self, activated=self.add)
        self.refresh()
        if self.table.rowCount():
            self.table.selectRow(0)

    # -- list
    @property
    def model(self):
        return self.doc.model

    def refresh(self, *_, select: Optional[Sequence[str]] = None):
        m = self.model
        du = DisplayUnits.of(m)
        keep = list(select) if select is not None else self.selected_names()
        flt = self.filter.text().strip().lower()
        grp = self.group.currentText()
        use: Dict[str, int] = {}
        for fr in m.frames.values():
            use[fr.section] = use.get(fr.section, 0) + 1
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        rows = []
        for name, sec in m.frame_sections.items():
            g = section_group(sec, m)
            if grp != "All types" and g != grp:
                continue
            texts = (name, type_label(sec), sec.material, dims_text(sec, du), str(use.get(name, 0)))
            if flt and not any(flt in t.lower() for t in texts[:4]):
                continue
            rows.append(texts)
        self.table.setRowCount(len(rows))
        for i, texts in enumerate(rows):
            for j, t in enumerate(texts):
                it = QtWidgets.QTableWidgetItem(t)
                it.setToolTip(t)
                if j == 4:
                    it.setData(QtCore.Qt.DisplayRole, int(t))
                if j == 0:
                    sec = m.frame_sections[t]
                    lo = section_outline_cached(sec)
                    if lo is not None:
                        mat = m.materials.get(sec.material)
                        it.setIcon(QtGui.QIcon(paint_icon(lo[0], lo[1], mat is not None and mat.type in STEEL_TYPES, 20)))
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
        self.count.setText(f"{len(rows)} of {len(m.frame_sections)} sections")
        if keep:
            self.select_names(keep)
        self._selected()

    def selected_names(self) -> List[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        out = []
        for r in rows:
            it = self.table.item(r, 0)
            if it is not None:
                out.append(it.text())
        return out

    def select_names(self, names: Sequence[str]):
        sm = self.table.selectionModel()
        sm.clearSelection()
        first = None
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() in names:
                self.table.selectRow(r) if first is None else sm.select(
                    self.table.model().index(r, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
                first = r if first is None else first
        if first is not None:
            self.table.scrollToItem(self.table.item(first, 0))

    def _selected(self):
        names = self.selected_names()
        one = len(names) == 1
        self.btn_copy.setEnabled(one)
        self.btn_modify.setEnabled(one)
        self.btn_delete.setEnabled(bool(names))
        if not one:
            self.preview.set_section(None, self.model)
            self.preview.message = f"{len(names)} sections selected" if names else "No section selected"
            self.props.setRowCount(0)
            self.title.setText("")
            return
        sec = self.model.frame_sections.get(names[0])
        self.title.setText(f"{sec.name}  -  {type_label(sec)}")
        self.preview.set_section(sec, self.model)
        props = section_properties(sec, True, self.model.materials) if (shape_of(sec) or sec.geometry) else None
        if props is not None and not sec.geometry and (sec.source or abs(props.A - sec.A) > 1e-9 * max(sec.A, 1e-30)):
            # tabulated / stored values win (library, imported)
            props.A, props.I33, props.I22, props.J, props.As2, props.As3 = sec.A, sec.I33, sec.I22, sec.J, sec.As2, sec.As3
            props.J_method = f"stored ({sec.source})" if sec.source else "stored"
        n = len(self.model.frame_section_users(sec.name))
        rows = property_rows(self.model, sec, props)
        rows.append(("Used by", f"{n} member(s)", ""))
        fill_table(self.props, rows)

    # -- actions
    def _store(self, sec: FrameSection, old: Optional[str] = None):
        def edit(m):
            if old is not None and old != sec.name:
                m.rename_frame_section(old, sec.name)
            m.frame_sections[sec.name] = sec
        self.doc.edit(edit)
        self.refresh(select=[sec.name])

    def edit_section(self, sec: FrameSection, new: bool) -> Optional[FrameSection]:
        """Open the form of the section's kind; the edited section or None."""
        names = list(self.model.frame_sections)
        if composite_dims(sec) is not None:
            d = CompositeSectionDialog(self.model, sec, self, existing=names, new=new)
        elif sec.geometry or sec.shape == DESIGNER:
            from .section_designer import SectionDesignerDialog
            d = SectionDesignerDialog(self.model, sec, self, existing=names)
        elif shape_of(sec) is not None and not sec.nonprismatic:
            d = ShapeSectionDialog(self.model, sec, self, existing=names, new=new)
        elif sec.nonprismatic:
            QtWidgets.QMessageBox.information(self, "Nonprismatic", "Nonprismatic sections are edited in "
                                              "Define > Section Properties > Nonprismatic Sections.")
            return None
        else:
            d = GeneralSectionDialog(self.model, sec, self, existing=names)
        if d.exec() != QtWidgets.QDialog.Accepted:
            return None
        return getattr(d, "result_section", None)

    def new_section(self, key: str, over: dict, kind: str) -> Optional[FrameSection]:
        m = self.model
        if kind == "Steel" and not any(x.type in STEEL_TYPES for x in m.materials.values()):
            st = steel_material(m.units)  # the first steel section: a steel to go with it (undoable)
            self.doc.edit(lambda mm: mm.materials.__setitem__(st.name, st))
            m = self.model
        mat = material_for(m, kind)
        if key == DESIGNER:
            return FrameSection(name=unique_name("SD1", list(m.frame_sections)), material=mat, shape=DESIGNER)
        if key == GENERAL:
            return FrameSection(name=unique_name("GEN1", list(m.frame_sections)), material=mat, shape=GENERAL)
        if key in (COMPOSITE_RECT, COMPOSITE_CIRCLE):
            if not any(x.type in STEEL_TYPES for x in m.materials.values()):
                st = steel_material(m.units)
                self.doc.edit(lambda mm: mm.materials.__setitem__(st.name, st))
                m = self.model
            L = 1.0 / m.units.factor("length")
            kind = "circle" if key == COMPOSITE_CIRCLE else "rect"
            outline = {"t3": 0.7 * L} if kind == "circle" else {"t3": 0.6 * L, "t2": 0.6 * L}
            steel = {"t3": 0.3 * L, "t2": 0.3 * L, "tf": 0.015 * L, "tw": 0.01 * L, "r": 0.018 * L}  # H300x300x10x15
            sec = FrameSection(name=unique_name("SRC1", list(m.frame_sections)), material=material_for(m, "Concrete"),
                               shape="Composite", geometry=composite_geometry(kind, outline, steel,
                                                                              material_for(m, "Steel")),
                               params={"composite": kind}, design_type="Column")
            sec.update_properties(True, m.materials)
            return sec
        v = si_dims(key, over, m.units)
        name = suggest_name(key, {k: (x or 0) * m.units.factor("length") for k, x in v.items()}, list(m.frame_sections))
        return FrameSection.from_shape(key, name, mat, **v)

    def add(self):
        d = SectionTypeDialog(self)
        if d.exec() != QtWidgets.QDialog.Accepted or d.choice is None:
            return
        key, over, kind = d.choice
        if key == "library":
            self.import_library()
            return
        sec = self.new_section(key, over, kind)
        out = self.edit_section(sec, new=True)
        if out is not None:
            self._store(out)

    def copy_section(self):
        names = self.selected_names()
        if len(names) != 1:
            return
        sec = copy.deepcopy(self.model.frame_sections[names[0]])
        sec.name = unique_name(f"{sec.name}-copy", list(self.model.frame_sections))
        out = self.edit_section(sec, new=False)
        if out is not None:
            if out.name in self.model.frame_sections:
                out.name = unique_name(out.name, list(self.model.frame_sections))
            self._store(out)

    def modify(self):
        names = self.selected_names()
        if len(names) != 1:
            return
        old = names[0]
        out = self.edit_section(copy.deepcopy(self.model.frame_sections[old]), new=False)
        if out is not None:
            self._store(out, old=old)

    def delete(self):
        names = self.selected_names()
        if not names:
            return
        used = {n: len(self.model.frame_section_users(n)) for n in names}
        busy = [f"{n} ({k} members)" for n, k in used.items() if k]
        free = [n for n, k in used.items() if not k]
        if busy:
            QtWidgets.QMessageBox.warning(self, "Sections in use",
                                          "Not deleted - assigned to members:\n" + "\n".join(busy) +
                                          "\n\nAssign another section to those members first.")
        if not free:
            return
        if len(free) > 1 and QtWidgets.QMessageBox.question(
                self, "Delete sections", f"Delete {len(free)} sections?\n" + ", ".join(free)) != QtWidgets.QMessageBox.Yes:
            return
        self.doc.edit(lambda m: [m.frame_sections.pop(n, None) for n in free])
        self.refresh(select=[])

    def import_library(self):
        if not any(x.type in STEEL_TYPES for x in self.model.materials.values()):
            st = steel_material(self.model.units)  # steel sections need a steel (undoable)
            self.doc.edit(lambda mm: mm.materials.__setitem__(st.name, st))
        d = LibraryDialog(self.model, self)
        if d.exec() != QtWidgets.QDialog.Accepted or not d.sections:
            return
        secs = d.sections

        def edit(m):
            for s in secs:
                m.frame_sections[s.name] = s
        self.doc.edit(edit)
        self.refresh(select=[s.name for s in secs])
