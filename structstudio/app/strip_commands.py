"""Design strip commands (Draw > Draw Design Strips, Edit > Add/Edit Design Strips,
Reshape of a strip, the strip's properties form).

A design strip is design data: every command here is an :meth:`Document.edit_design`
edit - one undo step that keeps the analysis results and the lock (the analysis does
not depend on the strips)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..design import strips as S
from ..model import ModelError

XYZ = Tuple[float, float, float]


def _widths_auto(model, ds) -> None:
    """Work out the widths of one strip again (its ``auto_width`` on): a column strip
    from the supports and the other column strips of its layer (ACI 318 8.4.1.5), a
    middle strip to the column strips beside it (or the slab edge)."""
    if ds.story is None or ds.story not in model.stories:
        return
    if ds.kind == "Middle":
        ws = S.middle_widths(model, ds.story, ds.points, ds.layer)
    else:
        got, _, _ = S.auto_widths(model, ds.story, names=[ds.name])
        ws = got.get(ds.name)
    if ws and len(ws) == len(ds.points) - 1:
        ds.widths = [tuple(float(v) for v in w) for w in ws]


def draw_strip(doc, points: Sequence[XYZ], layer: str = "A", kind: str = "Column", widths: str = "Auto",
               left: float = 1.0, right: float = 1.0) -> str:
    """A strip along the clicked points (Draw Design Strips); ``widths`` "Auto" or
    "Specified" (``left`` / ``right`` for every segment)."""
    pts = [tuple(float(c) for c in p) for p in points]
    if len(pts) < 2:
        raise ModelError("STRIP_POINTS", "A design strip needs at least two points")
    auto = widths == "Auto"
    if not auto and (left < 0 or right < 0 or not (math.isfinite(left) and math.isfinite(right))):
        raise ModelError("STRIP_WIDTHS", "Strip widths must be zero or more")
    if not auto and left + right <= 0:
        raise ModelError("STRIP_WIDTHS", "A strip of no width designs nothing: give it a width (or Auto)")
    made = []

    def mutate(model):
        ds = model.add_design_strip(pts, (0.0, 0.0) if auto else (left, right), layer=layer, kind=kind,
                                    auto_width=auto)
        if ds.story is None:
            del model.design_strips[ds.name]
            raise ModelError("NOT_AT_STORY", "Draw a design strip in a plan of a story (on the slab)")
        if auto:
            _widths_auto(model, ds)
            if not any(v > 0 for w in ds.widths for v in w):
                del model.design_strips[ds.name]
                raise ModelError("NO_SLAB", "No slab under the strip to work its width out from - draw it over "
                                            "a floor slab, or give it a width (Widths: Specified)")
        made.append(ds.name)

    doc.edit_design(mutate)
    return made[0] if made else ""


@dataclass
class AutoReport:
    made: Dict[str, int] = field(default_factory=dict)  # story -> strips made
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    removed: int = 0


def coverage_note(model, story: str) -> List[str]:
    """Warnings when the strips of a layer leave part of the slab out, or overlap."""
    out = []
    layers = sorted({ds.layer for ds in model.design_strips.values() if ds.story == story})
    for L in layers:
        if L == "Other":
            continue
        c = S.coverage(model, story, L)
        if not c["samples"]:
            continue
        if c["uncovered"] > 0.02:
            x, y = c["where_uncovered"]
            out.append(f"{story} layer {L}: {100 * c['uncovered']:.0f}% of the slab is under no strip "
                       f"(e.g. at x = {x:.3g}, y = {y:.3g}) - its reinforcement would not be designed")
        if c["overlap"] > 0.02:
            x, y = c["where_overlap"]
            out.append(f"{story} layer {L}: strips overlap over {100 * c['overlap']:.0f}% of the slab "
                       f"(e.g. at x = {x:.3g}, y = {y:.3g}) - that slab would be designed twice")
    return out


def auto_strips(doc, stories: Sequence[str], options: Optional[S.StripOptions] = None) -> AutoReport:
    """Edit > Add/Edit Design Strips > Auto Design Strips: the strips of each story laid
    out again from its supports (one undo step).  The strips Auto made before are
    replaced; hand-drawn ones stay (``keep_user``) or go too."""
    opt = options or S.StripOptions()
    rep = AutoReport()
    layouts = {}
    for st in stories:
        if st not in doc.model.stories:
            continue
        lay = S.layout(doc.model, st, opt)
        layouts[st] = lay
        rep.notes += [f"{st}: {n}" if not n.startswith(st) else n for n in lay.notes]
        rep.warnings += lay.warnings
    if not any(lay.strips for lay in layouts.values()):
        if not rep.warnings:
            rep.warnings.append("No strip was made")
        return rep

    def mutate(model):
        for st, lay in layouts.items():
            if not lay.strips:
                continue
            gone = [n for n, ds in model.design_strips.items() if ds.story == st and
                    (ds.auto or not opt.keep_user) and ds.layer in tuple(opt.layers) + ("Other",)]
            for n in gone:
                del model.design_strips[n]
            rep.removed += len(gone)
            keep = [ds for ds in model.design_strips.values() if ds.story == st]
            n = 0
            for kw in lay.strips:
                if opt.keep_user and _duplicates_user(kw, keep, model):
                    continue  # a hand-drawn strip along the same line stays (no second one)
                model.add_design_strip(story=st, **kw)
                n += 1
            rep.made[st] = n

    doc.edit_design(mutate)
    for st in layouts:
        rep.warnings += coverage_note(doc.model, st)
    return rep


def _duplicates_user(kw, keep, model) -> bool:
    tol = max(model.options.merge_tolerance, 1e-9)
    on = max(0.3 / model.units.factor("length"), tol)
    pts = [(p[0], p[1]) for p in kw["points"]]
    for ds in keep:
        if ds.layer != kw["layer"] or ds.kind != kw["kind"]:
            continue
        q = [(p[0], p[1]) for p in ds.points]
        cum = S._cum(q)
        if all(S._project(q, cum, p)[1] <= on for p in pts):
            return True
    return False


def auto_widths(doc, stories: Sequence[str], names: Optional[Sequence[str]] = None,
                remake_middle: bool = True) -> AutoReport:
    """Edit > Add/Edit Design Strips > Auto Strip Widths: the widths of the column strips
    worked out again (``names``, else those with automatic widths) and the middle strips
    made again in the gaps (the middle strips with automatic widths are replaced)."""
    rep = AutoReport()
    results = {}
    for st in stories:
        if st not in doc.model.stories:
            continue
        got, middles, notes = S.auto_widths(doc.model, st, names=None if names is None else
                                            [n for n in names if doc.model.design_strips.get(n) is not None
                                             and doc.model.design_strips[n].story == st])
        results[st] = (got, middles)
        rep.notes += notes
    if not any(got or (remake_middle and mids) for got, mids in results.values()):
        rep.warnings.append("No column strip with automatic widths on these stories (Auto Design Strips makes "
                            "them; a strip drawn with Specified widths keeps its widths)")
        return rep

    def mutate(model):
        for st, (got, middles) in results.items():
            for n, ws in got.items():
                if n in model.design_strips:
                    model.design_strips[n].widths = [tuple(float(v) for v in w) for w in ws]
            rep.made[st] = len(got)
            if not remake_middle:
                continue
            # the middle strips Auto made go and are made again in the gaps; hand-drawn ones stay
            # (with automatic widths: worked out again, to the column strips as they are now)
            gone = [n for n, ds in model.design_strips.items() if ds.story == st and ds.kind == "Middle"
                    and ds.auto]
            for n in gone:
                del model.design_strips[n]
            rep.removed += len(gone)
            manual = [ds for ds in model.design_strips.values() if ds.story == st and ds.kind == "Middle"]
            for ds in manual:
                if ds.auto_width:
                    _widths_auto(model, ds)
            z = model.stories[st].elevation
            for fam, pts, ws in middles:
                kw = dict(points=[(p[0], p[1], z) for p in pts], widths=ws, layer=fam, kind="Middle", auto=True)
                if _duplicates_user(kw, manual, model):
                    continue
                model.add_design_strip(story=st, **kw)

    doc.edit_design(mutate)
    for st in results:
        rep.warnings += coverage_note(doc.model, st)
    return rep


def set_strip(doc, name: str, new_name: Optional[str] = None, layer: Optional[str] = None,
              kind: Optional[str] = None, widths: Optional[List[Tuple[float, float, float, float]]] = None,
              auto_width: Optional[bool] = None, label: Optional[str] = None,
              points: Optional[List[XYZ]] = None) -> str:
    """The strip's properties form: rename, layer, type, widths (or automatic ones)."""
    from ..model.objects import STRIP_KINDS, STRIP_LAYERS
    m = doc.model
    if name not in m.design_strips:
        raise ModelError("UNKNOWN_STRIP", f"No design strip {name!r}")
    if new_name is not None:
        new_name = new_name.strip()
        if not new_name:
            raise ModelError("BAD_NAME", "A strip needs a name")
        if new_name != name and new_name in m.design_strips:
            raise ModelError("DUPLICATE_NAME", f"A design strip named {new_name!r} already exists")
    if layer is not None and layer not in STRIP_LAYERS:
        raise ModelError("BAD_OPTION", f"Strip layer {layer!r}: use {'/'.join(STRIP_LAYERS)}")
    if kind is not None and kind not in STRIP_KINDS:
        raise ModelError("BAD_OPTION", f"Strip type {kind!r}: use {'/'.join(STRIP_KINDS)}")
    if widths is not None:
        widths = [tuple(float(v) for v in w) for w in widths]
        if any(len(w) != 4 or not all(math.isfinite(v) and v >= 0 for v in w) for w in widths):
            raise ModelError("STRIP_WIDTHS", "Strip widths must be zero or more")
    out = [name]

    def mutate(model):
        ds = model.design_strips[name]
        before = (list(ds.points), list(ds.widths), ds.layer, ds.kind, ds.label, ds.auto_width, ds.name)
        _change(model, ds)
        after = (list(ds.points), list(ds.widths), ds.layer, ds.kind, ds.label, ds.auto_width, ds.name)
        if after != before:
            ds.auto = False  # (changed by hand: a new Auto run keeps it)

    def _change(model, ds):
        if points is not None:
            pts = [tuple(float(c) for c in p) for p in points]
            if len(pts) < 2:
                raise ModelError("STRIP_POINTS", "A design strip needs at least two points")
            if widths is None and len(pts) - 1 != len(ds.widths):
                ds.widths = _rewidth(ds.widths, len(pts) - 1)
            ds.points = pts
            ds.story = model.strip_story(ds)
        if layer is not None:
            ds.layer = layer
        if kind is not None:
            ds.kind = kind
        if label is not None:
            ds.label = label or None
        if widths is not None:
            if len(widths) != len(ds.points) - 1:
                raise ModelError("STRIP_WIDTHS", f"The strip has {len(ds.points) - 1} segment(s): "
                                                 f"{len(widths)} width row(s) given")
            ds.widths = widths
        if auto_width is not None:
            ds.auto_width = bool(auto_width)
        if ds.auto_width and widths is None and (points is not None or layer is not None or kind is not None
                                                  or auto_width):
            _widths_auto(model, ds)
        if new_name is not None and new_name != name:
            model.design_strips = {(new_name if k == name else k): v for k, v in model.design_strips.items()}
            ds.name = new_name
            out[0] = new_name

    doc.edit_design(mutate)
    return out[0]


def _rewidth(widths, n: int):
    """Widths for ``n`` segments from those of a strip with another number of segments."""
    if not widths:
        return [(0.0, 0.0, 0.0, 0.0)] * n
    if len(widths) >= n:
        return list(widths[:n])
    return list(widths) + [widths[-1]] * (n - len(widths))


def reshape_strip(doc, name: str, k: int, xy: Tuple[float, float], insert: bool = False) -> None:
    """Reshape of a strip: centre-line point ``k`` to ``xy`` (``insert``: a new point after
    point k, the segment k split there).  A strip with automatic widths gets them again."""
    m = doc.model
    if name not in m.design_strips:
        raise ModelError("UNKNOWN_STRIP", f"No design strip {name!r}")

    def mutate(model):
        ds = model.design_strips[name]
        z = float(ds.points[0][2])
        pts = list(ds.points)
        ws = list(ds.widths)
        p = (float(xy[0]), float(xy[1]), z)
        if insert:
            if not 0 <= k < len(pts) - 1:
                raise ModelError("STRIP_POINTS", "No such strip segment")
            w = ds.segment_widths(k)
            mid = (0.5 * (w[0] + w[2]), 0.5 * (w[1] + w[3]))
            pts.insert(k + 1, p)
            ws[k:k + 1] = [(w[0], w[1], mid[0], mid[1]), (mid[0], mid[1], w[2], w[3])]
        else:
            if not 0 <= k < len(pts):
                raise ModelError("STRIP_POINTS", "No such strip point")
            pts[k] = p
        tol = max(model.options.merge_tolerance, 1e-9)
        if any(math.hypot(b[0] - a[0], b[1] - a[1]) <= tol for a, b in zip(pts[:-1], pts[1:])):
            # two points on one place: the point is dropped (with its segment)
            keep_p, keep_w = [pts[0]], []
            for q, w in zip(pts[1:], ws):
                if math.hypot(q[0] - keep_p[-1][0], q[1] - keep_p[-1][1]) > tol:
                    keep_p.append(q)
                    keep_w.append(w)
            pts, ws = keep_p, keep_w
            if len(pts) < 2:
                raise ModelError("STRIP_POINTS", "The strip would have no length - nothing was changed")
        ds.points = pts
        ds.widths = ws
        ds.story = model.strip_story(ds)
        if ds.story is None:
            raise ModelError("NOT_AT_STORY", "The strip left its story - nothing was changed")
        if ds.auto_width:
            _widths_auto(model, ds)
        ds.auto = False  # (reshaped by hand: a new Auto run keeps it)

    doc.edit_design(mutate)


def delete_strips(doc, names: Sequence[str]) -> int:
    names = [n for n in names if n in doc.model.design_strips]
    if names:
        doc.edit_design(lambda m: [m.design_strips.pop(n, None) for n in names])
    return len(names)
