"""Area loads are load panels (one form for every surface load).

A panel made from an area object carries ``panel.area``.  With the distribution
"shells" it is the ETABS / SAP2000 *uniform area load* on that area; "one_way" /
"two_way" carry the load to the frames around the area (CYPE 3D panels).  Point
and frame loads stay on their joints and frames.

Area loads belong to their area; drawn panels (no ``panel.area``) are load areas
of their own, independent of the floors: they move or go away only when the user
moves or deletes them.

Rules after an edit (:func:`sync`, called by the document after every change):

* the area exists: its panels take the area's current polygon (moved, replicated,
  reshaped areas keep their loads; Divide Shells gives every piece the loads);
* the area is gone: its area loads go with it;
* drawn panels are never changed.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from .objects import AreaLoad, PanelLoad, PanelLoadCase


def _tol(model) -> float:
    return max(float(getattr(model.options, "merge_tolerance", 1e-3) or 1e-3), 1e-6)


def area_polygon(model, a) -> List[tuple]:
    return [tuple(float(v) for v in model.points[p].xyz) for p in a.points]


def sync(model) -> None:
    for panel in list(model.panel_loads.values()):
        if not panel.area:
            continue
        a = model.areas.get(panel.area)
        if a is not None and all(p in model.points for p in a.points):
            panel.points = area_polygon(model, a)
            panel.story = a.story
            continue
        del model.panel_loads[panel.name]  # the area was deleted: its loads go with it


def areas_in_polygon(model, poly) -> List[str]:
    """Areas in the plane of ``poly`` whose centroid lies inside it."""
    from ..mesh.panel_loads import point_in_poly2, polygon_frame
    P = np.asarray(poly, float)
    if len(P) < 3:
        return []
    try:
        o, e1, e2, n = polygon_frame(P)
    except ValueError:
        return []
    P2 = np.array([[(p - o) @ e1, (p - o) @ e2] for p in P])
    tol = 2 * _tol(model)
    out = []
    for a in model.areas.values():
        if a.type == "Opening":
            continue
        try:
            A = np.array([model.points[p].xyz for p in a.points], float)
        except KeyError:
            continue
        if all(abs(float((q - o) @ n)) <= tol for q in A):
            c = A.mean(axis=0) - o
            if point_in_poly2(np.array([c @ e1, c @ e2]), P2):
                out.append(a.name)
    return out


def convert_area_loads(model) -> int:
    """Move the loads stored on area objects into area-load panels ("shells", one
    panel per area and load pattern).  Returns the number of panels made."""
    made = 0
    for a in model.areas.values():
        if not a.loads:
            continue
        by_pat: Dict[str, List[AreaLoad]] = {}
        for ld in a.loads:
            by_pat.setdefault(ld.pattern, []).append(ld)
        a.loads = []
        for items in by_pat.values():
            for ld in items:
                model._require_pattern(ld.pattern)
            model.add_area_panel(a.name, [PanelLoadCase(ld.pattern, ld.value, ld.direction) for ld in items])
            made += 1
    return made


def area_items(panel: PanelLoad) -> List[AreaLoad]:
    """Area loads of a "shells" area panel on its own area.  "Normal" is the
    area's local 3 axis; a load varying with height takes its value at the
    area centroid."""
    out = []
    zs = [p[2] for p in panel.points] or [0.0]
    zc = sum(zs) / len(zs)
    z0, z1 = min(zs), max(zs)
    for case in panel.loads:
        d = case.direction
        if str(d).strip().lower() in ("normal", "local 3", "local3"):
            d = "3"
        if case.varying:
            v = case.value_lower if z1 - z0 <= 1e-12 else \
                case.value_lower + (case.value_upper - case.value_lower) * (zc - z0) / (z1 - z0)
        else:
            v = case.value
        out.append(AreaLoad(case.pattern, float(v), d))
    return out


def panels_of(model, area: str) -> List[str]:
    return [p.name for p in model.panel_loads.values() if p.area == area]


def copy_area_panels(model, src: str, dst: str) -> List[str]:
    """Copy the panels made from area ``src`` onto area ``dst`` (replicate, paste)."""
    made = []
    for p in [p for p in model.panel_loads.values() if p.area == src]:
        q = model.add_area_panel(dst, copy.deepcopy(p.loads), distribution=p.distribution, span_dir=p.span_dir)
        q.receivers = None if p.receivers is None else list(p.receivers)
        q.label = p.label
        made.append(q.name)
    return made
