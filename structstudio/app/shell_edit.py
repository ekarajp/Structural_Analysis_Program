"""*Edit > Edit Shells*: Merge Shells, Expand/Shrink Shells, Divide Walls for Openings.

Like :mod:`.commands`, every function is one undoable :class:`~.document.Document`
edit with no Qt dependency.  Each returns the affected area names and a list of
notes (objects that were skipped and why) for the status bar / message box.
Area loads (load panels made from an area) follow the area automatically
(:func:`structstudio.model.area_panels.sync`).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from ..mesh.geometry import point_on_segment
from ..mesh import polygon2d as pg
from ..model import Model
from .commands import _area_kwargs, _unused_points
from .document import Document

Result = Tuple[List[str], List[str]]


def _xyz(m: Model, names: Sequence[str]) -> np.ndarray:
    return np.array([m.points[p].xyz for p in names], float)


def _tol(m: Model) -> float:
    return max(float(m.options.merge_tolerance or 1e-3), 1e-6)


def _used_outside(m: Model, exclude_areas: Set[str]) -> Set[str]:
    """Points used by anything except the areas in ``exclude_areas`` (or carrying
    their own supports, loads, springs or mass)."""
    used: Set[str] = set()
    for f in m.frames.values():
        used |= {f.point_i, f.point_j}
    for a in m.areas.values():
        if a.name not in exclude_areas:
            used |= set(a.points)
    for s in getattr(m, "line_springs", {}).values():
        used |= {s.point_i, s.point_j}
    for n, p in m.points.items():
        if any(p.restraints) or p.loads or any(p.springs) or any(p.mass):
            used.add(n)
    return used


def _drop_points(model: Model, candidates: Set[str]) -> None:
    for n in set(_unused_points(model)) & candidates:
        model.points.pop(n, None)


def _panels_of(m: Model, area: str):
    return [(p.distribution, p.loads) for p in m.panel_loads.values() if p.area == area]


# ------------------------------------------------------------- Merge Shells
def merge_shells(doc: Document, areas: Sequence[str]) -> Result:
    """Selected coplanar shells with the same section and type that share edges
    become one shell (the first selected keeps its name, properties and loads).

    Edges that only partly coincide (a T-junction) are handled: the joints of the
    neighbours on an edge are inserted first.  Joints left in the middle of a
    straight edge are removed unless something else uses them.  Groups that would
    leave a hole, only touch at a corner, or overlap are skipped.
    """
    m = doc.model
    tol = _tol(m)
    names = [n for n in dict.fromkeys(areas) if n in m.areas]
    notes: List[str] = []
    groups: List[dict] = []
    for n in names:
        a = m.areas[n]
        P = _xyz(m, a.points)
        try:
            fr = pg.plane_frame(P)
        except ValueError:
            notes.append(f"{n}: degenerate shape")
            continue
        for g in groups:
            if g["section"] == a.section and g["type"] == a.type and abs(fr[3] @ g["frame"][3]) > 1 - 1e-6 \
                    and pg.off_plane(P, g["frame"]) <= tol:
                g["names"].append(n)
                break
        else:
            groups.append({"section": a.section, "type": a.type, "frame": fr, "names": [n]})

    plans = []  # (keep, new point list, removed areas)
    for g in groups:
        if len(g["names"]) < 2:
            continue
        nrm = g["frame"][3]
        polys: Dict[str, List[str]] = {}
        flipped: Dict[str, bool] = {}
        for n in g["names"]:
            pts = list(m.areas[n].points)
            flip = pg.newell_normal(_xyz(m, pts)) @ nrm < 0
            polys[n] = pts[::-1] if flip else pts
            flipped[n] = flip
        group_pts = {p for pts in polys.values() for p in pts}
        # T-junctions: joints of the neighbours lying on an edge become vertices
        for n, pts in polys.items():
            out = []
            for i, a_ in enumerate(pts):
                b_ = pts[(i + 1) % len(pts)]
                xa, xb = np.array(m.points[a_].xyz), np.array(m.points[b_].xyz)
                on = []
                for q in group_pts - {a_, b_}:
                    t = point_on_segment(np.array(m.points[q].xyz), xa, xb, tol)
                    if t is not None:
                        on.append((t, q))
                out.append(a_)
                out += [q for _, q in sorted(on)]
            polys[n] = out
        # components of shells joined through shared edges
        count: Dict[frozenset, int] = {}
        owner: Dict[frozenset, List[str]] = {}
        for n, pts in polys.items():
            for i in range(len(pts)):
                k = frozenset((pts[i], pts[(i + 1) % len(pts)]))
                count[k] = count.get(k, 0) + 1
                owner.setdefault(k, []).append(n)
        parent = {n: n for n in polys}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for k, own in owner.items():
            for o in own[1:]:
                parent[find(o)] = find(own[0])
        comps: Dict[str, List[str]] = {}
        for n in g["names"]:
            comps.setdefault(find(n), []).append(n)
        for comp in comps.values():
            if len(comp) < 2:
                notes.append(f"{comp[0]}: shares no edge with the other selected shells")
                continue
            nxt: Dict[str, List[str]] = {}
            for n in comp:
                pts = polys[n]
                for i in range(len(pts)):
                    a_, b_ = pts[i], pts[(i + 1) % len(pts)]
                    if count[frozenset((a_, b_))] == 1:
                        nxt.setdefault(a_, []).append(b_)
            label = "+".join(comp)
            if any(len(v) > 1 for v in nxt.values()):
                notes.append(f"{label}: the shells only touch at a corner - not merged")
                continue
            start = polys[comp[0]][0] if polys[comp[0]][0] in nxt else next(iter(nxt))
            loop, cur = [start], nxt[start][0]
            while cur is not None and cur != start and len(loop) <= len(nxt):
                loop.append(cur)
                cur = nxt[cur][0] if cur in nxt else None
            if cur != start or len(loop) != len(nxt):
                notes.append(f"{label}: the merged shape would have a hole - not merged")
                continue
            fr = g["frame"]
            total = sum(abs(pg.signed_area(pg.to_2d(_xyz(m, polys[n]), fr))) for n in comp)
            merged_area = abs(pg.signed_area(pg.to_2d(_xyz(m, loop), fr)))
            if abs(merged_area - total) > 1e-6 * max(total, 1.0) + tol * tol:
                notes.append(f"{label}: the shells overlap - not merged")
                continue
            # joints left in the middle of a straight edge go, unless used elsewhere
            keep_used = _used_outside(m, set(comp))
            changed = True
            while changed and len(loop) > 3:
                changed = False
                for i, p in enumerate(loop):
                    if p in keep_used:
                        continue
                    a_, b_ = loop[i - 1], loop[(i + 1) % len(loop)]
                    if point_on_segment(np.array(m.points[p].xyz), np.array(m.points[a_].xyz),
                                        np.array(m.points[b_].xyz), tol) is not None:
                        loop.pop(i)
                        changed = True
                        break
            if flipped[comp[0]]:
                loop = [loop[0]] + loop[1:][::-1]
            keep = comp[0]
            for n in comp[1:]:
                if m.areas[n].loads != m.areas[keep].loads or _panels_of(m, n) != _panels_of(m, keep):
                    notes.append(f"{n}: its loads differ from {keep}'s - the merged shell has {keep}'s loads")
            plans.append((keep, loop, comp[1:]))

    if not plans:
        return [], notes

    def mutate(model: Model):
        old: Set[str] = set()
        for keep, loop, removed in plans:
            for n in [keep] + list(removed):
                old |= set(model.areas[n].points)
            a = model.areas[keep]
            a.points = list(loop)
            if len(loop) != 4:
                a.mesh_divisions = None
            for n in removed:
                model.areas.pop(n)
            a.story = model.story_for(a, "area") or a.story
        _drop_points(model, old)

    doc.edit(mutate)
    return [k for k, _, _ in plans], notes


# ------------------------------------------------------- Expand/Shrink Shells
def expand_shells(doc: Document, areas: Sequence[str], offset: float) -> Result:
    """Every edge of each selected shell moves outward by ``offset`` (negative:
    inward), ETABS *Expand/Shrink Areas*.  Neighbouring objects are not changed;
    new corners snap to an existing joint at the same place.  A shell that would
    turn inside out or cross itself is skipped."""
    m = doc.model
    tol = _tol(m)
    names = [n for n in dict.fromkeys(areas) if n in m.areas]
    notes: List[str] = []
    plans = []
    if abs(offset) <= tol:
        return [], ["offset is zero"]
    for n in names:
        a = m.areas[n]
        P = _xyz(m, a.points)
        try:
            fr = pg.plane_frame(P)
        except ValueError:
            notes.append(f"{n}: degenerate shape")
            continue
        Q = pg.to_2d(P, fr)  # counter-clockwise by construction (Newell normal)
        R = pg.offset_polygon(Q, offset)
        ok = pg.signed_area(R) > tol * tol and not pg.self_intersects(R, 1e-9)
        n_ = len(Q)
        for i in range(n_):
            d0 = Q[(i + 1) % n_] - Q[i]
            d1 = R[(i + 1) % n_] - R[i]
            if d0 @ d1 <= 0:
                ok = False
        if not ok:
            notes.append(f"{n}: shrinking by {abs(offset):g} would turn it inside out - skipped")
            continue
        plans.append((n, pg.to_3d(R, fr)))
    if not plans:
        return [], notes

    def mutate(model: Model):
        old: Set[str] = set()
        for n, X in plans:
            a = model.areas[n]
            old |= set(a.points)
            a.points = [model.add_point(*map(float, x)).name for x in X]
            a.story = model.story_for(a, "area") or a.story
        _drop_points(model, old)

    doc.edit(mutate)
    return [n for n, _ in plans], notes


# ------------------------------------------------ Divide Walls for Openings
def _openings_in(m: Model, host: str, fr, Q: np.ndarray, tol: float):
    found, notes = [], []
    for o in m.areas.values():
        if o.name == host or not o.is_opening:
            continue
        P = _xyz(m, o.points)
        if pg.off_plane(P, fr) > tol:
            continue
        H = pg.to_2d(P, fr)
        if not all(pg.inside(q, Q, edge_tol=tol) for q in H):
            c = H.mean(axis=0)
            if pg.inside(c, Q):
                notes.append(f"{o.name}: runs past the edge of {host} - ignored")
            continue
        if not pg.axis_aligned(H, tol):
            notes.append(f"{o.name}: not a rectangle lined up with {host} - ignored")
            continue
        found.append((o.name, H))
    return found, notes


def divide_for_openings(doc: Document, areas: Sequence[str]) -> Result:
    """ETABS *Divide Walls for Openings*: each selected wall (or floor) with
    openings drawn in it is cut along the lines through the opening corners into
    rectangular pieces; the pieces inside an opening are left out.  The pieces
    keep the shell's section, properties (pier / spandrel labels included) and
    loads, and share whole edges so they mesh compatibly.  Selecting an opening
    also divides the shell it is in.  The opening objects stay in the model.

    The analysis cuts openings out automatically (:mod:`structstudio.mesh.openings`);
    this command is for when the pieces themselves are wanted (separate pier /
    spandrel labels, assignments per piece).  Shell and openings must be
    rectangular and lined up.
    """
    m = doc.model
    tol = _tol(m)
    sel = [n for n in dict.fromkeys(areas) if n in m.areas]
    hosts = [n for n in sel if not m.areas[n].is_opening]
    for o in [n for n in sel if m.areas[n].is_opening]:
        Po = _xyz(m, m.areas[o].points)
        c = Po.mean(axis=0)
        for a in m.areas.values():
            if a.is_opening or a.name in hosts:
                continue
            P = _xyz(m, a.points)
            try:
                fr = pg.plane_frame(P)
            except ValueError:
                continue
            if pg.off_plane(Po, fr) <= tol and pg.inside(pg.to_2d(c[None], fr)[0], pg.to_2d(P, fr)):
                hosts.append(a.name)
    notes: List[str] = []
    plans = []
    for n in hosts:
        a = m.areas[n]
        P = _xyz(m, a.points)
        try:
            fr = pg.plane_frame(P)
        except ValueError:
            notes.append(f"{n}: degenerate shape")
            continue
        Q = pg.to_2d(P, fr)
        if not pg.axis_aligned(Q, tol):
            notes.append(f"{n}: not rectangular (only rectangular / L-shaped walls and floors can be divided)")
            continue
        ops, why = _openings_in(m, n, fr, Q, tol)
        notes += why
        if not ops:
            notes.append(f"{n}: no openings in it")
            continue
        U, V, cells = pg.grid_cells(Q, [h for _, h in ops], tol)
        plans.append((n, fr, U, V, cells))
    if not plans:
        return [], notes
    created: List[str] = []

    def mutate(model: Model):
        old: Set[str] = set()
        for n, fr, U, V, cells in plans:
            a = model.areas.pop(n)
            old |= set(a.points)
            originals = [p for p in model.panel_loads.values() if p.area == n]
            for p in originals:
                del model.panel_loads[p.name]
            for k, l in cells:
                X = pg.to_3d(pg.cell_corners(U, V, k, l), fr)
                pts = [model.add_point(*map(float, x)).name for x in X]
                b = model.add_area(pts, a.section, type=a.type, **_area_kwargs(a))
                b.loads = copy.deepcopy(a.loads)
                b.label = a.label
                b.group = a.group
                b.mesh_divisions = None
                for p in originals:  # every piece keeps the shell's area loads
                    q = model.add_area_panel(b.name, copy.deepcopy(p.loads), distribution=p.distribution,
                                             span_dir=p.span_dir)
                    q.receivers, q.label = p.receivers, p.label
                created.append(b.name)
        _drop_points(model, old)

    doc.edit(mutate)
    return created, notes
