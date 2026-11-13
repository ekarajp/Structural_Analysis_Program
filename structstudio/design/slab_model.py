"""Slab Design Model: one floor taken out of the building to design its slabs
(Design > Concrete Slab Design > Slab Design Model).

A *membrane* floor of the building model only hands its loads to the beams and
columns, so it has no bending moments of its own to design for.  Its slabs are
designed on a separate model of that floor (as a floor is sent from ETABS to SAFE):

* every floor slab at the story level, membrane slabs becoming plates (ShellThick,
  their gravity stiffness modifiers kept); shell floors at the level are kept as
  they are (the slab is continuous; ``include_shell=False`` leaves them out);
* the beams at the level: *flexible* (their own stiffness: the slab moments follow
  the beam deflections, as SAFE) or *rigid* (stiffness x 1000 in bending and axial:
  the slab spans between unyielding supports, as the coefficient method);
* the columns and walls one story below and above as stubs fixed (or pinned) at
  their far ends - the rotational restraint the floor gets from them;
* the gravity load patterns only (wind / seismic / notional out), the loads on the
  kept objects, the load panels of the level carried by the slab ("shells"), the
  linear static gravity cases (P-Delta / nonlinear gravity cases as linear ones)
  and the combinations made only of those.

The result is an ordinary model: it is analysed and displayed like any other.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..model.base import ModelError
from ..model.loads import LATERAL_PATTERN_TYPES, LinearStaticCase, LoadCombination, PDeltaCase
from ..model.objects import Point

FIXED = (True, True, True, True, True, True)
PINNED = (True, True, True, False, False, False)
RIGID_FACTOR = 1000.0
SIN_VERTICAL = 1e-3


@dataclass
class SlabModelResult:
    model: object
    story: str
    notes: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)


def _tol(model) -> float:
    return max(model.options.merge_tolerance, 1e-9)


def _is_slab_floor(model, a) -> bool:
    if a.type not in ("Floor", "Ramp") or a.section is None:
        return False
    sec = model.shell_sections.get(a.section)
    return sec is not None and sec.type == "Slab"


def _is_membrane(model, a) -> bool:
    return _is_slab_floor(model, a) and model.shell_sections[a.section].model_type == "Membrane"


def _area_at(model, a, z, tol) -> bool:
    return bool(a.points) and all(p in model.points and abs(model.points[p].z - z) <= tol for p in a.points)


def membrane_stories(model) -> List[str]:
    """Stories (bottom up) with membrane floor slabs - the floors a Slab Design Model is for."""
    tol = _tol(model)
    out = []
    for st in model.stories_bottom_up():
        if any(_is_membrane(model, a) and _area_at(model, a, st.elevation, tol) for a in model.areas.values()):
            out.append(st.name)
    return out


def build_slab_model(model, story: str, beams: str = "flexible", columns: str = "fixed",
                     include_shell: bool = True) -> SlabModelResult:
    """The Slab Design Model of ``story`` (see the module text).  ``beams``: "flexible" |
    "rigid"; ``columns``: "fixed" | "pinned" (the far ends of the column / wall stubs)."""
    if beams not in ("flexible", "rigid") or columns not in ("fixed", "pinned"):
        raise ModelError("BAD_OPTION", f"beams {beams!r} / columns {columns!r}: use flexible|rigid, fixed|pinned")
    if story not in model.stories:
        raise ModelError("NO_STORY", f"Story {story!r} does not exist")
    from ..model import Model
    m = Model.from_dict(model.to_dict())
    m.name = f"{model.name} - Slab Design Model {story}" if getattr(model, "name", "") else f"Slab Design Model {story}"
    notes: List[str] = []
    tol = _tol(m)
    z = m.stories[story].elevation
    story_z = sorted(s.elevation for s in m.stories.values())

    def at_level(pt: str) -> bool:
        return abs(m.points[pt].z - z) <= tol

    # ---- the floor
    floors = [a for a in m.areas.values() if _is_slab_floor(m, a) and _area_at(m, a, z, tol)]
    if not any(_is_membrane(m, a) for a in floors):
        raise ModelError("NO_MEMBRANE_FLOOR", f"Story {story} has no membrane floor slab: its slabs (shells) are "
                                              "designed with the forces of the building model")
    if not include_shell:
        floors = [a for a in floors if _is_membrane(m, a)]
    keep_areas: Set[str] = {a.name for a in floors}
    for a in m.areas.values():  # openings of the level
        if a.name not in keep_areas and (a.type == "Opening" or a.section is None) and _area_at(m, a, z, tol):
            keep_areas.add(a.name)
    level_pts: Set[str] = {p for n in keep_areas for p in m.areas[n].points}

    def vertical(f) -> bool:
        a, b = m.points[f.point_i], m.points[f.point_j]
        v = np.array([b.x - a.x, b.y - a.y, b.z - a.z])
        L = np.linalg.norm(v)
        return L > 0 and np.linalg.norm(v[:2]) / L < SIN_VERTICAL

    # the beams: frames lying at the level and touching the floor
    keep_frames: Set[str] = set()
    for f in m.frames.values():
        if f.type == "Null" or not (at_level(f.point_i) and at_level(f.point_j)):
            continue
        keep_frames.add(f.name)
    level_pts |= {p for n in keep_frames for p in (m.frames[n].point_i, m.frames[n].point_j)}
    # ---- supports: column and wall stubs one story up and down, fixed at the far end
    adj: Dict[str, List[str]] = {}
    for f in m.frames.values():
        adj.setdefault(f.point_i, []).append(f.name)
        adj.setdefault(f.point_j, []).append(f.name)
    far_pts: Set[str] = set()
    n_cols = 0
    new_frames = []
    for pt in sorted(level_pts):
        for up in (False, True):
            first = None
            cur, seen = pt, {pt}
            while True:
                nxt_frame = None
                for fn in adj.get(cur, []):
                    f = m.frames[fn]
                    if f.type in ("Null",) or not vertical(f):
                        continue
                    other = f.point_j if f.point_i == cur else f.point_i
                    if (m.points[other].z > m.points[cur].z) == up and other not in seen:
                        nxt_frame = (fn, other)
                        break
                if nxt_frame is None:
                    break
                fn, other = nxt_frame
                first = first or fn
                seen.add(other)
                cur = other
                if any(abs(m.points[cur].z - zs) <= tol for zs in story_z):
                    break
            if first is None:
                continue
            f0 = m.frames[first]
            stub = dataclasses.replace(f0, name=f"{first}~{'up' if up else 'dn'}",
                                       point_i=cur if not up else pt, point_j=pt if not up else cur,
                                       loads=[], releases_i=(False,) * 6, releases_j=(False,) * 6,
                                       group=None, pier=None, spandrel=None, nonlinear=None)
            if f0.point_i == cur or f0.point_j == cur:  # a single object: keep its own releases at the slab
                stub = dataclasses.replace(stub, releases_i=f0.releases_i, releases_j=f0.releases_j,
                                           end_offset_i=f0.end_offset_i, end_offset_j=f0.end_offset_j,
                                           point_i=f0.point_i, point_j=f0.point_j)
            new_frames.append(stub)
            far_pts.add(cur)
            n_cols += 1
    # walls: the wall panels touching the level, their far points fixed
    n_walls = 0
    for a in m.areas.values():
        if a.type != "Wall" or a.name in keep_areas:
            continue
        if any(p in level_pts or (p in m.points and at_level(p)) for p in a.points) and \
                not all(at_level(p) for p in a.points):
            if not any(p in level_pts for p in a.points):
                continue
            keep_areas.add(a.name)
            far_pts |= {p for p in a.points if not at_level(p)}
            level_pts |= {p for p in a.points if at_level(p)}
            n_walls += 1
    # ---- remove everything else
    for n in list(m.frames):
        if n not in keep_frames:
            del m.frames[n]
    for f in new_frames:
        m.frames[f.name] = f
    for n in list(m.areas):
        if n not in keep_areas:
            del m.areas[n]
    used = set(level_pts) | far_pts
    for f in m.frames.values():
        used |= {f.point_i, f.point_j}
    for a in m.areas.values():
        used |= set(a.points)
    for n in list(m.points):
        if n not in used:
            del m.points[n]
    for p in far_pts:
        P = m.points[p]
        P.restraints = FIXED if columns == "fixed" else PINNED
        P.loads = []
        P.ground_displacements = []
    for n in list(m.line_springs):
        ls = m.line_springs[n]
        if ls.point_i not in m.points or ls.point_j not in m.points:
            del m.line_springs[n]
    if m.tendons:
        notes.append(f"{len(m.tendons)} tendon(s) left out (tendons act on frame objects)")
        m.tendons.clear()
    if m.links:
        notes.append(f"{len(m.links)} link(s) left out")
        m.links.clear()
    for g in m.groups.values():
        g.points = [p for p in g.points if p in m.points]
        g.frames = [f for f in g.frames if f in m.frames]
        g.areas = [a for a in g.areas if a in m.areas]
    # ---- the slabs: plates; no diaphragm (the floor carries its own loads)
    changed = set()
    for a in m.areas.values():
        a.diaphragm = None
        if a.name in keep_areas and _is_slab_floor(m, a):
            sec = m.shell_sections[a.section]
            if sec.model_type == "Membrane":
                sec.model_type = "ShellThick"
                changed.add(sec.name)
    if changed:
        notes.append("membrane slab section(s) " + ", ".join(sorted(changed)) + " analysed as plates (ShellThick)")
    if beams == "rigid":
        from ..model.properties import FrameModifiers
        for n in keep_frames:
            f = m.frames[n]
            base = f.modifiers or FrameModifiers()
            f.modifiers = dataclasses.replace(base, A=base.A * RIGID_FACTOR, I22=base.I22 * RIGID_FACTOR,
                                              I33=base.I33 * RIGID_FACTOR)
        notes.append(f"beams: rigid supports (A, I22, I33 x {RIGID_FACTOR:g}; torsion as the beam)")
    # ---- loads: gravity only
    gravity = {n for n, p in m.load_patterns.items() if p.type not in LATERAL_PATTERN_TYPES}
    dropped = sorted(set(m.load_patterns) - gravity)
    for n in dropped:
        del m.load_patterns[n]
    for ms in m.mass_sources.values():
        ms.load_patterns = {k: v for k, v in ms.load_patterns.items() if k in gravity}
    for P in m.points.values():
        P.loads = [l for l in P.loads if l.pattern in gravity]
        P.ground_displacements = [g for g in P.ground_displacements if g.pattern in gravity]
    for f in m.frames.values():
        f.loads = [l for l in f.loads if l.pattern in gravity]
    for a in m.areas.values():
        a.loads = [l for l in a.loads if l.pattern in gravity]
    n_panels = 0
    for n in list(m.panel_loads):
        pl = m.panel_loads[n]
        on_level = (pl.area in keep_areas) if pl.area else \
            (bool(pl.points) and all(abs(p[2] - z) <= tol for p in pl.points))
        if not on_level:
            del m.panel_loads[n]
            continue
        pl.loads = [c for c in pl.loads if c.pattern in gravity]
        if pl.distribution != "shells":
            n_panels += 1
        pl.distribution, pl.receivers = "shells", None
    if n_panels:
        notes.append(f"{n_panels} load panel(s) carried by the slab (in the building model they load the beams)")
    # cases: linear static gravity cases; P-Delta / nonlinear gravity cases as linear ones
    kept_cases: Dict[str, object] = {}
    for n, c in m.load_cases.items():
        loads = getattr(c, "loads", None)
        if not isinstance(loads, dict) or not loads:
            continue
        if any(p not in gravity for p, f in loads.items() if f):
            continue
        acc = {d for d, f in (getattr(c, "acceleration_loads", None) or {}).items() if f}
        if acc & {"UX", "UY"}:
            continue
        if isinstance(c, LinearStaticCase):
            if c.stiffness_from:
                notes.append(f"case {n}: P-Delta stiffness of {c.stiffness_from} not used")
            kept_cases[n] = dataclasses.replace(c, stiffness_from=None, stiffness="gravity", cd=None)
        elif isinstance(c, PDeltaCase) or getattr(c, "type", "") == "NonlinearStatic":
            kept_cases[n] = LinearStaticCase(name=n, loads=dict(loads), stiffness="gravity")
            notes.append(f"case {n}: analysed as a linear static case")
    m.load_cases = kept_cases
    if not kept_cases:
        raise ModelError("NO_GRAVITY_CASE", "No linear static gravity load case to analyse the floor for")
    combos = dict(m.load_combinations)
    kept_combos: Dict[str, LoadCombination] = {}
    changed_any = True
    while changed_any:
        changed_any = False
        for n, c in combos.items():
            if n in kept_combos:
                continue
            if c.items and all(it.name in kept_cases or it.name in kept_combos for it in c.items):
                kept_combos[n] = c
                changed_any = True
    left = sorted(set(combos) - set(kept_combos))
    m.load_combinations = {n: combos[n] for n in combos if n in kept_combos}
    if left:
        notes.append("combination(s) with lateral cases left out: " + ", ".join(left[:8]) + (" ..." if len(left) > 8 else ""))
    if dropped:
        notes.append("lateral load pattern(s) left out: " + ", ".join(dropped))
    counts = {"slabs": sum(1 for a in m.areas.values() if _is_slab_floor(m, a)), "beams": len(keep_frames),
              "column stubs": n_cols, "wall panels": n_walls}
    notes.insert(0, f"{counts['slabs']} slab(s), {counts['beams']} beam(s), {n_cols} column stub(s), "
                    f"{n_walls} wall panel(s); stubs {columns} at the far end, beams {beams}")
    return SlabModelResult(model=m, story=story, notes=notes, counts=counts)
