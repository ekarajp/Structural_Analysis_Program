"""Reshape (Edit > Reshape, right-click > Reshape): drag the joints of the selection
the CAD way (AutoCAD grips / STRETCH, Revit shape handles).  The Select tool never
drags anything; only this command moves joints by dragging.

What a grip drag moves depends on what was selected before the command:

=====================  =============================================================
selection              a grip drag
=====================  =============================================================
joints only            the dragged joint only; every member on it follows (several
                       joints moved together, STRETCH: Move does that)
frames / shells only   the dragged joint of the selected objects only: an unselected
                       member on that joint keeps its own joint where it was (shown
                       dashed orange while dragging); the middle grip of a floor edge
                       moves that edge square to itself (a rectangle stays one)
joints and members     the dragged joint only; of the members on it only the selected
                       ones follow (a selected joint with no selected member on it
                       takes all its members along)
=====================  =============================================================

The drag stays in the window's plane (plan: X / Y only, elevation: in the elevation,
3D: at the joint's level); Shift while dragging keeps it along X / Y (the window's
axes) or along the dragged member.  A typed distance + Enter works as in Move.  A
joint dropped on another joint becomes that joint; dropped on a member it is a joint
of that member (the analysis connects it).  Joint loads go with their joint, frame
loads keep their place along the member (relative), drawn load panels stay where
they are.  One edit = one undo step.  Reshape changes the shape one joint at a time; moving the
selection as a whole is the Move command.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..model import Model, ModelError
from .commands import _restory
from .document import Document
from .modify import ModifyTool, _merge_points
from .tools import TOOLS, Preview, next_step

XYZ = Tuple[float, float, float]
GripKey = Tuple  # ("joint", name) | ("edge", area, k)  (edge k: corner k to corner k+1)


@dataclass
class ReshapeSet:
    """What the selection lets a grip drag move."""

    mode: str  # joints | elements | mixed
    joints: List[str] = field(default_factory=list)  # selected joints (joints / mixed)
    frames: Set[str] = field(default_factory=set)  # selected members (follow their joints)
    areas: Set[str] = field(default_factory=set)

    @property
    def follow_all(self) -> bool:
        return self.mode == "joints"


def reshape_set(model: Model, frames: Sequence[str], areas: Sequence[str], points: Sequence[str]) -> ReshapeSet:
    fr = {n for n in frames if n in model.frames}
    ar = {n for n in areas if n in model.areas}
    pt = [n for n in points if n in model.points]
    if pt and not (fr or ar):
        return ReshapeSet("joints", pt)
    if pt:
        return ReshapeSet("mixed", pt, fr, ar)
    return ReshapeSet("elements", [], fr, ar)


def _users(model: Model, n: str):
    fr = [f.name for f in model.frames.values() if n in (f.point_i, f.point_j)]
    ar = [a.name for a in model.areas.values() if n in a.points]
    return fr, ar


def element_joints(model: Model, rs: ReshapeSet) -> List[str]:
    names: List[str] = []
    for n in sorted(rs.frames):
        f = model.frames[n]
        names += [f.point_i, f.point_j]
    for n in sorted(rs.areas):
        names += list(model.areas[n].points)
    return list(dict.fromkeys(names))


def grips(model: Model, rs: ReshapeSet) -> List[Tuple[GripKey, XYZ]]:
    """The handles: joints (selected joints, or the corners / ends of the selected
    objects) and the middle of every edge of a selected shell."""
    out: List[Tuple[GripKey, XYZ]] = []
    names = rs.joints if rs.mode != "elements" else element_joints(model, rs)
    out += [(("joint", n), model.points[n].xyz) for n in names if n in model.points]
    if rs.mode == "elements":
        for n in sorted(rs.areas):
            a = model.areas[n]
            k_n = len(a.points)
            for k in range(k_n):
                p, q = model.points[a.points[k]].xyz, model.points[a.points[(k + 1) % k_n]].xyz
                if math.dist(p, q) > 1e-9:
                    out.append((("edge", n, k), tuple((np.add(p, q) / 2).tolist())))
    return out


def _area_normal(model: Model, a) -> Optional[np.ndarray]:
    P = np.array([model.points[q].xyz for q in a.points], float)
    c = P.mean(axis=0)
    n = np.zeros(3)
    for i in range(len(P)):
        n += np.cross(P[i] - c, P[(i + 1) % len(P)] - c)
    L = float(np.linalg.norm(n))
    return n / L if L > 1e-12 else None


def edge_normal(model: Model, area: str, k: int) -> Optional[np.ndarray]:
    """Unit direction square to edge k of the shell, in the shell's plane."""
    a = model.areas[area]
    nrm = _area_normal(model, a)
    if nrm is None:
        return None
    p = np.array(model.points[a.points[k]].xyz, float)
    q = np.array(model.points[a.points[(k + 1) % len(a.points)]].xyz, float)
    e = np.cross(nrm, q - p)
    L = float(np.linalg.norm(e))
    return e / L if L > 1e-12 else None


def grip_joints(model: Model, rs: ReshapeSet, key: GripKey) -> List[str]:
    """The joints a drag of the grip moves: that joint only (an edge grip: its two)."""
    if key[0] == "edge":
        a = model.areas[key[1]]
        return [a.points[key[2]], a.points[(key[2] + 1) % len(a.points)]]
    return [key[1]]


def constrain(model: Model, rs: ReshapeSet, key: GripKey, delta, view: str, lock: bool = False) -> np.ndarray:
    """The move vector of a drag: in the window's plane (3D: level), a floor edge
    square to itself, Shift (``lock``) along X / Y (the window's axes) or a member."""
    d = np.asarray(delta, float).copy()
    nrm = plane_normal_of(view)
    d = d - nrm * float(np.dot(d, nrm))
    if key[0] == "edge":
        e = edge_normal(model, key[1], key[2])
        if e is None:
            return np.zeros(3)
        d = e * float(np.dot(d, e))
        d = d - nrm * float(np.dot(d, nrm))
        return d
    if lock and float(np.linalg.norm(d)) > 0:
        dirs = [np.array(v, float) for v in _view_axes(view)]
        if key[0] == "joint":
            n = key[1]
            for f in model.frames.values():
                if n in (f.point_i, f.point_j) and (rs.follow_all or f.name in rs.frames):
                    other = f.point_j if f.point_i == n else f.point_i
                    v = np.subtract(model.points[n].xyz, model.points[other].xyz)
                    v = v - nrm * float(np.dot(v, nrm))
                    L = float(np.linalg.norm(v))
                    if L > 1e-9:
                        dirs.append(v / L)
        best = max(dirs, key=lambda u: abs(float(np.dot(d, u))))
        d = best * float(np.dot(d, best))
    return d


def plane_normal_of(view: str) -> np.ndarray:
    """The direction a drag in this window never goes (3D: vertical - the joint stays
    at its level)."""
    return {"plan": np.array((0.0, 0.0, 1.0)), "xz": np.array((0.0, 1.0, 0.0)),
            "yz": np.array((1.0, 0.0, 0.0))}.get(view, np.array((0.0, 0.0, 1.0)))


def _view_axes(view: str):
    return {"xz": ((1, 0, 0), (0, 0, 1)), "yz": ((0, 1, 0), (0, 0, 1))}.get(view, ((1, 0, 0), (0, 1, 0)))


def detached(model: Model, rs: ReshapeSet, moved: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Unselected members on the moved joints that keep their joint (stay)."""
    if rs.follow_all:
        return [], []
    fr: List[str] = []
    ar: List[str] = []
    for n in moved:
        ufr, uar = _users(model, n)
        if rs.mode == "mixed" and not (set(ufr) & rs.frames or set(uar) & rs.areas):
            continue  # a selected joint no selected member is on: all its members follow
        fr += [x for x in ufr if x not in rs.frames]
        ar += [x for x in uar if x not in rs.areas]
    return list(dict.fromkeys(fr)), list(dict.fromkeys(ar))


def _length(model: Model, f) -> float:
    return math.dist(model.points[f.point_i].xyz, model.points[f.point_j].xyz)


def _inside_poly(pt2, poly2, eps) -> bool:
    from ..model.checks import _inside_2d
    return _inside_2d(pt2, poly2, eps) >= 0


def openings_outside(model: Model, names: Optional[Set[str]] = None) -> List[str]:
    """Openings not inside any floor / wall in their plane (a reshape moved them off)."""
    tol = max(model.options.merge_tolerance, 1e-6)
    out = []
    solids = [a for a in model.areas.values() if not a.is_opening and len(a.points) >= 3]
    for o in model.areas.values():
        if not o.is_opening or (names is not None and o.name not in names) or len(o.points) < 3:
            continue
        nrm = _area_normal(model, o)
        if nrm is None:
            continue
        O = np.array([model.points[q].xyz for q in o.points], float)
        u = np.cross(nrm, (0, 0, 1.0)) if abs(nrm[2]) < 0.9 else np.array((1.0, 0, 0))
        u = u / np.linalg.norm(u)
        v = np.cross(nrm, u)
        ok = False
        for a in solids:
            n2 = _area_normal(model, a)
            if n2 is None or abs(float(np.dot(n2, nrm))) < 1 - 1e-6:
                continue
            A = np.array([model.points[q].xyz for q in a.points], float)
            if abs(float(np.dot(A[0] - O[0], nrm))) > tol * 10:
                continue
            poly = [(float(np.dot(p, u)), float(np.dot(p, v))) for p in A]
            if all(_inside_poly((float(np.dot(p, u)), float(np.dot(p, v))), poly, tol * 10) for p in O):
                ok = True
                break
        if not ok:
            out.append(o.name)
    return out


def _on_member(model: Model, n: str, tol: float) -> Optional[str]:
    """A member the joint lies on (between its ends), if any."""
    p = np.array(model.points[n].xyz, float)
    for f in model.frames.values():
        if n in (f.point_i, f.point_j):
            continue
        a = np.array(model.points[f.point_i].xyz, float)
        b = np.array(model.points[f.point_j].xyz, float)
        d = b - a
        L2 = float(np.dot(d, d))
        if L2 <= tol * tol:
            continue
        t = float(np.dot(p - a, d)) / L2
        if tol / math.sqrt(L2) < t < 1 - tol / math.sqrt(L2) and float(np.linalg.norm(a + t * d - p)) <= tol:
            return f.name
    return None


def reshape(doc: Document, rs: ReshapeSet, moves: Dict[str, XYZ]) -> dict:
    """Put the joints of ``moves`` at their new places (one undo step).  Unselected
    members on them keep a joint of their own where they were (unless every member
    follows: a joints-only selection).  Returns what happened (for the status bar)."""
    moves = {n: tuple(float(c) for c in xyz) for n, xyz in moves.items() if n in doc.model.points}
    if not moves:
        return {"moved": 0}
    info: dict = {"moved": len(moves), "stayed": [], "merged": {}, "on": {}, "cleared": [], "stories": [],
                  "openings": []}

    def mutate(model: Model):
        tol = max(model.options.merge_tolerance, 1e-6)
        names = list(moves)
        lengths = {f.name: _length(model, f) for f in model.frames.values()}
        story0 = {n: model.points[n].story for n in names}
        # 1. split: unselected members keep the joint where it is
        fr_stay, ar_stay = detached(model, rs, names)
        fs, as_ = set(fr_stay), set(ar_stay)
        for n in names:
            ufr, uar = _users(model, n)
            stay_f = [x for x in ufr if x in fs]
            stay_a = [x for x in uar if x in as_]
            if not (stay_f or stay_a):
                continue
            p = model.points[n]
            q = model.add_point(p.x, p.y, p.z, merge=False)
            q.story = p.story
            for x in stay_f:
                f = model.frames[x]
                f.point_i = q.name if f.point_i == n else f.point_i
                f.point_j = q.name if f.point_j == n else f.point_j
            for x in stay_a:
                a = model.areas[x]
                a.points = [q.name if c == n else c for c in a.points]
        info["stayed"] = sorted(fs | as_)
        # 2. the joints to their new places
        for n, xyz in moves.items():
            p = model.points[n]
            p.x, p.y, p.z = xyz
        # 3. dropped on a joint: that joint (merged); on a member: a joint of it
        had_f, had_a = set(model.frames), set(model.areas)
        _merge_points(model, names, tol)
        lost = sorted(had_f - set(model.frames)) + sorted(had_a - set(model.areas))
        if lost:  # a member with both ends / a shell with two corners on one joint
            raise ModelError("ZERO_LENGTH", f"{', '.join(lost[:6])} would have no length / area (two of its joints on "
                                            f"one place) - nothing was changed")
        for n in names:
            if n not in model.points:
                q = model.find_point(*moves[n])
                info["merged"][n] = q.name if q is not None else "?"
        alive = [n for n in names if n in model.points]
        for n in alive:
            on = _on_member(model, n, tol)
            if on:
                info["on"][n] = on
        # 4. what the new shape must not be
        for f in model.frames.values():
            if f.point_i == f.point_j or _length(model, f) <= tol:
                raise ModelError("ZERO_LENGTH", f"Frame {f.name} would have no length (its two ends on one joint) - "
                                                f"nothing was changed")
        for a in model.areas.values():
            if len(set(a.points)) < len(a.points) or len(a.points) < 3:
                raise ModelError("BAD_AREA", f"Shell {a.name} would have two corners on one joint - nothing was "
                                             f"changed")
        moved_now = {info["merged"].get(n, n) for n in names}
        for f in model.frames.values():
            if f.point_i not in moved_now and f.point_j not in moved_now:
                continue
            L0, L1 = lengths.get(f.name), _length(model, f)
            # loads given as distances keep their place along the member (relative)
            if L0 and L1 and abs(L1 - L0) > 1e-12:
                for ld in f.loads:
                    if getattr(ld, "absolute", False):
                        ld.rel_a = ld.rel_a * L1 / L0
                        ld.rel_b = ld.rel_b * L1 / L0
            if getattr(f, "real_axis", None):
                # a beam moved by Move Beams to Real Axis: reshaped, the move is no longer known
                f.real_axis = None
                f.real_axis_offset_i = f.real_axis_offset_j = (0.0, 0.0, 0.0)
                info["cleared"].append(f.name)
        _restory(model)
        for n in alive:
            s0, s1 = story0.get(n), model.points[n].story
            if s0 != s1:
                info["stories"].append((n, s0, s1))
        touched = {a.name for a in model.areas.values() if set(a.points) & moved_now}
        info["openings"] = openings_outside(model, {n for n in touched if model.areas[n].is_opening} |
                                            ({o.name for o in model.areas.values() if o.is_opening}
                                             if any(not model.areas[n].is_opening for n in touched) else set()))

    doc.edit(mutate)
    _notes(doc, info)
    return info


def _notes(doc: Document, info: dict):
    m = doc.model
    parts = [f"Reshape: {info['moved']} joint(s) moved"]
    if info["merged"]:
        parts.append("joined to " + ", ".join(sorted(set(info["merged"].values()))))
    if info["on"]:
        parts.append("on member " + ", ".join(sorted(set(info["on"].values()))) + " (connected there)")
    if info["stayed"]:
        st = info["stayed"]
        parts.append("stayed at the old joint: " + ", ".join(st[:8]) + (f" +{len(st) - 8}" if len(st) > 8 else ""))
    doc.note("; ".join(parts))
    if info["cleared"]:
        doc.note(f"{', '.join(info['cleared'])}: no longer remembered as moved by Move Beams to Real Axis "
                 f"(reshaped)")
    for n, s0, s1 in info["stories"]:
        doc.note(f"Joint {n} is now on story {s1 or '-'} (was {s0 or '-'})")
    if info["openings"]:
        doc.note("Opening " + ", ".join(info["openings"]) + " is no longer inside a floor / wall - it cuts "
                 "nothing. Undo (Ctrl+Z) or move it back.", "warning")
    del m


# ------------------------------------------------------------------ the tool
class ReshapeTool(ModifyTool):
    """Drag a grip (press, drag, release) or click a grip, then the new place (or
    type a distance + Enter).  Stays on for more drags; Esc = done."""

    name = "reshape"
    title = "Reshape"
    needs = "joints, frames or shells to reshape"
    hot: Optional[GripKey] = None
    shift_lock = False

    def activate(self, state) -> None:
        super().activate(state)
        self.hot = None

    def done_selecting(self, state) -> bool:
        ok = super().done_selecting(state)
        if ok:
            state.selection_changed.emit()  # the windows draw the grips
        return ok

    def rset(self) -> ReshapeSet:
        st = self.state
        return reshape_set(self.model, list(st.frames), list(st.areas), list(st.points))

    def grips(self) -> List[Tuple[GripKey, XYZ]]:
        if self.selecting or not hasattr(self, "state"):
            return []
        try:
            return grips(self.model, self.rset())
        except Exception:
            return []

    def grip_xyz(self, key: GripKey) -> Optional[XYZ]:
        return next((xyz for k, xyz in self.grips() if k == key), None)

    def pick_grip(self, key: GripKey) -> bool:
        xyz = self.grip_xyz(key)
        if xyz is None:
            return False
        self.hot = key
        self.points = [tuple(float(c) for c in xyz)]  # base: tracking / typed distances from here
        self._raw_last = None
        self.state.update_hint()
        return True

    def cancel(self):
        self.hot = None
        super().cancel()

    def step_hint(self) -> str:
        if self.hot is None:
            return next_step(self.title, "drag a grip (square) to its new place - or click it, then the new place",
                             "Shift while dragging = along X / Y / the member", "Esc = done")
        return next_step(self.title, "the new place (release / click), or type a distance + Enter",
                         "Shift = along X / Y / the member", "Esc = cancel this grip")

    # the move a cursor position means
    def _delta(self, cursor) -> np.ndarray:
        c = self._p3(cursor)
        rs = self.rset()
        return constrain(self.model, rs, self.hot, np.subtract(c, self.points[0]), self.plane.kind, self.shift_lock)

    def moves(self, delta) -> Dict[str, XYZ]:
        m = self.model
        rs = self.rset()
        return {n: tuple((np.array(m.points[n].xyz, float) + delta).tolist()) for n in grip_joints(m, rs, self.hot)}

    def click(self, pt, snap=None) -> bool:
        if self.selecting:
            return False
        if self.hot is None:
            self.state.status.emit("Reshape: drag a grip (the squares on the selection) - or click one first")
            return False
        d = self._delta(pt)
        if float(np.linalg.norm(d)) < 1e-12:
            if self.hot[0] == "edge":
                self.state.status.emit("Reshape: a floor edge moves square to itself (in this window it cannot: "
                                       "use a window the floor lies in)")
            self.cancel()
            return False
        try:
            reshape(self.doc, self.rset(), self.moves(d))
        finally:
            self.cancel()
        self.state.update_hint()
        return True

    def preview(self, cursor) -> Preview:
        if self.hot is None or self.selecting or cursor is None:
            return Preview(marker=None if cursor is None else self._p3(cursor))
        m = self.model
        rs = self.rset()
        d = self._delta(cursor)
        mv = self.moves(d)
        fr_stay, ar_stay = detached(m, rs, list(mv))
        stay = set(fr_stay) | set(ar_stay)

        def P(n):
            return mv.get(n, m.points[n].xyz)
        lines, dash, dims = [], [], []
        for f in m.frames.values():
            if f.name in stay or not (f.point_i in mv or f.point_j in mv):
                continue
            a, b = P(f.point_i), P(f.point_j)
            lines.append((a, b))
            if len(dims) < 4:
                dims.append(f"{f.name} {_length(m, f):.3f}>{math.dist(a, b):.3f}")
        for a in m.areas.values():
            if a.name in stay or not (set(a.points) & set(mv)):
                continue
            q = [P(c) for c in a.points]
            lines += [(q[k], q[(k + 1) % len(q)]) for k in range(len(q))]
        for n in fr_stay:
            f = m.frames[n]
            dash.append((m.points[f.point_i].xyz, m.points[f.point_j].xyz))
        for n in ar_stay:
            a = m.areas[n]
            q = [m.points[c].xyz for c in a.points]
            dash += [(q[k], q[(k + 1) % len(q)]) for k in range(len(q))]
        base = self.points[0]
        to = tuple((np.array(base) + d).tolist())
        text = f"Reshape  {self._measure(base, to)}"
        if dims:
            text += "   " + "  ".join(dims)
        if stay:
            text += f"   ({len(stay)} unselected stay: dashed)"
        pv = Preview(lines=lines, marker=to, text=text, guides=[(base, to)], detach=dash)
        return pv


TOOLS.update({"reshape": ReshapeTool})
