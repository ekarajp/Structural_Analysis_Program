"""Analysis diagnostics: what is wrong, with which object, and where.

Every problem found while meshing, assembling or solving becomes an
:class:`Issue` naming the object (joint / frame / shell), its story and grid
position and its coordinates, with a hint on how to fix it - so the UI can
list them and jump to the object (ETABS "analysis warnings" + locate).

* :func:`pre_checks` - after meshing, before solving: loads that reach nothing
  (a loaded joint or a null line connected to no member, floor or support) and
  parts of the structure that are not connected to any support;
* :func:`explain` - an exception from meshing / element set-up / the solver
  turned into issues (zero-length or warped elements, an unstable structure
  with the joints and directions that can move freely);
* :func:`from_warnings` - the analysis model's text warnings as issues.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

DOF_NAMES = ("UX", "UY", "UZ", "RX", "RY", "RZ")


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    message: str
    attr: Optional[str] = None  # "points" | "frames" | "areas" (for locate)
    name: Optional[str] = None
    xyz: Optional[Tuple[float, float, float]] = None  # model units
    where: str = ""  # story / grid text
    hint: str = ""

    def text(self) -> str:
        loc = f" [{self.where}]" if self.where else ""
        hint = f" -> {self.hint}" if self.hint else ""
        return f"{self.severity.upper()}: {self.message}{loc}{hint}"


from .model.base import ModelError  # noqa: E402


class AnalysisError(ModelError):
    """The analysis cannot run; ``issues`` say what and where.  ``code`` is the
    code of the underlying error (e.g. SINGULAR_STIFFNESS) when there is one."""

    def __init__(self, issues: Sequence[Issue], code: str = "ANALYSIS_ERROR"):
        self.issues = list(issues)
        errs = [i for i in self.issues if i.severity == "error"] or self.issues
        msg = "; ".join(i.message + (f" [{i.where}]" if i.where else "") for i in errs[:3])
        msg += f" (+{len(errs) - 3} more)" if len(errs) > 3 else ""
        super().__init__(code, msg)


# ------------------------------------------------------------------ locating
def _where(model, xyz) -> str:
    from .app.locate import grid_ref
    st = model.story_at(float(xyz[2]), model.options.merge_tolerance, xy=(float(xyz[0]), float(xyz[1]))) \
        if model.stories else None
    u = model.units.label("length")
    s = f"{st.name}, " if st else ""
    return f"{s}grid {grid_ref(model, xyz)}, ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) {u}"


def node_object(model, am, nid: int, node_shells=None, node_frames=None) -> Tuple[Optional[str], Optional[str]]:
    """The model object a mesh node belongs to: its joint, else a shell, else a frame."""
    nd = am.nodes[nid]
    if nd.point is not None and nd.point in model.points:
        return "points", nd.point
    if node_shells is not None and nid in node_shells:
        return "areas", sorted(node_shells[nid])[0]
    if node_frames is not None and nid in node_frames:
        return "frames", sorted(node_frames[nid])[0]
    for s in am.shells:
        if nid in s.nodes:
            return "areas", s.object
    for f in am.frames:
        if nid in (f.node_i, f.node_j):
            return "frames", f.object
    return None, None


def _node_xyz(model, am, nid) -> Tuple[float, float, float]:
    Lf = model.units.factor("length")
    return tuple(float(c) / Lf for c in am.nodes[nid].xyz)


def _issue_at_node(model, am, nid, severity, message, hint="", maps=None) -> Issue:
    ns, nf = maps if maps else (None, None)
    attr, name = node_object(model, am, nid, ns, nf)
    xyz = _node_xyz(model, am, nid)
    return Issue(severity, message, attr, name, xyz, _where(model, xyz), hint)


def _maps(am):
    node_shells: Dict[int, Set[str]] = {}
    node_frames: Dict[int, Set[str]] = {}
    for s in am.shells:
        for n in s.nodes:
            node_shells.setdefault(n, set()).add(s.object)
    for f in am.frames:
        for n in (f.node_i, f.node_j):
            node_frames.setdefault(n, set()).add(f.object)
    return node_shells, node_frames


# ------------------------------------------------------------- before solving
def pre_checks(model, am) -> List[Issue]:
    issues: List[Issue] = []
    ns, nf = _maps(am)
    connected = set(ns) | set(nf)
    for link in list(getattr(am, "links", []) or []) + list(getattr(am, "rigid_links", []) or []):
        for n in (getattr(link, "node_i", None), getattr(link, "node_j", None)):
            if n is not None:
                connected.add(n)
    tied = {t.node for t in getattr(am, "ties", [])}
    for body in getattr(am, "bodies", []) or []:
        tied.update(body.nodes)
    supported = {nd.id for nd in am.nodes if np.any(nd.restraints) or np.any(nd.springs)}
    in_diaphragm = {n for c in am.constraints for n in c.nodes}
    # 1) loads that reach nothing
    lost: Dict[int, Dict[str, float]] = {}
    for pattern, loads in am.nodal_loads.items():
        for nid, vec in loads.items():
            if nid in connected or nid in tied or nid in supported:
                continue
            f = float(np.linalg.norm(vec[:3]))
            if f <= 0 and not np.any(vec[3:]):
                continue
            if nid in in_diaphragm and abs(vec[2]) <= 1e-12 * max(f, 1.0) and not np.any(vec[3:5]):
                continue  # a lateral load on a diaphragm point (ETABS story load): it is carried
            lost.setdefault(nid, {})[pattern] = f
    Ff = model.units.factor("force")
    uF = model.units.force
    null_lines = _null_line_nodes(model, am)
    for nid, pats in lost.items():
        loads = ", ".join(f"{p} {v / Ff:.3g} {uF}" for p, v in pats.items())
        src = null_lines.get(nid)
        if src:
            iss = _issue_at_node(model, am, nid, "warning",
                                 f"Part of the line load on null line {src} ({loads}) reaches nothing here - "
                                 f"the line runs outside the floors / members, that part is not applied",
                                 "keep line loads on a floor, or draw a member under them", (ns, nf))
            iss.attr, iss.name = "frames", src
        else:
            nd = am.nodes[nid]
            near = _nearest_area(model, _node_xyz(model, am, nid))
            more = f"; it is {near[1]:.3f} {model.units.label('length')} from {near[0]}" if near else ""
            iss = _issue_at_node(model, am, nid, "warning",
                                 f"Joint {nd.point or nid} carries a load ({loads}) but is not connected to any "
                                 f"member, floor or support - the load is not applied{more}",
                                 "move the joint onto the floor / member, or connect it", (ns, nf))
        issues.append(iss)
    # 2) parts of the structure that rest on nothing
    issues += _floating_parts(model, am, ns, nf, supported)
    return issues


def _null_line_nodes(model, am) -> Dict[int, str]:
    """Mesh nodes lying on a loaded null line -> the null line's name."""
    out: Dict[int, str] = {}
    lines = [f for f in model.frames.values() if f.type == "Null" and f.loads]
    if not lines:
        return out
    Lf = model.units.factor("length")
    tol = model.options.merge_tolerance * Lf
    X = np.array([n.xyz for n in am.nodes], float)
    for f in lines:
        a = np.array(model.points[f.point_i].xyz) * Lf
        b = np.array(model.points[f.point_j].xyz) * Lf
        d = b - a
        L2 = float(d @ d)
        if L2 <= 0:
            continue
        t = ((X - a) @ d) / L2
        off = np.linalg.norm(X - (a + np.outer(t, d)), axis=1)
        for nid in np.nonzero((off <= tol) & (t >= -1e-9) & (t <= 1 + 1e-9))[0]:
            out.setdefault(int(nid), f.name)
    return out


def _nearest_area(model, xyz) -> Optional[Tuple[str, float]]:
    best = None
    for a in model.areas.values():
        if a.is_opening or any(p not in model.points for p in a.points):
            continue
        P = np.array([model.points[p].xyz for p in a.points], float)
        if abs(P[:, 2].mean() - xyz[2]) > 1e-6 and np.ptp(P[:, 2]) <= 1e-6:
            continue
        d = min(np.linalg.norm(np.cross(P[(i + 1) % len(P)] - P[i], P[i] - np.asarray(xyz))) /
                max(np.linalg.norm(P[(i + 1) % len(P)] - P[i]), 1e-12) for i in range(len(P)))
        if best is None or d < best[1]:
            best = (a.name, float(d))
    return best


def _floating_parts(model, am, ns, nf, supported) -> List[Issue]:
    """Connected groups of elements (shared nodes, ties) with no support, spring
    or ground link: they would fly away (a column standing on nothing, a slab
    not attached to anything)."""
    n = len(am.nodes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for f in am.frames:
        union(f.node_i, f.node_j)
    for s in am.shells:
        for k in s.nodes[1:]:
            union(s.nodes[0], k)
    for t in getattr(am, "ties", []):
        for m_, _ in t.masters:
            union(t.node, m_)
    for link in list(getattr(am, "links", []) or []) + list(getattr(am, "rigid_links", []) or []):
        a, b = getattr(link, "node_i", None), getattr(link, "node_j", None)
        if a is not None and b is not None:
            union(a, b)
    for body in getattr(am, "bodies", []) or []:
        for k in body.nodes:
            union(body.master, k)
    has_support: Set[int] = set()
    for nid in supported:
        has_support.add(find(nid))
    for link in getattr(am, "links", []) or []:  # grounded (one-joint) links
        a, b = getattr(link, "node_i", None), getattr(link, "node_j", None)
        if a is not None and b is None:
            has_support.add(find(a))
    groups: Dict[int, List[int]] = {}
    for nid in set(ns) | set(nf):
        groups.setdefault(find(nid), []).append(nid)
    issues = []
    for root, nodes in groups.items():
        if root in has_support:
            continue
        objs = sorted({o for nid in nodes for o in ns.get(nid, ())} | {o for nid in nodes for o in nf.get(nid, ())})
        X = np.array([am.nodes[i].xyz for i in nodes]) / model.units.factor("length")
        lowest = nodes[int(np.argmin(X[:, 2]))]
        listed = ", ".join(objs[:6]) + (f" (+{len(objs) - 6} more)" if len(objs) > 6 else "")
        iss = _issue_at_node(model, am, lowest, "error",
                             f"{len(objs)} object(s) are not connected to any support: {listed} - "
                             f"they rest on nothing (the structure is unstable)",
                             "connect them to the rest of the structure (a shared joint, the floor they stand "
                             "on) or add a support", (ns, nf))
        if objs:
            first = objs[0]
            iss.attr = "frames" if first in model.frames else "areas" if first in model.areas else iss.attr
            iss.name = first
        issues.append(iss)
    return issues


# --------------------------------------------------------------- exceptions
_EL = re.compile(r"(shell|frame) element (\d+)")


def explain(model, am, exc: Exception, asm=None, fac=None) -> List[Issue]:
    """Issues for an exception raised while setting up or solving."""
    msg = getattr(exc, "message", None) or str(exc)
    issues: List[Issue] = []
    m = _EL.search(msg)
    if m and am is not None:
        kind, i = m.group(1), int(m.group(2))
        els = am.shells if kind == "shell" else am.frames
        if 0 <= i < len(els):
            el = els[i]
            nodes = list(el.nodes) if kind == "shell" else [el.node_i, el.node_j]
            xyz = tuple(np.mean([am.nodes[n].xyz for n in nodes], axis=0) / model.units.factor("length"))
            obj = el.object
            what = re.sub(r"(shell|frame) element \d+", f"the mesh of {obj}", msg)
            hint = {"shell": "check the object's corners (order, flatness, overlaps) or remesh it finer",
                    "frame": "check the member's joints (two joints at the same place?)"}[kind]
            issues.append(Issue("error", f"{'Shell' if kind == 'shell' else 'Frame'} {obj}: {what}",
                                "areas" if kind == "shell" else "frames", obj, xyz, _where(model, xyz), hint))
            return issues
    code = getattr(exc, "code", "")
    if code in ("SINGULAR_STIFFNESS", "UNSTABLE_STRUCTURE") and am is not None:
        issues += mechanism(model, am, asm, fac)
        if not issues:
            issues.append(Issue("error", msg, hint="check supports, releases and connections"))
        return issues
    issues.append(Issue("error", f"{code + ': ' if code else ''}{msg}"))
    return issues


def mechanism(model, am, asm=None, fac=None, limit: int = 8) -> List[Issue]:
    """Where is the structure unstable?  The joints and directions that have no
    stiffness: from the factorisation's vanishing pivots, or else from the
    lowest eigenvector of the stiffness (the mechanism shape)."""
    issues: List[Issue] = []
    if asm is None:
        try:
            from .engine.assembly import Assembler
            asm = Assembler(am)
        except Exception:
            return issues
    dofs = asm.dofs
    keep = np.where(dofs.reduced_of_full >= 0)[0]
    free = list(getattr(fac, "unstable", None) or [])
    pairs: List[Tuple[int, int]] = []
    if free:
        for j in free:
            red = int(dofs.free[j])
            pairs.append(divmod(int(keep[red]), 6))
    else:
        try:
            from scipy.sparse.linalg import eigsh
            K = asm.stiffness_reduced()
            f = dofs.free
            Kf = K[f][:, f].tocsc()
            d = np.abs(Kf.diagonal())
            s = -1e-8 * float(d.max() if d.size else 1.0)
            w, v = eigsh(Kf, k=1, sigma=s, which="LM")
            phi = np.abs(v[:, 0])
            for j in np.argsort(phi)[::-1][: 4 * limit]:
                if phi[j] < 0.2 * phi.max():
                    break
                red = int(f[j])
                pairs.append(divmod(int(keep[red]), 6))
        except Exception:
            return issues
    ns, nf = _maps(am)
    seen: Dict[int, List[str]] = {}
    for node, comp in pairs:
        seen.setdefault(node, []).append(DOF_NAMES[comp])
    for node, comps in list(seen.items())[:limit]:
        objs = sorted(ns.get(node, set()) | nf.get(node, set()))
        att = f" (joins {', '.join(objs[:4])})" if objs else ""
        label = am.nodes[node].point or f"mesh node {node}"
        issues.append(_issue_at_node(model, am, node, "error",
                                     f"Unstable: {label} can move freely in {'/'.join(dict.fromkeys(comps))}{att}",
                                     "add a support or a member there, or remove a release that makes it a hinge",
                                     (ns, nf)))
    if len(seen) > limit:
        issues.append(Issue("error", f"... and {len(seen) - limit} more unstable joint(s)"))
    return issues


# ------------------------------------------------------------------- warnings
_AT = re.compile(r"(?:at|Joint|Area|Frame|joint|area|frame) ([A-Za-z]*\d+[A-Za-z0-9_\-]*)")


def from_warnings(model, warnings: Sequence[str], skip_null_dof: bool = True) -> List[Issue]:
    """The analysis model's text warnings as issues, with the object they name."""
    out: List[Issue] = []
    null_dofs: Dict[str, List[str]] = {}
    for w in warnings:
        m = re.match(r"Null DOF (\w+) at (\S+) has no stiffness", w)
        if m:
            null_dofs.setdefault(m.group(2), []).append(m.group(1))
            continue
        attr = name = None
        for mm in _AT.finditer(w):
            cand = mm.group(1)
            for a in ("points", "frames", "areas"):
                if cand in getattr(model, a):
                    attr, name = a, cand
                    break
            if attr:
                break
        out.append(Issue("warning", w, attr, name))
    for label, comps in null_dofs.items():
        if label not in model.points:
            continue
        if skip_null_dof:  # ETABS / SAP2000 restrain these silently; keep a note
            out.append(Issue("info", f"Joint {label}: {'/'.join(comps)} has no stiffness - restrained automatically",
                             "points", label))
    return out
