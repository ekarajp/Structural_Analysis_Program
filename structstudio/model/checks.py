"""Model check (Analyze > Check Model): geometry problems the analysis would not
stop on but that change the results - members drawn twice, overlapping
members, joints that should be one, shells or frames with no size.

:func:`geometry_issues` checks the whole model; :func:`duplicates_of` checks
only the given frames / areas (fast: used after every drawing / copy / move).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .point_ops import coincident_groups


@dataclass
class GeometryIssue:
    severity: str  # error | warning | info
    code: str
    message: str
    frames: List[str] = field(default_factory=list)
    areas: List[str] = field(default_factory=list)
    points: List[str] = field(default_factory=list)


def _xyz(model, n) -> np.ndarray:
    return np.array(model.points[n].xyz, float)


def area_size(model, a) -> float:
    pts = [_xyz(model, n) for n in a.points if n in model.points]
    if len(pts) < 3:
        return 0.0
    v = np.zeros(3)
    for k in range(len(pts)):
        v += np.cross(pts[k], pts[(k + 1) % len(pts)])
    return float(np.linalg.norm(v) / 2)


def _line_key(a: np.ndarray, b: np.ndarray, tol: float):
    """A key shared by every segment on the same infinite line (rounded)."""
    d = b - a
    n = np.linalg.norm(d)
    if n <= tol:
        return None
    d = d / n
    # canonical direction: first non-zero component positive
    for c in d:
        if abs(c) > 1e-9:
            if c < 0:
                d = -d
            break
    foot = a - np.dot(a, d) * d  # closest point of the line to the origin
    q = max(tol, 1e-9) * 10
    return (tuple(np.round(d, 6)), tuple(np.round(foot / q).astype(int)))


def _frame_index(model, tol: float):
    pairs: Dict[frozenset, List[str]] = {}
    lines: Dict[tuple, List[Tuple[str, float, float]]] = {}
    for f in model.frames.values():
        if f.point_i not in model.points or f.point_j not in model.points:
            continue
        pairs.setdefault(frozenset((f.point_i, f.point_j)), []).append(f.name)
        a, b = _xyz(model, f.point_i), _xyz(model, f.point_j)
        key = _line_key(a, b, tol)
        if key is None:
            continue
        d = np.array(key[0], float)
        t0, t1 = sorted((float(np.dot(a, d)), float(np.dot(b, d))))
        lines.setdefault(key, []).append((f.name, t0, t1))
    return pairs, lines


def _overlaps(seg: List[Tuple[str, float, float]], tol: float) -> List[Tuple[str, str, float]]:
    """(f, g, overlap length) of segments on one line that overlap by more than tol."""
    out = []
    seg = sorted(seg, key=lambda s: s[1])
    active: List[Tuple[str, float, float]] = []
    for s in seg:
        active = [a for a in active if a[2] > s[1] + tol]
        for a in active:
            ov = min(a[2], s[2]) - s[1]
            if ov > tol:
                out.append((a[0], s[0], ov))
        active.append(s)
    return out


def _link_in_joint_zone(model, f, tol: float) -> bool:
    """A rigid link from a column's joint to a point inside that column's section (in
    plan, same level): the beam joined to the column inside its joint zone."""
    if f.type != "Rigid":
        return False
    from ..mesh.geometry import column_containing
    for p, q in ((f.point_i, f.point_j), (f.point_j, f.point_i)):
        # (a beam ending there whose body still overlaps the column bears on it: half its width more)
        grow = max([0.5 * float(getattr(model.frame_sections.get(g.section), "t2", 0.0) or 0.0)
                    for g in model.frames.values() if g.type == "Beam" and q in (g.point_i, g.point_j)] or [0.0])
        if column_containing(model, p, model.points[q].xyz, tol, grow=grow) is not None:
            return True
    return False


def duplicates_of(model, frames: Iterable[str] = (), areas: Iterable[str] = ()) -> List[GeometryIssue]:
    """Duplicate / overlapping members and duplicate shells involving the given ones."""
    tol = max(model.options.merge_tolerance, 1e-9)
    frames = [n for n in frames if n in model.frames]
    areas = [n for n in areas if n in model.areas]
    out: List[GeometryIssue] = []
    if frames:
        pairs, lines = _frame_index(model, tol)
        mine = set(frames)
        seen: Set[frozenset] = set()
        for n in frames:
            f = model.frames[n]
            others = [o for o in pairs.get(frozenset((f.point_i, f.point_j)), []) if o != n]
            if others:
                key = frozenset([n] + others)
                if key not in seen:
                    seen.add(key)
                    out.append(GeometryIssue("warning", "DUPLICATE_FRAME",
                                             f"{n} lies exactly on {', '.join(others)} (same two joints): a member "
                                             "drawn twice doubles its stiffness and weight", [n] + others))
        for key, seg in lines.items():
            if not any(s[0] in mine for s in seg) or len(seg) < 2:
                continue
            for a, b, ov in _overlaps(seg, tol):
                if (a in mine or b in mine) and frozenset((a, b)) not in seen:
                    fa, fb = model.frames[a], model.frames[b]
                    if {fa.point_i, fa.point_j} == {fb.point_i, fb.point_j}:
                        continue
                    if fa.type == "Rigid" and fb.type == "Rigid":
                        continue  # rigid links over each other: one rigid body, nothing counted twice
                    if _link_in_joint_zone(model, fa, tol) or _link_in_joint_zone(model, fb, tol):
                        continue  # a rigid link inside a column: the joint zone, rigid anyway
                    seen.add(frozenset((a, b)))
                    out.append(GeometryIssue("warning", "OVERLAPPING_FRAMES",
                                             f"{a} and {b} overlap along {ov:.4g} (same line): part of the member "
                                             "is there twice", [a, b]))
    if areas:
        sets: Dict[frozenset, List[str]] = {}
        for a in model.areas.values():
            sets.setdefault(frozenset(a.points), []).append(a.name)
        seen_a: Set[frozenset] = set()
        for n in areas:
            others = [o for o in sets.get(frozenset(model.areas[n].points), []) if o != n]
            key = frozenset([n] + others)
            if others and key not in seen_a:
                seen_a.add(key)
                out.append(GeometryIssue("warning", "DUPLICATE_AREA",
                                         f"{n} lies exactly on {', '.join(others)} (same corners): a slab / wall "
                                         "drawn twice doubles its stiffness and weight", areas=[n] + others))
    return out


def drawing_issues(model, frames: Iterable[str] = (), areas: Iterable[str] = ()) -> List[GeometryIssue]:
    """What the last command made that is probably not what was meant (shown right after
    it): members / shells on top of others, a member end left hanging just beside a joint
    or member (a missed snap), a floor / wall overlapping another."""
    tol = max(model.options.merge_tolerance, 1e-9)
    frames = [n for n in frames if n in model.frames]
    areas = [n for n in areas if n in model.areas]
    out = duplicates_of(model, frames, areas)
    for check, names in ((_near_misses, frames), (_area_overlaps, areas)):
        if names:
            try:
                out += check(model, tol, set(names))
            except Exception:
                pass
    return out


def geometry_issues(model) -> List[GeometryIssue]:
    """Every geometry problem of the model (Check Model)."""
    tol = max(model.options.merge_tolerance, 1e-9)
    out: List[GeometryIssue] = []
    for f in model.frames.values():
        if f.point_i in model.points and f.point_j in model.points and \
                math.dist(model.points[f.point_i].xyz, model.points[f.point_j].xyz) <= tol:
            out.append(GeometryIssue("error", "ZERO_LENGTH_FRAME", f"{f.name} has no length (its two ends coincide)",
                                     [f.name]))
    for a in model.areas.values():
        if len(set(a.points)) != len(a.points):
            out.append(GeometryIssue("error", "REPEATED_CORNER", f"{a.name} uses a joint twice", areas=[a.name]))
        elif area_size(model, a) <= tol * tol * 10:
            out.append(GeometryIssue("error", "ZERO_AREA", f"{a.name} has no area (its corners are on one line)",
                                     areas=[a.name]))
    out += duplicates_of(model, list(model.frames), list(model.areas))
    for grp in coincident_groups(model.points.values(), tol):
        out.append(GeometryIssue("warning", "COINCIDENT_JOINTS",
                                 f"joints {', '.join(sorted(grp))} are at the same place but not connected "
                                 "(Edit > Merge Joints)", points=sorted(grp)))
    used = set()
    for f in model.frames.values():
        used |= {f.point_i, f.point_j}
    for a in model.areas.values():
        used |= set(a.points)
    for s in getattr(model, "line_springs", {}).values():
        used |= {s.point_i, s.point_j}
    for lk in getattr(model, "links", {}).values():
        used |= {lk.point_i, lk.point_j}
    from .point_ops import has_assignments
    lonely = [n for n, p in model.points.items() if n not in used and not has_assignments(p)]
    if lonely:
        out.append(GeometryIssue("info", "UNUSED_JOINTS",
                                 f"{len(lonely)} joint(s) are used by nothing and carry nothing "
                                 f"({', '.join(lonely[:8])}{'...' if len(lonely) > 8 else ''})", points=lonely))
    for check in (_near_misses, _area_overlaps, _base_without_support, _far_away, _section_sizes,
                  _material_moduli, _self_weight, _upward_gravity_loads, _membrane_cantilevers,
                  _membrane_without_beams, _stiffness_sets):
        try:
            out += check(model, tol)
        except Exception:  # a check must never stop Check Model / the analysis
            pass
    return out


# ------------------------------------------------------------------ the mistakes of a new user
def _metre(model) -> float:
    """One metre in model length units."""
    try:
        return 1.0 / model.units.factor("length")
    except Exception:
        return 1.0


def _near_misses(model, tol, only: Optional[Set[str]] = None) -> List[GeometryIssue]:
    """A member end left hanging just beside a joint or a member (a snap missed): the two
    are not joined, so the load goes another way than drawn.  Only free ends - an end
    joint used by that one member and nothing else, with no support - are looked at."""
    near = 0.12 * _metre(model)
    use: Dict[str, int] = {}
    for f in model.frames.values():
        for q in (f.point_i, f.point_j):
            use[q] = use.get(q, 0) + 1
    for a in model.areas.values():
        for q in a.points:
            use[q] = use.get(q, 0) + 2
    for s in getattr(model, "line_springs", {}).values():
        for q in (s.point_i, s.point_j):
            use[q] = use.get(q, 0) + 2
    for lk in getattr(model, "links", {}).values():
        for q in (lk.point_i, lk.point_j):
            use[q] = use.get(q, 0) + 2
    ends = []
    for f in model.frames.values():
        if f.type == "Null" or (only is not None and f.name not in only):
            continue
        for q in (f.point_i, f.point_j):
            p = model.points.get(q)
            if p is not None and use.get(q) == 1 and not any(p.restraints) and not any(p.springs) \
                    and not any(getattr(p, "rotational_springs", (0, 0, 0))):
                ends.append((f, q))
    if not ends:
        return []
    names = [n for n in model.points if n in use]
    X = np.array([model.points[n].xyz for n in names], float)
    frames = [g for g in model.frames.values() if g.type != "Null" and g.point_i in model.points and g.point_j in model.points]
    A = np.array([model.points[g.point_i].xyz for g in frames], float)
    B = np.array([model.points[g.point_j].xyz for g in frames], float)
    out = []
    for f, q in ends:
        x = np.array(model.points[q].xyz, float)
        other = f.point_j if q == f.point_i else f.point_i
        d = np.linalg.norm(X - x, axis=1)
        best = None
        for k in np.argsort(d)[:6]:
            if names[k] in (q, other) or not (tol < d[k] <= near):
                continue
            best = (float(d[k]), f"joint {names[k]}")
            break
        if best is None and len(frames):
            ab = B - A
            L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30)
            t = np.clip(np.einsum("ij,ij->i", x - A, ab) / L2, 0.0, 1.0)
            dist = np.linalg.norm(A + t[:, None] * ab - x, axis=1)
            for k in np.argsort(dist)[:6]:
                g = frames[k]
                if g.name == f.name or q in (g.point_i, g.point_j) or other in (g.point_i, g.point_j):
                    continue
                if tol < dist[k] <= near:
                    best = (float(dist[k]), f"{g.type.lower()} {g.name}")
                break
        if best is not None:
            out.append(GeometryIssue(
                "warning", "NEAR_MISS",
                f"the end of {f.name} at joint {q} is {best[0]:.3f} from {best[1]} but not joined to it - "
                f"it hangs free, the load does not go into {best[1].split()[-1]} (a missed snap?). Move the end onto "
                f"it, or merge the joints (Edit > Merge Joints)", [f.name], points=[q]))
    return out


def _plane_key(model, a, tol):
    P = np.array([model.points[q].xyz for q in a.points], float)
    if len(P) < 3:
        return None, None
    c = P.mean(axis=0)
    n = np.zeros(3)
    for i in range(len(P)):  # Newell's normal
        p0, p1 = P[i], P[(i + 1) % len(P)]
        n += np.cross(p0 - c, p1 - c)
    L = np.linalg.norm(n)
    if L <= 1e-12:
        return None, None
    n = n / L
    k = int(np.argmax(np.abs(n)))
    if n[k] < 0:
        n = -n
    return n, P


def _inside_2d(pt, poly, eps) -> int:
    """1 strictly inside, 0 on the boundary (within eps), -1 outside."""
    x, y = pt
    m = len(poly)
    for i in range(m):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % m]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
        if math.hypot(x1 + t * dx - x, y1 + t * dy - y) <= eps:
            return 0
    inside = False
    for i in range(m):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % m]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return 1 if inside else -1


def _cross_properly(p1, p2, q1, q2, eps) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    s = eps * max(1.0, math.dist(p1, p2), math.dist(q1, q2))
    return ((d1 > s and d2 < -s) or (d1 < -s and d2 > s)) and ((d3 > s and d4 < -s) or (d3 < -s and d4 > s))


def _area_overlaps(model, tol, only: Optional[Set[str]] = None) -> List[GeometryIssue]:
    """Two floors (or two walls) in one plane covering part of the same surface: that
    part has twice the slab - its weight, its loads and its stiffness counted twice.
    (Exactly the same corners is the duplicate check's.)"""
    groups: Dict[tuple, list] = {}
    for a in model.areas.values():
        if a.type not in ("Floor", "Wall") or len(a.points) < 3 or any(q not in model.points for q in a.points):
            continue
        n, P = _plane_key(model, a, tol)
        if n is None:
            continue
        d = float(n @ P.mean(axis=0))
        key = (a.type, tuple(np.round(n, 3)), round(d / max(tol * 10, 1e-9)))
        groups.setdefault(key, []).append((a, n, P))
    out = []
    eps = max(tol, 1e-9)
    for items in groups.values():
        if len(items) < 2 or (only is not None and not any(a.name in only for a, _, _ in items)):
            continue
        n = items[0][1]
        u = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        polys = [(a, [(float(p @ u), float(p @ v)) for p in P]) for a, _, P in items]
        boxes = [(min(x for x, _ in pg), min(y for _, y in pg), max(x for x, _ in pg), max(y for _, y in pg))
                 for _, pg in polys]
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                bi, bj = boxes[i], boxes[j]
                if bi[0] >= bj[2] - eps or bj[0] >= bi[2] - eps or bi[1] >= bj[3] - eps or bj[1] >= bi[3] - eps:
                    continue
                (a, pa), (b, pb) = polys[i], polys[j]
                if only is not None and a.name not in only and b.name not in only:
                    continue
                if set(a.points) == set(b.points):
                    continue  # (drawn twice: the duplicate check says so)
                hit = any(_inside_2d(p, pb, eps) == 1 for p in pa) or any(_inside_2d(p, pa, eps) == 1 for p in pb)
                if not hit:
                    ca = (sum(x for x, _ in pa) / len(pa), sum(y for _, y in pa) / len(pa))
                    cb = (sum(x for x, _ in pb) / len(pb), sum(y for _, y in pb) / len(pb))
                    hit = (_inside_2d(ca, pb, eps) == 1 and _inside_2d(ca, pa, eps) == 1) or \
                          (_inside_2d(cb, pa, eps) == 1 and _inside_2d(cb, pb, eps) == 1)
                if not hit:
                    hit = any(_cross_properly(pa[k], pa[(k + 1) % len(pa)], pb[l], pb[(l + 1) % len(pb)], 1e-9)
                              for k in range(len(pa)) for l in range(len(pb)))
                if hit:
                    what = "floors" if a.type == "Floor" else "walls"
                    out.append(GeometryIssue(
                        "warning", "AREAS_OVERLAP",
                        f"{what} {a.name} and {b.name} overlap: where they overlap the {a.type.lower()} is there twice "
                        f"- its weight, loads and stiffness counted twice. Reshape one so they only meet at their edges",
                        areas=[a.name, b.name]))
    return out


def _base_without_support(model, tol) -> List[GeometryIssue]:
    """A column or wall standing on a joint at the lowest level that has no support."""
    used = [q for f in model.frames.values() for q in (f.point_i, f.point_j)] + \
        [q for a in model.areas.values() for q in a.points]
    zs = [model.points[q].z for q in used if q in model.points]
    if not zs:
        return []
    z0 = min(zs)
    if not any(any(p.restraints) or any(p.springs) for p in model.points.values()):
        return []  # nothing supported at all: the analysis says so plainly
    bad: Dict[str, str] = {}
    for f in model.frames.values():
        if f.type != "Column":
            continue
        for q in (f.point_i, f.point_j):
            p = model.points.get(q)
            if p is not None and abs(p.z - z0) <= tol and not any(p.restraints) and not any(p.springs) \
                    and not any(getattr(p, "rotational_springs", (0, 0, 0))):
                bad.setdefault(q, f.name)
    for a in model.areas.values():
        if a.type != "Wall":
            continue
        for q in a.points:
            p = model.points.get(q)
            if p is not None and abs(p.z - z0) <= tol and not any(p.restraints) and not any(p.springs):
                bad.setdefault(q, a.name)
    if not bad:
        return []
    pts = sorted(bad)
    return [GeometryIssue("warning", "BASE_NOT_SUPPORTED",
                          f"{len(pts)} joint(s) at the base level carry a column / wall but have no support: "
                          + ", ".join(f"{q} ({bad[q]})" for q in pts[:6]) + ("..." if len(pts) > 6 else "")
                          + " - the member hangs from the rest of the structure. Assign > Joint > Restraints",
                          frames=[bad[q] for q in pts if bad[q] in model.frames][:1], points=pts)]


def _far_away(model, tol) -> List[GeometryIssue]:
    """Joints far outside the grid (a slip of the mouse, or a length in the wrong units)."""
    xs = [g.ordinate for g in model.grid_lines if g.direction == "X"]
    ys = [g.ordinate for g in model.grid_lines if g.direction == "Y"]
    if len(xs) < 2 or len(ys) < 2:
        return []
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    margin = max(math.hypot(x1 - x0, y1 - y0), 20.0 * _metre(model))
    used = {q for f in model.frames.values() for q in (f.point_i, f.point_j)} | \
        {q for a in model.areas.values() for q in a.points}
    far = [q for q in used if q in model.points and (
        model.points[q].x < x0 - margin or model.points[q].x > x1 + margin or
        model.points[q].y < y0 - margin or model.points[q].y > y1 + margin)]
    if not far:
        return []
    owners = [f.name for f in model.frames.values() if f.point_i in far or f.point_j in far]
    return [GeometryIssue("warning", "FAR_FROM_GRID",
                          f"{len(far)} joint(s) lie far outside the grid ({', '.join(sorted(far)[:5])}) - "
                          f"a click that slipped, or a length typed in other units?"
                          + (f" Used by {', '.join(owners[:4])}" if owners else ""), owners[:1], points=sorted(far))]


def _section_sizes(model, tol) -> List[GeometryIssue]:
    """A frame section of a catalogue shape whose dimensions are not valid (zero depth
    ...): its properties are not those of any real section."""
    from .section_shapes import SHAPES, check_dims
    used = {f.section for f in model.frames.values()}
    out = []
    for s in model.frame_sections.values():
        sh = SHAPES.get(getattr(s, "shape", "General"))
        if sh is None or s.name not in used:
            continue
        dims = {k: getattr(s, k, None) for k in ("t3", "t2", "tf", "tw", "t2b", "tfb", "dis", "lip", "r", "r2")}
        dims.update({k: v for k, v in (getattr(s, "params", None) or {}).items() if k == "slope"})
        try:
            errs = check_dims(sh, dims)
        except Exception:
            errs = []
        if errs:
            out.append(GeometryIssue("error", "INVALID_SECTION",
                                     f"section {s.name}: " + "; ".join(errs) + " - Define > Section Properties",
                                     [f.name for f in model.frames.values() if f.section == s.name][:1]))
    return out


_E_RANGE = {"Concrete": (5e9, 6e10), "Steel": (1.5e11, 2.5e11), "Rebar": (1.5e11, 2.5e11),
            "Tendon": (1.5e11, 2.5e11), "Aluminum": (5e10, 9e10), "Masonry": (1e9, 3e10)}


def _material_moduli(model, tol) -> List[GeometryIssue]:
    """E far from that of the material type (a value typed in other units: MPa for kN/m2 ...)."""
    try:
        to_pa = model.units.factor("stress")
    except Exception:
        return []
    used = {getattr(s, "material", None) for s in model.frame_sections.values()} | \
        {getattr(s, "material", None) for s in model.shell_sections.values()}
    out = []
    for m in model.materials.values():
        r = _E_RANGE.get(m.type)
        if r is None or m.name not in used or m.E <= 0:
            continue
        e = m.E * to_pa
        if not (r[0] <= e <= r[1]):
            ratio = e / math.sqrt(r[0] * r[1])
            hint = f" ({ratio:.3g} times the usual)" if ratio < 0.05 or ratio > 20 else ""
            out.append(GeometryIssue("warning", "UNUSUAL_E",
                                     f"material {m.name} ({m.type}): E = {m.E:.6g} is far from a {m.type.lower()}'s "
                                     f"{r[0] / to_pa:.4g} - {r[1] / to_pa:.4g}{hint}: typed in other units? "
                                     f"Define > Materials"))
    return out


def _self_weight(model, tol) -> List[GeometryIssue]:
    sw = [p for p in model.load_patterns.values() if abs(getattr(p, "self_weight_multiplier", 0.0)) > 0]
    if len(sw) > 1:
        return [GeometryIssue("warning", "SELF_WEIGHT_TWICE",
                              f"self weight is in {len(sw)} load patterns ({', '.join(p.name for p in sw)}): "
                              f"combined, it is counted {len(sw)} times. Keep the multiplier 1 in one pattern "
                              f"(Dead) only - Define > Load Patterns")]
    if not sw and (model.frames or model.areas) and model.load_patterns:
        used = {getattr(x, "material", None) for x in model.frame_sections.values()} | \
            {getattr(x, "material", None) for x in model.shell_sections.values()}
        if any(getattr(m, "weight_density", 0.0) > 0 for n, m in model.materials.items() if n in used):
            # (a note, not a warning: test models leave it out on purpose)
            return [GeometryIssue("info", "NO_SELF_WEIGHT",
                                  "no load pattern includes the self weight, though the materials have weight: "
                                  "the members' own weight is in no load. Usually the Dead pattern has self "
                                  "weight multiplier 1 - Define > Load Patterns")]
    return []


def _stiffness_sets(model, tol) -> List[GeometryIssue]:
    """Shells with other modifiers in the lateral load cases: which cases use them."""
    from .loads import (LATERAL_PATTERN_TYPES, LinearStaticCase, ModalCase, ResponseSpectrumCase,
                        stiffness_set_of)
    from .nonlinear import NonlinearStaticCase, TimeHistoryCase
    secs = {n for n, s in model.shell_sections.items() if s.lateral_modifiers is not None}
    areas = [a.name for a in model.areas.values()
             if not a.is_opening and (a.lateral_modifiers is not None or a.section in secs)]
    if not areas:
        return []
    out = []
    lateral, gravity = [], []
    for name, case in model.load_cases.items():
        if isinstance(case, (NonlinearStaticCase, TimeHistoryCase)):
            gravity.append(name)
            continue
        try:
            s = stiffness_set_of(model, name)
        except Exception as exc:
            out.append(GeometryIssue("error", "BAD_STIFFNESS_SET", str(exc)))
            continue
        (lateral if s == "lateral" else gravity).append(name)
        if isinstance(case, LinearStaticCase) and s == "gravity" and (case.stiffness or "auto") == "auto":
            types = {model.load_patterns[p].type for p, f in case.loads.items() if f and p in model.load_patterns}
            if types & set(LATERAL_PATTERN_TYPES):
                out.append(GeometryIssue(
                    "warning", "MIXED_STIFFNESS_CASE",
                    f"Load case {name} has both gravity and lateral load patterns: it is solved with the gravity "
                    f"stiffness (the shells' modifiers, not their lateral ones). Split it, or set its Stiffness "
                    f"(Define > Load Cases) to lateral"))
        if s == "gravity" and isinstance(case, (ModalCase, ResponseSpectrumCase)):
            out.append(GeometryIssue(
                "info", "GRAVITY_SET_DYNAMIC",
                f"Load case {name} ({case.type}) is set to the gravity stiffness: the shells' lateral modifiers "
                f"are not used in it"))
    nl = [n for n, c in model.load_cases.items() if isinstance(c, (NonlinearStaticCase, TimeHistoryCase))]
    out.append(GeometryIssue(
        "info", "STIFFNESS_SETS",
        f"{len(areas)} slab / wall object(s) have other stiffness modifiers in the lateral cases: "
        f"lateral stiffness: {', '.join(lateral) or '(no case)'}; gravity stiffness: {', '.join(gravity) or '-'}"
        + (f" (nonlinear / time history cases {', '.join(nl)} always use the gravity stiffness)" if nl else "")
        + ". The combinations add the results of both (linear superposition)", areas=areas[:50]))
    return out


def _membrane_without_beams(model, tol) -> List[GeometryIssue]:
    """A membrane floor with no beam under it: its loads would be lost (an error)."""
    from ..mesh.panel_loads import membrane_floors_without_beams
    names = membrane_floors_without_beams(model)
    out = []
    for name in names:
        a = model.areas[name]
        out.append(GeometryIssue(
            "error", "MEMBRANE_NO_BEAMS",
            f"Membrane floor {name} (section {a.section}) has no beam under it: a membrane carries nothing out "
            f"of its plane, so its loads and self weight would go nowhere. A flat slab / post-tensioned slab "
            f"without beams must be a shell (Define > Shell Sections: Shell-Thin / Shell-Thick), or draw the "
            f"beams under it", areas=[name]))
    return out


def _membrane_cantilevers(model, tol) -> List[GeometryIssue]:
    """Membrane floors that cantilever past a beam: where the moment at the root goes
    (a membrane has no bending stiffness - the analysis puts it on the frames)."""
    from ..mesh.panel_loads import membrane_cantilever_notes
    notes = membrane_cantilever_notes(model)
    agg: Dict[tuple, list] = {}
    for floor, bar, mode, far, S, pattern, F in notes:
        k = (floor, bar, mode, far)
        a = agg.setdefault(k, [0.0, 0.0])
        a[0] += abs(S)
        a[1] += abs(F)
    out = []
    m = _metre(model)
    for (floor, bar, mode, far), (S, F) in sorted(agg.items(), key=lambda kv: str(kv[0])):
        lever = S / F if F > 0 else 0.0
        if lever < 0.15 * m:  # a slab edge a little past the beam: nothing worth saying
            continue
        if mode == "torsion":
            out.append(GeometryIssue(
                "info", "MEMBRANE_CANTILEVER_TORSION",
                f"Membrane floor {floor} cantilevers about {2 * lever / m:.2f} m past {bar} with no floor behind it: "
                f"the moment at the root goes into {bar} as torsion (design {bar} for torsion, or make {floor} a "
                f"shell / add a beam at its free edge)", frames=[bar], areas=[floor]))
        else:
            out.append(GeometryIssue(
                "info", "MEMBRANE_CANTILEVER",
                f"Membrane floor {floor} cantilevers about {2 * lever / m:.2f} m past {bar}: the moment at the root is "
                f"balanced by the floor behind it - more load on {bar}, less on {far} (the span behind as simply "
                f"supported). The floor's own moments come from a shell model (Slab Design Model)",
                frames=[bar, far] if far else [bar], areas=[floor]))
    return out


def _upward_gravity_loads(model, tol) -> List[GeometryIssue]:
    """Gravity loads with a negative value act upward: usually a sign typed by mistake."""
    frames = sorted({f.name for f in model.frames.values() for ld in getattr(f, "loads", ())
                     if getattr(ld, "direction", "") == "Gravity"
                     and (getattr(ld, "value_a", 0.0) < 0 or getattr(ld, "value_b", 0.0) < 0)})
    panels = sorted({p.area or p.name for p in getattr(model, "panel_loads", {}).values() for c in p.loads
                     if getattr(c, "direction", "") == "Gravity" and getattr(c, "value", 0.0) < 0})
    pts = sorted({n for n, p in model.points.items() for ld in getattr(p, "loads", ())
                  if getattr(ld, "direction", "Gravity") == "Gravity" and getattr(ld, "value", 0.0) < 0})
    out = []
    if frames or panels or pts:
        where = ", ".join(frames[:4] + panels[:4] + pts[:4])
        out.append(GeometryIssue("warning", "UPWARD_GRAVITY_LOAD",
                                 f"gravity loads with a negative value push UP on {where}"
                                 f"{'...' if len(frames) + len(panels) + len(pts) > 12 else ''} - a minus typed by "
                                 f"mistake? (a downward load is positive)", frames[:1],
                                 areas=[a for a in panels if a in model.areas][:1]))
    return out


__all__ = ["GeometryIssue", "geometry_issues", "duplicates_of", "drawing_issues", "area_size"]
