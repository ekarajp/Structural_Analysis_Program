"""Find an object from typed text: a unique name / id ("B23", "P104"), a label
("C5" - with a story name "C5 Story3" when the label repeats on every story),
coordinates ("12.5, 6, 9" or "12.5 6" = the nearest object in plan) or a grid
intersection ("B/3", "B/3 Story2")."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..model import Model
from . import locate

KIND_NAMES = {"points": "Joint", "frames": "Frame", "areas": "Shell", "panel_loads": "Load panel"}


@dataclass
class Hit:
    attr: str            # points | frames | areas | panel_loads | "xyz" (a bare location)
    name: str
    text: str            # one line for the result list
    xyz: Optional[tuple] = None   # the searched location (coordinates / grid)
    score: float = 0.0   # lower = better
    query: str = "name"  # name | location (what was typed)


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_COORD = re.compile(rf"^\(?\s*({_NUM})\s*[,;\s]\s*({_NUM})(?:\s*[,;\s]\s*({_NUM}))?\s*\)?$")


def _story_token(model: Model, tokens: List[str]):
    """Split a trailing / leading story name off the search text."""
    names = {s.lower(): s for s in model.stories}
    keep, story = [], None
    for t in tokens:
        if story is None and t.lower() in names:
            story = names[t.lower()]
        else:
            keep.append(t)
    return keep, story


def _grid_xy(model: Model, text: str):
    """'B/3', 'B-3', 'B 3' -> (x, y) of the grid intersection (either order)."""
    parts = [p for p in re.split(r"\s*[/\\-]\s*|\s+", text.strip()) if p]
    if len(parts) != 2:
        return None
    gx = {g.name.lower(): g.ordinate for g in model.grid_lines if g.direction == "X"}
    gy = {g.name.lower(): g.ordinate for g in model.grid_lines if g.direction == "Y"}
    a, b = parts[0].lower(), parts[1].lower()
    if a in gx and b in gy:
        return gx[a], gy[b]
    if b in gx and a in gy:
        return gx[b], gy[a]
    return None


def _objects(model: Model):
    for attr in locate.KINDS:
        for name, obj in getattr(model, attr, {}).items():
            yield attr, name, obj


def _describe(model: Model, attr: str, name: str) -> str:
    return locate.location_text(model, attr, name)


def _dist_to(X: np.ndarray, p: np.ndarray, plan: bool) -> float:
    """Distance from p to an object's points / segment / polygon (plan: x-y only)."""
    if plan:
        X, p = X[:, :2], p[:2]
    if len(X) == 1:
        return float(np.linalg.norm(X[0] - p))
    if len(X) == 2:
        a, b = X
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
        return float(np.linalg.norm(a + t * ab - p))
    # polygon: 0 inside (projected on its plane), else distance to the nearest edge
    d = min(_dist_to(np.array([X[i], X[(i + 1) % len(X)]]), p, False) for i in range(len(X)))
    c = X.mean(axis=0)
    n = np.zeros(X.shape[1] if X.shape[1] == 3 else 3)
    if X.shape[1] == 3:
        for i in range(len(X)):
            n += np.cross(X[i] - c, X[(i + 1) % len(X)] - c)
        nn = np.linalg.norm(n)
        if nn > 0:
            n /= nn
            h = float(np.dot(p - c, n))
            q = p - h * n
            if _inside(X, q, n):
                return abs(h)
    else:
        q3 = np.c_[X, np.zeros(len(X))]
        if _inside(q3, np.r_[p, 0.0], np.array([0, 0, 1.0])):
            return 0.0
    return d


def _inside(X: np.ndarray, q: np.ndarray, n: np.ndarray) -> bool:
    k = int(np.argmax(np.abs(n)))
    ax = [i for i in range(3) if i != k]
    P, x = X[:, ax], q[ax]
    inside = False
    for i in range(len(P)):
        (x1, y1), (x2, y2) = P[i], P[(i + 1) % len(P)]
        if (y1 > x[1]) != (y2 > x[1]) and x[0] < x1 + (x[1] - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _near(model: Model, p: np.ndarray, plan: bool, story: Optional[str], limit: int) -> List[Hit]:
    """Objects nearest to a location: joints first, then members / shells."""
    rank = {"points": 0, "frames": 1, "areas": 2, "panel_loads": 3}
    out = []
    for attr, name, obj in _objects(model):
        if story and locate.object_story(model, attr, name) != story:
            continue
        X = locate.object_xyz(model, attr, name)
        if X is None:
            continue
        d = _dist_to(X, p, plan)
        out.append((round(d, 9), rank[attr], _natural(name), attr, name))
    out.sort()
    u = model.units.shown("length")
    return [Hit(attr, name, f"{_describe(model, attr, name)}   ({d:.3f} {u} away)",
                tuple(p), score=d, query="location") for d, _r, _n, attr, name in out[:limit]]


def find_objects(model: Model, text: str, limit: int = 30) -> List[Hit]:
    """Everything that matches the typed text, best first."""
    text = (text or "").strip()
    if not text:
        return []
    tokens, story = _story_token(model, text.split())
    body = " ".join(tokens)
    tol = max(model.options.merge_tolerance, 1e-3) * 10
    # 1) coordinates
    m = _COORD.match(body) if body else None
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if m.group(3) is not None:
            p, plan = np.array([x, y, float(m.group(3))]), False
        elif story:
            p, plan = np.array([x, y, model.stories[story].elevation]), False
        else:
            p, plan = np.array([x, y, 0.0]), True
        return _location_hits(model, p, plan, story, limit, tol)
    # 2) grid intersection
    g = _grid_xy(model, body) if body else None
    if g is not None:
        z = model.stories[story].elevation if story else 0.0
        return _location_hits(model, np.array([g[0], g[1], z]), story is None, story, limit, tol)
    # 3) names, labels (exact before partial)
    key = body.lower()
    hits = []
    for attr, name, obj in _objects(model):
        label = (getattr(obj, "label", None) or "")
        if story and locate.object_story(model, attr, name) != story:
            continue
        nl, ll = name.lower(), label.lower()
        if not key:
            score = 5.0
        elif nl == key:
            score = 0.0
        elif ll == key:
            score = 1.0
        elif nl.startswith(key) or ll.startswith(key):
            score = 2.0 + len(nl) / 1000
        elif key in nl or key in ll:
            score = 3.0 + len(nl) / 1000
        else:
            continue
        line = _describe(model, attr, name)
        if label and label != name:
            line = f"{line}   [label {label}]"
        hits.append(Hit(attr, name, line, score=score))
    hits.sort(key=lambda h: (h.score, locate.KINDS.index(h.attr), _natural(h.name)))
    return hits[:limit]


def _location_hits(model, p, plan, story, limit, tol) -> List[Hit]:
    near = _near(model, p, plan, story, limit)
    u = model.units.shown("length")
    where = (f"x = {p[0]:g}, y = {p[1]:g}" + ("" if plan else f", z = {p[2]:g}") + f" {u}"
             + f"  ·  grid {locate.grid_ref(model, p)}" + (f"  ·  {story}" if story else ""))
    exact = [h for h in near if h.score <= tol]
    loc = Hit("xyz", "", f"Location {where}" + ("" if exact else "   (no object exactly here)"),
              tuple(p), score=-1.0 if not exact else tol, query="location")
    return (exact + [loc] + [h for h in near if h.score > tol])[:limit]


def _natural(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def completions(model: Model) -> List[str]:
    """Words for the search box's completer: names, labels, stories, grid lines."""
    words = set()
    for _attr, name, obj in _objects(model):
        words.add(name)
        if getattr(obj, "label", None):
            words.add(obj.label)
    words.update(model.stories)
    return sorted(words, key=_natural)
