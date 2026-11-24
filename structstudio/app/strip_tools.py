"""Draw > Draw Design Strips: click the points of a strip's centre line in a plan."""
from __future__ import annotations

from typing import Optional

from ..model import ModelError
from .tools import TOOLS, Preview, SnapResult, Tool, _dist, next_step


class StripTool(Tool):
    """A design strip: click the points of its centre line (it may turn), then Enter,
    double-click or right-click.  Widths: Auto (worked out from the supports and the
    column strips - ACI 318 8.4.1.5) or the specified left / right widths.  The strip
    is design data: drawing it keeps the analysis results."""

    name = "strip"
    title = "Draw Design Strips"
    prompt = "click the first point of the strip's centre line (in a plan)"
    next_prompt = "click the next point (the strip may turn)"
    views = ("plan",)
    shows_strips = True  # the windows show the strips while drawing them
    finishes_on_right = True  # right click = this strip is done (not cancelled)

    def hint(self, state=None) -> str:
        s = self.settings
        what = f"layer {s.strip_layer} {s.strip_kind.lower()} strip, widths {s.strip_widths.lower()}"
        if self.points:
            return next_step(self.title, self.next_prompt, "Enter / double-click / right-click = finish",
                             "Esc = cancel this strip", what)
        return next_step(self.title, self.prompt, what, "Esc = back to Select")

    def click(self, pt, snap: Optional[SnapResult] = None) -> bool:
        p = self._p3(pt)
        if self.points and _dist(p, self.points[-1]) <= self.tol:
            return False
        self.points.append(p)
        return False

    def double_click(self, pt) -> bool:
        p = self._p3(pt)
        if not self.points or _dist(p, self.points[-1]) > self.tol:
            self.points.append(p)
        return self.finish()

    def finish(self) -> bool:
        if len(self.points) < 2:
            self.points = []
            return False
        pts = list(self.points)
        self.points = []
        if self.plane.kind != "plan":
            raise ModelError("PLAN_ONLY", "Design strips are drawn in a plan window")
        z = self.elevation()
        pts = [(p[0], p[1], z) for p in pts]
        s = self.settings
        from .strip_commands import draw_strip
        name = draw_strip(self.doc, pts, layer=s.strip_layer, kind=s.strip_kind, widths=s.strip_widths,
                          left=s.strip_width_left, right=s.strip_width_right)
        if name:
            self.created.append(name)
        return bool(name)

    def preview(self, cursor) -> Preview:
        pv = Preview(marker=None if cursor is None else self._p3(cursor))
        pts = list(self.points)
        if cursor is not None and pts:
            c = self._p3(cursor)
            pts.append(c)
            pv.text = f"L = {_dist(pts[-2], c):.3f}"
        pv.lines = [(a, b) for a, b in zip(pts[:-1], pts[1:])]
        s = self.settings
        if s.strip_widths == "Specified" and len(pts) >= 2:
            import math
            for a, b in zip(pts[:-1], pts[1:]):  # the band's edges, dashed
                dx, dy = b[0] - a[0], b[1] - a[1]
                L = math.hypot(dx, dy)
                if L <= 0:
                    continue
                nx, ny = -dy / L, dx / L
                for w in (s.strip_width_left, -s.strip_width_right):
                    pv.guides.append(((a[0] + nx * w, a[1] + ny * w, a[2]), (b[0] + nx * w, b[1] + ny * w, b[2])))
        return pv


TOOLS["strip"] = StripTool
