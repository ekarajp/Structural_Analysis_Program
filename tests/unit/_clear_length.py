"""The weighed length of a frame, worked out here from the geometry (independently of the
mesher): ETABS weighs a beam/brace only between the faces of the columns at its ends - half
the column's size along the beam at each end (auto end offsets) - and a column on its whole
length."""
import numpy as np

from structstudio.mesh.geometry import frame_axes


def weighed_length(m, f):
    a = np.array(m.points[f.point_i].xyz, float)
    b = np.array(m.points[f.point_j].xyz, float)
    L = float(np.linalg.norm(b - a))
    if f.type == "Column" or not m.frame_auto_end_offsets(f) and not (f.end_offset_i or f.end_offset_j):
        return L
    if not m.frame_auto_end_offsets(f):
        return L - f.end_offset_i - f.end_offset_j
    d = (b - a) / L
    off = 0.0
    for p in (f.point_i, f.point_j):
        half = 0.0
        for c in m.frames.values():
            if c.type != "Column" or p not in (c.point_i, c.point_j):
                continue
            s = m.frame_sections[c.section]
            ax = frame_axes(np.array(m.points[c.point_i].xyz, float), np.array(m.points[c.point_j].xyz, float), c.angle)
            half = max(half, 0.5 * (abs(d @ ax[1]) * (s.t3 or 0.0) + abs(d @ ax[2]) * (s.t2 or 0.0)))
        off += half
    return L - off
