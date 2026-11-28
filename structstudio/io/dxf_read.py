"""Reading a DXF drawing as *reference geometry*: the line work of the model space as
plain 2D segments (file coordinates, file units), with the layer and the colour of each,
plus the points worth snapping to (ends of arcs, centres, insertion points).

Only what a drawing to trace over needs is read - no dependency on a CAD library:

* LINE, LWPOLYLINE / POLYLINE (with bulges), ARC, CIRCLE, ELLIPSE, SPLINE, SOLID / TRACE /
  3DFACE (outline), LEADER, POINT;
* INSERT of blocks (nested, scaled, rotated, mirrored, arrays) and DIMENSION (its drawn
  block); layer "0" and colour BYBLOCK / BYLAYER are inherited as in CAD;
* the layers (name, colour, off / frozen), the drawing units ($INSUNITS).

Curves become short chords (marked: a chord end is no end point to snap to).  Texts,
hatches, images and 3D solids are not drawn; :attr:`Drawing.skipped` counts them.  The
paper space (layouts) is left out.  Binary DXF and DWG are refused with a message that
says what to do (save as ASCII DXF).
"""
from __future__ import annotations

import codecs
import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# $INSUNITS -> unit name (the ones a building drawing comes in)
INSUNITS = {1: "in", 2: "ft", 3: "mi", 4: "mm", 5: "cm", 6: "m", 7: "km", 10: "yd", 14: "dm"}
UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "dm": 0.1, "m": 1.0, "km": 1000.0, "in": 0.0254, "ft": 0.3048,
             "yd": 0.9144, "mi": 1609.344}
UNIT_NAMES = ("mm", "cm", "m", "in", "ft", "dm", "km", "yd", "mi")

BYLAYER, BYBLOCK = -1, -2
_LAYER_OF = -100  # colour "of layer k" is _LAYER_OF - k (an insert that is BYLAYER gives it to BYBLOCK entities)
PT_END, PT_CENTER, PT_INSERT, PT_POINT = 0, 1, 2, 3
CHORD_DEG = 7.5  # a curve is drawn as chords of this angle
MAX_DEPTH = 12  # blocks inside blocks
MAX_SEGMENTS = 6_000_000

# AutoCAD colour index 1..9 and the grey ramp; the others come from the hue table below
_ACI_BASE = {1: (255, 0, 0), 2: (255, 255, 0), 3: (0, 255, 0), 4: (0, 255, 255), 5: (0, 0, 255), 6: (255, 0, 255),
             7: (255, 255, 255), 8: (128, 128, 128), 9: (192, 192, 192), 250: (51, 51, 51), 251: (80, 80, 80),
             252: (105, 105, 105), 253: (130, 130, 130), 254: (190, 190, 190), 255: (255, 255, 255)}


def aci_rgb(index: int) -> Tuple[int, int, int]:
    """RGB of an AutoCAD colour index (1..255)."""
    index = int(index)
    if index in _ACI_BASE:
        return _ACI_BASE[index]
    if not 10 <= index <= 249:
        return (255, 255, 255)
    # 24 hues x 10 shades: even = full saturation, odd = half; brightness 100 / 80 / 60 / 50 / 30 %
    hue = (index - 10) // 10 * 15.0
    shade = (index - 10) % 10
    value = (1.0, 0.8, 0.6, 0.5, 0.3)[shade // 2]
    sat = 1.0 if shade % 2 == 0 else 0.5
    h = hue / 60.0
    c = value * sat
    x = c * (1 - abs(h % 2 - 1))
    r, g, b = ((c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x))[int(h) % 6]
    m = value - c
    return (int(round((r + m) * 255)), int(round((g + m) * 255)), int(round((b + m) * 255)))


_SERIAL = itertools.count(1)


class DxfError(Exception):
    """The file cannot be read as a drawing (the message says why, for the user)."""


@dataclass
class Drawing:
    """The line work of a DXF file (model space), in file coordinates."""

    layers: List[str] = field(default_factory=list)
    layer_rgb: List[Tuple[int, int, int]] = field(default_factory=list)
    layers_off: List[str] = field(default_factory=list)  # off or frozen in the file
    seg: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # x1, y1, x2, y2
    seg_layer: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    seg_curve: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # a chord of a curve
    seg_rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pts_layer: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    pts_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    unit: Optional[str] = None  # from $INSUNITS; None: the file does not say
    metric: bool = True  # $MEASUREMENT
    version: str = ""
    skipped: Dict[str, int] = field(default_factory=dict)  # entity type -> how many were not drawn
    counts: Dict[str, int] = field(default_factory=dict)
    # the texts (drawn as letters, never snapped to): (x, y, height, angle in degrees, string, layer index,
    # (r, g, b), across 0 left / 1 centre / 2 right of the point, up 0 baseline / 1 bottom / 2 middle / 3 top,
    # width: the length of the string when the file states it, else 0, width factor)
    texts: list = field(default_factory=list)
    # line work that is text (the glyphs of a PDF font with no characters): drawn, never snapped to
    text_seg: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    text_seg_layer: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    text_seg_rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    # a page of a PDF (see pdf_read): the coordinates are millimetres on the sheet
    source: str = "dxf"  # "dxf" | "pdf"
    pages: int = 1
    page: int = 1
    sheet: Optional[Tuple[float, float]] = None  # the sheet, mm (PDF)
    plot_scale: Optional[float] = None  # 1 : n as the PDF states it; None: it does not say
    serial: int = field(default_factory=lambda: next(_SERIAL))  # (one number per drawing read, for caches)

    @property
    def extents(self) -> Optional[Tuple[float, float, float, float]]:
        """(xmin, ymin, xmax, ymax) of the line work; None for an empty drawing."""
        if not len(self.seg):
            if not len(self.pts):
                return None
            return (float(self.pts[:, 0].min()), float(self.pts[:, 1].min()),
                    float(self.pts[:, 0].max()), float(self.pts[:, 1].max()))
        x = self.seg[:, (0, 2)]
        y = self.seg[:, (1, 3)]
        return (float(x.min()), float(y.min()), float(x.max()), float(y.max()))

    def size(self) -> Tuple[float, float]:
        e = self.extents
        return (0.0, 0.0) if e is None else (e[2] - e[0], e[3] - e[1])

    def guess_unit(self) -> str:
        """The unit to offer when the file does not say: a building drawing is tens of
        metres across - thousands of drawing units mean millimetres, hundreds
        centimetres (inches when the file is imperial)."""
        if self.unit:
            return self.unit
        w = max(self.size())
        if not self.metric:
            return "in" if w > 400 else "ft"
        if w > 5000:
            return "mm"
        if w > 500:
            return "cm"
        return "m"

    def text_counts(self) -> Dict[str, int]:
        """Texts per layer (the glyph outlines of a layer count as one text)."""
        out = {name: 0 for name in self.layers}
        for t in self.texts:
            out[self.layers[t[5]]] += 1
        for k in np.unique(self.text_seg_layer).tolist() if len(self.text_seg) else []:
            out[self.layers[k]] += 1
        return out

    def layer_counts(self) -> Dict[str, int]:
        n = np.bincount(self.seg_layer, minlength=len(self.layers)) if len(self.seg) else np.zeros(len(self.layers), int)
        return {name: int(n[k]) for k, name in enumerate(self.layers)}


# --------------------------------------------------------------------------- reading
def _decodable(data: bytes, enc: str) -> bool:
    """Can all of ``data`` be read as ``enc``?  (Piece by piece: no second copy of a large file.)"""
    try:
        dec = codecs.getincrementaldecoder(enc)()
    except LookupError:
        return False
    try:
        for k in range(0, len(data), 1 << 23):
            dec.decode(data[k:k + (1 << 23)])
        dec.decode(b"", True)
    except UnicodeDecodeError:
        return False
    return True


def _text_bytes(data: bytes) -> Tuple[bytes, str]:
    """(the bytes of the file's text, the encoding of its values); refuses what is no
    ASCII DXF with what to do."""
    if data[:22].startswith(b"AutoCAD Binary DXF"):
        raise DxfError("This is a binary DXF file. Save it from the CAD program as an ASCII (text) DXF and try again.")
    if data[:6] in (b"AC1012", b"AC1014", b"AC1015", b"AC1018", b"AC1021", b"AC1024", b"AC1027", b"AC1032") or \
            data[:2] == b"AC" and data[2:6].isdigit():
        raise DxfError("This is a DWG file. Save it from the CAD program as DXF (Save As > DXF) and try again.")
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):  # (a UTF-16 text file, as some editors save it)
        try:
            return data.decode("utf-16").encode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            pass
    if data[:3] == b"\xef\xbb\xbf":  # (a UTF-8 byte order mark)
        data = data[3:]
    if _decodable(data, "utf-8"):
        return data, "utf-8"
    page = "cp1252"
    k = data.find(b"$DWGCODEPAGE")
    if k >= 0:
        tail = data[k:k + 80].split(b"\n")
        if len(tail) >= 3:
            name = tail[2].strip().decode("ascii", "replace").upper()
            if name.startswith("ANSI_"):
                page = "cp" + name[5:]
            elif name.startswith("DOS"):
                page = "cp" + name[3:]
    for enc in (page, "cp874"):
        if _decodable(data, enc):
            return data, enc
    return data, "latin-1"


class _Values:
    """The value lines of a DXF file: read as text when asked for (a file of 100 MB is
    ten million lines, most of them never looked at - the pattern lines of hatches)."""

    __slots__ = ("data", "s", "e", "codec")

    def __init__(self, data: bytes, s: np.ndarray, e: np.ndarray, codec: str):
        self.data, self.s, self.e, self.codec = data, s, e, codec

    def __len__(self) -> int:
        return len(self.s)

    def __getitem__(self, k):
        d, c = self.data, self.codec
        if isinstance(k, slice):
            return [d[a:b].decode(c, "replace") for a, b in zip(self.s[k].tolist(), self.e[k].tolist())]
        return d[int(self.s[k]):int(self.e[k])].decode(c, "replace")


def _index(data: bytes, codec: str) -> Tuple[np.ndarray, _Values]:
    """(the group codes, their values) of the text of a DXF file: pairs of lines."""
    arr = np.frombuffer(data, np.uint8)
    k = len(data.rstrip())
    off = np.int32 if len(data) < 2_000_000_000 else np.int64  # (the offsets of ten million lines: kept small)
    nl = np.flatnonzero(arr[:k] == 10).astype(off)
    L = len(nl) + 1
    if L < 4:
        raise DxfError("This is not a DXF file (it is empty).")
    starts = np.empty(L, off)
    starts[0] = 0
    starts[1:] = nl + 1
    ends = np.empty(L, off)
    ends[:-1] = nl
    ends[-1] = k
    del nl
    n = min(len(range(0, L - 1, 2)), len(range(1, L, 2)))
    cs, ce = starts[0:L - 1:2][:n].copy(), ends[0:L - 1:2][:n].copy()
    vs, ve = starts[1::2][:n].copy(), ends[1::2][:n].copy()
    del starts, ends
    bad = DxfError("This is not an ASCII DXF file (a group code is not a number).")
    width = int((ce - cs).max())
    if width > 24 or width < 1:
        raise bad
    val = np.zeros(n, np.int32)
    last = len(arr) - 1
    for i in range(0, n, 1 << 20):  # (a million lines at a time)
        a0, a1 = cs[i:i + (1 << 20)].astype(np.int64), ce[i:i + (1 << 20)].astype(np.int64)
        v = np.zeros(len(a0), np.int32)
        neg = np.zeros(len(a0), bool)
        seen = np.zeros(len(a0), bool)
        for j in range(int((a1 - a0).max())):
            at = a0 + j
            ch = arr[np.minimum(at, last)]
            ch[at >= a1] = 32
            digit = (ch >= 48) & (ch <= 57)
            minus = ch == 45
            if not (digit | minus | (ch == 32) | (ch == 9) | (ch == 13) | (ch == 43)).all():
                raise bad
            v[digit] = v[digit] * 10 + (ch[digit].astype(np.int32) - 48)
            neg |= minus
            seen |= digit
        if not seen.all():
            raise bad
        v[neg] = -v[neg]
        val[i:i + (1 << 20)] = v
    del cs, ce
    return val, _Values(data, vs, ve, codec)


def _ocs(n: Sequence[float]) -> Optional[np.ndarray]:
    """The 2x2 matrix taking (x, y) of an entity's own coordinate system to world X, Y
    (arbitrary axis algorithm); None for the usual extrusion (0, 0, 1)."""
    nx, ny, nz = (float(v) for v in n)
    L = math.sqrt(nx * nx + ny * ny + nz * nz)
    if L < 1e-12:
        return None
    nx, ny, nz = nx / L, ny / L, nz / L
    if abs(nx) < 1e-9 and abs(ny) < 1e-9 and nz > 0:
        return None
    N = np.array([nx, ny, nz])
    ax = np.cross((0.0, 1.0, 0.0), N) if abs(nx) < 1 / 64 and abs(ny) < 1 / 64 else np.cross((0.0, 0.0, 1.0), N)
    ax = ax / np.linalg.norm(ax)
    ay = np.cross(N, ax)
    ay = ay / np.linalg.norm(ay)
    return np.array([[ax[0], ay[0]], [ax[1], ay[1]]])


def _arc_points(cx: float, cy: float, r: float, a0: float, sweep: float) -> np.ndarray:
    """Points of an arc from angle a0 over ``sweep`` (radians, signed)."""
    n = max(2, int(math.ceil(abs(math.degrees(sweep)) / CHORD_DEG)))
    t = a0 + np.linspace(0.0, sweep, n + 1)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _bulge_points(p: np.ndarray, q: np.ndarray, bulge: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The arc from p to q of a polyline bulge (= tan of a quarter of the angle; + = to
    the left).  Returns (points, centre)."""
    c = q - p
    d = float(np.hypot(c[0], c[1]))
    if d < 1e-14 or abs(bulge) < 1e-12:
        return np.array([p, q]), None
    theta = 4.0 * math.atan(bulge)
    h = 0.5 * d * (1.0 - bulge * bulge) / (2.0 * bulge)
    nrm = np.array([-c[1], c[0]]) / d
    cen = 0.5 * (p + q) + nrm * h
    r = float(np.hypot(*(p - cen)))
    a0 = math.atan2(p[1] - cen[1], p[0] - cen[0])
    pts = _arc_points(cen[0], cen[1], r, a0, theta)
    pts[0], pts[-1] = p, q
    return pts, cen


def _spline_points(degree: int, knots: List[float], ctrl: np.ndarray, weights: Optional[np.ndarray],
                   fit: np.ndarray) -> np.ndarray:
    n = len(ctrl)
    if n < 2 or degree < 1 or len(knots) < n + degree + 1:
        return _through_points(fit) if len(fit) >= 2 else ctrl
    k = np.asarray(knots[:n + degree + 1], float)
    u0, u1 = k[degree], k[n]
    if not u1 > u0:
        return ctrl
    m = int(min(max(16, 8 * n), 2000))
    u = np.linspace(u0, u1, m + 1)
    u[-1] = u1 - (u1 - u0) * 1e-12
    # Cox - de Boor, all sample points at once
    B = ((u[:, None] >= k[None, :-1]) & (u[:, None] < k[None, 1:])).astype(float)
    for p in range(1, degree + 1):
        left_den = k[p:-1] - k[:-p - 1]
        right_den = k[p + 1:] - k[1:-p]
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(left_den > 0, (u[:, None] - k[None, :-p - 1]) / left_den, 0.0)
            right = np.where(right_den > 0, (k[None, p + 1:] - u[:, None]) / right_den, 0.0)
        B = left * B[:, :-1] + right * B[:, 1:]
    B = B[:, :n]
    if weights is not None and len(weights) == n:
        B = B * weights[None, :]
    s = B.sum(axis=1)
    s[s == 0] = 1.0
    return (B @ ctrl) / s[:, None]


def _through_points(fit: np.ndarray) -> np.ndarray:
    """A smooth curve through the fit points of a spline saved without control points
    (a cubic spline over the chord lengths, as CAD programs draw it)."""
    fit = np.asarray(fit, float)
    keep = np.r_[True, np.hypot(*np.diff(fit, axis=0).T) > 0]
    fit = fit[keep]
    if len(fit) < 3:
        return fit
    try:
        from scipy.interpolate import CubicSpline
    except Exception:  # (no SciPy: the points joined by straight lines)
        return fit
    t = np.r_[0.0, np.cumsum(np.hypot(*np.diff(fit, axis=0).T))]
    cs = CubicSpline(t, fit, bc_type="natural")
    return cs(np.linspace(0.0, t[-1], int(min(max(16, 12 * len(fit)), 2000)) + 1))


class _Group:
    """What a container (the model space or a block) holds, before blocks are put in."""

    __slots__ = ("seg", "curve", "layer", "color", "pts", "pkind", "player", "inserts", "base", "flat", "texts",
                 "tflat")

    def __init__(self):
        self.seg: List[np.ndarray] = []
        self.curve: List[np.ndarray] = []
        self.layer: List[np.ndarray] = []
        self.color: List[np.ndarray] = []
        self.pts: List[Tuple[float, float]] = []
        self.pkind: List[int] = []
        self.player: List[int] = []
        self.inserts: List[tuple] = []
        self.base = (0.0, 0.0)
        self.flat = None
        self.texts: List[tuple] = []  # (x, y, height, angle rad, string, layer, colour, across, up, width factor)
        self.tflat = None

    def add(self, pts: np.ndarray, layer: int, color: int, curve, closed: bool = False):
        pts = np.asarray(pts, float).reshape(-1, 2)
        if closed and len(pts) >= 2:
            pts = np.vstack([pts, pts[:1]])
        if len(pts) < 2:
            return
        s = np.hstack([pts[:-1], pts[1:]])
        self.seg.append(s)
        n = len(s)
        cv = np.broadcast_to(np.asarray(curve, bool), (n,)) if np.ndim(curve) == 0 else np.asarray(curve, bool)
        self.curve.append(np.array(cv, bool))
        self.layer.append(np.full(n, layer, np.int32))
        self.color.append(np.full(n, color, np.int64))

    def point(self, xy, kind: int, layer: int):
        self.pts.append((float(xy[0]), float(xy[1])))
        self.pkind.append(kind)
        self.player.append(layer)


def _f(d: Dict[int, List[str]], code: int, default: float = 0.0) -> float:
    v = d.get(code)
    if not v:
        return default
    try:
        return float(v[0])
    except ValueError:
        return default


def _i(d: Dict[int, List[str]], code: int, default: int = 0) -> int:
    v = d.get(code)
    if not v:
        return default
    try:
        return int(float(v[0]))
    except ValueError:
        return default


def _fl(d: Dict[int, List[str]], code: int) -> List[float]:
    out = []
    for v in d.get(code, ()):
        try:
            out.append(float(v))
        except ValueError:
            out.append(0.0)
    return out


MAX_TEXTS = 60_000
_MTEXT_CODE = re.compile(r"\\[ACcFfHhQqTtWwpP][^;\\]*;|\\[LlOoKk]|[{}]")
_SKIPPED = {"ATTDEF", "HATCH", "IMAGE", "WIPEOUT", "VIEWPORT", "3DSOLID", "BODY", "REGION",
            "MESH", "SURFACE", "MLINE", "MLEADER", "MULTILEADER", "ACAD_TABLE", "TOLERANCE", "XLINE", "RAY", "SHAPE",
            "OLE2FRAME", "OLEFRAME", "ACAD_PROXY_ENTITY", "HELIX", "LIGHT", "SUN", "UNDERLAY", "PDFUNDERLAY",
            "DWFUNDERLAY", "DGNUNDERLAY", "SECTION", "ARC_DIMENSION", "LARGE_RADIAL_DIMENSION"}


def read_dxf(source) -> Drawing:
    """Read a DXF file (a path, or its bytes) into a :class:`Drawing`.

    Raises :class:`DxfError` with a message for the user when the file is not an ASCII
    DXF (binary DXF, DWG, something else) or holds more line work than can be shown."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        try:
            with open(source, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise DxfError(f"The file cannot be read: {exc}") from None
    data, codec = _text_bytes(data)
    codes, vals = _index(data, codec)
    n = len(codes)
    starts = np.flatnonzero(codes == 0)
    if not len(starts):
        raise DxfError("This is not a DXF file (no entities).")
    ends = np.append(starts[1:], n).tolist()
    starts = starts.tolist()

    dwg = Drawing()
    layer_ids: Dict[str, int] = {}
    layer_color: List[int] = []  # packed rgb per layer

    def layer_id(name: str) -> int:
        k = layer_ids.get(name)
        if k is None:
            k = layer_ids[name] = len(dwg.layers)
            dwg.layers.append(name)
            layer_color.append(0xFFFFFF)
        return k

    L0 = layer_id("0")
    blocks: Dict[str, _Group] = {}
    model = _Group()
    section = ""
    table = ""
    current: Optional[_Group] = None  # the block being read
    poly = None  # an old-style POLYLINE being read: (group, common, flags, vertices, bulges)
    skip_seq = False  # the attributes after an INSERT
    counts: Dict[str, int] = {}
    skipped: Dict[str, int] = {}

    def record(a: int, b: int) -> Dict[int, List[str]]:
        d: Dict[int, List[str]] = {}
        for c, v in zip(codes[a:b].tolist(), vals[a:b]):
            # (the spaces of a text belong to it: a long text is cut into pieces anywhere)
            d.setdefault(c, []).append(v.strip() if c != 1 and c != 3 else v.strip("\r\n"))
        return d

    def color_of(d) -> int:
        tc = d.get(420)
        if tc:
            try:
                return int(tc[0]) & 0xFFFFFF
            except ValueError:
                pass
        c = _i(d, 62, 256)
        if c == 256:
            return BYLAYER
        if c == 0:
            return BYBLOCK
        r, g, b = aci_rgb(abs(c))
        return (r << 16) | (g << 8) | b

    def plain(text: str) -> str:
        """The codes of CAD texts as characters (%%d, \\U+0E01 ...)."""
        if "%%" in text:
            for code, ch in (("%%d", "\u00b0"), ("%%D", "\u00b0"), ("%%p", "\u00b1"), ("%%P", "\u00b1"),
                             ("%%c", "\u2300"), ("%%C", "\u2300"), ("%%u", ""), ("%%U", ""), ("%%o", ""),
                             ("%%O", ""), ("%%%", "%")):
                text = text.replace(code, ch)
        if "\\U+" in text:
            text = re.sub(r"\\U\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), text)
        return text

    def add_text(g: _Group, d, lay: int, col: int, up_code: int):
        text = plain((d.get(1) or [""])[0]).strip()
        h = _f(d, 40)
        if not text or not h > 0:
            return
        across, up = _i(d, 72, 0), _i(d, up_code, 0)
        at = (_f(d, 10), _f(d, 20))
        if across in (3, 5):  # aligned / fit between two points: from the first
            across, up = 0, 0
        elif across == 4:  # middle
            across, up, at = 1, 2, (_f(d, 11, at[0]), _f(d, 21, at[1]))
        elif (across or up) and 11 in d:
            at = (_f(d, 11), _f(d, 21))
        xy = to_world(d, np.array([at]))[0]
        g.texts.append((float(xy[0]), float(xy[1]), h, math.radians(_f(d, 50)), text, lay, col,
                        min(max(across, 0), 2), min(max(up, 0), 3), _f(d, 41, 1.0) or 1.0))

    def add_mtext(g: _Group, d, lay: int, col: int):
        raw = "".join(d.get(3) or []) + (d.get(1) or [""])[0]
        h = _f(d, 40)
        if not raw.strip() or not h > 0:
            return
        raw = re.sub(r"\\S([^;^/#]*)[\^/#]([^;]*);", r"\1/\2", raw)  # stacked: a/b
        raw = raw.replace("\\P", "\n").replace("\\~", " ").replace("\\\\", "\x00")
        raw = _MTEXT_CODE.sub("", raw).replace("\x00", "\\")
        lines = [plain(t).strip() for t in raw.split("\n")]
        if 11 in d or 21 in d:
            ang = math.atan2(_f(d, 21), _f(d, 11, 1.0))
        else:
            ang = _f(d, 50)  # (radians in an MTEXT)
        att = min(max(_i(d, 71, 1), 1), 9)
        across, row = (att - 1) % 3, (att - 1) // 3  # row: 0 top, 1 middle, 2 bottom
        step = 1.667 * h
        n = len(lines)
        first = {0: 0.0, 1: 0.5 * (n - 1) * step, 2: (n - 1) * step}[row]  # the first line above the point by this
        up = {0: 3, 1: 2, 2: 0}[row]
        x0, y0 = to_world(d, np.array([[_f(d, 10), _f(d, 20)]]))[0]
        for k, text in enumerate(lines):
            if not text:
                continue
            off = first - k * step
            g.texts.append((float(x0 - math.sin(ang) * off), float(y0 + math.cos(ang) * off), h, ang, text, lay, col,
                            across, up, 1.0))

    def to_world(d, pts: np.ndarray, z: float = 0.0) -> np.ndarray:
        if 210 not in d and 230 not in d:
            return pts
        W = _ocs((_f(d, 210, 0.0), _f(d, 220, 0.0), _f(d, 230, 1.0)))
        return pts if W is None else pts @ W.T

    def polyline(g: _Group, d, verts: np.ndarray, bulges: Sequence[float], closed: bool, lay: int, col: int,
                 smooth: bool = False):
        # smooth: the vertices are points along a curve (a spline-fit polyline), its chords no straight edges
        if len(verts) < 2:
            return
        verts = to_world(d, verts)
        mirror = False
        if 210 in d or 230 in d:
            W = _ocs((_f(d, 210, 0.0), _f(d, 220, 0.0), _f(d, 230, 1.0)))
            mirror = W is not None and float(np.linalg.det(W)) < 0
        m = len(verts)
        last = m if closed else m - 1
        pts: List[np.ndarray] = []
        curve: List[np.ndarray] = []
        for k in range(last):
            p, q = verts[k], verts[(k + 1) % m]
            b = bulges[k] if k < len(bulges) else 0.0
            if mirror:
                b = -b
            if abs(b) > 1e-12:
                arc, cen = _bulge_points(p, q, b)
                if cen is not None:
                    g.point(cen, PT_CENTER, lay)
                    g.point(p, PT_END, lay)
                    g.point(q, PT_END, lay)
                pts.append(arc[:-1])
                curve.append(np.ones(len(arc) - 1, bool))
            else:
                pts.append(p[None, :])
                curve.append(np.zeros(1, bool))
        if not pts:
            return
        P = np.vstack(pts + [verts[last % m][None, :]])
        cv = np.concatenate(curve)
        if smooth:
            cv[:] = True
            if not closed:
                g.point(verts[0], PT_END, lay)
                g.point(verts[-1], PT_END, lay)
        g.add(P, lay, col, cv)

    for a, b in zip(starts, ends):
        etype = vals[a].strip().upper()
        if etype == "SECTION":
            d = record(a + 1, b)
            section = (d.get(2) or [""])[0].upper()
            if section == "HEADER":
                # the variables: (9, name) then its values
                name = ""
                for c, v in zip(codes[a + 1:b].tolist(), vals[a + 1:b]):
                    v = v.strip()
                    if c == 9:
                        name = v
                    elif name == "$INSUNITS" and c == 70:
                        try:
                            dwg.unit = INSUNITS.get(int(v))
                        except ValueError:
                            pass
                    elif name == "$MEASUREMENT" and c == 70:
                        dwg.metric = v != "0"
                    elif name == "$ACADVER" and c == 1:
                        dwg.version = v
            continue
        if etype == "ENDSEC":
            section, table = "", ""
            continue
        if etype == "EOF":
            break
        if section == "TABLES":
            if etype == "TABLE":
                table = (record(a + 1, b).get(2) or [""])[0].upper()
            elif etype == "ENDTAB":
                table = ""
            elif etype == "LAYER" and table == "LAYER":
                d = record(a + 1, b)
                name = (d.get(2) or [""])[0]
                if not name:
                    continue
                k = layer_id(name)
                c = _i(d, 62, 7)
                tc = d.get(420)
                if tc:
                    try:
                        layer_color[k] = int(tc[0]) & 0xFFFFFF
                    except ValueError:
                        tc = None
                if not tc:
                    r, g_, b_ = aci_rgb(abs(c) or 7)
                    layer_color[k] = (r << 16) | (g_ << 8) | b_
                if c < 0 or _i(d, 70, 0) & 1:
                    dwg.layers_off.append(name)
            continue
        if section == "BLOCKS":
            if etype == "BLOCK":
                d = record(a + 1, b)
                name = (d.get(2) or d.get(3) or [""])[0].strip()
                current = _Group()
                current.base = (_f(d, 10), _f(d, 20))
                blocks[name.upper()] = current
                continue
            if etype == "ENDBLK":
                current = None
                poly = None
                continue
            g = current
            if g is None:
                continue
        elif section == "ENTITIES":
            g = model
        else:
            continue

        # ---- an entity of the model space or of a block
        if etype == "VERTEX":
            if poly is not None:
                d = record(a + 1, b)
                vf = _i(d, 70, 0)
                if vf & 16:
                    pass  # a control point of the frame of a spline-fit polyline: not on the curve
                elif not vf & 128 or vf & 64:  # (not a face record of a polyface mesh)
                    poly[3].append((_f(d, 10), _f(d, 20)))
                    poly[4].append(_f(d, 42))
            continue
        if etype == "SEQEND":
            if poly is not None:
                pg, pd, flags, verts, bulges = poly
                poly = None
                if not flags & (16 | 64):  # (a polygon / polyface mesh is not line work)
                    lay = layer_id((pd.get(8) or ["0"])[0])
                    polyline(pg, pd if not flags & 8 else {}, np.array(verts, float).reshape(-1, 2), bulges,
                             bool(flags & 1), lay, color_of(pd), smooth=bool(flags & 4))
            skip_seq = False
            continue
        if skip_seq:
            if etype == "ATTRIB":  # the text an insert carries (a grid bubble's letter)
                d = record(a + 1, b)
                if _i(d, 67, 0) != 1 and not _i(d, 70, 0) & 1:
                    add_text(g, d, layer_id((d.get(8) or ["0"])[0]), color_of(d), 74)
            continue
        if etype in _SKIPPED:  # (not read at all: the hatches can be most of a file)
            if not ((codes[a + 1:b] == 67).any() and _i(record(a + 1, b), 67, 0) == 1):
                skipped[etype] = skipped.get(etype, 0) + 1
            continue
        d = record(a + 1, b)
        if _i(d, 67, 0) == 1:
            continue  # paper space
        counts[etype] = counts.get(etype, 0) + 1
        lay = layer_id((d.get(8) or ["0"])[0])
        col = color_of(d)
        if etype == "LINE":
            g.add(np.array([[_f(d, 10), _f(d, 20)], [_f(d, 11), _f(d, 21)]]), lay, col, False)
        elif etype == "LWPOLYLINE":
            xs, ys = _fl(d, 10), _fl(d, 20)
            m = min(len(xs), len(ys))
            if m >= 2:
                # the bulge (42) of a vertex follows its 10 / 20: read them in order
                bulges = [0.0] * m
                k = -1
                for c, v in zip(codes[a + 1:b].tolist(), vals[a + 1:b]):
                    if c == 10:
                        k += 1
                    elif c == 42 and 0 <= k < m:
                        try:
                            bulges[k] = float(v)
                        except ValueError:
                            pass
                polyline(g, d, np.column_stack([xs[:m], ys[:m]]), bulges, bool(_i(d, 70, 0) & 1), lay, col)
        elif etype == "POLYLINE":
            poly = (g, d, _i(d, 70, 0), [], [])
        elif etype in ("ARC", "CIRCLE"):
            r = _f(d, 40)
            if r > 0:
                cx, cy = _f(d, 10), _f(d, 20)
                if etype == "CIRCLE":
                    a0, sweep = 0.0, 2 * math.pi
                else:
                    a0 = math.radians(_f(d, 50))
                    sweep = (math.radians(_f(d, 51)) - a0) % (2 * math.pi)
                    if sweep < 1e-12:
                        sweep = 2 * math.pi
                P = to_world(d, _arc_points(cx, cy, r, a0, sweep))
                g.add(P, lay, col, True)
                g.point(to_world(d, np.array([[cx, cy]]))[0], PT_CENTER, lay)
                if etype == "ARC":
                    g.point(P[0], PT_END, lay)
                    g.point(P[-1], PT_END, lay)
        elif etype == "ELLIPSE":
            cx, cy = _f(d, 10), _f(d, 20)
            mx, my = _f(d, 11), _f(d, 21)
            ratio = _f(d, 40, 1.0)
            t0, t1 = _f(d, 41, 0.0), _f(d, 42, 2 * math.pi)
            sweep = (t1 - t0) % (2 * math.pi)
            if sweep < 1e-12:
                sweep = 2 * math.pi
            nz = _f(d, 230, 1.0)
            sgn = -1.0 if nz < 0 else 1.0  # (the minor axis is normal x major)
            nx_, ny_ = -my * ratio * sgn, mx * ratio * sgn
            nseg = max(2, int(math.ceil(math.degrees(sweep) / CHORD_DEG)))
            t = t0 + np.linspace(0.0, sweep, nseg + 1)
            P = np.column_stack([cx + mx * np.cos(t) + nx_ * np.sin(t), cy + my * np.cos(t) + ny_ * np.sin(t)])
            g.add(P, lay, col, True)
            g.point((cx, cy), PT_CENTER, lay)
            if sweep < 2 * math.pi - 1e-9:
                g.point(P[0], PT_END, lay)
                g.point(P[-1], PT_END, lay)
        elif etype == "SPLINE":
            ctrl = np.column_stack([_fl(d, 10), _fl(d, 20)]) if d.get(10) and len(d[10]) == len(d.get(20, ())) else \
                np.zeros((0, 2))
            fit = np.column_stack([_fl(d, 11), _fl(d, 21)]) if d.get(11) and len(d[11]) == len(d.get(21, ())) else \
                np.zeros((0, 2))
            w = np.array(_fl(d, 41), float) if d.get(41) else None
            P = _spline_points(_i(d, 71, 3), _fl(d, 40), ctrl, w, fit)
            if len(P) >= 2:
                g.add(P, lay, col, True)
                g.point(P[0], PT_END, lay)
                g.point(P[-1], PT_END, lay)
        elif etype in ("SOLID", "TRACE", "3DFACE"):
            P = [(_f(d, 10), _f(d, 20)), (_f(d, 11), _f(d, 21))]
            third, fourth = (_f(d, 12), _f(d, 22)), (_f(d, 13, _f(d, 12)), _f(d, 23, _f(d, 22)))
            P += [third, fourth] if etype == "3DFACE" else [fourth, third]
            P = np.array(P, float)
            keep = [0] + [k for k in range(1, len(P)) if np.hypot(*(P[k] - P[k - 1])) > 1e-12]
            P = P[keep]
            if etype != "3DFACE":
                P = to_world(d, P)
            g.add(P, lay, col, False, closed=len(P) > 2)
        elif etype == "LEADER":
            xs, ys = _fl(d, 10), _fl(d, 20)
            m = min(len(xs), len(ys))
            if m >= 2:
                g.add(np.column_stack([xs[:m], ys[:m]]), lay, col, False)
        elif etype == "POINT":
            g.point((_f(d, 10), _f(d, 20)), PT_POINT, lay)
        elif etype in ("INSERT", "DIMENSION"):
            name = (d.get(2) or [""])[0].upper()
            if etype == "DIMENSION":
                if name:
                    g.inserts.append((name, 0.0, 0.0, 1.0, 1.0, 0.0, 1, 1, 0.0, 0.0, lay, col, None, False))
                continue
            W = _ocs((_f(d, 210, 0.0), _f(d, 220, 0.0), _f(d, 230, 1.0))) if (210 in d or 230 in d) else None
            g.inserts.append((name, _f(d, 10), _f(d, 20), _f(d, 41, 1.0), _f(d, 42, 1.0), math.radians(_f(d, 50)),
                              max(_i(d, 71, 1), 1), max(_i(d, 70, 1), 1), _f(d, 45), _f(d, 44), lay, col, W, True))
            if _i(d, 66, 0) == 1:
                skip_seq = True  # its attributes, up to SEQEND
        elif etype in ("TEXT", "ATTRIB"):
            add_text(g, d, lay, col, 73 if etype == "TEXT" else 74)
        elif etype == "MTEXT":
            add_mtext(g, d, lay, col)
        elif etype in _SKIPPED:
            skipped[etype] = skipped.get(etype, 0) + 1
            counts[etype] -= 1
        else:
            skipped[etype] = skipped.get(etype, 0) + 1
            counts[etype] -= 1

    # ---- blocks put in place
    def flatten(g: _Group, depth: int, chain: Tuple[int, ...]):
        """(seg, curve, layer, colour, pts, pkind, player) of a container with its blocks."""
        if g.flat is not None:
            return g.flat
        seg = [np.vstack(g.seg)] if g.seg else []
        curve = [np.concatenate(g.curve)] if g.curve else []
        layer = [np.concatenate(g.layer)] if g.layer else []
        color = [np.concatenate(g.color)] if g.color else []
        pts = [np.array(g.pts, float).reshape(-1, 2)] if g.pts else []
        pkind = [np.array(g.pkind, np.uint8)] if g.pkind else []
        player = [np.array(g.player, np.int32)] if g.player else []
        held = len(seg[0]) if seg else 0  # the segments of this container so far (each counted once)
        for (name, ix, iy, sx, sy, rot, rows, cols, dr, dc, ilay, icol, W, mark) in g.inserts:
            blk = blocks.get(name)
            if blk is None or depth >= MAX_DEPTH or id(blk) in chain:
                continue
            bs, bc, bl, bcol, bp, bk, bpl = flatten(blk, depth + 1, chain + (id(blk),))
            cr, sr = math.cos(rot), math.sin(rot)
            A = np.array([[cr * sx, -sr * sy], [sr * sx, cr * sy]])
            base = np.array(blk.base, float)
            for r_ in range(rows):
                for c_ in range(cols):
                    off = np.array([ix, iy]) + np.array([cr * c_ * dc - sr * r_ * dr, sr * c_ * dc + cr * r_ * dr])
                    M, t = A, off - A @ base
                    if W is not None:
                        M, t = W @ A, W @ t
                    if len(bs):
                        held += len(bs)
                        if held > MAX_SEGMENTS:
                            raise DxfError("The drawing holds more line work than can be shown as a reference "
                                           f"(over {MAX_SEGMENTS:,} segments). Save a simpler copy (fewer layers, "
                                           "no hatches exploded into lines) and try again.")
                        P1 = bs[:, 0:2] @ M.T + t
                        P2 = bs[:, 2:4] @ M.T + t
                        seg.append(np.hstack([P1, P2]))
                        curve.append(bc)
                        layer.append(np.where(bl == L0, ilay, bl).astype(np.int32))
                        cc = bcol.copy()
                        if icol == BYLAYER:
                            # BYBLOCK takes the colour of the insert: that of the insert's layer
                            cc[cc == BYBLOCK] = _LAYER_OF - ilay
                        else:
                            cc[cc == BYBLOCK] = icol
                        cc[cc == _LAYER_OF - L0] = _LAYER_OF - ilay  # (layer "0" in a block: the insert's)
                        color.append(cc)
                    if len(bp):
                        pts.append(bp @ M.T + t)
                        pkind.append(bk)
                        player.append(np.where(bpl == L0, ilay, bpl).astype(np.int32))
                    if mark:
                        pts.append((off if W is None else W @ off)[None, :])
                        pkind.append(np.array([PT_INSERT], np.uint8))
                        player.append(np.array([ilay], np.int32))
        g.flat = (np.vstack(seg) if seg else np.zeros((0, 4)),
                  np.concatenate(curve) if curve else np.zeros(0, bool),
                  np.concatenate(layer) if layer else np.zeros(0, np.int32),
                  np.concatenate(color) if color else np.zeros(0, np.int64),
                  np.vstack(pts) if pts else np.zeros((0, 2)),
                  np.concatenate(pkind) if pkind else np.zeros(0, np.uint8),
                  np.concatenate(player) if player else np.zeros(0, np.int32))
        return g.flat

    def flat_texts(g: _Group, depth: int, chain: Tuple[int, ...]) -> List[tuple]:
        """The texts of a container with those of its blocks, put in place."""
        if g.tflat is not None:
            return g.tflat
        out = list(g.texts)
        for (name, ix, iy, sx, sy, rot, rows, cols, dr, dc, ilay, icol, W, mark) in g.inserts:
            blk = blocks.get(name)
            if blk is None or depth >= MAX_DEPTH or id(blk) in chain:
                continue
            inner = flat_texts(blk, depth + 1, chain + (id(blk),))
            if not inner or len(out) > MAX_TEXTS:
                continue
            cr, sr = math.cos(rot), math.sin(rot)
            A = np.array([[cr * sx, -sr * sy], [sr * sx, cr * sy]])
            base = np.array(blk.base, float)
            for r_ in range(rows):
                for c_ in range(cols):
                    off = np.array([ix, iy]) + np.array([cr * c_ * dc - sr * r_ * dr, sr * c_ * dc + cr * r_ * dr])
                    M, t = A, off - A @ base
                    if W is not None:
                        M, t = W @ A, W @ t
                    for (x, y, h, ang, text, lay, col, across, up, wf) in inner:
                        p = M @ np.array([x, y]) + t
                        u = M @ np.array([math.cos(ang), math.sin(ang)])
                        v = M @ np.array([-math.sin(ang), math.cos(ang)])
                        lu, lv = float(np.hypot(*u)), float(np.hypot(*v))
                        if not (lu > 0 and lv > 0):
                            continue
                        if col == BYBLOCK:
                            col2 = (_LAYER_OF - ilay) if icol == BYLAYER else icol
                        elif col == _LAYER_OF - L0:
                            col2 = _LAYER_OF - ilay
                        elif col == BYLAYER and lay == L0:
                            col2 = _LAYER_OF - ilay
                        else:
                            col2 = col
                        out.append((float(p[0]), float(p[1]), h * lv, math.atan2(u[1], u[0]), text,
                                    ilay if lay == L0 else lay, col2, across, up, wf * lu / lv))
        g.tflat = out
        return out

    seg, curve, layer, color, pts, pkind, player = flatten(model, 0, ())
    if len(seg) > MAX_SEGMENTS:
        raise DxfError(f"The drawing holds more line work than can be shown as a reference (over {MAX_SEGMENTS:,} "
                       "segments). Save a simpler copy (fewer layers) and try again.")
    ok = np.all(np.isfinite(seg), axis=1) if len(seg) else np.zeros(0, bool)
    # (zero-length pieces draw nothing and snap to nothing)
    if len(seg):
        ok &= np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) > 0.0
    seg, curve, layer, color = seg[ok], curve[ok], layer[ok], color[ok]
    lay_rgb = np.array(layer_color, np.int64)
    # colours: BYLAYER -> the layer's; "of layer k"; BYBLOCK left at the top -> white (7)
    final = color.copy()
    by = final == BYLAYER
    final[by] = lay_rgb[layer[by]]
    of = final <= _LAYER_OF
    final[of] = lay_rgb[np.clip(_LAYER_OF - final[of], 0, len(lay_rgb) - 1)]
    final[final == BYBLOCK] = 0xFFFFFF
    dwg.seg = np.ascontiguousarray(seg, float)
    dwg.seg_curve = curve.astype(bool)
    dwg.seg_layer = layer.astype(np.int32)
    dwg.seg_rgb = np.column_stack([(final >> 16) & 255, (final >> 8) & 255, final & 255]).astype(np.uint8)
    okp = np.all(np.isfinite(pts), axis=1) if len(pts) else np.zeros(0, bool)
    dwg.pts, dwg.pts_kind, dwg.pts_layer = pts[okp], pkind[okp], player[okp]
    dwg.layer_rgb = [((c >> 16) & 255, (c >> 8) & 255, c & 255) for c in layer_color]
    def rgb_of(col: int, lay: int) -> Tuple[int, int, int]:
        if col == BYLAYER:
            col = layer_color[lay]
        elif col <= _LAYER_OF:
            col = layer_color[min(max(_LAYER_OF - col, 0), len(layer_color) - 1)]
        elif col == BYBLOCK:
            col = 0xFFFFFF
        return ((col >> 16) & 255, (col >> 8) & 255, col & 255)

    for (x, y, h, ang, text, lay, col, across, up, wf) in flat_texts(model, 0, ())[:MAX_TEXTS]:
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(h) and h > 0:
            dwg.texts.append((x, y, h, math.degrees(ang), text, int(lay), rgb_of(col, lay), across, up, 0.0, wf))
    dwg.counts = {k: v for k, v in counts.items() if v > 0}
    dwg.skipped = skipped
    return dwg
