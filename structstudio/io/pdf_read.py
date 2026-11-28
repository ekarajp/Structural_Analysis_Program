"""A page of a PDF drawing read as line work, to place as a reference to draw over.

Only *vector* PDFs give line work: a drawing plotted / exported from a CAD program (its
lines are lines in the file).  A scanned sheet is a picture - it has nothing to snap to
and is refused with a message.

What is read from the page's content (and the forms it draws, nested):

* stroked paths - lines, rectangles and Bezier curves (as chords, like the arcs of a DXF
  file), with the colour and the pen width they are stroked with;
* small filled shapes without curves (a column drawn solid) as their outlines, on a
  layer of their own; larger fills (hatches, letters turned into shapes) are counted
  and left out, as are pictures and shadings;
* the texts written with fonts (where the file says which characters they are): their
  string, place, size and direction - the program draws them with a font of its own, they
  are never snapped to.  (Texts a CAD program plots as strokes are line work already.)
  Text written invisibly is read too: programs that turn letters into filled shapes
  (left out here) keep the words that way;
* what is covered by a white shape drawn after it (a mask behind a title or a stamp) is
  left out, as on the sheet - a line that only crosses its edge keeps the part outside;
* the clipping paths of the page (a hatch drawn as long lines clipped to its area, a
  viewport) - only what is inside them is on the sheet;
* hatch patterns (many parallel lines at an even, close spacing) are found and put on
  layers of their own, off at first: what is drawn over is the outline, not the pattern;
* texts written with fonts whose codes are no characters (no ToUnicode table): read from
  the font's own table of characters (TrueType), else drawn as the outlines of its glyphs
  (TrueType, Type 3) - seen as on the sheet, though not readable as text;
* the layers of the file (PDF "optional content", as CAD programs write them when
  plotting with layers) - their names, and which are off.  A PDF without layers gets
  one layer per pen (width and colour), so thick and thin lines can be shown apart.

The result is a :class:`~structstudio.io.dxf_read.Drawing` in **millimetres on the
sheet**, the origin at the lower left corner of the page as it is viewed (the page's
rotation applied).  A sheet plotted 1:100 is placed with scale 100.

The file itself is opened with *pypdf* (pure Python); without it a PDF cannot be read
and the message says so.
"""
from __future__ import annotations

import io
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dxf_read import CHORD_DEG, MAX_SEGMENTS, PT_CENTER, PT_END, UNIT_TO_M, Drawing, DxfError

PT_TO_MM = 25.4 / 72.0
MAX_DEPTH = 12  # forms drawn in forms
FILL_MAX_VERTICES = 16  # a filled shape of straight edges up to this size is outlined
MAX_PEN_LAYERS = 24
HATCH_SPACING = 4.0  # mm on the sheet: slanted lines this close, evenly spaced, can be a hatch pattern
HATCH_SPACING_SQUARE = 2.0  # ... lines along the sheet's edges this close (further apart: the rows of a table)
HATCH_LINES = 8  # ... and at least this many of them
MAX_TEXTS = 60_000
MAX_MASKS = 400  # white shapes looked at for the lines they cover
CAP_HEIGHT = 0.72  # the height of capital letters in a font of size 1
NO_LAYER = "(no layer)"

_WS = rb"\x00\t\n\x0c\r "
_DELIM = rb"()<>\[\]{}/%"
_STR = rb"[^()\\]|\\[\s\S]"
_TOKEN = re.compile(
    rb"(?P<c>%[^\r\n]*)"
    rb"|(?P<s>\((?:" + _STR + rb"|\((?:" + _STR + rb"|\((?:" + _STR + rb")*\))*\))*\))"
    rb"|(?P<d><<|>>|[\[\]{}])"
    rb"|(?P<h><[0-9A-Fa-f" + _WS + rb"]*>)"
    rb"|(?P<n>/[^" + _WS + _DELIM + rb"]*)"
    rb"|(?P<r>[^" + _WS + _DELIM + rb"]+)")
_INLINE_IMAGE = re.compile(rb"(?<![^" + _WS + rb"])BI[" + _WS + rb"].*?[" + _WS + rb"]ID[" + _WS + rb"].*?["
                           + _WS + rb"]EI(?![^" + _WS + rb"])", re.S)

_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_STROKE = frozenset((b"S", b"s", b"B", b"B*", b"b", b"b*"))
_FILL = frozenset((b"f", b"F", b"f*"))
_CLOSING = frozenset((b"s", b"b", b"b*"))


def is_pdf(data: bytes) -> bool:
    """Do these bytes start a PDF file?"""
    return b"%PDF-" in bytes(data[:1024])


def _mul(m, n):
    """Matrix m applied first, then n (PDF: [x y 1] m n)."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + b * C, a * B + b * D, c * A + d * C, c * B + d * D, e * A + f * C + E, e * B + f * D + F)


def _pack(r: float, g: float, b: float) -> int:
    q = lambda v: 0 if not v > 0 else 255 if v >= 1 else int(round(v * 255))  # noqa: E731
    return (q(r) << 16) | (q(g) << 8) | q(b)


def _cmyk(c, m, y, k) -> int:
    return _pack((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))


def _turn(u, v) -> float:
    return abs(math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1]))


def _bezier(p0, p1, p2, p3) -> List[Tuple[float, float]]:
    """The points after ``p0`` of a cubic Bezier curve as chords: one per CHORD_DEG of
    turning (one chord = it is straight)."""
    legs = [(q[0] - p[0], q[1] - p[1]) for p, q in ((p0, p1), (p1, p2), (p2, p3))]
    legs = [v for v in legs if v[0] or v[1]]
    turn = sum(_turn(a, b) for a, b in zip(legs, legs[1:]))
    n = int(math.ceil(math.degrees(turn) / CHORD_DEG - 1e-9))
    if n <= 1 and math.degrees(turn) < 1.0:
        return [p3]
    n = max(2, min(n, 48))
    out = []
    for k in range(1, n):
        t = k / n
        s = 1.0 - t
        a, b, c, d = s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    out.append(p3)
    return out


def _get(d, key):
    """d[key] with its reference followed; None when it is not there / not readable."""
    try:
        v = d[key]
        return v.get_object() if hasattr(v, "get_object") else v
    except Exception:
        return None


def _name(raw: bytes) -> str:
    """A name token (/Layer#201) as the key of a dictionary."""
    if b"#" in raw:
        raw = re.sub(rb"#([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]), raw)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


_ESC = re.compile(rb"\\([nrtbf()\\]|[0-7]{1,3}|\r\n|\r|\n)")
_ESC_OF = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}
_GLYPHS = {"space": " ", "period": ".", "comma": ",", "hyphen": "-", "minus": "-", "slash": "/", "colon": ":",
           "semicolon": ";", "parenleft": "(", "parenright": ")", "plus": "+", "equal": "=", "degree": "\u00b0",
           "multiply": "\u00d7", "quotesingle": "'", "quotedbl": '"', "underscore": "_", "at": "@", "percent": "%",
           "ampersand": "&", "less": "<", "greater": ">", "bracketleft": "[", "bracketright": "]",
           "numbersign": "#", "asterisk": "*", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
           "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "plusminus": "\u00b1"}


def _string_bytes(tok) -> bytes:
    """The bytes of a string token: ("s", b"(...)") or ("h", b"<...>")."""
    kind, raw = tok[0], tok[1]
    if kind == "c":  # (already bytes: read through a font program)
        return raw
    if kind == "h":
        hx = re.sub(rb"[^0-9A-Fa-f]", b"", raw)
        return bytes.fromhex((hx + b"0" * (len(hx) % 2)).decode("ascii"))
    body = raw[1:-1]
    if b"\\" not in body:
        return body

    def one(m):
        g = m.group(1)
        if g[:1] in b"01234567":
            return bytes([int(g, 8) & 255])
        return _ESC_OF.get(g, b"")
    return _ESC.sub(one, body)


def _u16(hx: bytes) -> str:
    hx = re.sub(rb"[^0-9A-Fa-f]", b"", hx)
    try:
        return bytes.fromhex((hx + b"0" * (len(hx) % 2)).decode("ascii")).decode("utf-16-be", "replace")
    except ValueError:
        return ""


def _to_unicode(data: bytes) -> Dict[int, str]:
    """The character of each code, from a font's ToUnicode table."""
    out: Dict[int, str] = {}
    for blk in re.findall(rb"beginbfchar(.*?)endbfchar", data, re.S):
        for src, dst in re.findall(rb"<([0-9A-Fa-f\s]*)>\s*<([0-9A-Fa-f\s]*)>", blk):
            try:
                out[int(src, 16)] = _u16(dst)
            except ValueError:
                pass
    for blk in re.findall(rb"beginbfrange(.*?)endbfrange", data, re.S):
        for lo, hi, rest in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[^\]]*\]|<[0-9A-Fa-f\s]*>)", blk):
            a, b = int(lo, 16), int(hi, 16)
            if not 0 <= b - a <= 70000:
                continue
            if rest[:1] == b"[":
                for k, it in enumerate(re.findall(rb"<([0-9A-Fa-f\s]*)>", rest)[:b - a + 1]):
                    out[a + k] = _u16(it)
            else:
                first = _u16(rest[1:-1])
                if first:
                    for k in range(b - a + 1):
                        try:
                            out[a + k] = first[:-1] + chr(ord(first[-1]) + k)
                        except ValueError:
                            break
    return out


def _text(v) -> str:
    if isinstance(v, bytes):
        return v.decode("latin-1", "replace")
    return str(v) if v is not None else ""


class _Page:
    """The line work of one page, gathered while its content is interpreted."""

    def __init__(self, box=(0.0, 0.0, 0.0, 0.0)):
        self.box = box  # the page (x0, y0, x1, y1), points
        self.raw: List[float] = []  # x1, y1, x2, y2 ... in the space of their matrix
        self.curve: List[bool] = []
        self.runs: List[tuple] = []  # (segments, matrix id, layer, rgb, pen, filled)
        self.praw: List[float] = []  # snap points x, y ...
        self.pruns: List[tuple] = []  # (points, matrix id, layer, rgb, pen, filled, kind)
        self.ctm_id: Dict[tuple, int] = {}
        self.ctms: List[tuple] = []
        self.skipped: Dict[str, int] = {}
        self.strokes = 0
        self.has_layers = False
        self.texts: List[tuple] = []  # (x, y, ux, uy, size, length, string, layer, rgb) under their matrices
        self.clips: List[list] = []  # clipping paths: rings in page points
        self.chains: Dict[tuple, int] = {(): 0}
        self.chain_list: List[tuple] = [()]
        self.glyph = 0  # > 0: drawing the glyph of a Type 3 font (its paths are text, not line work)
        self.graw: List[np.ndarray] = []  # glyph outlines, page points (K, 4) each
        self.gmeta: List[tuple] = []  # (segments, layer, rgb, clip chain)
        self.masks: List[tuple] = []  # (segments drawn before it, polygon (K, 2)) of white shapes
        self._fonts: Dict[tuple, Optional[dict]] = {}
        self._keep: list = []  # (the objects the tables below are keyed by stay alive)
        self._data: Dict[int, bytes] = {}
        self._cs: Dict[tuple, tuple] = {}
        self._lw: Dict[tuple, Optional[float]] = {}
        self._oc: Dict[tuple, Optional[str]] = {}

    def skip(self, what: str, n: int = 1):
        self.skipped[what] = self.skipped.get(what, 0) + n

    # ------------------------------------------------------------- resources
    def layer_of(self, res, name: bytes) -> Optional[str]:
        """The layer (optional content group) a /Properties name stands for."""
        key = (id(res), name)
        if key not in self._oc:
            got = None
            props = _get(res, "/Properties") if res is not None else None
            oc = _get(props, _name(name)) if props is not None else None
            got = self.layer_name(oc)
            self._oc[key] = got
        return self._oc[key]

    def layer_name(self, oc) -> Optional[str]:
        if oc is None:
            return None
        if _text(_get(oc, "/Type")) == "/OCMD":
            g = _get(oc, "/OCGs")
            if isinstance(g, (list, tuple)):
                g = g[0].get_object() if g and hasattr(g[0], "get_object") else (g[0] if g else None)
            oc = g
            if oc is None:
                return None
        name = _get(oc, "/Name")
        if name is None:
            return None
        self.has_layers = True
        return _text(name) or NO_LAYER

    def colour_space(self, res, name: bytes) -> tuple:
        """("n",) plain components | ("sep",) a tint | ("idx", palette bytes, components)."""
        if name in (b"/DeviceGray", b"/DeviceRGB", b"/DeviceCMYK", b"/G", b"/RGB", b"/CMYK"):
            return ("n",)
        key = (id(res), name)
        if key not in self._cs:
            got: tuple = ("n",)
            try:
                cs = _get(_get(res, "/ColorSpace"), _name(name))
                kind = _text(cs[0]) if isinstance(cs, (list, tuple)) and cs else _text(cs)
                if kind in ("/Separation", "/DeviceN"):
                    got = ("sep",)
                elif kind == "/Indexed":
                    base = cs[1].get_object() if hasattr(cs[1], "get_object") else cs[1]
                    bk = _text(base[0]) if isinstance(base, (list, tuple)) else _text(base)
                    ncomp = 3
                    if bk == "/ICCBased":
                        ncomp = int(_get(base[1].get_object(), "/N") or 3)
                    elif bk in ("/DeviceGray", "/CalGray"):
                        ncomp = 1
                    elif bk == "/DeviceCMYK":
                        ncomp = 4
                    table = cs[3].get_object() if hasattr(cs[3], "get_object") else cs[3]
                    data = table.get_data() if hasattr(table, "get_data") else (
                        table if isinstance(table, bytes) else str(table).encode("latin-1", "replace"))
                    got = ("idx", bytes(data), ncomp)
            except Exception:
                got = ("n",)
            self._cs[key] = got
        return self._cs[key]

    def colour(self, cs: tuple, nums: List[float], old: int) -> int:
        try:
            if cs[0] == "sep":
                t = nums[-1] if nums else 1.0
                return _pack(1 - t, 1 - t, 1 - t)
            if cs[0] == "idx":
                k, n = int(nums[-1]), cs[2]
                v = [b / 255.0 for b in cs[1][k * n:k * n + n]]
                nums = v
            if len(nums) == 1:
                return _pack(nums[0], nums[0], nums[0])
            if len(nums) == 3:
                return _pack(*nums)
            if len(nums) == 4:
                return _cmyk(*nums)
        except Exception:
            pass
        return old

    def pen_of(self, res, name: bytes) -> Optional[float]:
        key = (id(res), name)
        if key not in self._lw:
            got = None
            try:
                lw = _get(_get(_get(res, "/ExtGState"), _name(name)), "/LW")
                got = float(lw) if lw is not None else None
            except Exception:
                got = None
            self._lw[key] = got
        return self._lw[key]

    def font_of(self, res, name) -> Optional[dict]:
        """How the strings of a font are read: its codes as characters, their widths."""
        key = (id(res), name)
        if key in self._fonts:
            return self._fonts[key]
        info: Optional[dict] = None
        try:
            f = _get(_get(res, "/Font"), _name(name))
            if f is not None:
                info = {"two": False, "map": None, "codec": "cp1252", "diff": {}, "w": {}, "dw": 0.5, "skip": False}
                tu = _get(f, "/ToUnicode")
                if tu is not None and hasattr(tu, "get_data"):
                    info["map"] = _to_unicode(bytes(tu.get_data())) or None
                info["names"] = {}
                sub = _text(_get(f, "/Subtype"))
                if sub == "/Type0":
                    info["two"] = True
                    d0 = _get(f, "/DescendantFonts")
                    d0 = d0[0].get_object() if d0 else None
                    if info["map"] is None:  # codes of glyphs, not of characters: the font program
                        info["tt"], info["gid"] = self._font_program(d0)
                        info["skip"] = info["tt"] is None
                    info["dw"] = float(_get(d0, "/DW") or 1000) / 1000.0 if d0 is not None else 1.0
                    W = list(_get(d0, "/W") or ()) if d0 is not None else []
                    k = 0
                    while k + 1 < len(W):
                        c = int(W[k])
                        nxt = W[k + 1].get_object() if hasattr(W[k + 1], "get_object") else W[k + 1]
                        if isinstance(nxt, (list, tuple)):
                            for j, w in enumerate(nxt):
                                info["w"][c + j] = float(w) / 1000.0
                            k += 2
                        else:
                            if k + 2 < len(W) and 0 <= int(nxt) - c <= 70000:
                                for cc in range(c, int(nxt) + 1):
                                    info["w"][cc] = float(W[k + 2]) / 1000.0
                            k += 3
                else:
                    if sub == "/Type3":  # glyphs drawn by small content streams of their own
                        fm = _get(f, "/FontMatrix")
                        try:
                            fm = tuple(float(v) for v in fm)
                        except Exception:
                            fm = (0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
                        info["t3"] = {"procs": _get(f, "/CharProcs"), "fm": fm if len(fm) == 6 else
                                      (0.001, 0.0, 0.0, 0.001, 0.0, 0.0), "res": _get(f, "/Resources")}
                    enc = _get(f, "/Encoding")
                    base = _text(enc if not hasattr(enc, "keys") else _get(enc, "/BaseEncoding"))
                    if base == "/MacRomanEncoding":
                        info["codec"] = "mac_roman"
                    if hasattr(enc, "keys"):
                        code = 0
                        for it in (_get(enc, "/Differences") or ()):
                            it = it.get_object() if hasattr(it, "get_object") else it
                            if isinstance(it, (int, float)):
                                code = int(it)
                                continue
                            nm = _text(it).lstrip("/")
                            info["names"][code] = nm
                            if nm.startswith("uni") and len(nm) >= 7:
                                try:
                                    info["diff"][code] = chr(int(nm[3:7], 16))
                                except ValueError:
                                    pass
                            elif len(nm) == 1:
                                info["diff"][code] = nm
                            elif nm in _GLYPHS:
                                info["diff"][code] = _GLYPHS[nm]
                            code += 1
                    first = int(_get(f, "/FirstChar") or 0)
                    unit = info["t3"]["fm"][0] if "t3" in info else 0.001  # (Type 3: widths in glyph space)
                    for j, w in enumerate(_get(f, "/Widths") or ()):
                        info["w"][first + j] = float(w) * unit
        except Exception:
            info = None
        self._keep.append(res)
        self._fonts[key] = info
        return info

    def _font_program(self, d0):
        """(the embedded TrueType program of a CID font, its CID -> glyph map); (None, None)."""
        try:
            fd = _get(d0, "/FontDescriptor")
            ff = _get(fd, "/FontFile2") if fd is not None else None
            if ff is None:
                return None, None
            tt = _TrueType(bytes(ff.get_data()))
            gm = _get(d0, "/CIDToGIDMap")
            gid = None
            if gm is not None and hasattr(gm, "get_data"):
                raw = bytes(gm.get_data())
                gid = {k // 2: (raw[k] << 8) | raw[k + 1] for k in range(0, len(raw) - 1, 2)}
            return tt, gid
        except Exception:
            return None, None

    def show(self, items, font, size, tm, ctm, tc, tw, tz, rise, mode, layer, rgb, ctx=None):
        """A string shown (Tj / TJ): remembered with its place; returns the text matrix after it."""
        if font is None or not size:
            self.skip("TEXT")
            return tm
        if font["map"] is None and ("t3" in font or font.get("tt") is not None):
            # no table of characters: the font's own (TrueType), else the glyphs as drawn
            text = self._tt_text(items, font) if font.get("tt") is not None else None
            if text is None:
                return self.glyphs(items, font, size, tm, ctm, tc, tw, tz, rise, mode, layer, rgb, ctx)
            items = text
        elif font["skip"]:
            self.skip("TEXT")
            return tm
        parts: List[str] = []
        adv = 0.0
        m = font["map"]
        for it in items:
            if isinstance(it, float):
                if parts and it < -180:
                    parts.append(" ")  # (a gap as wide as a space)
                adv -= it / 1000.0 * size * tz
            elif isinstance(it, tuple):
                raw = _string_bytes(it)
                codes = [(raw[k] << 8) | raw[k + 1] for k in range(0, len(raw) - 1, 2)] if font["two"] else list(raw)
                known_chars = it[2] if len(it) > 2 else None
                for j, c in enumerate(codes):
                    if known_chars is not None:
                        parts.append(known_chars[j])
                    elif m is not None and c in m:
                        parts.append(m[c])
                    elif not font["two"]:
                        parts.append(font["diff"].get(c) or bytes([c]).decode(font["codec"], "replace"))
                w = sum(font["w"].get(c, font["dw"]) for c in codes)
                spaces = 0 if font["two"] else codes.count(32)
                adv += (w * size + len(codes) * tc + spaces * tw) * tz
        string = "".join(parts).replace("\x00", "").rstrip()
        if string.strip() and len(self.texts) < MAX_TEXTS:
            M = _mul(tm, ctm)
            lu, lv = math.hypot(M[0], M[1]), math.hypot(M[2], M[3])
            if lu > 0 and lv > 0:
                # (a font that states no widths - one of the standard ones: the length is the drawing font's)
                known = bool(font["w"]) or font["two"]
                self.texts.append((M[4] + rise * M[2], M[5] + rise * M[3], M[0] / lu, M[1] / lu, abs(size) * lv,
                                   max(adv, 0.0) * lu if known else 0.0, string, layer, rgb, mode == 3))
        return _mul((1.0, 0.0, 0.0, 1.0, adv, 0.0), tm)

    def _tt_text(self, items, font):
        """The items of a TJ with the characters of a CID font read from its TrueType program
        (as ("c", raw, chars)); None when a glyph has no character."""
        tt, gid = font["tt"], font.get("gid")
        out = []
        for it in items:
            if isinstance(it, tuple):
                raw = _string_bytes(it)
                chars = []
                for k in range(0, len(raw) - 1, 2):
                    c = (raw[k] << 8) | raw[k + 1]
                    ch = tt.chars.get(gid.get(c, 0) if gid is not None else c)
                    if ch is None:
                        return None
                    chars.append(ch)
                out.append(("c", raw, chars))
            else:
                out.append(it)
        return out

    def glyphs(self, items, font, size, tm, ctm, tc, tw, tz, rise, mode, layer, rgb, ctx):
        """A string of a font with no characters: its glyphs drawn as on the sheet (outlines)."""
        if mode == 3 or mode == 7:  # (invisible: nothing is seen of it)
            return tm
        t3 = font.get("t3")
        tt = font.get("tt")
        fm = t3["fm"] if t3 else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        drawn = False
        for it in items:
            if isinstance(it, float):
                tm = _mul((1.0, 0.0, 0.0, 1.0, -it / 1000.0 * size * tz, 0.0), tm)
                continue
            if not isinstance(it, tuple):
                continue
            raw = _string_bytes(it)
            codes = [(raw[k] << 8) | raw[k + 1] for k in range(0, len(raw) - 1, 2)] if font["two"] else list(raw)
            for c in codes:
                G = _mul(_mul(fm, (size * tz, 0.0, 0.0, size, 0.0, rise)), _mul(tm, ctm))
                if t3 is not None and ctx is not None:
                    name = font["names"].get(c)
                    proc = _get(t3["procs"], "/" + name) if (name and t3["procs"] is not None) else None
                    if proc is not None and ctx["depth"] < MAX_DEPTH:
                        self.glyph += 1
                        try:
                            self.run(self.content(proc), t3["res"] if t3["res"] is not None else ctx["res"], G, rgb,
                                     rgb, 0.0, layer, ctx["depth"] + 1, ctx["chain"], ctx["clip"])
                        finally:
                            self.glyph -= 1
                        drawn = True
                elif tt is not None:
                    gid = font["gid"].get(c, 0) if font.get("gid") is not None else c
                    for C in tt.outline(gid):
                        X = C[:, 0] * G[0] + C[:, 1] * G[2] + G[4]
                        Y = C[:, 0] * G[1] + C[:, 1] * G[3] + G[5]
                        P = np.column_stack([X, Y])
                        P = np.vstack([P, P[:1]])
                        self.graw.append(np.hstack([P[:-1], P[1:]]))
                        self.gmeta.append((len(P) - 1, layer, rgb, self.chain_id(ctx["clip"] if ctx else ())))
                        drawn = True
                w = font["w"].get(c, font["dw"])
                spaces = 1 if (not font["two"] and c == 32) else 0
                tm = _mul((1.0, 0.0, 0.0, 1.0, (w * size + tc + spaces * tw) * tz, 0.0), tm)
        if not drawn:
            self.skip("TEXT")
        return tm

    def chain_id(self, clip: tuple) -> int:
        """A number for a set of clipping paths (0: none)."""
        if not clip:
            return 0
        k = self.chains.get(clip)
        if k is None:
            k = self.chains[clip] = len(self.chain_list)
            self.chain_list.append(clip)
        return k

    def add_clip(self, subs, ctm) -> Optional[int]:
        """A clipping path (page points); None when it clips nothing of the page."""
        a, b, c, d, e, f = ctm
        rings = []
        for pts, _flags in subs:
            P = np.array([(a * x + c * y + e, b * x + d * y + f) for x, y in pts], float)
            if len(P) > 1 and (P[0] == P[-1]).all():
                P = P[:-1]
            if len(P) >= 3:
                rings.append(P)
        if not rings:
            return None
        if len(rings) == 1 and len(rings[0]) == 4:
            R = rings[0]
            x0, y0, x1, y1 = self.box
            xs, ys = np.unique(np.round(R[:, 0], 6)), np.unique(np.round(R[:, 1], 6))
            if len(xs) == 2 and len(ys) == 2 and xs[0] <= x0 + 1e-6 and xs[1] >= x1 - 1e-6 and \
                    ys[0] <= y0 + 1e-6 and ys[1] >= y1 - 1e-6:
                return None  # (the whole page)
        self.clips.append(rings)
        return len(self.clips) - 1

    def content(self, obj) -> bytes:
        key = id(obj)
        self._keep.append(obj)
        if key not in self._data:
            try:
                data = bytes(obj.get_data())
            except Exception:
                data = b""
            if b"BI" in data and b"ID" in data and b"EI" in data:
                data, n = _INLINE_IMAGE.subn(b" ", data)
                if n:
                    self.skip("IMAGE", n)
            self._data[key] = data
        return self._data[key]

    # ------------------------------------------------------------- what is drawn
    def _matrix(self, ctm) -> int:
        k = self.ctm_id.get(ctm)
        if k is None:
            k = self.ctm_id[ctm] = len(self.ctms)
            self.ctms.append(ctm)
        return k

    def paint(self, subs, ends, ctm, layer, rgb, pen, filled: bool, clip: tuple = ()):
        if self.glyph:  # the strokes / shapes of a glyph: its outline, as text
            a, b, c, d, e, f = ctm
            for pts, flags in subs:
                if len(pts) >= 2:
                    P = np.array([(a * x + c * y + e, b * x + d * y + f) for x, y in pts], float)
                    self.graw.append(np.hstack([P[:-1], P[1:]]))
                    self.gmeta.append((len(P) - 1, layer, rgb, self.chain_id(clip)))
            return
        raw, curve = self.raw, self.curve
        n0 = len(curve)
        cid = self._matrix(ctm)
        centres = []
        for pts, flags in subs:
            if not flags:
                continue
            for a, b in zip(pts, pts[1:]):
                raw += a + b
            curve += flags
            if len(flags) >= 8 and all(flags) and pts[0] == pts[-1]:
                # a closed curve whose points are all as far from its middle: a circle
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
                r = [math.hypot(x - cx, y - cy) for x, y in pts]
                if max(r) > 0 and (max(r) - min(r)) <= 0.02 * max(r):
                    centres.append((cx, cy))
        n = len(curve) - n0
        if not n:
            return
        if len(curve) > MAX_SEGMENTS:
            raise DxfError("The page holds more line work than can be shown as a reference "
                           f"(over {MAX_SEGMENTS:,} segments). Plot a simpler sheet (fewer layers, no hatches).")
        # the pen in sheet points: the width under the matrix
        w = pen * math.sqrt(abs(ctm[0] * ctm[3] - ctm[1] * ctm[2]))
        ch = self.chain_id(clip)
        self.runs.append((n, cid, layer, rgb, w, filled, ch))
        if not filled:
            self.strokes += 1
        for kind, where in ((PT_END, ends), (PT_CENTER, centres)):
            if where:
                for p in where:
                    self.praw += p
                self.pruns.append((len(where), cid, layer, rgb, w, filled, kind, ch))

    def run(self, data: bytes, res, ctm, srgb: int, frgb: int, pen: float, layer: Optional[str], depth: int,
            chain: tuple, clip: tuple = ()):
        """Interpret one content stream (a page, or a form drawn in it)."""
        clipping = False  # W / W*: the path being made clips what is drawn after it
        stack: list = []
        push = stack.append
        saved: list = []
        marks: list = []
        scs: tuple = ("n",)
        fcs: tuple = ("n",)
        subs: list = []
        pts: Optional[list] = None
        flags: Optional[list] = None
        ends: list = []
        font, fsize, tm, tlm = None, 0.0, _IDENTITY, _IDENTITY
        lead, tc, tw, tz, rise, tmode = 0.0, 0.0, 0.0, 1.0, 0.0, 0
        for m in _TOKEN.finditer(data):
            kind = m.lastgroup
            if kind == "r":
                op = m.group()
                try:
                    push(float(op))
                    continue
                except ValueError:
                    pass
            elif kind == "n":
                push(m.group())
                continue
            elif kind == "s" or kind == "h":
                push((kind, m.group()))
                continue
            else:
                continue
            try:
                if op == b"l":
                    if pts is not None:
                        pts.append((stack[-2], stack[-1]))
                        flags.append(False)
                elif op == b"m":
                    pts, flags = [(stack[-2], stack[-1])], []
                    subs.append((pts, flags))
                elif op == b"c" or op == b"v" or op == b"y":
                    if pts is not None:
                        p0 = pts[-1]
                        if op == b"c":
                            p1, p2, p3 = (stack[-6], stack[-5]), (stack[-4], stack[-3]), (stack[-2], stack[-1])
                        elif op == b"v":
                            p1, p2, p3 = p0, (stack[-4], stack[-3]), (stack[-2], stack[-1])
                        else:
                            p1, p3 = (stack[-4], stack[-3]), (stack[-2], stack[-1])
                            p2 = p3
                        chords = _bezier(p0, p1, p2, p3)
                        pts += chords
                        if len(chords) == 1:
                            flags.append(False)
                        else:
                            flags += [True] * len(chords)
                            ends.append(p0)
                            ends.append(p3)
                elif op == b"h":
                    if pts is not None and flags and pts[-1] != pts[0]:
                        pts.append(pts[0])
                        flags.append(False)
                elif op == b"re":
                    x, y, w, h = stack[-4], stack[-3], stack[-2], stack[-1]
                    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
                    flags = [False, False, False, False]
                    subs.append((pts, flags))
                elif op in _STROKE:
                    if op in _CLOSING and pts is not None and flags and pts[-1] != pts[0]:
                        pts.append(pts[0])
                        flags.append(False)
                    if subs:
                        self.paint(subs, ends, ctm, layer, srgb, pen, False, clip)
                    if clipping and subs and not self.glyph:
                        c = self.add_clip(subs, ctm)
                        clip = clip + (c,) if c is not None else clip
                    clipping = False
                    subs, pts, flags, ends = [], None, None, []
                elif op in _FILL:
                    if subs:
                        self._fill(subs, ctm, layer, frgb, clip)
                    if clipping and subs and not self.glyph:
                        c = self.add_clip(subs, ctm)
                        clip = clip + (c,) if c is not None else clip
                    clipping = False
                    subs, pts, flags, ends = [], None, None, []
                elif op == b"n":
                    if clipping and subs and not self.glyph:
                        c = self.add_clip(subs, ctm)
                        clip = clip + (c,) if c is not None else clip
                    clipping = False
                    subs, pts, flags, ends = [], None, None, []
                elif op == b"W" or op == b"W*":
                    clipping = True
                elif op == b"q":
                    saved.append((ctm, srgb, frgb, pen, scs, fcs, clip))
                elif op == b"Q":
                    if saved:
                        ctm, srgb, frgb, pen, scs, fcs, clip = saved.pop()
                elif op == b"cm":
                    ctm = _mul(tuple(stack[-6:]), ctm)
                elif op == b"w":
                    pen = float(stack[-1])
                elif op == b"RG":
                    srgb, scs = _pack(stack[-3], stack[-2], stack[-1]), ("n",)
                elif op == b"G":
                    srgb, scs = _pack(stack[-1], stack[-1], stack[-1]), ("n",)
                elif op == b"K":
                    srgb, scs = _cmyk(stack[-4], stack[-3], stack[-2], stack[-1]), ("n",)
                elif op == b"rg":
                    frgb, fcs = _pack(stack[-3], stack[-2], stack[-1]), ("n",)
                elif op == b"g":
                    frgb, fcs = _pack(stack[-1], stack[-1], stack[-1]), ("n",)
                elif op == b"k":
                    frgb, fcs = _cmyk(stack[-4], stack[-3], stack[-2], stack[-1]), ("n",)
                elif op == b"CS":
                    scs, srgb = self.colour_space(res, stack[-1]), 0
                elif op == b"cs":
                    fcs, frgb = self.colour_space(res, stack[-1]), 0
                elif op == b"SC" or op == b"SCN":
                    srgb = self.colour(scs, [v for v in stack if isinstance(v, float)], srgb)
                elif op == b"sc" or op == b"scn":
                    frgb = self.colour(fcs, [v for v in stack if isinstance(v, float)], frgb)
                elif op == b"gs":
                    w = self.pen_of(res, stack[-1])
                    if w is not None:
                        pen = w
                elif op == b"BDC":
                    name = None
                    if len(stack) >= 2 and stack[-2] == b"/OC" and isinstance(stack[-1], bytes):
                        name = self.layer_of(res, stack[-1])
                    marks.append(layer)
                    if name is not None:
                        layer = name
                elif op == b"BMC":
                    marks.append(layer)
                elif op == b"EMC":
                    if marks:
                        layer = marks.pop()
                elif op == b"Do":
                    xo = _get(_get(res, "/XObject"), _name(stack[-1]))
                    sub = _text(_get(xo, "/Subtype")) if xo is not None else ""
                    if sub == "/Image":
                        self.skip("IMAGE")
                    elif sub == "/Form" and depth < MAX_DEPTH and id(xo) not in chain:
                        mat = _get(xo, "/Matrix")
                        try:
                            mat = tuple(float(v) for v in mat) if mat is not None else _IDENTITY
                        except Exception:
                            mat = _IDENTITY
                        if len(mat) != 6:
                            mat = _IDENTITY
                        own = self.layer_name(_get(xo, "/OC"))
                        res2 = _get(xo, "/Resources")
                        self.run(self.content(xo), res2 if res2 is not None else res, _mul(mat, ctm), srgb, frgb, pen,
                                 own if own is not None else layer, depth + 1, chain + (id(xo),), clip)
                elif op == b"Tj" or op == b"TJ" or op == b"'" or op == b'"':
                    if op == b'"':
                        tw, tc = float(stack[-3]), float(stack[-2])
                    if op != b"Tj" and op != b"TJ":
                        tlm = _mul((1.0, 0.0, 0.0, 1.0, 0.0, -lead), tlm)
                        tm = tlm
                    tm = self.show(stack if op == b"TJ" else stack[-1:], font, fsize, tm, ctm, tc, tw, tz, rise,
                                   tmode, layer, frgb, {"res": res, "depth": depth, "chain": chain, "clip": clip})
                elif op == b"BT":
                    tm = tlm = _IDENTITY
                elif op == b"Tf":
                    font, fsize = self.font_of(res, stack[-2]), float(stack[-1])
                elif op == b"Tm":
                    t6 = tuple(float(v) for v in stack[-6:])
                    if len(t6) == 6:
                        tm = tlm = t6
                elif op == b"Td" or op == b"TD":
                    if op == b"TD":
                        lead = -float(stack[-1])
                    tlm = _mul((1.0, 0.0, 0.0, 1.0, float(stack[-2]), float(stack[-1])), tlm)
                    tm = tlm
                elif op == b"T*":
                    tlm = _mul((1.0, 0.0, 0.0, 1.0, 0.0, -lead), tlm)
                    tm = tlm
                elif op == b"TL":
                    lead = float(stack[-1])
                elif op == b"Tc":
                    tc = float(stack[-1])
                elif op == b"Tw":
                    tw = float(stack[-1])
                elif op == b"Tz":
                    tz = float(stack[-1]) / 100.0
                elif op == b"Ts":
                    rise = float(stack[-1])
                elif op == b"Tr":
                    tmode = int(stack[-1])
                elif op == b"sh":
                    self.skip("FILL")
            except (IndexError, TypeError, ValueError, AttributeError):
                pass
            del stack[:]

    def _fill(self, subs, ctm, layer, rgb: int, clip: tuple = ()):
        if self.glyph:
            closed = [(pts + [pts[0]] if pts[-1] != pts[0] else pts, flags) for pts, flags in subs if len(pts) >= 2]
            self.paint(closed, [], ctm, layer, rgb, 0.0, True, clip)
            return
        if (rgb >> 16) >= 247 and ((rgb >> 8) & 255) >= 247 and (rgb & 255) >= 247:
            # white: a background or a mask - the lines drawn before it under it are not seen
            if len(self.curve) and len(self.masks) < MAX_MASKS:
                a, b, c, d, e, f = ctm
                for pts, _flags in subs:
                    if len(pts) >= 3:
                        self.masks.append((len(self.curve), len(self.texts),
                                           np.array([(a * x + c * y + e, b * x + d * y + f) for x, y in pts], float)))
            return
        if len(subs) == 1 and len(subs[0][0]) in (4, 5) and not any(subs[0][1]):
            # one four-sided shape over the whole page: the sheet's background colour
            a, b, c, d, e, f = ctm
            xs = [a * x + c * y + e for x, y in subs[0][0]]
            ys = [b * x + d * y + f for x, y in subs[0][0]]
            x0, y0, x1, y1 = self.box
            if max(xs) - min(xs) >= 0.95 * (x1 - x0) > 0 and max(ys) - min(ys) >= 0.95 * (y1 - y0) > 0:
                return
        small = len(subs) <= 4
        for pts, flags in subs:
            if any(flags) or len(pts) > FILL_MAX_VERTICES + 1 or len(pts) < 3:
                small = False
                break
        if not small:
            self.skip("FILL")
            return
        closed = []
        for pts, flags in subs:
            if pts[-1] != pts[0]:
                pts, flags = pts + [pts[0]], flags + [False]
            closed.append((pts, flags))
        self.paint(closed, [], ctm, layer, rgb, 0.0, True, clip)


def _open(source):
    try:
        import pypdf
    except ImportError:
        raise DxfError("PDF drawings need the 'pypdf' package of Python, which is not installed here "
                       "(pip install pypdf). DXF drawings do not need it.") from None
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        try:
            with open(source, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise DxfError(f"The file cannot be read: {exc}") from None
    if not is_pdf(data):
        raise DxfError("This is not a PDF file.")
    try:
        reader = pypdf.PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise DxfError("The PDF is protected with a password. Save a copy without the password "
                               "(or plot the drawing again) and try again.")
        n = len(reader.pages)
    except DxfError:
        raise
    except Exception as exc:
        raise DxfError(f"The PDF cannot be read ({type(exc).__name__}: {exc}).") from None
    if n < 1:
        raise DxfError("The PDF has no pages.")
    return reader, n


def pdf_page_count(source) -> int:
    return _open(source)[1]


def _box(page) -> Tuple[float, float, float, float]:
    for name in ("cropbox", "mediabox"):
        try:
            b = getattr(page, name)
            x0, y0, x1, y1 = float(b.left), float(b.bottom), float(b.right), float(b.top)
            if x1 > x0 and y1 > y0:
                return x0, y0, x1, y1
        except Exception:
            continue
    return 0.0, 0.0, 595.276, 841.89  # (A4, as viewers assume)


def _plot_scale(page, user_unit: float) -> Optional[float]:
    """The scale a measuring viewport of the page states (1 : n); None: it does not say."""
    try:
        vps = _get(page, "/VP")
        best, area = None, -1.0
        for vp in vps or ():
            vp = vp.get_object() if hasattr(vp, "get_object") else vp
            ms = _get(vp, "/Measure")
            if ms is None or _text(_get(ms, "/Subtype") or "/RL") != "/RL":
                continue
            x = _get(ms, "/X")
            fmt = x[0].get_object() if hasattr(x[0], "get_object") else x[0]
            c = float(_get(fmt, "/C"))
            unit = _text(_get(fmt, "/U")).strip().lower().rstrip(".")
            unit = {"metre": "m", "meter": "m", "metres": "m", "meters": "m", "millimetre": "mm", "millimeter": "mm",
                    "feet": "ft", "foot": "ft", "'": "ft", "inch": "in", "inches": "in", '"': "in"}.get(unit, unit)
            if unit not in UNIT_TO_M or not c > 0:
                continue
            bb = _get(vp, "/BBox")
            a = abs((float(bb[2]) - float(bb[0])) * (float(bb[3]) - float(bb[1]))) if bb is not None else 0.0
            if a > area:
                best, area = c * UNIT_TO_M[unit] / (PT_TO_MM / 1000.0 * user_unit), a
        if best is not None and 0.001 < best < 1.0e6:
            return float(f"{best:.6g}")
    except Exception:
        pass
    return None


def _layers_off(reader) -> List[str]:
    try:
        ocp = _get(_get(reader.trailer, "/Root"), "/OCProperties")
        if ocp is None:
            return []
        d = _get(ocp, "/D")
        name = lambda o: _text(_get(o.get_object() if hasattr(o, "get_object") else o, "/Name"))  # noqa: E731
        off = {name(o) for o in (_get(d, "/OFF") or ())} if d is not None else set()
        if d is not None and _text(_get(d, "/BaseState")) == "/OFF":
            on = {name(o) for o in (_get(d, "/ON") or ())}
            off |= {name(o) for o in (_get(ocp, "/OCGs") or ())} - on
        return sorted(n for n in off if n)
    except Exception:
        return []


class _TrueType:
    """What is needed of a TrueType font program embedded in a PDF (FontFile2): the
    character of each glyph (its own table of characters, read backwards) and the outlines
    of the glyphs (in units of the em)."""

    def __init__(self, data: bytes):
        import struct
        self.d = data
        self.u16 = lambda o: struct.unpack_from(">H", data, o)[0]  # noqa: E731
        self.i16 = lambda o: struct.unpack_from(">h", data, o)[0]  # noqa: E731
        self.u32 = lambda o: struct.unpack_from(">I", data, o)[0]  # noqa: E731
        self.tables: Dict[bytes, Tuple[int, int]] = {}
        for k in range(self.u16(4)):
            o = 12 + 16 * k
            self.tables[data[o:o + 4]] = (self.u32(o + 8), self.u32(o + 12))
        head = self.tables.get(b"head")
        self.upem = float(self.u16(head[0] + 18)) if head else 1000.0
        self.long_loca = bool(self.i16(head[0] + 50)) if head else False
        maxp = self.tables.get(b"maxp")
        self.glyphs = self.u16(maxp[0] + 4) if maxp else 0
        self.chars = self._chars()
        self._outlines: Dict[int, List[np.ndarray]] = {}

    def _chars(self) -> Dict[int, str]:
        """glyph -> character, from the Unicode tables of 'cmap' (formats 4 and 12)."""
        out: Dict[int, str] = {}
        cm = self.tables.get(b"cmap")
        if not cm:
            return out
        base = cm[0]
        subs = []
        for k in range(self.u16(base + 2)):
            o = base + 4 + 8 * k
            pid, eid, off = self.u16(o), self.u16(o + 2), self.u32(o + 4)
            if (pid, eid) in ((3, 10), (0, 4), (0, 6)):
                subs.insert(0, base + off)
            elif (pid, eid) in ((3, 1), (0, 3), (0, 0), (0, 1), (0, 2)):
                subs.append(base + off)
        for o in subs:
            fmt = self.u16(o)
            if fmt == 4:
                n = self.u16(o + 6) // 2
                ends, starts = o + 14, o + 16 + 2 * n
                deltas, ranges = starts + 2 * n, starts + 4 * n
                for i in range(n):
                    a, b = self.u16(starts + 2 * i), self.u16(ends + 2 * i)
                    delta, ro = self.i16(deltas + 2 * i), self.u16(ranges + 2 * i)
                    if a == 0xFFFF or b - a > 0x3000:
                        continue
                    for ch in range(a, b + 1):
                        if ro:
                            at = ranges + 2 * i + ro + 2 * (ch - a)
                            g = self.u16(at) if at + 2 <= len(self.d) else 0
                            g = (g + delta) & 0xFFFF if g else 0
                        else:
                            g = (ch + delta) & 0xFFFF
                        if g and g not in out:
                            out[g] = chr(ch)
            elif fmt == 12:
                for i in range(self.u32(o + 12)):
                    a, b, g0 = self.u32(o + 16 + 12 * i), self.u32(o + 20 + 12 * i), self.u32(o + 24 + 12 * i)
                    if b - a > 0x3000:
                        continue
                    for ch in range(a, b + 1):
                        out.setdefault(g0 + ch - a, chr(ch))
            if out:
                break
        return out

    def outline(self, gid: int, depth: int = 0) -> List[np.ndarray]:
        """The contours of glyph ``gid`` as closed polylines (em units / upem), curves as chords."""
        got = self._outlines.get(gid)
        if got is not None:
            return got
        out: List[np.ndarray] = []
        try:
            out = self._read_glyph(gid, depth)
        except Exception:
            out = []
        self._outlines[gid] = out
        return out

    def _read_glyph(self, gid: int, depth: int) -> List[np.ndarray]:
        loca, glyf = self.tables.get(b"loca"), self.tables.get(b"glyf")
        if not loca or not glyf or not 0 <= gid < self.glyphs or depth > 8:
            return []
        if self.long_loca:
            a, b = self.u32(loca[0] + 4 * gid), self.u32(loca[0] + 4 * gid + 4)
        else:
            a, b = 2 * self.u16(loca[0] + 2 * gid), 2 * self.u16(loca[0] + 2 * gid + 2)
        if b <= a:
            return []
        o = glyf[0] + a
        nc = self.i16(o)
        k = 1.0 / self.upem
        if nc < 0:  # made of other glyphs
            out = []
            p = o + 10
            while True:
                flags, g = self.u16(p), self.u16(p + 2)
                p += 4
                if flags & 1:
                    dx, dy = self.i16(p), self.i16(p + 2)
                    p += 4
                else:
                    dx, dy = (self.d[p] ^ 0x80) - 0x80, (self.d[p + 1] ^ 0x80) - 0x80
                    p += 2
                m = (1.0, 0.0, 0.0, 1.0)
                if flags & 8:
                    v = self.i16(p) / 16384.0
                    m = (v, 0.0, 0.0, v)
                    p += 2
                elif flags & 0x40:
                    m = (self.i16(p) / 16384.0, 0.0, 0.0, self.i16(p + 2) / 16384.0)
                    p += 4
                elif flags & 0x80:
                    m = tuple(self.i16(p + 2 * j) / 16384.0 for j in range(4))
                    p += 8
                for C in self.outline(g, depth + 1):
                    x = C[:, 0] * m[0] + C[:, 1] * m[2] + (dx * k if flags & 2 else 0.0)
                    y = C[:, 0] * m[1] + C[:, 1] * m[3] + (dy * k if flags & 2 else 0.0)
                    out.append(np.column_stack([x, y]))
                if not flags & 0x20:
                    break
            return out
        ends = [self.u16(o + 10 + 2 * j) for j in range(nc)]
        npts = ends[-1] + 1 if ends else 0
        p = o + 10 + 2 * nc
        p += 2 + self.u16(p)
        fl = []
        while len(fl) < npts:
            f = self.d[p]
            p += 1
            fl.append(f)
            if f & 8:
                r = self.d[p]
                p += 1
                fl.extend([f] * r)
        fl = fl[:npts]
        xs, ys = [], []
        for coords, short, same in ((xs, 2, 16), (ys, 4, 32)):
            v = 0
            for f in fl:
                if f & short:
                    dv = self.d[p]
                    p += 1
                    v += dv if f & same else -dv
                elif not f & same:
                    v += self.i16(p)
                    p += 2
                coords.append(v)
        out = []
        start = 0
        for e in ends:
            P = [(xs[j] * k, ys[j] * k, bool(fl[j] & 1)) for j in range(start, e + 1)]
            start = e + 1
            if len(P) < 2:
                continue
            # quadratic contour -> chords (an off-curve point between two off-curve ones: their middle is on it)
            pts: List[Tuple[float, float]] = []
            if not P[0][2]:
                P = ([P[-1]] if P[-1][2] else [((P[0][0] + P[-1][0]) / 2, (P[0][1] + P[-1][1]) / 2, True)]) + P
            pts.append(P[0][:2])
            q = P[1:] + [P[0]]
            i, cur = 0, P[0]
            while i < len(q):
                nxt = q[i]
                if nxt[2]:
                    pts.append(nxt[:2])
                    cur = nxt
                    i += 1
                    continue
                ctrl = nxt
                after = q[i + 1] if i + 1 < len(q) else P[0]
                end = after if after[2] else ((ctrl[0] + after[0]) / 2, (ctrl[1] + after[1]) / 2, True)
                for t in (0.25, 0.5, 0.75, 1.0):
                    u = 1 - t
                    pts.append((u * u * cur[0] + 2 * u * t * ctrl[0] + t * t * end[0],
                                u * u * cur[1] + 2 * u * t * ctrl[1] + t * t * end[1]))
                cur = end
                i += 2 if after[2] and i + 1 < len(q) else 1
            out.append(np.array(pts, float))
        return out


def _inside_rings(rings, P: np.ndarray) -> np.ndarray:
    """Points ``P`` inside a set of rings, even-odd (a ring in a ring is a hole)."""
    out = np.zeros(len(P), bool)
    for ring in rings:
        out ^= _inside(ring, P)
    return out


def _clip_segments(S: np.ndarray, rings, inside: bool) -> Tuple[np.ndarray, np.ndarray]:
    """The parts of segments ``S`` (N, 4) inside (``inside``) or outside the area of
    ``rings``: (pieces (M, 4), the segment each piece is of)."""
    n = len(S)
    if not n:
        return np.zeros((0, 4)), np.zeros(0, np.int64)
    allp = np.vstack(rings)
    bx0, by0 = allp.min(axis=0)
    bx1, by1 = allp.max(axis=0)
    # wholly beside the box of the area: all out
    away = (np.maximum(S[:, 0], S[:, 2]) < bx0) | (np.minimum(S[:, 0], S[:, 2]) > bx1) | \
        (np.maximum(S[:, 1], S[:, 3]) < by0) | (np.minimum(S[:, 1], S[:, 3]) > by1)
    near = np.flatnonzero(~away)
    keep_whole = np.flatnonzero(away) if not inside else np.zeros(0, np.int64)
    if not len(near):
        return S[keep_whole], keep_whole
    T = S[near]
    A = T[:, 0:2]
    D = T[:, 2:4] - A
    ts = [np.zeros(len(T)), np.ones(len(T))]
    with np.errstate(divide="ignore", invalid="ignore"):
        for ring in rings:
            Q = np.roll(ring, -1, axis=0)
            for (px, py), (qx, qy) in zip(ring.tolist(), Q.tolist()):
                ex, ey = qx - px, qy - py
                den = D[:, 0] * ey - D[:, 1] * ex
                wx, wy = px - A[:, 0], py - A[:, 1]
                t = (wx * ey - wy * ex) / den
                u = (wx * D[:, 1] - wy * D[:, 0]) / den
                ok = (np.abs(den) > 1e-12) & (t > 0) & (t < 1) & (u >= -1e-12) & (u <= 1 + 1e-12)
                ts.append(np.where(ok, t, np.nan))
    Tt = np.sort(np.column_stack(ts), axis=1)  # (the NaN last)
    t0, t1 = Tt[:, :-1], Tt[:, 1:]
    valid = np.isfinite(t0) & np.isfinite(t1) & (t1 - t0 > 1e-9)
    rows, cols = np.nonzero(valid)
    a, b = t0[rows, cols], t1[rows, cols]
    mid = A[rows] + (0.5 * (a + b))[:, None] * D[rows]
    keep = _inside_rings(rings, mid) == inside
    rows, a, b = rows[keep], a[keep], b[keep]
    # pieces of one segment that meet (split at a corner of the area that only touches it) as one
    if len(rows) > 1:
        join = (rows[1:] == rows[:-1]) & (np.abs(a[1:] - b[:-1]) <= 1e-9)
        if join.any():
            start = np.r_[True, ~join]
            first = np.flatnonzero(start)
            last = np.r_[first[1:] - 1, len(rows) - 1]
            rows, a, b = rows[first], a[first], b[last]
    pieces = np.column_stack([A[rows] + a[:, None] * D[rows], A[rows] + b[:, None] * D[rows]])
    src = near[rows]
    if len(keep_whole):
        return np.vstack([S[keep_whole], pieces]), np.concatenate([keep_whole, src])
    return pieces, src


def _hatched(S: np.ndarray, group: np.ndarray, spacing: float = HATCH_SPACING, min_lines: int = HATCH_LINES,
             tol: float = 0.05, cover: float = 0.15, square: float = HATCH_SPACING_SQUARE) -> np.ndarray:
    """True per segment of a hatch pattern: one of many parallel lines of one ``group``
    (a pen, a layer) beside each other at an even spacing no larger than ``spacing``
    (sheet mm) - what a CAD program plots a patterned hatch as - and the short pieces
    between such lines (the joints of a brick pattern).  Walls, grid lines and the rows of
    a table are further apart; the strokes of texts are short and scattered.

    Looked at in strips across the lines (``6 x spacing`` wide along them): a line of a
    strip counts when it covers ``cover`` of the strip; a hatch is ``min_lines`` such lines
    in an even progression (two hatches over each other, or a pattern of pairs, are two
    progressions)."""
    n = len(S)
    out = np.zeros(n, bool)
    if n < min_lines:
        return out
    d = S[:, 2:4] - S[:, 0:2]
    deg = np.degrees(np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi))
    # whole degrees; a line is looked at in the two nearest (the ends of plotted lines are rounded to the
    # plotter's grid: the lines of one hatch differ in angle by a fraction of a degree)
    b0 = np.floor(deg).astype(np.int64)
    both = np.r_[np.arange(n), np.arange(n)]
    bins = np.r_[b0, b0 + 1] % 180
    th = np.radians(bins.astype(float))
    c, s = np.cos(th), np.sin(th)
    X0, Y0, X1, Y1 = S[both, 0], S[both, 1], S[both, 2], S[both, 3]
    rho = -s * 0.5 * (X0 + X1) + c * 0.5 * (Y0 + Y1)
    ta, tb = c * X0 + s * Y0, c * X1 + s * Y1
    t0, t1 = np.minimum(ta, tb), np.maximum(ta, tb)
    A = 6.0 * spacing
    c0 = np.floor(t0 / A).astype(np.int64)
    c1 = np.floor(t1 / A).astype(np.int64)
    span = np.minimum(c1 - c0 + 1, 64)  # (a very long line: its first strips)
    pick = np.repeat(np.arange(len(both)), span)
    seg = both[pick]
    cell = c0[pick] + (np.arange(len(pick)) - np.repeat(np.cumsum(span) - span, span))
    covered = np.minimum(t1[pick], (cell + 1) * A) - np.maximum(t0[pick], cell * A)
    key = (np.asarray(group, np.int64)[seg] * 180 + bins[pick]) * 4_000_003 + cell
    r = rho[pick]
    o = np.lexsort((r, key))
    key, r, seg, covered = key[o], r[o], seg[o], covered[o]
    # distinct lines in each strip, and how much of the strip each covers
    new_line = np.r_[True, (key[1:] != key[:-1]) | (r[1:] - r[:-1] > tol)]
    line = np.cumsum(new_line) - 1
    lk, lr = key[new_line], r[new_line]
    cov = np.bincount(line, weights=np.maximum(covered, 0.0)) / A
    L = len(lk)
    starts = np.flatnonzero(np.r_[True, lk[1:] != lk[:-1]])
    gidx = np.cumsum(np.r_[True, lk[1:] != lk[:-1]]) - 1  # the strip of each line, numbered
    comp = gidx * 1.0e6 + (lr - np.minimum.reduceat(lr, starts)[gidx])
    usable = cov >= cover
    ang_line = (lk // 4_000_003) % 180
    lim = np.where((ang_line <= 1) | (ang_line >= 179) | (np.abs(ang_line - 90) <= 1), square, spacing)
    flag = np.zeros(L, bool)
    for j in range(1, 5):
        if L <= j:
            break
        i = np.flatnonzero((gidx[j:] == gidx[:-j]) & usable[:-j] & usable[j:])
        g = lr[i + j] - lr[i]
        keep = (g > tol) & (g <= lim[i])
        i, g = i[keep], g[keep]
        if not len(i):
            continue
        cur = i + j
        found = [i, cur]
        alive = np.ones(len(i), bool)
        for _m in range(2, min_lines):
            target = comp[cur] + g
            pos = np.clip(np.searchsorted(comp, target), 1, L - 1)
            near = np.where(np.abs(comp[pos] - target) <= np.abs(comp[pos - 1] - target), pos, pos - 1)
            alive &= (gidx[near] == gidx[i]) & (np.abs(comp[near] - target) <= tol) & (near > cur) & usable[near]
            cur = np.where(alive, near, cur)
            found.append(cur)
        for f in found:
            flag[f[alive]] = True
    out[seg[flag[line]]] = True
    if out.any():
        out |= _between(S, group, out, 2.0 * spacing, tol)
    return out


def _between(S: np.ndarray, group: np.ndarray, hatch: np.ndarray, longest: float, tol: float) -> np.ndarray:
    """Short segments both of whose ends lie on hatch lines of their group: the pieces of a
    pattern between its lines (brick joints)."""
    cand = np.flatnonzero(~hatch & (np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]) <= longest))
    out = np.zeros(len(S), bool)
    if not len(cand):
        return out
    H = np.flatnonzero(hatch)
    cellsz = max(4 * tol, 0.1)
    # the cells the hatch lines pass through (points every half cell along them)
    Lh = np.hypot(S[H, 2] - S[H, 0], S[H, 3] - S[H, 1])
    k = np.maximum(np.ceil(Lh / (0.5 * cellsz)).astype(np.int64), 1)
    if int(k.sum()) > 4_000_000:
        return out
    hh = np.repeat(H, k + 1)
    t = (np.arange(len(hh)) - np.repeat(np.cumsum(k + 1) - (k + 1), k + 1)) / np.repeat(k, k + 1)
    X = S[hh, 0] + t * (S[hh, 2] - S[hh, 0])
    Y = S[hh, 1] + t * (S[hh, 3] - S[hh, 1])
    g = np.asarray(group, np.int64)

    def cells(x, y, grp):
        return (grp * 1_000_003 + np.floor(x / cellsz).astype(np.int64)) * 1_000_003 + np.floor(y / cellsz).astype(np.int64)
    have = np.unique(cells(X, Y, g[hh]))
    ok = np.ones(len(cand), bool)
    for cx, cy in ((0, 1), (2, 3)):
        x, y, gg = S[cand, cx], S[cand, cy], g[cand]
        hit = np.zeros(len(cand), bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = cells(x + dx * cellsz, y + dy * cellsz, gg)
                pos = np.clip(np.searchsorted(have, key), 0, len(have) - 1)
                hit |= have[pos] == key
        ok &= hit
    out[cand[ok]] = True
    return out


def _inside(poly: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Points ``P`` (N, 2) inside (or on) polygon ``poly`` (K, 2) - even-odd."""
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), bool)
    n = len(poly)
    eps = 1e-9 * max(float(np.abs(poly).max()), 1.0)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        cross = ((y1 <= y) != (y2 <= y)) & (x <= (x2 - x1) * (y - y1) / (y2 - y1) + x1 + eps)
        inside ^= cross
    return inside


def _pen_text(w_pt: float, user_unit: float) -> str:
    return f"Pen {w_pt * user_unit * PT_TO_MM:.2f} mm"


def read_pdf(source, page: int = 1) -> Drawing:
    """Read page ``page`` (1 = the first) of a PDF file (a path, or its bytes) into a
    :class:`Drawing` in millimetres on the sheet.

    Raises :class:`DxfError` with a message for the user when the file cannot be read,
    is protected, or the page holds no line work (a scanned sheet)."""
    reader, count = _open(source)
    number = min(max(int(page), 1), count)
    try:
        pg = reader.pages[number - 1]
        x0, y0, x1, y1 = _box(pg)
        try:
            rot = int(round(float(pg.rotation or 0) / 90.0)) % 4
        except Exception:
            rot = 0
        try:
            uu = float(pg.user_unit or 1.0)
        except Exception:
            uu = 1.0
        if not uu > 0:
            uu = 1.0
        work = _Page((x0, y0, x1, y1))
        # the page's content: one stream, or several that read as one
        contents = _get(pg, "/Contents")
        parts = list(contents) if isinstance(contents, (list, tuple)) else ([contents] if contents is not None else [])
        data = b"\n".join(work.content(c.get_object() if hasattr(c, "get_object") else c) for c in parts)
        work.run(data, _get(pg, "/Resources"), _IDENTITY, 0, 0, 1.0, None, 0, ())
    except DxfError:
        raise
    except RecursionError:
        raise DxfError("The PDF page is nested too deeply to read.") from None
    except Exception as exc:
        raise DxfError(f"The PDF page cannot be read ({type(exc).__name__}: {exc}).") from None

    dwg = Drawing()
    dwg.source, dwg.pages, dwg.page = "pdf", count, number
    dwg.unit, dwg.metric = "mm", True
    try:
        dwg.version = "PDF " + str(reader.pdf_header).replace("%PDF-", "")
    except Exception:
        dwg.version = "PDF"
    W, H = (x1 - x0), (y1 - y0)
    k = uu * PT_TO_MM
    dwg.sheet = ((H if rot % 2 else W) * k, (W if rot % 2 else H) * k)
    dwg.plot_scale = _plot_scale(pg, uu)
    dwg.skipped = dict(work.skipped)

    def placed(raw: List[float], runs: List[tuple], width: int) -> np.ndarray:
        """Coordinates under their matrices, then as the page is viewed, in mm."""
        try:
            A = np.array(raw, float).reshape(-1, width)
        except (TypeError, ValueError):
            raise DxfError("The PDF page cannot be read (its content is damaged).") from None
        if not len(A):
            return A
        C = np.array(work.ctms, float).reshape(-1, 6)
        M = C[np.repeat(np.array([r[1] for r in runs], np.int64), np.array([r[0] for r in runs], np.int64))]
        out = np.empty_like(A)
        for c in range(0, width, 2):
            x = M[:, 0] * A[:, c] + M[:, 2] * A[:, c + 1] + M[:, 4] - x0
            y = M[:, 1] * A[:, c] + M[:, 3] * A[:, c + 1] + M[:, 5] - y0
            if rot == 1:
                x, y = y, W - x
            elif rot == 2:
                x, y = W - x, H - y
            elif rot == 3:
                x, y = H - y, x
            out[:, c], out[:, c + 1] = x * k, y * k
        return out

    seg = placed(work.raw, work.runs, 4)
    pts = placed(work.praw, work.pruns, 2)

    def viewed(x, y):
        """A point of the page (points) as the page is viewed, in mm."""
        x, y = x - x0, y - y0
        if rot == 1:
            x, y = y, W - x
        elif rot == 2:
            x, y = W - x, H - y
        elif rot == 3:
            x, y = H - y, x
        return x * k, y * k

    def viewed_ring(R: np.ndarray) -> np.ndarray:
        return np.column_stack(viewed(R[:, 0], R[:, 1]))

    clip_rings = [[viewed_ring(R) for R in rings] for rings in work.clips]

    def clipped(S: np.ndarray, chain_of: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The parts of segments inside the clipping paths they were drawn under:
        (pieces, the segment each is of)."""
        if not len(S) or not chain_of.any():
            return S, np.arange(len(S))
        parts, src = [S[chain_of == 0]], [np.flatnonzero(chain_of == 0)]
        for c in np.unique(chain_of[chain_of > 0]).tolist():
            idx = np.flatnonzero(chain_of == c)
            T = S[idx]
            for cid in work.chain_list[c]:
                T, j = _clip_segments(T, clip_rings[cid], True)
                idx = idx[j]
            parts.append(T)
            src.append(idx)
        src_all = np.concatenate(src)
        order_ = np.argsort(src_all, kind="stable")  # (in the order they were drawn)
        return np.vstack(parts)[order_], src_all[order_]

    def run_of(runs) -> np.ndarray:
        return np.repeat(np.arange(len(runs)), np.array([r[0] for r in runs], np.int64)) if runs else np.zeros(0, int)

    seg_run = run_of(work.runs)
    seg_chain = np.array([r[6] for r in work.runs], np.int64)[seg_run] if len(seg_run) else np.zeros(0, np.int64)
    seg, src = clipped(seg, seg_chain)
    seg_run, orig = seg_run[src], src
    curve = np.array(work.curve, bool)[src] if len(src) else np.zeros(0, bool)
    if len(pts):  # snap points outside the clipping paths are not on the sheet
        p_run = run_of(work.pruns)
        p_chain = np.array([r[7] for r in work.pruns], np.int64)[p_run]
        keep = np.ones(len(pts), bool)
        for c in np.unique(p_chain[p_chain > 0]).tolist():
            idx = np.flatnonzero(p_chain == c)
            for cid in work.chain_list[c]:
                keep[idx] &= _inside_rings(clip_rings[cid], pts[idx])
    else:
        p_run, keep = np.zeros(0, int), np.zeros(0, bool)

    # what a white shape drawn after it covers is not on the sheet as it is seen
    text_hidden = set()
    for upto, tupto, poly in work.masks:
        if not upto and not tupto:
            continue
        Q = viewed_ring(poly)
        qx0, qy0, qx1, qy1 = Q[:, 0].min(), Q[:, 1].min(), Q[:, 0].max(), Q[:, 1].max()
        if (qx1 - qx0) >= 0.95 * dwg.sheet[0] and (qy1 - qy0) >= 0.95 * dwg.sheet[1]:
            continue  # (the paper itself)
        cand = np.flatnonzero((orig < upto) &
                              (np.maximum(seg[:, 0], seg[:, 2]) >= qx0) & (np.minimum(seg[:, 0], seg[:, 2]) <= qx1) &
                              (np.maximum(seg[:, 1], seg[:, 3]) >= qy0) & (np.minimum(seg[:, 1], seg[:, 3]) <= qy1))
        if len(cand):
            pieces, j = _clip_segments(seg[cand], [Q], False)
            rest = np.ones(len(seg), bool)
            rest[cand] = False
            take = np.concatenate([np.flatnonzero(rest), cand[j]])
            seg = np.vstack([seg[rest], pieces])
            seg_run, orig, curve = seg_run[take], orig[take], curve[take]
            o = np.argsort(orig, kind="stable")
            seg, seg_run, orig, curve = seg[o], seg_run[o], orig[o], curve[o]
        for j in range(min(tupto, len(work.texts))):
            tx, ty = viewed(work.texts[j][0], work.texts[j][1])
            if qx0 <= tx <= qx1 and qy0 <= ty <= qy1 and _inside(Q, np.array([[tx, ty]]))[0]:
                text_hidden.add(j)
    if not len(seg):
        if work.skipped.get("IMAGE"):
            raise DxfError(f"Page {number} of the PDF is a picture (a scanned sheet), not line work: there is nothing "
                           "to draw over or snap to. Only PDFs plotted from a CAD program can be placed.")
        raise DxfError(f"Page {number} of the PDF has no lines"
                       + (" - only text." if (work.skipped.get("TEXT") or work.texts) else "."))

    # ---- layers: those of the file, else one per pen
    all_runs = work.runs + [r[:6] for r in work.pruns]
    text_layer_names = [(t[7] or NO_LAYER) if work.has_layers else "Text" for t in work.texts]
    if work.has_layers:
        names = [(r[2] or NO_LAYER) + (" (filled)" if r[5] else "") for r in all_runs]
    else:
        pens = sorted({(round(r[4] * k, 2), r[3]) for r in all_runs if not r[5]})
        if len(pens) <= MAX_PEN_LAYERS:
            black_only = all(c == 0 for _, c in pens)
            label = lambda w, c: f"Pen {w:.2f} mm" + ("" if black_only else f" #{c:06x}")  # noqa: E731
            names = ["Filled shapes" if r[5] else label(round(r[4] * k, 2), r[3]) for r in all_runs]
        elif len({w for w, _ in pens}) <= MAX_PEN_LAYERS:
            names = ["Filled shapes" if r[5] else f"Pen {round(r[4] * k, 2):.2f} mm" for r in all_runs]
        else:
            names = ["Filled shapes" if r[5] else "PDF" for r in all_runs]
    order: Dict[str, int] = {}
    ids = np.array([order.setdefault(nm, len(order)) for nm in names], np.int64)
    ns = len(work.runs)
    run_rgb = np.array([r[3] for r in work.runs], np.int64)
    run_filled = np.array([r[5] for r in work.runs], bool)
    layer = ids[:ns][seg_run]
    rgb = run_rgb[seg_run]
    # what lies wholly off the page is not on the sheet; zero-length pieces draw nothing
    sw, sh = dwg.sheet
    pad = 1.0e-6 * max(sw, sh)
    ok = np.all(np.isfinite(seg), axis=1)
    ok &= np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) > 0.0
    ok &= ~((np.maximum(seg[:, 0], seg[:, 2]) < -pad) | (np.minimum(seg[:, 0], seg[:, 2]) > sw + pad) |
            (np.maximum(seg[:, 1], seg[:, 3]) < -pad) | (np.minimum(seg[:, 1], seg[:, 3]) > sh + pad))
    # letters drawn as filled shapes with their words written invisibly over them (as some programs plot
    # texts): the words are kept as text, the shapes are left out
    filled_seg = run_filled[seg_run]
    if filled_seg.any() and any(t[9] for t in work.texts):
        mid = 0.5 * (seg[:, 0:2] + seg[:, 2:4])
        fi = np.flatnonzero(filled_seg & ok)
        for (tx, ty, ux, uy, size, length, string, _lay, _rgb, hid) in work.texts:
            if not hid or not len(fi):
                continue
            ln = length if length > 0 else 0.6 * size * len(string)
            vx, vy = -uy, ux
            corners = [(tx + ux * a_ + vx * b_, ty + uy * a_ + vy * b_)
                       for a_, b_ in ((-0.1 * size, -0.3 * size), (ln + 0.1 * size, -0.3 * size),
                                      (ln + 0.1 * size, 1.1 * size), (-0.1 * size, 1.1 * size))]
            Q = viewed_ring(np.array(corners, float))
            box = (mid[fi, 0] >= Q[:, 0].min()) & (mid[fi, 0] <= Q[:, 0].max()) & \
                (mid[fi, 1] >= Q[:, 1].min()) & (mid[fi, 1] <= Q[:, 1].max())
            if box.any():
                inner = fi[box][_inside(Q, mid[fi[box]])]
                ok[inner] = False
    seg, layer, rgb, curve, seg_run = seg[ok], layer[ok], rgb[ok], curve[ok], seg_run[ok]
    if not len(seg):
        raise DxfError(f"Page {number} of the PDF has no lines on the sheet.")
    # hatch patterns: on layers of their own, off at first
    hatch = _hatched(seg, np.where(run_filled[seg_run], -1, layer)) & ~run_filled[seg_run] & ~curve
    hatch_layers = set()
    if hatch.any():
        names_of = list(order)
        for lid in np.unique(layer[hatch]).tolist():
            nm = (names_of[lid] + " (hatch)") if work.has_layers else "Hatch patterns"
            h = order.setdefault(nm, len(order))
            layer[hatch & (layer == lid)] = h
            hatch_layers.add(nm)
    # the texts: where they start, their direction and size on the sheet as it is viewed
    for j, (tx, ty, ux, uy, size, length, string, _lay, trgb, _hid) in enumerate(work.texts):
        if j in text_hidden:
            continue
        px, py = viewed(tx, ty)
        if not (math.isfinite(px) and math.isfinite(py) and -pad <= px <= sw + pad and -pad <= py <= sh + pad):
            continue
        if rot == 1:
            ux, uy = uy, -ux
        elif rot == 2:
            ux, uy = -ux, -uy
        elif rot == 3:
            ux, uy = -uy, ux
        lid = order.setdefault(text_layer_names[j], len(order))
        dwg.texts.append((px, py, size * k * CAP_HEIGHT, math.degrees(math.atan2(uy, ux)), string, lid,
                          ((trgb >> 16) & 255, (trgb >> 8) & 255, trgb & 255), 0, 0, length * k, 1.0))
    # the glyphs of fonts with no characters: their outlines, as text (drawn, never snapped to)
    if work.graw:
        G = np.vstack(work.graw)
        G = np.column_stack([*viewed(G[:, 0], G[:, 1]), *viewed(G[:, 2], G[:, 3])])
        meta = work.gmeta
        g_chain = np.repeat(np.array([m_[3] for m_ in meta], np.int64), np.array([m_[0] for m_ in meta], np.int64))
        g_of = np.repeat(np.arange(len(meta)), np.array([m_[0] for m_ in meta], np.int64))
        G, j = clipped(G, g_chain)
        g_of = g_of[j]
        g_lay = np.array([order.setdefault((m_[1] or NO_LAYER) if work.has_layers else "Text", len(order))
                          for m_ in meta], np.int32)
        g_rgb = np.array([m_[2] for m_ in meta], np.int64)
        okg = np.all(np.isfinite(G), axis=1) & (np.hypot(G[:, 2] - G[:, 0], G[:, 3] - G[:, 1]) > 0)
        G, g_of = G[okg], g_of[okg]
        c_ = g_rgb[g_of]
        dwg.text_seg = np.ascontiguousarray(G, float)
        dwg.text_seg_layer = g_lay[g_of]
        dwg.text_seg_rgb = np.column_stack([(c_ >> 16) & 255, (c_ >> 8) & 255, c_ & 255]).astype(np.uint8)
    dwg.layers = list(order)
    first = {}
    for lid, c in zip(layer.tolist(), rgb.tolist()):
        if lid not in first:
            first[lid] = c
            if len(first) == len(order):
                break
    for t in dwg.texts:
        first.setdefault(t[5], (t[6][0] << 16) | (t[6][1] << 8) | t[6][2])
    dwg.layer_rgb = [((first.get(i, 0) >> 16) & 255, (first.get(i, 0) >> 8) & 255, first.get(i, 0) & 255)
                     for i in range(len(order))]
    off = set(_layers_off(reader)) if work.has_layers else set()
    dwg.layers_off = [nm for nm in dwg.layers if nm in off or nm in hatch_layers or
                      (nm.endswith(" (filled)") and nm[:-9] in off) or (nm.endswith(" (hatch)") and nm[:-8] in off)]
    dwg.seg = np.ascontiguousarray(seg, float)
    dwg.seg_layer = layer.astype(np.int32)
    dwg.seg_curve = curve
    dwg.seg_rgb = np.column_stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255]).astype(np.uint8)
    if len(pts):
        pl = ids[ns:][np.repeat(np.arange(len(work.pruns)), np.array([r[0] for r in work.pruns], np.int64))]
        pk = np.repeat(np.array([r[6] for r in work.pruns], np.uint8), np.array([r[0] for r in work.pruns], np.int64))
        okp = keep & np.all(np.isfinite(pts), axis=1) & (pts[:, 0] >= -pad) & (pts[:, 0] <= sw + pad) & \
            (pts[:, 1] >= -pad) & (pts[:, 1] <= sh + pad)
        dwg.pts, dwg.pts_layer, dwg.pts_kind = pts[okp], pl[okp].astype(np.int32), pk[okp]
    dwg.counts = {"PATH": int(work.strokes)}
    return dwg
