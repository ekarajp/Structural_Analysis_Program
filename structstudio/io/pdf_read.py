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
* lines wholly under a white shape drawn after them (a mask behind a title or a stamp)
  are left out, as on the sheet;
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
    kind, raw = tok
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
                if _text(_get(f, "/Subtype")) == "/Type0":
                    info["two"] = True
                    info["skip"] = info["map"] is None  # (codes of glyphs, not of characters)
                    d0 = _get(f, "/DescendantFonts")
                    d0 = d0[0].get_object() if d0 else None
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
                    for j, w in enumerate(_get(f, "/Widths") or ()):
                        info["w"][first + j] = float(w) / 1000.0
        except Exception:
            info = None
        self._keep.append(res)
        self._fonts[key] = info
        return info

    def show(self, items, font, size, tm, ctm, tc, tw, tz, rise, mode, layer, rgb):
        """A string shown (Tj / TJ): remembered with its place; returns the text matrix after it."""
        if font is None or font["skip"] or not size:
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
                for c in codes:
                    if m is not None and c in m:
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
                                   max(adv, 0.0) * lu if known else 0.0, string, layer, rgb))
        return _mul((1.0, 0.0, 0.0, 1.0, adv, 0.0), tm)

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

    def paint(self, subs, ends, ctm, layer, rgb, pen, filled: bool):
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
        self.runs.append((n, cid, layer, rgb, w, filled))
        if not filled:
            self.strokes += 1
        for kind, where in ((PT_END, ends), (PT_CENTER, centres)):
            if where:
                for p in where:
                    self.praw += p
                self.pruns.append((len(where), cid, layer, rgb, w, filled, kind))

    def run(self, data: bytes, res, ctm, srgb: int, frgb: int, pen: float, layer: Optional[str], depth: int,
            chain: tuple):
        """Interpret one content stream (a page, or a form drawn in it)."""
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
                        self.paint(subs, ends, ctm, layer, srgb, pen, False)
                    subs, pts, flags, ends = [], None, None, []
                elif op in _FILL:
                    if subs:
                        self._fill(subs, ctm, layer, frgb)
                    subs, pts, flags, ends = [], None, None, []
                elif op == b"n":
                    subs, pts, flags, ends = [], None, None, []
                elif op == b"q":
                    saved.append((ctm, srgb, frgb, pen, scs, fcs))
                elif op == b"Q":
                    if saved:
                        ctm, srgb, frgb, pen, scs, fcs = saved.pop()
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
                                 own if own is not None else layer, depth + 1, chain + (id(xo),))
                elif op == b"Tj" or op == b"TJ" or op == b"'" or op == b'"':
                    if op == b'"':
                        tw, tc = float(stack[-3]), float(stack[-2])
                    if op != b"Tj" and op != b"TJ":
                        tlm = _mul((1.0, 0.0, 0.0, 1.0, 0.0, -lead), tlm)
                        tm = tlm
                    tm = self.show(stack if op == b"TJ" else stack[-1:], font, fsize, tm, ctm, tc, tw, tz, rise,
                                   tmode, layer, frgb)
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

    def _fill(self, subs, ctm, layer, rgb: int):
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
        self.paint(closed, [], ctm, layer, rgb, 0.0, True)


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

    # lines wholly under a white shape drawn after them are not on the sheet as it is seen
    hidden = np.zeros(len(seg), bool)
    text_hidden = set()
    for upto, tupto, poly in work.masks:
        if not upto and not tupto:
            continue
        Q = np.array([viewed(x, y) for x, y in poly], float)
        qx0, qy0, qx1, qy1 = Q[:, 0].min(), Q[:, 1].min(), Q[:, 0].max(), Q[:, 1].max()
        if (qx1 - qx0) >= 0.95 * dwg.sheet[0] and (qy1 - qy0) >= 0.95 * dwg.sheet[1]:
            continue  # (the paper itself)
        part = seg[:upto]
        cand = np.flatnonzero((np.minimum(part[:, 0], part[:, 2]) >= qx0) & (np.maximum(part[:, 0], part[:, 2]) <= qx1) &
                              (np.minimum(part[:, 1], part[:, 3]) >= qy0) & (np.maximum(part[:, 1], part[:, 3]) <= qy1))
        if len(cand):
            inside = _inside(Q, part[cand, 0:2]) & _inside(Q, part[cand, 2:4]) & \
                _inside(Q, 0.5 * (part[cand, 0:2] + part[cand, 2:4]))
            hidden[cand[inside]] = True
        for j in range(min(tupto, len(work.texts))):
            tx, ty = viewed(work.texts[j][0], work.texts[j][1])
            if qx0 <= tx <= qx1 and qy0 <= ty <= qy1 and _inside(Q, np.array([[tx, ty]]))[0]:
                text_hidden.add(j)
    if not len(seg):
        if work.skipped.get("IMAGE"):
            raise DxfError(f"Page {number} of the PDF is a picture (a scanned sheet), not line work: there is nothing "
                           "to draw over or snap to. Only PDFs plotted from a CAD program can be placed.")
        raise DxfError(f"Page {number} of the PDF has no lines"
                       + (" - only text." if work.skipped.get("TEXT") else "."))

    # ---- layers: those of the file, else one per pen
    def per(runs, col):
        return [r[col] for r in runs]

    def spread(runs, values, dtype):
        return np.repeat(np.array(values, dtype), np.array([r[0] for r in runs], np.int64))

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
    ids = [order.setdefault(nm, len(order)) for nm in names]
    ns = len(work.runs)
    layer = spread(work.runs, ids[:ns], np.int32)
    rgb = spread(work.runs, per(work.runs, 3), np.int64)
    curve = np.array(work.curve, bool)
    # what lies wholly off the page is not on the sheet; zero-length pieces draw nothing
    sw, sh = dwg.sheet
    pad = 1.0e-6 * max(sw, sh)
    ok = np.all(np.isfinite(seg), axis=1) & ~hidden
    ok &= np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) > 0.0
    ok &= ~((np.maximum(seg[:, 0], seg[:, 2]) < -pad) | (np.minimum(seg[:, 0], seg[:, 2]) > sw + pad) |
            (np.maximum(seg[:, 1], seg[:, 3]) < -pad) | (np.minimum(seg[:, 1], seg[:, 3]) > sh + pad))
    seg, layer, rgb, curve = seg[ok], layer[ok], rgb[ok], curve[ok]
    if not len(seg):
        raise DxfError(f"Page {number} of the PDF has no lines on the sheet.")
    # the texts: where they start, their direction and size on the sheet as it is viewed
    for j, (tx, ty, ux, uy, size, length, string, _lay, trgb) in enumerate(work.texts):
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
    dwg.layers = list(order)
    first = {}
    for lid, c in zip(layer.tolist(), rgb.tolist()):
        if lid not in first:
            first[lid] = c
            if len(first) == len(order):
                break
    dwg.layer_rgb = [((first.get(i, 0) >> 16) & 255, (first.get(i, 0) >> 8) & 255, first.get(i, 0) & 255)
                     for i in range(len(order))]
    off = set(_layers_off(reader)) if work.has_layers else set()
    dwg.layers_off = [nm for nm in dwg.layers if nm in off or (nm.endswith(" (filled)") and nm[:-9] in off)]
    dwg.seg = np.ascontiguousarray(seg, float)
    dwg.seg_layer = layer.astype(np.int32)
    dwg.seg_curve = curve
    dwg.seg_rgb = np.column_stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255]).astype(np.uint8)
    if len(pts):
        pl = spread(work.pruns, ids[ns:], np.int32)
        pk = spread(work.pruns, [r[6] for r in work.pruns], np.uint8)
        okp = np.all(np.isfinite(pts), axis=1) & (pts[:, 0] >= -pad) & (pts[:, 0] <= sw + pad) & \
            (pts[:, 1] >= -pad) & (pts[:, 1] <= sh + pad)
        dwg.pts, dwg.pts_layer, dwg.pts_kind = pts[okp], pl[okp], pk[okp]
    dwg.counts = {"PATH": int(work.strokes)}
    return dwg
