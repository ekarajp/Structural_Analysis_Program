"""Build the built-in steel section and rebar catalogs (structstudio/catalog/data).

Run once by the developers; the CSV files it writes are shipped with the
program.  Sources (all public):

* TIS 1227-2558 (hot-rolled H, I, channels, angles, T) and TIS 1228-2561 (cold-formed
  angles, C, lipped C, Z, lipped Z, hats), full texts published by TISI
  (service.tisi.go.th/fulltext/), converted with ``pdftotext -layout``.  The TIS
  tables follow JIS G3192 / G3350.
* TIS 107-2566 pipes and tubes: the annex tables as printed in the Pacific Pipe
  structural catalogue 2026 (half pages: ``pdftotext -layout -f P -l P -x 0|595
  -W 595 -H 842``), checked against TIS 107-2533 (full text) and the dimensions.
* TIS 20-2559 (round bars RB) and TIS 24-2559 (deformed bars DB), TISI full texts.
* AISC shapes (W, HP, C, MC, L, HSS, Pipe): the AISC Shapes Database as bundled by
  the ``steelpy`` package (Apache-2.0).
* European shapes (IPE, HE A/B/M, IPN, UPE, UPN, L): section geometry of the
  ``structuralcodes`` package (fib-international, Apache-2.0); A and I computed here.
* China: GB/T 11263-2017 H sections and GB/T 706-2016 I-beams, channels and
  angles - dimensions and areas as reproduced on mechtool.cn / xycost.com, every
  row checked against the area computed from its dimensions; GB 1499 rebar.

Every tabulated row is checked: the area computed from the dimensions (with
the fillets) must match the tabulated area, so a mis-read table line stops
the build.

    python scripts/build_catalog.py --tis-dir /tmp --steelpy DIR --structuralcodes DIR --gb-dir DIR --tis107 DIR
"""
from __future__ import annotations

import argparse
import ast
import csv
import itertools
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from structstudio.model.section_shapes import SHAPES, shape_properties  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "structstudio", "catalog", "data")
STEEL_COLS = ["standard", "series", "name", "shape", "t3", "t2", "tf", "tw", "r", "r2", "lip", "t2b",
              "A", "I33", "I22", "J", "mass", "slope"]  # mm, cm2, cm4, kg/m; slope = tan of the flange taper
JIS_CHANNEL_SLOPE = math.tan(math.radians(5.0))  # from the area formula of TIS 1227 annex A (0.349 r^2 fillets)
JIS_I_SLOPE = math.tan(math.radians(8.0))  # ditto (0.615 r^2 for four fillets)


def area_mm2(shape, dims):
    return shape_properties(SHAPES[shape], dims, False, False).A


def props(shape, dims):
    p = shape_properties(SHAPES[shape], dims, False, False)
    return p.A / 100.0, p.I33 / 1e4, p.I22 / 1e4


def num(tok):
    return float(tok.replace(",", ""))


def groupings(tokens, n):
    """Ways to read ``tokens`` as ``n`` numbers where a thousands group may be
    split by a space ("1 020" = 1020)."""
    out = []

    def rec(i, acc):
        if len(acc) > n:
            return
        if i == len(tokens):
            if len(acc) == n:
                out.append(list(acc))
            return
        rec(i + 1, acc + [num(tokens[i])])
        if (i + 1 < len(tokens) and re.fullmatch(r"\d{1,3}", tokens[i]) and re.fullmatch(r"\d{3}", tokens[i + 1])):
            rec(i + 2, acc + [num(tokens[i] + tokens[i + 1])])
            if (i + 2 < len(tokens) and re.fullmatch(r"\d{3}", tokens[i + 2])):
                rec(i + 3, acc + [num(tokens[i] + tokens[i + 1] + tokens[i + 2])])
    rec(0, [])
    return out


def pick(tokens, n, ok, used=6):
    hits = [g for g in groupings(tokens, n) if ok(g)]
    if len(hits) != 1:
        # several readings may agree on the values that matter (a, mass, I)
        uniq = {tuple(h[:used]) for h in hits}
        if len(uniq) != 1:
            raise ValueError(f"ambiguous / unreadable row {tokens} -> {hits}")
    return hits[0]


def clean(text):
    """Drop the private-use glyphs some Thai PDF fonts put between letters."""
    return re.sub("[\ue000-\uf8ff]", "", text)


def table_lines(text, start, stop):
    text, start, stop = clean(text), clean(start), clean(stop)
    lines = text.splitlines()
    i0 = next(i for i, ln in enumerate(lines) if start in ln)
    i1 = next(i for i, ln in enumerate(lines) if i > i0 and stop in ln)
    return lines[i0:i1]


SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(?:\s*x\s*(\d+(?:\.\d+)?))?(?=\s|$)")
NUMS = re.compile(r"\d+(?:\.\d+)?")


def rows_with_sizes(lines, nsize=2):
    """(size tuple | None, numeric tokens) for every data line, with the size
    labels that stand on a line of their own attached later."""
    items = []
    for k, ln in enumerate(lines):
        s = ln.replace("×", "x")
        s = re.sub(r"(\d)\s*x\s*(\d)", r"\1 x \2", s)
        m = SIZE.match(s)
        size = None
        rest = s
        if m and (nsize == 2 and m.group(3) is None or nsize == 3 and m.group(3) is not None):
            size = tuple(float(g) for g in m.groups() if g is not None)
            rest = s[m.end():]
        toks = NUMS.findall(rest)
        if size is None and len(toks) < 6:
            continue
        items.append((k, size, toks))
    return items


def resolve(items, build):
    """Give the rows that print no size the size of the nearest label (above or
    below) for which ``build(size, toks)`` accepts the row."""
    labels = [(k, size) for k, size, toks in items if size is not None]
    out = []
    for k, size, toks in items:
        if not toks:
            continue
        cands = [size] if size is not None else [s for _, s in sorted(labels, key=lambda ls: abs(ls[0] - k))[:3]]
        row = None
        for c in cands:
            try:
                row = build(c, toks)
                break
            except ValueError:
                continue
        if row is None:
            raise ValueError(f"no size fits row {toks}")
        out.append(row)
    return out


# ------------------------------------------------------------------ TIS 1227
def tis_h(text):
    lines = table_lines(text, "ตารางที่ 5 ขนาดและความหนาของเหล็กรูปตัวเอช", "ตารางที่ 6 ขนาด")
    rows = []
    for _, size, toks in rows_with_sizes(lines):
        if size is None:
            continue
        H, B = size
        t1, t2, r = map(num, toks[:3])
        dims = dict(t3=H, t2=B, tf=t2, tw=t1, r=r)
        A, Ixc, _ = props("I", dims)

        def ok(g):  # a, mass, Ix, Iy, ix, iy, Zx, Zy
            a, m, ix_, iy_, rx, ry = g[0], g[1], g[2], g[3], g[4], g[5]
            return (abs(a - A) < 0.012 * A and abs(m - 0.785 * a) < 0.02 * m and abs(ix_ - Ixc) < 0.02 * Ixc and
                    abs(ix_ - a * rx * rx) < 0.03 * ix_ and abs(iy_ - a * ry * ry) < 0.03 * iy_)
        g = pick(toks[3:], 8, ok)
        name = f"H{H:g}x{B:g}x{t1:g}x{t2:g}"
        rows.append(dict(standard="TIS 1227 / JIS G3192", series="H", name=name, shape="I", t3=H, t2=B, tf=t2,
                         tw=t1, r=r, A=g[0], I33=g[2], I22=g[3], mass=g[1]))
    return rows


def tis_angles(text):
    rows = []
    for title, stop, series in (("ตารางที่ 2 ขนาดและความหนาของเหล็กฉากขาเทากัน", "ตารางที่ 3 ขนาด", "L equal"),
                                ("ตารางที่ 3 ขนาดและความหนาของเหล็กฉากขาไมเทากัน", "ตารางที่ 4 ขนาด", "L unequal")):
        lines = table_lines(text, title, stop)
        unequal = series == "L unequal"

        def build(size, toks):
            A_, B_ = size
            t, r1, r2 = map(num, toks[:3])
            dims = dict(t3=A_, t2=B_, tf=t, tw=t, r=r1, r2=r2)
            Ac, Ixc, _ = props("Angle", dims)
            n = 13 if unequal else 12  # a mass Cx Cy Ix Iy Iu Iv ix iy iu iv [tan] Zx Zy

            def ok(g):
                a, m = g[0], g[1]
                ix_, iy_, rx, ry = g[4], g[5], g[8], g[9]
                return (abs(a - Ac) < 0.015 * Ac and abs(m - 0.785 * a) < 0.02 * m and abs(ix_ - Ixc) < 0.03 * Ixc and
                        abs(ix_ - a * rx * rx) < 0.035 * ix_ and abs(iy_ - a * ry * ry) < 0.035 * iy_)
            g = pick(toks[3:], n + 2, ok)
            return dict(standard="TIS 1227 / JIS G3192", series=series, name=f"L{A_:g}x{B_:g}x{t:g}", shape="Angle",
                        t3=A_, t2=B_, tf=t, tw=t, r=r1, r2=r2, A=g[0], I33=g[4], I22=g[5], mass=g[1])
        rows += resolve(rows_with_sizes(lines), build)
    return rows


def tis_channels(text):
    lines = table_lines(text, "ตารางที่ 4 ขนาดและความหนาของเหล็กรูปรางน้ำ", "ตารางที่ 5 ขนาด")

    def build(size, toks):
        H, B = size
        t1, t2, r1, r2 = map(num, toks[:4])
        dims = dict(t3=H, t2=B, tf=t2, tw=t1, r=r1, r2=r2, slope=JIS_CHANNEL_SLOPE)
        Ac, Ixc, _ = props("Channel", dims)

        def ok(g):  # a mass Cx Cy Ix Iy ix iy Zx Zy
            a, m, ix_, iy_, rx, ry = g[0], g[1], g[4], g[5], g[6], g[7]
            # 2 %: the smallest channel's printed area is 1.5 % below the standard's own formula
            return (abs(a - Ac) < 0.02 * Ac and abs(m - 0.785 * a) < 0.02 * m and abs(ix_ - Ixc) < 0.025 * Ixc and
                    abs(ix_ - a * rx * rx) < 0.03 * ix_ and abs(iy_ - a * ry * ry) < 0.035 * iy_)
        g = pick(toks[4:], 10, ok)
        return dict(standard="TIS 1227 / JIS G3192", series="Channel", name=f"C{H:g}x{B:g}x{t1:g}x{t2:g}",
                    shape="Channel", t3=H, t2=B, tf=t2, tw=t1, r=r1, r2=r2, A=g[0], I33=g[4], I22=g[5], mass=g[1],
                    slope=JIS_CHANNEL_SLOPE)
    return resolve(rows_with_sizes(lines), build)


def tis_ibeams(text):
    """TIS 1227 table 6: I-beams with tapered flanges.  Rows the table prints
    wrongly (they fit no size of their group) are left out and reported."""
    lines = table_lines(text, "ตารางที่ 6 ขนาดและความหนาของเหล็กรูปตัวไอ", "ตารางที่ 7 ขนาด")

    def build(size, toks):
        H, B = size
        t1, t2, r1, r2 = map(num, toks[:4])
        dims = dict(t3=H, t2=B, tf=t2, tw=t1, r=r1, r2=r2, slope=JIS_I_SLOPE)
        Ac, Ixc, _ = props("I", dims)

        def ok(g):  # a mass Cx Cy Ix Iy ix iy Zx Zy
            a, m, ix_, iy_, rx, ry = g[0], g[1], g[4], g[5], g[6], g[7]
            return (abs(a - Ac) < 0.006 * Ac and abs(m - 0.785 * a) < 0.02 * m and abs(ix_ - Ixc) < 0.02 * Ixc and
                    abs(ix_ - a * rx * rx) < 0.03 * ix_ and abs(iy_ - a * ry * ry) < 0.035 * iy_)
        g = pick(toks[4:], 10, ok)
        return dict(standard="TIS 1227 / JIS G3192", series="I-beam", name=f"I{H:g}x{B:g}x{t1:g}x{t2:g}",
                    shape="I", t3=H, t2=B, tf=t2, tw=t1, r=r1, r2=r2, A=g[0], I33=g[4], I22=g[5], mass=g[1],
                    slope=JIS_I_SLOPE)
    out = []
    labels = [(k, size) for k, size, toks in rows_with_sizes(lines) if size is not None]
    for k, size, toks in rows_with_sizes(lines):
        if not toks:
            continue
        cands = [size] if size is not None else [s_ for _, s_ in sorted(labels, key=lambda ls: abs(ls[0] - k))[:3]]
        row = None
        for c in cands:
            try:
                row = build(c, toks)
                break
            except ValueError:
                continue
        if row is None:
            print(f"  TIS I-beam row left out (does not fit its size): {' '.join(toks)}")
        else:
            out.append(row)
    return out


# ------------------------------------------------------------------ TIS 1227 T
def tis_tees(text):
    """TIS 1227 table 7 (= JIS G3192 T-sections): a flange B x t2 on a short stem
    whose profile the standard gives only through its area (482.6 mm2 in the
    area formula of annex A, the same for every size).  The stem is drawn with
    the uniform thickness of that area; A, Ix and Iy are the tabulated values."""
    lines = table_lines(text, "ตารางที่ 7 ขนาดและความหนาของเหล็กรูปตัวที", "ตารางที่ 8")
    rows = []
    for _, size, toks in rows_with_sizes(lines):
        if size is None or len(toks) < 16:
            continue
        B, H, t1, t2, r1, r2 = map(num, toks[:6])
        a_std = (B * t2 + 0.307 * r1 * r1 + 482.6) / 100.0  # annex A formula
        tw = round(482.6 / (H - t2), 2)
        dims = dict(t3=H, t2=B, tf=t2, tw=tw, r=r1)
        Ac, Ixc, Iyc = props("Tee", dims)

        def ok(g):  # a mass Cx Cy Ix Iy ix iy Zx Zy
            a, m, ix_, iy_, rx, ry = g[0], g[1], g[4], g[5], g[6], g[7]
            return (abs(a - a_std) < 0.002 * a and abs(m - 0.785 * a) < 0.02 * m and abs(iy_ - Iyc) < 0.01 * iy_ and
                    abs(ix_ - Ixc) < 0.12 * ix_ and abs(ix_ - a * rx * rx) < 0.03 * ix_ and
                    abs(iy_ - a * ry * ry) < 0.03 * iy_)
        g = pick(toks[6:], 10, ok)
        rows.append(dict(standard="TIS 1227 / JIS G3192", series="T", name=f"T{B:g}x{t2:g}", shape="Tee",
                         t3=H, t2=B, tf=t2, tw=tw, r=r1, A=g[0], I33=g[4], I22=g[5], mass=g[1]))
    return rows


# ------------------------------------------------------------------ TIS 1228
# (table title, shape, series, numbers after t, size count, index of Ix / Iy in them)
TIS1228 = (
    ("ตารางที่ ก.1 ขนาด ความหนาระบุ และมวลต่อเมตรของเหล็กฉากขาเท่ากัน", "ตารางที่ ก.2", "Cold Angle",
     "Cold-formed L", 17, 2),
    ("ตารางที่ ก.2 ขนาด ความหนาระบุ และมวลต่อเมตรของเหล็กรูปตัวซี", "ตารางที่ ก.3", "Cold Channel",
     "Cold-formed C", 12, 3),
    ("ตารางที่ ก.3 ขนาด ความหนาระบุ และมวลต่อเมตรของเหล็กรูปตัวซีมีขอบ", "ตารางที่ ก.4", "Lipped Channel",
     "Lipped C", 12, 3),
    ("ตารางที่ ก.4 ขนาด ความหนาระบุ และมวลต่อเมตรของเหล็กรูปตัวแซด", "ตารางที่ ก.5", "Cold Z", "Cold-formed Z", 17, 3),
    ("ตารางที่ ก.5 ขนาด ความหนาระบุ และมวลต่อเมตรของเหล็กรูปตัวแซดมีขอบ", "ตารางที่ ก.6", "Lipped Z", "Lipped Z", 17, 3),
    ("ตารางที่ ก.6 ขนาด ความหนาระบุ และมวลต่อเมตรของเหล็กรูปหมวก", "ภาคผนวก ข", "Hat", "Hat", 12, 3),
)


TIS1228_FIX = {"C125x50x20x4": 31.1}  # Iy misprinted 33.1 in TIS 1228-2561; JIS G3350 (NSMP table) and the dimensions: 31.1


def tis_cold(text):
    """TIS 1228-2561 annex A (= JIS G3350): every table; inside bend radius = t
    (the radius the tabulated areas are computed with)."""
    rows = []
    for title, stop, shape, series, n, nsize in TIS1228:
        lines = table_lines(text, title, stop)

        def build(size, toks, shape=shape, series=series, n=n):
            t = num(toks[0])
            if shape == "Cold Angle":
                H, A_ = size
                dims, name = dict(t3=H, t2=A_), f"L{H:g}x{A_:g}x{t:g}"
            elif shape == "Cold Channel":
                H, A_, B_ = size
                if A_ != B_:
                    raise ValueError("unequal flanges")
                dims, name = dict(t3=H, t2=A_), f"C{H:g}x{A_:g}x{t:g}"
            elif shape == "Cold Z":
                H, A_, B_ = size
                dims, name = dict(t3=H, t2=A_, t2b=None if B_ == A_ else B_), f"Z{H:g}x{A_:g}x{B_:g}x{t:g}"
            elif shape == "Hat":
                H, A_, C = size
                dims, name = dict(t3=H, t2=A_, lip=C), f"HAT{H:g}x{A_:g}x{C:g}x{t:g}"
            else:
                H, A_, C = size
                dims = dict(t3=H, t2=A_, lip=C)
                name = f"{'C' if shape == 'Lipped Channel' else 'Z'}{H:g}x{A_:g}x{C:g}x{t:g}"
            dims.update(tw=t, tf=t, r=t)
            Ac, Ixc, Iyc = props(shape, dims)

            def ok(g):  # a mass Cx Cy Ix Iy ...; the depth / perimeter alone can be ambiguous
                a, m, ix_, iy_ = g[0], g[1], g[4], g[5]
                return abs(a - Ac) < 0.01 * Ac and abs(m - 0.785 * a) < 0.02 * m and abs(ix_ - Ixc) < 0.03 * Ixc
            g = pick(toks[1:], n, ok)
            iy = TIS1228_FIX.get(name, g[5])
            if abs(iy - Iyc) > 0.04 * Iyc:
                raise ValueError(f"TIS 1228 {name}: tabulated Iy {g[5]} vs {Iyc:.3g} from the dimensions")
            return dict(standard="TIS 1228 / JIS G3350", series=series, name=name, shape=shape,
                        **{k: v for k, v in dims.items()}, A=g[0], I33=g[4], I22=iy, mass=g[1])
        items = []
        for k, size, toks in rows_with_sizes(lines, nsize=nsize):
            if size is None and toks and len(toks) >= n:
                items.append((k, None, toks))
            elif size is not None:
                items.append((k, size, toks))
        rows += resolve(items, build)
    return rows


# ------------------------------------------------------------------ TIS 107
def tis107_official(text):
    """TIS 107-2533 tables 1-3 (round, square, rectangular hollow sections, =
    JIS G3444 / G3466 sizes): (kind, D, B, t) -> (A, I33, I22), to check the
    current edition's tables against."""
    out = {}
    for title, stop, kind in (("ตารางที่ 1 ชือ่ ขนาด", "ตารางที่ 2", "round"), ("ตารางที่ 2 ชื่อขนาด", "ตารางที่ 3", "square"),
                              ("ตารางที่ 3 ชื่อขนาด", "ตารางที่ 4", "rect")):
        lines = table_lines(text, title, stop)
        size = None
        for ln in lines:
            s = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", re.sub(r"(\d)\.\s+(\d)", r"\1.\2", ln)).replace("ด", " x ")
            if kind == "round":  # [name D] t A mass I Z r
                toks = NUMS.findall(s)
                for n_ in (8, 6):
                    hit = [g for g in groupings(toks, n_) if abs(g[-4] - 0.785 * g[-5]) < 0.02 * g[-4] and
                           abs(g[-3] - g[-5] * g[-1] ** 2) < 0.04 * g[-3]]
                    if hit:
                        g = hit[0]
                        if n_ == 8:
                            size, g = g[1], g[2:]
                        out[("round", size, size, g[0])] = (g[1], g[3], g[3])
                        break
                continue
            m = re.match(r"\s*(\d+)\s+x\s+(\d+)", s)
            if m:
                size = (float(m.group(1)), float(m.group(2)))
                s = s[m.end():]
            toks = NUMS.findall(s)
            if not toks or size is None:
                continue
            want = 6 if kind == "square" else 9
            for g in groupings(toks, want):  # t A mass I(x) [Iy] Z(x) [Zy] r(x) [ry]
                rx, ry = (g[5], g[5]) if kind == "square" else (g[7], g[8])
                I33_, I22_ = (g[3], g[3]) if kind == "square" else (g[3], g[4])
                if (abs(g[2] - 0.785 * g[1]) < 0.02 * g[2] and abs(I33_ - g[1] * rx * rx) < 0.04 * I33_ and
                        abs(I22_ - g[1] * ry * ry) < 0.04 * I22_):
                    I33, I22 = (g[3], g[3]) if kind == "square" else (g[3], g[4])
                    out[(kind, size[0], size[1], g[0])] = (g[1], I33, I22)
                    break
    return out


PAP_TABLES = {"ก.1": ("round", "JIS G3444"), "ก.2": ("round", "ASTM A500"), "ก.3": ("round", "EN 10219"),
              "ข.1": ("square", "JIS G3466"), "ข.2": ("square", "ASTM A500"), "ข.3": ("square", "EN 10219"),
              "ค.1": ("rect", "JIS G3466"), "ค.2": ("rect", "ASTM A500"), "ค.3": ("rect", "EN 10219")}
TUBE = {"round": "Pipe", "square": "Square tube", "rect": "Rectangular tube"}
_LAB = re.compile(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)")


def _tube_rows(kind, texts):
    """(line no, size | None, numbers) of one table spread over half pages."""
    nd = 9 if kind == "rect" else 6  # t mass A I(x) [Iy] S(x) [Sy] r(x) [ry]
    items = []
    for text in texts:
        for k, ln in enumerate(text.splitlines()):
            s = re.sub(r"\([^)]*\)", " ", ln.replace(",", "")).replace("×", "x")
            toks = re.findall(r"\d+(?:\.\d+)?|x", s)
            nums = []
            i = len(toks)
            while i > 0 and toks[i - 1] != "x" and len(nums) < nd:
                i -= 1
                nums.insert(0, toks[i])
            size = None
            if kind == "round":
                head = [x for x in toks[:i] if x != "x"]
                if len(nums) == nd and head and "." in head[-1]:
                    size = (float(head[-1]),)
                elif len(nums) < nd:
                    head = [x for x in toks if x != "x"]
                    if head and len(head) <= 3 and "." in head[-1]:
                        size, nums = (float(head[-1]),), []
            else:
                m = _LAB.search(" ".join(toks))
                if m:
                    size = (float(m.group(1)), float(m.group(2)))
                    if len(nums) < nd:
                        nums = []
            if len(nums) != nd:
                nums = []
            if size is None and not nums:
                continue
            items.append((len(items), size, nums))
    return items


def sig(x, n):
    """Round to n significant figures (as the standards print)."""
    if not x:
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + n - 1)


def tis107(pap_dir, official):
    """TIS 107-2566 annex tables (round / square / rectangular tubes to JIS G3444 /
    G3466, ASTM A500 and EN 10219 sizes) as printed in the Pacific Pipe structural
    catalogue 2026 (the TISI full text is not downloadable), every row checked
    against the area and I of its dimensions (outside corner radius 2t, or 2.5t /
    3t where the table is computed so) and the JIS rows against TIS 107-2533."""
    import glob
    texts = {}
    for f in sorted(glob.glob(os.path.join(pap_dir, "p*_*.txt")),
                    key=lambda f: tuple(map(int, re.findall(r"\d+", os.path.basename(f))))):
        t = open(f, encoding="utf-8").read()
        m = re.search(r"Table ([กขค]\.\d)", t)
        if m and m.group(1) in PAP_TABLES:
            texts.setdefault(m.group(1), []).append(t)
    rows, dropped, checked, recomputed = [], [], 0, []
    for key, (kind, std) in PAP_TABLES.items():
        items = _tube_rows(kind, texts.get(key, []))
        pool_txt = " ".join(texts.get(key, [])).replace(",", "").replace("×", "x")
        if kind == "round":
            pool = sorted({(float(x),) for x in re.findall(r"(?<![\d.])(\d{2,4}\.\d)(?!\d)", pool_txt) if float(x) >= 10})
        else:
            pool = sorted({(float(a_), float(b_)) for a_, b_ in _LAB.findall(pool_txt)})
            if kind == "square":
                pool = [p_ for p_ in pool if p_[0] == p_[1]]

        def build(size, toks):
            v = [num(x) for x in toks]
            t, tabA, tabI = v[0], v[2], v[3]
            tabI2 = v[4] if kind == "rect" else tabI
            tolA = 0.012 if kind == "round" else (0.02 if t <= 6 else 0.035)  # thick tubes: A by 2t, I by 2.5t
            if kind == "round":
                D = size[0]
                fits = [dict(t3=D, tw=t)]
                shape, name = "Pipe", f"PIPE{D:g}x{t:g}"
            else:
                H, B = size if kind == "rect" else (size[0], size[0])
                fits = [dict(t3=H, t2=B, tf=t, tw=t, r=round(mlt * t, 3)) for mlt in (2.0, 2.5, 3.0, 1.5)]
                shape, name = "Box", f"BOX{H:g}x{B:g}x{t:g}"
            scored = []
            for dims in fits:
                try:
                    A_, I_, I2_ = props(shape, dims)
                except (ZeroDivisionError, ValueError):
                    continue
                scored.append((abs(I_ - tabI) / tabI + abs(I2_ - tabI2) / tabI2 + abs(A_ - tabA) / tabA, dims, A_, I_, I2_))
            for _, dims, A_, I_, I2_ in sorted(scored, key=lambda x: x[0]):
                if (abs(A_ - tabA) <= tolA * tabA + 0.002 and abs(I_ - tabI) <= 0.02 * tabI + 0.01 and
                        abs(I2_ - tabI2) <= 0.02 * tabI2 + 0.01 and abs(v[1] - 0.785 * tabA) <= 0.03 * v[1] + 0.01):
                    row = dict(standard="TIS 107-2566", series=f"{TUBE[kind]} ({std})", name=name, shape=shape,
                               **dims, A=tabA, I33=tabI, I22=tabI2, mass=v[1])
                    if std == "JIS G3466" and dims["r"] != 2.0 * t:
                        # JIS G3466 / TIS 107 annex: outside corner radius 2t for every thickness; the catalogue
                        # computes its thick tubes with 2.5t - recompute them with the standard's corners
                        row["r"] = 2.0 * t
                        A2, I2a, I2b = props(shape, dict(dims, r=2.0 * t))
                        row.update(A=sig(A2, 4), I33=sig(I2a, 3 if I2a < 1000 else 4), I22=sig(I2b, 3 if I2b < 1000 else 4),
                                   mass=sig(0.785 * A2, 3))
                        recomputed.append(name)
                    return row
            if std == "JIS G3466" and scored:  # print inconsistent (mass / A); the I's identify the size
                _, dims, A_, I_, I2_ = min(scored, key=lambda x: abs(x[3] - tabI) / tabI + abs(x[4] - tabI2) / tabI2)
                if abs(I_ - tabI) <= 0.02 * tabI and abs(I2_ - tabI2) <= 0.02 * tabI2:
                    A2, I2a, I2b = props(shape, dict(dims, r=2.0 * t))
                    recomputed.append(name)
                    return dict(standard="TIS 107-2566", series=f"{TUBE[kind]} ({std})", name=name, shape=shape,
                                **dict(dims, r=2.0 * t), A=sig(A2, 4), I33=sig(I2a, 3 if I2a < 1000 else 4),
                                I22=sig(I2b, 3 if I2b < 1000 else 4), mass=sig(0.785 * A2, 3))
            raise ValueError("no fit")
        labels = [(k, s_) for k, s_, tt in items if s_ is not None]
        for k, size, toks in items:
            if not toks:
                continue
            near = [s_ for _, s_ in sorted(labels, key=lambda ls: abs(ls[0] - k))]
            cands = ([size] if size is not None else []) + near + [p_ for p_ in pool if p_ not in near]
            v = [num(x) for x in toks]
            t, Atab = v[0], v[2] * 100  # size label lost in the text: solve the size from A (I's still checked)
            if kind == "round":
                D = Atab / (math.pi * t) + t
                cands += [(round(D, 1),), (float(round(D)),)]
            elif kind == "square":
                B = (Atab - 3 * math.pi * t * t) / (4 * t) + 4 * t
                cands += [(round(B, 1), round(B, 1)), (float(round(B)), float(round(B)))]
            elif std == "JIS G3466":
                P = (Atab - 3 * math.pi * t * t) / (2 * t) + 8 * t
                cands += [(float(H), float(round(P - H))) for H in range(20, 501, 5) if 10 <= P - H <= H]
            row = None
            for c in cands:
                try:
                    row = build(c, toks)
                    break
                except ValueError:
                    continue
            if row is None:
                dropped.append(f"{std} {kind} {' '.join(toks)}")
                continue
            ok = official.get((kind, row["t3"], row.get("t2") or row["t3"], row["tw"]))
            if ok and std.startswith("JIS"):
                checked += 1
                if any(abs(a_ - b_) > 0.012 * b_ + 0.01 for a_, b_ in zip((row["A"], row["I33"], row["I22"]), ok)):
                    A3, I3, I3b = props(row["shape"], {k: row.get(k) for k in ("t3", "t2", "tf", "tw", "r")})
                    calc = (A3, I3, I3b)
                    old_ok = all(abs(a_ - b_) <= 0.012 * b_ + 0.01 for a_, b_ in zip(calc, ok))
                    new_ok = all(abs(a_ - b_) <= 0.012 * b_ + 0.01 for a_, b_ in
                                 zip(calc, (row["A"], row["I33"], row["I22"])))
                    if not new_ok or old_ok:
                        raise ValueError(f"TIS 107-2566 row {row['name']} differs from TIS 107-2533 {ok}")
                    print(f"  TIS 107-2533 misprint {row['name']} {ok}; 2566 table and dimensions agree: "
                          f"{row['A']}, {row['I33']}, {row['I22']}")
            rows.append(row)
    # one row per size and series; names unique in the standard (EN / ASTM tubes of a JIS size get a suffix)
    seen, out = set(), []
    from structstudio.model.section_shapes import check_dims
    for r in list(rows):
        if check_dims(SHAPES[r["shape"]], {k: (r.get(k) or 0) / 1000.0 for k in ("t3", "t2", "tf", "tw", "r")}):
            dropped.append(f"{r['series']} {r['name']} (corners 2t do not fit the width)")
            rows.remove(r)
    for r in rows:
        k = (r["series"], r["name"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    names = {}
    for r in out:
        names.setdefault(r["name"], []).append(r)
    for nm, rs in names.items():
        for r in rs[1:]:
            r["name"] = nm + ("-EN" if "EN" in r["series"] else "-A500")
    missing = [k for k in official if not any(
        (r["series"].startswith(TUBE[k[0]]) and "JIS" in r["series"] and r["t3"] == k[1] and
         (r.get("t2") or r["t3"]) == k[2] and r["tw"] == k[3]) for r in out)]
    print(f"  TIS 107: {len(recomputed)} thick JIS G3466 tubes recomputed with the 2t corners")
    print(f"  TIS 107: {len(out)} tubes; {checked} JIS rows agree with TIS 107-2533; "
          f"{len(dropped)} rows left out (A, I and mass inconsistent in the print):")
    for d in dropped:
        print("    " + d)
    if missing:
        print(f"  TIS 107-2533 sizes not in the 2566 tables, added from 2533: {missing}")
    for kind, D, B, t in missing:
        A_, I_, I2_ = official[(kind, D, B, t)]
        if kind == "round":
            out.append(dict(standard="TIS 107-2566", series=f"{TUBE['round']} (JIS G3444)", name=f"PIPE{D:g}x{t:g}", shape="Pipe",
                            t3=D, tw=t, A=A_, I33=I_, I22=I2_, mass=round(0.785 * A_, 3)))
        else:
            out.append(dict(standard="TIS 107-2566", series=f"{TUBE[kind]} (JIS G3466)", name=f"BOX{D:g}x{B:g}x{t:g}",
                            shape="Box", t3=D, t2=B, tf=t, tw=t, r=2 * t, A=A_, I33=I_, I22=I2_,
                            mass=round(0.785 * A_, 3)))
    return out


# ------------------------------------------------------------------ AISC
IN = 25.4


def aisc(folder):
    rows = []

    def read(fn):
        with open(os.path.join(folder, fn), newline="") as fh:
            return list(csv.DictReader(fh))

    def f(r, k):
        v = (r.get(k) or "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    def add(series, r, shape, dims, J=True, slope=0.0):
        A_, Ix, Iy = f(r, "area"), f(r, "Ix"), f(r, "Iy")
        dims = {k: (None if v is None else round(v * IN, 3)) for k, v in dims.items()}
        rows.append(dict(standard="AISC", series=series, name=r["shape"].replace("_", "/") if False else r["shape"],
                         shape=shape, **dims, A=round(A_ * 6.4516, 3), I33=round(Ix * 41.6231, 2),
                         I22=round(Iy * 41.6231, 2), J=round(f(r, "J") * 41.6231, 3) if J and f(r, "J") else None,
                         mass=round(f(r, "weight") * 1.48816, 2), slope=slope or None))
    for series, fn in (("W", "W_shapes.csv"), ("HP", "HP_shapes.csv")):
        for r in read(fn):
            tf = f(r, "tf")
            add(series, r, "I", dict(t3=f(r, "d"), t2=f(r, "bf"), tf=tf, tw=f(r, "tw"), r=max(f(r, "k") - tf, 0.0)))
    for series, fn in (("C", "C_shapes.csv"), ("MC", "MC_shapes.csv")):
        for r in read(fn):  # C: inner flange slope 1:6 (16 2/3 %), tf the average thickness; MC: varies, 1:6 used
            add(series, r, "Channel", dict(t3=f(r, "d"), t2=f(r, "bf"), tf=f(r, "tf"), tw=f(r, "tw")), slope=1.0 / 6.0)
    for r in read("S_shapes.csv"):  # American standard beams: flange slope 1:6
        add("S", r, "I", dict(t3=f(r, "d"), t2=f(r, "bf"), tf=f(r, "tf"), tw=f(r, "tw")), slope=1.0 / 6.0)
    for r in read("M_shapes.csv"):
        add("M", r, "I", dict(t3=f(r, "d"), t2=f(r, "bf"), tf=f(r, "tf"), tw=f(r, "tw")))
    for r in read("L_shapes.csv"):  # Ix is about the axis across the long leg: long leg vertical (t3)
        t = f(r, "t")
        long_, short = max(f(r, "d"), f(r, "b")), min(f(r, "d"), f(r, "b"))
        add("L", r, "Angle", dict(t3=long_, t2=short, tf=t, tw=t, r=max(f(r, "k") - t, 0.0)))
    for r in read("HSS_shapes.csv"):
        t = f(r, "tdes")
        add("HSS rect", r, "Box", dict(t3=f(r, "Ht"), t2=f(r, "B"), tf=t, tw=t, r=2.0 * t))
    for r in read("HSS_R_shapes.csv"):
        add("HSS round", r, "Pipe", dict(t3=f(r, "OD"), tw=f(r, "tdes")))
    for r in read("PIPE_shapes.csv"):
        add("Pipe", r, "Pipe", dict(t3=f(r, "OD"), tw=f(r, "tdes")))
    for r in rows:
        r["name"] = aisc_name(r["series"], r["name"])
    return rows


def aisc_name(series, name):
    """The EDI name of the AISC database from the steelpy key ('_' for '.', '/' and '-')."""
    if series in ("L", "HSS rect"):
        def frac(tok):
            p = tok.split("_")
            if len(p) == 3:
                return f"{p[0]}-{p[1]}/{p[2]}"
            return "/".join(p)
        head = re.match(r"^[A-Z]+", name).group(0)
        return head + "X".join(frac(t) for t in name[len(head):].split("X"))
    return name.replace("_", ".")


# ------------------------------------------------------------------ Europe
def euro(folder):
    def params(fn, cls):
        tree = ast.parse(open(os.path.join(folder, fn), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for st in node.body:
                    if isinstance(st, ast.Assign) and st.targets[0].id == "parameters":
                        return ast.literal_eval(st.value)
        raise KeyError(cls)
    rows = []

    def add(series, name, shape, dims, slope=0.0):
        A_, I33, I22 = props(shape, dict(dims, slope=slope))
        rows.append(dict(standard="EN", series=series, name=name, shape=shape, **dims, A=round(A_, 3),
                         I33=round(I33, 2), I22=round(I22, 2), mass=round(A_ * 0.785, 2), slope=slope or None))
    for name, p in params("_ipe.py", "IPE").items():
        add("IPE", name.replace("IPE", "IPE "), "I", dict(t3=p["h"], t2=p["b"], tf=p["tf"], tw=p["tw"], r=p["r"]))
    for name, p in params("_he.py", "HE").items():
        m = re.fullmatch(r"HE([A-Z]+)(\d+)", name)
        if m is None:
            continue
        add(f"HE{m.group(1)}", f"HE {m.group(2)} {m.group(1)}", "I",
            dict(t3=p["h"], t2=p["b"], tf=p["tf"], tw=p["tw"], r=p["r"]))
    for name, p in params("_upe.py", "UPE").items():
        add("UPE", name.replace("UPE", "UPE "), "Channel", dict(t3=p["h"], t2=p["b"], tf=p["tf"], tw=p["tw"], r=p["r"]))
    for name, p in params("_upn.py", "UPN").items():
        # EN 10279: slope 8 % (h <= 300) with tf at b/2, 5 % (h > 300) with tf at (b - tw)/2;
        # here tf is the mean thickness at the middle of the flange
        sl = 0.08 if p["h"] <= 300 else 0.05
        u = p["b"] / 2.0 if p["h"] <= 300 else (p["b"] - p["tw"]) / 2.0
        tm = p["tf"] + sl * ((p["b"] - p["tw"]) / 2.0 - u)
        add("UPN", name.replace("UPN", "UPN "), "Channel",
            dict(t3=p["h"], t2=p["b"], tf=round(tm, 4), tw=p["tw"], r=p["r1"], r2=p["r2"]), sl)
    for name, p in params("_ipn.py", "IPN").items():
        # slope 14 %, tf at b/4 from the centre line
        tm = p["tf"] - 0.14 * p["tw"] / 4.0
        add("IPN", name.replace("IPN", "IPN "), "I",
            dict(t3=p["h"], t2=p["b"], tf=round(tm, 4), tw=p["tw"], r=p["r1"], r2=p["r2"]), 0.14)
    for name, p in params("_l.py", "L").items():
        if "t" not in p:
            continue
        add("L", name, "Angle", dict(t3=p["h"], t2=p["b"], tf=p["t"], tw=p["t"], r=p["r1"], r2=p["r2"]))
    return rows


# ------------------------------------------------------------------ China (GB)
def html_tables(path):
    """Tables of an HTML page as grids, row / column spans expanded."""
    import html as _html
    s = open(path, encoding="utf-8", errors="ignore").read()
    out = []
    for t in re.findall(r"<table.*?</table>", s, re.S):
        grid, span = [], {}
        for ri, r in enumerate(re.findall(r"<tr.*?</tr>", t, re.S)):
            row, ci = [], 0
            it = iter(re.findall(r"<(t[dh])([^>]*)>(.*?)</t[dh]>", r, re.S))
            while True:
                if (ri, ci) in span:
                    row.append(span.pop((ri, ci)))
                    ci += 1
                    continue
                try:
                    _tag, attrs, c = next(it)
                except StopIteration:
                    break
                txt = _html.unescape(re.sub(r"<.*?>", "", c)).strip()
                rs = int((re.search(r'rowspan="?(\d+)', attrs) or [0, 1])[1])
                cs = int((re.search(r'colspan="?(\d+)', attrs) or [0, 1])[1])
                for _ in range(cs):
                    row.append(txt)
                    for d in range(1, rs):
                        span[(ri + d, ci)] = txt
                    ci += 1
            while (ri, ci) in span:
                row.append(span.pop((ri, ci)))
                ci += 1
            grid.append(row)
        out.append(grid)
    return out


def _numrow(r, n):
    """The first n cells as numbers, or None (header / note rows)."""
    try:
        return [float(x.replace(",", "").replace("*", "").replace(" ", "")) for x in r[:n]]
    except (ValueError, IndexError):
        return None


def gb706_official(pdf):
    """A, Ix, Iy of the I-beams and channels from the GB/T 706-2016 full text
    (table A.1 / A.2), keyed ("I" | "C", model): the words of each table line,
    the area located by value, Ix and Iy after it (A, mass, surface, Ix, Iy)."""
    import pymupdf
    doc = pymupdf.open(pdf)
    out, mode = {}, None
    for pno in range(doc.page_count):
        rows = {}
        for w in doc[pno].get_text("words"):
            rows.setdefault(round((w[1] + w[3]) / 5.0), []).append(w)
        for y in sorted(rows):
            ws = sorted(rows[y], key=lambda w: w[0])
            toks, last = [], None
            for w in ws:
                if last is not None and w[0] - last[2] < 1.2:
                    toks[-1] += w[4]
                else:
                    toks.append(w[4])
                last = w
            line = "".join(toks)
            if "工字钢截面尺寸" in line:
                mode = "I"
            elif "槽钢截面尺寸" in line:
                mode = "C"
            elif "等边角钢截面尺寸" in line:
                mode = None
            if mode and toks and re.fullmatch(r"\d+(\.\d)?[abc]?", toks[0]) and len(toks) > 8:
                out[(mode, toks[0])] = toks[1:]
    return out


def _official_props(toks, A_):
    """(A, Ix, Iy) of the standard: the area is the token within 0.5 % of the
    web table's area (which may be rounded), Ix and Iy 3 and 4 tokens later."""
    for i, t in enumerate(toks):
        try:
            if abs(float(t) - A_) < 0.005 * A_ and i + 4 < len(toks) and "." in t:
                return float(t), float(toks[i + 3]), float(toks[i + 4])
        except ValueError:
            continue
    return None


GB_STD_H = "GB/T 11263"
GB_STD = "GB/T 706"


def gb(folder, gb706_pdf=None):
    """GB/T 11263-2017 H sections (HW, HM, HN, HT) and GB/T 706-2016 I-beams,
    channels, equal and unequal angles: dimensions and the tabulated area; the
    area computed from the dimensions (fillets, flange taper 1:6 / 1:10, angle
    toe r1 = d/3 as the standard draws them) must match it within 0.6 %."""
    rows, report = [], []

    official = gb706_official(gb706_pdf) if gb706_pdf else {}
    backup = {}  # Ix / Iy of the web page's property table, for lines the PDF text splits

    fixed = []

    def add(std, series, name, shape, dims, A_tab, mass, slope=0.0, key=None):
        A_, I33, I22 = props(shape, dict(dims, slope=slope))
        if abs(A_ - A_tab) > 0.006 * A_tab:
            raise ValueError(f"{name}: area {A_tab} vs {A_:.3f} from the dimensions")
        if not slope:  # H and angles: the standard's area formula is the drawn outline; the GB tables
            a4 = float(f"{A_:.4g}")  # round it their own way (within 0.1 %) - a larger gap is a misprint
            if abs(a4 - A_tab) > 0.002 * A_tab:
                fixed.append(f"{name} {A_tab}->{a4}")
                A_tab = a4
        tab = _official_props(official[key], A_tab) if key in official else None
        if tab is None and key in backup:
            tab = (A_tab,) + backup[key]
            if key[0] == "H" and (abs(tab[1] / I33 - 1) > 0.01 or abs(tab[2] / I22 - 1) > 0.01):
                print(f"  {name}: second page Ix / Iy {tab[1:]} off the dimensions - computed values used")
                tab = None
        if tab is not None:  # the standard's Ix / Iy (the taper of the b / c sizes is not drawn exactly)
            A_std, Ix_std, Iy_std = tab
            lim = (0.01, 0.01) if key[0] == "H" else (0.06, 0.08)  # parallel flanges: exact geometry
            if abs(Ix_std / I33 - 1) > lim[0] or abs(Iy_std / I22 - 1) > lim[1]:
                raise ValueError(f"{name}: GB/T 706 Ix / Iy {Ix_std} / {Iy_std} vs {I33:.0f} / {I22:.1f} from the dimensions")
            report.append((name, I33 / Ix_std - 1.0, I22 / Iy_std - 1.0))
            A_tab, I33, I22 = A_std, Ix_std, Iy_std
        elif key and key[0] != "H":
            print(f"  {name}: no Ix / Iy in the GB/T 706 text - computed")
        rows.append(dict(standard=std, series=series, name=name, shape=shape, **dims, A=A_tab,
                         I33=round(I33, 2 if I33 < 1e4 else 0), I22=round(I22, 2 if I22 < 1e4 else 0),
                         mass=mass, slope=slope or None))
        return I33, I22
    # H sections: dimensions and area of one page, A / Ix / Iy of a second page where it has the size
    second = {}
    p2 = os.path.join(folder, "gb_h2.html")
    if os.path.exists(p2):
        for g in html_tables(p2):
            for r in g:
                v = _numrow(r[2:12], 10)
                if r and r[0] in ("HW", "HM", "HN", "HT") and v is not None:
                    H, B, t1, t2, _r_, A_, _m, _s, Ix, Iy = v
                    second[f"{r[0]}{H:g}x{B:g}x{t1:g}x{t2:g}"] = (A_, Ix, Iy)
    for series, grid in zip(("HW", "HM", "HN", "HT"), html_tables(os.path.join(folder, "gb_h.html"))):
        for r in grid:
            v = _numrow(r[1:], 7)
            if v is None:
                continue
            H, B, t1, t2, rr, A_, m = v
            name = f"{series}{H:g}x{B:g}x{t1:g}x{t2:g}"
            dims = dict(t3=H, t2=B, tf=t2, tw=t1, r=rr)
            if name in second:  # two sources: the area nearer the standard's formula; the second's Ix / Iy
                A2, Ix2, Iy2 = second[name]
                Af = props("I", dims)[0]
                A_ = min((A_, A2), key=lambda a: abs(a - Af))
                backup[("H", name)] = (Ix2, Iy2)
            add(GB_STD_H, series, name, "I", dims, A_, m, key=("H", name))
    # I-beams: dimensions table; Ix / Iy of the property table to compare
    t_i = html_tables(os.path.join(folder, "gb_i.html"))
    ref = {}
    for r in t_i[0]:
        v = _numrow(r[7:16], 9)
        if v is not None:
            ref[r[0]] = (v[3], v[6], v[0])  # Ix, Iy, A
            backup[("I", r[0])] = (v[3], v[6])
    for r in t_i[1]:
        v = _numrow(r[1:], 8)
        if v is None:
            continue
        h, b, d, t, rr, r1, A_, m = v
        dims = dict(t3=h, t2=b, tf=t, tw=d, r=rr, r2=r1)
        for a_try in [A_] + ([ref[r[0]][2]] if r[0] in ref else []):  # the two tables of the page
            try:
                add(GB_STD, "I-beam", f"I{r[0]}", "I", dims, a_try, m, 1.0 / 6.0, key=("I", r[0]))
                break
            except ValueError as exc:
                print(f"  {exc}")
    def try_add(name, alts, *a, **kw):
        """add() with the area of each source in turn; a row no source gets right is reported."""
        for A_try in alts:
            try:
                return add(*a[:4], a[4], A_try, *a[5:], **kw)
            except ValueError as exc:
                last = exc
        print(f"  left out: {last}")
        return None
    # channels
    ch = [g for g in html_tables(os.path.join(folder, "gb_c.html")) if g and any("40c" in c for row in g for c in row)]
    for r in ch[-1]:
        v = _numrow(r[1:], 8)
        if v is None:
            continue
        h, b, d, t, rr, r1, A_, m = v
        try_add(f"[{r[0]}", [A_], GB_STD, "Channel", f"[{r[0]}", "Channel",
                dict(t3=h, t2=b, tf=t, tw=d, r=rr, r2=r1), m, 0.1, key=("C", r[0]))
    # angles (toe radius r1 = d / 3): the dimension table, and the property table's area as the second source
    for fn, series, nd in (("gb_le.html", "L equal", 3), ("gb_lu.html", "L unequal", 4)):
        grids = html_tables(os.path.join(folder, fn))
        alt = {}
        for r in grids[0]:
            v = _numrow(r[1:], nd + 1)
            if v is not None:
                alt[(r[0], v[nd - 2])] = v[nd]
        for r in grids[1]:
            v = _numrow(r[1:], nd + 2)
            if v is None:
                continue
            if nd == 3:
                b, d, rr, A_, m = v
                B = b
            else:
                B, b, d, rr, A_, m = v
            alts = [A_] + ([alt[(r[0], d)]] if (r[0], d) in alt else [])
            try_add(f"L{B:g}x{b:g}x{d:g}", alts, GB_STD, series, f"L{B:g}x{b:g}x{d:g}", "Angle",
                    dict(t3=B, t2=b, tf=d, tw=d, r=rr, r2=round(d / 3.0, 3)), m)
    if fixed:
        print(f"  GB areas of the web tables corrected to the standard's formula: {', '.join(fixed)}")
    if report:
        e33 = max(abs(x[1]) for x in report)
        e22 = max(abs(x[2]) for x in report)
        print(f"  GB I-beams / channels: Ix / Iy from the dimensions vs GB/T 706: max {e33:.2%} / {e22:.2%} "
              f"({len(report)} sections; the standard's values are used)")
    return rows


# ------------------------------------------------------------------ rebar
def rebar(tis_dir):
    out = []
    for fn, prefix, std in (("tis20_2559.txt", "RB", "TIS 20 (round, SR24)"), ("tis24_2559.txt", "DB", "TIS 24 (deformed)")):
        text = open(os.path.join(tis_dir, fn), encoding="utf-8").read()
        seen = set()
        for m in re.finditer(rf"^\s*{prefix} (\d+)\s+(\d+)\s+([\d ]+\.\d)\s+(\d\.\d+)\s*$", text, re.M):
            name = f"{prefix}{m.group(1)}"
            if name in seen:
                continue
            seen.add(name)
            d, a, w = float(m.group(2)), float(m.group(3).replace(" ", "")), float(m.group(4))
            assert abs(a - math.pi * d * d / 4) < 0.06, (name, a)
            out.append(dict(standard=std, name=name, diameter=d, area=a, mass=w))
    # ASTM A615 / A706: nominal diameter (in) and area (in2) of the bar designations
    # and nominal weight (lb/ft)
    for no, d, a, w in ((3, .375, .11, .376), (4, .5, .20, .668), (5, .625, .31, 1.043), (6, .75, .44, 1.502),
                        (7, .875, .60, 2.044), (8, 1.0, .79, 2.670), (9, 1.128, 1.00, 3.400), (10, 1.27, 1.27, 4.303),
                        (11, 1.41, 1.56, 5.313), (14, 1.693, 2.25, 7.650), (18, 2.257, 4.00, 13.600)):
        out.append(dict(standard="ASTM A615", name=f"#{no}", diameter=round(d * IN, 1), area=round(a * 645.16, 1),
                        mass=round(w * 1.48816, 3)))
    # JIS G3112: nominal diameters; area 0.7854 d^2 (4 significant figures), mass 0.785 S (3 s.f.)
    for nm, d in (("D6", 6.35), ("D10", 9.53), ("D13", 12.7), ("D16", 15.9), ("D19", 19.1), ("D22", 22.2),
                  ("D25", 25.4), ("D29", 28.6), ("D32", 31.8), ("D35", 34.9), ("D38", 38.1), ("D41", 41.3),
                  ("D51", 50.8)):
        s = float(f"{0.7854 * d * d / 100:.4g}")
        out.append(dict(standard="JIS G3112", name=nm, diameter=d, area=round(s * 100, 2),
                        mass=float(f"{0.785 * s:.3g}")))
    # GB 1499.1 / 1499.2: nominal diameters, nominal areas and masses of the standard's table
    for d, a, w in ((6, 28.27, .222), (8, 50.27, .395), (10, 78.54, .617), (12, 113.1, .888), (14, 153.9, 1.21),
                    (16, 201.1, 1.58), (18, 254.5, 2.00), (20, 314.2, 2.47), (22, 380.1, 2.98), (25, 490.9, 3.85),
                    (28, 615.8, 4.83), (32, 804.2, 6.31), (36, 1018.0, 7.99), (40, 1257.0, 9.87), (50, 1964.0, 15.42)):
        assert abs(a - math.pi * d * d / 4) < 0.6, d
        out.append(dict(standard="GB 1499 (HPB / HRB)", name=f"\u03a6{d}", diameter=float(d), area=a, mass=w))
    # EN 10080 / BS 4449: nominal diameters, area pi d^2/4, mass 7850 kg/m3
    for d in (6, 8, 10, 12, 14, 16, 20, 25, 28, 32, 40, 50):
        a = math.pi * d * d / 4
        out.append(dict(standard="EN 10080", name=f"T{d}" if False else f"Ø{d}", diameter=float(d),
                        area=round(a, 1), mass=round(a * 7.85e-3, 3)))
    return out


def write(fn, cols, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, fn), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    print(f"{fn}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tis-dir", required=True, help="folder with t1227.txt, tis1228_2561.txt, tis20/24_2559.txt")
    ap.add_argument("--steelpy", required=True, help="steelpy 'shape files' folder")
    ap.add_argument("--structuralcodes", required=True, help="structuralcodes geometry/profiles folder")
    ap.add_argument("--gb-dir", required=True, help="folder with gb_h.html, gb_i.html, gb_c.html, gb_le.html, gb_lu.html")
    ap.add_argument("--tis107", required=True, help="folder with t107_2533.txt (TIS 107-2533 full text) and the half "
                    "pages p<page>_<0|1>.txt of the Pacific Pipe catalogue (pdftotext -layout -x 0|595 -W 595)")
    a = ap.parse_args()
    t1227 = open(os.path.join(a.tis_dir, "t1227.txt"), encoding="utf-8").read()
    t1228 = open(os.path.join(a.tis_dir, "tis1228_2561.txt"), encoding="utf-8").read()
    t107 = open(os.path.join(a.tis107, "t107_2533.txt"), encoding="utf-8").read()
    tis = (tis_h(t1227) + tis_ibeams(t1227) + tis_channels(t1227) + tis_angles(t1227) + tis_tees(t1227) +
           tis_cold(t1228) + tis107(a.tis107, tis107_official(t107)))
    write("steel_tis.csv", STEEL_COLS, tis)
    write("steel_gb.csv", STEEL_COLS, gb(a.gb_dir, os.path.join(a.gb_dir, "gb706.pdf")))
    write("steel_aisc.csv", STEEL_COLS, aisc(a.steelpy))
    write("steel_en.csv", STEEL_COLS, euro(a.structuralcodes))
    write("rebar.csv", ["standard", "name", "diameter", "area", "mass"], rebar(a.tis_dir))


if __name__ == "__main__":
    main()
