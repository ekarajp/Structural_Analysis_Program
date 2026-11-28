"""CAD references (References menu): DXF drawings and PDF sheets placed on plans /
elevations to draw over.

* the DXF reader (``structstudio.io.dxf_read``): the entities of a building drawing as
  plain segments, blocks put in place (nested, turned, mirrored), layers and colours
  inherited as in CAD, the drawing unit, what is refused and why;
* a reference in the model: where its picture lies, align / scale / move / turn keep the
  right points fixed, saved with the model (linked and / or embedded), units, undo;
* it is no part of the structure: placing or changing one keeps the analysis results;
* snapping to it;
* the PDF reader (``structstudio.io.pdf_read``): a page of a PDF plotted from CAD as line
  work in millimetres on the sheet - strokes, curves, colours, pens and layers, forms,
  the page as it is viewed, what is refused - and a PDF as a reference (page, plot scale).

The window part (the menu, placing through it, what each window shows, the tools) runs
in a child interpreter, as the other window tests."""
import math
import os
import pickle
import subprocess
import sys
import zlib

import numpy as np
import pytest

from structstudio.app import cadref
from structstudio.app.document import Document
from structstudio.app.templates import new_building
from structstudio.io.dxf_read import PT_CENTER, PT_END, PT_INSERT, DxfError, aci_rgb, read_dxf
from structstudio.io.pdf_read import PT_TO_MM, is_pdf, read_pdf
from structstudio.model import Model, ModelError
from structstudio.model.references import Blob, CadReference, align_two_points

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"
gui = pytest.mark.skipif(not CHILD, reason="runs in the child interpreter (test_gui_in_subprocess)")
core = pytest.mark.skipif(CHILD, reason="parent only")


# ------------------------------------------------------------ a DXF written by hand
def _g(*pairs) -> str:
    return "".join(f"{c:>3}\n{v}\n" for c, v in pairs)


def _common(layer, color=None, extra=()):
    out = [(8, layer)]
    if color is not None:
        out.append((62, color))
    return out + list(extra)


def line(x1, y1, x2, y2, layer="0", color=None, extra=()):
    return _g((0, "LINE"), *_common(layer, color, extra), (10, x1), (20, y1), (30, 0), (11, x2), (21, y2), (31, 0))


def lwpoly(pts, layer="0", closed=False, color=None, extra=()):
    body = [(0, "LWPOLYLINE"), *_common(layer, color, extra), (90, len(pts)), (70, 1 if closed else 0)]
    for p in pts:
        body += [(10, p[0]), (20, p[1])]
        if len(p) > 2 and p[2]:
            body.append((42, p[2]))
    return _g(*body)


def arc(cx, cy, r, a0, a1, layer="0", extra=()):
    return _g((0, "ARC"), *_common(layer, None, extra), (10, cx), (20, cy), (40, r), (50, a0), (51, a1))


def circle(cx, cy, r, layer="0", color=None):
    return _g((0, "CIRCLE"), *_common(layer, color), (10, cx), (20, cy), (40, r))


def insert(name, x, y, layer="0", color=None, sx=1.0, sy=1.0, rot=0.0, extra=()):
    return _g((0, "INSERT"), *_common(layer, color, extra), (2, name), (10, x), (20, y), (41, sx), (42, sy), (50, rot))


def block(name, body, base=(0, 0)):
    return _g((0, "BLOCK"), (8, "0"), (2, name), (70, 0), (10, base[0]), (20, base[1]), (30, 0), (3, name)) + \
        body + _g((0, "ENDBLK"), (8, "0"))


def layer_def(name, color=7, flags=0):
    return _g((0, "LAYER"), (2, name), (70, flags), (62, color), (6, "CONTINUOUS"))


def dxf(entities="", layers="", blocks="", insunits=None, measurement=1, codepage=None) -> str:
    head = _g((0, "SECTION"), (2, "HEADER"), (9, "$ACADVER"), (1, "AC1015"), (9, "$MEASUREMENT"), (70, measurement))
    if insunits is not None:
        head += _g((9, "$INSUNITS"), (70, insunits))
    if codepage:
        head += _g((9, "$DWGCODEPAGE"), (3, codepage))
    head += _g((0, "ENDSEC"))
    tabs = _g((0, "SECTION"), (2, "TABLES"), (0, "TABLE"), (2, "LAYER"), (70, 8)) + layer_def("0") + layers + \
        _g((0, "ENDTAB"), (0, "ENDSEC"))
    blk = _g((0, "SECTION"), (2, "BLOCKS")) + blocks + _g((0, "ENDSEC"))
    ent = _g((0, "SECTION"), (2, "ENTITIES")) + entities + _g((0, "ENDSEC"))
    return head + tabs + blk + ent + _g((0, "EOF"))


def _segs(d, layer=None):
    """The segments as a sorted list of rounded ((x1, y1), (x2, y2)) with the lower end first."""
    out = []
    for k, s in enumerate(d.seg):
        if layer is not None and d.layers[d.seg_layer[k]] != layer:
            continue
        a, b = (round(float(s[0]), 6), round(float(s[1]), 6)), (round(float(s[2]), 6), round(float(s[3]), 6))
        out.append((min(a, b), max(a, b)))
    return sorted(out)


def _floor_plan(unit=4) -> str:
    """A 2 x 1 bay plan in mm: grid lines, 400 mm columns as a block, a wall, a round column."""
    layers = layer_def("GRID", 1) + layer_def("COLUMN", 3) + layer_def("WALL", 5) + layer_def("NOTES", -2) + \
        layer_def("FROZEN", 4, flags=1)
    col = block("COL", lwpoly([(-200, -200), (200, -200), (200, 200), (-200, 200)], closed=True))
    ents = ""
    for x in (0, 6000, 12000):
        ents += line(x, -1000, x, 6000, "GRID")
        for y in (0, 5000):
            ents += insert("COL", x, y, "COLUMN")
    for y in (0, 5000):
        ents += line(-1000, y, 13000, y, "GRID")
    ents += lwpoly([(0, 200), (0, 4800)], "WALL") + circle(9000, 2500, 300, "COLUMN") + \
        line(100, 100, 200, 200, "NOTES") + line(0, 0, 50, 50, "FROZEN")
    return dxf(ents, layers, col, insunits=unit)


# ------------------------------------------------------------------- the DXF reader
@core
def test_lines_layers_colours_and_the_unit():
    d = read_dxf(_floor_plan().encode())
    assert d.unit == "mm" and d.guess_unit() == "mm" and d.version == "AC1015"
    assert d.layers[:1] == ["0"] and {"GRID", "COLUMN", "WALL", "NOTES", "FROZEN"} <= set(d.layers)
    assert sorted(d.layers_off) == ["FROZEN", "NOTES"]  # off (negative colour) and frozen
    assert _segs(d, "GRID") == sorted([((x, -1000.0), (x, 6000.0)) for x in (0.0, 6000.0, 12000.0)] +
                                      [((-1000.0, y), (13000.0, y)) for y in (0.0, 5000.0)])
    assert d.extents == (-1000.0, -1000.0, 13000.0, 6000.0) and d.size() == (14000.0, 7000.0)
    grid = d.layers.index("GRID")
    assert {tuple(c) for c in d.seg_rgb[d.seg_layer == grid].tolist()} == {(255, 0, 0)}  # BYLAYER: the layer's red
    assert d.layer_counts()["COLUMN"] == 6 * 4 + 48  # six columns of four sides, the circle as 48 chords
    assert aci_rgb(1) == (255, 0, 0) and aci_rgb(5) == (0, 0, 255) and aci_rgb(8) == (128, 128, 128)


@core
def test_a_block_goes_where_it_is_inserted_on_the_layer_of_the_insert():
    d = read_dxf(_floor_plan().encode())
    col = d.layers.index("COLUMN")
    straight = d.seg[(d.seg_layer == col) & ~d.seg_curve]
    assert len(straight) == 24
    # the column at (6000, 5000): its four corners
    c = straight[(np.abs(straight[:, [0, 2]] - 6000).max(axis=1) <= 200) & (np.abs(straight[:, [1, 3]] - 5000).max(axis=1) <= 200)]
    assert {(float(x), float(y)) for x, y in np.vstack([c[:, :2], c[:, 2:]])} == \
        {(5800.0, 4800.0), (6200.0, 4800.0), (6200.0, 5200.0), (5800.0, 5200.0)}
    # (drawn on layer "0" inside the block: green as layer COLUMN)
    assert {tuple(v) for v in d.seg_rgb[(d.seg_layer == col) & ~d.seg_curve].tolist()} == {(0, 255, 0)}
    ins = d.pts[d.pts_kind == PT_INSERT]
    assert sorted(map(tuple, ins.tolist())) == sorted((float(x), float(y)) for x in (0, 6000, 12000) for y in (0, 5000))
    assert [tuple(p) for p in d.pts[d.pts_kind == PT_CENTER].tolist()] == [(9000.0, 2500.0)]


@core
def test_blocks_turned_scaled_mirrored_and_nested_colours_by_block():
    inner = block("ARROW", line(0, 0, 100, 0) + line(100, 0, 100, 50, "DETAIL", color=0))  # second line: BYBLOCK
    outer = block("PAIR", insert("ARROW", 0, 0) + insert("ARROW", 0, 200, "WALL", color=1), base=(0, 0))
    ents = insert("ARROW", 1000, 1000, "A", rot=90.0, sx=2.0) + \
        insert("ARROW", 0, 0, "A", extra=((210, 0), (220, 0), (230, -1))) + \
        insert("PAIR", 5000, 0, "B", color=3)
    d = read_dxf(dxf(ents, layer_def("A", 2) + layer_def("B", 4) + layer_def("WALL", 5) + layer_def("DETAIL", 6),
                     inner + outer).encode())
    # turned 90 degrees and twice as long: (0,0)-(100,0) -> (1000,1000)-(1000,1200); (100,0)-(100,50) -> (1000,1200)-(950,1200)
    assert ((1000.0, 1000.0), (1000.0, 1200.0)) in _segs(d, "A")
    assert ((950.0, 1200.0), (1000.0, 1200.0)) in _segs(d, "DETAIL")
    # mirrored (extrusion 0, 0, -1): x -> -x
    assert ((-100.0, 0.0), (0.0, 0.0)) in _segs(d, "A") and ((-100.0, 0.0), (-100.0, 50.0)) in _segs(d, "DETAIL")
    # nested: the layer-0 line of the first inner insert is on B (the outer insert's layer),
    # of the second on WALL (its own insert's layer)
    assert ((5000.0, 0.0), (5100.0, 0.0)) in _segs(d, "B") and ((5000.0, 200.0), (5100.0, 200.0)) in _segs(d, "WALL")

    def rgb_of(a, b):
        k = next(i for i, s in enumerate(d.seg) if {(round(s[0]), round(s[1])), (round(s[2]), round(s[3]))} == {a, b})
        return tuple(d.seg_rgb[k].tolist())
    # BYBLOCK: the colour of the insert - layer A (yellow) / red given to the insert / the
    # outer insert's green handed down through an insert that is itself BYBLOCK ... BYLAYER
    assert rgb_of((1000, 1200), (950, 1200)) == (255, 255, 0)
    assert rgb_of((5100, 200), (5100, 250)) == (255, 0, 0)
    assert rgb_of((5100, 0), (5100, 50)) == (0, 255, 255)  # insert BYLAYER on "0" in PAIR -> PAIR's layer B (cyan)
    assert rgb_of((5000, 0), (5100, 0)) == (0, 255, 255)


@core
def test_curves_are_chords_with_their_ends_and_centres_to_snap_to():
    ents = arc(0, 0, 1000, 0, 90, "A") + lwpoly([(2000, 0, 1.0), (4000, 0)], "A") + \
        arc(0, 0, 1000, 0, 90, "M", extra=((210, 0), (220, 0), (230, -1)))
    d = read_dxf(dxf(ents, layer_def("A") + layer_def("M")).encode())
    a = d.seg[d.seg_layer == d.layers.index("A")]
    assert d.seg_curve.all()
    r = np.hypot(a[:12, 0], a[:12, 1])
    assert len(a) == 12 + 24 and np.allclose(r, 1000.0)  # 90 degrees in 7.5 degree chords; a half circle
    half = a[12:]
    assert np.allclose(np.hypot(half[:, 0] - 3000.0, half[:, 1]), 1000.0) and half[:, 1].max() < 1e-9 and \
        abs(half[:, 1].min() + 1000.0) < 1e-6  # bulge 1 = a half circle to the left of 2000 -> 4000: below
    ends = sorted(map(tuple, np.round(d.pts[d.pts_kind == PT_END], 6).tolist()))
    assert ends == sorted([(1000.0, 0.0), (0.0, 1000.0), (2000.0, 0.0), (4000.0, 0.0), (-1000.0, 0.0), (0.0, 1000.0)])
    assert sorted(map(tuple, np.round(d.pts[d.pts_kind == PT_CENTER], 6).tolist())) == [(0.0, 0.0), (0.0, 0.0), (3000.0, 0.0)]
    m = d.seg[d.seg_layer == d.layers.index("M")]
    assert m[:, [0, 2]].max() < 1e-9 and m[:, [0, 2]].min() > -1000.0 - 1e-9  # the mirrored arc is on the -x side


@core
def test_what_is_not_line_work_is_counted_and_the_paper_space_left_out():
    ents = line(0, 0, 10, 0) + _g((0, "TEXT"), (8, "0"), (10, 0), (20, 0), (40, 2.5), (1, "A")) + \
        _g((0, "HATCH"), (8, "0"), (10, 0), (20, 0)) + _g((0, "MTEXT"), (8, "0"), (10, 0), (20, 0), (1, "note")) + \
        line(0, 0, 0, 999, extra=((67, 1),))
    d = read_dxf(dxf(ents).encode())
    assert _segs(d) == [((0.0, 0.0), (10.0, 0.0))]
    assert d.skipped == {"TEXT": 1, "HATCH": 1, "MTEXT": 1}
    assert d.unit is None and d.guess_unit() == "m"  # 10 units across: metres
    big = read_dxf(dxf(line(0, 0, 42000, 0)).encode())
    assert big.guess_unit() == "mm" and read_dxf(dxf(line(0, 0, 4200, 0)).encode()).guess_unit() == "cm"
    assert read_dxf(dxf(line(0, 0, 1650, 0), measurement=0).encode()).guess_unit() == "in"


@core
def test_files_that_are_not_ascii_dxf_say_what_to_do(tmp_path):
    for data, word in ((b"AutoCAD Binary DXF\r\n\x1a\x00rest", "binary DXF"), (b"AC1032\x00\x00\x00data", "DWG"),
                       (b"hello\nworld\nthis\nis text\n", "not an ASCII DXF"), (b"", "not a DXF")):
        with pytest.raises(DxfError) as e:
            read_dxf(data)
        assert word in str(e.value), (word, str(e.value))
    with pytest.raises(DxfError) as e:
        read_dxf(str(tmp_path / "missing.dxf"))
    assert "cannot be read" in str(e.value)
    # a Thai layer name in the code page of an older file
    text = dxf(line(0, 0, 1, 0, "เสา"), layer_def("เสา"), codepage="ANSI_874")
    assert "เสา" in read_dxf(text.encode("cp874")).layers
    assert "เสา" in read_dxf(text.encode("utf-8")).layers


@core
def test_the_reader_agrees_with_a_cad_library(tmp_path):
    """The same drawing through ezdxf (when it is installed): every point of one lies on
    the lines of the other, within the sag of a chord."""
    ezdxf = pytest.importorskip("ezdxf")
    from ezdxf import disassemble, path as ezpath
    doc = ezdxf.new("R2010")
    doc.units = 4
    msp = doc.modelspace()
    doc.layers.add("A", color=1)
    blk = doc.blocks.new("C")
    blk.add_lwpolyline([(-200, -200), (200, -200), (200, 200), (-200, 200)], close=True)
    blk.add_circle((0, 0), 50)
    nest = doc.blocks.new("N")
    nest.add_blockref("C", (0, 0))
    nest.add_blockref("C", (3000, 0), dxfattribs={"rotation": 20})
    msp.add_blockref("C", (9000, 9000), dxfattribs={"layer": "A", "rotation": 30, "xscale": 2, "yscale": 0.5})
    msp.add_blockref("N", (0, -8000), dxfattribs={"xscale": -1})
    msp.add_blockref("N", (5000, -8000), dxfattribs={"extrusion": (0, 0, -1), "rotation": 15})
    msp.add_lwpolyline([(0, 0, 0), (6000, 0, 0.5), (6000, 5000, 0), (0, 5000, -1)], format="xyb", close=True)
    msp.add_arc((3000, 8000), 1500, 20, 200)
    msp.add_arc((3000, 8000), 1500, 20, 200, dxfattribs={"extrusion": (0, 0, -1)})
    msp.add_ellipse((16000, 3000), major_axis=(2000, 500), ratio=0.4, start_param=0.3, end_param=4.0)
    msp.add_open_spline([(7000, 9000), (8000, 11000), (10000, 9000), (11000, 12000), (12500, 10000)])
    msp.add_polyline2d([(13000, 0), (14000, 0), (14000, 1000)])
    fn = str(tmp_path / "lib.dxf")
    doc.saveas(fn)
    ref = []
    for e in disassemble.recursive_decompose(ezdxf.readfile(fn).modelspace()):
        pts = [(v.x, v.y) for v in ezpath.make_path(e).flattening(distance=0.5)]
        ref += [(*a, *b) for a, b in zip(pts, pts[1:]) if a != b]
    E, M = np.array(ref, float), read_dxf(fn).seg

    def far(P, S):
        A, B = S[:, :2], S[:, 2:]
        D = B - A
        L2 = np.maximum((D * D).sum(1), 1e-30)
        out = 0.0
        for p in P:
            t = np.clip(((p - A) * D).sum(1) / L2, 0, 1)
            out = max(out, float(np.sqrt((((A + t[:, None] * D) - p) ** 2).sum(1)).min()))
        return out
    mid = lambda S: np.vstack([S[:, :2], S[:, 2:], 0.5 * (S[:, :2] + S[:, 2:])])  # noqa: E731
    # the largest radius is 3125 (the bulge): its 7.5 degree chord sags 6.7
    assert far(mid(M), E) < 7.0 and far(mid(E), M) < 7.0


# -------------------------------------------------------------- a reference's place
@core
def test_where_a_reference_lies_and_back():
    r = CadReference("R", unit="mm", base=(1000.0, 2000.0), origin=(5.0, 7.0), rotation=90.0)
    assert r.to_plane((1000.0, 2000.0), "m") == pytest.approx((5.0, 7.0))
    assert r.to_plane((2000.0, 2000.0), "m") == pytest.approx((5.0, 8.0))  # 1 m along the drawing's x: up
    assert r.to_file((5.0, 8.0), "m") == pytest.approx((2000.0, 2000.0))
    assert r.to_plane((2000.0, 2000.0), "mm") == pytest.approx((5.0, 1007.0))  # (a model in mm: origin is in mm)
    r.scale = 2.0
    assert r.to_plane((2000.0, 2000.0), "m") == pytest.approx((5.0, 9.0))
    # another fixed point: the picture stays
    before = [r.to_plane(p, "m") for p in ((0.0, 0.0), (3000.0, -500.0))]
    r.rebase((11.0, -4.0), "m")
    assert [r.to_plane(p, "m") for p in ((0.0, 0.0), (3000.0, -500.0))] == [pytest.approx(b) for b in before]
    assert r.origin == (11.0, -4.0)


@core
def test_align_two_points():
    r = CadReference("R", unit="mm", origin=(100.0, 50.0), rotation=10.0)
    a, b = r.to_plane((0.0, 0.0), "m"), r.to_plane((6000.0, 0.0), "m")  # two grid crossings of the drawing
    ratio = align_two_points(r, "m", a, (0.0, 0.0), b, (0.0, 6.0))
    assert ratio == pytest.approx(1.0)
    assert r.to_plane((0.0, 0.0), "m") == pytest.approx((0.0, 0.0), abs=1e-9)
    assert r.to_plane((6000.0, 0.0), "m") == pytest.approx((0.0, 6.0), abs=1e-9)
    # a drawing 1 % too small: turned onto the ray, its size kept - or resized
    r = CadReference("R", unit="mm")
    ratio = align_two_points(r, "m", (0.0, 0.0), (1.0, 1.0), (5.94, 0.0), (7.0, 1.0))
    assert ratio == pytest.approx(6.0 / 5.94) and r.scale == 1.0
    assert r.to_plane((5940.0, 0.0), "m") == pytest.approx((6.94, 1.0))
    r = CadReference("R", unit="mm")
    align_two_points(r, "m", (0.0, 0.0), (1.0, 1.0), (5.94, 0.0), (7.0, 1.0), scale=True)
    assert r.to_plane((5940.0, 0.0), "m") == pytest.approx((7.0, 1.0)) and r.scale == pytest.approx(6.0 / 5.94)
    with pytest.raises(ValueError):
        align_two_points(r, "m", (0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (7.0, 1.0))


def _dxf_file(tmp_path, name="plan.dxf") -> str:
    p = tmp_path / name
    p.write_bytes(_floor_plan().encode())
    return str(p)


def _doc():
    return Document(new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5, 3.5), grid_only=False))


@core
def test_saved_with_the_model_linked_and_embedded_and_in_other_units(tmp_path):
    d = _doc()
    fn = _dxf_file(tmp_path)
    name = cadref.add_reference(d, fn, stories=["Story1", "Story2"], embed=True, origin=(1.0, 2.0), base=(0.0, 0.0))
    ref = d.model.references[name]
    assert (ref.unit, ref.stories, ref.level, ref.pinned) == ("mm", ["Story1", "Story2"], 3.5, False)
    assert ref.path == os.path.abspath(fn) and ref.embedded in d.model.attachments
    assert sorted(ref.layers_off) == ["FROZEN", "NOTES"]  # as in the file
    blob = d.model.attachments[ref.embedded]
    assert blob.data == open(fn, "rb").read() and blob.size == os.path.getsize(fn) and blob.stored_size < blob.size
    model_file = str(tmp_path / "m.ssj.json")
    d.save(model_file)
    assert d.model.references[name].rel_path == "plan.dxf"
    m2 = Model.from_json(model_file)
    r2 = m2.references[name]
    assert r2 == d.model.references[name] and isinstance(r2.base, tuple) and isinstance(r2.origin, tuple)
    assert m2.attachments[ref.embedded].data == blob.data
    # other units: the picture stays where it is in the building
    mm = d.model.convert_units("kN mm")
    assert mm.references[name].to_plane((6000.0, 5000.0), "mm") == pytest.approx(
        tuple(v * 1000.0 for v in ref.to_plane((6000.0, 5000.0), "m")))
    assert mm.references[name].level == pytest.approx(3500.0) and mm.attachments == d.model.attachments
    # an undo step keeps the key of the embedded file, not its bytes
    big = Blob(os.urandom(300_000), "big.bin")
    d.model.attachments[big.key] = big
    snap = pickle.dumps(d.model)
    assert len(pickle.dumps(big)) < 200 and pickle.loads(snap).attachments[big.key] is big


@core
def test_a_linked_drawing_is_found_beside_the_moved_model_and_a_missing_one_is_told(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    d = _doc()
    name = cadref.add_reference(d, _dxf_file(a), stories=["Story1"])
    d.save(str(a / "m.ssj.json"))
    b = tmp_path / "b"
    os.rename(a, b)
    cadref.reload()
    d2 = Document()
    d2.open(str(b / "m.ssj.json"))
    ref = d2.model.references[name]
    assert not os.path.isfile(ref.path) and cadref.source_of(ref, d2.model, d2.path) == ("file", str(b / "plan.dxf"))
    dwg, why = cadref.drawing_of(ref, d2.model, d2.path)
    assert dwg is not None and why == "" and not d2.notes
    os.remove(b / "plan.dxf")
    cadref.reload()
    d3 = Document()
    d3.open(str(b / "m.ssj.json"))
    assert cadref.drawing_of(d3.model.references[name], d3.model, d3.path)[0] is None
    assert any("not found" in t and name in t for _, t in d3.notes)
    assert "FILE NOT FOUND" in cadref.describe(d3.model.references[name], d3.model, d3.path)


@core
def test_linked_embedded_or_both(tmp_path):
    d = _doc()
    fn = _dxf_file(tmp_path)
    name = cadref.add_reference(d, fn, stories=["Story1"])
    assert d.model.references[name].embedded is None and not d.model.attachments
    assert cadref.describe(d.model.references[name], d.model) == "linked"
    cadref.set_storage(d, name, link=True, embed=True)
    key = d.model.references[name].embedded
    assert key in d.model.attachments and cadref.describe(d.model.references[name], d.model) == "linked + embedded"
    cadref.set_storage(d, name, link=False, embed=True)
    ref = d.model.references[name]
    assert ref.path == "" and cadref.source_of(ref, d.model, None) == ("embedded", key)
    assert cadref.drawing_of(ref, d.model)[0] is not None and cadref.describe(ref, d.model) == "embedded"
    with pytest.raises(ModelError):  # neither
        cadref.set_storage(d, name, link=False, embed=False)
    with pytest.raises(ModelError):  # the copy is the only one left: it is not thrown away
        cadref.set_storage(d, name, link=True, embed=False)
    cadref.relink(d, name, fn)
    cadref.set_storage(d, name, link=True, embed=False)
    assert d.model.references[name].embedded is None and not d.model.attachments  # (the unused copy is gone)
    # the file changed on disk is seen, and read again on Reload
    ref = d.model.references[name]
    n0 = len(cadref.drawing_of(ref, d.model)[0].seg)
    assert not cadref.changed_on_disk(ref, d.model)
    with open(fn, "wb") as fh:
        fh.write(dxf(line(0, 0, 1, 1)).encode())
    os.utime(fn, ns=(os.stat(fn).st_atime_ns, os.stat(fn).st_mtime_ns + 5_000_000_000))
    assert cadref.changed_on_disk(ref, d.model) and len(cadref.drawing_of(ref, d.model)[0].seg) == n0
    cadref.reload(ref, d.model)
    assert len(cadref.drawing_of(ref, d.model)[0].seg) == 1
    with pytest.raises(ModelError) as e:
        cadref.add_reference(d, str(tmp_path / "nothing.dxf"))
    assert "cannot be read" in e.value.message


@core
def test_a_reference_is_no_part_of_the_structure_results_and_lock_stay(tmp_path):
    from structstudio import analyze
    d = _doc()
    for p in d.model.points.values():
        if abs(p.z) < 1e-9:
            p.restraint = (True,) * 6
    d.set_results(analyze(d.model))
    res = d.results
    assert d.locked and res is not None
    name = cadref.add_reference(d, _dxf_file(tmp_path), stories=["Story1"])
    cadref.move_reference(d, name, (0.0, 0.0), (3.0, 4.0))
    cadref.change_reference(d, name, color="layers", strength=0.9, layers_off=["GRID"])
    assert d.locked and d.results is res and d.model.references[name].origin == (3.0, 4.0)
    assert d.undo_name() == "Reference" if hasattr(d, "undo_name") else True
    for _ in range(3):
        assert d.undo()
    assert d.locked and d.results is res and not d.model.references and not d.can_undo
    assert d.redo() and name in d.model.references and d.results is res
    cadref.delete_references(d, [name])
    assert d.locked and d.results is res and not d.model.references


@core
def test_move_turn_scale_and_the_pin(tmp_path):
    d = _doc()
    name = cadref.add_reference(d, _dxf_file(tmp_path), stories=["Story1"])
    ref = lambda: d.model.references[name]  # noqa: E731
    at = lambda xy: ref().to_plane(xy, "m")  # noqa: E731
    cadref.move_reference(d, name, at((6000.0, 0.0)), (10.0, 10.0))
    assert at((6000.0, 0.0)) == pytest.approx((10.0, 10.0)) and at((0.0, 0.0)) == pytest.approx((4.0, 10.0))
    cadref.rotate_reference(d, name, (10.0, 10.0), 90.0)
    assert at((6000.0, 0.0)) == pytest.approx((10.0, 10.0)) and at((0.0, 0.0)) == pytest.approx((10.0, 4.0))
    # the bay drawn 6000 is really 6.3 m: resized about the first point
    f = cadref.scale_by_length(d, name, at((6000.0, 0.0)), at((0.0, 0.0)), 6.3)
    assert f == pytest.approx(1.05) and ref().scale == pytest.approx(1.05)
    assert at((6000.0, 0.0)) == pytest.approx((10.0, 10.0)) and at((0.0, 0.0)) == pytest.approx((10.0, 3.7))
    cadref.align_reference(d, name, at((0.0, 0.0)), (0.0, 0.0), at((12000.0, 0.0)), (12.0, 0.0), scale=True, pin=True)
    assert at((0.0, 0.0)) == pytest.approx((0.0, 0.0), abs=1e-9) and at((12000.0, 5000.0)) == pytest.approx((12.0, 5.0))
    assert ref().scale == pytest.approx(1.0) and ref().rotation == pytest.approx(0.0, abs=1e-9) and ref().pinned
    # pinned: every way of moving it is refused (and nothing changes) until it is unpinned or forced
    was = (ref().origin, ref().rotation, ref().scale)
    for go in (lambda: cadref.move_reference(d, name, (0, 0), (1, 1)),
               lambda: cadref.rotate_reference(d, name, (0, 0), 5.0),
               lambda: cadref.scale_reference(d, name, (0, 0), 2.0),
               lambda: cadref.align_reference(d, name, (0, 0), (1, 1), (1, 0), (2, 1))):
        with pytest.raises(ModelError) as e:
            go()
        assert e.value.code == "REFERENCE_PINNED"
    assert (ref().origin, ref().rotation, ref().scale) == was and ref().pinned
    cadref.move_reference(d, name, (0.0, 0.0), (1.0, 0.0), force=True, pin=True)
    assert at((0.0, 0.0)) == pytest.approx((1.0, 0.0)) and ref().pinned
    with pytest.raises(ModelError):
        cadref.scale_reference(d, name, (0, 0), 0.0, force=True)
    with pytest.raises(ModelError):
        cadref.scale_by_length(d, name, (0, 0), (0, 0), 5.0, force=True)


@core
def test_which_windows_show_it_and_its_story_renamed(tmp_path):
    d = _doc()
    fn = _dxf_file(tmp_path)
    plan = cadref.add_reference(d, fn, stories=["Story1"])
    free = cadref.add_reference(d, fn, level=1.25)  # a plan at a level without a story
    elev = cadref.add_reference(d, fn, view="xz", level=5.0)
    m = d.model
    names = lambda *a: [r.name for r in cadref.shown_in(m, *a)]  # noqa: E731
    assert names("plan", 3.5, "Story1") == [plan] and names("plan", 7.0, "Story2") == []
    assert names("plan", 1.25, None) == [free] and names("xz", 5.0, None) == [elev]
    assert names("xz", 0.0, None) == [] and names("yz", 5.0, None) == []
    cadref.change_reference(d, plan, visible=False)
    assert names("plan", 3.5, "Story1") == []
    cadref.change_reference(d, plan, visible=True, stories=["Story1", "Story2"])
    assert names("plan", 7.0, "Story2") == [plan]
    # the stories renamed and moved (Edit Stories): the drawing stays with them
    def stories(mm):
        old = [(s.name, s.elevation) for s in mm.stories_bottom_up()]
        new = [("Base", 0.0), ("L1", 4.0), ("L2", 8.0)]
        mm.carry_levels(old, new)
        mm.set_stories([("L1", 4.0), ("L2", 4.0)])
    d.edit(stories)
    m = d.model
    assert m.references[plan].stories == ["L1", "L2"] and m.references[plan].level == 4.0
    assert [r.name for r in cadref.shown_in(m, "plan", 4.0, "L1")] == [plan]


# ------------------------------------------------------------------------- snapping
@core
def test_snap_points_of_a_reference(tmp_path):
    d = _doc()
    name = cadref.add_reference(d, _dxf_file(tmp_path), stories=["Story1"])
    m = d.model
    ref = m.references[name]
    dwg, _ = cadref.drawing_of(ref, m)
    sn = cadref.RefSnapper([(ref, dwg)], "m")

    def kinds(uv, r=0.05, **kw):
        return sorted((k, tuple(round(v, 6) for v in p), n) for p, k, n in sn.hits(uv, r, **kw))
    # the corner of the column at (6, 5): an end of two of its sides
    assert kinds((6.21, 5.19)) == [("end", (6.2, 5.2), f"{name}: COLUMN")] * 2
    assert ("midpoint", (6.0, 5.2), f"{name}: COLUMN") in kinds((6.0, 5.21))
    assert ("intersection", (6.0, 5.2), f"{name}: GRID") in kinds((6.0, 5.21)) or \
        ("intersection", (6.0, 5.2), f"{name}: COLUMN") in kinds((6.0, 5.21))  # grid line x column side
    assert kinds((9.0, 2.5)) == [("center", (9.0, 2.5), f"{name}: COLUMN")]  # the round column's centre
    assert kinds((9.3, 2.5)) == []  # on the circle: no end of a chord
    assert [k for k, _, _ in kinds((9.3, 2.5), nearest=True)] == ["nearest"]
    assert ("nearest", (3.0, 0.0), f"{name}: GRID") in kinds((3.0, 0.02), nearest=True)
    assert ("perpendicular", (3.0, 5.0), f"{name}: GRID") in kinds((3.0, 5.0), previous=(3.0, 2.0))
    assert kinds((6.0, 5.21), kinds=("midpoint",)) == [("midpoint", (6.0, 5.2), f"{name}: COLUMN")]
    assert kinds((3.0, 2.0)) == [] and kinds((0.15, 0.15)) == []  # (layer NOTES is off: its line is no target)
    assert sn.which((3.0, 0.01), 0.05) == name and sn.which((3.0, 2.0), 0.05) is None
    # layers switched off, the reference moved / not snapped to
    cadref.change_reference(d, name, layers_off=["COLUMN", "NOTES", "FROZEN"])
    ref = d.model.references[name]
    sn = cadref.RefSnapper([(ref, dwg)], "m")
    assert kinds((6.21, 5.19)) == [] and kinds((9.0, 2.5)) == []
    cadref.change_reference(d, name, layers_off=[], origin=(100.0, 0.0), rotation=90.0)
    ref = d.model.references[name]
    sn = cadref.RefSnapper([(ref, dwg)], "m")
    assert ("end", (94.8, 6.2), f"{name}: COLUMN") in kinds((94.81, 6.21))  # (6.2, 5.2) turned: (100 - 5.2, 6.2)
    assert not cadref.RefSnapper([(cadref.replace(ref, snap=False), dwg)], "m")


# ------------------------------------------------------------------------ the window
# ------------------------------------------------- found by the audit of the feature
@core
def test_files_as_editors_save_them_and_curve_fit_polylines(monkeypatch):
    plan = _floor_plan()
    n = len(read_dxf(plan.encode()).seg)
    # a byte order mark (a DXF saved again from a text editor), UTF-16
    assert len(read_dxf(b"\xef\xbb\xbf" + plan.encode()).seg) == n
    assert len(read_dxf(plan.encode("utf-16")).seg) == n
    assert len(read_dxf(plan.replace("\n", "\r\n").encode()).seg) == n
    # a spline-fit polyline: the control points of its frame (flag 16) are no part of the
    # curve; the fitted vertices are chords of a curve (no midpoints to snap to), its ends are ends
    v = lambda x, y, f: _g((0, "VERTEX"), (8, "0"), (10, x), (20, y), (70, f))  # noqa: E731
    pl = _g((0, "POLYLINE"), (8, "0"), (66, 1), (70, 4)) + v(0, 0, 16) + v(5, 9, 16) + v(10, 0, 16) + \
        v(0, 0, 8) + v(5, 4, 8) + v(10, 0, 8) + _g((0, "SEQEND"))
    d = read_dxf(dxf(pl).encode())
    assert _segs(d) == [((0.0, 0.0), (5.0, 4.0)), ((5.0, 4.0), (10.0, 0.0))] and d.seg_curve.all()
    assert sorted(map(tuple, d.pts.tolist())) == [(0.0, 0.0), (10.0, 0.0)] and set(d.pts_kind.tolist()) == {PT_END}
    # the size limit counts the lines a drawing ends up with - blocks in blocks once, not
    # once per level (a drawing of 200 lines under a limit of 250 was refused)
    from structstudio.io import dxf_read
    inner = block("A", "".join(line(i, 0, i, 1) for i in range(10)))
    mid = block("B", "".join(insert("A", 0, 2 * k) for k in range(10)))
    top = block("C", insert("B", 0, 0) + insert("B", 50, 0))
    text = dxf(insert("C", 0, 0), "", inner + mid + top).encode()
    monkeypatch.setattr(dxf_read, "MAX_SEGMENTS", 250)
    assert len(read_dxf(text).seg) == 200
    monkeypatch.setattr(dxf_read, "MAX_SEGMENTS", 150)
    with pytest.raises(DxfError) as e:
        read_dxf(text)
    assert "more line work" in str(e.value)


@core
def test_the_copy_in_the_model_follows_reload_and_stands_in_for_a_broken_file(tmp_path):
    d = _doc()
    fn = _dxf_file(tmp_path)
    name = cadref.add_reference(d, fn, stories=["Story1"], embed=True)
    ref = lambda: d.model.references[name]  # noqa: E731
    old = ref().embedded
    n0 = len(cadref.drawing_of(ref(), d.model)[0].seg)
    # the file is drawn again: Reload shows it, and the copy in the model is renewed (the
    # model sent alone showed the old drawing) - one undo step, the results kept
    with open(fn, "wb") as fh:
        fh.write(dxf(line(0, 0, 1, 1) + line(0, 0, 2, 0)).encode())
    cadref.reload()
    steps = len(d._undo)
    assert cadref.refresh_embedded(d) == [name] and len(d._undo) == steps + 1
    assert ref().embedded != old and list(d.model.attachments) == [ref().embedded]
    assert len(read_dxf(d.model.attachments[ref().embedded].data).seg) == 2
    assert cadref.refresh_embedded(d) == []  # nothing new: no undo step
    assert len(d._undo) == steps + 1
    # the linked file is damaged (half written): the copy in the model is shown
    with open(fn, "wb") as fh:
        fh.write(b"not a drawing")
    cadref.reload()
    dwg, why = cadref.drawing_of(ref(), d.model)
    assert dwg is not None and len(dwg.seg) == 2 and why == ""
    assert cadref.refresh_embedded(d) == []  # (a file that cannot be read replaces no copy)
    # linked only: the damaged file is told
    cadref.set_storage(d, name, link=False, embed=True)
    lone = cadref.add_reference(d, _dxf_file(tmp_path, "b.dxf"), stories=["Story1"])
    with open(tmp_path / "b.dxf", "wb") as fh:
        fh.write(b"not a drawing")
    cadref.reload()
    dwg, why = cadref.drawing_of(d.model.references[lone], d.model)
    assert dwg is None and "DXF" in why
    # Choose File on an embedded reference: the copy is the new file's
    fn2 = _dxf_file(tmp_path, "c.dxf")
    cadref.set_storage  # (name is embedded only now)
    cadref.relink(d, name, fn2)
    assert ref().path == fn2 and len(read_dxf(d.model.attachments[ref().embedded].data).seg) == n0
    assert not cadref.changed_on_disk(ref(), d.model)
    # another model: the drawings read for this one are not kept
    assert cadref._CACHE
    d.new()
    assert not cadref._CACHE and not cadref._STAMPS


# ------------------------------------------------------------ a PDF written by hand
class _Pdf:
    """A small PDF file: objects by number, streams plain or deflated."""

    def __init__(self):
        self.objs = []

    def reserve(self) -> int:
        self.objs.append(None)
        return len(self.objs)

    def put(self, body, number=None) -> int:
        body = body.encode("latin-1") if isinstance(body, str) else body
        if number is None:
            self.objs.append(body)
            return len(self.objs)
        self.objs[number - 1] = body
        return number

    def stream(self, data, extra="", flate=False, number=None) -> int:
        data = data.encode("latin-1") if isinstance(data, str) else data
        if flate:
            data, extra = zlib.compress(data), extra + " /Filter /FlateDecode"
        return self.put(f"<< /Length {len(data)} {extra} >>\nstream\n".encode("latin-1") + data + b"\nendstream", number)

    def build(self, root: int) -> bytes:
        out = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
        at = []
        for k, body in enumerate(self.objs, 1):
            at.append(len(out))
            out += f"{k} 0 obj\n".encode() + body + b"\nendobj\n"
        x = len(out)
        out += f"xref\n0 {len(self.objs) + 1}\n".encode() + b"0000000000 65535 f \n"
        for a in at:
            out += f"{a:010d} 00000 n \n".encode()
        out += f"trailer\n<< /Size {len(self.objs) + 1} /Root {root} 0 R >>\nstartxref\n{x}\n%%EOF\n".encode()
        return bytes(out)


def pdf(pages, catalog="", flate=False) -> bytes:
    """A PDF of ``pages``: each (content, page dictionary entries, resources entries)
    or just the content (A3 landscape).  Objects named in the entries: see the tests."""
    f = _Pdf()
    return _pdf_into(f, pages, catalog, flate)


def _pdf_into(f, pages, catalog="", flate=False) -> bytes:
    root, tree = f.reserve(), f.reserve()
    kids = []
    for pg in pages:
        content, entries, res = (pg, "", "") if isinstance(pg, str) else (tuple(pg) + ("", ""))[:3]
        if "/MediaBox" not in entries:
            entries += " /MediaBox [0 0 1190.55 841.89]"
        c = f.stream(content, flate=flate)
        kids.append(f.put(f"<< /Type /Page /Parent {tree} 0 R /Contents {c} 0 R /Resources << {res} >> {entries} >>"))
    f.put(f"<< /Type /Pages /Count {len(kids)} /Kids [{' '.join(f'{k} 0 R' for k in kids)}] >>", tree)
    f.put(f"<< /Type /Catalog /Pages {tree} 0 R {catalog} >>", root)
    return f.build(root)


def _mm(*pt):
    return tuple(round(v * PT_TO_MM, 6) for v in pt)


def _psegs(d, layer=None):
    return [(tuple(round(v / PT_TO_MM, 3) for v in a), tuple(round(v / PT_TO_MM, 3) for v in b))
            for a, b in _segs(d, layer)]


pdf_core = pytest.mark.skipif(CHILD, reason="parent only")


@pytest.fixture
def pypdf():
    return pytest.importorskip("pypdf")


@pdf_core
def test_a_pdf_page_is_its_strokes_in_millimetres_on_the_sheet(pypdf):
    content = """
    0.72 w 0 0 0 RG
    100 100 m 400 100 l S                         % a line
    2 0 0 2 10 20 cm                              % everything after: twice the size, shifted
    q 1 0 0 RG 0.36 w 50 50 100 40 re S Q         % a red rectangle, a thinner pen
    200 200 m 260 200 l 260 240 l h S             % a triangle closed with h
    300 200 m 340 200 l 340 230 l s               % ... and with s
    BT /F1 12 Tf (GRID A) Tj ET                   % a text: not line work
    0.5 g 400 300 20 20 re f                      % a column drawn solid: its outline
    1 g 0 0 500 400 re f                          % a white background: nothing
    q 0.5 0 0 0.5 -10 -20 cm 0.1 g 0 0 1190.55 841.89 re f Q   % the sheet's own dark background: nothing
    0 g 10 10 m 30 10 l 30 30 l 20 40 l 10 30 l 12 28 l 14 26 l 16 24 l 18 22 l 20 20 l 22 18 l 24 16 l
        26 14 l 28 12 l 29 11 l 30 12 l 31 13 l 32 14 l f   % a large fill (a hatch): counted, left out
    5 5 m 9 9 l W n                               % a clip path: nothing
    0 0 m 0 0 l S                                 % a dot: nothing to draw
    """
    d = read_pdf(pdf([content]))
    assert (d.source, d.unit, d.pages, d.page) == ("pdf", "mm", 1, 1) and d.guess_unit() == "mm"
    assert d.sheet == pytest.approx((420.0, 297.0), abs=0.01) and d.plot_scale is None
    assert d.skipped == {"TEXT": 1, "FILL": 1}
    # no layers in the file: one per pen (width, colour), and the outlines of the solids
    assert d.layers == ["Pen 0.25 mm #000000", "Pen 0.25 mm #ff0000", "Pen 0.51 mm #000000", "Filled shapes"]
    assert d.layer_counts() == {"Pen 0.25 mm #000000": 1, "Pen 0.25 mm #ff0000": 4, "Pen 0.51 mm #000000": 6,
                                "Filled shapes": 4}
    assert _psegs(d, "Pen 0.25 mm #000000") == [((100.0, 100.0), (400.0, 100.0))]
    assert _psegs(d, "Pen 0.25 mm #ff0000") == [((110.0, 120.0), (110.0, 200.0)), ((110.0, 120.0), (310.0, 120.0)),
                                                 ((110.0, 200.0), (310.0, 200.0)), ((310.0, 120.0), (310.0, 200.0))]
    tri = _psegs(d, "Pen 0.51 mm #000000")
    assert ((410.0, 420.0), (530.0, 500.0)) in tri and ((610.0, 420.0), (690.0, 480.0)) in tri and len(tri) == 6
    assert _psegs(d, "Filled shapes")[0] == ((810.0, 620.0), (810.0, 660.0))
    assert d.seg_rgb[d.seg_layer == 1].tolist() == [[255, 0, 0]] * 4 and not d.seg_curve.any()
    assert d.seg[0].tolist() == pytest.approx([*_mm(100, 100), *_mm(400, 100)])
    # deflated, as files are written: the same
    assert np.allclose(read_pdf(pdf([content], flate=True)).seg, d.seg)
    assert is_pdf(pdf([content])) and not is_pdf(_floor_plan().encode())


@pdf_core
def test_pdf_curves_are_chords_with_ends_and_circle_centres(pypdf):
    k = 0.5522847498 * 50
    circle_ = f"""450 300 m 450 {300 + k} {400 + k} 350 400 350 c {400 - k} 350 350 {300 + k} 350 300 c
                 350 {300 - k} {400 - k} 250 400 250 c {400 + k} 250 450 {300 - k} 450 300 c S"""
    d = read_pdf(pdf([circle_ + " 0 0 m 30 0 60 0 90 0 c S  100 0 m 150 100 200 100 v S"]))
    c = d.seg[d.seg_curve]
    # 7.5 degrees of turning a chord, as the arcs of a DXF (the last curve turns by 63.4); a flat curve: a line
    assert len(c) == 48 + 9 and len(d.seg) == 58
    r = np.hypot(c[:48, 0] / PT_TO_MM - 400, c[:48, 1] / PT_TO_MM - 300)
    assert r == pytest.approx(50.0, abs=0.02)
    assert _psegs(d)[0] == ((0.0, 0.0), (90.0, 0.0)) and not d.seg_curve[48]
    ends = {tuple(round(v / PT_TO_MM, 3) for v in p) for p, kk in zip(d.pts.tolist(), d.pts_kind.tolist()) if kk == PT_END}
    assert {(450.0, 300.0), (400.0, 350.0), (350.0, 300.0), (400.0, 250.0), (100.0, 0.0), (200.0, 100.0)} <= ends
    centres = [tuple(round(v / PT_TO_MM, 3) for v in p) for p, kk in zip(d.pts.tolist(), d.pts_kind.tolist())
               if kk == PT_CENTER]
    assert centres == [(400.0, 300.0)]


@pdf_core
def test_pdf_layers_forms_and_the_page_as_it_is_viewed(pypdf):
    # the layers CAD writes when plotting with layers (optional content), one of them off;
    # a block as a form with its own matrix and layer, drawn twice
    f = _Pdf()
    walls = f.put("<< /Type /OCG /Name (S-WALL) >>")
    grid = f.put("<< /Type /OCG /Name (S-GRID) >>")
    hidden = f.put("<< /Type /OCG /Name (A-FURN) >>")
    form = f.stream("0 0 10 10 re S", f"/Type /XObject /Subtype /Form /BBox [0 0 10 10] /Matrix [2 0 0 2 0 0] "
                                      f"/OC {walls} 0 R")
    res = f"/Properties << /MC0 {walls} 0 R /MC1 {grid} 0 R /MC2 {hidden} 0 R >> /XObject << /Col {form} 0 R >>"
    content = """
    /OC /MC1 BDC 0 1 0 RG 0 0 m 100 0 l S EMC
    /OC /MC0 BDC 0 0 m 0 50 l S  /Span BMC 5 5 m 6 6 l S EMC  EMC
    /OC /MC2 BDC 7 7 m 8 8 l S EMC
    1 0 0 1 20 20 m 30 30 l S
    q 1 0 0 1 100 100 cm /Col Do Q   q 0 1 -1 0 300 100 cm /Col Do Q
    """
    cat = f"/OCProperties << /OCGs [{walls} 0 R {grid} 0 R {hidden} 0 R] /D << /OFF [{hidden} 0 R] >> >>"
    d = read_pdf(_pdf_into(f, [(content, "", res)], cat))
    assert d.layers == ["S-GRID", "S-WALL", "A-FURN", "(no layer)"] and d.layers_off == ["A-FURN"]
    assert d.layer_counts() == {"S-GRID": 1, "S-WALL": 2 + 8, "A-FURN": 1, "(no layer)": 1}
    assert d.layer_rgb[0] == (0, 255, 0)
    w = _psegs(d, "S-WALL")
    assert ((100.0, 100.0), (120.0, 100.0)) in w and ((100.0, 100.0), (100.0, 120.0)) in w  # twice the size
    assert ((280.0, 100.0), (300.0, 100.0)) in w and ((300.0, 100.0), (300.0, 120.0)) in w  # ... and turned
    # the page turned by 90 (a portrait page shown landscape), a crop box, a user unit
    line_ = "50 100 m 250 100 l S"
    turned = read_pdf(pdf([(line_, "/MediaBox [0 0 600 800] /Rotate 90")]))
    assert turned.sheet == pytest.approx((800 * PT_TO_MM, 600 * PT_TO_MM))
    assert _psegs(turned) == [((100.0, 350.0), (100.0, 550.0))]
    assert _psegs(read_pdf(pdf([(line_, "/MediaBox [0 0 600 800] /Rotate 180")]))) == [((350.0, 700.0), (550.0, 700.0))]
    assert _psegs(read_pdf(pdf([(line_, "/MediaBox [0 0 600 800] /Rotate 270")]))) == [((700.0, 50.0), (700.0, 250.0))]
    crop = read_pdf(pdf([(line_ + " 0 0 m 5 5 l S", "/MediaBox [0 0 600 800] /CropBox [40 90 440 390] /UserUnit 2")]))
    assert crop.sheet == pytest.approx((800 * PT_TO_MM, 600 * PT_TO_MM))
    assert _psegs(crop) == [((20.0, 20.0), (420.0, 20.0))]  # from the corner of the crop box; off the sheet: left out


@pdf_core
def test_pdf_pages_the_scale_it_states_and_what_is_refused(pypdf, tmp_path, monkeypatch):
    two = pdf(["0 0 m 10 0 l S", ("0 0 m 0 20 l 20 20 l S", "/MediaBox [0 0 842 595]")])
    assert [len(read_pdf(two, n).seg) for n in (1, 2)] == [1, 2] and read_pdf(two, 2).pages == 2
    assert read_pdf(two, 2).sheet == pytest.approx((297.04, 209.9), abs=0.01)
    assert read_pdf(two, 9).page == 2  # (past the end: the last page - said by .page)
    fn = tmp_path / "two.pdf"
    fn.write_bytes(two)
    assert len(read_pdf(str(fn), 2).seg) == 2 and len(cadref.read_drawing(str(fn), 2).seg) == 2
    assert cadref.read_drawing(_floor_plan().encode()).source == "dxf"
    # a measuring viewport (as some programs write it): 1 point = 0.035278 m -> 1 : 100
    vp = "/VP [<< /Type /Viewport /BBox [0 0 800 500] /Measure << /Type /Measure /Subtype /RL " \
         "/R (1 mm = 100 mm) /X [<< /Type /NumberFormat /U (m) /C 0.0352778 >>] >> >>]"
    assert read_pdf(pdf([("0 0 m 10 0 l S", vp)])).plot_scale == pytest.approx(100.0, rel=1e-4)
    # a scanned sheet: a picture and no lines
    f = _Pdf()
    img = f.stream(b"\x00", "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8")
    with pytest.raises(DxfError) as e:
        read_pdf(_pdf_into(f, [("q 800 0 0 600 0 0 cm /Im0 Do Q", "", f"/XObject << /Im0 {img} 0 R >>")]))
    assert "scanned" in str(e.value) and "plotted from a CAD program" in str(e.value)
    with pytest.raises(DxfError) as e:
        read_pdf(pdf(["BT /F1 12 Tf (only words) Tj ET"]))
    assert "no lines" in str(e.value) and "only text" in str(e.value)
    # an inline picture between the lines: skipped, the lines stay
    d = read_pdf(pdf([b"0 0 m 10 0 l S BI /W 1 /H 1 /CS /G /BPC 8 ID \xff 5 5 m S Q \nEI 0 0 m 0 10 l S".decode("latin-1")]))
    assert len(d.seg) == 2 and d.skipped == {"IMAGE": 1}
    for bad in (b"", b"%PDF-1.4\nnothing", _floor_plan().encode()):
        with pytest.raises(DxfError):
            read_pdf(bad)
    # protected with a password: told; protected with none to open it (printing only): read
    try:
        w = pypdf.PdfWriter()
        w.append(pypdf.PdfReader(__import__("io").BytesIO(two)))
        w.encrypt(user_password="secret", owner_password="owner", algorithm="RC4-128")
        buf = __import__("io").BytesIO()
        w.write(buf)
        locked = buf.getvalue()
        w = pypdf.PdfWriter()
        w.append(pypdf.PdfReader(__import__("io").BytesIO(two)))
        w.encrypt(user_password="", owner_password="owner", algorithm="RC4-128")
        buf = __import__("io").BytesIO()
        w.write(buf)
        open_ = buf.getvalue()
    except Exception:
        locked = open_ = None
    if locked is not None:
        with pytest.raises(DxfError) as e:
            read_pdf(locked)
        assert "password" in str(e.value)
        assert len(read_pdf(open_).seg) == 1
    # the package that opens PDF files is not there: said so, DXF still works
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(DxfError) as e:
        read_pdf(two)
    assert "pypdf" in str(e.value)
    assert len(cadref.read_drawing(_floor_plan().encode()).seg) == 80


def _plan_pdf(tmp_path, name="plan.pdf", scale=100.0, pages=1) -> str:
    """The floor plan of the DXF tests plotted 1 : ``scale`` on an A3 sheet, its (0, 0) at
    (60 mm, 80 mm) of the sheet; grid lines thin and red, the rest black."""
    d = read_dxf(_floor_plan().encode())
    k = 1.0 / scale / PT_TO_MM  # drawing mm -> points
    ox, oy = 60 / PT_TO_MM, 80 / PT_TO_MM
    out = []
    for lay, pen, rgb in (("GRID", 0.5, "1 0 0"), ("COLUMN", 1.0, "0 0 0"), ("WALL", 1.0, "0 0 0")):
        out.append(f"{pen} w {rgb} RG")
        for x1, y1, x2, y2 in d.seg[d.seg_layer == d.layers.index(lay)].tolist():
            out.append(f"{ox + x1 * k:.5f} {oy + y1 * k:.5f} m {ox + x2 * k:.5f} {oy + y2 * k:.5f} l S")
    p = tmp_path / name
    p.write_bytes(pdf(["\n".join(out)] + ["0 0 m 10 0 l S"] * (pages - 1), flate=True))
    return str(p)


@pdf_core
def test_a_pdf_sheet_as_a_reference_page_and_plot_scale(pypdf, tmp_path):
    d = _doc()
    fn = _plan_pdf(tmp_path, pages=2)
    name = cadref.add_reference(d, fn, stories=["Story1"], embed=True, scale=100.0, base=(60.0, 80.0))
    ref = d.model.references[name]
    assert (ref.unit, ref.scale, ref.page) == ("mm", 100.0, 1) and ref.pinned is False
    dwg, _ = cadref.drawing_of(ref, d.model)
    assert dwg.source == "pdf" and len(dwg.seg) == 78 and dwg.layers == ["Pen 0.18 mm #ff0000", "Pen 0.35 mm #000000"]
    # sheet millimetres x the plot scale = the building: the grid crossing C-2 at (12, 5) m
    assert ref.to_plane((60.0 + 120.0, 80.0 + 50.0), "m") == pytest.approx((12.0, 5.0))
    assert ref.factor("m") == pytest.approx(0.1)
    sn = cadref.RefSnapper([(ref, dwg)], "m")
    assert ((6.2, 5.2), "end") in [((round(q[0], 4), round(q[1], 4)), kind) for q, kind, _ in sn.hits((6.21, 5.19), 0.05)]
    # without a scale: what the file states, else 1:100 (to be set with Scale by a Known Length)
    assert d.model.references[cadref.add_reference(d, fn, stories=["Story1"])].scale == 100.0
    with pytest.raises(ModelError):
        cadref.add_reference(d, fn, stories=["Story1"], scale=0.0)
    # saved and read: the page and the scale; the copy in the model shows it without the file
    d.save(str(tmp_path / "m.ssj.json"))
    os.remove(fn)
    d2 = Document()
    d2.open(str(tmp_path / "m.ssj.json"))
    r2 = d2.model.references[name]
    assert (r2.unit, r2.scale, r2.page, r2.base) == ("mm", 100.0, 1, (60.0, 80.0))
    assert len(cadref.drawing_of(r2, d2.model, d2.path)[0].seg) == 78
    # another page (one undo step, the results kept), a page that is not there
    steps = len(d2._undo)
    cadref.set_page(d2, name, 2, d2.path)
    assert d2.model.references[name].page == 2 and len(cadref.drawing_of(d2.model.references[name], d2.model, d2.path)[0].seg) == 1
    assert len(d2._undo) == steps + 1
    with pytest.raises(ModelError) as e:
        cadref.set_page(d2, name, 3, d2.path)
    assert "2 page(s)" in e.value.message
    # a model saved before PDF sheets: no page in the file -> the first
    data = d2.model.to_dict()
    for r in data["references"].values():
        r.pop("page", None)
    assert Model.from_dict(data).references[name].page == 1


@pdf_core
def test_a_pdf_drawn_by_another_program_reads_the_same(pypdf, tmp_path):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    fn = str(tmp_path / "r.pdf")
    c = canvas.Canvas(fn, pagesize=(1190.55, 841.89))
    c.setLineWidth(0.7)
    c.setStrokeColorRGB(0, 0, 1)
    c.line(72, 72, 360, 72)
    c.rect(100, 100, 50, 80)
    c.circle(400, 300, 50)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.rect(500, 500, 20, 20, stroke=0, fill=1)
    c.drawString(100, 500, "A text")
    c.save()
    d = read_pdf(fn)
    assert d.layers == ["Pen 0.25 mm #0000ff", "Filled shapes"] and d.skipped == {"TEXT": 1}
    assert d.layer_counts() == {"Pen 0.25 mm #0000ff": 1 + 4 + 48, "Filled shapes": 4}
    assert _psegs(d)[0] == ((72.0, 72.0), (360.0, 72.0))
    assert [tuple(round(v / PT_TO_MM, 2) for v in p) for p, kk in zip(d.pts.tolist(), d.pts_kind.tolist())
            if kk == PT_CENTER] == [(400.0, 300.0)]



@core
def test_the_picture_of_a_drawing_is_drawn_with_numpy():
    pytest.importorskip("PySide6.QtWidgets")
    from structstudio.app import ref_preview
    from structstudio.app.ref_preview import raster_lines, render
    P = np.array([[2.0, 3.0, 12.0, 3.0], [5.0, 0.0, 5.0, 9.0], [-50.0, 8.0, 500.0, 8.0], [100.0, 100.0, 200.0, 200.0],
                  [np.nan, 0.0, 1.0, 1.0]])
    rgb = np.array([[255, 0, 0], [0, 0, 255], [0, 128, 0], [9, 9, 9], [1, 1, 1]], np.uint8)
    img = raster_lines(P, rgb, 20, 10, (255, 255, 255))
    assert img.shape == (10, 20, 3)
    assert (img[3, 2:13] == [255, 0, 0]).all() or (img[3, [2, 3, 4, 6, 12]] == [255, 0, 0]).all()
    assert (img[[0, 1, 2, 4, 9], 5] == [0, 0, 255]).all()  # the later line over the earlier where they cross
    assert (img[8, :] == [0, 128, 0]).all()  # a line far longer than the picture: only what shows is drawn
    assert (img[0, 0] == [255, 255, 255]).all() and int((img != 255).any(axis=2).sum()) == 11 + 10 + 20 - 2  # (two crossings)
    assert (raster_lines(P[:1], np.array([7, 8, 9], np.uint8), 20, 10)[3, 7] == [7, 8, 9]).all()  # one colour for all
    assert (raster_lines(np.zeros((0, 4)), rgb[:0], 4, 4, (1, 2, 3)) == [1, 2, 3]).all()
    # more lines than one picture can take: every line still shows (thinner), none is dropped
    many = np.column_stack([np.zeros(400), np.arange(400.0), np.full(400, 399.0), np.arange(400.0)])
    full = raster_lines(many, np.array([0, 0, 0], np.uint8), 400, 400)
    old = ref_preview.MAX_SAMPLES
    try:
        ref_preview.MAX_SAMPLES = 20_000
        thin = raster_lines(many, np.array([0, 0, 0], np.uint8), 400, 400)
    finally:
        ref_preview.MAX_SAMPLES = old
    assert (full == 0).all() and ((thin == 0).any(axis=(1, 2))).all() and (thin == 0).sum() < (full == 0).sum()
    # a whole drawing: its lines in their colours (white lines in ink on a white picture), a layer left out
    d = read_dxf(_floor_plan().encode())
    a = render(d, 300, 160)
    b = render(d, 300, 160, layers_off=["GRID"])
    dark = lambda im: int((im.sum(axis=2) < 600).sum())  # noqa: E731
    assert dark(a) > dark(b) > 0 and (a[0, 0] == [255, 255, 255]).all()
    assert ((a == [255, 0, 0]).all(axis=2)).any() and not ((b == [255, 0, 0]).all(axis=2)).any()  # the red grid


@core
def test_gui_in_subprocess():
    import shutil
    pytest.importorskip("vtk")
    pytest.importorskip("PySide6.QtWidgets")
    env = dict(os.environ, STRUCTSTUDIO_GUI_CHILD="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__]
    if not env.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1600x1000x24"] + cmd
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if "WARN|" not in ln]
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower(), \
        "\n".join(lines[-60:])


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app, tmp_path):
    from PySide6 import QtCore
    from structstudio.app.mainwindow import MainWindow
    w = MainWindow(_doc(), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.resize(1500, 950)
    w.show()
    app.processEvents()
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    app.processEvents()


def _line_offset(actor) -> float:
    """The depth offset of the lines of an actor (negative: towards the eye)."""
    import vtk
    f, u = vtk.reference(0.0), vtk.reference(0.0)
    actor.GetMapper().GetRelativeCoincidentTopologyLineOffsetParameters(f, u)
    return float(u)


def _plan(win, app, story="Story1"):
    vp = next(v for v in win.views.viewports if v.config.kind == "plan")
    win.views.set_active(vp)
    vp.set_view("plan", story=story)
    app.processEvents()
    return vp


def _click(vp, app, xyz, button=None):
    from PySide6 import QtCore, QtTest
    x, y = vp._display(xyz)
    dpr = vp._dpr()
    pos = QtCore.QPoint(int(round(x / dpr)), int(round(vp.vtk_widget.height() - 1 - y / dpr)))
    QtTest.QTest.mouseMove(vp.vtk_widget, pos)
    app.processEvents()
    QtTest.QTest.mouseClick(vp.vtk_widget, button or QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, pos)
    app.processEvents()


def _place(win, app, monkeypatch, path, answer=None):
    """References > Place DXF Drawing with the two forms answered."""
    from PySide6 import QtWidgets
    from structstudio.app import ref_ui
    seen = {}
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path, "")))

    def run(dlg):
        seen.update(size=dlg.size.text(), unit=dlg.unit.currentData(), note=dlg.unit_note.text(),
                    own=dlg.pos_own.isChecked(), view=dlg.values()["view"], stories=dlg.values()["stories"],
                    layers=dlg.layers.off())
        if answer:
            answer(dlg)
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(ref_ui.PlaceReferenceDialog, "exec", run)
    before = set(win.doc.model.references)
    win.menu_actions["ref.place"].trigger()
    app.processEvents()
    made = sorted(set(win.doc.model.references) - before)
    return (made[0] if made else None), seen


@gui
def test_the_references_menu_in_the_order_of_the_work(win, app):
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert titles[titles.index("Define") + 1:titles.index("Draw") + 1] == ["References", "Draw"]
    items = [(a.text(), a.shortcut().toString()) for a in win.menus["References"].actions() if not a.isSeparator()]
    assert items == [("Place Drawing (DXF / PDF)...", ""), ("Align (2 Points)", ""), ("Scale by a Known Length", ""),
                     ("Move", ""), ("Rotate", ""), ("Pin / Unpin", ""), ("Show in This Window", "Ctrl+Alt+R"),
                     ("Snap to References", "Ctrl+Alt+S"), ("Layers...", ""), ("Zoom to Reference", ""),
                     ("Reload Linked Drawings", ""), ("Manage References...", "")]
    assert all(a.isEnabled() for a in win.menus["References"].actions())  # none is only planned
    # the snap switch is the "CAD references" snap of the snap list (one switch, not two)
    assert win.menu_actions["ref.snap"] is win.snap_actions["cad"] and win.state.snap.cad
    win.menu_actions["ref.snap"].trigger()
    assert not win.state.snap.cad
    # no other program is named in the menu
    text = " ".join(t for t, _ in items).lower()
    assert not any(w in text for w in ("autocad", "revit", "archicad", "cype", "etabs"))


@gui
def test_placed_through_the_menu_shown_on_its_story_only(win, app, monkeypatch, tmp_path):
    vp = _plan(win, app)
    fn = _dxf_file(tmp_path)
    name, seen = _place(win, app, monkeypatch, fn)
    assert seen == {"size": "14.00 m x 7.00 m", "unit": "mm", "note": "(from the file)", "own": True, "view": "plan",
                    "stories": ["Story1"], "layers": ["FROZEN", "NOTES"]}
    ref = win.doc.model.references[name]
    assert (ref.view, ref.stories, ref.unit, ref.origin, ref.base) == ("plan", ["Story1"], "mm", (0.0, 0.0), (0.0, 0.0))
    assert win.state.tool.name == "ref_align" and "click a point of the drawing" in win.state.tool.hint()
    assert len(vp._ref_actors) == 1
    lines = vp._ref_actors[0].GetMapper().GetInput()
    assert lines.GetNumberOfLines() == 5 + 24 + 1 + 48  # grid, columns, wall, the circle (NOTES / FROZEN are off)
    assert not vp._ref_actors[0].GetPickable() and not vp._ref_actors[0].GetUseBounds()
    b = lines.GetBounds()
    assert (b[0], b[1], b[2], b[3], b[4], b[5]) == pytest.approx((-1.0, 13.0, -1.0, 6.0, 3.5, 3.5))  # mm -> m, on the story
    # the other windows: the 3D window and the plan of another story show none
    for w in win.views.viewports:
        if w is not vp:
            assert not getattr(w, "_ref_actors", []) and w.shown_references() == []
    vp.set_view("plan", story="Story2")
    app.processEvents()
    assert vp._ref_actors == [] and vp.reference_snapper() is None
    vp.set_view("plan", story="Story1")
    app.processEvents()
    assert len(vp._ref_actors) == 1
    # Show in This Window: off and on again
    win.menu_actions["ref.show_window"].trigger()
    app.processEvents()
    assert vp._ref_actors == [] and vp.references_hidden and not win.menu_actions["ref.show_window"].isChecked()
    win.menu_actions["ref.show_window"].trigger()
    app.processEvents()
    assert len(vp._ref_actors) == 1
    # an edit of the model does not make the drawing's lines again
    actor = vp._ref_actors[0]
    sec = next(iter(win.doc.model.frame_sections))
    win.doc.edit(lambda m: m.add_frame((1.0, 1.0, 3.5), (2.0, 1.0, 3.5), sec))
    app.processEvents()
    assert vp._ref_actors == [actor]
    assert win.doc.undo() and win.doc.undo() and not win.doc.model.references
    app.processEvents()
    assert vp._ref_actors == []


@gui
def test_drawn_over_extruded_sections_or_under_and_in_its_colours(win, app, monkeypatch, tmp_path):
    vp = _plan(win, app)
    name, _ = _place(win, app, monkeypatch, _dxf_file(tmp_path))
    mapper = vp._ref_actors[0].GetMapper()
    assert _line_offset(vp._ref_actors[0]) < -60000  # in front of what is at its level
    assert not mapper.GetScalarVisibility()  # one faded colour
    win.act_extrude.trigger()
    app.processEvents()
    assert len(vp._ref_actors) == 1 and vp.display_options.extrude
    cadref.change_reference(win.doc, name, on_top=False, color="layers")
    app.processEvents()
    mapper = vp._ref_actors[0].GetMapper()
    assert _line_offset(vp._ref_actors[0]) > 60000 and mapper.GetScalarVisibility()
    rgb = vp._ref_actors[0].GetMapper().GetInput().GetCellData().GetScalars()
    assert rgb.GetNumberOfTuples() == 78 and rgb.GetNumberOfComponents() == 3
    cadref.change_reference(win.doc, name, color="#ff0000", strength=1.0)
    app.processEvents()
    assert vp._ref_actors[0].GetProperty().GetColor() == pytest.approx((1.0, 0.0, 0.0))


@gui
def test_snapping_to_the_drawing_and_the_model_first(win, app, monkeypatch, tmp_path):
    vp = _plan(win, app)
    name, _ = _place(win, app, monkeypatch, _dxf_file(tmp_path))
    win.state.set_tool("beam")
    app.processEvents()

    def best(xyz):
        x, y = vp._display(xyz)
        hits = vp._snap_engine(int(x), int(y)).find(x, y)
        return (hits[0].kind, hits[0].target, tuple(round(c, 4) for c in hits[0].xyz), hits[0].cad) if hits else None
    assert best((6.2, 5.2, 3.5)) == ("end", f"{name}: COLUMN", (6.2, 5.2, 3.5), True)  # a column corner of the drawing
    assert best((9.0, 2.5, 3.5)) == ("center", f"{name}: COLUMN", (9.0, 2.5, 3.5), True)
    b = best((6.0, 5.0, 3.5))  # the model's own joint, not the drawing's point at the same place
    assert b[0] in ("joint", "grid") and b[2] == (6.0, 5.0, 3.5) and not b[3]
    win.snap_actions["cad"].trigger()  # Snap to References off
    assert best((6.2, 5.2, 3.5)) != ("end", f"{name}: COLUMN", (6.2, 5.2, 3.5), True)
    win.snap_actions["cad"].trigger()
    cadref.change_reference(win.doc, name, snap=False)
    app.processEvents()
    assert vp.reference_snapper() is None and (best((9.0, 2.5, 3.5)) or (None,))[0] != "center"


@gui
def test_align_with_two_points_then_scale_by_a_known_length(win, app, monkeypatch, tmp_path):
    vp = _plan(win, app)

    def off_place(dlg):
        dlg.pos_mid.setChecked(True)
    name, seen = _place(win, app, monkeypatch, _dxf_file(tmp_path), off_place)
    ref = lambda: win.doc.model.references[name]  # noqa: E731
    at = lambda xy: ref().to_plane(xy, "m")  # noqa: E731
    assert at((0.0, 0.0)) != pytest.approx((0.0, 0.0)) and not ref().pinned  # in the middle of the window
    tool = win.state.tool
    assert tool.name == "ref_align"
    # the drawing's grid crossings (0, 0) and (12000, 5000) onto the model's A-1 and C-2
    _click(vp, app, (*at((0.0, 0.0)), 3.5))
    _click(vp, app, (0.0, 0.0, 3.5))
    assert at((0.0, 0.0)) == pytest.approx((0.0, 0.0), abs=1e-6) and "second point" in win.state.tool.hint()
    _click(vp, app, (*at((12000.0, 5000.0)), 3.5))
    _click(vp, app, (12.0, 5.0, 3.5))
    assert at((12000.0, 5000.0)) == pytest.approx((12.0, 5.0), abs=1e-6) and ref().pinned
    assert ref().scale == 1.0 and ref().rotation == pytest.approx(0.0, abs=1e-6)
    assert win.doc.undo_name() == "Align Reference" if hasattr(win.doc, "undo_name") else True
    # Scale: the 6 m bay of the drawing is really 6.6 m - the pinned reference asks first
    asked = []
    win.menu_actions["ref.scale"].trigger()
    app.processEvents()
    tool = win.state.tool
    assert tool.name == "ref_scale"
    tool.confirm_unpin = lambda n: asked.append(n) or False
    _click(vp, app, (0.0, 0.0, 3.5))
    assert asked == [name] and tool.points == []  # answered No: nothing started
    tool.confirm_unpin = lambda n: asked.append(n) or True
    _click(vp, app, (0.0, 0.0, 3.5))
    _click(vp, app, (6.0, 0.0, 3.5))
    assert "6 m long now" in tool.hint() and "type its real length" in tool.hint()
    vp._typed = "6.6"
    vp._apply_typed()
    app.processEvents()
    assert ref().scale == pytest.approx(1.1) and at((6000.0, 0.0)) == pytest.approx((6.6, 0.0), abs=1e-6)
    assert at((0.0, 0.0)) == pytest.approx((0.0, 0.0), abs=1e-6) and ref().pinned


@gui
def test_align_asks_about_the_scale_when_the_distances_differ(win, app, monkeypatch, tmp_path):
    vp = _plan(win, app)
    name, _ = _place(win, app, monkeypatch, _dxf_file(tmp_path))
    ref = lambda: win.doc.model.references[name]  # noqa: E731
    at = lambda xy: ref().to_plane(xy, "m")  # noqa: E731
    for answer, scale in ((False, 1.0), (True, 12.0 / 13.0)):
        cadref.change_reference(win.doc, name, pinned=False, scale=1.0, rotation=0.0, origin=(0.0, 0.0), base=(0.0, 0.0))
        win.menu_actions["ref.align"].trigger()
        app.processEvents()
        tool = win.state.tool
        asked = []
        tool.ask_scale = lambda drawn, real: asked.append((round(drawn, 3), round(real, 3))) or answer
        _click(vp, app, (0.0, 0.0, 3.5))
        _click(vp, app, (0.0, 0.0, 3.5))
        _click(vp, app, (*at((12000.0, -1000.0)), 3.5))  # an end of grid line C of the drawing: 12.04 from A-1
        tool.points[2] = (*at((13000.0, 0.0)), 3.5)  # (the drawing's point 13 m out, the model's 12 m out)
        _click(vp, app, (12.0, 0.0, 3.5))
        assert asked == [(13.0, 12.0)]
        assert ref().scale == pytest.approx(scale) and ref().pinned
        assert at((13000.0, 0.0)) == pytest.approx((12.0, 0.0) if answer else (13.0, 0.0), abs=1e-6)


@gui
def test_rotate_takes_the_typed_angle_as_degrees_also_with_lengths_shown_in_mm(win, app, monkeypatch, tmp_path):
    vp = _plan(win, app)
    name, _ = _place(win, app, monkeypatch, _dxf_file(tmp_path))
    win.doc.model.display_units = {"coordinates": "mm"}
    assert vp._length_unit()[1] == pytest.approx(1000.0)
    cadref.change_reference(win.doc, name, pinned=False)
    win.menu_actions["ref.rotate"].trigger()
    app.processEvents()
    _click(vp, app, (0.0, 0.0, 3.5))
    vp._typed = "30"
    vp._apply_typed()
    app.processEvents()
    assert win.doc.model.references[name].rotation == pytest.approx(30.0)
    # Edit > Rotate of the model's own objects: the same (30 typed was turned into 0.03)
    f = next(n for n, fr in win.doc.model.frames.items() if fr.story == "Story1" and fr.type == "Beam")
    a = win.doc.model.points[win.doc.model.frames[f].point_i]
    b = win.doc.model.points[win.doc.model.frames[f].point_j]
    before = math.degrees(math.atan2(b.y - a.y, b.x - a.x))
    win.state.set_selection([f], [])
    win.state.set_tool("rotate")
    app.processEvents()
    tool = win.state.tool
    if getattr(tool, "selecting", False):
        tool.done_selecting(win.state)
    _click(vp, app, (a.x, a.y, 3.5))
    vp._typed = "30"
    vp._apply_typed()
    app.processEvents()
    m = win.doc.model
    a2, b2 = m.points[m.frames[f].point_i], m.points[m.frames[f].point_j]
    assert (math.degrees(math.atan2(b2.y - a2.y, b2.x - a2.x)) - before) % 360 == pytest.approx(30.0, abs=1e-6)


@gui
def test_on_an_elevation(win, app, monkeypatch, tmp_path):
    vp = _plan(win, app)  # (the window on screen, turned to the elevation X-Z at Y = 5)
    vp.set_view("xz", ordinate=5.0)
    app.processEvents()
    name, seen = _place(win, app, monkeypatch, _dxf_file(tmp_path))
    ref = win.doc.model.references[name]
    assert seen["view"] == "xz" and (ref.view, ref.level, ref.stories) == ("xz", 5.0, [])
    b = vp._ref_actors[0].GetMapper().GetInput().GetBounds()
    assert b == pytest.approx((-1.0, 13.0, 5.0, 5.0, -1.0, 6.0))  # drawing x along X, drawing y up, on the plane Y = 5
    for w in win.views.viewports:
        if w is not vp:
            assert not getattr(w, "_ref_actors", [])
    vp.set_view("xz", ordinate=0.0)
    app.processEvents()
    assert vp._ref_actors == []
    vp.set_view("xz", ordinate=5.0)
    app.processEvents()
    win.state.set_tool("beam")
    x, y = vp._display((6.2, 5.0, 5.2))
    h = vp._snap_engine(int(x), int(y)).find(x, y)[0]
    assert (h.kind, h.cad, tuple(round(c, 4) for c in h.xyz)) == ("end", True, (6.2, 5.0, 5.2))


@gui
def test_the_list_of_references_and_the_layers(win, app, monkeypatch, tmp_path):
    from PySide6 import QtCore, QtWidgets
    from structstudio.app import ref_ui
    vp = _plan(win, app)
    name, _ = _place(win, app, monkeypatch, _dxf_file(tmp_path))
    dlg = ref_ui.ManageReferencesDialog(win)
    row = [dlg.table.item(0, c).text() for c in range(4)]
    assert row == [name, "plan.dxf", "Plan: Story1", "linked"] and dlg.current() == name
    assert dlg.size.text() == "14.00 x 7.00 m" and dlg.props.isEnabled()
    dlg.c_top.click()
    dlg.c_snap.click()
    dlg.color.setCurrentIndex(1)
    dlg._color()
    dlg.c_embed.click()
    app.processEvents()
    ref = win.doc.model.references[name]
    assert (ref.on_top, ref.snap, ref.color) == (False, False, "layers") and ref.embedded in win.doc.model.attachments
    assert dlg.table.item(0, 3).text() == "linked + embedded"
    dlg.unit.setCurrentIndex(ref_ui.UNIT_NAMES.index("cm"))
    dlg.unit.activated.emit(dlg.unit.currentIndex())
    assert win.doc.model.references[name].unit == "cm" and dlg.size.text() == "140.00 x 70.00 m"
    dlg.scale.setText("0.5")
    dlg._scale()
    assert win.doc.model.references[name].scale == 0.5 and dlg.size.text() == "70.00 x 35.00 m"
    # the layers: the window follows every tick, Cancel puts them back, OK is one undo step
    seen = []

    def layers(d2):
        d2.layers.set_all(False)
        app.processEvents()
        seen.append(len(vp._ref_actors))
        return QtWidgets.QDialog.Rejected
    monkeypatch.setattr(ref_ui.ReferenceLayersDialog, "exec", layers)
    dlg._layers()
    app.processEvents()
    assert seen == [0] and len(vp._ref_actors) == 1 and sorted(win.doc.model.references[name].layers_off) == ["FROZEN", "NOTES"]

    def layers_ok(d2):
        items = {d2.layers.list.item(k).data(QtCore.Qt.UserRole): d2.layers.list.item(k) for k in range(d2.layers.list.count())}
        assert set(items) == {"GRID", "COLUMN", "WALL", "NOTES", "FROZEN"} and items["GRID"].text() == "GRID    (5)"
        items["GRID"].setCheckState(QtCore.Qt.Unchecked)
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(ref_ui.ReferenceLayersDialog, "exec", layers_ok)
    steps = len(win.doc._undo)
    dlg._layers()
    app.processEvents()
    assert sorted(win.doc.model.references[name].layers_off) == ["FROZEN", "GRID", "NOTES"] and len(win.doc._undo) == steps + 1
    assert vp._ref_actors[0].GetMapper().GetInput().GetNumberOfLines() == 24 + 1 + 48
    # delete
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    dlg._delete()
    app.processEvents()
    assert not win.doc.model.references and not win.doc.model.attachments and dlg.table.rowCount() == 0
    assert not dlg.props.isEnabled() and vp._ref_actors == []
    dlg.close()
    dlg.deleteLater()


@gui
def test_the_tools_on_what_the_window_shows_and_after_an_undo(win, app, monkeypatch, tmp_path):
    vp = _plan(win, app)
    name, _ = _place(win, app, monkeypatch, _dxf_file(tmp_path), lambda dlg: dlg.align.setChecked(False))
    ref = lambda: win.doc.model.references[name]  # noqa: E731
    from PySide6 import QtWidgets
    said = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: said.append(a[2])))
    win.state.status.connect(said.append)
    # references hidden in the window: a tool chosen from the menu shows them again ...
    win.menu_actions["ref.show_window"].trigger()
    app.processEvents()
    assert vp.references_hidden and vp._ref_actors == []
    win.menu_actions["ref.move"].trigger()
    app.processEvents()
    assert not vp.references_hidden and len(vp._ref_actors) == 1 and win.menu_actions["ref.show_window"].isChecked()
    # ... and hidden while the tool is on: a click moves nothing that cannot be seen
    tool = win.state.tool
    tool.confirm_unpin = lambda n: True
    win.menu_actions["ref.show_window"].trigger()
    app.processEvents()
    _click(vp, app, (0.0, 0.0, 3.5))
    _click(vp, app, (1.0, 0.0, 3.5))
    assert ref().origin == (0.0, 0.0) and tool.points == [] and "hidden in this window" in " ".join(said)
    win.menu_actions["ref.show_window"].trigger()
    app.processEvents()
    # Scale by a Known Length: only the typed length resizes - a third click does nothing
    win.menu_actions["ref.scale"].trigger()
    app.processEvents()
    tool = win.state.tool
    tool.confirm_unpin = lambda n: True
    _click(vp, app, (0.0, 0.0, 3.5))
    _click(vp, app, (6.0, 0.0, 3.5))
    steps = len(win.doc._undo)
    _click(vp, app, (9.0, 3.0, 3.5))
    assert ref().scale == 1.0 and len(win.doc._undo) == steps and len(tool.points) == 2
    assert "type its real length" in tool.hint()
    vp._typed = "12"
    vp._apply_typed()
    app.processEvents()
    assert ref().scale == pytest.approx(2.0) and ref().pinned
    assert win.doc.undo() and ref().scale == 1.0
    # Undo in the middle of Align (after the first pair): the command starts again rather
    # than turn the drawing about a point it is no longer at
    cadref.change_reference(win.doc, name, pinned=False)
    win.menu_actions["ref.align"].trigger()
    app.processEvents()
    tool = win.state.tool
    _click(vp, app, (0.0, 0.0, 3.5))
    _click(vp, app, (3.0, 2.0, 3.5))
    assert ref().origin == pytest.approx((3.0, 2.0), abs=0.02) and len(tool.points) == 2
    assert win.doc.undo() and ref().origin == (0.0, 0.0)
    app.processEvents()
    del said[:]
    _click(vp, app, (6.0, 0.0, 3.5))
    assert tool.points == [] and "changed meanwhile" in " ".join(said) and ref().rotation == 0.0
    _click(vp, app, (6.0, 0.0, 3.5))  # the next click is the first point again
    assert len(tool.points) == 1
    win.state.set_tool("select")
    app.processEvents()
    assert not ref().pinned and ref().origin == (0.0, 0.0)  # (cancelled before a pair was done: nothing pinned)
    # a snap chosen for one pick (Shift + right-click) finds the drawing only while Snap to References is on
    win.state.set_tool("beam")
    app.processEvents()

    def end_at(xyz):
        x, y = vp._display(xyz)
        eng = vp._snap_engine(int(x), int(y))
        eng.override = "endpoint"
        return [(h.kind, h.cad) for h in eng.find(x, y)]
    assert ("end", True) in end_at((6.2, 5.2, 3.5))
    win.snap_actions["cad"].trigger()
    assert ("end", True) not in end_at((6.2, 5.2, 3.5))
    win.snap_actions["cad"].trigger()


@gui
def test_a_pdf_sheet_placed_through_the_menu_its_page_and_plot_scale(win, app, monkeypatch, tmp_path):
    pytest.importorskip("pypdf")
    from PySide6 import QtCore, QtTest, QtWidgets
    from structstudio.app import ref_ui
    vp = _plan(win, app)
    fn = _plan_pdf(tmp_path, pages=2)
    got = {}

    def answer(dlg):
        got.update(pdf=dlg.is_pdf, sheet=dlg.sheet.text(), plot=dlg.plot.text(), note=dlg.plot_note.text(),
                   pages=(dlg.page.minimum(), dlg.page.maximum()), info=dlg.info.text(), mid=dlg.pos_mid.isChecked(),
                   layers=[dlg.layers.list.item(k).text() for k in range(dlg.layers.list.count())])
        dlg.plot.setText("x")
        got["bad"] = dlg.size.text()
        with pytest.raises(ValueError):
            dlg.values()
        dlg.page.setValue(2)  # the other page: read, shown - and back
        got["page2"] = (dlg.info.text(), dlg.dwg.page)
        dlg.page.setValue(1)
        dlg.plot.setText("1:50")
        got["size50"] = dlg.size.text()
        dlg.plot.setText("100")
        # the picture: the lines of the layers ticked; a length measured on it sets the plot scale
        pv = dlg.preview
        dlg.resize(1180, 720)
        dlg.show()
        app.processEvents()
        got["shown"] = pv.shown_segments()
        dlg.layers.list.item(0).setCheckState(QtCore.Qt.Unchecked)
        got["shown_less"] = pv.shown_segments()
        dlg.layers.set_all(True)
        for xy in ((60.0, 80.0), (120.0, 80.0)):  # the grid crossings A-1 and B-1 of the sheet: 6 m apart
            x, y = pv.to_widget(xy)
            QtTest.QTest.mouseClick(pv, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(x) + 2, int(y) - 2))
        got["measured"] = (round(pv.measured_length(), 6), dlg.measure_text.text(), dlg.b_set.isEnabled())
        dlg.real.setText("12")
        dlg._calibrate()
        got["calibrated"] = (dlg.plot.text(), dlg.plot_note.text(), dlg.size.text())
        dlg.plot.setText("100")
        dlg.hide()
    name, seen = _place(win, app, monkeypatch, fn, answer)
    assert got["pdf"] and got["sheet"] == "420 x 297 mm (A3)" and got["plot"] == "100" and "does not say" in got["note"]
    assert got["pages"] == (1, 2) and got["info"].startswith("78 lines on 2 layer(s)") and got["mid"]
    assert got["layers"] == ["Pen 0.18 mm #ff0000    (5)", "Pen 0.35 mm #000000    (73)"]
    assert got["bad"] == "-" and got["page2"] == ("1 line on 1 layer(s)", 2) and got["size50"] == "7.00 m x 3.50 m"
    assert seen["size"] == "14.00 m x 7.00 m"
    assert (got["shown"], got["shown_less"]) == (78, 73)
    assert got["measured"][0] == 60.0 and got["measured"][1].startswith("Measured: 6 m") and got["measured"][2]
    assert got["calibrated"] == ("200", "(set by the measured length)", "28.00 m x 14.00 m")
    ref = lambda: win.doc.model.references[name]  # noqa: E731
    assert (ref().unit, ref().scale, ref().page, ref().pinned) == ("mm", 100.0, 1, False)
    assert win.state.tool.name == "ref_align" and len(vp._ref_actors) == 1
    assert vp._ref_actors[0].GetMapper().GetInput().GetNumberOfLines() == 78
    assert win.qsettings.value("references/plot_scale") in (100.0, "100.0", "100")
    # put in place with two points: the sheet's grid crossings A-1 and C-2 onto the model's
    at = lambda xy: ref().to_plane(xy, "m")  # noqa: E731
    _click(vp, app, (*at((60.0, 80.0)), 3.5))
    _click(vp, app, (0.0, 0.0, 3.5))
    _click(vp, app, (*at((180.0, 130.0)), 3.5))
    _click(vp, app, (12.0, 5.0, 3.5))
    assert at((180.0, 130.0)) == pytest.approx((12.0, 5.0), abs=1e-6) and ref().pinned and ref().scale == 100.0
    # snapping to the sheet's lines: a column corner
    win.state.set_tool("beam")
    app.processEvents()
    x, y = vp._display((6.2, 5.2, 3.5))
    hit = vp._snap_engine(int(x), int(y)).find(x, y)[0]
    assert (hit.kind, hit.cad) == ("end", True) and hit.xyz == pytest.approx((6.2, 5.2, 3.5), abs=1e-6)
    win.state.set_tool("select")
    # the list: a sheet has a plot scale and a page, no drawing unit to choose
    dlg = ref_ui.ManageReferencesDialog(win)
    dlg.show()
    app.processEvents()
    assert dlg.scale_label.text() == "Plot scale  1 :" and not dlg.unit.isEnabled() and dlg.scale.text() == "100"
    assert dlg.page.isVisible() and (dlg.page.minimum(), dlg.page.maximum(), dlg.page.value()) == (1, 2, 1)
    assert dlg.size.text() == "14.00 x 7.00 m"
    dlg.scale.setText("50")
    dlg._scale()
    assert ref().scale == 50.0 and dlg.size.text() == "7.00 x 3.50 m"
    dlg.page.setValue(2)
    app.processEvents()
    assert ref().page == 2 and vp._ref_actors[0].GetMapper().GetInput().GetNumberOfLines() == 1
    assert win.doc.undo() and ref().page == 1
    dlg.close()
    dlg.deleteLater()
    # a second, DXF reference: its unit is chosen, its scale is an extra scale
    name2, _ = _place(win, app, monkeypatch, _dxf_file(tmp_path), lambda d2: d2.align.setChecked(False))
    dlg = ref_ui.ManageReferencesDialog(win)
    dlg.show()
    dlg.table.selectRow([dlg.table.item(r, 0).text() for r in range(dlg.table.rowCount())].index(name2))
    app.processEvents()
    assert dlg.scale_label.text() == "Extra scale" and dlg.unit.isEnabled() and not dlg.page.isVisible()
    dlg.close()
    dlg.deleteLater()


@gui
def test_the_forms_show_the_drawing_and_a_measured_length_scales_a_dxf(win, app, monkeypatch, tmp_path):
    from PySide6 import QtCore, QtTest, QtWidgets
    from structstudio.app import ref_ui
    vp = _plan(win, app)
    got = {}

    def answer(dlg):
        pv = dlg.preview
        dlg.resize(1180, 720)
        dlg.show()
        app.processEvents()
        got["shown"] = pv.shown_segments()
        got["dark"] = int((pv._buf.sum(axis=2) < 600).sum())
        items = {dlg.layers.list.item(k).data(QtCore.Qt.UserRole): dlg.layers.list.item(k) for k in range(dlg.layers.list.count())}
        got["swatch"] = not items["GRID"].icon().isNull()
        items["NOTES"].setCheckState(QtCore.Qt.Checked)  # a layer that is off in the file, ticked: drawn at once
        got["more"] = pv.shown_segments()
        items["NOTES"].setCheckState(QtCore.Qt.Unchecked)
        # the wheel zooms about the cursor, a double click fits again
        k0 = pv._scale
        pv.wheelEvent(_wheel(pv, 120))
        got["zoomed"] = pv._scale / k0
        QtTest.QTest.mouseDClick(pv, QtCore.Qt.LeftButton)
        got["fit"] = pv._scale / k0
        # the 6 m bay measured on the picture (the clicks snap to the ends of the grid lines' crossing columns)
        for xy in ((-200.0, -200.0), (5800.0, -200.0)):
            x, y = pv.to_widget(xy)
            QtTest.QTest.mouseClick(pv, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(x) + 1, int(y) + 1))
        got["measured"] = (round(pv.measured_length(), 6), dlg.measure_text.text())
        dlg.unit.setCurrentIndex(ref_ui.UNIT_NAMES.index("cm"))  # a wrong unit shows on the picture and in the length
        dlg._measured(pv.measured_length())
        got["as_cm"] = dlg.measure_text.text()
        dlg.unit.setCurrentIndex(ref_ui.UNIT_NAMES.index("mm"))
        dlg.real.setText("zero")
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: got.setdefault("bad", a[2])))
        dlg._calibrate()
        dlg.real.setText("6.6")  # that bay is really 6.6 m: the drawing is scaled by 1.1
        dlg._calibrate()
        got["size"] = dlg.size.text()
        dlg.align.setChecked(False)
        dlg.hide()
    name, _ = _place(win, app, monkeypatch, _dxf_file(tmp_path), answer)
    assert got["shown"] == 78 and got["more"] == 79 and got["dark"] > 500 and got["swatch"]
    assert got["zoomed"] == pytest.approx(1.25) and got["fit"] == pytest.approx(1.0)
    assert got["measured"][0] == 6000.0 and got["measured"][1].startswith("Measured: 6 m")
    assert got["as_cm"].startswith("Measured: 60 m") and "positive number" in got["bad"]
    assert got["size"].startswith("15.40 m x 7.70 m") and "scaled x 1.1" in got["size"]
    ref = win.doc.model.references[name]
    assert ref.scale == pytest.approx(1.1) and ref.unit == "mm"
    assert ref.to_plane((6000.0, 0.0), "m") == pytest.approx((6.6, 0.0))
    # the layers form: the picture beside the ticks (and the window) follows them
    seen = []

    def layers(d2):
        d2.show()
        app.processEvents()
        seen.append(d2.preview.shown_segments())
        d2.layers.set_all(False)
        seen.append(d2.preview.shown_segments())
        d2.hide()
        return QtWidgets.QDialog.Rejected
    monkeypatch.setattr(ref_ui.ReferenceLayersDialog, "exec", layers)
    win.reference_layers(name)
    app.processEvents()
    assert seen == [78, 0] and len(vp._ref_actors) == 1
    # the list: the picture of the reference selected
    dlg = ref_ui.ManageReferencesDialog(win)
    dlg.show()
    app.processEvents()
    assert dlg.preview.shown_segments() == 78 and dlg.preview.dwg is cadref.drawing_of(ref, win.doc.model)[0]
    cadref.delete_references(win.doc, [name])
    dlg.refresh()
    assert dlg.preview.dwg is None and dlg.preview.shown_segments() == 0
    dlg.close()
    dlg.deleteLater()


def _wheel(widget, delta):
    from PySide6 import QtCore, QtGui
    pos = QtCore.QPointF(widget.width() / 2, widget.height() / 2)
    return QtGui.QWheelEvent(pos, widget.mapToGlobal(pos), QtCore.QPoint(0, 0), QtCore.QPoint(0, delta),
                             QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
