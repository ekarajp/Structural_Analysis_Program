"""Built-in catalogs (steel sections, rebar sizes) against the published
standards, CSV import, and the project's rebar sizes in the model."""
import math

import pytest

from structstudio.catalog import (CatalogError, find_steel, read_rebar, read_steel, rebar_catalog, rebar_csv,
                                  steel_catalog, steel_csv)
from structstudio.model import Model
from structstudio.model.section_shapes import SHAPES, shape_properties
from structstudio.units import UnitSystem


def test_catalog_contents():
    cat = steel_catalog()
    assert set(cat) == {"TIS 1227 / JIS G3192", "TIS 1228 / JIS G3350", "TIS 107-2566", "GB/T 11263", "GB/T 706",
                        "AISC", "EN"}
    series = {(e.standard, e.series) for es in cat.values() for e in es}
    for s in [("TIS 1227 / JIS G3192", "H"), ("TIS 1227 / JIS G3192", "Channel"), ("TIS 1227 / JIS G3192", "L equal"),
              ("TIS 1227 / JIS G3192", "L unequal"), ("TIS 1228 / JIS G3350", "Lipped C"), ("AISC", "W"),
              ("AISC", "HSS rect"), ("EN", "IPE"), ("EN", "HEB"), ("EN", "UPN"), ("EN", "IPN"),
              ("TIS 1227 / JIS G3192", "I-beam"), ("GB/T 11263", "HN"), ("GB/T 706", "Channel"), ("AISC", "S"),
              ("TIS 1227 / JIS G3192", "T"), ("TIS 1228 / JIS G3350", "Cold-formed C"),
              ("TIS 1228 / JIS G3350", "Cold-formed Z"), ("TIS 1228 / JIS G3350", "Cold-formed L"),
              ("TIS 1228 / JIS G3350", "Hat"), ("TIS 107-2566", "Pipe (JIS G3444)"),
              ("TIS 107-2566", "Square tube (JIS G3466)"), ("TIS 107-2566", "Rectangular tube (JIS G3466)"),
              ("TIS 107-2566", "Square tube (ASTM A500)"), ("TIS 107-2566", "Rectangular tube (EN 10219)")]:
        assert s in series
    for es in cat.values():
        names = [e.name for e in es]
        assert len(names) == len(set(names))


@pytest.mark.parametrize("name, A, I33, I22", [
    # TIS 1227-2558 table 5 (= JIS G3192)
    ("H300x150x6.5x9", 46.78, 7210, 508),
    ("H400x400x13x21", 218.7, 66600, 22400),
    ("H588x300x12x20", 192.5, 118000, 9020),
    # TIS 1227 table 2 / 3 / 4 / 6 (channels and I-beams with tapered flanges)
    ("L100x100x10", 19.00, 175, 175),
    ("L150x90x12", 27.36, 619, 167),
    ("C150x75x6.5x10", 23.71, 861, 117),
    ("I300x150x10x18.5", 83.47, 12700, 886),
    # TIS 1228-2561 tables A.1 - A.6
    ("C150x65x20x2.3", 7.012, 248, 41.1),
    ("C100x50x20x4.5", 9.469, 139, 30.9),
    ("C125x50x20x4", 9.548, 217, 31.1),  # Iy misprinted 33.1 in TIS 1228; JIS G3350: 31.1
    ("L60x60x3.2", 3.672, 13.1, 13.1),
    ("L75x30x3.2", 3.192, 18.9, 1.94),
    ("C450x75x6", 34.82, 8400, 122),
    ("C100x50x2.3", 4.426, 69.9, 11.1),
    ("Z100x50x50x3.2", 6.063, 93.6, 24.2),
    ("Z75x40x30x2.3", 3.161, 26.8, 6.15),
    ("Z100x50x20x3.2", 7.007, 107, 44.8),
    ("HAT60x30x25x2.3", 4.358, 20.9, 14.7),
    ("HAT50x40x30x3.2", 5.932, 20.9, 35.9),
    # TIS 107 (JIS G3444 / G3466 sizes; TIS 107-2533 tables 1-3)
    ("PIPE21.7x2", 1.238, 0.607, 0.607),
    ("PIPE216.3x8", 52.35, 2840, 2840),
    ("BOX100x100x3.2", 12.13, 187, 187),
    ("BOX200x200x9", 66.67, 3990, 3990),
    ("BOX200x100x6", 33.63, 1700, 577),
    ("BOX100x50x4", 10.95, 134, 45.0),  # TIS 107-2533 prints 142 / 46.7; JIS G3466 and TIS 107-2566: 134 / 45.0
])
def test_tis_values_as_published(name, A, I33, I22):
    e = find_steel(name)
    assert e.standard.startswith("TIS")
    assert e.A == pytest.approx(A, rel=0.001) and e.I33 == pytest.approx(I33, rel=0.003)
    assert e.I22 == pytest.approx(I22, rel=0.003)
    # and the drawn shape agrees with the table (channels taper: I22 differs)
    dims = dict(e.dims)
    if e.slope:
        dims["slope"] = e.slope
    p = shape_properties(SHAPES[e.shape], dims, False, False)
    assert p.A / 100 == pytest.approx(A, rel=0.006)
    assert p.I33 / 1e4 == pytest.approx(I33, rel=0.02)
    assert p.I22 / 1e4 == pytest.approx(I22, rel=0.07)  # lipped C: thin-walled corners


def test_tis_tee_and_tubes_drawn_to_the_table():
    """Every Thai catalog row: the drawn section gives the tabulated A and I."""
    for e in steel_catalog()["TIS 107-2566"] + steel_catalog()["TIS 1228 / JIS G3350"]:
        p = shape_properties(SHAPES[e.shape], dict(e.dims), False, False)
        assert p.A / 100 == pytest.approx(e.A, rel=0.036), e.name  # thick tubes: A printed with 2t, I with 2.5t
        assert p.I33 / 1e4 == pytest.approx(e.I33, rel=0.03, abs=0.01), e.name
    tee = find_steel("T150x9")  # TIS 1227 table 7; stem drawn with the uniform thickness of its 482.6 mm2
    assert tee.A == 18.52 and tee.I33 == 16.5 and tee.I22 == 254
    p = shape_properties(SHAPES["Tee"], dict(tee.dims), False, False)
    assert p.I22 / 1e4 == pytest.approx(254, rel=0.01)


def test_en_and_aisc_values():
    ipe = find_steel("IPE 300")  # EN 10365: A 53.81 cm2, Iy 8356 cm4, Iz 603.8 cm4
    assert ipe.A == pytest.approx(53.81, rel=0.002)
    assert ipe.I33 == pytest.approx(8356, rel=0.002) and ipe.I22 == pytest.approx(603.8, rel=0.002)
    heb = find_steel("HE 300 B")  # A 149.1, Iy 25170, Iz 8563
    assert heb.A == pytest.approx(149.1, rel=0.002) and heb.I33 == pytest.approx(25170, rel=0.002)
    w = find_steel("W14X90")  # AISC: A 26.5 in2, Ix 999 in4, J 4.06 in4
    assert w.A == pytest.approx(26.5 * 6.4516, rel=1e-3)
    assert w.I33 == pytest.approx(999 * 41.6231, rel=1e-3) and w.J == pytest.approx(4.06 * 41.6231, rel=1e-3)
    ang = find_steel("L8X4X1")  # long leg vertical: Ix 69.7 in4 > Iy 11.6 in4
    assert ang.dims["t3"] > ang.dims["t2"] and ang.I33 > ang.I22


@pytest.mark.parametrize("name, std, A, I33, I22", [
    # GB/T 11263-2017 (HN 1000x300: A 315.5, Ix 472000, Iy 11500 in the standard)
    ("HN980x298x17x26", "GB/T 11263", 315.5, 472000, 11500),
    ("HW100x100x6x8", "GB/T 11263", 21.58, 378, 134),  # GB r = 8 (JIS 10): smaller area than TIS 21.90
    # GB/T 706-2016 table A.1 / A.2 (Ix / Iy of the standard)
    ("I20a", "GB/T 706", 35.55, 2370, 158),
    ("I36b", "GB/T 706", 83.64, 16500, 582),
    ("[10", "GB/T 706", 12.74, 198, 25.6),
    ("[40c", "GB/T 706", 91.04, 19700, 688),
])
def test_gb_values(name, std, A, I33, I22):
    e = find_steel(name, std)
    assert e.A == pytest.approx(A) and e.I33 == pytest.approx(I33, rel=0.01) and e.I22 == pytest.approx(I22, rel=0.01)
    if e.series in ("I-beam", "Channel"):
        assert e.slope == pytest.approx(1 / 6 if e.series == "I-beam" else 0.1)


def test_tapered_european_and_american_shapes():
    upn = find_steel("UPN 100")  # EN 10279: A 13.5 cm2, Iy 206 cm4, Iz 29.3 cm4
    assert upn.slope == 0.08 and upn.A == pytest.approx(13.5, rel=0.003)
    assert upn.I33 == pytest.approx(206, rel=0.005) and upn.I22 == pytest.approx(29.3, rel=0.015)
    ipn = find_steel("IPN 200")  # A 33.4, Iy 2140, Iz 117
    assert ipn.A == pytest.approx(33.4, rel=0.003) and ipn.I33 == pytest.approx(2140, rel=0.005)
    c = find_steel("C15X50")  # AISC C: slope 1:6 drawn; the drawn I22 now near the table (11.0 in4)
    assert c.slope == pytest.approx(1 / 6)
    p = shape_properties(SHAPES["Channel"], dict(c.dims, slope=c.slope), False, False)
    assert p.I22 / 1e4 == pytest.approx(11.0 * 41.6231 / 1e0, rel=0.03)


def test_gb_rebar():
    gb = {e.name: e for e in rebar_catalog()["GB 1499 (HPB / HRB)"]}
    assert gb["\u03a618"].area == 254.5 and gb["\u03a625"].mass == 3.85 and gb["\u03a650"].area == 1964


def test_entry_to_frame_section_in_model_units():
    e = find_steel("H300x150x6.5x9")
    for units in (UnitSystem("kN", "m"), UnitSystem("kgf", "cm"), UnitSystem("kip", "in")):
        sec = e.frame_section("SS400", units)
        L = units.factor("length")
        assert sec.A * L ** 2 == pytest.approx(46.78e-4)
        assert sec.I33 * L ** 4 == pytest.approx(7210e-8)
        assert sec.t3 * L == pytest.approx(0.300) and sec.r * L == pytest.approx(0.013)
        assert sec.J > 0 and sec.As2 > 0 and sec.source == "TIS 1227 / JIS G3192"


def test_rebar_catalog():
    cat = rebar_catalog()
    db = {e.name: e for e in cat["TIS 24 (deformed)"]}
    assert db["DB16"].area == 201.1 and db["DB16"].mass == 1.578 and db["DB40"].area == 1256.6
    rb = {e.name: e for e in cat["TIS 20 (round, SR24)"]}
    assert rb["RB9"].area == 63.6 and rb["RB6"].diameter == 6
    us = {e.name: e for e in cat["ASTM A615"]}
    assert us["#5"].area == pytest.approx(0.31 * 645.16, abs=0.1)
    jis = {e.name: e for e in cat["JIS G3112"]}
    assert jis["D25"].area == pytest.approx(506.7) and jis["D10"].diameter == 9.53
    assert "EN 10080" in cat


def test_read_steel_csv_and_round_trip():
    text = ("name,shape,t3,t2,tf,tw,r,A,I33,I22\n"
            "MY-H200,I,200,100,8,5.5,11,27.16,1840,134\n")
    es = read_steel(text, default_standard="Supplier")
    assert es[0].standard == "Supplier" and es[0].dims["t3"] == 200 and es[0].J is None
    again = read_steel(steel_csv(es))
    assert again[0].name == "MY-H200" and again[0].A == 27.16 and again[0].dims == es[0].dims
    with pytest.raises(CatalogError, match="unknown shape"):
        read_steel("name,shape,t3,A,I33,I22\nX,Hexagon,1,1,1,1\n")
    with pytest.raises(CatalogError, match="missing column"):
        read_steel("name,shape\nX,I\n")
    with pytest.raises(CatalogError):
        read_steel("name,shape,t3,t2,tf,tw,A,I33,I22\nX,I,200,100,120,5,1,1,1\n")  # flanges thicker than the depth


def test_read_rebar_csv():
    es = read_rebar("name,diameter\nB14,14\n")
    assert es[0].area == pytest.approx(math.pi * 49) and es[0].standard == "Imported"
    with pytest.raises(CatalogError):
        read_rebar("name,diameter\nB0,0\n")


def test_model_rebar_sizes_saved_and_converted():
    m = Model("t", "kN m")
    for e in rebar_catalog()["TIS 24 (deformed)"][:4]:
        s = e.rebar_size(m.units)
        m.rebar_sizes[s.name] = s
    back = Model.from_json(m.to_json())
    assert list(back.rebar_sizes) == ["DB6", "DB8", "DB10", "DB12"]
    assert back.rebar_sizes["DB12"].area == pytest.approx(113.1e-6)
    cm = m.convert_units("kgf cm")
    assert cm.rebar_sizes["DB12"].diameter == pytest.approx(1.2) and cm.rebar_sizes["DB12"].area == pytest.approx(1.131)
    text = rebar_csv(list(cm.rebar_sizes.values()), cm.units)
    again = read_rebar(text)
    assert again[3].name == "DB12" and again[3].diameter == pytest.approx(12) and again[3].area == pytest.approx(113.1)
