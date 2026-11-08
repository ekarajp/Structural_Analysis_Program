"""Import of ETABS / SAP2000 / SAFE files: database tables (text, Excel), legacy
keyword files, older ETABS text files, format detection."""
import os

import numpy as np
import pytest

from structstudio import analyze
from structstudio.io.csi_import import CsiImporter, parse_units
from structstudio.io.csi_tables import Table, TableSet, read_text_tables, write_text_tables
from structstudio.io.importers import detect, import_model

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "import")


def path(name):
    return os.path.join(DATA, name)


# ----------------------------------------------------------------------------
def test_text_tables_continuation_quotes_and_roundtrip():
    text = '''File x.s2k was saved on 1/1/20
TABLE:  "PROGRAM CONTROL"
   ProgramName=SAP2000   Version=23.0.0   CurrUnits="KN, m, C"
TABLE:  "CONNECTIVITY - AREA"
   Area=1   NumJoints=5   Joint1=1   Joint2=2 _
      Joint3=3   Joint4=4   Joint5=5
   Area=2   NumJoints=3   Joint1=1   Joint2=2   Joint3=3   Notes="two words"
END TABLE DATA
TABLE:  "IGNORED"
   A=1
'''
    ts = read_text_tables(text)
    assert ts.program_control()["currunits"] == "KN, m, C"
    rows = ts.rows("connectivity - area")
    assert rows[0]["joint5"] == "5" and rows[1]["notes"] == "two words"
    assert "IGNORED" not in ts
    back = read_text_tables(write_text_tables(ts))
    assert back.rows("Connectivity - Area") == rows
    assert parse_units("Kip-in-F").length == "in" and parse_units("kN_m_C").force == "kN"
    assert parse_units("Tonf, m, C").force == "tonf"


def test_detect_formats():
    assert detect(path("sap2000_v23_portal.s2k")) == "tables"
    assert detect(path("safe_v12_slab.f2k")) == "tables"
    assert detect(path("etabs_v9_column.e2k")) == "e2k"
    assert detect(path("sap2000_v7_cantilever.s2k")) == "s2k_legacy"


def test_sap2000_v23_tables_model_and_results():
    res = import_model(path("sap2000_v23_portal.s2k"))
    m = res.model
    assert res.program == "SAP2000" and res.version.startswith("23")
    assert m.units.force == "kN" and m.grid_mode == "xyz" and [n for n, _ in m.levels()] == ["Z1", "Z2"]
    assert m.frame_sections["BM3060"].modifiers.I33 == pytest.approx(0.35)
    assert m.frame_sections["COL40"].I33 == pytest.approx(0.0021333333)
    assert m.frames["3"].releases_i[4] and m.frames["3"].releases_j[4] and not m.frames["3"].releases_i[5]
    assert m.frames["3"].type == "Beam" and m.frames["1"].type == "Column"
    kinds = {n: type(c).__name__ for n, c in m.load_cases.items()}
    assert kinds == {"DEAD": "LinearStaticCase", "MODAL": "ModalCase", "LIVE": "LinearStaticCase",
                     "WIND": "LinearStaticCase", "RSX": "ResponseSpectrumCase", "PDL": "PDeltaCase"}
    assert m.load_cases["PDL"].loads == {"DEAD": 1.2, "LIVE": 1.6}
    rs = m.load_cases["RSX"]
    assert rs.modal_case == "MODAL" and rs.loads[0].direction == "U1" and rs.loads[0].scale == pytest.approx(9.80665)
    assert m.functions["FUNC1"].periods == [0.0, 0.5, 1.0, 4.0]
    env = m.load_combinations["ENV"]
    assert env.type == "Envelope" and [(i.name, i.is_combo) for i in env.items] == [("ULS1", True), ("WIND", False)]
    ms = m.mass_sources["MSSSRC1"]
    assert ms.include_vertical and ms.load_patterns == {"LIVE": 0.25}
    assert not m.frames["1"].auto_mesh and not m.areas["1"].auto_mesh  # SAP2000: objects as drawn
    r = analyze(m)
    assert r.base_reactions("LIVE")[2] == pytest.approx(10 * 6 + 30 + 2 * 30)
    wd = 23.5631216161854
    assert r.base_reactions("DEAD")[2] == pytest.approx(wd * (2 * 4 * 0.16 + 6 * 0.18 + 30 * 0.2), rel=1e-9)
    assert r.base_reactions("WIND")[0] == pytest.approx(-20.0)
    assert r.base_reactions("RSX")[0] > 0


def test_sap2000_v10_old_names_cantilever_exact():
    m = import_model(path("sap2000_v10_cantilever.s2k")).model
    assert set(m.load_patterns) == {"DEAD", "TIP"} and m.units.length == "in"
    assert m.load_combinations["COMB1"].items[1].scale == pytest.approx(1.7)
    r = analyze(m)
    assert r.joint_displacement("2", "TIP")[2] == pytest.approx(-120 ** 3 / (3 * 29000 * 200), rel=1e-9)


def test_safe_v12_slab_on_columns():
    res = import_model(path("safe_v12_slab.f2k"))
    m = res.model
    assert res.program == "SAFE"
    assert m.shell_sections["SLAB25"].model_type == "ShellThick"
    assert m.frame_sections["COL50"].t3 == pytest.approx(0.5)
    assert all(m.points[p].restraints == (True,) * 6 for p in "5678")
    assert any("SAFE" in w for w in res.warnings)
    r = analyze(m)
    assert r.base_reactions("LIVE")[2] == pytest.approx(3 * 64 + 50)
    assert r.base_reactions("DEAD")[2] == pytest.approx(64 * 0.25 * 24 + 4 * 3 * 0.25 * 24)


def test_safe_soil_springs_uniform_settlement():
    text = '''TABLE:  "PROGRAM CONTROL"
   ProgramName=SAFE   Version=16.0.0   CurrUnits="KN, m, C"
TABLE:  "ACTIVE DEGREES OF FREEDOM"
   UX=No   UY=No   UZ=Yes   RX=Yes   RY=Yes   RZ=No
TABLE:  "MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES"
   Material=C30   UnitWeight=0   UnitMass=0   E1=25000000   U12=0.2
TABLE:  "SLAB PROPERTIES 02 - SOLID SLABS"
   Slab=MAT60   Type=Mat   MatProp=C30   Thickness=0.6
TABLE:  "OBJ GEOM - POINT COORDINATES"
   Point=1   GlobalX=0   GlobalY=0   GlobalZ=0
   Point=2   GlobalX=10   GlobalY=0   GlobalZ=0
   Point=3   GlobalX=10   GlobalY=6   GlobalZ=0
   Point=4   GlobalX=0   GlobalY=6   GlobalZ=0
TABLE:  "OBJ GEOM - AREAS 01 - GENERAL"
   Area=1   NumPoints=4   Point1=1   Point2=2   Point3=3   Point4=4
TABLE:  "SLAB PROPERTY ASSIGNMENTS"
   Area=1   SlabProp=MAT60
TABLE:  "SOIL PROPERTIES"
   Soil=S1   Subgrade=20000   NonlinOpt="Compression Only"
TABLE:  "SOIL PROPERTY ASSIGNMENTS"
   Area=1   SoilProp=S1
TABLE:  "LOAD PATTERNS"
   LoadPat=Q   Type=LIVE   SelfWtMult=0
TABLE:  "LOAD ASSIGNMENTS - SURFACE LOADS"
   Area=1   LoadPat=Q   Dir=Gravity   UnifLoad=50
'''
    imp = CsiImporter.from_tables(read_text_tables(text))
    m = imp.model
    assert m.areas["1"].spring == pytest.approx(20000) and any("linear" in w for w in imp.warnings)
    assert "Q" in m.load_cases  # no case table: one linear case per pattern
    r = analyze(m)
    for p in "1234":
        assert r.joint_displacement(p, "Q")[2] == pytest.approx(-50 / 20000, rel=1e-6)
    assert r.base_reactions("Q")[2] == pytest.approx(50 * 60, rel=1e-9)


def _etabs_tables():
    """ETABS 21 style tables with Excel display names (no field-key file needed)."""
    def t(name, fields, rows, units=None):
        from structstudio.io.csi_tables import norm
        tab = Table(name=name, fields=list(fields))
        tab.rows = [{norm(f): str(v) for f, v in zip(fields, r) if v != ""} for r in rows]
        tab.units = units or {}
        return tab
    tabs = [
        t("Program Control", ["ProgramName", "Version", "CurrUnits"], [["ETABS", "21.1.0", "kN, m, C"]]),
        t("Story Definitions", ["Tower", "Name", "Height", "Master Story"], [["T1", "Story1", 3.5, "Yes"]]),
        t("Tower and Base Story Definition", ["Tower", "BSName", "BSElev"], [["T1", "Base", 0]]),
        t("Material Properties - General", ["Material", "Type"], [["C30", "Concrete"]]),
        t("Material Properties - Basic Mechanical Properties", ["Material", "UnitWeight", "UnitMass", "E1", "U12"],
          [["C30", 0, 0, 25e6, 0.2]]),
        t("Frame Section Property Definitions - Concrete Rectangular", ["Name", "Material", "Depth", "Width",
                                                                        "I33 Modifier"],
          [["C40", "C30", 0.4, 0.4, 1], ["B30", "C30", 0.6, 0.3, 0.5]]),
        t("Point Object Connectivity", ["UniqueName", "Is Auto Point", "Story", "PointBay", "X", "Y", "Z"],
          [[1, "No", "Base", 1, 0, 0, 0], [2, "No", "Story1", 1, 0, 0, 3.5], [3, "No", "Story1", 2, 6, 0, 3.5],
           [4, "No", "Base", 2, 6, 0, 0]], {"x": "m"}),
        t("Column Object Connectivity", ["Unique Name", "Story", "ColumnBay", "UniquePtI", "UniquePtJ"],
          [["C1", "Story1", "C1", 1, 2], ["C2", "Story1", "C2", 4, 3]]),
        t("Beam Object Connectivity", ["Unique Name", "Story", "BeamBay", "UniquePtI", "UniquePtJ"],
          [["B1", "Story1", "B1", 2, 3]]),
        t("Frame Assignments - Section Properties", ["Story", "Label", "UniqueName", "Shape", "Section Property"],
          [["Story1", "C1", "C1", "Concrete Rectangular", "C40"], ["Story1", "C2", "C2", "Concrete Rectangular", "C40"],
           ["Story1", "B1", "B1", "Concrete Rectangular", "B30"]]),
        t("Joint Assignments - Restraints", ["Story", "Label", "UniqueName", "UX", "UY", "UZ", "RX", "RY", "RZ"],
          [["Base", 1, 1] + ["Yes"] * 6, ["Base", 2, 4] + ["Yes"] * 6]),
        t("Load Pattern Definitions", ["Name", "Is Auto Load", "Type", "Self Weight Multiplier"],
          [["LAT", "No", "Other", 0], ["GRAV", "No", "Live", 0]]),
        t("Load Case Definitions - Summary", ["Name", "Type"], [["LAT", "Linear Static"], ["GRAV", "Linear Static"]]),
        t("Load Case Definitions - Linear Static", ["Name", "Load Type", "Load Name", "Load SF"],
          [["LAT", "Load Pattern", "LAT", 1], ["GRAV", "Load Pattern", "GRAV", 1]]),
        t("Joint Loads Assignments - Force", ["Story", "Label", "UniqueName", "Load Pattern", "FX", "FY", "FZ"],
          [["Story1", 1, 2, "LAT", 10, 0, 0]]),
        t("Frame Loads Assignments - Distributed", ["Story", "Label", "UniqueName", "Load Pattern", "Load Type",
                                                    "Direction", "Distance Type", "Relative Distance A",
                                                    "Relative Distance B", "Force A", "Force B"],
          [["Story1", "B1", "B1", "GRAV", "Force", "Gravity", "Relative", 0, 1, 12, 12]]),
        t("Load Combination Definitions", ["Name", "Type", "Load Name", "SF"],
          [["COMB1", "Linear Add", "LAT", 1.5], ["COMB1", "", "GRAV", 1.2]]),
    ]
    ts = TableSet(container="xlsx")
    for tab in tabs:
        ts.add(tab)
    return ts


def _write_xlsx(ts, file):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    from structstudio.io.csi_tables import norm
    for k, tab in enumerate(ts.tables.values()):
        ws = wb.create_sheet(f"S{k}")
        ws.append([f"TABLE:  {tab.name}"])
        ws.append(tab.fields)
        ws.append([tab.units.get(norm(f), "") for f in tab.fields] if tab.units else ["" for _ in tab.fields])
        for row in tab.rows:
            vals = []
            for f in tab.fields:
                v = row.get(norm(f), None)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    pass
                vals.append(v)
            ws.append(vals)
    wb.save(file)


def test_etabs_tables_from_excel_story_model(tmp_path):
    f = str(tmp_path / "frame.xlsx")
    _write_xlsx(_etabs_tables(), f)
    res = import_model(f)
    m = res.model
    assert res.program == "ETABS" and res.kind == "tables"
    assert m.grid_mode == "story" and [(s.name, s.elevation) for s in m.stories_bottom_up()] == [("Base", 0.0),
                                                                                                 ("Story1", 3.5)]
    assert m.points["2"].story == "Story1" and m.points["2"].label == "1"
    assert m.frames["C1"].type == "Column" and m.frames["B1"].section == "B30"
    assert m.frame_sections["B30"].modifiers.I33 == pytest.approx(0.5)
    assert m.frame_sections["C40"].t3 == pytest.approx(0.4) and m.frame_sections["C40"].A == pytest.approx(0.16)
    assert [(i.name, i.scale) for i in m.load_combinations["COMB1"].items] == [("LAT", 1.5), ("GRAV", 1.2)]
    r = analyze(m)
    assert r.base_reactions("LAT")[0] == pytest.approx(-10.0)
    assert r.base_reactions("GRAV")[2] == pytest.approx(72.0)


def test_etabs_and_sap_tables_of_one_frame_give_one_answer(tmp_path):
    """The same portal as ETABS tables and as SAP2000 tables analyses identically."""
    f = str(tmp_path / "frame.xlsx")
    _write_xlsx(_etabs_tables(), f)
    me = import_model(f).model
    sap = '''TABLE:  "PROGRAM CONTROL"
   ProgramName=SAP2000   Version=24.0.0   CurrUnits="KN, m, C"
TABLE:  "MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES"
   Material=C30   UnitWeight=0   UnitMass=0   E1=25000000   U12=0.2
TABLE:  "FRAME SECTION PROPERTIES 01 - GENERAL"
   SectionName=C40   Material=C30   Shape=Rectangular   t3=0.4   t2=0.4
   SectionName=B30   Material=C30   Shape=Rectangular   t3=0.6   t2=0.3   I3Mod=0.5
TABLE:  "JOINT COORDINATES"
   Joint=1   GlobalX=0   GlobalY=0   GlobalZ=0
   Joint=2   GlobalX=0   GlobalY=0   GlobalZ=3.5
   Joint=3   GlobalX=6   GlobalY=0   GlobalZ=3.5
   Joint=4   GlobalX=6   GlobalY=0   GlobalZ=0
TABLE:  "JOINT RESTRAINT ASSIGNMENTS"
   Joint=1   U1=Yes   U2=Yes   U3=Yes   R1=Yes   R2=Yes   R3=Yes
   Joint=4   U1=Yes   U2=Yes   U3=Yes   R1=Yes   R2=Yes   R3=Yes
TABLE:  "CONNECTIVITY - FRAME"
   Frame=C1   JointI=1   JointJ=2
   Frame=C2   JointI=4   JointJ=3
   Frame=B1   JointI=2   JointJ=3
TABLE:  "FRAME SECTION ASSIGNMENTS"
   Frame=C1   AnalSect=C40
   Frame=C2   AnalSect=C40
   Frame=B1   AnalSect=B30
TABLE:  "LOAD PATTERN DEFINITIONS"
   LoadPat=LAT   DesignType=Other   SelfWtMult=0
TABLE:  "LOAD CASE DEFINITIONS"
   Case=LAT   Type=LinStatic
TABLE:  "CASE - STATIC 1 - LOAD ASSIGNMENTS"
   Case=LAT   LoadType="Load pattern"   LoadName=LAT   LoadSF=1
TABLE:  "JOINT LOADS - FORCE"
   Joint=2   LoadPat=LAT   CoordSys=GLOBAL   F1=10
'''
    ms = CsiImporter.from_tables(read_text_tables(sap)).model
    ue = analyze(me).joint_displacement("2", "LAT")
    us = analyze(ms).joint_displacement("2", "LAT")
    assert ue[0] > 0 and np.allclose(ue, us, rtol=1e-9, atol=1e-15)


def test_etabs_v9_text_and_legacy_sap2000_v7():
    r9 = import_model(path("etabs_v9_column.e2k"))
    m = r9.model
    assert set(m.load_cases) == {"DEAD", "LATX"} and m.materials["CONC"].type == "Concrete"
    assert m.shell_sections["SLAB1"].thickness == pytest.approx(0.2)
    E, I, G, As = 25e6, 0.4 ** 4 / 12, 25e6 / 2.4, 5 / 6 * 0.16
    u = analyze(m).joint_displacement("1@STORY1", "LATX")[0]
    assert u == pytest.approx(10 * 3.5 ** 3 / (3 * E * I) + 10 * 3.5 / (G * As), rel=1e-9)
    r7 = import_model(path("sap2000_v7_cantilever.s2k"))
    m7 = r7.model
    assert r7.kind == "s2k_legacy" and m7.units.length == "in"
    res = analyze(m7)
    assert res.joint_displacement("3", "TIP")[2] == pytest.approx(-120 ** 3 / (3 * 29000 * 200), rel=1e-9)
    assert res.joint_displacement("3", "UDL")[2] == pytest.approx(-0.01 * 120 ** 4 / (8 * 29000 * 200), rel=1e-9)
    assert [(i.name, i.scale) for i in m7.load_combinations["C1"].items] == [("TIP", 2.0)]


def test_native_files_need_the_installed_program(tmp_path):
    f = tmp_path / "model.EDB"
    f.write_bytes(b"\x00\x01binary")
    assert detect(str(f)) == "native"
    if os.name != "nt":
        from structstudio.io.native import ConversionError
        with pytest.raises(ConversionError, match="Windows"):
            import_model(str(f))
    with pytest.raises(ValueError, match="native"):
        import_model(str(f), convert_native=False)


def test_unsupported_tables_are_reported():
    text = '''TABLE:  "PROGRAM CONTROL"
   ProgramName=SAP2000   Version=22.0.0   CurrUnits="KN, m, C"
TABLE:  "MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES"
   Material=S   UnitWeight=0   UnitMass=0   E1=200000000   U12=0.3
TABLE:  "JOINT COORDINATES"
   Joint=1   GlobalX=0   GlobalY=0   GlobalZ=0
   Joint=2   GlobalX=1   GlobalY=0   GlobalZ=0
TABLE:  "CONNECTIVITY - LINK"
   Link=1   JointI=1   JointJ=2
TABLE:  "LOAD CASE DEFINITIONS"
   Case=TH1   Type=LinModHist
'''
    imp = CsiImporter.from_tables(read_text_tables(text))
    rep = imp.report()
    assert "CONNECTIVITY - LINK" in rep and "time history" in rep


def test_corrupted_csi_numbers_are_reported(tmp_path):
    # SAFE 22 database tables drop trailing zeros of whole numbers (E = 3000 -> "3")
    from structstudio.io.csi_import import CsiImporter
    text = """TABLE:  "PROGRAM CONTROL"
   ProgramName=SAFE   Version=22.3.0   CurrUnits=kip_in_F   NumbersSuspect=Yes

TABLE:  "Material Properties - General"
   Material=C4   Type=Concrete   SymType=Isotropic

TABLE:  "Material Properties - Basic Mechanical Properties"
   Material=C4   UnitWeight=0   UnitMass=0   E1=3   G12=1153.8461538461538   U12=0.3   A1=5.5E-06

TABLE:  "Material Properties - Concrete Data"
   Material=C4   Fc=4

TABLE:  "Point Object Connectivity"
   UniqueName=1   X=0   Y=0   Z=0

END TABLE DATA
"""
    p = tmp_path / "bad.SAFE.tables.txt"
    p.write_text(text)
    im = CsiImporter.read(str(p))
    assert any("Implausible material data" in w for w in im.warnings)
    assert any("could not read reliable numbers" in w for w in im.warnings)


def _safe22_strip(null_load=True, stiff=False, spring="1E+10", springs_from=""):
    """A SAFE 22 style plate 360 x 240 on line springs along its edges, four floors
    named by UniqueName (the "Area" column is the measured area), a null line with
    a line load across the middle (SAFE example 1 layout, 1 kip/ft)."""
    pts = [(1, 0, 0), (2, 180, 0), (3, 360, 0), (4, 0, 120), (5, 180, 120), (6, 360, 120), (7, 0, 240), (8, 180, 240),
           (9, 360, 240)]
    rows = "\n".join(f"   UniqueName={n}   IsAuto=No   IsSpecial=No   X={x}   Y={y}   Z=0" for n, x, y in pts)
    floors = [(10, 1, 2, 5, 4), (11, 2, 3, 6, 5), (12, 4, 5, 8, 7), (13, 5, 6, 9, 8)]
    frows = "\n".join(f"   UniqueName={n}   UniquePt1={a}   UniquePt2={b}   UniquePt3={c}   UniquePt4={d}   Perimeter=600   "
                      f"Area=21600" for n, a, b, c, d in floors)
    prop = "Stiff" if stiff else "Slab"
    return f'''TABLE:  "PROGRAM CONTROL"
   ProgramName=SAFE   Version=22.3.0   CurrUnits=kip_in_F   TableRoute=DisplayArray{springs_from}
TABLE:  "Material Properties - Basic Mechanical Properties"
   Material=MAT1   UnitWeight=0   UnitMass=0   E1=3000   G12=1250   U12=0.2   A1=5.5E-06
TABLE:  "Slab Property Definitions"
   Name=SLAB   ModelType=Shell-Thick   PropType={prop}   Material=MAT1   Thickness=8
TABLE:  "Point Object Connectivity"
{rows}
TABLE:  "Floor Object Connectivity"
{frows}
TABLE:  "Area Assignments - Section Properties"
''' + "\n".join(f"   UniqueName={f[0]}   SectProp=SLAB   PropType=Slab" for f in floors) + f'''
TABLE:  "Null Line Object Connectivity"
   UniqueName=20   UniquePtI=1   UniquePtJ=3   Length=360
   UniqueName=21   UniquePtI=7   UniquePtJ=9   Length=360
   UniqueName=22   UniquePtI=1   UniquePtJ=7   Length=240
   UniqueName=23   UniquePtI=3   UniquePtJ=9   Length=240
   UniqueName=24   UniquePtI=4   UniquePtJ=6   Length=360
TABLE:  "Frame Assignments - Section Properties"
''' + "\n".join(f"   UniqueName={n}   Shape=N.A.   SectProp=None" for n in range(20, 25)) + f'''
TABLE:  "Spring Property Definitions - Line Springs"
   Name=EDGE   StiffOpt=User   VertStiff={spring}   RotStiff=0   NonlinOpt="None (Linear)"
TABLE:  "Frame Assignments - Line Springs"
''' + "\n".join(f"   UniqueName={n}   SpringProp=EDGE" for n in range(20, 24)) + '''
TABLE:  "Load Pattern Definitions"
   Name=LL   IsAuto=No   Type=Live   SelfWtMult=0
''' + ('''TABLE:  "Frame Loads Assignments - Distributed"
   UniqueName=24   LoadPattern=LL   LoadType=Force   Dir=Gravity   DistType=Relative   RelDistA=0   RelDistB=1   ForceA=0.0833333   ForceB=0.0833333
''' if null_load else "")


def test_safe22_floors_null_line_loads_and_spring_values():
    imp = CsiImporter.from_tables(read_text_tables(_safe22_strip()))
    m = imp.model
    # floors keep their UniqueName (the "Area" column is the area, 21600 in2)
    assert sorted(m.areas) == ["10", "11", "12", "13"] and all(a.section == "SLAB" for a in m.areas.values())
    # null lines: Null frames without stiffness; the middle one carries the line load
    assert m.frames["24"].type == "Null" and m.frames["24"].loads and len(m.line_springs) == 4
    r = analyze(m)
    assert r.base_reactions("LL")[2] == pytest.approx(0.0833333 * 360, rel=1e-9)
    # the load reaches the slab along the line (y = 120): every loaded node lies on it, the
    # total is the line load, and it is symmetric about the middle
    nl = r.am.nodal_loads["LL"]
    on_line = {nid: v for nid, v in nl.items() if abs(v[2]) > 0}
    assert all(abs(r.am.nodes[nid].xyz[1] - r.am.nodes[r.am.point_to_node["5"]].xyz[1]) < 1e-9 for nid in on_line)
    total = sum(v[2] for v in on_line.values())
    assert total == pytest.approx(-0.0833333 * 360 * m.units.factor("force"), rel=1e-9)
    end4, end6 = nl[r.am.point_to_node["4"]][2], nl[r.am.point_to_node["6"]][2]
    assert end4 == pytest.approx(end6) and len(on_line) >= 3
    assert r.joint_displacement("5", "LL")[2] < 0 and abs(r.joint_displacement("1", "LL")[2]) < 1e-9
    assert not any("Spring stiffness = 1" in w for w in imp.warnings)
    # a null line has no stiffness: the plate deflects the same with or without the unloaded line
    m2 = CsiImporter.from_tables(read_text_tables(_safe22_strip())).model
    del m2.frames["24"]
    m2.add_load_pattern("P", "Live")
    m.add_load_pattern("P", "Live")
    for mm in (m, m2):
        mm.add_point_load("5", "P", (0, 0, -10))
        mm.add_linear_static_case("P")
    d1, d2 = analyze(m).joint_displacement("5", "P")[2], analyze(m2).joint_displacement("5", "P")[2]
    assert d1 == pytest.approx(d2, rel=1e-12)


def test_safe22_lost_zeros_and_stiff_areas_are_reported():
    imp = CsiImporter.from_tables(read_text_tables(_safe22_strip(spring="1")))
    assert any("Spring stiffness = 1" in w and "EDGE.VertStiff" in w for w in imp.warnings)
    imp = CsiImporter.from_tables(read_text_tables(_safe22_strip(spring="1", springs_from="   SpringsFrom=ObjectAPI")))
    assert not any("Spring stiffness = 1" in w for w in imp.warnings)  # the converter read them from the API
    imp = CsiImporter.from_tables(read_text_tables(_safe22_strip(stiff=True)))
    sec = imp.model.shell_sections["SLAB"]
    assert sec.modifiers.m11 == pytest.approx(100.0) and sec.modifiers.f12 == pytest.approx(100.0)
    assert sec.modifiers.v13 == pytest.approx(1.0) and any("Stiff" in w for w in imp.warnings)
