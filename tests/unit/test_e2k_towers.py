"""ETABS 21 multiple towers in .e2k files (data: tests/data/import/etabs21_towers, written by
ETABS 21.1 through scripts/Make-TowerSample.ps1, with ETABS' own result tables):

* "$ TOWERS", stories "IN SEQUENCE FROM TOP - FOR EACH TOWER" (story names repeat per tower;
  the first line of each tower names it), grid systems per tower ("T1-G1"), assignments to
  "T1-Story1" - read as towers, our stories named as ETABS names them in its tables;
* a third number on a POINT line = how far the point lies below the story it is used at.
The model is then run and compared with ETABS' own results."""
import csv
import os

import numpy as np
import pytest

from structstudio import analyze
from structstudio.engine.kernel import SciPyKernel
from structstudio.io.e2k import E2KImporter

D = os.path.join(os.path.dirname(__file__), "..", "data", "import", "etabs21_towers")


def _rows(name):
    with open(os.path.join(D, f"TwoTowers.{name}.csv")) as fh:
        return list(csv.DictReader(fh))


def test_towers_stories_and_grids_are_read():
    imp = E2KImporter.read(os.path.join(D, "TwoTowers.e2k"))
    m = imp.model
    assert list(m.towers) == ["T1", "T2"]
    assert [(s.name, s.elevation) for s in m.stories_of("T1")] == [
        ("T1-Base", 0.0), ("T1-Story1", 4000.0), ("T1-Story2", 7000.0), ("T1-Story3", 10000.0)]
    assert [s.elevation for s in m.stories_of("T2")] == [0.0, 3000.0, 6000.0, 9000.0, 12000.0]
    assert m.grid_systems_of("T1") == ["T1-G1"] and m.grid_systems_of("T2") == ["T2-G1"]
    assert len([g for g in m.grid_lines if g.system == "T1-G1"]) == 8
    # points 11, 13, 15, 17 lie 500 mm below the story they are used at (floors at 3.5 m)
    zs = sorted({round(p.z) for p in m.points.values() if p.x >= 30000})
    assert zs == [0, 3500, 7000]
    assert "T1-Story1" in {f.story for f in m.frames.values()}


def test_two_towers_with_objects_in_each():
    m = E2KImporter.read(os.path.join(D, "TwoTowersT2.e2k")).model
    assert [s.name for s in m.stories_of("T2")] == ["T2-Base", "T2-Story1", "T2-Story2"]
    t2 = {n for n in m.frames if m.tower_of_object("frame", n) == "T2"}
    assert len(t2) == 16 and all(m.frames[n].story.startswith("T2-") for n in t2)
    assert {m.tower_of_object("area", n) for n in m.areas if m.areas[n].diaphragm == "D2"} == {"T2"}


@pytest.fixture(scope="module")
def etabs_run():
    m = E2KImporter.read(os.path.join(D, "TwoTowers.e2k")).model.convert_units("kN m")
    return m, analyze(m, kernel=SciPyKernel())


def test_lateral_case_matches_etabs(etabs_run):
    m, res = etabs_run
    env = res.displacement_envelope("EQX").max
    n = 0
    for r in _rows("Joint_Displacements"):
        if r["OutputCase"] != "EQX":
            continue
        pt = f"{r['Label']}@{r['Story']}"
        if pt not in m.points:
            continue
        u = env[res._node_of_point(pt)]
        for k, lab in enumerate(("Ux", "Uy")):
            e = float(r[lab])
            if abs(e) > 2e-5:
                assert u[k] == pytest.approx(e, rel=6e-3), (pt, lab)
                n += 1
    assert n > 40
    br = next(r for r in _rows("Base_Reactions") if r["OutputCase"] == "EQX")
    assert res.base_reactions("EQX")[:2] == pytest.approx([float(br["FX"]), float(br["FY"])], rel=1e-9)
    drift = {(r["Story"], r["Direction"]): float(r["Drift"]) for r in _rows("Story_Drifts") if r["OutputCase"] == "EQX"}
    for d in res.story_drifts("EQX"):
        if (d["Story"], d["Direction"]) in drift:  # (ETABS lists 3 digits)
            assert d["Drift"] == pytest.approx(drift[(d["Story"], d["Direction"])], rel=6e-3, abs=1.5e-6)


def test_modes_of_a_floor_between_the_levels_lumped_to_them(etabs_run):
    """The 3.5 m floor of D2 is between story levels: the mass source lumps its mass to
    the levels - its diaphragm's inertia correction goes with it (no negative mass)."""
    m, res = etabs_run
    pe = [float(r["Period"]) for r in _rows("Modal_Periods_And_Frequencies")]
    po = [d["Period"] for d in res.modal_periods("Modal")]
    assert po[0] == pytest.approx(pe[0], rel=0.03) and po[1] == pytest.approx(pe[1], rel=0.03)


def test_csi_tables_with_towers():
    """The CSI tables of the same ETABS model (.edb import): stories per tower from the
    "Story Definitions" (Tower column) and "Tower and Base Story Definitions" tables,
    grid systems per tower, the joints in the tower of their "Point Object Connectivity"
    row."""
    from structstudio.io.csi_import import CsiImporter
    imp = CsiImporter.read(os.path.join(D, "TwoTowers.tables_with_stories.txt"))
    m = imp.model
    assert list(m.towers) == ["T1", "T2"]
    assert [s.name for s in m.stories_of("T1")] == ["T1-Base", "T1-Story1", "T1-Story2", "T1-Story3"]
    assert [s.elevation for s in m.stories_of("T2")] == [0.0, 3000.0, 6000.0, 9000.0, 12000.0]
    assert {n: m.tower_of_system(n) for n in m.grid_systems} == {"T1-G1": "T1", "T2-G1": "T2"}
    # every joint of this model is in T1 (as ETABS put them): none on a T2 story
    assert all(p.story is None or p.story.startswith("T1-") for p in m.points.values())
    assert m.stories["T1-Story2"].similar_to == "T1-Story3"


def test_csi_tables_without_story_table_take_levels_per_tower():
    """An export without the story table (ETABS 21 marks it not importable with towers):
    each tower's levels from its joints."""
    from structstudio.io.csi_import import CsiImporter
    text = open(os.path.join(D, "TwoTowers.tables_with_stories.txt")).read()
    i, j = text.index('TABLE:  "Story Definitions"'), text.index('TABLE:  "Grid Definitions - General"')
    text = text[:i] + text[j:]
    # the joints at x >= 30 m in tower T2
    out = []
    for line in text.splitlines():
        if "Tower=T1" in line and "X=30000" in line or "Tower=T1" in line and "X=36000" in line:
            line = line.replace("Tower=T1", "Tower=T2")
        out.append(line)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.tables.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(out))
        m = CsiImporter.read(path).model
    assert m.multi_tower
    assert [s.elevation for s in m.stories_of("T2")] == [0.0, 3500.0, 7000.0]
    assert [s.elevation for s in m.stories_of("T1")] == [0.0, 4000.0, 7000.0, 10000.0]
    assert {p.story for p in m.points.values() if p.x >= 30000} == {"T2-Base", "T2-Story1", "T2-Story2"}
    assert m.grid_systems_of("T2") == ["T2-G1"]
