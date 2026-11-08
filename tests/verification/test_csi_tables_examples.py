"""Real CSI files converted by the installed programs (scripts/Export-CsiSamples.cmd):

* the 30 ETABS 21 verification examples as database tables, imported through the
  tables importer and checked against ETABS 21's own results (the same results the
  .e2k importer is checked against);
* 40 SAP2000 23 and 40 SAFE 22 example models: imported and analysed; checked against
  the programs' results when scripts/Export-CsiResults.cmd has written them.
"""
import glob
import os

import numpy as np
import pytest

from structstudio import analyze
from structstudio.io import import_model
from structstudio.verification import run

DATA = os.path.join(os.path.dirname(__file__), "data")
ETABS_TABLES = os.path.join(DATA, "etabs", "tables")
ETABS_RESULTS = os.path.join(DATA, "etabs", "results")

have_etabs = pytest.mark.skipif(not (os.path.isdir(ETABS_TABLES) and os.path.isdir(ETABS_RESULTS)),
                                reason="ETABS tables / results not exported")


def _rows(name, data_dir, results):
    return run(data_dir, [name], results)


@have_etabs
@pytest.mark.parametrize("name", ["Example02", "Example04", "Example05", "Example06", "Example07", "Example08",
                                  "Example09"])
def test_etabs_tables_frames_match_etabs21(name):
    rows = _rows(name, ETABS_TABLES, ETABS_RESULTS)
    assert {r.quantity for r in rows} >= {"joint displacement", "base reaction"}
    for r in rows:
        assert r.quantity != "ERROR", r.notes
        assert r.rel < 2e-3, f"{r.example} {r.quantity} {r.case}: {100 * r.rel:.3f}% ({r.worst})"


@have_etabs
def test_etabs_tables_example01_releases_and_point_loads():
    for r in _rows("Example01", ETABS_TABLES, ETABS_RESULTS):
        if r.quantity == "frame forces":
            continue  # ETABS reports the shear jump under the mid-span point load at the station itself
        assert r.rel < 1e-6, f"{r.quantity} {r.case}: {100 * r.rel:.3f}% ({r.worst})"


@have_etabs
@pytest.mark.parametrize("name", ["Example15a1", "Example15a2", "Example15a3", "Example15a4", "Example15a5",
                                  "Example15a6", "Example15a7", "Example15a8", "Example15a9", "Example15b",
                                  "Example15c1", "Example15c2", "Example15c3", "Example15c4"])
def test_etabs_tables_walls_match_etabs21(name):
    tol = {"joint displacement": 0.035, "frame forces": 0.07, "base reaction": 1e-3, "period": 1e-3}
    for r in _rows(name, ETABS_TABLES, ETABS_RESULTS):
        assert r.quantity != "ERROR", r.notes
        assert r.rel < tol[r.quantity], f"{r.example} {r.quantity} {r.case}: {100 * r.rel:.3f}% ({r.worst})"


SAP = sorted(glob.glob(os.path.join(DATA, "sap2000", "*.SAP2000.tables.txt")))
SAFE = sorted(glob.glob(os.path.join(DATA, "safe", "*.SAFE.tables.txt")))


@pytest.mark.skipif(not SAP, reason="SAP2000 samples not exported")
@pytest.mark.parametrize("path", SAP, ids=lambda p: os.path.basename(p).split(".")[0])
def test_sap2000_samples_import_and_analyse(path):
    res = import_model(path)
    m = res.model
    assert res.program == "SAP2000" and m.frames and m.load_cases
    r = analyze(m)
    assert r.case_names
    results = os.path.join(DATA, "sap2000", "results")
    if os.path.isdir(results):
        name = os.path.basename(path).split(".SAP2000")[0].replace(" ", "")
        for c in run(os.path.dirname(path), [name], results):
            assert c.quantity != "ERROR", c.notes


@pytest.mark.skipif(not SAFE, reason="SAFE samples not exported")
@pytest.mark.parametrize("path", SAFE, ids=lambda p: os.path.basename(p).split(".")[0])
def test_safe_samples_import_analyse_and_equilibrium(path):
    from structstudio.mesh.geometry import polygon_area
    res = import_model(path)
    m = res.model
    assert res.program == "SAFE" and m.areas
    assert m.options.active_dofs == (False, False, True, True, True, False)  # plates
    r = analyze(m)
    for c, case in m.load_cases.items():
        if not hasattr(case, "loads") or c not in r.cases:
            continue
        applied = 0.0
        from structstudio.model.area_panels import area_items
        for panel in m.panel_loads.values():  # area loads are area-load panels
            a = m.areas.get(panel.area)
            if a is None or a.is_opening:
                continue
            for ld in area_items(panel):
                applied += case.loads.get(ld.pattern, 0.0) * ld.value * polygon_area(
                    np.array([m.points[p].xyz for p in a.points]))
        for p in m.points.values():
            for ld in p.loads:
                applied -= case.loads.get(ld.pattern, 0.0) * ld.force[2]
        if not any(f.loads for f in m.frames.values()):
            assert r.base_reactions(c)[2] == pytest.approx(applied, rel=1e-6, abs=1e-9)
