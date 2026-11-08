"""CSI ETABS verification suite (Analysis Examples 2, 5, 6, 7, 8, 9, 15) run
from the ETABS text model files exported by scripts/Export-VerificationModels.cmd.

Reference values are taken from the ETABS Software Verification manual
("ETABS" column, with the independent references where they differ).  The
examples are skipped when the exported files are not present.

Section database: the examples use AISC sections from the CSI property
libraries; point STRUCTSTUDIO_SECTION_PATH at the "Property Libraries" folder
of an ETABS installation (the default search paths cover the usual locations).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from structstudio import analyze
from structstudio.io import E2KImporter

DATA = Path(__file__).parent / "data" / "etabs"


def _model(name: str):
    path = DATA / f"Example{name}.e2k"
    if not path.exists():
        pytest.skip(f"{path.name} not exported (run scripts/Export-VerificationModels.cmd)")
    try:
        return E2KImporter.read(str(path)).model
    except Exception as exc:  # missing section library etc.
        pytest.skip(f"cannot import {path.name}: {exc}")


def _periods(res, n, case="Modal"):
    return np.array([row["Period"] for row in res.modal_periods(case)][:n])


# --------------------------------------------------------------- Example 2
def test_example02_three_story_plane_frame_response_spectrum():
    m = _model("02")
    res = analyze(m)
    np.testing.assert_allclose(_periods(res, 3), [0.4414, 0.1575, 0.1090], rtol=5e-4)
    # response-spectrum displacements (Y, frame spans Y) and base moment of C1
    for point, expected in (("C1@ROOF", 2.139), ("C1@2ND", 1.716), ("C1@1ST", 0.955)):
        assert res.joint_displacement(point, "SPEC2")[1] == pytest.approx(expected, rel=1e-3)
    assert res.frame_end_forces("C1@1ST", "SPEC2")[0][5] == pytest.approx(11730, rel=1e-3)


# --------------------------------------------------------------- Example 5
def test_example05_three_story_3d_braced_frame():
    m = _model("05")
    res = analyze(m)
    # ETABS 0.32686 / 0.32061; Wilson & Habibullah and Peterson 0.32689 / 0.32064
    np.testing.assert_allclose(_periods(res, 2), [0.32689, 0.32064], rtol=2e-4)
    for frame, expected in (("C1@STORY1", 279.47), ("D1@STORY1", 194.51), ("D3@STORY1", 120.53)):
        assert abs(res.frame_end_forces(frame, "RS-CQC")[0][0]) == pytest.approx(expected, rel=5e-4)


# --------------------------------------------------------------- Example 6
def test_example06_nine_story_ten_bay_frame_eigenvalues():
    m = _model("06")
    res = analyze(m)
    eig = np.array([row["Eigenvalue"] for row in res.modal_periods("Modal")][:3])
    # ETABS 0.58964 / 5.53195 / 16.5962; Bathe & Wilson 0.58954 / 5.52695 / 16.5878
    np.testing.assert_allclose(eig, [0.58964, 5.53195, 16.5962], rtol=1e-3)


# --------------------------------------------------------------- Example 7
def test_example07_seven_story_frame_static_and_modal():
    m = _model("07")
    if "FUNC3" not in m.functions:
        # the El Centro spectrum file referenced by the model is not shipped
        for case in ("SPECSRSS", "SPECCQC"):
            m.load_cases.pop(case, None)
    res = analyze(m)
    assert res.joint_displacement("1@STORY7", "LAT")[0] == pytest.approx(1.4508, rel=2e-4)
    end = res.frame_end_forces("C2@STORY1", "LAT")[0]  # manual "C1" = left-hand column
    assert abs(end[0]) == pytest.approx(69.99, rel=2e-4)
    assert abs(end[5]) == pytest.approx(2324.68, rel=2e-4)
    np.testing.assert_allclose(
        _periods(res, 7), [1.27321, 0.43128, 0.24205, 0.16018, 0.11899, 0.09506, 0.07952], rtol=2e-4)


# --------------------------------------------------------------- Example 8
def test_example08_two_story_3d_frame_response_spectrum():
    m = _model("08")
    res = analyze(m)
    # ETABS 0.22708 / 0.21565 / 0.07335 / 0.07201; references 0.22706 / 0.21563 / ...
    np.testing.assert_allclose(_periods(res, 4), [0.22706, 0.21563, 0.07335, 0.07201], rtol=2e-4)
    cm = res.diaphragm_cm_displacement("D1", "STORY2", "RSCQC")
    assert cm[0] == pytest.approx(0.0201, rel=3e-3)


# --------------------------------------------------------------- Example 9
def test_example09_two_story_unsymmetric_building_angled_spectrum():
    m = _model("09")
    res = analyze(m)
    np.testing.assert_allclose(_periods(res, 6), [0.4146, 0.3753, 0.2436, 0.1148, 0.1103, 0.0729], rtol=5e-4)
    assert res.diaphragm_cm_displacement("D1", "2ND", "SPEC1")[0] == pytest.approx(0.1062, rel=1e-3)
    assert res.diaphragm_cm_displacement("D1", "2ND", "SPEC2")[0] == pytest.approx(0.0617, rel=1e-3)


# -------------------------------------------------------------- Example 15
def _loaded_point_displacement(m, res):
    point, pattern = next((p.name, ld.pattern) for p in m.points.values() for ld in p.loads)
    return res.joint_displacement(point, pattern)[0]


# (example, ETABS result at the loaded point, direction).  ETABS 21 values (identical to
# the "ETABS" column of the verification manual); ETABS models each wall panel with one
# shell element, as the importer does.
WALLS = [
    ("15a1", 2.3921, 0),   # 6-storey planar wall, 120" long
    ("15a4", 0.3071, 0),   # 3-storey, 120"
    ("15a7", 0.0145, 0),   # 1-storey, 120"
    ("15a9", 0.0011, 0),   # 1-storey, 720"
    ("15b", 0.0691, 0),    # wall supported on columns
    ("15c1", 0.0844, 0),   # wall-spandrel, 60" beams, 6 storeys
    ("15c2", 0.1456, 0),   # 240" beams
    ("15c3", 0.0188, 0),   # 60" beams, 3 storeys
    ("15c4", 0.0313, 0),
    ("15d1", 0.8637, 0),   # C-shaped wall, X load
    ("15e1", 0.2822, 0),   # wall with thickened edges
    ("15e2", 0.0464, 0),
    ("15f1", 0.3707, 0),   # E-shaped wall, X load
    ("15f2", 0.0602, 0),
]


@pytest.mark.parametrize("name,etabs,k", WALLS)
def test_example15_wall_objects_match_etabs(name, etabs, k):
    m = _model(name)
    res = analyze(m)
    point, pattern = next((p.name, ld.pattern) for p in m.points.values() for ld in p.loads)
    d = res.joint_displacement(point, pattern)[k]
    assert abs(d) == pytest.approx(etabs, rel=0.05, abs=5e-5)


ETABS_RESULTS = DATA / "results"


def _etabs_rows(name):
    if not (ETABS_RESULTS / f"Example{name}.joint_displacements.csv").exists():
        pytest.skip("ETABS results not exported (scripts/Export-VerificationResults.cmd)")
    from structstudio.verification import compare_example
    return compare_example(f"Example{name}", str(DATA / f"Example{name}.e2k"), str(ETABS_RESULTS))


@pytest.mark.parametrize("name", ["02", "04", "05", "06", "07", "08", "09"])
def test_frames_match_etabs21_results(name):
    """Every joint displacement, period, frame force station and base reaction of
    every case vs ETABS 21 (scripts/Export-VerificationResults.cmd)."""
    rows = _etabs_rows(name)
    assert {r.quantity for r in rows} >= {"joint displacement", "frame forces", "base reaction"}
    for r in rows:
        assert r.quantity != "ERROR", r.notes
        # 0.2 %: Ex.5 pinned braces carry ~5e-4 kip-in numerical moments where ETABS prints 0
        assert r.rel < 2e-3, f"{r.example} {r.quantity} {r.case}: {100 * r.rel:.3f}% ({r.worst})"


WALL_EXAMPLES = ["15a1", "15a2", "15a3", "15a4", "15a5", "15a6", "15a7", "15a8", "15a9", "15b",
                 "15c1", "15c2", "15c3", "15c4", "15d1", "15d2", "15e1", "15e2", "15f1", "15f2"]


@pytest.mark.parametrize("name", WALL_EXAMPLES)
def test_walls_match_etabs21_results(name):
    """Walls (one shell element per panel, like ETABS): every joint within 3.5 %,
    frame forces of frames framing into walls within 7 %, base reactions exact."""
    rows = _etabs_rows(name)
    tol = {"joint displacement": 0.035, "frame forces": 0.07, "base reaction": 1e-3, "period": 1e-3}
    for r in rows:
        assert r.quantity != "ERROR", r.notes
        assert r.rel < tol[r.quantity], f"{r.example} {r.quantity} {r.case}: {100 * r.rel:.3f}% ({r.worst})"
