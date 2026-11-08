"""Acceptance tests against the CSI ETABS verification manual.

Each test rebuilds a verification example from its published description
(and, where available, the ETABS text model file) and compares our results
with the *theoretical* values printed in the manual.  Tolerances are stated
per example.
"""
import os

import numpy as np
import pytest

from structstudio import Model, analyze
from structstudio.io import E2KImporter
from structstudio.model import FIXED, FrameSection, ShellSection

DATA = os.path.join(os.path.dirname(__file__), "data")


# ----------------------------------------------------------------------------
# Example 1 - plane frame with beam span loads (static gravity)
# ----------------------------------------------------------------------------
EX1_CASE_I = {  # station -> (V2, M3) for beam B1
    0.00: (-31.25, 0.0), 0.25: (-31.25, 1687.5), 0.50: (68.75, 3375.0), 0.75: (68.75, -337.5), 1.00: (68.75, -4050.0),
}
EX1_CASE_II = {
    0.00: (-67.5, 0.0), 0.25: (-22.5, 2430.0), 0.50: (22.5, 2430.0), 0.75: (67.5, 0.0), 1.00: (112.5, -4860.0),
}


def _check_beam_table(res, frame, case, expected, length):
    st = res.frame_forces(frame, case)
    for rel, (v2, m3) in expected.items():
        k = int(np.argmin(np.abs(st.station - rel * length)))
        assert st.forces[k, 1] == pytest.approx(v2, abs=0.01)
        assert st.forces[k, 5] == pytest.approx(m3, abs=0.05)


def test_example01_from_e2k():
    imp = E2KImporter.read(os.path.join(DATA, "Example01.e2k"))
    res = analyze(imp.model, cases=["I", "II"])
    _check_beam_table(res, "B1@1ST", "I", EX1_CASE_I, 216.0)
    _check_beam_table(res, "B1@1ST", "II", EX1_CASE_II, 216.0)


def build_example01() -> Model:
    m = Model("EX1", "kip in")
    m.use_csi_frame_defaults()  # as built in SAP2000 / ETABS
    m.options.active_dofs = (True, False, True, False, True, False)
    m.add_material("MP1", E=3000, nu=0.0)
    m.add_frame_section(FrameSection.general("CP1", "MP1", A=1e7, I33=13824))
    m.add_frame_section(FrameSection.general("BP1", "MP1", A=1e7, I33=27000))
    m.set_stories([("1ST", 120)], base_name="BASE")
    base = [m.add_point(x, 0, 0, restraints=FIXED) for x in (0, 216, 432)]
    top = [m.add_point(x, 0, 120) for x in (0, 216, 432)]
    pin = (False, False, False, False, False, True)
    m.add_frame(base[0], top[0], "CP1", name="C1", releases_j=pin)
    m.add_frame(base[1], top[1], "CP1", name="C2", releases_i=pin)
    m.add_frame(base[2], top[2], "CP1", name="C3", releases_j=pin)
    m.add_frame(top[0], top[1], "BP1", name="B1", max_station_spacing=54)
    m.add_frame(top[1], top[2], "BP1", name="B2", max_station_spacing=54)
    m.add_load_pattern("I", "Dead")
    m.add_load_pattern("II", "Live")
    for p, f in zip(top, (50, 100, 50)):
        m.add_point_load(p, "I", force=(0, 0, -f))
    for b in ("B1", "B2"):
        m.add_frame_point_load(b, "I", 100, 0.5)
        m.add_frame_distributed_load(b, "II", 10 / 12)
    m.add_linear_static_case("I")
    m.add_linear_static_case("II")
    return m


def test_example01_api_model():
    res = analyze(build_example01())
    _check_beam_table(res, "B1", "I", EX1_CASE_I, 216.0)
    _check_beam_table(res, "B1", "II", EX1_CASE_II, 216.0)


def test_example01_json_roundtrip():
    m = build_example01()
    m2 = Model.from_json(m.to_json())
    res = analyze(m2)
    _check_beam_table(res, "B1", "II", EX1_CASE_II, 216.0)


def test_example01_combination():
    m = build_example01()
    m.add_combination("COMB", {"I": 1.2, "II": 1.6})
    res = analyze(m)
    st = res.frame_forces("B1", "COMB")
    k = int(np.argmin(np.abs(st.station - 216.0)))
    assert st.forces[k, 5] == pytest.approx(1.2 * -4050 + 1.6 * -4860, rel=1e-6)


# ----------------------------------------------------------------------------
# Example 2 - three-story plane frame, periods and mass-normalised mode shapes
# ----------------------------------------------------------------------------
def build_example02() -> Model:
    m = Model("EX2", "kip in")
    m.use_csi_frame_defaults()  # as built in SAP2000 / ETABS
    m.options.active_dofs = (True, False, True, False, True, False)
    m.add_material("STL", E=29500, nu=0.3, type="Steel")
    # W14X90: I33 = 999 in^4; infinite axial area, zero shear area (neglect shear deformation)
    m.add_frame_section(FrameSection.general("W14X90", "STL", A=1e7, I33=999, I22=362, J=4.06))
    m.add_frame_section(FrameSection.general("RIGID", "STL", A=1e7, I33=1e9, I22=1e9, J=1e9))
    m.set_stories([("1ST", 144), ("2ND", 144), ("ROOF", 144)], base_name="BASE")
    xs = (0, 432)
    prev = [m.add_point(x, 0, 0, restraints=FIXED) for x in xs]
    for sname, z in (("1ST", 144), ("2ND", 288), ("ROOF", 432)):
        top = [m.add_point(x, 0, z, mass=(0.2, 0.2, 0.0), name=f"{'C1' if x == 0 else 'C2'}@{sname}") for x in xs]
        for i in range(2):
            m.add_frame(prev[i], top[i], "W14X90", name=f"C{i + 1}@{sname}", end_offset_j=24, rigid_factor=1.0)
        m.add_frame(top[0], top[1], "RIGID", name=f"B1@{sname}")
        prev = top
    m.add_modal_case("Modal", max_modes=3)
    return m


def test_example02_periods_and_shapes():
    res = analyze(build_example02())
    periods = [r["Period"] for r in res.modal_periods("Modal")]
    assert periods == pytest.approx([0.4414, 0.1575, 0.1090], abs=5e-5)
    expected = {1: (1.165, 0.934, 0.519), 2: (0.934, -0.519, -1.165), 3: (0.519, -1.165, 0.934)}
    for mode, shape in expected.items():
        phi = res.mode_shape("Modal", mode)
        vals = np.array([phi[res.am.point_to_node[f"C1@{s}"], 0] for s in ("ROOF", "2ND", "1ST")])
        sign = np.sign(vals[0]) * np.sign(shape[0])
        assert sign * vals == pytest.approx(shape, abs=1e-3)


# ----------------------------------------------------------------------------
# Example 4 - single-story 3D frame with automatic story mass (rigid diaphragm)
# ----------------------------------------------------------------------------
def build_example04() -> Model:
    m = Model("EX4", "kip in")
    m.use_csi_frame_defaults()  # as built in SAP2000 / ETABS
    m.options.gravity = 386.4  # CSI value for kip-in
    m.add_material("CONC", E=3000, nu=0.2)
    m.add_material("MASS", E=1.0, nu=0.0, weight_density=150 / 144 / 1000)  # 150 psf on a 1 in slab
    m.add_frame_section(FrameSection.general("C24", "CONC", A=1e7, I33=24 ** 4 / 12, I22=24 ** 4 / 12, J=1e3))
    m.add_frame_section(FrameSection.general("C18", "CONC", A=1e7, I33=18 ** 4 / 12, I22=18 ** 4 / 12, J=1e3))
    m.add_frame_section(FrameSection.general("RIGID", "CONC", A=1e7, I33=1e9, I22=1e9, J=1e9))
    m.add_shell_section(ShellSection("FLOOR", "MASS", type="Slab", model_type="Membrane", thickness=1.0))
    m.set_stories([("STORY1", 180)], base_name="BASE")
    m.add_diaphragm("D1")
    cols = {"C1": ((0, 0), "C24"), "C2": ((360, 0), "C24"), "C3": ((0, 360), "C18"), "C4": ((360, 360), "C18")}
    top = {}
    for n, ((x, y), sec) in cols.items():
        b = m.add_point(x, y, 0, restraints=FIXED)
        top[n] = m.add_point(x, y, 180, name=n)
        m.add_frame(b, top[n], sec, name=n, end_offset_j=36, rigid_factor=1.0)
    for a, b in (("C1", "C2"), ("C2", "C4"), ("C4", "C3"), ("C3", "C1")):
        m.add_frame(top[a], top[b], "RIGID")
    m.options.floor_mesh_max_size = 90
    m.add_area(["C1", "C2", "C4", "C3"], "FLOOR", diaphragm="D1", name="F1")
    m.add_modal_case("Modal", max_modes=3)
    return m


def test_example04_periods_and_shapes():
    res = analyze(build_example04())
    periods = [r["Period"] for r in res.modal_periods("Modal")]
    assert periods == pytest.approx([0.1389, 0.1254, 0.0702], abs=6e-5)
    master = res.am.constraints[0].master
    expected = {1: (-1.6244, 0.0, 0.0032), 2: (0.0, 1.6918, 0.0), 3: (0.4728, 0.0, 0.0111)}
    for mode, (ux, uy, rz) in expected.items():
        phi = res.mode_shape("Modal", mode)[master][[0, 1, 5]]
        ref = np.array([ux, uy, rz])
        k = int(np.argmax(np.abs(ref)))
        sign = np.sign(phi[k]) * np.sign(ref[k])
        assert sign * phi == pytest.approx(ref, abs=1.5e-3)
