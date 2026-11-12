"""Changing units (Options > Display Units, the units button): every value of the
model is converted, so the same structure gives the same physical answer in any
unit set - forces, lengths and temperatures (C / F).

Three checks:

* completeness: every number stored in a model object is either tagged with
  its physical quantity, converted by the object's own ``convert_units``, or
  listed below as dimensionless (a new field that is none of these fails);
* exactness: converting to every unit set and back gives the original numbers;
* physics: a model using every kind of load, support, spring, offset, section
  and analysis gives the same displacements, reactions, member forces, periods
  and spectrum response (in SI) in every unit set.
"""
import dataclasses
import importlib
import inspect
import math
import pkgutil

import numpy as np
import pytest

import structstudio.model as model_pkg
from structstudio.api import analyze
from structstudio.mesh import MeshOptions
from structstudio.model import Model
from structstudio.model.objects import GroundDisplacement, LineSpring, Tendon, TendonLoad, TendonPoint
from structstudio.model.properties import FrameSection
from structstudio.units import PRESETS, UnitSystem

# numbers that have no units (ratios, factors, angles, counts, coefficients of
# dimensionless quantities, times, frequencies, damping) or are converted by hand
DIMENSIONLESS = {
    "AnalysisOptions": {"floor_mesh_max_size", "wall_mesh_max_size", "merge_tolerance", "gravity",  # Model.convert_units
                        "rigid_zone_factor_concrete", "rigid_zone_factor_steel", "rigid_zone_factor_other",
                        "mesh_refine_tolerance", "active_dofs", "fine_mesh_factor",
                        "fine_mesh_size_min", "fine_mesh_size_max"},  # (sizes: Model.convert_units)
    "Area": {"angle", "mesh_divisions", "edge_releases"},
    "ComboItem": {"scale"},
    "Frame": {"angle", "rigid_factor", "auto_end_offsets", "display_cardinal",
              "real_axis"},  # (joint names and flags only; its lengths are real_axis_offset_i / _j)
    "FrameHingeAssignment": {"relative_distance"},
    "FrameModifiers": {"A", "As2", "As3", "J", "I22", "I33", "mass", "weight"},
    "GridSystem": {"rotation"},
    "LinearBucklingCase": {"loads"},
    "LinearStaticCase": {"loads", "acceleration_loads", "cd", "ie"},  # pattern factors; accelerations in g
    "LinkObject": {"angle"},
    "LoadPattern": {"self_weight_multiplier", "auto_load"},
    "MassSource": {"load_patterns"},
    "Material": {"nu", "damping"},
    "ModalCase": {"shift", "cutoff", "tolerance"},  # cycles / s
    "NonlinearStaticCase": {"loads", "acceleration_loads", "mode_loads", "tolerance"},
    "PDeltaCase": {"loads", "tolerance"},
    "PanelLoad": {"points", "span_dir"},  # points: Model.convert_units; span_dir: a direction
    "Point": {"local_axes"},  # angles
    "ResponseSpectrumCase": {"directional_abs_factor", "damping", "angle", "eccentricity_ratio",
                             "rigid_response", "cd", "ie"},
    "SlabColumnDriftPrefs": {"phi", "lam"},
    "ShellModifiers": {"f11", "f22", "f12", "m11", "m22", "m12", "v13", "v23", "mass", "weight"},
    "TendonLoad": {"curvature"},  # per radian
    "TimeHistoryCase": {"dt", "damping_ratio", "damping_periods", "mass_coefficient", "stiffness_coefficient",
                        "gamma", "beta", "alpha", "tolerance"},
    "TimeHistoryFunction": {"times", "values"},  # values x the load's scale (an acceleration)
    "ResponseSpectrumFunction": {"periods", "accelerations", "damping"},  # x the load's scale (an acceleration)
    "GroundDisplacement": {"rotations"},
    "FrameSection": {"rebar"},  # design data, not used by the analysis yet
}
NUMERIC = ("float", "int", "Tuple", "List", "Dict", "list", "dict", "tuple", "Vec3")


def _model_dataclasses():
    seen = {}
    for mi in pkgutil.iter_modules(model_pkg.__path__):
        mod = importlib.import_module("structstudio.model." + mi.name)
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if dataclasses.is_dataclass(cls) and cls.__module__.startswith("structstudio.model") \
                    and cls.__module__ != "structstudio.model.section_shapes":  # computed, never stored
                seen[name] = cls
    return seen


def test_every_number_in_the_model_has_a_unit_rule():
    missing = []
    for name, cls in _model_dataclasses().items():
        hooked = callable(getattr(cls, "convert_units", None))
        for f in dataclasses.fields(cls):
            t = str(f.type)
            if f.metadata.get("quantity") or not any(k in t for k in NUMERIC) or t in ("int", "Optional[int]"):
                continue  # tagged, not a number, or a count
            if "str" in t and not any(k in t for k in ("float", "Dict[str, float]")):
                continue  # names, lists of names
            if t in ("bool", "Optional[bool]") or "bool]" in t and "float" not in t:
                continue
            if f.name in DIMENSIONLESS.get(name, set()) or hooked:
                continue
            if dataclasses.is_dataclass(f.default_factory() if f.default_factory is not dataclasses.MISSING
                                        else f.default):
                continue
            if any(k in t for k in ("List[PointLoad]", "List[FrameLoad]", "List[AreaLoad]", "List[TendonPoint]",
                                    "List[TendonLoad]", "List[PanelLoadCase]", "List[GroundDisplacement]",
                                    "List[SpectrumLoad]", "List[ComboItem]", "List[TimeHistoryLoad]",
                                    "List[LinkDOF]", "Modifiers", "list")) and f.name not in ("points",):
                continue  # nested objects: converted on their own
            missing.append(f"{name}.{f.name}: {t}")
    assert not missing, "fields without a unit rule:\n" + "\n".join(missing)


# --------------------------------------------------------------- the model
def everything(units: str = "kN m C") -> Model:
    """A small building using every kind of input that carries units."""
    from structstudio.app.templates import new_building
    m = new_building(x_spacings=(6, 5), y_spacings=(5,), story_heights=(3.5, 3.0), grid_only=False)
    m.options.gravity = 9.81
    for mat in m.materials.values():
        mat.alpha = 1.0e-5
    beams = sorted(n for n, f in m.frames.items() if f.type == "Beam")
    cols = sorted(n for n, f in m.frames.items() if f.type == "Column")
    b0, b1, b2 = beams[0], beams[1], beams[2]
    m.add_load_pattern("Q", "Other")
    m.add_frame_distributed_load(b0, "Q", 4.0, 9.0, rel_a=1.0, rel_b=4.5, absolute=True)  # trapezoidal, absolute
    m.add_frame_distributed_load(b1, "Q", 2.5, direction="Z", load_type="Moment")  # distributed moment
    m.add_frame_point_load(b1, "Q", 30.0, rel=2.0, absolute=True)  # point force at 2 m
    m.add_frame_point_load(b2, "Q", 12.0, rel=0.3, load_type="Moment", direction="2")  # point moment
    top = max(m.points.values(), key=lambda p: (p.z, p.x, p.y))
    m.add_point_load(top.name, "Q", force=(15.0, -8.0, -20.0), moment=(3.0, -2.0, 5.0))
    floor = next(n for n, a in m.areas.items() if a.type == "Floor")
    m.add_area_load(floor, "Q", 3.0)
    # temperature: uniform change and a gradient across local 2
    m.add_load_pattern("TEMP", "Temperature")
    m.frames[b0].loads.append(type(m.frames[b0].loads[0])("TEMP", "temperature", "Gravity", "Temperature",
                                                          0.0, 1.0, False, 25.0, 25.0))
    m.frames[b2].loads.append(type(m.frames[b2].loads[0])("TEMP", "temperature", "Gravity", "Gradient2",
                                                          0.0, 1.0, False, 40.0, 40.0))
    # support settlement, springs, partial fixity, offsets, mass
    m.add_load_pattern("SET", "Other")
    base = sorted((p for p in m.points.values() if any(p.restraints)), key=lambda p: (p.x, p.y))
    base[0].ground_displacements.append(GroundDisplacement("SET", (0.0, 0.0, -0.01), (0.001, 0.0, 0.0)))
    base[1].restraints = (True, True, False, True, True, True)
    base[1].springs = (0.0, 0.0, 2.0e5)
    base[2].rotational_springs = (5.0e4, 5.0e4, 0.0)
    base[2].restraints = (True, True, True, False, False, True)
    f = m.frames[b1]
    f.releases_j = (False, False, False, False, False, True)
    f.fixity_j_rot = (0.0, 0.0, 8.0e3)  # partial fixity of the released M3
    m.frames[cols[0]].joint_offset_j = (0.05, 0.0, 0.0)
    m.frames[b2].end_offset_i, m.frames[b2].end_offset_j, m.frames[b2].auto_end_offsets = 0.2, 0.15, False
    top.mass = (2.0, 2.0, 0.0)
    m.line_springs["LS1"] = LineSpring("LS1", base[3].name, base[4].name if len(base) > 4 else base[0].name,
                                       stiffness=(0.0, 0.0, 1.0e4), rotational=50.0) if len(base) > 3 else None
    if m.line_springs.get("LS1") is None:
        del m.line_springs["LS1"]
    # a nonprismatic beam with an absolute-length haunch
    sec = m.frame_sections[m.frames[b0].section]
    deep = dataclasses.replace(sec, name="DEEP", t3=sec.t3 * 1.6, A=sec.A * 1.6, I33=sec.I33 * 4.1)
    m.add_frame_section(deep)
    m.add_frame_section(dataclasses.replace(sec, name="HAUNCH", nonprismatic=[
        {"start": "DEEP", "end": sec.name, "length": 1.2, "length_type": "Absolute", "ei33": "Parabolic", "ei22": "Linear"},
        {"start": sec.name, "end": sec.name, "length": 1.0, "length_type": "Variable", "ei33": "Linear", "ei22": "Linear"}]))
    m.frames[beams[3]].section = "HAUNCH"
    # a tendon (equivalent loads) with friction and wobble
    bt = m.frames[beams[4]]
    pi, pj = m.points[bt.point_i], m.points[bt.point_j]
    m.add_load_pattern("PT", "Other")
    m.tendons["T1"] = Tendon("T1", [TendonPoint("Start", (pi.x, pi.y, pi.z - 0.1)),
                                    TendonPoint("ParabolaIntermediate", ((pi.x + pj.x) / 2, (pi.y + pj.y) / 2, pi.z - 0.25)),
                                    TendonPoint("ParabolaEnd", (pj.x, pj.y, pj.z - 0.1))], area=1.0e-3,
                             frames=[bt.name], loads=[TendonLoad("PT", 800.0, "Both", 0.2, 0.004, 5.0e4)])
    for p in ("Q", "TEMP", "SET", "PT"):
        m.add_linear_static_case(p, {p: 1.0})
    m.add_linear_static_case("ACC", {}, acceleration_loads={"UX": 0.1})
    m.add_modal_case("Modal", max_modes=4)
    m.add_spectrum_function("G", [0.0, 0.5, 2.0], [0.4, 1.0, 0.25], normalized=True)
    m.add_spectrum_function("SI", [0.0, 0.5, 2.0], [3.0, 6.0, 1.5], normalized=False)  # m/s^2
    m.add_response_spectrum_case("RSG", "Modal", [("U1", "G", 9.81)])
    m.add_response_spectrum_case("RSA", "Modal", [("U2", "SI", 1.0)])
    m.add_combination("C1", {"Q": 1.2, "TEMP": 1.0, "SET": 1.0, "PT": 0.9})
    return m if units == "kN m C" else m.convert_units(units)


UNIT_SETS = [" ".join(p) for p in PRESETS] + ["kN m F", "kip in C"]
CASES = ("Q", "TEMP", "SET", "PT", "ACC", "RSG", "RSA", "C1")


def _numbers(obj, out, path=""):
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _numbers(obj[k], out, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _numbers(v, out, f"{path}[{i}]")
    elif isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    elif isinstance(obj, (int, float)):
        out[path] = float(obj)


@pytest.fixture(scope="module")
def base():
    return everything()


@pytest.mark.parametrize("units", UNIT_SETS)
def test_converting_there_and_back_is_exact(base, units):
    there = base.convert_units(units)
    back = there.convert_units("kN m C")
    a, b = {}, {}
    _numbers(base.to_dict(), a)
    _numbers(back.to_dict(), b)
    assert a.keys() == b.keys()
    bad = [k for k in a if not math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-12 * (abs(a[k]) + 1e-300))]
    assert not bad, bad[:10]
    # and the values really changed where they carry units (not a no-op)
    if units.split()[:2] != ["kN", "m"]:
        c = {}
        _numbers(there.to_dict(), c)
        changed = sum(1 for k in a if not math.isclose(a[k], c.get(k, a[k]), rel_tol=1e-9))
        assert changed > 10


def _si_results(m: Model):
    r = analyze(m, mesh_options=MeshOptions(floor_max_size=1.0, wall_max_size=1.0, refine_check=False))
    U = m.units
    L, F, M = U.factor("length"), U.factor("force"), U.factor("moment")
    out = {}
    for case in CASES:
        for p in sorted(m.points):
            d = np.asarray(r.joint_displacement(p, case), float)
            out[f"{case} U {p}"] = np.concatenate([d[:3] * L, d[3:]])
        for row in r.joint_reactions(case):
            out[f"{case} R {row['Point']}"] = np.array([row["FX"] * F, row["FY"] * F, row["FZ"] * F,
                                                        row["MX"] * M, row["MY"] * M, row["MZ"] * M])
        for fr in sorted(m.frames):
            st = r.frame_forces(fr, case)
            out[f"{case} F {fr}"] = np.concatenate([st.station * L, (st.forces[:, :3] * F).ravel(),
                                                    (st.forces[:, 3:] * M).ravel()])
    out["periods"] = np.array([row["Period"] for row in r.modal_periods("Modal")])
    return out


@pytest.fixture(scope="module")
def reference(base):
    return _si_results(base)


@pytest.mark.parametrize("units", ["N mm C", "kgf cm C", "tonf m C", "kip in F", "lb ft F", "kN m F"])
def test_same_structure_same_answer_in_every_unit_set(base, reference, units):
    got = _si_results(base.convert_units(units))
    assert got.keys() == reference.keys()
    # round-off of the solution (a unit mistake shows as a factor of 1.8, 10, 12, 1000 ...);
    # eigenvectors are iterative (converged to ~1e-9)
    for group, tol in (("static", 1e-7), ("dynamic", 1e-6)):
        worst, where = 0.0, ""
        for k, ref in reference.items():
            if (k.startswith("RS") or k == "periods") != (group == "dynamic"):
                continue
            scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1e-30)
            err = float(np.max(np.abs(got[k] - ref))) / scale if ref.size else 0.0
            if err > worst:
                worst, where = err, k
        assert worst < tol, f"{units}, {group}: largest difference {worst:.2e} at {where}"


def test_every_kind_of_input_is_in_the_test_model(base):
    """The physics check above is only as good as its model: it must really use them."""
    kinds = {(ld.kind, ld.load_type, ld.absolute) for f in base.frames.values() for ld in f.loads}
    assert {("distributed", "Force", True), ("distributed", "Moment", False), ("point", "Force", True),
            ("point", "Moment", False), ("temperature", "Temperature", False),
            ("temperature", "Gradient2", False)} <= kinds
    assert base.tendons and base.tendons["T1"].loads[0].wobble > 0
    assert any(s.nonprismatic for s in base.frame_sections.values())
    assert any(p.ground_displacements for p in base.points.values())
    assert not base.functions["SI"].normalized


def test_temperature_units():
    c, f = UnitSystem("kN", "m", "C"), UnitSystem("kN", "m", "F")
    assert c.factor("temperature_change") / f.factor("temperature_change") == pytest.approx(1.8)
    assert f.label("per_temperature") == "1/F" and f.label("temperature_gradient") == "F/m"
    m = everything().convert_units("kN m F")
    assert next(iter(m.materials.values())).alpha == pytest.approx(1.0e-5 / 1.8)
    t = next(ld for fr in m.frames.values() for ld in fr.loads if ld.load_type == "Temperature")
    assert t.value_a == pytest.approx(45.0)  # a 25 C change is a 45 F change (no offset)


def test_changing_units_in_the_program_keeps_the_results():
    """Options > Display Units / the units button: the model is converted, the
    results stay (reported in the new units), undo goes back."""
    from structstudio.app.document import Document
    m = everything()
    doc = Document(m)
    r = analyze(m, ["Q"], mesh_options=MeshOptions(floor_max_size=1.0, refine_check=False))
    doc.set_results(r)
    top = max(m.points.values(), key=lambda p: (p.z, p.x, p.y)).name
    beam = sorted(n for n, f in m.frames.items() if f.type == "Beam")[0]
    d0, f0 = r.joint_displacement(top, "Q"), r.frame_forces(beam, "Q").forces
    seen = []
    doc.on_change(seen.append)
    doc.change_units("kip in F")
    assert doc.model.units == UnitSystem("kip", "in", "F") and seen[:2] == ["units", "model"]
    assert doc.results is not None and doc.results.model is doc.model
    u = doc.model.units
    d1 = doc.results.joint_displacement(top, "Q")
    assert d1[:3] * u.factor("length") == pytest.approx(d0[:3], rel=1e-12)
    f1 = doc.results.frame_forces(beam, "Q").forces
    assert f1[:, 5] * u.factor("moment") == pytest.approx(f0[:, 5] * UnitSystem("kN", "m").factor("moment"),
                                                          rel=1e-12)
    assert doc.units_ratio == pytest.approx(1 / 0.0254)
    assert doc.locked  # locked after the analysis: undo asks to unlock (and deletes the results)
    doc.ask_unlock = lambda: True
    assert doc.undo() and doc.model.units == UnitSystem("kN", "m", "C") and not doc.locked


DATA = __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "verification", "data")


@pytest.mark.parametrize("name", ["etabs/Example02.e2k", "etabs/Example15d1.e2k", "sap2000/Example 1-004.SAP2000.tables.txt",
                                  "sap2000/Example 1-009a.SAP2000.tables.txt", "safe/S06a-Thin.SAFE.tables.txt"])
def test_imported_csi_models_give_the_same_answer_in_other_units(name):
    """ETABS / SAP2000 / SAFE verification models (kip, in, F) analysed again in N, mm, C."""
    import os
    from structstudio.io.importers import import_model
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip("verification data not present")
    m = import_model(path).model
    assert m.units.temperature == "F"  # the file's temperature unit is kept
    m2 = m.convert_units("N mm C")
    mo = MeshOptions(refine_check=False)
    r1, r2 = analyze(m, mesh_options=mo), analyze(m2, mesh_options=mo)
    for case in r1.cases:
        D1 = np.asarray(r1.displacement_envelope(case).max)[:, :3] * m.units.factor("length")
        D2 = np.asarray(r2.displacement_envelope(case).max)[:, :3] * m2.units.factor("length")
        assert D1.shape == D2.shape
        assert np.abs(D1 - D2).max() <= 1e-9 * max(np.abs(D1).max(), 1e-30), case


@pytest.mark.parametrize("units", UNIT_SETS)
def test_new_model_default_concrete_is_the_same_material_in_every_unit_set(units):
    """New Model > default properties: C28 (25 GPa, 24 kN/m3, 28 MPa) for metric
    lengths, 4000 psi (3600 ksi, 150 pcf) for inch / foot models - whatever the force unit."""
    from structstudio.app.templates import new_building
    m = new_building(units=units, grid_only=False)
    mat = next(iter(m.materials.values()))
    U = m.units
    E, w, fc = (mat.E * U.factor("stress"), mat.weight_density * U.factor("weight_per_volume"),
                mat.fc * U.factor("stress"))
    if U.length in ("in", "ft"):
        assert E == pytest.approx(3600 * 6894757.29, rel=1e-6) and w == pytest.approx(150 * 157.0874606, rel=1e-6)
        assert fc == pytest.approx(4000 * 6894.75729, rel=1e-6)
    else:
        assert (E, w, fc) == (pytest.approx(25e9), pytest.approx(24e3), pytest.approx(28e6))
    depth = max(s.t3 for s in m.frame_sections.values()) * U.factor("length")
    assert depth == pytest.approx(0.6 if U.length not in ("in", "ft") else 24 * 0.0254)
