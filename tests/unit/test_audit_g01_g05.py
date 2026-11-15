"""Regression tests for the audit findings G01, G02, G03 and G05
(comprehensive-audit 20261115-130000).  G04 (frame self weight on the clear
length) is a documented contract, tested in test_e2k_towers.py."""
import builtins
import os

import pytest

from structstudio import Model, ModelError, analyze
from structstudio.app.document import Document
from structstudio.engine.kernel import SciPyKernel
from structstudio.model import FIXED, FrameSection
from structstudio.model.nonlinear import FiberSectionDef, FrameNonlinearProperty, NonlinearMaterial


def bar(fixed=False):
    m = Model("Audit", "N m")
    m.use_csi_frame_defaults()
    m.add_material("S", E=200e9, nu=.3, type="Steel")
    m.add_frame_section(FrameSection.general("B", "S", A=.01, I33=2.5e-5, I22=2.5e-5, J=5e-5))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    b = m.add_point(3, 0, 0, restraints=FIXED if fixed else (False, True, True, True, True, True))
    f = m.add_frame(a, b, "B")
    m.add_load_pattern("P")
    m.add_point_load(b, "P", force=(1000, 0, 0))
    m.add_linear_static_case("P")
    return m, a, b, f


# ---------------------------------------------------------------- G01
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_point_load_is_refused_before_solving(value):
    m, a, b, f = bar()
    b.loads[0].force = (value, 0, 0)
    assert any(e.code == "NONFINITE_VALUE" and "loads" in str(e.path) for e in m.validate())
    with pytest.raises(ModelError) as e:
        analyze(m, kernel=SciPyKernel())
    assert e.value.code == "NONFINITE_INPUT"


def test_finite_model_still_runs():
    m, a, b, f = bar()
    assert not m.validate()
    r = analyze(m, kernel=SciPyKernel())
    # u = P L / E A = 1000 * 3 / (200e9 * 0.01)
    assert r.joint_displacement(b.name, "P")[0] == pytest.approx(1.5e-6, rel=1e-9)


# ---------------------------------------------------------------- G02
def _failing_open(folder):
    real_open = builtins.open

    class Writer:
        def __init__(self, fh):
            self.fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.fh.close()

        def write(self, text):
            self.fh.write(text[:100])
            self.fh.flush()
            raise OSError("simulated write failure")

        def __getattr__(self, name):
            return getattr(self.fh, name)

    def fail_open(name, mode="r", *args, **kw):
        fh = real_open(name, mode, *args, **kw)
        return Writer(fh) if "w" in mode and str(name).startswith(str(folder)) else fh
    return fail_open


def test_failed_save_keeps_previous_file_and_leaves_no_temp(tmp_path, monkeypatch):
    m, a, b, f = bar()
    d = Document(m)
    p = tmp_path / "saved.ssj.json"
    d.save(str(p))
    before = p.read_bytes()
    d.edit(lambda mm: mm.add_point(1, 1, 1))
    with monkeypatch.context() as mp:
        mp.setattr(builtins, "open", _failing_open(tmp_path))
        with pytest.raises(OSError):
            d.save()
    assert p.read_bytes() == before
    assert d.dirty
    assert sorted(os.listdir(tmp_path)) == ["saved.ssj.json"]  # no temporary file left behind


def test_save_keeps_previous_version_as_bak(tmp_path):
    m, a, b, f = bar()
    d = Document(m)
    p = tmp_path / "saved.ssj.json"
    d.save(str(p))
    first = p.read_bytes()
    d.edit(lambda mm: mm.add_point(1, 1, 1))
    d.save()
    assert (tmp_path / "saved.ssj.json.bak").read_bytes() == first
    assert len(Model.from_json(str(p)).points) == 3
    assert not d.dirty


# ---------------------------------------------------------------- G03
@pytest.mark.parametrize("end,dof", [("j", 5), ("i", 4), ("j", 0), ("i", 1)])
def test_fiber_member_with_end_release_is_refused(end, dof):
    m, a, b, f = bar()
    rel = [False] * 6
    rel[dof] = True
    setattr(f, f"releases_{end}", tuple(rel))
    m.load_cases.clear()
    b.loads[0].force = (3e6, 0, 0)
    m.add_nonlinear_material(NonlinearMaterial("S1", "Steel01", dict(Fy=250e6, E0=200e9, b=.01)))
    m.add_fiber_section(FiberSectionDef("F", "Custom",
                                        fibers=[[y, z, .0025, "S1"] for y in (-.05, .05) for z in (-.05, .05)]))
    m.add_frame_nonlinear(FrameNonlinearProperty("NLF", "F"), [f.name])
    m.add_nonlinear_static_case("NL", {"P": 1}, n_steps=30, geometric_nonlinearity="None")
    with pytest.raises(ModelError) as e:
        analyze(m, kernel=SciPyKernel())
    assert e.value.code == "UNSUPPORTED_FIBER_RELEASE"
    assert f.name in str(e.value)


def test_fiber_member_without_release_yields():
    """Same bar without the release: Steel01, Fy 250 MPa, b 0.01, P = 3 MN on A = 0.01 m2:
    eps = Fy/E + (sigma - Fy)/(b E) = 1.25e-3 + 50e6/2e9 = 0.02625;  u = 3 m x eps = 78.75 mm."""
    m, a, b, f = bar()
    m.load_cases.clear()
    b.loads[0].force = (3e6, 0, 0)
    m.add_nonlinear_material(NonlinearMaterial("S1", "Steel01", dict(Fy=250e6, E0=200e9, b=.01)))
    m.add_fiber_section(FiberSectionDef("F", "Custom",
                                        fibers=[[y, z, .0025, "S1"] for y in (-.05, .05) for z in (-.05, .05)]))
    m.add_frame_nonlinear(FrameNonlinearProperty("NLF", "F"), [f.name])
    m.add_nonlinear_static_case("NL", {"P": 1}, n_steps=30, geometric_nonlinearity="None")
    r = analyze(m, kernel=SciPyKernel())
    assert r.joint_displacement(b.name, "NL")[0] == pytest.approx(.07875, rel=1e-6)


# ---------------------------------------------------------------- G05
def test_failed_edit_with_full_history_keeps_every_undo_step():
    m, a, b, f = bar()
    d = Document(m)
    for k in range(d.MAX_UNDO):
        d.edit(lambda mm, v=k: setattr(mm.info, "project_name", f"v{v}"))
    d.undo()
    before = d.model.hash()
    undo, redo = list(d._undo), list(d._redo)
    with pytest.raises(ModelError):
        d.edit(lambda mm: setattr(mm.points[b.name], "x", float("inf")))
    assert d.model.hash() == before and d._undo == undo and d._redo == redo
    d.redo()  # the redo step survived the refused command
    assert d.model.info.project_name == f"v{d.MAX_UNDO - 1}"


def test_failed_design_edit_keeps_history():
    m, a, b, f = bar()
    d = Document(m)
    for k in range(d.MAX_UNDO):
        d.edit(lambda mm, v=k: setattr(mm.info, "project_name", f"v{v}"))
    undo = list(d._undo)

    def boom(mm):
        mm.info.project_name = "changed"
        raise RuntimeError("fail")
    with pytest.raises(RuntimeError):
        d.edit_design(boom)
    assert d._undo == undo and d.model.info.project_name == f"v{d.MAX_UNDO - 1}"
