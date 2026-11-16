"""Edit > Edit Frames > Join Frames through joints that other objects use."""
import numpy as np
import pytest

from structstudio import Model, analyze
from structstudio.app import commands as C
from structstudio.app.document import Document
from structstudio.engine.kernel import SciPyKernel
from structstudio.model import FIXED, FrameSection


def _model():
    m = Model("join", "kN m")
    m.use_csi_frame_defaults()
    m.add_material("C", E=25e6, nu=.2)
    m.add_frame_section(FrameSection.general("B", "C", A=.12, I33=1.6e-3, I22=4e-4, J=1e-3))
    a = m.add_point(0, 0, 0, restraints=FIXED)
    mid = m.add_point(5, 0, 0)
    b = m.add_point(10, 0, 0, restraints=FIXED)
    f = m.add_frame(a, mid, "B", type="Beam").name
    g = m.add_frame(mid, b, "B", type="Beam").name
    # a cross beam framing in at the joint between the two pieces
    s = m.add_point(5, 4, 0, restraints=FIXED)
    h = m.add_frame(mid, s, "B", type="Beam").name
    m.add_load_pattern("P")
    m.add_frame_distributed_load(f, "P", -10.0)
    m.add_frame_distributed_load(g, "P", -10.0)
    m.add_point_load(mid, "P", force=(0, 0, -20))
    m.add_linear_static_case("P")
    return m, f, g, h, mid.name


def test_join_through_a_joint_used_by_another_beam_keeps_it_and_the_results():
    m, f, g, h, mid = _model()
    ref = analyze(m, kernel=SciPyKernel()).joint_displacement(mid, "P")
    doc = Document(m)
    assert C.join_frames(doc, [f, g]) == [g]
    m2 = doc.model
    assert mid in m2.points and len(m2.frames) == 2  # the joint stays (cross beam, load)
    got = analyze(m2, kernel=SciPyKernel()).joint_displacement(mid, "P")
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_join_refused_across_a_column_with_the_reason():
    m, f, g, h, mid = _model()
    top = m.add_point(5, 0, 3)
    m.add_frame(m.points[mid], top, "B", type="Column")
    doc = Document(m)
    assert C.join_frames(doc, [f, g]) == []
    assert any("column" in p for p in C.join_frames.last_problems)


def test_join_reports_different_sections_and_no_pair():
    m, f, g, h, mid = _model()
    m.add_frame_section(FrameSection.general("B2", "C", A=.2, I33=2e-3, I22=4e-4, J=1e-3))
    m.frames[g].section = "B2"
    doc = Document(m)
    assert C.join_frames(doc, [f, g]) == []
    assert any("different section" in p for p in C.join_frames.last_problems)
    m, f, g, h, mid = _model()
    doc = Document(m)
    assert C.join_frames(doc, [f, h]) == []  # at a right angle: not a pair
    assert C.join_frames.last_pairs == 0


def test_join_beams_along_a_slab_edge_same_results():
    from structstudio.model import ShellSection
    def build():
        m, f, g, h, mid = _model()
        m.frames.pop(h)
        m.add_shell_section(ShellSection("S", "C", thickness=.2))
        q1, q2 = m.add_point(0, 3, 0, restraints=FIXED), m.add_point(5, 3, 0)
        m.add_area([m.points[m.frames[f].point_i], m.points[mid], q2, q1], "S")
        return m, f, g, mid
    m, f, g, mid = build()
    ref = analyze(m, kernel=SciPyKernel()).joint_displacement(mid, "P")
    m, f, g, mid = build()
    doc = Document(m)
    assert C.join_frames(doc, [f, g]) == [g] and mid in doc.model.points
    got = analyze(doc.model, kernel=SciPyKernel()).joint_displacement(mid, "P")
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-12)
