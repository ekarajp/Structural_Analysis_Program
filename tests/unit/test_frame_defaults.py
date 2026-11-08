"""Model defaults for frame end offsets / rigid-zone factors (Analyze > Analysis
Model Defaults) and their use by the mesher."""
from structstudio.app.templates import new_building
from structstudio.io.e2k import E2KImporter
from structstudio.mesh import build_analysis_model
from structstudio.model import Model
from structstudio.model.properties import FrameSection, Material


def _model():
    m = Model("d", "kN m")
    m.add_material("C", E=30e6, nu=0.2)
    m.add_material("S", E=200e6, nu=0.3)
    m.materials["C"].type = "Concrete"
    m.materials["S"].type = "Steel"
    m.add_frame_section(FrameSection.rectangular("RC", "C", 0.4, 0.4))
    m.add_frame_section(FrameSection.rectangular("ST", "S", 0.3, 0.3))
    return m


def test_recommended_defaults_by_material():
    m = _model()
    a, b = m.add_point(0, 0, 0), m.add_point(0, 0, 3)
    rc = m.add_frame(a, b, "RC")
    st = m.add_frame(a, m.add_point(5, 0, 3), "ST")
    assert m.frame_auto_end_offsets(rc) and m.frame_rigid_factor(rc) == 0.5
    assert m.frame_rigid_factor(st) == 0.0
    m.options.rigid_zone_factor_concrete = 0.75
    assert m.frame_rigid_factor(rc) == 0.75
    rc.rigid_factor = 0.0  # the frame's own value wins
    assert m.frame_rigid_factor(rc) == 0.0
    given = m.add_frame(b, m.add_point(5, 5, 3), "RC", end_offset_i=0.2)
    assert not m.frame_auto_end_offsets(given)  # offsets given are the offsets


def test_imported_and_csi_models_keep_the_csi_defaults():
    m = _model()
    m.use_csi_frame_defaults()
    f = m.add_frame(m.add_point(0, 0, 0), m.add_point(0, 0, 3), "RC")
    assert not m.frame_auto_end_offsets(f) and m.frame_rigid_factor(f) == 0.0
    imp = E2KImporter.parse("$ PROGRAM INFORMATION\n  PROGRAM  \"ETABS\"  VERSION \"21\"\n")
    assert not imp.model.options.frame_auto_end_offsets and imp.model.options.rigid_zone_factor_concrete == 0.0


def test_new_buildings_get_rigid_end_zones():
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.5,), grid_only=False)
    am = build_analysis_model(m)
    beams = [e for e in am.frames if m.frames[e.object].type == "Beam"]
    assert beams and any(getattr(e, "rigid_i", 0) > 0 or getattr(e, "rigid_j", 0) > 0 for e in beams) \
        or any(getattr(e, "end_offset_i", 0) > 0 for e in beams)
