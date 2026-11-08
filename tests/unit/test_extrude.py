"""Extruded view: real cross-sections and slab / wall thickness."""
import math

import numpy as np
import pytest

pytest.importorskip("vtk")

from structstudio.app import scene  # noqa: E402
from structstudio.app.extrude import area_solids, cardinal_offset, frame_solids, section_profile  # noqa: E402
from structstudio.model import FrameSection, Model, ShellSection  # noqa: E402


def _area(pieces):
    tot = 0.0
    for p in pieces:
        x, y = p[:, 0], p[:, 1]
        a = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert a > 0  # counter-clockwise: faces point outward
        tot += a
    return tot


@pytest.mark.parametrize("shape,dims,area", [
    ("Steel I/Wide Flange", dict(t3=0.4, t2=0.2, tf=0.02, tw=0.01), 2 * 0.2 * 0.02 + 0.36 * 0.01),
    ("Steel Tube", dict(t3=0.3, t2=0.2, tf=0.015, tw=0.01), 0.3 * 0.2 - 0.27 * 0.18),
    ("Steel Channel", dict(t3=0.3, t2=0.1, tf=0.015, tw=0.01), 0.3 * 0.01 + 2 * 0.09 * 0.015),
    ("Steel Angle", dict(t3=0.2, t2=0.15, tf=0.02, tw=0.02), 0.2 * 0.02 + 0.13 * 0.02),
    ("Steel Tee", dict(t3=0.3, t2=0.25, tf=0.02, tw=0.012), 0.25 * 0.02 + 0.28 * 0.012),
    ("Concrete Rectangular", dict(t3=0.6, t2=0.3), 0.18),
])
def test_profiles_have_the_section_area(shape, dims, area):
    sec = FrameSection(name="S", material="M", shape=shape, **dims)
    assert _area(section_profile(sec)) == pytest.approx(area, rel=1e-9)


def test_round_and_general_sections():
    pipe = section_profile(FrameSection(name="P", material="M", shape="Steel Pipe", t3=0.3, tw=0.02))
    assert _area(pipe) == pytest.approx(math.pi * (0.15 ** 2 - 0.13 ** 2), rel=0.02)
    circ = section_profile(FrameSection.circular("C", "M", 0.5))
    assert _area(circ) == pytest.approx(math.pi * 0.25 ** 2, rel=0.02)
    # General: rectangle with the same A and I33
    gen = section_profile(FrameSection.general("G", "M", A=0.18, I33=0.0054))
    lo, hi = np.vstack(gen).min(0), np.vstack(gen).max(0)
    assert (hi - lo) == pytest.approx([0.6, 0.3])
    assert section_profile(FrameSection.general("Z", "M", A=0, I33=0)) is None
    rect = section_profile(FrameSection.rectangular("R", "M", 0.6, 0.3))
    assert cardinal_offset(rect, 8) == pytest.approx([0.3, 0.0])  # top centre
    assert cardinal_offset(rect, 1) == pytest.approx([-0.3, -0.15])


def _bounds(poly):
    return np.array(poly.GetBounds()).reshape(3, 2)


def test_members_follow_local_axes_cardinal_point_and_slabs_hang_below_level():
    m = Model("t", "kN m")
    m.add_material("C", E=3e7)
    m.add_frame_section(FrameSection.rectangular("B", "C", 0.6, 0.3))
    m.add_frame_section(FrameSection.rectangular("COL", "C", 0.8, 0.4))
    m.add_shell_section(ShellSection("S", "C", thickness=0.2))
    m.set_stories([("L1", 3.0)])
    b = m.add_frame((0, 0, 3), (6, 0, 3), "B", type="Beam")
    c = m.add_frame((0, 0, 0), (0, 0, 3), "COL", type="Column", angle=90)
    poly, missing = frame_solids(m, [b.name])
    assert not missing
    bb = _bounds(poly)
    assert bb[2] == pytest.approx([2.4, 3.0])  # story model: beam top at the level (cardinal 8)
    assert bb[1] == pytest.approx([-0.15, 0.15])
    b.display_cardinal = 10
    assert _bounds(frame_solids(m, [b.name])[0])[2] == pytest.approx([2.7, 3.3])
    cb = _bounds(frame_solids(m, [c.name])[0])  # column, depth along local 2 = +Y after 90 deg
    assert cb[0] == pytest.approx([-0.2, 0.2]) and cb[1] == pytest.approx([-0.4, 0.4])
    a = m.add_area([(0, 0, 3), (6, 0, 3), (6, 5, 3), (0, 5, 3)], "S")
    ab = _bounds(area_solids(m, [a]))
    assert ab[2] == pytest.approx([2.8, 3.0])
    acts = scene.build_model_actors(m, extruded=True).actors
    assert "frames_Beam" in acts and "areas_Floor_solid" in acts and "frames_Beam_edges" in acts
    lines = scene.build_model_actors(m, extruded=False).actors
    assert "areas_Floor" in lines and "areas_Floor_solid" not in lines
