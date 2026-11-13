"""Slab Design Model: a membrane floor taken out of the building to design its slabs.

Checked against an independent solution (Navier: a simply supported square plate),
equilibrium of the loads, and what the model must contain / leave out."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from structstudio.app.templates import new_building
from structstudio.design.slab_model import build_slab_model, membrane_stories
from structstudio.model.base import ModelError
from structstudio.model.properties import FrameModifiers

sys.path.insert(0, str(Path(__file__).parent))
from test_stiffness_sets import _run  # noqa: E402


def _building(xs=(6, 6), ys=(5, 5), hs=(3.5, 3.0, 3.0), membrane=True):
    m = new_building(x_spacings=xs, y_spacings=ys, story_heights=hs, grid_only=False)
    if membrane:
        for s in m.shell_sections.values():
            if s.type == "Slab":
                s.model_type = "Membrane"
    return m


def _floors(m, story):
    z = m.stories[story].elevation
    return [a for a in m.areas.values() if a.type == "Floor" and all(abs(m.points[p].z - z) < 1e-9 for p in a.points)]


def test_contents_and_the_building_is_not_changed():
    m = _building()
    m.add_load_pattern("EQX", "Seismic")
    m.add_linear_static_case("EQX", {"EQX": 1.0})
    m.add_combination("G+E", {"Dead": 1.2, "Live": 0.5, "EQX": 1.0})
    m.add_combination("G", {"Dead": 1.4})
    before = m.to_dict()
    assert membrane_stories(m) == ["Story1", "Story2", "Story3"]
    r = build_slab_model(m, "Story2")
    assert m.to_dict() == before
    sm = r.model
    z = m.stories["Story2"].elevation
    # 4 slabs as plates, 12 beams of the level, 9 columns x (below + above)
    slabs = [a for a in sm.areas.values() if a.type == "Floor"]
    assert len(slabs) == 4 and all(sm.shell_sections[a.section].model_type == "ShellThick" for a in slabs)
    assert all(a.diaphragm is None for a in sm.areas.values())
    beams = [f for f in sm.frames.values() if f.type == "Beam"]
    assert len(beams) == 12 and all(abs(sm.points[f.point_i].z - z) < 1e-9 for f in beams)
    stubs = [f for f in sm.frames.values() if "~" in f.name]
    assert len(stubs) == 18 and r.counts["column stubs"] == 18
    far = {p for f in stubs for p in (f.point_i, f.point_j) if abs(sm.points[p].z - z) > 1e-9}
    assert len(far) == 18 and all(all(sm.points[p].restraints) for p in far)
    assert {round(sm.points[p].z, 6) for p in far} == {3.5, 9.5}
    # gravity only
    assert "EQX" not in sm.load_patterns and set(sm.load_cases) == {"Dead", "Live"}
    assert set(sm.load_combinations) == {"G", "1.2D+1.6L"}
    assert any("G+E" in n for n in r.notes)
    # the top story: stubs below only; pinned
    r3 = build_slab_model(m, "Story3", columns="pinned")
    stubs = [f for f in r3.model.frames.values() if "~" in f.name]
    assert len(stubs) == 9 and all(f.name.endswith("~dn") for f in stubs)
    far = [p for p in r3.model.points.values() if p.z < 9.5 - 1e-9]
    assert far and all(p.restraints == (True, True, True, False, False, False) for p in far)


def test_loads_reach_the_supports():
    m = _building()
    for a in _floors(m, "Story2"):
        m.add_area_load(a.name, "Live", 3.0)
    for a in _floors(m, "Story1"):  # another floor: not in the model
        m.add_area_load(a.name, "Live", 50.0)
    # a two-way load panel over part of the floor: carried by the slab
    from structstudio.model.objects import PanelLoadCase
    pl = m.add_panel_load([(1, 1, 6.5), (5, 1, 6.5), (5, 4, 6.5), (1, 4, 6.5)], distribution="two_way")
    pl.loads = [PanelLoadCase("Live", 2.0)]
    r = build_slab_model(m, "Story2")
    assert list(r.model.panel_loads.values())[-1].distribution == "shells"
    res = _run(r.model)
    R = res.base_reactions("Live")
    assert R[2] == pytest.approx(3.0 * 12 * 10 + 2.0 * 4 * 3, rel=1e-9)
    assert abs(R[0]) < 1e-6 and abs(R[1]) < 1e-6


def test_simply_supported_plate_navier():
    """One 6 x 6 m bay, rigid beams without torsional stiffness, pinned columns: the
    slab is a simply supported square plate.  Navier: M = 0.0368(1+ν) q a², w = 0.00406 q a⁴/D."""
    m = _building(xs=(6,), ys=(6,), hs=(3.0, 3.0))
    for f in m.frames.values():
        if f.type == "Beam":
            f.modifiers = FrameModifiers(J=1e-6)
    q, a = 10.0, 6.0
    for fl in _floors(m, "Story1"):
        m.add_area_load(fl.name, "Live", q)
    r = build_slab_model(m, "Story1", beams="rigid", columns="pinned")
    sm = r.model
    sec = next(sm.shell_sections[x.section] for x in sm.areas.values() if x.type == "Floor")
    mat = sm.materials[sec.material]
    D = mat.E * sec.thickness ** 3 / (12 * (1 - mat.nu ** 2))
    res = _run(sm)
    node = next(n.id for n in res.am.nodes if np.allclose(n.xyz, [3, 3, 3]))
    M = [row["M11"] for row in res.shell_forces("Live") if row["Node"] == node]
    w = res.cases["Live"].displacements[node, 2]
    assert np.mean(M) == pytest.approx(0.0368 * (1 + mat.nu) * q * a * a, rel=0.03)
    assert -w == pytest.approx(0.00406 * q * a ** 4 / D, rel=0.04)
    # flexible beams deflect: the slab deflects more
    r2 = build_slab_model(m, "Story1", beams="flexible", columns="pinned")
    res2 = _run(r2.model)
    node2 = next(n.id for n in res2.am.nodes if np.allclose(n.xyz, [3, 3, 3]))
    assert -res2.cases["Live"].displacements[node2, 2] > -w * 1.05


def test_walls_and_errors():
    m = _building()
    # a wall under Story2 between two column lines
    p = [m.find_point(0, 0, 3.5), m.find_point(6, 0, 3.5), m.find_point(6, 0, 6.5), m.find_point(0, 0, 6.5)]
    wsec = next(s.name for s in m.shell_sections.values() if s.type == "Wall")
    m.add_area([q.name for q in p], wsec, type="Wall")
    r = build_slab_model(m, "Story2")
    assert r.counts["wall panels"] == 1
    walls = [a for a in r.model.areas.values() if a.type == "Wall"]
    low = [r.model.points[q] for q in walls[0].points if r.model.points[q].z < 6.5 - 1e-9]
    assert len(low) == 2 and all(all(q.restraints) for q in low)
    _run(r.model)
    shell = _building(membrane=False)
    with pytest.raises(ModelError) as e:
        build_slab_model(shell, "Story2")
    assert e.value.code == "NO_MEMBRANE_FLOOR"
    with pytest.raises(ModelError):
        build_slab_model(m, "Nope")


# ------------------------------------------------------------- the window
CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"


@pytest.mark.skipif(CHILD, reason="parent only")
def test_slab_model_window_in_subprocess():
    pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtk")
    import shutil
    env = dict(os.environ, STRUCTSTUDIO_GUI_CHILD="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "gui_", __file__]
    if not env.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24"] + cmd
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower(), \
        (proc.stdout + proc.stderr)[-4000:]


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_slab_model_dialog(tmp_path, monkeypatch):
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    from structstudio.app.document import Document
    from structstudio.app.mainwindow import MainWindow
    from structstudio.app.slab_model_dialog import SlabModelDialog
    m = _building()
    w = MainWindow(Document(m), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.show()
    app.processEvents()
    assert "design.slab.model" in w.menu_commands()
    d = SlabModelDialog(w)
    names = [d.stories.item(i).text() for i in range(d.stories.count())]
    assert names == ["Story3", "Story2", "Story1"]
    assert d.chosen() == ["Story3"]
    d.stories.item(0).setCheckState(QtCore.Qt.Unchecked)
    d.stories.item(1).setCheckState(QtCore.Qt.Checked)
    d.beams.setCurrentIndex(1)
    d.run_box.setChecked(False)
    before = m.to_dict()
    d.create()
    app.processEvents()
    assert len(d.windows) == 1
    sw = d.windows[0]
    assert w.doc.model.design.slab_model.stories == ["Story2"] and w.doc.model.design.slab_model.beams == "rigid"
    after = w.doc.model.to_dict()
    after["design"] = before["design"]
    assert after == before  # only the preferences changed
    assert "Slab Design Model Story2" in sw.doc.model.name
    assert all(sw.doc.model.shell_sections[a.section].model_type == "ShellThick"
               for a in sw.doc.model.areas.values() if a.type == "Floor")
    res = _run(sw.doc.model)
    sw.doc.set_results(res)
    app.processEvents()
    assert sw.doc.results is res and w.doc.results is None
    for win in (sw, w):
        win.doc.dirty = False
        win.views.finalize()
        win.close()
    for _ in range(3):
        app.processEvents()
