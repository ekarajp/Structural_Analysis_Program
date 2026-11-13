"""Several grid systems: Cartesian (moved / turned) and cylindrical (ETABS: R = circles of
constant radius, T = radial lines of constant angle)."""
import math
import os
import subprocess
import sys

import pytest

from structstudio.app.templates import new_building
from structstudio.app.tools import grid_intersections, grid_ordinates
from structstudio.model import Model
from structstudio.model.grids import grid_geometry, grid_names_at, line_intersection
from structstudio.model.objects import GridLine, GridSystem


def _with_systems():
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.0, 3.0), grid_only=True)
    m.grid_systems["G2"] = GridSystem(name="G2", origin=(20.0, 0.0), rotation=30.0)
    for k, u in enumerate((0.0, 4.0)):
        m.grid_lines.append(GridLine(system="G2", name=f"P{k + 1}", direction="X", ordinate=u))
    for k, v in enumerate((0.0, 3.0)):
        m.grid_lines.append(GridLine(system="G2", name=f"Q{k + 1}", direction="Y", ordinate=v))
    m.grid_systems["C1"] = GridSystem(name="C1", type="Cylindrical", origin=(0.0, 30.0), rotation=10.0)
    for k, a in enumerate(range(0, 360, 45)):
        m.grid_lines.append(GridLine(system="C1", name=f"T{k + 1}", direction="T", ordinate=float(a)))
    for k, r in enumerate((4.0, 8.0)):
        m.grid_lines.append(GridLine(system="C1", name=f"R{k + 1}", direction="R", ordinate=r))
    return m


def test_geometry_of_the_systems():
    m = _with_systems()
    g = grid_geometry(m)
    # G1: 3 + 2 lines, G2: 2 + 2, C1: 8 radial lines + 2 full circles
    assert len(g.lines) == 5 + 4 + 8 and len(g.arcs) == 2
    assert all(abs((a1 - a0) - 360.0) < 1e-9 for _, _, a0, a1, _, _ in g.arcs)
    pts = {(round(x, 6), round(y, 6)) for x, y in g.points}
    # G2 point (4, 3) local -> turned 30 degrees about (20, 0)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    assert (round(20 + 4 * c - 3 * s, 6), round(4 * s + 3 * c, 6)) in pts
    # C1: R2 x T3 (90 + 10 degrees), and the centre
    t = math.radians(100)
    assert (round(8 * math.cos(t), 6), round(30 + 8 * math.sin(t), 6)) in pts
    assert (0.0, 30.0) in pts
    assert len(pts) == 6 + 4 + 16 + 1
    # only G1 (along the axes) gives global ordinates; G2 is turned
    assert grid_ordinates(m) == ([0.0, 6.0, 12.0], [0.0, 5.0])
    assert len(grid_intersections(m)) == 27


def test_partial_arcs_and_names():
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=True)
    m.grid_systems["C"] = GridSystem(name="C", type="Cylindrical", origin=(0.0, 0.0))
    for k, a in enumerate((0.0, 45.0, 90.0)):
        m.grid_lines.append(GridLine(system="C", name=f"t{k}", direction="T", ordinate=a))
    m.grid_lines.append(GridLine(system="C", name="r1", direction="R", ordinate=10.0))
    g = grid_geometry(m)
    (_, r, a0, a1, _, _), = g.arcs
    assert r == 10.0 and a0 < 0.0 and 90.0 < a1 < 120.0  # a quarter, run a little past the radial lines
    ln = {n: gl for gl in m.grid_lines for n in [gl.name]}
    assert line_intersection(m, ln["t1"], ln["r1"]) == pytest.approx((10 / math.sqrt(2), 10 / math.sqrt(2)))
    assert set(grid_names_at(m, 10 / math.sqrt(2), 10 / math.sqrt(2), 1e-6)) == {"t1", "r1"}


def test_moved_cartesian_system_ordinates():
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=True)
    m.grid_systems["G1"].origin = (100.0, 50.0)
    assert grid_ordinates(m) == ([100.0, 106.0], [50.0, 55.0])
    from structstudio.app.locate import grid_ref
    assert grid_ref(m, (106.0, 55.0, 0.0)) == "B/2"


def test_saved_and_units():
    m = _with_systems()
    m2 = Model.from_dict(m.to_dict())
    assert m2.grid_systems["C1"].type == "Cylindrical" and m2.grid_systems["G2"].rotation == 30.0
    m3 = m2.convert_units("kN mm C")
    t = [g for g in m3.grid_lines if g.direction == "T"]
    r = [g for g in m3.grid_lines if g.direction == "R"]
    assert [g.ordinate for g in t][:3] == [0.0, 45.0, 90.0]  # angles stay degrees
    assert [g.ordinate for g in r] == [4000.0, 8000.0]
    assert m3.grid_systems["C1"].origin == pytest.approx((0.0, 30000.0))


def test_snap_on_cylindrical_grid():
    from structstudio.app.snap import SnapEngine, SnapOptions
    from structstudio.app.tools import Plane
    m = _with_systems()
    eng = SnapEngine(m, Plane("plan", 3.0, "Story1"), SnapOptions(), lambda xyz: (xyz[0], xyz[1]),
                     lambda x, y: None, None, 1e-6, ())
    segs = [s for s in eng.segments() if s.kind == "grid"]
    names = {s.name for s in segs}
    assert {"R1", "R2", "T1", "P1", "Q2", "A"} <= names
    grid_pts = {(round(h.xyz[0], 6), round(h.xyz[1], 6)) for h in eng._point_candidates() if h.kind == "grid"}
    assert (0.0, 30.0) in grid_pts and len(grid_pts) == 27
    t = math.radians(100)
    assert (round(4 * math.cos(t), 6), round(30 + 4 * math.sin(t), 6)) in grid_pts
    # what the cursor finds at the centre of the cylindrical grid
    hits = eng.find(0.0, 30.0)
    assert hits and hits[0].kind == "grid" and hits[0].xyz[:2] == pytest.approx((0.0, 30.0))


def test_find_grid_point_of_any_system():
    from structstudio.app.find import _grid_xy
    m = _with_systems()
    assert _grid_xy(m, "B/2") == (6.0, 5.0)
    x, y = _grid_xy(m, "T3/R2")
    assert (x, y) == pytest.approx((8 * math.cos(math.radians(100)), 30 + 8 * math.sin(math.radians(100))))


# ------------------------------------------------------------- the window
CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"


@pytest.mark.skipif(CHILD, reason="parent only")
def test_grid_systems_window_in_subprocess():
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
def test_gui_grid_systems_dialog(tmp_path, monkeypatch):
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    from structstudio.app.dialogs import GridStoriesDialog
    from structstudio.app.document import Document
    from structstudio.app.mainwindow import MainWindow
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.0, 3.0), grid_only=True)
    w = MainWindow(Document(m), QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.show()
    app.processEvents()
    d = GridStoriesDialog(w.doc, w)
    assert [d.tabs.tabText(i) for i in range(d.tabs.count())] == ["X Grid", "Y Grid", "Story"]
    assert d.system_box.currentText() == "G1" and d.x_table.rowCount() == 3
    d._add_system("Cylindrical")
    assert d.system_box.currentText() == "C1" and d.tabs.tabText(0) == "Radial lines"
    assert d.x_table.rowCount() == 8 and d.y_table.rowCount() == 2
    d.ox.setText("30")
    d.rot.setText("15")
    d._add_row(d.y_table, ("C1-T3", 12.0))
    d.system_box.setCurrentText("G1")  # back: G1 untouched, C1 kept
    assert d.tabs.tabText(0) == "X Grid" and d.x_table.rowCount() == 3
    d._add_system("Cartesian")
    d.rot.setText("45")
    d.accept()
    mm = w.doc.model
    assert set(mm.grid_systems) == {"G1", "C1", "G2"}
    c1 = mm.grid_systems["C1"]
    assert c1.type == "Cylindrical" and c1.origin == (30.0, 0.0) and c1.rotation == 15.0
    assert sorted(g.ordinate for g in mm.grid_lines if g.system == "C1" and g.direction == "R") == [5.0, 10.0, 12.0]
    assert len([g for g in mm.grid_lines if g.system == "C1" and g.direction == "T"]) == 8
    assert mm.grid_systems["G2"].rotation == 45.0
    assert [g.name for g in mm.grid_lines if g.system == "G1"] == ["A", "B", "C", "1", "2"]
    app.processEvents()
    for vp in w.views.viewports:
        vp.render()
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    for _ in range(3):
        app.processEvents()
