"""Edit > Divide Shells: equal pieces / largest size / along frames drawn on the shell,
grid lines and story levels (like Divide Frames)."""
import numpy as np
import pytest

from structstudio import Model
from structstudio.app import shell_divide as SD
from structstudio.app.document import Document
from structstudio.model import FrameSection
from structstudio.model.objects import GridLine


def _doc():
    m = Model("div", "kN m")
    m.add_material("C", E=25e6, nu=.2)
    m.add_frame_section(FrameSection.general("B", "C", A=.1, I33=1e-3, I22=1e-3, J=1e-3))
    return Document(m)


def _slab(doc, xy, z=3.0):
    m = doc.model
    return m.add_area([m.add_point(x, y, z) for x, y in xy], None, type="Floor").name


def _beam(doc, a, b):
    m = doc.model
    return m.add_frame(m.add_point(*a), m.add_point(*b), "B").name


def _area(doc, names):
    m = doc.model
    return sum(SD._poly_area(SD._xyz(m, m.areas[n])) for n in names)


def _divide(doc, n, planes):
    m = doc.model
    return SD.divide(doc, {n: SD.cut(SD._xyz(m, m.areas[n]), planes, m.options.merge_tolerance)})


RECT = [(0, 0), (8, 0), (8, 6), (0, 6)]


def test_equal_and_max_size():
    doc = _doc()
    s = _slab(doc, RECT)
    m = doc.model
    assert len(SD.pieces_equal(m, m.areas[s], 2, 3)) == 6
    ps = SD.pieces_max_size(m, m.areas[s], 2.5)  # 8 / 2.5 -> 4 pieces, 6 / 2.5 -> 3
    assert len(ps) == 12 and all(max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) <= 2.5 for q in ps)
    # a shape other than a quadrilateral: along its local axes
    L = _slab(doc, [(20, 0), (24, 0), (24, 2), (22, 2), (22, 4), (20, 4)])
    ps = SD.pieces_equal(m, m.areas[L], 2, 2)
    assert len(ps) == 3 and sum(SD._poly_area(q) for q in ps) == pytest.approx(12.0)  # one quarter is outside


def test_along_frames_that_run_across_the_shell():
    doc = _doc()
    s = _slab(doc, RECT)
    m = doc.model
    across = _beam(doc, (4, 0, 3), (4, 6, 3))  # a secondary beam right across the slab
    _beam(doc, (0, 3, 3), (5, 3, 3))  # stops inside the slab: no cut
    _beam(doc, (2, 0, 0), (2, 6, 0))  # another level: no cut
    _beam(doc, (0, 0, 3), (8, 0, 3))  # along the edge: no cut
    planes = SD.planes_at_frames(m, m.areas[s])
    assert len(planes) == 1
    made = _divide(doc, s, planes)
    assert len(made) == 2 and _area(doc, made) == pytest.approx(48.0)
    f = m.frames[across]
    for n in made:  # both pieces have the beam's joints: the beam is the shared edge
        assert {f.point_i, f.point_j} <= set(m.areas[n].points)
    # only the selected frames
    s2 = _slab(doc, [(x + 20, y) for x, y in RECT])
    _beam(doc, (24, 0, 3), (24, 6, 3))
    assert SD.planes_at_frames(m, m.areas[s2], frames=[across]) == []


def test_l_shaped_slab_cut_at_its_inner_corner():
    doc = _doc()
    s = _slab(doc, [(0, 0), (8, 0), (8, 4), (4, 4), (4, 8), (0, 8)])
    _beam(doc, (4, 0, 3), (4, 4, 3))  # from the inner corner to the far edge
    m = doc.model
    made = _divide(doc, s, SD.planes_at_frames(m, m.areas[s]))
    assert sorted(round(_area(doc, [n]), 6) for n in made) == [16.0, 32.0]


def test_grid_lines_and_story_levels():
    doc = _doc()
    m = doc.model
    sysname = next(iter(m.grid_systems), "G1")
    for nm, d, o in (("A", "X", 0.0), ("B", "X", 3.0), ("C", "X", 8.0), ("1", "Y", 0.0), ("2", "Y", 6.0)):
        m.grid_lines.append(GridLine(sysname, nm, d, o))
    s = _slab(doc, RECT)
    planes = SD.planes_at_grids(m, m.areas[s])
    assert len(planes) == 1  # B only: the others run along the edges
    assert len(_divide(doc, s, planes)) == 2
    # a wall along grid 1 crossed by grid B, through two stories
    m.add_story("S1", 3.5)
    m.add_story("S2", 7.0)
    w = m.add_area([m.add_point(*p) for p in ((0, 0, 0), (8, 0, 0), (8, 0, 7), (0, 0, 7))], None, type="Wall").name
    planes = SD.planes_at_grids(m, m.areas[w]) + SD.planes_at_stories(m, m.areas[w])
    made = _divide(doc, w, planes)
    assert len(made) == 4 and _area(doc, made) == pytest.approx(56.0)


def test_pieces_keep_loads_and_one_undo():
    doc = _doc()
    m = doc.model
    m.add_load_pattern("SDL")
    s = _slab(doc, RECT)
    m.add_area_load(s, "SDL", 2.0)
    _beam(doc, (4, 0, 3), (4, 6, 3))
    n0 = len(m.areas)
    made = _divide(doc, s, SD.planes_at_frames(m, m.areas[s]))
    assert len(doc.model.areas) == n0 + 1
    panels = [p for p in doc.model.panel_loads.values()]
    assert sorted(p.area for p in panels) == sorted(made)  # each piece keeps the floor's area load
    doc.undo()
    assert s in doc.model.areas and len(doc.model.areas) == n0


def test_dialog_choices():
    from PySide6 import QtWidgets
    from structstudio.app.divide_dialog import DivideShellsDialog
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    d = DivideShellsDialog(0.01)
    d.n1.setValue(3)
    k, v = d.choice()
    assert k == "equal" and (v["n1"], v["n2"]) == (3, 2)
    d.set_kind("cross")
    d.grids.setChecked(True)
    k, v = d.choice()
    assert k == "cross" and v["frames"] and v["grids"] and not v["stories"] and not v["selected_frames"]


def _cyl(m, rs=(6.0,), ts=(0.0, 90.0)):
    from structstudio.model.objects import GridSystem
    m.grid_systems["C1"] = GridSystem(name="C1", type="Cylindrical")
    for k, r in enumerate(rs):
        m.grid_lines.append(GridLine("C1", f"R{k + 1}", "R", r))
    for k, t in enumerate(ts):
        m.grid_lines.append(GridLine("C1", f"T{k + 1}", "T", t))


def test_arc_grid_cuts_a_floor_and_a_wall():
    """A cylindrical grid's arc cuts a floor along short straight pieces (the curved edge
    is a polyline) and a wall straight up where the arc crosses it."""
    doc = _doc()
    m = doc.model
    _cyl(m, rs=(6.0,), ts=(0.0, 90.0))
    s = _slab(doc, [(0, 0), (10, 0), (10, 10), (0, 10)])
    cuts = SD.planes_at_grids(m, m.areas[s])
    paths = [c for c in cuts if isinstance(c, SD.Path)]
    assert len(paths) == 1  # the arc; the radial lines T1 / T2 lie along the edges
    made = _divide(doc, s, cuts)
    areas = sorted(_area(doc, [n]) for n in made)
    assert len(made) == 2 and areas[0] == pytest.approx(np.pi * 36 / 4, rel=2e-3)
    assert sum(areas) == pytest.approx(100.0)
    # a wall along y = 2 from x = 0 to 10: the arc (r = 6) crosses it at x = sqrt(32)
    w = m.add_area([m.add_point(*p) for p in ((0, 2, 0), (10, 2, 0), (10, 2, 3), (0, 2, 3))], None,
                   type="Wall").name
    made = _divide(doc, w, SD.planes_at_grids(m, m.areas[w]))
    xs = sorted({round(m.points[p].x, 6) for n in made for p in m.areas[n].points})
    assert len(made) == 2 and xs == [0, pytest.approx(np.sqrt(32), abs=1e-6), 10]


def test_which_grids_shown_all_or_chosen():
    """Divide at the grids shown, every grid (hidden ones too) or chosen grid lines -
    for shells and frames alike."""
    from structstudio.app import frame_divide as FD
    doc = _doc()
    m = doc.model
    sy = next(iter(m.grid_systems), "G1")
    for nm, d, o in (("A", "X", 0.0), ("B", "X", 2.0), ("C", "X", 4.0), ("D", "X", 8.0),
                     ("1", "Y", 0.0), ("2", "Y", 6.0)):
        m.grid_lines.append(GridLine(sy, nm, d, o))
    next(g for g in m.grid_lines if g.name == "C").visible = False  # hidden in the grid form
    s = _slab(doc, RECT)
    a = m.areas[s]
    xs = lambda cuts: sorted(round(float(c[0][0]), 6) for c in cuts)
    assert xs(SD.planes_at_grids(m, a)) == [2.0]  # shown: B (C hidden)
    assert xs(SD.planes_at_grids(m, a, grids="all")) == [2.0, 4.0]
    assert xs(SD.planes_at_grids(m, a, grids=[(sy, "C")])) == [4.0]
    assert SD.planes_at_grids(m, a, grids=[]) == []
    f = _beam(doc, (0, 3, 3), (8, 3, 3))
    assert FD.at_grid_lines(m, [f])[f] == pytest.approx([.25])
    assert FD.at_grid_lines(m, [f], grids="all")[f] == pytest.approx([.25, .5])
    assert FD.at_grid_lines(m, [f], grids=[(sy, "C")])[f] == pytest.approx([.5])
    rows = FD.grid_lines_list(m)
    assert (sy, "C", "X", False) in rows and (sy, "B", "X", True) in rows


def test_grid_pick_widget():
    from PySide6 import QtWidgets
    from structstudio.app.divide_dialog import DivideFramesDialog, DivideShellsDialog
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    doc = _doc()
    for D in (DivideFramesDialog, DivideShellsDialog):
        d = D(0.01, model=doc.model)
        d.set_kind("cross")
        d.grids.setChecked(True)
        assert d.choice()[1]["grid_which"] == "shown"
        d.grid_pick.set_value("all")
        assert d.choice()[1]["grid_which"] == "all"
        d.grid_pick.set_value([("G1", "B")])
        assert d.choice()[1]["grid_which"] == [("G1", "B")]
