"""Area loads are load panels made from the areas (PanelLoad.area).

Checks: identical analysis to the loads stored on the areas, migration of older
files and imports, the panel follows edits (move, replicate, divide, delete,
copy / paste assigns), one-way / two-way area panels on frames."""
import copy

import numpy as np
import pytest

from structstudio import Model, analyze
from structstudio.app.document import Document
from structstudio.model import FIXED, FrameSection, ShellSection
from structstudio.model.objects import AreaLoad, PanelLoadCase


def _slab_model(attached: bool = False) -> Model:
    m = Model("t", "kN m")
    m.add_material("C", E=2.5e7, nu=0.2, weight_density=0.0)
    m.add_frame_section(FrameSection.general("B", "C", A=0.12, I33=1e-3, I22=1e-3, J=1e-4))
    m.add_shell_section(ShellSection("S20", "C", thickness=0.2))
    base = [m.add_point(x, y, 0, restraints=FIXED) for x, y in ((0, 0), (6, 0), (6, 5), (0, 5))]
    top = [m.add_point(x, y, 3) for x, y in ((0, 0), (6, 0), (6, 5), (0, 5))]
    for a, b in zip(base, top):
        m.add_frame(a, b, "B", type="Column")
    for i in range(4):
        m.add_frame(top[i], top[(i + 1) % 4], "B", type="Beam", name=f"BM{i}")
    m.add_area(top, "S20", name="F1")
    m.add_load_pattern("SDL", "SuperDead")
    m.add_load_pattern("W", "Wind")
    if attached:  # the old storage: loads on the area object
        m.areas["F1"].loads += [AreaLoad("SDL", 2.0, "Gravity"), AreaLoad("SDL", 1.5, "Gravity"),
                                AreaLoad("W", 0.8, "X")]
    else:
        m.add_area_load("F1", "SDL", 2.0)
        m.add_area_load("F1", "SDL", 1.5)
        m.add_area_load("F1", "W", 0.8, direction="X")
    m.add_linear_static_case("SDL")
    m.add_linear_static_case("W")
    return m


def test_area_load_is_a_panel_and_gives_the_same_analysis():
    m = _slab_model()
    assert not m.areas["F1"].loads
    panels = [p for p in m.panel_loads.values() if p.area == "F1"]
    assert len(panels) == 2 and all(p.distribution == "shells" for p in panels)  # one per pattern
    assert [c.value for c in m.area_panel("F1", "SDL").loads] == [2.0, 1.5]
    old = analyze(_slab_model(attached=True))  # attached loads are still honoured by the mesher
    new = analyze(m)
    for case in ("SDL", "W"):
        assert np.allclose(new.base_reactions(case), old.base_reactions(case), atol=1e-9)
    assert new.base_reactions("SDL")[2] == pytest.approx(3.5 * 30.0, rel=1e-9)
    rep = new.am.panel_report
    assert sum(r["SDL"][0] for r in rep.values() if "SDL" in r) == pytest.approx(105.0)


def test_older_files_and_scripts_convert_to_panels():
    m = _slab_model(attached=True)
    d = m.to_dict()
    d["schema_version"] = "2.0"  # a file written before area loads were panels
    m2 = Model.from_dict(d)
    assert not m2.areas["F1"].loads and len(m2.panel_loads) == 2
    assert np.allclose(analyze(m2).base_reactions("SDL"), analyze(m).base_reactions("SDL"))
    m3 = Model.from_dict(Model.from_dict(d).to_dict())  # current files are not converted twice
    assert len(m3.panel_loads) == 2
    assert m3._counters.get("PL", 0) >= 2 and m3.add_panel_load([(0, 0, 0), (1, 0, 0), (1, 1, 0)]).name == "PL3"
    # unit change keeps the polygon in step (panel corners are lengths)
    mm = m3.convert_units("kN mm")
    assert mm.panel_loads["PL1"].points[1][0] == pytest.approx(6000.0)


def test_area_panels_follow_edits():
    from structstudio.app import commands as C
    from structstudio.app.tools import delete_objects, move_objects, replicate_linear
    doc = Document()
    doc.model = _slab_model()
    m = doc.model
    move_objects(doc, [], ["F1"], [], 0, 0, 1.0)  # the area moves: its panels follow
    assert all(p.points[0][2] == pytest.approx(4.0) for p in doc.model.panel_loads.values())
    # divided floor: the panel stays on its pieces (no load lost)
    total = analyze(doc.model).base_reactions("SDL")[2]
    assert total == pytest.approx(105.0)
    C.divide_shells(doc, ["F1"], 2, 2)
    assert len(doc.model.areas) == 4
    assert all(p.area in doc.model.areas for p in doc.model.panel_loads.values())  # no free panel made
    assert len(doc.model.panel_loads) == 8  # 4 pieces x 2 patterns
    assert analyze(doc.model).base_reactions("SDL")[2] == pytest.approx(total, rel=1e-9)
    # a drawn panel is a load area of its own: it does not move with the floors
    from structstudio.app.tools import move_panels
    free = doc.model.add_panel_load([(0, 0, 4), (6, 0, 4), (6, 5, 4), (0, 5, 4)], "two_way",
                                    loads=[PanelLoadCase("SDL", 1.0)]).name
    pieces = list(doc.model.areas)
    move_objects(doc, [], pieces, [], 0, 0, 1.0)
    assert doc.model.panel_loads[free].points[0] == (0, 0, 4)
    delete_objects(doc, areas=pieces[:1])
    assert free in doc.model.panel_loads  # deleting a floor never deletes a drawn panel
    assert move_panels(doc, [free], 1.0, 0, 0) == 1 and doc.model.panel_loads[free].points[0][0] == 1.0
    # replicated areas get their own panels
    piece = next(a for a in doc.model.areas if doc.model.area_panel(a) is not None)
    doc.model.add_area_load(piece, "W", 1.0, "X")
    new = replicate_linear(doc, [], [piece], [], 10, 0, 0, count=1)
    assert len(new) == 1 and doc.model.area_panel(new[0], "W") is not None
    assert doc.model.area_panel(new[0], "W").points[0][0] >= 10.0
    # deleting an area deletes its loads
    delete_objects(doc, areas=new)
    assert all(p.area != new[0] for p in doc.model.panel_loads.values())
    # undo restores them
    doc.undo()
    assert doc.model.area_panel(new[0], "W") is not None


def test_copy_paste_assigns_and_clipboard_carry_area_loads():
    from structstudio.app import commands as C
    doc = Document()
    doc.model = _slab_model()
    m = doc.model
    pts = [m.add_point(x, y, 3) for x, y in ((6, 0), (12, 0), (12, 5), (6, 5))]
    m.add_area(pts, "S20", name="F2")
    clip = C.copy_assigns(m, "area", "F1")
    C.paste_assigns(doc, clip, ["F2"])
    assert sorted(c.value for p in doc.model.panel_loads.values() if p.area == "F2" for c in p.loads) == [0.8, 1.5, 2.0]
    C.paste_assigns(doc, clip, ["F2"])  # paste replaces, does not add
    assert len([p for p in doc.model.panel_loads.values() if p.area == "F2"]) == 2
    cb = C.copy_objects(doc.model, [], ["F1"], [])
    sel = C.paste_objects(doc, cb, 0, 20, 0)
    assert len([p for p in doc.model.panel_loads.values() if p.area == sel[1][0]]) == 2


def test_area_panel_two_way_on_frames_and_on_a_null_area():
    m = _slab_model()
    m.panel_loads.clear()
    m.areas["F1"].section = None  # a load-only (null) area: its panel carries the load to the beams
    m.areas["F1"].type = "Null"
    pn = m.add_area_panel("F1", [PanelLoadCase("SDL", 10.0)], distribution="two_way").name
    r = analyze(m)
    assert r.base_reactions("SDL")[2] == pytest.approx(300.0, rel=5e-3)
    rep = r.am.panel_report[pn]["SDL"]
    assert rep[0] == pytest.approx(300.0) and rep[1] == pytest.approx(300.0, rel=5e-3)
