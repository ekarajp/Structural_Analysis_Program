"""Slab design strips: the model object, the automatic layout (ACI 318 8.4.1.5 widths),
the strip commands (design data: the analysis results stay) and the file round trip."""
import math

import pytest

from structstudio.app.document import Document
from structstudio.app.templates import build_model
from structstudio.design import strips as S
from structstudio.model import Model, ModelError


def _layout(m, story="Story1", **kw):
    lay = S.layout(m, story, S.StripOptions(**kw) if kw else None)
    for n in list(m.design_strips):
        del m.design_strips[n]
    for s in lay.strips:
        m.add_design_strip(**s)
    return lay


def _strips(m, layer, kind):
    return [ds for ds in m.design_strips.values() if ds.layer == layer and ds.kind == kind]


def _covered(m, story="Story1"):
    for L in ("A", "B"):
        c = S.coverage(m, story, L)
        assert c["uncovered"] < 0.005 and c["overlap"] < 0.005, (L, c)


# ------------------------------------------------------------------ model object
def test_strip_object_round_trip_units_and_story():
    m = Model("t", "kN m")
    m.add_story("Base", 0.0)
    m.add_story("S1", 3.0, 3.0)
    ds = m.add_design_strip([(0, 0, 3), (8, 0, 3), (16, 0.5, 3)], (1.0, 1.5))
    assert ds.name == "CSA1" and ds.story == "S1" and len(ds.widths) == 2
    assert ds.widths[0] == (1.0, 1.5, 1.0, 1.5)
    m2 = Model.from_json(m.to_json())
    assert m2.design_strips["CSA1"].points == ds.points and m2.design_strips["CSA1"].widths == ds.widths
    assert m2.hash() == m.hash()
    mm = m.convert_units("kN mm")
    assert mm.design_strips["CSA1"].points[2] == (16000.0, 500.0, 3000.0)
    assert mm.design_strips["CSA1"].widths[0] == (1000.0, 1500.0, 1000.0, 1500.0)
    # names by type and layer; a repeated point is dropped
    ms = m.add_design_strip([(0, 3, 3), (0, 3, 3), (8, 3, 3)], (1, 1), layer="B", kind="Middle")
    assert ms.name == "MSB1" and len(ms.points) == 2
    with pytest.raises(ModelError):
        m.add_design_strip([(0, 0, 3), (1, 0, 4)], (1, 1))  # not at one level
    with pytest.raises(ModelError):
        m.add_design_strip([(0, 0, 3)], (1, 1))
    with pytest.raises(ModelError):
        m.add_design_strip([(0, 0, 3), (1, 0, 3)], (-1, 1))
    with pytest.raises(ModelError):
        m.add_design_strip([(0, 0, 3), (1, 0, 3)], (1, 1), layer="C")


# ------------------------------------------------------------------ layout
@pytest.mark.parametrize("kind", ["flat_slab", "rc_frame", "flat_slab_edge", "steel_deck"])
def test_regular_bays_aci_widths(kind):
    """8 m x 6 m bays: A strips (along X, l1 = 8, l2 = 6) 1.5 each side; B strips (along
    Y, l1 = 6, l2 = 8) 1.5 each side; edge strips reach the slab edge on the outside;
    middle strips fill the rest; nothing left out, nothing twice."""
    m = build_model(structure=kind, x_ordinates=(0, 8, 16, 24), y_ordinates=(0, 6, 12))
    _layout(m)
    ca, cb = _strips(m, "A", "Column"), _strips(m, "B", "Column")
    assert len(ca) == 3 and len(cb) == 4
    for ds in ca:
        y = ds.points[0][1]
        for w in ds.widths:
            left, right = w[0], w[1]
            assert left == pytest.approx(0.0 if y == 12 else 1.5)
            assert right == pytest.approx(0.0 if y == 0 else 1.5)
    for ds in cb:
        for w in ds.widths:
            assert max(w) == pytest.approx(1.5)
    mid_a = sorted(ds.points[0][1] for ds in _strips(m, "A", "Middle"))
    assert mid_a == pytest.approx([3.0, 9.0])
    assert all(w[0] == pytest.approx(1.5) for ds in _strips(m, "A", "Middle") for w in ds.widths)
    assert all(w[0] == pytest.approx(2.5) for ds in _strips(m, "B", "Middle") for w in ds.widths)
    _covered(m)


def test_column_strip_spans_split_at_supports_and_stiff_secondary_beam():
    """A stiff beam across the bays is a line of supports: the column strips crossing it
    get a span of 4 there (0.25 x 4 = 1.0), a column strip of its own."""
    m = build_model(structure="grid", x_ordinates=(0, 8, 16), y_ordinates=(0, 6, 12))
    for x in (0, 8, 16):
        for y in (0, 6, 12):
            m.add_frame((x, y, 0), (x, y, 3.5), "C50x50", type="Column")
    for y in (0, 6, 12):
        m.add_frame((0, y, 3.5), (8, y, 3.5), "B30x60")
        m.add_frame((8, y, 3.5), (16, y, 3.5), "B30x60")
    for x in (0, 4, 8, 16):
        m.add_frame((x, 0, 3.5), (x, 6, 3.5), "B30x60")
        m.add_frame((x, 6, 3.5), (x, 12, 3.5), "B30x60")
    m.add_area([(0, 0, 3.5), (16, 0, 3.5), (16, 12, 3.5), (0, 12, 3.5)], "S20", type="Floor")
    _layout(m)
    xs = sorted(ds.points[0][0] for ds in _strips(m, "B", "Column"))
    assert xs == pytest.approx([0, 4, 8, 16])
    a6 = next(ds for ds in _strips(m, "A", "Column") if ds.points[0][1] == 6)
    assert [round(p[0], 6) for p in a6.points] == [0, 4, 8, 16]
    assert a6.widths[0][0] == pytest.approx(1.0) and a6.widths[2][0] == pytest.approx(1.5)
    _covered(m)


def test_flexible_beam_is_not_a_support():
    """A beam no deeper than the slab is part of the slab (alpha_f small): no strip on it."""
    m = build_model(structure="grid", x_ordinates=(0, 8, 16), y_ordinates=(0, 6))
    from structstudio.model.properties import FrameSection
    m.add_frame_section(FrameSection("BH", "C240", "Rectangular", t3=0.2, t2=0.3, A=0.06, I33=0.0002,
                                     I22=0.00045, J=0.0003))
    for x in (0, 8, 16):
        for y in (0, 6):
            m.add_frame((x, y, 0), (x, y, 3.5), "C50x50", type="Column")
    m.add_frame((4, 0, 3.5), (4, 6, 3.5), "BH")  # a hidden beam mid-bay
    m.add_area([(0, 0, 3.5), (16, 0, 3.5), (16, 6, 3.5), (0, 6, 3.5)], "S20", type="Floor")
    lay = _layout(m)
    assert sorted(ds.points[0][0] for ds in _strips(m, "B", "Column")) == pytest.approx([0, 8, 16])
    assert any("too flexible" in n for n in lay.notes)
    # a lower threshold takes it as a support
    _layout(m, beam_alpha=0.0)
    assert sorted(ds.points[0][0] for ds in _strips(m, "B", "Column")) == pytest.approx([0, 4, 8, 16])


def test_staggered_columns_without_grid_and_cantilever():
    """Columns off line by 0.4 m (no grid): the column strip turns through them; a slab
    running 2 m past the last columns: the strips run on to its edge (cantilever) with
    the edge side reaching the edge."""
    m = build_model(structure="blank", x_ordinates=(0, 8), y_ordinates=(0, 6))
    m.grid_lines.clear()
    for x, y in [(0, 0), (8, 0.4), (16, 0), (0, 6), (8, 6), (16, 6)]:
        m.add_frame((x, y, 0), (x, y, 3.5), "C50x50", type="Column")
    m.add_area([(0, -2, 3.5), (16, -2, 3.5), (16, 6, 3.5), (0, 6, 3.5)], "S20", type="Floor")
    _layout(m)
    a0 = next(ds for ds in _strips(m, "A", "Column") if ds.points[0][1] < 1)
    assert [tuple(round(c, 6) for c in p[:2]) for p in a0.points] == [(0, 0), (8, 0.4), (16, 0)]
    for ds in _strips(m, "B", "Column"):
        assert min(p[1] for p in ds.points) == pytest.approx(-2.0)  # on to the slab edge
    for L in ("A", "B"):
        c = S.coverage(m, "Story1", L)
        assert c["uncovered"] < 0.02 and c["overlap"] < 0.02, (L, c)


def test_walls_slab_on_piles_and_slab_on_ground():
    m = build_model(structure="grid", x_ordinates=(0, 8, 16), y_ordinates=(0, 12))
    for a, b in [((0, 0), (16, 0)), ((16, 0), (16, 12)), ((16, 12), (0, 12)), ((0, 12), (0, 0)), ((8, 0), (8, 12))]:
        m.add_area([(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], 3.5), (a[0], a[1], 3.5)], "W20", type="Wall")
    m.add_area([(0, 0, 3.5), (16, 0, 3.5), (16, 12, 3.5), (0, 12, 3.5)], "S20", type="Floor")
    _layout(m)
    b8 = next(ds for ds in _strips(m, "B", "Column") if ds.points[0][0] == 8)
    assert b8.widths[0][:2] == pytest.approx((2.0, 2.0))  # 0.25 min(l1 = 12, l2 = 8)
    _covered(m)
    # piles: supports of the joints at the slab level
    p = build_model(structure="grid", x_ordinates=(0, 8, 16), y_ordinates=(0, 6, 12))
    for x in (0, 8, 16):
        for y in (0, 6, 12):
            p.add_point(x, y, 3.5, restraints=(True, True, True, False, False, False))
    p.add_area([(-1, -1, 3.5), (17, -1, 3.5), (17, 13, 3.5), (-1, 13, 3.5)], "S20", type="Floor")
    lay = _layout(p)
    assert lay.supports["columns / supports"] == 9 and not lay.warnings
    _covered(p)
    # on ground (no support): nothing made, the reason said
    g = build_model(structure="grid", x_ordinates=(0, 8), y_ordinates=(0, 6))
    g.add_area([(0, 0, 3.5), (8, 0, 3.5), (8, 6, 3.5), (0, 6, 3.5)], "S20", type="Floor")
    lay = _layout(g)
    assert not lay.strips and any("slab on ground" in w for w in lay.warnings)
    lay = _layout(g, source="grids")  # every grid line over the slab
    assert lay.strips


def test_rotated_grid_and_l_shape():
    m = build_model(structure="grid", x_ordinates=(0, 8, 16), y_ordinates=(0, 6, 12))
    m.grid_systems["G1"].rotation = 30.0
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))

    def P(u, v):
        return (u * c - v * s, u * s + v * c)
    for u in (0, 8, 16):
        for v in (0, 6, 12):
            m.add_frame((*P(u, v), 0), (*P(u, v), 3.5), "C50x50", type="Column")
    m.add_area([(*P(0, 0), 3.5), (*P(16, 0), 3.5), (*P(16, 12), 3.5), (*P(0, 12), 3.5)], "S20", type="Floor")
    lay = _layout(m)
    assert any("30 deg" in n for n in lay.notes)
    _covered(m)
    L = build_model(structure="grid", x_ordinates=(0, 8, 16), y_ordinates=(0, 6, 12))
    for x, y in [(0, 0), (8, 0), (16, 0), (0, 6), (8, 6), (16, 6), (0, 12), (8, 12)]:
        L.add_frame((x, y, 0), (x, y, 3.5), "C50x50", type="Column")
    L.add_area([(0, 0, 3.5), (16, 0, 3.5), (16, 6, 3.5), (8, 6, 3.5), (8, 12, 3.5), (0, 12, 3.5)], "S20",
               type="Floor")
    _layout(L)
    a6 = next(ds for ds in _strips(L, "A", "Column") if ds.points[0][1] == 6)
    assert a6.points[-1][0] == pytest.approx(16)  # the edge strip over the re-entrant part
    assert a6.widths[-1][0] == pytest.approx(0.0)  # nothing on that side
    _covered(L)


# ------------------------------------------------------------------ commands
def _doc():
    m = build_model(structure="flat_slab", x_ordinates=(0, 8, 16, 24), y_ordinates=(0, 6, 12))
    return Document(m)


def test_commands_keep_results_and_undo():
    from structstudio.app import strip_commands as SC
    doc = _doc()
    rep = SC.auto_strips(doc, ["Story1", "Story2"])
    assert rep.made == {"Story1": 12, "Story2": 12} and not rep.warnings
    sentinel = object()
    doc.results, doc.locked = sentinel, True
    doc.ask_unlock = lambda: pytest.fail("a strip edit must not ask to unlock the model")
    n = SC.draw_strip(doc, [(2, 0, 3.5), (2, 12, 3.5)], layer="B", kind="Middle")
    assert doc.model.design_strips[n].widths[0][:2] == pytest.approx((0.5, 4.5))  # to the column strips
    assert doc.results is sentinel and doc.locked
    doc.undo()
    assert n not in doc.model.design_strips and doc.results is sentinel
    doc.redo()
    assert n in doc.model.design_strips and doc.results is sentinel
    SC.delete_strips(doc, [n])
    # an Auto run again replaces what Auto made, keeps a hand-drawn strip
    hand = SC.draw_strip(doc, [(4, -0, 3.5), (4, 12, 3.5)], layer="B", kind="Middle", widths="Specified",
                         left=2.5, right=2.5)
    rep = SC.auto_strips(doc, ["Story1"])
    assert hand in doc.model.design_strips
    assert sum(1 for ds in doc.model.design_strips.values() if ds.story == "Story1") == 12  # (one auto skipped)
    assert doc.results is sentinel
    # no change = no undo step
    k = len(doc._undo)
    SC.set_strip(doc, hand)
    assert len(doc._undo) == k


def test_reshape_and_auto_widths():
    from structstudio.app import strip_commands as SC
    doc = _doc()
    SC.auto_strips(doc, ["Story1"])
    name = next(ds.name for ds in doc.model.design_strips.values()
                if ds.kind == "Column" and ds.layer == "A" and ds.points[0][1] == 6)
    SC.reshape_strip(doc, name, 1, (8, 6.4))
    ds = doc.model.design_strips[name]
    assert ds.points[1][:2] == pytest.approx((8, 6.4)) and not ds.auto  # kept by a new Auto run
    SC.reshape_strip(doc, name, 0, (4, 5.0), insert=True)
    assert len(ds.points) == 5 and len(ds.widths) == 4
    with pytest.raises(ModelError):
        SC.reshape_strip(doc, name, 99, (0, 0))
    rep = SC.auto_widths(doc, ["Story1"])
    assert rep.made["Story1"] >= 1
    # the properties form: specified widths stop the automatic ones
    SC.set_strip(doc, name, widths=[(1, 1, 1, 1)] * 4, auto_width=False)
    assert doc.model.design_strips[name].widths[0] == (1, 1, 1, 1)
    new = SC.set_strip(doc, name, new_name="CS-6")
    assert new == "CS-6" and "CS-6" in doc.model.design_strips and name not in doc.model.design_strips
    with pytest.raises(ModelError):
        SC.set_strip(doc, "CS-6", new_name=next(n for n in doc.model.design_strips if n != "CS-6"))


def test_draw_strip_refusals():
    from structstudio.app import strip_commands as SC
    doc = _doc()
    with pytest.raises(ModelError):
        SC.draw_strip(doc, [(0, 0, 3.5)])
    with pytest.raises(ModelError):
        SC.draw_strip(doc, [(0, 0, 3.5), (8, 0, 3.5)], widths="Specified", left=0, right=0)
    with pytest.raises(ModelError):
        SC.draw_strip(doc, [(100, 100, 3.5), (108, 100, 3.5)])  # no slab under it: no width to work out
    with pytest.raises(ModelError):
        SC.draw_strip(doc, [(0, 0, 1.0), (8, 0, 1.0)], widths="Specified")  # not at a story
    assert not doc.model.design_strips and not doc.can_undo


def test_slab_design_model_takes_the_floors_strips():
    from structstudio.app import strip_commands as SC
    from structstudio.design.slab_model import build_slab_model
    m = build_model(structure="rc_frame", x_ordinates=(0, 8, 16), y_ordinates=(0, 6))
    for sec in m.shell_sections.values():
        if sec.type == "Slab":
            sec.model_type = "Membrane"
    doc = Document(m)
    SC.auto_strips(doc, ["Story1", "Story2"])
    res = build_slab_model(doc.model, "Story1")
    z = doc.model.stories["Story1"].elevation
    assert res.model.design_strips
    assert all(abs(ds.points[0][2] - z) < 1e-9 for ds in res.model.design_strips.values())


def test_cylindrical_grid_rings():
    """Columns on rings of a cylindrical grid, a quarter-annulus slab: circumferential
    strips along the rings (layer A), radial strips along the radial lines (layer B);
    l1 along the inner ring is its arc between columns (pi x 6 / 6 = 3.14 -> 0.79)."""
    from structstudio.model.objects import GridLine, GridSystem
    m = build_model(structure="blank", x_ordinates=(0, 8), y_ordinates=(0, 6))
    m.grid_systems["C1"] = GridSystem(name="C1", type="Cylindrical")
    for r in (6, 12, 18):
        m.grid_lines.append(GridLine(system="C1", name=f"R{r}", direction="R", ordinate=r))
    for k, t in enumerate((0, 30, 60, 90)):
        m.grid_lines.append(GridLine(system="C1", name=f"T{k}", direction="T", ordinate=t))
    for r in (6, 12, 18):
        for t in (0, 30, 60, 90):
            x, y = r * math.cos(math.radians(t)), r * math.sin(math.radians(t))
            m.add_frame((x, y, 0), (x, y, 3.5), "C50x50", type="Column")
    outer = [(18 * math.cos(math.radians(a)), 18 * math.sin(math.radians(a)), 3.5) for a in range(0, 91, 5)]
    inner = [(6 * math.cos(math.radians(a)), 6 * math.sin(math.radians(a)), 3.5) for a in range(90, -1, -5)]
    m.add_area(outer + inner, "S20", type="Floor")
    _layout(m)
    rings = sorted(round(math.hypot(*ds.points[1][:2]), 1) for ds in _strips(m, "A", "Column"))
    assert rings == [6.0, 12.0, 18.0]
    inner_strip = next(ds for ds in _strips(m, "A", "Column") if abs(math.hypot(*ds.points[1][:2]) - 6) < 0.1)
    assert max(w[1] for w in inner_strip.widths) == pytest.approx(0.25 * math.pi, abs=0.02)
    assert len(_strips(m, "B", "Column")) == 4
    for L in ("A", "B"):
        c = S.coverage(m, "Story1", L)
        assert c["uncovered"] < 0.02 and c["overlap"] < 0.02, (L, c)
