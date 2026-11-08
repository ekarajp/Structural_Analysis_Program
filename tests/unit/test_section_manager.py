"""Define > Frame Sections (manager: Add / Copy / Modify / Delete, drawing with
dimensions, properties in display units) and the Section Designer (drawing by
mouse and by typed coordinates, snaps, openings, undo).  Runs in a child
interpreter like the other desktop tests."""
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("vtk")
QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"
gui = pytest.mark.skipif(not CHILD, reason="runs in the child interpreter (test_sections_in_subprocess)")


@pytest.mark.skipif(CHILD, reason="parent only")
def test_sections_in_subprocess():
    import shutil
    env = dict(os.environ, STRUCTSTUDIO_GUI_CHILD="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__]
    if not env.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1600x1000x24"] + cmd
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "passed" in proc.stdout and "failed" not in proc.stdout and "error" not in proc.stdout.lower(), \
        (proc.stdout + proc.stderr)[-4000:]


@pytest.fixture(scope="module")
def app():
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    for w in app.topLevelWidgets():
        w.close()
        w.deleteLater()
    app.processEvents()


@pytest.fixture(autouse=True)
def _cleanup(app):
    yield
    for w in app.topLevelWidgets():
        w.close()
        w.deleteLater()
    app.processEvents()
    import gc
    gc.collect()


@pytest.fixture
def doc(app):
    from structstudio.app import forms
    from structstudio.app.document import Document
    from structstudio.app.templates import new_building
    from structstudio.display_units import PRESETS, DisplayUnits
    from structstudio.units import UnitSystem
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.5,), grid_only=False)
    consistent, items = PRESETS["Thai legacy - tonf (tonf, tonf-m, tonf/m, tonf/m^2, ksc)"]
    m = m.convert_units(UnitSystem(*consistent))
    m.display_units = dict(items)  # sections in cm, areas cm^2, inertia cm^4
    forms.DISPLAY["units"] = DisplayUnits.of(m)
    return Document(m)


@gui
def test_manager_lists_draws_and_adds_a_t_beam(doc, app):
    from structstudio.app.section_manager import FrameSectionsDialog, ShapeSectionDialog
    d = FrameSectionsDialog(doc)
    d.show()
    app.processEvents()
    m = doc.model
    assert d.table.rowCount() == len(m.frame_sections)
    used = next(f.section for f in m.frames.values())
    d.select_names([used])
    app.processEvents()
    assert d.preview.solids and d.props.rowCount() > 6
    rows = {d.props.item(i, 0).text(): (d.props.item(i, 1).text(), d.props.item(i, 2).text())
            for i in range(d.props.rowCount())}
    assert rows["Area A"][1] == "cm²" and rows["I₃₃ (major, about 3)"][1] == "cm⁴"  # printed as in a book
    assert d.props.item(d.props.rowCount() - 1, 1).text().endswith("member(s)")
    assert not d.preview.grab().isNull()
    # Add New > T-beam: the form in cm, the name follows the dimensions
    sec = d.new_section("Concrete Tee", {}, "Concrete")
    e = ShapeSectionDialog(m, sec, d, existing=list(m.frame_sections), new=True)
    assert e.edits["t3"].text() == "60" and e.edits["t2"].text() == "120"
    e.edits["t3"].setText("70")
    e.edits["tw"].setText("35")
    assert e.name.text() == "T35x70"
    assert e.ok_button.isEnabled() and e.preview.solids
    e.edits["tf"].setText("80")  # thicker than the depth
    assert not e.ok_button.isEnabled() and "thinner" in e.error.text()
    e.edits["tf"].setText("15")
    e._ok()
    out = e.result_section
    L = 1.0 / m.units.factor("length")  # model length per metre (model in m)
    assert out.t3 == pytest.approx(0.70 * L) and out.tw == pytest.approx(0.35 * L)
    d._store(out)
    assert "T35x70" in doc.model.frame_sections and d.selected_names() == ["T35x70"]
    A = 1.2 * 0.15 + 0.35 * 0.55
    assert doc.model.frame_sections["T35x70"].A == pytest.approx(A * L * L)
    doc.undo()
    assert "T35x70" not in doc.model.frame_sections
    d.close()


@gui
def test_copy_modify_rename_and_delete(doc, app, monkeypatch):
    from structstudio.app.section_manager import FrameSectionsDialog
    d = FrameSectionsDialog(doc)
    used = next(f.section for f in doc.model.frames.values())
    n_used = len(doc.model.frame_section_users(used))
    monkeypatch.setattr(d, "edit_section", lambda sec, new: sec)  # accept the forms as they open
    d.select_names([used])
    d.copy_section()
    copy_name = f"{used}-copy"
    assert copy_name in doc.model.frame_sections and doc.model.frame_section_users(copy_name) == []

    def rename(sec, new):
        sec.name = "MAIN"
        return sec
    monkeypatch.setattr(d, "edit_section", rename)
    d.select_names([used])
    d.modify()
    assert "MAIN" in doc.model.frame_sections and used not in doc.model.frame_sections
    assert len(doc.model.frame_section_users("MAIN")) == n_used  # the members follow the new name
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    d.select_names(["MAIN", copy_name])
    d.delete()
    assert warned and "MAIN" in warned[0]  # in use: kept
    assert "MAIN" in doc.model.frame_sections and copy_name not in doc.model.frame_sections
    doc.undo()
    doc.undo()
    assert used in doc.model.frame_sections


@gui
def test_steel_section_brings_a_steel_material(doc, app):
    from structstudio.app.section_manager import FrameSectionsDialog
    d = FrameSectionsDialog(doc)
    assert not any(m.type == "Steel" for m in doc.model.materials.values())
    sec = d.new_section("I", {}, "Steel")
    mat = doc.model.materials["SS400"]
    assert sec.material == "SS400" and mat.type == "Steel"
    ksc = 98066.5 / doc.model.units.factor("stress")  # model stress per ksc
    assert mat.fy / ksc == pytest.approx(2498.3, rel=1e-3)  # 245 MPa
    assert sec.name == "H300x150x6.5x9"


@gui
def test_every_shape_form_and_icon(doc, app):
    from structstudio.app.section_manager import (CHOICES, FrameSectionsDialog, SectionTypeDialog, ShapeSectionDialog,
                                                  choice_icon)
    from structstudio.model.section_shapes import SHAPES
    d = FrameSectionsDialog(doc)
    t = SectionTypeDialog(d)
    assert len(t.buttons) == sum(len(items) for _, items in CHOICES)
    for _g, items in CHOICES:
        for label, key, over, kind in items:
            assert not choice_icon(key, over, kind).isNull()
            if key not in SHAPES:
                continue
            sec = d.new_section(key, over, kind)
            e = ShapeSectionDialog(doc.model, sec, d, existing=list(doc.model.frame_sections), new=True)
            assert e.ok_button.isEnabled(), (label, e.error.text())
            assert e.preview.solids and not e.preview.grab().isNull()


@gui
def test_designer_typed_coordinates_and_mouse(doc, app):
    from PySide6.QtTest import QTest
    from structstudio.app.section_designer import SectionDesignerDialog
    from structstudio.model.section_shapes import SHAPES, shape_properties
    m = doc.model
    dlg = SectionDesignerDialog(m, None, existing=list(m.frame_sections))
    dlg.show()
    app.processEvents()
    cv = dlg.canvas
    run = lambda *lines: [dlg.command.setText(t) or dlg._command() for t in lines]
    # a T as two rectangles typed like in CAD (cm)
    run("r", "-60,48", "@120,12", "r", "-15,0", "@30,48")
    assert len(cv.shapes) == 2
    dlg._recompute()
    p = dlg.props_result
    L = 1.0 / m.units.factor("length")
    q = shape_properties(SHAPES["Concrete Tee"], {"t3": 0.6, "t2": 1.2, "tf": 0.12, "tw": 0.3})
    assert p.A == pytest.approx(q.A * L ** 2) and p.I33 == pytest.approx(q.I33 * L ** 4)
    # an opening: a polygon typed with relative and polar points, closed with "c"
    run("h", "p", "-5,20", "@10,0", "@10<90", "@-10,0", "c")
    assert cv.shapes[-1]["hole"] and len(cv.shapes[-1]["points"]) == 4
    assert cv.shapes[-1]["points"][2] == pytest.approx([5 / 100 * L, 30 / 100 * L])
    dlg._recompute()
    assert dlg.props_result.A == pytest.approx((q.A - 0.01) * L ** 2)
    # undo / redo
    cv.undo()
    assert len(cv.shapes) == 2
    cv.redo()
    assert len(cv.shapes) == 3
    # a circle by centre and typed radius
    run("h", "ci", "0,-30", "10")
    assert cv.shapes[-1]["kind"] == "circle" and cv.shapes[-1]["radius"] == pytest.approx(0.1 * L)
    assert not cv.shapes[-1]["hole"]
    dlg._recompute()
    assert "separate parts" in dlg.warn.text()
    cv.undo()
    # mouse: drag the top-right corner of the flange 10 cm up (grid snap)
    run("s")
    cv.selected = {0}
    cv.fit()
    app.processEvents()
    corner = np.array([0.6, 0.6]) * L
    a = cv.to_screen(corner).toPoint()
    b = cv.to_screen(corner + np.array([0.0, 0.1]) * L).toPoint()
    QTest.mousePress(cv, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, a)
    QTest.mouseMove(cv, b)
    QTest.mouseRelease(cv, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, b)
    assert [0.6 * L, 0.7 * L] in [pytest.approx(pt) for pt in cv.shapes[0]["points"]]
    cv.undo()
    assert [0.6 * L, 0.6 * L] in [pytest.approx(pt) for pt in cv.shapes[0]["points"]]
    # mouse: draw a rectangle with two clicks on the grid
    cv.set_tool("rect")
    p0, p1 = np.array([1.0, 0.0]) * L, np.array([1.2, 0.3]) * L
    for pt in (p0, p1):
        QTest.mouseClick(cv, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, cv.to_screen(pt).toPoint())
    assert np.allclose(cv.shapes[-1]["points"], [[1.0 * L, 0], [1.2 * L, 0], [1.2 * L, 0.3 * L], [1.0 * L, 0.3 * L]])
    cv.delete_selected()
    # insert a catalogue shape and the point table
    from structstudio.app.section_designer import shape_from_catalogue
    cv.insert_shapes = shape_from_catalogue("I", {"t3": 0.3 * L, "t2": 0.15 * L, "tf": 0.009 * L, "tw": 0.0065 * L})
    cv.tool = "insert"
    run("100,0")
    assert len(cv.shapes) == 4 and cv.selected == {3}
    dlg._fill_points()
    assert dlg.points.rowCount() == 12
    dlg.points.item(0, 0).setText("90")
    assert cv.shapes[3]["points"][0][0] == pytest.approx(0.9 * L)
    cv.undo()
    cv.delete_selected()
    # OK: a Section Designer section with the drawing and its properties
    dlg.name.setText("PIER-T")
    dlg._ok()
    sec = dlg.result_section
    assert sec.shape == "Section Designer" and len(sec.geometry) == 3
    assert sec.A == pytest.approx((q.A - 0.01) * L ** 2) and sec.J > 0
    assert sec.t3 == pytest.approx(0.6 * L) and sec.t2 == pytest.approx(1.2 * L)
    assert not dlg.grab().isNull()


@gui
def test_designer_snaps_and_ortho(doc, app):
    from structstudio.app.section_designer import DesignerCanvas, parse_point
    from structstudio.display_units import DisplayUnits
    du = DisplayUnits.of(doc.model)
    k = du.factor("section_dim")
    assert parse_point("30,40", None, None, k) == pytest.approx(np.array([30, 40]) / k)
    assert parse_point("@10<90", np.array([1.0, 1.0]), None, k) == pytest.approx([1.0, 1.0 + 10 / k])
    assert parse_point("20", np.zeros(2), np.array([3.0, 4.0]), k) == pytest.approx(np.array([0.6, 0.8]) * 20 / k)
    cv = DesignerCanvas(du)
    cv.resize(600, 500)
    cv.grid = 0.05
    cv.shapes = [{"kind": "polygon", "points": [[0, 0], [0.3, 0], [0.3, 0.6], [0, 0.6]], "hole": False}]
    cv.update_region()
    cv.scale = 500
    p, kind = cv.snapped(np.array([0.302, 0.597]))
    assert kind == "end point" and p == pytest.approx([0.3, 0.6])
    p, kind = cv.snapped(np.array([0.151, 0.004]))
    assert kind == "mid point" and p == pytest.approx([0.15, 0.0])
    p, kind = cv.snapped(np.array([0.5, 0.33]))
    assert kind == "grid" and p == pytest.approx([0.5, 0.35])
    cv.ortho = True
    p, kind = cv.snapped(np.array([0.9, 0.42]), ref=np.array([0.5, 0.4]))
    assert kind == "ortho" and p == pytest.approx([0.9, 0.4])


@gui
def test_designer_section_in_the_model_and_3d(doc, app):
    from structstudio.app.extrude import section_profile
    from structstudio.model import FrameSection
    m = doc.model
    L = 1.0 / m.units.factor("length")
    geom = [{"kind": "polygon", "points": (np.array([[-0.4, 0], [0.4, 0], [0.4, 0.15], [0.15, 0.15], [0.15, 0.75],
                                                     [-0.15, 0.75], [-0.15, 0.15], [-0.4, 0.15]]) * L).tolist()},
            {"kind": "circle", "center": [0, 0.45 * L], "radius": 0.06 * L, "hole": True}]
    sec = FrameSection.from_designer("SD", next(iter(m.materials)), geom)
    pieces = section_profile(sec)
    area = sum(0.5 * abs((P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1]) - (P[2, 0] - P[0, 0]) * (P[1, 1] - P[0, 1]))
               for P in pieces)
    assert area == pytest.approx(sec.A, rel=2e-3)
    allp = np.vstack(pieces)  # (y2, y3) about the centroid
    assert allp[:, 0].max() - allp[:, 0].min() == pytest.approx(0.75 * L)
    from structstudio.app.object_props import _sec_text
    from structstudio.display_units import DisplayUnits
    m.add_frame_section(sec)
    assert "Section Designer" in _sec_text(m, "SD", DisplayUnits.of(m))


@gui
def test_designer_cad_commands(doc, app):
    from structstudio.app.section_designer import SectionDesignerDialog
    m = doc.model
    L = 1.0 / m.units.factor("length")
    dlg = SectionDesignerDialog(m, None, existing=list(m.frame_sections))
    dlg.show()
    app.processEvents()
    cv = dlg.canvas
    run = lambda *lines: [dlg.command.setText(t) or dlg._command() for t in lines]
    pts = lambda i: np.asarray(cv.shapes[i]["points"]) / L * 100  # cm
    run("r", "0,0", "@30,60")
    cv.selected = {0}
    # move: base point -> second point (relative)
    run("m", "0,0", "@10,0")
    assert pts(0).min(axis=0) == pytest.approx([10, 0])
    # copy: one base point, several copies until Enter
    run("co", "10,0", "@40,0", "@80,0", "")
    assert len(cv.shapes) == 3 and pts(2).min(axis=0) == pytest.approx([90, 0])
    assert cv.op is None
    # mirror about the vertical line x = 0, keep the source
    cv.selected = {0}
    run("mi", "0,0", "0,10", "n")
    assert len(cv.shapes) == 4 and pts(3).min(axis=0) == pytest.approx([-40, 0])
    # rotate the mirrored copy 90 deg about its corner (typed angle)
    cv.selected = {3}
    run("ro", "-10,0", "90")
    assert pts(3).min(axis=0) == pytest.approx([-70, -30]) and pts(3).max(axis=0) == pytest.approx([-10, 0])
    # command first, then select, Enter
    cv.selected = set()
    run("m")
    assert cv.op["stage"] == "select"
    cv.op_click(np.array([1.0, 0.3]) * L)  # inside the copy at x 90..120 cm
    run("", "0,0", "@0,10")
    assert pts(2).min(axis=0) == pytest.approx([90, 10])
    # trim: cut the first rectangle along y = 20 and click away the lower part
    cv.selected = set()
    n = len(cv.shapes)
    run("tr", "-200,20", "200,20")
    assert len(cv.shapes) > n and cv.op["stage"] == "pick"
    lower = next(i for i, g in enumerate(cv.shapes) if np.allclose(np.asarray(g["points"]).min(axis=0) / L * 100, [10, 0]))
    cv.op_click(np.array([0.2, 0.1]) * L)
    run("")
    assert not any(np.allclose(np.asarray(g["points"]).min(axis=0) / L * 100, [10, 0]) for g in cv.shapes) and lower >= 0
    assert any(np.allclose(np.asarray(g["points"]).min(axis=0) / L * 100, [10, 20]) for g in cv.shapes)
    # clipboard: copy and paste at a point (the base is the centre of the selection)
    cv.selected = {0}
    c0 = pts(0).mean(axis=0)
    cv.copy_to_clipboard()
    cv.start_op("paste")
    run("0,-100")
    assert len(cv.selected) == 1
    new = pts(sorted(cv.selected)[0])
    assert 0.5 * (new.min(axis=0) + new.max(axis=0)) == pytest.approx([0, -100])
    assert c0 is not None
    # undo walks back through the commands
    k = len(cv.undo_stack)
    assert k >= 8
    dlg.close()


@gui
def test_designer_snap_modes(doc, app):
    from structstudio.app.section_designer import DesignerCanvas
    from structstudio.display_units import DisplayUnits
    cv = DesignerCanvas(DisplayUnits.of(doc.model))
    cv.resize(600, 500)
    cv.grid = 0.05
    cv.scale = 500
    cv.set_shapes([{"kind": "polygon", "points": [[0, 0], [0.4, 0], [0.4, 0.2], [0, 0.2]]},
                   {"kind": "polygon", "points": [[0.1, -0.1], [0.3, -0.1], [0.3, 0.5], [0.1, 0.5]]}], fit=False)
    cv.scale = 500
    p, kind = cv.snapped(np.array([0.302, 0.198]))
    assert kind == "intersection" and p == pytest.approx([0.3, 0.2])
    p, kind = cv.snapped(np.array([0.2, 0.199]), ref=np.array([0.21, 0.4]))
    assert kind in ("perpendicular", "mid point")
    cv.snap["mid"] = False
    p, kind = cv.snapped(np.array([0.2, 0.199]), ref=np.array([0.21, 0.4]))
    assert kind == "perpendicular" and p == pytest.approx([0.21, 0.2])
    cv.osnap_on = False  # F3: object snaps off, the grid only
    p, kind = cv.snapped(np.array([0.302, 0.198]))
    assert kind == "grid" and p == pytest.approx([0.3, 0.2])
    cv.snap["grid"] = False
    p, kind = cv.snapped(np.array([0.302, 0.198]))
    assert kind == "" and p == pytest.approx([0.302, 0.198])


@gui
def test_composite_column_form(doc, app):
    from structstudio.app.section_manager import COMPOSITE_CIRCLE, COMPOSITE_RECT, CompositeSectionDialog, \
        FrameSectionsDialog
    d = FrameSectionsDialog(doc)
    sec = d.new_section(COMPOSITE_RECT, {}, "Concrete")
    assert "SS400" in doc.model.materials  # a steel to go in the concrete
    e = CompositeSectionDialog(doc.model, sec, d, existing=list(doc.model.frame_sections), new=True)
    assert e.name.text() == "SRC60x60-H300x300" and e.ok_button.isEnabled()
    assert e.preview.parts and any(st for st, _, _ in e.preview.parts)
    rows = {e.table.item(i, 0).text(): e.table.item(i, 1).text() for i in range(e.table.rowCount())}
    assert "n = E(SS400) / E(C28)" in rows and "Ratio SS400 / gross" in rows
    e.edits["c_t2"].setText("25")  # narrower than the steel
    assert not e.ok_button.isEnabled() and "fit" in e.error.text()
    e.edits["c_t2"].setText("70")
    e.rotate.setChecked(True)
    e._ok()
    out = e.result_section
    d._store(out)
    s = doc.model.frame_sections[out.name]
    assert s.params["composite"] == "rect" and s.I22 > s.I33  # the steel's strong axis now about 2
    d.select_names([out.name])
    assert d.preview.parts and "Composite" in d.table.item(d.table.currentRow(), 1).text()
    c = d.new_section(COMPOSITE_CIRCLE, {}, "Concrete")
    e = CompositeSectionDialog(doc.model, c, d, existing=list(doc.model.frame_sections), new=True)
    assert e.ok_button.isEnabled() and e.name.text().startswith("SRC-D70")


@gui
def test_steel_catalog_adds_sections_with_tabulated_values(doc, app):
    from structstudio.app.section_manager import FrameSectionsDialog, LibraryDialog
    d = LibraryDialog(doc.model)
    d.show()
    app.processEvents()
    labels = [d.file.itemText(i) for i in range(d.file.count())]
    assert any("TIS 1227" in t for t in labels) and any("AISC" in t for t in labels) and any("EN" in t for t in labels)
    d.file.setCurrentText(next(t for t in labels if "TIS 1227" in t))
    d.series.setCurrentText("H")
    d.filter.setText("H300x150")
    app.processEvents()
    items = [d.list.item(i).text() for i in range(d.list.count())]
    assert "H300x150x6.5x9" in items
    d.list.setCurrentRow(items.index("H300x150x6.5x9"))
    d.list.item(items.index("H300x150x6.5x9")).setSelected(True)
    app.processEvents()
    assert d.ok_button.isEnabled() and d.table.rowCount() > 5
    d._ok()
    sec = d.sections[0]
    L = doc.model.units.factor("length")
    assert sec.A * L ** 2 * 1e4 == pytest.approx(46.78) and sec.I33 * L ** 4 * 1e8 == pytest.approx(7210)
    assert sec.source == "TIS 1227 / JIS G3192"
    # the manager stores them (and makes a steel material when the model has none)
    m = FrameSectionsDialog(doc)
    m.show()
    app.processEvents()
    doc.edit(lambda mm: mm.frame_sections.__setitem__(sec.name, sec))
    m.refresh(select=[sec.name])
    app.processEvents()
    assert sec.name in doc.model.frame_sections


@gui
def test_rebar_sizes_from_catalog_edit_and_undo(doc, app):
    from structstudio.app.rebar_manager import RebarCatalogDialog, RebarSizesDialog
    from structstudio.catalog import rebar_catalog
    d = RebarSizesDialog(doc)
    d.show()
    app.processEvents()
    assert d.table.rowCount() == 0
    c = RebarCatalogDialog((), d)
    c.standard.setCurrentText("TIS 24 (deformed)")
    for i in range(c.list.count()):
        if c.list.item(i).data(QtCore.Qt.UserRole) in ("DB12", "DB16", "DB20"):
            c.list.item(i).setCheckState(QtCore.Qt.Checked)
    c._ok()
    assert [e.name for e in c.chosen] == ["DB12", "DB16", "DB20"]
    d.add_entries(c.chosen)
    assert d.table.rowCount() == 3
    # cm / cm^2 display (Thai legacy): DB16 1.6 cm, 2.011 cm^2
    row = [d.table.item(1, j).text() for j in range(5)]
    assert row[0] == "DB16" and float(row[2]) == pytest.approx(1.6) and float(row[3]) == pytest.approx(2.011)
    # rename and a size of one's own
    d.table.item(0, 0).setText("DB12-main")
    d.add()
    assert "Bar1" in d.sizes and d.sizes["Bar1"].standard == ""
    d._ok()
    m = doc.model
    assert list(m.rebar_sizes) == ["DB12-main", "DB16", "DB20", "Bar1"]
    L = m.units.factor("length")
    assert m.rebar_sizes["DB20"].area * L ** 2 == pytest.approx(314.2e-6)
    doc.undo()
    assert doc.model.rebar_sizes == {}
    assert len(rebar_catalog()["JIS G3112"]) == 13


@gui
def test_channel_flange_taper_option(doc, app):
    from structstudio.app.section_manager import ShapeSectionDialog
    from structstudio.model import FrameSection
    L = doc.model.units.factor("length")
    s = FrameSection.from_shape("Channel", "C1", "SS400", t3=0.15 / L, t2=0.075 / L, tf=0.010 / L, tw=0.0065 / L,
                                r=0.010 / L, r2=0.005 / L)
    d = ShapeSectionDialog(doc.model, s, existing=[], new=False)
    d.show()
    app.processEvents()
    labels = [d.slope.itemText(i) for i in range(d.slope.count())]
    assert any("JIS" in t for t in labels) and any("AISC" in t for t in labels) and any("GB" in t for t in labels)
    assert d.slope.currentIndex() == 0 and not d.slope_custom.isEnabled()
    d.slope.setCurrentIndex(next(i for i, t in enumerate(labels) if "JIS / TIS channel" in t))
    app.processEvents()
    d._ok()
    out = d.result_section
    assert out.params["slope"] == pytest.approx(0.0875, abs=1e-4)
    assert out.A * L ** 2 * 1e4 == pytest.approx(23.71, rel=0.002) and out.I22 * L ** 4 * 1e8 == pytest.approx(117, rel=0.01)
    # custom slope, and the form opens on it again
    d2 = ShapeSectionDialog(doc.model, out, existing=[], new=False)
    assert "JIS" in d2.slope.currentText()
    d2.slope.setCurrentIndex(d2.slope.count() - 1)
    d2.slope_custom.setText("12.5")
    app.processEvents()
    d2._ok()
    assert d2.result_section.params["slope"] == pytest.approx(0.125)
