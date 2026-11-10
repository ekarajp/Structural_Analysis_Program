"""Regression tests for the UX audit of 2026-11-10 (UX01-UX06).

Every test here fails on the audited snapshot (= commit 0072afc) and passes on the
candidate.  Window tests (UX01-UX05) run in a child interpreter like the other
desktop tests; the parent only starts it.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import pytest

QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from structstudio.app import modify as M  # noqa: E402
from structstudio.app.document import Document  # noqa: E402
from structstudio.app.templates import build_model, new_building  # noqa: E402
from structstudio.app.tools import DrawSettings, Plane, PointTool  # noqa: E402
from structstudio.model import Model  # noqa: E402
from structstudio.model.base import ModelError  # noqa: E402

CHILD = os.environ.get("STRUCTSTUDIO_GUI_CHILD") == "1"


# ================================================================ UX06 typed input
@pytest.mark.parametrize("text", ["1e309,0,3", "#1e309,0,3", "0,-1e309,3", "#0,0,1e999"])
def test_ux06_overflowing_typed_point_is_refused_and_the_model_is_unchanged(text):
    doc = Document(new_building())
    t = PointTool(doc, DrawSettings())
    t.set_story("Story1")
    before, undo = doc.model.to_json(), len(doc._undo)
    with pytest.raises(ValueError, match="finite"):
        t.click(t.typed(text))
    assert doc.model.to_json() == before and len(doc._undo) == undo
    assert all(math.isfinite(v) for p in doc.model.points.values() for v in p.xyz)


def test_ux06_large_but_finite_and_normal_points_are_still_accepted():
    doc = Document(new_building())
    t = PointTool(doc, DrawSettings())
    t.set_story("Story1")
    assert t.typed("#1e300,0,3") == (1e300, 0.0, 3.0)  # not clamped, not zeroed
    assert t.typed("#0,0,3") == (0.0, 0.0, 3.0)
    t.click(t.typed("#2,1,3"))
    assert [p.xyz[:2] for p in doc.model.points.values()] == [(2.0, 1.0)]  # (z: the plan's story)


def test_ux06_relative_input_that_overflows_is_refused():
    plane = Plane("plan", 3.0)
    with pytest.raises(ValueError, match="finite|range"):
        M.parse_typed("1e308,0,0", (1e308, 0.0, 3.0), None, plane)  # 1e308 + 1e308 = inf
    with pytest.raises(ValueError, match="finite"):
        M.parse_typed("1e309", (0.0, 0.0, 3.0), (1.0, 0.0, 3.0), plane)
    assert M.parse_typed("3,4", (0.0, 0.0, 3.0), None, plane) == (3.0, 4.0, 3.0)


def test_ux06_the_model_refuses_nonfinite_coordinates_and_rolls_back():
    doc = Document(new_building(grid_only=False))
    before, undo = doc.model.to_json(), len(doc._undo)
    with pytest.raises(ModelError) as e:
        doc.edit(lambda m: m.add_point(float("inf"), 0, 0))
    assert e.value.code == "NONFINITE_COORDINATE"
    name = next(iter(doc.model.points))
    with pytest.raises(ModelError) as e:  # any command that would move a joint to inf / NaN
        doc.edit(lambda m: setattr(m.points[name], "x", float("nan")))
    assert e.value.code == "NONFINITE_COORDINATE"
    assert doc.model.to_json() == before and len(doc._undo) == undo


def test_ux06_validate_reports_nonfinite_values_and_open_refuses_them():
    m = new_building(grid_only=False)
    p = next(iter(m.points.values()))
    f = next(iter(m.frames.values()))
    p.x, f.angle = float("inf"), float("nan")
    codes = {(e.code, e.path) for e in m.validate()}
    assert ("NONFINITE_COORDINATE", f"points.{p.name}") in codes
    assert ("NONFINITE_VALUE", f"frames.{f.name}.angle") in codes
    with pytest.raises(ModelError) as e:
        Model.from_json(m.to_json())  # the file has Infinity in it
    assert e.value.code == "NONFINITE_COORDINATE"


# ================================================================ UX05 generator speed
def test_ux05_generated_models_are_the_same_with_the_point_index(monkeypatch):
    """The spatial hash only makes the joint search faster: the model is identical."""
    xs, ys = [0, 6, 12, 18], [0, 5, 10]
    st = [("S1", 3.0), ("S2", 3.0)]
    fast = {k: build_model(x_ordinates=xs, y_ordinates=ys, stories=st, structure=k).to_json()
            for k in ("rc_frame", "flat_slab_edge", "core_frame", "steel_deck")}

    class NoIndex:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(Model, "point_index", lambda self: NoIndex())
    for k, j in fast.items():
        assert build_model(x_ordinates=xs, y_ordinates=ys, stories=st, structure=k).to_json() == j, k


def test_ux05_generator_finds_joints_by_index_not_by_scanning(monkeypatch):
    calls = {"scan": 0}
    real = Model.find_point

    def spy(self, x, y, z, tol=None):
        if self._pindex is None:
            calls["scan"] += 1
        return real(self, x, y, z, tol)

    monkeypatch.setattr(Model, "find_point", spy)
    m = build_model(x_ordinates=[i * 6.0 for i in range(12)], y_ordinates=[i * 5.0 for i in range(12)],
                    stories=[("S1", 3.0), ("S2", 3.0)], structure="rc_frame")
    assert len(m.frames) > 500 and calls["scan"] == 0
    assert m._pindex is None  # the index is gone afterwards: later edits search normally


# ================================================================ window (child)
@pytest.mark.skipif(CHILD, reason="parent only")
def test_ux_window_cases_in_subprocess():
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


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    return app


def _settle(app, n=5):
    for _ in range(n):
        app.processEvents()


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    pytest.importorskip("vtk")
    monkeypatch.setenv("STRUCTSTUDIO_THUMBS", str(tmp_path / "thumbs"))
    from structstudio.app.mainwindow import MainWindow
    w = MainWindow(Document(new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3, 3), grid_only=False)),
                   QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    w.show()
    w.resize(1600, 950)
    _settle(qapp)
    yield w
    w.doc.dirty = False
    w.views.finalize()
    w.close()
    w.deleteLater()
    _settle(qapp)


def _palette_ok(w):
    """Every palette button is shown, whole, inside the window (none folded behind a >> arrow)."""
    g = w.palette_grid
    if g is None:
        return True
    win = w.rect()
    for b in g.buttons:
        if b is None:
            continue
        r = QtCore.QRect(b.mapTo(w, QtCore.QPoint(0, 0)), b.size())
        if not (b.isVisible() and g.rect().contains(b.geometry()) and win.contains(r)):
            return False
    return g.isVisible()


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_ux01_full_palette_lets_the_window_shrink_and_reflows(window, qapp):
    w = window
    w.set_palette_mode("full")
    _settle(qapp)
    for size in ((1280, 720), (1366, 768), (1024, 700), (1920, 1000), (1280, 720)):
        w.resize(*size)
        _settle(qapp, 8)
        assert (w.width(), w.height()) == size, (size, (w.width(), w.height()), w.minimumSizeHint())
        assert _palette_ok(w), size
    # short window: more columns, every tool still there; tall window: one column again
    w.resize(1280, 600)
    _settle(qapp, 8)
    assert w.height() == 600 and w.palette_grid.columns() >= 2 and _palette_ok(w)
    w.resize(1600, 1000)
    _settle(qapp, 8)
    assert w.palette_grid.columns() <= 2 and _palette_ok(w)


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_ux01_no_mode_switch_workaround_needed(window, qapp):
    w = window
    w.set_palette_mode("full")
    w.resize(1280, 720)
    _settle(qapp, 8)
    direct = (w.width(), w.height())
    w.set_palette_mode("compact")
    w.resize(1280, 720)
    w.set_palette_mode("full")
    w.resize(1280, 720)
    _settle(qapp, 8)
    assert direct == (w.width(), w.height()) == (1280, 720)


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
@pytest.mark.parametrize("layout", ["1", "2", "2v", "3", "4", "6", "9"])
def test_gui_ux02_many_windows_do_not_widen_the_program(window, qapp, layout):
    w = window
    w.set_palette_mode("compact")
    w.views.set_layout(layout)
    _settle(qapp, 8)
    w.resize(1280, 800)
    _settle(qapp, 8)
    assert w.width() == 1280, (layout, w.width(), w.minimumSizeHint())
    for v in w.views.visible_viewports():
        hdr = v.header
        # the controls that matter stay usable in the header
        for c in (v.kind_box, v.results_box) + ((v.where_box,) if v.config.kind != "3d" else ()):
            assert c.isVisible() and c.width() >= c.minimumSizeHint().width() - 1, (layout, type(c).__name__)
        assert all(hdr.rect().contains(QtCore.QRect(c.pos(), c.size())) for c in (v.kind_box, v.results_box))
    w.views.set_layout("1")
    _settle(qapp, 8)
    assert w.width() == 1280


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_ux02_header_title_is_elided_not_lost(window, qapp):
    w = window
    w.views.set_layout("9")
    w.resize(1280, 800)
    _settle(qapp, 8)
    v = w.views.visible_viewports()[0]
    assert v.title.text() and v.title.toolTip() == v.title.text()  # full text kept (tooltip)
    assert v.title.minimumSizeHint().width() == 0


def _buttons_on_screen(d, names, app):
    avail = (d.screen() or app.primaryScreen()).availableGeometry()
    found = {}
    for b in d.findChildren(QtWidgets.QAbstractButton):
        t = b.text().replace("&", "")
        if t in names and b.isVisible():
            r = QtCore.QRect(b.mapToGlobal(QtCore.QPoint(0, 0)), b.size())
            found[t] = d.rect().contains(QtCore.QRect(b.mapTo(d, QtCore.QPoint(0, 0)), b.size())) and avail.contains(r)
    return found


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_ux03_new_model_wizard_fits_small_screens(qapp, tmp_path):
    from structstudio.app.new_model import NewModelWizard
    w = NewModelWizard(None, QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    try:
        for page in range(3):
            w.go(page)
            w.show()
            _settle(qapp)
            for size in ((1024, 650), (900, 600)):
                w.resize(*size)
                _settle(qapp)
                assert (w.width(), w.height()) == size, (page, size, w.minimumSizeHint())
                names = {"Cancel", w.btn_next.text()} | ({"‹ Back"} if page else set())
                found = _buttons_on_screen(w, names, qapp)
                assert set(found) == names and all(found.values()), (page, size, found)
        # values survive Back / Next in the small window
        w.go(1)
        w.u_nx.setValue(7)
        w.go(2)
        w.go(1)
        assert w.u_nx.value() == 7
    finally:
        w.hide()
        w.deleteLater()
        _settle(qapp)


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_ux04_display_and_colour_dialogs_fit_a_short_screen(qapp):
    from structstudio.app import colors_dialog as cd
    from structstudio.app.display import DisplayOptions
    from structstudio.app.display_dialog import DisplayOptionsDialog
    d = DisplayOptionsDialog(DisplayOptions())
    c = cd.GraphicsColorsDialog(cd.ColorSettings(), new_building(grid_only=False))
    try:
        assert d.minimumSizeHint().height() <= 600 and c.minimumSizeHint().height() <= 600
        d.show()
        _settle(qapp)
        d.resize(d.width(), 560)
        _settle(qapp)
        assert d.height() == 560 and all(_buttons_on_screen(d, {"OK", "Cancel"}, qapp).values())
        c.show()
        for i in range(c.tabs.count()):
            c.tabs.setCurrentIndex(i)
            c.resize(c.width(), 560)
            _settle(qapp)
            found = _buttons_on_screen(c, {"OK", "Cancel", "Apply"}, qapp)
            assert c.height() == 560 and len(found) == 3 and all(found.values()), (c.tabs.tabText(i), found)
    finally:
        for x in (d, c):
            x.hide()
            x.deleteLater()
        _settle(qapp)


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_ux04_dialog_is_kept_on_its_screen(qapp):
    from structstudio.app.display import DisplayOptions
    from structstudio.app.display_dialog import DisplayOptionsDialog
    d = DisplayOptionsDialog(DisplayOptions())
    try:
        avail = qapp.primaryScreen().availableGeometry()
        d.resize(avail.width() + 400, avail.height() + 400)
        d.move(avail.right() - 50, avail.bottom() - 50)
        d.show()
        _settle(qapp, 8)
        assert avail.contains(d.frameGeometry()), (avail, d.frameGeometry())
    finally:
        d.hide()
        d.deleteLater()
        _settle(qapp)


@pytest.mark.skipif(not CHILD, reason="desktop test: run in the child interpreter")
def test_gui_ux05_large_grid_preview_keeps_the_window_answering(qapp, tmp_path):
    """30 x 30 lines, 8 stories: Next to the structure page used to block 16-19 s."""
    from structstudio.app.new_model import NewModelWizard
    w = NewModelWizard(None, QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat))
    try:
        w.show()
        w.u_nx.setValue(30)
        w.u_ny.setValue(30)
        w.u_ns.setValue(8)
        w.go(1)
        _settle(qapp)
        t = time.perf_counter()
        w.go(2)
        took = time.perf_counter() - t
        assert w.pages.currentIndex() == 2 and took < 2.0, took
        note = w.preview_note.text()
        assert "30 × 30" in note and "first 12 × 12" in note and "lowest 3 of 8" in note
        deadline = time.perf_counter() + 20
        while w.previews_pending() and time.perf_counter() < deadline:
            qapp.processEvents()
        assert not w.previews_pending()
        assert all(w.cards[k].sketch for k in ("grid", "rc_frame", "core_frame", "steel_deck"))
        # a change made while previews are drawn wins: old work is dropped (generation number)
        w.go(1)
        w.u_nx.setValue(4)
        w.go(2)
        assert "4 × 30" in w.preview_note.text()
    finally:
        w.hide()
        w.deleteLater()
        _settle(qapp)
