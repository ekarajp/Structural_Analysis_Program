"""Document.edit: a command that changes nothing is no edit; a locked model asks to
unlock only when the command changes something."""
import pytest

from structstudio import Model
from structstudio.app.document import Document
from structstudio.model.base import ModelError


def _doc():
    m = Model("d", "kN m")
    m.add_point(0, 0, 0)
    m.add_point(1, 0, 0)
    return Document(m)


def test_a_command_that_changes_nothing_is_no_edit():
    doc = _doc()
    doc.edit(lambda m: m.add_point(2, 0, 0))
    doc.undo()
    assert doc.can_redo
    doc.dirty = False
    seen = []
    doc.on_change(seen.append)
    doc.edit(lambda m: None)  # OK in a form that was only looked at
    doc.edit(lambda m: setattr(m.points["P1"], "x", m.points["P1"].x))  # the same value written back
    assert not doc.can_undo and doc.can_redo and not doc.dirty and seen == []
    doc.edit(lambda m: setattr(m.points["P1"], "x", 5.0))
    assert doc.can_undo and doc.dirty and not doc.can_redo and seen == ["model"]


def test_locked_model_asks_only_for_a_real_change():
    doc = _doc()
    asked = []
    m0 = doc.model
    doc.results = object()  # (stands for analysis results)
    doc.locked = True
    doc.ask_unlock = lambda: asked.append(1) or False
    doc.edit(lambda m: None)
    # objects made again with the same values (a form that rebuilds its list) change nothing either
    doc.edit(lambda m: m.points.update({"P1": type(m.points["P1"])(**vars(m.points["P1"]))}))
    assert asked == [] and doc.locked and doc.results is not None and not doc.can_undo
    # a real change: asked; refused -> the model is as it was, still locked, results kept
    with pytest.raises(ModelError) as e:
        doc.edit(lambda m: setattr(m.points["P1"], "x", 9.0))
    assert e.value.code == "MODEL_LOCKED" and asked == [1]
    assert doc.model is m0  # put back into the same object (the results hold it)
    assert doc.model.points["P1"].x == 0.0 and doc.locked and doc.results is not None and not doc.can_undo
    # accepted -> unlocked, the results are deleted, one undo step
    doc.ask_unlock = lambda: True
    doc.edit(lambda m: setattr(m.points["P1"], "x", 9.0))
    assert doc.model.points["P1"].x == 9.0 and not doc.locked and doc.results is None
    doc.undo()
    assert doc.model.points["P1"].x == 0.0


def test_a_failed_command_leaves_a_locked_model_alone():
    doc = _doc()
    doc.results, doc.locked = object(), True
    doc.ask_unlock = lambda: True

    def bad(m):
        m.points["P1"].x = 3.0
        raise ValueError("no")
    with pytest.raises(ValueError):
        doc.edit(bad)
    assert doc.model.points["P1"].x == 0.0 and doc.locked and doc.results is not None and not doc.can_undo


def _defined():
    from structstudio.model import FrameSection, ShellSection
    m = Model("d", "kN m")
    m.add_material("C", E=25e6, nu=.2)
    m.add_frame_section(FrameSection.general("B", "C", A=.1, I33=1e-3, I22=1e-3, J=1e-3))
    m.add_shell_section(ShellSection("S", "C", thickness=0.2))
    a, b = m.add_point(0, 0, 0), m.add_point(4, 0, 0)
    m.add_frame(a, b, "B")
    m.add_area([a, b, m.add_point(4, 3, 0), m.add_point(0, 3, 0)], "S", type="Floor")
    return Document(m)


def test_checked_edit_refuses_a_change_that_breaks_the_model():
    """Define forms: E = 0, a section without thickness, a deleted material that is still
    used, a tolerance larger than the members - refused with the reason, nothing changed
    (they were accepted and found only when the analysis was run)."""
    from structstudio.app.commands import checked_edit
    doc = _defined()
    for change, word in ((lambda m: setattr(m.materials["C"], "E", 0.0), "E > 0"),
                         (lambda m: setattr(m.shell_sections["S"], "thickness", 0.0), "thickness"),
                         (lambda m: m.materials.pop("C"), "unknown material"),
                         (lambda m: setattr(m.options, "merge_tolerance", 100.0), "no length")):
        with pytest.raises(ModelError) as e:
            checked_edit(doc, change)
        assert word in e.value.message
    m = doc.model
    assert m.materials["C"].E == 25e6 and m.shell_sections["S"].thickness == 0.2 and not doc.can_undo
    checked_edit(doc, lambda m: setattr(m.materials["C"], "E", 30e6))  # a good change goes through
    assert doc.model.materials["C"].E == 30e6 and doc.can_undo
    # an error the model has already does not stop other changes
    doc.model.shell_sections["S"].thickness = 0.0
    checked_edit(doc, lambda m: setattr(m.materials["C"], "nu", 0.25))
    assert doc.model.materials["C"].nu == 0.25


def test_define_form_keeps_open_on_a_bad_value(monkeypatch):
    from PySide6 import QtWidgets
    from structstudio.app import dialogs
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    said = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: said.append(a[2])))
    doc = _defined()
    d = dialogs.DefineDialog(doc, "materials", "Materials", dialogs.material_factory)
    d.table["C"].E = 0.0
    d.form.set_object(d.table["C"])
    d.accept()
    assert said and "E > 0" in said[0] and d.result() != QtWidgets.QDialog.Accepted
    assert doc.model.materials["C"].E == 25e6


def test_absurd_coordinates_are_refused():
    doc = _doc()
    with pytest.raises(ModelError) as e:
        doc.edit(lambda m: setattr(m.points["P1"], "x", 1e12))
    assert e.value.code == "NONFINITE_COORDINATE" and doc.model.points["P1"].x == 0.0
