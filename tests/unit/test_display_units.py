"""Display units item by item (Options > Display Units, second tab): e.g. the Thai
legacy sets - every force in tonf or every force in kgf, lengths in m, section
dimensions in cm, areas in cm^2, deflections in mm, strengths in ksc."""
import os
import re

import pytest

from structstudio.display_units import CHOICES, ITEMS, PRESETS, DisplayUnits, item_of, unit_si
from structstudio.model import Model
from structstudio.units import UnitSystem

THAI = PRESETS["Thai legacy - tonf (tonf, tonf-m, tonf/m, tonf/m^2, ksc)"]
THAI_KG = PRESETS["Thai legacy - kgf (kgf, kgf-m, kgf/m, kgf/m^2, ksc)"]


def test_unit_labels_are_parsed():
    g = 9.80665
    assert unit_si("ksc") == pytest.approx(g / 1e-4)  # kg/cm^2
    assert unit_si("kg/cm^2") == pytest.approx(unit_si("ksc"))
    assert unit_si("ton-m") == pytest.approx(1000 * g)
    assert unit_si("kg/m^2") == pytest.approx(g)
    assert unit_si("ton/m^3") == pytest.approx(1000 * g)
    assert unit_si("MPa") == unit_si("N/mm^2") == pytest.approx(1e6)
    assert unit_si("kip-ft") == pytest.approx(4448.2216152605 * 0.3048)
    assert unit_si("cm^4") == pytest.approx(1e-8)
    with pytest.raises(ValueError):
        unit_si("furlong")
    for q, labels in CHOICES.items():  # every offered unit is understood
        for lab in labels:
            assert unit_si(lab) > 0


def test_thai_legacy_values():
    consistent, items = THAI
    assert consistent == ("tonf", "m", "C")
    du = DisplayUnits(UnitSystem("kN", "m"), items)  # whatever the model units are
    assert du.label("stress") == "ksc" and du.label("displacement") == "mm" and du.label("force") == "tonf"
    assert du.show(28e3, "stress") == pytest.approx(285.52, rel=1e-4)  # 28 MPa = 285.5 ksc
    assert du.show(0.3, "section_dim") == pytest.approx(30.0)          # 0.3 m = 30 cm
    assert du.show(4.9e-4, "section_area") == pytest.approx(4.9)       # rebar 4.9 cm^2
    assert du.show(9.80665, "force") == pytest.approx(1.0)             # 9.81 kN = 1 tonf
    assert du.show(2.0, "area_load") == pytest.approx(0.20394, rel=1e-4)  # 2 kN/m^2 = 0.204 tonf/m^2
    assert du.show(1.0, "line_load") == pytest.approx(0.10197, rel=1e-4)  # tonf/m
    assert du.show(24.0, "unit_weight") == pytest.approx(2.4473, rel=1e-4)  # tonf/m^3
    assert du.parse(du.show(12.5, "moment"), "moment") == pytest.approx(12.5)
    assert du.per_length("moment") == ("tonf-m/m", pytest.approx(1 / 9.80665))
    # SI results (deflection in m) -> mm
    assert du.si_factor("displacement") == pytest.approx(1000.0)


def test_thai_legacy_sets_keep_one_force_unit():
    """Two Thai sets: every force-based unit in kgf, or every one in tonf - never mixed."""
    for (consistent, items), force in ((THAI_KG, "kgf"), (THAI, "tonf")):
        assert consistent[0] == force and consistent[1] == "m"
        for key in ("force", "moment", "line_load", "area_load", "unit_weight"):
            assert items[key].split("-")[0].split("/")[0] == force, (key, items[key])
        assert items["stress"] == "ksc" and items["displacement"] == "mm" and items["section_dim"] == "cm"
        assert items["section_area"] == "cm^2" and items["coordinates"] == "m"
    du = DisplayUnits(UnitSystem("kN", "m"), THAI_KG[1])
    assert du.show(2.0, "area_load") == pytest.approx(203.94, rel=1e-4)  # 2 kN/m^2 = 204 kgf/m^2
    assert du.show(9.80665, "force") == pytest.approx(1000.0) and du.show(9.80665, "moment") == pytest.approx(1000.0)
    assert du.show(28e3, "stress") == pytest.approx(285.52, rel=1e-4)


def test_items_of_model_fields():
    assert item_of("length", "FrameSection", "t3") == "section_dim"
    assert item_of("length", "ShellSection", "thickness") == "section_dim"
    assert item_of("length", "Point", "x") == "coordinates"
    assert item_of("area", "FiberSectionDef", "bar_area") == "section_area"
    assert item_of("stress", "Material", "fc") == "stress"
    assert item_of("force_per_area", "AreaLoad", "value") == "area_load"
    assert item_of("force_per_area", "LineSpring", "stiffness") is None  # a spring, not a load
    assert {k for k, _l, _q in ITEMS} >= {"coordinates", "section_dim", "section_area", "displacement", "force",
                                          "moment", "line_load", "area_load", "stress", "unit_weight"}


def test_display_units_are_saved_with_the_model_and_survive_unit_changes(tmp_path):
    m = Model("t", "kN m")
    m.display_units = dict(THAI[1])
    p = tmp_path / "m.ssj.json"
    m.to_json(str(p))
    m2 = Model.from_json(str(p))
    assert m2.display_units == THAI[1]
    assert m2.convert_units("tonf m C").display_units == THAI[1]


def test_result_tables_in_display_units():
    from structstudio.app.panels import table_in_display_units
    m = Model("t", "kN m")
    m.display_units = dict(THAI[1])
    rows = [{"Point": "P1", "Ux": 0.001, "Uy": 0.0, "Uz": -0.0025, "Rx": 0.001}]
    out = table_in_display_units(m, "Joint displacements", rows)
    assert out[0]["Uz [mm]"] == pytest.approx(-2.5) and out[0]["Rx"] == 0.001
    out = table_in_display_units(m, "Frame forces (selected frame)", [{"Station": 0.3, "P": -98.0665, "M3": 9.80665,
                                                                        "M3Min": -19.6133}])
    assert out[0]["Station [m]"] == pytest.approx(0.3) and out[0]["P [tonf]"] == pytest.approx(-10.0)
    assert out[0]["M3 [tonf-m]"] == pytest.approx(1.0) and out[0]["M3Min [tonf-m]"] == pytest.approx(-2.0)
    out = table_in_display_units(m, "Shell forces", [{"M11": 9.80665, "F11": 9.80665}])  # tonf set
    assert out[0]["M11 [tonf-m/m]"] == pytest.approx(1.0) and out[0]["F11 [tonf/m]"] == pytest.approx(1.0)


def test_document_change_of_display_units_only():
    from structstudio.app.document import Document
    m = Model("t", "kN m")
    doc = Document(m)
    events = []
    doc.on_change(events.append)
    doc.change_units("kN m", THAI[1])  # same model units, only the display
    assert doc.model.units == UnitSystem("kN", "m") and doc.model.display_units == THAI[1]
    assert events == ["units", "model"] and doc.units_ratio == 1.0
    assert doc.undo() and doc.model.display_units == {}


@pytest.mark.skipif(not os.environ.get("QT_QPA_PLATFORM"), reason="needs Qt (offscreen)")
def test_forms_show_and_take_values_in_display_units():
    from PySide6 import QtWidgets
    from structstudio.app import forms
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    forms.DISPLAY["units"] = DisplayUnits(UnitSystem("kN", "m"), THAI_KG[1])
    try:
        d = forms.FormDialog("t", [("w", "Line load", 9.80665, {"item": "line_load"}),
                                   ("o", "Offset", 0.25, {"item": "coordinates"}), ("n", "Plain", 2.0)])
        w = d._widgets["w"]
        assert float(w.text()) == pytest.approx(1000.0)  # 9.81 kN/m shown as 1000 kgf/m
        w.setText("500")
        v = d.values()
        assert v["w"] == pytest.approx(9.80665 / 2) and v["o"] == pytest.approx(0.25) and v["n"] == 2.0
    finally:
        forms.DISPLAY["units"] = None
    assert app is not None


def test_forces_are_labelled_kgf_and_tonf_never_kg_or_ton():
    from structstudio.display_units import CHOICES, force_label
    for name, (_c, items) in PRESETS.items():
        for key in ("force", "moment", "line_load", "area_load", "unit_weight"):
            label = items.get(key, "")
            assert not re.search(r"\b(kg|ton|t)\b", label), (name, key, label)
    for q in ("force", "moment", "force_per_length", "force_per_area", "stress", "weight_per_volume"):
        assert not any(re.search(r"\b(kg|ton)\b", c) for c in CHOICES[q])
    # labels from older files / typed by hand are made explicit
    assert force_label("kg/m^2") == "kgf/m^2" and force_label("ton-m") == "tonf-m" and force_label("t/m") == "tonf/m"
    assert force_label("kgf-cm") == "kgf-cm" and force_label("ksc") == "ksc" and force_label("MPa") == "MPa"
    du = DisplayUnits(UnitSystem("kN", "m", "C"), {"force": "ton", "area_load": "kg/m^2"})
    assert du.label("force") == "tonf" and du.label("area_load") == "kgf/m²"  # printed with a superscript
    assert du.show(9.80665, "force") == pytest.approx(1.0)


def test_old_saved_force_labels_load_as_kgf_tonf():
    from structstudio.model import Model
    m = Model()
    data = m.to_dict()
    data["display_units"] = {"force": "ton", "moment": "ton-m", "line_load": "kg/m"}
    m2 = Model.from_dict(data)
    assert m2.display_units == {"force": "tonf", "moment": "tonf-m", "line_load": "kgf/m"}
