"""The ETABS / SAFE comparison: specs carry every object and load, and the
report reads CSI result tables (here: our own numbers written as CSI CSVs)."""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import csi_compare as cc  # noqa: E402


def test_spec_carries_all_objects_and_loads():
    m, programs = cc.model_inclined()
    s = cc.to_spec(m, "INCLINED", programs)
    assert len(s["points"]) == len(m.points) and len(s["frames"]) == len(m.frames) and len(s["areas"]) == len(m.areas)
    assert not s["skipped"]
    assert {a["type"] for a in s["areas"]} >= {"Wall", "Ramp", "Floor"}
    assert any(f["section"] == "None" for f in s["frames"]) and s["frame_loads"] and s["point_loads"]
    assert s["stories"] == [0.0, 3.5, 7.0]
    m2, _ = cc.model_two_slabs()
    s2 = cc.to_spec(m2, "TWOSLABS", ("ETABS",))
    assert any(a["opening"] for a in s2["areas"])


def test_report_from_csi_tables(tmp_path):
    folder = str(tmp_path)
    m, programs = cc.model_two_slabs()
    spec = cc.to_spec(m, "TWOSLABS", ("ETABS",))
    spec["checkpoints"] = cc.checkpoints(m)
    ours = cc.our_results(m, programs)
    assert ours["auto"]["mesh_check"] is not None  # the converged reference is there
    ours.pop("auto")  # this test compares against the h = 0.5 mesh
    json.dump(spec, open(os.path.join(folder, "TWOSLABS.spec.json"), "w"))
    json.dump(ours, open(os.path.join(folder, "TWOSLABS.ours.json"), "w"))
    open(os.path.join(folder, "models.txt"), "w").write("TWOSLABS\n")
    os.makedirs(os.path.join(folder, "etabs"))
    fine = ours["0.5"]
    with open(os.path.join(folder, "etabs", "TWOSLABS.joint_displacements.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["UniqueName", "Element", "Case", "StepType", "StepNum", "Ux", "Uy", "Uz", "Rx", "Ry", "Rz"])
        for n, d in fine["disp"].items():
            w.writerow([n, n, "Live", "", 0] + [v * 1.02 for v in d])   # CSI 2 % stiffer... softer
    with open(os.path.join(folder, "etabs", "TWOSLABS.joint_reactions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["UniqueName", "Case", "StepType", "StepNum", "F1", "F2", "F3", "M1", "M2", "M3"])
        for n, r in fine["react"].items():
            w.writerow([n, "Live", "", 0] + r)
    with open(os.path.join(folder, "etabs", "TWOSLABS.base_reactions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Case", "StepType", "StepNum", "FX", "FY", "FZ", "MX", "MY", "MZ"])
        w.writerow(["Live", "", 0] + fine["base"])
    text = cc.compare(folder)
    assert "TWOSLABS in ETABS" in text and "diff +0.000 %" in text
    line = next(l for l in text.splitlines() if l.strip().startswith("TWOSLABS  ETABS"))
    assert "largest deflection diff   1.96 %" in line and "reaction diff   0.00 %" in line
    assert os.path.exists(os.path.join(folder, "report.txt"))


@pytest.mark.parametrize("key", ["FREEFORM", "TWOSLABS"])
def test_rigid_zone_areas_reuse_coincident_joints(key):
    """ETABS / SAFE rename an existing joint when a new point lands on it, so the stiff
    areas over the columns must reuse joints at slab corners and shared footprint corners."""
    m, progs = cc.MODELS[key]()
    rz = cc.with_rigid_zones(m, cc.to_spec(m, key, progs))
    seen = {}
    for p in rz["points"]:
        k = (round(p["x"], 5), round(p["y"], 5), round(p["z"], 5))
        assert k not in seen, f"{p['name']} coincides with {seen[k]}"
        seen[k] = p["name"]
    names = {p["name"] for p in rz["points"]}
    stiff = [a for a in rz["areas"] if a["section"] == "COLSTIFF"]
    assert stiff and all(set(a["points"]) <= names and len(set(a["points"])) >= 3 for a in stiff)
