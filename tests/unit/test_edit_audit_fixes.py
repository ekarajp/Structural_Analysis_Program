"""Edit menu audit fixes: objects follow their stories, design strips in Move / Copy /
Rotate / Mirror / Replicate / Clipboard / Delete / Find, and the Replicate checks."""
import math

import numpy as np
import pytest

from structstudio.app import commands as C
from structstudio.app import find
from structstudio.app.document import Document
from structstudio.app.modify import copy_transformed, mirror, rotation, transform_in_place, translation
from structstudio.app.templates import build_model, new_building
from structstudio.app.tools import delete_objects, replicate_to_stories


def _zs(m):
    return sorted({round(p.z, 6) for p in m.points.values()})


# ------------------------------------------------------------- stories carry objects
def test_carry_levels_keeps_the_distance_under_each_level():
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.0, 3.0, 3.0), grid_only=False)
    p = m.add_point(0.0, 0.0, 5.5)  # 0.5 under Story2 (belongs to it, as in ETABS)
    q = m.add_point(0.0, 0.0, 10.0)  # above the top: with the top
    old = [(s.name, s.elevation) for s in m.stories_bottom_up()]
    new = [("Base", 0.0), ("Story1", 4.0), ("Story2", 7.0), ("Story3", 10.0)]
    moved = m.carry_levels(old, new)
    assert moved > 0
    assert p.z == pytest.approx(6.5) and q.z == pytest.approx(11.0)
    m._stories_from_levels(new)
    assert _zs(m) == [0.0, 4.0, 6.5, 7.0, 10.0, 11.0]
    assert {f.story for f in m.frames.values() if f.type == "Beam"} == {"Story1", "Story2", "Story3"}


def test_carry_levels_renamed_and_xyz_on_levels_only():
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0, 3.0), grid_only=False)
    old = [(s.name, s.elevation) for s in m.stories_bottom_up()]
    # every story renamed (same count): paired in order
    pairs, gone = m.pair_levels(old, [("B", 0.0), ("L1", 3.5), ("L2", 6.5)])
    assert pairs == {"Base": "B", "Story1": "L1", "Story2": "L2"} and not gone
    mid = m.add_point(0.0, 0.0, 1.5)
    m.carry_levels(old, [("Base", 0.0), ("Story1", 3.5), ("Story2", 6.5)], on_levels_only=True)
    assert mid.z == 1.5 and _zs(m) == [0.0, 1.5, 3.5, 6.5]


def test_deleted_story_takes_its_objects_the_story_above_comes_down():
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.0, 3.0, 3.0), grid_only=False)
    old = [(s.name, s.elevation) for s in m.stories_bottom_up()]
    new = [("Base", 0.0), ("Story1", 3.0), ("Story3", 6.0)]
    _pairs, gone = m.pair_levels(old, new)
    assert gone == ["Story2"]
    objs = m.objects_on_levels(gone)
    assert objs["frames"] and objs["areas"]
    m.delete_on_levels(gone)
    snapped = []
    m.carry_levels(old, new, snapped=snapped)
    from structstudio.app.modify import _merge_points
    _merge_points(m, snapped, 1e-6)
    m._stories_from_levels(new)
    assert _zs(m) == [0.0, 3.0, 6.0]
    locs = [tuple(round(c, 6) for c in p.xyz) for p in m.points.values()]
    assert len(locs) == len(set(locs))
    assert all(math.dist(m.points[f.point_i].xyz, m.points[f.point_j].xyz) > 1e-6 for f in m.frames.values())
    cols = [f for f in m.frames.values() if f.story == "Story3" and f.type == "Column"]
    assert cols and all(sorted((m.points[f.point_i].z, m.points[f.point_j].z)) == [3.0, 6.0] for f in cols)


def test_strips_and_panels_follow_their_story():
    m = build_model(structure="flat_slab", x_ordinates=(0, 8, 16), y_ordinates=(0, 6))
    ds = m.add_design_strip([(0, 3, 3.5), (16, 3, 3.5)], (1.0, 1.0))
    z1 = m.stories["Story1"].elevation
    assert ds.story == "Story1" and z1 == 3.5
    old = [(s.name, s.elevation) for s in m.stories_bottom_up()]
    new = [(n, z + (0.5 if z >= z1 else 0.0)) for n, z in old]
    m.carry_levels(old, new)
    m._stories_from_levels(new)
    assert ds.points[0][2] == pytest.approx(4.0) and m.strip_story(ds) == "Story1"


# ------------------------------------------------------------------ design strips
def _strip_doc():
    from structstudio.app import strip_commands as SC
    m = build_model(structure="flat_slab", x_ordinates=(0, 8, 16, 24), y_ordinates=(0, 6, 12))
    doc = Document(m)
    SC.auto_strips(doc, ["Story1"])
    sentinel = object()
    doc.results, doc.locked = sentinel, True
    doc.ask_unlock = lambda: pytest.fail("a strip edit must not ask to unlock the model")
    return doc, sentinel


def test_strips_move_copy_rotate_mirror_keep_results():
    doc, sentinel = _strip_doc()
    m = doc.model
    name = next(ds.name for ds in m.design_strips.values() if ds.layer == "A" and ds.kind == "Column")
    p0 = list(m.design_strips[name].points)
    w0 = list(m.design_strips[name].widths)
    transform_in_place(doc, [], [], [], translation((0.0, 0.5, 0.0)), strips=[name])
    assert m.design_strips[name].points[0][1] == pytest.approx(p0[0][1] + 0.5)
    assert doc.results is sentinel and doc.locked
    doc.undo()
    assert doc.model.design_strips[name].points == p0 and doc.results is sentinel
    # copy one storey up is refused off a story, kept on a story
    fr, ar, pts = copy_transformed(doc, [], [], [], translation((0.0, 0.0, 3.5)), strips=[name])
    new = copy_transformed.last_strips
    assert len(new) == 1 and doc.model.design_strips[new[0]].story == "Story2" and doc.results is sentinel
    fr, ar, pts = copy_transformed(doc, [], [], [], translation((0.0, 0.0, 1.0)), strips=[name])
    assert copy_transformed.last_strips == []
    # mirror: left and right widths swap
    T = mirror((0.0, 0.0, 3.5), (1.0, 0.0, 3.5), np.array([0.0, 0.0, 1.0]))
    copy_transformed(doc, [], [], [], T, strips=[name])
    made = doc.model.design_strips[copy_transformed.last_strips[0]]
    assert [tuple(w) for w in made.widths] == [(w[1], w[0], w[3], w[2]) for w in w0]
    # rotate in an elevation: not level any more - left where it was
    before = list(doc.model.design_strips[name].points)
    transform_in_place(doc, [], [], [], rotation((0.0, 0.0, 3.5), np.array([0.0, 1.0, 0.0]), 20.0), strips=[name])
    assert doc.model.design_strips[name].points == before


def test_strips_replicate_clipboard_delete_find():
    doc, sentinel = _strip_doc()
    m = doc.model
    names = sorted(n for n, ds in m.design_strips.items() if ds.story == "Story1")[:3]
    made = replicate_to_stories(doc, [], [], [], ["Story2"], strips=names)
    assert len(made) == 3 and all(doc.model.design_strips[n].story == "Story2" for n in made)
    assert doc.results is sentinel
    made = C.replicate_transform(doc, [], [], [], C.linear_transforms(0.0, 0.0, 3.5, 1), strips=names[:1])
    assert len(made) == 1 and doc.results is sentinel
    # clipboard: strips only - a design edit
    clip = C.copy_objects(doc.model, [], [], [], strips=names[:2])
    assert not clip.empty
    C.paste_objects(doc, clip, 0.0, 0.0, 3.5)
    assert len(C.paste_objects.last_strips) == 2 and doc.results is sentinel
    assert all(n.startswith(("CS", "MS")) for n in C.paste_objects.last_strips)
    clip_mm = clip.in_units(type(doc.model.units).parse("kN mm"))
    assert clip_mm.strips[0].points[0][2] == pytest.approx(clip.strips[0].points[0][2] * 1000)
    # delete: strips alone keep the results; with a beam one undo step
    k = len(doc._undo)
    assert delete_objects(doc, strips=[names[0]]) == 1 and doc.results is sentinel
    beam = next(iter(doc.model.frames))
    doc.ask_unlock = lambda: True  # (a frame deleted: the model is unlocked)
    k = len(doc._undo)
    delete_objects(doc, frames=[beam], strips=[names[1]])
    assert len(doc._undo) == k + 1 and names[1] not in doc.model.design_strips
    doc.undo()
    assert names[1] in doc.model.design_strips and beam in doc.model.frames
    # find by name / label
    hits = find.find_objects(doc.model, names[2])
    assert hits and hits[0].attr == "design_strips" and hits[0].name == names[2]
    assert "Design strip" in hits[0].text


# ------------------------------------------------------------------------- labels
def test_new_objects_get_the_next_labels():
    m = new_building(x_spacings=(6, 6), y_spacings=(5,), story_heights=(3.0, 3.0), grid_only=False)
    doc = Document(m)
    C.auto_relabel(doc)
    beams = sorted(f.label for f in doc.model.frames.values() if f.type == "Beam" and f.story == "Story1")
    top = max(int(lab[1:]) for lab in beams)
    a = doc.model.points
    p1 = next(p for p in a.values() if p.story == "Story1" and p.x == 0 and p.y == 0)
    p2 = next(p for p in a.values() if p.story == "Story1" and p.x == 12 and p.y == 5)

    def draw(model):
        model.add_frame(model.points[p1.name], model.points[p2.name], next(iter(model.frame_sections)))
    doc.edit(draw)
    new = [f for f in doc.model.frames.values() if f.point_i == p1.name and f.point_j == p2.name][0]
    assert new.label == f"B{top + 1}"
    # copied to the same story: the next label; to another story: the same label
    fl = next(n for n, ar in doc.model.areas.items() if ar.story == "Story1")
    lab = doc.model.areas[fl].label
    made = C.replicate_transform(doc, [], [fl], [], C.linear_transforms(0.0, 0.0, 3.0, 1))
    up = [n for n in made if n in doc.model.areas]
    assert doc.model.areas[up[0]].story == "Story2"
    # (a floor is there on Story2 already: the copy gets the next label of that story)
    labels2 = [ar.label for n, ar in doc.model.areas.items() if ar.story == "Story2"]
    assert len(labels2) == len(set(labels2))
    # a divided shell: every piece its own label
    from structstudio.app import shell_divide as SD
    pieces = SD.pieces_for(doc.model, [fl], "equal", {"n1": 2, "n2": 2})
    SD.divide(doc, pieces)
    labels1 = [ar.label for ar in doc.model.areas.values() if ar.story == "Story1"]
    assert len(labels1) == len(set(labels1)) and lab in labels1


def test_no_labels_model_stays_unlabelled_and_messages_show_labels():
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    doc = Document(m)
    assert not any(f.label for f in m.frames.values())
    C.replicate_transform(doc, list(m.frames)[:1], [], [], C.linear_transforms(1.0, 0.0, 0.0, 1))
    assert not any(f.label for f in doc.model.frames.values())
    from structstudio.app import locate
    f = next(iter(doc.model.frames.values()))
    C.change_labels(doc, "frame", {f.name: "XB9"})
    assert f"{f.name} (label XB9)" in locate.location_text(doc.model, "frames", f.name)


def test_prefix_clashes():
    from structstudio.model.objects import LabelRules
    r = LabelRules()
    assert C.prefix_clashes(r) == []
    r.prefixes["Brace"] = "B"
    assert C.prefix_clashes(r) == [("B", ["Beam", "Brace"])]
    r.by_type = False
    assert C.prefix_clashes(r) == []


# ------------------------------------------------------- strips: speed and stability
def test_auto_strips_typical_floor_once_progress_and_cancel():
    from structstudio.app import strip_commands as SC
    st = tuple((f"Story{k}", 3.5) for k in range(1, 5))
    m = build_model(structure="flat_slab", x_ordinates=(0, 8, 16, 24), y_ordinates=(0, 6, 12), stories=st)
    doc = Document(m)
    seen = []
    rep = SC.auto_strips(doc, ["Story4", "Story1", "Story2", "Story3"], progress=lambda k, n, s: seen.append(s))
    assert seen[:4] == ["Story4", "Story1", "Story2", "Story3"]
    assert any("the same floor as Story1" in n for n in rep.notes)
    lay = {s: sorted((ds.layer, ds.kind, tuple((round(p[0], 6), round(p[1], 6)) for p in ds.points))
                     for ds in doc.model.design_strips.values() if ds.story == s) for s in ("Story1", "Story2")}
    assert lay["Story1"] == lay["Story2"] and lay["Story1"]
    assert all(p[2] == doc.model.stories[ds.story].elevation for ds in doc.model.design_strips.values()
               for p in ds.points)
    # a fresh layout of Story2 gives the same strips as the reused one
    from structstudio.design import strips as S
    fresh = S.layout(doc.model, "Story2")
    assert sorted((s["layer"], s["kind"], tuple((round(p[0], 6), round(p[1], 6)) for p in s["points"]))
                  for s in fresh.strips) == lay["Story2"]
    k = len(doc._undo)
    rep = SC.auto_strips(doc, ["Story1"], progress=lambda k, n, s: False)
    assert rep.cancelled and len(doc._undo) == k


def test_auto_strip_widths_keeps_the_middle_strips():
    from structstudio.app import strip_commands as SC
    m = build_model(structure="flat_slab", x_ordinates=(0, 8, 16, 24), y_ordinates=(0, 6, 12))
    doc = Document(m)
    SC.auto_strips(doc, ["Story1"])
    mids = {n: ds for n, ds in doc.model.design_strips.items() if ds.kind == "Middle" and ds.story == "Story1"}
    doc.model.design_strips[next(iter(mids))].label = "M-keep"
    SC.auto_widths(doc, ["Story1"])
    after = {n for n, ds in doc.model.design_strips.items() if ds.kind == "Middle" and ds.story == "Story1"}
    assert after == set(mids)
    assert any(ds.label == "M-keep" for ds in doc.model.design_strips.values())


# -------------------------------------------------------------------- undo names
def test_undo_redo_name_the_command():
    from structstudio.app.menus import command_title, guarded_call
    assert command_title("Copy by Base Point  (CO)") == "Copy by Base Point"
    assert command_title("Array / Replicate to Stories...") == "Array / Replicate to Stories"
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    doc = Document(m)

    class W:
        pass
    w = W()
    w.doc = doc
    f = next(iter(m.frames))
    guarded_call(lambda: delete_objects(doc, frames=[f]), w, name="Delete")
    assert doc.undo_name() == "Delete" and doc.command_name is None
    doc.name_hint = lambda: "Draw Beam / Column / Brace"
    doc.edit(lambda mm: mm.add_point(50.0, 50.0, 0.0))
    assert doc.undo_name() == "Draw Beam / Column / Brace"
    doc.undo()
    assert doc.redo_name() == "Draw Beam / Column / Brace" and doc.undo_name() == "Delete"
    doc.redo()
    assert doc.undo_name() == "Draw Beam / Column / Brace"
