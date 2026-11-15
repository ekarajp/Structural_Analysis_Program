"""G Tower performance audit (20261115): the mesh of a building whose columns move from
story to story, the size guard of the analysis, and the faster model checks."""
import numpy as np
import pytest

from structstudio import Model, ModelError, analyze
from structstudio.api import build_analysis_model
from structstudio.app.templates import build_model
from structstudio.model import checks


def _tower(n_stories=6, shift=0.56, zones=True):
    """Flat-slab floors on columns that step ``shift`` along X at every story (as the
    leaning columns of a tower): each floor has its own column lines."""
    m = build_model(structure="grid", x_ordinates=(0, 6, 12), y_ordinates=(0, 5, 10),
                    stories=[(f"S{k}", 3.5) for k in range(1, n_stories + 1)])
    col = next(n for n in m.frame_sections if n.startswith("C"))
    slab = next(iter(m.shell_sections))
    m.options.slab_column_rigid_zone = zones
    zs = [0.0] + [3.5 * k for k in range(1, n_stories + 1)]
    for k in range(1, n_stories + 1):
        dx0, dx1 = shift * (k - 1), shift * k
        for x in (0.0, 6.0, 12.0):
            for y in (0.0, 5.0, 10.0):
                a = m.add_point(x + dx0, y, zs[k - 1])
                if k == 1:
                    a.restraints = (True,) * 6
                b = m.add_point(x + dx1, y, zs[k])
                m.add_frame(a, b, col)
        for i in range(2):
            for j in range(2):
                pts = [m.add_point(6.0 * (i + di) + dx1, 5.0 * (j + dj), zs[k])
                       for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1))]
                m.add_area(pts, slab)
    m.add_load_pattern("DL", "Dead", 1.0)
    m.add_linear_static_case("DL")
    return m


def _floor_x_lines(am, z, tol=1e-6):
    X = np.array([n.xyz for n in am.nodes])
    return np.unique(np.round(X[np.abs(X[:, 2] - z) <= tol][:, 0], 6))


def test_floors_do_not_take_the_lines_of_other_stories():
    m = _tower()
    am = build_analysis_model(m)
    lines1 = _floor_x_lines(am, 3.5)
    # the first floor is meshed for its own columns (x = 0.56 + 0, 6, 12): no line at the
    # column positions of the floors above (x = 1.12, 1.68 ...)
    for k in range(2, 7):
        assert not np.any(np.abs(lines1 - 0.56 * k) < 1e-6), k
    # and every floor has the same number of mesh lines (the same slab, shifted)
    counts = {len(_floor_x_lines(am, 3.5 * k)) for k in range(1, 7)}
    assert len(counts) == 1, counts


def test_element_count_grows_with_the_stories_not_with_their_square():
    n3 = len(build_analysis_model(_tower(3)).shells)
    n6 = len(build_analysis_model(_tower(6)).shells)
    assert n6 == pytest.approx(2 * n3, rel=0.02), (n3, n6)


def test_shifted_tower_still_carries_its_weight():
    """Vertical equilibrium of the shifted tower (floors of different mesh lines, columns
    tied into them): the base reaction is the weight of the slabs and columns."""
    m = _tower(4)
    r = analyze(m)
    slab = m.shell_sections[next(iter(m.shell_sections))]
    w = 4 * 12.0 * 10.0 * slab.thickness * m.materials[slab.material].weight_density
    for f in m.frames.values():
        sec = m.frame_sections[f.section]
        a, b = m.points[f.point_i], m.points[f.point_j]
        w += sec.A * float(np.linalg.norm(np.subtract(b.xyz, a.xyz))) * m.materials[sec.material].weight_density
    Rz = float(r.base_reactions("DL")[2])
    assert Rz == pytest.approx(w, rel=2e-3)


def test_regular_building_floors_all_mesh_alike():
    """A regular building (the same columns and walls at every story): every floor has the
    same mesh lines, as when all the floors shared them."""
    m = build_model(structure="core_frame", x_ordinates=(0, 6, 12, 18), y_ordinates=(0, 5, 10, 15),
                    stories=[(f"S{k}", 3.5) for k in range(1, 4)])
    am = build_analysis_model(m)
    l1, l2, l3 = (_floor_x_lines(am, z) for z in (3.5, 7.0, 10.5))
    assert len(l1) > 4 and np.array_equal(l1, l2) and np.array_equal(l2, l3)


def test_out_of_memory_is_reported_not_raised_raw(monkeypatch):
    import structstudio.api as api
    m = _tower(2, zones=False)

    def no_memory(self, *a, **k):
        raise MemoryError("no memory")
    monkeypatch.setattr(api.Engine, "run", no_memory)
    with pytest.raises(ModelError) as e:
        analyze(m)
    assert e.value.code == "OUT_OF_MEMORY"
    assert "too large for the memory" in e.value.message
    assert e.value.issues[0].hint and "mesh size" in e.value.issues[0].hint


def test_large_model_note_names_the_rigid_zones(monkeypatch):
    import structstudio.api as api
    monkeypatch.setattr(api, "LARGE_MODEL_SHELLS", 10)
    r = analyze(_tower(2))
    notes = [i.message for i in r.issues if i.message.startswith("Large analysis model")]
    assert notes and "rigid zone over columns" in notes[0]
    r2 = analyze(_tower(2, zones=False))
    notes2 = [i.message for i in r2.issues if i.message.startswith("Large analysis model")]
    assert all("rigid zone over columns" not in t for t in notes2)


def test_drawing_checks_on_a_changed_object_match_the_full_check():
    m = build_model(structure="rc_frame", x_ordinates=(0, 6, 12), y_ordinates=(0, 5, 10),
                    stories=[("S1", 3.5), ("S2", 3.5)])
    P = m.points
    f0 = next(f for f in m.frames.values() if f.type == "Beam")
    a, b = P[f0.point_i], P[f0.point_j]
    dup = m.add_frame(a, b, f0.section)  # drawn twice
    q = m.add_point(a.x + 0.4 * (b.x - a.x), a.y + 0.4 * (b.y - a.y), a.z)
    r = m.add_point(a.x + 1.4 * (b.x - a.x), a.y + 1.4 * (b.y - a.y), a.z)
    over = m.add_frame(q, r, f0.section)  # overlapping on the same line
    ar = next(x for x in m.areas.values() if x.type == "Floor")
    X = [P[p].xyz for p in ar.points]
    shifted = m.add_area([m.add_point(x + 1.0, y + 1.0, z) for x, y, z in X], ar.section)
    issues = checks.drawing_issues(m, [dup.name, over.name], [shifted.name])
    codes = sorted(i.code for i in issues)
    assert "DUPLICATE_FRAME" in codes and "OVERLAPPING_FRAMES" in codes and "AREAS_OVERLAP" in codes
    key = lambda i: (i.code, frozenset(i.frames), frozenset(i.areas))  # noqa: E731
    full = {key(i) for i in checks.geometry_issues(m)}
    assert {key(i) for i in issues if i.code in ("DUPLICATE_FRAME", "OVERLAPPING_FRAMES", "AREAS_OVERLAP")} <= full
    # an area far from the changed one is not reported for it
    assert all(shifted.name in i.areas for i in issues if i.code == "AREAS_OVERLAP")
