"""Automatic mesh size from the geometry and the mesh refinement check."""
import numpy as np

from structstudio.api import analyze
from structstudio.mesh import MeshOptions, build_analysis_model
from structstudio.mesh import area_mesh as am_
from structstudio.model.properties import ShellSection


def test_default_auto_size_is_the_etabs_like_round_value():
    """Default automatic mesh: approximate maximum size 0.5 m (user decision 2026-11: the
    ETABS default 48 in = 1.2192 m is too coarse), floors and walls; a mesh check halves it."""
    assert am_.AUTO_RULE == "etabs" and am_.DEFAULT_MAX_SIZE == 0.5
    sq = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], float)
    strip = np.array([[0, 0], [20, 0], [20, 2], [0, 2]], float)
    cols = am_.Features(points=[(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)])
    for Q, f in ((sq, None), (strip, None), (sq, cols)):
        assert am_.auto_size(Q, f) == 0.5
    assert am_.auto_size(sq, factor=0.5) == 0.25
    assert am_.auto_size(sq, factor=0.25) == 0.125


def test_spans_split_into_equal_elements_up_to_the_size():
    """6 m span -> 12 x 0.5 m, 5 m -> 10 x 0.5 m (mesh lines through the columns)."""
    from structstudio.app.templates import new_building
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m.options.slab_column_rigid_zone = False  # the plain ETABS-like mesh
    am = build_analysis_model(m)
    X = am.coords()
    floor = [s for s in am.shells if m.areas[s.object].type == "Floor"]
    xs = sorted({round(float(X[n][0]), 6) for s in floor for n in s.nodes})
    ys = sorted({round(float(X[n][1]), 6) for s in floor for n in s.nodes})
    assert np.allclose(np.diff(xs), 0.5) and np.allclose(np.diff(ys), 0.5)


def test_fine_auto_size_follows_width_and_spans():
    sq = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], float)
    assert am_.auto_size(sq, rule="fine") == 1.0  # 8 m / 8 elements
    strip = np.array([[0, 0], [20, 0], [20, 2], [0, 2]], float)
    assert am_.auto_size(strip, rule="fine") == 0.25  # 2 m wide -> 0.25, never 2.5
    cols = am_.Features(points=[(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)])
    assert am_.auto_size(sq, cols, rule="fine") == 0.5  # 3 m spans / 6 -> 0.5 (power of two of 1 m)
    assert am_.auto_size(sq, factor=0.5, rule="fine") == 0.5


def test_new_shell_sections_are_thick_plates():
    assert ShellSection("S", "C").model_type == "ShellThick"


def _plate():
    from structstudio.app.templates import new_building
    return new_building(x_spacings=(6,), y_spacings=(6,), story_heights=(3.0,), grid_only=False)


def test_analysis_reports_mesh_check():
    m = _plate()
    m.options.mesh_refine_check = True  # Analyze > Analysis Model Defaults (off by default)
    m.options.mesh_refine_tolerance = 0.01
    r = analyze(m)
    chk = r.am.mesh_check
    assert chk is not None and chk["converged"] and chk["estimated_error"] <= 0.01
    assert any("Mesh check" in i.text() for i in r.issues)
    assert r.am.auto_sizes  # the floor was sized automatically
    # an explicit size switches the automatic sizing (and the check) off
    am2 = build_analysis_model(m, MeshOptions(floor_max_size=1.5))
    assert not am2.auto_sizes


def test_column_rigid_zones_tie_the_slab_inside_the_column():
    m = _plate()
    am = build_analysis_model(m, MeshOptions(floor_max_size=0.1))
    assert am.bodies
    for b in am.bodies:
        c = am.nodes[b.master].xyz
        for n in b.nodes:  # inside the column's cross-section (template columns are <= 0.6 m)
            assert np.all(np.abs(am.nodes[n].xyz[:2] - c[:2]) <= 0.3 + 1e-6)
    m.options.slab_column_rigid_zone = False
    assert not build_analysis_model(m, MeshOptions(floor_max_size=0.1)).bodies


def test_rigid_zones_make_the_column_joint_converge():
    m = _plate()
    case = next(n for n, c in m.load_cases.items() if type(c).__name__ == "LinearStaticCase")
    mid = None
    w = []
    for h in (0.2, 0.1):
        r = analyze(m, [case], mesh_options=MeshOptions(floor_max_size=h, refine_check=False))
        X = np.array([n.xyz for n in r.am.nodes])
        top = X[:, 2].max()
        if mid is None:
            mid = np.array([X[:, 0].mean(), X[:, 1].mean(), top])
        slab = sorted({n for s in r.am.shells for n in s.nodes})  # not the diaphragm's master
        k = slab[int(np.argmin(np.linalg.norm(X[slab] - mid, axis=1)))]
        w.append(r.cases[case].displacements[k][2])
    assert w[0] != 0 and abs(w[1] - w[0]) / abs(w[1]) < 0.02


def test_mesh_check_reports_an_observed_order():
    r = analyze(_plate(), mesh_options=MeshOptions(refine_check=True))
    chk = r.am.mesh_check
    assert len(chk["levels"]) >= 3  # 4h, 2h, h
    assert chk["order"] is None or 0.5 <= chk["order"] <= 4.0


def test_no_refinement_check_by_default_is_fast():
    """ETABS: the automatic mesh is used as it is unless the check is asked for."""
    import time
    m = _plate()
    assert not m.options.mesh_refine_check
    t = time.time()
    r = analyze(m)
    assert r.am.mesh_check is None or not r.am.mesh_check.get("levels")
    assert time.time() - t < 30


def _floor_spacing(m, opts=None):
    am = build_analysis_model(m, opts)
    X = am.coords()
    floor = [s for s in am.shells if m.areas[s.object].type == "Floor"]
    xs = sorted({round(float(X[n][0]), 6) for s in floor for n in s.nodes})
    return round(float(np.max(np.diff(xs))), 6), am


def test_per_area_manual_maximum_and_fine_mesh():
    """Assign > Floor Auto Mesh Options: a manual maximum size and the fine mesh found by
    Find Fine Mesh are used for the areas they are assigned to."""
    from structstudio.app.templates import new_building
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    m.options.slab_column_rigid_zone = False
    floor = next(a for a in m.areas.values() if a.type == "Floor")
    assert _floor_spacing(m)[0] == 0.5
    floor.max_mesh_size = 0.3  # manual maximum mesh size (model units: m)
    assert _floor_spacing(m)[0] <= 0.3 + 1e-9
    floor.max_mesh_size = None
    floor.fine_mesh = True  # fine mesh not found yet: the automatic mesh
    assert _floor_spacing(m)[0] == 0.5
    m.options.fine_mesh_factor = 0.5
    h, am = _floor_spacing(m)
    assert h == 0.25 and am.auto_sizes[floor.name] == 0.25
    # the mesh check looks for the fine mesh itself: the area's fine flag is not applied
    assert _floor_spacing(m, MeshOptions(area_fine=False))[0] == 0.5


def test_insertion_offsets_only_in_the_analysis_when_included():
    """Frame.offsets_in_analysis: the cardinal point and joint offsets are rigid arms in
    the analysis; off = drawn there only (the same results as without them)."""
    import copy
    from structstudio.app.templates import new_building
    base = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3.0,), grid_only=False)
    beams = [f for f in base.frames.values() if f.type == "Beam"]

    def run(include, offset):
        m = copy.deepcopy(base)
        for f in m.frames.values():
            if f.type == "Beam":
                f.cardinal_point, f.display_cardinal = 8, 8
                f.joint_offset_i = f.joint_offset_j = (0.0, offset, 0.0)
                f.joint_offset_local = True
                f.offsets_in_analysis = include
        r = analyze(m)
        case = next(iter(r.cases)) if hasattr(r, "cases") else None
        return r, case
    plain = analyze(copy.deepcopy(base))
    off_r, _ = run(False, -0.3)
    on_r, _ = run(True, -0.3)
    import numpy as np

    case = plain.case_names[0]
    def disp(r):
        return np.array([r.joint_displacement(n, case) for n in sorted(base.points)], float)
    a, b, c = disp(plain), disp(off_r), disp(on_r)
    assert np.allclose(a, b, atol=1e-12) and not np.allclose(a, c, atol=1e-9)
    del beams
