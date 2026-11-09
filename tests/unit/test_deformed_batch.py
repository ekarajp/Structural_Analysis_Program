"""The batched deformed / mode shape (cached frame sampling and Hermite basis,
array evaluation) against the scalar reference it replaced: same points, same
values, same connectivity - with released members, a plan subset and modes."""
import math

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

from vtkmodules.util import numpy_support as nps  # noqa: E402

from structstudio import analyze  # noqa: E402
from structstudio.app import scene  # noqa: E402
from tests.unit.test_app_scene import small_building  # noqa: E402


def _reference(model, results, case, mode=None, frames=None, areas=None, shells=True, releases=False):
    """The former per-sample loop of build_deformed_mesh_actors (kept as the oracle)."""
    am = results.am
    Lf = results.U.factor("length")
    if mode is not None:
        D = np.asarray(results.mode_shape(case, mode, normalize="max"), float)
        fac = 1.0
    else:
        D = np.asarray(results.displacement_envelope(case).max, float)
        fac, _ = scene.display_length(model)
    U3 = D[:, :3]
    mag = np.linalg.norm(U3, axis=1)
    ext = scene.model_extent(model)
    dmax = float(np.max(np.abs(U3)))
    scale = (0.08 * ext / dmax) if dmax > 0 else 1.0
    X = np.array([n.xyz for n in am.nodes], float) / Lf + scale * U3
    sel_sh = [s for s in am.shells if areas is None or s.object in areas] if shells else []
    sel_fr = [f for f in am.frames if frames is None or f.object in frames]
    used = sorted({n for s in sel_sh for n in s.nodes} | {n for f in sel_fr for n in (f.node_i, f.node_j)})
    idx = {n: k for k, n in enumerate(used)}
    X0 = np.array([n.xyz for n in am.nodes], float) / Lf
    seg = max(ext / 80.0, 1e-9)

    def deflected(f, x):
        a, b = X0[f.node_i], X0[f.node_j]
        L = float(np.linalg.norm(b - a))
        e = (b - a) / L
        di, dj = D[f.node_i, :3], D[f.node_j, :3]
        hinged = bool(np.any(f.releases_i[3:]) or np.any(f.releases_j[3:]))
        ri = np.cross(D[f.node_i, 3:6], e) * L if not hinged else np.zeros(3)
        rj = np.cross(D[f.node_j, 3:6], e) * L if not hinged else np.zeros(3)
        ti, tj = di - (di @ e) * e, dj - (dj @ e) * e
        h1, h2, h3, h4 = 1 - 3 * x * x + 2 * x ** 3, x - 2 * x * x + x ** 3, 3 * x * x - 2 * x ** 3, -x * x + x ** 3
        if hinged:
            h1, h2, h3, h4 = 1 - x, 0.0, x, 0.0
        return ((1 - x) * (di @ e) + x * (dj @ e)) * e + h1 * ti + h2 * ri + h3 * tj + h4 * rj

    fpts, fval, flines = [], [], []
    for f in sel_fr:
        a, b = X0[f.node_i], X0[f.node_j]
        L = float(np.linalg.norm(b - a))
        if L <= 0:
            continue
        n = int(min(10, max(1, math.ceil(L / seg))))
        start = len(fpts)
        for k in range(n + 1):
            x = k / n
            d = deflected(f, x)
            fpts.append(a + x * (b - a) + scale * d)
            fval.append(float(np.linalg.norm(d)) * fac)
        flines.append(list(range(start, len(fpts))))
    rings = []
    if releases:
        by_obj = {}
        for f in sel_fr:
            by_obj.setdefault(f.object, []).append(f)
        for name, els in by_obj.items():
            obj = model.frames.get(name)
            ends = [t for t, rel in ((scene.RELEASE_RING_AT, obj.releases_i), (1 - scene.RELEASE_RING_AT, obj.releases_j))
                    if any(rel[1:3]) or any(rel[3:])]
            for t in ends:
                f = next((el for el in els if el.rel_start - 1e-9 <= t <= el.rel_end + 1e-9), None)
                if f is None:
                    continue
                span = max(f.rel_end - f.rel_start, 1e-12)
                x = min(max((t - f.rel_start) / span, 0.0), 1.0)
                a, b = X0[f.node_i], X0[f.node_j]
                if float(np.linalg.norm(b - a)) <= 0:
                    continue
                rings.append(a + x * (b - a) + scale * deflected(f, x))
    shells_conn = [[idx[n] for n in s.nodes] for s in sel_sh]
    return dict(scale=scale, shell_pts=X[used], shell_vals=mag[used] * fac, shell_conn=shells_conn,
                frame_pts=np.array(fpts).reshape(-1, 3), frame_vals=np.array(fval), frame_lines=flines,
                rings=np.array(rings).reshape(-1, 3))


def _cells(cells):
    out, ids = [], vtk.vtkIdList()
    cells.InitTraversal()
    while cells.GetNextCell(ids):
        out.append([ids.GetId(i) for i in range(ids.GetNumberOfIds())])
    return out


def _check(m, res, case, **kw):
    ref = _reference(m, res, case, **kw)
    actors, scale = scene.build_deformed_mesh_actors(m, res, case, **kw)
    assert scale == pytest.approx(ref["scale"], rel=1e-15)
    polys = [a.GetMapper().GetInput() for a in actors
             if isinstance(a, vtk.vtkActor) and a.GetMapper() is not None
             and a.GetMapper().GetInput() is not None and a.GetMapper().GetLookupTable() is not None]
    sh = [p for p in polys if p.GetNumberOfPolys()]
    fr = [p for p in polys if p.GetNumberOfLines()]
    if ref["shell_conn"]:
        p = sh[0]
        assert np.allclose(nps.vtk_to_numpy(p.GetPoints().GetData()), ref["shell_pts"], rtol=1e-12, atol=1e-12)
        assert np.allclose(nps.vtk_to_numpy(p.GetPointData().GetScalars()), ref["shell_vals"].astype(np.float32))
        assert _cells(p.GetPolys()) == ref["shell_conn"]
    if ref["frame_lines"]:
        p = fr[0]
        assert np.allclose(nps.vtk_to_numpy(p.GetPoints().GetData()), ref["frame_pts"], rtol=1e-12, atol=1e-12)
        assert np.allclose(nps.vtk_to_numpy(p.GetPointData().GetScalars()), ref["frame_vals"].astype(np.float32))
        assert _cells(p.GetLines()) == ref["frame_lines"]
    return actors, ref


@pytest.fixture(scope="module")
def solved():
    m = small_building()
    # a released beam: straight (linear) between hinges, pin symbols on the curve
    beam = next(f for f in m.frames.values() if abs(m.points[f.point_i].z - m.points[f.point_j].z) < 1e-9)
    beam.releases_i = [False, False, False, False, True, True]
    beam.releases_j = [False, False, False, False, True, True]
    return m, analyze(m), beam.name


def test_batched_deflection_matches_the_scalar_reference(solved):
    m, res, beam = solved
    _check(m, res, "DL")
    _check(m, res, "DL", releases=True)
    _, ref = _check(m, res, "DL", releases=True, shells=False)
    assert len(ref["rings"]) == 2


def test_batched_modes_and_subsets_match(solved):
    m, res, beam = solved
    for mode in (1, 2, 3):
        _check(m, res, "Modal", mode=mode)
    _check(m, res, "DL", frames={beam}, areas=set())
    _check(m, res, "DL", frames=set(), areas={"F1"})


def test_plan_is_reused_and_read_only(solved):
    m, res, beam = solved
    p1 = scene._deformed_plan(m, res, None, None, True, False)
    p2 = scene._deformed_plan(m, res, None, None, True, False)
    assert p1 is p2 and not p1.base.flags.writeable and not p1.H.flags.writeable
    p3 = scene._deformed_plan(m, res, {beam}, None, True, False)
    assert p3 is not p1 and len(p3.fe) < len(p1.fe)


def test_plan_cache_keeps_to_its_byte_budget(solved, monkeypatch):
    m, res, beam = solved
    monkeypatch.setattr(scene, "PLAN_CACHE_BYTES", 1)  # every new plan pushes the older ones out
    monkeypatch.setattr(scene, "_PLAN_CACHE", None)
    scene._deformed_plan(m, res, None, None, True, False)
    scene._deformed_plan(m, res, {beam}, None, True, False)
    assert len(scene._PLAN_CACHE) == 1
