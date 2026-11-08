"""High-level API: the boundary used by the CLI, the desktop app and agents."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from ..engine import Engine
from ..engine.kernel import Kernel
from ..mesh import MeshOptions, build_analysis_model
from ..model import Model, ModelError
from ..results import Results


def analyze(model: Model, cases: Optional[Iterable[str]] = None, kernel: Optional[Kernel] = None,
            mesh_options: Optional[MeshOptions] = None, progress=None) -> Results:
    """Mesh the model, run the requested (default: all) load cases and return
    a :class:`~structstudio.results.Results` object in the model's units.

    ``progress(percent, message)`` is called as the analysis goes.  Problems
    are reported as :class:`~structstudio.diagnostics.Issue` objects naming the
    object and where it is: ``results.issues`` (warnings / notes), or an
    :class:`~structstudio.diagnostics.AnalysisError` (its ``issues``) when the
    analysis cannot run (an unstable structure, a bad element...)."""
    from .. import diagnostics as dg

    def report(pct, msg):
        if progress is not None:
            progress(float(min(max(pct, 0.0), 100.0)), msg)
    report(0, "Preparing the analysis model")
    fine_note = None
    o = model.options
    if mesh_options is None and getattr(o, "use_fine_mesh", False) and getattr(o, "fine_mesh_factor", None):
        f = min(1.0, float(o.fine_mesh_factor))  # never coarser than the automatic mesh
        mesh_options = MeshOptions(size_factor=f, refine_check=False)
        fine_note = (f"Fine mesh from Analyze > Check Mesh: automatic element size x {f:g}"
                     + ("" if o.fine_mesh_signature == mesh_signature(model) else
                        " - the model has changed since the check: run Check Mesh again to confirm it"))
    try:
        am = build_analysis_model(model, mesh_options, progress=lambda f, msg: report(2 + 23 * f, msg))
    except dg.AnalysisError:
        raise
    except (ModelError, ValueError) as exc:
        raise dg.AnalysisError(_located(model, dg.explain(model, None, exc)), getattr(exc, "code", "MESH_ERROR")) from exc
    am = _refine_until_converged(model, am, mesh_options, kernel, cases, report)
    if fine_note:
        (am.info if "changed" not in fine_note else am.warnings).append(fine_note)
    s = am.summary()
    report(35, f"Mesh: {s['nodes']} nodes, {s['frame_elements']} frame and {s['shell_elements']} shell elements")
    report(36, "Checking connections and loads")
    issues = dg.pre_checks(model, am)
    if any(i.severity == "error" for i in issues):
        raise dg.AnalysisError(_located(model, issues), "UNSTABLE_STRUCTURE")
    engine = None
    try:
        report(38, "Building element matrices")
        engine = getattr(am, "_engine", None)  # from the mesh check: matrices already built
        if engine is not None:
            del am._engine
        else:
            engine = Engine(model, am, kernel)
        names = list(cases) if cases else None

        def case_progress(k, n, name):
            if name:
                report(42 + 55 * k / max(n, 1), f"Solving load case {name} ({k + 1}/{n})")
        case_results = engine.run(names, progress=case_progress)
    except (ModelError, ValueError, ArithmeticError) as exc:
        asm = getattr(engine, "asm", None)
        fac = getattr(engine, "_factor", None)
        raise dg.AnalysisError(_located(model, issues + dg.explain(model, am, exc, asm, fac)),
                               getattr(exc, "code", "ANALYSIS_ERROR")) from exc
    report(98, "Collecting results")
    res = Results(model, am, case_results, engine)
    extra = dg.from_warnings(model, am.warnings)
    for name, r in case_results.items():
        extra += [dg.Issue("warning", f"{name}: {w}") for w in getattr(r, "warnings", [])]
    extra += [dg.Issue("info", t) for t in getattr(am, "info", [])]
    from ..engine import kernel as _k
    solver = f"Equation solver: {_k.describe(engine.kernel)}"
    if _k.KERNEL_NOTE and getattr(engine.kernel, "name", "") == "scipy":
        solver += f" - {_k.KERNEL_NOTE}"
    extra.append(dg.Issue("info", solver))
    res.issues = _located(model, issues + extra)
    report(100, "Analysis complete")
    return res


def mesh_signature(model) -> str:
    """Fingerprint of what the mesh depends on (shell and frame geometry, supports),
    in SI so a change of units does not count as a change."""
    import hashlib
    import json
    k = model.units.factor("length")
    pts = model.points
    r = lambda xyz: [round(float(v) * k, 3) for v in xyz]  # noqa: E731
    data = {
        "areas": sorted([a.name, a.type, [r(pts[q].xyz) for q in a.points]] for a in model.areas.values()),
        "frames": sorted([f.name, r(pts[f.point_i].xyz), r(pts[f.point_j].xyz)] for f in model.frames.values()),
        "supports": sorted([p.name, list(p.restraints)] for p in pts.values() if any(p.restraints)),
        "mesh": [model.options.floor_mesh_max_size, model.options.wall_mesh_max_size],
    }
    from ..mesh import area_mesh as _am
    if _am.AUTO_RULE != "fine":  # the automatic size rule (a check made with another rule is out of date)
        data["auto"] = [_am.AUTO_RULE, _am.ETABS_MAX_SIZE]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def check_mesh(model: Model, kernel: Optional[Kernel] = None, progress=None) -> dict:
    """Analyze > Check Mesh: find the fine mesh of the automatically meshed floors and
    walls once (h-convergence of the largest slab / wall deflection of the linear static
    cases; see :func:`_refine_until_converged`).  Returns ``factor`` (the automatic
    element size x factor, at most 1 - never coarser than the automatic mesh), whether the
    estimate met the allowed error, the levels solved and a text for the user."""
    import dataclasses

    def report(pct, msg):
        if progress is not None:
            progress(float(min(max(pct, 0.0), 100.0)), msg)
    o = model.options
    opts = MeshOptions(refine_check=True, refine_tolerance=float(getattr(o, "mesh_refine_tolerance", 0.02)))
    report(0, "Preparing the analysis model")
    am = build_analysis_model(model, opts, progress=lambda f, msg: report(2 + 20 * f, msg))
    if not am.auto_sizes or not am.shells:
        return {"factor": 1.0, "converged": True, "levels": [], "error": 0.0,
                "text": "No automatically meshed floors / walls: nothing to refine (the mesh sizes are set)"}
    am = _refine_until_converged(model, am, opts, kernel, None,
                                 lambda pct, msg: report(min(95.0, 25 + (pct - 26) * 6), msg))
    chk = getattr(am, "mesh_check", None) or {}
    levels = chk.get("levels") or [(1.0, len(am.shells))]
    factor = min(1.0, float(levels[-1][0]))
    sizes = sorted(set(am.auto_sizes.values()))
    size_txt = f"{sizes[0]:g}" + (f"-{sizes[-1]:g}" if len(sizes) > 1 else "") + " m"
    err = float(chk.get("estimated_error", 0.0))
    ok = bool(chk.get("converged", True))
    step = "h" if factor >= 1.0 else f"h/{round(1 / factor):d}"
    text = (f"Fine mesh: {step} = element size {size_txt}, {len(am.shells)} shell elements; estimated error of the "
            f"largest deflection {100 * err:.2f} % (allowed {100 * opts.refine_tolerance:g} %)"
            + ("" if ok else " - not reached" + (" (stopped at the 12,000 element limit)" if chk.get("capped") else "")))
    report(100, text)
    return {"factor": factor, "converged": ok, "levels": levels, "error": err, "text": text,
            "signature": mesh_signature(model)}


def _shell_peak(model, am, res) -> dict:
    """Largest translation of the nodes of automatically meshed shells, per case."""
    nodes = sorted({n for a in am.auto_sizes for e in am.object_to_shells.get(a, ())
                    for n in am.shells[e].nodes})
    out = {}
    for name, r in res.items():
        if nodes and r.displacements is not None and len(r.displacements):
            out[name] = float(np.max(np.linalg.norm(r.displacements[nodes, :3], axis=1)))
    return out


def _refine_until_converged(model, am, mesh_options, kernel, cases, report):
    """Mesh refinement check (h-convergence) of the automatically meshed shells.

    The linear static cases are solved on the automatic mesh (size h) and on
    meshes twice and four times as coarse (2h, 4h: a quarter and a sixteenth of
    the elements - cheap).  From the three largest shell displacements of each
    case the observed order of convergence p = log2(|u4h - u2h| / |u2h - uh|)
    and the error left at h, |u2h - uh| / (2^p - 1), are estimated
    (Richardson).  No assumed order: near column joints and point loads plates
    converge slower than O(h^2).  When the three values do not approach a limit
    the error is taken as the whole last change.  While the estimate is above
    ``refine_tolerance`` the size is halved (at most ``refine_levels`` times,
    never past ``refine_max_shells`` elements) and the last three meshes are used
    again.  The finest mesh solved is kept; the outcome goes to ``am.mesh_check``
    and to the analysis notes / warnings."""
    import dataclasses
    import math as _m
    if mesh_options is None:  # Analyze > Analysis Model Defaults
        o = model.options
        opts = MeshOptions(refine_check=bool(getattr(o, "mesh_refine_check", False)),
                           refine_tolerance=float(getattr(o, "mesh_refine_tolerance", 0.02)))
    else:
        opts = mesh_options
    if not (opts.refine_check and am.auto_sizes and am.shells):
        return am
    names = [n for n in (list(cases) if cases else list(model.load_cases))
             if type(model.load_cases[n]).__name__ == "LinearStaticCase"]
    if not names:
        return am

    engines = {}

    def solve(a):
        eng = Engine(model, a, kernel)
        eng.displacements_only = True  # no stress resultants: only the deflections are compared
        out = _shell_peak(model, a, eng.run(names))
        engines.clear()  # keep only the latest (finest) one: its matrices serve the analysis proper
        engines[id(a)] = eng
        return out

    def estimate(seq):
        """(relative error at the finest, observed order or None) from the last
        two or three solutions (coarse -> fine)."""
        worst, order = 0.0, None
        for c in seq[-1]:
            vals = [s[c] for s in seq[-3:] if c in s]
            if len(vals) < 2 or abs(vals[-1]) <= 1e-12:
                continue
            d2 = abs(vals[-1] - vals[-2])
            p = None
            if len(vals) == 3:
                d1 = abs(vals[-2] - vals[-3])
                if d2 <= 1e-9 * abs(vals[-1]):
                    p, err = 4.0, 0.0
                elif d1 > d2:
                    p = min(max(_m.log2(d1 / d2), 0.5), 4.0)
                    err = d2 / (2 ** p - 1)
                else:
                    err = d2  # not settling: the whole last change
            else:
                err = d2  # order unknown: the whole last change
            rel = err / abs(vals[-1])
            if rel >= worst:
                worst = rel
                order = p
        return worst, order

    factor = opts.size_factor
    seq, levels = [], []
    try:
        for k, f in enumerate((4 * factor, 2 * factor)):
            report(26 + 2 * k, f"Mesh refinement check: coarser mesh ({round(f / factor):d}h)")
            try:
                a = build_analysis_model(model, dataclasses.replace(opts, size_factor=f))
                seq.append(solve(a))
                levels.append((f, len(a.shells)))
            except Exception:
                if f == 2 * factor:
                    raise
        report(30, "Mesh refinement check: automatic mesh (h)")
        seq.append(solve(am))
        levels.append((factor, len(am.shells)))
    except Exception:
        return am  # the analysis proper reports any problem
    est, order = estimate(seq)
    best, capped = am, False
    for k in range(max(opts.refine_levels, 0)):
        if est <= opts.refine_tolerance:
            break
        factor /= 2
        report(32 + 3 * k, f"Mesh refinement check: element size h/{round(opts.size_factor / factor):d}")
        try:
            fine = build_analysis_model(model, dataclasses.replace(opts, size_factor=factor))
        except Exception:
            break
        if len(fine.shells) > opts.refine_max_shells:
            capped = True
            break
        try:
            seq.append(solve(fine))
        except Exception:
            break
        best = fine
        levels.append((factor, len(fine.shells)))
        est, order = estimate(seq)
    ok = est <= opts.refine_tolerance
    eng = engines.get(id(best))
    if eng is not None:  # the element matrices and the factorised stiffness of the mesh kept
        eng.displacements_only = False
        eng.results = {}
        best._engine = eng
    best.mesh_check = {"levels": levels, "estimated_error": est, "order": order, "converged": ok,
                       "tolerance": opts.refine_tolerance, "capped": capped}
    sizes = sorted(set(best.auto_sizes.values()))
    size_txt = f"{sizes[0]:g}" + (f"-{sizes[-1]:g}" if len(sizes) > 1 else "") + " m"
    p_txt = f"observed order {order:.1f}" if order is not None else "order not established - error taken as the last change"
    if ok:
        best.info.append(f"Mesh check: automatic shell mesh {size_txt} ({len(best.shells)} shells), estimated "
                         f"error of the largest shell displacement {100 * est:.2f} % (limit "
                         f"{100 * opts.refine_tolerance:g} %, {p_txt})")
    else:
        why = f" (not refined further: more than {opts.refine_max_shells} elements)" if capped else ""
        best.warnings.append(f"Mesh check: estimated mesh error of the largest shell displacement is still "
                             f"{100 * est:.1f} % with the automatic mesh {size_txt} ({len(best.shells)} shells, "
                             f"{p_txt}){why} - set a smaller mesh size")
    return best


def _located(model, issues):
    """Fill in coordinates / story / grid for issues that only name an object."""
    from .. import diagnostics as dg
    from ..app.locate import object_xyz
    for i in issues:
        if i.xyz is None and i.attr and i.name:
            X = object_xyz(model, i.attr, i.name)
            if X is not None:
                i.xyz = tuple(float(c) for c in X.mean(axis=0))
        if i.xyz is not None and not i.where:
            i.where = dg._where(model, i.xyz)
    return issues


__all__ = ["analyze", "check_mesh", "mesh_signature"]
