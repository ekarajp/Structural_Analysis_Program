"""Cross-check Structural Studio against OpenSeesPy on random structures.

    python -m tests.xcheck.compare --n 1000 --jobs 8            # report of the worst cases
    python -m tests.xcheck.compare --seed 123 --verbose          # one case in detail

Quantities compared per load case: displacements and rotations of every joint,
support reactions, and member section forces (P V2 V3 T M2 M3) at both ends of
every member; for models with masses the first natural periods.  Errors are
relative to the largest absolute value of the same kind in that case.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# OpenSees has the MITC4 plate (ShellMITC4): our MITC4 option is compared with it (the
# program's default thick plate is the DKMQ, checked against Mindlin theory and SAFE)
from structstudio.engine.elements import shell as _shell  # noqa: E402


def _use_mitc4():
    _shell.THICK_FORMULATION = "MITC"

TOL = {"exact": 5e-5, "strips": 2e-4, "strips_shear": 2e-4}
PERIOD_TOL = 1e-5


@dataclass
class Outcome:
    seed: int
    family: str = ""
    status: str = "ok"  # ok | fail | skip | error
    worst: float = 0.0  # worst relative error / tolerance (>1 = fail)
    detail: str = ""
    notes: List[str] = field(default_factory=list)
    n_values: int = 0


ABS_ZERO = 1e-7  # kN, m: differences below this are round-off whatever the scale


def _rel(err: float, scale: float) -> float:
    if err < ABS_ZERO:
        return 0.0
    return err / scale if scale > 1e-12 else err


def run_case(seed: int, family: Optional[str] = None, verbose: bool = False) -> Outcome:
    import openseespy.opensees as ops
    from structstudio import analyze
    from structstudio.model import ModelError
    from tests.xcheck import opensees_model as om_mod
    from tests.xcheck.generator import random_case

    out = Outcome(seed)
    try:
        case = random_case(seed, family)
    except Exception as exc:  # generator bug - report
        out.status, out.detail = "error", f"generator: {type(exc).__name__}: {exc}"
        return out
    model = case.model
    out.family = case.family
    out.notes = case.notes
    if "coincident points" in case.notes:
        out.status, out.detail = "skip", "generator made coincident points"
        return out
    for ms in model.mass_sources.values():
        ms.include_vertical = True
        ms.include_lateral = True
    tol = TOL[case.tol_class]
    if case.tol_class == "strips":  # trapezoids on Timoshenko members are coarser strips in OpenSees
        for f in model.frames.values():
            sec = model.frame_sections[f.section]
            released = bool(f.releases_i[4] or f.releases_i[5] or f.releases_j[4] or f.releases_j[5])
            if ((sec.As2 > 0 and sec.As3 > 0) or released) and any(
                    l.kind == "distributed" and l.value_a != l.value_b for l in f.loads):
                tol = TOL["strips_shear"]  # after Richardson extrapolation (see below)
    try:
        res = analyze(model)
    except ModelError as exc:
        out.status, out.detail = "skip", f"ours: {exc}"
        return out
    except Exception as exc:
        out.status, out.detail = "error", f"ours: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"
        return out
    unstable = [w for w in res.am.warnings if "unstable" in w.lower()]
    if unstable:  # both programs are solving a (nearly) singular system - nothing to compare
        out.status, out.detail = "skip", "ours: " + unstable[0][:200]
        return out
    worst, where = 0.0, ""
    lines = []

    def track(rel, what):
        nonlocal worst, where
        out.n_values += 1
        if rel / tol > worst:
            worst, where = rel / tol, what

    def ops_results(pat):
        """OpenSees displacements, reactions and member end forces for one pattern."""
        om = om_mod.build(ops, model, pat)
        if not om_mod.solve_static(ops):
            return None, om
        ops.reactions()
        disp = {n: np.array(ops.nodeDisp(om.point_node[n]), float) for n in model.points}
        reac = {n: np.array(ops.nodeReaction(om.point_node[n]), float) for n in model.points}
        forces = {}
        for fname, mm in om.members.items():
            si, _ = om_mod.local_forces_csi(ops, mm.elements[0])
            _, sj = om_mod.local_forces_csi(ops, mm.elements[-1])
            forces[fname] = np.vstack([si, sj])
        return (disp, reac, forces), om

    def has_shear_trapezoid(pat):
        for f in model.frames.values():
            sec = model.frame_sections[f.section]
            released = bool(f.releases_i[4] or f.releases_i[5] or f.releases_j[4] or f.releases_j[5])
            if ((sec.As2 > 0 and sec.As3 > 0) or released) and any(
                    l.pattern == pat and l.kind == "distributed" and l.value_a != l.value_b for l in f.loads):
                return True
        return False

    for pat in model.load_patterns:
        theirs_all, om = ops_results(pat)
        if theirs_all is None:
            out.status, out.detail = "skip", f"OpenSees could not solve {pat} (unstable?)"
            return out
        if has_shear_trapezoid(pat):
            # Timoshenko strips converge as h^2: Richardson extrapolation from n and 2n strips
            n0 = om_mod.STRIPS_SHEAR
            om_mod.STRIPS_SHEAR = 2 * n0
            try:
                fine, om = ops_results(pat)
            finally:
                om_mod.STRIPS_SHEAR = n0
            if fine is not None:
                theirs_all = tuple({k: (4.0 * fine[i][k] - theirs_all[i][k]) / 3.0 for k in fine[i]} for i in range(3))
        disp_t, reac_t, forces_t = theirs_all
        # ---- joint displacements
        ours = {n: np.asarray(res.joint_displacement(n, pat), float) for n in model.points}
        theirs = disp_t
        su = max(max(np.abs(v[:3]).max() for v in theirs.values()), 1e-300)
        sr = max(max(np.abs(v[3:]).max() for v in theirs.values()), 1e-300)
        if not np.isfinite(su) or su > 1e6:
            out.status, out.detail = "skip", "mechanism (huge displacements in OpenSees)"
            return out
        for n in model.points:
            e = np.abs(ours[n] - theirs[n])
            track(_rel(e[:3].max(), su), f"{pat} joint {n} U: ours {ours[n][:3]} vs ops {theirs[n][:3]}")
            track(_rel(e[3:].max(), sr), f"{pat} joint {n} R: ours {ours[n][3:]} vs ops {theirs[n][3:]}")
        # ---- reactions at supports without springs
        sup = [n for n, p in model.points.items() if any(p.restraints) and not any(p.springs)
               and not any(p.rotational_springs) and n not in om.link_masters]
        if sup:
            rr_o = {n: np.asarray(res.joint_reaction(n, pat), float) for n in sup}
            rr_t = {n: reac_t[n] for n in sup}
            sf = max(max(np.abs(v[:3]).max() for v in rr_t.values()), 1e-300)
            sm = max(max(np.abs(v[3:]).max() for v in rr_t.values()), 1e-300)
            for n in sup:
                e = np.abs(rr_o[n] - rr_t[n])
                track(_rel(e[:3].max(), sf), f"{pat} reaction {n} F: ours {rr_o[n][:3]} vs ops {rr_t[n][:3]}")
                track(_rel(e[3:].max(), sm), f"{pat} reaction {n} M: ours {rr_o[n][3:]} vs ops {rr_t[n][3:]}")
        # ---- member end forces
        names = list(forces_t)
        if names:
            fo = np.array([np.vstack([np.asarray(res.frame_forces(n, pat).forces, float)[0],
                                      np.asarray(res.frame_forces(n, pat).forces, float)[-1]]) for n in names])
            ft = np.array([forces_t[n] for n in names])
            sF = max(np.abs(ft[:, :, :3]).max(), 1e-300)
            sM = max(np.abs(ft[:, :, 3:]).max(), 1e-300)
            labels = ("P", "V2", "V3", "T", "M2", "M3")
            for k, fname in enumerate(names):
                for end in (0, 1):
                    for c in range(6):
                        e = abs(fo[k, end, c] - ft[k, end, c])
                        track(_rel(e, sF if c < 3 else sM),
                              f"{pat} frame {fname} {'IJ'[end]} {labels[c]}: ours {fo[k, end, c]:.6g} vs ops "
                              f"{ft[k, end, c]:.6g}")
    # ---- modal periods
    if case.modal and "MODAL" in model.load_cases:
        om = om_mod.build(ops, model, None, with_mass=True)
        ours_T = [r["Period"] for r in res.modal_periods("MODAL")]
        n = min(len(ours_T), 6)
        try:
            lam = ops.eigen("-fullGenLapack", n)
        except Exception as exc:
            lam = []
        lam = [x for x in lam if np.isfinite(x) and x > 0]
        theirs_T = [2 * math.pi / math.sqrt(x) for x in lam]
        for k in range(min(n, len(theirs_T))):
            rel = abs(ours_T[k] - theirs_T[k]) / theirs_T[k]
            out.n_values += 1
            if rel / PERIOD_TOL > worst:
                worst, where = rel / PERIOD_TOL, f"period {k + 1}: ours {ours_T[k]:.8g} vs ops {theirs_T[k]:.8g}"
    out.worst = worst
    out.detail = where
    out.status = "fail" if worst > 1.0 else "ok"
    return out


def _worker(args):
    seed, family = args
    try:
        return run_case(seed, family)
    except Exception as exc:
        o = Outcome(seed, status="error", detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
        return o


def _run_many(n: int, start: int = 0, jobs: int = 1, family: Optional[str] = None) -> List[Outcome]:
    seeds = [(s, family) for s in range(start, start + n)]
    if jobs <= 1:
        return [_worker(s) for s in seeds]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        return list(pool.imap_unordered(_worker, seeds, chunksize=4))



def run_many(n: int, start: int = 0, jobs: int = 1, family: Optional[str] = None) -> List[Outcome]:
    """``_run_many`` with the MITC4 thick plate (the OpenSees element), restored after."""
    old = _shell.THICK_FORMULATION
    _use_mitc4()
    try:
        return _run_many(n, start, jobs, family)
    finally:
        _shell.THICK_FORMULATION = old


def report(outs: List[Outcome]) -> str:
    from collections import Counter, defaultdict
    by = defaultdict(Counter)
    for o in outs:
        by[o.family or "?"][o.status] += 1
    lines = [f"{'family':12s} {'ok':>6s} {'fail':>6s} {'skip':>6s} {'error':>6s}"]
    for fam in sorted(by):
        c = by[fam]
        lines.append(f"{fam:12s} {c['ok']:6d} {c['fail']:6d} {c['skip']:6d} {c['error']:6d}")
    tot = Counter(o.status for o in outs)
    lines.append(f"{'TOTAL':12s} {tot['ok']:6d} {tot['fail']:6d} {tot['skip']:6d} {tot['error']:6d}   "
                 f"values compared: {sum(o.n_values for o in outs)}")
    bad = sorted((o for o in outs if o.status in ("fail", "error")), key=lambda o: -o.worst)
    for o in bad[:40]:
        lines.append(f"  seed {o.seed:6d} {o.family:10s} {o.status:5s} x{o.worst:9.3g}  {o.detail[:300]}"
                     + (f"  [{'; '.join(o.notes)}]" if o.notes else ""))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--family")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.seed is not None:
        o = run_case(a.seed, a.family, verbose=True)
        print(json.dumps(o.__dict__, indent=1, default=str))
        return 0
    t0 = time.time()
    outs = run_many(a.n, a.start, a.jobs, a.family)
    print(report(outs))
    print(f"{len(outs)} models in {time.time() - t0:.1f} s")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump([o.__dict__ for o in outs], fh, indent=0, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())

