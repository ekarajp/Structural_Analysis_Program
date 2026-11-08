"""Compare Structural Studio results with ETABS results exported by
``scripts/Export-VerificationResults.cmd`` (CSV tables in kip-in units).

    python -m structstudio.verification tests/verification/data/etabs

prints, per example, the worst relative difference of joint displacements,
modal periods and base reactions for every load case ETABS reports.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

INCH = 0.0254
KIP = 4448.2216152605


def read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class Comparison:
    example: str
    quantity: str
    case: str
    n: int = 0
    max_abs_ref: float = 0.0
    max_err: float = 0.0  # max |ours - etabs| / max|etabs| over the compared items
    worst: str = ""
    notes: List[str] = field(default_factory=list)

    NUMERICAL_ZERO = 1e-6  # kip, inch: both results below this are equal (round-off)

    @property
    def rel(self) -> float:
        if self.max_err <= self.NUMERICAL_ZERO and self.max_abs_ref <= 1e3 * self.NUMERICAL_ZERO:
            return 0.0
        return self.max_err / self.max_abs_ref if self.max_abs_ref > 0 else 0.0


def _step_ok(row: dict, envelope: bool) -> bool:
    """ETABS StepType: "Single Value" (static / linear-add), "Max"/"Min"
    (spectrum, envelopes), "Mode" (modal cases)."""
    st = (row.get("StepType") or "").strip().lower()
    if envelope:
        return st == "max"
    return st in ("", "single value")


def _object_name(row: dict, names) -> Optional[str]:
    """Our object for a CSI result row: ETABS e2k models use "Label@Story", models
    imported from database tables (ETABS / SAP2000 / SAFE) the unique names."""
    label, story = (row.get("Label") or "").strip(), (row.get("Story") or "").strip()
    if label and story and f"{label}@{story}" in names:
        return f"{label}@{story}"
    u = (row.get("UniqueName") or "").strip()
    return u if u in names else None


def compare_example(name: str, e2k_path: str, results_dir: str, analyze_kw: Optional[dict] = None) -> List[Comparison]:
    """Compare a model file (``.e2k`` or any file :func:`~structstudio.io.import_model`
    reads) with the CSI results ``<results_dir>/<name>.<table>.csv`` (kip-in)."""
    from . import analyze
    from .io import E2KImporter

    if e2k_path.lower().endswith((".e2k", ".$et")):
        model = E2KImporter.read(e2k_path).model
    else:
        from .io import import_model
        model = import_model(e2k_path).model
    # drop cases our engine cannot run (missing spectrum files, time history ...)
    for cname, c in list(model.load_cases.items()):
        for ld in getattr(c, "loads", []) or []:
            fn = getattr(ld, "function", None)
            if fn is not None and fn not in model.functions:
                model.load_cases.pop(cname, None)
                break
    res = analyze(model, **(analyze_kw or {}))
    L = res.U.factor("length") / INCH  # model length -> inch
    F = res.U.factor("force") / KIP
    out: List[Comparison] = []

    # joints that are part of the structure: SAFE also reports free points (design-strip
    # ends ...) with the slab displacement interpolated at their location
    used = {n for f in model.frames.values() for n in (f.point_i, f.point_j)}
    for a in model.areas.values():
        used.update(a.points)
    used.update(n for n, p in model.points.items() if any(p.restraints) or any(p.springs))

    # ---- joint displacements
    path = os.path.join(results_dir, f"{name}.joint_displacements.csv")
    if os.path.exists(path):
        rows = read_csv(path)
        by_case: Dict[str, List[dict]] = {}
        for r in rows:
            by_case.setdefault(r["Case"], []).append(r)
        for case, rr in by_case.items():
            if case not in res.cases and case not in model.load_combinations:
                continue
            r0 = res.cases.get(case)
            envelope = bool(r0 is not None and r0.is_envelope)
            cmp_ = Comparison(name, "joint displacement", case)
            for r in rr:
                st = (r.get("StepType") or "").strip().lower()
                if st == "max":
                    envelope = True
                if not _step_ok(r, envelope):
                    continue
                point = _object_name(r, res.am.point_to_node)
                if point is None or point not in used:
                    continue
                ours = res.joint_displacement(point, case)
                nd = res.am.nodes[res.am.point_to_node[point]]
                if nd.local_axes is not None:  # CSI reports joint results in the joint local axes
                    ours = np.concatenate([nd.local_axes @ ours[:3], nd.local_axes @ ours[3:]])
                ref = np.array([_f(r[k]) for k in ("Ux", "Uy", "Uz", "Rx", "Ry", "Rz")])
                mine = np.concatenate([ours[:3] * L, ours[3:]])
                if envelope:
                    mine, ref = np.abs(mine), np.abs(ref)
                for k in range(3):  # translations decide the comparison
                    if np.isnan(ref[k]):
                        continue
                    cmp_.n += 1
                    cmp_.max_abs_ref = max(cmp_.max_abs_ref, abs(ref[k]))
                    err = abs(mine[k] - ref[k])
                    if err > cmp_.max_err:
                        cmp_.max_err = err
                        cmp_.worst = f"{point} U{'xyz'[k]}: ours {mine[k]:.6g} vs ETABS {ref[k]:.6g}"
            if cmp_.n:
                out.append(cmp_)

    # ---- modal periods
    path = os.path.join(results_dir, f"{name}.modal_periods.csv")
    if os.path.exists(path):
        rows = read_csv(path)
        by_case = {}
        for r in rows:
            by_case.setdefault(r["Case"], []).append(r)
        for case, rr in by_case.items():
            if case not in res.cases or res.cases[case].modal is None:
                continue
            ours = [row["Period"] for row in res.modal_periods(case)]
            cmp_ = Comparison(name, "period", case)
            for r in rr:
                k = int(round(_f(r["Mode"]))) - 1
                if k < 0 or k >= len(ours):
                    continue
                ref = _f(r["Period"])
                cmp_.n += 1
                cmp_.max_abs_ref = max(cmp_.max_abs_ref, abs(ref))
                rel = abs(ours[k] - ref)
                # periods: relative to each period, not to the largest one
                rel_k = rel / abs(ref) * cmp_.max_abs_ref if ref else 0.0
                if rel_k > cmp_.max_err:
                    cmp_.max_err = rel_k
                    cmp_.worst = f"mode {k + 1}: ours {ours[k]:.6g} vs ETABS {ref:.6g}"
            if cmp_.n:
                out.append(cmp_)

    # ---- frame internal forces at the ETABS output stations
    path = os.path.join(results_dir, f"{name}.frame_forces.csv")
    if os.path.exists(path):
        rows = read_csv(path)
        by_case = {}
        for r in rows:
            by_case.setdefault(r["Case"], []).append(r)
        M = F * L  # model moment -> kip-in
        for case, rr in by_case.items():
            if case not in res.cases and case not in model.load_combinations:
                continue
            r0 = res.cases.get(case)
            if r0 is not None and r0.modal is not None and not r0.is_envelope:
                continue  # modal case: mode-by-mode forces not compared
            cmp_ = Comparison(name, "frame forces", case)
            cache = {}
            for r in rr:
                st = (r.get("StepType") or "").strip().lower()
                envelope = st in ("max", "min")
                if st not in ("max", "single value", ""):
                    continue
                frame = _object_name(r, model.frames)
                if frame is None:
                    continue
                x = _f(r["ObjSta"]) / L  # inch -> model length
                # a station listed twice (point load, element end) is the value just
                # before and just after the discontinuity
                key = (frame, round(x, 9))
                k_seen = cache.get(key, 0)
                cache[key] = k_seen + 1
                fl = model.frames[frame]
                Lf = float(np.linalg.norm(np.subtract(model.points[fl.point_j].xyz, model.points[fl.point_i].xyz)))
                dup = sum(1 for q in rr if _object_name(q, model.frames) == frame
                          and abs(_f(q["ObjSta"]) / L - x) < 1e-9 * max(Lf, 1.0)
                          and (q.get("StepType") or "") == (r.get("StepType") or "")) > 1
                if dup:
                    x = x - 1e-7 * Lf if k_seen == 0 else x + 1e-7 * Lf
                    x = min(max(x, 0.0), Lf)
                try:
                    mine_raw = res.frame_forces_at(frame, case, [x]).forces[0]
                except Exception:
                    continue
                mine = mine_raw * np.array([F, F, F, M, M, M])
                ref = np.array([_f(r[c]) for c in ("P", "V2", "V3", "T", "M2", "M3")])
                if envelope:
                    mine, ref = np.abs(mine), np.abs(ref)
                for j, lab in enumerate(("P", "V2", "V3", "T", "M2", "M3")):
                    if np.isnan(ref[j]):
                        continue
                    cmp_.n += 1
                    scale_key = j >= 3
                    cmp_.max_abs_ref = max(cmp_.max_abs_ref, abs(ref[j]) if not scale_key else 0.0)
                    cmp_.notes.append("")  # placeholder to keep the dataclass simple
                    err = abs(mine[j] - ref[j])
                    # moments are judged relative to the largest moment, forces to the largest force
                    cmp_._m = max(getattr(cmp_, "_m", 0.0), abs(ref[j]) if scale_key else 0.0)
                    key = "_em" if scale_key else "_ef"
                    if err > getattr(cmp_, key, 0.0):
                        setattr(cmp_, key, err)
                        setattr(cmp_, key + "w", f"{frame} x={_f(r['ObjSta']):g} {lab}: ours {mine[j]:.6g} vs ETABS {ref[j]:.6g}")
            if cmp_.n:
                # numerical zeros: moments are judged against at least 1e-3 x (max force x 1 in)
                cmp_.max_abs_ref = max(cmp_.max_abs_ref, 1e-9)
                cmp_._m = max(getattr(cmp_, "_m", 0.0), 1e-3 * cmp_.max_abs_ref, 1e-9)
                rf = getattr(cmp_, "_ef", 0.0) / cmp_.max_abs_ref
                rm = getattr(cmp_, "_em", 0.0) / cmp_._m
                cmp_.notes = []
                if rf >= rm:
                    cmp_.max_err, cmp_.worst = getattr(cmp_, "_ef", 0.0), getattr(cmp_, "_efw", "")
                else:
                    cmp_.max_err, cmp_.max_abs_ref, cmp_.worst = cmp_._em, cmp_._m, getattr(cmp_, "_emw", "")
                out.append(cmp_)

    # ---- base reactions
    path = os.path.join(results_dir, f"{name}.base_reactions.csv")
    if os.path.exists(path):
        for r in read_csv(path):
            case = r["Case"]
            if case not in res.cases or case.startswith("~"):
                continue
            r0 = res.cases[case]
            if r0.modal is not None and r0.type.lower().startswith("modal"):
                continue
            envelope = r0.is_envelope or (r.get("StepType") or "").strip().lower() in ("max", "min")
            if not _step_ok(r, envelope):
                continue
            ours = res.base_reactions(case)
            ref = np.array([_f(r[k]) for k in ("FX", "FY", "FZ")])
            mine = ours[:3] * F
            if envelope:
                mine, ref = np.abs(mine), np.abs(ref)
            cmp_ = Comparison(name, "base reaction", case, n=3, max_abs_ref=float(np.nanmax(np.abs(ref))) if len(ref) else 0)
            err = np.abs(mine - ref)
            k = int(np.nanargmax(err))
            cmp_.max_err = float(err[k])
            cmp_.worst = f"F{'XYZ'[k]}: ours {mine[k]:.6g} vs ETABS {ref[k]:.6g}"
            out.append(cmp_)
    return out


MODEL_SUFFIXES = (".e2k", ".tables.txt")


def _models(data_dir: str) -> Dict[str, str]:
    """name -> model file; ``Example 1-001.SAP2000.tables.txt`` -> ``Example1-001``."""
    out = {}
    for f in sorted(os.listdir(data_dir)):
        for suf in MODEL_SUFFIXES:
            if f.endswith(suf):
                base = f[: -len(suf)]
                if suf == ".tables.txt" and "." in base:
                    base = base.rsplit(".", 1)[0]
                out.setdefault(base.replace(" ", ""), os.path.join(data_dir, f))
    return out


def run(data_dir: str, examples: Optional[List[str]] = None, results_dir: Optional[str] = None) -> List[Comparison]:
    results_dir = results_dir or os.path.join(data_dir, "results")
    models = _models(data_dir)
    names = examples or sorted(models)
    out: List[Comparison] = []
    for n in names:
        e2k = models.get(n)
        if not e2k or not os.path.exists(e2k):
            continue
        try:
            out += compare_example(n, e2k, results_dir)
        except Exception as exc:  # report, keep going
            c = Comparison(n, "ERROR", "-")
            c.notes.append(f"{type(exc).__name__}: {exc}")
            out.append(c)
    return out


def format_table(rows: List[Comparison]) -> str:
    lines = [f"{'example':12s} {'quantity':20s} {'case':14s} {'n':>5s} {'rel.diff':>9s}  worst"]
    for c in rows:
        if c.quantity == "ERROR":
            lines.append(f"{c.example:12s} {'ERROR':20s} {'':14s} {'':>5s} {'':>9s}  {'; '.join(c.notes)[:120]}")
            continue
        lines.append(f"{c.example:12s} {c.quantity:20s} {c.case[:14]:14s} {c.n:5d} {100 * c.rel:8.3f}%  {c.worst}")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    results = None
    if "--results" in argv:
        k = argv.index("--results")
        results = argv[k + 1]
        del argv[k:k + 2]
    data_dir = argv[0] if argv else os.path.join("tests", "verification", "data", "etabs")
    rows = run(data_dir, argv[1:] or None, results)
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
