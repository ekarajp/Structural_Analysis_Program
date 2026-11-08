"""SAFE engine check: how close is the Structural Studio analysis to SAFE once the
numbers lost by the SAFE 22 table export are restored?

* decimals at full precision from SAFE's own ``<model>.$sf`` text file (written by
  SAFE next to the model; it keeps decimals but drops the zeros of whole numbers);
* spring stiffnesses exported as 1 (a lost power of ten): 10^n fitted against SAFE's
  displacements at the spring joints (log scale); rotational line springs 10^10.

This is a check of the analysis engine, not an import fix: re-converting the models
with the current ``scripts/Export-SafeSamples.cmd`` gives the true values.

    python scripts/safe_engine_check.py [S01a-Thick S05d-Thin ...]

Columns: worst |Uz| difference over the joints of the structure, relative to the
largest SAFE |Uz| of the case - as imported, with the decimals restored, and with
the springs restored too."""
import csv, glob, os, re, sys, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from structstudio.io.importers import import_model
from structstudio.io.csi_tables import read_tables
from structstudio import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "tests", "verification", "data", "safe")
SF = D  # SAFE writes its .$sf text file next to the model


def norm(k): return re.sub(r"[^a-z0-9]", "", k.lower())


def sf_tables(path):
    """{table: [ {normkey: text} ]} from a .$sf file."""
    out, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\r\n")
        if line.startswith("TABLE:"):
            cur = norm(line.split('"')[1]); out[cur] = []; continue
        if cur is None or not line.strip():
            continue
        row = {}
        for m in re.finditer(r'(?:"([^"]+)"|(\S+?))="([^"]*)"', line):
            row[norm(m.group(1) or m.group(2))] = m.group(3)
        out[cur].append(row)
    return out


def restore_decimals(model, tables_path, sf_path):
    """Area / frame / joint load values: full-precision decimals from the .$sf."""
    sf = sf_tables(sf_path)
    fixed = 0
    # area loads (now panels on areas)
    rows = sf.get(norm("AREA LOAD ASSIGNMENTS - UNIFORM"), [])
    vals = {}
    for r in rows:
        v = r.get("load", "")
        if "." in v or "e" in v.lower():
            vals.setdefault((r.get("uniquename"), r.get("loadpattern")), []).append(float(v))
    for p in model.panel_loads.values():
        for c in p.loads:
            cands = vals.get((p.area, c.pattern), [])
            for v in cands:
                if abs(v - c.value) <= 0.02 * abs(v) and v != c.value:
                    c.value = v; fixed += 1; break
    rows = sf.get(norm("FRAME LOADS ASSIGNMENTS - DISTRIBUTED"), [])
    for r in rows:
        f = model.frames.get(r.get("uniquename"))
        if f is None: continue
        for ld in f.loads:
            for key, attr in (("forcea", "value_a"), ("forceb", "value_b")):
                v = r.get(key, "")
                if ("." in v or "e" in v.lower()) and abs(float(v) - getattr(ld, attr)) <= 0.02 * abs(float(v)):
                    setattr(ld, attr, float(v)); fixed += 1
    rows = sf.get(norm("JOINT LOADS ASSIGNMENTS - FORCE"), [])
    for r in rows:
        p = model.points.get(r.get("uniquename"))
        if p is None: continue
        for ld in p.loads:
            F = list(ld.force)
            for i, key in enumerate(("fx", "fy", "fz")):
                v = r.get(key, "")
                if ("." in v or "e" in v.lower()) and abs(float(v) - F[i]) <= 0.02 * abs(float(v)):
                    F[i] = float(v); fixed += 1
            ld.force = tuple(F)
    return fixed


def safe_disp(name):
    out = {}
    p = os.path.join(D, "results", f"{name}.joint_displacements.csv")
    for r in csv.DictReader(open(p)):
        out.setdefault(r["Case"], {})[r["UniqueName"]] = float(r["Uz"])
    return out


def connected(model):
    used = {n for f in model.frames.values() for n in (f.point_i, f.point_j)}
    for a in model.areas.values():
        used.update(a.points)
    return used


WORST = {}


def error(model, ref, cases):
    try:
        res = analyze(model)
    except Exception as exc:
        return 1e9, None
    used = connected(model)
    worst, where = 0.0, None
    for c in cases:
        if c not in res.cases: continue
        d = {n: v for n, v in ref[c].items() if n in used}  # SAFE also reports free points (interpolated)
        if not d: continue
        big = max(abs(v) for v in d.values()) or 1.0
        for n, v in d.items():
            try:
                o = res.joint_displacement(n, c)[2]
            except Exception:
                continue
            e = abs(o - v) / big
            if e > worst:
                worst, where = e, (c, n, v, o)
    WORST["last"] = where
    return worst, res


def springs(model):
    """{kind: (setters, joints)} for spring values exported as 1 (a lost power of ten);
    joints = model points the springs act on (for the fit)."""
    out = {}
    tol = 1e-6
    def on_line(p, a, b):
        a, b, q = np.asarray(a), np.asarray(b), np.asarray(p)
        d = b - a; L = np.linalg.norm(d)
        t = float((q - a) @ d) / (L * L)
        return -1e-9 <= t <= 1 + 1e-9 and np.linalg.norm(a + t * d - q) <= max(tol, 1e-6 * L)
    for ls in model.line_springs.values():
        a, b = model.points[ls.point_i].xyz, model.points[ls.point_j].xyz
        pts = [p.name for p in model.points.values() if on_line(p.xyz, a, b)]
        if ls.stiffness[2] == 1.0:
            e = out.setdefault("line", ([], set()))
            e[0].append(lambda v, ls=ls: setattr(ls, "stiffness", (0.0, 0.0, v))); e[1].update(pts)
        if getattr(ls, "rotational", 0.0) == 1.0:
            e = out.setdefault("line-rot", ([], set()))
            e[0].append(lambda v, ls=ls: setattr(ls, "rotational", v)); e[1].update(pts)
    for p in model.points.values():
        for i, k in enumerate(p.springs):
            if k == 1.0:
                def setv(v, p=p, i=i):
                    s = list(p.springs); s[i] = v; p.springs = tuple(s)
                e = out.setdefault("point", ([], set())); e[0].append(setv); e[1].add(p.name)
    for a in model.areas.values():
        if a.spring == 1.0:
            e = out.setdefault("area", ([], set()))
            e[0].append(lambda v, a=a: setattr(a, "spring", v)); e[1].update(a.points)
    return out


def log_error(model, ref, cases, joints):
    try:
        res = analyze(model)
    except Exception:
        return 1e9
    errs = []
    for c in cases:
        if c not in res.cases: continue
        for n in joints:
            if n not in ref[c]: continue
            try:
                o = res.joint_displacement(n, c)[2]
            except Exception:
                continue
            errs.append(abs(math.log10(abs(o) + 1e-14) - math.log10(abs(ref[c][n]) + 1e-14)))
    return float(np.mean(errs)) if errs else 0.0


def main(names):
    print(f"{'sample':12s} {'as imported':>12s} {'decimals':>9s} {'+springs':>9s}  springs (10^n)")
    for name in names:
        path = os.path.join(D, f"{name}.SAFE.tables.txt")
        ref = safe_disp(name)
        cases = [c for c in ref if not c.startswith("DCON")]
        m = import_model(path).model
        e0, _ = error(m, ref, cases)
        restore_decimals(m, path, os.path.join(SF, f"{name}.$sf"))
        e1, _ = error(m, ref, cases)
        groups = springs(m)
        best = {}
        for _ in range(2):
            for kind, (setters, joints) in groups.items():
                trial = []
                for n in ((10,) if kind == "line-rot" else range(0, 13)):
                    for st in setters: st(10.0 ** n)
                    trial.append((log_error(m, ref, cases, joints), n))
                e, n = min(trial)
                best[kind] = n
                for st in setters: st(10.0 ** n)
        e2, _ = error(m, ref, cases)
        w = WORST.get("last")
        ws = f"{w[0]} joint {w[1]}: SAFE {w[2]:.4g} ours {w[3]:.4g}" if w else "-"
        print(f"{name:12s} {100*e0:11.2f}% {100*e1:8.2f}% {100*e2:8.2f}%  {best}  worst {ws}")
        sys.stdout.flush()


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(D, "*.SAFE.tables.txt")))
    main(names)
