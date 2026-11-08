"""Compare Structural Studio with ETABS / SAFE on the free-form, inclined and
"strange" models (slabs of any shape, columns / walls / loads anywhere on them,
ramps, skew walls, braces through slabs).

    python csi_compare.py build   <folder>   # models -> spec JSON + our results
    python csi_compare.py compare <folder>   # + ETABS / SAFE CSVs -> report.txt

``Compare-With-ETABS-SAFE.cmd`` runs everything: build, then
``Run-CsiCompare.ps1`` builds the same models in ETABS / SAFE through their
official API, runs them and exports the results, then compare.

The spec is a plain description (stories, materials, sections, points,
frames, areas, loads) in kN-m; nothing CSI-specific is read from the
programs' files.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from structstudio.api import analyze  # noqa: E402
from structstudio.app.templates import new_building  # noqa: E402
from structstudio.mesh.mesher import MeshOptions  # noqa: E402

CASE = "Live"
MESH_SIZES = (1.25, 0.5)   # ETABS' default floor mesh is about 1.25 m; 0.5 m = converged


# ------------------------------------------------------------------- models
def _secs(m):
    return (next(n for n in m.frame_sections if n.startswith("C")), next(n for n in m.frame_sections if n.startswith("B")),
            next(n for n in m.shell_sections if n.startswith("S")), next(n for n in m.shell_sections if n.startswith("W")))


def _column(m, col, x, y, z, z0=0.0):
    b = m.add_point(x, y, z0)
    b.restraints = (True,) * 6
    m.add_frame(b, m.add_point(x, y, z), col, type="Column")


def _clean(m):
    """Same model in both programs: no diaphragms (none is assigned in ETABS
    through the API either), loads in Live only, no self weight."""
    for a in m.areas.values():
        a.diaphragm = None
    for p in m.points.values():
        p.diaphragm = None
    for lp in m.load_patterns.values():
        lp.self_weight_multiplier = 0.0
    return m


def model_freeform():
    """Free-form slab on 7 odd columns, a column planted on it carrying a load,
    a curved line load, point loads anywhere, a uniform load."""
    m = new_building(x_spacings=(8,), y_spacings=(8,), story_heights=(4.0, 3.0), grid_only=True)
    col, bm, slab, _ = _secs(m)
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    R = np.array([9, 7, 10, 6, 9.5, 8, 11, 7, 8.5, 6.5, 10, 7.5])
    poly = [(round(R[i] * np.cos(ang[i]) + 0.3 * i, 4), round(0.8 * R[i] * np.sin(ang[i]), 4)) for i in range(12)]
    f = m.add_area([m.add_point(x, y, 4.0) for x, y in poly], slab, type="Floor")
    for xy in [(0.3, 0.2), (4.7, -2.1), (-2.6, -3.6), poly[0], poly[3], poly[6], poly[9]]:
        _column(m, col, *xy, 4.0)
    t = m.add_point(2.7, -1.3, 7.0)
    m.add_frame(m.add_point(2.7, -1.3, 4.0), t, col, type="Column")
    t.restraints = (True, True, False, False, False, False)
    m.add_point_load(t.name, CASE, force=(0, 0, -200))
    pts = [(-6 + 1.1 * k, round(2.5 * np.sin(0.6 * k) - 1, 4)) for k in range(11)]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = m.add_frame(m.add_point(x0, y0, 4.0), m.add_point(x1, y1, 4.0), None, type="Null")
        m.add_frame_distributed_load(n.name, CASE, 6.0)
    for xy in [(1.37, -1.84), (3.13, 3.30), (1.07, 1.84)]:
        m.add_point_load(m.add_point(*xy, 4.0).name, CASE, force=(0, 0, -30))
    m.add_area_load(f.name, CASE, 2.0)
    return _clean(m), ("ETABS", "SAFE")


def model_two_slabs():
    """Two free-form slabs sharing a curved edge, an octagonal opening, a line
    load running from one slab into the other, columns at odd places."""
    m = new_building(x_spacings=(8,), y_spacings=(8,), story_heights=(4.0,), grid_only=True)
    col, bm, slab, _ = _secs(m)
    A = [(5, 9)] + [(round(5 - 5.5 * np.sin(t), 4), round(4.5 + 4.5 * np.cos(t), 4))
                    for t in np.linspace(0.25, np.pi - 0.25, 9)] + [(5, 0)]
    B = [(5, 0), (9, -0.8), (12.5, 1.5), (13.2, 5.0), (11, 8.6), (7.5, 9.8), (5, 9)]
    for poly in (A, B):
        a = m.add_area([m.add_point(x, y, 4.0) for x, y in poly], slab, type="Floor")
        m.add_area_load(a.name, CASE, 3.0)
    oc = [(round(9.5 + 1.2 * np.cos(t), 4), round(4.5 + 0.9 * np.sin(t), 4)) for t in np.linspace(0, 2 * np.pi, 9)[:-1]]
    m.add_area([m.add_point(x, y, 4.0) for x, y in oc], None, type="Opening")
    for xy in [(5, 4.2), oc[2], (2.0, 4.5), (10.5, 8.0), (3.0, 1.5), (12.0, 3.0), (1.0, 6.0), (8, 0.5)]:
        _column(m, col, *xy, 4.0)
    pts = [(1.5 + k, round(7.0 - 0.07 * k * k, 4)) for k in range(10)]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = m.add_frame(m.add_point(x0, y0, 4.0), m.add_point(x1, y1, 4.0), None, type="Null")
        m.add_frame_distributed_load(n.name, CASE, 5.0)
    return _clean(m), ("ETABS", "SAFE")


def model_inclined():
    """Frame building with a skew wall standing on a slab, a ramp with a column
    planted under it mid-ramp, a point and a line load on the ramp, a beam lying
    on the ramp and a brace passing through a slab mid-panel."""
    m = new_building(x_spacings=(8, 8), y_spacings=(8, 8), story_heights=(3.5, 3.5), grid_only=False)
    col, bm, slab, wall = _secs(m)
    z1, z2 = 3.5, 7.0
    m.add_area([m.add_point(2, 2, z1), m.add_point(6.5, 5.3, z1), m.add_point(6.5, 5.3, z2), m.add_point(2, 2, z2)],
               wall, type="Wall")
    m.add_area([m.add_point(0, 16, z1), m.add_point(4, 16, z1), m.add_point(4, 22, z2), m.add_point(0, 22, z2)],
               slab, type="Ramp")
    for x in (0, 4):
        _column(m, col, x, 22, z2)
    m.add_frame(m.add_point(0, 22, z2), m.add_point(4, 22, z2), bm, type="Beam")
    zr = lambda y: round(z1 + (z2 - z1) * (y - 16) / 6, 6)
    _column(m, col, 2.7, 19.3, zr(19.3))
    m.add_point_load(m.add_point(1.1, 17.4, zr(17.4)).name, CASE, force=(0, 0, -25))
    n = m.add_frame(m.add_point(0.3, 18, zr(18)), m.add_point(3.6, 20.5, zr(20.5)), None, type="Null")
    m.add_frame_distributed_load(n.name, CASE, 4.0)
    m.add_frame(m.add_point(0, 16, z1), m.add_point(4, 22, z2), bm, type="Brace")
    a = m.add_point(10, 3, 0)
    a.restraints = (True,) * 6
    m.add_frame(a, m.add_point(14, 6, z2), bm, type="Brace")
    for ar in list(m.areas.values()):
        if ar.type in ("Floor", "Ramp"):
            m.add_area_load(ar.name, CASE, 2.0)
    return _clean(m), ("ETABS",)


MODELS = {"FREEFORM": model_freeform, "TWOSLABS": model_two_slabs, "INCLINED": model_inclined}
# models also sent with rigid zones over the columns (a stiff slab area over each
# column, as SAFE's "Col_Stiff" areas: bending stiffness x 100) - the CSI side of
# a like-for-like check of our column rigid zones
RIGID_ZONE_MODELS = ("FREEFORM", "TWOSLABS")
STIFF_FACTOR = 100.0


def _clip_convex(subject, clip):
    """Sutherland-Hodgman: polygon ``subject`` (any) clipped by the convex ``clip`` (2D, CCW)."""
    out = [tuple(p) for p in subject]
    n = len(clip)
    for k in range(n):
        a, b = np.asarray(clip[k], float), np.asarray(clip[(k + 1) % n], float)
        inside = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12
        src, out = out, []
        for i in range(len(src)):
            P, Q = np.asarray(src[i - 1], float), np.asarray(src[i], float)
            if inside(Q):
                if not inside(P):
                    out.append(_cross_point(P, Q, a, b))
                out.append(tuple(Q))
            elif inside(P):
                out.append(_cross_point(P, Q, a, b))
        if not out:
            break
    return out


def _cross_point(P, Q, a, b):
    d1, d2 = Q - P, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    t = ((a[0] - P[0]) * d2[1] - (a[1] - P[1]) * d2[0]) / den
    return tuple(P + t * d1)


def with_rigid_zones(m, spec: dict) -> dict:
    """The spec plus a stiff slab area over every column end on a floor (the
    column's cross-section, clipped to the slab) for the CSI programs."""
    import copy
    from structstudio.mesh.mesher import Mesher
    from structstudio.mesh import polygon2d as pg
    out = copy.deepcopy(spec)
    out["name"] = spec["name"] + "_RZ"
    floors = [a for a in m.areas.values() if a.type == "Floor" and not a.is_opening]
    if not floors:
        return out
    sec0 = m.shell_sections[floors[0].section]
    out["shell_sections"].append({"name": "COLSTIFF", "material": sec0.material, "t": sec0.thickness,
                                  "kind": "Slab", "shell": sec0.model_type,
                                  "modifiers": [1, 1, 1, STIFF_FACTOR, STIFF_FACTOR, STIFF_FACTOR, 1, 1, 0, 0]})
    k = 0
    # a footprint corner on an existing joint (a column at a slab corner) or on the
    # corner of another footprint (a column on the edge between two slabs) must reuse
    # that joint: ETABS / SAFE merge a coincident new point into the old one and give
    # the old one the new name, which unhooks the slab and the column from it
    known = [(p["name"], p["x"], p["y"], p["z"]) for p in out["points"]]

    def point_at(x, y, z):
        for nm, px, py, pz in known:
            if abs(px - x) < 1e-5 and abs(py - y) < 1e-5 and abs(pz - z) < 1e-5:
                return nm
        return None

    for pt, c, corners, _ in Mesher(m)._column_footprints():
        for a in floors:
            P = np.array([m.points[q].xyz for q in a.points])
            if abs(P[0, 2] - c[2]) > 1e-6 or np.ptp(P[:, 2]) > 1e-6:
                continue
            if not pg.inside(c[:2], P[:, :2], edge_tol=1e-6):
                continue
            box = corners[:, :2]
            if pg.signed_area(box) < 0:
                box = box[::-1]
            poly = _clip_convex(P[:, :2] if pg.signed_area(P[:, :2]) > 0 else P[::-1, :2], box)
            if len(poly) < 3 or abs(pg.signed_area(np.array(poly))) < 1e-6:
                continue
            names = []
            for x, y in poly:
                nm = point_at(float(x), float(y), float(c[2]))
                if nm is None:
                    k += 1
                    nm = f"RZ{k}"
                    out["points"].append({"name": nm, "x": float(x), "y": float(y), "z": float(c[2]),
                                          "restraint": None})
                    known.append((nm, float(x), float(y), float(c[2])))
                if nm not in names:
                    names.append(nm)
            out["areas"].append({"name": f"RZ_{pt}_{a.name}", "points": names, "type": "Floor",
                                 "section": "COLSTIFF", "opening": False})
    return out


# --------------------------------------------------------------------- spec
def to_spec(m, name: str, programs) -> dict:
    """The model as a plain description for Run-CsiCompare.ps1 (kN, m)."""
    assert m.units.force == "kN" and m.units.length == "m"
    used_sec = {f.section for f in m.frames.values() if f.type != "Null"}
    used_sh = {a.section for a in m.areas.values() if not a.is_opening}
    used_mat = {m.frame_sections[s].material for s in used_sec} | {m.shell_sections[s].material for s in used_sh}
    elev = sorted(s.elevation for s in m.stories.values())
    zmax = max(p.z for p in m.points.values())
    if zmax > elev[-1] + 1e-6:
        elev.append(zmax)
    spec = {
        "name": name, "programs": list(programs), "case": CASE,
        "stories": [round(z, 6) for z in elev],
        "materials": [{"name": k, "E": m.materials[k].E, "nu": m.materials[k].nu} for k in sorted(used_mat)],
        "frame_sections": [{"name": s, "material": m.frame_sections[s].material,
                            "t3": m.frame_sections[s].t3, "t2": m.frame_sections[s].t2} for s in sorted(used_sec)],
        "shell_sections": [{"name": s, "material": m.shell_sections[s].material,
                            "t": m.shell_sections[s].thickness, "kind": m.shell_sections[s].type,
                            "shell": m.shell_sections[s].model_type} for s in sorted(used_sh)],
        "points": [], "frames": [], "areas": [], "point_loads": [], "frame_loads": [], "area_loads": [],
        "skipped": [],
    }
    # SAFE models one floor at its datum Z = 0 (the stories above / below are only
    # heights): Run-CsiCompare.ps1 shifts the model by this elevation for SAFE
    zs = [round(m.points[p].z, 6) for a in m.areas.values() if a.type == "Floor" and not a.is_opening
          for p in a.points]
    spec["slab_z"] = max(set(zs), key=zs.count) if zs else 0.0
    for p in m.points.values():
        spec["points"].append({"name": p.name, "x": p.x, "y": p.y, "z": p.z,
                               "restraint": [bool(r) for r in p.restraints] if any(p.restraints) else None})
        for ld in p.loads:
            spec["point_loads"].append({"point": p.name, "pattern": ld.pattern,
                                        "F": list(ld.force) + list(ld.moment)})
    for f in m.frames.values():
        spec["frames"].append({"name": f.name, "i": f.point_i, "j": f.point_j,
                               "section": "None" if f.type == "Null" else f.section, "angle": f.angle,
                               "cardinal": f.cardinal_point})
        for ld in f.loads:
            if ld.kind == "distributed" and ld.direction == "Gravity" and not ld.absolute:
                spec["frame_loads"].append({"frame": f.name, "pattern": ld.pattern, "a": ld.rel_a, "b": ld.rel_b,
                                            "wa": ld.value_a, "wb": ld.value_b})
            else:
                spec["skipped"].append(f"frame load on {f.name} ({ld.kind}, {ld.direction})")
    for a in m.areas.values():
        spec["areas"].append({"name": a.name, "points": list(a.points), "type": a.type,
                              "section": None if a.is_opening else a.section, "opening": a.is_opening})
    for pl in m.panel_loads.values():
        if pl.area and pl.distribution == "shells":
            for lc in pl.loads:
                if lc.direction == "Gravity" and not lc.varying:
                    spec["area_loads"].append({"area": pl.area, "pattern": lc.pattern, "q": lc.value})
                    continue
                spec["skipped"].append(f"area load {pl.name} ({lc.direction})")
        else:
            spec["skipped"].append(f"load panel {pl.name} ({pl.distribution})")
    return spec


def our_results(m, programs) -> dict:
    out = {}
    for h in MESH_SIZES:
        r = analyze(m, [CASE], mesh_options=MeshOptions(floor_max_size=h, wall_max_size=h))
        disp = {n: [float(v) for v in r.joint_displacement(n, CASE)] for n in m.points}
        react = {n: [float(v) for v in r.joint_reaction(n, CASE)] for n, p in m.points.items() if any(p.restraints)}
        out[str(h)] = {"disp": disp, "react": react, "base": [float(v) for v in r.base_reactions(CASE)],
                       "shells": len(r.am.shells), "warnings": list(r.am.warnings)}
    # the automatic mesh with its refinement check and the column rigid zones:
    # the converged answer the other two are judged against
    r = analyze(m, [CASE], mesh_options=MeshOptions(refine_check=True))
    out["auto"] = {"disp": {n: [float(v) for v in r.joint_displacement(n, CASE)] for n in m.points},
                   "react": {n: [float(v) for v in r.joint_reaction(n, CASE)] for n, p in m.points.items()
                             if any(p.restraints)},
                   "base": [float(v) for v in r.base_reactions(CASE)], "shells": len(r.am.shells),
                   "warnings": list(r.am.warnings), "mesh_check": r.am.mesh_check,
                   "sizes": sorted(set(r.am.auto_sizes.values()))}
    return out


def checkpoints(m) -> List[str]:
    """Joints worth comparing: loaded joints, column / wall joints on the slabs,
    the free ends of members (not supports)."""
    names = []
    for n, p in m.points.items():
        if any(p.restraints) and all(p.restraints):
            continue
        if p.loads or any(n in (f.point_i, f.point_j) for f in m.frames.values() if f.type != "Null") \
                or any(n in a.points for a in m.areas.values() if not a.is_opening):
            names.append(n)
    return names


def build(folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    index = []
    for key, fn in MODELS.items():
        m, programs = fn()
        spec = to_spec(m, key, programs)
        spec["checkpoints"] = checkpoints(m)
        with open(os.path.join(folder, f"{key}.spec.json"), "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=1)
        ours = our_results(m, programs)
        with open(os.path.join(folder, f"{key}.ours.json"), "w", encoding="utf-8") as fh:
            json.dump(ours, fh)
        index.append(key)
        if key in RIGID_ZONE_MODELS:
            rz = with_rigid_zones(m, spec)
            for name, data in ((f"{rz['name']}.spec.json", rz), (f"{rz['name']}.ours.json", ours)):
                with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=1 if name.endswith("spec.json") else None)
            index.append(rz["name"])
            print(f"{rz['name']}: the same with {sum(a['section'] == 'COLSTIFF' for a in rz['areas'])} "
                  f"stiff areas over the columns -> {', '.join(programs)}")
        print(f"{key}: {len(spec['points'])} joints, {len(spec['frames'])} frames, {len(spec['areas'])} areas"
              f" -> {', '.join(programs)}; ours: " +
              ", ".join(f"h={h} m {ours[str(h)]['shells']} shells" for h in MESH_SIZES)
              + f", auto {ours['auto']['shells']} shells")
        for s in spec["skipped"]:
            print("   not sent:", s)
    with open(os.path.join(folder, "models.txt"), "w") as fh:
        fh.write("\n".join(index) + "\n")


# ------------------------------------------------------------------ compare
def _read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def csi_results(folder: str, key: str, program: str):
    base = os.path.join(folder, program.lower(), key)
    disp = _read_csv(base + ".joint_displacements.csv")
    react = _read_csv(base + ".joint_reactions.csv")
    breact = _read_csv(base + ".base_reactions.csv")
    if disp is None:
        return None
    pick = lambda rows: [r for r in rows if r.get("Case", "") == CASE] if rows else []
    D = {r["UniqueName"]: [_f(r[k]) for k in ("Ux", "Uy", "Uz", "Rx", "Ry", "Rz")] for r in pick(disp)}
    R = {}
    for r in pick(react):
        v = np.array([_f(r[k]) for k in ("F1", "F2", "F3", "M1", "M2", "M3")])
        R[r["UniqueName"]] = R.get(r["UniqueName"], 0) + v
    B = None
    rows = pick(breact)
    if rows:
        B = [_f(rows[0][k]) for k in ("FX", "FY", "FZ", "MX", "MY", "MZ")]
    return {"disp": D, "react": {k: list(v) for k, v in R.items()}, "base": B}


def compare(folder: str) -> str:
    keys = [k.strip() for k in open(os.path.join(folder, "models.txt")) if k.strip()]
    L: List[str] = []
    w = L.append
    w("Structural Studio vs ETABS / SAFE - free-form and inclined models (load case Live, kN-m)")
    w("=" * 88)
    w("Uz = vertical deflection [mm]; diff = (ours - CSI) / CSI.  'ours h=1.25' uses about the")
    w("default ETABS floor mesh size, 'ours h=0.5' a fine mesh.  The CSI mesh is the program's")
    w("own automatic mesh.  'ours auto' is our automatic mesh with its refinement check and")
    w("the column rigid zones (the converged answer).  Reactions [kN] at every support (ours auto).")
    w("")
    summary = []
    for key in keys:
        spec = json.load(open(os.path.join(folder, f"{key}.spec.json"), encoding="utf-8"))
        ours = json.load(open(os.path.join(folder, f"{key}.ours.json"), encoding="utf-8"))
        fine, coarse = ours[str(MESH_SIZES[-1])], ours[str(MESH_SIZES[0])]
        for prog in spec["programs"]:
            c = csi_results(folder, key, prog)
            head = f"{key} in {prog}" + (" (rigid zones over the columns on both sides)" if key.endswith("_RZ") else "")
            w(head)
            w("-" * len(head))
            if c is None:
                w(f"  no {prog} results (see {prog.lower()}\\{key}.calls.log)")
                w("")
                summary.append((key, prog, None, None, None))
                continue
            total = -fine["base"][2]
            if c["base"] is not None and abs(c["base"][2]) < 1e-6 * max(abs(fine["base"][2]), 1.0):
                w(f"  {prog} carried no load (total reaction 0): its model was not built correctly - "
                  f"not compared (see {prog.lower()}\\{key}.calls.log and {key}.LOG)")
                w("")
                summary.append((key, prog, None, None, None))
                continue
            if c["base"] is not None:
                w(f"  total vertical reaction   ours {fine['base'][2]:12.3f}   {prog} {c['base'][2]:12.3f}   "
                  f"diff {100 * (fine['base'][2] - c['base'][2]) / (abs(c['base'][2]) or 1):+.3f} %")
            missing = [n for n in spec["checkpoints"] if n not in c["disp"]]
            # a joint the CSI program did not connect to anything (all six
            # displacements exactly 0, not a support): not part of its mesh
            fixed = {p["name"] for p in spec["points"] if p.get("restraint")}
            # ... or moving by kilometres: a loaded joint the program left floating
            loose = [n for n in spec["checkpoints"] if n in c["disp"] and n not in fixed
                     and ((not any(c["disp"][n]) and any(fine["disp"].get(n, [0] * 6)))
                          or max(abs(v) for v in c["disp"][n][:3]) > 1e3)]
            pts = [n for n in spec["checkpoints"] if n in c["disp"] and n not in loose]
            if missing:
                w(f"  {len(missing)} joint(s) not in the {prog} output: {', '.join(missing[:12])}")
            if loose:
                w(f"  {len(loose)} joint(s) not connected to the {prog} mesh (zero or huge displacement, not compared): "
                  f"{', '.join(loose[:12])}")
            umax = max((abs(c["disp"][n][2]) for n in pts), default=0.0) or 1.0
            auto = ours.get("auto")
            if auto is not None:
                chk = auto.get("mesh_check") or {}
                est = chk.get("estimated_error")
                w(f"  ours auto = automatic mesh {', '.join(f'{v:g}' for v in auto.get('sizes', []))} m "
                  f"({auto['shells']} shells) with column rigid zones"
                  + (f", estimated mesh error {100 * est:.1f} %" if est is not None else ""))
            w(f"  {'joint':>8} {'x':>8} {'y':>8} {'z':>7} | {'ours h=1.25':>11} {'ours h=0.5':>11} "
              + (f"{'ours auto':>11} " if auto else "") + f"{prog:>11} | "
              f"{'diff h=1.25':>11} {'diff h=0.5':>11}" + (f" {'diff auto':>10}" if auto else ""))
            xyz = {p["name"]: (p["x"], p["y"], p["z"]) for p in spec["points"]}
            worst = []
            for n in pts:
                u_c = c["disp"][n][2] * 1000
                u0, u1 = coarse["disp"][n][2] * 1000, fine["disp"][n][2] * 1000
                ua = auto["disp"][n][2] * 1000 if auto else None
                rel = lambda u: (u - u_c) / (umax * 1000)   # relative to the largest deflection
                worst.append((abs(rel(ua if auto else u1)), n))
                x, y, z = xyz[n]
                w(f"  {n:>8} {x:8.2f} {y:8.2f} {z:7.2f} | {u0:11.4f} {u1:11.4f} "
                  + (f"{ua:11.4f} " if auto else "") + f"{u_c:11.4f} | "
                  f"{100 * rel(u0):+10.2f}% {100 * rel(u1):+10.2f}%" + (f" {100 * rel(ua):+9.2f}%" if auto else ""))
            w("  (diff of each joint relative to the largest deflection, so tiny deflections near")
            w("   supports do not give huge percentages)")
            rws = []
            rmax = max((abs(v[2]) for v in c["react"].values()), default=0.0) or 1.0
            ref = ours.get("auto") or fine
            for n, v in sorted(c["react"].items()):
                if n in ref["react"]:
                    rws.append((n, ref["react"][n][2], v[2]))
            if rws:
                w(f"  {'support':>8} {'Fz ours':>11} {'Fz ' + prog:>11} {'diff':>9}")
                for n, a, b in rws:
                    w(f"  {n:>8} {a:11.3f} {b:11.3f} {100 * (a - b) / rmax:+8.2f}%")
            dmax = max(worst)[0] if worst else None
            rdiff = max((abs(a - b) / rmax for _n, a, b in rws), default=None)
            summary.append((key, prog, dmax, rdiff, total))
            w("")
    w("SUMMARY (ours: automatic mesh with rigid zones when available, else h=0.5)")
    w("-------------------")
    for key, prog, dmax, rdiff, total in summary:
        if dmax is None:
            w(f"  {key:9} {prog:6} no results")
        else:
            rd = "n/a (no matching supports)" if rdiff is None else f"{100 * rdiff:6.2f} %"
            w(f"  {key:9} {prog:6} largest deflection diff {100 * dmax:6.2f} %   largest support reaction diff {rd}")
    text = "\n".join(L) + "\n"
    with open(os.path.join(folder, "report.txt"), "w", encoding="utf-8-sig", newline="\r\n") as fh:
        fh.write(text)
    return text


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "compare"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "build":
        build(sys.argv[2])
    else:
        print(compare(sys.argv[2]))
