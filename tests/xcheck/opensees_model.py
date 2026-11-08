"""Independent OpenSees model of a Structural Studio :class:`Model` (3D frames).

Nothing here reuses the Structural Studio mesher: local axes, load
transformation, self weight, projected loads, releases and joint offsets are
re-implemented from the CSI definitions, so a bug in either program shows up
as a difference.

Mapping
-------
* frame  -> ``elasticBeamColumn`` (As2 = As3 = 0) or ``elasticTimoshenkoBeamColumn``;
  OpenSees local y = CSI 2, z = CSI 3 (``vecxz`` = CSI axis 3);
* releases M2 / M3 -> ``-releasey`` / ``-releasez``;
* joint offsets -> ``geomTransf ... -jntOffset`` (rigid links);
* member point forces / moments -> the member is split at the load point and the
  load is applied to the new node (exact);
* uniform / partial loads -> ``-beamUniform`` (partial form); trapezoids -> 64
  partial uniform strips (error ~1e-4);
* point springs -> ``zeroLength`` to a fixed node; point masses -> ``mass``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

STRIPS = 512        # trapezoid strips on one Euler-Bernoulli element (partial uniform loads)
STRIPS_SHEAR = 32   # trapezoid strips as sub-elements of a Timoshenko member
EZ = np.array([0.0, 0.0, 1.0])
EX = np.array([1.0, 0.0, 0.0])


def csi_axes(a: np.ndarray, b: np.ndarray, angle: float) -> np.ndarray:
    """CSI default frame axes (rows 1, 2, 3), rotated by ``angle`` about axis 1."""
    v1 = (b - a) / np.linalg.norm(b - a)
    if np.linalg.norm(np.cross(v1, EZ)) < 1e-3:  # vertical (CSI: within ~0.057 deg)
        v2 = EX.copy()
        v3 = np.cross(v1, v2)
        v3 /= np.linalg.norm(v3)
        v2 = np.cross(v3, v1)
    else:
        v3 = np.cross(v1, EZ)
        v3 /= np.linalg.norm(v3)
        v2 = np.cross(v3, v1)
    if angle:
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        v2, v3 = c * v2 + s * v3, -s * v2 + c * v3
    return np.vstack([v1, v2, v3])


def offset_axes(nominal: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """CSI: with joint offsets the element axis changes; v3~ = v1~ x v2, v2~ = v3~ x v1~."""
    v1 = v1 / np.linalg.norm(v1)
    v3 = np.cross(v1, nominal[1])
    v3 /= np.linalg.norm(v3)
    v2 = np.cross(v3, v1)
    return np.vstack([v1, v2, v3])


def _dir_vector(direction: str):
    d = direction.lower()
    proj = d.endswith("projected")
    if proj:
        d = d[: -len("projected")]
    g = {"gravity": (0, 0, -1), "x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    if d in g:
        return np.array(g[d], float), proj
    return int(d), proj


@dataclass
class MemberMap:
    frame: str
    elements: List[int]          # OpenSees element tags in order from I to J
    stations: List[float]        # split positions (0..L) of the chain nodes
    L: float
    axes: np.ndarray
    timoshenko: bool = False


@dataclass
class OpsModel:
    point_node: Dict[str, int] = field(default_factory=dict)
    members: Dict[str, MemberMap] = field(default_factory=dict)
    spring_elements: List[int] = field(default_factory=list)
    link_masters: set = field(default_factory=set)
    shell_nodes: dict = field(default_factory=dict)


def build(ops, model, pattern: Optional[str], with_mass: bool = False) -> OpsModel:
    """Build the OpenSees domain for one load pattern (``None``: no loads)."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    om = OpsModel()
    tag = [0]

    def new_node(xyz):
        tag[0] += 1
        ops.node(tag[0], *map(float, xyz))
        return tag[0]

    om.sp = []  # (node, dof 1..6, value): imposed ground displacements of this pattern
    for name, p in model.points.items():
        om.point_node[name] = new_node(p.xyz)
        dg = ground_displacement(p, pattern)
        if any(p.restraints):
            fx = [int(bool(r)) for r in p.restraints]
            for d in range(6):
                if fx[d] and dg[d]:
                    fx[d] = 0
                    om.sp.append((om.point_node[name], d + 1, float(dg[d])))
            ops.fix(om.point_node[name], *fx)
    # point loads on members -> split positions
    split: Dict[str, List[float]] = {}
    timo: Dict[str, bool] = {}
    for fname, f in model.frames.items():
        sec = model.frame_sections[f.section]
        released = bool(f.releases_i[4] or f.releases_i[5] or f.releases_j[4] or f.releases_j[5])
        # "timo": loads must cover whole sub-elements - Timoshenko elements have no partial
        # loads, and OpenSees does not condense partial loads of released elements correctly
        timo[fname] = (sec.As2 > 0 and sec.As3 > 0) or released
        a = np.array(model.points[f.point_i].xyz, float)
        b = np.array(model.points[f.point_j].xyz, float)
        L = float(np.linalg.norm(b - a))
        xs = set()
        for ld in f.loads:
            if pattern is None or ld.pattern != pattern:
                continue
            xa = ld.rel_a * L if not ld.absolute else ld.rel_a
            xb = ld.rel_b * L if not ld.absolute else ld.rel_b
            if ld.kind == "point":
                xs.add(round(xa, 12))
            elif not timo[fname]:
                continue  # Euler-Bernoulli element: partial uniform loads on the whole element
            elif ld.value_a == ld.value_b:
                xs |= {round(xa, 12), round(xb, 12)}
            else:  # Timoshenko element (no partial loads in OpenSees): strips as sub-elements
                xs |= {round(float(x), 12) for x in np.linspace(xa, xb, STRIPS_SHEAR + 1)}
        if sec.nonprismatic:  # sub-elements end at the segment boundaries (smooth I inside)
            xs |= {round(float(x), 12) for x in nonprismatic_bounds(sec, L)}
        split[fname] = sorted(x for x in xs if 1e-9 < x < L - 1e-9)

    mat_tag = [0]
    ele = [0]
    transf = [0]
    for fname, f in model.frames.items():
        sec = model.frame_sections[f.section]
        mat = model.materials[sec.material]
        mod = sec.modifiers
        if f.modifiers is not None:
            mod = mod.combined(f.modifiers)
        E = mat.E
        G = mat.shear_modulus
        A, J, I22, I33 = sec.A * mod.A, sec.J * mod.J, sec.I22 * mod.I22, sec.I33 * mod.I33
        As2, As3 = sec.As2 * mod.As2, sec.As3 * mod.As3
        pa = np.array(model.points[f.point_i].xyz, float)
        pb = np.array(model.points[f.point_j].xyz, float)
        nominal = csi_axes(pa, pb, f.angle)
        oi = np.array(f.joint_offset_i, float)
        oj = np.array(f.joint_offset_j, float)
        ea, eb = pa + oi, pb + oj
        axes = offset_axes(nominal, eb - ea) if (oi.any() or oj.any()) else nominal
        L = float(np.linalg.norm(pb - pa))
        stations = [0.0] + split[fname] + [L]
        # joint offsets as rigid links (OpenSees' Timoshenko element ignores -jntOffset;
        # penalty members would ruin the conditioning).  OpenSees does not report the
        # support reaction of a rigid-link master node: those reactions are not compared.
        ni, nj = om.point_node[f.point_i], om.point_node[f.point_j]
        if oi.any():
            n_off = new_node(ea)
            ops.rigidLink("beam", ni, n_off)
            om.link_masters.add(f.point_i)
            ni = n_off
        if oj.any():
            n_off = new_node(eb)
            ops.rigidLink("beam", nj, n_off)
            om.link_masters.add(f.point_j)
            nj = n_off
        chain = [ni]
        for x in split[fname]:
            chain.append(new_node(ea + (eb - ea) * x / L))
        chain.append(nj)
        tags = []
        for k in range(len(chain) - 1):
            transf[0] += 1
            first, last = k == 0, k == len(chain) - 2
            ops.geomTransf("Linear", transf[0], *map(float, axes[2]))
            ele[0] += 1
            rel = []
            rz = (1 if (first and f.releases_i[5]) else 0) + (2 if (last and f.releases_j[5]) else 0)
            ry = (1 if (first and f.releases_i[4]) else 0) + (2 if (last and f.releases_j[4]) else 0)
            if rz:
                rel += ["-releasez", rz]
            if ry:
                rel += ["-releasey", ry]
            if sec.nonprismatic:  # force-based element, elastic sections at 10 Gauss points
                x0, x1 = stations[k], stations[k + 1]
                gx, gw = np.polynomial.legendre.leggauss(10)
                tags_s = []
                for xg in gx:
                    xx = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * xg
                    Ax, Jx, I22x, I33x = nonprismatic_at(model, sec, L, xx)
                    mat_tag[0] += 1
                    ops.section("Elastic", 10000 + mat_tag[0], E, Ax * mod.A, I33x * mod.I33, I22x * mod.I22, G,
                                Jx * mod.J)
                    tags_s.append(10000 + mat_tag[0])
                ops.beamIntegration("UserDefined", ele[0] + 1, 10, *tags_s, *map(float, 0.5 * (gx + 1)),
                                    *map(float, 0.5 * gw))
                ops.element("forceBeamColumn", ele[0], chain[k], chain[k + 1], transf[0], ele[0] + 1)
            elif As2 > 0 and As3 > 0 and not (f.releases_i[4] or f.releases_i[5] or f.releases_j[4] or f.releases_j[5]):
                ops.element("ElasticTimoshenkoBeam", ele[0], chain[k], chain[k + 1], E, G, A, J, I22, I33,
                            As2, As3, transf[0])
            else:
                ops.element("elasticBeamColumn", ele[0], chain[k], chain[k + 1], A, E, G, J, I22, I33, transf[0],
                            *rel)
            tags.append(ele[0])
        om.members[fname] = MemberMap(fname, tags, stations, L, axes, timo[fname])

    # shells: ShellMITC4 (only flat "ShellThick" plates are generated)
    sec_tags: Dict[str, int] = {}
    om.shell_nodes = {}
    for aname, a in model.areas.items():
        sec = model.shell_sections[a.section]
        if sec.name not in sec_tags:
            mat = model.materials[sec.material]
            sec_tags[sec.name] = len(sec_tags) + 1
            ops.section("ElasticMembranePlateSection", sec_tags[sec.name], mat.E, mat.nu, sec.thickness, 0.0)
        nodes = [om.point_node[p] for p in a.points]
        xyz = np.array([model.points[p].xyz for p in a.points], float)
        nrm = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        if nrm[2] < 0:
            nodes, xyz = nodes[::-1], xyz[::-1]
        ele[0] += 1
        ops.element("ShellMITC4", ele[0], *nodes, sec_tags[sec.name])
        om.shell_nodes[aname] = (nodes, xyz)

    # springs (global, to a fixed ground node)
    for name, p in model.points.items():
        k = list(p.springs) + list(p.rotational_springs)
        if any(v > 0 for v in k):
            g = new_node(p.xyz)
            dg = ground_displacement(p, pattern)
            fx = [1] * 6
            for d in range(6):
                if k[d] > 0 and dg[d] and not p.restraints[d]:
                    fx[d] = 0
                    om.sp.append((g, d + 1, float(dg[d])))
            ops.fix(g, *fx)
            mats, dirs = [], []
            for d, v in enumerate(k):
                if v > 0:
                    mat_tag[0] += 1
                    ops.uniaxialMaterial("Elastic", mat_tag[0], float(v))
                    mats.append(mat_tag[0])
                    dirs.append(d + 1)
            ele[0] += 1
            ops.element("zeroLength", ele[0], g, om.point_node[name], "-mat", *mats, "-dir", *dirs)
            om.spring_elements.append(ele[0])
    # rigid diaphragms: master node at the centroid of each (diaphragm, level) group
    groups: Dict[Tuple[str, float], List[str]] = {}
    for name, p in model.points.items():
        if p.diaphragm:
            groups.setdefault((p.diaphragm, round(p.z, 6)), []).append(name)
    for (dname, z), names in groups.items():
        if len(names) < 2:
            continue
        c = np.mean([model.points[n].xyz for n in names], axis=0)
        master = new_node(c)
        ops.fix(master, 0, 0, 1, 1, 1, 0)
        ops.rigidDiaphragm(3, master, *[om.point_node[n] for n in names])
    if with_mass:
        for name, p in model.points.items():
            mm = list(p.mass) + list(p.rotational_mass)
            if any(mm):
                ops.mass(om.point_node[name], *map(float, mm))
    if pattern is not None:
        _loads(ops, model, om, pattern)
    return om


def ground_displacement(p, pattern) -> np.ndarray:
    out = np.zeros(6)
    for gd in getattr(p, "ground_displacements", []):
        if gd.pattern == pattern:
            out += np.concatenate([np.array(gd.translations, float), np.array(gd.rotations, float)])
    return out


def _np_lengths(sec, L):
    rows = sec.nonprismatic
    abs_len = sum(r["length"] for r in rows if r["length_type"] == "Absolute")
    var_sum = sum(r["length"] for r in rows if r["length_type"] != "Absolute")
    lengths = [r["length"] if r["length_type"] == "Absolute" else max(L - abs_len, 0) * r["length"] / var_sum
               for r in rows]
    scale = L / sum(lengths)
    return [ln * scale for ln in lengths]


def nonprismatic_bounds(sec, L: float):
    return list(np.cumsum(_np_lengths(sec, L))[:-1])


def nonprismatic_at(model, sec, L: float, x: float):
    """(A, J, I22, I33) of a nonprismatic section at x along a member of length L."""
    rows = sec.nonprismatic
    abs_len = sum(r["length"] for r in rows if r["length_type"] == "Absolute")
    var_sum = sum(r["length"] for r in rows if r["length_type"] != "Absolute")
    lengths = [r["length"] if r["length_type"] == "Absolute" else max(L - abs_len, 0) * r["length"] / var_sum
               for r in rows]
    scale = L / sum(lengths)
    x0 = 0.0
    order = {"linear": 1, "parabolic": 2, "cubic": 3}
    for r, ln in zip(rows, lengths):
        ln *= scale
        if x <= x0 + ln + 1e-12 or r is rows[-1]:
            xi = min(max((x - x0) / ln, 0.0), 1.0)
            s0, s1 = model.frame_sections[r["start"]], model.frame_sections[r["end"]]

            def lin(a, b):
                return a + (b - a) * xi

            def var(a, b, n):
                return (a ** (1 / n) + (b ** (1 / n) - a ** (1 / n)) * xi) ** n
            return (lin(s0.A, s1.A), lin(s0.J, s1.J), var(s0.I22, s1.I22, order[str(r["ei22"]).lower()]),
                    var(s0.I33, s1.I33, order[str(r["ei33"]).lower()]))
        x0 += ln


def _loads(ops, model, om: OpsModel, pattern: str):
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for node, dof, value in getattr(om, "sp", []):
        ops.sp(node, dof, value)
    lp = model.load_patterns[pattern]
    for name, p in model.points.items():
        for ld in p.loads:
            if ld.pattern == pattern:
                ops.load(om.point_node[name], *map(float, ld.force), *map(float, ld.moment))
    # area loads -> consistent nodal loads (integral of the bilinear shape functions)
    g = 1.0 / math.sqrt(3.0)
    for aname, a in model.areas.items():
        nodes, xyz = om.shell_nodes[aname]
        trib = np.zeros(4)
        for xi in (-g, g):
            for eta in (-g, g):
                N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
                dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                                      [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
                J = dN @ xyz[:, :2]
                trib += N * abs(np.linalg.det(J))
        from structstudio.model.area_panels import area_items
        area_loads = list(a.loads) + [ld for p in model.panel_loads.values()
                                      if p.area == aname and p.distribution == "shells" for ld in area_items(p)]
        for ld in area_loads:
            if ld.pattern != pattern:
                continue
            d, _ = _dir_vector(ld.direction)
            for n, ta in zip(nodes, trib):
                ops.load(n, *map(float, d * ld.value * ta), 0.0, 0.0, 0.0)
    for fname, f in model.frames.items():
        mm = om.members[fname]
        R = mm.axes
        L = mm.L
        # self weight
        if lp.self_weight_multiplier:
            sec = model.frame_sections[f.section]
            mat = model.materials[sec.material]
            mod = sec.modifiers if f.modifiers is None else sec.modifiers.combined(f.modifiers)
            w = lp.self_weight_multiplier * (mat.weight_density * sec.A * mod.weight
                                             + sec.weight_per_length * mod.weight)
            if w:
                q = R @ np.array([0.0, 0.0, -w])
                for e in mm.elements:
                    ops.eleLoad("-ele", e, "-type", "-beamUniform", float(q[1]), float(q[2]), float(q[0]))
        for ld in f.loads:
            if ld.pattern != pattern:
                continue
            d, proj = _dir_vector(ld.direction)
            if ld.kind == "point":
                x = ld.rel_a * L if not ld.absolute else ld.rel_a
                vec = np.zeros(3)
                if isinstance(d, int):
                    vec = R[d - 1] * ld.value_a  # local -> global
                else:
                    vec = d * ld.value_a
                if x <= 1e-9:
                    node = ops_node(om, model, f.point_i)
                elif x >= L - 1e-9:
                    node = ops_node(om, model, f.point_j)
                else:
                    k = mm.stations.index(min(mm.stations, key=lambda s: abs(s - x)))
                    node = ops.eleNodes(mm.elements[k])[0] if k < len(mm.elements) else ops.eleNodes(mm.elements[-1])[1]
                if ld.load_type == "Force":
                    ops.load(node, *map(float, vec), 0.0, 0.0, 0.0)
                else:
                    ops.load(node, 0.0, 0.0, 0.0, *map(float, vec))
                continue
            if ld.load_type != "Force":
                raise NotImplementedError("distributed moments")
            xa = ld.rel_a * L if not ld.absolute else ld.rel_a
            xb = ld.rel_b * L if not ld.absolute else ld.rel_b

            def local(val):
                if isinstance(d, int):
                    v = np.zeros(3)
                    v[d - 1] = val
                    return v
                g = d * val
                if proj:
                    g = g * math.sqrt(max(0.0, 1.0 - float(R[0] @ d) ** 2))
                return R @ g

            pieces = [(xa, xb, ld.value_a)] if ld.value_a == ld.value_b else []
            if not pieces:
                edges = np.linspace(xa, xb, (STRIPS_SHEAR if mm.timoshenko else STRIPS) + 1)
                for s0, s1 in zip(edges[:-1], edges[1:]):
                    t = ((s0 + s1) / 2 - xa) / (xb - xa)
                    pieces.append((s0, s1, ld.value_a + (ld.value_b - ld.value_a) * t))
            for s0, s1, val in pieces:
                q = local(val)
                for k, e in enumerate(mm.elements):
                    e0, e1 = mm.stations[k], mm.stations[k + 1]
                    if mm.timoshenko:  # loads start / end at sub-element nodes
                        mid = 0.5 * (e0 + e1)
                        if s0 - 1e-9 <= mid <= s1 + 1e-9:
                            ops.eleLoad("-ele", e, "-type", "-beamUniform", float(q[1]), float(q[2]), float(q[0]))
                        continue
                    lo, hi = max(s0, e0), min(s1, e1)
                    if hi - lo <= 1e-12:
                        continue
                    Le = e1 - e0
                    ops.eleLoad("-ele", e, "-type", "-beamUniform", float(q[1]), float(q[2]), float(q[0]),
                                float((lo - e0) / Le), float((hi - e0) / Le))


def ops_node(om: OpsModel, model, point: str) -> int:
    return om.point_node[point]


def solve_static(ops) -> bool:
    ops.system("UmfPack")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.integrator("LoadControl", 1.0)
    if ops.getEleTags() and any(ops.eleType(e) == "ForceBeamColumn3d" for e in ops.getEleTags()):
        # force-based elements need element state iterations even when linear
        ops.test("NormDispIncr", 1e-14, 50, 0)
        ops.algorithm("Newton")
    else:
        ops.algorithm("Linear")
    ops.analysis("Static")
    return ops.analyze(1) == 0


def local_forces_csi(ops, ele: int) -> Tuple[np.ndarray, np.ndarray]:
    """CSI section forces [P V2 V3 T M2 M3] at the I and J ends of an OpenSees element."""
    f = np.array(ops.eleResponse(ele, "localForce"), float)
    fi, fj = f[:6], f[6:]
    # OpenSees: end forces on the element in local x y z.  Section force at x=0+
    # is minus the I-end force; at x=L- it is the J-end force.  CSI M2 is the
    # opposite of the right-hand moment about local 2 (= OpenSees y).
    si = -fi
    sj = fj.copy()
    si[4] *= -1.0
    sj[4] *= -1.0
    return si, sj
