"""OpenSees twin of a Structural Studio nonlinear domain (for cross-checks).

The OpenSees model is generated from the *analysis model* (nodes, rigid
diaphragms, lumped masses) and from the engine's nonlinear elements: fiber
sections are exported fiber by fiber, elastic members become
``elasticBeamColumn`` / ``elasticTimoshenkoBeamColumn``, ElasticHingeFrame
members become an elastic element with zero-length rotational springs (Steel01
for bilinear kinematic hinges), links become ``zeroLength`` elements.  The
check therefore covers the nonlinear elements, the constraint handling, the
P-Delta treatment and the solution algorithms, while the linear cross-check
(``compare.py``) covers the mesher.

Supported: frames without joint offsets / rigid zones / releases, joint
loads, fiber force-based (Lobatto, Legendre) and displacement-based members,
materials Steel01, Steel02, Concrete01, Concrete02, ElasticPP, Hysteretic.
"""
from __future__ import annotations

import numpy as np
import openseespy.opensees as ops

from structstudio.engine.nonlinear.elements import DispBeamColumn, ElasticHingeFrame, ForceBeamColumn
from structstudio.engine.nonlinear.materials import ElasticPP, HingeMaterial, Steel01

_MAT_ARGS = {
    "Steel01": lambda m: (m.Fy, m.E0, m.b, m.a1, m.a2, m.a3, m.a4),
    "Steel02": lambda m: (m.Fy, m.E0, m.b, m.R0, m.cR1, m.cR2, m.a1, m.a2, m.a3, m.a4),
    "Concrete01": lambda m: (m.fpc, m.epsc0, m.fpcu, m.epscu),
    "Concrete02": lambda m: (m.fc, m.epsc0, m.fcu, m.epscu, m.rat, m.ft, m.Ets),
    "ElasticPP": lambda m: (m.E, m.epsyP, m.epsyN, m.eps0),
}


class OpenSeesTwin:
    def __init__(self, dom, geometry: str = "PDelta"):
        self.dom = dom
        am = dom.am
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        self.tag = lambda n: n + 1
        M = dom.M_full.reshape(-1, 6)
        for node in am.nodes:
            ops.node(self.tag(node.id), *map(float, node.xyz))
            if np.any(M[node.id]):
                ops.mass(self.tag(node.id), *map(float, M[node.id]))
        masters = {c.master for c in am.constraints if c.rigid}
        for node in am.nodes:
            fix = [int(b) for b in node.restraints]
            if node.id in masters:
                fix = [0, 0, 1, 1, 1, 0]
            if any(fix):
                ops.fix(self.tag(node.id), *fix)
        for c in am.constraints:
            if c.rigid and c.nodes:
                ops.rigidDiaphragm(3, self.tag(c.master), *[self.tag(n) for n in c.nodes])
        self._mat_tag = 0
        self._mats = {}
        self._sec_tag = 0
        self._transf = 0
        self._next_node = len(am.nodes) + 10
        g0 = geometry.lower().replace("-", "")
        geo = "PDelta" if g0 == "pdelta" else ("Corotational" if g0.startswith("corot") else "Linear")
        for fe in dom.asm.frames:
            el = fe.el
            if np.any(el.offset_i) or np.any(el.offset_j) or el.rigid_i or el.rigid_j:
                raise NotImplementedError("offsets / rigid zones are not translated")
            self._transf += 1
            ops.geomTransf(geo, self._transf, *map(float, el.axes[2]))
            i, j = self.tag(el.node_i), self.tag(el.node_j)
            nl = dom.nl_frames.get(el.id)
            if isinstance(nl, (ForceBeamColumn, DispBeamColumn)):
                sec = self._section(nl)
                kind = "Lobatto" if isinstance(nl, ForceBeamColumn) else "Legendre"
                self._sec_tag += 1
                ops.beamIntegration(kind, self._sec_tag, sec, nl.bi.n)
                name = "forceBeamColumn" if isinstance(nl, ForceBeamColumn) else "dispBeamColumn"
                extra = ("-iter", 100, 1e-12) if name == "forceBeamColumn" else ()
                ops.element(name, el.id + 1, i, j, self._transf, self._sec_tag, *extra)
            elif isinstance(nl, ElasticHingeFrame):
                ii, jj = i, j
                for hg in nl.hinges:
                    if hg.dof != "M3":
                        raise NotImplementedError("only M3 hinges are translated")
                    new = self._next_node
                    self._next_node += 1
                    ref = el.node_i if hg.end == "I" else el.node_j
                    ops.node(new, *map(float, am.nodes[ref].xyz))
                    mt = self._hinge_material(hg.material)
                    # the other five DOFs are rigid through stiff springs: an equalDOF from a
                    # diaphragm slave would chain constraints, which OpenSees does not support
                    if not hasattr(self, "_rigid_tag"):
                        self._mat_tag += 1
                        self._rigid_tag = self._mat_tag
                        ops.uniaxialMaterial("Elastic", self._rigid_tag, 1e17)
                    r = self._rigid_tag
                    ops.element("zeroLength", 100000 + new, self.tag(ref), new, "-mat", r, r, r, r, r, mt,
                                "-dir", 1, 2, 3, 4, 5, 6, "-orient", *map(float, el.axes[0]),
                                *map(float, el.axes[1]), "-doRayleigh", 0)
                    if hg.end == "I":
                        ii = new
                    else:
                        jj = new
                ops.element("elasticBeamColumn", el.id + 1, ii, jj, el.A, el.E, el.G, el.J, el.I22, el.I33,
                            self._transf)
            else:
                if el.As2 > 0 and el.As3 > 0:
                    ops.element("ElasticTimoshenkoBeam", el.id + 1, i, j, el.E, el.G, el.A, el.J, el.I22, el.I33,
                                el.As2, el.As3, self._transf)
                else:
                    ops.element("elasticBeamColumn", el.id + 1, i, j, el.A, el.E, el.G, el.J, el.I22, el.I33,
                                self._transf)

    def _material(self, mat) -> int:
        key = (mat.type_name, tuple(sorted((k, v) for k, v in mat.params().items() if not isinstance(v, list))))
        if key in self._mats:
            return self._mats[key]
        self._mat_tag += 1
        t = self._mat_tag
        if mat.type_name == "Hysteretic":
            p = mat.params()
            ops.uniaxialMaterial("Hysteretic", t, p["s1p"], p["e1p"], p["s2p"], p["e2p"], p["s3p"], p["e3p"],
                                 p["s1n"], p["e1n"], p["s2n"], p["e2n"], p["s3n"], p["e3n"], p["pinchX"],
                                 p["pinchY"], p["damage1"], p["damage2"], p["beta"])
        else:
            ops.uniaxialMaterial(mat.type_name, t, *map(float, _MAT_ARGS[mat.type_name](mat)))
        self._mats[key] = t
        return t

    def _hinge_material(self, m) -> int:
        """Bilinear kinematic HingeMaterial -> Steel01."""
        if isinstance(m, HingeMaterial) and m.hysteresis.lower() == "kinematic":
            (d1, f1), (d2, f2) = m.pos[0], m.pos[1]
            k0 = f1 / d1
            b = ((f2 - f1) / (d2 - d1)) / k0
            self._mat_tag += 1
            ops.uniaxialMaterial("Steel01", self._mat_tag, f1, k0, b)
            return self._mat_tag
        return self._material(m)

    def _section(self, nl) -> int:
        blk, idx = nl.secs.blocks[0]
        tpl = blk.template
        self._sec_tag += 1
        t = self._sec_tag
        ops.section("Fiber", t, "-GJ", float(tpl.GJ))
        for mat, y, z, A in tpl.groups:
            mt = self._material(mat)
            for k in range(len(y)):
                ops.fiber(float(y[k]), float(z[k]), float(A[k]), mt)
        return t

    # ------------------------------------------------------------- analysis
    def nodal_pattern(self, tag: int, ts: int, F_full: np.ndarray):
        ops.pattern("Plain", tag, ts)
        F = F_full.reshape(-1, 6)
        for nid in range(len(F)):
            if np.any(F[nid]):
                ops.load(self.tag(nid), *map(float, F[nid]))

    @staticmethod
    def solver(test_tol=1e-10, iters=100):
        ops.system("UmfPack")
        ops.numberer("RCM")
        ops.constraints("Transformation")
        ops.test("NormDispIncr", test_tol, iters)
        ops.algorithm("Newton")

    @staticmethod
    def analyze_robust(dt=None):
        ok = ops.analyze(1) if dt is None else ops.analyze(1, dt)
        for alg in (("NewtonLineSearch",), ("KrylovNewton",), ("ModifiedNewton", "-initial")):
            if ok == 0:
                break
            ops.algorithm(*alg)
            ok = ops.analyze(1) if dt is None else ops.analyze(1, dt)
            ops.algorithm("Newton")
        return ok
