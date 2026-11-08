"""Find the single load (or self weight) responsible for a cross-check difference."""
import copy, sys
import numpy as np
import openseespy.opensees as ops
from structstudio import analyze
from tests.xcheck.generator import random_case
from tests.xcheck import opensees_model as om_mod


def single_load_models(model):
    for pat in model.load_patterns:
        lp = model.load_patterns[pat]
        if lp.self_weight_multiplier:
            m = copy.deepcopy(model)
            for p in m.points.values(): p.loads = []
            for f in m.frames.values(): f.loads = []
            for q, l in m.load_patterns.items(): l.self_weight_multiplier = lp.self_weight_multiplier if q == pat else 0
            yield f"{pat} selfweight x{lp.self_weight_multiplier:.3g}", m, pat
        for pn, p in model.points.items():
            for k, ld in enumerate(p.loads):
                if ld.pattern != pat: continue
                m = copy.deepcopy(model)
                for q in m.points.values(): q.loads = []
                for f in m.frames.values(): f.loads = []
                for l in m.load_patterns.values(): l.self_weight_multiplier = 0
                m.points[pn].loads = [copy.deepcopy(ld)]
                yield f"{pat} joint {pn} F{ld.force} M{ld.moment}", m, pat
        for fn, f in model.frames.items():
            for k, ld in enumerate(f.loads):
                if ld.pattern != pat: continue
                m = copy.deepcopy(model)
                for q in m.points.values(): q.loads = []
                for g in m.frames.values(): g.loads = []
                for l in m.load_patterns.values(): l.self_weight_multiplier = 0
                m.frames[fn].loads = [copy.deepcopy(ld)]
                yield f"{pat} frame {fn} {ld.kind} {ld.load_type} dir={ld.direction} {ld.rel_a:.3g}-{ld.rel_b:.3g} {ld.value_a:.3g}/{ld.value_b:.3g}", m, pat


def err(m, pat):
    res = analyze(m, cases=[pat])
    om = om_mod.build(ops, m, pat)
    om_mod.solve_static(ops)
    e = 0; s = 1e-300
    for n in m.points:
        a = np.asarray(res.joint_displacement(n, pat))[:3]; b = np.array(ops.nodeDisp(om.point_node[n]))[:3]
        e = max(e, np.abs(a - b).max()); s = max(s, np.abs(b).max())
    return e / s


if __name__ == "__main__":
    seed = int(sys.argv[1]); fam = sys.argv[2] if len(sys.argv) > 2 else None
    c = random_case(seed, fam)
    print(c.family, c.notes, {f: (x.angle, x.type) for f, x in c.model.frames.items()})
    for label, m, pat in single_load_models(c.model):
        r = err(m, pat)
        print(f"{r:10.3e}  {label}")
