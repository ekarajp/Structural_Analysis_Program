"""Build nonlinear elements from the object model's nonlinear assignments."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...mesh.geometry import frame_axes
from ...model.base import ModelError
from . import hinges as H
from . import sections as S
from .elements import DispBeamColumn, ElasticHingeFrame, ForceBeamColumn, FrameHinge, Link, NLFrame
from .materials import (BoucWen, Elastic, ElasticPP, HingeMaterial, UniaxialMaterial, Viscous, make_material,
                        Parallel)

_DOF_INDEX = {"U1": 0, "U2": 1, "U3": 2, "R1": 3, "R2": 4, "R3": 5}


class NonlinearBuilder:
    def __init__(self, model, am, asm):
        self.model, self.am, self.asm = model, am, asm
        self.U = model.units
        self._mat: Dict[str, UniaxialMaterial] = {}
        self._sec: Dict[str, S.FiberSection3D] = {}
        self.warnings: List[str] = []
        # load drops (CSI C-D, after E) are given a slope of k_member / drop_regularisation
        self.drop_regularisation = 10.0

    # ---------------------------------------------------------------- pieces
    def material(self, name: str) -> UniaxialMaterial:
        if name not in self._mat:
            nm = self.model.nonlinear_materials.get(name)
            if nm is None:
                raise ModelError("UNKNOWN_NL_MATERIAL", f"Nonlinear material {name!r} is not defined")
            self._mat[name] = make_material(nm.type, nm.si_params(self.U))
            w = getattr(self._mat[name], "warning", None)
            if w:
                self.warnings.append(f"Material {name}: {w}")
        return self._mat[name]

    def fiber_section(self, name: str, fe=None) -> S.FiberSection3D:
        key = (name, None if fe is None else id(fe))
        if name in self._sec and fe is None:
            return self._sec[name]
        d = self.model.fiber_sections.get(name)
        if d is None:
            raise ModelError("UNKNOWN_FIBER_SECTION", f"Fiber section {name!r} is not defined")
        L = self.U.factor("length")
        A2 = L * L
        fs = self.model.frame_sections.get(d.frame_section) if d.frame_section else None

        def dim(v, attr):
            if v is not None:
                return v * L
            if fs is not None and getattr(fs, attr, None) is not None:
                return getattr(fs, attr) * L
            raise ModelError("FIBER_SECTION_DIM", f"Fiber section {name!r}: {attr} is not given")

        if d.GJ is not None:
            GJ = d.GJ * self.U.factor("flexural_rigidity")
        elif fs is not None:
            mat = self.model.materials[fs.material]
            GJ = mat.shear_modulus * self.U.factor("stress") * fs.J * self.U.factor("inertia") * fs.modifiers.J
        elif fe is not None:
            GJ = fe.el.G * fe.el.J
        else:
            GJ = 0.0
        shape = d.shape.lower()
        ny, nz = d.mesh
        if shape == "rcrectangle":
            sec = S.rc_rectangle(dim(d.depth, "t3"), dim(d.width, "t2"), d.cover * L, self.material(d.core_material),
                                 self.material(d.cover_material or d.core_material), self.material(d.steel_material),
                                 d.n_top, d.n_bottom, d.bar_area * A2, d.n_side,
                                 None if d.side_bar_area is None else d.side_bar_area * A2, GJ, ny, nz, name)
        elif shape == "rccircle":
            D = d.diameter * L if d.diameter is not None else dim(None, "t3")
            sec = S.rc_circle(D, d.cover * L, self.material(d.core_material),
                              self.material(d.cover_material or d.core_material), self.material(d.steel_material),
                              d.n_bars, d.bar_area * A2, GJ, max(ny * 2, 16), max(nz, 4), 2, name)
        elif shape == "steeli":
            sec = S.steel_i(dim(d.depth, "t3"), dim(d.width, "t2"), dim(d.tf, "tf"), dim(d.tw, "tw"),
                            self.material(d.steel_material), GJ, max(ny // 3, 2), ny, nz, name=name)
        elif shape == "steelbox":
            sec = S.steel_box(dim(d.depth, "t3"), dim(d.width, "t2"), dim(d.tf, "tf"), dim(d.tw, "tw"),
                              self.material(d.steel_material), GJ, name=name)
        elif shape == "steelpipe":
            D = d.diameter * L if d.diameter is not None else dim(None, "t3")
            sec = S.steel_pipe(D, dim(d.tw, "tw"), self.material(d.steel_material), GJ, name=name)
        elif shape == "custom":
            sets = {}
            for y, z, a, m in d.fibers:
                sets.setdefault(m, ([], [], []))
                sets[m][0].append(y * L)
                sets[m][1].append(z * L)
                sets[m][2].append(a * A2)
            sec = S.FiberSection3D([S.FiberSet(self.material(m), np.array(v[0]), np.array(v[1]), np.array(v[2]))
                                    for m, v in sets.items()], GJ, name)
        else:
            raise ModelError("FIBER_SECTION_SHAPE", f"Fiber section {name!r}: unknown shape {d.shape!r}")
        self._sec[name] = sec
        return sec

    # -------------------------------------------------------------- frames
    def frames(self) -> Dict[int, NLFrame]:
        out: Dict[int, NLFrame] = {}
        for name, fr in self.model.frames.items():
            eids = self.am.object_to_frames.get(name, [])
            if not eids:
                continue
            if fr.nonlinear:
                prop = self.model.frame_nonlinear.get(fr.nonlinear)
                if prop is None:
                    raise ModelError("UNKNOWN_FRAME_NONLINEAR", f"Frame {name}: nonlinear property {fr.nonlinear!r} "
                                                                "is not defined")
                for k, eid in enumerate(eids):
                    fe = self.asm.frames[eid]
                    if np.any(fe.el.releases_i[3:]) or np.any(fe.el.releases_j[3:]) or fe.el.releases_i[0] \
                            or fe.el.releases_j[0]:
                        self.warnings.append(f"Frame {name}: end releases are not supported on fiber members; "
                                             "the member stays elastic")
                        continue
                    out[eid] = self._fiber_element(fe, prop, first=(k == 0), last=(k == len(eids) - 1))
            if fr.hinges:
                placed = self._hinge_targets(name, fr, eids)
                for eid, items in placed.items():
                    if eid in out:
                        self.warnings.append(f"Frame {name}: concentrated hinges ignored on a fiber member")
                        continue
                    fe = self.asm.frames[eid]
                    pmm = [(p, e) for p, e in items if p.is_pmm]
                    lumped = [(p, e) for p, e in items if not p.is_pmm]
                    if pmm:
                        if lumped:
                            self.warnings.append(f"Frame {name}: M/P hinges ignored where a fiber P-M2-M3 hinge "
                                                 "is assigned on the same element")
                        out[eid] = self._pmm_element(fe, pmm, fr)
                    else:
                        out[eid] = ElasticHingeFrame(fe, [self.hinge(p, fe, e, fr) for p, e in lumped])
        return out

    def _pmm_element(self, fe, pmm, frame_obj) -> NLFrame:
        """Force-based member with fiber P-M2-M3 hinges at the assigned ends."""
        L = self.U.factor("length")
        el = fe.el
        Lf = fe.kin.Lf
        secE = S.ElasticSection3D(el.E * el.A, el.E * el.I33, el.E * el.I22, el.G * el.J)
        sec = {"I": secE, "J": secE}
        lp = {"I": 0.0, "J": 0.0}
        info = []
        for prop, end in pmm:
            if not prop.fiber_section:
                raise ModelError("PMM_HINGE", f"Hinge {prop.name}: a P-M2-M3 hinge needs a fiber section")
            sec[end] = self.fiber_section(prop.fiber_section, fe)
            lpe = prop.hinge_length * L
            if lpe <= 0.0:
                fs = self.model.frame_sections.get(frame_obj.section) if frame_obj is not None else None
                depth = (fs.t3 * L) if fs is not None and getattr(fs, "t3", None) else 0.1 * Lf
                lpe = 0.5 * depth
            lp[end] = lpe
            info.append({"end": end, "name": prop.name, "acceptance": prop.acceptance})
        if 4.0 * (lp["I"] + lp["J"]) >= 0.9 * Lf:
            f = 0.9 * Lf / (4.0 * (lp["I"] + lp["J"]))
            self.warnings.append(f"Frame {frame_obj.name if frame_obj else el.object}: fiber hinge lengths reduced "
                                 f"by {f:.2f} to fit the member")
            lp = {k: v * f for k, v in lp.items()}
        bi = S.hinge_radau(lp["I"], lp["J"], Lf)
        e = ForceBeamColumn(fe, [sec["I"], sec["I"], secE, secE, sec["J"], sec["J"]], bi)
        for h in info:
            h["lp"] = lp[h["end"]]
            h["max"] = 0.0
        e.fiber_hinges = info
        e.EIz, e.EIy = el.E * el.I33, el.E * el.I22
        return e

    def _place_hinges(self, name, fr, eids) -> Dict[int, List[FrameHinge]]:
        return {eid: [self.hinge(p, self.asm.frames[eid], e, fr) for p, e in items]
                for eid, items in self._hinge_targets(name, fr, eids).items()}

    def _fiber_element(self, fe, prop, first: bool, last: bool) -> NLFrame:
        U = self.U
        sec_i = self.fiber_section(prop.section, fe)
        sec_j = self.fiber_section(prop.section_j, fe) if prop.section_j else sec_i
        kind = prop.integration.lower()
        Lf = fe.kin.Lf
        if kind == "hingeradau":
            lpI = prop.hinge_length_i * U.factor("length") if first else 0.0
            lpJ = prop.hinge_length_j * U.factor("length") if last else 0.0
            if 4 * (lpI + lpJ) >= Lf:
                raise ModelError("HINGE_LENGTH", f"Element of frame {fe.el.object}: 4 (lpI + lpJ) must be below the "
                                                 "member length")
            bi = S.hinge_radau(lpI, lpJ, Lf)
            el = fe.el
            secE = S.ElasticSection3D(el.E * el.A, el.E * el.I33, el.E * el.I22, el.G * el.J)
            secs = [sec_i, sec_i, secE, secE, sec_j, sec_j]
        else:
            bi = S.make_integration(prop.integration, prop.n_points)
            secs = [sec_i] * bi.n
        if prop.formulation.lower().startswith("disp"):
            return DispBeamColumn(fe, secs, bi)
        return ForceBeamColumn(fe, secs, bi, max_iters=prop.max_iterations)

    def _hinge_targets(self, name, fr, eids):
        """Element and end of every hinge assignment: {eid: [(property, "I" | "J")]}."""
        out = {}
        els = [self.am.frames[e] for e in eids]
        for asg in fr.hinges:
            prop = self.model.hinge_properties.get(asg.property)
            if prop is None:
                raise ModelError("UNKNOWN_HINGE", f"Frame {name}: hinge property {asg.property!r} is not defined")
            rd = min(max(asg.relative_distance, 0.0), 1.0)
            target = None
            for el in els:
                if el.rel_start - 1e-9 <= rd <= el.rel_end + 1e-9:
                    end = "I" if (rd - el.rel_start) <= (el.rel_end - rd) else "J"
                    target = (el.id, end)
                    break
            if target is None:
                target = (els[-1].id, "J")
            eid, end = target
            el = self.am.frames[eid]
            pos = el.rel_start if end == "I" else el.rel_end
            if abs(pos - rd) > 0.02:
                self.warnings.append(f"Frame {name}: hinge {prop.name} at {rd:.3f} placed at the element end {pos:.3f}")
            out.setdefault(eid, []).append((prop, end))
        return out

    def hinge(self, prop, fe, end: str, frame_obj=None) -> FrameHinge:
        U = self.U
        dof = prop.dof.upper()
        el = fe.el
        Lf = fe.kin.Lf
        if dof == "M3":
            k_el = 6.0 * el.E * el.I33 / Lf
            fq, dq = U.factor("moment"), 1.0
        elif dof == "M2":
            k_el = 6.0 * el.E * el.I22 / Lf
            fq, dq = U.factor("moment"), 1.0
        elif dof == "P":
            k_el = el.E * el.A / fe.kin.L
            fq, dq = U.factor("force"), U.factor("length")
        else:
            raise ModelError("HINGE_DOF", f"Hinge {prop.name}: unsupported DOF {prop.dof!r}")
        if prop.auto:
            pts, acc = self._auto_hinge(prop, fe, frame_obj, k_el)
            pts_n, acc_n = None, None
        else:
            sf_f = prop.force_sf * fq if prop.force_sf else fq
            sf_d = prop.deformation_sf * dq if prop.deformation_sf else dq
            if not prop.points:
                raise ModelError("HINGE_POINTS", f"Hinge {prop.name}: no backbone points")
            pts = [(d * sf_d, f * sf_f) for d, f in prop.points]
            pts_n = None if prop.points_neg is None else [(abs(d) * sf_d, abs(f) * sf_f) for d, f in prop.points_neg]
            acc = None if prop.acceptance is None else tuple(a * sf_d for a in prop.acceptance)
            acc_n = None if prop.acceptance_neg is None else tuple(a * sf_d for a in prop.acceptance_neg)
        kh = prop.rigid_factor * k_el
        # member stiffness seen by the hinge (4EI/L or EA/L): a load drop steeper than half
        # of it would make the member snap back; CSI's vertical drops C-D and after E are
        # regularised to that slope (event-to-event solvers jump instead)
        k_mem = (4.0 / 6.0) * k_el if dof in ("M3", "M2") else k_el
        reg = getattr(prop, "drop_regularisation", None) or self.drop_regularisation

        def total(points, vertical=False):
            """Total-deformation backbone; ``vertical``: drops as vertical slivers,
            returns (points, [(deformation at drop, force loss, sliver width)])."""
            out, drops, last_d, last_f = [], [], 0.0, None
            for dp, f in points:
                f = abs(f)
                d = dp + f / kh
                if last_f is not None and f < last_f:
                    need = reg * (last_f - f) / k_mem
                    if d - last_d < need:
                        if vertical:
                            eps = 1e-9 * max(last_d, 1e-12)
                            drops.append((last_d, last_f - f, eps))
                            out.append((last_d + eps, f))
                            last_d, last_f = last_d + eps, f
                            if d <= last_d * (1 + 1e-6):
                                continue
                        else:
                            d = last_d + need
                if d <= last_d:
                    d = last_d * (1 + 1e-9) + 1e-12
                out.append((d, f))
                last_d, last_f = d, f
            if prop.drop_after_e and out and out[-1][1] > 0:
                fres = prop.residual * out[-1][1]
                if vertical:
                    eps = 1e-9 * max(out[-1][0], 1e-12)
                    drops.append((out[-1][0], out[-1][1] - fres, eps))
                    out.append((out[-1][0] + eps, fres))
                else:
                    out.append((out[-1][0] + reg * (out[-1][1] - fres) / k_mem, fres))
            if out:  # flat beyond the last point
                out.append((out[-1][0] * 10.0 + 1.0, out[-1][1]))
            return (out, drops) if vertical else out

        pos = total(pts)
        neg = total(pts_n) if pts_n else None
        vpos, dpos = total(pts, True)
        vneg, dneg = total(pts_n, True) if pts_n else (None, None)
        if dneg is None:
            dneg = list(dpos)
        vertical = None
        if dpos or dneg:
            vertical = {"pos": vpos, "neg": vneg, "drops_pos": dpos, "drops_neg": dneg}
        mat = HingeMaterial(pos, neg, prop.hysteresis, drop=False, vertical=vertical)
        return FrameHinge(dof, end, mat, name=prop.name, acceptance=acc, yield_deformation=pos[0][0],
                          backbone_points=pos)

    def _auto_hinge(self, prop, fe, frame_obj, k_el):
        """Backbone (plastic deformation, force) in SI and acceptance for automatic hinges."""
        U = self.U
        p = prop.auto_params
        kind = prop.auto.lower()
        sec = self.model.frame_sections.get(frame_obj.section) if frame_obj is not None else None
        if kind in ("asce41-steelbeam", "asce41-steel-beam"):
            if sec is None or sec.shape != "I":
                raise ModelError("AUTO_HINGE", f"Hinge {prop.name}: ASCE 41 steel beam hinges need an I section")
            L = U.factor("length")
            fy = p.get("fy", None)
            if fy is None:
                mat = self.model.materials[sec.material]
                fy = mat.fy
            if fy is None:
                raise ModelError("AUTO_HINGE", f"Hinge {prop.name}: steel yield stress fy is not given")
            fy = fy * U.factor("stress") * p.get("expected_factor", 1.1)
            Z = H.i_plastic_modulus(sec.t3 * L, sec.t2 * L, sec.tf * L, sec.tw * L)
            My = Z * fy
            theta_y = My / k_el  # ASCE 41 Eq. 9-1: Z Fye lb / (6 E I)
            a, b, c, io, ls, cp = H.asce41_steel_beam(theta_y, p.get("compactness", 0.0))
            hard = 0.03 * k_el * a / My  # 3 % strain hardening between B and C
            return H.csi_points_from_abc(My, a, b, c, hard), (io, ls, cp)
        if kind in ("asce41-rcbeam", "asce41-rc-beam"):
            My = p.get("My")
            if My is None:
                fsn = p.get("fiber_section")
                if not fsn:
                    raise ModelError("AUTO_HINGE", f"Hinge {prop.name}: give My or a fiber section for RC beam hinges")
                mc = H.moment_curvature(self.fiber_section(fsn, fe), 0.0, "z", kappa_max=0.08, n=160, eps_cu=0.004)
                My = H.yield_and_ultimate(mc)["My"]
            else:
                My = My * U.factor("moment")
            a, b, c, io, ls, cp = H.asce41_rc_beam(p.get("rho_ratio", 0.0), bool(p.get("conforming", 1)),
                                                   p.get("shear_index", 3.0))
            return H.csi_points_from_abc(My, a, b, c, p.get("hardening", 0.1)), (io, ls, cp)
        raise ModelError("AUTO_HINGE", f"Hinge {prop.name}: unknown automatic hinge {prop.auto!r}")

    # ---------------------------------------------------------------- links
    def links(self) -> List[Link]:
        out = []
        U = self.U
        for k, (name, lk) in enumerate(self.model.links.items()):
            prop = self.model.link_properties.get(lk.property)
            if prop is None:
                raise ModelError("UNKNOWN_LINK_PROPERTY", f"Link {name}: property {lk.property!r} is not defined")
            ni = self.am.point_to_node.get(lk.point_i)
            nj = self.am.point_to_node.get(lk.point_j) if lk.point_j else None
            if ni is None or (lk.point_j and nj is None):
                raise ModelError("LINK_POINT", f"Link {name}: point not found in the analysis model")
            xi = self.am.nodes[ni].xyz
            if nj is not None and np.linalg.norm(self.am.nodes[nj].xyz - xi) > 1e-9:
                axes = frame_axes(xi, self.am.nodes[nj].xyz, lk.angle)
            else:  # zero length / grounded: axis 1 = +Z (CSI)
                axes = frame_axes(np.zeros(3), np.array([0.0, 0.0, 1.0]), lk.angle)
            mats = {}
            for d in prop.dofs:
                mats[_DOF_INDEX[d.dof.upper()]] = self._link_material(d)
            out.append(Link(k, ni, nj, axes, mats, name))
        return out

    def _link_material(self, d) -> UniaxialMaterial:
        U = self.U
        rot = d.dof.upper().startswith("R")
        kq = U.factor("rotational_stiffness" if rot else "translational_stiffness")
        fq = U.factor("moment" if rot else "force")
        dq = 1.0 if rot else U.factor("length")
        t = d.type.lower()
        k = d.k * kq
        if t == "linear":
            return Elastic(k, eta=d.c * kq)
        if t == "wen":
            m = BoucWen.csi_wen(k, d.yield_force * fq, d.post_yield_ratio, d.exponent if d.exponent != 1.0 else 2.0)
            return Parallel([m, Viscous(d.c * kq, 1.0)]) if d.c else m
        if t == "elasticpp":
            return ElasticPP(k, d.yield_force * fq / k)
        if t == "damper":
            c = d.c * fq / dq ** d.exponent
            v = Viscous(c, d.exponent)
            return Parallel([v, Elastic(k)]) if k else v
        if t == "multilinear":
            pts = [(abs(x) * dq, abs(f) * fq) for x, f in d.points if x != 0]
            return HingeMaterial(pts, None, d.hysteresis, drop=False)
        raise ModelError("LINK_TYPE", f"Unknown link DOF type {d.type!r}")
