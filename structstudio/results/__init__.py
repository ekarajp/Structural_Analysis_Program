"""Result queries in the model's unit system.

:class:`Results` wraps the raw SI :class:`~structstudio.engine.results.CaseResult`
objects and the analysis model, and answers the questions an engineer (or
an agent) asks: joint displacements/reactions, frame forces at stations,
modal tables, story drifts, diaphragm displacements and load combinations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..engine.results import CaseResult, ModalData
from ..mesh.analysis_model import AnalysisModel
from ..model import Model, ModelError
from ..model.loads import LoadCombination

DOF_LABELS = ("Ux", "Uy", "Uz", "Rx", "Ry", "Rz")
FORCE_LABELS = ("P", "V2", "V3", "T", "M2", "M3")


def _exact_end(r: CaseResult, eid: int):
    """Stored end forces of nonlinear cases (station forces then follow from equilibrium)."""
    if r.diagnostics.get("end_forces_exact"):
        return r.frame_end_forces[eid]
    return None


_DOFS = ("UX", "UY", "UZ", "RX", "RY", "RZ")


@dataclass
class Envelope:
    """Max/min pair of arrays (CSI combo semantics)."""

    max: np.ndarray
    min: np.ndarray

    @classmethod
    def single(cls, value: np.ndarray) -> "Envelope":
        return cls(np.array(value, float), np.array(value, float))

    @classmethod
    def symmetric(cls, value: np.ndarray) -> "Envelope":
        v = np.abs(np.array(value, float))
        return cls(v, -v)

    def scaled(self, sf: float) -> "Envelope":
        if sf >= 0:
            return Envelope(self.max * sf, self.min * sf)
        return Envelope(self.min * sf, self.max * sf)


@dataclass
class FrameStationForces:
    frame: str
    case: str
    station: np.ndarray  # distance from end I of the object (model length units)
    forces: np.ndarray  # (n, 6) P V2 V3 T M2 M3 (model units)
    forces_min: Optional[np.ndarray] = None  # for envelopes/combos

    def table(self) -> List[dict]:
        rows = []
        for k in range(len(self.station)):
            row = {"Frame": self.frame, "OutputCase": self.case, "Station": float(self.station[k])}
            for j, lab in enumerate(FORCE_LABELS):
                row[lab] = float(self.forces[k, j])
                if self.forces_min is not None:
                    row[lab + "Min"] = float(self.forces_min[k, j])
            rows.append(row)
        return rows

    def max_abs(self, component: str) -> float:
        j = FORCE_LABELS.index(component)
        vals = self.forces[:, j] if self.forces_min is None else np.concatenate([self.forces[:, j], self.forces_min[:, j]])
        return float(np.max(np.abs(vals)))


class ShellForcesTable:
    """Shell forces rows held as arrays (read-only): row r is node ``kk[r]`` of
    element ``el[r]`` of the stacked envelope.  ``row(r)`` gives the same dict
    as the list form; ``value(r, column)`` one cell ("" where a row has no such
    column: an element without a distinct minimum has no ...Min columns)."""

    def __init__(self, case, components, env, two, el, kk, node, eid, obj):
        self.case, self.components = case, list(components)
        self.env, self.two = env, two
        self.el, self.kk, self.node, self.eid, self.obj = el, kk, node, eid, obj
        self._col = {lab: j for j, lab in enumerate(self.components)}

    def __len__(self) -> int:
        return len(self.el)

    def columns(self, r: int = 0) -> List[str]:
        """Column names of row ``r`` (the table shows those of its first row)."""
        cols = ["Area", "Element", "Node", "Case"]
        two = len(self.el) and bool(self.two[self.el[r]])
        for lab in self.components:
            cols.append(lab)
            if two:
                cols.append(lab + "Min")
        return cols

    def value(self, r: int, col: str):
        if col == "Area":
            return self.obj[r]
        if col == "Element":
            return int(self.eid[r])
        if col == "Node":
            return int(self.node[r])
        if col == "Case":
            return self.case
        k = int(self.el[r])
        if col.endswith("Min") and col[:-3] in self._col:
            if not self.two[k]:
                return ""
            return float(self.env.min[k, self.kk[r], self._col[col[:-3]]])
        j = self._col.get(col)
        return "" if j is None else float(self.env.max[k, self.kk[r], j])

    def row(self, r: int) -> dict:
        return {c: self.value(r, c) for c in self.columns(r)}


class Results:
    def __init__(self, model: Model, am: AnalysisModel, cases: Dict[str, CaseResult], engine=None):
        self.model = model
        self.am = am
        self.cases = cases
        self.engine = engine
        self.U = model.units
        self._station_cache: Dict[Tuple[str, str], FrameStationForces] = {}

    def in_units(self, model: Model) -> "Results":
        """The same results reported in the units of ``model`` (the analysed model
        converted to other units): the solution is kept in SI, only the output
        changes - as ETABS / SAP2000 keep their results when the units change."""
        out = Results(model, self.am, self.cases, self.engine)
        out.issues = list(getattr(self, "issues", []))
        return out

    # ----------------------------------------------------------------- utils
    @property
    def case_names(self) -> List[str]:
        return list(self.cases)

    def _case(self, name: str) -> CaseResult:
        try:
            return self.cases[name]
        except KeyError:
            raise ModelError("NO_RESULTS", f"No results for case {name!r}; available: {list(self.cases)}")

    def _disp_factor(self) -> np.ndarray:
        L = self.U.factor("length")
        return np.array([1 / L, 1 / L, 1 / L, 1.0, 1.0, 1.0])

    def _force_factor(self) -> np.ndarray:
        F = self.U.factor("force")
        M = self.U.factor("moment")
        return np.array([1 / F, 1 / F, 1 / F, 1 / M, 1 / M, 1 / M])

    def _node_of_point(self, point: str) -> int:
        try:
            return self.am.point_to_node[point]
        except KeyError:
            raise ModelError("UNKNOWN_POINT", f"Point {point!r} is not in the analysis model")

    # ----------------------------------------------------- joint results
    def joint_displacement(self, point: str, case: str) -> np.ndarray:
        """(Ux, Uy, Uz, Rx, Ry, Rz) in model units (max values for envelopes)."""
        env = self.displacement_envelope(case)
        return env.max[self._node_of_point(point)]

    def displacement_envelope(self, case: str) -> Envelope:
        if case in self.model.load_combinations:
            return self._combo_displacements(case)
        r = self._case(case)
        d = r.displacements * self._disp_factor()
        return Envelope.symmetric(d) if r.is_envelope else Envelope.single(d)

    def joint_displacements(self, case: str, points: Optional[Iterable[str]] = None) -> List[dict]:
        env = self.displacement_envelope(case)
        rows = []
        names = points if points is not None else list(self.model.points)
        # Min columns when the envelope has distinct max / min anywhere in the model
        # (decided once per query, not per point and DOF)
        with_min = not np.array_equal(env.max, env.min)
        for p in names:
            nid = self._node_of_point(p)
            row = {"Point": p, "Story": self.model.points[p].story, "OutputCase": case}
            for j, lab in enumerate(DOF_LABELS):
                row[lab] = float(env.max[nid, j])
                if with_min:
                    row[lab + "Min"] = float(env.min[nid, j])
            rows.append(row)
        return rows

    def reaction_envelope(self, case: str) -> Dict[int, Envelope]:
        if case in self.model.load_combinations:
            return self._combo_reactions(case)
        r = self._case(case)
        ff = self._force_factor()
        out = {}
        for nid, vec in r.reactions.items():
            v = vec * ff
            out[nid] = Envelope.symmetric(v) if r.is_envelope else Envelope.single(v)
        return out

    def joint_reaction(self, point: str, case: str) -> np.ndarray:
        env = self.reaction_envelope(case)
        nid = self._node_of_point(point)
        return env[nid].max if nid in env else np.zeros(6)

    def joint_reactions(self, case: str) -> List[dict]:
        env = self.reaction_envelope(case)
        rows = []
        for nid, e in sorted(env.items()):
            node = self.am.nodes[nid]
            row = {"Point": node.point, "Story": node.story, "OutputCase": case}
            for j, lab in enumerate(("FX", "FY", "FZ", "MX", "MY", "MZ")):
                row[lab] = float(e.max[j])
                if not np.array_equal(e.max, e.min):
                    row[lab + "Min"] = float(e.min[j])
            rows.append(row)
        return rows

    def base_reactions(self, case: str) -> np.ndarray:
        """Sum of support reactions (FX, FY, FZ) in model units.  For spectrum
        cases the modal base totals are combined (CQC/SRSS...), as in ETABS."""
        r = self.cases.get(case)
        if r is not None and getattr(r, "spectrum_base", None) is not None:
            return r.spectrum_base[:3] / self.U.factor("force")
        env = self.reaction_envelope(case)
        tot = np.zeros(3)
        for e in env.values():
            tot += e.max[:3]
        dist = self._distributed_reaction(case)
        if dist is not None:
            tot += dist.max[:3]
        return tot

    def _distributed_reaction(self, case: str) -> Optional[Envelope]:
        """Reaction of distributed supports (frame foundations), model units."""
        if case in self.model.load_combinations:
            combo = self._combo(case)
            parts = {it.name: self._distributed_reaction(it.name) for it in combo.items}
            if all(v is None for v in parts.values()):
                return None
            return self._combine(combo, lambda n: parts[n] or Envelope.single(np.zeros(6)))
        r = self.cases.get(case)
        if r is None or getattr(r, "distributed_reaction", None) is None:
            return None
        v = r.distributed_reaction * self._force_factor()
        return Envelope.symmetric(v) if r.is_envelope else Envelope.single(v)

    # ------------------------------------------------------ frame results
    def _object_stations(self, frame: str) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """For each element of the object: (element id, x along element (SI),
        x along object (SI)) at the output stations.  Stations exclude the
        end offsets (CSI convention)."""
        obj = self.model.frames[frame]
        elems = self.am.object_to_frames.get(frame, [])
        # a beam piece inside a column's cross-section, where the slab is tied to
        # the column as a rigid body, does not deform: its forces would read zero.
        # Like an end offset, it gets no output stations (the results start at the face)
        body_of = getattr(self, "_body_of", None)
        if body_of is None:
            body_of = self._body_of = {}
            for k, b in enumerate(getattr(self.am, "bodies", []) or []):
                body_of[b.master] = k
                for n in b.nodes:
                    body_of[n] = k
        if body_of:
            inside = [eid for eid in elems
                      if body_of.get(self.am.frames[eid].node_i, -1) == body_of.get(self.am.frames[eid].node_j, -2)]
            if inside and len(inside) < len(elems):
                elems = [e for e in elems if e not in inside]
        L_obj = None
        out = []
        for eid in elems:
            el = self.am.frames[eid]
            fe = self.engine.asm.frames[eid]
            L = fe.kin.L
            if L_obj is None:
                L_obj = L / max(el.rel_end - el.rel_start, 1e-12)
            x0 = el.end_offset_i
            x1 = L - el.end_offset_j
            n = max(2, el.stations)
            xs = np.linspace(x0, x1, n)
            x_obj = el.rel_start * L_obj + xs
            out.append((eid, xs, x_obj))
        return out

    def frame_forces(self, frame: str, case: str) -> FrameStationForces:
        """Internal forces P V2 V3 T M2 M3 at the output stations of a frame
        object (model units, CSI sign convention)."""
        key = (frame, case)
        if key in self._station_cache:
            return self._station_cache[key]
        if frame not in self.model.frames:
            raise ModelError("UNKNOWN_FRAME", f"Frame {frame!r} is not defined")
        if case in self.model.load_combinations:
            res = self._combo_frame_forces(frame, case)
        else:
            r = self._case(case)
            stations, forces, forces_min = [], [], []
            for eid, xs, x_obj in self._object_stations(frame):
                fe = self.engine.asm.frames[eid]
                if r.is_envelope and r.modal_displacements is not None:
                    # combine station forces mode by mode and direction by direction
                    f = self._rs_station_forces(r, fe, xs)
                    forces.append(f)
                    forces_min.append(-f)
                else:
                    d = self.engine.asm._element_dofs(fe.el.nodes)
                    u = r.displacements.reshape(-1)[d]
                    f = fe.internal_forces(u, r.frame_loads.get(eid, []), xs, _exact_end(r, eid))
                    forces.append(f)
                stations.append(x_obj)
            st = np.concatenate(stations) / self.U.factor("length")
            ff = self._force_factor()
            F = np.concatenate(forces) * ff
            Fmin = np.concatenate(forces_min) * ff if forces_min else None
            res = FrameStationForces(frame, case, st, F, Fmin)
        self._station_cache[key] = res
        return res

    def frame_forces_at(self, frame: str, case: str, stations: Sequence[float]) -> FrameStationForces:
        """Internal forces at arbitrary distances from end I of the object
        (model length units).  Envelope cases are evaluated mode by mode at
        each station (as ETABS does), so no interpolation of envelopes occurs.
        Points inside an end offset are evaluated at the face of the offset."""
        if frame not in self.model.frames:
            raise ModelError("UNKNOWN_FRAME", f"Frame {frame!r} is not defined")
        if case in self.model.load_combinations:
            combo = self._combo(case)

            def getter(n):
                st = self.frame_forces_at(frame, n, stations)
                return Envelope.single(st.forces) if st.forces_min is None else Envelope(st.forces, st.forces_min)

            env = self._combine(combo, getter)
            return FrameStationForces(frame, case, np.asarray(stations, float), env.max, env.min)
        r = self._case(case)
        Lf = self.U.factor("length")
        elems = self.am.object_to_frames.get(frame, [])
        spans = []
        L_obj = None
        for eid in elems:
            el = self.am.frames[eid]
            fe = self.engine.asm.frames[eid]
            L = fe.kin.L
            if L_obj is None:
                L_obj = L / max(el.rel_end - el.rel_start, 1e-12)
            spans.append((el.rel_start * L_obj, el.rel_end * L_obj, eid, fe, el))
        ff = self._force_factor()
        out = np.zeros((len(stations), 6))
        out_min = np.zeros((len(stations), 6)) if r.is_envelope else None
        for k, x in enumerate(stations):
            xs = float(x) * Lf
            for a, b, eid, fe, el in spans:
                if xs <= b + 1e-9 or (a, b) == spans[-1][:2]:
                    xl = min(max(xs - a, el.end_offset_i), fe.kin.L - el.end_offset_j)
                    if r.is_envelope and r.modal_displacements is not None:
                        f = self._rs_station_forces(r, fe, np.array([xl]))[0]
                        out[k] = f * ff
                        out_min[k] = -f * ff
                    else:
                        d = self.engine.asm._element_dofs(fe.el.nodes)
                        u = r.displacements.reshape(-1)[d]
                        out[k] = fe.internal_forces(u, r.frame_loads.get(eid, []), np.array([xl]),
                                                    _exact_end(r, eid))[0] * ff
                    break
        return FrameStationForces(frame, case, np.asarray(stations, float), out, out_min)

    def _rs_station_forces(self, r: CaseResult, fe, xs: np.ndarray) -> np.ndarray:
        from ..engine.solvers import _combine_directions, _correlation, _combine_modes
        case = self.model.load_cases[r.case]
        rho = _correlation(r.modal.omega, case.damping, case.modal_combination)
        d = self.engine.asm._element_dofs(fe.el.nodes)
        per_dir = []
        for u_modes in r.modal_displacements:  # (n_modes, n_nodes, 6)
            vals = np.array([fe.internal_forces(u_modes[n].reshape(-1)[d], [], xs) for n in range(u_modes.shape[0])])
            per_dir.append(_combine_modes(vals, rho))
        return _combine_directions(per_dir, case)

    def frame_end_forces(self, frame: str, case: str) -> np.ndarray:
        """Forces at the two ends of the object (first element I, last element J)
        in local axes [P V2 V3 T M2 M3] x 2, model units, CSI sign convention."""
        st = self.frame_forces(frame, case)
        return np.vstack([st.forces[0], st.forces[-1]])

    # --------------------------------------------------------- nonlinear
    def _hist(self, case: str) -> dict:
        r = self._case(case)
        if not r.history:
            raise ModelError("NOT_NONLINEAR", f"Case {case!r} has no step / time history")
        return r.history

    def pushover_curve(self, case: str, direction: str = "auto") -> List[dict]:
        """Capacity curve of a nonlinear static case: one row per step with the
        monitored displacement, the base reaction (model units; base shear is the
        negative of the reaction) and the number of hinges in each state /
        performance range (CSI "Pushover Curve" table)."""
        h = self._hist(case)
        fF = self.U.factor("force")
        fL = self.U.factor("length")
        fM = self.U.factor("moment")
        dof = (h.get("control_dof") or "UX").upper()
        rows = [{"Step": 0, "Displacement": 0.0, "BaseForce": 0.0}]
        if direction == "auto":
            direction = dof if dof in ("UX", "UY", "UZ") else "UX"
        k = ("UX", "UY", "UZ").index(direction)
        for st in h["steps"]:
            br = st["base_reaction"]
            row = {"Step": st["step"], "LoadFactor": st["lambda"],
                   "Displacement": (st["control"] / (fL if dof.startswith("U") else 1.0)) if st["control"] is not None else None,
                   "BaseForce": -br[k] / fF, "FX": br[0] / fF, "FY": br[1] / fF, "FZ": br[2] / fF,
                   "MX": br[3] / fM, "MY": br[4] / fM, "MZ": br[5] / fM}
            counts = {}
            for hg in st["hinges"]:
                counts[hg["state"] or "n/a"] = counts.get(hg["state"] or "n/a", 0) + 1
                if hg["performance"]:
                    counts[hg["performance"]] = counts.get(hg["performance"], 0) + 1
            row.update(counts)
            rows.append(row)
        return rows

    def hinge_results(self, case: str, step: int = -1) -> List[dict]:
        """Hinge deformations / forces / states of a nonlinear case (model units).
        ``step``: pushover step (1-based, -1 = last); time histories report the end
        state with the maximum and minimum deformations over the record."""
        h = self._hist(case)
        hs = h["steps"][step]["hinges"] if h.get("kind") == "NonlinearStatic" else h.get("hinges", [])
        out = []
        for x in hs:
            el = self.am.frames[x["element"]]
            fq = "force" if x["dof"] == "P" else "moment"
            dq = self.U.factor("length") if x["dof"] == "P" else 1.0
            extra = {}
            if x["dof"] == "PMM":
                extra = {"P": x["P"] / self.U.factor("force"), "M2": x["M2"] / self.U.factor("moment"),
                         "M3": x["M3"] / self.U.factor("moment"), "R2": x["R2"], "R3": x["R3"]}
            out.append({"Frame": el.object, "Element": x["element"], "Hinge": x["name"], "DOF": x["dof"],
                        "End": x["end"], "Deformation": x["deformation"] / dq,
                        "PlasticDeformation": x["plastic_deformation"] / dq,
                        "Force": x["force"] / self.U.factor(fq), "State": x["state"], "Performance": x["performance"],
                        "MaxDeformation": x["max_deformation"] / dq, "MinDeformation": x["min_deformation"] / dq,
                        **extra})
        return out

    def time_history(self, case: str, point: str, component: str = "UX", quantity: str = "displacement"):
        """(time, values) of a monitored point: ``quantity`` displacement | velocity |
        acceleration (relative) | absolute_acceleration; model units."""
        h = self._hist(case)
        nid = self._node_of_point(point)
        mon = h["monitor_nodes"]
        if nid not in mon:
            raise ModelError("NOT_MONITORED", f"Point {point!r} was not monitored in case {case!r}; add it to "
                                              "monitor_points")
        i = mon.index(nid)
        c = _DOFS.index(component.upper())
        key = {"displacement": "displacements", "velocity": "velocities", "acceleration": "accelerations",
               "absolute_acceleration": "accelerations"}[quantity]
        if key not in h:
            raise ModelError("NOT_RECORDED", f"{quantity} is not recorded for case {case!r}")
        v = h[key][:, i, c].copy()
        if quantity == "absolute_acceleration" and c < 3:
            v = v + h["ground_acceleration"][:, c]
        f = (self.U.factor("length") if c < 3 else 1.0)
        if quantity != "displacement":
            f = self.U.factor("length") if c < 3 else 1.0
        return h["time"].copy(), v / f

    def base_reaction_history(self, case: str) -> Tuple[np.ndarray, np.ndarray]:
        h = self._hist(case)
        br = h["base_reaction"].copy()
        br[:, :3] /= self.U.factor("force")
        br[:, 3:] /= self.U.factor("moment")
        t = h["time"] if "time" in h else np.arange(1, len(br) + 1)
        return t, br

    def time_history_envelope(self, case: str) -> Dict[str, Envelope]:
        """Max / min joint displacements over a time history (model units)."""
        h = self._hist(case)
        env = h["envelope"]
        f = self._disp_factor()
        return {"displacements": Envelope(env["disp_max"] / f, env["disp_min"] / f)}

    def time_history_frame_envelope(self, frame: str, case: str) -> np.ndarray:
        """Max / min local end forces of the elements of a frame object over a time
        history: array (n_elements, 2, 12) [max, min] in model units."""
        h = self._hist(case)
        env = h["envelope"]
        ff = np.concatenate([self._force_factor()] * 2)
        out = []
        for eid in self.am.object_to_frames.get(frame, []):
            out.append(np.vstack([env["frame_max"][eid] / ff, env["frame_min"][eid] / ff]))
        return np.array(out)

    def story_drift_history(self, case: str) -> List[dict]:
        """Peak story drift ratios over a time history from the diaphragm (or monitored) joints."""
        h = self._hist(case)
        out = []
        disp = h["displacements"]
        mon = h["monitor_nodes"]
        stories = []
        for c in self.am.constraints:
            if c.master in mon and c.story in self.am.story_elevations:
                stories.append((self.am.story_elevations[c.story], c.story, mon.index(c.master)))
        stories.sort()
        prev_z, prev_i = None, None
        base = min(self.am.story_elevations.values()) if self.am.story_elevations else 0.0
        for z, name, i in stories:
            dz = z - (prev_z if prev_z is not None else base)
            for comp, k in (("X", 0), ("Y", 1)):
                d = disp[:, i, k] - (disp[:, prev_i, k] if prev_i is not None else 0.0)
                out.append({"Story": name, "Direction": comp, "MaxDrift": float(np.max(np.abs(d)) / dz) if dz else 0.0})
            prev_z, prev_i = z, i
        return out

    # ------------------------------------------------------------- modal
    def modal(self, case: str) -> ModalData:
        r = self._case(case)
        if r.modal is None:
            raise ModelError("NOT_MODAL", f"Case {case!r} has no modal data")
        return r.modal

    def modal_periods(self, case: str) -> List[dict]:
        md = self.modal(case)
        return [{"Case": case, "Mode": k + 1, "Period": float(md.periods[k]), "Frequency": float(1 / md.periods[k]),
                 "CircFreq": float(md.omega[k]), "Eigenvalue": float(md.eigenvalues[k])} for k in range(md.n_modes)]

    def modal_participating_mass_ratios(self, case: str) -> List[dict]:
        md = self.modal(case)
        rows = []
        cum = np.zeros(6)
        for k in range(md.n_modes):
            cum += md.mass_ratios[k]
            row = {"Case": case, "Mode": k + 1, "Period": float(md.periods[k])}
            for j, lab in enumerate(("UX", "UY", "UZ", "RX", "RY", "RZ")):
                row[lab] = float(md.mass_ratios[k, j])
            for j, lab in enumerate(("SumUX", "SumUY", "SumUZ", "SumRX", "SumRY", "SumRZ")):
                row[lab] = float(cum[j])
            rows.append(row)
        return rows

    def mode_shape(self, case: str, mode: int, point: Optional[str] = None, normalize: str = "mass") -> np.ndarray:
        """Mode shape (n_nodes, 6) or for a point (6,).  ``normalize``:
        "mass" (phi^T M phi = 1, CSI default), "max" (max translation = 1)."""
        md = self.modal(case)
        phi = md.shapes[:, mode - 1].reshape(-1, 6)
        if normalize == "max":
            mx = np.max(np.abs(phi[:, :3]))
            phi = phi / mx if mx > 0 else phi
        else:
            # mass normalised shapes have units 1/sqrt(mass) (translations) and
            # 1/sqrt(rotational mass) (rotations): express in model units
            fm = np.sqrt(self.U.factor("mass"))
            fr = np.sqrt(self.U.factor("rotational_mass"))
            phi = phi * np.array([fm, fm, fm, fr, fr, fr])
        if point is not None:
            return phi[self._node_of_point(point)]
        return phi

    def participation_factors(self, case: str) -> np.ndarray:
        """(n_modes, 6) in model units (mass-normalised modes)."""
        md = self.modal(case)
        fm = self.U.factor("mass") ** 0.5
        L = self.U.factor("length")
        return md.participation / np.array([fm, fm, fm, fm * L, fm * L, fm * L])

    # ------------------------------------------------------- diaphragms
    def diaphragm_displacement(self, diaphragm: str, story: str, case: str) -> np.ndarray:
        """Displacement (Ux, Uy, Rz) of the diaphragm master (geometric centre) in model units."""
        for c in self.am.constraints:
            if c.name == diaphragm and c.story == story:
                env = self.displacement_envelope(case)
                d = env.max[c.master]
                return np.array([d[0], d[1], d[5]])
        raise ModelError("UNKNOWN_DIAPHRAGM", f"No rigid diaphragm {diaphragm!r} at story {story!r}")

    def diaphragm_center(self, diaphragm: str, story: str) -> np.ndarray:
        for c in self.am.constraints:
            if c.name == diaphragm and c.story == story:
                return self.am.nodes[c.master].xyz / self.U.factor("length")
        raise ModelError("UNKNOWN_DIAPHRAGM", f"No rigid diaphragm {diaphragm!r} at story {story!r}")

    def point_displacement_at(self, xyz: Sequence[float], diaphragm: str, story: str, case: str) -> np.ndarray:
        """Rigid-body in-plane displacement (Ux, Uy) of any point of a rigid
        diaphragm, e.g. its centre of mass, in model units."""
        d = self.diaphragm_displacement(diaphragm, story, case)
        c = self.diaphragm_center(diaphragm, story)
        dx, dy = xyz[0] - c[0], xyz[1] - c[1]
        return np.array([d[0] - d[2] * dy, d[1] + d[2] * dx])

    # ---------------------------------------------------------- stories
    def story_drifts(self, case: str) -> List[dict]:
        """Maximum drift ratio per story in X and Y (point-based, like CSI)."""
        env = self.displacement_envelope(case)
        stories = self.model.stories_bottom_up()
        rows = []
        for k in range(1, len(stories)):
            s, below = stories[k], stories[k - 1]
            h = s.elevation - below.elevation
            if h <= 0:
                continue
            best = {"X": (0.0, None), "Y": (0.0, None)}
            for p in self.model.points.values():
                if p.story != s.name:
                    continue
                q = self.model.find_point(p.x, p.y, below.elevation)
                if q is None:
                    continue
                dp = env.max[self._node_of_point(p.name)]
                dq = env.max[self._node_of_point(q.name)]
                for j, lab in ((0, "X"), (1, "Y")):
                    drift = abs(dp[j] - dq[j]) / h
                    if drift > best[lab][0]:
                        best[lab] = (drift, p.name)
            for lab in ("X", "Y"):
                rows.append({"Story": s.name, "OutputCase": case, "Direction": lab, "Drift": float(best[lab][0]),
                             "Label": best[lab][1]})
        return rows

    def story_shears(self, case: str) -> List[dict]:
        """Story shear (VX, VY) and axial sum (P) just below each story level,
        from the internal forces of all frame elements (columns, braces,
        inclined members) crossing a horizontal plane slightly below the
        level.  Wall (shell) contributions are not included yet."""
        stories = self.model.stories_bottom_up()
        L = self.U.factor("length")
        F = self.U.factor("force")
        rows = []
        r = self._case(case) if case not in self.model.load_combinations else None
        frames = self.engine.asm.frames
        ZA, ZB = self._frame_end_z()
        ZLO, ZHI = np.minimum(ZA, ZB), np.maximum(ZA, ZB)
        for k in range(1, len(stories)):
            s, below = stories[k], stories[k - 1]
            z_cut = (s.elevation - 0.02 * (s.elevation - below.elevation)) * L
            total = np.zeros(3)
            total_min = np.zeros(3)
            # the elements crossing the cut (in element order, as the sums were made)
            for i in np.flatnonzero((ZLO < z_cut) & (z_cut < ZHI)):
                fe = frames[int(i)]
                za, zb = float(ZA[i]), float(ZB[i])
                t = (z_cut - za) / (zb - za)
                x = np.array([t * fe.kin.L])
                env = self._element_force_envelope(case, fe, x)
                for arr, acc in ((env.max, total), (env.min, total_min)):
                    f_local = arr[0, :3]
                    f_global = fe.kin.axes.T @ f_local
                    # section force on the +1 face = force of the J-side on the I-side.
                    # we want the force transferred from below to above: flip when I is above
                    sign = 1.0 if za < zb else -1.0
                    acc += sign * f_global
            row = {"Story": s.name, "OutputCase": case, "VX": float(total[0] / F), "VY": float(total[1] / F),
                   "P": float(total[2] / F)}
            if not np.allclose(total, total_min):
                row.update({"VXMin": float(total_min[0] / F), "VYMin": float(total_min[1] / F), "PMin": float(total_min[2] / F)})
            rows.append(row)
        return rows

    def _frame_end_z(self):
        """Z of the two ends (with offsets) of every analysis frame element (SI)."""
        z = getattr(self, "_frame_z_cache", None)
        if z is None:
            frames = self.engine.asm.frames
            nz = np.array([n.xyz[2] for n in self.am.nodes], float)
            za = np.array([nz[fe.el.node_i] + fe.el.offset_i[2] for fe in frames], float)
            zb = np.array([nz[fe.el.node_j] + fe.el.offset_j[2] for fe in frames], float)
            za.setflags(write=False)
            zb.setflags(write=False)
            z = self._frame_z_cache = (za, zb)
        return z

    def _element_force_envelope(self, case: str, fe, x: np.ndarray) -> Envelope:
        """Internal forces (SI) of one element at stations x for a case or combo."""
        if case in self.model.load_combinations:
            combo = self._combo(case)
            return self._combine(combo, lambda n: self._element_force_envelope(n, fe, x))
        r = self._case(case)
        eid = fe.el.id
        if r.is_envelope and r.modal_displacements is not None:
            f = self._rs_station_forces(r, fe, x)
            return Envelope(f, -f)
        d = self.engine.asm._element_dofs(fe.el.nodes)
        u = r.displacements.reshape(-1)[d]
        return Envelope.single(fe.internal_forces(u, r.frame_loads.get(eid, []), x, _exact_end(r, eid)))

    # ------------------------------------------------------ centre of mass
    def diaphragm_center_of_mass(self, diaphragm: str, story: str) -> np.ndarray:
        """Centre of lateral mass of a rigid diaphragm (model units)."""
        M = self.engine.asm.mass_full()
        for c in self.am.constraints:
            if c.name == diaphragm and c.story == story:
                ids = list(c.nodes) + [c.master]
                m = np.array([M[6 * n] for n in ids])
                xy = np.array([self.am.nodes[n].xyz for n in ids])
                if m.sum() <= 0:
                    return self.am.nodes[c.master].xyz / self.U.factor("length")
                return (m[:, None] * xy).sum(axis=0) / m.sum() / self.U.factor("length")
        raise ModelError("UNKNOWN_DIAPHRAGM", f"No rigid diaphragm {diaphragm!r} at story {story!r}")

    def diaphragm_cm_displacement(self, diaphragm: str, story: str, case: str) -> np.ndarray:
        """(Ux, Uy, Rz) at the diaphragm centre of mass, like the ETABS
        "Diaphragm Center of Mass Displacements" table."""
        cm = self.diaphragm_center_of_mass(diaphragm, story)
        ux, uy = self.point_displacement_at(cm, diaphragm, story, case)
        rz = self.diaphragm_displacement(diaphragm, story, case)[2]
        return np.array([ux, uy, rz])

    # ------------------------------------------------------------ shells
    SHELL_COMPONENTS = ("F11", "F22", "F12", "M11", "M22", "M12", "V13", "V23")

    def _shell_factor(self) -> np.ndarray:
        fl = self.U.factor("force_per_length")
        fm = self.U.factor("force")  # moment per length = force
        return np.array([1 / fl, 1 / fl, 1 / fl, 1 / fm, 1 / fm, 1 / fm, 1 / fl, 1 / fl])

    SHELL_CACHE_ITEMS = 6  # cases / combinations whose shell resultants stay prepared

    def _shell_cache(self) -> "OrderedDict":
        c = getattr(self, "_shell_lru", None)
        if c is None:
            from collections import OrderedDict
            c = self._shell_lru = OrderedDict()
        return c

    def _cached(self, key, make):
        """A small LRU of derived shell results (the whole model, all components):
        switching component / window / story reuses them instead of rebuilding."""
        c = self._shell_cache()
        if key in c:
            c.move_to_end(key)
            return c[key]
        v = make()
        c[key] = v
        while len(c) > self.SHELL_CACHE_ITEMS:
            c.popitem(last=False)
        return v

    def _shell_stack(self, case: str) -> Tuple[np.ndarray, "Envelope"]:
        """Element ids and the Envelope of the stacked (n, 4, 8) nodal resultants
        (model units) of a case or combination - computed once, read-only."""
        fac = self._shell_factor()
        key = ("stack", case, tuple(np.round(fac, 15)))

        def make():
            if case in self.model.load_combinations and case not in self.cases:
                combo = self._combo(case)
                parts = {it.name: self._shell_stack(it.name) for it in combo.items}  # each case once
                common = None
                for ids_, _ in parts.values():
                    common = set(ids_.tolist()) if common is None else common & set(ids_.tolist())
                ids = np.array(sorted(common or ()), dtype=int)

                def pick(name):
                    ids_n, env = parts[name]
                    pos = np.searchsorted(ids_n, ids)
                    return Envelope(env.max[pos], env.min[pos])
                env = self._combine(combo, pick) if len(ids) else Envelope(np.zeros((0, 4, 8)), np.zeros((0, 4, 8)))
            else:
                r = self._case(case)
                ids = np.array(sorted(r.shell_resultants), dtype=int)
                if len(ids):
                    sr = r.shell_resultants
                    vals = np.concatenate([np.stack([sr[i]["F_nodes"] for i in ids]),
                                           np.stack([sr[i]["M_nodes"] for i in ids]),
                                           np.stack([sr[i]["V_nodes"] for i in ids])], axis=2) * fac
                else:
                    vals = np.zeros((0, 4, 8))
                env = Envelope.symmetric(vals) if r.is_envelope else Envelope.single(vals)
            for a in (env.max, env.min):
                a.setflags(write=False)  # shared by every view: must not be changed in place
            return ids, env
        return self._cached(key, make)

    def shell_element_envelope(self, case: str) -> Dict[int, Envelope]:
        """Element id -> Envelope of (4 nodes, 8) nodal stress resultants
        F11,F22,F12 (force/length), M11,M22,M12 (moment/length), V13,V23."""
        ids, env = self._shell_stack(case)
        return {int(e): Envelope(env.max[k], env.min[k]) for k, e in enumerate(ids)}

    def _combo_shells(self, name: str) -> Dict[int, Envelope]:
        return self.shell_element_envelope(name)

    def shell_forces(self, case: str, areas: Optional[Iterable[str]] = None) -> List[dict]:
        """Table like ETABS "Shell Forces": one row per element node."""
        t = self.shell_forces_table(case, areas)
        return [t.row(r) for r in range(len(t))]

    def shell_forces_table(self, case: str, areas: Optional[Iterable[str]] = None) -> "ShellForcesTable":
        """The rows of :meth:`shell_forces` as arrays (a row made only when asked
        for): a large model has hundreds of thousands of element nodes."""
        ids, env = self._shell_stack(case)
        key = ("shell_rows", len(ids), None if areas is None else frozenset(areas))

        def make():
            # element nodes in the order of the mesh (shell by shell, node by node)
            sh = self.am.shells
            sid = np.fromiter((se.id for se in sh), np.int64, len(sh))
            lens = np.fromiter((len(se.nodes) for se in sh), np.int64, len(sh))
            conn = np.fromiter((n for se in sh for n in se.nodes), np.int64, int(lens.sum()))
            objs = np.array([se.object for se in sh], dtype=object)
            pos = np.searchsorted(ids, sid) if len(ids) else np.zeros(len(sh), np.int64)
            keep = (pos < len(ids)) & (ids[np.minimum(pos, max(len(ids) - 1, 0))] == sid) if len(ids) \
                else np.zeros(len(sh), bool)
            if areas is not None:
                wanted = set(areas)
                keep &= np.fromiter((o in wanted for o in objs), bool, len(sh))
            start = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
            n_rows = lens[keep]
            el = np.repeat(pos[keep], n_rows)
            first = np.repeat(start[keep], n_rows)
            kk = np.arange(int(n_rows.sum()), dtype=np.int64) - np.repeat(
                np.concatenate([[0], np.cumsum(n_rows)[:-1]]).astype(np.int64), n_rows)
            node = conn[first + kk]
            eid = np.repeat(sid[keep], n_rows)
            obj = np.repeat(objs[keep], n_rows)
            arrays = (el.astype(np.int64), kk, node, eid, obj)
            for a in arrays:
                a.setflags(write=False)
            return arrays
        el, kk, node, eid, obj = self._cached(key, make)
        # Min columns for the elements whose minimum differs (as np.allclose decides)
        if env.min is None or not len(ids):
            two = np.zeros(len(ids), bool)
        else:
            two = ~np.isclose(env.min, env.max).reshape(len(ids), -1).all(axis=1)
        return ShellForcesTable(case, self.SHELL_COMPONENTS, env, two, el, kk, node, eid, obj)

    def shell_node_values(self, case: str, component: str, average: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Per-element nodal values (n_shells, 4) of one component (max of the
        envelope) and the per-element node ids (n_shells, 4); nodal averaging
        over the elements of the same area object when ``average``."""
        j = self.SHELL_COMPONENTS.index(component)
        fac = self._shell_factor()
        key = ("nodes", case, component, bool(average), tuple(np.round(fac, 15)))

        def make():
            ids, env = self._shell_stack(case)
            have = set(ids.tolist())
            shells = [se for se in self.am.shells if se.id in have]
            if not shells:
                return np.zeros((0, 4)), np.zeros((0, 4), int)
            pos = np.searchsorted(ids, np.array([se.id for se in shells]))
            vals = np.array(env.max[pos, :, j], float)
            nodes = np.array([se.nodes for se in shells], int)
            if average:
                # the slab inside a column (rigid zone) does not bend: its elements do not
                # take part in the averaging, so the values at the column face are the slab's
                rigid = self.rigid_zone_shells()
                use = np.array([se.id not in rigid for se in shells])
                objs = {}
                obj = np.array([objs.setdefault(se.object, len(objs)) for se in shells], int)
                # one key per (area object, node): averages never cross objects
                keys = obj[:, None].astype(np.int64) * (int(nodes.max()) + 1) + nodes
                k_use = keys[use].ravel()
                uniq, inv = np.unique(k_use, return_inverse=True)
                tot = np.bincount(inv, weights=vals[use].ravel(), minlength=len(uniq))
                cnt = np.bincount(inv, minlength=len(uniq))
                vals[use] = (tot / cnt)[inv].reshape(-1, 4)
            vals.setflags(write=False)
            nodes.setflags(write=False)
            return vals, nodes
        vals, nodes = self._cached(key, make)
        return vals.copy(), nodes.copy()

    def rigid_zone_shells(self) -> set:
        """Ids of the shell elements lying inside a column's cross-section (all
        their nodes in one column-slab rigid zone)."""
        out = getattr(self, "_rigid_shells", None)
        if out is None:
            body_of = {}
            for k, b in enumerate(getattr(self.am, "bodies", []) or []):
                body_of[b.master] = k
                for n in b.nodes:
                    body_of[n] = k
            out = set()
            if body_of:
                for se in self.am.shells:
                    ks = {body_of.get(n, -1 - i) for i, n in enumerate(se.nodes)}
                    if len(ks) == 1:
                        out.add(se.id)
            self._rigid_shells = out
        return out

    # ----------------------------------------------------------- combos
    def _combo(self, name: str) -> LoadCombination:
        try:
            return self.model.load_combinations[name]
        except KeyError:
            raise ModelError("UNKNOWN_COMBO", f"Combination {name!r} is not defined")

    def _combine(self, combo: LoadCombination, getter) -> Envelope:
        """Generic CSI combination over Envelope-valued contributions."""
        parts: List[Envelope] = []
        for it in combo.items:
            env = getter(it.name)
            parts.append(env.scaled(it.scale))
        if not parts:
            raise ModelError("EMPTY_COMBO", f"Combination {combo.name!r} has no items")
        t = combo.type.replace(" ", "").lower()
        if t in ("linearadd", "additive", "add"):
            mx = sum(p.max for p in parts)
            mn = sum(p.min for p in parts)
            return Envelope(mx, mn)
        if t in ("envelope",):
            return Envelope(np.max([p.max for p in parts], axis=0), np.min([p.min for p in parts], axis=0))
        if t in ("absolute", "abs"):
            mx = sum(np.maximum(np.abs(p.max), np.abs(p.min)) for p in parts)
            return Envelope(mx, -mx)
        if t in ("srss",):
            mx = np.sqrt(sum(np.maximum(np.abs(p.max), np.abs(p.min)) ** 2 for p in parts))
            return Envelope(mx, -mx)
        if t in ("range",):
            mx = sum(np.maximum(p.max, 0.0) for p in parts)
            mn = sum(np.minimum(p.min, 0.0) for p in parts)
            return Envelope(mx, mn)
        raise ModelError("INVALID_COMBO", f"Unknown combination type {combo.type!r}")

    def _combo_displacements(self, name: str) -> Envelope:
        return self._combine(self._combo(name), self.displacement_envelope)

    def _combo_reactions(self, name: str) -> Dict[int, Envelope]:
        combo = self._combo(name)
        nodes = set()
        tables = {}
        for it in combo.items:
            tables[it.name] = self.reaction_envelope(it.name)
            nodes |= set(tables[it.name])
        out = {}
        for nid in nodes:
            out[nid] = self._combine(combo, lambda n: tables[n].get(nid, Envelope.single(np.zeros(6))))
        return out

    def _combo_frame_forces(self, frame: str, name: str) -> FrameStationForces:
        combo = self._combo(name)

        def getter(n):
            st = self.frame_forces(frame, n)
            if st.forces_min is None:
                return Envelope.single(st.forces)
            return Envelope(st.forces, st.forces_min)

        env = self._combine(combo, getter)
        st0 = self.frame_forces(frame, combo.items[0].name)
        return FrameStationForces(frame, name, st0.station, env.max, env.min)

    # ---------------------------------------------------------- export
    def to_dict(self, cases: Optional[Iterable[str]] = None) -> dict:
        names = list(cases) if cases else list(self.cases)
        out = {"units": f"{self.U.force}-{self.U.length}", "cases": {}}
        for n in names:
            r = self.cases[n]
            entry = {"type": r.type, "joint_displacements": self.joint_displacements(n),
                     "joint_reactions": self.joint_reactions(n), "warnings": r.warnings}
            if r.modal is not None and r.type == "Modal":
                entry["modal_periods"] = self.modal_periods(n)
                entry["modal_mass_ratios"] = self.modal_participating_mass_ratios(n)
            if r.type != "Modal":
                entry["frame_forces"] = [row for f in self.model.frames for row in self.frame_forces(f, n).table()]
            out["cases"][n] = entry
        return out
