"""Object model -> finite element analysis model.

Responsibilities (all in SI):

* node creation and merging;
* frame objects: local axes, insertion points, end offsets / rigid zones,
  auto-meshing at points, intersections and floor mesh nodes;
* area objects: structured quad meshing (rectangular or polygon-to-quads);
* rigid diaphragm constraints with virtual master nodes;
* object loads -> element / nodal loads; self weight; mass source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from . import area_mesh as am_
from . import openings as opn
from . import polygon2d as pg

from ..model import Model, ModelError
from ..model.loads import LoadPattern, MassSource
from ..model.objects import Area, Frame, Point
from ..model.properties import FrameModifiers, FrameSection, ShellModifiers, ShellSection
from ..units import UnitSystem
from . import tendons as TD
from .panel_loads import membrane_floors, panel_loads_to_members
from .analysis_model import (AnalysisModel, Body, Constraint, RigidLink, ElementLoad, FrameElement, Node, ShellElement,
                             ShellLoad, Tie)
from .geometry import (beam_axis_shift, column_containing, cross3, member_through, frame_axes, joint_axes, point_on_segment, polygon_centroid, polygon_normal, polygon_polar_inertia,
                       quad_tributary_areas, segment_intersection, shell_axes, transformed_axes, unit)

G_SI = 9.80665

# cardinal point -> (offset of centroid from the cardinal point) in local (2, 3)
# expressed as multiples of (t3 = depth along 2, t2 = width along 3)
# ("left" cardinal points lie on the -3 side; verified with SAP2000 1-011)
_CARDINAL = {
    1: (0.5, 0.5), 2: (0.5, 0.0), 3: (0.5, -0.5),
    4: (0.0, 0.5), 5: (0.0, 0.0), 6: (0.0, -0.5),
    7: (-0.5, 0.5), 8: (-0.5, 0.0), 9: (-0.5, -0.5),
    10: (0.0, 0.0), 11: (0.0, 0.0),
}


def _bilinear_weights(P: np.ndarray, x: np.ndarray, nrm: np.ndarray, tol: float):
    """Shape-function weights of point ``x`` in the planar quad ``P`` (None if
    outside): Newton inversion of the bilinear map in the quad's plane."""
    e1 = P[1] - P[0]
    e1 /= np.linalg.norm(e1) or 1.0
    e2 = cross3(nrm, e1)
    Q = np.column_stack([(P - P[0]) @ e1, (P - P[0]) @ e2])
    q = np.array([(x - P[0]) @ e1, (x - P[0]) @ e2])
    s = t = 0.0
    for _ in range(30):
        N = np.array([(1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t])
        dNs = np.array([-(1 - t), (1 - t), t, -t])
        dNt = np.array([-(1 - s), -s, s, (1 - s)])
        r = N @ Q - q
        J = np.column_stack([dNs @ Q, dNt @ Q])
        try:
            d = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return None
        s, t = s - d[0], t - d[1]
        if np.linalg.norm(d) < 1e-12:
            break
    size = max(np.linalg.norm(Q[2] - Q[0]), 1e-12)
    eps = tol / size
    if not (-eps <= s <= 1 + eps and -eps <= t <= 1 + eps):
        return None
    s, t = min(max(s, 0.0), 1.0), min(max(t, 0.0), 1.0)
    return np.array([(1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t])


@dataclass
class MeshOptions:
    floor_max_size: Optional[float] = None  # SI; None => taken from model options or 4 divisions
    wall_max_size: Optional[float] = None
    frame_intersections: bool = True
    default_divisions: int = 4
    # no mesh size given: choose one per shell from its geometry (spans between
    # columns / supports, narrowest width) - see area_mesh.auto_size
    auto_size: bool = True
    size_factor: float = 1.0  # scales the automatic sizes (mesh refinement check)
    # api.analyze: mesh refinement check - halve the automatic sizes until the
    # estimated error of the largest shell displacement of every linear static
    # case is below ``refine_tolerance`` (see api._refine_until_converged)
    refine_check: bool = False  # Analyze > Check Mesh (api.check_mesh) runs it on purpose
    refine_tolerance: float = 0.02
    refine_levels: int = 2  # at most this many halvings
    refine_max_shells: int = 12000  # never refine past this many shell elements (time)
    # areas assigned the fine mesh (Assign > Floor / Wall Auto Mesh) use it; off for the
    # mesh refinement check (it looks for the fine mesh) and the legacy global fine mesh
    area_fine: bool = True
    # membrane floors on auto mesh: no element in a rigid diaphragm, no slab-column rigid
    # zones (False: meshed like a shell - the reference of the tests)
    membrane_coarse: bool = True


class Mesher:
    def __init__(self, model: Model, options: Optional[MeshOptions] = None):
        self.model = model
        self.U: UnitSystem = model.units
        self.opts = options or MeshOptions()
        self.tol = model.options.merge_tolerance * self.U.factor("length")
        self.am = AnalysisModel()
        self._node_index: Dict[Tuple[int, int, int], int] = {}
        self._frame_cuts: Dict[str, List[float]] = {}  # frame name -> parameters t of extra nodes
        self._unmeshed_membranes: set = set()  # membrane floors in a rigid diaphragm: no elements
        self._membrane_nodes: Dict[str, set] = {}  # their joints (in their diaphragm)

        self._frame_geom: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._diaphragm_nodes: Dict[Tuple[str, Optional[str], float], set] = {}
        # per diaphragm key: list of (polygon pts, mass per area, {pattern: force per area (gravity)},
        #                             {node id: tributary area})
        self._diaphragm_areas: Dict[Tuple[str, Optional[str], float], list] = {}
        self._openings = None

    # ============================================================== public
    def build(self) -> AnalysisModel:
        m = self.model
        m.require_valid()
        am = self.am
        am.active_dofs = np.array(m.options.active_dofs, dtype=bool)
        if m.options.gravity:
            am.gravity = float(m.options.gravity) * self.U.factor("length")
        am.story_elevations = {s.name: s.elevation * self.U.factor("length") for s in m.stories.values()}
        if m.multi_tower:
            am.story_towers = {s.name: m.tower_of_story(s.name) for s in m.stories.values()}
        for p in m.load_patterns.values():
            if p.self_weight_multiplier:
                am.self_weight[p.name] = float(p.self_weight_multiplier)
        self._apply_mass_source()
        # load panels -> frame / area loads (model units), before any element is built
        self._panel_frame_loads, self._panel_area_loads, am.panel_report = panel_loads_to_members(m)
        # membrane floors: their self weight and the loads normal to them went to the frames
        self._membrane_floors = set(membrane_floors(m))
        for name, rep in am.panel_report.items():
            for pat, (applied, carried) in rep.items():
                if abs(applied) > 1e-12 and abs(carried - applied) > 1e-3 * abs(applied):
                    am.warnings.append(
                        f"Load {name}, pattern {pat}: {100.0 * (1.0 - carried / applied):.1f}% of the load is not applied "
                        f"(no floor / beam under that part of the panel, or an opening there)")

        # 1. nodes for point objects.  Points used by frames / areas come first; an
        #    unconnected point that coincides with another point object stays a separate
        #    joint (ETABS keeps distinct point objects distinct: a load point placed on a
        #    wall corner is tied to the structure only through its diaphragm).
        used = {n for f in m.frames.values() for n in (f.point_i, f.point_j)}
        for a in m.areas.values():
            used.update(a.points)
        self._separate_points = set()
        ordered = [p for p in m.points.values() if p.name in used] + [p for p in m.points.values() if p.name not in used]
        for p in ordered:
            xyz = self._xyz(p)
            nid = None
            if p.name not in used:
                existing = self._find_node(xyz)
                if existing is not None and am.nodes[existing].point not in (None, p.name):
                    nid = len(am.nodes)
                    am.nodes.append(Node(id=nid, xyz=np.array(xyz, float), point=p.name, story=p.story))
                    am.point_to_node[p.name] = nid
                    self._separate_points.add(p.name)
            if nid is None:
                nid = self._node_at(xyz, point=p.name, story=p.story)
            node = am.nodes[nid]
            node.restraints |= np.array(p.restraints, dtype=bool)
            node.springs += np.concatenate([
                np.array(p.springs, float) * self.U.factor("translational_stiffness"),
                np.array(p.rotational_springs, float) * self.U.factor("rotational_stiffness"),
            ])
            node.mass += np.concatenate([
                np.array(p.mass, float) * self.U.factor("mass"),
                np.array(p.rotational_mass, float) * self.U.factor("rotational_mass"),
            ])
            if p.diaphragm:
                self._diaphragm_nodes.setdefault((p.diaphragm, p.story, round(node.xyz[2] / self.tol)), set()).add(nid)
            if p.local_axes is not None and any(p.local_axes):
                node.local_axes = joint_axes(*p.local_axes)
            for gd in p.ground_displacements:
                vec = np.concatenate([np.array(gd.translations, float) * self.U.factor("length"),
                                      np.array(gd.rotations, float)])
                if node.local_axes is not None and str(gd.coord_sys).lower() != "local":
                    R = node.local_axes  # stored in the joint local axes
                    vec = np.concatenate([R @ vec[:3], R @ vec[3:]])
                node.ground_disp[gd.pattern] = node.ground_disp.get(gd.pattern, np.zeros(6)) + vec

        # 2. frame geometry & cuts at points / intersections
        # a beam whose insertion offsets are in the analysis is meshed on its true axis
        # (horizontal part of the offsets; floors share its nodes there) and joined to
        # its joints by rigid links; _frame_line keeps the joint-to-joint line
        self._frame_line: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._axis_shift: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._joint_cuts: Dict[str, Dict[float, int]] = {}
        tol_model = self.tol / self.U.factor("length")
        for f in m.frames.values():
            pi = self._xyz(m.points[f.point_i])
            pj = self._xyz(m.points[f.point_j])
            self._frame_line[f.name] = (pi, pj)
            sh = beam_axis_shift(m, f, tol_model)
            if sh is not None:
                hi, hj = sh[0] * self.U.factor("length"), sh[1] * self.U.factor("length")
                self._axis_shift[f.name] = (hi, hj)
                pi, pj = pi + hi, pj + hj
            self._frame_geom[f.name] = (pi, pj)
            self._frame_cuts[f.name] = []
            self._joint_cuts[f.name] = {}
        if self._axis_shift:
            am.info.append(f"{len(self._axis_shift)} beam(s) meshed on their true (offset) axis, "
                           f"joined to their joints by rigid links")
        self._cut_frames_at_points()
        self._frames_into: Dict[Tuple[str, int], str] = {}
        self._find_axis_ends_on_members()
        if self.opts.frame_intersections:
            self._cut_frames_at_intersections()

        # 3. areas -> shells (also cuts frames at floor mesh nodes); openings cut holes
        self._openings = opn.OpeningIndex(m, self.U.factor("length"))
        self._prepare_features()
        self._area_order: Dict[str, int] = {}
        # floors first: walls standing on / hanging from them then take the floor
        # nodes along their edges, so the two meshes match
        areas = [a for a in sorted(m.areas.values(), key=lambda a: a.type != "Floor") if not a.is_opening]
        prog = getattr(self, "progress", None)
        for k, a in enumerate(areas):
            if prog is not None and (k % max(1, len(areas) // 50) == 0):
                prog(0.1 + 0.6 * k / max(len(areas), 1), f"Meshing {a.type.lower()} {a.name} ({k + 1}/{len(areas)})")
            self._mesh_area(a)
        if prog is not None:
            prog(0.72, "Tying joints and edges to the shells (auto edge constraints)")
        # anything left between mesh nodes (a joint, a member end, the edge of a
        # shell meshed later) follows the element it lies in: no floating parts
        self._tie_to_shells()
        self._report_untied_joints()

        # 4. frames -> elements (bonded "Elements" tendons cut their frames first)
        self._internal_nodes = set()
        self._tendon_points = {t.name: self._tendon_samples(t) for t in m.tendons.values()}
        for t in m.tendons.values():
            if t.model_as.lower().startswith("elem"):
                self._cut_frames_for_tendon(t)
        for f in m.frames.values():
            self._build_frame_elements(f)
        for t in m.tendons.values():
            self._apply_tendon(t)

        self._check_beams_beside_floors()

        # 4a. line springs -> nodal springs by tributary length along the line
        self._apply_line_springs()

        # 4b. ETABS-style automatic restraints of generated nodes at the base level
        if m.options.auto_base_restraints and m.stories:
            base_z = min(am.story_elevations.values())
            for node in am.nodes:
                if node.point is None and node.id not in self._internal_nodes and abs(node.xyz[2] - base_z) <= self.tol:
                    node.restraints[:] = True

        # several towers: the tower of every node (from its point / the objects of its
        # elements) and the story of generated nodes at a level of their own tower
        if m.multi_tower:
            self._assign_node_towers()
            # a tower standing on the ground at its own (higher) base level - all its joints
            # there supported - has its generated base joints restrained too (a tower on a
            # podium has not: its base joints are the podium's)
            if m.options.auto_base_restraints and am.story_elevations:
                base_z = min(am.story_elevations.values())
                for t in m.tower_names():
                    ss = m.stories_of(t)
                    if not ss:
                        continue
                    zt = ss[0].elevation * self.U.factor("length")
                    if abs(zt - base_z) <= self.tol:
                        continue
                    pts = [n for n in am.nodes if n.point is not None and n.tower == t
                           and abs(n.xyz[2] - zt) <= self.tol]
                    if not pts or not all(n.restraints[:3].all() for n in pts):
                        continue
                    for node in am.nodes:
                        if node.point is None and node.tower == t and node.id not in self._internal_nodes \
                                and abs(node.xyz[2] - zt) <= self.tol:
                            node.restraints[:] = True

        # 4c. generated nodes at a story level join the diaphragm defined at that level
        # (CSI: wall/floor mesh joints at a story are part of the story diaphragm) - of
        # their own tower when the model has several
        level_diaphragms = {}
        for (dname, story, zkey), nodes in self._diaphragm_nodes.items():
            tw = am.story_towers.get(story) if story is not None else None
            if tw is None and nodes:
                tw = am.nodes[next(iter(nodes))].tower
            level_diaphragms.setdefault((tw, zkey), (dname, story))
        for node in am.nodes:
            if node.point is not None or node.id in self._internal_nodes:
                continue
            zkey = round(node.xyz[2] / self.tol)
            hit = level_diaphragms.get((node.tower, zkey))
            if hit is not None:
                self._diaphragm_nodes.setdefault((hit[0], hit[1], zkey), set()).add(node.id)

        # 4d. slab inside each column's cross-section: a rigid zone
        if m.options.slab_column_rigid_zone:
            self._column_rigid_zones()
            self._releases_at_column_faces()

        # 5. constraints
        self._build_constraints()

        # 6. point loads
        for p in m.points.values():
            nid = am.point_to_node[p.name]
            for ld in p.loads:
                vec = np.concatenate([np.array(ld.force, float) * self.U.factor("force"),
                                      np.array(ld.moment, float) * self.U.factor("moment")])
                am.add_nodal_load(ld.pattern, nid, vec)
        return am

    def _apply_line_springs(self):
        m, am = self.model, self.am
        if not m.line_springs:
            return
        kf = self.U.factor("force_per_area")
        rf = self.U.factor("force")
        for ls in m.line_springs.values():
            if ls.point_i not in m.points or ls.point_j not in m.points:
                continue
            a, b = self._xyz(m.points[ls.point_i]), self._xyz(m.points[ls.point_j])
            L = float(np.linalg.norm(b - a))
            if L <= self.tol:
                continue
            e = (b - a) / L
            hits = {0.0: am.point_to_node[ls.point_i], 1.0: am.point_to_node[ls.point_j]}
            for node in am.nodes:
                if node.is_master:
                    continue
                t = point_on_segment(node.xyz, a, b, self.tol)
                if t is not None:
                    hits.setdefault(round(t, 12), node.id)
            ts = sorted(hits)
            k = np.array(ls.stiffness, float) * kf
            kr = float(ls.rotational) * rf
            for idx, t in enumerate(ts):
                lo = ts[idx - 1] if idx > 0 else t
                hi = ts[idx + 1] if idx + 1 < len(ts) else t
                trib = 0.5 * (hi - lo) * L
                node = am.nodes[hits[t]]
                node.springs[:3] += k * trib
                if kr:
                    node.springs[3:6] += kr * trib * e ** 2

    # ============================================================ helpers
    def _xyz(self, p: Point) -> np.ndarray:
        return np.array([p.x, p.y, p.z], float) * self.U.factor("length")

    def _key(self, xyz: np.ndarray) -> Tuple[int, int, int]:
        return tuple(int(round(c / self.tol)) for c in xyz)

    def _find_node(self, xyz: np.ndarray) -> Optional[int]:
        key = self._key(xyz)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand = self._node_index.get((key[0] + dx, key[1] + dy, key[2] + dz))
                    if cand is not None and np.linalg.norm(self.am.nodes[cand].xyz - xyz) <= self.tol:
                        return cand
        return None

    def _node_at(self, xyz: np.ndarray, point: Optional[str] = None, story: Optional[str] = None) -> int:
        """Return the node id at ``xyz``; create it if needed (merging within tol)."""
        key = self._key(xyz)
        nid = self._node_index.get(key)
        if nid is None:
            # second chance: neighbouring keys (rounding boundary)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        cand = self._node_index.get((key[0] + dx, key[1] + dy, key[2] + dz))
                        if cand is not None and np.linalg.norm(self.am.nodes[cand].xyz - xyz) <= self.tol:
                            nid = cand
                            break
                    if nid is not None:
                        break
                if nid is not None:
                    break
        if nid is None:
            nid = len(self.am.nodes)
            if story is None:
                story = self._story_of(xyz[2])
            self.am.nodes.append(Node(id=nid, xyz=np.array(xyz, float), point=point, story=story))
            self._node_index[key] = nid
        node = self.am.nodes[nid]
        if point is not None:
            node.point = point
            self.am.point_to_node[point] = nid
        if story is not None and node.story is None:
            node.story = story
        return nid

    def _assign_node_towers(self):
        """Several towers: node.tower for every node and, for generated nodes, the story
        at their level in that tower (None between its levels)."""
        m, am = self.model, self.am
        cache = {}

        def tw(kind, name):
            key = (kind, name)
            if key not in cache:
                try:
                    cache[key] = m.tower_of_object(kind, name)
                except KeyError:
                    cache[key] = None
            return cache[key]

        for fe in am.frames:
            t = tw("frame", fe.object)
            for n in (fe.node_i, fe.node_j):
                if am.nodes[n].tower is None:
                    am.nodes[n].tower = t
        for se in am.shells:
            t = tw("area", se.object)
            for n in se.nodes:
                if am.nodes[n].tower is None:
                    am.nodes[n].tower = t
        for name, nodes in self._membrane_nodes.items():
            t = tw("area", name)
            for n in nodes:
                if am.nodes[n].tower is None:
                    am.nodes[n].tower = t
        for node in am.nodes:
            if node.point is not None and node.point in m.points:
                p = m.points[node.point]
                node.tower = m.tower_of_story(p.story) if p.story in m.stories else (node.tower or m.tower_of_point(p.name))
            elif node.tower is None:
                node.tower = m.default_tower()
            if node.point is None and node.story is not None and am.story_towers.get(node.story) != node.tower:
                node.story = None
                for sname, elev in am.story_elevations.items():
                    if am.story_towers.get(sname) == node.tower and abs(elev - node.xyz[2]) <= self.tol:
                        node.story = sname
                        break

    def _story_of(self, z: float) -> Optional[str]:
        for name, elev in self.am.story_elevations.items():
            if abs(elev - z) <= self.tol:
                return name
        return None

    # ---------------------------------------------------------- mass source
    def _apply_mass_source(self):
        ms: Optional[MassSource] = None
        for cand in self.model.mass_sources.values():
            if cand.is_default:
                ms = cand
                break
        if ms is None and self.model.mass_sources:
            ms = next(iter(self.model.mass_sources.values()))
        if ms is None:
            return
        am = self.am
        am.mass_include_elements = ms.include_element_mass
        am.mass_include_added = ms.include_added_mass
        am.mass_lateral = ms.include_lateral
        am.mass_vertical = ms.include_vertical
        am.mass_from_loads = {k: float(v) for k, v in ms.load_patterns.items()}
        am.mass_lump_at_stories = bool(getattr(ms, "lump_at_stories", False))

    MEMBRANE_BEAM_DIVISION = 0.75  # m: beams under a membrane floor without elements, divided (mass)

    def _frames_at_levels(self):
        """(level key -> frame end points there, level key -> horizontal frames there):
        found once (membrane floors without elements look for what is under them)."""
        c = self.__dict__.get("_levels_cache")
        if c is None:
            ends: Dict[int, list] = {}
            flat: Dict[int, list] = {}
            for f in self.model.frames.values():
                ga, gb = self._frame_geom[f.name]
                ka, kb = round(float(ga[2]) / self.tol), round(float(gb[2]) / self.tol)
                ends.setdefault(ka, []).append(ga)
                ends.setdefault(kb, []).append(gb)
                if ka == kb and f.type not in ("Rigid", "Null"):
                    flat.setdefault(ka, []).append(f.name)
            c = self._levels_cache = (ends, flat)
        return c

    def _column_joints(self) -> set:
        c = self.__dict__.get("_col_joints")
        if c is None:
            c = self._col_joints = {p for g in self.model.frames.values() if g.type == "Column"
                                    for p in (g.point_i, g.point_j)}
        return c

    def _end_offsets_of(self, f) -> Tuple[float, float]:
        cache = self.__dict__.setdefault("_end_off_cache", {})
        if f.name not in cache:
            m = self.model
            try:
                ga, gb = self._frame_geom[f.name]
                cache[f.name] = (self._auto_end_offsets(f, ga, gb) if m.frame_auto_end_offsets(f)
                                 else (f.end_offset_i * self.U.factor("length"), f.end_offset_j * self.U.factor("length")))
            except Exception:
                cache[f.name] = (0.0, 0.0)
        return cache[f.name]

    def _membrane_auto(self, a: Area) -> bool:
        """A horizontal membrane floor on auto mesh (no element in a rigid diaphragm,
        coarse elsewhere - see _mesh_area)."""
        if not self.opts.membrane_coarse or a.type != "Floor" or not a.auto_mesh or a.max_mesh_size \
                or getattr(a, "fine_mesh", False) or a.mesh_divisions is not None:
            return False
        if a.name not in getattr(self, "_membrane_floors", ()):
            return False
        sec = self.model.shell_sections.get(a.section)
        if sec is None or sec.model_type != "Membrane":
            return False
        try:
            P = np.array([self._xyz(self.model.points[p]) for p in a.points])
            return abs(polygon_normal(P)[2]) > 1 - 1e-6
        except Exception:
            return False

    # ------------------------------------------------------------ frame cuts
    def _cut_frames_at_points(self):
        pts = [(self.am.point_to_node[p.name], self._xyz(p)) for p in self.model.points.values()
               if p.name not in getattr(self, "_separate_points", ())]
        if not pts:
            return
        X = np.array([xyz for _, xyz in pts], float)
        tol = self.tol
        # joints rigidly linked to a column's joint from inside that column (a beam moved to
        # its real axis within the joint zone): a member framing into that column is not cut
        # there - that part of it is in the joint zone already (its end offset), exactly as
        # with the same beam analysed on its offset axis
        in_zone: Dict[str, set] = {}
        Lf = self.U.factor("length")
        for g in self.model.frames.values():
            if g.type != "Rigid":
                continue
            for e, q in ((g.point_i, g.point_j), (g.point_j, g.point_i)):
                if column_containing(self.model, e, self._xyz(self.model.points[q]), tol, Lf, self._xyz) is not None:
                    in_zone.setdefault(e, set()).add(self.am.point_to_node[q])
        for f in self.model.frames.values():
            if not (f.auto_mesh and f.mesh_at_intersections) or f.type == "Rigid":
                continue
            a, b = self._frame_geom[f.name]
            skip = in_zone.get(f.point_i, set()) | in_zone.get(f.point_j, set()) if in_zone and f.type != "Column" else ()
            # broad phase: the joints in the frame's box (+tol); the exact test is unchanged
            lo, hi = np.minimum(a, b) - tol, np.maximum(a, b) + tol
            near = np.nonzero(np.all((X >= lo) & (X <= hi), axis=1))[0]
            for k in near:
                if skip and pts[int(k)][0] in skip and pts[int(k)][0] not in (self.am.point_to_node[f.point_i],
                                                                              self.am.point_to_node[f.point_j]):
                    continue
                t = point_on_segment(X[k], a, b, tol)
                if t is not None:
                    self._frame_cuts[f.name].append(t)
            if f.name in self._axis_shift:
                # joints on the joint-to-joint line (a member framing in there): the beam
                # gets a node on its axis at the same place, rigidly linked to the joint
                a, b = self._frame_line[f.name]
                lo, hi = np.minimum(a, b) - tol, np.maximum(a, b) + tol
                near = np.nonzero(np.all((X >= lo) & (X <= hi), axis=1))[0]
                for k in near:
                    t = point_on_segment(X[k], a, b, tol)
                    if t is not None and 0.0 < t < 1.0:
                        self._frame_cuts[f.name].append(t)
                        self._joint_cuts[f.name][round(t, 12)] = pts[int(k)][0]

    def _find_axis_ends_on_members(self):
        """A beam meshed on its true axis whose axis end (or the axis node of a joint
        on its line) lies outside the column at the joint and on a member framing into
        that joint - the side beam, typically - frames into that member there, as the
        beam really is built: that member gets a node there and no rigid link joins the
        two (a link along it would make that part of it rigid).  The same rule as Move
        Beams to Real Axis, so both give the same model."""
        if not self._axis_shift:
            return
        m, Lf = self.model, self.U.factor("length")
        node_point = {nid: pn for pn, nid in self.am.point_to_node.items()}
        xyz_of = self._xyz
        for name, (hi, hj) in self._axis_shift.items():
            f = m.frames[name]
            a, b = self._frame_line[name]
            anchors = [(f.point_i, a + hi), (f.point_j, b + hj)]
            for t, nid in self._joint_cuts.get(name, {}).items():
                pn = node_point.get(nid)
                if pn is not None:
                    anchors.append((pn, xyz_of(m.points[pn]) + (1.0 - t) * hi + t * hj))
            for pn, q in anchors:
                if column_containing(m, pn, q, self.tol, Lf, xyz_of) is not None:
                    continue  # in the column's joint zone: the rigid link stays
                g = member_through(m, pn, q, self.tol, exclude=set(self._axis_shift) | {name},
                                   geom=lambda n: self._frame_geom[n], xyz_of=xyz_of)
                if g is None:
                    continue
                ga, gb = self._frame_geom[g]
                t = point_on_segment(q, ga, gb, self.tol)
                self._frame_cuts[g].append(t)
                self._frames_into[(name, self.am.point_to_node[pn])] = g
                self.am.info.append(f"Beam {name}: its axis frames into {g} {np.linalg.norm(q - xyz_of(m.points[pn])) / Lf:.3f} "
                                    f"from joint {pn} (outside the column) - joined there, no rigid link")

    def _cut_frames_at_intersections(self):
        names = [f.name for f in self.model.frames.values() if f.auto_mesh and f.mesh_at_intersections
                 and f.type != "Rigid"]
        null = {f.name for f in self.model.frames.values() if f.type == "Null"}
        if not names:
            return
        A = np.array([self._frame_geom[nm][0] for nm in names], float)
        B = np.array([self._frame_geom[nm][1] for nm in names], float)
        LO, HI = np.minimum(A, B), np.maximum(A, B)
        tol = self.tol
        for i in range(len(names)):
            a, b = self._frame_geom[names[i]]
            # the same bounding box reject as before, for all later frames at once
            later = np.arange(i + 1, len(names))
            ok = np.all((HI[i + 1:] >= LO[i] - tol) & (HI[i] >= LO[i + 1:] - tol), axis=1)
            for j in later[ok]:
                j = int(j)
                if names[i] in null and names[j] in null:
                    continue  # two null lines crossing make no joint (it would connect nothing)
                c, d = self._frame_geom[names[j]]
                hit = segment_intersection(a, b, c, d, self.tol)
                if hit is None:
                    continue
                t, s = hit
                xyz = a + t * (b - a)
                self._node_at(xyz)
                self._frame_cuts[names[i]].append(t)
                self._frame_cuts[names[j]].append(s)

    def _floor_frame_index(self):
        """Boxes (+tol) of the frames that floors cut, in model order: a vectorised
        broad phase for :meth:`_cut_frames_at_node` (built once per mesh build)."""
        key = (len(self.model.frames), len(self._frame_geom))
        idx = getattr(self, "_ffi", None)
        if idx is not None and idx[0] == key:
            return idx[1]
        names = [f.name for f in self.model.frames.values() if f.auto_mesh and f.mesh_at_floor and f.type != "Rigid"]
        if names:
            A = np.array([self._frame_geom[nm][0] for nm in names], float)
            B = np.array([self._frame_geom[nm][1] for nm in names], float)
            lo, hi = np.minimum(A, B) - self.tol, np.maximum(A, B) + self.tol
        else:
            lo = hi = np.zeros((0, 3))
        data = (names, lo, hi)
        self._ffi = (key, data)
        return data

    def _cut_frames_at_node(self, xyz: np.ndarray):
        """Register a mesh node that may lie on frame objects (floor meshing)."""
        names, lo, hi = self._floor_frame_index()
        if not names:
            return
        x = np.asarray(xyz, float)
        # broad phase: a point within tol of a segment lies in its box grown by tol
        near = np.nonzero(np.all((lo <= x) & (x <= hi), axis=1))[0]
        for k in near:
            nm = names[k]
            a, b = self._frame_geom[nm]
            t = point_on_segment(x, a, b, self.tol)
            if t is not None:
                self._frame_cuts[nm].append(t)
                fc = self.__dict__.setdefault("_floor_cut_count", {}).setdefault(nm, {})
                fc[round(t, 12)] = fc.get(round(t, 12), 0) + 1

    # -------------------------------------------------------- frame elements
    NONPRISMATIC_SUBDIVISIONS = 16  # elements per nonprismatic segment

    def _nonprismatic(self, f: Frame, L: float):
        """Segments [(r0, r1, props_start, props_end, n33, n22)] of a nonprismatic member
        in relative coordinates, or None for a prismatic one."""
        sec = self.model.frame_sections[f.section]
        if not sec.nonprismatic:
            return None
        rows = sec.nonprismatic
        lf = self.U.factor("length")
        abs_len = sum(r["length"] * lf for r in rows if r["length_type"] == "Absolute")
        var_sum = sum(r["length"] for r in rows if r["length_type"] != "Absolute")
        rest = L - abs_len
        lengths = []
        for r in rows:
            if r["length_type"] == "Absolute":
                lengths.append(r["length"] * lf)
            else:
                lengths.append(max(rest, 0.0) * r["length"] / var_sum if var_sum else 0.0)
        tot = sum(lengths)
        if tot <= 0:
            return None
        if abs(tot - L) > 1e-6 * L:
            self.am.warnings.append(f"Frame {f.name}: nonprismatic segment lengths scaled to the member length")
        order = {"linear": 1, "parabolic": 2, "cubic": 3}
        out, x = [], 0.0
        for r, ln in zip(rows, lengths):
            if ln <= 0:
                continue
            out.append((x / tot, (x + ln) / tot, self._section_props(f, r["start"]), self._section_props(f, r["end"]),
                        order.get(str(r["ei33"]).lower(), 1), order.get(str(r["ei22"]).lower(), 1)))
            x += ln
        return out

    @staticmethod
    def _nonprismatic_props(segs, t0: float, t1: float, L: float = 0.0) -> dict:
        """Element properties over [t0, t1] (inside one segment): stiffness terms as
        harmonic means (flexibilities in series), mass and weight as averages."""
        seg = next(sg for sg in segs if sg[0] - 1e-12 <= t0 and t1 <= sg[1] + 1e-12)
        r0, r1, p0, p1, n33, n22 = seg
        xg, wg = np.polynomial.legendre.leggauss(4)
        tm = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * xg
        xi = (tm - r0) / (r1 - r0)
        w = 0.5 * wg

        def at(key, n=1):
            a, b = p0[key], p1[key]
            if n == 1:
                return a + (b - a) * xi
            ra, rb = max(a, 0.0) ** (1.0 / n), max(b, 0.0) ** (1.0 / n)
            return (ra + (rb - ra) * xi) ** n

        def harmonic(v):
            return 0.0 if np.any(v <= 0) else 1.0 / float(np.sum(w / v))

        out = dict(p0)
        for key in ("A", "J"):
            out[key] = harmonic(at(key))
        out["I33"] = harmonic(at("I33", n33))
        out["I22"] = harmonic(at("I22", n22))
        for key in ("mass_per_length", "weight_per_length"):
            out[key] = float(np.sum(w * at(key)))
        # SAP2000 neglects shear deformation in nonprismatic members (verified with
        # SAP2000 example 1-006: the tip deflections match only without it)
        out["As2"] = out["As3"] = 0.0
        if L > 0:  # exact bending flexibility of the element (variable EI)
            xg8, wg8 = np.polynomial.legendre.leggauss(8)
            tq = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * xg8
            xi = (tq - r0) / (r1 - r0)
            x = (tq - t0) * L  # from the element end I
            wq = 0.5 * wg8 * (t1 - t0) * L
            flex = {}
            for key, n, name in (("I33", n33, "33"), ("I22", n22, "22")):
                Iq = at(key, n)
                if np.all(Iq > 0):
                    inv = wq / (p0["E"] * Iq)
                    flex[name] = (float(np.sum(inv)), float(np.sum(inv * x)), float(np.sum(inv * x * x)))
            out["bending_flex"] = flex
        return out

    def _section_props(self, f: Frame, section: Optional[str] = None) -> dict:
        m = self.model
        sec: FrameSection = m.frame_sections[section or f.section]
        mat = m.materials[sec.material]
        mods: FrameModifiers = sec.modifiers.combined(f.modifiers)
        U = self.U
        E = mat.E * U.factor("stress")
        G = mat.shear_modulus * U.factor("stress")
        area_f = U.factor("area")
        inertia_f = U.factor("inertia")
        weight_density = mat.weight_density * U.factor("weight_per_volume")
        if mat.mass_density is not None:
            mass_density = mat.mass_density * U.factor("mass_per_volume")
        else:
            mass_density = weight_density / self.am.gravity
        A = sec.A * area_f
        props = (sec.A, sec.As2, sec.As3, sec.J, sec.I22, sec.I33)
        from ..model.section_shapes import is_composite
        if sec.geometry and is_composite(sec.geometry):
            # composite (e.g. steel I in concrete): transformed stiffness with the
            # current materials, weight and mass from each material's own area
            from ..model.section_shapes import material_areas, section_properties
            p = section_properties(sec, True, m.materials)
            if p is not None and p.A > 0:
                props = (p.A, p.As2, p.As3, p.J, p.I22, p.I33)
                A = p.A * area_f
            wd = md = 0.0
            for k, a in material_areas(sec.geometry).items():
                mk = m.materials.get(k) if k else mat
                mk = mk or mat
                g = mk.weight_density * U.factor("weight_per_volume")
                wd += g * a * area_f
                md += (mk.mass_density * U.factor("mass_per_volume") if mk.mass_density is not None
                       else g / self.am.gravity) * a * area_f
            weight_density, mass_density = wd / A if A else 0.0, md / A if A else 0.0
        mass_pl = (mass_density * A + sec.mass_per_length * U.factor("mass_per_length")) * mods.mass \
            + f.mass_per_length * U.factor("mass_per_length")
        weight_pl = (weight_density * A + sec.weight_per_length * U.factor("force_per_length")) * mods.weight
        sA, sAs2, sAs3, sJ, sI22, sI33 = props
        return dict(
            E=E, G=G, A=A * mods.A, As2=sAs2 * area_f * mods.As2, As3=sAs3 * area_f * mods.As3,
            J=sJ * inertia_f * mods.J, I22=sI22 * inertia_f * mods.I22, I33=sI33 * inertia_f * mods.I33,
            mass_per_length=mass_pl, weight_per_length=weight_pl,
            t3=(sec.t3 or 0.0) * U.factor("length"), t2=(sec.t2 or 0.0) * U.factor("length"),
            alpha=mat.alpha,
        )

    def _build_frame_elements(self, f: Frame):
        am = self.am
        U = self.U
        a, b = self._frame_geom[f.name]
        L = float(np.linalg.norm(b - a))
        if L <= self.tol:
            raise ModelError("ZERO_LENGTH", f"Frame {f.name!r} has zero length", f"frames.{f.name}")
        if f.type == "Null":  # CSI null line: no stiffness, no mass; its loads go to the nodes along it
            self._null_line_loads(f, a, b, L)
            return
        if f.type == "Rigid":  # rigid link: its two joints move as one rigid body (no element)
            am.rigid_links.append(RigidLink(am.point_to_node[f.point_i], am.point_to_node[f.point_j], f.name))
            am.object_to_frames[f.name] = []
            if f.loads or self._panel_frame_loads.get(f.name):
                am.warnings.append(f"Rigid link {f.name}: its loads are ignored (a rigid link has no element)")
            return
        shift = self._axis_shift.get(f.name)
        props = self._section_props(f)
        # the local axes come from the joint-to-joint line (CSI), also for a beam meshed on
        # its true axis: unequal end offsets turn its axis in plan, not its section
        nominal = frame_axes(*self._frame_line[f.name], f.angle) if shift is not None else frame_axes(a, b, f.angle)

        # insertion point offsets (global vectors from joint to element centroid)
        c2, c3 = _CARDINAL.get(f.cardinal_point, (0.0, 0.0))
        cardinal = c2 * props["t3"] * nominal[1] + c3 * props["t2"] * nominal[2]
        off_i = np.array(f.joint_offset_i, float) * U.factor("length")
        off_j = np.array(f.joint_offset_j, float) * U.factor("length")
        if getattr(f, "joint_offset_local", False):  # components along the member local 1, 2, 3
            off_i, off_j = nominal.T @ off_i, nominal.T @ off_j
        off_i = off_i + cardinal
        off_j = off_j + cardinal
        if not getattr(f, "offsets_in_analysis", True):  # drawn off the joints only
            off_i = off_j = np.zeros(3)
        elif f.type == "Beam":
            vz = 0.5 * float(off_i[2] + off_j[2])
            if abs(vz) > 0.5 * props["t3"] + self.tol:
                am.warnings.append(
                    f"Beam {f.name}: {(abs(vz) - 0.5 * props['t3']) / U.factor('length'):.3f} {'above' if vz > 0 else 'below'} "
                    f"its joints' level, outside its depth - joined to the joints (and the slab there) by "
                    f"rigid arms that are not in the real building: use Edit > Edit Frames > Move Beams to "
                    f"Real Axis (the same as drawing the beam there)")
        if shift is not None:  # meshed on the true axis: only the rest (vertical) is an element arm
            off_i = off_i - shift[0]
            off_j = off_j - shift[1]
            off_i[np.abs(off_i) < 1e-12] = 0.0
            off_j[np.abs(off_j) < 1e-12] = 0.0
        axes = transformed_axes(nominal, (b + off_j) - (a + off_i))

        # end offsets and rigid zones along the (transformed) axis
        if self.model.frame_auto_end_offsets(f):
            end_off_i, end_off_j = self._auto_end_offsets(f, a, b)
        else:
            end_off_i = f.end_offset_i * U.factor("length")
            end_off_j = f.end_offset_j * U.factor("length")
        if end_off_i + end_off_j > 0.99 * L:
            scale = 0.99 * L / (end_off_i + end_off_j)
            end_off_i *= scale
            end_off_j *= scale
            am.warnings.append(f"Frame {f.name}: end offsets reduced to keep 1% clear length")
        rzf = self.model.frame_rigid_factor(f)
        rigid_i = rzf * end_off_i
        rigid_j = rzf * end_off_j
        # CSI: a moment/shear release at an end makes that end offset fully rigid for bending
        if any(f.releases_i[1:3]) or any(f.releases_i[4:6]):
            rigid_i = end_off_i
        if any(f.releases_j[1:3]) or any(f.releases_j[4:6]):
            rigid_j = end_off_j

        # split parameters (nonprismatic members: segment ends and subdivisions)
        np_segs = self._nonprismatic(f, L)
        extra = []
        if np_segs:
            n = self.NONPRISMATIC_SUBDIVISIONS
            for r0, r1, *_ in np_segs:
                extra += [r0 + (r1 - r0) * k / n for k in range(n + 1)]
        gap0 = self.tol / L  # cuts closer than the merge tolerance are one joint
        # a floor mesh node inside a rigid zone (the slab inside the column) does not cut
        # the member there: the piece would be rigid, and the slab there moves with the
        # column anyway; joints, crossing members and load points still cut it
        floor_count = self.__dict__.get("_floor_cut_count", {}).get(f.name, {})
        all_count: Dict[float, int] = {}
        for t in self._frame_cuts[f.name]:
            all_count[round(t, 12)] = all_count.get(round(t, 12), 0) + 1
        zi, zj = rigid_i / L, 1.0 - rigid_j / L
        cuts = []
        for t in sorted(set(round(t, 12) for t in list(self._frame_cuts[f.name]) + extra if 0.0 < t < 1.0)):
            if t <= gap0 or t >= 1.0 - gap0 or (cuts and t - cuts[-1] <= gap0):
                continue
            if (t <= zi + gap0 or t >= zj - gap0) and (rigid_i or rigid_j) and \
                    floor_count.get(t, 0) >= all_count.get(t, 0) and \
                    ((t <= zi + gap0 and rigid_i) or (t >= zj - gap0 and rigid_j)):
                continue  # only floor nodes / integration subdivisions there
            cuts.append(t)
        params = [0.0] + cuts + [1.0]
        given_cuts = set(round(t, 12) for t in self._frame_cuts[f.name])
        if extra:  # generated subdivisions must not create joints closer than the merge tolerance
            gap = 2.0 * self.tol / L
            keep = [0.0]
            given = set(round(t, 12) for t in self._frame_cuts[f.name])
            for t in params[1:]:
                if t - keep[-1] > gap:
                    keep.append(t)
                elif t in given or t == 1.0:  # a real cut (joint / load point / end) wins over a subdivision
                    keep[-1] = t if len(keep) > 1 else keep[-1]
            if keep[-1] != 1.0:
                keep[-1] = 1.0
            params = keep
        node_ids = []
        joint_cuts = self._joint_cuts.get(f.name, {})
        for k, t in enumerate(params):
            if shift is not None:
                # true axis: a node there, rigidly linked to the joint at the ends and to
                # the joints of members framing in along the joint-to-joint line
                nid = self._node_at(a + t * (b - a))
                node_ids.append(nid)
                joint = am.point_to_node[f.point_i] if k == 0 else am.point_to_node[f.point_j] \
                    if k == len(params) - 1 else joint_cuts.get(round(t, 12))
                if joint is not None and joint != nid and (f.name, joint) not in self._frames_into:
                    am.rigid_links.append(RigidLink(joint, nid, f.name))
                continue
            if k == 0:
                node_ids.append(am.point_to_node[f.point_i])
            elif k == len(params) - 1:
                node_ids.append(am.point_to_node[f.point_j])
            else:
                before = len(self.am.nodes)
                node_ids.append(self._node_at(a + t * (b - a)))
                if len(self.am.nodes) > before and np_segs and round(t, 12) not in given_cuts:
                    # a joint inside a nonprismatic member exists only for the integration:
                    # it does not join diaphragms (SAP2000 keeps such members as one element)
                    self._internal_nodes.add(node_ids[-1])

        if shift is not None:  # the axis off the column at an end: no rigid zone there (see _length_inside_column)
            for end_point, h in ((f.point_i, shift[0]), (f.point_j, shift[1])):
                hn = float(np.linalg.norm(h))
                if hn <= self.tol or (f.name, am.point_to_node[end_point]) in self._frames_into:
                    continue
                for other in self.model.frames.values():
                    if other.type != "Column" or end_point not in (other.point_i, other.point_j):
                        continue
                    sec = self.model.frame_sections.get(other.section)
                    if sec is None:
                        continue
                    try:
                        ax = frame_axes(self._xyz(self.model.points[other.point_i]),
                                        self._xyz(self.model.points[other.point_j]), other.angle)
                    except ValueError:
                        continue
                    hu = h / hn
                    half = 0.5 * (abs(float(hu @ ax[1])) * (sec.t3 or 0.0) + abs(float(hu @ ax[2])) * (sec.t2 or 0.0)) \
                        * U.factor("length")
                    own = self.model.frame_sections.get(f.section)
                    bears = 0.5 * float(getattr(own, "t2", 0.0) or 0.0) * U.factor("length")  # the beam still overlaps the column
                    if hn > half + bears + self.tol:
                        am.warnings.append(
                            f"Beam {f.name}: its axis is {hn / U.factor('length'):.3f} off joint {end_point}, outside "
                            f"column {other.name} ({half / U.factor('length'):.3f} to its face) - joined to the joint by a "
                            f"rigid link, no rigid end zone at that end - a rigid arm that is not in the real "
                            f"building: use Edit > Edit Frames > Move Beams to Real Axis (the same as drawing "
                            f"the beam there)")
        if shift is not None:  # a joint whose cut merged with a nearby one: the nearest axis node
            used = {round(t, 12) for t in params}
            for t, joint in joint_cuts.items():
                if t not in used and (f.name, joint) not in self._frames_into:
                    k = int(np.argmin([abs(t - p) for p in params]))
                    am.rigid_links.append(RigidLink(joint, node_ids[k], f.name))
        fix_i = np.concatenate([np.array(f.fixity_i_trans, float) * U.factor("translational_stiffness"),
                                np.array(f.fixity_i_rot, float) * U.factor("rotational_stiffness")])
        fix_j = np.concatenate([np.array(f.fixity_j_trans, float) * U.factor("translational_stiffness"),
                                np.array(f.fixity_j_rot, float) * U.factor("rotational_stiffness")])

        elem_ids = []
        n_seg = len(params) - 1
        for k in range(n_seg):
            t0, t1 = params[k], params[k + 1]
            first, last = k == 0, k == n_seg - 1
            if np_segs:
                props = self._nonprismatic_props(np_segs, t0, t1, L)
            # rigid zones by position: the part of this piece inside each zone (a piece
            # wholly inside one - cut there by a slab mesh node - stays 99% rigid; the
            # arms and the flexible length always add up to the piece's length, so a
            # rigid-body motion strains nothing)
            s0, s1 = t0 * L, t1 * L
            seg = s1 - s0
            ri = max(0.0, min(s1, rigid_i) - s0)
            rj = max(0.0, s1 - max(s0, L - rigid_j))
            if ri + rj > 0.99 * seg:
                sc = 0.99 * seg / (ri + rj)
                ri, rj = ri * sc, rj * sc
            span = None
            if f.type != "Column" and (end_off_i > 0 or end_off_j > 0):
                wa = min(max(end_off_i - s0, 0.0), seg)
                wb = min(max(L - end_off_j - s0, wa), seg)
                if wa > 1e-12 or wb < seg - 1e-12:
                    span = (wa, wb)
            el = FrameElement(
                id=len(am.frames), object=f.name, node_i=node_ids[k], node_j=node_ids[k + 1],
                E=props["E"], G=props["G"], A=props["A"], As2=props["As2"], As3=props["As3"], J=props["J"],
                I22=props["I22"], I33=props["I33"], mass_per_length=props["mass_per_length"],
                weight_per_length=props["weight_per_length"], axes=axes,
                releases_i=np.array(f.releases_i, bool) if first else np.zeros(6, bool),
                releases_j=np.array(f.releases_j, bool) if last else np.zeros(6, bool),
                fixity_i=fix_i if first else np.zeros(6), fixity_j=fix_j if last else np.zeros(6),
                offset_i=off_i.copy(), offset_j=off_j.copy(),
                rigid_i=ri, rigid_j=rj,
                end_offset_i=end_off_i if first else 0.0, end_offset_j=end_off_j if last else 0.0,
                weight_span=span,
                rel_start=t0, rel_end=t1,
                stations=max(2, f.min_stations),
                tension_limit=None if f.tension_limit is None else f.tension_limit * U.factor("force"),
                compression_limit=None if f.compression_limit is None else f.compression_limit * U.factor("force"),
                foundation=np.array(f.foundation, float) * U.factor("force_per_area"),
                alpha=props.get("alpha", 0.0),
                bending_flex=props.get("bending_flex"),
            )
            seg_len = (t1 - t0) * L
            if f.max_station_spacing:
                sp = f.max_station_spacing * U.factor("length")
                el.stations = max(el.stations, int(math.ceil(seg_len / sp)) + 1)
            self._assign_frame_loads(f, el, L, t0, t1, axes)
            am.frames.append(el)
            elem_ids.append(el.id)
        am.object_to_frames[f.name] = elem_ids

    def _null_line_loads(self, f: Frame, a: np.ndarray, b: np.ndarray, L: float):
        """Loads on a null line (SAFE / ETABS): no element; each piece of the line
        between two consecutive nodes on it carries its share of the load, lumped
        statically to those nodes (forces and moments).  The nodes are the line's
        joints and the mesh nodes the line passes (floor mesh, other members)."""
        am = self.am
        am.object_to_frames[f.name] = []
        loads = list(f.loads) + self._panel_frame_loads.get(f.name, [])
        if not loads:
            return
        params = [0.0, 1.0]
        for t in sorted(set(round(t, 12) for t in self._frame_cuts[f.name] if 0.0 < t < 1.0)):
            if self._find_node(a + t * (b - a)) is not None:  # only nodes that exist (no free joints)
                params.insert(-1, t)
        ids = [am.point_to_node[f.point_i]] + [self._find_node(a + t * (b - a)) for t in params[1:-1]] + \
              [am.point_to_node[f.point_j]]
        axes = frame_axes(a, b, f.angle)

        class _Seg:  # what _assign_frame_loads fills
            def __init__(self):
                self.loads, self.weight_per_length = [], 0.0

        for k in range(len(params) - 1):
            t0, t1 = params[k], params[k + 1]
            Ls = (t1 - t0) * L
            seg = _Seg()
            self._assign_frame_loads(f, seg, L, t0, t1, axes)
            for ld in seg.loads:
                if ld.kind not in ("distributed", "point") or ld.load_type not in ("Force", "Moment"):
                    continue  # temperature / self-weight multipliers: a null line has neither
                if ld.kind == "point":
                    x = ld.x_a
                    fi, fj = ld.value_a * (1 - x / Ls), ld.value_a * (x / Ls)
                else:
                    xa, xb = ld.x_a, ld.x_b
                    wa, wb = ld.value_a, ld.value_b
                    ln = xb - xa
                    total = ln * (wa + wb) / 2.0
                    slope = (wb - wa) / ln if ln > 0 else 0.0 * wa
                    first = wa * (xb ** 2 - xa ** 2) / 2.0 + slope * ((xb ** 3 - xa ** 3) / 3.0 - xa * (xb ** 2 - xa ** 2) / 2.0)
                    fj = first / Ls
                    fi = total - fj
                for nid, v in ((ids[k], fi), (ids[k + 1], fj)):
                    g = axes.T @ np.asarray(v, float)
                    vec = np.concatenate([g, np.zeros(3)]) if ld.load_type == "Force" else np.concatenate([np.zeros(3), g])
                    am.add_nodal_load(ld.pattern, nid, vec)

    # --------------------------------------------------------------- tendons
    def _tendon_samples(self, t) -> np.ndarray:
        lf = self.U.factor("length")
        pts = [(p.kind, np.asarray(p.xyz, float) * lf) for p in t.path]
        if len(pts) < 2:
            return np.zeros((0, 3))
        total = TD.path_length(pts)
        if t.model_as.lower().startswith("elem"):
            h = (t.max_discretization or 0.0) * lf or total / 10
        else:  # equivalent loads: a fine subdivision (smooth load distribution)
            h = total / 400
        return TD.sample_path(pts, max(h, 1e-6))

    def _tendon_frames(self, t):
        return [f for f in self.model.frames.values() if not t.frames or f.name in t.frames]

    def _bond_object(self, t, x):
        """(frame, relative position) of the bonded frame object nearest to x."""
        best = None
        for f in self._tendon_frames(t):
            a, b = self._frame_geom[f.name]
            d = b - a
            L2 = float(d @ d)
            if L2 <= 0:
                continue
            r = float((x - a) @ d) / L2
            if r < -1e-9 or r > 1 + 1e-9:
                continue
            dist = float(np.linalg.norm(a + min(max(r, 0.0), 1.0) * d - x))
            if best is None or dist < best[0]:
                best = (dist, f, min(max(r, 0.0), 1.0))
        return (best[1], best[2]) if best else (None, None)

    def _cut_frames_for_tendon(self, t):
        for x in self._tendon_points[t.name]:
            f, r = self._bond_object(t, x)
            if f is not None and 1e-9 < r < 1 - 1e-9:
                self._frame_cuts[f.name].append(r)

    def _bond_element(self, t, x):
        """(element, station from element end I, axis point) nearest to x."""
        am = self.am
        names = {f.name for f in self._tendon_frames(t)}
        best = None
        for el in am.frames:
            if el.object not in names:
                continue
            pa = am.nodes[el.node_i].xyz + el.offset_i
            pb = am.nodes[el.node_j].xyz + el.offset_j
            d = pb - pa
            L = float(np.linalg.norm(d))
            if L <= 0:
                continue
            e1 = d / L
            xs = float((x - pa) @ e1)
            if xs < -1e-6 * L or xs > L * (1 + 1e-6):
                continue
            xs = min(max(xs, 0.0), L)
            dist = float(np.linalg.norm(pa + xs * e1 - x))
            if best is None or dist < best[0] - 1e-12:
                best = (dist, el, xs, pa, e1, L)
        return best

    def _tendon_load_force(self, t, X, ld):
        U = self.U
        loss = ld.loss_stress * U.factor("stress") * t.area * U.factor("area")
        return TD.chord_forces(X, ld.force * U.factor("force"), ld.jack_from, ld.curvature,
                               ld.wobble / U.factor("length"), loss)

    def _apply_tendon(self, t):
        am = self.am
        U = self.U
        X = self._tendon_points[t.name]
        if len(X) < 2 or not t.loads:
            if len(X) < 2:
                am.warnings.append(f"Tendon {t.name}: path needs at least two points")
            return
        bonds = [self._bond_element(t, x) for x in X]
        if any(b is None for b in bonds):
            am.warnings.append(f"Tendon {t.name}: part of the path is not bonded to a frame and is ignored")
        if t.model_as.lower().startswith("elem"):
            self._tendon_elements(t, X, bonds)
            return
        for ld in t.loads:
            if ld.pattern not in self.model.load_patterns:
                continue
            F = TD.node_forces(X, self._tendon_load_force(t, X, ld))
            n = len(X)
            for k in range(n):
                b = bonds[k]
                if b is None:
                    continue
                _, el, xs, pa, e1, L = b
                r = X[k] - (pa + xs * e1)
                M = np.cross(r, F[k])
                fl, ml = el.axes @ F[k], el.axes @ M
                if k == 0 or k == n - 1:  # anchors: concentrated
                    el.loads.append(ElementLoad(ld.pattern, "point", "Force", xs, xs, fl, fl.copy()))
                    if np.any(ml):
                        el.loads.append(ElementLoad(ld.pattern, "point", "Moment", xs, xs, ml, ml.copy()))
                    continue
                lo = min(max(float((X[k - 1] - pa) @ e1), 0.0), L)
                hi = min(max(float((X[k + 1] - pa) @ e1), 0.0), L)
                lo, hi = sorted((lo, hi))
                a_, b_ = 0.5 * (lo + xs), 0.5 * (xs + hi)
                if b_ - a_ <= 1e-12:
                    el.loads.append(ElementLoad(ld.pattern, "point", "Force", xs, xs, fl, fl.copy()))
                    el.loads.append(ElementLoad(ld.pattern, "point", "Moment", xs, xs, ml, ml.copy()))
                    continue
                w = 1.0 / (b_ - a_)
                el.loads.append(ElementLoad(ld.pattern, "distributed", "Force", a_, b_, fl * w, fl * w))
                if np.any(ml):
                    el.loads.append(ElementLoad(ld.pattern, "distributed", "Moment", a_, b_, ml * w, ml * w))

    def _tendon_elements(self, t, X, bonds):
        """Bonded tendon members between the discretisation points (rigid links to the
        frame nodes) prestressed by an imposed shortening strain."""
        am = self.am
        U = self.U
        mat = self.model.materials.get(t.material) if t.material else None
        if mat is None:
            am.warnings.append(f"Tendon {t.name}: material missing, tendon ignored")
            return
        E = mat.E * U.factor("stress")
        G = mat.shear_modulus * U.factor("stress")
        A = t.area * U.factor("area")
        d = math.sqrt(4 * A / math.pi)
        I = math.pi * d ** 4 / 64
        wd = mat.weight_density * U.factor("weight_per_volume")
        md = mat.mass_density * U.factor("mass_per_volume") if mat.mass_density is not None else wd / am.gravity
        forces = {ld.pattern: self._tendon_load_force(t, X, ld) for ld in t.loads
                  if ld.pattern in self.model.load_patterns}
        for k in range(len(X) - 1):
            bi, bj = bonds[k], bonds[k + 1]
            if bi is None or bj is None:
                continue
            ni = self._node_at(bi[3] + bi[2] * bi[4])
            nj = self._node_at(bj[3] + bj[2] * bj[4])
            if ni == nj:
                continue
            axes = frame_axes(X[k], X[k + 1], 0.0)
            el = FrameElement(
                id=len(am.frames), object=f"tendon:{t.name}", node_i=ni, node_j=nj,
                E=E, G=G, A=A, As2=0.0, As3=0.0, J=2 * I, I22=I, I33=I,
                mass_per_length=md * A, weight_per_length=wd * A, axes=axes,
                offset_i=X[k] - am.nodes[ni].xyz, offset_j=X[k + 1] - am.nodes[nj].xyz)
            Lk = float(np.linalg.norm(X[k + 1] - X[k]))
            for pat, P in forces.items():
                eps = np.array([-P[k] / (E * A), 0.0, 0.0])  # N = -EA eps0 when restrained
                el.loads.append(ElementLoad(pat, "temperature", "Strain", 0.0, Lk, eps, eps.copy()))
            am.frames.append(el)

    def _auto_end_offsets(self, f: Frame, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """ETABS-style automatic end offsets from the sizes of the members
        connecting at each end (SI).

        * column: at its top joint the overlap is the depth of the deepest
          beam framing in (ETABS beams insert at their top, so the joint sits
          at the beam top); at the bottom joint 0.
        * beam/brace: half the dimension of the connecting column measured
          along the beam direction.
        """
        m = self.model
        U = self.U
        Lf = U.factor("length")
        direction = unit(b - a)
        out = []
        for end_point, end_xyz, inward in ((f.point_i, a, direction), (f.point_j, b, -direction)):
            offset = 0.0
            # the members at the end joint, and for a beam also the columns one rigid link
            # away (a beam moved to its real axis, joined to the column joint by a link)
            end_node = self.am.point_to_node.get(end_point)
            near = [(other, self._xyz(m.points[end_point])) for other in m.frames.values()
                    if other.name != f.name and end_point in (other.point_i, other.point_j)
                    and (other.name, end_node) not in self._frames_into]  # (framing into a side beam)
            if f.type in ("Beam", "Brace"):  # a column the end lands on partway up (moved up / down)
                near += [(c, end_xyz) for c in m.frames.values()
                         if c.type == "Column" and end_point not in (c.point_i, c.point_j)
                         and c.auto_mesh and c.mesh_at_intersections
                         and point_on_segment(end_xyz, *self._frame_geom[c.name], self.tol) is not None]
            # (a column likewise sees the beams moved to their real axis: its top end
            # offset is the depth of the deepest beam framing in, linked or not)
            want = "Column" if f.type in ("Beam", "Brace") else "Beam" if f.type == "Column" else None
            if want is not None:
                for lk in [g for g, _ in near if g.type == "Rigid"]:
                    jn = lk.point_j if lk.point_i == end_point else lk.point_i
                    near += [(other, self._xyz(m.points[jn])) for other in m.frames.values()
                             if other.type == want and other.name != f.name and jn in (other.point_i, other.point_j)]
            for other, joint_xyz in near:
                sec = m.frame_sections.get(other.section)
                if sec is None:
                    continue
                t3 = (sec.t3 or 0.0) * Lf
                t2 = (sec.t2 or 0.0) * Lf
                if f.type == "Column" and other.type == "Beam":
                    is_top = end_xyz[2] >= max(a[2], b[2]) - self.tol
                    if is_top:
                        offset = max(offset, t3)
                elif f.type in ("Beam", "Brace") and other.type == "Column":
                    oa = self._xyz(m.points[other.point_i])
                    ob = self._xyz(m.points[other.point_j])
                    try:
                        axes = frame_axes(oa, ob, other.angle)
                    except ValueError:
                        continue
                    offset = max(offset, _length_inside_column(end_xyz - joint_xyz, inward, axes, t3, t2, self.tol))
            out.append(offset)
        return out[0], out[1]

    def _assign_frame_loads(self, f: Frame, el: FrameElement, L: float, t0: float, t1: float, axes: np.ndarray):
        """Map object loads (model units, object coordinates) to this element."""
        U = self.U
        seg_len = (t1 - t0) * L
        for ld in list(f.loads) + self._panel_frame_loads.get(f.name, []):
            # loaded range in object relative coordinates
            if ld.absolute:
                ra = ld.rel_a * U.factor("length") / L
                rb = ld.rel_b * U.factor("length") / L
            else:
                ra, rb = ld.rel_a, ld.rel_b
            if ld.kind == "temperature":
                # temperatures (deg) or gradients (deg / length), linear from end I to end J of the object
                qf = 1.0 if ld.load_type == "Temperature" else 1.0 / U.factor("length")
                va = (ld.value_a + (ld.value_b - ld.value_a) * t0) * qf
                vb = (ld.value_a + (ld.value_b - ld.value_a) * t1) * qf
                el.loads.append(ElementLoad(ld.pattern, "temperature", ld.load_type, 0.0, seg_len,
                                            np.array([va, 0.0, 0.0]), np.array([vb, 0.0, 0.0])))
                continue
            if ld.kind == "gravity":
                # CSI "frame gravity load": multiplier x member self weight (per length) in a
                # global direction, over the whole member, independent of the pattern self weight
                vec = self._load_vector(_parse_direction(ld.direction)[0], ld.value_a * el.weight_per_length,
                                        axes, False)
                el.loads.append(ElementLoad(ld.pattern, "distributed", "Force", 0.0, seg_len, vec, vec.copy()))
                continue
            if ld.load_type == "Moment":
                qf = U.factor("moment_per_length") if ld.kind == "distributed" else U.factor("moment")
            else:
                qf = U.factor("force_per_length") if ld.kind == "distributed" else U.factor("force")
            direction, projected = _parse_direction(ld.direction)
            if ld.kind == "point":
                if not (t0 - 1e-12 <= ra <= t1 + 1e-12):
                    continue
                if ra >= t1 - 1e-12 and t1 < 1.0:  # belongs to the next segment (avoid double count)
                    continue
                x = (ra - t0) * L
                vec = self._load_vector(direction, ld.value_a * qf, axes, projected)
                el.loads.append(ElementLoad(ld.pattern, "point", ld.load_type, x, x, vec, vec.copy()))
            else:
                lo, hi = max(ra, t0), min(rb, t1)
                if hi - lo <= 1e-12:
                    continue
                # linear interpolation of the trapezoid at lo and hi
                def interp(r):
                    if rb == ra:
                        return ld.value_a
                    return ld.value_a + (ld.value_b - ld.value_a) * (r - ra) / (rb - ra)
                va = self._load_vector(direction, interp(lo) * qf, axes, projected)
                vb = self._load_vector(direction, interp(hi) * qf, axes, projected)
                el.loads.append(ElementLoad(ld.pattern, "distributed", ld.load_type,
                                            (lo - t0) * L, (hi - t0) * L, va, vb))

    @staticmethod
    def _load_vector(direction, magnitude: float, axes: np.ndarray, projected: bool) -> np.ndarray:
        """Return the load as a local (1,2,3) vector."""
        if isinstance(direction, int):  # local axis 1..3
            vec = np.zeros(3)
            vec[direction - 1] = magnitude
            return vec
        gvec = np.asarray(direction, float) * magnitude
        if projected:
            # intensity per unit projected length -> per unit element length
            sin_theta = math.sqrt(max(0.0, 1.0 - float(axes[0] @ unit(np.asarray(direction, float))) ** 2))
            gvec = gvec * sin_theta
        return axes @ gvec

    # ----------------------------------------------------------------- areas
    def _mesh_area(self, a: Area):
        m = self.model
        U = self.U
        pts = np.array([self._xyz(m.points[p]) for p in a.points])
        normal = polygon_normal(pts)
        horizontal = abs(normal[2]) > 1 - 1e-6
        if horizontal and normal[2] < 0:  # enforce +Z normal for floors
            pts = pts[::-1]
            a_points = list(reversed(a.points))
        else:
            a_points = list(a.points)
        sec: ShellSection = m.shell_sections[a.section]
        mat = m.materials[sec.material]
        mods: ShellModifiers = sec.modifiers.combined(a.modifiers)
        E = mat.E * U.factor("stress")
        G = mat.shear_modulus * U.factor("stress")
        t = sec.thickness * U.factor("length")
        tb = sec.bending_thickness_value * U.factor("length")
        wd = mat.weight_density * U.factor("weight_per_volume")
        md = mat.mass_density * U.factor("mass_per_volume") if mat.mass_density is not None else wd / self.am.gravity
        mass_pa = (md * t + sec.mass_per_area * U.factor("mass_per_area")) * mods.mass + a.mass_per_area * U.factor("mass_per_area")
        weight_pa = (wd * t + sec.weight_per_area * U.factor("force_per_area")) * mods.weight
        membrane_floor = a.name in getattr(self, "_membrane_floors", ())
        if membrane_floor:  # carried to the frames as loads (panel_loads.membrane_floors)
            weight_pa = 0.0
        own_loads = [] if membrane_floor else list(a.loads)
        modarr = np.array([mods.f11, mods.f22, mods.f12, mods.m11, mods.m22, mods.m12, mods.v13, mods.v23])
        modarr_lat = None
        if sec.lateral_modifiers is not None or a.lateral_modifiers is not None:
            lm = (sec.lateral_modifiers or sec.modifiers).combined(
                a.lateral_modifiers if a.lateral_modifiers is not None else a.modifiers)
            modarr_lat = np.array([lm.f11, lm.f22, lm.f12, lm.m11, lm.m22, lm.m12, lm.v13, lm.v23])
            if np.allclose(modarr_lat, modarr):
                modarr_lat = None
        # a membrane floor on auto mesh carries nothing out of its plane (its loads went to
        # the frames): in a rigid diaphragm it is no element at all (the diaphragm holds its
        # plane; its mass goes to the joints under it; the beams in it keep the rigid part
        # inside the columns its slab-column rigid zones gave them).  In a semi-rigid
        # diaphragm or none it is meshed as before (its in-plane stiffness: the chord /
        # collector forces of the beams need the mesh)
        membrane_auto = self._membrane_auto(a)
        dia = m.diaphragms.get(a.diaphragm) if a.diaphragm else None
        membrane_rigid = membrane_auto and a.diaphragm is not None and (dia.rigid if dia is not None else True)

        # mesh size
        if a.auto_mesh and a.max_mesh_size:
            size = a.max_mesh_size * U.factor("length")  # manual maximum mesh size
        elif a.type == "Wall":
            size = self.opts.wall_max_size or (m.options.wall_mesh_max_size and m.options.wall_mesh_max_size * U.factor("length"))
        else:
            size = self.opts.floor_max_size or (m.options.floor_mesh_max_size and m.options.floor_mesh_max_size * U.factor("length"))
        quads = self._quad_patches(pts)
        patch_div = None
        exact_pieces = [pts]  # polygons making up the meshed area (diaphragm inertia)
        holes, fr = [], None
        gsign, gkey = 1.0, None
        try:
            fr = pg.plane_frame(pts)
            planar = pg.off_plane(pts, fr) <= self.tol
            if planar:
                gfr, gsign, gkey = self._group_frame(a, pts)
                if gfr is not None and pg.axis_aligned(pg.to_2d(pts, gfr), self.tol):
                    fr = gfr  # shells of one plane share their mesh lines
                else:
                    gkey = None
        except ValueError:
            planar = False
        if self._openings and fr is not None:
            holes = self._openings.holes(pts, fr, self.tol, exclude=(a.name,))
        Q2 = pg.to_2d(pts, fr) if fr is not None else None
        if membrane_rigid and planar and fr is not None and self._membrane_without_elements(a, pts, fr, Q2, holes,
                                                                                            mass_pa):
            return
        # columns, loaded joints, beams, null lines, wall edges, neighbouring mesh nodes
        feats = self._area_features(a, pts, fr, Q2) if planar and a.auto_mesh else am_.Features()
        if (not size and self.opts.auto_size and planar and a.auto_mesh
                and not (a.mesh_divisions is not None and len(pts) == 4)):
            size = self._auto_area_size(a.name, gkey)
        if planar and a.auto_mesh and horizontal and m.options.slab_column_rigid_zone:
            zones, extra = self._zones(fr, Q2, [H for _, H in holes], size)
            feats.zones += zones
            if extra:
                feats.soft += extra
        if feats and not a.auto_mesh:
            self.am.warnings.append(f"Area {a.name}: auto mesh is off - joints / lines inside it are not connected")
        how = opn.split(Q2, holes, self.tol) if holes else None
        explicit = a.mesh_divisions if (a.mesh_divisions is not None and len(pts) == 4) else None
        rect = (planar and pg.axis_aligned(Q2, self.tol) and (how is None or how[0] == "grid")
                and a.auto_mesh and (feats or holes or not pg.is_convex(Q2)
                                     or (gkey is not None and self._group_has_features(gkey))))
        # any other shape that the patch split cannot mesh well: non-convex outlines,
        # openings, many-sided slabs with things on them -> triangulate and split to quads
        freeform = (planar and a.auto_mesh and explicit is None and not rect
                    and (not pg.is_convex(Q2) or bool(holes) or (bool(feats) and len(pts) > 4)
                         or bool(feats.zones)))
        self._area_order[a.name] = len(self._area_order)
        P3: List[np.ndarray] = []
        cells: List[Tuple[int, int, int, int]] = []
        local: Dict[Tuple[int, int, int], int] = {}

        def node_index(xyz) -> int:
            k = self._key(xyz)
            i = local.get(k)
            if i is None:
                i = local[k] = len(P3)
                P3.append(np.asarray(xyz, float))
            return i

        if freeform:
            hf = size if size and size > 0 else max(float(np.ptp(Q2[:, 0])), float(np.ptp(Q2[:, 1]))) / max(
                self.opts.default_divisions, 1)
            P2, C = am_.freeform_mesh(Q2, [H for _, H in holes], feats, hf, self.tol)
            X = pg.to_3d(P2, fr) if len(P2) else np.zeros((0, 3))
            ids = [node_index(x) for x in X]
            cells = [tuple(ids[i] for i in c) for c in C]
            if not len(C):
                self.am.warnings.append(f"Area {a.name}: covered by openings - not meshed")
        elif rect:  # grid through corners, openings and features (exact, conforming)
            hs = [H for _, H in holes]
            levels = None
            if gkey is not None and explicit is None:
                lv = self._group_levels(gkey, size)
                if lv is not None:
                    levels = (lv[0], sorted(v * gsign for v in lv[1]))
            grading = self._group_grading(gkey) if levels is not None else None
            for Z, _h in feats.zones:  # a turned column: its faces are reached by moving nodes
                if not pg.axis_aligned(Z, self.tol):
                    feats.lines += [((float(Z[i][0]), float(Z[i][1])),
                                     (float(Z[(i + 1) % len(Z)][0]), float(Z[(i + 1) % len(Z)][1])))
                                    for i in range(len(Z))]
            P2, C = am_.tensor_mesh(Q2, hs, feats, size, self.opts.default_divisions, self.tol,
                                    fixed_div=explicit, levels=levels, grading=grading)
            X = pg.to_3d(P2, fr) if len(P2) else np.zeros((0, 3))
            ids = [node_index(x) for x in X]
            cells = [tuple(ids[i] for i in c) for c in C]
            if not len(C):
                self.am.warnings.append(f"Area {a.name}: covered by openings - not meshed")
        else:
            if how is not None and how[0] == "cells":
                self.am.warnings.append(
                    f"Area {a.name}: opening(s) {', '.join(n for n, _ in holes)} not rectangular / not lined up "
                    f"with the shell - the hole follows the mesh (refine the mesh for a closer outline)")
            for qi, quad in enumerate(quads):
                if explicit is not None:
                    n1, n2 = explicit
                elif not a.auto_mesh:
                    n1 = n2 = 1
                else:
                    n1, n2 = self._divisions(quad, size)
                if len(quads) > 1 and a.auto_mesh and explicit is None:
                    # polygon split: use a common division count so shared edges match
                    n1 = n2 = max(n1, n2)
                grid = {}
                for i in range(n1):
                    for j in range(n2):
                        if how is not None:  # filter the cells inside openings
                            c = _bilinear(quad, (i + 0.5) / n1, (j + 0.5) / n2)
                            if opn.in_hole(pg.to_2d(c[None], fr)[0], holes):
                                continue
                        for (ii, jj) in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                            if (ii, jj) not in grid:
                                grid[(ii, jj)] = None
                # nodes in the original creation order (i, then j), then the cells
                for i in range(n1 + 1):
                    for j in range(n2 + 1):
                        if (i, j) in grid:
                            grid[(i, j)] = node_index(_bilinear(quad, i / n1, j / n2))
                for i in range(n1):
                    for j in range(n2):
                        q = (grid.get((i, j)), grid.get((i + 1, j)), grid.get((i + 1, j + 1)), grid.get((i, j + 1)))
                        if None in q:
                            continue
                        if how is not None:
                            c = _bilinear(quad, (i + 0.5) / n1, (j + 0.5) / n2)
                            if opn.in_hole(pg.to_2d(c[None], fr)[0], holes):
                                continue
                        cells.append(q)
        # features not on a mesh node / line yet: move the nearest nodes onto them
        if feats and cells:
            P2 = pg.to_2d(np.array(P3), fr)
            h = size if size and size > 0 else max(float(np.ptp(Q2[:, 0])), float(np.ptp(Q2[:, 1]))) / max(
                self.opts.default_divisions, 1)
            off = np.array([(x - fr[0]) @ fr[3] for x in P3])
            rep = am_.relocate(P2, np.array(cells, int), feats, h, self.tol, move_boundary=not (rect or freeform))
            if rep.moved:
                X = pg.to_3d(P2, fr) + np.outer(off, fr[3])
                P3 = [X[i] for i in range(len(X))]
            Lf = self.U.factor("length")
            if rep.missed:
                # reported after the auto edge constraints (_tie_to_shells), which tie most of
                # these joints into the element they lie in: only the ones still left are
                miss3 = pg.to_3d(np.array(rep.missed, float), fr) + np.outer(
                    np.full(len(rep.missed), float(np.mean(off)) if len(off) else 0.0), fr[3])
                self.__dict__.setdefault("_pending_missed", []).append((a.name, miss3))
            if rep.line_gap > 3.0 * h:
                self.am.warnings.append(
                    f"Area {a.name}: a skew beam / line load is tied into the slab only every "
                    f"{rep.line_gap / Lf:.2f} (mesh size {h / Lf:.2f}) - use a smaller mesh size")
        if rect or freeform or feats or how is not None:
            exact_pieces = [np.array([P3[i] for i in c]) for c in cells]

        axes = shell_axes(pts, a.angle)
        elem_ids = []
        trib_nodes: Dict[tuple, Dict[int, float]] = {}
        used = sorted({i for c in cells for i in c})
        nid_of: Dict[int, int] = {}
        for i in used:
            nid_of[i] = self._node_at(P3[i])
            self._cut_frames_at_node(P3[i])
        for c in cells:
            n_ids = tuple(int(nid_of[i]) for i in c)
            if len(set(n_ids)) < 4:
                continue  # degenerate (collapsed) cell
            el = ShellElement(
                id=len(self.am.shells), object=a.name, nodes=n_ids, E=E, nu=mat.nu, G=G,
                thickness=t, bending_thickness=tb, model_type=sec.model_type, mass_per_area=mass_pa,
                weight_per_area=weight_pa, modifiers=modarr, axes=axes, modifiers_lateral=modarr_lat,
            )
            for ld in own_loads + self._panel_area_loads.get(a.name, []):
                direction, projected = _parse_direction(ld.direction)
                value = ld.value * U.factor("force_per_area")
                if projected and not isinstance(direction, int):
                    value *= _projection_factor(normal, direction)  # per projected area -> per area
                region = getattr(ld, "region", None)
                if region is not None:  # a load panel over part of the area: this element's part
                    value *= _part_inside(np.array([P3[i] for i in c]),
                                          np.asarray(region, float) * U.factor("length"))  # model units -> SI
                    if value == 0.0:
                        continue
                if isinstance(direction, int):
                    if direction != 3:
                        raise ModelError("UNSUPPORTED_LOAD", f"Area {a.name}: only local 3 (pressure) supported for local loads")
                    el.loads.append(ShellLoad(ld.pattern, "pressure", np.array([value])))
                else:
                    el.loads.append(ShellLoad(ld.pattern, "force", np.asarray(direction, float) * value))
            self.am.shells.append(el)
            elem_ids.append(el.id)
        # area springs -> nodal springs by tributary area (normal direction)
        if a.spring:
            k_area = a.spring * U.factor("weight_per_volume")
            nrm = np.abs(normal) / (np.linalg.norm(normal) or 1.0)
            for c in cells:
                ids = tuple(int(nid_of[i]) for i in c)
                if len(set(ids)) < 4:
                    continue
                cell = np.array([P3[i] for i in c])
                for nid, ta in zip(ids, quad_tributary_areas(cell)):
                    self.am.nodes[nid].springs[:3] += k_area * float(ta) * nrm ** 2
        # register diaphragm membership of all mesh nodes
        if a.diaphragm and cells:
            z = float(np.mean(pts[:, 2]))
            key = (a.diaphragm, a.story, round(z / self.tol))
            self._diaphragm_nodes.setdefault(key, set()).update(int(nid_of[i]) for i in used)
            trib = trib_nodes.setdefault(key, {})
            for c in cells:
                ids = tuple(int(nid_of[i]) for i in c)
                if len(set(ids)) < 4:
                    continue
                for nid, ta in zip(ids, quad_tributary_areas(np.array([P3[i] for i in c]))):
                    trib[nid] = trib.get(nid, 0.0) + float(ta)
        if a.diaphragm:
            gravity_loads = {}
            for ld in own_loads + self._panel_area_loads.get(a.name, []):
                direction, projected = _parse_direction(ld.direction)
                if not isinstance(direction, int):
                    pf = _projection_factor(normal, direction) if projected else 1.0
                    pf *= float(getattr(ld, "fraction", 1.0))
                    gravity_loads[ld.pattern] = gravity_loads.get(ld.pattern, 0.0) - float(direction[2]) * ld.value * pf * U.factor("force_per_area")
            for key, trib in trib_nodes.items():
                self._diaphragm_areas.setdefault(key, []).append((exact_pieces, mass_pa, weight_pa, gravity_loads, trib))
        self.am.object_to_shells[a.name] = elem_ids

    def _membrane_without_elements(self, a: Area, pts: np.ndarray, fr, Q2: np.ndarray, holes, mass_pa: float) -> bool:
        """A membrane floor in a rigid diaphragm, as no element: the joints of the frames /
        walls on it join the diaphragm and take its mass (each part of the floor to the
        nearest of them; total mass exact, the rotational inertia corrected exactly by the
        diaphragm).  False (meshed after all) when no joint is under it."""
        m = self.model
        tol = self.tol
        normal = fr[3]
        o = fr[0]
        # joints of everything standing on / framing into the floor, in its plane
        cand: Dict[int, np.ndarray] = {}

        def take(xyz):
            xyz = np.asarray(xyz, float)
            if abs(float((xyz - o) @ normal)) > tol:
                return
            nid = self._node_at(xyz)
            cand[nid] = self.am.nodes[nid].xyz
        h = self.MEMBRANE_BEAM_DIVISION
        ends_at, flat_at = self._frames_at_levels()
        zk = round(float(np.mean(pts[:, 2])) / tol)
        for dz in (-1, 0, 1):
            for xyz in ends_at.get(zk + dz, ()):
                take(xyz)
        for f in (m.frames[n] for dz in (-1, 0, 1) for n in flat_at.get(zk + dz, ())):
            ga, gb = self._frame_geom[f.name]
            if abs(float((ga - o) @ normal)) > tol or abs(float((gb - o) @ normal)) > tol:
                continue
            for t in self._frame_cuts.get(f.name, ()):
                take(ga + t * (gb - ga))
            # a beam under the floor is divided as the floor's mesh would divide it: the
            # floor's mass (and the beam's own) spread along it, its nodes in the diaphragm
            mid2 = pg.to_2d(((ga + gb) / 2)[None], fr)[0]
            if not pg.inside(mid2, Q2, edge_tol=tol):
                continue
            L = float(np.linalg.norm(gb - ga))
            ei, ej = self._end_offsets_of(f)
            if m.options.slab_column_rigid_zone:
                # a node on the column face (the floor's slab-column rigid zone holds the
                # beam inside the column there, as with a meshed floor)
                cols = self._column_joints()
                for t, e, p in ((ei / L, ei, f.point_i), (1 - ej / L, ej, f.point_j)):
                    if p in cols and e > tol and 0.0 < t < 1.0:
                        xyz = ga + t * (gb - ga)
                        self._cut_frames_at_node(xyz)
                        take(xyz)
            n = int(math.ceil(L / h - 1e-9))
            if n < 2:
                continue
            for k in range(1, n):
                t = k / n
                if t * L <= ei + tol or (1 - t) * L <= ej + tol:
                    continue  # inside the joint zone: not a node of the member
                xyz = ga + t * (gb - ga)
                self._cut_frames_at_node(xyz)
                take(xyz)
        for w in m.areas.values():
            if w.name == a.name or w.is_opening or w.type == "Floor":
                continue
            for q in w.points:
                take(self._xyz(m.points[q]))
        if not cand:
            return False
        ids = list(cand)
        X2 = pg.to_2d(np.array([cand[n] for n in ids]), fr)
        ok = np.array([pg.inside(x, Q2, edge_tol=tol) and not any(pg.inside(x, H) for _, H in holes)
                       for x in X2], bool)
        ids = [n for n, k in zip(ids, ok) if k]
        X2 = X2[ok]
        if len(ids) < 2:
            return False
        # each part of the floor (less its openings) to the nearest joint
        from .panel_loads import _inside_many
        g = 96
        lo, hi = Q2.min(axis=0), Q2.max(axis=0)
        span = hi - lo
        ii, jj = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        C = lo + span * np.column_stack(((ii.ravel() + 0.5) / g, (jj.ravel() + 0.5) / g))
        keep = _inside_many(C, Q2)
        for _, H in holes:
            keep &= ~_inside_many(C, H, edge_tol=0.0)
        C = C[keep]
        area = abs(pg.signed_area(Q2)) - sum(abs(pg.signed_area(H)) for _, H in holes)
        if not len(C) or area <= 0:
            return False
        d2 = ((C[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2)
        near = np.argmin(d2, axis=1)
        w = np.bincount(near, minlength=len(ids)).astype(float)
        w *= area / w.sum()
        # first moment exact (the centre of mass of the floor where it is - the sampling puts
        # it within a cell): the least change of the shares, in proportion to them
        def first_moment(P):
            A2 = pg.signed_area(P)
            cx = cy = 0.0
            for k in range(len(P)):
                x0, y0 = P[k]
                x1, y1 = P[(k + 1) % len(P)]
                cr = x0 * y1 - x1 * y0
                cx += (x0 + x1) * cr
                cy += (y0 + y1) * cr
            sgn = 1.0 if A2 >= 0 else -1.0
            return np.array([cx, cy]) / 6.0 * sgn
        S = first_moment(Q2) - sum((first_moment(H) for _, H in holes), np.zeros(2))
        X = np.column_stack([np.ones(len(ids)), X2])  # rows: 1, u, v
        r = np.array([0.0, S[0] - float(w @ X2[:, 0]), S[1] - float(w @ X2[:, 1])])
        Wd = np.maximum(w, 1e-12 * area)
        G = (X.T * Wd) @ X
        try:
            lam = np.linalg.solve(G, r)
            dw = Wd * (X @ lam)
            if np.all(w + dw >= 0):
                w = w + dw
        except np.linalg.LinAlgError:
            pass  # joints on one line: the shares stay as sampled
        trib = {int(n): float(wi) for n, wi in zip(ids, w)}
        for n, wi in trib.items():
            self.am.area_node_mass[n] = self.am.area_node_mass.get(n, 0.0) + mass_pa * wi
        z = float(np.mean(pts[:, 2]))
        key = (a.diaphragm, a.story, round(z / self.tol))
        self._diaphragm_nodes.setdefault(key, set()).update(trib)
        hole_pieces = [pg.to_3d(H, fr) for _, H in holes]
        self._diaphragm_areas.setdefault(key, []).append(([pts], mass_pa, 0.0, {}, trib))
        if hole_pieces:
            self._diaphragm_areas[key].append((hole_pieces, -mass_pa, 0.0, {}, {}))
        self._area_order[a.name] = len(self._area_order)
        self.am.object_to_shells[a.name] = []
        self._unmeshed_membranes.add(a.name)
        self._membrane_nodes[a.name] = set(trib)
        return True

    # ------------------------------------------- auto edge constraints (ETABS)
    def _check_beams_beside_floors(self):
        """A beam whose real axis (joint offsets / insertion point in the analysis, or the
        beam moved there and joined by rigid links) lies outside the floor it belongs to
        - its joints are on that floor, its axis is not - shares no node with that floor:
        the slab edge along it has no beam under it.  Not an error the analysis can see,
        so it is said."""
        m, am = self.model, self.am
        floors = [a for a in m.areas.values() if a.type == "Floor" and all(p in m.points for p in a.points)]
        if not floors or not (am.shells or self._unmeshed_membranes):
            return
        Lf = self.U.factor("length")
        tol = max(self.tol, 1e-6)
        shell_nodes: Dict[str, set] = {}
        for sh in am.shells:
            shell_nodes.setdefault(sh.object, set()).update(sh.nodes)
        for name, nodes in self._membrane_nodes.items():  # membrane floors without elements
            shell_nodes.setdefault(name, set()).update(nodes)
        tied = {t.node for t in am.ties}
        polys = {}
        for a in floors:
            P = np.array([self._xyz(m.points[p]) for p in a.points], float)
            if np.ptp(P[:, 2]) > tol:
                continue  # an inclined floor: not checked
            polys[a.name] = P
        rigid_at: Dict[str, List[str]] = {}
        for g in m.frames.values():
            if g.type == "Rigid":
                rigid_at.setdefault(g.point_i, []).append(g.point_j)
                rigid_at.setdefault(g.point_j, []).append(g.point_i)
        for f in m.frames.values():
            if f.type != "Beam" or f.name not in am.object_to_frames:
                continue
            shifted = f.name in self._axis_shift
            # per end, the joints it stands for: its own (offset beam), or the joints its
            # rigid links go to (a moved beam; an end without a link: the end itself)
            rec = getattr(f, "real_axis", None)  # a beam moved to its real axis: the joints it came from
            came = [p for p in (rec or {}).get("joints", ()) if p in m.points]
            if not shifted and len(came) == 2 and [f.point_i, f.point_j] == list(rec.get("ends", ())):
                ends = [[p] for p in came]
            else:
                ends = [[p] for p in (f.point_i, f.point_j)] if shifted else \
                    [rigid_at.get(p, []) or [p] for p in (f.point_i, f.point_j)]
                if not shifted and not any(rigid_at.get(p) for p in (f.point_i, f.point_j)):
                    continue  # on its joints: the usual case, the floor mesh follows the beam
            nodes = {n for e in am.object_to_frames[f.name] for n in (am.frames[e].node_i, am.frames[e].node_j)}
            ga, gb = self._frame_geom[f.name]
            for name, P in polys.items():
                z = P[0, 2]
                if abs(ga[2] - z) > tol or abs(gb[2] - z) > tol:
                    continue
                Q = P[:, :2]
                if not all(any(_plan_distance(self._xyz(m.points[p])[:2], Q) <= tol for p in end) for end in ends):
                    continue  # the beam's joints (both ends) are not on this floor
                if nodes & (shell_nodes.get(name, set()) | tied):
                    continue  # it is connected
                mid = 0.5 * (ga + gb)
                d = _plan_distance(mid[:2], Q)
                if d <= tol:
                    continue
                am.warnings.append(
                    f"Beam {f.name}: its real axis is {d / Lf:.3f} outside floor {name} - the beam is not "
                    f"connected to the slab, and the slab edge along it has no beam under it. Draw the floor "
                    f"to the real slab edge (at least to the beam axis), or put the beam back on the floor edge")

    def _report_untied_joints(self):
        """Joints inside an area that got no mesh node and that no auto edge constraint
        ties to the mesh either: these are really not connected."""
        pend = self.__dict__.pop("_pending_missed", [])
        if not pend:
            return
        am = self.am
        X = np.array([n.xyz for n in am.nodes], float) if am.nodes else np.zeros((0, 3))
        in_mesh = {n for sh in am.shells for n in sh.nodes} | {t.node for t in am.ties}
        from scipy.spatial import cKDTree
        tree = cKDTree(X) if len(X) else None
        Lf = self.U.factor("length")
        for name, pts in pend:
            left = []
            for x in pts:
                near = tree.query_ball_point(x, 2.0 * self.tol) if tree is not None else []
                if not any(n in in_mesh for n in near):
                    left.append(x)
            if left:
                at = tuple(round(float(v) / Lf, 3) for v in left[0])
                am.warnings.append(f"Area {name}: {len(left)} joint(s) inside it could not be tied into its mesh "
                                   f"(e.g. at {at}) - use a smaller mesh size")

    def _tie_to_shells(self):
        """Ties for nodes lying in / on a shell element without being one of its
        nodes: a joint or member end that missed the mesh, and the nodes of a
        shell meshed later along an edge it shares with an earlier one.  The node
        follows the element's bilinear shape functions (ETABS auto edge
        constraint), so loads reach the shell and the meshes stay compatible."""
        am = self.am
        if not am.shells:
            return
        m = self.model
        X = np.array([n.xyz for n in am.nodes], float)
        order = self._area_order
        node_areas: Dict[int, Set[str]] = {}
        for sh in am.shells:
            for n in sh.nodes:
                node_areas.setdefault(n, set()).add(sh.object)
        cand = set(node_areas)
        for nd in am.nodes:  # joints that matter (members, loads, supports...)
            if nd.point is not None and nd.point in self._active_points:
                cand.add(nd.id)
        sep = {self.am.point_to_node.get(p) for p in getattr(self, "_separate_points", ())}
        cand -= sep
        cand = np.array(sorted(cand), int)
        if not len(cand):
            return
        # broad phase (C speed): candidate nodes within the sphere around each
        # shell's box (+tol) - a superset of the nodes the exact tests below accept
        from scipy.spatial import cKDTree
        tol = self.tol
        S = np.array([sh.nodes for sh in am.shells], int)
        PA = X[S]  # (n_shells, 4, 3)
        lo, hi = PA.min(axis=1) - tol, PA.max(axis=1) + tol
        ctr = 0.5 * (lo + hi)
        rad = 0.5 * np.linalg.norm(hi - lo, axis=1) + 2.0 * tol
        lists = cKDTree(X[cand]).query_ball_point(ctr, rad)
        # the object of a node that belongs to one mesh only (-1: none or several)
        objs = {}
        sh_obj = np.array([objs.setdefault(sh.object, len(objs)) for sh in am.shells], int)
        single = -np.ones(len(X), int)
        for n, own in node_areas.items():
            if len(own) == 1:
                single[n] = objs[next(iter(own))]
        tied: Set[int] = set()
        for si, sh in enumerate(am.shells):
            near = lists[si]
            if not near:
                continue
            near = cand[np.asarray(near, int)]
            near = near[single[near] != sh_obj[si]]  # nodes of this same mesh only: never tied to it
            if not len(near):
                continue
            P = PA[si]
            nrm = cross3(P[2] - P[0], P[3] - P[1])
            ln = np.linalg.norm(nrm)
            if ln <= 0:
                continue
            nrm /= ln
            my = order.get(sh.object, -1)
            for n in np.sort(near):
                n = int(n)
                if n in tied or n in sh.nodes:
                    continue
                own = node_areas.get(n, ())
                if sh.object in own:
                    continue  # a node of the same mesh
                if any(order.get(o, 1 << 30) < my for o in own):
                    continue  # the earlier mesh is the master side
                x = X[n]
                if abs((x - P[0]) @ nrm) > self.tol or np.min(np.linalg.norm(P - x, axis=1)) <= self.tol:
                    continue
                w = _bilinear_weights(P, x, nrm, self.tol)
                if w is None:
                    continue
                nd = am.nodes[n]
                if np.any(nd.restraints[:3]):
                    continue  # a support stays a support
                # a horizontal membrane holds nothing out of its plane: a node tied into it
                # follows it in its plane only (its own floor / beam carries it vertically)
                in_plane = (0, 1, 5) if sh.model_type == "Membrane" and abs(nrm[2]) > 1.0 - 1e-9 else None
                am.ties.append(Tie(n, [(int(sh.nodes[i]), float(w[i])) for i in range(4) if abs(w[i]) > 1e-12],
                                   sh.object, in_plane))
                tied.add(n)
        if tied:
            pts = [am.nodes[n].point for n in tied if am.nodes[n].point]
            note = f" (joints {', '.join(sorted(pts)[:6])}{'...' if len(pts) > 6 else ''})" if pts else ""
            am.info = getattr(am, "info", [])
            am.info.append(f"{len(tied)} node(s) tied to the shell they lie on (auto edge constraints){note}")

    # ------------------------------------------------- mesh features (ETABS / SAFE)
    def _prepare_features(self):
        """Segments that can be meshing lines of a floor / wall: frame objects (beams,
        braces, null lines carrying line loads) and the edges of area objects."""
        m = self.model
        segs, names = [], []
        for f in m.frames.values():
            if not f.auto_mesh or f.type == "Rigid":
                continue
            a, b = self._frame_geom[f.name]
            segs.append((a, b))
            names.append(("frame", f.name))
        for ar in m.areas.values():
            if ar.is_opening or any(p not in m.points for p in ar.points):
                continue
            P = [self._xyz(m.points[p]) for p in ar.points]
            for i in range(len(P)):
                segs.append((P[i], P[(i + 1) % len(P)]))
                names.append(("area", ar.name))
        self._plane_groups: Dict[tuple, List[str]] = {}
        self._plane_of: Dict[str, tuple] = {}
        for ar in m.areas.values():
            if ar.is_opening or any(p not in m.points for p in ar.points):
                continue
            P = np.array([self._xyz(m.points[p]) for p in ar.points])
            try:
                nrm = pg.newell_normal(P)
            except ValueError:
                continue
            k = int(np.argmax(np.abs(nrm)))
            if nrm[k] < 0:
                nrm = -nrm
            # parallel planes (every floor, every wall facing one way) share their mesh lines -
            # within one tower when the model has several (towers do not share their meshes)
            key = tuple(int(round(c * 1e5)) for c in nrm)
            if m.multi_tower:
                key = key + (m.tower_of_object("area", ar.name),)
            self._plane_groups.setdefault(key, []).append(ar.name)
            self._plane_of[ar.name] = key
        self._group_cache: Dict[tuple, tuple] = {}
        # joints that matter to a slab under them: members, other objects' corners,
        # loads, supports, springs, mass (a bare joint changes nothing - SAFE ignores it)
        active = set()
        for f in m.frames.values():
            active |= {f.point_i, f.point_j}
        for ar in m.areas.values():
            active |= set(ar.points)
        for sp in getattr(m, "line_springs", {}).values():
            active |= {sp.point_i, sp.point_j}
        for n, pt in m.points.items():
            if pt.loads or any(pt.restraints or ()) or any(pt.springs or ()) or any(pt.rotational_springs or ()) \
                    or any(pt.mass or ()) or pt.diaphragm or pt.ground_displacements:
                active.add(n)
        self._active_points = active
        self._frame_slab_crossings()
        self._feat_A = np.array([s[0] for s in segs], float).reshape(-1, 3)
        self._feat_B = np.array([s[1] for s in segs], float).reshape(-1, 3)
        self._feat_owner = names
        self._node_xyz_cache = np.zeros((0, 3))

    def _frame_slab_crossings(self):
        """A member passing through a floor / wall / ramp between its joints (an
        inclined beam or brace through a slab) is joined to it there: the member
        is cut at the crossing and the shell meshes at that point.  Not inside an
        opening, and not for null lines."""
        m = self.model
        shells = []
        for ar in m.areas.values():
            if ar.is_opening or any(p not in m.points for p in ar.points):
                continue
            P = np.array([self._xyz(m.points[p]) for p in ar.points])
            try:
                fr = pg.plane_frame(P)
            except ValueError:
                continue
            if pg.off_plane(P, fr) > self.tol:
                continue
            holes = [H for _, H in self._openings.holes(P, fr, self.tol, exclude=(ar.name,))] if self._openings else []
            shells.append((ar.name, fr, pg.to_2d(P, fr), holes, P.min(axis=0), P.max(axis=0)))
        if not shells:
            return
        self._crossings = []
        for f in m.frames.values():
            if f.type in ("Null", "Rigid") or not f.auto_mesh:
                continue
            a, b = self._frame_geom[f.name]
            d = b - a
            L = float(np.linalg.norm(d))
            if L <= self.tol:
                continue
            lo, hi = np.minimum(a, b) - self.tol, np.maximum(a, b) + self.tol
            for name, fr, Q, holes, plo, phi in shells:
                if np.any(hi < plo) or np.any(lo > phi):
                    continue
                n = fr[3]
                da, db = float((a - fr[0]) @ n), float((b - fr[0]) @ n)
                if abs(da - db) <= 1e-12 or da * db >= 0 or min(abs(da), abs(db)) <= self.tol:
                    continue  # parallel / not crossing / crossing at its own joint
                t = da / (da - db)
                x = a + t * d
                uv = pg.to_2d(x[None], fr)[0]
                if not pg.inside(uv, Q, edge_tol=self.tol) or any(pg.inside(uv, H) for H in holes):
                    continue
                self._node_at(x)
                self._frame_cuts[f.name].append(t)
                self._crossings.append((f.name, name))
        if self._crossings:
            self.am.info.append(f"{len(self._crossings)} member(s) joined to the floor / wall they pass through "
                                f"(e.g. {self._crossings[0][0]} through {self._crossings[0][1]})")

    def _node_coords(self) -> np.ndarray:
        n = len(self.am.nodes)
        c = self._node_xyz_cache
        if len(c) < n:
            c = np.vstack([c, np.array([nd.xyz for nd in self.am.nodes[len(c):]], float).reshape(-1, 3)])
            self._node_xyz_cache = c
        return c[:n]

    def _group_frame(self, a: Area, pts: np.ndarray):
        """The frame shared by all shells in the plane of ``a`` (global X along
        floors, horizontal along walls), turned to ``a``'s own normal; and the
        sign of v relative to the shared frame (+1 / -1)."""
        key = self._plane_of.get(a.name)
        if key is None:
            return None, 1.0, None
        g = self._group_cache.get(key)
        if g is None:
            first = self.model.areas[self._plane_groups[key][0]]
            P0 = np.array([self._xyz(self.model.points[p]) for p in first.points])
            nrm = np.array(key[:3], float) / 1e5
            nrm /= np.linalg.norm(nrm)
            if abs(nrm[2]) > 1 - 1e-6:
                hint = np.array([1.0, 0.0, 0.0])
            elif abs(nrm[2]) < 1e-6:
                hint = np.cross([0.0, 0.0, 1.0], nrm)
            else:
                hint = None
            g = [pg.plane_frame(P0, normal=nrm, e1_hint=hint), None]
            self._group_cache[key] = g
        _, ge1, ge2, gn = g[0]
        # this shell's exact normal (the key is rounded: an inclined ramp would be
        # meshed slightly off its plane), the shared in-plane directions
        n = pg.newell_normal(pts)
        e1 = ge1 - (ge1 @ n) * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        sign = 1.0 if e2 @ ge2 > 0 else -1.0
        o = n * float(n @ pts[0])  # u / v measured from the global origin
        return (o, e1, e2, n), sign, key

    def _group_levels(self, key, size):
        """Shared mesh lines of one plane: built the first time a shell of the plane
        is meshed, from every shell in it (their corners, openings and features)."""
        g = self._group_cache[key]
        if g[1] is None:
            _, e1, e2, n = g[0]
            polys, hs, feats = [], [], am_.Features()
            for name in self._plane_groups[key]:
                ar = self.model.areas[name]
                P = np.array([self._xyz(self.model.points[p]) for p in ar.points])
                fr = (n * float(n @ P[0]), e1, e2, n)
                Q = pg.to_2d(P, fr)
                if not pg.axis_aligned(Q, self.tol):
                    continue
                polys.append(Q)
                if self._openings:
                    hs += [H for _, H in self._openings.holes(P, fr, self.tol, exclude=(name,))]
                if ar.auto_mesh:
                    f = self._area_features(ar, P, fr, Q)
                    feats.points += f.points
                    feats.lines += f.lines
                    if abs(n[2]) > 1 - 1e-6 and self.model.options.slab_column_rigid_zone:
                        hq = [H for _, H in self._openings.holes(P, fr, self.tol, exclude=(name,))] \
                            if self._openings else []
                        z, soft = self._zones(fr, Q, hq, size)
                        feats.zones += z
                        feats.soft += soft
            g[1] = am_.feature_levels(polys, hs, feats, size, self.tol) if polys else None
            g.append(am_.zone_grading(feats.zones, self.tol))
        return g[1]

    def _group_grading(self, key):
        g = self._group_cache.get(key)
        return g[2] if g is not None and len(g) > 2 else None

    def _group_has_features(self, key) -> bool:
        """Whether another shell of a plane group has openings or features: then
        every shell of the plane is meshed on the shared lines (no hanging nodes
        where a plain piece meets a piece with features)."""
        cache = self.__dict__.setdefault("_group_feats", {})
        if key not in cache:
            names = self._plane_groups.get(key, [])
            found = False
            if len(names) > 1:
                fr0 = self._group_cache[key][0]
                for nm in names:
                    ar = self.model.areas[nm]
                    if not ar.auto_mesh:
                        continue
                    P = np.array([self._xyz(self.model.points[p]) for p in ar.points])
                    fr = (fr0[3] * float(fr0[3] @ P[0]), fr0[1], fr0[2], fr0[3])
                    Q = pg.to_2d(P, fr)
                    if (self._openings and self._openings.holes(P, fr, self.tol, exclude=(nm,))) \
                            or self._area_features(ar, P, fr, Q):
                        found = True
                        break
            cache[key] = found
        return cache[key]

    # ------------------------------------------------ slab-column rigid zones
    def _column_footprints(self):
        """Cross-sections of the columns (frames within 10 degrees of vertical with
        a rectangular / circular section): list of (end point, centre, corners
        (k, 3), smallest dimension), one per column end."""
        out = self.__dict__.get("_footprints")
        if out is not None:
            return out
        out = self._footprints = []
        m, L = self.model, self.U.factor("length")
        for f in m.frames.values():
            if f.type == "Null":
                continue
            sec = m.frame_sections.get(f.section)
            if sec is None or sec.nonprismatic or not sec.t3 or not sec.t2:
                continue
            shape = (sec.shape or "").lower()
            if not (shape.startswith("rect") or shape.startswith("circ")):
                continue
            a, b = self._xyz(m.points[f.point_i]), self._xyz(m.points[f.point_j])
            ax = b - a
            n = np.linalg.norm(ax)
            if n <= 0 or abs(ax[2]) / n < math.cos(math.radians(10)):
                continue
            from .geometry import frame_axes
            e = frame_axes(a, b, f.angle)
            d3, d2 = float(sec.t3) * L, float(sec.t2) * L
            if shape.startswith("circ"):
                ang = np.linspace(0, 2 * np.pi, 9)[:-1]
                rel = [0.5 * d3 * (np.cos(t) * e[1] + np.sin(t) * e[2]) for t in ang]
            else:
                rel = [0.5 * (s2 * d3 * e[1] + s3 * d2 * e[2]) for s2, s3 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for pt, c in ((f.point_i, a), (f.point_j, b)):
                corners = np.array([c + r for r in rel])
                corners[:, 2] = c[2]  # in the horizontal plane of the joint
                out.append((pt, c, corners, min(d2, d3)))
        return out

    ZONE_RATIO = 0.25  # element size at a column face: this x the column's smaller dimension (x the size factor)

    def _zones(self, fr, Q, holes, size):
        """Column cross-sections in a horizontal shell (2D, in the frame ``fr``),
        each cut to the shell: (zones [(polygon, element size at its edges)],
        corner points for the columns that cannot be zones).  The mesh follows the
        zones' outlines exactly, so the rigid zone of every column has its real
        size whatever the mesh size, and the elements grow from the column face
        outward.  A column too close to an edge, an opening or another column for
        clean elements falls back to corner nodes where they fit (soft)."""
        zones, soft = [], []
        if not size:
            return zones, soft
        o, nrm = fr[0], fr[3]
        f = max(self.opts.size_factor, 1e-9)
        taken = []
        fps = self._column_footprints()
        if not fps:
            return zones, soft
        C = self.__dict__.get("_footprint_centres")
        if C is None or len(C) != len(fps):
            C = self._footprint_centres = np.array([fp[1] for fp in fps], float)
        # the columns in this shell's plane (the same test as below, all at once)
        on = np.nonzero(np.abs((C - o) @ nrm) <= self.tol)[0]
        for k in on:
            _, c, corners, dmin = fps[k]
            if abs((c - o) @ nrm) > self.tol:
                continue
            uc = pg.to_2d(c[None], fr)[0]
            if not pg.inside(uc, Q, edge_tol=self.tol):
                continue
            uv = pg.to_2d(corners, fr)
            Z = pg.clip_convex(Q, uv)
            ok = len(Z) >= 3 and abs(pg.signed_area(Z)) > (0.05 * dmin) ** 2
            gap = 0.15 * dmin
            if ok:  # a corner just inside a curved / skew edge: put it on the edge (no slivers)
                Zs = []
                for p in Z:
                    d = self._dist_outline(p, Q)
                    if self.tol < d < gap:
                        p = self._nearest_on_outline(p, Q)
                    if not Zs or np.linalg.norm(p - Zs[-1]) > gap:
                        Zs.append(np.asarray(p, float))
                if len(Zs) > 1 and np.linalg.norm(Zs[0] - Zs[-1]) <= gap:
                    Zs.pop()
                Z = np.array(Zs).reshape(-1, 2)
                ok = len(Z) >= 3 and abs(pg.signed_area(Z)) > (0.05 * dmin) ** 2
            if ok:
                for H in holes:
                    if (any(pg.inside(p, H, edge_tol=gap) for p in Z) or any(pg.inside(p, Z, edge_tol=gap) for p in H)
                            or len(pg.clip_convex(H, Z)) >= 3):
                        ok = False
                        break
            if ok:
                for Zo in taken:
                    if len(pg.clip_convex(Zo, pg.offset_polygon(Z, gap))) >= 3:
                        ok = False
                        break
            if ok:
                taken.append(Z)
                zones.append((Z, min(size, self.ZONE_RATIO * dmin * f)))
            elif size <= 0.75 * dmin * f:
                soft += [(float(p[0]), float(p[1])) for p in uv if pg.inside(p, Q, edge_tol=self.tol)]
        return zones, soft

    def _on_outline(self, p, Q) -> bool:
        return self._dist_outline(p, Q) <= self.tol

    @staticmethod
    def _nearest_on_outline(p, Q):
        best, out = math.inf, np.asarray(p, float)
        for i in range(len(Q)):
            a, b = Q[i], Q[(i + 1) % len(Q)]
            d = b - a
            L2 = float(d @ d)
            t = 0.0 if L2 <= 0 else min(max(float((p - a) @ d) / L2, 0.0), 1.0)
            q = a + t * d
            dd = float(np.linalg.norm(q - p))
            if dd < best:
                best, out = dd, q
        return out

    @staticmethod
    def _dist_outline(p, Q) -> float:
        best = math.inf
        for i in range(len(Q)):
            a, b = Q[i], Q[(i + 1) % len(Q)]
            d = b - a
            L2 = float(d @ d)
            t = 0.0 if L2 <= 0 else min(max(float((p - a) @ d) / L2, 0.0), 1.0)
            best = min(best, float(np.linalg.norm(a + t * d - p)))
        return best

    def _footprint_points(self, fr, Q, size) -> list:
        """Corners of the column cross-sections in a horizontal shell (2D): mesh
        nodes there make the rigid zone the real column size.  Only once the mesh
        is fine enough (automatic size <= 0.75 x column size) - coarser meshes
        would be distorted for little gain."""
        out = []
        if not size:
            return out
        o, nrm = fr[0], fr[3]
        for _, c, corners, dmin in self._column_footprints():
            # judged on the unrefined automatic size, so that every mesh of a
            # refinement check (2h, 4h...) models the same rigid zone
            if size > 0.75 * dmin * max(self.opts.size_factor, 1.0) or abs((c - o) @ nrm) > self.tol:
                continue
            uv = pg.to_2d(corners, fr)
            if not pg.inside(pg.to_2d(c[None], fr)[0], Q, edge_tol=self.tol):
                continue
            out += [(float(p[0]), float(p[1])) for p in uv if pg.inside(p, Q, edge_tol=self.tol)]
        return out

    def _releases_at_column_faces(self):
        """End releases of a beam that meets a column inside a slab: the slab within
        the column's cross-section moves with the column (rigid zone), so the pieces
        of the beam inside it cannot rotate apart from the column - a pin on the
        piece at the joint would do nothing and the beam would act fixed-ended.  The
        release goes to the first piece outside the rigid zone, i.e. the hinge is at
        the column face and a pinned beam spans face to face as a simple span."""
        am = self.am
        if not am.bodies:
            return
        body_of = {}
        for k, b in enumerate(am.bodies):
            body_of[b.master] = k
            for n in b.nodes:
                body_of[n] = k
        moved = 0
        for name, eids in am.object_to_frames.items():
            if len(eids) < 2:
                continue
            els = sorted((am.frames[e] for e in eids), key=lambda e: e.rel_start)
            for end in ("i", "j"):
                seq = els if end == "i" else els[::-1]
                first = seq[0]
                rel = getattr(first, "releases_" + end)
                if not np.any(rel):
                    continue
                joint = first.node_i if end == "i" else first.node_j
                b = body_of.get(joint)
                if b is None:
                    continue
                # walk away from the joint while the far node is still in the column's zone
                k = 0
                while k < len(seq) - 1:
                    far = seq[k].node_j if end == "i" else seq[k].node_i
                    if body_of.get(far) != b:
                        break
                    k += 1
                if k == 0:
                    continue
                target = seq[k]
                setattr(target, "releases_" + end, np.array(rel, bool).copy())
                setattr(target, "fixity_" + end, np.array(getattr(first, "fixity_" + end), float).copy())
                setattr(first, "releases_" + end, np.zeros(6, bool))
                setattr(first, "fixity_" + end, np.zeros(6))
                moved += 1
        if moved:
            am.info.append(f"{moved} beam end release(s) placed at the column face (end of the slab-column rigid zone)")

    def _column_rigid_zones(self):
        """Tie the slab nodes inside each column's cross-section to the column
        joint (out-of-plane DOFs, rigid body)."""
        am = self.am
        if not am.shells and not self._membrane_nodes:
            return
        tied = {t.node for t in am.ties}
        floor_nodes = set()
        for sh in am.shells:
            ar = self.model.areas.get(sh.object)
            if ar is not None and ar.type == "Floor":
                floor_nodes.update(sh.nodes)
        for nodes in self._membrane_nodes.values():  # membrane floors without elements: their joints
            floor_nodes.update(nodes)
        if not floor_nodes:
            return
        ids = np.array(sorted(floor_nodes))
        X = np.array([am.nodes[i].xyz for i in ids])
        count = 0
        for pt, c, corners, _ in self._column_footprints():
            master = am.point_to_node.get(pt)
            if master is None or master not in floor_nodes or master in tied:
                continue
            near = np.abs(X[:, 2] - c[2]) <= self.tol
            lo, hi = corners.min(axis=0) - self.tol, corners.max(axis=0) + self.tol
            box = near & np.all((X[:, :2] >= lo[:2]) & (X[:, :2] <= hi[:2]), axis=1)
            if not box.any():
                continue
            poly = corners[:, :2]
            slaves = [int(ids[k]) for k in np.nonzero(box)[0]
                      if int(ids[k]) != master and int(ids[k]) not in tied
                      and pg.inside(X[k, :2], poly, edge_tol=self.tol)]
            if slaves:
                am.bodies.append(Body(master, slaves, pt))
                count += 1
        if count:
            am.info.append(f"{count} column-slab rigid zone(s): the slab inside each column's cross-section "
                           f"moves with the column")

    def _support_features(self, P: np.ndarray, fr, Q) -> "am_.Features":
        """Where a shell is held: columns / braces standing on or hanging from it
        and supported joints in it (the spans that set the automatic mesh size).
        Loaded joints and the nodes of neighbouring meshes are not spans."""
        X = self.__dict__.get("_support_xyz")
        if X is None:
            m = self.model
            pts = set()
            for f in m.frames.values():
                if f.type == "Null":
                    continue
                pts.update((f.point_i, f.point_j))
            for name, p in m.points.items():
                if any(p.restraints) or any(p.springs) or any(p.rotational_springs):
                    pts.add(name)
            X = self._support_xyz = (np.array([self._xyz(m.points[p]) for p in pts if p in m.points])
                                     if pts else np.zeros((0, 3)))
            self._support_dir = {}
        out = am_.Features()
        if not len(X):
            return out
        o, n = fr[0], fr[3]
        near = np.abs((X - o) @ n) <= self.tol
        if not near.any():
            return out
        uv = pg.to_2d(X[near], fr)
        for p in uv:
            if pg.inside(p, Q, edge_tol=self.tol):
                out.points.append((float(p[0]), float(p[1])))
        # frames lying in the plane (beams) share joints with the slab too: keep
        # only joints that are not just beam-to-beam joints is not worth the cost;
        # beams are lines and the refinement check covers them
        return out

    def _fine_factor(self, ar) -> float:
        """Size factor of an area assigned the fine mesh (1 when none was found)."""
        if not (self.opts.area_fine and getattr(ar, "fine_mesh", False)):
            return 1.0
        f = getattr(self.model.options, "fine_mesh_factor", None)
        return float(f) if f and 0 < f <= 1 else 1.0

    def _auto_area_size(self, name: str, gkey=None) -> float:
        """Automatic mesh size of an area (SI); the smallest of its plane group so
        that shells of one plane share their mesh lines."""
        cache = self.__dict__.setdefault("_auto_sizes", {})
        names = list(getattr(self, "_plane_groups", {}).get(gkey, ())) if gkey is not None else []
        if name not in names:
            names.append(name)
        out = []
        for nm in names:
            if nm not in cache:
                ar = self.model.areas[nm]
                h = None
                try:
                    P = np.array([self._xyz(self.model.points[p]) for p in ar.points])
                    fr = pg.plane_frame(P)
                    Q = pg.to_2d(P, fr)
                    h = am_.auto_size(Q, self._support_features(P, fr, Q), self.tol,
                                      factor=self.opts.size_factor * self._fine_factor(ar))
                    # fine enough for the columns' rigid zones to have their real size
                    if (ar.type == "Floor" and abs(fr[3][2]) > 1 - 1e-6
                            and self.model.options.slab_column_rigid_zone):
                        dm = [d for _, c, _, d in self._column_footprints()
                              if abs((c - fr[0]) @ fr[3]) <= self.tol
                              and pg.inside(pg.to_2d(c[None], fr)[0], Q, edge_tol=self.tol)]
                        # the columns' own cross-sections are meshed finely (and graded
                        # out to h) by the zones: no need to refine the whole slab for them
                        del dm
                except ValueError:
                    pass
                cache[nm] = h
                if h is not None:
                    self.am.auto_sizes[nm] = h
            if cache[nm] is not None:
                out.append(cache[nm])
        return min(out) if out else None

    def _area_features(self, a: Area, pts: np.ndarray, fr, Q2) -> "am_.Features":
        """Joints and lines in the plane of this shell and inside it: nodes that
        already exist there (columns standing on it, loaded joints, supports,
        frame crossings, the mesh of neighbouring shells along shared edges) and
        frame / area-edge segments (beams, null lines with line loads, walls)."""
        feats = am_.Features()
        tol = self.tol
        o, n = fr[0], fr[3]
        lo, hi = pts.min(axis=0) - tol, pts.max(axis=0) + tol
        own = {self._key(p) for p in pts}
        X = self._node_coords()
        if len(X):
            sel = np.all((X >= lo) & (X <= hi), axis=1) & (np.abs((X - o) @ n) <= tol)
            sep = {self.am.point_to_node.get(p) for p in getattr(self, "_separate_points", ())}
            nodes = self.am.nodes
            for i in np.nonzero(sel)[0]:
                if int(i) in sep or self._key(X[i]) in own:
                    continue
                pt = nodes[int(i)].point
                if pt is not None and pt not in self._active_points:
                    continue  # a bare joint
                uv = pg.to_2d(X[i][None], fr)[0]
                if pg.inside(uv, Q2, edge_tol=tol):
                    feats.points.append((float(uv[0]), float(uv[1])))
        A, B = self._feat_A, self._feat_B
        if len(A):
            inplane = (np.abs((A - o) @ n) <= tol) & (np.abs((B - o) @ n) <= tol)
            box = np.all((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo), axis=1)
            for k in np.nonzero(inplane & box)[0]:
                kind, name = self._feat_owner[k]
                if kind == "area" and name == a.name:
                    continue
                for seg in am_.clip_segment(pg.to_2d(A[k][None], fr)[0], pg.to_2d(B[k][None], fr)[0], Q2, tol):
                    feats.lines.append(seg)
        return feats

    def _divisions(self, quad: np.ndarray, size: Optional[float]) -> Tuple[int, int]:
        l1 = 0.5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
        l2 = 0.5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
        if size is None or size <= 0:
            n = self.opts.default_divisions
            return n, n
        return max(1, int(math.ceil(l1 / size - 1e-9))), max(1, int(math.ceil(l2 / size - 1e-9)))

    @staticmethod
    def _grid_patches(pts: np.ndarray, tol: float):
        """A non-convex polygon whose edges all run in two perpendicular directions
        (L, T, U shapes, merged floors) is split into the rectangles of the grid
        through its corners; returns (quads, [(length along u, along v)]) with
        each rectangle's divisions to be taken per grid interval, or None when
        the polygon is convex or not rectilinear (then :meth:`_quad_patches`)."""
        from . import polygon2d as pg
        if len(pts) <= 4:
            return None
        try:
            fr = pg.plane_frame(pts)
        except ValueError:
            return None
        Q = pg.to_2d(pts, fr)
        if pg.is_convex(Q) or not pg.axis_aligned(Q, tol):
            return None
        U, V, cells = pg.grid_cells(Q, (), tol)
        if not cells:
            return None
        quads = [pg.to_3d(pg.cell_corners(U, V, k, l), fr) for k, l in cells]
        spans = [(U[k + 1] - U[k], V[l + 1] - V[l]) for k, l in cells]
        return quads, spans

    @staticmethod
    def _quad_patches(pts: np.ndarray) -> List[np.ndarray]:
        """Split a planar polygon into quads.

        4-node polygons are used directly.  Other polygons are split into one
        quad per corner using edge midpoints and the centroid (a Catmull-Clark
        style subdivision), which gives an all-quad mesh for any star-shaped
        polygon.
        """
        n = len(pts)
        if n == 4:
            return [pts.copy()]
        c = polygon_centroid(pts)
        mids = [(pts[i] + pts[(i + 1) % n]) / 2.0 for i in range(n)]
        quads = []
        for i in range(n):
            quads.append(np.array([pts[i], mids[i], c, mids[i - 1]]))
        return quads

    # ------------------------------------------------------------ constraints
    def _drop_loose_area_nodes(self):
        """Nodes a membrane floor without elements gave its mass to that no element uses
        after all (a cut the member did not take): their mass to the nearest used node of
        the same diaphragm, and out of the diaphragm."""
        am = self.am
        if not self._unmeshed_membranes:
            return
        used = set()
        for fe in am.frames:
            used.update((fe.node_i, fe.node_j))
        for sh in am.shells:
            used.update(sh.nodes)
        for rl in am.rigid_links:
            used.update((rl.node_i, rl.node_j))
        for lk in getattr(am, "links", ()) or ():
            used.update(getattr(lk, "nodes", ()) or ())
        # the corners of such a floor that nothing else uses: held quietly
        for name in self._unmeshed_membranes:
            for q in self.model.areas[name].points:
                nid = am.point_to_node.get(q)
                p = self.model.points[q]
                if nid is not None and nid not in used and not (p.loads or any(p.restraints) or any(p.springs)
                                                                or any(p.mass)):
                    am.quiet_nodes.append(nid)
        used.update(n.id for n in am.nodes if n.point is not None)
        for key, nodes in self._diaphragm_nodes.items():
            loose = [n for n in nodes if n not in used]
            if not loose:
                continue
            keep = [n for n in nodes if n in used]
            for n in loose:
                nodes.discard(n)
                mass = am.area_node_mass.pop(n, 0.0)
                if mass and keep:
                    d = [float(np.linalg.norm(am.nodes[k].xyz - am.nodes[n].xyz)) for k in keep]
                    tgt = keep[int(np.argmin(d))]
                    am.area_node_mass[tgt] = am.area_node_mass.get(tgt, 0.0) + mass
                    for entry in self._diaphragm_areas.get(key, []):
                        trib = entry[4]
                        if n in trib:
                            trib[tgt] = trib.get(tgt, 0.0) + trib.pop(n)

    def _build_constraints(self):
        am = self.am
        self._drop_loose_area_nodes()
        for (name, story, _zkey), nodes in self._diaphragm_nodes.items():
            node_list = sorted(nodes)
            if len(node_list) < 2:
                continue
            diaph = self.model.diaphragms.get(name)
            rigid = diaph.rigid if diaph else True
            if not rigid:
                am.warnings.append(f"Diaphragm {name!r} is semi-rigid; treated as unconstrained (meshed slab)")
                continue
            coords = np.array([am.nodes[n].xyz for n in node_list])
            center = coords.mean(axis=0)
            mid = len(am.nodes)
            master = Node(id=mid, xyz=center, story=story, is_master=True, diaphragm=name,
                          tower=am.story_towers.get(story) or am.nodes[node_list[0]].tower)
            am.nodes.append(master)
            for n in node_list:
                am.nodes[n].diaphragm = name
            con = Constraint(name=name, story=story, nodes=node_list, master=mid, normal=2, rigid=True)
            # analytical rotational inertia of area objects vs. lumped mesh masses
            for pieces, mass_pa, weight_pa, gravity_loads, trib in self._diaphragm_areas.get((name, story, _zkey), []):
                j_exact = sum(polygon_polar_inertia(p, center) for p in pieces)
                j_disc = sum(ta * float(np.sum((am.nodes[n].xyz[:2] - center[:2]) ** 2)) for n, ta in trib.items())
                diff = j_exact - j_disc
                con.inertia_correction_self += mass_pa * diff
                for pat, p in gravity_loads.items():
                    con.inertia_correction_loads[pat] = con.inertia_correction_loads.get(pat, 0.0) + p / am.gravity * diff
                # self weight pattern mass (pattern multiplier applied in the assembler)
                if weight_pa:
                    con.inertia_correction_loads["__self_weight__"] = con.inertia_correction_loads.get("__self_weight__", 0.0) + weight_pa / am.gravity * diff
            am.constraints.append(con)


def _plan_distance(p: np.ndarray, Q: np.ndarray) -> float:
    """Plan distance of point ``p`` from polygon ``Q`` (0 inside or on it)."""
    if pg.inside(np.asarray(p, float), Q):
        return 0.0
    best = np.inf
    n = len(Q)
    for i in range(n):
        a, b = Q[i], Q[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else min(1.0, max(0.0, float((p - a) @ ab) / L2))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _length_inside_column(h: np.ndarray, d: np.ndarray, axes: np.ndarray, t3: float, t2: float,
                          tol: float) -> float:
    """Length of a beam axis inside a column (the automatic end offset), from the beam end
    that lies ``h`` from the column's axis, going along ``d`` into the span.  At the
    column's axis (h = 0): the ETABS rule, half the column dimension along the beam;
    off it (a beam on its real axis, an offset beam): the exact length to where the axis
    leaves the column's rectangle, 0 when the end is outside the column."""
    u, v = float(h @ axes[1]), float(h @ axes[2])
    du, dv = float(d @ axes[1]), float(d @ axes[2])
    if abs(u) <= tol and abs(v) <= tol:
        return 0.5 * (abs(du) * t3 + abs(dv) * t2)
    h3, h2 = 0.5 * t3, 0.5 * t2
    if abs(u) > h3 + tol or abs(v) > h2 + tol:
        return 0.0
    s = np.inf
    for c, dc, half in ((u, du, h3), (v, dv, h2)):
        if abs(dc) > 1e-12:
            s = min(s, ((half if dc > 0 else -half) - c) / dc)
    return float(max(0.0, s)) if np.isfinite(s) else 0.0


def _part_inside(cell3: np.ndarray, region3: np.ndarray) -> float:
    """Fraction of the (planar) element polygon inside the region polygon (same plane)."""
    from .panel_loads import overlap_area_2d, polygon_frame
    o, e1, e2, _ = polygon_frame(np.asarray(region3, float))
    to = lambda Q: np.array([[(q - o) @ e1, (q - o) @ e2] for q in np.asarray(Q, float)])
    C = to(cell3)
    full = abs(pg.signed_area(C))
    if full <= 0:
        return 0.0
    return min(1.0, overlap_area_2d(C, to(region3)) / full)


def _projection_factor(normal, direction) -> float:
    """|n . d|: an intensity per unit area projected on the plane normal to the load
    direction (CSI "Gravity Projected") -> per unit area of the inclined surface."""
    n = np.asarray(normal, float)
    ln = float(np.linalg.norm(n))
    if ln <= 0:
        return 1.0
    return abs(float(n @ np.asarray(direction, float))) / ln


def _parse_direction(text: str):
    """Return (direction, projected) where direction is a global unit vector
    or an int 1..3 for element local axes."""
    t = str(text).strip()
    projected = False
    low = t.lower()
    if low.endswith("projected"):
        projected = True
        low = low[: -len("projected")].strip(" _-")
    if low.startswith("vec:"):  # arbitrary global direction (panel loads)
        v = np.array([float(x) for x in low[4:].split(",")], float)
        return v / (np.linalg.norm(v) or 1.0), projected
    if low in ("gravity", "grav", "-z"):
        return np.array([0.0, 0.0, -1.0]), projected
    if low in ("x", "+x", "global x"):
        return np.array([1.0, 0.0, 0.0]), projected
    if low in ("y", "+y", "global y"):
        return np.array([0.0, 1.0, 0.0]), projected
    if low in ("z", "+z", "global z"):
        return np.array([0.0, 0.0, 1.0]), projected
    if low in ("-x",):
        return np.array([-1.0, 0.0, 0.0]), projected
    if low in ("-y",):
        return np.array([0.0, -1.0, 0.0]), projected
    if low in ("1", "local 1", "local1"):
        return 1, projected
    if low in ("2", "local 2", "local2"):
        return 2, projected
    if low in ("3", "local 3", "local3"):
        return 3, projected
    raise ModelError("INVALID_DIRECTION", f"Unknown load direction {text!r}")


def _bilinear(quad: np.ndarray, xi: float, eta: float) -> np.ndarray:
    p0, p1, p2, p3 = quad
    return ((1 - xi) * (1 - eta) * p0 + xi * (1 - eta) * p1 + xi * eta * p2 + (1 - xi) * eta * p3)


def build_analysis_model(model: Model, options: Optional[MeshOptions] = None, progress=None) -> AnalysisModel:
    """Mesh the model; ``progress(fraction 0..1, message)`` reports the steps."""
    ms = Mesher(model, options)
    ms.progress = progress
    return ms.build()
