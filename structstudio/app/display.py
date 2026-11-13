"""Per-window display options (ETABS "Set Display Options", Ctrl+W).

Each model window keeps its own :class:`DisplayOptions` (saved with the window
layout); the dialog can apply a choice to one window or to all of them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Tuple

GRID_3D = ("all", "base", "top")
VERSION = 6  # v6: vertical grid lines and the grid at every story in 3D on again; v2: line view by default, vertical grid lines in 3D; v3: 3D grid at every story; v4: joints shown;
# v5: spring supports shown
GRID_3D_LABELS = {"base": "At the base level", "all": "At every story / level", "top": "At the top level"}


@dataclass
class DisplayOptions:
    # ---- object visibility
    joints: bool = True  # small squares (ETABS)
    beams: bool = True
    columns: bool = True
    braces: bool = True
    null_lines: bool = True
    rigid_links: bool = True
    floors: bool = True
    walls: bool = True
    ramps: bool = True
    openings: bool = True
    load_panels: bool = True  # drawn load panels (Load > Load Panel); always shown while a panel command runs
    # ---- assignments
    supports: bool = True
    springs: bool = True  # spring supports (elastic): drawn like the rigid supports
    releases: bool = True  # pins / releases: rings near the released ends
    local_axes: bool = False
    # ---- labels
    joint_labels: bool = False
    frame_labels: bool = False
    area_labels: bool = False
    section_names: bool = False
    # ---- other
    extrude: bool = False
    fill_areas: bool = True
    edges: bool = True
    grid_lines: bool = True
    grid_bubbles: bool = True
    grid_3d: str = "all"  # 3D windows: grid at every level (base drawn stronger) / the base / the top
    grid_vertical: bool = True  # 3D windows: vertical lines at the grid intersections (column lines)
    story_plane: bool = True  # 3D windows: shade the plane of the story shown in the plan window
    # CYPE 3D reference lines: through every joint, parallel to the global axes, in their own colour
    reference_lines: bool = False
    reference_x: bool = True
    reference_y: bool = True
    reference_z: bool = True
    # the mesh the analysis used (floors / walls); nothing before an analysis
    analysis_mesh: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["version"] = VERSION
        return d

    @classmethod
    def from_dict(cls, d) -> "DisplayOptions":
        o = cls()
        if isinstance(d, dict):
            ver = int(d.get("version", 1) or 1)
            for f in fields(cls):
                if ver < 2 and f.name == "extrude":
                    continue  # v1 windows were saved with extrusion on by default
                if ver < 3 and f.name == "grid_3d":
                    continue
                if ver < 6 and f.name in ("grid_3d", "grid_vertical"):
                    continue  # (the story grids / column lines were often switched off and not found again)
                if ver < 4 and f.name in ("joints", "releases"):
                    continue  # older windows were saved with the joints hidden by default  # v1/v2 windows were saved with the grid at the base only
                if ver < 5 and f.name == "springs":
                    continue  # older windows were saved with the spring supports hidden
                if f.name in d:
                    v = d[f.name]
                    if f.type in ("bool", bool):
                        v = bool(v)
                    setattr(o, f.name, v)
        if o.grid_3d not in GRID_3D:
            o.grid_3d = "all"
        return o

    def copy(self) -> "DisplayOptions":
        return DisplayOptions.from_dict(self.to_dict())

    # ---- helpers for the scene builder
    FRAME_TYPES = {"Beam": "beams", "Column": "columns", "Brace": "braces", "Null": "null_lines", "Rigid": "rigid_links"}
    AREA_TYPES = {"Floor": "floors", "Wall": "walls", "Ramp": "ramps", "Opening": "openings", "Null": "openings"}

    def shows_frame_type(self, ftype: str) -> bool:
        return bool(getattr(self, self.FRAME_TYPES.get(ftype, "beams")))

    def shows_area_type(self, atype: str) -> bool:
        return bool(getattr(self, self.AREA_TYPES.get(atype, "floors")))


# dialog layout: (group title, [(field, label), ...]) - follows the ETABS dialog
GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Object Visibility", [("joints", "Joints"), ("beams", "Beams"), ("columns", "Columns"), ("braces", "Braces"),
                           ("null_lines", "Null lines"), ("rigid_links", "Rigid links"), ("floors", "Floors"), ("walls", "Walls"),
                           ("ramps", "Ramps"), ("openings", "Openings / null areas"),
                           ("load_panels", "Load panels (drawn load areas)")]),
    ("Assignments", [("supports", "Joint restraints"), ("springs", "Joint springs"),
                     ("releases", "Frame releases"), ("local_axes", "Frame local axes")]),
    ("Labels", [("joint_labels", "Joint labels"), ("frame_labels", "Frame labels"), ("area_labels", "Area labels"),
                ("section_names", "Frame section names")]),
    ("Other", [("extrude", "Extrude frames and areas (real sections)"), ("fill_areas", "Fill areas"),
               ("edges", "Show edges"), ("grid_lines", "Grid lines"), ("grid_bubbles", "Grid bubbles (A, B, 1, 2)"),
               ("grid_vertical", "Vertical grid lines in 3D (column lines)"),
               ("story_plane", "Show the plane of the active plan / elevation window (blue) in 3D"),
               ("analysis_mesh", "Analysis mesh of floors / walls (after an analysis)")]),
    ("Reference lines", [("reference_lines", "Reference lines through joints"),
                                   ("reference_x", "   parallel to X"), ("reference_y", "   parallel to Y"),
                                   ("reference_z", "   parallel to Z")]),
]
LABELS: Dict[str, str] = {k: v for _, items in GROUPS for k, v in items}
