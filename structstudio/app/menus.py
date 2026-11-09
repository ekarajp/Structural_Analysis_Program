"""The menu bar, laid out like ETABS 21.

The structure (menus, order, sub-menus, names and default keyboard shortcuts)
follows the public ETABS help ("Menus" chapter and *Options > Default Keyboard
Shortcuts*).  Each entry names a command id; the main window registers a
handler for every command it implements.  Entries without a handler are shown
disabled and marked *planned* so the menu already reads like ETABS while the
features are being built.

Entry forms::

    "-"                                  separator
    ("Label", "cmd.id")                  command
    ("Label", "cmd.id", "Ctrl+R")        command with its ETABS shortcut
    ("Label", [entries...])              sub-menu
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from PySide6 import QtGui, QtWidgets

Entry = Union[str, tuple]

# loads other than the four Load commands (joint / frame / area loads, load panels)
_OTHER_LOADS = [("Joint Ground Displacement...", "assign.ground_disp"),
                ("Joint Temperature...", "assign.joint_temperature"),
                ("Frame Temperature...", "assign.frame_temperature"),
                ("Shell Temperature...", "assign.shell_temperature"),
                ("Shell Non-uniform...", "assign.shell_nonuniform"),
                ("Wind Pressure Coefficients...", "assign.wind_cp"),
                ("Open Structure Wind Parameters...", "assign.open_wind")]


def _frame_design(prefix: str, lateral: bool = False, auto_null: bool = True, change: bool = True):
    items = [("View/Revise Preferences...", f"{prefix}.prefs"), ("View/Revise Overwrites...", f"{prefix}.overwrites")]
    if lateral:
        items.append(("Lateral Bracing...", f"{prefix}.bracing"))
    items += [("Select Design Groups...", f"{prefix}.groups"), ("Select Design Combinations...", f"{prefix}.combos"),
              "-", ("Start Design/Check", f"{prefix}.start"), ("Interactive Design", f"{prefix}.interactive"),
              ("Display Design Info...", f"{prefix}.info"), "-"]
    if auto_null:
        items.append(("Make Auto Select Section Null", f"{prefix}.null_auto"))
    if change:
        items.append(("Change Design Section...", f"{prefix}.change"))
    items += [("Reset Design Section to Last Analysis", f"{prefix}.reset_section"),
              ("Verify Analysis vs Design Section", f"{prefix}.verify_section"),
              ("Verify All Members Passed", f"{prefix}.verify_passed"), "-",
              ("Reset All Overwrites", f"{prefix}.reset_overwrites"), ("Delete Design Results", f"{prefix}.delete")]
    return items


OBJECT_TYPES = ("Joints", "Columns", "Beams", "Braces", "Floors", "Walls", "Openings", "Null Lines", "Supports")


def _select_items(mode: str) -> List[Entry]:
    """Select > Select / Deselect sub-menu (ETABS lists the same filters in both)."""
    s = f"select.{mode}"
    items: List[Entry] = []
    if mode == "add":
        items += [("Pointer/Window", "tool.select"), ("Poly", "select.poly", "Ctrl+Shift+O"),
                  ("Intersecting Poly", "select.intersecting_poly", "Ctrl+Shift+P"),
                  ("Intersecting Line", "tool.select_line", "Ctrl+Shift+L"),
                  ("Coordinate Specification", [("Click Joint in XY Plane", "select.coord_xy"),
                                                ("Click Joint in XZ Plane", "select.coord_xz"),
                                                ("Click Joint in YZ Plane", "select.coord_yz")]), "-"]
    items += [
        ("Object Type", [(t, f"{s}.type.{t}") for t in OBJECT_TYPES]),
        ("Properties", [("Material Properties...", f"{s}.material"), ("Frame Sections...", f"{s}.frame_section"),
                        ("Slab Sections...", f"{s}.slab_section"), ("Deck Sections...", f"{s}.deck_section"),
                        ("Wall Sections...", f"{s}.wall_section"), ("Tendon Sections...", f"{s}.tendon_section"),
                        ("Link/Support Properties...", f"{s}.link"), ("Point Springs...", f"{s}.point_spring"),
                        ("Line Springs...", f"{s}.line_spring"), ("Area Springs...", f"{s}.area_spring"),
                        ("Panel Zone...", f"{s}.panel_zone")]),
        ("Labels", [("Pier Labels...", f"{s}.pier"), ("Spandrel Labels...", f"{s}.spandrel"), "-",
                    ("Joint Labels...", f"{s}.label.point"), ("Frame Labels...", f"{s}.label.frame"),
                    ("Shell Labels...", f"{s}.label.area"), "-",
                    ("Joint Unique Names...", f"{s}.name.point"), ("Frame Unique Names...", f"{s}.name.frame"),
                    ("Shell Unique Names...", f"{s}.name.area"), ("Link Unique Names...", f"{s}.name.link")]),
        ("Groups...", f"{s}.groups", "Ctrl+G" if mode == "add" else None),
        ("Towers...", f"{s}.towers"),
        ("Stories...", f"{s}.stories"),
        "-",
        ("All", f"{s}.all", "Ctrl+A" if mode == "add" else None),
    ]
    return items


ETABS_MENUS: List[Tuple[str, List[Entry]]] = [
    ("&File", [
        ("New Model...", "file.new", "Ctrl+N"),
        ("Open...", "file.open", "Ctrl+O"),
        ("Close", "file.close", "Ctrl+Shift+E"),
        "-",
        ("Save", "file.save", "Ctrl+S"),
        ("Save As...", "file.save_as", "Ctrl+Shift+S"),
        "-",
        ("Import", [("ETABS / SAP2000 / SAFE Model (.e2k .$et .edb .s2k .$2k .sdb .f2k .fdb)...",
                     "file.import_csi", "Ctrl+Shift+I"),
                    ("CSI Database Tables (Excel / Access / text)...", "file.import_tables"),
                    "-",
                    ("DXF/DWG Architectural Grids...", "file.import_dxf_grids"),
                    ("DXF/DWG Floor Plan...", "file.import_dxf_plan"),
                    ("DXF/DWG 3D Model...", "file.import_dxf_3d"),
                    ("IFC File...", "file.import_ifc"),
                    ("Revit Structure .exr File...", "file.import_revit")]),
        ("Export", [("Results (JSON)...", "file.export_results"),
                    ("Model Tables to Text File (.tables.txt)...", "file.export_tables"),
                    ("ETABS .e2k Text File...", "file.export_e2k"),
                    ("Model to Excel...", "file.export_excel"),
                    ("DXF/DWG File...", "file.export_dxf"),
                    ("IFC File...", "file.export_ifc")]),
        "-",
        ("Create Video", [("Cyclic Animation...", "file.video_cyclic"),
                          ("Time History Animation...", "file.video_th")]),
        ("Print Graphics...", "file.print", "Ctrl+P"),
        ("Create Report...", "file.report"),
        ("Capture Picture", [("Entire Screen", "file.capture_screen", "Ctrl+Shift+5"),
                             ("Main Window", "file.capture_main", "Ctrl+Shift+6"),
                             ("Current Window", "file.capture_window", "Ctrl+Shift+7")]),
        "-",
        ("Project Information...", "file.project_info"),
        ("Comments and Log...", "file.comments", "Ctrl+Shift+C"),
        ("Show Input/Output Text Files...", "file.text_files", "Ctrl+Shift+F"),
        "-",
        ("Exit", "file.exit", "Alt+F4"),
    ]),
    ("&Edit", [
        ("Undo", "edit.undo", "Ctrl+Z"),
        ("Redo", "edit.redo", "Ctrl+Y"),
        "-",
        ("Cut", "edit.cut", "Ctrl+X"),
        ("Copy to Clipboard", "edit.copy", "Ctrl+C"),
        ("Paste (click a point, or type dx,dy,dz)", "edit.paste", "Ctrl+V"),
        ("Delete", "edit.delete", "Del"),
        ("Find Object (name, label, grid, x,y,z)...", "edit.find", "Ctrl+F"),
        "-",
        # one place for every modify command (CAD style; numbers can be typed in the window)
        ("Move  (MV)", "edit.move"),  # Ctrl+M and M V: the action's own shortcuts
        ("Copy  (CO)", "tool.copy"),
        ("Rotate  (RO)", "tool.rotate"),
        ("Mirror  (MM)", "tool.mirror"),
        ("Offset  (OF)", "tool.offset"),
        ("Trim  (TR)", "tool.trim"),
        ("Extend  (EX)", "tool.extend"),
        ("Array / Replicate to Stories...", "edit.replicate", "Ctrl+R"),
        "-",
        ("Add to Model From Template", [("2D Frames...", "edit.template_2d"), ("3D Structures...", "edit.template_3d")]),
        "-",
        ("Edit Stories and Grid Systems...", "edit.grid_stories"),
        ("Grid System Mode", "edit.grid_mode_menu"),
        ("Add Grid Lines at Selected Joints", "edit.grid_at_joints"),
        ("Grid Options", [("Glue Joints to Grids", "edit.glue_joints"),
                          ("Lock Onscreen Grid System Edit", "edit.lock_grid")]),
        "-",
        ("Interactive Database Editing...", "edit.database"),
        "-",
        ("Extrude", [("Extrude Joints to Frames...", "edit.extrude_joints"),
                     ("Extrude Frames to Shells...", "edit.extrude_frames")]),
        ("Merge Joints...", "edit.merge_joints"),
        ("Align Joints/Frames/Edges...", "edit.align", "Ctrl+Shift+M"),
        ("Edit Joint Coordinates...", "edit.point_coordinates"),
        "-",
        ("Edit Frames", [("Divide Frames...", "edit.divide_frames"), ("Join Frames", "edit.join_frames"),
                         ("Reverse Frame Connectivity", "edit.reverse_frames"),
                         ("Modify/Show Frame Type...", "edit.frame_type")]),
        ("Edit Shells", [("Divide Shells...", "edit.divide_shells"), ("Merge Shells", "edit.merge_shells"),
                         ("Expand/Shrink Shells...", "edit.expand_shells"), ("Split Shell Edge...", "edit.split_edge"),
                         ("Remove Joints From Shells", "edit.remove_shell_joints"),
                         ("Chamfer Slab Corners...", "edit.chamfer"), "-",
                         ("Reverse Wall Local 3 Axis", "edit.reverse_wall"),
                         ("Divide Walls for Openings", "edit.divide_wall_openings"),
                         ("Modify/Show Slab Edge Type...", "edit.slab_edge_type"),
                         ("Modify/Show Wall Curve Type...", "edit.wall_curve_type")]),
        ("Edit Links", [("Reverse Link Connectivity", "edit.reverse_links")]),
        ("Add/Edit Tendons", [("Add Tendons in Strips...", "edit.tendons_strips"),
                              ("Edit Plan Layout (Horizontal)...", "edit.tendon_plan"),
                              ("Edit Vertical Profile...", "edit.tendon_profile")]),
        ("Add/Edit Slab Panels...", "edit.slab_panels"),
        ("Add/Edit Support Lines...", "edit.support_lines"),
        ("Add/Edit Design Strips", [("Add Design Strips Using Grids...", "edit.strips_grids"),
                                    ("Add Design Strips Using Support Lines...", "edit.strips_support"),
                                    ("Edit Strip Widths...", "edit.strip_widths")]),
        "-",
        ("Auto Relabel All", "edit.relabel"),
        ("Change Labels...", "edit.change_labels"),
    ]),
    ("&View", [
        ("Set Default 3D View", "view.default_3d"),
        ("Set 3D View...", "view.3d", "Ctrl+Shift+F3"),
        ("Set Plan View...", "view.plan", "Ctrl+Shift+F1"),
        ("Set Elevation View...", "view.elevation", "Ctrl+Shift+F2"),
        ("Set Building View Limits...", "view.limits"),
        ("Set Display Options...", "view.display", "Ctrl+W"),
        "-",
        ("Rubber Band Zoom", "view.zoom_window", "F2"),
        ("Restore Full View", "view.full", "F3"),
        ("Zoom In One Step", "view.zoom_in", "Shift+F2"),
        ("Zoom Out One Step", "view.zoom_out", "Shift+F3"),
        ("Pan", "view.pan", "F10"),
        ("Rotate 3D View", "view.rotate"),
        ("Measure", [("Line", "tool.measure_line"), ("Area", "tool.measure_area"), ("Angle", "tool.measure_angle")]),
        "-",
        ("Set Grid System Visibility", "view.grid_menu"),
        ("Show Axes", "view.axes"),
        ("Object Labels", "view.labels"),
        ("Extruded View (real sections)", "view.extrude"),
        "-",
        ("Show Selected Objects Only", "view.show_selected", "Ctrl+Shift+J"),
        ("Invert Visibility of Objects", "view.invert_visibility"),
        ("Make Selected Objects Invisible", "view.hide_selected"),
        ("Restore Visibility of Previous Selection", "view.restore_visibility"),
        ("Show All Objects", "view.show_all", "Ctrl+Shift+A"),
        "-",
        ("Refresh Window", "view.refresh_window", "Ctrl+Shift+W"),
        ("Refresh View", "view.refresh_view", "Ctrl+Shift+V"),
        "-",
        ("Show Rendered View (DirectX)", "view.rendered"),
        ("Show Rendered View (Ray Tracing)", "view.raytrace"),
    ]),
    ("&Define", [
        ("Material Properties...", "define.materials"),
        ("Reinforcing Bar Sizes...", "define.rebar"),
        ("Section Properties", [("Frame Sections...", "define.frame_sections"),
                                ("Slab Sections...", "define.slab_sections"),
                                ("Deck Sections...", "define.deck_sections"),
                                ("Wall Sections...", "define.wall_sections"),
                                "-",
                                ("Tendon Sections...", "define.tendon_sections"),
                                ("Link/Support Properties...", "define.links"),
                                ("Frame Nonlinear Hinges...", "define.hinges"),
                                ("Panel Zone...", "define.panel_zone")]),
        ("Spring Properties", [("Point Springs...", "define.point_springs"), ("Line Springs...", "define.line_springs"),
                               ("Area Springs...", "define.area_springs"), ("Soil Profiles...", "define.soil"),
                               ("Isolated Column Footings...", "define.footings")]),
        ("Diaphragms...", "define.diaphragms"),
        ("Pier Labels...", "define.piers"),
        ("Spandrel Labels...", "define.spandrels"),
        ("Group Definitions...", "define.groups"),
        ("Section Cuts...", "define.section_cuts"),
        ("Strain Gauge Properties", [("Line Gauges...", "define.line_gauges"), ("Quad Gauges...", "define.quad_gauges")]),
        "-",
        ("Functions", [("Response Spectrum...", "define.rs_functions"), ("Time History...", "define.th_functions")]),
        ("Generalized Displacements...", "define.gen_disp"),
        ("Mass Source...", "define.mass_source"),
        ("P-Delta Options...", "define.pdelta"),
        ("Modal Cases...", "define.modal"),
        "-",
        ("Load Patterns...", "define.load_patterns"),
        ("Load Cases...", "define.load_cases"),
        ("Load Combinations...", "define.combos"),
        ("Auto Construction Sequence Case...", "define.construction"),
        ("Walking Vibrations...", "define.walking"),
        "-",
        ("Performance Checks...", "define.performance"),
        ("Table Named Sets...", "define.table_sets"),
        ("Named Assignment Sets...", "define.assign_sets"),
    ]),
    ("D&raw", [
        ("Select Object", "tool.select"),
        ("Reshape Object", "draw.reshape"),
        "-",
        ("Draw Joint Objects", "tool.point"),
        ("Draw Beam/Column/Brace Objects", [
            ("Draw Beam/Column/Brace (Plan, Elev, 3D)", "tool.beam"),
            ("Quick Draw Beams (Plan, Elev)", "tool.quick_beam"),
            ("Draw Columns (Plan, Elev, 3D)", "tool.column"),
            ("Quick Draw Columns (Plan, Elev)", "tool.quick_column"),
            ("Quick Draw Secondary Beams (Plan)", "tool.secondary_beams"),
            ("Quick Draw Braces (Elev)", "tool.quick_brace")]),
        ("Draw Floor/Wall Objects", [
            ("Draw Floor/Wall (Plan, Elev, 3D)", "tool.slab"),
            ("Draw Rectangular Floor/Wall (Plan, Elev)", "tool.rect_slab"),
            ("Quick Draw Floor/Wall (Plan, Elev)", "tool.quick_slab"),
            ("Draw Walls (Plan, Elev, 3D)", "tool.wall"),
            ("Quick Draw Walls (Plan, Elev)", "tool.quick_wall"),
            ("Draw Openings (Plan, Elev)", "tool.opening"),
            ("Draw Wall Openings (Elev)", "draw.wall_openings")]),
        ("Draw Links", "draw.links"),
        ("Draw Tendons", "draw.tendons"),
        ("Draw Slab Rebar", "draw.slab_rebar"),
        ("Draw Design Strips", "draw.strips"),
        "-",
        ("Draw Grids", "draw.grids"),
        ("Draw Dimension Lines", "draw.dimensions"),
        ("Draw Reference Points", "draw.ref_points"),
        ("Draw Reference Planes", "draw.ref_planes"),
        ("Draw Section Cuts", "draw.section_cuts"),
        ("Draw Developed Elevation Definition", "draw.developed"),
        ("Draw Wall Stacks", "draw.wall_stacks"),
        ("Auto Draw Cladding", "draw.cladding"),
        ("Draw Line Strain Gauge", "draw.line_gauge"),
        ("Draw Quad Strain Gauge", "draw.quad_gauge"),
        "-",
        ("Snaps and Guides (on / off, Options...)", "draw.snap_menu"),
    ]),
    ("&Select", [
        ("Select", _select_items("add")),
        ("Deselect", _select_items("remove")),
        "-",
        ("Invert Selection", "select.invert", "Ctrl+K"),
        ("Get Previous Selection", "select.previous", "Ctrl+J"),
        ("Clear Selection", "select.clear", "Ctrl+Q"),
    ]),
    ("&Assign", [
        ("Joint", [("Restraints...", "assign.restraints"), ("Springs...", "assign.point_springs"),
                   ("Diaphragms...", "assign.point_diaphragm"), ("Panel Zone...", "assign.panel_zone"),
                   ("Additional Mass...", "assign.point_mass"),
                   ("Joint Floor Meshing Options...", "assign.joint_meshing")]),
        ("Frame", [("Section Property...", "assign.frame_section"),
                   ("Property Modifiers...", "assign.frame_modifiers"),
                   ("Releases/Partial Fixity...", "assign.releases"),
                   ("End Length Offsets...", "assign.end_offsets"),
                   ("Insertion Point...", "assign.insertion_point"),
                   ("Local Axes...", "assign.frame_axes"),
                   ("Output Stations...", "assign.output_stations"),
                   ("Tension/Compression Limits...", "assign.tc_limits"),
                   ("Hinges...", "assign.hinges"), ("Hinge Overwrites...", "assign.hinge_overwrites"),
                   ("Line Springs...", "assign.line_springs"),
                   ("Additional Mass...", "assign.frame_mass"),
                   ("Pier Label...", "assign.frame_pier"), ("Spandrel Label...", "assign.frame_spandrel"),
                   ("Frame Auto Mesh Options...", "assign.frame_mesh"),
                   ("Frame Floor Meshing Options...", "assign.frame_floor_mesh"),
                   ("Moment Frame Beam Type...", "assign.beam_type"),
                   ("Column Splice Overwrite...", "assign.splice"),
                   ("Nonprismatic Property Parameters...", "assign.nonprismatic"),
                   ("Material Overwrites...", "assign.frame_material")]),
        ("Shell", [("Slab Section...", "assign.slab_section"), ("Deck Section...", "assign.deck_section"),
                   ("Wall Section...", "assign.wall_section"), ("Openings...", "assign.openings"),
                   ("Stiffness Modifiers...", "assign.shell_modifiers"),
                   ("Thickness Overwrites...", "assign.thickness"),
                   ("Insertion Point...", "assign.shell_insertion"),
                   ("Diaphragms...", "assign.area_diaphragm"),
                   ("Edge Releases...", "assign.edge_releases"), ("Line Releases...", "assign.line_releases"),
                   ("Local Axes...", "assign.shell_axes"),
                   ("Area Springs...", "assign.area_springs"),
                   ("Additional Mass...", "assign.shell_mass"),
                   ("Pier Label...", "assign.shell_pier"), ("Spandrel Label...", "assign.shell_spandrel"),
                   ("Wall Hinge...", "assign.wall_hinge"),
                   ("Floor Auto Mesh Options...", "assign.floor_mesh"),
                   ("Wall Auto Mesh Options...", "assign.wall_mesh"),
                   ("Auto Edge Constraints...", "assign.edge_constraints"),
                   ("Material Overwrite...", "assign.shell_material")]),
        ("Link", [("Link Properties...", "assign.link_props"), ("Local Axes...", "assign.link_axes")]),
        ("Gauge Properties", [("Line Gauge Properties...", "assign.line_gauge"),
                              ("Quad Gauge Properties...", "assign.quad_gauge")]),
        ("Pier Force Acceptance Criteria...", "assign.pier_criteria"),
        ("Spandrel Force Acceptance Criteria...", "assign.spandrel_criteria"),
        "-",

        ("Assign Objects to Tower...", "assign.tower"),
        ("Assign Objects to Group...", "assign.group"),
        ("Clear Display of Assigns", "assign.clear_display"),
        ("Copy Assigns", "assign.copy"),
        ("Paste Assigns", "assign.paste"),
    ]),
    # Load menu (CYPE 3D concept: loads are a step of their own between assigning
    # properties and analysing).  Area loads are load panels: click a floor / wall to
    # make one on it, or draw a panel over any polygon; joint and frame loads go on
    # the joints and frames (click, or the forms for the selection).
    ("&Load", [
        # one command per load kind.  Command first: set the load in Properties of
        # Object, pick the objects, Enter / right-click.  Objects first: select them,
        # then the command - the form shows their load to edit.
        ("Area Load (Floors / Walls)", "tool.area_load"),
        ("Load Panel (drawn load area)", "tool.panel_load"),
        ("Frame Load", "tool.frame_load"),
        ("Joint Load", "tool.joint_load"),
        ("Other Loads", _OTHER_LOADS),
        "-",
        ("Delete Loads on Selection...", "load.delete"),
        ("Load Panel / Area Load Report...", "load.report"),
        ("Show Loads", "load.show_menu"),
    ]),
    ("A&nalyze", [
        ("Check Model...", "analyze.check"),
        ("Set Active Degrees of Freedom...", "analyze.dof"),
        ("Set Load Cases to Run...", "analyze.cases_to_run"),
        ("Advanced SAPFire Options...", "analyze.solver"),
        ("Advanced Design and Response Recovery Options...", "analyze.recovery"),
        ("Analysis Model Defaults (end offsets, rigid zones, mesh check)...", "analyze.model_defaults"),
        ("Automatic Mesh Settings for Floors...", "analyze.floor_mesh"),
        ("Automatic Mesh Settings for Walls...", "analyze.wall_mesh"),
        ("Cracking Analysis Options...", "analyze.cracking"),
        "-",
        ("Check Mesh (find the fine mesh)...", "analyze.check_mesh"),
        ("Use Fine Mesh", "analyze.use_fine_mesh"),
        ("Run Analysis", "analyze.run", "F5"),
        ("Model Alive", "analyze.alive"),
        ("Modify Undeformed Geometry...", "analyze.modify_geometry"),
        "-",
        ("Last Analysis Run Log...", "analyze.log"),
        ("Last Analysis Problems...", "analyze.problems"),
        ("Lock/Unlock Model", "analyze.lock"),
        ("Analysis Model for Nonlinear Hinges", "analyze.hinge_model"),
    ]),
    ("Dis&play", [
        ("Undeformed Shape", "display.undeformed", "F4"),
        ("Show Analysis Mesh", "display.mesh"),
        ("Load Assigns", [("Joint...", "display.loads.point"), ("Frame...", "display.loads.frame"),
                          ("Shell...", "display.loads.area")]),
        ("Deformed Shape...", "display.deformed", "F6"),
        ("Mode Shape...", "display.mode"),
        ("Force/Stress Diagrams", [("Support/Spring Reactions...", "display.reactions", "F7"),
                                   ("Frame/Pier/Spandrel/Link Forces...", "display.frame_forces", "F8"),
                                   ("Shell Stresses/Forces...", "display.shell_forces", "F9"),
                                   ("Diaphragm Forces...", "display.diaphragm_forces"),
                                   ("Soil Pressure...", "display.soil_pressure")]),
        ("Energy/Virtual Work Diagram...", "display.energy"),
        ("Story Response Plots...", "display.story_response"),
        ("Combined Story Response...", "display.combined_story"),
        ("Response Spectrum Curves...", "display.rs_curves"),
        ("Plot Functions...", "display.plot_functions", "F12"),
        ("Static Pushover Curve...", "display.pushover"),
        ("Hinge Results...", "display.hinges"),
        "-",
        ("Save Named Display...", "display.save_named"),
        ("Show Named Display...", "display.show_named"),
        "-",
        ("Show Tables...", "display.tables", "Ctrl+T"),
    ]),
    ("Desi&gn", [
        ("Steel Frame Design", _frame_design("design.steel", lateral=True)),
        ("Concrete Frame Design", _frame_design("design.concrete", auto_null=False)),
        ("Composite Beam Design", _frame_design("design.composite_beam")),
        ("Composite Column Design", _frame_design("design.composite_column", change=False)),
        ("Steel Joist Design", _frame_design("design.joist", lateral=True)),
        ("Overwrite Frame Design Procedure...", "design.procedure"),
        "-",
        ("Shear Wall Design", [("View/Revise Preferences...", "design.wall.prefs"),
                               ("Assign Pier Sections...", "design.wall.pier_sections"),
                               ("View/Revise Pier Overwrites...", "design.wall.pier_overwrites"),
                               ("View/Revise Spandrel Overwrites...", "design.wall.spandrel_overwrites"),
                               ("Select Design Combinations...", "design.wall.combos"), "-",
                               ("Start Design/Check", "design.wall.start"),
                               ("Interactive Wall Design", "design.wall.interactive"),
                               ("Display Design Info...", "design.wall.info"), "-",
                               ("Reset All Pier/Spandrel Overwrites", "design.wall.reset"),
                               ("Delete Design Results", "design.wall.delete")]),
        ("Concrete Slab Design", [("View/Revise Preferences...", "design.slab.prefs"),
                                  ("View/Revise Flexural Design Overwrites...", "design.slab.flex_overwrites"),
                                  ("View/Revise Punching Check Overwrites...", "design.slab.punch_overwrites"),
                                  ("Select Design Combinations...", "design.slab.combos"),
                                  ("Select Stories for Slab Design...", "design.slab.stories"), "-",
                                  ("Start Design", "design.slab.start"),
                                  ("Display Flexural Design...", "design.slab.flexure"),
                                  ("Display Punching Check...", "design.slab.punching"), "-",
                                  ("Reset All Flexural Design Overwrites", "design.slab.reset_flex"),
                                  ("Reset All Punching Overwrites", "design.slab.reset_punch"),
                                  ("Delete Design Results", "design.slab.delete")]),
        "-",
        ("Live Load Reduction Factors...", "design.llrf"),
        ("Set Lateral Displacement Targets...", "design.drift_targets"),
        ("Set Time Period Targets...", "design.period_targets"),
    ]),
    ("De&tailing", [
        ("Bar Selection and Curtailment Rules", [("Beams...", "detail.rules_beams"),
                                                 ("Slabs / Mats / Footings...", "detail.rules_slabs"),
                                                 ("Column Stacks...", "detail.rules_columns"),
                                                 ("Walls...", "detail.rules_walls")]),
        ("Clear Detailing", "detail.clear"),
        ("Start Concrete Detailing", "detail.concrete"),
        ("Start Steel Detailing", "detail.steel"),
        ("Drawing Management", [("Drawing Sheet Setup...", "detail.sheets"),
                                ("Drawing Format Properties...", "detail.format"),
                                ("Drawing Sheet List...", "detail.sheet_list"),
                                ("Open Drawing Manager...", "detail.manager")]),
        ("Generate / Show Beam Groups", "detail.beam_groups"),
        ("Detailing Comments / Log...", "detail.log"),
    ]),
    ("&Window", [
        ("Panels (show / hide)", "options.panels_menu"),
        ("Show Model Explorer", "options.model_explorer"),
        ("Draw Toolbar (left palette)", "options.palette_menu"),
        ("Reset Default Layout", "options.reset_toolbars"),
        "-",
        ("Window Layout (1 - 4 model windows)", "options.windows_menu"),
        ("Arrange Windows...", "options.arrange"),
        "-",
        ("Dark / Light Mode", "window.theme"),
        ("Customize Toolbars...", "options.toolbars"),
        ("Graphics", [
            ("Graphics Preferences", "options.theme_menu"),
            ("Color Objects By", "options.color_by_menu"),
            ("Graphics Colors", [("Display Colors...", "options.colors_display"),
                                 ("Output Colors...", "options.colors_output")]),
            ("Graphics Sizes (supports, lines, text)...", "options.sizes"),
            ("Moment Diagrams on Tension Side", "options.moment_tension"),
            ("Plan Diagrams: Positive Below / Right", "options.plan_positive_below"),
            ("Architectural Plan Options...", "options.arch_plan"),
            ("Show Crosshairs", "options.crosshairs"),
            ("Show Bounding Plane", "options.bounding_plane"),
            ("Show Value at Cursor for Assign/Design Info", "options.value_at_cursor"),
            "-",
            ("Restore Graphics Defaults...", "options.graphics_defaults"),
        ]),
    ]),
    ("&Options", [
        ("Display Units...", "options.units", "Ctrl+U"),
        ("Tolerances...", "options.tolerances"),
        "-",
        ("Auto Save Model...", "options.autosave"),
        ("Show Start Page at Startup", "options.start_page"),
        ("Allow Multiple Towers", "options.towers"),
        ("Animation Sound", "options.sound"),
        ("Use Multi-processing for Design/Results", "options.multiprocessing"),
        "-",
        ("Default Keyboard Shortcuts...", "options.shortcuts"),
        ("Save User Default Settings", "options.save_defaults"),
        ("Reset User Default Settings", "options.reset_defaults"),
    ]),
    ("&Help", [
        ("Structural Studio Help", "help.help", "F1"),
        ("Documentation...", "help.docs"),
        ("Structural Studio on the Web", "help.web"),
        ("News", "help.news"),
        ("Watch and Learn", "help.learn"),
        "-",
        ("About Structural Studio", "help.about"),
        ("Change Language", "help.language", "Ctrl+L"),
    ]),
]


def iter_commands(entries: Sequence[Entry] = None, path: Tuple[str, ...] = ()):
    """(menu path, label, command id, shortcut) for every command in the spec."""
    if entries is None:
        for title, items in ETABS_MENUS:
            yield from iter_commands(items, (title.replace("&", ""),))
        return
    for e in entries:
        if e == "-":
            continue
        label, target = e[0], e[1]
        if isinstance(target, list):
            yield from iter_commands(target, path + (label,))
        else:
            yield path, label, target, (e[2] if len(e) > 2 else None)


PLANNED_TIP = "Planned - not available in this version yet"


class MenuBuilder:
    """Builds the ETABS menu bar from :data:`ETABS_MENUS`.

    ``actions`` : command id -> existing QAction (shared with toolbars);
    ``commands``: command id -> callable (an action is made for it);
    ``menus``   : command id -> callable(parent menu) that fills a sub-menu
    (e.g. the theme choices).  Anything else becomes a disabled, *planned* entry.
    """

    def __init__(self, window: QtWidgets.QMainWindow, actions: Dict[str, QtGui.QAction],
                 commands: Dict[str, Callable], menus: Dict[str, Callable[[QtWidgets.QMenu], None]],
                 make: Callable[..., QtGui.QAction]):
        self.window = window
        self.actions = actions
        self.commands = commands
        self.menus = menus
        self.make = make
        self.planned: List[str] = []
        self.implemented: List[str] = []

    def build(self, menubar: QtWidgets.QMenuBar) -> Dict[str, QtWidgets.QMenu]:
        out = {}
        for title, items in ETABS_MENUS:
            menu = menubar.addMenu(title)
            menu.setToolTipsVisible(True)
            self._fill(menu, items)
            out[title.replace("&", "")] = menu
        return out

    def _fill(self, menu: QtWidgets.QMenu, items: Sequence[Entry]):
        for e in items:
            if e == "-":
                menu.addSeparator()
                continue
            label, target = e[0], e[1]
            shortcut = e[2] if len(e) > 2 else None
            if isinstance(target, list):
                sub = menu.addMenu(label)
                sub.setToolTipsVisible(True)
                self._fill(sub, target)
                if not any(a.isEnabled() for a in sub.actions() if not a.isSeparator()):
                    sub.menuAction().setToolTip(PLANNED_TIP)
                    sub.menuAction().setEnabled(False)
                continue
            if target in self.menus:
                sub = menu.addMenu(label)
                sub.setToolTipsVisible(True)
                self.menus[target](sub)
                self.implemented.append(target)
                continue
            menu.addAction(self.action(target, label, shortcut))

    def action(self, cid: str, label: str, shortcut: Optional[str]) -> QtGui.QAction:
        a = self.actions.get(cid)
        if a is not None:
            if a.property("menu_text") is None:  # ETABS wording; the first menu that uses it wins
                a.setProperty("menu_text", label)
                a.setText(label)
            if shortcut:
                a.setShortcut(QtGui.QKeySequence(shortcut))
            self.implemented.append(cid)
            return a
        fn = self.commands.get(cid)
        if fn is not None:
            a = self.make(label, lambda _=False, f=fn: f(), shortcut)
            self.actions[cid] = a
            self.implemented.append(cid)
            return a
        a = QtGui.QAction(label, self.window)
        a.setEnabled(False)
        a.setToolTip(PLANNED_TIP)
        a.setStatusTip(PLANNED_TIP)
        a.setProperty("planned", True)
        self.actions[cid] = a
        self.planned.append(cid)
        return a
