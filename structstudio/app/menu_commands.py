"""Handlers of the ETABS-style menu commands (mixed into :class:`MainWindow`).

The menu layout lives in :mod:`.menus`; the model operations in
:mod:`.commands`.  This module only asks for input (small forms) and calls
those operations on the current selection.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..model import ModelError
from . import commands as C
from . import dialogs
from .forms import TabbedFormDialog, ask
from .menus import OBJECT_TYPES, iter_commands

CARDINAL_POINTS = ["1 Bottom left", "2 Bottom center", "3 Bottom right", "4 Middle left", "5 Middle center",
                   "6 Middle right", "7 Top left", "8 Top center", "9 Top right", "10 Centroid",
                   "11 Shear center"]
DOF_LABELS = ("UX", "UY", "UZ", "RX", "RY", "RZ")


class MenuCommandsMixin:
    """ETABS menu commands.  ``menu_commands()`` maps command ids to handlers."""

    clipboard: Optional[C.Clipboard] = None
    assign_clipboard: Optional[C.AssignClipboard] = None

    # ------------------------------------------------------------ registry
    def menu_commands(self) -> Dict[str, Callable]:
        m: Dict[str, Callable] = {
            # File
            "file.close": self.close_model,
            "file.import_tables": self.import_csi,
            "file.export_results": self.export_results,
            "file.capture_window": lambda: self.capture_picture("window"),
            "file.capture_region": lambda: self.capture_picture("region"),
            "file.capture_main": self.capture_main,
            "file.project_info": self.project_info,
            "file.exit": self.close,
            # Edit
            "edit.cut": self.edit_cut,
            "edit.copy": self.edit_copy,
            "edit.paste": self.edit_paste,
            "edit.grid_stories": self.edit_grid_stories,
            "edit.grid_at_joints": self.edit_grid_at_joints,
            "edit.replicate": self.edit_replicate_tabs,
            "edit.extrude_joints": self.edit_extrude_joints,
            "edit.extrude_frames": self.edit_extrude_frames,
            "edit.merge_joints": self.edit_merge_joints,
            "edit.align": self.edit_align,
            "edit.point_coordinates": self.edit_point,
            "edit.join_frames": self.edit_join_frames,
            "edit.reverse_frames": self.edit_reverse_frames,
            "edit.beam_axis": self.edit_beam_axis,
            "edit.frame_type": self.edit_frame_type,
            "edit.divide_shells": self.edit_divide_shells,
            "edit.merge_shells": self.edit_merge_shells,
            "edit.expand_shells": self.edit_expand_shells,
            "edit.divide_wall_openings": self.edit_divide_for_openings,
            "edit.reverse_wall": self.edit_reverse_wall,
            "edit.relabel": self.edit_relabel,
            # View
            "view.plan": self.view_set_plan,
            "view.elevation": self.view_set_elevation,
            "view.show_selected": lambda: self.viewport.show_selected_only(),
            "view.invert_visibility": lambda: self.viewport.invert_visibility(),
            "view.hide_selected": lambda: self.viewport.make_selected_invisible(),
            "view.restore_visibility": lambda: self.viewport.restore_previous_visibility(),
            "view.show_all": lambda: self.viewport.show_all_objects(),
            "view.refresh_window": lambda: self.viewport.rebuild(),
            "view.refresh_view": self.view_refresh_all,
            "view.rendered": lambda: self.views.apply_display(self._with(extrude=True), self.viewport),
            # Define
            "define.materials": lambda: self._define("materials", "Materials", dialogs.material_factory),
            "define.rebar": self.define_rebar_sizes,
            "define.frame_sections": self.define_frame_sections,
            "define.slab_sections": self.define_shell_sections,
            "define.wall_sections": self.define_shell_sections,
            "define.diaphragms": self.define_diaphragms,
            "define.groups": self.define_groups,
            "define.mass_source": self.define_mass_source,
            "define.modal": self.add_modal_case,
            "define.load_patterns": lambda: self._define("load_patterns", "Load Patterns",
                                                         dialogs.load_pattern_factory),
            "define.load_cases": lambda: self._define("load_cases", "Load Cases", dialogs.static_case_factory),
            "define.combos": lambda: self._define("load_combinations", "Load Combinations", dialogs.combo_factory),
            # Draw
            "draw.grids": self.edit_grid_stories,
            "draw.snap_options": self.draw_snap_options,
            "draw.snap_only": self.toggle_snap_only,
            # Select
            "select.invert": self.state.invert_selection,
            "select.previous": self.state.previous_selection,
            "select.clear": self.state.clear_selection,
            # Assign - panels
            "assign.panel_loads": self.assign_panel_loads,
            # Assign - joint
            "assign.restraints": self.assign_restraints,
            "assign.point_springs": self.assign_point_springs,
            "assign.point_diaphragm": lambda: self.assign_diaphragm(("point",)),
            "assign.point_mass": self.assign_point_mass,
            # Assign - frame
            "assign.frame_section": self.assign_frame_section,
            "assign.frame_modifiers": self.assign_frame_modifiers,
            "assign.releases": self.assign_releases_full,
            "assign.end_offsets": self.assign_end_offsets,
            "assign.insertion_point": self.assign_insertion_point,
            "assign.frame_axes": lambda: self._assign_angle("frame"),
            "assign.output_stations": self.assign_output_stations,
            "assign.tc_limits": self.assign_tc_limits,
            "assign.frame_mass": lambda: self._assign_number("frame", "mass_per_length", "Frame Additional Mass",
                                                             "Mass per unit length"),
            "assign.frame_pier": lambda: self._assign_label("frame", "pier"),
            "assign.frame_spandrel": lambda: self._assign_label("frame", "spandrel"),
            "assign.frame_mesh": self.assign_frame_mesh,
            "assign.frame_floor_mesh": self.assign_frame_floor_mesh,
            # Assign - shell
            # commands in Properties of Object, like Assign Frame Section (pick, Enter)
            "assign.slab_section": lambda: self.load_command("slab_section"),
            "assign.wall_section": lambda: self.load_command("wall_section"),
            "assign.deck_section": lambda: self.load_command("deck_section"),
            "assign.openings": self.assign_openings,
            "assign.slab_openings": lambda: self.assign_openings("Slab"),
            "assign.wall_openings": lambda: self.assign_openings("Wall"),
            "assign.shell_modifiers": self.assign_shell_modifiers,
            "assign.slab_modifiers": lambda: self.assign_shell_modifiers("Slab"),
            "assign.wall_modifiers": lambda: self.assign_shell_modifiers("Wall"),
            "assign.area_diaphragm": lambda: self.assign_diaphragm(("area",)),
            "assign.shell_axes": lambda: self._assign_angle("area"),
            "assign.slab_axes": lambda: self._assign_angle("area", "Slab"),
            "assign.wall_axes": lambda: self._assign_angle("area", "Wall"),
            "assign.area_springs": lambda: self._assign_number("area", "spring", "Slab Area Springs",
                                                               "Subgrade modulus (force / length^3)", "Slab"),
            "assign.wall_springs": lambda: self._assign_number("area", "spring", "Wall Area Springs",
                                                               "Spring modulus (force / length^3)", "Wall"),
            "assign.shell_mass": lambda: self._assign_number("area", "mass_per_area", "Shell Additional Mass",
                                                             "Mass per unit area"),
            "assign.slab_mass": lambda: self._assign_number("area", "mass_per_area", "Slab Additional Mass",
                                                            "Mass per unit area", "Slab"),
            "assign.wall_mass": lambda: self._assign_number("area", "mass_per_area", "Wall Additional Mass",
                                                            "Mass per unit area", "Wall"),
            "assign.shell_pier": lambda: self._assign_label("area", "pier", "Wall"),
            "assign.shell_spandrel": lambda: self._assign_label("area", "spandrel", "Wall"),
            "assign.floor_mesh": lambda: self.assign_area_mesh("Floor"),
            "assign.wall_mesh": lambda: self.assign_area_mesh("Wall"),
            # Assign - loads
            # the ETABS ids run the Load-menu commands (one command per load kind)
            "assign.joint_force": lambda: self.load_command("joint_load"),
            "assign.frame_point_load": lambda: self.load_command("frame_load", frame_load_kind="Point"),
            "assign.frame_dist_load": lambda: self.load_command("frame_load", frame_load_kind="Uniform"),
            "assign.shell_uniform": lambda: self.load_command("area_load"),
            # Load menu
            "load.delete": self.delete_loads_on_selection,
            "load.report": self.load_report,
            "assign.group": self.assign_group,
            "assign.clear_display": self.clear_assign_display,
            "assign.copy": self.copy_assigns,
            "assign.paste": self.paste_assigns,
            # Analyze
            "analyze.check": self.validate_model,
            "analyze.dof": self.analyze_dof,
            "analyze.model_defaults": self.analyze_model_defaults,
            "analyze.floor_mesh": lambda: self.analyze_mesh("floor"),
            "analyze.wall_mesh": lambda: self.analyze_mesh("wall"),
            "analyze.log": self.show_log,
            # Display
            "display.undeformed": self.display_undeformed,
            "load.show": self.show_loads_dialog,
            "display.loads.point": lambda: self.display_loads("point"),
            "display.loads.frame": lambda: self.display_loads("frame"),
            "display.loads.area": lambda: self.display_loads("area"),
            "display.deformed": self.display_deformed,
            "display.mode": self.display_mode_shape,
            "display.reactions": self.display_reactions,
            "display.frame_forces": lambda: self.display_forces("frame"),
            "display.shell_forces": lambda: self.display_forces("shell"),
            "display.tables": self.show_tables,
            # Options
            "options.units": self.options_units,
            "options.tolerances": self.options_tolerances,
            "options.arrange": self.arrange_windows,
            "options.colors_display": lambda: self.edit_colors_tab(0),
            "options.colors_output": lambda: self.edit_colors_tab(1),
            "options.sizes": lambda: self.edit_colors_tab(2),
            "options.graphics_defaults": self.restore_graphics_defaults,
            "options.reset_toolbars": self.reset_toolbars,
            "options.shortcuts": self.show_shortcuts,
            # Help
            "help.help": self.show_help,
            "help.docs": self.show_help,
            "help.about": self.show_about,
        }
        for mode in ("add", "remove"):
            for t in OBJECT_TYPES:
                m[f"select.{mode}.type.{t}"] = lambda mode=mode, t=t: self._select(mode, *self._by_type(t))
            m[f"select.{mode}.material"] = lambda mode=mode: self.select_by_property(mode, "material")
            m[f"select.{mode}.frame_section"] = lambda mode=mode: self.select_by_property(mode, "frame_section")
            m[f"select.{mode}.slab_section"] = lambda mode=mode: self.select_by_property(mode, "slab_section")
            m[f"select.{mode}.wall_section"] = lambda mode=mode: self.select_by_property(mode, "wall_section")
            m[f"select.{mode}.pier"] = lambda mode=mode: self.select_by_pier(mode, False)
            m[f"select.{mode}.spandrel"] = lambda mode=mode: self.select_by_pier(mode, True)
            for kind in ("point", "frame", "area"):
                m[f"select.{mode}.label.{kind}"] = lambda mode=mode, k=kind: self.select_by_label(mode, k)
                m[f"select.{mode}.name.{kind}"] = lambda mode=mode, k=kind: self.select_by_label(mode, k, True)
            m[f"select.{mode}.groups"] = lambda mode=mode: self.select_by_group(mode)
            m[f"select.{mode}.stories"] = lambda mode=mode: self.select_by_stories(mode)
            m[f"select.{mode}.all"] = (lambda: self.viewport.select_all()) if mode == "add" else \
                self.state.clear_selection
        return m

    def menu_fillers(self) -> Dict[str, Callable[[QtWidgets.QMenu], None]]:
        """Command ids that open a sub-menu of existing (checkable) actions."""
        def grid_mode(menu):
            for a in self.grid_mode_actions.values():
                menu.addAction(a)

        def grid(menu):
            for a in (self.act_grid, self.act_bubbles, self.act_grid_vertical, self.act_story_plane):
                menu.addAction(a)
            menu.addMenu(self.grid3d_menu)
            menu.addMenu(self.reference_menu)

        def snaps(menu):
            self._fill_snap_menu(menu)

        def windows(menu):
            menu.addAction(self.act_new_window)
            menu.addSeparator()
            for a in self.layout_actions.values():
                menu.addAction(a)

        def panels(menu):
            for a in self.panel_actions.values():
                menu.addAction(a)

        def themes(menu):
            for a in self.theme_actions.values():
                menu.addAction(a)

        def color_by(menu):
            for a in self.color_by_actions.values():
                menu.addAction(a)

        def palette(menu):
            group = QtGui.QActionGroup(self)
            self.palette_mode_actions = {}
            for key, label in (("compact", "Compact (grouped drop-downs)"),
                               ("full", "Full (every tool as a button)")):
                a = self._make(label, lambda _=False, k=key: self.set_palette_mode(k), checkable=True,
                               checked=(getattr(self, "palette_mode", "compact") == key))
                group.addAction(a)
                menu.addAction(a)
                self.palette_mode_actions[key] = a

        def load_kinds(menu):
            def fill():  # the actions are made with the display toolbar, after the menu bar
                if not menu.actions():
                    for a in getattr(self, "load_kind_actions", {}).values():
                        menu.addAction(a)
            fill()
            menu.aboutToShow.connect(fill)

        def move_window(menu):
            def fill():
                menu.clear()
                self.views.fill_move_menu(menu)
            fill()
            menu.aboutToShow.connect(fill)

        def toolbars(menu):
            self.fill_toolbars_menu(menu)
            menu.aboutToShow.connect(lambda: self.fill_toolbars_menu(menu))

        def recent(menu):
            self._fill_recent_menu(menu)
            menu.aboutToShow.connect(lambda: self._fill_recent_menu(menu))

        return {"file.recent_menu": recent, "options.palette_menu": palette, "edit.grid_mode_menu": grid_mode, "view.grid_menu": grid, "draw.snap_menu": snaps,
                "load.show_menu": load_kinds,
                "options.windows_menu": windows, "options.move_window_menu": move_window, "options.toolbars_menu": toolbars, "options.panels_menu": panels, "options.theme_menu": themes, "options.color_by_menu": color_by}

    # -------------------------------------------------------------- utils
    def _info(self, text: str):
        self.statusBar().showMessage(text, 6000)
        if hasattr(self, "logger"):
            try:
                self.logger.log(text)
            except Exception:
                pass

    def _with(self, **kw):
        o = self.viewport.display_options
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def _select(self, mode: str, frames=(), areas=(), points=()):
        if mode == "add":
            self.state.set_selection(frames, areas, points, add=True)
        else:
            self.state.deselect(frames, areas, points)

    def _by_type(self, t: str):
        from .tools import select_by_type
        return select_by_type(self.doc.model, t)

    def _need_any(self) -> bool:
        if any(self._sel()):
            return True
        return self._wait_for("any", 2)

    def _try(self, title: str, fn):
        try:
            return fn()
        except (ModelError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, title, str(exc))
            return None

    # --------------------------------------------------------------- File
    def close_model(self):
        if not self.maybe_save_changes("Close"):
            return
        u = self.doc.model.units
        self.doc.new(f"{u.force} {u.length}")

    def capture_picture(self, what: str = "main"):
        """File > Capture Picture: the program window / the active model window (with its
        drawing - the native OpenGL views are painted in from their render windows) or a
        region dragged over the program window."""
        from . import capture
        vps = list(self.views.viewports)
        if what == "window":
            vp = self.views.active
            pm = capture.compose(vp, [vp])
            return capture.save_picture(self, pm, "Capture Current Window", "view.png")
        pm = capture.compose(self, vps)
        if what == "region":
            g = self.geometry()
            top_left = self.mapToGlobal(QtCore.QPoint(0, 0))
            d = capture.RegionPicker(pm, QtCore.QRect(top_left, g.size()), self)
            if not d.exec():
                return None
            part = d.region()
            if part is None:
                return None
            return capture.save_picture(self, part, "Capture Selected Region", "region.png")
        return capture.save_picture(self, pm, "Capture Main Window", "main.png")

    def capture_main(self):
        return self.capture_picture("main")

    def project_info(self):
        """File > Project Information (reports, drawing title blocks)."""
        from .project_info import edit_project_info, info_dict
        v = edit_project_info(info_dict(self.doc.model.info), self)
        if v:
            self.doc.edit(lambda m: [setattr(m.info, k, val) for k, val in v.items()])

    def show_log(self):
        self.dock_log.show()
        self.dock_log.raise_()

    def show_tables(self):
        """Display > Show Tables: open the Results panel (hidden by default so the
        drawing gets the whole window)."""
        self.dock_results.show()
        self.dock_results.raise_()

    # --------------------------------------------------------------- Edit
    def edit_copy(self):
        if self._need_any():
            self.state.clipboard = C.copy_objects(self.doc.model, *self._sel())
            f, a, p = self._sel()
            self._info(f"Copied {len(f)} frame(s), {len(a)} shell(s), {len(p)} joint(s) - Ctrl+V to place them")

    def edit_cut(self):
        if self._need_any():
            self.edit_copy()
            self.state.delete_selected()

    def edit_paste(self):
        """Ctrl+V: the copied objects follow the cursor; click where they go (CAD paste)."""
        if not self.state.clipboard or self.state.clipboard.empty:
            QtWidgets.QMessageBox.information(self, "Paste", "Nothing to paste - use Edit > Copy first.")
            return
        self.state.set_tool("paste")

    @property
    def clipboard(self):
        return self.state.clipboard

    @clipboard.setter
    def clipboard(self, clip):
        self.state.clipboard = clip

    def edit_grid_at_joints(self):
        if not self._need("point"):
            return
        n = C.add_grid_lines_at_points(self.doc, list(self.state.points))
        self._info(f"Added {n} grid line(s)")

    def edit_replicate_tabs(self):
        if not self._need_any():
            return
        stories = [s.name for s in self.doc.model.stories_bottom_up()][::-1]
        # mirror copies: Edit > Mirror (MM) draws the axis in the window
        d = TabbedFormDialog("Array / Replicate to Stories", [
            ("linear", "Linear", [("dx", "dx", 0.0), ("dy", "dy", 0.0), ("dz", "dz", 0.0),
                                  ("n", "Number", 1, {"min": 1, "max": 1000})]),
            ("radial", "Radial", [("cx", "Rotate about point  X", 0.0), ("cy", "Y", 0.0),
                                  ("angle", "Angle (deg)", 90.0), ("n", "Number", 1, {"min": 1, "max": 360})]),
            ("story", "Story", [("stories", "Stories (comma separated, blank = all above)", "")]),
        ], self, extra=[("delete", "Delete original objects", False)])
        if not d.exec():
            return
        kind, v = d.result_values()
        sel = self._sel()
        if kind == "story":
            names = [s.strip() for s in v["stories"].split(",") if s.strip()] or stories
            from .tools import replicate_to_stories
            made = self._try("Replicate", lambda: replicate_to_stories(self.doc, *sel, names,
                                                                       delete_original=v.get("delete", False)))
        else:
            if kind == "linear":
                T, rev = C.linear_transforms(v["dx"], v["dy"], v["dz"], v["n"]), False
            else:
                T, rev = C.radial_transforms(v["cx"], v["cy"], v["angle"], v["n"]), False
            made = self._try("Replicate", lambda: C.replicate_transform(self.doc, *sel, T, reverse_areas=rev,
                                                                        delete_original=v.get("delete", False)))
        self._info(f"Replicate: created {len(made or [])} object(s)")

    def edit_extrude_joints(self):
        if not self._need("point"):
            return
        secs = list(self.doc.model.frame_sections)
        if not secs:
            return QtWidgets.QMessageBox.information(self, "Extrude", "Define a frame section first.")
        v = ask(self, "Extrude Joints to Frames", [("section", "Property", secs[0], {"choices": secs}),
                                                   ("dx", "dx", 0.0), ("dy", "dy", 0.0), ("dz", "dz", 3.0),
                                                   ("n", "Number", 1, {"min": 1, "max": 1000})])
        if v:
            made = self._try("Extrude", lambda: C.extrude_points_to_frames(
                self.doc, list(self.state.points), v["dx"], v["dy"], v["dz"], v["n"], v["section"])) or []
            self.state.set_selection(made)

    def edit_extrude_frames(self):
        if not self._need("frame"):
            return
        secs = list(self.doc.model.shell_sections)
        v = ask(self, "Extrude Frames to Shells", [("section", "Property", secs[0] if secs else "None",
                                                    {"choices": secs + ["None"]}),
                                                   ("dx", "dx", 0.0), ("dy", "dy", 0.0), ("dz", "dz", 3.0),
                                                   ("n", "Number", 1, {"min": 1, "max": 1000}),
                                                   ("delete", "Delete source frames", False)])
        if v:
            sec = None if v["section"] == "None" else v["section"]
            made = self._try("Extrude", lambda: C.extrude_frames_to_shells(
                self.doc, list(self.state.frames), v["dx"], v["dy"], v["dz"], v["n"], sec,
                delete_frames=v["delete"])) or []
            self.state.set_selection(areas=made)

    def edit_merge_joints(self):
        if not self._need("point"):
            return
        v = ask(self, "Merge Selected Joints", [("tol", "Merge tolerance", 10 * self.doc.model.options.merge_tolerance)])
        if v:
            n = C.merge_joints(self.doc, list(self.state.points), v["tol"])
            self.state.clear_selection()
            self._info(f"Merged {n} joint(s)")

    def edit_align(self):
        if not self._need("point"):
            return
        v = ask(self, "Align Selected Joints", [("axis", "Align joints to", "Z-Ordinate",
                                                 {"choices": ["X-Ordinate", "Y-Ordinate", "Z-Ordinate"]}),
                                                ("value", "Ordinate", 0.0)])
        if v:
            C.align_points(self.doc, list(self.state.points), v["axis"][0], v["value"])

    def edit_join_frames(self):
        if not self._need("frame"):
            return
        joined = C.join_frames(self.doc, list(self.state.frames))
        self.state.set_selection([f for f in self.state.frames if f in self.doc.model.frames])
        self._info(f"Joined {len(joined)} frame(s)")

    def edit_reverse_frames(self):
        if self._need("frame"):
            C.reverse_frames(self.doc, list(self.state.frames))

    def edit_beam_axis(self):
        """Edit > Edit Frames > Move Beams to Real Axis (Rigid Links)."""
        if not self._need("frame"):
            return
        m = self.doc.model
        names = [n for n in self.state.frames if n in m.frames]
        lu = m.units.length
        from_offsets, _ = C.beam_axis_moves(m, names, None)
        use = "The beam's own joint offsets / insertion point" if from_offsets else "A distance"
        v = ask(self, "Move Beams to Real Axis (Rigid Links)", [
            ("how", "Move by", use, {"choices": ["The beam's own joint offsets / insertion point", "A distance"]}),
            ("d", f"Distance [{lu}] (+ = to the left of I -> J seen from above)", 0.0)],
            note="Each selected beam is moved in plan to where it really is, and joined to its old joints by "
                 "rigid links (drawn, and analysed as rigid). The beam line you see is then the real beam axis, "
                 "so draw or reshape the floor to the real slab edge. Joints on the old line used by other "
                 "members get a rigid link to the new line too. One Undo restores everything.")
        if not v:
            return
        dist = None if v["how"].startswith("The beam") else float(v["d"])
        if dist is not None and abs(dist) <= m.options.merge_tolerance:
            QtWidgets.QMessageBox.information(self, "Move Beams to Real Axis", "The distance is zero: nothing to move.")
            return
        moved, links, notes = C.move_beams_to_axis(self.doc, names, dist)
        self.state.set_selection([n for n in moved + links if n in self.doc.model.frames])
        self._info(f"Moved {len(moved)} beam(s) to their real axis with {len(links)} rigid link(s)")
        if notes or not moved:
            QtWidgets.QMessageBox.information(self, "Move Beams to Real Axis",
                                              (f"Moved {len(moved)} beam(s), {len(links)} rigid link(s).\n\n" if moved else
                                               "Nothing was moved.\n\n") + "\n".join(notes[:20]))

    def edit_frame_type(self):
        if not self._need("frame"):
            return
        cur = self.doc.model.frames[next(iter(self.state.frames))].type
        v = ask(self, "Modify/Show Frame Type", [("type", "Frame type", cur,
                                                  {"choices": ["Beam", "Column", "Brace", "Null", "Rigid"]})])
        if v:
            C.assign(self.doc, "frame", list(self.state.frames), type=v["type"])

    def edit_divide_shells(self):
        if not self._need("area"):
            return
        v = ask(self, "Divide Shells", [("n1", "Divide along edge 1-2 into", 2, {"min": 1, "max": 100}),
                                        ("n2", "Divide along edge 1-3 into", 2, {"min": 1, "max": 100})],
                note="Quadrilateral shells are divided into n1 x n2 objects.")
        if v:
            made = C.divide_shells(self.doc, list(self.state.areas), v["n1"], v["n2"])
            self.state.set_selection(areas=made)

    def _shell_edit_done(self, title: str, done: list, notes: list, what: str):
        if done:
            self.state.set_selection(areas=[n for n in done if n in self.doc.model.areas])
        self._info(f"{title}: {what}" + (f" ({len(notes)} skipped)" if notes else ""))
        if notes:
            QtWidgets.QMessageBox.information(self, title, "\n".join(notes[:20]))

    def edit_merge_shells(self):
        if not self._need("area"):
            return
        from . import shell_edit as SE
        done, notes = SE.merge_shells(self.doc, list(self.state.areas))
        self._shell_edit_done("Merge Shells", done, notes,
                              f"merged into {', '.join(done)}" if done else "nothing merged")

    def edit_expand_shells(self):
        if not self._need("area"):
            return
        v = ask(self, "Expand/Shrink Shells", [("offset", "Offset all edges by", 0.1, {"item": "coordinates"})],
                note="Positive expands, negative shrinks.  Neighbouring objects are not changed.")
        if not v:
            return
        from . import shell_edit as SE
        done, notes = SE.expand_shells(self.doc, list(self.state.areas), v["offset"])
        self._shell_edit_done("Expand/Shrink Shells", done, notes, f"{len(done)} shell(s) changed")

    def edit_divide_for_openings(self):
        if not self._need("area"):
            return
        from . import shell_edit as SE
        done, notes = SE.divide_for_openings(self.doc, list(self.state.areas))
        self._shell_edit_done("Divide Walls for Openings", done, notes, f"{len(done)} piece(s)")

    def edit_reverse_wall(self):
        if self._need("area"):
            C.reverse_area_axis(self.doc, list(self.state.areas))

    def edit_relabel(self):
        if QtWidgets.QMessageBox.question(self, "Auto Relabel All",
                                          "Relabel every joint, frame and shell (per story, by position)?") \
                == QtWidgets.QMessageBox.Yes:
            C.auto_relabel(self.doc)

    # --------------------------------------------------------------- View
    def view_set_plan(self):
        stories = [s.name for s in self.doc.model.stories_bottom_up()][::-1]
        if not stories:
            return
        cur = self.viewport.config.story if self.viewport.config.story in stories else stories[0]
        v = ask(self, "Set Plan View", [("story", "Plan view at", cur, {"choices": stories})])
        if v:
            self.viewport.set_view("plan", story=v["story"])

    def view_set_elevation(self):
        lines = [(g.name, g.direction) for g in self.doc.model.grid_lines if g.direction in ("X", "Y")]
        if not lines:
            return
        labels = [f"{n}  ({'Y-Z' if d == 'X' else 'X-Z'} plane)" for n, d in lines]
        v = ask(self, "Set Elevation View", [("grid", "Elevation at grid line", labels[0], {"choices": labels})])
        if v:
            name, d = lines[labels.index(v["grid"])]
            self.viewport.set_view("yz" if d == "X" else "xz", grid=name)

    def view_refresh_all(self):
        for vp in self.views.viewports:
            vp.refresh()  # the windows not on screen when they are shown again

    # ------------------------------------------------------------- Define
    def define_shell_sections(self):
        self._define("shell_sections", "Slab / Wall Sections", dialogs.shell_section_factory)

    def define_groups(self):
        m = self.doc.model
        items = list(m.groups)
        name, ok = QtWidgets.QInputDialog.getItem(
            self, "Group Definitions", "Existing groups (type a new name to add it; "
            "the current selection is added):", items + [""], len(items), True)
        if ok and name.strip():
            C.assign_to_group(self.doc, name.strip(), *self._sel())
            self._info(f"Group {name.strip()}: {sum(len(x) for x in self._sel())} object(s) added")

    def define_mass_source(self):
        m = self.doc.model
        ms = next(iter(m.mass_sources.values()))
        pats = ", ".join(f"{k}={v:g}" for k, v in ms.load_patterns.items())
        v = ask(self, f"Mass Source {ms.name}", [
            ("-", "Mass source"),
            ("element", "Element self mass", ms.include_element_mass),
            ("added", "Additional mass", ms.include_added_mass),
            ("patterns", "Specified load patterns (name=multiplier, ...)", pats),
            ("-", "Mass options"),
            ("lateral", "Include lateral mass", ms.include_lateral),
            ("vertical", "Include vertical mass", ms.include_vertical),
            ("lump", "Lump lateral mass at story levels", ms.lump_at_stories)])
        if not v:
            return
        loads = {}
        for item in v["patterns"].split(","):
            if "=" in item:
                k, val = item.split("=", 1)
                if k.strip():
                    loads[k.strip()] = float(val)

        def mutate(model):
            s = model.mass_sources[ms.name]
            s.include_element_mass, s.include_added_mass = v["element"], v["added"]
            s.include_lateral, s.include_vertical, s.lump_at_stories = v["lateral"], v["vertical"], v["lump"]
            s.load_patterns = loads

        self._try("Mass Source", lambda: self.doc.edit(mutate))

    def assign_panel_loads(self):
        """Assign > Panel Loads: edit the selected panel (or every panel one by one)."""
        names = list(self.state.panels) or list(self.doc.model.panel_loads)
        if not names:
            self.state.status.emit("Draw a load panel first (Draw Load Panel)")
            return
        for n in names:
            self.edit_panel_load(n)

    # --------------------------------------------------------------- Load
    def load_command(self, tool: str, **settings):
        """Start a load command (with preset form values)."""
        for k, v in settings.items():
            setattr(self.state.settings, k, v)
        self.state.set_tool(tool)

    def delete_loads_on_selection(self):
        """Load > Delete Loads on Selection: the loads of one pattern (or all) on the
        selected joints, frames, areas (their area-load panels) and panels."""
        m = self.doc.model
        st = self.state
        pts, frs, ars, pls = set(st.points), set(st.frames), set(st.areas), set(st.panels)
        if not (pts or frs or ars or pls):
            self.state.status.emit("Select the objects whose loads are to be deleted")
            return
        pats = sorted({ld.pattern for n in pts for ld in m.points[n].loads} |
                      {ld.pattern for n in frs for ld in m.frames[n].loads} |
                      {c.pattern for p in m.panel_loads.values() if p.area in ars or p.name in pls for c in p.loads})
        if not pats:
            self.state.status.emit("The selection carries no loads")
            return
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Delete Loads", "Load pattern:", ["(all patterns)"] + pats,
                                                    0, False)
        if not ok:
            return
        keep = (lambda pat: False) if choice == "(all patterns)" else (lambda pat: pat != choice)

        def mutate(model):
            for n in pts:
                model.points[n].loads = [ld for ld in model.points[n].loads if keep(ld.pattern)]
            for n in frs:
                model.frames[n].loads = [ld for ld in model.frames[n].loads if keep(ld.pattern)]
            for p in list(model.panel_loads.values()):
                if p.area in ars or p.name in pls:
                    p.loads = [c for c in p.loads if keep(c.pattern)]
                    if not p.loads and p.area:  # an area load with nothing left goes away
                        del model.panel_loads[p.name]

        self._try("Delete loads", lambda: self.doc.edit(mutate))

    def load_report(self):
        """Load > Load Panel / Area Load Report: per panel and pattern, the load
        applied and the load the members (frames / shells) carry."""
        from ..mesh.panel_loads import panel_loads_to_members
        m = self.doc.model
        try:
            _, _, rep = panel_loads_to_members(m)
        except Exception as exc:
            return QtWidgets.QMessageBox.warning(self, "Load report", str(exc))
        u = m.units.shown("force")
        lines = [f"{'Panel':<8} {'On':<10} {'Distribution':<9} {'Pattern':<10} {'Applied':>12} {'Carried':>12}"]
        totals = {}
        for name, per in rep.items():
            p = m.panel_loads[name]
            for pat, (app, car) in per.items():
                lines.append(f"{name:<8} {(p.area or 'polygon'):<10} {p.distribution:<9} {pat:<10} "
                             f"{app:>12.4g} {car:>12.4g}" + ("   <- not all carried" if abs(app - car) > 1e-3 * max(abs(app), 1e-9) else ""))
                t = totals.setdefault(pat, [0.0, 0.0])
                t[0] += app
                t[1] += car
        lines.append("")
        for pat, (app, car) in totals.items():
            lines.append(f"Total {pat}: applied {app:.6g} {u}, carried {car:.6g} {u}")
        box = QtWidgets.QDialog(self)
        box.setWindowTitle("Load Panel / Area Load Report")
        lay = QtWidgets.QVBoxLayout(box)
        txt = QtWidgets.QPlainTextEdit("\n".join(lines) if rep else "No load panels or area loads in the model.")
        txt.setReadOnly(True)
        txt.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        lay.addWidget(txt)
        box.resize(760, 420)
        box.show()
        self._load_report_dialog = box

    # --------------------------------------------------------------- Draw
    def draw_snap_options(self):
        """Draw > Snap Options: the full form (stays open, Apply applies)."""
        from .snap_dialog import SnapOptionsDialog
        d = getattr(self, "_snap_dialog", None)
        if d is None or not d.isVisible():
            d = SnapOptionsDialog(self.state.snap, self)
            d.applied.connect(self.apply_snap_options)
            self._snap_dialog = d
        d.show()
        d.raise_()

    def apply_snap_options(self, options):
        self.state.set_snap_options(options)
        self._sync_snap_actions()
        if self.qsettings is not None:
            self.qsettings.setValue("draw/snap", options.to_json())

    def toggle_snap_only(self, on=None):
        o = self.state.snap
        o.snap_only = (not o.snap_only) if on is None else bool(on)
        self.apply_snap_options(o)

    def _sync_snap_actions(self):
        o = self.state.snap
        for key, act in getattr(self, "snap_actions", {}).items():
            act.blockSignals(True)
            act.setChecked(o.snap_only if key == "snap_only" else o.osnap if key == "osnap" else o.enabled(key))
            act.blockSignals(False)

    # ------------------------------------------------------------- Select
    def select_by_property(self, mode: str, what: str):
        m = self.doc.model
        if what == "material":
            names, fn = list(m.materials), C.objects_with_material
        elif what == "frame_section":
            names, fn = list(m.frame_sections), C.objects_with_frame_section
        else:
            t = "Slab" if what == "slab_section" else "Wall"
            names = [s.name for s in m.shell_sections.values() if s.type == t]
            fn = C.objects_with_shell_section
        if not names:
            return QtWidgets.QMessageBox.information(self, "Select", "Nothing of that kind is defined.")
        d = dialogs.MultiChoiceDialog(("Select" if mode == "add" else "Deselect") + " by property", names, self)
        if d.exec():
            f, a, p = set(), set(), set()
            for n in d.values():
                ff, aa, pp = fn(m, n)
                f |= set(ff)
                a |= set(aa)
                p |= set(pp)
            self._select(mode, f, a, p)

    def select_by_pier(self, mode: str, spandrel: bool):
        m = self.doc.model
        attr = "spandrel" if spandrel else "pier"
        names = sorted({getattr(o, attr) for o in list(m.frames.values()) + list(m.areas.values())
                        if getattr(o, attr)})
        if not names:
            return QtWidgets.QMessageBox.information(self, "Select", f"No {attr} labels are assigned.")
        d = dialogs.MultiChoiceDialog(f"{'Select' if mode == 'add' else 'Deselect'} {attr} labels", names, self)
        if d.exec():
            f, a = set(), set()
            for n in d.values():
                ff, aa, _ = C.objects_with_pier(m, n, spandrel)
                f |= set(ff)
                a |= set(aa)
            self._select(mode, f, a)

    def select_by_label(self, mode: str, kind: str, unique: bool = False):
        what = {"point": "joint", "frame": "frame", "area": "shell"}[kind]
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Select by " + ("unique name" if unique else "label"),
            f"{what.title()} {'unique names' if unique else 'labels'} (comma separated):")
        if ok and text.strip():
            self._select(mode, *C.objects_with_label(self.doc.model, kind, text))

    def select_by_group(self, mode: str):
        m = self.doc.model
        if not m.groups:
            return QtWidgets.QMessageBox.information(self, "Select", "No groups are defined (Define > Group Definitions).")
        d = dialogs.MultiChoiceDialog("Select groups", list(m.groups), self)
        if d.exec():
            f, a, p = set(), set(), set()
            for g in d.values():
                ff, aa, pp = C.objects_in_group(m, g)
                f |= set(ff)
                a |= set(aa)
                p |= set(pp)
            self._select(mode, f, a, p)

    def select_by_stories(self, mode: str):
        m = self.doc.model
        d = dialogs.MultiChoiceDialog("Select stories", [s.name for s in m.stories_bottom_up()][::-1], self)
        if d.exec():
            st = set(d.values())
            self._select(mode, [f.name for f in m.frames.values() if f.story in st],
                         [a.name for a in m.areas.values() if a.story in st],
                         [p.name for p in m.points.values() if p.story in st])

    # ------------------------------------------------------------- Assign
    def _first(self, kind: str):
        table = {"point": "points", "frame": "frames", "area": "areas"}[kind]
        names = {"point": self.state.points, "frame": self.state.frames, "area": self.state.areas}[kind]
        return getattr(self.doc.model, table)[sorted(names)[0]]

    def _names(self, kind: str) -> List[str]:
        return list({"point": self.state.points, "frame": self.state.frames, "area": self.state.areas}[kind])

    def assign_form(self, key: str, title: str, picks, apply, fields=None, widget=None, note=None):
        """Start an Assign command (see tools.AssignFormTool): its form opens at once in a
        window of its own, with the values of the objects selected before; pick / unpick
        objects in the model; Apply assigns and keeps the window, OK / Enter assigns and
        closes, Cancel closes without assigning."""
        from .tools import AssignFormTool, AssignSpec
        if getattr(self, "_pending_assign", None) is not None:
            self.cancel_pending_assign()
        spec = AssignSpec(key, title, tuple(picks), apply, fields=fields, widget=widget, note=note)
        self.state.start_tool(lambda: AssignFormTool(self.doc, self.state.settings, spec))

    # the unit of an object value (shown with the number to type)
    _ATTR_QUANTITY = {"mass_per_length": "mass_per_length", "mass_per_area": "mass_per_area",
                      "spring": "weight_per_volume"}

    def _assign_number(self, kind: str, attr: str, title: str, label: str, of: str = ""):
        q = self._ATTR_QUANTITY.get(attr)
        if q:
            label = f"{label.split(' (')[0]} [{self.doc.model.units.shown(q)}]"
        self.assign_form(f"{kind}.{attr}.{of}", title, (kind,),
                         lambda names, v: C.assign(self.doc, kind, self._of_kind(names, of), **{attr: v["v"]}),
                         fields=lambda o: [("v", label, float(getattr(o, attr) or 0.0))])

    def _assign_angle(self, kind: str, of: str = ""):
        what = "Frame" if kind == "frame" else (of or "Shell")
        self.assign_form(f"{kind}.angle.{of}", what + " Local Axes", (kind,),
                         lambda names, v: C.assign(self.doc, kind, self._of_kind(names, of), angle=v["angle"]),
                         fields=lambda o: [("angle", "Rotate local axes about axis 1 / 3 by [deg]",
                                            float(o.angle))])

    def _assign_label(self, kind: str, attr: str, of: str = ""):
        self.assign_form(f"{kind}.{attr}.{of}", f"{attr.title()} Label", (kind,),
                         lambda names, v: C.assign(self.doc, kind, self._of_kind(names, of),
                                                   **{attr: v["label"].strip() or None}),
                         fields=lambda o: [("label", f"{attr.title()} label (blank = none)",
                                            str(getattr(o, attr) or ""))])

    def assign_point_springs(self):
        u = self.doc.model.units
        ku, kr = u.label("translational_stiffness"), u.label("rotational_stiffness")

        def fields(p):
            out = [("-", f"Translational springs ({ku})")]
            vals = tuple(p.springs) + tuple(p.rotational_springs)
            for i, lab in enumerate(DOF_LABELS):
                if i == 3:
                    out.append(("-", f"Rotational springs ({kr})"))
                out.append((f"k{i}", f"{lab} stiffness  [{ku if i < 3 else kr}]", float(vals[i])))
            return out

        def apply(names, v):
            k = [v[f"k{i}"] for i in range(6)]
            C.assign(self.doc, "point", names, springs=tuple(k[:3]), rotational_springs=tuple(k[3:]))
        self.assign_form("point.springs", "Joint Springs", ("point",), apply, fields=fields)

    def assign_point_mass(self):
        mu = self.doc.model.units.shown("mass")
        mi = f"{mu}·{self.doc.model.units.shown('length')}²"
        self.assign_form("point.mass", "Joint Additional Mass", ("point",),
                         lambda names, v: C.assign(self.doc, "point", names, mass=(v["ux"], v["uy"], v["uz"]),
                                                   rotational_mass=(v["rx"], v["ry"], v["rz"])),
                         fields=lambda p: [
                             ("ux", f"Mass UX  [{mu}]", float(p.mass[0])), ("uy", f"Mass UY  [{mu}]", float(p.mass[1])),
                             ("uz", f"Mass UZ  [{mu}]", float(p.mass[2])),
                             ("rx", f"Mass moment RX  [{mi}]", float(p.rotational_mass[0])),
                             ("ry", f"Mass moment RY  [{mi}]", float(p.rotational_mass[1])),
                             ("rz", f"Mass moment RZ  [{mi}]", float(p.rotational_mass[2]))])

    def define_rebar_sizes(self):
        """Define > Reinforcing Bar Sizes: the project's bar sizes, with the
        standards' catalogs and CSV import / export."""
        from .rebar_manager import RebarSizesDialog
        RebarSizesDialog(self.doc, self).exec()

    def define_frame_sections(self):
        """Define > Frame Sections: the section manager (Add / Copy / Modify /
        Delete, drawing and properties of the selected section)."""
        from .section_manager import FrameSectionsDialog
        FrameSectionsDialog(self.doc, self).exec()

    def assign_frame_modifiers(self):
        from ..model.properties import FrameModifiers
        names_ = ("A", "As2", "As3", "J", "I22", "I33", "mass", "weight")
        labels = ("Cross-section (axial) area", "Shear area in 2 direction", "Shear area in 3 direction",
                  "Torsional constant", "Moment of inertia about 2 axis", "Moment of inertia about 3 axis",
                  "Mass", "Weight")

        def fields(f):
            cur = f.modifiers or FrameModifiers()
            return [(n, f"{l} [x section value]", float(getattr(cur, n))) for n, l in zip(names_, labels)]
        self.assign_form("frame.modifiers", "Frame Property/Stiffness Modifiers", ("frame",),
                         lambda names, v: C.assign(self.doc, "frame", names, modifiers=FrameModifiers(**v)),
                         fields=fields)

    def assign_shell_modifiers(self, kind: str = "Shell"):
        from ..model.properties import ShellModifiers
        keys = ("f11", "f22", "f12", "m11", "m22", "m12", "v13", "v23", "mass", "weight")

        def fields(a):
            cur = a.modifiers or ShellModifiers()
            return [(n, {"mass": "Mass", "weight": "Weight"}.get(
                n, f"{n[0].upper()}{n[1:]} {'membrane' if n[0] == 'f' else 'bending' if n[0] == 'm' else 'shear'}")
                + " [x section value]", float(getattr(cur, n))) for n in keys]
        self.assign_form(f"{kind.lower()}.modifiers", f"{kind} Stiffness Modifiers", ("area",),
                         lambda names, v: C.assign(self.doc, "area", self._of_kind(names, kind),
                                                   modifiers=ShellModifiers(**v)),
                         fields=fields)

    def _of_kind(self, names, kind: str):
        """The picked areas a Slab / Wall command changes: walls for Wall, the others
        (floors, ramps) for Slab; every one for the plain Shell commands."""
        m = self.doc.model
        if kind not in ("Slab", "Wall", "Floor"):
            return list(names)
        wall = kind == "Wall"
        mine = [n for n in names if n in m.areas and (m.areas[n].type == "Wall") == wall]
        if not mine:
            self.state.status.emit(f"None of the picked shells is a {'wall' if wall else 'floor'} "
                                   f"(Assign > {'Wall' if wall else 'Slab'}): nothing changed")
        return mine

    def assign_releases_full(self):
        labels = ("Axial load (P)", "Shear force 2 (V2)", "Shear force 3 (V3)", "Torsion (T)",
                  "Moment 22 (M2)", "Moment 33 (M3)")

        def fields(f):
            out = [("-", "Release  (Start | End)")]
            for k, lab in enumerate(labels):
                out.append((f"i{k}", f"{lab} - Start", bool(f.releases_i[k])))
                out.append((f"j{k}", f"{lab} - End", bool(f.releases_j[k])))
            return out

        def apply(names, v):
            ri = tuple(v[f"i{k}"] for k in range(6))
            rj = tuple(v[f"j{k}"] for k in range(6))
            C.assign(self.doc, "frame", names, releases_i=ri, releases_j=rj)
        self.assign_form("frame.releases", "Frame Releases", ("frame",), apply, fields=fields)

    def assign_end_offsets(self):
        m = self.doc.model

        def fields(f):
            default = f.auto_end_offsets is None and f.rigid_factor is None
            return [("default", "Use the model defaults (Analyze > Analysis Model Defaults)", default),
                    ("auto", "Automatic from connectivity", m.frame_auto_end_offsets(f)),
                    ("i", "End-I length offset", float(f.end_offset_i), {"item": "coordinates"}),
                    ("j", "End-J length offset", float(f.end_offset_j), {"item": "coordinates"}),
                    ("rz", "Rigid-zone factor (0 = flexible ... 1 = fully rigid)", m.frame_rigid_factor(f))]

        def apply(names, v):
            if v["default"]:
                C.assign(self.doc, "frame", names, auto_end_offsets=None, rigid_factor=None,
                         end_offset_i=v["i"], end_offset_j=v["j"])
            else:
                C.assign(self.doc, "frame", names, auto_end_offsets=v["auto"], end_offset_i=v["i"],
                         end_offset_j=v["j"], rigid_factor=max(0.0, min(1.0, v["rz"])))
        self.assign_form("frame.end_offsets", "Frame End Length Offsets", ("frame",), apply, fields=fields,
                         note="Rigid-zone factor: the fraction of each end offset taken as rigid for bending and "
                              "shear (recommended: 0.5 for concrete frames; steel 0, or 1.0 only together with "
                              "panel zones).")

    def analyze_model_defaults(self):
        """Analyze > Analysis Model Defaults: what frames and slabs get unless the
        object says otherwise, and the mesh refinement check."""
        o = self.doc.model.options
        v = ask(self, "Analysis Model Defaults", [
            ("auto", "Frames: end offsets automatic from connectivity", bool(o.frame_auto_end_offsets)),
            ("rc", "Frames: rigid-zone factor, concrete", float(o.rigid_zone_factor_concrete)),
            ("rs", "Frames: rigid-zone factor, steel", float(o.rigid_zone_factor_steel)),
            ("ro", "Frames: rigid-zone factor, other materials", float(o.rigid_zone_factor_other)),
            ("slab", "Slabs: rigid zone over columns (slab inside the column section moves with it)",
             bool(o.slab_column_rigid_zone)),
            ("tol", "Check Mesh: allowed estimated error of the largest slab / wall deflection (%)",
             100.0 * float(o.mesh_refine_tolerance))],
            note="Used for every frame / slab that does not set its own value (Assign > Frame > End Length "
                 "Offsets sets one per frame).  Recommended: concrete 0.5; steel 0, or 1.0 only together "
                 "with panel zones; rigid zones over columns on (deflections and slab reinforcement lower, "
                 "punching ratios higher).  Imported ETABS / SAP2000 / SAFE models start with no automatic "
                 "offsets and factor 0, as in the original file.")
        if not v:
            return
        clip = lambda x: max(0.0, min(1.0, float(x)))

        def apply(mm):
            oo = mm.options
            oo.frame_auto_end_offsets = bool(v["auto"])
            oo.rigid_zone_factor_concrete = clip(v["rc"])
            oo.rigid_zone_factor_steel = clip(v["rs"])
            oo.rigid_zone_factor_other = clip(v["ro"])
            oo.slab_column_rigid_zone = bool(v["slab"])
            oo.mesh_refine_tolerance = max(0.001, float(v["tol"]) / 100.0)
        self.doc.edit(apply)

    def assign_insertion_point(self):
        """Assign > Frame > Insertion Point: the cardinal point on a picture of the section,
        whether its offset goes into the analysis, and the joint offsets."""
        from .insertion_dialog import InsertionPointWidget, frame_fields, frame_state, offset_fields, offset_state
        m = self.doc.model

        def widget(f):
            sec = f.section if f.section in m.frame_sections else next(iter(m.frame_sections), None)
            cp, transform = frame_state(f, m) if f.name else (10, False)
            local, shift, oi, oj = offset_state(f)
            axes = None
            if f.name:
                import numpy as np
                from ..mesh.geometry import frame_axes
                try:
                    axes = frame_axes(np.array(m.points[f.point_i].xyz, float),
                                      np.array(m.points[f.point_j].xyz, float), f.angle)
                except (ValueError, KeyError):
                    axes = None
            w = InsertionPointWidget(m, sec, cp, transform, oi, oj, shift=shift, local=local, axes=axes)
            return w, w.values

        def apply(names, v):
            fields = {**offset_fields(v), **frame_fields(v["cp"], v["transform"])}
            if not v.get("move"):
                C.assign(self.doc, "frame", names, **fields)
                return
            moved, links, notes = C.move_beams_to_axis(self.doc, names, None, assign=fields)
            self._info(f"Insertion point assigned; {len(moved)} beam(s) moved to their real axis with "
                       f"{len(links)} rigid link(s)")
            if notes and not moved:
                QtWidgets.QMessageBox.information(self, "Move Beams to Real Axis",
                                                  "The insertion point was assigned; no beam was moved:\n\n"
                                                  + "\n".join(notes[:20]))
        self.assign_form("frame.insertion", "Frame Insertion Point", ("frame",), apply, widget=widget)

    def assign_output_stations(self):
        self.assign_form("frame.stations", "Frame Output Stations", ("frame",),
                         lambda names, v: C.assign(self.doc, "frame", names, max_station_spacing=v["spacing"],
                                                   min_stations=v["n"]),
                         fields=lambda f: [
                             ("spacing", "Max station spacing (blank = by number)", f.max_station_spacing,
                              {"optional": True, "item": "coordinates"}),
                             ("n", "Min number of stations", int(f.min_stations), {"min": 2, "max": 100})])

    def assign_tc_limits(self):
        self.assign_form("frame.tc_limits", "Frame Tension/Compression Limits", ("frame",),
                         lambda names, v: C.assign(self.doc, "frame", names, tension_limit=v["t"],
                                                   compression_limit=v["c"]),
                         fields=lambda f: [
                             ("t", "Tension limit (blank = none)", f.tension_limit, {"optional": True, "item": "force"}),
                             ("c", "Compression limit (blank = none)", f.compression_limit,
                              {"optional": True, "item": "force"})])

    def assign_frame_mesh(self):
        self.assign_form("frame.mesh", "Frame Auto Mesh Options", ("frame",),
                         lambda names, v: C.assign(self.doc, "frame", names, auto_mesh=v["auto"],
                                                   mesh_at_intersections=v["inter"]),
                         fields=lambda f: [
                             ("auto", "Auto mesh frame at intermediate joints / intersections", bool(f.auto_mesh)),
                             ("inter", "Mesh at intersections with other frames and area edges",
                              bool(f.mesh_at_intersections))])

    def assign_frame_floor_mesh(self):
        self.assign_form("frame.floor_mesh", "Frame Floor Meshing Options", ("frame",),
                         lambda names, v: C.assign(self.doc, "frame", names, mesh_at_floor=v["floor"]),
                         fields=lambda f: [("floor", "Include frame in floor meshing (split at mesh nodes)",
                                            bool(f.mesh_at_floor))])

    def assign_shell_section_of(self, kind: str):
        if not self._need("area"):
            return
        secs = [s.name for s in self.doc.model.shell_sections.values() if s.type == kind] or \
            list(self.doc.model.shell_sections)
        if not secs:
            return QtWidgets.QMessageBox.information(self, "Assign", "Define a shell section first.")
        v = ask(self, f"Assign {kind} Section", [("sec", "Section", secs[0], {"choices": secs})])
        if v:
            names = [n for n in self._names("area")
                     if (self.doc.model.areas[n].type == "Wall") == (kind == "Wall")] or self._names("area")
            C.assign(self.doc, "area", names, section=v["sec"])

    def assign_openings(self, of: str = ""):
        self.assign_form(f"shell.openings.{of}", f"{of or 'Shell'} Openings", ("area",),
                         lambda names, v: C.assign(self.doc, "area", self._of_kind(names, of), type="Opening",
                                                   section=None),
                         fields=lambda a: [],
                         note="The picked shells become openings (holes in the floors / walls they lie in).")

    def assign_area_mesh(self, kind: str):
        """Assign > Slab / Wall > Auto Mesh Options, in the box: one choice for the picked
        areas - auto mesh, manual maximum mesh size, fine mesh (once found) or no mesh -
        and the allowed error of Find Fine Mesh."""
        from .mesh_dialog import AreaMeshDialog, area_mode
        m = self.doc.model

        def widget(a):
            mode = area_mode(a) if a.name else "auto"
            if mode == "auto" and m.options.use_fine_mesh and m.options.fine_mesh_factor:
                mode = "fine"  # (models saved with the earlier whole-model fine mesh)
            dlg = AreaMeshDialog(kind, m, mode, a.max_mesh_size, self.check_mesh, self)
            bb = dlg.findChild(QtWidgets.QDialogButtonBox)
            if bb is not None:
                bb.hide()
            self._mesh_dialog = dlg  # Find Fine Mesh refreshes it when done

            def values():
                v = dlg.values()  # (ValueError: shown, nothing assigned)
                if dlg.apply_tolerance():
                    self.doc.dirty = True
                return v
            return dlg, values

        def apply(names, v):
            mm = self.doc.model
            if mm.options.use_fine_mesh:
                # the earlier whole-model fine mesh becomes the fine mesh of each auto-meshed area
                rest = [n for n, ar in mm.areas.items() if ar.auto_mesh and not ar.max_mesh_size and not ar.fine_mesh]
                if rest:
                    C.assign(self.doc, "area", rest, fine_mesh=True)
                self.set_fine_mesh(False)
            C.assign(self.doc, "area", self._of_kind(names, kind), auto_mesh=v["auto"],
                     max_mesh_size=v["size"], fine_mesh=v["fine"])
            self._sync_fine_mesh()
        self.assign_form(f"{kind.lower()}.mesh", f"{kind} Auto Mesh Options", ("area",), apply, widget=widget)

    def assign_frame_point_load(self):
        if not self._need("frame"):
            return
        m = self.doc.model
        if not m.load_patterns:
            return QtWidgets.QMessageBox.information(self, "No load pattern", "Define a load pattern first.")
        v = ask(self, "Assign Frame Point Loads", [
            ("pattern", "Load pattern", next(iter(m.load_patterns)), {"choices": list(m.load_patterns)}),
            ("type", "Load type", "Force", {"choices": ["Force", "Moment"]}),
            ("dir", "Direction", "Gravity", {"choices": ["Gravity", "X", "Y", "Z", "Local 1", "Local 2", "Local 3"]}),
            ("value", "Load", 0.0), ("rel", "Relative distance from End-I", 0.5),
            ("replace", "Replace existing point loads in this pattern", True)])
        if not v:
            return
        names = self._names("frame")

        def mutate(model):
            for n in names:
                f = model.frames[n]
                if v["replace"]:
                    f.loads = [ld for ld in f.loads if not (ld.pattern == v["pattern"] and ld.kind == "point")]
                model.add_frame_point_load(n, v["pattern"], v["value"], rel=max(0.0, min(1.0, v["rel"])),
                                           direction=v["dir"], load_type=v["type"])

        self._try("Frame point load", lambda: self.doc.edit(mutate))

    def assign_group(self):
        if not self._need_any():
            return
        m = self.doc.model
        name, ok = QtWidgets.QInputDialog.getItem(self, "Assign Objects to Group", "Group:", list(m.groups) or ["GROUP1"],
                                                  0, True)
        if ok and name.strip():
            C.assign_to_group(self.doc, name.strip(), *self._sel())

    def clear_assign_display(self):
        self.load_box.setCurrentIndex(0)
        self.views.apply_display(self._with(supports=True, springs=False, releases=False, local_axes=False,
                                            section_names=False), self.viewport)

    def copy_assigns(self):
        st = self.state
        kinds = [(k, n) for k, names in (("frame", st.frames), ("area", st.areas), ("point", st.points)) for n in names]
        if len(kinds) != 1:
            return QtWidgets.QMessageBox.information(self, "Copy Assigns", "Select exactly one joint, frame or shell.")
        kind, name = kinds[0]
        self.assign_clipboard = C.copy_assigns(self.doc.model, kind, name)
        self._info(f"Copied the assignments of {kind} {name}")

    def paste_assigns(self):
        clip = self.assign_clipboard
        if clip is None:
            return QtWidgets.QMessageBox.information(self, "Paste Assigns", "Use Assign > Copy Assigns first.")
        names = self._names(clip.kind)
        if not names:
            return self._need(clip.kind)
        v = ask(self, "Paste Assigns", [("loads", "Include loads", True)])
        if v:
            C.paste_assigns(self.doc, clip, names, loads=v["loads"])

    # ------------------------------------------------------------ Analyze
    def analyze_dof(self):
        from .dof_dialog import DofDialog
        d = DofDialog(self.doc.model.options.active_dofs, self)
        if d.exec():
            dofs = d.value()
            self.doc.edit(lambda m: setattr(m.options, "active_dofs", dofs))

    def analyze_mesh(self, which: str):
        o = self.doc.model.options
        attr = f"{which}_mesh_max_size"
        v = ask(self, f"Automatic Mesh Settings for {which.title()}s",
                [("size", "Approximate maximum mesh size (blank = automatic from the geometry, refinement checked)", getattr(o, attr), {"optional": True})])
        if v:
            self.doc.edit(lambda m: setattr(m.options, attr, v["size"]))

    # ------------------------------------------------------------ Display
    def _cases(self) -> List[str]:
        return [self.case_box.itemText(i) for i in range(self.case_box.count()) if self.case_box.itemText(i)]

    def _need_results(self) -> bool:
        if self.doc.results is None or not self._cases():
            QtWidgets.QMessageBox.information(self, "No results", "Run the analysis first (Analyze > Run Analysis, F5).")
            return False
        return True

    def _pick_case(self, title: str, extra=()) -> Optional[dict]:
        cases = self._cases()
        cur = self.case_box.currentText() or cases[0]
        return ask(self, title, [("case", "Load case / combination", cur, {"choices": cases})] + list(extra))

    def display_undeformed(self):
        for a in (self.act_deformed, self.act_reactions):
            a.setChecked(False)
        self.mode_box.setValue(0)
        self.diagram_box.setCurrentText("None")
        self._display_changed()

    def display_loads(self, kind: str):
        pats = list(self.doc.model.load_patterns)
        if not pats:
            return QtWidgets.QMessageBox.information(self, "Load Assigns", "No load patterns are defined.")
        v = ask(self, f"Show {'Joint' if kind == 'point' else kind.title()} Loads",
                [("pattern", "Load pattern", pats[0], {"choices": pats})])
        if v:
            self.load_box.setCurrentText(v["pattern"])

    def display_deformed(self):
        if not self._need_results():
            return
        v = self._pick_case("Deformed Shape")
        if v:
            self.case_box.setCurrentText(v["case"])
            self.mode_box.setValue(0)
            self.diagram_box.setCurrentText("None")
            self.act_deformed.setChecked(True)
            self._display_changed()

    def display_mode_shape(self):
        if not self._need_results():
            return
        v = self._pick_case("Mode Shape", [("mode", "Mode number", max(1, self.mode_box.value()), {"min": 1, "max": 999})])
        if v:
            self.case_box.setCurrentText(v["case"])
            self.mode_box.setValue(v["mode"])

    def display_reactions(self):
        if not self._need_results():
            return
        v = self._pick_case("Support/Spring Reactions")
        if v:
            self.case_box.setCurrentText(v["case"])
            self.act_reactions.setChecked(True)
            self._display_changed()

    def display_forces(self, kind: str):
        if not self._need_results():
            return
        comps = ["P", "V2", "V3", "T", "M2", "M3"] if kind == "frame" else \
            ["F11", "F22", "F12", "M11", "M22", "M12", "V13", "V23"]
        cur = self.diagram_box.currentText()
        v = self._pick_case("Frame Forces" if kind == "frame" else "Shell Stresses/Forces",
                            [("comp", "Component", cur if cur in comps else comps[-1], {"choices": comps})])
        if v:
            self.case_box.setCurrentText(v["case"])
            self.mode_box.setValue(0)
            self.diagram_box.setCurrentText(v["comp"])

    # ------------------------------------------------------------ Options
    def options_units(self):
        """Options > Display Units (Ctrl+U), also the units button > Show Units Form."""
        from .units_ui import UnitsDialog
        d = UnitsDialog(self.doc.model.units, self, display=self.doc.model.display_units)
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.change_units(d.units(), d.display())

    def set_display_units(self, display: dict, units=None):
        """Units button > Display unit sets (Thai legacy ...): the item units, and the
        consistent units of the set when it has some."""
        self.change_units(units or self.doc.model.units, display)

    def change_units(self, units, display=None):
        """Express the model (and what is being drawn / snapped) in other units, and /
        or show the values item by item in other units (``display``)."""
        from ..units import UnitSystem
        units = units if isinstance(units, UnitSystem) else UnitSystem.parse(units)
        old = self.doc.model.units
        if units == old and (display is None or dict(display) == self.doc.model.display_units):
            return
        try:  # the drawing / snap values follow in _on_doc_change (_follow_model_units)
            self.doc.change_units(units, display)
        except (ModelError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, "Units", str(exc))
            return
        btn = getattr(self, "units_button", None)
        if btn is not None:
            btn.refresh()
        self.logger.log(f"Units: {old.text} -> {units.text} (all values converted; results kept)")
        self.statusBar().showMessage(f"Units {units.text}: every value converted, results kept - "
                                     f"Edit > Undo goes back", 8000)

    def _convert_session_values(self, old, new):
        """Values typed in the Properties of Object form, the values each tool keeps,
        the snap settings and a command in progress are in the model's units:
        convert them too, so the next object drawn is the same (ViewState.convert_units)."""
        self.state.convert_units(old, new)

    def options_tolerances(self):
        o = self.doc.model.options
        v = ask(self, "Tolerances", [("tol", "Auto merge tolerance", float(o.merge_tolerance))])
        if v and v["tol"] > 0:
            self.doc.edit(lambda m: setattr(m.options, "merge_tolerance", v["tol"]))

    def edit_colors_tab(self, index: int):
        d = self.colors_dialog()
        tabs = d.findChild(QtWidgets.QTabWidget)
        if tabs is not None and index < tabs.count():
            tabs.setCurrentIndex(index)
        d.exec()

    def reset_toolbars(self):
        self.show_all_toolbars()

    def show_shortcuts(self):
        rows = []
        for path, label, cid, sc in iter_commands():
            a = self.menu_actions.get(cid)
            key = a.shortcut().toString() if a is not None and not a.shortcut().isEmpty() else (sc or "")
            if key:
                rows.append((" > ".join(path + (label,)), key))
        for k, a in self.tool_actions.items():
            if not a.shortcut().isEmpty():
                rows.append(("Draw tool: " + a.text(), a.shortcut().toString()))
        rows += [("Mouse wheel", "Zoom at the cursor"), ("Middle-drag", "Pan"),
                 ("Right-drag / Shift + middle-drag / Shift + left-drag", "Rotate 3D view"),
                 ("Double-click the wheel", "Restore full view"), ("Arrow keys", "Rotate 3D (Ctrl: pan)"),
                 ("Esc / right click", "End the current draw command")]
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Default Keyboard Shortcuts")
        lay = QtWidgets.QVBoxLayout(d)
        t = QtWidgets.QTableWidget(len(rows), 2)
        t.setHorizontalHeaderLabels(["Command", "Shortcut"])
        for r, (c, k) in enumerate(rows):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(c))
            t.setItem(r, 1, QtWidgets.QTableWidgetItem(k))
        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(t)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(d.reject)
        lay.addWidget(bb)
        d.resize(640, 640)
        self._shortcut_rows = rows
        d.exec()

    # --------------------------------------------------------------- Help
    def show_help(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        readme = os.path.join(root, "README.md")
        if os.path.exists(readme):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(readme))

    def show_about(self):
        QtWidgets.QMessageBox.about(self, "About Structural Studio",
                                    f"Structural Studio {__version__}\nObject-based building analysis program.\n"
                                    f"Entries marked planned are not available yet.")
