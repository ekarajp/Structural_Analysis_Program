# Structural Studio (new core) — `studio/`

Object-based building analysis engine, Python-first and compiled-kernel-ready.
This folder is the new code base; the old `source/StructuralStudio` (WPF + worker) is
kept only for reference and its EtabsBridge capture tool.

## Run on Windows (E:\REA ETABS)

```powershell
. 'E:\REA ETABS\scripts\Environment.ps1'
cd 'E:\REA ETABS\studio'
& 'E:\REA ETABS\tools\python\Scripts\python.exe' -m pip install -e .[dev]
& 'E:\REA ETABS\tools\python\Scripts\python.exe' -m pytest -q          # 18 tests
& 'E:\REA ETABS\tools\python\Scripts\python.exe' -m structstudio.cli analyze tests\verification\data\Example01.e2k --cases I II --frames "B1@1ST"
```

(needs numpy >= 1.26, scipy >= 1.11, pytest; Python >= 3.11)

## Desktop app (Phase 2)

```powershell
E:\REA ETABS\tools\python\Scripts\python.exe -m pip install -e .[app]
E:\REA ETABS\tools\python\Scripts\python.exe -m structstudio.app examples\office.ssj.json
```

PySide6 + VTK, ETABS / SAP2000-style workflow:

* **Grid system switch** (top-left, like Affinity's personas) – *X · Y · Story* or
  *X · Y · Z*; switch back and forth at any time (also *Edit / Define > Grid system*,
  undoable, saved in the project).  In X · Y · Z mode the Z grid lines are the levels (plans are
  "X-Y Plane @ Z"), in X · Y · Story mode the stories; *Edit > Edit grid lines* (Ctrl+G) has X / Y tabs
  plus a Story or Z Grid tab.  Analysis results are identical in both modes.
* **Menus laid out like ETABS 21** – File, Edit, View, Define, Draw, Select, Assign, Analyze, Display,
  Design, Detailing, Options, Tools, Help with the ETABS command names, order and default
  shortcuts (F5 run, F4 / F6 undeformed / deformed, F7–F9 reactions / frame / shell forces, Ctrl+R replicate,
  Ctrl+M move, Ctrl+W display options, Ctrl+K / Ctrl+J / Ctrl+Q invert / previous / clear selection,
  Ctrl+T tables, Ctrl+U units ...; *Options > Default Keyboard Shortcuts* lists them).  Working now:
  cut / copy / paste, replicate (linear, radial, mirror, story), extrude joints→frames and
  frames→shells, merge / align joints, divide / join / reverse frames, divide shells, grid lines at
  joints, auto relabel; set plan / elevation view, measure line / area / angle, show selected only /
  hide / invert / restore / show all objects; select and deselect by type, material, section, label,
  pier / spandrel, group, story; joint springs and mass, frame modifiers, releases, end offsets,
  insertion point, local axes, output stations, T/C limits, mesh options, shell modifiers, openings,
  area springs, groups, copy / paste assigns; active DOF, mesh settings, units, tolerances, mass source.
  Commands not built yet (design, detailing, links, tendons, hinges ...) are shown greyed out as *planned*.
* **Zoom / Pan / Rotate** like ETABS – mouse wheel zooms at the cursor, **middle-drag pans**,
  **right-drag** (or Shift + middle / Shift + left drag) **rotates the 3D view** about the vertical
  axis (Z stays up; plan and elevation windows pan), double-click the wheel = full view.  Keys:
  arrows rotate a 3D window (Ctrl + arrows pan), + / − zoom, Home restores the view, F fits.
  Toolbar / *View* menu: Rubber Band Zoom, Restore Full View, Zoom In / Out, Pan, Rotate 3D View
  (left-drag modes, Esc returns to Select), Default 3D View.
* **Grid** – *View > Show Grid lines* (Ctrl+Shift+G, toolbar), grid bubbles, vertical grid
  (column) lines in 3D, grid at the base / every / top level in 3D.
* **Extruded view** (Ctrl+E, toolbar; off by default – the line view) – members with their real cross-sections (rectangle, circle,
  I / wide flange, box / tube, pipe, channel, angle, tee; general sections as the equivalent
  rectangle) oriented by the member local axes and placed by the insertion (cardinal) point; slabs
  and walls with their thickness (slab top at the level, walls centred).
* **Set Display Options** (Ctrl+W, the eye button on the toolbar and in every window header) – per window,
  like ETABS: object visibility (joints, beams, columns, braces, null lines, floors, walls, ramps,
  openings), assignments (restraints, springs, releases, frame local axes), labels (joints, frames,
  areas, section names), extrusion, fill / edges, grid lines and bubbles, *Apply to all windows*.
  3D windows draw the grid at the base level (or every / top level) and shade the plane of the story
  shown in the plan window (switchable).
* **Dark / Light mode** (light by default) – *Options > Theme* or the sun/moon button; each mode has its own default
  colours for every object and output (beams, columns, braces, floors, walls, supports, grid,
  selection, deformed shape, diagrams, loads ...).  *Options > Graphics Colors* changes any colour per
  mode (Reset returns to the defaults) and the colour of each section / material;
  *Color objects by* object type, section or material.  Saved between sessions.
* **Graphics sizes** – *Options > Graphics Sizes* (the *Sizes* tab of Graphics Colors): support /
  spring symbols, joint dots, beam / column / null line widths, area edges, grid and deformed-shape
  lines, load arrows and the text size of load values and labels.  The *Display* tab also sets the
  transparency of floors, walls, grid lines, load diagrams, result diagrams and the label boxes.
  Saved between sessions.
* **Back to the defaults** – every row has a *Default* button (enabled once the row is changed);
  *Defaults for this tab*; *All defaults (light + dark)*; and *Options > Restore Graphics Defaults*.
  Light and dark mode keep separate colours; sizes and transparencies are shared.
* **ViewCube** – 3D windows have a CAD ViewCube (top right): click a face (Top, Front, Right ...),
  an edge or a corner (isometric) and the view turns smoothly to it; drag the cube to orbit.
* **CAD modify commands** (*Edit* menu, *Modify* fly-out of the left palette, Revit / AutoCAD
  two-letter shortcuts typed in a window): **Move** `MV`, **Copy** `CO` (multiple), **Rotate** `RO`,
  **Mirror** `MM`, **Offset** `OF`, **Trim** `TR`, **Extend** `EX`.  With nothing selected the
  command first asks for objects (click / window / crossing, then Enter or right-click).  Base
  point and destination use all snaps; moved objects detach from unselected ones unless
  *Stretch connected objects* is on, and joints landing on existing joints are merged.
  Column = `CL`, Opening = `OP` (Revit), other draw keys unchanged.
* **Edit > Edit Frames** – Divide, Join, Reverse Connectivity, Modify Frame Type.
  **Edit > Edit Shells** – Divide Shells, **Merge Shells** (coplanar shells with the same section
  and type that share edges become one; T-junctions handled, mid-edge joints removed unless
  something uses them; holes, corner-only contacts and overlaps are refused), **Expand/Shrink
  Shells** (every edge offset by a distance; neighbours unchanged), **Divide Walls for Openings**
  (the wall - or floor - is cut along the opening corners into rectangles that share whole edges,
  the pieces inside openings are left out; selecting an opening finds its wall), Reverse Wall
  Local 3 Axis.  Non-convex rectilinear shells (L / T / U floors, e.g. after Merge Shells) are meshed as
  the rectangles of their corner grid (conforming, matches the separate pieces exactly).
* **Floor / wall meshing around what stands on them** (ETABS "mesh at beams and other meshing
  lines", SAFE meshing at columns / point / line loads).  Orthogonal floors and walls are cut along
  lines through every corner, opening, beam / null line (line loads), wall edge, planted column,
  loaded joint, support and spring, then subdivided to the maximum element size; all floors share
  their plan lines (all walls facing one way share theirs), so neighbouring panels and slab-wall
  joints always match (no hanging nodes).  Joints and skew lines not on a line get the nearest
  mesh node moved onto them (corner angles kept within 30-150 degrees; panel edges never move).
  Bare joints with nothing on them are ignored (as SAFE).  Checked: an off-grid point load on a
  simply supported plate converges to the Navier solution (3.9 % at 1 m mesh, 0.42 % at 0.25 m,
  0.13 % at 0.125 m); point, line and planted-column loads all reach the supports exactly; the
  SAFE comparison is unchanged.  *Display > Show Analysis Mesh* draws the mesh (red dots: where
  members and joints are tied into it).
* **Model Explorer** (left, as ETABS): *Structure Layout* (stories with elevations - click one to show
  it in the active plan window - and grid lines), *Structural Objects* (joints; frames as Columns /
  Beams / Braces; shells as Floors / Walls / Ramps / Openings; load panels - each listed by story,
  top story first, with its section), *Groups*, *Properties*, *Loads*.  Click an object: it is
  selected, plan windows go to its story, it flashes magenta in every window that shows it and the
  status bar says where it is (`Beam B68 · B30×60 · Story3 · B/2 to C/2 · L = 6.000 m`).
  Double-click: zoom onto it.  Click a heading (e.g. *Beams > Story3*): select all of them.  The
  find box filters by name or section.
* **Deflection view** (*Display > Deformed Shape*, or the Deformed button) – the analysis mesh is
  drawn deformed and coloured by the displacement with a colour scale: slabs sag between their
  beams, beams and columns bend with the joint rotations (cubic shape; straight at released ends)
  and are drawn as solid bars; the undeformed model stays as a faint shadow.  Mode shapes the same.
* **Openings** (`OP`, *Draw Openings*) – a 2-click rectangle (or click + type `width,height`) in
  plan (shafts, stairs) or in an elevation window (doors, windows); *Shape: Polygon* in Properties
  of Object for other outlines.  Openings cut holes automatically, in the analysis and in the
  drawing: rectangular openings lined up with the shell give an exact hole and a conforming mesh
  (also openings running past the edge, or covering a whole floor); other shapes drop the mesh
  cells whose centre is inside (a warning says so).  Area loads are not applied in the hole and the
  diaphragm rotational inertia leaves it out.
* **One command per action** – *Move* is the CAD Move (Ctrl+M or `MV`; exact offsets typed as
  `dx,dy,dz` replace the old Move form); *Array / Replicate to Stories...* (Ctrl+R) keeps only what
  the modify commands cannot do (numbered linear / radial arrays, copies to other stories);
  mirroring is the Mirror command.  The left palette has the modify commands in its *Modify*
  fly-out only.
* **Ctrl+C / Ctrl+V** – Paste places the copied objects at the cursor (by their lower-left corner),
  or at a typed offset `dx,dy,dz`; in another story's plan they go to that story.
* **Snaps and guides** – one place: the status bar strip (SNAP = all object snaps, GRID, ORTHO,
  POLAR, TRACK = alignment lines, EXT, PERP, INC, and ▾ for every other snap and Snap Options),
  the same switches as *Draw > Snaps and Guides*.
* **Direct editing** – with the select tool, drag a selected object to move the selection
  (hold Ctrl when releasing = copy); drag a blue grip (a joint of the selection) to move that
  joint - dropped on another joint it connects there.
* **Guides and typed input** – snap tooltip at the cursor (snap kind, length, angle); alignment
  lines with nearby joints (object snap tracking), polar tracking (45°), extensions and
  perpendiculars on by default; after the first point type `6` (distance along the cursor),
  `6<30` (distance and angle) or `3,4` (relative) and Enter - in every drawing and modify command.
* **Plan and elevation windows** show only their story / frame line: model, loads, deformed
  shape, diagrams, contours and reactions (columns are left out of plan diagrams).
* **Moment diagrams** are drawn on the tension side by default (positive below a beam, negative
  above): *Options > Moment Diagrams on Tension Side* or the Output tab of Graphics Colors.
* **Result diagrams** – coloured by sign (positive / negative, split where the diagram crosses
  zero) in colours apart from the member colours; envelopes show the maximum and minimum curves.
* **Panels** – the drawing gets the whole window: the bottom *Results* / *Log* and the right
  *Properties* panel start hidden.  The *Tables* and *Properties panel* buttons (top toolbar),
  *Display > Show Tables* (Ctrl+T) and *Options > Panels* show / hide them; every panel can be
  closed with its own button.  Validation and analysis errors open the Log by themselves.
* **Load display** – like ETABS / CYPE 3D: frame loads as a filled load diagram with arrows onto the
  member and one value with units ("12.5 kN/m"), point / joint loads as a solid arrow ("50 kN"),
  area loads as a few arrows and "2.5 kN/m²" at the centre; values sit on a filled label box.  A
  load along the line of sight (gravity in plan) is turned into the plan (frame diagrams) or drawn
  as the force-into-the-screen symbol (circle and cross).  Plan and elevation windows show only the
  loads of the objects they show.
* **Model windows** – default two windows: **plan (left) + 3D (right)**.  *Window* menu / top
  toolbar: 1, 2 side-by-side, 2 stacked, 3 or 4 windows; each window header chooses what it
  shows (Plan + story, 3D, Elevation X-Z / Y-Z + grid line, results on/off).
  *Window > Arrange windows* sets everything at once; the layout is remembered.
  Tool, selection and display options are shared by all windows; the active window has a blue header.
* **Draw / Select palette** (left, ETABS workflow order; a button with a corner triangle is a
  CAD-style fly-out: press it and the group's tools open in a row beside it; the face remembers
  the last tool; *Show every tool* lists them all) – **Select** (click, Ctrl-add, drag left→right =
  window, right→left = crossing / intersecting line; the selection is highlighted dashed in the
  selection colour, the drag box is a translucent blue – colour and transparency in *Graphics
  Colors*) | **Draw** joints · beams (line, quick, secondary) · columns (plan click – WF-section
  icon, quick) · braces (line, quick X, /, \, V, inverted V) · floors (polygon, rectangle, quick)
  · walls (line, quick) · openings · load panels | **Edit** move, replicate, divide, delete |
  **Selection** all, previous, clear | **Snap** buttons + *Draw > Snap Options*.
  Lines, braces and quick braces / columns / walls also work in elevation views; in 3D they
  snap to model points and grid x story intersections.  Esc or right-click ends a chain.
* **Properties of Object** (under the Model explorer) – the ETABS form for the current draw
  tool: type, section, line type, moment releases, cardinal point, drawing control (fixed length /
  angle, parallel / perpendicular), offsets X/Y/Z, special joint ...  Empty with the select tool;
  the *Properties* panel shows nothing when nothing is selected.
* **Snap Options** (ETABS / CAD object snaps, all windows): joints, ends & midpoints, grid
  intersections, lines, edges, perpendicular, intersections, extensions, parallels, fine grid,
  reference lines, area centres, ortho, polar, length / angle increments, *Draw Using Snap Only*;
  **N** cycles the candidates under the cursor; AutoCAD markers (square = end / joint, triangle =
  midpoint, X = intersection, circle = centre, ...) in the form and at the cursor.  **Priority as
  in CAD**: a joint, end, midpoint, intersection or grid point inside the aperture always wins over
  tracking / alignment / reference lines (3D: of points on top of each other the front one), so a
  click lands on the point under the cursor.  **Shift / Ctrl + right-click** while drawing or
  moving / copying: the Object Snap menu (Endpoint, Midpoint, Intersection, Extension, Center,
  Node, Perpendicular, Parallel, Grid, Nearest, None) - that snap only, for the next point.
  Typed distances are in the display length unit (shown in the cursor tip, e.g. `> 3.5_ m`).
* **Drawing lines**: *Line shape* Straight or arcs the AutoCAD way (3 points; start, centre, end;
  start, end, radius), an arc made of *n* straight members; the chosen section is used whatever
  the slope.  Floors / walls / load panels: Enter or right-click closes the outline back to the
  first corner.  Moment releases: Continuous, Pinned, Pinned at start / end (start / end instead
  of I / J everywhere); pinned ends show a ring by the joint.
* **Left palette**: every tool its own button by default, flowing into a second column when the
  window is short, thin lines between the categories; the compact palette's buttons work like
  Photoshop (click = the tool shown, press-and-hold / right-click = the others; the last picked
  takes the face).
* **Analysis**: a modal progress window with **Cancel**, *Finished* at the end; nothing can be
  edited while it runs; afterwards the model is **locked** (padlock on the toolbar / status bar,
  Analyze > Lock / Unlock Model) - a change asks to unlock and deletes the results.  The mesh
  refinement check is off by default (Analysis Model Defaults): the automatic mesh is solved once.
* **Views**: joints as squares (on by default), selected joints as large squares drawn over
  everything (also extruded); joint / area labels; grid bubbles in circles (size and distance:
  Options > graphics sizes); ▲ ▼ (Page Up / Down) step through the stories of a plan or the grid
  lines of an elevation; right-click an object: Information, Select, Show only this object,
  Hide, Show all.  Set Active Degrees of Freedom shows the DOFs on a picture with presets.
  Units print with superscripts (m², cm⁴) and symbols with subscripts (I₃₃, f′c).
* **Right-click** on a selected joint / frame / area (select tool) opens its editable
  *Information* form (Geometry / Assignments / Loads, ETABS-SAP-SAFE style; Shift+right-click
  chooses among overlapping objects).  *Assign > Joint > Restraints* shows pictures of every
  support (fixed, pinned, rollers, sliders, guided, custom DOF) and inclined rollers via joint
  local axes; 3D windows draw the matching symbols.
* **Load panels** (CYPE 3D style, `docs/CYPE3D_STUDY.md`) – *Draw > Load Panel* polygon (flat or
  inclined) carrying loads per pattern, **one-way** (strips on the crossing bars), **two-way**
  (tributary areas) or onto shells, uniform or varying with height (wind / soil), projected or
  not, with a list of receiving frames.  Converted to member loads at analysis time
  (`AnalysisModel.panel_report`).  The **Show** button of the load display picks joint / frame /
  area loads / free panels or *as applied to members* (panels and area loads resolved onto the
  bars, no slab loads).
* **Load menu** (between Assign and Analyze; CYPE 3D concept) – one command per load kind:
  *Area Load*, *Load Panel*, *Frame Load*, *Joint Load*, plus *Other Loads* (temperature, ground
  displacement ...).  Each command works both ways: **command first** – set the load in
  *Properties of Object*, pick the objects (click = add / remove, drag = window / crossing), then
  Enter, right-click or the Apply button; **objects first** – select them, then the command: the
  form shows their current load of the pattern to edit (Replace), and applies the same way.  One
  undo step per apply; Esc drops the picks, a second Esc leaves the command.
  **Area loads are load panels made from the area** (on the area itself = the ETABS area load,
  or one-way / two-way to the frames around it, also for load-only Null areas); they follow the
  area (move, replicate, copy / paste, reshape, Divide Shells gives every piece the load) and go
  with it when it is deleted.  A **drawn Load Panel** is a load area of its own: it never follows
  the floors - move it with *Edit > Move* or delete it yourself.  *Delete Loads on Selection*,
  *Load Panel / Area Load Report* (applied vs carried).  Imports (ETABS / SAP2000 / SAFE) and older
  project files turn area loads into panels.
* **Guidance (bottom left)** – every command says what to do next and how to finish / leave
  (e.g. "Area Load > 2 floors / walls picked > Enter / right-click / Apply = add").
* **Free 2D views** (SAP2000 *Set 2D View*) – the window header's *Custom Z... / Custom X... /
  Custom Y...* entry shows a plan or elevation at any coordinate, not only at stories / grid
  lines; drawing tools work there (a column runs from the level below).  *View > Reference Lines*
  draws CYPE 3D reference lines through every joint (X / Y / Z selectable) in their own colour,
  distinct from the building grid; they are snap targets.
* **File > New Model** – grid-only or frame+slab template; **Edit > Grid lines and stories**.
* **Define** – materials, frame sections, shell sections, diaphragms, load patterns / cases /
  combinations.
* **Define > Frame Sections** (section manager) – every section with its type, material, size
  and the number of members using it; filter / search; **Add New, Add Copy, Modify / Show,
  Delete** (not while members use it; renaming follows the members; all undoable).  The
  selected section is **drawn with its dimensions** (overall width / depth, flange and web
  thickness) and local axes 2 / 3 at the centroid, with its properties in the display units:
  A, I33, I22 (I23 and the principal angle when not symmetric), J, As2, As3, S, Z, r, weight
  per length, material strengths, modifiers.
  *Add New* opens a chooser with drawn icons: **concrete** rectangle, circle, **T-beam** (with
  the slab); **steel, hot-rolled** I / H (unequal flanges allowed), channel, tee, angle, double
  angle, double channel, box / tube, pipe, plate, round bar; **steel, cold-formed** lipped C,
  **lipped Z** and plain **Z** (inclined principal axes: I23, Imax, Imin and the angle are shown),
  plain channel C, angle L and **hat** (TIS 1228 / JIS G3350 shapes, inside bend radius);
  **composite (SRC)** columns - a WF / H encased in rectangular or circular concrete;
  **From catalog** (see below), **General** (type the properties) and the
  **Section Designer**.  (L-beams, hollow concrete boxes / circles stay readable in older
  models but are not offered.)  Each shape form shows the drawing and the properties live
  while the dimensions are typed (arithmetic allowed, e.g. `60-12`), names itself after the
  dimensions (B30x60, T30x60, H300x150x6.5x9, C150x65x20x2.3, Z150x65x20x2.3,
  SRC60x60-H300x300) and checks them.  The first steel section of a model without steel adds
  SS400 (TIS 1227 / JIS G3101).
* **Steel section catalog** (Frame Sections > **Catalog...**, or Add > From catalog) –
  built-in tables, several sections at once, with the drawing and properties of the one
  pointed at: **Thailand / Japan** TIS 1227-2558 = JIS G3192 (80 H, 20 I-beams, 17 channels,
  47 equal and 13 unequal angles, 11 T) and TIS 1228-2561 = JIS G3350 (every table: 6 angles,
  31 C, 37 lipped C, 7 Z, 2 lipped Z, 8 hats), taken from the TISI full texts; **TIS 107-2566**
  pipes and tubes - round (JIS G3444), square and rectangular (JIS G3466) and the ASTM A500 /
  EN 10219 sizes of its annex tables, about 1,240 sizes, checked against TIS 107-2533;
  **China** GB/T 11263 (HW, HM, HN, HT) and GB/T 706
  (I-beams, channels, equal / unequal angles); **USA** AISC (W, HP, S, M, C, MC, L, HSS, Pipe);
  **Europe** EN (IPE, HE A / B / M, IPN, UPN, UPE, L).  The tabulated A, I33, I22 (and J where the table has it) are used in the
  analysis, the dimensions (with the fillets) draw the section and give the shear areas,
  S, Z and J.  **Import...** reads a CSV of the same columns (standard, series, name, shape,
  t3, t2, tf, tw, r, r2, lip, t2b in mm, A cm2, I33 / I22 / J cm4, mass kg/m - a supplier's or the
  office's table) or a CSI xml database.  Sources: `structstudio/catalog/data/SOURCES.md`.
* **Define > Reinforcing Bar Sizes** – the project's bar sizes (saved in the model), edited
  in place (name, diameter, area in the display units); **Catalog...** ticks sizes of
  TIS 20 (RB6-RB34), TIS 24 (DB6-DB40), ASTM A615 (#3-#18), JIS G3112 (D6-D51),
  GB 1499 (6-50 mm) or EN 10080 (6-50 mm); **Add Size** for one's own; **Import / Export** CSV (name, diameter mm,
  area mm2, standard).  Undoable.
* **Tapered flanges** – I and channel shapes have a *Flanges* option: parallel, or tapered
  like the rolled shapes of a standard (JIS / TIS channel 5 deg, JIS / TIS I-beam 8 deg,
  GB channel 1:10, GB I-beam and AISC C / S 1:6, EN UPN 8 % / 5 %, IPN 14 %) or a custom
  slope; the flange thickness is then the mean thickness (middle of the flange), as the
  standards define it.  TIS C150x75x6.5x10 drawn with the 5 deg taper gives A 23.71 cm2,
  Ix 861, Iy 117 cm4 - the table's values (parallel flanges: Iy 133).
* **Corner radii of steel shapes** – rolled I / H, channel and tee have the web-flange fillet
  r (channels also the flange tip r2), angles the root r1 and toe r2, tubes the outside
  corner radius (inside = outside - thickness), cold-formed C / Z the inside bend radius;
  the drawing shows the arcs and A, I, S, Z include them: H300x150x6.5x9 r13 gives
  A 46.78 cm2, Ix 7210 cm4 (JIS 46.78 / 7210), L100x100x10 A 19.00, Iu 278, Iv 71.9
  (JIS 19.00 / 278 / 72.0), C150x65x20x2.3 A 7.011 cm2 (JIS 7.012).  J of rolled shapes with
  fillets by finite elements (W14x90: 4.064 in4, AISC 4.06), of tubes and cold-formed
  sections by the thin-walled formulas on the rounded centre line.  0 / blank = sharp corners
  (older models unchanged); library sections keep their tabulated values.
* **Composite (SRC) columns** – concrete and steel materials, the steel I's dimensions and
  fillet, optionally rotated 90 deg; properties transformed to the concrete with
  n = Es / Ec (G for J, solved by finite elements with each material's G), shear areas
  from the shear-flow integral over the transformed widths, weight and mass from each
  material's own area; the analysis recomputes n from the current materials.  Each
  material is drawn in its colour; "Open in Section Designer" continues on the outline.
  Reinforcing bars are not included.
* **Section Designer** – any cross-section drawn like in CAD: polygon, rectangle, circle,
  *insert shape* (any catalogue shape as an editable outline), openings; select (click,
  window from the left, crossing from the right), drag points or shapes; CAD commands with
  base points: **Move** (M), **Copy** (CO, repeated copies), **Mirror** about a picked line
  (MI, keep or erase the source), **Rotate** (RO, angle typed or picked), **Trim** (TR: cut
  along a line, click the parts away), Ctrl+C / Ctrl+X / Ctrl+V with the paste following
  the cursor, flip / rotate 90, solid <-> opening, delete, undo / redo; commands work
  selection-first or command-first (pick objects, Enter).  Snaps: end point, mid point,
  centre / quadrant / centroid, intersection, perpendicular, nearest, each switchable, all
  on / off with F3; grid display F7, grid snap F9, ortho F8 / Shift.  Typing on the command
  line: `x,y`, `@dx,dy`, `@length<angle`, a distance along the cursor, `c` to close,
  `p r ci i h u f`, `g 5` for the grid; the points of the selected shape in an editable
  table; a material per shape (composite sections).  Overlapping / touching shapes are united (a T as two rectangles), openings removed;
  A, centroid, I33, I22, I23 exact, **J by finite elements** (Saint-Venant warping function,
  quadratic triangles on a conforming Delaunay mesh: 0.03 % on a rectangle), As2 / As3 from
  the shear-flow integral, S, Z, r - updated while drawing.  The drawing is saved with the
  section, scales with unit changes, and is used in the 3-D extruded view.  **Assign** – restraints, diaphragms,
  sections, releases (loads: *Load* menu).  **Select** – all, invert, previous, by type,
  story or section.
* **Analyze > Run (F5)**, deformed shape (D), mode shapes, frame diagrams, shell contours
  (F11…V23), reactions (R), assigned-load display, result tables with CSV/XLSX export,
  ETABS / SAP2000 / SAFE import (all versions, text, tables, native), undo/redo, `.ssj.json` project files.
* **Automatic meshing (ETABS / SAFE style)** – floors and walls are meshed around everything
  that touches them: columns and walls standing on a slab, beams in or crossing it (also mid-slab,
  inclined, ramps), loaded joints, line loads / Null lines, springs and supports get nodes
  *on* the mesh, so loads reach the slab where they really act.  Rectangular shells: a tensor
  mesh with shared levels; free-form / non-convex / holed shells: conforming Delaunay split into
  quads, smoothed and repaired (`mesh/area_mesh.py`).  Shells meshed apart are tied at shared
  edges (bilinear constraints), a member passing through a slab is cut where it crosses it.
  *Display > Show Analysis Mesh* draws the mesh and the tie points.  Checks: the reaction equals
  the applied load on free-form slabs, ramps, skew walls and braces through slabs; Navier plate
  on flat, 30°-tilted and skewed slabs; the SAFE samples unchanged (`tests/unit/test_mesh_freeform.py`).
* **Analysis window** – percent, a moving green bar, the current step, elapsed time and a live
  log (meshing, checks, each load case, results); it can run in the background.
* **Problems found before / during the analysis** (`diagnostics.py`) – loads that reach nothing
  (load in the air, distance to the nearest floor), objects resting on nothing, a model without
  supports, mechanisms ("P14 can move freely in RZ"), lines off the slab... each with the object,
  **where it is** (story, grid, coordinates) and what to do.  *Analyze > Last Analysis Problems*
  lists them; double-click / Locate selects the object and shows it; **Save as text file** writes
  the whole report (errors / warnings / notes) for checking or sending.
* **Find (Ctrl+F, main toolbar / Edit > Find Object)** – type a name / id (`B23`), a label (`C5`,
  `C5 Story3`), a grid intersection (`B/3`, `B/3 Story2`) or coordinates (`12.5, 6, 9`; `x, y` =
  nearest in plan) and press Enter: it selects the object, takes the plan to its story, zooms and
  flashes it; several matches give a list; a location with nothing on it is marked.
* **Results as lines or extruded** – the Extrude button (Ctrl+E, also on the Display bar) applies
  to the results too: shell contours on slabs / walls with their thickness, the deformed shape
  with the real sections bending with the joints, and **member colours** (Display bar *Colour*:
  P, V2, V3, T, M2, M3 along each member, or the **stress ratio** |P|/A + |M2|/S22 + |M3|/S33
  over fy / fc' in the CSI ratio colours 0.5 / 0.7 / 0.9 / 1.0 - an elastic indication, not a code
  design check; frame design ratios will use the same colours).  The objects a result is drawn
  on are hidden meanwhile, the undeformed model stays as a faint shadow under a deformed shape.

* **Compare with ETABS / SAFE yourself** – double-click `scripts\Compare-With-ETABS-SAFE.cmd`: it builds
  the free-form slab, the two slabs with an opening and the inclined building (ramp, skew wall,
  column under the ramp, brace through a slab) in Structural Studio, builds the *same* models in
  ETABS and SAFE through their official API (`Run-CsiCompare.ps1`, every call logged), runs
  them and writes `tests\verification\data\compare\report.txt`: total reaction, deflection of
  every loaded / column / slab joint (ours at h = 1.25 m, 0.5 m and with the automatic mesh +
  rigid zones = the converged answer) and every support reaction.  Each model runs in its own
  ETABS / SAFE with a 10 minute limit (on a timeout: screenshot `<model>.hang.png`, the last
  API call in `<model>.calls.log`, then the next model).  SAFE gets the model shifted so that
  the slab is at its datum Z = 0.  Joints a CSI program left unconnected (zero or huge
  displacement) are listed, not compared.  FREEFORM and TWOSLABS are also sent as `*_RZ`: the
  same models with a stiff slab area (bending stiffness x 100, like SAFE's Col_Stiff areas) over
  every column, so that our column rigid zones are compared like for like.
* **Thick plates (ShellThick) = DKMQ** (Katili 1993): converges to the Mindlin solution
  (Navier, a/t = 5..100, distorted meshes) and becomes the thin-plate DKQ as the plate gets thin.
  Against the 19 SAFE thick samples it is closer than the former MITC4 in 14 (sum of the worst
  differences 84 % instead of 112 %; fine meshes within 1.3 % instead of 5.2 %).  On very coarse
  meshes (a few elements per panel) SAFE's own thick element still differs by up to ~20 %
  (S02a, S06a): its exact formulation is not public; `shell.THICK_FORMULATION = "MITC"` gives
  MITC4 (checked against OpenSees ShellMITC4).
* **New slab / wall sections are ShellThick (DKMQ) by default**: no thin / thick choice to make -
  a thin slab gives the DKQ (Kirchhoff) answer, a thick one includes shear deformation.  Imported
  ETABS / SAFE models keep the type they were defined with; `ShellThin` (DKQ) stays available,
  e.g. to reproduce a SAFE thin-plate model exactly.
* **Automatic mesh size + refinement check**: with no mesh size set, each shell gets a size
  from its geometry (`area_mesh.auto_size`): at least 8 elements across its narrowest width,
  6 between neighbouring columns / supports, at most 1 m, rounded to 1, 0.5, 0.25... m so that
  meshes of neighbouring shells nest (one size per plane).  Columns no longer refine the whole
  slab: the mesh follows each column's cross-section and is graded around it (below).
  Analyze then solves the linear static cases on that mesh (h) and on meshes 2h and 4h (cheap),
  estimates the *observed* order of convergence p and the error left at h,
  |u2h - uh| / (2^p - 1) (Richardson, no assumed order - near column joints plates converge
  slower than h^2; values that do not settle count the whole last change).  Above 1 % the size
  is halved (twice at most, never past 40 000 shells).  The outcome is in the analysis notes
  ("Mesh check: ... observed order ...", a warning when not reached) and in
  `results.am.mesh_check`.  Moments at a point support do not converge (a singularity): the
  check uses displacements.  A size you set yourself (Analyze > Automatic Mesh Settings) is
  used as given.
* **Column rigid zones in slabs** (`model.options.slab_column_rigid_zone`, on by default, as
  SAFE): the slab inside a column's cross-section (rectangular / circular columns within 10
  degrees of vertical) moves with the column as a rigid body (out-of-plane DOFs).  The mesh
  follows the cross-section's outline exactly at every mesh size (cut to the slab for edge and
  corner columns; a corner just inside a curved edge is put on the edge), with elements of
  0.25 x the column size at the face growing to the slab's size: in rectangular slabs mesh
  lines at h, 2h, 4h ... from each face (the regular lines stay elsewhere, so floor and wall
  meshes still nest), in free-form slabs rings of points around the column, 1.5 x wider each.  Before, the rigid zone was
  "the mesh nodes that fall inside the column", so its real size changed with the mesh: the
  `*_RZ` comparison showed TWOSLABS P1 at -14.7 / -12.5 / -14.0 mm for 0.25 / 0.125 / 0.10 m
  while the mesh check claimed 0.5 %; now -13.1 / -13.6 / -13.6 / -13.6 mm at 1 / 0.5 / 0.25 /
  0.125 m.  A column on the edge of an opening or overlapping another column still falls back to
  corner nodes.  Beam pieces inside a column's cross-section (rigid) get no output stations,
  like an end offset: beam results start at the column face.  Shell contours leave out the
  elements inside the columns (and their values out of the nodal averaging), so they show and
  scale to the values at the column faces.  Without it the column-slab joint is
  one node whose rotational stiffness falls as the mesh is refined, so deflections keep growing
  with refinement (FREEFORM: -24.1 / -24.7 / -25.2 mm at 0.5 / 0.25 / 0.125 m) instead of
  settling (with rigid zones: -21.7 / -21.1 mm at 0.25 / 0.125 m).  ETABS joins column and slab
  at one node, so its slab deflections near columns are larger than ours and depend on its mesh.
* **Analysis model defaults** (Analyze > Analysis Model Defaults...): what every frame / slab gets
  unless the object sets its own - frames: end offsets automatic from connectivity (on),
  rigid-zone factor 0.5 for concrete, 0 for steel and other materials (CSI: steel frames
  never fully rigid unless panel zones are modelled, 0.5 for concrete frames); slabs: rigid
  zone over columns (on); mesh: refinement check (on) and its tolerance (1 %).  Assign > Frame >
  End Length Offsets sets a frame's own values or "use the model defaults"; the Object Info
  panel shows the value in use.  Imported ETABS / SAP2000 / SAFE models start with the CSI
  defaults (no automatic offsets, factor 0) so they reproduce the original; models built in
  code can do the same with `model.use_csi_frame_defaults()`.
* **Mesh check speed**: its solves recover no stress resultants (only deflections are compared)
  and the element matrices / factorised stiffness of the finest mesh are reused by the analysis;
  with columns no longer refining whole slabs, `examples/office.ssj.json` analyses in ~7 s
  (2 040 shells, estimated error 0.03 %) instead of ~12 s.
* **Result display**: colour scales on round limits (0, 0.25, 0.5 ...), deflections in mm,
  one scale for all windows (the whole model's range), the model moved left while a scale is
  shown; frame diagrams show the end and largest values and a caption (component, unit, max /
  min); in a plan, beam diagrams are turned into the plan with positive below horizontal
  members and right of vertical ones (Options > Plan Diagrams: Positive Below / Right, or the
  Graphics settings); with the Select tool the Properties panel shows a results summary
  (largest deflection and where, base / support reactions, mesh and mesh check) - replaced:
  with the Select tool the panel now shows the selected object in short (frame: section with its
  size, material, length, joints, releases, offsets, loads; shell: section and thickness,
  material, area, loads, diaphragm; joint: coordinates, support, loads, connected frames;
  several: counts, common section, total length), in the display units; with a draw / load tool
  it shows that tool's settings as before.
* **Toolbars**, one per category in the order of the work (after the ETABS / SAP2000 / SAFE
  toolbars): row 1, the model - File (new, open, save) | Edit (undo, redo) | Define (materials,
  frame / slab / wall sections) | Loads (patterns, cases, combinations) | Analyze (run) | Find,
  and the X·Y·Story / X·Y·Z switch alone at the right end; row 2, looking at it - View (plan,
  3D, elevations, zoom, pan, rotate) | Display (display options, grid, extruded sections, load
  assigns) | Results (case, mode, undeformed / deformed / reactions / frame forces F4-F8,
  diagram, member colours, tables).  Window layouts (1-4 windows), panels and dark / light
  mode are in the new Window menu.
* **Units** (as ETABS / SAP2000): the units button in the lower right corner shows the model's
  units ("kN, m, C"); click it for the consistent sets (N / kN / kgf / tonf with mm / cm / m,
  lb / kip with in / ft, temperature C or F) or Show Units Form... (= Options > Display Units,
  Ctrl+U: force, length, temperature, quick sets Metric SI / MKS / U.S., and the unit of every
  kind of value before and after); hovering lists them.  Every value of the model is converted,
  the structure does not change, the analysis results are kept and shown in the new units, the
  view stays the same, values in the Properties of Object form and the snap spacings follow,
  and Edit > Undo goes back.  Checked by `tests/unit/test_unit_conversion.py`: every number in
  the model has a unit rule (a new field without one fails); converting to all 16 sets and back
  is exact; a model with every kind of input (trapezoidal / moment / point frame loads at
  absolute distances, temperature changes and gradients, settlements, springs, partial fixity,
  offsets, a haunched section, a tendon with wobble, acceleration loads, spectra) gives the same
  displacements, reactions, member forces, periods and spectrum response in every set; imported
  ETABS / SAP2000 / SAFE verification models give the same answer in N-mm-C as in kip-in-F.
  Fixed on the way: frame loads were not converted (a 8 kN/m load became 8 kN/mm in mm); the
  thermal coefficient, temperature loads (C <-> F), tendon wobble, nonprismatic absolute lengths
  and the gravity option were not converted; imported ETABS / SAP2000 / SAFE files lost their
  temperature unit (F read as C).
* **Display units item by item** (Options > Display Units, tab "Display units per item"; units
  button > Display unit sets): each kind of value is shown and typed in its own unit on top of
  the model's consistent units - structure dimensions, section dimensions, section / rebar
  areas, inertias, displacements, forces, moments, line loads, area loads, stresses / strengths
  / E, unit weight.  Sets: consistent, Metric SI (kN, m, mm, MPa), **Thai legacy - kgf** (every
  force in kgf: kgf, kgf-m, kgf/m, kgf/m^2, kgf/m^3; model units kgf-m-C), **Thai legacy - tonf**
  (every force in tonf: tonf, tonf-m, tonf/m, tonf/m^2, tonf/m^3; model units tonf-m-C) - both with lengths in
  m, sections in cm / cm^2 / cm^4, deflections in mm, strengths and E in ksc - and U.S. (kip,
  kip-ft, ft, in, ksi, psf, pcf); any label can be typed ("kgf-cm", "tonf/m^3").
  Forces are always labelled kgf / tonf (kilogram-force, tonne-force), never kg / ton (masses);
  "kg" / "ton" typed by hand or saved by older versions are read as kgf / tonf.
  Used by the material / section forms, the Properties panel and Object Information, the
  Properties of Object load and offset values, the load panel form, the frame-load / offset /
  limit forms, load labels, frame diagrams and their caption, shell contours (tonf-m/m), member
  colours, reactions, deflection legends, the results summary and the result tables (unit in
  each column name).  Display only: saved with the model (``display_units``), the model is not
  changed.  Tested in `tests/unit/test_display_units.py`.
* **Freeform mesh smoothing** runs on plain Python lists: a 17 000-element slab meshes in ~7 s
  (was ~2 min).

## Python API

```python
from structstudio import Model, analyze
from structstudio.model import FrameSection, ShellSection, FIXED

m = Model("office", "kN m")
m.add_material("C30", E=27e6, nu=0.2, weight_density=24)
m.add_frame_section(FrameSection.rectangular("B30x60", "C30", depth=0.6, width=0.3))
m.add_frame_section(FrameSection.rectangular("C40x40", "C30", depth=0.4, width=0.4))
m.add_shell_section(ShellSection("S20", "C30", thickness=0.20))
m.set_stories([("L1", 3.5), ("L2", 3.5)])
m.add_diaphragm("D1")
for x, y in ((0, 0), (6, 0), (6, 5), (0, 5)):
    m.add_point(x, y, 0, restraints=FIXED)
for z, story in ((3.5, "L1"), (7.0, "L2")):
    pts = [m.add_point(x, y, z).name for x, y in ((0, 0), (6, 0), (6, 5), (0, 5))]
    for p in pts:
        q = m.points[p]
        m.add_frame((q.x, q.y, z - 3.5), p, "C40x40")
    for k in range(4):
        m.add_frame(pts[k], pts[(k + 1) % 4], "B30x60")
    m.add_area(pts, "S20", diaphragm="D1", name=f"F{story}")
m.add_load_pattern("DL", "Dead", self_weight_multiplier=1.0)
m.add_load_pattern("LL", "Live")
for a in m.areas:
    m.add_area_load(a, "LL", 2.5)           # kN/m², gravity
m.add_linear_static_case("DL"); m.add_linear_static_case("LL")
m.add_modal_case("Modal", max_modes=6)
m.add_combination("U1", {"DL": 1.4, "LL": 1.7})

res = analyze(m)
print(res.modal_periods("Modal")[0])
print(res.frame_forces("B5", "U1").table()[:3])    # station forces, max/min
print(res.story_drifts("LL"))
```

## Import ETABS, SAP2000 and SAFE models

*File > Import ETABS / SAP2000 / SAFE...* (Ctrl+Shift+I), *File > Open*, or drag a file onto
`Start Structural Studio.cmd`.  Files are recognised by content, so any version works:

| Program | Text model | Table exports | Native file |
|---|---|---|---|
| ETABS | `.e2k` / `.$et` (v9 – 22) | Excel `.xlsx`, Access `.mdb`/`.accdb`, text tables | `.edb` * |
| SAP2000 / CSiBridge | `.s2k` / `.$2k` tables (v8 – 26), keyword `.s2k` v6–v7 (beta) | `.xlsx`, `.mdb`/`.accdb`, text | `.sdb` * |
| SAFE | `.f2k` / `.$f2` (v12 – 2016), SAFE 20+ table exports | `.xlsx`, `.mdb`/`.accdb`, text | `.fdb` * |

\* Native binary files are opened by the CSI program installed on this computer through its official
API (`structstudio/io/csi_convert.ps1`; also `scripts/Convert-CsiModels.cmd` for drag and drop), which
writes the model tables (and for ETABS the `.e2k`) to `%LOCALAPPDATA%\StructuralStudio\import`; your
file is not changed.  Access files need `pip install pyodbc` and the 64-bit Access Database Engine.

Every import ends with a report: approximations (warnings) and anything not supported yet (skipped) –
nothing is dropped silently.  SAP2000 objects keep SAP2000 behaviour (frames / areas not meshed unless an
auto-mesh is assigned, vertical mass), ETABS objects ETABS behaviour (stories, auto mesh, lateral mass),
SAFE slabs are thick plates with their soil springs (area springs) and fixed far ends of columns / walls.

```python
from structstudio.io import import_model
res = import_model("frame.s2k")      # or .e2k .f2k .xlsx .mdb .edb .sdb .fdb
print(res.report)
results = analyze(res.model)
```

## Cross-check against OpenSees

`tests/xcheck` builds thousands of random structures - cantilevers, crooked chains, plane and
3D frames with sloped roofs and leaning columns, pin-jointed plane / space trusses, arches,
closed rings, ribbed domes, helices and spiral stairs, lattice towers, skewed grillages, random
space frames, beams on springs, flat plates (Mindlin, ShellThick) with edge beams - with random
sections (Euler-Bernoulli and Timoshenko), rotated local axes, end releases, joint offsets, rigid
diaphragms, springs, point masses and odd loads (trapezoidal, partial span, projected, local and
global directions, member point forces and moments, self weight).  Each model is analysed by
Structural Studio and by an independent OpenSeesPy model; joint displacements, reactions,
member end forces and natural periods must agree to 5e-5 (trapezoids: OpenSees needs strips,
2e-4).

```powershell
python -m pip install openseespy
python -m tests.xcheck.compare --n 10000 --jobs 8
```

The campaign found and fixed: wrong sign of M2 (1-3 plane), wrong equivalent loads of point /
distributed moments about local axis 2 away from the member ends, point loads at end J missing
from the end section force, and missing instability warnings (mechanisms).  Last run: 10 000
models, see the table printed by the command.

## Layout / docs

* `docs/ARCHITECTURE.md` — layers and the compiled-kernel boundary
* `docs/CONVENTIONS.md` — every ETABS behaviour reproduced, with its public source
* `docs/ROADMAP.md` — phases 2–6
* `tests/verification` — CSI verification examples (acceptance tests)

## Nonlinear & seismic analysis

Pushover (gravity staging, load / mode / acceleration patterns, displacement control,
ETABS-style vertical hinge drops with load redistribution, arc-length), nonlinear time
history (Newmark / HHT, Rayleigh or modal damping), FNA (fast nonlinear analysis), exact
linear modal time history, P-Delta and corotational large displacements, fiber
force-/displacement-based members, fiber P-M2-M3 hinges, CSI-style hinges with ASCE 41
automatic backbones, Wen isolators and viscous dampers, plus
spectra, ASCE 41 / EC8 N2 target displacements, IDA and fragility fitting.  Materials,
elements and solvers are cross-checked step by step against OpenSees 3.8
(`pytest tests/xcheck`, needs `pip install openseespy`).

```powershell
cd 'E:\REA ETABS\studio'
& 'E:\REA ETABS\tools\python\Scripts\python.exe' examples\nonlinear_rc_frame.py
& 'E:\REA ETABS\tools\python\Scripts\python.exe' -m pip install openseespy   # optional: OpenSees cross-checks
& 'E:\REA ETABS\tools\python\Scripts\python.exe' -m pytest -q tests\unit\test_nonlinear.py tests\xcheck
```

Details, verification table and known differences: `docs/NONLINEAR.md`.
