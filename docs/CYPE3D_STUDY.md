# CYPE 3D study: what it does well, and what Structural Studio takes from it

Sources: the CYPE 3D on-line help (`info.cype.com`, *Geometry*, *Load*, *Loads
(panels)*, *Surface loads* pages, versions 2025), the CYPE 3D user and
calculation manuals (`nm3dmu09.pdf`, `nm3dmemo09.pdf`), and the installed
program (version 2025.d, `C:\CYPE Ingenieros`, examples in
`C:\CYPE Ingenieros\Examples\CYPE 3D`).  The installed program has no local
help files; the on-line help is the reference.  Nothing here comes from the
program's code.

## 1. Modelling workflow

CYPE 3D is a bar (and shell) program for steel, aluminium and timber
structures with a very free geometry: no stories, no plan grid.  The work
happens in a *Structure* tab with upper tabs *Geometry*, *Load*, *Analysis*,
*Results*, *Joints*.

### 1.1 Reference lines and planes

* Every node that is entered (directly, or as the end of a bar or the vertex
  of a shell) creates **reference lines** through it, parallel to the three
  global axes, when *Show new* (Planes group) is on.  They are drawn as thin
  dashed lines in a colour different from the bars, optionally one colour per
  direction (*Draw with colour depending on direction*).
* The **References** dialog toggles: *Display the grid*, *Reference lines*,
  *Only those with a node*, *Parallel to X / Y / Z*.
* *Visibility* shows or hides the reference lines of the planes containing a
  group of nodes; *Group* / *Ungroup* link parallel planes so that their
  contents move together.
* **Grid**: lines parallel to X and Y (coordinate, label, label position;
  *Generate gridlines* with number / spacing / initial label).  **Levels**:
  horizontal planes (reference + elevation; *Generate* with prefix, number,
  initial level, spacing).  Elements at a level belong to it; columns between
  levels go to the lower one.
* Nodes are entered on a reference line, on a bar, by coordinates (numeric
  keypad opens a *Calculator* for X, Y, Z) or anywhere in a 2D view.
* Dimensions are live: *Add* (set *Distance between lines*, pick two reference
  lines / bars / nodes) moves the geometry; *Show* adds informative dimensions.

→ Structural Studio: reference lines through joints, `View > Reference lines`
with per-direction toggles and their own colour (*Graphics Colors >
Reference lines*), snapping to them (*Snap Options > Reference lines*); the
free X · Y · Z grid mode (SAP2000 style) with independent Z levels; joint
entry with plan offsets and typed coordinates (*Properties of Object* form).

### 1.2 Snapping

Object snap, template object snaps (DXF/DWG), orthogonality, polar tracking,
dimension entry while drawing, coordinate entry.  Disabled receiving bars are
drawn dashed.

→ Structural Studio: the *Snap Options* form (ETABS list plus the CAD snaps:
nearest, perpendicular, intersection, extension, parallel, fine grid,
reference lines, ortho / polar tracking, length and angle increments) and the
`N` key to cycle between snaps.

## 2. Loads: panels and surface loads

This is the part worth copying: a load that lives on its own object and is
carried to the bars at analysis time, independent of any slab or shell.

### 2.1 Panels (*Load > Panels*)

* **New**: click the corners of a closed polygon (coplanar nodes, any
  orientation), right-click, then pick one of the auxiliary lines along the
  sides to set the **distribution direction** (*Parallel to the selected side*
  or *Perpendicular to the selected side*).  The *Loads on panel* window then
  takes one or more loads, each with a load case.
* **Positive direction** follows the load case: wind cases act normal to the
  panel (the program shows the unit vector), the others vertically along
  global Z (*Vertical downwards* is the default).
* Loads are **uniform**, or, when the panel is not horizontal, **vary with the
  height**: *Value at the lower level* and *Value at the upper level*
  (interpolated linearly in Z).
* Distribution: the panel is treated as a **one-way slab, simply supported**
  on the bars it crosses.  The load goes to the bars inside the panel that
  have a component perpendicular to the distribution direction; bars parallel
  to the direction get nothing.  Shells inside the panel receive the load as
  pressure.  The resulting line loads can be displayed on the bars.
* Tools: *Edit* (several panels at once: the first one's loads are copied to
  all), *Delete* (removes the surface loads too), *Move* vertices, *Divide* a
  panel by a line (2025.b; its surface loads are divided too), *Distribution
  direction*.

### 2.2 Surface loads (*Load > Surface loads*)

* A surface load is a polygon **inside one panel** (it must not overlap two
  panels); several surface loads with different geometries and values can sit
  on one panel (wind pressure zones on a roof slope).  If no panel exists the
  program offers to create one with the same shape, the distribution direction
  being the line through the first two points.
* Same load types as the panel (uniform / varying with height), same
  distribution rule (only the bars nearest to the loaded surface).
* **Assign** enables or disables bars as receivers (disabled bars are dashed;
  ties never receive).  Edit / Copy (same shape, other case and value) / Move
  (optionally the loads of the same zone in other cases) / Delete / change the
  load case.

### 2.3 What Structural Studio does with it

* `PanelLoad` object (`model.panel_loads`): polygon in any plane, loads per
  pattern (uniform or varying with height), direction *Gravity / X / Y / Z /
  Normal / projected*, **receivers** list (the CYPE *Assign* option), and a
  **distribution** option:
  * `one_way` - CYPE: strips parallel to the span direction, simply supported
    on every bar they cross (load between two supports split half-half,
    overhangs to the nearest support); the span direction is set from an edge
    of the panel (parallel / perpendicular), stored as a global vector.
  * `two_way` - **new**: tributary areas (every point of the panel loads the
    nearest receiving bar measured in the panel plane) - the ETABS / SAFE
    45-degree split, triangles on the short beams and trapezoids on the long
    ones; works for any polygon and any number of interior beams.
  * `shells` - pressure on the area objects inside the panel.
* The panel adds no stiffness.  At meshing time its loads become frame
  distributed loads (piecewise linear, 11 stations) and area pressures;
  `AnalysisModel.panel_report` holds (applied, carried) totals per panel and
  pattern so the transfer can be checked (the two-way sampling is within
  0.5 % of the exact tributary areas, the one-way strips are exact).
* So there are two ways to load a floor: the load **attached to the slab
  object** (ETABS / SAP2000 area loads, carried by the shell elements), and
  the load as **its own object** (the panel, carried to the beams before the
  analysis, CYPE style).  Both can be used in one model.
* Display: the *Loads* toolbar shows the pattern, and *Show* picks what is
  drawn - joint / frame / area loads, the **panel loads on the panel**, or the
  **panel loads as applied to the members** (the line loads after
  distribution, as CYPE shows them after *Load > Visible loadcase*).

## 3. Joint supports

CYPE 3D supports (*Node > Supports*) are free, pinned, fixed, or with elastic
springs, and the restraint can be given in a rotated system (an inclined
roller is a support rotated by an angle).  → Structural Studio draws a
picture per support type (fixed, pinned, roller in each direction, slider,
guided, free) in the *Assign Joint Restraints* form, with a distinct 3D symbol
each, and an **inclined support** option: the joint local axes angles a, b, c
(CSI convention), so a roller on a 30-degree slope is *Roller (vertical)* with
b = 30.

## 4. Still to take from CYPE 3D

* Surface loads inside a panel (sub-polygons with their own values), *Divide
  panels*, *Move* panel vertices.
* Dimensions that drive the geometry (parametric dimensions between reference
  lines).
* Node ties (equal displacements between nodes in chosen directions).
* Buckling lengths / lateral-torsional buckling data per bar (design).
