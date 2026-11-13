# ETABS-compatible conventions implemented

Source: *CSI Analysis Reference Manual* (rev. 18, 2017) and the *ETABS Software
Verification* manual shipped with ETABS 21 (public documentation).  Page numbers refer
to the Analysis Reference.

| Topic | Rule implemented | Where | Source |
|---|---|---|---|
| Global axes | right-handed, +Z up | everywhere | ch. III |
| Frame local axes | 1 = I→J; 1-2 plane vertical, 2 has +Z sense; vertical members (sin < 1e-3): 2 = +X; coordinate angle rotates 2-3 about +1 | `mesh/geometry.frame_axes` | p.109-110 |
| Section properties | A, J, I33 (bending in 1-2), I22 (bending in 1-3), As2, As3; zero shear area = shear deformation ignored | `model/properties.FrameSection`, `engine/elements/frame` | p.116 |
| Property modifiers | section-level × object-level multipliers for A, As2, As3, J, I22, I33, mass, weight | `mesh/mesher._section_props` | p.123 |
| Insertion points | cardinal points 1-11 + joint offsets; rigid links joint→centroid; local axes re-derived when the axis direction changes | `mesher._build_frame_elements`, `FrameFormulation.T_off` | p.125-127 |
| End offsets / rigid zone | flexible length Lf = L − rigid·(ioff+joff) for bending and shear only; axial and torsion use the full length; a moment/shear release makes that end offset fully rigid; output only on the clear length | `FrameFormulation` | p.127-131 |
| End releases | U1..R3 at either end, partial fixity springs, unstable combinations rejected | `FrameFormulation.condense` | p.131-132 |
| Span loads | concentrated / trapezoidal partial loads, global or local, projected option; consistent (interpolation-based) equivalent nodal loads | `FrameFormulation.fixed_end_load_vector` | p.135-139 |
| Internal force sign | P tension +, V2/V3/T positive on the +1 face in the + local directions; positive M3 causes compression on the +2 face; M2 is positive for the 1-3 plane bending a positive V3 produces (opposite of the right-hand moment about +2 - checked against SAP2000 Ex. 1-004 and OpenSees) | `FrameFormulation.internal_forces` | p.143-144 |
| Section force at a point load | value just after the load, except at end J (the element end force, value just before) - SAP2000 lists both values at a load station | `_load_resultant_left` | SAP2000 Ex. 1-001, 1-003 |
| Unstable structures | equations whose pivot drops below 1e-12 of the largest are reported (joint + DOF) as "unstable or ill-conditioned", results are still produced (SAPFire behaviour) | `engine/kernel.py`, `Solver._report_instability` | OpenSees cross-check |
| Frame mass | lumped half-half at I and J, translational only; ρA·mod + mpl | `FrameFormulation.lumped_mass` | p.134 |
| Self weight | w·A·mod + wpl, global −Z, pattern self-weight multiplier; beams/braces (not columns) weigh — and carry mass — only on the clear length between end offsets (as ETABS) | `mesher`, `assembly.load_vector_full` | p.134 |
| Diaphragm constraint | u1j = u1i − r3i Δx2, u2j = u2i + r3i Δx1, r3i = r3j (master-slave transformation); virtual master; mixing restraints and constraints rejected | `assembly.Assembler._number_dofs` | p.53-54 |
| Automatic story mass | area mass lumped to mesh nodes; analytical polar inertia of each area object enforced at the diaphragm master | `mesher._build_constraints` | p.334 (mass source), verification Ex.4 |
| Mass source | element self mass, added mass, mass from load patterns (gravity loads / g), lateral-only option (ETABS default) | `MassSource`, `assembly.mass_full` | p.334-336 |
| Gravity constant | 9.80665 m/s² SI; ETABS text import sets 386.4 in/s² (kip-in) / 32.2 ft/s² | `AnalysisOptions.gravity` | verification Ex.4 |
| Modal analysis | eigenvectors, mass-normalised (φᵀMφ = 1); participation factors and mass ratios about the centre of mass | `solvers.modal` | p.375-378 |
| Response spectrum | function in g × scale; constant extrapolation outside the defined periods; CQC/SRSS/ABS modal combination; SRSS/ABS directional combination; U1/U2 rotated by the case angle | `solvers.response_spectrum` | p.385-392 |
| Load combinations | Linear Add / Envelope / Absolute / SRSS / Range with max-min pairs; spectrum cases contribute ±max | `results.Results._combine` | p.351-353 |
| Shell element | planar 4-node; MITC4 transverse shear; membrane = bilinear + QM6 incompatible modes + drilling DOF tied by a Hughes-Brezzi penalty γ = 0.7·G·t, fully integrated, with the incompatible modes included in the penalty's continuum rotation (no locking in in-plane bending); thin = DKQ discrete Kirchhoff quadrilateral (Batoz & Tahar 1982, right-hand nodal rotations, shears from moment equilibrium), thick = MITC4 with the actual shear; f11…v23 modifiers | `engine/elements/shell` | ch. X; calibrated on ETABS 21 Ex.15 |
| Shell resultants sign | CSI: S11 = F11/t − 12·M11·x3/t³ - positive M11/M22/M12 compress the +3 face (gravity slab: positive at mid-span, negative over beams/walls); V13/V23 from equilibrium (sign not yet checked against CSI) | `engine/elements/shell.resultants` | CSI Analysis Reference, shell internal force output |
| Wall meshing (ETABS import) | `OBJMESHTYPE DEFAULT` walls are one shell element per wall panel, as ETABS does (finer meshes converge to the SAP2000 refined-mesh values instead) | `io/e2k._area_assign` | ETABS 21 Ex.15 results |
| Insertion point stiffness | cardinal-point offsets transform the frame stiffness only with `TRANSFORMSTIFF "Yes"`; otherwise the member acts on the joint line (spandrels in a rigid diaphragm carry P = 0, as in ETABS) | `io/e2k._line_assign` | ETABS 21 Ex.15c |
| Coincident joints | an unconnected point object at the location of another point stays a separate joint (load points tied only through the diaphragm) | `mesher.build` step 1 | ETABS 21 Ex.15 |
| Released DOFs without stiffness | a released rotation of a member with I = 0 (truss) is disconnected, not an error | `FrameFormulation.condense` | ETABS 21 Ex.10/11 |
| RS base reactions | combined from the modal base totals (CQC/SRSS/ABS/NRC 10 %), not summed from support envelopes | `solvers.response_spectrum` | ETABS 21 |
| NRC 10 % modal combination | modes within 10 % in frequency add 2·|Ri·Rj| | `solvers._correlation` | Reg. Guide 1.92 / ETABS `NRC10` |
| Automatic end offsets | rigid-zone factor with "automatic from connectivity": column top offset = largest beam depth framing in; beam/brace end offset = half the column dimension in the beam direction | `mesher._auto_end_offsets` | p.128 |
| Cardinal point 0 / absent | centroid (10) | `io/e2k._line_assign` | verification Ex.9 |
| Diaphragm master | placed at the centre of lateral mass of the diaphragm (eccentric masses keep their coupling) | `assembly._place_masters_at_center_of_mass` | p.53 |
| Auto base restraints | ETABS text import: generated mesh nodes at the base level are fully restrained; mesh nodes generated at a story level join that story's diaphragm; restrained joints are dropped from diaphragms with a warning | `mesher.build` | ETABS behaviour |
| Area local axes | 3 = normal from node order (floors forced to +Z); 1 horizontal (global +X for horizontal elements) | `geometry.shell_axes` | p.185-186 |

## Nonlinear analysis conventions

| Topic | Rule implemented | Where | Source |
|---|---|---|---|
| Hinge sign | hinge moment and rotation in the CSI M3 / M2 sign convention of the member end (positive M3 compresses the +2 face) | `elements.FrameHinge` | CSI Analysis Ref., frame hinges |
| Hinge backbone | B-C-D-E as (plastic deformation, force), optional scale factors; rigid-plastic (stiffness = rigid_factor x 6EI/L) | `builder.NonlinearBuilder.hinge` | CSI hinge properties |
| Hinge states | A-B, B-C, C-D, D-E, >E from the total deformation; performance <IO, IO-LS, LS-CP, >CP from the plastic deformation | `FrameHinge.state_label/performance` | CSI pushover tables |
| Acceleration loads in pushover | F = -M r a (as linear acceleration loads): use a negative scale to push in +X | `runner.nonlinear_static` | CSI acceleration loads |
| Mode load | F = scale x M phi, phi normalised to a unit maximum translation | `runner.nonlinear_static` | CSI "Mode" load |
| Continue from a case | all loads of the previous case stay applied; geometric nonlinearity of the new case | `NonlinearRunner._start` | CSI nonlinear staging |
| Time-history base motion | relative-displacement formulation, uniform excitation; U1/U2 rotated by the load angle | `runner.time_history` | CSI time history |
| Linear TH | zero initial conditions; modal method exact for piecewise-linear loading | `runner.modal_time_history` | CSI linear modal TH |

## Verification status (tests/verification)

Reference: the **ETABS 21 results** of the CSI verification models, exported with
`scripts/Export-VerificationResults.cmd` (CSV in `tests/verification/data/etabs/results`),
compared item by item by `python -m structstudio.verification` and by the test suite.

| Example | Compared | Max. difference vs ETABS 21 |
|---|---|---|
| 2, 4, 6, 7, 8, 9 | every joint displacement, period, frame-force station (P V2 V3 T M2 M3) and base reaction of every static, spectrum and combination case | 0.000 % (Ex.6 periods 0.04 %) |
| 5 | same | 0.000 %; 0.2 % = 5e-4 kip-in numerical moments in pinned braces where ETABS prints 0 |
| 15a1–a9, 15d, 15e, 15f | every wall joint, base reactions | ≤ 3.0 % (most < 1.5 %), reactions exact |
| 15b, 15c1–c4 | wall-on-columns / wall-spandrel joints and frame forces | joints ≤ 2.6 %, frame forces ≤ 6.2 % |
| 3 | periods | ≤ 0.7 % (ETABS uses Ritz vectors); auto UBC97 / ASCE 7-02 / wind loads not implemented |
| 10, 11, 12–14 | — | panel-zone links not implemented; FNA (Wilson, eigen + link Ritz vectors) is implemented and checked against direct integration; Wen / damper links and nonlinear direct-integration time histories are implemented and verified against OpenSees (docs/NONLINEAR.md) |

## Known differences / not yet implemented

* The exact CSI membrane drilling formulation is not public; ours is calibrated to the
  ETABS 21 wall results (≤ 3 % joints, ≤ 6 % spandrel moments).  An Allman-type variant
  (`shell.ALLMAN`) is kept for research but bends a wall on columns under its own
  weight, which ETABS does not.
* Auto lateral loads (Ex.3), link elements, panel-zone links and nonlinear / time-history
  cases (Ex.10–14) are not implemented yet.

* Ritz vectors (eigenvectors used), rigid-response (Gupta f1/f2), CQC3 angle envelope,
  accidental eccentricity, damping scaling of spectra.
* Semi-rigid diaphragms, auto line constraints, edge releases, partial end fixity springs on shells.
* Non-prismatic sections, tension/compression limits, hinges, links, cables, tendons.
* Story drift uses point-to-point differences (max over points), like ETABS "Story Drifts" table,
  but without the "max over average" statistic.
* Membrane floors: ETABS distributes membrane loads to beams tributary-wise without meshing
  out-of-plane DOFs; here membranes are meshed with a tiny out-of-plane stabilisation.

## Importing SAP2000 / SAFE / ETABS tables

Source: the public CSI database-table format (`TABLE:` text files written by SAP2000 v8+,
SAFE v12+, ETABS), the table / field keys of ETABS 21 `Table and Field Keys.xml`, and the
official APIs (`DatabaseTables.GetAvailableTables / GetTableForEditingArray`).

| Behaviour | SAP2000 / CSiBridge | ETABS / SAFE 20+ | SAFE v12–2016 |
|---|---|---|---|
| Grid system after import | X · Y · Z (Z grid lines, else point elevations) | X · Y · Story | X · Y · Z |
| Frames divided at crossing objects | only with *Frame Auto Mesh* (AtJoints/AtFrames) | yes | yes |
| Area objects meshed | only with *Area Auto Mesh* | floors yes, walls one element per panel | yes |
| Default mass source | elements + added mass, all directions, not lumped | lateral, lumped at stories | as SAP2000 |
| g used for weight -> mass | 9.80665 m/s² | 386.4 in/s² / 32.2 ft/s² (ETABS) | 9.80665 m/s² |
| Far ends of columns / walls | as modelled | as modelled (base auto-restraints) | fixed when free (SAFE) |
| Soil / area springs | area springs by tributary area | area springs | subgrade modulus by tributary area (linear) |

* Versions differ in names, e.g. SAP2000 before v14 calls load patterns "load cases"
  (`LOAD CASE DEFINITIONS` with `SelfWtMult`) and load cases "analysis cases"; field `LoadCase`
  instead of `LoadPat`.  Every table / field is looked up through alias lists; ETABS Excel exports
  use display names ("Section Property") that are mapped to keys ("SectProp").
* Not yet imported (reported in the import report): links, solids, cables, non-diaphragm
  constraints, time-history / staged / moving-load cases, auto lateral loads, area / joint
  temperature loads, non-uniform area loads, tendon anchorage-set loss.  Spectrum functions defined by code parameters are used through their
  tabulated Period / Value rows.
* SAFE 22 returns numbers through `DatabaseTables.GetTableForEditingArray` with the trailing
  zeros of whole numbers removed (a 360 in span arrives as "36", E = 3000 ksi as "3"; text
  fields and numbers with decimals are intact).  `csi_convert.ps1` therefore checks each table
  route (editing / display array, editing / display CSV file) against the object API
  (`PointObj.GetCoordCartesian`, `PropMaterial.GetMPIsotropic`) and uses the first that agrees;
  the route is recorded as `TableRoute=` in PROGRAM CONTROL.  When none agrees, coordinates and
  materials are repaired from the object API and `NumbersSuspect=Yes` is written; the importer
  then warns, and it also warns whenever a modulus is below 50 x the material strength.
  Results exported through the `Results` object API are not affected.
* ETABS / SAFE object tables name an area "UniqueName" and use "Area" for the measured area
  (SAFE 22 "Floor Object Connectivity": `UniqueName=2 ... Perimeter=240 Area=3600`); SAP2000
  names it "Area".  The importer takes "UniqueName" when the row has a "Perimeter" (or a numeric
  "Area" beside a "UniqueName").  Before this, every SAFE 22 slab of the same size collapsed into
  one Null area without a section, and 18 of the 40 SAFE verification samples lost their loads.
* SAFE 22 number transfer, measured on the 40 SAFE verification models (S01-S07, thick and
  thin plates): the display route (`TableRoute=DisplayArray`) **rounds decimals** to the display
  format (0.0006944445 -> "0.00069", 1/12 -> "0.0833", about 0.6 % on loads) and, like the editing
  route, **loses the zeros of large whole numbers**: every spring stiffness of 1E+10 arrived as
  "1" (confirmed from SAFE's own reactions / displacements: k = 1.0E+10 kip/in/in on every
  edge).  `csi_convert.ps1` therefore merges the routes value by value (the editing value when
  it has decimals and rounds to the display value: `DecimalsFrom=EditArray`) and reads spring
  properties from the object API (`PropLineSpring` / `PropAreaSpring` / `PropPointSpring`:
  `SpringsFrom=ObjectAPI`; `SpringsSuspect=Yes` when a call fails).  The importer warns about
  SAFE spring stiffnesses equal to 1 unless the springs came from the API.  The API calls were
  written from the CSI API documentation and are not yet run against SAFE here.
* SAFE / ETABS **null lines** are imported as frames of type Null: no stiffness, no mass, no
  section; their loads go to the nodes along them (each piece between two nodes lumped statically
  to its nodes), their line springs and mesh lines are kept.  Before this, a line load on a null
  line (SAFE example 1, LL) was lost.  Two null lines crossing make no joint.
* SAFE **"Stiff" slab property** (column regions): 100 x the membrane and bending stiffness of
  the slab (`csi_import.STIFF_AREA_FACTOR`).  The SAFE manuals give no factor; 100 reproduces
  S05d (1.2 % thick, 5.9 % thin; 42 % with 1 x), larger factors change the results by < 1 %.
  The importer reports it.
* SAFE engine check (`scripts/safe_engine_check.py`, decimals from SAFE's `.$sf` text file and
  spring exponents fitted; worst |Uz| difference over the joints of the structure, relative to
  the largest SAFE |Uz| of the case):

  | family | a (coarse) thick / thin | b thick / thin | c (fine) thick / thin |
  |---|---|---|---|
  | S01 plate, simply supported | 3.2 / 0.04 % | 2.8 / 0.04 % | 1.7 / 0.04 % |
  | S02 plate, clamped | 14.5 / 0.04 % | 7.0 / 0.04 % | 4.5 / 0.04 % |
  | S03 | 1.7 / 0.00 % | 0.7 / 0.00 % | 0.3 / 0.00 % |
  | S04 slab with beams | 1.7 / 0.00 % | 0.7 / 0.00 % | 0.2 / 0.00 % |
  | S05 flat plate on columns | 27 / 0.00 % | 7.0 / 0.00 % | 3.4 / 0.00 % (S05d 1.2 / 0.00 %) |
  | S06 slab on grade | 20 / 0.00 % | 8.7 / 0.00 % | 5.2 / 0.00 % |

  **Thin slabs (DKQ, Nov 2026) match SAFE to the printed digits on every mesh**, including the
  coarse ones, beams in slabs (S04) and soil springs (S06): SAFE's thin plate is a discrete
  Kirchhoff element, and the earlier differences (12.8 / 31 / 33 / 43 % on the coarse meshes)
  came from the penalised-shear MITC4 that stood in for it.  The DKQ also makes beam moment
  diagrams under meshed slabs continuous (no jumps at the slab mesh nodes).  An early version
  with the rotation sign of a left-hand rule matched plates alone but stiffened beams in slabs
  (S04: 21 %); `test_thin_shell_dkq_rotation_sign_with_a_frame` guards it.
  The **thick** column is unchanged (MITC4): still stiffer than SAFE on coarse meshes, most at
  point loads; SAFE's thick-plate element is the next thing to match (the mesh-refinement trend
  is the same as for the old thin results, so the element, not localized meshing, is the likely
  cause).  S06 uses compression-only soil springs, imported as linear; S07a has only its four
  corner joints in the structure (other points are free strip ends that SAFE reports
  interpolated); both programs give |Uz| ~ 1E-21 there, so its joint comparison carries no
  information.  `structstudio.verification` skips joints that are not part of the structure.
* Validation: hand solutions (cantilever, uniform settlement on springs, load totals) and the same
  frame imported as ETABS tables and as SAP2000 tables giving identical displacements
  (`tests/unit/test_import_csi.py`).  Real files from the installed programs are collected with
  `scripts/Export-CsiSamples.cmd`.

## Supports, temperature, foundations, limits, nonprismatic members, tendons

Each rule below was checked against the SAP2000 verification models (results exported from
SAP2000 itself) and, where OpenSees can model it, against OpenSees in the random campaign.

| Feature | Rule | Checked with |
|---|---|---|
| Joint local axes | angles a, b, c: about +3, then the new +2, then the new +1; restraints, springs and ground displacements act in the local axes; joint displacements are reported in the local axes (CSI) | SAP2000 1-005e/f |
| Ground displacement | imposed on restrained DOFs (K_ff u_f = F_f - K_fs u_s); through a spring it moves the spring's ground end (load k dg); ignored on free DOFs; given in GLOBAL or joint LOCAL | SAP2000 1-005a-d, OpenSees `sp` |
| Frame temperature | uniform change: N = -E A alpha T when restrained; gradient across 2 / 3: hot side expands; a joint pattern scales the value at each end (linear between) | SAP2000 1-002a/b |
| Frame gravity load | multiplier x member self weight per length, in a global direction | SAP2000 1-008 |
| Frame springs (Winkler) | consistent element stiffness k integral(N^T N); the foundation reaction enters the base reaction. SAP2000 lumps the springs at the mesh joints, so its unmeshed results (1-013a/b) are coarser than ours; with a fine mesh (1-013c) both agree | SAP2000 1-013c, Hetenyi |
| Tension / compression limits | CSI nonlinear static case: a member past its limit carries the limit force without axial stiffness; iterated on the set of capped members, recovered exactly through an imposed axial strain | SAP2000 1-012b/c |
| Joint offsets | `CoordSys=Local` offsets are member local 1, 2, 3 components | SAP2000 1-011 |
| Nonprismatic sections | A and J vary linearly; I22 / I33 vary linearly, parabolically or cubically (I^1/n linear); shear deformation neglected (as SAP2000); bending stiffness of each element exact from the flexibility integrals of 1/EI | SAP2000 1-006a/b (0.000 %), OpenSees forceBeamColumn |
| Tendons ("Loads") | P(s) = P exp(-(mu alpha + K s)) minus uniform stress losses x area; the tendon pushes +P t / -P t at the anchors and d(P t)/ds along its length, eccentric to the member axis. SAP2000 discretises the tendon into chords (MaxDiscLen) and differs by up to 0.7 % from the continuous solution | SAP2000 1-009a |
| Tendons ("Elements") | bonded axial members between the discretisation points (rigid links to the frame), prestressed by an imposed shortening strain; elastic shortening follows from the analysis. SAP2000's base reactions for this option exclude the tendon members | SAP2000 1-009b/c |

OpenSees cross-check notes: force-based elements (`forceBeamColumn`) need Newton iterations
even when linear (a single `Linear` step leaves multi-element members ~2 % off); a ground
displacement that moves a shell support in its own plane engages the fictitious drilling
stiffness, which is program dependent, and is not compared.

## Load panels (CYPE 3D style surface loads on bars)

A load panel (`PanelLoad`) is a planar polygon (flat or inclined) that carries loads per
pattern and is resolved onto the members at analysis time (`mesh/panel_loads.py`,
`AnalysisModel.panel_report`); the panel itself has no stiffness.

| Rule | Detail |
|---|---|
| one_way | strips parallel to the span direction, simply supported on the bars they cross (exact: each strip load splits between its two supporting bars; strips with one support dump on it) |
| two_way | tributary areas of the receiving bars (nearest bar in the panel plane, sampled on a 64x64 grid with 4x4 sub-samples at the boundary cells, 0.1-0.5 % of the total) |
| shells | pressure on the areas the panel overlaps (the mesher distributes it as an area load) |
| receivers | all bars in the panel plane by default, or the listed frames only (CYPE "receiving frames") |
| direction | Gravity, +-X/Y/Z, projected (per unit horizontal / vertical projection) or any vector `vec:x,y,z` |
| varying | linear with height between `value_lower` (at the lowest point) and `value_upper` (highest): wind pressure, earth / water pressure |
| totals | `applied_total` (panel load) equals `carried_total` (member + shell loads) to the sampling accuracy; checked by `tests/unit/test_panel_loads.py` and a full analysis (base reaction) |

### Area loads are panels (schema 2.1)

Surface loads have one form: the load panel.  An *area load* is a panel made from an area
object (`PanelLoad.area`, `Model.add_area_load` / `add_area_panel`, `model/area_panels.py`).
Joint and frame loads stay on the joints and frames (`Point.loads`, `Frame.loads`).

| Rule | Detail |
|---|---|
| "shells" area panel | exactly the ETABS / SAP2000 uniform area load on that area only (direction passed through, "Normal" = local 3); checked equal to the old storage (`tests/unit/test_area_panels.py`) |
| one_way / two_way area panel | carries the load to the frames around the area's current polygon (also a load-only Null area) |
| edits (`Document.end_edit` -> `Model.sync_panels`) | the panel takes the area's current polygon (move, replicate, reshape); replicate / copy-paste / paste assigns copy the area's panels; Divide Shells gives every piece the area's panels (no load lost); deleting an area deletes its panels |
| drawn panels (`area` None) | load areas of their own: never changed by edits of floors; *Edit > Move* moves selected ones (`tools.move_panels`), Delete deletes them |
| migration | files with schema < 2.1 and every import (ETABS e2k, CSI tables, SAP2000 legacy s2k) move `Area.loads` into "shells" panels, one per area and pattern; `Area.loads` written by scripts are still analysed |
| loads on openings / unmeshed Null areas with "shells" | not analysed (as before); the load report shows them as not carried |

The "as applied to members" load display shows the resolved member loads instead of the
panels and floor loads.
