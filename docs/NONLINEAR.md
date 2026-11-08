# Nonlinear and seismic analysis

> **สรุปภาษาไทย** — เพิ่ม engine วิเคราะห์ไม่เชิงเส้นแบบ OpenSees เข้าไปใน Structural Studio:
> วัสดุ uniaxial 8 ชนิดที่ให้ผลตรงกับ OpenSees 3.8 ระดับ machine precision, fiber section,
> force-based / displacement-based beam-column, ข้อหมุนพลาสติกแบบ ETABS (backbone B-C-D-E,
> IO/LS/CP, ASCE 41 auto hinge), link (Wen isolator, viscous damper), P-Delta,
> Newton / line search / Anderson-accelerated Newton / arc-length, load & displacement control,
> Newmark / HHT, Rayleigh / modal damping, modal time history แบบ exact และเครื่องมือ
> แผ่นดินไหว (spectrum, ASCE 41 / EC8 N2 target displacement, IDA, fragility MLE)
> ทุกอย่างกำหนดผ่าน Model แบบ ETABS (load case, hinge assignment) และตรวจกับ OpenSees
> โดยตรงในชุดทดสอบ (`tests/xcheck/test_nonlinear_xcheck.py`)

## 1. What is implemented

| Area | Implementation | Module |
|---|---|---|
| Uniaxial materials | Elastic, ElasticPP, Steel01, Steel02 (GMP + isotropic hardening), Concrete01 (Kent-Scott-Park), Concrete02 (tension softening), Hysteretic (pinching, damage), BoucWen, Viscous, Parallel; CSI hinge law (Kinematic / Takeda / nonlinear elastic) | `engine/nonlinear/materials.py` |
| Material helpers | Mander confined concrete (fitted to Concrete02), cover concrete, reinforcing steel | `model/nonlinear.NonlinearMaterial` |
| Fiber sections | rectangular / circular RC, I, box, pipe, custom fibers; OpenSees patch / layer discretisation; vectorised per element | `engine/nonlinear/sections.py` |
| Beam integration | Gauss-Lobatto, Gauss-Legendre, Gauss-Radau, HingeRadau (Scott & Fenves 2006) | `sections.py` |
| Frame elements | force-based (iterative state determination with subdivision, under-relaxation at kinks), displacement-based, elastic member + CSI hinges (M3, M2, P), fiber P-M2-M3 hinges (force-based member with HingeRadau over the hinge length) | `engine/nonlinear/elements.py`, `builder.py` |
| Geometric nonlinearity | Linear, P-Delta (leaning-column string stiffness on every member, elastic or nonlinear), Corotational 3D large displacements and rotations (Crisfield / de Souza, OpenSees `Corotational` local rotations; every frame, elastic ones included) | `engine/nonlinear/corotational.py`, `elements.py`, `domain.py` |
| Links | zero-length two-joint or grounded links, one material per DOF (Linear k + c, Wen plastic, ElasticPP, MultiLinear, Damper) | `elements.Link`, `builder.py` |
| Domain | same DOF numbering and rigid-diaphragm constraints as the linear engine; shells / elastic members in one constant sparse matrix; fixed sparsity pattern | `engine/nonlinear/domain.py` |
| Static solution | load control, displacement control, arc-length continuation (Crisfield) through strength drops; CSI-style vertical hinge drops released at a fixed control displacement (load redistribution); Newton, NewtonLineSearch, Anderson-accelerated modified Newton, ModifiedNewton, InitialNewton; automatic step subdivision | `engine/nonlinear/analysis.py` |
| Transient solution | Newmark (γ, β), HHT-α; Rayleigh damping on initial / committed / current stiffness; constant modal damping (low rank, Woodbury solve); uniform base excitation in 3 directions; time-dependent load patterns | `analysis.py` |
| Linear time history | modal superposition, exact for piecewise-linear excitation (Nigam & Jennings); linear direct integration | `runner.py` |
| FNA | fast nonlinear analysis (Wilson): linear structure + nonlinear links, eigenvectors + Ritz vectors of the link deformations and ground loads, exact modal integration, Newton iteration on the modal link forces | `runner.fna_time_history` |
| Load cases | `NonlinearStaticCase` (gravity staging, pushover by load pattern, uniform acceleration or mode shape, continue from another case), `TimeHistoryCase` (linear / nonlinear, modal / direct) | `model/nonlinear.py` |
| Results | pushover curve with hinge state counts (A-B ... >E) and performance ranges (<IO ... >CP), hinge table, joint time histories (relative / absolute), base reaction history, envelopes, story drift history; station forces of nonlinear members from equilibrium | `results/__init__.py` |
| Seismic tools | PEER AT2 reader, baseline correction, intensity measures (PGA, PGV, PGD, Arias, D5-95, CAV), elastic spectra, scaling to a target spectrum, ASCE 41 bilinearisation and coefficient method, EC8 N2, IDA driver, lognormal fragility (moments and MLE) | `seismic/` |

## 2. Verification against OpenSees 3.8

All comparisons run the *same* model in both programs and compare every
step.  The material, element and solver checks are part of the test suite
(`pytest tests/xcheck`, skipped when `openseespy` is not installed).

| Check | Quantity compared | Max. difference (relative to peak) |
|---|---|---|
| Steel01, Steel02, Concrete01, Concrete02, ElasticPP, Hysteretic, BoucWen | stress and tangent over random cyclic strain histories | 0 – 8e-16 |
| RC fiber column, force-based Lobatto(5), P-Delta, through softening | base shear at every pushover step | 6.4e-9 |
| RC fiber column, displacement-based Legendre(3) | base shear | 4.2e-9 |
| RC fiber portal frame, Newmark, Rayleigh (initial K), 1500 steps | roof displacement history | 2.3e-11 |
| Same, Rayleigh on current / committed tangent | roof displacement | 7.7e-13 / 3.9e-8 |
| Same, HHT α = 0.9 | roof displacement | 1.4e-4 |
| Hinged portal frame (bilinear kinematic and Hysteretic hinges), Linear and P-Delta | base shear at every pushover step | 5e-16 – 3.2e-12 |
| Hinged portal frame, time history (mass-proportional damping) | roof displacement, all steps OpenSees solves with plain Newton | 3.4e-13 |
| Wen isolator + nonlinear viscous damper | displacement history | 9.8e-14 |
| FNA vs our Newmark direct integration, base-isolated 12-storey stick, 2 eigenvectors + Ritz | joint displacement history | 1.3e-3 (3.5e-4 with 4 eigenvectors; dt error of Newmark) |
| 3D two-storey building, rigid diaphragms, fiber columns, elastic beams | pushover base shear | 2.3e-12 |
| 3D L-frame, Corotational, 0.5 rad / 1.5 m tip motion, elastic and fiber members | joint displacements and rotations | 1.2e-7 / 6.7e-8 |
| Planar cantilever, Corotational, 10 / 20 / 40 elements | tip displacements | 9.4e-10 / 1.5e-11 / 2.4e-13 |
| Same with hinged beams | pushover base shear | 2.7e-11 |
| Same, nonlinear time history with P-Delta | roof displacement | 5.1e-11 |

Notes on the comparisons:

* Where OpenSees needs its fallback algorithms (KrylovNewton, ModifiedNewton)
  with a displacement-increment test, its unbalanced force may remain large,
  and the histories drift apart by 1–60 %.  Up to that step the programs agree
  to 1e-11.  Our solver tests force unbalance *and* displacement increment.
* Stiffness-proportional Rayleigh damping differs by construction when hinges
  are zero-length springs with their own internal nodes in OpenSees (springs
  damped in series) and condensed into the member here (damping of the combined
  stiffness).  With mass-proportional damping the two agree to 3e-13.
* OpenSees cannot run displacement control on a rigid-diaphragm slave DOF; the
  test controls the master instead (the load is torsion free).
* Concrete02 in OpenSees returns the previous trial stress when the strain does
  not change; we return the committed state.  With a reloading focus point in
  compression (εr < 0) the Concrete02 unloading rule is discontinuous in both
  programs; the material reports it and the Mander helper chooses λ so that
  εr ≥ 0.

Speed: the Python engine needs about 9 times the OpenSees run time on the
fiber portal frame (9.2 s vs 1.0 s for 1500 steps).  The compiled kernel of
Phase 5 targets this gap.

## 3. Comparison with ETABS / SAP2000 / OpenSees (current state)

| Capability | ETABS / SAP2000 | OpenSees | Structural Studio |
|---|---|---|---|
| Fiber force-based members | fiber hinges only (P-M2-M3 at a point) | yes | yes, identical to OpenSees |
| Concentrated M3 / M2 / P hinges, ASCE 41 auto | yes | via zeroLength + material | yes (ASCE 41 RC and steel beams) |
| P-M-M interaction hinges | yes | via fiber sections | fiber P-M2-M3 hinges (`HingeProperty(dof="PMM", fiber_section=...)`); interaction-surface hinges not yet |
| Strength-drop handling | event-to-event + unloading methods | user scripts | vertical drops released at fixed displacement (static), regularised drops in time history (see §4) |
| Pushover: load / displacement / mode / acceleration | yes | yes (scripts) | yes |
| Direct-integration NLTH, HHT, Rayleigh | yes | yes | yes |
| FNA (fast nonlinear analysis) | yes | no | yes (links nonlinear, frames linear, as CSI) |
| Large displacements (corotational) | yes | yes | yes, matches OpenSees `Corotational` (shells stay small-displacement) |
| Linear modal TH (exact) | yes | no (scripts) | yes |
| IDA, fragility, target displacement | no | scripts | built in (`seismic/`) |
| Verification against OpenSees inside the test suite | — | — | yes |

## 4. Modelling choices and known differences

* **Rigid-plastic hinges.**  As in CSI products, hinges have no elastic
  flexibility of their own: the hinge elastic stiffness is `rigid_factor`
  (default 1000) × 6EI/L of the member, so the member flexibility is changed by
  about 0.1 %.  Results with factors 50 and 1000 differ by 0.03 % on the
  building example.
* **Load drops.**  CSI drops the hinge force vertically (C→D, after E).
  Static cases do the same by default (`hinge_unloading="Redistribute"`): a
  hinge that passes a drop keeps the strength it had before the drop until the
  step has converged; the drop is then released in sub-steps (25 %, 50 %,
  100 %, halved when a sub-step fails) with the control displacement (or, under
  load control, the load) held, and every sub-step is a point of the capacity
  curve.  The curve therefore shows vertical steps as in ETABS, and the other
  hinges pick up the released load.  Under load control a drop that the
  structure cannot redistribute stops the case (mechanism) with a warning.
  The drop is triggered at the end of the step in which the hinge passed point
  C, so a coarse step can overshoot C along the plateau; use enough steps
  (ETABS stops exactly at the event).  Time history cannot hold time still, so
  drops keep a finite slope `k_member / drop_regularisation` (default 10, the
  force drops over a plastic rotation of 10 ΔF / (4EI/L)); the same applies to
  static cases with `hinge_unloading="Regularised"` (the previous behaviour).
* **Corotational.**  Large displacements and rotations of every frame member
  (nodal triads updated by the exponential map, basic system as OpenSees
  `CorotCrdTransf3d`, consistent tangent from complex-step derivatives).
  Members with joint offsets / rigid zones are not supported in this mode, and
  shells remain small-displacement (with a warning).
* **FNA.**  `TimeHistoryCase(method="FNA", modal_case=...)`: the number of
  eigenvectors is `max_modes` of the modal case (12 without one); one Ritz
  vector per nonlinear link DOF and per ground direction is added.  Damping
  is modal (constant ratio, or the Rayleigh coefficients converted per vector;
  ratios are limited to 0.999).  Frame hinges and fiber members are linear in
  FNA.  An FNA case can only continue from another FNA case, so gravity is
  applied as a ramped load in the case, as in CSI.  Link deformation / force
  histories are in `history["links"]`.
* **Fiber P-M2-M3 hinges.**  A PMM hinge turns the member into a force-based
  element with HingeRadau integration: the fiber section over the hinge length
  `lp` (default half the depth) at each hinged end, elastic in between; hinge
  rotations reported are curvature × lp at the end sections.
* **Hinge location.**  Hinges sit at the ends of the flexible span of the
  element nearest to the assigned relative distance (a warning is issued when
  that is more than 2 % away); interior hinges need a mesh point.
* **Fiber members and rigid zones.**  Fiber members deform over the flexible
  length in every direction; CSI's linear frame lets axial and torsion act
  over the full length (the hinged elastic member follows CSI).  End releases
  are not supported on fiber members (the member stays elastic, with a
  warning).
* **Shear deformation** is included in elastic and hinged members (Timoshenko)
  but not in fiber members (Euler-Bernoulli sections, as OpenSees fiber
  sections without a shear aggregator).
* **Damping in time history.**  Default Rayleigh damping on the initial
  stiffness, ζ at T1 and at the period where 90 % of the lateral mass
  participates (from an eigen solution of the initial stiffness); explicit
  coefficients, periods, the committed / current tangent or constant modal
  damping can be chosen.
* **Reactions** in nonlinear cases are the element forces at the supports
  (damping forces of stiffness-proportional damping are not added).
* **Linear time history** starts from zero initial conditions (CSI rule).

## 5. Using it

```python
from structstudio.model.nonlinear import *
m.add_nonlinear_material(NonlinearMaterial.confined_concrete_mander("Core", 30e3, 400e3, 0.012))
m.add_fiber_section(FiberSectionDef("C40F", "RCRectangle", frame_section="C40", core_material="Core", ...))
m.add_frame_nonlinear(FrameNonlinearProperty("COL", "C40F", "ForceBased", "Lobatto", 5), columns)
m.add_hinge_property(HingeProperty("BH", "M3", auto="ASCE41-RCBeam", auto_params={"My": 260.0}))
m.assign_hinge("B1", "BH", 0.0)
m.add_nonlinear_static_case("GRAV", {"DL": 1.0, "SDL": 1.0}, n_steps=5)
m.add_nonlinear_static_case("PUSH", {}, mode_loads=[("Modal", 1, 1.0)], initial_case="GRAV",
                            control="Displacement", monitored_point=roof, target_displacement=0.25, n_steps=50)
res = analyze(m)
res.pushover_curve("PUSH"); res.hinge_results("PUSH")
```

A complete example: `examples/nonlinear_rc_frame.py`.

## 6. References

* Filippou, Popov & Bertero (1983), EERC 83/19 — Steel02.
* Kent & Park (1971); Scott, Park & Priestley (1982) — Concrete01.
* Mohd Yassin (1994) — Concrete02.
* Mander, Priestley & Park (1988), J. Struct. Eng. 114(8) — confined concrete.
* Spacone, Filippou & Taucer (1996); Neuenhofer & Filippou (1997) — force-based elements.
* Scott & Fenves (2006), J. Struct. Eng. 132(2) — HingeRadau.
* Scott & Fenves (2010), J. Struct. Eng. 136(5) — Krylov subspace accelerated Newton.
* Crisfield (1981), Comput. Struct. 13 — arc-length method.
* Hilber, Hughes & Taylor (1977) — HHT-α.
* Nigam & Jennings (1969), BSSA 59 — exact piecewise-linear integration.
* Zareian & Medina (2010), Comput. Struct. 88 — damping of concentrated-plasticity models.
* ASCE/SEI 41-17 — modelling parameters, acceptance criteria, coefficient method.
* EN 1998-1:2004 Annex B — N2 method.
* Vamvatsikos & Cornell (2002), EESD 31 — incremental dynamic analysis.
* Baker (2015), Earthquake Spectra 31(1) — fragility fitting.
* CSI Analysis Reference Manual — hinge definitions, nonlinear cases.
