# Architecture — "Python-first, compiled-kernel-ready"

```
structstudio/
  units.py            unit systems (kN-m, kip-in, ...) — the engine is SI inside
  model/              OBJECT MODEL — what the engineer draws
    properties.py     Material, FrameSection (+modifiers), ShellSection
    objects.py        Point, Frame (beam/column/brace), Area (floor/wall), Story, Grid, Diaphragm
    loads.py          LoadPattern, MassSource, functions, LoadCase (static/modal/RS/P-Delta), LoadCombination
    model.py          Model container, validation, JSON persistence, unit conversion
  mesh/               OBJECT MODEL -> ANALYSIS MODEL
    analysis_model.py Node, FrameElement, ShellElement, Constraint (plain SI data)
    mesher.py         local axes, insertion points, end offsets/rigid zones, auto-mesh at
                      points / intersections / floor mesh nodes, quad meshing, rigid diaphragms
    geometry.py       vector helpers (CSI local axes rules, polygon meshing, inertia)
  engine/             FINITE ELEMENT ENGINE (SI)
    elements/frame.py Timoshenko/Euler frame, releases + partial fixity, rigid zones,
                      consistent partial trapezoidal loads, station forces, geometric stiffness
    elements/shell.py MITC4 Mindlin plate + QM6 membrane, modifiers, lumped mass, loads
    assembly.py       DOF numbering, master-slave diaphragm transformation, K, M, F
    kernel.py         *** the swap point for a compiled kernel ***  factor/solve + eigen
    solvers.py        linear static, P-Delta (iterative initial stress), modal, response spectrum
    results.py        raw CaseResult / ModalData
    nonlinear/        NONLINEAR ENGINE (see docs/NONLINEAR.md)
      materials.py    vectorised uniaxial materials (Steel01/02, Concrete01/02, Hysteretic, BoucWen, hinge laws)
      sections.py     fiber sections, beam integration, stacked per-element section blocks
      elements.py     force-/displacement-based beam-columns, elastic member + CSI hinges, links
      domain.py       nonlinear domain on the linear DOF map (diaphragms), fixed sparsity pattern
      analysis.py     Newton family, Anderson acceleration, load/displacement/arc-length control, Newmark/HHT
      builder.py      model assignments -> elements;  hinges.py  ASCE 41 hinges, moment-curvature
      runner.py       NonlinearStatic / TimeHistory cases (direct and exact modal)
  results/            RESULT QUERIES in model units: joints, frame station forces, modal tables,
                      story drift, diaphragm displacement, CSI-style load combinations (max/min)
  seismic/            records (PEER AT2), spectra, ASCE 41 / N2 target displacement, IDA, fragility
  io/e2k.py           ETABS text file (.e2k / .$et) importer
  api/                analyze(model) — the boundary used by CLI, desktop app and AI agents
  cli.py              structstudio info | import | analyze | verify
tests/
  unit/               closed-form physics checks of the engine
  verification/       CSI ETABS verification manual examples (acceptance tests)
```

## Layer rules

1. `model` knows nothing about finite elements; `engine` knows nothing about objects.
   Only `mesh` connects the two.  This is what makes slab design, auto meshing,
   drawings and BIM possible without touching the solver.
2. Everything the engine sees is SI numpy data.  A Rust/C++ kernel replaces
   `engine/kernel.py` (and later `engine/elements/*`) without changing anything else.
3. Results are queried through `results.Results` in the model's units.  UI and agents never
   touch raw arrays.
4. Every behaviour that imitates ETABS is documented in `docs/CONVENTIONS.md` with the
   public source it comes from (CSI Analysis Reference Manual, verification manual).
   No decompiled code is used anywhere.

## Verification strategy

* `tests/unit` — closed-form mechanics (cantilevers, fixed-end moments, Timoshenko,
  Navier plate, SDOF spectrum, beam-column amplification...).
* `tests/verification` — the CSI verification examples rebuilt from the manual, compared
  against the *theoretical* columns of the manual tables (Examples 1, 2, 4 today).
* EtabsBridge captures (`..\evidence\*`) can be imported through their `.$et` text model and
  compared table by table.
