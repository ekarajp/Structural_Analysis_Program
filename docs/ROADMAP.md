# Roadmap

เป้าหมาย: โปรแกรมวิเคราะห์/ออกแบบอาคารระดับสากล (ETABS + SAFE + SAP2000 + IDEA StatiCa +
optimization + BIM + drawings + AI agent) — สร้างบนสถาปัตยกรรมเดียวตั้งแต่ต้น แล้วเพิ่มทีละเฟส

## Phase 1 — Core engine (ทำแล้วในรอบนี้) ✅
- Object model แบบ ETABS (Story/Grid/Point/Frame/Area/Diaphragm, properties, patterns, cases, combos)
- Mesher: insertion points, end offsets/rigid zones, auto-mesh ที่จุด/จุดตัด/โหนดพื้น, quad meshing, rigid diaphragm
- Engine: Timoshenko frame (releases, partial loads, station forces), MITC4+QM6 shell, static, modal, RS, P-Delta
- Results API หน่วยตามโมเดล, combos แบบ CSI, e2k importer, CLI, verification tests (CSI Ex.1, 2, 4) ผ่าน

## Phase 2 — Desktop (PySide6 + VTK)
- [x] 3D viewport (VTK): วาด/เลือก/snap grid, plan/elevation, story filter, column markers
- [x] Draw beam/column/wall/slab + quick draw, window selection, delete, undo/redo
- [x] New Model wizard, grid/stories editor, Define (materials, sections, CSI libraries, loads, cases, combos)
- [x] Assign (restraints, diaphragms, sections, releases, point/frame/area loads)
- [x] Run analysis, deformed shape, diagrams, result tables, .e2k import, .ssj project files
- [ ] Shell contours (M11/M22/V13...), reactions display, load display, tables export (xlsx/csv)
- [ ] Replicate / move / edit-point tools, elevation-view drawing (braces), extrude view with sections
- [ ] Report export (PDF/docx), units switch in UI, Thai UI strings

## Phase 3 — Loads & RC design
- Auto load combos (ACI/มยผ./EIT), live-load reduction, wind (มยผ.1311), seismic static & RS (มยผ.1301/1302)
- RC beam/column/wall design ACI 318-19/25 + มยผ. 1301/1302; demand ดึงจาก analysis อัตโนมัติ
- Steel design AISC 360 (เริ่มจาก beam/column/brace)

## Phase 4 — Slab & foundation (SAFE-like)
- Slab FE design strips, punching shear, deflection (cracked), mat foundation + soil springs, pile caps

## Phase 5 — Advanced analysis & kernel
- [x] Nonlinear engine แบบ OpenSees (ตรวจกับ OpenSees 3.8 ในชุดทดสอบ, ดู docs/NONLINEAR.md):
      วัสดุ Steel01/02, Concrete01/02, Hysteretic, BoucWen, Viscous; fiber section; force/disp-based
      beam-column; CSI hinges (ASCE 41 auto); links; P-Delta
- [x] Nonlinear static: gravity staging, pushover (load / mode / acceleration), displacement control,
      arc-length ผ่านจุดที่กำลังตก, hinge states + IO/LS/CP
- [x] Time history: nonlinear direct (Newmark/HHT, Rayleigh/modal damping), linear modal (exact), linear direct
- [x] เครื่องมือแผ่นดินไหว: AT2 reader, spectrum, ASCE 41 / EC8 N2 target displacement, IDA, fragility MLE
- [x] Hinge ตกกำลังแนวดิ่งแบบ ETABS ใน static (ปล่อยแรงทีละขั้นที่ control displacement คงที่, redistribution);
      time history ใช้ regularised drop
- [x] Corotational 3D (large displacement/rotation) ตรงกับ OpenSees 1e-7; fiber P-M2-M3 hinges (HingeRadau)
- [x] FNA (Wilson): eigen + Ritz vectors ของ link, exact modal integration, Newton บนแรง link
- [ ] Event-to-event แท้ (หยุดตรงจุด C พอดี), P-M2-M3 interaction-surface hinges, shear hinges, FNA แบบ compiled
- [ ] วัสดุเพิ่ม: Concrete04/07 (Chang-Mander), IMK / Pinching4, Steel4; staged construction
- Compiled kernel (Rust/C++): assembly + PARDISO/MKL; target 500k DOF
- Connections (StatiCa-like) CBFEM

## Phase 6 — Integration
- BIM: IFC import/export, Revit link; drawings: DXF/DWG beam & column schedules
- Optimization (section sizing, cost), batch/fragility API สำหรับงานวิจัย
- MCP server สำหรับ AI agent: model query/edit, run, read results, design

## Quality gates (ทุกเฟส)
- ทุก feature ที่เลียนแบบ ETABS ต้องมีแหล่งอ้างอิงใน docs/CONVENTIONS.md
- Verification examples จาก CSI + ETABS จริง (EtabsBridge) ก่อนประกาศว่า "ตรง ETABS"
- ไม่มีโค้ดจาก decompile ใน product
