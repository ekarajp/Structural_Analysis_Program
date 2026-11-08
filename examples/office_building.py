"""Two-storey RC office box: slabs with rigid diaphragms, self weight, live load,
modal analysis and a load combination.  Run: python examples/office_building.py"""
from structstudio import Model, analyze
from structstudio.model import FIXED, FrameSection, ShellSection

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
    m.add_area_load(a, "LL", 2.5)  # kN/m2, gravity
m.add_linear_static_case("DL")
m.add_linear_static_case("LL")
m.add_modal_case("Modal", max_modes=6)
m.add_combination("U1", {"DL": 1.4, "LL": 1.7})

res = analyze(m)
print(res.am.summary())
for row in res.modal_periods("Modal")[:3]:
    print(row)
print("base reactions DL:", res.base_reactions("DL"))
beam = next(f for f in m.frames.values() if f.type == "Beam").name
for row in res.frame_forces(beam, "U1").table():
    print(f"{row['Station']:6.2f}  V2={row['V2']:8.2f}  M3={row['M3']:8.2f}  (min {row['M3Min']:8.2f})")
m.to_json("office.ssj.json")
