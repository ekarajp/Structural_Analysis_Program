"""Nonlinear seismic assessment of a two-storey RC frame.

Fiber (force-based) columns with Mander confined concrete, ASCE 41 automatic
beam hinges, rigid diaphragms; gravity staging -> modal pushover ->
ASCE 41 / EC8 N2 target displacement -> nonlinear time history.
Run: python examples/nonlinear_rc_frame.py
"""
import numpy as np

from structstudio import Model, analyze
from structstudio.model import FIXED, FrameSection
from structstudio.model.nonlinear import (FiberSectionDef, FrameNonlinearProperty, HingeProperty, NonlinearMaterial,
                                          TimeHistoryLoad)
from structstudio.seismic import GroundMotionRecord, asce41_target_displacement, response_spectrum

m = Model("rc-frame", "kN m")
m.add_material("C30", E=27e6, nu=0.2, weight_density=24, fc=30e3)
m.add_frame_section(FrameSection.rectangular("C40", "C30", depth=0.4, width=0.4))
m.add_frame_section(FrameSection.rectangular("B30x60", "C30", depth=0.6, width=0.3))
m.set_stories([("L1", 3.5), ("L2", 3.2)])
m.add_diaphragm("D1")
xs, ys = (0, 6, 12), (0, 5)
for x in xs:
    for y in ys:
        m.add_point(x, y, 0, restraints=FIXED)
for z, st, h in ((3.5, "L1", 3.5), (6.7, "L2", 3.2)):
    for x in xs:
        for y in ys:
            p = m.add_point(x, y, z, diaphragm="D1")
            m.add_frame((x, y, z - h), p, "C40", name=f"C{x}{y}{st}")
    for y in ys:
        for x0, x1 in zip(xs[:-1], xs[1:]):
            m.add_frame((x0, y, z), (x1, y, z), "B30x60", name=f"BX{x0}{y}{st}")
    for x in xs:
        m.add_frame((x, 0, z), (x, 5, z), "B30x60", name=f"BY{x}{st}")
m.add_load_pattern("DL", "Dead", self_weight_multiplier=1.0)
m.add_load_pattern("SDL", "SuperDead")
for f in m.frames.values():
    if f.name.startswith("B"):
        m.add_frame_distributed_load(f.name, "SDL", 20.0)
m.mass_sources["MsSrc1"].load_patterns = {"DL": 1.0, "SDL": 1.0}

# --- nonlinear properties --------------------------------------------------
m.add_nonlinear_material(NonlinearMaterial.confined_concrete_mander("Core", 30e3, 400e3, 0.012))
m.add_nonlinear_material(NonlinearMaterial.unconfined_concrete("Cover", 30e3))
m.add_nonlinear_material(NonlinearMaterial.rebar("SD40", 460e3, 200e6, 0.01))
m.add_fiber_section(FiberSectionDef("C40F", "RCRectangle", frame_section="C40", cover=0.05, core_material="Core",
                                    cover_material="Cover", steel_material="SD40", n_top=3, n_bottom=3, n_side=1,
                                    bar_area=4.91e-4, mesh=(10, 6)))
m.add_frame_nonlinear(FrameNonlinearProperty("COL", "C40F", "ForceBased", "Lobatto", 5),
                      [f for f in m.frames if f.startswith("C")])
m.add_hinge_property(HingeProperty("BEAM-M3", "M3", auto="ASCE41-RCBeam", hysteresis="Takeda",
                                   auto_params={"My": 260.0, "rho_ratio": 0.1, "conforming": 1, "shear_index": 3}))
for f in m.frames:
    if f.startswith("B"):
        m.assign_hinge(f, "BEAM-M3", 0.0)
        m.assign_hinge(f, "BEAM-M3", 1.0)

# --- cases -----------------------------------------------------------------
roof = m.find_point(12, 5, 6.7).name
m.add_modal_case("Modal", max_modes=9)
m.add_nonlinear_static_case("GRAV", {"DL": 1.0, "SDL": 1.0}, n_steps=5)
res0 = analyze(m, ["Modal"])
mode_x = int(max(res0.modal_participating_mass_ratios("Modal"), key=lambda r: r["UX"])["Mode"])
m.add_nonlinear_static_case("PUSH-X", {}, mode_loads=[("Modal", mode_x, 1.0)], initial_case="GRAV",
                            control="Displacement", monitored_point=roof, monitored_dof="UX",
                            target_displacement=0.25, n_steps=50)
rng = np.random.default_rng(2)
t = np.arange(0, 8, 0.01)
ag = sum(a * np.sin(2 * np.pi * f * t + ph) for f, a, ph in zip(rng.uniform(0.5, 6, 20), rng.uniform(0.2, 1, 20),
                                                                rng.uniform(0, 6, 20)))
ag = ag * np.minimum(1, t / 1.5) * np.exp(-np.maximum(t - 5, 0))
ag *= 0.35 / np.abs(ag).max()  # synthetic record, PGA 0.35 g
m.add_time_history_function("GM", t, ag)
m.add_time_history_case("NLTH", [TimeHistoryLoad("Acceleration", "U1", "GM", 9.80665)], n_steps=len(t) - 1,
                        dt=0.01, initial_case="GRAV", damping="Rayleigh", damping_ratio=0.05)

res = analyze(m)
print("periods (s):", [round(r["Period"], 3) for r in res.modal_periods("Modal")[:3]])
curve = res.pushover_curve("PUSH-X")
for row in curve[::10]:
    print(f"step {row['Step']:3d}  d = {row['Displacement']:.3f} m  V = {row['BaseForce']:8.1f} kN")
d = np.array([r["Displacement"] for r in curve])
V = np.array([r["BaseForce"] for r in curve])
T1 = res.modal_periods("Modal")[mode_x - 1]["Period"]
spec = lambda T: 2.5 * 0.35 if T < 0.6 else 2.5 * 0.35 * 0.6 / T  # illustrative design spectrum (g)
ta = asce41_target_displacement(d, V, weight=abs(res.base_reactions("GRAV")[2]), Sa_of_T=spec, Ti=T1, n_stories=2)
print("ASCE 41 target displacement: %.3f m (C0 %.2f C1 %.2f C2 %.2f, Te %.3f s)"
      % (ta["target_displacement"], ta["C0"], ta["C1"], ta["C2"], ta["Te"]))
_, ux = res.time_history("NLTH", roof, "UX")
print("NLTH roof: peak %.1f mm, residual %.1f mm" % (1e3 * np.abs(ux).max(), 1e3 * ux[-1]))
print("NLTH peak story drifts:", [(r["Story"], round(r["MaxDrift"], 4)) for r in res.story_drift_history("NLTH")
                                  if r["Direction"] == "X"])
rec = GroundMotionRecord(0.01, ag)
print("record Sa(T1) = %.3f g" % response_spectrum(rec, [T1])["Sa"][0])
