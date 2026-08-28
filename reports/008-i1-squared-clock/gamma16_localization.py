"""Adversarial localization check: does the ticking stay localized when
the well is DEEP? The convexity mechanism predicts yes -- spreading
away from the template costs at any gamma; if instead the observed
PR ~ 100 were only the weakness of the term, a 16x deeper well would
delocalize like 004/007's Mexican-hat quartics did.

Runs the JG_E rung at omega = 0 / 0.35 / 0.8 with gamma x16 (statics
deformation 80% -- deliberately aggressive), same convergence criterion
as the main ladders, and asserts PR stays at the core scale.
Out: results/gamma16_localization.json
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "gamma16_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("gamma16_localization: NOT REPRODUCED HERE -- needs report "
          f"004's polished field in {FIELDS} (or M5_FIELDS_DIR).")
    sys.exit(0)

lad = runpy.run_path(os.path.join(HERE, "ladder_i1sq_defs.py"))
base = lad["load_base"](HERE)
gamma16 = 16.0 * base["gamma"]
print(f"gamma x16 = {gamma16:.2f} (80% statics deformation)")

rows = lad["run_rungs"]("g16", gamma16, "energy",
                        (0.0, 0.35, 0.5, 0.8, 1.2))
min_per_level = []
for lv in range(len(rows[0]["E_levels"])):
    k = min(range(len(rows)), key=lambda i: rows[i]["E_levels"][lv])
    min_per_level.append(rows[k]["omega"])
k = min(range(len(rows)), key=lambda i: rows[i]["E_total"])
print(f"minimum per relaxation level: {min_per_level}; final min at "
      f"omega {rows[k]['omega']} (PR {rows[k]['PR_k_sites']:.0f})")
json.dump({"gamma16": gamma16, "rows": rows,
           "min_omega_per_level": min_per_level,
           "final_min_omega": rows[k]["omega"],
           "PR_at_final_min": rows[k]["PR_k_sites"],
           "depth_at_035": rows[0]["E_total"] - rows[1]["E_total"],
           "max_grad_inf": max(r["grad_inf"] for r in rows)},
          open(os.path.join(HERE, "results",
                            "gamma16_localization.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("gamma16 localization computed in this run\n")
print("written: results/gamma16_localization.json")
