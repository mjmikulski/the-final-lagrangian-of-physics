"""Confirmation that the JG_E well is the predicted physics, not
relaxation noise. From E(omega) ~ -2 g C1 omega^2 + g C2 omega^4: the
minimum position omega_E = sqrt(C1/C2) is INDEPENDENT of gamma, while
the well depth g C1^2/C2 scales linearly. Rerun the JG_E ladder at
gamma x4 (20% statics deformation), same convergence criterion as the
main ladders (Adam + L-BFGS to |g|_inf < 2e-4), and check both.

Needs report 004's polished field (or M5_FIELDS_DIR) plus
results/i1sq_ladders.json. Out: results/gamma_scaling.json
"""
import json
import os
import runpy
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
MP = os.path.join(FIELDS, "M_G_polished.npz")
FLAG = os.path.join(HERE, "results", "gamma_scaling_ran.flag")
if not os.path.exists(MP):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("confirm_gamma_scaling: NOT REPRODUCED HERE -- needs report "
          f"004's polished field in {FIELDS} (or M5_FIELDS_DIR).")
    sys.exit(0)

lad = runpy.run_path(os.path.join(HERE, "ladder_i1sq_defs.py"))
run_rungs, base = lad["run_rungs"], lad["load_base"](HERE)
gamma4 = 4.0 * base["gamma"]
print(f"gamma x4 = {gamma4:.4f} (20% statics deformation); prediction: "
      f"same omega_E = {base['omega_pred_E']:.3f}, depth x4")

rows = run_rungs("gx4", gamma4, "energy", (0.0, 0.2, 0.35, 0.5, 0.8))
k = min(range(len(rows)), key=lambda i: rows[i]["E_total"])
depth = rows[0]["E_total"] - rows[k]["E_total"]
base_rows = {r["omega"]: r["E_total"] for r in base["JG_E"]["rungs"]}
base_depth = base_rows[0.0] - min(base_rows.values())
ratio = depth / max(base_depth, 1e-12)
print(f"minimum at omega = {rows[k]['omega']} "
      f"(interior: {0 < k < len(rows)-1}); depth {depth:.5f} vs base "
      f"{base_depth:.5f} (ratio {ratio:.2f}, predicted ~4)")
out = {"gamma4": gamma4, "rows": rows, "min_omega": rows[k]["omega"],
       "interior": bool(0 < k < len(rows) - 1), "depth": depth,
       "base_depth": base_depth, "depth_ratio": ratio,
       "max_grad_inf": max(r["grad_inf"] for r in rows)}
json.dump(out, open(os.path.join(HERE, "results",
                                 "gamma_scaling.json"), "w"), indent=1)
with open(FLAG, "w") as f:
    f.write("gamma scaling computed in this run\n")
print("written: results/gamma_scaling.json")
