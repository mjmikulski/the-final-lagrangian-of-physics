"""The fundamental-Lagrangian reading of +gamma*(I1_G)^2 is measured to
be statically UNSTABLE (review round 2, point 1).

The correct Legendre image of L_extra = gamma*(s - k)^2 (s velocity-
independent, k homogeneous of degree two in the velocity) is

    H_extra = -gamma*s^2 - 2*gamma*s*k + 3*gamma*k^2

(the static square flips sign in H = sum p qdot - L). The -gamma*s^2
term is unbounded below once the static density exceeds ~1/(2 gamma)
against the linear e_static cost; on this lattice max i1s ~ 0.0102 vs
1/gamma = 0.0142, and relaxation under the correct H_extra blows
through the threshold immediately. This script documents the runaway:
1000 Adam steps at omega = 0 and omega = 0.19, recording the energy and
max static density every 100 steps. (Choosing -gamma for the overall
Lagrangian sign flips the drive and brake and removes the clock
analytically, so no ladder is run for it.)

Consequence for the report: gamma*(I1_G)^2 works as an ENERGY-FUNCTIONAL
ansatz (JG_E); its naive fundamental-Lagrangian reading does not define
a stable theory. Out: results/fundamental_runaway.json
"""
import json
import os
import runpy
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "runaway_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("fundamental_runaway: NOT REPRODUCED HERE -- needs report "
          f"004's polished field in {FIELDS} (or M5_FIELDS_DIR).")
    sys.exit(0)

lad = runpy.run_path(os.path.join(HERE, "ladder_i1sq_defs.py"))
field, e_static = lad["field"], lad["e_static"]
densities, A0, H = lad["densities"], lad["A0"], lad["H"]
M_pol = lad["M_pol"]
base = lad["load_base"](HERE)
gamma = base["gamma"]
print(f"gamma = {gamma:.5f}; unboundedness threshold 1/gamma = "
      f"{1/gamma:.5f} vs initial max i1s")

out = {"gamma": gamma, "threshold_inv_gamma": 1.0 / gamma, "runs": []}
for om in (0.0, 0.19):
    def e_tot(Mr, om=om):
        Mf = field(Mr)
        i1s, k = densities(Mf, A0, om, "G")
        dens = -i1s ** 2 - 2.0 * i1s * k + 3.0 * k ** 2
        return e_static(Mf, "G") + gamma * H ** 3 * dens.sum()

    M_raw = M_pol.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    traj = []
    for it in range(1000):
        opt.zero_grad()
        e_tot(M_raw).backward()
        opt.step()
        if (it + 1) % 100 == 0:
            with torch.no_grad():
                Mf = field(M_raw)
                i1s, _ = densities(Mf, A0, max(om, 1e-9), "G")
                traj.append({"step": it + 1,
                             "E": float(e_tot(M_raw).detach()),
                             "max_i1s": float(i1s.max())})
            print(f"  om {om} step {it+1}: E {traj[-1]['E']:.3e}, "
                  f"max i1s {traj[-1]['max_i1s']:.3e}", flush=True)
    out["runs"].append({"omega": om, "trajectory": traj})

json.dump(out, open(os.path.join(HERE, "results",
                                 "fundamental_runaway.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("runaway documented in this run\n")
print("written: results/fundamental_runaway.json")
