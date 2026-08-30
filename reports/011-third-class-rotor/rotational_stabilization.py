"""Producer 5: does rotation stabilize the core deformation?

Statics alone removes both core-deformation types (relax_all,
frame_twist): nothing in E_stat favors a symmetry-breaking core. But
at fixed J the term J^2/(2 I[M]) REWARDS inertia, so a core
deformation that raises I lowers E_J -- centrifugal stabilization.
Test: minimize E_J[M] = E_stat + J^2/(2 I[M]) from the CB SEED (not
the relaxed field) at J = 0, 0.4, 0.8 and at the deformation-threshold estimate J = 4
(J_thr ~ sqrt(2 I dE_def) with dE_def ~ 0.05-0.09 the seed
deformation costs), same deep-ish protocol (Adam 1000 + 3 L-BFGS
cycles), measuring the surviving inertia and the excess kinetic
density over the J = 0 endpoint.
Out: results/rot_stabilization.json
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
FLAG = os.path.join(HERE, "results", "rotstab_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("rotational_stabilization: NOT REPRODUCED HERE.")
    sys.exit(0)

d = runpy.run_path(os.path.join(HERE, "defs_011.py"))
L = d["load_stack"](N=24)
M0 = d["seed"](L, "CB")
d["install_seed"](L, M0)
field, e_static = L["field"], L["e_static"]
H = L["H"]

out = {"rows": []}
kd_final = {}
for J in (0.0, 0.4, 0.8, 4.0):
    def e_tot(Mr, J=J):
        Mf = field(Mr)
        E = e_static(Mf, "G")
        if J > 0:
            z = d["zeta_of"](L, Mf)
            I = 2.0 * H ** 3 * d["kin_density"](L, Mf, z).sum()
            E = E + J ** 2 / (2.0 * I)
        return E
    M_raw = M0.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(1000):
        opt.zero_grad()
        e_tot(M_raw).backward()
        opt.step()
    E_levels = [float(e_tot(M_raw).detach())]
    for cyc in range(3):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_tot(M_raw)
            E.backward()
            return E
        opt2.step(closure)
        E_levels.append(float(e_tot(M_raw).detach()))
    Mf = field(M_raw.detach())
    z = d["zeta_of"](L, Mf)
    kz = d["kin_density"](L, Mf, z)
    kd_final[J] = kz
    I = float(2.0 * H ** 3 * kz.sum())
    pr = float((kz.sum() ** 2) / (kz ** 2).sum().clamp_min(1e-30))
    Es = float(e_static(Mf, "G"))
    out["rows"].append({"J": J, "E_levels": E_levels,
                        "E_total": E_levels[-1], "E_stat": Es,
                        "I": I, "omega": J / I if J > 0 else 0.0,
                        "PR_kin": pr})
    print(f"J {J}: E {E_levels[-1]:.6f} (stat {Es:.6f}), I {I:.4e}, "
          f"PR {pr:.0f}", flush=True)
    del M_raw, Mf, z
    torch.cuda.empty_cache()

for J in (0.4, 0.8, 4.0):
    dk = (kd_final[J] - kd_final[0.0]).clamp_min(0.0)
    Ix = float(2.0 * H ** 3 * dk.sum())
    prx = float((dk.sum() ** 2) / (dk ** 2).sum().clamp_min(1e-30))
    out[f"excess_J{J}"] = {"I_excess": Ix, "PR_excess": prx}
    print(f"excess at J={J}: I_excess {Ix:.4e}, PR {prx:.0f}")
json.dump(out, open(os.path.join(HERE, "results",
                                 "rot_stabilization.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("rotational stabilization computed in this run\n")
print("written: results/rot_stabilization.json")
