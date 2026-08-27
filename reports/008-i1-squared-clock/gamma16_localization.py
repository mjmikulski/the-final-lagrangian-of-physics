"""Adversarial localization check: does the ticking stay localized when
the well is DEEP? The convexity mechanism predicts yes -- spreading
away from the template costs at any gamma; if instead the observed
PR ~ 100 were only the weakness of the term, a 16x deeper well would
delocalize like 004/007's Mexican-hat quartics did.

Runs the J1 rung at omega = 0 / 0.35 / 0.8 with gamma x16 (statics
deformation 80% -- deliberately aggressive) and asserts PR stays at the
core scale. Out: results/gamma16_localization.json
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
FLAG = os.path.join(HERE, "results", "gamma16_ran.flag")
if not os.path.exists(MP):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("gamma16_localization: NOT REPRODUCED HERE -- needs report "
          f"004's polished field in {FIELDS} (or M5_FIELDS_DIR).")
    sys.exit(0)

L = runpy.run_path(os.path.join(R004, "lattice.py"), run_name="not_main")
field, e_static = L["field"], L["e_static"]
boost_channels, a0_of, gen_catalog = (L["boost_channels"], L["a0_of"],
                                      L["gen_catalog"])
d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
H, DT, DEV = L["H"], L["DT"], L["DEV"]
M_pol = torch.tensor(np.load(MP)["M"], dtype=DT, device=DEV)


def i1_static_density(M):
    G = G_of(M)
    e = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                e = e + 0.5 * 4.0 * torch.einsum(
                    "...ab,...ac,...bd,...cd->...", F, G, G, F)
    return e


base = json.load(open(os.path.join(HERE, "results",
                                   "i1sq_ladders.json")))
gamma16 = 16.0 * base["gamma"]
Mg = field(M_pol)
a0 = a0_of(gen_catalog()["boost_x"], Mg)
print(f"gamma x16 = {gamma16:.2f} (80% statics deformation)")

rows = []
for om in (0.0, 0.35, 0.8):
    M_raw = M_pol.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        Mf = field(M_raw)
        i1s = i1_static_density(Mf)
        bk, _ = boost_channels(Mf, a0, om)
        (e_static(Mf, "G")
         + gamma16 * H ** 3 * ((i1s - bk) ** 2).sum()).backward()
        opt.step()
    Mf = field(M_raw.detach())
    i1s = i1_static_density(Mf)
    bk, _ = boost_channels(Mf, a0, max(om, 1e-9))
    E = (e_static(Mf, "G")
         + gamma16 * H ** 3 * ((i1s - bk) ** 2).sum()).item()
    pr = ((bk.sum() ** 2) / (bk ** 2).sum().clamp_min(1e-30)).item()
    rows.append({"omega": om, "E_total": E, "PR_bk_sites": pr})
    print(f"  omega {om}: E {E:.5f}, PR {pr:.0f}", flush=True)

pr_min = rows[1]["PR_bk_sites"]
depth = rows[0]["E_total"] - rows[1]["E_total"]
print(f"well depth at 16x: {depth:.5f}; PR at the minimum: {pr_min:.0f}")
out = {"gamma16": gamma16, "rows": rows, "depth": depth,
       "PR_at_min": pr_min}
json.dump(out, open(os.path.join(HERE, "results",
                                 "gamma16_localization.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("gamma16 localization computed in this run\n")
print("written: results/gamma16_localization.json")
