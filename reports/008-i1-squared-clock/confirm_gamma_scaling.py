"""Confirmation that the J1 well is the predicted physics, not
relaxation noise. From E(omega) ~ -2 g C1 omega^2 + g C2 omega^4:
the minimum position omega_* = sqrt(C1/C2) is INDEPENDENT of gamma,
while the well depth g C1^2/C2 scales linearly. Rerun the J1 ladder at
gamma x4 (20% statics deformation) and check both.

Needs report 004's polished field (or M5_FIELDS_DIR) plus
results/i1sq_ladders.json from ladder_i1sq.py.
Out: results/gamma_scaling.json
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
gamma4 = 4.0 * base["gamma"]
Mg = field(M_pol)
a0 = a0_of(gen_catalog()["boost_x"], Mg)
print(f"gamma x4 = {gamma4:.4f} (20% statics deformation); prediction: "
      f"same omega_* = {base['omega_pred_frozen']:.3f}, depth x4")

rows = []
for om in (0.0, 0.2, 0.35, 0.5, 0.8):
    M_raw = M_pol.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        Mf = field(M_raw)
        i1s = i1_static_density(Mf)
        bk, _ = boost_channels(Mf, a0, om)
        (e_static(Mf, "G")
         + gamma4 * H ** 3 * ((i1s - bk) ** 2).sum()).backward()
        opt.step()
    Mf = field(M_raw.detach())
    i1s = i1_static_density(Mf)
    bk, _ = boost_channels(Mf, a0, max(om, 1e-9))
    E = (e_static(Mf, "G")
         + gamma4 * H ** 3 * ((i1s - bk) ** 2).sum()).item()
    pr = ((bk.sum() ** 2) / (bk ** 2).sum().clamp_min(1e-30)).item()
    rows.append({"omega": om, "E_total": E, "PR_bk_sites": pr})
    print(f"  omega {om}: E {E:.5f}, PR {pr:.0f}", flush=True)

k = min(range(len(rows)), key=lambda i: rows[i]["E_total"])
depth = rows[0]["E_total"] - rows[k]["E_total"]
base_rows = {r["omega"]: r["E_total"]
             for r in base["J1_local_covariant"]["rungs"]}
base_depth = base_rows[0.0] - min(base_rows.values())
ratio = depth / max(base_depth, 1e-12)
print(f"minimum at omega = {rows[k]['omega']} "
      f"(interior: {0 < k < len(rows)-1}); depth {depth:.5f} vs base "
      f"{base_depth:.5f} (ratio {ratio:.2f}, predicted ~4)")
out = {"gamma4": gamma4, "rows": rows, "min_omega": rows[k]["omega"],
       "interior": bool(0 < k < len(rows) - 1), "depth": depth,
       "base_depth": base_depth, "depth_ratio": ratio}
json.dump(out, open(os.path.join(HERE, "results",
                                 "gamma_scaling.json"), "w"), indent=1)
with open(FLAG, "w") as f:
    f.write("gamma scaling computed in this run\n")
print("written: results/gamma_scaling.json")
