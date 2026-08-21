"""Extended C3 ladder + delocalization diagnostic.

Continues stage_ladder (same a, b, generator, frozen a0) to larger omega
to locate the interior minimum the first ladder left at its edge, and
records per rung the participation ratio of the local boost density
PR = (sum B)^2 / sum B^2  [sites]: ~core-size = localized clock,
~N^3 = the extensive-floor delocalized condensate.
"""
import json
import os
import runpy

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
L = runpy.run_path(os.path.join(HERE, "lattice.py"), run_name="not_main")
field, e_static, e_condensate = L["field"], L["e_static"], L["e_condensate"]
boost_channels, a0_of, gen_catalog = (L["boost_channels"], L["a0_of"],
                                      L["gen_catalog"])
H, DT, DEV = L["H"], L["DT"], L["DEV"]

res = json.load(open(os.path.join(HERE, "results", "lattice_results.json")))
setup = res["ladder_setup"]
a_c, b_c = setup["a"], setup["b"]
Mr = torch.tensor(np.load(os.path.join(HERE, os.path.join("results", "M_G.npz")))["M"],
                  dtype=DT, device=DEV)
a0 = a0_of(gen_catalog()[setup["generator"]], field(Mr))

rungs = []
for om in (0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.4, 2.8):
    Mr = Mr.clone().requires_grad_(True)
    opt = torch.optim.Adam([Mr], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        Mf = field(Mr)
        E = e_static(Mf, "G") + e_condensate(Mf, a0, om, a_c, b_c)
        E.backward()
        opt.step()
    Mr = Mr.detach()
    Mf = field(Mr)
    Es = e_static(Mf, "G").item()
    Ec = e_condensate(Mf, a0, om, a_c, b_c).item()
    bk, _ = boost_channels(Mf, a0, om)
    pr = ((bk.sum() ** 2) / (bk ** 2).sum().clamp_min(1e-30)).item()
    row = {"omega": om, "E_total": Es + Ec, "E_stat": Es, "E_cond": Ec,
           "participation_sites": pr}
    rungs.append(row)
    print(f"omega {om}: E {Es+Ec:.4f} (stat {Es:.4f}, cond {Ec:.4f}), "
          f"PR {pr:.0f} sites", flush=True)

E0 = res["ladder"]["rungs"][0]["E_total"]
k = min(range(len(rungs)), key=lambda i: rungs[i]["E_total"])
res["ladder_ext"] = {"rungs": rungs, "E0": E0,
                     "min_omega": rungs[k]["omega"],
                     "interior": 0 < k < len(rungs) - 1}
json.dump(res, open(os.path.join(HERE, "results", "lattice_results.json"), "w"),
          indent=1)
print("verdict: min at omega =", rungs[k]["omega"],
      "interior:", res["ladder_ext"]["interior"])
