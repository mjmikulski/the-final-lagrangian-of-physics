"""Producer 3 (review round 1, finding 3): the protocol-matched boost
contrast, supplied as a committed record instead of a citation to a
nonexistent artifact.

The energy-reading boost clock of report 008 (its merged JG_E ladder
ran four L-BFGS cycles) is continued here under THIS report's deep
protocol: the persisted bracket fields jge_rung_om{02,035,05}.npz and
a fresh omega = 0 endpoint, 24 L-BFGS(150) cycles each, per-cycle
energies and gradients recorded -- exactly the budget and cycle
length of continue_window.py, so the fundamental-reading drift and
the energy-reading stability can be compared like for like.
Out: results/boost_contrast.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R008 = os.path.join(HERE, "..", "008-i1-squared-clock")
R004 = os.path.join(HERE, "..", "004-lattice-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "boost_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("boost_contrast: NOT REPRODUCED HERE -- needs the 004 stack.")
    sys.exit(0)

import runpy
lad = runpy.run_path(os.path.join(R008, "ladder_i1sq_defs.py"))
field, e_static = lad["field"], lad["e_static"]
densities, H, A0 = lad["densities"], lad["H"], lad["A0"]
M_pol = lad["M_pol"]
res = json.load(open(os.path.join(R008, "results",
                                  "i1sq_ladders.json")))
gamma = res["gamma"]
MAXC = 24


def e_tot_of(om):
    def e_tot(Mr):
        Mf = field(Mr)
        i1s, k = densities(Mf, A0, max(om, 1e-9), "G")
        return (e_static(Mf, "G")
                + gamma * H ** 3 * ((i1s - k) ** 2).sum())
    return e_tot


out = {"gamma": gamma, "max_cycles": MAXC, "rungs": {}}
for om, src in ((0.0, None), (0.2, "jge_rung_om02.npz"),
                (0.35, "jge_rung_om035.npz"),
                (0.5, "jge_rung_om05.npz")):
    e_tot = e_tot_of(om)
    if src:
        M0 = torch.tensor(
            np.load(os.path.join(R008, "results", src))["M"],
            dtype=M_pol.dtype, device=M_pol.device)
        M_raw = M0.clone().requires_grad_(True)
    else:
        M_raw = M_pol.clone().requires_grad_(True)
        opt = torch.optim.Adam([M_raw], lr=1e-3)
        for it in range(500):
            opt.zero_grad()
            e_tot(M_raw).backward()
            opt.step()
        M_raw = M_raw.detach().requires_grad_(True)
    Es, Gs = [], []

    def snap():
        Mv = M_raw.detach().requires_grad_(True)
        E = e_tot(Mv)
        g = torch.autograd.grad(E, Mv)[0]
        return float(E.detach()), float(g.abs().max())
    e, g = snap()
    Es.append(e)
    Gs.append(g)
    for cyc in range(MAXC):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=150,
                                 history_size=25, tolerance_grad=1e-9,
                                 tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_tot(M_raw)
            E.backward()
            return E
        opt2.step(closure)
        e, g = snap()
        Es.append(e)
        Gs.append(g)
    out["rungs"][str(om)] = {"E": Es, "ginf": Gs}
    print(f"[boost om {om}] E {Es[0]:.9f} -> {Es[-1]:.9f} "
          f"({MAXC} cycles, |g| {Gs[-1]:.1e})", flush=True)
    del M_raw
    torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(HERE, "results",
                                     "boost_contrast.json"), "w"),
              indent=1)

E = {om: out["rungs"][om]["E"] for om in out["rungs"]}
n = len(E["0.35"])
d02 = [E["0.2"][i] - E["0.35"][i] for i in range(n)]
d00 = [E["0.0"][i] - E["0.35"][i] for i in range(n)]
d05 = [E["0.5"][i] - E["0.35"][i] for i in range(n)]
out["diffs_final"] = {"0.0": d00[-1], "0.2": d02[-1], "0.5": d05[-1]}
out["ordering_held_every_cycle"] = bool(
    all(d00[i] > 0 and d02[i] > 0 and d05[i] > 0 for i in range(n)))
print("final diffs vs 0.35:", out["diffs_final"],
      "| ordering held every cycle:",
      out["ordering_held_every_cycle"])
json.dump(out, open(os.path.join(HERE, "results",
                                 "boost_contrast.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("boost contrast computed in this run\n")
print("written: results/boost_contrast.json")
