"""Producer 2: extended clock ladders at delta = 1/8 for every g in
{8, 64, 512} -- seven rungs at (0, 0.5, 1, 1.5, 2, 3, 4) x om_pred,
starting from the persisted grid fields, with the participation ratio
(PR, effective site count) of the ticking density recorded per rung.
Notation as in grid_scan.py. This resolves the 3-rung mini-ladder's
top-rung minima at g >= 64 (well shifted vs prediction, not absent)
and provides the g = 8 control that attributes the large PR to the
grid protocol (relative potential + short relax), not to g.
Out: results/extended_ladders_all.json
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
LAT = os.path.join(R004, "lattice.py")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "ladders_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    import sys
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("extended_ladders: NOT REPRODUCED HERE -- needs the 004 "
          "stack and the persisted grid fields.")
    sys.exit(0)
SRC = open(LAT).read()


def load_stack(delta, g):
    src = SRC.replace("SG, DELTA, W1 = 8.0, 0.3, 0.000724023879",
                      f"SG, DELTA, W1 = {float(g)}, {delta}, "
                      "0.000724023879")
    src = src.replace("v4 = v4 + (t - C_P[p]) ** 2",
                      "v4 = v4 + (t / C_P[p] - 1.0) ** 2")
    ns = {"__name__": "not_main", "__file__": LAT}
    exec(compile(src, "lattice_patched", "exec"), ns)
    return ns


grid = json.load(open(os.path.join(HERE, "results", "grid.json")))
P = {(r["delta"], r["g"]): r for r in grid["points"]}

out = {"cases": []}
for g in (8, 64, 512):
    delta = 0.125
    L = load_stack(delta, g)
    field, e_static = L["field"], L["e_static"]
    H, DT, DEV = L["H"], L["DT"], L["DEV"]
    d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
    Mf0 = torch.tensor(
        np.load(os.path.join(HERE, "results",
                             f"M_d{delta:.6f}_g{g}.npz"))["M"],
        dtype=DT, device=DEV)
    a0 = L["a0_of"](L["gen_catalog"]()["boost_x"], Mf0)
    rec0 = P[(delta, g)]
    om_p, gamma = rec0["om_pred"], rec0["gamma"]

    def densities(M, om):
        G = G_of(M)
        V = om * a0
        i1s = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
        k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
        for st in ("fwd", "bwd"):
            A = [d1(M, ax, st) for ax in range(3)]
            for i in range(3):
                F0 = comm(V, A[i])
                k = k + 0.5 * 4.0 * torch.einsum(
                    "...ab,...ac,...bd,...cd->...", F0, G, G, F0)
                for j in range(i + 1, 3):
                    F = comm(A[i], A[j])
                    i1s = i1s + 0.5 * 4.0 * torch.einsum(
                        "...ab,...ac,...bd,...cd->...", F, G, G, F)
        return i1s, k

    rows = []
    for f in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        om = round(f * om_p, 3)
        def e_tot(Mr, om=om):
            Mfx = field(Mr)
            i1s, k = densities(Mfx, max(om, 1e-9))
            return (e_static(Mfx, "G")
                    + gamma * H ** 3 * ((i1s - k) ** 2).sum())
        Mr2 = Mf0.clone().requires_grad_(True)
        opt = torch.optim.Adam([Mr2], lr=1e-3)
        for it in range(500):
            opt.zero_grad()
            e_tot(Mr2).backward()
            opt.step()
        opt2 = torch.optim.LBFGS([Mr2], max_iter=200, history_size=25,
                                 tolerance_grad=1e-9,
                                 tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_tot(Mr2)
            E.backward()
            return E
        opt2.step(closure)
        Mfx = field(Mr2.detach())
        _, kd = densities(Mfx, max(om, 1e-9))
        pr = ((kd.sum() ** 2)
              / (kd ** 2).sum().clamp_min(1e-30)).item()
        rows.append({"omega": om, "E_total": float(e_tot(Mr2)),
                     "PR": pr})
        print(f"[g={g}] om {om}: E {rows[-1]['E_total']:.6f} "
              f"(PR {pr:.0f})", flush=True)
        del Mr2
        torch.cuda.empty_cache()
    k = min(range(len(rows)), key=lambda i: rows[i]["E_total"])
    interior = bool(0 < k < len(rows) - 1)
    out["cases"].append({"delta": delta, "g": g, "om_pred": om_p,
                         "gamma": gamma, "rows": rows,
                         "min_omega": rows[k]["omega"],
                         "interior": interior})
    print(f"[g={g}] verdict: min at {rows[k]['omega']} "
          f"(pred {om_p:.3f}), interior {interior}", flush=True)
    del L, Mf0
    torch.cuda.empty_cache()

json.dump(out, open(os.path.join(HERE, "results",
                                 "extended_ladders_all.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("extended ladders computed in this run\n")
print("written: results/extended_ladders_all.json")
