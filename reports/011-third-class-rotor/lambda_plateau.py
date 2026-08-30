"""Finding-2 response: continue the persisted EQ endpoints with
per-cycle recording of the tube observable itself. For each of
EQ_N16 / EQ_N24 / EQ_N32 / EQ_d125 / EQ_h075: up to 24 further L-BFGS
cycles (stop when the per-cycle energy change < 5e-4), recording at
EVERY cycle the total energy, |g|_inf, lambda_z, lambda_x and the
excess -- the plateau (or its honest absence) of the local observable,
which the round-1 data could not show. Out: results/lambda_plateau.json
(+ updated persisted fields M_<tag>_deep.npz)
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
FLAG = os.path.join(HERE, "results", "plateau_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("lambda_plateau: NOT REPRODUCED HERE.")
    sys.exit(0)

d = runpy.run_path(os.path.join(HERE, "defs_011.py"))
CASES = [("EQ_N16", dict(N=16), 0.3), ("EQ_N24", dict(N=24), 0.3),
         ("EQ_N32", dict(N=32), 0.3), ("EQ_d125", dict(N=32), 0.125),
         ("EQ_h075", dict(N=32, Lbox=24.0), 0.3)]
TOL, MAXC = 5e-4, 24

out = {"tol": TOL, "max_cycles": MAXC, "cases": {}}
for tag, kw, delta in CASES:
    L = d["load_stack"](delta=delta, **kw)
    M0 = d["seed"](L, "EQ", delta=delta)
    d["install_seed"](L, M0)
    field, e_static = L["field"], L["e_static"]
    Mp = torch.tensor(
        np.load(os.path.join(HERE, "results", f"M_{tag}.npz"))["M"],
        dtype=L["DT"], device=L["DEV"])
    M_raw = Mp.clone().requires_grad_(True)
    traj = []

    def snap(Mr):
        Mf = field(Mr.detach() if Mr.requires_grad else Mr)
        m = d["measures"](L, Mf)
        g = torch.autograd.grad(
            e_static(field(Mr), "G"), Mr)[0] if Mr.requires_grad \
            else None
        return {"E": float(e_static(Mf, "G")),
                "ginf": float(g.abs().max()) if g is not None else None,
                "lambda_z": m["lambda_z"], "lambda_x": m["lambda_x"],
                "excess": m["lambda_axis_excess"]}
    traj.append(snap(M_raw))
    for cyc in range(MAXC):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_static(field(M_raw), "G")
            E.backward()
            return E
        opt2.step(closure)
        traj.append(snap(M_raw))
        dE = traj[-2]["E"] - traj[-1]["E"]
        if abs(dE) < TOL:
            break
    out["cases"][tag] = {"trajectory": traj,
                         "cycles": len(traj) - 1,
                         "final": traj[-1]}
    exs = [f"{t['excess']:.2e}" for t in traj]
    print(f"[{tag}] {len(traj)-1} cycles; E {traj[0]['E']:.5f}->"
          f"{traj[-1]['E']:.5f} (|g| {traj[-1]['ginf']:.1e}); "
          f"excess: {' '.join(exs)}", flush=True)
    np.savez_compressed(
        os.path.join(HERE, "results", f"M_{tag}_deep.npz"),
        M=field(M_raw.detach()).cpu().numpy())
    del L, M0, Mp, M_raw
    torch.cuda.empty_cache()

json.dump(out, open(os.path.join(HERE, "results",
                                 "lambda_plateau.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("lambda plateau computed in this run\n")
print("written: results/lambda_plateau.json")
