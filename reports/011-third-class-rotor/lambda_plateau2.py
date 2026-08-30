"""Round-2 response, finding 2: continue L = 36 and L = 48 from the
persisted deep endpoints under an OBSERVABLE-LEVEL stopping rule --
the run stops only when the tube excess lambda_z - lambda_x drifts
by less than 1% relative over four consecutive cycles (max 30 more
cycles); the total energy is recorded but does not gate. Out:
results/lambda_plateau2.json
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
FLAG = os.path.join(HERE, "results", "plateau2_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("lambda_plateau2: NOT REPRODUCED HERE.")
    sys.exit(0)

d = runpy.run_path(os.path.join(HERE, "defs_011.py"))
DRIFT, WIN, MAXC = 0.01, 4, 30

out = {"drift_tol": DRIFT, "window": WIN, "max_cycles": MAXC,
       "cases": {}}
for tag, kw in (("EQ_N24", dict(N=24)), ("EQ_N32", dict(N=32))):
    L = d["load_stack"](**kw)
    M0 = d["seed"](L, "EQ")
    d["install_seed"](L, M0)
    field, e_static = L["field"], L["e_static"]
    src = os.path.join(HERE, "results", f"M_{tag}_deep.npz")
    Mp = torch.tensor(np.load(src)["M"], dtype=L["DT"],
                      device=L["DEV"])
    M_raw = Mp.clone().requires_grad_(True)

    def snap():
        Mf = field(M_raw.detach())
        m = d["measures"](L, Mf)
        g = torch.autograd.grad(e_static(field(M_raw), "G"), M_raw)[0]
        return {"E": float(e_static(Mf, "G")),
                "ginf": float(g.abs().max()),
                "excess": m["lambda_axis_excess"]}
    traj = [snap()]
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
        traj.append(snap())
        if len(traj) > WIN:
            ex = [t["excess"] for t in traj[-(WIN + 1):]]
            drifts = [abs(ex[i + 1] - ex[i]) / abs(ex[-1])
                      for i in range(WIN)]
            if max(drifts) < DRIFT:
                break
    out["cases"][tag] = {"trajectory": traj,
                         "cycles": len(traj) - 1,
                         "final_excess": traj[-1]["excess"],
                         "stopped_on_observable":
                             len(traj) - 1 < MAXC}
    exs = " ".join(f"{t['excess']:.3e}" for t in traj)
    print(f"[{tag}] {len(traj)-1} cycles "
          f"(observable-stop {len(traj)-1 < MAXC}); excess: {exs}",
          flush=True)
    np.savez_compressed(os.path.join(HERE, "results",
                                     f"M_{tag}_deep2.npz"),
                        M=field(M_raw.detach()).cpu().numpy())
    del L, M0, Mp, M_raw
    torch.cuda.empty_cache()

e36 = out["cases"]["EQ_N24"]["final_excess"]
e48 = out["cases"]["EQ_N32"]["final_excess"]
out["agreement_rel"] = abs(e48 - e36) / abs(e36)
print(f"plateaus: L36 {e36:.3e}, L48 {e48:.3e} "
      f"(rel diff {out['agreement_rel']:.1%})")
json.dump(out, open(os.path.join(HERE, "results",
                                 "lambda_plateau2.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("observable-level plateau computed in this run\n")
print("written: results/lambda_plateau2.json")
