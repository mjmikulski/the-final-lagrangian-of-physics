"""Blocker-1 response: converge the rotational bracket for real.
Continue L-BFGS cycles on the persisted rung fields (and the omega = 0
endpoint refreshed from scratch at matching depth) until the per-cycle
energy change drops below 1e-7 or 24 cycles, recording the full
trajectory. Verdict: where the minimum actually sits once the local
bracket is converged. Out: results/deep_converge.json
"""
import json
import os
import runpy

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
R008 = os.path.join(HERE, "..", "008-i1-squared-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "deep_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    import sys
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("deep_converge: NOT REPRODUCED HERE -- needs report 004's "
          f"polished field in {FIELDS} (or M5_FIELDS_DIR).")
    sys.exit(0)
lad = runpy.run_path(os.path.join(R008, "ladder_i1sq_defs.py"))
field, e_static = lad["field"], lad["e_static"]
densities, H = lad["densities"], lad["H"]
M_pol = lad["M_pol"]
DT, DEV = lad["field"], None
import torch as _t
A0R = _t.tensor(np.load(os.path.join(HERE, "results",
                                     "a0r_frozen.npz"))["a0"],
                dtype=lad["M_pol"].dtype, device=lad["M_pol"].device)
res = json.load(open(os.path.join(HERE, "results", "rot_ladders.json")))
gamma = res["gamma"]
TOL, MAXC = 1e-7, 24


def e_tot_of(om):
    def e_tot(Mr):
        Mf = field(Mr)
        i1s, k = densities(Mf, A0R, om, "G")
        return (e_static(Mf, "G")
                + gamma * H ** 3 * ((i1s - k) ** 2).sum())
    return e_tot


out = {"tol": TOL, "max_cycles": MAXC, "runs": {}}
for om, src in ((0.0, None), (0.152, "rot_rung_om0152.npz"),
                (0.217, "rot_rung_om0217.npz"),
                (0.304, "rot_rung_om0304.npz")):
    e_tot = e_tot_of(om)
    if src:
        M0 = _t.tensor(np.load(os.path.join(HERE, "results", src))["M"],
                       dtype=M_pol.dtype, device=M_pol.device)
        # persisted fields are field()-outputs; optimize a raw copy
        M_raw = M0.clone().requires_grad_(True)
    else:
        M_raw = M_pol.clone().requires_grad_(True)
        opt = torch.optim.Adam([M_raw], lr=1e-3)
        for it in range(500):
            opt.zero_grad()
            e_tot(M_raw).backward()
            opt.step()
        M_raw = M_raw.detach().requires_grad_(True)
    traj = [float(e_tot(M_raw).detach())]
    for cyc in range(MAXC):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200,
                                 history_size=25, tolerance_grad=1e-9,
                                 tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_tot(M_raw)
            E.backward()
            return E
        opt2.step(closure)
        traj.append(float(e_tot(M_raw).detach()))
        if abs(traj[-1] - traj[-2]) < TOL:
            break
    out["runs"][str(om)] = {"trajectory": traj,
                            "E_final": traj[-1],
                            "cycles": len(traj) - 1,
                            "last_change": traj[-1] - traj[-2]}
    print(f"om {om}: E {traj[0]:.9f} -> {traj[-1]:.9f} in "
          f"{len(traj)-1} cycles (last change {traj[-1]-traj[-2]:+.1e})",
          flush=True)

E = {k: v["E_final"] for k, v in out["runs"].items()}
order = sorted(E, key=lambda k: E[k])
print(f"converged ordering: {order}; "
      f"E(0.152)-E(0.217) = {E['0.152']-E['0.217']:+.3e}; "
      f"E(0.0)-E(min) = {E['0.0']-min(E.values()):+.3e}")
out["converged_energies"] = E
json.dump(out, open(os.path.join(HERE, "results",
                                 "deep_converge.json"), "w"), indent=1)
with open(FLAG, "w") as f:
    f.write("deep convergence computed in this run\n")
print("written: results/deep_converge.json")
