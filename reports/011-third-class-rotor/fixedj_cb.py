"""Producer 2: fixed-J on the relaxed third-class (CB) configuration.

With the equivariant frozen boundary the global combined rotation
preserves the boundary data (the shell is equivariant, zeta M_shell = 0
up to the 1/r discretization residual), so -- unlike report 009's
masked surrogate -- the rotation acts on the configuration space and
the angle is cyclic up to that residual; J is the corresponding
(approximate at finite h) Noether charge. The functional
E_J[M] = E_stat[M] + J^2 / (2 I[M]) with I from the FULL zeta tangent
of the current field (equivariant-quotient direction) is minimized at
prescribed J.
Out: results/fixedj_cb.json
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
FLAG = os.path.join(HERE, "results", "fixedj_ran.flag")
SRC_FIELD = os.path.join(HERE, "results", "M_CB_N32.npz")
if not (os.path.exists(os.path.join(FIELDS, "M_G_polished.npz"))
        and os.path.exists(SRC_FIELD)):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("fixedj_cb: NOT REPRODUCED HERE -- needs the 004 stack and "
          "the persisted M_CB_N32.npz (relax_all.py).")
    sys.exit(0)

d = runpy.run_path(os.path.join(HERE, "defs_011.py"))
L = d["load_stack"](N=32)
M0 = d["seed"](L, "CB")
d["install_seed"](L, M0)     # equivariant boundary for field()
field, e_static = L["field"], L["e_static"]
H = L["H"]
M_cb = torch.tensor(np.load(SRC_FIELD)["M"], dtype=L["DT"],
                    device=L["DEV"])

I_rigid = float(2.0 * H ** 3
                * d["kin_density"](L, M_cb, d["zeta_of"](L, M_cb)).sum())
print(f"relaxed-CB rigid inertia I_0 = {I_rigid:.6e}")

out = {"I_0": I_rigid, "rows": []}
for J in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
    def e_tot(Mr, J=J):
        Mf = field(Mr)
        E = e_static(Mf, "G")
        if J > 0:
            z = d["zeta_of"](L, Mf)
            I = 2.0 * H ** 3 * d["kin_density"](L, Mf, z).sum()
            E = E + J ** 2 / (2.0 * I)
        return E
    M_raw = M_cb.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        e_tot(M_raw).backward()
        opt.step()
    for cyc in range(2):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_tot(M_raw)
            E.backward()
            return E
        opt2.step(closure)
    Mf = field(M_raw.detach())
    z = d["zeta_of"](L, Mf)
    kz = d["kin_density"](L, Mf, z)
    I = float(2.0 * H ** 3 * kz.sum())
    pr = float((kz.sum() ** 2) / (kz ** 2).sum().clamp_min(1e-30))
    E = float(e_tot(M_raw.detach()))
    out["rows"].append({"J": J, "E_total": E, "I": I,
                       "omega": J / I if J > 0 else 0.0, "PR_kin": pr})
    print(f"  J {J}: E {E:.6f}, I {I:.4e}, omega {J/I if J>0 else 0:.5f},"
          f" PR {pr:.0f}", flush=True)
    del M_raw, Mf, z, kz
    torch.cuda.empty_cache()

E0 = out["rows"][0]["E_total"]
for r in out["rows"][1:]:
    pred = r["J"] ** 2 / (2 * I_rigid)
    print(f"  J {r['J']}: dE {r['E_total']-E0:.3e} vs rigid {pred:.3e}"
          f" (ratio {(r['E_total']-E0)/pred:.3f})")
json.dump(out, open(os.path.join(HERE, "results", "fixedj_cb.json"),
                    "w"), indent=1)
with open(FLAG, "w") as f:
    f.write("fixed-J on relaxed CB computed in this run\n")
print("written: results/fixedj_cb.json")
