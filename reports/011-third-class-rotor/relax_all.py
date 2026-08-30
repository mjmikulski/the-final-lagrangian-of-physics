"""Producer 1: all relaxations of report 011, persisted.

Grid (the L-sweep carries BOTH the inertia and the static-energy /
line-tension scaling -- the latter added on MJ's plan review: a finite
inertia must not be bought with a pathologically extensive axial
statics):
  EQ, CB at N = 16, 24, 32 (h = 1.5)      -> L-sweep
  EQ, CB at N = 32, L = 24 (h = 0.75)     -> h-test pair vs N=16
  EQ at delta = 1/8 (N = 32)              -> axial-cost comparison
Each: deep protocol (Adam 1000 + 6 L-BFGS cycles, E_levels), then the
measurement kit; fields persisted for the later producers.
Out: results/relax_all.json + results/M_<tag>.npz
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
FLAG = os.path.join(HERE, "results", "relax_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("relax_all: NOT REPRODUCED HERE -- needs report 004's stack "
          f"(M5_FIELDS_DIR); committed results carry the record.")
    sys.exit(0)

d = runpy.run_path(os.path.join(HERE, "defs_011.py"))

CASES = [
    ("EQ_N16", dict(N=16), "EQ", 0.3),
    ("CB_N16", dict(N=16), "CB", 0.3),
    ("EQ_N24", dict(N=24), "EQ", 0.3),
    ("CB_N24", dict(N=24), "CB", 0.3),
    ("EQ_N32", dict(N=32), "EQ", 0.3),
    ("CB_N32", dict(N=32), "CB", 0.3),
    ("EQ_h075", dict(N=32, Lbox=24.0), "EQ", 0.3),
    ("CB_h075", dict(N=32, Lbox=24.0), "CB", 0.3),
    ("EQ_d125", dict(N=32), "EQ", 0.125),
]

out = {"rho0": d["RHO0"], "eps": d["EPS"], "r0": d["R0"], "cases": {}}
for tag, kw, kind, delta in CASES:
    L = d["load_stack"](delta=delta, **kw)
    M0 = d["seed"](L, kind, delta=delta)
    d["install_seed"](L, M0)
    pre = d["measures"](L, M0)
    Mf, E_levels, ginf = d["relax"](L)
    post = d["measures"](L, Mf)
    rec = {"N": L["N"], "Lbox": float(L["N"] * L["H"]), "H": L["H"],
           "kind": kind, "delta": delta,
           "E_levels": E_levels, "E_final": E_levels[-1],
           "ginf": ginf, "pre": pre, "post": post}
    out["cases"][tag] = rec
    np.savez_compressed(os.path.join(HERE, "results", f"M_{tag}.npz"),
                        M=Mf.cpu().numpy())
    print(f"[{tag}] E {E_levels[0]:.6f}->{E_levels[-1]:.6f} "
          f"(|g| {ginf:.1e}); I_comb {pre['I_comb']:.3e}->"
          f"{post['I_comb']:.3e} (PR {post['PR']:.0f}); "
          f"lam_z-lam_x {post['lambda_axis_excess']:.4e}", flush=True)
    del L, M0, Mf
    torch.cuda.empty_cache()

json.dump(out, open(os.path.join(HERE, "results", "relax_all.json"),
                    "w"), indent=1)
with open(FLAG, "w") as f:
    f.write("relaxations computed in this run\n")
print("written: results/relax_all.json")
