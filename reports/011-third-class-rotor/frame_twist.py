"""Producer 4: the spectrally-neutral third-class candidate.

The spectral core deformation (CB: eps*chi*D with D = xx-yy) changes
the eigenvalues, so V4 punishes it and relaxation largely removes it
(relax_all.py: I_CB - I_EQ collapses to ~5 on a diffuse background).
The natural V4-NEUTRAL third-class representative is a pure FRAME
TWIST: rotate the local eigenframe by beta(r) = beta0 * exp(-(r/r0)^2)
about the fixed x axis,

    S3_CB2(x) = R_x(beta(r)) S3_EQ(x) R_x(beta(r))^T .

Eigenvalues are untouched pointwise (V4 identical to EQ's); the twist
axis x != z breaks axial equivariance in the core only. Measured here:
relax EQ and CB2 at N = 24 with the same deep protocol, compare
I_comb, E, and the localization of the EXCESS kinetic density
k[zeta](CB2) - k[zeta](EQ). Out: results/frame_twist.json
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
FLAG = os.path.join(HERE, "results", "twist_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("frame_twist: NOT REPRODUCED HERE -- needs the 004 stack.")
    sys.exit(0)

d = runpy.run_path(os.path.join(HERE, "defs_011.py"))
BETA0 = 0.6

out = {"beta0": BETA0, "cases": {}}
kd_post = {}
for kind in ("EQ", "CB2"):
    L = d["load_stack"](N=24)
    M0 = d["seed"](L, "EQ")
    if kind == "CB2":
        DT, DEV = L["DT"], L["DEV"]
        _, _, _, r, _, _, _ = d["geometry"](L)
        beta = BETA0 * torch.exp(-(r / d["R0"]) ** 2)
        c, s_ = torch.cos(beta), torch.sin(beta)
        Rx = torch.zeros(*beta.shape, 3, 3, dtype=DT, device=DEV)
        Rx[..., 0, 0] = 1.0
        Rx[..., 1, 1], Rx[..., 2, 2] = c, c
        Rx[..., 1, 2], Rx[..., 2, 1] = -s_, s_
        S3 = M0[..., 1:4, 1:4]
        M0 = M0.clone()
        M0[..., 1:4, 1:4] = Rx @ S3 @ Rx.transpose(-1, -2)
    d["install_seed"](L, M0)
    pre = d["measures"](L, M0)
    Mf, E_levels, ginf = d["relax"](L)
    post = d["measures"](L, Mf)
    z = d["zeta_of"](L, Mf)
    kd_post[kind] = d["kin_density"](L, Mf, z)
    out["cases"][kind] = {"E_levels": E_levels,
                          "E_final": E_levels[-1], "ginf": ginf,
                          "pre": pre, "post": post}
    print(f"[{kind}] E {E_levels[0]:.6f}->{E_levels[-1]:.6f} "
          f"(|g| {ginf:.1e}); I {pre['I_comb']:.3e}->"
          f"{post['I_comb']:.3e} (PR {post['PR']:.0f})", flush=True)
    np.savez_compressed(os.path.join(HERE, "results",
                                     f"M_{kind}_twist24.npz"),
                        M=Mf.cpu().numpy())
    del M0, Mf, z

# the third-class signal: the EXCESS kinetic density over the shared
# equivariant background, and its localization
L = d["load_stack"](N=24)
H = L["H"]
dk = (kd_post["CB2"] - kd_post["EQ"]).clamp_min(0.0)
I_excess = float(2.0 * H ** 3 * dk.sum())
pr_excess = float((dk.sum() ** 2) / (dk ** 2).sum().clamp_min(1e-30))
out["I_excess"] = I_excess
out["PR_excess"] = pr_excess
out["I_diff_raw"] = float(2.0 * H ** 3
                          * (kd_post["CB2"] - kd_post["EQ"]).sum())
print(f"third-class signal: I_excess {I_excess:.4e} "
      f"(raw diff {out['I_diff_raw']:.4e}), PR_excess {pr_excess:.0f}")
json.dump(out, open(os.path.join(HERE, "results", "frame_twist.json"),
                    "w"), indent=1)
with open(FLAG, "w") as f:
    f.write("frame twist computed in this run\n")
print("written: results/frame_twist.json")
