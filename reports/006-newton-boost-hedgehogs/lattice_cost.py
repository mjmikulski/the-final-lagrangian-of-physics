"""Cost meter: how large the I4 channel is on the WORKING 3x3 physics.

The no-go forces any quadratic Newton-sign modification outside the
exact-3x3-preserving family, i.e. to carry a nonzero I4-channel component
on spatial fields. This measures integral(I4)/integral(I1) on the relaxed,
gradient-polished 3x3 electron hedgehog of report 004.

REPRODUCIBILITY STATUS (explicit, per review): the input field
results/M_G_polished.npz of report 004 is regenerated only by
../004-lattice-clock/reproduce.sh (a GPU-hour run), so this measurement is
an EXTERNAL RESULT relative to this report's own reproduce.sh. The
recorded values live in results/lattice_cost_external.json and are marked
as such; when the 004 artifact IS present, this script recomputes the
ratio and ASSERTS agreement with the external record, writing
results/lattice_cost.json. Without the artifact it exits with a loud
NOT-REPRODUCED-HERE notice (and reproduce.sh reports the cost claim as
external, never as independently certified).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
MPATH = os.path.join(R004, "results", "M_G_polished.npz")
EXTERNAL = json.load(open(os.path.join(HERE, "results",
                                       "lattice_cost_external.json")))
if not os.path.exists(MPATH):
    stale = os.path.join(HERE, "results", "lattice_cost.json")
    if os.path.exists(stale):
        os.remove(stale)            # a leftover from an earlier run must
        print("lattice_cost: removed stale results/lattice_cost.json")
    print("lattice_cost: NOT REPRODUCED HERE -- report 004's regenerated "
          "artifact results/M_G_polished.npz is absent (run "
          "../004-lattice-clock/reproduce.sh first). The I4/I1 = "
          f"{EXTERNAL['free_bulk_mean_ratio']} figure is an external "
          "result recorded in results/lattice_cost_external.json.")
    sys.exit(0)

sys.path.insert(0, R004)
from lattice import DT, DEV, ETA, H, FREE, d1, comm    # noqa: E402

M = torch.tensor(np.load(MPATH)["M"], dtype=DT, device=DEV)
out = {}
for st in ("fwd", "bwd"):
    A = [d1(M, ax, st) for ax in range(3)]
    e1 = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    Phi = torch.zeros(*M.shape[:3], 3, 4, dtype=DT, device=DEV)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            F = comm(A[i], A[j])
            e1 += torch.einsum("...ab,ac,bd,...cd->...", F, ETA, ETA, F)
            # Phi_{j b} = sum_i F_{ij, i b}: matrix-row index i+1
            Phi[..., j, :] += F[..., i + 1, :]
    e4 = torch.einsum("...jb,bc,...jc->...", Phi, ETA, Phi)
    for tag, mask in (("all", torch.ones_like(e1, dtype=torch.bool)),
                      ("free", FREE)):
        I1 = (e1 * mask).sum().item() * H ** 3
        I4 = (e4 * mask).sum().item() * H ** 3
        out[f"{st}_{tag}"] = {"I1": I1, "I4": I4, "ratio": I4 / I1}
        print(f"{st} {tag:4s}: int I1 = {I1:12.5f}   int I4 = {I4:12.5f}"
              f"   I4/I1 = {I4 / I1:.4f}")

ratio = float(np.mean([v["ratio"] for k, v in out.items()
                       if k.endswith("free")]))
out["free_bulk_mean_ratio"] = ratio
print(f"I4/I1 on the relaxed 3x3 hedgehog (free bulk): {ratio:.4f}")
# the regenerated 004 polish is stochastic-free but not bit-identical
# across machines; agree with the external record to a loose tolerance
drift = abs(ratio - EXTERNAL["free_bulk_mean_ratio"])
print(f"drift vs external record: {drift:.4f}")
assert drift < 0.1, "regenerated cost disagrees with the external record"
with open(os.path.join(HERE, "results", "lattice_cost.json"), "w") as f:
    json.dump(out, f, indent=1)
print("written: results/lattice_cost.json (external record confirmed)")
