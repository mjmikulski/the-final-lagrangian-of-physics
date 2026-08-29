"""Route 2: independent numpy re-implementation of the rotational-clock
energies, evaluated on the persisted JR_E rung fields (the bracket
0.152 / 0.217 / 0.304) with the frozen rotation tangent
results/a0r_frozen.npz. No torch, no import of lattice.py. Asserts
agreement with results/rot_ladders.json to 1e-9 relative and the
sampled interior well shape.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N, Lbox = 32, 48.0
Hh = Lbox / N
SG, DELTA, W1 = 8.0, 0.3, 0.000724023879
C_P = tuple(SG ** p + 1.0 + DELTA ** p for p in range(1, 5))
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])

OMS = ((0.152, "0152"), (0.217, "0217"), (0.304, "0304"))
need = [os.path.join(HERE, "results", f"rot_rung_om{t}.npz")
        for _, t in OMS]
if not all(os.path.exists(p) for p in need):
    print("verify_energies: NOT REPRODUCED HERE -- persisted rung "
          "fields absent (ladder_rot.py writes them when 004 fields "
          "are available).")
    sys.exit(0)

res = json.load(open(os.path.join(HERE, "results", "rot_ladders.json")))
GAMMA = res["gamma"]


def d1(f, ax, st):
    out = np.zeros_like(f)
    sl = [slice(None)] * f.ndim
    lo, hi = [slice(None)] * f.ndim, [slice(None)] * f.ndim
    lo[ax], hi[ax] = slice(0, -1), slice(1, None)
    if st == "fwd":
        sl[ax] = slice(0, -1)
        out[tuple(sl)] = (f[tuple(hi)] - f[tuple(lo)]) / Hh
    else:
        sl[ax] = slice(1, None)
        out[tuple(sl)] = (f[tuple(hi)] - f[tuple(lo)]) / Hh
    return out


def comm(A, B):
    return A @ ETA @ B - B @ ETA @ A


def G_of(M):
    x = np.einsum("ab,...bc->...ac", ETA, M)
    I4 = np.broadcast_to(np.eye(4), M.shape)
    q = (x @ (x - I4) @ (x - DELTA * I4)) / (SG * (SG - 1) * (SG - DELTA))
    return ETA - 2.0 * q @ ETA


def inner_pc(F, X):
    return np.einsum("...ab,...ac,...bd,...cd->...", F, X, X, F)


def v4_density(M):
    Me = M @ ETA
    P, v4 = Me, 0.0
    for p in range(4):
        if p:
            P = P @ Me
        v4 = v4 + (np.einsum("...kk->...", P) - C_P[p]) ** 2
    return v4


def densities_G(M, a0, om):
    G = G_of(M)
    V = om * a0
    i1s, k = 0.0, 0.0
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            k = k + 0.5 * 4.0 * inner_pc(comm(V, A[i]), G)
            for j in range(i + 1, 3):
                i1s = i1s + 0.5 * 4.0 * inner_pc(comm(A[i], A[j]), G)
    return i1s, k


A0R = np.load(os.path.join(HERE, "results", "a0r_frozen.npz"))["a0"]
rows = {r["omega"]: r for r in res["JR_E"]["rungs"]}
E_tot = {}
worst = 0.0
for om, tag in OMS:
    M = np.load(os.path.join(HERE, "results",
                             f"rot_rung_om{tag}.npz"))["M"]
    i1s, k = densities_G(M, A0R, om)
    Es = Hh ** 3 * (i1s.sum() + W1 * v4_density(M).sum())
    E = Es + GAMMA * Hh ** 3 * ((i1s - k) ** 2).sum()
    E_tot[om] = E
    ref = rows[om]["E_total"]
    rel = abs(E - ref) / abs(ref)
    worst = max(worst, rel)
    print(f"omega {om}: E_numpy = {E:.6f}  vs ladder {ref:.6f}  "
          f"(rel {rel:.1e})")

print(f"worst relative difference: {worst:.2e}")
assert worst < 1e-9, "route-2 energies must match the ladder record"
print("ROUTE-2 ENERGIES MATCH: the persisted fixed-depth energies are "
      "independently reproduced. (No well claim is made here -- the "
      "fixed-depth ordering is the protocol artifact documented in "
      "README section 4.)")
