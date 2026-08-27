"""Route 2 for the load-bearing lattice numbers: an independent numpy
re-implementation of the (I1)^2 energies, evaluated on the persisted J1
rung fields.

No torch, no import of lattice.py -- the definitions (one-sided
differences, eta-commutator, Lagrange-projector Euclideanizer G, pinned
potential V4, static I1 density, boost-channel density, the local
(I1)^2 term) are coded from scratch against the report-002/004/007
formulas. Asserts:
- E_total of the persisted J1 rungs (omega = 0.2, 0.35, 0.5) matches
  results/i1sq_ladders.json to 1e-9 relative;
- the sampled well shape E(0.2) > E(0.35) < E(0.5) holds in this
  independent evaluation.
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

need = [os.path.join(HERE, "results", f"j1_rung_om{t}.npz")
        for t in ("02", "035", "05")]
if not all(os.path.exists(p) for p in need):
    print("verify_energies: NOT REPRODUCED HERE -- persisted rung fields "
          "absent (ladder_i1sq.py writes them when 004 fields are "
          "available).")
    sys.exit(0)

res = json.load(open(os.path.join(HERE, "results", "i1sq_ladders.json")))
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


def inner_const(F, X):
    return np.einsum("...ab,ac,bd,...cd->...", F, X, X, F)


def i1_stat_density(M):
    G = G_of(M)
    e = 0.0
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                e = e + 0.5 * 4.0 * inner_pc(comm(A[i], A[j]), G)
    return e


def v4_density(M):
    Me = M @ ETA
    P, v4 = Me, 0.0
    for p in range(4):
        if p:
            P = P @ Me
        v4 = v4 + (np.einsum("...kk->...", P) - C_P[p]) ** 2
    return v4


def e_static(M):
    return Hh ** 3 * (i1_stat_density(M).sum() + W1 * v4_density(M).sum())


def bk_of(M, a0, omega):
    G = G_of(M)
    V = omega * a0
    bk = 0.0
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F = comm(V, A[i])
            bk = bk + 0.5 * 4.0 * (inner_pc(F, G) - inner_const(F, ETA)) / 2
    return bk


A0_FROZEN = np.load(os.path.join(HERE, "results", "a0_frozen.npz"))["a0"]

rows = {r["omega"]: r for r in res["J1_local_covariant"]["rungs"]}
E_tot = {}
worst = 0.0
for om, tag in ((0.2, "02"), (0.35, "035"), (0.5, "05")):
    M = np.load(os.path.join(HERE, "results", f"j1_rung_om{tag}.npz"))["M"]
    Es = e_static(M)
    i1s = i1_stat_density(M)
    bk = bk_of(M, A0_FROZEN, om)
    Ex = GAMMA * Hh ** 3 * ((i1s - bk) ** 2).sum()
    E = Es + Ex
    E_tot[om] = E
    ref = rows[om]["E_total"]
    rel = abs(E - ref) / abs(ref)
    worst = max(worst, rel)
    print(f"omega {om}: E_numpy = {E:.6f}  vs ladder {ref:.6f}  "
          f"(rel {rel:.1e})")

print(f"worst relative difference: {worst:.2e}")
assert worst < 1e-9, "route-2 energies must match the ladder record"
assert E_tot[0.2] > E_tot[0.35] < E_tot[0.5], \
    "sampled interior well must survive"
print("ROUTE-2 ENERGIES MATCH; sampled interior well confirmed "
      "independently")
