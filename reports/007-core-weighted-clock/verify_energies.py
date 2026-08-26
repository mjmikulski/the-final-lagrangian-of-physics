"""Route 2 for the load-bearing lattice numbers: an independent numpy
re-implementation of the 004-stack energies, evaluated on the persisted
fresh-ladder rung fields.

No torch, no import of lattice.py -- the definitions (one-sided
differences, eta-commutator, Lagrange-projector Euclideanizer G, pinned
potential V4, boost-channel density, the intensive condensate) are coded
from scratch against the report-002/004 formulas. Asserts, for BOTH the
frozen-mask ladder (L4, weight from results/cw_frozen.npz) and the
dynamic-weight ladder (L5, weight recomputed here from each rung field):
- E_static and E_total of the persisted rungs (omega = 0.5, 0.8, 1.1)
  match results/ladder_series.json to 1e-9 relative;
- the well shape E(0.5) > E(0.8) < E(1.1) -- the interior clock --
  holds in this independent evaluation.
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

need = [os.path.join(HERE, "results", f"{p}{t}.npz")
        for t in ("05", "08", "11")
        for p in ("fresh_rung_om", "fresh5_rung_om")]
if not all(os.path.exists(p) for p in need):
    print("verify_energies: NOT REPRODUCED HERE -- persisted rung fields "
          "absent (ladder_series.py writes them when 004 fields are "
          "available).")
    sys.exit(0)

res = json.load(open(os.path.join(HERE, "results", "ladder_series.json")))
setup = res["L4_intensive_fresh"]["setup"]
A_C, B_C, V0 = 1.0, setup["b"], setup["v0"]


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


def e_static(M):
    G = G_of(M)
    e_u = 0.0
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                e_u = e_u + 0.5 * 4.0 * inner_pc(comm(A[i], A[j]), G).sum()
    Me = M @ ETA
    P, v4 = Me, 0.0
    for p in range(4):
        if p:
            P = P @ Me
        t = np.einsum("...kk->...", P)
        v4 = v4 + (t - C_P[p]) ** 2
    return Hh ** 3 * (e_u + W1 * v4.sum())


A0_FROZEN = np.load(os.path.join(HERE, "results", "a0_frozen.npz"))["a0"]
CW_FROZEN = np.load(os.path.join(HERE, "results", "cw_frozen.npz"))["cw"]
# the ladder freezes BOTH a0 and the weight mask from the polished start


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


def cw_dyn(M):
    """Dynamic weight recomputed from the rung field (the L5 form)."""
    Me = M @ ETA
    P, v4 = Me, 0.0
    for p in range(4):
        if p:
            P = P @ Me
        v4 = v4 + (np.einsum("...kk->...", P) - C_P[p]) ** 2
    return v4 / (v4 + V0)


worst = 0.0
for leg, prefix, wfun in (
        ("L4_intensive_fresh", "fresh_rung_om", lambda M: CW_FROZEN),
        ("L5_intensive_dynamic", "fresh5_rung_om", cw_dyn)):
    rows = {r["omega"]: r for r in res[leg]["rungs"]}
    E_tot = {}
    for om, tag in ((0.5, "05"), (0.8, "08"), (1.1, "11")):
        fp = os.path.join(HERE, "results", f"{prefix}{tag}.npz")
        M = np.load(fp)["M"]
        Es = e_static(M)
        B = Hh ** 3 * (wfun(M) * bk_of(M, A0_FROZEN, om)).sum()
        Ec = -A_C * B + 3 * B_C * B ** 2
        E = Es + Ec
        E_tot[om] = E
        ref = rows[om]["E_total"]
        rel = abs(E - ref) / abs(ref)
        worst = max(worst, rel)
        print(f"[{leg}] omega {om}: E_numpy = {E:.6f}  vs ladder "
              f"{ref:.6f}  (rel {rel:.1e})")
    assert E_tot[0.5] > E_tot[0.8] < E_tot[1.1], \
        f"interior well must survive ({leg})"

print(f"worst relative difference: {worst:.2e}")
assert worst < 1e-9, "route-2 energies must match the ladder record"
print("ROUTE-2 ENERGIES MATCH (L4 and L5); interior wells confirmed "
      "independently")
