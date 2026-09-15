"""Route 2 for the box ladder: the report's independent numpy energy route (verify_energies.py) extended to
any box, evaluated on the persisted bracket fields of every box.

No torch, no import of lattice_L.py: the definitions are the same from-scratch numpy ones as in
verify_energies.py (one-sided differences, eta-commutator, Lagrange-projector Euclideanizer, V4, the static
I1_G density and the G-metric time density, the energy reading of the quartic), with N read from the field
and h = 1.5. The frozen tangent is rebuilt here in numpy from the box's polished field (envelope
exp(-(r/10)^4) times the boost-x conjugation tangent, unit Frobenius norm). For each box the persisted rungs
omega in {0, 0.1, 0.2, 0.35} are evaluated and compared with the records: the first ladder run
(ladder_N*.json) for the rungs it persisted (0 and 0.35), the bracket rerun (rungs_N*.json) for 0.1 and 0.2.
Asserts: every persisted energy matches its record to 1e-9 relative, and in every box the sampled well
(minimum below its neighbours and below omega = 0) holds in this independent evaluation.
Writes results/L_ladder/independent_route.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "L_ladder")
Hh = 1.5
SG, DELTA, W1 = 8.0, 0.3, 0.000724023879
C_P = tuple(SG ** p + 1.0 + DELTA ** p for p in range(1, 5))
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
GAMMA = json.load(open(os.path.join(HERE, "results", "i1sq_ladders.json")))["gamma"]
BRACKET = (0.0, 0.1, 0.2, 0.35)


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


def energy(M, a0, om):
    i1s, k = densities_G(M, a0, om)
    Es = Hh ** 3 * (i1s.sum() + W1 * v4_density(M).sum())
    return Es + GAMMA * Hh ** 3 * ((i1s - k) ** 2).sum()


def tangent(M):
    """Frozen boost-x tangent of the polished field, as in 004's lattice.py, rebuilt in numpy."""
    N = M.shape[0]
    x = (np.arange(N) - (N - 1) / 2.0) * Hh
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    env = np.exp(-((np.sqrt(X * X + Y * Y + Z * Z) / 10.0) ** 4))
    W = np.zeros((4, 4)); W[0, 1] = W[1, 0] = 1.0
    a = env[..., None, None] * (np.einsum("ab,...bc->...ac", W, M) + np.einsum("...ab,cb->...ac", M, W))
    return a / np.linalg.norm(a)


def pinned_field(M_raw, N, seed3):
    """The field as the stack sees it: shell of physical depth 1.6 frozen at the embedded seed."""
    wc = max(1, int(np.ceil(1.6 / Hh)))
    mask = np.zeros((N, N, N), dtype=bool)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = slice(0, wc); mask[tuple(sl)] = True
        sl[ax] = slice(N - wc, N); mask[tuple(sl)] = True
    seed4 = np.zeros((N, N, N, 4, 4)); seed4[..., 1:, 1:] = seed3; seed4[..., 0, 0] = -SG
    Ms = 0.5 * (M_raw + np.swapaxes(M_raw, -1, -2))
    return np.where(mask[..., None, None], seed4, Ms)


out, worst = {}, 0.0
for N in (32, 48, 64):
    lad = json.load(open(os.path.join(RES, f"ladder_N{N}.json")))
    rer_path = os.path.join(RES, f"rungs_N{N}.json")
    rer = json.load(open(rer_path)) if os.path.exists(rer_path) else None
    seed3 = np.load(os.path.join(RES, "seeds", f"m5_21_2b_end_A_T2_sym_e0_n{N}_d0.3_pinned.npz"))["M"].astype(np.float64)
    pol = np.load(os.path.join(RES, f"M_G_polished_N{N}.npz"))["M"]
    a0 = tangent(pinned_field(pol, N, seed3))
    rows = {}
    for om in BRACKET:
        f = os.path.join(RES, f"jge_N{N}_om{str(om).replace('.', '')}.npz")
        if not os.path.exists(f):
            print(f"N = {N} omega {om}: field absent")
            continue
        M = pinned_field(np.load(f)["M"], N, seed3)
        E = energy(M, a0, om)
        # the record this field belongs to: the first ladder run persisted 0 and 0.35, the rerun 0.1 and 0.2
        src = rer if (rer and om in [r["omega"] for r in rer["rungs"]]) else lad
        ref = [r for r in src["rungs"] if r["omega"] == om][0]["E_total"]
        rel = abs(E - ref) / abs(ref)
        worst = max(worst, rel)
        rows[om] = {"E_numpy": E, "E_record": ref, "rel_dev": rel, "record": "rungs rerun" if src is rer else "ladder"}
        print(f"N = {N} omega {om}: E_numpy {E:.9f} vs record {ref:.9f} (rel {rel:.1e}, {rows[om]['record']})")
    out[str(N)] = rows
    if all(o in rows for o in BRACKET):
        e = {o: rows[o]["E_numpy"] for o in BRACKET}
        kmin = min(e, key=e.get)
        out[str(N)]["well"] = {"min_at": kmin, "E_minus_E0": {str(o): e[o] - e[0.0] for o in BRACKET}}
        print(f"   independent evaluation: minimum of the bracket at omega = {kmin}, E - E(0) = {e[kmin] - e[0.0]:+.3e}")
out["worst_rel_dev"] = worst
json.dump(out, open(os.path.join(RES, "independent_route.json"), "w"), indent=1)
if "--strict" in sys.argv:
    assert worst < 1e-9, worst
    for N in (32, 48, 64):
        assert out[str(N)]["well"]["min_at"] in (0.2, 0.35) and out[str(N)]["well"]["E_minus_E0"][str(out[str(N)]["well"]["min_at"])] < 0
    print("ROUTE-2 ENERGIES MATCH IN EVERY BOX; the sampled wells hold independently")
