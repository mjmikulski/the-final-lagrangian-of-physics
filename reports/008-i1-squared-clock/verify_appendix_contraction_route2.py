"""Route 2 for APPENDIX-contraction-release: the report's from-scratch numpy energy route (the functions of verify_L_ladder_energies.py
copied here, no torch) evaluated on the persisted fields of this appendix (float32, so energies agree with the float64 records
up to a common offset of 1.7e-5 from the float32 rounding of the stored fields; the offset is the same for every
field to 1e-7, so every energy difference is reproduced to that level, well below the depths of 3-7e-5).

Frobenius ladder: the same energy with the identity in place of G in both densities of the (I1)^2 term (statics
in G). Frozen tangent: rebuilt from the polished field. Released tangent: rebuilt from the evaluated field itself.
Asserts the qualitative content in this route: the 0.35 rung lies below omega = 0 by more than 5e-5 for the
Frobenius ladder, for the frozen and for the released tangent, and the 0.5 and 0.8 Frobenius rungs lie above.
Writes results/contraction_release/route2.json.
"""
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "contraction_release")
I4 = np.eye(4)

# ---- the report's numpy route, copied verbatim from verify_L_ladder_energies.py (whose module body runs the L-ladder
# check on release-asset fields, so it is not imported) ----
Hh = 1.5
SG, DELTA, W1 = 8.0, 0.3, 0.000724023879
C_P = tuple(SG ** p + 1.0 + DELTA ** p for p in range(1, 5))
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
GAMMA = json.load(open(os.path.join(HERE, "results", "i1sq_ladders.json")))["gamma"]


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


class v:  # namespace so the calls below read as in the source route
    pass


for _n in ("d1", "comm", "G_of", "inner_pc", "v4_density", "densities_G", "energy", "tangent", "pinned_field", "Hh", "W1", "GAMMA"):
    setattr(v, _n, globals()[_n])


def e_frob(M, a0, om):
    G = v.G_of(M)
    i1G, i1F, kF = 0.0, 0.0, 0.0
    for st in ("fwd", "bwd"):
        A = [v.d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            kF = kF + 0.5 * 4.0 * v.inner_pc(v.comm(om * a0, A[i]), np.broadcast_to(I4, M.shape))
            for j in range(i + 1, 3):
                F = v.comm(A[i], A[j])
                i1G = i1G + 0.5 * 4.0 * v.inner_pc(F, G)
                i1F = i1F + 0.5 * 4.0 * v.inner_pc(F, np.broadcast_to(I4, M.shape))
    Es = v.Hh ** 3 * (i1G.sum() + v.W1 * v.v4_density(M).sum())
    return Es + v.GAMMA * v.Hh ** 3 * ((i1F - kF) ** 2).sum()


def load(name):
    return np.load(os.path.join(RES, name))["M"].astype(np.float64)


if __name__ == "__main__":
    seed = np.load(os.path.join(HERE, "results", "L_ladder", "seeds", "m5_21_2b_end_A_T2_sym_e0_n32_d0.3_pinned.npz"))["M"]
    a0 = v.tangent(v.pinned_field(np.load(os.path.join(HERE, "results", "L_ladder", "M_G_polished_N32.npz"))["M"].astype(np.float64),
                                  32, seed.astype(np.float64)))
    out = {"frob": {}, "frozen8": {}, "released8": {}}
    rec = {r["omega"]: r["E_total"] for r in json.load(open(os.path.join(RES, "frob.json")))["rungs"]}
    for om in (0.0, 0.1, 0.2, 0.35, 0.5, 0.8):
        E = float(e_frob(load(f"frob_om{str(om).replace('.', '')}.npz"), a0, om))
        out["frob"][str(om)] = {"E": E, "record": rec[om], "dev": abs(E - rec[om])}
    rec = {r["omega"]: r["E_total"] for r in json.load(open(os.path.join(RES, "frozen8.json")))["rungs"]}
    for om in (0.0, 0.35):
        E = float(v.energy(load(f"frozen8_om{str(om).replace('.', '')}.npz"), a0, om))
        out["frozen8"][str(om)] = {"E": E, "record": rec[om], "dev": abs(E - rec[om])}
    rec = {r["omega"]: r["E_total"] for r in json.load(open(os.path.join(RES, "released8.json")))["rungs"]}
    for om in (0.2, 0.35):
        M = load(f"released8_om{str(om).replace('.', '')}.npz")
        E = float(v.energy(M, v.tangent(M), om))
        out["released8"][str(om)] = {"E": E, "record": rec[om], "dev": abs(E - rec[om])}
    devs = [x["E"] - x["record"] for grp in out.values() for x in grp.values()]
    out["common_offset"] = float(np.mean(devs))           # float32 storage of the fields: the same for every field
    out["offset_spread"] = float(max(devs) - min(devs))   # what matters for energy differences
    f, fz, rl = out["frob"], out["frozen8"], out["released8"]
    out["depths"] = {"frob 0.35": f["0.0"]["E"] - f["0.35"]["E"], "frozen 0.35": fz["0.0"]["E"] - fz["0.35"]["E"],
                     "released 0.35": fz["0.0"]["E"] - rl["0.35"]["E"], "released 0.2": fz["0.0"]["E"] - rl["0.2"]["E"]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RES, "route2.json"), "w"), indent=1)
    assert out["offset_spread"] < 2e-7, "route 2 reproduces every energy difference of the records (float32 fields)"
    assert all(out["depths"][k] > 5e-5 for k in ("frob 0.35", "frozen 0.35", "released 0.35"))
    assert out["depths"]["released 0.2"] > 2e-5
    assert f["0.5"]["E"] > f["0.0"]["E"] and f["0.8"]["E"] > f["0.0"]["E"]
    print("route 2: all assertions pass")
