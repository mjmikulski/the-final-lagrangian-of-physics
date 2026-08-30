"""Round-1 fixes, computed:

(A) absolute-potential control row: delta = 1/8, g in {8, 64, 512}
    with the ORIGINAL absolute potential W1*sum_p (t_p - C_p)^2 --
    the same observables as the grid; comparing the g-trends of the
    two potential variants tests whether the measured slopes belong
    to g or to the relative variant's g-dependent pinning stiffness.
    A per-point cancellation diagnostic (ulp(t_p) vs |t_p - C_p|)
    replaces the assumption that g = 512 is numerically unsafe.
(C) delta extension toward the target: delta in {1e-6, 1e-9} at
    g = 8 (relative variant, same protocol) -- sampling NEAR the
    proposed 1e-10 rather than extrapolating across unbounded
    crossover scales.
Out: results/fix_round1.json
"""
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
LAT = os.path.join(R004, "lattice.py")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "fix1_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("fix_round1: NOT REPRODUCED HERE -- needs the 004 stack.")
    sys.exit(0)
SRC = open(LAT).read()


def load_stack(delta, g, potential):
    src = SRC.replace("SG, DELTA, W1 = 8.0, 0.3, 0.000724023879",
                      f"SG, DELTA, W1 = {float(g)}, {delta}, "
                      "0.000724023879")
    if potential == "relative":
        src = src.replace("v4 = v4 + (t - C_P[p]) ** 2",
                          "v4 = v4 + (t / C_P[p] - 1.0) ** 2")
    ns = {"__name__": "not_main", "__file__": LAT}
    exec(compile(src, "lattice_patched", "exec"), ns)
    return ns


def relax(L, adam=1000, cycles=4):
    field, e_static = L["field"], L["e_static"]
    M_raw = L["seed_embedded"]().clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(adam):
        opt.zero_grad()
        e_static(field(M_raw), "G").backward()
        opt.step()
    E_levels = [float(e_static(field(M_raw), "G").detach())]
    for cyc in range(cycles):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_static(field(M_raw), "G")
            E.backward()
            return E
        opt2.step(closure)
        E_levels.append(float(e_static(field(M_raw), "G").detach()))
    g_ = torch.autograd.grad(e_static(field(M_raw), "G"), M_raw)[0]
    return M_raw.detach(), E_levels, float(g_.abs().max())


def densities(L, M, a0, om):
    d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
    DT, DEV = L["DT"], L["DEV"]
    G = G_of(M)
    V = om * a0
    i1s = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F0 = comm(V, A[i])
            k = k + 0.5 * 4.0 * torch.einsum(
                "...ab,...ac,...bd,...cd->...", F0, G, G, F0)
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                i1s = i1s + 0.5 * 4.0 * torch.einsum(
                    "...ab,...ac,...bd,...cd->...", F, G, G, F)
    return i1s, k


def true_dual_precision(L, Mf):
    """(B) the honest float32 evaluation: recompute the static energy
    entirely in float32 numpy (derivatives, commutators, G, potential,
    sums) and compare with the float64 value."""
    ETA = L["ETA"].cpu().numpy()
    SG, DELTA, W1, H = (float(L["SG"]), float(L["DELTA"]),
                        float(L["W1"]), float(L["H"]))
    C_P = [float(c) for c in L["C_P"]]
    rel = "v4 = v4 + (t / C_P[p] - 1.0) ** 2" in L.get(
        "__patched_rel", "") if False else None

    def e_np(M, dtype):
        M = M.astype(dtype)
        eta = ETA.astype(dtype)
        x = np.einsum("ab,...bc->...ac", eta, M)
        I4 = np.broadcast_to(np.eye(4, dtype=dtype), M.shape)
        q = (x @ (x - I4) @ (x - dtype(DELTA) * I4)) / dtype(
            SG * (SG - 1) * (SG - DELTA))
        G = eta - dtype(2.0) * q @ eta
        e_u = np.zeros(M.shape[:3], dtype=dtype)
        for st in ("fwd", "bwd"):
            A = []
            for ax in range(3):
                o = np.zeros_like(M)
                lo = [slice(None)] * 5
                hi = [slice(None)] * 5
                sl = [slice(None)] * 5
                lo[ax], hi[ax] = slice(0, -1), slice(1, None)
                sl[ax] = slice(0, -1) if st == "fwd" else slice(1, None)
                o[tuple(sl)] = (M[tuple(hi)] - M[tuple(lo)]) / dtype(H)
                A.append(o)
            for i in range(3):
                for j in range(i + 1, 3):
                    F = (A[i] @ eta @ A[j] - A[j] @ eta @ A[i])
                    e_u = e_u + dtype(0.5 * 4.0) * np.einsum(
                        "...ab,...ac,...bd,...cd->...", F, G, G, F)
        Me = M @ eta
        P, v4 = Me, np.zeros(M.shape[:3], dtype=dtype)
        for p in range(4):
            if p:
                P = P @ Me
            t = np.einsum("...kk->...", P)
            v4 = v4 + (t / dtype(C_P[p]) - dtype(1.0)) ** 2
        return float(dtype(H) ** 3 * (e_u.sum() + dtype(W1) * v4.sum()))

    Mnp = Mf.cpu().numpy()
    E64 = e_np(Mnp, np.float64)
    E32 = e_np(Mnp, np.float32)
    return abs(E32 - E64) / abs(E64), E64, E32


def measure_point(delta, g, potential):
    L = load_stack(delta, g, potential)
    field, e_static = L["field"], L["e_static"]
    H = L["H"]
    M_raw, E_levels, ginf = relax(L)
    Mf = field(M_raw)
    a0 = L["a0_of"](L["gen_catalog"]()["boost_x"], Mf)
    i1s, k = densities(L, Mf, a0, 1.0)
    tG = float(H ** 3 * k.sum())
    C1 = float(H ** 3 * (i1s * k).sum())
    C2 = float(H ** 3 * (k ** 2).sum())
    rec = {"delta": delta, "g": g, "potential": potential,
           "E_stat": E_levels[-1], "ginf": ginf,
           "time_part_G": -tG, "C1": C1, "C2": C2,
           "om_pred": (max(C1, 0) / C2) ** 0.5 if C2 > 0 else 0.0}
    # cancellation diagnostic for the ABSOLUTE potential: signal vs ulp
    Me = Mf @ L["ETA"]
    P = Me
    canc = []
    for p in range(4):
        if p:
            P = P @ Me
        t = torch.einsum("...kk->...", P)
        sig = float((t - L["C_P"][p]).abs().median())
        ulp = math.ulp(float(L["C_P"][p]))
        canc.append({"p": p + 1, "median_signal": sig, "ulp_C": ulp,
                     "signal_over_ulp": sig / ulp if ulp else None})
    rec["cancellation"] = canc
    # (B) true dual-precision on this field (relative-potential energy)
    f32rel, E64, E32 = true_dual_precision(L, Mf)
    rec["true_float32_rel"] = f32rel
    print(f"[{potential} d={delta:g} g={g}] E {rec['E_stat']:.4f} "
          f"(|g| {ginf:.1e}); tG {rec['time_part_G']:+.3e}; om_pred "
          f"{rec['om_pred']:.3f}; sig/ulp(p4) "
          f"{canc[3]['signal_over_ulp']:.1e}; TRUE f32rel {f32rel:.1e}",
          flush=True)
    del L, M_raw, Mf
    torch.cuda.empty_cache()
    return rec


out = {"A_absolute_row": [], "C_delta_extension": []}
for g in (8, 64, 512):
    out["A_absolute_row"].append(measure_point(0.125, g, "absolute"))
    json.dump(out, open(os.path.join(HERE, "results",
                                     "fix_round1.json"), "w"),
              indent=1)
for delta in (1e-6, 1e-9):
    out["C_delta_extension"].append(measure_point(delta, 8, "relative"))
    json.dump(out, open(os.path.join(HERE, "results",
                                     "fix_round1.json"), "w"),
              indent=1)
with open(FLAG, "w") as f:
    f.write("round-1 fixes computed in this run\n")
print("written: results/fix_round1.json")
