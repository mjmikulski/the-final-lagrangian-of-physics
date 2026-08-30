"""Round-2 fixes, computed:

(1) Converged absolute-potential rows at g = 8 and g = 64
    (delta = 1/8): relaxation continued under an OBSERVABLE-level
    stopping rule -- both the drive (time_part_G) and om_pred must
    drift < 1% over four consecutive L-BFGS cycles (max 30) -- with
    the full trajectory recorded, so the g-trend of the original
    theory is a statement about (observable-)converged profiles, not
    optimizer outputs. g = 512 stays excluded (documented breakdown).
(3) The full sign structure on those converged profiles: BOTH
    time_part_G and time_part_eta are measured.
(2) Sampling AT the proposed target and below: relative-variant
    points at delta = 1e-10 and 1e-11 (g = 8). All quantities remain
    representable in float64 (the delta^p terms of C_p round away
    exactly when their physical contribution is below one ulp).
Out: results/fix_round2.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
LAT = os.path.join(R004, "lattice.py")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "fix2_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("fix_round2: NOT REPRODUCED HERE -- needs the 004 stack.")
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


def densities(L, M, a0, om, metric):
    d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
    DT, DEV, ETA = L["DT"], L["DEV"], L["ETA"]
    X = G_of(M) if metric == "G" else ETA.expand_as(M)
    V = om * a0
    k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    i1s = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F0 = comm(V, A[i])
            k = k + 0.5 * 4.0 * torch.einsum(
                "...ab,...ac,...bd,...cd->...", F0, X, X, F0)
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                i1s = i1s + 0.5 * 4.0 * torch.einsum(
                    "...ab,...ac,...bd,...cd->...", F, X, X, F)
    return i1s, k


def observables(L, Mf):
    H = L["H"]
    a0 = L["a0_of"](L["gen_catalog"]()["boost_x"], Mf)
    i1sG, kG = densities(L, Mf, a0, 1.0, "G")
    _, kE = densities(L, Mf, a0, 1.0, "eta")
    C1 = float(H ** 3 * (i1sG * kG).sum())
    C2 = float(H ** 3 * (kG ** 2).sum())
    # sign conventions matching the grid producer: the G-contraction
    # time part is reported as an energy drive (negative = drives);
    # the raw eta matrix-slot contraction of the time pairs is
    # negative on the field, and with the outer eta^00 = -1 the net
    # eta time part is its negation -- positive = inert (report 008).
    return {"time_part_G": -float(H ** 3 * kG.sum()),
            "time_part_eta": -float(H ** 3 * kE.sum()),
            "C1": C1, "C2": C2,
            "om_pred": (max(C1, 0) / C2) ** 0.5 if C2 > 0 else 0.0}


def run_point(delta, g, potential, adam=1000, maxc=30):
    L = load_stack(delta, g, potential)
    field, e_static = L["field"], L["e_static"]
    M_raw = L["seed_embedded"]().clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(adam):
        opt.zero_grad()
        e_static(field(M_raw), "G").backward()
        opt.step()
    traj = []

    def snap():
        Mf = field(M_raw.detach())
        o = observables(L, Mf)
        g_ = torch.autograd.grad(e_static(field(M_raw), "G"), M_raw)[0]
        o["E"] = float(e_static(Mf, "G"))
        o["ginf"] = float(g_.abs().max())
        return o
    traj.append(snap())
    stopped = False
    for cyc in range(maxc):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_static(field(M_raw), "G")
            E.backward()
            return E
        opt2.step(closure)
        traj.append(snap())
        if len(traj) > 4:
            ok = True
            for key in ("time_part_G", "om_pred"):
                vals = [t[key] for t in traj[-5:]]
                ref = abs(vals[-1])
                if ref and max(abs(vals[i + 1] - vals[i])
                               for i in range(4)) / ref > 0.01:
                    ok = False
            if ok:
                stopped = True
                break
    rec = {"delta": delta, "g": g, "potential": potential,
           "trajectory": traj, "final": traj[-1],
           "cycles": len(traj) - 1, "stopped_on_observable": stopped}
    print(f"[{potential} d={delta:g} g={g}] {rec['cycles']} cycles "
          f"(obs-stop {stopped}); tG {traj[-1]['time_part_G']:+.4e} "
          f"tE {traj[-1]['time_part_eta']:+.4e}; om_pred "
          f"{traj[-1]['om_pred']:.4f}; |g| {traj[-1]['ginf']:.1e}",
          flush=True)
    np.savez_compressed(
        os.path.join(HERE, "results",
                     f"M_fix2_{potential}_d{delta:g}_g{g}.npz"),
        M=field(M_raw.detach()).cpu().numpy())
    del L, M_raw
    torch.cuda.empty_cache()
    return rec


out = {"absolute_converged": [], "delta_at_target": []}
for g in (8, 64):
    out["absolute_converged"].append(run_point(0.125, g, "absolute"))
    json.dump(out, open(os.path.join(HERE, "results",
                                     "fix_round2.json"), "w"),
              indent=1)
for delta in (1e-10, 1e-11):
    out["delta_at_target"].append(
        run_point(delta, 8, "relative", maxc=8))
    json.dump(out, open(os.path.join(HERE, "results",
                                     "fix_round2.json"), "w"),
              indent=1)
with open(FLAG, "w") as f:
    f.write("round-2 fixes computed in this run\n")
print("written: results/fix_round2.json")
