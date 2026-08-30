"""Producer 1: the (delta, g) scaling grid. See PREREG.md (committed
in the working repository BEFORE the run, commit 9648b14).

Notation, introduced in full (this report assumes no notation from
earlier reports): M(x) is the symmetric 4x4 field of the model on a
32^3 lattice; the vacuum is diag(-g, 1, delta, 0), where g is the
timelike eigenvalue (sigma of earlier reports; the model author's
proposal has g ~ 1e10) and delta the smaller transverse eigenvalue
(author's proposal ~1e-10). V4 is the potential pinning the traces of
powers of (M eta) to their vacuum targets C_p; this run uses the
RELATIVE variant V4_rel = sum_p (tr((M eta)^p)/C_p - 1)^2 (dimension-
less; avoids catastrophic cancellation at g = 512, where the absolute
variant squares numbers ~1e21). The clock tangent a0 is the frozen
boost conjugation direction; i1s and k are the static and kinetic
parts of the invariant I1 = F.F contracted with the working metric G
(the field-dependent positive metric of the statics repair); omega is
the clock frequency, om_pred = sqrt(C1/C2) the frozen-profile
prediction of the energy-reading well position; I_pure and I_comb are
the rotational-channel inertias (internal generator, and combined
space-internal generator, both interior-masked); PR is the
participation ratio (effective site count) of a density.
"""
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
LAT = os.path.join(R004, "lattice.py")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "grid_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    import sys
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("grid_scan: NOT REPRODUCED HERE -- needs report 004's stack "
          "(the seed file it loads); committed results carry the "
          "record.")
    sys.exit(0)
SRC = open(LAT).read()
assert "v4 = v4 + (t - C_P[p]) ** 2" in SRC


def load_stack(delta, g):
    src = SRC.replace("SG, DELTA, W1 = 8.0, 0.3, 0.000724023879",
                      f"SG, DELTA, W1 = {float(g)}, {delta}, "
                      "0.000724023879")
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


def densities(L, M, a0, om, metric):
    d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
    DT, DEV, ETA = L["DT"], L["DEV"], L["ETA"]
    X = G_of(M) if metric == "G" else ETA.expand_as(M)
    V = om * a0
    i1s = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    sgn = 1.0 if metric == "G" else -1.0
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F0 = comm(V, A[i])
            k = k + sgn * 0.5 * 4.0 * torch.einsum(
                "...ab,...ac,...bd,...cd->...", F0, X, X, F0)
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                i1s = i1s + 0.5 * 4.0 * torch.einsum(
                    "...ab,...ac,...bd,...cd->...", F, X, X, F)
    return i1s, k


def point(delta, g):
    t0 = time.time()
    L = load_stack(delta, g)
    field, e_static = L["field"], L["e_static"]
    H, DT, DEV = L["H"], L["DT"], L["DEV"]
    M_raw, E_levels, ginf = relax(L)
    Mf = field(M_raw)
    rec = {"delta": delta, "g": g, "E_levels": E_levels,
           "E_stat": E_levels[-1], "ginf": ginf}

    # precision diagnostics
    Me = Mf @ L["ETA"]
    P = Me
    tr_rel = []
    for p in range(4):
        if p:
            P = P @ Me
        t = torch.einsum("...kk->...", P)
        tr_rel.append(float((t / L["C_P"][p] - 1.0).abs().max()))
    rec["tr_rel_max"] = tr_rel
    E64 = float(e_static(Mf, "G"))
    E32 = float(e_static(Mf.float().double(), "G"))
    M32 = Mf.float()
    L32 = L
    rec["float32_rel"] = abs(
        float(e_static(M32.double(), "G")) - E64) / abs(E64)

    # channel measurements
    a0 = L["a0_of"](L["gen_catalog"]()["boost_x"], Mf)
    i1sG, kG = densities(L, Mf, a0, 1.0, "G")
    _, kE = densities(L, Mf, a0, 1.0, "eta")
    tG = float((H ** 3 * kG.sum()))
    tE = float((H ** 3 * kE.sum()))
    rec["time_part_G"] = -tG          # drive convention: negative ticks
    rec["time_part_eta"] = tE
    C1 = float(H ** 3 * (i1sG * kG).sum())
    C2 = float(H ** 3 * (kG ** 2).sum())
    rec["C1"], rec["C2"] = C1, C2
    om_p = (max(C1, 0.0) / C2) ** 0.5 if C2 > 0 else 0.0
    rec["om_pred"] = om_p

    # mix-3/4 curvature
    env = L["envelope"]()
    P34 = torch.zeros(4, 4, dtype=DT, device=DEV)
    P34[2, 3] = P34[3, 2] = 1.0
    dM = env[..., None, None] * P34
    dM = dM / dM.norm()
    eps = 1e-3
    Ep = float(e_static(field(M_raw + eps * dM), "G"))
    Em = float(e_static(field(M_raw - eps * dM), "G"))
    rec["mix34_curv"] = (Ep - 2 * E_levels[-1] + Em) / eps ** 2

    # rotational inertias (fixed L = 48)
    W = L["gen_catalog"]()["rot_xy"]
    interior = (1.0 - L["SHELL"].to(DT))[..., None, None]
    Xc, Yc, _ = L["coords"]()
    g_pure = interior * (torch.einsum("ab,...bc->...ac", W, Mf)
                         - torch.einsum("...ab,bc->...ac", Mf, W))
    dMy, dMx = L["d1"](Mf, 1, "fwd"), L["d1"](Mf, 0, "fwd")
    orbital = -(Xc[..., None, None] * dMy - Yc[..., None, None] * dMx)
    g_comb = g_pure + interior * orbital
    for tag, tan in (("I_pure", g_pure), ("I_comb", g_comb)):
        G_ = L["G_of"](Mf)
        kk = torch.zeros(Mf.shape[:3], dtype=DT, device=DEV)
        for st in ("fwd", "bwd"):
            A = [L["d1"](Mf, ax, st) for ax in range(3)]
            for i in range(3):
                F0 = L["comm"](tan, A[i])
                kk = kk + 0.5 * 4.0 * torch.einsum(
                    "...ab,...ac,...bd,...cd->...", F0, G_, G_, F0)
        rec[tag] = float(2.0 * H ** 3 * kk.sum())

    # mini-ladder (energy-reading G-form quartic, 5% budget gamma)
    gamma = 0.05 * E_levels[-1] / float(H ** 3 * (i1sG ** 2).sum())
    rec["gamma"] = gamma
    rows = []
    for om in (0.0, round(om_p, 3), round(1.5 * om_p, 3)):
        def e_tot(Mr, om=om):
            Mfx = field(Mr)
            i1s, k = densities(L, Mfx, a0, max(om, 1e-9), "G")
            extra = gamma * H ** 3 * ((i1s - k) ** 2).sum()
            return e_static(Mfx, "G") + extra
        Mr2 = M_raw.clone().requires_grad_(True)
        opt = torch.optim.Adam([Mr2], lr=1e-3)
        for it in range(500):
            opt.zero_grad()
            e_tot(Mr2).backward()
            opt.step()
        opt2 = torch.optim.LBFGS([Mr2], max_iter=200, history_size=25,
                                 tolerance_grad=1e-9,
                                 tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_tot(Mr2)
            E.backward()
            return E
        opt2.step(closure)
        rows.append({"omega": om, "E_total": float(e_tot(Mr2))})
        del Mr2
        torch.cuda.empty_cache()
    kmin = min(range(3), key=lambda i: rows[i]["E_total"])
    rec["ladder"] = rows
    rec["interior"] = bool(kmin == 1)
    rec["depth"] = rows[0]["E_total"] - rows[1]["E_total"]
    rec["minutes"] = (time.time() - t0) / 60
    np.savez_compressed(
        os.path.join(HERE, "results", f"M_d{delta:.6f}_g{g}.npz"),
        M=Mf.cpu().numpy())
    print(f"[d={delta:.5f} g={g}] E {rec['E_stat']:.4f} "
          f"(|g| {ginf:.1e}); tG {rec['time_part_G']:+.3e} "
          f"tE {rec['time_part_eta']:+.3e}; om_pred {om_p:.3f}; "
          f"interior {rec['interior']} depth {rec['depth']:.2e}; "
          f"I_p {rec['I_pure']:.3e} I_c {rec['I_comb']:.3e}; "
          f"f32rel {rec['float32_rel']:.1e}; {rec['minutes']:.0f} min",
          flush=True)
    del L, M_raw, Mf
    torch.cuda.empty_cache()
    return rec


results = {"prereg": "prereg_012.md", "points": []}
for g in (8, 64, 512):
    for delta in (0.125, 1.0 / 64, 1.0 / 512):
        try:
            results["points"].append(point(delta, g))
        except Exception as e:
            results["points"].append({"delta": delta, "g": g,
                                      "FAILED": repr(e)})
            print(f"[d={delta} g={g}] FAILED: {e!r}", flush=True)
        json.dump(results, open(os.path.join(HERE, "results",
                                             "grid.json"),
                                "w"), indent=1)
with open(FLAG, "w") as f:
    f.write("grid computed in this run\n")
print("grid complete: results/grid.json")
