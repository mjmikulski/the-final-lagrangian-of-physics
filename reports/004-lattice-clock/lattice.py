"""Lattice port of the G-metric statics and the C3 condensate clock.

Torch/autograd version of the m5_21_3 stack conventions (sym stencils,
pinned shell, embedded 2b electron seed): no hand-derived gradients, so
the field-dependent metric G(M) costs nothing extra. CLI stages write
partial results so long runs are resumable:

  gate    vacuum-zero + baseline eta relax (must match the m5_21_3
          reproduction character: E ~ 6.3, offblock = 0)
  statG   relax the all-G statics, Coulomb-tail fit, spectral gap
  kin     kin table on the relaxed G profile (all channels > 0?)
  ladder  C3 condensate omega-ladder with per-rung re-relaxation
  hessq1  lowest Hessian eigenvalue of the G statics (P240 Q1)

Out: lattice_results.json (merged across stages), M_eta.npz, M_G.npz.
"""
import json
import os
import sys
import time

import torch

torch.manual_seed(1)
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DT = torch.float64
N, L = 32, 48.0
H = L / N
SG, DELTA, W1 = 8.0, 0.3, 0.000724023879
C_P = tuple(SG ** p + 1.0 + DELTA ** p for p in range(1, 5))
ETA = torch.diag(torch.tensor([-1.0, 1, 1, 1], dtype=DT, device=DEV))
M_VAC = torch.diag(torch.tensor([-SG, 1.0, DELTA, 0.0], dtype=DT,
                                device=DEV))
SEED3 = os.path.join(HERE, "results", "seed_3x3_electron.npz")
RESULTS = os.path.join(HERE, "results", "lattice_results.json")


def d1(f, ax, st):
    """One-sided difference along grid axis ax (m5_21_2b convention)."""
    out = torch.zeros_like(f)
    idx = [slice(None)] * f.ndim

    def at(i):
        s = list(idx); s[ax] = i; return tuple(s)
    if st == "fwd":
        out[at(slice(0, -1))] = (f[at(slice(1, None))]
                                 - f[at(slice(0, -1))]) / H
    else:
        out[at(slice(1, None))] = (f[at(slice(1, None))]
                                   - f[at(slice(0, -1))]) / H
    return out


def coords():
    x = (torch.arange(N, dtype=DT, device=DEV) - (N - 1) / 2.0) * H
    return torch.meshgrid(x, x, x, indexing="ij")


def pin_shell(depth=1.6):
    wc = max(1, int(-(-depth // H)))
    P = torch.zeros(N, N, N, dtype=torch.bool, device=DEV)
    for ax in range(3):
        s = [slice(None)] * 3
        s[ax] = slice(0, wc); P[tuple(s)] = True
        s[ax] = slice(N - wc, N); P[tuple(s)] = True
    return P


SHELL = pin_shell()
FREE = ~SHELL


def sym4(X):
    return 0.5 * (X + X.transpose(-1, -2))


def G_of(M):
    """Lagrange-projector Euclideanizer per cell (report 002)."""
    x = torch.einsum("ab,...bc->...ac", ETA, M)
    I4 = torch.eye(4, dtype=DT, device=DEV).expand_as(M)
    q = (x @ (x - I4) @ (x - DELTA * I4)) / (SG * (SG - 1) * (SG - DELTA))
    return ETA - 2.0 * q @ ETA


def inner_X(F, X):
    """<F,F>_X per cell; X constant (4,4) or per-cell (...,4,4)."""
    if X.dim() == 2:
        return torch.einsum("...ab,ac,bd,...cd->...", F, X, X, F)
    return torch.einsum("...ab,...ac,...bd,...cd->...", F, X, X, F)


def comm(A, B):
    return A @ ETA @ B - B @ ETA @ A


SHELL_VALS = None    # shell frozen at SEED values (m5 pinned-shell BC:
                     # the hedgehog boundary keeps its radial texture)


def field(M_raw):
    """Symmetric field with the shell frozen at the embedded-seed values."""
    global SHELL_VALS
    if SHELL_VALS is None:
        SHELL_VALS = seed_embedded()
    Msym = sym4(M_raw)
    mask = SHELL[..., None, None].to(DT)
    return mask * SHELL_VALS + (1 - mask) * Msym


def e_static(M, metric):
    """u (sym-stencil averaged) + V4; metric 'eta' or 'G'."""
    X = ETA if metric == "eta" else G_of(M)
    e_u = 0.0
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                e_u = e_u + 0.5 * 4.0 * inner_X(F, X).sum()
    Me = M @ ETA
    P, v4 = Me, 0.0
    for p in range(4):
        if p:
            P = P @ Me
        t = torch.einsum("...kk->...", P)
        v4 = v4 + (t - C_P[p]) ** 2
    return H ** 3 * (e_u + W1 * v4.sum())


def relax(M_raw, metric, steps, lr=5e-4, tag="", log_every=500):
    M_raw = M_raw.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=lr)
    t0 = time.time()
    traj = []
    for it in range(steps):
        opt.zero_grad()
        E = e_static(field(M_raw), metric)
        E.backward()
        opt.step()
        if (it + 1) % log_every == 0:
            traj.append(E.item())
            print(f"  {tag} it {it+1:5d}  E {E.item():.6f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return M_raw.detach(), traj


def load_results():
    return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}


def save_results(r):
    with open(RESULTS, "w") as f:
        json.dump(r, f, indent=1)


def seed_embedded():
    import numpy as np
    M3 = torch.tensor(np.load(SEED3)["M"], dtype=DT, device=DEV)
    M4 = torch.zeros(N, N, N, 4, 4, dtype=DT, device=DEV)
    M4[..., 1:4, 1:4] = M3
    M4[..., 0, 0] = -SG
    return M4


def offblock(M):
    return M[..., 0, 1:4].abs().max().item()


def gen_catalog():
    gens = {}
    for name, (i, j), boost in (("rot_xy", (1, 2), 0), ("rot_xz", (1, 3), 0),
                                ("rot_yz", (2, 3), 0), ("boost_x", (0, 1), 1),
                                ("boost_y", (0, 2), 1), ("boost_z", (0, 3), 1)):
        W = torch.zeros(4, 4, dtype=DT, device=DEV)
        if boost:
            W[i, j] = W[j, i] = 1.0
        else:
            W[i, j], W[j, i] = -1.0, 1.0
        gens[name] = W
    return gens


def envelope():
    X, Y, Z = coords()
    r = torch.sqrt(X * X + Y * Y + Z * Z)
    return torch.exp(-((r / 10.0) ** 4))


def a0_of(W, M):
    """Envelope-localized conjugation tangent, unit Frobenius norm."""
    a = envelope()[..., None, None] * (torch.einsum("ab,...bc->...ac", W, M)
                                       + torch.einsum("...ab,cb->...ac", M, W))
    return a / a.norm()


def boost_channels(M, a0, omega):
    """(B_k density per cell, rot-kinetic density) for dM/dt = omega a0."""
    G = G_of(M)
    V = omega * a0
    bk = torch.zeros(N, N, N, dtype=DT, device=DEV)
    rk = torch.zeros(N, N, N, dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F = comm(V, A[i])
            lG = inner_X(F, G)
            le = inner_X(F, ETA)
            bk = bk + 0.5 * 4.0 * (lG - le) / 2
            rk = rk + 0.5 * 4.0 * (lG + le) / 2
    return bk, rk


def e_condensate(M, a0, omega, a_c, b_c):
    bk, _ = boost_channels(M, a0, omega)
    return H ** 3 * (-a_c * bk + 3 * b_c * bk ** 2).sum()


# ==================== stages ====================
def stage_gate(res):
    Mv = M_VAC.expand(N, N, N, 4, 4)
    res["vacuum_energy"] = {m: e_static(Mv, m).item() for m in ("eta", "G")}
    print("vacuum energies:", res["vacuum_energy"])
    M0 = seed_embedded()
    oracle = e_static(field(M0), "eta").item()
    print(f"seed E_eta = {oracle:.4f}")
    # oracle: the FIRE reference value of the same embedded seed
    ORACLE_REF = 9.263660060
    res["gate"] = {"oracle_seed_E": oracle, "oracle_ref": ORACLE_REF,
                   "oracle_rel": abs(oracle - ORACLE_REF) / ORACLE_REF}
    Mr, traj = relax(M0, "eta", 3000, tag="eta")
    res["gate"]["baseline_trajectory"] = [oracle] + traj
    Me = field(Mr)
    res["baseline_eta"] = {"E": e_static(Me, "eta").item(),
                           "offblock": offblock(Me)}
    print("baseline:", res["baseline_eta"])
    import numpy as np
    np.savez_compressed(os.path.join(HERE, os.path.join("results", "M_eta.npz")),
                        M=Mr.cpu().numpy())
    return res


def tail_fit(M, metric):
    """log-log slope + coefficient of the u density on shells r in [8,16]."""
    X = ETA if metric == "eta" else G_of(M)
    dens = torch.zeros(N, N, N, dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                dens = dens + 0.5 * 4.0 * inner_X(F, X)
    Xc, Yc, Zc = coords()
    r = torch.sqrt(Xc ** 2 + Yc ** 2 + Zc ** 2)
    rs, us = [], []
    for k in range(10):
        lo, hi = 8.0 + 0.8 * k, 8.0 + 0.8 * (k + 1)
        m = (r >= lo) & (r < hi)
        if m.sum() > 0 and dens[m].mean() > 0:
            rs.append((lo + hi) / 2)
            us.append(dens[m].mean().item())
    import numpy as np
    lr_, lu = np.log(rs), np.log(us)
    slope, logc = np.polyfit(lr_, lu, 1)
    return {"slope": float(slope), "coeff": float(np.exp(logc)),
            "shells": list(zip(rs, us))}


def stage_statG(res):
    import numpy as np
    M0 = torch.tensor(np.load(os.path.join(HERE, os.path.join("results", "M_eta.npz")))["M"],
                      dtype=DT, device=DEV)
    Mr, _ = relax(M0, "G", 3000, tag="G  ")
    Mg = field(Mr)
    lam = torch.linalg.eigvals(torch.einsum(
        "ab,...bc->...ac", ETA, Mg)).real
    top2 = lam.sort(dim=-1, descending=True).values[..., :2]
    gap = (top2[..., 0] - top2[..., 1])[FREE].min().item()
    res["statG"] = {"E": e_static(Mg, "G").item(),
                    "E_eta_of_same": e_static(Mg, "eta").item(),
                    "offblock": offblock(Mg),
                    "spectral_gap_min": gap,
                    "tail_eta_profile": tail_fit(
                        field(torch.tensor(np.load(os.path.join(
                            HERE, os.path.join("results", "M_eta.npz")))["M"], dtype=DT, device=DEV)),
                        "eta"),
                    "tail_G_profile": tail_fit(Mg, "G")}
    print("statG:", {k: v for k, v in res["statG"].items()
                     if not isinstance(v, dict)})
    print("  tail eta:", res["statG"]["tail_eta_profile"]["slope"],
          " tail G:", res["statG"]["tail_G_profile"]["slope"])
    np.savez_compressed(os.path.join(HERE, os.path.join("results", "M_G.npz")), M=Mr.cpu().numpy())
    return res


def stage_kin(res):
    import numpy as np
    Mg = field(torch.tensor(np.load(os.path.join(HERE, os.path.join("results", "M_G.npz")))["M"],
                            dtype=DT, device=DEV))
    G = G_of(Mg)
    table = {}
    for name, W in gen_catalog().items():
        a0 = a0_of(W, Mg)
        kin = 0.0
        for st in ("fwd", "bwd"):
            A = [d1(Mg, ax, st) for ax in range(3)]
            for i in range(3):
                F = comm(a0, A[i])
                kin = kin + 0.5 * 4.0 * inner_X(F, G).sum()
        table[name] = (H ** 3 * kin).item()
    res["kin_table_G"] = table
    print("kin (G, relaxed profile):",
          {k: round(v, 4) for k, v in table.items()})
    return res


def stage_ladder(res):
    import numpy as np
    Mr0 = torch.tensor(np.load(os.path.join(HERE, os.path.join("results", "M_G.npz")))["M"],
                       dtype=DT, device=DEV)
    Mg = field(Mr0)
    # pick the boost generator with the largest K1 on the profile
    best, K1, K2 = None, 0.0, 0.0
    for name in ("boost_x", "boost_y", "boost_z"):
        a0 = a0_of(gen_catalog()[name], Mg)
        bk, _ = boost_channels(Mg, a0, 1.0)
        k1 = (H ** 3 * bk.sum()).item()
        if k1 > K1:
            best, K1 = name, k1
            K2 = (H ** 3 * (bk ** 2).sum()).item()
    a_c, om_t = 1.0, 0.8
    b_c = a_c * K1 / (6 * K2 * om_t ** 2)
    a0 = a0_of(gen_catalog()[best], Mg)          # frozen (m5 protocol)
    res["ladder_setup"] = {"generator": best, "K1": K1, "K2": K2,
                           "a": a_c, "b": b_c, "omega_target": om_t}
    print(f"ladder: gen {best}, K1 {K1:.4f}, K2 {K2:.6f}, b {b_c:.4f}")
    rungs, M_raw = [], Mr0.clone()
    E_stat0 = e_static(Mg, "G").item()
    for om in (0.0, 0.2, 0.5, 0.8, 1.1, 1.4):
        if om > 0:
            M_raw = M_raw.clone().requires_grad_(True)
            opt = torch.optim.Adam([M_raw], lr=1e-3)
            for it in range(500):
                opt.zero_grad()
                Mf = field(M_raw)
                E = e_static(Mf, "G") + e_condensate(Mf, a0, om, a_c, b_c)
                E.backward()
                opt.step()
            M_raw = M_raw.detach()
        Mf = field(M_raw)
        Es = e_static(Mf, "G").item()
        Ec = e_condensate(Mf, a0, om, a_c, b_c).item() if om > 0 else 0.0
        bk, rk = boost_channels(Mf, a0, max(om, 1e-9))
        rungs.append({"omega": om, "E_total": Es + Ec, "E_stat": Es,
                      "E_cond": Ec,
                      "rot_kin_coeff": (H ** 3 * rk.sum()).item()
                      / max(om, 1e-9) ** 2})
        print(f"  omega {om}: E_total {Es+Ec:.4f} (stat {Es:.4f}, "
              f"cond {Ec:.4f})", flush=True)
    res["ladder"] = {"E_stat_start": E_stat0, "rungs": rungs}
    Ets = [r["E_total"] for r in rungs]
    k = Ets.index(min(Ets))
    res["ladder"]["omega_min"] = rungs[k]["omega"]
    res["ladder"]["interior_minimum"] = bool(0 < k < len(rungs) - 1)
    print("ladder verdict: min at omega =", rungs[k]["omega"],
          "interior:", res["ladder"]["interior_minimum"])
    return res


def stage_hessq1(res):
    import numpy as np
    Mr = torch.tensor(np.load(os.path.join(HERE, os.path.join("results", "M_G.npz")))["M"],
                      dtype=DT, device=DEV)
    Mr = Mr.clone().requires_grad_(True)

    def grad_of(m):
        return torch.autograd.grad(e_static(field(m), "G"), m,
                                   create_graph=True)[0]

    g = grad_of(Mr)
    mask = FREE[..., None, None].to(DT)

    def hvp(v):
        (Hv,) = torch.autograd.grad(g, Mr, grad_outputs=v,
                                    retain_graph=True)
        return sym4(Hv) * mask

    def power(op, iters=60):
        v = sym4(torch.randn_like(Mr)) * mask
        v = v / v.norm()
        lam = 0.0
        for _ in range(iters):
            w = op(v)
            lam = (v * w).sum().item()
            v = w / w.norm()
        return lam

    lam_max = power(hvp)
    lam_min = lam_max - power(lambda v: lam_max * v - hvp(v))
    res["hessian_q1"] = {"lam_max": lam_max, "lam_min_est": lam_min}
    print(f"Q1: lam_max {lam_max:.3f}, lam_min {lam_min:.4f} "
          f"({'SADDLE' if lam_min < -1e-3 else 'PSD within tolerance'})")
    return res


if __name__ == "__main__":
    stages = sys.argv[1:] or ["gate", "statG", "kin", "ladder", "hessq1"]
    res = load_results()
    for s in stages:
        print(f"==== stage {s} [{DEV}] ====", flush=True)
        res = {"gate": stage_gate, "statG": stage_statG, "kin": stage_kin,
               "ladder": stage_ladder, "hessq1": stage_hessq1}[s](res)
        save_results(res)
    print("written:", RESULTS)
