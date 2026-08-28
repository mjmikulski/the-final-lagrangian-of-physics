"""Does the simplest quartic term, (F_abcd F^abcd)^2 = (I1)^2, make the
localized clock work? (The model author's candidate for the article-1
Lagrangian, asked directly in correspondence of 2026-08-26.)

Two inequivalent realizations of I1 on a clock configuration
(dM/dt = omega * a0, frozen tangent), review round 1 having established
that the distinction is decisive:

- RAW ETA FORM (report 001's I1, all four slots plus the matrix
  contractions with eta): the time pairs carry the outer eta^00 = -1,
  but the matrix-slot eta contraction of F_{0i} is itself negative on
  the field, so the NET time part is POSITIVE (measured: +0.020832 vs
  the +0.020833 magnitude of the channel density at omega = 0.35).
  I1_eta = i1s_eta + k_eta with k_eta > 0: the square has no negative
  cross term and cannot tick.
- G FORM (the same contraction with the working Euclideanizer G of
  reports 002/004 on the matrix slots, outer indices by eta): the
  matrix-slot G contraction of F_{0i} is positive, the outer eta^00
  makes the net time part NEGATIVE. I1_G = i1s_G - k_G: the square
  gamma*(i1s_G - k_G)^2 contains the clock drive -2*gamma*i1s*k_G and
  the quartic brake gamma*k_G^2. The same eta -> G repair that fixed
  the statics in 004 is what lets the simplest quartic tick.

Anti-delocalization is a convexity fact: the completed square is convex
in k with pointwise minimum at the static template k = i1s, so
spreading the ticking density away from the template only costs (the
Mexican-hat local quartic of 004/007 is instead concave in its linear
drive -- dilution pays there).

TWO READINGS of the G-form term (review round 1, point 2 -- the
report-002 distinction for velocity-quartic terms):
- ENERGY-FUNCTIONAL ansatz: E_extra = gamma*int (i1s - k)^2; reduced
  frozen-profile energy -2 g C1 w^2 + g C2 w^4, minimum at
  omega_E = sqrt(C1/C2);
- FUNDAMENTAL-LAGRANGIAN reading: L_extra = gamma*(I1_G)^2 with k
  quadratic in the velocity; the Legendre transform maps
  L = -2 g s k + g k^2 (in the k-dependent part) to
  H = -2 g s k + 3 g k^2, so the frozen minimum sits at
  omega_H = sqrt(C1/(3 C2)).
Both predictions are gamma-independent; both ladders are run.

CONVERGENCE (review round 1, point 3): every rung runs the SAME
fixed-depth protocol -- 500 Adam steps + two L-BFGS cycles (strong
Wolfe, 200 iterations each) -- with the energy recorded after each
level. Absolute stationarity is not reachable here (measured: further
L-BFGS cycles keep creeping ~1.5e-6 each with |g|_inf stuck at a few
1e-3 -- a long flat valley of the 327k-dof landscape), so the asserted
criterion is BRACKET STABILITY: the location of the minimum must be
identical at every protocol level (the common creep cancels in energy
differences across rungs). Final |g|_inf is recorded per rung. The
omega = 0 endpoint is relaxed identically (a first run relaxed every
rung but omega = 0, faking an interior minimum in both signs; the sign
control caught it).

Ladders:
  JG_E  : G form, energy reading (the well at omega_E)     [persisted]
  JG_H  : G form, fundamental-Lagrangian reading (omega_H) [persisted]
  J_ETA : faithful raw-eta form, energy reading (expects min at 0)
  J0    : G form with the cross sign flipped (control, min at 0)
  J2    : intensive (int I1_G)^2 / V (diagnosis: zeroes its integral)
gamma fixes the statics deformation at 5% of E_stat.

Needs report 004's polished field (or M5_FIELDS_DIR); NOT-REPRODUCED
notice otherwise. Out: results/i1sq_ladders.json
"""
import json
import os
import runpy
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
MP = os.path.join(FIELDS, "M_G_polished.npz")
FLAG = os.path.join(HERE, "results", "i1sq_ran.flag")
if not os.path.exists(MP):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("ladder_i1sq: NOT REPRODUCED HERE -- needs report 004's "
          f"polished field in {FIELDS} (or M5_FIELDS_DIR). Committed "
          "results carry the recorded values.")
    sys.exit(0)

L = runpy.run_path(os.path.join(R004, "lattice.py"), run_name="not_main")
field, e_static = L["field"], L["e_static"]
a0_of, gen_catalog = L["a0_of"], L["gen_catalog"]
d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
H, DT, DEV, ETA = L["H"], L["DT"], L["DEV"], L["ETA"]

M_pol = torch.tensor(np.load(MP)["M"], dtype=DT, device=DEV)


def inner_G(F, G):
    return torch.einsum("...ab,...ac,...bd,...cd->...", F, G, G, F)


def inner_eta(F):
    return torch.einsum("...ab,ac,bd,...cd->...", F, ETA, ETA, F)


def densities(M, a0, om, metric):
    """(static I1 density, time density k >= 0 as it enters the form).

    metric='G':  i1s with G on the matrix slots; k_G = sum_i <F0i>_GG
                 (enters I1_G with the outer eta^00 minus).
    metric='eta': all-eta contractions; k_eta = sum_i -<F0i>_etaeta
                 (measured positive; enters I1_eta with a PLUS).
    """
    G = G_of(M) if metric == "G" else None
    V = om * a0
    i1s = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F0 = comm(V, A[i])
            if metric == "G":
                k = k + 0.5 * 4.0 * inner_G(F0, G)
            else:
                k = k + 0.5 * 4.0 * (-1.0) * inner_eta(F0)
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                if metric == "G":
                    i1s = i1s + 0.5 * 4.0 * inner_G(F, G)
                else:
                    i1s = i1s + 0.5 * 4.0 * inner_eta(F)
    return i1s, k


Mg = field(M_pol)
a0 = a0_of(gen_catalog()["boost_x"], Mg)
i1s0, k1 = densities(Mg, a0, 1.0, "G")
Es0 = e_static(Mg, "G").item()
gamma = 0.05 * Es0 / (H ** 3 * (i1s0 ** 2).sum()).item()
C1 = (H ** 3 * (i1s0 * k1).sum()).item()
C2 = (H ** 3 * (k1 ** 2).sum()).item()
om_E = (C1 / C2) ** 0.5
om_H = (C1 / (3.0 * C2)) ** 0.5
print(f"gamma = {gamma:.5f} (5% statics deformation)")
print(f"frozen-profile predictions: omega_E = {om_E:.3f} (energy "
      f"reading), omega_H = {om_H:.3f} (fundamental-L reading)")

results = {"gamma": gamma, "C1": C1, "C2": C2,
           "omega_pred_E": om_E, "omega_pred_H": om_H, "E_stat0": Es0}


def relax(e_total_fn):
    """Fixed-depth protocol, identical for every rung: 500 Adam steps,
    then two L-BFGS cycles (strong Wolfe, 200 iterations each). The
    landscape has a long flat valley: absolute stationarity is NOT
    reached (measured: the energy keeps creeping ~1.5e-6 per further
    L-BFGS cycle with |g|_inf stuck at a few 1e-3), so instead of a
    residual threshold the report asserts BRACKET STABILITY: the energy
    is recorded after each protocol level and the well's shape must be
    the same at every level (the common creep mode cancels in the
    differences). Returns (field, E_levels, ginf)."""
    M_raw = M_pol.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        e_total_fn(M_raw).backward()
        opt.step()
    E_levels = [float(e_total_fn(M_raw).detach())]
    for cycle in range(2):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200,
                                 history_size=25, tolerance_grad=1e-9,
                                 tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_total_fn(M_raw)
            E.backward()
            return E
        opt2.step(closure)
        E_levels.append(float(e_total_fn(M_raw).detach()))
    g = torch.autograd.grad(e_total_fn(M_raw), M_raw)[0]
    return M_raw.detach(), E_levels, float(g.abs().max())


def run_ladder(tag, e_extra, omegas, save_prefix=None, save=()):
    rungs = []
    for om in omegas:
        M_raw, E_levels, ginf = relax(
            lambda Mr, om=om: e_static(field(Mr), "G")
            + e_extra(field(Mr), om))
        Mf = field(M_raw)
        Es = e_static(Mf, "G").item()
        Ex = e_extra(Mf, om).item()
        _, kdens = densities(Mf, a0, max(om, 1e-9), "G")
        pr = ((kdens.sum() ** 2)
              / (kdens ** 2).sum().clamp_min(1e-30)).item()
        rungs.append({"omega": om, "E_total": Es + Ex, "E_stat": Es,
                      "E_extra": Ex, "PR_k_sites": pr,
                      "grad_inf": ginf, "E_levels": E_levels})
        print(f"  [{tag}] omega {om}: E {Es+Ex:.6f} (extra {Ex:+.4f}), "
              f"PR {pr:.0f}, |g|inf {ginf:.1e}, "
              f"levels {['%.6f' % e for e in E_levels]}", flush=True)
        if save_prefix and om in save:
            tagom = str(om).replace(".", "")
            np.savez_compressed(
                os.path.join(HERE, "results",
                             f"{save_prefix}{tagom}.npz"),
                M=Mf.cpu().numpy())
    k = min(range(len(rungs)), key=lambda i: rungs[i]["E_total"])
    # bracket stability across protocol depths: the minimum's rung
    # index must be the same at every relaxation level
    min_idx_per_level = [
        min(range(len(rungs)), key=lambda i: rungs[i]["E_levels"][lv])
        for lv in range(len(rungs[0]["E_levels"]))]
    v = {"rungs": rungs, "min_omega": rungs[k]["omega"],
         "interior": bool(0 < k < len(rungs) - 1),
         "max_grad_inf": max(r["grad_inf"] for r in rungs),
         "min_omega_per_level": [rungs[i]["omega"]
                                 for i in min_idx_per_level]}
    print(f"  [{tag}] verdict: min at omega {v['min_omega']}, interior "
          f"{v['interior']}, worst |g|inf {v['max_grad_inf']:.1e}")
    return v


def eG_energy(Mf, om):
    i1s, k = densities(Mf, a0, om, "G")
    return gamma * H ** 3 * ((i1s - k) ** 2).sum()


def eG_fundamental(Mf, om):
    # Legendre image of L = gamma*(i1s - k)^2 with k ~ omega^2:
    # H = gamma*i1s^2 - 2 gamma i1s k + 3 gamma k^2
    i1s, k = densities(Mf, a0, om, "G")
    return gamma * H ** 3 * (i1s ** 2 - 2.0 * i1s * k
                             + 3.0 * k ** 2).sum()


def eG_flipped(Mf, om):
    i1s, k = densities(Mf, a0, om, "G")
    return gamma * H ** 3 * ((i1s + k) ** 2).sum()


def e_eta(Mf, om):
    i1s, k = densities(Mf, a0, om, "eta")
    return gamma * H ** 3 * ((i1s + k) ** 2).sum()


def e_intensive(Mf, om):
    i1s, k = densities(Mf, a0, om, "G")
    T = H ** 3 * (i1s - k).sum()
    return gamma * T ** 2 / (H ** 3 * i1s0.numel())


np.savez_compressed(os.path.join(HERE, "results", "a0_frozen.npz"),
                    a0=a0.cpu().numpy())

print("JG_E: G form, energy reading:")
results["JG_E"] = run_ladder(
    "JG_E", eG_energy, (0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2),
    save_prefix="jge_rung_om", save=(0.2, 0.35, 0.5))
print("JG_H: G form, fundamental-Lagrangian reading:")
results["JG_H"] = run_ladder(
    "JG_H", eG_fundamental, (0.0, 0.07, 0.13, 0.19, 0.26, 0.35, 0.5),
    save_prefix="jgh_rung_om", save=(0.13, 0.19, 0.26))
print("J_ETA: faithful raw-eta form, energy reading:")
results["J_ETA"] = run_ladder(
    "J_ETA", e_eta, (0.0, 0.1, 0.2, 0.35, 0.5, 0.8))
print("J0: G form, flipped cross sign (control):")
results["J0"] = run_ladder("J0", eG_flipped, (0.0, 0.2, 0.35, 0.8))
print("J2: intensive (int I1_G)^2 / V (diagnosis):")
results["J2_intensive"] = run_ladder(
    "J2", e_intensive, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

with open(os.path.join(HERE, "results", "i1sq_ladders.json"), "w") as f:
    json.dump(results, f, indent=1)
with open(FLAG, "w") as f:
    f.write("ladders computed in this run\n")
print("written: results/i1sq_ladders.json")
