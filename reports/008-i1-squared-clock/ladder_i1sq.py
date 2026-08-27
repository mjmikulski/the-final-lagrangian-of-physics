"""The simplest quartic term: does (F_abcd F^abcd)^2 = (I1)^2 make the
localized clock work? (The model author's candidate for the article-1
Lagrangian, asked directly in correspondence of 2026-08-26.)

Structure. On a clock configuration the full I1 density splits into a
static (spatial-derivative) part and a kinetic (time-derivative) part;
in the covariant eta convention the kinetic part enters with a minus
sign (eta^00 = -1), so

    I1_full(x) = i1_stat(x) - i1_kin(x, omega),
    gamma * I1_full^2 = gamma*i1_stat^2          (statics deformation)
                      - 2*gamma*i1_stat*i1_kin   (clock DRIVE)
                      + gamma*i1_kin^2           (local quartic brake).

The cross term supplies the negative kinetic coefficient (the drive)
with no hand-added structure. Anti-delocalization is a convexity fact:
the completed square is convex in i1_kin with pointwise minimum at the
static template i1_kin = i1_stat, so spreading the ticking density away
from the template only costs (the Mexican-hat local quartic of 004/007
is instead concave in its linear drive -- dilution pays there).
Frozen-profile prediction: with i1_kin = omega^2 * bk1 the reduced
energy is E(omega) ~ -2 g C1 omega^2 + g C2 omega^4 with
C1 = sum(i1s*bk1), C2 = sum(bk1^2), so the minimum sits at
omega_* = sqrt(C1/C2), INDEPENDENT of gamma, and the well depth
g*C1^2/C2 scales linearly with gamma (tested separately in
confirm_gamma_scaling.py).

Ladders (report-004 protocol, fresh-start from the polished field,
EVERY rung relaxed including omega = 0 -- the statics deformation must
relax at omega = 0 too, or the endpoint is artificially high and fakes
an interior minimum; that failure mode was caught by the sign control
in a first run and is exactly what J0 guards against):
  J1: local (I1)^2, covariant sign  -- the faithful simplest term;
  J0: local, flipped (positive) cross term -- control, expects min at 0;
  J2: intensive (integral I1_full)^2 / V -- diagnosis of the intensive
      variant (expected: minimum by trivially zeroing the integral).
gamma is set so the statics deformation is a mild perturbation
(gamma * sum i1s^2 = 5% of E_stat).

Persisted for the independent route: J1 rung fields at
omega = 0.2, 0.35, 0.5 (results/j1_rung_om*.npz).

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
boost_channels, a0_of, gen_catalog = (L["boost_channels"], L["a0_of"],
                                      L["gen_catalog"])
d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
H, DT, DEV = L["H"], L["DT"], L["DEV"]

M_pol = torch.tensor(np.load(MP)["M"], dtype=DT, device=DEV)


def i1_static_density(M):
    """Static I1 density per cell: the spatial-pair (e_u) part of
    e_static in the working G metric, both one-sided stencils averaged
    -- identical contractions to lattice.e_static."""
    G = G_of(M)
    e = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                e = e + 0.5 * 4.0 * torch.einsum(
                    "...ab,...ac,...bd,...cd->...", F, G, G, F)
    return e


Mg = field(M_pol)
i1s0 = i1_static_density(Mg)
Es0 = e_static(Mg, "G").item()
gamma = 0.05 * Es0 / (H ** 3 * (i1s0 ** 2).sum()).item()
a0 = a0_of(gen_catalog()["boost_x"], Mg)
bk1, _ = boost_channels(Mg, a0, 1.0)
C1 = (H ** 3 * (i1s0 * bk1).sum()).item()
C2 = (H ** 3 * (bk1 ** 2).sum()).item()
om_pred = (C1 / C2) ** 0.5
print(f"gamma = {gamma:.5f} (5% statics deformation), frozen-profile "
      f"prediction omega_* = sqrt(C1/C2) = {om_pred:.3f}")

results = {"gamma": gamma, "C1": C1, "C2": C2,
           "omega_pred_frozen": om_pred, "E_stat0": Es0}
OM_FINE = (0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2)


def run_ladder(tag, e_extra, omegas, save=()):
    rungs = []
    for om in omegas:
        M_raw = M_pol.clone().requires_grad_(True)
        opt = torch.optim.Adam([M_raw], lr=1e-3)
        for it in range(500):
            opt.zero_grad()
            Mf = field(M_raw)
            (e_static(Mf, "G") + e_extra(Mf, om)).backward()
            opt.step()
        M_raw = M_raw.detach()
        Mf = field(M_raw)
        Es = e_static(Mf, "G").item()
        Ex = e_extra(Mf, om).item()
        bk, _ = boost_channels(Mf, a0, max(om, 1e-9))
        pr = ((bk.sum() ** 2) / (bk ** 2).sum().clamp_min(1e-30)).item()
        rungs.append({"omega": om, "E_total": Es + Ex, "E_stat": Es,
                      "E_extra": Ex, "PR_bk_sites": pr})
        print(f"  [{tag}] omega {om}: E {Es+Ex:.5f} (stat {Es:.4f}, "
              f"extra {Ex:+.4f}), PR {pr:.0f}", flush=True)
        if om in save:
            tagom = str(om).replace(".", "")
            np.savez_compressed(
                os.path.join(HERE, "results", f"j1_rung_om{tagom}.npz"),
                M=Mf.cpu().numpy())
    k = min(range(len(rungs)), key=lambda i: rungs[i]["E_total"])
    v = {"rungs": rungs, "min_omega": rungs[k]["omega"],
         "interior": bool(0 < k < len(rungs) - 1)}
    print(f"  [{tag}] verdict: min at omega {v['min_omega']}, "
          f"interior {v['interior']}")
    return v


def e_local(sign):
    def e_extra(Mf, om):
        i1s = i1_static_density(Mf)
        bk, _ = boost_channels(Mf, a0, om)
        return gamma * H ** 3 * ((i1s + sign * bk) ** 2).sum()
    return e_extra


def e_intensive(Mf, om):
    i1s = i1_static_density(Mf)
    bk, _ = boost_channels(Mf, a0, om)
    T = H ** 3 * (i1s - bk).sum()
    return gamma * T ** 2 / (H ** 3 * i1s0.numel())


# also persist the frozen tangent and the static reference density for
# the independent route
np.savez_compressed(os.path.join(HERE, "results", "a0_frozen.npz"),
                    a0=a0.cpu().numpy())

print("J1: local (I1)^2, covariant sign (the faithful simplest term):")
results["J1_local_covariant"] = run_ladder("J1", e_local(-1.0), OM_FINE,
                                           save=(0.2, 0.35, 0.5))
print("J0: local, flipped cross sign (control):")
results["J0_local_control"] = run_ladder("J0", e_local(+1.0), OM_FINE)
print("J2: intensive (int I1_full)^2 / V (diagnosis):")
results["J2_intensive"] = run_ladder(
    "J2", e_intensive, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

with open(os.path.join(HERE, "results", "i1sq_ladders.json"), "w") as f:
    json.dump(results, f, indent=1)
with open(FLAG, "w") as f:
    f.write("ladders computed in this run\n")
print("written: results/i1sq_ladders.json")
