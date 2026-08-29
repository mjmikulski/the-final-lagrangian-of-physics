"""Fixed-J (relative-equilibrium) reading of the rotational sector --
dev run for report 010.

Collective coordinate: the PURE rotation generator acting on the
interior (the frozen shell does not rotate),
    dM/dt = thetadot * g(M),  g(M) = (1 - shell_mask) * (W M - M W),
with NO envelope and NO normalization -- the two conventions that made
the report-009 channel proxy ill-defined. The channel kinetic energy is
T = I[M]/2 * thetadot^2 with I[M] = 2 H^3 sum k1(M), where k1 is the
G-metric kinetic density of g(M) at unit rate. The fixed-J energy is

    E_J[M] = E_stat[M] + J^2 / (2 I[M]),

the Routhian of L = I/2 thetadot^2 - V: a BOUNDED, well-posed
constrained functional (the Lagrangian-reading answer), with
thetadot_* = J / I[M*] as an output.

Scan: J in a grid; for each J minimize E_J over M (Adam 500 + 2 L-BFGS
cycles, E_levels recorded); record E(J), I(J), omega(J) = J/I, the
rotational-energy split J^2/2I, and the PR of the kinetic density.
Rigid-profile prediction: E(J) - E(0) ~ J^2/(2 I_0) and
omega ~ J/I_0 for small J. Out: results_fixedj.json
"""
import json
import os
import runpy

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
R008 = os.path.join(HERE, "..", "008-i1-squared-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "fixedj_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    import sys
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("fixedj_scan: NOT REPRODUCED HERE -- needs report 004's "
          f"polished field in {FIELDS} (or M5_FIELDS_DIR).")
    sys.exit(0)
lad = runpy.run_path(os.path.join(R008, "ladder_i1sq_defs.py"))
field, e_static = lad["field"], lad["e_static"]
H = lad["H"]
M_pol = lad["M_pol"]

L004 = runpy.run_path(os.path.join(R004, "lattice.py"),
                      run_name="not_main")
SHELL = L004["SHELL"]
d1, comm, G_of = L004["d1"], L004["comm"], L004["G_of"]
DT, DEV, ETA = L004["DT"], L004["DEV"], L004["ETA"]

W = L004["gen_catalog"]()["rot_xy"]
interior = (1.0 - SHELL.to(DT))[..., None, None]


def kin_density_unit(M):
    """G-metric kinetic density of the pure interior generator at unit
    rate: g(M) = interior * (W M - M W)."""
    g = interior * (torch.einsum("ab,...bc->...ac", W, M)
                    - torch.einsum("...ab,bc->...ac", M, W))
    G = G_of(M)
    k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F0 = comm(g, A[i])
            k = k + 0.5 * 4.0 * torch.einsum(
                "...ab,...ac,...bd,...cd->...", F0, G, G, F0)
    return k


def inertia(M):
    return 2.0 * H ** 3 * kin_density_unit(M).sum()


Mg = field(M_pol)
I0 = float(inertia(Mg))
print(f"rigid inertia I_0 = {I0:.6e}")

results = {"I_0": I0}
Js = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
rows = []
for J in Js:
    def e_tot(Mr, J=J):
        Mf = field(Mr)
        E = e_static(Mf, "G")
        if J > 0:
            E = E + J ** 2 / (2.0 * inertia(Mf))
        return E
    M_raw = M_pol.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        e_tot(M_raw).backward()
        opt.step()
    E_levels = [float(e_tot(M_raw).detach())]
    for cyc in range(2):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200,
                                 history_size=25, tolerance_grad=1e-9,
                                 tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_tot(M_raw)
            E.backward()
            return E
        opt2.step(closure)
        E_levels.append(float(e_tot(M_raw).detach()))
    Mf = field(M_raw.detach())
    I = float(inertia(Mf))
    kd = kin_density_unit(Mf)
    pr = float((kd.sum() ** 2) / (kd ** 2).sum().clamp_min(1e-30))
    E = E_levels[-1]
    om = J / I if J > 0 else 0.0
    Erot = J ** 2 / (2 * I) if J > 0 else 0.0
    Estat = E - Erot
    rows.append({"J": J, "E_total": E, "E_levels": E_levels,
                 "I": I, "omega": om, "E_rot": Erot,
                 "E_stat_part": Estat, "PR_kin": pr})
    print(f"  J {J}: E {E:.6f} (rot {Erot:.2e}, stat {Estat:.6f}), "
          f"I {I:.4e}, omega {om:.4f}, PR {pr:.0f}", flush=True)

results["rows"] = rows
E0 = rows[0]["E_total"]
print("\nrigid-profile check: dE vs J^2/(2 I_0):")
for r in rows[1:]:
    pred = r["J"] ** 2 / (2 * I0)
    print(f"  J {r['J']}: dE {r['E_total']-E0:.3e} vs pred {pred:.3e} "
          f"(ratio {(r['E_total']-E0)/pred:.3f}), omega {r['omega']:.4f}"
          f" vs J/I0 {r['J']/I0:.4f}")
json.dump(results, open(os.path.join(HERE, "results", "fixedj.json"),
                        "w"), indent=1)
with open(FLAG, "w") as f:
    f.write("fixed-J scan computed in this run\n")
print("written: results/fixedj.json")
