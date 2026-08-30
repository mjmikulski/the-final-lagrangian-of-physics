"""Finding-1 response: the two-branch qualitative record of the
centrifugal response, with a scalar-return continuation and a
deformation order parameter. (Round 2: all eight runs hit the
16-cycle cap without meeting TOL, so this is NOT a matched-accuracy
comparison and no branch is selected -- see the README scoping.)

For J in (0, 2, 4, 6), TWO branches at N = 24:
  EQ-start: minimize E_J from the equivariant seed;
  CB-start: minimize E_J from the core-broken spectral seed.
Protocol per run: Adam 1000 + L-BFGS cycles until the per-cycle
energy change < 1e-3 (max 16), with the final residual recorded.
Branch comparison at each J: the E_J values are recorded but, at
these residuals, do not resolve any branch ordering. Hysteresis: continue the CB-start J = 4 endpoint at
J = 0 (same stopping rule) -- does the inertia SCALAR return? (this
tests one scalar of one branch; configuration-space reversibility
would need a field-distance comparison against a converged
reference, which is not done here)
Order parameter: the radial shell profile of the signed kinetic-density
excess k[zeta](branch) - k[zeta](EQ-start J=0 endpoint), plus its
centroid radius -- showing WHERE the observed peripheral excess lives.
Out: results/centrifugal_branches.json
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
FLAG = os.path.join(HERE, "results", "branches_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("centrifugal_branches: NOT REPRODUCED HERE.")
    sys.exit(0)

d = runpy.run_path(os.path.join(HERE, "defs_011.py"))
L = d["load_stack"](N=24)
field, e_static = L["field"], L["e_static"]
H, DT, DEV, N = L["H"], L["DT"], L["DEV"], L["N"]
M_EQ0 = d["seed"](L, "EQ")
M_CB0 = d["seed"](L, "CB")
d["install_seed"](L, M_EQ0)   # equivariant boundary everywhere
TOL, MAXC = 1e-3, 16


def e_tot_of(J):
    def e_tot(Mr):
        Mf = field(Mr)
        E = e_static(Mf, "G")
        if J > 0:
            z = d["zeta_of"](L, Mf)
            I = 2.0 * H ** 3 * d["kin_density"](L, Mf, z).sum()
            E = E + J ** 2 / (2.0 * I)
        return E
    return e_tot


def minimize(M_start, J, adam=1000):
    e_tot = e_tot_of(J)
    M_raw = M_start.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(adam):
        opt.zero_grad()
        e_tot(M_raw).backward()
        opt.step()
    traj = [float(e_tot(M_raw).detach())]
    for cyc in range(MAXC):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_tot(M_raw)
            E.backward()
            return E
        opt2.step(closure)
        traj.append(float(e_tot(M_raw).detach()))
        if abs(traj[-1] - traj[-2]) < TOL:
            break
    g = torch.autograd.grad(e_tot(M_raw), M_raw)[0]
    return M_raw.detach(), traj, float(g.abs().max())


def shells(dens):
    c = (N - 1) / 2
    idx = torch.stack(torch.meshgrid(
        torch.arange(N), torch.arange(N), torch.arange(N),
        indexing="ij")).to(DT).to(DEV)
    rr = torch.sqrt(((idx - c) ** 2).sum(dim=0)) * H
    prof = []
    for r0 in np.arange(0, 18, 3.0):
        sel = (rr >= r0) & (rr < r0 + 3.0)
        prof.append(float(dens[sel].sum() * H ** 3))
    w = dens.clamp_min(0.0)
    cen = float((rr * w).sum() / w.sum().clamp_min(1e-30))
    return prof, cen


out = {"tol": TOL, "branches": {}}
k_ref = None
for J in (0.0, 2.0, 4.0, 6.0):
    for br, M0 in (("EQ", M_EQ0), ("CB", M_CB0)):
        Mr, traj, ginf = minimize(M0, J)
        Mf = field(Mr)
        z = d["zeta_of"](L, Mf)
        kz = d["kin_density"](L, Mf, z)
        I = float(2.0 * H ** 3 * kz.sum())
        Es = float(e_static(Mf, "G"))
        rec = {"J": J, "branch": br, "E_J": traj[-1], "E_stat": Es,
               "I": I, "cycles": len(traj) - 1,
               "last_dE": traj[-1] - traj[-2], "ginf": ginf}
        if J == 0.0 and br == "EQ":
            k_ref = kz.clone()
            np.savez_compressed(
                os.path.join(HERE, "results", "M_branch_EQ_J0.npz"),
                M=Mf.cpu().numpy())
        dk = kz - k_ref
        prof, cen = shells(dk)
        rec["excess_shell_profile"] = prof
        rec["excess_centroid_r"] = cen
        rec["I_excess_over_EQ0"] = float(2.0 * H ** 3
                                         * dk.clamp_min(0).sum())
        out["branches"][f"{br}_J{J}"] = rec
        if J == 4.0 and br == "CB":
            M_hyst_start = Mr.clone()
        print(f"[{br} J={J}] E_J {traj[-1]:.6f} (stat {Es:.6f}), "
              f"I {I:.4e}, cyc {len(traj)-1}, "
              f"last dE {traj[-1]-traj[-2]:+.1e}, |g| {ginf:.1e}; "
              f"centroid r {cen:.1f}", flush=True)
        del Mr, Mf, z, kz
        torch.cuda.empty_cache()

# hysteresis: melt the CB J=4 endpoint back at J=0
Mr, traj, ginf = minimize(M_hyst_start, 0.0, adam=200)
Mf = field(Mr)
z = d["zeta_of"](L, Mf)
kz = d["kin_density"](L, Mf, z)
I = float(2.0 * H ** 3 * kz.sum())
dk = kz - k_ref
prof, cen = shells(dk)
out["hysteresis"] = {"E": traj[-1], "I": I,
                     "cycles": len(traj) - 1, "ginf": ginf,
                     "excess_centroid_r": cen,
                     "I_excess_over_EQ0": float(
                         2.0 * H ** 3 * dk.clamp_min(0).sum())}
print(f"[hysteresis CB@J4 -> J0] E {traj[-1]:.6f}, I {I:.4e} "
      f"(from 5.9e2), cyc {len(traj)-1}", flush=True)

json.dump(out, open(os.path.join(HERE, "results",
                                 "centrifugal_branches.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("centrifugal branches computed in this run\n")
print("written: results/centrifugal_branches.json")
