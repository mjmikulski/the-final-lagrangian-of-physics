"""The rotational clock: angular momentum from energy minimization.

Report 008 established the clock mechanism for the boost channel: the
G-form energy ansatz E_extra = gamma*int (i1s - k)^2 with the kinetic
density k of a frozen conjugation tangent. Here the SAME functional is
run with the ROTATION tangent (rot_xy generator),
dM/dt = omega * a_rot, a_rot = env*(W M - M W)/||.||, W antisymmetric.
This is the angular-momentum realization: the defect does not just
tick, it rotates, with a canonical channel angular momentum.

Conventions and protocol are inherited from report 008 verbatim
(densities and relaxation via runpy of ../008-i1-squared-clock/
ladder_i1sq_defs.py): fresh-start rungs from the polished hedgehog,
Adam 500 + L-BFGS cycles (4 for the main ladder, 8 for the omega = 0
reference endpoint -- the reference creeps more slowly than the
well minimum converges, so it gets a deeper budget), E_levels per rung,
bracket stability + depth trend as the convergence evidence.

Frozen-profile prediction: omega_R = sqrt(C1r/C2r) with
C1r = int i1s*k1r, C2r = int k1r^2. Channel angular momentum in the
quadratic-kinetic reading (author-gated interpretation): the channel
kinetic energy is T = I_R/2 * omega^2 * (normalization absorbed), with
I_R = 2 * int k1r, giving J = I_R * omega at the sampled minimum.

A methodological correction recorded here: an earlier dev run using
report-004's channel construct bk = (<F>_GG - <F>_etaeta)/2 found a
spurious omega_R ~ 24.5, because for ROTATION tangents the G and eta
contractions nearly coincide and the construct cancels the channel
density; the clean G contraction used here (and in 008) has no such
cancellation.

Ladders:
  JR_E : rotation tangent, energy reading (deep protocol) [persisted]
  JR0  : rotation tangent, flipped cross sign (control)

Needs report 004's polished field (or M5_FIELDS_DIR); NOT-REPRODUCED
notice otherwise. Out: results/rot_ladders.json
"""
import json
import os
import runpy
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
R008 = os.path.join(HERE, "..", "008-i1-squared-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "rot_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("ladder_rot: NOT REPRODUCED HERE -- needs report 004's "
          f"polished field in {FIELDS} (or M5_FIELDS_DIR). Committed "
          "results carry the recorded values.")
    sys.exit(0)

lad = runpy.run_path(os.path.join(R008, "ladder_i1sq_defs.py"))
field, e_static = lad["field"], lad["e_static"]
densities, H = lad["densities"], lad["H"]
M_pol = lad["M_pol"]
a0_of, gen_catalog = lad["a0_of"], lad["gen_catalog"]

_Mg = field(M_pol)
A0R = a0_of(gen_catalog()["rot_xy"], _Mg)
i1s0, k1r = densities(_Mg, A0R, 1.0, "G")
Es0 = e_static(_Mg, "G").item()
gamma = 0.05 * Es0 / (H ** 3 * (i1s0 ** 2).sum()).item()
C1r = (H ** 3 * (i1s0 * k1r).sum()).item()
C2r = (H ** 3 * (k1r ** 2).sum()).item()
om_R = (C1r / C2r) ** 0.5
I_R = 2.0 * (H ** 3 * k1r.sum()).item()
print(f"gamma = {gamma:.5f}; rot prediction omega_R = {om_R:.3f}; "
      f"channel inertia I_R = {I_R:.6e}")

np.savez_compressed(os.path.join(HERE, "results", "a0r_frozen.npz"),
                    a0=A0R.cpu().numpy())
results = {"gamma": gamma, "C1r": C1r, "C2r": C2r,
           "omega_pred_R": om_R, "I_R": I_R, "E_stat0": Es0}


def relax(e_total_fn, cycles):
    M_raw = M_pol.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        e_total_fn(M_raw).backward()
        opt.step()
    E_levels = [float(e_total_fn(M_raw).detach())]
    for cycle in range(cycles):
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


def run_ladder(tag, sign, omegas, cycles, endpoint_cycles=None,
               save=()):
    rungs = []
    for om in omegas:
        def e_tot(Mr, om=om):
            Mf = field(Mr)
            i1s, k = densities(Mf, A0R, om, "G")
            return (e_static(Mf, "G")
                    + gamma * H ** 3 * ((i1s + sign * k) ** 2).sum())
        cyc = endpoint_cycles if (om == 0.0 and endpoint_cycles) \
            else cycles
        M_raw, E_levels, ginf = relax(e_tot, cyc)
        Mf = field(M_raw)
        E = float(e_tot(M_raw))
        _, kd = densities(Mf, A0R, max(om, 1e-9), "G")
        pr = ((kd.sum() ** 2) / (kd ** 2).sum().clamp_min(1e-30)).item()
        rungs.append({"omega": om, "E_total": E, "PR_k_sites": pr,
                      "grad_inf": ginf, "E_levels": E_levels})
        print(f"  [{tag}] omega {om}: E {E:.6f}, PR {pr:.0f}, "
              f"|g|inf {ginf:.1e}, levels "
              f"{['%.6f' % e for e in E_levels]}", flush=True)
        if om in save:
            np.savez_compressed(
                os.path.join(HERE, "results",
                             f"rot_rung_om{str(om).replace('.','')}.npz"),
                M=Mf.cpu().numpy())
    k = min(range(len(rungs)), key=lambda i: rungs[i]["E_total"])
    ncmp = min(len(r["E_levels"]) for r in rungs)
    min_per_level = [
        min(range(len(rungs)), key=lambda i: rungs[i]["E_levels"][lv])
        for lv in range(ncmp)]
    v = {"rungs": rungs, "min_omega": rungs[k]["omega"],
         "interior": bool(0 < k < len(rungs) - 1),
         "min_omega_per_level": [rungs[i]["omega"]
                                 for i in min_per_level],
         "max_grad_inf": max(r["grad_inf"] for r in rungs)}
    print(f"  [{tag}] verdict: min at {v['min_omega']}, interior "
          f"{v['interior']}, per-level {v['min_omega_per_level']}")
    return v


rungs_R = tuple(round(f * om_R, 3)
                for f in (0.0, 0.35, 0.7, 1.0, 1.4, 2.0, 3.0))
print(f"JR_E rungs: {rungs_R}")
results["JR_E"] = run_ladder("JR_E", -1.0, rungs_R, cycles=4,
                             endpoint_cycles=8,
                             save=(rungs_R[2], rungs_R[3], rungs_R[4]))
_r = {r["omega"]: r for r in results["JR_E"]["rungs"]}
ncmp = min(len(_r[0.0]["E_levels"]), len(_r[rungs_R[3]]["E_levels"]))
_depths = [_r[0.0]["E_levels"][lv] - _r[rungs_R[3]]["E_levels"][lv]
           for lv in range(ncmp)]
# reference-extended depth: the deep endpoint against the converged min
_depths_deep = [_r[0.0]["E_levels"][lv] - _r[rungs_R[3]]["E_total"]
                for lv in range(len(_r[0.0]["E_levels"]))]
results["JR_E"]["depth_per_level"] = _depths
results["JR_E"]["depth_deep_endpoint"] = _depths_deep
results["JR_E"]["depth_changes"] = [_depths[i + 1] - _depths[i]
                                    for i in range(len(_depths) - 1)]
print(f"  [JR_E] depth per level: {['%.3e' % d for d in _depths]}")
print(f"  [JR_E] depth vs deep endpoint: "
      f"{['%.3e' % d for d in _depths_deep]}")
om_min = results["JR_E"]["min_omega"]
results["J_at_min"] = I_R * om_min
print(f"  J at sampled minimum: I_R * {om_min} = "
      f"{results['J_at_min']:.6e}")

print("JR0: rotational sign control:")
results["JR0"] = run_ladder("JR0", +1.0,
                            (0.0, rungs_R[2], rungs_R[3], rungs_R[5]),
                            cycles=2)

with open(os.path.join(HERE, "results", "rot_ladders.json"), "w") as f:
    json.dump(results, f, indent=1)
with open(FLAG, "w") as f:
    f.write("rotational ladders computed in this run\n")
print("written: results/rot_ladders.json")
