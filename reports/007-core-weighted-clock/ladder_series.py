"""Lattice ladder series: does a core-supported coefficient localize the
clock, and which quartic form does it take?

Four ladders, all in the report-004 protocol (same stack, generator
selection, calibration b = a*K1/(6*K2*omega_t^2) at omega_t = 0.8, 500
Adam steps per rung, participation ratio PR = (sum B_k)^2 / sum B_k^2):

  L1 dynamic-local : local quartic density * dynamic weight c(M) =
                     v4/(v4 + 0.05*max v4)  -- the naive c(M) proposal;
  L2 frozen-local  : local quartic * FROZEN sharp mask (frac = 0.5);
  L3 intensive     : global quartic E_c = -a*B + 3b*B^2,
                     B = H^3 sum(c_w b_k), frozen mask, transfer ladder
                     (field carried between rungs);
  L4 intensive-fresh: same, every rung restarted from the polished field
                     (no hysteresis). Rung fields at omega = 0.5/0.8/1.1
                     are persisted for the independent energy route
                     (verify_energies.py);
  L5 intensive-DYNAMIC: the review-requested decisive variant -- the
                     intensive quartic with the weight recomputed from
                     the CURRENT field at every optimization step
                     (a genuine functional of M, no external mask),
                     fresh-start protocol. Note the intensive form
                     removes the self-widening incentive of L1: B is
                     driven to B* = a/(6b), so growing the weight's
                     support past that point is penalized, not paid.

Field inputs: report 004's regenerated artifacts results/M_G.npz and
results/M_G_polished.npz (run ../004-lattice-clock/reproduce.sh first),
or a directory given in the env var M5_FIELDS_DIR containing those files
(used to produce the committed results from the 004-line working fields;
provenance recorded in the JSON). Without either, exits with a loud
NOT-REPRODUCED notice.

Out: results/ladder_series.json (+ results/fresh_rung_om{05,08,11}.npz)
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
if not (os.path.exists(os.path.join(FIELDS, "M_G.npz"))
        and os.path.exists(os.path.join(FIELDS, "M_G_polished.npz"))):
    flag = os.path.join(HERE, "results", "ladder_ran.flag")
    if os.path.exists(flag):
        os.remove(flag)
    print("ladder_series: NOT REPRODUCED HERE -- needs report 004's "
          "regenerated fields (M_G.npz, M_G_polished.npz) in "
          f"{FIELDS} (or set M5_FIELDS_DIR). Committed results carry the "
          "recorded values with provenance.")
    sys.exit(0)

L = runpy.run_path(os.path.join(R004, "lattice.py"), run_name="not_main")
field, e_static = L["field"], L["e_static"]
boost_channels, a0_of, gen_catalog = (L["boost_channels"], L["a0_of"],
                                      L["gen_catalog"])
H, DT, DEV, ETA, C_P = L["H"], L["DT"], L["DEV"], L["ETA"], L["C_P"]

M_start = torch.tensor(np.load(os.path.join(FIELDS, "M_G.npz"))["M"],
                       dtype=DT, device=DEV)
M_pol = torch.tensor(np.load(os.path.join(FIELDS, "M_G_polished.npz"))["M"],
                     dtype=DT, device=DEV)
OMEGAS = (0.0, 0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.4, 2.8)
A_C, OM_T = 1.0, 0.8


def v4_density(M):
    Me = M @ ETA
    P, v4 = Me, 0.0
    for p in range(4):
        if p:
            P = P @ Me
        t = torch.einsum("...kk->...", P)
        v4 = v4 + (t - C_P[p]) ** 2
    return v4


def sigmoid_w(M, v0):
    v4 = v4_density(M)
    return v4 / (v4 + v0)


def pick_generator(Mg, w):
    best, K1, K2 = None, 0.0, 0.0
    for name in ("boost_x", "boost_y", "boost_z"):
        a0 = a0_of(gen_catalog()[name], Mg)
        bk, _ = boost_channels(Mg, a0, 1.0)
        k1 = (H ** 3 * (w * bk).sum()).item()
        if k1 > K1:
            best, K1 = name, k1
            K2 = (H ** 3 * (w * bk ** 2).sum()).item()
    return best, K1, K2


def run_ladder(tag, M0, e_cond_fn, fresh=False, save_rungs=None):
    rungs, M_raw = [], M0.clone()
    for om in OMEGAS:
        if fresh:
            M_raw = M0.clone()
        if om > 0:
            M_raw = M_raw.clone().requires_grad_(True)
            opt = torch.optim.Adam([M_raw], lr=1e-3)
            for it in range(500):
                opt.zero_grad()
                Mf = field(M_raw)
                (e_static(Mf, "G") + e_cond_fn(Mf, om)).backward()
                opt.step()
            M_raw = M_raw.detach()
        Mf = field(M_raw)
        Es = e_static(Mf, "G").item()
        Ec = e_cond_fn(Mf, om).item() if om > 0 else 0.0
        bk, _ = boost_channels(Mf, a0_glob, max(om, 1e-9))
        pr = ((bk.sum() ** 2) / (bk ** 2).sum().clamp_min(1e-30)).item()
        rungs.append({"omega": om, "E_total": Es + Ec, "E_stat": Es,
                      "E_cond": Ec, "PR_bk_sites": pr})
        print(f"  [{tag}] omega {om}: E {Es+Ec:.4f} (stat {Es:.4f}, "
              f"cond {Ec:.4f}), PR {pr:.0f}", flush=True)
        if save_rungs and om in save_rungs:
            np.savez_compressed(
                os.path.join(HERE, "results",
                             f"fresh_rung_om{str(om).replace('.', '')}.npz"),
                M=Mf.cpu().numpy())
    k = min(range(len(rungs)), key=lambda i: rungs[i]["E_total"])
    return {"rungs": rungs, "min_omega": rungs[k]["omega"],
            "interior": bool(0 < k < len(rungs) - 1)}


results = {"provenance": {
    "fields_dir": os.path.abspath(FIELDS),
    "note": ("fields from report 004's reproduce path, or the 004-line "
             "working fields (M5_FIELDS_DIR); see README provenance")}}

Mg0 = field(M_start)
v4h0 = v4_density(Mg0)
Mgp = field(M_pol)
v4hp = v4_density(Mgp)

# L1: dynamic-local
v0_1 = (0.05 * v4h0.max()).item()
gen, K1, K2 = pick_generator(Mg0, sigmoid_w(Mg0, v0_1))
b1 = A_C * K1 / (6 * K2 * OM_T ** 2)
a0_glob = a0_of(gen_catalog()[gen], Mg0)
print(f"L1 dynamic-local: gen {gen}, K1 {K1:.4f}, b {b1:.2f}")
results["L1_dynamic_local"] = run_ladder(
    "L1", M_start,
    lambda Mf, om: H ** 3 * (sigmoid_w(Mf, v0_1) * (
        -A_C * boost_channels(Mf, a0_glob, om)[0]
        + 3 * b1 * boost_channels(Mf, a0_glob, om)[0] ** 2)).sum())
results["L1_dynamic_local"]["setup"] = {"gen": gen, "K1": K1, "K2": K2,
                                        "b": b1, "v0": v0_1}

# L2: frozen-local, sharp mask
v0_2 = (0.5 * v4h0.max()).item()
CW2 = sigmoid_w(Mg0, v0_2).detach()
gen, K1, K2 = pick_generator(Mg0, CW2)
b2 = A_C * K1 / (6 * K2 * OM_T ** 2)
a0_glob = a0_of(gen_catalog()[gen], Mg0)
print(f"L2 frozen-local: gen {gen}, K1 {K1:.4f}, b {b2:.2f}, "
      f"core {int((CW2 > 0.5).sum())} sites")
results["L2_frozen_local"] = run_ladder(
    "L2", M_start,
    lambda Mf, om: H ** 3 * (CW2 * (
        -A_C * boost_channels(Mf, a0_glob, om)[0]
        + 3 * b2 * boost_channels(Mf, a0_glob, om)[0] ** 2)).sum())
results["L2_frozen_local"]["setup"] = {"gen": gen, "K1": K1, "K2": K2,
                                       "b": b2, "v0": v0_2}

# L3/L4: intensive quartic, frozen mask from the polished field
v0_3 = (0.5 * v4hp.max()).item()
CW3 = sigmoid_w(Mgp, v0_3).detach()
gen, K1, _ = pick_generator(Mgp, CW3)
b3 = A_C / (6 * K1 * OM_T ** 2)
a0_glob = a0_of(gen_catalog()[gen], Mgp)
np.savez_compressed(os.path.join(HERE, "results", "a0_frozen.npz"),
                    a0=a0_glob.cpu().numpy())   # for verify_energies.py
np.savez_compressed(os.path.join(HERE, "results", "cw_frozen.npz"),
                    cw=CW3.cpu().numpy())       # frozen mask, ditto
print(f"L3/L4 intensive: gen {gen}, K1 {K1:.4f}, b {b3:.3f}, "
      f"core {int((CW3 > 0.5).sum())} sites")


def e_cond_int(Mf, om):
    bk, _ = boost_channels(Mf, a0_glob, om)
    B = H ** 3 * (CW3 * bk).sum()
    return -A_C * B + 3 * b3 * B ** 2


results["L3_intensive_transfer"] = run_ladder("L3", M_pol, e_cond_int)
results["L3_intensive_transfer"]["setup"] = {"gen": gen, "K1": K1,
                                             "b": b3, "v0": v0_3}
results["L4_intensive_fresh"] = run_ladder("L4", M_pol, e_cond_int,
                                           fresh=True,
                                           save_rungs=(0.5, 0.8, 1.1))
results["L4_intensive_fresh"]["setup"] = results[
    "L3_intensive_transfer"]["setup"]


def e_cond_dyn(Mf, om):
    """Intensive quartic with the weight as a genuine functional of the
    CURRENT field (review round 1, P1): no frozen mask anywhere."""
    bk, _ = boost_channels(Mf, a0_glob, om)
    cw = sigmoid_w(Mf, v0_3)
    B = H ** 3 * (cw * bk).sum()
    return -A_C * B + 3 * b3 * B ** 2


results["L5_intensive_dynamic"] = run_ladder("L5", M_pol, e_cond_dyn,
                                             fresh=True)
results["L5_intensive_dynamic"]["setup"] = results[
    "L3_intensive_transfer"]["setup"]

print("\nverdicts:")
for k in ("L1_dynamic_local", "L2_frozen_local", "L3_intensive_transfer",
          "L4_intensive_fresh", "L5_intensive_dynamic"):
    v = results[k]
    print(f"  {k}: min at omega {v['min_omega']}, interior {v['interior']}")
assert not results["L1_dynamic_local"]["interior"]
assert not results["L2_frozen_local"]["interior"]
assert results["L4_intensive_fresh"]["interior"]
assert results["L4_intensive_fresh"]["min_omega"] == OM_T

with open(os.path.join(HERE, "results", "ladder_series.json"), "w") as f:
    json.dump(results, f, indent=1)
open(os.path.join(HERE, "results", "ladder_ran.flag"), "w").write("ran\n")
print("written: results/ladder_series.json")
