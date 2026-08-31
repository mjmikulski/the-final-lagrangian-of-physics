"""Report 013 producer 1: the converged-level fate of report 010's
candidate wells.

Question (the sharp open question of report 010): inside the measured
two-sided coupling window, do the fixed-depth interior minima of the
canonical Hamiltonian survive relaxation deepened to an
OBSERVABLE-level criterion, or does the slow dilution drift (seen at
six cycles) continue indefinitely?

Notation, self-contained: report 010 scans Lagrangians
L = -1/2 I1 + gamma (I_j)^2 - V over a family of invariants; its C10
cell (the covariant completion of report 003's B_k: the eta-norm
squared of the time-leg of F along the field's clock axis u) showed
interior minima of the canonical H at relaxation depth of six L-BFGS
cycles, with the minimum at x14 coupling migrating to the top sampled
rung at the sixth cycle. Here the same functional (imported verbatim
from report 010's committed stack) is continued:

  x14: from the PERSISTED sixth-cycle rung fields deep14_om*.npz,
       +18 more L-BFGS(150) cycles per rung (24 total);
  x10: fresh from the base profile, Adam 500 + up to 24 cycles;
  stopping rule per gamma arm: the two bracket DIFFERENCES
  E(0.1)-E(0.15) and E(0.2)-E(0.15) [and E(0.28)-E(0.15)] must each
  drift by < 5% of the larger |difference| over four consecutive
  cycles -- an observable-level criterion on the well shape itself
  (the lesson of reports 011-012), not on the total energy.

Per cycle per rung: E, |grad|_inf. Out: results/window_deep.json
(+ final fields persisted).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R010 = os.path.join(HERE, "..", "010-fundamental-grid-clock")
sys.path.insert(0, R010)
os.chdir(R010)   # lattice_grid_defs uses relative results/ paths
FLAG = os.path.join(HERE, "results", "deep_ran.flag")
try:
    from lattice_grid_defs import (DEV, a0_of, field, gen_catalog,
                                   load_or_make_base)
    from e4_ladders import e_cell_fused
except Exception as e:
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print(f"continue_window: NOT REPRODUCED HERE -- report 010 stack "
          f"unavailable ({e!r}).")
    sys.exit(0)
if not os.path.exists(os.path.join(R010, "results", "pre_e4.json")):
    print("continue_window: NOT REPRODUCED HERE -- pre_e4.json absent.")
    sys.exit(0)

RUNGS = [0.0, 0.1, 0.15, 0.2, 0.28]
MAXC, WIN, DRIFT = 24, 4, 0.05

with open(os.path.join(R010, "results", "pre_e4.json")) as f:
    pe = json.load(f)
Mr = load_or_make_base()
a0 = a0_of(gen_catalog()[pe["generator"]], field(Mr))
gam1 = pe["cells"]["C10"]["gamma"]

out = {"rungs": RUNGS, "max_cycles": MAXC, "window": WIN,
       "drift_tol": DRIFT, "arms": {}}


def snap(M_raw, om, gam):
    Mv = M_raw.detach().requires_grad_(True)
    E = e_cell_fused(field(Mv), a0, om, gam, "C10")
    g = torch.autograd.grad(E, Mv)[0]
    return float(E.detach()), float(g.abs().max())


def one_cycle(M_raw, om, gam):
    opt2 = torch.optim.LBFGS([M_raw], max_iter=150, history_size=25,
                             tolerance_grad=1e-9, tolerance_change=0,
                             line_search_fn="strong_wolfe")

    def closure():
        opt2.zero_grad()
        E = e_cell_fused(field(M_raw), a0, om, gam, "C10")
        E.backward()
        return E
    opt2.step(closure)


def run_arm(tag, gam, starts, pre_adam):
    fields = {}
    hist = {str(om): {"E": [], "ginf": []} for om in RUNGS}
    for om in RUNGS:
        if starts == "persisted":
            src = os.path.join(R010, "results",
                               f"deep14_om{str(om).replace('.', '')}.npz")
            M0 = torch.tensor(np.load(src)["M"], dtype=Mr.dtype,
                              device=Mr.device)
            M_raw = M0.clone().requires_grad_(True)
        else:
            M_raw = Mr.clone().requires_grad_(True)
            opt = torch.optim.Adam([M_raw], lr=1e-3)
            for it in range(pre_adam):
                opt.zero_grad()
                e_cell_fused(field(M_raw), a0, om, gam,
                             "C10").backward()
                opt.step()
        fields[om] = M_raw
        E, g = snap(M_raw, om, gam)
        hist[str(om)]["E"].append(E)
        hist[str(om)]["ginf"].append(g)
    stopped = False
    for cyc in range(MAXC):
        for om in RUNGS:
            one_cycle(fields[om], om, gam)
            E, g = snap(fields[om], om, gam)
            hist[str(om)]["E"].append(E)
            hist[str(om)]["ginf"].append(g)
        d1 = [hist["0.1"]["E"][i] - hist["0.15"]["E"][i]
              for i in range(len(hist["0.1"]["E"]))]
        d2 = [hist["0.2"]["E"][i] - hist["0.15"]["E"][i]
              for i in range(len(d1))]
        d3 = [hist["0.28"]["E"][i] - hist["0.15"]["E"][i]
              for i in range(len(d1))]
        print(f"[{tag}] cycle {cyc+1}: "
              f"d(0.1) {d1[-1]:+.2e} d(0.2) {d2[-1]:+.2e} "
              f"d(0.28) {d3[-1]:+.2e}", flush=True)
        if len(d1) > WIN:
            ok = True
            for dd in (d1, d2, d3):
                ref = max(abs(dd[-1]), 1e-12)
                if max(abs(dd[-1 - i] - dd[-2 - i])
                       for i in range(WIN)) > DRIFT * ref:
                    ok = False
            if ok:
                stopped = True
                break
    for om in RUNGS:
        np.savez_compressed(
            os.path.join(HERE, "results",
                         f"win_{tag}_om{str(om).replace('.', '')}.npz"),
            M=field(fields[om].detach()).cpu().numpy())
    E_last = {om: hist[str(om)]["E"][-1] for om in RUNGS}
    kmin = min(RUNGS, key=lambda om: E_last[om])
    verdict = {"min_omega": kmin,
               "interior": bool(kmin not in (RUNGS[0], RUNGS[-1])),
               "stopped_on_observable": stopped,
               "cycles_run": len(hist["0.0"]["E"]) - 1}
    print(f"[{tag}] VERDICT: min at {kmin}, interior "
          f"{verdict['interior']}, obs-stop {stopped}", flush=True)
    out["arms"][tag] = {"gamma": gam, "history": hist,
                        "verdict": verdict}
    json.dump(out, open(os.path.join(HERE, "results",
                                     "window_deep.json"), "w"),
              indent=1)
    for om in RUNGS:
        del fields[om]
    torch.cuda.empty_cache()


run_arm("x14_continued", gam1 * 14.0, "persisted", 0)
run_arm("x10_fresh", gam1 * 10.0, "fresh", 500)
with open(FLAG, "w") as f:
    f.write("window deep continuation computed in this run\n")
print("written: results/window_deep.json")
