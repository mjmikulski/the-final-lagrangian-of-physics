#!/usr/bin/env bash
# Reproduction pipeline for report 012. With report 004's stack (the
# hedgehog seed the producers load) the lattice producers rerun;
# without it the committed artifacts are checked. Sentinels
# distinguish a fresh run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -f results/grid_ran.flag results/ladders_ran.flag results/fix1_ran.flag results/fix2_ran.flag

$PY grid_scan.py
$PY extended_ladders.py
$PY fix_round1.py
$PY fix_round2.py
$PY analysis.py
$PY make_figures.py

$PY - <<'EOF'
import json
import os

R = "results"
G = json.load(open(os.path.join(R, "grid.json")))
V = json.load(open(os.path.join(R, "scaling_verdicts.json")))
E = json.load(open(os.path.join(R, "extended_ladders_all.json")))

assert len(G["points"]) == 9
assert not any("FAILED" in p for p in G["points"])

# delta-flatness of the headline observables (three octaves, < 2%)
for ob in ("time_part_G", "time_part_eta", "om_pred", "C1",
           "I_pure", "I_comb", "mix34_curv", "E_stat"):
    sp = max(V["observables"][ob]["delta_spread_per_g"].values())
    assert sp < 0.02, (ob, sp)

# both time-part signs stable at all nine points
assert V["sign_stable_G"] and V["sign_stable_eta"]

# the g-slope sign pattern (coarse trends; signs are the claim)
sl = V["observables"]
assert sl["time_part_G"]["g_loglog_slope"] > 0.05
assert sl["om_pred"]["g_loglog_slope"] < -0.05
assert sl["C2"]["g_loglog_slope"] > 0.3
assert abs(sl["E_stat"]["g_loglog_slope"]) < 0.01

# extended ladders: interior minimum at every g; the prediction
# degrades monotonically; the g = 8 PR control is large (protocol,
# not g, drives the delocalization measured here)
lad = V["ladders"]
ratios = []
for g in ("8", "64", "512"):
    assert lad[g]["interior"], g
    ratios.append(lad[g]["min_over_pred"])
assert ratios[0] < ratios[1] < ratios[2]
assert abs(ratios[0] - 1.0) < 0.05
assert lad["8"]["PR_at_min"] > 800, "the g=8 control must be large"

# round-1 fixes: delta flat at the target scale; true float32
# degradation at machine-epsilon level; the absolute-potential row's
# documented breakdown at g = 512
FX = json.load(open(os.path.join(R, "fix_round1.json")))
base = next(p for p in G["points"]
            if abs(p["delta"] - 0.125) < 1e-9 and p["g"] == 8)
for r in FX["C_delta_extension"]:
    assert abs(r["time_part_G"] - base["time_part_G"]) \
        < 0.01 * abs(base["time_part_G"]), r["delta"]
    assert abs(r["om_pred"] - base["om_pred"]) < 0.01 * base["om_pred"]
    assert r["true_float32_rel"] < 1e-6
A = {r["g"]: r for r in FX["A_absolute_row"]}
assert all(r["time_part_G"] < 0 for r in FX["A_absolute_row"])
# g = 512 is documented as broken and is NOT used in any trend claim
assert A[512]["ginf"] > 1.0, "documented absolute-potential breakdown"
assert A[512]["cancellation"][3]["signal_over_ulp"] < 100
for r in FX["A_absolute_row"]:
    assert r["true_float32_rel"] < 1e-6

# round-2 fixes: the original-theory trend on observable-CONVERGED
# profiles (g = 8, 64 only), with the full sign structure; and the
# at-target delta points
F2 = json.load(open(os.path.join(R, "fix_round2.json")))
AC = {r["g"]: r for r in F2["absolute_converged"]}
for g_ in (8, 64):
    assert AC[g_]["stopped_on_observable"], g_
    assert AC[g_]["final"]["time_part_G"] < 0
    assert AC[g_]["final"]["time_part_eta"] > 0, "eta inert (absolute)"
assert abs(AC[8]["final"]["time_part_G"]) \
    < abs(AC[64]["final"]["time_part_G"]), "drive grows (converged)"
assert AC[64]["final"]["om_pred"] > AC[8]["final"]["om_pred"]
for r in F2["delta_at_target"]:
    assert r["stopped_on_observable"]
    assert abs(r["final"]["time_part_G"] - base["time_part_G"]) \
        < 0.01 * abs(base["time_part_G"]), r["delta"]
    assert abs(r["final"]["om_pred"] - base["om_pred"]) \
        < 0.01 * base["om_pred"]
# the old input-quantization diagnostic is retained in grid.json but
# no longer carries the cleanliness claim
assert V["max_float32_rel"] < 1e-9

for f in ("fig_grid.png", "fig_ladders.png"):
    assert os.path.getsize(os.path.join(R, f)) > 10000, f

ran = all(os.path.exists(os.path.join(R, f))
          for f in ("grid_ran.flag", "ladders_ran.flag",
                    "fix1_ran.flag", "fix2_ran.flag"))
print("REPRODUCED" if ran else "STRUCTURAL CHECKS PASS on committed "
      "artifacts (004 stack absent)")
EOF
