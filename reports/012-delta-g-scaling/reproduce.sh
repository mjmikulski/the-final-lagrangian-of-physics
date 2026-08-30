#!/usr/bin/env bash
# Reproduction pipeline for report 012. With report 004's stack (the
# hedgehog seed the producers load) the lattice producers rerun;
# without it the committed artifacts are checked. Sentinels
# distinguish a fresh run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -f results/grid_ran.flag results/ladders_ran.flag

$PY grid_scan.py
$PY extended_ladders.py
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

# precision diagnostic across the grid
assert V["max_float32_rel"] < 1e-9

for f in ("fig_grid.png", "fig_ladders.png"):
    assert os.path.getsize(os.path.join(R, f)) > 10000, f

ran = all(os.path.exists(os.path.join(R, f))
          for f in ("grid_ran.flag", "ladders_ran.flag"))
print("REPRODUCED" if ran else "STRUCTURAL CHECKS PASS on committed "
      "artifacts (004 stack absent)")
EOF
