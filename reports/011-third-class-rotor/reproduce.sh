#!/usr/bin/env bash
# Reproduction pipeline for report 011. With report 004's stack (or
# M5_FIELDS_DIR) reruns the producers; without it the committed
# artifacts are checked. Sentinels distinguish a fresh run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -f results/relax_ran.flag results/twist_ran.flag \
      results/rotstab_ran.flag results/fixedj_ran.flag \
      results/plateau_ran.flag results/branches_ran.flag

$PY relax_all.py
$PY analysis.py
$PY frame_twist.py
$PY rotational_stabilization.py
$PY fixedj_cb.py
$PY lambda_plateau.py
$PY centrifugal_branches.py
$PY make_figures.py

$PY - <<'EOF'
import json
import os

R = "results"
an = json.load(open(os.path.join(R, "analysis.json")))
ra = json.load(open(os.path.join(R, "relax_all.json")))
tw = json.load(open(os.path.join(R, "frame_twist.json")))
rs = json.load(open(os.path.join(R, "rot_stabilization.json")))

# (A) statics: the deep-continued tube observable plateaus at a
# CONSTANT line tension for L >= 36 (the corrected claim)
lp = json.load(open(os.path.join(R, "lambda_plateau.json")))
p36 = lp["cases"]["EQ_N24"]["trajectory"][-1]["excess"]
p48 = lp["cases"]["EQ_N32"]["trajectory"][-1]["excess"]
assert p36 > 0 and p48 > 0
assert abs(p48 - p36) < 0.1 * p36, "constant line tension L>=36"
# the L=48 trajectory must show the rise that falsified round 1's
# shrinking claim
t48 = [t["excess"] for t in lp["cases"]["EQ_N32"]["trajectory"]]
assert t48[-1] > 5 * t48[0]
# delta = 1/8: negative along the WHOLE continued trajectory
t125 = [t["excess"] for t in lp["cases"]["EQ_d125"]["trajectory"]]
assert all(v < 0 for v in t125)

# (B) neither core deformation survives statics: the seed inertia
# excess collapses to the diffuse-background level
seed_dI = (ra["cases"]["CB_N24"]["pre"]["I_comb"]
           - ra["cases"]["EQ_N24"]["pre"]["I_comb"])
rel_dI = (ra["cases"]["CB_N24"]["post"]["I_comb"]
          - ra["cases"]["EQ_N24"]["post"]["I_comb"])
assert seed_dI > 300 and abs(rel_dI) < 0.05 * seed_dI
assert abs(tw["I_diff_raw"]) < 15.0, "frame twist collapse magnitude"
assert tw["PR_excess"] > 300, "no core-localized excess remains"

# (C) the two-branch record: spontaneity (EQ-start grows inertia),
# reversibility (hysteresis melts back), and the PERIPHERAL order
# parameter that withdrew the core-rotor headline
cbj = json.load(open(os.path.join(R, "centrifugal_branches.json")))
B = cbj["branches"]
assert B["EQ_J4.0"]["I"] > 4 * B["EQ_J0.0"]["I"], "spontaneous"
assert abs(B["EQ_J4.0"]["I"] - B["CB_J4.0"]["I"]) \
    < 0.15 * B["CB_J4.0"]["I"], "branches agree"
assert cbj["hysteresis"]["I"] < 1.3 * B["EQ_J0.0"]["I"], "reversible"
for J in ("2.0", "4.0", "6.0"):
    assert B[f"EQ_J{J}"]["excess_centroid_r"] > 12, "peripheral"
    prof = B[f"EQ_J{J}"]["excess_shell_profile"]
    assert sum(prof[:2]) < 0.1 * sum(prof[3:]), "not core-localized"
# the small-J record of rot_stabilization is retained as-is
rows = {r["J"]: r for r in rs["rows"]}
assert rows[0.8]["I"] - rows[0.0]["I"] < 0.1 * rows[0.0]["I"]

for f in ("fig_statics.png", "fig_survival.png",
          "fig_centrifugal.png"):
    assert os.path.getsize(os.path.join(R, f)) > 10000, f

ran = all(os.path.exists(os.path.join(R, f))
          for f in ("relax_ran.flag", "twist_ran.flag",
                    "rotstab_ran.flag", "fixedj_ran.flag",
                    "plateau_ran.flag", "branches_ran.flag"))
print("REPRODUCED" if ran else "STRUCTURAL CHECKS PASS on committed "
      "artifacts (004 stack absent)")
EOF
