#!/usr/bin/env bash
# Reproduction pipeline for report 011. With report 004's stack (or
# M5_FIELDS_DIR) reruns the producers; without it the committed
# artifacts are checked. Sentinels distinguish a fresh run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -f results/relax_ran.flag results/twist_ran.flag \
      results/rotstab_ran.flag results/fixedj_ran.flag

$PY relax_all.py
$PY analysis.py
$PY frame_twist.py
$PY rotational_stabilization.py
$PY fixedj_cb.py
$PY make_figures.py

$PY - <<'EOF'
import json
import os

R = "results"
an = json.load(open(os.path.join(R, "analysis.json")))
ra = json.load(open(os.path.join(R, "relax_all.json")))
tw = json.load(open(os.path.join(R, "frame_twist.json")))
rs = json.load(open(os.path.join(R, "rot_stabilization.json")))

# (A) statics-extensivity check: the axial line-tension excess shrinks
# with box size (no extensive axial cost) ...
ex = an["lam_excess_EQ"]
assert ex[0] > ex[1] > ex[2] > 0, "axial excess must shrink with L"
assert ex[2] < 0.1 * ex[0]
# ... and falls with delta (negative at 1/8)
dd = an["axial_cost_delta"]
assert dd["d0125"] < 0 < dd["d0300"]

# (B) neither core deformation survives statics: the seed inertia
# excess collapses to the diffuse-background level
seed_dI = (ra["cases"]["CB_N24"]["pre"]["I_comb"]
           - ra["cases"]["EQ_N24"]["pre"]["I_comb"])
rel_dI = (ra["cases"]["CB_N24"]["post"]["I_comb"]
          - ra["cases"]["EQ_N24"]["post"]["I_comb"])
assert seed_dI > 300 and abs(rel_dI) < 0.05 * seed_dI
assert tw["I_diff_raw"] < 5.0, "frame twist must not survive statics"
assert tw["PR_excess"] > 300, "no core-localized excess remains"

# (C) centrifugal stabilization absent at moderate J, threshold point
# recorded; the qualitative I(J) rise is the only surviving trace
rows = {r["J"]: r for r in rs["rows"]}
assert rows[0.8]["I"] - rows[0.0]["I"] < 0.1 * rows[0.0]["I"]
# ... and PRESENT above it: the minimizer rebuilds a core-localized
# breaking deformation at J = 4
assert rows[4.0]["I"] > 3.0 * rows[0.0]["I"], "centrifugal jump"
ex4 = rs["excess_J4.0"]
assert ex4["PR_excess"] < 300, "excess must be core-localized"
assert rows[4.0]["E_stat"] - rows[0.0]["E_stat"] < 0.1

for f in ("fig_statics.png", "fig_survival.png"):
    assert os.path.getsize(os.path.join(R, f)) > 10000, f

ran = all(os.path.exists(os.path.join(R, f))
          for f in ("relax_ran.flag", "twist_ran.flag",
                    "rotstab_ran.flag", "fixedj_ran.flag"))
print("REPRODUCED" if ran else "STRUCTURAL CHECKS PASS on committed "
      "artifacts (004 stack absent)")
EOF
