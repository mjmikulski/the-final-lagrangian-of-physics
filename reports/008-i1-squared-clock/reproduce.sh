#!/usr/bin/env bash
# Reproduction pipeline for report 008. With report 004's fields (or
# M5_FIELDS_DIR) reruns the lattice producers; without them the
# producers print NOT-REPRODUCED and the committed artifacts are
# checked for internal consistency. Run-status sentinels distinguish a
# fresh reproduction from committed results.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -f results/i1sq_ran.flag results/gamma_scaling_ran.flag \
      results/gamma16_ran.flag

$PY ladder_i1sq.py
$PY confirm_gamma_scaling.py
$PY gamma16_localization.py
$PY verify_energies.py
$PY make_figures.py

$PY - <<'EOF'
import json
import os

R = "results"
lad = json.load(open(os.path.join(R, "i1sq_ladders.json")))
gsc = json.load(open(os.path.join(R, "gamma_scaling.json")))

# frozen-profile prediction inside the sampled bracket
assert 0.2 < lad["omega_pred_frozen"] < 0.5

# J1: interior well at the 0.35 rung, localized
j1 = lad["J1_local_covariant"]
assert j1["interior"] and j1["min_omega"] == 0.35
rows = {r["omega"]: r for r in j1["rungs"]}
assert rows[0.2]["E_total"] > rows[0.35]["E_total"] < rows[0.5]["E_total"]
assert rows[0.35]["PR_bk_sites"] < 300, "must stay core-localized"

# J0 control: no interior well, minimum at zero
j0 = lad["J0_local_control"]
assert (not j0["interior"]) and j0["min_omega"] == 0.0
r0 = {r["omega"]: r["E_total"] for r in j0["rungs"]}
assert r0[0.0] <= min(r0.values()) + 1e-12

# gamma scaling: same minimum position, depth ratio near 4
assert gsc["interior"] and gsc["min_omega"] == 0.35
assert 3.0 < gsc["depth_ratio"] < 5.0

# deep-well (16x) localization: depth ~16x base, PR stays at core scale
g16 = json.load(open(os.path.join(R, "gamma16_localization.json")))
base_depth = gsc["base_depth"]
assert 12.0 < g16["depth"] / base_depth < 20.0
assert g16["PR_at_min"] < 300, "deep well must stay core-localized"

# J2 diagnosis: the intensive term's E_extra reaches ~0 inside the range
j2min = min(r["E_extra"] for r in lad["J2_intensive"]["rungs"])
assert abs(j2min) < 1e-4

# figures exist and are non-trivial
for f in ("fig_i1sq_ladders.png", "fig_mechanism.png"):
    assert os.path.getsize(os.path.join(R, f)) > 10000, f

ran = all(os.path.exists(os.path.join(R, f))
          for f in ("i1sq_ran.flag", "gamma_scaling_ran.flag",
                    "gamma16_ran.flag"))
if ran:
    print("REPRODUCED: lattice producers ran here and all structural "
          "checks pass.")
else:
    print("STRUCTURAL CHECKS PASS on committed artifacts; lattice "
          "producers did NOT run here (004 fields absent) -- see "
          "NOT-REPRODUCED notices above.")
EOF
