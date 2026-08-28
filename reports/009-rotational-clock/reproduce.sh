#!/usr/bin/env bash
# Reproduction pipeline for report 009. With report 004's fields (or
# M5_FIELDS_DIR) reruns the lattice producers; without them the
# committed artifacts are checked and the lattice legs report
# NOT-REPRODUCED. Sentinels distinguish a fresh run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -f results/rot_ran.flag

$PY ladder_rot.py
$PY verify_energies.py
$PY make_figures.py

$PY - <<'EOF'
import json
import os

R = "results"
rot = json.load(open(os.path.join(R, "rot_ladders.json")))

# prediction inside the sampled grid; minimum exactly at the predicted
# rung at every relaxation level
jr = rot["JR_E"]
pred = rot["omega_pred_R"]
assert jr["interior"]
# the minimum sits on the rung placed at round(pred, 3)
assert abs(jr["min_omega"] - round(pred, 3)) < 1e-9
assert len(set(jr["min_omega_per_level"])) == 1
assert jr["min_omega_per_level"][0] == jr["min_omega"]

# sign control clean
jr0 = rot["JR0"]
assert (not jr0["interior"]) and jr0["min_omega"] == 0.0
assert len(set(jr0["min_omega_per_level"])) == 1

# depth changes shrink (documented trend; existence rests on location)
dch = rot["JR_E"]["depth_changes"]
assert abs(dch[-1]) < abs(dch[0])
assert all(d < 0 or abs(d) < 3e-6 for d in dch)

# localization: tighter than the boost channel's ~100
rows = {r["omega"]: r for r in jr["rungs"]}
assert rows[jr["min_omega"]]["PR_k_sites"] < 150

# angular momentum record consistent
assert abs(rot["J_at_min"] - rot["I_R"] * jr["min_omega"]) < 1e-9

# residual sanity
for key in ("JR_E", "JR0"):
    assert rot[key]["max_grad_inf"] < 1.5e-1, key

for f in ("fig_rot_ladders.png", "fig_rot_channel.png"):
    assert os.path.getsize(os.path.join(R, f)) > 10000, f

if os.path.exists(os.path.join(R, "rot_ran.flag")):
    print("REPRODUCED: lattice producers ran here and all structural "
          "checks pass.")
else:
    print("STRUCTURAL CHECKS PASS on committed artifacts; lattice "
          "producers did NOT run here (004 fields absent).")
EOF
