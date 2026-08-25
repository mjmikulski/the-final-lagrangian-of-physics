#!/bin/bash
# Regenerates all results and asserts the structural claims. The lattice
# scripts need report 004's regenerated fields (or M5_FIELDS_DIR); without
# them they print NOT-REPRODUCED-HERE and the committed results stand as
# the (provenance-pinned) record. Run-status sentinels (results/*.flag)
# are cleared here and written only by producers that actually computed
# in THIS run, so committed artifacts can never masquerade as a fresh
# reproduction (review round 1).
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
rm -f results/ladder_ran.flag results/kinetic_ran.flag

$PY check_blindness.py
$PY canonical_reduced.py
$PY kinetic_forms.py
$PY ladder_series.py
$PY verify_energies.py
$PY make_figures.py

$PY - <<'EOF'
import json, os
b = json.load(open("results/blindness.json"))
assert b["invariant_drift_rel"] < 1e-10
assert all(abs(x - 8) < 0.8 for x in b["topo_scaling"])

k = json.load(open("results/kinetic_forms.json"))
for c in k["cells"]:
    assert c["negs_c0"] == 0 and -1.0 < c["c_clock"] < -0.3
if "fd_route_rel" in k:
    assert k["fd_route_rel"] < 1e-6
cs = k["core_sweep"]
assert cs["all_exactly_one_negative"]
assert cs["a0_channel_defined_cells"] == cs["n_cells"]
assert -1.0 < cs["c_clock_min"] and cs["c_clock_max"] < -0.3
assert -2.5 < cs["c_a0_min"] and cs["c_a0_max"] < -0.9

r = json.load(open("results/ladder_series.json"))
assert not r["L1_dynamic_local"]["interior"]
assert not r["L2_frozen_local"]["interior"]
for key in ("L4_intensive_fresh", "L5_intensive_dynamic"):
    assert r[key]["interior"], key
    assert r[key]["min_omega"] == 0.8, key
    rungs = {x["omega"]: x["E_total"] for x in r[key]["rungs"]}
    assert rungs[0.5] > rungs[0.8] < rungs[1.1], key
    pr = {x["omega"]: x["PR_bk_sites"] for x in r[key]["rungs"]}
    assert pr[0.8] < 300, key                # localized, not extensive
l1 = r["L1_dynamic_local"]["rungs"]
assert l1[-1]["PR_bk_sites"] > 1500          # the delocalized contrast

for f in ("fig_ladders.png", "fig_kinetic_window.png"):
    assert os.path.getsize(os.path.join("results", f)) > 10000

ran = all(os.path.exists(f"results/{f}")
          for f in ("ladder_ran.flag", "kinetic_ran.flag"))
print("REPRODUCED: all structural checks pass"
      + ("." if ran else
         " (lattice results from the committed, provenance-pinned "
         "record -- 004 fields absent, producers did NOT run here)."))
EOF
