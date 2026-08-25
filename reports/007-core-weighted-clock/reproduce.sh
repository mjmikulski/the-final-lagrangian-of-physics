#!/bin/bash
# Regenerates all results and asserts the structural claims. The lattice
# scripts need report 004's regenerated fields (or M5_FIELDS_DIR); without
# them they print NOT-REPRODUCED-HERE and the committed results stand as
# the (provenance-pinned) record -- the final block reports which state
# this run is in.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY check_blindness.py
$PY canonical_reduced.py
$PY kinetic_forms.py
$PY ladder_series.py
$PY verify_energies.py

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

r = json.load(open("results/ladder_series.json"))
assert not r["L1_dynamic_local"]["interior"]
assert not r["L2_frozen_local"]["interior"]
assert r["L4_intensive_fresh"]["interior"]
assert r["L4_intensive_fresh"]["min_omega"] == 0.8
rungs = {x["omega"]: x["E_total"] for x in r["L4_intensive_fresh"]["rungs"]}
assert rungs[0.5] > rungs[0.8] < rungs[1.1]
pr = {x["omega"]: x["PR_bk_sites"] for x in r["L4_intensive_fresh"]["rungs"]}
assert pr[0.8] < 300                       # localized, not extensive
l1 = r["L1_dynamic_local"]["rungs"]
assert l1[-1]["PR_bk_sites"] > 1500        # the delocalized contrast

fresh = all(os.path.exists(f"results/fresh_rung_om{t}.npz")
            for t in ("05", "08", "11"))
print("REPRODUCED: all structural checks pass"
      + ("." if fresh else
         " (lattice results from the committed, provenance-pinned "
         "record -- 004 fields absent in this run)."))
EOF
