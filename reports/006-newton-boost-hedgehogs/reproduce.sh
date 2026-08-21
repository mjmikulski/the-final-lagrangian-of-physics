#!/bin/bash
# Regenerates all results and asserts the structural claims of the report.
# Each script self-asserts its claims; the block below re-asserts the
# headline numbers from the persisted JSONs. lattice_cost.py runs only if
# report 004's regenerated artifact is present (soft-skips otherwise; the
# committed results/lattice_cost.json records the measured values).
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY check_structure.py
$PY verify_symbolic.py
$PY measure_energies.py
$PY lattice_cost.py

$PY - <<'EOF'
import json
s = json.load(open("results/structure_results.json"))
assert s["F_time_components"] == 0.0
assert s["pseudoscalar_max"] < 1e-12
assert all(abs(v) < 1e-9 for v in s["identities"].values())
assert s["eig_identity_lo"] < 1e-9 and s["eig_identity_hi"] < 1e-9
assert s["rho_sampled"][0] > 1.0 and s["rho_sampled"][1] < 4.0 + 1e-9
assert s["random_sweep_worst"] < 1e-9

r = json.load(open("results/energy_results.json"))
assert all(abs(v - 4/3) < 1e-4 for v in r["virial"].values())
for k, v in r["tails"].items():
    assert v["all_repulsive"] and v["t1_min"] > 4/3
    assert all(row["X"] > 0 for row in v["rows"])
clean_ceiling = r["t1_ceiling_clean"]
assert r["cluster_witness"] > clean_ceiling
assert r["ir_divergence_slope"] > 10
assert r["gauss_pocket"]["1.0"]["E1int"] < 0 < r["gauss_pocket"]["0.5"]["E1int"]
assert all(x > 0 for x in r["X_stability"]["variants"].values())
ng = r["no_go"]
assert ng["attraction_needs_t_above"] > ng["stability_singles_boundary_t"]
assert ng["alpha_neg_needs_beta_ratio_at_least"] > \
    ng["alpha_neg_attraction_allows_at_most"]
print("REPRODUCED: all structural checks pass.")
EOF
