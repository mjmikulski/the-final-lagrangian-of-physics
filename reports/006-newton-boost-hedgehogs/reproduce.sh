#!/bin/bash
# Regenerates all results and asserts the structural claims of the report.
# Each script self-asserts its claims; the block below re-asserts the
# headline numbers from the persisted JSONs. The I4/I1 cost figure is an
# EXTERNAL result (needs report 004's regenerated artifact): lattice_cost.py
# certifies it only when that artifact is present, and this run reports its
# status explicitly either way.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY check_structure.py
$PY verify_symbolic.py
$PY measure_energies.py
$PY verify_measures.py
$PY lattice_cost.py

$PY - <<'EOF'
import json, os
s = json.load(open("results/structure_results.json"))
assert s["F_time_components"] == 0.0
assert s["pseudoscalar_max"] < 1e-12
assert all(abs(v) < 1e-9 for v in s["identities"].values())
assert s["eig_identity_lo"] < 1e-9 and s["eig_identity_hi"] < 1e-9
assert s["rho_sampled"][0] > 1.0 and s["rho_sampled"][1] < 4.0 + 1e-9
assert s["random_sweep_worst"] < 1e-9

r = json.load(open("results/energy_results.json"))
assert all(abs(v - 4/3) < 1e-4 for v in r["virial"].values())
clean = 0
for k, v in r["tails"].items():
    assert v["all_repulsive"] and v["t1_min"] > 4/3
    assert all(row["X"] > 0 for row in v["rows"])
    clean += 0 if v["uv_flagged"] else 1
assert clean == 3                     # p in {0.3, 0.5} x mu: strict p < 3/4
clean_ceiling = r["t1_ceiling_clean"]
assert r["cluster_witness"] > clean_ceiling
assert r["ir_divergence_slope"] > 10
assert r["gauss_pocket"]["1.0"]["E1int"] < 0 < r["gauss_pocket"]["0.5"]["E1int"]
assert all(x > 0 for x in r["X_stability"]["variants"].values())
ng = r["no_go"]
assert ng["attraction_needs_t_above"] > ng["stability_singles_boundary_t"]
assert ng["alpha_neg_needs_beta_ratio_at_least"] > \
    ng["alpha_neg_attraction_allows_at_most"]

v = json.load(open("results/verify_results.json"))   # route 2 (measured)
assert all(abs(x - 4/3) < 1e-3 for x in v["virial"].values())
for row in v["pairs"].values():
    assert row["E1int"] > 0 and row["E4int"] > 0
    assert row["X"] > 0 and row["t1"] > 4/3
    assert max(row["rel_vs_route1"]) < 0.05
assert all(row["X"] > 0 for row in v["gauss_pair"].values())
assert v["chain7_ratio"] > 1.5

ext = json.load(open("results/lattice_cost_external.json"))
assert abs(ext["free_bulk_mean_ratio"] - 0.763) < 5e-3   # README quote
if os.path.exists("results/lattice_cost.json"):
    lc = json.load(open("results/lattice_cost.json"))
    assert abs(lc["free_bulk_mean_ratio"] - ext["free_bulk_mean_ratio"]) < 0.1
    print("REPRODUCED: all structural checks pass "
          "(I4/I1 cost regenerated and confirmed).")
else:
    print("REPRODUCED: all structural checks pass. NOTE: the I4/I1 cost "
          "figure is an EXTERNAL result (results/lattice_cost_external"
          ".json), not regenerated in this run -- see lattice_cost.py.")
EOF
