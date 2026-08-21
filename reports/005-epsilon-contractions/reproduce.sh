#!/bin/bash
# Regenerates all results and asserts the structural claims of the report.
# Floating-point tails may differ across machines/BLAS; everything asserted
# here may not. verify_exact.py additionally asserts its claims internally,
# in exact integer arithmetic.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY check_torch.py
$PY verify_exact.py

$PY - <<'EOF'
import json
r = json.load(open("results/numerical_results.json"))

c = r["counts"]
assert c["total"] == 210 and c["identically_zero"] == 54
assert c["n_classes"] == 13
assert sorted(c["class_sizes"]) == [2, 2, 4, 4] + [16] * 9

assert r["rank_generic"] == 4 and r["rank_physical"] == 3
assert r["dead_classes_generic"] == [] and len(r["dead_classes_physical"]) == 1

assert r["pdd_physical_max"] < 1e-8 and r["pdd_cyclic_identity_max"] < 1e-8
assert r["pdd_generic_mean"] > 1.0          # nonzero off the model fields

assert r["named_rank"] == 4
assert r["worst_expansion_residual"] < 1e-12

assert r["proper_invariance"]["dI"] < 1e-9 and r["proper_invariance"]["dP"] < 1e-9
assert r["parity"]["dI"] < 1e-12 and r["parity"]["dP_flip_resid"] < 1e-12
assert r["parity_on_A"]["even"] < 1e-9 and r["parity_on_A"]["odd"] < 1e-9

assert r["spatial_max"] == 0.0
assert r["clock_I"] == [4.0, 4.0, 2.0, 2.0, 2.0, 4.0]
assert r["clock_pseudo_max"] == 0.0 and r["clock_orbit_pseudo_max"] < 1e-12

assert r["two_eps_nonzero"] == 70 and r["two_eps_worst_residual"] < 1e-12

for ens in ("generic", "physical"):
    assert all(v < 1e-12 for v in r["identities"][ens].values())

assert r["jac_rank_generic"] == 9 and r["jac_rank_physical"] == 8

nl = r["null_lagrangian"]
for k in ("phi", "chi"):
    assert nl[k]["el_max"] < 1e-9 * nl[k]["scale"]          # null
for k in ("P_mm", "P_dm", "P_cp", "I1_control"):
    assert nl[k]["el_max"] > 1e-3 * nl[k]["scale"]          # dynamical

print("REPRODUCED: all structural checks pass.")
EOF
