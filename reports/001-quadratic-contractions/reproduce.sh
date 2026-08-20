#!/bin/bash
# Regenerates all results and asserts the structural claims of the report.
# Floating-point tails may differ across machines/BLAS; everything asserted
# here may not.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY enumerate_sympy.py
$PY check_torch.py
$PY verify_3x3_nogo.py

$PY - <<'EOF'
import json
c = json.load(open("results/contraction_classes.json"))
assert c["n_zero"] == 45 and len(c["classes"]) == 6
assert sorted(len(x["members"]) for x in c["classes"]) == [4, 4, 4, 16, 16, 16]
r = json.load(open("results/numerical_results.json"))
s = r["symmetries_physical_F"]
assert s["antisym_mn"] == 0.0 and s["antisym_ab"] == 0.0
assert s["pair_exchange"] > 0.1 and s["bianchi"] > 0.1
assert r["class_consistency_worst"] < 1e-12
assert r["rank_generic"]["rank"] == 6 and r["rank_physical"]["rank"] == 6
assert r["jacobian_generic"]["rank"] == 6 and r["jacobian_physical"]["rank"] == 6
assert all(v < 1e-9 for v in r["channel_identities"].values())
print("REPRODUCED: all structural checks pass.")
EOF
