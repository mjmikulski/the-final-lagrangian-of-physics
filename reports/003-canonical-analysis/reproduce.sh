#!/bin/bash
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
$PY hamiltonian_tests.py
$PY - <<'PYEOF'
import json
r = json.load(open("results/hamiltonian_results.json"))
s = r["symbolic"]
assert "-b*s**2" in s["H_C2"].replace(" ", "") or "- b*s**2" in s["H_C2"]
assert s["H_C3"].replace(" ", "") == "-a*Bk+3*b*Bk**2"
assert s["omega_star_C3_sq"].replace(" ", "") == "a/(6*b*k)"
assert abs(r["C3_omega_star"] - (1.0 / (6 * 0.05 * 4.0)) ** 0.5) < 1e-3
assert r["Bk_spatial"] == 0.0
assert r["C3_covariance"] < 1e-12
assert r["d2H_domega2_at_star"] > 0
assert min(r["hessian_H_at_omega_star"]) > -1e-9
assert r["hessian_quadratic_min"] > -1e-9
assert any(e < -1 for e in r["hessian_C3"]["clock_w*"])   # branched zone real
print("REPRODUCED: all structural checks pass.")
PYEOF
