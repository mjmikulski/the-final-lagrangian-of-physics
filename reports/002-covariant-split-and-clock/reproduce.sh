#!/bin/bash
# Regenerates all results and figures, then asserts every structural claim
# of the report. Floating-point tails are machine-dependent; the asserted
# structure is not.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY proto.py
$PY clock_tests.py
$PY make_figures.py
$PY dual_identity.py

$PY - <<'PYEOF'
import json
p = json.load(open("results/proto_results.json"))
assert p["covariance_exact"]["rel"] < 1e-9 < p["covariance_exact"]["neg_control"]
assert p["covariance_soft_n4"]["rel"] < 1e-9
assert p["reduction_3x3"]["exact"] == 0.0
assert p["A1_lagrange"]["reduction_onpotential"] < 1e-12
assert p["complex_step_soft"] < 1e-12
for ch, row in p["kin_table"].items():
    if ch.startswith("rot"):
        assert row["eta"] > 0 and abs(row["eta"] - row["G"]) < 1e-9
    else:
        assert row["eta"] < 0 < row["G"]
        assert abs(row["G"] + row["eta"]) < 1e-9     # same magnitude
        assert row["B"] < -1e3                       # literal -g^4 is worse
c = json.load(open("results/clock_results.json"))
assert c["B1_hamiltonian_dropout"] < 1e-12
b3 = c["B3_counterexample"]
assert abs(b3["omega_star"] - b3["omega_pred"]) <= 0.05 + 1e-9
assert b3["E_min"] < b3["E_0"]
d = c["B3_dive_scan"]
assert d["min_density"] >= d["floor"] - 1e-9
assert c["B3_spatial_guard"] == 0.0
lg = c["legendre"]
assert lg["caustic_at_minimum"] is True
assert abs(lg["omega_star_H"] - (1.0 / (6 * 0.05 * 4.0)) ** 0.5) < 1e-6
assert c["kinetic_matrix_allG"]["min_eig"] > 0
print("REPRODUCED: all structural checks pass.")
PYEOF
