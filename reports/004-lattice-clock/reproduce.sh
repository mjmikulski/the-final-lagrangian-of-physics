#!/bin/bash
# Full lattice run: ~45 min on a CUDA GPU (float64), several hours on CPU.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
$PY lattice.py gate statG kin ladder hessq1
$PY ladder_ext.py
$PY - <<'PYEOF'
import json
r = json.load(open("results/lattice_results.json"))
assert r["vacuum_energy"]["eta"] == 0.0 and r["vacuum_energy"]["G"] == 0.0
assert r["baseline_eta"]["offblock"] == 0.0 and r["baseline_eta"]["E"] < 5.5
sg = r["statG"]
assert sg["offblock"] == 0.0 and sg["spectral_gap_min"] > 1.0
assert abs(sg["E"] - sg["E_eta_of_same"]) / sg["E"] < 1e-2
assert all(v > 0 for v in r["kin_table_G"].values())
assert r["hessian_q1"]["lam_min_est"] > 0
ext = r["ladder_ext"]
assert ext["interior"] is False                     # the honest negative
prs = [x["participation_sites"] for x in ext["rungs"]]
assert prs[-1] / prs[0] > 5                          # delocalization is real
print("REPRODUCED: all structural checks pass.")
PYEOF
