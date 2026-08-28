#!/usr/bin/env bash
# Reproduction pipeline for report 008. With report 004's fields (or
# M5_FIELDS_DIR) reruns the lattice producers; without them the
# producers print NOT-REPRODUCED and the committed artifacts are
# checked for internal consistency. Run-status sentinels distinguish a
# fresh reproduction from committed results.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -f results/i1sq_ran.flag results/gamma_scaling_ran.flag \
      results/gamma16_ran.flag results/runaway_ran.flag

$PY ladder_i1sq.py
$PY fundamental_runaway.py
$PY confirm_gamma_scaling.py
$PY gamma16_localization.py
$PY verify_energies.py
$PY make_figures.py

$PY - <<'EOF'
import json
import os

R = "results"
lad = json.load(open(os.path.join(R, "i1sq_ladders.json")))
gsc = json.load(open(os.path.join(R, "gamma_scaling.json")))
g16 = json.load(open(os.path.join(R, "gamma16_localization.json")))

# prediction inside the sampled bracket
assert 0.2 < lad["omega_pred_E"] < 0.5

# convergence criteria (README section 6): bracket stability at every
# relaxation level, and a depth plateau for the deep JG_E run
for key in ("JG_E", "J_ETA", "J0"):
    lv = lad[key]["min_omega_per_level"]
    assert len(set(lv)) == 1 and lv[0] == lad[key]["min_omega"], key
dch = lad["JG_E"]["depth_changes"]
dpl = lad["JG_E"]["depth_per_level"]
assert abs(dch[-1]) < abs(dch[0]), "depth changes must shrink"
assert abs(dch[-1]) < 0.1 * abs(dpl[-1]), "final depth change < 10%"
# residuals recorded for transparency; sanity bound only (the steep
# high-omega rungs sit at a few 1e-2)
for key in ("JG_E", "J_ETA", "J0", "J2_intensive"):
    assert lad[key]["max_grad_inf"] < 1.5e-1, key

# fundamental-reading runaway: documented instability
rw = json.load(open(os.path.join(R, "fundamental_runaway.json")))
for run in rw["runs"]:
    assert run["trajectory"][-1]["E"] < -1e6, "runaway must be measured"
    assert run["trajectory"][-1]["max_i1s"] > 10 * rw["threshold_inv_gamma"]

# JG_E: interior well at the 0.35 rung, localized
jge = lad["JG_E"]
assert jge["interior"] and jge["min_omega"] == 0.35
rows = {r["omega"]: r for r in jge["rungs"]}
assert rows[0.2]["E_total"] > rows[0.35]["E_total"] < rows[0.5]["E_total"]
assert rows[0.35]["PR_k_sites"] < 300, "must stay core-localized"

# J_ETA: the faithful raw form is inert -- minimum at zero
jeta = lad["J_ETA"]
assert (not jeta["interior"]) and jeta["min_omega"] == 0.0
re = {r["omega"]: r["E_total"] for r in jeta["rungs"]}
assert re[0.0] <= min(re.values()) + 1e-12

# J0 control: no interior well, minimum at zero
j0 = lad["J0"]
assert (not j0["interior"]) and j0["min_omega"] == 0.0

# gamma scaling: same minimum position at every protocol level,
# depth ratio near 4
assert gsc["interior"] and gsc["min_omega"] == 0.35
assert 3.0 < gsc["depth_ratio"] < 5.0
for lv in range(3):
    k = min(range(len(gsc["rows"])),
            key=lambda i: gsc["rows"][i]["E_levels"][lv])
    assert gsc["rows"][k]["omega"] == 0.35, "4-gamma bracket unstable"

# 16x probe: the documented regime BREAK -- the minimum is unstable
# across protocol levels (frozen-profile prediction fails at 80%
# statics deformation); this bounds the gamma budget
assert len(set(g16["min_omega_per_level"])) > 1, \
    "expected level instability at 16x"
assert g16["final_min_omega"] > 0.35

# J2 diagnosis: the intensive term's E_extra reaches ~0 inside the range
j2min = min(r["E_extra"] for r in lad["J2_intensive"]["rungs"])
assert abs(j2min) < 1e-4

# figures exist and are non-trivial
for f in ("fig_i1sq_ladders.png", "fig_mechanism.png"):
    assert os.path.getsize(os.path.join(R, f)) > 10000, f

ran = all(os.path.exists(os.path.join(R, f))
          for f in ("i1sq_ran.flag", "gamma_scaling_ran.flag",
                    "gamma16_ran.flag"))
if ran:
    print("REPRODUCED: lattice producers ran here and all structural "
          "checks pass.")
else:
    print("STRUCTURAL CHECKS PASS on committed artifacts; lattice "
          "producers did NOT run here (004 fields absent) -- see "
          "NOT-REPRODUCED notices above.")
EOF
