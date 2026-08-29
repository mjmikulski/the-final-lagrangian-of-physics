#!/usr/bin/env bash
# Reproduction pipeline for report 009 (revised). With report 004's
# fields (or M5_FIELDS_DIR) reruns all producers; without them the
# committed artifacts are checked. Sentinels distinguish a fresh run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -f results/rot_ran.flag results/deep_ran.flag \
      results/fixedj_ran.flag results/scaling_ran.flag \
      results/texture_ran.flag

$PY ladder_rot.py
$PY deep_converge.py
$PY fixedj_scan.py
$PY inertia_scaling.py
$PY combined_texture.py
$PY verify_energies.py
$PY make_figures.py

$PY - <<'EOF'
import json
import os

R = "results"
rot = json.load(open(os.path.join(R, "rot_ladders.json")))
dc = json.load(open(os.path.join(R, "deep_converge.json")))
fj = json.load(open(os.path.join(R, "fixedj.json")))

# ladder record: internal consistency of the fixed-depth run (a record,
# not a well claim -- see README)
jr = rot["JR_E"]
assert abs(jr["min_omega"] - round(rot["omega_pred_R"], 3)) < 1e-9
jr0 = rot["JR0"]
assert (not jr0["interior"]) and jr0["min_omega"] == 0.0

# deep run: the reordering that withdraws the well claim must be
# reproduced -- 0.217 converged, the others still descending below it
E = dc["converged_energies"]
assert E["0.304"] < E["0.152"] < E["0.217"] < E["0.0"]
assert dc["runs"]["0.217"]["cycles"] <= 2
for om in ("0.0", "0.152", "0.304"):
    assert dc["runs"][om]["last_change"] < -1e-7, om  # still creeping

# fixed-J: bounded Routhian with rigid scaling, and the
# vacuum-domination signature (extensive inertia, delocalized density)
rows = fj["rows"]
E0 = rows[0]["E_total"]
I0 = fj["I_0"]
top = rows[-1]
ratio = (top["E_total"] - E0) / (top["J"] ** 2 / (2 * I0))
assert 0.9 < ratio < 1.2, ratio
assert all(r["PR_kin"] > 300 for r in rows), "vacuum domination"
assert max(r["I"] for r in rows) / min(r["I"] for r in rows) < 1.05

# box-size scaling: the extensivity claim rests on the exponent
sc = json.load(open(os.path.join(R, "inertia_scaling.json")))
assert sc["exponent_I_pure"] > 2.5, "volume-law scaling expected"
assert sc["rows"][-1]["I_pure"] > 5 * sc["rows"][0]["I_pure"]

# texture test: equivariant analytic textures decay ~1/r, the working
# texture stays O(1) -- at BOTH delta values
tx = json.load(open(os.path.join(R, "combined_texture.json")))
for tag in ("uniaxial", "spherical_biax"):
    zp = tx[tag]["zeta_profile"]
    assert zp[-1] < 0.35 * zp[0], tag  # falls off
    assert zp[-1] < 0.15, tag
wz = tx["working"]["zeta_profile"]
assert min(wz) > 0.5 and wz[0] > 1.0, "working texture stays O(1)"
assert tx["working"]["I_comb"] > 3 * tx["spherical_biax"]["I_comb"]

for f in ("fig_rot_ladders.png", "fig_rot_channel.png",
          "fig_fixedj.png", "fig_scaling.png"):
    assert os.path.getsize(os.path.join(R, f)) > 10000, f

ran = all(os.path.exists(os.path.join(R, f))
          for f in ("rot_ran.flag", "deep_ran.flag", "fixedj_ran.flag",
                    "scaling_ran.flag", "texture_ran.flag"))
if ran:
    print("REPRODUCED: producers ran here and all structural checks "
          "pass.")
else:
    print("STRUCTURAL CHECKS PASS on committed artifacts; producers "
          "did NOT run here (004 fields absent).")
EOF
