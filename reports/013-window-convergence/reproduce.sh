#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -f results/deep_ran.flag

$PY continue_window.py
$PY analysis.py
$PY make_figures.py

$PY - <<'EOF'
import json
import os

R = "results"
W = json.load(open(os.path.join(R, "window_deep.json")))
V = json.load(open(os.path.join(R, "verdicts.json")))

for tag in ("x14_continued", "x10_fresh"):
    v = W["arms"][tag]["verdict"]
    assert v["min_omega"] == 0.28 and not v["interior"], tag
    assert not v["stopped_on_observable"], tag

a14 = V["arms"]["x14_continued"]
assert a14["final_diffs"]["0.1"] < 0
assert a14["final_diffs"]["0.2"] < 0
assert a14["final_diffs"]["0.28"] < -3e-3
assert a14["drift_last5"]["0.28"] < -5e-5, "still deepening at the end"

a10 = V["arms"]["x10_fresh"]
assert a10["final_diffs"]["0.1"] > 0 and a10["final_diffs"]["0.2"] > 0
assert a10["final_diffs"]["0.28"] < 0, "the wide bracket cracks"

assert os.path.getsize(os.path.join(R, "fig_drift.png")) > 10000
ran = os.path.exists(os.path.join(R, "deep_ran.flag"))
print("REPRODUCED" if ran else "STRUCTURAL CHECKS PASS on committed "
      "artifacts (010 stack absent)")
EOF
