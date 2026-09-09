#!/usr/bin/env bash
# Report 016: regenerate the committed records and assert their structure.
# CPU suite (minutes for the symbolic and lattice-diagnostic legs; the radial problems take up to an hour each,
# run in parallel below). The lattice minimisations that produced the committed endpoint fields
# (results/fields/*.pt) need a GPU (about an hour each at n = 32); set M5_RUN_GPU=1 to redo them.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${M5_RUN_GPU:+0}
$PY prop4_check.py
$PY null_probe.py
$PY tilt_sector.py
if [ -n "$M5_RUN_GPU" ]; then
  $PY unfrozen_test.py 100 0.05 results/fields/static_base_n32.pt E0_100
  $PY unfrozen_test.py 10 0.05 results/fields/static_base_n32.pt E0_10
  $PY unfrozen_test.py 1000 0.05 results/fields/static_base_n32.pt E0_1000
  for c in 0.03 0.1 0.3 3.0; do
    $PY run_static.py --n 32 --box 12 --unfreeze --tilt $c --init results/fields/static_unfrozen_E0_100.pt --iters 1200 --rounds 1 --tag tilt$c
  done
fi
$PY field_diagnostics.py
$PY time_sector_check.py --E0 100 --tilt 0 0.001 0.01 0.03 0.1
$PY time_sector_check.py --E0 10 3 --tilt 0 --out results/time_sector_check_E0.json
if [ -z "$M5_SKIP_1D" ]; then
  $PY radial_screened.py --spline 240 --E0 100 &
  $PY radial_screened.py --spline 480 --E0 100 --which screened &
  for c in 0.003 0.01 0.03 0.1 0.3 1 3; do $PY radial_screened.py --spline 240 --E0 100 --tilt $c --which screened & done
  wait
fi
$PY make_figures.py
$PY verify_artifacts.py
test -s results/fig_screening.png && test -s results/fig_cure.png
echo "ALL PASS"
