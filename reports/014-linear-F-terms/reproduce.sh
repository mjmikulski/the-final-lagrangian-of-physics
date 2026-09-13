#!/usr/bin/env bash
# Report 014: regenerate the committed records and assert their structure.
# CPU (about 20 minutes): enumeration (float and exact), the exact classification, the Euler-Lagrange test,
# the orbit checks, the static kernels, and the E0 = 100 classification of the second route.
# GPU (hours): M5_RUN_LATTICE=1 reruns the lambda-scan on the electron, its restarts and the vacuum twist scan
# (they need report 010's lattice stack, ../010-fundamental-grid-clock), and the E0 = 100 scan of X = c X_a.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY enumerate_linear.py
$PY exact_linear.py
$PY exact_classification.py
$PY null_test_linear.py
$PY orbit_linear_exact.py
$PY orbit1_linear_exact.py
$PY static_kernel_signs.py
(cd e0_route && $PY candidates.py)
if [ "${M5_RUN_LATTICE:-0}" = "1" ]; then
  $PY lattice_linear_v2.py A
  $PY restart_check.py A
  $PY twist_scan_generic.py A
  (cd e0_route && $PY run_task4.py --cands "-1,0" "-0.5,0" "-0.3,0" "-0.1,0" "0.1,0" "0.3,0" "0.5,0" "1,0" --out results/task4_scan.json)
fi
$PY make_figures.py
$PY verify_artifacts.py
for f in fig_family fig_vacuum_saddle fig_electron_scan fig_e0_scan; do test -s results/$f.png; done
echo "ALL PASS"
