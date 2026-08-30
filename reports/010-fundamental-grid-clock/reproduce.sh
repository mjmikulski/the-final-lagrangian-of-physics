#!/usr/bin/env bash
# Report 010 reproduction. CPU part (~10 min): exact structural suites -- each
# script asserts its claims and regenerates its results JSON. GPU part
# (optional, hours): set M5_RUN_LATTICE=1 with a CUDA device; regenerates the
# base profile against the 004 oracle and reruns the lattice runs. With
# M5_FRESH=1 the cached base profile and the resumable ladder state are
# removed ONCE up front; pre_e4.py then regenerates the base a single time
# and every later stage loads that shared result (review round 3). Without
# it interrupted runs resume from the committed state.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

echo "== E0: grid Legendre theorem (sympy, exact) =="
$PY legendre_theorem_check.py

echo "== E1: u-family enumeration (float route) =="
$PY enumerate_u_family.py
echo "== E1: exact route over Q =="
$PY verify_u_family_exact.py

echo "== E2: velocity splits, statics filter, leaks, channels =="
$PY velocity_split.py
echo "== E2: exact split checks =="
$PY exact_split_checks.py
echo "== E2: matrix-cap orbit zeros (exact) =="
$PY orbit_zeros_exact.py
echo "== T1': span-level static kernel (exact) =="
$PY static_kernel_exact.py

echo "== E3: reduced-family Legendre (sympy, exact) =="
$PY e3_reduced_legendre.py

if [ "${M5_RUN_LATTICE:-0}" = "1" ]; then
  if [ "${M5_FRESH:-0}" = "1" ]; then
    echo "== fresh run: removing cached base profile and ladder state =="
    rm -f results/M_eta_base.npz results/e4_cells.json
    unset M5_FRESH   # later stages share the one regenerated base
  fi
  echo "== E4: lattice runs (GPU; hours) =="
  $PY pre_e4.py
  $PY e4_ladders.py
  $PY e4_gamma_arm.py
  $PY e5_arms.py
  $PY e5_deep_bracket.py
  $PY gyro_family_lattice.py
else
  echo "== E4 lattice legs SKIPPED (set M5_RUN_LATTICE=1 to rerun) =="
fi

echo "== artifact-only assertions of every lattice headline =="
$PY verify_artifacts.py

echo "== figures from committed JSONs =="
$PY make_figures.py

echo "ALL PASS"
