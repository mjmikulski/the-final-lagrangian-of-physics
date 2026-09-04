#!/usr/bin/env bash
# Report 014 reproduction. CPU (~15 min): enumeration (float + exact), the
# Euler-Lagrange null test, orbit checks, static kernels, artifact assertions,
# figures. GPU (hours): M5_RUN_LATTICE=1 reruns the two lattice routes (they
# resume from committed state; delete results/lattice_linear*.json for a
# fresh run).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
echo "== L0: enumeration, float route =="; $PY enumerate_linear.py
echo "== L0: exact route over Q =="; $PY exact_linear.py
echo "== L0: Euler-Lagrange (null) test =="; $PY null_test_linear.py
echo "== L1: rank-rich orbit values =="; $PY orbit_linear_exact.py
echo "== L1: rank-1 orbit theorem =="; $PY orbit1_linear_exact.py
echo "== L1: static kernel signatures =="; $PY static_kernel_signs.py
if [ "${M5_RUN_LATTICE:-0}" = "1" ]; then
  echo "== L2: lattice routes (GPU) =="; $PY lattice_linear_v2.py A; $PY lattice_linear_v2.py B
else
  echo "== L2 lattice legs SKIPPED (set M5_RUN_LATTICE=1) =="
fi
echo "== artifact assertions =="; $PY verify_artifacts.py
echo "== figures =="; $PY make_figures.py
echo "ALL PASS"
