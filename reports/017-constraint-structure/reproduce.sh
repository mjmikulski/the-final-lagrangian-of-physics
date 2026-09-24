#!/usr/bin/env bash
# Report 017: regenerate the committed records and assert their structure. CPU only, a few minutes.
# Reads the committed lattice fields of report 016 (../016-time-axis-screening/results/fields/).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY symbolic.py
$PY kinetic_rank.py
$PY trace_constraint.py
$PY radial_profile.py
$PY make_figures.py
$PY verify_artifacts.py
test -s results/fig_kinetic_rank.png && test -s results/fig_speeds.png
echo "ALL PASS"
