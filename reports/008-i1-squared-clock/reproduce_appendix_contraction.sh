#!/usr/bin/env bash
# APPENDIX-contraction-release: by default (CPU, seconds) checks the committed records of results/contraction_release/
# and redraws the figure; with M5_RUN=1 regenerates them on a GPU (structure: seconds; frob: six rungs, about
# 20 min; release: four rungs with eight L-BFGS cycles, about 30 min, alone on an RTX 4070).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
if [ "${M5_RUN:-0}" = "1" ]; then
  $PY appendix_contraction_release.py all
fi
$PY make_appendix_contraction_figure.py
$PY verify_appendix_contraction.py
$PY verify_appendix_contraction_route2.py    # the independent numpy route on the persisted fields (CPU, about a minute)
echo "APPENDIX CONTRACTION-RELEASE CHECKS OK"
