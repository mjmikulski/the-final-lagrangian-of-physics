#!/usr/bin/env bash
# APPENDIX-walls-kappa: default (seconds) redraws the figure and checks the records; M5_RUN=1 regenerates them on a GPU
# (hours per part; at most two parts side by side on a 12 GB card).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
if [ "${M5_RUN:-0}" = "1" ]; then
  for part in seeds kick continue16 kick16kappa kappa split; do $PY strand_walls.py $part; done
fi
[ "${M5_SKIP_SADDLE:-0}" = "1" ] || $PY strand_saddle.py      # the saddle witness (CPU, a few minutes)
$PY make_appendix_walls_figure.py
$PY verify_appendix_walls.py
echo "APPENDIX WALLS-KAPPA CHECKS OK"
