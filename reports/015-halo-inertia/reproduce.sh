#!/usr/bin/env bash
# Report 015: regenerate the committed records and assert their structure.
# CPU only (minutes): the symbolic halo formulas, the exterior reduction, the tilt measurement on the committed
# hedgehog fields, and the re-evaluation of the committed rotating branch. Finding the rotating states
# themselves needs a GPU (run_spinning.py, hours); set M5_RUN_GPU=1 to redo it.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${M5_RUN_GPU:+0}
$PY halo_check.py
$PY exterior_check.py
$PY halo_inertia_check.py
$PY halo_scaling.py --static results/fields/static_base_n48.pt --tag n48_box18
$PY halo_scaling.py --static results/fields/static_base_n32.pt --tag n32_box12
if [ -n "$M5_RUN_GPU" ]; then
  $PY run_spinning.py --static results/fields/static_base_n32.pt --J 1 4 10 18 --tag base_n32
fi
$PY rotor_branch.py
$PY make_figures.py
$PY verify_artifacts.py
test -s results/fig_halo.png && test -s results/fig_rotor.png
echo "ALL PASS"
