#!/usr/bin/env bash
# APPENDIX-L-ladder: by default (CPU, seconds) checks the committed records of results/L_ladder/ and redraws the
# figure; with M5_RUN_LADDER=1 regenerates everything: the 2b seeds (CPU, numpy: 20 min / 1 h / 4 h for
# N = 32 / 48 / 64, run in parallel), then the chains and ladders on the GPU (about 1 h / 6 h / 20 h).
# Separate from reproduce.sh so the merged report's path stays untouched (METHOD section 7).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
if [ "${M5_RUN_LADDER:-0}" = "1" ]; then
  for n in 32 48 64; do
    L=$($PY -c "print($n*1.5)")
    (cd instrument && OMP_NUM_THREADS=8 $PY m5_21_2b_a_instrument.py relax seed=A term=T2 stencil=sym eps=0 n=$n L=$L delta=0.3 bc=pinned maxit=8000 w2=0.0027581) &
  done
  wait
  $PY appendix_L_ladder.py all --fresh      # every stage regenerated; without --fresh the committed records are resumed
fi
# the 64^3 fields (13 MB each) and the 48^3 bracket rungs (5 MB each) live in the repository release
# appendix-008-L-ladder-fields; fetch if absent
for f in M_G_polished_N64 jge_N64_om00 jge_N64_om01 jge_N64_om02 jge_N64_om035 jge_N48_om00 jge_N48_om01 jge_N48_om02 jge_N48_om035; do
  if [ ! -f results/L_ladder/$f.npz ]; then
    gh release download appendix-008-L-ladder-fields --repo mjmikulski/the-final-lagrangian-of-physics --pattern "$f.npz" --dir results/L_ladder \
      || curl -sSL -o results/L_ladder/$f.npz "https://github.com/mjmikulski/the-final-lagrangian-of-physics/releases/download/appendix-008-L-ladder-fields/$f.npz"
  fi
done
$PY verify_L_ladder_energies.py --strict    # the independent numpy route on the persisted bracket of every box
$PY make_appendix_L_figure.py
$PY verify_L_ladder.py
echo "APPENDIX L-LADDER CHECKS OK"
