#!/usr/bin/env bash
# Report 018. Default (CPU, minutes): the Bogomolny bound, the electron record if the release fields are present,
# the strand localisation if the single-charge field is present, the figures and the assertions.
# M5_RUN=1 also redoes the 2D strand scan (CPU, a few hours) and the 3D runs on a GPU (the single charge about
# 1.5 h, each pair flow 2-4 h; three run side by side on a 12 GB card).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
mkdir -p results/fields
for f in static_base_n48 static_base_n64_box24 static_base_n64_box16 single_biaxial_b0.3_n32; do
  if [ ! -f results/fields/$f.pt ]; then
    gh release download report-018-fields --repo mjmikulski/the-final-lagrangian-of-physics --pattern "$f.pt" --dir results/fields \
      || curl -sSLf -o results/fields/$f.pt "https://github.com/mjmikulski/the-final-lagrangian-of-physics/releases/download/report-018-fields/$f.pt" \
      || echo "release asset $f.pt not available"
  fi
done
$PY strand_bogomolny.py
if [ "${M5_RUN:-0}" = "1" ]; then
  $PY strand2d.py
  $PY single_biaxial.py --beta 0.3 --n 32 --box 12
  $PY pair_flow.py --beta 0 --d 4
  for b in 0.3 0.1; do for d in 4 6 8; do $PY pair_flow.py --beta $b --d $d; done; done
fi
[ -f results/fields/static_base_n48.pt ] && $PY electron_record.py
[ -f results/fields/single_biaxial_b0.3_n32.pt ] && $PY single_strands.py
$PY make_figures.py
$PY verify_artifacts.py
test -s results/fig_strand.png && test -s results/fig_charge.png && test -s results/fig_pairs.png
echo "ALL PASS"
