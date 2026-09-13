#!/usr/bin/env bash
# APPENDIX-volume-weights: regenerates the appendix record (CPU, about 25 minutes; the vacuum script
# dominates). Separate from reproduce.sh so the merged report's path stays untouched (METHOD section 7).
# Needs the E0 = 100 snapshot in e0_route/ and the committed fields of report 016.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY appendix_volume_identities.py
$PY appendix_volume_el_test.py
$PY appendix_volume_lattice_phi.py
$PY appendix_volume_vacuum.py
$PY - <<'PYEOF'
import json
R = 'results/'
idn = json.load(open(R + 'appendix_volume_identities.json'))
el = json.load(open(R + 'appendix_volume_el_test.json'))
lp = json.load(open(R + 'appendix_volume_lattice_phi.json'))
vc = json.load(open(R + 'appendix_volume_vacuum.json'))
# identities exact, decomposition to round-off, vacuum values
assert all(idn['identities'][k] for k in ('newton_diag', 'newton_general_etaM', 'cayley_hamilton_adj', 'tr_adj_is_e3'))
assert idn['identities']['vacuum']['e3'] == '1' and idn['identities']['vacuum']['det'] == '0'
assert idn['decomposition']['worst_relative'] < 1e-10
# the weight on the electron: positive and core-enhanced in the frozen sector, negative somewhere with the time sector free
f = idn['lattice']['frozen_static_base_n32']
assert f['e3_over_vac_min'] > 0.99 and f['e3_over_vac_max'] > 10 and f['fraction_e3_nonpositive'] == 0
assert idn['lattice']['unfrozen_E0_100_report016']['fraction_e3_nonpositive'] > 0
# Euler-Lagrange structure
g, p = el['generic'], el['pure_frame_constant_spectrum']
assert g['phi']['null'] and not g['I1']['null'] and not g['L1']['null'] and not g['L2']['null']
assert p['L1']['frame_projection'] < 1e-8 and p['L1']['eigenvalue_projection'] > 1e-3
assert p['L2']['frame_projection'] > 1e-3
# electron: phi is a boundary term with a definite sign, the weighted integral is a sizeable fraction of E_stat
assert abs(lp['int_phi']) == lp['int_abs_phi'] and abs(lp['X1_signed']) > 5 * lp['E_stat']
# vacuum: phi integrates to zero on every twist, L1 as well at small amplitude, L2 has a converged cubic term
for row in vc['A_twists']['grid_convergence']:
    assert abs(row['I_phi']) < 1e-6 * row['phi_abs'], row
fits = vc['A_twists']['cubic_fit']
for s in range(3):
    c32 = [r for r in fits if r['seed'] == s and r['n'] == 32][0]
    c48 = [r for r in fits if r['seed'] == s and r['n'] == 48][0]
    assert c32['max_abs_I_L1'] < 1e-8 and c48['max_abs_I_L1'] < 1e-8
    assert abs(c32['c3'] - c48['c3']) < 1e-2 * abs(c48['c3']) and abs(c48['c3']) > 1e3
    assert abs(c48['c2']) < 1e-6 * abs(c48['c3'])
# threshold: the Cauchy-Schwarz bound is attained, kappa_c of order 1e-2 in both conventions
assert abs(vc['B_threshold']['rho_min'] - 1 / 6) < 1e-4
for lab in ('E3=0', 'E3=E2 (lattice)'):
    b = vc['B_threshold'][lab]
    assert 0.003 < b['kappa_c'] < 0.1, b
    assert abs(b['kappa_c'] - b['kappa_lin']) < 0.5 * b['kappa_lin'], b
print('APPENDIX REPRODUCTION OK')
PYEOF
