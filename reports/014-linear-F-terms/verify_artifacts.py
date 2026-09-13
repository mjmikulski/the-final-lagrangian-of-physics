"""Assertions on the committed records of report 014 (structure, not floating-point tails)."""
import json
import os

def load(n, d='results'):
    with open(os.path.join(d, n + '.json')) as f:
        return json.load(f)

fl = load('linear_float'); ex = load('linear_exact'); ec = load('exact_classification'); nt = load('null_test')
ob = load('orbit_linear'); o1 = load('orbit1_linear'); ks = load('static_kernel_signs')
assert (fl['n_diagrams'], fl['n_classes']) == (675, 38) and (fl['rank_all'], fl['rank_even'], fl['rank_odd']) == (12, 6, 6)
assert (ex['rank_all'], ex['rank_even'], ex['rank_odd']) == (12, 6, 6)
assert (ec['n_diagrams'], ec['n_zero'], ec['n_classes'], ec['rank_all'], ec['rank_even'], ec['rank_odd']) == (675, 424, 38, 12, 6, 6)
assert fl['rank_spatial_3x3'] == 3 and fl['rank_static_generic'] == 12
alive = [k for k, v in fl['reps'].items() if v['spatial_3x3'] == 'alive']
assert len(alive) == 7 and all('P0' not in fl['reps'][k]['label'] and 'eps' not in fl['reps'][k]['label'] for k in alive)
assert nt['CONTROL_phi']['null'] and nt['L11']['null'] and nt['L0']['null'] and not nt['CONTROL_I1']['null']
dec = [k for k in nt if k.startswith('L') and k not in ('L0', 'L11')]
assert len(dec) == 36 and all(not nt[k]['null'] for k in dec), 'all 36 decorated classes dynamical'
assert ob['orbit_zero'] == [] and len(ob['orbit_nonzero']) == 38
assert o1['even_with_Pt_cap_all_zero'] is True and len(o1['nonzero_odd']) > 0
assert all(abs(v['trace']) < 1e-9 and v['n_pos'] > 0 and v['n_neg'] > 0 for v in ks.values())
print('family: 675 diagrams, 38 classes, 12 = 6 + 6 exactly; phi, chi null; 36 decorated classes dynamical; rank-1 orbit even sector inert; 3x3 kernels traceless indefinite')

# the electron scan (route A, vacuum-pinned projectors)
lt = load('lattice_linear_A'); v = lt['validation']
assert all(f['rel_err'] < 1e-5 for f in v['directional_fd'].values()) and v['lattice_EL_ratio_phi_over_dyn'] < 1e-12
runs = {k: r for k, r in lt.items() if isinstance(r, dict) and 'status' in r}
base = runs['baseline']
assert len(runs) == 13 and all(r['status'] == 'ok' for r in runs.values())
for k, r in runs.items():
    if k == 'baseline':
        continue
    frozen = r['lambda'] * lt['base_integrals'][k.split('_')[0]]
    shift = r['E_total'] - base['E_total']
    assert shift * frozen > 0 and abs(shift / frozen - 1) < 0.3, (k, shift, frozen)      # sign-weighted, within 30% of the frozen value
    assert 0.001 < r['small_gap_min'] < 0.01                                             # the core's exchange gap survives
    assert abs(r.get('continuation_dE', 0.0)) < 0.01 * abs(shift)
rc = load('restart_check_A')
assert all(abs(x['dE_vs_main']) < 0.01 * abs(x['E_main'] - base['E_total']) for x in rc.values()), 'restarts re-land within 1% of the effect'
print(f'electron scan: {len(runs)} runs, shifts sign-weighted and within 30% of the frozen values, gaps 1e-3..1e-2, restarts within 1%')

# the vacuum: cubic saddle
tg = load('twist_generic_A')
ts = [t for t in next(iter(tg.values())) if t.startswith('0.')]
for k, vv in tg.items():
    c3 = [vv[t]['c3_lin'] for t in ts]
    assert min(abs(x) for x in c3) > 1e-3 and max(abs(c3[0] - x) for x in c3) < 1e-3 * abs(c3[0]), (k, c3)   # nonzero and constant in t
    assert 50 < vv[ts[0]]['c4_eta'] < 80
for s in sorted({k.split('seed')[1] for k in tg}):
    c3 = {c: tg[f'{c}_seed{s}']['0.001']['c3_lin'] for c in ('P1P2', 'P1P3', 'P2P3')}
    assert abs(sum(c3.values())) < 1e-3 * max(abs(x) for x in c3.values()), ('null combination', s, c3)
print('vacuum: cubic coefficient nonzero for every class and twist, constant over t, cancelling in the null combination')

# second route, E0 = 100 conventions
cd = load('candidates', 'e0_route/results')
assert cd['decomposition_residual_cross'] < 1e-8 and cd['decomposition_residual_eps'] < 1e-8, '12 frame structures span every candidate'
assert cd['n_null'] == 2 and cd['n_nontrivial'] > 0
sc = sorted(load('task4_scan', 'e0_route/results'), key=lambda r: r['ca'])
good = [r for r in sc if abs(r['ca']) <= 0.5]
assert all(r['grad'] < 1e-4 and r['rest_energy'] and r['charge'] for r in good), 'static tests pass for |c| <= 0.5'
assert all(good[i]['E'] > good[i + 1]['E'] for i in range(len(good) - 1)), 'mass decreases monotonically with c'
assert all(0.8 < x < 1.2 for r in good for x in r['tail_ratio']), 'far field within 20% of the Coulomb law'
assert [r for r in sc if r['ca'] == 1.0][0]['E'] < 0 and [r for r in sc if r['ca'] == -1.0][0]['grad'] > 1e-3
print('E0 = 100 route: every candidate in the span of the 12 frame structures; X = c X_a scan: static tests pass for |c| <= 0.5, mass monotone, c = +1 negative energy, c = -1 not converged')
print('VERIFY 014 OK')
