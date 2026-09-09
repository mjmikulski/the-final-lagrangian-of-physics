"""Assertions on the committed records of report 015 (structure, not floating-point tails)."""
import json

p4 = json.load(open('results/prop4_check.json'))
assert p4['F_at_m_Delta_max_entry'] == '0' and p4['trace_dNdN_at_m_Delta_max'] == '0' and p4['P_squared_zero']
assert p4['farfield_at_screening_rapidity'] == '0'
assert abs(p4['eigenvalue_shift_from_vacuum'][1]) < 0.02, 'eigenvalues move by O(Delta^2/E_0) only'
npb = json.load(open('results/null_probe.json'))
t = npb['table']
blind = ['quartic F.F (the model)', 'quadratic, eta norm  tr(d_iN d_iN)', 'tr(d_iN d_jN) tr(d_iN d_jN)', 'eigenvalue gradients sum (d_i e_a)^2']
assert all(abs(t[k]['screened']) < 1e-8 for k in blind), 'invariants of the spectrum and of the jet are blind'
assert t['time-axis tilt K_u = |Q dN P_0|^2']['frozen'] == 0 and t['time-axis tilt K_u = |Q dN P_0|^2']['screened'] > 1, 'K_u sees the tilt'
assert all(abs(b['c_star'] / b['prediction'] - 1) < 0.02 for b in npb['breakeven']), 'break-even c* = 2 Delta^2 / r^2'
ts = json.load(open('results/time_sector_check.json'))
for row in ts:
    assert row['grad_time'] < 1e-8, 'the frozen hedgehog is stationary in the time sector'
    assert abs(row['E_full'] - row['E_frozen']) < 1e-9
base = [r for r in ts if r['tilt'] == 0]
assert len({round(r['E_frozen'], 9) for r in base}) == 1, 'frozen energy independent of E_0'
assert all(r['rayleigh']['radial M_0i ~ x_i'] < 0 for r in base), 'radial tilt is unstable without the frame term'
assert all(v > 0 for r in base for k, v in r['rayleigh'].items() if not k.startswith('radial')), 'the other directions are stable'
e100 = sorted([r for r in ts if r['E0'] == 100.0], key=lambda r: r['tilt'])
ks = [r['rayleigh']['radial M_0i ~ x_i'] for r in e100]
assert ks[0] < 0 < ks[-1] and all(ks[i] < ks[i + 1] for i in range(len(ks) - 1)), 'stiffness grows with c and changes sign'
sect = json.load(open('results/tilt_sector.json'))
vac0 = [s for s in sect if s['tag'].startswith('VACUUM') and 'base' in s['tag']][0]
vac1 = [s for s in sect if s['tag'].startswith('VACUUM') and 'K_u' in s['tag']][0]
assert max(abs(v) for v in vac0['kinetic_eigenvalues']) < 1e-12, 'no linear dynamics on the vacuum without K_u'
assert vac1['dynamical'] == ['M01', 'M02', 'M03'] and all(abs(v - 1) < 1e-9 for v in vac1['squared_speeds']), 'three light-speed tilt modes'
assert not any(s['ghost'] for s in sect), 'no ghost anywhere'
assert all(max(s['squared_speeds']) < 1 + 1e-9 for s in sect if s['squared_speeds']), 'subluminal'
fd = {r['file']: r for r in json.load(open('results/field_diagnostics.json'))}
frozen = fd['static_base_n32.pt']
assert frozen['Ku_unit'] < 1e-30, 'K_u vanishes on the frozen hedgehog'
for e0 in (10, 100, 1000):
    r = fd[f'static_unfrozen_E0_{e0}.pt']
    assert r['E'] < frozen['E'] - 4 and r['tilt_max'] > 0.6, f'screening at E_0 = {e0}'
    assert r['profiles'][-1]['coulomb_ratio'] > 1.5, 'far field inflated by the box-forced return of the tilt'
cure = sorted([(r['tilt'], r['E']) for r in fd.values() if r['tilt'] > 0 and r['E0'] == 100.0])
assert all(cure[i][1] < cure[i + 1][1] for i in range(len(cure) - 1)), 'energy rises monotonically with c'
assert abs(cure[-1][1] - frozen['E']) < 1e-4, 'at the largest coupling the minimiser returns to the frozen hedgehog'
rad = json.load(open('results/radial_E0_100_nk480.json'))
assert abs(rad['frozen']['E'] - 33.3) < 0.3 and rad['screened']['E'] < 0.05 * rad['frozen']['E'], '1D: screened mass a few percent of the frozen one'
rad2 = json.load(open('results/radial_E0_100_nk240.json'))
assert abs(rad2['screened']['E'] - rad['screened']['E']) < 0.1, 'screened mass converged in the knots'
import glob, re
scan = sorted((float(re.search(r'tilt([0-9.]+)\.json', f).group(1)), json.load(open(f))['screened']['E']) for f in glob.glob('results/radial_E0_100_nk240_tilt*.json'))
assert all(scan[i][1] < scan[i + 1][1] for i in range(len(scan) - 1)), '1D: mass rises monotonically with c'
assert abs(scan[-1][1] - rad2['frozen']['E']) < 0.05, '1D: mass fully restored at the largest coupling'
assert [c for c, E in scan if abs(E - rad2['frozen']['E']) < 0.05][0] <= 0.3, '1D: restored by c = 0.3'
print('VERIFY 015 OK')
