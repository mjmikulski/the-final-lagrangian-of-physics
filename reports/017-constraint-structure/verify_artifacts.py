"""Assertions on the committed records of report 017 (structure: ranks, kernels, bounds, identities)."""
import json

s = json.load(open('results/symbolic.json'))
assert s['velocity_powers'] == [0, 2], 'L is exactly quadratic in the velocity'
assert s['F_blind_to_trace'] and s['K_u_blind_to_trace'] and s['symbol_identity']

k = json.load(open('results/kinetic_rank.json'))
assert k['vacuum']['c = 0']['rank'] == 0, 'no kinetic term on the vacuum'
v1 = k['vacuum']['c = 1']
assert v1['rank'] == 3 and v1['dynamical_components'] == ['M01', 'M02', 'M03'] and v1['speed2'] == [1.0]
expected = {('random', 'generic'): 9, ('random', 'frozen'): 8,
            ('lattice', 'frozen hedgehog'): 8, ('lattice', 'time sector free (E0 = 100)'): 9,
            ('lattice', 'K_u, c = 0.1'): 9, ('lattice', 'K_u, c = 3'): 8}
for (grp, name), rank in expected.items():
    r = k[grp][name]
    assert r['rank_counts'] == {str(rank): r['points']}, (name, r['rank_counts'])
    assert r['kernel_contains_trace_min'] > 1 - 1e-9, 'the trace is always in the kernel'
    if rank == 8:
        assert r['kernel_contains_trace_and_P0_min'] > 1 - 1e-9, 'frozen locus: kernel = trace + P_0'
    assert -1e-9 < r['speed2_min'] and r['speed2_max'] < 1 + 1e-9, 'squared speeds in [0, 1]'
    assert r['symbol_on_kernel_max'] < 1e-7, 'kernel directions carry no principal symbol'
    assert r['eta_norm_points_with_negative_K'] == r['points'], 'the eta norm makes K indefinite'
    assert r['route1_route2_max_relative_deviation'] < 1e-12, 'autograd and closed form agree'

sp = k['special']
for name in ('boost twist', 'rotation twist'):
    assert sp[name]['c = 0']['rank'] == 5, 'one-direction twists have rank 5'
assert sp['rotation twist']['M03_kinetic_c0'] == 0.0, 'a time-space component without kinetic term'
for c in ('c = 0', 'c = 1'):
    h = sp['uniaxial hedgehog, z axis, k = x'][c]
    assert h['rank'] == 8 and h['K_XX'] > 1 and h['G_X_norm'] < 1e-12 and abs(h['speed2_min']) < 1e-12, 'exact zero speed'

t = json.load(open('results/trace_constraint.json'))
for name, r in t.items():
    if name.startswith('rotor'):
        continue
    assert r['quartic_change_under_trace_shift'] < 1e-13 and (r['K_u_change_under_trace_shift'] or 0) < 1e-13
    assert r['potential_change_under_trace_shift'] > 0.1
    assert r['trace_residual_max'] < 1e-7 and r['e1_max'] - r['e1_min'] > 0.5, name
rot = t['rotor, frozen hedgehog']
assert all(rot[w]['b'] == 0.0 for w in rot), 'no term linear in the angular velocity'
assert rot['combined']['a'] < 0.02 * rot['internal']['a'], 'combined rotation is (up to the lattice) a symmetry'
assert abs(rot['spatial']['a'] / rot['internal']['a'] - 1) < 0.01
print('verify_artifacts: all assertions pass')
