"""Assertions on the committed records of report 015 (structure, not floating-point tails)."""
import json
import math

hc = json.load(open('results/halo_check.json'))
assert hc['cost_is_32pi_over_3'] and hc['inertia_leading_is_64pi_over_3'] and hc['inertia_closed_form_verified'], 'exact halo formulas'
hi = json.load(open('results/halo_inertia_check.json'))
assert hi['worst_relative_deviation'] < 1e-6, 'the model kinetic density on the analytic exterior matches the closed form'
placed = {p['profile'][:12]: p['exact_over_leading'] for p in hi['profiles']}
far = [p['exact_over_leading'] for p in hi['profiles'] if '(r-100)' in p['profile']][0]
near = [p['exact_over_leading'] for p in hi['profiles'] if '(r-3)' in p['profile']][0]
assert near < 1.5 and far > 5, 'the radial kinetic term grows with the placement radius'
ex = json.load(open('results/exterior_check.json'))
assert ex['worst_relative_deviation'] < 1e-10, 'exterior reduces to the Faddeev-Skyrme quartic'
for tag in ('n48_box18', 'n32_box12'):
    d = json.load(open(f'results/halo_scaling_{tag}.json'))
    # every tilt costs energy and adds inertia; both quadratic in the angle; measured slopes within 40% of the exterior formula
    assert all(r['cost'] > 0 and r['inertia_excess'] > 0 for r in d['rows']), tag
    assert abs(d['exponent_delta_cost'] - 2) < 0.15 and abs(d['exponent_delta_inertia'] - 2) < 0.15, tag
    assert 0.6 < d['ratio_cost'] < 1.4 and 0.6 < d['ratio_inertia_exact'] < 1.4, tag
    # the halo mechanism: at fixed angle the cost falls with the twist-zone thickness, the inertia grows with the tilted length
    small = sorted([r for r in d['rows'] if r['delta'] == min(x['delta'] for x in d['rows'])], key=lambda r: r['L'])
    assert small[0]['cost'] > small[-1]['cost'] and small[0]['inertia_excess'] > small[-1]['inertia_excess'], tag
rb = json.load(open('results/rotor_branch.json'))
rows = sorted(rb['rows'], key=lambda r: r['J'])
assert all(r['Omega'] > 0 and r['E_kin'] > 0 and r['E_def'] > 0 for r in rows)
assert all(rows[i]['Omega'] > rows[i + 1]['Omega'] for i in range(len(rows) - 1)), 'frequency falls with J'
assert all(r['inertia_shape'] > 5 * rb['rigid_inertia'] for r in rows), 'inertia far above the rigid value'
assert all(r['E_over_Omega'] > 10 * 2 * r['J'] for r in rows), 'clock condition E = 2 J Omega never approached'
assert all(rows[i]['E_over_Omega'] < rows[i + 1]['E_over_Omega'] for i in range(len(rows) - 1)), 'E/Omega moves away from 2J'
print('VERIFY 015 OK')
