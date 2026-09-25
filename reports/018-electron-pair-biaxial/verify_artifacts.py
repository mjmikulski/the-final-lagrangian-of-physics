"""Assertions on the committed records of report 018 (structure and the conclusions, not floating-point tails)."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')

er = json.load(open(os.path.join(R, 'electron_record.json')))
for k in ('n32_box12', 'n48_box18', 'n64_box24', 'n64_box16'):
    r = er[k]
    assert all(abs(abs(v) - 1) < 0.01 for v in r['degree'].values()), 'unit charge on every sphere'
    assert 0.95 < r['virial_with_tail'] < 1.06
boxes = [er[k]['E_box'] for k in ('n32_box12', 'n48_box18', 'n64_box24')]
assert boxes[0] < boxes[1] < boxes[2], 'box energy grows with the box'
ex = er['extrapolation']
assert 33.2 < ex['linear_in_1overR'] + ex['spacing_correction_at_R8'] < 33.7, 'frozen mass 33.4 +- 0.3'

b = json.load(open(os.path.join(R, 'strand_bogomolny.json')))
assert b['density_identity'], 'planar density 64 k^2 (b b\'/rho)^2 checked symbolically'
rows = json.load(open(os.path.join(R, 'strand2d.json')))
for r in rows:
    assert 1.0 - 1e-4 < r['T_over_bound'] < 1.003, ('lattice attains the Bogomolny bound', r['beta'], r['k'])
    assert abs(r['r_half_split'] / r['r_half_bogomolny'] - 1) < 0.03, 'core radius on the Bogomolny value'
    assert r['e1_z_at_line'] > 0.999, 'no escape of the charge direction'
betas = sorted({r['beta'] for r in rows})
assert betas == [0.01, 0.03, 0.1, 0.3] and {r['k'] for r in rows} == {0.5, 1.0}

s = json.load(open(os.path.join(R, 'single_strands.json')))
for rad, sp in s['spheres'].items():
    assert sp['n_minima'] == 4, ('four half-disclination piercings', rad)
    assert all(0.4 < m['rho_from_axis'] < 0.9 for m in sp['minima'])

pairs = sorted(glob.glob(os.path.join(R, 'pair_flow_b*.json')))
assert len(pairs) == 10, 'uniaxial d = 4 and biaxial beta = 0.3, 0.1 at d = 4, 6, 8; three runs from the resolved seed'
assert sum(json.load(open(f)).get('melt_centre', False) for f in pairs) == 3
cc = json.load(open(os.path.join(R, 'central_check.json')))
for r in cc:
    e = [r['energies'][k] for k in ('8', '16', '32')]
    if r['melt_centre'] and r['beta'] == 0:
        assert e[0] > e[1] > e[2], 'uniaxial resolved seed: midpoint energy falling with h'
    elif r['melt_centre']:
        pass                                   # biaxial: only partly resolved (transverse splitting not melted)
    else:
        assert e[1] > 1.8 * e[0] and e[2] > 1.8 * e[1], 'electrostatic seed: midpoint energy grows like 1/h'
for f in pairs:
    tr = json.load(open(f))['trace']
    assert all(tr[i + 1]['E'] <= tr[i]['E'] + 1e-9 for i in range(len(tr) - 1)), ('energy never rises', f)
    end = tr[-1]['degrees']
    assert abs(end[0][0]) < 0.05 and abs(end[1][0]) < 0.05, ('the held charges melt', f)
print('verify_artifacts: all assertions pass')
