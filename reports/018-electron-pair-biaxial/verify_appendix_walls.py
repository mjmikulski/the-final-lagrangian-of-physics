"""Assertions on the records of APPENDIX-walls-kappa (structure and the stated conclusions)."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
RH = math.sqrt(2 * (4 * 0.5 * 0.15 / math.sqrt(2)) * (2 * math.log(2) - 1))
load = lambda n: json.load(open(os.path.join(R, n)))
div = lambda r: round(RH / r['h'])

seeds = load('strand_walls_seeds.json')
for r in seeds:
    if r['seed'] in ('smooth', 'unmelted'):
        assert abs(r['T_over_bound'] - 1) < 2e-3, ('smooth and unmelted seeds relax to T_B', r['seed'], div(r))
um = {div(r): r['T_start_over_bound'] for r in seeds if r['seed'] == 'unmelted' and r['k'] == 0.5}
assert 3.5 < um[16] / um[8] < 4.5, 'the unmelted line costs ~ 1/h^2 on the lattice'

kick = [r for r in load('strand_walls_kick.json') if r['kappa'] == 0]
k8 = [r for r in kick if div(r) == 8][0]
assert k8['T_over_bound'] < 0.35 and k8['trace'][-1] < k8['trace'][-2], 'h/8: below T_B and still falling'
c16 = load('strand_walls_continue16.json')[0]
assert c16['T_over_bound'] < 0.95 and c16['trace'][-1] < c16['trace'][-2], 'h/16: below T_B after continuation, still falling'

kk = [r for r in load('strand_walls_kick.json') if r['kappa'] > 0][0]
assert 0.8 < kk['T_over_bound'] < 0.9 and kk['grad_inf'] < 1e-5, 'kappa = 1e-2 at h/8: converged below T_B'

k16k = load('strand_walls_kick16kappa.json')[0]
assert abs(k16k['T_over_bound'] - kk['T_over_bound']) < 0.01 * kk['T_over_bound'], 'kappa state: the same at h/8 and h/16'

kap = load('strand_walls_kappa.json')
def added(k, kappa, Rm):
    a = [r for r in kap if r['k'] == k and r['kappa'] == kappa and abs(r['R'] / RH - Rm) < 1e-6][0]
    z = [r for r in kap if r['k'] == k and r['kappa'] == 0.0 and abs(r['R'] / RH - Rm) < 1e-6][0]
    return a['T'] - z['T']
pred = 16 * math.pi * 1e-3 * 0.25 * 0.15 ** 2
for lo, hi in ((4, 8), (8, 16)):
    slope = (added(0.5, 1e-3, hi) - added(0.5, 1e-3, lo)) / math.log(2)
    assert abs(slope / pred - 1) < 0.05, 'k = 1/2 winding cost 16 pi kappa k^2 b0^2 ln R'
r1 = (added(1.0, 1e-2, 16) - added(1.0, 1e-2, 8)) / (added(0.5, 1e-2, 16) - added(0.5, 1e-2, 8))
assert 1.5 < r1 < 3.0, 'k = 1 grows about twice as fast as k = 1/2, not four times (it splits)'
sp = load('strand_walls_split.json')[0]
assert sp['k'] == 1.0 and sp['kappa'] == 1e-2
print('verify_appendix_walls: all assertions pass')
