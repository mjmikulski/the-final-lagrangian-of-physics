"""Task 4: scan X = c_a X_a + c_b X_b in the electron sector, static tests for every candidate.

python run_task4.py --cands "1,0" "-1,0" "0.3,0" "-0.3,0" "0,0.3" "0,-0.3" --n 32 --box 12 --init results/static_base_n32.pt
For each candidate: warm-started static minimisation, then existence, charge, virial (with E2) and the
symmetry report; results appended to results/task4_scan.json. Rotating branches for the survivors are
run separately (run_spinning.py --pin 100 on the saved static file).
"""
import argparse
import json
import math
import os
import torch
import physics_tests as T
from soliton import Grid, refine

torch.set_default_dtype(torch.float64)
ap = argparse.ArgumentParser()
ap.add_argument('--cands', nargs='+', required=True, help='pairs "c_a,c_b"')
ap.add_argument('--n', type=int, default=32)
ap.add_argument('--box', type=float, default=12.0)
ap.add_argument('--init', default='results/static_base_n32.pt')
ap.add_argument('--iters', type=int, default=1500)
ap.add_argument('--rounds', type=int, default=2)
ap.add_argument('--out', default='results/task4_scan.json')
a = ap.parse_args()

out = a.out
scan = json.load(open(out)) if os.path.exists(out) else []
src = torch.load(a.init)
E = src['E']
for cand in a.cands:
    ca, cb = (float(t) for t in cand.split(','))
    tag = f'xa{ca:g}_xb{cb:g}'
    grid = Grid(a.n, a.box, E=E, X=(ca, cb))
    u = refine(src['u'].cuda(), src['n'], src['box'], grid)
    for r in range(a.rounds):
        u, gnorm = grid.minimize(u, grid.energy, iters=a.iters, verbose=True)
    E4, EV, E2 = (float(t) for t in grid.energy_terms(u))
    d = {'u': u.cpu(), 'n': a.n, 'box': a.box, 'E': E, 'split': False, 'frozen': True, 'X': (ca, cb),
         'E4': E4, 'EV': EV, 'E2': E2, 'grad': gnorm}
    torch.save(d, f'results/static_{tag}.pt')
    sol = (grid, u, d)
    row = {'ca': ca, 'cb': cb, 'E4': E4, 'EV': EV, 'E2': E2, 'E': E4 + EV + E2, 'grad': gnorm,
           'tail_ratio': T.tail_ratio(grid, u), 'virial': (E4 + 16 * math.pi * (E[1] - E[2]) ** 4 / (a.box / 2) - E2) / (3 * EV),
           'centre_of_potential': grid.centres(u)[0].cpu().tolist()}
    print(f'candidate X = {ca:g} X_a + {cb:g} X_b: E4={E4:.3f} EV={EV:.3f} E2={E2:.3f} E={row["E"]:.3f} virial={row["virial"]:.3f} |grad|={gnorm:.1e}', flush=True)
    row['rest_energy'] = T.test_rest_energy(sol)
    row['charge'] = T.test_charge_quantization(sol)
    row['symmetry'] = T.symmetry_report(grid, u)
    print(f'   existence {row["rest_energy"]}, charge {row["charge"]}, symmetry {row["symmetry"]}', flush=True)
    scan = [s for s in scan if not (s['ca'] == ca and s['cb'] == cb)] + [row]
    json.dump(scan, open(out, 'w'), indent=1)
