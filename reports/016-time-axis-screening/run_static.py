"""Find the static hedgehog by energy minimisation and save it.

python run_static.py --n 32 --box 12 --iters 3000 --tag base_n32 [--init results/x.pt] [--split] [--unfreeze]
"""
import argparse
import torch
from soliton import Grid, refine

torch.set_default_dtype(torch.float64)
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=32)
ap.add_argument('--box', type=float, default=12.0)
ap.add_argument('--E0', type=float, default=100.0)
ap.add_argument('--E2', type=float, default=0.01)
ap.add_argument('--E3', type=float, default=0.01)
ap.add_argument('--split', action='store_true', help='E_2 != E_3 with the (v2, v3) frame in the boundary condition')
ap.add_argument('--unfreeze', action='store_true', help='let the time-like sector move (full 4x4)')
ap.add_argument('--iters', type=int, default=2000)
ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--init', default=None)
ap.add_argument('--tag', required=True)
ap.add_argument('--xa', type=float, default=0.0)
ap.add_argument('--xb', type=float, default=0.0)
ap.add_argument('--degree', type=int, default=1)
ap.add_argument('--sigma', type=float, default=0.0)
ap.add_argument('--frame', type=float, default=0.0)
ap.add_argument('--tilt', type=float, default=0.0, help='coefficient of the time-axis tilt term K_u (unfrozen sector)')
a = ap.parse_args()

E = (a.E0, 1.0, a.E2, a.E3)
X = {k: v for k, v in dict(xa=a.xa, xb=a.xb, sigma=a.sigma, frame=a.frame, tilt=a.tilt).items() if v} or None
grid = Grid(a.n, a.box, E=E, split=a.split, freeze_time=not a.unfreeze, X=X, degree=a.degree)
if a.init:
    src = torch.load(a.init)
    u = refine(src['u'].cuda(), src['n'], src['box'], grid)
else:
    u = grid.initial(width=1.0)
for r in range(a.rounds):
    u, gnorm = grid.minimize(u, grid.energy, iters=a.iters, verbose=True)
    k, v, x2 = grid.energy_terms(u)
    print(f'round {r}: E4={float(k):.6f} EV={float(v):.6f} E2={float(x2):.6f} E={float(k + v + x2):.6f} virial={float((k - x2) / (3 * v)):.5f} |grad|={gnorm:.2e}', flush=True)
    torch.save({'u': u.cpu(), 'n': a.n, 'box': a.box, 'E': E, 'split': a.split, 'frozen': not a.unfreeze, 'X': X, 'degree': a.degree,
                'E4': float(k), 'EV': float(v), 'E2': float(x2), 'grad': gnorm}, f'results/fields/static_{a.tag}.pt')
