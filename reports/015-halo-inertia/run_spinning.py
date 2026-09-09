"""Rigidly rotating stationary states: minimise H_J = (J - b)^2 / (4a) - c at fixed angular momentum J.

python run_spinning.py --static results/static_base_n32.pt --J 0.5 1 2 4 8 --tag base_n32
Each point is warm-started from the previous one; the first from the static solution plus a small
non-symmetric perturbation (a rotation-invariant field has a = 0 and infinite H_J).
"""
import argparse
import torch
from soliton import Grid


def load(path):
    d = torch.load(path)
    grid = Grid(d['n'], d['box'], E=d['E'], split=d['split'], freeze_time=d['frozen'], X=d.get('X'), degree=d.get('degree', 1))
    return grid, d['u'].cuda(), d

torch.set_default_dtype(torch.float64)
ap = argparse.ArgumentParser()
ap.add_argument('--static', required=True)
ap.add_argument('--J', type=float, nargs='+', required=True)
ap.add_argument('--iters', type=int, default=1500)
ap.add_argument('--rounds', type=int, default=2)
ap.add_argument('--tag', required=True)
ap.add_argument('--init', default=None, help='warm start from the last point of a saved branch')
ap.add_argument('--pin', type=float, default=0.0, help='penalty strength pinning the energy centres to the axis (0 = free)')
a = ap.parse_args()

grid, u, d = load(a.static)
gen = torch.Generator(device='cuda').manual_seed(0)
u = u + 0.03 * torch.randn(u.shape, generator=gen, device='cuda') * grid.mask * torch.exp(-grid.x.norm(dim=-1) ** 2 / 4)[..., None]
branch = []
if a.init:
    prev = torch.load(a.init)
    branch = prev['branch']
    u = branch[-1]['u'].cuda()
for J in a.J:
    for r in range(a.rounds):
        obj = (lambda v: grid.energy_at_J_pinned(v, J, a.pin)) if a.pin else (lambda v: grid.energy_at_J(v, J))
        u, gnorm = grid.minimize(u, obj, iters=a.iters, verbose=True)
    with torch.no_grad():
        A, B, C = (float(t) for t in grid.rotor(u))
    om = (J - B) / (2 * A)
    pt = {'J': J, 'a': A, 'b': B, 'Estatic': -C, 'E': (J - B) ** 2 / (4 * A) - C, 'Omega': om, 'grad': gnorm, 'pin': a.pin}
    with torch.no_grad():
        pt['centres'] = [c.cpu().tolist() for c in grid.centres(u)]
    pt['clock'] = pt['E'] - 2 * J * om
    print(f'J={J:.3f}: E={pt["E"]:.5f} Estatic={-C:.5f} Omega={om:.5f} a={A:.5f} b={B:.2e} E-2J*Omega={pt["clock"]:.5f} |grad|={gnorm:.1e} centres={[[round(x, 3) for x in c] for c in pt["centres"]]}', flush=True)
    pt['u'] = u.cpu()
    branch.append(pt)
    torch.save({'branch': branch, 'static': a.static}, f'results/fields/spinning_{a.tag}.pt')
