"""Perturb the frozen electron along M_0i ~ x_i f(r) and minimise the full 4x4 energy.

python unfrozen_test.py E0 eps [static_file] [tag]
"""
import sys, torch
from soliton import Grid, to_matrix, to_vector
torch.set_default_dtype(torch.float64)
E0 = float(sys.argv[1]); eps = float(sys.argv[2])
src = sys.argv[3] if len(sys.argv) > 3 else 'results/fields/static_base_n32.pt'
tag = sys.argv[4] if len(sys.argv) > 4 else f'E0_{E0:g}'
d = torch.load(src)
E = (E0, d['E'][1], d['E'][2], d['E'][3])
grid = Grid(d['n'], d['box'], E=E, freeze_time=False)
M = to_matrix(d['u'].cuda()); M[..., 0, 0] = E0; M[..., 0, 1:] = 0; M[..., 1:, 0] = 0
x = grid.x; f = torch.exp(-x.norm(dim=-1, keepdim=True) ** 2 / 8)
M[..., 0, 1:] += eps * f * x; M[..., 1:, 0] += eps * f * x
u = to_vector(M)
with torch.no_grad():
    print(f'E0 = {E0}: start energy {float(grid.energy(u)):.6f} (frozen {float(Grid(d["n"], d["box"], E=E).energy(to_vector(to_matrix(d["u"].cuda())))):.6f})', flush=True)
for r in range(3):
    u, g = grid.minimize(u, grid.energy, iters=400, verbose=False)
    with torch.no_grad():
        Mn = to_matrix(u)
        print(f'round {r}: E = {float(grid.energy(u)):.6f}, |grad| = {g:.1e}, max |M_0i| = {float(Mn[..., 0, 1:].abs().max()):.4f}, max |M_00 - E0| = {float((Mn[..., 0, 0] - E0).abs().max()):.4f}', flush=True)
torch.save({'u': u.cpu(), 'n': d['n'], 'box': d['box'], 'E': E, 'split': False, 'frozen': False, 'X': None, 'degree': 1}, f'results/fields/static_unfrozen_{tag}.pt')
