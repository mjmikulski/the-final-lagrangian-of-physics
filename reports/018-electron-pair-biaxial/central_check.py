"""The midpoint of the pair seed (review round 1): lattice energy of a unit box around the midpoint at h = 1/8,
1/16, 1/32, for the electrostatic seed as in pair.py (V vanishes at the midpoint: a degree-0 defect cut off only
by the lattice) and for the seed with that defect given the melted core of the single hedgehog
(pair_flow.py --melt-centre). The ghost layer of the unit box carries the same analytic seed.
Output: results/central_check.json.
"""
import json
import os

import torch
from soliton import Grid
from pair import single_profiles
from pair_flow import seed, biaxial_vacuum_field

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))


def central_energy(E, beta, d, melt, n, prof):
    g = Grid(n, 1.0, E=E, degree=0, device='cpu')
    gp = Grid(n + 2, 1.0 + 2.0 / n, E=E, degree=0, device='cpu')        # its nodes are g's padded nodes
    u, _ = seed(g, E, d, prof, 1.2, beta, melt)
    up, _ = seed(gp, E, d, prof, 1.2, beta, melt)
    g.boundary = up
    return float(g.energy(u))


if __name__ == '__main__':
    prof = single_profiles(os.path.join(HERE, 'results', 'fields', 'static_base_n48.pt'))
    out = []
    for beta, d in ((0.0, 4.0), (0.3, 6.0)):
        E = (100.0, 1.0, 0.01, 0.01) if beta == 0 else (100.0, 1.0, beta, 0.0)
        for melt in (False, True):
            row = dict(beta=beta, d=d, melt_centre=melt,
                       energies={str(n): central_energy(E, beta, d, melt, n, prof) for n in (8, 16, 32)})
            print(json.dumps(row), flush=True)
            out.append(row)
    json.dump(out, open(os.path.join(HERE, 'results', 'central_check.json'), 'w'), indent=1)
