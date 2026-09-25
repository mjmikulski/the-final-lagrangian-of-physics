"""The static electron of the E_0 = 100 conventions (frozen sector): energies in three boxes at one spacing and
one finer spacing, the analytic exterior tail, the charge, and the virial.

Fields (relaxed in the development repository new-duda-lagrangian, commit b757030, `run_static.py`): n = 32 box
12, n = 48 box 18, n = 64 box 24 (spacing 0.375) and n = 64 box 16 (spacing 0.25). The first is committed in
report 016; the others are release assets of this report (fetched by reproduce.sh).

For each: E_4 (quartic) and E_V (potential) on the lattice; the analytic tail outside the inscribed sphere of
radius R = box/2, 16 pi Delta^4 / R with Delta = E_1 - E_2 (the hedgehog exterior density 4 Delta^4 / r^4);
the virial (E_4 + tail) / (3 E_V) (Derrick: 1 for a stationary point of the quartic + potential energy in
infinite space); the degree of the charge direction on spheres r = 1, 2, 3, 4 (signed, from pair.py).
Output: results/electron_record.json.
"""
import json
import os

import torch
from soliton import Grid
from pair import signed_degree

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = {
    'n32_box12': os.path.join(HERE, '..', '016-time-axis-screening', 'results', 'fields', 'static_base_n32.pt'),
    'n48_box18': os.path.join(HERE, 'results', 'fields', 'static_base_n48.pt'),
    'n64_box24': os.path.join(HERE, 'results', 'fields', 'static_base_n64_box24.pt'),
    'n64_box16': os.path.join(HERE, 'results', 'fields', 'static_base_n64_box16.pt'),
}

if __name__ == '__main__':
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    out = {}
    for name, path in FIELDS.items():
        if not os.path.exists(path):
            print(f'{name}: {path} missing (release asset), skipped', flush=True)
            continue
        d = torch.load(path, map_location='cpu', weights_only=False)
        g = Grid(d['n'], d['box'], E=d['E'], device=dev)
        u = d['u'].double().to(dev)
        E4, EV, _ = (float(t) for t in g.energy_terms(u))
        D = d['E'][1] - d['E'][2]
        R = d['box'] / 2
        tail = 16 * torch.pi * D ** 4 / R
        deg = {str(r): signed_degree(g, u, (0.0, 0.0, 0.0), r) for r in (1.0, 2.0, 3.0, 4.0) if r < R - 0.5}
        out[name] = dict(n=d['n'], box=d['box'], h=d['box'] / d['n'], E4=E4, EV=EV, E_box=E4 + EV, tail=float(tail),
                         E_with_tail=E4 + EV + float(tail), virial_with_tail=(E4 + float(tail)) / (3 * EV),
                         virial_box=E4 / (3 * EV), degree=deg)
        print(name, json.dumps(out[name]), flush=True)
    # extrapolation in 1/R through the three boxes at h = 0.375 (E_inf + a/R), and the spacing correction
    pts = [(out[k]['box'] / 2, out[k]['E_with_tail']) for k in ('n32_box12', 'n48_box18', 'n64_box24') if k in out]
    if len(pts) == 3:
        import numpy as np
        R_, E_ = np.array(pts).T
        a1 = np.polyfit(1 / R_, E_, 1)
        a2 = np.polyfit(1 / R_, E_, 2)
        out['extrapolation'] = {'linear_in_1overR': float(a1[-1]), 'quadratic_in_1overR': float(a2[-1])}
        if 'n64_box16' in out:
            # the h = 0.25 point against the h = 0.375 value interpolated to R = 8 by the linear fit
            ref = float(np.polyval(a1, 1 / 8.0))
            out['extrapolation']['spacing_correction_at_R8'] = out['n64_box16']['E_with_tail'] - ref
        print('extrapolation', out['extrapolation'], flush=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'electron_record.json'), 'w'), indent=1)
