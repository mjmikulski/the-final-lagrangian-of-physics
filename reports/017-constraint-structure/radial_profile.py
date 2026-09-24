"""Kinetic-form spectrum and characteristic speeds along a ray through the lattice fields (for the figures).

For the frozen hedgehog and the endpoint with the time sector free (report 016, E_0 = 100), at the quadrature
points closest to a generic ray (direction (1, 2, 3)/sqrt(14)): the ten eigenvalues of K (normalised by the
largest), the same with the indefinite eta norm, and the squared speeds for propagation along the radius and
along a tangent. Output: results/radial_profile.json.
"""
import json
import numpy as np
import torch
from forms import forms_formula, speeds2
from soliton import Grid

torch.set_default_dtype(torch.float64)
FIELDS = '../016-time-axis-screening/results/fields/'
RAY = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
TANGENT = np.cross(RAY, [0.0, 0.0, 1.0])
TANGENT /= np.linalg.norm(TANGENT)
RADII = np.round(np.arange(0.25, 5.51, 0.25), 2)

if __name__ == '__main__':
    out = {'ray': RAY.tolist(), 'tangent': TANGENT.tolist(), 'fields': {}}
    for name, fn in (('frozen hedgehog', 'static_base_n32.pt'), ('time sector free (E0 = 100)', 'static_unfrozen_E0_100.pt')):
        d = torch.load(FIELDS + fn)
        grid = Grid(d['n'], d['box'], E=d['E'], device='cpu')
        Mq, dMq = grid.derivatives(d['u'].double())
        xq = grid.xq.reshape(-1, 3).numpy()
        Mq, dMq = Mq.reshape(-1, 4, 4).numpy(), dMq.reshape(-1, 4, 4, 4).numpy()
        rows = []
        for rad in RADII:
            j = int(np.linalg.norm(xq - rad * RAY, axis=1).argmin())
            K, _ = forms_formula(Mq[j], dMq[j])
            kd = np.sort(np.linalg.eigvalsh(K))[::-1]
            Ke, _ = forms_formula(Mq[j], dMq[j], metric='eta')
            ke = np.sort(np.linalg.eigvalsh(0.5 * (Ke + Ke.T)))[::-1]
            sp = {}
            for lab, k in (('radial', xq[j] / np.linalg.norm(xq[j])), ('tangential', TANGENT)):
                sp[lab] = speeds2(*forms_formula(Mq[j], dMq[j], 0.0, k))[0].tolist()
            rows.append(dict(r=float(np.linalg.norm(xq[j])), K_delta=(kd / kd[0]).tolist(),
                             K_eta=(ke / abs(ke).max()).tolist(), speed2=sp))
        out['fields'][name] = rows
        print(name, 'done')
    json.dump(out, open('results/radial_profile.json', 'w'), indent=1)
