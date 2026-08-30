"""Review round 2, finding 1: the gyroscopic lambda-families on the lattice.

The span contains I_j = Q + lambda*P with P = C0 - 4*C2 + C5 (s == 0, k == 0,
m != 0 exactly -- a combination of cap-free classes, so the identity holds on
the lattice verbatim). Squaring changes only the m-dependent coefficients:
  A(lambda) = A_Q + gamma*(2*lambda*<m_Q m_P> + lambda^2*<m_P^2>),
  B(lambda) = B_Q + 4*gamma*lambda*<m_P k_Q>;
s_j, k_j (hence the F4 leak and the omega^4 brake) are lambda-invariant.
This script measures the lambda-family coefficients on the eta electron
profile and boost-x tangent: <m_P^2>, and <m_P k_Q>, <m_P m_Q> for every
j-cell. Writes results/gyro_family_lattice.json.
"""

import json

import torch

from lattice_grid_defs import (H, REPS, a0_of, field, gen_catalog,
                               load_or_make_base, split_densities)

J_CELLS = ['C10', 'C9', 'C11', 'C18', 'C19', 'C20', 'C15', 'C16', 'C17',
           'C6', 'C7', 'C8', 'C12', 'C13', 'C14']


def main():
    with open('results/pre_e4.json') as f:
        pe = json.load(f)
    Mr = load_or_make_base()
    Mg = field(Mr)
    a0 = a0_of(gen_catalog()[pe['generator']], Mg)

    parts = {}
    for n, c in (('C0', 1.0), ('C2', -4.0), ('C5', 1.0)):
        parts[n] = (c, split_densities(Mg, a0, 1.0, n))
    sP = sum(c * p[0] for c, p in parts.values())
    mP = sum(c * p[1] for c, p in parts.values())
    kP = sum(c * p[2] for c, p in parts.values())
    scale = max(float(x.abs().max()) for _, p in parts.values() for x in p)
    out = {
        's_P_max': float(sP.abs().max()), 'k_P_max': float(kP.abs().max()),
        'density_scale': scale,
        'MP2': float(H ** 3 * (mP ** 2).sum()),
        'm_P_max': float(mP.abs().max()),
    }
    assert out['s_P_max'] < 1e-9 * scale and out['k_P_max'] < 1e-9 * scale, \
        'P split identity broken on the lattice'
    cross = {}
    for n in J_CELLS:
        s, m, k = split_densities(Mg, a0, 1.0, n)
        cross[n] = {'mP_kQ': float(H ** 3 * (mP * k).sum()),
                    'mP_mQ': float(H ** 3 * (mP * m).sum())}
    out['cross'] = cross
    with open('results/gyro_family_lattice.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f"s_P max {out['s_P_max']:.2e}, k_P max {out['k_P_max']:.2e} "
          f"(scale {scale:.2e}) -> identity holds on the lattice")
    print(f"<m_P^2> = {out['MP2']:.3e}, max|m_P| = {out['m_P_max']:.3e}")
    worst = max(abs(v['mP_kQ']) for v in cross.values())
    print(f"largest |<m_P k_Q>| over the 15 j-cells: {worst:.3e}")


if __name__ == '__main__':
    main()
