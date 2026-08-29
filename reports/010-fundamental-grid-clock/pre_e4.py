"""pre-E4: integrated cell coefficients on the eta-relaxed electron profile.

Regenerates the 004 eta base profile (gate: seed oracle 9.263660060), validates
the lattice family port, computes per-class integrals
  D1 (record drive), S2, SK, M2, MK, K2, s_max, PR(k^2)
and per cell the gamma choice, frozen-profile well prediction and the ranked
E4 order. Writes results/pre_e4.json.
"""

import json
import time

import numpy as np
import torch

from lattice_grid_defs import (DEV, DT, ETA, FREE, H, N, REPS, RUNS, W1,
                               a0_of, class_density, d1, e_static, field,
                               gen_catalog, load_or_make_base, comm,
                               split_densities, validate)

J_AXIS = ['C10', 'C9', 'C11', 'C18', 'C19', 'C20', 'C15', 'C16', 'C17',
          'C6', 'C7', 'C8', 'C12', 'C13', 'C14', 'P_dm']

EPS4 = torch.zeros(4, 4, 4, 4, dtype=DT, device=DEV)
from itertools import permutations
for p in permutations(range(4)):
    sgn, q = 1, list(p)
    for i in range(4):
        for j in range(i + 1, 4):
            if q[i] > q[j]:
                sgn = -sgn
    EPS4[p] = float(sgn)


def pdm_density(M, a0, om):
    """P_dm = eps^{mn gd} F_{mn ab} F^{ab}_{gd}; split via lambda like classes."""
    from lattice_grid_defs import F4_of
    s = m = k = None
    Z = torch.zeros_like(M[..., 0, 0])
    s, dp, dm_ = Z.clone(), Z.clone(), Z.clone()
    for st in ('fwd', 'bwd'):
        A = [d1(M, ax, st) for ax in range(3)]
        for lam, tag in ((0.0, 's'), (1.0, 'dp'), (-1.0, 'dm')):
            V = lam * om * a0
            F4 = F4_of([V] + A)
            d = 0.5 * torch.einsum('mngd,...mnab,aA,bB,...ABgd->...',
                                   EPS4, F4, ETA, ETA, F4)
            if tag == 's':
                s = s + d
            elif tag == 'dp':
                dp = dp + d
            else:
                dm_ = dm_ + d
    return s, (dp - dm_) / 2.0, (dp + dm_) / 2.0 - s


def splits(M, a0, om, name):
    if name == 'P_dm':
        return pdm_density(M, a0, om)
    return split_densities(M, a0, om, name)


def well_from_coeffs(A, B, C):
    """Interior stationary point of A w^2 + B w^3 + C w^4 (E3 formulas)."""
    if C <= 0:
        return None
    disc = 9 * B * B - 32 * A * C
    if disc < 0:
        return None
    r = [(-3 * B + s * np.sqrt(disc)) / (8 * C) for s in (+1, -1)]
    pos = [x for x in r if x > 0]
    if not pos:
        return None
    w = max(pos)  # H' goes - to + at the largest positive root: the minimum
    depth = A * w**2 + B * w**3 + C * w**4
    if depth >= 0:
        return None
    return {'omega_star': float(w), 'depth': float(depth)}


def main():
    t0 = time.time()
    Mr = load_or_make_base()
    Mg = field(Mr)
    E_stat = e_static(Mg, 'eta').item()
    print(f'base profile ready: E_stat = {E_stat:.6f}  [{time.time()-t0:.0f}s]')

    val = validate(Mg, torch.zeros_like(Mg))
    print('validation:', {k: f'{v:.3e}' for k, v in val.items()})
    assert val['statics_identity_rel'] < 1e-12
    assert val['pointwise_crosscheck_rel'] < 1e-10

    # generator choice: largest record drive D1 among the boosts
    D1s = {}
    for gname in ('boost_x', 'boost_y', 'boost_z'):
        a0 = a0_of(gen_catalog()[gname], Mg)
        _, _, k1 = split_densities(Mg, a0, 1.0, 'C3')
        D1s[gname] = (2.0 * H ** 3 * k1.sum()).item()
    gen = max(D1s, key=D1s.get)
    D1 = D1s[gen]
    print('record drive D1 by generator:', {k: f'{v:.5f}' for k, v in D1s.items()},
          '-> using', gen)
    assert D1 > 0, 'no drive on the electron profile: grid dead (blocker)'
    a0 = a0_of(gen_catalog()[gen], Mg)

    e_dens_scale = (e_static(Mg, 'eta') / H ** 3 / (N ** 3)).item()
    rows = {}
    for name in J_AXIS:
        s, m, k = splits(Mg, a0, 1.0, name)
        S2 = (H ** 3 * (s ** 2).sum()).item()
        SK = (H ** 3 * (s * k).sum()).item()
        M2 = (H ** 3 * (m ** 2).sum()).item()
        MK = (H ** 3 * (m * k).sum()).item()
        K2 = (H ** 3 * (k ** 2).sum()).item()
        smax = s.abs().max().item()
        pr = ((k ** 2).sum() ** 2 / (k ** 4).sum().clamp_min(1e-300)).item()
        # gamma: hit omega* = 0.35 at frozen profile, capped by the 5% budget
        gam_t = (D1 / 2) / (6 * 0.1225 * K2 + (M2 + 2 * SK)) \
            if (6 * 0.1225 * K2 + (M2 + 2 * SK)) > 0 else np.inf
        gam_b = 0.05 * E_stat / S2 if S2 > 1e-30 else np.inf
        gam = min(gam_t, gam_b)
        capped = bool(gam_b < gam_t)
        if not np.isfinite(gam) or gam <= 0:
            rows[name] = {'K2': K2, 'no_gamma': True}
            continue
        A = -D1 / 2 + gam * (M2 + 2 * SK)
        B = 4 * gam * MK
        C = 3 * gam * K2
        well = well_from_coeffs(A, B, C)
        rows[name] = {
            'S2': S2, 'SK': SK, 'M2': M2, 'MK': MK, 'K2': K2,
            's_max': smax, 'PR_k2': pr, 'gamma': gam,
            'gamma_budget_capped': capped,
            'statics_deformation_frac': gam * S2 / E_stat,
            'A': A, 'B': B, 'C': C, 'well': well,
            'runaway_proxy': gam * smax ** 2 / e_dens_scale,
        }
        w = well['omega_star'] if well else float('nan')
        d = well['depth'] if well else float('nan')
        print(f'{name:5s} gam {gam:9.3g}{"(cap)" if capped else "     "} '
              f'A {A:+.4e} C {C:.3e} w* {w:6.3f} depth {d:+.3e} '
              f'deform {rows[name]["statics_deformation_frac"]*100:5.2f}% '
              f'runaway~{rows[name]["runaway_proxy"]:.2e}')

    # ranking: well first, then smaller runaway proxy, then deeper well
    def key(nm):
        r = rows[nm]
        if r.get('no_gamma') or not r.get('well'):
            return (1, 0, 0)
        return (0, r['runaway_proxy'], r['well']['depth'])
    order = sorted(J_AXIS, key=key)
    print('E4 order:', order)

    out = {'E_stat': E_stat, 'D1_by_gen': D1s, 'generator': gen,
           'validation': val, 'cells': rows, 'e4_order': order,
           'omega_target': 0.35, 'budget_frac': 0.05}
    with open('results/pre_e4.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'pre-E4 done [{time.time()-t0:.0f}s] -> results/pre_e4.json')


if __name__ == '__main__':
    main()
