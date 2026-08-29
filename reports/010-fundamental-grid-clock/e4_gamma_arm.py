"""E4 gamma-robustness arm: can ANY brake strength stop the evasion?

The main sweep found no interior well at the frozen-tuned gamma: creep cells
slide gently to the omega cap, dive cells run away at omega ~ 1.2-1.6. If the
runaway persists at gamma x100 and x10^4 (statics budget still untouched, the
deformation was ~0%), the brake is evaded structurally — the spatial profile
reorganizes so the j-density vanishes where the record drive pays — and the
no-go is gamma-robust, not a tuning accident. Representatives: C10 (creep,
flagship), C19 (creep, p=4), C9 (dive). Writes results/e4_gamma_arm.json.
"""

import json

import torch

from lattice_grid_defs import (a0_of, field, gen_catalog, load_or_make_base)
from e4_ladders import ladder, verdict

CONFIGS = [
    ('C10', 100.0, [0.0, 0.35, 1.0, 2.0, 2.97]),
    ('C10', 1e4, [0.0, 0.35, 1.0, 2.0, 2.97]),
    ('C19', 100.0, [0.0, 0.35, 1.0, 2.0, 2.97]),
    ('C9', 100.0, [0.0, 0.6, 0.9, 1.2, 1.63]),
    ('C9', 1e4, [0.0, 0.6, 0.9, 1.2, 1.63]),
]


def main():
    with open('results/pre_e4.json') as f:
        pe = json.load(f)
    Mr = load_or_make_base()
    Mg = field(Mr)
    a0 = a0_of(gen_catalog()[pe['generator']], Mg)
    out = {}
    for name, mult, omegas in CONFIGS:
        gam = pe['cells'][name]['gamma'] * mult
        tag = f'{name}x{mult:g}'
        print(f'== {tag}: gamma {gam:.4g} ==', flush=True)
        rows = ladder(Mr, a0, gam, name, omegas, tag=tag)
        v = verdict(rows)
        out[tag] = {'gamma': gam, 'mult': mult, 'rows': rows, 'verdict': v}
        json.dump(out, open('results/e4_gamma_arm.json', 'w'), indent=1,
                  default=float)
        print(f'  {tag} verdict: {v}', flush=True)
    print('gamma arm complete', flush=True)


if __name__ == '__main__':
    main()
