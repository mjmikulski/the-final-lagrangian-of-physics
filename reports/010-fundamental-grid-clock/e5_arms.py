"""E5 prep arms (GPU, sequential):

A. Depth-plateau probe of the C10 x10 well, 008 criterion: Adam 500 + four
   L-BFGS(150) cycles at omega in {0, 0.2, 0.28}, energy recorded after every
   level; the well depth per level must settle (final level-to-level change
   smaller than the first and below 10% of the depth).
B. gamma-window map for C10: mult in {5, 7, 14, 20} at rungs
   {0, 0.1, 0.15, 0.2, 0.28, 0.42} (standard protocol), plus the two missing
   fine rungs {0.1, 0.15} at mult 10.

Writes results/e5_arms.json.
"""

import json
import time

import numpy as np
import torch

from lattice_grid_defs import a0_of, field, gen_catalog, load_or_make_base
import e4_ladders
from e4_ladders import e_cell_fused, ladder, verdict


def relax_deep(M_seed, a0, om, gam, jname, cycles=4):
    M_raw = M_seed.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        e_cell_fused(field(M_raw), a0, om, gam, jname).backward()
        opt.step()
    levels = [e_cell_fused(field(M_raw.detach()), a0, om, gam, jname).item()]
    for cy in range(cycles):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=150, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn='strong_wolfe')

        def closure():
            opt2.zero_grad()
            E = e_cell_fused(field(M_raw), a0, om, gam, jname)
            E.backward()
            return E
        opt2.step(closure)
        levels.append(e_cell_fused(field(M_raw.detach()), a0, om, gam,
                                   jname).item())
    return levels


def main():
    from lattice_grid_defs import DEV
    print('device:', DEV, flush=True)
    assert str(DEV).startswith('cuda'), 'CUDA not available at import time'
    pe = json.load(open('results/pre_e4.json'))
    Mr = load_or_make_base()
    a0 = a0_of(gen_catalog()[pe['generator']], field(Mr))
    out = {}
    gam10 = pe['cells']['C10']['gamma'] * 10.0

    # --- arm A: depth plateau ---
    lv = {}
    for om in (0.0, 0.2, 0.28):
        t0 = time.time()
        lv[str(om)] = relax_deep(Mr, a0, om, gam10, 'C10')
        print(f'[deep] om {om}: levels {[f"{x:.6f}" for x in lv[str(om)]]} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    depth_per_level = [l2 - l0 for l0, l2 in zip(lv['0.0'], lv['0.2'])]
    changes = list(np.diff(depth_per_level))
    plateau = (abs(changes[-1]) < abs(changes[0])
               and abs(changes[-1]) < 0.1 * abs(depth_per_level[-1]))
    out['deep_well'] = {'levels': lv, 'depth_per_level': depth_per_level,
                        'depth_changes': changes, 'plateau': bool(plateau),
                        'bracket_holds_last_level':
                            bool(lv['0.2'][-1] < lv['0.0'][-1]
                                 and lv['0.2'][-1] < lv['0.28'][-1])}
    print('deep-well:', {k: v for k, v in out['deep_well'].items()
                         if k != 'levels'}, flush=True)
    json.dump(out, open('results/e5_arms.json', 'w'), indent=1, default=float)

    # --- arm B: gamma-window map ---
    win = {}
    for mult in (5.0, 7.0, 14.0, 20.0):
        gam = pe['cells']['C10']['gamma'] * mult
        rows = ladder(Mr, a0, gam, 'C10',
                      [0.0, 0.1, 0.15, 0.2, 0.28, 0.42], tag=f'w{mult:g}')
        win[f'x{mult:g}'] = {'gamma': gam, 'rows': rows,
                             'verdict': verdict(rows)}
        print(f'  x{mult:g}: {win[f"x{mult:g}"]["verdict"]}', flush=True)
        out['window_map'] = win
        json.dump(out, open('results/e5_arms.json', 'w'), indent=1,
                  default=float)
    rows = ladder(Mr, a0, gam10, 'C10', [0.1, 0.15], tag='w10fine')
    out['x10_fine_extra'] = {'rows': rows}
    json.dump(out, open('results/e5_arms.json', 'w'), indent=1, default=float)
    print('e5 arms complete', flush=True)


if __name__ == '__main__':
    main()
