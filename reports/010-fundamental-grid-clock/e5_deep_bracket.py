"""Review finding 2 (PR #12): a converged-level demonstration of one well
inside the window. Deep protocol (Adam 500 + six L-BFGS(150) cycles) at
C10 x14, rungs {0, 0.1, 0.15, 0.2, 0.28}: per-level energies AND |grad|_inf
recorded, final rung fields persisted, and BOTH bracket inequalities checked
per level with margins compared against the per-level energy creep.

Writes results/e5_deep_bracket.json and results/deep14_om*.npz.
"""

import json
import time

import numpy as np
import torch

from lattice_grid_defs import (DEV, a0_of, field, gen_catalog,
                               load_or_make_base)
from e4_ladders import e_cell_fused

CYCLES = 6
RUNGS = [0.0, 0.1, 0.15, 0.2, 0.28]


def relax_deep(M_seed, a0, om, gam, jname):
    M_raw = M_seed.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        e_cell_fused(field(M_raw), a0, om, gam, jname).backward()
        opt.step()

    def snap():
        Mv = M_raw.detach().requires_grad_(True)
        E = e_cell_fused(field(Mv), a0, om, gam, jname)
        g = torch.autograd.grad(E, Mv)[0]
        return float(E.detach()), float(g.abs().max())
    levels, grads = [], []
    e0, g0 = snap()
    levels.append(e0); grads.append(g0)
    for cy in range(CYCLES):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=150, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn='strong_wolfe')

        def closure():
            opt2.zero_grad()
            E = e_cell_fused(field(M_raw), a0, om, gam, jname)
            E.backward()
            return E
        opt2.step(closure)
        e, g = snap()
        levels.append(e); grads.append(g)
    return M_raw.detach(), levels, grads


def main():
    print('device:', DEV, flush=True)
    assert str(DEV).startswith('cuda')
    with open('results/pre_e4.json') as f:
        pe = json.load(f)
    Mr = load_or_make_base()
    a0 = a0_of(gen_catalog()[pe['generator']], field(Mr))
    gam = pe['cells']['C10']['gamma'] * 14.0
    out = {'gamma': gam, 'cycles': CYCLES, 'rungs': {}}
    for om in RUNGS:
        t0 = time.time()
        Mf, levels, grads = relax_deep(Mr, a0, om, gam, 'C10')
        np.savez_compressed(f'results/deep14_om{str(om).replace(".", "")}.npz',
                            M=Mf.cpu().numpy())
        out['rungs'][str(om)] = {'levels': levels, 'grad_inf': grads}
        print(f'[deep14] om {om}: E {levels[-1]:.9f} '
              f'(levels {[f"{x:.6f}" for x in levels]}) '
              f'ginf {grads[-1]:.2e} [{time.time()-t0:.0f}s]', flush=True)
        json.dump(out, open('results/e5_deep_bracket.json', 'w'), indent=1)

    # bracket analysis per level
    L = {om: out['rungs'][str(om)]['levels'] for om in RUNGS}
    nlev = len(L[0.0])
    rows = []
    for lv in range(nlev):
        Es = {om: L[om][lv] for om in RUNGS}
        mn = min(RUNGS, key=lambda o: Es[o])
        rows.append({'level': lv, 'min_omega': mn,
                     'E': {str(o): Es[o] for o in RUNGS}})
    final = rows[-1]
    Ef = {float(k): v for k, v in final['E'].items()}
    mn = final['min_omega']
    ridx = RUNGS.index(mn)
    interior = 0 < ridx < len(RUNGS) - 1
    creep = {str(o): abs(L[o][-1] - L[o][-2]) for o in RUNGS}
    left_gap = Ef[RUNGS[ridx - 1]] - Ef[mn] if interior else None
    right_gap = Ef[RUNGS[ridx + 1]] - Ef[mn] if interior else None
    max_creep = max(creep.values())
    out['bracket'] = {
        'min_omega_per_level': [r['min_omega'] for r in rows],
        'final_min_omega': mn, 'interior': bool(interior),
        'left_gap': left_gap, 'right_gap': right_gap,
        'per_rung_last_level_change': creep,
        'gaps_resolve_creep': bool(interior and left_gap > max_creep
                                   and right_gap > max_creep),
        'depth': Ef[mn] - Ef[0.0] if interior else None,
    }
    json.dump(out, open('results/e5_deep_bracket.json', 'w'), indent=1)
    print('bracket:', out['bracket'], flush=True)


if __name__ == '__main__':
    main()
