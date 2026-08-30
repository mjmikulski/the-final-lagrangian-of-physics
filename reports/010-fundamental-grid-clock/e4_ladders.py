"""E4: fundamental-H omega-ladders for the 16 grid cells, in pre-E4 order.

Per cell: fresh-start relaxation of the fundamental Hamiltonian
  E_cell(M; om) = 2 H^3 sum_st (2 D3(0) - D3(V)) + V4          (record part)
                + gamma H^3 sum [ -s^2 + m^2 + 2sk + 4mk + 3k^2 ]  (j-quartic)
at each rung (Adam + one L-BFGS cycle; two protocol levels recorded), with a
runaway guard and, for every cell with an interior well, the sign_cross = -1
control. One shared gamma = 0 control documents the unbraked record disease.

Resumable: results/e4_cells.json is updated after every cell. Usage:
  python e4_ladders.py            # all cells in pre-E4 order
  python e4_ladders.py C10 C18    # subset
"""

import json
import os
import sys
import time

import numpy as np
import torch

from lattice_grid_defs import (DEV, DT, ETA, H, REPS, RUNS, W1, C_P,
                               F4_of, a0_of, class_density, d1, field,
                               gen_catalog, load_or_make_base)
from pre_e4 import EPS4, well_from_coeffs

ADAM_STEPS, ADAM_LR, LBFGS_ITER = 300, 1e-3, 80
DRIVE_SIGN = 1.0  # -1.0: drive-flip control (record kinetic sign reversed)
GUARD_E, GUARD_M = -60.0, 1e3
RESULTS = 'results/e4_cells.json'
C3_CAPS, C3_PAIRS = REPS['C3']


def j_density(F4, U, name):
    if name == 'P_dm':
        return torch.einsum('mngd,...mnab,aA,bB,...ABgd->...',
                            EPS4, F4, ETA, ETA, F4)
    caps, pairs = REPS[name]
    return class_density(F4, U if caps else None, caps, pairs)


def e_cell_fused(M, a0, om, gam, jname, sign_cross=1.0, parts=False):
    from lattice_grid_defs import F4_stack, U_of
    caps_j = () if jname == 'P_dm' else REPS[jname][0]
    U = U_of(M) if gam != 0.0 and (caps_j or jname == 'P_dm') else None
    rec, sj, dpj, dmj = 0.0, 0.0, 0.0, 0.0
    for st in ('fwd', 'bwd'):
        A = [d1(M, ax, st) for ax in range(3)]
        F3 = F4_stack(A, om * a0)                       # (0, +V, -V)
        D3 = 0.5 * class_density(F3[:2], None, C3_CAPS, C3_PAIRS)
        if DRIVE_SIGN > 0:
            rec = rec + 2.0 * (2.0 * D3[0] - D3[1])
        else:
            rec = rec + 2.0 * D3[1]  # statics + POSITIVE kinetic (drive off)
        if gam != 0.0:
            dj = 0.5 * j_density(F3, U, jname)
            sj = sj + dj[0]
            dpj = dpj + dj[1]
            dmj = dmj + dj[2]
    Me = M @ ETA
    P, v4 = Me, 0.0
    for p in range(4):
        if p:
            P = P @ Me
        t = torch.einsum('...kk->...', P)
        v4 = v4 + (t - C_P[p]) ** 2
    E_rec = H ** 3 * (rec.sum() + W1 * v4.sum())
    if gam == 0.0:
        if parts:
            return E_rec, {'E_record': E_rec.item(), 'E_quartic': 0.0,
                           'PR_k2': 0.0, 's_max': 0.0}
        return E_rec
    m = (dpj - dmj) / 2.0
    k = (dpj + dmj) / 2.0 - sj
    quart = (-sj ** 2 + m ** 2 + sign_cross * (2 * sj * k + 4 * m * k)
             + 3 * k ** 2)
    E_q = gam * H ** 3 * quart.sum()
    if parts:
        pr = ((k ** 2).sum() ** 2 / (k ** 4).sum().clamp_min(1e-300)).item()
        return E_rec + E_q, {'E_record': E_rec.item(), 'E_quartic': E_q.item(),
                             'PR_k2': pr, 's_max': sj.abs().max().item()}
    return E_rec + E_q


def relax_rung(M_seed, a0, om, gam, jname, sign_cross=1.0):
    M_raw = M_seed.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=ADAM_LR)
    status = 'ok'
    for it in range(ADAM_STEPS):
        opt.zero_grad()
        E = e_cell_fused(field(M_raw), a0, om, gam, jname, sign_cross)
        E.backward()
        opt.step()
        if (it + 1) % 50 == 0:
            e = E.item()
            if not np.isfinite(e) or e < GUARD_E \
               or M_raw.abs().max().item() > GUARD_M:
                status = 'runaway'
                break
    E_adam = e_cell_fused(field(M_raw.detach()), a0, om, gam, jname,
                          sign_cross).item()
    if status == 'ok':
        opt2 = torch.optim.LBFGS([M_raw], max_iter=LBFGS_ITER, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn='strong_wolfe')

        def closure():
            opt2.zero_grad()
            E = e_cell_fused(field(M_raw), a0, om, gam, jname, sign_cross)
            E.backward()
            return E
        try:
            opt2.step(closure)
        except Exception as ex:  # LBFGS line-search failure on a runaway slope
            status = f'lbfgs_fail:{type(ex).__name__}'
    Mf = field(M_raw.detach())
    E, extra = e_cell_fused(Mf, a0, om, gam, jname, sign_cross, parts=True)
    e = E.item()
    if not np.isfinite(e) or e < GUARD_E or M_raw.abs().max().item() > GUARD_M:
        status = 'runaway'
    return {'omega': om, 'E_total': e, 'E_adam_level': E_adam,
            'status': status, **extra}


def ladder(M_seed, a0, gam, jname, omegas, sign_cross=1.0, tag=''):
    rows = []
    for om in omegas:
        t0 = time.time()
        r = relax_rung(M_seed, a0, om, gam, jname, sign_cross)
        r['seconds'] = round(time.time() - t0, 1)
        rows.append(r)
        print(f'  [{tag}] om {om:6.3f}: E {r["E_total"]:+.6f} '
              f'(adam {r["E_adam_level"]:+.6f}) {r["status"]} '
              f'PR {r["PR_k2"]:.0f} [{r["seconds"]}s]', flush=True)
    return rows


def verdict(rows):
    okp = []
    for r in rows:
        if r['status'] != 'ok':
            break
        okp.append(r)
    runaway_above = len(okp) < len(rows)
    if len(okp) < 3:
        return {'interior_well': False, 'reason': 'too few ok rungs',
                'runaway': True, 'min_at_top': False}
    Es = [r['E_total'] for r in okp]
    Ea = [r['E_adam_level'] for r in okp]
    i = int(np.argmin(Es))
    ia = int(np.argmin(Ea))
    interior = 0 < i < len(okp) - 1
    return {'interior_well': bool(interior and i == ia),
            'min_omega': okp[i]['omega'], 'min_omega_adam': okp[ia]['omega'],
            'level_stable': bool(i == ia), 'depth': Es[i] - Es[0],
            'runaway': bool(runaway_above),
            'min_at_top': bool(i == len(okp) - 1)}


OMEGA_CAP, MAX_EXT = 3.0, 6


def extend_until_turn(rows, Mr, a0, gam, name, tag):
    """While the minimum sits on the top ok rung, append 1.35x rungs until the
    energy turns up, a rung runs away, or the omega cap is hit."""
    v = verdict(rows)
    ext = 0
    while (v['min_at_top'] and not v['runaway'] and ext < MAX_EXT
           and rows[-1]['omega'] * 1.35 <= OMEGA_CAP):
        om_next = round(rows[-1]['omega'] * 1.35, 4)
        rows += ladder(Mr, a0, gam, name, [om_next], tag=tag + ':ext')
        v = verdict(rows)
        ext += 1
    return rows, v


def main():
    with open('results/pre_e4.json') as f:
        pe = json.load(f)
    Mr = load_or_make_base()
    Mg = field(Mr)
    a0 = a0_of(gen_catalog()[pe['generator']], Mg)
    fresh = bool(os.environ.get('M5_FRESH'))  # ignore resumable state
    done = {} if fresh else (
        json.load(open(RESULTS)) if os.path.exists(RESULTS) else {})

    cells = sys.argv[1:] or pe['e4_order']

    if 'CONTROL_gamma0' not in done:
        print('== shared control: gamma = 0 (record theory, no brake) ==')
        rows = ladder(Mr, a0, 0.0, 'C10', [0.0, 0.2, 0.4, 0.6, 0.8],
                      tag='g0')
        done['CONTROL_gamma0'] = {'rows': rows, 'verdict': verdict(rows)}
        json.dump(done, open(RESULTS, 'w'), indent=1, default=float)

    for name in cells:
        redo_ext = False
        if name in done:
            if 'rows' not in done[name]:
                continue
            v_saved = verdict(done[name]['rows'])
            if v_saved['min_at_top'] and not v_saved['runaway']:
                redo_ext = True
            else:
                print(f'== {name}: already done, skipping ==', flush=True)
                continue
        cell = pe['cells'][name]
        if cell.get('no_gamma'):
            done[name] = {'skipped': 'no viable gamma'}
            continue
        gam = cell['gamma']
        w = cell['well']
        ws = w['omega_star'] if w else 0.35
        if redo_ext:  # fix-up: extend a saved min-at-top ladder
            print(f'== {name}: extending saved ladder ==', flush=True)
            rows = done[name]['rows']
        else:
            omegas = [0.0] + [round(f * ws, 4) for f in (0.4, 0.7, 1.0, 1.4)]
            print(f'== {name}: gamma {gam:.4g}, predicted w* '
                  f'{w["omega_star"] if w else None} ==', flush=True)
            rows = ladder(Mr, a0, gam, name, omegas, tag=name)
        rows, v = extend_until_turn(rows, Mr, a0, gam, name, name)
        entry = {'gamma': gam, 'predicted': w, 'rows': rows, 'verdict': v}
        if v['interior_well']:
            print(f'  {name}: interior well at {v["min_omega"]} -> control')
            wm = v['min_omega']
            crows = ladder(Mr, a0, gam, name,
                           [0.0, wm, round(1.35 * wm, 4)], sign_cross=-1.0,
                           tag=name + ':ctrl')
            cv = verdict(crows)
            entry['control_sign_flip'] = {'rows': crows, 'verdict': cv}
        done[name] = entry
        json.dump(done, open(RESULTS, 'w'), indent=1, default=float)
        print(f'  {name} verdict: {v}', flush=True)
    print('E4 pass complete', flush=True)


if __name__ == '__main__':
    main()
