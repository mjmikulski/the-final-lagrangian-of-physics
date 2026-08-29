"""E4 analysis: verdict table + two-panel figure from committed JSONs.

Reads results/pre_e4.json and results/e4_cells.json (works on partial runs:
analyzes whatever cells are done). Writes results/fig_e4_ladders.png and
prints the ranked verdict table.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BAND = {**{c: ('p=4', 'tab:green') for c in ('C18', 'C19', 'C20')},
        **{c: ('dd p=2', 'tab:orange') for c in ('C9', 'C10', 'C11')},
        **{c: ('p=2 orbit-0', 'tab:gray')
           for c in ('C6', 'C7', 'C8', 'C12', 'C13', 'C14',
                     'C15', 'C16', 'C17')}}


def main():
    pe = json.load(open('results/pre_e4.json'))
    e4 = json.load(open('results/e4_cells.json'))
    cells = [c for c in pe['e4_order'] if c in e4 and 'rows' in e4[c]]

    print(f"{'cell':6s} {'band':12s} {'well?':6s} {'w_min':6s} {'w_pred':7s} "
          f"{'depth':>10s} {'stable':7s} {'ctrl kills':10s} {'runaway':8s}")
    rows_fig = []
    for c in cells:
        d = e4[c]
        v = d['verdict']
        pred = d['predicted']['omega_star'] if d.get('predicted') else np.nan
        ctrl = d.get('control_sign_flip')
        ctrl_kills = '-' if ctrl is None else \
            ('YES' if not ctrl['verdict']['interior_well'] else 'NO(!)')
        print(f"{c:6s} {BAND.get(c, ('', ''))[0]:12s} "
              f"{str(v.get('interior_well')):6s} "
              f"{str(v.get('min_omega', '-')):6s} {pred:7.3f} "
              f"{v.get('depth', float('nan')):+10.2e} "
              f"{str(v.get('level_stable', '-')):7s} {ctrl_kills:10s} "
              f"{str(v.get('runaway')):8s}")
        rows_fig.append(c)

    if 'CONTROL_gamma0' in e4:
        g0 = e4['CONTROL_gamma0']['verdict']
        print(f"gamma=0 control: interior_well={g0.get('interior_well')} "
              f"min_omega={g0.get('min_omega')} (record disease: expect "
              f"monotone decrease, no interior well)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)

    # panel A: relaxed ladders, dE(omega) = E - E(0), predictions dashed
    for c in rows_fig:
        d = e4[c]
        om = [r['omega'] for r in d['rows']]
        E = [r['E_total'] for r in d['rows']]
        dE = [e - E[0] for e in E]
        col = BAND.get(c, ('', 'tab:blue'))[1]
        ax1.plot(om, dE, 'o-', ms=3.5, lw=1.2, color=col, alpha=0.85)
        bad = [(o, e) for o, e, r in zip(om, dE, d['rows'])
               if r['status'] != 'ok']
        if bad:
            ax1.plot(*zip(*bad), 'x', ms=7, mew=1.8, color=col)
        cell = pe['cells'][c]
        A, B, C = cell['A'], cell['B'], cell['C']
        og = np.linspace(0, max(om), 120)
        ax1.plot(og, A * og**2 + B * og**3 + C * og**4, '--', lw=0.7,
                 color=col, alpha=0.45)
        i = int(np.argmin(E))
        if 0 < i < len(E) - 1:
            ax1.annotate(c, (om[i], dE[i]), fontsize=6.5,
                         textcoords='offset points', xytext=(3, -7))
    if 'CONTROL_gamma0' in e4:
        rows = e4['CONTROL_gamma0']['rows']
        om = [r['omega'] for r in rows]
        dE = [r['E_total'] - rows[0]['E_total'] for r in rows]
        ax1.plot(om, dE, 's:', ms=3.5, lw=1.2, color='tab:red')
        ax1.annotate('γ=0 (record, no brake)', (om[-1], dE[-1]), fontsize=7,
                     color='tab:red', textcoords='offset points',
                     xytext=(-95, 4))
    ax1.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax1.set_yscale('symlog', linthresh=0.05)
    ax1.set_xlabel('ω'); ax1.set_ylabel('E(ω) − E(0)   [symlog]')
    ax1.set_title('Fundamental-H ladders (relaxed; frozen prediction dashed)',
                  fontsize=9.5)
    handles = [plt.Line2D([], [], color=c, lw=2, label=l) for l, c in
               [('p=4 band', 'tab:green'), ('∂–∂ p=2 band', 'tab:orange'),
                ('orbit-0 p=2 band', 'tab:gray'),
                ('γ=0 control', 'tab:red')]]
    ax1.legend(handles=handles, fontsize=7, loc='lower left')

    # panel B: depth vs runaway proxy (log), verdict-coded
    for c in rows_fig:
        d = e4[c]
        v = d['verdict']
        cell = pe['cells'][c]
        x = cell['runaway_proxy']
        y = abs(v.get('depth', np.nan))
        col = BAND.get(c, ('', 'tab:blue'))[1]
        mk = 'o' if v.get('interior_well') else 'x'
        ax2.scatter([x], [y], c=col, marker=mk, s=42)
        ax2.annotate(c, (x, y), fontsize=6.5, textcoords='offset points',
                     xytext=(4, 3))
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('runaway proxy  γ·s²_max / ⟨e_stat⟩')
    ax2.set_ylabel('|well depth|')
    ax2.set_title('Safety vs depth (o = interior well, x = none)',
                  fontsize=9.5)
    fig.tight_layout()
    fig.savefig('results/fig_e4_ladders.png', bbox_inches='tight')
    print('figure -> results/fig_e4_ladders.png')


if __name__ == '__main__':
    main()
