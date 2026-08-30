"""Report 010 figures, regenerated from the committed results JSONs only.

Fig 1 (fig_grid_ladders.png): the campaign in two panels — (a) the main sweep
at frozen-tuned gamma (no wells: creep vs dive, gamma=0 control), (b) the
two-sided gamma-window of the flagship C10 (evasion below, tick at x10, drive
death above; drive-flip control).

Fig 2 (fig_well_anatomy.png): the winning well — (a) fine rungs at x10 with
the drive-flip control, (b) the 008-style depth-per-level plateau.
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BAND = {**{c: ('p=4', 'tab:green') for c in ('C18', 'C19', 'C20')},
        **{c: ('dd p=2', 'tab:orange') for c in ('C9', 'C10', 'C11')},
        **{c: ('orbit-0 p=2', 'tab:gray')
           for c in ('C6', 'C7', 'C8', 'C12', 'C13', 'C14',
                     'C15', 'C16', 'C17')}}


def load(name):
    with open(f'results/{name}.json') as f:
        return json.load(f)


def dE(rows):
    om = [r['omega'] for r in rows]
    E = [r['E_total'] for r in rows]
    ok = [r['status'] == 'ok' for r in rows]
    return om, [e - E[0] for e in E], ok


def fig1(e4, garm, e5, conf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=150)

    for c, d in e4.items():
        if c == 'CONTROL_gamma0' or 'rows' not in d:
            continue
        om, de, ok = dE(d['rows'])
        col = 'tab:blue' if c == 'C10' else BAND.get(c, ('', 'k'))[1]
        lw = 2.0 if c == 'C10' else 1.0
        ax1.plot(om, de, 'o-', ms=2.5, lw=lw, color=col,
                 alpha=0.9 if c == 'C10' else 0.5)
        bad = [(o, e) for o, e, k in zip(om, de, ok) if not k]
        if bad:
            ax1.plot(*zip(*bad), 'x', ms=7, mew=1.8, color='k', zorder=6)
        if c == 'C10':
            ax1.annotate('C10', (om[-1], de[-1]), fontsize=8.5,
                         color='tab:blue', fontweight='bold',
                         textcoords='offset points', xytext=(4, 4))
    om, de, _ = dE(e4['CONTROL_gamma0']['rows'])
    ax1.plot(om, de, 's:', ms=4, lw=1.4, color='tab:red', zorder=5)
    ax1.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax1.set_yscale('symlog', linthresh=0.05)
    ax1.set_ylim(-160, 0.03)
    ax1.set_yticks([-100, -10, -1, -0.1, 0])
    ax1.set_xlabel('\u03c9')
    ax1.set_ylabel('E(\u03c9) \u2212 E(0)   [symlog, linear below 0.05]')
    ax1.set_title('(a) Main sweep, frozen-tuned \u03b3: no interior wells\n'
                  '(ladders end at the first runaway rung \u00d7; '
                  'inset: linear zoom at the well-depth scale)', fontsize=9.5)
    ax1.legend(handles=[
        plt.Line2D([], [], color='tab:blue', lw=2, label='C10 (flagship)'),
        plt.Line2D([], [], color='tab:orange', lw=2,
                   label='\u2202\u2013\u2202 caps \u2014 2 more cells'),
        plt.Line2D([], [], color='tab:green', lw=2,
                   label='double caps \u2014 3 cells'),
        plt.Line2D([], [], color='tab:gray', lw=2,
                   label='orbit-0 bands \u2014 9 cells'),
        plt.Line2D([], [], color='k', marker='x', ls='none',
                   label='runaway rung'),
        plt.Line2D([], [], color='tab:red', marker='s', ls=':',
                   label='\u03b3=0 control'),
    ], fontsize=7, loc='upper right')

    axi = ax1.inset_axes([0.56, 0.12, 0.42, 0.34])
    for c, d in e4.items():
        if c == 'CONTROL_gamma0' or 'rows' not in d:
            continue
        om, de, ok = dE(d['rows'])
        col = 'tab:blue' if c == 'C10' else BAND.get(c, ('', 'k'))[1]
        axi.plot(om, de, '-', lw=1.4 if c == 'C10' else 0.8, color=col,
                 alpha=0.9 if c == 'C10' else 0.5)
    axi.set_xlim(0, 0.25)
    axi.set_ylim(-8e-3, 1e-3)
    axi.axhline(0, color='k', lw=0.5, alpha=0.5)
    axi.axhline(-2.3e-3, color='tab:red', lw=0.7, ls='--')
    axi.annotate('the \u00d710 well depth', (0.02, -2.9e-3), fontsize=6.5,
                 color='tab:red')
    axi.tick_params(labelsize=6)
    axi.set_title('linear zoom: all descend, no minima', fontsize=7)

    # (b) the gamma window of C10
    fam = {'\u00d71 (tuned)': (e4['C10']['rows'], 'dimgray', ':'),
           '\u00d73': (garm['C10x3']['rows'], 'steelblue', '--')}
    if 'window_map' in e5:
        for tag, lab, sty in (('x5', '\u00d75 (monotone)', ('teal', '--')),
                              ('x7', '\u00d77 (monotone)', ('goldenrod', '--')),
                              ('x14', '\u00d714', ('tab:purple', '-.')),
                              ('x20', '\u00d720', ('tab:brown', '-.'))):
            if tag in e5['window_map']:
                fam[lab] = (e5['window_map'][tag]['rows'], *sty)
    fine = list(conf['C10_x10_fine']['rows'])
    if 'x10_fine_extra' in e5:
        fine += e5['x10_fine_extra']['rows']
    fine.sort(key=lambda r: r['omega'])
    fam['\u00d730'] = (garm['C10x30']['rows'], 'navy', '-.')
    XMAX, YW = 0.75, 0.05
    label_dx = {'\u00d720': (-18, 5), '\u00d714': (4, 2)}
    for lab, (rows, col, ls) in fam.items():
        om, de, _ = dE(rows)
        ax2.plot(om, de, ls, marker='.', ms=4, lw=1.1, color=col, alpha=0.85)
        vis = [(o, e) for o, e in zip(om, de) if o <= XMAX and abs(e) <= YW]
        if vis:
            dx = label_dx.get(lab, (4, -2))
            ax2.annotate(lab, vis[-1], fontsize=7.5, color=col,
                         textcoords='offset points', xytext=dx)
    om0 = fine[0]['E_total']
    omf = [r['omega'] for r in fine]
    def_ = [r['E_total'] - om0 for r in fine]
    ax2.plot(omf, def_, 'o-', ms=5, lw=2.0, color='tab:red', zorder=5)
    i = int(np.argmin(def_))
    ax2.annotate('\u00d710: the well', (omf[i], def_[i]), fontsize=9,
                 color='tab:red', fontweight='bold',
                 textcoords='offset points', xytext=(-30, -16))
    for tag, col in (('x14', 'tab:purple'), ('x20', 'tab:brown')):
        if 'window_map' in e5 and tag in e5['window_map']:
            om, de, _ = dE(e5['window_map'][tag]['rows'])
            j = int(np.argmin(de))
            ax2.plot([om[j]], [de[j]], 'v', ms=7, color=col, zorder=6,
                     markeredgecolor='k', markeredgewidth=0.5)
    cr = conf['C10_x10_drive_flip_control']['rows']
    om, de, _ = dE(cr)
    ax2.plot(om, de, 'd--', ms=5, lw=1.3, color='k')
    ax2.annotate('\u00d710, drive flipped', (om[2], de[2]), fontsize=8,
                 color='k', textcoords='offset points', xytext=(-96, 2))
    ax2.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax2.set_xlim(-0.02, 0.75)
    ax2.set_ylim(-0.05, 0.055)
    ax2.set_xlabel('\u03c9')
    ax2.set_ylabel('E(\u03c9) \u2212 E(0)')
    ax2.set_title('(b) C10: the two-sided \u03b3-window (multipliers of the '
                  'frozen-tuned \u03b3)\n\u25bc = minima of \u00d714, '
                  '\u00d720; the \u00d71 and \u00d730 curves exit the '
                  'frame', fontsize=9.5)
    fig.tight_layout()
    fig.savefig('results/fig_grid_ladders.png', bbox_inches='tight')


def fig2(conf, e5, db=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 4.4), dpi=150)
    fine = list(conf['C10_x10_fine']['rows'])
    if 'x10_fine_extra' in e5:
        fine += e5['x10_fine_extra']['rows']
    fine.sort(key=lambda r: r['omega'])
    om = [r['omega'] for r in fine]
    E0 = fine[0]['E_total']
    de = [(r['E_total'] - E0) * 1e3 for r in fine]
    dea = [(r['E_adam_level'] - fine[0]['E_adam_level']) * 1e3 for r in fine]
    ax1.plot(om, de, 'o-', ms=5, lw=1.8, color='tab:red', zorder=3,
             label='after L-BFGS')
    ax1.plot(om, dea, 's--', ms=3, lw=0.9, color='dimgray', zorder=4,
             label='Adam level')
    cr = conf['C10_x10_drive_flip_control']['rows']
    omc = [r['omega'] for r in cr]
    dec = [(r['E_total'] - cr[0]['E_total']) * 1e3 for r in cr]
    ax1.plot(omc, dec, 'd--', ms=5, lw=1.2, color='k', label='drive flipped')
    ax1.annotate('+41 at \u03c9=0.55 \u2192', (0.26, 8.6), fontsize=7.5,
                 color='k')
    ax1.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax1.set_ylim(-3.4, 9.5)
    ax1.set_xlabel('\u03c9')
    ax1.set_ylabel('E(\u03c9) \u2212 E(0)   [\u00d710\u207b\u00b3]')
    ax1.set_title('(a) C10 \u00d710 well, standard protocol '
                  '(Adam 300 + 1 L-BFGS);\nupper curves clipped',
                  fontsize=9.5)
    ax1.legend(fontsize=8, loc='upper left')

    dw = e5.get('deep_well')
    if dw:
        dpl = np.array(dw['depth_per_level']) * 1e3
        lv = np.arange(len(dpl))
        tol = 0.1 * abs(dpl[-1])
        ax2.axhspan(dpl[-1] - tol, dpl[-1] + tol, color='tab:red',
                    alpha=0.12, lw=0)
        ax2.plot(lv, dpl, 'o-', lw=1.5, color='tab:red')
        ax2.axhline(-2.13, color='dimgray', lw=1.0, ls='--')
        ax2.annotate('standard-protocol value at \u03c9=0.2 (panel a): '
                     '\u22122.13', (0.1, -2.16), fontsize=7.5,
                     color='dimgray')
        ax2.annotate('shaded: 008 plateau tolerance (10% of depth);\n'
                     'spread over L-BFGS levels: 1.5%', (1.6, -2.36),
                     fontsize=7.5, color='tab:red')
        ax2.set_xticks(lv)
        ax2.set_xticklabels(['Adam 500'] + [f'L-BFGS {i+1}'
                                            for i in range(len(lv) - 1)],
                            fontsize=8)
        ax2.set_ylim(-2.55, -2.05)
        ax2.set_ylabel('E(\u03c9=0.2) \u2212 E(0)   [\u00d710\u207b\u00b3]')
        ax2.set_title('(b) SEPARATE deep-protocol run (Adam 500 + 4 L-BFGS):\n'
                      'value at probe \u03c9=0.2 per level \u2014 settled',
                      fontsize=9.5)
    if db:
        rungs = sorted(db['rungs'], key=float)
        L0 = db['rungs']['0.0']['levels']
        nlev = len(L0)
        cols = {'0.1': 'tab:blue', '0.15': 'tab:red', '0.2': 'tab:purple',
                '0.28': 'dimgray'}
        for om in rungs:
            if om == '0.0':
                continue
            lv = db['rungs'][om]['levels']
            rel = [(l - l0) * 1e3 for l, l0 in zip(lv, L0)]
            ax3.plot(range(nlev), rel, 'o-', ms=4, lw=1.4,
                     color=cols.get(om, 'k'), label=f'\u03c9={om}')
        ax3.axhline(0, color='k', lw=0.6, alpha=0.5)
        ax3.set_xticks(range(nlev))
        ax3.set_xticklabels(['Adam'] + [f'L{i+1}' for i in range(nlev - 1)],
                            fontsize=8)
        ax3.set_ylabel('E(\u03c9) \u2212 E(0), same level   [\u00d710\u207b\u00b3]')
        ax3.set_title('(c) \u00d714 deep bracket (Adam 500 + 6 L-BFGS):\n'
                      'interior minimum at 0.15 for four levels,\n'
                      'then \u03c9=0.28 (unconverged) overtakes \u2014 '
                      'NOT certified', fontsize=9)
        ax3.legend(fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig('results/fig_well_anatomy.png', bbox_inches='tight')


def main():
    e4 = load('e4_cells')
    garm = load('e4_gamma_arm')
    conf = load('e4_confirm')
    try:
        e5 = load('e5_arms')
    except FileNotFoundError:
        e5 = {}
    try:
        db = load('e5_deep_bracket')
    except FileNotFoundError:
        db = None
    fig1(e4, garm, e5, conf)
    fig2(conf, e5, db)
    print('figures -> results/fig_grid_ladders.png, results/fig_well_anatomy.png')


if __name__ == '__main__':
    main()
