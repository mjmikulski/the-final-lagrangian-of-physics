"""Figures of report 014 from the committed JSON records.

fig_family.png        : Euler-Lagrange test per linear class (null controls), and the static kernels of the 3x3-alive classes
fig_vacuum_saddle.png : the uniform vacuum under a generic compact frame twist -- cubic odd part of the linear integral against
                        the quartic eta energy, the cubic coefficients per class and seed, and the cubic-quartic energy along a ray
fig_electron_scan.png : the lambda-scan on the eta-relaxed electron (route A): energy shift against the frozen value, far-field
                        exponent and eigenvalue-exchange gap
fig_e0_scan.png       : second route in the E0 = 100 conventions -- the one-parameter family X = c X_a of the electron sector
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

R = 'results'
MARK = {'P1P2': 'o', 'P1P3': 's', 'P2P3': '^'}
CLASS_LABEL = {'P1P2': '$F_{12}$', 'P1P3': '$F_{13}$', 'P2P3': '$F_{23}$'}


def load(n, d=R):
    with open(os.path.join(d, n + '.json')) as f:
        return json.load(f)


def pretty(label):
    return label.replace('02-13', '').replace('03-12', '').replace('eta', '$\\eta$').replace('eps[', '$\\varepsilon$[')


def fig_family():
    nt = load('null_test'); ks = load('static_kernel_signs')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={'width_ratios': [2.4, 1]})
    names = list(nt.keys())
    vals = [max(nt[n]['el_ratio'], 1e-18) for n in names]
    def color(n):
        if n.startswith('CONTROL'):
            return 'tab:red'
        if nt[n]['null']:
            return 'tab:gray'
        return 'tab:blue' if nt[n]['label'].startswith('02') else 'tab:purple'
    ax1.bar(range(len(names)), vals, color=[color(n) for n in names])
    ax1.set_yscale('log'); ax1.set_ylim(1e-19, 5)
    ax1.axhline(1e-8, color='k', lw=0.8, ls='--')
    ax1.text(len(names) - 0.5, 2e-8, 'null threshold $10^{-8}$', ha='right', va='bottom', fontsize=8)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([('control: ' if n.startswith('CONTROL') else '') + (pretty(nt[n]['label']) if n != 'CONTROL_I1' else 'I₁')
                         for n in names], rotation=90, fontsize=7)
    ax1.set_ylabel('|Euler–Lagrange expression| / scale (dimensionless)')
    ax1.set_title('(a) every class with a projector coefficient is dynamical;\nthe constant-coefficient φ and χ are null (total derivatives)', fontsize=10)
    ax1.legend(handles=[Patch(color='tab:red', label='controls: I₁ (dynamical), φ (null)'),
                        Patch(color='tab:gray', label='constant coefficients: φ, χ (null, report 005; values clipped at $10^{-18}$)'),
                        Patch(color='tab:blue', label='even classes with projectors'),
                        Patch(color='tab:purple', label='odd (ε) classes with projectors')],
               fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    labs = [('φ = [$\\eta$,$\\eta$]' if v['label'].endswith('[eta,eta]') else ('$F_{%s%s}$' % (v['label'][-5], v['label'][-2]) if '[P' in v['label'] else pretty(v['label']))) for v in ks.values()]
    pos = [v['n_pos'] for v in ks.values()]; neg = [v['n_neg'] for v in ks.values()]
    ax2.bar(range(len(labs)), pos, color='tab:green', label='positive eigenvalues')
    ax2.bar(range(len(labs)), [-x for x in neg], color='tab:red', label='negative eigenvalues')
    for i, (p, q) in enumerate(zip(pos, neg)):
        ax2.text(i, p + 0.2, f'{18 - p - q} zero', ha='center', fontsize=7, color='0.3')
    ax2.set_xticks(range(len(labs))); ax2.set_xticklabels(labs, rotation=60, fontsize=8)
    yt = [-8, -4, 0, 4, 8]; ax2.set_yticks(yt); ax2.set_yticklabels([str(t) for t in yt])
    ax2.axhline(0, color='k', lw=0.6); ax2.set_ylim(-9.5, 10.5)
    ax2.set_ylabel('18×18 kernel eigenvalues: positive (up), negative (down)')
    ax2.set_title('(b) on the 3×3 sector the surviving classes\nare traceless and indefinite', fontsize=10)
    ax2.legend(fontsize=8, loc='lower right')
    fig.suptitle('The linear-in-F family: 675 diagrams, 38 classes, 12 generators (6 even + 6 odd)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(R, 'fig_family.png'), dpi=150)


def fig_vacuum_saddle():
    tg = load('twist_generic_A')
    seeds = sorted({int(k.split('seed')[1]) for k in tg})
    classes = ['P1P2', 'P1P3', 'P2P3']
    ts = sorted(float(t) for t in next(iter(tg.values())) if t.replace('.', '').replace('e-', '').isdigit() or t.startswith('0.'))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    cols = {456: 'C0', 7: 'C1', 99: 'C2'}
    for k, v in tg.items():
        cname, seed = k.split('_seed'); seed = int(seed)
        odd = [abs(v[str(t)]['lin_plus'] - v[str(t)]['lin_minus']) / 2 for t in ts]
        even = [(v[str(t)]['E_plus'] + v[str(t)]['E_minus']) / 2 for t in ts]
        ax[0].loglog(ts, odd, marker=MARK[cname], color=cols.get(seed, 'k'), ls='-', ms=5, lw=1)
        if cname == 'P1P2':
            ax[0].loglog(ts, even, marker='x', color=cols.get(seed, 'k'), ls='--', ms=5, lw=1)
    tt = np.array(ts)
    ax[0].loglog(tt, 0.2 * tt ** 3, 'k-', lw=1, label='∝ t³ (guide)')
    ax[0].loglog(tt, 50 * tt ** 4, 'k--', lw=1, label='∝ t⁴ (guide)')
    ax[0].set(xlabel='twist amplitude t', ylabel='integral (model units)',
              title='(a) odd part of the linear integral ∝ t³ (solid),\neven part of the η static energy ∝ t⁴ (dashed)')
    h = [Line2D([], [], color='k', marker=MARK[c], ls='none', label=f'class {CLASS_LABEL[c]}') for c in classes]
    h += [Line2D([], [], color=cols[s], lw=2, label=f'twist seed {s}') for s in seeds if s in cols]
    h += [Line2D([], [], color='k', lw=1, label='∝ t³'), Line2D([], [], color='k', lw=1, ls='--', label='∝ t⁴')]
    ax[0].legend(handles=h, fontsize=7, loc='lower right')
    ax[0].set_xticks(ts); ax[0].set_xticklabels([f'{t:g}' for t in ts]); ax[0].minorticks_off()
    # (b) c3 per class and seed, and the null combination
    w = 0.2
    for i, s in enumerate(seeds):
        c3 = [tg[f'{c}_seed{s}']['0.001']['c3_lin'] for c in classes]
        ax[1].bar(np.arange(3) + (i - 1) * w, c3, w, color=cols.get(s, 'k'), label=f'seed {s}')
        ax[1].bar(3 + (i - 1) * w, sum(c3), w, color=cols.get(s, 'k'), edgecolor='k', hatch='//')
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([CLASS_LABEL[c] for c in classes] + ['$F_{12}+F_{13}+F_{23}$\n(= φ/2, null)'])
    ax[1].set(ylabel='cubic coefficient $c_3$ of the linear integral', title='(b) the cubic coefficient is nonzero for every class and twist,\nand cancels exactly in the null combination')
    ax[1].legend(fontsize=8)
    # (c) cubic-quartic energy along one ray at the 5% coupling
    v = tg['P1P2_seed456']; lam = v['lambda5']; c3 = v['0.001']['c3_lin']; c4 = v['0.001']['c4_eta']
    tstar = -3 * lam * c3 / (4 * c4)
    t = np.linspace(-1.6, 1.1, 400) * abs(tstar)
    E = lam * c3 * t ** 3 + c4 * t ** 4
    ax[2].plot(t / abs(tstar), E / abs(lam * c3 * tstar ** 3), color='C3', lw=1.8, label='λ c₃ t³ + c₄ t⁴ along the ray')
    ax[2].axhline(0, color='k', lw=0.6); ax[2].axvline(0, color='k', lw=0.6)
    ax[2].set(xlabel='t / |t*|,   t* = −3λc₃/(4c₄)', ylabel='energy along the ray (units of |λc₃t*³|)', ylim=(-0.5, 2.5),
              title='(c) energy along one ray at the 5% coupling:\nnegative for small t of one sign ($F_{12}$, seed 456)')
    ax[2].legend(fontsize=8, loc='upper center')
    ax[2].annotate('E = −¼ |λc₃t*³| at t = t*', xy=(-1, -0.25), xytext=(-1.5, 0.9), fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
    for a in ax:
        a.grid(alpha=0.3, which='both')
    fig.suptitle('The uniform vacuum under a generic compact frame twist M = R M_vac Rᵀ, R = exp(t W(x)) (32³ lattice, three random W)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(R, 'fig_vacuum_saddle.png'), dpi=150)


def fig_electron_scan():
    lt = load('lattice_linear_A'); rc = load('restart_check_A') if os.path.exists(os.path.join(R, 'restart_check_A.json')) else {}
    runs = {k: r for k, r in lt.items() if isinstance(r, dict) and 'status' in r}
    base = runs['baseline']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for k, r in runs.items():
        if k == 'baseline':
            continue
        cls = k.split('_')[0]
        w = r['lambda'] * lt['base_integrals'][cls] / lt['E_stat_base']
        kw = dict(color='C0', marker=MARK[cls], ms=7, ls='none')
        jit = {'P1P2': -0.006, 'P1P3': 0.0, 'P2P3': 0.006}[cls]
        ax[0].plot(w + jit, r['E_total'] - base['E_total'], **kw)
        ax[1].plot(w, r['tail_eta'], **kw)
        ax[2].plot(w, 1e3 * r['small_gap_min'], **kw)
        if k in rc:
            ax[0].plot(w + jit + 0.012, rc[k]['E_restart'] - base['E_total'], marker='x', color='k', ms=6, mew=1.2, ls='none')
    ww = np.linspace(-0.22, 0.22, 10)
    ax[0].plot(ww, ww * lt['E_stat_base'], 'k--', lw=1, label='frozen value λ·∫dens (no relaxation)')
    ax[0].set(xlabel='weight λ·∫dens / E_static on the base profile', ylabel='E(λ) − E(0) (model units)',
              title='(a) the energy shift follows the frozen value;\nthe relaxed field pulls it back toward zero')
    ax[1].axhline(base['tail_eta'], color='k', ls=':', lw=1, label='baseline (λ = 0)')
    ax[1].set(xlabel='weight λ·∫dens / E_static', ylabel='far-field exponent of the η density (shells r = 8–16)', title='(b) the far-field exponent moves with the weight,\nclass-dependently')
    ax[2].axhline(1e3 * base['small_gap_min'], color='k', ls=':', lw=1, label='baseline (λ = 0)')
    ax[2].set(xlabel='weight λ·∫dens / E_static', ylabel='smallest gap of the small eigenvalue pair (×10⁻³)', title="(c) the core's eigenvalue-exchange gap narrows\nbut does not close")
    h = [Line2D([], [], color='C0', marker=MARK[c], ls='none', label=f'class {CLASS_LABEL[c]}') for c in MARK]
    ax[0].legend(handles=h + [Line2D([], [], color='k', ls='--', label='frozen value'), Line2D([], [], color='k', marker='x', ls='none', label='perturbed restart (20% runs)')], fontsize=8, loc='upper left', title='classes offset slightly in x', title_fontsize=7)
    for a in ax[1:]:
        a.legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle('The three surviving classes on the η-relaxed electron (32³ lattice, (g, δ) = (8, 0.3)): a sign-weighted reweighting of the static density', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(R, 'fig_electron_scan.png'), dpi=150)


def fig_e0_scan():
    rows = sorted(load('task4_scan', 'e0_route/results'), key=lambda r: r['ca'])
    c = np.array([r['ca'] for r in rows]); E = np.array([r['E'] for r in rows])
    ok = np.array([r['grad'] < 1e-4 for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.6))
    ax[0].plot(c[ok], E[ok], 'o-', color='C0', label='total energy E (converged)')
    for cc in c[~ok]:
        ax[0].annotate('c = %g:\nno converged\nsolution' % cc, xy=(cc, 0), xytext=(cc, 28), ha='center', fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax[0].set_xlim(-1.15, 1.15)
    ax[0].plot(c[ok], [r['E4'] for r in rows if r['grad'] < 1e-4], 's--', color='C2', ms=4, label='quartic gradient term')
    ax[0].plot(c[ok], [r['EV'] for r in rows if r['grad'] < 1e-4], '^--', color='C1', ms=4, label='potential')
    ax[0].plot(c[ok], [r['E2'] for r in rows if r['grad'] < 1e-4], 'v--', color='C4', ms=4, label='−c ∫X_a (the linear term)')
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set(xlabel='coupling c of X = c X_a', ylabel='energy in box 12 (model units)', title='(a) for |c| ≤ 0.5 the mass changes smoothly and stays\npositive; at c = ±1 the minimisation finds no solution')
    ax[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    for i, sh in enumerate(('inner', 'middle', 'outer')):
        ax[1].plot(c[ok], [r['tail_ratio'][i] for r in rows if r['grad'] < 1e-4], 'o-', ms=4, label=f'{sh} of the three outer shells')
    ax[1].axhline(1, color='k', ls='--', lw=1, label='Coulomb law of the hedgehog')
    ax[1].set(xlabel='coupling c of X = c X_a', ylabel='far-field density / 4Δ⁴ r⁻⁴', ylim=(0.7, 1.3), title='(b) the far field stays within 20% of the Coulomb law\nfor |c| ≤ 0.5')
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle('Second route, E₀ = 100 conventions: the constant-coefficient family of the electron sector is X = c X_a (n = 32, box 12)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(R, 'fig_e0_scan.png'), dpi=150)


if __name__ == '__main__':
    fig_family(); fig_vacuum_saddle(); fig_electron_scan(); fig_e0_scan()
    print('figures written')
