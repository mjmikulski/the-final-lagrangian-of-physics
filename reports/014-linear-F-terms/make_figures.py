"""Report 014 figures from committed JSONs.

Fig 1 (fig_linear_family.png): (a) Euler-Lagrange ratio per linear class with
the null controls; (b) static-kernel signatures of the 3x3-alive classes.
Fig 2 (fig_lattice_lambda.png): lattice lambda-scan, routes A and B -- energy
change, far-field exponent and the small-pair gap versus the linear term's
weight on the base profile.
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

MARK = {'P1P2': 'o', 'P1P3': 's', 'P2P3': '^', 'F1Q': 'D', 'FQQ': 'v', 'F1T': 'P', 'FtQ': '*'}
WLABEL = 'weight  λ·∫dens / E_stat  (on the base profile)'


def load(n):
    with open(f'results/{n}.json') as f:
        return json.load(f)


def pretty(label):
    return label.replace('02-13', '').replace('eta', 'η').replace('eps[', 'ε[')


def fig1():
    nt = load('null_test'); ks = load('static_kernel_signs')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150,
                                   gridspec_kw={'width_ratios': [2.3, 1]})
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
    ax1.axhline(1e-8, color='k', lw=0.7, ls='--')
    ax1.text(len(names) - 0.5, 2e-8, 'null threshold 1e-8', ha='right', va='bottom', fontsize=7)
    ax1.text(len(names) - 0.5, 2e-18, 'values clipped at 1e-18', ha='right', fontsize=6, color='0.4')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([('ctrl: ' if n.startswith('CONTROL') else '') + (pretty(nt[n]['label']) if n != 'CONTROL_I1' else 'I₁')
                         for n in names], rotation=90, fontsize=7)
    ax1.set_ylabel('|EL| / scale')
    ax1.set_title('(a) Euler–Lagrange test per linear class', fontsize=10)
    ax1.legend(handles=[Patch(color='tab:red', label='controls (I₁ dynamical, φ null)'),
                        Patch(color='tab:gray', label='constant-coefficient φ, χ (null, report 005)'),
                        Patch(color='tab:blue', label='even classes with projectors'),
                        Patch(color='tab:purple', label='odd (ε) classes with projectors')],
               fontsize=7, loc='center right', bbox_to_anchor=(0.995, 0.30))
    labs = [pretty(v['label']) for v in ks.values()]
    pos = [v['n_pos'] for v in ks.values()]; neg = [v['n_neg'] for v in ks.values()]
    ax2.bar(range(len(labs)), pos, color='tab:green')
    ax2.bar(range(len(labs)), [-x for x in neg], color='tab:red')
    for i, (p, q) in enumerate(zip(pos, neg)):
        ax2.text(i, p + 0.2, f'{18 - p - q} zero', ha='center', fontsize=6, color='0.3')
    ax2.set_xticks(range(len(labs))); ax2.set_xticklabels(labs, rotation=60, fontsize=7)
    yt = [-8, -4, 0, 4, 8]; ax2.set_yticks(yt); ax2.set_yticklabels([str(abs(t)) for t in yt])
    ax2.axhline(0, color='k', lw=0.6); ax2.set_ylim(-9.5, 10.5)
    ax2.set_ylabel('count of positive (up) / negative (down)\neigenvalues of the 18×18 static kernel')
    ax2.set_title('(b) 3×3-alive classes: traceless, indefinite', fontsize=10)
    fig.tight_layout(); fig.savefig('results/fig_linear_family.png', bbox_inches='tight')


def fig2():
    routes = [('route A: vacuum-pinned', 'lattice_linear_A', 'tab:orange'),
              ('route B: spectral, differentiable', 'lattice_linear_B', 'tab:blue')]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=150)
    any_data = False; seen = {}
    for lab, fn, col in routes:
        if not os.path.exists(f'results/{fn}.json'):
            continue
        lt = load(fn); any_data = True
        runs = {k: r for k, r in lt.items() if isinstance(r, dict) and 'status' in r}
        base = runs.get('baseline')
        if base is None or base['status'] != 'ok':
            continue
        rc = load(f'restart_check_{fn[-1]}') if os.path.exists(f'results/restart_check_{fn[-1]}.json') else {}
        for k, r in runs.items():
            if k == 'baseline':
                continue
            cls = k.split('_')[0]; seen[cls] = True
            w = r['lambda'] * lt['base_integrals'][cls] / lt['E_stat_base']
            ok = r['status'] == 'ok'
            eff = abs(r['E_total'] - base['E_total'])
            relaxed = ok and r.get('grad_inf_free', 1.0) <= 0.1 and (eff < 1e-9 or abs(r.get('continuation_dE', 1.0)) < 0.01 * eff)
            kw = dict(color=col, marker=MARK[cls], ms=7, ls='none', mfc=col if relaxed else 'none', mew=1.3)
            axes[0].plot(w, r['E_total'] - base['E_total'], **kw)
            if ok:
                axes[1].plot(w, r['tail_eta'], **kw)
                axes[2].plot(w, 1e3 * r['small_gap_min'], **kw)
            if k in rc:
                axes[0].plot(w + 0.008, rc[k]['E_restart'] - base['E_total'], marker='x', color='k', ms=6, mew=1.2, ls='none')
        for ax, key, sc in ((axes[1], 'tail_eta', 1.0), (axes[2], 'small_gap_min', 1e3)):
            ax.axhline(sc * base[key], color=col, lw=0.8, ls=':')
    for ax in axes:
        ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2]); ax.set_xlabel(WLABEL, fontsize=8)
    axes[0].axhline(0, color='k', lw=0.6)
    axes[0].set_ylabel('E_total − E_baseline'); axes[0].set_title('(a) total energy after the protocol', fontsize=10)
    axes[1].set_ylabel('far-field exponent of the η density, r ∈ [8,16]'); axes[1].set_title('(b) tail exponent', fontsize=10)
    axes[2].set_ylabel('min gap of the small eigenvalue pair (×10⁻³)'); axes[2].set_title('(c) eigenvalue-exchange gap', fontsize=10)
    handles = [Line2D([], [], color=c, lw=6, label=l) for l, _, c in routes]
    handles += [Line2D([], [], color='k', marker=MARK[c], ls='none', mfc='none', label=c) for c in MARK if c in seen]
    handles += [Line2D([], [], color='k', marker='o', ls='none', label='filled: relaxed (|∇E|∞ ≤ 0.1, continuation gate)'),
                Line2D([], [], color='k', marker='o', ls='none', mfc='none', label='open: stalled / gate failed'),
                Line2D([], [], color='k', marker='x', ls='none', label='spatial-block perturbed restart'),
                Line2D([], [], color='k', ls=':', label='dotted: baseline (λ = 0)')]
    fig.legend(handles=handles, fontsize=7, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    if not any_data:
        axes[0].text(0.5, 0.5, 'no lattice records', transform=axes[0].transAxes, ha='center')
    fig.tight_layout(); fig.savefig('results/fig_lattice_lambda.png', bbox_inches='tight')


if __name__ == '__main__':
    fig1(); fig2(); print('figures -> results/fig_linear_family.png, results/fig_lattice_lambda.png')
