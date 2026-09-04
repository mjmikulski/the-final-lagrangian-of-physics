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
import numpy as np

def load(n):
    with open(f'results/{n}.json') as f:
        return json.load(f)

def fig1():
    nt = load('null_test'); ks = load('static_kernel_signs')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=150,
                                   gridspec_kw={'width_ratios': [2.2, 1]})
    names = list(nt.keys())
    vals = [max(nt[n]['el_ratio'], 1e-18) for n in names]
    cols = ['tab:gray' if nt[n]['null'] else ('tab:blue' if nt[n]['label'].startswith('02') else 'tab:purple') for n in names]
    cols = ['tab:red' if n.startswith('CONTROL') else c for n, c in zip(names, cols)]
    ax1.bar(range(len(names)), vals, color=cols)
    ax1.set_yscale('log'); ax1.axhline(1e-8, color='k', lw=0.7, ls='--')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([nt[n]['label'].replace('02-13', '') for n in names], rotation=90, fontsize=6)
    ax1.set_ylabel('|EL| / scale')
    ax1.set_title('(a) Euler–Lagrange test: null threshold 1e-8; red = controls (I₁, φ),\n'
                  'gray = null (φ, χ), blue = even, purple = odd decorated classes', fontsize=9)
    labs = [v['label'] for v in ks.values()]
    ax2.bar(range(len(labs)), [v['n_pos'] for v in ks.values()], color='tab:green', label='positive')
    ax2.bar(range(len(labs)), [-v['n_neg'] for v in ks.values()], color='tab:red', label='negative')
    ax2.set_xticks(range(len(labs))); ax2.set_xticklabels([l.replace('02-13', '') for l in labs], rotation=60, fontsize=7)
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_ylabel('eigenvalue count (18×18 static kernel)')
    ax2.set_title('(b) 3×3-alive classes: traceless, indefinite kernels', fontsize=9)
    ax2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig('results/fig_linear_family.png', bbox_inches='tight')

def fig2():
    routes = [('A: vacuum-pinned', 'lattice_linear_A', 'tab:orange'), ('B: spectral, differentiable', 'lattice_linear_B', 'tab:blue')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), dpi=150)
    any_data = False
    for lab, fn, col in routes:
        if not os.path.exists(f'results/{fn}.json'):
            continue
        lt = load(fn); any_data = True
        runs = {k: r for k, r in lt.items() if isinstance(r, dict) and 'status' in r}
        base = runs.get('baseline')
        if base is None or base['status'] != 'ok':
            continue
        markers = {}
        for k, r in runs.items():
            if k == 'baseline':
                continue
            cls = k.split('_')[0]
            mk = markers.setdefault(cls, ['o', 's', '^', 'D', 'v'][len(markers)])
            w = r['lambda'] * lt['base_integrals'][cls] / lt['E_stat_base']
            ok = r['status'] == 'ok'
            relaxed = ok and abs(r.get('continuation_dE', 1.0)) < 0.01 * max(abs(r['E_total'] - base['E_total']), 1e-12)
            kw = dict(color=col, marker=mk, ms=6, ls='none', mfc=col if relaxed else 'none', mew=1.2)
            axes[0].plot(w, r['E_total'] - base['E_total'], **kw)
            if ok:
                axes[1].plot(w, r['tail_eta'], **kw)
                axes[2].plot(w, r['small_gap_min'], **kw)
            else:
                axes[0].annotate(r['status'][:8], (w, r['E_total'] - base['E_total']), fontsize=6, color=col)
            rc_file = f"results/restart_check_{fn[-1]}.json"
            if os.path.exists(rc_file):
                rc = load(f"restart_check_{fn[-1]}")
                if k in rc:
                    axes[0].plot(w, rc[k]['E_restart'] - base['E_total'], marker='x', color=col, ms=7, mew=1.5, ls='none')
        for ax, key in ((axes[1], 'tail_eta'), (axes[2], 'small_gap_min')):
            ax.axhline(base[key], color=col, lw=0.8, ls=':')
        for cls, mk in markers.items():
            axes[0].plot([], [], color=col, marker=mk, ls='none', label=f'{lab}: {cls}')
    axes[0].plot([], [], color='k', marker='x', ls='none', label='spatial-block perturbed restart')
    axes[0].axhline(0, color='k', lw=0.6)
    axes[0].set_xlabel('λ·∫dens / E_stat  (weight on the base profile)'); axes[0].set_ylabel('E_total − E_baseline')
    axes[0].set_title('(a) relaxed total energy (filled = continuation gate passed)', fontsize=9)
    axes[1].set_xlabel('weight'); axes[1].set_ylabel('far-field exponent of the η density, r ∈ [8,16]')
    axes[1].set_title('(b) tail exponent (dotted: baseline)', fontsize=9)
    axes[2].set_xlabel('weight'); axes[2].set_ylabel('min gap of the small eigenvalue pair')
    axes[2].set_title('(c) eigenvalue-exchange gap (dotted: baseline)', fontsize=9)
    axes[0].legend(fontsize=6, loc='best')
    if not any_data:
        axes[0].text(0.5, 0.5, 'no lattice records', transform=axes[0].transAxes, ha='center')
    fig.tight_layout(); fig.savefig('results/fig_lattice_lambda.png', bbox_inches='tight')

if __name__ == '__main__':
    fig1(); fig2(); print('figures -> results/fig_linear_family.png, results/fig_lattice_lambda.png')
