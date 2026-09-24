"""Figures of report 017 from results/radial_profile.json (no lattice work).

fig_kinetic_rank.png : the ten eigenvalues of the kinetic form K along a ray, frozen hedgehog and time sector free
fig_speeds.png       : squared characteristic speeds along the same ray, radial and tangential propagation
                       (the two directions are offset by -0.04 and +0.04 in r so that coinciding points stay visible)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#222222', '#8a8a86'
FLOOR = 1e-16
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
d = json.load(open('results/radial_profile.json'))['fields']
TITLES = {'frozen hedgehog': 'time sector frozen (hedgehog)',
          'time sector free (E0 = 100)': 'time sector free (screened endpoint, report 016)'}


def fig_rank():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for a, (name, rows) in zip(ax, d.items()):
        r = np.array([row['r'] for row in rows])
        lam = np.clip(np.abs(np.array([row['K_delta'] for row in rows])), FLOOR, None)
        zeros = [j for j in range(10) if lam[:, j].max() <= 1e-12]
        for j in range(10):
            if j not in zeros:
                a.semilogy(r, lam[:, j], color=BLUE, lw=1.2, label='nonzero eigenvalues' if j == 0 else None)
        for n, j in enumerate(zeros):
            a.semilogy(r, np.full_like(r, FLOOR) * 2.5 ** n, color=ORANGE, lw=0, marker='os'[n], ms=5,
                       label='exact zeros (drawn at the floor)' if n == 0 else None)
        a.axhspan(FLOOR / 3, 1e-13, color=MUTED, alpha=0.15, lw=0)
        a.text(0.15, 2e-13, '|λ| < 10⁻¹³: numerically zero', fontsize=10, color=INK, va='bottom')
        a.text(5.4, 1e-8, f'{10 - len(zeros)} nonzero, {len(zeros)} zero\n(rank {10 - len(zeros)})', ha='right',
               va='center', color=INK, fontsize=11)
        a.set(xlabel='distance from the centre r (model units)', title=TITLES[name], ylim=(FLOOR / 3, 3), xlim=(0, 5.6))
    ax[0].set_ylabel('|eigenvalue of K| / largest')
    ax[0].legend(loc='center left', bbox_to_anchor=(0.0, 0.62), fontsize=10, frameon=False)
    fig.suptitle('Kinetic form K = ∂²L/∂Ṁ²: two exact zeros (trace, P₀) with the time sector frozen,\n'
                 'one (trace) with it free', fontsize=12)
    fig.tight_layout()
    fig.savefig('results/fig_kinetic_rank.png', dpi=150)


def fig_speeds():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for a, (name, rows) in zip(ax, d.items()):
        for lab, col, mk, off in (('radial', BLUE, 'o', -0.04), ('tangential', ORANGE, 's', 0.04)):
            rr = [row['r'] + off for row in rows for _ in row['speed2'][lab]]
            ss = [s for row in rows for s in row['speed2'][lab]]
            a.scatter(rr, ss, s=16, color=col, marker=mk, label=f'propagation {lab}', edgecolor='white', linewidth=0.5,
                      zorder=3)
        a.axhspan(1, 1.25, color=MUTED, alpha=0.15, lw=0)
        a.axhspan(-0.25, 0, color=MUTED, alpha=0.15, lw=0)
        a.axhline(1, color=MUTED, lw=0.8)
        a.axhline(0, color=MUTED, lw=0.8)
        a.text(0.1, 1.08, 'excluded: faster than light', color=INK, fontsize=10)
        a.text(0.1, -0.17, 'excluded: gradient instability', color=INK, fontsize=10)
        a.set(xlabel='distance from the centre r (model units)', title=TITLES[name], ylim=(-0.25, 1.25), xlim=(0, 5.6))
    ax[0].set_ylabel('squared speed v²/c²')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, fontsize=11)
    fig.suptitle('Characteristic speeds on the range of K: all in [0, 1], as the symbol identity requires', fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig('results/fig_speeds.png', dpi=150)


if __name__ == '__main__':
    fig_rank()
    fig_speeds()
