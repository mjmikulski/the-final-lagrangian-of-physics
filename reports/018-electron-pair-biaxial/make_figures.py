"""Figures of report 018 from the committed JSON records (no lattice work).

fig_strand.png  : tension of the straight strand against beta (2D lattice, k = 1/2 and 1) with the Bogomolny line
                  (2 sqrt2 pi / 3) k beta^3, and the radial splitting profile against the Bogomolny profile
fig_charge.png  : one charge in the biaxial vacuum beta = 0.3: e_2 - e_3 on the plane x = 0 (four half-lines)
fig_pairs.png   : the held pair under the gradient flow: energy and the charge of the held cores against steps
"""
import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#c98500', '#1a1a19', '#6b6a63'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    tmp = os.path.join(R, name + '.tmp.png')
    fig.savefig(tmp, dpi=150)
    os.replace(tmp, os.path.join(R, name))


def bogomolny_profile(beta, k, rmax):
    b0 = beta / 2
    sol = solve_ivp(lambda s, q: [math.sqrt(2) * (b0 - math.sqrt(max(q[0], 0))) / (4 * k)], [0, rmax ** 2 / 2], [0.0],
                    max_step=1e-4, dense_output=True)
    r = np.linspace(0, rmax, 400)
    return r, 2 * np.sqrt(np.maximum(sol.sol(r ** 2 / 2)[0], 0))


def fig_strand():
    rows = json.load(open(os.path.join(R, 'strand2d.json')))
    main = [r for r in rows if abs(r['R'] / r['r_half_bogomolny'] - 8 * (0.5 if r['k'] == 1.0 else 1)) < 1e-6 or True]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))
    bb = np.logspace(-2.1, -0.4, 50)
    ax.loglog(bb, 2 * math.sqrt(2) * math.pi / 3 * 0.5 * bb ** 3, color=MUTED, lw=1.2, label='Bogomolny bound, k = 1/2')
    ax.loglog(bb, 2 * math.sqrt(2) * math.pi / 3 * 1.0 * bb ** 3, color=MUTED, lw=1.2, ls='--', label='Bogomolny bound, k = 1')
    seen = set()
    for r in rows:
        key = (r['beta'], r['k'])
        if key in seen:
            continue                     # the first row per (beta, k) is the scan; later rows are the box/grid checks
        seen.add(key)
        col, mk = (BLUE, 'o') if r['k'] == 0.5 else (ORANGE, 's')
        ax.loglog(r['beta'], r['T'], mk, color=col, ms=7, mec='white', mew=1,
                  label=('lattice, half-disclination k = 1/2' if r['k'] == 0.5 else 'lattice, full disclination k = 1')
                  if (r['beta'] == 0.3) else None)
    ax.set(xlabel='vacuum biaxiality β  (spectrum 1, β, 0)', ylabel='tension T (energy per unit length)',
           title='strand tension T = (2√2π/3) k β³, k = 1/2 and k = 1')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.title.set_fontsize(11)
    for r in rows:
        if r['k'] == 0.5 and r['beta'] in (0.3, 0.03) and 'profile_r' in r and r is next(x for x in rows if x['k'] == 0.5 and x['beta'] == r['beta']):
            col = BLUE if r['beta'] == 0.3 else AQUA
            rh = r['r_half_bogomolny']
            bx.plot(np.array(r['profile_r']) / rh, np.array(r['profile_split']) / r['beta'], 'o', color=col, ms=5,
                    mfc='none', mew=1.2, label=f'lattice, β = {r["beta"]:g}', zorder=2)
            rr, sp = bogomolny_profile(r['beta'], 0.5, 6 * rh)
            bx.plot(rr / rh, sp / r['beta'], '-', color=col, lw=1.6, label=f'Bogomolny profile, β = {r["beta"]:g}', zorder=3)
    bx.set(xlabel='distance from the line / ρ½  (ρ½ ∝ √β)', ylabel='transverse splitting (e₂ − e₃) / β', xlim=(0, 5),
           title='core profile of the half-disclination')
    bx.legend(frameon=False, fontsize=9, loc='lower right')
    bx.title.set_fontsize(11)
    fig.tight_layout()
    save(fig, 'fig_strand.png')


def fig_charge():
    d = json.load(open(os.path.join(R, 'single_strands.json')))
    s = d['slice_x0']
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(np.array(s['split']).T, origin='lower', extent=[s['y'][0], s['y'][-1], s['z'][0], s['z'][-1]],
                   cmap='Blues_r', vmin=0, vmax=d['beta'])
    fig.colorbar(im, ax=ax, label='transverse splitting e₂ − e₃  (vacuum: β = 0.3)')
    ax.text(3.2, 4.4, 'half-disclination\nlines (e₂ = e₃)', ha='center', fontsize=9, color=INK)
    for sgn in (1, -1):
        ax.annotate('', xy=(0.75 * sgn, 2.2), xytext=(2.6, 4.2), arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.4))
    ax.plot([0], [0], '+', color=ORANGE, ms=10, mew=2)
    ax.text(0.35, -0.5, 'charge', color=ORANGE, fontsize=9)
    ax.set(xlabel='y', ylabel='z (model units)', title='one charge, biaxial vacuum β = 0.3, plane x = 0')
    ax.title.set_fontsize(11)
    fig.tight_layout()
    save(fig, 'fig_charge.png')


def fig_pairs():
    files = sorted(glob.glob(os.path.join(R, 'pair_flow_b*.json')))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.0))
    style = {0.0: BLUE, 0.3: ORANGE, 0.1: AQUA}
    dash = {4.0: '-', 6.0: (0, (5, 2)), 8.0: (0, (1, 1.2))}
    for f in files:
        d = json.load(open(f))
        tr = d['trace']
        st = [t['step'] for t in tr]
        lab = ('uniaxial vacuum' if d['beta'] == 0 else f'β = {d["beta"]:g}') + f', d = {d["d"]:g}'
        ax.plot(st, [t['E'] for t in tr], linestyle=dash[d['d']], color=style[d['beta']], lw=2.2, label=lab)
        bx.plot(st, [0.5 * (abs(t['degrees'][0][0]) + abs(t['degrees'][1][0])) for t in tr], linestyle=dash[d['d']],
                color=style[d['beta']], lw=2.2, label=lab)
    ax.set(xlabel='gradient-flow step', ylabel='energy (model units)', title='held pair: the energy only falls', ylim=(0, 60))
    bx.set(xlabel='gradient-flow step', ylabel='charge of the held cores (sphere r = 1.6)',
           title='the charge of the held cores melts away', ylim=(-0.03, 1.03))
    h, l = bx.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: l[i])
    fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=4, frameon=False, fontsize=9)
    for a in (ax, bx):
        a.title.set_fontsize(11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, 'fig_pairs.png')


if __name__ == '__main__':
    fig_strand()
    fig_charge()
    if glob.glob(os.path.join(R, 'pair_flow_b*.json')):
        fig_pairs()
