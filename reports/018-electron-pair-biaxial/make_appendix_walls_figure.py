"""Figure for APPENDIX-walls-kappa, from the committed JSON records of strand_walls.py (no lattice work).

Left: T / T_B against L-BFGS iterations for the kicked seed (k = 1/2, beta = 0.3) at h = rho_1/2 / 8 and / 16
(the / 16 run continued from its saved field), without and with kappa = 1e-2.
Right: the energy the two-derivative term adds to the smooth strand, T(kappa) - T(0), against ln(R / rho_1/2)
for k = 1/2 and 1, with the single-line slope 16 pi k^2 b0^2 drawn through the kappa = 1e-3 point at R = 8 of each k.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
BLUE, ORANGE, AQUA, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#1a1a19', '#6b6a63'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
BETA, B0 = 0.3, 0.15
RH = math.sqrt(2 * (4 * 0.5 * B0 / math.sqrt(2)) * (2 * math.log(2) - 1))


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else []


kick = load('strand_walls_kick.json') + load('strand_walls_kick16kappa.json')
cont = load('strand_walls_continue16.json')
kap = load('strand_walls_kappa.json')

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6))
for r in kick:
    lab = f'h = ρ½/{round(RH / r["h"])}' + (f', κ = {r["kappa"]:g}' if r['kappa'] else ', κ = 0')
    col = (ORANGE if round(RH / r['h']) == 8 else '#8a4f00') if r['kappa'] else (BLUE if round(RH / r['h']) == 8 else AQUA)
    it = [r['block'] * (i + 1) for i in range(len(r['trace']))]
    y = [t / r['T_bound'] for t in r['trace']]
    if round(RH / r['h']) == 16 and not r['kappa'] and cont:
        c = cont[0]
        it += [it[-1] + c['block'] * (i + 1) for i in range(len(c['trace']))]
        y += [t / c['T_bound'] for t in c['trace']]
        lab += ' (continued)'
    ax.plot(it, y, '-o', color=col, ms=4, lw=1.8, label=lab)
ax.axhline(1, color=MUTED, lw=1)
ax.text(24500, 1.04, 'Bogomolny value T_B', color=MUTED, fontsize=9, ha='right')
ax.set(xlabel='L-BFGS iterations', ylabel='T / T_B', ylim=(0, 2.0), title='kicked seed leaves the saddle at both spacings; with κ it levels off')
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.title.set_fontsize(11)

for k, col, mk in ((0.5, BLUE, 'o'), (1.0, ORANGE, 's')):
    for kappa, ls in ((1e-3, '-'), (1e-2, '--')):
        pts = []
        for Rm in (4, 8, 16):
            a = [r for r in kap if r['k'] == k and r['kappa'] == kappa and abs(r['R'] / RH - Rm) < 1e-6]
            z = [r for r in kap if r['k'] == k and r['kappa'] == 0.0 and abs(r['R'] / RH - Rm) < 1e-6]
            if a and z:
                pts.append((math.log(Rm), (a[0]['T'] - z[0]['T']) / kappa))
        if not pts:
            continue
        x, y = zip(*pts)
        bx.plot(x, y, ls, marker=mk, color=col, ms=6, lw=1.6, label=f'k = {k:g}, κ = {kappa:g}')
xx = np.linspace(math.log(4), math.log(16), 10)
for k, col in ((0.5, BLUE), (1.0, ORANGE)):
    slope = 16 * math.pi * k * k * B0 ** 2
    ref = [r for r in kap if r['k'] == k and r['kappa'] == 1e-3 and abs(r['R'] / RH - 8) < 1e-6]
    z = [r for r in kap if r['k'] == k and r['kappa'] == 0.0 and abs(r['R'] / RH - 8) < 1e-6]
    if ref and z:
        y8 = (ref[0]['T'] - z[0]['T']) / 1e-3
        bx.plot(xx, y8 + slope * (xx - math.log(8)), linestyle=(0, (1, 1.5)), color=INK, lw=2.0, zorder=5,
                label=f'single-line slope 16π k² b₀², k = {k:g}')
bx.set(xlabel='ln(R / ρ½)', ylabel='[T(κ) − T(0)] / κ', title='two-derivative term: log winding cost; k = 1 splits')
bx.legend(frameon=False, fontsize=8.5, loc='upper left')
bx.title.set_fontsize(11)
fig.tight_layout()
tmp = os.path.join(R, 'fig_appendix_walls.tmp.png')
fig.savefig(tmp, dpi=150)
os.replace(tmp, os.path.join(R, 'fig_appendix_walls.png'))
