"""Figures of report 016 from the committed JSON records (no lattice or GPU work needed).

fig_screening.png : the electron with the time sector free -- 1D profiles, the energy density, the lattice tilt
fig_cure.png      : the frame term K_u against its coupling -- restored mass (1D and lattice), the two thresholds
"""
import glob
import json
import math
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'results'
E0, E1, E2 = 100.0, 1.0, 0.01
Delta = E1 - E2
chi_star = math.asinh(math.sqrt(Delta ** 2 / ((E0 - E2) ** 2 - Delta ** 2)))


def load(name):
    return json.load(open(os.path.join(R, name)))


def fig_screening():
    rad = {'frozen': load('radial_E0_100_nk240.json')['frozen'], 'screened': load('radial_E0_100_nk480.json')['screened']}
    fd = {row['file']: row for row in load('field_diagnostics.json')}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    # (a) 1D profiles
    for tag, col, lab in (('frozen', 'k', 'time axis frozen'), ('screened', 'C3', 'time axis free')):
        p = rad[tag]['profile']
        r, e1, chi = np.array(p['r']), np.array(p['e1']), np.array(p['chi'])
        ax[0].semilogx(r, e1, color=col, lw=1.6, label=f'{lab}: $e_1$')
        ax[0].semilogx(r, chi / chi_star, color=col, lw=1.6, ls='--', label=f'{lab}: $\\chi/\\chi_*$')
    ax[0].set(xlabel='r (model units)', ylabel='profile', xlim=(2e-2, 300), ylim=(-0.1, 1.25),
              title='radial problem, no box: the charge eigenvalue $e_1$\nand the boost rapidity $\\chi$ of the time axis')
    ax[0].legend(fontsize=8, loc='center right')
    # (b) energy per unit radius
    for tag, col, lab in (('frozen', 'k', 'time axis frozen'), ('screened', 'C3', 'time axis free')):
        p = rad[tag]['profile']
        r, dens = np.array(p['r']), np.array(p['density'])
        m = (r > 0.02) & (dens > 0)
        ax[1].loglog(r[m], dens[m], color=col, lw=1.6, label=f'{lab}: mass {rad[tag]["E"]:.1f}')
    rr = np.logspace(0, 2.5, 50)
    ax[1].loglog(rr, 16 * math.pi * Delta ** 4 / rr ** 2, 'k:', lw=1, label='Coulomb law $16\\pi\\Delta^4/r^2$')
    ax[1].set(xlabel='r (model units)', ylabel='energy per unit radius $4\\pi r^2\\,\\mathcal{H}$', xlim=(2e-2, 300), ylim=(1e-8, 1e2),
              title='energy per unit radius: with the time axis free\nthe Coulomb tail is gone')
    ax[1].legend(fontsize=8, loc='lower left')
    # (c) lattice tilt profiles
    for name, col, lab in (('static_unfrozen_E0_10.pt', 'C0', '$E_0 = 10$'), ('static_unfrozen_E0_100.pt', 'C3', '$E_0 = 100$'),
                           ('static_unfrozen_E0_1000.pt', 'C1', '$E_0 = 1000$'), ('static_tilt0.1.pt', 'C2', '$E_0 = 100$, with $K_u$, c = 0.1'),
                           ('static_tilt1.pt', 'C4', '$E_0 = 100$, with $K_u$, c = 1')):
        if name not in fd:
            continue
        prof = fd[name]['profiles']
        ax[2].plot([p['r'] for p in prof], [p['m0i_radial'] for p in prof], 'o-', ms=3.5, color=col, lw=1.4, label=lab)
    ax[2].axhline(Delta, color='k', ls=':', lw=1, label='screening value $\\Delta$')
    ax[2].set(xlabel='r (model units)', ylabel='radial tilt $M_{0i}\\hat x_i$ (shell mean)', ylim=(-0.05, 1.1),
              title='lattice, box 12: the tilt of the time axis in the relaxed\nelectron, and its expulsion by the frame term $K_u$')
    ax[2].legend(fontsize=8, loc='upper right')
    for a in ax:
        a.grid(alpha=0.3, which='both')
    fig.suptitle('The electron with the time sector free: the radial tilt of the time axis screens the Coulomb energy '
                 f'($E_0 = 100$, $E_2 = E_3 = 0.01$)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(R, 'fig_screening.png'), dpi=150)


def fig_cure():
    base = load('radial_E0_100_nk240.json')
    Ef, Es = base['frozen']['E'], base['screened']['E']
    rows = []
    for f in sorted(glob.glob(os.path.join(R, 'radial_E0_100_nk240_tilt*.json'))):
        c = float(re.search(r'tilt([0-9.]+)\.json', f).group(1))
        rows.append((c, json.load(open(f))['screened']['E']))
    rows.sort()
    c1 = np.array([r[0] for r in rows]); E1d = np.array([r[1] for r in rows])
    fd = {row['file']: row for row in load('field_diagnostics.json')}
    Ef_lat = fd['static_base_n32.pt']['E']; Es_lat = fd['static_unfrozen_E0_100.pt']['E']
    lat = sorted((row['tilt'], row['E']) for row in fd.values() if row['tilt'] > 0 and row['E0'] == 100.0)
    cl = np.array([x[0] for x in lat]); El = np.array([x[1] for x in lat])
    ts = load('time_sector_check.json')
    stiff = sorted((row['tilt'], row['rayleigh']['radial M_0i ~ x_i']) for row in ts if row['E0'] == 100.0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].semilogx(c1, (E1d - Es) / (Ef - Es), 'o-', color='C0', label='radial problem, no box (r up to 300)')
    ax[0].semilogx(cl, (El - Es_lat) / (Ef_lat - Es_lat), 's--', color='C1', label='lattice, box 12')
    ax[0].axhline(1, color='k', ls='--', lw=1, label='frozen hedgehog (mass fully restored)')
    ax[0].axhline(0, color='C3', ls=':', lw=1.2, label='no term: fully screened')
    ax[0].axvspan(0.1, 0.3, color='0.85', label='global threshold (between 0.1 and 0.3)')
    ax[0].set(xlabel='coupling c of the frame term $K_u$', ylabel='restored fraction of the Coulomb energy\n$(E(c) - E_{screened})/(E_{frozen} - E_{screened})$',
              ylim=(-0.05, 1.1), title='the frame term restores the mass:\nradial problem and lattice agree on the threshold')
    ax[0].legend(fontsize=8, loc='center left')
    cs = np.array([s[0] for s in stiff]); ks = np.array([s[1] for s in stiff])
    ax[1].plot(cs, ks, 'o-', color='C2', label='Rayleigh quotient of the full Hessian\nalong the radial tilt $M_{0i}\\propto x_i f(r)$')
    if len(cs) > 1:
        slope, icpt = np.polyfit(cs, ks, 1)
        cc = np.linspace(0, cs.max(), 50)
        ax[1].plot(cc, icpt + slope * cc, 'k:', lw=1, label=f'linear fit: zero at c = {-icpt / slope:.3f}')
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set(xlabel='coupling c of the frame term $K_u$', ylabel='stiffness of the frozen hedgehog\nalong the radial tilt',
              title='local threshold: the frozen hedgehog becomes a local\nminimum of the full energy once the stiffness is positive')
    ax[1].legend(fontsize=8, loc='lower right')
    for a in ax:
        a.grid(alpha=0.3, which='both')
    fig.suptitle('The frame term $K_u$ against its coupling ($E_0 = 100$; lattice box 12, n = 32)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(R, 'fig_cure.png'), dpi=150)


if __name__ == '__main__':
    fig_screening()
    fig_cure()
    print('figures written')
