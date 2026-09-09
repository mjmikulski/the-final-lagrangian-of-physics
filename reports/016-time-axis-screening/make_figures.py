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
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    # (a) 1D profiles
    pf, ps = rad['frozen']['profile'], rad['screened']['profile']
    ax[0].semilogx(pf['r'], pf['e1'], color='k', lw=1.6, label='$e_1$, time axis frozen')
    ax[0].semilogx(ps['r'], ps['e1'], color='C3', lw=1.6, label='$e_1$, time axis free')
    ax[0].semilogx(ps['r'], np.array(ps['chi']) / chi_star, color='C4', lw=1.6, ls='--', label='$\\chi/\\chi_*$, time axis free (frozen: 0)')
    ax[0].set(xlabel='r (model units)', ylabel='$e_1$ and $\\chi/\\chi_*$ (dimensionless)', xlim=(2e-2, 300), ylim=(-0.1, 1.25),
              title='radial problem, no box ($E_0 = 100$): $e_1$ and the boost\nrapidity $\\chi$ of the time axis ($\\chi_* \\approx \\Delta/E_0$, screening value)')
    ax[0].legend(fontsize=8, loc='lower right')
    # (b) energy per unit radius
    r, dens = np.array(pf['r']), np.array(pf['density']); m = (r > 0.02) & (dens > 0)
    ax[1].loglog(r[m], dens[m], color='k', lw=1.6, label=f'time axis frozen: mass {rad["frozen"]["E"]:.1f} (model units)')
    r, dens = np.array(ps['r']), np.array(ps['density']); m = (r > 0.02) & (dens > 0)
    ax[1].loglog(r[m], dens[m], color='C3', lw=0.9, alpha=0.85, label=f'time axis free: mass {rad["screened"]["E"]:.1f} (model units);\ntail below $10^{{-3}}$ is solver noise')
    rr = np.logspace(0.3, 2.5, 50)
    ax[1].loglog(rr, 16 * math.pi * Delta ** 4 / rr ** 2, '-', color='0.75', lw=5, zorder=0, label='Coulomb law $16\\pi\\Delta^4/r^2$ (the frozen tail lies on it)')
    ax[1].set(xlabel='r (model units)', ylabel='energy per unit radius $4\\pi r^2\\,\\mathcal{H}$ (model units)', xlim=(2e-2, 300), ylim=(1e-8, 1e2),
              title='energy per unit radius ($E_0 = 100$): with the time axis\nfree the Coulomb tail is gone')
    ax[1].legend(fontsize=8, loc='lower left')
    # (c) lattice tilt profiles
    for name, col, ls, mk, lab in (('static_unfrozen_E0_10.pt', 'C0', '--', 's', '$E_0 = 10$'), ('static_unfrozen_E0_100.pt', 'C3', '-', 'o', '$E_0 = 100$'),
                                   ('static_unfrozen_E0_1000.pt', 'C1', ':', '^', '$E_0 = 1000$'), ('static_tilt0.1.pt', 'C2', '-', 'o', '$E_0 = 100$, with $K_u$, c = 0.1'),
                                   ('static_tilt3.0.pt', 'C4', '-', 'o', '$E_0 = 100$, with $K_u$, c = 3')):
        if name not in fd:
            continue
        prof = fd[name]['profiles']
        ax[2].plot([p['r'] for p in prof], [p['m0i_radial'] for p in prof], marker=mk, ls=ls, ms=4, color=col, lw=1.4, label=lab)
    ax[2].axhline(Delta, color='k', ls=':', lw=1, label='screening value $\\Delta$')
    ax[2].set(xlabel='r (model units)', ylabel='radial tilt $M_{0i}\\hat x_i$ (shell mean, model units)', ylim=(-0.05, 1.1),
              title='lattice, box 12: the tilt of the time axis (curves for\n$E_0$ = 10 to 1000 overlap) and its expulsion by $K_u$')
    ax[2].legend(fontsize=8, loc='lower left')
    for a in ax:
        a.grid(alpha=0.3, which='both')
    fig.suptitle('The electron with the time sector free: the radial tilt of the time axis screens the Coulomb energy ($E_2 = E_3 = 0.01$)', fontsize=11)
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
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax[0].semilogx(c1, (E1d - Es) / (Ef - Es), 'o-', color='k', label='radial problem, no box (r up to 300)')
    ax[0].semilogx(cl, (El - Es_lat) / (Ef_lat - Es_lat), 's', color='C3', ms=11, mfc='none', mew=1.8, label='lattice, box 12')
    ax[0].axhline(1, color='k', ls='--', lw=1, label='frozen hedgehog (mass fully restored)')
    ax[0].axhline(0, color='C3', ls=':', lw=1.2, label='no term: fully screened')
    ax[0].axvspan(0.1, 0.3, color='0.85', label='full restoration reached between c = 0.1 and 0.3')
    ax[0].set(xlabel='coupling c of the frame term $K_u$', ylabel='restored fraction of the Coulomb energy\n$(E(c) - E_{screened})/(E_{frozen} - E_{screened})$',
              ylim=(-0.05, 1.1), title='the frame term restores the mass: lattice checks fall on the\nradial curve where the tilt fits in the box (c ≥ 0.1)')
    ax[0].legend(fontsize=8, loc='lower right')
    cs = np.array([s[0] for s in stiff]); ks = np.array([s[1] for s in stiff])
    ax[1].plot(cs, ks, 'o', color='C2', ms=7, label='Rayleigh quotient of the full Hessian\nalong the radial tilt $M_{0i}\\propto x_i f(r)$')
    if len(cs) > 1:
        slope, icpt = np.polyfit(cs, ks, 1)
        cc = np.linspace(0, cs.max(), 50)
        ax[1].plot(cc, icpt + slope * cc, 'k:', lw=1.2, label=f'linear fit: zero at c = {-icpt / slope:.3f}')
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set(xlabel='coupling c of the frame term $K_u$', ylabel='stiffness along the radial tilt\n(Rayleigh quotient, model units)',
              title='local threshold: the stiffness of the frozen hedgehog\nalong the radial tilt turns positive at c ≈ 0.013')
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
