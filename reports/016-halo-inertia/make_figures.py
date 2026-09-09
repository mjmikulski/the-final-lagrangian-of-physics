"""Figures of report 016 from the committed JSON records.

fig_halo.png  : cost and inertia of an internal tilt of the halo against the two one-dimensional integrals,
                with the exact coefficients (32 pi/3) Delta^4 and (64 pi/3) Delta^4 (no fit)
fig_rotor.png : the rigidly rotating branch at fixed J in box 12 -- rotational energy, frequency, and E/Omega
                against the clock condition E = 2 J Omega
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'results'


def load(name):
    return json.load(open(os.path.join(R, name)))


def fig_halo():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    marks = {1.0: 'o', 2.0: 's', 3.0: '^'}
    for name, col, lab in (('halo_scaling_n48_box18.json', 'C0', 'box 18, n = 48'), ('halo_scaling_n32_box12.json', 'C1', 'box 12, n = 32')):
        if not os.path.exists(os.path.join(R, name)):
            continue
        d = load(name)
        for row in d['rows']:
            ax[0].loglog(row['x_cost'], row['cost'], marks[row['L']], color=col, ms=6)
            ax[1].loglog(row['x_inertia'], row['inertia_excess'], marks[row['L']], color=col, ms=6)
        ax[0].plot([], [], 'o', color=col, label=f'{lab}: measured, slope {d["slope_cost"]:.1f}')
        ax[1].plot([], [], 'o', color=col, label=f'{lab}: measured, slope {d["slope_inertia"]:.1f}')
    d = load('halo_scaling_n48_box18.json')
    xc = np.logspace(-4.2, -0.8, 10)
    xi = np.logspace(-4.4, -0.5, 10)
    ax[0].loglog(xc, d['predicted_cost'] * xc, 'k-', lw=1.2, label=f'exact exterior formula, slope $(32\\pi/3)\\Delta^4$ = {d["predicted_cost"]:.1f} (no fit)')
    ax[1].loglog(xi, d['predicted_inertia'] * xi, 'k-', lw=1.2, label=f'exact exterior formula, slope $(64\\pi/3)\\Delta^4$ = {d["predicted_inertia"]:.1f} (no fit)')
    for L, m in marks.items():
        ax[0].plot([], [], m, color='0.4', label=f'twist zone thickness L = {L:g}')
        ax[1].plot([], [], m, color='0.4', label=f'twist zone thickness L = {L:g}')
    ax[0].set(xlabel="$\\alpha^2\\int s'(r)^2\\,dr$   (tilt angle $\\alpha$, radial profile $s$)", ylabel='static energy cost of the tilt (model units)',
              title='cost of tilting the halo against the radial derivative\nof the twist angle: exact slope 32.2, measured 22 to 25')
    ax[1].set(xlabel='$\\alpha^2\\int s(r)^2\\,dr$   (tilted length)', ylabel='inertia gained for rotation about z (model units)',
              title='inertia of the tilted halo grows linearly with the tilted\nlength: exact slope 64.4, measured 72 to 78')
    for a in ax:
        a.legend(fontsize=7.5, loc='upper left'); a.grid(alpha=0.3, which='both')
    fig.suptitle('The halo mechanism on the relaxed lattice hedgehog: internal tilt of the exterior by an angle $\\alpha\\,s(r)$ '
                 '($E_0 = 100$, $\\Delta = 0.99$)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(R, 'fig_halo.png'), dpi=150)


def fig_rotor():
    d = load('rotor_branch.json')
    J = np.array([r['J'] for r in d['rows']]); Er = np.array([r['E_above_static'] for r in d['rows']])
    Ek = np.array([r['E_kin'] for r in d['rows']]); Ed = np.array([r['E_def'] for r in d['rows']])
    Om = np.array([r['Omega'] for r in d['rows']]); EO = np.array([r['E_over_Omega'] for r in d['rows']])
    I2a = np.array([r['inertia_shape'] for r in d['rows']])
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))
    ax[0].plot(J, Er, 'o-', color='C0', label='$E(J) - E_{static}$: total')
    ax[0].plot(J, Ek, '^-', color='C2', label='kinetic part $J^2/(4a)$')
    ax[0].plot(J, Ed, 's-', color='C1', label='deformation energy of the shape')
    ax[0].set(xlabel='angular momentum J (model units)', ylabel='energy above the static hedgehog (model units)',
              title='the rotating states deform the shape;\nthe energy grows sublinearly with J')
    ax[0].margins(y=0.1)
    ax[1].semilogy(J, Om, 'o-', color='C3', label='frequency $\\Omega$ of the eigenframe (1/time)')
    ax[1].semilogy(J, I2a, 'x:', color='C4', label='inertia $2a$ of the deformed shape')
    ax[1].axhline(d['rigid_inertia'], color='k', ls='--', lw=1, label=f'rigid inertia of the static hedgehog ({d["rigid_inertia"]:.1f})')
    ax[1].set(xlabel='angular momentum J (model units)', ylabel='$\\Omega$ and inertia (model units, magnitudes only)',
              title=f'the frequency falls with J; the inertia is\n{I2a.min() / d["rigid_inertia"]:.0f} to {round(I2a.max() / d["rigid_inertia"], -2):.0f} times the rigid value')
    ax[1].legend(fontsize=8, loc='center right')
    ax[2].loglog(J, EO, 'o-', color='k', label='$E(J)/\\Omega$ of the branch (total energy)')
    ax[2].loglog(J, 2 * J, '--', color='C3', lw=1.2, label='clock condition $E/\\Omega = 2J$')
    ax[2].set(xlabel='angular momentum J (model units)', ylabel='$E/\\Omega$ and $2J$ (model units)',
              title='the clock condition is not met anywhere\non this branch: $E/\\Omega$ moves away from $2J$')
    ax[0].legend(fontsize=8, loc='upper left'); ax[2].legend(fontsize=8, loc='center right')
    for a in ax:
        a.grid(alpha=0.3, which='both')
    fig.suptitle('Rigidly rotating stationary states at fixed J on the lattice (box 12, n = 32, $E_0 = 100$)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(R, 'fig_rotor.png'), dpi=150)


if __name__ == '__main__':
    fig_halo()
    fig_rotor()
    print('figures written')
