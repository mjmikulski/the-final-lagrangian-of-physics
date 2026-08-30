"""Artifact-only assertions for the main lattice results (PR #12 finding 3).

This does NOT reproduce the GPU runs; it structurally checks that the
committed results JSONs actually contain the verdicts the README claims, so
`reproduce.sh` cannot say ALL PASS against inconsistent artifacts. A CUDA
rerun is the M5_RUN_LATTICE=1 path.
"""

import json
import os


def load(name):
    with open(f'results/{name}.json') as f:
        return json.load(f)


def main():
    pe = load('pre_e4')
    e4 = load('e4_cells')
    garm = load('e4_gamma_arm')
    conf = load('e4_confirm')
    e5 = load('e5_arms')

    # validations and drive
    v = pe['validation']
    assert v['statics_identity_rel'] < 1e-12
    assert v['pointwise_crosscheck_rel'] < 1e-10
    assert v['U_vs_eigen_max'] < 1e-3
    assert pe['generator'] == 'boost_x' and pe['D1_by_gen']['boost_x'] > 0
    print('artifact check: base validations + record drive D1 > 0')

    # main sweep: 0/15 interior wells; gamma=0 control is the record disease
    cells = [c for c in e4 if c != 'CONTROL_gamma0' and 'rows' in e4[c]]
    assert len(cells) == 15
    assert all(not e4[c]['verdict']['interior_well'] for c in cells)
    g0 = e4['CONTROL_gamma0']
    Es = [r['E_total'] for r in g0['rows']]
    assert not g0['verdict']['interior_well']
    assert all(b < a for a, b in zip(Es, Es[1:])), 'gamma=0 not monotone'
    ndive = sum(e4[c]['verdict'].get('runaway', False) for c in cells)
    assert ndive == 11, f'expected 11 dive cells, artifacts say {ndive}'
    print('artifact check: main sweep 0/15 wells; gamma=0 control monotone; '
          f'{ndive} dive cells')

    # two-sided window
    assert garm['C10x100']['verdict']['min_omega'] == 0.0
    assert garm['C10x10000']['verdict']['min_omega'] == 0.0
    assert garm['C9x10000']['verdict']['runaway']
    assert not garm['C10x3']['verdict']['interior_well']  # below the window:
    # min migrates between protocol levels (0.7 vs 1.2), no certified well
    assert garm['C10x10']['verdict']['interior_well']
    assert garm['C10x10']['verdict']['level_stable']
    print('artifact check: window bounded (evasion below, drive-kill above, '
          '-s^2 ignition at extreme gamma); x10 interior well')

    # confirmation arm: fine well, sibling cells, drive-flip control
    fv = conf['C10_x10_fine']['verdict']
    assert fv['interior_well'] and fv['level_stable']
    for c in ('C19_x10', 'C13_x10', 'C16_x10'):
        cv = conf[c]['verdict']
        assert cv['interior_well'] and cv['level_stable'], c
    df = conf['C10_x10_drive_flip_control']['verdict']
    assert not df['interior_well'] and df['min_omega'] == 0.0
    print('artifact check: fine well + three sibling wells level-stable; '
          'drive-flip control kills the well')

    # cross-file fine bracket at x10 with the extra rungs
    rows = {r['omega']: r['E_total'] for r in conf['C10_x10_fine']['rows']}
    rows.update({r['omega']: r['E_total']
                 for r in e5['x10_fine_extra']['rows']})
    assert rows[0.1] > rows[0.15] < rows[0.2], 'x10 fine bracket broken'

    # window map + deep probe
    assert e5['deep_well']['plateau'] is True
    assert e5['window_map']['x5']['verdict']['min_at_top']
    for tag, om in (('x14', 0.15), ('x20', 0.1)):
        wv = e5['window_map'][tag]['verdict']
        assert wv['interior_well'] and wv['level_stable']
        assert wv['min_omega'] == om
    print('artifact check: window map (x5 monotone; wells x10/x14/x20 at '
          '0.15/0.15/0.1); x10 depth plateau settled')

    # deep-bracket run (finding 2), if present
    if os.path.exists('results/e5_deep_bracket.json'):
        db = load('e5_deep_bracket')
        b = db['bracket']
        print(f"artifact check: deep bracket at x14 -> interior={b['interior']} "
              f"min_omega={b['final_min_omega']} "
              f"gaps_resolve_creep={b['gaps_resolve_creep']}")
        # the committed outcome is NEGATIVE and the README is scoped to it:
        # interior minimum at 0.15 through four levels, then migration to the
        # still-descending top rung -- assert the artifact says exactly that
        assert not b['interior'] and b['final_min_omega'] == 0.28
        assert b['min_omega_per_level'][:4] == [0.15] * 4

    # span-level static kernel (finding 1)
    sk = load('static_kernel_exact')
    assert sk['counterexample_confirmed'] and not sk['pure_kinetic_exists']
    assert sk['kernel_dim_mod_identities'] == 1

    # gyroscopic lambda-family lattice coefficients (round 2, finding 1)
    gf = load('gyro_family_lattice')
    assert gf['s_P_max'] < 1e-9 * gf['density_scale']
    assert gf['k_P_max'] < 1e-9 * gf['density_scale']
    assert gf['MP2'] > 0
    for c in ('C10', 'C13', 'C16'):
        assert gf['cross'][c]['mP_mQ'] == 0.0, c
    print('artifact check: gyro family -- s_P = k_P = 0 on the lattice, '
          '<m_P^2> > 0, <m_P m_Q> = 0 for C10/C13/C16')

    print('ARTIFACT CHECKS PASS (structural consistency of committed JSONs; '
          'not a GPU reproduction — set M5_RUN_LATTICE=1 for that)')


if __name__ == '__main__':
    main()
