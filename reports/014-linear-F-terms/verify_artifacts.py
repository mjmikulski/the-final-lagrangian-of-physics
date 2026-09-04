"""Artifact-only assertions for the results the README quotes (structural
consistency of the committed JSONs; not a rerun)."""
import json, os

def load(n):
    with open(f'results/{n}.json') as f:
        return json.load(f)

def main():
    fl = load('linear_float'); ex = load('linear_exact'); nt = load('null_test')
    ob = load('orbit_linear'); o1 = load('orbit1_linear'); ks = load('static_kernel_signs')
    assert (fl['n_diagrams'], fl['n_classes']) == (675, 38)
    assert (fl['rank_all'], fl['rank_even'], fl['rank_odd']) == (12, 6, 6)
    assert (ex['rank_all'], ex['rank_even'], ex['rank_odd']) == (12, 6, 6)
    assert fl['rank_spatial_3x3'] == 3 and fl['rank_static_generic'] == 12
    alive = [k for k, v in fl['reps'].items() if v['spatial_3x3'] == 'alive']
    assert all('P0' not in fl['reps'][k]['label'] and 'eps' not in fl['reps'][k]['label'] for k in alive)
    print('artifact check: family 675/38, ranks 12 = 6 + 6 (float and exact); 3x3 sector rank 3, no P_t / no odd class alive there')
    assert nt['CONTROL_phi']['null'] and nt['L11']['null'] and not nt['CONTROL_I1']['null']
    dec = [k for k in nt if k.startswith('L') and k not in ('L0', 'L11')]
    assert all(not nt[k]['null'] for k in dec) and len(dec) == 36
    print('artifact check: phi and chi null, I1 dynamical, all 36 decorated classes dynamical')
    assert ob['orbit_zero'] == [] and len(ob['orbit_nonzero']) == 38
    assert o1['even_with_Pt_cap_all_zero'] is True
    print('artifact check: rank-rich orbit all nonzero; rank-1 orbit P_t-capped even classes exactly zero')
    for k, v in ks.items():
        assert abs(v['trace']) < 1e-9 and v['n_pos'] > 0 and v['n_neg'] > 0
    print('artifact check: 3x3 static kernels traceless and indefinite')
    for name in ('lattice_linear_A', 'lattice_linear_B'):
        if os.path.exists(f'results/{name}.json'):
            lt = load(name)
            v = lt['validation']
            for c, fdv in v['directional_fd'].items():
                assert fdv['rel_err'] < 1e-5, (name, c)
            assert v['lattice_EL_ratio_phi_over_dyn'] < 1e-12
            if name == 'lattice_linear_B':
                m = v['metric_vs_exact']
                assert max(m[k]['max'] for k in ('Pt', 'P1', 'Q', 'T')) < 1e-10
                assert abs(lt['base_integrals']['FtQ']) < 1e-12
            else:
                m = v['metric_vs_exact']
                assert m['P0']['max'] < 1e-3 and m['P2']['max'] > 0.5
            runs = {k: r for k, r in lt.items() if isinstance(r, dict) and 'status' in r}
            base = runs['baseline']
            n_relaxed = sum(1 for k, r in runs.items() if k != 'baseline' and r['status'] == 'ok'
                            and abs(r.get('continuation_dE', 1.0)) < 0.01 * max(abs(r['E_total'] - base['E_total']), 1e-12))
            print(f'artifact check: {name}: {len(runs)} runs, {sum(r["status"] != "ok" for r in runs.values())} non-ok, '
                  f'{n_relaxed} pass the continuation gate; derivative check and null control recorded')
    # vacuum stability, twist scan, radial profile, restart checks (when present)
    for rt in ('A', 'B'):
        fn = f'results/vacuum_condensation_{rt}.json'
        if os.path.exists(fn):
            vc = load(f'vacuum_condensation_{rt}')
            assert all(abs(r['E_relaxed']) < 1e-4 for r in vc.values()), 'vacuum did not return to E ~ 0'
            print(f'artifact check: vacuum_condensation_{rt}: all {len(vc)} runs relax back to |E| < 1e-4 (no condensate)')
        fn = f'results/restart_check_{rt}.json'
        if os.path.exists(fn):
            rc = load(f'restart_check_{rt}'); lt = load(f'lattice_linear_{rt}')
            base = lt['baseline']['E_total']
            bad = [k for k, r in rc.items() if abs(r['dE_vs_main']) > 0.01 * abs(r['E_main'] - base)]
            print(f'artifact check: restart_check_{rt}: {len(rc)} restarts, {len(bad)} outside the 1% gate {bad}')
    if os.path.exists('results/twist_scan_B.json'):
        ts = load('twist_scan_B'); t = ['0.002', '0.005']
        ratio_lin = ts[t[1]]['lin_integral'] / ts[t[0]]['lin_integral']; ratio_stat = ts[t[1]]['E_stat'] / ts[t[0]]['E_stat']
        assert 30 < ratio_lin < 50 and 30 < ratio_stat < 50, (ratio_lin, ratio_stat)   # quartic: (2.5)^4 = 39
        print(f'artifact check: twist scan: linear integral and E_stat both scale as t^4 (ratios {ratio_lin:.1f}, {ratio_stat:.1f})')
    for rt in ('A', 'B'):
        fn = f'results/radial_profile_{rt}.json'
        if os.path.exists(fn):
            rp = load(f'radial_profile_{rt}')['classes']
            fo = {k: v['frac_outer'] for k, v in rp.items() if v['frac_outer'] is not None}
            assert 0.3 < fo['eta_static'] < 0.7
            print(f'artifact check: radial_profile_{rt}: outer-zone fraction eta_static {fo["eta_static"]:.2f}, '
                  + ', '.join(f'{k} {v:.2f}' for k, v in fo.items() if k not in ('eta_static',)))
    if os.path.exists('results/stall_diagnostics_B.json'):
        sd = load('stall_diagnostics_B')
        stalled = {k: r for k, r in sd.items() if r['grad_inf_free'] > 0.1}
        relaxed = {k: r for k, r in sd.items() if r['grad_inf_free'] <= 0.1}
        assert all(r['gaps_at_argmax']['1-2'] < 1e-3 and r['gaps_min_free']['1-2'] < 1e-6 and r['gaps_at_argmax']['2-3'] > 0.5 for r in stalled.values())
        assert all(r['directional']['rel_err']['1e-06'] > 1.0 for r in stalled.values())
        assert all(r['directional']['rel_err']['1e-06'] < 1e-5 for r in relaxed.values())
        print(f'artifact check: stall diagnostics: {len(stalled)} stalled endpoints all sit on 1-2 collisions (gap < 1e-3 at the '
              f'max-gradient site, < 1e-6 somewhere on the free sites; 2-3 gap > 0.5 there) with a failed FD check; {len(relaxed)} relaxed endpoints pass FD to < 1e-5')
    if os.path.exists('results/continuation_A_P1P3_f0.2_s-1.json'):
        ct = load('continuation_A_P1P3_f0.2_s-1')
        assert ct['history'][0]['g_inf'] > 0.1 and ct['passes_gate'] and ct['history'][-1]['g_inf'] <= 0.1
        print(f"artifact check: continuation of A:P1P3 -20%: |grad|inf {ct['history'][0]['g_inf']:.3f} -> {ct['history'][-1]['g_inf']:.3f} (gate passed)")
    print('ARTIFACT CHECKS PASS (committed JSON consistency; not a GPU rerun)')

if __name__ == '__main__':
    main()
