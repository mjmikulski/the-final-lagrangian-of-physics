"""L2 lattice scan of linear-in-F static terms, unified routes (review round 0).

  route A  vacuum-pinned Lagrange projectors (004's G route; a fixed polynomial
           in eta M -- exactly differentiable, but not the spectral projectors
           in the cores); classes P1P2, P1P3, P2P3.
  route B  spectral coefficients with DIFFERENTIABLE eigenvalue nodes
           (torch.linalg.eigvals in the graph): P_t, the small-pair cluster
           projector Q = P_2 + P_3 (smooth through the 2-3 crossing), P_1 =
           eta - P_t - Q, and the gap-weighted splitting T = (X - lbar) Q,
           lbar = tr(XQ)/2, which equals (l2 - l3)/2 (P_2 - P_3) in the local
           eigenbasis; classes F1Q, FQQ (= 2 F23), F1T (= (l2-l3)/2 (F12-F13)),
           FtQ (control, vanishes on the eta profile).

E_lambda(M) = e_static(M, eta) + lambda H^3 sum dens. Per run: Adam 2000 +
L-BFGS 100, then the gates the review asked for -- final true-objective
free-site gradient norm, a +100-iteration L-BFGS continuation (energy, linear
integral and tail change), and for the 20% runs a perturbed restart. Fields
persisted under ../runs/linear_F_terms/. Before the scan: a directional
finite-difference check of the full derivative (eigenvalue-changing direction
included). Usage: python lattice_linear_v2.py A|B
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '010-fundamental-grid-clock'))
import numpy as np
import torch
from lattice_grid_defs import (DEV, DT, ETA, FREE, H, N, F4_of, U_of, _L, d1,
                               e_static, field, load_or_make_base, offblock)
tail_fit = _L['tail_fit']
ROUTE = sys.argv[1] if len(sys.argv) > 1 else 'B'
TARGETS = [8.0, 1.0, 0.3, 0.0]
RESULTS = f'results/lattice_linear_{ROUTE}.json'
FIELDS = os.environ.get('M5_FIELDS_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'fields'))
os.makedirs(FIELDS, exist_ok=True)
CLASSES = ({'P1P2': (1, 2), 'P1P3': (1, 3), 'P2P3': (2, 3)} if ROUTE == 'A'
           else {'F1Q': (1, 2), 'FQQ': (2, 2), 'F1T': (1, 3), 'FtQ': (0, 2)})
ADAM, LR, LB, LB_CONT = 2000, 5e-4, 100, 100


def metrics_A(M):
    x = torch.einsum('ab,...bc->...ac', ETA, M)
    I = torch.eye(4, dtype=DT, device=DEV).expand_as(M)
    out = []
    for a in range(4):
        P = I
        for b in range(4):
            if b != a:
                P = P @ (x - TARGETS[b] * I) / (TARGETS[a] - TARGETS[b])
        out.append(P @ ETA)
    return out


def metrics_B(M):
    """[P_t, P_1, Q, T] as (2,0) tensors, all differentiable in M."""
    x = torch.einsum('ab,...bc->...ac', ETA, M)
    lam = torch.linalg.eigvals(x.cpu()).real.to(DEV)          # differentiable nodes
    lam = torch.sort(lam, dim=-1, descending=True).values
    I = torch.eye(4, dtype=DT, device=DEV).expand_as(M)
    lt, l1, l2, l3 = [lam[..., k][..., None, None] for k in range(4)]
    Pt = ((x - l1 * I) @ (x - l2 * I) @ (x - l3 * I)) / ((lt - l1) * (lt - l2) * (lt - l3))
    g2 = 1.0 / ((l2 - lt) * (l2 - l1))
    g3 = 1.0 / ((l3 - lt) * (l3 - l1))
    d = l2 - l3
    small = d.abs() < 1e-9
    g2p = -g2 * (1.0 / (l2 - lt) + 1.0 / (l2 - l1))
    alpha = torch.where(small, g2p, (g2 - g3) / torch.where(small, torch.ones_like(d), d))
    beta = g2 - alpha * l2
    Q = ((x - lt * I) @ (x - l1 * I)) @ (alpha * x + beta * I)
    P1 = I - Pt - Q
    lbar = 0.5 * torch.einsum('...aa->...', x @ Q)[..., None, None]
    T = (x - lbar * I) @ Q
    return [Pt @ ETA, P1 @ ETA, Q @ ETA, T @ ETA]


metrics = metrics_A if ROUTE == 'A' else metrics_B


def dens_class(M, a, b, mets=None):
    mets = metrics(M) if mets is None else mets
    X, Y = mets[a], mets[b]
    acc = 0.0
    for st in ('fwd', 'bwd'):
        A = [d1(M, ax, st) for ax in range(3)]
        F4 = F4_of([torch.zeros_like(M)] + A)
        acc = acc + 0.5 * torch.einsum('...mnab,...ma,...nb->...', F4, X, Y)
    return acc


def E_of(Mr, lam, a, b):
    Mf = field(Mr)
    return e_static(Mf, 'eta') + lam * H ** 3 * dens_class(Mf, a, b).sum()


def grad_inf_free(Mr, lam, a, b):
    Mv = Mr.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(E_of(Mv, lam, a, b), Mv)[0]
    g = 0.5 * (g + g.transpose(-1, -2))
    return float(g[FREE].abs().max())


def validate(Mg, Mr):
    out = {'route': ROUTE}
    mets = metrics(Mg)
    out['P0_vs_minusU_max'] = float((mets[0] + U_of(Mg)).abs().max())
    x = torch.einsum('ab,...bc->...ac', ETA, Mg).cpu()
    lam, vec = torch.linalg.eig(x)
    lam, vec = lam.real, vec.real
    order = torch.argsort(lam, dim=-1, descending=True)
    Pe = []
    for a in range(4):
        idx = order[..., a]
        e = torch.gather(vec, -1, idx[..., None, None].expand(*x.shape[:3], 4, 1))[..., 0]
        nrm = torch.einsum('...a,ab,...b->...', e, ETA.cpu(), e)
        Pe.append(torch.einsum('...a,...b->...ab', e, e) / nrm[..., None, None])
    if ROUTE == 'A':
        pairs = [(f'P{a}', mets[a], Pe[a]) for a in range(4)]
    else:
        ls = torch.sort(lam, dim=-1, descending=True).values
        Texact = 0.5 * (ls[..., 2] - ls[..., 3])[..., None, None] * (Pe[2] - Pe[3])
        pairs = [('Pt', mets[0], Pe[0]), ('P1', mets[1], Pe[1]), ('Q', mets[2], Pe[2] + Pe[3]), ('T', mets[3], Texact)]
        out['node_gaps'] = {'t-1': float((ls[..., 0] - ls[..., 1])[FREE.cpu()].min()),
                            '1-2': float((ls[..., 1] - ls[..., 2])[FREE.cpu()].min()),
                            '2-3': float((ls[..., 2] - ls[..., 3])[FREE.cpu()].min())}
    dev = {}
    for name, Pl, Pex in pairs:
        dd = (Pl.cpu() - Pex).abs().amax(dim=(-1, -2))[FREE.cpu()]
        dev[name] = {'max': float(dd.max()), 'mean': float(dd.mean()), 'q99': float(dd.quantile(0.99))}
    out['metric_vs_exact'] = dev
    # discrete Euler-Lagrange control: gradient of the lattice sum of phi
    Mv = Mg.clone().requires_grad_(True)
    gphi = torch.autograd.grad(dens_class(field(Mv), 0, 0, [ETA.expand_as(Mv)] * 4).sum(), Mv)[0]
    Mv = Mg.clone().requires_grad_(True)
    a, b = list(CLASSES.values())[0]
    gdyn = torch.autograd.grad(dens_class(field(Mv), a, b).sum(), Mv)[0]
    out['lattice_EL_ratio_phi_over_dyn'] = float(gphi.abs().max() / gdyn.abs().max())
    # directional finite-difference check of the FULL derivative (review round 0)
    fd = {}
    torch.manual_seed(0)
    V = torch.randn_like(Mr); V = 0.5 * (V + V.transpose(-1, -2)); V = V / V.norm()
    for cname, (a, b) in CLASSES.items():
        base = abs(float(H ** 3 * dens_class(Mg, a, b).sum()))
        # a class vanishing on the profile gets the probe coupling, not 5%/|0|
        lam0 = (0.05 * e_static(Mg, 'eta').item() / base) if base > 1e-10 else 1e-3 * e_static(Mg, 'eta').item()
        Mv = Mr.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(E_of(Mv, lam0, a, b), Mv)[0]
        auto = float((g * V).sum())
        eps = 1e-4
        fdv = (E_of(Mr + eps * V, lam0, a, b).item() - E_of(Mr - eps * V, lam0, a, b).item()) / (2 * eps)
        fd[cname] = {'autograd': auto, 'fd': fdv, 'rel_err': abs(auto - fdv) / max(abs(fdv), 1e-30)}
    out['directional_fd'] = fd
    return out


def relax(M_seed, lam, a, b, tag):
    M_raw = M_seed.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=LR)
    status = 'ok'
    for it in range(ADAM):
        opt.zero_grad(); E = E_of(M_raw, lam, a, b); E.backward(); opt.step()
        if (it + 1) % 100 == 0 and (not torch.isfinite(E) or E.item() < -50 or M_raw.abs().max() > 1e3):
            status = 'runaway'; break
    levels = [E_of(M_raw.detach(), lam, a, b).item()]
    if status == 'ok':
        for cyc in range(2):     # main cycle + continuation cycle
            opt2 = torch.optim.LBFGS([M_raw], max_iter=LB if cyc == 0 else LB_CONT, history_size=25,
                                     tolerance_grad=1e-9, tolerance_change=0, line_search_fn='strong_wolfe')
            def closure():
                opt2.zero_grad(); E = E_of(M_raw, lam, a, b); E.backward(); return E
            try:
                opt2.step(closure)
            except Exception as ex:
                status = f'lbfgs:{type(ex).__name__}'; break
            levels.append(E_of(M_raw.detach(), lam, a, b).item())
    E = levels[-1]
    if not np.isfinite(E) or E < -50:
        status = 'runaway'
    np.savez_compressed(os.path.join(FIELDS, f'{ROUTE}_{tag}.npz'), M=M_raw.detach().cpu().numpy())
    return M_raw.detach(), levels, status


def diagnostics(Mr, lam, a, b):
    Mf = field(Mr)
    d = {'E_stat_eta': e_static(Mf, 'eta').item()}
    dens = dens_class(Mf, a, b)
    d['lin_integral'] = float(H ** 3 * dens.sum())
    d['PR_lin'] = float((dens.abs().sum() ** 2 / (dens ** 2).sum().clamp_min(1e-300)))
    d['tail_eta'] = tail_fit(Mf, 'eta')['slope']
    d['offblock'] = offblock(Mf)
    d['grad_inf_free'] = grad_inf_free(Mr, lam, a, b)
    x = torch.einsum('ab,...bc->...ac', ETA, Mf).cpu()
    ls = torch.sort(torch.linalg.eigvals(x).real, dim=-1).values
    d['spectral_gap_top_min'] = float((ls[..., 3] - ls[..., 2])[FREE.cpu()].min())
    d['small_gap_min'] = float((ls[..., 1] - ls[..., 0])[FREE.cpu()].min())
    d['lam4_max'] = float(ls[..., 0].abs()[FREE.cpu()].max())
    return d


def main():
    print('route', ROUTE, 'device', DEV, flush=True)
    Mr = load_or_make_base()
    Mg = field(Mr)
    done = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    if 'validation' not in done:
        done['validation'] = validate(Mg, Mr)
        print('validation:', json.dumps(done['validation'])[:900], flush=True)
        for c, v in done['validation']['directional_fd'].items():
            assert v['rel_err'] < 1e-5, f'derivative check failed for {c}: {v}'
    E_stat = e_static(Mg, 'eta').item()
    base_int = {c: float(H ** 3 * dens_class(Mg, a, b).sum()) for c, (a, b) in CLASSES.items()}
    done['E_stat_base'] = E_stat; done['base_integrals'] = base_int
    print('E_stat', E_stat, 'base integrals', base_int, flush=True)
    runs = [('baseline', 0.0, *list(CLASSES.values())[0])]
    for c, (a, b) in CLASSES.items():
        if abs(base_int[c]) < 1e-10:      # class vanishing on the profile: one small-lambda probe only
            runs.append((f'{c}_probe', 1e-3 * E_stat, a, b)); continue
        for frac in (0.05, 0.2):
            lam0 = frac * E_stat / abs(base_int[c])
            for sgn in (+1, -1):
                runs.append((f'{c}_f{frac}_s{sgn:+d}', sgn * lam0, a, b))
    for tag, lam, a, b in runs:
        if tag in done:
            continue
        t0 = time.time()
        Mf, levels, status = relax(Mr, lam, a, b, tag)
        d = diagnostics(Mf, lam, a, b) if status == 'ok' else {}
        entry = {'lambda': lam, 'class': [a, b], 'E_levels': levels, 'E_total': levels[-1], 'status': status, **d}
        if status == 'ok' and len(levels) == 3:
            entry['continuation_dE'] = levels[2] - levels[1]
        if status == 'ok' and 'f0.2' in tag:      # restart check from a perturbed start
            torch.manual_seed(1)
            P = torch.randn_like(Mr); P = 0.5 * (P + P.transpose(-1, -2))
            Mp = Mr + 1e-2 * P * FREE[..., None, None].to(DT)
            Mf2, lv2, st2 = relax(Mp, lam, a, b, tag + '_restart')
            entry['restart'] = {'E_total': lv2[-1], 'status': st2,
                                'dE_vs_main': lv2[-1] - levels[-1],
                                'tail_eta': tail_fit(field(Mf2), 'eta')['slope'] if st2 == 'ok' else None}
        done[tag] = {k: v for k, v in entry.items()}
        done[tag]['seconds'] = round(time.time() - t0)
        json.dump(done, open(RESULTS, 'w'), indent=1)
        print(f"[{ROUTE}:{tag}] lam {lam:+.4g} E {levels[-1]:.6f} ({status}) levels {['%.6f' % x for x in levels]} "
              + (f"g_inf {d['grad_inf_free']:.2e} tail {d['tail_eta']:.3f} small_gap {d['small_gap_min']:.4f} lin {d['lin_integral']:+.3e}" if d else '')
              + (f" restart dE {entry['restart']['dE_vs_main']:+.2e}" if 'restart' in entry else '')
              + f" [{time.time()-t0:.0f}s]", flush=True)
    print('lattice_linear_v2 complete', flush=True)


if __name__ == '__main__':
    main()
