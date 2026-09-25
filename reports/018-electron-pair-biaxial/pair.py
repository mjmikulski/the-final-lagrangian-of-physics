"""Two hedgehogs of opposite charge: the interaction energy E(d), and their annihilation.

The director of a defect-antidefect pair is built from the electrostatic analogy: with the two "charges" at
+-d/2 on the z axis, the field V = z_hat + lambda [ (x-a)/|x-a|^3 - (x+a)/|x+a|^3 ] has a radial hedgehog at
+a, a hyperbolic one at -a, and tends to z_hat far away, which is the vacuum of the constant-vacuum sector
(degree 0 boundary). The tensor is the uniaxial M with that director and a melted core at each defect.

For E(d) the cores are held apart by a penalty on the centroids of the potential-energy density in the two
half-spaces, so the field itself relaxes completely. For the annihilation the penalty is dropped and the
configuration follows the gradient flow.

python pair.py --scan 1.5 2 3 4 5           # interaction energy
python pair.py --anim 4                     # frames of the annihilation from d = 4
"""
import argparse
import json
import math
import torch
from lagrangian import terms, eigvalsh3
from soliton import Grid, to_matrix, to_vector

torch.set_default_dtype(torch.float64)


def pair_field(x, E, d, lam=None, width=0.5):
    """Uniaxial M with a hedgehog at +d/2 z_hat, an antihedgehog at -d/2 z_hat and the vacuum far away."""
    lam = 0.5 * (d / 2) ** 2 if lam is None else lam
    a = torch.zeros(3, dtype=x.dtype, device=x.device)
    a[2] = d / 2
    r1, r2 = x - a, x + a
    n1, n2 = r1.norm(dim=-1, keepdim=True).clamp_min(1e-9), r2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    z = torch.zeros_like(x)
    z[..., 2] = 1.0
    V = z + lam * (r1 / n1 ** 3 - r2 / n2 ** 3)
    nn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    f = (torch.tanh(n1 / width) ** 2 * torch.tanh(n2 / width) ** 2)[..., 0]
    M = torch.zeros(*x.shape[:-1], 4, 4, dtype=x.dtype, device=x.device)
    M[..., 0, 0] = E[0]
    aniso = E[2] * torch.eye(3, dtype=x.dtype, device=x.device) + f[..., None, None] * (E[1] - E[2]) * nn[..., :, None] * nn[..., None, :]
    M[..., 1:, 1:] = -aniso
    return to_vector(M)


def core_centres(grid, u):
    """Centroids of the potential-energy density in the upper and lower half-space, and the two weights."""
    _, pot, _ = terms(*grid.derivatives(u), grid.E, frozen=grid.freeze, X=grid.X)
    xq = grid.xq
    up = (xq[..., 2] > 0).to(pot)
    w1, w2 = pot * up, pot * (1 - up)
    c1 = (w1[..., None] * xq).sum((0, 1, 2, 3)) / w1.sum().clamp_min(1e-30)
    c2 = (w2[..., None] * xq).sum((0, 1, 2, 3)) / w2.sum().clamp_min(1e-30)
    return c1, c2, float(w1.sum() * grid.vol), float(w2.sum() * grid.vol)


def hold_cores(grid, d, radius=0.8):
    """Mask that freezes the field inside balls of the given radius around the two defects: the charges are
    held apart, as in an electrostatics calculation, and everything outside relaxes."""
    a = torch.zeros(3, dtype=grid.x.dtype, device=grid.x.device)
    a[2] = d / 2
    inside = ((grid.x - a).norm(dim=-1) < radius) | ((grid.x + a).norm(dim=-1) < radius)
    return (~inside)[..., None] & grid.mask


def signed_degree(grid, u, centre, r, n_th=48, n_ph=96):
    """Signed degree of the charge-direction line field on a sphere of radius r about `centre`."""
    th = (torch.arange(n_th, dtype=torch.float64) + 0.5) * math.pi / n_th
    ph = torch.arange(n_ph, dtype=torch.float64) * 2 * math.pi / n_ph
    T, P = torch.meshgrid(th, ph, indexing='ij')
    pts = r * torch.stack([T.sin() * P.cos(), T.sin() * P.sin(), T.cos()], -1).to(u.device) + torch.as_tensor(centre, device=u.device)
    m = -to_matrix(grid.sample(u, pts))[..., 1:, 1:]
    e, V = torch.linalg.eigh(m.cpu())
    v = V[..., :, 2].to(u.device)
    for j in range(1, v.shape[1]):                       # lift the line field to a vector field
        v[:, j] *= (torch.sign((v[:, j] * v[:, j - 1]).sum(-1, keepdim=True)).clamp(min=0) * 2 - 1)
    for i in range(1, v.shape[0]):
        v[i] *= (torch.sign((v[i] * v[i - 1]).sum(-1, keepdim=True)).clamp(min=0) * 2 - 1)
    dth, dph = float(th[1] - th[0]), float(ph[1] - ph[0])
    v_th = (v[2:] - v[:-2]) / (2 * dth)
    v_ph = (v.roll(-1, 1) - v.roll(1, 1))[1:-1] / (2 * dph)
    dens = (v[1:-1] * torch.cross(v_th, v_ph, dim=-1)).sum(-1)
    return float(dens.sum() * dth * dph / (4 * math.pi))


def director_of(grid, nn, E):
    """Uniaxial M with the given unit director field and the eigenvalues at their vacuum values."""
    n = nn / nn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    I3 = torch.eye(3, dtype=nn.dtype, device=nn.device)
    M = torch.zeros(*n.shape[:-1], 4, 4, dtype=nn.dtype, device=nn.device)
    M[..., 0, 0] = E[0]
    M[..., 1:, 1:] = -(E[2] * I3 + (E[1] - E[2]) * n[..., :, None] * n[..., None, :])
    return to_vector(M)


def ward(n, h, alpha, line=False):
    """Ward's nearest-neighbour term (hep-th/0512024): with `line=False` it penalises (1 - n.n')^2 on every
    link, so neighbours stay at acute angles and the degree of the map to S^2 cannot change on the lattice.

    With `line=True` the absolute value is taken, which is the natural term for a line field - but then the
    term permits disclinations, around which the director turns by pi with every pair of neighbours nearly
    parallel, and a disclination loop sweeping through a hedgehog unwinds its charge. That is the lattice
    face of the Alice problem of uniaxial nematics: the point charge is only defined up to the holonomy of
    the director. The charge of this model is defined with a lift, i.e. in the S^2 sector, so `line=False`
    is the term that matches the definition."""
    tot = 0
    for ax in range(3):
        xi = (n * n.roll(-1, dims=ax)).sum(-1)[tuple(slice(0, -1) if k == ax else slice(None) for k in range(3))]
        tot = tot + ((1 - (xi.abs() if line else xi)) ** 2).sum()
    return alpha * h * tot


def relax_director(grid, n, E, hold=None, steps=1500, tau=0.02, alpha=25.0, report=0):
    """Normalised gradient flow of the director alone (the sigma-model limit: the eigenvalues stay at their
    vacuum values, so the energy is exactly the quartic energy 4 Delta^4 |B|^2 of the emergent field)."""
    n = n / n.norm(dim=-1, keepdim=True)
    frozen = n.clone()
    for k in range(steps + 1):
        p = n.clone().requires_grad_(True)
        e = grid.energy(director_of(grid, p, E)) + (ward(p / p.norm(dim=-1, keepdim=True), grid.h, alpha) if alpha else 0)
        g = torch.autograd.grad(e, p)[0]
        g = g - (g * n).sum(-1, keepdim=True) * n
        if hold is not None:
            g = g * hold[..., None]
        if report and k % report == 0:
            print(f'    flow {k}: E = {float(grid.energy(director_of(grid, n, E))):.4f}, |grad| = {float(g.norm()):.2e}', flush=True)
        n = n - tau * g / (g.abs().max() + 1e-12)
        n = n / n.norm(dim=-1, keepdim=True)
        if hold is not None:
            n = torch.where(hold[..., None], n, frozen)
    return n


def relax_director_lbfgs(grid, n0, E, hold=None, iters=800, alpha=25.0):
    """Same relaxation by L-BFGS on the unnormalised vector field (normalised inside the energy). Ward's
    term keeps the texture from unwinding under the long steps of a quasi-Newton method."""
    m = n0.clone()
    if hold is not None:
        frozen = n0.clone()
    p = m.clone().requires_grad_(True)
    if hold is not None:
        p.register_hook(lambda g: g * hold[..., None])

    def objective():
        n = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if hold is not None:
            n = torch.where(hold[..., None], n, frozen)
        return grid.energy(director_of(grid, n, E)) + (ward(n, grid.h, alpha) if alpha else 0)

    opt = torch.optim.LBFGS([p], lr=1, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-14,
                            history_size=40, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        f = objective()
        f.backward()
        return f

    opt.step(closure)
    n = (p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)).detach()
    if hold is not None:
        n = torch.where(hold[..., None], n, frozen)
    return n, float((p.grad * (hold[..., None] if hold is not None else 1)).norm())


def pair_director(x, d, lam=None):
    """The director of the pair, without any melting: a genuine defect at each core.

    lambda sets the radius sqrt(lambda) out to which each defect's winding reaches; it has to stay inside the
    half separation, or a compensating defect appears next to each core, so lambda = (d/2)^2 / 2 by default."""
    lam = 0.5 * (d / 2) ** 2 if lam is None else lam
    a = torch.zeros(3, dtype=x.dtype, device=x.device)
    a[2] = d / 2
    r1, r2 = x - a, x + a
    n1, n2 = r1.norm(dim=-1, keepdim=True).clamp_min(1e-9), r2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    z = torch.zeros_like(x)
    z[..., 2] = 1.0
    V = z + lam * (r1 / n1 ** 3 - r2 / n2 ** 3)
    return V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def single_profiles(path, nbins=60):
    """Radial profiles of the two spatial eigenvalues of the relaxed single hedgehog: the gap e_1 - e_2 and
    the middle eigenvalue e_2, used to give each defect of the pair a physical melted core."""
    d = torch.load(path)
    g = Grid(d['n'], d['box'], E=d['E'], device='cpu')
    e = eigvalsh3(-to_matrix(d['u'].double())[..., 1:, 1:])
    r = g.x.norm(dim=-1).flatten()
    top, mid = e[..., 0].flatten(), e[..., 1].flatten()
    edges = torch.linspace(0, float(r.max()), nbins + 1)
    rs, gaps, mids = [], [], []
    for i in range(nbins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.any():
            rs.append(float(0.5 * (edges[i] + edges[i + 1])))
            gaps.append(float((top[m] - mid[m]).mean()))
            mids.append(float(mid[m].mean()))
    return torch.tensor(rs), torch.tensor(gaps), torch.tensor(mids)


def interp1(xs, ys, q, far):
    """Linear interpolation of a radial profile, saturating at `far` beyond the tabulated range."""
    idx = torch.searchsorted(xs.to(q.device), q.clamp(max=float(xs[-1]) - 1e-9)).clamp(1, len(xs) - 1)
    x0, x1 = xs.to(q.device)[idx - 1], xs.to(q.device)[idx]
    y0, y1 = ys.to(q.device)[idx - 1], ys.to(q.device)[idx]
    out = y0 + (y1 - y0) * (q - x0) / (x1 - x0)
    return torch.where(q > float(xs[-1]), torch.full_like(q, far), out)


def pair_full(grid, E, d, prof, hold_radius):
    """The pair in the full model: the director of the two defects, each with the melted core of the relaxed
    single hedgehog. Returns the field and the mask that holds the two cores."""
    rs, gaps, mids = prof
    a = torch.zeros(3, dtype=grid.x.dtype, device=grid.x.device)
    a[2] = d / 2
    r1 = (grid.x - a).norm(dim=-1)
    r2 = (grid.x + a).norm(dim=-1)
    n = pair_director(grid.x, d)
    D = E[1] - E[2]
    A = interp1(rs, gaps, r1, D) * interp1(rs, gaps, r2, D) / D
    c = interp1(rs, mids, r1, E[2]) + interp1(rs, mids, r2, E[2]) - E[2]
    I3 = torch.eye(3, dtype=grid.x.dtype, device=grid.x.device)
    M = torch.zeros(*grid.x.shape[:-1], 4, 4, dtype=grid.x.dtype, device=grid.x.device)
    M[..., 0, 0] = E[0]
    M[..., 1:, 1:] = -(c[..., None, None] * I3 + A[..., None, None] * n[..., :, None] * n[..., None, :])
    hold = ((r1 >= hold_radius) & (r2 >= hold_radius))[..., None] & grid.mask
    return to_vector(M), hold


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--box', type=float, default=12.0)
    ap.add_argument('--E0', type=float, default=100.0)
    ap.add_argument('--scan', type=float, nargs='*', default=None)
    ap.add_argument('--anim', type=float, default=None, help='separation to start the free relaxation from')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--every', type=int, default=8)
    ap.add_argument('--iters', type=int, default=1200)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--tag', default='pair')
    ap.add_argument('--flow', type=int, default=0, help='small-step gradient flow instead of L-BFGS (barrier check)')
    ap.add_argument('--full', action='store_true', help='the full model with physical melted cores held by Dirichlet balls')
    ap.add_argument('--single', default='results/static_base_n48.pt', help='relaxed single hedgehog for the core profile')
    ap.add_argument('--sigma', action='store_true', help='sigma-model limit: relax the director alone')
    ap.add_argument('--alpha', type=float, default=25.0, help="Ward's lattice coefficient")
    ap.add_argument('--hold', type=float, default=0.8, help='radius of the balls in which the cores are held')
    a = ap.parse_args()
    E = (a.E0, 1.0, 0.01, 0.01)
    grid = Grid(a.n, a.box, E=E, degree=0)               # degree 0: the vacuum boundary of the pair sector
    if a.sigma:
        rows = []
        for d in a.scan or []:
            # the held cores carry the pure (anti)hedgehog, the same data at every separation, so the
            # self-energy inside the balls does not depend on d and drops out of the interaction
            aa = torch.zeros(3, dtype=grid.x.dtype, device=grid.x.device)
            aa[2] = d / 2
            r1, r2 = grid.x - aa, grid.x + aa
            d1, d2 = r1.norm(dim=-1, keepdim=True).clamp_min(1e-9), r2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            n0 = pair_director(grid.x, d)
            n0 = torch.where(d1 < a.hold, r1 / d1, n0)
            n0 = torch.where(d2 < a.hold, -r2 / d2, n0)
            hold = (d1[..., 0] >= a.hold) & (d2[..., 0] >= a.hold)
            if a.flow:      # small steps: L-BFGS jumps the lattice barrier and unwinds the texture
                n = relax_director(grid, n0, E, hold=hold, steps=a.flow, alpha=a.alpha, report=a.flow // 4)
                gn = 0.0
            else:
                n, gn = relax_director_lbfgs(grid, n0, E, hold=hold, iters=a.iters, alpha=a.alpha)
            u = director_of(grid, n, E)
            E4, EV, _ = (float(t) for t in grid.energy_terms(u))
            aa = torch.zeros(3, dtype=u.dtype, device=u.device); aa[2] = d / 2
            deg = [[signed_degree(grid, u, c, rr) for rr in (0.75, 0.9, min(1.4, 0.45 * d))] for c in (aa, -aa)]
            print(f'd = {d:4.1f}: E = {E4 + EV:.4f} (E4 {E4:.3f}, EV {EV:.4f}), degrees at r = 0.75/0.9/outer: '
                  f'{[round(x, 3) for x in deg[0]]} and {[round(x, 3) for x in deg[1]]}, |grad| {gn:.1e}', flush=True)
            rows.append(dict(d=d, E=E4 + EV, E4=E4, EV=EV, degrees=deg, grad=gn))
            torch.save({'u': u.cpu(), 'n': a.n, 'box': a.box, 'E': E, 'split': False, 'frozen': True, 'X': None,
                        'degree': 0, 'E4': E4, 'EV': EV, 'E2': 0.0, 'grad': 0.0, 'd': d}, f'results/{a.tag}_sigma_d{d:g}.pt')
            json.dump(rows, open(f'results/{a.tag}_sigma_scan.json', 'w'), indent=1)
        if a.anim is not None:
            # free flow: nothing holds the cores, so the pair follows the gradient of its own energy
            n = pair_director(grid.x, a.anim)
            n = n / n.norm(dim=-1, keepdim=True)
            frames, tau = [], 0.02
            for k in range(a.steps + 1):
                p = n.clone().requires_grad_(True)
                e = grid.energy(director_of(grid, p, E)) + ward(p / p.norm(dim=-1, keepdim=True), grid.h, a.alpha)
                g = torch.autograd.grad(e, p)[0]
                g = g - (g * n).sum(-1, keepdim=True) * n
                if k % a.every == 0:
                    u = director_of(grid, n, E)
                    E4, EV, _ = (float(t) for t in grid.energy_terms(u))
                    frames.append(dict(step=k, E=E4 + EV, n=n.cpu().clone()))
                    print(f'step {k}: E = {E4 + EV:.4f}', flush=True)
                n = n - tau * g / (g.abs().max() + 1e-12)
                n = n / n.norm(dim=-1, keepdim=True)
            torch.save({'frames': frames, 'n': a.n, 'box': a.box, 'E': E, 'd0': a.anim}, f'results/{a.tag}_sigma_annihilation.pt')
        raise SystemExit
    if a.full and a.flow:
        # careful check of the barrier: small-step gradient flow with the cores held, watching the charge
        prof = single_profiles(a.single)
        d = a.scan[0]
        u, hold = pair_full(grid, E, d, prof, a.hold)
        aa = torch.zeros(3, dtype=u.dtype, device=u.device); aa[2] = d / 2
        tau = 2e-3
        for k in range(a.flow + 1):
            with torch.no_grad():
                Ecur = float(grid.energy(u))
            if k % max(1, a.flow // 12) == 0:
                deg = [signed_degree(grid, u, c, 1.6) for c in (aa, -aa)]
                print(f'flow {k:5d}: E = {Ecur:9.4f}, degrees {deg[0]:+.3f} {deg[1]:+.3f}', flush=True)
            p_ = u.clone().requires_grad_(True)
            g = torch.autograd.grad(grid.energy(p_), p_)[0] * hold
            while True:
                trial = (u - tau * g).detach()
                with torch.no_grad():
                    if float(grid.energy(trial)) <= Ecur or tau < 1e-7:
                        break
                tau *= 0.5
            u = trial
        torch.save({'u': u.cpu(), 'n': a.n, 'box': a.box, 'E': E, 'd': d}, f'results/{a.tag}_full_flow_d{d:g}.pt')
        raise SystemExit
    if a.full:
        prof = single_profiles(a.single)
        rows = []
        for d in a.scan or []:
            u, hold = pair_full(grid, E, d, prof, a.hold)
            tau, gn = 2e-3, 0.0
            for k in range(a.steps):          # small-step flow: L-BFGS jumps the barrier and unwinds the pair
                with torch.no_grad():
                    Ecur = float(grid.energy(u))
                p_ = u.clone().requires_grad_(True)
                g = torch.autograd.grad(grid.energy(p_), p_)[0] * hold
                gn = float(g.norm())
                while True:
                    trial = (u - tau * g).detach()
                    with torch.no_grad():
                        if float(grid.energy(trial)) <= Ecur or tau < 1e-7:
                            break
                    tau *= 0.5
                u = trial
            E4, EV, _ = (float(t) for t in grid.energy_terms(u))
            aa = torch.zeros(3, dtype=u.dtype, device=u.device); aa[2] = d / 2
            deg = [[signed_degree(grid, u, c, rr) for rr in (1.4, 1.8)] for c in (aa, -aa)]  # outside the held ball
            print(f'd = {d:4.1f}: E = {E4 + EV:.4f} (E4 {E4:.3f}, EV {EV:.3f}), degrees at r = 1.4/1.8: '
                  f'{[round(x, 3) for x in deg[0]]} and {[round(x, 3) for x in deg[1]]}, |grad| {gn:.1e}', flush=True)
            rows.append(dict(d=d, E=E4 + EV, E4=E4, EV=EV, degrees=deg, grad=gn))
            torch.save({'u': u.cpu(), 'n': a.n, 'box': a.box, 'E': E, 'split': False, 'frozen': True, 'X': None,
                        'degree': 0, 'E4': E4, 'EV': EV, 'E2': 0.0, 'grad': gn, 'd': d}, f'results/{a.tag}_full_d{d:g}.pt')
            json.dump(rows, open(f'results/{a.tag}_full_scan.json', 'w'), indent=1)
        raise SystemExit
    if a.scan:
        rows = []
        for d in a.scan:
            u = pair_field(grid.x, E, d)
            free = grid.mask
            grid.mask = hold_cores(grid, d, a.hold)
            for _ in range(a.rounds):
                u, g = grid.minimize(u, grid.energy, iters=a.iters)
            grid.mask = free
            E4, EV, _ = (float(t) for t in grid.energy_terms(u))
            c1, c2, w1, w2 = core_centres(grid, u)
            deg = [signed_degree(grid, u, c.detach(), 0.8) for c in (c1, c2)]
            sep = float((c1 - c2).norm())
            row = dict(d=d, separation=sep, E=E4 + EV, E4=E4, EV=EV, grad=g, degrees=deg,
                       centres=[c1.tolist(), c2.tolist()])
            print(f'd = {d:4.1f}: separation {sep:.3f}, E = {E4 + EV:.4f} (E4 {E4:.3f}, EV {EV:.3f}), '
                  f'degrees {deg[0]:+.2f} {deg[1]:+.2f}, |grad| {g:.1e}', flush=True)
            rows.append(row)
            torch.save({'u': u.cpu(), 'n': a.n, 'box': a.box, 'E': E, 'split': False, 'frozen': True, 'X': None,
                        'degree': 0, 'E4': E4, 'EV': EV, 'E2': 0.0, 'grad': g, 'd': d}, f'results/{a.tag}_d{d:g}.pt')
            json.dump(rows, open(f'results/{a.tag}_scan.json', 'w'), indent=1)
    if a.anim is not None:
        u = pair_field(grid.x, E, a.anim)
        frames, tau = [], 2e-3
        for k in range(a.steps + 1):
            with torch.no_grad():
                Ecur = float(grid.energy(u))
            if k % a.every == 0:
                c1, c2, w1, w2 = core_centres(grid, u)
                frames.append(dict(step=k, E=Ecur, sep=float((c1 - c2).norm()), u=u.cpu().clone()))
                print(f'step {k}: E = {Ecur:.4f}, core separation {float((c1 - c2).norm()):.3f}', flush=True)
            p = u.clone().requires_grad_(True)
            g = torch.autograd.grad(grid.energy(p), p)[0] * grid.mask
            while True:
                trial = (u - tau * g).detach()
                with torch.no_grad():
                    if float(grid.energy(trial)) <= Ecur or tau < 1e-7:
                        break
                tau *= 0.5
            u = trial
        torch.save({'frames': frames, 'n': a.n, 'box': a.box, 'E': E}, f'results/{a.tag}_annihilation.pt')
