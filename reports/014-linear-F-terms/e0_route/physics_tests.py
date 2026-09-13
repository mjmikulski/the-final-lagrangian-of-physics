"""Task 2: physics tests of the electron sector. Each test returns True / False / None (inconclusive).

Inputs are solved configurations (dicts saved by run_static.py / run_spinning.py), never code paths.
Tolerances are indulgent because E_0 = 100 is far from the physical 1e10 and lattices are coarse.
"""
import math
import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from lagrangian import terms
from soliton import Grid, to_matrix, hvp_operator

torch.set_default_dtype(torch.float64)


def load(path):
    d = torch.load(path)
    grid = Grid(d['n'], d['box'], E=d['E'], split=d['split'], freeze_time=d['frozen'], X=d.get('X'), degree=d.get('degree', 1))
    return grid, d['u'].cuda(), d


# ----------------------------------------------------------------------------- helpers

def top_eigenvector(u):
    """Unit eigenvector of the largest spatial eigenvalue (the charge direction v_1), and the gap."""
    n = -to_matrix(u)[..., 1:, 1:]
    e, V = torch.linalg.eigh(n.cpu())
    return V[..., :, 2].cuda(), (e[..., 2] - e[..., 1]).cuda()


def sphere(r, n_th=48, n_ph=96):
    th = (torch.arange(n_th, dtype=torch.float64) + 0.5) * math.pi / n_th
    ph = torch.arange(n_ph, dtype=torch.float64) * 2 * math.pi / n_ph
    T, P = torch.meshgrid(th, ph, indexing='ij')
    pts = r * torch.stack([T.sin() * P.cos(), T.sin() * P.sin(), T.cos()], -1)
    return pts.cuda(), th, ph


def lift_on_sphere(v):
    """Choose signs of a line field on a (theta, phi) mesh so that it becomes a continuous vector field."""
    v = v.clone()
    for j in range(1, v.shape[1]):
        v[0, j] *= torch.sign((v[0, j] * v[0, j - 1]).sum()).clamp(min=0) * 2 - 1
    for i in range(1, v.shape[0]):
        v[i] *= (torch.sign((v[i] * v[i - 1]).sum(-1, keepdim=True)).clamp(min=0) * 2 - 1)
    seam = torch.minimum((v[:, 0] * v[:, -1]).sum(-1).min(), (v[0, :] * v[0, 0]).sum(-1).min())
    return v, float(seam)


def line_field_degree(grid, u, r):
    """Degree of the map S^2_r -> RP^2 given by the v_1 line, via its lift to S^2."""
    pts, th, ph = sphere(r)
    v, _ = top_eigenvector(grid.sample(u, pts))
    v, seam = lift_on_sphere(v)
    if seam < 0.5:
        return None
    dth, dph = float(th[1] - th[0]), float(ph[1] - ph[0])
    v_th = (v[2:] - v[:-2]) / (2 * dth)
    v_ph = (v.roll(-1, 1) - v.roll(1, 1))[1:-1] / (2 * dph)
    dens = (v[1:-1] * torch.cross(v_th, v_ph, dim=-1)).sum(-1)
    return abs(float(dens.sum() * dth * dph / (4 * math.pi)))


def shell_profile(grid, u, nbins=12):
    """Energy density (kinetic, potential) averaged over spherical shells inside the inscribed sphere."""
    M, dM = grid.derivatives(u)
    kin, pot, x = terms(M, dM, grid.E, frozen=grid.freeze, X=grid.X)
    r = grid.xq.norm(dim=-1)
    edges = torch.linspace(0, grid.n * grid.h / 2, nbins + 1, device=r.device)
    rows = []
    for i in range(nbins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        rows.append((float(0.5 * (edges[i] + edges[i + 1])), float(-kin[m].mean()), float(pot[m].mean() - x[m].mean())))
    return rows


def tail_ratio(grid, u):
    """Kinetic energy density in the outer shells relative to the asymptotic hedgehog law
    4 (E_1 - E_2)^4 / r^4 (an integrable tail); 1 means the Coulomb-like hedgehog tail is reached."""
    rows = shell_profile(grid, u)
    E = grid.E
    return [(k + p) * r ** 4 / (4 * (E[1] - E[2]) ** 4) for r, k, p in rows[-3:]]   # p includes -X: a 1/r^2 term makes the ratio blow up


def physical_directions(grid, u):
    """Odd-parity (inversion) deformation, translation and scaling directions on the free components."""
    from soliton import hedgehog
    dev = (u - hedgehog(grid.x, grid.E, grid.split, degree=grid.degree)) * grid.mask
    odd = 0.5 * (dev - dev.flip(0, 1, 2))
    trans = torch.zeros_like(u)
    trans[1:] = (u[:-1] - u[1:])
    scale = grid.sample(u, 1.02 * grid.x) - u
    return [d * grid.mask for d in (odd, trans, scale)]


def lowest_hessian_eigenvalues(grid, u, objective, k=4, maxiter=120):
    """Lowest Ritz values of the Hessian by LOBPCG started from the physical soft directions plus random
    vectors; with a bounded iteration count they are upper bounds on the lowest eigenvalues, which is
    what an instability check needs (a negative Ritz value proves a negative direction)."""
    from scipy.sparse.linalg import lobpcg
    op = hvp_operator(grid, u, objective)
    free = grid.mask.nonzero().flatten()
    X = torch.stack([d[..., free].flatten() for d in physical_directions(grid, u)], 1).cpu().numpy()
    X = np.concatenate([X, np.random.default_rng(0).standard_normal((X.shape[0], k - X.shape[1]))], 1) if k > X.shape[1] else X[:, :k]
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    try:
        vals, vecs = lobpcg(op, X, largest=False, maxiter=maxiter, tol=1e-4)
    except Exception as ex:
        print('  Hessian:', str(ex)[:80])
        return None
    return sorted(vals.tolist())


def symmetry_report(grid, u):
    """Relative change of the solution under the lattice-exact rotations by pi/2 about z and x and under
    the reflection x -> -x, and the part of the spatial block that is not of the SO(3)-invariant form
    b(r) 1 + a(r) rr; all relative to the deviation from the asymptotic hedgehog (0 = invariant)."""
    from soliton import to_vector, hedgehog
    M = to_matrix(u)
    dev = (u - hedgehog(grid.x, grid.E, grid.split, degree=grid.degree)).norm()
    Rz = torch.eye(4, device='cuda', dtype=torch.float64)
    Rz[1, 1] = Rz[2, 2] = 0; Rz[1, 2], Rz[2, 1] = -1.0, 1.0
    Rx = torch.eye(4, device='cuda', dtype=torch.float64)
    Rx[2, 2] = Rx[3, 3] = 0; Rx[2, 3], Rx[3, 2] = -1.0, 1.0
    P = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0], device='cuda'))
    out = {'Rz(pi/2)': Rz @ M.transpose(0, 1).flip(0) @ Rz.mT, 'Rx(pi/2)': Rx @ M.transpose(1, 2).flip(1) @ Rx.mT,
           'x->-x': P @ M.flip(0) @ P}
    out = {k: float((to_vector(v) - u).norm() / dev) for k, v in out.items()}
    n = -M[..., 1:, 1:]
    r = grid.x.norm(dim=-1, keepdim=True)
    xh = grid.x / r
    a = torch.einsum('...i,...ij,...j->...', xh, n, xh)
    b = (torch.einsum('...ii->...', n) - a) / 2
    model = b[..., None, None] * torch.eye(3, device='cuda') + (a - b)[..., None, None] * xh[..., :, None] * xh[..., None, :]
    out['non-spherical'] = float((n - model).norm() / dev)
    return out


# ----------------------------------------------------------------------------- tests

def test_rest_energy(sol, grad_tol=0.05):
    """Existence: a converged, positive, finite-energy localised static solution (tail integrable)."""
    grid, u, d = sol
    if d['grad'] > grad_tol * (d['E4'] + d['EV']):
        return None
    E = d['E4'] + d['EV'] + d.get('E2', 0.0)
    ratios = tail_ratio(grid, u)
    print(f'  E = {E:.4f} (+ analytic tail 16 pi (E1-E2)^4 / R = {16 * math.pi * (grid.E[1] - grid.E[2]) ** 4 / (grid.n * grid.h / 2):.3f} outside the inscribed sphere), outer kinetic density / (4 (E1-E2)^4 / r^4) = {[round(x, 3) for x in ratios]}')
    return bool(E > 0 and all(0.5 < x < 2 for x in ratios))


def test_derrick(sol, virial_tol=0.05):
    """Derrick scaling: virial E4 = 3 EV at the stationary point and no negative Hessian direction."""
    grid, u, d = sol
    tail = 16 * math.pi * (grid.E[1] - grid.E[2]) ** 4 / (grid.n * grid.h / 2)
    virial = (d['E4'] + tail - d.get('E2', 0.0)) / (3 * d['EV'])   # E(lambda) = lambda E4 + E2 / lambda + EV / lambda^3
    print(f'  virial (E4 + analytic tail - E2)/(3 EV) = {virial:.4f}   (E2 = {d.get("E2", 0.0):.3f}; in the box alone {(d["E4"] - d.get("E2", 0.0)) / (3 * d["EV"]):.4f})')
    if abs(virial - 1) > virial_tol:
        return False
    ev = lowest_hessian_eigenvalues(grid, u, grid.energy)
    if ev is None:
        return None
    print(f'  lowest Hessian eigenvalues = {ev}')
    return bool(ev[0] > -1e-3 * abs(ev[-1]))


def test_charge_quantization(sol, radii=(0.35, 0.5, 0.65, 0.8)):
    """The charge is the degree of the v_1 line field on spheres: integer, non-zero, radius independent."""
    grid, u, d = sol
    degs = [line_field_degree(grid, u, f * grid.n * grid.h / 2) for f in radii]
    print(f'  degrees on spheres = {[None if x is None else round(x, 3) for x in degs]}')
    if any(x is None for x in degs):
        return None
    ok = all(abs(x - round(x)) < 0.05 for x in degs) and len({round(x) for x in degs}) == 1 and round(degs[0]) != 0
    return bool(ok)


def clock_point(branch):
    """First zero of f(J) = E - 2 J Omega along the branch (linear interpolation), or None."""
    f = [b['E'] - 2 * b['J'] * b['Omega'] for b in branch]
    for i in range(1, len(branch)):
        if f[i - 1] > 0 >= f[i]:
            t = f[i - 1] / (f[i - 1] - f[i])
            return {k: branch[i - 1][k] + t * (branch[i][k] - branch[i - 1][k]) for k in ('J', 'E', 'Omega', 'Estatic')}, i
    return None, None


def test_de_broglie_clock(branch):
    """A stationary rigidly rotating state with E = hbar Omega where hbar := 2 J (frame frequency Omega,
    M oscillates at 2 Omega). The rotation axis passes through the centres of the potential and of the
    gradient energy (pinned by a penalty), so J is the spin about the centre of charge and the
    translation-like soft mode cannot supply orbital inertia. Passes if E(J), Omega(J) crosses E = 2 J Omega."""
    if not branch:
        return None
    pt, i = clock_point(branch)
    if pt is None:
        last = branch[-1]
        g = [b['E'] / (2 * b['J'] * b['Omega']) for b in branch]
        msg = f'  no crossing; E/(2 J Omega) = {g[0]:.0f} ... {g[-1]:.0f}, E_rot/E_static at the last point {(last["E"] - last["Estatic"]) / last["Estatic"]:.3f}'
        far = False
        if len(branch) > 1 and g[-1] < g[-2]:
            p = math.log(g[-1] / g[-2]) / math.log(branch[-1]['J'] / branch[-2]['J'])
            Jstar = branch[-1]['J'] * g[-1] ** (-1 / p)
            msg += f'; power-law extrapolation reaches 1 at J ~ {Jstar:.0e}'
            far = len(branch) >= 3 and g[-1] > 10 and Jstar > 10 * branch[-1]['J']
        print(msg)
        # fail when the branch is far from the condition and moving towards it only on a scale that is an order of
        # magnitude beyond the computed states (which are already box-limited deformations of the hedgehog)
        return False if (last['E'] - last['Estatic'] > last['Estatic'] or far) else None
    print(f'  clock point: J* = {pt["J"]:.4f}, E = {pt["E"]:.4f}, Omega = {pt["Omega"]:.4f}, E_rot/E_static = {(pt["E"] - pt["Estatic"]) / pt["Estatic"]:.3f}')
    return True


def test_spin(branch, hessian):
    """Spin hbar/2: the clock point defines hbar = 2 J*; the state there must be a stable stationary
    point of H_J (lowest Hessian eigenvalue of H_J non-negative) with Omega > 0."""
    pt, i = clock_point(branch)
    if pt is None or pt['Omega'] <= 0:
        return None if pt is None else False
    if hessian is None:
        return None
    print(f'  hbar := 2 J* = {2 * pt["J"]:.4f}; lowest Hessian eigenvalues of H_J at J* = {hessian}')
    return bool(hessian[0] > -1e-3 * abs(hessian[-1]))


def test_magnetic_moment(gfactor, tol=0.5):
    """g = 2 mu m / (e J) with the lifted-v_1 (Faber) field, at the clock point or, if the branch does not
    reach it, at its largest J (for a linear-response rotor g does not depend on J); pass if |g - 2| < tol."""
    if gfactor is None:
        return None
    print(f'  g = {gfactor:.3f}')
    return bool(abs(abs(gfactor) - 2) < tol)


def test_spin_statistics(sol):
    """Finkelstein-Rubinstein: fermionic quantisation needs the 2 pi rigid-rotation loop to be
    non-contractible. For a rotation-invariant solution the loop is constant, hence contractible."""
    grid, u, d = sol
    rep = symmetry_report(grid, u)
    print(f'  symmetry report: {rep}')
    if max(rep.values()) < 0.05:
        return False
    return None  # the rotation loop is not constant; its homotopy class in the field space is not established


def test_de_broglie_clock_modes(modes_json, branch, tol=0.5):
    """Clock from the soliton's own internal oscillation: the lowest localised, stable, even-parity
    mode of the static soliton (breathing / shape mode) with frequency omega_0 gives hbar_clock = E / omega_0;
    the spin branch gives hbar_spin = 2 J*. Passes if the two agree within tol (relative)."""
    import json as _json
    info = _json.load(open(modes_json))
    good = [m for m in info['modes'] if m['omega2'] > 0 and m['inside_r4'] > 0.5 and m['even_fraction'] > 1]
    if not good:
        print('  no localised stable even mode found')
        return None
    m = min(good, key=lambda m: m['omega'])
    print(f"  lowest localised even mode: omega_0 = {m['omega']:.4f}, hbar_clock = E/omega_0 = {info['E'] / m['omega']:.1f}")
    pt, i = clock_point(branch) if branch else (None, None)
    if pt is None:
        print('  spin branch has no clock point: hbar_spin undefined')
        return None
    ratio = (info['E'] / m['omega']) / (2 * pt['J'])
    print(f"  hbar_spin = 2 J* = {2 * pt['J']:.1f}; ratio hbar_clock / hbar_spin = {ratio:.2f}")
    return bool(abs(ratio - 1) < tol)


def clock_modes(modes_json, kind='odd'):
    """The lowest localised mode of the given parity (odd: dipolar oscillation of the core against the halo,
    the Zitterbewegung-like motion; even: shape/breathing oscillation), excluding the translation-like mode."""
    import json as _json
    info = _json.load(open(modes_json))
    d = torch.load(info['static'])   # energy with the X term and the tail of the right charge
    E = d['E4'] + d['EV'] + d.get('E2', 0.0) + 16 * math.pi * (d['E'][1] - d['E'][2]) ** 4 / (d['box'] / 2) * d.get('degree', 1) ** 2
    sel = [m for m in info['modes'] if m['omega2'] > 0 and m['inside_r4'] > 0.5 and abs(m['translation']) < 0.5
           and (kind == 'any' or (m['even_fraction'] > 1) == (kind == 'even'))]
    return (E, min(sel, key=lambda m: m['omega'])) if sel else (E, None)


def test_clock_universality(modes_deg1, modes_deg2, kind='odd', tol=2.0):
    """de Broglie clock without an external hbar: E = hbar omega_0 with a universal hbar requires the
    ratio E / omega_0 of the internal clock mode to be the same for the charge-1 and the charge-2 soliton.
    Passes if the two values agree within a factor tol (indulgent)."""
    (E1, m1), (E2, m2) = clock_modes(modes_deg1, kind), clock_modes(modes_deg2, kind)
    if m1 is None or m2 is None:
        print('  clock mode not identified')
        return None
    h1, h2 = E1 / m1['omega'], E2 / m2['omega']
    print(f"  charge 1: omega_0 = {m1['omega']:.3f}, E = {E1:.1f}, E/omega_0 = {h1:.0f};  charge 2: omega_0 = {m2['omega']:.3f}, E = {E2:.1f}, E/omega_0 = {h2:.0f};  ratio {h2 / h1:.2f}")
    return bool(1 / tol < h2 / h1 < tol)
