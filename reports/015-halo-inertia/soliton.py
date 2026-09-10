"""3D lattice: static hedgehog and rigidly rotating states (infrastructure for Task 2).

Cell-centred cubic grid of n^3 points with spacing h; Dirichlet boundary given by the asymptotic
hedgehog on one ghost layer. The field is stored as the 10 components of the symmetric M
(order 00,01,02,03,11,12,13,22,23,33). The time-like sector (components with index 0) can be
frozen at the vacuum, which is the E_0 -> infinity reduction.
"""
import time
import torch
from lagrangian import ETA, E_DEFAULT, terms, lagrangian

IDX = torch.tensor([[0, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]])
SPATIAL = torch.tensor([False, False, False, False, True, True, True, True, True, True])
G_Z = torch.zeros(4, 4, dtype=torch.float64)
G_Z[1, 2], G_Z[2, 1] = -1.0, 1.0
QUAD = (0.5 + 0.5 / 3 ** 0.5 * torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float64)).tolist()


def to_matrix(u):
    return u[..., IDX.to(u.device)]


def to_vector(M):
    return torch.stack([M[..., a, b] for a, b in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]], -1)


def hedgehog(x, E, split, width=None, degree=1):
    """Covariant M of the hedgehog: N = diag(E_0) + n(x), n = E_1 rr + E_2 tt + E_3 pp in the
    spherical frame (r, theta, phi); without split E_2 = E_3 and the frame is not needed.
    A finite width interpolates the anisotropy to zero at the origin (initial condition)."""
    r = x.norm(dim=-1).clamp_min(1e-12)
    rho = x[..., :2].norm(dim=-1).clamp_min(1e-12)
    rh = x / r[..., None]
    if degree == 0:  # constant vacuum: the charge direction along z everywhere (Task 6 sector)
        rh = torch.zeros_like(x)
        rh[..., 2] = 1.0
    elif degree != 1:  # winding number `degree` in the azimuth: v_1 = (sin th cos k ph, sin th sin k ph, cos th)
        ph = torch.atan2(x[..., 1], x[..., 0]) * degree
        rh = torch.stack([rho / r * ph.cos(), rho / r * ph.sin(), x[..., 2] / r], -1)
    th = torch.stack([x[..., 2] * x[..., 0] / rho, x[..., 2] * x[..., 1] / rho, -rho], -1) / r[..., None]
    f = 1.0 if width is None else (torch.tanh(r / width) ** 2)[..., None, None]
    E3 = E[3] if split else E[2]
    n = E3 * torch.eye(3, dtype=x.dtype, device=x.device) + f * (E[1] - E3) * rh[..., :, None] * rh[..., None, :]
    if split:
        n = n + f * (E[2] - E3) * th[..., :, None] * th[..., None, :]
    M = torch.zeros(*x.shape[:-1], 4, 4, dtype=x.dtype, device=x.device)
    M[..., 0, 0] = E[0]
    M[..., 1:, 1:] = -n
    return to_vector(M)


class Grid:
    def __init__(self, n, box, E=E_DEFAULT, split=False, freeze_time=True, device='cuda', X=None, degree=1):
        self.n, self.h, self.E, self.split, self.freeze, self.X, self.degree = n, box / n, E, split, freeze_time, X, degree
        c = (torch.arange(n, dtype=torch.float64) + 0.5) * self.h - box / 2
        cp = torch.cat([c[:1] - self.h, c, c[-1:] + self.h])
        self.x = torch.stack(torch.meshgrid(c, c, c, indexing='ij'), -1).to(device)
        lo = torch.stack(torch.meshgrid(cp[:-1], cp[:-1], cp[:-1], indexing='ij'), -1)
        self.xq = torch.stack([lo + self.h * torch.tensor(q) for q in QUAD]).to(device)
        self.vol = self.h ** 3 / len(QUAD)
        W = torch.zeros(len(QUAD), 4, 2, 2, 2, dtype=torch.float64)
        for qi, q in enumerate(QUAD):
            w = [(1 - q[d], q[d]) for d in range(3)]
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        W[qi, :, a, b, c] = torch.tensor([w[0][a] * w[1][b] * w[2][c], (2 * a - 1) * w[1][b] * w[2][c] / self.h,
                                                          (2 * b - 1) * w[0][a] * w[2][c] / self.h, (2 * c - 1) * w[0][a] * w[1][b] / self.h])
        self.stencil = W.reshape(1, 4 * len(QUAD), 1, 2, 2, 2).repeat(10, 1, 1, 1, 1, 1).reshape(-1, 1, 2, 2, 2).to(device)
        xp = torch.stack(torch.meshgrid(cp, cp, cp, indexing='ij'), -1).to(device)
        self.boundary = hedgehog(xp, E, split, degree=degree)
        self.mask = (SPATIAL if freeze_time else torch.ones(10, dtype=torch.bool)).to(device)

    def initial(self, width=1.0):
        return hedgehog(self.x, self.E, self.split, width, self.degree)

    def derivatives(self, u):
        """Trilinear interpolant of the padded lattice field and its gradient at the four
        tetrahedral quadrature points of every cube (second-order accurate and, unlike schemes that
        average differences over cube edges, free of sub-lattice null modes), as one convolution."""
        Mp = self.boundary.clone()
        Mp[1:-1, 1:-1, 1:-1] = u
        n1 = self.n + 1
        out = torch.nn.functional.conv3d(Mp.permute(3, 0, 1, 2)[None], self.stencil, groups=10)[0]
        out = out.reshape(10, len(QUAD), 4, n1, n1, n1).permute(1, 3, 4, 5, 2, 0)
        return to_matrix(out[..., 0, :]), to_matrix(torch.cat([torch.zeros_like(out[..., :1, :]), out[..., 1:, :]], -2))

    def rotation(self, M, dM):
        """d/dtheta of the rigid rotation about z acting on the tensor field."""
        G = G_Z.to(M)
        x, y = self.xq[..., 0, None, None], self.xq[..., 1, None, None]
        return G @ M + M @ G.mT + y * dM[..., 1, :, :] - x * dM[..., 2, :, :]

    def energy_terms(self, u):
        """(E4, EV, E2): quartic gradient term, potential and -integral of X of the static energy."""
        kin, pot, x = terms(*self.derivatives(u), self.E, frozen=self.freeze, X=self.X)
        return -kin.sum() * self.vol, pot.sum() * self.vol, -x.sum() * self.vol

    def energy(self, u):
        return sum(self.energy_terms(u))

    def action_density(self, u, omega):
        M, dM = self.derivatives(u)
        dM = torch.cat([(omega * self.rotation(M, dM))[..., None, :, :], dM[..., 1:, :, :]], -3)
        return lagrangian(M, dM, self.E, frozen=self.freeze, X=self.X) * self.vol

    def rotor(self, u):
        """L(omega) = a omega^2 + b omega + c for the rigid rotation; J = 2 a omega + b,
        H = omega J - L = a omega^2 - c, so at fixed J: H_J = (J - b)^2 / (4a) - c."""
        Lp, Lm, L0 = (self.action_density(u, w).sum() for w in (1.0, -1.0, 0.0))
        return 0.5 * (Lp + Lm) - L0, 0.5 * (Lp - Lm), L0

    def energy_at_J(self, u, J):
        a, b, c = self.rotor(u)
        return (J - b) ** 2 / (4 * a) - c

    def centres(self, u):
        """Centres (x, y, z) of the potential and of the gradient energy densities."""
        kin, pot, x = terms(*self.derivatives(u), self.E, frozen=self.freeze, X=self.X)
        return [(w[..., None] * self.xq).sum((0, 1, 2, 3)) / w.sum() for w in (pot, -kin)]

    def energy_at_J_pinned(self, u, J, lam=100.0):
        """H_J with the rotation axis (z) forced through the centres of the potential and of the gradient
        energy: rigid rotation about the centre of charge, so that the translation-like soft mode cannot
        supply orbital inertia. Spin is the angular momentum about that axis."""
        return self.energy_at_J(u, J) + lam * sum((c[:2] ** 2).sum() for c in self.centres(u))

    def minimize(self, u0, objective, iters=500, tol=1e-7, verbose=False, lr=1.0):
        u = u0.contiguous().clone().requires_grad_(True)
        u.register_hook(lambda gr: gr * self.mask)
        opt = torch.optim.LBFGS([u], lr=lr, max_iter=iters, tolerance_grad=tol, tolerance_change=1e-14,
                                history_size=40, line_search_fn='strong_wolfe')
        hist = []

        def closure():
            opt.zero_grad()
            f = objective(u)
            f.backward()
            hist.append(float(f.detach()))
            if verbose and len(hist) % 200 == 0:
                print(f'  {len(hist)} evaluations: {hist[-1]:.6f}  [{time.strftime("%H:%M:%S")}]', flush=True)
            return f

        opt.step(closure)
        if verbose:
            print(f'  {len(hist)} evaluations, {hist[0]:.6f} -> {hist[-1]:.6f}, |grad| = {float((u.grad * self.mask).norm()):.2e}')
        return u.detach(), float((u.grad * self.mask).norm())

    def sample(self, u, points):
        """Trilinear interpolation of the field at points [..., 3] (returns [..., 10])."""
        vol = u.permute(3, 0, 1, 2)[None]
        p = (points / (self.n * self.h / 2)).flip(-1).reshape(1, -1, 1, 1, 3)
        out = torch.nn.functional.grid_sample(vol, p, align_corners=False, padding_mode='border')
        return out[0, :, :, 0, 0].mT.reshape(*points.shape[:-1], u.shape[-1])


def hvp_operator(grid, u, objective):
    """scipy LinearOperator for the Hessian of objective on the free components."""
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    free = grid.mask.nonzero().flatten()
    base = u.clone()

    def full(p):
        v = base.clone()
        v[..., free] = p.reshape(*u.shape[:-1], len(free))
        return v

    def matvec(vec):
        p = base[..., free].clone().requires_grad_(True)
        v = torch.as_tensor(vec, dtype=u.dtype, device=u.device).reshape(p.shape)
        gr = torch.autograd.grad(objective(full(p)), p, create_graph=True)[0]
        hv = torch.autograd.grad((gr * v).sum(), p)[0]
        return hv.detach().cpu().numpy().ravel()

    dim = u[..., free].numel()
    return LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)


def refine(u_src, n_src, box_src, grid):
    """Interpolate a solution from another (coarser or smaller) lattice onto grid; outside the
    source box the asymptotic hedgehog is used."""
    vol = u_src.permute(3, 0, 1, 2)[None]
    p = (grid.x / (box_src / 2)).flip(-1).reshape(1, -1, 1, 1, 3)
    out = torch.nn.functional.grid_sample(vol, p, align_corners=False, padding_mode='border')
    out = out[0, :, :, 0, 0].mT.reshape(grid.n, grid.n, grid.n, 10)
    inside = (grid.x.abs().max(-1).values < box_src / 2 - box_src / n_src)[..., None]
    return torch.where(inside, out, hedgehog(grid.x, grid.E, grid.split, degree=grid.degree)).contiguous()
