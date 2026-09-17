"""Report-004/008 lattice stack with the box size as a parameter (fixed spacing h = 1.5).

Everything is defined exactly as in reports/004-lattice-clock/lattice.py, polish_hess.py and
reports/008-i1-squared-clock/ladder_i1sq.py, with N (and hence L = N h) an argument instead of a module
constant: symmetrized one-sided stencils, pinned shell of physical depth 1.6, the 4-target trace potential
V4, the Lagrange-projector Euclideanizer G, the envelope-localized frozen tangent (radius 10), the (I1^G)^2
energy-functional densities and the fixed-depth relaxation protocols. The seed is the analytic hedgehog of
the M5.21.2b instrument (seed A: eigenvalue 1 on r-hat, delta on phi-hat, 0 on theta-hat, core radius 4,
escaped isotropic centre) embedded in 4x4 with M_00 = -g; the 004 chain (eta relax, G relax, polish) is run
from it at every N. At N = 32 the same seed values sit on the pinned shell as in the committed 004 field.
"""
import time

import numpy as np
import torch

SG, DELTA, W1 = 8.0, 0.3, 0.000724023879
C_P = tuple(SG ** p + 1.0 + DELTA ** p for p in range(1, 5))
DT = torch.float64


class Stack:
    def __init__(self, N, H=1.5, dev=None, seed3=None):
        """seed3: the 3x3 endpoint of the M5.21.2b instrument on this grid (numpy [N, N, N, 3, 3]); without it
        the analytic ansatz is embedded (only for the seed-shortcut check, which fails at N = 32)."""
        self.N, self.H, self.L = N, H, N * H
        self.dev = dev or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed3_given = seed3
        self.ETA = torch.diag(torch.tensor([-1.0, 1, 1, 1], dtype=DT, device=self.dev))
        self.M_VAC = torch.diag(torch.tensor([-SG, 1.0, DELTA, 0.0], dtype=DT, device=self.dev))
        wc = max(1, int(np.ceil(1.6 / H)))
        P = torch.zeros(N, N, N, dtype=torch.bool, device=self.dev)
        for ax in range(3):
            s = [slice(None)] * 3
            s[ax] = slice(0, wc); P[tuple(s)] = True
            s[ax] = slice(N - wc, N); P[tuple(s)] = True
        self.SHELL, self.FREE = P, ~P
        self.SEED = self.seed_embedded(None if seed3 is None else torch.tensor(np.asarray(seed3), dtype=DT, device=self.dev))
        self.mask = self.SHELL[..., None, None].to(DT)
        self.fmask = self.FREE[..., None, None].to(DT)

    # ---------- grid and seed ----------
    def coords(self):
        x = (torch.arange(self.N, dtype=DT, device=self.dev) - (self.N - 1) / 2.0) * self.H
        return torch.meshgrid(x, x, x, indexing="ij")

    def seed3_analytic(self, r_c=4.0):
        """M5.21.2b seed A in physical coordinates: lam = (1, delta, 0) on (r-hat, phi-hat, theta-hat)."""
        X, Y, Z = self.coords()
        r = torch.sqrt(X * X + Y * Y + Z * Z)
        rs = torch.where(r < 1e-12, torch.full_like(r, 1e-12), r)
        rhat = torch.stack([X / rs, Y / rs, Z / rs], -1)
        rho = torch.sqrt(X * X + Y * Y)
        rhos = torch.where(rho < 1e-12, torch.full_like(rho, 1e-12), rho)
        phihat = torch.stack([-Y / rhos, X / rhos, torch.zeros_like(Z)], -1)
        near = rho < 1e-9
        phihat[near] = torch.tensor([0.0, 1.0, 0.0], dtype=DT, device=self.dev)
        that = torch.cross(phihat, rhat, dim=-1)
        S = (1.0 * rhat[..., :, None] * rhat[..., None, :] + DELTA * phihat[..., :, None] * phihat[..., None, :]
             + 0.0 * that[..., :, None] * that[..., None, :])
        a = (1.0 + DELTA) / 3.0
        w = (1.0 - torch.exp(-((r / r_c) ** 2)))[..., None, None]
        return w * S + (1.0 - w) * (a * torch.eye(3, dtype=DT, device=self.dev))

    def seed_embedded(self, M3=None):
        M3 = self.seed3_analytic() if M3 is None else M3
        M4 = torch.zeros(self.N, self.N, self.N, 4, 4, dtype=DT, device=self.dev)
        M4[..., 1:4, 1:4] = M3
        M4[..., 0, 0] = -SG
        return M4

    # ---------- the 004 functional ----------
    def d1(self, f, ax, st):
        out = torch.zeros_like(f)
        idx = [slice(None)] * f.ndim

        def at(i):
            s = list(idx); s[ax] = i; return tuple(s)
        if st == "fwd":
            out[at(slice(0, -1))] = (f[at(slice(1, None))] - f[at(slice(0, -1))]) / self.H
        else:
            out[at(slice(1, None))] = (f[at(slice(1, None))] - f[at(slice(0, -1))]) / self.H
        return out

    @staticmethod
    def sym4(X):
        return 0.5 * (X + X.transpose(-1, -2))

    def field(self, M_raw):
        return self.mask * self.SEED + (1 - self.mask) * self.sym4(M_raw)

    def G_of(self, M):
        x = torch.einsum("ab,...bc->...ac", self.ETA, M)
        I4 = torch.eye(4, dtype=DT, device=self.dev).expand_as(M)
        q = (x @ (x - I4) @ (x - DELTA * I4)) / (SG * (SG - 1) * (SG - DELTA))
        return self.ETA - 2.0 * q @ self.ETA

    def inner_X(self, F, X):
        if X.dim() == 2:
            return torch.einsum("...ab,ac,bd,...cd->...", F, X, X, F)
        return torch.einsum("...ab,...ac,...bd,...cd->...", F, X, X, F)

    def comm(self, A, B):
        return A @ self.ETA @ B - B @ self.ETA @ A

    def e_static(self, M, metric):
        X = self.ETA if metric == "eta" else self.G_of(M)
        e_u = 0.0
        for st in ("fwd", "bwd"):
            A = [self.d1(M, ax, st) for ax in range(3)]
            for i in range(3):
                for j in range(i + 1, 3):
                    F = self.comm(A[i], A[j])
                    e_u = e_u + 0.5 * 4.0 * self.inner_X(F, X).sum()
        Me = M @ self.ETA
        P, v4 = Me, 0.0
        for p in range(4):
            if p:
                P = P @ Me
            t = torch.einsum("...kk->...", P)
            v4 = v4 + (t - C_P[p]) ** 2
        return self.H ** 3 * (e_u + W1 * v4.sum())

    # ---------- tangent and the 008 densities ----------
    def envelope(self):
        X, Y, Z = self.coords()
        r = torch.sqrt(X * X + Y * Y + Z * Z)
        return torch.exp(-((r / 10.0) ** 4))

    def gen_boost_x(self):
        W = torch.zeros(4, 4, dtype=DT, device=self.dev)
        W[0, 1] = W[1, 0] = 1.0
        return W

    def a0_of(self, W, M):
        a = self.envelope()[..., None, None] * (torch.einsum("ab,...bc->...ac", W, M)
                                                 + torch.einsum("...ab,cb->...ac", M, W))
        return a / a.norm()

    def densities(self, M, a0, om, metric="G"):
        G = self.G_of(M) if metric == "G" else None
        V = om * a0
        i1s = torch.zeros(M.shape[:3], dtype=DT, device=self.dev)
        k = torch.zeros(M.shape[:3], dtype=DT, device=self.dev)
        for st in ("fwd", "bwd"):
            A = [self.d1(M, ax, st) for ax in range(3)]
            for i in range(3):
                F0 = self.comm(V, A[i])
                if metric == "G":
                    k = k + 0.5 * 4.0 * self.inner_X(F0, G)
                else:
                    k = k + 0.5 * 4.0 * (-1.0) * self.inner_X(F0, self.ETA)
                for j in range(i + 1, 3):
                    F = self.comm(A[i], A[j])
                    if metric == "G":
                        i1s = i1s + 0.5 * 4.0 * self.inner_X(F, G)
                    else:
                        i1s = i1s + 0.5 * 4.0 * self.inner_X(F, self.ETA)
        return i1s, k

    def tail_fit(self, M, metric="G"):
        X = self.ETA if metric == "eta" else self.G_of(M)
        dens = torch.zeros(M.shape[:3], dtype=DT, device=self.dev)
        for st in ("fwd", "bwd"):
            A = [self.d1(M, ax, st) for ax in range(3)]
            for i in range(3):
                for j in range(i + 1, 3):
                    F = self.comm(A[i], A[j])
                    dens = dens + 0.5 * 4.0 * self.inner_X(F, X)
        Xc, Yc, Zc = self.coords()
        r = torch.sqrt(Xc ** 2 + Yc ** 2 + Zc ** 2)
        rs, us = [], []
        for kk in range(10):
            lo, hi = 8.0 + 0.8 * kk, 8.0 + 0.8 * (kk + 1)
            m = (r >= lo) & (r < hi)
            if m.sum() > 0 and dens[m].mean() > 0:
                rs.append((lo + hi) / 2)
                us.append(dens[m].mean().item())
        slope, logc = np.polyfit(np.log(rs), np.log(us), 1)
        return {"slope": float(slope), "coeff": float(np.exp(logc))}

    # ---------- the 004 relaxations ----------
    def relax_adam(self, M_raw, metric, steps, lr=5e-4, tag="", log_every=500):
        M_raw = M_raw.clone().requires_grad_(True)
        opt = torch.optim.Adam([M_raw], lr=lr)
        t0 = time.time()
        for it in range(steps):
            opt.zero_grad()
            E = self.e_static(self.field(M_raw), metric)
            E.backward()
            opt.step()
            if (it + 1) % log_every == 0:
                print(f"  {tag} it {it+1:5d}  E {E.item():.6f} [{time.time()-t0:.0f}s]", flush=True)
        return M_raw.detach()

    def grad_inf(self, m):
        m = m.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(self.e_static(self.field(m), "G"), m)
        g = self.sym4(g) * self.fmask
        return g.abs().max().item()

    def polish(self, M_raw, tag=""):
        """polish_hess.py: Adam annealed (1e-3 x3000, 2e-4 x3000, 5e-5 x2000) then L-BFGS outer loops
        (40 iterations each, up to 12) stopping at |g|_inf < 1e-4; gradient masked to the free cells."""
        M = M_raw.clone().requires_grad_(True)
        t0 = time.time()
        for lr, steps in ((1e-3, 3000), (2e-4, 3000), (5e-5, 2000)):
            opt = torch.optim.Adam([M], lr=lr)
            for it in range(steps):
                opt.zero_grad()
                self.e_static(self.field(M), "G").backward()
                M.grad.mul_(self.fmask)
                opt.step()
            print(f"  {tag} polish adam lr {lr}: |g|_inf {self.grad_inf(M.detach()):.5f} [{time.time()-t0:.0f}s]", flush=True)
        opt = torch.optim.LBFGS([M], max_iter=40, history_size=25, tolerance_grad=1e-9, tolerance_change=1e-14)
        gi = None
        for outer in range(12):
            def closure():
                opt.zero_grad()
                E = self.e_static(self.field(M), "G")
                E.backward()
                M.grad.mul_(self.fmask)
                return E
            opt.step(closure)
            gi = self.grad_inf(M.detach())
            print(f"  {tag} polish lbfgs outer {outer}: |g|_inf {gi:.6f} [{time.time()-t0:.0f}s]", flush=True)
            if gi < 1e-4:
                break
        return M.detach(), gi

    # ---------- the 008 rung relaxation ----------
    def relax_rung(self, e_total_fn, M_start, cycles=4):
        M_raw = M_start.clone().requires_grad_(True)
        opt = torch.optim.Adam([M_raw], lr=1e-3)
        for it in range(500):
            opt.zero_grad()
            e_total_fn(M_raw).backward()
            opt.step()
        E_levels = [float(e_total_fn(M_raw).detach())]
        for cycle in range(cycles):
            opt2 = torch.optim.LBFGS([M_raw], max_iter=200, history_size=25, tolerance_grad=1e-9,
                                     tolerance_change=0, line_search_fn="strong_wolfe")

            def closure():
                opt2.zero_grad()
                E = e_total_fn(M_raw)
                E.backward()
                return E
            opt2.step(closure)
            E_levels.append(float(e_total_fn(M_raw).detach()))
        g = torch.autograd.grad(e_total_fn(M_raw), M_raw)[0]
        return M_raw.detach(), E_levels, float(g.abs().max())
