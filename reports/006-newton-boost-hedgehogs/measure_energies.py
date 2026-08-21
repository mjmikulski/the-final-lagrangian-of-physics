"""Measurements on the canonical boost-hedgehog ansatz (float64, GPU if
available; falls back to CPU).

1. Virial check: single-hedgehog S1/S4 = 4/3 for every profile, two
   independent integration routes (2D cylindrical grid, 1D radial
   quadrature).
2. Tailed profiles f = exp(-mu r)/r^p: both interaction channels are
   REPULSIVE in the Newton window (E1_int, E4_int > 0), and the per-
   separation ratio t1(d) = E1_int/E4_int exceeds 4/3.
3. Cluster witness: multi-center configurations with S1/S4 well above the
   measured tail ratios (kills the alpha<0 branch of the no-go).
4. The author's notebook protocol (p=1/2, unscreened): the self-energy
   diverges linearly in the domain radius (S ~ 85*R) -- the notebook's
   fitted numbers are an artifact of an IR-divergent integral; the
   qualitative sign conclusion is unaffected.
5. Curiosity: compact (gaussian) dressings show a short-range attraction
   pocket in both channels near d ~ core size -- molecular-type binding,
   not a long-range tail.
6. Convergence scans (grid step, domain, core cutoff).

Out: results/energy_results.json
"""
import json
import os

import numpy as np
import torch
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
DT = torch.float64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
results = {}


def V_field(pts, centers, fdf):
    V = torch.zeros(pts.shape[0], 3, 3, dtype=DT, device=DEV)
    eye = torch.eye(3, dtype=DT, device=DEV)
    for c in centers:
        rv = pts - torch.tensor(c, dtype=DT, device=DEV)
        r = rv.norm(dim=1).clamp_min(1e-300)
        f, df = fdf(r)
        V += (df / r)[:, None, None] * rv[:, :, None] * rv[:, None, :] \
            + f[:, None, None] * eye
    return V


def dens_pair(V):
    G = V @ V
    trG = G.diagonal(dim1=1, dim2=2).sum(1)
    e1 = 2 * (trG ** 2 - (G * G.transpose(1, 2)).sum((1, 2)))
    trV = V.diagonal(dim1=1, dim2=2).sum(1)
    Phi = G - trV[:, None, None] * V
    e4 = (Phi * Phi.transpose(1, 2)).sum((1, 2))
    return e1, e4


def integrate_cyl(zcenters, fdf, R, h=0.02, eps=0.05, chunk=400):
    """Axisymmetric configs: centers on the z axis."""
    nr, nz = int(R / h), int(2 * R / h)
    rho1d = (torch.arange(nr, dtype=DT, device=DEV) + 0.5) * h
    z1d = -R + (torch.arange(nz, dtype=DT, device=DEV) + 0.5) * h
    tot = torch.zeros(2, dtype=DT, device=DEV)
    for i0 in range(0, nz, chunk):
        z = z1d[i0:i0 + chunk]
        Z, RHO = torch.meshgrid(z, rho1d, indexing="ij")
        rho_f, z_f = RHO.flatten(), Z.flatten()
        mask = torch.ones_like(rho_f, dtype=torch.bool)
        for zc in zcenters:
            mask &= (rho_f ** 2 + (z_f - zc) ** 2) > eps ** 2
        pts = torch.stack([rho_f[mask], torch.zeros_like(rho_f[mask]),
                           z_f[mask]], 1)
        e1, e4 = dens_pair(V_field(pts, [[0, 0, zc] for zc in zcenters], fdf))
        w = 2 * torch.pi * rho_f[mask] * h * h
        tot += torch.stack([(e1 * w).sum(), (e4 * w).sum()])
    return tot.cpu()


def integrate_3d(centers, fdf, R=10.0, h=0.04):
    n = int(2 * R / h)
    ax = -R + (torch.arange(n, dtype=DT, device=DEV) + 0.5) * h
    tot = torch.zeros(2, dtype=DT, device=DEV)
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    for iz in range(n):
        pts = torch.stack([X.flatten(), Y.flatten(),
                           ax[iz].expand(n * n)], 1)
        e1, e4 = dens_pair(V_field(pts, centers, fdf))
        tot += torch.stack([e1.sum(), e4.sum()]) * h ** 3
    return tot.cpu()


def power_fdf(p, mu):
    def fdf(r):
        f = torch.exp(-mu * r) * r ** (-p)
        return f, f * (-p / r - mu)
    return fdf


def radial_self(p, mu, eps, Rmax=400.0):
    def dens(r, which):
        f = np.exp(-mu * r) * r ** (-p)
        lr, lt = f * (1 - p - mu * r), f
        if which == 1:
            return 4 * np.pi * r * r * 4 * lt * lt * (2 * lr * lr + lt * lt)
        return 4 * np.pi * r * r * (4 * lr * lr * lt * lt
                                    + 2 * lt * lt * (lr + lt) ** 2)
    S1 = quad(lambda r: dens(r, 1), eps, Rmax, limit=500)[0]
    S4 = quad(lambda r: dens(r, 4), eps, Rmax, limit=500)[0]
    return S1, S4


# --- 1. virial: S1/S4 = 4/3 -------------------------------------------------
print("1. virial S1/S4 = 4/3 on single hedgehogs (quad route, eps -> 0):")
vir = {}
for (p, mu) in [(0.3, 0.2), (0.5, 0.1), (0.5, 0.2), (0.6, 0.3)]:
    S1a, S4a = radial_self(p, mu, eps=1e-8)
    vir[f"p{p}_mu{mu}"] = S1a / S4a
    print(f"   p={p} mu={mu}: S1/S4 = {S1a / S4a:.8f}")
    assert abs(S1a / S4a - 4 / 3) < 1e-5
gf = lambda r: (torch.exp(-r ** 2), -2 * r * torch.exp(-r ** 2))
Sg = integrate_cyl([0.0], gf, R=12.0, h=0.01, eps=0.0)
vir["gauss_grid"] = (Sg[0] / Sg[1]).item()
print(f"   gaussian, 2D grid, no cutoff: S1/S4 = {Sg[0] / Sg[1]:.8f}")
assert abs(Sg[0] / Sg[1] - 4 / 3) < 1e-4
results["virial"] = vir

# --- 2. tailed profiles: repulsive tails, t1 > 4/3 -------------------------
# Self-energies from the SAME grid scheme as E(d) (consistent core cutoff);
# the radial-quadrature route cross-checks S for the UV-convergent profiles
# (p < 3/4, where the eps-scheme dependence vanishes).
print("2. tailed profiles, Newton window (mu*d <= 0.6):")
CLEAN = [(0.3, 0.2), (0.5, 0.1), (0.5, 0.2), (0.75, 0.2)]
FLAGGED = [(1.0, 0.2)]          # UV-cutoff-dependent: excluded from assembly
tails = {}
for (p, mu) in CLEAN + FLAGGED:
    R = min(45.0, 14.0 / mu)
    eps = 0.05
    fdf = power_fdf(p, mu)
    Sgrid = integrate_cyl([0.0], fdf, R, h=0.02, eps=eps)
    Sg1, Sg4 = Sgrid[0].item(), Sgrid[1].item()
    if p <= 0.6:
        Sq1, Sq4 = radial_self(p, mu, eps=eps)
        rel = max(abs(Sq1 - Sg1) / Sg1, abs(Sq4 - Sg4) / Sg4)
        assert rel < 2e-2, f"S cross-route mismatch {rel}"
    rows = []
    for d in (1.0, 1.5, 2.0, 2.5, 3.0):
        if mu * d > 0.6:
            continue
        E = integrate_cyl([-d, d], fdf, R, h=0.02, eps=eps)
        i1, i4 = E[0].item() - 2 * Sg1, E[1].item() - 2 * Sg4
        rows.append({"d": d, "E1int": i1, "E4int": i4, "t1": i1 / i4,
                     "X": 3 * i1 - 4 * i4})
    assert all(r["X"] > 0 for r in rows)     # marginal direction repels
    ts = [r["t1"] for r in rows]
    pos = all(r["E1int"] > 0 and r["E4int"] > 0 for r in rows)
    flagged = (p, mu) in FLAGGED
    tails[f"p{p}_mu{mu}"] = {"rows": rows, "t1_min": min(ts),
                             "t1_max": max(ts), "all_repulsive": pos,
                             "uv_flagged": flagged}
    print(f"   p={p:4.2f} mu={mu}: E_int>0 both: {pos};  "
          f"t1(d) in [{min(ts):.4f}, {max(ts):.4f}]"
          f"{'   [UV-cutoff-flagged]' if flagged else ''}")
    assert pos and min(ts) > 4 / 3
results["tails"] = tails
t1_ceiling = max(v["t1_max"] for v in tails.values()
                 if not v["uv_flagged"])
results["t1_ceiling_clean"] = t1_ceiling

# --- 3. cluster witnesses ---------------------------------------------------
print("3. cluster witnesses (gaussian profile, 3D grid, no cutoff):")
gf3 = lambda r: (torch.exp(-r ** 2), -2 * r * torch.exp(-r ** 2))
clusters = {
    "trio": [[-1.3, 0, 0], [0.4, 0, 0.6], [1.9, 0, -0.4]],
    "five": [[0, 0, 0], [1.1, 0, 0], [-0.6, 0.9, 0],
             [0.2, -1.0, 0.6], [-0.4, -0.3, -1.0]],
    "chain7": [[i * 1.1 - 3.3, 0, 0] for i in range(7)],
    "ring6": [[1.2 * np.cos(k * np.pi / 3), 1.2 * np.sin(k * np.pi / 3), 0]
              for k in range(6)],
}
cl = {}
for tag, cen in clusters.items():
    S = integrate_3d(cen, gf3, R=10.0, h=0.04)
    cl[tag] = (S[0] / S[1]).item()
    print(f"   {tag}: S1/S4 = {S[0] / S[1]:.4f}")
results["clusters"] = cl
witness = max(cl.values())
results["cluster_witness"] = witness
assert witness > t1_ceiling, "cluster witness must exceed measured t1"

# --- 4. the notebook protocol is IR-divergent ------------------------------
print("4. unscreened p=1/2 self-energy vs domain radius:")
fdf0 = power_fdf(0.5, 0.0)
Rs, Ss = [10.0, 20.0, 30.0, 40.0], []
for R in Rs:
    S = integrate_cyl([0.0], fdf0, R, h=0.02, eps=0.0316)
    Ss.append(S[0].item())
    print(f"   R={R:5.1f}: S1 = {S[0]:9.2f}")
slope = np.polyfit(Rs, Ss, 1)[0]
print(f"   linear IR divergence, slope dS1/dR = {slope:.2f}")
assert slope > 10
results["ir_divergence_slope"] = slope

# --- 5. compact-dressing pocket --------------------------------------------
print("5. gaussian pair: short-range pocket (not a tail):")
Sg1, Sg4 = Sg[0].item(), Sg[1].item()
pocket = {}
for d in (0.5, 1.0, 1.5, 2.0):
    E = integrate_cyl([-d, d], gf, R=12.0, h=0.01, eps=0.0)
    pocket[d] = {"E1int": E[0].item() - 2 * Sg1,
                 "E4int": E[1].item() - 2 * Sg4}
    print(f"   d={d}: E1int = {pocket[d]['E1int']:+.4f}   "
          f"E4int = {pocket[d]['E4int']:+.4f}")
results["gauss_pocket"] = {str(k): v for k, v in pocket.items()}
assert pocket[1.0]["E1int"] < 0 < pocket[0.5]["E1int"]

# --- 6. convergence ---------------------------------------------------------
print("6. convergence at p=0.5 mu=0.2, d=1.5:")
fdf = power_fdf(0.5, 0.2)
base = integrate_cyl([-1.5, 1.5], fdf, R=45.0, h=0.02, eps=0.05)
conv = {}
for tag, kw in [("h=0.01", dict(h=0.01)), ("R=60", dict(R=60.0)),
                ("eps=0.025", dict(eps=0.025))]:
    v = integrate_cyl([-1.5, 1.5], fdf,
                      kw.get("R", 45.0), h=kw.get("h", 0.02),
                      eps=kw.get("eps", 0.05))
    conv[tag] = {"dE1_rel": ((v[0] - base[0]) / base[0]).item(),
                 "dE4_rel": ((v[1] - base[1]) / base[1]).item()}
    print(f"   {tag:9s}: dE1/E1 = {conv[tag]['dE1_rel']:+.2e}   "
          f"dE4/E4 = {conv[tag]['dE4_rel']:+.2e}")
results["convergence"] = conv

# 6b. sign stability of the marginal-direction excess X = 3 E1int - 4 E4int
print("6b. X = 3 E1int - 4 E4int sign stability (p=0.5 mu=0.1, d=2):")
fdfx = power_fdf(0.5, 0.1)
Xvals = {}
for tag, (hh, ee) in [("base", (0.02, 0.05)), ("h=0.01", (0.01, 0.05)),
                      ("eps=0.025", (0.02, 0.025))]:
    Sx = integrate_cyl([0.0], fdfx, 45.0, h=hh, eps=ee)
    Ex = integrate_cyl([-2.0, 2.0], fdfx, 45.0, h=hh, eps=ee)
    X = 3 * (Ex[0] - 2 * Sx[0]).item() - 4 * (Ex[1] - 2 * Sx[1]).item()
    Xvals[tag] = X
    print(f"   {tag:9s}: X = {X:+.3f}")
    assert X > 0
Xg = {d: 3 * v["E1int"] - 4 * v["E4int"] for d, v in pocket.items()}
print(f"   gaussian pair (cutoff-free): X(d) = "
      + ", ".join(f"{d}: {x:+.3f}" for d, x in Xg.items()))
assert all(x > 0 for x in Xg.values())
results["X_stability"] = {"variants": Xvals,
                          "gauss_cutoff_free": {str(k): v
                                                for k, v in Xg.items()}}

# --- verdicts ---------------------------------------------------------------
results["no_go"] = {
    "stability_singles_boundary_t": 4 / 3,
    "attraction_needs_t_above": min(v["t1_min"] for v in tails.values()),
    "alpha_neg_needs_beta_ratio_at_least": witness,
    "alpha_neg_attraction_allows_at_most": t1_ceiling,
}
print(f"\nno-go summary: attraction needs t > "
      f"{results['no_go']['attraction_needs_t_above']:.4f} > 4/3 = single-"
      f"hedgehog stability bound; alpha<0 branch: needs beta/|alpha| >= "
      f"{witness:.3f} (cluster) but attraction allows < {t1_ceiling:.3f}")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "energy_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("written: results/energy_results.json")
