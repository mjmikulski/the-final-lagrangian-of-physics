"""The rot/boost channels are the magnetic/electric norms of the
matrix-pair 2-form w.r.t. u -- verified identities:

  boost2 = 2 |E|^2,  E_{mn|b} = u^a F_{mn a b}          (no epsilon)
  rot2   = 2 |B|^2,  B^{mn|g} = 1/2 eps^{g a b d} u_d F_{mn a b}

The dual part carries a Levi-Civita, but squared: two epsilons collapse
to metrics (delta identity), so both channels are parity-EVEN and none
of the four genuine one-epsilon pseudoscalars (report 001 scope limit,
catalogued by P239) is used -- the E.B-type invariant is printed as the
unused parity-odd witness.
"""
import os
from itertools import permutations

import torch

torch.manual_seed(3)
HERE = os.path.dirname(os.path.abspath(__file__))
ETA = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
DELTA, SG = 0.3, 8.0
M_VAC = torch.diag(torch.tensor([-SG, 1.0, DELTA, 0.0], dtype=torch.float64))


def G_lagrange(M):
    x = ETA @ M
    I4 = torch.eye(4, dtype=M.dtype)
    q = (x @ (x - I4) @ (x - DELTA * I4)) / (SG * (SG - 1) * (SG - DELTA))
    return ETA - 2.0 * q @ ETA


def F_of(A):
    AeA = torch.einsum("mac,cd,ndb->mnab", A, ETA, A)
    return AeA - AeA.transpose(0, 1)


def channels(F, M):
    G = G_lagrange(M)
    lGG = torch.einsum("mnab,mp,nq,ac,bd,pqcd->", F, G, G, G, G, F)
    lGe = torch.einsum("mnab,mp,nq,ac,bd,pqcd->", F, G, G, ETA, ETA, F)
    return (lGG + lGe) / 2, (lGG - lGe) / 2


def perm_sign(p):
    return (-1) ** sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4))


eps = torch.zeros(4, 4, 4, 4, dtype=torch.float64)
for p in permutations(range(4)):
    eps[p] = perm_sign(p)                        # eps_{0123} = +1
eps_up = torch.einsum("abcd,ae,bf,cg,dh->efgh", eps, ETA, ETA, ETA, ETA)

A = torch.randn(4, 4, 4, dtype=torch.float64)
A = A + A.transpose(-1, -2)
F = F_of(A)
u = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)    # vacuum frame
Gm = G_lagrange(M_VAC)

E = torch.einsum("a,mnab->mnb", u, F)            # (4,4,4) electric part
B = 0.5 * torch.einsum("gabd,d,mnab->mng", eps_up, ETA @ u, F)   # dual part
E2 = torch.einsum("mnb,mp,nq,bd,pqd->", E, Gm, Gm, Gm, E)
B2 = torch.einsum("mng,mp,nq,gh,pqh->", B, Gm, Gm, ETA, B)
r2, b2 = channels(F, M_VAC)

rel_b = ((b2 - 2 * E2).abs() / b2.abs()).item()
rel_r = ((r2 - 2 * B2).abs() / r2.abs()).item()
eb = torch.einsum("mnb,mnb->", E, B).item()      # one-epsilon, parity-odd
print(f"boost2 vs 2|E|^2: rel {rel_b:.1e};  rot2 vs 2|B|^2: rel {rel_r:.1e}")
print(f"unused parity-odd E.B witness: {eb:+.4f} (nonzero, not in L_C)")
assert rel_b < 1e-12 and rel_r < 1e-12 and abs(eb) > 1e-3
print("VERIFIED: channel/dual identities hold; no pseudoscalar is used.")
