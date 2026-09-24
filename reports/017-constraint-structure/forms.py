"""Kinetic form K and principal symbol of the model at a point, by two independent routes.

Route 1 (`forms_autograd`): second derivatives of the model's own Lagrangian (lagrangian.py, copied from
report 016) with respect to the time derivative dM/dt and to the spatial derivative along a unit vector k.

Route 2 (`forms_formula`): the closed form derived in the README. With W_i(X) = [eta X, d_i N] and the
field's Euclidean norm <A, B> = Tr(A dM B^T dM^-1) (dM = delta_M),
    A_ab = sum_i <W_i(X_a), W_i(X_b)>,   B(k)_ab = <k.W(X_a), k.W(X_b)>,   C_ab = <Y(X_a), Y(X_b)>,
    Y(X) = (1 - P_0) eta X P_0,
    K = 4 A + 2 c C,   G(k) = 4 (|k|^2 A - B(k)) + 2 c |k|^2 C,
where c is the coupling of the frame term K_u of report 016. Plain numpy, no autograd and no torch.

Both return (K, G) as 10 x 10 matrices in the basis of the symmetric M (order 00,01,02,03,11,12,13,22,23,33),
with G the gradient form of the energy (positive semidefinite), so that the characteristic equation of a plane
wave exp(i(k.x - w t)) at principal order is det(w^2 K - G(k)) = 0.
"""
import numpy as np
import torch
from lagrangian import lagrangian
from soliton import to_matrix

torch.set_default_dtype(torch.float64)
ETA = np.diag([1.0, -1.0, -1.0, -1.0])
PAIRS = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
NAMES = ['M%d%d' % p for p in PAIRS]


def basis():
    """Unit symmetric matrices, one per stored component (off-diagonal: both entries 1)."""
    out = []
    for a, b in PAIRS:
        X = np.zeros((4, 4))
        X[a, b] = X[b, a] = 1.0
        out.append(X)
    return np.array(out)


BASIS = basis()


# ---------------------------------------------------------------- route 1: autograd on the model code
def forms_autograd(M, dM, E, c=0.0, k=(0.0, 0.0, 1.0)):
    M, dM = torch.as_tensor(M), torch.as_tensor(dM)
    k = torch.as_tensor(k, dtype=torch.float64)
    X = {'tilt': c} if c else None

    def L_of(v0, vk):
        d = dM.clone()
        d[0] = d[0] + to_matrix(v0)
        for i in range(3):
            d[1 + i] = d[1 + i] + k[i] * to_matrix(vk)
        return lagrangian(M[None], d[None], E, frozen=False, X=X)[0]

    z = torch.zeros(10)
    K = torch.autograd.functional.hessian(lambda v: L_of(v, z), z.clone())
    G = -torch.autograd.functional.hessian(lambda v: L_of(z, v), z.clone())
    return K.numpy(), G.numpy()


# ---------------------------------------------------------------- route 2: closed form, numpy
def frame(M):
    """delta_M, its inverse and the projector P_0 on the time-like eigenvector of N = eta M (numpy eig)."""
    N = ETA @ M
    w, V = np.linalg.eig(N)
    w, V = w.real, V.real
    # the time-like eigenvector is the one with v^T eta v > 0 (exactly one for the spectra used here)
    norms = np.einsum('ai,ab,bi->i', V, ETA, V)
    i0 = int(np.argmax(norms))
    v0 = V[:, i0] / np.sqrt(norms[i0])
    P0 = np.outer(v0, v0) @ ETA
    dMet = (2 * P0 - np.eye(4)) @ ETA
    return dMet, ETA @ dMet @ ETA, P0


def inner(Aa, Ab, dMet, dMi):
    """<A, B> = Tr(A dM B^T dM^-1), batched over leading indices of both arguments -> Gram matrix."""
    return np.einsum('...ab,bc,...dc,da->...', Aa, dMet, Ab, dMi) if Aa.ndim == 2 else \
        np.einsum('iab,bc,jdc,da->ij', Aa, dMet, Ab, dMi)


def gram(Ws, dMet, dMi):
    return np.einsum('iab,bc,jdc,da->ij', Ws, dMet, Ws, dMi)


def pieces(M, dM, metric='delta'):
    """A, the three W_i blocks and C at a point. metric='eta' replaces delta_M by eta in the matrix norm
    (the indefinite contraction <F, G>_eta of the thread), for comparison only."""
    dMet, dMi, P0 = frame(M)
    if metric == 'eta':
        dMet = dMi = ETA
    NX = np.einsum('ab,xbc->xac', ETA, BASIS)                       # eta X_a
    D = [ETA @ dM[1 + i] for i in range(3)]                           # d_i N
    W = np.array([NX @ Di - Di @ NX for Di in D])                     # W[i, a] = [eta X_a, d_i N]
    A = sum(gram(W[i], dMet, dMi) for i in range(3))
    Y = (np.eye(4) - P0) @ NX @ P0
    C = gram(Y, dMet, dMi)
    return A, W, C, dMet, dMi


def forms_formula(M, dM, c=0.0, k=(0.0, 0.0, 1.0), metric='delta'):
    A, W, C, dMet, dMi = pieces(np.asarray(M), np.asarray(dM), metric)
    k = np.asarray(k, float)
    kW = np.einsum('i,iaxy->axy', k, W)
    B = gram(kW, dMet, dMi)
    K = 4 * A + 2 * c * C
    G = 4 * (k @ k * A - B) + 2 * c * (k @ k) * C
    return K, G


# ---------------------------------------------------------------- shared analysis
def rank_kernel(K, rtol=1e-9):
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    scale = max(abs(w).max(), 1e-300)
    live = w > rtol * scale
    return int(live.sum()), w, V[:, ~live]


def speeds2(K, G, rtol=1e-9):
    """Squared characteristic speeds |w|^2/|k|^2 on the range of K (k is a unit vector), and the size of G on
    the kernel of K relative to G (zero means the kernel directions have no principal symbol at all)."""
    K, G = 0.5 * (K + K.T), 0.5 * (G + G.T)
    w, V = np.linalg.eigh(K)
    scale = max(abs(w).max(), 1e-300)
    live = w > rtol * scale
    R, Z = V[:, live], V[:, ~live]
    Kl = R.T @ K @ R
    Li = np.linalg.inv(np.linalg.cholesky(Kl))
    s = np.linalg.eigvalsh(Li @ (R.T @ G @ R) @ Li.T)
    gker = float(np.abs(Z.T @ G @ Z).max() / max(np.abs(G).max(), 1e-300)) if Z.shape[1] else 0.0
    gmix = float(np.abs(Z.T @ G @ R).max() / max(np.abs(G).max(), 1e-300)) if Z.shape[1] else 0.0
    return np.sort(s), max(gker, gmix)


def trace_direction():
    """The symmetric M = eta: N -> N + 1, the shift that the whole derivative sector cannot see."""
    v = np.zeros(10)
    v[0], v[4], v[7], v[9] = 1.0, -1.0, -1.0, -1.0
    return v / 2.0


def p0_direction(M):
    """delta M along the time-like projector: N -> N + P_0 (eta P_0 as the covariant matrix, symmetrised)."""
    _, _, P0 = frame(M)
    X = ETA @ P0
    X = 0.5 * (X + X.T)
    return np.array([X[a, b] for a, b in PAIRS])
