"""Lagrangian of the symmetric tensor field M (Task 1, X = 0).

Conventions: signature (+,-,-,-). M is stored as the covariant symmetric matrix M_{mu nu};
the operator acting on vectors is N = eta M (mixed tensor). Everything is batched over
leading dimensions; the last two dimensions are matrix indices, a derivative index
(when present) sits just before them: dM[..., mu, a, b] = d_mu M_ab.
"""
import torch

ETA = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))
E_DEFAULT = (100.0, 1.0, 0.01, 0.0)
PAIRS = ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])
PAIR_SIGN = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)


def eta(like):
    return ETA.to(like)


def operator(M):
    return eta(M) @ M


def eigenvalues(N, c):
    """Descending eigenvalues of the eta-self-adjoint N.

    g = eta (N - c) is symmetric and positive definite when exactly one eigenvalue exceeds c,
    and N is g-self-adjoint, so S = L^T N L^-T (g = L L^T) is symmetric with the spectrum of N.
    Only eigenvalues are used downstream, which keeps autograd finite at degeneracies.
    """
    g = eta(N) @ N - c * eta(N)
    L = torch.linalg.cholesky(g)
    S = torch.linalg.solve_triangular(L.mT, L.mT @ N, upper=True, left=False)
    return eigvalsh4(0.5 * (S + S.mT))


def time_projector(N, e):
    """P_0 = prod_{b>0} (N - e_b) / (e_0 - e_b): the projector on the time-like eigenvector."""
    I = torch.eye(4, dtype=N.dtype, device=N.device)
    P = I
    for b in (1, 2, 3):
        P = P @ (N - e[..., b, None, None] * I) / (e[..., 0] - e[..., b])[..., None, None]
    return P


def delta_M(N, e):
    """delta_M = 2 v_0 v_0^T - eta = (2 P_0 - 1) eta, the field-defined Euclidean metric."""
    I = torch.eye(4, dtype=N.dtype, device=N.device)
    return (2 * time_projector(N, e) - I) @ eta(N)


def field_strength(dN):
    """F[..., mu, nu, a, b] = [d_mu N, d_nu N]_ab."""
    NN = torch.einsum('...mab,...nbc->...mnac', dN, dN)
    return NN - NN.transpose(-4, -3)


def sq_norm(F, dM):
    """||F||^2 = Tr(F dM F^T dM^-1), the Euclidean norm in the eigenframe; dM^-1 = eta dM eta."""
    dMi = eta(dM) @ dM @ eta(dM)
    return torch.einsum('...ab,...bc,...dc,...da->...', F, dM, F, dMi)


def kinetic(F, dM, deriv_metric='eta'):
    """- G^{mu mu'} G^{nu nu'} <F_{mu nu}, F_{mu' nu'}> with G = eta (default) or delta_M."""
    if deriv_metric == 'eta':
        s = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=F.dtype, device=F.device)
        return -torch.einsum('...mn,m,n->...', sq_norm(F, dM[..., None, None, :, :]), s, s)
    dMi = eta(dM) @ dM @ eta(dM)
    inner = torch.einsum('...mnab,...bc,...pqdc,...da->...mnpq', F, dM, F, dMi)
    return -torch.einsum('...mnpq,...mp,...nq->...', inner, dM, dM)


def potential(e, E):
    return ((e - torch.as_tensor(E, dtype=e.dtype, device=e.device)) ** 2).sum(-1)


def eigvalsh3(A):
    """Eigenvalues (descending) of batched symmetric 3x3 matrices, closed trigonometric form.
    The clamp keeps the gradient finite at degeneracies, where it is exact for the isotropic part."""
    q = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1) / 3
    B = A - q[..., None, None] * torch.eye(3, dtype=A.dtype, device=A.device)
    p = torch.sqrt((B * B).sum((-2, -1)) / 6 + 1e-300)
    r = (torch.linalg.det(B / p[..., None, None]) / 2).clamp(-1 + 1e-12, 1 - 1e-12)
    phi = torch.acos(r) / 3
    e1 = q + 2 * p * torch.cos(phi)
    e3 = q + 2 * p * torch.cos(phi + 2 * torch.pi / 3)
    return torch.stack([e1, 3 * q - e1 - e3, e3], -1)


def terms(M, dM_cov, E=E_DEFAULT, deriv_metric='eta', frozen=False, X=None):
    """(-F.F, V, X) pointwise. M[..., 4, 4] covariant, dM_cov[..., mu, 4, 4] = d_mu M_{ab}.
    frozen: the time-like sector is at its vacuum value (M_00 = E_0, M_0i = 0), so the spectrum is
    {E_0} + spec(-M_ij) and delta_M is the identity."""
    N, dN = operator(M), operator(dM_cov)
    if frozen:
        e = torch.cat([N[..., 0, 0, None], eigvalsh3(N[..., 1:, 1:])], -1)
        A, B = dN[..., PAIRS[0], :, :], dN[..., PAIRS[1], :, :]
        F = A @ B - B @ A
        kin = -2 * torch.einsum('...pab,p->...', F * F, PAIR_SIGN.to(F))
        return kin, potential(e, E), x_term(N, e, F, X) + extra_terms(N, dN, X)
    e = eigenvalues(N, c=(E[0] * E[1]) ** 0.5)
    dM = delta_M(N, e)
    x = torch.zeros_like(e[..., 0])
    if isinstance(X, dict) and X.get('tilt'):
        x = x + X['tilt'] * time_tilt(N, dN, e, dM)
    return kinetic(field_strength(dN), dM, deriv_metric), potential(e, E), x + extra_terms(N, dN, X)


def extra_terms(N, dN, X):
    """Terms quadratic in the derivatives that are not linear in F (outside the Task 4 class), in the
    frozen sector: 'sigma' * eta^{mu nu} Tr(d_mu N d_nu N) (sigma model) and
    'frame' * eta^{mu nu} Tr(A_mu A_nu) with A_mu = 1/2 [N, d_mu N] (the composite potential; in the
    eigenframe it weights the frame rotations by the squared eigenvalue differences)."""
    if not isinstance(X, dict):
        return 0
    s = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=N.dtype, device=N.device)
    x = 0
    if X.get('sigma'):
        x = x + X['sigma'] * torch.einsum('...m,m->...', (dN * dN).sum((-2, -1)), s)
    if X.get('frame'):
        A = 0.5 * (N[..., None, :, :] @ dN - dN @ N[..., None, :, :])
        x = x + X['frame'] * torch.einsum('...m,m->...', (A * A).sum((-2, -1)), s)
    return x


def time_tilt(N, dN, e, dMet):
    """K_u = eta^{mu nu} <A_mu, A_nu>, A_mu = (1 - P_0) d_mu N P_0: the part of the jet that rotates the
    field's own time-like axis. It vanishes identically wherever that axis is constant, so it is zero on
    every static texture of the frozen sector and on every rigid rotation of one."""
    P0 = time_projector(N, e)
    I = torch.eye(4, dtype=N.dtype, device=N.device)
    A = (I - P0)[..., None, :, :] @ dN @ P0[..., None, :, :]
    s = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=N.dtype, device=N.device)
    return torch.einsum('...m,m->...', sq_norm(A, dMet[..., None, :, :]), s)


def x_term(N, e, F, X):
    """X = c_a X_a + c_b X_b in the frozen sector (Task 4): X_a = F_{ij,ab} P_1^{ia} Q^{jb}, X_b = F_{ij,ab} Q^{ia} Q^{jb}
    with P_1 the projector on the charge direction and Q = 1 - P_1 (summed over all i, j; F antisymmetric).
    With constant coefficients and the degenerate vacuum pair these are the two structures that survive for
    static fields with the time-like sector at its vacuum; their constant-coefficient sum is a total derivative.
    Contractions with eigenvalue-dependent weights (powers of M) are independent of them and not scanned here."""
    if isinstance(X, dict):
        X = (X.get('xa', 0.0), X.get('xb', 0.0))
    if X is None or not (X[0] or X[1]):
        return torch.zeros_like(e[..., 0])
    n, es = N[..., 1:, 1:], e[..., 1:]
    I3 = torch.eye(3, dtype=N.dtype, device=N.device)
    P1 = (n - es[..., 1, None, None] * I3) @ (n - es[..., 2, None, None] * I3) / ((es[..., 0] - es[..., 1]) * (es[..., 0] - es[..., 2]))[..., None, None]
    Q = I3 - P1
    xa = xb = 0
    for p, (i, j) in zip((3, 4, 5), ((0, 1), (0, 2), (1, 2))):
        Fp = F[..., p, 1:, 1:]
        xa = xa + torch.einsum('...ab,...a,...b->...', Fp, P1[..., i, :], Q[..., j, :]) - torch.einsum('...ab,...a,...b->...', Fp, P1[..., j, :], Q[..., i, :])
        xb = xb + torch.einsum('...ab,...a,...b->...', Fp, Q[..., i, :], Q[..., j, :]) - torch.einsum('...ab,...a,...b->...', Fp, Q[..., j, :], Q[..., i, :])
    return X[0] * xa + X[1] * xb


def lagrangian(M, dM_cov, E=E_DEFAULT, deriv_metric='eta', frozen=False, X=None):
    kin, pot, x = terms(M, dM_cov, E, deriv_metric, frozen, X)
    return kin + x - pot


def static_energy_density(M, dM_cov, E=E_DEFAULT, deriv_metric='eta', frozen=False, X=None):
    """Energy density for time-independent fields (dM_cov[..., 0] must be zero): H = -L."""
    return -lagrangian(M, dM_cov, E, deriv_metric, frozen, X)


def eigvalsh4(S, sweeps=7):
    """Eigenvalues (descending) of batched symmetric 4x4 matrices by cyclic Jacobi rotations.

    cuSOLVER's batched eigensolver needs ~0.5 MB per matrix, so it cannot run on a lattice;
    seven sweeps bring the off-diagonal part of a 4x4 matrix to round-off.
    """
    tiny = 1e-300
    for _ in range(sweeps):
        for p, q in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            d = S[..., q, q] - S[..., p, p]
            t = 2 * S[..., p, q] * torch.where(d < 0, -1.0, 1.0) / (d.abs() + torch.sqrt(d * d + 4 * S[..., p, q] ** 2 + tiny))
            c = torch.rsqrt(1 + t * t)
            s = t * c
            J = torch.eye(4, dtype=S.dtype, device=S.device).expand(S.shape).clone()
            J[..., p, p], J[..., q, q], J[..., p, q], J[..., q, p] = c, c, s, -s
            S = J.mT @ S @ J
    return torch.diagonal(S, dim1=-2, dim2=-1).sort(-1, descending=True).values
