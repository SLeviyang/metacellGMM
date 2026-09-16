from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import mixandmix as mm
import matplotlib.pyplot as plt


def create_ncells_per_mixture(n, alphas):
    n_a = np.round(n * np.asarray(alphas)).astype(int)
    diff = n - np.sum(n_a)
    if diff != 0:
        n_a[np.argmax(n_a)] += diff
    return n_a


########################################################
# Data structure conventions
#
# m_model : dict — mixture model parameters
#   "covariance"          : list of K arrays, each of shape (p,) — per-gene variance
#                           vector for each cell type (diagonal of covariance matrix)
#   "mixture_frequencies" : array of shape (K,) — mixture weights (alpha_a = n_a / n),
#                           must sum to 1
#   "gamma"               : float — aspect ratio p / n
#   "mixture_labels"      : array of shape (K,) — cell type label for each mixture
#                           component (e.g. strings from s.obs["cell_type"])
#
# spike : dict — low-rank signal structure (built by make_spike_matrix in lognorm.py)
#   "u_list"        : list of k arrays, each of shape (n,) — left singular vectors
#   "v_list"        : list of k arrays, each of shape (p,) — right singular vectors
#   "s"             : array of shape (k,) — singular values
#   "sample_labels" : array of shape (n,) — cell type label for each cell,
#                     same dtype as m_model["mixture_labels"]
#
########################################################
# functions that compute quantities associated with the bulk
# Propositions 1.5 and 1.6 from Benaych-Georges and Nadakuditi (2012?)
#  and Benaych-Georges, Coulliet 2016

# equations (1.6) and (1.7) in Proposition 1.5 (B-G 2016)
def make_Q_bar(m_model, g):
    """
    Returns the two deterministic equivalents from Proposition 1.5 (B-G 2016).

    Parameters
    ----------
    m_model : dict with keys "covariance", "mixture_frequencies", "gamma"
    g       : array of K complex values g_a(z)
    """
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    p = Lambda_list[0].shape[0]
    K = len(Lambda_list)
    alphas = np.asarray(alphas)

    n = int(round(p / gamma))
    n_a = create_ncells_per_mixture(n, alphas)

    # Eq (1.6): diagonal of Q_bar
    Q_bar = np.concatenate([gamma * g[a] * np.ones(n_a[a]) for a in range(K)])
    return Q_bar


def make_tilde_Q_bar(m_model, g, z):
    """
    Returns the deterministic equivalent from Proposition 1.5 (B-G 2016).

    Parameters
    ----------
    m_model : dict with keys "covariance", "mixture_frequencies", "gamma"
    g       : array of K complex values g_a(z)
    z       : scalar (real or complex)

    Returns
    -------
    Q_tilde_bar : (p,) array — diagonal of eq (1.7): -1/z * (I + sum_a c_a g_a C_a)^{-1}
    """
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    p = Lambda_list[0].shape[0]
    K = len(Lambda_list)

    # Eq (1.7): diagonal of Q_tilde_bar (all C_a diagonal => result is diagonal)
    diag_inner = np.ones(p, dtype=np.complex128)
    for a in range(K):
        diag_inner += alphas[a] * g[a] * Lambda_list[a]
    Q_tilde_bar = -1.0 / z / diag_inner

    return Q_tilde_bar


def make_R(m_model, g, z, Q_tilde_bar=None):
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    K = len(Lambda_list)
    p = Lambda_list[0].shape[0]
    alphas = np.asarray(alphas)
    g = np.asarray(g)

    if Q_tilde_bar is None:
        Q_tilde_bar = make_tilde_Q_bar(m_model, g, z)

    # Omega_{ab} = c0 * c_b * z^2 * g_a * g_b * (1/p) * tr[C_a Q_tilde C_b Q_tilde]
    # Since all diagonal: tr[C_a Q_tilde C_b Q_tilde] = Lambda_a . (Q_tilde^2 * Lambda_b)
    Qt2 = Q_tilde_bar ** 2
    Omega = np.zeros((K, K), dtype=np.complex128)
    for a in range(K):
        for b in range(K):
            trace_term = np.dot(Lambda_list[a] * Qt2, Lambda_list[b]) / p
            Omega[a, b] = gamma * alphas[b] * z**2 * g[a] * g[a] * trace_term

    # T_a = row_sum of M solves T_a = sum_b Omega[a,b] * (1 + T_b),
    # i.e. T = (I - Omega)^{-1} Omega * ones.
    IminusOmega_inv = np.linalg.inv(np.eye(K) - Omega)
    R = IminusOmega_inv @ Omega
    for i in range(K):
        for j in range(K):
            R[i, j] = alphas[i]/alphas[j] * R[i, j]

    return R


# first equation in proposition 1.6
def compute_Q2(m_model, g, z, u1, u2, restrict_to_a=None, isotropic=False, Q_bar=None, R=None):
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    if Q_bar is None:
        Q_bar = make_Q_bar(m_model, g)
    if R is None:
        R = make_R(m_model, g, z)
    K = len(alphas)
    n = len(Q_bar)
    n_a = create_ncells_per_mixture(n, alphas)

    starts = np.cumsum(np.concatenate([[0], n_a]))

    # s_b = u^T Qbar D_b Qbar u
    s = np.zeros(K, dtype=np.complex128)

    for b in range(K):
        start = starts[b]
        end = starts[b + 1]
        if not isotropic:
            s[b] = np.sum((Q_bar[start:end] ** 2) * u1[start:end] * u2[start:end])
        else:
            inner = np.dot(u1, u2)
            s[b] = (inner / n) * np.sum(Q_bar[start:end] ** 2)

    # After summing over a:
    # sum_a QD_aQ ~ sum_b (1 + sum_a R[a,b]) Qbar D_b Qbar
    coeffs = 1.0 + np.sum(R, axis=0)

    q2_by_block = s + R @ s
    if restrict_to_a is None:
        return np.sum(q2_by_block)
    else:
        return q2_by_block[restrict_to_a]


def compute_tQWWtQ(m_model, g, z, w1, w2, restrict_to_a=None, isotropic=False, Q_tilde_bar=None):
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    if Q_tilde_bar is None:
        Q_tilde_bar = make_tilde_Q_bar(m_model, g, z)
    K = len(Lambda_list)
    Qt2 = Q_tilde_bar ** 2
    w12 = w1*w2
    p = Lambda_list[0].shape[0]

    n = len(w1)
    n_a = create_ncells_per_mixture(n, alphas)
    starts = np.cumsum(np.concatenate([[0], n_a]))

    result = np.zeros(K, dtype=np.complex128)
    for a in range(K):
        if not isotropic:
            result[a] = z**2 * gamma * alphas[a] * g[a]**2 * np.dot(Qt2 * Lambda_list[a], w12)
        else:
            inner = np.dot(w1, w2)
            result[a] = z**2 * gamma * alphas[a] * g[a]**2 * (inner / p) * np.sum(Qt2 * Lambda_list[a])

    if restrict_to_a is None:
        return np.sum(result)
    else:
        return result[restrict_to_a]


def compute_g(m_model, z, outside_spectrum=False):
    params = mm.SolveParams(
        eta0=1.0,
        beta=2.0,
        eta_target=1e-8,
        max_iter=5000,
        tol=1e-8,
        resid_tol=1e-8)

    _, _, _, g, _ = mm.mixandmix_density(x_grid=np.array([z]), m_model=m_model,
        params=params, outside_spectrum=outside_spectrum)
    g = g[0]

    return g


# u_list is a list of vectors, w_list is a list of vectors, rho is a vector of spike strengths
# z is the value at which Q_bar, Q_tilde_bar are computed
def compute_spike_Q_moments(m_model, g, z, u_list, w_list, isotropic=False, U=None, W=None):
    #Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    Q_bar = make_Q_bar(m_model, g)
    Q_tilde_bar = make_tilde_Q_bar(m_model, g, z)

    k = len(u_list)
    if isotropic:
        u_moments = np.zeros((k, k), dtype=np.complex128)
        w_moments = np.zeros((k, k), dtype=np.complex128)
        np.fill_diagonal(u_moments, np.mean(Q_bar))
        np.fill_diagonal(w_moments, np.mean(Q_tilde_bar))
        return {'u_moments': u_moments, 'w_moments': w_moments}

    if U is None:
        U = np.column_stack(u_list)
    if W is None:
        W = np.column_stack(w_list)

    u_moments = U.T @ (Q_bar[:, None] * U)
    w_moments = W.T @ (Q_tilde_bar[:, None] * W)
    return {'u_moments': u_moments, 'w_moments': w_moments}


def compute_spike_Q2_moments(m_model, g, z, u_list, w_list, isotropic=False):
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    k = len(u_list)
    n = len(u_list[0])
    p = len(w_list[0])

    if isotropic:
        u_moments = np.zeros((k, k), dtype=np.complex128)
        w_moments = np.zeros((k, k), dtype=np.complex128)
        u_iso = 1 / np.sqrt(n) * np.ones(n)
        w_iso = 1 / np.sqrt(p) * np.ones(p)
        Q_bar = make_Q_bar(m_model, g)
        Q_tilde_bar = make_tilde_Q_bar(m_model, g, z)
        R = make_R(m_model, g, z, Q_tilde_bar=Q_tilde_bar)
        u_out = compute_Q2(m_model, g, z, u_iso, u_iso, isotropic=True, Q_bar=Q_bar, R=R)
        w_out = compute_tQWWtQ(m_model, g, z, w_iso, w_iso, isotropic=True, Q_tilde_bar=Q_tilde_bar)
        np.fill_diagonal(u_moments, u_out)
        np.fill_diagonal(w_moments, w_out)
        return {'u_moments': u_moments, 'w_moments': w_moments}

    K = len(alphas)
    n_a = create_ncells_per_mixture(n, alphas)
    starts = np.cumsum(np.concatenate([[0], n_a]))

    Q_bar = make_Q_bar(m_model, g)
    Q_tilde_bar = make_tilde_Q_bar(m_model, g, z)
    R = make_R(m_model, g, z, Q_tilde_bar=Q_tilde_bar)
    U = np.column_stack(u_list)
    W = np.column_stack(w_list)

    u_by_block = np.zeros((K, k, k), dtype=np.complex128)
    for b in range(K):
        start, end = starts[b], starts[b + 1]
        weighted_U = (Q_bar[start:end] ** 2)[:, None] * U[start:end, :]
        u_by_block[b] = U[start:end, :].T @ weighted_U

    q2_by_block = u_by_block + np.einsum('ab,bij->aij', R, u_by_block)
    u_moments = np.sum(q2_by_block, axis=0)

    Qt2 = Q_tilde_bar ** 2
    w_by_block = np.zeros((K, k, k), dtype=np.complex128)
    for a in range(K):
        weighted_W = (Qt2 * Lambda_list[a])[:, None] * W
        w_by_block[a] = z**2 * gamma * alphas[a] * g[a]**2 * (W.T @ weighted_W)
    w_moments = np.sum(w_by_block, axis=0)

    return {'u_moments': u_moments, 'w_moments': w_moments}


def make_M_matrix(m_model, g, z, z0, u_list, w_list, isotropic=False, U=None, W=None):
    #Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    k = len(u_list)
    d = compute_spike_Q_moments(m_model, g, z**2, u_list, w_list, isotropic=isotropic, U=U, W=W)
    M = np.eye(2*k, dtype=float)
    M_w = np.real(d['u_moments']) * z * z0
    M_u = np.real(d['w_moments']) * z * z0
    M[:k, k:] = M_w
    M[k:, :k] = M_u
    return M

########################################################
# functions that depend on the spike

# z0 is a vector of spike strengths
# z is \lambda in the Benaych-Georges/Nadakuditi paper
def compute_spike_eigenvalues(m_model, spike, accuracy=0.1, isotropic=False):
    z0_vec, u_list, w_list = spike["s"], spike["u_list"], spike["v_list"]

    k = len(u_list)
    U = None if isotropic else np.column_stack(u_list)
    W = None if isotropic else np.column_stack(w_list)

    b_plus = mm.compute_b_plus(m_model, accuracy=accuracy, debug=True)

    f_cache = {}

    def f(z2):
        z2_key = float(z2)
        if z2_key in f_cache:
            return f_cache[z2_key]
        z = np.sqrt(z2)
        g = compute_g(m_model, z2, outside_spectrum=True)
        M = make_M_matrix(m_model, g, z, z0_vec, u_list, w_list, isotropic=isotropic, U=U, W=W)
        out = float(np.linalg.det(M))
        f_cache[z2_key] = out
        return out

    z_lo = b_plus + accuracy
    z_hi = np.max(z0_vec**2) + 5

    n_grid = 2*int((z_hi - z_lo)) + 2
    roots = []

    while len(roots) < k:
        print("grid size: ", n_grid)
        print("grid width: ", z_hi - z_lo)
        print("grid accuracy: ", (z_hi - z_lo) / n_grid)
        print("grid:", z_lo, z_hi)

        z_grid = np.linspace(z_lo, z_hi, n_grid)
        f_grid = np.array([f(z) for z in z_grid])

        roots = []
        for i in range(len(z_grid) - 1):
            if f_grid[i] * f_grid[i + 1] < 0:
                root = brentq(f, z_grid[i], z_grid[i + 1], xtol=1e-10)
                roots.append(float(root))

        print(f'grid size {n_grid}: found {len(roots)} / {k} roots: {roots}')

        if len(roots) < k:
            n_grid += 2*n_grid

        if (z_grid[1] - z_grid[0] < accuracy) or n_grid > 1000:
            break

        print("not all roots found, refining grid")

    # sort roots from greatest to least
    roots = sorted(roots, key=lambda x: -x)

    return roots


def compute_spike_projection_coef(m_model, spike, z, isotropic=False):
    z0_vec, u_list, w_list = spike["s"], spike["u_list"], spike["v_list"]

    g = compute_g(m_model, z**2, outside_spectrum=True)
    alpha, beta = compute_alphas_betas(m_model, spike, g, z, isotropic=isotropic)
    alpha0 = compute_alpha0(m_model, spike, g, alpha, beta, z, isotropic=isotropic)
    delta = compute_delta(m_model, spike, g, alpha, beta, z)

    return {'alpha': alpha, 'beta': beta, 'alpha0': alpha0, 'delta': delta}


def compute_alphas_betas(m_model, spike, g, z, isotropic=False, debug=False):
    z0_vec, u_list, w_list = spike["s"], spike["u_list"], spike["v_list"]

    k = len(u_list)

    d = compute_spike_Q2_moments(m_model, g, z**2, u_list, w_list, isotropic=isotropic)
    u_moments = (d['u_moments'].T * z0_vec).T * z0_vec
    w_moments = (d['w_moments'].T * z0_vec).T * z0_vec

    M = make_M_matrix(m_model, g, z, z0_vec, u_list, w_list, isotropic=isotropic)
    u, s, v = np.linalg.svd(M, full_matrices=False)
    v = v.T
    # find the s that are 0 up to 1e-6
    zero_s = s < 1e-6
    if debug:
        print("computing alphas s: ", s)
    if np.sum(zero_s) > 1:
        raise ValueError(f"More than one zero singular value of M")
    if np.sum(zero_s) == 0:
        if debug:
           print("No zero singular value, choosing smallest singular value", s[len(s) - 1])
        zero_s[len(zero_s) - 1] = True

    alpha_beta_vec = v[:, zero_s]
    alpha = alpha_beta_vec[:k].reshape(k, 1)
    beta = alpha_beta_vec[k:].reshape(k, 1)
    val = alpha.T @ w_moments @ alpha + z**2 * beta.T @ u_moments @ beta

    def f(c, val):
        out = c**2 * val - 1.0
        return float(np.real(out))

    c_lo = 0
    c_hi = 1.0
    while f(c_hi, val) * f(c_lo, val) > 0:
        c_hi *= 2.0

    root = brentq(f, c_lo, c_hi, args=(val,), xtol=1e-10)

    alpha = np.real_if_close(root * alpha, tol=1000).reshape(-1)
    beta  = np.real_if_close(root * beta,  tol=1000).reshape(-1)

    alpha = alpha.astype(float, copy=False)
    beta  = beta.astype(float, copy=False)

    return alpha, beta


def compute_alpha0(m_model, spike, g, alpna, beta, z, isotropic=False):
    z0_vec, u_list, w_list = spike["s"], spike["u_list"], spike["v_list"]

    Q = make_Q_bar(m_model, g)

    n = len(u_list[0])
    k = len(u_list)

    alpha0 = 0
    for j in range(k):
        if not isotropic:
            alpha0 -= beta[j] * z0_vec[j] * z * np.sum(Q.real * u_list[j])/np.sqrt(n)
        else:
            alpha0 -= beta[j] * z0_vec[j] * z * np.sum(Q.real) * np.sum(u_list[j]) / (n * np.sqrt(n))

    return alpha0


# compute coefficients of left singular vector for isotropic vectors restricted to a mixture
def compute_delta(m_model, spike, g, alpha, beta, z, isotropic=False):
    Lambda_list, pop_alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]
    z0_vec, u_list, w_list = spike["s"], spike["u_list"], spike["v_list"]

    Q = make_Q_bar(m_model, g)
    n = len(u_list[0])
    r = len(u_list)          # number of spikes
    k = len(pop_alphas)      # number of mixtures
    n_a = create_ncells_per_mixture(n, pop_alphas)
    starts = np.concatenate([[0], np.cumsum(n_a)])

    Q_tilde_bar = make_tilde_Q_bar(m_model, g, z**2)
    R = make_R(m_model, g, z**2, Q_tilde_bar=Q_tilde_bar)
    U = np.column_stack(u_list)
    W = np.column_stack(w_list)

    weighted_alpha = alpha * z0_vec
    weighted_beta = beta * z0_vec

    # compute ||D_a x||^2 over the mixtures a
    q2_blocks = np.zeros((k, r, r), dtype=np.complex128)
    qbar_sq = Q ** 2
    for b in range(k):
        start, end = starts[b], starts[b + 1]
        weighted_U = qbar_sq[start:end, None] * U[start:end, :]
        q2_blocks[b] = U[start:end, :].T @ weighted_U
    q2_blocks = q2_blocks + np.einsum('ab,bij->aij', R, q2_blocks)

    tq_blocks = np.zeros((k, r, r), dtype=np.complex128)
    Qt2 = Q_tilde_bar ** 2
    for a in range(k):
        weighted_W = (Qt2 * Lambda_list[a])[:, None] * W
        tq_blocks[a] = z**4 * gamma * pop_alphas[a] * g[a]**2 * (W.T @ weighted_W)

    Da_x_norm2 = (
        np.einsum('i,j,aij->a', weighted_alpha, weighted_alpha, tq_blocks)
        + np.einsum('i,j,aij->a', weighted_beta, weighted_beta, z**2 * q2_blocks)
    )

    # compute cross term (ignore alpha0)
    cross_term = np.zeros(k, dtype=np.complex128)
    last_term = np.zeros(k, dtype=np.complex128)
    for a in range(k):
        start, end = starts[a], starts[a + 1]
        U_block = U[start:end, :]
        q_block = Q.real[start:end]
        cross_matrix = U_block.T @ (q_block[:, None] * U_block)
        last_matrix = U_block.T @ U_block
        cross_term[a] = 2 * z * alpha @ cross_matrix @ weighted_beta
        last_term[a] = alpha @ last_matrix @ alpha

    delta2 = Da_x_norm2 + cross_term + last_term
    delta = np.sqrt(delta2)

    return delta
