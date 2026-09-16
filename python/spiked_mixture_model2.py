# This files computes the cross correlation between the noise components of
# two different spikes.   Functions in spiked_mixture_model.py are used to
# compute the noise components of a spike.

# The main generalization is to let z_1 and z_2 differ in Proposition 1.6
# of the Benaych-Georges, Coulliet paper.

import numpy as np
import scipy.linalg as la
import scipy.stats as stats

import spiked_mixture_model as smm


# def make_R_simple(m_model, g1, g2, z1, z2, Q_tilde_bar1=None, Q_tilde_bar2=None):
#     Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]
#
#     K = len(Lambda_list)
#     p = Lambda_list[0].shape[0]
#     alphas = np.asarray(alphas)
#     g1 = np.asarray(g1)
#     g2 = np.asarray(g2)
#
#     if Q_tilde_bar1 is None:
#         Q_tilde_bar1 = smm.make_tilde_Q_bar(m_model, g1, z1)
#     if Q_tilde_bar2 is None:
#         Q_tilde_bar2 = smm.make_tilde_Q_bar(m_model, g2, z2)
#
#
#     Omega = np.zeros((K, K), dtype=np.complex128)
#     for a in range(K):
#         for b in range(K):
#             trace_term = np.dot(Lambda_list[a] * Q_tilde_bar1 * Q_tilde_bar2, Lambda_list[b]) / p
#             Omega[a, b] = gamma * alphas[b] * z1 * z2 * g1[a] * g2[a] * trace_term
#
#     # T_a = row_sum of M solves T_a = sum_b Omega[a,b] * (1 + T_b),
#     # i.e. T = (I - Omega)^{-1} Omega * ones.
#     IminusOmega_inv = np.linalg.inv(np.eye(K) - Omega)
#     R = IminusOmega_inv @ Omega
#     for i in range(K):
#         for j in range(K):
#             R[i, j] = alphas[i]/alphas[j] * R[i, j]
#
#     return R


def make_R(m_model, g1, g2, z1, z2, Q_tilde_bar1=None, Q_tilde_bar2=None):
    """Vectorized version of make_R_simple (see the commented-out reference
    copy above): replaces the (a, b) double loop building Omega, and the
    (i, j) double loop rescaling R, with matrix operations.
    """
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    K = len(Lambda_list)
    p = Lambda_list[0].shape[0]
    alphas = np.asarray(alphas)
    g1 = np.asarray(g1)
    g2 = np.asarray(g2)

    if Q_tilde_bar1 is None:
        Q_tilde_bar1 = smm.make_tilde_Q_bar(m_model, g1, z1)
    if Q_tilde_bar2 is None:
        Q_tilde_bar2 = smm.make_tilde_Q_bar(m_model, g2, z2)

    # trace_term[a, b] = dot(Lambda_list[a] * Q_tilde_bar1 * Q_tilde_bar2, Lambda_list[b]) / p
    Lambda_matrix = np.asarray(Lambda_list)  # (K, p)
    QtQ = Q_tilde_bar1 * Q_tilde_bar2
    trace_term = (Lambda_matrix * QtQ[None, :]) @ Lambda_matrix.T / p  # (K, K)
    Omega = (gamma * z1 * z2) * (g1 * g2)[:, None] * trace_term * alphas[None, :]

    # T_a = row_sum of M solves T_a = sum_b Omega[a,b] * (1 + T_b),
    # i.e. T = (I - Omega)^{-1} Omega * ones.
    IminusOmega_inv = np.linalg.inv(np.eye(K) - Omega)
    R = IminusOmega_inv @ Omega
    R = R * (alphas[:, None] / alphas[None, :])

    return R


# first equation in proposition 1.6
def compute_Q2(m_model, R, Q_bar1, Q_bar2, u1, u2, restrict_to_a=None, isotropic=False):
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    K = len(alphas)
    n = len(Q_bar1)
    n_a = smm.create_ncells_per_mixture(n, alphas)

    starts = np.cumsum(np.concatenate([[0], n_a]))

    # s_b = u^T Qbar D_b Qbar u
    s = np.zeros(K, dtype=np.complex128)

    for b in range(K):
        start = starts[b]
        end = starts[b + 1]
        if not isotropic:
            s[b] = np.sum((Q_bar1[start:end] * Q_bar2[start:end]) * u1[start:end] * u2[start:end])
        else:
            inner = np.dot(u1, u2)
            s[b] = (inner / n) * np.sum(Q_bar1[start:end] * Q_bar2[start:end])

    # After summing over a:
    # sum_a QD_aQ ~ sum_b (1 + sum_a R[a,b]) Qbar D_b Qbar
    coeffs = 1.0 + np.sum(R, axis=0)

    q2_by_block = s + R @ s
    if restrict_to_a is None:
        return np.sum(q2_by_block)
    else:
        return q2_by_block[restrict_to_a]


def compute_tQWWtQ(m_model, g1, g2, z1, z2, w1, w2, restrict_to_a=None,
                   isotropic=False, Q_tilde_bar1=None, Q_tilde_bar2=None):
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    if Q_tilde_bar1 is None:
        Q_tilde_bar1 = smm.make_tilde_Q_bar(m_model, g1, z1)
    if Q_tilde_bar2 is None:
        Q_tilde_bar2 = smm.make_tilde_Q_bar(m_model, g2, z2)

    K = len(Lambda_list)
    p = Lambda_list[0].shape[0]
    result = np.zeros(K, dtype=np.complex128)
    for a in range(K):
        q_lambda_q = Q_tilde_bar1 * Lambda_list[a] * Q_tilde_bar2
        if not isotropic:
            result[a] = (
                z1 * z2 * gamma * alphas[a] * g1[a] * g2[a]
                * np.dot(q_lambda_q, w1 * w2)
            )
        else:
            inner = np.dot(w1, w2)
            result[a] = (
                z1 * z2 * gamma * alphas[a] * g1[a] * g2[a]
                * (inner / p) * np.sum(q_lambda_q)
            )

    if restrict_to_a is None:
        return np.sum(result)
    else:
        return result[restrict_to_a]


def _block_grams(U, starts):
    """The per-mixture Gram matrices Ub.T @ Ub, stacked (K, r, r).

    Independent of the eigenvalue pair, so built once per model and reused
    across all n_eigs*(n_eigs+1)/2 inner iterations.
    """
    K = len(starts) - 1
    return np.stack([U[starts[b]:starts[b + 1], :].T @ U[starts[b]:starts[b + 1], :]
                     for b in range(K)]).astype(np.complex128)


def compute_Q2_all(R, Q_bar1, Q_bar2, U, starts, mixture_indices=None, isotropic=False,
                   block_grams=None):
    """Vectorized version of compute_Q2 above: instead of being called once
    per (i, j, a) triple, computes the full r x r matrix of q2 values --
    across every (i, j) pair of spike-rank columns of U at once, via matrix
    multiplication -- for every requested mixture in one pass.

    Returns an array of shape (len(rows), r, r), aligned with `rows` (=
    range(K) if mixture_indices is None, else mixture_indices).
    """
    K = len(starts) - 1
    r = U.shape[1]
    rows = list(range(K)) if mixture_indices is None else list(mixture_indices)

    if isotropic:
        n = U.shape[0]
        Inner = U.T @ U  # (r, r), same for every block
        block_weight = np.array([
            np.sum(Q_bar1[starts[b]:starts[b + 1]] * Q_bar2[starts[b]:starts[b + 1]])
            for b in range(K)
        ])
        S = (block_weight[:, None, None] / n) * Inner[None, :, :]
    else:
        # make_Q_bar returns gamma*g[a] repeated over block a, so Q_bar is
        # CONSTANT within each block. The weighted Gram therefore factorizes
        # as (Q_bar1[b] * Q_bar2[b]) * (Ub.T @ Ub), and the Gram does not
        # depend on the (i, j) eigenvalue pair -- it is passed in, built once
        # per model rather than once per pair.
        grams = _block_grams(U, starts) if block_grams is None else block_grams
        wb = np.array([Q_bar1[starts[b]] * Q_bar2[starts[b]] for b in range(K)])
        S = wb[:, None, None] * grams

    # R @ S needs every block of S (cross-mixture coupling), but we only
    # need the output rows for the requested mixtures. Written as a single
    # (K, K) @ (K, r*r) matmul so it lands in BLAS -- einsum does not, and
    # was ~25x slower on a K=311 model.
    R_rows = R[rows, :]
    coupled = (R_rows @ S.reshape(K, -1)).reshape(len(rows), r, r)
    return S[rows] + coupled


def compute_tQWWtQ_all(m_model, g1, g2, z1, z2, W, mixture_indices=None,
                       isotropic=False, Q_tilde_bar1=None, Q_tilde_bar2=None):
    """Vectorized version of compute_tQWWtQ above: computes the full r x r
    matrix of tq values for every requested mixture in one pass, instead of
    being called once per (i, j, a) triple.
    """
    Lambda_list, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]

    if Q_tilde_bar1 is None:
        Q_tilde_bar1 = smm.make_tilde_Q_bar(m_model, g1, z1)
    if Q_tilde_bar2 is None:
        Q_tilde_bar2 = smm.make_tilde_Q_bar(m_model, g2, z2)

    K = len(Lambda_list)
    p = Lambda_list[0].shape[0]
    r = W.shape[1]
    rows = list(range(K)) if mixture_indices is None else list(mixture_indices)

    result = np.zeros((len(rows), r, r), dtype=np.complex128)

    if isotropic:
        Inner = W.T @ W  # (r, r), same for every mixture
        for idx, a in enumerate(rows):
            q_lambda_q = Q_tilde_bar1 * Lambda_list[a] * Q_tilde_bar2
            prefactor = z1 * z2 * gamma * alphas[a] * g1[a] * g2[a] * (np.sum(q_lambda_q) / p)
            result[idx] = prefactor * Inner
    else:
        for idx, a in enumerate(rows):
            q_lambda_q = Q_tilde_bar1 * Lambda_list[a] * Q_tilde_bar2
            weighted_W = W * q_lambda_q[:, None]
            Ta = weighted_W.T @ W
            result[idx] = (z1 * z2 * gamma * alphas[a] * g1[a] * g2[a]) * Ta

    return result


def compute_cross_terms(Q1, Q2, U, starts, rows, block_grams=None):
    """Vectorized replacement for the per-(i, j) q2_cross/q1_cross/last_ij
    scalar dot products inside compute_delta_dot_simple: computes the full
    r x r matrix for each requested mixture at once via matrix multiplication.
    """
    # Q1/Q2 are block-constant (see compute_Q2_all), so each of these is just
    # a scalar multiple of that block's Gram matrix -- and last_ij IS the Gram,
    # which is why it was being rebuilt identically on every (i, j) pair.
    grams = _block_grams(U, starts) if block_grams is None else block_grams
    G = grams[rows]
    c2 = np.array([Q2.real[starts[a]] for a in rows])
    c1 = np.array([Q1.real[starts[a]] for a in rows])
    return c2[:, None, None] * G, c1[:, None, None] * G, G


# def compute_delta_simple(m_model, spike, alpha, beta, z, isotropic=False):
#     g = smm.compute_g(m_model, z**2, outside_spectrum=True)

#     Lambda_list, pop_alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]
#     z0_vec, u_list, w_list = spike["s"], spike["u_list"], spike["v_list"]

#     Q = smm.make_Q_bar(m_model, g)
#     n = len(u_list[0])
#     r = len(u_list)          # number of spikes
#     k = len(pop_alphas)      # number of mixtures
#     n_a = smm.create_ncells_per_mixture(n, pop_alphas)
#     starts = np.concatenate([[0], np.cumsum(n_a)])

#     Q_tilde_bar = smm.make_tilde_Q_bar(m_model, g, z**2)
#     U = np.column_stack(u_list)
#     W = np.column_stack(w_list)

#     weighted_alpha = alpha * z0_vec
#     weighted_beta = beta * z0_vec

#     # compute ||D_a x||^2 over the mixtures a
#     R = smm.make_R(m_model, g, z**2, Q_tilde_bar=Q_tilde_bar)
#     Da_x_norm2 = np.zeros(k, dtype=np.complex128)
#     for a in range(k):
#         for i in range(r):
#             for j in range(r):
#                 q2_ij = smm.compute_Q2(
#                     m_model,
#                     g,
#                     z**2,
#                     U[:, i],
#                     U[:, j],
#                     restrict_to_a=a,
#                     isotropic=isotropic,
#                     Q_bar=Q,
#                     R=R,
#                 )
#                 tq_ij = smm.compute_tQWWtQ(
#                     m_model,
#                     g,
#                     z**2,
#                     W[:, i],
#                     W[:, j],
#                     restrict_to_a=a,
#                     isotropic=isotropic,
#                     Q_tilde_bar=Q_tilde_bar,
#                 )
#                 Da_x_norm2[a] += weighted_alpha[i] * tq_ij * weighted_alpha[j]
#                 Da_x_norm2[a] += weighted_beta[i] * z**2 * q2_ij * weighted_beta[j]

#     # compute cross term (ignore alpha0)
#     cross_term = np.zeros(k, dtype=np.complex128)
#     last_term = np.zeros(k, dtype=np.complex128)
#     for a in range(k):
#         start, end = starts[a], starts[a + 1]
#         for i in range(r):
#             for j in range(r):
#                 cross_ij = np.dot(Q.real[start:end] * U[start:end, i], U[start:end, j])
#                 last_ij = np.dot(U[start:end, i], U[start:end, j])
#                 cross_term[a] += 2 * z * alpha[i] * cross_ij * weighted_beta[j]
#                 last_term[a] += alpha[i] * last_ij * alpha[j]

#     delta2 = Da_x_norm2 + cross_term + last_term
#     delta = np.sqrt(delta2)

#     return delta


# def compute_delta_dot_simple(m_model, spike, alpha1, beta1, z1, alpha2, beta2, z2,
#                       isotropic=False, mixture_indices=None):
#     g1 = smm.compute_g(m_model, z1**2, outside_spectrum=True)
#     g2 = smm.compute_g(m_model, z2**2, outside_spectrum=True)
#
#     pop_alphas = m_model["mixture_frequencies"]
#     z0_vec, u_list, w_list = spike["s"], spike["u_list"], spike["v_list"]
#
#     Q1 = smm.make_Q_bar(m_model, g1)
#     Q2 = smm.make_Q_bar(m_model, g2)
#     n = len(u_list[0])
#     r = len(u_list)
#     k = len(pop_alphas)
#     n_a = smm.create_ncells_per_mixture(n, pop_alphas)
#     starts = np.concatenate([[0], np.cumsum(n_a)])
#
#     Q_tilde_bar1 = smm.make_tilde_Q_bar(m_model, g1, z1**2)
#     Q_tilde_bar2 = smm.make_tilde_Q_bar(m_model, g2, z2**2)
#     R = make_R(
#         m_model,
#         g1,
#         g2,
#         z1**2,
#         z2**2,
#         Q_tilde_bar1=Q_tilde_bar1,
#         Q_tilde_bar2=Q_tilde_bar2,
#     )
#
#     U = np.column_stack(u_list)
#     W = np.column_stack(w_list)
#
#     weighted_alpha1 = alpha1 * z0_vec
#     weighted_alpha2 = alpha2 * z0_vec
#     weighted_beta1 = beta1 * z0_vec
#     weighted_beta2 = beta2 * z0_vec
#
#     dot_by_mixture = np.zeros(k, dtype=np.complex128)
#     for a in (range(k) if mixture_indices is None else mixture_indices):
#         start, end = starts[a], starts[a + 1]
#         for i in range(r):
#             for j in range(r):
#                 q2_ij = compute_Q2(
#                     m_model,
#                     R,
#                     Q1,
#                     Q2,
#                     U[:, i],
#                     U[:, j],
#                     restrict_to_a=a,
#                     isotropic=isotropic,
#                 )
#                 tq_ij = compute_tQWWtQ(
#                     m_model,
#                     g1,
#                     g2,
#                     z1**2,
#                     z2**2,
#                     W[:, i],
#                     W[:, j],
#                     restrict_to_a=a,
#                     isotropic=isotropic,
#                     Q_tilde_bar1=Q_tilde_bar1,
#                     Q_tilde_bar2=Q_tilde_bar2,
#                 )
#
#                 dot_by_mixture[a] += weighted_alpha1[i] * tq_ij * weighted_alpha2[j]
#                 dot_by_mixture[a] += weighted_beta1[i] * z1 * z2 * q2_ij * weighted_beta2[j]
#
#                 q2_cross = np.dot(Q2.real[start:end] * U[start:end, i], U[start:end, j])
#                 q1_cross = np.dot(Q1.real[start:end] * U[start:end, i], U[start:end, j])
#                 last_ij = np.dot(U[start:end, i], U[start:end, j])
#                 dot_by_mixture[a] += alpha1[i] * z2 * q2_cross * weighted_beta2[j]
#                 dot_by_mixture[a] += weighted_beta1[i] * z1 * q1_cross * alpha2[j]
#                 dot_by_mixture[a] += alpha1[i] * last_ij * alpha2[j]
#
#     return dot_by_mixture


def compute_delta_dot(m_model, U, W, starts, ctx1, ctx2, isotropic=False, mixture_indices=None,
                      block_grams=None):
    """Same computation as compute_delta_dot_simple (see the commented-out
    reference copy above), but:
    - takes precomputed per-eigenvalue quantities (ctx1/ctx2, one per
      eigenvector -- see compute_covariance_matrix) and precomputed
      model-level quantities (U, W, starts) instead of recomputing
      g/Q_bar/Q_tilde_bar/U/W/starts from scratch on every call;
    - computes the full r x r matrix of q2/tq/cross terms for every
      requested mixture in one batch (via compute_Q2_all/compute_tQWWtQ_all/
      compute_cross_terms), instead of calling compute_Q2/compute_tQWWtQ
      once per (i, j, a) triple.
    """
    pop_alphas = m_model["mixture_frequencies"]
    k = len(pop_alphas)

    z1, g1, Q1, Q_tilde_bar1 = ctx1["z"], ctx1["g"], ctx1["Qbar"], ctx1["Qtilde"]
    z2, g2, Q2, Q_tilde_bar2 = ctx2["z"], ctx2["g"], ctx2["Qbar"], ctx2["Qtilde"]
    alpha1, weighted_alpha1, weighted_beta1 = ctx1["alpha"], ctx1["weighted_alpha"], ctx1["weighted_beta"]
    alpha2, weighted_alpha2, weighted_beta2 = ctx2["alpha"], ctx2["weighted_alpha"], ctx2["weighted_beta"]

    R = make_R(
        m_model,
        g1,
        g2,
        z1**2,
        z2**2,
        Q_tilde_bar1=Q_tilde_bar1,
        Q_tilde_bar2=Q_tilde_bar2,
    )

    rows = list(range(k)) if mixture_indices is None else list(mixture_indices)

    Q2_mat = compute_Q2_all(R, Q1, Q2, U, starts, mixture_indices=rows, isotropic=isotropic,
                            block_grams=block_grams)
    TQ_mat = compute_tQWWtQ_all(
        m_model, g1, g2, z1**2, z2**2, W, mixture_indices=rows, isotropic=isotropic,
        Q_tilde_bar1=Q_tilde_bar1, Q_tilde_bar2=Q_tilde_bar2,
    )
    q2_cross, q1_cross, last_ij = compute_cross_terms(Q1, Q2, U, starts, rows,
                                                      block_grams=block_grams)

    wa1_wa2 = np.outer(weighted_alpha1, weighted_alpha2)
    wb1_wb2 = np.outer(weighted_beta1, weighted_beta2)
    a1_wb2 = np.outer(alpha1, weighted_beta2)
    wb1_a2 = np.outer(weighted_beta1, alpha2)
    a1_a2 = np.outer(alpha1, alpha2)

    term = (
        wa1_wa2[None, :, :] * TQ_mat
        + (z1 * z2) * wb1_wb2[None, :, :] * Q2_mat
        + z2 * a1_wb2[None, :, :] * q2_cross
        + z1 * wb1_a2[None, :, :] * q1_cross
        + a1_a2[None, :, :] * last_ij
    )
    dot_by_rows = term.sum(axis=(1, 2))

    dot_by_mixture = np.zeros(k, dtype=np.complex128)
    for idx, a in enumerate(rows):
        dot_by_mixture[a] = dot_by_rows[idx]

    return dot_by_mixture


def compute_covariance_matrix(m_model, spike, evs, alpha_matrix, beta_matrix,
                              isotropic=False, mixture_indices=None):
    n_eigs = len(evs)
    n_mixtures = len(m_model["mixture_frequencies"])

    # model/spike-level quantities that don't depend on i or j at all --
    # computed once here instead of once per (i, j) pair as before
    z0_vec = spike["s"]
    U = np.column_stack(spike["u_list"])
    W = np.column_stack(spike["v_list"])
    n = U.shape[0]
    pop_alphas = m_model["mixture_frequencies"]
    n_a = smm.create_ncells_per_mixture(n, pop_alphas)
    starts = np.concatenate([[0], np.cumsum(n_a)])

    # per-eigenvalue quantities that only depend on i (or j), not the pair --
    # computed once per eigenvector instead of once per (i, j) pair as before
    contexts = []
    for i in range(n_eigs):
        print(f"Computing context for eigenvalue {i} of {n_eigs}")
        g_i = smm.compute_g(m_model, evs[i], outside_spectrum=True)
        contexts.append({
            "z": np.sqrt(evs[i]),
            "g": g_i,
            "Qbar": smm.make_Q_bar(m_model, g_i),
            "Qtilde": smm.make_tilde_Q_bar(m_model, g_i, evs[i]),
            "alpha": alpha_matrix[:, i],
            "weighted_alpha": alpha_matrix[:, i] * z0_vec,
            "weighted_beta": beta_matrix[:, i] * z0_vec,
        })

    # (i, j)-independent, so built once here rather than inside every iteration
    block_grams = None if isotropic else _block_grams(U, starts)

    delta_covariance = np.zeros((n_mixtures, n_eigs, n_eigs), dtype=np.complex128)

    iters = 0
    total_iters = n_eigs * (n_eigs + 1) // 2
    for i in range(n_eigs):
        for j in range(i, n_eigs):
            print(f"Computing covariance for eigenvalue {i},{j}  of {n_eigs}")
            print(f"Progress: {iters}/{total_iters}")
            cov_ij = compute_delta_dot(
                m_model, U, W, starts, contexts[i], contexts[j],
                isotropic=isotropic,
                mixture_indices=mixture_indices,
                block_grams=block_grams,
            )
            iters += 1

            delta_covariance[:, i, j] = cov_ij
            if j != i:
                delta_covariance[:, j, i] = cov_ij

    return delta_covariance
