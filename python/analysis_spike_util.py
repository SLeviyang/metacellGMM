import numpy as np

import lognorm_workflow_analytic as lwa


def compute_projection_matrices(spec_proj, spec_base, cell_labels):
    """Project one spec's PC scores onto another's gene basis, then find the
    best per-axis rescaling of that projection against the base's METACELL
    MEANS.

    spec_proj and spec_base are spec dicts ({"U", "sv", "V"}, as stored in an
    analysis_spike pkl); cell_labels gives the metacell of each row and must
    be aligned positionally with BOTH specs' rows. Every spec in one pkl
    shares that row order (anndata2spike realigns spec_analytic to match), so
    the pkl's own "cell_labels" groups either spec correctly -- but nothing
    here can check the two specs came from the same pkl, only that their
    shapes are consistent, so pairing specs across pkls is the caller's
    responsibility.

    Q = V_proj' V_base                              (k x k)

    is the same construction as the deleted project_spike: for a vector with
    spec_proj's PC coordinates c, V_proj c is its gene-space vector, and
    V_base' (V_proj c) = Q c is that vector's coordinates in spec_base's
    V-basis. When spec_proj is spec_analytic its V is pca_theory's
    mean_gene_matrix, which is not orthonormal, so Q is a looser notion of
    "project onto the base's basis" than it would be for an SVD-derived V --
    see project_spike's docstring history for the same caveat.

    D is the diagonal minimising sum_i || mu_i - mu'_i Q D ||^2 (applied in
    that order, mu'_i Q D: rescale AFTER rotating into spec_base's basis,
    i.e. correct each of the base's own axes by one scalar), where mu_i is
    spec_base's mean for metacell i and mu'_i is spec_proj's -- MU_base and
    MU_proj stack those means as (n_mix, k) matrices, one metacell per row.
    Both are the empirical per-metacell mean of that spec's own scores
    U[:, :k] * sv[:k]. D is fit against the metacell CENTROIDS, not
    individual cells: a method with the right per-cell spread but off-centre
    metacells is scored purely on the centres. (An earlier version fit D
    against every cell instead, minimising a per-cell objective that turned
    out to systematically overcorrect the metacell-level comparison this
    function exists for. A later version also tried fitting D before Q, i.e.
    mu'_i D Q -- on real data that order gave a worse mu vs mu' error than
    mu'_i Q D on most datasets, so only mu'_i Q D remains.)

    k is taken from the specs themselves rather than from pca_theory:
    anndata2spike already truncates every spec to k columns, so V's second
    dimension is k. spec_proj and spec_base must agree on it.

    D acts on Q's OUTPUT columns, so the Frobenius norm separates column by
    column, || MU_base[:,j] - d_j M[:,j] ||^2 with M = MU_proj @ Q, an
    ordinary scalar least squares problem per column with closed form

        d_j = <M[:,j], MU_base[:,j]> / <M[:,j], M[:,j]>

    D=I is a feasible choice for the diagonal being optimised, so the fitted D
    can only do at least as well as Q alone: it cannot increase
    sum_i||mu_i - mu'_i Q D||^2 relative to D=I.

    Sign invariance: every term contributing to the objective is a product of
    something from spec_proj (its scores or a column of them) and something
    from Q, or of something from Q and something from spec_base -- flipping a
    method's column j sign always flips BOTH factors of such a product
    together (the score column j and Q's matching row/column, since Q is built
    from the same V pair), so every term, and hence the whole objective as a
    function of d, is unchanged and the argmin d does not move.

    Returns (Q, D): Q is (k, k), D is (k, k) diagonal.
    """
    U_proj = np.asarray(spec_proj["U"])
    sv_proj = np.asarray(spec_proj["sv"])
    V_proj = np.asarray(spec_proj["V"])
    U_base = np.asarray(spec_base["U"])
    sv_base = np.asarray(spec_base["sv"])
    V_base = np.asarray(spec_base["V"])
    cell_labels = np.asarray(cell_labels)

    k = V_proj.shape[1]
    if V_base.shape[1] != k:
        raise ValueError(f"spec_proj and spec_base disagree on k: "
                         f"{k} vs {V_base.shape[1]}")
    # the two specs are only interchangeable row for row if they describe the
    # same cells in the same order, which cell_labels is then used to group
    if U_proj.shape[0] != U_base.shape[0]:
        raise ValueError(f"spec_proj and spec_base have different row counts: "
                         f"{U_proj.shape[0]} vs {U_base.shape[0]}")
    if len(cell_labels) != U_proj.shape[0]:
        raise ValueError(f"cell_labels has {len(cell_labels)} entries but the specs "
                         f"have {U_proj.shape[0]} rows")

    Q = V_proj.T @ V_base

    mixture_labels = np.unique(cell_labels)
    Y_proj = U_proj[:, :k] * sv_proj[:k]        # U*sv, spec_proj's own scores
    Y_base = U_base[:, :k] * sv_base[:k]        # U'*sv', spec_base's own scores
    MU_proj = np.stack([Y_proj[cell_labels == lb, :].mean(axis=0) for lb in mixture_labels])
    MU_base = np.stack([Y_base[cell_labels == lb, :].mean(axis=0) for lb in mixture_labels])

    M = MU_proj @ Q                         # mu'_i Q, one row per metacell
    # d_j = argmin_d || MU_base[:,j] - d * M[:,j] ||^2, closed form per column
    numer = np.einsum("ij,ij->j", M, MU_base)
    denom = np.einsum("ij,ij->j", M, M)
    d = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    D = np.diag(d)

    return Q, D


def compute_spike_projection(pca_theory, spec_base, cell_labels_base):
    """The analytic model's per-metacell mean and covariance AFTER the Q, D
    transform onto spec_base's basis.

    pca_theory is lognorm_workflow_analytic.log_normalized_PCA's output;
    spec_base is a spec dict ({"U", "sv", "V"}) and cell_labels_base gives the
    metacell of each of ITS rows.

    mu'_i, Sigma'_i are the model's own noise-free moments, straight from
    lwa.get_GMM(pca_theory, label) -- not an empirical mean/covariance of
    sampled scores. get_GMM already returns them in PC-score units (mean_matrix
    scaled by values, covariance scaled by outer(values, values) and by the
    mixture's cell count), so they are directly comparable to a spec's own
    U*sv moments. With A = Q @ D:

        mean_i       = mu'_i @ A                    (row-vector convention,
                                                       matching mu'_i Q D in
                                                       compute_projection_matrices)
        covariance_i = A' @ Sigma'_i @ A             (standard covariance
                                                       transform for a linear
                                                       map y = x A on a row
                                                       vector x)

    Q and D come from compute_projection_matrices, which needs a spec on the
    projected side, so one is assembled from pca_theory:

        V  = mean_gene_matrix[:, :k]    the model's predicted gene loadings
        sv = values[:k]
        U  = mean_matrix, REALIGNED into cell_labels_base's row order

    making that spec's scores U*sv = mean_matrix * values, get_GMM's own
    convention, so D is fit against the model's noise-free metacell means.

    The realignment matters: pca_theory's rows follow its own sample_labels
    (contiguous metacell blocks) while spec_base's follow the h5ad's shuffled
    cell order, and the two agree in only a few percent of positions -- so
    cell_labels_base would mis-group pca_theory's rows as they sit. Rather
    than permute positionally, each cell's row is looked up BY METACELL from
    mean_matrix, which is exact here because mean_matrix is constant within a
    mixture (see pca_to_mixture_pca): rows sharing a label are identical, so
    any 1-to-1 pairing of them -- what a positional realignment would pick --
    gives this same result. The rebuilt U therefore has one row per entry of
    cell_labels_base, which is also what compute_projection_matrices's
    row-count and cell_labels-length checks require.

    Returns {"means": (n_mix, k) array, "covariances": (n_mix, k, k) array,
    "mixture_labels": (n_mix,) array}, mixture_labels being np.unique of
    cell_labels_base, so sorted and matching the grouping
    compute_projection_matrices fits D against.
    """
    cell_labels_base = np.asarray(cell_labels_base)
    k = len(pca_theory["values"])

    mean_matrix = np.asarray(pca_theory["mean_matrix"])[:, :k]
    sample_labels = np.asarray(pca_theory["sample_labels"])

    missing = set(np.unique(cell_labels_base)) - set(np.unique(sample_labels))
    if missing:
        raise ValueError(f"cell_labels_base has metacells absent from "
                         f"pca_theory['sample_labels']: {sorted(missing)}")

    # one representative row per metacell, then broadcast to cell_labels_base's
    # order -- exact because mean_matrix is constant within a mixture
    row_by_label = {lb: mean_matrix[sample_labels == lb][0]
                    for lb in np.unique(cell_labels_base)}
    U_proj = np.stack([row_by_label[lb] for lb in cell_labels_base])

    spec_proj = {"U": U_proj,
                 "sv": np.asarray(pca_theory["values"])[:k],
                 "V": np.asarray(pca_theory["mean_gene_matrix"])[:, :k]}

    Q, D = compute_projection_matrices(spec_proj, spec_base, cell_labels_base)

    mixture_labels = np.unique(cell_labels_base)
    gmms = [lwa.get_GMM(pca_theory, lb) for lb in mixture_labels]
    MU_proj = np.stack([np.asarray(g["mean"])[:k] for g in gmms])
    SIGMA_proj = np.stack([np.asarray(g["covariance"])[:k, :k] for g in gmms])

    A = Q @ D
    means = MU_proj @ A
    covariances = np.einsum("ab,nbc,cd->nad", A.T, SIGMA_proj, A)

    return {"means": means, "covariances": covariances,
            "mixture_labels": mixture_labels}


def simulate_spike_projection(sp, cell_labels, rng=None):
    """Draw one Gaussian score vector per entry of cell_labels from the
    projected per-metacell moments in `sp`.

    sp is compute_spike_projection's output ({"means", "covariances",
    "mixture_labels"}); row i of the returned (n, k) matrix, n =
    len(cell_labels), is a draw from N(mean_a, covariance_a) for the mixture
    a = cell_labels[i]. cell_labels need not be in any particular order and
    need not use every mixture in sp; each label is matched to sp's rows BY
    VALUE, since sp's rows follow sorted mixture_labels while cell_labels
    follows whatever cell order the caller has.

    Sampling is done one block per mixture rather than one call per cell, so
    each covariance is factorised once -- the same pattern
    lognorm_workflow_analytic.sample_mixture_gaussian uses to build
    pca_theory's own vectors.

    The covariances are used as they come from compute_spike_projection, with
    no symmetrising or eigenvalue flooring: A' Sigma A of a PSD Sigma is PSD,
    and in practice these land well inside numpy's tolerance (asymmetry of
    order 1e-19, eigenvalues positive), so multivariate_normal accepts them
    unaltered. Where A is badly conditioned some mixtures do have very small
    eigenvalues, and draws are then nearly degenerate along those directions
    -- that is the model's own shape coming through, not a numerical fault.

    rng is any object with a multivariate_normal method (a
    numpy.random.Generator or the numpy.random module itself); None uses a
    fresh default_rng, so successive calls give independent draws.
    """
    rng = np.random.default_rng() if rng is None else rng

    cell_labels = np.asarray(cell_labels)
    means = np.asarray(sp["means"])
    covariances = np.asarray(sp["covariances"])
    mixture_labels = np.asarray(sp["mixture_labels"])

    missing = set(np.unique(cell_labels)) - set(mixture_labels)
    if missing:
        raise ValueError(f"cell_labels has metacells absent from "
                         f"sp['mixture_labels']: {sorted(missing)}")

    row_of = {lb: i for i, lb in enumerate(mixture_labels)}
    Y = np.empty((len(cell_labels), means.shape[1]))
    for lb in np.unique(cell_labels):
        idx = np.where(cell_labels == lb)[0]
        i = row_of[lb]
        Y[idx, :] = rng.multivariate_normal(means[i], covariances[i], size=len(idx))
    return Y


def rank_adjust_spike_projection(sp, spec_base, cell_labels, rank_adjust=1):
    """Add the best PSD rank-<=rank_adjust correction to each of sp's
    covariances, fitted against spec_base's own empirical covariances.

    For metacell i, Sigma_i is the empirical covariance of spec_base's scores
    U[:, :k] * sv[:k] over that metacell's cells, and M_i is sp's covariance
    for it (compute_spike_projection's output). This finds the b_{i,j} >= 0
    and unit q_{i,j} minimising

        || (Sigma_i - M_i) - sum_j b_{i,j} q_{i,j} q_{i,j}^T ||_F^2

    over j = 1..rank_adjust, and returns sp with covariance i replaced by
    M_i + sum_j b_{i,j} q_{i,j} q_{i,j}^T -- the model's covariance corrected
    toward the empirical one along at most rank_adjust directions.

    Closed form: b_j >= 0 with arbitrary unit q_j means sum_j b_j q_j q_j^T
    ranges over exactly the PSD matrices of rank <= rank_adjust, so this is
    the best PSD rank-constrained Frobenius approximation of the symmetric
    residual R_i = Sigma_i - M_i. It is obtained by eigendecomposing R_i and
    keeping its rank_adjust largest ALGEBRAIC eigenvalues, floored at zero:
    b_j = max(lambda_j, 0), q_j the matching eigenvector. Where fewer than
    rank_adjust of those are positive the surplus terms come out at b_j = 0,
    which is correct -- a PSD term cannot correct a direction along which the
    residual is already negative, so the fit simply uses a lower rank there.
    (rank_adjust=1 is the single largest positive eigenvalue, which is the
    rank-1 case this generalises.)

    Note this is the ALGEBRAIC top of the spectrum, not the largest by
    magnitude: an unconstrained fit would chase a large negative eigenvalue,
    but the b >= 0 constraint rules that out.

    Both M_i and the correction are PSD, so every returned covariance is PSD
    and the adjusted sp stays usable with simulate_spike_projection.

    sp is not mutated; means and mixture_labels are carried through unchanged,
    and cell_labels (which labels spec_base's rows) is matched to sp's rows by
    value. rank_adjust=0 is allowed and returns an unadjusted copy.
    """
    means = np.asarray(sp["means"])
    covariances = np.asarray(sp["covariances"])
    mixture_labels = np.asarray(sp["mixture_labels"])
    cell_labels = np.asarray(cell_labels)

    k = covariances.shape[1]
    if not (isinstance(rank_adjust, (int, np.integer)) and 0 <= rank_adjust <= k):
        raise ValueError(f"rank_adjust must be an integer in [0, {k}], got {rank_adjust!r}")

    U_base = np.asarray(spec_base["U"])
    sv_base = np.asarray(spec_base["sv"])
    if np.asarray(spec_base["V"]).shape[1] != k:
        raise ValueError(f"spec_base has k={np.asarray(spec_base['V']).shape[1]} but sp's "
                         f"covariances are {k} x {k}")
    if len(cell_labels) != U_base.shape[0]:
        raise ValueError(f"cell_labels has {len(cell_labels)} entries but spec_base has "
                         f"{U_base.shape[0]} rows")

    # every mixture in sp needs at least two of spec_base's cells for np.cov
    counts = {lb: int((cell_labels == lb).sum()) for lb in mixture_labels}
    thin = sorted(lb for lb, n in counts.items() if n < 2)
    if thin:
        raise ValueError(f"these mixtures of sp have fewer than 2 cells in cell_labels, "
                         f"so no empirical covariance can be formed: {thin}")

    Y_base = U_base[:, :k] * sv_base[:k]
    adjusted = np.empty_like(covariances)
    for i, lb in enumerate(mixture_labels):
        M = covariances[i]
        if rank_adjust == 0:
            adjusted[i] = M
            continue
        Sigma = np.cov(Y_base[cell_labels == lb, :], rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(Sigma - M)     # ascending
        b = np.maximum(eigvals[-rank_adjust:], 0.0)
        Qr = eigvecs[:, -rank_adjust:]
        adjusted[i] = M + (Qr * b) @ Qr.T

    return {"means": means.copy(), "covariances": adjusted,
            "mixture_labels": mixture_labels.copy()}
