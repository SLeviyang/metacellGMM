import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import mixandmix as mm
import spiked_mixture_model as smm
import spiked_mixture_model2 as smm2


########################################################
# functions to compute the mean and variance of log(1 + NB(mu_ij, theta_ij) * L0/Li_i)



def compute_NB_log_moments(mu, theta, Li, L0):
    """Compute mean and variance of log(1 + NB(mu_ij, theta_ij) * L0/Li_i).

    NB parameterization: mean = mu_ij, variance = mu_ij + mu_ij^2/theta_ij.
    When theta_ij = inf the distribution reduces to Poisson(mu_ij).

    Parameters
    ----------
    mu    : (n_cells, n_genes) array — NB means
    theta : (n_cells, n_genes) array — NB dispersions (inf → Poisson)
    Li    : (n_cells,) array        — per-cell library sizes
    L0    : float                   — reference library size

    Returns
    -------
    M : (n_cells, n_genes) — E[log(1 + X * L0/Li)]
    V : (n_cells, n_genes) — Var[log(1 + X * L0/Li)]
    """
    L_ratio = (L0 / Li)[:, np.newaxis]          # (n_cells, 1) — broadcasts over genes

    M  = np.zeros_like(mu)   # first moment accumulator
    E2 = np.zeros_like(mu)   # second moment accumulator

    mask_nz = mu > 0

    def _nb_log_moments(mask):
        mu_g    = mu[mask]
        theta_g = theta[mask]
        Lr_g    = np.broadcast_to(L_ratio, mu.shape)[mask]

        is_pois = ~np.isfinite(theta_g)
        is_nb   = ~is_pois

        std_g          = np.sqrt(mu_g)                              # Poisson default
        std_g[is_nb]   = np.sqrt(mu_g[is_nb] + mu_g[is_nb]**2 / theta_g[is_nb])
        k_max   = max(1, int((mu_g + 15.0 * std_g + 20).max()))

        # P(X = 0)
        prob           = np.exp(-mu_g)                              # Poisson default
        th_nb, mu_nb   = theta_g[is_nb], mu_g[is_nb]
        prob[is_nb]    = np.exp(th_nb * np.log(th_nb / (th_nb + mu_nb)))
        M_g  = np.zeros_like(mu_g)
        E2_g = np.zeros_like(mu_g)

        for k in range(1, k_max + 1):
            ratio          = mu_g / k                               # Poisson default
            ratio[is_nb]   = (k - 1 + th_nb) / k * (mu_nb / (th_nb + mu_nb))
            prob  = prob * ratio
            lv    = np.log(1.0 + Lr_g * k)
            M_g  += prob * lv
            E2_g += prob * lv**2
            if prob.max() < 1e-10:
                break

        return M_g, E2_g, k

    mask_vlo = mask_nz & (mu <  0.1)
    mask_lo  = mask_nz & (mu >= 0.1) & (mu <= 1.0)
    mask_mid = mask_nz & (mu >  1.0) & (mu <  5.0)
    mask_hi  = mask_nz & (mu >= 5.0)

    for label, mask in (("vlo", mask_vlo), ("lo", mask_lo),
                        ("mid", mask_mid), ("hi", mask_hi)):
        if not mask.any():
            continue
        M_g, E2_g, k_used = _nb_log_moments(mask)
        print(f"group={label}  m_max={mu[mask].max():.3f}  k_used={k_used}")
        M[mask]  = M_g
        E2[mask] = E2_g

    V = E2 - M**2   # Var[Y] = E[Y^2] - (E[Y])^2

    return M, V


def MC_logPoiss(L0, Li, m_grid=None, n_samples=100_000, theta=None):
    """Monte Carlo estimate of E and Var of log(1 + Y·Lᵢ/L₀) over a mu grid,
    where Y ~ Pois(Lᵢ·mu) when theta is None, or Y ~ NegBin(mean=Lᵢ·mu,
    var=Lᵢ·mu + (Lᵢ·mu)²/theta) when theta is provided.

    Parameters
    ----------
    L0       : float             — reference library size
    Li       : float             — cell library size
    m_grid   : array-like or None — grid of mu values; defaults to
                                    np.arange(0, 1.01, 0.01)
    n_samples: int               — Monte Carlo draws per mu (default 100 000)
    theta    : float or None     — NegBin dispersion; None → Poisson

    Returns
    -------
    expected : array, same shape as m_grid — MC mean of log(1 + Y·Lᵢ/L₀)
    var      : array, same shape as m_grid — MC variance of log(1 + Y·Lᵢ/L₀)
    """
    if m_grid is None:
        m_grid = np.arange(0, 1.01, 0.01)
    m_grid = np.asarray(m_grid, dtype=float)

    rng   = np.random.default_rng()
    scale = L0 / Li

    expected = np.empty(len(m_grid))
    var      = np.empty(len(m_grid))

    for i, mu in enumerate(m_grid):
        lam = Li * mu
        if lam == 0:
            expected[i] = 0.0
            var[i]      = 0.0#
        else:
            if theta is None:
                draws = rng.poisson(lam=lam, size=n_samples)
            else:
                # NegBin: mean=lam, var=lam + lam²/theta
                draws = rng.negative_binomial(n=theta, p=theta / (theta + lam),
                                              size=n_samples)
            X           = np.log1p(draws * scale)
            expected[i] = X.mean()
            var[i]      = X.var()

    return expected, var


def plot_logPoiss(m_grid, expected, var, L0=None, Li=None, theta=None,
                  n_samples=None):
    """Plot MC mean and SD of log(1 + Y·Lᵢ/L₀) over a mu grid.

    Parameters
    ----------
    m_grid   : array — mu values (x-axis)
    expected : array — MC mean of log(1 + Y·Lᵢ/L₀)
    var      : array — MC variance of log(1 + Y·Lᵢ/L₀)
    L0, Li, theta, n_samples : optional, used only for the plot title
    """
    sd = np.sqrt(var)

    # sort by mu for clean line plots without changing the caller's arrays
    order = np.argsort(m_grid)
    m_sorted  = m_grid[order]
    e_sorted  = expected[order]
    sd_sorted = sd[order]

    dist_label = "Pois" if theta is None else f"NegBin(θ={theta})"
    title_parts = [dist_label]
    if L0 is not None:
        title_parts.append(f"L₀={L0}")
    if Li is not None:
        title_parts.append(f"Lᵢ={Li}")
    if n_samples is not None:
        title_parts.append(f"{n_samples:,} samples")

    fig, ax = plt.subplots(figsize=(8, 5))
    msize = 10
    ax.plot(m_sorted, e_sorted,  lw=1.5, color="steelblue", label="mean", marker=".", markersize=msize)
    ax.plot(m_sorted, sd_sorted, lw=1.5, color="tomato",    label="sd",   marker=".", markersize=msize)
    ax.set_xlabel("μ")
    ax.set_ylabel("value")
    ax.set_title("log(1 + Y·Lᵢ/L₀)\n" + ",  ".join(title_parts))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def MC_grid(L0, Li, m_grid1, m_grid2, n_samples=100_000, theta=None):
    """Pairwise MC comparison of log(1 + Y·Lᵢ/L₀) between two mu grids.

    For each index i, computes the difference in expected values and returns
    the variance at each point for both grids.

    Parameters
    ----------
    L0, Li    : float         — reference and cell library sizes
    m_grid1   : array-like    — first grid of mu values
    m_grid2   : array-like    — second grid of mu values (same length as m_grid1)
    n_samples : int           — Monte Carlo draws per mu (default 100 000)
    theta     : float or None — NegBin dispersion; None → Poisson

    Returns
    -------
    diff  : array (n,) — expected1[i] - expected2[i] for each i
    var1  : array (n,) — MC variance of log(1 + Y·Lᵢ/L₀) at m_grid1[i]
    var2  : array (n,) — MC variance of log(1 + Y·Lᵢ/L₀) at m_grid2[i]
    """
    m_grid1 = np.asarray(m_grid1, dtype=float)
    m_grid2 = np.asarray(m_grid2, dtype=float)

    expected1, var1 = MC_logPoiss(L0=L0, Li=Li, m_grid=m_grid1,
                                  n_samples=n_samples, theta=theta)
    expected2, var2 = MC_logPoiss(L0=L0, Li=Li, m_grid=m_grid2,
                                  n_samples=n_samples, theta=theta)

    diff = expected1 - expected2

    # ── scatter plot: m_grid1 vs m_grid2, coloured by diff / (var1 + var2) ──
    with np.errstate(invalid="ignore", divide="ignore"):
        denom     = var1 + var2 + diff**2
        mask      = denom != 0
        color_val = (var1[mask] + var2[mask]) / denom[mask]

    dist_label = "Pois" if theta is None else f"NegBin(θ={theta})"
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(m_grid1[mask], m_grid2[mask], c=color_val, cmap="plasma", s=40,
                    edgecolors="none", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, label="(var1 + var2) / (var1 + var2 + diff²)")
    ax.set_xlabel("μ₁  (m_grid1)")
    ax.set_ylabel("μ₂  (m_grid2)")
    ax.set_title(f"MC_grid scatter\n{dist_label},  L₀={L0},  Lᵢ={Li}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    return diff, var1, var2


def mesh_mgrid(max_mu, d_mu=0.01):
    """Return two flat vectors whose paired entries cover the 2-D grid
    [0, max_mu] x [0, max_mu] with spacing d_mu.

    Parameters
    ----------
    max_mu : float — upper bound of both axes
    d_mu   : float — grid spacing (default 0.01)

    Returns
    -------
    mu1_grid : (n²,) array — mu1 coordinate of each grid point
    mu2_grid : (n²,) array — mu2 coordinate of each grid point
    """
    axis = np.arange(0, max_mu + d_mu / 2, d_mu)
    mg1, mg2 = np.meshgrid(axis, axis)
    return mg1.ravel(), mg2.ravel()


########################################################
# functions to analytically compute the singular values and vectors of 
# a negative binomial mixture model passed through the log-normalization workflow

def check_noise_covariance(pca_theory, rtol=1e-6, atol=1e-8):
    """Verify that the diagonal of pca_theory["covariance_matrix"] (see
    log_normalized_PCA) is consistent with pca_theory["delta_matrix"]**2.

    Raises ValueError if they disagree beyond the given tolerance.
    """
    covariance_matrix = pca_theory["covariance_matrix"]
    delta_matrix = pca_theory["delta_matrix"]

    n_eigs = delta_matrix.shape[1]
    expected_diag = delta_matrix ** 2
    observed_diag = np.zeros_like(expected_diag, dtype=np.complex128)
    for i in range(n_eigs):
        observed_diag[:, i] = covariance_matrix[:, i, i]

    if not np.allclose(observed_diag, expected_diag, rtol=rtol, atol=atol):
        max_abs_diff = np.max(np.abs(observed_diag - expected_diag))
        raise ValueError(
            "check_noise_covariance: diagonal of covariance_matrix does not match "
            f"delta_matrix**2; max_abs_diff={max_abs_diff}"
        )


def sample_mixture_gaussian(mean_matrix, covariance_matrix, sample_labels, mixture_labels):
    """Draw one Gaussian sample per cell (row of mean_matrix), jointly across
    the k coordinates, using each mixture's own mean (mean_matrix, constant
    within a mixture's block of rows) and k x k covariance.

    covariance_matrix[a] is n_a times the per-cell covariance for mixture a
    (see check_noise_covariance: diag(covariance_matrix[a]) ==
    delta_matrix[a]**2 == n_a * std_matrix[a]**2 exactly), so it is divided
    by n_a here to get single-cell units before sampling.
    """
    n, k = mean_matrix.shape
    U = np.zeros((n, k))
    for a, label in enumerate(mixture_labels):
        idx = np.where(sample_labels == label)[0]
        n_a = len(idx)
        cov_a = covariance_matrix[a].real / n_a
        U[idx, :] = np.random.multivariate_normal(mean_matrix[idx[0], :], cov_a, size=n_a)
    return U


def log_normalized_PCA(nb, n, L0=10000, compactify_spike=False):
    model, spike = log_normalized_spiked_mixture_model(nb, n, L0)
    evs = analytic_eigenvalues(model, spike)
    k = len(evs)
    if compactify_spike:
        spike = cut_spike(spike, k)
    p = spike["v_list"][0].shape[0]

    r = len(spike["u_list"])
    nm = len(model["mixture_labels"])
    mean_matrix = np.zeros((n, k))
    mean_gene_matrix = np.zeros((p, k))
    std_matrix = np.zeros((n, k))
    alpha_matrix = np.zeros((r, k))
    beta_matrix = np.zeros((r, k))
    delta_matrix = np.zeros((nm, k))
    for i in range(k):
        print(f"Computing left singular vectors for PC {i} of {k}...")
        mean, mean_gene, std, alpha, beta, delta = analytic_left_singular_vectors(evs[i], model, spike)
        mean_matrix[:,i] = mean
        mean_gene_matrix[:,i] = mean_gene
        std_matrix[:,i] = std
        alpha_matrix[:,i] = alpha
        beta_matrix[:,i] = beta
        delta_matrix[:,i] = delta

    # same p-scaling convention used everywhere else in this file before
    # calling into spiked_mixture_model2.py / spiked_mixture_model.py
    p = spike["v_list"][0].shape[0]
    lambda_list = model["covariance"].copy()
    lambda_list = [p*lambda_list[i] for i in range(len(lambda_list))]
    model_scaled = model.copy()
    model_scaled["covariance"] = lambda_list

    # covariance_matrix[a] is the k x k covariance of the leading eigenvectors
    # for mixture a; its row order matches model["mixture_labels"] (== nb
    # ["celltypes"]), which is also the order of the mixture blocks in
    # mean_matrix/std_matrix/sample_labels (make_spike_matrix lays out
    # sample_labels as contiguous blocks in exactly that order) and of
    # delta_matrix's rows.
    print("Computing covariance matrix...")
    covariance_matrix = smm2.compute_covariance_matrix(
        model_scaled, spike, evs, alpha_matrix, beta_matrix)

    print("Computing covariance matrix... done")

    U = sample_mixture_gaussian(mean_matrix, covariance_matrix, spike["sample_labels"],
                                 model["mixture_labels"])

    return {"vectors": U, "values": np.sqrt(evs),
            "mean_matrix": mean_matrix, "std_matrix": std_matrix,
            "mean_gene_matrix": mean_gene_matrix,
            "sample_labels": spike["sample_labels"],
            "alpha_matrix": alpha_matrix, "beta_matrix": beta_matrix,
            "delta_matrix": delta_matrix, "covariance_matrix": covariance_matrix}


# Returns the mean vector and covariance matrix (in PC-score units) for the
# rows of pca whose sample_labels match the given label. mean_matrix is
# constant within each mixture, so the first matching row is used.
# covariance_matrix is indexed by mixture (not by cell); its row order
# matches the order mixture blocks first appear in sample_labels (see
# log_normalized_PCA), so that order is reconstructed here to find the
# right slice. The covariance is scaled by the singular values (the same
# per-coordinate scaling mean_matrix gets) and divided by the number of
# cells in this mixture to convert it to a per-cell scale.
def get_GMM(pca, label):
    values = np.asarray(pca["values"])
    mean_matrix = np.asarray(pca["mean_matrix"]) * values
    covariance_matrix = np.asarray(pca["covariance_matrix"])
    sample_labels = np.asarray(pca["sample_labels"])

    idx = np.where(sample_labels == label)[0]
    if len(idx) == 0:
        raise ValueError(f"label '{label}' not found in pca['sample_labels']")
    row = idx[0]
    n_cells = len(idx)

    unique_labels, first_idx = np.unique(sample_labels, return_index=True)
    labels_in_order = unique_labels[np.argsort(first_idx)]
    mixture_idx = np.where(labels_in_order == label)[0][0]

    mean = mean_matrix[row, :]
    covariance = covariance_matrix[mixture_idx] * np.outer(values, values) / n_cells
    covariance = covariance.real

    return {"mean": mean, "covariance": covariance}


# Plots one panel per coordinate (3 columns, as many rows as needed), each
# showing the mean-4*std to mean+4*std density of every gmm in gmm_list for
# that coordinate, with a shared legend keyed by label_list.
def plot_GMM(gmm_list, label_list, n_grid=500):
    means = [np.asarray(gmm["mean"]) for gmm in gmm_list]
    stds = [np.asarray(gmm["std"]) for gmm in gmm_list]

    k = min(len(mean) for mean in means)
    if any(len(mean) != k for mean in means):
        print(f"plot_GMM: gmm_list entries have differing numbers of coordinates "
              f"{[len(mean) for mean in means]}; truncating to the first {k}")
    colors = plt.get_cmap("tab10")

    x_min = min((mean - 4 * std).min() for mean, std in zip(means, stds))
    x_max = max((mean + 4 * std).max() for mean, std in zip(means, stds))
    x_grid = np.linspace(x_min, x_max, n_grid)

    n_cols = 3
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for i in range(k):
        ax = axes.flat[i]

        for j, (mean, std, label) in enumerate(zip(means, stds, label_list)):
            color = colors(j % 10)
            ax.plot(x_grid, norm.pdf(x_grid, mean[i], std[i]), color=color, label=label)

        ax.set_xlim(x_min, x_max)
        ax.set_title(f"PC {i + 1}")
        ax.set_xlabel("PC score")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)

    for ax in axes.flat[k:]:
        ax.axis("off")

    fig.tight_layout()
    return fig


# Collapse a per-cell PCA (as returned by log_normalized_PCA) to one row per
# mixture. mean_matrix and std_matrix are required to be constant within each
# mixture (otherwise a ValueError is raised); for vectors, which carry per-row
# random noise, a single row is drawn at random from each mixture's block.
def pca_to_mixture_pca(pca):
    sample_labels = np.asarray(pca["sample_labels"])
    vectors = np.asarray(pca["vectors"])
    mean_matrix = np.asarray(pca["mean_matrix"])
    std_matrix = np.asarray(pca["std_matrix"])

    # unique mixtures in first-appearance order (rows are contiguous per mixture)
    _, first_idx = np.unique(sample_labels, return_index=True)
    mixtures = sample_labels[np.sort(first_idx)]

    new_vectors = np.zeros((len(mixtures), vectors.shape[1]), dtype=vectors.dtype)
    new_mean = np.zeros((len(mixtures), mean_matrix.shape[1]), dtype=mean_matrix.dtype)
    new_std = np.zeros((len(mixtures), std_matrix.shape[1]), dtype=std_matrix.dtype)

    for i, mix in enumerate(mixtures):
        rows = np.where(sample_labels == mix)[0]

        for name, M in (("mean_matrix", mean_matrix), ("std_matrix", std_matrix)):
            block = M[rows]
            if not np.allclose(block, block[0]):
                raise ValueError(
                    f"{name} is not constant within mixture '{mix}'")

        rep = np.random.choice(rows)
        new_vectors[i] = vectors[rep]
        new_mean[i] = mean_matrix[rep]
        new_std[i] = std_matrix[rep]

    new_pca = dict(pca)
    new_pca["vectors"] = new_vectors
    new_pca["mean_matrix"] = new_mean
    new_pca["std_matrix"] = new_std
    new_pca["sample_labels"] = mixtures
    new_pca["values"] = pca["values"]
    return new_pca


# Given a negative binomial mixture model, computes the spike and noise covariance structure
# that results from the log-normalization workflow
def log_normalized_spiked_mixture_model(nb, n, L0=10000, include_spike=True):
    # nb["L"] is per metacell (one entry per row of mu_matrix), so each
    # mixture's log moments are evaluated at its own library size
    L_vec = np.asarray(nb["L"], dtype=float).ravel()
    if len(L_vec) != nb["mu_matrix"].shape[0]:
        raise ValueError(f"nb['L'] must have one entry per metacell: got {len(L_vec)} for "
                         f"{nb['mu_matrix'].shape[0]} rows of mu_matrix")
    Lmu, theta = L_vec[:, np.newaxis] * nb["mu_matrix"], nb["theta_matrix"]
    alphas = nb["pop_alphas"]

    #Y = np.log(1 + X * (L0 / Li[:, np.newaxis]))
    Y_mean, Y_variance = compute_NB_log_moments(Lmu, theta, L_vec, L0)
    Y_total_mean = np.zeros(Y_mean.shape[1])
    for i in range(len(alphas)):
        Y_total_mean += alphas[i] * Y_mean[i]

    Y_mean_var = np.zeros(Y_mean.shape[1])
    Y_variance_ave = np.zeros(Y_variance.shape[1])
    for i in range(len(alphas)):
        Y_mean_var += alphas[i] * (Y_mean[i] - Y_total_mean)**2
        Y_variance_ave += alphas[i] * Y_variance[i]

    normalizer = Y_mean_var + Y_variance_ave + 1e-10
    spike_mixtures = (Y_mean - Y_total_mean) / np.sqrt(normalizer) / np.sqrt(n)

    p = Y_mean.shape[1]
    lambda_list = [Y_variance[i]/normalizer/n for i in range(len(alphas))]

    m_model = {"covariance": lambda_list, "mixture_frequencies": alphas, "gamma": p/n,
               "mixture_labels": nb["celltypes"]}

    if include_spike:
        spike = make_spike_matrix(spike_mixtures, alphas, n, nb["celltypes"])
    else:
        spike = None

    return m_model, spike

def simulate_PCA_embedding(pca_theory, cell_labels):
    """Draw a fresh embedding from the analytic per-mixture GMM (get_GMM's
    mean/covariance, in U*sv units), with rows in cell_labels' order rather
    than pca_theory's own sample_labels order -- so the result is
    row-comparable to the other specs in an analysis_spike pkl, which are
    all realigned to cell_labels the same way (see anndata2spike).

    Returns a spec dict {"U", "sv", "V"}: the drawn (n, k) matrix is
    column-normalized to get U, with the column norms as sv, so U*sv
    reconstructs the draw exactly. V is None -- a GMM draw has no gene-space
    basis behind it, so this spec can be consumed anywhere a spec's U*sv
    scores are used but cannot serve as a projection basis (the same status
    the analysis_spike *_proj_* specs have).
    """
    cell_labels = np.asarray(cell_labels)
    sample_labels = np.asarray(pca_theory["sample_labels"])
    k = len(np.asarray(pca_theory["values"]))

    mixture_labels, first_idx = np.unique(sample_labels, return_index=True)
    mixture_labels = mixture_labels[np.argsort(first_idx)]

    missing = np.setdiff1d(np.unique(cell_labels), mixture_labels)
    if len(missing):
        raise ValueError(f"cell_labels contains labels not in pca_theory: {sorted(missing)}")

    E = np.zeros((len(cell_labels), k))
    for label in mixture_labels:
        idx = np.where(cell_labels == label)[0]
        if len(idx) == 0:
            continue
        gmm = get_GMM(pca_theory, label)
        E[idx, :] = np.random.multivariate_normal(gmm["mean"], gmm["covariance"], size=len(idx))

    sv = np.linalg.norm(E, axis=0)
    return {"U": E / sv, "sv": sv, "V": None}


def simulate_Z_guassian(m_model, spike):
    celltypes = m_model["mixture_labels"]
    lambda_list = m_model["covariance"]
    u_list, s, v_list = spike["u_list"], spike["s"], spike["v_list"]
    n = u_list[0].shape[0]
    p = v_list[0].shape[0]

    sample_labels = spike["sample_labels"]
    spike_matrix = spike_svd2_matrix(spike)

    Z = np.zeros((n, p))
    for i in range(n):
        idx = np.where(sample_labels[i] == celltypes)[0][0]
        mean = spike_matrix[i]
        var = lambda_list[idx]
        Z[i, :] = np.random.normal(loc=mean, scale=np.sqrt(var), size=p)
    return Z

def simulate_noise_gaussian(m_model, mixture_labels):
    celltypes = np.asarray(m_model["mixture_labels"])
    lambda_list = m_model["covariance"]
    mixture_labels = np.asarray(mixture_labels)
    n = len(mixture_labels)
    p = lambda_list[0].shape[0]

    Z = np.zeros((n, p))
    for i, label in enumerate(mixture_labels):
        idx = np.where(label == celltypes)[0]
        if len(idx) != 1:
            raise ValueError(f"mixture label '{label}' does not match exactly one mixture")
        var = lambda_list[idx[0]]
        Z[i, :] = np.random.normal(loc=0, scale=np.sqrt(var), size=p)
    return Z


def compute_snr(m_model, mixture_labels):
    celltypes = np.asarray(m_model["mixture_labels"])
    lambda_list = m_model["covariance"]
    mixture_labels = np.asarray(mixture_labels)
    p = lambda_list[0].shape[0]

    snr = np.zeros(p)
    for label, var in zip(celltypes, lambda_list):
        n_label = np.sum(mixture_labels == label)
        snr += n_label * var

    return 1 - snr

def spike_svd2_matrix(spike):
    u_list, s, v_list = spike["u_list"], spike["s"], spike["v_list"]
    n = u_list[0].shape[0]
    p = v_list[0].shape[0]
    spike_matrix = np.zeros((n, p))
    for i in range(len(s)):
        spike_matrix += s[i] * np.outer(u_list[i], v_list[i])
    return spike_matrix


def create_ncells_per_celltype(n, pop_alphas):
    ncells_per_celltype = np.round(n * pop_alphas).astype(int)

    # Adjust rounded counts so they sum exactly to n.
    diff = n - np.sum(ncells_per_celltype)
    if diff > 0:
        idx = np.argmax(ncells_per_celltype)
        ncells_per_celltype[idx] += diff
    elif diff < 0:
        idx = np.argmax(ncells_per_celltype)
        ncells_per_celltype[idx] += diff

    if np.any(ncells_per_celltype < 1):
        raise ValueError(f"ncells_per_celltype must be at least 1, got {ncells_per_celltype}")
    if n != np.sum(ncells_per_celltype):
        raise ValueError(
            f"n must be equal to the sum of ncells_per_celltype, "
            f"got {n} and {np.sum(ncells_per_celltype)}"
        )

    return ncells_per_celltype


def make_spike_matrix(spike_mixtures, pop_alphas, n, mixture_labels):
    p = spike_mixtures.shape[1]
    spike_matrix = np.zeros((n, p))
    sample_labels = np.empty(n, dtype=np.array(mixture_labels).dtype)
    ncells_per_celltype = create_ncells_per_celltype(n, pop_alphas)
    start = 0
    for i in range(len(pop_alphas)):
        n_i = int(ncells_per_celltype[i])
        spike_matrix[start:start + n_i, :] = spike_mixtures[i]
        sample_labels[start:start + n_i] = mixture_labels[i]
        start += n_i
    U, s, V = np.linalg.svd(spike_matrix, full_matrices=False)
    k = np.sum(s > 1e-8)
    U = U[:, :k]
    s = s[:k]
    V = V[:k, :].T
    
    u_list = [U[:, i] for i in range(k)]
    v_list = [V[:, i] for i in range(k)]

    return {"u_list": u_list, "s": s, "v_list": v_list, "sample_labels": sample_labels,
            "spike_mixtures": spike_mixtures, "mixture_labels": mixture_labels}

def cut_spike(spike, k_new):
    k = len(spike["s"])
    if k_new > k:
        raise ValueError(f"k_new must be less than or equal to k, got {k_new} and {k}")
    u_list = spike["u_list"][:k_new]
    s = spike["s"][:k_new]
    v_list = spike["v_list"][:k_new]
    return {"u_list": u_list, "s": s, "v_list": v_list, "sample_labels": spike["sample_labels"],
            "spike_mixtures": spike["spike_mixtures"], "mixture_labels": spike["mixture_labels"]}

def compute_b_plus(m_model):
    lambda_list = m_model["covariance"]
    p = lambda_list[0].shape[0]
    lambda_list = [p*lambda_list[i] for i in range(len(lambda_list))]
   
    m_model_scaled = m_model.copy()
    m_model_scaled["covariance"] = lambda_list

    b_plus = mm.compute_b_plus(m_model_scaled, accuracy=0.1, debug=True)
    return b_plus

def analytic_spectral_density(m_model, x_grid, n, p):
    lambda_list, pop_alphas = m_model["covariance"], m_model["mixture_frequencies"]
    gamma = p / n
    lambda_list = [p*lambda_list[i] for i in range(len(lambda_list))]
   
    m_model_scaled = m_model.copy()
    m_model_scaled["covariance"] = lambda_list

    params = mm.SolveParams(warm_start=True)
    _, _, f, _, _ = mm.mixandmix_density(m_model_scaled, x_grid, params)
    return f

def make_E_matrix(sample_labels, celltypes):
    r = len(celltypes)  
    n = len(sample_labels)
    E = np.zeros((n, r))
    for i in range(r):
        idx = np.where(sample_labels == celltypes[i])[0]
        E[idx, i] = 1/np.sqrt(len(idx))
    return E

def analytic_eigenvalues(m_model, spike):
    lambda_list, pop_alpha = m_model["covariance"], m_model["mixture_frequencies"]
    u_list, s, v_list = spike["u_list"], spike["s"], spike["v_list"]

    p = v_list[0].shape[0]
    n = u_list[0].shape[0]
    gamma = p / n
    lambda_list = [p*lambda_list[i] for i in range(len(lambda_list))]

    m_model_scaled = m_model.copy()
    m_model_scaled["covariance"] = lambda_list

    evs = smm.compute_spike_eigenvalues(m_model_scaled, spike, isotropic=False)

    return evs

# eig_values is determined by the analytic_singular_values function
def analytic_left_singular_vectors(eig_value, m_model, spike):
    lambda_list, pop_alphas = m_model["covariance"], m_model["mixture_frequencies"]
    u_list, s, v_list = spike["u_list"], spike["s"], spike["v_list"]
    celltypes = m_model["mixture_labels"]
    sample_labels = spike["sample_labels"]
    
    z = np.sqrt(eig_value)
    p = v_list[0].shape[0]
    n = u_list[0].shape[0]
    gamma = p / n
    lambda_list = [p*lambda_list[i] for i in range(len(lambda_list))]
    m_model_scaled = m_model.copy()
    m_model_scaled["covariance"] = lambda_list
    d = smm.compute_spike_projection_coef(m_model_scaled, spike, z, isotropic=False)

    
    U_spike = np.array(u_list).T
    V_spike = np.array(v_list).T
    E = make_E_matrix(sample_labels, celltypes)

    alpha = d['alpha']
    beta = d['beta']
    delta = d['delta']

    mean = U_spike @ alpha
    mean_gene = V_spike @ beta

    std = E @ delta
    return mean, mean_gene, std, alpha, beta, delta
