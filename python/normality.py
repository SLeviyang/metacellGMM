import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


DEFAULT_N_BINS = 8
DEFAULT_N_REP = 2000
DEFAULT_N_MIN = 10

# This script computes a null for the normality score of a metacell.
# The null is used to compute a z-score for the metacell, which
# standardizes the degree of normality across metacells with different
# number of cells and different number of PCs.

# The main external functions are:
# - compute_normality_null: fit a null for a given k over a given range of n
# - normality_z: compute the z-score of a metacell's normality against a null
# - visualize_normality_null: visualize the null

########################################################
#  The basic normality statistic : Kolmogorov-Smirnov statistic
#  The metacell normality score is the max over the PC eigenvectors

def max_ecdf_normal_diff(x, mean, std):
    """Two-sided Kolmogorov-Smirnov statistic between the empirical CDF of
    `x`, standardized to (x - mean) / std, and the standard Normal(0, 1) CDF:
    the max absolute gap between the two, checked on both sides of each
    empirical CDF step.
    """
    z_sorted = np.sort((x - mean) / std)
    n = len(z_sorted)
    normal_cdf = norm.cdf(z_sorted, loc=0, scale=1)
    ecdf_upper = np.arange(1, n + 1) / n
    ecdf_lower = np.arange(0, n) / n
    return max(np.max(np.abs(ecdf_upper - normal_cdf)), np.max(np.abs(ecdf_lower - normal_cdf)))


def metacell_eigenvectors(X):
    """Eigenvectors of this metacell's own PC-score covariance, as columns,
    ordered by descending eigenvalue.
    """
    eigvals, eigvecs = np.linalg.eigh(np.cov(X, rowvar=False))
    return eigvecs[:, np.argsort(eigvals)[::-1]]


def metacell_normality(X, return_per_axis=False):
    """Normality score of one metacell: the max over its OWN covariance
    eigenvector directions of max_ecdf_normal_diff (a Kolmogorov-Smirnov
    statistic against a normal fitted to that same projection).

    X is that metacell's PC scores, spec["U"][mask, :] * spec["sv"].

    The projection basis is the metacell's own covariance eigenvectors, not
    the raw PC axes. Those raw axes belong to the GLOBAL PCA and are
    arbitrary relative to any one metacell, whereas the eigenvectors are
    intrinsic to it; they also diagonalize the covariance, so the k
    projections are uncorrelated rather than redundantly correlated. This
    additionally puts the score in the same basis the splitter searches
    over, which the raw-axis version did not (empirically the two bases were
    near-orthogonal).

    Two ambiguities are harmless here: the score is self-referential
    (mean/sd fit from the same sample), so it is invariant to the per-axis
    sign ambiguity of each method's independent SVD; and the two-sided KS
    statistic is itself sign-invariant, since max_ecdf_normal_diff checks
    both sides of every ECDF step, so eigenvector sign does not matter
    either.

    return_per_axis returns the whole per-eigenvector array instead of just
    its max.
    """
    proj = X @ metacell_eigenvectors(X)
    per_axis = np.array([max_ecdf_normal_diff(proj[:, j], proj[:, j].mean(), proj[:, j].std())
                         for j in range(proj.shape[1])])
    return per_axis if return_per_axis else float(per_axis.max())

########################################################


def _collect_simulated_null(n_vals, k, n_rep=1, seed=0):
    """Simulate the null distribution of the normality score at a FIXED k,
    once per size in n_vals (n_rep times each).

    Each draw is standard normal (n, k) data scored with metacell_normality.
    Standard normal is a valid stand-in for any covariance because the score
    is rotation- and scale-invariant: for orthogonal Q the eigenvectors of
    cov(XQ) are Q.T times those of cov(X), so the projections are unchanged,
    and each projection is standardized by its own sample mean/sd before the
    KS statistic. Only the SHAPE of the eigenvalue spectrum could matter,
    and an isotropic spectrum is the natural reference.

    Sizes with n <= k are skipped: the sample covariance is then rank
    deficient, so the trailing eigenvectors are arbitrary and their
    projections have ~zero spread, which makes the standardization in
    max_ecdf_normal_diff meaningless.

    Returns (n_out, values), parallel arrays of length <= len(n_vals)*n_rep.
    """
    rng = np.random.default_rng(seed)
    n_out, values = [], []
    for n in np.atleast_1d(n_vals):
        n = int(n)
        if n <= k:
            continue
        for _ in range(n_rep):
            n_out.append(n)
            values.append(metacell_normality(rng.standard_normal((n, k))))
    if not n_out:
        raise ValueError(f"no usable sizes in n_vals for k={k} (all n <= k)")
    return np.array(n_out, dtype=float), np.array(values, dtype=float)


def _fit_power_law(n_cells, values, n_bins=DEFAULT_N_BINS):
    """mu(n) and sigma(n) as power laws in n.

    mu:    log(value) regressed on log(n), so mu(n) = exp(a) * n**b. A KS
           statistic scales as ~1/sqrt(n), so b is expected near -0.5.

    sigma: n is split into n_bins equal-count bins in log(n); the sd of the
           values within each bin is computed, and log(sd) is regressed on
           log(n) to give sigma(n) = exp(c) * n**d. Estimating sigma from
           binned spread (rather than as a fixed multiple of mu) lets
           sigma/mu vary with n, which it empirically does.
    """
    log_n = np.log(n_cells)
    b, a = np.polyfit(log_n, np.log(values), 1)

    pred = a + b * log_n
    ss_res = np.sum((np.log(values) - pred) ** 2)
    ss_tot = np.sum((np.log(values) - np.log(values).mean()) ** 2)
    r2_mu = 1.0 - ss_res / ss_tot

    edges = np.quantile(log_n, np.linspace(0.0, 1.0, n_bins + 1))
    centers, sds = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (log_n >= lo) & (log_n <= hi) if i == n_bins - 1 else (log_n >= lo) & (log_n < hi)
        if in_bin.sum() >= 3:
            sd = values[in_bin].std(ddof=1)
            if sd > 0:
                centers.append(log_n[in_bin].mean())
                sds.append(sd)
    if len(centers) < 2:
        raise ValueError(f"only {len(centers)} usable bins for the sigma fit; "
                         "reduce n_bins or supply more datasets")

    centers, sds = np.array(centers), np.array(sds)
    d, c = np.polyfit(centers, np.log(sds), 1)

    pred_s = c + d * centers
    ss_res_s = np.sum((np.log(sds) - pred_s) ** 2)
    ss_tot_s = np.sum((np.log(sds) - np.log(sds).mean()) ** 2)
    r2_sigma = 1.0 - ss_res_s / ss_tot_s if ss_tot_s > 0 else np.nan

    def mu(n):
        n = np.asarray(n, dtype=float)
        out = np.exp(a) * n ** b
        return float(out) if out.ndim == 0 else out

    def sigma(n):
        n = np.asarray(n, dtype=float)
        out = np.exp(c) * n ** d
        return float(out) if out.ndim == 0 else out

    params = {"mu_log_intercept": float(a), "mu_log_slope": float(b),
              "sigma_log_intercept": float(c), "sigma_log_slope": float(d),
              "r2_mu_loglog": float(r2_mu), "r2_sigma_loglog": float(r2_sigma),
              "n_bins_used": len(centers)}
    return mu, sigma, params


def _simulation_sizes(n_vals, n_rep=DEFAULT_N_REP, n_min=DEFAULT_N_MIN):
    """Geometric grid of max(n_rep, len(n_vals)) sizes spanning n_min to
    max(n_vals).

    Only the UPPER end comes from n_vals. The lower end is always n_min,
    regardless of how small the supplied sizes go: refinement keeps
    splitting and produces children far below any originally observed
    metacell, and a z-score for an n below the fitted range would be an
    extrapolation. Simulating down there costs nothing because it is
    simulation rather than data.

    Simulating across a dense grid of DISTINCT sizes, rather than repeating
    draws at the supplied sizes, is what pins down a smooth fit: it
    constrains the curve over the whole range instead of stacking replicates
    at a handful of points.

    Geometric spacing matches the power-law form (uniform in log n).
    Rounding to integers naturally repeats the smallest sizes, which is
    where the score is noisiest and replication is most useful.
    """
    n_vals = np.asarray(n_vals, dtype=float)
    if len(n_vals) == 0:
        return n_vals
    hi = float(n_vals.max())
    lo = float(min(n_min, hi))
    return np.geomspace(lo, hi, max(int(n_rep), len(n_vals), 2))


def compute_normality_null(k, n_vals, n_rep=DEFAULT_N_REP, n_bins=DEFAULT_N_BINS, seed=0):
    """Fit mu(n) and sigma(n) for the metacell normality score at a FIXED k.

    The score depends strongly on both n and k: a KS statistic scales as
    ~1/sqrt(n) even for perfectly Gaussian data, and the score is a max over
    k such statistics, so more axes means a larger max. A null is therefore
    only valid for the k it was fitted at -- pooling across k leaves a
    systematic residual (measured at spearman +0.21 against k when pooled).

    The null is SIMULATED: draws of standard normal (n, k) data scored with
    metacell_normality. That gives arbitrarily many observations per size,
    and is valid because the score is rotation- and scale-invariant (see
    _collect_simulated_null).

    n_vals supplies only the UPPER end of the size range -- the grid runs from
    DEFAULT_N_MIN up to max(n_vals), with max(n_rep, len(n_vals)) points
    (see _simulation_sizes). n_rep is therefore a grid DENSITY, not a
    repetition count: sizes are distinct, and replication only arises where
    integer rounding collapses neighbouring grid points, which happens at
    the small-n end where the score is noisiest.

    Both mu and sigma are fitted as power laws, exp(intercept) * n**slope
    (see _fit_power_law). That form is not an approximation of convenience:
    a KS statistic scales as ~1/sqrt(n), and the fitted mu slopes land near
    -0.49, with log-log R^2 ~0.98.

    Returns a dict with:
      k              -- the k this null is valid for
      mu, sigma      -- callables, accept scalar or array n
      params         -- fit coefficients
      n_cells, values -- the simulated null observations
      z_median_on_null, z_sd_on_null -- calibration diagnostics. The median
                        sits near 0 as it should. The sd runs ~0.75-0.79
                        rather than 1.0 because sigma is estimated from
                        binned spread of a right-skewed residual
                        distribution, so it overstates the typical
                        deviation -- a nominal z cutoff is therefore
                        stricter than the same number of standard units.
    """
    sizes = _simulation_sizes(n_vals, n_rep=n_rep)
    if len(sizes) == 0:
        raise ValueError("n_vals is empty")

    n_cells, values = _collect_simulated_null(sizes, k, n_rep=1, seed=seed)
    mu, sigma, params = _fit_power_law(n_cells, values, n_bins=n_bins)

    z_null = (values - mu(n_cells)) / sigma(n_cells)

    return {"k": int(k), "mu": mu, "sigma": sigma, "params": params,
            "n_cells": n_cells, "values": values,
            "n_obs": len(n_cells), "n_rep": int(n_rep),
            "z_median_on_null": float(np.median(z_null)),
            "z_sd_on_null": float(z_null.std())}


def normality_z(X, null):
    """z-score of one metacell's normality against a fitted null (the dict
    returned by compute_normality_null).

    The null is k-specific, so X's number of axes must match the k it was
    fitted for -- using a mismatched null would silently mis-scale the
    score.
    """
    n, k = X.shape
    if k != null["k"]:
        raise ValueError(f"null was fitted for k={null['k']} but X has k={k}; "
                         "fit a null for this k (see compute_normality_null)")
    return float((metacell_normality(X) - null["mu"](n)) / null["sigma"](n))


def _draw_null_panel(ax, null, n_obs, values, label, n_sd, log_scale, grid_lo, grid_hi,
                     cmap_lim):
    """One panel: the fitted mu(n) curve and +/- sd bands, with the supplied
    observations scattered on top and colored by their own z-score.
    """
    mu_fn, sigma_fn = null["mu"], null["sigma"]

    grid = np.geomspace(grid_lo, grid_hi, 300)
    mu_grid = np.asarray(mu_fn(grid), dtype=float)
    sd_grid = np.asarray(sigma_fn(grid), dtype=float)

    shades = np.linspace(0.10, 0.26, len(n_sd))
    for m, shade in zip(sorted(n_sd, reverse=True), shades):
        ax.fill_between(grid, mu_grid - m * sd_grid, mu_grid + m * sd_grid,
                        color="black", alpha=shade, lw=0, label=rf"$\mu \pm {m}\sigma$")
    ax.plot(grid, mu_grid, color="black", lw=2.2, label=r"$\mu(n)$")

    sc = None
    if len(n_obs):
        z = (values - np.asarray(mu_fn(n_obs), dtype=float)) / \
            np.asarray(sigma_fn(n_obs), dtype=float)
        sc = ax.scatter(n_obs, values, c=z, cmap="coolwarm", vmin=-cmap_lim, vmax=cmap_lim,
                        s=30, alpha=0.8, edgecolors="none", zorder=3, label=label)
        stat = (f"n={len(n_obs)}\nz median={np.median(z):+.3f}\nz sd={z.std():.3f}")
    else:
        stat = "no observations\nat this k"
    ax.text(0.02, 0.02, stat, transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
           bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("n_cells (metacell)")
    ax.set_ylabel("normality score (max |ECDF - normal CDF|)")
    ax.legend(fontsize=8, loc="upper right")
    return sc


def visualize_normality_null(null, n_sd=(1, 2), log_scale=True, ax=None):
    """The fitted mu(n) curve with +/- sd bands, and the simulated draws the
    null was fitted to scattered on top, colored by their own z-score.

    Returns the matplotlib figure.
    """
    sim_n, sim_v = null["n_cells"], null["values"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    # the grid runs below the smallest simulated size because refinement
    # produces children there and the extrapolation behaviour matters
    grid_lo, grid_hi = sim_n.min() * 0.5, sim_n.max() * 1.1

    z = (sim_v - np.asarray(null["mu"](sim_n), dtype=float)) / \
        np.asarray(null["sigma"](sim_n), dtype=float)
    cmap_lim = float(np.abs(z).max())

    sc = _draw_null_panel(ax, null, sim_n, sim_v,
                          f"simulated draws (n_rep={null['n_rep']})",
                          n_sd, log_scale, grid_lo, grid_hi, cmap_lim)
    ax.set_title(f"simulated null, k={null['k']} ({null['n_obs']} draws)", fontsize=10)

    if sc is not None:
        fig.colorbar(sc, ax=ax, label="z-score of the observation")

    p = null["params"]
    subtitle = (rf"$\mu(n)={np.exp(p['mu_log_intercept']):.3f}\,n^{{{p['mu_log_slope']:.3f}}}$"
                rf" ($R^2$={p['r2_mu_loglog']:.3f}),  "
                rf"$\sigma(n)={np.exp(p['sigma_log_intercept']):.4f}\,n^{{{p['sigma_log_slope']:.3f}}}$"
                rf" ($R^2$={p['r2_sigma_loglog']:.3f})")

    fig.suptitle(f"Normality null for k={null['k']}\n{subtitle}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
