import os
import pickle
import numpy as np

import lognorm_workflow_analytic as lwa
import analysis_spike as asp
import nb_model
import datasets
import datasets_refined as dr

import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# bulk pkls are split by provenance the same way the HVG h5ads they are built
# from are: refined ones come from analysis_HVG_refined, unrefined from
# analysis_HVG
BULK_DIR = "../analysis_bulk/"
BULK_REFINED_DIR = "../analysis_bulk_refined/"


def _as_list(x):
    """Wrap a single value in a list, leaving an existing sequence alone.

    Lets callers pass either visualize_dataset_bulk("PBMC", 500) or
    visualize_dataset_bulk("PBMC", [500, 1000]). Strings are wrapped rather
    than iterated, so a bare dataset name does not silently loop over its
    characters.
    """
    if isinstance(x, str) or not hasattr(x, "__len__"):
        return [x]
    return x


def get_bulk_filename(dataset, n_genes, refined=False):
    """Path of the bulk pkl, creating the folder if it does not exist."""
    out_dir = BULK_REFINED_DIR if refined else BULK_DIR
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{dataset}_HVG_{n_genes}.pkl")


def anndata2bulk(dataset, n_genes, overwrite=False, refined=False):
    """Bulk (noise) eigenvalue spectrum for one (dataset, n_genes), cached
    as a pkl.

    Compares the empirical PCA eigenvalues below the signal components
    against three references: the analytic mixture spectral density, the
    Marchenko-Pastur law, and a simulated isotropic-noise spectrum. Also
    includes three further empirical spectra, evs_perm, evs_nb and evs_normal.

    evs_perm comes from data with each gene independently permuted within
    metacell -- the same permutation analysis_spike.permute_genes_within_celltype
    applies to build spec_perm, reused here rather than duplicated. This
    destroys within-metacell gene-gene correlations while preserving each
    gene's own marginal distribution, giving a bulk null with real per-gene
    noise structure but no real covariance signal, unlike evs_isotropic's
    fully synthetic Gaussian noise.

    evs_nb comes from counts drawn from the fitted NB model at the true
    metacell labels -- the same nb_model.simulate_X call analysis_spike uses
    for spec_nb, so the two agree by construction rather than by coincidence.
    It keeps the FITTED mean-variance relationship and has no gene-gene
    correlation at all, which places it between the other two nulls: evs_perm
    keeps each real gene's own marginal, evs_isotropic keeps no count
    structure whatever. Because it is simulated at the metacell labels it
    carries real between-metacell mean structure, so it has genuine spikes and
    is truncated at k alongside evs_pca and evs_perm; evs_isotropic is pure
    noise and is deliberately left whole. Comparing evs_nb against evs_mixture
    is the direct check on the analytic mixture density, since both are
    derived from the same nb object -- one by simulation, one in closed form.

    evs_normal is the Gaussian spiked-mixture model: the analytic spike matrix
    as each cell's mean plus per-mixture heteroscedastic Gaussian noise, the
    same lwa.simulate_Z_guassian draw analysis_spike uses for spec_n. It is the
    fully parametric end of the series -- evs_nb keeps discrete counts and
    their fitted mean-variance link, evs_normal replaces both with the Gaussian
    the analytic theory actually assumes -- so the gap between the two measures
    what the Gaussian approximation costs, and the gap between evs_normal and
    evs_mixture measures the analytic density's own error with the distribution
    held exactly right.

    Note evs_normal is NOT run through normalize/log1p/scale and is NOT divided
    by n: unlike evs_nb it is drawn directly in log-normalized standardized
    space, and the model's covariance already carries the 1/n (see
    lwa.log_normalized_spiked_mixture_model). Its cell-to-mixture assignment
    comes from the model's population frequencies rather than the observed
    metacell sizes, so it is not conditioned on quite the same thing as evs_nb.

    Returns a dict with keys x_grid, evs_mixture, evs_MP, evs_isotropic,
    evs_pca, evs_perm, evs_nb, evs_normal, b_plus and k -- named rather than a
    positional tuple, since same-typed arrays are easy to mis-order.
    """
    out_f = get_bulk_filename(dataset, n_genes, refined=refined)
    if os.path.isfile(out_f) and not overwrite:
        with open(out_f, "rb") as f:
            return pickle.load(f)

    s = (dr.load_HVG_refined_dataset(dataset, n_genes) if refined
         else datasets.load_HVG_dataset(dataset, n_genes))

    nb = s.uns["nb_model"]

    n, p = s.shape
    print("Computing analytical eigenvalues")
    # include_spike=True only ADDS the spike factorisation; m_model itself is
    # built identically either way, so b_plus and evs_mixture below are
    # unaffected. Taking the spike here lets evs_normal reuse this call rather
    # than fitting the same mixture model a second time.
    model, spike = lwa.log_normalized_spiked_mixture_model(nb, n, L0=1e4, include_spike=True)
    b_plus = lwa.compute_b_plus(model)
    x_grid = np.linspace(.1, b_plus + 0.5, 1000)
    evs_mixture = lwa.analytic_spectral_density(model, x_grid, n, p)

    print("Computing the Marchenko-Pastur density")
    gamma = p / n
    if gamma > 1:
        gamma = 1 / gamma
    a = (1 - np.sqrt(gamma)) ** 2
    b = (1 + np.sqrt(gamma)) ** 2
    evs_MP = np.sqrt(np.maximum(0.0, (b - x_grid) * (x_grid - a)))
    evs_MP = evs_MP / (2 * np.pi * gamma * np.maximum(x_grid, 1e-12))

    cell_labels = s.obs["metacell"].to_numpy()

    print("Computing Gaussian isotropic noise spectrum")
    snr = lwa.compute_snr(model, cell_labels)
    Z_isotropic = np.zeros((n, p))
    if Z_isotropic.shape[1] != p:
        raise ValueError(f"Z_isotropic.shape[1] != p: {Z_isotropic.shape[1]} != {p}")
    for i in range(p):
        Z_isotropic[:, i] = np.random.normal(loc=0, scale=np.sqrt((1-snr[i]) / n), size=n)
    evs_isotropic = np.linalg.svd(Z_isotropic, full_matrices=False)[1]**2
    del Z_isotropic

    print("Computing the PCA eigenvalues")
    s_new = s.copy()
    sc.pp.normalize_total(s_new, target_sum=1e4)
    sc.pp.log1p(s_new)
    sc.pp.scale(s_new)
    U, sv, V = np.linalg.svd(s_new.X, full_matrices=False)
    evs_pca = sv**2 / n
    del s_new, U, V

    print("Computing the permuted-gene PCA eigenvalues")
    X_permuted = asp.permute_genes_within_celltype(s.X.toarray(), cell_labels)
    s_perm = sc.AnnData(X_permuted)
    sc.pp.normalize_total(s_perm, target_sum=1e4)
    sc.pp.log1p(s_perm)
    sc.pp.scale(s_perm)
    U_perm, sv_perm, V_perm = np.linalg.svd(s_perm.X, full_matrices=False)
    evs_perm = sv_perm**2 / n
    del X_permuted, s_perm, U_perm, V_perm

    # simulated from the fitted NB model at the true metacell labels, each
    # metacell at its own library size nb["L"][i], then put through the SAME
    # normalize -> log1p -> scale pipeline as the true data above; comparing
    # the two spectra only means something if the transform applied to each
    # is identical. This is the call analysis_spike makes for spec_nb, so the
    # bulk and spike views of the NB model stay in step, and evs_mixture and
    # evs_normal are built at the same per-metacell L.
    print("Computing the NB-model PCA eigenvalues")
    X_nb = nb_model.simulate_X(nb, cell_labels)
    s_nb = sc.AnnData(X_nb)
    sc.pp.normalize_total(s_nb, target_sum=1e4)
    sc.pp.log1p(s_nb)
    sc.pp.scale(s_nb)
    U_nb, sv_nb, V_nb = np.linalg.svd(s_nb.X, full_matrices=False)
    evs_nb = sv_nb**2 / n
    del X_nb, s_nb, U_nb, V_nb

    # NOT put through normalize -> log1p -> scale, unlike evs_nb: the normal
    # model already lives in log-normalized standardized space, and log1p of a
    # signed Gaussian is undefined for half its entries. For the same reason
    # there is no /n here -- log_normalized_spiked_mixture_model builds its
    # covariance as Y_variance/normalizer/n, so the 1/n is already inside the
    # draw and svd(Z)**2 is directly comparable to the sv**2/n above. This is
    # the convention analysis_spike uses for spec_n.
    #
    # Row order is the model's own (spike["sample_labels"]) rather than the
    # h5ad's; analysis_spike realigns it because it keeps U, but singular
    # values are invariant under row permutation so the bulk does not need to.
    print("Computing the normal-model PCA eigenvalues")
    Z_normal = lwa.simulate_Z_guassian(model, spike)
    evs_normal = np.linalg.svd(Z_normal, full_matrices=False)[1]**2
    del Z_normal

    # the signal components are exactly those above the ANALYTIC bulk edge,
    # so k is counted against b_plus rather than read from obsm["X_pca"],
    # whose k comes from the elbow of the singular values and is a different
    # quantity. Dropping them leaves the bulk, which is what the reference
    # densities are compared against. SVD returns eigenvalues descending, so
    # the ones above the edge are a prefix.
    #k = int(np.sum(evs_pca > b_plus))
    # use a conservative k to eliminate eigenvalues near the bulk edge
    k = int(1.5*s.obsm["X_pca"].shape[1])
    evs_pca = evs_pca[k:]
    evs_perm = evs_perm[k:]
    # evs_nb is simulated at the metacell labels and evs_normal carries the
    # spike matrix as its per-cell mean, so both have real spikes of their own
    # and are truncated with the others. evs_isotropic is drawn at loc=0, has
    # no mean structure to spike on, and is left whole.
    evs_nb = evs_nb[k:]
    evs_normal = evs_normal[k:]

    result = {"x_grid": x_grid, "evs_mixture": evs_mixture, "evs_MP": evs_MP,
              "evs_isotropic": evs_isotropic, "evs_pca": evs_pca,
              "evs_perm": evs_perm, "evs_nb": evs_nb, "evs_normal": evs_normal,
              "b_plus": b_plus, "k": k, "snr": snr}
    with open(out_f, "wb") as f:
        pickle.dump(result, f)

    return result


def make_all_bulk_pkl(datasets_list=datasets.DEFAULT_DATASETS,
                      n_genes_list=datasets.DEFAULT_N_GENES,
                      overwrite=False, refined=True):
    """Build the bulk pkl for every (dataset, n_genes).

    A combination whose HVG h5ad has not been built is skipped with a
    warning rather than aborting the sweep.
    """
    for dataset in _as_list(datasets_list):
        for n_genes in _as_list(n_genes_list):
            print(f"Processing {dataset} HVG {n_genes}")
            try:
                anndata2bulk(dataset, n_genes, overwrite=overwrite, refined=refined)
            except ValueError as e:
                print(f"  SKIPPING {dataset}/{n_genes}: {e}")
    return None


## visualization functions ##

def visualize_bulk(bulk, ax=None, title=None, histogram_type="true"):
    """Plot one bulk spectrum: an empirical eigenvalue histogram with the
    three reference densities rescaled onto it.

    bulk is the dict returned by anndata2bulk. The reference curves are
    rescaled to the histogram's peak because only their SHAPE is being
    compared -- they are densities over different normalizations.

    histogram_type picks which empirical spectrum is histogrammed:
    "true" (default) uses bulk["evs_pca"] (the true data's own PCA
    eigenvalues), "perm" uses bulk["evs_perm"] (the gene-permuted-within-
    metacell null -- see anndata2bulk).
    """
    if histogram_type not in ("true", "perm"):
        raise ValueError(f'histogram_type must be "true" or "perm", got {histogram_type!r}')
    if ax is None:
        ax = plt.gca()

    evs_key = "evs_pca" if histogram_type == "true" else "evs_perm"
    hist_label = "PCA" if histogram_type == "true" else "permuted"
    counts, _, _ = ax.hist(bulk[evs_key], bins=100, alpha=0.5, label=hist_label)

    scale = counts.max() if counts.max() > 0 else 1.0

    def rescaled(curve):
        peak = curve.max()
        return curve * (scale / peak) if peak > 0 else curve

    x_grid = bulk["x_grid"]
    ax.plot(x_grid, rescaled(bulk["evs_mixture"]), label="mixture")
    ax.plot(x_grid, rescaled(bulk["evs_MP"]), label="MP")
    kde = gaussian_kde(bulk["evs_isotropic"])
    ax.plot(x_grid, rescaled(kde(x_grid)), label="isotropic")

    ax.set_xlim(0, bulk["b_plus"] + 1)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return ax


def available_n_genes(dataset, n_genes_list=datasets.DEFAULT_N_GENES, refined=False):
    """The n_genes in n_genes_list whose bulk pkl already exists on disk.

    Availability is judged on the PKL, not on the HVG h5ad it would be built
    from: anndata2bulk builds a missing pkl on demand, which takes minutes,
    so gating on the source would silently turn a plotting call into a long
    computation. Use make_all_bulk_pkl to build them deliberately.
    """
    found = []
    for n_genes in _as_list(n_genes_list):
        if os.path.isfile(get_bulk_filename(dataset, n_genes, refined=refined)):
            found.append(n_genes)
    return found


def visualize_full_dataset(dataset, n_genes_list=datasets.DEFAULT_N_GENES):
    """One dataset's bulk spectra, unrefined on the top row and refined on
    the bottom, one column per n_genes.

    Columns are the UNION of the gene counts available in either row, so the
    two rows stay aligned and a panel missing from one row is visibly empty
    rather than shifting the others along. An n_genes absent from both is
    dropped entirely.

    Purely a reader: an n_genes whose bulk pkl does not exist gets no panel,
    and nothing is computed. Build the missing ones with make_all_bulk_pkl
    first.
    """
    requested = list(_as_list(n_genes_list))
    rows = [(False, "unrefined"), (True, "refined")]
    have = {refined: available_n_genes(dataset, requested, refined=refined)
            for refined, _ in rows}

    columns = [ng for ng in requested if any(ng in have[r] for r, _ in rows)]
    if not columns:
        raise ValueError(
            f"no bulk pkl found for {dataset} among n_genes={requested}; "
            "run make_all_bulk_pkl to build them")

    for refined, tag in rows:
        missing = [ng for ng in columns if ng not in have[refined]]
        if missing:
            print(f"  no {tag} bulk pkl for {dataset} n_genes={missing} -- leaving blank")

    fig, axes = plt.subplots(len(rows), len(columns),
                             figsize=(5 * len(columns), 4 * len(rows)), squeeze=False)

    for i, (refined, tag) in enumerate(rows):
        for j, n_genes in enumerate(columns):
            ax = axes[i, j]
            if n_genes not in have[refined]:
                ax.axis("off")
                ax.set_title(f"{n_genes} genes\n({tag}: not built)", fontsize=9, color="grey")
                continue
            print(f"Visualizing {dataset} {tag} with {n_genes} genes")
            bulk = anndata2bulk(dataset, n_genes, refined=refined)
            visualize_bulk(bulk, ax=ax,
                           title=f"{tag} - {n_genes} genes  (k={bulk['k']})")

    fig.suptitle(f"{dataset} bulk spectrum -- unrefined (top) vs refined (bottom)")
    fig.tight_layout()
    plt.show()
    return fig


def visualize_dataset_bulk(dataset, n_genes_list=datasets.DEFAULT_N_GENES,
                           refined=True, overwrite=False, histogram_type="true"):
    """One panel per n_genes for a dataset, read from the cached bulk pkls
    (built on demand if absent).

    histogram_type is passed through to visualize_bulk: "true" (default)
    histograms each panel's evs_pca, "perm" histograms evs_perm.
    """
    n_genes_list = _as_list(n_genes_list)

    fig, axes = plt.subplots(1, len(n_genes_list), figsize=(5 * len(n_genes_list), 4))
    if len(n_genes_list) == 1:
        axes = [axes]

    for ax, ng in zip(axes, n_genes_list):
        print(f"Visualizing {dataset} with {ng} genes")
        bulk = anndata2bulk(dataset, ng, overwrite=overwrite, refined=refined)
        visualize_bulk(bulk, ax=ax, title=f"{dataset} - {ng} genes", histogram_type=histogram_type)

    fig.suptitle(f"{dataset} bulk spectrum" + (" (refined)" if refined else ""))
    fig.tight_layout()
    plt.show()
    return fig
