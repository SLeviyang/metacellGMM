"""Bulk (noise) eigenvalue spectra for the cell-type-as-metacell HVG
datasets (datasets_celltype.py), cached as pkls under
../analysis_bulk_celltype/.

This is to analysis_bulk.py what analysis_spike_celltype.py is to
analysis_spike.py: a self-contained copy with the loader and the output
folder swapped, so that nothing in the existing modules needs to change.
anndata2bulk_celltype is analysis_bulk.anndata2bulk verbatim apart from

  * the h5ad coming from datasets_celltype.load_HVG_celltype_dataset rather
    than the refined/unrefined branch, and
  * the within-metacell permutation coming from
    analysis_spike_celltype.permute_genes_within_celltype (identical to the
    analysis_spike one, but it keeps this module's imports on the cell-type
    side).

Because obs["metacell"] IS the cell type in those h5ads, "within metacell"
below means within cell type: the permutation null and the NB simulation
both condition on the cell-type labels, and the analytic mixture has one
component per cell type (9-70) rather than per SuperCell metacell (80-500).

The result dict carries exactly the keys analysis_bulk.anndata2bulk
produces, so analysis_bulk.visualize_bulk and the figure code that reads
those dicts can consume it unchanged; make_bulk_figure_celltype below is
figures_util.make_bulk_figure with the one loading call replaced.

Build everything with make_all_bulk_pkl_celltype(); the pkls are not built
by make.py.
"""

import os
import pickle
import numpy as np

import scanpy as sc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import lognorm_workflow_analytic as lwa
import analysis_bulk as ab
import analysis_spike_celltype as asc
import nb_model
import datasets
import datasets_celltype


# cell-type bulk pkls are built from ../analysis_HVG_celltype/ (see
# datasets_celltype.py) rather than ../analysis_HVG/ or
# ../analysis_HVG_refined/ -- there is no refined/unrefined split here, since
# a cell-type metacell is not something refinement operates on
BULK_CELLTYPE_DIR = "../analysis_bulk_celltype/"

# only the 1000-gene cell-type h5ads exist (and none for Thymus_large), so
# the default sweep is narrower than datasets.DEFAULT_N_GENES; a wider list
# would only print skip warnings for the missing combinations
DEFAULT_N_GENES_CELLTYPE = [1000]

# the empirical spectra make_bulk_figure_celltype draws, one row each; the
# keys are anndata2bulk_celltype's own names (evs_pca is the true data's PCA
# spectrum). Same as figures_util.BULK_SPECTRA.
BULK_SPECTRA = [("NB", "evs_nb"), ("true", "evs_pca")]


def _as_list(x):
    """Wrap a single value in a list, leaving an existing sequence alone.

    Strings are wrapped rather than iterated, so a bare dataset name does not
    silently loop over its characters. Same as analysis_bulk._as_list.
    """
    if isinstance(x, str) or not hasattr(x, "__len__"):
        return [x]
    return x


def get_bulk_celltype_filename(dataset, n_genes):
    """Path of the cell-type bulk pkl, creating the folder if it does not
    exist. Same filename pattern as analysis_bulk.get_bulk_filename, so the
    folder is a drop-in sibling of analysis_bulk_refined."""
    os.makedirs(BULK_CELLTYPE_DIR, exist_ok=True)
    return os.path.join(BULK_CELLTYPE_DIR, f"{dataset}_HVG_{n_genes}.pkl")


def has_bulk_celltype_result(dataset, n_genes):
    """True when the pkl is on disk."""
    return os.path.isfile(get_bulk_celltype_filename(dataset, n_genes))


def load_bulk_celltype_result(dataset, n_genes):
    """The cached bulk dict, read only -- never built.

    anndata2bulk_celltype builds a missing pkl on demand, which takes
    minutes, so a plotting call that went through it would silently turn
    into a long computation. Use make_all_bulk_pkl_celltype to build them
    deliberately, and this to read them.
    """
    out_f = get_bulk_celltype_filename(dataset, n_genes)
    if not os.path.isfile(out_f):
        raise FileNotFoundError(
            f"{out_f} not found; build it with "
            f"anndata2bulk_celltype('{dataset}', {n_genes})")
    with open(out_f, "rb") as f:
        return pickle.load(f)


def anndata2bulk_celltype(dataset, n_genes, overwrite=False):
    """Bulk (noise) eigenvalue spectrum for one (dataset, n_genes) of the
    cell-type-as-metacell HVG datasets, cached as a pkl.

    A copy of analysis_bulk.anndata2bulk; see its docstring for what each
    of the spectra is and how they relate. The only differences are the
    source h5ad (datasets_celltype) and the output folder, and hence that
    every "metacell" here is exactly one cell type.

    Returns a dict with keys x_grid, evs_mixture, evs_MP, evs_isotropic,
    evs_pca, evs_perm, evs_nb, evs_normal, b_plus, k and snr -- the same
    keys as analysis_bulk.anndata2bulk.
    """
    out_f = get_bulk_celltype_filename(dataset, n_genes)
    if os.path.isfile(out_f) and not overwrite:
        with open(out_f, "rb") as f:
            return pickle.load(f)

    s = datasets_celltype.load_HVG_celltype_dataset(dataset, n_genes)

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

    # the h5ad's metacell column, which datasets_celltype sets to the cell
    # type itself
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
    X_permuted = asc.permute_genes_within_celltype(s.X.toarray(), cell_labels)
    s_perm = sc.AnnData(X_permuted)
    sc.pp.normalize_total(s_perm, target_sum=1e4)
    sc.pp.log1p(s_perm)
    sc.pp.scale(s_perm)
    U_perm, sv_perm, V_perm = np.linalg.svd(s_perm.X, full_matrices=False)
    evs_perm = sv_perm**2 / n
    del X_permuted, s_perm, U_perm, V_perm

    # simulated from the fitted NB model at the cell-type labels, each
    # cell type at its own library size nb["L"][i], then put through the SAME
    # normalize -> log1p -> scale pipeline as the true data above; comparing
    # the two spectra only means something if the transform applied to each
    # is identical. This is the call analysis_spike_celltype makes for
    # spec_nb, so the bulk and spike views of the NB model stay in step, and
    # evs_mixture and evs_normal are built at the same per-metacell L.
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
    # the convention analysis_spike_celltype uses for spec_n.
    #
    # Row order is the model's own (spike["sample_labels"]) rather than the
    # h5ad's; singular values are invariant under row permutation so the bulk
    # does not need to realign.
    print("Computing the normal-model PCA eigenvalues")
    Z_normal = lwa.simulate_Z_guassian(model, spike)
    evs_normal = np.linalg.svd(Z_normal, full_matrices=False)[1]**2
    del Z_normal

    # the signal components are dropped so what is left is the bulk, which is
    # what the reference densities are compared against. SVD returns
    # eigenvalues descending, so the ones above the edge are a prefix. As in
    # analysis_bulk, k is a conservative 1.5x the elbow k stored in the h5ad
    # rather than a count against b_plus, to eliminate eigenvalues near the
    # bulk edge.
    k = int(1.5*s.obsm["X_pca"].shape[1])
    evs_pca = evs_pca[k:]
    evs_perm = evs_perm[k:]
    # evs_nb is simulated at the cell-type labels and evs_normal carries the
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


def make_all_bulk_pkl_celltype(datasets_list=datasets.DEFAULT_DATASETS,
                               n_genes_list=DEFAULT_N_GENES_CELLTYPE,
                               overwrite=False):
    """Build the cell-type bulk pkl for every (dataset, n_genes).

    A combination whose cell-type HVG h5ad has not been built is skipped
    with a warning rather than aborting the sweep, exactly as
    analysis_spike_celltype.make_all_spike_pkl_celltype does.
    """
    for dataset in _as_list(datasets_list):
        for n_genes in _as_list(n_genes_list):
            print(f"Processing {dataset} HVG {n_genes} (cell-type metacells)")
            try:
                anndata2bulk_celltype(dataset, n_genes, overwrite=overwrite)
            except ValueError as e:
                print(f"  SKIPPING {dataset}/{n_genes}: {e}")
    return None


## visualization functions ##

def available_n_genes_celltype(dataset, n_genes_list=DEFAULT_N_GENES_CELLTYPE):
    """The n_genes in n_genes_list whose cell-type bulk pkl already exists
    on disk. Judged on the PKL, not the h5ad, for the reason given in
    load_bulk_celltype_result."""
    return [n_genes for n_genes in _as_list(n_genes_list)
            if has_bulk_celltype_result(dataset, n_genes)]


def visualize_dataset_bulk_celltype(dataset, n_genes_list=DEFAULT_N_GENES_CELLTYPE,
                                    histogram_type="true", show=True):
    """One panel per n_genes for a dataset, read from the cached cell-type
    bulk pkls. Purely a reader: a missing pkl raises rather than being built.

    Draws through analysis_bulk.visualize_bulk, which takes the dict and an
    axis and knows nothing about where the dict came from. histogram_type is
    passed through: "true" histograms evs_pca, "perm" histograms evs_perm.
    """
    n_genes_list = _as_list(n_genes_list)

    fig, axes = plt.subplots(1, len(n_genes_list), figsize=(5 * len(n_genes_list), 4))
    if len(n_genes_list) == 1:
        axes = [axes]

    for ax, ng in zip(axes, n_genes_list):
        print(f"Visualizing {dataset} (cell-type metacells) with {ng} genes")
        bulk = load_bulk_celltype_result(dataset, ng)
        ab.visualize_bulk(bulk, ax=ax, title=f"{dataset} - {ng} genes  (k={bulk['k']})",
                          histogram_type=histogram_type)

    fig.suptitle(f"{dataset} bulk spectrum (cell-type metacells)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def make_bulk_figure_celltype(dataset_list=datasets.DEFAULT_DATASETS, n_genes=1000,
                              smoothing=0.05, include_unsmoothed_analytic=False,
                              min_ev=.001, show=True):
    """The NB and true eigenvalue histograms per dataset, each against the
    analytic mixture density (drawn as RMT) and the Marchenko-Pastur law,
    for the cell-type bulk pkls.

    figures_util.make_bulk_figure with its ab.anndata2bulk call replaced by
    load_bulk_celltype_result, so the two figures are laid out identically
    and can be compared panel for panel: rows are the entries of
    BULK_SPECTRA, columns datasets. n_genes defaults to 1000 rather than
    2000 because that is the only cell-type panel built.

    A dataset without a cell-type bulk pkl gets an empty column rather than
    aborting the figure. Every curve is rescaled to its OWN panel's histogram
    peak, so only its SHAPE and support are meaningful, never its height.
    """
    n_rows, n_cols = len(BULK_SPECTRA), len(dataset_list)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for j, dataset in enumerate(dataset_list):
        if not has_bulk_celltype_result(dataset, n_genes):
            print(f"  no cell-type bulk pkl for {dataset} n_genes={n_genes} -- leaving blank")
            for i in range(n_rows):
                axes[i, j].axis("off")
            axes[0, j].set_title(f"{dataset}\n(not built)", fontsize=12, color="grey")
            continue

        bulk = load_bulk_celltype_result(dataset, n_genes)
        x_grid = bulk["x_grid"]
        analytic = gaussian_filter1d(bulk["evs_mixture"],
                                     smoothing / (x_grid[1] - x_grid[0]))

        for i, (spectrum, key) in enumerate(BULK_SPECTRA):
            ax = axes[i, j]
            evs = bulk[key][bulk[key] >= min_ev]
            counts, _, _ = ax.hist(evs, bins=100, alpha=0.7, color="#a9a9a9")
            scale = counts.max()

            if include_unsmoothed_analytic:
                ax.plot(x_grid, bulk["evs_mixture"] * scale / bulk["evs_mixture"].max(),
                        color="orange", label="RMT (unsmoothed)")
            ax.plot(x_grid, analytic * scale / analytic.max(), color="blue",
                    linewidth=1.95, label="RMT")
            ax.plot(x_grid, bulk["evs_MP"] * scale / bulk["evs_MP"].max(),
                    color="red", linestyle="--", linewidth=1.95, label="MP")

            ax.set_xlim(0, bulk["b_plus"] + 1)
            ax.tick_params(labelsize=13)
            if i == 0:
                ax.set_title(dataset, fontsize=24)
            if j == 0:
                ax.set_ylabel(spectrum, fontsize=20)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    make_all_bulk_pkl_celltype()
