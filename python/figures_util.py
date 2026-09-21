import os
import pickle
import zlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.ndimage import gaussian_filter1d

import analysis_bulk as ab
import analysis_knn as ak
import analysis_knn_celltype as akc
import analysis_spike as asp
import analysis_spike_celltype as asc
import analysis_spike_util as asu
import datasets
import datasets_refined as dr
import lognorm_workflow_analytic as lwa
import normality as nrm

# How dataset keys are shown in figures and tables. The keys themselves (as
# in datasets.DEFAULT_DATASETS and every cached file name) stay as they are;
# only the rendered name changes. The key "Heart" is taken by the older,
# smaller heart dataset, which is why Heart_large cannot simply be renamed.
DISPLAY_NAMES = {"Heart_large": "Heart", "Breast_small": "Breast (1 donor)"}

# Datasets that must never reach a manuscript figure or table: "Heart" is the
# older, smaller heart dataset, superseded by Heart_large (which is what the
# manuscript calls Heart). display_name raises on these so that any figure
# or table code that is handed one fails loudly instead of mislabelling it.
EXCLUDED_DATASETS = {"Heart"}


def display_name(dataset):
    """The name a dataset is shown under in figures and tables."""
    if dataset in EXCLUDED_DATASETS:
        raise ValueError(f"dataset {dataset!r} is not part of the manuscript "
                         f"(see figures_util.EXCLUDED_DATASETS); did you mean Heart_large?")
    return DISPLAY_NAMES.get(dataset, dataset)


# How the methods are named in every figure, keyed by the names the pkls and
# the analysis code use. The manuscript introduces these names in the
# Results ("we refer to these eigenvalues as LML-RMT, metacell, true, and
# MP"), and its \normal, \nb, \perm and \true subscripts follow the same
# scheme, so every axis label, title and legend entry goes through here.
METHOD_LABELS = {"analytic": "LML-RMT", "RMT": "LML-RMT", "normal": "normal",
                 "nb": "metacell", "NB": "metacell", "perm": "permuted",
                 "true": "true", "MP": "MP"}


def method_label(method):
    """The name a method (analytic/normal/nb/perm/true, or an existing figure
    label such as RMT/NB) is shown under in figures."""
    return METHOD_LABELS.get(method, method)


## Spike pkl selection


# Every function below that reads the spike pkls takes the older `refined`
# flag (True: analysis_spike_refined, False: analysis_spike) and an HVG_type
# that, when given, overrides it: "refined", "unrefined" or "celltype", the
# last being analysis_spike_celltype's pkls, whose metacells are cell types
# (datasets_celltype.py) rather than SuperCell/refinement groups. All three
# carry the same "cell_labels", "spec_analytic", "spec_n", "spec_nb",
# "spec_perm" and "spec_true" keys, so nothing downstream of the load needs
# to know which it got. figures_celltype.py is the caller that passes
# "celltype"; every call in figures.py, make.py and table.py leaves
# HVG_type at None and so behaves as it always did.
HVG_TYPES = ("refined", "unrefined", "celltype")


def resolve_HVG_type(refined=True, HVG_type=None):
    """The partition an (refined, HVG_type) pair selects; HVG_type wins."""
    if HVG_type is None:
        return "refined" if refined else "unrefined"
    if HVG_type not in HVG_TYPES:
        raise ValueError(f"HVG_type must be one of {HVG_TYPES}, got {HVG_type!r}")
    return HVG_type


def has_spike(dataset, n_genes, refined=True, HVG_type=None):
    HVG_type = resolve_HVG_type(refined, HVG_type)
    if HVG_type == "celltype":
        return asc.has_spike_celltype_result(dataset, n_genes)
    return asp.has_spike_result(dataset, n_genes, refined=(HVG_type == "refined"))


def load_spike(dataset, n_genes, refined=True, HVG_type=None):
    HVG_type = resolve_HVG_type(refined, HVG_type)
    if HVG_type == "celltype":
        return asc.load_spike_celltype_result(dataset, n_genes)
    return asp.load_spike_result(dataset, n_genes, refined=(HVG_type == "refined"))


def load_spike_entries(refined=True, dataset_list=None, n_genes_list=None, HVG_type=None):
    """(dataset, n_genes, result) for every cached spike pkl of the partition,
    as analysis_spike._load_spike_entries returns them: purely a reader, a
    combination without a pkl is skipped."""
    HVG_type = resolve_HVG_type(refined, HVG_type)
    if HVG_type != "celltype":
        return asp._load_spike_entries(HVG_type == "refined", dataset_list, n_genes_list)
    dataset_list = datasets.DEFAULT_DATASETS if dataset_list is None else dataset_list
    n_genes_list = datasets.DEFAULT_N_GENES if n_genes_list is None else n_genes_list
    return [(dataset, n_genes, asc.load_spike_celltype_result(dataset, n_genes))
            for dataset in dataset_list for n_genes in n_genes_list
            if asc.has_spike_celltype_result(dataset, n_genes)]


def load_knn(dataset, n_genes, k_nn=10, refined=True, HVG_type=None):
    """The partition's knn pkl (analysis_knn.anndata2knn, or
    analysis_knn_celltype.anndata2knn_celltype for "celltype"), built on
    first use."""
    HVG_type = resolve_HVG_type(refined, HVG_type)
    if HVG_type == "celltype":
        return akc.anndata2knn_celltype(dataset, n_genes, k_nn=k_nn)
    return ak.anndata2knn(dataset, n_genes, k_nn=k_nn, refined=(HVG_type == "refined"))


## Bulk figure


def _dataset_snr_and_genes(dataset, n_genes, refined=True):
    """(var_names, snr) for one HVG dataset, the two aligned element for
    element -- lwa.compute_snr returns one entry per gene in the object's own
    column order, so var_names[i] names snr[i].

    Split out of compute_dataset_snr so compute_snr_gene_bins can group the
    values by gene without loading the h5ad and refitting the model twice.
    """
    s = (dr.load_HVG_refined_dataset(dataset, n_genes) if refined
         else datasets.load_HVG_dataset(dataset, n_genes))
    n = s.shape[0]
    model, _ = lwa.log_normalized_spiked_mixture_model(s.uns["nb_model"], n, L0=1e4,
                                                       include_spike=False)
    return np.asarray(s.var_names), lwa.compute_snr(model, s.obs["metacell"].to_numpy())


def compute_dataset_snr(dataset, n_genes, refined=True):
    return _dataset_snr_and_genes(dataset, n_genes, refined=refined)[1]


def compute_snr_gene_bins(dataset, n_genes_list, refined=True):
    """(labels, values): the SNR of each HVG panel's OWN new genes, one entry
    per step in n_genes_list.

    n_genes_list is walked in ascending order. A gene belongs to the bin of
    the first panel it appears in, so bin j holds the genes of panel j that
    were in no smaller panel -- "genes 1001-2000" is the n_genes=2000 build
    minus everything already in the n_genes=1000 build, and so on. The bins
    are therefore disjoint, and (because the panels are nested in practice)
    they partition the largest panel's gene set exactly.

    Membership is tested against the union of ALL smaller panels rather than
    just the previous one. The panels are very nearly nested but not exactly
    -- one gene of Heart's n_genes=1000 build is absent from its n_genes=2000
    build -- and the union keeps such a gene from re-entering a later bin as
    though it were new.

    The SNR values come from the build the gene ENTERED in, which is what
    makes the bins comparable to their own panel rather than to each other:
    lwa.compute_snr is fitted per dataset, so the same gene carries a
    different value in the n_genes=2000 and n_genes=5000 builds.

    Bin sizes fall short of the nominal 1000/2000/5000 because each build
    drops its low-count genes (n_genes=1000 leaves ~950).

    Returns (labels, values) with labels like "1-1000", "1001-2000" and
    values a list of arrays, ready for a single violinplot call.
    """
    n_genes_list = sorted(n_genes_list)
    labels, values, seen = [], [], set()

    for i, n_genes in enumerate(n_genes_list):
        var_names, snr = _dataset_snr_and_genes(dataset, n_genes, refined=refined)
        is_new = np.array([name not in seen for name in var_names])
        labels.append(f"1-{n_genes}" if i == 0
                      else f"{n_genes_list[i - 1] + 1}-{n_genes}")
        values.append(snr[is_new])
        seen.update(var_names)

    return labels, values


# the empirical spectra the histogram block draws, one column each; the keys
# are anndata2bulk's own names (evs_pca is the true data's PCA spectrum)
BULK_SPECTRA = [("NB", "evs_nb"), ("true", "evs_pca")]


def make_bulk_figure(dataset_list=datasets.DEFAULT_DATASETS, n_genes=2000,
                     refined=True, smoothing=0.05,
                     include_unsmoothed_analytic=False, min_ev=.001,
                     show=True):
    """The NB and true eigenvalue histograms per dataset, each against the
    analytic mixture density (drawn as RMT) and the Marchenko-Pastur law.

    Rows are the entries of BULK_SPECTRA, columns datasets. This replaces the
    n_genes sweep the figure used to hold: the panels of a column are now
    several spectra at ONE n_genes rather than one spectrum at three n_genes.
    The SNR violins that used to sit in a trailing column now live in
    make_MP_figure's panel B, so this figure is spectra only.

    The dataset loop stays OUTSIDE the spectrum loop even though datasets are
    now the columns: anndata2bulk and the smoothed analytic curve are per
    dataset, and depend on n_genes rather than on which empirical spectrum
    they are drawn over, so walking datasets outermost still builds them once
    each and reuses them down the column.

    Every curve is rescaled to its OWN panel's histogram peak, so only its
    SHAPE and support are meaningful, never its height.
    """
    n_rows, n_cols = len(BULK_SPECTRA), len(dataset_list)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for j, dataset in enumerate(dataset_list):
        bulk = ab.anndata2bulk(dataset, n_genes, refined=refined)
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
                        color="orange", label=method_label("RMT") + " (unsmoothed)")
            ax.plot(x_grid, analytic * scale / analytic.max(), color="blue",
                    linewidth=1.95, label=method_label("RMT"))
            ax.plot(x_grid, bulk["evs_MP"] * scale / bulk["evs_MP"].max(),
                    color="red", linestyle="--", linewidth=1.95, label=method_label("MP"))

            ax.set_xlim(0, bulk["b_plus"] + 1)
            ax.tick_params(labelsize=13)
            if i == 0:
                ax.set_title(display_name(dataset), fontsize=24)
            if j == 0:
                ax.set_ylabel(method_label(spectrum), fontsize=20)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _add_snr_column(ax, dataset, n_genes_list, refined=True, title=None):
    """The SNR violins for one dataset, drawn into ax.

    Binned by the gene panel each gene FIRST entered rather than by panel
    size, so the violins hold disjoint gene sets instead of nested supersets
    -- see compute_snr_gene_bins.
    """
    snr_labels, snr_values = compute_snr_gene_bins(dataset, n_genes_list,
                                                   refined=refined)
    ax.violinplot(snr_values, positions=range(len(snr_values)), showextrema=False)
    ax.set_xticks(range(len(snr_values)))
    ax.set_xticklabels(snr_labels)
    ax.tick_params(labelsize=13)
    if title is not None:
        ax.set_title(title, fontsize=24)
    return ax


def make_MP_figure(dataset_list=datasets.DEFAULT_DATASETS,
                   n_genes_list=[1000, 2000, 5000], refined=True,
                   smoothing=0.05, min_ev=.001, show=True):
    """A: the true PCA eigenvalue histogram per (dataset, n_genes) against
    the Marchenko-Pastur law and the analytic mixture density (drawn as RMT).
    B: the SNR of each gene panel's own new genes, titled "signal strength"
    as the manuscript calls it.

    Rows are datasets, columns the entries of n_genes_list plus the SNR
    column. This is the n_genes sweep make_bulk_figure used to carry, kept
    deliberately spare: only the true spectrum is histogrammed, and only over
    the same two curves make_bulk_figure draws, so the panels show how the
    fit of both holds up as the gene panel grows. The other empirical
    spectra are make_bulk_figure's job.

    Both references are rebuilt for every (dataset, n_genes) rather than once
    per dataset: x_grid, evs_MP and evs_mixture all change with n_genes, so
    hoisting them out of the inner loop would draw one panel's curves over
    another's histogram. The analytic curve is smoothed onto x_grid exactly
    as make_bulk_figure smooths it, so the two figures draw the same RMT.

    Every curve is rescaled to its panel's histogram peak, so only its SHAPE
    and support are meaningful, never its height.
    """
    n_cols = len(n_genes_list) + 1
    fig, axes = plt.subplots(len(dataset_list), n_cols,
                             figsize=(5 * n_cols, 4 * len(dataset_list)),
                             squeeze=False)

    for i, dataset in enumerate(dataset_list):
        for j, n_genes in enumerate(n_genes_list):
            ax = axes[i, j]
            bulk = ab.anndata2bulk(dataset, n_genes, refined=refined)
            x_grid = bulk["x_grid"]
            analytic = gaussian_filter1d(bulk["evs_mixture"],
                                         smoothing / (x_grid[1] - x_grid[0]))

            evs = bulk["evs_pca"][bulk["evs_pca"] >= min_ev]
            counts, _, _ = ax.hist(evs, bins=100, alpha=0.7, color="#a9a9a9")
            scale = counts.max()

            ax.plot(x_grid, analytic * scale / analytic.max(), color="blue",
                    linewidth=1.95, label=method_label("RMT"))
            ax.plot(x_grid, bulk["evs_MP"] * scale / bulk["evs_MP"].max(),
                    color="red", linestyle="--", linewidth=1.95, label=method_label("MP"))

            ax.set_xlim(0, bulk["b_plus"] + 1)
            ax.tick_params(labelsize=13)
            if i == 0:
                ax.set_title(f"{n_genes} genes", fontsize=24)
            if j == 0:
                ax.set_ylabel(display_name(dataset), fontsize=20)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

        _add_snr_column(axes[i, -1], dataset, n_genes_list, refined=refined,
                        title="signal strength" if i == 0 else None)

    # A labels the histogram block, B the SNR column; placed above and to the
    # left of each axes rather than inside it, so they read as figure-part
    # labels rather than as panel annotations
    for ax, letter in ((axes[0, 0], "A"), (axes[0, -1], "B")):
        ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=30,
                fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


### Normality figures


# the four spectra compared against the analytic reference; the spec_* keys
# are the pkl's own names (spec_n is the normal one)
NORMALITY_METHODS = [("normal", "spec_n"), ("nb", "spec_nb"),
                     ("perm", "spec_perm"), ("true", "spec_true")]


def compute_eigenvalue_spectrum(dataset, n_genes=1000, refined=True,
                                min_metacell_size=None, HVG_type=None):
    """Per-metacell covariance eigenvalues, paired between each method and
    analytic on the same metacell.

    For one metacell, the method's PC scores U*sv give a k x k empirical
    covariance whose eigenvalues (np.linalg.eigvalsh, sorted descending so
    index 0 is the largest -- the same "rank 0 = highest variance"
    convention as metacell_eigenvectors elsewhere in this file) are paired
    by rank against analytic's eigenvalues for that SAME metacell: x is
    always analytic's j-th largest eigenvalue, y the method's. This assumes
    the two spectra's j-th ranked directions are the right match to compare
    -- not that they are the same direction, only that they carry the same
    rank.

    analytic's own per-metacell eigenvalues are computed once and reused for
    every method's pairing, rather than recomputed per method, since they
    do not depend on which method is being compared.

    Uses the same k and the same min_metacell_size floor (default k + 1) as
    the other normality figures, for the same reason: below it the
    covariance is rank deficient and its smallest eigenvalues are ~0 by
    construction rather than meaningfully estimated.

    Every eigenvalue is scaled by n, the FULL dataset's cell count (not the
    metacell's own size) -- the same normalization anndata2spike applies to
    its singular values (sv/sqrt(n), see analysis_spike.py), so these
    eigenvalues sit on that same scale rather than the raw per-cell
    covariance one.

    HVG_type selects the partition's spike pkl (see load_spike); None
    follows refined.

    Returns {method: (x, y)}, parallel arrays of length n_metacells * k --
    x analytic's eigenvalues, y the method's.
    """
    result = load_spike(dataset, n_genes, refined=refined, HVG_type=HVG_type)
    cell_labels = np.asarray(result["cell_labels"])
    spec_analytic = result["spec_analytic"]
    method_specs = [(method, result[key]) for method, key in NORMALITY_METHODS]

    k = min([len(spec_analytic["sv"])] + [len(spec["sv"]) for _, spec in method_specs])
    if min_metacell_size is None:
        min_metacell_size = k + 1

    # the FULL dataset's cell count, not any one metacell's -- matches the
    # sv/sqrt(n) convention anndata2spike itself uses (analysis_spike.py),
    # so these eigenvalues sit on the same scale as that normalization
    n = len(cell_labels)

    labels, sizes = np.unique(cell_labels, return_counts=True)
    labels = labels[sizes >= min_metacell_size]
    scores_analytic = spec_analytic["U"][:, :k] * spec_analytic["sv"][:k]

    analytic_eigs = {
        label: np.linalg.eigvalsh(np.cov(scores_analytic[cell_labels == label, :],
                                         rowvar=False))[::-1] * n
        for label in labels}

    def spectrum_xy(scores_method):
        x, y = [], []
        for label in labels:
            mask = cell_labels == label
            eig_m = np.linalg.eigvalsh(np.cov(scores_method[mask, :], rowvar=False))[::-1] * n
            x.append(analytic_eigs[label])
            y.append(eig_m)
        return np.concatenate(x), np.concatenate(y)

    out = {}
    for method, spec in method_specs:
        out[method] = spectrum_xy(spec["U"][:, :k] * spec["sv"][:k])
    return out


# the spectra the max-variance figure histograms, top row to bottom. analytic's
# value is the LML-RMT prediction itself (lwa.get_GMM), so it carries no
# pkl key; the others are sample values from the pkl's PC scores
MAX_VAR_METHODS = [("analytic", None), ("nb", "spec_nb"), ("true", "spec_true")]


def compute_metacell_max_var(dataset, n_genes=1000, refined=True,
                            min_metacell_size=None, HVG_type=None):
    """Per-metacell variance along the metacell's maximal principal
    direction in PCA space -- the largest eigenvalue of its covariance
    Sigma -- for the LML-RMT prediction and for the metacell (nb) and true
    PCA embeddings (the entries of MAX_VAR_METHODS), each divided by that
    method's smallest value, so the arrays are unitless ratios >= 1.

    The reference is taken per method and per dataset: v_0 is the
    smallest value over the metacells of that method's own embedding, and
    every metacell's value is v_i / v_0. Each panel of the figure
    therefore starts at exactly 1, and what a panel shows is the spread of
    metacell widths relative to its own narrowest metacell -- the absolute
    scale, and any overall under-estimate of the true widths, is removed.
    (Scaling every method by one shared reference metacell would instead
    let nb and true dip below 1 where another metacell is narrower in that
    embedding.)

    For analytic, Sigma is the predicted per-cell covariance of the
    metacell (lwa.get_GMM on the pkl's pca_theory), truncated to its first
    k coordinates: the same matrix whose normal density the visualize
    figure draws. For nb and true, Sigma is the sample covariance of the
    metacell's rows of U[:, :k] * sv[:k], i.e. the cells centered on the
    metacell's own mean. The maximal principal direction is each Sigma's
    OWN top eigenvector, so the direction can differ between methods for
    the same metacell; only the variance along it is compared.

    Unlike compute_eigenvalue_spectrum, nothing here is rescaled by the
    cell count: these are per-cell values on the scale of the PC scores
    themselves, which is the scale get_GMM's covariance is on.

    Uses the same k and the same min_metacell_size floor (default k + 1)
    as the other normality figures, and applies the floor to all three
    methods, so the rows of the figure histogram the SAME metacells.

    HVG_type selects the partition's spike pkl (see load_spike); None
    follows refined.

    Returns {method: values}, one array of scaled values per entry of
    MAX_VAR_METHODS, each of length n_metacells in np.unique's label order.
    """
    result = load_spike(dataset, n_genes, refined=refined, HVG_type=HVG_type)
    cell_labels = np.asarray(result["cell_labels"])
    pca_theory = result["pca_theory"]
    specs = {method: result[key] for method, key in MAX_VAR_METHODS if key is not None}

    k = min([len(pca_theory["values"])] + [len(spec["sv"]) for spec in specs.values()])
    if min_metacell_size is None:
        min_metacell_size = k + 1

    labels, sizes = np.unique(cell_labels, return_counts=True)
    labels = labels[sizes >= min_metacell_size]

    def max_var(cov):
        # eigvalsh sorts ascending, so [-1] is the largest eigenvalue
        return np.linalg.eigvalsh(cov)[-1]

    out = {"analytic": np.array([
        max_var(np.asarray(lwa.get_GMM(pca_theory, label)["covariance"])[:k, :k])
        for label in labels])}
    for method, spec in specs.items():
        scores = np.asarray(spec["U"])[:, :k] * np.asarray(spec["sv"])[:k]
        out[method] = np.array([
            max_var(np.cov(scores[cell_labels == label, :], rowvar=False))
            for label in labels])

    # each method scaled by its own narrowest metacell, so every panel's
    # minimum is exactly 1
    return {method: values / values.min() for method, values in out.items()}


def make_metacell_max_var_figure(dataset_list=datasets.DEFAULT_DATASETS,
                                  n_genes=1000, refined=True, HVG_type=None,
                                  n_bins=30, show=True):
    """Histograms over metacells of the scaled variance along each
    metacell's maximal principal direction in PCA space
    (compute_metacell_max_var: the variance divided by the panel's
    smallest, so every panel starts at 1): rows are the entries of
    MAX_VAR_METHODS -- the LML-RMT prediction, then the metacell and true
    embeddings -- and columns are datasets.

    The x axis is logarithmic in base 2, with the ticks labelled by the
    ratio itself (1, 2, 4, 8, ...) rather than by its exponent, and no
    minor tick labels. Every panel shares ONE x range and one set of
    log-spaced bins, from 1 to the largest ratio over all datasets and
    methods, so widths are read on the same scale across the whole figure;
    y is shared down each column only, since the datasets have different
    metacell counts. The scaling removes the absolute scale, so what a
    panel shows is how spread out its metacell widths are relative to its
    own narrowest metacell.

    Each column's top panel is titled with the dataset and the number of
    metacells retained by compute_metacell_max_var's k + 1 floor; the
    rows are labelled on the first column through method_label.
    """
    n_rows, n_cols = len(MAX_VAR_METHODS), len(dataset_list)

    # every dataset first: the shared bins need the global maximum
    ratios = {dataset: compute_metacell_max_var(dataset, n_genes, refined=refined,
                                                 HVG_type=HVG_type)
              for dataset in dataset_list}
    hi = max(v.max() for per_method in ratios.values() for v in per_method.values())
    bins = np.geomspace(1, hi, n_bins + 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey="col", squeeze=False)

    for j, dataset in enumerate(dataset_list):
        values = ratios[dataset]
        for i, (method, _) in enumerate(MAX_VAR_METHODS):
            ax = axes[i, j]
            ax.hist(values[method], bins=bins, alpha=0.7, color="#a9a9a9")
            ax.set_xscale("log", base=2)
            # powers of two, labelled as plain ratios rather than 2^k, and
            # the minor ticks left unlabelled
            ax.xaxis.set_major_locator(LogLocator(base=2))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(labelsize=13)
            if i == 0:
                ax.set_title(f"{display_name(dataset)} ({len(values[method])} metacells)",
                             fontsize=20)
            if i == n_rows - 1:
                ax.set_xlabel("scaled maximal variance", fontsize=16)
            if j == 0:
                ax.set_ylabel(method_label(method), fontsize=20)

    axes[0, 0].set_xlim(1, hi)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def compute_outlier_spectrum(dataset, n_genes, refined=True, methods=None,
                             HVG_type=None):
    """{method: (x, y)} of outlier singular values for one dataset: x is
    the LML-RMT (analytic) prediction and y the method's numerically
    computed value at the same rank, both sorted descending, read from the
    cached spike pkl (analysis_spike.load_spike_result). The two arrays are
    aligned rank by rank, as anndata2spike stores them: the top k singular
    values of each matrix against the k analytic outliers.

    methods are the keys of analysis_spike.SPEC_KEYS ("normal", "nb",
    "perm", "true", ...); None takes every method present in the pkl. A
    method absent from the pkl is left out rather than raising.

    The same (x, y) shape as compute_eigenvalue_spectrum and
    compute_neighbor_distance_ratios return, so median_relative_error
    applies to it unchanged.
    """
    result = load_spike(dataset, n_genes, refined=refined, HVG_type=HVG_type)
    x = np.asarray(result["spec_analytic"]["sv"], dtype=float)
    if methods is None:
        methods = [m for m, key in asp.SPEC_KEYS.items() if key in result]
    out = {}
    for method in methods:
        key = asp.SPEC_KEYS[method]
        if key not in result:
            continue
        y = np.asarray(result[key]["sv"], dtype=float)
        k = min(len(x), len(y))
        out[method] = (x[:k], y[:k])
    return out


def median_relative_error(x, y):
    """Median relative error of a method's values y against analytic's x
    (two parallel arrays, e.g. from compute_eigenvalue_spectrum or
    compute_neighbor_distance_ratios): the median over points of
    |y - x| / |x|, i.e. each value's deviation from the matching analytic
    one as a fraction of that analytic value. The median rather than the
    mean, so a handful of points far off the x = y line does not set the
    value.

    Taken on the raw values, not their logs -- dividing by x already puts
    large and small x on the same footing, which is what a log transform
    would otherwise be for. Summarizing over points is what lets datasets
    with very different point counts be compared, or pooled, on an equal
    footing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.median(np.abs(y - x) / np.abs(x)))


# axis label size of the normality spectrum and neighbor distance figures,
# which are read side by side: 16.9 is 30% over the 13 their ticks use
SCATTER_LABEL_FONTSIZE = 16.9
# the one colour the pooled scatters (normality spectrum and neighbor
# distance figures) draw every point in: the datasets are not distinguished
# and no legend is drawn. It is the steelblue analysis_knn._plot_adjacency_values
# uses when given no groups, so the knn figure matches.
SCATTER_COLOR = "steelblue"


def make_normality_spectrum_figure(dataset_list=datasets.DEFAULT_DATASETS,
                                   n_genes=1000, refined=True,
                                   min_metacell_size=None, methods=None,
                                   HVG_type=None, show=True):
    """Each method's per-metacell covariance eigenvalues against analytic's
    on the same metacell and the same rank, pooled across every dataset in
    dataset_list into one panel per method: a single row with one panel
    per entry of methods, rather than one row per dataset.

    HVG_type selects the partition's spike pkl (see load_spike); None
    follows refined.

    methods lists which of compute_eigenvalue_spectrum's methods ("normal",
    "nb", "perm", "true") get a panel, in that left-to-right order; None
    draws all four.

    Each panel is annotated in its lower right with "REL ERR: value",
    where value is the per-point median relative error |y - x| / |x|
    (median_relative_error) computed separately for every dataset in the
    panel and then the mean of those over datasets -- a mean of
    per-dataset medians, so each dataset counts once regardless of how
    many points it contributes.

    Every point is drawn in SCATTER_COLOR and no legend is drawn, so the
    datasets are not distinguished. The scatter is still one call per
    dataset, since the per-dataset relative errors are collected in the
    same loop.

    All panels share one axis range, computed from every dataset's points
    together, so the methods are read on an identical scale;
    the x = y line is where a method's spectrum exactly reproduces
    analytic's.

    Both axes are log-scaled. A metacell's covariance eigenvalues span
    several orders of magnitude between rank 0 (the dominant direction) and
    the trailing ranks -- on a linear axis the largest point sets the
    limits and compresses every other rank's points into a sliver near the
    origin where they are indistinguishable. Log-log spreads that same
    range out evenly instead.
    """
    if methods is None:
        methods = [method for method, _ in NORMALITY_METHODS]
    methods = list(methods)

    spectra_by_dataset = {
        dataset: compute_eigenvalue_spectrum(
            dataset, n_genes, refined=refined, min_metacell_size=min_metacell_size,
            HVG_type=HVG_type)
        for dataset in dataset_list}

    # the shared axis range only looks at the methods actually drawn, so a
    # left-out method cannot stretch the limits of the panels that remain
    drawn = [spectrum[method] for spectrum in spectra_by_dataset.values()
             for method in methods]
    lo = min(min(x.min(), y.min()) for x, y in drawn) / 1.5
    hi = max(max(x.max(), y.max()) for x, y in drawn) * 1.5
    lim = [lo, hi]

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                             squeeze=False)

    for j, method in enumerate(methods):
        ax = axes[0, j]
        rel_err_by_dataset = []
        for dataset in dataset_list:
            x, y = spectra_by_dataset[dataset][method]
            rel_err_by_dataset.append(median_relative_error(x, y))
            ax.scatter(x, y, color=SCATTER_COLOR, s=6, alpha=0.4, edgecolors="none")
        ax.plot(lim, lim, color="black", ls="--", lw=1)
        ax.text(0.97, 0.03, f"REL ERR: {np.mean(rel_err_by_dataset):.3g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.tick_params(labelsize=13)
        # the y label alone names the panel's method; there is no title
        ax.set_xlabel(f"{method_label('analytic')} variance",
                      fontsize=SCATTER_LABEL_FONTSIZE)
        ax.set_ylabel(f"{method_label(method)} variance",
                      fontsize=SCATTER_LABEL_FONTSIZE)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# the KS figures project each metacell onto KS_DIRECTIONS_PER_DIM * k random
# unit directions of the k-dimensional PCA space rather than onto the k
# eigenvectors of its own covariance (normality.metacell_normality, which the
# refinement step in datasets_refined still uses). The eigenvector basis
# singles out the few low-variance directions of a large metacell that load
# on genes the model says are absent from it, along which the metacell
# model's count noise is discrete and far from normal; random directions
# blend those with the rest and so test the typical direction. The
# directions are drawn once per metacell, from a generator seeded by
# KS_DIRECTION_SEED and the dataset name, and reused for every spectrum of
# that metacell, so the spectra are compared on the same directions.
KS_DIRECTIONS_PER_DIM = 10
KS_DIRECTION_SEED = 0


def random_unit_directions(k, n_directions, rng):
    """n_directions independent unit vectors of R^k, as columns."""
    G = rng.standard_normal((k, n_directions))
    return G / np.linalg.norm(G, axis=0)


def ks_along_directions(X, directions):
    """The KS statistic of X (cells x k) projected onto each column of
    directions: normality.max_ecdf_normal_diff against a normal fitted to
    that projection's own mean and sd, so every value is self-referential
    and invariant to the sign and length of the direction."""
    proj = X @ directions
    return np.array([nrm.max_ecdf_normal_diff(proj[:, j], proj[:, j].mean(), proj[:, j].std())
                     for j in range(proj.shape[1])])


def compute_normality_KS(dataset, n_genes=1000, HVG_type="refined", min_metacell_size=None):
    """Per-metacell, per-direction KS statistic, self-referential for each
    spectrum: analytic, normal, nb, perm, true each projected onto the SAME
    KS_DIRECTIONS_PER_DIM * k random unit directions of PCA space (see the
    note above KS_DIRECTIONS_PER_DIM), ks_along_directions giving the KS
    statistic of every projection against a normal fitted to its own mean
    and sd -- so one metacell contributes 10k values per spectrum, not one.

    HVG_type selects which spike pkl the spectra are read from (see
    load_spike): "refined", "unrefined" or "celltype".

    Uses the same k and the same min_metacell_size floor (default k + 1) as
    the other normality figures, for the same reason: below it the
    covariance is rank deficient.

    Returns {spectrum: values}, one flat array per spectrum ("analytic",
    "normal", "nb", "perm", "true") of length n_metacells * 10k, pooling
    every metacell's per-direction values together.
    """
    result = load_spike(dataset, n_genes, HVG_type=HVG_type)
    cell_labels = np.asarray(result["cell_labels"])
    spec_analytic = result["spec_analytic"]
    all_specs = [("analytic", spec_analytic)] + [(method, result[key])
                                                  for method, key in NORMALITY_METHODS]

    k = min(len(spec["sv"]) for _, spec in all_specs)
    if min_metacell_size is None:
        min_metacell_size = k + 1

    labels, sizes = np.unique(cell_labels, return_counts=True)
    labels = labels[sizes >= min_metacell_size]

    # zlib.crc32 rather than hash(): Python's string hash changes from one
    # process to the next, and the directions must not
    rng = np.random.default_rng([KS_DIRECTION_SEED, zlib.crc32(dataset.encode())])
    directions = {label: random_unit_directions(k, KS_DIRECTIONS_PER_DIM * k, rng)
                  for label in labels}

    out = {}
    for name, spec in all_specs:
        scores = spec["U"][:, :k] * spec["sv"][:k]
        values = [ks_along_directions(scores[cell_labels == label, :], directions[label])
                  for label in labels]
        out[name] = np.concatenate(values)
    return out


# how each spectrum is named on the x axis of make_normality_KS_figure;
# a spectrum absent here is labelled by its own name. analytic is drawn as
# RMT to match the bulk and MP figures, which already call it that.
# kept for callers that pass their own overrides; the ticks otherwise go
# through method_label, so analytic reads as LML-RMT and nb as metacell
KS_SPECTRUM_LABELS = {}


def make_normality_KS_figure(dataset_list=datasets.DEFAULT_DATASETS,
                             n_genes=1000, HVG_type="refined",
                             min_metacell_size=None, n_cols=4, spectra=None,
                             show=True):
    """One box per spectrum per dataset: the pooled distribution over
    metacells (and random direction) of the KS statistic against a normal
    fitted along that direction (see compute_normality_KS), with the chosen
    spectra as boxplots side by side in a single panel per dataset, laid
    out with n_cols dataset panels per row. Outlier points are suppressed
    (showfliers=False) since these pool thousands of values and would
    otherwise be mostly a cloud of dots.

    HVG_type is "refined", "unrefined", or "celltype" -- see
    compute_normality_KS for what each one loads. figures.make_KS_figure
    draws the refined one and figures.make_KS_figure_celltype the cell-type
    one, with identical layout so the two read as a pair.

    spectra lists which of compute_normality_KS's spectra ("analytic",
    "normal", "nb", "perm", "true") to draw, in that left-to-right order;
    None draws all five. The x tick under each box is the spectrum's entry
    in KS_SPECTRUM_LABELS when it has one, so analytic reads as RMT.

    Putting the five spectra in one panel (rather than one histogram panel
    each, as an earlier version of this figure did) is what makes them
    directly comparable within a dataset: the eye can line up medians and
    interquartile ranges across boxes in a way it cannot across separate
    histograms with their own bins and counts.
    """
    if spectra is None:
        spectra = ["analytic"] + [method for method, _ in NORMALITY_METHODS]
    panels = list(spectra)
    tick_labels = [KS_SPECTRUM_LABELS.get(name, method_label(name)) for name in panels]

    n_rows = int(np.ceil(len(dataset_list) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for i, dataset in enumerate(dataset_list):
        ax = axes[i // n_cols, i % n_cols]
        ks_by_spectrum = compute_normality_KS(
            dataset, n_genes, HVG_type=HVG_type, min_metacell_size=min_metacell_size)
        values = [ks_by_spectrum[name] for name in panels]

        ax.boxplot(values, positions=range(len(panels)), showfliers=False)
        ax.set_xticks(range(len(panels)))
        ax.set_xticklabels(tick_labels)
        # the tick labels (the method names under the boxes and the y ticks)
        # and the y label are 25% over the 13 and 14 the other per-dataset
        # figures use; the titles stay at 20
        ax.tick_params(labelsize=16.25)
        ax.set_title(f"{display_name(dataset)} ({len(values[0])} points)", fontsize=20)
        if i % n_cols == 0:
            ax.set_ylabel("KS", fontsize=17.5)

    # a dataset count that does not fill the last row leaves empty panels
    for j in range(len(dataset_list), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


### Neighbor distance figures


def compute_neighbor_moments(dataset, n_genes=1000, refined=True,
                             min_metacell_size=None, n_neighbors=5, HVG_type=None):
    """Per-metacell diff and sd, evaluated independently in the method's own
    space and in analytic's own space, on the neighbor pair(s) each space's
    OWN topology picks out.

    This is the shared pass behind compute_neighbor_distance_ratios (which
    reports diff/sd) and make_neighbor_sd_figure (which plots sd and diff
    separately) -- the neighbor search is the expensive part and is
    identical for both, so it happens once here and each consumer is a thin
    projection of the result.

    For metacell i, nn_method[i] is its n_neighbors nearest OTHER metacells
    by Euclidean distance between means in the method's own embedding,
    nn_analytic[i] the same n_neighbors computed in analytic's own
    embedding -- two independent nearest-neighbor searches, each using
    only that space's own means. The pair set for i is
    set(nn_method[i]) | set(nn_analytic[i]): as few as n_neighbors labels
    if the two spaces' top-n_neighbors lists coincide exactly, as many as
    2 * n_neighbors if they share none.

    For every (i, j) pair that survives, both quantities are computed on
    each side, entirely self-contained in its own space -- no basis
    rotation between method and analytic is needed here, unlike figures
    that compare directions across the two spaces, because both diff and
    sd are recomputed from scratch on each side rather than one side's
    direction being reused for the other's covariance:

        diff = ||mu[j] - mu[i]||           (that space's own means)
        d_hat = (mu[j] - mu[i]) / diff
        sd = sqrt(d_hat @ Sigma[i] @ d_hat)  (metacell i's OWN covariance,
                                              in that same space -- never
                                              the neighbor's covariance or
                                              an average of the two)

    Mirrored pairs (i's search finds j, j's separate search finds i) are
    not deduplicated -- they are not redundant anyway, since the reference
    covariance (metacell i's vs metacell j's) differs between them.

    Uses the same k and the same min_metacell_size floor (default k + 1) as
    the other normality figures, for the same reason: below it the
    covariance is rank deficient. n_neighbors is clamped to at most
    n_metacells - 1, so a dataset with very few metacells still works.

    HVG_type selects the partition's spike pkl (see load_spike); None
    follows refined.

    Returns {method: (diff_analytic, sd_analytic, diff_method, sd_method)},
    parallel arrays with n_neighbors to 2 * n_neighbors entries per
    metacell, depending on how much the two spaces' neighbor lists overlap.
    """
    result = load_spike(dataset, n_genes, refined=refined, HVG_type=HVG_type)
    cell_labels = np.asarray(result["cell_labels"])
    spec_analytic = result["spec_analytic"]
    method_specs = [(method, result[key]) for method, key in NORMALITY_METHODS]

    k = min([len(spec_analytic["sv"])] + [len(spec["sv"]) for _, spec in method_specs])
    if min_metacell_size is None:
        min_metacell_size = k + 1

    labels, sizes = np.unique(cell_labels, return_counts=True)
    labels = labels[sizes >= min_metacell_size]
    scores_analytic = spec_analytic["U"][:, :k] * spec_analytic["sv"][:k]
    n_neighbors = min(n_neighbors, len(labels) - 1)

    def nearest_neighbors(means):
        # for each label, its n_neighbors closest OTHER labels, nearest first
        label_list = list(labels)
        M = np.stack([means[l] for l in label_list])
        dist = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1)[:, :n_neighbors]
        return {label_list[i]: [label_list[o] for o in order[i]]
               for i in range(len(label_list))}

    def diff_and_sd(means, scores, i, j):
        direction = means[j] - means[i]
        diff = np.linalg.norm(direction)
        d_hat = direction / diff
        cov_i = np.cov(scores[cell_labels == i, :], rowvar=False)
        sd = np.sqrt(max(d_hat @ cov_i @ d_hat, 0))
        return diff, sd

    means_analytic = {label: scores_analytic[cell_labels == label, :].mean(axis=0)
                      for label in labels}
    nn_analytic = nearest_neighbors(means_analytic)

    out = {}
    for method, spec in method_specs:
        scores_method = spec["U"][:, :k] * spec["sv"][:k]
        means_method = {label: scores_method[cell_labels == label, :].mean(axis=0)
                        for label in labels}
        nn_method = nearest_neighbors(means_method)

        diff_a, sd_a, diff_m, sd_m = [], [], [], []
        for i in labels:
            for j in set(nn_method[i]) | set(nn_analytic[i]):
                da, sa = diff_and_sd(means_analytic, scores_analytic, i, j)
                dm, sm = diff_and_sd(means_method, scores_method, i, j)
                diff_a.append(da); sd_a.append(sa)
                diff_m.append(dm); sd_m.append(sm)
        out[method] = (np.array(diff_a), np.array(sd_a),
                       np.array(diff_m), np.array(sd_m))
    return out


def compute_neighbor_distance_ratios(dataset, n_genes=1000, refined=True,
                                     min_metacell_size=None, n_neighbors=5,
                                     HVG_type=None):
    """diff/sd per neighbor pair, in analytic's space (x) and the method's
    (y) -- see compute_neighbor_moments for how the pairs and the two
    quantities are defined.

    Returns {method: (x, y)}.
    """
    moments = compute_neighbor_moments(dataset, n_genes, refined=refined,
                                       min_metacell_size=min_metacell_size,
                                       n_neighbors=n_neighbors, HVG_type=HVG_type)
    return {method: (diff_a / sd_a, diff_m / sd_m)
           for method, (diff_a, sd_a, diff_m, sd_m) in moments.items()}




def make_neighbor_distance_figure(dataset_list=datasets.DEFAULT_DATASETS,
                                  n_genes=1000, refined=True,
                                  min_metacell_size=None, n_neighbors=5, HVG_type=None,
                                  show=True):
    """Each method's per-neighbor-pair diff/sd against analytic's on the
    same pair (see compute_neighbor_distance_ratios), pooled across every
    dataset in dataset_list into one panel per method: a single row of
    four panels (normal/nb/perm/true), rather than one row per dataset.

    HVG_type selects the partition's spike pkl (see load_spike); None
    follows refined.

    Every point is drawn in SCATTER_COLOR and no legend is drawn, so the
    datasets are not distinguished; the scatter is still one call per
    dataset, as in make_normality_spectrum_figure.

    All four panels share one axis range, set from the 99th percentile of
    every method's pooled points together rather than the max, since the
    ratio's tail can run to 10-20x the median (a handful of very
    well-separated pairs) while the bulk sits near 1 -- a max-based limit
    would crush that bulk into a corner. Points beyond the view still
    enter the fit and the relative error, just not the drawing.

    A dashed red least-squares line is fit on ALL of a panel's pooled
    points (not just the ones inside the clipped view), alongside the
    black x = y reference.

    Each panel is annotated in its lower right with "REL ERR: value",
    where value is the per-pair median relative error |y - x| / |x|
    (median_relative_error) computed separately for every dataset in the
    panel and then the mean of those over datasets -- a mean of
    per-dataset medians, so each dataset counts once regardless of how
    many pairs it contributes.
    """
    methods = [method for method, _ in NORMALITY_METHODS]

    ratios_by_dataset = {
        dataset: compute_neighbor_distance_ratios(
            dataset, n_genes, refined=refined, min_metacell_size=min_metacell_size,
            n_neighbors=n_neighbors, HVG_type=HVG_type)
        for dataset in dataset_list}
    pooled = {
        method: (np.concatenate([ratios_by_dataset[d][method][0] for d in dataset_list]),
                 np.concatenate([ratios_by_dataset[d][method][1] for d in dataset_list]))
        for method in methods}

    hi = max(np.percentile(np.concatenate([x, y]), 99)
            for x, y in pooled.values()) * 1.05
    lim = [0, hi]

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                             squeeze=False)

    for j, method in enumerate(methods):
        ax = axes[0, j]
        rel_err_by_dataset = []
        for dataset in dataset_list:
            x, y = ratios_by_dataset[dataset][method]
            rel_err_by_dataset.append(median_relative_error(x, y))
            ax.scatter(x, y, color=SCATTER_COLOR, s=6, alpha=0.3, edgecolors="none")
        ax.plot(lim, lim, color="black", ls="--", lw=1)

        # fitted on every pooled point, not just the ones inside the view,
        # so the clipped axis range cannot flatter or bias the fit
        x, y = pooled[method]
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(lim, [slope * lim[0] + intercept, slope * lim[1] + intercept],
                color="red", ls="--", lw=1.5)

        ax.text(0.97, 0.03, f"REL ERR: {np.mean(rel_err_by_dataset):.3g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.tick_params(labelsize=13)
        # the y label alone names the panel's method; there is no title
        ax.set_xlabel(f"{method_label('analytic')} diff/sd",
                      fontsize=SCATTER_LABEL_FONTSIZE)
        ax.set_ylabel(f"{method_label(method)} diff/sd",
                      fontsize=SCATTER_LABEL_FONTSIZE)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def make_neighbor_sd_figure(dataset_list=datasets.DEFAULT_DATASETS,
                            n_genes=1000, refined=True, min_metacell_size=None,
                            n_neighbors=5, show=True):
    """The two halves of the diff/sd ratio, plotted separately against
    analytic on the same neighbor pairs (see compute_neighbor_moments):
    a 2 x 4 grid, one column per method, top row the sd along the
    mu - mu' direction and bottom row ||mu - mu'|| itself. Every dataset in
    dataset_list is pooled into each panel, coloured by dataset.

    make_neighbor_distance_figure plots their ratio, which can hide a
    method matching analytic for the wrong reason -- inflating spread and
    separation together leaves the ratio unchanged. Splitting them shows
    which of the two actually moved.

    Both axes of both rows are log-scaled and square, with a row-shared
    range: sd spans a couple of decades across datasets, so a linear axis
    would collapse the smaller ones. The dashed red line is a least-squares
    fit in LOG space (so it is a power law, y = e^c * x^s, not a straight
    line in the raw units), fitted on every pooled point.
    """
    methods = [method for method, _ in NORMALITY_METHODS]
    rows = [("sd along mu - mu'", 1, 3), (r"$\|\mu - \mu'\|$", 0, 2)]

    moments = {dataset: compute_neighbor_moments(
                   dataset, n_genes, refined=refined,
                   min_metacell_size=min_metacell_size, n_neighbors=n_neighbors)
              for dataset in dataset_list}

    cmap = plt.get_cmap("tab10" if len(dataset_list) <= 10 else "tab20")
    colors = {dataset: cmap(i % cmap.N) for i, dataset in enumerate(dataset_list)}

    fig, axes = plt.subplots(len(rows), len(methods),
                             figsize=(5 * len(methods), 4.6 * len(rows)),
                             squeeze=False)

    for i, (label, ia, im) in enumerate(rows):
        pooled = np.concatenate([moments[d][m][idx] for d in dataset_list
                                for m in methods for idx in (ia, im)])
        lim = [pooled.min() / 1.5, pooled.max() * 1.5]

        for j, method in enumerate(methods):
            ax = axes[i, j]
            n_points = 0
            for dataset in dataset_list:
                x, y = moments[dataset][method][ia], moments[dataset][method][im]
                n_points += len(x)
                ax.scatter(x, y, color=colors[dataset], s=6, alpha=0.4,
                          edgecolors="none", label=display_name(dataset))
            ax.plot(lim, lim, color="black", ls="--", lw=1, label="x = y")

            # the fit is in log space, matching the log axes -- fitting in
            # raw units would be dominated by the largest points
            x_all = np.concatenate([moments[d][method][ia] for d in dataset_list])
            y_all = np.concatenate([moments[d][method][im] for d in dataset_list])
            slope, intercept = np.polyfit(np.log(x_all), np.log(y_all), 1)
            grid = np.geomspace(lim[0], lim[1], 100)
            ax.plot(grid, np.exp(intercept) * grid ** slope, color="red", ls="--",
                    lw=1.5, label="fit (log-log)")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.tick_params(labelsize=12)
            ax.set_xlabel(f"analytic {label}")
            ax.set_ylabel(f"{method} {label}")
            if i == 0:
                ax.set_title(f"{method} ({n_points} points)", fontsize=20)
            if j == 0:
                ax.set_ylabel(f"{label}\n{method}", fontsize=15)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, markerscale=2.5, loc="upper left")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


### k-NN figures


def compute_knn_edge_fractions(dataset, n_genes, k_nn=10, refined=True, HVG_type=None):
    """{method: edge fractions} for one (dataset, n_genes), analytic included.

    Each method's per-cell k-NN indices are aggregated to a mixture x mixture
    graph, row-normalized, and flattened over the OFF-DIAGONAL entries only
    (analysis_knn.build_mixture_graph / normalize_adjacency_rows) -- so entry
    [a, b] is the fraction of mixture a's k-NN edges landing in mixture b,
    and the result is about how much metacells bleed into each other rather
    than about within-metacell structure.

    Every method shares the pkl's one mixture_order, so the arrays are
    row-aligned across methods and can be scattered against each other.

    HVG_type selects the partition's spike and knn pkls (see load_spike and
    load_knn); None follows refined.

    A method the pkl does not carry is simply absent from the returned dict;
    the caller blanks that panel. Returns None if the pkl cannot be built or
    read at all.
    """
    if not has_spike(dataset, n_genes, refined=refined, HVG_type=HVG_type):
        return None
    try:
        knn_result = load_knn(dataset, n_genes, k_nn=k_nn, refined=refined, HVG_type=HVG_type)
    except ValueError as e:
        print(f"  SKIPPING {dataset}/{n_genes}: {e}")
        return None

    cell_labels = knn_result["cell_labels"]
    mixture_order = knn_result["mixture_order"]
    off_diagonal = ~np.eye(len(mixture_order), dtype=bool)

    out = {}
    for method in ak.KNN_METHODS:
        key = f"indices_{method}"
        if key not in knn_result:
            continue
        out[method] = ak.normalize_adjacency_rows(ak.build_mixture_graph(
            knn_result[key], cell_labels, mixture_order))[off_diagonal]
    return out


def compute_knn_edge_fraction_pairs(dataset, n_genes_list=datasets.DEFAULT_N_GENES,
                                    k_nn=10, refined=True, HVG_type=None):
    """{method: (x, y)} for one dataset: analytic's k-NN mixture-pair edge
    fractions (x) against the method's on the same pairs (y), as returned by
    compute_knn_edge_fractions, pooled over every n_genes in n_genes_list
    that has a readable pkl (only 1000 is cached in practice).

    HVG_type selects the partition's spike pkl (see load_spike); None
    follows refined.

    A method absent from every pkl is absent from the dict, and a dataset
    with no readable pkl gives an empty dict, so callers can leave such a
    dataset out rather than raise. Shared by make_knn_figure and
    table.make_knn_table so the two report the same numbers.
    """
    methods = [method for method, _ in NORMALITY_METHODS]
    pooled = {}
    for n_genes in n_genes_list:
        fractions = compute_knn_edge_fractions(dataset, n_genes, k_nn=k_nn,
                                               refined=refined, HVG_type=HVG_type)
        if fractions is None or "analytic" not in fractions:
            continue
        for method in methods:
            if method in fractions:
                xs, ys = pooled.setdefault(method, ([], []))
                xs.append(fractions["analytic"])
                ys.append(fractions[method])
    return {method: (np.concatenate(xs), np.concatenate(ys))
            for method, (xs, ys) in pooled.items()}


def knn_relative_error(x, y):
    """The per-dataset statistic of the knn figure and table: the median
    relative error |y - x| / |x| (median_relative_error) of a method's edge
    fractions y against analytic's x, over the pairs with x > 0 only, since
    the ratio is undefined where the analytic fraction is zero. Pairs the
    analytic model gives no edges but the method does are left out. NaN if
    no pair has x > 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nonzero = x > 0
    return median_relative_error(x[nonzero], y[nonzero]) if nonzero.any() else np.nan


def knn_replicate_filename(dataset, n_genes, refined=True):
    """Path of the cached replicate errors, beside the dataset's knn pkl."""
    out_dir = ak.KNN_REFINED_DIR if refined else ak.KNN_DIR
    return os.path.join(out_dir, f"{dataset}_HVG_{n_genes}_replicates.pkl")


def compute_knn_replicate_errors(dataset, n_genes=1000, n_rep=20, k_nn=10,
                                 refined=True, seed=0, overwrite=False):
    """knn_relative_error of n_rep fresh draws of Z_normal against the cached
    analytic embedding, one value per draw: the spread of the table's Normal
    entry, which is itself one such draw. Cached in knn_replicate_filename;
    a cache built with a different n_rep, k_nn, seed or replicate kind is
    rebuilt.

    Each replicate is built exactly as analysis_spike.anndata2spike builds
    spec_n, the pkl entry behind the "normal" method: the spike S and the
    per-entry noise variances of B come from the dataset's fitted nb model
    (lwa.log_normalized_spiked_mixture_model), Z = S + B_normal is drawn with
    lwa.simulate_Z_guassian, its SVD taken, and the leading k left singular
    vectors scaled by their singular values form the embedding W (the same
    U * sv that anndata2knn feeds to the k-NN for every method), which then
    gets the same NNDescent k-NN (ak.knn_indices). So unlike a draw from the
    predicted Gaussian mixture in PCA space, a replicate carries the
    finite-sample effect of the PCA step, as the Normal entry does.

    The replicate's rows follow the spike's own sample_labels order rather
    than the h5ad's; the mixture graph only needs each row's label, so
    nothing is realigned. The sampler uses numpy's global RNG, seeded with
    seed + replicate index for reproducibility.

    Returns the errors as an array, or None if the dataset has no spike or
    knn pkl (as compute_knn_edge_fractions).
    """
    kind = "Z_normal"
    cache_f = knn_replicate_filename(dataset, n_genes, refined=refined)
    if os.path.isfile(cache_f) and not overwrite:
        with open(cache_f, "rb") as f:
            cache = pickle.load(f)
        if ((cache["n_rep"], cache["k_nn"], cache["seed"], cache.get("kind"))
                == (n_rep, k_nn, seed, kind)):
            return cache["errors"]

    if not asp.has_spike_result(dataset, n_genes, refined=refined):
        return None
    try:
        knn_result = ak.anndata2knn(dataset, n_genes, k_nn=k_nn, refined=refined)
    except ValueError as e:
        print(f"  SKIPPING {dataset}/{n_genes}: {e}")
        return None

    s = (dr.load_HVG_refined_dataset(dataset, n_genes) if refined
         else datasets.load_HVG_dataset(dataset, n_genes))
    nb = s.uns["nb_model"]
    m_model, spike = lwa.log_normalized_spiked_mixture_model(nb, s.n_obs, include_spike=True)
    sample_labels = np.asarray(spike["sample_labels"])

    mixture_order = knn_result["mixture_order"]
    if set(sample_labels) != set(mixture_order):
        raise ValueError(f"{dataset}: the nb model's mixtures differ from the knn pkl's")
    k = knn_result["k"]

    off_diagonal = ~np.eye(len(mixture_order), dtype=bool)
    x = ak.normalize_adjacency_rows(ak.build_mixture_graph(
        knn_result["indices_analytic"], knn_result["cell_labels"], mixture_order))[off_diagonal]

    errors = np.empty(n_rep)
    for i in range(n_rep):
        print(f"  knn replicate {i + 1} of {n_rep} for {dataset}/{n_genes}", flush=True)
        np.random.seed(seed + i)
        Z = lwa.simulate_Z_guassian(m_model, spike)
        U, sv, _ = np.linalg.svd(Z, full_matrices=False)
        W = U[:, :k] * sv[:k]
        y = ak.normalize_adjacency_rows(ak.build_mixture_graph(
            ak.knn_indices(W, k_nn), sample_labels, mixture_order))[off_diagonal]
        errors[i] = knn_relative_error(x, y)

    with open(cache_f, "wb") as f:
        pickle.dump({"errors": errors, "n_rep": n_rep, "k_nn": k_nn, "seed": seed,
                     "kind": kind}, f)
    return errors


# the k-NN figure's axis label and panel title sizes: 50% over matplotlib's
# defaults of 10 ("medium") and 12 ("large"), which _plot_adjacency_values
# otherwise leaves in place for analysis_knn's own diagnostic plots
KNN_LABEL_FONTSIZE = 15
KNN_TITLE_FONTSIZE = 18


def make_knn_figure(dataset_list=datasets.DEFAULT_DATASETS,
                    n_genes_list=datasets.DEFAULT_N_GENES, k_nn=10,
                    refined=True, HVG_type=None, show=True):
    """Each method's k-NN mixture-pair edge fractions against analytic's on
    the same pairs, pooled across every dataset in dataset_list into one
    panel per method: a single row of four panels (normal/nb/perm/true), as
    in make_neighbor_distance_figure, rather than one row per dataset.

    Note the axis convention is the REVERSE of analysis_knn.plot_knn, which
    puts the panel's method on x and one fixed reference on y. Here analytic
    is always x and the panel's method is y, so the columns are four
    different y axes read against one common reference.

    Panels are drawn by analysis_knn._plot_adjacency_values, so they carry
    that function's cube-root axes, y = x line and linear-space red dashed
    fit curve, the fit taken on the pooled points of every dataset -- the
    edge fractions pile up near zero (only ~3% of pairs reach 0.1), which a
    linear axis would crush into the corner. No groups are passed, so every
    point is drawn in that function's single colour and no legend is drawn:
    the datasets are not distinguished. Each panel's axis range is set from
    its own points, as that function does.

    Its axis labels and panel titles are drawn at KNN_LABEL_FONTSIZE and
    KNN_TITLE_FONTSIZE; the panels are titled by method and there is no
    figure-level title.

    Its fit summary text is replaced by "REL ERR: value" in the lower right,
    as in make_normality_spectrum_figure and make_neighbor_distance_figure:
    the per-pair median relative error |y - x| / |x| over the pairs with
    x > 0 (knn_relative_error) computed separately for every dataset in the
    panel and then the mean of those over datasets -- a mean of per-dataset
    medians, so each dataset counts once regardless of how many mixture
    pairs it contributes. Pairs the analytic model gives no edges but the
    method does are drawn (on the y axis) but do not enter the value. The
    per-dataset values themselves are what table.make_knn_table tabulates.

    HVG_type selects the partition's spike pkl (see load_spike); None
    follows refined.

    Several n_genes are pooled for a dataset if n_genes_list has more than
    one entry (compute_knn_edge_fraction_pairs); only n_genes=1000 is
    cached in practice.

    A dataset with no readable k-NN pkl is left out of every panel, and a
    method absent from every pkl blanks just that panel,
    rather than raising -- so a partly built cache still plots.
    """
    methods = [method for method, _ in NORMALITY_METHODS]

    # per method: one x and y array per contributing dataset, plus that
    # dataset's relative error, all appended in dataset_list order
    pooled = {method: {"x": [], "y": [], "rel_err": []} for method in methods}
    for dataset in dataset_list:
        pairs = compute_knn_edge_fraction_pairs(dataset, n_genes_list, k_nn=k_nn,
                                                refined=refined, HVG_type=HVG_type)
        for method, (x, y) in pairs.items():
            pooled[method]["x"].append(x)
            pooled[method]["y"].append(y)
            pooled[method]["rel_err"].append(knn_relative_error(x, y))

    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 5.5),
                             squeeze=False)

    for j, method in enumerate(methods):
        ax = axes[0, j]
        entry = pooled[method]
        if not entry["x"]:
            ax.axis("off")
            ax.set_title(f"{method} (absent)", fontsize=12, color="grey")
            continue

        ak._plot_adjacency_values(
            ax, np.concatenate(entry["x"]), np.concatenate(entry["y"]),
            xlabel=f"fraction of edges ({method_label('analytic')})",
            ylabel=f"fraction of edges ({method_label(method)})",
            title=method_label(method),
            annotation=f"REL ERR: {np.nanmean(entry['rel_err']):.3g}",
            label_fontsize=KNN_LABEL_FONTSIZE, title_fontsize=KNN_TITLE_FONTSIZE)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
