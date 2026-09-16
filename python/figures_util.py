import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import analysis_bulk as ab
import analysis_knn as ak
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
    B: the SNR of each gene panel's own new genes.

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
                        title="SNR" if i == 0 else None)

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
                                min_metacell_size=None):
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

    Returns {method: (x, y)}, parallel arrays of length n_metacells * k --
    x analytic's eigenvalues, y the method's.
    """
    result = asp.load_spike_result(dataset, n_genes, refined=refined)
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


def compute_outlier_spectrum(dataset, n_genes, refined=True, methods=None):
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
    result = asp.load_spike_result(dataset, n_genes, refined=refined)
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


def make_normality_spectrum_figure(dataset_list=datasets.DEFAULT_DATASETS,
                                   n_genes=1000, refined=True,
                                   min_metacell_size=None, methods=None,
                                   show=True):
    """Each method's per-metacell covariance eigenvalues against analytic's
    on the same metacell and the same rank, pooled across every dataset in
    dataset_list into one panel per method: a single row with one panel
    per entry of methods, rather than one row per dataset.

    methods lists which of compute_eigenvalue_spectrum's methods ("normal",
    "nb", "perm", "true") get a panel, in that left-to-right order; None
    draws all four.

    Each panel is annotated in its lower right with "REL ERR: value",
    where value is the per-point median relative error |y - x| / |x|
    (median_relative_error) computed separately for every dataset in the
    panel and then the mean of those over datasets -- a mean of
    per-dataset medians, so each dataset counts once regardless of how
    many points it contributes.

    Points are coloured by dataset (a legend on the first panel maps colour
    to dataset), so points from the same dataset are not merged into the
    same scatter -- this is a per-dataset plot call, not a pooled one,
    unlike the earlier version of this figure.

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
            dataset, n_genes, refined=refined, min_metacell_size=min_metacell_size)
        for dataset in dataset_list}

    # the shared axis range only looks at the methods actually drawn, so a
    # left-out method cannot stretch the limits of the panels that remain
    drawn = [spectrum[method] for spectrum in spectra_by_dataset.values()
             for method in methods]
    lo = min(min(x.min(), y.min()) for x, y in drawn) / 1.5
    hi = max(max(x.max(), y.max()) for x, y in drawn) * 1.5
    lim = [lo, hi]

    # tab10/tab20 are qualitative palettes meant for exactly this many
    # categories; falls back to cycling tab20 if there are more datasets
    # than either provides
    cmap = plt.get_cmap("tab10" if len(dataset_list) <= 10 else "tab20")
    colors = {dataset: cmap(i % cmap.N) for i, dataset in enumerate(dataset_list)}

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                             squeeze=False)

    for j, method in enumerate(methods):
        ax = axes[0, j]
        rel_err_by_dataset = []
        for dataset in dataset_list:
            x, y = spectra_by_dataset[dataset][method]
            rel_err_by_dataset.append(median_relative_error(x, y))
            ax.scatter(x, y, color=colors[dataset], s=6, alpha=0.4,
                      edgecolors="none", label=display_name(dataset))
        ax.plot(lim, lim, color="black", ls="--", lw=1)
        ax.text(0.97, 0.03, f"REL ERR: {np.mean(rel_err_by_dataset):.3g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.tick_params(labelsize=13)
        # the y label alone names the panel's method; there is no title.
        # 13 is 30% over matplotlib's default "medium" label size of 10
        ax.set_xlabel(f"{method_label('analytic')} variance", fontsize=13)
        ax.set_ylabel(f"{method_label(method)} variance", fontsize=13)
        if j == 0:
            ax.legend(fontsize=7, markerscale=2.5, loc="upper left")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def compute_normality_KS(dataset, n_genes=1000, HVG_type="refined", min_metacell_size=None):
    """Per-metacell, per-eigenvector-axis KS statistic, self-referential
    for each spectrum: analytic, normal, nb, perm, true each projected onto
    their OWN metacell covariance eigenvectors, not a shared or rotated
    basis.

    For one metacell and one spectrum, this is exactly
    normality.metacell_normality(X, return_per_axis=True): X's own
    covariance eigenvectors (descending eigenvalue), X projected onto them,
    and normality.max_ecdf_normal_diff (the KS statistic against a normal
    fitted to that projection's own mean/std) evaluated on every axis
    rather than maxed over them -- so one metacell contributes k values,
    not one.

    HVG_type selects which spike pkl the spectra are read from: "refined"
    and "unrefined" both come from analysis_spike.load_spike_result (the
    old refined=True/False), "celltype" from
    analysis_spike_celltype.load_spike_celltype_result, whose metacells are
    cell types (see datasets_celltype.py) rather than SuperCell/refinement
    groups. All three return a dict with the same "cell_labels",
    "spec_analytic", "spec_n", "spec_nb", "spec_perm", "spec_true" keys, so
    nothing below this dispatch needs to know which one it got.

    Uses the same k and the same min_metacell_size floor (default k + 1) as
    the other normality figures, for the same reason: below it the
    covariance is rank deficient and its trailing eigenvectors are
    arbitrary.

    Returns {spectrum: values}, one flat array per spectrum ("analytic",
    "normal", "nb", "perm", "true") of length n_metacells * k, pooling
    every metacell's per-axis values together.
    """
    if HVG_type == "celltype":
        result = asc.load_spike_celltype_result(dataset, n_genes)
    elif HVG_type in ("refined", "unrefined"):
        result = asp.load_spike_result(dataset, n_genes, refined=(HVG_type == "refined"))
    else:
        raise ValueError(f"HVG_type must be 'refined', 'unrefined', or 'celltype', "
                         f"got {HVG_type!r}")
    cell_labels = np.asarray(result["cell_labels"])
    spec_analytic = result["spec_analytic"]
    all_specs = [("analytic", spec_analytic)] + [(method, result[key])
                                                  for method, key in NORMALITY_METHODS]

    k = min(len(spec["sv"]) for _, spec in all_specs)
    if min_metacell_size is None:
        min_metacell_size = k + 1

    labels, sizes = np.unique(cell_labels, return_counts=True)
    labels = labels[sizes >= min_metacell_size]

    out = {}
    for name, spec in all_specs:
        scores = spec["U"][:, :k] * spec["sv"][:k]
        values = [nrm.metacell_normality(scores[cell_labels == label, :], return_per_axis=True)
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
    metacells (and eigenvector axis) of the KS statistic against a normal
    fitted along that axis (see compute_normality_KS), with the chosen
    spectra as boxplots side by side in a single panel per dataset, laid
    out with n_cols dataset panels per row. Outlier points are suppressed
    (showfliers=False) since these pool thousands of values and would
    otherwise be mostly a cloud of dots.

    HVG_type is "refined", "unrefined", or "celltype" -- see
    compute_normality_KS for what each one loads.

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
        ax.tick_params(labelsize=13)
        ax.set_title(f"{display_name(dataset)} ({len(values[0])} points)", fontsize=20)
        if i % n_cols == 0:
            ax.set_ylabel("max |ECDF - normal CDF|", fontsize=14)

    # a dataset count that does not fill the last row leaves empty panels
    for j in range(len(dataset_list), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


### Neighbor distance figures


def compute_neighbor_moments(dataset, n_genes=1000, refined=True,
                             min_metacell_size=None, n_neighbors=5):
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

    Returns {method: (diff_analytic, sd_analytic, diff_method, sd_method)},
    parallel arrays with n_neighbors to 2 * n_neighbors entries per
    metacell, depending on how much the two spaces' neighbor lists overlap.
    """
    result = asp.load_spike_result(dataset, n_genes, refined=refined)
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
                                     min_metacell_size=None, n_neighbors=5):
    """diff/sd per neighbor pair, in analytic's space (x) and the method's
    (y) -- see compute_neighbor_moments for how the pairs and the two
    quantities are defined.

    Returns {method: (x, y)}.
    """
    moments = compute_neighbor_moments(dataset, n_genes, refined=refined,
                                       min_metacell_size=min_metacell_size,
                                       n_neighbors=n_neighbors)
    return {method: (diff_a / sd_a, diff_m / sd_m)
           for method, (diff_a, sd_a, diff_m, sd_m) in moments.items()}




def make_neighbor_distance_figure(dataset_list=datasets.DEFAULT_DATASETS,
                                  n_genes=1000, refined=True,
                                  min_metacell_size=None, n_neighbors=5, show=True):
    """Each method's per-neighbor-pair diff/sd against analytic's on the
    same pair (see compute_neighbor_distance_ratios), pooled across every
    dataset in dataset_list into one panel per method: a single row of
    four panels (normal/nb/perm/true), rather than one row per dataset.

    Points are coloured by dataset (a legend on the first panel maps colour
    to dataset; the x = y and fit lines are unlabelled so the legend lists
    only datasets), one scatter call per dataset rather than one pooled
    call, as in make_normality_spectrum_figure.

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
            n_neighbors=n_neighbors)
        for dataset in dataset_list}
    pooled = {
        method: (np.concatenate([ratios_by_dataset[d][method][0] for d in dataset_list]),
                 np.concatenate([ratios_by_dataset[d][method][1] for d in dataset_list]))
        for method in methods}

    hi = max(np.percentile(np.concatenate([x, y]), 99)
            for x, y in pooled.values()) * 1.05
    lim = [0, hi]

    # tab10/tab20 are qualitative palettes meant for exactly this many
    # categories; falls back to cycling tab20 if there are more datasets
    # than either provides
    cmap = plt.get_cmap("tab10" if len(dataset_list) <= 10 else "tab20")
    colors = {dataset: cmap(i % cmap.N) for i, dataset in enumerate(dataset_list)}

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                             squeeze=False)

    for j, method in enumerate(methods):
        ax = axes[0, j]
        rel_err_by_dataset = []
        for dataset in dataset_list:
            x, y = ratios_by_dataset[dataset][method]
            rel_err_by_dataset.append(median_relative_error(x, y))
            ax.scatter(x, y, color=colors[dataset], s=6, alpha=0.3,
                       edgecolors="none", label=display_name(dataset))
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
        # the y label alone names the panel's method; there is no title.
        # 13 matches make_normality_spectrum_figure's label size
        ax.set_xlabel(f"{method_label('analytic')} diff/sd", fontsize=13)
        ax.set_ylabel(f"{method_label(method)} diff/sd", fontsize=13)
        if j == 0:
            ax.legend(fontsize=7, markerscale=2.5, loc="upper left")

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


def compute_knn_edge_fractions(dataset, n_genes, k_nn=10, refined=True):
    """{method: edge fractions} for one (dataset, n_genes), analytic included.

    Each method's per-cell k-NN indices are aggregated to a mixture x mixture
    graph, row-normalized, and flattened over the OFF-DIAGONAL entries only
    (analysis_knn.build_mixture_graph / normalize_adjacency_rows) -- so entry
    [a, b] is the fraction of mixture a's k-NN edges landing in mixture b,
    and the result is about how much metacells bleed into each other rather
    than about within-metacell structure.

    Every method shares the pkl's one mixture_order, so the arrays are
    row-aligned across methods and can be scattered against each other.

    A method the pkl does not carry is simply absent from the returned dict;
    the caller blanks that panel. Returns None if the pkl cannot be built or
    read at all.
    """
    if not asp.has_spike_result(dataset, n_genes, refined=refined):
        return None
    try:
        knn_result = ak.anndata2knn(dataset, n_genes, k_nn=k_nn, refined=refined)
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
                                    k_nn=10, refined=True):
    """{method: (x, y)} for one dataset: analytic's k-NN mixture-pair edge
    fractions (x) against the method's on the same pairs (y), as returned by
    compute_knn_edge_fractions, pooled over every n_genes in n_genes_list
    that has a readable pkl (only 1000 is cached in practice).

    A method absent from every pkl is absent from the dict, and a dataset
    with no readable pkl gives an empty dict, so callers can leave such a
    dataset out rather than raise. Shared by make_knn_figure and
    table.make_knn_table so the two report the same numbers.
    """
    methods = [method for method, _ in NORMALITY_METHODS]
    pooled = {}
    for n_genes in n_genes_list:
        fractions = compute_knn_edge_fractions(dataset, n_genes, k_nn=k_nn,
                                               refined=refined)
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


def make_knn_figure(dataset_list=datasets.DEFAULT_DATASETS,
                    n_genes_list=datasets.DEFAULT_N_GENES, k_nn=10,
                    refined=True, show=True):
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
    linear axis would crush into the corner. Points are coloured by dataset
    through its groups/group_colors arguments, and the first panel carries
    a legend of the datasets that contributed (the x = y line, which that
    function also labels, is kept out of it). Each panel's axis range is
    set from its own points, as that function does.

    Its fit summary text is replaced by "REL ERR: value" in the lower right,
    as in make_normality_spectrum_figure and make_neighbor_distance_figure:
    the per-pair median relative error |y - x| / |x| over the pairs with
    x > 0 (knn_relative_error) computed separately for every dataset in the
    panel and then the mean of those over datasets -- a mean of per-dataset
    medians, so each dataset counts once regardless of how many mixture
    pairs it contributes. Pairs the analytic model gives no edges but the
    method does are drawn (on the y axis) but do not enter the value. The
    per-dataset values themselves are what table.make_knn_table tabulates.

    Several n_genes are pooled for a dataset if n_genes_list has more than
    one entry (compute_knn_edge_fraction_pairs); only n_genes=1000 is
    cached in practice.

    A dataset with no readable k-NN pkl is left out of every panel (and of
    the legend), and a method absent from every pkl blanks just that panel,
    rather than raising -- so a partly built cache still plots.
    """
    methods = [method for method, _ in NORMALITY_METHODS]

    # per method: one x, y and dataset-label array per contributing dataset,
    # plus that dataset's relative error, all appended in dataset_list order
    pooled = {method: {"x": [], "y": [], "ds": [], "rel_err": []}
              for method in methods}
    plotted_datasets = []
    for dataset in dataset_list:
        pairs = compute_knn_edge_fraction_pairs(dataset, n_genes_list, k_nn=k_nn,
                                                refined=refined)
        if pairs:
            plotted_datasets.append(dataset)
        for method, (x, y) in pairs.items():
            pooled[method]["x"].append(x)
            pooled[method]["y"].append(y)
            pooled[method]["ds"].append(np.full(x.shape, display_name(dataset)))
            pooled[method]["rel_err"].append(knn_relative_error(x, y))

    # tab10/tab20 are qualitative palettes meant for exactly this many
    # categories, as in make_neighbor_distance_figure; indexed by position
    # in dataset_list so a dataset keeps its colour if a later one drops out.
    # Keyed by display name, which is what the group labels above carry.
    cmap = plt.get_cmap("tab10" if len(dataset_list) <= 10 else "tab20")
    colors = {display_name(dataset): cmap(i % cmap.N)
              for i, dataset in enumerate(dataset_list)}

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
            groups=np.concatenate(entry["ds"]), group_colors=colors,
            xlabel=f"fraction of edges ({method_label('analytic')})",
            ylabel=f"fraction of edges ({method_label(method)})",
            title=method_label(method),
            annotation=f"REL ERR: {np.nanmean(entry['rel_err']):.3g}")
        if j == 0:
            # the dataset scatters only: _plot_adjacency_values also labels
            # its x = y line, which would otherwise join the legend
            handles, labels = ax.get_legend_handles_labels()
            shown = [(h, l) for h, l in zip(handles, labels) if l in colors]
            ax.legend([h for h, _ in shown], [l for _, l in shown],
                      fontsize=7, loc="upper left")

    fig.suptitle(f"k-NN mixture-pair edge fractions: {method_label('analytic')} (x) vs each "
                 f"method (y)\n{'refined' if refined else 'unrefined'} "
                 f"partition, k_nn={k_nn}, {len(plotted_datasets)} datasets",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    if show:
        plt.show()
    return fig
