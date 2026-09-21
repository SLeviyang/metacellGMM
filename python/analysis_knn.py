# Assumptions:
# - mixture_labels is a length-n array giving the mixture/metacell identity of each
#   row, shared by every spec in an analysis_spike pkl (their rows are aligned one to
#   one, including spec_analytic -- anndata2spike realigns it).
# - Every empirical PCA is truncated to the same number of components k as the
#   analytic one.
# - k_nn=10 nearest neighbors are used, excluding the query point itself.

import os
import pickle

import numpy as np
from pynndescent import NNDescent
import matplotlib.pyplot as plt

import datasets
import datasets_refined as dr
import analysis_spike as asp


# knn pkls are split by provenance the same way the spike pkls they read are:
# refined ones come from analysis_spike_refined / analysis_HVG_refined,
# unrefined from analysis_spike / analysis_HVG
KNN_DIR = "../analysis_knn/"
KNN_REFINED_DIR = "../analysis_knn_refined/"

# every spec in the pkl, with analytic first: it is the reference the others
# are compared against.
KNN_METHODS = ["analytic"] + list(asp.SPIKE_METHODS)

# the methods plot_knn draws, independent of which method is on the y axis:
# the five real embeddings, plus the rank-0 simulations of perm and true (the
# analytic model projected onto that method's basis with no covariance
# correction), so a simulated stand-in sits beside the thing it stands in for.
# The higher-rank <base>_proj_i are cached by anndata2knn but not drawn.
# Whichever method is the y axis also appears as a panel, giving a trivial
# y = x -- kept rather than dropped so the grid is the same in every call.
PLOT_METHODS = ["analytic", "normal", "nb", "perm", "true",
                "perm_proj_0", "true_proj_0"]
PLOT_GRID = (4, 2)   # 8 cells for 7 panels; the last is blanked

# the methods anndata2knn also builds a split-half k-NN for (see
# split_half_rows): these are the ones plot_knn uses as its y axis, and the
# split gives that axis a noise floor to be read against.
SPLIT_METHODS = ["true", "perm"]

def knn_indices(W, k_nn=10, random_state=0):
    """Approximate k_nn nearest neighbors of each row of W within W itself
    (self excluded), via pynndescent's NNDescent.

    This replaces an earlier exact sklearn NearestNeighbors implementation:
    once W's dimensionality gets into the tens (as it does for several of
    the analysis_spike.py embeddings, e.g. k=37 for Zebrafish/5000), a tree
    search degrades toward brute force's O(n^2) cost, which is prohibitively
    slow for the tens-of-thousands-of-cells datasets this is run on.
    NNDescent instead runs in roughly O(n log n), at the cost of the
    neighbor lists being approximate rather than exact -- acceptable here
    since this feeds a diagnostic edge-count comparison, not something
    requiring exact neighbor identity. random_state makes the result
    reproducible across runs.
    """
    index = NNDescent(W, n_neighbors=k_nn + 1, random_state=random_state)
    indices, _ = index.neighbor_graph
    return indices[:, 1:]


def split_half_rows(mixture_labels, random_state=0):
    """Split the rows into two halves, each holding half of EVERY metacell.

    The split is stratified rather than global so both halves see the same
    metacells in the same proportions: a global shuffle could leave a small
    metacell badly unbalanced, and a metacell absent from one half would give
    it an all-zero adjacency row (and so a divide-by-zero in
    normalize_adjacency_rows). A metacell with an odd count sends the extra
    cell to the first half, so the halves differ by at most one cell per
    metacell.

    Returns (rows_a, rows_b), each an index array into mixture_labels, sorted.
    random_state makes the split reproducible, so a cached split is stable
    across runs.
    """
    mixture_labels = np.asarray(mixture_labels)
    rng = np.random.default_rng(random_state)

    rows_a, rows_b = [], []
    for label in np.unique(mixture_labels):
        idx = np.where(mixture_labels == label)[0]
        shuffled = rng.permutation(idx)
        half = int(np.ceil(len(idx) / 2))
        rows_a.append(shuffled[:half])
        rows_b.append(shuffled[half:])

    return (np.sort(np.concatenate(rows_a)), np.sort(np.concatenate(rows_b)))


def mixture_order_from_labels(mixture_labels):
    mixture_labels = np.asarray(mixture_labels)
    _, first_idx = np.unique(mixture_labels, return_index=True)
    return mixture_labels[np.sort(first_idx)]


def build_mixture_graph(indices, mixture_labels, mixture_order=None):
    mixture_labels = np.asarray(mixture_labels)
    if mixture_order is None:
        mixture_order = mixture_order_from_labels(mixture_labels)

    M = len(mixture_order)
    label_to_idx = {mix: i for i, mix in enumerate(mixture_order)}

    row_idx = np.repeat(
        np.vectorize(label_to_idx.get)(mixture_labels), indices.shape[1]
    )
    col_idx = np.vectorize(label_to_idx.get)(mixture_labels[indices]).ravel()

    adjacency = np.zeros((M, M), dtype=int)
    np.add.at(adjacency, (row_idx, col_idx), 1)

    return adjacency


def normalize_adjacency_rows(adjacency):
    """Row-normalize an adjacency matrix so each row sums to 1: entry [a, b]
    becomes the fraction of mixture a's k-NN edges that land in mixture b,
    instead of a raw count. Every row of build_mixture_graph's output sums
    to n_a * k_nn by construction (each of a mixture's n_a cells contributes
    exactly k_nn outgoing edges), so this divides each row by that
    row-specific constant -- always positive, since every mixture in
    mixture_order has at least one cell.
    """
    return adjacency / adjacency.sum(axis=1, keepdims=True)


# candidate axis ticks for the cube-root scale, filtered to the visible range
# kept sparse: under x^(1/3) these are roughly evenly spaced on screen, and a
# denser set overlaps at the low end where the transform stretches hardest
_CBRT_TICKS = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0]


def _plot_adjacency_values(ax, x, y, groups=None, group_colors=None,
                            xlabel="x", ylabel="y", title="", annotation="fit",
                            label_fontsize=None, title_fontsize=None):
    """Scatter x vs y on cube-root axes, with a y = x reference and the
    linear-space least-squares fit drawn as a red dashed curve.

    Points at exactly (0, 0) are dropped from the PLOT: they are mixture pairs
    with no k-NN edges under either embedding, they are the large majority of
    the pairs (~86% of off-diagonal entries), and being identical rather than
    merely small no axis transform separates them. The number dropped is
    annotated.

    The REGRESSION is stricter, using only pairs with x > 0 and y > 0. A pair
    with edges under one embedding and none under the other sits on an axis,
    and there are enough such points to drag the line and depress R^2 while
    saying nothing about how the two edge fractions scale together where both
    are present. They are still drawn, just not fitted; the annotation reports
    how many of the plotted points the fit actually used.

    Both axes use x^(1/3) via a "function" scale. The edge fractions pile up
    near zero -- only ~3% of pairs reach 0.1 -- so a linear axis crushes
    everything into the corner, and a log axis cannot represent the exact
    zeros that survive the filter above (a pair with edges one way but not the
    other). Cube root maps 0 to 0 exactly, is smooth and monotone with no
    linear/log junction, and gives the interval (0, 0.1) about 46% of the axis
    instead of 10%.

    The regression and its R^2 are computed on the UNTRANSFORMED data, so they
    describe the linear-space relationship; the line is then sampled densely
    and drawn through the transform, which renders it as a curve. Drawing it
    from two endpoints -- correct on linear axes -- would render a straight
    segment here and misstate the fit.

    If `groups` (an array aligned with x/y) and `group_colors` (dict group ->
    color) are given, points are colored by group (e.g. dataset). The `label`
    kwargs are kept on the artists so a caller can build a figure-level
    legend, but no per-axes legend is drawn.
    instead of a single color.

    `annotation` picks the lower-right text: "fit" (default) is the fit
    summary described above -- slope, intercept, R^2 and the point counts;
    None draws no text; any other string is drawn verbatim in the same
    corner at the larger size the REL ERR annotations of figures_util's
    other figures use, so a caller can put its own statistic there (see
    figures_util.make_knn_figure). The fit line itself is drawn regardless.

    `label_fontsize` and `title_fontsize` set the axis label and title
    sizes; None (the default) leaves matplotlib's own axes.labelsize and
    axes.titlesize in place, so only a caller that asks for larger text
    (figures_util.make_knn_figure) gets it.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~((x == 0) & (y == 0))
    n_dropped = int((~keep).sum())
    x, y = x[keep], y[keep]
    if groups is not None:
        groups = np.asarray(groups)[keep]

    lim = [0, max(x.max(), y.max())] if len(x) else [0, 1]

    # fit in LINEAR space (the transform is display-only) and only on pairs
    # present under BOTH embeddings -- see the docstring on why the on-axis
    # points are plotted but not fitted
    fit_mask = (x > 0) & (y > 0)
    x_fit, y_fit = x[fit_mask], y[fit_mask]
    have_fit = len(x_fit) >= 2
    if have_fit:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        y_pred = slope * x_fit + intercept
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if groups is None:
        ax.scatter(x, y, s=15, alpha=0.6, color="steelblue")
    else:
        for group in sorted(set(groups)):
            mask = groups == group
            ax.scatter(x[mask], y[mask], s=15, alpha=0.6,
                       color=group_colors[group], label=group)

    # np.cbrt is the exact inverse of cubing and handles 0 (and negatives,
    # which the fitted line can reach when the intercept is below 0)
    ax.set_xscale("function", functions=(np.cbrt, lambda v: np.power(v, 3)))
    ax.set_yscale("function", functions=(np.cbrt, lambda v: np.power(v, 3)))

    ax.plot(lim, lim, color="black", ls="--", lw=1, label="x = y")
    if have_fit:
        fit_x = np.linspace(lim[0], lim[1], 400)
        ax.plot(fit_x, slope * fit_x + intercept, color="red", ls="--", lw=1.5)
        fit_annotation = (f"fit (linear space, x>0 & y>0):\n"
                      f"y = {slope:.2f}x + {intercept:.2f}\n"
                      f"$R^2$ = {r2:.2g}\n"
                      f"n fit = {len(x_fit):,} of {len(x):,} plotted\n"
                      f"(0,0) dropped: {n_dropped:,}")
    else:
        fit_annotation = (f"too few points with x>0 & y>0 to fit "
                      f"({len(x_fit):,})\n"
                      f"n plotted = {len(x):,}\n"
                      f"(0,0) dropped: {n_dropped:,}")
    if annotation == "fit":
        ax.text(0.95, 0.05, fit_annotation,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    elif annotation is not None:
        ax.text(0.97, 0.03, annotation,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

    ticks = [t for t in _CBRT_TICKS if lim[0] <= t <= lim[1]]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(labelsize=8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    # an explicit None must not reach matplotlib: Text.set_fontsize(None)
    # falls back to font.size rather than to axes.labelsize / axes.titlesize
    label_kw = {} if label_fontsize is None else {"fontsize": label_fontsize}
    title_kw = {} if title_fontsize is None else {"fontsize": title_fontsize}
    ax.set_xlabel(xlabel, **label_kw)
    ax.set_ylabel(ylabel, **label_kw)
    ax.set_title(title, **title_kw)


def get_knn_filename(dataset, n_genes, refined=False):
    """Path of the knn pkl, creating the folder if it does not exist."""
    out_dir = KNN_REFINED_DIR if refined else KNN_DIR
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{dataset}_HVG_{n_genes}.pkl")


def anndata2knn(dataset, n_genes, k_nn=10, overwrite=False, refined=False):
    """Compute (or load, if already cached) the information needed to define
    the k-NN graph for each of the six analysis_spike.py views (analytic,
    nb, perm, true, gmm, normal) for this dataset/n_genes, caching the
    result to get_knn_filename(dataset, n_genes).

    This caches the raw per-cell k-NN index arrays (the expensive step --
    see knn_indices) rather than the aggregated mixture-level adjacency
    matrices: those are cheap to recompute from the indices via
    build_mixture_graph, while the raw indices stay reusable for more than
    just an adjacency-count comparison (e.g. graph-based clustering).

    Requires the corresponding analysis_spike.py pkl (see
    analysis_spike.get_spike_filename) to already exist.
    """
    out_f = get_knn_filename(dataset, n_genes, refined=refined)
    if os.path.isfile(out_f) and not overwrite:
        with open(out_f, "rb") as f:
            return pickle.load(f)

    if not asp.has_spike_result(dataset, n_genes, refined=refined):
        raise ValueError(f"missing analysis_spike pkl for {dataset} HVG {n_genes} "
                         f"({'refined' if refined else 'unrefined'}): needs both "
                         f"{asp.get_pca_filename(dataset, n_genes, refined=refined)} "
                         f"and {asp.get_spike_filename(dataset, n_genes, refined=refined)}")

    result = asp.load_spike_result(dataset, n_genes, refined=refined)

    # the h5ad must come from the same partition as the pkl -- cell_labels is
    # matched against the spec rows positionally
    s = (dr.load_HVG_refined_dataset(dataset, n_genes) if refined
         else datasets.load_HVG_dataset(dataset, n_genes))
    cell_labels = s.obs["metacell"].to_numpy()
    mixture_order = mixture_order_from_labels(cell_labels)
    k = result["spec_nb"]["U"].shape[1]

    # spec_analytic is realigned to the h5ad cell order in anndata2spike, like
    # every other spec, so no realignment is needed here
    spec_analytic_aligned = result["spec_analytic"]

    # analytic stays special-cased: it is the reference the other methods are
    # compared against and is not in SPIKE_METHODS/SPEC_KEYS. The rest follow
    # SPIKE_METHODS -- NOT KNN_METHODS, which prepends "analytic" and so would
    # look it up in SPEC_KEYS, where it deliberately does not appear -- so
    # methods added in analysis_spike appear here too; a pkl written before a
    # method existed simply lacks its key and is skipped.
    specs = {"analytic": spec_analytic_aligned}
    for method in asp.SPIKE_METHODS:
        key = asp.SPEC_KEYS[method]
        if key in result:
            specs[method] = result[key]
        else:
            print(f"  SKIPPING {method}: {key} not in this spike pkl (rebuild it "
                  f"with overwrite=True to add it)")

    knn_result = {"cell_labels": cell_labels, "mixture_order": mixture_order,
                  "k": k, "k_nn": k_nn}
    tag = f"{dataset} HVG {n_genes} ({'refined' if refined else 'unrefined'})"
    for method, spec in specs.items():
        W = spec["U"][:, :k] * spec["sv"][:k]
        print(f"Computing {method} k-NN for {tag}")
        knn_result[f"indices_{method}"] = knn_indices(W, k_nn)

        # SPLIT_METHODS also get a split-half k-NN: the same embedding cut
        # into two halves of every metacell, each half's neighbours drawn only
        # from itself. Two independent samples of one embedding, so comparing
        # them measures how much of the disagreement between methods is just
        # k-NN sampling noise. Only these methods get one -- they are the ones
        # plot_knn puts on its y axis.
        if method in SPLIT_METHODS:
            rows_a, rows_b = split_half_rows(cell_labels)
            print(f"Computing {method} split-half k-NN for {tag}")
            knn_result[f"{method}_split"] = {
                "rows": (rows_a, rows_b),
                "indices": (knn_indices(W[rows_a], k_nn),
                            knn_indices(W[rows_b], k_nn))}

    with open(out_f, "wb") as f:
        pickle.dump(knn_result, f)

    return knn_result


def make_all_knn_pkl(dataset_names=datasets.DEFAULT_DATASETS,
                     n_genes_list=datasets.DEFAULT_N_GENES,
                     k_nn=10, overwrite=False, refined=False):
    """Build the knn pkl for every (dataset, n_genes) in the given partition.

    A combination whose analysis_spike pkl has not been built is skipped
    silently, since that is the normal state for a partition that has not been
    swept yet; one whose HVG h5ad is missing is skipped with a warning rather
    than aborting the sweep.
    """
    for dataset in dataset_names:
        for n_genes in n_genes_list:
            if not asp.has_spike_result(dataset, n_genes, refined=refined):
                continue
            print(f"Processing {dataset} HVG {n_genes} "
                  f"({'refined' if refined else 'unrefined'})")
            try:
                anndata2knn(dataset, n_genes, k_nn=k_nn, overwrite=overwrite,
                            refined=refined)
            except ValueError as e:
                print(f"  SKIPPING {dataset}/{n_genes}: {e}")
    return None


def plot_knn(dataset_names=datasets.DEFAULT_DATASETS,
             n_genes_list=datasets.DEFAULT_N_GENES, k_nn=10, refined=True,
             y_method="true"):
    """Two figures comparing k-NN mixture-pair edge fractions for one
    partition, returned as (fig, fig_split).

    The first is a panel grid, each panel putting one method's edge fractions
    on x against the reference method's on y. The second is a single panel
    giving that reference a NOISE FLOOR: the y_method embedding is split into
    two halves of every metacell, each half's k-NN graph built from itself
    alone (see split_half_rows and anndata2knn), and the two halves' edge
    fractions plotted against each other. Two independent samples of one
    embedding should agree exactly in expectation, so how far this scatters
    off y = x is how much of the first figure's disagreement is k-NN sampling
    noise rather than a real difference between methods. Only SPLIT_METHODS
    have a cached split, so fig_split is None for any other y_method.

    The halves hold half the cells each, so their k-NN graphs are noisier than
    a full-data one -- the floor is pessimistic, and the panels of the first
    figure are not on identical footing with it.

    y_method selects what goes on the y axis of every panel, "true" by
    default. It may be ANY method anndata2knn cached -- the five real
    embeddings or any simulated <base>_proj_i -- and it does not change the
    panels.

    The panels are always PLOT_METHODS (the five real embeddings plus
    perm_proj_0 and true_proj_0) in a 4 x 2 grid with the last cell blank.
    Whichever method is the y axis also appears as a panel, so that one is a
    trivial y = x; it is kept rather than dropped so the grid reads the same
    across calls. The higher-rank <base>_proj_i are cached but not drawn.

    refined selects which partition to read and plot -- the refined and
    unrefined metacell partitions have different metacells, so their edge
    fractions are not comparable point for point and are not mixed.

    Each adjacency matrix is row-normalized before pooling (see
    normalize_adjacency_rows), so entry [a, b] is the fraction of mixture a's
    k-NN edges landing in mixture b rather than a raw count. That keeps every
    point on the same [0, 1] scale regardless of mixture size, k_nn, or how
    many cells a given dataset/n_genes combo has, so pooling across datasets
    of very different sizes is an apples-to-apples comparison. Only
    off-diagonal entries are used, so this is about how much metacells bleed
    into each other, not about within-metacell structure.

    Reads the per-cell k-NN indices from the analysis_knn cache (see
    anndata2knn), computing and caching it first if it isn't there yet -- only
    the (cheap) mixture-level adjacency aggregation/normalization happens here.
    """
    known = list(KNN_METHODS)
    if y_method not in known:
        raise ValueError(f"y_method {y_method!r} is not a cached method; "
                         f"expected one of {known}")

    plot_methods = list(PLOT_METHODS)

    tag = "refined" if refined else "unrefined"
    pooled = {m: {"x": [], "y": [], "ds": []} for m in plot_methods}
    pooled_split = {"x": [], "y": [], "ds": []}

    for dataset in dataset_names:
        for n_genes in n_genes_list:
            if not asp.has_spike_result(dataset, n_genes, refined=refined):
                continue
            try:
                knn_result = anndata2knn(dataset, n_genes, k_nn=k_nn,
                                         refined=refined)
            except ValueError as e:
                print(f"  SKIPPING {dataset}/{n_genes} ({tag}): {e}")
                continue

            if f"indices_{y_method}" not in knn_result:
                print(f"  SKIPPING {dataset}/{n_genes} ({tag}): no "
                      f"indices_{y_method}, which is the reference axis")
                continue

            cell_labels = knn_result["cell_labels"]
            mixture_order = knn_result["mixture_order"]
            off_diagonal = ~np.eye(len(mixture_order), dtype=bool)

            ref_vals = normalize_adjacency_rows(build_mixture_graph(
                knn_result[f"indices_{y_method}"], cell_labels,
                mixture_order))[off_diagonal]

            for method in plot_methods:
                key = f"indices_{method}"
                if key not in knn_result:
                    continue
                method_vals = normalize_adjacency_rows(build_mixture_graph(
                    knn_result[key], cell_labels, mixture_order))[off_diagonal]
                pooled[method]["x"].append(method_vals)
                pooled[method]["y"].append(ref_vals)
                pooled[method]["ds"].append(np.full(method_vals.shape, dataset))

            # the split-half noise floor for this same y axis; each half is
            # graphed on its own rows, so its adjacency is built from that
            # half's labels rather than the full cell_labels
            split = knn_result.get(f"{y_method}_split")
            if split is not None:
                half_vals = [normalize_adjacency_rows(build_mixture_graph(
                    indices, cell_labels[rows], mixture_order))[off_diagonal]
                    for rows, indices in zip(split["rows"], split["indices"])]
                pooled_split["x"].append(half_vals[0])
                pooled_split["y"].append(half_vals[1])
                pooled_split["ds"].append(np.full(half_vals[0].shape, dataset))

    if not any(pooled[m]["x"] for m in plot_methods):
        raise ValueError(f"no cached analysis_knn pkls found for the {tag} "
                         f"partition with y_method={y_method!r}")

    plotted_datasets = sorted(dataset_names)
    cmap = plt.get_cmap("tab10")
    colors = {dataset: cmap(i % 10) for i, dataset in enumerate(plotted_datasets)}

    nrow, ncol = PLOT_GRID
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 5.5 * nrow),
                             squeeze=False)

    for i, method in enumerate(plot_methods):
        ax = axes[i // ncol, i % ncol]
        entry = pooled[method]
        if not entry["x"]:
            ax.axis("off")
            ax.set_title(f"{method} (absent)", fontsize=12, color="grey")
            continue
        _plot_adjacency_values(
            ax, np.concatenate(entry["x"]), np.concatenate(entry["y"]),
            groups=np.concatenate(entry["ds"]), group_colors=colors,
            xlabel=f"fraction of edges ({method})",
            ylabel=f"fraction of edges ({y_method})",
            title=f"{y_method} vs {method}")

    # any unused cells in the grid, if the panel set is shorter than it
    for j in range(len(plot_methods), nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")

    fig.suptitle(f"k-NN mixture-pair edge fractions: each method (x) vs "
                 f"{y_method} (y)\n{tag} partition, k_nn={k_nn}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # second figure: the same y axis against itself, one half against the
    # other. Only SPLIT_METHODS have a cached split, so for any other y_method
    # this is None and only the panel grid is returned.
    fig_split = None
    if pooled_split["x"]:
        fig_split, ax_split = plt.subplots(1, 1, figsize=(5.5, 5.5))
        _plot_adjacency_values(
            ax_split, np.concatenate(pooled_split["x"]),
            np.concatenate(pooled_split["y"]),
            groups=np.concatenate(pooled_split["ds"]), group_colors=colors,
            xlabel=f"fraction of edges ({y_method} half A)",
            ylabel=f"fraction of edges ({y_method} half B)",
            title=f"{y_method}: half A vs half B")
        fig_split.suptitle(f"k-NN sampling noise floor: {y_method} split in half\n"
                           f"{tag} partition, k_nn={k_nn}", fontsize=14)
        fig_split.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    return fig, fig_split


# _ = plot_knn()
