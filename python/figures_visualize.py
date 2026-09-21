"""Utilities behind figures.make_visualize_figure: a qualitative look at the
LML-RMT, metacell (nb) and true PCA embeddings of one or more datasets.

Panel A (left): the cached 2D UMAP of each embedding (analysis_umap pkls),
colored by cell type, with the metacell used in panel B outlined in black.

Panel B (right, same rows): for one metacell of roughly target_size cells,
the cells' PCA scores in each embedding, centered on that embedding's own
metacell mean, projected onto ONE random direction shared by all three
embeddings, one grey histogram per embedding on a shared x axis with the
LML-RMT normal prediction overlaid in blue.

Everything is read from caches: the refined analysis_spike pkls (for the
scores, the cell labels and pca_theory) and the refined analysis_umap pkls
(for the 2D embeddings and cell types). Nothing here recomputes a UMAP.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

import analysis_spike as asp
import analysis_umap as au
import figures_util as fu
import lognorm_workflow_analytic as lwa


# (pkl method, panel label), left to right. analytic is the model's own
# predicted embedding and is labelled the way the manuscript names it.
VISUALIZE_METHODS = [("analytic", "LML-RMT"), ("nb", "metacell"), ("true", "true")]

# cell types beyond the MAX_CELL_TYPES most frequent are pooled as OTHER_LABEL
MAX_CELL_TYPES = 8
OTHER_LABEL = "other"
OTHER_COLOR = (0.8, 0.8, 0.8)

# font sizes of the panel titles and dataset row labels, the histogram axis
# labels ("density", "projection onto random direction") and the A/B
# figure-part letters: 50% over the 15, 12 and 22 the figure first used.
# The ticks and legends are not scaled with them. The chosen metacell's
# cell count and k are printed and kept on fig.visualize_metacells rather
# than written inside the panel.
TITLE_FONTSIZE = 22.5
AXIS_LABEL_FONTSIZE = 18
PART_LABEL_FONTSIZE = 33

def _spec_for(result, method):
    """The spec dict a pkl method name refers to; analytic is the reference
    and lives under its own key rather than in asp.SPEC_KEYS."""
    key = "spec_analytic" if method == "analytic" else asp.SPEC_KEYS[method]
    return result[key]


def choose_metacell_near_size(cell_labels, target_size=200):
    """The metacell whose cell count is closest to target_size; ties go to
    the first label in sorted order, so the choice is reproducible."""
    labels, sizes = np.unique(np.asarray(cell_labels), return_counts=True)
    gap = np.abs(sizes - target_size)
    return labels[np.argmin(gap)]


def group_cell_types(cell_type, max_types=MAX_CELL_TYPES):
    """Keep the max_types most frequent cell types and relabel the rest as
    OTHER_LABEL.

    Returns (grouped, categories): grouped is the relabelled array and
    categories lists the kept types by descending frequency, followed by
    OTHER_LABEL when anything was pooled.
    """
    cell_type = np.asarray(cell_type).astype(str)
    types, counts = np.unique(cell_type, return_counts=True)
    order = np.argsort(counts)[::-1]
    kept = list(types[order[:max_types]])
    grouped = np.where(np.isin(cell_type, kept), cell_type, OTHER_LABEL)
    categories = kept + ([OTHER_LABEL] if len(types) > max_types else [])
    return grouped, categories


def category_colors(categories):
    """tab10 for the named types, grey for OTHER_LABEL."""
    cmap = plt.get_cmap("tab10")
    colors = {}
    i = 0
    for category in categories:
        if category == OTHER_LABEL:
            colors[category] = OTHER_COLOR
        else:
            colors[category] = cmap(i % cmap.N)
            i += 1
    return colors


def metacell_outline(embedding, mask, link_factor=10, min_fraction=0.1):
    """Convex-hull outline(s) of the 2D points embedding[mask].

    A metacell need not be one blob in a UMAP: in the true embedding it can
    split into pieces far apart, and one hull around all of them would
    enclose mostly empty space. So the points are first cut into
    single-linkage components at distance link_factor times their median
    nearest-neighbor spacing, and every component holding at least
    min_fraction of the cells gets its own hull. Smaller components are
    stray cells and are left unoutlined. A compact metacell comes out as a
    single hull, as before.

    Returns a list of closed (m+1, 2) polygons (last row repeats the first),
    possibly empty if no component has 3 or more points.
    """
    points = np.asarray(embedding)[np.asarray(mask)]
    if len(points) < 3:
        return []
    D = squareform(pdist(points))
    np.fill_diagonal(D, np.inf)
    nn = np.median(D.min(axis=1))
    if nn > 0:
        component = fcluster(linkage(points, "single"), t=link_factor * nn,
                             criterion="distance")
    else:
        component = np.ones(len(points), dtype=int)

    polygons = []
    for label in np.unique(component):
        piece = points[component == label]
        if len(piece) < max(3, min_fraction * len(points)):
            continue
        hull = ConvexHull(piece)
        poly = piece[hull.vertices]
        polygons.append(np.vstack([poly, poly[:1]]))
    return polygons


# outline styling: a black line over a white halo so it reads against dense
# colored points, and a ring of at least MIN_OUTLINE_FRACTION of the panel
# span around any hull smaller than that so a compact metacell does not
# collapse to a dot
OUTLINE_LW = 3.5
OUTLINE_HALO_LW = 6.5
MIN_OUTLINE_FRACTION = 0.05
# the outline, its halo, and the metacell's own cells: a heavy black line
# with a white halo so it reads against dense colored points, and the
# metacell cells in black on top of the colored ones
OUTLINE_COLOR = "black"
OUTLINE_HALO_COLOR = "white"
METACELL_CELL_COLOR = "black"
METACELL_CELL_EDGE = "none"


def _outline_effects():
    return [patheffects.withStroke(linewidth=OUTLINE_HALO_LW,
                                   foreground=OUTLINE_HALO_COLOR)]


def _draw_umap_panel(ax, embedding, grouped, categories, colors, outline,
                     metacell_mask=None, point_size=1.5):
    """One UMAP scatter: OTHER_LABEL first so the named types sit on top,
    then the metacell's own cells in black, then the outline polygons
    (metacell_outline) above everything, each drawn as a black line with
    a white halo. A polygon whose extent is under MIN_OUTLINE_FRACTION of
    the panel's span gets a ring of that size around its centroid instead
    of (not in addition to) the hull, since the hull itself would be
    invisible at panel scale. Rasterized, since the panels hold tens of
    thousands of points."""
    embedding = np.asarray(embedding)
    draw_order = ([OTHER_LABEL] if OTHER_LABEL in categories else []) + \
                 [c for c in categories if c != OTHER_LABEL]
    for category in draw_order:
        m = grouped == category
        if m.any():
            ax.scatter(embedding[m, 0], embedding[m, 1], color=colors[category],
                       s=point_size, alpha=0.6, linewidths=0, rasterized=True)
    if metacell_mask is not None and np.asarray(metacell_mask).any():
        pts = embedding[np.asarray(metacell_mask)]
        ax.scatter(pts[:, 0], pts[:, 1], color=METACELL_CELL_COLOR,
                   edgecolors=METACELL_CELL_EDGE, linewidths=0.3,
                   s=point_size * 4, zorder=4, rasterized=True)

    span = np.ptp(embedding, axis=0).max()
    min_radius = 0.5 * MIN_OUTLINE_FRACTION * span
    for poly in outline:
        if np.ptp(poly, axis=0).max() < 2 * min_radius:
            center = poly[:-1].mean(axis=0)
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(center[0] + min_radius * np.cos(theta),
                    center[1] + min_radius * np.sin(theta),
                    color=OUTLINE_COLOR, lw=OUTLINE_LW, zorder=5,
                    path_effects=_outline_effects())
        else:
            ax.plot(poly[:, 0], poly[:, 1], color=OUTLINE_COLOR, lw=OUTLINE_LW,
                    zorder=5, path_effects=_outline_effects())
    ax.set_xticks([])
    ax.set_yticks([])


def compute_random_direction_projection(dataset, n_genes=1000, metacell=None,
                                        seed=0, refined=True,
                                        methods=VISUALIZE_METHODS,
                                        target_size=200):
    """Project one metacell's centered PCA scores onto a single random
    direction, in each embedding.

    One unit vector d in R^k is drawn from a generator seeded with `seed` and
    reused for every method. For each method the metacell's rows of
    U[:, :k] * sv[:k] are centered on their own mean in that embedding, so
    the metacell sits at the origin of its own PCA space, then projected
    onto d. Centering is what makes the three comparable: the embeddings
    have different bases and arbitrary PC signs, so only the spread and
    shape along d carry over, not the location.

    metacell=None picks choose_metacell_near_size(cell_labels, target_size).

    Returns {"projections": {method: 1D array}, "direction": d,
    "analytic_sd": sd, "metacell": label, "k": k, "n_cells": n}, where sd is
    the LML-RMT predicted standard deviation along d, sqrt(d' Sigma d) with
    Sigma the metacell's covariance from lwa.get_GMM.
    """
    result = asp.load_spike_result(dataset, n_genes, refined=refined)
    cell_labels = np.asarray(result["cell_labels"])
    if metacell is None:
        metacell = choose_metacell_near_size(cell_labels, target_size)
    mask = cell_labels == metacell
    if not mask.any():
        raise ValueError(f"metacell {metacell!r} not in {dataset} HVG {n_genes}")

    specs = {method: _spec_for(result, method) for method, _ in methods}
    k = min(len(spec["sv"]) for spec in specs.values())

    rng = np.random.default_rng(seed)
    d = rng.standard_normal(k)
    d /= np.linalg.norm(d)

    projections = {}
    for method, spec in specs.items():
        scores = np.asarray(spec["U"])[:, :k] * np.asarray(spec["sv"])[:k]
        rows = scores[mask]
        projections[method] = (rows - rows.mean(axis=0)) @ d

    gmm = lwa.get_GMM(result["pca_theory"], metacell)
    cov = np.asarray(gmm["covariance"])[:k, :k]
    analytic_sd = float(np.sqrt(max(d @ cov @ d, 0.0)))

    return {"projections": projections, "direction": d, "analytic_sd": analytic_sd,
            "metacell": metacell, "k": k, "n_cells": int(mask.sum())}


def _draw_projection_panel(ax, values, bins, analytic_sd, hist_label=None,
                           curve_label=None):
    """One embedding's projection as a grey filled density histogram on the
    shared bins, with the LML-RMT normal density along the same direction
    overlaid in blue -- the bulk figure's grey-histogram-plus-RMT-curve look.
    The curve is drawn at its true density, not rescaled to the histogram
    peak, so a variance mismatch shows in both width and height."""
    ax.hist(values, bins=bins, density=True, alpha=0.7, color="#a9a9a9",
            label=hist_label)
    if analytic_sd > 0:
        x = np.linspace(bins[0], bins[-1], 400)
        ax.plot(x, np.exp(-0.5 * (x / analytic_sd) ** 2)
                / (analytic_sd * np.sqrt(2 * np.pi)),
                color="blue", lw=1.95, label=curve_label)
    ax.tick_params(labelsize=11)


def make_visualize_figure(dataset_list=("PBMC", "Thymus"), n_genes=1000,
                          refined=True, methods=VISUALIZE_METHODS,
                          target_size=200, metacell_overrides=None, seed=0,
                          top_n_legend=MAX_CELL_TYPES, n_bins=40,
                          clip_quantile=0.995, show=True):
    """One row per dataset. Panel A (left): one UMAP per method, colored by
    cell type (top top_n_legend types, rest pooled as other) with the panel
    B metacell outlined in black. Panel B (right): one histogram per method
    of that metacell's centered projection onto a single random direction,
    see compute_random_direction_projection, with the LML-RMT normal
    overlaid. The legend column sits between the two panels.

    Every histogram uses ONE bin array, symmetric about zero and spanning
    the clip_quantile quantile of |projection| pooled over all datasets and
    methods, and the histogram axes share x, so a wider distribution is
    wider on the page whichever panel it is in. The few cells beyond that
    quantile fall outside the bins and are left out of the histograms
    (their count is printed); clip_quantile=1 keeps every cell. Within a
    row the histograms also share y.

    metacell_overrides maps dataset -> metacell label to pin the choice;
    otherwise the metacell nearest target_size cells is used. The chosen
    metacells are printed and stored on the figure as fig.visualize_metacells.
    """
    dataset_list = list(dataset_list)
    methods = list(methods)
    metacell_overrides = dict(metacell_overrides or {})
    n_rows, n_methods = len(dataset_list), len(methods)

    # everything per dataset is gathered first: the pooled histogram bins
    # need every projection before any panel is drawn
    per_dataset = []
    for dataset in dataset_list:
        result = asp.load_spike_result(dataset, n_genes, refined=refined)
        cell_labels = np.asarray(result["cell_labels"])
        metacell = metacell_overrides.get(
            dataset, choose_metacell_near_size(cell_labels, target_size))
        mask = cell_labels == metacell
        print(f"{dataset}: panel B metacell {metacell!r} ({mask.sum()} cells)")

        umap_result = au.anndata2umap(dataset, n_genes, refined=refined)
        if not np.array_equal(np.asarray(umap_result["metacell"]).astype(str),
                              cell_labels.astype(str)):
            raise ValueError(f"{dataset}: umap pkl metacells do not match the spike "
                             f"pkl's cell_labels; rebuild the umap pkl")

        proj = compute_random_direction_projection(
            dataset, n_genes, metacell=metacell, seed=seed, refined=refined,
            methods=methods)
        per_dataset.append((dataset, mask, umap_result, proj))

    pooled = np.concatenate([v for _, _, _, proj in per_dataset
                             for v in proj["projections"].values()])
    half_width = np.quantile(np.abs(pooled), clip_quantile)
    bins = np.linspace(-half_width, half_width, n_bins + 1)
    n_clipped = int((np.abs(pooled) > half_width).sum())
    if n_clipped:
        print(f"{n_clipped} of {len(pooled)} projected cells lie beyond the shared "
              f"x range (+/-{half_width:.4f}) and are not drawn")

    # columns: n_methods UMAPs, the legend, n_methods histograms. The legend
    # column separates the two panels; wspace is padded on the histogram
    # side via the legend column's own width.
    width_ratios = [1] * n_methods + [0.72] + [0.8] * n_methods
    fig = plt.figure(figsize=(4.2 * sum(width_ratios) + 0.8, 4.2 * n_rows + 1.0))
    # top leaves room above the titles for the A/B letters at
    # PART_LABEL_FONTSIZE, which sit 0.04 of the figure height above the axes
    gs = fig.add_gridspec(n_rows, 2 * n_methods + 1, width_ratios=width_ratios,
                          left=0.03, right=0.99, top=0.90, bottom=0.08,
                          hspace=0.18, wspace=0.12)
    col_b0 = n_methods + 1

    chosen = {}
    first_hist_ax = None
    for i, (dataset, mask, umap_result, proj) in enumerate(per_dataset):
        chosen[dataset] = (proj["metacell"], proj["n_cells"])
        grouped, categories = group_cell_types(umap_result["cell_type"], top_n_legend)
        colors = category_colors(categories)

        for j, (method, label) in enumerate(methods):
            ax = fig.add_subplot(gs[i, j])
            key = f"embedding_{method}"
            if key not in umap_result:
                ax.axis("off")
                ax.set_title(f"{label} (absent)", color="grey")
                continue
            embedding = umap_result[key]
            _draw_umap_panel(ax, embedding, grouped, categories, colors,
                             metacell_outline(embedding, mask),
                             metacell_mask=mask)
            if i == 0:
                ax.set_title(label, fontsize=TITLE_FONTSIZE)
            if j == 0:
                ax.set_ylabel(fu.display_name(dataset), fontsize=TITLE_FONTSIZE)

        ax_leg = fig.add_subplot(gs[i, n_methods])
        ax_leg.axis("off")
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                              color=colors[c], label=c) for c in categories]
        ax_leg.legend(handles=handles, loc="center left", fontsize=8.5,
                      title="cell type", title_fontsize=10, frameon=False)

        row_first_ax = None
        for j, (method, label) in enumerate(methods):
            ax = fig.add_subplot(gs[i, col_b0 + j], sharex=first_hist_ax,
                                 sharey=row_first_ax)
            if first_hist_ax is None:
                first_hist_ax = ax
            if row_first_ax is None:
                row_first_ax = ax
            is_first = i == 0 and j == 0
            _draw_projection_panel(ax, proj["projections"][method], bins,
                                   proj["analytic_sd"],
                                   hist_label="embedding" if is_first else None,
                                   curve_label="LML-RMT normal" if is_first else None)
            if i == 0:
                ax.set_title(label, fontsize=TITLE_FONTSIZE)
            if j == 0:
                ax.set_ylabel("density", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if i == n_rows - 1:
                # the histograms share x, so the block gets ONE label, under
                # its middle panel: a label under each panel would run into
                # its neighbours at AXIS_LABEL_FONTSIZE
                if j == n_methods // 2:
                    ax.set_xlabel("projection onto random direction",
                                  fontsize=AXIS_LABEL_FONTSIZE)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            if is_first:
                ax.legend(fontsize=9, frameon=False, loc="upper right")

    first_hist_ax.set_xlim(bins[0], bins[-1])
    # shared x, so one locator thins the ticks on every histogram; the
    # default density runs the labels together at this range
    first_hist_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    x_a = gs[0, 0].get_position(fig).x0
    x_b = gs[0, col_b0].get_position(fig).x0
    y_top = gs[0, 0].get_position(fig).y1
    fig.text(x_a - 0.015, y_top + 0.04, "A", fontsize=PART_LABEL_FONTSIZE,
             fontweight="bold", ha="right", va="bottom")
    fig.text(x_b - 0.015, y_top + 0.04, "B", fontsize=PART_LABEL_FONTSIZE,
             fontweight="bold", ha="right", va="bottom")

    fig.visualize_metacells = chosen
    if show:
        plt.show()
    return fig
