# Assumptions:
# - dataset/n_genes identify a cached analysis_spike.py pkl, from which only
#   spec_true and cell_labels are read.
# - The metacell overlap graph (nodes = metacells, edge weight = overlap of the
#   two metacells' EMPIRICAL normals projected onto the line joining their
#   means) is built once per dataset/n_genes from spec_true (see
#   anndata2metacell_graph) and reused for every UMAP layout -- only each
#   layout's node positions differ.

import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import networkx as nx
from scipy.stats import norm

import analysis_spike as asp
import datasets


# Unit vector from gmm_a's mean to gmm_b's mean.
def direction_between_means(gmm_a, gmm_b):
    delta = np.asarray(gmm_b["mean"]) - np.asarray(gmm_a["mean"])
    norm_delta = np.linalg.norm(delta)
    if norm_delta == 0:
        raise ValueError("gmm_a and gmm_b have identical means; direction is undefined")
    return delta / norm_delta


# Projects a GMM {"mean", "covariance"} onto a unit direction, returning the
# (mean, std) of the induced univariate normal. Uses the general quadratic
# form for the variance of a linear projection (direction @ covariance @
# direction) rather than a diagonal-only formula, since the analytic
# covariance (see lognorm_workflow_analytic.get_GMM) has genuine off-diagonal
# (cross-PC) structure.
def project_onto_direction(gmm, direction):
    mean_proj = float(np.dot(gmm["mean"], direction))
    covariance = np.asarray(gmm["covariance"]).real
    var_proj = float(direction @ covariance @ direction)
    return mean_proj, np.sqrt(max(var_proj, 0.0))


# [0,1] overlap between two univariate normals using a combined-variance
# z-score: z = |mean1-mean2| / sqrt(std1^2+std2^2), weight = 2*Phi(-z/2).
# Unlike the overlapping coefficient, this does not penalize a variance
# mismatch by itself -- a narrow distribution centered inside a wide one
# still scores 1.
def combined_z_overlap(mean1, std1, mean2, std2):
    combined_std = np.sqrt(std1 ** 2 + std2 ** 2)
    if combined_std == 0:
        return 1.0 if mean1 == mean2 else 0.0
    z = abs(mean1 - mean2) / combined_std
    return 2 * norm.cdf(-z / 2)


# Projects gmm_a, gmm_b onto the line joining their means, returning
# (mean1, std1, mean2, std2) for the induced univariate normals. Falls back
# to the first standard-basis direction when the means coincide exactly
# (every direction gives the same projected means in that degenerate case).
def project_gmm_pair(gmm_a, gmm_b):
    mean_a, mean_b = np.asarray(gmm_a["mean"]), np.asarray(gmm_b["mean"])
    if np.array_equal(mean_a, mean_b):
        direction = np.zeros_like(mean_a)
        direction[0] = 1.0
    else:
        direction = direction_between_means(gmm_a, gmm_b)
    mean1, std1 = project_onto_direction(gmm_a, direction)
    mean2, std2 = project_onto_direction(gmm_b, direction)
    return mean1, std1, mean2, std2


# Edge weight between two mixtures: project both GMMs onto the line joining
# their means, then score the resulting univariate normals.
def overlap_score(gmm_a, gmm_b):
    mean1, std1, mean2, std2 = project_gmm_pair(gmm_a, gmm_b)
    return combined_z_overlap(mean1, std1, mean2, std2)


def metacell_moments(Y, cell_labels, k=None):
    """(labels, means, covariances) per metacell, empirically from scores Y.

    labels is np.unique(cell_labels), so sorted and shared with everything
    else that groups by that convention; means is (n_mix, k) and covariances
    is (n_mix, k, k), row/slice i belonging to labels[i].

    Raises on a metacell with fewer than 2 cells -- np.cov cannot form a
    covariance from one row and would return NaN, which would propagate
    silently into every edge weight touching that metacell. Metacells with
    fewer than k+1 cells give a rank-deficient covariance, which is usable
    but degenerate along some directions, so those are warned about rather
    than rejected.
    """
    Y = np.asarray(Y)
    cell_labels = np.asarray(cell_labels)
    if k is None:
        k = Y.shape[1]

    labels = np.unique(cell_labels)
    counts = np.array([(cell_labels == lb).sum() for lb in labels])

    tiny = labels[counts < 2]
    if len(tiny):
        raise ValueError(f"metacells with fewer than 2 cells have no empirical "
                         f"covariance: {list(tiny)} (sizes {list(counts[counts < 2])})")
    thin = labels[counts < k + 1]
    if len(thin):
        print(f"  WARNING: {len(thin)} metacells have <= k={k} cells, so their "
              f"empirical covariance is rank deficient: {list(thin[:5])}"
              f"{' ...' if len(thin) > 5 else ''}")

    means = np.stack([Y[cell_labels == lb, :k].mean(axis=0) for lb in labels])
    covariances = np.stack([np.cov(Y[cell_labels == lb, :k], rowvar=False)
                            for lb in labels])
    return labels, means, covariances


# Builds the complete weighted graph over metacells from their empirical
# moments. One node per metacell; every pair gets an edge, weighted by the
# combined_z_overlap of the two metacells' normals projected onto the line
# joining their means -- the raw score, with no cutoff applied. Thresholding
# which edges to actually use is left to the consumer, not baked into the
# graph itself.
#
# Each metacell is taken as normal with its own empirical mean and covariance,
# so the projection onto the joining line is exactly normal too and the pair
# reduces to two univariate normals -- which is what combined_z_overlap scores.
def build_metacell_graph(labels, means, covariances):
    gmms = {label: {"mean": means[i], "covariance": covariances[i]}
            for i, label in enumerate(labels)}

    G = nx.Graph()
    G.add_nodes_from(labels)
    for label_a, label_b in combinations(labels, 2):
        weight = overlap_score(gmms[label_a], gmms[label_b])
        G.add_edge(label_a, label_b, weight=weight)

    return G

# Runs scanpy's neighbors + PAGA on one method's embedding, and returns the
# complete graph over metacell_labels' categories (every pair present,
# including 0-weight ones -- same "cache the full raw graph" convention as
# build_metacell_graph): nodes are the metacell labels, edge weight is
# PAGA's connectivity confidence between the two metacells.
def _build_paga_graph(embedding, metacell_labels):
    adata = ad.AnnData(X=np.zeros((embedding.shape[0], 1)),
                       obs=pd.DataFrame({"metacell": pd.Categorical(metacell_labels)}))
    adata.obsm["X_pca"] = embedding

    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.paga(adata, groups="metacell")

    categories = list(adata.obs["metacell"].cat.categories)
    connectivities = adata.uns["paga"]["connectivities"].toarray()

    G = nx.Graph()
    G.add_nodes_from(categories)
    for i, label_a in enumerate(categories):
        for j in range(i + 1, len(categories)):
            G.add_edge(label_a, categories[j], weight=float(connectivities[i, j]))

    return G


# metacell-graph pkls are split by provenance the same way the spike pkls they
# are built from are, mirroring analysis_spike/analysis_knn/analysis_umap
METACELL_GRAPH_DIR = "../analysis_metacell_graph/"
METACELL_GRAPH_REFINED_DIR = "../analysis_metacell_graph_refined/"


def get_metacell_graph_filename(dataset, n_genes, refined=True):
    out_dir = METACELL_GRAPH_REFINED_DIR if refined else METACELL_GRAPH_DIR
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{dataset}_HVG_{n_genes}.pkl")


def anndata2metacell_graph(dataset, n_genes, overwrite=False, refined=True):
    """Compute (or load, if already cached) both metacell graphs for this
    dataset/n_genes, caching them together as
    {"normal_graph": ..., "paga_graph": ...}.

    Both are built from the SAME embedding -- the spike pkl's spec_true PC
    scores U[:, :k]*sv[:k], grouped by that pkl's cell_labels (stored in the
    same row order, see anndata2spike) -- so they share a node set and their
    edge weights are directly comparable:

      normal_graph  each metacell is taken as normal with its own EMPIRICAL
                    mean and covariance; the weight of a pair is the overlap
                    of the two normals projected onto the line joining their
                    means (see build_metacell_graph). A closed-form property
                    of the fitted moments.

      paga_graph    scanpy's neighbour graph on the same scores, aggregated by
                    PAGA into a per-metacell connectivity (see
                    _build_paga_graph). Driven by the k-NN structure of the
                    actual cells rather than by fitted moments.

    Both graphs are complete -- every pair of metacells has an edge, including
    weight-0 ones -- so thresholding which edges to use stays a display-time
    concern and this cache never goes stale against a cutoff.
    """
    out_f = get_metacell_graph_filename(dataset, n_genes, refined=refined)
    if os.path.isfile(out_f) and not overwrite:
        with open(out_f, "rb") as f:
            return pickle.load(f)

    if not asp.has_spike_result(dataset, n_genes, refined=refined):
        raise ValueError(f"missing analysis_spike pkl for {dataset} HVG {n_genes} "
                         f"({'refined' if refined else 'unrefined'}): needs both "
                         f"{asp.get_pca_filename(dataset, n_genes, refined=refined)} and "
                         f"{asp.get_spike_filename(dataset, n_genes, refined=refined)}")

    result = asp.load_spike_result(dataset, n_genes, refined=refined)
    spec = result["spec_true"]
    cell_labels = np.asarray(result["cell_labels"])

    k = spec["U"].shape[1]
    Y = spec["U"][:, :k] * spec["sv"][:k]
    tag = f"{dataset} HVG {n_genes} ({'refined' if refined else 'unrefined'})"

    print(f"Computing normal overlap graph for {tag}")
    labels, means, covariances = metacell_moments(Y, cell_labels, k=k)
    normal_graph = build_metacell_graph(labels, means, covariances)

    print(f"Computing PAGA graph for {tag}")
    paga_graph = _build_paga_graph(Y, cell_labels)

    graph_result = {"normal_graph": normal_graph, "paga_graph": paga_graph}
    with open(out_f, "wb") as f:
        pickle.dump(graph_result, f)

    return graph_result


def make_all_metacell_graph_pkl(dataset_names=datasets.DEFAULT_DATASETS,
                                n_genes_list=datasets.DEFAULT_N_GENES,
                                overwrite=False, refined=True):
    for dataset in dataset_names:
        for n_genes in n_genes_list:
            if not asp.has_spike_result(dataset, n_genes, refined=refined):
                continue
            print(f"Processing {dataset} HVG {n_genes}")
            anndata2metacell_graph(dataset, n_genes, overwrite=overwrite, refined=refined)
    return None


def plot_metagraph_weights(dataset, n_genes, refined=True, ax=None, show=True):
    """The two graphs' edge weights against each other: PAGA on x, normal
    overlap on y, one point per metacell pair.

    Both graphs are complete over the same node set (see
    anndata2metacell_graph), so the pairs line up exactly and each point is
    one pair scored two ways -- by the k-NN connectivity PAGA aggregates, and
    by the overlap of the two metacells' fitted normals. Weights are read off
    the normal graph's edge order and looked up in the PAGA graph by node
    pair, so the two are matched by identity rather than by iteration order.

    Pairs scoring exactly 0 under BOTH graphs are dropped, as
    analysis_knn._plot_adjacency_values does: they are pairs neither method
    connects at all, they are the bulk of a complete graph over many
    metacells, and being identical rather than merely small no axis transform
    separates them. The count dropped is annotated. A pair that is zero under
    one graph but not the other is kept -- that disagreement is the
    interesting part.

    Both axes use x^(1/3), as analysis_knn's adjacency plots do: the weights
    pile up hard against zero (most metacell pairs barely touch), so a linear
    axis puts nearly every point in the corner, while a log axis cannot show
    the surviving one-sided zeros. Cube root maps 0 to 0 and is smooth and
    monotone throughout.

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    graphs = anndata2metacell_graph(dataset, n_genes, refined=refined)
    g_normal, g_paga = graphs["normal_graph"], graphs["paga_graph"]

    pairs = [(u, v) for u, v, _ in g_normal.edges(data=True)]
    y = np.array([g_normal[u][v]["weight"] for u, v in pairs], dtype=float)
    x = np.array([g_paga[u][v]["weight"] for u, v in pairs], dtype=float)

    keep = ~((x == 0) & (y == 0))
    n_dropped = int((~keep).sum())
    x, y = x[keep], y[keep]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig = ax.figure

    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.scatter(x, y, s=12, alpha=0.45, color="steelblue", linewidths=0)
    lim = [0.0, max(x.max(), y.max(), 1e-12)]
    ax.plot(lim, lim, color="black", ls="--", lw=1, label="y = x")
    ax.set_xscale("function", functions=(np.cbrt, lambda v: np.power(v, 3)))
    ax.set_yscale("function", functions=(np.cbrt, lambda v: np.power(v, 3)))
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax.text(0.97, 0.03,
            f"n plotted = {len(x):,}\n(0,0) dropped = {n_dropped:,}\n"
            f"Spearman = {_spearman(x, y):.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    ax.set_xlabel("PAGA connectivity")
    ax.set_ylabel("normal overlap")
    ax.set_title(f"{dataset} HVG {n_genes} "
                 f"({'refined' if refined else 'unrefined'}): metacell-pair weights")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _spearman(x, y):
    """Rank correlation, which is what suits these weights: both are bounded
    scores on incomparable scales (a PAGA connectivity and a normal overlap
    are not the same units), so how they co-rank is the meaningful question,
    not how they co-vary."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def metacell_median_positions(umap_coords, metacell_labels):
    """{label: (x, y)} at each metacell's median UMAP point.

    Median rather than mean: a metacell's cells can straddle two lobes of the
    embedding, and a mean would then park the node in the empty space between
    them, where no cell of that metacell actually sits.
    """
    metacell_labels = np.asarray(metacell_labels)
    return {label: tuple(np.median(umap_coords[metacell_labels == label], axis=0))
            for label in np.unique(metacell_labels)}


def _metacell_cell_types(metacell_labels, cell_type_labels):
    """{metacell: its cell type} plus the minimum purity behind that call.

    Taken from the DATA rather than parsed off the label prefix: metacells are
    named "<cell_type>_<n>" only when the partition happened to be cell type
    (see nb_model.apply_metacell_decomposition), refinement appends further
    suffixes, and cell-type names may themselves contain underscores -- so
    splitting the string is fragile in a way counting cells is not.

    The type is the most common one in the metacell. That is exact when the
    metacells were built cell-type-partitioned (every purity is 1.0) and a
    plurality call otherwise, which is what the returned minimum purity is for.
    """
    metacell_labels = np.asarray(metacell_labels)
    cell_type_labels = np.asarray(cell_type_labels)

    mapping, purities = {}, []
    for label in np.unique(metacell_labels):
        types, counts = np.unique(cell_type_labels[metacell_labels == label],
                                  return_counts=True)
        mapping[label] = types[np.argmax(counts)]
        purities.append(counts.max() / counts.sum())
    return mapping, float(min(purities)) if purities else 1.0


def _draw_graph_on_umap(ax, embedding, cell_colors, positions, g, title,
                        min_edge_value, node_type=None,
                        inter_edges=True, intra_edges=True, percentile=0.0):
    """One panel: the UMAP cloud coloured by cell type, then metacell nodes at
    `positions` and the surviving edges of `g` on top.

    Drawing order is background cells (zorder 0), edges (1), nodes (2). Edges
    are dark grey and nodes black-on-white rather than coloured, so they stay
    legible over whatever cell-type colour they happen to cross.

    An edge is drawn when its weight is > 0, > min_edge_value, and at or above
    the `percentile` quantile of the graph's NON-ZERO weights. Zeros are
    excluded from that quantile deliberately: PAGA zeroes out most pairs
    outright, so a quantile taken over all weights would sit at 0 for any
    percentile below ~0.75 and the parameter would do nothing on that panel
    while biting hard on the other. Taken over the non-zero weights it means
    the same thing in both -- "the strongest (1-percentile) share of the edges
    that exist".

    The quantile is computed per graph, which is the point of offering it
    alongside min_edge_value: a PAGA connectivity and a normal overlap are not
    the same units, so one absolute cutoff cannot be equally strict on both.

    inter_edges / intra_edges then select which of the survivors to keep, by
    whether an edge's two metacells have different cell types or the same one.
    They are applied AFTER the weight cutoffs, so the percentile is always a
    quantile of the whole graph: turning one of them off removes edges from the
    picture but never re-ranks the ones that remain, and the strongest 10% means
    the same set of edges whichever flags are set.

    Edge width and opacity likewise scale with weight relative to THIS graph's
    own maximum, for the same reason.
    """
    from matplotlib.collections import LineCollection

    ax.scatter(embedding[:, 0], embedding[:, 1], s=2.0, c=cell_colors,
               linewidths=0, alpha=0.55, zorder=0, rasterized=True)

    nonzero = np.array([d["weight"] for _, _, d in g.edges(data=True)
                        if d["weight"] > 0])
    cutoff = float(np.quantile(nonzero, percentile)) if len(nonzero) else 0.0

    edges = [(u, v, d["weight"]) for u, v, d in g.edges(data=True)
             if d["weight"] > 0 and d["weight"] > min_edge_value
             and d["weight"] >= cutoff]
    if not inter_edges:
        edges = [(u, v, w) for u, v, w in edges if node_type[u] == node_type[v]]
    if not intra_edges:
        edges = [(u, v, w) for u, v, w in edges if node_type[u] != node_type[v]]
    if edges:
        w_max = max(w for _, _, w in edges)
        segments = [[positions[u], positions[v]] for u, v, _ in edges]
        widths = [0.2 + 2.3 * (w / w_max) for _, _, w in edges]
        colors = [(0.10, 0.10, 0.10, 0.12 + 0.68 * (w / w_max)) for _, _, w in edges]
        ax.add_collection(LineCollection(segments, linewidths=widths,
                                         colors=colors, zorder=1))

    xy = np.array([positions[node] for node in g.nodes()])
    ax.scatter(xy[:, 0], xy[:, 1], s=26, c="black", edgecolors="white",
               linewidths=0.6, zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"{title}  ({len(edges):,} of {g.number_of_edges():,} edges)",
                 fontsize=11)


def plot_graph_on_umap(dataset, n_genes, refined=True, min_edge_value=0.0,
                       percentile=0.0, inter_edges=True, intra_edges=True,
                       top_n_legend=12, show=True):
    """Both metacell graphs drawn on the true UMAP, PAGA left, normal right.

    The cells are scattered underneath coloured by cell type, with the metacell
    nodes and edges on top. Nodes sit at each metacell's median UMAP coordinate
    (see metacell_median_positions); edges are drawn for every pair scoring
    above min_edge_value, which defaults to 0 and so keeps every non-zero edge.

    percentile keeps only the strongest edges: a fraction in [0, 1], it drops
    everything below that quantile of the graph's NON-ZERO weights, computed
    per graph so it is equally strict on both panels despite their different
    scales (see _draw_graph_on_umap). percentile=0 keeps every non-zero edge;
    0.9 keeps the top tenth. It composes with min_edge_value -- an edge must
    clear both -- so leave one at its default unless you want both.

    inter_edges and intra_edges say which kinds of edge to draw. An edge is
    inter if its two metacells have different cell types -- the edges that
    BRIDGE populations, i.e. the ones that say a method thinks two different
    populations are adjacent -- and intra if they share one. Both default to
    True, which draws everything; inter_edges=False, intra_edges=True gives the
    within-population structure alone, and the reverse gives only the bridges.

    Both filters apply AFTER percentile and min_edge_value, which are therefore
    always taken over the whole graph. Switching one off subsets the picture
    without changing what "the strongest 10% of edges" refers to, so panels made
    with different flags at one percentile are directly comparable.

    Each metacell's cell type is counted from the cells rather than parsed off
    its label (see _metacell_cell_types), and the minimum purity behind those
    calls is reported whenever either filter is engaged, since they are only as
    meaningful as the metacells are pure.

    Every cell type is plotted, but only the top_n_legend most frequent get a
    legend entry -- some datasets have dozens (BMHemato has 48) and a full
    legend would be unreadable. The colour map is built once and shared by both
    panels, so a cell type is the same colour on each.

    Both panels use the SAME embedding and the SAME node positions -- only the
    edges differ -- so anything that changes between them is the graph, not
    the layout.

    Note the two graphs are not equally sparse at min_edge_value=0. PAGA
    zeroes out most pairs outright, while the normal overlap is 2*Phi(-z/2),
    which is strictly positive for any finite z, so no pair is ever exactly
    zero and the right panel draws the complete graph. Raise min_edge_value to
    make that panel readable; the drawn/total edge counts are in each title.

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import analysis_umap as au

    graphs = anndata2metacell_graph(dataset, n_genes, refined=refined)
    umap_result = au.anndata2umap(dataset, n_genes, refined=refined)

    embedding = np.asarray(umap_result["embedding_true"])
    metacell = np.asarray(umap_result["metacell"])
    cell_type = np.asarray(umap_result["cell_type"])

    # the graphs are keyed by the spike pkl's cell_labels; the umap pkl carries
    # the same labels, but check rather than assume -- a mismatch would place
    # nodes at another metacell's centre and silently misdraw every edge
    nodes = set(map(str, graphs["paga_graph"].nodes()))
    if nodes != set(map(str, np.unique(metacell))):
        raise ValueError(
            f"{dataset}/{n_genes}: the graph's metacells do not match the umap "
            f"pkl's ({len(nodes)} vs {len(np.unique(metacell))} labels); rebuild "
            "whichever is stale with overwrite=True")

    positions = metacell_median_positions(embedding, metacell)

    # one colour map for both panels, ranked by frequency so the legend shows
    # the types that actually cover the embedding
    types, counts = np.unique(cell_type, return_counts=True)
    types_by_freq = types[np.argsort(counts)[::-1]]
    palette = au._qualitative_colors(len(types_by_freq))
    type_color = {t: palette[i] for i, t in enumerate(types_by_freq)}
    cell_colors = np.array([type_color[t] for t in cell_type])

    if not 0.0 <= percentile <= 1.0:
        raise ValueError(f"percentile is a fraction in [0, 1], got {percentile!r}")
    if not inter_edges and not intra_edges:
        raise ValueError("inter_edges and intra_edges are both False, which "
                         "excludes every edge; set at least one to True")

    node_type, min_purity = _metacell_cell_types(metacell, cell_type)
    if (not inter_edges or not intra_edges) and min_purity < 1.0:
        print(f"  NOTE: metacells are not cell-type pure (minimum purity "
              f"{min_purity:.2f}); each is assigned its most common cell type, "
              f"so the inter/intra edge split is approximate")

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    _draw_graph_on_umap(axes[0], embedding, cell_colors, positions,
                        graphs["paga_graph"], "PAGA", min_edge_value,
                        node_type=node_type, inter_edges=inter_edges,
                        intra_edges=intra_edges, percentile=percentile)
    _draw_graph_on_umap(axes[1], embedding, cell_colors, positions,
                        graphs["normal_graph"], "normal overlap", min_edge_value,
                        node_type=node_type, inter_edges=inter_edges,
                        intra_edges=intra_edges, percentile=percentile)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                          color=type_color[t], label=str(t)[:30])
               for t in types_by_freq[:top_n_legend]]
    n_types = len(types_by_freq)
    legend_title = "cell type" + (f" (top {top_n_legend} of {n_types})"
                                  if n_types > top_n_legend else "")
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.86, 0.5),
               fontsize=8, frameon=False, title=legend_title, title_fontsize=9)

    fig.suptitle(f"{dataset} HVG {n_genes} "
                 f"({'refined' if refined else 'unrefined'}): metacell graphs on the "
                 f"true UMAP" + (f", edges > {min_edge_value:g}" if min_edge_value else "")
                 + (f", top {(1-percentile)*100:g}% of non-zero edges" if percentile else "")
                 + ("" if inter_edges and intra_edges else
                    (", cross-cell-type edges only" if inter_edges
                     else ", within-cell-type edges only")),
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    if show:
        plt.show()
    return fig
