# Assumptions:
# - The analysis_spike.py pkl for a given (dataset, n_genes) already exists (see
#   analysis_spike.anndata2spike / make_all_spike_pkl).
# - "cell_type" (verified identical to "celltype") is the biological cell-type
#   annotation column in the HVG AnnData's .obs, distinct from "metacell" (the
#   finer-grained grouping used to build the NB model / analytic PCA).
# - UMAP computes its own internal neighbor graph per fit; it does not reuse
#   analysis_knn.py's cached k-NN indices (those exclude self and distances, and
#   use fewer neighbors than UMAP requires -- see the discussion that led here).

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

import datasets
import datasets_refined as dr
import analysis_spike as asp


# umap pkls are split by provenance the same way the spike pkls they embed
# are: refined ones come from analysis_spike_refined / analysis_HVG_refined,
# unrefined from analysis_spike / analysis_HVG
UMAP_DIR = "../analysis_umap/"
UMAP_REFINED_DIR = "../analysis_umap_refined/"

# every spec in the pkl is embedded. analytic is included: it is a genuine
# embedding of the model's own predicted PCA, and it is the warm-start
# reference the others are initialised from.
UMAP_METHODS = ["analytic"] + list(asp.SPIKE_METHODS)

# the methods plot_umap draws. anndata2umap still embeds everything in
# UMAP_METHODS, so this can be widened again without recomputing any umap.
UMAP_PLOT_METHODS = ["analytic", "normal", "nb", "perm", "true"]


def get_umap_filename(dataset, n_genes, refined=False):
    """Path of the umap pkl, creating the folder if it does not exist."""
    out_dir = UMAP_REFINED_DIR if refined else UMAP_DIR
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{dataset}_HVG_{n_genes}.pkl")


def anndata2umap(dataset, n_genes, overwrite=False, refined=False, random_state=0):
    """Compute (or load, if already cached) a 2D UMAP embedding of each of
    the analytic, normal, nb, perm, and true PCA embeddings (U*sv, across all
    cells/mixtures) for this dataset/n_genes, caching the result to
    get_umap_filename(dataset, n_genes, refined).

    refined selects the refined (refined=True) or unrefined (refined=False)
    analysis_spike pkl and HVG h5ad; the two must come from the same
    partition, since the .obs annotations below are matched against the
    spec rows positionally.

    gmm is fit first with UMAP's default initialization ("spectral"); the
    other four are each warm-started from gmm's final embedding
    (init=embedding_gmm) instead of being computed independently, so all
    five layouts stay in a comparable orientation/scale to one another. All
    five specs share the same row order (see anndata2spike), so gmm's
    embedding is a valid, row-aligned starting point for the rest.

    Requires the corresponding analysis_spike.py pkl to already exist.
    """
    out_f = get_umap_filename(dataset, n_genes, refined=refined)
    if os.path.isfile(out_f) and not overwrite:
        with open(out_f, "rb") as f:
            return pickle.load(f)

    if not asp.has_spike_result(dataset, n_genes, refined=refined):
        raise ValueError(f"missing analysis_spike pkl for {dataset} HVG {n_genes} "
                         f"({'refined' if refined else 'unrefined'}): needs both "
                         f"{asp.get_pca_filename(dataset, n_genes, refined=refined)} "
                         f"and {asp.get_spike_filename(dataset, n_genes, refined=refined)}")

    result = asp.load_spike_result(dataset, n_genes, refined=refined)

    s = (dr.load_HVG_refined_dataset(dataset, n_genes) if refined
         else datasets.load_HVG_dataset(dataset, n_genes))
    cell_type = s.obs["cell_type"].to_numpy()
    metacell = s.obs["metacell"].to_numpy()

    k = result["spec_nb"]["U"].shape[1]
    # driven by UMAP_METHODS (analysis_spike's list plus pert_1) rather than a
    # local literal, so methods added there get embedded here too. A spike pkl
    # written before a method existed simply lacks its key; those are skipped
    # with a warning rather than raising, so stale pkls still work.
    specs = {}
    for method in UMAP_METHODS:
        key = "spec_analytic" if method == "analytic" else asp.SPEC_KEYS[method]
        if key in result:
            specs[method] = result[key]
        else:
            print(f"  SKIPPING {method}: {key} not in this spike pkl (rebuild it "
                  f"with overwrite=True to add it)")

    umap_result = {"cell_type": cell_type, "metacell": metacell}
    tag = f"{dataset} HVG {n_genes} ({'refined' if refined else 'unrefined'})"

    # analytic is fit first and every other method is warm-started from its
    # final embedding, so it is pulled out of the loop rather than iterated
    # over. Its rows are aligned to the h5ad cell order by anndata2spike, like
    # every other spec, so it is a valid row-aligned starting point.
    W_analytic = specs["analytic"]["U"][:, :k] * specs["analytic"]["sv"][:k]
    print(f"Computing analytic UMAP for {tag}")
    embedding_analytic = UMAP(random_state=random_state).fit_transform(W_analytic)
    umap_result["embedding_analytic"] = embedding_analytic

    for method in [m for m in UMAP_METHODS if m != "analytic" and m in specs]:
        spec = specs[method]
        W = spec["U"][:, :k] * spec["sv"][:k]
        print(f"Computing {method} UMAP for {tag}")
        embedding = UMAP(init=embedding_analytic, random_state=random_state).fit_transform(W)
        umap_result[f"embedding_{method}"] = embedding

    with open(out_f, "wb") as f:
        pickle.dump(umap_result, f)

    return umap_result


def make_all_umap_pkl(dataset_names=datasets.DEFAULT_DATASETS, n_genes_list=datasets.DEFAULT_N_GENES,
                      overwrite=False, refined=False):
    """Build the umap pkl for every (dataset, n_genes) in the given partition.

    A combination whose analysis_spike pkl has not been built is skipped
    silently, since that is the normal state for a partition that has not
    been swept yet; one whose HVG h5ad is missing is skipped with a warning
    rather than aborting the sweep.
    """
    for dataset in dataset_names:
        for n_genes in n_genes_list:
            if not asp.has_spike_result(dataset, n_genes, refined=refined):
                continue
            print(f"Processing {dataset} HVG {n_genes} "
                  f"({'refined' if refined else 'unrefined'})")
            try:
                anndata2umap(dataset, n_genes, overwrite=overwrite, refined=refined)
            except ValueError as e:
                print(f"  SKIPPING {dataset}/{n_genes}: {e}")
    return None


def _qualitative_colors(n):
    """n visually distinguishable colors, built by concatenating matplotlib's
    tab20/tab20b/tab20c qualitative palettes (60 colors total, cycling if
    n > 60) rather than sampling a continuous colormap -- sampling e.g.
    nipy_spectral at many points puts adjacent categories at very similar
    hues, which is hard to tell apart once there are more than a handful of
    categories (BMHemato has 48 cell types).

    Each of tab20/tab20b/tab20c pairs a dark and a light shade of the same
    hue at adjacent indices (0&1, 2&3, ...), so taking them in native order
    puts near-identical-hue colors right next to each other for the first
    few categories -- exactly the common case here, since most datasets have
    well under 10 cell types. Reordering each palette to all its "dark"
    shades first, then all its "light" shades, keeps the first ~10 colors
    maximally distinct in hue (tab20's darks alone are the classic tab10
    palette).
    """
    base = []
    for name in ["tab20", "tab20b", "tab20c"]:
        cmap_colors = list(plt.get_cmap(name).colors)
        base.extend(cmap_colors[0::2] + cmap_colors[1::2])
    return [base[i % len(base)] for i in range(n)]


def plot_umap(dataset, n_genes, top_n_legend=10, refined=True, base_method="true"):
    """Single-row panel plot of each method's 2D UMAP embedding (see
    anndata2umap) for ONE partition, colored by cell_type (not metacell).

    refined picks the partition, refined by default. The two partitions have
    different metacells -- refinement drops cells -- so their embeddings are
    not comparable panel for panel, and only the requested one is drawn.

    base_method chooses the panels:

      base_method in PROJ_BASES ("perm" or "true")
          base_method itself leads, followed by its simulated series,
          <base_method>_proj_0 .. _proj_3: the analytic model projected onto
          that method's basis with a rank-i covariance correction (see
          analysis_spike.simulate_projected_specs). The base panel is what the
          series is trying to reproduce, so reading left to right shows how
          many correction ranks it takes before the simulation's embedding
          looks like the real thing's -- without it there is nothing in the
          figure to judge the simulations against.

      anything else
          the panels fall back to UMAP_PLOT_METHODS.

    anndata2umap still embeds every method in UMAP_METHODS, so either panel
    set is available without recomputing any umap. A requested method missing
    from the pkl (one written before that method existed) is drawn as an empty
    "(absent)" panel rather than dropped, so it is obvious what a rebuild
    would add.

    Every cell type gets its own color and is plotted in every panel, but only
    the top_n_legend most frequent get a legend entry -- for datasets with
    many cell types (e.g. BMHemato has 48) a full legend would be unreadable.
    Ranking and colors are computed over this partition's cell types and
    shared across the panels, and a single legend is drawn for the figure.
    """
    known = list(UMAP_METHODS)
    if base_method not in known:
        raise ValueError(f"base_method {base_method!r} is not an embedded method; "
                         f"expected one of {known}")

    tag = "refined" if refined else "unrefined"
    umap_result = anndata2umap(dataset, n_genes, refined=refined)

    # a base with a simulated proj series is shown as itself followed by that
    # series, so the reference sits beside what is imitating it; anything else
    # keeps the fixed UMAP_PLOT_METHODS panels
    panels = ([base_method] + [f"{base_method}_proj_{i}" for i in asp.PROJ_RANKS]
              if base_method in asp.PROJ_BASES else list(UMAP_PLOT_METHODS))

    present = [m for m in panels if f"embedding_{m}" in umap_result]
    if not present:
        raise ValueError(
            f"none of the requested panels {panels} are in the {tag} umap pkl for "
            f"{dataset} HVG {n_genes}; rebuild it with overwrite=True (and its "
            f"spike pkl first, if that predates these methods)")

    cell_type = np.asarray(umap_result["cell_type"])
    types, counts = np.unique(cell_type, return_counts=True)
    types_by_freq = types[np.argsort(counts)[::-1]]
    n_types = len(types_by_freq)

    color_list = _qualitative_colors(n_types)
    colors = {ct: color_list[i] for i, ct in enumerate(types_by_freq)}

    # width scales with the panel count -- the figure was sized for the full
    # method list, and a fixed width stretches a short list unreadably
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5.0625 * len(panels), 10.3125),
                             squeeze=False)

    for ax, method in zip(axes[0], panels):
        if f"embedding_{method}" not in umap_result:
            ax.axis("off")
            ax.set_title(f"{method} (absent)", fontsize=14, color="grey")
            continue
        embedding = umap_result[f"embedding_{method}"]
        for ct in types_by_freq:
            mask = cell_type == ct
            if not mask.any():
                continue
            ax.scatter(embedding[mask, 0], embedding[mask, 1], color=colors[ct],
                      s=3, alpha=0.6)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(method)

    # build the legend from the shared color map rather than harvesting it off
    # an axis, so it covers every top type even when one is absent from the
    # panel that happened to be drawn first
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=12,
                          color=colors[ct], label=ct)
               for ct in types_by_freq[:top_n_legend]]

    legend_title = "cell type" + (f" (top {top_n_legend} of {n_types})" if n_types > top_n_legend else "")
    fig.suptitle(f"{dataset} HVG {n_genes} ({tag}): UMAP by method, colored by "
                 f"cell type", fontsize=26)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.85, 0.5),
              fontsize=16, title=legend_title, title_fontsize=18)
    # leave headroom for the suptitle, which would otherwise land on top of
    # the panel titles
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    plt.show()
    return fig


#_ = plot_umap("PBMC", 1000, top_n_legend=20)