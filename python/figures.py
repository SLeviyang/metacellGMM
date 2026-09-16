import os

import matplotlib.pyplot as plt
import numpy as np

import analysis_bulk_celltype as abc
import analysis_spike as asp
import datasets
import figures_util as fu
import figures_visualize as fv

FIGURE_DIR = "../figures/"
# the datasets given their own panel in the per-dataset figures (bulk, KS);
# must be a subset of datasets.DEFAULT_DATASETS, since only those have pkls
# built by make.py
DEFAULT_FIGURE_DATASETS = ["PBMC", "Heart_large", "BMHemato", "Thymus"]
# the two datasets given a row in the qualitative visualize figure
VISUALIZE_DATASETS = ["PBMC", "Thymus"]
# the panel B metacell is the one closest to VISUALIZE_TARGET_SIZE cells
# (fv.choose_metacell_near_size); VISUALIZE_METACELLS pins a dataset to a
# named metacell instead, e.g. when the size rule lands on a metacell that
# splits into pieces in the true UMAP. Thymus has two 200-cell metacells
# and the tie-break picks CD8aa(II)_2_0_1_1, which splits 70/26 in the true
# UMAP; T(agonist)_1_0_0 is the other one and stays a single blob, so it is
# pinned. Metacell labels are regenerated with the h5ads, so after a
# rebuild check that the pinned label still exists and still looks right
# (it reproduced unchanged across the per-metacell-L rebuild).
VISUALIZE_TARGET_SIZE = 200
VISUALIZE_METACELLS = {"Thymus": "T(agonist)_1_0_0"}

# the spectra plot_spike_spectrum compares against analytic, as (panel label,
# method); asp.SPEC_KEYS maps each method to its key in the spike pkl. The
# label is carried alongside because the pkl's key for the NB model is the
# lowercase "nb" while the panel is titled "NB". analysis_spike's own version
# of that figure walks asp.SPIKE_METHODS instead, which additionally carries
# the eight <base>_proj_i entries -- those are deliberately left out here.
SPIKE_SPECTRUM_METHODS = [("NB", "nb"), ("true", "true")]

# Figure 1
def make_bulk_figure():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig = fu.make_bulk_figure(dataset_list=DEFAULT_FIGURE_DATASETS, show=False)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_bulk.pdf"))
    return fig


def make_bulk_figure_SI():
    # the complement of DEFAULT_FIGURE_DATASETS, computed rather than listed
    # so the split follows automatically if that constant changes
    os.makedirs(FIGURE_DIR, exist_ok=True)
    si_datasets = [d for d in datasets.DEFAULT_DATASETS
                   if d not in DEFAULT_FIGURE_DATASETS]
    fig = fu.make_bulk_figure(dataset_list=si_datasets,
                              include_unsmoothed_analytic=True, show=False)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_SI_bulk.pdf"))
    return fig


def make_bulk_figure_celltype():
    # the cell-type-metacell counterpart of make_bulk_figure: same datasets
    # and layout, read from the analysis_bulk_celltype pkls instead of the
    # refined ones. n_genes is left at abc's default of 1000, the only gene
    # panel built for the cell-type h5ads (make_bulk_figure uses 2000).
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig = abc.make_bulk_figure_celltype(dataset_list=DEFAULT_FIGURE_DATASETS,
                                        show=False)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_bulk_celltype.pdf"))
    return fig


def make_MP_figure():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig = fu.make_MP_figure(dataset_list=DEFAULT_FIGURE_DATASETS, show=False)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_MP.pdf"))
    return fig


def make_spike_spectrum_figure(dataset_list=None, n_genes_list=None, refined=True,
                        show=True):
    """Each method's outlier singular values against the LML-RMT prediction
    at the same rank, pooled over datasets into one panel per method.

    Each panel is annotated in its lower right with "REL ERR: value", the
    per-point median relative error |y - x| / |x| (fu.median_relative_error)
    computed separately for every dataset and then averaged over datasets --
    the same summary the within-metacell variance and neighbor-distance
    figures carry, so the four quantitative figures read alike.
    """
    entries = asp._load_spike_entries(refined, dataset_list, n_genes_list)
    if not entries:
        raise ValueError("no spike pkls found; run make_all_spike_pkl to build them")

    dataset_names = sorted({dataset for dataset, *_ in entries})
    # tab20 once there are more than ten datasets, as in figures_util, so
    # no two datasets share a colour
    cmap = plt.get_cmap("tab10" if len(dataset_names) <= 10 else "tab20")
    colors = {dataset: cmap(i % cmap.N) for i, dataset in enumerate(dataset_names)}

    fig, axes = plt.subplots(1, len(SPIKE_SPECTRUM_METHODS),
                             figsize=(5 * len(SPIKE_SPECTRUM_METHODS), 5),
                             squeeze=False)

    # one limit for the whole row rather than per panel: x is the SAME
    # analytic spectrum in every panel, so per-panel limits would differ only
    # by each method's own y max and leave the panels unable to be compared
    # by eye
    hi = max(max(result["spec_analytic"]["sv"].max(), result[key]["sv"].max())
             for _, method in SPIKE_SPECTRUM_METHODS
             for key in [asp.SPEC_KEYS[method]]
             for _, _, result in entries if key in result)
    lim = [0, hi * 1.1]

    for j, (label, method) in enumerate(SPIKE_SPECTRUM_METHODS):
        ax = axes[0, j]
        key = asp.SPEC_KEYS[method]
        available = [entry for entry in entries if key in entry[2]]
        if not available:
            ax.axis("off")
            ax.set_title(f"{label}\n(not in these pkls)", fontsize=9, color="grey")
            continue

        seen = set()
        # the same statistic the variance and neighbor-distance figures
        # report: the median over a dataset's points of |y - x| / |x|
        # (fu.median_relative_error), then the mean of those over datasets,
        # so each dataset counts once regardless of how many outliers it has
        rel_err_by_dataset = []
        for dataset, n_genes, result in available:
            x = np.asarray(result["spec_analytic"]["sv"], dtype=float)
            y = np.asarray(result[key]["sv"], dtype=float)
            rel_err_by_dataset.append(fu.median_relative_error(x, y))
            ax.scatter(x, y, color=colors[dataset], alpha=0.7,
                       label=fu.display_name(dataset) if dataset not in seen else None)
            seen.add(dataset)

        ax.plot(lim, lim, color="black", ls="--", lw=1)
        ax.text(0.97, 0.03, f"REL ERR: {np.mean(rel_err_by_dataset):.3g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{fu.method_label('RMT')} sv")
        ax.set_ylabel(f"{fu.method_label(label)} sv")
        ax.set_title(fu.method_label(label))
        if j == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"{'refined' if refined else 'unrefined'} "
                 f"({len(dataset_names)} datasets)", fontsize=14)
    fig.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_spectrum.pdf"))

    if show:
        plt.show()
    return fig


# the spectra the KS figure draws, left to right; analytic is labelled RMT
# on the axis (see fu.KS_SPECTRUM_LABELS). normal and perm are left out.
KS_SPECTRA = ["analytic", "nb", "true"]


def make_KS_figure(show=True):
    """One row of the normality-KS boxplots, one panel per dataset in
    DEFAULT_FIGURE_DATASETS, on refined metacells, showing only the
    KS_SPECTRA boxes.

    n_cols is set to the dataset count so fu.make_normality_KS_figure lays
    every dataset out in a single row.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig = fu.make_normality_KS_figure(dataset_list=DEFAULT_FIGURE_DATASETS,
                                      HVG_type="refined",
                                      n_cols=len(DEFAULT_FIGURE_DATASETS),
                                      spectra=KS_SPECTRA, show=False)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_normality_KS.pdf"))

    if show:
        plt.show()
    return fig


# the methods the normality spectrum figure gives a panel, left to right
NORMALITY_SPECTRUM_METHODS = ["normal", "nb", "perm", "true"]


def make_normality_spectrum_figure():
    # every dataset, not just DEFAULT_FIGURE_DATASETS: this figure pools its
    # datasets into one panel per method rather than giving each its own, so
    # the full set costs only a denser scatter
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig = fu.make_normality_spectrum_figure(
        dataset_list=datasets.DEFAULT_DATASETS, n_genes=1000, refined=True,
        methods=NORMALITY_SPECTRUM_METHODS, show=False)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_normality_spectrum.pdf"))
    return fig


def make_neighbor_distance_figure():
    # every dataset, for the same reason as make_normality_spectrum_figure
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig = fu.make_neighbor_distance_figure(
        dataset_list=datasets.DEFAULT_DATASETS, n_genes=1000, refined=True,
        show=False)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_neighbor_distance.pdf"))
    return fig


def make_knn_figure():
    # every dataset, for the same reason as make_normality_spectrum_figure:
    # the datasets are pooled into one panel per method rather than given a
    # row each, so the full set costs only a denser scatter
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig = fu.make_knn_figure(dataset_list=datasets.DEFAULT_DATASETS, show=False)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_knn_adjacency.pdf"))
    return fig


def make_visualize_figure(show=False):
    # qualitative orientation figure: UMAPs of the LML-RMT, metacell and true
    # embeddings with one metacell outlined, plus that metacell's projection
    # onto a random direction. dpi matters here because the UMAP scatters are
    # rasterized inside the pdf.
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig = fv.make_visualize_figure(dataset_list=VISUALIZE_DATASETS, n_genes=1000,
                                   refined=True,
                                   target_size=VISUALIZE_TARGET_SIZE,
                                   metacell_overrides=VISUALIZE_METACELLS,
                                   show=False)
    fig.savefig(os.path.join(FIGURE_DIR, "figure_visualize.pdf"), dpi=200)
    if show:
        plt.show()
    return fig
