"""The manuscript's Supplementary Information figures: Figures 4-8 rebuilt on
the cell-type metacell partition (datasets_celltype.py: one metacell per
annotated cell type, at N_GENES genes), read from the analysis_spike_celltype
and analysis_knn_celltype caches.

Each function mirrors its figures.py counterpart -- the same datasets,
methods, layout and fonts, through the same figures_util function called
with HVG_type="celltype" -- and writes the same file name with a _celltype
suffix into figures.FIGURE_DIR. Figures 1-3 have no cell-type counterpart:
the UMAP, bulk and 2000/5000-gene caches they need were never built for
this partition (make.py builds the cell-type h5ads and spike pkls at N_GENES
only).

FIGURE_STEPS lists the five in their order in the Supplementary Information
(S1-S5); make.py appends it to its own list and make_all runs it here.
"""
import os

import matplotlib.pyplot as plt

import datasets
import figures
import figures_util as fu

# the one gene count the cell-type caches exist at (make.CELLTYPE_N_GENES)
N_GENES = 1000


def make_spike_spectrum_figure_celltype(show=False):
    """Figure S1: figures.make_spike_spectrum_figure on the cell-type pkls."""
    return figures.make_spike_spectrum_figure(n_genes_list=[N_GENES], HVG_type="celltype",
                                              filename="figure_spectrum_celltype.pdf", show=show)


def make_KS_figure_celltype(show=False):
    """Figure S2: the KS figure on cell-type metacells, with the same
    datasets, layout and boxes as figures.make_KS_figure.

    The raw KS statistic falls with metacell size, and cell-type metacells
    are 10-100x larger than refined ones, so this figure's boxes sit lower
    than Figure 5's and the two y axes are not comparable by eye. What the
    figure is for is the comparison WITHIN a panel: the true box against the
    metacell and LML-RMT boxes.
    """
    return figures._make_KS_figure("celltype", "figure_normality_KS_celltype.pdf", show)


def make_normality_spectrum_figure_celltype():
    """Figure S3: figures.make_normality_spectrum_figure on the cell-type pkls."""
    os.makedirs(figures.FIGURE_DIR, exist_ok=True)
    fig = fu.make_normality_spectrum_figure(
        dataset_list=datasets.DEFAULT_DATASETS, n_genes=N_GENES, HVG_type="celltype",
        methods=figures.NORMALITY_SPECTRUM_METHODS, show=False)
    fig.savefig(os.path.join(figures.FIGURE_DIR, "figure_normality_spectrum_celltype.pdf"))
    return fig


def make_neighbor_distance_figure_celltype():
    """Figure S4: figures.make_neighbor_distance_figure on the cell-type pkls."""
    os.makedirs(figures.FIGURE_DIR, exist_ok=True)
    fig = fu.make_neighbor_distance_figure(
        dataset_list=datasets.DEFAULT_DATASETS, n_genes=N_GENES, HVG_type="celltype",
        show=False)
    fig.savefig(os.path.join(figures.FIGURE_DIR, "figure_neighbor_distance_celltype.pdf"))
    return fig


def make_knn_figure_celltype():
    """Figure S5: figures.make_knn_figure on the cell-type knn pkls
    (analysis_knn_celltype), which are built on first use."""
    os.makedirs(figures.FIGURE_DIR, exist_ok=True)
    fig = fu.make_knn_figure(dataset_list=datasets.DEFAULT_DATASETS, n_genes_list=[N_GENES],
                             HVG_type="celltype", show=False)
    fig.savefig(os.path.join(figures.FIGURE_DIR, "figure_knn_adjacency_celltype.pdf"))
    return fig


# (name, function, keyword arguments), in Supplementary Information order
FIGURE_STEPS = [
    ("spike spectrum celltype", make_spike_spectrum_figure_celltype, {}),
    ("KS celltype", make_KS_figure_celltype, {}),
    ("normality spectrum celltype", make_normality_spectrum_figure_celltype, {}),
    ("neighbor distance celltype", make_neighbor_distance_figure_celltype, {}),
    ("knn celltype", make_knn_figure_celltype, {}),
]


def make_all():
    """Draw the five Supplementary Information figures into figures.FIGURE_DIR."""
    for name, fn, kwargs in FIGURE_STEPS:
        fn(**kwargs)
        plt.close("all")
    return None
