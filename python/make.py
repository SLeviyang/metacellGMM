"""Top-level build script: turns raw datasets into every cached artifact the
analysis modules read, then draws the figures.

Stages, in dependency order, and the scope each one sweeps:

  make_datasets  ->  ../analysis_base/                                (h5ad, kept)
                     ../analysis_HVG/, ../analysis_HVG_refined/       (h5ad, DEFAULT_N_GENES)
                     ../analysis_HVG_celltype/                        (h5ad, CELLTYPE_N_GENES)
  make_bulk      ->  ../analysis_bulk_refined/                        (pkl, DEFAULT_N_GENES)
  make_spike     ->  ../analysis_spike_refined/                       (pkl, SPIKE_N_GENES)
                     ../analysis_spike_celltype/                      (pkl, CELLTYPE_N_GENES)
  make_downstream->  ../analysis_knn_refined/, ../analysis_umap_refined/ (pkl, SPIKE_N_GENES)
  make_figures   ->  ../figures/                                      (pdf)

make_all runs all five. Stage 1 must precede 2 and 3, which both read the
HVG h5ads; 2 and 3 are independent of each other; 4 reads the refined
spike pkls; 5 reads everything.

Every stage takes an `overwrite` flag, defaulting to False: the sweeps skip
every dataset whose files already exist and build only what is missing, so
adding a dataset to datasets.DEFAULT_DATASETS and running make_all() builds
just that dataset. Existing h5ads, pkls and info csvs are neither rewritten
nor re-read; only the figures are always redrawn. Pass overwrite=True after
a model change (e.g. the per-metacell nb["L"]) to regenerate every
artifact. A combination whose input is missing is skipped with a warning
by the sweep itself rather than aborting the run.

Not built here: the unrefined bulk and spike pkls, the cell-type bulk pkls
and the metacell graph pkls -- no figure in the manuscript reads them. Their
make_all_* functions remain importable for building by hand.
"""

import time

import analysis_bulk as abulk
import analysis_spike as asp
import analysis_spike_celltype as asc

import datasets
import datasets_refined as dr
import datasets_celltype as dcelltype

import analysis_umap as aumap
import analysis_knn as aknn

import figures
import figures_util as fu
import table

# the spike pkls and everything downstream of them are built at one gene
# count; the HVG h5ads and bulk pkls sweep datasets.DEFAULT_N_GENES
SPIKE_N_GENES = [1000]
CELLTYPE_N_GENES = [1000]


def make_datasets(overwrite=False):
    """Build the HVG h5ads: unrefined, refined, then cell-type. The base
    h5ads are kept -- they carry no nb model, so a model change does not
    touch them."""
    print("\n=== base datasets ===")
    datasets.create_base_datasets(overwrite=False)

    print("\n=== HVG datasets: unrefined ===")
    datasets.create_all_HVG_datasets(overwrite=overwrite)

    print("\n=== HVG datasets: refined ===")
    dr.create_all_HVG_refined_datasets(overwrite=overwrite)

    print("\n=== HVG datasets: celltype ===")
    dcelltype.create_all_HVG_celltype_datasets(n_genes_list=CELLTYPE_N_GENES,
                                               overwrite=overwrite)
    return None


def make_bulk(overwrite=False):
    """Build the refined bulk pkls at every gene count."""
    print("\n=== bulk pkls: refined ===")
    abulk.make_all_bulk_pkl(overwrite=overwrite, refined=True)
    return None


def make_spike(overwrite=False):
    """Build the refined spike pkls, then the cell-type ones."""
    print("\n=== spike pkls: refined ===")
    asp.make_all_spike_pkl(n_genes_list=SPIKE_N_GENES, overwrite=overwrite,
                           refined=True)

    print("\n=== spike pkls: celltype ===")
    asc.make_all_spike_pkl_celltype(n_genes_list=CELLTYPE_N_GENES,
                                    overwrite=overwrite)
    return None


def make_downstream(overwrite=False):
    """Build the knn and umap pkls from the refined spike pkls, and the knn
    replicate errors the knn table reads (beside the knn pkls)."""
    print("\n=== knn pkls: refined ===")
    aknn.make_all_knn_pkl(n_genes_list=SPIKE_N_GENES, overwrite=overwrite,
                          refined=True)

    print("\n=== knn replicate errors: refined ===")
    for dataset in datasets.DEFAULT_DATASETS:
        for n_genes in SPIKE_N_GENES:
            fu.compute_knn_replicate_errors(dataset, n_genes, n_rep=table.KNN_REPLICATES,
                                            overwrite=overwrite, refined=True)

    print("\n=== umap pkls: refined ===")
    aumap.make_all_umap_pkl(n_genes_list=SPIKE_N_GENES, overwrite=overwrite,
                            refined=True)
    return None


# the manuscript's figures in order of appearance; each entry is the
# figures.py function and the keyword arguments that keep it headless.
# make_bulk_figure_celltype is left out: its cell-type bulk pkls are not
# built by this script.
FIGURE_STEPS = [
    ("bulk", figures.make_bulk_figure, {}),
    ("bulk SI", figures.make_bulk_figure_SI, {}),
    ("MP", figures.make_MP_figure, {}),
    ("spike spectrum", figures.make_spike_spectrum_figure, {"show": False}),
    ("KS", figures.make_KS_figure, {"show": False}),
    ("normality spectrum", figures.make_normality_spectrum_figure, {}),
    ("neighbor distance", figures.make_neighbor_distance_figure, {}),
    ("knn", figures.make_knn_figure, {}),
    ("visualize", figures.make_visualize_figure, {"show": False}),
    # the dataset table is a "figure" too: ../figures/datasets.tex
    ("datasets table", table.make_datasets_table, {}),
    ("knn table", table.make_knn_table, {}),
    ("processing table", table.make_processing_table, {}),
]


def make_figures():
    """Draw every manuscript figure into ../figures/, printing each one's
    name and elapsed time so a long run stays readable."""
    print("\n=== figures ===")
    for name, fn, kwargs in FIGURE_STEPS:
        t0 = time.time()
        fn(**kwargs)
        print(f"  figure {name}: done in {time.time() - t0:.0f}s")
    return None


def make_all_for_bulk(overwrite=False):
    make_datasets(overwrite=overwrite)
    make_bulk(overwrite=overwrite)
    return None


def make_all(overwrite=False):
    """Run every stage in dependency order, then draw the figures.

    With the default overwrite=False only the missing artifacts are built,
    so the run's cost is that of the datasets added since the last run;
    overwrite=True rebuilds everything for every dataset in
    datasets.DEFAULT_DATASETS.
    """
    # every stage sweeps datasets.DEFAULT_DATASETS, so this is the whole run's
    # scope; printed up front because the stages take hours and a dataset
    # missing from the list would otherwise only become apparent from what is
    # absent at the end
    print("=== datasets to be processed ===")
    for dataset in datasets.DEFAULT_DATASETS:
        print(f"  {dataset}")
    print(f"HVG / bulk n_genes: {datasets.DEFAULT_N_GENES}")
    print(f"spike / knn / umap n_genes: {SPIKE_N_GENES}")
    print(f"celltype n_genes: {CELLTYPE_N_GENES}")
    print(f"overwrite: {overwrite}")

    make_datasets(overwrite=overwrite)
    make_bulk(overwrite=overwrite)
    make_spike(overwrite=overwrite)
    make_downstream(overwrite=overwrite)
    make_figures()
    return None
