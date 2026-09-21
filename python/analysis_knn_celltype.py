"""k-NN index cache for the cell-type-as-metacell partition: the counterpart
of analysis_knn.anndata2knn for the analysis_spike_celltype pkls, read by
figures_celltype.make_knn_figure_celltype through figures_util.load_knn.

Mirrors anndata2knn's construction exactly -- the same k (the pkl's PCA
dimension), the same k_nn, the same U[:, :k] * sv[:k] embedding and the same
analysis_knn.knn_indices call -- for analytic and the four spectra the
manuscript figures draw, and writes the same dict layout, so
figures_util.compute_knn_edge_fractions reads either cache. Left out: the
<base>_proj_i families and the split-half k-NNs anndata2knn also computes,
which no cell-type figure reads. cell_labels come from the spike pkl itself,
which stores them in the specs' own row order, rather than from the h5ad.
"""
import os
import pickle

import numpy as np

import analysis_knn as ak
import analysis_spike_celltype as asc
import datasets

KNN_CELLTYPE_DIR = "../analysis_knn_celltype/"

# the spectra given a k-NN besides analytic, in asc.SPEC_KEYS' names
KNN_CELLTYPE_METHODS = ["normal", "nb", "perm", "true"]


def get_knn_celltype_filename(dataset, n_genes):
    """Path of the cell-type knn pkl, creating the folder if needed."""
    os.makedirs(KNN_CELLTYPE_DIR, exist_ok=True)
    return os.path.join(KNN_CELLTYPE_DIR, f"{dataset}_HVG_{n_genes}.pkl")


def has_knn_celltype_result(dataset, n_genes):
    return os.path.isfile(get_knn_celltype_filename(dataset, n_genes))


def anndata2knn_celltype(dataset, n_genes, k_nn=10, overwrite=False):
    """Compute (or load, if cached) the per-cell k-NN index arrays of the
    analytic, normal, nb, perm and true embeddings of the cell-type spike
    pkl, in the dict layout analysis_knn.anndata2knn writes: "cell_labels",
    "mixture_order", "k", "k_nn" and one "indices_<method>" per method.

    Requires the cell-type spike pkl (analysis_spike_celltype) to exist.
    """
    out_f = get_knn_celltype_filename(dataset, n_genes)
    if os.path.isfile(out_f) and not overwrite:
        with open(out_f, "rb") as f:
            return pickle.load(f)

    if not asc.has_spike_celltype_result(dataset, n_genes):
        raise ValueError(f"missing analysis_spike_celltype pkl for {dataset} HVG {n_genes}")
    result = asc.load_spike_celltype_result(dataset, n_genes)

    cell_labels = np.asarray(result["cell_labels"])
    mixture_order = ak.mixture_order_from_labels(cell_labels)
    k = np.asarray(result["spec_nb"]["U"]).shape[1]

    specs = {"analytic": result["spec_analytic"]}
    specs.update({method: result[asc.SPEC_KEYS[method]] for method in KNN_CELLTYPE_METHODS})

    knn_result = {"cell_labels": cell_labels, "mixture_order": mixture_order,
                  "k": k, "k_nn": k_nn}
    for method, spec in specs.items():
        W = np.asarray(spec["U"])[:, :k] * np.asarray(spec["sv"])[:k]
        print(f"Computing {method} k-NN for {dataset} HVG {n_genes} (celltype)")
        knn_result[f"indices_{method}"] = ak.knn_indices(W, k_nn)

    with open(out_f, "wb") as f:
        pickle.dump(knn_result, f)
    return knn_result


def make_all_knn_pkl_celltype(datasets_list=datasets.DEFAULT_DATASETS,
                              n_genes_list=[1000], k_nn=10, overwrite=False):
    """Build the cell-type knn pkl for every (dataset, n_genes) whose spike
    pkl exists; a combination without one is skipped with a warning."""
    for dataset in datasets_list:
        for n_genes in n_genes_list:
            if not asc.has_spike_celltype_result(dataset, n_genes):
                print(f"  SKIPPING {dataset}/{n_genes}: no cell-type spike pkl")
                continue
            anndata2knn_celltype(dataset, n_genes, k_nn=k_nn, overwrite=overwrite)
    return None
