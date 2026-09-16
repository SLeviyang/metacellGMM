import os

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import datasets
import nb_model

# cell-type-as-metacell HVG datasets live in their own flat folder, parallel
# to datasets.HVG_DIR -- filenames follow the exact same pattern so the two
# directories are drop-in swappable for any consumer that takes a directory
CELLTYPE_DIR = "../analysis_HVG_celltype/"

# a cell type with fewer than this many cells is dropped entirely rather
# than kept as an undersized metacell -- the direct analogue of
# min_super_cell in nb_model.apply_metacell_decomposition, and of
# _supercell_labels_by_partition's "partitions smaller than min_super_cell
# are skipped" rule. There is no SuperCell step here to otherwise fail on a
# handful of cells, but the filter is kept anyway so a tiny cell type cannot
# masquerade as a well-estimated metacell downstream.
DEFAULT_MIN_CELLTYPE = 100


def get_HVG_celltype_filename(dataset, n_genes):
    """Path of the cell-type HVG h5ad, creating the folder if needed."""
    os.makedirs(CELLTYPE_DIR, exist_ok=True)
    return os.path.join(CELLTYPE_DIR, f"{dataset}_HVG_genes{n_genes}.h5ad")


def get_HVG_celltype_info_filename(dataset):
    """Path of a dataset's HVG_info.csv under CELLTYPE_DIR."""
    os.makedirs(CELLTYPE_DIR, exist_ok=True)
    return os.path.join(CELLTYPE_DIR, f"{dataset}_HVG_info.csv")


def load_HVG_celltype_dataset(dataset, n_genes):
    out_f = get_HVG_celltype_filename(dataset, n_genes)
    if os.path.isfile(out_f):
        return ad.read_h5ad(out_f)
    else:
        raise ValueError(f"Cell-type dataset {dataset} with {n_genes} genes not found")


def _apply_celltype_decomposition(s, partition, min_celltype=DEFAULT_MIN_CELLTYPE):
    """Cell-type analogue of nb_model.apply_metacell_decomposition: the same
    cell/gene QC and the same undersized-group drop, but "metacell" is set
    directly from partition (the cell type) instead of running SuperCell.

    partition is a categorical vector with one entry per row of `s` (as
    passed in, before any filtering here) -- s.obs["cell_type"], following
    the same calling convention datasets.create_HVG_dataset already uses
    for nb_model.apply_metacell_decomposition's own partition argument.

    There is no SuperCell call to protect from a degenerate small partition,
    so a cell type below min_celltype is simply excluded by the same
    undersized-group filter that would otherwise run after SuperCell --
    one filter does the work _supercell_labels_by_partition's skip and
    apply_metacell_decomposition's post-hoc size filter did together there.
    """
    print("Running cell-type decomposition")
    print("stabilizing library size and gene count")

    L_true = np.sum(s.X.toarray(), axis=1)
    n_feature = np.sum(s.X.toarray() > 0, axis=1)
    print("number of cells before preprocessing: ", s.shape[0])

    partition = np.asarray(partition)
    if len(partition) != s.n_obs:
        raise ValueError(f"partition has {len(partition)} entries but s has "
                         f"{s.n_obs} rows")

    # the cell mask has to carry the partition with it, or every later
    # assignment would be against the wrong cells
    keep = (L_true > 100) & (n_feature > 40)
    s = s[keep, :]
    partition = partition[keep]
    print("number of cells after cell filtering: ", s.shape[0])
    gene_counts = s.X.sum(axis=0)
    s = s[:, gene_counts > 100]
    print("number of cells after gene filtering: ", s.shape[0])
    print("min library size:", np.min(np.sum(s.X.toarray(), axis=1)))
    print("min gene count:", np.min(np.sum(s.X.toarray(), axis=0)))

    s = s.copy()
    s.obs["partition"] = pd.Categorical(partition.astype(str))
    s.obs["metacell"] = s.obs["partition"]

    # drop undersized cell types. One pass is enough: removing a cell type
    # cannot change any other cell type's size.
    sizes = s.obs["metacell"].astype(str).value_counts()
    big = set(sizes[sizes >= min_celltype].index)
    keep = s.obs["metacell"].astype(str).isin(big).to_numpy()
    print(f"dropping {int((~keep).sum())} cells in cell types smaller than "
          f"{min_celltype} ({len(sizes) - len(big)} of {len(sizes)} cell types)")
    s = s[keep, :].copy()

    if s.n_obs == 0:
        raise ValueError(
            f"every cell type was smaller than min_celltype={min_celltype}, "
            f"leaving no cells. Lower min_celltype.")

    # and the genes those removals left too sparse; this cannot change the
    # rows, so it needs no second pass either
    gene_counts = s.X.sum(axis=0)
    if not np.all(gene_counts > 50):
        s = s[:, np.asarray(gene_counts).ravel() > 50].copy()

    print("final number of cells: ", s.shape[0])
    print("min library size:", np.min(np.sum(s.X.toarray(), axis=1)))
    print("min feature count:", np.min(np.sum(s.X.toarray() > 0, axis=1)))
    print("min gene count:", np.min(np.sum(s.X.toarray(), axis=0)))
    print("number of cell-type metacells: ", s.obs["metacell"].nunique())

    return s


def create_HVG_celltype_dataset(dataset, n_genes, overwrite=False,
                                poisson=False, s=None):
    """Build the cell-type HVG h5ad for a single (dataset, n_genes), or load
    it if it already exists and overwrite is False.

    Identical to datasets.create_HVG_dataset in every respect except the
    decomposition step: s.obs["metacell"] is set to s.obs["cell_type"]
    (via _apply_celltype_decomposition) instead of being built by SuperCell,
    so every metacell is exactly one cell type's cells rather than a
    SuperCell-grouped subset of one. Everything downstream of that --
    post-decomposition filtering, the elbow-selected PCA, the NB fit -- is
    the same code datasets.create_HVG_dataset runs.

    s is the base AnnData, passed in when building several n_genes for the
    same dataset so it is read from disk once; loaded here only if needed
    and not supplied.

    Returns the summary row for HVG_info.csv -- the caller aggregates those
    across n_genes (see create_all_HVG_celltype_datasets).
    """
    out_f = get_HVG_celltype_filename(dataset, n_genes)

    if not os.path.isfile(out_f) or overwrite:
        if s is None:
            s = datasets.load_base_dataset(dataset)

        print(f"Creating cell-type HVG dataset with {n_genes} genes")
        print("Computing highly variable genes")
        s_new = s.copy()
        sc.pp.highly_variable_genes(s_new, flavor="seurat_v3",
                                    inplace=True,
                                    n_top_genes=n_genes)
        s_new = s_new[:, s_new.var["highly_variable"]]
        s_new = _apply_celltype_decomposition(s_new, s.obs["cell_type"])

        # just some safety filtering
        gene_counts = s_new.X.sum(axis=0)
        s_new = s_new[:, gene_counts > 50]
        L_all = s_new.X.sum(axis=1)
        s_new = s_new[L_all > 100, :]

        # PCA of the log-normalized, scaled counts. sc.pp.scale centers each
        # gene, so the SVD of s_copy.X IS the PCA: U * sv are the scores and
        # the rows of Vt are the loadings.
        s_copy = s_new.copy()
        sc.pp.normalize_total(s_copy, target_sum=1e4)
        sc.pp.log1p(s_copy)
        sc.pp.scale(s_copy)
        U, sv, _ = np.linalg.svd(s_copy.X, full_matrices=False)

        # choose the elbow of the singular values, capped at MAX_K
        k_elbow = datasets.find_elbow(sv)
        k = min(k_elbow, datasets.MAX_K)
        if k < k_elbow:
            print(f"PCA: {len(sv)} singular values, elbow at k={k_elbow}, capped to k={k}")
        else:
            print(f"PCA: {len(sv)} singular values, elbow at k={k}")

        # keep only the leading k columns of U -- the full U is (n_cells x
        # n_genes) and dwarfs the rest of the file, whereas the full sv is a
        # few KB and worth keeping so the elbow can be revisited without
        # redoing the SVD
        s_new.obsm["X_pca"] = U[:, :k] * sv[np.newaxis, :k]
        s_new.uns["pca"] = {"U": U[:, :k], "sv": sv[:k], "k": k}
        s_new.uns["full_sv"] = sv

        print("number of cells/genes after filtering: ", s_new.shape)
        print("Computing NB model")
        nb = nb_model.compute_metacell_NB(s_new, poisson=poisson)
        s_new.uns["nb_model"] = nb
        s_new.uns["poisson"] = poisson

        s_new.write(out_f, compression="gzip")
    else:
        s_new = ad.read_h5ad(out_f)

    X = s_new.X.toarray() if hasattr(s_new.X, "toarray") else np.asarray(s_new.X)
    return {
        "n_genes": n_genes,
        "n_cells": s_new.n_obs,
        "n_metacells": s_new.obs["metacell"].nunique(),
        "min_library_size": np.min(X.sum(axis=1)),
        "min_gene_count": np.min((X > 0).sum(axis=0)),
    }


def create_all_HVG_celltype_datasets(dataset_list=datasets.DEFAULT_DATASETS,
                                     n_genes_list=datasets.DEFAULT_N_GENES,
                                     overwrite=False, poisson=False):
    """Build every (dataset, n_genes) cell-type HVG h5ad, writing one
    HVG_info.csv per dataset under CELLTYPE_DIR.
    """
    for dataset in dataset_list:
        print("Processing dataset: ", dataset)

        # a dataset whose every cell-type h5ad and info csv already exist is
        # skipped outright, so a fill-in-the-gaps run neither re-reads its
        # h5ads nor rewrites the same csv
        complete = all(os.path.isfile(get_HVG_celltype_filename(dataset, n_genes))
                       for n_genes in n_genes_list)
        if not overwrite and complete and os.path.isfile(get_HVG_celltype_info_filename(dataset)):
            print(f"  {dataset}: every cell-type h5ad and the info csv exist, skipping")
            continue

        # load the base dataset once for the whole n_genes sweep, and only if
        # something actually needs building -- it is the expensive read here
        needs_build = any(
            overwrite or not os.path.isfile(get_HVG_celltype_filename(dataset, n_genes))
            for n_genes in n_genes_list)
        s = datasets.load_base_dataset(dataset) if needs_build else None

        info_rows = [create_HVG_celltype_dataset(dataset, n_genes, overwrite=overwrite,
                                                  poisson=poisson, s=s)
                     for n_genes in n_genes_list]

        pd.DataFrame(info_rows).to_csv(get_HVG_celltype_info_filename(dataset), index=False)

    return None
