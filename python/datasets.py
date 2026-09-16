import dataset_pbmc
import dataset_heart_large
import dataset_BMhemato
import dataset_BMMC
import dataset_Zebrafish
import dataset_Breast
import dataset_thymus
# single-donor datasets built from ../data/<Name>_dataset/ (see their READMEs)
import dataset_Fat
import dataset_Skin
import dataset_Ocular
import dataset_Breast_small

import os
import nb_model
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

DEFAULT_DATASETS = ["PBMC",  "Heart_large", "BMHemato", "BMMC",
                    "Zebrafish", "Thymus", "Breast",
                    # single-donor datasets added 2026-09-07 (see
                    # ../data/<Name>_dataset/README.md)
                    "Fat", "Skin", "Ocular", "Breast_small"]
#DEFAULT_N_GENES = [1000, 2000, 5000]
DEFAULT_N_GENES = [1000, 2000, 5000]

# upper bound on the number of PCs kept, regardless of where the elbow lands
MAX_K = 20

# how far up the elbow to sit; see find_elbow. 1.0 is the unweighted
# chord-distance elbow, larger values pick an earlier, larger singular value
ELBOW_WEIGHT = 20.0

# derived datasets live in flat folders of their own, separate from the
# per-dataset raw data directories that load_dataset_directory points at --
# those still hold the untouched source files each base dataset is built from
BASE_DIR = "../analysis_base/"
HVG_DIR = "../analysis_HVG/"

# the base h5ad each dataset_*.py module writes
BASE_FILENAMES = {"PBMC": "pbmc_base.h5ad",
                  "Heart_large": "heart_large_base.h5ad",
                  "BMHemato": "BMHemato_base.h5ad", "BMhemato": "BMHemato_base.h5ad",
                  "BMMC": "BMMC_base.h5ad", "Zebrafish": "Zebrafish_base.h5ad",
                  "Breast": "breast_base.h5ad",
                  "Thymus": "Thymus_base.h5ad",
                  "Fat": "Fat_base.h5ad", "Skin": "Skin_base.h5ad",
                  "Ocular": "Ocular_base.h5ad",
                  "Breast_small": "breast_small_base.h5ad"}


def get_base_filename(dataset):
    """Path of the base h5ad, creating the folder if it does not exist.

    Only the OUTPUT moves here: each dataset_*.py still reads its raw source
    files from load_dataset_directory(dataset), which is why they take the
    output path separately.
    """
    if dataset not in BASE_FILENAMES:
        raise ValueError(f"Dataset {dataset} not found")
    os.makedirs(BASE_DIR, exist_ok=True)
    return os.path.join(BASE_DIR, BASE_FILENAMES[dataset])


def get_HVG_filename(dataset, n_genes):
    """Path of the HVG h5ad, creating the folder if it does not exist."""
    os.makedirs(HVG_DIR, exist_ok=True)
    return os.path.join(HVG_DIR, f"{dataset}_HVG_genes{n_genes}.h5ad")


def get_HVG_info_filename(dataset):
    """Path of a dataset's HVG_info.csv. Prefixed by dataset because all of
    them now share one folder.
    """
    os.makedirs(HVG_DIR, exist_ok=True)
    return os.path.join(HVG_DIR, f"{dataset}_HVG_info.csv")

def load_dataset_directory(dataset):

    if dataset == "PBMC":
        dir = "../data/PBMC_dataset/"
    elif dataset == "Heart_large":
        dir = "../data/Heart_large_dataset/"
    elif dataset in ("BMHemato", "BMhemato"):
        dir = "../data/BMHemato_dataset/"
    elif dataset == "BMMC":
        dir = "../data/BMMC_dataset/"
    elif dataset == "Zebrafish":
        dir = "../data/Zebrafish_dataset/"
    elif dataset == "Breast":
        dir = "../data/Breast_dataset/"
    elif dataset == "Thymus":
        dir = "../data/Thymus_dataset/"
    elif dataset == "Fat":
        dir = "../data/Fat_dataset/"
    elif dataset == "Skin":
        dir = "../data/Skin_dataset/"
    elif dataset == "Ocular":
        dir = "../data/Ocular_dataset/"
    elif dataset == "Breast_small":
        dir = "../data/Breast_small_dataset/"
    else:
        raise ValueError(f"Dataset {dataset} not found")

    return dir


def load_HVG_dataset(dataset, n_genes):
    out_f = get_HVG_filename(dataset, n_genes)
    if os.path.isfile(out_f):
        return ad.read_h5ad(out_f)
    else:
        raise ValueError(f"Dataset {dataset} with {n_genes} genes not found")


def load_base_dataset(dataset, overwrite=False):
    base_f = load_dataset_directory(dataset)
    if dataset == "PBMC":
        return dataset_pbmc.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset == "Heart_large":
        return dataset_heart_large.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset in ("BMHemato", "BMhemato"):
        return dataset_BMhemato.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset == "BMMC":
        return dataset_BMMC.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset == "Zebrafish":
        return dataset_Zebrafish.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset == "Breast":
        return dataset_Breast.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset == "Thymus":
        return dataset_thymus.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset == "Fat":
        return dataset_Fat.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset == "Skin":
        return dataset_Skin.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset == "Ocular":
        return dataset_Ocular.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    elif dataset == "Breast_small":
        return dataset_Breast_small.load_base_dataset(base_f, overwrite=overwrite, out_f=get_base_filename(dataset))
    else:
        raise ValueError(f"Dataset {dataset} not found")

    return None

# the order create_base_datasets builds in; the single-donor datasets (see
# ../data/<Name>_dataset/README.md) come last.
BASE_BUILD_ORDER = ["PBMC", "Heart_large", "BMHemato", "BMMC", "Zebrafish",
                    "Breast", "Thymus", "Fat", "Skin", "Ocular", "Breast_small"]


def create_base_datasets(overwrite=False):
    """Build every base h5ad in BASE_BUILD_ORDER. With overwrite=False a
    dataset whose base h5ad exists is skipped without being read."""
    for dataset in BASE_BUILD_ORDER:
        if not overwrite and os.path.isfile(get_base_filename(dataset)):
            print(f"  base {dataset}: exists, skipping")
            continue
        load_base_dataset(dataset, overwrite=overwrite)
    return None


def find_elbow(values, weight=ELBOW_WEIGHT):
    """Index (as a count, so 1-based) of the elbow of a decreasing curve.

    Both axes are rescaled to [0, 1] first, so neither dominates on account
    of its units, and the elbow is the index minimizing

        weight * x_n + y_n

    A singular-value curve is convex and so lies entirely below the chord
    joining its endpoints; the usual "furthest from the chord" criterion is
    then exactly this expression with weight = 1, since maximizing
    |x_n + y_n - 1| reduces to minimizing x_n + y_n.

    weight controls how far up the elbow to sit. Raising it penalizes the
    index axis more, moving the minimum earlier -- fewer components, at a
    larger singular value. On PBMC/1000, weight 1 gives k=20 and weight 20
    gives k=8.
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return len(v)
    x = np.arange(len(v), dtype=float)
    x_n = (x - x[0]) / (x[-1] - x[0])
    y_n = (v - v[-1]) / (v[0] - v[-1])
    return int(np.argmin(weight * x_n + y_n)) + 1


def default_n_super_cells(dataset):
    if dataset in ("BMHemato", "BMhemato"):
        return dataset_BMhemato.DEFAULT_N_SUPER_CELLS
    elif dataset == "BMMC":
        return dataset_BMMC.DEFAULT_N_SUPER_CELLS
    elif dataset == "Zebrafish":
        return dataset_Zebrafish.DEFAULT_N_SUPER_CELLS
    elif dataset == "Breast":
        return dataset_Breast.DEFAULT_N_SUPER_CELLS
    elif dataset == "Thymus":
        return dataset_thymus.DEFAULT_N_SUPER_CELLS
    elif dataset == "Fat":
        return dataset_Fat.DEFAULT_N_SUPER_CELLS
    elif dataset == "Skin":
        return dataset_Skin.DEFAULT_N_SUPER_CELLS
    elif dataset == "Ocular":
        return dataset_Ocular.DEFAULT_N_SUPER_CELLS
    elif dataset == "Breast_small":
        return dataset_Breast_small.DEFAULT_N_SUPER_CELLS
    else:
        return 80


def create_HVG_dataset(dataset, n_genes, n_super_cells=None, overwrite=False,
                       poisson=False, s=None):
    """Build the HVG h5ad for a single (dataset, n_genes), or load it if it
    already exists and overwrite is False.

    s is the base AnnData. Pass it in when building several n_genes for the
    same dataset so it is read from disk once rather than once per n_genes;
    it is loaded here only if needed and not supplied.

    Returns the summary row for HVG_info.csv -- the caller aggregates those
    across n_genes (see create_all_HVG_datasets).
    """
    if n_super_cells is None:
        n_super_cells = default_n_super_cells(dataset)

    out_f = get_HVG_filename(dataset, n_genes)

    if not os.path.isfile(out_f) or overwrite:
        if s is None:
            s = load_base_dataset(dataset)

        print(f"Creating HVG dataset with {n_genes} genes")
        print("Computing highly variable genes")
        s_new = s.copy()
        sc.pp.highly_variable_genes(s_new, flavor="seurat_v3",
                                    inplace=True,
                                    n_top_genes=n_genes)
        s_new = s_new[:,s_new.var["highly_variable"]]
        s_new = nb_model.apply_metacell_decomposition(s_new, n_super_cells, partition=s.obs["cell_type"])

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
        k_elbow = find_elbow(sv)
        k = min(k_elbow, MAX_K)
        if k < k_elbow:
            print(f"PCA: {len(sv)} singular values, elbow at k={k_elbow}, capped to k={k}")
        else:
            print(f"PCA: {len(sv)} singular values, elbow at k={k}")

        # keep only the leading k columns of U -- the full U is (n_cells x
        # n_genes) and dwarfs the rest of the file (135 MB vs a 10 MB h5ad
        # for PBMC/1000), whereas the full sv is a few KB and worth keeping
        # so the elbow can be revisited without redoing the SVD
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


def create_all_HVG_datasets(dataset_list=DEFAULT_DATASETS, n_genes_list=DEFAULT_N_GENES,
                            overwrite=False, poisson=False):
    """Build every (dataset, n_genes) HVG h5ad, writing one HVG_info.csv per
    dataset.
    """
    for dataset in dataset_list:
        print("Processing dataset: ", dataset)

        # a dataset whose every h5ad and info csv already exist is skipped
        # outright: re-reading its h5ads only to rewrite the same csv is
        # slow (the summary row densifies X) and touches files that a
        # fill-in-the-gaps run should leave alone
        complete = all(os.path.isfile(get_HVG_filename(dataset, n_genes))
                       for n_genes in n_genes_list)
        if not overwrite and complete and os.path.isfile(get_HVG_info_filename(dataset)):
            print(f"  {dataset}: every HVG h5ad and the info csv exist, skipping")
            continue

        # load the base dataset once for the whole n_genes sweep, and only if
        # something actually needs building -- it is the expensive read here
        needs_build = any(
            overwrite or not os.path.isfile(get_HVG_filename(dataset, n_genes))
            for n_genes in n_genes_list)
        s = load_base_dataset(dataset) if needs_build else None

        info_rows = [create_HVG_dataset(dataset, n_genes, overwrite=overwrite,
                                        poisson=poisson, s=s)
                     for n_genes in n_genes_list]

        pd.DataFrame(info_rows).to_csv(get_HVG_info_filename(dataset), index=False)

    return None


