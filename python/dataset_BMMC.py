import anndata as ad
import numpy as np
import os
from scipy import sparse

import util


RAW_RNA_FILE = "Leucken2021.h5ad"
BASE_DATASET_FILE = "BMMC_base.h5ad"
DEFAULT_N_SUPER_CELLS = 80

MIN_TOTAL_COUNTS = 500
MAX_TOTAL_COUNTS = 30000
MIN_GENES_BY_COUNTS = 200
MAX_GENES_BY_COUNTS = 8000
MAX_PCT_COUNTS_MT = 20
MIN_GENE_COUNTS = 50


def compute_qc_metrics(s, mt_prefix="MT-"):
    X = s.X
    total_counts = np.asarray(X.sum(axis=1)).ravel()

    if sparse.issparse(X):
        n_genes_by_counts = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        n_genes_by_counts = np.sum(X > 0, axis=1)

    mt_gene = np.asarray(s.var_names.str.startswith(mt_prefix))
    mt_counts = np.asarray(X[:, mt_gene].sum(axis=1)).ravel()
    pct_counts_mt = np.divide(mt_counts,
                              total_counts,
                              out=np.zeros_like(mt_counts, dtype=float),
                              where=total_counts > 0) * 100

    s.obs["total_counts"] = total_counts
    s.obs["n_genes_by_counts"] = n_genes_by_counts
    s.obs["pct_counts_mt"] = pct_counts_mt
    return s


def filter_cells(s,
                 min_total_counts=MIN_TOTAL_COUNTS,
                 max_total_counts=MAX_TOTAL_COUNTS,
                 min_genes_by_counts=MIN_GENES_BY_COUNTS,
                 max_genes_by_counts=MAX_GENES_BY_COUNTS,
                 max_pct_counts_mt=MAX_PCT_COUNTS_MT):
    if "total_counts" not in s.obs or "n_genes_by_counts" not in s.obs or "pct_counts_mt" not in s.obs:
        compute_qc_metrics(s)

    idx = (
        (s.obs["total_counts"] >= min_total_counts)
        & (s.obs["total_counts"] <= max_total_counts)
        & (s.obs["n_genes_by_counts"] >= min_genes_by_counts)
        & (s.obs["n_genes_by_counts"] <= max_genes_by_counts)
        & (s.obs["pct_counts_mt"] <= max_pct_counts_mt)
    )
    return s[idx, :].copy()


def filter_genes(s, min_gene_counts=MIN_GENE_COUNTS):
    gene_counts = np.asarray(s.X.sum(axis=0)).ravel()
    idx = gene_counts > min_gene_counts
    return s[:, idx].copy()


def load_base_dataset(base_f, overwrite=False, out_f=None):
    out_f = out_f or os.path.join(base_f, BASE_DATASET_FILE)
    os.makedirs(os.path.dirname(out_f) or '.', exist_ok=True)
    in_f = os.path.join(base_f, RAW_RNA_FILE)

    if os.path.isfile(out_f) and not overwrite:
        return ad.read_h5ad(out_f)

    # Leucken2021.h5ad is already RNA-only, raw counts in .X, with a
    # ready-made "cell_type" column -- no relabeling needed, unlike BMHemato.
    s_raw = ad.read_h5ad(in_f)
    s = util.counts2ann(s_raw.X,
                        genes=s_raw.var_names,
                        barcodes=s_raw.obs_names,
                        min_gene_count=0,
                        min_cell_count=0,
                        obs=s_raw.obs)

    s.var_names_make_unique()
    s.obs_names_make_unique()
    compute_qc_metrics(s)
    s = filter_cells(s)
    s = filter_genes(s)

    s.uns["BMMC_qc_filter"] = {
        "min_total_counts": MIN_TOTAL_COUNTS,
        "max_total_counts": MAX_TOTAL_COUNTS,
        "min_genes_by_counts": MIN_GENES_BY_COUNTS,
        "max_genes_by_counts": MAX_GENES_BY_COUNTS,
        "max_pct_counts_mt": MAX_PCT_COUNTS_MT,
        "min_gene_counts": MIN_GENE_COUNTS,
    }

    print("Number of cells: ", s.n_obs)
    print("Number of genes: ", s.n_vars)
    print("Writing dataset to: ", out_f)


    s.write(out_f, compression="gzip")
    return s
