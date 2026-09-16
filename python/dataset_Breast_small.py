"""Breast_small: one donor (HBCA_Donor_33) of the Human Breast Cell Atlas,
kept whole.

Source: ../data/Breast_small_dataset/HBCA_Donor_33_breast_raw.h5ad, a
single-donor extraction of the same HBCA global raw file that
dataset_Breast.py is built from, with raw UMI counts in X for all 33,604
genes (see the README in that folder). The donor (38-year-old BRCA1
carrier, prophylactic mastectomy, 10x 3' v3) has 39,446 cells in four
libraries: two epithelial-enriched unsorted libraries processed the same
day, one stroma-enriched library three weeks later, and one small
LASP-sorted fraction. obs["sample"] names the library.

The point of this dataset is the downsample that dataset_Breast.py
applies: it keeps 200,000 of the 782,667 QC-passing cells of all 55 donors
at random, which leaves only 9,440 of this donor's cells. Here the SAME
cell and gene QC as dataset_Breast.py is applied (800-50,000 UMIs,
500-6,500 genes, at most 8% mitochondrial reads, genes detected in at
least 100 cells) and nothing is downsampled, so the two breast datasets
stay comparable. The cuts keep 36,645 of 39,446 cells (92.9%).
"""
import anndata as ad
import numpy as np
import os
from scipy import sparse


RAW_RNA_FILE = "HBCA_Donor_33_breast_raw.h5ad"
BASE_DATASET_FILE = "breast_small_base.h5ad"
DEFAULT_N_SUPER_CELLS = 80

# the column naming one 10x library
SAMPLE_COLUMN = "sampleID"

# identical to dataset_Breast.py
MIN_TOTAL_COUNTS = 800
MAX_TOTAL_COUNTS = 50000
MIN_GENES_BY_COUNTS = 500
MAX_GENES_BY_COUNTS = 6500
MAX_PCT_COUNTS_MT = 8
MIN_CELLS_PER_GENE = 100


def _gene_symbols(s):
    """Gene symbols for MT- detection; the var index is Ensembl IDs."""
    if "feature_name" in s.var:
        return s.var["feature_name"].astype(str)
    return s.var_names.astype(str)


def _preserve_source_column(s, column):
    """Rename an authors' obs column out of the way before we overwrite it."""
    if column in s.obs and f"{column}_source" not in s.obs:
        s.obs[f"{column}_source"] = s.obs[column].to_numpy()


def compute_qc_metrics(s, mt_prefix="MT-"):
    X = s.X
    total_counts = np.asarray(X.sum(axis=1)).ravel()

    if sparse.issparse(X):
        n_genes_by_counts = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        n_genes_by_counts = np.sum(X > 0, axis=1)

    mt_gene = np.asarray(_gene_symbols(s).str.startswith(mt_prefix))
    mt_counts = np.asarray(X[:, mt_gene].sum(axis=1)).ravel()
    # 100 * mt / total, not (mt / total) * 100: the latter rounds a cell
    # sitting exactly on the cap (the source's own QC boundary) a hair above
    # it and drops it, which is not the intended "at most the cap"
    pct_counts_mt = np.divide(100.0 * mt_counts,
                              total_counts,
                              out=np.zeros_like(mt_counts, dtype=float),
                              where=total_counts > 0)

    for column in ("total_counts", "n_genes_by_counts", "pct_counts_mt"):
        _preserve_source_column(s, column)
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
    needed = {"total_counts", "n_genes_by_counts", "pct_counts_mt"}
    if not needed.issubset(s.obs.columns):
        compute_qc_metrics(s)

    idx = (
        (s.obs["total_counts"] >= min_total_counts)
        & (s.obs["total_counts"] <= max_total_counts)
        & (s.obs["n_genes_by_counts"] >= min_genes_by_counts)
        & (s.obs["n_genes_by_counts"] <= max_genes_by_counts)
        & (s.obs["pct_counts_mt"] <= max_pct_counts_mt)
    )
    return s[idx.to_numpy(), :].copy()


def filter_genes(s, min_cells_per_gene=MIN_CELLS_PER_GENE):
    """Same rule as dataset_Breast.py: keep genes detected in at least
    min_cells_per_gene of the QC-passing cells."""
    X = s.X
    if sparse.issparse(X):
        cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        cells_per_gene = np.sum(X > 0, axis=0)
    idx = cells_per_gene >= min_cells_per_gene
    s = s[:, idx].copy()
    s.var["cells_by_counts"] = cells_per_gene[idx]
    return s


def add_cell_type_annotations(s):
    """cell_type (CELLxGENE label), its older-module alias celltype, sample
    (one 10x library) and batch (the donor, constant in a single-donor
    dataset, matching the batch = donor convention of the other loaders)."""
    if "cell_type" not in s.obs:
        raise ValueError('Missing Breast_small annotation column: "cell_type"')
    if SAMPLE_COLUMN not in s.obs:
        raise ValueError(f'Missing Breast_small library column: "{SAMPLE_COLUMN}"')

    s.obs["cell_type"] = s.obs["cell_type"].astype(str).to_numpy()
    s.obs["celltype"] = s.obs["cell_type"].to_numpy()
    for column in ("sample", "batch"):
        _preserve_source_column(s, column)
    s.obs["sample"] = s.obs[SAMPLE_COLUMN].astype(str).to_numpy()
    s.obs["batch"] = s.obs["donor_id"].astype(str).to_numpy()
    return s


def load_base_dataset(base_f, overwrite=False, out_f=None):
    out_f = out_f or os.path.join(base_f, BASE_DATASET_FILE)
    os.makedirs(os.path.dirname(out_f) or '.', exist_ok=True)
    in_f = os.path.join(base_f, RAW_RNA_FILE)

    if os.path.isfile(out_f) and not overwrite:
        s = ad.read_h5ad(out_f)
        if "cell_type" not in s.obs or "sample" not in s.obs or "batch" not in s.obs:
            s = add_cell_type_annotations(s)
            s.write(out_f, compression="gzip")
        return s

    s = ad.read_h5ad(in_f)
    n_cells_start, n_genes_start = s.shape
    s.var_names_make_unique()
    s.obs_names_make_unique()
    compute_qc_metrics(s)
    s = filter_cells(s)
    n_cells_after_qc = s.n_obs
    s = filter_genes(s)
    s = add_cell_type_annotations(s)

    s.uns["Breast_small_qc_filter"] = {
        "source_file": RAW_RNA_FILE,
        "min_total_counts": MIN_TOTAL_COUNTS,
        "max_total_counts": MAX_TOTAL_COUNTS,
        "min_genes_by_counts": MIN_GENES_BY_COUNTS,
        "max_genes_by_counts": MAX_GENES_BY_COUNTS,
        "max_pct_counts_mt": MAX_PCT_COUNTS_MT,
        "min_cells_per_gene_after_cell_qc": MIN_CELLS_PER_GENE,
        "n_cells_start": int(n_cells_start),
        "n_cells_after_cell_qc": int(n_cells_after_qc),
        "n_genes_start": int(n_genes_start),
        "n_genes_after_gene_qc": int(s.n_vars),
        "downsampled": False,
    }

    print("Number of cells: ", s.n_obs)
    print("Number of genes: ", s.n_vars)
    print("Writing dataset to: ", out_f)

    s.write(out_f, compression="gzip")
    return s
