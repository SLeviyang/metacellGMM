import os

import anndata as ad
import h5py
import numpy as np
from scipy import sparse


RAW_RNA_FILE = "HBCA_global_raw.h5ad"
BASE_DATASET_FILE = "breast_base.h5ad"

MIN_TOTAL_COUNTS = 800
MAX_TOTAL_COUNTS = 50000
MIN_GENES_BY_COUNTS = 500
MAX_GENES_BY_COUNTS = 6500
MAX_PCT_COUNTS_MT = 8
MIN_CELLS_PER_GENE = 100
MT_PREFIX = "MT-"
TARGET_N_CELLS = 200000
RANDOM_SEED = 1

# Target metacell count handed to supercell; clean_supercells then drops the
# ones below its 100-cell floor, so the realized count comes out lower (on the
# other datasets roughly 60% of the target survives). Matches BMHemato's and
# Thymus's setting, which gives Breast coarser metacells than either -- it is
# downsampled to 200k cells against their ~60k, so expect on the order of
# 2700 cells per metacell rather than their ~750.
DEFAULT_N_SUPER_CELLS = 120


def _raw_x_group(h5):
    if "raw" in h5 and "X" in h5["raw"]:
        return h5["raw/X"]
    if "X" in h5:
        return h5["X"]
    raise ValueError("Could not find X or raw/X in h5ad file.")


def _mitochondrial_gene_mask(raw_var, mt_prefix=MT_PREFIX):
    if "feature_name" in raw_var.columns:
        gene_names = raw_var["feature_name"].astype(str)
    else:
        gene_names = raw_var.index.astype(str)
    return np.asarray(gene_names.str.startswith(mt_prefix))


def _compute_qc_metrics_from_raw_counts(in_f, raw_var, mt_prefix=MT_PREFIX,
                                        chunk_rows=5000):
    mt_gene = _mitochondrial_gene_mask(raw_var, mt_prefix=mt_prefix)

    with h5py.File(in_f, "r") as h5:
        group = _raw_x_group(h5)
        if not {"data", "indices", "indptr"}.issubset(group.keys()):
            raise ValueError("Expected raw.X to be backed by a CSR sparse matrix.")

        data_ds = group["data"]
        indices_ds = group["indices"]
        indptr = group["indptr"][:]
        n_obs = len(indptr) - 1
        n_vars = raw_var.shape[0]

        total_counts = np.zeros(n_obs, dtype=np.float64)
        n_genes_by_counts = np.diff(indptr).astype(np.int32)
        mt_counts = np.zeros(n_obs, dtype=np.float64)

        max_abs_fractional = 0.0
        mt_lookup = np.zeros(n_vars, dtype=bool)
        mt_lookup[np.flatnonzero(mt_gene)] = True

        for row_start in range(0, n_obs, chunk_rows):
            row_end = min(row_start + chunk_rows, n_obs)
            p0 = int(indptr[row_start])
            p1 = int(indptr[row_end])
            data = data_ds[p0:p1]
            indices = indices_ds[p0:p1]
            row_nnz = n_genes_by_counts[row_start:row_end]

            if data.size == 0:
                continue

            if not np.all(np.isfinite(data)):
                raise ValueError("raw.X contains non-finite values.")
            if np.min(data) < 0:
                raise ValueError("raw.X contains negative values.")

            frac = np.abs(data - np.rint(data))
            max_abs_fractional = max(max_abs_fractional, float(np.max(frac)))

            offsets = np.r_[0, np.cumsum(row_nnz[:-1])]
            nonempty = row_nnz > 0
            sums = np.zeros(row_end - row_start, dtype=np.float64)
            sums[nonempty] = np.add.reduceat(data, offsets[nonempty])
            total_counts[row_start:row_end] = sums

            mt_mask = mt_lookup[indices]
            if np.any(mt_mask):
                local_rows = np.repeat(np.arange(row_end - row_start,
                                                 dtype=np.int32),
                                       row_nnz)
                mt_counts[row_start:row_end] = np.bincount(
                    local_rows[mt_mask],
                    weights=data[mt_mask],
                    minlength=row_end - row_start,
                )

        if max_abs_fractional > 1e-8:
            raise ValueError(
                "raw.X does not look like raw counts: max abs(value - round(value)) "
                f"is {max_abs_fractional}."
            )

    pct_counts_mt = np.divide(mt_counts,
                              total_counts,
                              out=np.zeros_like(mt_counts, dtype=float),
                              where=total_counts > 0) * 100

    return {
        "total_counts": total_counts,
        "n_genes_by_counts": n_genes_by_counts,
        "pct_counts_mt": pct_counts_mt,
    }


def _cell_filter(qc,
                 min_total_counts=MIN_TOTAL_COUNTS,
                 max_total_counts=MAX_TOTAL_COUNTS,
                 min_genes_by_counts=MIN_GENES_BY_COUNTS,
                 max_genes_by_counts=MAX_GENES_BY_COUNTS,
                 max_pct_counts_mt=MAX_PCT_COUNTS_MT):
    return (
        (qc["total_counts"] >= min_total_counts)
        & (qc["total_counts"] <= max_total_counts)
        & (qc["n_genes_by_counts"] >= min_genes_by_counts)
        & (qc["n_genes_by_counts"] <= max_genes_by_counts)
        & (qc["pct_counts_mt"] <= max_pct_counts_mt)
    )


def _gene_filter_after_cell_qc(X, min_cells_per_gene=MIN_CELLS_PER_GENE):
    if sparse.issparse(X):
        cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        cells_per_gene = np.sum(X > 0, axis=0)
    return cells_per_gene >= min_cells_per_gene, cells_per_gene


def load_base_dataset(base_f, overwrite=False, out_f=None):
    out_f = out_f or os.path.join(base_f, BASE_DATASET_FILE)
    os.makedirs(os.path.dirname(out_f) or ".", exist_ok=True)
    in_f = os.path.join(base_f, RAW_RNA_FILE)

    if os.path.isfile(out_f) and not overwrite:
        return ad.read_h5ad(out_f)

    s_raw = ad.read_h5ad(in_f, backed="r")
    if s_raw.raw is None:
        s_raw.file.close()
        raise ValueError("Expected the breast source h5ad to contain .raw.X.")

    raw_var = s_raw.raw.var.copy()
    qc = _compute_qc_metrics_from_raw_counts(in_f, raw_var)
    keep_cells = _cell_filter(qc)
    cell_idx = np.flatnonzero(keep_cells)

    X_cell = s_raw.raw.X[cell_idx, :]
    if not sparse.issparse(X_cell):
        X_cell = sparse.csr_matrix(np.asarray(X_cell))
    else:
        X_cell = X_cell.tocsr()

    keep_genes, cells_per_gene = _gene_filter_after_cell_qc(X_cell)
    X = X_cell[:, keep_genes].copy()

    obs = s_raw.obs.iloc[cell_idx].copy()
    obs["total_counts"] = qc["total_counts"][cell_idx]
    obs["n_genes_by_counts"] = qc["n_genes_by_counts"][cell_idx]
    obs["pct_counts_mt"] = qc["pct_counts_mt"][cell_idx]
    # both spellings are carried: analysis_umap reads "cell_type", the older
    # modules read "celltype". Missing here would only surface much later, in
    # the umap step, so it is caught now.
    if "cell_type" not in obs.columns:
        s_raw.file.close()
        raise ValueError('Missing Breast annotation column: "cell_type"')
    obs["celltype"] = obs["cell_type"].to_numpy()

    var = raw_var.iloc[np.flatnonzero(keep_genes)].copy()
    var["cells_by_counts"] = cells_per_gene[keep_genes]

    s = ad.AnnData(X=X, obs=obs, var=var)
    s.obs_names = s_raw.obs_names[cell_idx].astype(str)
    s.var_names = s_raw.raw.var_names[np.flatnonzero(keep_genes)].astype(str)
    s.obs_names_make_unique()
    s.var_names_make_unique()

    s.uns["Breast_qc_filter"] = {
        "source_file": RAW_RNA_FILE,
        "count_matrix": "raw.X",
        "min_total_counts": MIN_TOTAL_COUNTS,
        "max_total_counts": MAX_TOTAL_COUNTS,
        "min_genes_by_counts": MIN_GENES_BY_COUNTS,
        "max_genes_by_counts": MAX_GENES_BY_COUNTS,
        "max_pct_counts_mt": MAX_PCT_COUNTS_MT,
        "min_cells_per_gene_after_cell_qc": MIN_CELLS_PER_GENE,
        "mt_prefix": MT_PREFIX,
        "n_cells_before_qc": int(s_raw.n_obs),
        "n_genes_before_qc": int(s_raw.raw.n_vars),
        "n_cells_after_cell_qc": int(keep_cells.sum()),
        "n_genes_after_gene_qc": int(keep_genes.sum()),
    }

    s_raw.file.close()

    if s.n_obs > TARGET_N_CELLS:
        rng = np.random.default_rng(RANDOM_SEED)
        keep = np.sort(rng.choice(s.n_obs, size=TARGET_N_CELLS, replace=False))
        s = s[keep, :].copy()

    s.uns["Breast_qc_filter"]["target_n_cells"] = TARGET_N_CELLS
    s.uns["Breast_qc_filter"]["random_seed"] = RANDOM_SEED
    s.uns["Breast_qc_filter"]["n_cells_after_downsampling"] = int(s.n_obs)

    print("Number of cells: ", s.n_obs)
    print("Number of genes: ", s.n_vars)
    print("Writing dataset to: ", out_f)

    s.write(out_f, compression="gzip")
    return s
