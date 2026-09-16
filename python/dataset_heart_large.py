import scanpy as sc
import pandas as pd
import anndata as ad
import pdb
import matplotlib.pyplot as plt

import numpy as np
import os

import util
import datasets

base_f = "../data/Heart_large_dataset/"

def load_base_dataset(base_f, overwrite=False, out_f=None):

    tissue = "heart_large"
    out_f = out_f or (base_f + "heart_large_base.h5ad")
    os.makedirs(os.path.dirname(out_f) or '.', exist_ok=True)
    in_f = base_f + tissue + ".h5ad"

    if os.path.isfile(out_f) and not overwrite:
        return ad.read_h5ad(out_f)

    s = sc.read_h5ad(in_f)
    # .to_numpy() matters: scipy's sparse indexing calls .nonzero() on a
    # boolean index, and pandas Series dropped that method in 2.x, so
    # X[series, :] raises AttributeError. AnnData accepts either, but the
    # sparse matrix is indexed first, so the mask has to be an ndarray.
    idx = (s.obs["cell_source"] == "Sanger-Cells").to_numpy()
    X = s.raw.X[idx, :]
    s_raw = s[idx, :]

    s = util.counts2ann(X,
                        genes=s_raw.raw.var_names,
                        barcodes=s_raw.raw.obs_names,
                        min_gene_count=0,
                        min_cell_count=0,
                        obs=s_raw.obs)

    # per-cell QC. s.X is sparse, so both reductions come back as (n_obs, 1)
    # np.matrix rather than 1-D arrays -- np.asarray(...).ravel() flattens them
    # to the (n_obs,) that .obs needs. (np.matrix.flatten() would give (1, n_obs),
    # which is why the previous total_counts line did not assign cleanly.)
    # `X > 0` is used rather than getnnz(axis=1) so that any explicitly stored
    # zeros are not counted as detected genes.

    Xr = s.X
    gene_counts = Xr.sum(axis=0).flatten()
    idx = gene_counts > 50
    s = s[:,idx]
    s.obs["celltype"] = s.obs["cell_type"].to_numpy()


    print("Number of cells: ", s.n_obs)
    print("Number of genes: ", s.n_vars)
    print("Writing dataset to: ", out_f)


    s.write(out_f, compression="gzip")
    return s
   
