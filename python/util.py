import numpy as np
from scipy import sparse
import anndata as ad


def counts2ann(X, genes=None, barcodes=None, 
               min_gene_count=0, min_cell_count=0,
               obs=None, check_counts=False):
    
    if type(X) == np.ndarray:
        X = sparse.csr_matrix(X)
    s = ad.AnnData(X)
    
    if obs is not None:
        s.obs = obs
        s.obs.index = s.obs.index.astype(str)

    if barcodes is None:
        barcodes = np.array(["cell" + str(i) for i in range(X.shape[0])])
    if genes is None:
        genes = np.array(["gene" + str(i) for i in range(X.shape[1])])
    
    s.obs_names = barcodes
    s.var_names = genes
    s.var["gene"] = genes
    
    if check_counts:
        cell_counts = np.array(s.X.sum(axis=1)).flatten()  # sum rows (cells)
        s = s[cell_counts >= min_cell_count,:]
        gene_counts = np.array(s.X.sum(axis=0)).flatten()  # sum columns (genes)
        s = s[:,gene_counts >= min_gene_count]
 
    return s
