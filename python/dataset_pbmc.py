from numpy.ma import outerproduct
import scanpy as sc
import pandas as pd
import anndata as ad
import pdb

import numpy as np
import os
import util
import datasets


def load_base_dataset(base_f, overwrite=False, out_f=None):
    
    dataset_name = "PBMC"
    out_f = out_f or (base_f + "pbmc_base.h5ad")
    os.makedirs(os.path.dirname(out_f) or '.', exist_ok=True)
    metadata_f = base_f + '68k_pbmc_barcodes_annotation.tsv'

    if os.path.isfile(out_f) and not overwrite:
        return ad.read_h5ad(out_f)
  
    m = pd.read_csv(metadata_f, header=0, sep="\t")
    # X is initially genes by cells
    s = sc.read_10x_mtx(base_f)
    s.obs["cell_type"] = m['celltype'].to_numpy()

    mito_gene = s.var_names.str.startswith("MT-")
  
    # Zheng specific processing
    s.obs['count'] = np.sum(s.X, axis=1)
    s.obs['pct_mt'] = (s.X @ (1*mito_gene))/s.obs['count']
    
    pct_mt = s.obs["pct_mt"].to_numpy()
    s = s[pct_mt < 0.05,:]
    

    cell_counts = s.obs['count'].to_numpy()
    s = s[(cell_counts > 1000) & (cell_counts < 3000),:]
    gene_counts = np.sum(s.X, axis=0)
    s = s[:,gene_counts > 50]
    
    s = util.counts2ann(s.X, 
                        genes=s.var_names, 
                        barcodes=s.obs_names, 
                        min_gene_count=50, 
                        min_cell_count=500,
                        obs=s.obs)

    print("Number of cells: ", s.n_obs)
    print("Number of genes: ", s.n_vars)
    print("Writing dataset to: ", out_f)
    s.write(out_f, compression="gzip")
    return s
    
      
     
     