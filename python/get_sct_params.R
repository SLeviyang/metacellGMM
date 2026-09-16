#!/usr/bin/env Rscript
# Usage: Rscript get_sct_params.R in1.csv out1.csv [in2.csv out2.csv ...]
#
# Each input CSV is a cells x genes count matrix (no header, no row index).
# Each output CSV has two columns (bg0, theta), one row per gene in the same
# order.  Genes not modelled by SCTransform get bg0 = -Inf, theta = Inf.

suppressPackageStartupMessages({
  library(Seurat)
  library(sctransform)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0 || length(args) %% 2 != 0)
  stop("Usage: Rscript get_sct_params.R in1.csv out1.csv [in2.csv out2.csv ...]")

process_one <- function(input_csv, output_csv) {
  X <- as.matrix(read.csv(input_csv, header = FALSE))
  n_genes <- ncol(X)          # input is cells x genes
  X <- t(X)                   # Seurat expects genes x cells

  pbmc <- CreateSeuratObject(counts = X)
  pbmc <- SCTransform(pbmc, verbose = TRUE, min_cells = 5)

  df <- pbmc@assays$SCT@SCTModel.list$model1@feature.attributes

  bg0_out   <- rep(-Inf, n_genes)
  theta_out <- rep( Inf, n_genes)

  for (i in seq_len(nrow(df))) {
    gene_name           <- rownames(df)[i]
    gene_idx            <- as.integer(substr(gene_name, 2L, nchar(gene_name)))
    bg0_out[gene_idx]   <- df[i, "(Intercept)"]
    theta_out[gene_idx] <- df[i, "theta"]
  }

  write.csv(data.frame(mu = exp(bg0_out), theta = theta_out),
            output_csv, row.names = FALSE)
}

pairs <- matrix(args, nrow = 2)          # 2 x n_matrices
for (j in seq_len(ncol(pairs))) {
  cat(sprintf("\n--- Matrix %d of %d ---\n", j, ncol(pairs)))
  process_one(pairs[1, j], pairs[2, j])
}
