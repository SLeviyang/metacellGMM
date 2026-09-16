#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop(
    paste(
      "Usage: Rscript SuperCell.R <matrix.mtx> <genes.tsv> <cells.tsv>",
      "<membership.csv> [gamma=20] [k.knn=5] [n.var.genes=1000]"
    )
  )
}

matrix_path <- args[[1]]
genes_path <- args[[2]]
cells_path <- args[[3]]
output_path <- args[[4]]
gamma <- if (length(args) >= 5) as.numeric(args[[5]]) else 20
k.knn <- if (length(args) >= 6) as.integer(args[[6]]) else 5
n.var.genes <- if (length(args) >= 7) as.integer(args[[7]]) else 1000

suppressPackageStartupMessages({
  library(Matrix)
  library(SuperCell)
})

GE <- Matrix::readMM(matrix_path)
genes <- readLines(genes_path, warn = FALSE)
cells <- readLines(cells_path, warn = FALSE)

if (nrow(GE) != length(genes)) {
  stop("The number of matrix rows does not match genes.tsv")
}
if (ncol(GE) != length(cells)) {
  stop("The number of matrix columns does not match cells.tsv")
}

rownames(GE) <- make.unique(genes)
colnames(GE) <- make.unique(cells)

# Mirrors the SuperCell vignette section:
# "Simplify single-cell data at the graining level gamma = 20".
SC <- SCimplify(
  GE,
  k.knn = k.knn,
  gamma = gamma,
  n.var.genes = n.var.genes
)

membership <- as.integer(SC$membership)
supercell_size <- as.integer(table(factor(membership, levels = sort(unique(membership)))))
size_by_supercell <- setNames(supercell_size, sort(unique(membership)))

result <- data.frame(
  cell_id = cells,
  cell_index = seq_along(cells) - 1L,
  supercell = membership,
  supercell_size = as.integer(size_by_supercell[as.character(membership)]),
  stringsAsFactors = FALSE
)

write.csv(result, output_path, row.names = FALSE)
