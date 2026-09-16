suppressPackageStartupMessages({
    library(Matrix)
    library(Seurat)
    library(data.table)
    library(mcRigor)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 7) {
    stop(
        "Usage: Rscript run_mcrigor_detect.R ",
        "counts.mtx genes.tsv cells.tsv partition.csv output.csv alpha Nrep"
    )
}

matrix_path <- args[[1]]
genes_path <- args[[2]]
cells_path <- args[[3]]
partition_path <- args[[4]]
output_path <- args[[5]]
alpha <- as.numeric(args[[6]])
Nrep <- as.integer(args[[7]])

counts <- Matrix::readMM(matrix_path)
counts <- as(counts, "CsparseMatrix")

genes <- fread(genes_path, header = FALSE)[[1]]
cells <- fread(cells_path, header = FALSE)[[1]]

rownames(counts) <- make.unique(as.character(genes))
colnames(counts) <- as.character(cells)

partition_long <- fread(partition_path)
partition_long$cell_id <- as.character(partition_long$cell_id)
partition_long$metacell <- as.character(partition_long$metacell)

missing_cells <- setdiff(partition_long$cell_id, colnames(counts))
if (length(missing_cells) > 0) {
    stop(
        "Some partition cells are not in the Seurat object. First few missing: ",
        paste(head(missing_cells, 10), collapse = ", ")
    )
}

partition_long <- partition_long[match(colnames(counts), partition_long$cell_id), ]

if (any(is.na(partition_long$metacell))) {
    stop("Some cells have NA metacell assignments after alignment.")
}

cell_membership <- data.frame("50" = partition_long$metacell, check.names = FALSE)
rownames(cell_membership) <- partition_long$cell_id

seu <- CreateSeuratObject(counts = counts, project = "mcRigor_from_python")

detect_res <- mcRigor_DETECT(
    obj_singlecell = seu,
    cell_membership = cell_membership,
    tgamma = "50",
    assay_type = "RNA",
    aggregate_method = "mean",
    Nrep = Nrep,
    feature_use = min(2000, nrow(seu)),
    prePro = TRUE,
    test_cutoff = alpha,
    thre_smooth = TRUE,
    thre_bw = 1 / 6,
    draw = FALSE
)

mc_res <- detect_res$mc_res
names(mc_res) <- colnames(detect_res$obj_metacell)

if (is.null(mc_res)) {
    stop(
        "mcRigor_DETECT returned no mc_res field. Returned names: ",
        paste(names(detect_res), collapse = ", ")
    )
}

extract_trustworthy <- function(mc_res, alpha) {
    df <- as.data.frame(mc_res)

    if (!is.null(rownames(df))) {
        mc_ids <- rownames(df)
    } else {
        stop("mc_res has no rownames, so metacell IDs cannot be recovered.")
    }

    if (ncol(df) == 1 && names(df)[1] == "mc_res") {
        status <- tolower(as.character(df[[1]]))

        is_dubious <- status %in% c(
            "dubious",
            "true",
            "t",
            "1",
            "yes",
            "y",
            "reject",
            "rejected",
            "fail",
            "failed"
        )

        is_trustworthy <- status %in% c(
            "trustworthy",
            "non-dubious",
            "nondubious",
            "not_dubious",
            "false",
            "f",
            "0",
            "no",
            "n",
            "pass",
            "passed"
        )

        if (any(is_trustworthy) || any(is_dubious)) {
            return(mc_ids[!is_dubious])
        }

        stop(
            "Found mc_res column, but statuses were not recognized. Unique values: ",
            paste(unique(status), collapse = ", ")
        )
    }

    cn <- names(df)
    cn_lower <- tolower(cn)

    find_col <- function(candidates) {
        for (cand in candidates) {
            hit <- which(cn_lower == cand)
            if (length(hit) > 0) {
                return(cn[[hit[[1]]]])
            }
        }
        return(NA_character_)
    }

    mc_col <- find_col(c(
        "metacell",
        "mc",
        "mc_id",
        "metacell_id",
        "metacell_name",
        "metacell_index"
    ))

    trust_col <- find_col(c(
        "trustworthy",
        "is_trustworthy",
        "true_metacell"
    ))

    dub_col <- find_col(c(
        "dubious",
        "is_dubious",
        "dubious_metacell",
        "dub_mc",
        "dubious_test",
        "test_res",
        "test_result",
        "reject",
        "rejected"
    ))

    p_col <- find_col(c(
        "pvalue",
        "p_value",
        "p.val",
        "p_val",
        "padj",
        "adj_p_value",
        "qvalue",
        "q_value"
    ))

    if (!is.na(mc_col)) {
        mc_ids <- as.character(df[[mc_col]])
    }

    if (!is.na(trust_col)) {
        vals <- df[[trust_col]]
        if (is.character(vals)) {
            vals <- tolower(vals) %in% c("true", "t", "1", "yes", "y", "trustworthy", "pass", "passed")
        } else {
            vals <- as.logical(vals)
        }
        return(mc_ids[which(vals)])
    }

    if (!is.na(dub_col)) {
        vals <- df[[dub_col]]
        if (is.character(vals)) {
            vals <- tolower(vals) %in% c("true", "t", "1", "yes", "y", "dubious", "reject", "rejected", "fail", "failed")
        } else {
            vals <- as.logical(vals)
        }
        return(mc_ids[which(!vals)])
    }

    if (!is.na(p_col)) {
        vals <- as.numeric(df[[p_col]])
        return(mc_ids[which(vals >= alpha)])
    }

    stop(
        "Could not infer trustworthy metacells from mc_res. Columns are: ",
        paste(cn, collapse = ", "),
        "\nFirst few rows:\n",
        paste(capture.output(print(head(df))), collapse = "\n")
    )
}

trustworthy <- extract_trustworthy(mc_res, alpha)

out <- data.frame(metacell = unique(as.character(trustworthy)))
write.csv(out, output_path, row.names = FALSE)