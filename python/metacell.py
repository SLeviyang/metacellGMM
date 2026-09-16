from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite


def _copy_anndata_with_integer_cell_ids(s):
    adata = s.copy()
    n = adata.n_obs
    adata.obs_names = [f"cell{i}" for i in range(n)]

    if adata.var_names is None or len(adata.var_names) != adata.n_vars:
        adata.var_names = [f"gene_{j}" for j in range(adata.n_vars)]
    else:
        adata.var_names = [
            str(x) if str(x) else f"gene_{j}"
            for j, x in enumerate(adata.var_names)
        ]

    adata.var_names_make_unique()

    # the metacells package requires a CSR float32 matrix; CSC or other
    # dtypes trigger obscure failures deep inside its internals.
    if sparse.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    return adata


def _ensure_pca(adata, n_pcs: int = 50, pca_key: str = "X_pca"):
    if pca_key in adata.obsm:
        return

    if "highly_variable" not in adata.var:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(3000, adata.n_vars),
            flavor="seurat_v3",
        )

    if sparse.issparse(adata.X):
        nonzero = adata.X.nnz
    else:
        nonzero = np.count_nonzero(adata.X)

    if nonzero == 0:
        raise ValueError("The AnnData matrix appears to contain all zeros.")

    adata_for_pca = adata.copy()
    sc.pp.normalize_total(adata_for_pca, target_sum=1e4)
    sc.pp.log1p(adata_for_pca)

    if (
        "highly_variable" in adata_for_pca.var
        and np.any(adata_for_pca.var["highly_variable"])
    ):
        adata_for_pca = adata_for_pca[
            :, adata_for_pca.var["highly_variable"]
        ].copy()

    sc.pp.scale(adata_for_pca, max_value=10)
    sc.tl.pca(
        adata_for_pca,
        n_comps=min(n_pcs, adata_for_pca.n_obs - 1, adata_for_pca.n_vars - 1),
    )

    adata.obsm[pca_key] = adata_for_pca.obsm["X_pca"]


def _write_matrix_for_r(adata, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    X = X.tocsc()

    matrix_path = out_dir / "counts.mtx"
    genes_path = out_dir / "genes.tsv"
    cells_path = out_dir / "cells.tsv"

    # Python AnnData stores cells x genes.
    # Seurat expects genes x cells.
    mmwrite(matrix_path, X.T)

    pd.Series(adata.var_names).to_csv(
        genes_path,
        sep="\t",
        index=False,
        header=False,
    )
    pd.Series(adata.obs_names).to_csv(
        cells_path,
        sep="\t",
        index=False,
        header=False,
    )

    return matrix_path, genes_path, cells_path


def _write_partition_csv(adata, labels: Iterable, out_path: Path) -> pd.DataFrame:
    labels = pd.Series(labels).astype(str).to_numpy()
    n = len(labels)

    df = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str).to_numpy(),
            "cell_index": np.arange(n, dtype=int),
            "metacell": ["mc" + label for label in labels],
        }
    )

    df.to_csv(out_path, index=False)
    return df


def _run_mcrigor_detect(
    adata,
    labels,
    *,
    r_script_path: str | Path = "mcRigor.R",
    temp_dir: Optional[str | Path] = None,
    alpha: float = 0.05,
    Nrep: int = 1,
    rscript_bin: str = "Rscript",
) -> list[list[int]]:
    parent_tmp = Path(temp_dir) if temp_dir is not None else None

    with tempfile.TemporaryDirectory(dir=parent_tmp) as td:
        td = Path(td)

        matrix_path, genes_path, cells_path = _write_matrix_for_r(adata, td)

        partition_path = td / "partition.csv"
        output_path = td / "trustworthy_metacells.csv"

        partition_df = _write_partition_csv(adata, labels, partition_path)

        r_script_path = Path(r_script_path)

        cmd = [
            rscript_bin,
            str(r_script_path),
            str(matrix_path),
            str(genes_path),
            str(cells_path),
            str(partition_path),
            str(output_path),
            str(alpha),
            str(Nrep),
        ]

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "mcRigor R helper failed.\n\n"
                f"Command:\n{' '.join(cmd)}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )

        trustworthy = pd.read_csv(output_path)

        if "metacell" not in trustworthy.columns:
            raise RuntimeError(
                "R helper output did not contain a 'metacell' column. "
                f"Available columns: {list(trustworthy.columns)}"
            )

        good = set(trustworthy["metacell"].astype(str))
        filtered = partition_df[partition_df["metacell"].astype(str).isin(good)]

        return trustworthy
        result = []
        for _, group in filtered.groupby("metacell", sort=True):
            result.append(group["cell_index"].astype(int).tolist())

        return result


def _get_n_metacells(n_cells: int, gamma: int) -> int:
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    return max(1, int(round(n_cells / gamma)))


def _extract_assignment_column(obs: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in obs.columns:
            return obs[col]

    lower_map = {c.lower(): c for c in obs.columns}

    for col in candidates:
        if col.lower() in lower_map:
            return obs[lower_map[col.lower()]]

    raise RuntimeError(
        "Could not find a metacell assignment column. "
        f"Tried {candidates}. Available obs columns: {list(obs.columns)}"
    )


def _run_seacells(
    adata,
    *,
    gamma: int = 50,
    n_pcs: int = 50,
    pca_key: str = "X_pca",
    n_waypoint_eigs: int = 10,
    convergence_epsilon: float = 1e-5,
    min_iter: int = 10,
    max_iter: int = 100,
    random_seed: int = 0,
):
    try:
        import SEACells
    except ImportError as e:
        raise ImportError(
            "Could not import SEACells. Install it with:\n"
            "    pip install SEACells"
        ) from e

    _ensure_pca(adata, n_pcs=n_pcs, pca_key=pca_key)

    n_seacells = _get_n_metacells(adata.n_obs, gamma)

    np.random.seed(random_seed)

    model = SEACells.core.SEACells(
        adata,
        build_kernel_on=pca_key,
        n_SEACells=n_seacells,
        n_waypoint_eigs=n_waypoint_eigs,
        convergence_epsilon=convergence_epsilon,
    )

    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit(min_iter=min_iter, max_iter=max_iter)

    labels = _extract_assignment_column(
        adata.obs,
        candidates=[
            "SEACell",
            "SEACells",
            "seacell",
            "seacells",
            "metacell",
        ],
    )

    return labels.astype(str).to_numpy()


def _run_metacell_python(
    adata,
    *,
    gamma: int = 50,
    random_seed: int = 0,
):
    try:
        import metacells as mc
    except ImportError as e:
        raise ImportError(
            "Could not import the Python metacells package. Install it with:\n"
            "    pip install metacells\n\n"
            "If you want the older R MetaCell package instead, you would need "
            "a separate R-side MetaCell runner."
        ) from e

    np.random.seed(random_seed)

    if "lateral_gene" not in adata.var:
        adata.var["lateral_gene"] = False

    # The Tanay-lab Python metacells API has changed across versions.
    # Try common high-level pipelines.
    if hasattr(mc, "pl") and hasattr(mc.pl, "divide_and_conquer_pipeline"):
        mc.pl.divide_and_conquer_pipeline(
            adata,
            what="__x__",
            random_seed=random_seed,
            target_metacell_size=gamma,
        )
    elif hasattr(mc, "pl") and hasattr(mc.pl, "compute_metacells"):
        mc.pl.compute_metacells(
            adata,
            what="__x__",
            random_seed=random_seed,
            target_metacell_size=gamma,
        )
    else:
        raise RuntimeError(
            "Imported 'metacells', but did not find a known high-level pipeline. "
            "Expected mc.pl.divide_and_conquer_pipeline or mc.pl.compute_metacells. "
            f"Top-level attributes include: {dir(mc)[:40]}"
        )

    labels = _extract_assignment_column(
        adata.obs,
        candidates=[
            "metacell",
            "metacells",
            "MetaCell",
            "MetaCells",
            "metacell_name",
            "metacell_index",
            "metacell_assignment",
        ],
    )

    return labels.astype(str).to_numpy()


def apply_SEACells(
    s,
    *,
    gamma: int = 50,
    n_pcs: int = 50,
    pca_key: str = "X_pca",
    alpha: float = 0.05,
    Nrep: int = 1,
    r_script_path: str | Path = "mcRigor.R",
    temp_dir: Optional[str | Path] = None,
    rscript_bin: str = "Rscript",
    random_seed: int = 0,
) -> list[list[int]]:
    adata = _copy_anndata_with_integer_cell_ids(s)

    labels = _run_seacells(
        adata,
        gamma=gamma,
        n_pcs=n_pcs,
        pca_key=pca_key,
        random_seed=random_seed,
    )

    return _run_mcrigor_detect(
        adata,
        labels,
        r_script_path=r_script_path,
        temp_dir=temp_dir,
        alpha=alpha,
        Nrep=Nrep,
        rscript_bin=rscript_bin,
    )


def apply_MetaCell(
    s,
    *,
    gamma: int = 50,
    alpha: float = 0.05,
    r_script_path: str | Path = "mcRigor.R",
    temp_dir: Optional[str | Path] = None,
    rscript_bin: str = "Rscript",
    random_seed: int = 0,
) -> list[list[int]]:
    adata = _copy_anndata_with_integer_cell_ids(s)

    labels = _run_metacell_python(
        adata,
        gamma=gamma,
        random_seed=random_seed,
    )

    return labels

    # return _run_mcrigor_detect(
    #     adata,
    #     labels,
    #     r_script_path=r_script_path,
    #     temp_dir=temp_dir,
    #     alpha=alpha,
    #     rscript_bin=rscript_bin,
    # )
