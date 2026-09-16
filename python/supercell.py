import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt




def apply_SuperCell(
    s,
    gamma: int = 20,
    *,
    k_knn: int = 5,
    n_var_genes: int = 1000,
    r_script_path: str = "SuperCell.R",
    rscript_bin: str = "Rscript",
) -> pd.DataFrame:
    """
    Run the R SuperCell package on a Scanpy/AnnData object.

    Returns a data frame with one row per cell and columns:
    cell_id, cell_index, supercell, supercell_size.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    from scipy import sparse
    from scipy.io import mmwrite

    genes = [str(x) for x in s.var_names]
    cells = [str(x) for x in s.obs_names]

    r_script_path = Path(r_script_path)
    if not r_script_path.is_absolute():
        r_script_path = Path(__file__).resolve().parent / r_script_path

    X = s.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        matrix_path = tmp / "expression.mtx"
        genes_path = tmp / "genes.tsv"
        cells_path = tmp / "cells.tsv"
        output_path = tmp / "supercell_membership.csv"

        # AnnData is cells x genes; SuperCell expects genes x cells.
        mmwrite(matrix_path, X.T.tocoo())
        genes_path.write_text("\n".join(genes) + "\n")
        cells_path.write_text("\n".join(cells) + "\n")

        cmd = [
            rscript_bin,
            str(r_script_path),
            str(matrix_path),
            str(genes_path),
            str(cells_path),
            str(output_path),
            str(gamma),
            str(k_knn),
            str(n_var_genes),
        ]

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "SuperCell.R failed.\n\n"
                f"Command:\n{' '.join(cmd)}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )

        return pd.read_csv(output_path)


def decompose_supercells(s) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two summaries of how cell types and supercells overlap.

    The first data frame has one row per cell type and columns sc1..sc5, sorted
    from the most frequent to the fifth most frequent supercell for that cell
    type. The second data frame has one row per supercell and columns ct1..ct3
    with the top cell-type fractions, plus ct1_name..ct3_name with the
    corresponding cell-type names.
    """
    if "cell_type" not in s.obs.columns:
        raise ValueError('decompose_supercells requires s.obs["cell_type"].')
    if "supercell" not in s.obs.columns:
        raise ValueError('decompose_supercells requires s.obs["supercell"].')

    membership = s.obs["supercell"].to_numpy()

    if len(membership) != s.n_obs:
        raise ValueError(
            f"membership has {len(membership)} entries, but s has {s.n_obs} cells."
        )

    df = pd.DataFrame(
        {
            "cell_type": s.obs["cell_type"].astype(str).to_numpy(),
            "supercell": pd.Series(membership).astype(str).to_numpy(),
        }
    )

    rows = []
    for cell_type, group in df.groupby("cell_type", sort=True):
        fractions = group["supercell"].value_counts(normalize=True)
        top_fractions = fractions.iloc[:5].to_list()
        top_fractions += [0.0] * (5 - len(top_fractions))

        rows.append(
            {
                "cell_type": cell_type,
                "size": len(group),
                "sc1": top_fractions[0],
                "sc2": top_fractions[1],
                "sc3": top_fractions[2],
                "sc4": top_fractions[3],
                "sc5": top_fractions[4],
            }
        )

    cell_type_by_top_supercells = (
        pd.DataFrame(rows)
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )

    rows = []
    for supercell, group in df.groupby("supercell", sort=True):
        fractions = group["cell_type"].value_counts(normalize=True)
        top_fractions = fractions.iloc[:3].to_list()
        top_names = fractions.index.astype(str).to_list()[:3]
        top_fractions += [0.0] * (3 - len(top_fractions))
        top_names += [None] * (3 - len(top_names))

        rows.append(
            {
                "supercell": supercell,
                "size": len(group),
                "ct1": top_fractions[0],
                "ct2": top_fractions[1],
                "ct3": top_fractions[2],
                "ct1_name": top_names[0],
                "ct2_name": top_names[1],
                "ct3_name": top_names[2],
            }
        )

    supercell_by_top_cell_types = (
        pd.DataFrame(rows)
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )

    return cell_type_by_top_supercells, supercell_by_top_cell_types

def supercell(s, number_super_cells):
    s.var["lateral_gene"] = False

    gamma = s.n_obs // number_super_cells
    d = apply_SuperCell(s, gamma)
    if len(d) != s.n_obs:
        raise ValueError(
            f"SuperCell returned {len(d)} assignments for {s.n_obs} cells."
        )

    d = d.sort_values("cell_index")
    if not np.array_equal(d["cell_index"].to_numpy(), np.arange(s.n_obs)):
        raise ValueError("SuperCell assignments are missing or duplicating cell indices.")

    s.obs["supercell"] = pd.Categorical(d["supercell"].astype(str).to_numpy())
    return s

def clean_supercells(s, min_super_cell: int = 100, min_cell_type: int = 100):
    if "supercell" not in s.obs.columns:
        raise ValueError('clean_supercells requires s.obs["supercell"].')
    if "cell_type" not in s.obs.columns:
        raise ValueError('clean_supercells requires s.obs["cell_type"].')

    obs = s.obs[["supercell", "cell_type"]].copy()
    if obs["supercell"].isna().any():
        raise ValueError('clean_supercells requires non-missing values in s.obs["supercell"].')
    if obs["cell_type"].isna().any():
        raise ValueError('clean_supercells requires non-missing values in s.obs["cell_type"].')

    obs["supercell"] = obs["supercell"].astype(str)
    obs["cell_type"] = obs["cell_type"].astype(str)

    counts = obs.groupby(["supercell", "cell_type"]).size()
    supercell_sizes = obs.groupby("supercell").size()
    denominators = counts.index.get_level_values("supercell").map(supercell_sizes).astype(float)
    frequencies = counts.astype(float) / denominators

    keep_cell_type_in_supercell = set(
        frequencies[frequencies >= 0.01].index.to_list()
    )
    keep_mask = np.array(
        [
            (supercell, cell_type) in keep_cell_type_in_supercell
            for supercell, cell_type in zip(obs["supercell"], obs["cell_type"])
        ],
        dtype=bool,
    )

    filtered = s[keep_mask, :].copy()
    filtered_supercell_sizes = filtered.obs["supercell"].astype(str).groupby(
        filtered.obs["supercell"].astype(str)
    ).size()
    large_supercells = set(
        filtered_supercell_sizes[filtered_supercell_sizes >= min_super_cell].index
    )

    final_mask = filtered.obs["supercell"].astype(str).isin(large_supercells).to_numpy()
    filtered = filtered[final_mask, :].copy()

    cell_type_sizes = filtered.obs["cell_type"].astype(str).groupby(
        filtered.obs["cell_type"].astype(str)
    ).size()
    large_cell_types = set(
        cell_type_sizes[cell_type_sizes >= min_cell_type].index
    )

    final_mask = filtered.obs["cell_type"].astype(str).isin(large_cell_types).to_numpy()
    return filtered[final_mask, :].copy()

# FORCE_RECOMPUTE = True
# _CACHE_PATH = "metacell_cell_type_table.pkl"
# if FORCE_RECOMPUTE or not os.path.exists(_CACHE_PATH):
#     #s = Zheng.dataset_Zheng()
#     with open(_CACHE_PATH, "wb") as _f:
#         pickle.dump({"s": s}, _f)

# print(f"Loading from cache: {_CACHE_PATH}")
# with open(_CACHE_PATH, "rb") as _f:
#     _cache = pickle.load(_f)
# s = _cache["s"]
# s.var["lateral_gene"] = False

# number_super_cells = 40
# s = supercell(s, number_super_cells)
# s = clean_supercells(s, min_super_cell=100, min_cell_type=100)
# dd1, dd2 = decompose_supercells(s)
# print(dd1)
# print(dd2)
