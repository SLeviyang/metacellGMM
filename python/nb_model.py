import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import supercell
import lognorm_workflow_analytic as lwa

########################################################
# methods to construct nb object

def shuffle_scanpy(s):
    new_order = np.arange(s.shape[0])
    for ct in s.obs["cell_type"].unique():
        idx = np.where(s.obs["cell_type"].to_numpy() == ct)[0]
        new_order[idx] = idx[np.random.permutation(len(idx))]
    return s[new_order].copy()


def apply_SCT(X_list):
    """Run SCTransform on a list of cells x genes count matrices.

    Calls get_sct_params.R once via Rscript, passing all matrices together
    so that R libraries are loaded only once.

    Parameters
    ----------
    X_list : list of array-like, each shape (n_cells, n_genes)

    Returns
    -------
    list of dict, one per input matrix, each with keys:
        'bg0'   : np.ndarray of length n_genes  (-inf for unmodelled genes)
        'theta' : np.ndarray of length n_genes  ( inf for Poisson/unmodelled)
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "get_sct_params.R")

    tmp_inputs  = []
    tmp_outputs = []
    try:
        for X in X_list:
            fi = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            fo = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            fi.close(); fo.close()
            pd.DataFrame(X).to_csv(fi.name, header=False, index=False)
            tmp_inputs.append(fi.name)
            tmp_outputs.append(fo.name)

        # Build alternating in/out argument list for the R script
        r_args = []
        for inp, out in zip(tmp_inputs, tmp_outputs):
            r_args += [inp, out]

        subprocess.run(["Rscript", script] + r_args, check=True)

        results = []
        for out in tmp_outputs:
            df = pd.read_csv(out)
            results.append({
                "mu":    df["mu"].to_numpy(dtype=float),
                "theta": df["theta"].to_numpy(dtype=float),
            })
    finally:
        for path in tmp_inputs + tmp_outputs:
            if os.path.exists(path):
                os.unlink(path)

    return results

def _supercell_labels_by_partition(s, partition, num_super_cells, min_super_cell):
    """SuperCell run separately within each partition, so every supercell holds
    cells of a single partition value.

    num_super_cells is split across partitions in proportion to their cell
    counts, which keeps supercell.supercell's gamma = n_obs // n_p -- and so
    the average supercell size -- roughly constant across partitions. Splitting
    it evenly instead would make supercells from a small partition far smaller
    than those from a large one. n_p is floored at 1 and capped at n_p_cells//2
    so gamma stays >= 2 and SuperCell is never asked for one supercell per cell.

    Partitions smaller than min_super_cell are skipped rather than sent to
    SuperCell: they cannot yield a supercell that survives the size filter
    anyway, and SuperCell's k_knn=5 graph is not meaningful on a handful of
    cells. Their cells get a None label and are dropped by the caller.

    A partition that works out to a single supercell is labelled directly
    rather than sent to SuperCell -- there is nothing to partition, and
    SCimplify on a small cell count with gamma = n_obs is exactly the
    degenerate case it handles worst.

    Genes with no variance INSIDE a partition are dropped before the call.
    The HVGs were chosen across the whole dataset, so a gene that varies
    globally can be flat -- usually all-zero -- within one cell type, and
    SCimplify hands the matrix to prcomp, which fails on a constant column
    ("cannot rescale a constant/zero column to unit variance"). Which genes
    are constant differs by partition, so this has to be redone per call.
    Only the matrix SuperCell clusters on is trimmed; it returns one label per
    CELL, so `s` keeps every gene and nothing downstream sees the difference.

    Labels are "<partition>_<n>" -- SuperCell numbers its supercells from
    scratch within each call, so the prefix is what keeps them distinct across
    partitions.

    Returns an object array of labels aligned with s's rows (None where the
    partition was skipped).
    """
    partition = np.asarray(partition)
    labels = np.full(s.n_obs, None, dtype=object)
    n_total = s.n_obs

    for value in pd.unique(partition):
        idx = np.where(partition == value)[0]
        if len(idx) < min_super_cell:
            print(f"  SKIPPING partition '{value}': {len(idx)} cells "
                  f"< min_super_cell={min_super_cell}")
            continue

        n_p = max(1, int(round(num_super_cells * len(idx) / n_total)))
        n_p = min(n_p, len(idx) // 2)

        if n_p <= 1:
            print(f"  partition '{value}': {len(idx)} cells -> 1 supercell "
                  f"(assigned directly, SuperCell not called)")
            labels[idx] = f"{value}_0"
            continue

        s_p = s[idx].copy()
        X_p = s_p.X.toarray() if hasattr(s_p.X, "toarray") else np.asarray(s_p.X)
        varying = X_p.std(axis=0) > 0
        n_dropped = int((~varying).sum())
        print(f"  partition '{value}': {len(idx)} cells -> {n_p} supercells"
              + (f" ({n_dropped} of {len(varying)} genes constant here, dropped "
                 f"for the SuperCell call)" if n_dropped else ""))
        if varying.sum() < 2:
            raise ValueError(
                f"partition '{value}' has {int(varying.sum())} genes with any "
                f"variance across its {len(idx)} cells; SuperCell cannot cluster it")

        # supercell.supercell writes to s.var and s.obs, so hand it a copy --
        # gene-trimmed, since the labels come back per cell either way
        s_p = supercell.supercell(s_p[:, varying].copy(), number_super_cells=n_p)
        labels[idx] = [f"{value}_{lb}" for lb in s_p.obs["supercell"].astype(str)]

    return labels


def apply_metacell_decomposition(s, num_super_cells, partition=None,
                                 min_super_cell=100):
    """Group cells into metacells with SuperCell, then drop the undersized ones.

    partition, if given, is a categorical vector with one entry per row of `s`
    (as passed in, before any filtering here). SuperCell is then run separately
    within each partition value, so every metacell contains cells of a single
    partition -- see _supercell_labels_by_partition. With partition=None a
    single global SuperCell call is made, as before.

    min_super_cell is both the size below which a finished supercell is
    discarded and the size below which a partition is skipped outright.

    This no longer calls supercell.clean_supercells, which also applied two
    filters that are now gone: cells whose (supercell, cell_type) pair made up
    under 1% of its supercell were dropped, and cell types with fewer than 100
    cells dataset-wide were dropped entirely. Only the supercell size filter
    remains. Rows are also no longer sorted by metacell and shuffled; nothing
    downstream depends on row order (every consumer groups by label value, and
    the SVD is order-equivariant), and the shuffle drew from an unseeded global
    RNG, so removing it makes the build reproducible.
    """
    print("Running metacell decomposition")
    print("stablizing library size and gene count")

    L_true = np.sum(s.X.toarray(), axis=1)
    n_feature = np.sum(s.X.toarray() > 0, axis=1)
    print("number of cells before preprocessing: ", s.shape[0])

    if partition is not None:
        partition = np.asarray(partition)
        if len(partition) != s.n_obs:
            raise ValueError(f"partition has {len(partition)} entries but s has "
                             f"{s.n_obs} rows")

    # the cell mask has to carry the partition with it, or every later
    # assignment would be against the wrong cells
    keep = (L_true > 100) & (n_feature > 40)
    s = s[keep, :]
    if partition is not None:
        partition = partition[keep]
    print("number of cells after cell filtering: ", s.shape[0])
    gene_counts = s.X.sum(axis=0)
    s = s[:, gene_counts > 100]
    print("number of cells after gene filtering: ", s.shape[0])
    print("min library size:", np.min(np.sum(s.X.toarray(), axis=1)))
    print("min gene count:", np.min(np.sum(s.X.toarray(), axis=0)))

    print("Running supercell")
    if partition is None:
        s = supercell.supercell(s, number_super_cells=num_super_cells)
    else:
        labels = _supercell_labels_by_partition(s, partition, num_super_cells,
                                                min_super_cell)
        assigned = np.array([lb is not None for lb in labels])
        print(f"cells in skipped partitions: {int((~assigned).sum())}")
        s = s[assigned, :].copy()
        s.obs["partition"] = pd.Categorical(partition[assigned].astype(str))
        s.obs["supercell"] = pd.Categorical(labels[assigned].astype(str))

    s.obs["metacell"] = s.obs["supercell"]

    # drop undersized supercells. One pass is enough: removing a supercell
    # cannot change any other supercell's size, unlike clean_supercells whose
    # global cell-type filter fed back into its own size filter.
    sizes = s.obs["metacell"].astype(str).value_counts()
    big = set(sizes[sizes >= min_super_cell].index)
    keep = s.obs["metacell"].astype(str).isin(big).to_numpy()
    print(f"dropping {int((~keep).sum())} cells in supercells smaller than "
          f"{min_super_cell} ({len(sizes) - len(big)} of {len(sizes)} supercells)")
    s = s[keep, :].copy()

    if s.n_obs == 0:
        raise ValueError(
            f"every supercell was smaller than min_super_cell={min_super_cell}, "
            f"leaving no cells. Lower min_super_cell, or raise num_super_cells "
            f"(={num_super_cells}) so fewer, larger supercells are built.")

    # and the genes those removals left too sparse; this cannot change the rows,
    # so it needs no second pass either
    gene_counts = s.X.sum(axis=0)
    if not np.all(gene_counts > 50):
        s = s[:, np.asarray(gene_counts).ravel() > 50].copy()

    print("final number of cells: ", s.shape[0])
    print("min library size:", np.min(np.sum(s.X.toarray(), axis=1)))
    print("min feature count:", np.min(np.sum(s.X.toarray() > 0, axis=1)))
    print("min gene count:", np.min(np.sum(s.X.toarray(), axis=0)))

    print("number of super cells: ", s.obs["metacell"].nunique())

    return s


def compute_metacell_NB(s, poisson=False):
    """Fit the per-metacell NB model of s.

    "L" is a per-metacell array, aligned with "celltypes" like mu_matrix,
    theta_matrix and pop_alphas: each metacell's mean library size (mean
    row sum of its cells). It is the library size every consumer uses --
    simulate_X draws each metacell at its own L, and the analytic model
    (lognorm_workflow_analytic) evaluates the log moments at it. There is
    no scalar form.
    """
    celltypes = s.obs["metacell"].unique()
    X = s.X.toarray()
    pop_alphas = np.zeros(len(celltypes))
    for i,t in enumerate(celltypes):
        pop_alphas[i] = np.mean(s.obs["metacell"] == t)

    X_list = []
    L = np.zeros(len(celltypes))
    for i, ct in enumerate(celltypes):
        idx = s.obs["metacell"] == ct
        X_ct = X[idx, :]
        X_list.append(X_ct)
        L[i] = X_ct.sum(axis=1).mean()

    n_genes = X.shape[1]
    mu_celltypes    = np.zeros((len(celltypes), n_genes))
    theta_celltypes = np.zeros((len(celltypes), n_genes))

    if poisson:
        for i, X_ct in enumerate(X_list):
            X_norm = X_ct / X_ct.sum(axis=1, keepdims=True)
            mu_celltypes[i, :] = X_norm.mean(axis=0)
            theta_celltypes[i, :] = np.inf
    else:
        results = apply_SCT(X_list)
        for i, r in enumerate(results):
            mu_celltypes[i, :]    = r["mu"]
            theta_celltypes[i, :] = r["theta"]

    return {"celltypes": celltypes, "mu_matrix": mu_celltypes, "theta_matrix": theta_celltypes,
    "pop_alphas": pop_alphas, "L": L}

########################################################
# methods for nb object

def simulate_X(nb, mixture_labels):
    """Draw a count matrix from the fitted NB model, one row per entry of
    mixture_labels. Every cell of metacell i is drawn at that metacell's own
    library size nb["L"][i] (see compute_metacell_NB), with per-gene means
    nb["L"][i] * mu_matrix[i] and per-gene dispersion theta_matrix[i].
    """
    celltypes, mu_celltypes, theta_celltypes = nb["celltypes"], nb["mu_matrix"], nb["theta_matrix"]
    L = np.asarray(nb["L"], dtype=float).ravel()
    if len(L) != len(celltypes):
        raise ValueError(f"nb['L'] must have one entry per metacell: got {len(L)} for "
                         f"{len(celltypes)} celltypes")

    mixture_labels = np.asarray(mixture_labels)
    n = len(mixture_labels)
    p = mu_celltypes.shape[1]
    X = np.zeros((n, p))

    unknown = set(np.unique(mixture_labels)) - set(celltypes)
    if unknown:
        raise ValueError(f"mixture_labels contains labels not in nb['celltypes']: {unknown}")

    print("Creating NB samples for number of celltypes: ", len(celltypes))
    for i, ct in enumerate(celltypes):
        idx = np.where(mixture_labels == ct)[0]
        if len(idx) == 0:
            continue

        mu_g = L[i] * mu_celltypes[i, :]   # per-gene means, same for every cell of this metacell
        theta_i = theta_celltypes[i, :]    # shape (p,)
        nb_mask = np.isfinite(theta_i)     # genes to sample as NB

        # NB genes: variance = mu + mu^2/theta
        if nb_mask.any():
            mu_nb    = mu_g[nb_mask]
            theta_nb = theta_i[nb_mask]
            X[np.ix_(idx, nb_mask)] = np.random.negative_binomial(
                n=theta_nb,
                p=theta_nb / (theta_nb + mu_nb),
                size=(len(idx), nb_mask.sum()),
            )

        # Poisson genes: theta = inf
        if (~nb_mask).any():
            X[np.ix_(idx, ~nb_mask)] = np.random.poisson(
                lam=mu_g[~nb_mask], size=(len(idx), (~nb_mask).sum()),
            )

    return X