import os

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import datasets
import normality as nrm
import nb_model


DEFAULT_Z_CUTOFF = 3.0
DEFAULT_MAX_ROUNDS = 20

# the end states a metacell can be in once the iteration stops
METACELL_STATES = ["refined", "refractory", "unfinished", "too_small"]
STATE_COLORS = {"refined": "mediumseagreen", "refractory": "indianred",
                "unfinished": "darkorange", "too_small": "grey"}
DEFAULT_MIN_METACELL_SIZE = 40

# a metacell whose worst direction departs from a Gaussian by less than this
# much -- as a maximum CDF gap, i.e. the raw KS statistic -- is accepted as
# refined whatever its z-score. The z-score is a SIGNIFICANCE measure: since
# mu(n) shrinks as n^-0.5, a fixed departure yields z growing as sqrt(n), so
# a large metacell gets flagged for a departure a small one would not. This
# floor is the matching EFFECT-SIZE requirement. Empirically every accepted
# metacell under 500 cells has a gap of at least 4.2%, so a 5% floor only
# ever prevents splits, never causes them.
DEFAULT_MAX_GAP = 0.05

# refined HVG datasets live in their own flat folder
REFINED_DIR = "../analysis_HVG_refined/"


def load_HVG_pca(dataset, n_genes):
    """(s, metacell_labels, X, k) for one HVG dataset.

    X is obsm["X_pca"], the leading k principal-component scores (U * sv)
    computed once by datasets.create_HVG_dataset, so nothing is recomputed
    here; k is X.shape[1], set there by the elbow of the singular values
    (capped at datasets.MAX_K).
    """
    s = datasets.load_HVG_dataset(dataset, n_genes)
    if "X_pca" not in s.obsm:
        raise ValueError(
            f"{dataset}/{n_genes} has no obsm['X_pca'] -- it predates the PCA being "
            "stored on the HVG object; rebuild it with datasets.create_HVG_dataset")

    X = np.asarray(s.obsm["X_pca"])
    return s, s.obs["metacell"].to_numpy(), X, X.shape[1]


def compute_z_scores(X, metacells, null, min_metacell_size=None):
    """(labels, z, gaps) for every metacell: the labels in np.unique order,
    their normality z-scores, and the raw statistic behind those scores --
    the maximum CDF gap from a Gaussian over the metacell's eigenvector
    directions -- as parallel arrays.

    The gap is the EFFECT SIZE and z its significance; both are needed,
    since z alone flags a vanishing departure once n is large (see
    DEFAULT_MAX_GAP).

    The labels are returned alongside the scores so downstream code can
    confirm which metacell each score belongs to rather than relying on an
    assumed ordering.

    X is the (n_cells, k) PC-score matrix and metacells the per-cell label
    array, so X[metacells == label] are that metacell's scores. null must
    have been fitted for the same k (normality_z checks this).

    A metacell below min_metacell_size is not scored and gets np.nan: it is
    terminal (too_small) and will not be analysed further, and once n
    approaches k its covariance is rank deficient anyway, so the score would
    be meaningless. Splits can produce such metacells, since a split is
    accepted on the strength of ONE viable child.
    """
    metacells = np.asarray(metacells)
    labels = np.unique(metacells)

    z, gaps = [], []
    for label in labels:
        Xi = X[metacells == label]
        if min_metacell_size is not None and len(Xi) < min_metacell_size:
            z.append(np.nan)
            gaps.append(np.nan)
            continue
        # the raw statistic and its z are one computation, not two: z is just
        # the gap standardized by the null at this n
        gap = nrm.metacell_normality(Xi)
        gaps.append(gap)
        z.append((gap - null["mu"](len(Xi))) / null["sigma"](len(Xi)))

    return labels, np.array(z, dtype=float), np.array(gaps, dtype=float)


def split_metacell(X_metacell, zscore, null, min_metacell_size=None, seed=0):
    """Split one metacell in two, along the direction that carries its
    non-normality, and score the two children.

    X_metacell is that metacell's PC scores and zscore its normality
    z-score, carried through so the caller can compare it against the
    children's. null scores the children and must be for the same k.

    A child below min_metacell_size is not scored -- its z is nan. Scoring
    it would be meaningless once n approaches k, and nan keeps it out of the
    accept test in call_split_metacell without needing an explicit check.

    The direction is the metacell's own covariance eigenvector whose 1-D
    projection is least normal. No null is needed to pick it: all k
    candidate projections come from the SAME metacell and so share the same
    n, making mu(n) and sigma(n) identical positive constants across them --
    z is then a monotone increasing function of the raw KS statistic, so
    argmax z is argmax raw. This is deliberately NOT the largest-eigenvalue
    eigenvector; empirically that one carried the non-normality only about
    40% of the time.

    Cells are assigned by SAMPLING each one's GMM posterior rather than by
    taking its argmax, so the boundary is not a hard threshold and cells in
    the overlap are divided in proportion to their membership probability.

    Returns a dict with:
      idx         -- [rows of child 0, rows of child 1], as index arrays into
                     X_metacell; None if the split degenerated
      child_n     -- [n of child 0, n of child 1]; None if degenerate
      child_z     -- [z of child 0, z of child 1]; nan for a child below
                     min_metacell_size, which is not scored
      parent_n    -- cells in the metacell being split
      parent_z    -- zscore, unchanged
      direction   -- the unit eigenvector split along
      eig_rank    -- its rank by eigenvalue (0 = largest)
      projection  -- X_metacell @ direction
      per_axis    -- the per-eigenvector KS statistics
    """
    n = X_metacell.shape[0]
    result = {"idx": None, "child_n": None, "child_z": None,
              "parent_n": int(n), "parent_z": zscore,
              "direction": None, "eig_rank": None, "projection": None, "per_axis": None}

    eigvecs = nrm.metacell_eigenvectors(X_metacell)
    per_axis = nrm.metacell_normality(X_metacell, return_per_axis=True)
    j = int(np.argmax(per_axis))

    direction = eigvecs[:, j]
    projection = X_metacell @ direction

    gmm = GaussianMixture(n_components=2, random_state=seed, n_init=4)
    gmm.fit(projection.reshape(-1, 1))
    posterior = gmm.predict_proba(projection.reshape(-1, 1))

    rng = np.random.default_rng(seed)
    assign = (rng.random(n)[:, None] < np.cumsum(posterior, axis=1)).argmax(axis=1)

    result.update({"direction": direction, "eig_rank": j, "projection": projection,
                   "per_axis": per_axis})
    if len(np.unique(assign)) < 2:
        return result

    idx = [np.flatnonzero(assign == c) for c in (0, 1)]
    child_n = [int(len(i)) for i in idx]
    result["idx"] = idx
    result["child_n"] = child_n

    # only children that reach min_metacell_size are scored; the others are
    # nan, which keeps them out of the accept test (nan comparisons are
    # False) without needing an explicit check
    result["child_z"] = [
        nrm.normality_z(X_metacell[i], null)
        if (min_metacell_size is None or len(i) >= min_metacell_size) else np.nan
        for i in idx]
    return result


def call_split_metacell(split, parent_metacell):
    """Decide whether to accept a proposed split, and return the resulting
    per-cell labels.

    split is the dict from split_metacell and parent_metacell the label of
    the metacell it came from. The returned vector always has one entry per
    cell of the parent, in the parent's own row order: all equal to
    parent_metacell if the split is rejected, or "<parent>_0" / "<parent>_1"
    at the corresponding indices if it is accepted.

    A split is accepted when it did not degenerate and at least ONE child
    both reaches min_metacell_size and scores below the parent. There is no
    requirement on the other child: a split that carves off one clean,
    adequately sized piece is worth taking even if the remainder is small or
    still poor. Children below min_metacell_size have z = nan, and since nan
    comparisons are False they can never satisfy the test on their own.

    Cells are never dropped -- both children are kept. An undersized one
    simply becomes a too_small metacell, terminal and not analysed further.
    """
    n = split["parent_n"]
    reject = np.full(n, parent_metacell, dtype=object)

    if split["idx"] is None:
        return reject

    child_z = np.asarray(split["child_z"], dtype=float)
    if not np.any(child_z < split["parent_z"]):
        return reject

    labels = np.empty(n, dtype=object)
    for c, rows in enumerate(split["idx"]):
        labels[rows] = f"{parent_metacell}_{c}"
    return labels


def classify_metacells(metacell_labels, z_scores, z_cutoff=DEFAULT_Z_CUTOFF,
                       min_metacell_size=None, terminal=None,
                       gaps=None, max_gap=DEFAULT_MAX_GAP):
    """The end state of each metacell, as an array in np.unique order.

    The four states are mutually exclusive and tested in this order:

      too_small   -- below min_metacell_size, so it cannot yield two viable
                     children and is done regardless of its score
      refined     -- passes on EITHER criterion: its maximum CDF gap is
                     below max_gap (the departure is too small to be worth
                     acting on, whatever its significance -- see
                     DEFAULT_MAX_GAP), or z <= z_cutoff
      refractory  -- fails both, but a split was proposed and rejected
                     (it appears in `terminal`)
      unfinished  -- fails both and was never resolved, i.e. the iteration
                     stopped at max_rounds before reaching it
    """
    labels, counts = np.unique(metacell_labels, return_counts=True)
    z = np.asarray(z_scores, dtype=float)
    g = np.full(len(labels), np.nan) if gaps is None else np.asarray(gaps, dtype=float)
    terminal = terminal or {}

    states = []
    for lab, n, zi, gi in zip(labels, counts, z, g):
        small_gap = max_gap is not None and np.isfinite(gi) and gi < max_gap
        if min_metacell_size is not None and n < min_metacell_size:
            states.append("too_small")
        elif small_gap or zi <= z_cutoff:
            states.append("refined")
        elif lab in terminal:
            states.append("refractory")
        else:
            states.append("unfinished")
    return np.array(states, dtype=object)


def visualize_refinement(metacell_labels, z_scores, z_cutoff=DEFAULT_Z_CUTOFF,
                         min_metacell_size=None, terminal=None,
                         gaps=None, max_gap=DEFAULT_MAX_GAP,
                         title=None, axes=None, show=True):
    """Two panels: metacell size against normality z-score, and the fraction
    of CELLS sitting in each end state.

    metacell_labels is the per-cell label vector; z_scores is the
    per-metacell vector in np.unique(metacell_labels) order, i.e. exactly
    what compute_z_scores returns.

    Both panels share the state coloring (see classify_metacells), so the
    scatter shows where each state lives in size/score space while the bar
    shows how much of the data it accounts for -- metacell counts and cell
    fractions differ sharply, since the states are strongly size-dependent.

    Returns the matplotlib figure.
    """
    labels, counts = np.unique(metacell_labels, return_counts=True)
    z = np.asarray(z_scores, dtype=float)
    if len(z) != len(labels):
        raise ValueError(f"{len(z)} z-scores for {len(labels)} metacells; z_scores must be "
                         "in np.unique(metacell_labels) order (see compute_z_scores)")

    states = classify_metacells(metacell_labels, z, z_cutoff=z_cutoff, gaps=gaps,
                                max_gap=max_gap,
                                min_metacell_size=min_metacell_size, terminal=terminal)
    total_cells = counts.sum()

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                                 gridspec_kw={"width_ratios": [2, 1]})
    else:
        fig = np.atleast_1d(axes)[0].figure
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for state in METACELL_STATES:
        sel = states == state
        if sel.any():
            ax.scatter(counts[sel], z[sel], s=32, alpha=0.75, color=STATE_COLORS[state],
                       edgecolors="none", label=f"{state} ({sel.sum()})")
    ax.axhline(0.0, color="black", ls=":", lw=1.2, label="null (z=0)")
    ax.axhline(z_cutoff, color="purple", ls="-.", lw=1.2, label=f"cutoff ({z_cutoff:g})")
    ax.axhline(np.nanmedian(z), color="black", ls="--", lw=1.4,
               label=f"median ({np.nanmedian(z):.2f})")
    if min_metacell_size is not None:
        ax.axvline(min_metacell_size, color="grey", ls=":", lw=1.2,
                   label=f"min size ({min_metacell_size})")
    ax.set_xscale("log")
    ax.set_xlabel("n_cells (metacell)")
    ax.set_ylabel("normality z-score")
    ax.set_title(f"{len(labels)} metacells")
    ax.legend(fontsize=8)

    ax = axes[1]
    fracs = [counts[states == state].sum() / total_cells for state in METACELL_STATES]
    bars = ax.bar(range(len(METACELL_STATES)), fracs,
                  color=[STATE_COLORS[s] for s in METACELL_STATES], alpha=0.85)
    for i, (state, bar) in enumerate(zip(METACELL_STATES, bars)):
        sel = states == state
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{fracs[i]:.1%}\n{int(counts[sel].sum()):,} cells\n{sel.sum()} mc",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(METACELL_STATES)))
    ax.set_xticklabels(METACELL_STATES, rotation=20, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("fraction of cells")
    ax.set_title(f"{total_cells:,} cells by end state")

    fig.suptitle(title if title is not None else "metacell refinement", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if show:
        plt.show()
    return fig


def refined_filename(dataset, n_genes):
    """Path of the refined h5ad, creating the folder if it does not exist."""
    os.makedirs(REFINED_DIR, exist_ok=True)
    return os.path.join(REFINED_DIR, f"{dataset}_HVG_genes{n_genes}_refined.h5ad")


def refined_info_filename(dataset):
    """Path of a dataset's HVG_refined_info.csv. Prefixed by dataset because
    all of them share one folder.
    """
    os.makedirs(REFINED_DIR, exist_ok=True)
    return os.path.join(REFINED_DIR, f"{dataset}_HVG_refined_info.csv")


def load_HVG_refined_dataset(dataset, n_genes):
    """The refined HVG AnnData, as written by create_HVG_refined_dataset.

    Mirrors datasets.load_HVG_dataset, which reads only the unrefined file;
    the refined path lives here rather than as a flag over there.
    """
    out_f = refined_filename(dataset, n_genes)
    if os.path.isfile(out_f):
        return ad.read_h5ad(out_f)
    raise ValueError(f"Refined dataset {dataset} with {n_genes} genes not found")


def create_HVG_refined_dataset(dataset, n_genes, z_cutoff=DEFAULT_Z_CUTOFF,
                               min_metacell_size=DEFAULT_MIN_METACELL_SIZE,
                               max_gap=DEFAULT_MAX_GAP, max_rounds=DEFAULT_MAX_ROUNDS,
                               seed=0, poisson=False, overwrite=False, verbose=True):
    """Refine the metacell partition of one HVG dataset by repeatedly
    splitting metacells whose PC scores are too far from Gaussian.

    Writes the result to "<dataset>_HVG_genes<n_genes>_refined.h5ad" and
    returns it. If that file already exists and overwrite is False it is
    loaded and returned untouched, as datasets.create_HVG_dataset does for
    the unrefined file.

    The normality null is fitted here, for this dataset: k comes from the
    data (obsm["X_pca"] was truncated to it when the HVG dataset was built)
    and the size range from the dataset's own metacell sizes. Only the
    largest of those matters for the upper end -- the null always extends
    down to normality.DEFAULT_N_MIN, which is what makes it valid for the
    small children refinement goes on to create.

    Each round scores every metacell, and for those above z_cutoff proposes
    a split (split_metacell) and accepts or rejects it (call_split_metacell).
    A metacell whose split is rejected is marked terminal and not revisited:
    the proposal is deterministic given the seed, so retrying it would
    propose the same split and reject it again.

    X is the fixed global PCA and never changes, so scores stay in one
    coordinate system throughout and are comparable across rounds.

    Returns the refined AnnData. It carries uns["final_z"] (the retained
    metacells re-scored in the new PCA) and uns["refined_min_metacell_size"]
    (the effective size floor, after the k+20 adjustment below), which is what
    visualize_HVG_dataset_refinement needs to redraw the final diagnostic
    without rerunning anything.
    """
    out_f = refined_filename(dataset, n_genes)
    if os.path.isfile(out_f) and not overwrite:
        return ad.read_h5ad(out_f)

    s, metacell_start, X, k = load_HVG_pca(dataset, n_genes)
    metacell_sizes = np.unique(metacell_start, return_counts=True)[1].astype(float)
    null = nrm.compute_normality_null(k, metacell_sizes)

    min_metacell_size = max(min_metacell_size, k+20)

    metacells = np.asarray(metacell_start, dtype=object).copy()
    terminal = {}   # label -> why it stopped: too_small / degenerate /
                    # child_too_small / no_improvement
    history = []

    for rnd in range(1, max_rounds + 1):
        labels, z, gaps = compute_z_scores(X, metacells, null, min_metacell_size)
        sizes = np.array([(metacells == lab).sum() for lab in labels])

        # a metacell that has fallen below min_metacell_size cannot yield two
        # viable children, so it is done regardless of its score
        for lab, n in zip(labels, sizes):
            if lab not in terminal and n < min_metacell_size:
                terminal[lab] = "too_small"

        # a departure smaller than max_gap is not worth splitting on however
        # significant it is, so those are left alone as refined
        flagged = [(lab, zi) for lab, zi, gi in zip(labels, z, gaps)
                   if lab not in terminal and zi > z_cutoff
                   and not (max_gap is not None and gi < max_gap)]
        if not flagged:
            if verbose:
                print(f"round {rnd}: nothing flagged -- converged")
            break

        n_split = 0
        for lab, zi in flagged:
            rows = np.flatnonzero(metacells == lab)
            split = split_metacell(X[rows], zi, null,
                                   min_metacell_size=min_metacell_size, seed=seed)
            new_labels = call_split_metacell(split, lab)

            if len(np.unique(new_labels)) > 1:
                metacells[rows] = new_labels
                n_split += 1
            elif split["idx"] is None:
                terminal[lab] = "degenerate"
            elif not np.any(np.isfinite(split["child_z"])):
                terminal[lab] = "child_too_small"
            else:
                terminal[lab] = "no_improvement"

        _, z_after, g_after = compute_z_scores(X, metacells, null, min_metacell_size)
        history.append({"round": rnd, "n_flagged": len(flagged), "n_split": n_split,
                        "n_terminal": len(terminal), "n_metacells": len(z_after),
                        "median_z": float(np.nanmedian(z_after)),
                        "max_z": float(np.nanmax(z_after)),
                        "n_above_cutoff": int(np.nansum(z_after > z_cutoff))})

        if verbose:
            h = history[-1]
            print(f"round {rnd}: flagged {h['n_flagged']:>4} | split {h['n_split']:>4} "
                  f"| terminal {h['n_terminal']:>4} | metacells {h['n_metacells']:>4} "
                  f"| median z {h['median_z']:>7.2f} | max z {h['max_z']:>8.2f}")

        if n_split == 0:
            if verbose:
                print(f"round {rnd}: every flagged metacell rejected its split -- converged")
            break

    # keep only cells whose metacell came out refined; everything else --
    # refractory, too_small, or left unfinished -- is dropped, so the
    # returned object contains exactly the metacells the refinement stands
    # behind
    labels, z_final, gaps_final = compute_z_scores(X, metacells, null, min_metacell_size)
    states = classify_metacells(metacells, z_final, z_cutoff=z_cutoff,
                                min_metacell_size=min_metacell_size, terminal=terminal,
                                gaps=gaps_final, max_gap=max_gap)
    refined_labels = set(labels[states == "refined"])
    keep = np.array([lab in refined_labels for lab in metacells])

    s_refined = s[keep].copy()
    s_refined.obs["metacell"] = pd.Categorical(metacells[keep])

    if verbose:
        counts = np.array([(metacells == lab).sum() for lab in labels])
        print(f"\nkept {int(keep.sum()):,}/{len(keep):,} cells "
              f"({keep.mean():.1%}) in {len(refined_labels)} refined metacells")
        for st in METACELL_STATES:
            sel = states == st
            if sel.any() and st != "refined":
                print(f"  dropped {st}: {int(sel.sum())} metacells, "
                      f"{int(counts[sel].sum()):,} cells")

    

    # recompute the PCA on the retained cells. s[keep] subsets X/obs/obsm but
    # NOT uns, so obsm["X_pca"] and uns["pca"]["U"] would otherwise disagree
    # on the number of rows; and dropping whole metacells changes the
    # covariance the PCA is built from, so the old axes are no longer the
    # ones this subset would produce
    if verbose:
        print("number of cells/genes after filtering: ", s_refined.shape)
        print("Recomputing PCA")
    s_copy = s_refined.copy()
    sc.pp.normalize_total(s_copy, target_sum=1e4)
    sc.pp.log1p(s_copy)
    sc.pp.scale(s_copy)
    U_new, sv_new, _ = np.linalg.svd(s_copy.X, full_matrices=False)

    k_elbow = datasets.find_elbow(sv_new)
    k_new = min(k_elbow, datasets.MAX_K)
    if verbose:
        capped = f", capped to k={k_new}" if k_new < k_elbow else ""
        print(f"PCA: {len(sv_new)} singular values, elbow at k={k_elbow}{capped} "
              f"(was k={k})")

    s_refined.obsm["X_pca"] = U_new[:, :k_new] * sv_new[np.newaxis, :k_new]
    s_refined.uns["pca"] = {"U": U_new[:, :k_new], "sv": sv_new[:k_new], "k": k_new}
    s_refined.uns["full_sv"] = sv_new

    if verbose:
        print("Computing NB model")
    nb = nb_model.compute_metacell_NB(s_refined, poisson=poisson)
    s_refined.uns["nb_model"] = nb
    s_refined.uns["poisson"] = poisson

    # final check: re-score the retained metacells in the NEW PCA. The
    # refinement ran against the old axes, so this is an independent
    # verification rather than a restatement -- k has changed, which means a
    # fresh null too, and metacells that passed before can fail here
    X_new = np.asarray(s_refined.obsm["X_pca"])
    mc_new = s_refined.obs["metacell"].to_numpy()
    null_new = nrm.compute_normality_null(
        k_new, np.unique(mc_new, return_counts=True)[1].astype(float))
    labels_new, z_new, gaps_new = compute_z_scores(X_new, mc_new, null_new,
                                                   min_metacell_size)
    s_refined.uns["final_z"] = {"labels": np.asarray(labels_new, dtype=str),
                                "z": z_new, "gap": gaps_new}
    # the EFFECTIVE floor, after the k+20 adjustment above -- stored because it
    # feeds classify_metacells, so redrawing the diagnostic without it would
    # colour the states differently than this run did
    s_refined.uns["refined_min_metacell_size"] = int(min_metacell_size)

    if verbose:
        n_over = int(np.nansum((z_new > z_cutoff) & (gaps_new > max_gap)))
        print(f"re-scored in the new k={k_new} PCA: median z={np.nanmedian(z_new):.2f}, "
              f"max z={np.nanmax(z_new):.2f}, {n_over}/{len(labels_new)} would now be flagged")

    s_refined.write(out_f, compression="gzip")
    return s_refined


def visualize_HVG_dataset_refinement(dataset, n_genes, z_cutoff=DEFAULT_Z_CUTOFF,
                                     min_metacell_size=None,
                                     max_gap=DEFAULT_MAX_GAP, show=True):
    """The refinement diagnostic for an already-built refined HVG dataset:
    metacell size against normality z-score, scored in the refined PCA.

    A pure reader. Everything comes off the h5ad create_HVG_refined_dataset
    wrote -- obs["metacell"], uns["final_z"] and uns["refined_min_metacell_size"]
    -- so nothing is refined, re-scored or re-fitted here. This is the same
    figure create_HVG_refined_dataset used to draw at the end of a run.

    z_cutoff and max_gap only affect how the metacells are coloured (see
    classify_metacells); pass the values the refinement ran with to see what
    it saw, or different ones to ask how the result would be judged under
    them.

    min_metacell_size is read from the h5ad. The argument is a fallback for
    files written before that key existed; it is ignored otherwise.

    Returns the matplotlib figure.
    """
    s = load_HVG_refined_dataset(dataset, n_genes)

    if "final_z" not in s.uns:
        raise ValueError(
            f"{dataset}/{n_genes} has no uns['final_z'] -- it predates the final "
            "re-scoring being stored; rebuild it with "
            "create_HVG_refined_dataset(..., overwrite=True)")

    metacells = s.obs["metacell"].to_numpy()
    final_z = s.uns["final_z"]
    z = np.asarray(final_z["z"], dtype=float)
    gaps = np.asarray(final_z["gap"], dtype=float)

    # visualize_refinement pairs z with np.unique(metacells) positionally and
    # only checks the length, so verify the labels themselves line up -- a
    # mismatch would otherwise plot every metacell against another's score
    stored = np.asarray(final_z["labels"], dtype=str)
    present = np.unique(metacells).astype(str)
    if not np.array_equal(stored, present):
        raise ValueError(
            f"{dataset}/{n_genes}: uns['final_z']['labels'] does not match "
            f"np.unique(obs['metacell']) ({len(stored)} vs {len(present)} labels); "
            "the file is inconsistent, rebuild it with overwrite=True")

    if "refined_min_metacell_size" in s.uns:
        effective_min = int(s.uns["refined_min_metacell_size"])
    else:
        effective_min = (DEFAULT_MIN_METACELL_SIZE if min_metacell_size is None
                         else min_metacell_size)
        print(f"  {dataset}/{n_genes} predates uns['refined_min_metacell_size']; "
              f"using {effective_min}, so the state colouring may not match the "
              "refinement. Rebuild with overwrite=True to store it.")

    k_new = int(s.uns["pca"]["k"]) if "pca" in s.uns else s.obsm["X_pca"].shape[1]
    return visualize_refinement(
        metacells, z, z_cutoff=z_cutoff, min_metacell_size=effective_min,
        gaps=gaps, max_gap=max_gap, show=show,
        title=f"{dataset}/{n_genes} -- refined, re-scored in k={k_new} PCA")


def create_all_HVG_refined_datasets(dataset_list=datasets.DEFAULT_DATASETS,
                                    n_genes_list=datasets.DEFAULT_N_GENES,
                                    overwrite=False, verbose=True, **kwargs):
    """Refine every (dataset, n_genes) HVG h5ad, writing one
    HVG_refined_info.csv per dataset.

    A combination whose UNREFINED h5ad has not been built is skipped with a
    warning rather than raising, so one missing input does not abort the
    sweep. kwargs are passed to create_HVG_refined_dataset (z_cutoff,
    min_metacell_size, max_gap, max_rounds, seed, poisson).

    The null is refitted inside every call rather than cached across
    combinations that happen to share a k -- it takes a few seconds against
    a refinement that takes longer, and keeping it per-call means the size
    range always matches the dataset being refined.
    """
    for dataset in dataset_list:
        if verbose:
            print(f"\n{'=' * 64}\nRefining dataset: {dataset}")

        # a dataset whose every refined h5ad and info csv already exist is
        # skipped outright, so a fill-in-the-gaps run neither re-reads its
        # h5ads nor rewrites the same csv
        complete = all(os.path.isfile(refined_filename(dataset, n_genes))
                       for n_genes in n_genes_list)
        if not overwrite and complete and os.path.isfile(refined_info_filename(dataset)):
            if verbose:
                print(f"  {dataset}: every refined h5ad and the info csv exist, skipping")
            continue
        info_rows = []

        for n_genes in n_genes_list:
            try:
                s_start = datasets.load_HVG_dataset(dataset, n_genes)
            except ValueError as e:
                print(f"  SKIPPING {dataset}/{n_genes}: {e}")
                continue

            if verbose:
                print(f"\n-- {dataset}/{n_genes} --")
            s_ref = create_HVG_refined_dataset(dataset, n_genes, overwrite=overwrite,
                                               verbose=verbose, **kwargs)

            info_rows.append({
                "n_genes": n_genes,
                "k": int(s_start.obsm["X_pca"].shape[1]) if "X_pca" in s_start.obsm else None,
                "n_cells_start": int(s_start.n_obs),
                "n_cells_refined": int(s_ref.n_obs),
                "frac_cells_kept": s_ref.n_obs / s_start.n_obs,
                "n_metacells_start": int(s_start.obs["metacell"].nunique()),
                "n_metacells_refined": int(s_ref.obs["metacell"].nunique()),
                "min_metacell_size": int(s_ref.obs["metacell"].value_counts().min()),
                "median_metacell_size": float(s_ref.obs["metacell"].value_counts().median()),
            })

        if info_rows:
            info_f = refined_info_filename(dataset)
            pd.DataFrame(info_rows).to_csv(info_f, index=False)
            if verbose:
                print(f"\n  wrote {info_f}")

    return None
