
This repo contains the code used in the manuscript, **A Metacell Model of Single Cell RNA-seq Counts Yields a Gaussian
Mixture Model in PCA Space**, *BioRxiv*

The code is arranged as a pipeline composed of five stages:

1. Dataset Construction
2. Bulk Eigenvalue Analysis
3. Spike Eigenvalue and Eigenvector Analysis
4. construction of knn, UMAP, and other data structures used by the manuscript
5. figure and table construction for the manuscript


## Workflow at a glance

`make.make_all()` runs these five stages in dependency order. Each stage sweeps through the datasets specified in
`datasets.DEFAULT_DATASETS`.   

### Dataset Construction

The datasets are originally h5ad files (stored in `../data/` by default) with raw count matrices.  Dataset construction then proceeds as follows through scripts with a `dataset` prefix.   `datasets.py` is the main control script.  Each datasets QC is done in `dataset_{datasetname}.py`.

* For each dataset, a base dataset anndata file is created in `../analysis_base/`.  The function reads the downloaded file (an h5ad from CELLxGENE, GEO or Zenodo, or the 10x matrix files in the case of PBMC), keeps only the cells the manuscript uses (e.g. a random subsample of 200,000 cells for Breast), and then QC.   The base anndata also standardizes the annotations, so that every base dataset has a `cell_type` column in `s.obs` (and, where the source provides them, `sample` and `batch` columns)
* From each base dataset, a 'non-refined' anndata object is created and saved in `../analysis_HVG'.   These are anndata objects with 1000, 2000 or 5000 HVG selected and an initial metacell partition generated just by Supercell.  These files correspond to the first step of the metacell construction as described in the manuscript Methods.  
* From the `non-refined` anndata objects we create `refined` anndata objects.   These have metacell partitions that include the second step of the metacell construction described in the Methods.  These anndata object also have `s.uns['nb_model']` which contains the negative binomial distributions of the metacells.  All 'refined' anndata objects are saved in `../analysis_refined_HVG'.
* From the base dataset, we also create 'celltype' anndata objects, which build a metacell partition based on the cell types, as mentioned in the Methods.  This metacell partition serves to show that our metacell refinement algorithm is not important in the main result of the paper characterizing the metcall PCA embeddings as GMM distributed.  These anndata objects are save in `../analysis_celltype_HVG/`.

### Bulk Eigenvalue Analysis

This stage compares the bulk (noise) eigenvalues of the scaled count matrix to the LML-RMT prediction of the bulk spectrum.  `analysis_bulk.py` is the control script.  The prediction comes from `lognorm_workflow_analytic.py`, which builds the spike-bulk decomposition from the `s.uns['nb_model']` of a refined anndata object and evaluates the bulk density with the MIXANDMIX solver in `mixandmix.py`.

* For each dataset and each of the 1000, 2000 and 5000 HVG refined anndata objects, a pkl is saved in `../analysis_bulk_refined/`.  It holds the empirical eigenvalues of the true scaled count matrix (`evs_pca`) and of the scaled count matrices formed from metacell model counts (`evs_nb`), from counts with genes permuted within metacells (`evs_perm`) and from the Gaussian model Z_normal (`evs_normal`), each with the top k outlier eigenvalues removed.  Alongside these it holds the analytic mixture density on a grid (`evs_mixture`), the Marchenko-Pastur density, and the right edge of the bulk `b_plus`.
* These pkls are the inputs of the bulk figure and the Marchenko-Pastur figure (Results, Bulk and Outlier Eigenvalues) and correspond to the "Computing the Bulk Spectrum" section of the Methods.

### Spike Eigenvalue and Eigenvector Analysis

This stage computes the LML-RMT predictions for the outlier eigenvalues and eigenvectors, i.e. the predicted PCA embedding, and the PCA embeddings of the four count matrices the predictions are compared against.  `analysis_spike.py` is the control script.  The prediction itself is computed by `lognorm_workflow_analytic.log_normalized_PCA`, which calls into `spiked_mixture_model.py` and `spiked_mixture_model2.py` (see "From the Methods to the code" below).  Only the 1000 HVG anndata objects are used, see `make.SPIKE_N_GENES`.

* For each dataset, two pkls are saved in `../analysis_spike_refined/`.  `{dataset}_HVG_1000_pca.pkl` holds `pca_theory`, the LML-RMT prediction: the outlier eigenvalues, the predicted left singular vectors, and each metacell's mean and covariance in PCA space, which together form the GMM.  `{dataset}_HVG_1000.pkl` holds the `spec_*` dictionaries: the SVD (U, singular values and V) of the scaled count matrix formed from Gaussian noise (`spec_n`, called normal in the manuscript), from counts sampled from the metacell model (`spec_nb`), from the true counts with genes permuted within metacells (`spec_perm`) and from the true counts (`spec_true`), plus `spec_analytic`, the prediction rearranged into the same cell order, and the metacell label of every cell.  Everything downstream reads the two pkls together through `analysis_spike.load_spike_result`.
* `analysis_spike_celltype.py` does the same for the celltype anndata objects and saves the results in `../analysis_spike_celltype/`.  These support the statement in the Methods that the GMM result does not depend on our metacell refinement; no manuscript figure reads them.
* The spike pkls feed the outlier eigenvalue figure, the KS normality figure, the metacell variance figure and the neighbor distance figure (Results, Bulk and Outlier Eigenvalues, Geometry Within Metacells and Geometry Across Metacells) and correspond to the "Computing the Outlier Eigenvalues" and "Computing the Outlier Eigenvectors" sections of the Methods.

### Construction of knn, UMAP, and Other Data Structures

This stage builds the data structures derived from the PCA embeddings stored in the spike pkls.  `analysis_knn.py` and `analysis_umap.py` are the control scripts; both require a dataset's spike pkl to exist.

* For each dataset, `analysis_knn.py` computes the k-nearest neighbor graph (k = 10, using pynndescent) of every embedding (analytic, normal, nb, perm and true) and saves the per-cell neighbor indices in `../analysis_knn_refined/`.  The metacell-to-metacell edge fractions used in the knn figure and the knn table are cheap to recompute from these indices, see `analysis_knn.build_mixture_graph` and `figures_util.compute_knn_edge_fractions`.  For the true and permuted embeddings the pkl also holds a split-half knn graph that serves as a noise floor for the comparison.
* Next to each knn pkl, `figures_util.compute_knn_replicate_errors` saves the knn relative error of `table.KNN_REPLICATES` fresh draws of Z_normal.  These give the spread reported for the normal column of the knn table.
* `analysis_umap.py` computes a 2D UMAP of every embedding and saves it in `../analysis_umap_refined/`.  The UMAPs of the four empirical embeddings are warm-started from the UMAP of the analytic embedding so that all layouts are comparable.  These are the UMAP panels of the qualitative visualization figure (Results, Figure 1).
* The knn structures correspond to the knn analysis in the Results (Geometry Across Metacells); the UMAPs are only used for visualization.

### Figure and Table Construction

This stage draws every figure and table in the manuscript into `../figures/` from the pkls and anndata objects above; nothing here recomputes a model.  `figures.py` is the control script, with one function per figure, and `make.FIGURE_STEPS` lists the functions in the manuscript's order of appearance, which is what `make.make_figures` runs.  The computations behind the panels are in `figures_util.py`, the visualization figure in `figures_visualize.py`, and the tables in `table.py`.

* Each figure is saved as a pdf: `figure_visualize.pdf` (UMAPs and the random projection of one metacell, Figure 1), `figure_bulk.pdf` and `figure_MP.pdf` (bulk eigenvalues against the LML-RMT density and Marchenko-Pastur), `figure_spectrum.pdf` (outlier eigenvalues), `figure_normality_KS.pdf` and `figure_normality_spectrum.pdf` (normality and variance of individual metacells), `figure_neighbor_distance.pdf` (scaled distances between neighboring metacells) and `figure_knn_adjacency.pdf` (knn edge fractions).  `figure_SI_bulk.pdf` is the bulk figure for the datasets not shown in the main text and is currently not included in the manuscript.
* The tables are saved as LaTeX fragments that the manuscript `\input`s: `datasets.tex` (the dataset table), `knn_table.tex` (per-dataset knn relative errors) and `processing_table.tex` (download, subsetting and QC of each dataset).
* Figures with one panel per dataset use the four datasets in `figures.DEFAULT_FIGURE_DATASETS`; figures that pool datasets into one panel per method use all of `datasets.DEFAULT_DATASETS`.  Each quantitative panel is annotated with REL ERR, the median relative error of the method against the LML-RMT prediction (`figures_util.median_relative_error`), averaged over datasets.

Run the whole thing from this folder with

```bash
python3 -c "import make; make.make_all()"
```

or a single stage or figure, e.g. `make.make_spike()` or `figures.make_knn_figure()`.
Only the figures are redrawn on every run; h5ads and pkls are read from the cache.

Two conventions run through all of the code. Every cached artifact is keyed by
`(dataset, n_genes)`, and every function that reads one takes a `refined` flag
selecting the refined metacell partition (what the manuscript uses) or the
unrefined SuperCell one. The five spectra are always named `analytic`, `normal`,
`nb`, `perm`, `true`; `figures_util.method_label` maps them to the names printed
in the paper (LML-RMT, normal, metacell, permuted, true) and
`figures_util.display_name` maps dataset keys to the names in the tables
(`Heart_large` is shown as Heart, `Breast_small` as Breast (1 donor)).

## RMT computations

The RMT computations can be split into the following tasks, described mathematically in the Methods,

1. Compute the spike-bulk decomposition of `Z_\normal = S + B_\normal`.
2. Compute the eigenvalues of `B_\normal`.
3. Compute the outlier eigenvalus of 'Z_\normal'
4. Compute the `\alpha` and `\beta` coefficients in the expansion for the outlier left singular vectors of 'Z_\normal'.  These coefficients determine the mean of the metacells.
5. Compute the `delta` coefficients in the expansion for the outlier left singular vectors of 'Z_\normal'.  These coefficients determine the covariance of the metacells.

Tasks 1, 3, 4 and 5 are run together, for one dataset, by `lognorm_workflow_analytic.log_normalized_PCA(nb, n, L0)`, where `nb` is the `s.uns['nb_model']` of a refined anndata object.  It returns the `pca_theory` dictionary stored in the spike pkls.  Task 2 is run separately by `analysis_bulk.anndata2bulk`, but its solver is also called inside tasks 3 to 5.

### Task 1: The spike-bulk decomposition

Everything for this task is in `lognorm_workflow_analytic.py`, starting from the negative binomial parameters of the metacells.

* `compute_NB_log_moments(mu, theta, Li, L0)` computes E[Y_ig] and V[Y_ig] for Y_ig = log(1 + L0 X_ig / L_i) with X_ig negative binomial, the entry-wise moments defined in the Methods.
* `log_normalized_spiked_mixture_model(nb, n, L0)` forms mu_tilde and sigma_tilde from those moments and returns the pair `(m_model, spike)`.  `spike` is the matrix S, one identical row per cell of a metacell.  `m_model` is the mixture description of B_normal that all the RMT code consumes: the entry-wise variances of each metacell (`covariance`), the metacell frequencies (`mixture_frequencies`), the metacell labels and the aspect ratio `gamma`.
* `simulate_Z_guassian(m_model, spike)` draws a realization of Z_normal = S + B_normal; this is the `normal` count matrix of the manuscript.  `simulate_noise_gaussian` draws B_normal alone.
* These are called for each dataset by `analysis_spike.anndata2spike` and `analysis_bulk.anndata2bulk`.

### Task 2: The eigenvalues of B_normal

The solver is `mixandmix.py`; `lognorm_workflow_analytic.py` wraps it and `analysis_bulk.py` caches its output.

* `mixandmix.mixandmix_density(m_model, x_grid, params)` solves the BG-C fixed point equations for the Stieltjes transform at every point of `x_grid`, using Anderson mixing and a homotopy in the imaginary part (`solve_e_and_m`, `fixed_point_map`, `anderson_step`), and returns the spectral density on the grid together with the per-metacell solutions g_k(z).  Tasks 3 to 5 reuse the same solver at single points through `spiked_mixture_model.compute_g`.
* `mixandmix.compute_b_plus(m_model)` locates the right edge of the bulk, b+.
* `lognorm_workflow_analytic.analytic_spectral_density(m_model, x_grid, n, p)` and `lognorm_workflow_analytic.compute_b_plus(m_model)` are the entry points; they rescale the model to the manuscript's normalization and call the solver.
* `analysis_bulk.anndata2bulk` stores the results as `evs_mixture` and `b_plus` in the bulk pkl, which `figures_util.make_bulk_figure` and `figures_util.make_MP_figure` draw.

### Task 3: The outlier eigenvalues of Z_normal

The computation is in `spiked_mixture_model.py`; the entry point is `lognorm_workflow_analytic.analytic_eigenvalues(m_model, spike)`.

* `lognorm_workflow_analytic.spike_svd2_matrix(spike)` computes the SVD of S, giving the rho_i, u_i and v_i (`cut_spike` truncates the rank when needed).
* `spiked_mixture_model.compute_g(m_model, z)` evaluates the fixed point of Task 2 at a single z; `make_Q_bar`, `make_tilde_Q_bar` and `make_R` build the BG-C deterministic equivalents of Q and Q_tilde; `compute_spike_Q_moments`, `compute_Q2` and `compute_tQWWtQ` evaluate the quadratic forms u_i^T Q u_j and v_i^T Q_tilde v_j that enter M'.
* `spiked_mixture_model.make_M_matrix(...)` assembles the 2r x 2r matrix M'.
* `spiked_mixture_model.compute_spike_eigenvalues(m_model, spike, accuracy, isotropic=False)` performs the grid search over [b+ + 0.1, max rho_i^2 + 5] with brentq root finding on det(M') = 0, refining the grid until the k roots are found, and returns the outlier eigenvalues.
* `log_normalized_PCA` stores them as singular values in `pca_theory["values"]`; `figures_util.compute_outlier_spectrum` compares them with the empirical outliers for the outlier eigenvalue figure.

### Task 4: The alpha and beta coefficients (metacell means)

The coefficients are computed in `spiked_mixture_model.py` and turned into metacell means in `lognorm_workflow_analytic.py`.

* `spiked_mixture_model.compute_alphas_betas(m_model, spike, g, z)` solves M' (alpha, beta) = 0 for one outlier z by taking the null vector of M' from its SVD, with the scale fixed by the normalization ||x|| = 1 (a one-dimensional root find); it returns alpha and beta.
* `spiked_mixture_model.compute_alpha0(...)` computes alpha_0, the component along the all-ones vector, which the Methods note is negligible.
* `spiked_mixture_model.compute_spike_projection_coef(m_model, spike, z)` is the per-outlier entry point: it calls `compute_g`, `compute_alphas_betas`, `compute_alpha0` and `compute_delta` (Task 5) and returns a dictionary with `alpha`, `beta`, `alpha0` and `delta`.
* `lognorm_workflow_analytic.analytic_left_singular_vectors(eig_value, m_model, spike)` converts the coefficients into the predicted left singular vector's value in each metacell, i.e. the sum of alpha_i (u_i)_a (using `make_E_matrix` to expand metacell-constant vectors to cells), and returns the per-metacell mean, the predicted gene loading, the standard deviation and the raw alpha, beta and delta.
* `log_normalized_PCA` assembles these over the k outliers into `pca_theory["mean_matrix"]`, `alpha_matrix` and `beta_matrix`; `get_GMM(pca, label)` returns the mean (and covariance) of a single metacell.

### Task 5: The delta coefficients (metacell covariances)

The variances come from `spiked_mixture_model.py`, the covariances across PCA coordinates from `spiked_mixture_model2.py`, and both are assembled by `lognorm_workflow_analytic.log_normalized_PCA`.

* `spiked_mixture_model.compute_delta(m_model, spike, g, alpha, beta, z)` computes delta_a^2 for every metacell a at one outlier, i.e. the variance of that PCA coordinate within each metacell; it is returned through `compute_spike_projection_coef` and stored as `pca_theory["delta_matrix"]` and `std_matrix`.
* `spiked_mixture_model2.compute_covariance_matrix(m_model, spike, evs, alpha_matrix, beta_matrix)` computes the covariance between PCA coordinates within each metacell, the large matrix limit of the product of the two bracketed terms in the Methods, vectorized over metacells and coordinate pairs (`compute_delta_dot`, `compute_Q2_all`, `compute_tQWWtQ_all`, `compute_cross_terms`); it returns one k x k covariance matrix per metacell, stored as `pca_theory["covariance_matrix"]`.
* `lognorm_workflow_analytic.check_noise_covariance(pca_theory)` checks that the diagonal of that covariance agrees with the delta_a^2 of `compute_delta`.
* `lognorm_workflow_analytic.sample_mixture_gaussian` and `simulate_PCA_embedding` draw a PCA embedding from the resulting GMM, which is how the e_a of the Methods are simulated as independent Gaussians.