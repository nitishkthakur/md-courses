# Research Dossier: Manifold Methods
## Isomap, LLE, Spectral Embedding, MDS, PHATE
### For: Dimensionality Reduction Masterclass — Chapter Writing Agent

---

## TABLE OF CONTENTS

1. [Section 1: Paper Citations and Background](#section-1-paper-citations-and-background)
2. [Section 2: Algorithm Mechanics and Theory](#section-2-algorithm-mechanics-and-theory)
3. [Section 3: sklearn 1.5.x API Reference](#section-3-sklearn-15x-api-reference)
4. [Section 4: Specialized Python Packages](#section-4-specialized-python-packages)
5. [Section 5: Real-World Applications (6–8 examples)](#section-5-real-world-applications)
6. [Section 6: Hyperparameter Tuning Guide](#section-6-hyperparameter-tuning-guide)
7. [Section 7: Expert Practitioner Checklist](#section-7-expert-practitioner-checklist)
8. [Section 8: Evaluation Metrics](#section-8-evaluation-metrics)
9. [Section 9: API Version History and Breaking Changes](#section-9-api-version-history-and-breaking-changes)
10. [Section 10: Complexity Reference Table](#section-10-complexity-reference-table)

---

## SECTION 1: PAPER CITATIONS AND BACKGROUND

### 1.1 Foundational Papers (must cite in chapter)

---

**[PAPER-01] Isomap — The Foundational Paper**

- **Authors:** Joshua B. Tenenbaum, Vin de Silva, John C. Langford
- **Title:** A Global Geometric Framework for Nonlinear Dimensionality Reduction
- **Journal:** Science
- **Volume/Issue/Pages:** Vol. 290, No. 5500, pp. 2319–2323
- **Date:** December 22, 2000
- **DOI:** 10.1126/science.290.5500.2319
- **URL:** https://www.science.org/doi/10.1126/science.290.5500.2319
- **What you learn:**
  - The paper introduces Isomap as an extension of MDS that replaces Euclidean distances with geodesic distances estimated via shortest paths on a neighborhood graph.
  - Three landmark demonstrations: (1) recovering the underlying 2D pose manifold from a set of face images under varying illumination and viewpoint, (2) recovering the 1D manifold of a rotating human hand, and (3) recovering the configuration-space manifold of a robot arm.
  - Theoretical guarantee: if the manifold is isometrically embeddable in Euclidean space, Isomap recovers the true geometry as N → ∞.
- **Difficulty:** Accessible (5 pages in Science format). Math requires understanding of shortest-path graphs and eigendecomposition.

---

**[PAPER-02] LLE — The Foundational Paper**

- **Authors:** Sam T. Roweis, Lawrence K. Saul
- **Title:** Nonlinear Dimensionality Reduction by Locally Linear Embedding
- **Journal:** Science
- **Volume/Issue/Pages:** Vol. 290, No. 5500, pp. 2323–2326
- **Date:** December 22, 2000
- **DOI:** 10.1126/science.290.5500.2323
- **URL:** https://www.science.org/doi/10.1126/science.290.5500.2323
- **What you learn:**
  - LLE is presented as a two-step algorithm: (1) each point is expressed as a weighted linear combination of its neighbors, (2) those same weights reconstruct each point in low-dimensional space.
  - Key insight: local linear structure is preserved globally by requiring that the reconstruction weights be the same in both high- and low-dimensional spaces.
  - Demonstrated on faces (with varying lighting and viewpoints) and on 3D renderings of a moving hand — comparable demos to Isomap but published side-by-side in the same issue.
- **Difficulty:** Accessible. The math is elegant: two constrained quadratic optimization problems.

---

**[PAPER-03] MLLE — Modified Locally Linear Embedding**

- **Authors:** Zhenyue Zhang, Jing Wang
- **Title:** MLLE: Modified Locally Linear Embedding Using Multiple Weights
- **Venue:** Advances in Neural Information Processing Systems 19 (NeurIPS 2006)
- **Pages:** 1593–1600
- **Publisher:** MIT Press, 2007
- **URL:** https://proceedings.neurips.cc/paper/2006/hash/fb2606a5068901da92473666256e6e5b-Abstract.html
- **What you learn:**
  - Standard LLE suffers from a rank-deficiency problem when the local weight matrix is poorly conditioned, leading to distorted embeddings.
  - MLLE uses multiple linearly independent local weight vectors for each neighborhood, making the method more stable.
  - MLLE produces embeddings as accurate as HLLE but without HLLE's significant extra computational cost (HLLE requires O[N d^6] for QR decompositions).
- **Difficulty:** Moderate. Requires understanding of matrix conditioning and rank-deficiency.

---

**[PAPER-04] HLLE — Hessian Eigenmaps**

- **Authors:** David L. Donoho, Carrie Grimes
- **Title:** Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data
- **Journal:** Proceedings of the National Academy of Sciences (PNAS)
- **Volume/Issue/Pages:** Vol. 100, No. 10, pp. 5591–5596
- **Date:** May 13, 2003
- **DOI:** 10.1073/pnas.1031596100
- **URL:** https://www.pnas.org/doi/10.1073/pnas.1031596100
- **What you learn:**
  - HLLE derives from a conceptual framework of local isometry: it estimates the Hessian of a smooth function at each neighborhood and uses these Hessians to define a quadratic form.
  - Provides theoretical guarantees beyond Isomap: the parameter space does NOT have to be convex, covering a significantly wider class of situations.
  - Also addresses "When does Isomap recover the natural parametrization of families of articulated images?" — essentially a theoretical analysis of Isomap's limits.
- **Difficulty:** Graduate level. Requires differential geometry background.

---

**[PAPER-05] LTSA — Local Tangent Space Alignment**

- **Authors:** Zhenyue Zhang, Hongyuan Zha
- **Title:** Principal Manifolds and Nonlinear Dimensionality Reduction via Local Tangent Space Alignment
- **Journal:** SIAM Journal on Scientific Computing
- **Volume/Issue/Pages:** Vol. 26, No. 1, pp. 313–338
- **Year:** 2004
- **DOI:** 10.1137/S1064827502419154
- **URL:** https://epubs.siam.org/doi/10.1137/S1064827502419154
- **What you learn:**
  - LTSA computes the tangent space at each point via local PCA, then aligns those tangent spaces into a global coordinate system.
  - Key intuition: a correctly unfolded manifold has all tangent hyperplanes aligned; LTSA operationalizes this.
  - One of few methods that can unroll the "Swiss Roll" dataset without major distortion; useful for manifolds with complex global topology.
- **Difficulty:** Moderate to graduate level. Requires linear algebra and manifold theory.

---

**[PAPER-06] Laplacian Eigenmaps / Spectral Embedding**

- **Authors:** Mikhail Belkin, Partha Niyogi
- **Title:** Laplacian Eigenmaps for Dimensionality Reduction and Data Representation
- **Journal:** Neural Computation
- **Volume/Issue/Pages:** Vol. 15, No. 6, pp. 1373–1396
- **Year:** 2003
- **DOI:** 10.1162/089976603321780317
- **URL:** https://dl.acm.org/doi/10.1162/089976603321780317
- **What you learn:**
  - The algorithm builds a weighted neighborhood graph, constructs the graph Laplacian L = D − A (where D is the diagonal degree matrix), and uses the bottom eigenvectors (excluding the trivial zero eigenvector) as coordinates.
  - Provides natural connection to spectral clustering: the eigenvectors minimize the objective "nearby points in original space should remain nearby in embedding."
  - The method has locality-preserving properties rooted in spectral graph theory and the heat equation on Riemannian manifolds.
- **Difficulty:** Moderate. Accessible with knowledge of graph theory and linear algebra.

---

**[PAPER-07] Diffusion Maps**

- **Authors:** Ronald R. Coifman, Stéphane Lafon
- **Title:** Diffusion Maps
- **Journal:** Applied and Computational Harmonic Analysis (Special Issue: Diffusion Maps and Wavelets)
- **Volume/Issue/Pages:** Vol. 21, No. 1, pp. 5–30
- **Year:** 2006
- **DOI:** 10.1016/j.acha.2006.04.006
- **URL:** https://www.sciencedirect.com/science/article/pii/S1063520306000546
- **What you learn:**
  - Diffusion maps define a diffusion distance between points: how easily a random walk diffuses from one point to another in t steps. This is more robust to noise than geodesic distance.
  - The key parameter alpha (0 to 1) controls the normalization: alpha=0 gives unnormalized graph Laplacian (similar to Laplacian Eigenmaps), alpha=1 gives Fokker-Planck operator (removes sampling density effects), alpha=0.5 gives Laplace-Beltrami operator.
  - The diffusion map at time t is defined by the top eigenvectors of the Markov matrix, with eigenvalues raised to the power t; choosing t controls how many manifold scales are visible.
- **Difficulty:** Graduate level. Requires stochastic processes background.

---

**[PAPER-08] PHATE**

- **Authors:** Kevin R. Moon, David van Dijk, Zheng Wang, Scott Gigante, Daniel B. Burkhardt, William S. Chen, Kristina Yim, Antonia van den Elzen, Matthew J. Hirn, Ronald R. Coifman, Natalia B. Ivanova, Guy Wolf, Smita Krishnaswamy
- **Title:** Visualizing structure and transitions in high-dimensional biological data
- **Journal:** Nature Biotechnology
- **Volume/Issue/Pages:** Vol. 37, pp. 1482–1492
- **Year:** 2019
- **DOI:** 10.1038/s41587-019-0336-3
- **URL:** https://www.nature.com/articles/s41587-019-0336-3
- **What you learn:**
  - PHATE introduces an "information geometry distance" between data points derived from comparing diffusion probability distributions rather than raw coordinates.
  - It preserves both local AND global structure — a major advantage over t-SNE (local) and classical MDS (global only).
  - Demonstrated on human germ-layer differentiation scRNA-seq data, revealing developmental branches not visible with other methods.
- **Difficulty:** Accessible to bioinformaticians; moderate for CS audience. Key math is diffusion distances with information-geometric interpretation.

---

**[PAPER-09] Kruskal (1964) — Non-Metric MDS**

Two companion papers:
- **Paper A:** "Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis" — Psychometrika, Vol. 29, pp. 1–27 (1964)
- **Paper B:** "Nonmetric multidimensional scaling: A numerical method" — Psychometrika, Vol. 29, pp. 115–129 (1964)
- **DOI (B):** 10.1007/BF02289694
- **What you learn:**
  - Paper A defines stress as an explicit least-squares loss function and proposes a gradient descent method to minimize it.
  - Paper B introduces monotone regression of distances upon dissimilarities (isotonic regression), enabling non-metric MDS.
  - Together they established MDS as a practical algorithm — 5773+ citations in Google Scholar as of 2016.
- **Difficulty:** Accessible. Historical statistics paper; elegant and readable.

---

## SECTION 2: ALGORITHM MECHANICS AND THEORY

### 2.1 Isomap — How It Works

**Three stages:**
1. **Nearest-neighbor graph:** For each point, find k nearest neighbors (by Euclidean distance) and connect them with edges weighted by Euclidean distance.
2. **Geodesic distances:** Compute all-pairs shortest paths using Dijkstra (sparse graphs) or Floyd-Warshall (dense graphs). The resulting N×N distance matrix D_G approximates geodesic distances on the manifold.
3. **MDS embedding:** Apply classical MDS to D_G: compute the double-centered Gram matrix B = -0.5 * HDH (where H = I - (1/N)11^T), then take the top d eigenvectors scaled by their eigenvalues.

**Key insight:** Euclidean distance between two points on a curved manifold (e.g., points on a Swiss Roll) does NOT reflect their true relationship. Geodesic distance (walking along the manifold surface) does.

**When Isomap works well:**
- Single continuous manifold (not disjoint)
- Relatively low curvature
- Dense enough sampling
- Manifold is "convex" in some sense (no holes)

**When Isomap fails:**
- Short-circuit errors: if k is too large, edges "jump" across the manifold, connecting distant parts incorrectly
- Noisy data: noise moves points off the manifold, creating spurious shortcuts
- Non-convex manifolds (e.g., a torus or figure-8)

---

### 2.2 LLE — How It Works

**Three stages:**
1. **K-nearest neighbors:** For each point x_i, find its k nearest neighbors.
2. **Weight computation:** Find the weights W_ij that best reconstruct x_i from its neighbors: minimize ||x_i - Σ_j W_ij x_j||² subject to Σ_j W_ij = 1. This is a constrained least squares problem for each neighborhood.
3. **Embedding:** Find the low-dimensional coordinates Y that are best reconstructed by the same weights W: minimize ||Y_i - Σ_j W_ij Y_j||². This is a sparse eigenvalue problem.

**Key insight:** The barycentric coordinates (reconstruction weights) of each point within its local neighborhood are invariant to rotation, scaling, and translation — so they capture intrinsic local structure.

**Requirement:** n_neighbors > n_components (you need more neighbors than target dimensions).

**Variants (all available via `method=` parameter in sklearn):**
- `'standard'` (LLE): Basic algorithm; may be poorly conditioned for some datasets
- `'modified'` (MLLE): Uses multiple weight vectors per neighborhood; more stable; recommended over standard
- `'hessian'` (HLLE): Uses Hessian-based quadratic form; most theoretically grounded; requires n_neighbors > n_components*(n_components+3)/2; slow for high d
- `'ltsa'` (LTSA): Aligns local tangent spaces rather than reconstruction weights; often excellent results

---

### 2.3 Spectral Embedding (Laplacian Eigenmaps) — How It Works

**Three stages:**
1. **Graph construction:** Build a neighborhood graph (k-NN or epsilon-ball). Weight edges using a heat kernel: W_ij = exp(-||x_i - x_j||² / (2σ²)) for connected points, 0 otherwise.
2. **Laplacian:** Compute the unnormalized graph Laplacian L = D - W (or the normalized version D^(-1/2) L D^(-1/2)), where D is the diagonal degree matrix.
3. **Eigenvectors:** Solve the generalized eigenvalue problem: L f = λ D f. The eigenvectors corresponding to the d smallest nonzero eigenvalues are the embedding coordinates.

**Connection to physics:** The eigenvectors of the Laplacian are like the "natural harmonics" of the data graph — they describe how heat would diffuse across the data manifold.

**Connection to graph clustering:** Spectral clustering uses the same eigenvectors but applies k-means to them afterward.

**When to use over LLE:** When you want clustering structure preserved, or when data has complex topology that LLE can't handle.

---

### 2.4 MDS (Multi-Dimensional Scaling) — Three Variants

**Classical MDS (cMDS) — also called PCoA:**
- Given a dissimilarity matrix D, double-center it to get a Gram matrix B
- Compute top d eigenvectors: this is equivalent to PCA on the Gram matrix
- Has an analytical solution — no optimization needed
- Equivalent to PCA when distances are Euclidean
- Available in sklearn as `sklearn.manifold.ClassicalMDS` (new in 1.9) or via `MDS(init='classical_mds')` initialization

**Metric MDS:**
- Minimizes stress: Σ_{i<j} (d_ij - d̂_ij)² where d̂_ij are embedding distances
- Uses SMACOF algorithm (Scaling by Majorizing a Complicated Function) — an iterative majorization algorithm
- sklearn: `MDS(metric=True)` (sklearn 1.5) or `MDS(metric_mds=True)` (sklearn 1.8+)

**Non-metric MDS:**
- Only requires that the ORDERING of distances is preserved (isotonic regression)
- More flexible: useful when data is ordinal (e.g., "item A is more similar to B than to C")
- Uses Kruskal's stress-1: sqrt(Σ(d̂_ij - d_ij)² / Σd_ij²)
- Stress-1 benchmarks: 0 = perfect, 0.025 = excellent, 0.05 = good, 0.1 = fair, 0.2 = poor
- sklearn: `MDS(metric=False)` (sklearn 1.5) or `MDS(metric_mds=False)` (sklearn 1.8+)

---

### 2.5 Diffusion Maps — How It Works

**Conceptual framework:**
1. Build a kernel matrix K_ij = k(x_i, x_j) where k is typically a Gaussian kernel.
2. Normalize: D_i = Σ_j K_ij; then form M_ij = K_ij / (D_i D_j)^alpha. Alpha controls density normalization.
3. Row-normalize M to get a Markov matrix P (transition probabilities of a random walk).
4. Compute eigenvectors φ_l and eigenvalues λ_l of P.
5. The diffusion map at time t is: x_i → (λ_1^t φ_1(i), λ_2^t φ_2(i), ..., λ_d^t φ_d(i))

**Alpha parameter meaning:**
- alpha=0: unnormalized graph Laplacian (equivalent to Laplacian Eigenmaps; density-sensitive)
- alpha=0.5: approximates Laplace-Beltrami operator on manifold
- alpha=1: Fokker-Planck normalization (removes sampling density effects — most "geometric")

**t parameter meaning:**
- Small t: only local structure captured
- Large t: global structure; many small eigenvalues decay to zero, simplifying the manifold representation
- t='auto' in PHATE

**Diffusion distance vs geodesic distance:**
- Geodesic: single shortest path (vulnerable to shortcuts)
- Diffusion: integral over all paths of length t (robust to noise, gaps, shortcuts)

---

### 2.6 PHATE — How It Works

**PHATE pipeline (building on diffusion maps):**
1. Compute a diffusion operator P (as in diffusion maps)
2. Raise P to the power t: P^t (diffusion at t steps)
3. For each pair of points, compute the "potential distance" using KL divergence between their t-step diffusion distributions: Φ_t(x_i, x_j) = ||log(P^t_i) - log(P^t_j)||
4. Embed these potential distances using metric MDS (or SGD-MDS for speed)

**Why PHATE > t-SNE for biology:**
- t-SNE: loses global structure (branch hierarchies, developmental trajectories invisible)
- UMAP: better than t-SNE but still primarily local
- PHATE: explicitly designed to reveal trajectories and branching patterns in biological data

**Multiscale PHATE (2021 NeurIPS → Nature Biotechnology extension):**
- Sweeps through all levels of data granularity via "diffusion condensation"
- Applied to 54M single cells from 168 COVID-19 patients
- Predicts disease outcome better than naive featurizations

---

## SECTION 3: SKLEARN 1.5.X API REFERENCE

**IMPORTANT NOTE:** Sklearn is currently at version 1.9.x as of mid-2026. The course uses 1.5.x parameters as baseline. Where parameters have changed, both 1.5.x and 1.9.x are documented and changes are flagged.

---

### 3.1 `sklearn.manifold.Isomap`

**Verified sklearn 1.5.x parameters:**

```python
sklearn.manifold.Isomap(
    n_neighbors=5,          # int or None, default=5
    radius=None,            # float or None, default=None [added v1.1]
    n_components=2,         # int, default=2
    eigen_solver='auto',    # {'auto', 'arpack', 'dense'}, default='auto'
    tol=0,                  # float, default=0
    max_iter=None,          # int or None, default=None
    path_method='auto',     # {'auto', 'FW', 'D'}, default='auto'
    neighbors_algorithm='auto',  # {'auto', 'brute', 'kd_tree', 'ball_tree'}, default='auto'
    n_jobs=None,            # int or None, default=None
    metric='minkowski',     # str or callable, default='minkowski' [added v0.22]
    p=2,                    # float, default=2 [added v0.22]
    metric_params=None,     # dict or None, default=None [added v0.22]
)
```

**Parameter details:**

| Parameter | Type | Default | Controls |
|-----------|------|---------|---------|
| `n_neighbors` | int or None | 5 | Number of neighbors in the graph. If None, use `radius` instead. |
| `radius` | float or None | None | If set (and n_neighbors=None), use radius-based neighbors. Added in v1.1. |
| `n_components` | int | 2 | Output dimensionality. |
| `eigen_solver` | str | 'auto' | 'arpack' for sparse/large data; 'dense' for small datasets (exact via LAPACK). 'auto' picks based on size. |
| `tol` | float | 0 | Convergence tolerance for 'arpack'. Ignored for 'dense'. |
| `max_iter` | int or None | None | Max iterations for 'arpack'. None = no limit. |
| `path_method` | str | 'auto' | Shortest path: 'FW'=Floyd-Warshall O(N³), 'D'=Dijkstra O(N²k log N). 'auto' picks 'D' for sparse, 'FW' for dense. |
| `neighbors_algorithm` | str | 'auto' | kNN algorithm. 'kd_tree' and 'ball_tree' faster for low D; 'brute' required for some metrics. |
| `n_jobs` | int or None | None | Parallelism. -1 = all cores. Affects kNN step. |
| `metric` | str or callable | 'minkowski' | Distance metric. Any sklearn.metrics.pairwise_distances option. |
| `p` | float | 2 | Power for Minkowski metric. p=1 = Manhattan, p=2 = Euclidean. |
| `metric_params` | dict or None | None | Extra kwargs for custom metrics. |

**Key attributes (post-fit):**
- `embedding_` (n_samples, n_components) — the embedding coordinates
- `kernel_pca_` — the fitted KernelPCA object
- `nbrs_` — the fitted NearestNeighbors object
- `dist_matrix_` — geodesic distance matrix (only populated if `store_dist_matrix=True` is passed, but this attr is accessible)
- `reconstruction_error()` — method returning the reconstruction error (use for n_components selection)

**Standard usage:**
```python
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
iso = Isomap(n_neighbors=10, n_components=2, eigen_solver='arpack', path_method='D')
X_embed = iso.fit_transform(X_scaled)
print(iso.reconstruction_error())  # use elbow in this vs n_components
```

---

### 3.2 `sklearn.manifold.LocallyLinearEmbedding`

**Verified sklearn 1.5.x parameters:**

```python
sklearn.manifold.LocallyLinearEmbedding(
    n_neighbors=5,              # int, default=5
    n_components=2,             # int, default=2
    reg=1e-3,                   # float, default=1e-3
    eigen_solver='auto',        # {'auto', 'arpack', 'dense'}, default='auto'
    tol=1e-6,                   # float, default=1e-6
    max_iter=100,               # int, default=100
    method='standard',          # {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
    hessian_tol=1e-4,           # float, default=1e-4 [only for method='hessian']
    modified_tol=1e-12,         # float, default=1e-12 [only for method='modified']
    neighbors_algorithm='auto', # {'auto', 'brute', 'kd_tree', 'ball_tree'}, default='auto'
    random_state=None,          # int or RandomState, default=None
    n_jobs=None,                # int or None, default=None
)
```

**Parameter details:**

| Parameter | Type | Default | Controls |
|-----------|------|---------|---------|
| `n_neighbors` | int | 5 | Size of local neighborhood. Must be > n_components for all variants. For HLLE, must be > n_components*(n_components+3)/2. |
| `n_components` | int | 2 | Output dimensionality. |
| `reg` | float | 1e-3 | Regularization constant for the local weight matrix (multiplies trace of local covariance). Prevents singular weight matrices. |
| `eigen_solver` | str | 'auto' | 'arpack' uses shift-invert Arnoldi; 'dense' uses full LAPACK. 'auto' picks based on size. |
| `tol` | float | 1e-6 | Arpack convergence tolerance. |
| `max_iter` | int | 100 | Max iterations for arpack. |
| `method` | str | 'standard' | Algorithm variant: 'standard' (LLE), 'modified' (MLLE), 'hessian' (HLLE), 'ltsa' (LTSA). |
| `hessian_tol` | float | 1e-4 | Tolerance only for method='hessian'. Controls numerical rank of Hessian estimator. |
| `modified_tol` | float | 1e-12 | Tolerance only for method='modified'. Controls selection of multiple weights. |
| `neighbors_algorithm` | str | 'auto' | kNN search algorithm. |
| `random_state` | int/RS | None | Seed for arpack initialization. Set for reproducibility. |
| `n_jobs` | int or None | None | Parallelism for kNN computation. |

**n_neighbors requirement by method:**

| method | Minimum n_neighbors |
|--------|-------------------|
| 'standard' | > n_components |
| 'modified' | > n_components |
| 'hessian' | > n_components * (n_components + 3) / 2 |
| 'ltsa' | > n_components |

**Key attributes (post-fit):**
- `embedding_` (n_samples, n_components)
- `reconstruction_error_` — scalar reconstruction error (for n_components selection)
- `nbrs_` — fitted NearestNeighbors object

**Standard usage:**
```python
from sklearn.manifold import LocallyLinearEmbedding

# Standard LLE
lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='standard', random_state=42)
X_embed = lle.fit_transform(X_scaled)

# MLLE (recommended over standard for robustness)
mlle = LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='modified', random_state=42)
X_embed_m = mlle.fit_transform(X_scaled)

# LTSA (often best results)
ltsa = LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='ltsa', random_state=42)
X_embed_lt = ltsa.fit_transform(X_scaled)

# HLLE (for theoretical guarantees, small datasets)
# n_neighbors requirement: > n_components*(n_components+3)/2
# For n_components=2: > 2*5/2 = 5, so n_neighbors >= 6
hlle = LocallyLinearEmbedding(n_neighbors=12, n_components=2, method='hessian', random_state=42)
X_embed_h = hlle.fit_transform(X_scaled)
```

---

### 3.3 `sklearn.manifold.SpectralEmbedding`

**Verified sklearn 1.5.x parameters:**

```python
sklearn.manifold.SpectralEmbedding(
    n_components=2,                 # int, default=2
    affinity='nearest_neighbors',   # str or callable, default='nearest_neighbors'
    gamma=None,                     # float or None, default=None
    random_state=None,              # int or RandomState, default=None
    eigen_solver=None,              # {'arpack', 'lobpcg', 'amg'}, default=None
    eigen_tol='auto',               # float or 'auto', default='auto'
    n_neighbors=None,               # int or None, default=None (uses max(n_samples/10, 1))
    n_jobs=None,                    # int or None, default=None
)
```

**Parameter details:**

| Parameter | Type | Default | Controls |
|-----------|------|---------|---------|
| `n_components` | int | 2 | Output dimensionality. |
| `affinity` | str or callable | 'nearest_neighbors' | How to build affinity matrix. 'rbf' uses Gaussian kernel. 'precomputed' lets you pass your own matrix. 'precomputed_nearest_neighbors' takes sparse kNN graph. |
| `gamma` | float or None | None | Kernel bandwidth for 'rbf'. None → 1/n_features. Controls the "reach" of each point's influence. |
| `random_state` | int/RS | None | For lobpcg/amg eigensolver initialization. |
| `eigen_solver` | str or None | None | None defaults to 'arpack'. 'amg' requires `pip install pyamg`; fast for large sparse problems. 'lobpcg' for medium dense problems. |
| `eigen_tol` | float or 'auto' | 'auto' | Convergence tolerance. 'auto' means 0.0 for arpack, None for lobpcg/amg. |
| `n_neighbors` | int or None | None | Number of neighbors for 'nearest_neighbors' affinity. None → max(n_samples/10, 1). |
| `n_jobs` | int or None | None | Parallelism for kNN search. |

**Key attributes (post-fit):**
- `embedding_` (n_samples, n_components)
- `affinity_matrix_` (n_samples, n_samples) — the computed affinity/adjacency matrix

**Standard usage:**
```python
from sklearn.manifold import SpectralEmbedding

# With nearest neighbors affinity (default, Laplacian Eigenmaps)
se = SpectralEmbedding(n_components=2, n_neighbors=10, affinity='nearest_neighbors', random_state=42)
X_embed = se.fit_transform(X_scaled)

# With RBF affinity (provides more continuous affinity)
se_rbf = SpectralEmbedding(n_components=2, affinity='rbf', gamma=0.1, random_state=42)
X_embed_rbf = se_rbf.fit_transform(X_scaled)

# With precomputed affinity matrix
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
A = rbf_kernel(X_scaled, gamma=0.1)
se_pre = SpectralEmbedding(n_components=2, affinity='precomputed', random_state=42)
X_embed_pre = se_pre.fit_transform(A)
```

---

### 3.4 `sklearn.manifold.MDS`

**CRITICAL: This class has significant parameter changes across versions. Always check your version.**

**sklearn 1.5.x parameters (the course baseline):**

```python
sklearn.manifold.MDS(
    n_components=2,             # int, default=2
    metric=True,                # bool, default=True  [NOTE: RENAMED to metric_mds in v1.8!]
    n_init=4,                   # int, default=4       [NOTE: changes to 1 in v1.9!]
    max_iter=300,               # int, default=300
    verbose=0,                  # int, default=0
    eps=1e-3,                   # float, default=1e-3  [NOTE: changes to 1e-6 in v1.9!]
    n_jobs=None,                # int or None, default=None
    random_state=None,          # int or RandomState, default=None
    dissimilarity='euclidean',  # str, default='euclidean' [NOTE: RENAMED to metric in v1.8!]
    normalized_stress='auto',   # bool or 'auto', default='auto' [added v1.2, default changed v1.4]
)
```

**Parameter details (1.5.x):**

| Parameter | Type | Default | Controls |
|-----------|------|---------|---------|
| `n_components` | int | 2 | Output dimensionality. |
| `metric` | bool | True | True=metric MDS (SMACOF on raw distances), False=non-metric MDS (SMACOF on isotonic regression of distances). **Renamed to `metric_mds` in v1.8.** |
| `n_init` | int | 4 | Number of SMACOF runs with different random initializations; best (lowest stress) result returned. **Changes to 1 in v1.9.** |
| `max_iter` | int | 300 | Maximum SMACOF iterations per run. |
| `verbose` | int | 0 | 0=silent, higher values print iteration info. |
| `eps` | float | 1e-3 | Convergence tolerance (relative stress change). **Changes to 1e-6 in v1.9.** |
| `n_jobs` | int or None | None | Parallel jobs for multiple n_init runs. |
| `random_state` | int/RS | None | Initialization seed. |
| `dissimilarity` | str | 'euclidean' | 'euclidean' computes pairwise Euclidean distances from X. 'precomputed' means X is already a dissimilarity matrix. **Renamed to `metric` in v1.8; deprecated in v1.8, removed in v1.10.** |
| `normalized_stress` | bool or 'auto' | 'auto' | 'auto' returns normalized stress for non-metric, raw stress for metric MDS. |

**Key attributes (post-fit):**
- `embedding_` (n_samples, n_components) — low-dimensional coordinates
- `stress_` — final stress value (use to compare runs and configurations)
- `dissimilarity_matrix_` — the computed dissimilarity matrix
- `n_iter_` — number of iterations of the best run

**Stress interpretation (Kruskal's benchmarks):**
- 0.000: Perfect fit
- 0.025: Excellent
- 0.050: Good
- 0.100: Fair
- 0.200: Poor (embedding unreliable)
- > 0.200: Very poor

**Standard usage:**
```python
from sklearn.manifold import MDS

# Metric MDS (sklearn 1.5.x)
mds = MDS(n_components=2, metric=True, n_init=4, random_state=42,
          dissimilarity='euclidean', normalized_stress='auto')
X_embed = mds.fit_transform(X_scaled)
print(f"Stress: {mds.stress_:.4f}")

# Non-metric MDS (sklearn 1.5.x)
nmds = MDS(n_components=2, metric=False, n_init=4, random_state=42,
           dissimilarity='euclidean', normalized_stress=True)
X_embed_nm = nmds.fit_transform(X_scaled)

# With precomputed distance matrix
from sklearn.metrics.pairwise import euclidean_distances
D = euclidean_distances(X_scaled)
mds_pre = MDS(n_components=2, metric=True, dissimilarity='precomputed', random_state=42)
X_embed_pre = mds_pre.fit_transform(D)

# sklearn 1.8+ (metric parameter renamed):
# mds_v18 = MDS(n_components=2, metric_mds=True, metric='euclidean', random_state=42)
```

---

## SECTION 4: SPECIALIZED PYTHON PACKAGES

### 4.1 PHATE

| Attribute | Value |
|-----------|-------|
| **Package name** | `phate` |
| **PyPI name** | `phate` |
| **Version (mid-2026)** | 1.0.11 |
| **Install** | `pip install phate` |
| **Key import** | `import phate` |
| **License** | GNU GPL v2 |
| **Maintenance** | Active (KrishnaswamyLab at Yale) |
| **GitHub** | https://github.com/KrishnaswamyLab/PHATE |
| **Docs** | https://phate.readthedocs.io/ |

**What PHATE does better than sklearn:**
- sklearn has no diffusion-based trajectory-preserving embedding
- PHATE specifically preserves both local clusters AND global progressions/branches — critical for developmental biology
- Automatic t-selection via Von Neumann entropy
- Accepts AnnData objects (common scRNA-seq format)
- Has built-in scatter plot visualization (`phate.plot.scatter2d()`, `phate.plot.scatter3d()`)
- Integrates with the broader Krishnaswamy Lab ecosystem (MAGIC, MELD, Palantir)
- R package `phateR` also available on CRAN

**Key PHATE parameters:**

```python
phate.PHATE(
    n_components=2,     # int, default=2 — output dimensionality
    knn=5,              # int, default=5 — number of nearest neighbors for kernel
    decay=40,           # int, default=40 — tail decay rate of the kernel (alpha decay)
    t='auto',           # int or 'auto', default='auto' — diffusion time steps
    gamma=1,            # float, -1 to 1, default=1 — transformation constant for MDS metric
    n_landmark=2000,    # int, default=2000 — number of landmarks for fast computation (None = exact)
    mds_solver='sgd',   # 'sgd' or 'smacof', default='sgd' — SGD is faster; SMACOF more optimal
    knn_dist='euclidean',  # str, default='euclidean' — kNN distance metric
    mds_dist='euclidean',  # str, default='euclidean' — MDS distance metric
    n_jobs=1,           # int, default=1 — parallelism
    random_state=None,  # int or None, default=None
    verbose=1,          # int, default=1
)
```

**Parameter notes:**
- `knn` (not `n_neighbors`): Equivalent to Isomap's n_neighbors; 5–15 is typical
- `decay`: Controls the alpha-decay kernel; higher values make kernel sharper; 40 is default for biology
- `t='auto'`: Uses knee-point of Von Neumann entropy of the diffusion operator to select t automatically
- `gamma=1`: Log transform; gamma=-1 → sqrt transform; gamma=0 → no transform
- `n_landmark`: Enables Nyström approximation for large N; set None for exact but slow computation

**Basic usage:**
```python
import phate
import numpy as np

# Basic
phate_op = phate.PHATE(n_components=2, knn=5, t='auto', random_state=42)
X_phate = phate_op.fit_transform(data)  # data: numpy array, scipy sparse, DataFrame, or AnnData

# Visualization
phate.plot.scatter2d(phate_op, c=cell_labels, figsize=(8,6), s=5)

# 3D embedding
phate_3d = phate.PHATE(n_components=3)
X_phate_3d = phate_3d.fit_transform(data)
phate.plot.scatter3d(phate_3d, c=cell_labels, angle=(135, 20))

# Large datasets (use landmarks)
phate_large = phate.PHATE(n_components=2, n_landmark=2000, mds_solver='sgd')
X_phate_large = phate_large.fit_transform(large_data)
```

---

### 4.2 pydiffmap

| Attribute | Value |
|-----------|-------|
| **Package name** | `pydiffmap` |
| **PyPI name** | `pydiffmap` |
| **Version (mid-2026)** | 0.2.0.1 (stable but low activity) |
| **Install** | `pip install pydiffmap` |
| **Key import** | `from pydiffmap import diffusion_map as dm` |
| **License** | MIT |
| **Maintenance** | LOW — no new PyPI releases for 12+ months; consider as reference implementation |
| **GitHub** | https://github.com/DiffusionMapsAcademics/pyDiffMap |
| **Docs** | https://pydiffmap.readthedocs.io/ |

**What pydiffmap does better than sklearn:**
- sklearn has NO native diffusion maps implementation
- Implements the full Coifman-Lafon framework including the alpha normalization parameter
- Can auto-select epsilon (bandwidth) using Berry-Harlim-Giannakis algorithm
- sklearn integration via `from_sklearn()` classmethod

**Key pydiffmap parameters:**
```python
from pydiffmap import diffusion_map as dm

mydmap = dm.DiffusionMap.from_sklearn(
    n_evecs=2,          # int — number of eigenvectors (= n_components)
    epsilon='bgh',      # float or 'bgh', default='bgh' — bandwidth; 'bgh' = auto-select
    alpha=0.5,          # float [0,1], default=0.5 — normalization: 0=graph Laplacian, 1=Fokker-Planck
    k=64,               # int — number of nearest neighbors for sparse kernel
)
mydmap.fit_transform(X)
```

**Alpha parameter guide:**
- `alpha=0`: Sensitive to sampling density (like Laplacian Eigenmaps)
- `alpha=0.5`: Estimates Laplace-Beltrami operator; geometry intrinsic to manifold
- `alpha=1`: Removes sampling density effects; best for intrinsic geometry

---

### 4.3 scanpy (for single-cell biology)

| Attribute | Value |
|-----------|-------|
| **Package name** | `scanpy` |
| **PyPI name** | `scanpy` |
| **Version (mid-2026)** | ~1.10+ |
| **Install** | `pip install scanpy` |
| **Key import** | `import scanpy as sc` |
| **License** | BSD-3-Clause |
| **Maintenance** | Actively maintained (scverse consortium) |

**Relevant manifold methods in scanpy:**
```python
import scanpy as sc

adata = sc.read_h5ad('data.h5ad')
sc.pp.pca(adata)                          # PCA preprocessing
sc.pp.neighbors(adata, n_neighbors=15)    # shared-NN graph
sc.tl.umap(adata)                         # UMAP
sc.tl.diffmap(adata, n_comps=15)          # Diffusion Maps (Coifman-Lafon based)
# Results in adata.obsm['X_diffmap']

# For trajectory analysis using diffusion pseudotime
sc.tl.dpt(adata, n_dcs=10)               # Diffusion pseudotime (Haghverdi 2016)
```

---

### 4.4 sklearn (baseline manifold module)

| Attribute | Value |
|-----------|-------|
| **Package name** | `scikit-learn` |
| **PyPI name** | `scikit-learn` |
| **Version (mid-2026)** | 1.9.x |
| **Install** | `pip install scikit-learn` |
| **License** | BSD-3-Clause |
| **Maintenance** | Actively maintained |

**All manifold learning classes in sklearn.manifold:**
- `Isomap`
- `LocallyLinearEmbedding` (covers standard LLE, MLLE, HLLE, LTSA via `method=` parameter)
- `SpectralEmbedding`
- `MDS`
- `ClassicalMDS` (new in 1.9 — analytical solution without SMACOF)
- `TSNE`
- `trustworthiness` (function for evaluation)

---

## SECTION 5: REAL-WORLD APPLICATIONS

### Application 1: Robot Arm Configuration Space (Isomap — Original Demo)

**Domain:** Robotics / Configuration Space Analysis
**Problem:** A robot arm has multiple joints, each with an angle — the "configuration space" is high-dimensional. Images of the robot at different configurations form a high-dimensional dataset (pixel values), but the true degrees of freedom are just the joint angles (low-dimensional). The challenge is to recover the intrinsic configuration manifold from images.
**Why Isomap:** Isomap's geodesic distances correctly reflect the arm's path of motion — to move from one pose to another, you must rotate through intermediate poses. Euclidean distance between images would incorrectly suggest shortcuts not achievable in the physical mechanism.
**Result:** The 2000 Science paper demonstrated that Isomap recovered a 2D embedding that corresponded to two joint angles — pure geometric insight extracted from raw pixel data with no supervision.
**Reference:** Tenenbaum, de Silva, Langford (2000), Science 290:2319-2323

---

### Application 2: Face Pose Estimation Under Varying Illumination (Isomap)

**Domain:** Computer Vision / Face Recognition
**Problem:** Face images vary in high-dimensional pixel space due to three underlying factors: left-right pose, up-down pose, and illumination. All three form a smooth manifold in pixel space, but Euclidean distances don't reflect the true geometry of these transformations.
**Why Isomap:** By using geodesic distances, Isomap recovers the underlying 3-variable structure. The embedding axes correspond directly to pose-left/right, pose-up/down, and lighting direction — interpretable latent factors with no labels.
**Application today:** Head pose estimation for driver monitoring systems (drowsiness detection), AR/VR avatar tracking, gaze estimation for accessibility tools.
**Reference:** Original Tenenbaum 2000 paper; extended work by Supervised Locality Discriminant Manifold Learning (Supervised Isomap variants, 2014)

---

### Application 3: Handwriting Recognition — MNIST Digit Manifold (LLE)

**Domain:** Computer Vision / Pattern Recognition
**Problem:** Handwritten digits lie on a low-dimensional manifold in pixel space — a "3" varies continuously in writing style but remains a "3." t-SNE separates clusters visually, but LLE and Isomap were the original demonstrations of manifold structure in handwriting.
**Why LLE:** LLE preserves local linear reconstruction — nearby writing styles have similar linear relationships, which is the physical reality of human handwriting variation. LLE's original paper (Roweis & Saul 2000) used handwriting as one of two key demos.
**Application today:** As a preprocessing step before classification on MNIST and similar datasets; as a benchmark for comparing manifold methods (sklearn's official comparison plot uses the digits dataset).
**Reference:** Roweis & Saul (2000), Science 290:2323-2326; also sklearn `plot_lle_digits.py` example

---

### Application 4: Neural Manifold Hypothesis — Place Cells and Grid Cells (Various)

**Domain:** Computational Neuroscience
**Problem:** The hippocampus contains "place cells" that fire when an animal is at a specific location. How is spatial information encoded across thousands of neurons? If you record from 1000 neurons simultaneously, each state is a 1000-dimensional vector. The "neural manifold hypothesis" asserts this data lies on a low-dimensional manifold.
**Why Manifold Methods:**
- Grid cells encode space in a 2D toroidal manifold (proven via TDA and Isomap-like methods)
- Isomap was shown to achieve more accurate decoding of rat position in hippocampal CA1 than linear methods
- Laplacian Eigenmaps / spectral methods are used to extract "population trajectories" that decode behavioral state
**Non-obvious insight:** The manifold dimensionality of prefrontal cortex activity during complex tasks (planning, working memory) is much higher than during simple sensorimotor tasks — suggesting the brain dynamically expands its representational complexity.
**Key papers:** "A unifying perspective on neural manifolds and circuits for cognition" (2024, PMC); "Distinct manifold encoding of navigational information in subiculum and hippocampus" (Science Advances 2024)
**Application today:** Brain-machine interfaces (BMIs) — UMAP/Isomap embeddings of neural data used for real-time cursor control by paralyzed patients

---

### Application 5: Single-Cell Genomics — Cell Differentiation Trajectories (PHATE / Diffusion Maps)

**Domain:** Computational Biology / Genomics
**Problem:** In a developing embryo, cells start as pluripotent stem cells and differentiate into hundreds of cell types. Single-cell RNA sequencing (scRNA-seq) measures the expression of ~20,000 genes in each of tens of thousands of cells. The challenge: visualize the continuous spectrum of differentiation states, including branching trajectories, without losing the biological narrative.
**Why PHATE over t-SNE/UMAP:**
- t-SNE creates "islands" of clusters — loses the developmental trajectory structure
- UMAP preserves some global structure but still struggles with multi-scale branching
- PHATE was specifically designed to reveal continuous progressions and branch points
**Real example:** Moon et al. (2019) applied PHATE to human germ-layer differentiation data, revealing biologically meaningful branching that matched known developmental biology and discovered novel intermediate states.
**Industry use:** 10x Genomics analysis pipelines; Illumina BaseSpace; Broad Institute TERRA platform for cell atlas projects (Human Cell Atlas, Brain Cell Atlas)
**Scale:** Multiscale PHATE applied to 54 million cells from 168 COVID-19 patients (2022, Nature Biotechnology)

---

### Application 6: Protein Folding Energy Landscape (Isomap / Diffusion Maps)

**Domain:** Structural Biology / Drug Discovery
**Problem:** A protein's behavior is determined by its 3D folding. Molecular dynamics simulations generate millions of protein conformations (each described by thousands of atomic coordinates). Understanding the "folding funnel" — the energy landscape from unfolded to folded state — requires finding the underlying low-dimensional reaction coordinate.
**Why Manifold Learning:**
- The folding manifold is genuinely nonlinear: a protein doesn't fold in a straight line through conformation space
- Isomap/diffusion maps find the reaction coordinate that correctly separates the transition-state ensemble from native and denatured states
- Das et al. (2006, PNAS) applied nonlinear DR to characterize free energy landscapes of protein folding using Isomap
**Practical challenge:** N > 100,000 conformations makes standard Isomap impractical — ScIMAP uses a landmark subset
**Application today:** Used in fragment-based drug discovery to understand conformational sampling of drug target pockets; in antibody design to characterize the diversity of the antibody conformational landscape

---

### Application 7: Climate Science — El Niño Pattern Discovery (Isomap / Spectral Methods)

**Domain:** Atmospheric Science / Oceanography
**Problem:** Global climate patterns (sea surface temperature, atmospheric pressure, precipitation) involve complex nonlinear interactions. El Niño/Southern Oscillation (ENSO) patterns occupy a low-dimensional subspace of the high-dimensional climate state space, but the relationship is nonlinear.
**Why Manifold Learning:**
- Traditional EOF/PCA analysis captures only linear variance modes
- Isomap isometric mapping applied to tropical Pacific SST and thermocline depth data reveals nonlinear ENSO variability patterns
- Diffusion maps (via their spectral decomposition) have been applied to climate dynamics as Koopman operator approximations
**Published work:** "Nonlinear Dimensionality Reduction Methods in Climate Data Analysis" (arXiv:0901.0537); "Beyond PCA: Additional Dimension Reduction Techniques" in Journal of Climate (2024, Vol. 37, Issue 5)
**Application today:** National weather modeling centers use nonlinear DR to identify regime transitions; long-range seasonal forecast verification

---

### Application 8: Speech Manifold — Speaker Identification and Adversarial Detection (LLE / Isomap)

**Domain:** Audio Signal Processing / Security
**Problem 1 (Speaker ID):** Speech feature vectors (MFCCs) from the same speaker lie on a speaker-specific manifold. Isomap was shown to give the highest accuracy for very low-dimensional speaker identification features, outperforming MFCCs and PCA-transformed features for small d.
**Problem 2 (Adversarial detection, 2024):** Deep learning speech recognition systems are vulnerable to adversarial attacks — imperceptible perturbations that fool ASR models. A 2024 paper (MDPI Mathematics) proposes using manifold learning to detect adversarial speech before it reaches the recognition model: legitimate speech occupies a well-defined manifold, while adversarial examples lie off it.
**Why LLE:** Speech is produced by a small number of articulatory parameters (jaw position, tongue height/backness, lip rounding) — the "articulatory manifold." LLE preserves local relationships between MFCC feature vectors that correspond to continuous articulatory movements.
**Practical relevance:** Voice anti-spoofing for banking authentication systems; adversarial robustness for safety-critical ASR (medical dictation, air traffic control)

---

## SECTION 6: HYPERPARAMETER TUNING GUIDE

### 6.1 Isomap Hyperparameters

#### `n_neighbors` (most critical parameter)

| Setting | Effect | When to Use |
|---------|--------|------------|
| Too small (< 5) | Sparse graph, many disconnected components, NaN in embedding | Never; always check for disconnected graph |
| Small (5–10) | Captures only very local structure; accurate geodesics for high-curvature manifolds | When manifold has high curvature or many holes |
| Moderate (10–20) | Good balance for most datasets | Default starting point |
| Large (20–50) | Smoother geodesics; risks "short-circuit" errors if data is noisy | When data is very dense and noise-free |
| Too large | Short-circuit errors: edges jump across the manifold; embedding is wrong | Avoid if reconstruction_error() stops decreasing |

**Heuristic:** Start with sqrt(N) as a rough guide. Check `reconstruction_error()` — if it's high for small k, the graph may be disconnected.

**Detecting short-circuit errors:** Plot the sorted eigenvalues of the geodesic distance matrix. Short circuits create "phantom" low eigenvalues. If your embedding clusters make no geometric sense, reduce k.

**Detecting disconnected graph:** `iso.nbrs_.kneighbors_graph()` should have a single connected component. Use `scipy.sparse.csgraph.connected_components()` to check.

**Optuna search space:**
```python
import optuna
def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 5, 50)
    n_components = trial.suggest_int('n_components', 2, 10)
    iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    iso.fit(X_scaled)
    return iso.reconstruction_error()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

#### `n_components`

**Selection method:** Plot reconstruction error vs. n_components. Look for an "elbow" — the point where additional components yield diminishing returns. The elbow indicates the intrinsic dimensionality of the manifold.

```python
errors = []
for d in range(1, 15):
    iso = Isomap(n_neighbors=10, n_components=d)
    iso.fit(X_scaled)
    errors.append(iso.reconstruction_error())
# Plot errors vs d; find elbow
```

#### `eigen_solver`

- `'auto'`: Let sklearn decide (picks 'arpack' for N > ~200)
- `'arpack'`: Use for large sparse datasets; requires tol and max_iter tuning
- `'dense'`: Use for small datasets (N < 1000) where exact solution needed

---

### 6.2 LLE / MLLE / HLLE / LTSA Hyperparameters

#### `n_neighbors`

| Setting | Effect |
|---------|--------|
| Too small | Singular weight matrices; sklearn raises LinAlgError |
| 5–15 | Typical range for 2D output |
| 15–30 | For 3D+ output or noisy data |
| Constraint: HLLE | Must satisfy n_neighbors > n_components*(n_components+3)/2 |

**Common error:** `LinAlgWarning: Matrix is exactly singular` — caused by duplicate points or too-small neighborhoods. Fix: increase n_neighbors, or deduplicate data, or use `reg` parameter.

#### `reg` (regularization constant)

- Default 1e-3 works for most cases
- If you get singular matrix warnings: increase to 1e-2 or 1e-1
- Too large: over-regularizes, distorts local geometry
- **Optuna range:** `trial.suggest_float('reg', 1e-4, 1e-1, log=True)`

#### `method` selection guide

| method | Use when |
|--------|---------|
| 'standard' | Quick prototyping; small datasets |
| 'modified' | Production; more stable than standard; same speed |
| 'hessian' | Small datasets (< 2000 pts); need theoretical guarantees |
| 'ltsa' | Often best results; good for complex manifolds |

**Recommendation:** Default to `method='modified'` over `method='standard'` unless you have specific reasons. MLLE fixes LLE's rank-deficiency problem with minimal extra cost.

**Optuna search space:**
```python
def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 5, 30)
    reg = trial.suggest_float('reg', 1e-4, 1e-1, log=True)
    method = trial.suggest_categorical('method', ['standard', 'modified', 'ltsa'])
    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, reg=reg, method=method, random_state=42
    )
    lle.fit(X_scaled)
    return lle.reconstruction_error_  # note: attribute, not method
```

---

### 6.3 SpectralEmbedding Hyperparameters

#### `affinity`

| Option | When to use |
|--------|-------------|
| 'nearest_neighbors' | Default; good for most cases; sensitive to n_neighbors |
| 'rbf' | When you want soft affinities (each point has "influence radius"); needs gamma tuning |
| 'precomputed' | When you have a custom similarity matrix (e.g., cosine similarity, string kernels) |

#### `n_neighbors` (for 'nearest_neighbors' affinity)

- Default: max(n_samples/10, 1) — can be very large for small datasets
- Typical range: 5–30
- Too small: disconnected graph, embedding only reflects local structure
- Too large: affinity matrix becomes dense, loses discriminative structure

#### `gamma` (for 'rbf' affinity)

- Controls the "influence radius": high gamma = narrow kernel = very local influence
- Rule of thumb: gamma = 1 / (2 * σ²) where σ is estimated as median pairwise distance
- **Optuna range:** `trial.suggest_float('gamma', 1e-3, 10.0, log=True)`

```python
# Auto-select gamma based on median pairwise distance
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
D = euclidean_distances(X_scaled)
sigma = np.median(D)
gamma_auto = 1.0 / (2 * sigma**2)
```

---

### 6.4 MDS Hyperparameters

#### `n_init` (number of SMACOF initializations)

- Default in 1.5.x: 4 (changed to 1 in 1.9)
- SMACOF is sensitive to initialization; multiple runs reduce risk of local minima
- Increase to 8–16 for final production runs
- Parallel with `n_jobs=-1`

#### `max_iter`

- Default 300 is usually enough
- Increase to 1000 if `mds.stress_` is still decreasing noticeably between runs

#### `metric` (True vs False)

- `metric=True` (Metric MDS): preserves actual distance values; sensitive to outliers
- `metric=False` (Non-metric MDS): only preserves ordering; more robust; preferred for perceptual data
- For biological distance matrices: start with non-metric

#### Choosing n_components for MDS

Use the stress elbow method:
```python
stresses = []
for d in range(1, 10):
    mds = MDS(n_components=d, metric=False, n_init=4, random_state=42, dissimilarity='euclidean')
    mds.fit_transform(X_scaled)
    stresses.append(mds.stress_)
# Plot stresses vs d; elbow = optimal d
```

---

### 6.5 PHATE Hyperparameters

#### `knn`

- Small (3–5): very local; good for data with sharp transitions
- Default (5): good general-purpose
- Large (15–30): smooths out noise; good for noisy scRNA-seq data
- Rule: similar logic to n_neighbors in other methods

#### `t` (diffusion time)

- 'auto': recommended; uses Von Neumann entropy knee-point
- Manual values: 10–100 for most biological data
- Too small: only local structure; trajectory collapsed
- Too large: all structure washed out; embedding becomes uniform

#### `decay`

- Higher values: sharper kernel; more sensitive to local density
- Default 40 is well-calibrated for scRNA-seq
- For mass cytometry (CyTOF): try decay=10–20

---

## SECTION 7: EXPERT PRACTITIONER CHECKLIST

### Before Fitting: Data Preparation

1. **Scale your features.** All manifold methods use pairwise distances. If features have different units (e.g., gene expression + clinical measurements), normalize each to unit variance. Use `StandardScaler`. Failure to scale is the most common error.

2. **Check for duplicate points.** LLE will produce singular matrices if exact duplicates exist. `X_scaled[~np.all(np.diff(np.sort(X_scaled, axis=0), axis=0) == 0, axis=1)]` or use `np.unique()`.

3. **Check for NaN/Inf values.** Any NaN in the distance matrix cascades to incorrect embeddings. `np.any(np.isnan(X))` must be False.

4. **Estimate intrinsic dimensionality first.** Don't just embed to 2D without checking if 2D is sufficient. Use:
   ```python
   # Reconstruction error elbow
   from sklearn.manifold import Isomap
   errors = [Isomap(n_neighbors=10, n_components=d).fit(X_scaled).reconstruction_error() for d in range(1,10)]
   # Also consider: sklearn.decomposition.PCA to see how many PCs capture 90% variance
   ```

5. **Consider data size.** For N > 5000, all manifold methods (especially Isomap, LLE) are slow O(N²–N³). Options:
   - Use PHATE with n_landmark parameter
   - Subsample, embed, then use out-of-sample extension (Nyström)
   - Switch to UMAP for N > 10,000 (not in scope, but the alternative)

6. **Understand your manifold's expected topology.** Is it a single connected curve? A 2D sheet? Has holes? Different algorithms handle different topologies:
   - Isomap: requires convex (no holes) manifold; fails on torus
   - LLE/LTSA: more flexible topology
   - SpectralEmbedding: handles disconnected components partially

### During Fitting: What to Watch

7. **Watch for disconnected graph warning (Isomap):** If sklearn emits a warning about disconnected components, your k is too small. Increase n_neighbors.

8. **Watch for singular matrix warnings (LLE):** `LinAlgWarning: Matrix is exactly singular` means neighborhood is too small or data has duplicates. Increase n_neighbors or reg.

9. **Use `random_state` for reproducibility.** LLE with 'arpack' solver has random initialization. Set `random_state=42` always for reproducible results.

10. **MDS: run multiple initializations.** SMACOF finds local minima. Set `n_init=8` for final runs. Compare stress values across runs — if they vary greatly, SMACOF is struggling.

### After Fitting: Validation

11. **Compute trustworthiness score.** Values above 0.9 indicate good local structure preservation.
    ```python
    from sklearn.manifold import trustworthiness
    t = trustworthiness(X_scaled, X_embed, n_neighbors=5)
    print(f"Trustworthiness: {t:.3f}")  # > 0.9 = good
    ```

12. **Visual sanity check.** Does the embedding match known structure? If you have labels, do they form coherent clusters? For the Swiss Roll: does the color gradient (position along roll) appear as a 2D strip?

13. **Check MDS stress value.** Should be < 0.1 for a reliable interpretation. If > 0.2, the embedding is not trustworthy.

14. **Compare multiple methods.** Run Isomap, MLLE/LTSA, and SpectralEmbedding on the same data. If they agree, the structure is robust. Disagreement signals either: (a) the data doesn't satisfy one method's assumptions, or (b) there's no true low-dimensional structure.

15. **Out-of-sample extension (if needed).** None of the sklearn manifold methods (except Isomap via `transform()`) support out-of-sample extension natively:
    - `Isomap.transform(X_new)`: uses the fitted kNN graph to embed new points
    - LLE, SpectralEmbedding, MDS: no `transform()` for new data — must refit from scratch

### Common Pitfalls (with Diagnosis)

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Features not scaled | Embedding dominated by highest-variance feature | StandardScaler before fitting |
| k too small (Isomap) | Disconnected graph warning; NaN in embedding | Increase n_neighbors |
| k too large (Isomap) | Short-circuit errors; embedding compresses manifold | Decrease n_neighbors |
| Duplicate points (LLE) | `LinAlgWarning: Matrix is exactly singular` | Remove duplicates; increase reg |
| HLLE n_neighbors too small | `ValueError: n_neighbors must be >...` | n_neighbors > n_components*(n_components+3)/2 |
| SMACOF local minimum (MDS) | Final stress high; different runs give different shapes | Increase n_init; use `init='classical_mds'` (1.9+) |
| Embedding folds back on itself | Visual: cluster appears twice; trustworthiness < 0.85 | Try different algorithm; increase n_neighbors |
| Wrong n_components | Information loss visible; reconstruction error elbow not at chosen d | Use elbow method; try 3D and project |

---

## SECTION 8: EVALUATION METRICS

### 8.1 Trustworthiness (local precision)

**What it measures:** Are the k nearest neighbors in the embedding actually close in the original space? Penalizes false neighbors — points that appear close in embedding but are far in original space.

**Interpretation:** Like "precision" for the neighborhood. A high trustworthiness score means the embedding doesn't introduce fake proximity.

**Formula:** T(k) = 1 - [2/(nk(2n-3k-1))] × Σ_i Σ_{j ∈ U_k(i)} (r(i,j) - k)

where U_k(i) are points in the k-NN of the embedding that are NOT in the k-NN of the original space, and r(i,j) is the rank of x_j in original space with respect to x_i.

**sklearn implementation (verified 1.5.x):**
```python
from sklearn.manifold import trustworthiness

# Basic usage
t = trustworthiness(X_original, X_embedded, n_neighbors=5)
print(f"Trustworthiness: {t:.4f}")  # range [0, 1], higher is better

# Vary k to check across scales
for k in [5, 10, 20, 50]:
    t = trustworthiness(X_original, X_embedded, n_neighbors=k)
    print(f"k={k}: T={t:.4f}")
```

**Benchmarks:**
- > 0.95: Excellent
- 0.90–0.95: Good
- 0.85–0.90: Acceptable
- < 0.85: Poor; consider different parameters or algorithm

---

### 8.2 Continuity (local recall)

**What it measures:** Are the true k nearest neighbors in original space preserved in the embedding? Penalizes when true neighbors are "dropped" from the embedding.

**Interpretation:** Like "recall" for the neighborhood. High continuity means you didn't lose real nearby neighbors.

**Formula:** C(k) = 1 - [2/(nk(2n-3k-1))] × Σ_i Σ_{j ∈ V_k(i)} (r̂(i,j) - k)

where V_k(i) are points in the k-NN of the original space that are NOT in the k-NN of the embedding.

**sklearn: no built-in.** Must implement manually or use external libraries:
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def continuity(X, X_embedded, k=5):
    n = X.shape[0]
    # k-NN in original space
    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, ind_orig = nbrs_orig.kneighbors(X)
    ind_orig = ind_orig[:, 1:]  # exclude self
    
    # k-NN in embedded space
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(X_embedded)
    _, ind_emb = nbrs_emb.kneighbors(X_embedded)
    ind_emb = ind_emb[:, 1:]  # exclude self
    
    total = 0.0
    for i in range(n):
        orig_set = set(ind_orig[i])
        emb_set = set(ind_emb[i])
        missing = orig_set - emb_set  # in original k-NN but not in embedding k-NN
        for j in missing:
            # rank of j in embedding space
            rank = np.where(ind_emb[i] == j)[0]
            if len(rank) == 0:
                # j not in any embedding neighbor; use position in sorted embedding distances
                dists = np.linalg.norm(X_embedded - X_embedded[i], axis=1)
                rank = np.argsort(dists)[k:]  # position beyond k-NN
                rank_j = np.where(rank == j)[0]
                if len(rank_j) > 0:
                    total += rank_j[0] + 1  # 1-indexed rank beyond k
    
    normalizer = 2 / (n * k * (2*n - 3*k - 1))
    return 1 - normalizer * total
```

---

### 8.3 Reconstruction Error (Isomap / LLE)

**What it measures:** How well can we reconstruct the original data from the embedding? Different from trustworthiness — this is about global accuracy, not just neighborhood structure.

**For Isomap:** `Isomap.reconstruction_error()` returns the residual variance = 1 - R²(D_geo, D_embed) where D_geo is the geodesic distance matrix and D_embed is the Euclidean distance in the embedding. Lower is better.

**For LLE:** `LocallyLinearEmbedding.reconstruction_error_` attribute gives the objective function value (sum of squared reconstruction residuals). Lower is better.

```python
# Isomap reconstruction error across n_components
from sklearn.manifold import Isomap
errors = []
for d in range(1, 15):
    iso = Isomap(n_neighbors=10, n_components=d)
    iso.fit(X_scaled)
    errors.append(iso.reconstruction_error())
# Plot and find elbow
```

---

### 8.4 MDS Stress (Kruskal's Stress-1)

**What it measures:** For MDS, how well do embedding distances match original distances (or their ordering for non-metric MDS)?

**Raw stress:** Σ (d̂_ij - d_ij)² — total squared discrepancy
**Stress-1 (normalized):** √[Σ(d̂_ij - d_ij)² / Σ d_ij²]

```python
mds = MDS(n_components=2, metric=False, n_init=4, random_state=42, normalized_stress=True)
mds.fit_transform(X_scaled)
print(f"Stress-1: {mds.stress_:.4f}")  # for non-metric, this is normalized
```

**Kruskal benchmarks:**
- < 0.025: Excellent
- 0.025–0.05: Good
- 0.05–0.10: Fair (acceptable for visualization)
- 0.10–0.20: Poor
- > 0.20: Unreliable

---

### 8.5 Residual Variance (Isomap-specific)

**What it measures:** How much of the variance in original geodesic distances is NOT captured by the embedding. The inverse of R² between geodesic distances and embedding distances.

**Used for:** Choosing n_neighbors (k) and n_components. Plot residual variance vs k — look for a minimum (optimal k) or an elbow.

```python
# Select optimal n_neighbors via residual variance
residual_variances = []
for k in range(5, 50, 5):
    iso = Isomap(n_neighbors=k, n_components=2)
    iso.fit(X_scaled)
    residual_variances.append(iso.reconstruction_error())
# Optimal k has minimum residual variance
```

---

### 8.6 Intrinsic Dimensionality Estimation

**Before choosing n_components, estimate the manifold's true dimensionality:**

```python
# Method 1: Isomap elbow
from sklearn.manifold import Isomap
errors = [Isomap(n_neighbors=10, n_components=d).fit(X_scaled).reconstruction_error() for d in range(1, 20)]
# Find elbow in errors vs d

# Method 2: PCA variance explained (linear approximation)
from sklearn.decomposition import PCA
pca = PCA().fit(X_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)
intrinsic_d_linear = np.argmax(cumvar >= 0.95) + 1  # PCA dims for 95% variance

# Method 3: sklearn's n_components_  (not available for all methods)
# Practical: use intrinsic_d_linear as an upper bound on nonlinear n_components
```

---

### 8.7 Comprehensive Evaluation Code Template

```python
import numpy as np
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Preprocessing
X_scaled = StandardScaler().fit_transform(X)

# 1. Fit multiple methods
methods = {
    'Isomap': Isomap(n_neighbors=10, n_components=2),
    'LLE-MLLE': LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='modified', random_state=42),
    'SpectralEmbedding': SpectralEmbedding(n_components=2, n_neighbors=10, random_state=42),
    'MDS': MDS(n_components=2, metric=False, n_init=4, random_state=42),
}

embeddings = {}
for name, method in methods.items():
    embeddings[name] = method.fit_transform(X_scaled)

# 2. Trustworthiness
print("=== Trustworthiness Scores ===")
for name, X_embed in embeddings.items():
    t = trustworthiness(X_scaled, X_embed, n_neighbors=10)
    print(f"{name}: {t:.4f}")

# 3. Reconstruction error (where available)
print("\n=== Reconstruction Errors ===")
iso = methods['Isomap']
lle = methods['LLE-MLLE']
print(f"Isomap reconstruction_error(): {iso.reconstruction_error():.4f}")
print(f"LLE reconstruction_error_: {lle.reconstruction_error_:.4e}")
print(f"MDS stress_: {methods['MDS'].stress_:.4f}")

# 4. n_components elbow plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
errors_iso = [Isomap(n_neighbors=10, n_components=d).fit(X_scaled).reconstruction_error() for d in range(1, 10)]
ax[0].plot(range(1, 10), errors_iso, 'o-')
ax[0].set_xlabel('n_components'); ax[0].set_ylabel('Reconstruction Error')
ax[0].set_title('Isomap: Elbow for n_components')
plt.tight_layout()
```

---

## SECTION 9: API VERSION HISTORY AND BREAKING CHANGES

### Key changes to know (for backward compatibility in the course)

| Version | Class | Change | Impact |
|---------|-------|--------|--------|
| 0.22 | Isomap | Added `metric`, `p`, `metric_params` params | Minor addition |
| 1.1 | Isomap | Added `radius` parameter (alternative to n_neighbors) | Minor addition |
| 1.2 | MDS | Added `normalized_stress` parameter | New feature |
| 1.4 | MDS | Changed default of `normalized_stress` from False to 'auto' | **Breaking if relying on False default** |
| 1.5 | MDS | `metric=True/False` still works | Stable |
| 1.8 | MDS | `metric` parameter renamed to `metric_mds`; `dissimilarity` renamed to `metric` | **Breaking rename — old code fails** |
| 1.8 | MDS | `dissimilarity='euclidean'` renamed to `metric='euclidean'` | **Breaking rename** |
| 1.9 | MDS | Default `n_init` changed from 4 to 1 | **Potentially breaking: results may differ** |
| 1.9 | MDS | Default `eps` changed from 1e-3 to 1e-6 | **Stricter convergence default** |
| 1.9 | MDS | New `init='classical_mds'` option | New feature |
| 1.9 | sklearn | `ClassicalMDS` class added as separate class | New class |

### Version-safe code pattern

```python
import sklearn
from packaging.version import Version

sklearn_version = Version(sklearn.__version__)

# MDS - version-safe
if sklearn_version >= Version("1.8"):
    # v1.8+ API
    mds = MDS(n_components=2, metric_mds=True, metric='euclidean',
              n_init=4, random_state=42)
else:
    # v1.5.x API
    mds = MDS(n_components=2, metric=True, dissimilarity='euclidean',
              n_init=4, random_state=42)
```

**RECOMMENDATION FOR CHAPTER:** Write code examples using sklearn 1.5.x syntax (as specified in scope) but add a clearly labeled note: "Note: In sklearn 1.8+, the `metric` parameter in MDS was renamed to `metric_mds`, and `dissimilarity` was renamed to `metric`. The examples above use the sklearn 1.5.x API."

---

## SECTION 10: COMPLEXITY REFERENCE TABLE

| Method | Time Complexity | Space Complexity | N Practical Limit | Out-of-Sample |
|--------|----------------|------------------|-------------------|---------------|
| Isomap | O[D·log(k)·N·log(N)] + O[N²·(k+log(N))] + O[d·N²] | O[N²] | ~5,000 (dense) | Yes (transform()) |
| LLE (standard) | O[D·log(k)·N·log(N)] + O[D·N·k³] + O[d·N²] | O[N²] | ~5,000 | No |
| MLLE | Same as LLE + O[N·(k-D)·k²] | O[N²] | ~5,000 | No |
| HLLE | Same as LLE + O[N·d⁶] | O[N²] | ~2,000 | No |
| LTSA | O[D·log(k)·N·log(N)] + O[D·N·k³] + O[k²·d] + O[d·N²] | O[N²] | ~5,000 | No |
| SpectralEmbedding | O[D·log(k)·N·log(N)] + O[D·N·k³] + O[d·N²] | O[N²] | ~10,000 | No |
| MDS (SMACOF) | O[N²·max_iter] per init | O[N²] | ~1,000 | No |
| ClassicalMDS | O[N²·D] + O[N²·d] | O[N²] | ~2,000 | No |
| PHATE (landmarks) | O[N·n_landmark] | O[N·n_landmark] | ~500,000 | Yes (transform()) |

---

## APPENDIX: QUICK REFERENCE — WHICH METHOD TO USE?

| Situation | Recommended Method | Why |
|-----------|-------------------|-----|
| Single continuous manifold, geodesic structure matters | Isomap | Best geodesic preservation |
| Single continuous manifold, more robust needed | LTSA or MLLE | Better conditioned than standard LLE |
| Need cluster structure preserved | SpectralEmbedding | Related to spectral clustering |
| Have a precomputed distance/similarity matrix | MDS with dissimilarity='precomputed' | Works directly on distances |
| Data is ordinal (ranked comparisons only) | Non-metric MDS (metric=False) | Only order matters |
| Single-cell genomics, trajectory analysis | PHATE | Purpose-built for biological trajectories |
| Large dataset (N > 10,000) | PHATE with n_landmark | Nyström approximation; others are too slow |
| Neuroscience neural population data | Isomap or SpectralEmbedding + PHATE | Different methods capture different aspects |
| Protein conformations | Diffusion Maps (pydiffmap) or Isomap | Robust to conformational noise |
| Need out-of-sample extension | Isomap (transform()) or PHATE | Only methods with native transform |
| Interpretability of axes needed | LTSA or Isomap | Axes often correspond to physical variables |
| Noisy data | PHATE or MLLE | More robust to noise than Isomap or standard LLE |

---

*Dossier compiled from: Tenenbaum et al. 2000, Roweis & Saul 2000, Zhang & Wang 2006, Donoho & Grimes 2003, Zhang & Zha 2004, Belkin & Niyogi 2003, Coifman & Lafon 2006, Moon et al. 2019; sklearn 1.5.x documentation; PHATE 1.0.11 documentation; pydiffmap 0.2.0.1 documentation; PMC literature search 2023–2025; Journal of Climate 2024; PNAS protein folding literature; neuroscience manifold hypothesis literature (2023–2024); speech processing literature.*

*API flags: MDS `metric`/`dissimilarity` parameter name change verified against sklearn 1.5.2 and 1.8.0 docs. MDS `n_init` default change verified against sklearn 1.9.0 changelog. All code examples tested against sklearn 1.5.x API.*
