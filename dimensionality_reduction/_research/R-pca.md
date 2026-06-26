# Research Dossier: PCA and All Variants
## For: Dimensionality Reduction Masterclass — Chapter on PCA

**Research Date:** June 2026
**Target chapter length:** 18,000+ tokens
**Audience:** The writing agent has only this dossier and the course bible.

---

## TABLE OF CONTENTS

1. [Algorithm Landscape & Taxonomy](#1-algorithm-landscape--taxonomy)
2. [Must-Read Papers (Top 8)](#2-must-read-papers-top-8)
3. [Sklearn 1.5.x API — Every Parameter Documented](#3-sklearn-15x-api--every-parameter-documented)
4. [Specialized Python Packages](#4-specialized-python-packages)
5. [Real-World Applications (6–8 Concrete Examples)](#5-real-world-applications-68-concrete-examples)
6. [Hyperparameter Tuning Guide](#6-hyperparameter-tuning-guide)
7. [Expert Practitioner Checklist](#7-expert-practitioner-checklist)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Common Pitfalls With Concrete Examples](#9-common-pitfalls-with-concrete-examples)
10. [Mathematical Foundations (Quick Reference)](#10-mathematical-foundations-quick-reference)

---

## 1. ALGORITHM LANDSCAPE & TAXONOMY

### The PCA Family Tree

```
PCA (Pearson 1901, Hotelling 1933)
├── Exact SVD solver  ("full")
├── Eigenvalue decomposition ("covariance_eigh")  [NEW in sklearn 1.5]
├── Truncated ARPACK solver  ("arpack")
└── Randomized SVD solver  ("randomized") [Halko/Martinsson/Tropp 2011]
    └── IncrementalPCA  (Ross et al. 2008) — online/streaming variant
        └── MiniBatch approximation

Kernel PCA  (Schölkopf/Smola/Müller 1998)
└── Kernels: linear, poly, rbf, sigmoid, cosine, precomputed

Sparse PCA  (Zou/Hastie/Tibshirani 2006)
└── MiniBatchSparsePCA  — stochastic dictionary-learning variant

Factor Analysis  — latent-variable generalization of probabilistic PCA
└── Rotations: varimax, quartimax
```

### Core Intuition: What Each Variant Solves

| Variant | Core Problem It Solves | When to Use |
|---|---|---|
| **Classical PCA** | Find orthogonal directions of maximum variance | Dense data, moderate size, linear structure |
| **Randomized PCA** | Full PCA is too slow for large matrices | n_samples × n_features > 500×500, n_components ≪ min(n,p) |
| **Incremental PCA** | Data doesn't fit in RAM; arrives in streams | Out-of-core learning, online updates |
| **Kernel PCA** | Data has nonlinear structure PCA can't capture | Manifold data, nonlinear feature spaces |
| **Sparse PCA** | Want interpretable, sparse loading vectors | Gene expression, finance factor interpretation |
| **Factor Analysis** | Want per-feature noise estimates + latent variables | Survey data, psychometrics, mixed-noise settings |
| **LSA (text PCA)** | Reduce document-term matrix; handle synonymy | NLP, information retrieval |

---

## 2. MUST-READ PAPERS (TOP 8)

### Paper 1 — Pearson (1901): The Origin
**Full Citation:** Pearson, K. (1901). LIII. On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11), 559–572.
**DOI:** 10.1080/14786440109462720
**arXiv/URL:** https://www.tandfonline.com/doi/abs/10.1080/14786440109462720
**What You'll Learn:** Pearson geometrically derives the problem of finding the line (or plane) of closest fit to a scatter of points in p-dimensional space — the first formal statement of PCA. He uses least-squares projection rather than the covariance eigenvalue approach. Reading this shows that PCA started as a geometry problem, not a statistics problem; the two formulations are equivalent but give different intuitions.
**Difficulty:** Intermediate (classical notation, but short and readable at 13 pages)

### Paper 2 — Hotelling (1933): The Statistical Formalization
**Full Citation:** Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417–441.
**DOI:** Not available online (historical)
**URL:** http://cda.psych.uiuc.edu/hotelling_principal_components.pdf
**What You'll Learn:** Hotelling reformulates PCA as finding linear combinations of correlated variables that explain maximum variance, introduces the covariance/correlation matrix approach, defines eigenvalues as "latent roots," and establishes the orthogonality of principal components. This is the statistical canonical form that sklearn implements.
**Difficulty:** Intermediate (dense but historically critical)

### Paper 3 — Halko, Martinsson, Tropp (2011): Randomized PCA
**Full Citation:** Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217–288.
**DOI:** 10.1137/090771806
**arXiv:** https://arxiv.org/abs/0909.4061 (also ResearchGate)
**What You'll Learn:** This is the foundational paper for all modern fast PCA. The key insight: randomly project the matrix onto a lower-dimensional sketch, then run exact SVD on the small sketch. The paper proves rigorous error bounds, shows the power iteration trick (QR-factored power iterations improve accuracy on matrices with slowly decaying singular values), and benchmarks against LAPACK. The `svd_solver='randomized'` in sklearn directly implements Algorithm 4.3 from this paper.
**Difficulty:** Advanced (requires linear algebra background, but the algorithm boxes are self-contained)

### Paper 4 — Ross, Lim, Lin, Yang (2008): Incremental PCA
**Full Citation:** Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77(1–3), 125–141.
**DOI:** 10.1007/s11263-007-0075-7
**URL:** https://www.cs.ait.ac.th/~mdailey/cvreadings/Ross-Incremental.pdf
**What You'll Learn:** Presents an incremental SVD algorithm that updates the subspace model as new data arrives, including a forgetting factor for non-stationary distributions. The mean and variance must also be updated incrementally — this paper provides the correct update formulas that sklearn's IncrementalPCA implements. Key insight: you cannot simply run PCA on each batch separately; the merging step requires careful bookkeeping of accumulated statistics.
**Difficulty:** Intermediate (computer vision context, but the math is self-contained)

### Paper 5 — Schölkopf, Smola, Müller (1998): Kernel PCA
**Full Citation:** Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation*, 10(5), 1299–1319.
**DOI:** 10.1162/089976698300017467
**URL:** https://papers.nips.cc/paper/1491-kernel-pca-and-de-noising-in-feature-spaces (related NeurIPS work)
**What You'll Learn:** The kernel trick applied to PCA: instead of mapping data explicitly to a high-dimensional feature space and running PCA there, you compute a kernel matrix K (where K_{ij} = k(x_i, x_j)) and perform eigendecomposition on the centered kernel matrix. The paper proves that the eigenvectors of the kernel-PCA problem in feature space correspond to the eigenvectors of the centered K matrix in input space. This explains why sklearn's KernelPCA can find nonlinear structure without ever computing the high-dimensional feature vectors explicitly.
**Difficulty:** Advanced (requires kernel methods background, Mercer's theorem)

### Paper 6 — Zou, Hastie, Tibshirani (2006): Sparse PCA
**Full Citation:** Zou, H., Hastie, T., & Tibshirani, R. (2006). Sparse principal component analysis. *Journal of Computational and Graphical Statistics*, 15(2), 265–286.
**DOI:** 10.1198/106186006X113430
**URL:** https://hastie.su.domains/Papers/spc_jcgs.pdf
**What You'll Learn:** Shows that PCA can be re-cast as a regression problem (LASSO/elastic net), enabling sparse loading vectors. The SCoTLASS formulation (earlier work) is intractable; SPCA reformulates it as iterating between a ridge regression (fixing sparse loadings, solving for scores) and a LASSO (fixing scores, solving for loadings). Provides the first efficient algorithm for sparse PCA. sklearn's `SparsePCA` uses dictionary learning / coordinate descent, but the theoretical motivation comes from this paper.
**Difficulty:** Intermediate-Advanced (requires elastic net background)

### Paper 7 — Turk & Pentland (1991): Eigenfaces
**Full Citation:** Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. *Journal of Cognitive Neuroscience*, 3(1), 71–86.
**DOI:** 10.1162/jocn.1991.3.1.71
**URL:** https://www.face-rec.org/algorithms/PCA/jcn.pdf
**What You'll Learn:** The iconic application of PCA to face recognition. Each face image is treated as a point in a high-dimensional pixel space; PCA finds the "eigenfaces" (principal components). New face images are classified by projection distance in eigenface space. This paper established the reconstruction-error approach that is now standard for anomaly detection, and showed that >97% of variance is captured in the first ~20–50 components for face images. Extremely readable and pedagogically excellent.
**Difficulty:** Beginner-Intermediate (very accessible)

### Paper 8 — Deerwester, Dumais, Furnas, Landauer, Harshman (1990): LSA
**Full Citation:** Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science*, 41(6), 391–407.
**DOI:** 10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9
**URL:** http://wordvec.colorado.edu/papers/Deerwester_1990.pdf
**What You'll Learn:** Applies truncated SVD (equivalent to PCA on a term-document matrix) to information retrieval. The key insight is that SVD captures latent semantic structure: synonyms end up close together in the low-dimensional space even if they never co-occur in the same document, because they co-occur with similar words. This paper launched NLP dimensionality reduction and directly inspired word2vec, GloVe, and transformer pre-training objectives. This is where "topic = principal component" originated.
**Difficulty:** Beginner-Intermediate (well-written, minimal math)

### Bonus Paper (NeurIPS 2024): Kernel PCA for OoD Detection
**Full Citation:** Fang, K., Tao, Q., & Lv, K. (2024). Kernel PCA for Out-of-Distribution Detection. *NeurIPS 2024*.
**arXiv:** https://arxiv.org/abs/2402.02949
**GitHub:** https://github.com/fanghenshaometeor/ood-kernel-pca
**What You'll Learn:** Modern application of KPCA to deep learning. Shows that linear PCA is insufficient to separate in-distribution (InD) from out-of-distribution (OoD) features in DNN penultimate layers. KPCA with task-specific kernels achieves state-of-the-art OoD detection. The reconstruction error in kernel feature space is the anomaly score. Extended in 2025 with approximate KPCA for large-scale settings.
**Difficulty:** Advanced (assumes deep learning familiarity)

---

## 3. SKLEARN 1.5.X API — EVERY PARAMETER DOCUMENTED

### 3.1 `sklearn.decomposition.PCA`

```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=None,
    copy=True,
    whiten=False,
    svd_solver='auto',
    tol=0.0,
    iterated_power='auto',
    n_oversamples=10,          # New in 1.1
    power_iteration_normalizer='auto',  # New in 1.1
    random_state=None
)
```

| Parameter | Type | Default | What It Controls |
|---|---|---|---|
| `n_components` | int, float, or `'mle'` | `None` | Number of components to keep. `None` → keep all `min(n_samples, n_features)`. Float in (0,1) → keep enough components to explain that fraction of variance (requires `svd_solver='full'`). `'mle'` → Minka's MLE to auto-select dimensionality (requires `svd_solver='full'`). |
| `copy` | bool | `True` | If `False`, input X is overwritten during fitting. Saves memory but `fit(X).transform(X)` gives wrong results; use `fit_transform(X)` instead. |
| `whiten` | bool | `False` | Divide each component by its singular value so that output has unit variance per component. Removes relative scale information but makes outputs assumption-compatible for SVMs, distance-based classifiers. |
| `svd_solver` | `{'auto','full','covariance_eigh','arpack','randomized'}` | `'auto'` | See solver comparison table below. **NEW in 1.5:** `'covariance_eigh'` solver. |
| `tol` | float | `0.0` | Convergence tolerance for singular values (only used with `arpack`). |
| `iterated_power` | int or `'auto'` | `'auto'` | Number of power iterations for randomized solver. More iterations = better accuracy at cost of speed. Auto sets a data-dependent value. |
| `n_oversamples` | int | `10` | Only for `randomized`. Extra random vectors to improve conditioning. Increasing from 10 to 20 rarely helps unless singular values decay slowly. Added in version 1.1. |
| `power_iteration_normalizer` | `{'auto','QR','LU','none'}` | `'auto'` | Only for `randomized`. Normalization method within power iterations. `'QR'` is most accurate, `'LU'` is faster, `'none'` is least stable. Added in version 1.1. |
| `random_state` | int or `None` | `None` | Seed for `arpack` and `randomized` solvers. Pass an int for reproducible results across runs. |

**SVD Solver Comparison Table:**

| Solver | Algorithm | When Auto-Selected | Best For |
|---|---|---|---|
| `'full'` | Exact SVD via `scipy.linalg.svd` | Small matrices | Ground truth, `n_components='mle'`, float n_components |
| `'covariance_eigh'` | Eigendecomp on X^T X covariance | n_features < 1000 AND n_samples > 10×n_features | Tall skinny matrices, e.g., longitudinal clinical data |
| `'arpack'` | Truncated SVD via ARPACK IRLM | Rarely auto-selected | Sparse matrices, few components on medium data |
| `'randomized'` | Halko/Martinsson/Tropp 2011 | Data > 500×500 AND n_components < 80% of min(n,p) | Large dense matrices, n_components ≪ n_features |
| `'auto'` | Policy-based selection | Always (default) | Let sklearn decide |

**Key Attributes After Fit:**
- `components_`: shape (n_components, n_features) — the principal axes (row = one PC)
- `explained_variance_`: shape (n_components,) — eigenvalues
- `explained_variance_ratio_`: fraction of total variance per component
- `singular_values_`: singular values of the centered input matrix
- `mean_`: per-feature mean subtracted during centering
- `n_components_`: actual number of components (may differ from n_components if 'mle')
- `noise_variance_`: estimated noise variance (for probabilistic interpretation)

**Version Change Log:**
- **v1.5**: Added `'covariance_eigh'` solver — significantly faster for n_samples >> n_features with small n_features
- **v1.1**: Added `n_oversamples` and `power_iteration_normalizer` parameters
- **v0.18**: Added `tol`, `iterated_power`, `random_state`

---

### 3.2 `sklearn.decomposition.IncrementalPCA`

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(
    n_components=None,
    whiten=False,
    copy=True,
    batch_size=None
)
```

| Parameter | Type | Default | What It Controls |
|---|---|---|---|
| `n_components` | int | `None` | Components to keep. If `None`, set to `min(n_samples, n_features)` in first batch. Must be set explicitly when calling `partial_fit` batch-by-batch. |
| `whiten` | bool | `False` | Same as PCA whitening. |
| `copy` | bool | `True` | If `False`, input X is overwritten per-batch. |
| `batch_size` | int | `None` | Samples per batch in `fit()`. Default infers as `5 * n_features` to balance accuracy vs. memory. In `partial_fit()`, each call is one batch. |

**Key Usage Patterns:**
```python
# Out-of-core: data arrives in batches
ipca = IncrementalPCA(n_components=50)
for chunk in data_generator:
    ipca.partial_fit(chunk)
X_transformed = ipca.transform(X_new)

# In-core but memory-efficient
ipca = IncrementalPCA(n_components=50, batch_size=1000)
ipca.fit(X_large)  # internally splits into batch_size chunks
```

**Important Constraint:** `n_components` must be less than or equal to `batch_size`. If batch_size is not set, and you call `partial_fit` with a batch of 100 samples and n_components=50, this is fine. But if n_components > batch_size, sklearn raises an error.

**Added:** v0.16
**Key Difference from PCA:** IncrementalPCA uses approximate incremental SVD (not exact), so results may differ slightly from batch PCA. Convergence improves with larger batch sizes and more passes.

---

### 3.3 `sklearn.decomposition.KernelPCA`

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(
    n_components=None,
    kernel='linear',
    gamma=None,
    degree=3,
    coef0=1,
    kernel_params=None,
    alpha=1.0,
    fit_inverse_transform=False,
    eigen_solver='auto',
    tol=0,
    max_iter=None,
    iterated_power='auto',
    remove_zero_eig=False,
    random_state=None,
    copy_X=True,
    n_jobs=None
)
```

| Parameter | Type | Default | What It Controls |
|---|---|---|---|
| `n_components` | int | `None` | Components to keep. If `None`, all non-zero eigenvalue components are kept (can be many for large datasets). |
| `kernel` | str or callable | `'linear'` | Kernel function. Options: `'linear'` (same as PCA), `'poly'`, `'rbf'`, `'sigmoid'`, `'cosine'`, `'precomputed'`. Pass a callable for custom kernels. |
| `gamma` | float | `None` | Kernel coefficient for `rbf`, `poly`, `sigmoid`. If `None`, uses `1/n_features`. **Critical to tune for rbf kernel.** |
| `degree` | float | `3` | Degree for polynomial kernel. Integer values only make sense mathematically, but sklearn accepts floats. |
| `coef0` | float | `1` | Independent term for `poly` (x·y + coef0)^degree and `sigmoid` tanh(gamma·x·y + coef0). |
| `kernel_params` | dict | `None` | Alternative way to pass kernel parameters when `kernel` is a callable. |
| `alpha` | float | `1.0` | Ridge regularization strength for the inverse transform regression (only when `fit_inverse_transform=True`). Higher alpha = smoother inverse but more distortion. |
| `fit_inverse_transform` | bool | `False` | Learn approximate inverse mapping for reconstruction. Adds O(n²) overhead. Required for reconstruction error–based anomaly detection. |
| `eigen_solver` | `{'auto','dense','arpack','randomized'}` | `'auto'` | Solver for kernel matrix eigendecomposition. `'dense'` = full eigendecomp (O(n³)), `'arpack'` = truncated, `'randomized'` = fast approximation. |
| `tol` | float | `0` | Convergence tolerance for `arpack`. |
| `max_iter` | int | `None` | Max iterations for `arpack`. |
| `iterated_power` | int or `'auto'` | `'auto'` | Power iterations for `randomized` solver. Auto sets 7 if n_components < 0.1 × min(n,p), else 4. |
| `remove_zero_eig` | bool | `False` | Remove components with zero eigenvalues. Useful when `n_components=None`. |
| `random_state` | int or `None` | `None` | Seed for `arpack` or `randomized` solvers. |
| `copy_X` | bool | `True` | Store a copy of X in `X_fit_`. Set to `False` to save memory (but X must not change after fitting). |
| `n_jobs` | int | `None` | Parallelism for kernel matrix computation. `-1` = all cores. |

**Critical KPCA Limitation:** The entire n_samples × n_samples kernel matrix must be computed and stored. For n_samples = 10,000, this is 100M floats = ~800MB. For n_samples = 100,000, it's impractical. Use `eigen_solver='randomized'` + `n_components` to mitigate.

**Inverse Transform Note:** `KernelPCA.inverse_transform()` uses a ridge regression (alpha hyperparameter) to learn the preimage. This is approximate — the true preimage may not exist in input space for all kernels (e.g., RBF is infinite-dimensional). The reconstruction is always an approximation.

---

### 3.4 `sklearn.decomposition.SparsePCA`

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(
    n_components=None,
    alpha=1,
    ridge_alpha=0.01,
    max_iter=1000,
    tol=1e-8,
    method='lars',
    n_jobs=None,
    U_init=None,
    V_init=None,
    verbose=False,
    random_state=None
)
```

| Parameter | Type | Default | What It Controls |
|---|---|---|---|
| `n_components` | int | `None` | Sparse atoms to extract. If `None`, set to `n_features`. |
| `alpha` | float | `1` | Sparsity strength. Higher → fewer nonzero loadings. This is the L1 regularization strength in the dictionary learning step. Key tuning parameter. |
| `ridge_alpha` | float | `0.01` | Ridge regularization for the `transform` step (code inference). Improves conditioning. Usually needs little tuning. |
| `max_iter` | int | `1000` | Max alternating-optimization iterations. |
| `tol` | float | `1e-8` | Convergence tolerance. |
| `method` | `{'lars','cd'}` | `'lars'` | Optimization for the sparse coding step. `'lars'` (Least Angle Regression) is faster when components are highly sparse; `'cd'` (coordinate descent) is more stable. |
| `n_jobs` | int | `None` | Parallelism for sparse coding across samples. |
| `U_init` | ndarray | `None` | Warm-start initialization for loadings (scores). |
| `V_init` | ndarray | `None` | Warm-start initialization for components (dictionary atoms). |
| `verbose` | int | `False` | Verbosity level. |
| `random_state` | int or `None` | `None` | Seed for dictionary initialization. |

**Warning:** SparsePCA is much slower than PCA. For n_samples=1000, n_features=100, n_components=20, expect 10–60 seconds. The components are not orthogonal (unlike PCA). Transform uses ridge regression to find sparse codes.

---

### 3.5 `sklearn.decomposition.MiniBatchSparsePCA`

```python
from sklearn.decomposition import MiniBatchSparsePCA

mbspca = MiniBatchSparsePCA(
    n_components=None,
    alpha=1,
    ridge_alpha=0.01,
    max_iter=1000,
    callback=None,
    batch_size=3,        # NOTE: batch_size here is in FEATURES, not samples
    verbose=False,
    shuffle=True,
    n_jobs=None,
    method='lars',
    random_state=None,
    tol=1e-3,            # Added in v1.1
    max_no_improvement=10  # Added in v1.1
)
```

| Parameter | Type | Default | What It Controls |
|---|---|---|---|
| `n_components` | int | `None` | Atoms to extract. |
| `alpha` | int | `1` | Sparsity parameter (same as SparsePCA). |
| `ridge_alpha` | float | `0.01` | Ridge for transform step. |
| `max_iter` | int | `1000` | Max full passes over the dataset. Added in v1.2 (replaces old `n_iter`). |
| `callback` | callable | `None` | Called every 5 iterations. Useful for monitoring. |
| `batch_size` | int | `3` | **Number of FEATURES** (not samples) per mini-batch. Different semantics from IncrementalPCA! |
| `shuffle` | bool | `True` | Shuffle features before batching. |
| `n_jobs` | int | `None` | Parallelism. |
| `method` | `{'lars','cd'}` | `'lars'` | Sparse coding method. |
| `random_state` | int or `None` | `None` | Seed. |
| `tol` | float | `1e-3` | Early stopping: stop if dictionary change norm < tol. Added v1.1. |
| `max_no_improvement` | int or `None` | `10` | Early stopping: stop if no improvement for this many consecutive batches. Added v1.1. Set `None` to disable. |

**API HAZARD:** `batch_size` in MiniBatchSparsePCA means features, NOT samples. This is the opposite of every other "mini-batch" interface in sklearn. Flag this for students.

---

### 3.6 `sklearn.decomposition.FactorAnalysis`

```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(
    n_components=None,
    tol=0.01,
    copy=True,
    max_iter=1000,
    noise_variance_init=None,
    svd_method='randomized',
    iterated_power=3,
    rotation=None,          # Added v0.24: 'varimax' or 'quartimax'
    random_state=0
)
```

| Parameter | Type | Default | What It Controls |
|---|---|---|---|
| `n_components` | int | `None` | Latent factor dimensionality. |
| `tol` | float | `0.01` | Stopping tolerance for log-likelihood increase in EM iterations. |
| `copy` | bool | `True` | Copy X before fitting. |
| `max_iter` | int | `1000` | Max EM iterations. |
| `noise_variance_init` | array-like | `None` | Initial diagonal noise variance. Default: `np.ones(n_features)`. |
| `svd_method` | `{'lapack','randomized'}` | `'randomized'` | SVD backend. |
| `iterated_power` | int | `3` | Power iterations for randomized SVD. |
| `rotation` | `{'varimax','quartimax'}` or `None` | `None` | Post-fit rotation for interpretability. Varimax maximizes variance of squared loadings (simple structure). |
| `random_state` | int | `0` | Seed (default 0, NOT None — unusual!). |

**FA vs. PCA Core Difference:**
- PCA assumes isotropic noise: all features have the same noise variance σ²
- Factor Analysis estimates per-feature noise variance ψ_i (diagonal Ψ matrix)
- FA maximizes the marginal log-likelihood p(X) via EM; PCA does not use a probabilistic model in the standard form

**When FA Beats PCA:** When features have very different noise levels (e.g., some sensors are noisier than others), FA finds latent factors that are robust to feature-specific noise. PCA will have its components distorted by high-noise features.

---

## 4. SPECIALIZED PYTHON PACKAGES

### 4.1 fbpca (Facebook PCA)
- **PyPI name:** `fbpca`
- **Version (2026):** 1.0 (released 2014; no updates since)
- **Install:** `pip install fbpca`
- **Key import:** `from fbpca import pca, svd`
- **License:** BSD + patent grant (from Facebook/Meta)
- **Maintenance:** DISCONTINUED — last release 2014, repo is archived
- **What it does better than sklearn:** At the time of release, was significantly faster and more numerically stable than sklearn's randomized SVD. It pre-dates sklearn's `n_oversamples` and `power_iteration_normalizer` additions.
- **Status (2026):** sklearn's randomized SVD has caught up and surpassed fbpca in both performance and correctness after the 1.1 improvements. **Not recommended for new projects.** Only worth using if you need the exact Facebook-paper results for comparison.
- **Example:**
```python
from fbpca import pca
U, s, Vt = pca(X, k=50, raw=False)  # Returns truncated SVD
```

### 4.2 prince (Factor Analysis Suite)
- **PyPI name:** `prince`
- **Version (2026):** 0.19.0 (released May 2026; actively maintained)
- **Install:** `pip install prince`
- **Key import:** `import prince`
- **License:** MIT
- **Author:** Max Halford
- **Maintenance:** ACTIVE — regular releases, sklearn API
- **What it does better than sklearn:**
  - Implements MCA (Multiple Correspondence Analysis) for purely categorical data
  - FAMD (Factor Analysis of Mixed Data) for combined numeric + categorical
  - MFA (Multiple Factor Analysis) for grouped variables
  - GPA (Generalized Procrustes Analysis) for shape data
  - Interactive Altair visualizations included
  - Supplementary rows/columns (project new samples without refitting)
  - Validated against R's FactoMineR
- **Example:**
```python
import prince
pca = prince.PCA(n_components=2, n_iter=3, random_state=42)
pca = pca.fit(X_df)
pca.plot(X_df)  # Returns interactive Altair chart
```

### 4.3 scikit-learn's TruncatedSVD (for sparse/text data)
- **PyPI name:** part of `scikit-learn`
- **Install:** `pip install scikit-learn`
- **Key import:** `from sklearn.decomposition import TruncatedSVD`
- **License:** BSD
- **What it does better than PCA:** Does NOT center the data before SVD, which is critical for sparse matrices (centering destroys sparsity). Use TruncatedSVD for text/LSA workflows, never `PCA` on a sparse TF-IDF matrix.
- **Example:**
```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000)
X_sparse = tfidf.fit_transform(documents)

lsa = TruncatedSVD(n_components=100, random_state=42)
X_lsa = lsa.fit_transform(X_sparse)  # Works on sparse; PCA would not
```

### 4.4 PyParSVD (Streaming/Distributed SVD)
- **PyPI name:** `PyParSVD`
- **Version (2026):** ~1.0 (research package)
- **Install:** `pip install PyParSVD`
- **Key import:** `from PyParSVD.streaming import StreamingSVD`
- **License:** MIT
- **Maintenance:** Research-grade (arXiv: 2108.08845)
- **What it does better:** True distributed/streaming SVD using data parallelism across nodes. Designed for HPC environments where data is distributed across machines and cannot be gathered.

### 4.5 pyDRMetrics (Evaluation Package)
- **PyPI name:** `pyDRMetrics`
- **Install:** `pip install pyDRMetrics`
- **What it does:** Comprehensive dimensionality reduction evaluation toolkit. Computes reconstruction error, trustworthiness, continuity, co-k-nearest neighbor size, LCMC, and rank-based metrics with a unified API.
- **License:** MIT
- **Reference:** Xia et al. (2021). pyDRMetrics. *Heliyon*, 7(2).

### 4.6 qrpca (GPU-Accelerated PCA)
- **PyPI name:** `qrpca`
- **Install:** `pip install qrpca`
- **What it does:** GPU-accelerated randomized PCA using QR decomposition. Claims 15× speedup over CPU implementations for large matrices (n > 10,000 features). arXiv: 2206.06797.
- **When to use:** Large-scale genomics, imaging, or NLP where n_features >> 10,000.

---

## 5. REAL-WORLD APPLICATIONS (6–8 CONCRETE EXAMPLES)

### Application 1: Genomics — Population Stratification (Classic)
**Domain:** Computational genomics / population genetics
**Problem:** Human genomics datasets have ~1M+ SNP (single nucleotide polymorphism) loci per individual and thousands of individuals. Direct analysis is computationally infeasible. More importantly, population structure (ancestry) confounds GWAS (genome-wide association studies) — if you don't control for ancestry, you'll find fake gene-disease associations driven by population differences.
**Why PCA:** The top 10 PCs of the SNP matrix capture population structure with remarkable fidelity. PC1 vs PC2 cleanly separates European, African, East Asian, and South Asian ancestries in HapMap/1000 Genomes data. Researchers include the top 10–20 PCs as covariates in GWAS regression to control for confounding.
**Algorithm:** Standard PCA (or FastPCA for large cohorts) on the n_samples × n_SNPs matrix, often with the FLASHPCA or EIGENSOFT tools.
**Company/Paper Reference:** Price et al. (2006) "Principal components analysis corrects for stratification in genome-wide association studies" *Nature Genetics*. Used routinely in Biobank UK (500K people) and by 23andMe.
**Data shape:** Typically 100K samples × 1M SNPs. Use randomized PCA or ARPACK.
**Sklearn code note:** For this scale, use `PCA(n_components=20, svd_solver='randomized')` or external tools like FLASHPCA.

### Application 2: Finance — Statistical Risk Factor Decomposition
**Domain:** Quantitative finance / risk management
**Problem:** A portfolio of 500 stocks has a 500×500 covariance matrix with 125,250 parameters to estimate. This is extremely unstable with limited historical data. Portfolio optimization (mean-variance) becomes numerically unstable.
**Why PCA:** The covariance matrix of stock returns has a low-rank structure: 1–5 PCs explain ~50–70% of variance. The first PC is almost always "market risk" (all loadings same sign). PC2 often captures growth vs. value, PC3 captures size, etc. These are the statistical factors analogous to Fama-French factors.
**Algorithm:** PCA on the returns covariance matrix, then reconstruct a regularized covariance using only top k PCs.
**Company/Paper Reference:** JP Morgan, BlackRock risk systems; Federal Reserve 2025 paper "Portfolio Margining Using PCA Latent Factors" (https://www.federalreserve.gov/econres/feds/files/2025016pap.pdf). Also QuestDB guide on PCA for portfolio risk.
**Practical approach:**
```python
from sklearn.decomposition import PCA
import numpy as np

# returns: shape (n_days, n_stocks)
pca = PCA(n_components=20)
pca.fit(returns)

# Reconstruct regularized covariance
# Full covariance = systematic risk (PCA) + idiosyncratic (diagonal residual)
systematic = pca.components_.T @ np.diag(pca.explained_variance_) @ pca.components_
residual_var = np.var(returns - pca.inverse_transform(pca.transform(returns)), axis=0)
reg_cov = systematic + np.diag(residual_var)
```
**Non-obvious insight:** The eigen-portfolios (rows of components_) are theoretically optimal mean-variance portfolios if returns follow a factor model. The first eigen-portfolio is essentially the market-cap-weighted index.

### Application 3: Anomaly Detection — Manufacturing Quality Control
**Domain:** Industrial manufacturing / quality inspection
**Problem:** A semiconductor fab has a wafer inspection system producing 10,000-dimensional feature vectors (optical reflectance measurements). Normal wafers form a low-dimensional manifold; defective wafers deviate from this manifold.
**Why PCA:** Train PCA on a batch of known-good wafers. The reconstruction error ||X - PCA.inverse_transform(PCA.transform(X))||² is near-zero for normal wafers and large for defective ones (their variation is not captured by the principal components of normal variation). This is the "Q-statistic" in statistical process control.
**Algorithm:** Standard PCA + reconstruction error threshold.
**Reference:** A 2024 industrial case study on steel manufacturing showed ~30% reduction in production downtime using PCA-based real-time monitoring. Fast Anomaly Detection Using Cascades of Null Subspace PCA Detectors (MDPI Sensors, 2025).
**Sklearn code:**
```python
pca = PCA(n_components=0.95)  # Keep 95% variance
pca.fit(X_normal_wafers)

def anomaly_score(X_new):
    X_reconstructed = pca.inverse_transform(pca.transform(X_new))
    return np.mean((X_new - X_reconstructed) ** 2, axis=1)

scores = anomaly_score(X_test)
threshold = np.percentile(scores, 99)  # Train on known-good
predictions = (scores > threshold).astype(int)
```
**Limitation:** PCA reconstruction may sometimes "generalize" too well, reconstructing anomalies that share linear structure with normal data. Use Kernel PCA for nonlinear anomaly boundaries.

### Application 4: Computer Vision — Eigenfaces and Face Recognition
**Domain:** Computer vision / biometrics
**Problem:** Recognize faces in a database given a probe image. Raw pixel space is too high-dimensional, contains lighting and pose variation.
**Why PCA:** Turk & Pentland (1991) showed that face images lie on a low-dimensional manifold. The top 20–50 PCs of a face image dataset capture the dominant modes of face variation (illumination, expression, pose). A face can be represented as a weighted sum of "eigenfaces."
**Algorithm:** PCA on flattened face images; classify by nearest neighbor in PC space.
**Historical significance:** This was state-of-the-art face recognition in the 1990s and established the reconstruction-error paradigm. Modern deep learning far surpasses it, but Eigenfaces remains the pedagogical gold standard for teaching PCA.
**Non-obvious extension:** The reconstruction error (not just the distance in eigenface space) can reject non-face images — if an image projects poorly onto the face subspace, it's probably not a face.
**Code pattern:**
```python
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data  # (n_images, n_pixels)

pca = PCA(n_components=150, svd_solver='randomized', whiten=True)
X_pca = pca.fit_transform(X)

# Eigenfaces are pca.components_ reshaped to image dimensions
eigenfaces = pca.components_.reshape((150, *faces.images.shape[1:]))
```

### Application 5: NLP / Information Retrieval — Latent Semantic Analysis
**Domain:** Natural language processing / search engines
**Problem:** Search engine gets query "car"; documents contain "automobile" but not "car." Keyword matching fails. Also, document-term matrices are extremely sparse and high-dimensional (100K+ unique terms).
**Why PCA/SVD:** Truncated SVD on the term-document matrix (LSA) maps synonyms to nearby locations in the latent semantic space, because synonyms co-occur with similar context words. This solves the synonym problem. The top 200–500 components typically suffice.
**Algorithm:** TruncatedSVD (NOT PCA — sparse data must not be centered) on TF-IDF matrix.
**Reference:** Deerwester et al. (1990). Also the foundation for all subsequent word embeddings.
**Code:**
```python
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

lsa_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, sublinear_tf=True)),
    ('svd', TruncatedSVD(n_components=300, random_state=42))
])
X_lsa = lsa_pipeline.fit_transform(documents)
```

### Application 6: Quantum Chemistry — Molecular Descriptor Compression
**Domain:** Computational chemistry / drug discovery
**Problem:** Molecules are described by thousands of quantum-mechanical descriptors (HOMO/LUMO energies, dipole moments, partial charges for each atom, fingerprints). Training QSAR (quantitative structure-activity relationship) models on all descriptors leads to overfitting.
**Why PCA:** PCA compresses correlated molecular descriptors into a small set of uncorrelated features. In practice, the top 20–50 PCs of molecular descriptor matrices explain >80% of variance and dramatically improve QSAR model performance while reducing the chance of overfitting.
**Algorithm:** Standard PCA on standardized descriptor matrix. Also: PCA on quantum mechanical wavefunctions to find the dominant modes of electronic structure variation.
**Reference:** A 2024 ChemRxiv paper compared PCA, t-SNE, UMAP, GTM for chemical space exploration. A 2021 ChemRxiv paper showed PCA reduces molecular orbital space for quantum computation, crucial for VQE (Variational Quantum Eigensolver) circuits with limited qubits. "Reduction of Orbital Space for Molecular Orbital Calculations with Quantum Computation."
**Non-obvious insight:** The principal components of molecular descriptor space often correspond to interpretable chemical properties: PC1 may track molecular size, PC2 may track polarity, PC3 may track ring aromaticity.

### Application 7: Neuroscience — EEG/BCI Artifact Removal and Feature Extraction
**Domain:** Neuroscience / brain-computer interfaces
**Problem:** EEG recordings from 64+ electrodes contain high-dimensional, highly correlated signals. Muscle artifacts, eye blinks, and cardiac signals contaminate the neural signal. Feature extraction for BCI classification requires dimensionality reduction.
**Why PCA:** PCA separates the high-variance artifacts (which dominate the first PCs due to their amplitude) from the lower-variance neural signals. Researchers inspect the first few PCs, identify artifact components, and reconstruct the signal without them. This is a faster alternative to ICA.
**Algorithm:** Standard PCA on time × channel EEG matrix. Also used as preprocessing before CSP (Common Spatial Patterns) in motor imagery BCI.
**Reference:** Multiple IEEE BCI papers; EEG signal classification using PCA with neural networks. Nature Scientific Reports (2025) hybrid deep learning + PCA for EEG classification.
**Caveat:** PCA assumes linear mixing and Gaussian signals; ICA (which uses non-Gaussianity) is more principled for artifact separation. PCA is faster and often sufficient for simple preprocessing.

### Application 8: Materials Science — High-Entropy Alloy Composition Space
**Domain:** Materials science / alloy design
**Problem:** High-entropy alloys have 5+ principal elements with continuous composition ranges. The composition space is 5+ dimensional. Predicting material properties (hardness, corrosion resistance) across this space for materials discovery requires efficient exploration.
**Why PCA:** PCA on a database of alloy compositions + properties reveals the principal axes of composition variation. These principal axes guide efficient experimental design (which compositions to try next). Also used to visualize clusters of alloys with similar properties.
**Algorithm:** Standard PCA for visualization and active learning guidance.
**Reference:** "Roadmap on Data-Centric Materials Science" (arXiv: 2402.10932, 2024). PCA-guided active learning for materials discovery.
**Non-obvious insight:** In materials science, the loadings (components_) are often directly interpretable as "alloying tendencies" — which elements tend to substitute for each other.

---

## 6. HYPERPARAMETER TUNING GUIDE

### 6.1 n_components — The Master Hyperparameter

**What it controls:** Latent space dimensionality. The single most important decision.

**Recommended selection strategies (in priority order):**

1. **Scree Plot (Elbow Method):** Plot `explained_variance_ratio_` vs. component index. Look for the "elbow" where the curve flattens. Subjective but robust for clear structure.

2. **Cumulative Variance Threshold:** Choose k such that sum(explained_variance_ratio_[:k]) ≥ threshold. Standard thresholds: 80% (exploration), 90% (balanced), 95% (conservative), 99% (near-lossless).

3. **Kaiser Rule:** Retain components where eigenvalue (explained_variance_) > 1.0 (if data was standardized). Heuristic — works poorly for high-dimensional data.

4. **Minka's MLE (n_components='mle'):** Only with `svd_solver='full'`. Uses Bayesian model selection to estimate dimensionality from the gap between retained and discarded eigenvalues. Best for moderate-dimensional data (n_features < 1000).

5. **Task-based selection:** Optimize n_components via cross-validation on downstream task (classification accuracy, reconstruction error on held-out data). Most principled for supervised applications.

**"Too high" symptoms:** Overfitting in downstream models; components capture noise rather than signal; eigenvalues become very small and nearly equal (noise floor).

**"Too low" symptoms:** High reconstruction error; downstream task performance degrades; you can't distinguish classes in PC space.

**Optuna search space:**
```python
import optuna
def objective(trial):
    n_components = trial.suggest_int('n_components', 2, min(n_samples, n_features) // 2)
    pca = PCA(n_components=n_components)
    # ... cross-validate downstream task
    return cv_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Code for scree plot:**
```python
pca = PCA(svd_solver='full').fit(X_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.bar(range(1, 21), pca.explained_variance_ratio_[:20])
ax1.set_xlabel('Principal Component'); ax1.set_ylabel('Variance Ratio')
ax1.set_title('Scree Plot')
ax2.plot(range(1, len(cumvar)+1), cumvar, 'b-o', markersize=3)
ax2.axhline(0.90, color='r', linestyle='--', label='90% threshold')
ax2.set_xlabel('Number of Components'); ax2.set_ylabel('Cumulative Variance')
plt.tight_layout()
```

---

### 6.2 svd_solver

**Recommended choice:**
- Use `'auto'` in most cases — sklearn's heuristic is well-calibrated
- Force `'randomized'` for: n_features > 1000 AND n_components < 0.3 × min(n,p)
- Force `'full'` when: need `n_components='mle'` or float n_components; verifying exact results
- Force `'covariance_eigh'` when: n_samples >> n_features (e.g., 100K samples, 50 features) — **new in 1.5, often fastest in this regime**
- Force `'arpack'` when: X is sparse OR n_components is very small (< 10)

**Too low iterated_power (0 or 1):** Fast but inaccurate when singular values decay slowly (slowly decaying spectrum = data doesn't have strong low-rank structure). Manifests as principal components that are not orthogonal in practice.

**Too high iterated_power (>7):** Minimal accuracy gain; the power method converges quickly.

**Recommended iterated_power:** `'auto'` (sklearn chooses based on n_components vs. matrix size). If overriding: start with 4, increase to 6-7 if you see component instability across runs.

**n_oversamples tuning:**
- Default `10` is sufficient for most cases
- Increase to `20-50` when singular value spectrum decays slowly
- Rarely need to go above 50
- Optuna: `trial.suggest_int('n_oversamples', 5, 50)`

---

### 6.3 KernelPCA Hyperparameters

**kernel selection:**
```
linear  → same as PCA (baseline; test this first)
rbf     → general nonlinear; most commonly useful
poly    → when features have polynomial interactions
sigmoid → rarely better than rbf
cosine  → text/NLP features normalized to unit sphere
```

**gamma (for rbf, poly, sigmoid):**
- Default `1/n_features` is a reasonable starting point
- **Too high gamma:** Kernel is very peaked (only nearby points interact) → overfitting, many near-zero eigenvalues
- **Too low gamma:** Kernel is flat → nearly linear behavior, barely different from linear PCA
- Use the median-heuristic as initialization: `gamma = 1 / (2 * np.median(pairwise_distances(X_sample))**2)`
- **Optuna search space:**
```python
gamma = trial.suggest_float('gamma', 1e-4, 10.0, log=True)
```

**degree (for poly):**
- Range: 2–5 for most applications
- Higher degrees → more expressive but numerically unstable
- `trial.suggest_int('degree', 2, 5)`

**coef0 (for poly, sigmoid):**
- Controls the "offset" in polynomial/sigmoid kernels
- Range: [0, 10] typically
- `trial.suggest_float('coef0', 0.0, 10.0)`

**n_components for KernelPCA:**
- Harder to choose than linear PCA because there's no clean eigenvalue-based variance explained
- Use: plot eigenvalues of kernel matrix K (after centering) and look for elbow
- Typical range: 10–100 for visualization; more for feature extraction

**Critical note:** KernelPCA has no natural way to choose n_components using "% variance explained" because the kernel matrix eigenvalues don't directly correspond to variance in input space.

---

### 6.4 SparsePCA alpha

**alpha is the most important SparsePCA hyperparameter.**

- **Too high alpha:** All loadings → 0; components contain no information
- **Too low alpha:** Loadings are dense; similar to regular PCA but slower
- **Target:** A sparsity of ~70-90% nonzeros-per-component is typical for gene expression / factor analysis
- **Check:** `np.mean(transformer.components_ == 0)` — aim for 0.7-0.95

**Optuna search space:**
```python
alpha = trial.suggest_float('alpha', 0.01, 10.0, log=True)
```

**Diagnostic:** After fitting, plot the loading heatmap. Each component should activate only a small subset of features.

---

### 6.5 whiten (PCA)

**When to use:** 
- whiten=True: downstream uses distance-based algorithms (SVM with RBF kernel, k-NN, k-means), or when components have very different scales
- whiten=False (default): downstream uses linear classifiers, or when you want to preserve the relative importance of components

**Trade-off:** Whitening removes information about relative variance (PC1 no longer "explains more" than PC2), but makes the feature space assumption-compatible for many classifiers.

---

## 7. EXPERT PRACTITIONER CHECKLIST

### BEFORE Fitting

**Data Quality:**
- [ ] Check for missing values: `X.isna().sum()`. PCA cannot handle NaNs. Impute with `SimpleImputer(strategy='mean')` or use IterativeImputer for better results.
- [ ] Check for constant features: `X.std(axis=0) == 0`. PCA on zero-variance features fails or produces unstable results.
- [ ] Check for outliers: plot boxplots or compute z-scores. PCA's covariance is sensitive to outliers. Consider `RobustScaler` or winsorization.
- [ ] Check data scale: plot histograms per feature. Wildly different scales → features with large range dominate.

**Preprocessing (CRITICAL):**
- [ ] **Standardize with StandardScaler if features have different units** (e.g., age in years vs. income in dollars). This is the most common mistake.
- [ ] If all features are in the same units (e.g., all pixel values, all gene expression levels), centering is sufficient.
- [ ] Do NOT use PCA on one-hot encoded categorical features without careful thought. Consider MCA (Multiple Correspondence Analysis) instead.
- [ ] For sparse data (text, genomics count matrices), use TruncatedSVD NOT PCA. PCA centers data, destroying sparsity.

**Algorithm Selection:**
- [ ] Is data linear or nonlinear? If you know the structure is nonlinear (e.g., Swiss roll, circular data), use Kernel PCA.
- [ ] How many samples? > 10K? Consider IncrementalPCA or Randomized PCA.
- [ ] Do you need interpretable loadings? Consider Sparse PCA.
- [ ] Do features have different noise levels? Consider Factor Analysis.
- [ ] Are you doing NLP/text? Use TruncatedSVD via LSA pipeline.

**Fit Train/Test Split:**
- [ ] CRITICAL: Fit PCA ONLY on training data. Then `transform()` on test data. Never fit on test data.
- [ ] Use sklearn Pipeline to avoid data leakage:
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('clf', LogisticRegression())
])
pipe.fit(X_train, y_train)  # Correct: PCA fit on train only
```

### DURING Fitting

**Monitor:**
- [ ] Check `pca.explained_variance_ratio_` — does it show clear signal (first few PCs >> rest)?
- [ ] Compute cumulative variance: `np.cumsum(pca.explained_variance_ratio_)`. Where does it cross 80%, 90%, 95%?
- [ ] For SparsePCA: check `pca.n_iter_` — if it hit `max_iter`, consider increasing it.
- [ ] For IncrementalPCA: `pca.n_samples_seen_` — ensures all data was processed.

**Solver Sanity:**
- [ ] If using `randomized` solver, verify results match `full` solver on a small subsample.
- [ ] Run PCA twice with different `random_state`. If components differ substantially → likely need more `iterated_power`.

### AFTER Fitting

**Verification:**
- [ ] Plot scree plot and cumulative variance curve.
- [ ] Visualize PC1 vs PC2 scatter colored by known labels — do clusters separate?
- [ ] Compute reconstruction error: `np.mean((X - pca.inverse_transform(pca.transform(X)))**2)`. How much information is lost?
- [ ] Check component loadings: `pd.DataFrame(pca.components_.T, index=feature_names)`. Do first few PCs have interpretable patterns?
- [ ] For KernelPCA: verify that `fit_inverse_transform=True` reconstruction is reasonable.

**Downstream Task Check:**
- [ ] Compare downstream model performance with n_components=k vs. original features. Ensure dimensionality reduction actually helps or at least doesn't hurt.
- [ ] If performance degrades significantly, you may be using too few components or the data structure doesn't suit PCA.

**Signs Something Is Wrong:**
- First few PCs explain < 20% variance each → data has distributed structure; PCA may not be appropriate
- PC2 explains almost the same variance as PC1 → data may have spherical structure; PCA arbitrary rotation
- After PCA, downstream accuracy is much worse → either too few components, or PCA destroyed discriminative features (try LDA instead)
- KernelPCA components look like noise → gamma is too large or n_components is too many

---

## 8. EVALUATION METRICS

### 8.1 Reconstruction Error (Primary Metric)

**Formula:** RE = (1/n) × Σ_i ||x_i - x̂_i||²  where x̂_i = PCA.inverse_transform(PCA.transform(x_i))

**Interpretation:** How much information is lost. Lower is better.

**Normalized version:** Relative reconstruction error = RE / Var(X) ≈ 1 - sum(explained_variance_ratio_[:k])

**Sklearn code:**
```python
X_reconstructed = pca.inverse_transform(pca.transform(X))
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
relative_error = 1 - np.sum(pca.explained_variance_ratio_)
```

**For anomaly detection (per-sample scores):**
```python
per_sample_error = np.mean((X - X_reconstructed) ** 2, axis=1)
```

**Relationship to explained variance:** For linear PCA, relative reconstruction error = 1 - Σ_k λ_k / Σ_total λ. The explained_variance_ratio_ directly gives you the complement.

---

### 8.2 Explained Variance Ratio

**Formula:** EVR_k = λ_k / Σ_i λ_i  where λ_k is the k-th eigenvalue of the covariance matrix

**Interpretation:** The fraction of total data variance captured by principal component k.

**Usage:**
```python
print(pca.explained_variance_ratio_)          # Per-component
print(pca.explained_variance_ratio_.cumsum())  # Cumulative
```

**Note:** This metric is specific to linear PCA. KernelPCA does not directly expose explained variance in input space.

---

### 8.3 Trustworthiness (Neighborhood Preservation)

**Definition:** Measures whether the k-nearest neighbors in the low-dimensional space are also neighbors in the high-dimensional space. Score ∈ [0, 1], where 1 = perfect neighborhood preservation.

**Formula:** T(k) = 1 - (2 / (nk(2n-3k-1))) × Σ_i Σ_{j∈U_k(i)} (r(i,j) - k)

where U_k(i) is the set of points that are in the k-neighborhood in low-D but not in high-D, and r(i,j) is the rank of point j from i in high-D space.

**Sklearn code:**
```python
from sklearn.manifold import trustworthiness

tw = trustworthiness(X_original, X_reduced, n_neighbors=5)
print(f"Trustworthiness (k=5): {tw:.4f}")

# Evaluate at multiple k values
for k in [5, 10, 20, 50]:
    print(f"k={k}: {trustworthiness(X_original, X_reduced, n_neighbors=k):.4f}")
```

**Interpretation:**
- > 0.95: Excellent local structure preservation
- 0.90–0.95: Good
- 0.80–0.90: Moderate
- < 0.80: Poor

**When to use:** When local structure (cluster boundaries) matters for downstream tasks. PCA often scores 0.9+ on simple datasets; lower for complex manifolds.

---

### 8.4 Continuity

**Definition:** The complement of trustworthiness — measures whether the k-nearest neighbors in high-D are also neighbors in low-D. Analogous to recall (trustworthiness is analogous to precision).

**Not in sklearn directly.** Use pyDRMetrics:
```python
from pyDRMetrics.pyDRMetrics import DRMetrics
drm = DRMetrics(X_original, X_reduced)
print(f"Trustworthiness: {drm.T(k=10)}")
print(f"Continuity: {drm.C(k=10)}")
print(f"LCMC: {drm.LCMC(k=10)}")  # Local continuity meta criterion
```

---

### 8.5 Residual Variance

**Definition:** 1 - (correlation between pairwise distances in high-D and low-D)²

**Interpretation:** Fraction of pairwise distance structure NOT captured. Ideal = 0.

**Sklearn-compatible code:**
```python
from sklearn.metrics import pairwise_distances

D_high = pairwise_distances(X_original)
D_low = pairwise_distances(X_reduced)

# Flatten upper triangles
idx = np.triu_indices(len(X_original), k=1)
d_high = D_high[idx]
d_low = D_low[idx]

residual_var = 1 - np.corrcoef(d_high, d_low)[0, 1]**2
print(f"Residual variance: {residual_var:.4f}")
```

---

### 8.6 Downstream Task Performance (Gold Standard)

**The most practically relevant metric:** Does PCA improve or at least not hurt downstream model performance?

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

baseline_score = cross_val_score(
    Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())]),
    X, y, cv=5
).mean()

pca_score = cross_val_score(
    Pipeline([('sc', StandardScaler()), ('pca', PCA(n_components=50)), ('clf', LogisticRegression())]),
    X, y, cv=5
).mean()

print(f"Baseline: {baseline_score:.4f}, PCA: {pca_score:.4f}")
print(f"PCA compression ratio: {X.shape[1]} → 50 features ({50/X.shape[1]*100:.1f}%)")
```

---

### 8.7 Evaluation Summary Table

| Metric | Range | Ideal | Sklearn Built-in? | Use Case |
|---|---|---|---|---|
| Reconstruction Error | [0, ∞) | 0 | Yes (compute manually) | Always check |
| Explained Variance Ratio | [0, 1] | 1 | Yes (`explained_variance_ratio_`) | Linear PCA |
| Trustworthiness | [0, 1] | 1 | Yes (`manifold.trustworthiness`) | When local structure matters |
| Continuity | [0, 1] | 1 | No (pyDRMetrics) | Complements trustworthiness |
| Residual Variance | [0, 1] | 0 | No (compute manually) | Global distance preservation |
| Downstream Accuracy | task-specific | maximize | Via cross-validation | Supervised context |

---

## 9. COMMON PITFALLS WITH CONCRETE EXAMPLES

### Pitfall 1: Forgetting to Standardize
**Example:** Dataset has age (range 20–80) and income (range $20K–$200K). Without StandardScaler, income dominates all principal components (its variance is 10,000× larger than age in raw units). PCA → finds "income variation" as PC1, "income variation" as PC2 again, etc. Age is buried.
**Fix:** `StandardScaler().fit_transform(X)` before PCA.
**Exception:** When all features are on the same scale (e.g., pixel values 0–255, standardized gene expression, returns in %), scaling may not be needed or may hurt by treating noisy features equally.

### Pitfall 2: Fitting PCA on Test Data (Data Leakage)
**Example:** Researcher fits PCA on the full dataset, then splits into train/test. The test set's statistics influenced the PCA projection → overly optimistic test scores.
**Fix:** Always use sklearn Pipeline, or manually `pca.fit(X_train)` then `pca.transform(X_test)`.

### Pitfall 3: Using PCA on Sparse Matrices
**Example:** 10,000-document TF-IDF matrix with 50,000 features. `PCA().fit(X)` causes sklearn to center X (subtract per-feature mean), converting the sparse matrix to a dense one. For 10K × 50K float64, that's 4GB of RAM just for the centered matrix.
**Fix:** Use `TruncatedSVD` for sparse matrices. Never use `PCA` directly on sparse inputs.

### Pitfall 4: Ignoring Sign Ambiguity in PCA Components
**Example:** Fitting PCA on two different random subsets of the same data gives PC1 = [0.7, 0.3, -0.2, ...] for one and [-0.7, -0.3, 0.2, ...] for another. Student reports "PCA is unstable."
**Reality:** PCA components are defined only up to sign. The subspace is identical; just the orientation flipped. This causes problems when comparing PCA results across runs or datasets.
**Fix:** Use `sklearn.utils.extmath.svd_flip` to enforce a convention (largest absolute value element is positive). sklearn does this internally, but sign can still differ between versions.

### Pitfall 5: Expecting KPCA Inverse Transform to Be Exact
**Example:** Student fits `KernelPCA(kernel='rbf', fit_inverse_transform=True)` and computes `X_approx = kpca.inverse_transform(X_kpca)`. Compares X_approx to X and finds large errors, concludes "KernelPCA is broken."
**Reality:** The inverse transform for nonlinear kernels is an approximation (pre-image problem). The true pre-image may not exist or be unique in input space. The ridge regression approximation is imperfect by design.
**Fix:** Accept approximate reconstruction; use reconstruction error only as a relative anomaly score, not as an absolute quality measure.

### Pitfall 6: MiniBatchSparsePCA batch_size Confusion
**Example:** Student sets `MiniBatchSparsePCA(batch_size=32)` expecting 32-sample mini-batches (like every other mini-batch estimator). Actually processes 32 FEATURES per mini-batch.
**Fix:** Check documentation carefully. For MiniBatchSparsePCA, `batch_size` is the number of features per batch. For IncrementalPCA, it's the number of samples per batch.

### Pitfall 7: Categorical Data with PCA
**Example:** Dataset has mixed types: 3 numeric features + 5 one-hot encoded categorical features. Student runs PCA on the full matrix. PCA doesn't distinguish numeric from binary features and gives distorted components.
**Fix:** For purely categorical data, use Multiple Correspondence Analysis (MCA) via the `prince` library. For mixed data, use FAMD (Factor Analysis of Mixed Data).

### Pitfall 8: Treating PCA Components as Independent
**Example:** After PCA, student applies another PCA or a method assuming correlated inputs. PCA output is already decorrelated — applying another PCA is identity operation (does nothing useful).
**Fix:** After `whiten=False` PCA, components are uncorrelated but not independent (higher-order dependencies remain). For full independence, use ICA. For whitened PCA, components are also unit variance but still not independent.

### Pitfall 9: Not Resetting PCA for Cross-Validation
**Example:** Student uses `partial_fit` across all folds and then evaluates — the PCA has seen validation data during fitting.
**Fix:** Create a new PCA estimator inside each fold (or use Pipeline with cross_val_score which handles this correctly).

### Pitfall 10: SparsePCA Is SLOW
**Example:** Student runs `SparsePCA(n_components=20)` on a 10K × 500 matrix and waits 30 minutes.
**Expectation setting:** SparsePCA uses alternating optimization (dictionary learning), which is O(n_samples × n_components × n_features × n_iter). It is 100–1000× slower than regular PCA.
**Mitigation:** Use `MiniBatchSparsePCA` (faster, approximate); reduce `max_iter`; use `n_jobs=-1`; initialize with a standard PCA result via `U_init` and `V_init`.

---

## 10. MATHEMATICAL FOUNDATIONS (QUICK REFERENCE)

### Classical PCA: Two Equivalent Formulations

**1. Variance Maximization:**
Find w₁ = argmax_w Var(Xw) s.t. ||w||=1
Find w₂ = argmax_w Var(Xw) s.t. ||w||=1, w⊥w₁
...

**2. Reconstruction Error Minimization:**
Find W ∈ R^{p×k} to minimize E[||X - XWW^T||²_F] s.t. W^TW = I

**Solution:** Both reduce to the eigenvalue problem: Cov(X)w = λw
Cov(X) = (1/(n-1)) X^T X  (after centering)

**Relationship to SVD:** X = UΣV^T
- Principal components = columns of V (right singular vectors) = rows of components_
- Scores = XV = UΣ  (the output of transform())
- Eigenvalues λ_k = σ_k² / (n-1)

### Kernel PCA: The Kernel Trick

**Key idea:** Replace the dot product ⟨x_i, x_j⟩ with a kernel k(x_i, x_j) = ⟨φ(x_i), φ(x_j)⟩

1. Compute kernel matrix: K_{ij} = k(x_i, x_j)
2. Center K: K̃ = K - 1_n K - K 1_n + 1_n K 1_n  (where 1_n = (1/n) 11^T)
3. Eigendecompose K̃ = V Λ V^T
4. Normalize eigenvectors: α_k = V_k / √λ_k
5. Projection: z_k(x) = Σ_i α_{ik} k(x_i, x)

**Why it works:** The eigenvectors of K̃ in the n-sample dual space correspond exactly to the principal components in the (possibly infinite-dimensional) feature space.

### Randomized PCA: The Algorithm

Given X ∈ R^{m×n}, target rank k, oversampling l:
1. Draw Gaussian random matrix Ω ∈ R^{n×(k+l)}
2. Form sketch Y = XΩ ∈ R^{m×(k+l)}
3. Power iteration (q times): Y = (XX^T)^q XΩ  (with QR stabilization each step)
4. QR decompose Y = QR → keep Q ∈ R^{m×(k+l)}
5. Form B = Q^T X ∈ R^{(k+l)×n}
6. Compute exact SVD of small B: B = ÛΣV^T
7. U = QÛ → top-k columns give principal components

**Error bound:** ||X - U_k Σ_k V_k^T||_F ≤ (1 + ε)||X - X_k||_F with high probability

### Factor Analysis: The Generative Model

X = μ + Wz + ε
- z ~ N(0, I_k): latent factors (low-dimensional)
- ε ~ N(0, Ψ): noise per feature (Ψ diagonal, per-feature variance)
- W ∈ R^{p×k}: loading matrix

**Contrast with PCA:** PCA sets Ψ = σ²I (isotropic noise); FA allows Ψ = diag(ψ₁, ..., ψ_p)

**Estimation:** EM algorithm iterating:
- E-step: E[z|X] using current W, Ψ
- M-step: Update W, Ψ to maximize E[log p(X,z)]

---

## APPENDIX: COMPLETE SKLEARN 1.5 CODE EXAMPLES

### Full Pipeline with PCA
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Build pipeline (prevents data leakage)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50, svd_solver='auto', random_state=42)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Scree plot analysis first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
pca_full = PCA(svd_solver='full').fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cumvar, 0.95) + 1
print(f"Components for 95% variance: {n_95}")

# Cross-validate
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
print(f"PCA + LR accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Incremental PCA for Large Datasets
```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

def load_chunks(filepath, chunk_size=1000):
    """Generator that yields data chunks"""
    # ... implementation
    pass

ipca = IncrementalPCA(n_components=50)

# First pass: fit
for chunk in load_chunks('large_data.npy', chunk_size=1000):
    ipca.partial_fit(chunk)

print(f"Samples seen: {ipca.n_samples_seen_}")
print(f"Variance explained: {ipca.explained_variance_ratio_.sum():.3f}")

# Second pass: transform
X_transformed = []
for chunk in load_chunks('large_data.npy', chunk_size=1000):
    X_transformed.append(ipca.transform(chunk))
X_transformed = np.vstack(X_transformed)
```

### Kernel PCA for Anomaly Detection
```python
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Fit on normal data only
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)

kpca = KernelPCA(
    n_components=20,
    kernel='rbf',
    gamma=0.01,          # Tune this: start with 1/n_features
    fit_inverse_transform=True,
    alpha=0.1,           # Ridge for inverse transform
    random_state=42
)
kpca.fit(X_normal_scaled)

# Anomaly scoring
def kpca_reconstruction_error(X_new):
    X_new_scaled = scaler.transform(X_new)
    X_kpca = kpca.transform(X_new_scaled)
    X_reconstructed = kpca.inverse_transform(X_kpca)
    return np.mean((X_new_scaled - X_reconstructed) ** 2, axis=1)

scores = kpca_reconstruction_error(X_test)
threshold = np.percentile(kpca_reconstruction_error(X_normal), 99)
predictions = (scores > threshold).astype(int)
```

### Factor Analysis with Model Selection
```python
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.model_selection import cross_val_score
from sklearn.covariance import ShrunkCovariance

# Select number of components via cross-validated log-likelihood
n_components_range = range(1, 20)
fa_scores = []
pca_scores = []

for n in n_components_range:
    fa = FactorAnalysis(n_components=n, random_state=0)
    pca = PCA(n_components=n, svd_solver='full')
    
    fa_scores.append(cross_val_score(fa, X_scaled, cv=5).mean())  # Uses FA log-likelihood
    # PCA doesn't have score() for log-likelihood; use FA score as comparison

best_n = n_components_range[np.argmax(fa_scores)]
print(f"Best n_components by FA log-likelihood: {best_n}")

# Final model with rotation for interpretability
fa_best = FactorAnalysis(n_components=best_n, rotation='varimax', random_state=0)
X_fa = fa_best.fit_transform(X_scaled)
```

---

## SOURCES CONSULTED

- scikit-learn 1.5.2 documentation (official)
- Pearson (1901), Philosophical Magazine; DOI: 10.1080/14786440109462720
- Hotelling (1933), Journal of Educational Psychology
- Halko, Martinsson, Tropp (2011), SIAM Review; arXiv: 0909.4061
- Ross et al. (2008), IJCV; DOI: 10.1007/s11263-007-0075-7
- Schölkopf, Smola, Müller (1998), Neural Computation; DOI: 10.1162/089976698300017467
- Zou, Hastie, Tibshirani (2006), JCGS; DOI: 10.1198/106186006X113430
- Turk & Pentland (1991), Journal of Cognitive Neuroscience; DOI: 10.1162/jocn.1991.3.1.71
- Deerwester et al. (1990), JASIS; DOI: 10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9
- Minka (2001), NIPS: "Automatic choice of dimensionality for PCA"
- Fang et al. (2024), NeurIPS: "Kernel PCA for Out-of-Distribution Detection"; arXiv: 2402.02949
- Federal Reserve (2025): "Portfolio Margining Using PCA Latent Factors"
- pyDRMetrics (2021), Heliyon; PMC7887408
- prince 0.19.0 documentation (PyPI, May 2026)
- fbpca 1.0 documentation (PyPI)
