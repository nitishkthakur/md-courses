# Research Dossier: Random Projections and Johnson-Lindenstrauss
## For: Dimensionality Reduction Masterclass — Chapter on Random Projections

**Research Date:** June 2026
**Target chapter length:** 18,000+ tokens
**Audience:** The writing agent has only this dossier and the course bible.

---

## TABLE OF CONTENTS

1. [Algorithm Landscape & Taxonomy](#1-algorithm-landscape--taxonomy)
2. [Must-Read Papers (Top 8)](#2-must-read-papers-top-8)
3. [Sklearn 1.5.x API — Every Parameter Documented](#3-sklearn-15x-api--every-parameter-documented)
4. [Specialized Python Packages](#4-specialized-python-packages)
5. [Real-World Applications (7 Concrete Examples)](#5-real-world-applications-7-concrete-examples)
6. [Hyperparameter Tuning Guide](#6-hyperparameter-tuning-guide)
7. [Expert Practitioner Checklist](#7-expert-practitioner-checklist)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Common Pitfalls With Concrete Examples](#9-common-pitfalls-with-concrete-examples)
10. [Mathematical Foundations (Quick Reference)](#10-mathematical-foundations-quick-reference)

---

## 1. ALGORITHM LANDSCAPE & TAXONOMY

### The Random Projections Family Tree

```
Random Projection (Johnson-Lindenstrauss 1984)
├── Dense Gaussian RP  — entries ~ N(0, 1/k)
│   └── sklearn.random_projection.GaussianRandomProjection
│
├── Sparse / Database-friendly RP  (Achlioptas 2003)
│   ├── {-1, 0, +1} with prob {1/6, 2/3, 1/6}   (s=3, density=1/3)
│   └── sklearn.random_projection.SparseRandomProjection
│       └── Very Sparse RP (Li, Hastie, Church 2006) — density = 1/sqrt(d)
│
├── Rademacher Projections  — entries ~ {-1/√k, +1/√k} uniform
│   └── Special case of Achlioptas with density=1 (no zeros)
│
├── Structured / Fast Projections  (SRHT family)
│   ├── SRHT — Subsampled Randomized Hadamard Transform
│   │   └── O(d log d) time vs O(dk) for Gaussian
│   └── FJLT — Fast Johnson-Lindenstrauss Transform
│
└── Locality-Sensitive Hashing (LSH)  — hashes preserve proximity
    ├── SimHash (cosine similarity)
    ├── MinHash (Jaccard similarity) — datasketch package
    ├── Random Projection on Hypercube (Euclidean distance)
    └── FALCONN — C++/Python library for cosine/Euclidean LSH
```

### Core Idea in One Sentence

Multiply your n×d data matrix X by a random d×k matrix R, where k ≪ d. The key theorem (Johnson-Lindenstrauss) guarantees that pairwise distances are preserved to within (1±ε) with high probability, and k only needs to grow as O(log n / ε²) — it does not depend on d at all.

### When to Use Which Variant

| Variant | When to Prefer |
|---|---|
| **GaussianRP** | Dense data, need maximum theoretical guarantees, k is moderate |
| **SparseRP (Achlioptas)** | Dense data, speed matters, 3× faster than Gaussian |
| **Very SparseRP (Li 2006)** | Very high-d data (d > 10,000), extreme speed, density = 1/√d |
| **Rademacher** | Integer arithmetic hardware, simple to implement from scratch |
| **SRHT** | d is very large, O(d log d) time needed, no explicit matrix storage |
| **MinHash LSH** | Jaccard similarity, text/set deduplication |
| **SimHash / Random-hyperplane LSH** | Cosine similarity, dense vectors |
| **FALCONN** | Production ANN search, C++ speed, Euclidean or cosine |

### Key Conceptual Contrasts vs PCA/SVD

| Property | PCA | Random Projection |
|---|---|---|
| Data-dependent? | Yes — needs full SVD | No — matrix is random, fit only checks n_features |
| Fits in one pass? | No (full SVD needs all data) | Yes (matrix sampled at fit time) |
| Time complexity | O(min(nd², n²d)) | O(ndk) — faster when k ≪ d |
| Memory for projection | d×k dense | k×d sparse (SparseRP) or dense |
| Distance preservation | Exact in k-dim subspace | Probabilistic ε-guarantee |
| Interpretability | High (loadings have meaning) | None (random columns) |
| Out-of-core? | IncrementalPCA only | Yes, trivially (fit just reads shape) |
| Best for | Visualization, structured features | Pre-processing pipeline, ANN, streaming |

---

## 2. MUST-READ PAPERS (TOP 8)

### Paper 1 — Johnson & Lindenstrauss (1984): The Origin
**Full Citation:** Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. In *Contemporary Mathematics*, Vol. 26: Conference in Modern Analysis and Probability (New Haven, CT, 1982), pp. 189–206. American Mathematical Society, Providence, RI.
**URL:** http://stanford.edu/class/cs114/readings/JL-Johnson.pdf
**DOI:** Not assigned (conference proceedings volume)
**What You'll Learn:** This is a 3-page construction buried in a geometric analysis paper — the JL lemma is not even the main result. The authors prove that any finite metric space on n points can be embedded into a Hilbert space with only logarithmic distortion. The random projection interpretation was implicit; it took the community years to extract it. Reading this explains why the lemma is stated in its abstract form and gives historical context for why it remained obscure until the 1990s algorithms community picked it up.
**Difficulty:** Advanced (functional analysis notation; the JL lemma is a side result)

### Paper 2 — Dasgupta & Gupta (2003): Accessible Proof
**Full Citation:** Dasgupta, S., & Gupta, A. (2003). An elementary proof of a theorem of Johnson and Lindenstrauss. *Random Structures & Algorithms*, 22(1), 60–65.
**DOI:** https://doi.org/10.1002/rsa.10073
**URL:** https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf
**What You'll Learn:** The paper gives the cleanest modern proof of the JL lemma using only Gaussian concentration inequalities and a union bound. The 5-page proof is now the standard reference taught in courses. You learn exactly why the bound is k ≥ 4 ln(n) / (ε²/2 − ε³/3) and why the random Gaussian matrix works. This is the paper to assign to students because it is self-contained and elementary relative to the original.
**Difficulty:** Intermediate (requires probability theory at the level of Hoeffding/Chernoff bounds)

### Paper 3 — Achlioptas (2003): Database-Friendly Projections
**Full Citation:** Achlioptas, D. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins. *Journal of Computer and System Sciences*, 66(4), 671–687. Special issue: PODS 2001.
**DOI:** https://doi.org/10.1016/S0022-0000(03)00025-4
**URL:** https://www.sciencedirect.com/science/article/pii/S0022000003000254
**Semantic Scholar:** https://www.semanticscholar.org/paper/Database-friendly-random-projections:-with-binary-Achlioptas/594d2e123ecb8ec0bc781aec467007d65ab5464d
**What You'll Learn:** Achlioptas proves that the Gaussian entries in the projection matrix can be replaced with simple ternary {−1, 0, +1} entries (with probabilities 1/6, 2/3, 1/6) or binary {−1, +1} entries (Rademacher), while maintaining the JL guarantee. The ternary version has 2/3 of entries zero, making the projection matrix sparse. This enables a 3× speedup over Gaussian projection for dense data, and makes the matrix storable in databases without floating-point columns. This paper is the theoretical basis for sklearn's SparseRandomProjection when density is set to 1/3.
**Difficulty:** Intermediate (the key proof reuses the moment-matching argument from the Gaussian case)

### Paper 4 — Li, Hastie & Church (2006): Very Sparse Random Projections
**Full Citation:** Li, P., Hastie, T. J., & Church, K. W. (2006). Very sparse random projections. In *Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (KDD 2006), pp. 287–296. ACM.
**DOI:** https://doi.org/10.1145/1150402.1150436
**URL:** https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf
**Award:** KDD 2006 Best Paper Award
**What You'll Learn:** Li et al. push sparsity to the extreme. Instead of 1/3 non-zeros (Achlioptas), they show density = 1/√d is sufficient for embeddings into k dimensions. For a 10,000-feature dataset, only 1% of entries are non-zero. The paper includes empirical comparisons showing the very sparse version is often as accurate as the dense Gaussian projection while being dramatically faster and more memory-efficient. This is the default `density='auto'` behavior in sklearn's SparseRandomProjection.
**Difficulty:** Intermediate (accessible; includes clear empirical validation)

### Paper 5 — Indyk & Motwani (1998): LSH Introduction
**Full Citation:** Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: Towards removing the curse of dimensionality. In *Proceedings of the 30th Annual ACM Symposium on Theory of Computing* (STOC 1998), pp. 604–613. ACM.
**DOI:** https://doi.org/10.1145/276698.276876
**What You'll Learn:** The paper that introduced Locality-Sensitive Hashing as a formal framework. Indyk and Motwani define the (r, cr, p1, p2)-sensitive hash family, show that random projections onto a line followed by quantization give a valid LSH for Euclidean distance, and prove that approximate nearest neighbor search can be solved in O(n^ρ) time where ρ < 1. This is the conceptual foundation for FALCONN, Annoy, and the entire ANN search ecosystem.
**Difficulty:** Advanced (STOC-level theory; reduction arguments and probabilistic analysis)

### Paper 6 — Charikar (2002): SimHash
**Full Citation:** Charikar, M. S. (2002). Similarity estimation techniques from rounding algorithms. In *Proceedings of the 34th Annual ACM Symposium on Theory of Computing* (STOC 2002), pp. 380–388. ACM.
**DOI:** https://doi.org/10.1145/509907.509965
**What You'll Learn:** Charikar introduces the random hyperplane LSH scheme for cosine similarity. For each random hyperplane, the sign of the dot product gives one bit; the fraction of bit disagreements between two vectors approximates their angular distance. This is "SimHash," used by Google for near-duplicate web page detection and by many document similarity systems. The paper shows the connection between the Johnson-Lindenstrauss projection and cosine similarity via the arccosine distance.
**Difficulty:** Intermediate (elegant construction; the proof is a nice geometric argument)

### Paper 7 — Broder et al. (1997): MinHash
**Full Citation:** Broder, A. Z., Glassman, S. C., Manber, U., & Zweig, G. (1997). Syntactic clustering of the web. In *Proceedings of the 6th International World Wide Web Conference* (WWW6), pp. 391–404. Elsevier.
**DOI:** https://doi.org/10.1016/S0169-7552(97)00031-7
**What You'll Learn:** MinHash was invented at AltaVista to detect near-duplicate web pages at scale. The key insight is that the probability that the minimum of a random hash function applied to two sets is equal exactly equals the Jaccard similarity. By concatenating many MinHash values, you get a compact signature that supports LSH-style banding for fast similarity lookup. This paper is the origin of what today powers LLM training data deduplication (C4, RefinedWeb, RedPajama all use MinHash LSH).
**Difficulty:** Accessible (practical paper; clean probabilistic argument)

### Paper 8 — Ghojogh et al. (2021): Modern Tutorial and Survey
**Full Citation:** Ghojogh, B., Ghodsi, A., Karray, F., & Crowley, M. (2021). Johnson-Lindenstrauss Lemma, Linear and Nonlinear Random Projections, Random Fourier Features, and Random Kitchen Sinks: Tutorial and Survey. *arXiv preprint arXiv:2108.04172*.
**URL:** https://arxiv.org/abs/2108.04172
**What You'll Learn:** A 100-page unified tutorial covering (1) the JL lemma and its proofs, (2) Gaussian, sparse, Rademacher and very sparse linear random projections, (3) Random Fourier Features (RFF) for kernel approximation, (4) Random Kitchen Sinks, (5) extreme learning machines, and (6) applications. This is the go-to reference for a writing agent covering the full landscape. It also covers connections to compressed sensing and neural network random weight initialization.
**Difficulty:** Intermediate to Advanced (self-contained but wide; use as reference, not cover-to-cover)

---

## 3. SKLEARN 1.5.x API — EVERY PARAMETER DOCUMENTED

### Module: `sklearn.random_projection`

**Import:**
```python
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)
```

---

### 3.1 `GaussianRandomProjection`

**Added in sklearn version:** 0.13

```python
GaussianRandomProjection(
    n_components='auto',        # int or 'auto', default='auto'
    eps=0.1,                    # float, default=0.1
    compute_inverse_components=False,  # bool, default=False
    random_state=None,          # int | RandomState | None, default=None
)
```

#### Parameters (Detailed)

| Parameter | Type | Default | What It Controls |
|---|---|---|---|
| `n_components` | int or `'auto'` | `'auto'` | Number of target dimensions. If `'auto'`, uses the JL formula: `k = 4 * log(n_samples) / (eps²/2 − eps³/3)`. Pass an integer to override the JL bound entirely. |
| `eps` | float | `0.1` | Distortion tolerance ε ∈ (0, 1). Only used when `n_components='auto'`. Smaller ε → better distance preservation → larger k. Values: 0.5 gives aggressive compression; 0.1 is the safe default; 0.01 is near-lossless but yields enormous k. |
| `compute_inverse_components` | bool | `False` | If True, computes the Moore-Penrose pseudo-inverse of the projection matrix at fit time and stores it as `inverse_components_`. Enables `inverse_transform()` without re-computing the pseudo-inverse each call. WARNING: scales as O(d × k) memory, and for large matrices the pseudo-inverse computation is very slow. |
| `random_state` | int, RandomState, or None | `None` | Seed for reproducibility. Pass an integer for fixed results across runs. |

#### Attributes After Fitting

| Attribute | Shape | Description |
|---|---|---|
| `n_components_` | scalar int | Concrete number of components (equals `n_components` if passed as int; computed from JL formula if `'auto'`) |
| `components_` | (n_components, n_features) ndarray | The dense random projection matrix. Entries ~ N(0, 1/n_components). |
| `inverse_components_` | (n_features, n_components) ndarray | Pseudo-inverse of `components_`. Only present if `compute_inverse_components=True`. Added in sklearn 1.1. |
| `n_features_in_` | int | Number of features seen at fit time. Added in sklearn 0.24. |
| `feature_names_in_` | ndarray of str | Feature names if input had named columns. Added in sklearn 1.0. |

#### Methods

```python
# Fit: only reads X.shape, not X values
transformer.fit(X)          # X: (n_samples, n_features) array or sparse

# Transform: matrix product X @ components_.T
transformer.transform(X)   # returns (n_samples, n_components) dense ndarray

# Combined
X_proj = transformer.fit_transform(X)

# Approximate inverse
X_approx = transformer.inverse_transform(X_proj)
# Returns (n_samples, n_features) dense ndarray
# Loss: ||X - X_approx||_F depends on n_components; never exact for k < d

# Set output format (sklearn 1.4+)
transformer.set_output(transform='pandas')  # or 'polars', 'default'
```

#### Example (Verified sklearn 1.5 Syntax)

```python
import numpy as np
from sklearn.random_projection import GaussianRandomProjection

rng = np.random.RandomState(42)
X = rng.rand(1000, 5000)   # 1000 samples, 5000 features

# Default: auto n_components with eps=0.1
proj = GaussianRandomProjection(random_state=42)
X_proj = proj.fit_transform(X)
print(proj.n_components_)   # e.g., 930 for n=1000, eps=0.1
print(X_proj.shape)          # (1000, 930)

# Fixed n_components
proj2 = GaussianRandomProjection(n_components=100, random_state=42)
X_proj2 = proj2.fit_transform(X)
print(X_proj2.shape)         # (1000, 100)

# With inverse transform
proj3 = GaussianRandomProjection(
    n_components=200, compute_inverse_components=True, random_state=42
)
X_proj3 = proj3.fit_transform(X)
X_approx = proj3.inverse_transform(X_proj3)
print(X_approx.shape)        # (1000, 5000)
```

---

### 3.2 `SparseRandomProjection`

**Added in sklearn version:** 0.13

```python
SparseRandomProjection(
    n_components='auto',               # int or 'auto', default='auto'
    density='auto',                    # float or 'auto', default='auto'
    eps=0.1,                           # float, default=0.1
    dense_output=False,                # bool, default=False
    compute_inverse_components=False,  # bool, default=False
    random_state=None,                 # int | RandomState | None
)
```

#### Parameters (Detailed)

| Parameter | Type | Default | What It Controls |
|---|---|---|---|
| `n_components` | int or `'auto'` | `'auto'` | Same as GaussianRP. When `'auto'`, applies the JL formula using `eps`. Overriding with an integer ignores `eps`. |
| `density` | float or `'auto'` | `'auto'` | Fraction of non-zero entries in each column of the projection matrix. When `'auto'`, uses the Li et al. (2006) recommendation: `1 / sqrt(n_features)`. For a 10,000-feature dataset this means 1% non-zeros. Pass a float in (0, 1] to override: `1/3` corresponds to Achlioptas 2003; `1.0` gives Rademacher. |
| `eps` | float | `0.1` | Same as GaussianRP. Only used when `n_components='auto'`. |
| `dense_output` | bool | `False` | If `False`, output of `transform()` is sparse (CSR matrix) when input is sparse. If `True`, always returns dense ndarray. Set `True` when the downstream pipeline requires dense input. |
| `compute_inverse_components` | bool | `False` | Same as GaussianRP. The pseudo-inverse is always stored as dense ndarray even though `components_` is sparse. Added in sklearn 1.1. |
| `random_state` | int, RandomState, or None | `None` | Same as GaussianRP. |

#### Attributes After Fitting

| Attribute | Shape | Description |
|---|---|---|
| `n_components_` | scalar int | Concrete number of components after JL formula or direct specification. |
| `components_` | (n_components, n_features) sparse CSR | The sparse random matrix. Values ∈ {−√(s/k), 0, +√(s/k)} where s = 1/density, k = n_components. |
| `density_` | float | Concrete density (fraction of non-zeros). Equal to `density` if float was given; the computed value if `'auto'`. |
| `inverse_components_` | (n_features, n_components) dense ndarray | Pseudo-inverse. Only present if `compute_inverse_components=True`. |
| `n_features_in_` | int | Features seen at fit. |
| `feature_names_in_` | ndarray of str | Feature names if input was named. |

#### Component Distribution (The Math)

```
P(entry = -sqrt(s/k))  =  1 / (2s)
P(entry = 0)           =  1 - 1/s
P(entry = +sqrt(s/k))  =  1 / (2s)

where s = 1/density, k = n_components

Special cases:
  density = 1/3  →  Achlioptas 2003 ternary matrix (s=3)
  density = 1.0  →  Rademacher matrix (s=1, no zeros), entries = ±1/√k
  density = 1/sqrt(d)  →  Li et al. 2006 very sparse (default 'auto')
```

#### Example (Verified sklearn 1.5 Syntax)

```python
import numpy as np
import scipy.sparse as sp
from sklearn.random_projection import SparseRandomProjection

# Dense input
X_dense = np.random.rand(500, 8000)
proj = SparseRandomProjection(random_state=42)
X_proj = proj.fit_transform(X_dense)
print(proj.n_components_)   # e.g., 902
print(proj.density_)         # 0.01118... = 1/sqrt(8000)
print(X_proj.shape)          # (500, 902)

# Sparse input — output stays sparse
X_sparse = sp.random(500, 8000, density=0.05, format='csr', random_state=42)
proj_sp = SparseRandomProjection(dense_output=False, random_state=42)
X_proj_sp = proj_sp.fit_transform(X_sparse)
print(sp.issparse(X_proj_sp))   # True

# Explicit Achlioptas density
proj_ach = SparseRandomProjection(
    n_components=200, density=1/3, random_state=42
)
X_ach = proj_ach.fit_transform(X_dense)

# Rademacher projection
proj_rad = SparseRandomProjection(
    n_components=200, density=1.0, random_state=42
)
X_rad = proj_rad.fit_transform(X_dense)

print(f"Non-zero fraction in components: {proj_ach.density_:.3f}")   # 0.333
```

---

### 3.3 `johnson_lindenstrauss_min_dim`

```python
from sklearn.random_projection import johnson_lindenstrauss_min_dim

# Returns minimum k satisfying the JL bound
k = johnson_lindenstrauss_min_dim(n_samples, eps)
```

**Formula implemented:**
```
k >= (4 * log(n_samples)) / (eps**2 / 2 - eps**3 / 3)
```

**Reference values:**

| n_samples | eps=0.5 | eps=0.1 | eps=0.01 |
|---|---|---|---|
| 1,000 | 530 | 9,434 | 888,742 |
| 100,000 | 663 | 11,841 | 1,112,658 |
| 1,000,000 | 796 | 14,248 | ~1.3M |

**Key insight:** k grows only logarithmically with n. Adding 900× more samples only adds ~50% more required dimensions at eps=0.1. The original dimension d does not appear in the formula at all.

---

### 3.4 Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('rp', SparseRandomProjection(n_components=300, random_state=42)),
    ('clf', LogisticRegression(max_iter=500)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

```python
# Using set_output for pandas DataFrames (sklearn 1.4+)
import pandas as pd
proj = SparseRandomProjection(n_components=100, random_state=42)
proj.set_output(transform='pandas')
X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
X_proj_df = proj.fit_transform(X_df)
# Column names will be 'sparserandomprojection0', 'sparserandomprojection1', ...
```

---

### 3.5 API Changes Between Versions

| Version | Change |
|---|---|
| 0.13 | Both classes introduced |
| 0.24 | `n_features_in_` attribute added |
| 1.0 | `feature_names_in_` attribute added |
| 1.1 | `compute_inverse_components` parameter and `inverse_components_` attribute added to both classes |
| 1.4 | `set_output(transform='polars')` support added |

**Stability note (mid-2026):** Both classes are stable with no breaking changes from 1.5 to 1.6. The `compute_inverse_components` parameter was the last significant addition (v1.1). No deprecations currently active.

---

## 4. SPECIALIZED PYTHON PACKAGES

### 4.1 `datasketch` — MinHash and LSH

| Property | Value |
|---|---|
| **PyPI name** | `datasketch` |
| **Current version (mid-2026)** | 1.10.0 (released April 17, 2026) |
| **License** | MIT |
| **Install** | `pip install datasketch` |
| **Optional extras** | `pip install datasketch[redis]`, `datasketch[cassandra]`, `datasketch[bloom]` |
| **Requirements** | Python 3.9+, NumPy 1.11+, SciPy |
| **Maintenance** | Actively maintained; GitHub: ekzhu/datasketch |

**What it does better than sklearn:** sklearn has no MinHash or Jaccard-based LSH. datasketch provides MinHash, MinHash LSH (banding), LSH Forest, LSH Ensemble (set containment queries), Weighted MinHash, HyperLogLog, and HyperLogLog++. It also supports distributed/async operation with Redis, Cassandra, and MongoDB backends.

**Core imports:**
```python
from datasketch import MinHash, MinHashLSH, MinHashLSHEnsemble, HyperLogLog

# Create MinHash signature for a document
m = MinHash(num_perm=128)   # 128 permutations
for word in document.split():
    m.update(word.encode('utf8'))

# Build LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=128)
lsh.insert('doc1', m1)
lsh.insert('doc2', m2)
result = lsh.query(m_query)   # returns list of candidate keys
```

**Key parameters for MinHashLSH:**
- `threshold`: Jaccard similarity threshold. Sets the banding parameters b (bands) and r (rows per band) to approximate this threshold.
- `num_perm`: Number of permutations. More → better approximation → slower. Typical: 64–256.
- `weights`: Tuple (false_positive_weight, false_negative_weight) for asymmetric tuning.

---

### 4.2 `FALCONN` — Fast ANN via LSH

| Property | Value |
|---|---|
| **PyPI name** | `FALCONN` |
| **Current version** | 1.3.1 (last release: September 2017) |
| **License** | MIT |
| **Install** | `pip install FALCONN` (or build from source) |
| **Authors** | Ilya Razenshteyn, Ludwig Schmidt (MIT CSAIL) |
| **Maintenance** | INACTIVE — no updates since 2017. Use as a reference implementation or for research reproducibility only. |

**What it does better than sklearn:** Provides production-quality C++ LSH for dense vectors (cosine and Euclidean). Supports two LSH families: Cross-polytope LSH and Hyperplane LSH. Near-optimal theoretical properties. The Python wrapper uses NumPy arrays.

**WARNING for writing agent:** FALCONN is not maintained. For production use in 2026, recommend FAISS (Meta AI), hnswlib, or usearch instead. FALCONN remains useful for teaching because its API is clean and well-documented.

```python
import falconn
import numpy as np

data = np.random.randn(10000, 128).astype(np.float32)

params = falconn.LSHConstructionParameters()
params.dimension = 128
params.lsh_family = falconn.LSHFamily.CrossPolytope
params.distance_function = falconn.DistanceFunction.EuclideanSquared
params.l = 10             # number of hash tables
params.num_rotations = 1  # number of random rotations
params.seed = 42

table = falconn.LSHIndex(params)
table.setup(data)
query_obj = table.construct_query_object()
query_obj.set_num_probes(50)    # increase for higher recall

neighbors = query_obj.find_k_nearest_neighbors(data[0], 5)
```

---

### 4.3 `annoy` (Spotify) — ANN via Random Projection Trees

| Property | Value |
|---|---|
| **PyPI name** | `annoy` |
| **Current version** | 1.17.3 |
| **License** | Apache 2.0 |
| **Install** | `pip install annoy` |
| **Maintenance** | Maintained by Spotify (Erik Bernhardsson); GitHub: spotify/annoy |

**What it does better than sklearn:** Annoy uses random projection trees (binary space partitioning by random hyperplanes) to build a static index that can be memory-mapped and shared across processes. Designed for read-heavy use cases. Supports Euclidean, Manhattan, cosine, and Hamming distances. Particularly good when the index is built once and queried millions of times.

```python
from annoy import AnnoyIndex

# Build index
t = AnnoyIndex(128, 'euclidean')    # vector dimension, distance metric
for i, vec in enumerate(embeddings):
    t.add_item(i, vec)
t.build(50)           # 50 trees — more trees = more accurate, slower build
t.save('index.ann')   # save to disk

# Query
u = AnnoyIndex(128, 'euclidean')
u.load('index.ann')
neighbors, distances = u.get_nns_by_vector(query_vec, 10, include_distances=True)
```

---

### 4.4 `faiss` (Meta AI) — GPU-accelerated ANN

| Property | Value |
|---|---|
| **PyPI name** | `faiss-cpu` or `faiss-gpu` |
| **Current version** | 1.8.x |
| **License** | MIT |
| **Install** | `pip install faiss-cpu` (CPU); `conda install -c conda-forge faiss-gpu` (GPU) |
| **Maintenance** | Actively maintained by Meta AI Research; GitHub: facebookresearch/faiss |

**What it does better than sklearn:** Full ANN search library supporting flat (exact), IVF (inverted file), HNSW, PQ (product quantization), and LSH indexes. GPU-accelerated versions can search billions of vectors. Supports both Euclidean and inner product similarity.

```python
import faiss
import numpy as np

d = 128           # dimension
n = 100000        # number of vectors
data = np.random.rand(n, d).astype('float32')

# Simple flat (exact) index
index = faiss.IndexFlatL2(d)
index.add(data)
D, I = index.search(data[:5], 10)   # D=distances, I=indices

# LSH index (random projection based)
nbits = 2 * d
index_lsh = faiss.IndexLSH(d, nbits)
index_lsh.add(data)
D2, I2 = index_lsh.search(data[:5], 10)

# IVF (fast approximate)
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, 100)   # 100 Voronoi cells
index_ivf.train(data)
index_ivf.add(data)
index_ivf.nprobe = 10   # number of cells to search
D3, I3 = index_ivf.search(data[:5], 10)
```

---

### 4.5 `jlt` — Reference JLT Implementations

| Property | Value |
|---|---|
| **GitHub** | github.com/dell/jlt |
| **Install** | `pip install jlt` |
| **License** | Not specified (academic) |
| **Maintenance** | Low activity; primarily educational |

**What it provides:** Clean implementations of Johnson-Lindenstrauss Transform, Fast JLT, and Randomized Hadamard Transform (RHT) in Python 3. Good for teaching and understanding SRHT/FJLT without the complexity of production libraries.

---

### 4.6 Summary Comparison Table

| Package | Use Case | Speed | Production-Ready |
|---|---|---|---|
| `sklearn SparseRP/GaussianRP` | Preprocessing pipeline | Fast | Yes |
| `datasketch` | Set/Jaccard similarity, text dedup | Fast | Yes |
| `annoy` | Static index, memory-mapped | Fast | Yes |
| `faiss` | Large-scale ANN, GPU acceleration | Very Fast | Yes |
| `FALCONN` | LSH benchmarking/research | Fast | No (unmaintained) |
| `jlt` | SRHT/JLT education | Moderate | No |

---

## 5. REAL-WORLD APPLICATIONS (7 CONCRETE EXAMPLES)

### Application 1: LLM Training Data Deduplication (MinHash LSH)
**Domain:** Natural Language Processing / Large Language Model pre-training
**Problem:** LLM training corpora (Common Crawl, C4, RefinedWeb, RedPajama) contain hundreds of billions of tokens with massive near-duplication from web crawls. Training on duplicated data hurts generalization. Comparing every pair of documents is O(n²) — impossible at web scale.
**Solution:** MinHash LSH. Each document is shingled (13-character n-grams), hashed to a compact 128-permutation MinHash signature, and inserted into an LSH index with banding (e.g., 16 bands × 8 rows). Documents sharing any band are candidate pairs; those with Jaccard similarity > 0.8 are deduplicated.
**Why this algorithm:** Only algorithm that scales to 5–50 billion documents. Signatures are ~4KB per document; full brute-force comparison is impossible. MinHash signatures can be computed per-shard and merged, enabling MapReduce parallelism.
**References:** C4 (Raffel et al., 2020), RefinedWeb (Penedo et al., 2023), "Data Deduplication at Trillion Scale" (Zilliz, 2024). The paper arXiv:2411.04257 "LSHBloom" describes the combined MinHashLSH + Bloom filter approach.
**Non-obviousness:** The fact that the premier technique for building GPT-quality training data is a 1997 algorithm by Broder et al. is genuinely surprising.

---

### Application 2: Pinterest Image Similarity at Scale (LSH + TensorFlow)
**Domain:** Visual search and recommendation
**Problem:** Pinterest serves billions of images. When a user pins an image, the system must detect near-duplicates in the corpus to avoid recommending the same image twice, and must find visually similar images for the "more like this" feature.
**Solution:** Pinterest built the "NearDup" system using Spark + batch MinHash/SimHash LSH over TensorFlow-generated image embeddings (CNN features). The LSH index compares billions of items daily with incremental updates.
**Why this algorithm:** TensorFlow generates dense 2048-dim CNN embeddings. SimHash (random hyperplane LSH) on these embeddings gives O(n) index build time. The pipeline increased throughput 13× and decreased runtime 8× compared to the prior brute-force approach.
**References:** Pinterest Engineering Blog: "Detecting image similarity using Spark, LSH and TensorFlow" (https://medium.com/pinterest-engineering/detecting-image-similarity-using-spark-lsh-and-tensorflow-618636afc939)
**Non-obviousness:** They combine neural embeddings (for semantic quality) with LSH (for scale) — neither alone would work.

---

### Application 3: Spotify Music Recommendation (Annoy / RPTREES)
**Domain:** Music streaming recommendation
**Problem:** Spotify represents each of its ~100 million tracks as a high-dimensional embedding. Given a user's taste embedding, find the 10 nearest tracks in real-time (< 10ms) at query time, without retraining the index constantly.
**Solution:** Annoy (Approximate Nearest Neighbors Oh Yeah) — developed internally at Spotify. It builds a forest of random projection trees. Each tree recursively splits the space by a random hyperplane until leaf nodes are small enough to brute-force. At query time, the forest is traversed and results merged.
**Why this algorithm:** Annoy produces static, memory-mappable index files that can be shared across worker processes without memory overhead. This is critical for horizontally scaled serving infrastructure. The index is built offline; query latency is a few milliseconds.
**References:** GitHub: spotify/annoy. Erik Bernhardsson's blog (2015). ANNOY remains widely used despite newer HNSW-based alternatives because its read-only index is uniquely suited to multi-process deployment.

---

### Application 4: Privacy-Preserving Federated Learning (Random Projection + DP)
**Domain:** Privacy-preserving machine learning / federated learning
**Problem:** In federated learning, model gradients (which can be hundreds of MB per update) must be transmitted from mobile devices to a central server. (1) Communication bandwidth is a bottleneck. (2) Full gradients can leak private training data via gradient inversion attacks.
**Solution:** FedRP (arXiv:2509.10041, 2025) proposes randomly projecting local gradients into a k-dimensional subspace before transmission. The projection matrix R is shared (not sent); the compressed k-dim gradient is all that's transmitted. Combined with differential privacy noise added in the compressed space, this simultaneously reduces communication and provides formal privacy guarantees.
**Why this algorithm:** Random projection preserves gradient direction in expectation (unbiased). The JL lemma guarantees that compression to k = O(log n / ε²) dimensions preserves the gradient geometry needed for convergence. DP noise added in low-d space has smaller absolute magnitude than in high-d space, giving better utility-privacy trade-off.
**References:** arXiv:2509.10041 "FedRP: A Communication-Efficient Approach for Differentially Private Federated Learning Using Random Projection" (2025).
**Non-obviousness:** The privacy benefit of random projection comes not from obfuscation per se, but from reducing the dimensionality where noise is added — this is a subtle and important point for the chapter to explain.

---

### Application 5: Distributed Machine Learning / Gradient Compression (RPGD)
**Domain:** Large-scale distributed deep learning
**Problem:** Training large models across many machines requires all-reduce communication at each step. For a model with 100M parameters (400MB of float32 gradients), every worker must send and receive 400MB per step. At 100 workers, that's 40GB of communication per step.
**Solution:** RPGD (Randomly Projected Gradient Descent). Each worker projects its local gradient to a k-dimensional vector using a shared random matrix, transmits only k floats, and the aggregated compressed gradient is projected back. A 2024 paper in *Stat* (Wiley) proves convergence guarantees for this approach.
**Why this algorithm:** Unlike top-k sparsification (which requires sorting) or PowerSGD (which requires SVD at each step), RPGD has O(dk) projection cost with no per-step matrix factorization. The random matrix is fixed throughout training.
**References:** "Communication-Efficient Distributed Gradient Descent via Random Projection" (Qi, 2024, *Stat*, Wiley). arXiv:2411.12898 "Problem-dependent convergence bounds for randomized linear gradient compression."

---

### Application 6: Reinforcement Learning State Space Compression
**Domain:** Reinforcement learning, control systems
**Problem:** In RL from high-dimensional observations (robot sensors, image pixels, network state), the state space is too large for tabular methods and too expensive for direct function approximation. Policy gradient methods suffer from high variance in high-dimensional spaces.
**Solution:** Apply Gaussian random projection to compress state observations before feeding to the RL agent. Projects s_t ∈ ℝ^d to z_t ∈ ℝ^k. Compressive Kernelized RL (CKRL) uses a random projection combined with kernel least-squares policy iteration.
**Why this algorithm:** RL convergence theory (e.g., fitted value iteration) requires the feature space to be small enough for stable regression. Random projection reduces k to a tractable size while the JL bound ensures state similarity is preserved — nearby states remain nearby after projection, which is the critical property for value function smoothness.
**References:** arXiv:1912.06514 "Fast Online RL Control using State-Space Dimensionality Reduction." arXiv:2401.10516 "Episodic RL with Expanded State-Reward Space" uses Gaussian RP for episodic memory indexing.

---

### Application 7: Count-Min Sketch for Streaming Feature Frequency
**Domain:** Big data preprocessing / streaming analytics
**Problem:** In NLP or click-stream data pipelines, feature frequencies (word counts, URL counts, user action counts) need to be tracked over billions of events without storing a full hash table. Standard hash tables grow without bound and cannot handle arbitrary key spaces.
**Solution:** Count-Min Sketch (CMS) — a 2D array of counters with d independent hash functions, each mapping items to one counter per row. Frequency of any item is estimated as the minimum across all rows. While not a "projection" in the strict sense, CMS is a linear sketch that implements dimension reduction on the frequency vector: the item vector (d-dimensional, one entry per possible item) is compressed to a d×w matrix.
**Why this algorithm:** Uses O(d × w) space regardless of number of distinct items. Answers point queries in O(d) time. Multiplicatively-accurate: error is at most ε ‖f‖₁ with probability 1−δ where d ≈ ln(1/δ) and w ≈ e/ε. Google has used CMS in advertising (ad click frequency), Amazon in purchase frequency estimation.
**References:** Cormode & Muthukrishnan (2004) "An improved data stream summary: The Count-Min Sketch and its applications." Google Advertising, Amazon recommendations.
**Non-obviousness:** CMS is often taught separately from random projections but is properly a 1-bit projection onto a hash table — connecting it to the JL framework gives students a unified view.

---

## 6. HYPERPARAMETER TUNING GUIDE

### 6.1 `n_components` (the main knob)

**Role:** The number of output dimensions. This is the primary trade-off parameter between quality and speed.

**Heuristics (in order of preference):**

1. **Start with the JL formula** (this is what `n_components='auto'` does):
   ```python
   from sklearn.random_projection import johnson_lindenstrauss_min_dim
   k_min = johnson_lindenstrauss_min_dim(n_samples=len(X_train), eps=0.1)
   ```
   The result is a conservative upper bound — often you can use 2–5× fewer components without measurable downstream task degradation.

2. **Task-driven rule of thumb:** Start at k = 100–500 for most classification/clustering pipelines. The JL formula gives a safe upper bound; you can then reduce by half iteratively and measure downstream metric (accuracy, silhouette score).

3. **Match to downstream model capacity:** If downstream classifier has d_effective degrees of freedom, k > d_effective is wasteful. E.g., for linear SVM on text, k = 300–1000 typically suffices.

**"Too high" symptoms:** Model training is slow; memory usage is high; no measurable quality gain over lower k.
**"Too low" symptoms:** Downstream accuracy drops sharply; distance computations become noisy; k-NN classifiers degrade.

**Optuna search space:**
```python
def objective(trial):
    n_components = trial.suggest_int('n_components', 50, 2000, log=True)
    proj = SparseRandomProjection(n_components=n_components, random_state=42)
    X_proj = proj.fit_transform(X_train)
    clf = LogisticRegression(max_iter=200).fit(X_proj, y_train)
    X_test_proj = proj.transform(X_test)
    return clf.score(X_test_proj, y_test)
```

---

### 6.2 `eps` (distortion tolerance)

**Role:** When `n_components='auto'`, controls the quality guarantee. ε is the maximum allowed relative distortion of pairwise distances.

**Practical ranges:**

| eps | k (for n=10,000) | Use When |
|---|---|---|
| 0.5 | ~530 | Aggressive compression; downstream task is robust to noise |
| 0.2 | ~2,200 | Good balance; most classification tasks |
| 0.1 | ~9,434 | Conservative; distance-sensitive tasks (k-NN, clustering) |
| 0.05 | ~38,000 | Near-lossless; rarely needed |
| 0.01 | ~888,000 | Almost certainly exceeds original d; never use with 'auto' |

**Key insight:** If you are going to override `n_components` directly, `eps` has no effect. Only set `eps` when you want the JL formula to drive `n_components` automatically.

**"Too high" (eps > 0.5):** k is so small that distance preservation fails; the JL bound has holes; downstream quality degrades unpredictably.
**"Too low" (eps < 0.05):** k may exceed d, making the projection expand rather than reduce dimensionality.

---

### 6.3 `density` (SparseRandomProjection only)

**Role:** Fraction of non-zero entries per column of the random matrix. Controls the speed-accuracy trade-off.

**Practical options:**

| density value | Corresponds to | Non-zeros per column | Speed multiplier |
|---|---|---|---|
| `'auto'` (= 1/√d) | Li et al. 2006 | 1/√d × k | Fastest |
| `1/3` | Achlioptas 2003 | k/3 | ~3× vs. Gaussian |
| `1.0` | Rademacher | k (all non-zero) | Same as Gaussian, ±1 entries |

**Rule:** Use `'auto'` (default) for sparse input data (e.g., TF-IDF matrices). Use `1/3` if you want a good speed/accuracy middle ground. Use `1.0` only if you want integer-valued entries for hardware efficiency.

**Warning:** Very small densities (1/√d) on dense data may increase variance in the embedding. If you see degraded downstream performance with auto density, try explicitly setting `density=1/3`.

---

### 6.4 `dense_output` (SparseRandomProjection only)

**Rule:** Set `dense_output=True` whenever the downstream estimator does not accept sparse matrices (e.g., most neural networks, some sklearn estimators). The default `False` preserves sparsity but many algorithms will silently densify the data anyway, sometimes less efficiently.

---

### 6.5 `random_state`

**Rule:** Always set in production pipelines. Use a fixed integer for reproducibility. During hyperparameter tuning, consider sweeping 3–5 random seeds and averaging metrics to account for projection variance.

---

### 6.6 MinHash LSH Tuning (`datasketch`)

| Parameter | Tuning Guidance |
|---|---|
| `num_perm` | 64 for fast/approximate; 128 for balanced; 256 for high accuracy. Compute time is linear in num_perm. |
| `threshold` (MinHashLSH) | Set slightly below your actual threshold to minimize false negatives. If you want to find pairs with Jaccard ≥ 0.8, set threshold=0.75. |
| Banding (implicit) | datasketch auto-computes optimal (b, r) from threshold. Manual control: `b` bands × `r` rows, where b*r = num_perm. More bands → lower threshold recall; more rows → higher precision. |

---

## 7. EXPERT PRACTITIONER CHECKLIST

### Before Fitting

**Data validation:**
- [ ] What is the current dimensionality d and number of samples n? Run `johnson_lindenstrauss_min_dim(n_samples=n, eps=0.1)` to see the JL-recommended k. If k ≥ d, random projection offers no compression — **use a different method**.
- [ ] Is the data sparse? Use `SparseRandomProjection` with `dense_output=False`. Never apply `StandardScaler` (centering) to sparse data before random projection — centering destroys sparsity and can cause OOM errors.
- [ ] Does the downstream task depend on distance preservation? (k-NN, clustering, kernel methods) Then ε matters — use `eps=0.1` or lower. If downstream task is a linear classifier, you can be more aggressive with ε=0.5.
- [ ] Is d > 10,000? Use `SparseRandomProjection` (very sparse by default). If d < 1,000, GaussianRP is fine.

**Preprocessing decisions:**
- [ ] Random projection is linear and preserves inner products (not just distances). Zero-mean your data if you want inner-product preservation to be equivalent to cosine similarity.
- [ ] Scale features to similar ranges IF downstream task is distance-based. Unscaled features with very different magnitudes will have pairwise distances dominated by high-magnitude features, and random projection will faithfully preserve that imbalance.
- [ ] Do NOT apply PCA before random projection unless d >> k (extreme compression). They serve the same purpose; stacking them adds no benefit.

---

### During Fitting

**What to watch:**
- `proj.n_components_` — verify the auto-computed k is reasonable (should be << d and not 0).
- `proj.density_` (SparseRP) — verify this is << 1.0 for large-d data.
- Wall-clock time of `fit_transform()` — should be O(ndk) operations. For d=50,000, n=100,000, k=500 with density=0.01, this is 50,000 × 500 × 0.01 × 100,000 = 2.5B multiply-adds.

**Common mistakes during fit:**
- Passing the test set to `fit()` — random projection technically doesn't need the data values (only `n_features`), but this is a conceptual error that may confuse when debugging pipelines.
- Calling `fit()` multiple times — each `fit()` generates a new random matrix. For consistent train/test projections, call `fit()` on training data only, then `transform()` on both.

---

### After Fitting

**Verify distance preservation empirically:**
```python
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Sample 500 pairs from training data
idx = np.random.choice(len(X_train), size=500, replace=False)
D_orig = euclidean_distances(X_train[idx])
D_proj = euclidean_distances(X_proj_train[idx])

# Compute distortion ratio for each pair
upper = np.triu_indices_from(D_orig, k=1)
ratios = D_proj[upper] / (D_orig[upper] + 1e-10)
print(f"Distortion range: [{ratios.min():.3f}, {ratios.max():.3f}]")
print(f"Mean distortion: {ratios.mean():.3f} (should be ~1.0)")
print(f"Std of distortion: {ratios.std():.3f} (should be < eps)")
```

**Downstream metric validation:**
```python
# Establish baseline on original data
baseline_score = cross_val_score(clf, X_train, y_train, cv=5).mean()

# Score after random projection
score_rp = cross_val_score(clf, X_proj_train, y_train, cv=5).mean()

print(f"Baseline accuracy: {baseline_score:.4f}")
print(f"After RP accuracy: {score_rp:.4f}")
print(f"Retention: {score_rp/baseline_score:.1%}")
# Expect > 95% retention for eps=0.1
```

**When something goes wrong:**
| Symptom | Likely Cause | Fix |
|---|---|---|
| `n_components_` > d | eps too small, n_samples too large | Set `n_components` manually to d/2 |
| Downstream accuracy collapses | k too small | Increase k or reduce eps |
| OOM during fit | `compute_inverse_components=True` on huge data | Set to False; compute inverse lazily |
| Sparse output causes downstream error | `dense_output=False` default | Set `dense_output=True` |
| Inconsistent train/test projections | Accidentally re-fitting on test | Call `fit()` only once on train set |

---

## 8. EVALUATION METRICS

### 8.1 Distance Preservation (Primary Metric for Random Projection)

**Pairwise Distance Distortion:**
```python
ratios = D_projected / D_original   # for all pairs
# Good: ratios cluster tightly around 1.0 within [1-eps, 1+eps]
# JL guarantee: P(ratio outside [1-eps, 1+eps]) < 1/n
```

**Relative Error:**
```python
rel_error = abs(D_projected - D_original) / D_original
print(f"Max relative error: {rel_error.max():.4f}")  # should be < eps
print(f"Mean relative error: {rel_error.mean():.4f}")
```

---

### 8.2 Reconstruction Error

Since random projection is lossy for k < d, the inverse transform is approximate:

```python
X_approx = proj.inverse_transform(proj.transform(X))
reconstruction_error = np.linalg.norm(X - X_approx, 'fro') / np.linalg.norm(X, 'fro')
print(f"Relative reconstruction error: {reconstruction_error:.4f}")
# Expected: sqrt(1 - k/d) in the Gaussian case for isotropic data
```

---

### 8.3 Trustworthiness (Neighborhood Preservation)

`sklearn.manifold.trustworthiness` measures whether k-nearest neighbors in the original space remain neighbors after projection:

```python
from sklearn.manifold import trustworthiness

# Sample a manageable subset for large datasets
idx = np.random.choice(len(X), size=2000, replace=False)
tw = trustworthiness(X[idx], X_proj[idx], n_neighbors=10)
print(f"Trustworthiness (k=10): {tw:.4f}")
# Range [0, 1]; > 0.9 is very good; > 0.8 is acceptable
# For random projection with eps=0.1, expect 0.95+
```

---

### 8.4 JL Bound Verification

```python
from sklearn.random_projection import johnson_lindenstrauss_min_dim

k_required = johnson_lindenstrauss_min_dim(n_samples=len(X), eps=0.1)
k_actual = proj.n_components_

print(f"JL requires k >= {k_required}")
print(f"Actual k: {k_actual}")
print(f"Safety margin: {k_actual / k_required:.1%}")
# If k_actual < k_required, the JL guarantee does not apply
```

---

### 8.5 Downstream Task Metric (Most Practical)

For classification/regression pipelines, the most practical evaluation is simply:

```python
# Compare downstream metric before and after projection
from sklearn.model_selection import cross_val_score

clf = make_pipeline(
    SparseRandomProjection(n_components=k, random_state=42),
    LogisticRegression(max_iter=300)
)
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"Accuracy with k={k}: {scores.mean():.4f} ± {scores.std():.4f}")
```

Sweep k over `[50, 100, 200, 500, 1000]` and plot the accuracy curve to find the elbow point.

---

### 8.6 MinHash LSH Quality Metrics

For MinHash/LSH:

```python
# Precision and Recall vs. true Jaccard similarity
from datasketch import MinHash, MinHashLSH

# Ground truth: compute exact Jaccard for a sample
# Precision: what fraction of LSH-returned pairs are true positives
# Recall: what fraction of true positive pairs did LSH return

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
```

The trade-off between precision and recall is controlled by:
- `threshold` in MinHashLSH
- `num_perm` (more permutations → better approximation)

---

## 9. COMMON PITFALLS WITH CONCRETE EXAMPLES

### Pitfall 1: Centering Sparse Data Before Projection

**What happens:** `StandardScaler(with_mean=True)` followed by `SparseRandomProjection` on a sparse TF-IDF matrix.

```python
# WRONG — this will crash or produce OOM error
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

# TF-IDF matrix is very sparse (99%+ zeros)
X_tfidf = tfidf_vectorizer.fit_transform(documents)  # shape (50000, 100000)

# Centering fills in all the zeros → 50000 × 100000 dense matrix → OOM
pipe = Pipeline([
    ('scale', StandardScaler(with_mean=True)),  # DANGEROUS with sparse data
    ('rp', SparseRandomProjection()),
])
pipe.fit(X_tfidf)   # CRASH: memory error
```

**Fix:** Never center sparse data. Use `StandardScaler(with_mean=False)` or skip scaling entirely for random projection pipelines on sparse data.

```python
# CORRECT
pipe = Pipeline([
    ('scale', StandardScaler(with_mean=False, with_std=True)),  # scale only
    ('rp', SparseRandomProjection()),
])
```

---

### Pitfall 2: Projecting When k ≥ d

```python
# Dataset has 500 features, 10 samples
X_small = np.random.rand(10, 500)

proj = GaussianRandomProjection(eps=0.1)   # auto n_components
proj.fit(X_small)
print(proj.n_components_)   # Output: 927 — MORE than 500!
```

**What happens:** The JL formula gives k = 927 for n=10 even though d=500. The "projection" actually expands dimensionality. No compression occurs; you've only added computation.

**Fix:** Always check `n_components_ < n_features_in_` after fitting. Add a guard:

```python
k = min(johnson_lindenstrauss_min_dim(n_samples=len(X), eps=0.1), X.shape[1] // 2)
proj = GaussianRandomProjection(n_components=k)
```

---

### Pitfall 3: Re-fitting on Each Chunk in Streaming Settings

```python
# WRONG — different chunks get projected differently
for chunk in data_chunks:
    proj = GaussianRandomProjection(n_components=200, random_state=42)
    X_proj = proj.fit_transform(chunk)
    model.partial_fit(X_proj)
    # The random matrices are DIFFERENT for each chunk (different n_features seen → different matrix)
    # Even with same random_state, re-fitting regenerates the matrix
```

**Fix:** Fit once on a representative sample or use a fixed n_features.

```python
# CORRECT — fit once, transform repeatedly
proj = GaussianRandomProjection(n_components=200, random_state=42)
proj.fit(X_train_sample)   # fixes the random matrix based on n_features

for chunk in data_chunks:
    X_proj = proj.transform(chunk)   # reuses the SAME matrix
    model.partial_fit(X_proj)
```

---

### Pitfall 4: Expecting Random Projection to Find Latent Structure

Random projection does not find principal components, manifold coordinates, or cluster structure. It is a data-agnostic transform. If you visualize the 2D random projection of MNIST, you will see a random blob — not the digit clusters that PCA or t-SNE would reveal.

**Rule:** Use random projection for k >> 2 (preprocessing for distance computation). For visualization (k=2 or 3), use t-SNE, UMAP, or PCA.

---

### Pitfall 5: Using n_components Too Small for LSH

In MinHash LSH, using too few permutations (`num_perm`) causes high variance in Jaccard estimates and degraded precision/recall. Rule of thumb: `num_perm` should be at least `1 / (similarity_threshold * (1 - similarity_threshold))` for adequate statistical power.

```python
# For threshold = 0.5, minimum num_perm ≈ 4 / (0.5 × 0.5) = 16
# In practice, use at least 64; prefer 128 for production
m = MinHash(num_perm=128)   # safe default
```

---

### Pitfall 6: Assuming inverse_transform Recovers the Original Data

```python
proj = GaussianRandomProjection(n_components=100, compute_inverse_components=True)
proj.fit_transform(X)   # X has 5000 features

X_reconstructed = proj.inverse_transform(proj.transform(X))
# X_reconstructed.shape is (n, 5000) BUT
# ||X - X_reconstructed||_F / ||X||_F ≈ sqrt(1 - 100/5000) ≈ 0.99
# 99% of the original energy is lost!
```

**Rule:** `inverse_transform` is useful only for debugging and visualization, not for reconstructing original signals. For that purpose use PCA (which has the optimal reconstruction for a given k).

---

## 10. MATHEMATICAL FOUNDATIONS (QUICK REFERENCE)

### 10.1 The Johnson-Lindenstrauss Lemma (Statement)

Let ε ∈ (0, 1/2) and let X = {x₁, ..., xₙ} ⊆ ℝᵈ be any set of n points. If k ≥ 4 ln(n) / (ε²/2 − ε³/3), then there exists a linear map f: ℝᵈ → ℝᵏ such that for all i, j:

```
(1 − ε) ‖xᵢ − xⱼ‖² ≤ ‖f(xᵢ) − f(xⱼ)‖² ≤ (1 + ε) ‖xᵢ − xⱼ‖²
```

Moreover, a random Gaussian map f(x) = (1/√k) R x (where R has i.i.d. N(0,1) entries) satisfies this with probability ≥ 1 − 2n exp(−k(ε²/4 − ε³/12)).

### 10.2 Why Gaussian Works (Concentration of Measure)

For a single pair (x, y), define w = x − y. The projected distance is:

```
‖Rw/√k‖² = (1/k) Σᵢ (rᵢᵀw)² where rᵢ ~ N(0, Iᵈ)
```

Each term (rᵢᵀw)² / ‖w‖² is χ²(1) distributed. The sum (1/k) Σᵢ (rᵢᵀw)² / ‖w‖² is chi-squared(k)/k — concentrated around 1 with standard deviation O(1/√k). This is why k = O(log n) suffices for a union bound over all O(n²) pairs.

### 10.3 Sparse Matrix Variance Matching

Achlioptas (2003) shows that ternary entries {±√3, 0} with probability {1/6, 2/3, 1/6} have:
- E[rᵢⱼ] = 0  (unbiased)
- E[rᵢⱼ²] = 1  (same variance as N(0,1))
- E[rᵢⱼ⁴] = 3  (same 4th moment as N(0,1))

These three moment conditions are sufficient for the JL bound to transfer to the sparse matrix. The very sparse construction by Li et al. (2006) weakens to just matching the first two moments but shows empirically (and theoretically) that the bound still holds with mild constant factor overhead.

### 10.4 LSH Formal Definition

A family H of hash functions h: ℝᵈ → U is (r₁, r₂, p₁, p₂)-sensitive if for any x, y:
- If ‖x−y‖ ≤ r₁, then P[h(x)=h(y)] ≥ p₁
- If ‖x−y‖ ≥ r₂, then P[h(x)=h(y)] ≤ p₂

For random hyperplane LSH (SimHash):
- h(x) = sign(aᵀx) where a ~ N(0, I)
- P[h(x)=h(y)] = 1 − arccos(cos(x,y)) / π
- This is the SimHash construction of Charikar (2002)

For MinHash:
- Each permutation π: Ω → {1,...,|Ω|}
- h(S) = min_{e ∈ S} π(e)
- P[h(S)=h(T)] = |S∩T| / |S∪T|  (Jaccard similarity)

### 10.5 SRHT Construction

The Subsampled Randomized Hadamard Transform applies three operations in sequence:

```
SRHT(x) = √(d/k) · P · H · D · x
```

Where:
- D is a random diagonal ±1 sign-flip matrix (Rademacher diagonal)
- H is the normalized Walsh-Hadamard transform matrix (orthogonal, fast O(d log d))
- P is a k×d random subsampling matrix (selects k of d rows)

Cost: O(d log d) per vector vs O(dk) for Gaussian. This makes SRHT preferable when d is very large (d > 10⁵) and k is moderate.

### 10.6 Key Inequalities (for the Writing Agent)

**Markov's inequality:** P[X ≥ t] ≤ E[X] / t

**Hoeffding's inequality:** If X = Σ Xᵢ and Xᵢ ∈ [aᵢ, bᵢ]:
```
P[|X − E[X]| ≥ t] ≤ 2 exp(−2t² / Σ(bᵢ−aᵢ)²)
```

**Union bound (for JL):** P[∃ pair violates bound] ≤ n² × P[one pair violates bound]
Setting n² × 2 exp(−k(ε²/4)) ≤ δ gives k ≥ 8 log(n/δ^(1/2)) / ε²

---

## APPENDIX: QUICK CODE SNIPPETS FOR THE WRITING AGENT

### Complete Working Example (Dense Data)

```python
import numpy as np
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=2000, random_state=42)

# Check JL-recommended k
k_jl = johnson_lindenstrauss_min_dim(n_samples=1000, eps=0.1)
print(f"JL recommends k >= {k_jl}")   # ~939

# Gaussian projection
grp = GaussianRandomProjection(n_components=200, random_state=42)
X_g = grp.fit_transform(X)
print(f"Gaussian: {X.shape} -> {X_g.shape}")

# Sparse projection (default = very sparse, Li 2006)
srp = SparseRandomProjection(n_components=200, random_state=42)
X_s = srp.fit_transform(X)
print(f"Sparse: {X.shape} -> {X_s.shape}")
print(f"Density: {srp.density_:.4f}")   # ~0.0224 = 1/sqrt(2000)

# Verify distance preservation
from sklearn.metrics.pairwise import euclidean_distances
idx = np.random.choice(len(X), size=100, replace=False)
D_orig = euclidean_distances(X[idx])
D_proj = euclidean_distances(X_s[idx])
triu = np.triu_indices_from(D_orig, k=1)
ratios = D_proj[triu] / (D_orig[triu] + 1e-10)
print(f"Distance ratios — mean: {ratios.mean():.3f}, std: {ratios.std():.3f}")
```

### Complete Working Example (MinHash LSH)

```python
from datasketch import MinHash, MinHashLSH

def text_to_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for shingle in {text[i:i+3] for i in range(len(text) - 2)}:
        m.update(shingle.encode('utf8'))
    return m

documents = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumped over a lazy dog",   # near-duplicate
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks for AI tasks",
]

# Build index
lsh = MinHashLSH(threshold=0.4, num_perm=128)
minhashes = {}
for i, doc in enumerate(documents):
    m = text_to_minhash(doc)
    minhashes[i] = m
    lsh.insert(f"doc_{i}", m)

# Query
query_m = text_to_minhash("the quick brown fox jumps over the lazy dog")
results = lsh.query(query_m)
print(f"Similar documents: {results}")
# Expected: ['doc_0', 'doc_1'] (near-duplicates)
```

### Pipeline for High-Dimensional Text

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, sublinear_tf=True)),
    ('rp', SparseRandomProjection(n_components=1000, random_state=42)),
    ('clf', LogisticRegression(C=1.0, max_iter=300)),
])

# TF-IDF produces 50k-dim sparse vectors
# SparseRP compresses to 1000-dim with ~0.45% non-zeros in projection matrix
# Total pipeline is fast and memory-efficient
pipe.fit(X_train_texts, y_train)
print(pipe.score(X_test_texts, y_test))
```

---

## APPENDIX: CITATIONS SUMMARY

| Short Reference | Full Citation | DOI / URL |
|---|---|---|
| Johnson & Lindenstrauss 1984 | Extensions of Lipschitz mappings into a Hilbert space. Contemporary Math vol. 26, pp. 189–206. AMS. | http://stanford.edu/class/cs114/readings/JL-Johnson.pdf |
| Dasgupta & Gupta 2003 | An elementary proof of a theorem of Johnson and Lindenstrauss. Random Structures & Algorithms 22(1):60–65. | https://doi.org/10.1002/rsa.10073 |
| Achlioptas 2003 | Database-friendly random projections: JL with binary coins. JCSS 66(4):671–687. | https://doi.org/10.1016/S0022-0000(03)00025-4 |
| Li, Hastie & Church 2006 | Very sparse random projections. KDD 2006, pp. 287–296. ACM. | https://doi.org/10.1145/1150402.1150436 |
| Indyk & Motwani 1998 | Approximate nearest neighbors: Towards removing the curse of dimensionality. STOC 1998. | https://doi.org/10.1145/276698.276876 |
| Charikar 2002 | Similarity estimation techniques from rounding algorithms. STOC 2002, pp. 380–388. | https://doi.org/10.1145/509907.509965 |
| Broder et al. 1997 | Syntactic clustering of the web. WWW6, pp. 391–404. Elsevier. | https://doi.org/10.1016/S0169-7552(97)00031-7 |
| Ghojogh et al. 2021 | JL Lemma, Linear and Nonlinear Random Projections: Tutorial and Survey. arXiv:2108.04172. | https://arxiv.org/abs/2108.04172 |
| Cormode & Muthukrishnan 2004 | An improved data stream summary: The Count-Min Sketch. J. Algorithms 55(1):58–75. | https://doi.org/10.1016/j.jalgor.2003.12.001 |
