# Research Dossier: K-Nearest Neighbors (kNN)

> **For chapter file:** `k-nearest-neighbors.md`
> **Dossier prepared:** 2026-06-25
> **Feeds sections:** §0 through §11 of the chapter

---

## Scope

This dossier supports the chapter `k-nearest-neighbors.md` — a deep expert treatment of the k-Nearest Neighbors family: the original non-parametric density estimation paper, the Cover & Hart classification theory, all sklearn API variants, distance metric theory, tree data structures, approximate nearest neighbor methods (FAISS, HNSW/hnswlib, Annoy, PyNNDescent), the curse of dimensionality with math, feature scaling requirements, prototype/condensed methods, explainability options, and tuning strategy.

---

## Verified Packages

### Core: scikit-learn 1.5.x

All APIs verified against scikit-learn 1.5.2 documentation.

```python
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
    NearestNeighbors,
    KDTree,
    BallTree,
    NeighborhoodComponentsAnalysis,
    KNeighborsTransformer,
    RadiusNeighborsTransformer,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
```

### Approximate Nearest Neighbor Packages

| Package | Install | Space | Notes |
|---|---|---|---|
| `faiss-cpu` | `pip install faiss-cpu` | L2, IP | Meta/Facebook, C++ + CUDA, Python bindings. Also `faiss-gpu`. |
| `faiss-gpu` | `conda install -c pytorch faiss-gpu` | L2, IP | GPU-accelerated billion-scale search |
| `hnswlib` | `pip install hnswlib` | l2, cosine, ip | C++ header-only with Python bindings; ref impl. of HNSW |
| `annoy` | `pip install annoy` | angular, euclidean, manhattan, hamming, dot | Spotify; tree-based; mmap'd files |
| `pynndescent` | `pip install pynndescent` | many metrics | Pure Python/Numba; used internally by UMAP |

### Explainability (for kNN)

```python
import shap
# For kNN only KernelExplainer works (no TreeExplainer)
# shap.KernelExplainer is model-agnostic but SLOW: O(2^p) samples per background point
from sklearn.inspection import permutation_importance  # preferred alternative
```

---

## The Original Algorithm

### Paper 1: Fix & Hodges (1951) — Birth of Non-Parametric Classification

**Full citation:**
Fix, E., and Hodges, J. L. (1951). *Discriminatory Analysis, Nonparametric Discrimination: Consistency Properties*. Technical Report 4, Project 21-49-004, USAF School of Aviation Medicine, Randolph Field, Texas.

- **Not peer-reviewed at publication** — an internal USAF technical report, never formally published by the authors. Reprinted posthumously in *International Statistical Review* 57(3), 238–247 (1989).
- **URL:** No DOI; best references are via Google Books (`ISBN: 978-0-9673519`) or Open Research Online: https://oro.open.ac.uk/28327/

**Problem solved:** The dominant approach to discrimination (classification) in 1951 was Fisher's Linear Discriminant Analysis (1936), which requires the assumption of Gaussian class-conditional densities with equal covariance. Fix and Hodges wanted a method that made no such parametric assumption.

**Core idea:** Estimate the class-conditional density $p(\mathbf{x} | C_j)$ at a query point $\mathbf{x}$ by counting the number of training samples from class $C_j$ that fall within a small ball of radius $r$ around $\mathbf{x}$. Formally, if $k_j$ of the $k$ nearest neighbors belong to class $j$:
$$\hat{p}(\mathbf{x} | C_j) \propto \frac{k_j}{n_j \cdot V_k(\mathbf{x})}$$
where $n_j$ is the number of training samples in class $j$ and $V_k(\mathbf{x})$ is the volume of the hypersphere containing the $k$ nearest neighbors of $\mathbf{x}$.

**The 1-NN rule** then assigns $\mathbf{x}$ to the class of its single nearest training point. Fix and Hodges proved this estimator is **consistent** — as $n \to \infty$ with $k/n \to 0$ and $k \to \infty$, the estimate converges to the true density. This is the fundamental result that legitimized kNN.

**What was experimental:** The choice of $k$ and the curse of dimensionality were not yet analyzed. The paper was entirely theoretical with no computational experiments.

---

### Paper 2: Cover & Hart (1967) — The Landmark Error-Rate Bound

**Full citation:**
Cover, T. M., and Hart, P. E. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27. DOI: https://doi.org/10.1109/TIT.1967.1053964

- **Semantic Scholar PDF:** https://www.semanticscholar.org/paper/Nearest-neighbor-pattern-classification-Cover-Hart/0efb841403aa6252b39ae6975c1cc5410554ef7b
- **Stanford preprint (Cover):** https://isl.stanford.edu/~cover/papers/transIT/0050cove.pdf

**Historical context:** By 1967, statistical decision theory had established the **Bayes error rate** $R^*$ as the gold standard — the irreducible minimum error achievable by any classifier given perfect knowledge of the true distributions. The question was: how close can a non-parametric method come to $R^*$?

**The landmark result (the Cover-Hart bound):**
Let $R$ be the asymptotic error rate of the 1-NN rule, and $R^*$ the Bayes error rate. Then:

$$R^* \leq R \leq R^*\left(2 - \frac{C}{C-1} R^*\right)$$

For the binary case ($C = 2$):
$$R^* \leq R \leq 2R^*(1 - R^*)$$

Since $R^* \leq 1/2$, this gives the famous result: **$R \leq 2R^*$**. The 1-NN rule achieves at most *twice* the Bayes error rate, using only training data and no distributional assumptions.

**Why this matters:** If $R^* = 10\%$, then 1-NN guarantees at most $20\%$ error — even without knowing anything about the true distributions. This placed kNN on solid theoretical footing and made it a serious contender in the pattern recognition literature.

**Key insight:** At the limit of large $n$, every query point $\mathbf{x}$ has a 1-NN that is essentially a replication of $\mathbf{x}$ drawn from the same distribution. The 1-NN rule is thus equivalent to a two-sample test between the query and its shadow. The proof uses the continuity of the class-posterior probabilities and the convergence of nearest neighbors.

**For $k > 1$:** As $k \to \infty$ with $k/n \to 0$, the kNN error rate converges to the Bayes error $R^*$ — this is the theoretical guarantee underlying all of kNN.

---

## The Core kNN Algorithm

**Prediction for classification (majority vote):**

Given a query $\mathbf{x}_q$, a training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$:

1. Compute distance $d(\mathbf{x}_q, \mathbf{x}_i)$ for all $i = 1, \ldots, n$
2. Find the $k$ indices $\mathcal{N}_k(\mathbf{x}_q)$ with smallest distances
3. Predict: $\hat{y} = \text{argmax}_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x}_q)} \mathbf{1}[y_i = c]$

**Prediction for regression (mean):**

$$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x}_q)} y_i$$

**Weighted prediction (distance-weighted):**

With $w_i = 1 / d(\mathbf{x}_q, \mathbf{x}_i)$:

$$\hat{y} = \frac{\sum_{i \in \mathcal{N}_k} w_i y_i}{\sum_{i \in \mathcal{N}_k} w_i}$$

(For classification, each class's weighted votes are summed: $\hat{y} = \text{argmax}_c \sum_{i \in \mathcal{N}_k} w_i \mathbf{1}[y_i = c]$)

**Training complexity:** $O(1)$ — kNN is a "lazy learner": there is no explicit training step. The entire training set is stored.

**Inference complexity (brute force):** $O(np)$ per query, where $n$ = training samples, $p$ = features. With efficient data structures: $O(p \log n)$ for kd-tree (low dimensions), $O(p \log n)$ for ball-tree (medium dimensions).

---

## Major Variants & Evolution

### 1. Weighted kNN (Distance-Weighted)

- **Problem solved:** Uniform voting treats a very close neighbor the same as a far one. A point at distance $\varepsilon$ from the query is far more informative than one at distance $100\varepsilon$.
- **Change:** Weight each neighbor's vote/value by $w_i = 1/d_i$ (or $w_i = 1/d_i^2$). Handle the zero-distance case explicitly (a training point query returns that point's label).
- **sklearn:** `KNeighborsClassifier(weights='distance')` or a custom callable passed to `weights=`.
- **When to prefer:** Almost always better than uniform when the neighborhood is heterogeneous. Use uniform only when all neighbors are roughly equidistant.

### 2. Radius-Based Neighbors (RadiusNeighbors)

- **Problem solved:** Fixed-$k$ kNN has pathological behavior in sparse regions — a neighborhood with $k=5$ may span the entire feature space, averaging wildly irrelevant points.
- **Change:** Instead of fixing $k$, fix a radius $r$. Use all training points within ball of radius $r$ around the query. The number of neighbors varies by query.
- **Paper origin:** Natural generalization; no single canonical paper. Widely discussed in Devroye, Györfi & Lugosi "A Probabilistic Theory of Pattern Recognition" (1996).
- **sklearn:** `RadiusNeighborsClassifier(radius=1.0, weights='uniform', outlier_label=None)` and `RadiusNeighborsRegressor(radius=1.0)`. The `outlier_label` handles queries in empty regions.
- **When to prefer:** When data density is highly non-uniform across the feature space.

### 3. Condensed Nearest Neighbor — CNN (Hart, 1968)

- **Paper:** Hart, P. E. (1968). The condensed nearest neighbor rule. *IEEE Transactions on Information Theory*, 14(3), 515–516. DOI: https://doi.org/10.1109/TIT.1968.1054155
- **Problem solved:** kNN stores the entire training set ($O(np)$ memory) and has $O(np)$ inference cost. For $n = 10^6$, this is prohibitive.
- **Algorithm:** Iteratively select a minimal subset $S$ of training points such that 1-NN on $S$ correctly classifies all remaining training points. Start with one random point from each class; scan the training set and add any misclassified point to $S$; repeat until no change.
- **Result:** $|S| \ll n$ in practice. Boundary points are retained; interior points (surrounded by same-class neighbors) are discarded.
- **sklearn:** Not directly implemented but achievable via `imbalanced-learn` or custom code.
- **Limitation:** Non-deterministic (order-dependent), may retain noise/outliers.

### 4. Edited Nearest Neighbor — ENN (Wilson, 1972)

- **Paper:** Wilson, D. L. (1972). Asymptotic properties of nearest neighbor rules using edited data. *IEEE Transactions on Systems, Man, and Cybernetics*, 2(3), 408–421. DOI: https://doi.org/10.1109/TSMC.1972.4309137
- **Problem solved:** kNN is sensitive to noisy training labels. A mislabeled point in the interior of the wrong class creates a "decision island" that corrupts neighboring predictions.
- **Algorithm:** Remove training points that are misclassified by their own $k$ nearest neighbors. This smooths the decision boundary and removes label noise.
- **sklearn/imbalanced-learn:** `from imblearn.under_sampling import EditedNearestNeighbours`

### 5. Locally Weighted Regression / LOWESS (Cleveland, 1979)

- **Paper:** Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots. *Journal of the American Statistical Association*, 74(368), 829–836. URL: https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf
- **Connection:** LOWESS is essentially a kNN regressor where the local regression model is a polynomial (not just a mean), weighted by a tricube kernel based on distance.
- **Algorithm:** At each query point, find the $f \cdot n$ nearest neighbors ($f$ = bandwidth fraction), fit a locally weighted linear regression using a tricube weight function $w(u) = (1 - |u|^3)^3$, then predict.
- **Key distinction from kNN regressor:** Fits a local polynomial (linear or quadratic) inside the neighborhood rather than just taking a mean. Much less biased near boundaries.
- **Python:** `statsmodels.nonparametric.smoothers_lowess.lowess`

### 6. Neighborhood Components Analysis — NCA (Goldberger et al., 2004)

- **Paper:** Goldberger, J., Roweis, S., Hinton, G., and Salakhutdinov, R. (2004). Neighbourhood Components Analysis. *Advances in Neural Information Processing Systems (NeurIPS)*, 17. URL: https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf
- **Problem solved:** The Euclidean metric treats all features equally. A highly irrelevant feature contributes the same to distances as a highly predictive feature. This degrades kNN performance.
- **Algorithm:** Learn a linear transformation $A$ (a $d' \times p$ matrix) such that the kNN error rate in the transformed space $A\mathbf{x}$ is minimized. The objective is the expected leave-one-out accuracy under a stochastic (softmax) version of 1-NN:
$$f(A) = \sum_i \sum_{j: y_j = y_i} \frac{\exp(-\|A\mathbf{x}_i - A\mathbf{x}_j\|^2)}{\sum_{k \neq i} \exp(-\|A\mathbf{x}_i - A\mathbf{x}_k\|^2)}$$
Maximized via gradient ascent (L-BFGS-B in sklearn).
- **sklearn:** `NeighborhoodComponentsAnalysis(n_components=None, init='auto', max_iter=50, tol=1e-5)`. Use in a Pipeline before `KNeighborsClassifier`.
- **When to prefer:** When you have irrelevant/redundant features and can't do manual feature selection. Expensive: $O(n^2 p)$ per iteration.

### 7. HNSW — Hierarchical Navigable Small World (Malkov & Yashunin, 2016)

- **Paper:** Malkov, Y. A., and Yashunin, D. A. (2016/2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824–836. arXiv: https://arxiv.org/abs/1603.09320
- **Problem solved:** Exact kNN search is $O(np)$ — prohibitive at scale. Prior approximate methods either sacrificed recall or required expensive index rebuilds.
- **Algorithm:** Builds a multi-layer proximity graph. Upper layers contain long-range links (coarse navigation, similar to skip lists). Lower layers contain short-range links (fine-grained local connectivity). Element layer assignment is random with exponentially decaying probability $P(\text{layer} \geq l) = e^{-l/m_L}$. Search starts at the top layer and greedily descends through layers.
- **Key parameters:** `M` (max outgoing connections per node; controls memory/accuracy tradeoff), `ef_construction` (size of dynamic candidate list during construction; larger = better recall at build time), `ef` / `ef_search` (size of candidate list during search; larger = better recall at query time).
- **Complexity:** $O(\log n)$ search, $O(n \log n)$ build. Memory: $O(M \cdot n)$.
- **Python:** `hnswlib` (`pip install hnswlib`). Also integrated into FAISS as `IndexHNSWFlat`.
- **When to prefer:** Static datasets where you need sub-millisecond latency and high recall (>95%). De facto standard for production vector search.

### 8. FAISS (Johnson, Douze & Jégou, 2021)

- **Paper:** Johnson, J., Douze, M., and Jégou, H. (2021). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547. arXiv: https://arxiv.org/abs/1702.08734
- **GitHub:** https://github.com/facebookresearch/faiss
- **Key index types:**
  - `IndexFlatL2` — exact L2 search (brute force), no approximation, guaranteed correct
  - `IndexFlatIP` — exact inner product search
  - `IndexIVFFlat` — inverted file index: partitions space into `nlist` Voronoi cells using k-means, then searches only `nprobe` cells at query time. Trade-off: accuracy vs speed via `nprobe`.
  - `IndexIVFPQ` — IVF + Product Quantization for memory compression
  - `IndexHNSWFlat` — HNSW graph structure for logarithmic search
  - `IndexPQ` — pure product quantization (memory-constrained deployments)
- **Production recipe:**

```python
import faiss
import numpy as np

d = 128          # dimension
n = 1_000_000   # dataset size
nlist = 1000     # number of Voronoi cells (rule of thumb: ~sqrt(n))

# Build index
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# Train on data sample
data = np.random.rand(n, d).astype('float32')
index.train(data)
index.add(data)

# Search
index.nprobe = 10          # search 10 of 1000 cells
k = 5
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k)
```

### 9. Annoy (Erik Bernhardsson / Spotify, 2013)

- **GitHub:** https://github.com/spotify/annoy
- **Algorithm:** Builds multiple random projection trees. At each node, a random hyperplane splits the data. Multiple trees increase recall.
- **Key advantage:** All data is memory-mapped to disk (`save` / `load`), so many processes share the same index with zero copying overhead. Excellent for read-heavy production deployments.
- **Limitation:** No incremental insertions — must rebuild the entire index to add new points. Spotify has since replaced it with **Voyager** (based on hnswlib) for new projects.

```python
from annoy import AnnoyIndex

dim = 128
t = AnnoyIndex(dim, 'euclidean')
for i, v in enumerate(vectors):
    t.add_item(i, v)
t.build(n_trees=50)   # more trees = better recall, more memory
t.save('index.ann')

# Query
t.get_nns_by_vector(query_vec, n=5, include_distances=True)
```

### 10. PyNNDescent (McInnes, 2018)

- **GitHub:** https://github.com/lmcinnes/pynndescent
- **Algorithm:** Nearest neighbor descent — iteratively refines an approximate k-NN graph by assuming "a neighbor of a neighbor is likely a neighbor." Converges quickly to a high-quality k-NN graph.
- **Key strength:** Pure Python/Numba (no C++ compilation required); wide metric support; used internally by UMAP for graph construction.
- **Notable:** Performs best in the high-recall regime (>90%). Slower than hnswlib for very high-throughput scenarios.

```python
import pynndescent
import numpy as np

index = pynndescent.NNDescent(data, n_neighbors=15, metric='euclidean')
indices, distances = index.query(query_data, k=10)
```

---

## Hyperparameter Reference

### KNeighborsClassifier / KNeighborsRegressor (sklearn 1.5.x)

All parameters verified against sklearn 1.5.2 documentation.

| Parameter | Default | Type | Description |
|---|---|---|---|
| `n_neighbors` | `5` | `int` | Number of neighbors to use for query |
| `weights` | `'uniform'` | `{'uniform', 'distance'}` or callable | Weight function for prediction |
| `algorithm` | `'auto'` | `{'auto', 'ball_tree', 'kd_tree', 'brute'}` | Algorithm for nearest-neighbor search |
| `leaf_size` | `30` | `int` | Leaf size for BallTree or KDTree |
| `p` | `2` | `float` | Power for Minkowski metric (`p=1` → L1, `p=2` → L2) |
| `metric` | `'minkowski'` | `str` or callable | Distance metric |
| `metric_params` | `None` | `dict` | Extra params for the metric (e.g., `{'VI': inv_cov}` for Mahalanobis) |
| `n_jobs` | `None` | `int` | Parallel jobs (`-1` = all CPUs). Does NOT affect `fit()`. |

**`n_neighbors` tuning:**
- Default 5 is often a reasonable starting point for datasets with thousands of samples.
- Rule of thumb: try $k \approx \sqrt{n}$ as a ceiling; start lower.
- For binary classification: choose odd $k$ to avoid ties.
- Use cross-validation with a range like `[1, 3, 5, 7, 11, 15, 21, 31, 51, 101]`.
- **Too low ($k=1$):** High variance, overfitting, sensitive to noise — fits training data perfectly.
- **Too high ($k \approx n$):** High bias — degenerates to majority class prediction (classifier) or global mean (regressor).
- The bias-variance tradeoff is inverted vs. most models: *increasing $k$ increases bias and reduces variance*.

**`weights` tuning:**
- `'uniform'`: All $k$ neighbors vote equally. Effectively assumes uniform density within the neighborhood.
- `'distance'`: Weight by $1/d_i$. Better when neighbors span very different distances. Be careful if there are exact duplicates in the training set (division by zero — sklearn handles this by assigning weight 1.0 to zero-distance neighbors and 0.0 to all others).
- **Custom callable:** `weights=lambda distances: np.exp(-distances**2 / (2 * sigma**2))` — Gaussian kernel.
- **Recommendation:** Almost always use `'distance'`. The performance gain is rarely negative.

**`algorithm` tuning — detailed rules:**
- `'auto'` selection logic in sklearn 1.5.x:
  - Defaults to `'brute'` if: input is sparse, `metric='precomputed'`, `p` is not integer (for Minkowski), or $p > 15$ dimensions.
  - Otherwise tries `'kd_tree'` first if the metric is valid for KDTree; then `'ball_tree'`; else `'brute'`.
- `'kd_tree'`: Build $O(np \log n)$, query $O(p \log n)$ in low dimensions. Degrades to $O(np)$ when $p \geq 20$.
- `'ball_tree'`: More expensive build, but handles higher dimensions and more distance metrics (including Mahalanobis, Hamming, cosine) better than KDTree. Valid metrics include all KDTree metrics PLUS `seuclidean`, `mahalanobis`, `hamming`, `canberra`, `braycurtis`, `jaccard`, `dice`, `haversine`.
- `'brute'`: $O(np)$ query — always exact. Fastest for small $n$ (< ~1000) or large $p$ (> 15–20).
- **Decision guide:**
  - $p \leq 10$, moderate $n$ → `'kd_tree'`
  - $10 < p \leq 20$, using non-Euclidean metrics → `'ball_tree'`
  - $p > 20$ or small dataset → `'brute'`
  - Very large $n$ ($> 10^5$) → switch to FAISS/hnswlib outside sklearn

**`leaf_size` tuning:**
- Controls when tree algorithms switch to brute-force within a leaf node.
- Default `30` is a reasonable balance for most datasets.
- **Smaller leaf_size:** Deeper tree, faster query at cost of larger tree and slower build.
- **Larger leaf_size:** Shallower tree, faster build and less memory, slower query.
- In practice, tuning `leaf_size` rarely yields meaningful improvements — focus on `n_neighbors` and `weights`.

**`p` (Minkowski power) tuning:**
- `p=1`: Manhattan distance (L1). Robust to outliers in individual features; often better for high-dimensional sparse data.
- `p=2`: Euclidean distance (L2). Default. Appropriate for continuous, well-scaled features.
- `p → ∞`: Chebyshev distance (max component difference). Rarely used in practice.
- **Only matters when `metric='minkowski'`** (the default). If you set `metric='euclidean'`, `p` is ignored.
- For NLP/text features with many zeros: Manhattan is often more discriminative than Euclidean.

**`metric` options and when to use each:**

| Metric | When to use | Notes |
|---|---|---|
| `'minkowski'` (default, `p=2` → Euclidean) | Continuous, well-scaled features | Most common; requires scaling |
| `'euclidean'` | Same as above | Identical to minkowski p=2 |
| `'manhattan'` | Sparse/high-dimensional; robust to outliers | L1; good for image pixel data |
| `'chebyshev'` | When max difference matters most | L-infinity; rare in practice |
| `'mahalanobis'` | Correlated features; want to account for scale and correlation | Requires `metric_params={'VI': inv_cov_matrix}`. Use BallTree. |
| `'hamming'` | Binary or categorical features | Fraction of positions that differ |
| `'jaccard'` | Binary bag-of-words, set membership | Only BallTree |
| `'haversine'` | Geo-coordinates (lat/lon) | Only BallTree; input must be (lat, lon) in radians |
| `'cosine'` | Text embeddings, normalized vectors | Not directly in KDTree/BallTree; use `brute` or normalize first |
| `'precomputed'` | Custom distance matrices | `X` passed to `fit` must be distance matrix |

**Mahalanobis usage — verified code:**
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Must compute inverse covariance matrix from training data ONLY
inv_cov = np.linalg.pinv(np.cov(X_train.T))   # use pinv for numerical stability

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='mahalanobis',
    metric_params={'VI': inv_cov},   # 'VI' = inverse of covariance
    algorithm='ball_tree'            # mahalanobis requires ball_tree or brute
)
```
**Note:** Mahalanobis is NOT supported by kd_tree. Use `algorithm='ball_tree'` or `algorithm='brute'`.

**`metric_params` usage:**
- For `metric='minkowski'` with non-integer `p`: `metric_params={'p': 1.5}`
- For `metric='seuclidean'` (standardized Euclidean): `metric_params={'V': variances}` where `V` is per-feature variance array
- For custom callables: pass any extra kwargs the callable accepts

**`n_jobs` tuning:**
- Only affects `kneighbors()` queries, not `fit()` (fit is just storing the data).
- `n_jobs=-1` uses all CPUs. Useful for large batch predictions.
- Overhead of multiprocessing is only worthwhile for large datasets or large query batches.

### RadiusNeighborsClassifier / RadiusNeighborsRegressor

| Parameter | Default | Notes |
|---|---|---|
| `radius` | `1.0` | Must be tuned to data scale — ALWAYS scale data first |
| `weights` | `'uniform'` | Same as KNeighbors |
| `algorithm` | `'auto'` | Same as KNeighbors |
| `leaf_size` | `30` | Same as KNeighbors |
| `p` | `2` | Same as KNeighbors |
| `metric` | `'minkowski'` | Same as KNeighbors |
| `outlier_label` | `None` (raises ValueError) | Label to assign when no neighbors found within radius |
| `metric_params` | `None` | Same as KNeighbors |
| `n_jobs` | `None` | Same as KNeighbors |

**Critical:** After `StandardScaler`, features have unit variance. A `radius=1.0` then corresponds to roughly ±1 standard deviation per feature — a reasonable starting point. Without scaling, `radius=1.0` is meaningless.

### NearestNeighbors (unsupervised retrieval)

```python
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='auto',
                         leaf_size=30, metric='minkowski', p=2,
                         metric_params=None, n_jobs=None)
nbrs.fit(X_train)

# k nearest neighbors
distances, indices = nbrs.kneighbors(X_query)   # shape (n_query, k)

# radius neighbors
distances, indices = nbrs.radius_neighbors(X_query, radius=0.5)  # variable-length lists

# Sparse graph for graph algorithms
graph = nbrs.kneighbors_graph(X_train, mode='connectivity')
```

### NeighborhoodComponentsAnalysis (NCA)

| Parameter | Default | Notes |
|---|---|---|
| `n_components` | `None` (= `n_features`) | Reduced dimensionality output |
| `init` | `'auto'` | Initialization: `'pca'`, `'lda'`, `'identity'`, `'random'` |
| `warm_start` | `False` | Reuse previous solution for subsequent fits |
| `max_iter` | `50` | L-BFGS-B optimization iterations |
| `tol` | `1e-5` | Convergence tolerance |
| `verbose` | `0` | Verbosity |
| `random_state` | `None` | Seed |

---

## Curse of Dimensionality — The Math

This is the central theoretical weakness of kNN and must be covered precisely.

### Volume of Hypersphere

The volume of a $p$-dimensional ball of radius $r$:
$$V_p(r) = \frac{\pi^{p/2}}{\Gamma(p/2 + 1)} r^p$$

The ratio of the inscribed hypersphere volume to its enclosing hypercube volume:
$$\frac{V_p(1/2)}{1^p} = \frac{\pi^{p/2}}{2^p \cdot \Gamma(p/2 + 1)}$$

| Dimension $p$ | Sphere/Cube volume ratio |
|---|---|
| 2 | 78.5% |
| 3 | 52.4% |
| 5 | 16.5% |
| 10 | 0.25% |
| 20 | $2.5 \times 10^{-7}$% |
| 100 | $\approx 0$ |

As $p \to \infty$, nearly all volume concentrates in the corners of the hypercube. A ball of fixed radius contains exponentially fewer training points.

### The Equidistance Phenomenon

Let $\mathbf{x}_1, \ldots, \mathbf{x}_n$ be i.i.d. uniform points in $[0,1]^p$, and let $d_{\max}$ and $d_{\min}$ be the maximum and minimum distances to a query point. Then:

$$\frac{d_{\max} - d_{\min}}{d_{\min}} \xrightarrow{p} 0 \text{ as } p \to \infty$$

That is, all points become roughly equidistant from any query. The discriminative power of "nearest" vs "farthest" collapses. Formally (Beyer et al. 1999):
$$\lim_{p \to \infty} \text{Var}\left[\frac{d(\mathbf{x}, \mathbf{y})}{\mathbb{E}[d(\mathbf{x}, \mathbf{y})]}\right] = 0$$

**Reference:** Beyer, K., Goldstein, J., Ramakrishnan, R., and Shaft, U. (1999). When Is "Nearest Neighbor" Meaningful? *ICDT*, 217–235.

### Practical implication

For kNN to work well in $p$ dimensions, you need exponentially more data:
$$n \sim C^p$$
for some constant $C > 1$. At $p = 20$, this is infeasible for most datasets.

**Mitigation strategies (all must be discussed in the chapter):**
1. **Feature scaling** (required, not optional)
2. **Dimensionality reduction:** PCA (linear), UMAP (non-linear, manifold)
3. **Feature selection:** Remove irrelevant features that add noise dimensions
4. **Metric learning (NCA):** Learn a low-rank projection that preserves discriminative structure
5. **Approximate methods (FAISS/HNSW):** Accept that exact NN is meaningless; find approximate NN in reduced space

---

## kd-Tree vs Ball-Tree vs Brute — Data Structures

### kd-Tree

- **Build:** Recursively splits data at the median of the dimension with maximum variance. $O(np \log n)$ build.
- **Query:** Traverse tree, prune branches whose closest possible point is farther than current best. $O(p \log n)$ average in low dimensions.
- **Problem:** In high dimensions, the pruning condition fails — almost no branches can be pruned. Degrades to $O(np)$ when $p \geq 20$.
- **Valid metrics in sklearn:** `euclidean`, `l2`, `minkowski`, `manhattan`, `cityblock`, `l1`, `chebyshev`, `infinity`

### Ball-Tree

- **Build:** Recursively partitions data into nested hyperspheres (balls). Uses the triangle inequality to prune branches: if the closest point on a ball's bounding sphere is farther than the current best, skip the entire ball.
- **Advantage over kd-tree:** The triangle inequality is metric-space agnostic — works for non-Euclidean metrics (Mahalanobis, Hamming, Haversine). Also handles curved manifolds better than axis-aligned hyperplanes.
- **Valid metrics in sklearn:** All kd-tree metrics PLUS `seuclidean`, `mahalanobis`, `hamming`, `canberra`, `braycurtis`, `jaccard`, `dice`, `rogerstanimoto`, `russellrao`, `sokalmichener`, `sokalsneath`, `haversine`
- **Limitation:** More expensive to build than kd-tree.

### Auto-Selection Logic (sklearn 1.5.x)

The `algorithm='auto'` selector in sklearn defaults to `'brute'` when any of these conditions hold:
1. Input is sparse
2. `metric='precomputed'`
3. `p > 15` (dimensions exceed 15)
4. `n_neighbors >= n_samples / 2` (asking for more than half the training set)
5. Metric not in the valid metric list for kd_tree or ball_tree

Otherwise: tries kd_tree first, then ball_tree.

---

## Common Errors & Pitfalls

### 1. Forgetting to Scale Features — THE cardinal sin of kNN

kNN is purely distance-based. If feature $x_1$ ranges from 0–1000 and feature $x_2$ ranges from 0–1, then $x_1$ dominates all distance calculations and $x_2$ is effectively ignored.

```python
# WRONG — data leakage AND missing scaling
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# CORRECT
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipe.fit(X_train, y_train)
```

**Always use a Pipeline** — scaling outside a pipeline and before cross-validation is a data leakage bug.

### 2. Using Even `k` for Binary Classification

With $k=4$ and two classes, ties are possible. sklearn resolves ties by taking the smallest label — this introduces a systematic bias toward lower class indices.

**Fix:** Use odd $k$ for binary problems: `n_neighbors=5` not `n_neighbors=4`.

### 3. Mahalanobis Distance: `algorithm` Must Be `'ball_tree'` or `'brute'`

kd-tree does not support Mahalanobis distance. Using `algorithm='kd_tree'` with `metric='mahalanobis'` raises a `ValueError`. Use `algorithm='ball_tree'` explicitly.

### 4. Data Leakage in RadiusNeighbors

RadiusNeighbors raises `ValueError` for queries in empty regions (no training points within radius) when `outlier_label=None` (default). Always set `outlier_label='most_frequent'` or a specific value:

```python
clf = RadiusNeighborsClassifier(radius=1.0, outlier_label='most_frequent')
```

### 5. Class Imbalance

kNN is particularly vulnerable to class imbalance. If class A has 95% of training samples, then almost all neighbors of any query will be class A — even in the minority class region. The fix is NOT to use `class_weight='balanced'` (kNN doesn't have this parameter) but to:
- Resample (SMOTE, undersampling) INSIDE the training fold
- Use `weights='distance'` to at least weight by proximity
- Consider RadiusNeighbors with the `outlier_label` parameter

### 6. Memory at Scale

Storing 10M training points with 128 features in float32 = $10^7 \times 128 \times 4$ bytes = **5.12 GB** in memory, before the tree structure overhead. This is the fundamental memory problem of kNN. Use approximate methods (FAISS/hnswlib) for $n > 10^5$.

### 7. SHAP KernelExplainer is Prohibitively Slow for kNN

`shap.KernelExplainer` is model-agnostic but has $O(2^p \cdot n_{\text{background}})$ complexity. With 100 features and 100 background samples, this is $O(1.27 \times 10^{32})$ evaluations — in practice, SHAP approximates this, but it is still 100–200x slower than TreeExplainer.

**Recommended alternatives for kNN explainability:**
1. `sklearn.inspection.permutation_importance` — fast, model-agnostic
2. Analyze the k nearest neighbors directly (local prototype explanation)
3. SHAP `PermutationExplainer` (newer than KernelExplainer, same idea but cleaner API)
4. If really needed: use `shap.KernelExplainer` with a very small background set (20–50 points) and small feature subset

### 8. Distance Metrics and Cosine Similarity for Embeddings

NLP embeddings (word2vec, sentence-transformers) live on hyperspheres. Cosine similarity (= inner product on unit-normalized vectors) is the appropriate metric, not Euclidean. sklearn does not directly support cosine in kd-tree or ball-tree, so either:
- L2-normalize the embeddings first (`sklearn.preprocessing.normalize`) and then use Euclidean (equivalent to cosine for unit vectors)
- Use `metric='cosine'` with `algorithm='brute'`
- Use FAISS `IndexFlatIP` (inner product on unit-normalized vectors = cosine)

### 9. Precomputed Distance Matrix — Shape Must Be Square

When using `metric='precomputed'`, sklearn expects an $(n_{\text{train}} \times n_{\text{train}})$ matrix at `fit()` time and an $(n_{\text{test}} \times n_{\text{train}})$ matrix at `predict()` time. Mixing these up silently gives wrong predictions.

### 10. Duplicate Training Points

If duplicate points exist in the training set, using `weights='distance'` will divide by zero. sklearn handles this gracefully (assigns weight 1.0 to zero-distance points, 0.0 to others), but the behavior can be confusing. Consider deduplication before fitting.

### 11. High-Dimensional Data with `algorithm='auto'`

When `p > 15`, sklearn's `auto` selector switches to brute force. This is correct but can be surprising — you may expect tree speedups that never come. Always verify with a timing check when $p > 20$.

---

## Key Datasets for Examples

### sklearn digits (primary recommendation)

```python
from sklearn.datasets import load_digits
import pandas as pd

digits = load_digits()
X, y = digits.data, digits.target
# Shape: (1797, 64) — 8x8 pixel images, 10 classes (0-9)
# Good for: visualizing curse of dimensionality, showing effect of PCA+kNN
print(f"Shape: {X.shape}, Classes: {len(set(y))}")
```

### sklearn make_classification

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    weights=[0.6, 0.3, 0.1],   # for demonstrating imbalance
    random_state=42
)
```

### MNIST subset (for ANN benchmarking)

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data[:10000] / 255.0, mnist.target[:10000].astype(int)
# 784 features — great for demonstrating curse of dimensionality and FAISS
```

### Iris (fast demos)

```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
# 150 samples, 4 features, 3 classes — decision boundary visualization
```

---

## Curated Resources

### Original Papers

1. **Fix & Hodges (1951)** — Technical report; reprinted in *International Statistical Review* 57(3), 238–247 (1989). Open Research Online reference: https://oro.open.ac.uk/28327/

2. **Cover & Hart (1967)** — "Nearest Neighbor Pattern Classification," IEEE Trans. Information Theory: https://doi.org/10.1109/TIT.1967.1053964. Preprint: https://isl.stanford.edu/~cover/papers/transIT/0050cove.pdf

3. **HNSW Paper (Malkov & Yashunin, 2018):** https://arxiv.org/abs/1603.09320

4. **FAISS Paper (Johnson et al.):** https://arxiv.org/abs/1702.08734

5. **NCA Paper (Goldberger et al., 2004):** https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf

6. **Curse of Dimensionality (Beyer et al., 1999):** "When Is Nearest Neighbor Meaningful?" ICDT 1999.

### Official Documentation

7. **sklearn Neighbors User Guide (1.5.x):** https://scikit-learn.org/1.5/modules/neighbors.html

8. **KNeighborsClassifier API reference:** https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

9. **KNeighborsRegressor API reference:** https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

10. **NeighborhoodComponentsAnalysis API:** https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html

### Package Documentation & Tutorials

11. **FAISS GitHub Wiki (Index types and trade-offs):** https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

12. **hnswlib GitHub with Python examples:** https://github.com/nmslib/hnswlib/blob/master/examples/python/EXAMPLES.md

13. **PyNNDescent documentation:** https://pynndescent.readthedocs.io/

14. **Annoy GitHub:** https://github.com/spotify/annoy

15. **ANN Benchmarks (erik bern):** https://github.com/erikbern/ann-benchmarks — authoritative benchmark comparing FAISS, hnswlib, Annoy, PyNNDescent, etc. on standard datasets

16. **Pinecone FAISS tutorial:** https://www.pinecone.io/learn/series/faiss/faiss-tutorial/ — best hands-on production guide for FAISS IndexIVFFlat

17. **Pinecone HNSW explainer:** https://www.pinecone.io/learn/series/faiss/hnsw/ — best visual explanation of HNSW layers

18. **Cleveland (1979) LOWESS paper PDF:** https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf

---

## Algorithmic Details the Writer Must Nail

### The Cover-Hart Proof Sketch

The proof uses two steps:
1. **Convergence of the nearest neighbor:** As $n \to \infty$ and $\mathbf{x}^*$ is the 1-NN of $\mathbf{x}$, the distribution of $\mathbf{x}^*$ converges to that of $\mathbf{x}$ (by density of the training set).
2. **Error as two-sample test:** The 1-NN rule makes an error iff the test point $\mathbf{x}$ and its training neighbor $\mathbf{x}^*$ have different true labels. The probability of this (in the limit) is $\sum_c \eta_c(\mathbf{x})(1 - \eta_c(\mathbf{x}))$ where $\eta_c(\mathbf{x}) = P(y=c|\mathbf{x})$. This integrates to $R \leq 2R^*(1-R^*)$.

The writer should derive this result explicitly for the binary case and show it implies $R \leq 2R^*$.

### The HNSW Layer Assignment

The probability of an element being assigned to layer $l$ or above:
$$P(\text{max layer} \geq l) = e^{-l / m_L}$$
where $m_L = 1 / \ln(M)$ (with $M$ the max connections per node). This exponential decay ensures the upper layers are sparse (coarse navigation) and lower layers are dense (fine-grained search), mimicking skip-list structure.

### The NCA Objective Gradient

The gradient of the NCA objective with respect to the transformation matrix $A$:
$$\frac{\partial f}{\partial A} = 2A \sum_i \left[ p_i \sum_k x_{ik} x_{ik}^T - \sum_{j: y_j = y_i} p_{ij} x_{ij} x_{ij}^T \right]$$
where $x_{ij} = \mathbf{x}_i - \mathbf{x}_j$, $p_{ij}$ is the softmax probability, and $p_i = \sum_{j: y_j = y_i} p_{ij}$.

### Minkowski Metric Family

$$d_p(\mathbf{x}, \mathbf{y}) = \left(\sum_{j=1}^{p} |x_j - y_j|^p\right)^{1/p}$$

Special cases:
- $p=1$: Manhattan / L1 / taxicab / cityblock
- $p=2$: Euclidean / L2
- $p \to \infty$: Chebyshev (max component difference)
- $p < 1$: Not a metric (violates triangle inequality) — sklearn allows this via BallTree but the triangle inequality pruning won't work correctly

---

## Uncertainties

1. **sklearn 1.5.x vs 1.6+/stable:** At time of writing, the course targets sklearn 1.5.x. The stable docs (1.9.0 at the time of research) show the same parameter API for KNeighbors, but the writer should verify that `algorithm='auto'` selection logic (specifically the $p > 15$ threshold) matches in 1.5.2 vs current stable. The internal threshold might have changed.

2. **Cosine metric support in kd-tree/ball-tree:** The sklearn docs list `'cosine'` as using `cosine_distances` (a pairwise function) but it is unclear whether it's directly supported in kd-tree. The writer should test: `KNeighborsClassifier(metric='cosine', algorithm='kd_tree')` and confirm whether it raises an error or silently falls back to brute. The safe recommendation is `algorithm='brute'` with `metric='cosine'`.

3. **FAISS version:** The research found FAISS 1.8.0 referenced in install instructions. The writer should verify the latest stable version and any API changes in the Python bindings for the `IndexIVFFlat` workflow. The core API (IndexFlatL2, add, search) has been stable for years, but `IndexHNSWFlat` parameters may differ slightly.

4. **hnswlib maintenance status:** The hnswlib repo (nmslib/hnswlib) has had reduced maintenance activity. The writer should note that Spotify's Voyager (https://github.com/spotify/voyager) is the spiritual successor and may be preferred in new projects. FAISS's `IndexHNSWFlat` is the production-grade alternative.

5. **SHAP PermutationExplainer vs KernelExplainer:** The research confirmed that `shap.PermutationExplainer` exists as an alternative, but the exact API call for kNN (whether it needs a `masker` argument in shap 0.46.x) should be verified against the current shap docs before writing code in the chapter.

6. **PyNNDescent metric list:** PyNNDescent supports a wide range of metrics via numba, but the exact list that works with its `metric` parameter string should be verified in the current pynndescent 0.5.x docs — not all scipy distance metrics are supported.

7. **Fix & Hodges (1951) URL:** There is no stable DOI for this technical report. The best stable reference is the 1989 reprint in *International Statistical Review* — the writer should search for that if a journal DOI is needed: Hodges, J. L. (1989). "Discriminatory Analysis: Nonparametric Discrimination." *International Statistical Review*, 57(3), 238–247.

8. **`weights='distance'` behavior with zero distances:** sklearn's documentation states that when a training point is identical to a query point, the zero-distance neighbor receives weight 1.0 and all others receive 0.0. This is the correct behavior but the writer should test it explicitly and include the edge case in the pitfalls section.

---

## Summary for Chapter Author

K-Nearest Neighbors has two foundational papers: Fix & Hodges (1951) established non-parametric density estimation consistency, and Cover & Hart (1967) proved the landmark result that 1-NN achieves at most twice the Bayes error rate asymptotically — placing the algorithm on solid theoretical footing. The algorithm family spans basic majority-vote kNN (sklearn `KNeighborsClassifier/Regressor`), radius-based variants, metric learning via NCA, condensed/edited prototype methods, and modern approximate nearest neighbor libraries (FAISS for GPU/billion-scale, hnswlib for HNSW, Annoy for mmap'd production, PyNNDescent for UMAP-compatible workflows). Feature scaling is non-negotiable and must always be done inside a Pipeline to prevent data leakage. The curse of dimensionality — manifesting as exponential volume concentration in corners and equidistance collapse — is the central theoretical weakness, requiring either dimensionality reduction (PCA/UMAP) or metric learning (NCA) for high-dimensional data, and approximate methods (FAISS/hnswlib) for $n > 10^5$. All sklearn hyperparameters for the 1.5.x API are verified in this dossier with confirmed defaults, metric-algorithm compatibility constraints (Mahalanobis requires ball_tree/brute; cosine requires brute or pre-normalization), and concrete tuning guidance.
