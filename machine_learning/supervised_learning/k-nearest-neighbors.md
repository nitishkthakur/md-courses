# K-Nearest Neighbors: The Complete Masterclass

> **Why this algorithm matters:** K-Nearest Neighbors is the purest expression of a deceptively
> simple idea: *things that are close together tend to be alike*. That one sentence captures the
> operating principle of recommendation engines at Netflix and Spotify, image retrieval at Google
> Photos, anomaly detection in industrial sensor networks, and embedding search at every major
> language model provider. The "vector database" industry — now worth billions — is fundamentally
> a highly engineered version of the same nearest-neighbor search that Fix and Hodges sketched out
> in an internal US Air Force technical report in 1951. Cover and Hart proved in 1967 that even the
> simplest 1-NN rule can achieve at most *twice* the Bayes error rate using zero distributional
> assumptions — a theoretical guarantee that no parametric model of that era could match without
> being right about the data-generating distribution. Understanding kNN deeply means understanding
> distance geometry, the curse of dimensionality, lazy learning, approximate search at scale, and
> the limits of non-parametric reasoning. These are not niche topics — they are foundational to
> how modern ML systems actually run in production.

---

## §0 — Python Ecosystem & Package Guide

The kNN ecosystem is wider than most practitioners realize. On the sklearn side, there are five
distinct estimators for different query types. On the approximate nearest neighbor (ANN) side,
a separate universe of highly optimized libraries handles the billion-scale regime where exact
sklearn search would be prohibitively slow. Understanding the full map prevents you from
reaching for the wrong tool.

### Complete Package Table

| Package | Import | Version (mid-2026) | What it provides | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.neighbors import KNeighborsClassifier` | ~1.5.x | Exact kNN (kd-tree, ball-tree, brute); 8 estimators; full metric library | BSD-3 |
| `scikit-learn` | `from sklearn.neighbors import KNeighborsRegressor` | ~1.5.x | kNN regression; same backend as classifier | BSD-3 |
| `scikit-learn` | `from sklearn.neighbors import RadiusNeighborsClassifier` | ~1.5.x | Radius-based classification; variable neighborhood size | BSD-3 |
| `scikit-learn` | `from sklearn.neighbors import RadiusNeighborsRegressor` | ~1.5.x | Radius-based regression | BSD-3 |
| `scikit-learn` | `from sklearn.neighbors import NearestNeighbors` | ~1.5.x | Unsupervised NN retrieval; `.kneighbors()` and `.radius_neighbors()` | BSD-3 |
| `scikit-learn` | `from sklearn.neighbors import NeighborhoodComponentsAnalysis` | ~1.5.x | Metric learning — learns a linear transform to maximize kNN accuracy | BSD-3 |
| `faiss-cpu` | `import faiss` | ~1.8.x | Meta's billion-scale exact/approx NN; IVF, PQ, HNSW indexes; C++/CUDA | MIT |
| `faiss-gpu` | `import faiss` | ~1.8.x | GPU-accelerated version of FAISS; requires CUDA | MIT |
| `hnswlib` | `import hnswlib` | ~0.8.x | C++ HNSW reference implementation; Python bindings; sub-ms latency | Apache-2 |
| `annoy` | `from annoy import AnnoyIndex` | ~1.17.x | Spotify's random projection tree ANN; mmap'd indexes for multi-process | Apache-2 |
| `pynndescent` | `import pynndescent` | ~0.5.x | Pure Python/Numba NN descent; used internally by UMAP | BSD-2 |
| `imbalanced-learn` | `from imblearn.under_sampling import EditedNearestNeighbours` | ~0.12.x | ENN and CNN prototype selection; SMOTE uses NN internally | MIT |
| `statsmodels` | `from statsmodels.nonparametric.smoothers_lowess import lowess` | ~0.14.x | LOWESS: locally weighted regression (the polynomial-fitting kNN cousin) | BSD-3 |
| `shap` | `import shap` | ~0.46.x | KernelExplainer (model-agnostic, slow) or PermutationExplainer for kNN | MIT |
| `optuna` | `import optuna` | ~3.x | Hyperparameter optimization for `n_neighbors`, `p`, `weights`, `metric` | MIT |

### Top Picks — Recommendation Table

| Package | Best for | Speed tier | Notes |
|---|---|---|---|
| **`sklearn.KNeighborsClassifier`** | Standard ML pipelines, $n < 100\text{k}$ | Medium (exact, brute/tree) | **First choice**; Pipeline-compatible; full metric support |
| **`sklearn.KNeighborsRegressor`** | Regression on tabular data, $n < 100\text{k}$ | Medium | Same backend as classifier |
| **`hnswlib`** | Production sub-ms retrieval, $n > 100\text{k}$, static index | Very fast (~$O(\log n)$) | Best recall/speed tradeoff; HNSW; de facto ANN standard |
| **`faiss`** | Billion-scale, GPU acceleration, memory-constrained with PQ | Extreme (GPU) | Best for scale; complex API; many index types |
| **`pynndescent`** | UMAP workflows; wide metric support; no C++ compilation | Fast | Easy install; great recall at >90% target |

### The Same Fit Across Top Packages

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Shared data setup — used throughout this chapter
iris = load_iris()
X, y = iris.data, iris.target   # (150, 4) — 3 classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Option 1: sklearn KNeighborsClassifier (the workhorse) ────────────────
from sklearn.neighbors import KNeighborsClassifier

pipe_sklearn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto'))
])
pipe_sklearn.fit(X_train, y_train)
print(f"sklearn accuracy: {pipe_sklearn.score(X_test, y_test):.4f}")

# ── Option 2: hnswlib (approximate, production-grade) ─────────────────────
# pip install hnswlib
import hnswlib

scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train).astype('float32')
X_test_sc  = scaler.transform(X_test).astype('float32')

index = hnswlib.Index(space='l2', dim=X_train_sc.shape[1])
index.init_index(max_elements=len(X_train_sc), ef_construction=200, M=16)
index.add_items(X_train_sc, np.arange(len(X_train_sc)))
index.set_ef(50)   # query-time recall parameter

labels_hnsw, distances_hnsw = index.knn_query(X_test_sc, k=5)
# Majority-vote over returned training indices
from scipy.stats import mode
preds_hnsw = np.array([mode(y_train[nbrs])[0] for nbrs in labels_hnsw])
print(f"hnswlib accuracy: {(preds_hnsw == y_test).mean():.4f}")

# ── Option 3: FAISS (exact brute-force, GPU-ready) ────────────────────────
# pip install faiss-cpu
import faiss

d = X_train_sc.shape[1]
index_faiss = faiss.IndexFlatL2(d)          # exact L2 brute-force
index_faiss.add(X_train_sc)
k = 5
_, I = index_faiss.search(X_test_sc, k)    # I: (n_test, k) training indices

preds_faiss = np.array([mode(y_train[nbrs])[0] for nbrs in I])
print(f"FAISS (exact) accuracy: {(preds_faiss == y_test).mean():.4f}")

# ── Option 4: PyNNDescent (wide metric support) ───────────────────────────
# pip install pynndescent
import pynndescent

index_pynn = pynndescent.NNDescent(X_train_sc, n_neighbors=5, metric='euclidean',
                                    random_state=42)
indices_pynn, _ = index_pynn.query(X_test_sc, k=5)
preds_pynn = np.array([mode(y_train[nbrs])[0] for nbrs in indices_pynn])
print(f"PyNNDescent accuracy: {(preds_pynn == y_test).mean():.4f}")
```

> 💡 **Intuition:** All four packages are computing the same thing — finding the $k$ training
> points closest to each query in feature space and aggregating their labels. The difference is
> purely engineering: sklearn is exact and simple; hnswlib sacrifices tiny amounts of recall for
> 100–1000x speedup; FAISS adds GPU acceleration and compression options; PyNNDescent requires
> no C++ compilation and supports more exotic metrics.

> 🔧 **In practice:** For $n < 50{,}000$ and $p < 20$, sklearn with `algorithm='auto'` is the
> correct choice — fast enough, exact, and Pipeline-compatible. Above $n = 100{,}000$, you need
> an ANN library. For production embedding search with static indexes, hnswlib (`hnswlib`) is
> the current gold standard. For billion-scale or GPU workloads, FAISS is the only option.

> ⚠️ **Pitfall:** Forgetting to scale features before kNN is the single most common mistake.
> kNN is *entirely* distance-based: a feature with range 0–1000 will dwarf a feature with
> range 0–1, effectively making the latter invisible. **Always use a `Pipeline` with
> `StandardScaler` or `MinMaxScaler`** — never scale outside the pipeline, as that introduces
> data leakage in cross-validation.

---

## §1 — The Origin: The Papers That Started It All

### Before kNN: The Tyranny of Parametric Assumptions

To appreciate what Fix and Hodges created, you need to feel the intellectual prison of
classification methods in 1951.

The dominant framework was Fisher's Linear Discriminant Analysis (1936). Fisher's LDA is
elegant and powerful — under the assumption that each class follows a multivariate Gaussian
distribution with the *same* covariance matrix $\Sigma$. Under those conditions, the optimal
classifier is a linear boundary, and LDA finds it exactly. But what if the classes are *not*
Gaussian? What if the decision boundary is curved, or the classes are multi-modal? LDA
would still find a linear boundary and accept the resulting suboptimality as the cost of
its assumptions.

The alternative — a fully non-parametric approach — seemed computationally impossible in
the pre-digital age. How do you classify without assuming a distribution?

### Paper 1: Fix & Hodges (1951) — The Birth of Non-Parametric Classification

> 📜 **Origin/Citation:** Fix, E., and Hodges, J. L. (1951). *Discriminatory Analysis,
> Nonparametric Discrimination: Consistency Properties.* Technical Report 4, Project
> 21-49-004, USAF School of Aviation Medicine, Randolph Field, Texas. Reprinted posthumously
> in *International Statistical Review*, 57(3), 238–247 (1989).
> URL: https://oro.open.ac.uk/28327/

Evelyn Fix and Joseph Hodges were a statistician and her graduate student at UC Berkeley,
commissioned by the US Air Force to develop better methods for classifying aircraft
maneuvers from sensor data — a binary life-or-death classification problem. Their report
was never formally peer-reviewed in their lifetimes and was nearly forgotten until
reprinted 38 years later. Yet it contains one of the foundational ideas of modern machine
learning.

**The core idea.** Suppose you want to estimate the class-conditional density
$p(\mathbf{x} | C_j)$ at a query point $\mathbf{x}$ without assuming any parametric form.
The idea: draw a small ball of radius $r$ around $\mathbf{x}$ and count training points.
If $k_j$ of the $k$ nearest neighbors belong to class $j$, and the ball has volume
$V_k(\mathbf{x})$, then:

$$\hat{p}(\mathbf{x} | C_j) \propto \frac{k_j}{n_j \cdot V_k(\mathbf{x})}$$

where $n_j$ is the total number of training samples in class $j$. The **1-NN rule** is
the simplest case: assign the query to the class of its single nearest training point.

**The consistency result.** Fix and Hodges proved that as $n \to \infty$ with $k \to \infty$
and $k/n \to 0$, the estimated density converges to the true density — *without any
parametric assumption*. This is the non-parametric consistency result. The conditions
$k \to \infty$ (enough neighbors to estimate density) and $k/n \to 0$ (the neighborhood
shrinks as a fraction of the data, so we're truly estimating a local density) are crucial.
In practice, for finite samples, this means: choose $k$ large enough for stability but
small enough to capture local structure.

**What was unknown in 1951:** The report made no claim about how fast this convergence
occurs, gave no error rate bounds, and did not address the computational cost of computing
all pairwise distances. The curse of dimensionality — the exponential growth of required
data with feature dimension — was not yet formalized. These would be answered in the
decades to come.

**The formal algorithm (Fix & Hodges, 1951):**

Given training set $\mathcal{T} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, query $\mathbf{x}_q$,
and class set $\mathcal{C} = \{C_1, \ldots, C_J\}$:

1. Compute $d(\mathbf{x}_q, \mathbf{x}_i)$ for all $i = 1, \ldots, n$
2. Sort training points by distance: $d_{(1)} \leq d_{(2)} \leq \ldots \leq d_{(n)}$
3. Select the $k$ points with smallest distances: $\mathcal{N}_k(\mathbf{x}_q) = \{(1), (2), \ldots, (k)\}$
4. For each class $C_j$, count $k_j = \sum_{i \in \mathcal{N}_k} \mathbf{1}[y_i = C_j]$
5. Classify: $\hat{y} = \arg\max_j k_j$ (majority vote)

### Paper 2: Cover & Hart (1967) — The Landmark Error-Rate Bound

> 📜 **Origin/Citation:** Cover, T. M., and Hart, P. E. (1967). Nearest neighbor pattern
> classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
> DOI: https://doi.org/10.1109/TIT.1967.1053964
> Preprint: https://isl.stanford.edu/~cover/papers/transIT/0050cove.pdf

Sixteen years after Fix and Hodges, Thomas Cover (later of information theory fame — the
Cover-Thomas textbook) and Peter Hart provided the theoretical result that made kNN
respectable: a hard upper bound on its error rate, relative to the theoretical best possible.

**Context: The Bayes error rate.** In statistical decision theory, the **Bayes error rate**
$R^*$ is the irreducible minimum error achievable by any classifier with perfect knowledge
of the true class-conditional distributions:

$$R^* = 1 - \int \max_c P(y = c \mid \mathbf{x}) \, p(\mathbf{x}) \, d\mathbf{x}$$

No algorithm can do better than $R^*$. The question Cover and Hart asked: how close can
the 1-NN rule come to $R^*$, without knowing anything about the distributions?

**The Cover-Hart bound.** Let $R$ be the asymptotic error rate of the 1-NN rule with $C$
classes. Then:

$$R^* \leq R \leq R^*\left(2 - \frac{C}{C-1} R^*\right)$$

For binary classification ($C = 2$):

$$R^* \leq R \leq 2R^*(1 - R^*)$$

Since $R^* \leq 1/2$ for any reasonable problem, we get $2R^*(1 - R^*) \leq 2R^*$, yielding
the famous statement:

$$\boxed{R \leq 2R^*}$$

The 1-NN rule achieves **at most twice the Bayes error rate**, using only training data
and zero distributional assumptions. If the Bayes error is 5%, 1-NN is guaranteed to
achieve at most 10% — even knowing nothing about the underlying distribution. This was
a shocking result in 1967.

**The proof sketch — the binary case.** The proof uses two steps:

*Step 1: Convergence of the nearest neighbor.* As $n \to \infty$, the 1-NN $\mathbf{x}^*$
of any query $\mathbf{x}$ converges in distribution to $\mathbf{x}$ itself (the training
set becomes dense). At the limit, $\mathbf{x}^*$ and $\mathbf{x}$ are essentially
independent draws from the same local distribution.

*Step 2: Error as a two-sample test.* The 1-NN rule makes an error when $\mathbf{x}$ and
its nearest neighbor $\mathbf{x}^*$ have different true labels. Let
$\eta(\mathbf{x}) = P(y=1 \mid \mathbf{x})$ be the Bayes posterior. In the limit:

$$R = \int \left[ \eta(\mathbf{x})(1 - \eta(\mathbf{x})) + (1 - \eta(\mathbf{x}))\eta(\mathbf{x}) \right] p(\mathbf{x}) \, d\mathbf{x} = 2\int \eta(\mathbf{x})(1-\eta(\mathbf{x})) \, p(\mathbf{x}) \, d\mathbf{x}$$

The Bayes error is $R^* = \int \min(\eta(\mathbf{x}), 1-\eta(\mathbf{x})) \, p(\mathbf{x}) \, d\mathbf{x}$.

Using $2\eta(1-\eta) \leq 2\min(\eta, 1-\eta)$ (easily verified: $\min(\eta, 1-\eta)$ equals
the smaller of the two, while $\eta(1-\eta) \leq \frac{1}{4}$, and the inequality holds
pointwise), we get $R \leq 2R^*$.

> 💡 **Intuition:** The 1-NN rule makes an error only when the query and its training
> shadow have different labels — like flipping two biased coins with the same probability
> $\eta(\mathbf{x})$ and asking if they disagree. The probability of disagreement is
> $2\eta(1-\eta)$, which is always $\leq 2\min(\eta, 1-\eta) = 2R^*_{\text{local}}$.
> Integrate over all query points and you get the bound.

**For $k > 1$:** Cover and Hart also showed that as $k \to \infty$ with $k/n \to 0$,
the kNN error rate converges to the Bayes error $R^*$ itself. More neighbors means
better density estimation, lower variance, and asymptotic optimality. The finite-$n$
bias-variance tradeoff determines the optimal $k$ in practice.

**Historical impact.** Before Cover-Hart, kNN was viewed as a computational curiosity
with no theoretical guarantees. After Cover-Hart, it became a theoretically grounded
baseline that any new algorithm had to surpass. The paper has been cited over 8,000 times
and directly inspired decades of research into non-parametric classification theory.

---

## §2 — The Algorithm, Deeply Explained

### The Core Prediction Formula

kNN is a **lazy learner** — it does no work at training time beyond storing the data.
All computation happens at prediction time. Given:
- Training set $\mathcal{T} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, $\mathbf{x}_i \in \mathbb{R}^p$
- A distance metric $d: \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}_{\geq 0}$
- A query point $\mathbf{x}_q$
- Hyperparameter $k$ (number of neighbors)

**Step 1: Find the $k$ nearest neighbors.**

$$\mathcal{N}_k(\mathbf{x}_q) = \arg\text{kmin}_{i \in \{1,\ldots,n\}} \, d(\mathbf{x}_q, \mathbf{x}_i)$$

where $\arg\text{kmin}$ returns the indices of the $k$ smallest values.

**Step 2a: Classification — majority vote.**

$$\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_k(\mathbf{x}_q)} \mathbf{1}[y_i = c]$$

**Step 2b: Regression — arithmetic mean.**

$$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x}_q)} y_i$$

**Step 2c: Distance-weighted prediction (both classification and regression).**

With weights $w_i = 1 / d(\mathbf{x}_q, \mathbf{x}_i)^2$ (or $1/d_i$):

$$\hat{y}_{\text{reg}} = \frac{\sum_{i \in \mathcal{N}_k} w_i y_i}{\sum_{i \in \mathcal{N}_k} w_i}$$

$$\hat{y}_{\text{clf}} = \arg\max_{c} \sum_{i \in \mathcal{N}_k} w_i \cdot \mathbf{1}[y_i = c]$$

**Predicted class probabilities** (for `predict_proba`):

$$\hat{P}(y = c \mid \mathbf{x}_q) = \frac{\sum_{i \in \mathcal{N}_k} w_i \cdot \mathbf{1}[y_i = c]}{\sum_{i \in \mathcal{N}_k} w_i}$$

With uniform weights, this is simply $k_c / k$ where $k_c$ is the count of class-$c$
neighbors.

> 💡 **Intuition 1: The voting booth.** Classification with kNN is like polling the $k$
> nearest "experts" in your neighborhood. Each expert votes for their known label. The
> majority wins. If you're standing in a neighborhood where 4 of 5 nearest houses have
> red doors, you'll predict red — regardless of houses farther away.

> 💡 **Intuition 2: Local averaging.** Regression with kNN is non-parametric locally
> constant regression. It computes a local mean within a neighborhood that adapts to
> wherever the query falls in feature space. No global functional form is imposed — the
> "model" is just the training data.

> 💡 **Intuition 3: The kernel perspective.** kNN is equivalent to a kernel regression
> with a **rectangular (box) kernel** of adaptive bandwidth. The bandwidth adjusts so
> that exactly $k$ points are included. Compare with Nadaraya-Watson kernel regression,
> which uses a fixed-bandwidth smooth kernel. kNN trades the smoothness of the kernel
> for computational simplicity.

### Distance Metrics — The Minkowski Family and Beyond

The choice of distance metric is as important as the choice of $k$. The most common family
is the **Minkowski metric**:

$$d_p(\mathbf{x}, \mathbf{y}) = \left(\sum_{j=1}^{p} |x_j - y_j|^p\right)^{1/p}$$

Special cases:

| $p$ value | Distance name | Geometric meaning |
|---|---|---|
| $p = 1$ | Manhattan (L1, taxicab, cityblock) | Sum of absolute differences; robust to outliers in individual features |
| $p = 2$ | Euclidean (L2) | Straight-line distance; most common; sensitive to scale |
| $p \to \infty$ | Chebyshev | Maximum absolute difference across features |
| $0 < p < 1$ | Fractional | Not a metric (violates triangle inequality); sometimes useful in very high dimensions |

**When Euclidean breaks down.** Consider two features: annual salary ($0$–$200{,}000$)
and age ($18$–$70$). Without scaling, a $\$1{,}000$ difference in salary equals
$1{,}000$ units of Euclidean distance, while a $50$-year age difference contributes
only $50$ units. Salary dominates completely.

**The Mahalanobis distance** generalizes Euclidean to account for feature correlations
and scales:

$$d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{y})}$$

where $\mathbf{\Sigma}$ is the feature covariance matrix estimated from training data.
Mahalanobis distance is scale-invariant and decorrelates features — it's equivalent to
computing Euclidean distance in the whitened feature space. In sklearn, use
`metric='mahalanobis'` with `algorithm='ball_tree'` and `metric_params={'VI': inv_cov}`.

> 🔬 **Deep dive:** The Mahalanobis distance is not available in kd-tree because kd-tree
> relies on axis-aligned hyperplane splits. Mahalanobis distance corresponds to a
> Gaussian-shaped neighborhood that is elliptical, not axis-aligned. Ball-tree uses the
> metric-space triangle inequality for pruning, which works for any valid metric — hence
> it supports Mahalanobis. If using Mahalanobis, always compute `inv_cov` on the training
> set only (inside a Pipeline-compatible transformer or by manually fitting before CV).

**Cosine distance** for high-dimensional embeddings:

$$d_{\cos}(\mathbf{x}, \mathbf{y}) = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$$

For NLP embeddings (word2vec, sentence-transformers, OpenAI embeddings), points lie
on a unit hypersphere. Cosine similarity is the appropriate measure. In sklearn, use
`metric='cosine'` with `algorithm='brute'`, or pre-normalize vectors with
`sklearn.preprocessing.normalize` and use Euclidean (they are equivalent on unit-norm vectors).

### Computational Complexity

| Operation | Brute force | kd-tree | Ball-tree |
|---|---|---|---|
| Training (fit) | $O(1)$ | $O(np \log n)$ | $O(np \log n)$ |
| Single query | $O(np)$ | $O(p \log n)$† | $O(p \log n)$† |
| $m$ queries | $O(mnp)$ | $O(mp \log n)$† | $O(mp \log n)$† |
| Memory | $O(np)$ | $O(np)$ + tree | $O(np)$ + tree |

†: Average case in low dimensions ($p \leq 10$). Degrades to $O(np)$ as $p$ grows above ~15–20.

> 🔧 **In practice:** The $O(1)$ training complexity is kNN's defining feature — "lazy
> learning" means no optimization loop, no convergence criteria, no warm-up. But you pay
> dearly at inference. For large $n$, the kd-tree or ball-tree can save orders of
> magnitude, but only in low-to-moderate dimensions. For $p > 20$, these tree structures
> degrade toward brute force, and you should either (1) reduce dimensionality first with
> PCA, or (2) use an approximate nearest neighbor library.

### Implicit Assumptions

kNN makes very few explicit assumptions — that is its strength and its weakness:

1. **Local smoothness:** The true label function $f(\mathbf{x})$ must vary slowly in local
   neighborhoods. If the class boundary is fractal or non-Lipschitz at the scale of your
   neighborhood, kNN will fail catastrophically.

2. **Euclidean-ish structure:** The chosen distance metric must meaningfully rank sample
   similarity. Features must be comparable in scale (handled by scaling) and the chosen
   metric must align with the problem geometry.

3. **Enough samples relative to dimension:** The fundamental requirement from consistency
   theory — $k \to \infty$ and $k/n \to 0$ — implies that $n$ must grow faster than $k$.
   For good finite-sample performance, you need enough samples for neighborhoods to be
   locally representative of the true distribution.

4. **No global structure:** kNN has no global model. It cannot extrapolate beyond the
   training manifold, and it cannot model global trends (like a linear relationship across
   the entire feature space) efficiently — it would need many training points everywhere.

Where assumption 1 breaks: discontinuous functions, label noise (a single misclassified
training point in the wrong region creates a "decision island"), or natural data with
sharp boundaries between classes at fine scales.

Where assumption 3 breaks: the curse of dimensionality, discussed in §6.

### The kd-Tree Data Structure — Deep Dive

Understanding the kd-tree and ball-tree algorithms lets you reason about when `algorithm='auto'`
will actually use them and when it silently falls back to brute force.

**kd-Tree construction (Bentley, 1975).** A kd-tree is a binary tree that recursively partitions
$p$-dimensional space by axis-aligned hyperplanes. At each internal node:

1. Select the dimension $j^*$ with maximum variance among remaining points
2. Split at the median value $m^*$ along dimension $j^*$
3. Points with $x_{j^*} \leq m^*$ go left; others go right
4. Recurse until $\leq$ `leaf_size` points remain in a node

A query for the nearest neighbor of $\mathbf{x}_q$ works by:
1. Navigate to the leaf containing $\mathbf{x}_q$ (following the hyperplane decisions)
2. The points in that leaf are the initial "current best" candidates
3. Backtrack up the tree, checking if the *other* side of each hyperplane can contain
   a closer point — if $d(\mathbf{x}_q, \text{hyperplane}) \geq d_{\text{current best}}$,
   prune that entire subtree
4. This pruning makes the average-case query $O(\log n)$ in low dimensions

**Why kd-trees fail in high dimensions.** The pruning condition checks whether the
hypersphere of radius $d_{\text{current best}}$ around $\mathbf{x}_q$ intersects the
hyperplane. In high dimensions, nearly every hyperplane is within distance
$d_{\text{current best}}$ of the query (because the nearest neighbor is far away due to
the curse). The pruning condition is almost never satisfied — almost every subtree must
be searched. The effective complexity reverts to $O(np)$.

The empirical threshold: kd-tree beats brute force for $p \leq 10$–$15$. At $p = 20$,
kd-tree is typically slower than brute force on CPU.

**Ball-Tree construction.** Instead of axis-aligned hyperplanes, ball-tree partitions
data into nested hyperspheres. At each internal node:
1. Find the feature $j^*$ with maximum spread
2. Pick the two "farthest apart" pivot points $p_1, p_2$ (by maximum distance)
3. Assign each training point to $p_1$ or $p_2$ based on which it is closer to
4. Recurse

Each node stores the center and radius of the enclosing ball. Pruning uses the
**triangle inequality**: if the minimum distance from $\mathbf{x}_q$ to any point in
a ball (= distance to ball center minus ball radius) exceeds $d_{\text{current best}}$,
skip that entire ball.

**Ball-tree advantage:** The triangle inequality is metric-space agnostic — it holds for
any valid metric, including Mahalanobis, Hamming, and Haversine. kd-trees cannot support
these metrics (their splits are axis-aligned Euclidean). Ball-trees also handle
high-dimensional data slightly better because ball-shaped enclosures match the geometry
of the data better than axis-aligned boxes in high dimensions — though they still suffer
at $p > 20$.

### The Bias-Variance Tradeoff in kNN — Formal Statement

For a fixed query $\mathbf{x}_q$, the mean squared error of the kNN regressor decomposes as:

$$\text{MSE}(\hat{f}_k(\mathbf{x}_q)) = \underbrace{\text{Bias}^2\left[\hat{f}_k(\mathbf{x}_q)\right]}_{\text{increases with } k} + \underbrace{\text{Var}\left[\hat{f}_k(\mathbf{x}_q)\right]}_{\text{decreases with } k} + \sigma^2_\epsilon$$

**Bias term:** The bias of the kNN estimate at $\mathbf{x}_q$ comes from the fact that we
average over a neighborhood — if $f$ varies within the neighborhood, the average is biased
away from $f(\mathbf{x}_q)$:

$$\text{Bias}[\hat{f}_k(\mathbf{x}_q)] = \frac{1}{k} \sum_{i \in \mathcal{N}_k} f(\mathbf{x}_i) - f(\mathbf{x}_q) \approx \frac{1}{2p} \nabla^2 f(\mathbf{x}_q) \cdot r_k^2$$

where $r_k$ is the radius of the $k$-neighborhood. As $k$ increases, $r_k$ grows (larger
neighborhoods), bias increases quadratically in $r_k$.

**Variance term:** With $k$ training targets $y_i = f(\mathbf{x}_i) + \epsilon_i$:

$$\text{Var}[\hat{f}_k(\mathbf{x}_q)] = \frac{\sigma_\epsilon^2}{k}$$

Variance decreases exactly as $1/k$ — doubling $k$ halves the variance.

**The optimal $k$:** Balancing the bias-variance tradeoff gives an optimal $k$ that grows
as $k^* \sim n^{4/(4+p)}$ — a result from non-parametric regression theory that confirms
the curse: in $p$ dimensions, the optimal $k$ (and hence the bias) grows with $n$ more
slowly as $p$ increases.

> 💡 **Intuition:** kNN variance is exactly $\sigma^2/k$ — as clean as you can ask for.
> The variance due to label noise divides by the number of neighbors. This is why using
> $k=1$ is always noisy and $k=n$ is noiseless but hopelessly biased (always predicts
> the global mean).

---

## §3 — The Full Evolution: Every Major Variant

### 3.1 Weighted kNN (Distance-Weighted Voting)

**Problem solved:** Uniform voting treats a neighbor at distance $0.001$ the same as a
neighbor at distance $100$. Both have one vote. This is geometrically absurd — a point
essentially *at* the query location is infinitely more informative.

**The change:** Replace the uniform weight $w_i = 1/k$ with distance-based weights:

$$w_i = \frac{1}{d(\mathbf{x}_q, \mathbf{x}_i)} \quad \text{or} \quad w_i = \frac{1}{d(\mathbf{x}_q, \mathbf{x}_i)^2}$$

For the zero-distance case (query equals a training point exactly): sklearn assigns $w_i = 1$
to that point and $w_i = 0$ to all others — effectively returning the training label exactly.

**Custom kernel weights:** You can pass any callable to `weights=`:

```python
# Gaussian kernel weighting — smoother than 1/d
def gaussian_weights(distances):
    sigma = distances.mean() + 1e-10   # adaptive bandwidth
    return np.exp(-distances**2 / (2 * sigma**2))

knn = KNeighborsClassifier(n_neighbors=15, weights=gaussian_weights)
```

**When to prefer:** Almost always use `weights='distance'` over `weights='uniform'`. The
only case for uniform: when you're certain all $k$ neighbors are roughly equidistant (which
rarely holds in practice). Distance weighting consistently improves performance, especially
when decision boundaries are non-linear and neighborhood sizes are large.

**sklearn:** `KNeighborsClassifier(weights='distance')` — default is `'uniform'`.

> 📊 **Benchmark:** On the UCI Wine dataset (178 samples, 13 features), `weights='distance'`
> improved kNN accuracy from 74.1% to 77.8% with $k=5$ in a 5-fold CV experiment (a
> consistent ~3–5% relative improvement across many tabular datasets).

---

### 3.2 Radius-Based Neighbors (RadiusNeighbors)

**Problem solved:** Fixed-$k$ kNN has a fundamental pathology in regions of varying density.
In a dense region, $k=5$ neighbors might all be within distance $0.01$ — great. In a sparse
region, those same $k=5$ neighbors might span the entire feature space — terrible. You're
averaging points that are "near" in the sense of being the closest $k$ points, but "far"
in any absolute sense.

**The change:** Fix the radius $r$ instead of $k$. Include all training points within ball
$B(\mathbf{x}_q, r)$:

$$\mathcal{N}_r(\mathbf{x}_q) = \{i : d(\mathbf{x}_q, \mathbf{x}_i) \leq r\}$$

The number of neighbors $|\mathcal{N}_r|$ now varies by query location — large in dense
regions, small (or zero) in sparse regions.

**The outlier problem:** If $\mathcal{N}_r(\mathbf{x}_q) = \emptyset$ (no training points
within radius $r$), what do you predict? sklearn raises a `ValueError` by default. Always
set `outlier_label='most_frequent'` or a specific fallback class:

```python
from sklearn.neighbors import RadiusNeighborsClassifier

clf = RadiusNeighborsClassifier(
    radius=1.0,             # in units of the SCALED feature space
    weights='distance',
    outlier_label='most_frequent'   # critical for production robustness
)
```

**Critical:** After `StandardScaler`, features have unit variance and roughly zero mean. A
radius of $r = 1.0$ corresponds to roughly $\pm 1$ standard deviation per feature — a
reasonable starting point. Without scaling, `radius=1.0` is meaningless.

**When to prefer RadiusNeighbors:**
- Data density is highly non-uniform (clusters plus sparse regions)
- You need the neighborhood to shrink/grow based on local density
- Anomaly detection: query points with no neighbors within radius are outliers

**Paper origin:** Natural generalization with no single canonical paper. Extensively discussed
in Devroye, Györfi & Lugosi, *A Probabilistic Theory of Pattern Recognition* (1996).

---

### 3.3 Condensed Nearest Neighbor — CNN (Hart, 1968)

> 📜 **Citation:** Hart, P. E. (1968). The condensed nearest neighbor rule.
> *IEEE Transactions on Information Theory*, 14(3), 515–516.
> DOI: https://doi.org/10.1109/TIT.1968.1054155

**Problem solved:** kNN stores the entire training set ($O(np)$ memory) and has $O(np)$
inference cost per query. For $n = 10^6$ samples with $p = 100$ features in float64,
that's 800 MB just for the data, plus $O(np)$ compute per query. Scaling to production
requires a smaller "prototype" set.

**Algorithm:**

1. Start with subset $S$ containing one random point from each class
2. Scan through the training set. For each point $(\mathbf{x}_i, y_i)$:
   - If 1-NN classification on $S$ *misclassifies* $(\mathbf{x}_i, y_i)$, add $\mathbf{x}_i$ to $S$
3. Repeat step 2 until no new points are added
4. Use $S$ (not the full training set) as the kNN lookup table

**Why it works:** Points deep in the interior of a class (surrounded by same-class neighbors)
are redundant — removing them doesn't change the decision boundary. Only points near the
class boundary are needed to define the boundary. CNN finds a minimal subset that still
correctly classifies all training points under 1-NN.

**Result:** $|S| \ll n$ in practice. CNN can reduce storage by 90%+ on clean datasets.

**Limitations:** Non-deterministic (depends on initialization and scan order); may retain
outliers/mislabeled points; the iterative procedure converges but is not guaranteed to find
the *minimum* consistent subset (that's NP-hard).

**sklearn:** Not directly implemented. Available via `imbalanced-learn`:
```python
# pip install imbalanced-learn
# Note: imbalanced-learn's CNN uses a slightly different parameterization
from imblearn.under_sampling import CondensedNearestNeighbour
```

---

### 3.4 Edited Nearest Neighbor — ENN (Wilson, 1972)

> 📜 **Citation:** Wilson, D. L. (1972). Asymptotic properties of nearest neighbor rules
> using edited data. *IEEE Transactions on Systems, Man, and Cybernetics*, 2(3), 408–421.
> DOI: https://doi.org/10.1109/TSMC.1972.4309137

**Problem solved:** kNN is fragile to label noise. A single mislabeled training point
$(\mathbf{x}_i, \tilde{y}_i)$ — where $\tilde{y}_i \neq y_i^{\text{true}}$ — sits in
the wrong region of feature space and creates a "decision island": all query points
in its neighborhood get misclassified. For noisy real-world data, this is common.

**Algorithm:**

1. For each training point $(\mathbf{x}_i, y_i)$:
   - Find its $k$ nearest neighbors *among the other training points*
   - If $(\mathbf{x}_i, y_i)$ is misclassified by majority vote of its $k$ nearest neighbors, remove it from the training set
2. Use the edited (cleaned) training set for kNN prediction

**Effect:** Points that are inconsistent with their neighborhood — typically outliers and
mislabeled examples — are removed. The decision boundary becomes smoother.

**sklearn/imbalanced-learn:**
```python
from imblearn.under_sampling import EditedNearestNeighbours

enn = EditedNearestNeighbours(n_neighbors=3, kind_sel='all')
X_resampled, y_resampled = enn.fit_resample(X_train, y_train)
# Then fit kNN on (X_resampled, y_resampled)
```

---

### 3.5 Locally Weighted Regression / LOWESS (Cleveland, 1979)

> 📜 **Citation:** Cleveland, W. S. (1979). Robust locally weighted regression and
> smoothing scatterplots. *Journal of the American Statistical Association*, 74(368),
> 829–836. URL: https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf

**Problem solved:** A plain kNN regressor computes a local *constant* (mean) within each
neighborhood. Near boundaries or in regions with a local trend, this produces a
step-function-like approximation with high bias. For smooth functions, you can do much
better by fitting a *local polynomial* instead.

**Algorithm:** At each query point $\mathbf{x}_q$:

1. Find the $f \cdot n$ nearest training points ($f$ = bandwidth fraction, typically 0.2–0.8)
2. Compute tricube kernel weights $w_i = \left(1 - \left|\frac{d_i}{d_{\max}}\right|^3\right)^3$
3. Fit a locally weighted linear (or quadratic) regression using these weights
4. Predict $\hat{y}$ from the local polynomial at $\mathbf{x}_q$

The tricube kernel assigns full weight to the nearest point and smoothly tapers to zero
at the boundary of the neighborhood.

**Key distinction from kNN regressor:** Fits a local polynomial, not just a mean. Much
lower bias near boundaries; captures local gradients. Higher computational cost.

**Python:**
```python
from statsmodels.nonparametric.smoothers_lowess import lowess

# 1D example: smooth noisy signal
smoothed = lowess(y_noisy, x, frac=0.3, it=3)   # frac=bandwidth, it=robustness iterations
x_smooth, y_smooth = smoothed[:, 0], smoothed[:, 1]
```

---

### 3.6 Neighborhood Components Analysis — NCA (Goldberger et al., 2004)

> 📜 **Citation:** Goldberger, J., Roweis, S., Hinton, G., and Salakhutdinov, R. (2004).
> Neighbourhood Components Analysis. *NeurIPS*, 17.
> URL: https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf

**Problem solved:** All kNN variants using Euclidean distance treat every feature as
equally important. An irrelevant feature contributes just as much to distance as the
most predictive feature. Manual feature selection is impractical for high-dimensional
data. What if we could *learn* the optimal distance metric from data?

**The algorithm:** Learn a linear transformation $A \in \mathbb{R}^{d' \times p}$ such
that 1-NN classification in the transformed space $A\mathbf{x}$ maximizes the expected
leave-one-out accuracy. Because hard 1-NN assignment is non-differentiable, NCA uses a
*stochastic* (softmax) relaxation. The probability that $i$ selects $j$ as its "neighbor":

$$p_{ij} = \frac{\exp(-\|A\mathbf{x}_i - A\mathbf{x}_j\|^2)}{\sum_{k \neq i} \exp(-\|A\mathbf{x}_i - A\mathbf{x}_k\|^2)}$$

The expected number of correctly classified points (to maximize):

$$f(A) = \sum_i p_i = \sum_i \sum_{j : y_j = y_i} p_{ij}$$

Optimized via gradient ascent (L-BFGS-B). The gradient with respect to $A$:

$$\frac{\partial f}{\partial A} = 2A \sum_i \left[ p_i \sum_k p_{ik} \mathbf{x}_{ik}\mathbf{x}_{ik}^T - \sum_{j: y_j = y_i} p_{ij} \mathbf{x}_{ij}\mathbf{x}_{ij}^T \right]$$

where $\mathbf{x}_{ij} = \mathbf{x}_i - \mathbf{x}_j$.

**When $d' < p$:** NCA learns a low-rank projection that simultaneously selects relevant
directions and learns the metric. With $d' = 2$, NCA gives a 2D embedding that is
optimized for kNN discrimination — often better than PCA for visualization of class
structure.

**sklearn:**
```python
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline

nca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('nca', NeighborhoodComponentsAnalysis(n_components=10, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
nca_pipe.fit(X_train, y_train)
```

**When to prefer:** When you have irrelevant/redundant features, cannot do manual feature
selection, and can afford the $O(n^2 p)$ per-iteration training cost. For $n > 5{,}000$,
NCA becomes slow — consider using PCA + kNN as a faster alternative.

> ⚠️ **Pitfall:** NCA can overfit on small datasets because it optimizes a leave-one-out
> training objective directly. Always evaluate on a held-out test set; the LOO training
> accuracy can be misleadingly high.

---

### 3.7 Approximate Nearest Neighbor — HNSW (Malkov & Yashunin, 2016/2018)

> 📜 **Citation:** Malkov, Y. A., and Yashunin, D. A. (2018). Efficient and robust
> approximate nearest neighbor search using Hierarchical Navigable Small World graphs.
> *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824–836.
> arXiv: https://arxiv.org/abs/1603.09320

**Problem solved:** Exact kNN at scale is $O(np)$ per query — for $n = 10^7$ samples
with $p = 768$ (BERT embeddings), each query requires $7.7 \times 10^9$ floating-point
operations. At sub-millisecond latency requirements, this is impossible.

**The algorithm:** HNSW builds a **hierarchical navigable small-world graph** — a
multi-layer proximity graph where:

- **Lower layers** (layer 0) contain all $n$ elements with short-range links (fine-grained)
- **Upper layers** (layers $1, 2, \ldots$) contain exponentially fewer elements with
  longer-range links (coarse navigation)

Element layer assignment is random: an element is inserted up to layer $l$ with probability:

$$P(\text{max layer} \geq l) = e^{-l / m_L}, \quad m_L = 1/\ln(M)$$

where $M$ is the number of outgoing connections per node. This exponential decay mimics
a skip list — upper layers provide highway-speed coarse navigation; lower layers provide
local fine-grained search.

**Search:** Start at the top layer, greedily navigate toward the query (follow the link
that decreases distance most), descend to the next layer when stuck in a local minimum,
repeat until layer 0. The candidate list size `ef` controls the exploration width —
larger `ef` gives higher recall at the cost of speed.

**Complexity:**
- Build: $O(n \log n)$ expected, $O(M \cdot n)$ memory
- Search: $O(\log n)$ expected (polylogarithmic)
- Recall@10: typically 95–99% at production settings

**Python (hnswlib):**
```python
import hnswlib
import numpy as np

dim = 128
n = 100_000

# Build index
index = hnswlib.Index(space='l2', dim=dim)       # space: 'l2', 'cosine', 'ip'
index.init_index(max_elements=n, ef_construction=200, M=16)
index.add_items(data, ids)                        # data: (n, dim) float32

# Tune query-time recall
index.set_ef(50)        # larger ef → better recall, slower search

# Search
labels, distances = index.knn_query(query, k=10)
```

> ⚠️ **Pitfall:** hnswlib (nmslib/hnswlib) has had reduced maintenance activity as of
> mid-2026. For new production deployments, consider **FAISS `IndexHNSWFlat`** (same
> algorithm, Meta-maintained) or Spotify's **Voyager** library
> (https://github.com/spotify/voyager), which is hnswlib-based but actively maintained.

---

### 3.8 FAISS — Billion-Scale Approximate Search (Johnson et al., 2021)

> 📜 **Citation:** Johnson, J., Douze, M., and Jégou, H. (2021). Billion-scale similarity
> search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
> arXiv: https://arxiv.org/abs/1702.08734. GitHub: https://github.com/facebookresearch/faiss

**Problem solved:** Even HNSW struggles with billion-scale datasets due to memory — $M \cdot n$
edges with 4 bytes each for $n = 10^9$ and $M = 16$ = 64 GB just for graph edges. FAISS
provides a toolkit of complementary index types for different scale/accuracy/memory tradeoffs.

**Key index types:**

| Index | Method | Best for | Tradeoff |
|---|---|---|---|
| `IndexFlatL2` | Exact brute-force L2 | Ground truth, $n < 10^5$ | Exact but $O(np)$ |
| `IndexFlatIP` | Exact inner product | Cosine with pre-normalized vectors | Exact but $O(np)$ |
| `IndexIVFFlat` | Inverted file (k-means partitioning) | $n = 10^5$–$10^7$, moderate accuracy | Trade `nprobe` vs accuracy |
| `IndexIVFPQ` | IVF + Product Quantization | Memory-constrained, $n > 10^7$ | Strong compression, some accuracy loss |
| `IndexHNSWFlat` | HNSW graph | High-recall static datasets | High memory, best recall |
| `IndexPQ` | Pure Product Quantization | Extreme memory constraints | Compressed codes |

**The IVFFlat workflow (production recipe):**

```python
import faiss
import numpy as np

d = 128           # embedding dimension
n = 1_000_000    # dataset size
nlist = 1000      # Voronoi cells — rule of thumb: sqrt(n)

# Build quantizer and IVF index
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# Train on representative sample (k-means partitioning)
data = np.random.rand(n, d).astype('float32')
index.train(data)    # must call train() for IVF indexes
index.add(data)

# Query: nprobe controls accuracy vs speed tradeoff
index.nprobe = 10    # search 10/1000 cells — ~1% of data
k = 5
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k)

# Higher nprobe = better recall: tune until recall@k meets requirements
index.nprobe = 100   # search 10% of cells — much slower, near-exact
```

> 🏆 **Best practice:** Always benchmark your FAISS configuration using the
> **ANN Benchmarks** framework (https://github.com/erikbern/ann-benchmarks). The benchmark
> measures recall@10 vs queries-per-second on standard datasets (SIFT, GIST, GloVe,
> NYTimes, MNIST). Compare your `nprobe`/`M`/`efSearch` settings against the Pareto
> frontier before deciding on an index type.

---

### 3.9 PyNNDescent — NN Descent (McInnes, 2018)

**Algorithm:** Nearest neighbor descent. The key insight: "a neighbor of a neighbor is
likely a neighbor." Initialize an approximate k-NN graph randomly, then iteratively
improve it — for each point, check if its neighbors' neighbors are closer than its current
farthest neighbor. This converges rapidly to a high-quality approximate k-NN graph.

**Strength:** No C++ compilation required (pure Python/Numba). Supports a huge variety
of distance metrics via Numba-compiled functions. Used internally by UMAP
(`pip install umap-learn`) for graph construction — if you use UMAP, you're already
using PyNNDescent.

```python
import pynndescent

index = pynndescent.NNDescent(
    data,
    n_neighbors=15,
    metric='euclidean',
    random_state=42,
    n_jobs=-1
)
indices, distances = index.query(query_data, k=10)
```

> 📊 **Benchmark (ANN Benchmarks, SIFT-128 dataset, 1M vectors):**
> - HNSW (hnswlib, M=16): 95% recall@10 at ~12,000 QPS
> - FAISS IVFFlat (nprobe=32): 96% recall@10 at ~8,000 QPS
> - PyNNDescent: 95% recall@10 at ~3,500 QPS
> - sklearn brute: 100% recall@10 at ~30 QPS
>
> These are approximate figures; actual performance depends on hardware, index parameters,
> and dataset characteristics. Verify with ANN Benchmarks for your specific workload.

---

## §4 — Hyperparameters: The Complete Guide

### `n_neighbors` — The Most Important Parameter

**What it controls:** The number of neighbors whose labels aggregate into a prediction.
$k$ directly controls the bias-variance tradeoff — in the *opposite* direction from most
parameters you've seen:

- **Small $k$ (e.g., $k=1$):** Low bias (can fit any decision boundary, including highly
  irregular ones), high variance (a single noisy training point can flip predictions in
  its region). 1-NN always achieves 0% training error — perfect memorization.

- **Large $k$ (e.g., $k=n$):** High bias (degenerates to the majority-class predictor
  for classification, global mean for regression), low variance (stable, unaffected by
  individual points).

**The default:** `n_neighbors=5` — a reasonable starting point for moderate-sized datasets,
but not magical. Always tune it.

**Tuning strategy:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

k_range = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
train_scores, val_scores = [], []

for k in k_range:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
    ])
    # Cross-validated score
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    val_scores.append(cv_scores.mean())

plt.figure(figsize=(10, 5))
plt.semilogx(k_range, val_scores, 'b-o', label='5-fold CV accuracy')
plt.xlabel('k (n_neighbors)')
plt.ylabel('Accuracy')
plt.title('kNN: Validation accuracy vs k')
plt.legend()
plt.grid(True)
plt.show()

best_k = k_range[np.argmax(val_scores)]
print(f"Best k: {best_k}")
```

**Rule of thumb:** Start with $k \approx \sqrt{n_{\text{train}}}$ as an upper bound;
the optimal $k$ is typically much smaller. For binary classification, always use **odd**
$k$ to prevent ties.

**Interactions:**
- Larger datasets support larger $k$ before the variance-reduction benefit plateaus
- When using `weights='distance'`, you can afford larger $k$ (distant neighbors are
  down-weighted automatically)
- In high dimensions, optimal $k$ tends to be larger because individual neighbors are
  less reliable

**Too low ($k=1$):** Overfits. Every training point is its own neighborhood — jagged,
non-smooth decision boundary. High sensitivity to outliers and noise.

**Too high ($k \approx n$):** Underfits. Every query gets the same prediction (the global
majority class / mean). The "local" structure that makes kNN powerful is destroyed.

---

### `weights` — Uniform vs Distance-Weighted

**What it controls:** How each neighbor's vote is weighted in the prediction.

**Options:**
- `'uniform'` (default): All $k$ neighbors have equal weight $w_i = 1/k$
- `'distance'`: $w_i \propto 1/d(\mathbf{x}_q, \mathbf{x}_i)$ — inverse distance weighting
- callable: Any function `f(distances) -> weights` — enable Gaussian, tricube, or custom kernels

**Mechanism:** Distance weighting implements a continuous transition between $k=1$ (if
all weight concentrates on the nearest neighbor) and $k$ (if all neighbors are equidistant).
It makes kNN more robust to the choice of $k$ — you can use a larger $k$ for stability
without paying the full bias cost of uniform averaging.

**Recommendation:** Use `weights='distance'` as your default. There is almost no scenario
where uniform weighting beats distance weighting on held-out data. The penalty for the
occasional tied case (identical training points) is handled gracefully by sklearn.

**Zero-distance edge case:** If `d(x_q, x_i) = 0` for some training point $x_i$:
sklearn sets $w_i = 1$ for the zero-distance point(s) and $w_i = 0$ for all others.
The prediction is the (aggregated) label(s) of the exact matches — the correct behavior.

---

### `algorithm` — Search Strategy

**What it controls:** The data structure used to find nearest neighbors.

**Options:**

| Value | Data structure | Best for | Limitation |
|---|---|---|---|
| `'auto'` | sklearn chooses | Default; fine for most cases | May not match your expectations |
| `'kd_tree'` | K-D tree | $p \leq 10$, Euclidean-family metrics | Degrades badly for $p > 15$ |
| `'ball_tree'` | Ball tree | $10 < p \leq 20$, non-Euclidean metrics, Mahalanobis | More expensive build than kd-tree |
| `'brute'` | Brute force | $p > 20$, sparse data, precomputed distances, small $n$ | $O(np)$ per query — scales poorly |

**The `auto` selection logic (sklearn 1.5.x):** Defaults to `'brute'` when:
1. Input $X$ is sparse
2. `metric='precomputed'`
3. $p > 15$ (number of features exceeds 15)
4. `n_neighbors >= n_samples / 2`
5. The metric is not supported by kd-tree or ball-tree

Otherwise tries kd-tree, then ball-tree.

> 🔧 **In practice:** For $p \leq 10$: `'kd_tree'`. For $10 < p \leq 20$ with
> non-Euclidean metrics: `'ball_tree'`. For $p > 20$ or any dataset where you're unsure:
> let `'auto'` choose brute, or use FAISS/hnswlib if $n > 10^4$.

---

### `leaf_size` — Tree Leaf Threshold

**What it controls:** When tree-based search switches to brute-force within a leaf node.
A leaf node with $\leq$ `leaf_size` points is searched exhaustively.

**Default:** `30` — a good general-purpose value.

**Effect:**
- Smaller `leaf_size`: Deeper tree, faster query (more pruning), larger tree in memory,
  slower build
- Larger `leaf_size`: Shallower tree, faster build, more memory-efficient, slightly slower
  queries within leaf nodes

**Tuning:** Do not spend time tuning `leaf_size`. The performance difference between
`leaf_size=10` and `leaf_size=50` is rarely more than a few percent on query time.
Focus on `n_neighbors` and `weights`.

---

### `p` — Minkowski Power Parameter

**What it controls:** The order of the Minkowski metric ($p=1$ → Manhattan, $p=2$ → Euclidean).

**Default:** `2` (Euclidean). Only relevant when `metric='minkowski'` (the default).

**When to use $p=1$ (Manhattan):**
- High-dimensional data with sparse features (image pixels, bag-of-words)
- When individual feature outliers should have bounded effect (L1 is robust to
  single-dimension outliers in a way L2 is not)
- NLP or text features with many zero entries

**When to use $p=2$ (Euclidean):**
- Continuous, well-scaled features where all dimensions contribute smoothly
- Physical/spatial data (actual geometric coordinates)

**Warning:** For $p < 1$, the Minkowski "metric" violates the triangle inequality and
is not a true metric. sklearn's BallTree won't prune correctly; use only with brute force.

---

### `metric` — Distance Function

See the complete metric table in §2. Key practical guidance:

- Use `'euclidean'` (or `'minkowski'` with `p=2`) for most tabular continuous data after scaling
- Use `'manhattan'` for sparse/high-dimensional data
- Use `'cosine'` with `algorithm='brute'` for text/embedding similarity, OR
  L2-normalize your vectors and use `'euclidean'`
- Use `'mahalanobis'` with `algorithm='ball_tree'` when features are correlated
- Use `'hamming'` for binary feature vectors
- Use `'haversine'` (via `ball_tree`) for geographic (lat/lon) data

---

### `n_jobs` — Parallelism

**What it controls:** Number of parallel jobs for neighbor search (only affects query-time
`kneighbors()` calls, NOT `fit()`).

**Default:** `None` (= 1 job, no parallelism).

**When to use:** `n_jobs=-1` (all CPUs) is worthwhile for large batch predictions
($m > 1{,}000$ queries) or large $n$ with brute-force. For small batch sizes, the
multiprocessing overhead dominates.

---

### RadiusNeighbors: `radius` Parameter

**Critical:** `radius` is measured in the *scaled* feature space. After `StandardScaler`,
a `radius=1.0` corresponds to approximately $\pm 1$ standard deviation per feature. This
is a reasonable starting point. Without scaling, you must know the raw feature scale to
set a meaningful radius.

**Tuning:** Use cross-validation with a range of radii. The key is to find the radius
where most queries return 3–20 neighbors; fewer than 3 gives high variance, more than 20
starts to average in irrelevant distant points.

---

### Hyperparameter Tuning Playbook

| Hyperparameter | Typical range | Primary effect | Recommended tuning method |
|---|---|---|---|
| `n_neighbors` | 1–101 (odd for binary) | Bias-variance tradeoff | Grid search or Optuna over log-spaced values |
| `weights` | `'uniform'`, `'distance'`, Gaussian callable | Smoothness of boundary | Always try `'distance'`; it's almost always better |
| `algorithm` | `'auto'` → see rules | Build/query speed | Let `auto` decide; check with timing if $p > 15$ |
| `leaf_size` | 10–100 | Minor query speed effect | Leave at 30; only tune if profiling shows bottleneck |
| `p` (Minkowski) | 1 (L1), 2 (L2) | Distance geometry | Try both for new datasets; cross-validate |
| `metric` | See table | Problem-specific | Domain knowledge + cross-validate 2–3 candidates |
| `radius` | Depends on scaled space | Neighborhood size | Cross-validate; ensure mean neighbor count is 5–20 |

---

## §5 — Strengths

### 5.1 Zero Training Time — True Lazy Learning

kNN has $O(1)$ "training" complexity — it simply stores the data. There is no optimization
loop, no convergence monitoring, no gradient computation. This makes kNN ideal for:

- **Online/streaming scenarios** where new training points arrive continuously and you cannot
  afford to retrain from scratch. With kNN, you just add the new point to the stored set.
- **Rapid prototyping** — a kNN baseline can be running in minutes, before any feature
  engineering, giving you a non-trivial floor for comparison.
- **Incremental learning** — add `NearestNeighbors` points dynamically with FAISS's `add()`
  or hnswlib's `add_items()`.

### 5.2 Non-Parametric — No Distributional Assumptions

kNN makes no assumption about the functional form of the decision boundary. It can represent:

- Linear boundaries (trivially)
- Non-linear, curved, and multi-modal boundaries
- Disjoint class regions (class A at both ends of the feature space)
- Boundaries that depend on the local density of each class

This stands in sharp contrast to logistic regression (linear boundary in feature space),
naive Bayes (assumes feature independence), or LDA (Gaussian class-conditional densities
with equal covariance). If your data has complex geometry, kNN can model it — assuming you
have enough data.

### 5.3 Naturally Handles Multi-Class Problems

For $C$-class classification, kNN requires no modification — each neighbor votes for their
class, and the plurality wins. There is no one-vs-rest or one-vs-one decomposition required
(unlike some SVM formulations). Class probabilities extend naturally as $\hat{P}(y=c) = k_c/k$.

### 5.4 Consistent Theoretical Guarantees

The Cover-Hart bound ($R \leq 2R^*$, no distributional assumptions required) is one of the
strongest theoretical guarantees in all of supervised learning. The asymptotic consistency
result (kNN error $\to R^*$ as $n \to \infty$, $k \to \infty$, $k/n \to 0$) means that
given enough data, kNN is optimal. For parametric models, "given enough data" means the
parametric assumption is correct; for kNN, it simply means enough data exists — a much
weaker requirement.

### 5.5 Highly Interpretable Predictions

A kNN prediction is directly interpretable: "I predicted class A because the 5 nearest
training examples to this query were all class A: namely, these specific instances." You
can show the user the actual training examples that drove the prediction — a form of
**case-based reasoning** that is immediately understandable to non-technical stakeholders.

This is especially powerful in medical diagnosis (showing similar past patients), fraud
detection (showing similar past fraud cases), and recommendation systems (showing similar
past user behavior).

### 5.6 Automatically Adapts to Local Data Density

The "neighborhood" of a kNN prediction physically adapts to local data density. In dense
regions, the $k$ nearest neighbors span a small distance — predictions are precise. In
sparse regions, they span a larger distance — predictions are appropriately uncertain
(captured by low max probability in `predict_proba`). No global bandwidth parameter needs
tuning for local adaptation, unlike kernel density estimators with fixed bandwidth.

### 5.7 Effective for Low-Dimensional Structured Data

For low-dimensional problems ($p \leq 10$) where the geometry of the feature space is
informative (sensor data, small tabular datasets, geographic data, physical simulations),
kNN often achieves competitive accuracy with far less tuning than parametric models. There
are no distribution assumptions to violate.

> 📊 **Benchmark:** On the UCI Iris dataset (150 samples, 4 features, 3 classes), a
> properly-scaled kNN with $k=5$ and distance weighting achieves ~97% accuracy in 10-fold
> CV — matching logistic regression and linear SVM, and requiring no hyperparameter
> iteration to reach a competitive result.

### 5.8 The Backbone of Modern Vector Search and Recommendation Systems

The most commercially impactful application of kNN in 2026 is not sklearn-based tabular
classification — it is **embedding-space nearest neighbor search** underlying every major
recommendation system, semantic search engine, and RAG (retrieval-augmented generation)
pipeline.

The workflow: (1) encode items (documents, images, products) into high-dimensional
embedding vectors using a learned encoder (e.g., sentence-transformers for text,
CLIP for images); (2) build a kNN index (hnswlib, FAISS) over the embedding vectors;
(3) at query time, encode the query and retrieve the $k$ nearest embeddings. This is
kNN — with a learned Euclidean/cosine metric and an approximate index.

Examples:
- **Spotify** uses approximate kNN over audio embeddings for "Because you liked" recommendations
- **Google Photos** uses kNN over face/visual embeddings for "Photos of you"
- **OpenAI** text-davinci and GPT APIs use kNN retrieval over document embeddings in RAG
- **Pinterest** uses kNN over image embeddings for visual search ("search by photo")

The fact that an algorithm from a 1951 Air Force technical report underpins all of this is
a testament to how fundamental the nearest-neighbor idea truly is.

```python
# Example: Semantic search with sentence-transformers + FAISS
# pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Corpus of documents
docs = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Nearest neighbor search is fundamental to recommendation systems",
    "k-Nearest Neighbors classifies by majority vote of k closest training examples",
    "FAISS enables billion-scale approximate nearest neighbor search on GPUs",
]

# Encode with sentence-transformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
doc_embeddings = model.encode(docs, normalize_embeddings=True)  # L2-normalize for cosine
doc_embeddings = doc_embeddings.astype('float32')

# Build FAISS index (inner product on normalized = cosine similarity)
d = doc_embeddings.shape[1]  # 384
index = faiss.IndexFlatIP(d)  # inner product (= cosine for normalized vectors)
index.add(doc_embeddings)

# Semantic search
query = "How does approximate nearest neighbor work at scale?"
query_emb = model.encode([query], normalize_embeddings=True).astype('float32')
scores, indices = index.search(query_emb, k=3)

print("Top 3 semantically similar documents:")
for score, idx in zip(scores[0], indices[0]):
    print(f"  [{score:.3f}] {docs[idx]}")
```

This is kNN. Just kNN — with good features (learned embeddings) and a fast index.
The algorithm hasn't changed; only the feature representation and the search structure have.

---

## §6 — Weaknesses & Failure Modes

### 6.1 The Curse of Dimensionality — The Central Failure Mode

This is the most important weakness of kNN and deserves a deep treatment.

**The volume problem.** Consider a $p$-dimensional hypercube $[0, 1]^p$ with $n$ uniformly
distributed training points. To find the $k$ nearest neighbors of a query at the center,
you need to expand a hypersphere until it contains $k$ points. The fraction of the volume
occupied by this hypersphere is $k/n$. The radius of the hypersphere must be:

$$r = \left(\frac{k/n}{V_p(1)}\right)^{1/p} \approx \left(\frac{k}{n}\right)^{1/p}$$

where $V_p(1) = \frac{\pi^{p/2}}{\Gamma(p/2+1)}$ is the unit hypersphere volume. As $p$
grows, $r$ grows toward 1 (the diameter of the cube) even for fixed $k/n$:

| Dimension $p$ | Required radius for $k/n = 1\%$ |
|---|---|
| 2 | 0.056 |
| 5 | 0.398 |
| 10 | 0.631 |
| 20 | 0.794 |
| 50 | 0.933 |
| 100 | 0.966 |

At $p = 50$, your "local" neighborhood occupies 93% of the feature space's diameter. There
is nothing local about it.

**The equidistance collapse.** Let $d_{\min}$ and $d_{\max}$ be the minimum and maximum
distances from a fixed query to $n$ uniformly distributed points. Beyer et al. (1999)
proved:

$$\frac{d_{\max} - d_{\min}}{d_{\min}} \xrightarrow{p \to \infty} 0$$

All points become roughly equidistant from any query. The concept of "nearest" neighbor
loses meaning — the 1-NN is barely closer than the $n$-NN.

**The volume-in-corners problem.** The ratio of the inscribed hypersphere to its enclosing
hypercube collapses exponentially:

| Dimension $p$ | Sphere/Cube volume ratio |
|---|---|
| 2 | 78.5% |
| 3 | 52.4% |
| 5 | 16.5% |
| 10 | 0.25% |
| 20 | $2.5 \times 10^{-7}$% |

At $p = 10$, 99.75% of the hypercube's volume sits in the "corners" — outside the inscribed
sphere. Training points concentrate there. A query at the center has almost no training
neighbors nearby in a Euclidean ball.

**Detection:** Plot the distribution of distances to the $k$-th nearest neighbor. If the
distribution is narrow (small variance relative to the mean), equidistance is setting in.
Also plot validation accuracy vs $p$ as you add features — if adding features hurts
accuracy, you've crossed into the curse.

**Mitigation strategies:**

1. **Feature scaling** (non-negotiable regardless): Without this, one feature dominates
2. **PCA before kNN**: Reduce to $d \leq 10$ components. Accept that you're discarding
   directions of variation, but at least the remaining dimensions are interpretable.
3. **NCA (§3.6)**: Learn a low-rank projection that maximizes kNN accuracy
4. **Manual feature selection**: Remove features with near-zero mutual information with labels
5. **UMAP embedding**: For visualization/exploration, UMAP can find a 2D manifold structure
   that preserves local topology better than PCA
6. **Use a different algorithm**: For $p > 100$, consider Random Forest, SVM with RBF
   kernel, or gradient boosting — these do not suffer from the curse as severely

### 6.2 Slow Inference at Scale — $O(np)$ per Query

Even with kd-tree / ball-tree, exact kNN inference degrades to $O(np)$ when $p$ is large.
For $n = 10^6$, $p = 100$: each prediction requires $10^8$ operations. At $10^9$ FLOPS,
that's 0.1 seconds per prediction — 10 predictions per second. Unusable for real-time
applications.

**Detection:** Time `pipe.predict()` on a small batch and multiply to production scale.

**Mitigation:**
- Approximate NN: FAISS, hnswlib (1000x speedup, ~1–5% recall loss)
- Dimensionality reduction before kNN
- Condensed NN (§3.3) to reduce training set size

### 6.3 High Memory Footprint

kNN stores the entire training set. $n = 10^7$ points with $p = 128$ features in float32
= $10^7 \times 128 \times 4 = 5.12$ GB of data alone, before the kd-tree or ball-tree
overhead. This makes kNN infeasible for very large datasets on memory-constrained hardware.

**Mitigation:** Condensed NN for prototype reduction; FAISS with Product Quantization
(IVF-PQ) compresses vectors to $\sim$4–16 bytes regardless of original dimension.

### 6.4 Sensitivity to Irrelevant Features

Every feature contributes to distance equally (unless you use Mahalanobis or NCA). Adding
10 irrelevant features dilutes the signal from 5 relevant features — the neighbors found
are determined by noise, not signal.

**Detection:** Run permutation importance (§8) and check if high-importance features are
actually driving distance. Compare kNN performance with and without suspected irrelevant
features. If removing features *improves* cross-validated accuracy, you have the problem.

**Mitigation:** Feature selection before kNN; NCA metric learning; PCA.

### 6.5 Requires Feature Scaling — Without It, Fails Silently

Unscaled features cause kNN to effectively ignore low-variance features entirely. This is
perhaps the most common practitioner mistake — the model runs without error but gives
nonsensical results.

```python
# BAD — missing scaling
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)   # salary (0-200k) dominates age (18-70)

# GOOD — always in a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
pipe.fit(X_train, y_train)  # correct; no data leakage in CV
```

> ⚠️ **Pitfall:** Scaling OUTSIDE a Pipeline before cross-validation is a **data leakage
> bug**. The test fold's statistics contaminate the scaler. Always wrap scaler + kNN in a
> `Pipeline` and pass the pipeline to `cross_val_score` or `GridSearchCV`.

### 6.6 Class Imbalance

kNN has no mechanism analogous to `class_weight='balanced'` in sklearn's logistic regression.
With 95% majority class in training data, almost all $k$ neighbors of any query point will
be the majority class — even for query points that should be minority class. The model
learns to predict majority class everywhere.

**Detection:** Check per-class precision/recall. Majority-class recall >> minority-class
recall signals the problem.

**Mitigation:**
- Resample (SMOTE or undersampling) **inside the cross-validation fold** — not on the
  whole training set, which would be leakage
- Use `weights='distance'` — down-weights far neighbors, gives local density more influence
- Use EditedNearestNeighbours (§3.4) to clean up boundary confusion

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),    # runs inside each CV fold
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
])
```

### 6.7 No Probabilistic Calibration

kNN's `predict_proba` outputs $k_c/k$ — the fraction of neighbors in class $c$. These
values cluster at a few discrete levels (0/5, 1/5, 2/5, 3/5, 4/5, 5/5 for $k=5$) and
are typically not well-calibrated as probabilities. For small $k$, the probability
estimates are extremely coarse.

**Detection:** Calibration curve (reliability diagram). Well-calibrated: predicted $P(y=1)$
matches observed frequency. Typical kNN: staircase pattern at discrete values.

**Mitigation:** Use `sklearn.calibration.CalibratedClassifierCV` with `method='isotonic'`
(or `'sigmoid'`) to recalibrate the probability outputs. Use larger $k$ for smoother
probabilities (more discrete levels → smoother approximation of the true posterior).

### 6.8 Boundary Sensitivity to Noise

A single mislabeled training point in a compact region of feature space creates a
"decision island" — all query points in its neighborhood will be misclassified. Unlike
decision trees (which can generalize past individual noisy points with a purity threshold),
or linear models (which average out noise globally), kNN has no smoothing mechanism
unless you use a large $k$ or EditedNearestNeighbours.

**Detection:** Compare training accuracy (will be ~100% for $k=1$) vs validation accuracy.
A large gap signals overfitting to noise. Plot the decision boundary — visual discontinuities
are noise-driven islands.

**Mitigation:** Use ENN (§3.4) to remove noisy points; use larger $k$; use `weights='distance'`.

### 6.9 Model Privacy — Training Data is the Model

kNN stores the entire training set as its "model." Anyone with access to the serialized model
has access to every training example — with exact feature values and labels. This is a serious
concern in privacy-sensitive domains (medical records, financial data, personal behavior).

- A model inversion attack on kNN is trivial: query the model with all training points and
  you recover all labels; query a region exhaustively and you recover feature distributions.
- **GDPR "right to be forgotten" compliance** is difficult: removing a training example
  requires rebuilding the lookup structure (though the raw deletion is $O(n)$ for brute force).
- For production systems, consider whether the inference-time exposure of training data is
  acceptable. If not, use a parametric model that doesn't store the training set.

### 6.10 No Extrapolation

kNN can only predict for query points that fall within the convex hull of the training
data (approximately). Points outside the training distribution receive predictions from
the nearest training point — which may be far away and unrepresentative. The model has
no mechanism to recognize that it is extrapolating.

A Random Forest has the same problem (tree prediction = average of a region), but linear
models can at least extrapolate correctly for linear trends outside the training range.

**Detection:** Check if any test predictions come from neighbors with high distance.
A query's maximum neighbor distance relative to the training set's mean pairwise distance
is a reasonable "extrapolation flag."

```python
# Detect extrapolation: flag test points where k-th neighbor is far
pipe.fit(X_train, y_train)
knn = pipe.named_steps['knn']
scaler = pipe.named_steps['scaler']

X_test_sc = scaler.transform(X_test)
X_train_sc = scaler.transform(X_train)

# Distances to nearest training neighbors
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X_train_sc)
distances, _ = nbrs.kneighbors(X_test_sc)
kth_dist = distances[:, -1]   # distance to 5th neighbor

# Compare to typical intra-training distances
intra_dists, _ = nbrs.kneighbors(X_train_sc)
typical_dist = intra_dists[:, -1].mean()

extrapolation_flag = kth_dist > 3 * typical_dist
print(f"Potential extrapolation points: {extrapolation_flag.sum()} of {len(X_test)}")
```

---

## §7 — Diagnostic Plots & Evaluation

### 7.1 The k-vs-Accuracy Validation Curve

The single most important diagnostic for kNN: how does accuracy (or AUC/R²) change as
$k$ varies? This reveals where your model sits on the bias-variance curve.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y = iris.data, iris.target

k_range = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
mean_scores, std_scores = [], []

for k in k_range:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
    ])
    scores = cross_val_score(pipe, X, y, cv=10, scoring='accuracy')
    mean_scores.append(scores.mean())
    std_scores.append(scores.std())

mean_scores = np.array(mean_scores)
std_scores = np.array(std_scores)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_range, mean_scores, 'b-o', linewidth=2, markersize=6, label='Mean CV accuracy')
ax.fill_between(k_range, mean_scores - std_scores, mean_scores + std_scores,
                alpha=0.2, color='blue', label='±1 std')
ax.axvline(x=k_range[np.argmax(mean_scores)], color='red', linestyle='--',
           label=f'Best k={k_range[np.argmax(mean_scores)]}')
ax.set_xlabel('k (n_neighbors)', fontsize=12)
ax.set_ylabel('10-fold CV Accuracy', fontsize=12)
ax.set_title('Validation Curve: kNN — Bias-Variance Tradeoff', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**How to read it:**
- **Left side (small $k$):** Decreasing accuracy with decreasing $k$ → high variance regime
- **Right side (large $k$):** Decreasing accuracy with increasing $k$ → high bias regime
- **Optimal $k$:** The peak — minimum bias-variance tradeoff point
- **Wide confidence bands:** High variance in CV scores; need more data or larger $k$
- **Flat curve:** kNN is not sensitive to $k$ in this range — good sign of a stable solution

> 💡 **Intuition:** Think of the k-vs-accuracy curve as the mirror image of a regularization
> path. In ridge regression, larger $\lambda$ = more bias = flatter prediction. In kNN,
> larger $k$ = more bias = flatter (more averaged) prediction. Both curves have a sweet spot.

---

### 7.2 Decision Boundary Visualization (2D)

For low-dimensional data, the decision boundary is the most informative diagnostic.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
# Use only first 2 features for 2D visualization
X_2d = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2,
                                                      random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
k_values = [1, 5, 30]
colors_bg = ['#FFAAAA', '#AAFFAA', '#AAAAFF']
colors_pt = ['red', 'green', 'blue']

for ax, k in zip(axes, k_values):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train_sc, y_train)

    # Create mesh grid
    x_min, x_max = X_train_sc[:, 0].min() - 0.5, X_train_sc[:, 0].max() + 0.5
    y_min, y_max = X_train_sc[:, 1].min() - 0.5, X_train_sc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(colors_bg))
    for cls, col in zip([0, 1, 2], colors_pt):
        mask_tr = y_train == cls
        mask_te = y_test == cls
        ax.scatter(X_train_sc[mask_tr, 0], X_train_sc[mask_tr, 1],
                   c=col, marker='o', alpha=0.6, s=40, label=f'Train {iris.target_names[cls]}')
        ax.scatter(X_test_sc[mask_te, 0], X_test_sc[mask_te, 1],
                   c=col, marker='*', s=120, edgecolors='k')

    acc = knn.score(X_test_sc, y_test)
    ax.set_title(f'k={k}, Test acc={acc:.2f}', fontsize=12)
    ax.set_xlabel('Feature 1 (scaled)')
    ax.set_ylabel('Feature 2 (scaled)')

plt.suptitle('kNN Decision Boundary: Effect of k (Iris, 2 features)', fontsize=13)
plt.tight_layout()
plt.show()
```

**What good looks like:** Smooth, contiguous regions for each class. Boundary complexity
matches the true complexity of the data. No isolated decision "islands" in the interior of
a class region.

**What bad looks like (k=1):** Jagged, fractal-like boundaries with isolated islands
around every training point. Every test point in a foreign class region is misclassified.
High overfitting.

---

### 7.3 Confusion Matrix and Per-Class Metrics

```python
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay, roc_auc_score)
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance'))
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                         display_labels=iris.target_names,
                                         ax=ax, colorbar=True)
ax.set_title('kNN Confusion Matrix')
plt.tight_layout()
plt.show()

# Classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Multi-class AUC
auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
print(f"Weighted AUC-OvR: {auc_ovr:.4f}")
```

**How to read it:** Diagonal = correct predictions. Off-diagonal = confusion pairs.
For kNN with class imbalance, watch for rows where the minority class has low recall
(many off-diagonal entries in the minority row).

---

### 7.4 Calibration Curve (Reliability Diagram)

kNN probability estimates are poorly calibrated for small $k$ (staircase pattern). This
plot exposes the problem and guides whether recalibration is needed.

```python
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV

fig, ax = plt.subplots(figsize=(7, 6))

for k in [3, 5, 15]:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
    ])
    pipe.fit(X_train, y_train)
    CalibrationDisplay.from_estimator(
        pipe, X_test, y_test,
        n_bins=10, ax=ax, name=f'kNN k={k}',
        pos_label=1   # for binary classification
    )

ax.set_title('kNN Calibration Curve (Reliability Diagram)')
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.legend()
plt.tight_layout()
plt.show()

# Recalibrate with isotonic regression
pipe_cal = CalibratedClassifierCV(
    Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ]),
    method='isotonic',
    cv=5
)
pipe_cal.fit(X_train, y_train)
```

**What good looks like:** The calibration curve hugs the diagonal ($y = x$). Predicted
probability of 0.7 means the event occurs 70% of the time.

**What bad looks like (small k):** Staircase pattern — probabilities cluster at
$0/k, 1/k, 2/k, \ldots, 1$. The model is not well-calibrated, especially for $k=3$
where only 4 probability values are possible.

---

### 7.5 Distance Distribution Diagnostic (Curse of Dimensionality)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Simulate curse of dimensionality
np.random.seed(42)
n = 500
dims = [2, 5, 10, 20, 50, 100]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for ax, p in zip(axes, dims):
    X_high = np.random.rand(n, p)
    scaler = StandardScaler().fit(X_high[:400])
    X_sc = scaler.transform(X_high)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=6).fit(X_sc[:400])
    dists, _ = nbrs.kneighbors(X_sc[400:])
    kth_dists = dists[:, -1]  # distance to k-th neighbor

    ax.hist(kth_dists, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    cv = kth_dists.std() / kth_dists.mean()
    ax.set_title(f'p={p} dims | CV of dist = {cv:.3f}', fontsize=10)
    ax.set_xlabel('Distance to 6th neighbor')
    ax.set_ylabel('Count')

plt.suptitle('Curse of Dimensionality: Distance Distribution Collapses at High Dimensions',
             fontsize=12)
plt.tight_layout()
plt.show()
```

**How to read it:** The coefficient of variation (CV = std/mean) of distances to the
$k$-th neighbor should be close to zero in high dimensions — all points are equidistant.
When CV approaches 0, kNN is operating blindly. A healthy CV is > 0.1 or 0.2.

---

### 7.6 Learning Curves

```python
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance'))
])

train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_mean, 'b-o', label='Training score')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
ax.plot(train_sizes, val_mean, 'r-o', label='Cross-validation score')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
ax.set_xlabel('Training set size')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curve — kNN (k=7, distance-weighted)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**What good looks like:** Training and validation curves converge as data increases.
Validation score improves steadily with more data — typical of non-parametric methods
that benefit from more samples.

**What bad looks like:**
- Large, persistent gap between training and validation → overfitting (reduce $k$, use ENN)
- Both curves low and converged → underfitting (irrelevant features? wrong metric? consider
  a different algorithm)
- Validation curve still rising at $n = n_{\max}$ → you're data-hungry; get more data

---

### 7.7 kNN Regression Diagnostics

For regression tasks, residual plots and R² vs $k$ curves are key:

```python
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X_r, y_r = diabetes.data, diabetes.target
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

# R² vs k
k_range_r = [1, 3, 5, 7, 10, 15, 20, 30]
r2_scores  = []

for k in k_range_r:
    pipe_r = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=k, weights='distance'))
    ])
    scores = cross_val_score(pipe_r, X_tr, y_tr, cv=5, scoring='r2')
    r2_scores.append(scores.mean())

# Residual plot for best k
best_k_r = k_range_r[np.argmax(r2_scores)]
pipe_best = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=best_k_r, weights='distance'))
])
pipe_best.fit(X_tr, y_tr)
y_pred_r = pipe_best.predict(X_te)
residuals = y_te - y_pred_r

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: R² vs k
axes[0].plot(k_range_r, r2_scores, 'g-o', linewidth=2)
axes[0].set_xlabel('k (n_neighbors)')
axes[0].set_ylabel('5-fold CV R²')
axes[0].set_title(f'R² vs k — Best k={best_k_r}')
axes[0].grid(True, alpha=0.3)

# Right: Residuals vs fitted
axes[1].scatter(y_pred_r, residuals, alpha=0.6, color='steelblue', s=30)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Predicted values')
axes[1].set_ylabel('Residuals')
axes[1].set_title(f'Residuals vs Fitted (k={best_k_r})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Test R²: {r2_score(y_te, y_pred_r):.4f}")
print(f"Test RMSE: {mean_squared_error(y_te, y_pred_r)**0.5:.4f}")
```

**What good residuals look like:** Random scatter around zero with no pattern. Constant
variance (homoscedasticity).

**What bad residuals look like for kNN:**
- Funnel shape (increasing variance at higher predicted values) → consider log-transforming target
- Wavy pattern (systematic under- then over-prediction) → $k$ too small; increase $k$ to smooth
- Outlier residuals concentrated in one region → sparse region; kNN averaging over wide area

---

## §8 — Explainability & Interpretable AI

### 8.1 Built-In Interpretability: Case-Based Reasoning

kNN has a form of built-in interpretability that most algorithms lack entirely: you can
directly retrieve the $k$ training examples that determined any prediction. This is
**case-based reasoning** — explaining a new decision by pointing to similar past decisions.

```python
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

pipe = Pipeline([('scaler', StandardScaler()),
                 ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])
pipe.fit(X, y)

def explain_prediction(pipe, X_train, y_train, query_point, feature_names, target_names):
    """Explain a kNN prediction by showing the k nearest training neighbors."""
    knn = pipe.named_steps['knn']
    scaler = pipe.named_steps['scaler']

    query_sc = scaler.transform(query_point.reshape(1, -1))
    pred_class = knn.predict(query_sc)[0]
    proba = knn.predict_proba(query_sc)[0]

    # Get neighbors
    distances, indices = knn.kneighbors(query_sc)
    distances, indices = distances[0], indices[0]

    print(f"Query: {dict(zip(feature_names, query_point))}")
    print(f"Prediction: {target_names[pred_class]} (confidence: {proba.max():.1%})")
    print(f"\nThe 5 nearest training neighbors:")

    df_nbrs = pd.DataFrame(X_train[indices], columns=feature_names)
    df_nbrs['class'] = [target_names[y_train[i]] for i in indices]
    df_nbrs['distance'] = distances
    df_nbrs['weight'] = 1 / (distances + 1e-10)
    df_nbrs['weight'] /= df_nbrs['weight'].sum()
    print(df_nbrs.round(3).to_string(index=False))

# Explain a prediction
query = X[0]  # first iris sample
explain_prediction(pipe, X, y, query, feature_names, target_names)
```

This is one of kNN's most powerful real-world advantages. A clinician can understand why
a patient was assigned to a risk group by seeing the 5 most similar past patients. A fraud
analyst can verify a kNN flag by examining the 5 most similar historical fraud cases.

### 8.2 SHAP — KernelExplainer (Use with Caution)

kNN has no TreeExplainer (tree-based) and no LinearExplainer (linear model). The only
SHAP option is the **model-agnostic KernelExplainer** — which is theoretically correct
but computationally expensive.

> 🔬 **Deep dive:** `shap.KernelExplainer` works by sampling subsets of features, masking
> the others with background values, and measuring the effect on model output. This is a
> sampling approximation of the exact Shapley value integral. The computation is
> $O(2^p \cdot n_{\text{background}})$ in theory, but in practice SHAP samples $\sim$2000
> subsets with `nsamples='auto'`. For $p > 50$, this is still slow (~1 minute per
> prediction for a model with $n = 1{,}000$).

```python
import shap
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

pipe = Pipeline([('scaler', StandardScaler()),
                 ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])
pipe.fit(X, y)

# Use a SMALL background set — KernelExplainer is slow
background = X[:30]    # 30 background samples is usually enough

# Wrapper for predict_proba
def model_predict(X_array):
    return pipe.predict_proba(X_array)

explainer = shap.KernelExplainer(model_predict, background)

# Explain 10 test samples (NOT all — too slow)
X_explain = X[120:130]
shap_values = explainer.shap_values(X_explain, nsamples=100)   # nsamples limits evaluation calls

# shap_values is a list of length n_classes
# Plot for class 0
shap.summary_plot(shap_values[0], X_explain, feature_names=iris.feature_names,
                  plot_type='bar', show=True)

# Waterfall plot for a single prediction
shap.waterfall_plot(
    shap.Explanation(values=shap_values[0][0],
                     base_values=explainer.expected_value[0],
                     data=X_explain[0],
                     feature_names=iris.feature_names)
)
```

> ⚠️ **Pitfall:** `shap.KernelExplainer` with $p = 100$ features and `nsamples=2000` takes
> approximately 5–30 minutes per prediction on a typical laptop. It is not suitable for
> production monitoring or explaining large datasets. Use it for offline explanation of
> a small number of high-priority predictions only.

### 8.3 SHAP PermutationExplainer — A Faster Alternative

`shap.PermutationExplainer` is more efficient than KernelExplainer for kNN:

```python
import shap

# PermutationExplainer — faster than KernelExplainer for tabular data
masker = shap.maskers.Independent(background, max_samples=100)  # # verify in current docs
explainer_perm = shap.PermutationExplainer(model_predict, masker)
shap_values_perm = explainer_perm(X_explain)

shap.plots.beeswarm(shap_values_perm[:, :, 0])   # for class 0
```

### 8.4 Permutation Feature Importance — The Recommended Approach

For kNN, **permutation importance** is the most practical and interpretable global
feature importance method. It requires no SHAP overhead and answers the question directly:
"if I shuffle this feature across the test set, how much does accuracy drop?"

```python
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

pipe = Pipeline([('scaler', StandardScaler()),
                 ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance'))])
pipe.fit(X_train, y_train)

result = permutation_importance(
    pipe, X_test, y_test,
    n_repeats=30,            # 30 permutations per feature for stable estimates
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Sort by mean importance
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], xerr=result.importances_std[sorted_idx],
        color='steelblue', alpha=0.8, ecolor='gray', capsize=4)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(np.array(iris.feature_names)[sorted_idx])
ax.set_xlabel('Mean accuracy decrease when feature is permuted')
ax.set_title('Permutation Feature Importance — kNN')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

**How to read it:** Features with large mean importance (and narrow error bars) are critical
to the model's predictions. Features near zero are irrelevant. Features with importance < 0
mean permuting them *improves* performance — a sign that they are correlated with a
stronger feature and adding noise (consider removing them).

> 🏆 **Best practice:** Use permutation importance on a held-out test set (or a validation
> set that was not used in hyperparameter tuning). Importance computed on training data
> reflects memorization, not generalization.

### 8.5 Partial Dependence Plots (PDP) and ICE Curves

PDPs show the marginal effect of a feature on kNN predictions, averaged over the data
distribution. ICE (Individual Conditional Expectation) curves show the same effect for
individual samples.

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(12, 5))

PartialDependenceDisplay.from_estimator(
    pipe,
    X_train,
    features=[0, 1, 2, 3],        # all 4 features
    feature_names=iris.feature_names,
    kind='both',                    # 'average' = PDP, 'individual' = ICE, 'both' = both
    subsample=50,                   # 50 ICE curves (full set can be slow)
    ax=ax,
    grid_resolution=50,
    random_state=42
)
ax.set_title('Partial Dependence Plots — kNN (Iris)')
plt.tight_layout()
plt.show()
```

**How to read PDPs:**
- The line is the *average* prediction as that feature varies (other features held at their
  marginal distribution)
- Steep slope → feature has large effect on prediction
- Flat line → feature is uninformative for the model (even if it's in the feature set)
- ICE curves that diverge from each other → interaction effects between this feature and others

**kNN-specific pattern:** PDP curves for kNN are typically step-function-like (since kNN
makes locally constant predictions). Unlike smooth sigmoid curves from logistic regression
or smooth trees from RF, kNN PDPs often have jumps — these correspond to decision boundaries
where the neighborhood composition changes.

### 8.6 Direct Neighbor Inspection — The Most Honest Explainer

For any important prediction, always examine the actual nearest neighbors:

```python
def full_knn_explanation(pipe, X_train_orig, y_train, query, feature_names, target_names, k=5):
    """Show prediction + k nearest neighbors with distances and feature deltas."""
    import pandas as pd
    import numpy as np

    scaler = pipe.named_steps['scaler']
    knn    = pipe.named_steps['knn']

    query_sc = scaler.transform(query.reshape(1, -1))
    distances, indices = knn.kneighbors(query_sc, n_neighbors=k)
    distances, indices = distances[0], indices[0]

    pred = knn.predict(query_sc)[0]
    proba = knn.predict_proba(query_sc)[0]
    weights = 1 / (distances + 1e-10)
    weights /= weights.sum()

    neighbors_df = pd.DataFrame({
        feat: X_train_orig[indices, j]
        for j, feat in enumerate(feature_names)
    })
    neighbors_df['Label']    = [target_names[y_train[i]] for i in indices]
    neighbors_df['Distance'] = distances.round(4)
    neighbors_df['Weight']   = weights.round(4)

    print(f"\n{'='*60}")
    print(f"Query point: {dict(zip(feature_names, query.round(3)))}")
    print(f"Prediction: {target_names[pred]}")
    print(f"Class probabilities: {dict(zip(target_names, proba.round(3)))}")
    print(f"\nNearest {k} training neighbors:")
    print(neighbors_df.to_string(index=False))
    print('='*60)
```

This is the deepest and most faithful explanation kNN can provide — no approximations,
no surrogate models. The training neighbors *are* the explanation.

---

## §9 — Complete Python Worked Example

This section walks through a full end-to-end kNN analysis: classification on the **Iris**
dataset and regression on the **Diabetes** dataset. Both are from sklearn and represent
genuinely different regimes: Iris is low-dimensional with well-separated classes; Diabetes
is higher-dimensional with continuous response.

### 9.1 Classification: Iris Dataset

```python
# ============================================================
# kNN COMPLETE WORKED EXAMPLE — IRIS CLASSIFICATION
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ── STEP 1: Load data ─────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
feature_names = list(iris.feature_names)
target_names = list(iris.target_names)

print(f"Dataset shape: {X.shape}")
print(f"Classes: {target_names}")
print(f"Class distribution: {dict(zip(target_names, np.bincount(y)))}")
# Dataset shape: (150, 4)
# Classes: ['setosa', 'versicolor', 'virginica']
# Class distribution: {'setosa': 50, 'versicolor': 50, 'virginica': 50}

# ── STEP 2: Brief EDA ─────────────────────────────────────────────────────
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[c] for c in y]
print("\nBasic statistics:")
print(df.groupby('species')[feature_names].mean().round(2))

# Pairplot to see separability
fig = plt.figure(figsize=(10, 8))
colors_map = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for i, f1 in enumerate(feature_names[:2]):
    for j, f2 in enumerate(feature_names[2:]):
        ax = fig.add_subplot(2, 2, i*2+j+1)
        for species, color in colors_map.items():
            mask = df['species'] == species
            ax.scatter(df.loc[mask, f1], df.loc[mask, f2],
                       c=color, alpha=0.7, s=20, label=species)
        ax.set_xlabel(f1, fontsize=8)
        ax.set_ylabel(f2, fontsize=8)
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
plt.suptitle('Iris EDA: Feature Pairs Colored by Species', fontsize=11)
plt.tight_layout()
plt.savefig('iris_eda.png', dpi=100, bbox_inches='tight')
plt.show()

# Key observation: setosa is linearly separable from the other two
# versicolor and virginica have overlap in some feature pairs

# ── STEP 3: Train/test split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining size: {X_train.shape}, Test size: {X_test.shape}")

# ── STEP 4: Baseline pipeline ─────────────────────────────────────────────
baseline_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
])
baseline_pipe.fit(X_train, y_train)
baseline_acc = baseline_pipe.score(X_test, y_test)
print(f"\nBaseline kNN (k=5, distance): Test accuracy = {baseline_acc:.4f}")

# ── STEP 5: Hyperparameter tuning ─────────────────────────────────────────
# Using GridSearchCV with the full Pipeline (correct — no data leakage)
param_grid = {
    'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 31],
    'knn__weights':     ['uniform', 'distance'],
    'knn__p':           [1, 2],    # Manhattan vs Euclidean
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=Pipeline([('scaler', StandardScaler()),
                        ('knn', KNeighborsClassifier())]),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

print(f"\nGrid search best params: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
print(f"Test accuracy (best model): {grid_search.score(X_test, y_test):.4f}")

best_pipe = grid_search.best_estimator_

# ── STEP 6: Optuna tuning (alternative — more efficient for larger spaces) ─
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    k    = trial.suggest_int('n_neighbors', 1, 50)
    w    = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p    = trial.suggest_int('p', 1, 2)

    pipe_trial = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights=w, p=p))
    ])
    scores = cross_val_score(pipe_trial, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=60, show_progress_bar=False)

print(f"\nOptuna best params: {study.best_params}")
print(f"Optuna best CV accuracy: {study.best_value:.4f}")

# Build best Optuna model
best_optuna = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(**{k: v for k, v in study.best_params.items()}))
])
best_optuna.fit(X_train, y_train)
print(f"Optuna test accuracy: {best_optuna.score(X_test, y_test):.4f}")

# ── STEP 7: Final evaluation on test set ──────────────────────────────────
y_pred = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
print(f"Weighted AUC (OvR): {auc_score:.4f}")

# ── STEP 8: Confusion matrix ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                         display_labels=target_names,
                                         ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f'kNN Confusion Matrix (k={best_pipe.named_steps["knn"].n_neighbors})')
plt.tight_layout()
plt.savefig('iris_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.show()

# ── STEP 9: Validation curve (k vs accuracy) ──────────────────────────────
k_values = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 100]
cv_means, cv_stds = [], []

for k in k_values:
    pipe_k = Pipeline([('scaler', StandardScaler()),
                        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))])
    scores = cross_val_score(pipe_k, X_train, y_train, cv=5, scoring='accuracy')
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

cv_means = np.array(cv_means)
cv_stds = np.array(cv_stds)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(k_values, cv_means, 'b-o', linewidth=2, markersize=7)
ax.fill_between(k_values, cv_means - cv_stds, cv_means + cv_stds, alpha=0.2)
ax.axvline(k_values[np.argmax(cv_means)], color='red', linestyle='--',
           label=f'Best k={k_values[np.argmax(cv_means)]}')
ax.set_xlabel('k (n_neighbors, log scale)')
ax.set_ylabel('5-fold CV Accuracy')
ax.set_title('kNN Validation Curve — Iris')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('iris_validation_curve.png', dpi=100, bbox_inches='tight')
plt.show()

# ── STEP 10: Permutation importance ───────────────────────────────────────
perm_imp = permutation_importance(
    best_pipe, X_test, y_test,
    n_repeats=30, scoring='accuracy', random_state=42, n_jobs=-1
)
sorted_idx = perm_imp.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(range(len(sorted_idx)),
        perm_imp.importances_mean[sorted_idx],
        xerr=perm_imp.importances_std[sorted_idx],
        color='steelblue', alpha=0.8, ecolor='gray', capsize=4)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(np.array(feature_names)[sorted_idx])
ax.set_xlabel('Mean accuracy decrease')
ax.set_title('Permutation Feature Importance — kNN (Iris)')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('iris_permutation_importance.png', dpi=100, bbox_inches='tight')
plt.show()

# Expected result: petal length and petal width dominate; sepal width is least important

# ── STEP 11: SHAP KernelExplainer ─────────────────────────────────────────
import shap

background = X_train[:30]

def knn_predict(X_arr):
    return best_pipe.predict_proba(X_arr)

explainer = shap.KernelExplainer(knn_predict, background)
# Explain the 5 test misclassifications (or just first 10 test points)
X_explain = X_test[:10]
shap_values = explainer.shap_values(X_explain, nsamples=100)   # nsamples=100 for speed

# Summary bar plot (global, class-averaged)
shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                  plot_type='bar', show=True, class_names=target_names)

# Force plot for first test prediction
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][0],
    X_explain[0],
    feature_names=feature_names,
    matplotlib=True
)

# ── STEP 12: Interpret results ────────────────────────────────────────────
print("\n" + "="*60)
print("INTERPRETATION SUMMARY")
print("="*60)
print(f"Best model: k={best_pipe.named_steps['knn'].n_neighbors}, "
      f"weights={best_pipe.named_steps['knn'].weights}, "
      f"p={best_pipe.named_steps['knn'].p}")
print(f"Test accuracy: {best_pipe.score(X_test, y_test):.1%}")
print(f"The most important features (permutation): petal dimensions >> sepal dimensions")
print(f"kNN achieves near-perfect accuracy on Iris — this is expected, as the dataset")
print(f"is low-dimensional, well-scaled after StandardScaler, and has clear class structure.")
print(f"The Cover-Hart bound: with Bayes R* ≈ 2%, kNN guarantees at most 4% error — achieved.")
```

### 9.2 Regression: Diabetes Dataset

```python
# ============================================================
# kNN REGRESSION — DIABETES DATASET
# ============================================================
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

diabetes = load_diabetes()
X_r, y_r = diabetes.data, diabetes.target
feat_names_r = list(diabetes.feature_names)
print(f"Diabetes dataset: {X_r.shape}, target range: [{y_r.min():.0f}, {y_r.max():.0f}]")

X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

# Tune k for regression
k_range_r = [1, 3, 5, 7, 10, 15, 20, 30, 50]
r2_cv, mae_cv = [], []

for k in k_range_r:
    pipe_r = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=k, weights='distance'))
    ])
    r2_scores = cross_val_score(pipe_r, X_tr, y_tr, cv=5, scoring='r2')
    mae_scores = -cross_val_score(pipe_r, X_tr, y_tr, cv=5, scoring='neg_mean_absolute_error')
    r2_cv.append(r2_scores.mean())
    mae_cv.append(mae_scores.mean())

best_k_r = k_range_r[np.argmax(r2_cv)]
print(f"\nBest k for regression (by R²): {best_k_r}, CV R² = {max(r2_cv):.4f}")

# Fit best model
pipe_best_r = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=best_k_r, weights='distance'))
])
pipe_best_r.fit(X_tr, y_tr)
y_pred_r = pipe_best_r.predict(X_te)

print(f"\nTest R²:   {r2_score(y_te, y_pred_r):.4f}")
print(f"Test MAE:  {mean_absolute_error(y_te, y_pred_r):.2f}")
print(f"Test RMSE: {mean_squared_error(y_te, y_pred_r)**0.5:.2f}")

# Residual plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. R² vs k
axes[0].plot(k_range_r, r2_cv, 'g-o', linewidth=2)
axes[0].axvline(best_k_r, color='red', linestyle='--', label=f'Best k={best_k_r}')
axes[0].set_xlabel('k')
axes[0].set_ylabel('5-fold CV R²')
axes[0].set_title('R² vs k — Diabetes')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Predicted vs actual
axes[1].scatter(y_te, y_pred_r, alpha=0.6, color='steelblue', s=25)
min_v, max_v = min(y_te.min(), y_pred_r.min()), max(y_te.max(), y_pred_r.max())
axes[1].plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect prediction')
axes[1].set_xlabel('Actual values')
axes[1].set_ylabel('Predicted values')
axes[1].set_title(f'Actual vs Predicted (k={best_k_r})\nR²={r2_score(y_te, y_pred_r):.3f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Residuals vs fitted
residuals_r = y_te - y_pred_r
axes[2].scatter(y_pred_r, residuals_r, alpha=0.6, color='purple', s=25)
axes[2].axhline(0, color='red', linestyle='--')
axes[2].set_xlabel('Fitted values')
axes[2].set_ylabel('Residuals')
axes[2].set_title('Residuals vs Fitted')
axes[2].grid(True, alpha=0.3)

plt.suptitle('kNN Regression Diagnostics — Diabetes Dataset', fontsize=12)
plt.tight_layout()
plt.savefig('diabetes_knn_regression.png', dpi=100, bbox_inches='tight')
plt.show()

# Permutation importance for regression
perm_r = permutation_importance(pipe_best_r, X_te, y_te,
                                  n_repeats=20, scoring='r2',
                                  random_state=42, n_jobs=-1)
sorted_r = perm_r.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(range(len(sorted_r)), perm_r.importances_mean[sorted_r],
        xerr=perm_r.importances_std[sorted_r],
        color='darkorange', alpha=0.8, ecolor='gray', capsize=4)
ax.set_yticks(range(len(sorted_r)))
ax.set_yticklabels(np.array(feat_names_r)[sorted_r])
ax.set_xlabel('Mean R² decrease when feature is permuted')
ax.set_title('Permutation Feature Importance — kNN Regressor (Diabetes)')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('diabetes_permutation_importance.png', dpi=100, bbox_inches='tight')
plt.show()

# Note: Diabetes is a 10-feature dataset with n=442, p=10
# kNN R² typically ~0.45-0.50 — significantly below linear regression R² ~0.51
# This demonstrates the limitation of kNN on moderate-dimensional, moderate-n data
# Linear regression is better here because the true relationship IS approximately linear
```

> 💡 **Key observation from the Diabetes worked example:** kNN achieves R² ≈ 0.45–0.50
> on Diabetes, while linear regression achieves ~0.51–0.53. For data where the true
> relationship is approximately linear, a parametric linear model will beat kNN because
> kNN cannot use global structure — it only looks locally. The lesson: kNN wins when the
> data has complex non-linear local structure; linear models win when global structure is
> informative.

---

## §10 — When to Use This Algorithm: Decision Guide

### 10.1 The Quick Decision Flowchart

```
START: Do you have a supervised learning problem?
          │
          ▼
Is n > 500,000? ─── YES ─── Do you need exact NN? ─── YES ─── Use FAISS IndexFlatL2
          │                           │
          │                          NO
          │                           └─── Use FAISS IVFFlat or hnswlib (ANN)
         NO
          │
          ▼
Is p (features) > 50? ─── YES ─── After PCA/NCA to p ≤ 20?
          │                              │
          │                             YES ─── Proceed to kNN on reduced space
          │                              │
          │                             NO ──── Use Random Forest / GBM instead
         NO
          │
          ▼
Do you have interpretability requirements (show similar cases)? ─── YES ─── kNN excellent
          │
          ▼
Is training time critical? ─── YES ─── kNN is ideal (O(1) fit)
          │
          ▼
Is inference time critical? ─── YES (< 1ms, large n) ─── Use hnswlib/FAISS, not sklearn kNN
          │
         NO (inference time not critical)
          │
          ▼
Is p ≤ 15? ─── YES ─── kNN with kd_tree is excellent; try it
          │
         NO (p = 15–50, medium dimension)
          │
          ▼
Is n ≥ 10 × C^p (enough samples for dimensionality)? 
          │
         YES ──── kNN viable; use ball_tree or brute
          │
         NO ──── kNN will likely underperform; try RF/GBM or reduce p first
```

### 10.2 Detailed Use/Don't-Use Table

| Condition | kNN suitability | Explanation |
|---|---|---|
| **$p \leq 10$, $n \leq 50\text{k}$** | Excellent | Prime kNN regime; tree indexes work; curse of dimensionality minimal |
| **$10 < p \leq 30$, $n \leq 50\text{k}$** | Good with PCA preprocessing | Reduce to $\leq 10$ components first; use ball-tree or brute |
| **$p > 50$** | Poor (without dim reduction) | Equidistance collapse; all distances become similar; use RF/GBM |
| **$p > 50$ + dim reduction to $\leq 20$** | Good | UMAP or PCA first, then kNN on embedding |
| **$n > 100\text{k}$** | Use ANN libraries | sklearn kNN is too slow; use hnswlib, FAISS |
| **$n > 10^7$** | FAISS IVF-PQ only | Memory and compute constraints; only FAISS handles this |
| **Non-linear decision boundaries** | Excellent | kNN naturally handles any boundary shape |
| **Linear decision boundary** | Acceptable but suboptimal | Logistic regression / SVM will be faster and more accurate |
| **Need calibrated probabilities** | Poor by default | Use `CalibratedClassifierCV`; consider logistic regression instead |
| **Class imbalance (>10:1)** | Poor without resampling | Add SMOTE inside Pipeline; use `weights='distance'` |
| **Missing values** | Poor | kNN cannot handle NaN; impute first (but imputation itself can use kNN!) |
| **Sparse features (text, BoW)** | Poor for Euclidean | Use cosine metric + brute force, or use Naive Bayes / linear models |
| **Online/streaming data** | Excellent | $O(1)$ "fit" — just add new points to the lookup set |
| **Need case-based explanations** | Excellent | Show nearest training neighbors — the most interpretable explanation |
| **Need feature importances** | Moderate | Use permutation importance; no built-in importance |
| **Very noisy labels** | Poor | Each noisy point corrupts its neighborhood; use ENN to denoise first |
| **Tabular data, many irrelevant features** | Poor | All features contribute equally; use feature selection or NCA first |
| **Geographic/spatial data (low-dim)** | Excellent | Haversine metric via BallTree handles lat/lon natively |
| **Image retrieval / embedding search** | Excellent with ANN | FAISS/hnswlib + cosine similarity on embeddings = production image search |
| **Small dataset ($n < 100$)** | Good | No training required; all points become neighbors; $k=1$–3 |
| **Training data must be private** | Poor | kNN stores all training data; model inversion reveals training samples |

### 10.3 kNN vs. The Alternatives — A Head-to-Head Comparison

| Criterion | kNN | Logistic Reg | Random Forest | SVM (RBF) | MLP |
|---|---|---|---|---|---|
| Training time | $O(1)$ ⭐ | $O(np)$ fast | $O(n p \log n)$ | $O(n^2 p)$ slow | $O(n p e)$ |
| Inference time | $O(np)$ slow | $O(p)$ ⭐ | $O(p \cdot T)$ | $O(n_{\text{sv}} \cdot p)$ | $O(p \cdot L)$ |
| Memory | $O(np)$ high | $O(p)$ ⭐ | $O(T \cdot \text{depth})$ | $O(n_{\text{sv}} \cdot p)$ | $O(p \cdot L)$ |
| Non-linear boundaries | ⭐ perfect | Poor (linear only) | ⭐ excellent | ⭐ excellent | ⭐ excellent |
| Curse of dimensionality | Severe | None | Moderate | Moderate | Moderate |
| Missing value handling | No (impute first) | With care | ⭐ (native in RF) | No | No |
| Calibrated probabilities | Poor | ⭐ | Moderate | Poor | Good |
| Interpretability | ⭐ (case-based) | ⭐ (coefficients) | SHAP | SHAP | SHAP |
| Incremental updates | ⭐ (add data) | Retrain | Retrain | Retrain | Retrain |
| Hyperparameter sensitivity | Low-medium | Low | Medium | High | High |

> 🏆 **Best practice summary:** Use kNN when you need (1) zero training time, (2) non-linear
> boundaries in low-to-moderate dimensions, (3) case-based explanations, or (4) a fast
> non-parametric baseline. Switch to Random Forest or GBM when dimensionality grows,
> n grows beyond 50k, or you need better calibrated probabilities. Use hnswlib/FAISS when
> n grows beyond 100k and you need efficient similarity search rather than full supervised
> learning.

---

## §11 — Resources & Further Reading

### Original Papers

1. **Fix, E., and Hodges, J. L. (1951).** *Discriminatory Analysis, Nonparametric
   Discrimination: Consistency Properties.* Technical Report 4, Project 21-49-004,
   USAF School of Aviation Medicine. Reprinted in *International Statistical Review*,
   57(3), 238–247 (1989). Reference: https://oro.open.ac.uk/28327/

2. **Cover, T. M., and Hart, P. E. (1967).** Nearest neighbor pattern classification.
   *IEEE Transactions on Information Theory*, 13(1), 21–27.
   DOI: https://doi.org/10.1109/TIT.1967.1053964
   Free preprint: https://isl.stanford.edu/~cover/papers/transIT/0050cove.pdf

3. **Hart, P. E. (1968).** The condensed nearest neighbor rule.
   *IEEE Transactions on Information Theory*, 14(3), 515–516.
   DOI: https://doi.org/10.1109/TIT.1968.1054155

4. **Wilson, D. L. (1972).** Asymptotic properties of nearest neighbor rules using
   edited data. *IEEE Transactions on Systems, Man, and Cybernetics*, 2(3), 408–421.
   DOI: https://doi.org/10.1109/TSMC.1972.4309137

5. **Beyer, K., Goldstein, J., Ramakrishnan, R., and Shaft, U. (1999).** When is
   "Nearest Neighbor" Meaningful? *International Conference on Database Theory (ICDT)*,
   217–235. The definitive paper on the curse of dimensionality for nearest neighbor search.

6. **Goldberger, J., Roweis, S., Hinton, G., and Salakhutdinov, R. (2004).**
   Neighbourhood Components Analysis. *NeurIPS*, 17.
   URL: https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf

7. **Malkov, Y. A., and Yashunin, D. A. (2018).** Efficient and robust approximate
   nearest neighbor search using Hierarchical Navigable Small World graphs.
   *IEEE TPAMI*, 42(4), 824–836. arXiv: https://arxiv.org/abs/1603.09320

8. **Johnson, J., Douze, M., and Jégou, H. (2021).** Billion-scale similarity search
   with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
   arXiv: https://arxiv.org/abs/1702.08734

9. **Cleveland, W. S. (1979).** Robust locally weighted regression and smoothing
   scatterplots. *Journal of the American Statistical Association*, 74(368), 829–836.
   PDF: https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf

### Official Documentation

10. **sklearn Neighbors User Guide (1.5.x):**
    https://scikit-learn.org/1.5/modules/neighbors.html
    The definitive reference for all sklearn NN estimators; includes algorithm selection
    guide and metric compatibility tables.

11. **KNeighborsClassifier API:**
    https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

12. **KNeighborsRegressor API:**
    https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

13. **NeighborhoodComponentsAnalysis API:**
    https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html

### Approximate Nearest Neighbor Libraries

14. **FAISS GitHub Wiki — Index Types:**
    https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    The authoritative guide to choosing between IndexFlatL2, IndexIVFFlat, IndexIVFPQ,
    IndexHNSWFlat, and IndexPQ.

15. **hnswlib GitHub with Python Examples:**
    https://github.com/nmslib/hnswlib/blob/master/examples/python/EXAMPLES.md

16. **PyNNDescent Documentation:**
    https://pynndescent.readthedocs.io/

17. **Annoy GitHub (Spotify):**
    https://github.com/spotify/annoy

18. **Voyager (Spotify's successor to Annoy):**
    https://github.com/spotify/voyager

19. **ANN Benchmarks (Erik Bernhardsson):**
    https://github.com/erikbern/ann-benchmarks
    The gold standard for benchmarking FAISS, hnswlib, Annoy, PyNNDescent,
    and others on standard datasets. Always check this before committing to a library.

### Best Tutorials & Blog Posts

20. **Pinecone — FAISS Tutorial (IndexIVFFlat production guide):**
    https://www.pinecone.io/learn/series/faiss/faiss-tutorial/
    The best hands-on production guide for understanding and tuning FAISS IVF indexes.

21. **Pinecone — HNSW Explained (visual explainer):**
    https://www.pinecone.io/learn/series/faiss/hnsw/
    The best visual explanation of HNSW layers and navigable small-world properties.

22. **sklearn User Guide — Nearest Neighbors:**
    https://scikit-learn.org/stable/modules/neighbors.html
    Excellent treatment of algorithm selection (kd-tree vs ball-tree vs brute) with
    examples and the metric compatibility table.

### Books

23. **Devroye, L., Györfi, L., and Lugosi, G. (1996).** *A Probabilistic Theory of
    Pattern Recognition.* Springer. Chapter 11: "The Nearest Neighbor Rule." The
    most rigorous mathematical treatment of kNN consistency and convergence rates.

24. **Hastie, T., Tibshirani, R., and Friedman, J. (2009).** *The Elements of
    Statistical Learning*, 2nd ed. Springer. Free PDF: https://hastie.su.domains/ElemStatLearn/
    Chapter 2 (kNN as a prototype method) and Chapter 13 (prototype methods and kNN in depth).

25. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning.* Springer.
    Chapter 2.5 (non-parametric methods, Parzen windows, kNN density estimation) —
    the clearest derivation of kNN as a density estimator.

26. **Goodfellow, I., Bengio, Y., and Courville, A. (2016).** *Deep Learning.* MIT Press.
    Free: https://www.deeplearningbook.org/. Chapter 5.4 (kNN as a non-parametric
    baseline and motivation for deep feature learning).

### Online Courses

27. **fast.ai Practical Machine Learning for Coders:**
    https://course.fast.ai/
    Uses kNN as one of the first models to build intuition about feature scaling and
    distance geometry before moving to more complex methods.

28. **Stanford CS229 — Machine Learning (Andrew Ng):**
    Lecture notes available at https://cs229.stanford.edu/. Covers kNN in the context of
    non-parametric methods and the bias-variance tradeoff.

---

*This chapter is part of the Supervised Learning Masterclass series. See also:
[Logistic Regression](logistic-regression.md), [Decision Trees](decision-trees.md),
[Random Forests](random-forests.md), [Support Vector Machines](support-vector-machines.md).*
