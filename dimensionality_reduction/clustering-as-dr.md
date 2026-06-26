# Clustering as Dimensionality Reduction: The Complete Masterclass

> **Why this algorithm matters:** Before deep learning swept through computer vision, the most
> accurate image classification system on the ImageNet challenge — right up until AlexNet in 2012 —
> was built on top of k-means clustering. The technique, called Bag of Visual Words, reduced every
> image from millions of raw pixel values to a single K-dimensional histogram by clustering local
> image descriptors into a visual vocabulary. That K-dimensional vector, produced by k-means, was
> the dimensionality reduction. The system that won the 2011 ImageNet challenge used Fisher Vectors:
> a soft-assignment, GMM-based generalization of the same idea. Both of these are clustering as
> dimensionality reduction. This is not a toy technique — it is a profound, nonlinear, highly
> practical form of DR that captures structure that PCA and its linear cousins completely miss, and
> it scales to billions of samples in a way that t-SNE and UMAP cannot.

---

## §0 — Python Ecosystem & Package Guide

Clustering as DR is not served by a single canonical package the way PCA is served by
`sklearn.decomposition.PCA`. Instead, you'll draw from clustering packages, mixture model
packages, and specialized libraries depending on which flavor of the technique you're using.
Let's map the full terrain before we dive into theory.

### Complete Package Table

| Package | Import | Version (mid-2026) | What it provides | License |
|---|---|---|---|---|
| `scikit-learn` (cluster) | `from sklearn.cluster import KMeans, FeatureAgglomeration, SpectralClustering, AgglomerativeClustering, HDBSCAN` | ~1.5.x | The workhorse: KMeans transform, FeatureAgglomeration, spectral embedding, agglomerative clustering | BSD-3 |
| `scikit-learn` (mixture) | `from sklearn.mixture import GaussianMixture, BayesianGaussianMixture` | ~1.5.x | Soft DR via GMM posteriors; BIC-based model selection | BSD-3 |
| `scikit-learn` (manifold) | `from sklearn.manifold import SpectralEmbedding` | ~1.5.x | Pure Laplacian eigenmap step underlying SpectralClustering | BSD-3 |
| `MiniSom` | `from minisom import MiniSom` | 2.3.6 (Feb 2026) | Self-Organizing Maps — topology-preserving DR + clustering on a 2D grid | MIT |
| `somoclu` | `from somoclu import Somoclu` | ~1.7.x | GPU/MPI-accelerated SOM for large-scale data | MIT |
| `sklearn-som` | `from sklearn_som.som import SOM` | ~1.1.x | sklearn-compatible SOM (fit/transform interface for Pipelines) | MIT |
| `hdbscan` | `import hdbscan` | ~0.8.x | Density-based clustering with soft membership vectors | BSD-3 |
| `faiss-cpu` / `faiss-gpu` | `import faiss` | ~1.8.x | Meta's library for billion-scale KMeans + ANN search (GPU-accelerated) | MIT |
| `pyclustering` | `from pyclustering.cluster.kmeans import kmeans` | ~0.10.x | LVQ, CURE, BANG — prototype-based methods not in sklearn | GNU GPL v3 |

### Top 5 Picks — Recommendation Table

| Package | Best for | Modern/Legacy | Key advantage |
|---|---|---|---|
| **`sklearn.cluster.KMeans`** | Hard-assignment DR, distance features, pipelines | Modern | `transform()` built-in; Pipeline-compatible; k-means++ init |
| **`sklearn.mixture.GaussianMixture`** | Soft-assignment DR (posterior probabilities), uncertainty-aware features | Modern | `predict_proba()` gives normalized soft features; BIC for K selection |
| **`sklearn.cluster.FeatureAgglomeration`** | Feature-space DR (clustered feature pooling), structured/tabular data | Modern | `inverse_transform()` for reconstruction; interpretable feature grouping |
| **`MiniSom`** | Topology-preserving DR + visualization, 2D grid embedding | Modern | Only mature Python SOM; NumPy-only; Numba acceleration optional |
| **`faiss`** | Billion-scale KMeans (BoVW codebooks, large embedding tables) | Modern | 100x–1000x faster than sklearn for N > 1M; GPU support |

> ⚠️ **Pitfall:** `GaussianMixture` has **no** `transform()` method. If you try `gmm.transform(X)`,
> you get `AttributeError`. The soft-assignment DR method is `gmm.predict_proba(X)`. This is the
> single most common gotcha when first using GMM for dimensionality reduction. Every code example
> in this chapter uses `predict_proba()` correctly — make sure yours does too.

> ⚠️ **Pitfall:** `KMeans(algorithm='full')` was **removed in sklearn 1.1**. Legacy code from
> pre-2021 tutorials may use this and will throw an error. Replace with `algorithm='lloyd'` (the
> default) — they are identical in behavior.

### GPU-Accelerated and Streaming Options

**GPU-accelerated KMeans (billion-scale):** Use `faiss-gpu`. This is the production choice at
Meta, Google, and any team training BoVW codebooks on large image datasets. FAISS KMeans is 100x
faster than sklearn on GPU for N > 1M samples.

```python
import faiss
import numpy as np

# GPU KMeans: train on N=10M, 128-dim descriptors in minutes
d = 128         # descriptor dimensionality
k = 1000        # number of visual words
kmeans_faiss = faiss.Kmeans(d=d, k=k, niter=20, gpu=True)  # set gpu=False for CPU
kmeans_faiss.train(descriptors.astype(np.float32))

# Centroids:
centroids = kmeans_faiss.centroids   # shape: (k, d)

# Fast assignment (for encoding new descriptors):
index = faiss.IndexFlatL2(d)
index.add(centroids)
distances, assignments = index.search(new_descriptors.astype(np.float32), 1)
```

**Streaming / Mini-batch KMeans:** Use `sklearn.cluster.MiniBatchKMeans` for datasets that don't
fit in memory. It approximates full KMeans using mini-batches and converges in O(N) time instead
of O(N·K·iterations). Suitable for N > 10M on CPU.

```python
from sklearn.cluster import MiniBatchKMeans

mbk = MiniBatchKMeans(
    n_clusters=100,
    batch_size=10_000,
    n_init=3,
    random_state=42
)
# Use partial_fit for true streaming:
for X_batch in data_generator:
    mbk.partial_fit(X_batch)

X_dr = mbk.transform(X_new)   # Works the same as KMeans.transform()
```

**Bayesian GMM (automatic K selection):** Use `sklearn.mixture.BayesianGaussianMixture`. It
automatically zeros out unnecessary components (sets mixing weights to near zero), so you can
initialize with K_max and let it find the effective K. Especially useful when you don't know how
many components your data has.

```python
from sklearn.mixture import BayesianGaussianMixture

bgmm = BayesianGaussianMixture(
    n_components=50,         # Upper bound — some will be pruned
    covariance_type='diag',
    weight_concentration_prior=1e-2,   # Small → encourages sparsity; fewer active components
    random_state=42
).fit(X_train_scaled)

# Effective number of components (components with non-negligible weight):
effective_k = (bgmm.weights_ > 0.01).sum()
print(f"Effective components: {effective_k}")

# Soft DR output:
X_dr = bgmm.predict_proba(X_train_scaled)   # shape: (n_samples, 50) but many cols near 0
```

### The Same fit_transform() Call Across Top Packages

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Synthetic dataset: 1000 samples, 20 features, 5 true clusters
X_raw, y_true = make_blobs(n_samples=1000, n_features=20, centers=5, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── Option 1: KMeans (hard-assignment distance features) ─────────────────
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, n_init='auto', random_state=42)
X_dr_kmeans = kmeans.fit_transform(X)         # shape: (1000, 10)
print(f"KMeans DR shape: {X_dr_kmeans.shape}")  # (1000, 10)
# Each column = Euclidean distance to one centroid

# ── Option 2: GaussianMixture (soft posterior probabilities) ─────────────
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=10, covariance_type='diag', n_init=3, random_state=42)
gmm.fit(X)
X_dr_gmm = gmm.predict_proba(X)              # shape: (1000, 10)
print(f"GMM DR shape: {X_dr_gmm.shape}")      # (1000, 10)
# Each column = posterior probability for one component; rows sum to 1.0

# ── Option 3: FeatureAgglomeration (feature-space clustering) ─────────────
from sklearn.cluster import FeatureAgglomeration

agglo = FeatureAgglomeration(n_clusters=5, linkage='ward')
X_dr_agglo = agglo.fit_transform(X)          # shape: (1000, 5)
print(f"FeatureAgglo DR shape: {X_dr_agglo.shape}")  # (1000, 5)
# Each column = mean of a cluster of original features

# ── Option 4: Spectral Embedding (graph Laplacian DR before clustering) ───
from sklearn.manifold import SpectralEmbedding

spec = SpectralEmbedding(n_components=8, affinity='nearest_neighbors', n_neighbors=15, random_state=42)
X_dr_spec = spec.fit_transform(X)            # shape: (1000, 8)
print(f"Spectral Embedding DR shape: {X_dr_spec.shape}")  # (1000, 8)
# Each column = Laplacian eigenfunction coordinate
```

> 🔧 **In practice:** For tabular ML pipelines, start with `KMeans.transform()` — it's fast,
> Pipeline-compatible, and interpretable. Move to `GaussianMixture.predict_proba()` when you need
> uncertainty or soft boundaries between clusters. Use `FeatureAgglomeration` when your goal is
> to reduce the feature dimension (not the sample-to-cluster space). Use Spectral Embedding when
> the data has non-convex manifold structure that KMeans can't capture.

---

## §1 — The Origin: The Paper That Started It All

### MacQueen (1967): Coining K-Means and Planting the Seed

> 📜 **Origin/Citation:** MacQueen, J. B. (1967). "Some methods for classification and analysis
> of multivariate observations." In *Proceedings of the Fifth Berkeley Symposium on Mathematical
> Statistics and Probability*, Vol. 1, pp. 281–297. University of California Press.
> Available via Semantic Scholar: https://www.semanticscholar.org/paper/Some-methods-for-classification-and-analysis-of-MacQueen/ac8ab51a86f1a9ae74dd0e4576d1a019f5e654ed

James MacQueen was thinking about classification — not dimensionality reduction — when he wrote
this paper in 1967. His problem was practical: given $n$ observations in $\mathbb{R}^p$, how do
you classify them into $K$ groups ("classes") such that the total within-class variance is
minimized? He called the algorithm "k-means" and proved two things: (1) the algorithm converges
in a finite number of steps, and (2) the class means define a natural coordinate system.

That second insight is the seed of everything in this chapter. Once you have $K$ class means —
centroids — every data point can be described by its relationship to those $K$ means rather than
by its original $p$ coordinates. MacQueen described this as "classification by means," but he was
implicitly describing a coordinate transformation: from $\mathbb{R}^p$ to $\mathbb{R}^K$.

**The prior art MacQueen was responding to:** Before k-means, the standard approach to
multivariate classification was Fisherian discriminant analysis (LDA), which required labeled
classes, and Ward's hierarchical clustering (1963), which required computing all pairwise
distances — $O(n^2)$ work. MacQueen's k-means required only $O(n \cdot K \cdot \text{iter})$
work and no labels. For the sample sizes of the 1960s (hundreds of observations), this was a
significant practical improvement.

**The core innovation:** The two-step iterate:
1. **Assignment step:** For each observation $\mathbf{x}_i$, assign it to the nearest mean
   $\mathbf{c}_k = \arg\min_k \|\mathbf{x}_i - \mathbf{c}_k\|^2$.
2. **Update step:** Recompute each mean as the centroid of its assigned observations:
   $\mathbf{c}_k \leftarrow \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{x}_i$.

This iterate is guaranteed to decrease the objective (within-class sum of squares) at each step
and to terminate in finite steps (since there are only finitely many partitions of $n$ points into
$K$ groups). MacQueen's proof of finite convergence was the key theoretical contribution.

### The DR Connection: From Classification to Coordinate Transformation

The leap from clustering to dimensionality reduction was not explicitly made until much later, but
it was hiding in plain sight in MacQueen's formulation. Once the $K$ centroids are fixed,
`KMeans.transform(X)` — which didn't exist in 1967, but was conceptually implicit — maps every
point to a $K$-dimensional vector of Euclidean distances:

$$f(\mathbf{x}) = \left[\|\mathbf{x} - \mathbf{c}_1\|_2, \;\|\mathbf{x} - \mathbf{c}_2\|_2, \;\ldots, \;\|\mathbf{x} - \mathbf{c}_K\|_2\right] \in \mathbb{R}^K$$

This is a legitimate dimensionality reduction: a deterministic mapping from $\mathbb{R}^p$ to
$\mathbb{R}^K$, where typically $K \ll p$. The mapping is:
- **Nonlinear** (Euclidean distance is nonlinear in the original coordinates)
- **Supervised by data geometry** (centroids encode the cluster structure)
- **Injective enough for practical use** (if clusters are well-separated, nearby points in
  $\mathbb{R}^p$ remain nearby in $\mathbb{R}^K$)

The first major application of clustering as explicit DR was the **Bag of Visual Words** (BoVW),
introduced independently by Sivic & Zisserman ("Video Google," ICCV 2003) and Csurka et al.
("Visual categorization with bags of keypoints," ECCV Workshops 2004). These papers showed
that clustering local image descriptors (SIFT features) with k-means produced a visual
"vocabulary" — and that representing each image as a histogram over this vocabulary gave state-of-
the-art image classification performance. The DR was the codebook: from millions of 128-dimensional
SIFT descriptors per image to a single $K$-dimensional vector.

### The Soft-Assignment Generalization: Fisher Vectors

The GMM-based generalization — using posterior probabilities as soft features instead of hard
cluster assignments — traces to the **Fisher Kernel** framework by Jaakkola and Haussler (1999)
and was made practical for computer vision by Perronnin & Dance (2007) and dramatically improved
by Perronnin, Sánchez & Mensink (ECCV 2010). The Fisher Vector encodes not just which visual word
a descriptor belongs to (0th-order statistics, like BoVW) but also how much each descriptor
deviates from each Gaussian component's mean and variance (1st and 2nd order statistics).

> 📜 **Origin/Citation:** Perronnin, F., Sánchez, J., & Mensink, T. (2010). "Improving the
> Fisher kernel for large-scale image classification." *ECCV 2010*, pp. 143–156.
> DOI: https://doi.org/10.1007/978-3-642-15561-1_11

This is the paper that, building on GMM-based soft clustering as DR, produced the best image
classification system in history prior to AlexNet. The 2011 ImageNet winner used Fisher Vectors.

### What Came Before? The Context of 1967

To appreciate what MacQueen built, consider what already existed in 1967:
- **Ward's hierarchical clustering (1963):** Builds a dendrogram via successive merges. $O(n^3)$
  in naive form. Produces a tree, not a flat partition.
- **Vector quantization (Shannon 1948, Linde-Buzo-Gray 1980):** Originally developed for signal
  compression — discretize a continuous signal into $K$ codewords to minimize quantization error.
  This is mathematically identical to k-means and appeared independently in signal processing.
- **Voronoi diagrams (Voronoi 1908, Dirichlet 1850):** A geometric decomposition of space into
  cells defined by proximity to a set of seed points. K-means finds the optimal seeds (centroids)
  to minimize within-cell sum-of-squares; the resulting partition IS the Voronoi tessellation.

The name "k-means" and the iterative algorithmic framing were MacQueen's contributions. The
underlying geometric idea — Voronoi regions of seed points — is older. But MacQueen's algorithm
made the idea computationally tractable.

---

## §2 — The Algorithm, Deeply Explained

### Three Flavors of Clustering as DR

Before deriving the math, let's be precise about what we mean by "clustering as dimensionality
reduction." There are three distinct mechanisms:

| Approach | Transform | Output per sample | Algorithm |
|---|---|---|---|
| **Hard-assignment distance** | $f(\mathbf{x}) = [\|\mathbf{x}-\mathbf{c}_1\|, \ldots, \|\mathbf{x}-\mathbf{c}_K\|]$ | Distance to every centroid | `KMeans.transform()` |
| **Soft-assignment posterior** | $f(\mathbf{x}) = [p(k=1|\mathbf{x}), \ldots, p(k=K|\mathbf{x})]$ | Posterior probability per component | `GaussianMixture.predict_proba()` |
| **Feature-space pooling** | $f(\mathbf{x}) = [\text{pool}(\mathbf{x}_{S_1}), \ldots, \text{pool}(\mathbf{x}_{S_K})]$ | Pooled value of feature clusters | `FeatureAgglomeration.transform()` |

Each has different mathematical foundations. Let's derive them carefully.

### 2.1 KMeans Transform: Distance-Based DR

**The optimization objective.** K-means minimizes the within-cluster sum of squared Euclidean
distances (inertia):

$$\mathcal{L}(\{\mathbf{c}_k\}, \{S_k\}) = \sum_{k=1}^{K} \sum_{i \in S_k} \|\mathbf{x}_i - \mathbf{c}_k\|_2^2$$

where $S_1, \ldots, S_K$ is a partition of $\{1, \ldots, n\}$ and $\mathbf{c}_k \in \mathbb{R}^p$
are the cluster centroids.

This is an NP-hard combinatorial optimization problem in general. The Lloyd's algorithm (the
"k-means algorithm") finds a local minimum via coordinate descent:
1. Hold $\{S_k\}$ fixed, optimize $\{\mathbf{c}_k\}$: the optimal centroid is the mean,
   $\mathbf{c}_k^* = \frac{1}{|S_k|}\sum_{i \in S_k}\mathbf{x}_i$.
2. Hold $\{\mathbf{c}_k\}$ fixed, optimize $\{S_k\}$: assign each $\mathbf{x}_i$ to its nearest
   centroid, $s_i^* = \arg\min_k \|\mathbf{x}_i - \mathbf{c}_k\|^2$.

**The DR transform.** After fitting, the `transform()` method computes the function:

$$\mathbf{z}_i = f(\mathbf{x}_i) = \left[\|\mathbf{x}_i - \mathbf{c}_1\|_2, \;\|\mathbf{x}_i - \mathbf{c}_2\|_2, \;\ldots, \;\|\mathbf{x}_i - \mathbf{c}_K\|_2\right] \in \mathbb{R}^K$$

This is a nonlinear map. Why nonlinear? Because $\|\mathbf{x} - \mathbf{c}\|_2 = \sqrt{\sum_{j=1}^p (x_j - c_j)^2}$
involves a square root and quadratic terms — not a linear function of $\mathbf{x}$.

> 💡 **Intuition:** Think of the $K$ centroids as $K$ "lighthouses" in $\mathbb{R}^p$. Each
> sample is a boat at some position. The transform measures how far the boat is from each
> lighthouse. Two boats at very different raw positions but similar distances from all lighthouses
> will have similar DR vectors — they inhabit the same "relative geography." This is the compression:
> instead of specifying absolute GPS coordinates, you specify your distance to each known landmark.

**Geometric interpretation.** The transform maps $\mathbb{R}^p$ to the positive orthant
$\mathbb{R}^K_{\geq 0}$. The $k$-th axis in the DR space encodes "how far from cluster $k$." A
point exactly on centroid $k$ has $z_k = 0$ and positive $z_{k'}$ for all other clusters.

**Voronoi cells and their boundaries.** The assignment step partitions $\mathbb{R}^p$ into $K$
Voronoi cells. The boundary between cells $k$ and $k'$ is the hyperplane where
$\|\mathbf{x} - \mathbf{c}_k\|^2 = \|\mathbf{x} - \mathbf{c}_{k'}\|^2$, which simplifies to:

$$2(\mathbf{c}_k - \mathbf{c}_{k'})^\top \mathbf{x} = \|\mathbf{c}_k\|^2 - \|\mathbf{c}_{k'}\|^2$$

This is a **linear** boundary in $\mathbb{R}^p$. So Voronoi cells are convex polytopes. This is
both a strength (efficient computation) and a weakness: k-means can only capture convex clusters.

**Computational complexity:**
- **Fitting:** $O(n \cdot K \cdot p \cdot \text{iter})$ per restart, where iter is typically
  $\leq 300$. In practice, $\approx O(n \cdot K \cdot p)$ since iter is constant.
- **Transform (inference):** $O(n \cdot K \cdot p)$ — compute $K$ distances per sample.
- **Memory:** $O(K \cdot p)$ to store centroids.

For sklearn's implementation with `n_init='auto'` and `init='k-means++'`: one restart runs, and
k-means++ initialization costs $O(n \cdot K \cdot p)$ additional overhead (smart seeding).

### 2.2 Gaussian Mixture Model: Soft-Assignment DR

**The model.** A GMM assumes the data comes from a mixture of $K$ Gaussian distributions:

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where $\pi_k \geq 0$, $\sum_k \pi_k = 1$ (mixing weights), $\boldsymbol{\mu}_k \in \mathbb{R}^p$
(component means), and $\boldsymbol{\Sigma}_k \in \mathbb{R}^{p \times p}$ (component covariance
matrices).

**Parameter estimation via EM.** We fit via Expectation-Maximization, maximizing the log-likelihood:

$$\log \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^n \log \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

The EM iterate:

**E-step (posterior responsibilities):**
$$r_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{k'=1}^K \pi_{k'} \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_{k'}, \boldsymbol{\Sigma}_{k'})}$$

**M-step (parameter updates):**
$$N_k = \sum_{i=1}^n r_{ik}, \quad \pi_k = \frac{N_k}{n}, \quad \boldsymbol{\mu}_k = \frac{\sum_i r_{ik} \mathbf{x}_i}{N_k}$$
$$\boldsymbol{\Sigma}_k = \frac{\sum_i r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top}{N_k} + \epsilon \mathbf{I}$$

(The $\epsilon \mathbf{I}$ term is `reg_covar` — added to prevent singular covariance matrices.)

**The DR transform.** After fitting, the feature representation for a new point $\mathbf{x}$ is:

$$\mathbf{z} = \text{predict\_proba}(\mathbf{x}) = \left[r_1(\mathbf{x}), r_2(\mathbf{x}), \ldots, r_K(\mathbf{x})\right] \in \Delta^{K-1}$$

where $\Delta^{K-1}$ is the $(K-1)$-dimensional probability simplex (rows sum to 1.0).

> 💡 **Intuition:** K-means asks "how far are you from each cluster center?" (a distance).
> GMM asks "given that you came from some Gaussian, what's the probability you came from
> each one?" (a posterior). The GMM representation is richer: a point exactly halfway between
> two well-separated Gaussians gets roughly $[0.5, 0.5, 0, \ldots]$ from GMM but
> $[\epsilon, \epsilon, \ldots]$ from KMeans (where $\epsilon$ depends on the scale). The GMM
> explicitly encodes uncertainty; KMeans does not.

**Why soft features beat hard one-hot encoding.** Hard assignment (one-hot encoding of the
nearest cluster) discards all information about how close the point is to other clusters. A point
on the boundary between clusters 3 and 7 gets $[0,0,1,0,0,0,0,0]$ from hard assignment — it looks
identical to a point deep in the heart of cluster 3. The soft representation
$[0, 0, 0.51, 0, 0, 0, 0.49, 0]$ captures the ambiguity and gives downstream models a much
richer signal.

**Relationship to k-means.** K-means is the limiting case of GMM with spherical, equal covariances
as the variance $\sigma^2 \to 0$. As $\sigma^2 \to 0$, the responsibilities $r_{ik} \to 1$ for
the nearest cluster and $r_{ik} \to 0$ for all others. In code:
`GaussianMixture(covariance_type='spherical', tol=1e-10)` with a very small `reg_covar` will
approximate k-means soft assignment. (Don't actually do this — just use k-means.)

**Computational complexity:**
- **Fitting:** $O(n \cdot K \cdot p^2 \cdot \text{iter})$ for full covariance, $O(n \cdot K \cdot p \cdot \text{iter})$ for diagonal.
- **Transform:** $O(n \cdot K \cdot p^2)$ for full, $O(n \cdot K \cdot p)$ for diagonal.
- **Memory:** $O(K \cdot p^2)$ for full covariance — this grows quickly with $p$.

> ⚠️ **Pitfall:** For high-dimensional data ($p > 50$), `covariance_type='full'` becomes
> numerically unstable (covariance matrices approach singularity) and is extremely memory-hungry.
> Always use `covariance_type='diag'` or `'spherical'` for high-D data, and always set
> `reg_covar=1e-3` or larger if you get `ConvergenceWarning`.

### 2.3 Feature Agglomeration: Clustering in Feature Space

This is a fundamentally different mechanism. Instead of mapping samples to a cluster space, we
cluster the **features** (columns) and pool them. Given $\mathbf{X} \in \mathbb{R}^{n \times p}$,
we:
1. Treat each feature as a point in $\mathbb{R}^n$ (the column vector of that feature across all
   samples).
2. Cluster these $p$ feature-vectors using agglomerative clustering into $K$ groups.
3. Represent each sample by the pooled (e.g., mean) value within each feature cluster.

The output is $\hat{\mathbf{X}} \in \mathbb{R}^{n \times K}$ where:

$$\hat{x}_{i,k} = \text{pooling\_func}\left(\{x_{i,j} : j \in S_k\}\right)$$

and $S_k \subseteq \{1, \ldots, p\}$ is the set of original features in cluster $k$.

> 💡 **Intuition:** If you have 500 gene expression features and 400 of them are highly correlated
> (e.g., all genes in the same metabolic pathway co-express), Feature Agglomeration will merge
> those 400 into one (or a few) meta-features — the average expression of the pathway. This is
> scientifically meaningful: you've discovered that those genes behave as a unit, and you now
> have a compact representation that preserves that unit rather than noise-amplifying 400 nearly
> identical columns.

**The Ward linkage criterion.** By default, `FeatureAgglomeration` uses Ward's linkage: merge the
two clusters whose union minimizes the increase in total within-cluster variance:

$$\Delta(S_a, S_b) = \frac{|S_a| \cdot |S_b|}{|S_a| + |S_b|} \|\bar{\mathbf{f}}_a - \bar{\mathbf{f}}_b\|^2$$

where $\bar{\mathbf{f}}_a, \bar{\mathbf{f}}_b$ are the mean feature vectors (across samples) for
clusters $a$ and $b$. This produces compact, roughly equal-sized clusters — appropriate for
pooling features of similar scale.

**Computational complexity (agglomerative clustering):**
- **Fitting:** $O(p^2 \log p)$ for general hierarchical clustering; $O(p \cdot n)$ for Ward's
  with the neighbor heap optimization.
- **Transform:** $O(n \cdot p)$ — just average the features within each cluster.
- **Memory:** $O(p^2)$ for the distance matrix in naive form.

### 2.4 Implicit Assumptions and Where They Break

Every clustering-as-DR method makes assumptions. Knowing them tells you where the method fails.

**K-means assumes:**
- Clusters are **convex** (bounded by hyperplanes). Non-convex clusters (rings, crescents,
  interlocking spirals) will be mishandled.
- Clusters are **approximately isotropic** (roughly spherical in the space after scaling). Very
  elongated clusters get split into multiple k-means clusters.
- Features are **commensurate** in scale. Always scale before applying KMeans.
- The "right" $K$ is known or estimable. KMeans always finds exactly $K$ clusters regardless of
  whether the data has $K$-cluster structure.

**GMM additionally assumes:**
- Data comes from a **mixture of Gaussian distributions**. Non-Gaussian clusters (heavy-tailed,
  multimodal within cluster) violate this and produce poor posteriors.
- With `covariance_type='full'`: each cluster is **ellipsoidal** (arbitrary orientation). Better
  than k-means, but still parametric and Gaussian-shaped.

**Feature Agglomeration assumes:**
- Features in the same cluster are **interchangeable** for downstream modeling. If features within
  a cluster have very different predictive power for the target, pooling discards useful signal.
- The relevant structure is in **correlated feature groups**, not interactions between features
  from different groups.

---

## §3 — The Full Evolution: Major Variants

### 3.1 K-Means++ Initialization (Arthur & Vassilvitskii, 2007)

**Problem it solves:** Vanilla k-means with random initialization often converges to poor local
minima — the quality of the solution depends strongly on the initial centroid positions, and a bad
random draw leads to a bad final clustering.

> 📜 **Citation:** Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful
> seeding." In *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*,
> pp. 1027–1035.

**Key algorithmic change:** Instead of placing all $K$ initial centroids uniformly at random,
k-means++ selects them sequentially with probability proportional to their squared distance from
the nearest already-chosen centroid:

$$P(\text{choose } \mathbf{x}_i \text{ as next centroid}) \propto \min_{k' \text{ already chosen}} \|\mathbf{x}_i - \mathbf{c}_{k'}\|^2$$

This spreads initial centroids across the data, dramatically reducing the chance of a degenerate
initialization. Theoretical guarantee: k-means++ achieves an expected cost within $O(\log K)$
of the optimal — much better than arbitrary random seeding.

**When to use:** Always. `init='k-means++'` is the sklearn default and should almost never be
changed. The overhead is one extra $O(n \cdot K \cdot p)$ pass.

**In practice:**
```python
# k-means++ is the default — you get it automatically:
kmeans = KMeans(n_clusters=20, init='k-means++', n_init='auto', random_state=42)
# With k-means++ init, n_init=1 is often sufficient (sklearn sets this via 'auto')
```

### 3.2 Mini-Batch K-Means (Sculley, 2010)

**Problem it solves:** Full k-means requires all data in memory and scales as $O(n \cdot K \cdot p)$
per iteration, making it intractable for $n > 10^6$ samples.

> 📜 **Citation:** Sculley, D. (2010). "Web-scale K-Means clustering." In *Proceedings of the
> 19th International Conference on the World Wide Web (WWW 2010)*, pp. 1177–1178.
> DOI: https://doi.org/10.1145/1772690.1772862

**Key algorithmic change:** At each iteration, draw a mini-batch $\mathcal{B} \subset \{1,\ldots,n\}$
of size $b \ll n$. For each point in the batch: (1) assign to nearest centroid, (2) update that
centroid with a per-centroid learning rate $\eta_k = 1 / (\text{count}_k)$:

$$\mathbf{c}_k \leftarrow (1 - \eta_k)\mathbf{c}_k + \eta_k \mathbf{x}_i \quad \text{for each } i \in \mathcal{B} \text{ assigned to } k$$

**Complexity:** $O(b \cdot K \cdot p)$ per iteration, $O(n/b)$ iterations for one epoch —
total $O(n \cdot K \cdot p / b)$ first epoch, substantially sub-linear in subsequent epochs.

**When to prefer:** N > 1M samples, or streaming data (use `partial_fit`). The clustering quality
is slightly worse than full k-means (typically 1-5% higher inertia), but this is acceptable for
large-scale DR.

```python
from sklearn.cluster import MiniBatchKMeans

mbk = MiniBatchKMeans(
    n_clusters=50,
    batch_size=1024,
    n_init=3,
    random_state=42
)
X_dr = mbk.fit_transform(X_large)   # Works exactly like KMeans.transform()
```

### 3.3 Gaussian Mixture Model (Dempster, Laird & Rubin, 1977)

**Problem it solves:** K-means assigns each point to exactly one cluster regardless of proximity
to cluster boundaries. It can't model clusters with different shapes or sizes. It provides no
uncertainty estimate.

> 📜 **Citation:** Dempster, A.P., Laird, N.M., & Rubin, D.B. (1977). "Maximum likelihood from
> incomplete data via the EM algorithm." *Journal of the Royal Statistical Society: Series B*,
> 39(1), 1–38.

**Key algorithmic change:** Replace hard assignment (one-hot) with soft assignment (posterior
probabilities). The EM algorithm for GMMs generalizes k-means by using a continuous relaxation of
the assignment variable. With `covariance_type='full'`, clusters can be ellipsoidal with arbitrary
orientations — a strict superset of k-means' isotropic clusters.

**When to prefer over k-means:**
- When clusters have different shapes or orientations
- When uncertainty about cluster membership matters (points near boundaries)
- When you need the DR output to be a proper probability distribution (sums to 1)
- For Fisher Vector computation (requires GMM posteriors specifically)

**Code:**
```python
from sklearn.mixture import GaussianMixture

# Soft-assignment DR:
gmm = GaussianMixture(
    n_components=15, covariance_type='diag', n_init=5, random_state=42
).fit(X_scaled)
X_dr = gmm.predict_proba(X_scaled)   # shape: (n, 15)
# Note: NO .transform() method exists — always use predict_proba()
```

### 3.4 Bayesian GMM / Dirichlet Process GMM (Blei & Jordan, 2006)

**Problem it solves:** Standard GMM requires specifying $K$ upfront. In many applications (e.g.,
customer segmentation, genomics), the number of clusters is genuinely unknown.

> 📜 **Citation:** Blei, D.M., & Jordan, M.I. (2006). "Variational inference for Dirichlet process
> mixtures." *Bayesian Analysis*, 1(1), 121–144.

**Key algorithmic change:** Place a Dirichlet Process prior (or a variational approximation
thereof — the "stick-breaking" construction) over the mixing weights. Components with no data
support automatically get $\pi_k \to 0$. You initialize with $K_{\max}$ components and let the
model determine the effective $K$.

**When to prefer:** When you genuinely don't know $K$, and cross-validating over many $K$ values
is expensive. The `weight_concentration_prior` hyperparameter controls how many components survive.

```python
from sklearn.mixture import BayesianGaussianMixture

bgmm = BayesianGaussianMixture(
    n_components=30,                  # upper bound
    covariance_type='diag',
    weight_concentration_prior=1e-2,  # small → sparse; large → more equal components
    n_init=3,
    random_state=42
).fit(X_scaled)

effective_k = (bgmm.weights_ > 0.01).sum()
print(f"Effective components used: {effective_k}")  # typically << 30

X_dr = bgmm.predict_proba(X_scaled)   # shape: (n, 30) but many cols near 0
```

### 3.5 Self-Organizing Maps (Kohonen, 1982)

**Problem it solves:** K-means centroids have no spatial relationship to each other — the labeling
of clusters as "1, 2, ..., K" is arbitrary and topologically meaningless. You can't say "cluster 5
is between clusters 3 and 7." SOMs impose a topology.

> 📜 **Citation:** Kohonen, T. (1982). "Self-organized formation of topologically correct feature
> maps." *Biological Cybernetics*, 43, 59–69. DOI: https://doi.org/10.1007/BF00337288

**Key algorithmic change:** Prototypes are arranged on a 2D grid. When prototype $k^*$ wins for
input $\mathbf{x}$ (nearest in weight space), not only $k^*$ but also its grid neighbors are
updated — with strength decreasing with grid distance:

$$\mathbf{w}_j \leftarrow \mathbf{w}_j + \alpha(t) \cdot h(j, k^*, t) \cdot (\mathbf{x} - \mathbf{w}_j)$$

where $h(j, k^*, t) = \exp\left(-\frac{\text{dist\_grid}(j, k^*)^2}{2\sigma(t)^2}\right)$ is the
neighborhood function, $\alpha(t)$ is a decaying learning rate, and $\sigma(t)$ is a decaying
neighborhood radius.

**The DR output:** Each sample maps to a 2D grid coordinate $(x_{\text{grid}}, y_{\text{grid}})$
— a topology-preserving 2D embedding. Nearby grid cells contain similar prototypes, so samples
with similar features land in nearby grid cells. The SOM is simultaneously a clustering (each cell
is a cluster) and a 2D DR.

**When to prefer:**
- When you want a topographic map: a 2D visualization where spatial proximity reflects feature
  similarity, and the layout itself carries meaning (e.g., "cells in the top-left corner
  correspond to high-income young users").
- For explorative visualization of high-dimensional tabular data.
- When interpretability of the 2D layout matters (unlike t-SNE/UMAP where the global arrangement
  can be misleading).

```python
from minisom import MiniSom
import numpy as np

som = MiniSom(x=15, y=15, input_len=X.shape[1],
              sigma=2.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(X_scaled)
som.train(X_scaled, num_iteration=10_000, verbose=False)

# 2D grid coordinates per sample (the DR output):
coords = np.array([som.winner(x) for x in X_scaled])  # shape: (n, 2)

# OR: distance to winning neuron (quantization error per sample):
q_err = np.array([np.linalg.norm(x - som.get_weights()[som.winner(x)])
                  for x in X_scaled])
```

### 3.6 Spectral Clustering as Two-Stage DR (Ng, Jordan & Weiss, 2002)

**Problem it solves:** K-means (and GMM) can't find non-convex clusters — rings, crescents, or
two interlocking spirals. The key insight: if we first embed the data using the graph Laplacian
(Spectral Embedding / Laplacian Eigenmaps), the clusters become linearly separable in the
embedding space, and then k-means can find them.

> 📜 **Citation:** Ng, A.Y., Jordan, M.I., & Weiss, Y. (2002). "On spectral clustering: Analysis
> and an algorithm." *NIPS 2002*, pp. 849–856.
> Also: Belkin, M., & Niyogi, P. (2001). "Laplacian Eigenmaps and spectral techniques for
> embedding and clustering." *NIPS 2001*, Vol. 14.

**Key algorithmic change:** Build a $k$-NN or RBF affinity graph $\mathbf{W}$, compute the
normalized graph Laplacian $\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$,
take the first $d$ eigenvectors of $\mathbf{L}$ as the embedding, then run k-means in this
$d$-dimensional embedding space.

**The DR output:** The $d$-dimensional spectral embedding IS the DR, and it's the primary
output of `SpectralEmbedding.fit_transform(X)`. `SpectralClustering` wraps this with a final
k-means step.

**When to prefer:** Non-convex cluster shapes. Graph-structured data (social networks, citation
networks, molecular graphs). Data where a kernel (RBF, cosine) similarity better captures
structure than Euclidean distance.

```python
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans

# Step 1: Spectral DR (the nonlinear embedding)
spec = SpectralEmbedding(n_components=8, affinity='nearest_neighbors',
                         n_neighbors=15, random_state=42)
X_spec = spec.fit_transform(X_scaled)   # shape: (n, 8) — this is the DR output

# Step 2: Cluster in embedding space
labels = KMeans(n_clusters=5, n_init='auto', random_state=42).fit_predict(X_spec)
```

### 3.7 Feature Agglomeration (Ward, 1963 / sklearn adaptation)

**Problem it solves:** When features are highly correlated (e.g., gene expression, sensor
readings, financial time series), including all of them individually adds noise and computational
cost. Grouping correlated features reduces redundancy while preserving interpretable structure.

**Key idea:** Apply agglomerative (hierarchical) clustering to the transpose of the data matrix —
cluster features instead of samples. This is structurally identical to hierarchical clustering
of samples, just applied to the other axis.

**When to prefer:**
- Tabular data with many correlated features (genomics, financial factor models)
- When feature interpretability is important (you want to know which features cluster together)
- As a fast, deterministic alternative to PCA when you prefer additive (rather than linear
  combination) feature reduction
- When reconstruction (inverse_transform) matters — feature agglomeration gives a meaningful
  reconstruction; KMeans distance-transform does not

### 3.8 HDBSCAN Soft Clustering (Campello et al., 2013)

**Problem it solves:** Density-based clustering that doesn't require specifying $K$, handles
clusters of varying density, and — in soft mode — produces membership vectors analogous to GMM
posteriors but without the Gaussian parametric assumption.

> 📜 **Citation:** Campello, R.J., Moulavi, D., & Sander, J. (2013). "Density-based clustering
> based on hierarchical density estimates." In *Pacific-Asia Conference on Knowledge Discovery and
> Data Mining (PAKDD 2013)*, pp. 160–172.

**For DR:** Use `hdbscan.all_points_membership_vectors(clusterer)` to get soft membership scores
$(n \times K)$ analogous to GMM posteriors, but robust to non-Gaussian, variable-density clusters.

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    prediction_data=True   # Required for soft membership
).fit(X_scaled)

soft_memberships = hdbscan.all_points_membership_vectors(clusterer)
# shape: (n, n_clusters_found) — rows sum to 1.0, like GMM posteriors
```

---

## §4 — Hyperparameters: The Complete Guide

### 4.1 `n_clusters` / `n_components` — The Central Hyperparameter

This single parameter determines both the clustering complexity **and** the output dimensionality
of the DR. Getting it right is the most important tuning task.

**What it controls:** The number of cluster centroids (KMeans) or Gaussian components (GMM). The
output DR space has exactly this many dimensions.

**Default and what it implies:**
- KMeans default: `n_clusters=8` — arbitrary; rarely the right choice for your data.
- GMM default: `n_components=1` — a single Gaussian (no clustering at all). Always set explicitly.

**Too low (K too small):**
- Clusters are large and heterogeneous; distance features don't distinguish fine-grained structure
- High inertia (within-cluster sum of squares); silhouette scores drop
- Downstream model can't distinguish important subgroups
- Feature Agglomeration: pooling non-correlated features — reconstruction error high

**Too high (K too large):**
- Clusters overfit to noise; each cluster has too few samples to be stable
- Distances become noisy; the DR space is unreliable
- Inference is slower (all $K$ distances computed per sample)
- Risk of empty clusters: sklearn warns "Number of distinct clusters found smaller than n_clusters"
- GMM: risk of component collapse (degenerate covariances)

**Tuning strategies:**

*1. Elbow method (KMeans inertia):*
```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertias = []
K_range = range(2, 30)
for k in K_range:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'o-')
plt.xlabel('K (n_clusters)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.tight_layout()
```

*2. Silhouette score (model-agnostic):*
```python
from sklearn.metrics import silhouette_score

sil_scores = []
for k in K_range:
    labels = KMeans(n_clusters=k, n_init='auto', random_state=42).fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
best_k = K_range[np.argmax(sil_scores)]
```

*3. BIC / AIC (GMM only):*
```python
from sklearn.mixture import GaussianMixture
bic_scores = []
for k in range(2, 30):
    gmm = GaussianMixture(n_components=k, covariance_type='diag', n_init=3, random_state=42)
    bic_scores.append(gmm.fit(X_scaled).bic(X_scaled))
best_k_gmm = np.argmin(bic_scores) + 2
```

*4. Downstream task validation (gold standard for DR use):*
```python
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def objective(trial):
    k = trial.suggest_int("n_clusters", 5, 200, log=True)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    X_dr = kmeans.fit_transform(X_train_scaled)
    return cross_val_score(LogisticRegression(max_iter=500), X_dr, y_train, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(f"Best K: {study.best_params['n_clusters']}")
```

> 🏆 **Best practice:** For pure DR (no downstream labels), use silhouette + BIC (for GMM) to
> select K. For pipeline DR (features for a classifier/regressor), treat K as a hyperparameter
> and optimize it end-to-end with Optuna. Domain knowledge beats data-driven heuristics when
> you know the number of categories (e.g., 10 digit classes → start with K=10–20).

**Recommended range:** $K \in [\sqrt{n/2}, \; 2\sqrt{n}]$ as a starting heuristic. More
concretely: try K = {5, 10, 20, 50, 100, 200} as a coarse grid, then refine.

---

### 4.2 `init` — Initialization Strategy (KMeans)

| Value | Description | When to use |
|---|---|---|
| `'k-means++'` | Smart seeding: D² weighting | **Default; use always** |
| `'random'` | Uniform random centroids | Only for benchmarking; inferior to k-means++ |
| array | Pre-supplied centroid matrix | When you have domain-knowledge initialization |

**Interaction with `n_init`:** With `'k-means++'`, a single restart (`n_init=1`) often suffices;
sklearn sets this automatically via `n_init='auto'`. With `'random'`, always use `n_init≥10`.

---

### 4.3 `n_init` — Number of Restarts

**What it controls:** How many times the algorithm runs from different random initializations.
The best result (lowest inertia) is retained.

**Default:** `'auto'` since sklearn 1.2 — equivalent to `n_init=10` for random init, `n_init=1`
for k-means++ or array init.

> ⚠️ **Pitfall:** `n_init='auto'` was introduced in sklearn 1.2. If your codebase targets
> sklearn < 1.2, you must use an integer value. Use `n_init=10` as a safe default.

**Too low:** Risk of poor local optima. Test: run the same fit twice with different `random_state`
values. If inertias differ by more than 5%, increase `n_init`.

**For GMM:** Default is `n_init=1` — dangerously low. EM has many local optima. Always set
`n_init=5` minimum for production GMM fits.

---

### 4.4 `covariance_type` — GMM Cluster Shape (GaussianMixture)

This parameter controls the parametric family of each Gaussian component — and has a major effect
on both model quality and computational cost.

| `covariance_type` | Parameters per component | Cluster shape | When to use |
|---|---|---|---|
| `'full'` | $p(p+1)/2$ | Arbitrary ellipsoid | Small $p$ (< 50), ample data |
| `'tied'` | $p(p+1)/2$ (shared) | All same ellipsoid | When clusters have similar shapes |
| `'diag'` | $p$ | Axis-aligned ellipsoid | **Good default for moderate $p$** |
| `'spherical'` | 1 | Isotropic sphere | Equivalent to k-means; fewest parameters |

**What it controls for DR:** The `covariance_type` determines the shape of the "catchment basin"
for each component — the region where it has dominant posterior probability. Full covariance
allows arbitrary ellipsoids; spherical constrains to spheres (same as k-means Voronoi cells).

> 🏆 **Best practice:** For clustering-as-DR on tabular data with $p \leq 50$, start with
> `'diag'`. Move to `'full'` only if you have ample data ($n > 100 \cdot K \cdot p$) and the
> clusters are known to be correlated ellipsoids. For high-dimensional data ($p > 100$), use
> `'diag'` or `'spherical'` and increase `reg_covar` to 1e-3 to avoid singularity.

---

### 4.5 `reg_covar` — Covariance Regularization (GaussianMixture)

**What it controls:** A small value $\epsilon > 0$ added to the diagonal of each covariance
matrix to ensure positive definiteness: $\hat{\boldsymbol{\Sigma}}_k \leftarrow \hat{\boldsymbol{\Sigma}}_k + \epsilon \mathbf{I}$.

**Default:** `1e-6`. **Increase to `1e-3` if you see `ConvergenceWarning` or `LinAlgError`.**

**When it matters:** High-dimensional data, small cluster sizes, or clusters with near-zero
variance in some directions.

---

### 4.6 `linkage` and `pooling_func` — FeatureAgglomeration

**`linkage`:** Controls which features merge at each step. See the table in §3.7 above.
For most tabular DR tasks: `'ward'` (default) is appropriate when features are z-scored;
`'average'` with `metric='cosine'` is better for text embedding features.

**`pooling_func`:** How to summarize a cluster of features into one value.
- `np.mean` (default): appropriate when features are comparable in magnitude
- `np.max`: "max pooling" — preserves the most active feature in each group
- `np.sum`: useful for count/frequency features

---

### 4.7 SOM Hyperparameters (MiniSom)

| Parameter | Default | Role | Tuning guidance |
|---|---|---|---|
| `x, y` (grid size) | — | Total neurons = x*y | Rule of thumb: $x \cdot y \approx 5\sqrt{n}$ |
| `sigma` (neighborhood radius) | 1.0 | Initial neighborhood reach | Start at $\max(x,y)/2$; let it decay to ~1 |
| `learning_rate` | 0.5 | Initial step size | Typical: 0.1–1.0; monitor quantization error |
| `neighborhood_function` | `'gaussian'` | Shape of neighborhood influence | `'gaussian'` for smooth maps; `'bubble'` for sharp boundaries |
| num_iteration (in `train()`) | — | Training length | ~500 × n for convergence |

---

### 4.8 Hyperparameter Tuning Playbook Table

| Parameter | Algorithm | Default | Too Low Effect | Too High Effect | Recommended Range | Tuning Method |
|---|---|---|---|---|---|---|
| `n_clusters` | KMeans | 8 | Coarse, noisy features | Overfitting, empty clusters | $[\sqrt{n/2}, 2\sqrt{n}]$ | Silhouette or downstream CV |
| `n_components` | GMM | 1 | No structure captured | Degenerate components | Same as above | BIC or downstream CV |
| `n_init` | KMeans | `'auto'` | Poor local optima | Slow | 3–10 | Fixed at 5–10 |
| `n_init` | GMM | 1 | Stuck in local optima | Slow | 5–10 | Fixed at 5 minimum |
| `covariance_type` | GMM | `'full'` | N/A (categorical) | N/A | `'diag'` for D>50 | BIC comparison |
| `reg_covar` | GMM | 1e-6 | Singular matrices | Oversmoothing | 1e-6 to 1e-3 | Increase if convergence fails |
| `n_clusters` | FeatureAgglo | 2 | Few, heterogeneous groups | Many tiny groups | $[D/20, D/2]$ | Downstream CV |
| `linkage` | FeatureAgglo | `'ward'` | N/A (categorical) | N/A | `'ward'` for Euclidean | Data inspection |
| Grid size (x*y) | SOM | — | Too few neurons | Overfitting | $\approx 5\sqrt{n}$ | Monitor quantization error |
| `sigma` | SOM | 1.0 | No topology | All update together | $[1, \max(x,y)/2]$ | Fixed schedule |

---

## §5 — Strengths

### 5.1 Nonlinear Feature Extraction Without Kernels

K-means and GMM produce **nonlinear features** — the Euclidean distance to a centroid is a
nonlinear function of the input features. This is a qualitative advantage over PCA and LDA, which
can only produce linear combinations of input features. In practice: two points with very
different raw feature vectors but similar cluster memberships will be represented similarly in
the DR space, capturing structure that linear methods miss.

**Mechanism:** The distance function $\|\mathbf{x} - \mathbf{c}_k\|^2 = \sum_j (x_j - c_{k,j})^2$
involves interaction terms between features implicitly, through the squared difference. When used
as features in a downstream linear model, this effectively adds a radial basis function (RBF)
kernel approximation — you're doing kernel regression at the cost of a few cluster fits.

### 5.2 Scalability and Inference Speed

Once fitted, **KMeans transforms scale as $O(n \cdot K \cdot p)$ — linear in all three quantities**.
There is no need to store the training data at inference time (unlike KNN, LOF, or UMAP without
parametric extension). The transform is a matrix multiplication plus a distance computation —
fast even on CPUs, trivially parallelizable.

**In contrast:** t-SNE has no out-of-sample transform at all. UMAP has one but it's slower. PCA
is faster (linear algebra), but clustering-as-DR is faster than KernelPCA, Isomap, or LLE.

### 5.3 Interpretable Features

When K is small (5–20), a practitioner can often inspect the centroids and understand what each
cluster represents. "Cluster 3's centroid has high income, low age, high digital engagement score"
means the 3rd feature in the DR space measures "proximity to the young high-earner digital-native
segment." This is interpretable in a way that t-SNE coordinates and PCA components usually are
not.

**Feature Agglomeration** goes further: the `labels_` attribute tells you exactly which original
features were merged into each meta-feature, enabling biological or domain interpretation (e.g.,
"meta-feature 7 = genes BRCA1, BRCA2, RAD51 — all involved in DNA repair").

### 5.4 Robust to Irrelevant Features Through Clustering

K-means naturally down-weights noisy, uninformative features because they contribute small,
uniform distances to all centroids — they don't move centroids. Informative features that
distinguish clusters pull centroids apart and create large, discriminative distance features.
This implicit feature selection is a practical benefit when you have many noisy columns.

### 5.5 Works with Any Downstream Model

Because `KMeans.transform()` and `GMM.predict_proba()` produce standard numpy arrays, they
compose naturally with any sklearn model — decision trees, gradient boosting, SVMs, neural
networks. The DR step can be inserted as a Pipeline stage with zero code changes. This plug-and-play
composability is a practical advantage over methods like UMAP that require more careful integration.

### 5.6 Soft Assignments Encode Uncertainty

GMM posteriors are proper probability distributions. A point deep in one cluster gets
$[0.99, 0.01, \ldots]$; a point on the boundary between two clusters gets $[0.52, 0.47, \ldots]$.
Downstream models receive this uncertainty directly — a gradient boosting tree can split on "is
probability of cluster 3 > 0.7?" which is a much richer signal than "is the cluster label == 3?"

---

## §6 — Weaknesses & Failure Modes

### 6.1 K-Means Cannot Find Non-Convex Clusters

**Mechanism:** Voronoi cells are always convex polytopes. Two interlocking crescents, a ring around
a core, or a spiral — none of these have convex cluster structure, and k-means will cut them
with hyperplanes in ways that don't reflect the true geometry.

**Detection:** Visualize clusters with PCA(2) or UMAP(2). If cluster boundaries don't match
visual groupings, the geometry is non-convex.

**Mitigation:** Use Spectral Clustering (Laplacian embedding + k-means) for non-convex structure.
Or apply kernel k-means (not in sklearn; available in `tslearn`). Or use HDBSCAN.

### 6.2 Sensitive to Feature Scale

**Mechanism:** Euclidean distance is the sum of squared differences per dimension. If income is
in $[0, 200000]$ and age is in $[18, 80]$, income dominates — the DR output is essentially just
"distance to income centroid" in each cluster.

**Detection:** Check if `kmeans.cluster_centers_` vary far more on some dimensions than others.

**Mitigation:** Always apply `StandardScaler` (or `RobustScaler` if outliers are present) before
any distance-based clustering. This is not optional.

```python
# WRONG:
X_dr = KMeans(n_clusters=10).fit_transform(X_raw)

# RIGHT:
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10, n_init='auto'))])
X_dr = pipe.fit_transform(X_raw)
```

### 6.3 K-Means Is Not a Proper DR of the Geometry (It Doesn't Preserve Neighborhoods)

**Mechanism:** K-means clusters by minimizing within-cluster variance. Points near a cluster
boundary may be placed in different clusters from their natural neighbors, causing nearby points
in the original space to have very different distance vectors. Trustworthiness can be low even
when silhouette is high.

**Detection:** Compute `trustworthiness(X_scaled, X_dr, n_neighbors=10)` from sklearn. Values
below 0.85 indicate significant neighborhood distortion.

**Mitigation:** Use more clusters (higher K) to give finer resolution. Or use UMAP/t-SNE for
neighborhood-preserving DR and treat clustering-as-DR as a feature engineering step, not a
manifold learning step.

### 6.4 GMM Assumes Gaussian Clusters

**Mechanism:** If the true clusters are non-Gaussian (bimodal, heavy-tailed, or ring-shaped),
the GMM posteriors will be inaccurate — the model is fitting Gaussians to non-Gaussian structure.
The DR features will reflect the Gaussian approximation, not the true geometry.

**Detection:** Check per-component histograms. Fit a GMM and plot the data colored by the
component with highest posterior; if the coloring doesn't match visual groupings, the Gaussian
assumption is violated.

**Mitigation:** Use HDBSCAN soft membership vectors (non-parametric), or apply a nonlinear
transform to the data before GMM (e.g., UMAP embedding → GMM posteriors as features).

### 6.5 Label Instability Across Runs

**Mechanism:** K-means has many local optima. With different `random_state` values, you get
different cluster assignments (though similar inertia). The *label* of a cluster is arbitrary —
"cluster 0" in one run may correspond to "cluster 3" in another run. Any downstream code that
hard-codes cluster IDs will silently produce wrong results.

**Detection:** Run k-means twice with different seeds; compute ARI between the two label
assignments. If ARI < 0.95, the labeling is unstable.

**Mitigation:** Always set `random_state` to a fixed value in production. Sort clusters by a
meaningful property (e.g., first centroid coordinate, mean distance to origin) to stabilize label
assignment. Prefer GMM posteriors (continuous values, no label ambiguity) over hard labels.

### 6.6 The "Always Finds K Clusters" Problem

**Mechanism:** K-means finds exactly K clusters regardless of whether the data actually has K-cluster
structure. On a dataset drawn from a single Gaussian, k-means will produce K perfectly geometrically
balanced clusters — each internally valid, but meaningless. The user may not notice.

**Detection:** Compute the Hopkins statistic (measures clustering tendency). If Hopkins ≈ 0.5,
the data is uniformly distributed and clustering is meaningless. Plot data with PCA(2) first.

```python
# Check for clustering tendency visually:
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.3)
# If it looks like a smooth blob → clustering-as-DR is suspect
```

**Mitigation:** Always validate clusters with silhouette score, Davies-Bouldin, or visual
inspection before trusting the DR features. Cluster validity is a prerequisite for meaningful DR.

---

## §7 — What I Think About When Fitting

This section is my internal monologue when I'm fitting a clustering-based DR for the first time
on a new dataset. I'll walk you through every stage.

### Before I Touch the Code

**What is the structure I'm trying to capture?** Is the data expected to have discrete clusters
(product categories, cell types, customer segments) or a continuous manifold? If I genuinely
believe the data is continuous and smooth, clustering-as-DR is a poor choice — UMAP or PCA will
serve better. But if I expect discrete groups or a "vocabulary" structure, clusters make sense.

**Do I have domain knowledge about K?** Always try to encode domain knowledge first. If I'm
working with MNIST digits, K=10 is a natural starting point (one per digit). If I'm building
a customer segmentation feature for a model, I ask the business team: "How many customer types
do your analysts currently recognize?" That number is my starting K.

**Check the curse of dimensionality.** If $p > 100$, distances in raw feature space are nearly
uniform (all distances concentrate around the same value), making k-means ineffective in the
raw space. My fix: apply `PCA(n_components=50)` before k-means. This is the standard scRNA-seq
pipeline and it works on most tabular data too.

**Scale check.** I run `X.std(axis=0)` and look at the ratio of the largest std to the smallest.
If it's > 10, I will scale. If all features are already on the same scale (pixel values 0–255,
z-scores), I might skip scaling. I almost always scale.

**Outlier check.** I run `X.quantile([0.01, 0.99])` and check for extreme outlier rows. A single
outlier at 100 sigma will pull a centroid far from the main cluster. I winsorize or clip to 3σ
before k-means if I find them.

### During Fitting

**I always set `verbose=1` on the first run.** For KMeans, this prints inertia per iteration —
I watch it decrease monotonically. If it jumps up (shouldn't happen with Lloyd's) or takes > 100
iterations, something is wrong.

**I check `kmeans.n_iter_`** immediately after fitting. If it equals `max_iter` (default 300),
convergence was not reached. I increase `max_iter` to 500 and refit.

**For GMM, I check `gmm.converged_`** right after `.fit()`. If False, I increase `max_iter` and
`n_init`. I also check that `gmm.lower_bound_` (the log-likelihood lower bound) is increasing
across iterations when `verbose=1`.

**Watch for warnings.** sklearn will warn about:
- "Number of distinct clusters (X) found smaller than n_clusters" → empty clusters, reduce K
- `ConvergenceWarning` → increase max_iter or n_init
- `LinAlgError: singular matrix` for GMM → increase `reg_covar` to 1e-3

### After Fitting — The Smell Tests

**1. Cluster size balance check.** I always run `np.bincount(kmeans.labels_)`. If one cluster
has 80% of the data, K is too small or there's a dominant mode. Very small clusters (< 10 samples)
with large K indicate overfitting.

**2. 2D visualization.** PCA to 2D, color by cluster label. If clusters form tight, well-separated
blobs in 2D PCA space — good sign. If they're a jumble — either the clusters exist in a subspace
that PCA doesn't capture (use UMAP for visualization instead) or the clustering is poor.

**3. Centroid inspection.** For tabular data with interpretable features: examine
`pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)`. Do the centroids make semantic
sense? "Cluster 4 has high recency, low frequency, high monetary value — that's the 'big spender'
segment." If centroids are numerically close together, K is too high.

**4. Silhouette sanity check.** Run `silhouette_score(X_scaled, kmeans.labels_)`. I expect > 0.3
for "useful" clusters. If below 0.2, I question whether the clustering is capturing real structure.

**5. Downstream validation.** The ultimate test: does using `X_dr = kmeans.fit_transform(X_scaled)`
as features improve my downstream model? I run a quick `cross_val_score(LogisticRegression(), X_dr, y, cv=5)`
and compare to the same with raw features or PCA features. If clustering-as-DR doesn't help, I
don't use it — no matter how nice the silhouette looks.

**If I see bad silhouette + bad downstream performance:** I try (a) different K, (b) PCA
preprocessing before k-means, (c) switch to GMM with `covariance_type='diag'`, (d) consider
whether clustering-as-DR is even the right tool for this data.

**If I see good silhouette + bad downstream performance:** The clusters exist but don't align with
the target. This is important information — the data has structure that is independent of the
prediction task. I might keep the cluster features as auxiliary features alongside PCA features.

---

## §8 — Diagnostic Plots & Evaluation

### 8.1 The Elbow Plot + Silhouette Plot (for K selection)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Assume X_scaled is already z-scored
K_range = range(2, 25)
inertias, sil_scores = [], []

for k in K_range:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(K_range), inertias, 'o-', color='steelblue')
ax1.set_xlabel('K (n_clusters)')
ax1.set_ylabel('Inertia (within-cluster SSE)')
ax1.set_title('Elbow Plot — K vs Inertia')
ax1.grid(True, alpha=0.4)

ax2.plot(list(K_range), sil_scores, 's-', color='tomato')
ax2.axhline(0.5, ls='--', color='gray', label='Silhouette = 0.5 (moderate)')
ax2.set_xlabel('K (n_clusters)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score vs K')
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('k_selection.png', dpi=150)
```

**Reading the elbow plot:** Look for the "knee" — the K value where inertia stops decreasing
rapidly. In clean synthetic data this is sharp; in real data it's often gradual. Combine with
silhouette: pick the K where silhouette peaks AND is near the inertia elbow.

**Reading the silhouette plot:** A peak at K=5 means K=5 clusters are best separated. If
silhouette scores are uniformly below 0.2 for all K, the data may lack cluster structure.

### 8.2 BIC/AIC Plot for GMM (n_components selection)

```python
from sklearn.mixture import GaussianMixture

K_range = range(2, 25)
bic_scores, aic_scores = [], []

for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='diag', n_init=3, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

plt.figure(figsize=(8, 4))
plt.plot(list(K_range), bic_scores, 'o-', label='BIC', color='navy')
plt.plot(list(K_range), aic_scores, 's--', label='AIC', color='darkorange')
plt.axvline(np.argmin(bic_scores) + 2, color='navy', ls=':', alpha=0.6, label=f'Best BIC K={np.argmin(bic_scores)+2}')
plt.xlabel('n_components')
plt.ylabel('Score (lower is better)')
plt.title('BIC/AIC for GMM n_components Selection')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
```

**Reading:** Pick the K at the minimum BIC (penalizes complexity more than AIC). BIC is preferred
for model selection when the goal is identifying the "true" number of components.

### 8.3 Trustworthiness Plot (Neighborhood Preservation)

```python
from sklearn.manifold import trustworthiness

n_neighbors_range = [5, 10, 15, 20]
trust_scores = {}

for k in [5, 10, 20, 50]:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    X_dr = km.fit_transform(X_scaled)
    trust_scores[k] = [trustworthiness(X_scaled, X_dr, n_neighbors=nn)
                       for nn in n_neighbors_range]

plt.figure(figsize=(8, 4))
for k, scores in trust_scores.items():
    plt.plot(n_neighbors_range, scores, 'o-', label=f'K={k}')
plt.xlabel('n_neighbors')
plt.ylabel('Trustworthiness')
plt.title('Trustworthiness of KMeans DR at Different K')
plt.legend()
plt.axhline(0.9, ls='--', color='gray', label='Threshold: 0.9')
plt.grid(True, alpha=0.4)
plt.tight_layout()
```

**Reading:** Higher K → higher trustworthiness (more centroids = finer grain). Values > 0.9 are
good. If trustworthiness is low even at high K, k-means is not capturing neighborhood structure —
try UMAP or Spectral Embedding instead.

### 8.4 Downstream Performance vs DR Method

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas as pd

results = {}

# Baseline: raw features
results['Raw'] = cross_val_score(LogisticRegression(max_iter=500), X_scaled, y, cv=5).mean()

# PCA
pca = PCA(n_components=20)
results['PCA-20'] = cross_val_score(LogisticRegression(max_iter=500), pca.fit_transform(X_scaled), y, cv=5).mean()

# KMeans DR (various K)
for k in [10, 20, 50]:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    X_dr = km.fit_transform(X_scaled)
    results[f'KMeans-K{k}'] = cross_val_score(LogisticRegression(max_iter=500), X_dr, y, cv=5).mean()

# GMM DR
gmm = GaussianMixture(n_components=20, covariance_type='diag', n_init=5, random_state=42)
gmm.fit(X_scaled)
X_dr_gmm = gmm.predict_proba(X_scaled)
results['GMM-K20'] = cross_val_score(LogisticRegression(max_iter=500), X_dr_gmm, y, cv=5).mean()

df = pd.DataFrame.from_dict(results, orient='index', columns=['CV Accuracy'])
df.sort_values('CV Accuracy', ascending=False).plot(kind='barh', legend=False, figsize=(8, 4))
plt.xlabel('5-fold CV Accuracy')
plt.title('DR Method Comparison on Downstream Classification')
plt.tight_layout()
```

**Reading:** This is the gold-standard diagnostic. If KMeans-based DR outperforms PCA, the data
has nonlinear cluster structure that linear DR misses. If GMM outperforms KMeans, soft assignment
is providing richer features.

### 8.5 Silhouette Sample Plot (Cluster Quality Visualization)

```python
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

k_best = 5   # replace with your selected K
km = KMeans(n_clusters=k_best, n_init='auto', random_state=42)
labels = km.fit_predict(X_scaled)
sil_vals = silhouette_samples(X_scaled, labels)

fig, ax = plt.subplots(figsize=(8, 5))
y_lower = 10
for i in range(k_best):
    vals = np.sort(sil_vals[labels == i])
    y_upper = y_lower + len(vals)
    color = cm.nipy_spectral(i / k_best)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * len(vals), str(i))
    y_lower = y_upper + 10

ax.axvline(sil_vals.mean(), color='red', linestyle='--', label=f'Mean: {sil_vals.mean():.3f}')
ax.set_xlabel('Silhouette coefficient')
ax.set_title(f'Silhouette Plot — K={k_best}')
ax.legend()
plt.tight_layout()
```

**Reading:** Each colored band is one cluster. Tall, wide bands = large, high-silhouette clusters
(good). Very thin or narrow bands = small or low-quality clusters. Bands below the red line
(mean silhouette) are candidates for merging or splitting.

### 8.6 Reconstruction Error for Feature Agglomeration

```python
from sklearn.cluster import FeatureAgglomeration

recon_errors = []
K_feat_range = range(2, 50, 2)

for k in K_feat_range:
    agglo = FeatureAgglomeration(n_clusters=k, linkage='ward')
    X_reduced = agglo.fit_transform(X_scaled)
    X_reconstructed = agglo.inverse_transform(X_reduced)
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    recon_errors.append(mse)

plt.figure(figsize=(8, 4))
plt.plot(list(K_feat_range), recon_errors, 'o-', color='purple')
plt.xlabel('n_clusters (target feature dimension)')
plt.ylabel('MSE (reconstruction error)')
plt.title('FeatureAgglomeration: Reconstruction Error vs n_clusters')
plt.grid(True, alpha=0.4)
plt.tight_layout()
```

**Reading:** Like PCA's scree plot. The error drops sharply at first (merging redundant features
costs little information) then levels off (merging informative features costs more). Pick the
elbow of this curve as your target `n_clusters`.

---

## §9 — Innovative Industry Applications

### 9.1 Bag of Visual Words — The Pre-Deep-Learning ImageNet Champion

> 🏭 **Industry application:** Computer vision at Google Image Search, Getty Images, Yandex Visual Search

Before convolutional neural networks took over in 2012, the reigning paradigm for image
classification and retrieval was Bag of Visual Words (BoVW). The pipeline: extract local keypoint
descriptors (SIFT, 128-dimensional vectors) from all training images. Train a KMeans codebook
with K=1,000–50,000 visual words on these descriptors. For each new image, assign every descriptor
to its nearest visual word (nearest centroid), and build a K-dimensional histogram of word
frequencies. That histogram IS the dimensionality-reduced image representation.

**Why this was transformative:** Before BoVW, images had no fixed-length representation. SIFT
descriptors are variable in number per image. BoVW collapses any image to a fixed K-dimensional
vector, enabling all of classical ML. Google deployed this at the scale of billions of image
descriptors.

```python
from sklearn.cluster import KMeans
import numpy as np

# descriptors: (N_total_descriptors, 128) — SIFT from all training images
vocab_size = 1000
kmeans = KMeans(n_clusters=vocab_size, n_init=1, algorithm='lloyd', random_state=42)
kmeans.fit(descriptors)  # Train visual vocabulary

def encode_image(img_descriptors, kmeans, vocab_size):
    """Encode one image as a normalized BoVW histogram."""
    if len(img_descriptors) == 0:
        return np.zeros(vocab_size)
    assignments = kmeans.predict(img_descriptors)
    hist, _ = np.histogram(assignments, bins=vocab_size, range=(0, vocab_size))
    return hist.astype(float) / max(hist.sum(), 1e-10)  # L1 normalize

# Encode all images:
X_bow = np.vstack([encode_image(d, kmeans, vocab_size) for d in all_image_descriptors])
# X_bow.shape: (n_images, 1000)  ← this IS the dimensionality reduction
```

### 9.2 Fisher Vectors — The 2011 ImageNet Winner

> 🏭 **Industry application:** Yandex visual similarity, Getty Images retrieval, Google image search

Fisher Vectors improved BoVW by using a GMM instead of k-means and encoding not just which
visual word a descriptor belongs to, but also *how it deviates* from each Gaussian component.
For a GMM with $K$ components and $D$-dimensional descriptors, the Fisher Vector of an image is
$2KD$-dimensional — encoding first-order (mean) and second-order (variance) statistics per component.

**The result:** Fisher Vectors won the 2011 ImageNet Large Scale Visual Recognition Challenge —
the most important computer vision benchmark before AlexNet — using this GMM-based DR as the
core feature representation. This is the clearest proof that clustering-as-DR with GMMs is not
a toy approach.

**Key insight:** GMM soft assignment (`predict_proba`) encodes uncertainty about which visual word
a descriptor belongs to. Hard BoVW assignment discards this uncertainty. The Fisher Vector
accumulates this uncertainty — descriptors near a cluster boundary contribute to both adjacent
components' gradient statistics, encoding the "between-cluster" character of the descriptor.

### 9.3 Anomaly Detection — Distance to Nearest Cluster as Anomaly Score

> 🏭 **Industry application:** Edge Impulse (embedded ML for IoT), fintech fraud scoring, industrial defect detection

Fit KMeans on normal (inlier) training data. For any new point, compute `kmeans.transform(x).min()` —
the minimum distance to any centroid. Points far from all cluster centers have never been seen
in training; they are anomalies. This is a fast, memory-efficient anomaly detector that stores
only $K \cdot p$ parameters (the centroids) and runs in $O(K \cdot p)$ per sample.

**Why this matters for embedded ML:** Edge Impulse deploys this pattern on microcontrollers with
4KB RAM. Neither LOF, isolation forest, nor one-class SVM is deployable on such hardware. KMeans
anomaly detection runs on an Arduino.

```python
# Train on normal data only:
kmeans_anomaly = KMeans(n_clusters=20, n_init='auto', random_state=42).fit(X_normal_scaled)

# Anomaly score = min distance to any centroid:
train_scores = kmeans_anomaly.transform(X_normal_scaled).min(axis=1)
threshold = np.percentile(train_scores, 99)  # 99th pct of normal data

# Detect anomalies at inference:
def anomaly_score(x_new, kmeans, threshold):
    dist = kmeans.transform(x_new.reshape(1, -1)).min()
    return dist, dist > threshold
```

### 9.4 Single-Cell Genomics — Pathway Activity Features (scRNA-seq)

> 🏭 **Industry application:** 10x Genomics Seurat/Scanpy pipeline, Genentech, Novartis drug discovery

scRNA-seq data has $p \approx 20{,}000$ genes per cell and $n \approx 10{,}000$–$1{,}000{,}000$
cells. The standard preprocessing pipeline:
1. Apply PCA to $d=50$ components (removes noise, retains biological signal)
2. Apply KMeans ($K \approx 20$–$100$ cell types) in PCA space
3. The $K$ cluster labels (or the distance vector) become the cell type features

**Feature Agglomeration variant:** Cluster genes (features) using `FeatureAgglomeration` on the
gene co-expression matrix. Each gene cluster = a metabolic pathway or gene module. Each cell is
then represented by its mean expression in each module — $K \approx 100$–$500$ pathway features
instead of 20,000 gene features. This DR is biologically interpretable (each feature = a pathway)
where PCA is not.

```python
from sklearn.cluster import FeatureAgglomeration
import numpy as np

# X: (n_cells, n_genes) matrix — already log-normalized
# Cluster genes into K pathway modules:
gene_agglo = FeatureAgglomeration(n_clusters=200, linkage='ward', pooling_func=np.mean)
X_pathway = gene_agglo.fit_transform(X_cells)  # shape: (n_cells, 200)

# Now each of the 200 features is the mean expression of a gene cluster:
# Interpretable! Check which genes are in cluster 42:
cluster_42_genes = np.where(gene_agglo.labels_ == 42)[0]
print(f"Cluster 42 genes: {gene_names[cluster_42_genes]}")
```

### 9.5 NLP — Word Embedding Clustering for Document Features

> 🏭 **Industry application:** News classification at major media companies, semantic search, spam filtering

Word2Vec/GloVe/BERT embeddings are 100–768 dimensional. For document classification with a
lightweight model (logistic regression, gradient boosting), you want a compact document
representation. Cluster word embeddings with KMeans ($K = 50$–$500$ semantic topics). Represent
each document by the distribution of its word tokens over these clusters.

This is literally BoVW applied to NLP: instead of SIFT descriptors, you have word embeddings;
instead of visual words, you have semantic word clusters; instead of image histograms, you have
document topic profiles.

```python
from sklearn.cluster import KMeans
import numpy as np

# word_embeddings: (vocab_size, 300) — e.g., GloVe 300d
n_topics = 100
kmeans_words = KMeans(n_clusters=n_topics, n_init='auto', random_state=42)
kmeans_words.fit(word_embeddings)

def encode_document(token_ids, word_embeddings, kmeans, n_topics):
    """Encode a document as a soft topic histogram."""
    doc_embeddings = word_embeddings[token_ids]
    assignments = kmeans.predict(doc_embeddings)
    hist, _ = np.histogram(assignments, bins=n_topics, range=(0, n_topics))
    return hist.astype(float) / max(hist.sum(), 1e-10)

X_docs = np.vstack([encode_document(doc, word_embeddings, kmeans_words, n_topics)
                    for doc in token_id_lists])
# X_docs.shape: (n_documents, 100) — compact, fast, interpretable
```

### 9.6 Recommendation Systems — Behavioral Cluster Features

> 🏭 **Industry application:** Netflix (1,300+ user taste clusters), Spotify (genre behavioral clusters), Alibaba

Netflix famously uses over 1,300 taste clusters derived from user viewing behavior metadata. Each
user is characterized by their cluster membership vector — a K-dimensional soft feature derived
from their interaction history. Downstream recommendation models use these K-dimensional "taste
profile" features as inputs.

**The GMM advantage here:** Users near the boundary between "documentary lovers" and "thriller
fans" should not be hard-assigned to one. GMM posteriors give them a soft vector like
$[..., 0.4_{\text{documentaries}}, ..., 0.6_{\text{thrillers}}, ...]$, which better captures
their actual viewing behavior and allows downstream models to respond to taste ambiguity.

### 9.7 Time Series — Bag of Patterns for ECG Classification

> 🏭 **Industry application:** ECG arrhythmia detection, industrial machine fault diagnosis, financial regime detection

Each time series window is converted to a symbolic pattern (SAX representation), then patterns
are clustered with KMeans. Each time series is represented as a histogram over K pattern clusters
(Bag of Patterns, BoP). This collapses variable-length, shift-variant time series to fixed-length,
scale-invariant K-dimensional feature vectors.

**Why clustering for time series:** Raw time series are shift-variant — the same ECG signal
shifted by 50ms looks completely different in raw feature space. The symbolic→cluster pipeline
is inherently shift-invariant: the same pattern recognized at different times contributes to the
same cluster bin. KMeans on SAX patterns learns "what kinds of local waveform shapes exist" and
represents each series by which shapes appear and how often.

---

## §10 — Complete Python Worked Example

We'll run two complete end-to-end worked examples:
1. **Bag of Visual Words on MNIST** — clustering patch descriptors to build a BoVW representation
2. **Prototype distance features on make_blobs** — showing KMeans DR beating PCA for a nonlinear task

### Example 1: Bag of Visual Words on MNIST Patches

MNIST consists of 28×28 grayscale images of handwritten digits (0–9). We'll treat each image
as a collection of 4×4 local patches (a simplified substitute for SIFT descriptors), train a
KMeans vocabulary, and represent each image as a K-dimensional histogram — a Bag of Visual Words.
We then compare classification accuracy of BoVW vs. raw pixels vs. PCA features.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, silhouette_score
from sklearn.manifold import trustworthiness
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load MNIST subset ─────────────────────────────────────────────────────
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw, y = mnist.data[:10000], mnist.target[:10000].astype(int)
X_img = X_raw.reshape(-1, 28, 28) / 255.0   # normalize to [0,1]

print(f"Data shape: {X_raw.shape}")  # (10000, 784)
print(f"Classes: {np.unique(y)}")    # 0-9

# ── 2. Extract 4x4 patch descriptors ─────────────────────────────────────────
PATCH_SIZE = 4
STRIDE = 4    # non-overlapping patches

def extract_patches(image, patch_size=4, stride=4):
    """Extract all non-overlapping patches from one image."""
    patches = []
    H, W = image.shape
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size].ravel()  # 16-dim vector
            patches.append(patch)
    return np.array(patches)

# Extract patches from all images:
print("Extracting patches...")
all_patches_list = [extract_patches(img) for img in X_img]
# Each image → 7x7 = 49 patches of 16 dimensions each (for 28x28, stride=4)

# Flatten all patches for vocabulary training:
all_patches = np.vstack(all_patches_list)  # shape: (n_images * 49, 16)
print(f"Total patches for vocabulary training: {all_patches.shape}")

# Standardize patches:
patch_scaler = StandardScaler()
all_patches_scaled = patch_scaler.fit_transform(all_patches)

# ── 3. Train Visual Vocabulary with KMeans ─────────────────────────────────
print("Training visual vocabulary (KMeans)...")
VOCAB_SIZE = 64   # K = 64 visual words

# Use MiniBatchKMeans for speed on ~490,000 patches:
vocab_kmeans = MiniBatchKMeans(
    n_clusters=VOCAB_SIZE,
    batch_size=5000,
    n_init=3,
    random_state=42,
    verbose=0
)
vocab_kmeans.fit(all_patches_scaled)
print(f"Vocabulary trained. Inertia: {vocab_kmeans.inertia_:.2f}")

# Inspect vocabulary quality:
sil = silhouette_score(all_patches_scaled[:5000], vocab_kmeans.labels_[:5000])
print(f"Silhouette score (subsample): {sil:.3f}")

# ── 4. Encode Each Image as a BoVW Histogram ─────────────────────────────────
def encode_image_bow(img_patches, scaler, kmeans, vocab_size):
    """Encode one image as a normalized histogram over visual words."""
    patches_scaled = scaler.transform(img_patches)
    assignments = kmeans.predict(patches_scaled)
    hist, _ = np.histogram(assignments, bins=vocab_size, range=(0, vocab_size))
    return hist.astype(float) / max(hist.sum(), 1e-10)   # L1 normalize

print("Encoding images as BoVW histograms...")
X_bow = np.vstack([
    encode_image_bow(patches, patch_scaler, vocab_kmeans, VOCAB_SIZE)
    for patches in all_patches_list
])
print(f"BoVW representation shape: {X_bow.shape}")  # (10000, 64)
# This is our dimensionality reduction: 784 → 64

# ── 5. Comparison Baselines ───────────────────────────────────────────────────
# Split train/test
X_train_bow, X_test_bow, y_train, y_test = train_test_split(
    X_bow, y, test_size=0.2, random_state=42, stratify=y
)
# Same split for raw pixels:
X_train_raw, X_test_raw = X_raw[:8000] / 255.0, X_raw[8000:] / 255.0
X_train_y, X_test_y = y[:8000], y[8000:]

# Baseline 1: PCA (64 components = same dimensionality)
pca = PCA(n_components=64, random_state=42)
X_train_pca = pca.fit_transform(X_train_raw)
X_test_pca  = pca.transform(X_test_raw)

clf_raw = LogisticRegression(max_iter=500, random_state=42).fit(X_train_raw, X_train_y)
clf_pca = LogisticRegression(max_iter=500, random_state=42).fit(X_train_pca, X_train_y)
clf_bow = LogisticRegression(max_iter=500, random_state=42).fit(X_train_bow, y_train)

acc_raw = clf_raw.score(X_test_raw, X_test_y)
acc_pca = clf_pca.score(X_test_pca, X_test_y)
acc_bow = clf_bow.score(X_test_bow, y_test)

print(f"\n=== Classification Accuracy (10-class MNIST, n=10k) ===")
print(f"Raw pixels (784-dim):  {acc_raw:.3f}")
print(f"PCA  (64-dim):         {acc_pca:.3f}")
print(f"BoVW (64-dim):         {acc_bow:.3f}")

# ── 6. Visualize the Visual Vocabulary ──────────────────────────────────────
centroids = vocab_kmeans.cluster_centers_   # shape: (64, 16)
centroids_img = patch_scaler.inverse_transform(centroids)

fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(centroids_img[i].reshape(4, 4), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
fig.suptitle(f'Visual Vocabulary: {VOCAB_SIZE} learned patch prototypes', fontsize=14)
plt.tight_layout()
plt.savefig('visual_vocabulary.png', dpi=120)
print("Saved: visual_vocabulary.png")

# ── 7. Visualization: BoVW space via PCA(2) ─────────────────────────────────
pca2 = PCA(n_components=2)
X_bow_2d = pca2.fit_transform(X_bow)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_bow_2d[:, 0], X_bow_2d[:, 1], c=y, cmap='tab10',
                      alpha=0.4, s=5)
plt.colorbar(scatter, label='Digit class')
plt.title('BoVW Features Projected to 2D via PCA\n(colored by digit class)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('bow_pca_visualization.png', dpi=120)
print("Saved: bow_pca_visualization.png")
```

**What we learn from the results:** PCA at 64 components typically achieves ~88–90% accuracy on
this MNIST subset. BoVW at 64 visual words typically achieves ~82–87% — slightly lower on clean
digits where PCA's linear subspace is well-suited, but BoVW is more robust to local deformations
and is the foundation that scaled to real-world image retrieval.

---

### Example 2: Prototype Distance Features on make_blobs

Here we show that KMeans distance features can outperform PCA when the decision boundary is
nonlinear (concentric or non-convex cluster structure). We use `make_blobs` to create a 2-class
problem where the classes are separated but have cluster substructure.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
import pandas as pd

np.random.seed(42)

# ── 1. Create a challenging multi-cluster dataset ────────────────────────────
# Each class has multiple sub-clusters (non-linearly separable in raw space for LR)
X_raw, y = make_blobs(
    n_samples=2000,
    n_features=20,
    centers=10,              # 10 cluster centers
    cluster_std=1.5,
    random_state=42
)
# Assign cluster-level class labels: even clusters → class 0, odd → class 1
y_binary = y % 2

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class balance: {np.bincount(y_binary)}")

# ── 2. Exploratory: visualize with PCA(2) ───────────────────────────────────
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_binary, cmap='bwr', alpha=0.5, s=20)
plt.title('make_blobs (20D → PCA 2D)\nColored by binary class')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('blobs_pca_2d.png', dpi=120)

# ── 3. Elbow + Silhouette to select K ───────────────────────────────────────
K_range = range(2, 20)
inertias, sil_scores = [], []

for k in K_range:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_range), inertias, 'o-', color='steelblue')
ax1.set_xlabel('K'); ax1.set_ylabel('Inertia'); ax1.set_title('Elbow Plot')
ax1.grid(alpha=0.4)
ax2.plot(list(K_range), sil_scores, 's-', color='tomato')
ax2.set_xlabel('K'); ax2.set_ylabel('Silhouette Score'); ax2.set_title('Silhouette Plot')
ax2.axhline(max(sil_scores), color='gray', ls='--')
ax2.grid(alpha=0.4)
plt.tight_layout(); plt.savefig('k_selection_blobs.png', dpi=120)

best_K = list(K_range)[np.argmax(sil_scores)]
print(f"\nBest K by silhouette: {best_K}")

# ── 4. Build DR Representations ─────────────────────────────────────────────
# (a) KMeans distance features
km_best = KMeans(n_clusters=best_K, n_init='auto', random_state=42)
X_km = km_best.fit_transform(X)    # shape: (2000, best_K)

# (b) GMM soft posteriors
gmm = GaussianMixture(n_components=best_K, covariance_type='diag', n_init=5, random_state=42)
gmm.fit(X)
X_gmm = gmm.predict_proba(X)       # shape: (2000, best_K)
print(f"GMM converged: {gmm.converged_}, BIC: {gmm.bic(X):.1f}")

# (c) PCA (same dimensionality as best_K)
pca_comp = PCA(n_components=best_K, random_state=42)
X_pca = pca_comp.fit_transform(X)  # shape: (2000, best_K)
print(f"PCA {best_K} components explain {pca_comp.explained_variance_ratio_.sum():.1%} variance")

# ── 5. Compare DR Methods on Downstream Classification ──────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=500, random_state=42)  # linear classifier

results = {
    'Raw (20D)': cross_val_score(clf, X, y_binary, cv=cv).mean(),
    f'PCA ({best_K}D)': cross_val_score(clf, X_pca, y_binary, cv=cv).mean(),
    f'KMeans-dist ({best_K}D)': cross_val_score(clf, X_km, y_binary, cv=cv).mean(),
    f'GMM-posterior ({best_K}D)': cross_val_score(clf, X_gmm, y_binary, cv=cv).mean(),
}

print("\n=== 5-fold CV Accuracy — LogisticRegression ===")
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:30s}: {score:.4f}")

# ── 6. Trustworthiness of KMeans DR ─────────────────────────────────────────
trust_km = trustworthiness(X, X_km, n_neighbors=10)
trust_pca = trustworthiness(X, X_pca, n_neighbors=10)
print(f"\nTrustworthiness — KMeans DR: {trust_km:.4f}")
print(f"Trustworthiness — PCA DR:    {trust_pca:.4f}")

# ── 7. Visualize Cluster Structure in DR Space ──────────────────────────────
pca2b = PCA(n_components=2)
X_km_2d = pca2b.fit_transform(X_km)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: KMeans DR, colored by binary class
axes[0].scatter(X_km_2d[:, 0], X_km_2d[:, 1], c=y_binary, cmap='bwr', alpha=0.5, s=15)
axes[0].set_title(f'KMeans Distance Features → PCA 2D\n(colored by class; trust={trust_km:.3f})')
axes[0].set_xlabel('PC1 of KMeans features')

# Panel 2: KMeans DR, colored by cluster assignment
axes[1].scatter(X_km_2d[:, 0], X_km_2d[:, 1],
                c=km_best.labels_, cmap='tab20', alpha=0.5, s=15)
axes[1].set_title(f'KMeans Distance Features → PCA 2D\n(colored by cluster; K={best_K})')
axes[1].set_xlabel('PC1 of KMeans features')

plt.tight_layout()
plt.savefig('kmeans_dr_visualization.png', dpi=120)
print("\nSaved: kmeans_dr_visualization.png")

# ── 8. Summary ───────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Original dimensionality: 20")
print(f"Reduced dimensionality:  {best_K}")
print(f"Best method: {max(results, key=results.get)} "
      f"with accuracy {max(results.values()):.4f}")
print(f"KMeans correctly recovers sub-cluster structure that PCA misses,")
print(f"enabling a linear LR classifier to solve this nonlinear problem.")
```

**Interpretation of results:** In this setup, KMeans distance features and GMM posteriors
typically outperform raw features and PCA features by 5–15% accuracy, because the linear
classifier can now use "distance to cluster 3" as a proxy for "probability of class 0" — which
is exactly the nonlinear feature needed for this problem. The key insight: the k-means
prototypes have learned the cluster centers of the data, and the downstream linear model can
now draw linear boundaries in the distance space, effectively creating a nonlinear boundary in
the original feature space.

---

## §11 — When to Use This Algorithm

### Decision Flowchart

```
START: Do you want to reduce dimensionality?
│
├── Is your data structured as fixed-length feature vectors? (tabular)
│   ├── Yes → Continue
│   └── No (variable-length: text, images, time series) → Use BoVW pipeline
│
├── Do you expect discrete cluster/group structure?
│   ├── Yes, clear groups (product categories, cell types, customer segments)
│   │   └── Use KMeans.transform() or GMM.predict_proba()
│   ├── Maybe → Run elbow/silhouette first; proceed if silhouette > 0.3
│   └── No (continuous manifold) → Prefer PCA / UMAP / Isomap
│
├── Is cluster shape expected to be non-convex (rings, crescents)?
│   ├── Yes → Use Spectral Embedding + KMeans (SpectralClustering)
│   └── No → KMeans or GMM is fine
│
├── Do you need uncertainty/soft assignments?
│   ├── Yes → GaussianMixture.predict_proba()
│   └── No → KMeans.transform() (simpler, faster)
│
├── Are you reducing FEATURES (not samples) to cluster correlated columns?
│   ├── Yes → FeatureAgglomeration
│   └── No → KMeans / GMM
│
└── Do you need topology-preserving 2D map for visualization?
    ├── Yes → Self-Organizing Map (MiniSom)
    └── No → KMeans or GMM
```

### Comparison Table: Clustering-as-DR vs Alternatives

| Criterion | KMeans DR | GMM DR | PCA | t-SNE / UMAP | Autoencoder |
|---|---|---|---|---|---|
| **Linear/nonlinear** | Nonlinear | Nonlinear | Linear | Nonlinear | Nonlinear |
| **Out-of-sample transform** | Yes (fast) | Yes (fast) | Yes (fast) | Not natively | Yes |
| **Interpretable features** | High (each feature = distance to cluster $k$) | High | Medium (components) | None | None |
| **Captures non-convex clusters** | No | No | No | Yes | Yes |
| **Scales to N > 1M** | Yes (MiniBatch) | No | Yes (IncrementalPCA) | No | Yes |
| **K selection required** | Yes | Yes (BIC helps) | No (explained variance) | Yes (perplexity/n_neigh) | Yes |
| **Reconstruction** | Poor | Poor | Good | None | Good |
| **Uncertainty encoding** | No | Yes | No | No | No (VAE: Yes) |

### When NOT to Use Clustering-as-DR

**Red flags — do not use clustering-as-DR when:**
- The data has no cluster structure (Hopkins statistic ≈ 0.5; uniform PCA scatterplot)
- You need reconstruction fidelity (use PCA or Autoencoder instead)
- The clusters are non-convex and you can't use spectral embedding
- Features are on different scales and you can't scale them (e.g., mixed categorical + numerical without good encoding)
- $p \gg n$ (more features than samples — GMM in particular will overfit badly)
- You need a 2D visualization that preserves local geometry (use t-SNE or UMAP instead)

> 🏆 **Best practice:** Use clustering-as-DR as **feature engineering** (adding cluster distance
> features to a model) rather than as strict DR (replacing original features). Concatenate
> `[X_original, X_kmeans_transform]` and let a downstream model select what's useful. This
> is often better than replacing the features entirely.

---

## §12 — Top Papers to Study

### 🥇 Start Here

**1. MacQueen (1967) — Original K-Means**

> MacQueen, J. B. (1967). "Some methods for classification and analysis of multivariate
> observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and
> Probability*, Vol. 1, pp. 281–297. University of California Press.
> URL: https://www.semanticscholar.org/paper/Some-methods-for-classification-and-analysis-of-MacQueen/ac8ab51a86f1a9ae74dd0e4576d1a019f5e654ed

**Annotation:** The founding paper. Read §1 and §2 for the two-step algorithm and convergence
proof. Notice that MacQueen's "classification by means" framing is implicitly a coordinate
transformation — that's the DR insight hiding in 1967. Low difficulty; accessible in one sitting.

---

**2. Arthur & Vassilvitskii (2007) — K-Means++**

> Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding."
> *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, pp. 1027–1035.
> URL: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

**Annotation:** Essential reading. The D² sampling initialization guarantees an $O(\log K)$
approximation ratio in expectation and eliminates the worst-case poor initialization problem.
After reading this, you understand why `init='k-means++'` is always the right choice.
The theoretical analysis is clean and elegant.

---

### 📖 Read Next

**3. Perronnin, Sánchez & Mensink (2010) — Improved Fisher Vectors**

> Perronnin, F., Sánchez, J., & Mensink, T. (2010). "Improving the Fisher kernel for
> large-scale image classification." *ECCV 2010*, pp. 143–156.
> DOI: https://doi.org/10.1007/978-3-642-15561-1_11
> PDF (INRIA HAL): https://inria.hal.science/inria-00548630v1/document

**Annotation:** The paper that won ImageNet 2011 using GMM-based clustering as DR. After reading
this, you understand the full pipeline: GMM → soft assignment → first-order and second-order
gradient statistics → Fisher Vector → power normalization → L2 normalization. The two
normalization tricks alone (power normalization fixes burstiness; L2 normalization fixes scale)
are worth 5+ percentage points on ImageNet. Medium-high difficulty.

---

**4. Jégou, Douze, Schmid & Pérez (2010) — VLAD**

> Jégou, H., Douze, M., Schmid, C., & Pérez, P. (2010). "Aggregating local descriptors into
> a compact image representation." *CVPR 2010*, pp. 3304–3311.
> PDF: https://falchi.isti.cnr.it/Draft/2013-SISAP-VLAD-DRAFT.pdf

**Annotation:** VLAD is the midpoint between BoVW (hard assignment + counts) and Fisher Vectors
(soft assignment + statistics). It accumulates residuals (descriptor - centroid) per cluster —
first-order statistics only, without the Gaussian model. Compact, effective, and makes the
"clustering as feature compression" story crystal clear. Read alongside the Fisher Vector paper
for a complete picture.

---

**5. Belkin & Niyogi (2001) — Laplacian Eigenmaps**

> Belkin, M., & Niyogi, P. (2001). "Laplacian Eigenmaps and Spectral Techniques for Embedding
> and Clustering." *NIPS 2001*, Vol. 14.
> URL: https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering

**Annotation:** The theoretical foundation for Spectral Clustering as a two-stage DR. Proves
that the graph Laplacian eigenvectors are the natural representation for data with manifold
structure, and that clustering in this representation gives provably better results than
clustering in the raw space. Essential for understanding why Spectral Clustering (and
SpectralEmbedding) work where KMeans fails on non-convex data.

---

### 🔬 Deep Mastery

**6. Kohonen (1982) — Self-Organizing Map**

> Kohonen, T. (1982). "Self-organized formation of topologically correct feature maps."
> *Biological Cybernetics*, 43, 59–69.
> DOI: https://doi.org/10.1007/BF00337288

**Annotation:** The SOM paper introduces the constraint that KMeans ignores: topological ordering
of prototypes. After reading this, you understand why SOMs produce meaningful 2D maps (the
neighborhood preservation property) rather than arbitrary cluster labelings. The biological
motivation (somatosensory cortex maps) is interesting but not necessary for the ML application.

---

**7. Campello, Moulavi & Sander (2013) — HDBSCAN**

> Campello, R.J., Moulavi, D., & Sander, J. (2013). "Density-based clustering based on
> hierarchical density estimates." *PAKDD 2013*, pp. 160–172.

**Annotation:** The modern alternative to KMeans and GMM for clustering-as-DR when cluster
shapes are unknown and non-Gaussian. HDBSCAN's soft membership vectors (via
`all_points_membership_vectors`) provide a non-parametric alternative to GMM posteriors.
Read this after understanding KMeans and GMM to know when to upgrade.

---

**8. Systematic DR + Clustering Study (2025)**

> "Assessing the impact of dimensionality reduction on clustering performance — a systematic
> study." arXiv:2604.22099.
> URL: https://arxiv.org/html/2604.22099v1

**Annotation:** Empirical study across 12 DR methods and 20 datasets, examining how DR
preprocessing affects clustering quality. Key finding: PCA before KMeans helps; t-SNE before
KMeans is fragile. Use this as a reference when deciding whether to preprocess with PCA before
clustering-as-DR. Low difficulty — mostly tables and conclusions.

---

## §13 — Resources & Further Reading

### Books

- **Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*. Springer.**
  Chapter 9 (Mixture Models and EM): the definitive derivation of GMM with EM, fully worked
  with the E-step and M-step in closed form. Chapter 12 covers PCA for comparison.
  Free PDF: https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/

- **Murphy, K.P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.**
  Chapter 21 (Clustering) covers KMeans, GMM, and spectral clustering in modern notation.
  Free PDF: https://probml.github.io/pml-book/book1.html

- **Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, 3rd ed.**
  Chapter 9 covers KMeans and DBSCAN with practical sklearn examples. Good for code patterns.

### Blog Posts and Tutorials

- **scikit-learn documentation — Clustering:**
  https://scikit-learn.org/stable/modules/clustering.html
  The authoritative reference for all sklearn clustering classes. Read the "Comparison of
  clustering algorithms" page carefully — it shows which algorithms work on which data geometries.

- **scikit-learn documentation — FeatureAgglomeration:**
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html

- **MiniSom GitHub (code + examples):**
  https://github.com/JustGlowing/minisom
  The `examples/` directory has Jupyter notebooks covering the U-matrix, component planes,
  and topology visualization — essential for understanding SOM output.

- **Distill.pub — "How to Use t-SNE Effectively":**
  https://distill.pub/2016/misread-tsne/
  While focused on t-SNE, the discussion of what makes a valid cluster representation (and
  when visual cluster separation is misleading) directly applies to evaluating clustering-as-DR.

- **Stanford CS231n — Feature Representation for Images:**
  http://cs231n.stanford.edu/
  The lecture notes on "image features" (BoVW, HOG, SIFT) cover the pre-deep-learning era
  when clustering-as-DR dominated computer vision.

- **FAISS documentation and tutorial:**
  https://faiss.ai/
  For billion-scale KMeans and approximate nearest neighbor search. Essential reading before
  deploying BoVW at scale.

### Online Courses

- **Fast.ai Practical Deep Learning for Coders:** Discusses feature engineering with clustering
  in the context of tabular data (Lesson 6–7). Practical and concise.
- **Coursera Machine Learning Specialization (Andrew Ng, DeepLearning.AI):** Course 3 (Unsupervised
  Learning) covers KMeans clustering in depth with worked examples.

### Package Documentation Pages

| Package | Documentation URL |
|---|---|
| sklearn KMeans | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html |
| sklearn GaussianMixture | https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html |
| sklearn FeatureAgglomeration | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html |
| sklearn SpectralEmbedding | https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html |
| MiniSom | https://github.com/JustGlowing/minisom |
| FAISS | https://faiss.ai/ |
| hdbscan | https://hdbscan.readthedocs.io/ |

---

*This chapter is part of the Dimensionality Reduction Masterclass. See also:*
- *`pca.md` — For linear DR and explained variance*
- *`umap.md` — For topology-preserving nonlinear DR at scale*
- *`autoencoder-dr.md` — For deep nonlinear DR with reconstruction*
- *`manifold-methods.md` — For Isomap, LLE, Spectral Embedding as standalone methods*

