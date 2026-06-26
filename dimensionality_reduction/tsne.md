# t-SNE: The Complete Masterclass

> **Why this algorithm matters:** In 2008, a paper quietly posted to the Journal of Machine Learning
> Research changed how biologists, linguists, and ML researchers visualize high-dimensional data.
> t-SNE — t-Distributed Stochastic Neighbor Embedding — produced embeddings of MNIST digits so
> strikingly organized that researchers immediately recognized something qualitatively different was
> happening. Within a decade, it became the default visualization tool for single-cell RNA sequencing
> (scRNA-seq), used in thousands of published Nature and Science papers to reveal immune cell
> subpopulations, tumor microenvironments, and developmental trajectories invisible to PCA.
> Understanding t-SNE deeply — every probability, every gradient, every hyperparameter trap — makes
> you a better scientist and a better engineer. This chapter gives you that depth.

---

## §0 — Python Ecosystem & Package Guide

Before touching the math, know your tools. The t-SNE landscape has three tiers: the standard
library entry point (sklearn), a production-grade accelerated implementation (openTSNE), and a
collection of specialized variants. The choice between them is not cosmetic — for datasets above
10,000 points, sklearn can take over an hour while openTSNE finishes in minutes.

### Complete Package Table

| Package | PyPI name | Version (mid-2026) | Algorithm | Best for | License | Maintained? |
|---|---|---|---|---|---|---|
| `sklearn.manifold.TSNE` | `scikit-learn` | ~1.5.x | Barnes-Hut (default) or exact | Teaching, small datasets (N < 10k), unified API | BSD-3 | Yes (very active) |
| `openTSNE` | `openTSNE` | 1.0.4 | FIt-SNE (FFT) or Barnes-Hut, auto-selected | Large datasets, incremental embedding, research | BSD-3 | Yes (active) |
| `MulticoreTSNE` | `MulticoreTSNE` | 0.1 | Barnes-Hut, multi-core | Medium datasets, drop-in sklearn replacement | BSD-3 | Mostly inactive (2018) |

**Installation:**

```bash
# Core (almost certainly already installed)
pip install scikit-learn

# Recommended for anything serious
pip install openTSNE
# or via conda (preferred — avoids BLAS/FFTW compilation):
conda install -c conda-forge opentsne
```

### Top Picks — Recommendation Table

| Package | Best for | Modern/Legacy |
|---|---|---|
| **`openTSNE`** | Large datasets (N > 10k), incremental embedding, production | **Modern — use this** |
| **`sklearn.TSNE`** | Quick exploration, small N, Pipeline consistency | Modern but limited at scale |
| `MulticoreTSNE` | Drop-in replacement when openTSNE is unavailable | Legacy — prefer openTSNE |

> 🏆 **Best practice:** Default to `openTSNE` for any serious work. Fall back to `sklearn.TSNE`
> only when you need a single consistent API across all DR algorithms (e.g., in a course or
> comparison notebook) or when N < 5,000 and installation overhead matters.

### GPU-Accelerated Options

As of mid-2026, there is no single stable, widely-adopted GPU-accelerated t-SNE package in the
Python ecosystem. The closest options are:

- **`rapids cuML`** (NVIDIA): Implements GPU t-SNE via the `cuml.manifold.TSNE` class, available
  if you have an NVIDIA GPU and the cuML environment. Uses the Barnes-Hut approximation on GPU.
  Syntax is sklearn-compatible. For N > 500k, this is the fastest option.
- **Custom PyTorch**: Implement the t-SNE loss manually in PyTorch and use autograd for gradients.
  Used in parametric t-SNE implementations.

For CPU-only workloads, openTSNE with FIt-SNE is close to theoretical optimality and practical
for datasets up to several million points.

### The Same `fit_transform()` Call Across Top Packages

```python
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target  # (1797, 64)

# ── Option 1: sklearn (Barnes-Hut, the standard) ──────────────────────────
from sklearn.manifold import TSNE

tsne_sklearn = TSNE(
    n_components=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate='auto',   # max(N/48, 50); best practice since sklearn v1.2
    max_iter=1000,          # renamed from n_iter in sklearn v1.5
    init='pca',             # best practice since sklearn v1.2 (was 'random')
    method='barnes_hut',
    angle=0.5,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
X_sk = tsne_sklearn.fit_transform(X)
print(f"sklearn  | shape={X_sk.shape} | KL={tsne_sklearn.kl_divergence_:.4f}")

# ── Option 2: openTSNE sklearn-compatible API (FIt-SNE backend) ───────────
from openTSNE import TSNE as openTSNE

tsne_open = openTSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,        # openTSNE uses n_iter (not max_iter)
    metric='euclidean',
    n_jobs=-1,
    random_state=42,
    verbose=True,
)
X_open = tsne_open.fit_transform(X)
print(f"openTSNE | shape={X_open.shape}")

# ── Option 3: openTSNE modular API (full control) ─────────────────────────
import openTSNE as otsne

# Step 1: compute affinities (the input-space probability distribution P)
affinities = otsne.affinity.PerplexityBasedNN(
    X,
    perplexity=30,
    n_jobs=-1,
    random_state=42,
)

# Step 2: PCA initialization
init = otsne.initialization.pca(X, n_components=2, random_state=42)

# Step 3: build embedding object
embedding = otsne.TSNEEmbedding(
    init,
    affinities,
    negative_gradient_method="fft",   # "fft" = FIt-SNE; "bh" = Barnes-Hut
    random_state=42,
)

# Step 4: two-phase optimization (explicit control)
embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)  # early exag
embedding.optimize(n_iter=750, exaggeration=1,  momentum=0.8, inplace=True)  # refinement

X_modular = np.array(embedding)
print(f"modular  | shape={X_modular.shape}")
```

> 💡 **Intuition:** The modular openTSNE API exposes exactly what t-SNE is doing under the hood:
> (1) compute pairwise similarities in high-D, (2) initialize positions, (3) run early exaggeration
> to organize clusters, (4) refine. The sklearn API hides steps 1-3 behind a single `fit_transform`
> call. Both produce the same result — the modular API just gives you a lever at each step.

### Key API Differences to Know

| Feature | sklearn TSNE | openTSNE |
|---|---|---|
| Parameter name | `max_iter` (was `n_iter` pre-v1.5) | `n_iter` |
| Default algorithm | Barnes-Hut | FIt-SNE (auto-selects based on N) |
| Incremental embedding | Not supported | `embedding.transform(X_new)` |
| Exaggeration schedule | Fixed (early only) | Fully programmable |
| Out-of-sample transform | Not possible | Supported via `prepare_partial` |
| Memory for large N | O(N log N) | O(N) (FFT backend) |
| Speed at N=200k | >90 minutes | <4 minutes |

> ⚠️ **Pitfall:** sklearn renamed `n_iter` → `max_iter` in version 1.5 and plans to remove
> `n_iter` entirely in v1.7. openTSNE still uses `n_iter`. If you copy-paste code between the
> two packages, watch this parameter name. Using `n_iter` in sklearn 1.5.x raises a
> `DeprecationWarning`; in v1.7+ it will raise an error.

### Version Change Summary (sklearn)

| Version | Change |
|---|---|
| sklearn 1.2 | `learning_rate` default: `200.0` → `'auto'` |
| sklearn 1.2 | `init` default: `'random'` → `'pca'` |
| sklearn 1.5 | `n_iter` renamed to `max_iter` (`n_iter` deprecated) |
| sklearn 1.7 (planned) | `n_iter` parameter removed entirely |

These are not cosmetic changes — the old defaults (`init='random'`, `learning_rate=200`) produced
systematically poor embeddings for large datasets. If you find t-SNE code online that uses these
old defaults, update it.

---

## §1 — The Origin: The Paper That Started It All

> 📜 **Origin/Citation:** Laurens van der Maaten and Geoffrey E. Hinton, "Visualizing
> High-Dimensional Data Using t-SNE," *Journal of Machine Learning Research*, 9(Nov):2579–2605,
> 2008. URL: https://jmlr.org/papers/v9/vandermaaten08a/vandermaaten08a.html

### The Problem Being Solved

By 2008, machine learning researchers had a serious visualization problem. The datasets they were
working with — MNIST (784 dimensions), gene expression arrays (thousands of genes), document-term
matrices (tens of thousands of features) — were fundamentally high-dimensional. The standard tool
for visualization was PCA, which projects data onto the directions of maximum variance. PCA is
honest about what it preserves: linear structure. But the most interesting patterns in biological
data, text data, and image data are often **non-linear** — they live on curved manifolds embedded
in the high-dimensional space.

The existing nonlinear methods — Isomap (Tenenbaum et al., 2000), Locally Linear Embedding
(Roweis & Saul, 2000), kernel PCA — all suffered from the same core failure mode: they struggled
to simultaneously reveal **both** local cluster structure and the broader relationships between
clusters in a single 2D plot. Isomap worked well on manifolds with uniform density but failed
when clusters had very different densities. LLE produced beautiful local structure but often
embedded everything into a single tangled mass at the center.

The direct predecessor to t-SNE was **SNE** (Stochastic Neighbor Embedding), introduced by Hinton
and Roweis at NeurIPS 2002. SNE had the right idea — model pairwise similarities as probabilities
and minimize the KL divergence between high-D and low-D distributions — but had two crippling
problems that prevented it from producing clean visualizations in practice.

### What SNE Got Right (and Wrong)

**SNE's core idea:** For each point $\mathbf{x}_i$, define a conditional probability distribution
$P_i$ over all other points, where the probability $p_{j|i}$ of picking point $j$ as a neighbor
of $i$ is proportional to the Gaussian similarity:

$$p_{j|i} = \frac{\exp\!\left(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2\right)}{\sum_{k \neq i} \exp\!\left(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2\right)}$$

Similarly, in the low-dimensional embedding space $\mathbf{Z}$:

$$q_{j|i} = \frac{\exp\!\left(-\|\mathbf{z}_i - \mathbf{z}_j\|^2\right)}{\sum_{k \neq i} \exp\!\left(-\|\mathbf{z}_i - \mathbf{z}_k\|^2\right)}$$

The goal: find an embedding $\mathbf{Z}$ that minimizes the sum of KL divergences between
the per-point distributions:

$$C_\text{SNE} = \sum_i \mathrm{KL}(P_i \| Q_i) = \sum_i \sum_j p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$$

This objective has an elegant interpretation: if $p_{j|i}$ is high (points $i$ and $j$ are
genuine neighbors in high-D) but $q_{j|i}$ is low (they are far apart in the embedding), the
KL divergence penalizes this heavily. The optimizer is forced to pull true neighbors together.

**SNE's failures:**

1. **Asymmetry:** The KL divergence $\mathrm{KL}(P_i \| Q_i)$ is computed per point $i$, using
   asymmetric conditionals. This makes the gradient messy: the "attraction force" from $p_{j|i}$
   and the "repulsion force" from $p_{i|j}$ play different roles. Gradient descent on this
   landscape is unstable and prone to poor local optima.

2. **The crowding problem** (explained deeply in §2): When projecting from 10+ dimensions to 2D,
   there simply is not enough area in the plane to accommodate all the "medium distance" neighbors
   from high-D. They crowd together in the center, producing a dense, uninterpretable blob.
   A Gaussian kernel in the embedding space cannot solve this because its probability mass decays
   too fast — it does not allow points to spread out far enough to accommodate the missing volume.

### Van der Maaten & Hinton's Two Innovations

The 2008 paper made two precise changes to SNE, each targeting one of its two core failures:

**Fix 1: Symmetrize the similarity distribution.** Replace the asymmetric conditionals with a
single symmetric joint distribution over pairs:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$

This ensures $\sum_{ij} p_{ij} = 1$ (a proper probability distribution) and that every point
contributes meaningfully to the cost function even if it is an outlier (because $p_{ij} \geq
\frac{1}{2N}$ even for very distant pairs). The gradient simplifies dramatically.

**Fix 2: Replace the Gaussian in the embedding with a Student-t distribution.** In the embedding
space, use the Student-t distribution with 1 degree of freedom (the Cauchy distribution):

$$q_{ij} = \frac{\left(1 + \|\mathbf{z}_i - \mathbf{z}_j\|^2\right)^{-1}}{\sum_{k \neq l} \left(1 + \|\mathbf{z}_k - \mathbf{z}_l\|^2\right)^{-1}}$$

This is the heavy-tail trick. The Cauchy distribution's tails fall as $r^{-2}$ rather than
$e^{-r^2}$. The result: "medium distance" pairs in high-D can be pushed much farther apart in the
2D embedding without incurring a large penalty — the heavy tail accommodates the distance. Close
pairs (genuine neighbors) are still pulled together by strong attractive forces. The net effect is
that clusters form with clear, empty space between them.

### The Original t-SNE Algorithm

**Pseudocode (from the 2008 paper):**

```
Input: data X ∈ ℝ^{n×p}, perplexity Perp, iterations T,
       learning rate η, momentum α(t), early exaggeration factor

1. Compute pairwise affinities p_{j|i} (with σ_i set via binary search for Perp(P_i) = Perp)
2. Set p_{ij} = (p_{j|i} + p_{i|j}) / 2N
3. Initialize Y ~ N(0, 10⁻⁴ I)
4. For t = 1 to T:
   a. If t ≤ 250: multiply all p_{ij} by early_exaggeration (default 12)
   b. Compute low-dim affinities q_{ij} using Cauchy kernel
   c. Compute gradient: ∂C/∂z_i = 4 Σ_{j≠i} (p_{ij} - q_{ij})(z_i - z_j)(1 + ||z_i - z_j||²)⁻¹
   d. Update Y with gradient descent + momentum
5. Return Y
```

### Historical Context

The t-SNE paper arrived at a moment when the ML community was wrestling with the "manifold
hypothesis" — the idea that real-world high-dimensional data lives on low-dimensional manifolds.
Isomap and LLE had demonstrated that non-linear methods could reveal structure invisible to PCA,
but they were fragile, difficult to tune, and computationally expensive. t-SNE was different: it
consistently produced stunning, interpretable visualizations on the first try, even for beginners.

Within two years, it had become the standard tool in genomics, NLP, and computer vision for
embedding visualization. The MNIST digit dataset became the canonical benchmark — t-SNE separated
the 10 digit classes into 10 clean clusters in 2D with no supervision, using only the raw pixel
values. This was compelling evidence that something real was being captured.

> 📜 **Origin/Citation:** Hinton, G. E. and Roweis, S. T., "Stochastic Neighbor Embedding,"
> *Advances in Neural Information Processing Systems 15 (NeurIPS 2002)*, pp. 857–864. The
> predecessor to t-SNE. Essential historical context but not needed for practical t-SNE use.

---

## §2 — The Algorithm, Deeply Explained

Let's derive t-SNE from first principles. You don't need to be a mathematician — you need to be a
practitioner who wants to understand WHY each design choice was made, so that you can debug
bad embeddings and tune the algorithm with confidence.

### 2.1 Modeling High-D Similarities as Probabilities

We have $n$ data points $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^p$. The first task is
to measure how similar each pair of points is in the original space. t-SNE converts pairwise
Euclidean distances into probabilities using a Gaussian kernel around each point $i$:

$$p_{j|i} = \frac{\exp\!\left(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2\right)}{\sum_{k \neq i} \exp\!\left(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2\right)}, \quad p_{i|i} = 0$$

Notice two things: (1) each point $i$ has its own bandwidth $\sigma_i$, and (2) the denominator
normalizes so the probabilities sum to 1 over all other points. The conditional $p_{j|i}$ is the
probability that point $i$ would pick $j$ as its neighbor if neighbors were picked in proportion
to their Gaussian similarity.

**Symmetrization:** Rather than working with asymmetric conditionals, t-SNE forms a joint:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$

Summing over all pairs: $\sum_{i \neq j} p_{ij} = 1$. This joint distribution is symmetric
($p_{ij} = p_{ji}$), which dramatically simplifies the gradient. It also ensures that even
outlier points (with very small $p_{j|i}$ for all $j$) have non-negligible $p_{ij}$ values and
therefore remain attracted to some portion of the data.

### 2.2 Setting Bandwidth via Perplexity

The bandwidth $\sigma_i$ is not set by hand — it is found automatically for each point $i$ via a
binary search on the **perplexity**:

$$\mathrm{Perp}(P_i) = 2^{H(P_i)}, \quad H(P_i) = -\sum_{j} p_{j|i} \log_2 p_{j|i}$$

Here $H(P_i)$ is the Shannon entropy of the conditional distribution around point $i$. The
perplexity is the "effective number of neighbors" that point $i$ considers.

> 💡 **Intuition:** Think of perplexity as the knob that sets the "scale" at which t-SNE looks
> at the data. Perplexity = 30 means each point acts as if it has about 30 meaningful neighbors.
> The algorithm finds the Gaussian width $\sigma_i$ so that point $i$'s neighborhood has exactly
> this entropy — no more, no less.

The binary search works differently for each point:
- In **dense regions**, the Gaussian must be narrow (small $\sigma_i$) to pick up only the
  truly close neighbors without spreading probability mass to farther points.
- In **sparse regions**, the Gaussian must be wide (large $\sigma_i$) to find enough neighbors
  to reach the target perplexity.

This adaptive bandwidth is one of t-SNE's great strengths: it treats every point's neighborhood
at a locally appropriate scale. Dense and sparse regions of the data are treated fairly.

### 2.3 The Crowding Problem — Why We Need Heavy Tails

This is the most important insight in t-SNE. Sit with this for a moment.

Consider data uniformly distributed on the surface of a $D$-dimensional sphere of radius $R$.
The volume of a thin shell at radius $r$ with thickness $\delta r$ scales as $r^{D-1} \delta r$.
As $D$ grows, the fraction of the sphere's volume that lies within distance $\epsilon$ of the
center shrinks to zero. Almost all the volume is in a thin shell near the surface.

In practice: **in high dimensions, most pairs of points are at "medium" distance** — neither very
close nor very far. There is very little "room" near any given point.

When we project this data to 2D, we need to represent all these "medium distance" pairs in a
2D plane. The area of a 2D circle is $\pi r^2$, while in $D$ dimensions the comparable volume
scales as $r^D$. For $D = 10$, the ratio of volumes is $r^{10} / r^2 = r^8$ — exponentially
more space in high-D than in low-D at any given radius. The 2D plane simply does not have enough
area to place all these medium-distance neighbors at the right distances from each other.

The result with a Gaussian kernel in the embedding space: the optimizer tries to place all medium
neighbors at medium distances, but there is not enough room. They collapse into the center,
forming a crowded, uninformative blob.

**The Student-t fix:** The Cauchy distribution (Student-t with 1 degree of freedom) has the
PDF:

$$f(z) \propto \left(1 + z^2\right)^{-1}$$

Its tails fall as $z^{-2}$ — much slower than the Gaussian's $e^{-z^2}$. In the embedding:

$$q_{ij} = \frac{\left(1 + \|\mathbf{z}_i - \mathbf{z}_j\|^2\right)^{-1}}{\sum_{k \neq l} \left(1 + \|\mathbf{z}_k - \mathbf{z}_l\|^2\right)^{-1}}$$

With this choice, a moderate similarity $p_{ij}$ can be matched by a **large** distance
$\|\mathbf{z}_i - \mathbf{z}_j\|$ in the embedding — the heavy tail allows $q_{ij}$ to remain
non-negligible even when points are far apart. This extra "room" is exactly what is needed to
accommodate the medium-distance pairs from high-D without crowding.

> 💡 **Intuition:** Imagine you have 1000 friends in a city (high-D) and need to seat them around
> a campfire (2D). With a Gaussian rule ("friends must sit close to people they know"), everyone
> crowds the fire and you can't tell who is close to whom. With a heavy-tail rule ("friends can
> sit far away from acquaintances"), you can spread them out and the genuine close friends are
> clearly visible as clusters. The Student-t distribution is the heavy-tail rule.

### 2.4 The Full t-SNE Objective and Gradient

**Objective:** Minimize the KL divergence between the joint distributions $P$ and $Q$:

$$C = \mathrm{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

Expanding this with the normalization constant $Z = \sum_{k \neq l}(1 + \|\mathbf{z}_k - \mathbf{z}_l\|^2)^{-1}$:

$$C = \sum_{i \neq j} p_{ij} \log p_{ij} - \sum_{i \neq j} p_{ij} \log q_{ij}$$

The first term is constant w.r.t. $\mathbf{Z}$, so minimizing $C$ is equivalent to maximizing
$\sum_{i \neq j} p_{ij} \log q_{ij}$.

**Gradient** (derived by van der Maaten and Hinton using the chain rule):

$$\frac{\partial C}{\partial \mathbf{z}_i} = 4 \sum_{j \neq i} (p_{ij} - q_{ij}) \left(1 + \|\mathbf{z}_i - \mathbf{z}_j\|^2\right)^{-1} (\mathbf{z}_i - \mathbf{z}_j)$$

This gradient has a beautifully interpretable physical structure. Think of it as a **force field**
acting on each embedding point $\mathbf{z}_i$:

- The direction $(\mathbf{z}_i - \mathbf{z}_j)$ is along the line between the two points.
- The sign of $(p_{ij} - q_{ij})$ determines whether the force is **attractive** or **repulsive**:
  - $p_{ij} > q_{ij}$: points $i$ and $j$ are genuine high-D neighbors but too far apart in the
    embedding → **attractive force** pulls them together.
  - $p_{ij} < q_{ij}$: points $i$ and $j$ are NOT genuine neighbors but are too close in the
    embedding → **repulsive force** pushes them apart.
- The factor $\left(1 + \|\mathbf{z}_i - \mathbf{z}_j\|^2\right)^{-1}$ is the heavy-tail weight:
  for distant pairs, the repulsive force decays slowly (unlike a Gaussian), which is what creates
  the clear empty space between clusters.

> 🔬 **Deep dive:** The factor of 4 comes from differentiating $\log q_{ij}$ with respect to
> $\mathbf{z}_i$, noting that the normalization $Z$ also depends on $\mathbf{z}_i$. Working this
> out: $\partial \log q_{ij} / \partial \mathbf{z}_i = -2(\mathbf{z}_i - \mathbf{z}_j)(1+\|\mathbf{z}_i-\mathbf{z}_j\|^2)^{-1} + 2\sum_k(\mathbf{z}_i-\mathbf{z}_k)(1+\|\mathbf{z}_i-\mathbf{z}_k\|^2)^{-1}/Z$. After simplification and using the symmetry of $q_{ij}$, you get the attractive and repulsive terms that combine into the formula above.

### 2.5 The Two-Phase Optimization

t-SNE uses gradient descent with momentum, but with a crucial two-phase schedule:

**Phase 1: Early Exaggeration (default: iterations 1–250)**

All $p_{ij}$ are multiplied by a factor $\alpha$ (default: 12). The modified objective becomes:

$$C_\text{exag} = \mathrm{KL}(\alpha P \| Q)$$

With $\alpha = 12$, the attractive forces are amplified 12-fold. The optimizer responds by
pulling genuine neighbors together very aggressively, forming compact cluster seeds separated by
large gaps. The repulsive forces also grow (since $q_{ij}$ adjusts), but they cannot keep up with
the amplified attraction. The result: the embedding organizes into a rough cluster structure
quickly and reliably.

> 💡 **Intuition:** Think of early exaggeration as the algorithm "pre-organizing" the room before
> refining seat assignments. It figures out roughly which group sits where before worrying about
> exactly who sits next to whom within the group.

There is a deep connection here to spectral clustering: early exaggeration with factor $\alpha$
is approximately equivalent to power iterations on the graph Laplacian of the k-nearest-neighbor
graph of the data. This connection partially explains why t-SNE often reveals the same cluster
structure as spectral methods, but with added ability to show fine-grained sub-structure.

**Phase 2: Normal Optimization (default: iterations 251–1000)**

The exaggeration factor is removed ($\alpha = 1$). The optimizer now refines the cluster
sub-structure: points within a cluster arrange themselves, and inter-cluster relationships
settle. The momentum schedule uses 0.8 in this phase (vs. 0.5 in Phase 1) to accelerate
convergence.

**Learning rate:** The step size $\eta$ controls how large each gradient descent step is.
The `'auto'` formula (sklearn ≥ 1.2):

$$\eta = \max\!\left(\frac{N}{\text{early\_exaggeration} \times 4},\ 50\right) = \max\!\left(\frac{N}{48},\ 50\right)$$

This ensures the learning rate scales with dataset size — a critical insight from Belkina et al.
(2019) and Kobak & Berens (2019). The old fixed default of 200 was systematically too small for
large $N$ and too large for small $N$.

### 2.6 Computational Complexity

The naive t-SNE algorithm requires $O(N^2)$ operations — computing all pairwise distances in
the input space, and all pairwise repulsive forces in the gradient. This becomes infeasible
for $N > 5{,}000$ or so.

| Method | Input similarity | Gradient | Memory | Practical limit |
|---|---|---|---|---|
| Exact | $O(N^2)$ | $O(N^2)$ | $O(N^2)$ | $N \leq 5{,}000$ |
| Barnes-Hut | $O(N \log N)$ | $O(N \log N)$ | $O(N \log N)$ | $N \leq 100{,}000$ |
| FIt-SNE | $O(N \log N)$ | $O(N)$ | $O(N + G\log G)$ | $N \leq 10{,}000{,}000$ |

**Barnes-Hut t-SNE** (van der Maaten, 2014) attacks both bottlenecks:
- **Input similarities:** Instead of computing all $N^2$ distances, build a vantage-point tree
  and find only the $k = 3 \times \text{perplexity}$ nearest neighbors for each point. Only these
  non-zero $p_{ij}$ values need to be stored.
- **Gradient:** The repulsive term $\sum_j q_{ij}(\mathbf{z}_i - \mathbf{z}_j)$ sums over all
  $N-1$ other points. The Barnes-Hut approximation (originally from N-body simulations in
  astrophysics) groups distant points into "summary cells" in a quadtree, replacing many
  point-level computations with a single cell-level computation. The `angle` parameter $\theta$
  controls the approximation quality: smaller $\theta$ = fewer cells, more accuracy, more compute.

**FIt-SNE** (Linderman et al., 2019) reduces the gradient computation further: it recognizes that
the repulsive sum is a **convolution** of the embedding point density with the Cauchy kernel.
Convolutions can be computed in $O(N)$ using the Fast Fourier Transform (after interpolating
onto a regular grid). This gives an approximately $O(N)$ gradient per iteration, regardless of $N$.

### 2.7 Implicit Assumptions and Where They Break

t-SNE makes several assumptions about your data:

1. **Euclidean distances are meaningful in the input space.** If your features are on wildly
   different scales, or if the natural similarity is cosine (text) or Jaccard (binary data),
   Euclidean distances are misleading. Always preprocess and consider the `metric` parameter.

2. **Local structure is the ground truth.** t-SNE explicitly optimizes local neighborhood
   preservation. Global structure (relative positions of distant clusters) is NOT preserved. If
   you need global structure, use PCA or UMAP.

3. **Manifold structure.** t-SNE assumes data lives on or near a low-dimensional manifold. If
   the data is uniformly distributed in high dimensions (no manifold structure), t-SNE will
   find spurious clusters — it will "force" structure that doesn't exist.

4. **Enough points.** With very few points ($N < 100$), the probability estimates are noisy
   and the resulting embedding is unreliable. At minimum, you need $N > 3 \times \text{perplexity}$.

> ⚠️ **Pitfall:** t-SNE is not scale-invariant by default. A feature measured in millions (e.g.,
> income in dollars) will dominate Euclidean distances compared to a feature measured in units
> (e.g., age in years). Always apply `StandardScaler` before t-SNE. The only exception: if you
> deliberately want certain features to dominate because of domain knowledge.

---

## §3 — The Full Evolution: Major Variants and Advancements

t-SNE spawned an entire lineage of improvements, each targeting a specific limitation of the
original algorithm. Understanding this lineage tells you which tool to reach for depending on
your dataset size, whether you need to embed new data, or whether you need interactive exploration.

### 3.1 Barnes-Hut t-SNE (2014) — Scaling to 100k Points

> 📜 **Origin:** Laurens van der Maaten, "Accelerating t-SNE using Tree-Based Algorithms,"
> *JMLR* 15(Oct):3221–3245, 2014. arXiv: https://arxiv.org/abs/1301.3342

**Problem solved:** The original $O(N^2)$ t-SNE is impractical for $N > 5{,}000$. Even modern
hardware cannot compute 25 million pairwise distances in reasonable time.

**Key algorithmic changes:**

*Input side (sparse P):* Build a vantage-point tree (VP-tree) on the input data, enabling
efficient nearest-neighbor search. For each point $i$, compute only the $k = \lfloor 3 \times \text{perplexity} \rfloor$ nearest neighbors, setting $p_{ij} = 0$ for all non-neighbors. This
reduces the similarity matrix from $O(N^2)$ dense to $O(N \cdot k)$ sparse.

*Output side (Barnes-Hut gradient):* The repulsive term in the gradient is:

$$F_{\text{rep}}(\mathbf{z}_i) = \sum_{j \neq i} q_{ij} \cdot q_{ij} \cdot Z \cdot (\mathbf{z}_i - \mathbf{z}_j)$$

This requires summing over all $N-1$ other points. The Barnes-Hut approximation: build a
quadtree (2D) or octree (3D) on the embedding points. For a cell whose points are all far from
$\mathbf{z}_i$ relative to the cell's size, replace all points in the cell with their center of
mass and treat them as a single "super-point." The approximation error is controlled by the
angle parameter $\theta$: a cell is summarized when $r_{\text{cell}} / d_{\text{cell}} < \theta$
where $r_{\text{cell}}$ is the cell's half-width and $d_{\text{cell}}$ is the distance from
$\mathbf{z}_i$ to the cell's center.

**When to prefer:** Default for N between 5,000 and 100,000. This is `sklearn`'s default
`method='barnes_hut'`. Beyond 100,000 points, FIt-SNE is faster.

**Python implementation:**

```python
# sklearn uses Barnes-Hut by default
from sklearn.manifold import TSNE
tsne = TSNE(method='barnes_hut', angle=0.5, n_jobs=-1)  # default
```

### 3.2 FIt-SNE (2019) — O(N) Gradient via FFT

> 📜 **Origin:** George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, and
> Yuval Kluger, "Fast Interpolation-Based t-SNE for Improved Visualization of Single-Cell RNA-Seq
> Data," *Nature Methods*, 16:243–245, 2019.
> URL: https://www.nature.com/articles/s41592-018-0308-4

**Problem solved:** Barnes-Hut reduces the gradient to $O(N \log N)$, but for $N > 100{,}000$,
this is still slow — a single iteration on 1 million points takes several seconds. FIt-SNE gets
the gradient computation to $O(N)$ per iteration.

**Key insight:** The repulsive part of the t-SNE gradient is a **convolution**:

$$F_{\text{rep}}(\mathbf{z}_i) \approx \int \frac{\mathbf{z}_i - \mathbf{z}}{(1 + \|\mathbf{z}_i - \mathbf{z}\|^2)^2} \cdot \hat{\rho}(\mathbf{z}) \, d\mathbf{z}$$

where $\hat{\rho}(\mathbf{z})$ is the (approximate continuous) density of embedding points.
Convolutions can be computed using the Fast Fourier Transform in $O(G \log G)$ where $G$ is the
grid size (independent of $N$), by:

1. Interpolating the $N$ discrete embedding points onto a regular $G \times G$ grid.
2. Computing the FFT of the grid-based density.
3. Multiplying pointwise by the FFT of the Cauchy kernel.
4. Inverse FFT to get the convolution result at all grid points.
5. Interpolating back from grid to the $N$ embedding points.

Since $G$ is fixed (chosen to give the desired approximation accuracy, typically a few hundred),
the total cost per iteration is $O(N) + O(G \log G) \approx O(N)$ for large $N$.

**Performance benchmark:**
- 200,000 cells: sklearn (Barnes-Hut) > 90 minutes; openTSNE (FIt-SNE) < 4 minutes (20x+ speedup)
- 1.3 million cells: FIt-SNE completes in hours; Barnes-Hut is impractical

**When to prefer:** Any dataset with $N > 100{,}000$. Standard choice in single-cell genomics.

**Python implementation (via openTSNE):**

```python
import openTSNE

# FIt-SNE backend (default in openTSNE for large N)
embedding = openTSNE.TSNEEmbedding(
    init,
    affinities,
    negative_gradient_method="fft",   # FIt-SNE
)
# or with sklearn-compatible API:
from openTSNE import TSNE as openTSNE
tsne = openTSNE(n_jobs=-1)   # auto-selects FIt-SNE for large N
X_emb = tsne.fit_transform(X)
```

> 🔧 **In practice:** openTSNE automatically selects between Barnes-Hut and FIt-SNE based on
> dataset size. You rarely need to specify this explicitly unless you want to benchmark the two.

### 3.3 Parametric t-SNE (2009) — Embedding New Data

> 📜 **Origin:** Laurens van der Maaten, "Learning a Parametric Embedding by Preserving Local
> Structure," *AISTATS 2009*, JMLR W&CP 5:384–391.
> URL: https://proceedings.mlr.press/v5/maaten09a.html

**Problem solved:** Standard t-SNE is **non-parametric** — it finds an embedding for the training
data $\mathbf{X}$, but cannot embed a new point $\mathbf{x}_\text{new}$ without re-running the
entire optimization. This makes t-SNE unusable for production systems, streaming data, or
train/test split evaluation.

**Key idea:** Train a neural network $f(\mathbf{x}; \boldsymbol{\theta}): \mathbb{R}^p \to \mathbb{R}^d$ to minimize the t-SNE cost function using backpropagation. The network learns a
parametric mapping that generalizes to new data.

**Original approach (2009):** Van der Maaten used a stack of restricted Boltzmann machines (RBMs)
pre-trained with contrastive divergence, then fine-tuned with t-SNE gradients. This was the state
of the art in 2009 but is now replaced by standard MLP training.

**Modern approach:** Use a standard MLP with standard backprop:

```python
import torch
import torch.nn as nn
import numpy as np

class ParametricTSNE(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500), nn.ReLU(),
            nn.Linear(500, 500), nn.ReLU(),
            nn.Linear(500, 2000), nn.ReLU(),
            nn.Linear(2000, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)

def tsne_loss(Y, P):
    """t-SNE cost (KL divergence) as a PyTorch loss."""
    # Compute pairwise squared distances in embedding
    sum_Y = (Y ** 2).sum(dim=1, keepdim=True)
    D = sum_Y + sum_Y.T - 2 * (Y @ Y.T)
    Q = (1 + D).pow(-1)
    Q = Q / Q.sum()
    Q = torch.clamp(Q, min=1e-12)
    return (P * torch.log(P / Q)).sum()
```

**Status in mid-2026:** No single stable, widely-maintained package for parametric t-SNE exists
in Python. Practitioners who need parametric mapping generally use:
1. **Parametric UMAP** (`umap-learn` with `parametric_umap=True`) — actively maintained,
   faster, arguably better global structure preservation.
2. **Custom PyTorch implementation** of the above pattern.

**When to prefer parametric t-SNE:** When you have a production requirement to embed new data
points into an existing t-SNE visualization, you need a reproducible mapping (same input → same
output), or you want to fine-tune an existing embedding.

### 3.4 openTSNE's Incremental Embedding — Practical Out-of-Sample Extension

> 📜 **Origin:** Poličar et al., "openTSNE: A Modular Python Library," *J. Stat. Software*
> 109(3):1–30, 2024. https://www.jstatsoft.org/article/view/v109i03

**Problem solved:** The same as parametric t-SNE — embedding new points — but without training
a neural network. openTSNE provides a practical approximation: given a fitted embedding, new
points can be placed by optimizing their positions while keeping the original embedding fixed.

**Mechanism:** For a new point $\mathbf{x}_\text{new}$:
1. Compute its affinities $p_{\text{new},j}$ to the original training points.
2. Initialize its embedding position (e.g., the weighted mean of its neighbors' positions).
3. Run a few iterations of gradient descent optimizing only $\mathbf{z}_\text{new}$, holding
   all original embedding points fixed.

This is faster than re-running the full t-SNE, but the result depends on the existing embedding.

**Python implementation:**

```python
from openTSNE import TSNE as openTSNE

# Fit on training data
tsne = openTSNE(perplexity=30, n_jobs=-1, random_state=42)
embedding_train = tsne.fit(X_train)   # note: .fit(), not .fit_transform()

# Embed new data into the existing embedding
embedding_test = embedding_train.transform(X_test)

# For finer control: prepare partial embedding and optimize
partial = embedding_train.prepare_partial(X_test, perplexity=5)
partial.optimize(n_iter=100, learning_rate=0.1)
embedding_new = np.array(partial)
```

> ⚠️ **Pitfall:** The `transform()` call in openTSNE is NOT equivalent to fitting fresh t-SNE on
> $X_\text{train} \cup X_\text{test}$. New points are placed relative to the training embedding
> only. If $X_\text{test}$ contains points very different from $X_\text{train}$, they may be
> poorly placed. Always check the output visually.

### 3.5 HSNE — Hierarchical SNE (2016) — Interactive Exploration at Scale

> 📜 **Origin:** Pezzotti et al., "Hierarchical Stochastic Neighbor Embedding," *Computer Graphics
> Forum (EuroVis 2016)*, 35(3):21–30, 2016.
> https://nicola17.github.io/publications/2016_hsne/preprint.pdf

**Problem solved:** Even FIt-SNE struggles with datasets of tens of millions of points. HSNE
addresses this by building a multi-resolution hierarchy, enabling interactive "zoom in" exploration.

**Key algorithmic idea:** HSNE builds a hierarchy of coarse-to-fine representations:
1. Randomly select $L$ "landmark" points that represent the full dataset.
2. Compute a t-SNE of the $L$ landmarks at the coarsest scale.
3. For each landmark, maintain an "area of influence" — the set of original points closest to it,
   computed using random walk-based proximity metrics.
4. When the user zooms into a cluster, recursively select new landmarks from within that cluster
   and compute a fresh t-SNE at finer scale.

**When to prefer:** Datasets with $N > 1{,}000{,}000$ where you need interactive exploration and
your goal is discovery (finding sub-populations) rather than a single static plot.

**Python availability:** HSNE is implemented in the C++ HDILib library and the Python
`cytosplore` package. As of mid-2026, Python integration requires some manual setup. Its primary
use is in specialized bioinformatics workflows.

### 3.6 opt-SNE (2019) — Automated Hyperparameter Calibration

> 📜 **Origin:** Belkina et al., "Automated Optimized Parameters for T-SNE," *Nature
> Communications* 10:5415, 2019. https://www.nature.com/articles/s41467-019-13055-y

**Problem solved:** The default t-SNE hyperparameters (especially learning rate) are
systematically wrong for datasets of very different sizes. The paper demonstrates that using
the correct learning rate dramatically improves embedding quality for $N > 10{,}000$.

**Key contribution:** opt-SNE uses real-time monitoring of the KL divergence curve during
optimization to:
1. Adaptively set `learning_rate = N / 12` (now the `'auto'` default in sklearn ≥ 1.2)
2. Dynamically extend the early exaggeration phase based on when the KL divergence plateaus

This paper's recommendations were essentially adopted as the new defaults in sklearn v1.2 and
in openTSNE. If you are using sklearn ≥ 1.2 with `learning_rate='auto'`, you are already
applying the core opt-SNE insight.

### Variants Summary Table

| Variant | Year | Key innovation | Gradient complexity | Python |
|---|---|---|---|---|
| SNE | 2002 | Probability-based neighborhoods | $O(N^2)$ | Historical only |
| t-SNE | 2008 | t-distribution + symmetrized P | $O(N^2)$ | `sklearn` (method='exact') |
| Barnes-Hut t-SNE | 2014 | VP-tree + Barnes-Hut gradient | $O(N \log N)$ | `sklearn` (default) |
| FIt-SNE | 2019 | FFT-accelerated gradient | $O(N)$ | `openTSNE` |
| Parametric t-SNE | 2009 | Neural network encoder | $O(N)$ train | Custom PyTorch |
| openTSNE incremental | 2019+ | Out-of-sample via partial optimization | $O(N_\text{new})$ | `openTSNE` |
| HSNE | 2016 | Multi-resolution hierarchy | Sublinear | `cytosplore` |
| opt-SNE | 2019 | Automated parameter calibration | $O(N \log N)$ | Adopted into sklearn/openTSNE defaults |

---

## §4 — Hyperparameters: The Complete Guide

t-SNE has fewer hyperparameters than many algorithms, but they interact in non-obvious ways and
their effects on the embedding can be dramatic. This section covers every parameter in sklearn's
TSNE API (v1.5.x), the mechanism behind each, and how to tune them.

### 4.1 `perplexity` — The Scale Parameter

**Default:** 30.0  
**Mechanism:** Sets the "effective number of neighbors" each point considers. For each point $i$,
the Gaussian bandwidth $\sigma_i$ is found via binary search so that the entropy of $P_i$ equals
$\log_2(\text{perplexity})$.

**What changes when you vary it:**
- **Too low (perplexity = 2–5):** Each point only considers 2–5 genuine neighbors. t-SNE
  produces many small, fragmented micro-clusters. Real structures get shattered. Noise looks
  like signal. Isolated points appear as their own clusters.
- **Optimal (perplexity = 5–50 for small-medium N):** Clean cluster structure with reasonable
  sub-structure visible within clusters.
- **Too high (perplexity = N/2):** Each point considers half the dataset as neighbors. Global
  structure is over-smoothed. Genuinely distinct clusters merge into a single blob.

**Tuning heuristics:**
- $N < 100$: perplexity $\approx \sqrt{N}$ or $N/5$
- $N = 100$–$10{,}000$: perplexity = 30–50 (default range)
- $N = 10{,}000$–$100{,}000$: perplexity = 50–200, or $\approx N/100$
- $N > 100{,}000$: perplexity = 500+ or use multi-scale kernels in openTSNE
- Hard constraint: perplexity must be strictly $< N$

> 🏆 **Best practice:** Run at 3–5 perplexity values and show all results side-by-side. Only
> interpret structure that appears consistently across perplexities. This is the single most
> important rule of responsible t-SNE use (Wattenberg et al., 2016).

**Interaction with `n_components`:** For 3D embeddings, you generally need slightly higher
perplexity than for 2D, because there is more "room" in 3D and fewer crowding artifacts.

### 4.2 `early_exaggeration` — Cluster Organization

**Default:** 12.0  
**Mechanism:** Multiplies all $p_{ij}$ by this factor during the first 250 iterations. Amplifies
attractive forces, causing cluster seeds to form rapidly.

**Effects:**
- **Too low (4–8):** Clusters form slowly and may not separate cleanly. Sub-structure may dominate
  over macro-structure.
- **Default (12):** Works well for the vast majority of datasets.
- **High (20–100):** Clusters become very tight with large gaps between them. Useful when you know
  there are real, well-separated clusters and you want them visually obvious. High values can
  obscure fine-grained sub-structure within clusters.

**Tuning:** Leave at 12 unless:
- Clusters are poorly separated → increase to 20–50
- Sub-structure within clusters is important and being hidden → decrease to 4–8

**Interaction with `learning_rate='auto'`:** The auto formula uses `early_exaggeration` in the
denominator: $\eta = \max(N / (\text{early\_exaggeration} \times 4), 50)$. Changing
`early_exaggeration` will change the auto learning rate. If you change one, be aware of the other.

### 4.3 `learning_rate` — Step Size

**Default:** `'auto'` (since sklearn 1.2)  
**Auto formula:** $\eta = \max(N / (\text{early\_exaggeration} \times 4),\ 50)$

For common dataset sizes:

```python
# Verify learning rate for your N
import numpy as np
for N in [500, 1000, 5000, 10_000, 50_000, 100_000]:
    lr = max(N / 12 / 4, 50)
    print(f"N = {N:8d}  →  learning_rate = {lr:.0f}")
# N =      500  →  learning_rate = 50
# N =     1000  →  learning_rate = 50
# N =     5000  →  learning_rate = 104
# N =    10000  →  learning_rate = 208
# N =    50000  →  learning_rate = 1042
# N =   100000  →  learning_rate = 2083
```

**Diagnostics for wrong learning rate:**
- **Too high:** Embedding looks like a ball or ring — all points roughly equidistant, no visible
  structure. The optimizer overshoots and equilibrium is achieved at maximum pairwise distance.
- **Too low:** One dense central mass with a few scattered outliers. Points are attracted toward
  the center but repulsion is too weak to spread them out.

> 🏆 **Best practice:** Always use `learning_rate='auto'` with sklearn ≥ 1.2. The only reason
> to override: if you are comparing results with pre-1.2 code (set `learning_rate=200.0`) or
> if you have a specific theoretical reason. For openTSNE, the equivalent is setting
> `learning_rate = max(N/12, 50)` manually.

### 4.4 `max_iter` — Number of Gradient Steps

**Default:** 1000 (renamed from `n_iter` in sklearn 1.5)  
**Minimum meaningful:** 500 (250 early exaggeration + 250 refinement)

**When to increase:**
- If the embedding looks different across different `random_state` values → not converged
- If `tsne.kl_divergence_` is still decreasing at iteration 1000 (check with `verbose=1`)
- For $N > 50{,}000$: use 2000–5000 iterations
- For very complex datasets with many sub-structures: up to 5000

**Early stopping:** `n_iter_without_progress` (default 300) stops the optimization if KL
divergence doesn't decrease by more than $10^{-5}$ for 300 consecutive iterations (checked every
50 iterations). In practice, this often triggers before `max_iter` is reached for well-behaved
data.

> ⚠️ **Pitfall:** If you are using sklearn < 1.5 and copy code using `max_iter=1000`, you need
> `n_iter=1000` instead. If you are using sklearn ≥ 1.5 and copy code using `n_iter=1000`, you
> will get a `DeprecationWarning`. Plan for sklearn 1.7 to remove `n_iter` entirely.

### 4.5 `init` — Initialization Strategy

**Default:** `'pca'` (since sklearn 1.2; was `'random'`)  
**Options:** `'random'`, `'pca'`, or an `ndarray` of shape `(n_samples, n_components)`

**`'pca'` (recommended):**
- Initializes embedding from the first `n_components` PCA components of $\mathbf{X}$,
  scaled to have standard deviation $10^{-4}$.
- Benefits: (1) preserves the coarse global structure from PCA, (2) much more reproducible
  across random seeds (the main variation comes from the stochastic gradient updates, not from
  a random starting point), (3) Kobak & Linderman (2021) showed that PCA init allows t-SNE to
  preserve global structure on top of local structure.
- Cannot be used with `metric='precomputed'` — raises `ValueError` because PCA cannot be
  computed from a precomputed distance matrix.

**`'random'`:**
- Samples each coordinate from $\mathcal{N}(0, 10^{-4})$.
- More exploratory — can escape local optima that PCA init might fall into.
- Required when `metric='precomputed'`.
- Less reproducible: different random seeds give qualitatively different embeddings even after
  convergence.

**Custom `ndarray`:**
- Provide your own initialization matrix of shape `(n_samples, n_components)`.
- Useful for: warm-starting from a previous embedding, initializing from UMAP, or adding new
  data to an existing embedding structure.

### 4.6 `method` and `angle` — Algorithm Backend

**`method`** (default: `'barnes_hut'`):
- `'barnes_hut'`: $O(N \log N)$, requires `n_components ≤ 3`. Use for $N > 5{,}000$.
- `'exact'`: $O(N^2)$, supports any `n_components`, supports sparse precomputed matrices.
  Use for $N ≤ 5{,}000$ or when you need `n_components > 3`.

**`angle`** (default: 0.5, only affects `method='barnes_hut'`):
- Controls the Barnes-Hut approximation granularity ($\theta$ in the original paper).
- $\theta = 0$: no approximation (exact, but then why use Barnes-Hut?).
- $\theta = 0.5$: the recommended default. Good speed/accuracy trade-off.
- $\theta \to 1$: very coarse approximation; fast but distorted embeddings.
- "Angle less than 0.2 has quickly increasing computation time; angle greater than 0.8 has
  quickly increasing error" (sklearn docs).
- **Tune this?** Almost never. The default 0.5 is robust. Only increase to 0.8 if you need
  maximum speed and can tolerate some distortion; only decrease to 0.2 if you need maximum
  accuracy and have time.

### 4.7 `metric` — Distance Metric in Input Space

**Default:** `'euclidean'`

This is often the most domain-critical parameter. Euclidean distance is appropriate when all
features are on similar scales and the geometry of the feature space is roughly isotropic. For
many real-world problems, this is not the case:

| Data type | Recommended metric | Reason |
|---|---|---|
| Continuous, same-scale features | `'euclidean'` (default) | Standard |
| Text / NLP embeddings (BERT, etc.) | `'cosine'` | Cosine captures semantic similarity |
| Gene expression (bulk RNA-seq) | `'correlation'` | Pearson correlation is the biological norm |
| Binary molecular fingerprints | `'jaccard'` | Tanimoto/Jaccard for set overlap |
| Tabular with scale differences | Preprocess → `'euclidean'` | StandardScaler first |
| Custom kernel matrix | `'precomputed'` | Pass your $N \times N$ matrix as X |
| Time series | Custom callable | DTW distance or similar |

> ⚠️ **Pitfall:** When using `metric='cosine'` or any non-Euclidean metric with sklearn's TSNE,
> you **must** set `init='random'` (or provide a custom ndarray). Setting `init='pca'` with a
> non-Euclidean metric raises a `ValueError` because PCA is not defined on the metric space
> (sklearn computes PCA from the raw features $\mathbf{X}$, not from the distance matrix).
> In openTSNE, you can initialize from PCA even with non-Euclidean metrics because PCA is
> computed separately.

### 4.8 `n_jobs` — Parallelism

**Default:** `None` (= 1 job)  
**Recommended:** `-1` (= use all available CPUs)

Parallelism applies to the **input similarity computation** (nearest-neighbor search), not to the
gradient computation itself. For `metric='euclidean'` with `method='exact'`, there is no speedup.
For large $N$ with `method='barnes_hut'`, setting `n_jobs=-1` can significantly reduce runtime.

### Tuning Playbook Table

| Hyperparameter | Default | Typical range | Golden rule | Too low effect | Too high effect |
|---|---|---|---|---|---|
| `perplexity` | 30 | 5–500+ | $\approx N/100$ for large N; always run 3+ values | Micro-clusters, fragmented, noise looks like structure | Over-smoothed, real clusters merge |
| `early_exaggeration` | 12 | 4–100 | Leave at 12; increase only if clusters poorly separated | Poor separation, sub-structure dominates | Clusters too tight, sub-structure hidden |
| `learning_rate` | `'auto'` | 10–5000 | Always use `'auto'` | Dense blob with outliers | Ball/ring with no structure |
| `max_iter` | 1000 | 500–5000 | 1000 for N<50k; 2000+ for larger | Not converged, seed-sensitive | More compute, usually fine |
| `angle` | 0.5 | 0.2–0.8 | Leave at 0.5 | Very slow | Distorted embedding |
| `init` | `'pca'` | `'pca'`, `'random'`, array | Use `'pca'` always except with `precomputed` | N/A | N/A |
| `n_jobs` | None | -1 | Set `-1` always | Slow for large N | No upper risk |

### Optuna Hyperparameter Search

```python
import optuna
from sklearn.manifold import TSNE, trustworthiness
import numpy as np

def tsne_objective(trial, X):
    perplexity = trial.suggest_float("perplexity", 5, min(200, len(X) - 1))
    early_exaggeration = trial.suggest_float("early_exaggeration", 4, 20)

    tsne = TSNE(
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate='auto',
        max_iter=500,       # use 500 for speed during search; refit with 2000 for best params
        init='pca',
        random_state=42,
        n_jobs=-1,
    )
    X_emb = tsne.fit_transform(X)
    return trustworthiness(X, X_emb, n_neighbors=10)   # maximize

study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: tsne_objective(trial, X), n_trials=20)
print(f"Best perplexity: {study.best_params['perplexity']:.1f}")
print(f"Best early_exaggeration: {study.best_params['early_exaggeration']:.1f}")

# Refit with best params and full iterations
best_tsne = TSNE(
    perplexity=study.best_params['perplexity'],
    early_exaggeration=study.best_params['early_exaggeration'],
    learning_rate='auto',
    max_iter=2000,
    init='pca',
    random_state=42,
    n_jobs=-1,
)
X_final = best_tsne.fit_transform(X)
```

> 🔧 **In practice:** Optuna for t-SNE is useful but slow — each trial runs the full t-SNE.
> Use `max_iter=500` during search, then refit with 2000 for the winner. Consider openTSNE
> for faster trials when $N$ is large.

---

## §5 — Strengths

### 5.1 Exceptional Local Structure Preservation

t-SNE is specifically designed to preserve the local neighborhood structure of your data. If two
points are genuine nearest neighbors in the high-dimensional space, they will almost certainly
appear close together in the t-SNE embedding. This property is quantified by the trustworthiness
score, and t-SNE consistently achieves near-perfect trustworthiness scores (>0.95 on clean data).

The mechanism: the KL divergence $\mathrm{KL}(P \| Q)$ is asymmetric — it penalizes very
heavily when $p_{ij}$ is high but $q_{ij}$ is low (a true neighbor separated in the embedding)
and penalizes less when $p_{ij}$ is low but $q_{ij}$ is high (a false neighbor). This
asymmetry biases the optimizer toward preserving true neighbors at the cost of occasionally
introducing false ones.

### 5.2 Reveals Non-Linear Manifold Structure

PCA finds the axes of maximum variance — it is optimal for linear structure. t-SNE finds
clusters and sub-clusters regardless of the shape of the manifold. Data living on a Swiss
roll, a torus, or a curved ribbon in high-D is correctly "unrolled" and separated by t-SNE.

This is why t-SNE became the standard in genomics: gene expression data lives on non-linear
manifolds (cellular differentiation pathways, transcriptional programs), and PCA flattens these
into an uninformative cloud while t-SNE reveals the curved structure.

### 5.3 Adaptive Density Handling

The per-point bandwidth $\sigma_i$ adapts to local density. A point in a dense cluster gets a
narrow Gaussian (only its closest neighbors contribute significantly), while an isolated point in
a sparse region gets a wide Gaussian (it must reach farther to find neighbors). This density
adaptation makes t-SNE equally informative in both dense and sparse regions — a critical feature
for data with wildly different local densities (typical in biology: common cell types vs. rare
cell types).

### 5.4 Striking, Interpretable Visualizations

The heavy-tail repulsion creates clear empty space between clusters. The resulting plots have a
visual clarity that makes them immediately compelling to non-experts. This is not just cosmetic:
the ability to communicate results to biologists, clinical teams, and business stakeholders is
a real practical strength.

### 5.5 Robustness to Hyperparameters

With modern defaults (`init='pca'`, `learning_rate='auto'`), t-SNE produces good results across
a wide range of datasets without manual tuning. The automatic bandwidth selection (via perplexity)
means you do not need to set $\sigma_i$ for each point manually. This robustness is part of why
t-SNE became so widely adopted — it "just works" for most visualization tasks.

---

## §6 — Weaknesses & Failure Modes

### 6.1 Non-Interpretable Inter-Cluster Distances

**Mechanism:** t-SNE optimizes LOCAL structure. The positions of clusters relative to each other
depend on the random initialization, the optimization trajectory, and the perplexity. Two clusters
that appear far apart in the embedding are NOT necessarily more different than two clusters that
appear close together.

**Detection:** Run t-SNE at multiple perplexities and multiple random seeds. If inter-cluster
distances change dramatically, they are not informative.

**Mitigation:** Never draw conclusions from cluster positions. Use `annotate=True` scatter plots
with centroids labeled. For global structure, use PCA or UMAP and report results explicitly.

### 6.2 Meaningless Cluster Sizes and Densities

**Mechanism:** The per-point bandwidth $\sigma_i$ normalizes local density. A dense cluster of
10 very similar points may appear the same size as a sparse cluster of 1,000 diverse points.
Cluster area in a t-SNE plot tells you nothing about the number of points in the cluster.

**Detection:** Always color by the number of points per cluster (use `plt.text()` to annotate
cluster centers with counts) rather than relying on visual area.

**Mitigation:** Always annotate clusters with sample counts. Never use cluster size to infer
population size.

### 6.3 Artifactual Structure from Random Noise

**Mechanism:** t-SNE's optimization always produces cluster-like structure, even from pure random
noise. The repulsive forces push all points away from each other, while the attractive forces pull
random nearby points together, resulting in apparent clusters from pure Gaussian noise (Wattenberg
et al., 2016, demonstrated this explicitly).

**Detection:** Run PCA on the same data. If PCA shows no structure (no clear separation in the
first 2 PCs), treat any t-SNE clusters with extreme skepticism.

**Mitigation:** Always compare t-SNE to PCA. Perform domain validation: do t-SNE clusters correspond
to known labels, experimental conditions, or other external ground truth?

### 6.4 Cannot Embed New Data (Non-Parametric)

**Mechanism:** t-SNE finds a specific embedding $\mathbf{Z}$ for the training data $\mathbf{X}$.
There is no parametric function $f: \mathbb{R}^p \to \mathbb{R}^d$ that can be applied to new
points $\mathbf{x}_\text{new}$ without re-running the full optimization.

**Detection:** The code fails at `tsne.transform(X_new)` — sklearn's TSNE has no `transform()`
method (only `fit_transform()`).

**Mitigation:**
- Use `openTSNE`'s incremental embedding for approximate out-of-sample extension.
- Use Parametric UMAP for a fully parametric, invertible mapping.
- Use PCA or UMAP if you need features for downstream ML (never use t-SNE features for ML models).

> ⚠️ **Critical pitfall:** Using t-SNE embeddings as features for a supervised model is almost
> always wrong. You cannot compute t-SNE for test data without the training data, creating a
> data leakage problem. Even ignoring leakage, t-SNE is optimized for visualization, not for
> preserving the information needed for prediction. Use PCA or UMAP for feature generation.

### 6.5 Computational Cost for Large N

**Mechanism:** Even with Barnes-Hut approximation, sklearn's t-SNE is $O(N \log N)$ per iteration
and requires many iterations. For $N > 100{,}000$, sklearn's implementation is impractically slow
(>1 hour).

**Detection:** You notice the progress bar moving very slowly, or the process hasn't finished
after 10+ minutes for $N < 100{,}000$.

**Mitigation:** Use openTSNE (FIt-SNE backend). For $N > 1{,}000{,}000$, use HSNE or subsample.

### 6.6 Sensitivity to Perplexity — Requires Multiple Runs

**Mechanism:** The topology of the t-SNE embedding (number of clusters, their sub-structure,
relative positions) changes substantially with perplexity. A single perplexity run can be
completely misleading.

**Detection:** Run at perplexities {5, 15, 30, 50, 100} and compare. If results differ
substantially, you have a perplexity-sensitivity problem.

**Mitigation:** Always run at multiple perplexities. Report the full range, not just the
"best looking" one. Only report structures that are consistent across all tested values.

### 6.7 Not Suitable for Global Structure or Downstream ML

This is not a bug — it is a feature of t-SNE's design. But practitioners frequently misuse t-SNE
for tasks it was not designed for:
- **"How different are cluster A and B?"** — Cannot be answered from t-SNE distances.
- **"How big is cluster C?"** — Cannot be answered from t-SNE area.
- **"Can I use t-SNE features to train a classifier?"** — No. Use PCA or UMAP.
- **"Does t-SNE preserve the ordering of clusters from left to right?"** — No.

> 💡 **Intuition:** Think of t-SNE as a high-fidelity microscope that reveals fine structure
> within its field of view, but cannot tell you how far apart two specimens are in the room.
> PCA is a wide-angle camera that preserves relative distances but cannot reveal fine details.
> Use the right instrument for the right question.

---

## §7 — What I Think About When Fitting

This is my actual mental checklist — the internal monologue I run before, during, and after every
t-SNE run. Internalize it and your embeddings will be more reliable and your interpretations more
honest.

### Before Fitting: Data Checks

**1. How large is my dataset?**
```
N < 1,000    → method='exact', perplexity = sqrt(N), max_iter=2000
N 1k–10k    → method='barnes_hut', perplexity=30–50, max_iter=1000
N 10k–100k  → openTSNE, perplexity ≈ N/100, n_jobs=-1
N > 100k    → openTSNE (FIt-SNE backend); if N > 1M consider HSNE
```

**2. How many dimensions?**
If $p > 50$: I apply PCA first, reducing to 50 components. This is not a shortcut — it is
standard practice recommended by every serious t-SNE paper. Reasons:
- Removes noise (high-variance components contain signal; low-variance ones often contain noise)
- Euclidean distances in high-D are dominated by random fluctuations; PCA removes this noise
- Dramatically speeds up the nearest-neighbor search in the input space
- The PCA initialization then uses these 50 PCs rather than the raw high-D data

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
if X.shape[1] > 50:
    X_input = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
else:
    X_input = X_scaled
```

**3. Is my data scaled?**
t-SNE uses Euclidean distances by default. If my features span different orders of magnitude,
I apply `StandardScaler`. If I'm using `metric='cosine'`, scaling matters less (cosine is
scale-invariant), but I still normalize to unit norm as a sanity check.

**4. Am I using the right distance metric?**
- Text data / NLP embeddings → `metric='cosine'`
- Gene expression → `metric='correlation'`
- Binary features → `metric='jaccard'`
- Default otherwise → `metric='euclidean'`

**5. Does my data have the right size relative to perplexity?**
perplexity must be strictly less than $N$. For $N = 50$, I can't use the default perplexity=30
without adjustment. I compute: `perplexity = min(30, N // 3)`.

### During Fitting: What I Monitor

I always run with `verbose=1` during development:

```python
tsne = TSNE(verbose=1, max_iter=2000, random_state=42)
X_emb = tsne.fit_transform(X_input)
```

**What I watch in the output:**
- The KL divergence printed every 50 iterations should decrease steadily.
- If it plateaus above 2.0 after 500 iterations, I consider: different perplexity, more
  iterations, or a different random seed.
- If it never decreases: learning rate is likely too high.

**The smell test after fitting:**
```python
kl = tsne.kl_divergence_
print(f"Final KL divergence: {kl:.4f}")
# < 0.5 → excellent
# 0.5–1.5 → good, typical range
# 1.5–3.0 → investigate: more iterations? different perplexity?
# > 3.0 → something is wrong
```

### After Fitting: The Diagnostic Checklist

**Step 1: Run at multiple perplexities (NON-NEGOTIABLE)**
```python
for perp in [5, 15, 30, 50, 100]:
    emb = TSNE(perplexity=perp, init='pca', random_state=42, n_jobs=-1).fit_transform(X_input)
    # Plot each
```
I only report structures that appear at ALL tested perplexities.

**Step 2: Check trustworthiness**
```python
from sklearn.manifold import trustworthiness
tw = trustworthiness(X_input, X_emb, n_neighbors=10)
print(f"Trustworthiness @ k=10: {tw:.3f}")
# < 0.85 → something is wrong (false neighbors being introduced)
# > 0.90 → good local preservation
# > 0.95 → excellent
```

**Step 3: Color by known labels or features**
The first thing I plot is the t-SNE colored by known ground truth (if available). If the known
classes are NOT separated: either the classes aren't real clusters in the data, the preprocessing
was wrong, or t-SNE needs different hyperparameters. I investigate all three.

**Step 4: Run multiple random seeds**
```python
embeddings = []
for seed in [42, 123, 456, 789, 1000]:
    emb = TSNE(perplexity=30, init='pca', random_state=seed).fit_transform(X_input)
    embeddings.append(emb)
# Compare: same macro-structure should appear (possibly rotated/reflected)
```
If the macro-structure (which groups cluster together) changes between seeds, the embedding has
not converged. Increase `max_iter`.

### Decision Tree: What I See → What I Do

```
Output = dense ball or ring (all points equidistant, no structure)
  → Learning rate too high. Use learning_rate='auto'.

Output = one dense blob + scattered outliers
  → Learning rate too low. Use learning_rate='auto'.

Output = 50+ tiny micro-clusters
  → Perplexity too low. Double or triple it.

Output = 1–2 large undifferentiated blobs
  → Perplexity too high. Halve it.

Output = structure changes dramatically between random seeds
  → Not converged. Increase max_iter to 2000–5000. Use init='pca'.

Output = clusters look right but domain knowledge says they're wrong
  → Wrong metric? Wrong preprocessing? Inspect what StandardScaler did.
  → Try cosine metric if features are embeddings.
  → Try comparing to UMAP on the same data.

Output = beautiful clusters but downstream ML doesn't improve
  → Expected! t-SNE is for visualization ONLY.
  → Switch to PCA (n_components=50) or UMAP for features.

Output = "I only have 200 points but I got 50 clusters"
  → Perplexity set too low (default 30 for N=200 is borderline).
  → Try perplexity = sqrt(200) ≈ 14.

Trustworthiness < 0.85
  → t-SNE is introducing many false neighbors.
  → Try lower perplexity (the current bandwidth is too wide, blending non-neighbors).
  → Check if the data has extreme outliers pulling bandwidths wide.
```

### The Three Things t-SNE Can and Cannot Tell You

**CAN tell you:**
- Which points are in the same local neighborhood
- Whether local neighborhoods form meaningful clusters
- The sub-structure within a cluster (at appropriate perplexity)

**CANNOT tell you:**
- How different two distant clusters are (inter-cluster distances are meaningless)
- How big a cluster is relative to another (cluster sizes are meaningless)
- The global ordering or topology of the data (perplexity-dependent)

Every time I present a t-SNE plot to a stakeholder, I state these three limitations explicitly.
Misinterpretation of t-SNE is far more common than misinterpretation of PCA, because t-SNE
looks so compelling that people believe it more than it deserves.

---

## §8 — Diagnostic Plots & Evaluation

t-SNE lacks a reconstruction error (it's not trying to reconstruct $\mathbf{X}$ from $\mathbf{Z}$).
Evaluation instead focuses on neighborhood preservation, sensitivity analysis, and the KL
divergence. Here is the full diagnostic suite with code.

### 8.1 KL Divergence (Direct from t-SNE)

```python
from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, init='pca', max_iter=1000, random_state=42, verbose=1, n_jobs=-1)
X_emb = tsne.fit_transform(X)

print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")
# Interpretation:
# < 0.5   → excellent embedding
# 0.5–1.5 → good (typical range for clean datasets)
# 1.5–3.0 → investigate: more iterations, different perplexity
# > 3.0   → something wrong: outliers, wrong scale, too few iterations
```

**Important caveat:** KL divergence is NOT comparable across datasets. Only use it to compare
different hyperparameter settings on the SAME dataset.

### 8.2 Trustworthiness Score

Trustworthiness (Venna & Kaski, 2006) measures: for each point $i$, are its $k$ nearest
neighbors in the embedding also true $k$-NN in the original space?

$$T(k) = 1 - \frac{2}{nk(2n - 3k - 1)} \sum_i \sum_{j \in \mathcal{U}_k(i)} (r(i,j) - k)$$

where $r(i,j)$ is the rank of point $j$ among $i$'s true neighbors (sorted by original distance),
and $\mathcal{U}_k(i)$ is the set of false neighbors (those in the embedding's $k$-NN but not
in the original $k$-NN). A score of 1.0 = perfect; typical good values > 0.90.

```python
from sklearn.manifold import trustworthiness

for k in [5, 10, 20]:
    tw = trustworthiness(X, X_emb, n_neighbors=k, metric='euclidean')
    print(f"Trustworthiness @ k={k:2d}: {tw:.4f}")
```

### 8.3 Perplexity Sensitivity Plot (MANDATORY)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, trustworthiness
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

perplexities = [5, 15, 30, 50, 100]
fig, axes = plt.subplots(1, len(perplexities), figsize=(5 * len(perplexities), 4))

for ax, perp in zip(axes, perplexities):
    tsne = TSNE(
        perplexity=perp, init='pca', max_iter=1000,
        random_state=42, n_jobs=-1
    )
    emb = tsne.fit_transform(X)
    tw = trustworthiness(X, emb, n_neighbors=10)

    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(f"perplexity={perp}\nTW={tw:.3f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("t-SNE Perplexity Sensitivity — Digits Dataset", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("tsne_perplexity_sensitivity.png", dpi=150, bbox_inches='tight')
plt.show()
```

**How to read it:** Structure that appears at ALL perplexities is real. Micro-clusters at perplexity
5 that merge into one blob at perplexity 50 are artifacts of the scale setting, not real
sub-populations.

### 8.4 Convergence Monitoring Plot

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Parse KL divergence from verbose output
# (sklearn 1.5.x does not expose per-iteration KL; use verbose=1 and parse stdout)
import io, sys

def run_tsne_verbose(X, max_iter=1000, **kwargs):
    """Run t-SNE with verbose=1 and capture KL divergence trace."""
    tsne = TSNE(max_iter=max_iter, verbose=1, **kwargs)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    X_emb = tsne.fit_transform(X)
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Parse KL divergence values from output lines like "Iteration 50: error = 4.2345"
    kl_values = []
    for line in output.split('\n'):
        if 'error' in line.lower() and '=' in line:
            try:
                kl = float(line.split('=')[-1].strip().split()[0])
                kl_values.append(kl)
            except (ValueError, IndexError):
                pass

    return X_emb, tsne.kl_divergence_, kl_values

X_emb, final_kl, kl_trace = run_tsne_verbose(
    X, max_iter=1000, perplexity=30, init='pca', random_state=42, n_jobs=-1
)

if kl_trace:
    iters = [50 * (i + 1) for i in range(len(kl_trace))]
    plt.figure(figsize=(8, 4))
    plt.plot(iters, kl_trace, 'b-o', markersize=4)
    plt.axvline(x=250, color='r', linestyle='--', label='End of early exaggeration')
    plt.xlabel("Iteration"); plt.ylabel("KL Divergence")
    plt.title("t-SNE Convergence Curve")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("tsne_convergence.png", dpi=150)
    plt.show()
```

**What "good" looks like:** A sharp drop during early exaggeration (first 250 iterations), then a
slower decline during normal optimization, finally plateauing. If the curve is still declining
steeply at the end, increase `max_iter`.

### 8.5 Neighborhood Preservation Score

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def neighborhood_preservation(X_orig, X_emb, k=10):
    """Fraction of k-NN in embedding that are also k-NN in original space."""
    nbrs_orig = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_orig)
    nbrs_emb  = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_emb)

    _, idx_orig = nbrs_orig.kneighbors(X_orig)
    _, idx_emb  = nbrs_emb.kneighbors(X_emb)

    preservation = 0.0
    for i in range(len(X_orig)):
        shared = len(set(idx_orig[i, 1:]) & set(idx_emb[i, 1:]))
        preservation += shared / k

    return preservation / len(X_orig)

for k in [5, 10, 20]:
    score = neighborhood_preservation(X, X_emb, k=k)
    print(f"{k}-NN preservation: {score:.3f}")  # 1.0 = perfect
```

### 8.6 Complete Diagnostic Dashboard

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, trustworthiness
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# Perplexity sweep with metrics
perplexities = [5, 10, 20, 30, 50, 100]
results = []
embeddings = {}

for perp in perplexities:
    tsne = TSNE(perplexity=perp, init='pca', max_iter=1000,
                random_state=42, n_jobs=-1)
    emb = tsne.fit_transform(X)
    tw  = trustworthiness(X, emb, n_neighbors=10)
    results.append({
        'perplexity': perp,
        'trustworthiness': tw,
        'kl_divergence': tsne.kl_divergence_,
    })
    embeddings[perp] = emb

df = pd.DataFrame(results)
print(df.to_string(index=False))

# --- Dashboard plot ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, perp in zip(axes, perplexities):
    emb = embeddings[perp]
    row = df[df.perplexity == perp].iloc[0]
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab10', s=8, alpha=0.7)
    ax.set_title(
        f"perplexity={perp}  |  TW={row.trustworthiness:.3f}  |  KL={row.kl_divergence:.3f}",
        fontsize=9
    )
    ax.set_xticks([]); ax.set_yticks([])

plt.colorbar(scatter, ax=axes[-1], label='Digit class')
plt.suptitle("t-SNE Diagnostic Dashboard — Digits Dataset", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("tsne_dashboard.png", dpi=150, bbox_inches='tight')
plt.show()
```

### 8.7 What Evaluation Metrics Cannot Tell You

These metrics measure **local** neighborhood preservation. They tell you nothing about:
- Whether inter-cluster distances are meaningful (they are not in t-SNE)
- Whether cluster sizes reflect population sizes (they do not)
- Whether the overall topology of the data is preserved (perplexity-dependent)

A t-SNE with trustworthiness = 0.98 but completely scrambled cluster ordering is still a valid
and potentially excellent visualization for local structure analysis. Trustworthiness is necessary
but not sufficient for good t-SNE results.

---

## §9 — Innovative Industry Applications

t-SNE's canonical use case — visualizing MNIST digits — undersells the algorithm. Here are eight
real-world applications, ranging from the well-known to the genuinely surprising.

### 9.1 Single-Cell Genomics — The Canonical Scientific Application

> 🏭 **Industry application:** Immunology, oncology, drug target discovery

CyTOF (Cytometry by Time-of-Flight) and scRNA-seq measure 40–60 protein markers or 20,000 genes
simultaneously in individual cells. A typical experiment produces 500,000–10,000,000 cells.
Researchers need to identify rare cell subpopulations: tumor-infiltrating lymphocytes, exhausted
T-cells, progenitor populations.

t-SNE became the standard tool for this after papers showed it revealed immune cell differentiation
trajectories that PCA completely missed. The Scanpy (Python) and Seurat (R) single-cell toolkits
both default to t-SNE/UMAP for their visualization step.

```python
# Standard scRNA-seq t-SNE pipeline (Scanpy)
import scanpy as sc

adata = sc.read_h5ad('cells.h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_pcs=50)
sc.tl.tsne(adata, use_rep='X_pca', perplexity=30)
sc.pl.tsne(adata, color=['cell_type', 'n_genes'], ncols=2)
```

**Why t-SNE (not PCA):** Cellular differentiation is a non-linear process. PCA projects cells
onto linear axes of variance, which flattens differentiation trajectories into uninformative
lines. t-SNE preserves the branching topology of these trajectories.

### 9.2 Drug Discovery — Chemical Space Mapping

> 🏭 **Industry application:** Pharmaceutical cheminformatics

Drug discovery requires navigating "chemical space" — the conceptual space of all drug-like
molecules, estimated at $10^{60}$ compounds. Medicinal chemists use t-SNE to:
- Visualize diversity in compound screening libraries
- Find unexplored regions of chemical space
- Cluster drugs by structural similarity and predict cross-reactivity

Molecular fingerprints (Morgan/ECFP, ~1024 bits) are naturally high-dimensional and binary.
t-SNE with Jaccard distance (Tanimoto similarity) preserves the chemically meaningful structure
that Euclidean distance misses.

```python
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.manifold import TSNE
import numpy as np

def get_ecfp4(smiles, n_bits=1024):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

X_fps = np.array([get_ecfp4(s) for s in smiles_list])  # (N_compounds, 1024)

tsne = TSNE(
    n_components=2,
    metric='jaccard',     # Tanimoto similarity for binary fingerprints
    init='random',        # required: PCA not available with non-Euclidean metric in sklearn
    perplexity=30,
    max_iter=1000,
    random_state=42,
)
X_chem_space = tsne.fit_transform(X_fps)
# Color by bioactivity, molecular weight, therapeutic area...
```

**Innovation — ChemPrint data splitting:** Some researchers use t-SNE clusters to define
train/test splits that maximize chemical diversity, testing whether models can truly generalize
beyond interpolation to novel chemical scaffolds.

### 9.3 NLP — Debugging and Interpreting Language Model Embeddings

> 🏭 **Industry application:** NLP engineering, model debugging

Modern language models (BERT, GPT, Sentence-Transformers) produce 768–1536 dimensional
embeddings. t-SNE is the standard tool for:
- Verifying that fine-tuned models have correctly separated semantic categories
- Debugging embedding drift after model updates
- Discovering unexpected clusters (domain confusion, demographic biases)

```python
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents, show_progress_bar=True)  # (N, 384)

tsne = TSNE(
    n_components=2,
    metric='cosine',   # cosine geometry of language model spaces
    init='random',     # init='pca' not available with cosine metric in sklearn
    perplexity=30,
    max_iter=1000,
    random_state=42,
)
emb_2d = tsne.fit_transform(embeddings)

# Interactive visualization with plotly
import plotly.express as px
fig = px.scatter(
    x=emb_2d[:, 0], y=emb_2d[:, 1],
    color=labels, hover_data={'text': documents},
    title="Sentence Embedding t-SNE",
)
fig.show()
```

### 9.4 Cybersecurity — Network Anomaly Detection

> 🏭 **Industry application:** Security operations, threat hunting

Network intrusion detection generates millions of connection events per day, each described by
40–80 features (packet sizes, timing, protocol flags). t-SNE is used to:
- Reveal cluster structure of normal traffic types
- Make anomalies visually obvious as isolated points or unusual clusters
- Allow security analysts to visually audit what clustering algorithms detected

The key insight: human analysts can visually inspect a 2D t-SNE plot and identify unusual clusters
far faster than reading tabular data. Recent work (APT-LLM, arXiv 2502.09385) uses t-SNE
visualization of LLM-generated security event embeddings to identify Advanced Persistent Threats
on the UNSW-NB15 and CICIDS2017 datasets.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

tsne = TSNE(
    perplexity=50,
    init='pca',
    max_iter=1500,
    random_state=42,
    n_jobs=-1,
)
X_2d = tsne.fit_transform(X_network_features)

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=anomaly_scores,
    cmap='RdYlGn_r',
    s=2, alpha=0.5,
    norm=plt.Normalize(vmin=0, vmax=1),
)
plt.colorbar(scatter, label='Anomaly Score (0=normal, 1=attack)')
ax.set_title('Network Traffic t-SNE — Color by Anomaly Score')
plt.savefig('network_tsne_anomaly.png', dpi=150)
```

### 9.5 Materials Science — Crystal Structure Phase Maps

> 🏭 **Industry application:** Computational materials science, battery research

ML models trained on crystal structure databases (Materials Project: ~150k crystals, AFLOW: ~3.5M)
learn high-dimensional representations of materials. t-SNE of these representations reveals:
- Which materials are structurally similar
- Unexplored regions of compositional space
- Phase diagram features in the learned representation

Recent work (arXiv 2312.13289) uses t-SNE on deep InfoMax representations of polymorphic
crystalline materials, with 2D plots colored by space group revealing clear structural clustering.

### 9.6 Financial Market Regime Analysis

> 🏭 **Industry application:** Quantitative finance, risk management

Financial markets exhibit "regime" behavior — periods of different statistical characteristics
(bull markets, bear markets, crisis periods). t-SNE of rolling correlation matrices reveals:
- When the current market state is entering an unusual regime
- How current conditions compare to historical periods
- Which historical crises were similar in their correlation structure

```python
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE

# Rolling correlation features from return series
def rolling_corr_features(returns_df, window=60):
    """Flatten upper triangle of rolling correlation matrix."""
    features = []
    for i in range(window, len(returns_df)):
        window_data = returns_df.iloc[i - window:i]
        corr = window_data.corr().values
        upper = corr[np.triu_indices(corr.shape[0], k=1)]
        features.append(upper)
    return np.array(features)

corr_features = rolling_corr_features(returns_df, window=60)

X_scaled = RobustScaler().fit_transform(corr_features)   # RobustScaler for outlier-heavy finance
tsne = TSNE(perplexity=30, init='pca', max_iter=1000, random_state=42)
regime_embedding = tsne.fit_transform(X_scaled)
# Color by VIX level, year, or recession indicator
```

### 9.7 Quality Control in Manufacturing — Sensor Fusion

> 🏭 **Industry application:** Industrial IoT, semiconductor manufacturing

Modern manufacturing equipment generates thousands of sensor readings per production cycle.
t-SNE of sensor data reveals:
- Whether a new batch of products is behaving like historical good batches
- The onset of equipment degradation (gradual drift in sensor clusters)
- Root cause patterns for defects (defective parts cluster together in t-SNE)

The non-parametric nature is sometimes a limitation here (can't embed new batches without
openTSNE's incremental embedding), but openTSNE's `transform()` solves this for monitoring.

### 9.8 Recommendation Systems — User Behavior Embedding

> 🏭 **Industry application:** E-commerce, streaming platforms

Collaborative filtering models (matrix factorization, neural CF) learn user and item embeddings
in high-dimensional latent spaces (typically 64–256 dimensions). t-SNE of user embeddings:
- Reveals behavioral segments (power users, casual browsers, bargain hunters)
- Helps product teams understand why certain recommendations are generated
- Enables manual auditing of fairness (do protected groups cluster separately?)

```python
from sklearn.manifold import TSNE
import plotly.express as px

# user_embeddings: (N_users, latent_dim) from your CF model
tsne = TSNE(
    n_components=2,
    perplexity=50,
    init='pca',
    max_iter=1000,
    random_state=42,
    n_jobs=-1,
)
user_2d = tsne.fit_transform(user_embeddings)

fig = px.scatter(
    x=user_2d[:, 0], y=user_2d[:, 1],
    color=user_segments,
    hover_data={'user_id': user_ids, 'avg_purchase': avg_purchases},
    title="User Behavior Embedding — t-SNE",
)
fig.write_html("user_tsne.html")
```

---

## §10 — Complete Python Worked Example

Three datasets, three different contexts, one complete pipeline each.

### 10.1 Example A: Digits Dataset (sklearn built-in)

The digits dataset is the go-to t-SNE benchmark: 1,797 samples, 64 features (8×8 pixel values),
10 classes (digits 0–9). Small enough to run quickly, complex enough to demonstrate real structure.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

# ── 1. Load and inspect ──────────────────────────────────────────────────────
digits = load_digits()
X, y = digits.data, digits.target
print(f"Shape: {X.shape}")          # (1797, 64)
print(f"Classes: {np.unique(y)}")   # [0 1 2 3 4 5 6 7 8 9]
print(f"Class counts: {np.bincount(y)}")

# Quick EDA: check for scale issues
print(f"Feature range: [{X.min():.1f}, {X.max():.1f}]")   # [0, 16]
# Pixels are already 0-16; StandardScaler is still good practice

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA preprocessing: 64 dimensions → 30 PCA components
# (64 < 50 threshold, so PCA is optional here, but let's show the pattern)
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance (30 components): {pca.explained_variance_ratio_.sum():.3f}")

# ── 3. t-SNE with modern defaults ─────────────────────────────────────────────
tsne = TSNE(
    n_components=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate='auto',
    max_iter=1000,
    init='pca',
    method='barnes_hut',
    angle=0.5,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
X_emb = tsne.fit_transform(X_pca)
print(f"\nEmbedding shape: {X_emb.shape}")          # (1797, 2)
print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")

# ── 4. Trustworthiness evaluation ─────────────────────────────────────────────
for k in [5, 10, 20]:
    tw = trustworthiness(X_pca, X_emb, n_neighbors=k)
    print(f"Trustworthiness @ k={k:2d}: {tw:.4f}")

# ── 5. Perplexity sensitivity sweep ───────────────────────────────────────────
perplexities = [5, 15, 30, 50, 100]
embs = {}
metrics = []

for perp in perplexities:
    t = TSNE(perplexity=perp, init='pca', max_iter=1000, random_state=42, n_jobs=-1)
    e = t.fit_transform(X_pca)
    embs[perp] = e
    metrics.append({
        'perplexity': perp,
        'trustworthiness@10': trustworthiness(X_pca, e, n_neighbors=10),
        'kl_divergence': t.kl_divergence_,
    })

df_metrics = pd.DataFrame(metrics)
print("\nPerplexity sweep results:")
print(df_metrics.to_string(index=False))

# ── 6. Visualization ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (perp, ax) in enumerate(zip(perplexities, axes)):
    emb = embs[perp]
    row = df_metrics[df_metrics.perplexity == perp].iloc[0]
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab10', s=10, alpha=0.8)
    ax.set_title(
        f"perplexity={perp}\nTW@10={row['trustworthiness@10']:.3f} | KL={row['kl_divergence']:.3f}",
        fontsize=10
    )
    ax.set_xticks([]); ax.set_yticks([])
    # Annotate cluster centroids with digit labels
    for digit in range(10):
        mask = y == digit
        cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
        ax.text(cx, cy, str(digit), fontsize=9, fontweight='bold', ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Use the last panel for the metrics table
axes[-1].axis('off')
table_data = df_metrics.round(4).values.tolist()
table = axes[-1].table(
    cellText=table_data,
    colLabels=df_metrics.columns.tolist(),
    loc='center', cellLoc='center'
)
table.auto_set_font_size(False); table.set_fontsize(9)
axes[-1].set_title("Perplexity Comparison Metrics", fontsize=10)

plt.colorbar(scatter, ax=axes[4], label='Digit class', fraction=0.03)
plt.suptitle("t-SNE Diagnostic Dashboard — Digits Dataset", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("tsne_digits_dashboard.png", dpi=150, bbox_inches='tight')
plt.show()
```

**What the digits embedding reveals:** At perplexity 30–50, the 10 digit classes form 10
well-separated clusters. Notably:
- Digits 1 and 7 (visually similar) are adjacent.
- Digit 9 often splits into two sub-clusters (the "closed" 9 vs. the "open" 9).
- Digits 4 and 9 are often nearby (they share a top-right stroke).

These are real structural features, not artifacts — they correspond to known visual similarity
patterns in handwritten digit recognition.

### 10.2 Example B: MNIST (Full 70k samples with openTSNE)

For the full 70,000-image MNIST dataset, sklearn is impractical. This is where openTSNE shines.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from openTSNE import TSNE as openTSNE

# ── 1. Load MNIST ─────────────────────────────────────────────────────────────
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_mnist, y_mnist = mnist.data, mnist.target.astype(int)
print(f"MNIST shape: {X_mnist.shape}")   # (70000, 784)

# ── 2. Preprocessing: Scale → PCA(50) ─────────────────────────────────────────
print("Preprocessing: StandardScaler + PCA(50)...")
X_scaled = StandardScaler().fit_transform(X_mnist.astype(np.float32))

pca = PCA(n_components=50, random_state=42)
X_pca50 = pca.fit_transform(X_scaled)
print(f"PCA(50) explained variance: {pca.explained_variance_ratio_.sum():.3f}")
print(f"Input to t-SNE: {X_pca50.shape}")   # (70000, 50)

# ── 3. openTSNE — sklearn-compatible API ──────────────────────────────────────
print("\nRunning openTSNE on 70k MNIST points...")
tsne = openTSNE(
    n_components=2,
    perplexity=700,        # ≈ N/100 for N=70k (Kobak & Berens recommendation)
    n_iter=1000,
    metric='euclidean',
    n_jobs=-1,
    random_state=42,
    verbose=True,
)
X_emb_mnist = tsne.fit_transform(X_pca50)
print(f"\nEmbedding shape: {X_emb_mnist.shape}")   # (70000, 2)

# ── 4. Visualization ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for digit in range(10):
    mask = y_mnist == digit
    ax.scatter(X_emb_mnist[mask, 0], X_emb_mnist[mask, 1],
               c=[colors[digit]], s=0.5, alpha=0.3, label=str(digit))

# Add digit labels at cluster centroids
for digit in range(10):
    mask = y_mnist == digit
    cx = X_emb_mnist[mask, 0].mean()
    cy = X_emb_mnist[mask, 1].mean()
    ax.text(cx, cy, str(digit), fontsize=18, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=2))

ax.legend(title='Digit', markerscale=10, loc='upper right')
ax.set_title("Full MNIST (70k) t-SNE via openTSNE\nperplexity=700, init='pca'", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("tsne_mnist_70k.png", dpi=150)
plt.show()
```

### 10.3 Example C: Single-Cell Style Demo with make_blobs

This demonstrates t-SNE's behavior on synthetic data — specifically, clusters of different
densities, which is common in biological data (rare cell types vs. common ones).

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, trustworthiness
from sklearn.decomposition import PCA

# ── 1. Simulate single-cell-like data ─────────────────────────────────────────
# 6 cell types: 2 rare (n=50), 2 medium (n=200), 2 common (n=500)
# Each type lives in 10-dimensional space (proxy for gene expression PCs)

np.random.seed(42)
n_features = 50  # simulate 50-dimensional gene expression PCs

cluster_configs = [
    # (n_samples, center_shift, std, cell_type)
    (500, [0]*n_features, 0.5, "Type A (common)"),
    (500, [5]*n_features, 0.5, "Type B (common)"),
    (200, [10]*n_features, 1.0, "Type C (medium)"),
    (200, [0, 5]+[0]*(n_features-2), 1.0, "Type D (medium)"),
    (50,  [15]*n_features, 0.3, "Type E (rare)"),
    (50,  [5, 15]+[0]*(n_features-2), 0.3, "Type F (rare)"),
]

X_parts, y_parts = [], []
for i, (n, center, std, name) in enumerate(cluster_configs):
    center_arr = np.array(center, dtype=float)
    X_part = np.random.randn(n, n_features) * std + center_arr
    X_parts.append(X_part)
    y_parts.extend([i] * n)

X_sc = np.vstack(X_parts)
y_sc = np.array(y_parts)
cell_type_names = [c[3] for c in cluster_configs]
print(f"Synthetic scRNA-seq data shape: {X_sc.shape}")   # (1500, 50)
print(f"Class sizes: {np.bincount(y_sc)}")                # [500, 500, 200, 200, 50, 50]

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
# Simulate log1p normalization (standard in scRNA-seq)
X_log = np.log1p(np.abs(X_sc))  # log1p transform
X_scaled = StandardScaler().fit_transform(X_log)

# PCA: 50D → 20D (already low, but show the pattern)
pca = PCA(n_components=20, random_state=42)
X_pca_sc = pca.fit_transform(X_scaled)

# ── 3. PCA baseline ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = plt.cm.tab10(np.linspace(0, 1, 6))

ax = axes[0]
for i, (n, _, _, name) in enumerate(cluster_configs):
    mask = y_sc == i
    ax.scatter(X_pca_sc[mask, 0], X_pca_sc[mask, 1],
               c=[colors[i]], s=10 if n > 100 else 30, alpha=0.7, label=name)
ax.set_title("PCA (first 2 PCs)", fontsize=11)
ax.legend(fontsize=7, loc='upper right')
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

# ── 4. t-SNE at two perplexities ──────────────────────────────────────────────
for ax, perp in zip(axes[1:], [10, 30]):
    tsne = TSNE(perplexity=perp, init='pca', max_iter=1000,
                random_state=42, n_jobs=-1)
    emb = tsne.fit_transform(X_pca_sc)
    tw = trustworthiness(X_pca_sc, emb, n_neighbors=10)

    for i, (n, _, _, name) in enumerate(cluster_configs):
        mask = y_sc == i
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[colors[i]], s=20 if n > 100 else 60, alpha=0.8, label=name)

    ax.set_title(f"t-SNE (perplexity={perp})\nTW@10={tw:.3f}", fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Synthetic Single-Cell Data: PCA vs t-SNE\n(rare types n=50, common types n=500)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("tsne_singlecell_synthetic.png", dpi=150, bbox_inches='tight')
plt.show()

# ── 5. Key lesson: rare cell types ─────────────────────────────────────────────
print("\nKey observation:")
print("PCA: Rare types (E, F, n=50) are barely visible — drowned out by common types.")
print("t-SNE: Rare types form distinct, clearly visible clusters DESPITE being outnumbered.")
print("This is why t-SNE (and UMAP) replaced PCA in single-cell genomics.")
print("\nNote: t-SNE cluster SIZES do not reflect n=50 vs n=500 — all clusters appear similar size.")
print("This is the adaptive bandwidth effect: smaller bandwidth for dense (common) clusters.")
```

**What this example teaches:**
- PCA drowns rare cell types in variance from common types.
- t-SNE's adaptive bandwidth gives rare types the same visual prominence as common types.
- BUT cluster sizes in the t-SNE are NOT proportional to cell counts — always annotate with
  actual numbers.

---

## §11 — When to Use t-SNE: Decision Guide

### 11.1 The Decision Flowchart

```
START: I need to visualize high-dimensional data
│
├─ Goal is visualization / exploratory EDA only?
│  └─ YES → t-SNE is a strong candidate. Continue ↓
│  └─ NO (need downstream ML features / global structure)
│     → Use PCA (n_components=2) or UMAP instead. Stop.
│
├─ N > 1,000,000?
│  └─ YES → Use HSNE or subsample + t-SNE or switch to UMAP
│  └─ NO → Continue ↓
│
├─ N > 100,000?
│  └─ YES → Use openTSNE (FIt-SNE). sklearn too slow.
│  └─ NO → Either sklearn or openTSNE will work. Continue ↓
│
├─ Need to embed new data after fitting (production system)?
│  └─ YES → Use openTSNE's incremental embedding OR Parametric UMAP. Stop.
│  └─ NO → Standard t-SNE. Continue ↓
│
├─ Data has known non-linear manifold structure (biology, text, vision)?
│  └─ YES → t-SNE is ideal. Use it.
│  └─ NO (data is genuinely linear, e.g., PCA shows clear structure already)
│     → PCA may be sufficient. Run both and compare. 
│
└─ Need reproducible, stable embeddings?
   └─ YES → Use init='pca' (modern default) + fixed random_state
   └─ NO → Either init is fine
```

### 11.2 Comparison Table: t-SNE vs Alternatives

| Criterion | t-SNE | UMAP | PCA | Isomap |
|---|---|---|---|---|
| Local structure preservation | Excellent | Excellent | Poor | Good |
| Global structure preservation | Poor | Fair | Excellent | Good |
| Speed (N=100k) | Slow (sklearn) / Fast (openTSNE) | Fast | Very fast | Slow |
| Embed new data | No (need openTSNE) | Yes | Yes | No |
| Deterministic | No (random init) | No | Yes | No |
| Hyperparameter sensitivity | High (perplexity) | Medium | Low | Medium |
| Interpretable parameters | Medium | Medium | High | Low |
| Best for | Visualization, biology, NLP | Viz + downstream ML | Noise removal, linear data | Geodesic geometry |
| Typical trustworthiness | 0.92–0.98 | 0.90–0.97 | 0.70–0.85 | 0.85–0.93 |

### 11.3 When NOT to Use t-SNE

- **You need downstream ML features.** t-SNE is not invertible and cannot embed new data.
  Use PCA (n_components=50) or UMAP.
- **You need to explain how much variance is captured.** t-SNE has no concept of explained
  variance. Use PCA.
- **Your dataset has N > 100,000 and no openTSNE.** sklearn TSNE will take hours. Install
  openTSNE or use UMAP.
- **You need to preserve global topology.** t-SNE scrambles inter-cluster distances. Use UMAP
  with `min_dist=0.0` and `metric='euclidean'`.
- **You want reproducible results across software updates.** t-SNE results can change between
  sklearn versions due to implementation changes. For production, use PCA or train a parametric
  model.
- **You have N < 50 samples.** t-SNE's probability estimates are too noisy to be reliable.
  Use PCA or MDS.

### 11.4 When t-SNE Beats UMAP

Despite UMAP's growing popularity, t-SNE has specific advantages:
- **Tighter cluster separation:** t-SNE's early exaggeration produces cleaner inter-cluster
  whitespace, making clusters more visually obvious to non-experts.
- **Better small-dataset behavior:** For N < 5,000, t-SNE and UMAP are comparable in quality,
  but t-SNE is more thoroughly studied and its failure modes are better understood.
- **Biological community standard:** Many bioinformatics tools and papers default to t-SNE;
  for comparability with published figures, t-SNE may be required.

---

## §12 — Top Papers to Study

Ranked from "start here" to "deep mastery."

### Start Here

**[1] The Original t-SNE Paper**
> Laurens van der Maaten and Geoffrey E. Hinton. "Visualizing High-Dimensional Data Using
> t-SNE." *Journal of Machine Learning Research*, 9(Nov):2579–2605, 2008.
> URL: https://jmlr.org/papers/v9/vandermaaten08a/vandermaaten08a.html

The essential foundation. Clean mathematics, accessible writing, demonstrates t-SNE on MNIST,
NYT text, and gene expression. Derive the gradient yourself after reading — it's within reach.
Estimated reading time: 3–4 hours for a thorough first reading.

**[2] How to Use t-SNE Effectively**
> Martin Wattenberg, Fernanda Viégas, and Ian Johnson. "How to Use t-SNE Effectively."
> *Distill*, 2016. URL: https://distill.pub/2016/misread-tsne/

Read this BEFORE you start using t-SNE in practice. Interactive visualizations demonstrate
exactly what happens at different perplexities, with different data shapes, and with random noise.
The six key lessons from this paper should be memorized. Estimated time: 1–2 hours.

### Read Next

**[3] Barnes-Hut t-SNE**
> Laurens van der Maaten. "Accelerating t-SNE using Tree-Based Algorithms." *JMLR*,
> 15(Oct):3221–3245, 2014. arXiv: https://arxiv.org/abs/1301.3342

Explains how the VP-tree and Barnes-Hut approximation reduce complexity from $O(N^2)$ to
$O(N \log N)$. Read this when you want to understand why the `angle` parameter exists and
what it controls. The description of the Barnes-Hut algorithm is clear and self-contained.

**[4] The Art of Using t-SNE for Single-Cell Transcriptomics**
> Dmitry Kobak and Philipp Berens. "The Art of Using t-SNE for Single-Cell Transcriptomics."
> *Nature Communications*, 10:5416, 2019.
> URL: https://www.nature.com/articles/s41467-019-13056-x

The definitive best-practices guide. Recommends PCA initialization, `learning_rate=N/12`,
and `perplexity≈N/100` for large datasets. These recommendations influenced the sklearn v1.2
defaults. Even if you never work with single-cell data, this paper is the most practical
t-SNE guide ever written.

**[5] Initialization is Critical for Preserving Global Data Structure**
> Dmitry Kobak and George C. Linderman. "Initialization Is Critical for Preserving Global
> Data Structure in Both t-SNE and UMAP." *Nature Biotechnology*, 39:156–157, 2021.
> URL: https://www.nature.com/articles/s41587-020-00809-z

Short (2 pages) but foundational. Proves that PCA initialization allows t-SNE to simultaneously
preserve global structure (inherited from PCA) and local non-linear structure. The theoretical
basis for sklearn's `init='pca'` default since v1.2.

### Deep Mastery

**[6] FIt-SNE**
> George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, and Yuval Kluger.
> "Fast Interpolation-Based t-SNE for Improved Visualization of Single-Cell RNA-Seq Data."
> *Nature Methods*, 16:243–245, 2019.
> URL: https://www.nature.com/articles/s41592-018-0308-4

The paper that made million-point t-SNE practical. Requires understanding of FFT and numerical
approximation. Dense but important for anyone working with large datasets. The key insight
(t-SNE gradient = convolution → FFT) is elegant and transferable to other algorithms.

**[7] Automated Optimized Parameters for t-SNE (opt-SNE)**
> Anna C. Belkina et al. "Automated Optimized Parameters for T-SNE." *Nature Communications*,
> 10:5415, 2019. URL: https://www.nature.com/articles/s41467-019-13055-y

The companion paper to Kobak & Berens 2019. Demonstrates that the default learning rate of 200
is systematically wrong for datasets with N > 5,000, and proposes the N/12 formula. Read this
to understand the theoretical basis of the `learning_rate='auto'` default.

**[8] openTSNE Library Paper**
> Pavlin G. Poličar, Martin Stražar, and Blaž Zupan. "openTSNE: A Modular Python Library for
> t-SNE Dimensionality Reduction and Embedding." *Journal of Statistical Software*, 109(3):1–30,
> 2024. URL: https://www.jstatsoft.org/article/view/v109i03

The paper describing the openTSNE implementation. Useful for understanding the modular API
design, the benchmarks against sklearn, and the incremental embedding algorithm. The most
practically useful "implementation paper" in the t-SNE literature.

**[9] Parametric t-SNE**
> Laurens van der Maaten. "Learning a Parametric Embedding by Preserving Local Structure."
> *AISTATS 2009*. URL: https://proceedings.mlr.press/v5/maaten09a.html

The original proposal for neural-network-based t-SNE. Historical interest and theoretical
foundation for parametric approaches. The modern replacement is Parametric UMAP (see
Sainburg et al. 2021 for the modern version).

**[10] HSNE — Hierarchical SNE**
> Pezzotti et al. "Hierarchical Stochastic Neighbor Embedding." *Computer Graphics Forum
> (EuroVis 2016)*, 35(3):21–30, 2016.
> https://nicola17.github.io/publications/2016_hsne/preprint.pdf

For those working with truly massive datasets (millions to hundreds of millions of points).
The hierarchical approach is conceptually elegant and practically impactful in mass cytometry
and other high-throughput biology applications.

---

## §13 — Resources & Further Reading

### Books

- **"Pattern Recognition and Machine Learning"** — Christopher Bishop (2006). Chapter 12
  covers dimensionality reduction including PCA and its probabilistic interpretation. No
  t-SNE (predates it), but essential background for understanding manifold methods.
- **"The Elements of Statistical Learning"** — Hastie, Tibshirani, Friedman (2nd ed., 2009).
  Chapter 14 covers unsupervised methods including self-organizing maps and spectral clustering.
  Free PDF at https://hastie.su.domains/ElemStatLearn/. Background for manifold methods.
- **"Python Machine Learning"** — Raschka & Mirjalili (3rd ed.). Chapter 13 covers
  dimensionality reduction with practical sklearn code. Good entry point for the sklearn API.

### Blog Posts and Tutorials

- **"How to Use t-SNE Effectively"** — Wattenberg, Viégas, Johnson (Distill, 2016).
  The single most important practical guide. Interactive. Required reading.
  https://distill.pub/2016/misread-tsne/

- **"Understanding t-SNE"** — Lilian Weng's blog.
  https://lilianweng.github.io/posts/2018-08-12-vae/ (covers latent space methods including
  t-SNE; comprehensive math with illustrations)

- **openTSNE Documentation** — The best practical reference for modern t-SNE usage.
  https://opentsne.readthedocs.io/en/stable/

- **openTSNE Algorithm Explanation** — Detailed walkthrough of the algorithm with visuals.
  https://opentsne.readthedocs.io/en/stable/tsne_algorithm.html

- **sklearn TSNE documentation** — Complete API reference with examples.
  https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html

- **sklearn trustworthiness documentation**
  https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.trustworthiness.html

- **Van der Maaten's t-SNE home page** — The original author's page with links to all papers,
  implementations, and FAQs. https://lvdmaaten.github.io/tsne/

### Courses

- **"Machine Learning"** by Andrew Ng (Coursera): Does not cover t-SNE but provides essential
  background on dimensionality reduction philosophy and PCA.
- **"Unsupervised Learning in Python"** (DataCamp): Practical sklearn-based introduction to
  clustering and DR including t-SNE.
- **"Single Cell RNA-seq Analysis"** (EMBL-EBI): Uses t-SNE and UMAP extensively in the
  context of bioinformatics. https://www.ebi.ac.uk/training/online/courses/

### Package Documentation Links

| Package | Documentation URL |
|---|---|
| sklearn TSNE | https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html |
| sklearn trustworthiness | https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.trustworthiness.html |
| openTSNE | https://opentsne.readthedocs.io/en/stable/ |
| openTSNE parameters | https://opentsne.readthedocs.io/en/latest/parameters.html |
| FIt-SNE GitHub | https://github.com/KlugerLab/FIt-SNE |
| Scanpy (scRNA-seq) | https://scanpy.readthedocs.io/en/stable/ |
| cuML t-SNE (GPU) | https://docs.rapids.ai/api/cuml/stable/ |

---

*Chapter complete. For the decision of which DR algorithm to use, see the course overview.md.*
*For the closest neighboring algorithm, see umap.md (UMAP was explicitly designed as a faster,
more globally-aware alternative to t-SNE).*









