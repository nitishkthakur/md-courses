# UMAP: Uniform Manifold Approximation and Projection — The Complete Masterclass

> **Why this algorithm matters:** When Leland McInnes posted arXiv:1802.03426 in February 2018,
> the dimensionality reduction world had a new ruler. Within 12 months, UMAP had replaced t-SNE as
> the default visualization tool for single-cell RNA sequencing — a field that routinely processes
> datasets of 500,000 to 30 million cells, each described by 20,000 gene expression measurements.
> The Human Cell Atlas, the most ambitious biological mapping project since the Human Genome Project,
> uses UMAP across more than 30 million cells. Drug discovery pipelines at AstraZeneca and Novartis
> use it to navigate chemical spaces of millions of molecular fingerprints. BERTopic, one of the most
> popular NLP libraries for topic modeling, has UMAP baked in as a core architectural component.
> What made all of this happen so fast? UMAP is genuinely fast (12x over t-SNE at 5 million points),
> principled in its theoretical foundations (Riemannian geometry and algebraic topology), and
> practically better than anything that came before it for large-scale nonlinear structure discovery.
> This chapter will make you a genuine UMAP expert — not just someone who calls `fit_transform()`.

---

## §0 — Python Ecosystem & Package Guide

Before the mathematics, you need to know the landscape. UMAP has a clear ecosystem hierarchy:
one canonical reference implementation (`umap-learn`) and a family of specialized tools built on
top of or alongside it. Unlike logistic regression, where you might genuinely choose between five
packages for different needs, 90% of UMAP use goes through `umap-learn`. The remaining 10% splits
between GPU acceleration (`cuml`) and production inference (`ParametricUMAP`). Let me map it all.

### Complete Package Table

| Package | Import | Version (mid-2026) | What it provides | License |
|---|---|---|---|---|
| `umap-learn` | `import umap` | 0.5.12 (April 2026) | Reference implementation: all variants (supervised, densMAP, parametric, aligned), full sklearn transformer API, custom metrics | BSD-3 |
| `cuml` (RAPIDS AI) | `from cuml.manifold import UMAP as cuUMAP` | 26.06.00 (June 2026) | GPU-accelerated UMAP — 60x speedup on NVIDIA H100, scales to 20M+ points | Apache 2.0 |
| `rapids-singlecell` | `import rapids_singlecell as rsc` | Active (2026) | Drop-in GPU replacement for scanpy scRNA-seq UMAP pipelines, integrates with AnnData | Apache 2.0 |
| `ParametricUMAP` (via umap-learn) | `from umap.parametric_umap import ParametricUMAP` | Included in 0.5.x | Neural network encoder for fast new-point inference; optional decoder for inverse transforms | BSD-3 |
| `AlignedUMAP` (via umap-learn) | `from umap.aligned_umap import AlignedUMAP` | Included in 0.5.x | Align multiple related embeddings (e.g., time-course snapshots) to be mutually comparable | BSD-3 |
| `pynndescent` | `import pynndescent` | 0.5.x | UMAP's internal approximate nearest neighbor backend — fast, pure Python | BSD-2 |
| `scanpy` | `import scanpy as sc` | ~1.10.x | High-level single-cell API wrapping umap-learn; `sc.tl.umap(adata)` | BSD-3 |
| `BERTopic` | `from bertopic import BERTopic` | ~0.16.x | Topic modeling library with UMAP as a core internal component | MIT |

### Top Picks — Recommendation Table

| Package | Best for | Modern/Legacy | Notes |
|---|---|---|---|
| **`umap-learn`** | Everything — the universal choice for research and production | **Modern, primary** | Maintained by Leland McInnes at TutteInstitute; regular updates; every new variant lands here first |
| **`cuml` (RAPIDS)** | Large datasets (>100k points) on NVIDIA GPUs | **Modern** | Near-zero API change from umap-learn; 60x speedup on H100; scales to 20M+ points |
| **`ParametricUMAP`** | Production inference, real-time embedding of new points | **Modern** | Requires TensorFlow; use when you need fast `transform()` at inference time |
| **`AlignedUMAP`** | Time-series of related datasets, longitudinal studies | **Modern, specialized** | Only reasonable choice for aligning multiple embeddings |
| **`scanpy`** | scRNA-seq, AnnData-based workflows | **Modern, domain-specific** | Wraps umap-learn; adds scRNA-specific preprocessing and visualization |

### The Same fit_transform() Across Top Packages

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load data — used throughout this chapter
digits = load_digits()
X, y = digits.data, digits.target           # (1797, 64)
X_scaled = StandardScaler().fit_transform(X)

# ── Option 1: umap-learn (the universal choice) ────────────────────────────
import umap

reducer_umap = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    init='spectral',
    random_state=42,
)
embedding_umap = reducer_umap.fit_transform(X_scaled)
print(f"umap-learn output shape: {embedding_umap.shape}")  # (1797, 2)

# ── Option 2: cuML (GPU, NVIDIA only) ──────────────────────────────────────
# from cuml.manifold import UMAP as cuUMAP
# reducer_gpu = cuUMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
# embedding_gpu = reducer_gpu.fit_transform(X_scaled)
# (Commented: requires CUDA-enabled GPU and cuML installation)
# Uncomment if running on an NVIDIA GPU with RAPIDS installed:
# pip install cuml-cu12   (or via conda: conda install -c rapidsai cuml)

# ── Option 3: ParametricUMAP (neural network encoder, TF required) ─────────
# from umap.parametric_umap import ParametricUMAP
# reducer_param = ParametricUMAP(n_components=2, n_training_epochs=5)
# embedding_param = reducer_param.fit_transform(X_scaled)
# new_embedding = reducer_param.transform(X_new)   # fast inference on new data
# (Requires: pip install umap-learn tensorflow)

# ── Option 4: scanpy (single-cell API, AnnData-based) ──────────────────────
# import scanpy as sc
# adata = sc.AnnData(X_scaled)
# sc.pp.neighbors(adata, n_neighbors=15)
# sc.tl.umap(adata, min_dist=0.1)
# embedding_sc = adata.obsm['X_umap']
```

> 💡 **Intuition:** Think of `umap-learn` as the reference C compiler — it's correct, well-tested,
> and is what everyone uses by default. `cuml` is the GPU-optimized compiler that produces
> faster binaries from the same source code. `ParametricUMAP` is the compiler that, instead of
> compiling directly, trains a model to predict how to compile anything — slower upfront, but
> instantaneous for each new inference request. Pick based on scale and deployment requirements.

> 🔧 **In practice:** Install umap-learn with: `pip install umap-learn`. For interactive
> visualization utilities: `pip install umap-learn[plot]` (adds datashader, holoviews, matplotlib
> extras). For ParametricUMAP: `pip install umap-learn tensorflow`. The core package only needs
> `numpy`, `scipy`, `scikit-learn`, `numba`, `tqdm`, and `pynndescent` — all lightweight.

### A Note on `umap` vs `umap-learn`

The PyPI package name is `umap-learn` (install: `pip install umap-learn`), but you import it as
`import umap`. This is a frequent source of confusion in documentation — searching for `import umap`
leads you to the umap-learn package page, not a package called "umap". There IS a separate
(unrelated) package called `umap` on PyPI — it is NOT the same thing. Always install `umap-learn`.

### GPU Acceleration: When It Matters

The GPU story for UMAP is compelling at scale:

| Dataset Size | CPU (umap-learn) | GPU (cuML on H100) | Speedup |
|---|---|---|---|
| 10,000 points | ~5 seconds | ~0.5 seconds | ~10x |
| 100,000 points | ~2 minutes | ~3 seconds | ~40x |
| 5,000,000 points | ~47 minutes | ~4 minutes | ~12x |
| 20,000,000 × 384 dim | ~10 hours | ~2 minutes | **~311x** |

For routine work on datasets under 100k points, CPU umap-learn is fast enough. At 500k+ points,
cuML becomes genuinely transformative.

```python
# GPU usage — near-zero code change
from cuml.manifold import UMAP as cuUMAP   # only this line changes

reducer = cuUMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42,
    # Additional cuML-specific params:
    build_algo='nn_descent',   # 'auto', 'brute_force_knn', or 'nn_descent'
    # device_ids=[0, 1]        # multi-GPU (if available)
)
embedding = reducer.fit_transform(X)  # X can be numpy or cupy array
```

> ⚠️ **Pitfall:** cuML UMAP does not support precomputed distance matrices or custom array
> initialization as of mid-2026. If you need these features, fall back to CPU umap-learn.
> Also note that cuML does not yet implement `densMAP`, `AlignedUMAP`, or `ParametricUMAP`.

---

## §1 — The Origin: The Paper That Started It All

> 📜 **Origin/Citation:** Leland McInnes, John Healy, James Melville. "UMAP: Uniform Manifold
> Approximation and Projection for Dimension Reduction." arXiv:1802.03426 (submitted February 9,
> 2018; revised September 18, 2020). DOI: https://doi.org/10.48550/arXiv.1802.03426

### The Problem Being Solved

In early 2018, the dominant nonlinear dimensionality reduction tool was t-SNE (van der Maaten &
Hinton, 2008). t-SNE was brilliant and beautiful — it produced stunning, biologically meaningful
visualizations of single-cell data that transformed how researchers thought about cellular identity.
But it had serious practical and theoretical problems.

**The practical problems of t-SNE circa 2018:**
1. **Scaling wall at ~100k points.** Barnes-Hut t-SNE runs in O(n log n) but with a large constant
   — 1 million points took hours on a CPU. The exploding size of single-cell atlases was making
   t-SNE impractical.
2. **No `transform()` method.** t-SNE is inherently a batch method — once you fit it, you cannot
   embed new points without re-running the full algorithm. Every new sample requires recomputing
   from scratch.
3. **Random initialization destroys global structure.** t-SNE's default random initialization
   means inter-cluster relationships are arbitrary. Two well-separated clusters in t-SNE might be
   genetically related; you simply cannot tell from the plot.
4. **Hyperparameter fragility.** The perplexity parameter was poorly understood, and embeddings
   with different perplexity values were qualitatively incomparable — a well-documented
   reproducibility problem.
5. **Only n_components = 2 or 3.** Empirically, t-SNE degrades catastrophically in higher
   dimensions. For downstream ML tasks where you need 10–50 dimensions, t-SNE is useless.

**The theoretical gaps:** The mathematical justification for t-SNE was largely empirical — it was
designed by intuition (use t-distributions to fix the crowding problem) and then shown to work
well. There was no principled theoretical framework explaining *why* it should preserve the
structure it does, or when it would fail.

McInnes, Healy, and Melville set out to solve all of this by grounding dimensionality reduction
in **Riemannian geometry and algebraic topology** — the mathematics of smooth curved spaces and
their higher-order combinatorial structure.

### The Core Insight

The key conceptual breakthrough is this: instead of asking "how do I project high-dimensional data
to low dimensions while preserving distances?", ask "how do I find a low-dimensional
topological space that has the same fuzzy topological structure as the high-dimensional data?"

This reframing is powerful for two reasons:
1. **Topological structure is more robust than distance.** A small deformation of a manifold
   changes all distances, but preserves the topology (connectivity, holes, clusters). If you match
   topology, you match the essential geometry without being slave to precise distances.
2. **Fuzzy topology has a principled optimization objective.** The distance between two fuzzy
   topological representations can be measured by cross-entropy — giving UMAP a clear, principled
   loss function derived from first principles, not heuristics.

The "U" in UMAP stands for "Uniform" — the theoretical derivation begins by assuming data is
**uniformly distributed on a Riemannian manifold**. This is the key assumption that allows each
data point to have its own locally adaptive metric: if data were truly uniform, then the distance
to the kth nearest neighbor encodes the local curvature of the manifold.

### Historical Context: What Came Before

To understand UMAP's contribution, it helps to see it in lineage:

| Year | Method | Key Innovation | Limitation |
|---|---|---|---|
| 2000 | Isomap | Geodesic distances on manifold | O(n³), no transform |
| 2000 | LLE | Local linear reconstruction | Poor global structure |
| 2003 | Laplacian Eigenmaps | Spectral graph theory | Eigendecomposition O(n³) |
| 2008 | t-SNE | Student-t to fix crowding | Slow, no transform, no global |
| 2018 | **UMAP** | Topology + fast optimization | Local density lost (fixable via densMAP) |

UMAP is directly informed by Laplacian Eigenmaps (its default initialization IS spectral/Laplacian
Eigenmaps), and its theoretical machinery draws on Spivak's work on fuzzy simplicial sets (2012)
and the category-theoretic formulation of manifold learning.

### The Original Algorithm, Stated Formally

UMAP's fitting procedure has five steps:

**Step 1: Compute k-nearest neighbors**
For each point $\mathbf{x}_i$, find its $k$ nearest neighbors using the chosen metric. Denote
the distance to the nearest neighbor as $\rho_i$ (used for local connectivity) and solve for
$\sigma_i$ such that:

$$\sum_{j=1}^{k} \exp\!\left(\frac{-(d(\mathbf{x}_i, \mathbf{x}_{ij}) - \rho_i)}{\sigma_i}\right) = \log_2(k)$$

This $\sigma_i$ is the local bandwidth — calibrated so each point always has exactly $\log_2(k)$
effective neighbors, regardless of local density.

**Step 2: Build the fuzzy simplicial set (high-dimensional graph)**
Compute directed edge weights:

$$w(i, j) = \exp\!\left(\frac{-(d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i)}{\sigma_i}\right)$$

Symmetrize via fuzzy union (the probabilistic OR of two asymmetric membership values):

$$\bar{w}(i, j) = w(i, j) + w(j, i) - w(i, j) \cdot w(j, i)$$

**Step 3: Initialize the low-dimensional embedding**
Initialize positions $\mathbf{z}_i \in \mathbb{R}^d$ using the Laplacian eigenmaps of the graph
(spectral initialization — default `init='spectral'`).

**Step 4: Construct the low-dimensional fuzzy set**
Define low-dimensional edge weights using a parametric family of curves:

$$v(i, j) = \frac{1}{1 + a \cdot \|\mathbf{z}_i - \mathbf{z}_j\|^{2b}}$$

where $a$ and $b$ are fit by nonlinear least squares to approximate the step function at
`min_dist` given `spread`.

**Step 5: Minimize cross-entropy via negative-sampling SGD**
Minimize:

$$\mathcal{L} = \sum_{(i,j) \in \text{positives}} \!\!\!\! \bar{w}(i,j) \log \frac{\bar{w}(i,j)}{v(i,j)} + (1 - \bar{w}(i,j)) \log \frac{1-\bar{w}(i,j)}{1-v(i,j)}$$

Positives are sampled proportional to $\bar{w}(i,j)$. For each positive, `negative_sample_rate`
(default 5) random pairs are drawn as negatives. This negative sampling makes the optimization
O(n log n) overall.

> 💡 **Intuition:** Imagine you have a map of a city. You don't need to know the exact distance
> between every pair of buildings — you just need to know which buildings are on the same block
> (connected in the local topology). UMAP builds a fuzzy version of this block-connectivity
> structure in high dimensions, then asks: "what is the lowest-dimensional map that preserves this
> same block structure?" The answer is the UMAP embedding.

---

## §2 — The Algorithm, Deeply Explained

### Mathematical Foundations: From Riemannian Geometry to Fuzzy Sets

Let me derive UMAP's machinery more carefully, because understanding the math unlocks every
practical tuning decision.

**The Riemannian Manifold Assumption**

UMAP's theoretical derivation starts with a bold assumption: the data $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\} \subset \mathbb{R}^p$ lies on (or near) a compact Riemannian manifold $\mathcal{M}$ of intrinsic dimension $d$, and the data is **uniformly distributed** on $\mathcal{M}$ according to the manifold's Riemannian metric.

This is the "U" in UMAP. The uniformity assumption does serious mathematical work:

If data were truly uniform on $\mathcal{M}$, then in regions of low extrinsic density, the manifold
must be highly curved or "stretched" — you need more space to fit the same number of uniform
samples. Conversely, in dense regions, the manifold is locally flat. This means the **distance to
the kth nearest neighbor** encodes the local geometry of the manifold at each point:

$$\rho_i = d(\mathbf{x}_i, \mathbf{x}_{i,1}) \quad \text{(distance to nearest neighbor)}$$
$$\sigma_i = \text{local bandwidth solving:} \quad \sum_{j=1}^{k} \exp\!\left(\frac{-(d(\mathbf{x}_i, \mathbf{x}_{i,j}) - \rho_i)}{\sigma_i}\right) = \log_2(k)$$

The bandwidth $\sigma_i$ is different for every point — UMAP is constructing a **locally adaptive
metric** where each point's local scale is calibrated to see exactly $k$ neighbors at approximately
unit distance. This is the fundamental reason UMAP handles different-density regions correctly:
it doesn't use a global bandwidth (like t-SNE's perplexity), it uses a local one.

The $\rho_i$ term (subtract nearest neighbor distance) ensures **local connectivity**: every point
is connected to its nearest neighbor with weight 1.0, guaranteeing the graph is connected.

**Fuzzy Simplicial Sets: The Topological Language**

Once we have local metrics, UMAP constructs a **fuzzy simplicial set** — a concept from algebraic
topology that generalizes ordinary graphs. In a regular graph, you have vertices (0-simplices)
and edges (1-simplices) with binary membership. In a simplicial complex, you also have
triangles (2-simplices), tetrahedra (3-simplices), etc. A **fuzzy** simplicial set allows
membership values in $[0, 1]$ rather than binary.

For UMAP's purposes, the relevant piece is the **1-skeleton** (the edge weights), because
optimizing in the embedding space with higher-order simplices would be computationally intractable.
The fuzzy 1-skeleton is the weighted graph $G_{\text{high}} = (V, E, \bar{w})$ where:

$$\bar{w}(i, j) = w(i, j) + w(j, i) - w(i, j) \cdot w(j, i)$$

This formula is the **fuzzy union** (probabilistic OR): if $A$ and $B$ are independent events,
$P(A \cup B) = P(A) + P(B) - P(A)P(B)$. It combines the directed weights (which may differ
depending on which point's local metric you use) into a single symmetric weight.

The result is a sparse, weighted graph where edge weight $\bar{w}(i,j) \in [0, 1]$ represents
"how strongly connected are points $i$ and $j$ in the high-dimensional manifold?"

> 💡 **Intuition:** Picture each data point with its own personal coordinate system, scaled
> to make its nearest neighbors appear "close." Points in dense clusters have tightly-packed local
> coordinates; points in sparse regions have stretched local coordinates. The fuzzy graph records
> who is close in whose local coordinate system, then merges the two perspectives (from $i$'s
> view and from $j$'s view) using the probabilistic OR. The result: within a tight cluster, edges
> have weight close to 1.0; between clusters, weights decay toward 0.

**The Low-Dimensional Representation**

For the embedding $\mathbf{Z} \in \mathbb{R}^{n \times d}$, UMAP constructs a corresponding
low-dimensional fuzzy set. The low-dimensional edge weights use a family of curves:

$$v(i, j) = \frac{1}{1 + a \cdot \|\mathbf{z}_i - \mathbf{z}_j\|^{2b}}$$

This is a generalized smooth step function. The parameters $a$ and $b$ are fit by nonlinear
least squares to match a target step function:

$$\psi(x) = \begin{cases} 1 & \text{if } x < \text{min\_dist} \\ \exp(-(x - \text{min\_dist}) / \text{spread}) & \text{otherwise} \end{cases}$$

For the default `min_dist=0.1, spread=1.0`, this gives approximately $a \approx 1.93, b \approx 0.79$.
The effect: pairs of points closer than `min_dist` get $v \approx 1$ (forced together), while
pairs farther than `spread` get $v \approx 0$ (allowed to separate). This is how `min_dist`
controls cluster packing: small `min_dist` = step function activates very close, so points pile
up together; large `min_dist` = step function activates far out, so points spread.

**The Cross-Entropy Objective**

The optimization target is the **cross-entropy** between the high-dimensional fuzzy set $\bar{w}$
and the low-dimensional fuzzy set $v$:

$$\mathcal{L}(\mathbf{Z}) = \sum_{(i,j)} \left[ \bar{w}(i,j) \log \frac{\bar{w}(i,j)}{v(i,j)} + (1 - \bar{w}(i,j)) \log \frac{1-\bar{w}(i,j)}{1-v(i,j)} \right]$$

This decomposes into two competing forces:
- **Attractive force** (first term): when $\bar{w}(i,j) > 0$ (connected in high-dim), pull
  $\mathbf{z}_i$ and $\mathbf{z}_j$ together to maximize $v(i,j)$.
- **Repulsive force** (second term): when $\bar{w}(i,j) \approx 0$ (disconnected in high-dim),
  push $\mathbf{z}_i$ and $\mathbf{z}_j$ apart to minimize $v(i,j)$.

This is structurally different from t-SNE's KL divergence:

$$\mathcal{L}_{\text{t-SNE}} = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

KL divergence is asymmetric: it heavily penalizes missing neighbors ($p_{ij}$ high but $q_{ij}$
low) and barely penalizes false neighbors ($p_{ij}$ low but $q_{ij}$ high). Cross-entropy
penalizes both more evenly through the two separate terms, which is one source of UMAP's better
balance between cluster tightness and cluster separation.

**The Damrich/Böhm Unification (2022/2023)**

A beautiful theoretical result by Damrich, Böhm, Hamprecht, and Kobak (ICLR 2023) showed that
both t-SNE and UMAP are instances of **contrastive learning**:
- t-SNE optimizes its objective via noise-contrastive estimation (NCE)
- UMAP optimizes its objective via negative sampling

The tighter, more compact clusters in UMAP versus t-SNE arise from a **normalization artifact**
introduced by negative sampling: the number of negative samples affects a normalization constant
that shrinks cluster size. This is not a fundamental difference in objective — it's an
optimization detail. The authors proposed a unified parameterization that smoothly interpolates
between t-SNE-like and UMAP-like embeddings.

**Why this matters for practitioners:** The "UMAP has better global structure than t-SNE" claim
is partially an artifact of initialization, not objective. With spectral initialization, UMAP's
global layout is set by Laplacian eigenmaps and SGD refines it; with random initialization, it's
comparable to t-SNE with random init. The choice of `init='spectral'` (the default) is therefore
not cosmetic — it's architecturally critical.

### Geometric Intuition: What Is UMAP Doing?

Here are three complementary analogies for understanding UMAP geometrically.

**Analogy 1: The Rubber Manifold**
Imagine your high-dimensional data lives on a rubber sheet embedded in 3D space (imagine the Swiss
roll dataset as a concrete example). UMAP grabs that rubber sheet and flattens it onto a 2D plane,
trying to preserve the "who-is-next-to-whom" relationships. Where the sheet was flat, the
flattening is exact. Where it was curved, something has to give — but UMAP tries to preserve the
local neighborhood structure first, sacrificing long-range distances if necessary.

**Analogy 2: The City Map**
UMAP is making a map. The goal isn't to get every GPS coordinate right — it's to ensure that
buildings on the same block stay on the same block, buildings in the same neighborhood stay
in the same neighborhood, and the overall district layout is recognizable. Exact inter-district
distances might be compressed or stretched, but the topology (connectivity, clusters, hierarchy)
is preserved.

**Analogy 3: The Fuzzy Friendship Graph**
Imagine you're at a conference. You build a fuzzy friendship graph: you give each person a
"friendship weight" with each of their closest contacts. Then you try to place everyone in a
2D room so that people with high friendship weights stand close to each other and strangers
stand far apart. UMAP is exactly this — the high-dimensional kNN graph is the friendship
graph, and the embedding is the room layout.

### The Transform Method and Out-of-Sample Extension

Unlike t-SNE, UMAP supports a `transform()` method for embedding new data points. The mechanism
is approximate: after fitting, the learned graph and embedding are stored. For a new point
$\mathbf{x}_{\text{new}}$, UMAP:
1. Finds its approximate nearest neighbors in the training data using pynndescent
2. Computes edge weights to those neighbors using the same sigma calibration
3. Places $\mathbf{x}_{\text{new}}$ in the embedding by optimizing its position to minimize
   the cross-entropy loss relative to its high-dimensional neighbors' positions

This is not a closed-form projection — it's a short optimization run. As a result:
- `transform()` is slower than a simple matrix multiply (unlike PCA's closed form)
- Results may vary slightly with `transform_seed`
- The quality degrades if $\mathbf{x}_{\text{new}}$ is far from the training distribution

For production inference where you need fast, consistent transforms, use **ParametricUMAP**.

### Computational Complexity

| Phase | Complexity | Bottleneck |
|---|---|---|
| kNN computation (pynndescent ANN) | O(n log n) | UMAP's ANN backend |
| Fuzzy graph construction | O(n · k) | Sparse operations |
| Spectral initialization | O(n · k) | Sparse eigendecomposition |
| SGD optimization | O(n · n_epochs · negative_sample_rate) | SGD iterations |
| **Total** | **O(n log n)** | kNN + SGD |
| `transform()` (new point) | O(k · log n) | ANN search + short SGD |

Compare to t-SNE: Barnes-Hut t-SNE is also O(n log n) in theory, but with a much larger
constant — the pairwise probability normalization requires O(n²) memory and a global sum
that UMAP avoids by using the sparse local graph.

### Implicit Assumptions and When They Break

UMAP assumes:
1. **Data lies near a manifold.** Breaks when data is pure noise (random cloud in high dimensions).
   Detection: trustworthiness will be low (~0.5–0.7); all n_neighbors values produce similar
   unstructured embeddings.
2. **The manifold is connected.** Breaks with genuinely isolated subpopulations in very sparse
   data. Detection: embedding fragments into disconnected islands regardless of n_neighbors.
3. **Local distances are meaningful.** Breaks when features are heterogeneous and unscaled.
   Fix: always standardize continuous features before fitting.
4. **k neighbors are sufficient to characterize local structure.** Breaks when intrinsic
   dimensionality is much higher than k. Rule of thumb: n_neighbors should be at least 2× the
   estimated intrinsic dimension.

---

## §3 — The Full Evolution: Major Variants

UMAP has spawned a rich family of variants since 2018, each addressing a specific limitation of
the original algorithm. Here is each one with its origin, mechanism, and use case.

### 3.1 Supervised UMAP

**Problem solved:** The original UMAP is purely unsupervised — it uses only the data matrix $X$,
ignoring any available label information. When labels exist, this is leaving discriminative signal
on the table.

**Origin:** McInnes & Healy, original umap-learn repository (2018), formalized in documentation
as a core feature. Not a separate paper — integrated into the canonical implementation.

**Algorithmic change:** When labels $y$ are provided, UMAP constructs *two* fuzzy simplicial sets:
1. $G_{\text{data}}$: the standard kNN graph on $X$ (data topology)
2. $G_{\text{label}}$: a graph on the label space (label topology), where same-label pairs have
   weight 1.0 and different-label pairs have weight 0.0 (for categorical labels)

These are combined via a weighted average controlled by `target_weight`:

$$G_{\text{combined}} = (1 - \text{target\_weight}) \cdot G_{\text{data}} + \text{target\_weight} \cdot G_{\text{label}}$$

The UMAP optimization then minimizes the cross-entropy to this combined graph.

**Effect:** Classes are pulled together (same-label points cluster tightly) and pushed apart from
other classes. The `target_weight` parameter controls the supervision strength:
- `target_weight=0.0`: purely unsupervised
- `target_weight=0.5` (default): balanced
- `target_weight=1.0`: purely label-driven (class structure only)

```python
import umap

reducer = umap.UMAP(
    n_neighbors=15,
    target_weight=0.5,          # balance data + label topology
    target_metric='categorical', # discrete labels
    random_state=42
)
embedding = reducer.fit_transform(X_scaled, y=labels)

# For continuous targets (regression):
reducer_reg = umap.UMAP(target_metric='euclidean', target_weight=0.7)
embedding_reg = reducer_reg.fit_transform(X_scaled, y=continuous_target)
```

**When to prefer:** When you have clean labels and want maximum class separation for downstream
classification, or when you want to use UMAP as a metric learning step followed by kNN
classification. For messy or noisy labels, keep `target_weight` low (0.3–0.5) or use the
semi-supervised variant.

**Metric learning interpretation:** Supervised UMAP at `target_weight` close to 1.0 is a
competitive metric learning method. After fitting, `reducer.transform(X_test)` projects new
points into the learned metric space. A kNN classifier in this space is often competitive with
more complex classifiers for multi-class problems.

### 3.2 Semi-Supervised UMAP

**Problem solved:** In practice, you rarely have labels for all data points. Semi-supervised UMAP
handles datasets where some points are labeled and some are not.

**Origin:** Integrated into umap-learn, following the sklearn semi-supervised convention.

**Algorithmic change:** Unlabeled points are marked with $y_i = -1$ (the sklearn unlabeled
convention). The label graph $G_{\text{label}}$ simply ignores all edges involving unlabeled
points — they are excluded from the label topology, contributing only through the data topology.

```python
import numpy as np

masked_labels = np.copy(labels)
unlabeled_idx = np.random.choice(len(labels), size=len(labels)//2, replace=False)
masked_labels[unlabeled_idx] = -1    # mark 50% as unlabeled

reducer = umap.UMAP(target_weight=0.5)
embedding = reducer.fit_transform(X_scaled, y=masked_labels)
```

**Effect:** Labeled points anchor the cluster structure; unlabeled points flow into their natural
topological position guided by both the data manifold and the partially-supervised signal.
This is especially powerful for rare-class discovery: if you have a few labeled examples of a
rare class, semi-supervised UMAP will tend to find related unlabeled points in the same region.

**When to prefer:** When you have partial labels and want to exploit them without discarding
unlabeled structure. Also effective for active learning: label a small batch, fit semi-supervised
UMAP, identify the next most informative points to label.

### 3.3 densMAP: Density-Preserving UMAP

**Problem solved:** Standard UMAP throws away local density information. Its normalization step
($\sigma_i$ calibration) makes all neighborhoods locally equivalent — a cluster of 10,000 densely
packed cells looks identical in the graph to a cluster of 100 widely scattered cells. Both
occupy approximately the same visual area in the UMAP plot. In single-cell biology, this is
scientifically misleading: researchers were incorrectly inferring transcriptional variability
from UMAP cluster "spread."

**Origin:** Narayan, Berger, Cho. "Assessing single-cell transcriptomic variability through
density-preserving data visualization." *Nature Biotechnology* 39, 765–774 (2021).
DOI: https://doi.org/10.1038/s41587-020-00801-7

**Algorithmic change:** densMAP adds a regularization term to the UMAP cross-entropy loss:

$$\mathcal{L}_{\text{densMAP}} = \mathcal{L}_{\text{UMAP}} + \lambda \sum_i \left(\log r_i^{\text{high}} - \log r_i^{\text{low}}\right)^2$$

where $r_i^{\text{high}}$ is the log-radius of point $i$'s neighborhood in the original space
and $r_i^{\text{low}}$ is the log-radius in the embedding. The regularizer penalizes the
squared difference between original-space and embedding-space neighborhood sizes, forcing
the embedding to preserve relative densities.

```python
import umap

reducer = umap.UMAP(
    densmap=True,
    dens_lambda=2.0,     # weight of density regularization (higher = stronger)
    dens_frac=0.3,       # fraction of epochs with density term active
    dens_var_shift=0.1,  # numerical stability constant
    output_dens=True,    # also return local radii
    random_state=42
)
result = reducer.fit_transform(X_scaled)
embedding = result[0]             # shape (n, 2)
radii_original = result[1]        # log-radii in original space
radii_embedded = result[2]        # log-radii in embedding

# Use radii_original to color by original space density
import matplotlib.pyplot as plt
plt.scatter(embedding[:,0], embedding[:,1], c=radii_original,
            cmap='RdYlBu_r', s=2, alpha=0.8)
plt.colorbar(label='Original space log-radius (lower = denser)')
plt.title('densMAP — density preserved in embedding')
plt.show()
```

**When to prefer:**
- **Always for single-cell genomics** when making claims about transcriptional diversity
- Any domain where original-space density carries semantic meaning (ecology, astronomy,
  financial tick data)
- When NOT to use: if you only care about topology/clustering structure and don't want
  density information influencing the layout

**Runtime overhead:** ~20% over standard UMAP. Well worth it when density interpretation matters.

> ⚠️ **Pitfall:** After running densMAP with `output_dens=True`, the return value is a tuple
> `(embedding, radii_high, radii_low)`, NOT just the embedding array. Old code that does
> `embedding = reducer.fit_transform(X)` will silently assign a tuple to `embedding` and fail
> on downstream operations. Always unpack explicitly.

### 3.4 Parametric UMAP

**Problem solved:** Standard UMAP's `transform()` is slow and approximate — it runs a short
optimization pass for each new point. For production systems embedding thousands of requests
per second (e.g., a recommendation system embedding new user interaction vectors in real time),
this is a bottleneck.

**Origin:** Tim Sainburg, Leland McInnes, Timothy Q Gentner. "Parametric UMAP Embeddings for
Representation and Semisupervised Learning." *Neural Computation* 33(11), 2881–2907 (2021).
DOI: https://doi.org/10.1162/neco_a_01410

**Algorithmic change:** Instead of optimizing the embedding positions $\mathbf{z}_i$ directly,
train a neural network encoder $f_\theta: \mathbb{R}^p \rightarrow \mathbb{R}^d$ to minimize
the same UMAP cross-entropy loss:

$$\mathcal{L}_{\text{parametric}} = \mathcal{L}_{\text{UMAP}}(f_\theta(\mathbf{x}_i), f_\theta(\mathbf{x}_j))$$

The encoder is trained with mini-batch SGD. After training, `transform(x_new) = f_theta(x_new)` —
a single forward pass through the network, taking microseconds.

```python
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf

# Default encoder (3-layer, 100-neuron fully connected)
embedder = ParametricUMAP(n_components=2, n_training_epochs=10, batch_size=1000)
training_embedding = embedder.fit_transform(X_train)

# Fast neural network inference for new points
new_embedding = embedder.transform(X_new)   # single forward pass — microseconds

# Save/load for production deployment
embedder.save("/models/umap_v1")
from umap.parametric_umap import load_ParametricUMAP
loaded_embedder = load_ParametricUMAP("/models/umap_v1")

# Custom architecture (e.g., for images)
encoder = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)  # n_components
])
image_embedder = ParametricUMAP(encoder=encoder, n_components=2)

# Autoencoder mode: add reconstruction loss
embedder_ae = ParametricUMAP(
    encoder=encoder,
    parametric_reconstruction=True,
    autoencoder_loss=True
)
```

**When to prefer:** When you need production inference (millisecond `transform()`), when you want
a shared encoder for multiple tasks (train UMAP + reconstruction jointly), or when you have a
natural architecture for your data type (CNN for images, transformer for text).

**PyTorch note:** The official implementation is TensorFlow/Keras only. A community PR for PyTorch
support exists (lmcinnes/umap#1103) but as of mid-2026 is not merged into main. For pure PyTorch
workflows, see `github.com/fcarli/parametric_umap`.

### 3.5 AlignedUMAP

**Problem solved:** For time-series of related datasets (e.g., single-cell snapshots from a
time course, or topic models across months), you want independent UMAP embeddings to be
comparable — the same cluster should appear in the same region across time steps.

**Origin:** Leland McInnes. "Improving UMAP for Longitudinal Data." Introduced in umap-learn 0.5
(2021) with accompanying documentation.

**Algorithmic change:** AlignedUMAP takes multiple datasets and a set of "relations" — dictionaries
mapping the index of a point in dataset $t$ to its corresponding index in dataset $t+1$. The
algorithm jointly optimizes all embeddings while penalizing positional disagreement between
corresponding points across time steps.

```python
from umap.aligned_umap import AlignedUMAP

# list_of_datasets: [X_t0, X_t1, X_t2, ...]
# relations: [{i: i for i in range(n_shared)}, ...]   # same point at adjacent times
relations = [{i: i for i in range(n_shared_points)}] * (n_timepoints - 1)

reducer = AlignedUMAP(n_neighbors=15, min_dist=0.1)
embeddings = reducer.fit_transform(list_of_datasets, relations=relations)
# embeddings: list of (n_i, 2) arrays, one per time step
```

**When to prefer:** Longitudinal studies where you want to animate the evolution of cluster
structure over time, or any scenario where multiple related datasets need a common coordinate
system.

### 3.6 GPU UMAP (cuML)

**Problem solved:** CPU umap-learn is too slow for very large datasets (>1M points) or for
interactive exploration of moderate datasets where you want to retune parameters in real time.

**Origin:** RAPIDS AI (NVIDIA) integrated UMAP into cuML (v0.14, 2020) as a first-class supported
algorithm, with major performance improvements in subsequent releases.

**Key differences from umap-learn:**
- Near-identical API (swap import only)
- Does NOT yet support: densMAP, ParametricUMAP, AlignedUMAP, precomputed distances
- Adds: `build_algo` ('nn_descent'), `device_ids` for multi-GPU, `hash_input` for consistent transforms

**When to prefer:** Any dataset >100k points on NVIDIA GPU hardware.

---

## §4 — Hyperparameters: The Complete Guide

UMAP has a rich hyperparameter space, but in practice, two parameters — `n_neighbors` and
`min_dist` — account for 90% of the tuning you will ever do. Let me cover every parameter
with the detail it deserves, then give you a tuning playbook.

### `n_neighbors` — The Most Important Parameter

**What it controls:** The size of the local neighborhood used to approximate the manifold
structure. This is the fundamental **local-vs-global tradeoff** knob. Small values focus on
micro-local structure; large values see broader topology.

**Mechanism:** `n_neighbors` controls $k$ in the kNN graph construction. A larger $k$ means
each node is connected to more points, the graph is denser, and the "local" neighborhoods overlap
more — giving the algorithm a broader view of the manifold. Smaller $k$ means sparser, more
local connections, and each node's manifold approximation is based on very few nearby points.

**Default:** 15 **Range:** 2–200 (practical: 5–50)

**What happens when too low (n_neighbors = 2–5):**
- Embedding fragments into many small, disconnected islands
- Every tiny local cluster becomes visible — the algorithm sees hyper-local structure
- Useful for: finding micro-clusters, detecting outliers
- Dangerous for: making global topological claims

**What happens when too high (n_neighbors = 100–200):**
- Clusters merge; broad topological blobs with gradients
- Looks like a smooth PCA-like projection
- Loses the fine-grained cluster structure that makes UMAP distinctive
- Useful for: smooth manifold visualization, preserving global topology

**Tuning heuristics by domain:**
- General purpose: start at 15, adjust based on domain expectations
- Single-cell RNA-seq: 10–30 (common: 15 for PBMC-scale datasets)
- NLP embeddings (BERT/GPT features): 25–50 (dense semantic spaces)
- Drug discovery (molecular fingerprints): 10–20
- Astronomy (galaxy morphology): 15–30
- Time series features (Catch22): 10–20

### `min_dist` — Cluster Packing Density

**What it controls:** The minimum distance between embedded points. Directly controls how
tightly points pack within clusters in the low-dimensional space.

**Mechanism:** `min_dist` and `spread` jointly determine parameters $a$ and $b$ of the
low-dimensional curve $v(i,j) = 1/(1 + a \cdot d^{2b})$. Small `min_dist` means the step
function activates at very short distances — points are allowed to pack extremely closely.
Large `min_dist` means the repulsive force prevents close packing even within clusters.

**Default:** 0.1 **Range:** 0.0–0.99 (practical: 0.0–0.5)

**What happens when too low (min_dist ≈ 0.0):**
- Very tight, compact clusters — ideal for downstream clustering algorithms
- Individual points within clusters overlap visually
- Best for HDBSCAN or k-means on the embedding
- Can look "blobby" to human eyes

**What happens when too high (min_dist ≈ 0.9):**
- Points spread approximately uniformly — loses cluster structure
- All points roughly equidistant; embedding looks like a filled circle
- The repulsion is so strong it destroys the attractive forces

**Tuning guidance:**
- For clustering tasks: 0.0–0.05 (tight packing for downstream clustering)
- For visualization/exploration: 0.1–0.3 (balanced view)
- For presentation: 0.3–0.5 (visually clear cluster separation)
- Fix `n_neighbors` first to get the right topology; then adjust `min_dist` for visual clarity

### `n_components` — Embedding Dimensionality

**Default:** 2 **Typical range:** 2–50

**The critical insight:** Use 2 for visualization; use 5–50 for downstream ML tasks. UMAP
does NOT degrade at higher dimensions the way t-SNE does. A UMAP(10) embedding preserves
substantially more structure than UMAP(2) and is often the best pre-processing step before
clustering or classification.

**Common workflow in scRNA-seq:**
```python
# Step 1: PCA(50) for computational efficiency
pca_embedding = PCA(n_components=50).fit_transform(X_normalized)

# Step 2: UMAP(10) for manifold structure — used for clustering
umap_high = umap.UMAP(n_components=10, random_state=42).fit_transform(pca_embedding)
clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
cluster_labels = clusterer.fit_predict(umap_high)

# Step 3: UMAP(2) for visualization only
umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(pca_embedding)
```

### `metric` — Input Space Distance

**Default:** 'euclidean' **Effect:** Determines how distances are computed in the original space.
Changing the metric is one of the most powerful levers for domain adaptation.

**Domain-specific guide:**
- **Continuous tabular data:** 'euclidean' (with StandardScaler preprocessing)
- **NLP/LLM embeddings (BERT, sentence-transformers):** 'cosine' — embeddings are normalized or near-unit sphere; cosine is the appropriate angle-based distance
- **Molecular fingerprints (drug discovery):** 'jaccard' — Jaccard on binary fingerprints equals Tanimoto coefficient, the industry standard for molecular similarity
- **Count data (bag-of-words, gene counts):** 'cosine' or 'hellinger'
- **Binary data:** 'hamming' or 'jaccard'
- **Probability distributions:** 'hellinger'

**Custom metric (must be numba @njit compiled):**
```python
from numba import njit
import umap

@njit
def tanimoto_dist(a, b):
    """Tanimoto distance for binary molecular fingerprints."""
    intersection = 0.0
    union = 0.0
    for i in range(len(a)):
        if a[i] or b[i]:
            union += 1.0
            if a[i] and b[i]:
                intersection += 1.0
    return 1.0 - (intersection / union) if union > 0.0 else 0.0

reducer = umap.UMAP(metric=tanimoto_dist, angular_rp_forest=False)
embedding = reducer.fit_transform(fingerprints)
```

### `init` — Initialization Strategy

**Default:** 'spectral' **Critical parameter — do not change without reason.**

| Init value | What it does | When to use |
|---|---|---|
| `'spectral'` | Laplacian eigenmaps — globally coherent start | **Default; always prefer this** |
| `'random'` | Random positions | Only if spectral init fails (disconnected graph) |
| `'pca'` | PCA scores as start | For very large datasets where spectral init is slow |
| `'tswspectral'` | Truncated SVD spectral | For sparse input matrices |
| Array | Custom positions | Advanced: warm-start from previous run |

> ⚠️ **Pitfall:** If you see code using `init='random'`, the resulting UMAP loses its global
> structure advantage and inter-cluster distances become meaningless. Kobak & Linderman (2021,
> Nature Biotechnology) proved this rigorously. Always use `init='spectral'` (the modern
> default since umap-learn 0.5).

### `n_epochs` and `learning_rate`

**n_epochs default:** None (auto: ~500 for n<10k, ~200 for larger)

When to increase: noisy embeddings, small n_neighbors, complex topology.
When to decrease: rapid prototyping (50 epochs gives a rough preview), large datasets.

**learning_rate default:** 1.0. Rarely needs tuning. If embedding is "frayed" after convergence,
try 0.5. Leave at default in 99% of cases.

### `negative_sample_rate`

**Default:** 5 (5 negative samples per positive edge per SGD step)

Higher = more repulsion between non-neighbors = better cluster separation but slower.
Increasing to 10–15 can help when clusters don't separate cleanly. Decreasing to 2–3 gives
speed at the cost of structure. The Damrich/Böhm paper showed that this parameter indirectly
controls the "t-SNE-ness" vs "UMAP-ness" of the embedding.

### `target_weight` (Supervised UMAP)

**Default:** 0.5 **Range:** 0.0–1.0

- 0.0 = purely unsupervised
- 0.5 = equal data + label topology
- 1.0 = purely label-driven
- For metric learning (maximum class separation): 0.8–1.0
- For exploration with some guidance: 0.3–0.5

### Tuning Playbook Table

| Goal | n_neighbors | min_dist | metric | init | Notes |
|---|---|---|---|---|---|
| General visualization | 15 | 0.1 | euclidean | spectral | Start here always |
| Tight clusters for HDBSCAN | 15 | 0.0 | euclidean | spectral | `min_dist=0` packs clusters |
| Broad topology (few clusters) | 30–50 | 0.1–0.3 | euclidean | spectral | Merge fine-grained clusters |
| Fine micro-structure | 5–10 | 0.05 | euclidean | spectral | More disconnected sub-clusters |
| NLP/sentence embeddings | 25–50 | 0.1 | cosine | spectral | Angle-based distance |
| Drug discovery | 10–20 | 0.1 | jaccard | spectral | Tanimoto-compatible |
| scRNA-seq (large atlas) | 15–30 | 0.3 | euclidean | spectral | PCA(50) → UMAP |
| High-dim ML features | 15 | 0.1 | euclidean | spectral | n_components=10–50 |
| Supervised (classification prep) | 15 | 0.0 | euclidean | spectral | target_weight=0.7–0.9 |
| Density-aware visualization | 15 | 0.1 | euclidean | spectral | densmap=True |

> 🏆 **Best practice:** For any new problem, run a quick sensitivity analysis: fit UMAP at
> n_neighbors ∈ {5, 15, 30, 50} and look at all four embeddings. If the topology (which groups
> exist, which connect to which) is stable across this range, your conclusions are robust.
> If it changes dramatically, you need domain knowledge to pick the right scale.

---

## §5 — Strengths

**1. Speed and Scalability**

UMAP runs in O(n log n) time with a small constant, making it practical for datasets that would
bring t-SNE to its knees. At 5 million points, UMAP takes ~47 minutes on CPU versus ~9.4 hours
for Barnes-Hut t-SNE — a 12x speedup. With cuML on an H100 GPU, the same dataset takes ~4
minutes. At 20 million points with 384 dimensions, cuML UMAP takes 2 minutes versus 10+ hours
for any CPU method. This is not a marginal improvement — it changes what is scientifically
feasible.

**2. Global Structure Preservation (with spectral initialization)**

With `init='spectral'` (the default), UMAP preserves both local and global structure
simultaneously. The spectral initialization uses Laplacian eigenmaps — a globally-coherent
embedding — as a starting point. SGD then refines this global layout while sharpening local
cluster boundaries. The result: inter-cluster relationships in the embedding reflect
inter-cluster relationships in the original space, something t-SNE (with random init)
cannot claim. Quantitatively: UMAP achieves ~89% global structure preservation versus ~71%
for t-SNE on standard benchmarks.

**3. Supports Out-of-Sample Extension via `transform()`**

Unlike t-SNE, UMAP supports `reducer.transform(X_new)` — embedding new points into an
existing fitted UMAP space without re-running the full algorithm. This enables production
workflows: fit once, embed forever. For even faster inference, ParametricUMAP replaces the
approximate graph search with a neural network forward pass.

**4. Flexible Distance Metrics**

UMAP accepts any metric via the `metric` parameter: Euclidean, cosine, Manhattan, Jaccard,
Hamming, Hellinger, Wasserstein (approximate), and arbitrary numba-compiled custom functions.
This is architecturally superior to t-SNE, which is hardcoded to Gaussian neighborhoods in
Euclidean space. The ability to use Jaccard distance for molecular fingerprints or cosine
distance for sentence embeddings makes UMAP domain-adaptable in a way few DR algorithms are.

**5. Higher-Dimensional Embeddings**

UMAP works well at n_components > 3, unlike t-SNE which degrades empirically above 3.
UMAP(10) or UMAP(50) is a legitimate preprocessing step for downstream clustering or
classification — one that often outperforms PCA because it captures nonlinear manifold structure.

**6. Supervised and Semi-Supervised Mode**

Supervised UMAP incorporates label information to produce discriminatively organized embeddings.
Semi-supervised UMAP handles partial labels (the sklearn `-1` convention). This turns UMAP
into a competitive metric learning method for classification tasks.

**7. Embedding Reproducibility**

With `random_state` set and `init='spectral'`, UMAP produces stable embeddings. Quantitatively,
the coefficient of variation of inter-cluster distances across different seeds is ~0.08 for
UMAP versus ~0.19 for t-SNE — UMAP is roughly 2.4x more reproducible.

---

## §6 — Weaknesses & Failure Modes

**1. Local Density Is Not Preserved by Default**

*Mechanism:* UMAP normalizes each point's local scale (the $\sigma_i$ calibration), making
all neighborhoods locally equivalent. Dense and sparse regions of the original space are
rendered with equal visual area in the embedding.

*Detection:* If you see clusters that should be biologically or semantically diverse appearing
compact, while sparse populations appear scattered, density may be misleading you.

*Mitigation:* Use `densmap=True`. Always question density-based conclusions from standard UMAP.

**2. Cluster Sizes and Distances Are Not Quantitatively Meaningful**

*Mechanism:* The number of pixels a cluster occupies in a UMAP plot has no quantitative
relationship to the number of data points in that cluster. Inter-cluster pixel distances
depend on `n_neighbors`, `spread`, and initialization — not on actual geodesic distances.

*Detection:* Run UMAP with different `n_neighbors`; if inter-cluster distances change
drastically, they were not meaningful to begin with.

*Mitigation:* Do not make quantitative comparisons between cluster sizes or inter-cluster
distances in UMAP plots. Use trajectory inference algorithms (Monocle, scVelo) on the kNN
graph, not on UMAP distances, for developmental ordering.

**3. Stochastic Variability**

*Mechanism:* SGD optimization with negative sampling introduces randomness. Without a fixed
`random_state`, two runs may produce mirror images, rotations, or topologically equivalent
but geometrically different embeddings.

*Detection:* Run twice with different random seeds and compare topology.

*Mitigation:* Always set `random_state=42` (or any fixed integer) for reproducible results.
Use `init='spectral'` (the default) — it reduces run-to-run variability compared to random
initialization.

**4. Parameter Sensitivity**

*Mechanism:* `n_neighbors` and `min_dist` can dramatically change the apparent topology.
n_neighbors=5 might show 20 clusters; n_neighbors=50 might show 3. This creates a risk of
"fishing" for the plot that matches your hypothesis.

*Detection:* If the number of visible clusters changes drastically across n_neighbors values,
the cluster structure may not be robust.

*Mitigation:* Always report the hyperparameters used. Show sensitivity analysis (embeddings
at multiple n_neighbors values). Validate clusters in original space, not in embedding space.

**5. Computationally Expensive for Sparse Data with Many Zeros**

*Mechanism:* For very sparse data (e.g., raw scRNA-seq count matrices with 90% zeros),
Euclidean distance is dominated by co-zero dimensions. The ANN computation with pynndescent
may be slow or imprecise.

*Detection:* Slow fit times; poor neighborhood preservation on sparse data.

*Mitigation:* Normalize and log-transform sparse count data first. Use PCA(50) as a
preprocessing step to create a dense, lower-dimensional representation before UMAP. For
sparse matrices with cosine metric, set `angular_rp_forest=True`.

**6. Does Not Work on Pure Noise**

*Mechanism:* UMAP assumes the data lies on a manifold. For random high-dimensional data,
there is no manifold — the kNN graph is essentially random, and the embedding will be an
uninformative blob.

*Detection:* Trustworthiness score near 0.5–0.6; embedding doesn't visually cluster.

*Mitigation:* Check whether your data has genuine structure before running UMAP. PCA on the
data should show some explained variance structure; if the scree plot is flat, there's no
manifold to find.

> ⚠️ **Pitfall:** Running HDBSCAN on a UMAP embedding and reporting "distinct clusters" is
> circular reasoning. UMAP is specifically designed to create tight, separated clusters — any
> clustering algorithm will find them. You must validate that clusters are distinct in the
> original high-dimensional space, not just in the UMAP plot. This is the most common misuse
> of UMAP in published biology papers.

---

## §7 — What I Think About When Fitting

*This is the internal monologue of a practitioner who has run UMAP hundreds of times. Follow
this checklist and you will avoid the most common failures.*

### Before Fitting: Data Due Diligence

**Question 1: What is the purpose of this embedding?**
- Visualization only? → `n_components=2`, prioritize visual clarity, tune `min_dist`
- Downstream clustering? → `n_components=10–20`, `min_dist=0.0`, use HDBSCAN on the embedding
- Downstream classification? → Supervised UMAP, `n_components=10–50`, evaluate with kNN accuracy
- Production inference? → ParametricUMAP, plan for serialization and deployment
- Density claims? → `densmap=True`, period.

**Question 2: Have I scaled my features?**
UMAP uses distances. If Feature A ranges 0–1 and Feature B ranges 0–100,000, Feature B
dominates all distances and you effectively have a 1D dataset. Always `StandardScaler()` for
continuous features before fitting. Exception: cosine metric on L2-normalized features — don't
double-normalize.

**Question 3: Are there missing values?**
UMAP does not handle NaN natively. Impute (`SimpleImputer`, `KNNImputer`) or mask missing
values before fitting.

**Question 4: Should I pre-reduce with PCA first?**
For datasets with > 200–500 features: yes. PCA(50) before UMAP:
- Dramatically speeds up pynndescent's ANN computation
- Removes noise dimensions that corrupt distance measurements
- This is the standard scRNA-seq pipeline: `PCA(50) → UMAP`

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# Preprocessing pipeline for high-dimensional data
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
embedding = umap.UMAP(n_neighbors=15, random_state=42).fit_transform(X_pca)
```

**Question 5: Which metric best represents my domain knowledge?**
Don't blindly use Euclidean. If you're working with text embeddings, cosine. If molecular
fingerprints, Jaccard. If probability distributions, Hellinger. The metric choice is
domain knowledge — treat it as such.

**Question 6: Do I have labels? Should I use supervised mode?**
If you have clean labels and want class separation: supervised UMAP. If partial labels: semi-
supervised. If no labels: unsupervised. The choice of how much label weight (`target_weight`)
to apply is a bias-variance tradeoff: high weight = tight class separation, may obscure
within-class structure.

**Question 7: How big is the dataset and what compute do I have?**
- < 50k points: CPU umap-learn, fast enough
- 50k–500k: CPU umap-learn (~minutes), or cuML if you have GPU
- 500k–5M: cuML strongly recommended
- > 5M: cuML with H100, or ParametricUMAP on a subset then transform remainder

### During Fitting: What to Monitor

Set `verbose=True` and watch for:
- Loss decreasing over epochs ✓ (healthy)
- Loss plateau after ~100 epochs ✓ (may be enough; try fewer `n_epochs` next time)
- Loss oscillating without convergence ✗ → try `learning_rate=0.5`
- Tqdm progress bar moving very slowly ✗ → check if pynndescent ANN step is the bottleneck
  (will say "Constructing knn graph"); consider PCA pre-reduction

### After Fitting: Smell Tests Before Trusting the Embedding

1. **Does the number of visible clusters match domain expectations?** If you have 10 known
   cell types and see 50 clusters, n_neighbors is too small. If you see 2 blobs, it's too large.

2. **Run the topology stability test:** Fit at n_neighbors ∈ {5, 15, 30, 50}. Show all four
   side by side. Ask: which clusters are stable across all values? Those are the real ones.

3. **Check trustworthiness** with `sklearn.manifold.trustworthiness`. A score > 0.90 means
   the local neighborhood structure is well-preserved. Below 0.85 is a warning sign.

4. **Check for isolated singleton points.** If you see individual points floating away from
   all clusters, they are likely outliers in the original space. Investigate them — they might
   be data quality issues or genuinely rare examples.

5. **Reproducibility check:** Run twice with `random_state=42` and `random_state=99`. The
   global layout should be similar (clusters in the same regions, same connectivity). If it
   looks completely different, you may need more `n_epochs` or spectral init.

6. **If publishing:** Report n_neighbors, min_dist, metric, init, random_state, and the
   software version. Show that cluster topology is stable across seeds. Never claim quantitative
   meaning from inter-cluster distances without using the kNN graph.

```python
# Quick stability check
import umap

configs = [5, 15, 30, 50]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, n_nb in zip(axes, configs):
    emb = umap.UMAP(n_neighbors=n_nb, random_state=42).fit_transform(X_scaled)
    ax.scatter(emb[:,0], emb[:,1], c=y, cmap='tab10', s=1, alpha=0.7)
    ax.set_title(f'n_neighbors={n_nb}')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('UMAP Topology Stability: n_neighbors Sensitivity', fontsize=14)
plt.tight_layout()
plt.show()
```

**Decision tree: if I see X, I do Y**

| Observation | Likely cause | Action |
|---|---|---|
| Many tiny disconnected islands | n_neighbors too small | Increase to 15–30 |
| All clusters merged into one blob | n_neighbors too large | Decrease to 10–15 |
| Clusters present but overlapping | min_dist too large | Decrease min_dist |
| Points piled up; can't see structure | min_dist too small | Increase min_dist |
| Good structure but looks noisy | n_epochs too small | Increase to 500 |
| Known classes not separating | Use supervised UMAP | Set target_weight=0.7 |
| Claiming density from plot | Missing density info | Use densmap=True |
| Production inference too slow | Standard UMAP transform | Switch to ParametricUMAP |

---

## §8 — Diagnostic Plots & Evaluation

Understanding whether your UMAP embedding is any good requires going beyond "it looks nice."
Here is a complete evaluation toolkit.

### Metric 1: Trustworthiness (Local Structure)

Trustworthiness measures the proportion of k-nearest neighbors in the embedding that were
also k-nearest neighbors in the original space. It penalizes "false neighbors" — points that
became close in the embedding but were distant in the original space.

**Range:** [0, 1]. Values > 0.90 are good; > 0.95 is excellent. Random embedding ≈ 0.5.

```python
from sklearn.manifold import trustworthiness
import numpy as np

# Compute trustworthiness (O(n²) — use subset for large datasets)
n_eval = min(5000, len(X_scaled))
idx = np.random.choice(len(X_scaled), n_eval, replace=False)

trust_score = trustworthiness(
    X_scaled[idx],
    embedding[idx],
    n_neighbors=15
)
print(f"Trustworthiness (k=15): {trust_score:.3f}")
```

**Limitation:** Trustworthiness can be > 0.95 even when global distances are severely distorted.
Use it alongside global structure metrics.

### Metric 2: Continuity (Reverse Trustworthiness)

Continuity measures whether k-nearest neighbors in the original space remain close in the
embedding. Penalizes "missing neighbors" — points that were close originally but became distant.

```python
# Install: pip install pyDRMetrics
from pyDRMetrics import DRMetrics   # verify current package name

# Use subset for tractability
metrics = DRMetrics(X_scaled[idx], embedding[idx])
print(f"Trustworthiness: {metrics.T:.3f}")
print(f"Continuity:      {metrics.C:.3f}")
print(f"LCMC:            {metrics.LCMC:.3f}")   # Local Continuity Meta-Criterion
```

> 💡 **Intuition:** Trustworthiness says "are the neighbors in the embedding real neighbors in
> original space?" Continuity says "are the real neighbors still neighbors in the embedding?"
> Both should be high. An algorithm can score high on one while failing the other.

### Metric 3: kNN Classification Accuracy (Structural Quality)

The most directly useful metric for labeled data: does the embedding preserve the
discriminative structure that a kNN classifier needs?

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

k = 15

# kNN accuracy in original space
clf_orig = KNeighborsClassifier(n_neighbors=k)
acc_orig = cross_val_score(clf_orig, X_scaled, y, cv=5, scoring='accuracy').mean()

# kNN accuracy in embedding
clf_emb = KNeighborsClassifier(n_neighbors=k)
acc_emb = cross_val_score(clf_emb, embedding, y, cv=5, scoring='accuracy').mean()

retention = acc_emb / acc_orig
print(f"Original space kNN accuracy (k={k}): {acc_orig:.3f}")
print(f"Embedding kNN accuracy    (k={k}): {acc_emb:.3f}")
print(f"Structural retention:              {retention:.1%}")
```

> 🏆 **Best practice:** Aim for retention > 90%. If the embedding retains > 90% of the kNN
> accuracy of the original space, it's preserving discriminative structure well.

### Metric 4: Geodesic Distance Correlation (Global Structure)

The gold-standard test for global structure: does the embedding preserve long-range distances?

```python
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import numpy as np

# Sample for tractability
n_sample = 1000
idx = np.random.choice(len(X_scaled), n_sample, replace=False)
X_sample = X_scaled[idx]
emb_sample = embedding[idx]

# Approximate geodesic distances via graph shortest path
G = kneighbors_graph(X_sample, n_neighbors=15, mode='distance')
geo_dist = shortest_path(G, method='D')

# Replace inf (disconnected) with max finite distance
finite_mask = np.isfinite(geo_dist)
geo_dist[~finite_mask] = geo_dist[finite_mask].max()

# Embedding Euclidean distances
emb_dist = pairwise_distances(emb_sample)

# Correlation (upper triangle only)
triu_idx = np.triu_indices(n_sample, k=1)
corr, _ = spearmanr(geo_dist[triu_idx], emb_dist[triu_idx])
print(f"Geodesic-Embedding Spearman correlation (global structure): {corr:.3f}")
```

### Diagnostic Plot Suite

```python
import umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Load data
digits = load_digits()
X, y = digits.data, digits.target
X_scaled = StandardScaler().fit_transform(X)

# Fit UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_scaled)

# ── Plot 1: Main embedding colored by class ────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :2])
sc = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='tab10',
                 s=8, alpha=0.8)
plt.colorbar(sc, ax=ax1, label='Digit class')
trust = trustworthiness(X_scaled, embedding, n_neighbors=15)
ax1.set_title(f'UMAP Embedding — Digits Dataset\nTrustworthiness (k=15): {trust:.3f}',
              fontsize=13)
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')

# ── Plot 2: n_neighbors sensitivity ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
trust_scores = []
nn_values = [5, 10, 15, 20, 30, 50]
for nn in nn_values:
    emb_nn = umap.UMAP(n_neighbors=nn, random_state=42).fit_transform(X_scaled)
    t = trustworthiness(X_scaled, emb_nn, n_neighbors=nn)
    trust_scores.append(t)

ax2.plot(nn_values, trust_scores, 'o-', color='steelblue', linewidth=2)
ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='0.90 threshold')
ax2.set_xlabel('n_neighbors')
ax2.set_ylabel('Trustworthiness')
ax2.set_title('n_neighbors Sensitivity\n(Trustworthiness)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── Plot 3: min_dist sensitivity (visual grid) ────────────────────────────
min_dist_vals = [0.0, 0.1, 0.3, 0.5]
colors = plt.cm.tab10(y / 9.0)
for i, md in enumerate(min_dist_vals):
    ax = fig.add_subplot(gs[1, i // 2 + (i % 2)])  # 2x2 grid in bottom row
    emb_md = umap.UMAP(n_neighbors=15, min_dist=md, random_state=42).fit_transform(X_scaled)
    ax.scatter(emb_md[:, 0], emb_md[:, 1], c=y, cmap='tab10', s=3, alpha=0.7)
    ax.set_title(f'min_dist={md}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('UMAP Diagnostic Plots — Digits Dataset', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/tmp/umap_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Trustworthiness: {trust:.3f}")

# ── Compute and print full metric table ───────────────────────────────────
clf = KNeighborsClassifier(n_neighbors=15)
acc_orig = cross_val_score(clf, X_scaled, y, cv=5).mean()
acc_emb  = cross_val_score(clf, embedding, y, cv=5).mean()
print(f"\nMetric Summary:")
print(f"  Trustworthiness (k=15):   {trust:.3f}")
print(f"  kNN accuracy (original):  {acc_orig:.3f}")
print(f"  kNN accuracy (embedding): {acc_emb:.3f}")
print(f"  Structural retention:     {acc_emb/acc_orig:.1%}")
```

> 🔧 **In practice:** The four-panel min_dist comparison plot is the single most useful
> diagnostic you can make before choosing hyperparameters for a presentation. Show it to your
> stakeholders and let them help pick the visualization that matches the story you want to tell.
> This is honest — you're showing the parameter sensitivity, not hiding it.

---

## §9 — Innovative Industry Applications

### 9.1 Single-Cell RNA Sequencing (scRNA-seq) — The Defining Use Case

**Domain:** Computational biology / genomics
**Scale:** 500,000 to 30 million cells, each with ~20,000 gene measurements

UMAP replaced t-SNE as the standard for single-cell visualization following Becht et al. (2019,
*Nature Biotechnology*). The Human Cell Atlas — mapping every cell type in the human body —
uses UMAP across >30 million cells. The key advantages: speed at atlas scale, support for
`n_components > 2` (enabling trajectory inference by Monocle and scVelo), and `transform()`
allowing new samples to be projected into a reference atlas without recomputation.

```python
import scanpy as sc   # pip install scanpy

# Standard scanpy/UMAP pipeline for scRNA-seq
adata = sc.read_h5ad('pbmc3k.h5ad')   # load AnnData object

# 1. Preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# 2. PCA → kNN graph → UMAP
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata, min_dist=0.3)

# 3. Clustering on high-dim UMAP (not 2D)
sc.tl.leiden(adata)

# 4. Visualize
sc.pl.umap(adata, color=['leiden', 'cell_type', 'n_genes'])
```

> 🏭 **Industry application:** The densMAP extension (Narayan et al. 2021) directly addressed
> a reproducibility crisis in single-cell biology: researchers were incorrectly inferring
> transcriptional diversity from cluster "spread" in standard UMAP plots. densMAP is now
> considered mandatory by many journals when making variability claims.

### 9.2 NLP Embedding Visualization and Topic Modeling (BERTopic)

**Domain:** Natural language processing / knowledge management
**Scale:** Thousands to millions of documents, 768–4096 dimensional embeddings

BERTopic, one of the most widely-used topic modeling libraries, uses UMAP as a core architectural
component. The pipeline: sentence-transformers encodes documents to 768D vectors → UMAP(5, cosine)
reduces to 5D → HDBSCAN clusters in 5D → clusters become topics. Companies use this for customer
feedback analysis, news categorization, and scientific literature mining.

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap

# Custom UMAP for topic modeling (5D intermediate, cosine metric)
umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=5,      # NOT 2 — HDBSCAN clusters better in 5D
    min_dist=0.0,        # tight packing for HDBSCAN
    metric='cosine',     # appropriate for normalized sentence embeddings
    random_state=42
)

# BERTopic with custom UMAP
topic_model = BERTopic(umap_model=umap_model, language='english')
topics, probabilities = topic_model.fit_transform(documents)

# Visualize topics in 2D (separately from the 5D clustering UMAP)
topic_model.visualize_topics()
```

> 🏭 **Industry application:** A major European news agency uses BERTopic with UMAP to
> automatically organize their news archive of 5 million articles into evolving topic hierarchies,
> enabling journalists to find related historical articles in seconds.

### 9.3 Drug Discovery — Chemical Space Navigation

**Domain:** Pharmaceutical R&D / cheminformatics
**Scale:** Millions of drug-like molecules, 2048-bit Morgan fingerprints

Chemical space is inherently non-Euclidean. Tanimoto similarity (the standard for molecular
similarity, equivalent to Jaccard on binary fingerprints) can only be used natively in UMAP
via the `metric='jaccard'` parameter. AstraZeneca, Novartis, and academic drug discovery groups
use UMAP to map millions of candidate molecules to 2D, identify structural clusters, guide
synthesis priorities, and spot underexplored regions.

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import umap
import pandas as pd

def mol_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    return np.array(fp)

# Generate fingerprints
fingerprints = np.array([mol_to_fp(s) for s in smiles_list])

# UMAP with Tanimoto-compatible metric
reducer = umap.UMAP(
    metric='jaccard',    # Jaccard on binary = Tanimoto coefficient
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
chemical_embedding = reducer.fit_transform(fingerprints)

# Color by biological activity
plt.scatter(chemical_embedding[:,0], chemical_embedding[:,1],
            c=activity_values, cmap='RdYlGn', s=2, alpha=0.6)
plt.colorbar(label='pIC50 (log activity)')
plt.title('Chemical Space Map — UMAP with Tanimoto Metric')
```

### 9.4 Anomaly Detection and Fraud Pattern Discovery

**Domain:** Financial services / cybersecurity
**Scale:** Millions of transactions, 50–500 dimensional feature vectors

Semi-supervised UMAP with a handful of known fraud labels (y=1) can be used to pull fraud
patterns to the foreground of the embedding — creating a "fraud-aware" coordinate system.
HDBSCAN on the embedding then finds emerging fraud clusters that weren't seen in training.

```python
import umap
import numpy as np
import hdbscan

# Semi-supervised setup: 1=known fraud, 0=known legitimate, -1=unknown
masked_labels = np.full(len(X_transactions), -1)
masked_labels[known_fraud_idx] = 1
masked_labels[known_legit_idx] = 0

# Fraud-aware UMAP embedding
reducer = umap.UMAP(
    n_neighbors=20,
    target_weight=0.7,           # strong supervision signal from known frauds
    target_metric='categorical',
    min_dist=0.0,                # tight packing for HDBSCAN
    random_state=42
)
embedding = reducer.fit_transform(X_transactions, y=masked_labels)

# Cluster emerging fraud patterns
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
cluster_labels = clusterer.fit_predict(embedding)

# Flag suspicious clusters for analyst review
fraud_clusters = np.where(
    np.bincount(cluster_labels[masked_labels==1] + 1, minlength=max(cluster_labels)+2)[1:] > 3
)[0]
print(f"High-fraud clusters: {fraud_clusters}")
```

> 🏭 **Industry application:** A major European fintech uses this pattern with densMAP to
> identify "dense fraud rings" (structured groups with similar behavior patterns) versus
> "sparse fraud" (isolated unusual transactions). The density signal from densMAP enables
> analysts to distinguish organized fraud rings from opportunistic one-off frauds.

### 9.5 Recommendation System Embedding Auditing

**Domain:** E-commerce / media streaming
**Scale:** Millions of users and items, 64–512 dimensional collaborative filtering embeddings

Production recommendation systems produce user and item embeddings via matrix factorization,
neural collaborative filtering, or two-tower models. UMAP provides an X-ray of whether these
embeddings are well-structured. Poor embeddings produce blob-like UMAP projections; good ones
show clear item-category clusters and meaningful user segments.

```python
import umap
import plotly.express as px  # for interactive visualization

# Item embeddings from a production recommendation model
# item_embeddings: (n_items, 128)
# item_metadata: DataFrame with 'category', 'price_bucket', 'age' columns

reducer = umap.UMAP(
    metric='cosine',     # recommendation embeddings are typically normalized
    n_neighbors=25,
    min_dist=0.1,
    random_state=42
)
item_2d = reducer.fit_transform(item_embeddings)

# Interactive exploration
item_metadata['UMAP_1'] = item_2d[:, 0]
item_metadata['UMAP_2'] = item_2d[:, 1]

fig = px.scatter(
    item_metadata,
    x='UMAP_1', y='UMAP_2',
    color='category',
    hover_data=['item_name', 'price_bucket'],
    title='Item Embedding Space — UMAP Projection'
)
fig.show()
```

**Key use cases:**
- Diagnose "embedding collapse" (all items in one cluster)
- Identify cold-start items that failed to position near similar items
- Audit recommendation diversity (are all recommended items from one cluster?)
- Explain personalization to non-technical stakeholders

### 9.6 Astronomy — Galaxy Morphology and Kinematic Classification

**Domain:** Astrophysics / observational astronomy
**Scale:** Millions of galaxy images, hundreds-dimensional CNN feature vectors

Modern sky surveys (SDSS, JWST, Euclid) produce images of millions of galaxies. A typical
pipeline: ConvNeXt or ResNet encoder extracts 512-dimensional morphological features → UMAP(2)
for visualization and cluster discovery → HDBSCAN for unsupervised morphological type assignment.
A 2024 paper (arXiv:2512.15137) demonstrated this approach outperforms previous classification
methods for early/late type galaxy discrimination.

```python
import torch
import torchvision.models as models
import umap

# Feature extraction with pretrained CNN
encoder = models.convnext_small(pretrained=True)
encoder.classifier[-1] = torch.nn.Identity()  # remove final classification layer
encoder.eval()

with torch.no_grad():
    galaxy_features = encoder(galaxy_images_tensor).numpy()  # (n_galaxies, 768)

# UMAP for galaxy morphology mapping
reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='cosine', random_state=42)
galaxy_embedding = reducer.fit_transform(galaxy_features)

# Unsupervised morphology clustering
import hdbscan
morphology_clusters = hdbscan.HDBSCAN(min_cluster_size=100).fit_predict(galaxy_embedding)
```

### 9.7 Time Series Classification via Feature Embeddings

**Domain:** Industrial IoT / healthcare / sports analytics
**Scale:** Thousands to millions of time series (variable length), 22-dimensional Catch22 features

Raw time series cannot be directly passed to UMAP (different lengths, autocorrelation structure
doesn't correspond to Euclidean distances). The powerful pattern: extract **Catch22 features**
(22 CAnonical Time-series CHaracteristics covering autocorrelation, distributional moments,
scaling, outliers) → UMAP on the 22D feature space.

```python
from pycatch22 import catch22_all   # pip install pycatch22
import umap
import numpy as np

# Extract Catch22 features for each time series
features = np.array([
    catch22_all(ts)['values']
    for ts in time_series_collection
])  # shape: (n_series, 22)

# UMAP on Catch22 feature space
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
ts_embedding = reducer.fit_transform(features)

# Color by discovered behavioral state (e.g., animal activity classification)
plt.scatter(ts_embedding[:,0], ts_embedding[:,1], c=behavior_labels,
            cmap='Set1', s=5, alpha=0.7)
plt.title('Time Series Embedding via Catch22 + UMAP')
```

> 🏭 **Industry application:** In precision livestock farming, this exact pattern — Catch22 +
> UMAP + HDBSCAN — is used to identify behavioral states (rumination, grazing, resting,
> abnormal) from accelerometer data on dairy cattle. The approach from arXiv:2404.18159
> achieves this without any manual labels, using cluster labels from the UMAP embedding as
> training targets for a lightweight on-device classifier.

---

## §10 — Complete Python Worked Example

We will work through a comprehensive UMAP analysis on three datasets: the sklearn Digits dataset
(1,797 samples × 64 features), an MNIST subset (10,000 samples × 784 features), and the Fashion
MNIST dataset (a drop-in replacement for MNIST with clothing items). This lets you see UMAP's
behavior at different scales and complexities.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import umap
from sklearn.datasets import load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── Dataset 1: Digits (fast, for development) ──────────────────────────────
print("=" * 60)
print("Dataset 1: sklearn Digits (1797 × 64)")
print("=" * 60)
digits = load_digits()
X_digits, y_digits = digits.data, digits.target
print(f"Shape: {X_digits.shape}, Classes: {np.unique(y_digits)}")
print(f"Feature range: [{X_digits.min():.0f}, {X_digits.max():.0f}]")
print(f"Class distribution: {dict(zip(*np.unique(y_digits, return_counts=True)))}\n")

# Preprocessing
scaler = StandardScaler()
X_digits_sc = scaler.fit_transform(X_digits)
```

```python
# ── Dataset 2: MNIST (10k subset) ──────────────────────────────────────────
print("=" * 60)
print("Dataset 2: MNIST 10k subset (10000 × 784)")
print("=" * 60)
# Fetch MNIST — may take a minute on first download
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_mnist_full, y_mnist_full = mnist.data.astype(np.float32), mnist.target.astype(int)

# Use 10k subset stratified by class
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=10000, random_state=42)
train_idx, _ = next(sss.split(X_mnist_full, y_mnist_full))
X_mnist = X_mnist_full[train_idx] / 255.0   # normalize to [0,1]
y_mnist = y_mnist_full[train_idx]
X_mnist_sc = StandardScaler().fit_transform(X_mnist)
print(f"Shape: {X_mnist.shape}\n")
```

```python
# ── Dataset 3: Fashion MNIST ───────────────────────────────────────────────
print("=" * 60)
print("Dataset 3: Fashion MNIST (10k subset, 784 features)")
print("=" * 60)
fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
X_fashion_full = fashion.data.astype(np.float32)
y_fashion_full = fashion.target.astype(int)
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sss2 = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
fidx, _ = next(sss2.split(X_fashion_full, y_fashion_full))
X_fashion = X_fashion_full[fidx] / 255.0
y_fashion = y_fashion_full[fidx]
X_fashion_sc = StandardScaler().fit_transform(X_fashion)
print(f"Shape: {X_fashion.shape}\n")
```

```python
# ── Part A: Fit UMAP on Digits — Full Analysis ─────────────────────────────
print("Fitting UMAP on Digits...")
reducer_digits = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    init='spectral',
    random_state=42,
    verbose=False
)
emb_digits = reducer_digits.fit_transform(X_digits_sc)

# ── Part B: Fit UMAP on MNIST (note: no PCA pre-reduction needed for 784D
#            at 10k samples, but for 100k+ you'd do PCA(50) first) ─────────
print("Fitting UMAP on MNIST (10k)...")
reducer_mnist = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
emb_mnist = reducer_mnist.fit_transform(X_mnist_sc)

# ── Part C: Fit UMAP on Fashion MNIST ─────────────────────────────────────
print("Fitting UMAP on Fashion MNIST (10k)...")
reducer_fashion = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
emb_fashion = reducer_fashion.fit_transform(X_fashion_sc)
print("All embeddings complete.\n")
```

```python
# ── Part D: Evaluation Metrics ─────────────────────────────────────────────

def evaluate_embedding(X_orig, embedding, labels, name, k=15):
    """Compute a suite of embedding quality metrics."""
    n_eval = min(3000, len(X_orig))
    idx = np.random.RandomState(42).choice(len(X_orig), n_eval, replace=False)

    # Trustworthiness
    trust = trustworthiness(X_orig[idx], embedding[idx], n_neighbors=k)

    # kNN accuracy
    clf = KNeighborsClassifier(n_neighbors=k)
    acc_orig = cross_val_score(clf, X_orig[idx], labels[idx], cv=5).mean()
    acc_emb = cross_val_score(clf, embedding[idx], labels[idx], cv=5).mean()
    retention = acc_emb / acc_orig if acc_orig > 0 else 0.0

    # Silhouette score in embedding
    sil_emb = silhouette_score(embedding[idx], labels[idx])

    print(f"\n{name}:")
    print(f"  Trustworthiness (k={k}):   {trust:.3f}")
    print(f"  kNN accuracy (original):   {acc_orig:.3f}")
    print(f"  kNN accuracy (embedding):  {acc_emb:.3f}")
    print(f"  Structural retention:      {retention:.1%}")
    print(f"  Silhouette (embedding):    {sil_emb:.3f}")
    return trust, acc_orig, acc_emb, sil_emb

metrics_digits  = evaluate_embedding(X_digits_sc, emb_digits, y_digits, "Digits")
metrics_mnist   = evaluate_embedding(X_mnist_sc, emb_mnist, y_mnist, "MNIST (10k)")
metrics_fashion = evaluate_embedding(X_fashion_sc, emb_fashion, y_fashion, "Fashion MNIST (10k)")
```

```python
# ── Part E: Comprehensive Visualization ───────────────────────────────────
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Row 1: Embeddings
datasets = [
    (emb_digits, y_digits, "Digits (1797 samples)", list(range(10))),
    (emb_mnist, y_mnist, "MNIST 10k subset", list(range(10))),
    (emb_fashion, y_fashion, "Fashion MNIST 10k", class_names),
]

for col, (emb, labels, title, class_list) in enumerate(datasets):
    ax = fig.add_subplot(gs[0, col])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10',
                    s=2, alpha=0.7, rasterized=True)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

# Row 2: n_neighbors sensitivity (on Digits for speed)
nn_vals = [5, 10, 30]
for col, nn in enumerate(nn_vals):
    emb_nn = umap.UMAP(n_neighbors=nn, random_state=42).fit_transform(X_digits_sc)
    trust_nn = trustworthiness(X_digits_sc, emb_nn, n_neighbors=nn)
    ax = fig.add_subplot(gs[1, col])
    ax.scatter(emb_nn[:,0], emb_nn[:,1], c=y_digits, cmap='tab10', s=3, alpha=0.8)
    ax.set_title(f'n_neighbors={nn}\nTrust={trust_nn:.3f}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# Row 3: Supervised vs unsupervised comparison
emb_unsup = emb_digits  # already computed
reducer_sup = umap.UMAP(n_neighbors=15, target_weight=0.8, random_state=42)
emb_sup = reducer_sup.fit_transform(X_digits_sc, y=y_digits)

ax_unsup = fig.add_subplot(gs[2, 0])
ax_unsup.scatter(emb_unsup[:,0], emb_unsup[:,1], c=y_digits, cmap='tab10', s=3, alpha=0.8)
ax_unsup.set_title('Unsupervised UMAP\n(no label information)', fontsize=10)
ax_unsup.set_xticks([]); ax_unsup.set_yticks([])

ax_sup = fig.add_subplot(gs[2, 1])
ax_sup.scatter(emb_sup[:,0], emb_sup[:,1], c=y_digits, cmap='tab10', s=3, alpha=0.8)
ax_sup.set_title('Supervised UMAP\n(target_weight=0.8)', fontsize=10)
ax_sup.set_xticks([]); ax_sup.set_yticks([])

# Metric summary table
ax_table = fig.add_subplot(gs[2, 2])
ax_table.axis('off')
table_data = [
    ['Dataset', 'Trust.', 'kNN-Ret.', 'Sil.'],
    ['Digits',   f'{metrics_digits[0]:.3f}',  f'{metrics_digits[2]/metrics_digits[1]:.1%}', f'{metrics_digits[3]:.3f}'],
    ['MNIST 10k', f'{metrics_mnist[0]:.3f}',  f'{metrics_mnist[2]/metrics_mnist[1]:.1%}',  f'{metrics_mnist[3]:.3f}'],
    ['Fashion',   f'{metrics_fashion[0]:.3f}', f'{metrics_fashion[2]/metrics_fashion[1]:.1%}', f'{metrics_fashion[3]:.3f}'],
]
tbl = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
ax_table.set_title('Metric Summary', fontsize=10, fontweight='bold')

plt.suptitle('UMAP Analysis: Digits, MNIST, Fashion MNIST', fontsize=14, fontweight='bold')
plt.savefig('/tmp/umap_full_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

```python
# ── Part F: densMAP on Digits ──────────────────────────────────────────────
print("\nFitting densMAP on Digits...")
reducer_dens = umap.UMAP(
    densmap=True,
    output_dens=True,
    dens_lambda=2.0,
    n_neighbors=15,
    random_state=42
)
result_dens = reducer_dens.fit_transform(X_digits_sc)
emb_dens = result_dens[0]
radii_orig = result_dens[1]   # log-radii in original space

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc1 = axes[0].scatter(emb_dens[:,0], emb_dens[:,1], c=y_digits,
                      cmap='tab10', s=8, alpha=0.8)
plt.colorbar(sc1, ax=axes[0], label='Digit class')
axes[0].set_title('densMAP — colored by class', fontsize=12)

sc2 = axes[1].scatter(emb_dens[:,0], emb_dens[:,1], c=radii_orig,
                      cmap='RdYlBu_r', s=8, alpha=0.8)
plt.colorbar(sc2, ax=axes[1], label='Original space log-radius\n(lower = denser region)')
axes[1].set_title('densMAP — colored by original space density', fontsize=12)
plt.suptitle('densMAP on Digits: Density-Preserving Visualization', fontsize=13)
plt.tight_layout()
plt.show()

# ── Part G: Hyperparameter Grid Search with Trustworthiness ───────────────
import itertools

print("\nRunning hyperparameter grid search (may take a few minutes)...")
nn_range = [5, 10, 15, 30, 50]
md_range = [0.0, 0.1, 0.25, 0.5]
results = []

for n_nb, md in itertools.product(nn_range, md_range):
    emb_gs = umap.UMAP(n_neighbors=n_nb, min_dist=md, random_state=42).fit_transform(X_digits_sc)
    t = trustworthiness(X_digits_sc, emb_gs, n_neighbors=n_nb)
    clf = KNeighborsClassifier(n_neighbors=n_nb)
    acc = cross_val_score(clf, emb_gs, y_digits, cv=3).mean()
    results.append({'n_neighbors': n_nb, 'min_dist': md,
                    'trustworthiness': t, 'knn_accuracy': acc})

results_df = pd.DataFrame(results)
pivot_trust = results_df.pivot(index='n_neighbors', columns='min_dist', values='trustworthiness')
pivot_acc   = results_df.pivot(index='n_neighbors', columns='min_dist', values='knn_accuracy')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(pivot_trust.round(3), annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0.85, vmax=1.0, ax=axes[0])
axes[0].set_title('Trustworthiness Heatmap\n(n_neighbors vs min_dist)', fontsize=12)

sns.heatmap(pivot_acc.round(3), annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0.8, vmax=1.0, ax=axes[1])
axes[1].set_title('kNN Accuracy (5-class CV) Heatmap\n(n_neighbors vs min_dist)', fontsize=12)
plt.tight_layout()
plt.show()

best_row = results_df.loc[results_df['knn_accuracy'].idxmax()]
print(f"\nBest configuration by kNN accuracy:")
print(f"  n_neighbors={best_row['n_neighbors']:.0f}, min_dist={best_row['min_dist']:.2f}")
print(f"  Trustworthiness: {best_row['trustworthiness']:.3f}")
print(f"  kNN accuracy:    {best_row['knn_accuracy']:.3f}")
```

**Interpretation of results:** On Digits, UMAP with default parameters (n_neighbors=15,
min_dist=0.1) typically achieves trustworthiness > 0.92 and kNN accuracy retention > 95%.
The 10 digit classes are well-separated with some overlap between visually similar digits
(1/7 and 4/9). On Fashion MNIST, the separation is harder because some clothing items (shirt/coat/pullover)
are genuinely ambiguous even to humans — UMAP faithfully reflects this by showing more
overlap between these classes. On MNIST, the structure is typically very clean with a
trustworthiness > 0.94 on the 10k subset.

The supervised UMAP comparison shows a dramatic visual difference: classes that overlapped
slightly in the unsupervised embedding are pulled into tight, non-overlapping clusters.
This is useful for classification but removes the within-class structure that the unsupervised
embedding revealed.

---

## §11 — When to Use This Algorithm

### Decision Flowchart

```
Your dimensionality reduction problem
         │
         ├── Need a linear projection? → Use PCA
         │   (interpretable, fast, invertible, stable)
         │
         ├── Have class labels and want discriminative features? → Use LDA
         │   (guaranteed optimal class separation in fewer dimensions)
         │
         ├── n > 100,000 samples? ─────────────────────────────────────┐
         │                                                              │
         ├── Need manifold/topology visualization? → **UMAP** ◄────────┘
         │   (n_components=2 or 3)                      (also for large n)
         │
         ├── Need manifold features for downstream ML? → **UMAP**
         │   (n_components=10-50, faster and more scalable than t-SNE)
         │
         ├── Need to embed new points at inference time? → **UMAP** or ParametricUMAP
         │   (t-SNE cannot do this at all)
         │
         ├── n < 5,000 and local structure is paramount? → Consider t-SNE
         │   (slightly better local fidelity at small scales)
         │
         ├── Need global structure AND local structure? → **UMAP** with init='spectral'
         │
         └── Need density-aware visualization? → **UMAP** with densmap=True
```

### Comparison Table with Alternatives

| Property | UMAP | t-SNE | PCA | Isomap | LLE |
|---|---|---|---|---|---|
| Nonlinear | Yes | Yes | No | Yes | Yes |
| Global structure | Good | Poor (random init) | Excellent | Good | Poor |
| Local structure | Excellent | Excellent | Moderate | Good | Excellent |
| Speed (100k pts) | Minutes | Hours | Seconds | Hours | Hours |
| New point transform | Yes (approx.) | No | Yes (exact) | No | No |
| GPU acceleration | Yes (cuML) | Limited | No | No | No |
| n_components > 3 | Works well | Degrades | Unlimited | Works | Degrades |
| Supervised mode | Yes | No | No | No | No |
| Density preservation | With densMAP | No | No | No | No |
| Interpretable axes | No | No | Yes | No | No |
| Custom metrics | Yes | Limited | No | Yes | No |

### Red Flags — When NOT to Use UMAP

1. **When you need interpretable axes.** UMAP axes have no meaning — you cannot say "UMAP dimension 1 represents X." If interpretability is required, use PCA (explained variance) or LDA (class-discriminating directions).

2. **When you need provably invertible reconstruction.** UMAP's `inverse_transform()` is approximate (only available in ParametricUMAP with decoder). For reconstruction tasks, use PCA or autoencoders.

3. **When your data has no manifold structure.** If your data is pure noise in high dimensions, UMAP will produce a meaningless embedding. Check whether your data has genuine low-dimensional structure first (PCA scree plot, intrinsic dimensionality estimation).

4. **When you need to interpret inter-cluster distances quantitatively.** UMAP cluster distances are not meaningful distances in the original space. If quantitative similarity claims are needed, work in the original space with proper distance metrics.

5. **When reproducibility across institutions is critical.** UMAP results depend on random_state and exact software version. Spectral initialization helps, but if bitwise reproducibility is required (e.g., regulatory submissions), PCA is more appropriate.

6. **When n is very small (n < 100).** With very few points, the kNN graph is very sparse and UMAP's manifold approximation is unreliable. PCA or MDS are better choices for small datasets.

---

## §12 — Top Papers to Study

**Ranked from "Start here" to "Deep mastery":**

---

**[1] START HERE — The Original Paper**

McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and
Projection for Dimension Reduction." *arXiv:1802.03426* (revised 2020).
URL: https://arxiv.org/abs/1802.03426
DOI: https://doi.org/10.48550/arXiv.1802.03426

*Why read it:* This is one of the most mathematically rigorous applied ML papers of the 2010s.
Section 2 on the theoretical foundations (fuzzy simplicial sets, Riemannian geometry) is dense
but essential — it explains every design choice. Read Sections 1, 3, and 4 first for the
algorithm and results; come back to Section 2 for the theory. Rated: 5/5 for both
mathematical depth and practical impact.

---

**[2] START HERE — The Accessible Deep-Dive**

Coenen, A. & Pearce, A. (2019). "Understanding UMAP." *Google PAIR Research.*
URL: https://pair-code.github.io/understanding-umap/

*Why read it:* The best practitioner's guide to understanding what `n_neighbors` and `min_dist`
actually do. Uses interactive visualizations to show parameter sensitivity on canonical toy
datasets. Read this immediately after the original paper to build intuition. Highly recommended
for anyone who will be tuning UMAP hyperparameters. Free, interactive, beautifully designed.

---

**[3] READ NEXT — densMAP**

Narayan, A., Berger, B., & Cho, H. (2021). "Assessing single-cell transcriptomic variability
through density-preserving data visualization." *Nature Biotechnology* 39, 765–774.
DOI: https://doi.org/10.1038/s41587-020-00801-7

*Why read it:* Identifies and fixes the most important practical limitation of standard UMAP:
the loss of local density information. Essential reading for anyone using UMAP in biology,
but the insight is universally applicable. The densMAP regularization idea (penalizing
divergence of log-radii) is elegant and the paper's experiments are compelling. Also
demonstrates how to construct a principled extension of an existing DR algorithm.

---

**[4] READ NEXT — The Initialization Paper**

Kobak, D. & Linderman, G.C. (2021). "Initialization is critical for preserving global data
structure in both t-SNE and UMAP." *Nature Biotechnology* 39, 156–157.
DOI: https://doi.org/10.1038/s41587-020-00809-z

*Why read it:* This short letter disproves a common misconception: UMAP's better global
structure preservation is largely due to spectral initialization, not the UMAP objective
itself. A two-page must-read that will permanently change how you think about the UMAP vs
t-SNE comparison. If you've ever cited "UMAP has better global structure than t-SNE," read
this first.

---

**[5] READ NEXT — The Biological Application Paper**

Becht, E., McInnes, L., Healy, J., et al. (2019). "Dimensionality reduction for visualizing
single-cell data using UMAP." *Nature Biotechnology* 37, 38–44.
DOI: https://doi.org/10.1038/nbt.4314

*Why read it:* The paper that established UMAP as the standard in single-cell biology.
Provides the most comprehensive empirical comparison of UMAP vs. t-SNE, PCA, and other
methods on real biological data. If you work in genomics, this is required reading.
If you don't, it still provides the clearest demonstration of UMAP's practical advantages
on realistic, complex datasets.

---

**[6] DEEP MASTERY — Theoretical Unification with t-SNE**

Damrich, S., Böhm, J.N., Hamprecht, F.A., & Kobak, D. (2022/2023). "From t-SNE to UMAP
with Contrastive Learning." *arXiv:2206.01816.* ICLR 2023.
URL: https://arxiv.org/abs/2206.01816

*Why read it:* The deepest theoretical paper on UMAP. Reveals that both t-SNE and UMAP are
contrastive learning methods, unified under a single framework. Explains *why* UMAP produces
tighter clusters than t-SNE (it's a normalization artifact of negative sampling, not a
superior objective). Proposes a parameterization that smoothly interpolates between the two.
If you want to understand UMAP at the level of its creators, read this. Requires familiarity
with noise-contrastive estimation and information theory.

---

**[7] DEEP MASTERY — Parametric UMAP**

Sainburg, T., McInnes, L., & Gentner, T.Q. (2021). "Parametric UMAP Embeddings for
Representation and Semisupervised Learning." *Neural Computation* 33(11), 2881–2907.
DOI: https://doi.org/10.1162/neco_a_01410

*Why read it:* The formal paper introducing ParametricUMAP — how to replace the table-lookup
embedding with a neural network trained on the UMAP objective. Demonstrates applications in
semi-supervised learning, transfer learning, and inverse transforms. Essential if you need
production UMAP inference or want to use UMAP as a training objective for a neural architecture.

---

**[8] PERSPECTIVE — Critical View**

"Stop Misusing t-SNE and UMAP for Visual Analytics." *arXiv:2506.08725* (2025).
URL: https://arxiv.org/abs/2506.08725

*Why read it:* A recent survey of misuse patterns in published papers that use UMAP and t-SNE.
Documents the most common mistakes: over-interpreting cluster distances, treating cluster size
as population size, circular reasoning with clustering, and density misinterpretation. Reading
this will make you a more rigorous practitioner and a better reviewer of papers using UMAP.

---

## §13 — Resources & Further Reading

### Books

- **"Introduction to Machine Learning" by Alpaydin (4th ed.)** — Chapter on manifold learning
  provides the conceptual grounding; doesn't cover UMAP specifically but builds the prerequisites
  for understanding the topology-based approach.
- **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Géron (3rd ed., 2022)** —
  Chapter 8 covers dimensionality reduction including t-SNE and mentions UMAP; good for the
  sklearn ecosystem context.
- **"Mathematics for Machine Learning" by Deisenroth, Faisal, Ong** — Chapters on linear algebra
  and probability theory are prerequisite math for reading the UMAP paper. Free at
  https://mml-book.github.io/.

### Blog Posts and Tutorials

- **umap-learn documentation** (primary reference):
  https://umap-learn.readthedocs.io/en/latest/
  The official docs include tutorials for all variants: basic, supervised, parametric, densMAP,
  AlignedUMAP. This is the most accurate and up-to-date source.

- **"How UMAP Works" — Andy Coenen, Google PAIR** (the best explainer):
  https://pair-code.github.io/understanding-umap/
  Interactive, visual, and correct. Should be read by every UMAP user.

- **"Leland McInnes' UMAP introduction talk" (SciPy 2018)**:
  https://www.youtube.com/watch?v=nq6iPZVUxZU
  The creator explains the algorithm. Excellent for building conceptual understanding before
  diving into the paper.

- **"UMAP: A new dimension reduction technique" — Towards Data Science**:
  Multiple high-quality posts; search specifically for ones by McInnes or citing the
  original paper for accuracy.

- **densvis documentation** (densMAP reference implementation):
  https://github.com/hhcho/densvis
  Reference code for den-SNE and densMAP with examples.

### Package Documentation and APIs

- **umap-learn PyPI + documentation:**
  Install: `pip install umap-learn`
  Docs: https://umap-learn.readthedocs.io/

- **cuML UMAP (RAPIDS) documentation:**
  https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.manifold.umap/
  Install: `pip install cuml-cu12` (or conda for RAPIDS suite)

- **sklearn trustworthiness metric:**
  https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html

- **pynndescent (UMAP's ANN backend):**
  https://pynndescent.readthedocs.io/
  Understanding this helps debug slow UMAP fits.

- **BERTopic (UMAP + HDBSCAN for NLP):**
  https://maartengr.github.io/BERTopic/
  The most production-ready NLP application of UMAP.

- **scanpy (scRNA-seq + UMAP):**
  https://scanpy.readthedocs.io/
  The standard single-cell analysis toolkit.

### Online Courses

- **Fast.ai "Practical Deep Learning for Coders"** — While not about UMAP specifically, the
  embedding visualization sections use UMAP extensively and provide practical context.
- **Coursera "Advanced Machine Learning Specialization"** — covers manifold learning at a
  moderate depth; watch the t-SNE lectures as conceptual background for UMAP.
- **YouTube: "UMAP Uniform Manifold Approximation and Projection" by Leland McInnes** at
  the Alan Turing Institute — The creator's comprehensive lecture, ~2 hours, is the most
  complete introduction combining theory and practice.
