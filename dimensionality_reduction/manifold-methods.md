# Manifold Methods: Isomap, LLE, Spectral Embedding, MDS, and PHATE

> **Why this family of algorithms matters:** In December 2000, two papers appeared side-by-side in
> *Science* — one of the most competitive journals on Earth — that changed how we think about
> high-dimensional data. Tenenbaum, de Silva, and Langford showed that 2,000 face images under
> varying lighting and viewpoint, each a 4,096-dimensional pixel vector, actually lived on a
> 3-dimensional manifold corresponding to left-right pose, up-down pose, and illumination direction.
> Roweis and Saul showed the same with handwritten digits. The insight: data you thought was
> "high-dimensional" is often secretly low-dimensional, but the low-dimensional structure is curved
> — a manifold embedded in high-dimensional space. PCA, which only finds linear subspaces, is blind
> to this. Manifold methods are the tools built specifically to find and unfurl that curved structure.
> Every robotics lab using configuration-space analysis, every single-cell biology lab running RNA
> sequencing, every neuroscience group analyzing population neural activity — they all reach for
> these tools when the geometry is nonlinear.

---

## §0 — Python Ecosystem & Package Guide

The manifold learning landscape in Python is dominated by `sklearn.manifold` for the classical
algorithms (Isomap, LLE and its variants, Spectral Embedding, MDS), plus the specialized `phate`
package for trajectory-aware biological embeddings, and `pydiffmap` for raw diffusion maps.
Before writing a single line of modeling code, you need to understand what you're choosing between.

### Complete Package Table

| Package | Import | Version (mid-2026) | Covers | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS` | ~1.9.x | Isomap, LLE+MLLE+HLLE+LTSA, Spectral Embedding, MDS (metric + non-metric), Classical MDS (1.9+) | BSD-3 |
| `phate` | `import phate` | 1.0.11 | PHATE (trajectory-preserving diffusion embedding for biology) | GPL-2 |
| `pydiffmap` | `from pydiffmap import diffusion_map as dm` | 0.2.0.1 | Full Coifman-Lafon diffusion maps with alpha normalization | MIT |
| `scanpy` | `import scanpy as sc` | ~1.10.x | Diffusion maps + pseudotime for single-cell workflows (AnnData-native) | BSD-3 |
| `megaman` | `import megaman` | 0.1.0 | Scalable manifold learning for very large datasets via geometric harmonics | BSD-3 |
| `umap-learn` | `import umap` | ~0.5.x | UMAP (a closely related but distinct method — covered in its own chapter) | BSD-3 |

> ⚠️ **Pitfall:** The `pydiffmap` package (v0.2.0.1) has had low release activity as of mid-2026.
> It remains the canonical reference implementation of Coifman-Lafon diffusion maps with the
> `alpha` normalization parameter, but treat it as a research/reference tool rather than production
> infrastructure. For production biology workflows, use `phate` or `scanpy`.

### Top Picks — Recommendation Table

| Package | Best for | Modern vs Legacy | Notes |
|---|---|---|---|
| **`sklearn.manifold`** | All classical methods; prototyping; benchmarking | Modern (actively maintained) | **Start here for 95% of use cases.** Consistent API, Pipeline-compatible. |
| **`phate`** | Single-cell genomics, trajectory analysis, biological data with branches | Modern (KrishnaswamyLab, Yale) | Purpose-built for developmental/differentiation data. Handles 500k+ cells with landmarks. |
| **`pydiffmap`** | Diffusion maps with full alpha parameter control | Legacy (low activity) | Use when you specifically need the Coifman-Lafon alpha normalization. |
| **`scanpy`** | Single-cell full workflow (QC → preprocessing → DR → clustering → trajectory) | Modern | Use when working within the scverse ecosystem; `sc.tl.diffmap()` for diffusion maps. |
| **`megaman`** | Very large datasets (>50k points) with manifold methods | Research-grade | Approximate methods with geometric harmonics; not well-maintained but unique. |

### Same Dataset, Five Methods Side-by-Side

Here is how the same call looks across the top packages, so you can see the API surface at a glance:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
    MDS,
    trustworthiness,
)

# ─── Shared setup ─────────────────────────────────────────────────────────────
n_samples = 1500
X_swiss, t_swiss = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
X_scaled = StandardScaler().fit_transform(X_swiss)

# ─── sklearn: Isomap ──────────────────────────────────────────────────────────
iso = Isomap(n_neighbors=10, n_components=2, eigen_solver='arpack')
Z_iso = iso.fit_transform(X_scaled)
print(f"Isomap recon error:    {iso.reconstruction_error():.4f}")
print(f"Isomap trustworthiness: {trustworthiness(X_scaled, Z_iso, n_neighbors=10):.4f}")

# ─── sklearn: LLE (MLLE variant — recommended default) ────────────────────────
lle = LocallyLinearEmbedding(
    n_neighbors=12, n_components=2, method='modified', random_state=42
)
Z_lle = lle.fit_transform(X_scaled)
print(f"MLLE recon error_:     {lle.reconstruction_error_:.4e}")
print(f"MLLE trustworthiness:  {trustworthiness(X_scaled, Z_lle, n_neighbors=10):.4f}")

# ─── sklearn: Spectral Embedding (Laplacian Eigenmaps) ────────────────────────
se = SpectralEmbedding(
    n_components=2, n_neighbors=10, affinity='nearest_neighbors', random_state=42
)
Z_se = se.fit_transform(X_scaled)
print(f"SpectralEmb trustworthiness: {trustworthiness(X_scaled, Z_se, n_neighbors=10):.4f}")

# ─── sklearn: MDS (non-metric recommended for exploratory work) ───────────────
# sklearn 1.5.x API: metric=False, dissimilarity='euclidean'
# sklearn 1.8+  API: metric_mds=False, metric='euclidean'  (see §4 for version note)
mds = MDS(
    n_components=2, metric=False, n_init=4,
    dissimilarity='euclidean', normalized_stress=True, random_state=42
)
Z_mds = mds.fit_transform(X_scaled)
print(f"MDS stress-1:          {mds.stress_:.4f}")
print(f"MDS trustworthiness:   {trustworthiness(X_scaled, Z_mds, n_neighbors=10):.4f}")

# ─── PHATE: trajectory-preserving embedding ───────────────────────────────────
# pip install phate
import phate  # noqa: E402 — install required
phate_op = phate.PHATE(n_components=2, knn=5, t='auto', random_state=42, verbose=0)
Z_phate = phate_op.fit_transform(X_swiss)   # PHATE prefers unscaled counts for biology;
                                             # for synthetic data, StandardScaler first is fine
print(f"PHATE trustworthiness: {trustworthiness(X_scaled, Z_phate, n_neighbors=10):.4f}")

# ─── Quick 2×3 comparison plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
embeddings = [Z_iso, Z_lle, Z_se, Z_mds, Z_phate]
titles      = ['Isomap', 'MLLE', 'Spectral\nEmbedding', 'MDS\n(non-metric)', 'PHATE']
for ax, Z, title in zip(axes, embeddings, titles):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=t_swiss, cmap='plasma', s=8, alpha=0.7)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(sc, ax=axes[-1], label='Swiss Roll position (t)')
plt.suptitle('Swiss Roll — Five Manifold Methods Compared', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('swiss_roll_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

> 🔧 **In practice:** Run all five on your dataset and compare. If they broadly agree on the
> structure, that structure is real and robust. If they disagree, you've learned something important
> about your data's geometry — or about which method's assumptions your data violates.

### Key Version Notes (Critical for Production Code)

The MDS class in sklearn has undergone significant API changes. If you write code today and run it
on a different sklearn version in the future, you will hit errors. Here is what changed:

```python
import sklearn
from packaging.version import Version

sklearn_ver = Version(sklearn.__version__)

# ─── MDS: version-safe instantiation ──────────────────────────────────────────
if sklearn_ver >= Version("1.8"):
    # v1.8+ renamed parameters:
    #   metric=True/False  →  metric_mds=True/False
    #   dissimilarity='euclidean'  →  metric='euclidean'
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, metric_mds=False, metric='euclidean',
              n_init=4, random_state=42)
else:
    # v1.5.x and earlier (course baseline):
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, metric=False, dissimilarity='euclidean',
              n_init=4, random_state=42)

# ─── sklearn 1.9+ added ClassicalMDS as a separate class (analytical solution) ─
if sklearn_ver >= Version("1.9"):
    from sklearn.manifold import MDS  # ClassicalMDS also available  # verify in current docs
```

The full chapter uses sklearn 1.5.x syntax as the baseline. All version differences are called out
explicitly. If you are on sklearn 1.8+, substitute the new parameter names wherever you see the
1.5.x names flagged with `# sklearn 1.5.x API`.

### GPU-Accelerated and Streaming Options

None of the sklearn manifold methods are GPU-accelerated — they are CPU-only algorithms. For
GPU-native manifold learning:

- **cuML** (RAPIDS): Implements UMAP and t-SNE on GPU; does not yet include Isomap or LLE.
- **PHATE (large scale)**: Uses landmark approximation (`n_landmark` parameter) with Nyström
  extension — effectively a O(N · n_landmark) algorithm that scales to 500k+ cells.
- **Streaming / online**: None of the classical manifold methods support true online learning.
  Isomap supports `transform()` for out-of-sample extension of new points but requires re-running
  the full geodesic computation on the original data. For truly streaming data at scale, UMAP
  (covered in the UMAP chapter) is the practical choice.

> 🏆 **Best practice:** For exploratory work on N < 5,000, use sklearn manifold methods directly.
> For N = 5,000–50,000, use PHATE with `n_landmark=2000`. For N > 50,000, consider UMAP (a
> different chapter) or PHATE's SGD solver. The O(N²) memory wall of classical methods is real.

---

## §1 — The Origin: The Papers That Started It All

### The December 2000 *Science* Double Publication

Picture the editorial situation: two research groups, working independently, submitted papers to
*Science* — one of the most competitive journals on Earth — that were so complementary the editors
published them back-to-back in the same issue, on pages 2319 and 2323 respectively. That doesn't
happen often. It happened because both papers were answering the same profound question that PCA
and linear methods had left unanswered for decades.

> 📜 **Origin/Citation:** Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). **A Global
> Geometric Framework for Nonlinear Dimensionality Reduction.** *Science*, 290(5500), 2319–2323.
> https://doi.org/10.1126/science.290.5500.2319

> 📜 **Origin/Citation:** Roweis, S. T., & Saul, L. K. (2000). **Nonlinear Dimensionality
> Reduction by Locally Linear Embedding.** *Science*, 290(5500), 2323–2326.
> https://doi.org/10.1126/science.290.5500.2323

### What Came Before — And Why It Wasn't Enough

By 2000, PCA had been the dominant dimensionality reduction tool for nearly a century (Pearson
1901, Hotelling 1933). It finds the directions of maximum variance — the linear subspace that
best preserves global spread. Classical MDS (Torgerson 1952) extended this to work on arbitrary
distance matrices. Both are beautiful, but they share a fundamental limitation: they only find
*linear* structure.

If your data lies on a curved surface in high-dimensional space — a Swiss Roll, a sphere, a
torus, or the manifold of face images under varying lighting — PCA will "flatten" it incorrectly.
It will project the two ends of the Swiss Roll onto nearby locations in 2D, even though those
points are far apart when measured along the surface of the roll. The Euclidean shortcut through
the air is not available to a data point that has to travel along the manifold.

The key insight that both papers shared: **the manifold assumption.** Real high-dimensional data
in many domains (computer vision, robotics, biology, speech) doesn't fill its ambient space. It
concentrates on a low-dimensional manifold — a curved surface embedded in high dimensions. And
the right notion of "distance" between points on that manifold is not Euclidean distance through
the ambient space, but *geodesic distance* — distance measured along the surface.

### Isomap — The Core Insight

Tenenbaum, de Silva, and Langford's insight was elegant: **replace Euclidean distances with
geodesic distances, then apply classical MDS.**

The geodesic distance between two points on a manifold is the length of the shortest path that
stays on the manifold. For a sphere, it's the great-circle distance. For a Swiss Roll, it's the
length of the curve you'd trace if you unrolled the paper.

You cannot compute true geodesic distances directly from a finite point cloud. But you can
*approximate* them: build a graph connecting each point to its $k$ nearest neighbors (using
Euclidean distance for nearby points where the manifold is approximately flat), then use
shortest-path algorithms to estimate the geodesic distance between all pairs.

**Isomap's three-step algorithm:**

1. **Neighborhood graph:** For each $\mathbf{x}_i$, find its $k$ nearest neighbors by Euclidean
   distance. Connect them with edges weighted by Euclidean distance $\|\mathbf{x}_i - \mathbf{x}_j\|$.

2. **All-pairs geodesic distances:** Run Dijkstra's algorithm (sparse graph) or Floyd-Warshall
   (dense graph) to compute the shortest-path distance matrix $\mathbf{D}_G \in \mathbb{R}^{n \times n}$.

3. **Classical MDS embedding:** Apply classical MDS to $\mathbf{D}_G$: double-center to form
   the Gram matrix $\mathbf{B} = -\frac{1}{2}\mathbf{H}\mathbf{D}_G^{(2)}\mathbf{H}$ where
   $\mathbf{H} = \mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$, then take the top $d$
   eigenvectors scaled by their square-root eigenvalues as coordinates.

The theoretical guarantee is powerful: if $\mathcal{M}$ is isometrically embeddable in Euclidean
space, as $n \to \infty$ and $k$ grows appropriately, Isomap recovers the true Euclidean geometry.

**The original demonstrations** remain striking: 2,000 face images (64×64 pixels = 4,096 dims)
under varying lighting and viewpoint → Isomap recovered a clean 3D embedding where the three axes
corresponded to left-right head rotation, up-down tilt, and illumination direction. PCA gave a
muddled projection. A robot arm's 2D configuration manifold was recovered from raw image pixels.

### LLE — A Different Route to the Same Destination

Roweis and Saul took a completely different approach. Instead of computing global geodesic
distances, they focused on *local linear structure* and asked: can I find a low-dimensional
embedding that preserves local geometry everywhere simultaneously?

Their insight: each data point $\mathbf{x}_i$ can be reconstructed as a weighted linear
combination of its $k$ nearest neighbors. Those reconstruction weights capture the local geometry.
The key claim: **if the low-dimensional coordinates $\mathbf{z}_i$ use the same weights to
reconstruct each point from its neighbors, the global embedding is geometrically consistent.**

Barycentric coordinates (the reconstruction weights) are invariant to rotation, scaling, and
translation. So the weights encode pure "shape" — the local geometry stripped of position.

**LLE's three-step algorithm:**

1. **$k$-nearest neighbors:** For each $\mathbf{x}_i$, find its $k$ nearest neighbors.

2. **Weight computation:** Solve for the weights $W_{ij}$ minimizing:
   $$\min_{\mathbf{W}} \sum_i \left\|\mathbf{x}_i - \sum_j W_{ij} \mathbf{x}_j\right\|^2 \quad \text{s.t.} \sum_j W_{ij} = 1, \; W_{ij} = 0 \text{ if } j \notin \mathcal{N}(i)$$

3. **Embedding:** Find coordinates $\mathbf{Z}$ minimizing:
   $$\min_{\mathbf{Z}} \sum_i \left\|\mathbf{z}_i - \sum_j W_{ij} \mathbf{z}_j\right\|^2 \quad \text{s.t.} \;\mathbf{Z}^\top\mathbf{Z} = \mathbf{I}$$
   This is the bottom $d+1$ eigenvectors (excluding the trivial zero eigenvector) of the sparse
   matrix $\mathbf{M} = (\mathbf{I} - \mathbf{W})^\top(\mathbf{I} - \mathbf{W})$.

> 💡 **Intuition:** Isomap is like a hiker who measures all trail lengths between every pair of
> campsites, then uses those trail distances to draw a map. LLE is like an orienteer who notes
> "I can reach campsite B from A as 30% of trail 1 plus 70% of trail 2" at each junction, then
> finds a map where those local directions are self-consistent everywhere. One is global-first;
> the other is local-first.

### Historical Context: MDS Comes First, Diffusion Maps Follow

**Classical MDS** (Torgerson 1952): given pairwise dissimilarities $\mathbf{D}$, find Euclidean
coordinates reproducing them. Used in psychometrics for spatial representations of similarity.

**Non-metric MDS** (Kruskal 1964): only reproduce the *ordering* of distances, not exact values.
This robustness makes it the default for exploratory social science and perception research today.

**Laplacian Eigenmaps / Spectral Embedding** (Belkin & Niyogi 2003): build a neighborhood graph,
compute the graph Laplacian $\mathbf{L} = \mathbf{D} - \mathbf{A}$, and use its bottom eigenvectors
as coordinates. Links manifold learning to spectral graph theory and the physics of heat diffusion.

**Diffusion Maps** (Coifman & Lafon 2006): define a diffusion distance between points based on
random-walk probability over $t$ steps. More robust than geodesic distance to noise and shortcuts.

**PHATE** (Moon et al. 2019, *Nature Biotechnology*): extends diffusion maps with an information-
geometric potential distance and MDS, specifically designed to reveal branching developmental
trajectories in single-cell RNA sequencing. Applied to 54 million cells from 168 COVID-19
patients in its 2022 follow-up.

> 📜 **Origin/Citation — Laplacian Eigenmaps:** Belkin, M., & Niyogi, P. (2003). Laplacian
> Eigenmaps for Dimensionality Reduction and Data Representation. *Neural Computation*, 15(6),
> 1373–1396. https://doi.org/10.1162/089976603321780317

> 📜 **Origin/Citation — Diffusion Maps:** Coifman, R. R., & Lafon, S. (2006). Diffusion Maps.
> *Applied and Computational Harmonic Analysis*, 21(1), 5–30.
> https://doi.org/10.1016/j.acha.2006.04.006

> 📜 **Origin/Citation — PHATE:** Moon, K. R., et al. (2019). Visualizing structure and
> transitions in high-dimensional biological data. *Nature Biotechnology*, 37, 1482–1492.
> https://doi.org/10.1038/s41587-019-0336-3

---

## §2 — The Algorithms, Deeply Explained

This section derives each algorithm properly. We'll build intuition layer by layer: what problem
is being solved, what mathematical structure exploits the manifold assumption, and what the
algorithm is actually computing. Let's use the notation from the course bible: $n$ samples,
$p$ original dimensionality, $d$ target dimensionality, $\mathbf{X} \in \mathbb{R}^{n \times p}$,
$\mathbf{Z} \in \mathbb{R}^{n \times d}$.

---

### 2.1 Isomap — Geodesic MDS

**The mathematical problem:** Given $\mathbf{X}$, find $\mathbf{Z}$ such that
$\|\mathbf{z}_i - \mathbf{z}_j\| \approx d_{\text{geo}}(\mathbf{x}_i, \mathbf{x}_j)$ for all pairs,
where $d_{\text{geo}}$ is the geodesic distance on the underlying manifold.

**Step 1: Neighborhood Graph Construction**

Build an adjacency graph $G = (V, E)$ where vertices are data points. Two points are connected if
they are among each other's $k$ nearest neighbors (kNN graph) or within a ball of radius $\epsilon$
(epsilon-ball graph). Edge weights equal the Euclidean distance:

$$w_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\| \quad \text{if } j \in \mathcal{N}_k(i), \text{ else } w_{ij} = \infty$$

The key insight: for nearby points on a smooth manifold, the Euclidean distance is a good
approximation to the geodesic distance. The manifold is locally flat. We exploit this by only
using Euclidean distances for pairs that are "close enough" to be in the same flat local patch.

**Step 2: All-Pairs Shortest Paths**

Compute the shortest-path distance matrix $\mathbf{D}_G$ where $D_G(i,j)$ is the length of the
shortest path from $i$ to $j$ in $G$:

$$D_G(i,j) = \min_{\text{path } i \to j \in G} \sum_{\text{edges on path}} w_{e}$$

Two algorithms:
- **Dijkstra's**: $O(kN \log N)$ per source node → $O(kN^2 \log N)$ total. Best for sparse graphs
  (small $k$). Used when `path_method='D'`.
- **Floyd-Warshall**: $O(N^3)$. Better for dense graphs. Used when `path_method='FW'`.

**Step 3: Classical MDS on Geodesic Distances**

Classical MDS takes a distance matrix $\mathbf{D}$ and finds Euclidean coordinates. The key
algebraic move: convert squared distances to an inner-product matrix via **double centering**.

Let $\mathbf{D}_G^{(2)}$ denote the element-wise squared distance matrix. The centering matrix is
$\mathbf{H} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$. The Gram matrix is:

$$\mathbf{B} = -\frac{1}{2}\mathbf{H}\mathbf{D}_G^{(2)}\mathbf{H}$$

If the true geodesic distances were exact and came from a Euclidean embedding, $\mathbf{B}$ would
be the inner-product matrix $\mathbf{Z}\mathbf{Z}^\top$ of that embedding. We extract $\mathbf{Z}$
via eigendecomposition: $\mathbf{B} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$, and then:

$$\mathbf{Z} = \mathbf{U}_d \boldsymbol{\Lambda}_d^{1/2}$$

where $\mathbf{U}_d$ contains the top $d$ eigenvectors and $\boldsymbol{\Lambda}_d$ the
corresponding eigenvalues. If $\mathbf{B}$ has negative eigenvalues (which happens when geodesic
distances aren't exactly Euclidean), we simply discard them.

> 💡 **Intuition:** The double-centering converts "how far apart are they?" into "how aligned are
> they with the center of mass?" — turning a distance matrix into a covariance-like matrix that PCA
> can decompose. Isomap is literally PCA, but operating on geodesic distances instead of raw features.

**Computational Complexity:**

| Step | Complexity | Bottleneck? |
|------|-----------|------------|
| kNN graph | $O(pN^2)$ brute force, $O(pN \log N)$ with ball tree | Only for large $p$ |
| Shortest paths (Dijkstra) | $O(kN^2 \log N)$ | For large $N$ |
| Eigendecomposition | $O(dN^2)$ | For large $N$ |
| **Total** | **$O(N^2)$ dominated** | **Practical limit: $N \lesssim 5000$** |

**Out-of-sample extension:** Isomap (uniquely among manifold methods in sklearn) supports
`transform(X_new)` for new data points, using the fitted kNN structure to place new points
into the existing embedding.

---

### 2.2 LLE — Locally Linear Embedding

**The mathematical problem:** Find low-dimensional coordinates $\mathbf{Z}$ such that the local
linear reconstruction weights $\mathbf{W}$ that work in high-dimensional space also work in
low-dimensional space.

**Step 1: kNN Search**

Find the $k$ nearest neighbors $\mathcal{N}(i)$ for each point $\mathbf{x}_i$.

**Step 2: Weight Computation (Local Least Squares)**

For each $\mathbf{x}_i$, solve the constrained least-squares problem:

$$\min_{\mathbf{w}_i} \left\|\mathbf{x}_i - \sum_{j \in \mathcal{N}(i)} w_{ij} \mathbf{x}_j\right\|^2 \quad \text{s.t.} \quad \sum_{j \in \mathcal{N}(i)} w_{ij} = 1$$

The constraint (weights sum to 1) enforces translation invariance — the weights encode the
"shape" of the local neighborhood rather than its absolute position.

This problem has a closed-form solution. Let $\mathbf{G}_i$ be the local Gram matrix of neighbor
differences: $G_i^{(jl)} = (\mathbf{x}_i - \mathbf{x}_j)^\top (\mathbf{x}_i - \mathbf{x}_l)$
for $j, l \in \mathcal{N}(i)$. Then:

$$\mathbf{w}_i = \frac{\mathbf{G}_i^{-1} \mathbf{1}}{\mathbf{1}^\top \mathbf{G}_i^{-1} \mathbf{1}}$$

The regularization parameter `reg` adds $\epsilon \cdot \text{tr}(\mathbf{G}_i) \cdot \mathbf{I}$
to $\mathbf{G}_i$ before inversion, preventing singular matrices when the neighborhood is
degenerate (e.g., when $k > p$ so the local neighborhood is overdetermined).

**Step 3: Global Embedding (Sparse Eigenvalue Problem)**

Define the sparse weight matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$ from step 2 (mostly zeros
— each row has at most $k$ nonzeros). The embedding minimizes:

$$\Phi(\mathbf{Z}) = \sum_{i=1}^n \left\|\mathbf{z}_i - \sum_{j} W_{ij} \mathbf{z}_j\right\|^2 = \text{tr}\left(\mathbf{Z}^\top \mathbf{M} \mathbf{Z}\right)$$

where $\mathbf{M} = (\mathbf{I} - \mathbf{W})^\top (\mathbf{I} - \mathbf{W})$. Subject to
$\mathbf{Z}^\top\mathbf{Z} = \mathbf{I}$ and $\sum_i \mathbf{z}_i = \mathbf{0}$ (zero mean),
the solution is the bottom $d+1$ eigenvectors of $\mathbf{M}$, discarding the trivial constant
eigenvector (eigenvalue = 0).

Note that $\mathbf{M}$ is sparse (band structure inherited from $\mathbf{W}$), so ARPACK's
shift-invert mode finds the smallest eigenvalues efficiently without materializing the full matrix.

> 🔬 **Deep dive:** The requirement $n\_neighbors > n\_components$ is fundamental: you need at
> least $d+1$ points to span a $d$-dimensional local coordinate system. For HLLE, you need even
> more: the local quadratic form requires $k > d(d+3)/2$ neighbors to be non-degenerate.

---

### 2.3 Spectral Embedding (Laplacian Eigenmaps)

**The mathematical problem:** Find coordinates $\mathbf{Z}$ that minimize the "elastic energy"
of nearby points being stretched apart: $\min_{\mathbf{Z}} \sum_{i,j} W_{ij} \|\mathbf{z}_i - \mathbf{z}_j\|^2$.

**Step 1: Affinity Matrix**

Build a weighted adjacency matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ using either:
- **kNN binary**: $A_{ij} = 1$ if $j \in \mathcal{N}_k(i)$, else 0.
- **Heat kernel (RBF)**: $A_{ij} = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)$
  for connected pairs, 0 otherwise. Gives soft, continuous affinities.

**Step 2: Graph Laplacian**

The **unnormalized graph Laplacian** is:
$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$

where $\mathbf{D}$ is the diagonal degree matrix: $D_{ii} = \sum_j A_{ij}$.

The **normalized graph Laplacian** (sklearn's default) is:
$$\mathbf{L}_{\text{sym}} = \mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$$

**Step 3: Generalized Eigenvalue Problem**

Solve:
$$\mathbf{L}\mathbf{f} = \lambda \mathbf{D}\mathbf{f}$$

The eigenvectors corresponding to the $d$ **smallest nonzero** eigenvalues are the embedding
coordinates. The trivial eigenvector $\mathbf{1}$ (with eigenvalue 0) is discarded.

**Connection to graph clustering:** The exact same eigenvectors used for spectral embedding are
used in spectral clustering — the only difference is that spectral clustering then applies k-means
to the eigenvector matrix. This is why SpectralEmbedding and SpectralClustering are so closely
related in sklearn's API.

**Connection to heat diffusion:** The Laplacian is the infinitesimal generator of the heat
equation $\partial_t u = -\mathbf{L}u$. The eigenvectors are "modes of cooling" on the graph.
Low-frequency modes (small eigenvalues) describe smooth, large-scale variation across the graph;
high-frequency modes capture rapid local oscillations. Using only the low-frequency modes as
coordinates is a form of graph smoothing.

> 💡 **Intuition:** Imagine spreading ink on a graph. The ink flows more easily between highly
> connected nodes. After time $t$, you get a "diffusion profile" from each node. Points that have
> similar diffusion profiles — even if not directly connected — have similar embeddings. This is
> the spectral embedding objective made physical.

---

### 2.4 MDS — Three Flavors

**Classical MDS (cMDS / PCoA):** Given dissimilarity matrix $\mathbf{D}$:

$$\mathbf{B} = -\frac{1}{2}\mathbf{H}\mathbf{D}^{(2)}\mathbf{H}, \quad \mathbf{Z} = \mathbf{U}_d\boldsymbol{\Lambda}_d^{1/2}$$

This is identical to Isomap's step 3, just with different distances. Equivalent to PCA when
distances are Euclidean. Has an **analytical solution** — no iteration needed.

**Metric MDS (SMACOF):** Minimize the raw stress:

$$\sigma(\mathbf{Z}) = \sum_{i < j} \left(d_{ij} - \hat{d}_{ij}\right)^2$$

where $d_{ij} = D_{ij}$ are given dissimilarities and $\hat{d}_{ij} = \|\mathbf{z}_i - \mathbf{z}_j\|$.
SMACOF (Scaling by MAjorizing a Complicated Function) iteratively minimizes an upper bound
(majorant) of $\sigma$ — a Guttman transform step:

$$\mathbf{Z}^{(t+1)} = \frac{1}{n}\mathbf{B}(\mathbf{Z}^{(t)})\mathbf{Z}^{(t)}$$

where $\mathbf{B}(\mathbf{Z})$ is a matrix depending on the current configuration. This is
guaranteed to decrease stress monotonically, though it may converge to a local minimum.

**Non-metric MDS (Kruskal's Stress-1):** Only require that the ordering of distances is preserved.
Minimize:

$$\sigma_1(\mathbf{Z}) = \sqrt{\frac{\sum_{i<j}(\hat{d}_{ij} - \hat{\delta}_{ij})^2}{\sum_{i<j}\hat{d}_{ij}^2}}$$

where $\hat{\delta}_{ij}$ are fitted "disparities" — monotone regression values found by isotonic
regression of $\hat{d}_{ij}$ on $d_{ij}$. This alternates between SMACOF steps (fix disparities,
update coordinates) and isotonic regression steps (fix coordinates, update disparities).

**Benchmarks for Kruskal's Stress-1:** 0 = perfect, 0.025 = excellent, 0.05 = good, 0.1 = fair,
0.2 = poor (embedding unreliable).

---

### 2.5 Diffusion Maps and PHATE

**Diffusion Maps (Coifman & Lafon 2006):** Define a random walk on the data graph. Starting
from point $i$, what is the probability of reaching point $j$ in exactly $t$ steps? This
"diffusion distance" integrates over all paths, making it robust to shortcuts and noise.

**Construction:**

1. Kernel matrix: $K_{ij} = \exp\left(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / \epsilon\right)$
2. Density-normalized: $\tilde{K}_{ij} = K_{ij} / (q_i q_j)^\alpha$ where
   $q_i = \sum_j K_{ij}$ and $\alpha \in [0,1]$ controls density normalization.
3. Row-normalize to get Markov matrix: $P_{ij} = \tilde{K}_{ij} / \sum_j \tilde{K}_{ij}$
4. Eigendecompose $\mathbf{P}$: right eigenvectors $\boldsymbol{\phi}_l$, eigenvalues $\lambda_l$,
   ordered $1 = \lambda_0 \geq \lambda_1 \geq \ldots$
5. **Diffusion map at time $t$:**
   $$\Psi_t(\mathbf{x}_i) = \left(\lambda_1^t \phi_1(i), \lambda_2^t \phi_2(i), \ldots, \lambda_d^t \phi_d(i)\right)$$

The $\alpha$ parameter controls what geometry you see:
- $\alpha = 0$: graph Laplacian (density-sensitive — clusters in dense regions)
- $\alpha = 0.5$: approximates the Laplace-Beltrami operator (intrinsic geometry)
- $\alpha = 1$: Fokker-Planck normalization (removes density effects — pure geometry)

**PHATE's Innovation:** Instead of using the eigenvectors as coordinates directly, PHATE computes
an **information-geometric potential distance** between diffusion distributions:

$$\tilde{d}(i,j) = \left\|\log \mathbf{P}^t_i - \log \mathbf{P}^t_j\right\|$$

where $\mathbf{P}^t_i$ is the row of $\mathbf{P}^t$ corresponding to point $i$ — the distribution
over all other points after $t$ diffusion steps. This KL-inspired distance preserves both local
density structure *and* global trajectory structure, which raw diffusion coordinates alone cannot.
The potential distances are then embedded with metric MDS (or SGD-MDS for speed).

> 💡 **Intuition:** Diffusion maps ask "how fast does a random walker spread from here?" PHATE
> asks "given two points, how similar are the probability landscapes a random walker sees from
> each of them?" Two cells at different stages of the same developmental trajectory see very
> different futures — PHATE captures that divergence.

**Complexity summary:**

| Method | Fit complexity | Space | Practical $N$ limit |
|--------|---------------|-------|-------------------|
| Isomap | $O(kN^2 \log N + dN^2)$ | $O(N^2)$ | ~5,000 |
| LLE (all variants) | $O(kN^2 + dN^2)$ | $O(N^2)$ | ~5,000 |
| HLLE | $O(kN^2 + Nd^6 + dN^2)$ | $O(N^2)$ | ~2,000 |
| Spectral Embedding | $O(kN^2 + dN^2)$ | $O(N^2)$ | ~10,000 |
| MDS (SMACOF) | $O(N^2 \cdot \text{max\_iter})$ | $O(N^2)$ | ~1,000 |
| PHATE (landmarks) | $O(N \cdot n\_\text{landmark})$ | $O(N \cdot n\_\text{landmark})$ | ~500,000 |

---

## §3 — The Full Evolution: Variants and Advances

The original LLE and Isomap papers spawned dozens of variants over the following decade. Here are
the ones that matter in practice, with the specific problem each solves.

---

### 3.1 MLLE — Modified LLE (Zhang & Wang, NeurIPS 2006)

**The problem with standard LLE:** When the local neighborhood is nearly coplanar (which happens
when $k$ is small relative to the local intrinsic dimensionality), the weight matrix $\mathbf{G}_i$
is nearly rank-deficient. The regularization `reg` helps, but doesn't fully solve it. The
embedding can fold back on itself in these degenerate neighborhoods.

**The fix:** Use multiple linearly independent weight vectors for each neighborhood, not just one.
MLLE finds the space of all valid reconstruction weight vectors (not just the minimum-norm one)
and uses them collectively to define a more stable eigenvalue problem.

> 📜 **Citation:** Zhang, Z., & Wang, J. (2006). MLLE: Modified Locally Linear Embedding Using
> Multiple Weights. *Advances in Neural Information Processing Systems 19*, 1593–1600.
> https://proceedings.neurips.cc/paper/2006/hash/fb2606a5068901da92473666256e6e5b-Abstract.html

**When to use:** Always prefer MLLE over standard LLE for production use. Same computational cost,
more robust output. In sklearn: `method='modified'`.

```python
mlle = LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='modified', random_state=42)
Z_mlle = mlle.fit_transform(X_scaled)
```

---

### 3.2 HLLE — Hessian Eigenmaps (Donoho & Grimes, PNAS 2003)

**The problem with standard LLE:** LLE's theoretical guarantees only hold for convex manifolds.
For manifolds with non-convex parameter spaces (e.g., a half-sphere, a manifold with an open
boundary), LLE can fail to recover the correct parameterization.

**The fix:** Instead of using the weight vector that reconstructs $\mathbf{x}_i$ from its
neighbors, HLLE estimates the local Hessian of a smooth function at each neighborhood via local
PCA, then assembles these Hessians into a quadratic form whose null space is the embedding.

The key theoretical advance: HLLE's recovery guarantee applies to **any open, connected Riemannian
manifold** — it does NOT require convexity. This is a strictly larger class than Isomap's guarantee.

> 📜 **Citation:** Donoho, D. L., & Grimes, C. (2003). Hessian eigenmaps: Locally linear
> embedding techniques for high-dimensional data. *PNAS*, 100(10), 5591–5596.
> https://doi.org/10.1073/pnas.1031596100

**When to use:** Small datasets (N < 2,000) where you have reason to believe the manifold is
non-convex. HLLE is significantly slower than MLLE due to QR decompositions per neighborhood.

**The n_neighbors constraint:** You need $k > d(d+3)/2$ neighbors to estimate the Hessian
reliably. For $d=2$: $k > 5$, so $k \geq 6$. For $d=3$: $k > 9$. For $d=5$: $k > 20$.

```python
# n_neighbors must be > n_components*(n_components+3)/2
# For n_components=2: must be > 5, so use >= 6
hlle = LocallyLinearEmbedding(
    n_neighbors=12, n_components=2, method='hessian',
    hessian_tol=1e-4, random_state=42
)
Z_hlle = hlle.fit_transform(X_scaled)
```

---

### 3.3 LTSA — Local Tangent Space Alignment (Zhang & Zha, SIAM 2004)

**The insight:** At each point on a smooth manifold, the tangent space tells you which directions
are "along the manifold." If you estimate the tangent space at every point using local PCA, and
then find a global coordinate system that is consistent with all local tangent spaces, you've
unrolled the manifold.

LTSA computes the local PCA at each neighborhood to get a $d$-dimensional tangent approximation,
then solves an alignment problem: find global coordinates $\mathbf{Z}$ such that each local
tangent coordinate is a rigid (rotation + translation) transformation of the global coordinates
restricted to that neighborhood.

> 📜 **Citation:** Zhang, Z., & Zha, H. (2004). Principal Manifolds and Nonlinear Dimensionality
> Reduction via Local Tangent Space Alignment. *SIAM Journal on Scientific Computing*, 26(1),
> 313–338. https://doi.org/10.1137/S1064827502419154

**When to use:** Often produces the cleanest results on well-sampled manifolds. LTSA is frequently
the best-performing LLE variant on benchmark tasks. The conceptual basis (align tangent spaces)
is physically meaningful and interpretable.

**A key advantage over Isomap:** LTSA handles manifolds with complex global topology (including
holes and non-convex parameter spaces) more gracefully than Isomap, because it only reasons
locally about tangent spaces.

```python
ltsa = LocallyLinearEmbedding(
    n_neighbors=15, n_components=2, method='ltsa', random_state=42
)
Z_ltsa = ltsa.fit_transform(X_scaled)
```

> 🏆 **Best practice:** When using LLE-family methods, try `method='ltsa'` first, then
> `method='modified'` as a fallback. Use `method='hessian'` only when you need the theoretical
> guarantees for non-convex manifolds and can afford the compute.

---

### 3.4 Landmark Isomap (de Silva & Tenenbaum, 2002)

**The problem:** Standard Isomap computes all-pairs shortest paths in an $N \times N$ matrix.
For $N > 5,000$, this is both time- ($O(N^2 k \log N)$) and memory-prohibitive ($O(N^2)$).

**The fix:** Select $L \ll N$ **landmark points** (e.g., via random sampling or max-min
selection). Run shortest-path only from the $L$ landmarks to all points. Use Nyström
approximation to place all $N$ points into the embedding defined by the $L$ landmarks.

> 📜 **Citation:** de Silva, V., & Tenenbaum, J. B. (2002). Global versus local methods in
> nonlinear dimensionality reduction. *Advances in Neural Information Processing Systems 15*.

**In sklearn:** Not directly implemented, but you can approximate it:
```python
# Approximate landmark Isomap via subsampling + Isomap.transform()
from sklearn.manifold import Isomap
import numpy as np

n_landmarks = 500
idx = np.random.choice(len(X_scaled), n_landmarks, replace=False)
X_landmarks = X_scaled[idx]

iso = Isomap(n_neighbors=10, n_components=2)
iso.fit(X_landmarks)           # fit on landmarks only
Z_all = iso.transform(X_scaled)  # place all points into the embedding
```

**PHATE's landmark extension** is more sophisticated — it uses Nyström approximation of the
diffusion kernel rather than just geodesic approximation. This is why PHATE scales to 500k cells.

---

### 3.5 Supervised Isomap and Variants

**The problem:** Standard manifold methods are unsupervised. Sometimes you have class labels and
want the embedding to reflect discriminative structure, not just geometric structure.

**Supervised Isomap:** Modify the neighborhood graph so that points of the same class are
preferentially connected (or penalize edges between classes). Several formulations exist:

- **Class-conditional geodesics**: use separate graphs per class, then align them.
- **Supervised metric**: replace Euclidean distances with a label-weighted metric before building
  the kNN graph (e.g., weight inter-class edges by a penalty factor).

**In sklearn:** No built-in supervised version. Approach: use Metric Learning (e.g., `metric-learn`
package) to learn a Mahalanobis distance from labels, then pass it to Isomap as a precomputed
distance matrix or custom metric.

```python
# Approximate supervised Isomap via class-aware kNN graph
from sklearn.manifold import Isomap
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# Penalize cross-class edges: multiply by a large constant
D = euclidean_distances(X_scaled)
# For each pair (i,j) with different labels, scale up the distance
same_class = (y[:, None] == y[None, :]).astype(float)
penalty = 10.0
D_supervised = D * (same_class + penalty * (1 - same_class))

iso = Isomap(n_neighbors=10, n_components=2)
# Pass as precomputed: must compute graph manually
# sklearn Isomap doesn't have dissimilarity='precomputed' directly,
# but you can use KernelPCA or a manual Gram matrix approach
```

---

### 3.6 Multiscale PHATE (2022)

**The innovation:** Standard PHATE gives one embedding at one diffusion timescale. Multiscale
PHATE applies "diffusion condensation" — progressively merging points that are similar at each
diffusion timescale — to create a hierarchical tree of data granularity. This allows you to
zoom in and out of the data structure, revealing structure at multiple scales simultaneously.

**Applied to:** 54 million single cells from 168 COVID-19 patients (Moon et al. 2022,
*Nature Biotechnology* extension). Predicted disease outcome (mortality) better than standard
single-scale embeddings by capturing both fine-grained cell population and coarse-grained
disease trajectory simultaneously.

```python
# Requires phate >= 1.0 with multiscale support  # verify in current docs
import phate

phate_op = phate.PHATE(n_components=2, t='auto', random_state=42)
phate_op.fit(X)

# Access the multiscale structure via the condensation tree
# (available as phate_op.optimal_t and associated eigenvalues)
```

---

### 3.7 Diffusion Maps with Auto Bandwidth (pydiffmap)

**The problem:** Choosing the kernel bandwidth $\epsilon$ for diffusion maps is critical and
non-obvious. Too small: graph is disconnected. Too large: local structure lost.

**The fix:** Berry, Harlim, and Giannakis (2015) developed an algorithm to automatically select
$\epsilon$ based on a log-log plot of $\sum_{ij} K_{ij}^{(\epsilon)}$ vs. $\log \epsilon$.
The optimal $\epsilon$ corresponds to the inflection point of this curve.

```python
from pydiffmap import diffusion_map as dm

# 'bgh' = Berry-Harlim-Giannakis auto bandwidth selection
dmap = dm.DiffusionMap.from_sklearn(
    n_evecs=2,
    epsilon='bgh',   # auto bandwidth
    alpha=0.5,       # Laplace-Beltrami normalization
    k=64             # kNN for sparse kernel
)
dmap.fit_transform(X_scaled)
```

---

## §4 — Hyperparameters: The Complete Guide

### 4.1 Isomap Hyperparameters

#### `n_neighbors` — The Most Critical Parameter

This single parameter controls the trade-off between local accuracy and global connectivity. Get
it wrong and your entire embedding is wrong.

**What it controls:** How many nearest neighbors define each node's local neighborhood in the
graph. Small $k$ → the graph is sparse; only nearby points are connected. Large $k$ → more edges,
potentially connecting points that aren't truly neighbors on the manifold (short-circuit errors).

**The short-circuit problem:** If $k$ is too large, an edge may be added between two points that
are geometrically distant on the manifold but happen to be close in Euclidean space (because the
manifold doubles back on itself, as in the Swiss Roll). The shortest path then passes through
this illegal shortcut, making two distant manifold regions appear close in the embedding. This is
the most common Isomap failure mode and it is invisible without careful inspection.

**The disconnected graph problem:** If $k$ is too small, the graph may have multiple disconnected
components. Points in different components have $\infty$ geodesic distance; sklearn will warn you.

```python
from sklearn.manifold import Isomap
from scipy.sparse.csgraph import connected_components
import numpy as np

def check_isomap_connectivity(X, k):
    iso = Isomap(n_neighbors=k, n_components=2)
    iso.fit(X)
    G = iso.nbrs_.kneighbors_graph(mode='connectivity')
    n_components, _ = connected_components(G)
    return n_components

# Find minimum k that gives a connected graph
for k in range(3, 30):
    n_comp = check_isomap_connectivity(X_scaled, k)
    print(f"k={k}: {n_comp} connected component(s)")
    if n_comp == 1:
        print(f"  --> Minimum connected k: {k}")
        break
```

| n_neighbors | Effect | When to use |
|-------------|--------|-------------|
| < 5 | Disconnected graph; NaN in embedding | Never; always check |
| 5–10 | Sparse graph; accurate for high-curvature manifolds | Dense, low-noise data |
| 10–20 | Good general-purpose range | **Start here** |
| 20–50 | Smoother geodesics; short-circuit risk | Dense, noisy data |
| > 50 | Almost certainly short-circuits | Avoid unless N is very large |

**Tuning heuristic:** `n_neighbors ≈ sqrt(N)` as a first guess. Then use `reconstruction_error()`
to sweep: plot error vs. $k$ and find the minimum — that's the sweet spot.

#### `n_components` — Target Dimensionality

Use the reconstruction error elbow method:
```python
errors = [Isomap(n_neighbors=10, n_components=d).fit(X_scaled).reconstruction_error()
          for d in range(1, 15)]
# The first elbow in this plot is your intrinsic dimensionality estimate
```

#### `eigen_solver`

- `'auto'`: sklearn picks ARPACK for large N, LAPACK for small. Trust the auto.
- `'arpack'`: Iterative, sparse. Good for large N. Tune `tol` and `max_iter` if it doesn't converge.
- `'dense'`: Full eigendecomposition (LAPACK). Exact but $O(N^3)$ — only for N < 1,000.

#### `path_method`

- `'auto'`: Dijkstra for sparse (small k), Floyd-Warshall for dense. Trust the auto.
- `'D'`: Dijkstra, $O(kN^2 \log N)$. Better for most cases.
- `'FW'`: Floyd-Warshall, $O(N^3)$. Use only if your graph is very dense.

---

### 4.2 LLE / MLLE / HLLE / LTSA Hyperparameters

#### `n_neighbors`

Same local-vs-global trade-off as Isomap, with an additional hard constraint: the local weight
system must be solvable, which requires $k > d$ (more neighbors than dimensions).

| method | Minimum n_neighbors | Recommended start |
|--------|-------------------|------------------|
| 'standard' | > n_components | n_components + 3 |
| 'modified' | > n_components | n_components + 3 |
| 'hessian' | > n_components*(n_components+3)/2 | Verify this formula! |
| 'ltsa' | > n_components | n_components + 3 |

The most common LLE error (`LinAlgWarning: Matrix is exactly singular`) is caused by either
$k$ being too small OR duplicate data points. Check both.

#### `reg` — Regularization

Controls how much we regularize the local Gram matrix to prevent singularity.

- Default `1e-3` is appropriate for most cases.
- Increase to `1e-2` or `1e-1` if you're getting singular matrix warnings.
- Too large: the embedding loses local geometric accuracy (reconstruction weights become uniform).

**Optuna search:**
```python
import optuna
from sklearn.manifold import LocallyLinearEmbedding

def lle_objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 5, 30)
    reg = trial.suggest_float('reg', 1e-4, 1e-1, log=True)
    method = trial.suggest_categorical('method', ['modified', 'ltsa'])
    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, reg=reg,
        method=method, random_state=42
    )
    lle.fit(X_scaled)
    return lle.reconstruction_error_

study = optuna.create_study(direction='minimize')
study.optimize(lle_objective, n_trials=50)
print(study.best_params)
```

#### `method` — Algorithm Variant

This is the most impactful choice after `n_neighbors`:

| method | Stability | Speed | Theoretical guarantee | Best for |
|--------|-----------|-------|----------------------|---------|
| 'standard' | Medium | Fast | Convex manifolds | Quick prototypes |
| 'modified' | High | Fast | Same as standard | **Production default** |
| 'hessian' | High | Slow | Non-convex manifolds | Small datasets, theory |
| 'ltsa' | High | Fast | Same as standard | Complex manifolds |

---

### 4.3 SpectralEmbedding Hyperparameters

#### `affinity`

The affinity matrix determines what "nearby" means. This choice is as important as `n_neighbors`.

- **`'nearest_neighbors'`**: Binary affinity — connected or not. Simple, sharp boundaries.
  Sensitive to `n_neighbors`.
- **`'rbf'`**: Gaussian kernel — soft, continuous affinity. Sensitive to `gamma`.
  Better when you expect smooth density.
- **`'precomputed'`**: Pass your own affinity matrix. Use this when you have domain-specific
  similarity (e.g., cosine similarity for text, string kernel for sequences, graph kernel).

```python
# Auto-tuning gamma for RBF affinity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

D = euclidean_distances(X_scaled)
sigma = np.median(D[D > 0])  # median non-zero pairwise distance
gamma_auto = 1.0 / (2 * sigma**2)

se = SpectralEmbedding(n_components=2, affinity='rbf', gamma=gamma_auto, random_state=42)
```

#### `n_neighbors` (for 'nearest_neighbors' affinity)

- Default: `max(n_samples/10, 1)` — can be very large for small datasets!
- Recommended: 5–30, same considerations as other methods.
- If the default gives a disconnected graph (warning from sklearn), increase it.

#### `eigen_solver`

- `None` → defaults to 'arpack'. Generally fine.
- `'amg'` (algebraic multigrid): fastest for large sparse problems. Requires `pip install pyamg`.
- `'lobpcg'`: good middle ground for medium dense problems.

---

### 4.4 MDS Hyperparameters

> ⚠️ **API Warning:** The `metric` and `dissimilarity` parameters were renamed in sklearn 1.8.
> See the version-safe code in §0. All examples below use the sklearn 1.5.x API.

#### `metric` (True/False in 1.5.x; `metric_mds` in 1.8+)

- `True` (metric MDS): minimizes raw stress. Assumes dissimilarities are ratio-scaled.
- `False` (non-metric MDS): minimizes rank-order stress. More flexible; robust to outlier distances.

**Rule of thumb:** Use non-metric (`metric=False`) for exploratory work and when dissimilarities
come from subjective ratings or ordinal measurements. Use metric (`metric=True`) when you need
to interpret absolute distance values in the embedding.

#### `n_init` — Number of SMACOF Initializations

SMACOF is not guaranteed to find the global minimum — it can get stuck in local minima that
produce "twisted" or "folded" embeddings. `n_init` runs SMACOF multiple times with different
random starts and returns the best result.

- Default in 1.5.x: 4 (changed to 1 in 1.9! — a potential regression in accuracy)
- **Recommendation for final results:** Use `n_init=8` or higher. The compute is parallelizable
  with `n_jobs=-1`.

```python
# Robust MDS with many initializations
mds = MDS(
    n_components=2, metric=False, n_init=16,
    dissimilarity='euclidean', normalized_stress=True,
    random_state=42, n_jobs=-1  # parallel initializations
)
```

#### `eps` — SMACOF Convergence Tolerance

- Default in 1.5.x: `1e-3` (changed to `1e-6` in 1.9!)
- The 1.9 change makes SMACOF run more iterations but converge more reliably.
- If your MDS results are noisy, try reducing `eps` to `1e-6` manually even on 1.5.x.

---

### 4.5 PHATE Hyperparameters

#### `knn` — Number of Nearest Neighbors

Same conceptual role as `n_neighbors` in other methods. Controls the graph connectivity.
Typical range: 3–30. Default 5 is good for dense biological data. For noisy data, try 10–15.

#### `t` — Diffusion Time

Controls how many steps the random walk takes. At $t=1$, you see only one-hop connections.
At large $t$, the random walk smooths out everything. `t='auto'` uses the knee of the Von
Neumann entropy curve — highly recommended.

```python
# Manually explore t's effect
import phate
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, t in zip(axes, [1, 5, 20, 50]):
    phate_op = phate.PHATE(n_components=2, knn=5, t=t, random_state=42, verbose=0)
    Z = phate_op.fit_transform(X)
    ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=5, alpha=0.6)
    ax.set_title(f't={t}', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('PHATE: Effect of Diffusion Time t', fontsize=12)
plt.tight_layout()
```

#### `decay` — Kernel Bandwidth (Alpha-Decay)

Controls how quickly the kernel decays with distance. Higher `decay` → sharper kernel → more
sensitive to fine-grained local structure. Default 40 is calibrated for scRNA-seq. For other
data types, try lower values (10–20 for mass cytometry, flow cytometry).

---

### Tuning Playbook Table

| Method | Primary parameter | Range | Metric | Strategy |
|--------|-----------------|-------|--------|---------|
| Isomap | `n_neighbors` | 5–50 | `reconstruction_error()` | Sweep; find minimum |
| Isomap | `n_components` | 1–15 | `reconstruction_error()` | Elbow plot |
| LLE all | `n_neighbors` | 5–30 | `reconstruction_error_` | Sweep; watch for warnings |
| LLE all | `reg` | 1e-4 to 1e-1 | Downstream task | Log-scale Optuna |
| LLE all | `method` | categorical | Trustworthiness | Try LTSA first |
| SpectralEmb | `n_neighbors` | 5–30 | Trustworthiness | Similar to Isomap |
| SpectralEmb | `gamma` (RBF) | 1e-3 to 10 | Trustworthiness | Median heuristic |
| MDS | `n_components` | 1–10 | `stress_` | Stress elbow |
| MDS | `n_init` | 4–16 | Lowest `stress_` | More is better |
| PHATE | `knn` | 3–20 | Trustworthiness | Similar to n_neighbors |
| PHATE | `t` | 'auto' or 5–100 | VN entropy knee | Use 'auto' |
| PHATE | `decay` | 10–60 | Domain knowledge | Biology: 40; other: tune |

---

## §5 — Strengths

### 5.1 Genuinely Nonlinear — No Linearity Assumption

Unlike PCA, LDA, and ICA, manifold methods make no assumption that the data's structure is a
linear subspace. They can unfold Swiss Rolls, recover circular manifolds, and embed toroidal
structures that would look like mush to any linear method. This is the defining strength.

**Mechanism:** By working with graph distances (Isomap), local weights (LLE), graph Laplacian
eigenvectors (Spectral Embedding), or diffusion operators (PHATE), these methods operate on the
*intrinsic* geometry of the data cloud — not on its coordinates in ambient space.

### 5.2 Isomap: Globally Consistent, Principled Geometry

Isomap's embedding is globally consistent: every pair of points has a well-defined geodesic
distance estimate, and the embedding is the unique best Euclidean approximation to those distances.
If the manifold is truly isometric to Euclidean space, Isomap is theoretically *exact* (as
$N \to \infty$). The theoretical guarantee gives you something to rely on.

**Consequence for interpretability:** When Isomap works well, the embedding axes often have
physical interpretability (as in the original face pose / lighting demo). This makes Isomap
valuable when you want to understand what the axes mean.

### 5.3 LLE: Sparse, Efficient Eigenvalue Problem

LLE's global embedding step solves for the bottom eigenvectors of the sparse matrix $\mathbf{M}$.
ARPACK's shift-invert solver handles this without materializing the full $N \times N$ dense matrix,
making LLE memory-efficient relative to its theoretical $O(N^2)$ cost.

### 5.4 Spectral Embedding: Directly Linked to Clustering

Because Spectral Embedding and Spectral Clustering share the same mathematical foundation (graph
Laplacian eigenvectors), the embedding explicitly reveals cluster structure. Cluster separation
in the Spectral Embedding is directly interpretable as disconnectedness in the affinity graph.
This is a unique advantage when your downstream task involves clustering.

### 5.5 MDS: Works on Arbitrary Dissimilarity Matrices

MDS is the only method in this family that can operate on data you never had in matrix form.
All you need is pairwise dissimilarities: expert ratings, edit distances between strings,
network hop counts, psychometric similarity judgments. This makes MDS applicable to problems
where you can't represent your data as a feature matrix at all.

### 5.6 PHATE: Preserves Both Local AND Global Structure

This is PHATE's unique selling point. t-SNE preserves local neighborhood structure but destroys
global trajectory structure. UMAP preserves both better than t-SNE but still loses some global
relationships. PHATE, through its information-geometric potential distance and MDS embedding,
explicitly preserves developmental trajectories and branching patterns that other methods lose.

### 5.7 sklearn Integration

All classical methods are Pipeline-compatible, have consistent `fit_transform()` APIs, support
`n_jobs` for parallelism in the kNN step, and integrate with sklearn's evaluation utilities
(`trustworthiness`). You can drop them into any sklearn workflow.

---

## §6 — Weaknesses & Failure Modes

### 6.1 O(N²) Memory Wall — The Fundamental Scalability Problem

Every classical method (Isomap, LLE, Spectral Embedding, MDS) stores and operates on an $N \times N$
distance or similarity matrix. For $N = 10,000$: that's 800 MB of float64. For $N = 100,000$:
80 GB. This is the hard wall.

**Detection:** Your fit call runs out of memory or takes hours.

**Mitigation:**
- Subsample to $N \leq 5,000$ and use `Isomap.transform()` for the rest.
- Use PHATE with `n_landmark=2000` for biological data.
- Switch to UMAP (covered in its own chapter) for large N — UMAP uses approximate nearest
  neighbors and stochastic gradient descent, keeping memory sub-quadratic.

> ⚠️ **Pitfall:** The $O(N^2)$ cost is deceptive — sklearn will happily start your Isomap fit
> on N=50,000 points, consume 20 GB of memory over 10 minutes, then crash. Always check N before
> fitting.

### 6.2 Isomap: Short-Circuit Errors and Topological Holes

If the manifold has a hole (torus, figure-8, Möbius strip), Isomap will connect points across
the hole with spurious short-path edges, creating a fundamentally incorrect geodesic distance
matrix. The embedding will look "pinched" or show unexpected collapses.

**Detection:** Plot the embedding colored by a known invariant. For the Swiss Roll, color by the
roll position `t` — if the gradient doesn't smoothly unfurl, you have short circuits.

**Mitigation:**
- Reduce `n_neighbors`.
- Switch to LLE or LTSA, which handle holes better.
- Use `reconstruction_error()` — short circuits cause it to increase as $k$ grows past the
  optimal value.

### 6.3 LLE: Sensitive to Noisy or Redundant Points

LLE's weight computation involves inverting a local Gram matrix. If the neighborhood is nearly
degenerate (e.g., $k$ points nearly collinear when you're embedding to 2D), the inversion is
ill-conditioned. Noise amplifies this problem.

**Detection:** `LinAlgWarning: Matrix is exactly singular` or near-zero `reconstruction_error_`
(paradoxically, numerical singularities can give artificially low reconstruction errors).

**Mitigation:**
- Increase `reg` (regularization).
- Switch to `method='modified'` (MLLE).
- Deduplicate data.

### 6.4 No Out-of-Sample Extension (LLE, Spectral Embedding, MDS)

Once fitted, LLE, Spectral Embedding, and MDS cannot embed new test points without re-fitting
from scratch. This is a fundamental mathematical limitation: the eigenvalue problem that defines
the embedding has no natural mechanism for projecting new points.

**Mitigation:**
- Isomap: use `iso.transform(X_new)` — it approximates new-point geodesic distances via the
  fitted kNN graph.
- PHATE: supports `phate_op.transform(X_new)` via Nyström extension.
- For a fully generalizable embedding: train a neural network to mimic the embedding (parametric
  manifold learning; covered in the autoencoder chapter).

### 6.5 MDS: SMACOF Local Minima

SMACOF's majorization algorithm converges to a local minimum. For small $n\_init$, different runs
may produce qualitatively different embeddings. This nondeterminism is a practical problem.

**Detection:** Run MDS twice with different seeds and compare. If the visual structure changes
significantly, SMACOF is finding different local minima.

**Mitigation:**
- Increase `n_init` to 8–16.
- Use `dissimilarity='precomputed'` with a classical MDS initialization (sklearn 1.9+:
  `init='classical_mds'`).
- Use fewer `n_components` — higher-dimensional targets have more local-minimum traps.

### 6.6 Manifold Assumption Failure

All of these methods assume your data lies on a low-dimensional manifold. If the data is:
- **Genuinely high-dimensional** (intrinsic dimensionality equals ambient dimensionality)
- **Scattered without manifold structure** (e.g., uniformly distributed in high-D space)
- **Multi-modal with disconnected clusters** (Isomap fails completely; others struggle)

...then manifold methods will produce garbage embeddings that look convincing but are meaningless.

**Detection:**
- Reconstruction error doesn't decrease as $n\_components$ increases — the "elbow" is absent.
- Trustworthiness is low (< 0.85) across all tested parameter values.
- PCA explained variance is spread across many components (no clear dimensionality reduction possible).

| Failure mode | Symptom | Mitigation |
|-------------|---------|-----------|
| Short-circuit (Isomap) | Embedding squishes distant regions together | Reduce k; use LLE |
| Disconnected graph | sklearn warning; NaN in embedding | Increase k |
| Singular weights (LLE) | LinAlgWarning; weird embedding | Increase reg; remove duplicates |
| SMACOF local min (MDS) | Run-to-run variation; twisted embedding | More n_init |
| Out-of-memory | Crash / swap thrash | Subsample; use PHATE landmarks |
| No manifold structure | Flat reconstruction error curve | Use PCA; reconsider DR goal |

---

## §7 — What I Think About When Fitting

This is the mental monologue I run through every time I fit a manifold method. If you internalize
this checklist, you'll avoid the 90% of mistakes that beginners make.

---

### Before I Touch the Data

**1. What is N?** If N > 5,000, I immediately think: "Can I use PHATE with landmarks? Should I
subsample? Is UMAP more appropriate here?" The $O(N^2)$ wall is a real constraint, not a
theoretical abstraction. I don't start fitting until I have an answer.

**2. What is p?** If p is very high (p > 1,000), I consider applying PCA first to reduce to ~50
dimensions while preserving most variance. Manifold methods using Euclidean distances in high-p
space suffer from the curse of dimensionality — all pairwise distances become similar, making
kNN meaningless. PCA preprocessing is standard practice for scRNA-seq (reduce from 20,000 genes
to 50 PCs before PHATE/Isomap).

**3. Are there NaN or Inf values?** I run `np.any(np.isnan(X))` and `np.any(np.isinf(X))` before
anything else. A single NaN in the distance matrix cascades into garbage.

**4. Are there duplicate points?** `np.unique(X, axis=0).shape[0] < X.shape[0]` means duplicates
exist. LLE will hit singular matrices. I deduplicate or add small noise.

**5. Have I standardized?** Manifold methods use pairwise distances. If feature 1 is in meters and
feature 2 is in millimeters, feature 2 dominates all distances. `StandardScaler().fit_transform()`
is almost always correct. Exception: count data (RNA-seq) where the preprocessing pipeline is
different (normalize → log1p → PCA).

**6. What do I expect the manifold to look like?** Is it a single connected surface? Does it have
branches (→ use PHATE)? Holes (→ prefer LLE over Isomap)? Multiple disconnected clusters
(→ Spectral Embedding)? Setting this expectation helps me interpret the output and choose the
right method.

---

### While I'm Fitting

**7. I check for warnings.** sklearn is verbose about problems:
   - `UserWarning: Graph is not fully connected` (Isomap/SpectralEmbedding): $k$ too small; increase it.
   - `LinAlgWarning: Matrix is exactly singular` (LLE): degenerate neighborhood; increase `reg` or `n_neighbors`.
   - `ConvergenceWarning` (MDS): SMACOF didn't converge; increase `max_iter` or reduce `eps`.

**8. For Isomap, I check connectivity before fit:**
```python
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
G = kneighbors_graph(X_scaled, n_neighbors=k, mode='connectivity')
n_comp, _ = connected_components(G)
assert n_comp == 1, f"Graph has {n_comp} components — increase k"
```

**9. For MDS, I run multiple initializations and inspect stress variance:**
```python
stresses = []
for seed in range(10):
    mds = MDS(n_components=2, metric=False, n_init=1, random_state=seed,
              dissimilarity='euclidean', normalized_stress=True)
    mds.fit_transform(X_scaled)
    stresses.append(mds.stress_)
print(f"Stress range: {min(stresses):.4f} – {max(stresses):.4f}")
# If range is large, SMACOF is finding different local minima; increase n_init
```

**10. I set `random_state=42` everywhere.** LLE uses ARPACK with random initialization. Without a
seed, every run gives a different result. Reproducibility is non-negotiable.

---

### After Fitting — The Validation Loop

**11. Trustworthiness first.** I compute `trustworthiness(X_scaled, Z, n_neighbors=10)` before
looking at any plots. If it's below 0.85, I don't trust the embedding regardless of how
pretty it looks. I change parameters.

**12. Visual sanity check with a ground truth.** For synthetic data: color by the ground-truth
manifold parameter (`t` for Swiss Roll). For real data: color by a known label or feature that
shouldn't be embedded but is correlated with the manifold structure. If the color gradient is
smooth and logical, the embedding is working.

**13. Compare at least two methods.** Isomap + MLLE at minimum. If they broadly agree → the
structure is robust. If they disagree → I investigate why. One method's assumption is violated,
and understanding which one tells me something about my data.

**14. Reconstruction error elbow.** I always plot the elbow to confirm I've chosen the right
`n_components`. If the elbow is at $d=3$ and I embedded to $d=2$, I'm losing information. I
try 3D and see if the structure clarifies.

**15. If the embedding looks "pinched" or "folded":** This is the short-circuit signature in
Isomap. I reduce `n_neighbors`. For LLE, a folded embedding often means `reg` is too high or
`n_neighbors` is too large.

**16. If trustworthiness is good but the embedding looks random:** The manifold assumption may
not hold. I compute a PCA and check the cumulative explained variance. If 95% of variance
requires > 10 PCs, the data may be genuinely high-dimensional with no manifold structure.

---

### Decision Tree: If I See X, I Do Y

| What I see | What I do |
|-----------|-----------|
| sklearn warning: disconnected graph | Increase `n_neighbors` by 5 |
| `LinAlgWarning: singular` | Increase `reg` (LLE); or deduplicate data |
| MDS embedding changes shape between runs | Increase `n_init`; use `n_init=16, n_jobs=-1` |
| Isomap: Swiss Roll collapses | Decrease `n_neighbors`; or switch to LTSA |
| Trustworthiness < 0.85 for all k | Try a different method; check if manifold assumption holds |
| Elbow in reconstruction error at d=5 | Embed to 5D for downstream tasks; 2D only for visualization |
| Colors don't match known structure | Scaling problem → re-check StandardScaler |
| Very slow fitting (> 10 min for N=2000) | Check `eigen_solver`; use 'arpack'; reduce N |

---

## §8 — Diagnostic Plots & Evaluation

Manifold learning has no single canonical evaluation metric like classification accuracy. You
need a portfolio of diagnostics. Here are the key ones, what they tell you, and how to read them.

---

### 8.1 Trustworthiness Score

**What it measures:** Of the $k$ nearest neighbors in the embedding, what fraction were also $k$
nearest neighbors in the original space? Penalizes "intrusions" — points that appear close in
the embedding but weren't close in the original space. Think of it as precision for neighborhoods.

**Reading it:** > 0.95 = excellent; 0.90–0.95 = good; 0.85–0.90 = acceptable; < 0.85 = poor.

**Multi-scale reading:** Compute at multiple values of $k$. A method that is trustworthy at
small $k$ but not large $k$ preserves local but not global structure.

```python
from sklearn.manifold import trustworthiness
import numpy as np
import matplotlib.pyplot as plt

def plot_trustworthiness_curve(X_orig, X_embedded, label='Method', ax=None):
    """Plot trustworthiness as a function of k."""
    ks = [3, 5, 10, 15, 20, 30, 50]
    ts = [trustworthiness(X_orig, X_embedded, n_neighbors=k) for k in ks]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, ts, 'o-', label=label, linewidth=2)
    ax.axhline(0.9, ls='--', color='gray', alpha=0.5, label='0.9 threshold')
    ax.set_xlabel('k (n_neighbors)'); ax.set_ylabel('Trustworthiness')
    ax.set_ylim(0.7, 1.01); ax.legend(); ax.grid(alpha=0.3)
    return ax
```

---

### 8.2 Reconstruction Error Elbow (Isomap / LLE)

**What it measures:** How much of the geodesic distance information is captured by the embedding.
For Isomap: `1 - R²(D_geo, D_embed)`. For LLE: sum of squared reconstruction residuals.

**How to read it:** Plot error vs. `n_components`. The "elbow" — where adding dimensions gives
diminishing returns — estimates the manifold's intrinsic dimensionality.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, LocallyLinearEmbedding

def plot_reconstruction_elbow(X, method='isomap', n_neighbors=10, max_d=15):
    """Elbow plot for intrinsic dimensionality estimation."""
    errors = []
    dims = range(1, max_d + 1)

    for d in dims:
        if method == 'isomap':
            m = Isomap(n_neighbors=n_neighbors, n_components=d)
            m.fit(X)
            errors.append(m.reconstruction_error())
        elif method == 'lle':
            m = LocallyLinearEmbedding(
                n_neighbors=n_neighbors, n_components=d,
                method='modified', random_state=42
            )
            m.fit(X)
            errors.append(m.reconstruction_error_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(dims), errors, 'o-', linewidth=2, color='steelblue')
    ax.set_xlabel('n_components'); ax.set_ylabel('Reconstruction Error')
    ax.set_title(f'{method.upper()} Reconstruction Error Elbow')
    ax.grid(alpha=0.3)
    # Find approximate elbow via first-difference
    diffs = np.diff(errors)
    elbow = np.argmin(diffs) + 2   # +2 because argmin of diff array is 0-indexed from d=1
    ax.axvline(elbow, ls='--', color='red', alpha=0.7, label=f'Elbow at d={elbow}')
    ax.legend()
    plt.tight_layout()
    return elbow
```

---

### 8.3 MDS Stress Elbow

**What it measures:** How well the embedding distances match the original dissimilarities (metric
MDS) or their ordering (non-metric MDS). Lower is better.

**How to read it:** Use Kruskal's benchmarks: < 0.025 excellent, < 0.05 good, < 0.1 fair,
> 0.2 unreliable.

```python
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def plot_mds_stress_elbow(X, max_d=10, metric=False, n_init=4):
    stresses = []
    for d in range(1, max_d + 1):
        mds = MDS(
            n_components=d, metric=metric, n_init=n_init,
            dissimilarity='euclidean', normalized_stress=True, random_state=42
        )
        mds.fit_transform(X)
        stresses.append(mds.stress_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, max_d + 1), stresses, 'o-', linewidth=2, color='darkorange')
    ax.axhline(0.05, ls='--', color='green', alpha=0.7, label='0.05 (good)')
    ax.axhline(0.10, ls='--', color='orange', alpha=0.7, label='0.10 (fair)')
    ax.axhline(0.20, ls='--', color='red', alpha=0.7, label='0.20 (poor)')
    ax.set_xlabel('n_components'); ax.set_ylabel("Kruskal's Stress-1")
    ax.set_title('MDS Stress Elbow'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
```

---

### 8.4 Shepherd Diagram (Residual Scatter Plot)

**What it measures:** Direct scatterplot of original distances vs. embedding distances. A perfect
embedding would give points on the diagonal $y=x$. Deviations show which pairs of points are
poorly preserved.

**How to read it:** Points above the diagonal = compressed in embedding (far points appear close).
Points below = expanded (close points appear far). A curved relationship → nonlinear distortion.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def shepherd_diagram(X_orig, X_embedded, sample_size=1000, ax=None):
    """Scatter plot of original vs. embedding pairwise distances (subsampled)."""
    n = X_orig.shape[0]
    # Subsample pairs for efficiency
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)

    D_orig = euclidean_distances(X_orig[idx])
    D_emb  = euclidean_distances(X_embedded[idx])

    # Upper triangle only
    triu = np.triu_indices(len(idx), k=1)
    d_orig = D_orig[triu]; d_emb = D_emb[triu]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(d_orig, d_emb, s=2, alpha=0.2, color='steelblue')
    lim_max = max(d_orig.max(), d_emb.max())
    ax.plot([0, lim_max], [0, lim_max], 'r--', alpha=0.7, label='y=x (perfect)')
    ax.set_xlabel('Original distance'); ax.set_ylabel('Embedding distance')
    ax.set_title('Shepherd Diagram'); ax.legend()
    return ax
```

---

### 8.5 Comprehensive Evaluation Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import (
    Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS, trustworthiness
)
from sklearn.metrics.pairwise import euclidean_distances

# ─── Setup ────────────────────────────────────────────────────────────────────
X, color = make_swiss_roll(n_samples=1000, noise=0.05, random_state=42)
X_sc = StandardScaler().fit_transform(X)

# ─── Fit methods ──────────────────────────────────────────────────────────────
methods = {
    'Isomap':    Isomap(n_neighbors=10, n_components=2),
    'MLLE':      LocallyLinearEmbedding(n_neighbors=15, n_components=2,
                                        method='modified', random_state=42),
    'LTSA':      LocallyLinearEmbedding(n_neighbors=15, n_components=2,
                                        method='ltsa', random_state=42),
    'Spectral':  SpectralEmbedding(n_components=2, n_neighbors=10, random_state=42),
    'MDS':       MDS(n_components=2, metric=False, n_init=4,
                     dissimilarity='euclidean', normalized_stress=True, random_state=42),
}

embeddings = {}
for name, m in methods.items():
    embeddings[name] = m.fit_transform(X_sc)

# ─── Trustworthiness across k ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ks = [5, 10, 20, 30, 50]
for name, Z in embeddings.items():
    ts = [trustworthiness(X_sc, Z, n_neighbors=k) for k in ks]
    ax.plot(ks, ts, 'o-', label=name, linewidth=2)
ax.axhline(0.9, ls='--', color='gray', alpha=0.5)
ax.set_xlabel('k'); ax.set_ylabel('Trustworthiness')
ax.set_title('Trustworthiness vs k — Five Manifold Methods')
ax.legend(loc='lower left'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('trustworthiness_comparison.png', dpi=150)

# ─── Reconstruction error elbows ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
dims = range(1, 10)
iso_errors = [Isomap(n_neighbors=10, n_components=d).fit(X_sc).reconstruction_error()
              for d in dims]
lle_errors = [LocallyLinearEmbedding(n_neighbors=15, n_components=d, method='modified',
                                      random_state=42).fit(X_sc).reconstruction_error_
              for d in dims]
axes[0].plot(list(dims), iso_errors, 'o-', color='steelblue')
axes[0].set_title('Isomap Reconstruction Error'); axes[0].set_xlabel('n_components')
axes[1].plot(list(dims), lle_errors, 'o-', color='darkorange')
axes[1].set_title('MLLE Reconstruction Error'); axes[1].set_xlabel('n_components')
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reconstruction_elbows.png', dpi=150)

# ─── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'Method':<14} {'Trust@5':<10} {'Trust@20':<10}")
print('-' * 36)
for name, Z in embeddings.items():
    t5  = trustworthiness(X_sc, Z, n_neighbors=5)
    t20 = trustworthiness(X_sc, Z, n_neighbors=20)
    print(f"{name:<14} {t5:<10.4f} {t20:<10.4f}")
```

**What to expect on the Swiss Roll:** Isomap and LTSA should achieve trustworthiness > 0.96 at
k=5 on a well-sampled Swiss Roll with noise=0.05. MDS (operating on Euclidean distances in the
folded 3D space) will score much lower — demonstrating exactly why you need geodesic methods.

---

## §9 — Innovative Industry Applications

These are the applications that go beyond the standard "visualize your clusters" use case.

---

### 9.1 Robot Arm Configuration Space Recovery (Isomap — Original Demo)

**Domain:** Robotics / Configuration Space Analysis

**The problem:** A robot arm has multiple joints, each with an angle. The configuration space is
low-dimensional (two joint angles for a 2-DOF arm), but images of the arm in different
configurations are 4,096-dimensional pixel vectors. Can we recover the configuration space
structure from raw images, with no labels?

**Why Isomap:** Moving between configurations requires rotating through intermediate poses — you
can't "teleport" from one arm configuration to another. Geodesic distance correctly captures this
physical constraint. Euclidean distance in pixel space would suggest illegal shortcuts.

**Result:** The original 2000 Science paper showed Isomap recovering the 2D configuration manifold
with axes corresponding directly to the two joint angles. This was a landmark demonstration that
nonlinear dimensionality reduction can extract physically meaningful latent variables from raw data.

**Today's applications:** Sim-to-real transfer in robotic manipulation (learning robot policies
in simulation; the configuration manifold helps bridge the domain gap), robot learning from
demonstration, and joint space trajectory planning from video.

```python
# Configuration space recovery sketch
# X: (N, H*W) pixel images of robot at different configurations
# Expected embedding: 2D grid corresponding to two joint angles

iso = Isomap(n_neighbors=8, n_components=2, eigen_solver='arpack')
Z_config = iso.fit_transform(X_images)

# Validate: color by known joint angles (if available for ground truth)
plt.scatter(Z_config[:, 0], Z_config[:, 1], c=joint_angle_1, cmap='viridis', s=20)
plt.colorbar(label='Joint angle 1 (degrees)')
```

---

### 9.2 Single-Cell RNA Sequencing — Developmental Trajectories (PHATE)

**Domain:** Computational Biology / Genomics

**The problem:** During embryonic development, cells differentiate from pluripotent stem cells
into hundreds of specialized cell types. Single-cell RNA sequencing measures 20,000+ genes per
cell across tens of thousands of cells. The challenge: visualize the continuous spectrum of
differentiation states, including branching points where one lineage gives rise to two distinct
fates.

**Why PHATE:** t-SNE creates disconnected "islands" — each cell type is a blob with no visible
connections. UMAP preserves some global structure but loses branching details. PHATE was
specifically designed for this: its information-geometric potential distance reveals the continuous
flow from one cell type to another, including the exact branching geometry.

**Real impact:** Moon et al. (2019) used PHATE to reveal novel intermediate cell states in human
germ-layer differentiation not previously documented in the literature. PHATE is now standard
in 10x Genomics analysis pipelines and the Human Cell Atlas project.

```python
import phate
import scanpy as sc

# Standard single-cell preprocessing
adata = sc.read_h5ad('pbmc3k.h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.pca(adata, n_comps=50)  # PCA preprocessing is standard before PHATE

# PHATE on PCA-reduced data
phate_op = phate.PHATE(n_components=2, knn=15, t='auto', random_state=42, n_jobs=-1)
X_phate = phate_op.fit_transform(adata.obsm['X_pca'])
adata.obsm['X_phate'] = X_phate

# Visualize colored by pseudotime or cell type
sc.pl.embedding(adata, basis='phate', color=['cell_type', 'pseudotime'])
```

---

### 9.3 Neural Manifold Decoding for Brain-Machine Interfaces (Isomap / Spectral)

**Domain:** Computational Neuroscience / Medical Devices

**The problem:** A paralyzed patient's motor cortex contains neurons that "intend" to move the
arm. Recording from 100+ neurons simultaneously gives a 100+-dimensional neural activity vector.
If you can decode the intended movement direction from this neural population state in real-time,
you can control a robotic arm or cursor.

**Why manifold methods:** The neural manifold hypothesis posits that motor cortex activity lies on
a low-dimensional manifold corresponding to movement direction. Isomap/SpectralEmbedding can find
this manifold and provide a 2D or 3D coordinate system that directly corresponds to movement
parameters — without any labels.

**Real deployment:** BrainGate2 and related BMI systems use dimensionality reduction of neural
population activity for real-time cursor control. The manifold structure of hippocampal place
cells (encoding spatial location) has been mapped using Isomap, revealing the toroidal geometry
of grid cell activity.

```python
# Neural manifold decoding sketch
# X_spikes: (n_timepoints, n_neurons) — binned spike counts from neural recording
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

X_normalized = StandardScaler().fit_transform(X_spikes)
iso = Isomap(n_neighbors=15, n_components=2, eigen_solver='arpack')
Z_neural = iso.fit_transform(X_normalized)

# If movement labels available, validate:
# correlation between Z_neural[:, 0] and actual x-velocity
# correlation between Z_neural[:, 1] and actual y-velocity
```

---

### 9.4 Protein Conformation Landscape in Drug Discovery (Isomap / Diffusion Maps)

**Domain:** Structural Biology / Pharmaceutical

**The problem:** Drug target proteins adopt multiple 3D conformations. Molecular dynamics
simulations generate millions of protein "snapshots" (each described by thousands of atomic
coordinates). Understanding the energy landscape — where the protein spends most time and which
conformations transition to which others — is essential for drug design.

**Why manifold methods:** The folding pathway is a nonlinear trajectory through conformation
space. Isomap and diffusion maps find the low-dimensional "reaction coordinate" (the path along
which folding proceeds) that can be used to:
- Identify distinct metastable states (binding-competent vs. occluded conformations)
- Design mutations that bias toward the drug-binding conformation
- Run computationally cheap enhanced sampling using the learned reaction coordinate

**Real impact:** Das et al. (2006, PNAS) used nonlinear DR to characterize free energy landscapes
of small protein folding. Modern workflows combine Isomap with Markov State Models (MSMs) for
drug target characterization at companies like Schrödinger and D.E. Shaw Research.

```python
# Protein conformation analysis sketch
# X_conf: (N_frames, 3*N_atoms) — flattened atomic coordinates from MD simulation

from sklearn.manifold import Isomap
import numpy as np

# Align conformations first (remove rotation/translation) using MDAnalysis or MDTraj
# import mdtraj as md
# traj = md.load('simulation.xtc', top='protein.pdb')
# X_conf = traj.xyz.reshape(len(traj), -1)

iso = Isomap(n_neighbors=20, n_components=3, path_method='D', n_jobs=-1)
Z_folding = iso.fit_transform(X_conf)

# Color by potential energy or RMSD to native structure
plt.scatter(Z_folding[:, 0], Z_folding[:, 1], c=potential_energy, cmap='RdYlBu', s=3, alpha=0.5)
plt.colorbar(label='Potential Energy (kcal/mol)')
```

---

### 9.5 Climate Science — Nonlinear ENSO Pattern Discovery (Isomap / Spectral)

**Domain:** Atmospheric Science / Long-range Forecasting

**The problem:** El Niño / Southern Oscillation (ENSO) involves complex, nonlinear interactions
between sea surface temperature, wind patterns, and ocean circulation. Traditional PCA / EOF
analysis finds only linear variance modes. Nonlinear DR can reveal the true structure of ENSO
variability and improve long-range seasonal forecasts.

**Why manifold methods:** Isomap applied to tropical Pacific sea surface temperature anomaly data
reveals nonlinear ENSO patterns (the warm El Niño phase and cold La Niña phase are distinct
regions of a curved manifold, not just opposite ends of a linear axis). Diffusion maps can be
interpreted as approximating Koopman operator decompositions of the climate dynamical system.

```python
# Climate manifold sketch
# X_sst: (n_months, n_grid_points) — sea surface temperature anomalies
# Rows = time steps, columns = spatial grid points

from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

X_climate = StandardScaler().fit_transform(X_sst)
iso = Isomap(n_neighbors=12, n_components=3)
Z_climate = iso.fit_transform(X_climate)

# Color by month of year to reveal seasonality
# Color by ENSO index to validate recovery of El Niño structure
```

---

### 9.6 Adversarial Speech Detection via Manifold Distance (LLE / Isomap)

**Domain:** Audio Security / ASR Robustness

**The problem (2024):** Deep learning speech recognition systems are vulnerable to adversarial
attacks — carefully crafted audio perturbations that fool the ASR model while being (nearly)
imperceptible to humans. Detecting adversarial examples before they reach the model is a security
challenge for banking voice authentication, medical dictation, and air traffic control.

**The manifold approach:** Legitimate speech utterances lie on a well-defined manifold in MFCC
feature space (corresponding to the articulatory manifold — the space of physically producible
speech sounds). Adversarial examples perturb audio in ways that move it off this manifold.
A detector can measure the distance of each new utterance from the learned speech manifold.

```python
# Adversarial speech detection sketch
import librosa
import numpy as np
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

# Extract MFCCs from legitimate training audio
def extract_mfcc(audio_path, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)  # temporal average → (n_mfcc,) vector

X_legit = np.stack([extract_mfcc(p) for p in legitimate_audio_paths])
X_sc = StandardScaler().fit_transform(X_legit)

iso = Isomap(n_neighbors=10, n_components=5)
iso.fit(X_sc)

# For a new utterance, embed it and measure reconstruction error
def is_adversarial(audio_path, threshold=0.15):
    mfcc = extract_mfcc(audio_path).reshape(1, -1)
    mfcc_sc = StandardScaler().fit(X_legit).transform(mfcc)  # use train stats
    Z_new = iso.transform(mfcc_sc)    # Isomap out-of-sample extension
    # Reconstruction error for this single point
    recon = iso.reconstruction_error()
    return recon > threshold
```

---

### 9.7 Material Science — Crystal Structure Manifolds (MDS / Spectral)

**Domain:** Materials Design / Computational Chemistry

**The problem:** High-throughput computational screening generates thousands of candidate crystal
structures for battery materials, catalysts, or superconductors. Comparing crystal structures
requires structural similarity metrics (e.g., XRD fingerprint overlap, Coulomb matrix distances)
that are inherently dissimilarity matrices — not feature vectors. MDS is the natural tool.

**Why MDS:** Given a pairwise structural similarity matrix between crystal structures, non-metric
MDS creates a map of "materials space" where nearby structures are chemically similar. This map
reveals families of related materials, enables visual navigation of the design space, and
identifies "unexplored regions" likely to contain novel materials.

```python
# Materials design sketch
# D_crystal: (N_materials, N_materials) precomputed structural similarity matrix

from sklearn.manifold import MDS

# Non-metric MDS on precomputed distances (1.5.x API)
mds = MDS(
    n_components=2, metric=False, n_init=8,
    dissimilarity='precomputed',   # X is already a distance matrix
    normalized_stress=True, random_state=42, n_jobs=-1
)
Z_materials = mds.fit_transform(D_crystal)
print(f"Stress-1: {mds.stress_:.4f}")

plt.scatter(Z_materials[:, 0], Z_materials[:, 1],
            c=formation_energy, cmap='plasma', s=30, alpha=0.7)
plt.colorbar(label='Formation Energy (eV/atom)')
```

---

## §10 — Complete Python Worked Example

In this section we run all five methods on three datasets that illustrate different aspects of
manifold learning: the **S-curve** (a simple 2D manifold in 3D), the **Swiss Roll** (a more
complex 2D manifold that challenges geodesic methods), and the **digits dataset** (a real-world
10-class problem in 64 dimensions).

```python
# ─────────────────────────────────────────────────────────────────────────────
# Manifold Methods: Complete Worked Example
# Datasets: S-curve, Swiss Roll, Digits
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_swiss_roll, make_s_curve, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import (
    Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS, trustworthiness
)
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and inspect datasets
# ─────────────────────────────────────────────────────────────────────────────

# S-curve: a 2D sheet bent into an S-shape in 3D
X_s, t_s = make_s_curve(n_samples=1500, noise=0.05, random_state=42)
print(f"S-curve:     shape={X_s.shape}, color range=[{t_s.min():.2f}, {t_s.max():.2f}]")

# Swiss Roll: a 2D sheet rolled into a cylinder
X_sw, t_sw = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)
print(f"Swiss Roll:  shape={X_sw.shape}, color range=[{t_sw.min():.2f}, {t_sw.max():.2f}]")

# Digits: 1797 handwritten digit images, 64 features (8x8 pixels), 10 classes
digits = load_digits()
X_dig, y_dig = digits.data, digits.target
print(f"Digits:      shape={X_dig.shape}, classes={np.unique(y_dig)}")
print(f"Digits per class: {np.bincount(y_dig)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

scaler_s  = StandardScaler()
scaler_sw = StandardScaler()
scaler_d  = StandardScaler()

X_s_sc  = scaler_s.fit_transform(X_s)
X_sw_sc = scaler_sw.fit_transform(X_sw)
X_dig_sc = scaler_d.fit_transform(X_dig)

# Sanity check: no NaN, no duplicates
assert not np.any(np.isnan(X_s_sc)), "NaN in S-curve"
assert not np.any(np.isnan(X_sw_sc)), "NaN in Swiss Roll"
assert not np.any(np.isnan(X_dig_sc)), "NaN in Digits"

# For Digits: first check how many PCA components capture 95% variance
pca = PCA().fit(X_dig_sc)
cumvar = np.cumsum(pca.explained_variance_ratio_)
d95 = np.searchsorted(cumvar, 0.95) + 1
print(f"\nDigits: PCA needs {d95} components for 95% explained variance")
print("This is our upper bound estimate for intrinsic dimensionality")
# Typical result: ~30 components — much lower than 64, but not 2D

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fit all five methods on S-curve
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("S-CURVE ANALYSIS")
print("="*60)

methods_sc = {
    'Isomap (k=10)':
        Isomap(n_neighbors=10, n_components=2, eigen_solver='arpack'),
    'MLLE (k=12)':
        LocallyLinearEmbedding(n_neighbors=12, n_components=2,
                                method='modified', random_state=42),
    'LTSA (k=12)':
        LocallyLinearEmbedding(n_neighbors=12, n_components=2,
                                method='ltsa', random_state=42),
    'Spectral (k=10)':
        SpectralEmbedding(n_components=2, n_neighbors=10, random_state=42),
    'MDS (non-metric)':
        MDS(n_components=2, metric=False, n_init=4,
            dissimilarity='euclidean', normalized_stress=True, random_state=42),
}

embeddings_sc = {}
for name, m in methods_sc.items():
    Z = m.fit_transform(X_s_sc)
    embeddings_sc[name] = Z
    t5  = trustworthiness(X_s_sc, Z, n_neighbors=5)
    t20 = trustworthiness(X_s_sc, Z, n_neighbors=20)
    print(f"  {name:<22}: Trust@5={t5:.4f}, Trust@20={t20:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fit all five methods on Swiss Roll
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SWISS ROLL ANALYSIS")
print("="*60)

methods_sw = {
    'Isomap (k=10)':
        Isomap(n_neighbors=10, n_components=2, eigen_solver='arpack'),
    'MLLE (k=15)':
        LocallyLinearEmbedding(n_neighbors=15, n_components=2,
                                method='modified', random_state=42),
    'LTSA (k=15)':
        LocallyLinearEmbedding(n_neighbors=15, n_components=2,
                                method='ltsa', random_state=42),
    'Spectral (k=10)':
        SpectralEmbedding(n_components=2, n_neighbors=10, random_state=42),
    'MDS (non-metric)':
        MDS(n_components=2, metric=False, n_init=4,
            dissimilarity='euclidean', normalized_stress=True, random_state=42),
}

embeddings_sw = {}
for name, m in methods_sw.items():
    Z = m.fit_transform(X_sw_sc)
    embeddings_sw[name] = Z
    t5  = trustworthiness(X_sw_sc, Z, n_neighbors=5)
    t20 = trustworthiness(X_sw_sc, Z, n_neighbors=20)
    iso_re = m.reconstruction_error() if hasattr(m, 'reconstruction_error') else 'N/A'
    lle_re = m.reconstruction_error_ if hasattr(m, 'reconstruction_error_') else 'N/A'
    mds_st = m.stress_ if hasattr(m, 'stress_') else 'N/A'
    print(f"  {name:<22}: Trust@5={t5:.4f}, Trust@20={t20:.4f}")
    if iso_re != 'N/A': print(f"    --> Isomap recon error: {iso_re:.4f}")
    if lle_re != 'N/A': print(f"    --> LLE recon error_:   {lle_re:.4e}")
    if mds_st != 'N/A': print(f"    --> MDS stress-1:       {mds_st:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Digits: Intrinsic Dimensionality + Comparison
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("DIGITS DATASET ANALYSIS")
print("="*60)

# Reconstruction error elbow for Isomap on digits
print("Computing Isomap reconstruction error elbow...")
iso_errors_dig = []
dims_dig = range(1, 20)
for d in dims_dig:
    iso_d = Isomap(n_neighbors=10, n_components=d)
    iso_d.fit(X_dig_sc)
    iso_errors_dig.append(iso_d.reconstruction_error())
    print(f"  d={d:2d}: {iso_d.reconstruction_error():.4f}")

# Find approximate elbow
diffs = np.diff(iso_errors_dig)
elbow_d = int(np.argmin(diffs)) + 2
print(f"\nEstimated intrinsic dimensionality (Isomap elbow): {elbow_d}")

# Embed digits with all methods (subset N=500 for MDS speed)
N_dig = 500
idx_dig = rng.choice(len(X_dig), N_dig, replace=False)
X_dig_sub = X_dig_sc[idx_dig]; y_dig_sub = y_dig[idx_dig]

methods_dig = {
    'Isomap':    Isomap(n_neighbors=10, n_components=2),
    'MLLE':      LocallyLinearEmbedding(n_neighbors=12, n_components=2,
                                         method='modified', random_state=42),
    'Spectral':  SpectralEmbedding(n_components=2, n_neighbors=10, random_state=42),
    'MDS':       MDS(n_components=2, metric=False, n_init=4,
                     dissimilarity='euclidean', normalized_stress=True,
                     random_state=42, n_jobs=-1),
}

embeddings_dig = {}
for name, m in methods_dig.items():
    Z = m.fit_transform(X_dig_sub)
    embeddings_dig[name] = Z
    t5 = trustworthiness(X_dig_sub, Z, n_neighbors=5)
    print(f"  {name:<14}: Trust@5={t5:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization — the main figure
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3)

method_names_sw = list(embeddings_sw.keys())
method_names_dig = list(embeddings_dig.keys())

# Row 1: Swiss Roll embeddings
for col, (name, Z) in enumerate(embeddings_sw.items()):
    ax = fig.add_subplot(gs[0, col])
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=t_sw, cmap='plasma', s=6, alpha=0.7)
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('Swiss Roll', fontsize=8)

# Row 2: S-curve embeddings
for col, (name, Z) in enumerate(embeddings_sc.items()):
    ax = fig.add_subplot(gs[1, col])
    sc2 = ax.scatter(Z[:, 0], Z[:, 1], c=t_s, cmap='viridis', s=6, alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('S-Curve', fontsize=8)

# Row 3: Digits embeddings (4 methods) + elbow plot
for col, (name, Z) in enumerate(embeddings_dig.items()):
    ax = fig.add_subplot(gs[2, col])
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y_dig_sub, cmap='tab10', s=10, alpha=0.7)
    ax.set_title(f'{name}', fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('Digits', fontsize=8)

# Elbow plot in position (2, 4)
ax_elbow = fig.add_subplot(gs[2, 4])
ax_elbow.plot(list(dims_dig), iso_errors_dig, 'o-', color='steelblue', linewidth=2)
ax_elbow.axvline(elbow_d, ls='--', color='red', alpha=0.7, label=f'Elbow d={elbow_d}')
ax_elbow.set_xlabel('n_components'); ax_elbow.set_ylabel('Reconstruction Error')
ax_elbow.set_title('Isomap Elbow\n(Digits)', fontsize=9, fontweight='bold')
ax_elbow.legend(fontsize=8); ax_elbow.grid(alpha=0.3)

fig.suptitle('Manifold Methods: Swiss Roll vs S-Curve vs Digits', fontsize=14, y=1.01)
plt.savefig('manifold_comparison_full.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Hyperparameter sweep: n_neighbors effect on Isomap (Swiss Roll)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ISOMAP n_neighbors SENSITIVITY (Swiss Roll)")
print("="*60)

neighbor_values = [3, 5, 7, 10, 15, 20, 30, 50]
iso_trust = []
iso_recon = []

for k in neighbor_values:
    iso_k = Isomap(n_neighbors=k, n_components=2)
    try:
        Z_k = iso_k.fit_transform(X_sw_sc)
        trust_k = trustworthiness(X_sw_sc, Z_k, n_neighbors=10)
        recon_k = iso_k.reconstruction_error()
        iso_trust.append(trust_k)
        iso_recon.append(recon_k)
        print(f"  k={k:3d}: Trust={trust_k:.4f}, Recon={recon_k:.4f}")
    except Exception as e:
        iso_trust.append(np.nan)
        iso_recon.append(np.nan)
        print(f"  k={k:3d}: ERROR — {e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(neighbor_values, iso_trust, 'o-', linewidth=2, color='steelblue')
axes[0].set_xlabel('n_neighbors'); axes[0].set_ylabel('Trustworthiness@10')
axes[0].set_title('Trustworthiness vs k (Swiss Roll)'); axes[0].grid(alpha=0.3)
axes[1].plot(neighbor_values, iso_recon, 'o-', linewidth=2, color='darkorange')
axes[1].set_xlabel('n_neighbors'); axes[1].set_ylabel('Reconstruction Error')
axes[1].set_title('Reconstruction Error vs k (Swiss Roll)'); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('isomap_k_sensitivity.png', dpi=150)
plt.show()

print("\nInterpretation:")
print("- Trustworthiness should peak at an intermediate k (not too sparse, not too dense)")
print("- Reconstruction error should decrease then flatten (or increase if short-circuits appear)")
print("- The k with lowest reconstruction error and high trustworthiness is optimal")
```

**What you'll observe:**

- **Swiss Roll:** Isomap and LTSA will both unfurl the roll cleanly, showing a smooth 2D sheet
  colored by the roll position. MDS will produce a muddled, partially overlapping embedding —
  directly demonstrating why geodesic distances matter. MLLE may show minor artifacts at the
  edges of the roll where the reconstruction becomes degenerate.

- **S-curve:** All methods except MDS should perform well on the S-curve. It's simpler than the
  Swiss Roll because the two halves don't overlap in 3D space, making kNN less likely to produce
  short circuits.

- **Digits:** No method achieves a clean 10-class separation in 2D — this tells us that 2D is
  too low for the digits manifold (the Isomap elbow will suggest ~7–12 dimensions). Isomap tends
  to spread the classes more evenly; Spectral Embedding tends to cluster them but with more
  overlap; MLLE can produce twisted embeddings for some classes.

- **k-sensitivity:** On the Swiss Roll, you'll see trustworthiness peak around k=10–15, then
  decline as short circuits appear at k=30+. The reconstruction error will show a corresponding
  uptick at larger k values. This is the empirical demonstration of the short-circuit effect.

---

## §11 — When to Use This Algorithm (Decision Guide)

### The Core Question: Do You Have a Nonlinear Manifold?

Before reaching for manifold methods, ask: "Is my data curved?" PCA and LDA are much faster,
more interpretable, and have out-of-sample extension. Use manifold methods only when you have
evidence (or strong reason to believe) that the low-dimensional structure is nonlinear.

**Evidence of manifold structure:**
- PCA requires many more components than expected for 90% variance.
- PCA embedding shows a curved, folded, or "banana-shaped" point cloud.
- Domain knowledge: the data varies along physical parameters (pose, articulation, temperature).
- Prior literature uses manifold methods for this data type.

### Decision Table

| Your situation | Best choice | Why |
|----------------|-------------|-----|
| Single continuous manifold, geodesic accuracy matters | **Isomap** | Best global geodesic preservation; interpretable axes |
| Continuous manifold, need robust embedding | **LTSA** or **MLLE** | More stable than Isomap; handles topology better |
| Cluster structure matters as much as manifold | **Spectral Embedding** | Directly linked to spectral clustering; gap in eigenvalues reveals k |
| Have precomputed distance/similarity matrix | **MDS (precomputed)** | MDS is the only method that works on arbitrary dissimilarities |
| Ordinal comparisons only (not exact distances) | **Non-metric MDS** | Preserves rank order; robust to scaling issues |
| Single-cell biology, developmental trajectories | **PHATE** | Preserves branching structure; state-of-the-art for scRNA-seq |
| Large N (> 10,000) | **PHATE with landmarks** or UMAP | O(N²) methods will run out of memory |
| Need to embed new test points | **Isomap** or **PHATE** | Only methods in this family with native transform() |
| Need eigenvalue spectrum insight | **Spectral Embedding** | Eigenvalue gap → number of clusters |
| Noisy data | **PHATE** or **MLLE** | More robust to noise than Isomap or standard LLE |
| Small dataset (N < 500), theoretical guarantees | **HLLE** | Strongest theoretical foundation; handles non-convex parameter spaces |

### When NOT to Use Manifold Methods

- **N > 5,000**: Switch to UMAP or PHATE with landmarks. The $O(N^2)$ memory wall is fatal.
- **You need a linear model or interpretation**: Use PCA or LDA. Manifold embeddings don't have
  loadings or explained variance fractions in the same sense.
- **You need reproducible out-of-sample predictions**: Only Isomap and PHATE support `transform()`.
  For production scoring, consider parametric approaches (autoencoders).
- **You expect multiple disconnected manifolds**: Isomap fails completely; Spectral Embedding
  partially handles this but the eigenvalue structure can be hard to interpret.
- **Your data is truly high-dimensional** (intrinsic d ≈ p): Manifold methods will produce
  meaningless embeddings. Check PCA explained variance first.

### Comparison with Alternatives

| Algorithm | Global structure | Local structure | Out-of-sample | Speed (N=2000) | Interpretable axes |
|-----------|-----------------|----------------|--------------|----------------|-------------------|
| PCA | ✓✓ | ✗ | ✓ (fast) | Very fast | ✓ (loadings) |
| Isomap | ✓✓ | ✓ | ✓ (slow) | ~30s | ✓ (often) |
| LLE/MLLE | ✗ | ✓✓ | ✗ | ~20s | ✗ |
| Spectral | ✗ | ✓✓ | ✗ | ~20s | ✗ |
| MDS | ✓✓ | ✓ | ✗ | ~60s | ✗ |
| PHATE | ✓✓ | ✓✓ | ✓ | ~45s | ✗ |
| t-SNE | ✗ | ✓✓ | ✗ | ~60s | ✗ |
| UMAP | ✓ | ✓✓ | ✓ | ~5s | ✗ |

---

## §12 — Top Papers to Study

### Start Here (Foundational, Accessible)

**[MUST-READ 1]** Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A Global Geometric
Framework for Nonlinear Dimensionality Reduction. *Science*, 290(5500), 2319–2323.
https://doi.org/10.1126/science.290.5500.2319
> **Annotation:** The Isomap paper. Read it in one sitting — only 5 pages. The face pose and
> robot arm demos are still the clearest possible illustration of why geodesic distance matters.
> The theoretical guarantee (Theorem 1) is where you'll learn how Isomap converges to the true
> geometry. Essential.

**[MUST-READ 2]** Roweis, S. T., & Saul, L. K. (2000). Nonlinear Dimensionality Reduction by
Locally Linear Embedding. *Science*, 290(5500), 2323–2326.
https://doi.org/10.1126/science.290.5500.2323
> **Annotation:** The LLE paper. Published the same issue as Isomap; read them together. The
> two-step algorithm (weight computation → eigenvalue problem) is explained with elegant clarity.
> Pay attention to the section on why the constraint $\sum W_{ij} = 1$ makes the method
> invariant to affine transformations. 4 pages.

**[MUST-READ 3]** Belkin, M., & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality
Reduction and Data Representation. *Neural Computation*, 15(6), 1373–1396.
https://doi.org/10.1162/089976603321780317
> **Annotation:** The Laplacian Eigenmaps / Spectral Embedding paper. Beautifully connects
> manifold learning to spectral graph theory and the heat kernel on Riemannian manifolds. Even
> if you only ever use Spectral Embedding as a tool, understanding the connection to the heat
> equation will change how you think about what eigenvectors mean. Moderate mathematical depth.

### Read Next (Important Extensions)

**[IMPORTANT 4]** Coifman, R. R., & Lafon, S. (2006). Diffusion Maps. *Applied and Computational
Harmonic Analysis*, 21(1), 5–30. https://doi.org/10.1016/j.acha.2006.04.006
> **Annotation:** Introduces the diffusion distance framework that underlies both Diffusion Maps
> and PHATE. The key insight — that diffusion distance integrates over all paths and is robust
> to shortcuts — explains why PHATE outperforms Isomap on noisy biological data. The alpha
> parameter and its relationship to different Laplacian operators (graph Laplacian, Fokker-Planck,
> Laplace-Beltrami) is the conceptual heart of the paper. Graduate-level math required.

**[IMPORTANT 5]** Zhang, Z., & Zha, H. (2004). Principal Manifolds and Nonlinear Dimensionality
Reduction via Local Tangent Space Alignment. *SIAM Journal on Scientific Computing*, 26(1),
313–338. https://doi.org/10.1137/S1064827502419154
> **Annotation:** The LTSA paper. Once you understand the reconstruction-weight approach of LLE,
> LTSA's tangent-space-alignment approach provides a beautiful complementary geometric perspective.
> The paper includes detailed comparisons to Isomap and LLE, showing when each wins. The
> mathematical connection to SVD and PCA at the local level is illuminating.

**[IMPORTANT 6]** Moon, K. R., et al. (2019). Visualizing structure and transitions in
high-dimensional biological data. *Nature Biotechnology*, 37, 1482–1492.
https://doi.org/10.1038/s41587-019-0336-3
> **Annotation:** The PHATE paper. Exceptionally well-written for a methods paper — the
> motivation (why t-SNE and UMAP lose trajectory structure) is argued convincingly with concrete
> biological examples. The information-geometric potential distance is derived clearly. Even if
> you're not a biologist, the general principle of using KL divergence between diffusion
> distributions as a distance metric has broad applicability.

### Deep Mastery

**[DEEP 7]** Donoho, D. L., & Grimes, C. (2003). Hessian eigenmaps: Locally linear embedding
techniques for high-dimensional data. *PNAS*, 100(10), 5591–5596.
https://doi.org/10.1073/pnas.1031596100
> **Annotation:** HLLE's theoretical guarantees are the strongest in the manifold learning
> literature. Reading this paper gives you the precise conditions under which LLE-type methods
> provably recover the true manifold parameterization. The critique of Isomap (showing why
> convexity matters) is particularly illuminating. Requires differential geometry background.

**[DEEP 8]** Zhang, Z., & Wang, J. (2006). MLLE: Modified Locally Linear Embedding Using
Multiple Weights. *NeurIPS 19*, 1593–1600.
https://proceedings.neurips.cc/paper/2006/hash/fb2606a5068901da92473666256e6e5b-Abstract.html
> **Annotation:** A short, focused paper that solves a specific practical problem with LLE
> (rank deficiency) in an elegant way. The argument for using multiple weight vectors is clear
> and the proof that MLLE dominates standard LLE in all cases is convincing. Read this alongside
> the original LLE paper to understand exactly what MLLE improves.

**[DEEP 9]** Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to
a nonmetric hypothesis. *Psychometrika*, 29, 1–27. (Also: Nonmetric multidimensional scaling:
A numerical method. *Psychometrika*, 29, 115–129.) https://doi.org/10.1007/BF02289694
> **Annotation:** The 1964 MDS papers are a historical gem — elegant, readable, and surprisingly
> modern. Kruskal's definition of stress and the SMACOF-like algorithm he proposes are still
> exactly what sklearn implements today, six decades later. Read these if you want to understand
> MDS at the level of the original algorithm rather than the sklearn abstraction.

**[CONTEXT 10]** Lee, J. A., & Verleysen, M. (2007). Nonlinear Dimensionality Reduction.
Springer. (Book — full reference below in §13)
> **Annotation:** The most comprehensive academic treatment of the field. Chapter 4 (LLE and
> variants) and Chapter 6 (spectral methods) are especially valuable. The mathematical
> unification across all methods — showing how many manifold methods can be seen as special
> cases of kernel PCA — is the kind of insight that changes how you think about the entire field.

---

## §13 — Resources & Further Reading

### Books

**Lee, J. A., & Verleysen, M. (2007).** *Nonlinear Dimensionality Reduction.* Springer.
ISBN 978-0-387-39350-6.
> The academic reference. Dense but rigorous. Chapters 4–7 cover LLE, spectral methods, and
> their relationships. The section on quality criteria (trustworthiness, continuity, LCMC) is
> the mathematical foundation for what sklearn's `trustworthiness` function computes.

**Izenman, A. J. (2008).** *Modern Multivariate Statistical Techniques.* Springer.
Chapter 16: Manifold Learning.
> An accessible introduction from a statistician's perspective. Good for understanding MDS
> in the context of classical multivariate analysis.

**Géron, A. (2023).** *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (3rd ed.)*
O'Reilly. Chapter 8: Dimensionality Reduction.
> The practitioner's entry point. Less mathematical than Lee & Verleysen, but covers the
> sklearn API thoroughly and explains the intuition clearly. Excellent first resource.

### Blog Posts & Tutorials

**sklearn official documentation — Manifold learning:**
https://scikit-learn.org/stable/modules/manifold.html
> The authoritative sklearn reference. The comparison figure (all methods on Swiss Roll and S-curve)
> is essential reading. The mathematical descriptions are terse but accurate.

**Distill.pub — "How to Use t-SNE Effectively" (Wattenberg, Viégas, Johnson, 2016):**
https://distill.pub/2016/misread-tsne/
> Officially about t-SNE, but the discussion of what neighborhood preservation means and how
> to interpret (and misinterpret) embeddings applies directly to Isomap, LLE, and Spectral
> Embedding. Required reading for any manifold learning practitioner.

**PHATE documentation & tutorials:**
https://phate.readthedocs.io/
> The official PHATE docs include Jupyter notebooks covering the full scRNA-seq workflow,
> including integration with Scanpy and AnnData. The tutorial on Multiscale PHATE is especially
> good.

**"Visualizing Data using t-SNE" (van der Maaten & Hinton, 2008) — supplementary:**
https://lvdmaaten.github.io/tsne/
> The t-SNE page includes comparison experiments against Isomap and LLE that are extremely
> informative about where each method fails. The landmark comparison on the MNIST dataset is
> still the canonical benchmark.

### Online Courses

**fast.ai Practical Deep Learning for Coders, Part 2:** Includes lectures on embeddings and
manifold learning in the context of representation learning. Not manifold-learning-specific but
provides excellent intuition about why the manifold assumption matters for modern deep learning.

**scikit-learn tutorial: Manifold Learning (official):**
https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
> The compare_methods example is runnable code that benchmarks all sklearn manifold methods
> on the Swiss Roll and S-curve. Run it yourself — it's the fastest way to see the qualitative
> differences between methods.

### Package Documentation

| Package | Documentation URL |
|---------|------------------|
| sklearn.manifold | https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold |
| phate | https://phate.readthedocs.io/ |
| pydiffmap | https://pydiffmap.readthedocs.io/ |
| scanpy (diffmap) | https://scanpy.readthedocs.io/en/stable/api/scanpy.tl.diffmap.html |
| PHATE GitHub | https://github.com/KrishnaswamyLab/PHATE |

### Key Evaluation Tools

```python
# Everything you need for manifold evaluation is in sklearn.manifold
from sklearn.manifold import trustworthiness  # local structure preservation

# For continuity (the "recall" complement to trustworthiness), implement manually
# (see §8.2 in this chapter for the implementation)

# Intrinsic dimensionality via PCA explained variance
from sklearn.decomposition import PCA
pca = PCA().fit(X_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)
d_95 = np.searchsorted(cumvar, 0.95) + 1  # linear upper bound on intrinsic d

# For MDS: stress_ attribute gives Kruskal stress directly
# For Isomap: reconstruction_error() method
# For LLE: reconstruction_error_ attribute
```

---

*Chapter complete. Algorithms covered: Isomap (Tenenbaum et al. 2000), LLE + MLLE + HLLE + LTSA
(Roweis & Saul 2000; Zhang & Wang 2006; Donoho & Grimes 2003; Zhang & Zha 2004), Laplacian
Eigenmaps / Spectral Embedding (Belkin & Niyogi 2003), MDS — metric and non-metric (Kruskal 1964),
Diffusion Maps (Coifman & Lafon 2006), and PHATE (Moon et al. 2019). Python stack: sklearn 1.5.x
baseline (1.8/1.9 API changes flagged), phate 1.0.11, pydiffmap 0.2.0.1, scanpy ~1.10.*
