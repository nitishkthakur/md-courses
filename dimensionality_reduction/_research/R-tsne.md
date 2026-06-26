# Research Dossier: t-SNE and Variants
**For chapter author use only — not for publication**
**Compiled:** 2026-06-26
**Coverage:** van der Maaten & Hinton 2008 original → BH-tSNE 2014 → FIt-SNE 2019 → openTSNE → parametric t-SNE → HSNE

---

## SECTION 1: Paper Citations & Annotations

### 1.1 Foundational Papers

---

**[P1] The original t-SNE paper**
- **Full citation:** Laurens van der Maaten and Geoffrey E. Hinton, "Visualizing High-Dimensional Data Using t-SNE," *Journal of Machine Learning Research*, 9(Nov):2579–2605, 2008.
- **URL:** https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.html
- **PDF:** https://jmlr.org/papers/v9/vandermaaten08a/vandermaaten08a.pdf
- **Difficulty:** Intermediate (accessible math, some familiarity with probability distributions needed)
- **What you'll learn:** The core algorithm from SNE (Hinton & Roweis 2002) through the two key innovations that make t-SNE work: (1) symmetrizing the joint probability P so that p_ij = (p_j|i + p_i|j)/2, and (2) replacing the Gaussian kernel in the embedding space with a Student-t distribution to resolve the crowding problem. The paper includes experiments on MNIST (60k images), text documents (NYT), and gene expression data showing t-SNE outperforms PCA, Sammon mapping, Isomap, and LLE. Essential reading — the equations in the paper are clean and self-contained.

---

**[P2] Barnes-Hut-SNE (O(N log N) t-SNE)**
- **Full citation:** Laurens van der Maaten, "Accelerating t-SNE using Tree-Based Algorithms," *Journal of Machine Learning Research*, 15(Oct):3221–3245, 2014.
- **arXiv:** https://arxiv.org/abs/1301.3342
- **DOI:** Published in JMLR 15
- **Difficulty:** Intermediate-Hard (requires understanding of tree structures and numerical approximations)
- **What you'll learn:** Two algorithmic improvements that reduce complexity from O(N²) to O(N log N): (1) a vantage-point tree (VP-tree) to compute sparse pairwise similarities in the input space, only computing the k nearest neighbors (default k = 3*perplexity) rather than all N² pairs; and (2) the Barnes-Hut algorithm applied to the gradient computation, which groups distant points in the embedding space into "summary nodes" (similar to how n-body simulations use tree codes). The angle parameter θ controls the granularity of this grouping. This is what sklearn's `method='barnes_hut'` implements.

---

**[P3] FIt-SNE (O(N) t-SNE via FFT)**
- **Full citation:** George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, and Yuval Kluger, "Fast Interpolation-Based t-SNE for Improved Visualization of Single-Cell RNA-Seq Data," *Nature Methods*, 16:243–245, 2019.
- **URL:** https://www.nature.com/articles/s41592-018-0308-4
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6402590/
- **arXiv (preprint):** https://arxiv.org/abs/1712.09005
- **Difficulty:** Hard (requires understanding of polynomial interpolation and FFT)
- **What you'll learn:** FIt-SNE reduces gradient computation from O(N log N) to O(N) per iteration by recognizing that the t-SNE gradient is a convolution. The key insight: instead of computing all pairwise (1 + ||y_i - y_j||²)⁻¹ sums, interpolate onto an equispaced grid in the embedding space and use the Fast Fourier Transform to compute the convolution in O(G log G) where G is the grid size (independent of N for fixed accuracy). This makes t-SNE on 1M+ points practical. The paper demonstrated it on single-cell RNA-seq datasets with up to 1.3M cells. Originally preprinted in 2017 (arXiv 1712.09005).

---

**[P4] openTSNE library paper**
- **Full citation:** Pavlin G. Poličar, Martin Stražar, and Blaž Zupan, "openTSNE: A Modular Python Library for t-SNE Dimensionality Reduction and Embedding," *Journal of Statistical Software*, 109(3):1–30, 2024.
- **URL:** https://www.jstatsoft.org/article/view/v109i03
- **bioRxiv preprint:** https://www.biorxiv.org/content/10.1101/731877v3
- **Difficulty:** Low-Intermediate (practical paper, implementation focused)
- **What you'll learn:** How to design a modular t-SNE implementation that supports multiple acceleration backends (Barnes-Hut, FIt-SNE interpolation), flexible optimization schedules, and crucially: adding new data points to an existing embedding without re-fitting everything. The paper documents the design philosophy and benchmarks showing openTSNE is 20x+ faster than sklearn on large datasets (200k cells: sklearn >90 minutes, openTSNE <4 minutes). Also covers the key tricks from Kobak & Berens 2019 (PCA init, high learning rate).

---

**[P5] Parametric t-SNE**
- **Full citation:** Laurens van der Maaten, "Learning a Parametric Embedding by Preserving Local Structure," *Proceedings of the 12th International Conference on Artificial Intelligence and Statistics (AISTATS)*, JMLR Workshop and Conference Proceedings 5:384–391, 2009.
- **URL:** https://proceedings.mlr.press/v5/maaten09a.html
- **PDF:** https://lvdmaaten.github.io/publications/papers/AISTATS_2009.pdf
- **Difficulty:** Intermediate (neural network optimization background helpful)
- **What you'll learn:** A deep neural network (originally a stack of restricted Boltzmann machines, now updated to use standard backprop) is trained to minimize the t-SNE cost function. The network learns a parametric function f(x; θ) that maps any new point to the embedding without rerunning the full optimization. This solves t-SNE's core limitation — inability to embed unseen data — at the cost of more complex training. Modern implementations (e.g., parametric UMAP) have revived and improved this idea substantially.

---

**[P6] "How to Use t-SNE Effectively" (Wattenberg, Viégas, Johnson 2016)**
- **Full citation:** Martin Wattenberg, Fernanda Viégas, and Ian Johnson, "How to Use t-SNE Effectively," *Distill*, 2016.
- **URL:** https://distill.pub/2016/misread-tsne/
- **Difficulty:** Low (interactive, visual)
- **What you'll learn:** The definitive practitioner's guide to t-SNE misinterpretation. Six key lessons demonstrated with interactive visualizations: (1) cluster shapes in t-SNE are meaningless; (2) distances between clusters are meaningless; (3) density differences are meaningless; (4) topology can change dramatically with perplexity; (5) random noise can produce cluster-like structures; (6) you must run at multiple perplexities before concluding anything. Every practitioner must read this. It prevents the most common and dangerous mistakes.

---

**[P7] "The Art of Using t-SNE for Single-Cell Transcriptomics" (Kobak & Berens 2019)**
- **Full citation:** Dmitry Kobak and Philipp Berens, "The Art of Using t-SNE for Single-Cell Transcriptomics," *Nature Communications*, 10:5416, 2019.
- **URL:** https://www.nature.com/articles/s41467-019-13056-x
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6882829/
- **Difficulty:** Intermediate
- **What you'll learn:** Comprehensive best-practices guide developed in the context of single-cell RNA-seq data but applicable generally. Key recommendations: (1) initialize from PCA (not random) for global structure preservation and reproducibility; (2) set learning rate = N/early_exaggeration (the "auto" default in sklearn ≥1.2); (3) use perplexity ≈ N/100 as a rule of thumb for large datasets; (4) use exaggeration during the late optimization phase (after early exaggeration) to form denser clusters. Benchmark against 5 other methods on 20 real single-cell datasets. The protocol used in this paper influenced the sklearn v1.2 default changes (init='pca', learning_rate='auto').

---

**[P8] HSNE — Hierarchical SNE**
- **Full citation:** Nicola Pezzotti, Thomas Höllt, Jan van Gemert, Boudewijn P.F. Lelieveldt, Elmar Eisemann, and Anna Vilanova, "Hierarchical Stochastic Neighbor Embedding," *Computer Graphics Forum* (EuroVis 2016), 35(3):21–30, 2016.
- **URL/PDF:** https://nicola17.github.io/publications/2016_hsne/preprint.pdf
- **Nature Comms application paper:** https://www.nature.com/articles/s41467-017-01689-9 (Analyzing 107 CyTOF cells)
- **Difficulty:** Hard (hierarchical algorithm design)
- **What you'll learn:** For datasets too large for even FIt-SNE (millions to hundreds of millions of points), HSNE builds a hierarchy of k-nearest neighbor graphs at progressively coarser scales. The user starts at the coarsest scale, sees a t-SNE of a few thousand landmarks, and can zoom in to reveal sub-structure at finer scales. The landmark selection uses random walk-based area-of-influence metrics. Has been applied to 10⁷ CyTOF cells — orders of magnitude beyond what flat t-SNE can handle.

---

### 1.2 Critical Analysis Papers

**[P9] "Automated Optimized Parameters for t-SNE" (Belkina et al. 2019)**
- **Full citation:** Anna C. Belkina et al., "Automated Optimized Parameters for T-Distributed Stochastic Neighbor Embedding Improve Visualization and Analysis of Large Datasets," *Nature Communications*, 10:5415, 2019.
- **URL:** https://www.nature.com/articles/s41467-019-13055-y
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6882880/
- **What you'll learn:** Demonstrates that default t-SNE parameters (especially learning rate) are systematically wrong for large datasets. Proposes opt-SNE: automated calibration of early exaggeration and learning rate based on real-time KL divergence evaluation. Shows that using learning_rate = N/12 (now the sklearn 'auto' default) dramatically improves results for datasets >10k points.

**[P10] "Initialization is Critical for Preserving Global Data Structure" (Kobak & Linderman 2021)**
- **Full citation:** Dmitry Kobak and George C. Linderman, "Initialization Is Critical for Preserving Global Data Structure in Both t-SNE and UMAP," *Nature Biotechnology*, 39:156–157, 2021.
- **URL:** https://www.nature.com/articles/s41587-020-00809-z
- **What you'll learn:** Short but important: shows that when t-SNE is initialized from PCA coordinates (rather than random noise), the resulting embedding preserves global structure (which PCA already captured), while still revealing local non-linear structure that PCA misses. This is the theoretical basis for sklearn's init='pca' default since v1.2.

---

## SECTION 2: The Algorithm — Mathematical Deep Dive

### 2.1 From SNE to t-SNE: The Lineage

**SNE (Hinton & Roweis, 2002)** — the predecessor. SNE converts high-dimensional Euclidean distances between points into conditional probabilities:

```
p_{j|i} = exp(-||x_i - x_j||² / 2σ_i²) / Σ_{k≠i} exp(-||x_i - x_k||² / 2σ_i²)
```

And in the low-dimensional embedding:

```
q_{j|i} = exp(-||y_i - y_j||²) / Σ_{k≠i} exp(-||y_i - y_k||²)
```

The cost function is the sum of KL divergences:
```
C_SNE = Σ_i KL(P_i || Q_i) = Σ_i Σ_j p_{j|i} log(p_{j|i} / q_{j|i})
```

**Problems with SNE:**
1. **Asymmetric cost** — the gradient has different "roles" for different points; hard to optimize
2. **Crowding problem** — (explained in §2.3)
3. **Difficult optimization** — cost function is hard to minimize without tricks

**t-SNE fixes (van der Maaten & Hinton 2008):**

**Fix 1: Symmetrize P.** Instead of per-point conditional distributions, use a single joint distribution:
```
p_{ij} = (p_{j|i} + p_{i|j}) / 2N    [ensures Σ_{ij} p_{ij} = 1]
```
This simplifies the gradient and makes outliers have more effect (an outlier's p_{ij} is not near zero for all j).

**Fix 2: Use Student-t distribution for Q** (with 1 degree of freedom = Cauchy distribution):
```
q_{ij} = (1 + ||y_i - y_j||²)⁻¹ / Σ_{k≠l} (1 + ||y_k - y_l||²)⁻¹
```
This is the heavy-tail trick that solves the crowding problem.

### 2.2 The Full t-SNE Objective and Gradient

**Objective:**
```
C = KL(P || Q) = Σ_{i≠j} p_{ij} log(p_{ij} / q_{ij})
```

**Gradient** (derived by van der Maaten & Hinton):
```
∂C/∂y_i = 4 Σ_{j≠i} (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||²)⁻¹
```

**Interpretation of the gradient:** Each pair (i,j) exerts a force on point i. The direction is (y_i - y_j) — along the line connecting the two embedding points. The magnitude is (p_{ij} - q_{ij}) × (1 + ||y_i - y_j||²)⁻¹:
- If p_{ij} > q_{ij}: points are too far apart in the embedding → attractive force
- If p_{ij} < q_{ij}: points are too close in embedding → repulsive force
- The (1 + ||y_i - y_j||²)⁻¹ factor is the heavy tail: for distant pairs, the repulsive force decays slowly (compared to Gaussian)

### 2.3 The Crowding Problem — Deep Explanation

**The volume intuition:** Consider points uniformly distributed on a 10-dimensional sphere. The volume of this sphere is enormous compared to a 2D circle of the same radius. When we project from 10D to 2D, we must compress many neighbors that were at "medium distance" in 10D into a 2D plane — they all cluster together. There is simply not enough area in 2D to faithfully represent the neighborhood structure of 10D.

**Formally:** In D dimensions, the volume of a shell at radius r with thickness dr scales as r^{D-1} dr. As D grows, more and more points occupy shells at intermediate distances — neither very close nor very far. In 2D, there is no room for "medium distance" neighbors; they all get squeezed together in the center (crowding).

**The t-distribution solution:** The Student-t distribution with 1 degree of freedom (the Cauchy distribution) has tails that fall as r⁻², much slower than the Gaussian's e^{-r²}. This means that in the embedding:
- A "medium distance" in high-D maps to a **large** distance in 2D (the heavy tail accommodates it)
- The effective range of the repulsive force extends further, pushing moderately similar points further apart
- Close neighbors are still close (attractive force from p_{ij} > q_{ij})
- The net effect: clusters form with clear gaps between them

### 2.4 Perplexity — What It Really Means

Perplexity is defined as:
```
Perp(P_i) = 2^{H(P_i)}
```
where H(P_i) = -Σ_j p_{j|i} log_2(p_{j|i}) is the Shannon entropy of the conditional distribution around point i.

**Intuition:** Perplexity is the "effective number of neighbors" point i has. If perplexity = 30, the Gaussian bandwidth σ_i is adjusted (via binary search) so that point i effectively sees about 30 nearby neighbors with meaningful probability mass.

**The binary search for σ_i:** For each point i, σ_i is found numerically (binary search) so that the conditional distribution P_i has the target perplexity. This is adaptive — in dense regions, σ_i is small (only close neighbors matter); in sparse regions, σ_i is larger.

**Practical implication:** Perplexity ~ the scale at which t-SNE "looks" at the data. Low perplexity = very local (few neighbors) → may split real clusters. High perplexity = more global → may merge real clusters. The authors recommend 5–50; for N > 10,000, values like N/100 are often better.

### 2.5 The Two-Phase Optimization

**Phase 1: Early Exaggeration (default: first 250 iterations)**
- All p_{ij} are multiplied by a factor α (default: 12)
- This amplifies attractive forces: points that are similar in high-D are pulled very strongly together
- Effect: points coalesce into loose clusters, widely separated (the early exaggeration "pre-organizes" the embedding)
- Theoretical connection: early exaggeration is equivalent to power iterations of the underlying graph Laplacian (connection to spectral clustering)

**Phase 2: Normal Optimization (default: iterations 251–1000)**
- Early exaggeration factor removed (α = 1)
- The optimizer refines cluster sub-structure and inter-point relationships
- Uses gradient descent with momentum (default 0.8), which speeds convergence

**Learning rate guidance:**
- Too high → "exploding" embedding; all points end up equidistant ("ball" appearance)
- Too low → points collapse to a dense cloud with a few outliers ("compressed blob")
- The 'auto' rule: learning_rate = max(N / early_exaggeration / 4, 50) = max(N/48, 50)
  - For N=1000: lr ≈ 50 (at the floor)
  - For N=10,000: lr ≈ 208
  - For N=100,000: lr ≈ 2083

### 2.6 Computational Complexity Summary

| Method | Input similarity | Gradient | Memory | Notes |
|--------|-----------------|----------|--------|-------|
| Exact t-SNE | O(N²) | O(N²) | O(N²) | Feasible for N ≤ 5,000 |
| Barnes-Hut t-SNE | O(N log N) | O(N log N) | O(N log N) | sklearn default, N up to ~100k |
| FIt-SNE | O(N log N) | O(N) | O(N + G log G) | G = grid size; practical for N up to millions |

---

## SECTION 3: Sklearn 1.5.x API — Complete Parameter Reference

**Import:**
```python
from sklearn.manifold import TSNE
```

**Class signature (sklearn 1.5.x):**
```python
TSNE(
    n_components=2,
    *,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate='auto',
    max_iter=1000,           # renamed from n_iter in v1.5
    n_iter_without_progress=300,
    min_grad_norm=1e-07,
    metric='euclidean',
    metric_params=None,
    init='pca',              # changed from 'random' default in v1.2
    verbose=0,
    random_state=None,
    method='barnes_hut',
    angle=0.5,
    n_jobs=None,
    n_iter='deprecated',     # removed in v1.7
)
```

### Parameter Details

**`n_components`** (int, default=2)
- Dimension of the embedded space
- Almost always 2 (for visualization). Use 3 for 3D interactive plots (plotly)
- Cannot be used for dimensionality reduction for downstream ML (not invertible, non-parametric)
- Note: method='barnes_hut' requires n_components ≤ 3

---

**`perplexity`** (float, default=30.0)
- Controls the effective number of neighbors per point
- Binary-searches σ_i for each point so that 2^H(P_i) = perplexity
- Range recommendation: 5–50 for typical datasets; up to N/100 for very large N
- Must be strictly < N (number of samples)
- Interacts with: n_components (higher dimensions → may need higher perplexity)

---

**`early_exaggeration`** (float, default=12.0)
- Multiplier applied to all p_{ij} during the first ~250 iterations
- Higher values → tighter clusters with larger gaps (stronger initial organization)
- For very large datasets (N > 100k), values up to 12 still recommended
- Some practitioners use values up to 100 for datasets with very strong cluster structure
- Interacts with: learning_rate='auto' uses this value

---

**`learning_rate`** (float or 'auto', default='auto')
- Step size for gradient descent
- 'auto' computes: max(N / early_exaggeration / 4, 50)
- Manual range: [10.0, 1000.0] — but 'auto' is almost always superior
- Too high: points equidistant (looks like a ball or ring)
- Too low: points collapse to a blob
- **Version change:** Default changed from 200.0 to 'auto' in v1.2. The old default of 200 was systematically wrong for large datasets.

---

**`max_iter`** (int, default=1000)
- Maximum gradient descent steps total
- Should be at least 250 (the early exaggeration phase)
- 1000 is usually sufficient; increase to 2000-5000 for large or complex datasets
- **Version change:** Renamed from `n_iter` in v1.5. `n_iter` is deprecated and will be removed in v1.7.
- Monitor: if the embedding looks different between runs or the KL divergence is still decreasing at 1000 iterations, increase this.

---

**`n_iter_without_progress`** (int, default=300)
- Early stopping: if KL divergence doesn't decrease by >1e-5 for this many iterations (after the first 250), stop
- Checked every 50 iterations
- Set higher (500-1000) if you suspect slow convergence
- Set lower for fast exploratory runs

---

**`min_grad_norm`** (float, default=1e-7)
- Another early stopping criterion: stop if ||gradient|| < min_grad_norm
- Usually not the binding constraint (n_iter_without_progress triggers first)
- Rarely needs tuning

---

**`metric`** (str or callable, default='euclidean')
- Distance metric for computing p_{ij} in input space
- Supported: any metric from scipy.spatial.distance.pdist (see scipy docs)
- Use 'precomputed' if you already have a distance or similarity matrix
- Use a custom callable for domain-specific distances
- Common non-Euclidean choices:
  - 'cosine' — for text/NLP embeddings (high-D sparse vectors)
  - 'correlation' — for gene expression (pearson correlation of profiles)
  - 'hamming' — for binary data
  - 'precomputed' — when you have a custom kernel matrix
- Note: method='exact' supports sparse matrices (csr, csc, coo) with metric='precomputed'

---

**`metric_params`** (dict, default=None)
- Additional keyword arguments for the metric function
- Example: `metric_params={'p': 3}` for Minkowski distance with p=3

---

**`init`** ({'random', 'pca'} or ndarray, default='pca')
- Initialization of the embedding before gradient descent
- 'pca': embeds the first n_components PCA components. Better global structure, more reproducible. The current sklearn default since v1.2.
- 'random': randomly samples from N(0, 1e-4). Can get trapped in poor local optima. May be needed if metric='precomputed' (can't compute PCA from distances alone).
- ndarray: shape (n_samples, n_components) — provide your own initialization
- **Cannot use 'pca' with metric='precomputed'** — sklearn will raise an error
- **Version change:** Default changed from 'random' to 'pca' in v1.2

---

**`verbose`** (int, default=0)
- 0 = silent; 1 = progress messages (iterations and KL divergence)
- Always set verbose=1 during development to see if the embedding is converging

---

**`random_state`** (int, RandomState, or None, default=None)
- Controls reproducibility of random number generation
- Always set for reproducibility in production
- Note: even with random_state set, results may differ between sklearn versions

---

**`method`** ({'barnes_hut', 'exact'}, default='barnes_hut')
- 'barnes_hut': O(N log N) approximation using Barnes-Hut tree. Default and recommended for N > ~5000. Requires n_components ≤ 3.
- 'exact': O(N²) exact computation. More accurate but impractical for N > 5000. Supports n_components > 3. Can use sparse matrices.
- When to use 'exact': N < 5000, n_components > 3, or when you need the exact gradient (e.g., theoretical studies)

---

**`angle`** (float, default=0.5)
- **Only used when method='barnes_hut'**
- Barnes-Hut approximation threshold (θ in the original Barnes-Hut paper)
- Controls when a cluster of far-away points is treated as a single point
- θ=0 → exact (no approximation); θ=1 → very coarse approximation
- Recommended range: 0.2–0.8
- "Angle less than 0.2 has quickly increasing computation time; angle greater than 0.8 has quickly increasing error" (sklearn docs)
- Default 0.5 is a good trade-off for most use cases

---

**`n_jobs`** (int or None, default=None)
- Number of parallel CPU jobs for nearest-neighbor search
- None = 1 job; -1 = use all available CPUs
- Only affects the input similarity computation (finding k-NN), not the gradient
- No effect when metric='precomputed' or metric='euclidean' with method='exact'
- Set n_jobs=-1 for large datasets

---

### Version Change Summary (important for students)

| Version | Change |
|---------|--------|
| sklearn 1.2 | `learning_rate` default changed from 200.0 to 'auto' |
| sklearn 1.2 | `init` default changed from 'random' to 'pca' |
| sklearn 1.5 | `n_iter` renamed to `max_iter` (`n_iter` deprecated) |
| sklearn 1.7 (planned) | `n_iter` parameter removed |

**Flag for chapter author:** If showing code examples that need to run on sklearn 1.4 or earlier, use `n_iter=1000` instead of `max_iter=1000`. For sklearn 1.5+, always use `max_iter`.

---

## SECTION 4: Specialized Python Packages

### 4.1 Package Comparison Table

| Package | PyPI name | Version (mid-2026) | Speed vs sklearn | Best for | License | Maintained? |
|---------|-----------|-------------------|-----------------|----------|---------|-------------|
| sklearn TSNE | scikit-learn | 1.5.x | Baseline | Small datasets, familiarity | BSD | Yes (very active) |
| openTSNE | openTSNE | 1.0.4 | 20x faster for 200k pts | Large datasets, incremental, research | BSD-3 | Yes (active) |
| MulticoreTSNE | MulticoreTSNE | 0.1 | ~5x | Medium datasets, drop-in replacement | BSD | Mostly inactive (2018) |

### 4.2 openTSNE — Primary Recommendation

**Package:** openTSNE
**PyPI:** https://pypi.org/project/openTSNE/
**Latest version (mid-2026):** 1.0.4 (released October 2025)
**License:** BSD-3-Clause
**Maintainer:** Pavlin Poličar
**Maintenance status:** Active

**Install:**
```bash
pip install openTSNE
# or via conda:
conda install -c conda-forge opentsne
```

**Key imports:**
```python
from openTSNE import TSNE as openTSNE   # sklearn-compatible API
import openTSNE                          # for full modular API
```

**What it does better than sklearn:**

1. **Speed (20x+):** Uses FIt-SNE (FFT-accelerated) by default for large datasets; automatically switches between algorithms. 200k cells: sklearn takes >90 minutes, openTSNE takes <4 minutes.

2. **Incremental embedding:** Can embed new data into an *existing* embedding without recomputing from scratch. This is the killer feature. Critical for: production deployments, batch effects in single-cell data, adding new points to an existing visualization.

3. **Flexible exaggeration scheduling:** Can apply exaggeration during normal optimization phase (not just early phase). Useful for forming tighter, more separated clusters.

4. **Support for latest algorithms:** FIt-SNE, Barnes-Hut, and a unified interface to switch between them.

5. **Memory efficiency:** Sparse matrix support for the k-NN graph.

6. **Multiscale kernels:** Supports multi-scale similarities (mixture of Gaussians at different bandwidths) for better global/local balance.

**Key API — sklearn-compatible:**
```python
from openTSNE import TSNE as openTSNE

tsne = openTSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    n_jobs=-1,
    random_state=42,
)
embedding = tsne.fit_transform(X)
```

**Key API — modular (full control):**
```python
import openTSNE

# Step 1: Compute affinities (input space similarities)
affinities = openTSNE.affinity.PerplexityBasedNN(
    X,
    perplexity=30,
    n_jobs=-1,
    random_state=42,
)

# Step 2: Initialize
init = openTSNE.initialization.pca(X, n_components=2, random_state=42)

# Step 3: Create and optimize embedding
embedding = openTSNE.TSNEEmbedding(
    init,
    affinities,
    negative_gradient_method="fft",  # or "bh"
    random_state=42,
)

# Phase 1: Early exaggeration
embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5)

# Phase 2: Normal optimization
embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8)
```

**Incremental embedding (unique to openTSNE):**
```python
# After initial embedding is computed:
embedding_train = tsne.fit(X_train)

# Add new points without rerunning full optimization:
embedding_new = embedding_train.transform(X_test)
# or
partial_embedding = embedding_train.prepare_partial(X_test, perplexity=5)
partial_embedding.optimize(n_iter=100)
```

---

### 4.3 sklearn TSNE — When to Use

**Best for:** Teaching, small datasets (N < 10,000), when you need a single consistent API across all DR algorithms.

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate='auto',
    max_iter=1000,
    init='pca',
    random_state=42,
    n_jobs=-1,
)
X_embedded = tsne.fit_transform(X)
```

**Limitations vs openTSNE:**
- No incremental/out-of-sample embedding
- Slower on large datasets
- No FIt-SNE acceleration
- Cannot embed new data after fitting

---

### 4.4 Parametric t-SNE Options

For parametric t-SNE (neural-network-based), the recommended modern approach is:

```python
# Modern parametric approach via openTSNE with custom encoder
# (no stable dedicated package; usually implemented manually in PyTorch/TensorFlow)
# Reference: van der Maaten 2009; also see Sainburg et al. 2021 (Parametric UMAP)
```

Note: As of mid-2026, there is no single stable Python package specifically for parametric t-SNE that is actively maintained. Practitioners who need parametric t-SNE typically:
1. Use Parametric UMAP (umap-learn with parametric=True) as a superior alternative
2. Implement it manually in PyTorch with the t-SNE loss function

---

## SECTION 5: Hyperparameter Tuning Guidance

### 5.1 The Master Table

| Hyperparameter | Default | Typical Range | Key Heuristic | Too Low | Too High |
|----------------|---------|---------------|---------------|---------|----------|
| `perplexity` | 30 | 5–200+ | ≈ N/100 for large N | Artificial micro-clusters, disconnected structure | Merges real clusters, loss of local detail |
| `early_exaggeration` | 12 | 4–100 | 12 for most cases | Poor cluster separation | Clusters too tight, may mask sub-structure |
| `learning_rate` | 'auto' | 10–2000 | Use 'auto' always | Dense blob with outliers | Ball of equidistant points |
| `max_iter` | 1000 | 250–5000 | 1000 for N<100k; 2000+ for larger | Underfitting, unstable | Just more compute, usually OK |
| `angle` (BH only) | 0.5 | 0.2–0.8 | 0.5 almost always fine | Very slow | Distorted embedding |
| `n_iter_without_progress` | 300 | 150–1000 | 300 for most; increase for complex | Premature stopping | Just more compute |

### 5.2 Perplexity — Detailed Tuning Guide

**What it controls:** The bandwidth σ_i for each point's Gaussian. Approximately = the number of effective neighbors each point "sees."

**Key heuristics:**
- For N < 100: perplexity = sqrt(N) or N/5
- For N = 100–10,000: perplexity = 30–50 (the default range)
- For N = 10,000–100,000: perplexity = 50–200, or N/100
- For N > 100,000: perplexity = 500+ or use multi-scale kernels (openTSNE)
- Rule: perplexity << N (always)

**Effects:**
- **Too low (perplexity = 5):** Each point only considers ~5 neighbors. Produces many small, tight clusters. Distances within clusters are accurate; everything else is meaningless. Points with no neighbors may float as isolated singletons. Can make noise look like structure.
- **Too high (perplexity = 200 for N=1000):** Each point considers 200/1000 = 20% of data as neighbors. Over-smoothed. Real clusters merge together. Local structure is lost.
- **The golden rule (Wattenberg 2016):** Always run at 3+ perplexity values. Only trust structure that appears consistently across multiple perplexity settings.

**Optuna search space:**
```python
import optuna
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness

def objective(trial):
    perplexity = trial.suggest_float("perplexity", 5, min(200, len(X) - 1))
    early_exaggeration = trial.suggest_float("early_exaggeration", 4, 20)
    
    tsne = TSNE(
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate='auto',
        max_iter=1000,
        init='pca',
        random_state=42,
        n_jobs=-1,
    )
    X_emb = tsne.fit_transform(X)
    
    # Use trustworthiness as objective (maximize)
    return trustworthiness(X, X_emb, n_neighbors=10)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

**Note:** Optuna for t-SNE is slow because each trial runs the full t-SNE. Use max_iter=500 in Optuna runs for speed, then refit with max_iter=2000 for the best parameters. Consider using openTSNE for faster trials.

### 5.3 Early Exaggeration

**Default 12 works for most datasets.** Only tune if:
- Clusters are not well-separated in the final embedding → try higher (20–100)
- Clusters are too tight and you can't see sub-structure → try lower (4–8)

**Theoretical note:** Early exaggeration with α = 12 is equivalent (approximately) to finding the 12 nearest neighbors in the high-dimensional space during the initial phase. Higher α = more aggressive initial clustering.

### 5.4 Learning Rate

**Always use 'auto' in sklearn ≥ 1.2 unless you have a very specific reason not to.**

The 'auto' formula (max(N / early_exaggeration / 4, 50)):
```python
# For common dataset sizes:
import numpy as np
for N in [500, 1000, 5000, 10000, 50000, 100000]:
    lr = max(N / 12 / 4, 50)
    print(f"N={N:8d}  →  learning_rate = {lr:.0f}")
# N=     500  →  learning_rate = 50
# N=    1000  →  learning_rate = 50
# N=    5000  →  learning_rate = 104
# N=   10000  →  learning_rate = 208
# N=   50000  →  learning_rate = 1042
# N=  100000  →  learning_rate = 2083
```

**Diagnostics for wrong learning rate:**
- Too high: All points roughly equidistant, embedding looks like a ball or ring with no structure
- Too low: One dense central mass with a few scattered outliers

### 5.5 `max_iter` and Convergence

**Minimum:** 500 iterations total (250 EE + 250 normal)
**Recommended default:** 1000
**For large/complex datasets:** 2000–5000

**How to check convergence:**
```python
tsne = TSNE(verbose=1, max_iter=2000)  # verbose=1 prints KL div per 50 iters
X_emb = tsne.fit_transform(X)
# Watch: KL divergence should decrease and plateau
```

If the embedding looks different across random seeds at the same `max_iter`, increase `max_iter`.

---

## SECTION 6: Evaluation Metrics

### 6.1 Trustworthiness (sklearn built-in)

```python
from sklearn.manifold import trustworthiness

score = trustworthiness(X, X_embedded, n_neighbors=10, metric='euclidean')
# Returns float in [0, 1]; higher is better
# 1.0 = perfect local structure preservation
# Typical good values: > 0.9
```

**What it measures:** For each point i, are its k nearest neighbors in the embedding *also* true neighbors in the original space? Trustworthiness penalizes "false neighbors" — points that become neighbors in the embedding but were far away in the original space (precision-like).

**Formula:**
```
T(k) = 1 - (2 / (Nk(2N - 3k - 1))) × Σ_i Σ_{j ∈ U_k(i)} (r(i,j) - k)
```
where r(i,j) is the rank of point j in the original space when sorted by distance from i, and U_k(i) is the set of false k-nearest neighbors in the embedding.

### 6.2 Continuity

**Complement to trustworthiness:** Penalizes "missing neighbors" — points that were true neighbors in original space but are no longer neighbors in the embedding (recall-like).

```python
# sklearn does not directly provide continuity, but it can be computed manually
# Reference: Venna & Kaski 2006
# Trustworthiness + Continuity together = full picture of local neighborhood preservation
```

### 6.3 KL Divergence (directly from t-SNE)

```python
tsne = TSNE(max_iter=1000)
X_emb = tsne.fit_transform(X)
print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")
# Lower is better; typical range: 0.3–2.0 for well-fit embeddings
# Very high KL div (>3) suggests poor embedding or needs more iterations
```

**Important caveat:** KL divergence is NOT comparable across datasets of different sizes. Use it only to compare the same dataset across different hyperparameter settings.

### 6.4 Perplexity Sensitivity Plot (diagnostic)

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

perplexities = [5, 15, 30, 50, 100]
fig, axes = plt.subplots(1, len(perplexities), figsize=(5*len(perplexities), 4))

for ax, perp in zip(axes, perplexities):
    tsne = TSNE(perplexity=perp, init='pca', random_state=42)
    emb = tsne.fit_transform(X)
    ax.scatter(emb[:, 0], emb[:, 1], c=y, s=5, cmap='tab10', alpha=0.6)
    ax.set_title(f'perplexity={perp}')
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig('tsne_perplexity_sensitivity.png', dpi=150)
```

**What to look for:** Structure that *persists across perplexities* is real. Structure that appears only at one perplexity setting is an artifact.

### 6.5 Neighborhood Preservation (k-NN Agreement)

```python
from sklearn.neighbors import NearestNeighbors

def neighborhood_preservation(X_orig, X_emb, k=10):
    """Fraction of k-NN in embedding that are also k-NN in original space."""
    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X_orig)
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(X_emb)
    
    _, idx_orig = nbrs_orig.kneighbors(X_orig)
    _, idx_emb = nbrs_emb.kneighbors(X_emb)
    
    preservation = 0
    for i in range(len(X_orig)):
        shared = len(set(idx_orig[i][1:]) & set(idx_emb[i][1:]))
        preservation += shared / k
    
    return preservation / len(X_orig)

score = neighborhood_preservation(X, X_embedded, k=10)
print(f"10-NN preservation: {score:.3f}")  # 1.0 = perfect
```

### 6.6 What Evaluation Metrics Cannot Tell You

These metrics tell you about **local** structure only. t-SNE is explicitly designed NOT to preserve global structure (cluster distances, relative cluster sizes, ordering of well-separated groups). Do not use these metrics to argue about global structure preservation — t-SNE will score poorly on such metrics compared to PCA, and that is expected.

---

## SECTION 7: Expert Practitioner's Mental Checklist

### 7.1 Before Fitting

**Data checks:**
1. **How many samples?**
   - N < 1,000: use `method='exact'`, perplexity ≈ sqrt(N)
   - N = 1,000–10,000: `method='barnes_hut'`, default perplexity 30–50
   - N > 10,000: use openTSNE, set perplexity ≈ N/100, n_jobs=-1
   - N > 100,000: use openTSNE with FIt-SNE backend, consider UMAP

2. **How many features?**
   - If D > 50: apply PCA first to 50 components. This is standard practice in single-cell genomics and strongly recommended elsewhere.
   - Reason: (a) removes noise, (b) makes t-SNE faster (computes distances in 50D not D), (c) the PCA embedding already captures most linear structure

3. **Is the data scaled?**
   - t-SNE uses Euclidean distances by default. Features on wildly different scales will dominate the distances. Apply StandardScaler first.
   - Exception: if using metric='cosine' or 'correlation', scaling matters less

4. **Are there enough samples?**
   - perplexity must be < N. For small datasets (N < 50), perplexity = N/5 or less.
   - With very small N, t-SNE results are unreliable (too few examples to form meaningful probability distributions)

5. **Is the data appropriate for local structure preservation?**
   - If you need to preserve global structure (e.g., for downstream regression), t-SNE is wrong; use PCA or UMAP.
   - t-SNE is for visualization and exploratory analysis of local cluster structure.

**Preprocessing pipeline template:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# If D > 50 (e.g., gene expression, image features):
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
X_tsne = TSNE(perplexity=30, init='pca', random_state=42, n_jobs=-1).fit_transform(X_pca)

# If D <= 50:
X_scaled = StandardScaler().fit_transform(X)
X_tsne = TSNE(perplexity=30, init='pca', random_state=42, n_jobs=-1).fit_transform(X_scaled)
```

### 7.2 During Fitting

```python
tsne = TSNE(verbose=1, max_iter=2000)  # Always verbose=1 during development
X_emb = tsne.fit_transform(X)
```

**What to watch in the verbose output:**
- KL divergence at iteration 50, 100, 150... (should decrease)
- If KL divergence plateaus above 2.0 → embedding may be stuck; try random_state variation or increase max_iter
- If KL divergence never decreases → learning rate too high

**The smell test:** Does the final KL divergence make sense?
```python
print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")
# < 0.5 → usually very good fit
# 0.5–1.5 → typical good range
# 1.5–3.0 → may need more iterations or different perplexity
# > 3.0 → something is wrong (outliers? wrong perplexity? too few iterations?)
```

### 7.3 After Fitting — Diagnostic Checklist

**Check 1: Run at multiple perplexities (MANDATORY)**
```python
for perp in [5, 15, 30, 50, 100]:
    tsne = TSNE(perplexity=perp, init='pca', random_state=42)
    emb = tsne.fit_transform(X)
    # Plot and compare
```
Only trust structure that appears at ALL tested perplexities.

**Check 2: Trustworthiness score**
```python
from sklearn.manifold import trustworthiness
score = trustworthiness(X, X_emb, n_neighbors=10)
print(f"Trustworthiness @ k=10: {score:.3f}")
# If < 0.85: t-SNE is introducing false neighbors; investigate
```

**Check 3: Color by known labels (if available)**
- If classes are NOT separated: either (a) the classes aren't real clusters in the data, or (b) t-SNE needs different hyperparameters
- If classes ARE separated but you expected them to be mixed: same two causes

**Check 4: Run with multiple random seeds**
```python
embeddings = []
for seed in [42, 123, 456, 789, 1000]:
    tsne = TSNE(perplexity=30, random_state=seed, init='pca')
    embeddings.append(tsne.fit_transform(X))
# Compare: same cluster structure should appear across seeds (possibly rotated/reflected)
# Procrustes alignment may be needed for quantitative comparison
```

**Check 5: The cluster size warning**
- NEVER interpret cluster size in a t-SNE plot as meaningful
- NEVER interpret inter-cluster distances as meaningful
- ONLY trust: which points are in the same cluster, and whether that agrees with domain knowledge

### 7.4 Decision Tree: What I See → What I Do

```
Output looks like a dense ball/ring with no structure:
  → Learning rate too high OR max_iter too low
  → Try: learning_rate='auto', increase max_iter

Output looks like one dense blob with scattered outliers:
  → Learning rate too low
  → Try: learning_rate='auto' (should fix this)

Too many tiny clusters (>30 small dots):
  → Perplexity too low
  → Try: increase perplexity (2x or 3x current value)

All points in one or two clusters:
  → Perplexity too high
  → Try: decrease perplexity

Structure changes dramatically across random seeds:
  → max_iter too low (not converged)
  → Try: increase to 2000, use init='pca'

Clusters appear but don't match domain knowledge:
  → Could be artifact; try different perplexities
  → Check if preprocessing is appropriate (scaling, PCA)
  → Try UMAP for comparison

Embedding looks perfect but downstream ML doesn't improve:
  → Expected! t-SNE embeddings are for visualization ONLY
  → Use PCA or UMAP for features fed into downstream ML
```

---

## SECTION 8: Innovative Industry Applications

### 8.1 Single-Cell Genomics (Mass Cytometry / CyTOF)

**Domain:** Immunology, oncology
**Problem solved:** CyTOF (Cytometry by Time-of-Flight) measures 40–60 protein markers simultaneously on individual cells. A typical experiment produces 500,000–10,000,000 cells, each described by 40+ dimensions. Researchers need to identify rare cell subpopulations (immune cell types, tumor infiltrating lymphocytes, exhausted T-cells).
**Why t-SNE:** The data has genuine non-linear manifold structure (immune cell differentiation pathways form curved trajectories in high-D space). PCA misses these non-linear relationships. t-SNE reveals the topology of immune cell diversity.
**Specific technique:** FIt-SNE or HSNE for the scale (millions of cells). Initialize from PCA. Perplexity 100–500 for large datasets.
**References:** 
- Belkina et al. 2019 (Nature Comms) — t-SNE for CyTOF with opt-SNE parameters
- Pezzotti et al. 2016 (HSNE) applied to 10^7 CyTOF cells
- 2024 studies continue using CyTOF + t-SNE for tumor microenvironment analysis (50-parameter panels)

**Code sketch:**
```python
# Typical CyTOF pipeline
import openTSNE
from sklearn.preprocessing import arcsinh  # not sklearn; use: X = np.arcsinh(X / 5)

# CyTOF-specific preprocessing: arcsinh transform with cofactor 5
X_transformed = np.arcsinh(X_raw / 5)

# t-SNE on large CyTOF data
tsne = openTSNE.TSNE(
    perplexity=500,     # high for ~500k cells
    n_jobs=-1,
    random_state=42,
    metric='euclidean',
)
embedding = tsne.fit_transform(X_transformed)
```

---

### 8.2 Drug Discovery — Chemical Space Mapping

**Domain:** Pharmaceutical industry (cheminformatics)
**Problem solved:** Drug discovery requires navigating "chemical space" — a conceptual space of all possible drug-like molecules, estimated to contain 10^60 compounds. Medicinal chemists need to visualize compound libraries, find novel regions of chemical space unexplored by existing drugs, and ensure diversity in high-throughput screening collections.
**Why t-SNE:** Molecular fingerprints (Morgan fingerprints, ECFP) are binary high-dimensional vectors (~1024 bits). t-SNE with `metric='jaccard'` or `metric='tanimoto'` preserves the chemically meaningful similarity structure. Non-linear projection is essential because chemical similarity is not linear.
**Specific company example:** ACS Omega 2021 paper on Reaction Difference Fingerprints and Parametric t-SNE for exploring reaction space. Drug Discovery Maps (DDM) by van der Maaten used t-SNE on kinase inhibitors to predict cross-kinome activity.
**Innovation:** ChemPrint (bioRxiv 2024) uses t-SNE data splitting (splitting train/test by t-SNE clusters) to maximize chemical dissimilarity and test model generalization beyond interpolation.

```python
# Chemical space visualization
from sklearn.manifold import TSNE
from rdkit.Chem import MolFromSmiles, AllChem
import numpy as np

def get_morgan_fp(smiles, radius=2, nbits=1024):
    mol = MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)

X_fps = np.array([get_morgan_fp(s) for s in smiles_list])

tsne = TSNE(
    n_components=2,
    metric='jaccard',      # appropriate for binary fingerprints
    perplexity=30,
    init='random',         # can't use PCA with non-Euclidean metric in sklearn
    max_iter=1000,
    random_state=42,
)
X_chem = tsne.fit_transform(X_fps)
```

---

### 8.3 NLP — Sentence/Document Embedding Visualization

**Domain:** Natural language processing, information retrieval
**Problem solved:** Modern NLP models (BERT, Sentence-BERT, GPT embeddings) produce 768–1536 dimensional embeddings. Practitioners need to: (1) debug embedding quality, (2) discover topic clusters in document collections, (3) visualize semantic similarity structure, (4) identify model failures.
**Why t-SNE:** The semantic geometry of language embeddings is genuinely non-linear (fine-grained topic distinctions, polysemy). t-SNE with `metric='cosine'` respects the cosine geometry of language model spaces.
**Practical use:** t-SNE visualization of BERT embeddings is standard practice in NLP papers to demonstrate that a fine-tuned model has separated semantic categories.

```python
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)  # shape: (N, 384)

tsne = TSNE(
    n_components=2,
    metric='cosine',    # critical: cosine for NLP embeddings
    perplexity=30,
    init='random',      # init='pca' not meaningful for cosine metric in sklearn
    max_iter=1000,
    random_state=42,
)
emb_2d = tsne.fit_transform(embeddings)
```

---

### 8.4 Cybersecurity — Network Anomaly Detection

**Domain:** Cybersecurity, network security
**Problem solved:** Network intrusion detection systems generate millions of connection events per day. Each event is described by 40–80 features (packet sizes, timing patterns, protocol flags). t-SNE reveals the cluster structure of normal traffic and makes anomalies visually obvious as isolated points or unusual clusters.
**Why t-SNE:** Security analysts need to understand WHAT kind of anomaly is occurring (not just that one exists). t-SNE allows visual auditing of clustering algorithms' outputs and human-in-the-loop anomaly classification.
**Applications (2024-2025):**
- APT-LLM (arXiv 2502.09385): t-SNE visualizations of LLM-generated embeddings for Advanced Persistent Threats on UNSW-NB15 and CICIDS2017 datasets
- Anomaly clusters clearly separated in t-SNE despite being embedded in the same 768-D LLM space as normal traffic

```python
# Anomaly detection visualization
tsne = TSNE(
    n_components=2,
    perplexity=50,
    metric='euclidean',
    init='pca',
    max_iter=1500,
    random_state=42,
    n_jobs=-1,
)
X_2d = tsne.fit_transform(X_network_features)

# Color by anomaly score or known label
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                      c=anomaly_scores, cmap='RdYlGn_r', 
                      s=2, alpha=0.5)
plt.colorbar(scatter, label='Anomaly Score')
```

---

### 8.5 Materials Science — Crystal Structure Phase Diagrams

**Domain:** Materials science, computational chemistry
**Problem solved:** Materials researchers train ML models on large databases (Materials Project: ~150k crystals; AFLOW: ~3.5M). They need to understand how materials cluster in the learned representation space, identify compositional trends, and find "white space" (unexplored material compositions).
**Why t-SNE:** Crystal structures are described by high-dimensional representations (crystal graph embeddings, XRD patterns, 192-dim latent codes). t-SNE of these representations reveals compositional and structural similarity, identifying families of materials that share structural motifs.
**Applications (2024):** arXiv 2312.13289 uses t-SNE on deep InfoMax representations of polymorphic crystalline materials; 2D t-SNE plots colored by space group or cell volume reveal clear structural clustering; arXiv 2407.00671 uses t-SNE to validate self-supervised material representations.

---

### 8.6 Protein Structure and Function Analysis

**Domain:** Structural biology, drug target identification
**Problem solved:** Protein sequence or structural embeddings (from ESM-2, AlphaFold features, contact maps) are high-dimensional. Biologists want to cluster proteins by function, visualize evolutionary relationships, identify outliers (novel folds), or understand the landscape of a protein family.
**Why t-SNE:** Protein similarity is complex and non-linear (distant homologs have low sequence similarity but similar structure). t-SNE reveals non-linear structure-function relationships missed by linear methods.
**Applications:** Standard tool in proteomics research; widely used in papers studying protein superfamilies.

---

### 8.7 Financial Market Regime Visualization

**Domain:** Quantitative finance, risk management
**Problem solved:** Financial markets switch between "regimes" (bull market, bear market, high-volatility crisis, sector rotation). Risk managers need to identify the current regime and monitor transitions. t-SNE of multivariate return data reveals these regime clusters and can show when the current market state is moving toward an unusual region.
**Why t-SNE (not PCA):** Market regimes are non-linear phenomena. PCA can detect linear structure (e.g., factor models), but regime transitions involve non-linear interactions across assets. t-SNE reveals clusters in return-correlation structure that PCA misses.
**Specific application (arXiv 1909.03808):** "Systemic Risk Clustering of China Internet Financial Based on t-SNE" — uses t-SNE to map Chinese internet financial platforms into 2D; US stock + FX markets cluster distinctly from crypto + Chinese stocks, consistent with their different risk profiles.

```python
# Financial regime visualization
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE

# Rolling correlation features from return series
window = 60  # 60-day rolling window
corr_features = compute_rolling_correlations(returns_df, window=window)  # shape: (T, N*(N-1)/2)

X_scaled = RobustScaler().fit_transform(corr_features)  # RobustScaler for outlier robustness in finance
tsne = TSNE(perplexity=30, init='pca', max_iter=1000, random_state=42)
regime_embedding = tsne.fit_transform(X_scaled)
# Color by VIX level, year, or crisis period for interpretation
```

---

### 8.8 Single-Cell RNA-seq (scRNA-seq) — The Canonical Application

**Domain:** Genomics, cell biology
**Problem solved:** scRNA-seq measures gene expression of 10,000–30,000 genes across tens of thousands of individual cells. The goal: identify cell types, developmental trajectories, and diseased vs. healthy states.
**Why t-SNE:** This is the domain where t-SNE had its biggest scientific impact. The Seurat (R) and Scanpy (Python) single-cell analysis toolkits made t-SNE (and later UMAP) their default visualization method. Thousands of published papers use t-SNE plots of scRNA-seq data.
**Standard pipeline:**
1. Filter low-quality cells and genes
2. Normalize (library size normalization + log transform)
3. Select highly variable genes (~2,000 genes)
4. PCA to 50 components
5. t-SNE or UMAP on PC coordinates
6. Cluster (Leiden algorithm)
7. Annotate clusters with marker genes

```python
# Standard scRNA-seq t-SNE pipeline (Scanpy)
import scanpy as sc

adata = sc.read_h5ad('cells.h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.tsne(adata, use_rep='X_pca', perplexity=30)
sc.pl.tsne(adata, color='cell_type')
```

---

## SECTION 9: Critical Pitfalls Reference

### 9.1 The Four Things t-SNE Cannot Tell You

**Pitfall 1: Cluster sizes are meaningless**
- Explanation: t-SNE uses adaptive bandwidths (σ_i varies per point). Dense regions get small bandwidths, sparse regions get large bandwidths. This normalizes local density differences. In the embedding, a cluster of 10 tightly-packed similar cells may appear the same size as a cluster of 1000 loosely-organized cells.
- Example: In a MNIST t-SNE, the "1" digit cluster often appears small (because 1s are similar to each other) while the "3" digit cluster appears large (3s have more internal variation). This does NOT mean there are fewer 1s than 3s.
- Rule: Color by class count or annotate with numbers. Never read population size from cluster area.

**Pitfall 2: Inter-cluster distances are meaningless**
- Explanation: t-SNE optimizes LOCAL structure. The positions of clusters relative to each other depend heavily on the random initialization, the order of optimization steps, and the perplexity. Two clusters that are far apart in the embedding are NOT necessarily more different than two clusters that are close together.
- Example: At perplexity 30, the "cats" and "dogs" clusters may be far from the "cars" cluster. At perplexity 50, they might be rearranged completely.
- Rule: Only interpret which points are IN the same cluster. Never interpret distance between cluster centroids.

**Pitfall 3: Topology changes with perplexity**
- Explanation: The number of clusters, their sub-structure, and their connectivity all change with perplexity. At very low perplexity (5), one real cluster may appear as 10 micro-clusters. At very high perplexity (200), 5 real clusters may appear as 1 blob.
- Rule: Run at perplexities {5, 15, 30, 50, 100} and show all results. Only report structures that are consistent.

**Pitfall 4: Noise can form clusters**
- Explanation: t-SNE will form clusters even from pure random noise (Wattenberg 2016 demonstrates this). A t-SNE plot of random Gaussian data shows multiple "clusters" — these are artifacts.
- Rule: Always compare t-SNE to PCA on the same data. If PCA shows no structure but t-SNE shows clusters, be highly skeptical.

### 9.2 Common Code Pitfalls

```python
# WRONG: Old default parameters (sklearn < 1.2)
tsne = TSNE(n_iter=1000, init='random', learning_rate=200)  # Bad defaults

# CORRECT: Modern sklearn 1.5.x
tsne = TSNE(max_iter=1000, init='pca', learning_rate='auto')

# WRONG: Using n_iter in sklearn 1.5+
tsne = TSNE(n_iter=1000)  # DeprecationWarning; removed in 1.7

# CORRECT:
tsne = TSNE(max_iter=1000)

# WRONG: Trying to use init='pca' with metric='precomputed'
tsne = TSNE(metric='precomputed', init='pca')  # Will raise ValueError

# CORRECT:
tsne = TSNE(metric='precomputed', init='random')

# WRONG: Using t-SNE embedding as features for downstream ML
# (produces test-time leakage and can't embed new data)
model.fit(X_tsne_train, y_train)
# t-SNE is NOT parametric; you can't compute X_tsne_test without X_train

# CORRECT: Use PCA or UMAP for downstream ML
from sklearn.decomposition import PCA
X_pca = PCA(n_components=50).fit_transform(X)
```

---

## SECTION 10: Variants Summary Table

| Variant | Year | Key Innovation | Complexity | When to Use |
|---------|------|---------------|------------|-------------|
| SNE | 2002 | Stochastic neighborhoods with Gaussians | O(N²) | Historical only |
| t-SNE | 2008 | t-distribution in embedding + symmetrized P | O(N²) | N < 5,000 with method='exact' |
| Barnes-Hut t-SNE | 2014 | VP-tree for k-NN + Barnes-Hut gradient | O(N log N) | Default for N up to ~100k |
| FIt-SNE | 2019 | FFT-accelerated gradient via interpolation | O(N) | N > 100k, single-cell genomics |
| Parametric t-SNE | 2009 | Neural network encoder for f(x)→embedding | O(N) train | Need to embed new data, stable mappings |
| HSNE | 2016 | Hierarchical landmarks for interactive exploration | Sublinear | N > 10M, interactive exploration |
| openTSNE | 2019+ | Modular Python library with incremental embedding | O(N) | Production use, large data, incremental |

---

## SECTION 11: Selected URLs for Chapter Author Reference

- Original t-SNE paper: https://jmlr.org/papers/v9/vandermaaten08a/vandermaaten08a.pdf
- van der Maaten's t-SNE page: https://lvdmaaten.github.io/tsne/
- Wattenberg 2016 (Distill): https://distill.pub/2016/misread-tsne/
- Kobak & Berens 2019: https://www.nature.com/articles/s41467-019-13056-x
- openTSNE docs: https://opentsne.readthedocs.io/en/stable/
- openTSNE algorithm explanation: https://opentsne.readthedocs.io/en/stable/tsne_algorithm.html
- openTSNE parameter guide: https://opentsne.readthedocs.io/en/latest/parameters.html
- sklearn 1.5.2 TSNE docs: https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html
- sklearn trustworthiness: https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.trustworthiness.html
- FIt-SNE Nature Methods paper: https://www.nature.com/articles/s41592-018-0308-4
- FIt-SNE GitHub: https://github.com/KlugerLab/FIt-SNE
- Barnes-Hut arXiv: https://arxiv.org/abs/1301.3342
- Parametric t-SNE: https://proceedings.mlr.press/v5/maaten09a.html
- HSNE preprint: https://nicola17.github.io/publications/2016_hsne/preprint.pdf
- Belkina opt-SNE: https://www.nature.com/articles/s41467-019-13055-y

---

## SECTION 12: Code Snippets — Verified for sklearn 1.5.x

### 12.1 Minimal Working Example
```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target  # (1797, 64)

# Fit t-SNE (sklearn 1.5.x syntax)
tsne = TSNE(
    n_components=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate='auto',     # auto = max(N/48, 50); best practice since v1.2
    max_iter=1000,            # was n_iter before v1.5
    init='pca',               # best practice since v1.2
    method='barnes_hut',
    angle=0.5,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
X_emb = tsne.fit_transform(X)
print(f"Embedding shape: {X_emb.shape}")        # (1797, 2)
print(f"KL divergence: {tsne.kl_divergence_:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter)
ax.set_title("t-SNE of Digits Dataset (perplexity=30)")
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
plt.tight_layout()
plt.savefig("digits_tsne.png", dpi=150)
```

### 12.2 PCA Preprocessing Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# For high-dimensional data (D > 50)
X_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50, random_state=42)),
]).fit_transform(X)

# Note: TSNE is NOT pipeline-compatible (no inverse_transform)
# So we apply it separately after the pipeline
tsne = TSNE(
    perplexity=30,
    learning_rate='auto',
    max_iter=1000,
    init='pca',
    n_jobs=-1,
    random_state=42,
)
X_emb = tsne.fit_transform(X_pca)
```

### 12.3 openTSNE for Large Datasets
```python
from openTSNE import TSNE as openTSNE

tsne = openTSNE(
    n_components=2,
    perplexity=50,
    n_iter=1000,
    metric='euclidean',
    n_jobs=-1,
    random_state=42,
    verbose=True,
)
X_emb = tsne.fit_transform(X)  # 20x+ faster than sklearn for N > 50k
```

### 12.4 Trustworthiness Evaluation
```python
from sklearn.manifold import trustworthiness

for k in [5, 10, 20]:
    score = trustworthiness(X, X_emb, n_neighbors=k)
    print(f"Trustworthiness @ k={k:2d}: {score:.4f}")
```

### 12.5 Perplexity Sensitivity Analysis
```python
from sklearn.manifold import TSNE, trustworthiness
import pandas as pd

results = []
for perp in [5, 10, 20, 30, 50, 100]:
    tsne = TSNE(perplexity=perp, init='pca', max_iter=1000, random_state=42, n_jobs=-1)
    X_emb = tsne.fit_transform(X)
    tw = trustworthiness(X, X_emb, n_neighbors=10)
    kl = tsne.kl_divergence_
    results.append({'perplexity': perp, 'trustworthiness': tw, 'kl_divergence': kl})

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### 12.6 Comparison Across Packages
```python
# Same fit_transform call across packages for direct comparison

# sklearn
from sklearn.manifold import TSNE as sklearn_TSNE
emb_sklearn = sklearn_TSNE(n_components=2, perplexity=30, random_state=42, 
                            init='pca', max_iter=1000, n_jobs=-1).fit_transform(X)

# openTSNE (sklearn-compatible API)
from openTSNE import TSNE as openTSNE_TSNE
emb_opentsne = openTSNE_TSNE(n_components=2, perplexity=30, random_state=42,
                               n_jobs=-1).fit_transform(X)
```

---

## SECTION 13: Key Numbers for the Chapter

- t-SNE paper year: 2008; JMLR volume 9
- Barnes-Hut paper year: 2014; JMLR volume 15
- FIt-SNE paper year: 2019; Nature Methods
- openTSNE latest: v1.0.4 (October 2025)
- openTSNE speedup: 20x+ over sklearn for 200k points (>90 min → <4 min)
- FIt-SNE complexity: O(N) per iteration gradient (vs O(N log N) BH)
- Perplexity range recommendation: 5–50 original paper; N/100 for large N (Kobak & Berens)
- sklearn default learning_rate changed: v1.2 (from 200 → 'auto')
- sklearn default init changed: v1.2 (from 'random' → 'pca')
- sklearn n_iter deprecated: v1.5 (replaced by max_iter)
- sklearn n_iter removed: v1.7 (planned)
- Typical KL divergence range for good embedding: 0.3–1.5
- Barnes-Hut angle optimal range: 0.2–0.8 (default 0.5)
- Early exaggeration phase: first 250 iterations (default)
- Early exaggeration factor default: 12
- t-SNE cannot embed new data (non-parametric): TRUE (use openTSNE partial or parametric t-SNE)
- t-SNE degrees of freedom of Student-t: 1 (this is the Cauchy distribution)

---

*End of Research Dossier R-tsne.md*
*Chapter author: use Sections 1–13 in full. Expand code with real dataset examples. Follow course bible structure (§0 packages first, then §1 history, etc.).*
