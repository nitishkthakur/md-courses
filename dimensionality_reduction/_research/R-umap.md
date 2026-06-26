# Research Dossier: UMAP and Variants
**For: Dimensionality Reduction Masterclass — Chapter on UMAP**
**Compiled: 2026-06-26**
**Status: COMPLETE — ready for writing agent**

---

## SECTION 1 — Paper Citations and Bibliographic Index

### Paper 1 (Primary): McInnes, Healy, Melville 2018 — The UMAP Paper

- **Full citation:** Leland McInnes, John Healy, James Melville. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv preprint arXiv:1802.03426 (2018, revised 2020).
- **arXiv URL:** https://arxiv.org/abs/1802.03426
- **DOI:** https://doi.org/10.48550/arXiv.1802.03426
- **Venue:** arXiv (stat.ML), also published as JOSS-adjacent reference; widely cited as conference-quality theory work
- **Year:** First submitted February 9, 2018; last revised September 18, 2020
- **What you learn from it (2-3 sentences):** The paper develops UMAP from first principles using Riemannian geometry and algebraic topology, specifically arguing that if data is assumed to be uniformly distributed on a Riemannian manifold, one can construct a fuzzy simplicial set representation of that manifold from finite samples. Cross-entropy between the high-dimensional and low-dimensional fuzzy topological representations is then minimized to produce an embedding. The result is an algorithm that is faster than t-SNE, scales better, and by default preserves more global structure due to spectral initialization of the optimization.
- **Difficulty level:** High — requires familiarity with Riemannian geometry, simplicial sets (category theory via Spivak), and stochastic gradient descent
- **Key theoretical innovations:**
  1. Modeling the manifold through Riemannian geometry instead of raw pairwise distances
  2. Constructing the high-dimensional topology as a fuzzy simplicial set (each data point gets its own locally adaptive metric using the distance to its kth nearest neighbor as a normalization constant)
  3. Using cross-entropy as the loss function between high- and low-dimensional fuzzy sets (rather than KL divergence as in t-SNE)
  4. Optimization via negative sampling (contrastive learning) instead of Barnes-Hut approximation

---

### Paper 2: Narayan, Berger, Cho 2021 — densMAP

- **Full citation:** Ashwin Narayan, Bonnie Berger, Hyunghoon Cho. "Assessing single-cell transcriptomic variability through density-preserving data visualization." *Nature Biotechnology* 39, 765–774 (2021).
- **DOI:** https://doi.org/10.1038/s41587-020-00801-7
- **Venue:** Nature Biotechnology (peer-reviewed, high-impact)
- **Year:** 2021 (preprint 2020)
- **What you learn from it (2-3 sentences):** Standard UMAP (and t-SNE) discard local density information from the original space — densely-populated subsets of cells get the same visual footprint as sparse ones, leading to misleading visualizations where all clusters look equally diverse. densMAP regularizes the UMAP cost function with an additional term that correlates the local radius of each point in the embedding to its local radius in the original space, explicitly preserving density. This is now natively supported in umap-learn via `densmap=True` with roughly 20% additional runtime overhead.
- **Difficulty level:** Medium — accessible to practitioners with UMAP knowledge
- **Key result:** densMAP outperforms standard UMAP at identifying transcriptionally variable vs. homogeneous cell populations in scRNA-seq datasets

---

### Paper 3: Damrich, Böhm, Hamprecht, Kobak 2022/2023 — Contrastive Learning Unifies t-SNE and UMAP

- **Full citation:** Sebastian Damrich, Jan Niklas Böhm, Fred A. Hamprecht, Dmitry Kobak. "From t-SNE to UMAP with Contrastive Learning." arXiv:2206.01816. Presented at ICLR 2023.
- **arXiv URL:** https://arxiv.org/abs/2206.01816
- **Year:** Submitted June 3, 2022; revised February 28, 2023
- **Venue:** ICLR 2023
- **Author affiliations:** Damrich and Hamprecht (IWR, Heidelberg University); Böhm and Kobak (University of Tübingen)
- **What you learn from it (2-3 sentences):** This paper provides the theoretical unification of t-SNE and UMAP by revealing they are both contrastive learning methods: t-SNE can be optimized via noise-contrastive estimation (NCE) while UMAP uses negative sampling — both are instances of the same family of objectives. The key difference is that negative sampling introduces a specific distortion (a normalization constant that depends on the number of negative samples), which mathematically explains why UMAP produces tighter, more compact clusters than t-SNE. The authors propose a generalization that enables smooth interpolation between t-SNE-like and UMAP-like embeddings, giving practitioners a knob to control the local-vs-global tradeoff.
- **Difficulty level:** High — requires understanding of noise-contrastive estimation and information theory

---

### Paper 4: Coenen & Pearce 2019 — "Understanding UMAP" (Google PAIR)

- **Full citation:** Andy Coenen, Adam Pearce. "Understanding UMAP." Google PAIR (People + AI Research), 2019.
- **URL:** https://pair-code.github.io/understanding-umap/
- **Also available as PDF:** https://www.ccs.neu.edu/home/vip/teach/DMcourse/3_dim_reduction/UMAP/Understanding%20UMAP.pdf
- **Venue:** Google PAIR interactive explainer / white paper (not peer-reviewed journal)
- **What you learn from it (2-3 sentences):** This is the most accessible deep-dive into UMAP's theory and behavior — it uses interactive visualizations to show how n_neighbors and min_dist affect embeddings on canonical toy datasets. The authors explicitly compare UMAP and t-SNE, showing UMAP's advantage in speed and global structure preservation while being honest about what UMAP cannot guarantee. Highly recommended as a practitioner reference for the "what does this hyperparameter actually do" question.
- **Difficulty level:** Low to medium — excellent for building intuition

---

### Paper 5: Kobak & Linderman 2021 — Initialization Paper

- **Full citation:** Dmitry Kobak, George C. Linderman. "Initialization is critical for preserving global data structure in both t-SNE and UMAP." *Nature Biotechnology* 39, 156–157 (2021).
- **DOI:** https://doi.org/10.1038/s41587-020-00809-z
- **Venue:** Nature Biotechnology (correspondence / letter)
- **Year:** 2021
- **What you learn from it (2-3 sentences):** This correspondence demonstrates that UMAP with random initialization preserves global structure just as poorly as t-SNE with random initialization — the widely-cited advantage of UMAP for global structure preservation is largely due to its default spectral (Laplacian eigenmaps) initialization, not the UMAP objective itself. This means t-SNE with PCA/spectral initialization performs comparably to UMAP on global structure. Critical reading for anyone comparing the two algorithms.
- **Difficulty level:** Low — a short, readable letter

---

### Paper 6: Becht et al. 2019 — UMAP for scRNA-seq (Nature Biotechnology application paper)

- **Full citation:** Etienne Becht, Leland McInnes, John Healy, Charles-Antoine Dutertre, Immanuel W. H. Kwok, Lai Guan Ng, Florent Ginhoux, Evan W. Newell. "Dimensionality reduction for visualizing single-cell data using UMAP." *Nature Biotechnology* 37, 38–44 (2019).
- **DOI:** https://doi.org/10.1038/nbt.4314
- **Year:** 2019
- **What you learn from it:** The paper that established UMAP as the standard for single-cell data visualization, directly comparing it to t-SNE, PCA, and other methods on flow cytometry and scRNA-seq datasets. Shows superior speed and comparably high local structure preservation.
- **Difficulty level:** Low — practitioner-focused

---

## SECTION 2 — Python Package Ecosystem (Comprehensive)

### Package 1: umap-learn (THE canonical package)

- **PyPI name:** `umap-learn`
- **Install:** `pip install umap-learn`
- **With optional plotting:** `pip install umap-learn[plot]`
- **With parametric UMAP:** `pip install umap-learn tensorflow`
- **Current version (mid-2026):** 0.5.12 (released April 8, 2026)
- **Python requirement:** Python >= 3.9
- **License:** BSD 3-Clause
- **Maintainer:** Leland McInnes (primary), TutteInstitute
- **Maintenance status:** Active, regularly updated
- **Key import:** `import umap; reducer = umap.UMAP()`
- **Core dependencies:** numpy, scipy, scikit-learn, numba, tqdm, pynndescent
- **Optional dependencies:** TensorFlow >= 2.0.0 (for ParametricUMAP), matplotlib/datashader/holoviews (for plotting)
- **What it does better than anything else:** It IS the reference implementation. Supports all variants: supervised UMAP, semi-supervised UMAP, densMAP, ParametricUMAP, AlignedUMAP, metric UMAP with custom distance functions, inverse_transform, and sparse input
- **Sklearn compatibility:** Full sklearn transformer API (fit, transform, fit_transform, inverse_transform)
- **Key submodules:**
  - `umap.UMAP` — main class
  - `umap.parametric_umap.ParametricUMAP` — neural network parametric extension
  - `umap.aligned_umap.AlignedUMAP` — aligning multiple embeddings
  - `umap.plot` — visualization utilities (requires optional deps)

```python
import umap
import numpy as np

# Basic usage
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X)

# Supervised
embedding = umap.UMAP().fit_transform(X, y=labels)

# densMAP
reducer = umap.UMAP(densmap=True, dens_lambda=2.0)
embedding = reducer.fit_transform(X)

# Transform new points
reducer.fit(X_train)
X_test_embedded = reducer.transform(X_test)
```

---

### Package 2: cuML (GPU-accelerated UMAP via RAPIDS)

- **PyPI name:** `cuml` (part of RAPIDS AI)
- **Install:** `pip install cuml-cu12` (or via conda: `conda install -c rapidsai cuml`)
- **Current version (mid-2026):** cuML 26.06.00 (June 2026 release — RAPIDS follows YY.MM versioning)
- **License:** Apache 2.0
- **Maintainer:** NVIDIA / RAPIDS AI team
- **Maintenance status:** Very active, monthly releases
- **Key import:** `from cuml.manifold import UMAP as cuUMAP`
- **What it does better:** GPU-accelerated — 60x speedup over CPU umap-learn on NVIDIA H100 GPUs; for datasets of 20M points and 384 dimensions, reduces processing time from 10 hours to 2 minutes (311x speedup). Scales to datasets that don't fit on GPU via batching (introduced in cuML 24.10). Supports batched approximate nearest neighbor (ANN) algorithm.
- **Zero-code-change acceleration:** Replace `import umap; umap.UMAP()` with `from cuml.manifold import UMAP as cuUMAP; cuUMAP()` and keep nearly identical parameter names
- **Key parameter differences from umap-learn:**
  - `hash_input` (bool, default=False): enable consistent transform results
  - `force_serial_epochs` (bool/None): force sequential optimization kernel
  - `build_algo` ('auto'|'brute_force_knn'|'nn_descent'): KNN construction algorithm
  - `device_ids` (list/str/None): GPU device IDs for multi-GPU fitting
  - Does NOT yet support: pre-computed distances (pairwise), manual array initialization
  - Supported metrics: l1, cityblock, manhattan, euclidean, l2, sqeuclidean, canberra, minkowski, chebyshev, cosine, correlation, hellinger, hamming, jaccard
- **Benchmark context:** cuML UMAP on H100 achieves 60x speedup vs CPU; HDBSCAN 175x

```python
from cuml.manifold import UMAP as cuUMAP
import cupy as cp  # or numpy — output_type controls this

reducer = cuUMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X)  # X can be numpy or cupy array
```

---

### Package 3: rapids-singlecell (scRNA-seq specific GPU pipeline)

- **PyPI name:** `rapids-singlecell`
- **Install:** `pip install rapids-singlecell`
- **License:** Apache 2.0
- **Key import:** `import rapids_singlecell as rsc`
- **What it does better:** Drop-in GPU replacement for scanpy's single-cell workflows including UMAP. Supports `init_pos` parameter as ndarray, 'paga', or obsm[key]. Integrates with AnnData objects directly.
- **Use case:** When running scRNA-seq pipelines at scale on GPUs

```python
import rapids_singlecell as rsc
rsc.tl.umap(adata, n_neighbors=15)
```

---

### Package 4: ParametricUMAP (via umap-learn)

- **No separate package** — included in umap-learn >= 0.5
- **Install:** `pip install umap-learn tensorflow`
- **Key import:** `from umap.parametric_umap import ParametricUMAP`
- **What it does better:** Learns a neural network that maps any new point to its embedding without re-running the full UMAP optimization. Supports: inverse transforms (decoder network), autoencoder joint training, custom CNN/RNN encoder architectures, early stopping via Keras callbacks
- **Backend:** TensorFlow + Keras ONLY (no native PyTorch — a PyTorch PR exists at lmcinnes/umap#1103 but is community-maintained)
- **For pure PyTorch:** Use `github.com/fcarli/parametric_umap` (unofficial)

```python
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf

# Default (3-layer, 100-neuron FC network)
embedder = ParametricUMAP(n_components=2)
embedding = embedder.fit_transform(X)

# Custom encoder
encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])
embedder = ParametricUMAP(encoder=encoder, n_components=2)
embedding = embedder.fit_transform(X)

# With reconstruction loss (autoencoder mode)
embedder = ParametricUMAP(
    encoder=encoder,
    parametric_reconstruction=True,
    autoencoder_loss=True
)
```

---

### Package 5: uwot (R package — for reference)

- Not Python, but the primary R implementation
- **CRAN:** `uwot`
- Provides `umap()` and `umap_transform()` functions
- Mentioned here because many scRNA-seq workflows use R (Seurat uses uwot under the hood)

---

### Package Summary Table

| Package | Best For | GPU | Version (mid-2026) | License |
|---------|----------|-----|-------------------|---------|
| `umap-learn` | Everything — reference implementation | No | 0.5.12 | BSD-3 |
| `cuml` (RAPIDS) | Large-scale GPU acceleration | Yes | 26.06 | Apache 2.0 |
| `rapids-singlecell` | scRNA-seq GPU pipelines | Yes | active | Apache 2.0 |
| `ParametricUMAP` (via umap-learn) | New-point inference, inverse transforms | No (TF) | included | BSD-3 |
| `pynndescent` | Fast ANN — umap-learn's backend | No | 0.5.x | BSD-2 |

---

## SECTION 3 — Full API Parameter Reference (umap-learn 0.5.12)

All parameters of `umap.UMAP()`. Verified against umap-learn 0.5.8 docs (stable as of 0.5.12 — no breaking API changes in this parameter set since 0.5.5).

### Core Structure Parameters

| Parameter | Type | Default | Controls |
|-----------|------|---------|----------|
| `n_neighbors` | float | 15 | Size of local neighborhood for manifold approximation. Fundamental local-vs-global tradeoff knob. |
| `n_components` | int | 2 | Dimensionality of the embedding space. |
| `metric` | str or callable | 'euclidean' | Distance metric in input space. |
| `metric_kwds` | dict | None | Keyword arguments for the metric function. |
| `output_metric` | str or callable | 'euclidean' | Distance metric in output embedding space. Use 'haversine' for geospatial embeddings onto sphere. |
| `output_metric_kwds` | dict | None | Keyword arguments for output_metric. |

### Optimization Parameters

| Parameter | Type | Default | Controls |
|-----------|------|---------|----------|
| `n_epochs` | int | None | Number of training epochs. Auto-selected: ~500 for small datasets (<10k points), ~200 for larger. |
| `learning_rate` | float | 1.0 | Initial learning rate for SGD embedding optimization. |
| `min_dist` | float | 0.1 | Minimum distance between embedded points. Controls clustering tightness in low-dim space. |
| `spread` | float | 1.0 | Scale of embedded points. Works with min_dist via a, b curve fitting. |
| `a` | float | None | Embedding shape parameter. Auto-set from min_dist/spread if None. |
| `b` | float | None | Embedding shape parameter. Auto-set from min_dist/spread if None. |
| `repulsion_strength` | float | 1.0 | Weight on negative samples in loss. Higher = more repulsion between non-neighbors. |
| `negative_sample_rate` | int | 5 | Negative samples per positive sample. Higher = slower but better structure. |
| `set_op_mix_ratio` | float | 1.0 | Interpolation between fuzzy union (1.0) and intersection (0.0). |
| `local_connectivity` | int | 1 | Number of locally assumed-connected nearest neighbors. |

### Initialization Parameters

| Parameter | Type | Default | Controls |
|-----------|------|---------|----------|
| `init` | str or array | 'spectral' | Initialization of embedding: 'spectral' (Laplacian eigenmaps), 'random', 'pca', 'tswspectral', or a numpy array of initial positions. |
| `random_state` | int, RandomState, None | None | Seed for reproducibility. |

**Critical note on `init`:** 'spectral' is the default and is strongly recommended. Spectral initialization uses Laplacian eigenmaps to provide a globally-coherent starting point that SGD then refines. Random initialization produces embeddings where inter-cluster distances are meaningless. The paper by Kobak & Linderman (2021, Nature Biotechnology) proves that UMAP's global structure advantage over t-SNE largely disappears if you use random initialization for UMAP.

### Memory and Performance Parameters

| Parameter | Type | Default | Controls |
|-----------|------|---------|----------|
| `low_memory` | bool | True | Memory-efficient mode for nearest neighbor computation. Uses pynndescent with approximate NN. Set False for exact NN (much slower). |
| `n_jobs` | int | -1 | Parallel jobs for computation. -1 = all CPU cores. |
| `angular_rp_forest` | bool | False | Use angular random projection forest for NN initialization. Better for cosine/angular metrics. |
| `force_approximation_algorithm` | bool | False | Force approximation even for small datasets. |
| `transform_queue_size` | float | 4.0 | Aggressiveness of NN search during transform(). Higher = slower but more accurate. |
| `transform_seed` | int | 42 | Random seed for stochastic elements in transform(). |
| `transform_mode` | str | 'embedding' | Mode for transform operation. |
| `unique` | bool | False | Remove duplicate rows before embedding. |

### Supervised/Semi-supervised Parameters

| Parameter | Type | Default | Controls |
|-----------|------|---------|----------|
| `target_n_neighbors` | int | -1 | Neighbors for target simplicial set. -1 means use n_neighbors. |
| `target_metric` | str or callable | 'categorical' | Distance metric for label space. 'categorical' treats labels as discrete categories. Use 'euclidean' for continuous targets. |
| `target_metric_kwds` | dict | None | Kwargs for target_metric. |
| `target_weight` | float | 0.5 | Weight between data topology (0.0) and label topology (1.0). 0.5 = equal balance. |

### densMAP Parameters

| Parameter | Type | Default | Controls |
|-----------|------|---------|----------|
| `densmap` | bool | False | Enable density-preserving objective (densMAP). |
| `dens_lambda` | float | 2.0 | Weight of density correlation regularization term. |
| `dens_frac` | float | 0.3 | Fraction of epochs using density-augmented objective. |
| `dens_var_shift` | float | 0.1 | Variance shift constant for numerical stability in density calculations. |
| `output_dens` | bool | False | If True, return (embedding, local_radii) instead of just embedding. |

### Advanced/Expert Parameters

| Parameter | Type | Default | Controls |
|-----------|------|---------|----------|
| `disconnection_distance` | float | np.inf | Maximum graph distance — points farther than this are considered disconnected components. |
| `precomputed_knn` | tuple | (None,None,None) | Pre-computed (knn_indices, knn_dists, knn_search_index). Skip ANN computation if you have it from pynndescent. |
| `verbose` | bool | False | Logging verbosity. |
| `tqdm_kwds` | dict | None | Keyword arguments for tqdm progress bar. |

### Version-specific notes (API changes)

- **0.5.x series (stable as of 0.5.12):** `low_memory` defaulted to False in older versions; changed to True around 0.5.3. If you see old code without `low_memory`, be aware.
- **Pre-0.5 (legacy):** `angular_rp_forest` not available. `n_jobs` not available.
- **0.5.0 introduced:** ParametricUMAP, AlignedUMAP, densMAP native support.
- **No sklearn integration issue:** umap-learn is NOT part of sklearn but implements the sklearn transformer protocol (fit, transform, fit_transform). Import is `import umap`, not `from sklearn.manifold import UMAP`.

---

## SECTION 4 — Detailed Hyperparameter Tuning Guide

### `n_neighbors` — The Most Important Parameter

**What it controls:** The size of the local neighborhood used to approximate the manifold structure. Determines the local-vs-global tradeoff.

**Recommended range:** 2 to 200 (practical: 5 to 50)

**Default:** 15

**Heuristics:**
- Start at 15 and adjust based on need
- **Smaller datasets** (< 1000 points): consider n_neighbors in 5–15
- **Single-cell RNA-seq:** 10–30 is standard; common choice is 15 for PBMC datasets
- **NLP embeddings:** 25–50 works well for dense semantic spaces
- **Drug discovery (molecular fingerprints):** 10–20 for compound clustering

**Too low (n_neighbors = 2–5):**
- Embedding fragments into disconnected components or "islands"
- Very tight, local clusters — the algorithm can see only 2 points as neighbors, so it focuses on hyper-local structure
- May appear as many small, disconnected blobs
- Useful for: finding micro-clusters, identifying outliers

**Too high (n_neighbors = 100–200):**
- All local detail is lost; embedding reflects only broad topological structure
- Clusters merge; you get a blob with gradients rather than distinct groups
- Looks similar to a PCA-like smooth embedding
- Useful for: smooth manifold visualization, global topology

**Optuna search space:**
```python
def objective(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
    reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
    embedding = reducer.fit_transform(X)
    # Evaluate with trustworthiness or downstream task metric
    score = sklearn.manifold.trustworthiness(X, embedding, n_neighbors=n_neighbors)
    return score
```

---

### `min_dist` — Cluster Packing Density

**What it controls:** The minimum distance between points in the low-dimensional embedding. Directly controls how tightly points are packed together.

**Recommended range:** 0.0 to 0.99 (practical: 0.0 to 0.5)

**Default:** 0.1

**Heuristics:**
- For **clustering tasks:** use 0.0 or very small values (0.0–0.05) — you want points to pack tightly so clustering algorithms work well on the embedding
- For **visualization/exploration:** use 0.1–0.3 — balanced view
- For **presentation/aesthetics:** use 0.5–0.8 — more visually spread out

**Too low (min_dist ~0.0):**
- Very tight, compact clusters — individual points within clusters overlap
- Excellent for downstream clustering (HDBSCAN, k-means on embedding)
- Can look "blobby" to human eyes

**Too high (min_dist ~0.9):**
- Points spread very evenly — loses cluster structure
- All points are approximately equidistant; embedding looks like a filled circle
- Loses the topological structure that makes UMAP useful

**Interaction with n_neighbors:** These two are the primary tuning knobs. Fix n_neighbors first to get the right topology, then adjust min_dist for visual clarity.

**Optuna search space:**
```python
min_dist = trial.suggest_float("min_dist", 0.0, 0.5)
```

---

### `n_components` — Embedding Dimensionality

**What it controls:** The number of dimensions in the output embedding.

**Default:** 2

**When to use more than 2:**
- For **ML pipelines** (not just visualization): 5–50 components gives a much better manifold representation
- The UMAP paper notes n_components scales well — UMAP doesn't degrade as badly as t-SNE at higher dimensions
- For **clustering:** 10–20 dimensions often gives the best downstream clustering results
- For **classification feature engineering:** 10–30 components

**Rule of thumb:** Use 2 for visualization; use 10–50 for downstream ML tasks. For scRNA-seq, a common workflow is: PCA(50) → UMAP(10) → Leiden clustering → UMAP(2) for visualization.

---

### `metric` — Input Space Distance

**What it controls:** How distances are measured in the original high-dimensional space.

**Default:** 'euclidean'

**Available metrics (umap-learn, via pynndescent):**
- **Minkowski family:** 'euclidean', 'manhattan', 'chebyshev', 'minkowski'
- **Angular:** 'cosine', 'correlation'
- **Binary/categorical:** 'hamming', 'jaccard', 'dice', 'matching', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'yule'
- **Distributional:** 'hellinger', 'wasserstein' (approximate)
- **Custom:** Any numba-compiled function

**Domain-specific guidance:**
- **Text / NLP embeddings (BERT etc.):** use 'cosine' — embeddings are normalized or near-unit-sphere
- **Molecular fingerprints (drug discovery):** use 'jaccard' or 'tanimoto' (Jaccard on binary fingerprints = Tanimoto coefficient)
- **Gene expression (count data):** 'euclidean' on normalized+log-transformed data works; 'cosine' if using raw normalized counts
- **Images:** 'euclidean' is standard; 'cosine' if using CNN features

**Custom metric example:**
```python
from numba import njit

@njit
def tanimoto_distance(a, b):
    # Returns 1 - Tanimoto similarity for binary fingerprints
    intersection = 0.0
    union = 0.0
    for i in range(len(a)):
        if a[i] or b[i]:
            union += 1
            if a[i] and b[i]:
                intersection += 1
    return 1.0 - (intersection / union) if union > 0 else 0.0

reducer = umap.UMAP(metric=tanimoto_distance)
```

---

### `n_epochs` — Optimization Epochs

**Default:** None (auto-selected: ~500 for < 10k points, ~200 for larger)

**When to increase:**
- If embeddings look noisy or poorly converged
- When using small n_neighbors (more sensitive optimization landscape)

**When to decrease:**
- For very large datasets where 200 epochs is slow — try 100
- For rapid prototyping: 50 epochs gives a rough but fast result

**Optuna search space:**
```python
n_epochs = trial.suggest_int("n_epochs", 100, 500, step=50)
```

---

### `learning_rate`

**Default:** 1.0

**Guidance:** Rarely needs tuning. If embedding looks "frayed" or doesn't converge, try 0.5. For large datasets, occasionally 1.5 helps escape local optima. Leave at default in most cases.

---

### `init` — Initialization Strategy

**Default:** 'spectral'

**Options:**
- `'spectral'` — Laplacian eigenmaps (STRONGLY RECOMMENDED). Preserves global structure in the final embedding because SGD refines rather than destroys the global layout.
- `'random'` — Random positions. UMAP loses its global structure advantage; inter-cluster distances become meaningless.
- `'pca'` — PCA scores as initial positions. Good alternative to spectral when spectral initialization is slow (very large datasets).
- `'tswspectral'` — Truncated SVD spectral embedding. For sparse matrices.
- Array of shape (n_samples, n_components) — custom starting positions.

**Critical practitioner note:** If you are comparing UMAP to t-SNE and claiming UMAP has better global structure, make sure you are using `init='spectral'` (the default). If you see old code using `init='random'`, this is a red flag — the comparison may be unfair.

---

### `target_weight` (Supervised UMAP)

**Default:** 0.5

**Range:** 0.0 to 1.0

**What it controls:**
- 0.0 = purely unsupervised (ignore labels)
- 1.0 = purely supervised (maximize class separation)
- 0.5 = balance data topology with label topology

**Guidance:**
- For **metric learning** (you want maximum class separation for a classifier): use 0.8–1.0
- For **exploratory analysis** (you want to see within-class structure): use 0.3–0.5
- When labels are noisy: stay closer to 0.5 or below

---

### `densmap` + density parameters

**`densmap=True`:** Enables densMAP. Adds ~20% runtime overhead.

**`dens_lambda`:** Default 2.0. Higher values enforce stronger density correlation. Too high can distort topological structure; too low has no density effect. Recommended range: 0.5–5.0.

**`dens_frac`:** Default 0.3. Fraction of total epochs where density term is active. Leave at default unless you have specific reason to change.

---

## SECTION 5 — Supervised, Semi-supervised, and Metric UMAP

### Supervised UMAP

When you have full label information, pass `y` to `fit_transform()`:

```python
import umap

# All points labeled
reducer = umap.UMAP(
    n_neighbors=15,
    target_weight=0.5,      # 0.0 = unsupervised, 1.0 = fully supervised
    target_metric='categorical',  # for discrete labels
    random_state=42
)
embedding = reducer.fit_transform(X, y=labels)
```

**What happens internally:** UMAP constructs two graphs — one from the data (X) and one from the labels (y) — and combines them with the `target_weight` parameter before optimizing. The label graph uses `target_metric` to measure distance between labels, with 'categorical' treating same-label points as distance 0 and different-label points as distance 1.

**Effect:** Classes are pulled together tightly and pushed apart from other classes. Within-class structure is partly preserved depending on `target_weight`.

**For continuous targets (regression):** Set `target_metric='euclidean'` and pass continuous y values. This creates embeddings where similar target values cluster together.

### Semi-supervised UMAP

When some points lack labels, mark unlabeled points with -1 (sklearn convention):

```python
import numpy as np

# Create masked target — -1 means "unknown"
masked_labels = np.copy(labels)
unlabeled_mask = np.random.rand(len(labels)) < 0.5  # 50% unlabeled
masked_labels[unlabeled_mask] = -1

reducer = umap.UMAP()
embedding = reducer.fit_transform(X, y=masked_labels)
```

**Effect:** The labeled points guide the embedding structure while unlabeled points find their position based on the data topology alone. Produces moderate class separation — less sharp boundaries than fully supervised.

### Metric UMAP (Custom Distance for Metric Learning)

Supervised UMAP doubles as a metric learning framework. After fitting on labeled training data, `transform()` applies the learned metric to new points:

```python
# Fit supervised UMAP on train data
reducer = umap.UMAP(target_weight=0.9)
reducer.fit(X_train, y=y_train)

# Transform new test points into the learned metric space
X_test_embedded = reducer.transform(X_test)

# Now run a simple kNN classifier in embedding space
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(reducer.embedding_, y_train)
pred = clf.predict(X_test_embedded)
```

This is particularly powerful for few-shot learning scenarios where you train the metric on seen classes and evaluate on unseen ones.

---

## SECTION 6 — densMAP: Density-Preserving UMAP

### The Problem densMAP Solves

Standard UMAP (and t-SNE) explicitly throw away local density information. The algorithm normalizes each point's distance to its kth neighbor to 1.0 before constructing the fuzzy graph — this makes all neighborhoods locally equivalent regardless of whether a point is in a dense cluster of 10,000 cells or a sparse population of 50 cells. The result: in a single-cell UMAP plot, a cell type with 10,000 highly similar cells occupies the same visual area as a rare cell type with 100 highly variable cells. Researchers misread this as equal transcriptional diversity.

### How densMAP Works

densMAP adds a regularization term to the UMAP cost function:

```
L_densMAP = L_UMAP + lambda * sum_i [rho_low(i) - rho_high(i)]^2
```

Where `rho_high(i)` is the log-radius of point i's neighborhood in the original space and `rho_low(i)` is the log-radius in the embedding. By penalizing the squared difference between these, densMAP forces the embedding to preserve the relative neighborhood sizes — dense regions stay dense, sparse regions stay sparse.

### Usage

```python
import umap

# Basic densMAP
reducer = umap.UMAP(
    densmap=True,
    dens_lambda=2.0,    # weight of density regularization
    dens_frac=0.3,      # fraction of epochs with density term active
    dens_var_shift=0.1, # numerical stability constant
    random_state=42
)
embedding = reducer.fit_transform(X)

# Get both embedding AND local radii
reducer_with_dens = umap.UMAP(densmap=True, output_dens=True)
result = reducer_with_dens.fit_transform(X)
# result is a tuple: (embedding, local_radii_high, local_radii_low)
```

### When to Use densMAP

- **Single-cell genomics:** Always. When you want to make claims about transcriptional variability or diversity, use densMAP to avoid misleading density artifacts.
- **Any domain where density carries meaning:** ecology (species abundance), astronomy (stellar density fields), market data (trading volume concentration)
- **When NOT to use:** If you only care about topology/clustering and don't want density information distorting your layout, use standard UMAP.

### Performance overhead

densMAP adds approximately 20% runtime vs standard UMAP. Still achieves fast runtimes for large datasets — the densvis GitHub repo reports both den-SNE and densMAP take < ~30 minutes for typical single-cell datasets.

---

## SECTION 7 — Parametric UMAP

### What Parametric UMAP Solves

Standard UMAP's `transform()` method for new points uses an approximate lookup in the learned graph — it's not a true function approximation. For production inference (e.g., embedding new user vectors in real time for a recommendation system), this is slow and potentially inconsistent. Parametric UMAP replaces the embedding lookup with a neural network that learns the same UMAP objective:

```
L_parametric_UMAP = L_UMAP(encoder(x_i), encoder(x_j))
```

The encoder maps any new input x to its embedding without re-running graph construction.

### Architecture and API

```python
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf

# Default (3-layer, 100-neuron FC network)
embedder = ParametricUMAP(n_components=2, batch_size=1000)
embedding = embedder.fit_transform(X)

# New points — fast neural network inference
new_embedding = embedder.transform(X_new)

# Save and load
embedder.save("my_umap_model")
from umap.parametric_umap import load_ParametricUMAP
embedder = load_ParametricUMAP("my_umap_model")
```

### Custom Architectures

```python
# Custom encoder for tabular data
encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(n_features,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(2)  # n_components
])
embedder = ParametricUMAP(encoder=encoder)

# CNN encoder for images
encoder = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])
embedder = ParametricUMAP(encoder=encoder)
```

### Autoencoder Mode

```python
# Joint encoder + decoder (reconstruction loss)
embedder = ParametricUMAP(
    encoder=encoder,
    parametric_reconstruction=True,
    autoencoder_loss=True
)
embedding = embedder.fit_transform(X)

# Inverse transform (reconstruct from embedding)
X_reconstructed = embedder.inverse_transform(embedding)
```

### Key Parameters for ParametricUMAP

- `batch_size` (int, default=1000): Edge batches — NOT data batches. Controls memory use during training.
- `n_training_epochs` (int): Equivalent to n_epochs in standard UMAP
- `optimizer`: Defaults to Adam; can pass any tf.keras optimizer
- `parametric_embedding` (bool, default=True): If False, falls back to non-parametric optimization
- `parametric_reconstruction` (bool, default=False): Enable decoder network
- `autoencoder_loss` (bool, default=False): Use reconstruction loss in addition to UMAP loss
- `loss_report_frequency` (int): How often to report loss

### PyTorch Alternative

The official umap-learn only supports TensorFlow/Keras for ParametricUMAP. For PyTorch:
- **Community implementation:** `github.com/fcarli/parametric_umap` (uses PyTorch + FAISS for ANN)
- A PR exists in the main repo (lmcinnes/umap#1103) adding PyTorch support but as of mid-2026 not merged into main

---

## SECTION 8 — UMAP Variants Reference

### Aligned UMAP

Aligns multiple UMAP embeddings of related datasets (e.g., time-series of snapshots) so they are comparable:

```python
from umap.aligned_umap import AlignedUMAP

# relations: list of dicts mapping indices of dataset[i] to dataset[i+1]
relations = [{i: i for i in range(n_points)}] * (n_timepoints - 1)
reducer = AlignedUMAP()
embeddings = reducer.fit_transform(list_of_datasets, relations=relations)
```

Use case: single-cell RNA-seq time courses, video frame embeddings.

### GPU UMAP (cuML variant)

See Section 2, Package 2 above. Key addition: supports `build_algo='nn_descent'` for faster NN graph construction on GPU. Does not support precomputed distance matrices or custom array initialization (as of mid-2026).

### Metric UMAP

Any `metric` parameter choice constitutes "Metric UMAP" — the ability to define a custom distance function is one of UMAP's strongest advantages over t-SNE. The metric must be either a string (recognized by pynndescent) or a numba `@njit`-compiled function.

---

## SECTION 9 — UMAP vs t-SNE: Detailed Comparison

| Property | UMAP | t-SNE |
|----------|------|-------|
| **Time complexity** | O(n log n) | O(n² log n) for exact; O(n log n) with Barnes-Hut |
| **Speed (100k points)** | Minutes | Hours |
| **Speed (5M points)** | ~47 min (CPU) | ~9.4 hours |
| **Speedup factor** | ~12x at 5M points | baseline |
| **Global structure** | Good (with spectral init) | Poor (with random init default) |
| **Local structure** | Excellent | Excellent |
| **Scalability** | To millions+ | Practical limit ~1M with tricks |
| **Initialization** | Spectral (default) | Random (default) |
| **Supports transform()** | Yes (approximate) | No |
| **Supports n_components > 3** | Yes | Unreliable above 3 |
| **Supports supervised** | Yes (target param) | No (standard) |
| **Reproducibility** | With random_state | With random_state |
| **Cluster size meaningful** | No (standard UMAP) | No |
| **Cluster distance meaningful** | Partially (global topology) | No |
| **Underlying objective** | Cross-entropy between fuzzy sets | KL divergence between distributions |
| **Theoretical basis** | Riemannian geometry / topology | Gaussian neighborhoods |
| **Embedding stability** | Higher (CV 0.08) | Lower (CV 0.19) |
| **GPU acceleration** | Yes (cuML, 60x) | Limited |

**Key practical takeaway:** For > 10k points, UMAP is the only practical choice. For < 10k points, t-SNE remains competitive if you want maximum local fidelity. Always use spectral initialization for UMAP if global structure matters.

**Theoretical note (Damrich/Böhm 2022):** Both t-SNE and UMAP are contrastive learning methods. The tighter clusters in UMAP vs t-SNE are not a feature of better structure preservation — they arise from negative sampling's normalization artifact. If you set UMAP's negative_sample_rate very high, you get t-SNE-like spread. If you use NCE-based t-SNE, you get UMAP-like tightness. This is important context for practitioners making algorithm choices.

---

## SECTION 10 — Innovative Real-World Applications

### Application 1: Single-Cell RNA-seq (scRNA-seq) — The Dominant Use Case

- **Domain:** Computational biology / genomics
- **Problem solved:** Visualizing 10,000–1,000,000 cells, each described by 20,000+ gene expression measurements. Researchers need to identify cell types, developmental trajectories, and transcriptional variability.
- **Why UMAP:** Replaced t-SNE as the standard around 2019 (following Becht et al. 2019, Nature Biotechnology). Key advantages: faster (essential for large atlases), better global layout for trajectory inference, supports n_components > 2 for trajectory algorithms (Monocle, scVelo), and `transform()` allows projecting new samples into a reference atlas.
- **Key tools:** Scanpy (Python), Seurat (R, uses uwot), rapids-singlecell (GPU), scvi-tools
- **Specific workflow:**
  ```python
  import scanpy as sc
  sc.pp.neighbors(adata, n_neighbors=15)  # builds kNN graph
  sc.tl.umap(adata, min_dist=0.3)         # runs UMAP
  sc.pl.umap(adata, color=['cell_type', 'leiden'])
  ```
- **Innovation:** densMAP (Narayan 2021) extended this by preserving density — critical when comparing transcriptional variability between cell types (a cell type with 10,000 similar cells should look denser than one with 100 variable cells)
- **Scale:** Human Cell Atlas uses UMAP across >30 million cells

---

### Application 2: NLP Embedding Visualization (BERT/GPT/LLMs)

- **Domain:** Natural language processing / model interpretability
- **Problem solved:** BERT, GPT, and other LLMs produce 768–4096 dimensional embeddings per token or document. Researchers need to understand semantic clusters, model layer behavior, and data distribution.
- **Why UMAP:** `cosine` metric naturally handles unit-sphere normalized embeddings; faster than t-SNE for large document collections; BERTopic (a popular topic modeling library) uses UMAP as a core component.
- **Key workflow (BERTopic):**
  ```python
  from bertopic import BERTopic
  from umap import UMAP
  
  umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
  topic_model = BERTopic(umap_model=umap_model)
  topics, probs = topic_model.fit_transform(documents)
  ```
- **Applications:** Visualizing semantic neighborhoods in embedding spaces; debugging model tokenization; identifying dataset biases; comparing embeddings across model versions
- **Research use:** Papers visualize BERT layer representations using UMAP to understand how linguistic structure emerges across layers (early layers = syntactic, later = semantic)
- **Company references:** BERTopic is used in production at multiple companies for customer feedback clustering, news categorization, and scientific paper organization

---

### Application 3: Drug Discovery — Chemical Space Mapping

- **Domain:** Pharmaceutical research / cheminformatics
- **Problem solved:** A drug discovery pipeline might screen millions of candidate molecules. Each molecule is represented as a molecular fingerprint (a binary or count vector of structural features, e.g., Morgan fingerprints, ECFP4, RDKit fingerprints). Chemists need to understand chemical diversity, cluster structurally similar compounds, and guide synthesis priorities.
- **Why UMAP:** Tanimoto similarity (= Jaccard similarity on binary fingerprints) is the industry standard for molecular similarity — UMAP supports it directly as its `metric`. UMAP respects the non-Euclidean structure of chemical space better than PCA.
- **Key workflow:**
  ```python
  import umap
  from rdkit import Chem
  from rdkit.Chem import rdMolDescriptors
  import numpy as np
  
  # Generate Morgan fingerprints as numpy arrays
  fingerprints = np.array([
      list(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048))
      for mol in molecules
  ])
  
  reducer = umap.UMAP(metric='jaccard', n_neighbors=15, min_dist=0.1)
  embedding = reducer.fit_transform(fingerprints)
  ```
- **Company/paper references:** AstraZeneca, Novartis, and academic groups use UMAP for compound library visualization; published in Journal of Cheminformatics; Elana Simon (blog post at elanapearl.github.io) provides a detailed walkthrough
- **Innovation:** UMAP models fitted on large drug-like compound databases can serve as universal chemical space coordinators — transform any new molecule into the canonical space for comparison

---

### Application 4: Astronomy — Galaxy Morphology Classification

- **Domain:** Astrophysics / observational astronomy
- **Problem solved:** Modern sky surveys (SDSS, JWST) produce images of millions of galaxies. CNN features (hundreds-dimensional) encode morphological information. Astronomers need to cluster galaxy types and find rare morphologies without exhaustive manual labeling.
- **Why UMAP:** Outperforms PCA on non-linear manifold structure of CNN feature spaces; hierarchical features from ConvNeXt/ResNet encoders project cleanly; compatible with HDBSCAN for downstream unsupervised clustering.
- **Recent paper:** "An updated efficient galaxy morphology classification model based on ConvNeXt encoding with UMAP dimensionality reduction" (arXiv 2512.15137, December 2024). Also: kinematic morphology classification using UMAP + HDBSCAN (arXiv 2212.03999).
- **Innovation:** UMAP for kinematic maps — projecting 2D velocity field maps from integral field spectroscopy to 2D for morphological similarity clustering

---

### Application 5: Materials Science — Crystal Structure Space

- **Domain:** Computational materials science / solid-state chemistry
- **Problem solved:** Materials databases (Materials Project: ~150k materials) contain crystal structures encoded as graph-neural-network embeddings or descriptor vectors. Researchers need to navigate chemical space, find materials with similar properties, and spot under-explored regions.
- **Why UMAP:** High-dimensional structure embeddings (from GNNs like MEGNet, CGCNN) are non-linearly organized; UMAP projects them to 2D while preserving neighborhood structure better than PCA.
- **Evidence:** ResearchGate figures show UMAP projections of 116k materials using different embedding models, colored by material property (band gap, formation energy).
- **Application pattern:** Fit UMAP on known materials → identify dense clusters of known functional materials → search database for unexplored regions in sparse areas

---

### Application 6: Fraud Detection and Anomaly Detection

- **Domain:** Financial services / cybersecurity
- **Problem solved:** Transaction datasets have millions of events. Fraud patterns are non-linear (no single feature discriminates; fraud lies in feature combinations). Analysts want to visually inspect suspicious clusters, understand fraud morphology, and identify new fraud types not seen in training.
- **Why UMAP:** Semi-supervised UMAP can incorporate the few known fraud labels (mark frauds as y=1, unknowns as y=-1) to guide the embedding toward fraud-discriminating structure. Then HDBSCAN on the embedding finds emerging fraud clusters.
- **Workflow:**
  ```python
  import umap
  import numpy as np
  
  # y has 1 for known frauds, 0 for known legit, -1 for unlabeled
  reducer = umap.UMAP(target_weight=0.7, metric='euclidean', n_neighbors=20)
  embedding = reducer.fit_transform(X_transactions, y=y_labels)
  
  # Cluster the embedding
  import hdbscan
  clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
  cluster_labels = clusterer.fit_predict(embedding)
  ```
- **Innovation:** Using densMAP for fraud detection — dense clusters of normal transactions should be visually dense, while sparse anomalous clusters should be visually sparse. This provides a natural "density signal" for analysts.

---

### Application 7: Recommendation Systems — Embedding Visualization and Auditing

- **Domain:** E-commerce / media streaming
- **Problem solved:** Collaborative filtering and matrix factorization produce user and item embeddings in 64–512 dimensional spaces. Product/ML teams need to understand embedding quality, spot biases, audit cold-start behavior, and explain recommendations.
- **Why UMAP:** Transforms high-dimensional user/item embeddings to 2D for interactive dashboards; `cosine` metric works well for normalized collaborative filtering embeddings; fast enough for periodic refreshes of production embedding spaces.
- **Evidence:** A Netflix-adjacent paper notes that "projection of good embeddings via UMAP corresponds to characteristic good attributes" — UMAP quality is used as a proxy for embedding quality assessment.
- **Specific use cases:**
  - Visualize product clusters to audit recommendation diversity
  - Identify "embedding collapse" (all items clustering at one point)
  - Debug cold-start items that fail to cluster with similar items
  - Map user segments to explain personalization to stakeholders

---

### Application 8: Time Series Classification via Feature Spaces

- **Domain:** Industrial IoT / healthcare / finance
- **Problem solved:** Raw time series can't be directly passed to UMAP (different lengths, high autocorrelation structure). The pattern: extract Catch22 features (22 CAnonical Time-series CHaracteristics), then run UMAP on the 22-dimensional feature space.
- **Why this works:** Catch22 covers autocorrelation, successive differences, value distributions, outliers, and fluctuation scaling — exactly the properties that make time series distinct. UMAP then finds the manifold structure in this feature space.
- **Reference:** Lubba et al. 2019, "catch22: CAnonical Time-series CHaracteristics" (Data Mining and Knowledge Discovery, DOI 10.1007/s10618-019-00647-x). HAL paper: "Improved Time-Series Clustering with UMAP dimension reduction method" (hal-03188503).
- **Workflow:**
  ```python
  from pycatch22 import catch22_all
  import umap
  
  # Extract Catch22 features for each time series
  features = np.array([catch22_all(ts)['values'] for ts in time_series_list])
  
  # Run UMAP
  reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
  embedding = reducer.fit_transform(features)
  ```
- **Innovation at scale:** For accelerometer data (e.g., calf behavior classification, arXiv 2404.18159), Catch22 + UMAP identifies behavioral states (rumination, grazing, resting) without labeled data.

---

## SECTION 11 — Mental Model: Expert Practitioner's Checklist

### Before Fitting

**Data preprocessing checklist:**
1. **Scale your features.** UMAP uses distances — if your features have very different magnitudes (age in years vs. salary in dollars), standardize first. Use `StandardScaler` for continuous features. EXCEPTION: If using `metric='cosine'` on L2-normalized features, don't double-normalize.
2. **Handle missing values.** UMAP does not handle NaN natively. Impute or use masking.
3. **Consider PCA pre-processing for very high dimensions.** For > 500 features, consider PCA(50) → UMAP(2). This speeds up pynndescent's ANN search dramatically and avoids the curse of dimensionality in the ANN step. Standard scanpy workflow: PCA(50) → UMAP.
4. **Check for constant features.** Zero-variance features add noise to distance computations.
5. **Consider the appropriate metric** for your data type (see Section 4, `metric`).
6. **Dataset size and compute:** For > 1M points, use cuML. For > 100k points on CPU, expect minutes. For > 10M points, only cuML with batching.

**Questions to answer before running:**
- Is this for visualization (n_components=2) or downstream ML (n_components=5-50)?
- Do I have labels? (Consider supervised/semi-supervised)
- Do I care about density information? (Consider densMAP)
- Do I need to embed new points later? (Consider ParametricUMAP)
- What distance best represents my domain knowledge?

---

### During Fitting

**Signs of healthy optimization:**
- Loss decreases monotonically (with minor oscillations) when `verbose=True`
- The embedding doesn't look like a random cloud after training

**Signs of problems:**
- Loss doesn't decrease → learning_rate too high or data issue
- Embedding collapses to a single point → n_neighbors too high for the dataset
- Embedding has one massive blob and a few isolated points → outliers pulling the manifold

**Memory issues:** If OOM, enable `low_memory=True` (now default). If still OOM, use cuML which handles out-of-core processing.

---

### After Fitting — What to Check

**Visual checks:**
1. **No unexplained isolated points** — if you see them, check for outliers in original data
2. **Cluster number makes domain sense** — if you expect 5 cell types and see 50 clusters, n_neighbors may be too small
3. **Reproducibility test** — run twice with different random_state; if cluster positions and topology change drastically, you may need more epochs or spectral init
4. **Run with different n_neighbors** — if the picture changes completely with n_neighbors=5 vs n_neighbors=50, your conclusions may not be robust

**Statistical checks:**
1. **Trustworthiness** — measures whether k-nearest neighbors in embedding were also neighbors in original space (see Section 12)
2. **Continuity** — reverse: whether neighbors in original space stayed close in embedding
3. **kNN classification accuracy** — train a kNN on the embedding and test it; if accuracy is much lower than in original space, UMAP is losing discriminative structure

**Common pitfalls to verify:**
- **Cluster size is NOT meaningful** — a cluster occupying 10% of the UMAP plot does NOT represent 10% of the data. Don't make quantitative claims about cluster sizes.
- **Inter-cluster distances are PARTIALLY meaningful with spectral init** — you can say "these two clusters are further from each other globally", but don't over-interpret the pixel distances.
- **Intra-cluster distances are NOT meaningful** — UMAP compresses clusters, so you cannot infer that within-cluster variation is small just because points look close.
- **UMAP + clustering is self-affirming** — running HDBSCAN on the UMAP embedding and then saying "UMAP shows these are distinct clusters" is circular: UMAP was designed to create tight clusters. Always validate clusters in the original space.

**The publication-readiness checklist:**
- [ ] Report hyperparameters used (n_neighbors, min_dist, metric, init)
- [ ] Report random_state for reproducibility
- [ ] Show that topology is stable across multiple random seeds
- [ ] Do not interpret inter-cluster distances as absolute similarities
- [ ] If claiming something about density, use densMAP

---

## SECTION 12 — Evaluation Metrics

### Metric 1: Trustworthiness (sklearn)

Measures the proportion of k-nearest neighbors in the embedding that were also among the true k-nearest neighbors in the original space. Penalizes "false neighbors" — points that became close in the embedding but were far apart originally.

**Range:** 0 to 1. Higher is better. Values above 0.9 are considered good.

**Formula:** For each sample i, rank all other samples by distance in the original space. For each of i's k-nearest neighbors in the embedding that were NOT among its k-nearest neighbors in the original space, penalize proportional to their rank in the original space.

```python
from sklearn.manifold import trustworthiness

# Compute on subset for speed (full computation is O(n²))
score = trustworthiness(X, embedding, n_neighbors=10)
print(f"Trustworthiness: {score:.3f}")
```

**Limitations:** Trustworthiness can be high (>0.95) even when global distances are severely distorted (geodesic distance correlation as low as 0.3). Local neighborhood structure can be excellent while global topology is destroyed.

---

### Metric 2: Continuity

Reverse of trustworthiness: measures whether k-nearest neighbors in the original space remain close in the embedding. Penalizes "missing neighbors" — points that were close originally but became far apart in the embedding.

```python
# sklearn doesn't have a built-in continuity function
# Use pyDRMetrics package
from pyDRMetrics import DRMetrics

metrics = DRMetrics(X, embedding)
print(f"Trustworthiness: {metrics.T}")
print(f"Continuity: {metrics.C}")
print(f"LCMC: {metrics.LCMC}")  # Local Continuity Meta-Criterion
```

**Installation:** `pip install pyDRMetrics`

---

### Metric 3: Neighborhood Preservation / kNN Accuracy

For labeled datasets: train a kNN classifier on the embedding coordinates and measure accuracy vs. training a kNN in the original space.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# kNN accuracy in original space
k = 15
clf_original = KNeighborsClassifier(n_neighbors=k)
acc_original = cross_val_score(clf_original, X, y, cv=5).mean()

# kNN accuracy in embedding
clf_embedding = KNeighborsClassifier(n_neighbors=k)
acc_embedding = cross_val_score(clf_embedding, embedding, y, cv=5).mean()

print(f"Original space kNN accuracy: {acc_original:.3f}")
print(f"Embedding kNN accuracy: {acc_embedding:.3f}")
print(f"Retention: {acc_embedding/acc_original:.1%}")
```

---

### Metric 4: Silhouette Score (for clustering quality)

If you have ground-truth cluster labels, measure how well the embedding separates them:

```python
from sklearn.metrics import silhouette_score

# In original space (using sample for speed)
sil_original = silhouette_score(X[:5000], y[:5000])

# In embedding
sil_embedding = silhouette_score(embedding[:5000], y[:5000])

print(f"Silhouette in original: {sil_original:.3f}")
print(f"Silhouette in embedding: {sil_embedding:.3f}")
```

High silhouette in embedding with low silhouette in original = UMAP is creating artificial separation. High silhouette in both = genuine cluster structure preserved.

---

### Metric 5: Co-ranking Matrix (comprehensive)

The co-ranking matrix generalizes trustworthiness and continuity across all neighborhood sizes simultaneously. The Q_NX score (area under the co-ranking curve for varying k) gives a global summary.

```python
# Using pyDRMetrics
from pyDRMetrics import DRMetrics
metrics = DRMetrics(X_subset, embedding_subset)
print(f"Q_NX (overall): {metrics.Q_NX}")
```

---

### Metric 6: Geodesic Distance Correlation (global structure)

For assessing global structure, compute the correlation between pairwise geodesic distances in the original manifold and pairwise Euclidean distances in the embedding. High correlation means UMAP preserved global topology.

```python
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import numpy as np

# On a sample to keep it tractable
idx = np.random.choice(len(X), 1000, replace=False)
X_sample = X[idx]
emb_sample = embedding[idx]

# Approximate geodesic with graph distance (using kNN graph)
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

G = kneighbors_graph(X_sample, n_neighbors=15, mode='distance')
geo_distances = shortest_path(G, method='D')

# Embedding distances
emb_distances = pairwise_distances(emb_sample)

# Correlation (using upper triangle only)
mask = np.triu_indices(len(X_sample), k=1)
corr, pval = spearmanr(geo_distances[mask], emb_distances[mask])
print(f"Geodesic-Embedding distance correlation (Spearman): {corr:.3f}")
```

---

### Metric 7: Reconstruction Error (for ParametricUMAP with decoder)

When using ParametricUMAP with autoencoder:

```python
# Access training history
history = embedder._history
print(f"Final reconstruction loss: {history['loss'][-1]:.4f}")
print(f"Final UMAP loss: {history['umap_loss'][-1]:.4f}")

# Explicit reconstruction error
X_reconstructed = embedder.inverse_transform(embedding)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"Mean squared reconstruction error: {reconstruction_error:.4f}")
```

---

## SECTION 13 — UMAP Mathematical Foundations (Summary for Writing Agent)

This section provides the mathematical concepts the writing agent must explain in depth.

### 1. Riemannian Manifold Assumption

UMAP assumes data lies on or near a Riemannian manifold (a smooth, curved space that locally looks like flat Euclidean space). The key theoretical assumption: data is **uniformly distributed on this manifold**. This is the "U" in UMAP.

If data were actually uniform on the manifold, points with many close neighbors live in a low-curvature flat region; points with sparse neighbors live in a high-curvature or high-dimensional region. UMAP uses this to adaptively set the local metric: for each point, the distance to its kth nearest neighbor sets the local scale factor (call it `sigma_i`). This makes all neighborhood sizes "locally equivalent" — each point sees n_neighbors neighbors at approximately the same local distance.

### 2. Fuzzy Simplicial Sets

Once the local adaptive metric is set, UMAP constructs a **fuzzy simplicial set** from the data. A simplicial set is a combinatorial object that generalizes graphs by including higher-order relationships (triangles, tetrahedra, etc.). The "fuzzy" qualifier means membership in each simplex is a value in [0,1] rather than 0 or 1.

In practice, the relevant piece is the **1-skeleton**: a weighted graph where the edge weight between points i and j represents the probability they are connected in the manifold:

```
w(i,j) = exp(-(d(i,j) - rho_i) / sigma_i)
```

where d(i,j) is the distance, rho_i is the distance to the nearest neighbor (ensuring continuity), and sigma_i is chosen so that `sum_j w(i,j) = log2(n_neighbors)`.

Because this is computed from each point's local perspective and may be asymmetric (w(i,j) ≠ w(j,i)), UMAP combines them using a **fuzzy union**:

```
w_sym(i,j) = w(i,j) + w(j,i) - w(i,j)*w(j,i)
```

This symmetric graph is the high-dimensional fuzzy simplicial set approximation.

### 3. Low-Dimensional Fuzzy Set and Cross-Entropy Optimization

UMAP then constructs a corresponding low-dimensional fuzzy set using a parametric family of curves:

```
v(i,j) = 1 / (1 + a * d_low(i,j)^(2b))
```

where d_low is the Euclidean distance in the embedding, and a and b are fit from min_dist and spread via nonlinear least squares.

The objective is to minimize the **cross-entropy** between the high-dim fuzzy set W and the low-dim fuzzy set V:

```
L = sum_{(i,j)} [w(i,j) * log(w(i,j)/v(i,j)) + (1-w(i,j)) * log((1-w(i,j))/(1-v(i,j)))]
```

This is optimized via **negative sampling** SGD: positive samples are drawn proportional to edge weights w(i,j); negative samples are random pairs. This is computationally O(n log n) thanks to the sparse graph.

### 4. Why Cross-Entropy, not KL Divergence (like t-SNE)

t-SNE minimizes KL(P||Q) where P is the high-dim distribution and Q is the low-dim. KL divergence has different behavior for zeros: KL(P||Q) is infinity if Q=0 when P>0, meaning t-SNE is heavily penalized for "missing" high-dim neighbors. Cross-entropy allows more flexibility — the second term (for pairs not connected in high-dim) acts as a repulsive force pushing embedded points apart.

The Damrich/Böhm paper (2022) shows that minimizing UMAP's cross-entropy with negative sampling is equivalent to a contrastive learning objective, making the theoretical connection explicit.

---

## SECTION 14 — Code Patterns for Writing Agent

### Pattern 1: Complete UMAP Pipeline with Evaluation

```python
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit UMAP
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    init='spectral',      # critical for global structure
    random_state=42,
    verbose=True
)
embedding = reducer.fit_transform(X_scaled)

# Evaluate
trust = trustworthiness(X_scaled, embedding, n_neighbors=15)
print(f"Trustworthiness (k=15): {trust:.3f}")

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0], embedding[:, 1],
    c=labels, cmap='tab10', s=1, alpha=0.7
)
plt.colorbar(scatter)
plt.title(f"UMAP Embedding (n_neighbors=15, min_dist=0.1)\nTrustworthiness: {trust:.3f}")
plt.show()
```

### Pattern 2: Hyperparameter Grid Search with Trustworthiness

```python
import umap
from sklearn.manifold import trustworthiness
import itertools
import pandas as pd

n_neighbors_range = [5, 10, 15, 30, 50]
min_dist_range = [0.0, 0.1, 0.25, 0.5]

results = []
for n_nb, md in itertools.product(n_neighbors_range, min_dist_range):
    reducer = umap.UMAP(n_neighbors=n_nb, min_dist=md, random_state=42)
    emb = reducer.fit_transform(X)
    trust = trustworthiness(X, emb, n_neighbors=n_nb)
    results.append({
        'n_neighbors': n_nb, 'min_dist': md, 'trustworthiness': trust
    })

df = pd.DataFrame(results)
pivot = df.pivot(index='n_neighbors', columns='min_dist', values='trustworthiness')
print(pivot.round(3))
```

### Pattern 3: densMAP with density visualization

```python
import umap
import matplotlib.pyplot as plt
import numpy as np

# densMAP with density output
reducer = umap.UMAP(
    densmap=True,
    output_dens=True,
    dens_lambda=2.0,
    random_state=42
)
result = reducer.fit_transform(X)
embedding = result[0]  # (n_samples, 2)
radii_original = result[1]  # log-radii in original space
radii_embedded = result[2]  # log-radii in embedding

# Color by original space density
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(embedding[:,0], embedding[:,1], c=radii_original, 
            cmap='RdYlBu_r', s=1, alpha=0.7)
plt.colorbar(label='Original space log-radius (lower = denser)')
plt.title('densMAP — colored by original space density')

plt.subplot(1, 2, 2)
plt.scatter(embedding[:,0], embedding[:,1], c=labels,
            cmap='tab10', s=1, alpha=0.7)
plt.title('densMAP — colored by cell type')
plt.tight_layout()
plt.show()
```

### Pattern 4: Supervised UMAP for metric learning

```python
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Train supervised UMAP
reducer = umap.UMAP(
    n_neighbors=15,
    target_weight=0.8,      # strong supervision
    target_metric='categorical',
    random_state=42
)
reducer.fit(X_train, y=y_train)

# Get training embedding
train_embedding = reducer.embedding_

# Transform test data
test_embedding = reducer.transform(X_test)

# Simple kNN in embedding space
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_embedding, y_train)
y_pred = knn.predict(test_embedding)
print(classification_report(y_test, y_pred))
```

### Pattern 5: ParametricUMAP for production inference

```python
from umap.parametric_umap import ParametricUMAP
import numpy as np

# Train ParametricUMAP
embedder = ParametricUMAP(
    n_components=2,
    n_training_epochs=5,
    batch_size=1000
)
training_embedding = embedder.fit_transform(X_train)

# Save
embedder.save("umap_model_v1")

# Load and use for inference (no re-training needed)
from umap.parametric_umap import load_ParametricUMAP
loaded = load_ParametricUMAP("umap_model_v1")

# Fast inference on new data
new_embedding = loaded.transform(X_new)  # milliseconds per point
```

### Pattern 6: GPU UMAP with cuML

```python
# Zero-code-change from umap-learn to cuML
from cuml.manifold import UMAP as cuUMAP  # swap this import
import numpy as np

# Identical API
reducer = cuUMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X)  # runs on GPU — 60x faster on H100

# Works with numpy arrays (converted internally to GPU memory)
# Or pass cupy arrays for zero-copy
import cupy as cp
X_gpu = cp.asarray(X)
embedding_gpu = reducer.fit_transform(X_gpu)
```

---

## SECTION 15 — Common Pitfalls with Concrete Examples

### Pitfall 1: Interpreting cluster distances as similarity

**Wrong:** "Cell type A and B are close on the UMAP, so they must be developmentally related."

**Why wrong:** Inter-cluster distances in UMAP depend heavily on `n_neighbors` and `spread`. With n_neighbors=50 the same cell types might appear far apart. The distance has no quantitative meaning.

**Right:** Use trajectory inference tools (Monocle, scVelo) that operate on the kNN graph, not UMAP distances.

---

### Pitfall 2: Claiming global structure from random initialization

**Wrong (old code):**
```python
reducer = umap.UMAP(init='random')  # breaks global structure
embedding = reducer.fit_transform(X)
# "UMAP shows better global structure than t-SNE"
```

**Wrong because:** With random init, UMAP loses its global structure advantage. If you're comparing UMAP to t-SNE on global structure, ensure UMAP uses `init='spectral'` (the modern default).

---

### Pitfall 3: Running UMAP without standardization

**Wrong:**
```python
# Feature 1: age (0-100), Feature 2: salary (20000-200000)
reducer = umap.UMAP()
embedding = reducer.fit_transform(raw_features)
```

**Wrong because:** Salary dominates all distances. The UMAP sees an essentially 1D dataset.

**Right:**
```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(raw_features)
```

---

### Pitfall 4: Trusting cluster count without cross-validation

**Wrong:** "UMAP shows 8 distinct clusters, so there are 8 cell types."

**Why wrong:** The number of clusters visible in UMAP depends critically on `n_neighbors`. With n_neighbors=5 you might see 20 clusters; with n_neighbors=100 you might see 3. The "right" number requires domain validation.

**Right:** Check cluster stability across n_neighbors ∈ {5, 10, 15, 30, 50}. Clusters that are robust across this range are likely real.

---

### Pitfall 5: Using UMAP embedding as input to another DR method

**Wrong:** Running t-SNE or another UMAP on the UMAP embedding.

**Wrong because:** You are reducing an already distorted representation. Distance relationships in UMAP space are not Euclidean in the original sense; a second DR compounds the distortions unpredictably.

---

### Pitfall 6: Not using densMAP when publishing density claims

**Wrong (common in biology papers):** "Rare cell population X is transcriptionally diverse because it appears scattered in the UMAP."

**Wrong because:** Standard UMAP throws away density. A "scattered" appearance could mean either high diversity OR low cell count (sparse sampling of a homogeneous population).

**Right:** Use `densmap=True` and report the local radii. If the local radius in original space is large, the population is genuinely variable.

---

## SECTION 16 — Key Numbers and Facts (Quick Reference)

- UMAP paper submitted: February 9, 2018 (arXiv 1802.03426)
- umap-learn current stable version: 0.5.12 (April 2026)
- Python requirement: >= 3.9
- License: BSD 3-Clause
- Default n_neighbors: 15
- Default min_dist: 0.1
- Default init: 'spectral' (Laplacian eigenmaps)
- Default metric: 'euclidean'
- Default n_components: 2
- Default n_epochs: auto (500 for small, 200 for large)
- Default learning_rate: 1.0
- GPU speedup (cuML H100 vs CPU): 60x for typical datasets; up to 311x for 20M×384 datasets
- densMAP runtime overhead: ~20% vs standard UMAP
- UMAP vs t-SNE on 5M points: UMAP 47 min, t-SNE 9.4 hours (~12x speedup)
- Embedding stability: UMAP CV=0.08 vs t-SNE CV=0.19 across different random seeds
- Global structure preservation: UMAP 89% vs t-SNE 71% (benchmark on specific datasets)
- densMAP paper DOI: 10.1038/s41587-020-00801-7
- densMAP adds density term active for `dens_frac` (0.3) fraction of epochs
- ParametricUMAP default encoder: 3-layer, 100-neuron FC network (TF/Keras)
- Semi-supervised: unlabeled points marked with y=-1
- Default target_weight: 0.5 (balanced data + label topology)
- Trustworthiness metric range: [0, 1]; > 0.9 is good
- UMAP complexity: O(n log n) for both NN computation and optimization
- t-SNE complexity: O(n² log n) exact; O(n log n) with Barnes-Hut

---

## SECTION 17 — Sources and References

1. McInnes, Healy, Melville (2018). arXiv:1802.03426. https://arxiv.org/abs/1802.03426
2. Narayan, Berger, Cho (2021). Nature Biotechnology 39:765-774. DOI:10.1038/s41587-020-00801-7
3. Damrich, Böhm, Hamprecht, Kobak (2022/2023). arXiv:2206.01816. ICLR 2023. https://arxiv.org/abs/2206.01816
4. Coenen & Pearce (2019). Understanding UMAP. Google PAIR. https://pair-code.github.io/understanding-umap/
5. Kobak & Linderman (2021). Nature Biotechnology 39:156-157. DOI:10.1038/s41587-020-00809-z
6. Becht et al. (2019). Nature Biotechnology 37:38-44. DOI:10.1038/nbt.4314
7. umap-learn documentation (v0.5.8/0.5.12): https://umap-learn.readthedocs.io/
8. cuML UMAP documentation (v26.06): https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.manifold.umap/
9. NVIDIA cuML blog (60x speedup): https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/
10. umap-learn PyPI page: https://pypi.org/project/umap-learn/
11. densvis GitHub (densMAP): https://github.com/hhcho/densvis
12. lmcinnes/umap GitHub: https://github.com/lmcinnes/umap
13. Lubba et al. (2019). catch22. Data Mining and Knowledge Discovery. DOI:10.1007/s10618-019-00647-x
14. "An updated efficient galaxy morphology classification model" arXiv:2512.15137 (2024)
15. "Assessing single-cell transcriptomic variability" arXiv:2306.13750 and Cell Reports 2021
16. sklearn trustworthiness: https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.trustworthiness.html
17. "Stop Misusing t-SNE and UMAP for Visual Analytics" arXiv:2506.08725 (2025)
18. "Simply Statistics: Biologists, stop putting UMAP plots in your papers" (Dec 2024)
