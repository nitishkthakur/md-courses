# Dimensionality Reduction Masterclass — Course Overview

> *A note from me to you: This page is your map. Before you open any algorithm deep-dive, read this from top to bottom once. You will understand every algorithm better — and make fewer bad choices — if you have the whole landscape in your head first. Come back here whenever you are unsure which tool to reach for.*

---

## §1 — What Is Dimensionality Reduction and Why Does It Matter?

### The Curse of Dimensionality (Bellman, 1961)

Richard Bellman coined the phrase in the context of dynamic programming, but the idea cuts across every field that works with high-dimensional vectors. The core observation is geometric: **volume grows exponentially with dimension**.

Picture a unit hypercube — a cube with side length 1 — in $d$ dimensions. Its total volume is always 1. Now shrink each side to $1 - \varepsilon$ and ask: what fraction of the volume lives in a thin shell of thickness $\varepsilon$ near the surface?

$$
\text{fraction in shell} = 1 - (1 - \varepsilon)^d
$$

For $\varepsilon = 0.01$ (a 1% shell) and $d = 1000$: $1 - (0.99)^{1000} \approx 1 - e^{-10} \approx 0.9999955$. Nearly all the volume is in the outermost 1% of the cube. In high dimensions, **the interior is essentially empty** — all your data points are near the boundary, far from one another.

This is not an abstract curiosity. It has three concrete consequences for machine learning:

**1. Sparsity.** To cover a $d$-dimensional unit hypercube with $n$ training points at average spacing $\delta$, you need $n \approx (1/\delta)^d$ points. Double the dimensions, square the data requirement. Your perfectly reasonable 10,000-sample dataset is extremely sparse when $d = 500$.

**2. Distance concentration.** In high dimensions, pairwise distances concentrate — the ratio of the maximum to minimum distance between any two points converges to 1 as $d \to \infty$. Algorithms that depend on meaningful distance comparisons (k-NN, kernel SVMs, clustering) silently degrade. Points that are "near" and "far" become indistinguishable.

**3. Overfitting pressure.** A model with more parameters than effective data constraints memorises noise. When $d$ is large relative to $n$, almost every model class overfits unless regularised aggressively or the effective dimensionality is reduced.

Dimensionality reduction is the principled answer to all three problems: find a lower-dimensional representation that preserves the structure you care about, discard the rest.

---

### The Three Goals of Dimensionality Reduction

Not every DR problem is the same. Before picking an algorithm, know which goal you are optimising for:

| Goal | What you want | Typical output dim | Examples |
|---|---|---|---|
| **Visualisation** | 2-D or 3-D scatter you can inspect | 2–3 | t-SNE, UMAP, PCA |
| **Compression / noise removal** | Compact representation that reconstructs well | 10–100 | PCA, Autoencoders, NMF |
| **Feature learning** | Downstream ML model performs better in the embedding | 10–200 | LDA, UMAP + classifier, VAE |

These goals sometimes conflict. t-SNE produces beautiful 2-D visualisations but its embeddings are **not useful as features for a downstream model**. PCA embeddings are excellent features but rarely produce interpretable 2-D plots. Know your goal before you start.

---

### Four Axes That Classify Every DR Algorithm

**Linear vs Nonlinear.**
Linear methods (PCA, LDA, NMF, ICA, Random Projections) find a matrix $W$ such that $Z = XW$. The embedding is a linear combination of the original features. They are fast, invertible, and interpretable — but they cannot "unfold" a curved manifold.

Nonlinear methods (t-SNE, UMAP, Isomap, Autoencoders) learn arbitrary mappings. They can discover structure that no linear method can see, but they are slower, harder to interpret, and rarely have a closed-form inverse.

**Global vs Local structure.**
Global methods (PCA, MDS, Isomap) try to preserve large-scale geometry — the relative positions of clusters far apart in the original space. Local methods (LLE, t-SNE) preserve neighbourhood relationships but may distort global layout. UMAP tries to balance both.

**Supervised vs Unsupervised.**
Most DR methods are unsupervised — they see only $X$. LDA is fully supervised (uses class labels $y$ to maximise between-class separation). UMAP supports optional supervision. Supervised DR almost always produces better features for classification, but requires labels at fit time.

**Generative vs Discriminative (for deep methods).**
Autoencoders learn a bottleneck representation that reconstructs input. Variational Autoencoders impose a structured latent prior and can generate new samples. These distinctions matter when you need uncertainty estimates or data augmentation.

---

## §2 — Algorithm Taxonomy

The table below covers every algorithm family in this course. Complexity is expressed in terms of $n$ (samples) and $p$ (original features); $k$ = output dimensions.

| Algorithm | Type | Preserves | Supervised? | Best For | Complexity |
|---|---|---|---|---|---|
| **PCA** | Linear | Global | No | Baseline compression, noise removal, correlated features | $O(np^2)$ or $O(n^2p)$ for $n \ll p$ |
| **LDA** | Linear | Global | Yes (class labels) | Classification pre-processing, maximising class separation | $O(np^2 + p^3)$ |
| **NMF** | Linear | Global (parts) | No | Topic modelling, spectral data, non-negative features | $O(npk \cdot \text{iters})$ |
| **ICA** | Linear | Statistical independence | No | Signal separation, EEG/audio blind source separation | $O(np^2 \cdot \text{iters})$ |
| **Random Projections** | Linear | Approximate distances (JL lemma) | No | Very large $n$, $p$; speed critical | $O(npk)$ |
| **t-SNE** | Nonlinear | Local | No | Visualisation of cluster structure (2-D only) | $O(n^2)$ exact, $O(n \log n)$ Barnes-Hut |
| **UMAP** | Nonlinear | Local + Global | Optional | Visualisation + downstream features, large datasets | $O(n^{1.14})$ approx |
| **Manifold methods** (Isomap, LLE, Spectral, MDS, PHATE) | Nonlinear | Local / Geodesic | No | Data on true low-dim manifold (images, trajectories) | $O(n^2)$–$O(n^3)$ |
| **Clustering as DR** (k-Means, GMM soft assignments) | Nonlinear | Prototype distance | No | When cluster membership is the feature | $O(nkp \cdot \text{iters})$ |
| **Autoencoders / VAE** | Nonlinear (deep) | Learned | Optional | Complex nonlinear structure, generation, anomaly detection | $O(n \cdot \text{network size} \cdot \text{epochs})$ |

---

## §3 — Workhorse Algorithms (Easiest → Hardest)

These six algorithms are what you should reach for first on any new DR problem. They are listed from easiest to use and interpret to most nuanced.

---

### 1. PCA — Principal Component Analysis

PCA finds the orthogonal directions of maximum variance in your data by computing the eigenvectors of the covariance matrix $X^TX$. It is linear, closed-form, fast, and losslessly invertible. The components are ranked by explained variance, giving you a natural criterion for choosing $k$. PCA is the universal starting point: run it first, inspect the scree plot, and only move to a more complex method if PCA genuinely fails to separate the structure you care about.

**Typical use case:** Noise removal in tabular data, speeding up downstream models, visualising gene expression or financial returns.

```python
from sklearn.decomposition import PCA
Z = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
```

📖 Deep dive: [pca.md](pca.md)

---

### 2. LDA — Linear Discriminant Analysis

LDA is the supervised counterpart to PCA. Instead of maximising variance, it maximises the ratio of between-class scatter to within-class scatter, producing at most $C - 1$ components (where $C$ is the number of classes). If you have labels and your goal is classification, LDA features almost always outperform PCA features on the same downstream model. The catch: it assumes Gaussian class-conditional distributions and equal covariance matrices, and it requires labels at fit time.

**Typical use case:** Multi-class classification pre-processing, face recognition pipelines, medical diagnosis with labelled cohorts.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Z = LinearDiscriminantAnalysis(n_components=C-1).fit_transform(X_scaled, y)
```

📖 Deep dive: [lda.md](lda.md)

---

### 3. NMF — Non-negative Matrix Factorisation

NMF decomposes $X \approx WH$ where both $W$ (basis) and $H$ (coefficients) are constrained to be non-negative. The non-negativity constraint forces a parts-based decomposition — components can only add, not cancel, so they tend to be interpretable (topics, spectral peaks, image patches). NMF is the right choice whenever your data is inherently non-negative (counts, pixel intensities, frequencies) and you want interpretable components rather than variance-maximising axes.

**Typical use case:** Topic modelling on TF-IDF matrices, hyperspectral image unmixing, audio source separation, single-cell RNA-seq.

```python
from sklearn.decomposition import NMF
W = NMF(n_components=20, init='nndsvda', random_state=42).fit_transform(X)
```

📖 Deep dive: [nmf.md](nmf.md)

---

### 4. ICA — Independent Component Analysis

ICA finds a linear unmixing matrix that produces components with maximally non-Gaussian, statistically independent distributions. Where PCA finds uncorrelated components (zero second-order correlation), ICA finds independent components (zero higher-order correlation too). ICA is the correct tool when your data is a linear mixture of independent non-Gaussian source signals — the canonical example is the cocktail-party problem: recovering individual speakers from microphone recordings that each capture a weighted sum of all voices.

**Typical use case:** EEG artefact removal (eye blinks, muscle noise), audio source separation, financial factor modelling, fMRI signal decomposition.

```python
from sklearn.decomposition import FastICA
S = FastICA(n_components=10, random_state=42).fit_transform(X_scaled)
```

📖 Deep dive: [ica.md](ica.md)

---

### 5. t-SNE — t-Distributed Stochastic Neighbour Embedding

t-SNE converts pairwise similarities in the original space to probabilities, then finds a 2-D (or 3-D) embedding that minimises the KL divergence between the high-dimensional and low-dimensional similarity distributions. Using a heavy-tailed Student-t kernel in the embedding space prevents crowding. t-SNE produces stunning cluster visualisations, but **the distances between clusters are not interpretable**, the algorithm is non-deterministic, has $O(n^2)$ memory (or $O(n \log n)$ with Barnes-Hut), and produces a different result every run. Use it for visual communication only — never as features for a downstream model.

**Typical use case:** Visualising MNIST, single-cell RNA-seq clusters, word embeddings, CIFAR-10 feature spaces.

```python
from sklearn.manifold import TSNE
Z = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca50)
```

📖 Deep dive: [tsne.md](tsne.md)

---

### 6. UMAP — Uniform Manifold Approximation and Projection

UMAP constructs a weighted k-nearest-neighbour graph in the original space (using Riemannian geometry and fuzzy simplicial sets) then optimises a low-dimensional layout that preserves that graph structure. It is faster than t-SNE, preserves more global structure, supports `transform()` on new data (t-SNE does not), and supports optional supervision via label information. UMAP embeddings are usable as features for downstream classifiers. The tradeoff: more hyperparameters than t-SNE and the theoretical underpinnings require more study to use responsibly.

**Typical use case:** Replacing t-SNE for large datasets, building embedding features for a classifier, trajectory analysis in genomics.

```python
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
Z = reducer.fit_transform(X_scaled)
Z_new = reducer.transform(X_test_scaled)  # works unlike t-SNE
```

📖 Deep dive: [umap.md](umap.md)

---

## §4 — Advanced / Specialised Algorithms

Reach for these when the workhorse algorithms have failed or when your problem has a specific structural property they are designed for.

---

### Manifold Methods — When Topology Matters

When your data genuinely lies on a low-dimensional curved manifold embedded in a high-dimensional space — think a coiled spring, a face rotating in 3-D, a robot arm's joint-angle trajectory — Euclidean distance misleads. Manifold methods compute geodesic (along-the-surface) distances instead.

| Method | Key idea | When to use |
|---|---|---|
| **Isomap** | Replace Euclidean distance with graph-shortest-path geodesic, then apply MDS | Data on a smooth, globally connected manifold |
| **LLE** (Locally Linear Embedding) | Express each point as a linear combo of its neighbours; preserve those weights in 2-D | Densely sampled, gently curved manifolds |
| **Spectral Embedding** | Embed via eigenvectors of the graph Laplacian | When you care about connectivity / graph structure |
| **MDS** (Multi-Dimensional Scaling) | Find embedding that preserves pairwise dissimilarity matrix | When you start from a distance/dissimilarity matrix, not raw features |
| **PHATE** | Diffusion operator + potential distance; preserves both local and global trajectory structure | Single-cell genomics, time-series trajectories |

📖 Deep dive: [manifold-methods.md](manifold-methods.md)

---

### Clustering as DR — Prototype-Based Features

When your downstream model only needs to know "which cluster does this point belong to?", cluster assignment is itself a form of dimensionality reduction. Soft k-Means or GMM posterior probabilities give you a $k$-dimensional feature vector (one dimension per cluster) that is often more useful than a continuous embedding.

Reach for this when: you want interpretable, discrete features; you are building a rule-based system on top; or when $k$ is very small (< 20) and the clusters are well-separated.

📖 Deep dive: [clustering-as-dr.md](clustering-as-dr.md)

---

### Autoencoders / VAE — Deep Nonlinear DR with Generation

An autoencoder trains an encoder $f_\theta: \mathbb{R}^p \to \mathbb{R}^k$ and decoder $g_\phi: \mathbb{R}^k \to \mathbb{R}^p$ jointly to minimise reconstruction loss. The bottleneck layer is the embedding. Autoencoders can learn arbitrary nonlinear structure that no linear method can capture — but they require more data, more tuning, and GPU training.

A Variational Autoencoder (VAE) adds a KL-divergence regulariser that forces the latent space to match a prior (typically $\mathcal{N}(0, I)$), enabling smooth interpolation and generation of new samples.

Reach for autoencoders when: PCA and UMAP both fail; you have > 50k samples; you need to generate new samples; or you need uncertainty estimates from the latent space.

📖 Deep dive: [autoencoder-dr.md](autoencoder-dr.md)

---

### Random Projections — Speed and Scale Above All Else

The Johnson-Lindenstrauss lemma guarantees that a random linear projection from $\mathbb{R}^p$ to $\mathbb{R}^k$ (with $k = O(\log n / \varepsilon^2)$) preserves all pairwise distances to within factor $(1 \pm \varepsilon)$ with high probability. No fitting required — just multiply by a random matrix.

Reach for random projections when: $n$ or $p$ is in the millions; you need a projection in milliseconds; approximate distance preservation is sufficient; or you are pre-processing before a more expensive method.

```python
from sklearn.random_projection import SparseRandomProjection
Z = SparseRandomProjection(n_components=256, random_state=42).fit_transform(X)
```

📖 Deep dive: [random-projections.md](random-projections.md)

---

## §5 — Quick-Reference Decision Table

Use this table when you have a specific constraint and need to narrow the field fast. ✅ = good fit, ⚠️ = possible with caveats, ❌ = poor fit or not supported.

| Problem characteristic | PCA | LDA | NMF | ICA | t-SNE | UMAP | Manifold | Autoencoder | Random Proj |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small dataset** ($n < 1000$) | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ❌ | ✅ |
| **Large dataset** ($n > 100k$) | ✅ | ✅ | ⚠️ | ⚠️ | ❌ | ✅ | ❌ | ✅ | ✅ |
| **Very high-dim** ($p > 10k$) | ✅ | ⚠️ | ✅ | ❌ | ⚠️ | ✅ | ❌ | ✅ | ✅ |
| **Have class labels** | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ |
| **Visualisation only (2-D)** | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Need transform() on new data** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| **Need interpretable components** | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Speed is critical** | ✅ | ✅ | ⚠️ | ⚠️ | ❌ | ✅ | ❌ | ❌ | ✅ |
| **Nonlinear manifold structure** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Sparse input matrix** | ✅ | ✅ | ✅ | ❌ | ⚠️ | ✅ | ❌ | ⚠️ | ✅ |
| **Non-negative data** | ⚠️ | ⚠️ | ✅ | ❌ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| **Downstream ML features** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ✅ | ✅ |
| **Generation / sampling** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (VAE) | ❌ |

---

## §6 — How to Evaluate a Dimensionality Reduction

Most practitioners eyeball a 2-D scatter and call it a day. Do not do that. Here are five principled evaluation strategies, in order of how much you should weight them.

---

### 1. Downstream Task Performance (Gold Standard)

If you are using DR as a preprocessing step, the only evaluation that matters is whether your downstream metric (accuracy, AUC, RMSE) improves. Fit your model on the embedding and compare to fitting on the raw features and on the baseline (no DR).

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([("pca", PCA(n_components=50)), ("clf", LogisticRegression())])
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
print(f"PCA+LR AUC: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

### 2. Reconstruction Error (Linear Methods)

For methods that have an inverse transform (PCA, NMF, Autoencoders), measure how well the embedding reconstructs the original data. For PCA this is the fraction of variance explained.

```python
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=50).fit(X_train)
X_reconstructed = pca.inverse_transform(pca.transform(X_test))
reconstruction_error = np.mean((X_test - X_reconstructed) ** 2)
variance_explained = pca.explained_variance_ratio_.sum()

print(f"Reconstruction MSE: {reconstruction_error:.4f}")
print(f"Variance explained: {variance_explained:.3%}")
```

💡 Plot the **scree plot** (explained variance vs number of components) and look for the "elbow" — the point where adding more components gives diminishing returns.

---

### 3. Trustworthiness and Continuity (sklearn metrics)

**Trustworthiness** measures whether nearby points in the embedding were also nearby in the original space (no false neighbours). **Continuity** measures the converse: nearby points in the original space remain nearby in the embedding (no tears). Both are in $[0, 1]$; higher is better.

```python
from sklearn.manifold import trustworthiness

# Trustworthiness: embedding is not creating false neighbours
t = trustworthiness(X_original, Z_embedded, n_neighbors=5)
print(f"Trustworthiness (n_neighbors=5): {t:.3f}")

# Continuity: swap original and embedded to measure tears
c = trustworthiness(Z_embedded, X_original, n_neighbors=5)
print(f"Continuity (n_neighbors=5): {c:.3f}")
```

⚠️ These metrics are $O(n^2)$ in memory; sub-sample to $n \leq 5000$ for large datasets.

---

### 4. Neighbourhood Preservation — k-NN Accuracy in Embedding

A practical alternative to trustworthiness: train a k-NN classifier in the original space and in the embedding, compare accuracy. If the embedding loses neighbourhood structure, k-NN degrades.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_emb = KNeighborsClassifier(n_neighbors=5)

acc_raw = cross_val_score(knn_raw, X_scaled, y, cv=5).mean()
acc_emb = cross_val_score(knn_emb, Z_embedded, y, cv=5).mean()

print(f"k-NN on raw features: {acc_raw:.3f}")
print(f"k-NN on embedding:    {acc_emb:.3f}")
```

---

### 5. Visual Inspection Checklist

Even when quantitative metrics look good, always make the 2-D plot. Check for:

- [ ] **Separation:** Do known clusters appear separated? (if labels exist)
- [ ] **Crowding:** Are all points collapsed into a single blob? (perplexity too high for t-SNE; `min_dist` too large for UMAP)
- [ ] **Fragmentation:** Is one true cluster split into many? (perplexity too low; `n_neighbors` too small)
- [ ] **Outliers:** Are there points far from all clusters that shouldn't be? (scaling issue or data error)
- [ ] **Reproducibility:** Run the algorithm twice with different `random_state` — do the structures persist? (especially important for t-SNE)

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", alpha=0.6, s=5)
plt.colorbar(scatter, ax=ax, label="Class")
ax.set_title("UMAP embedding coloured by label")
plt.tight_layout()
plt.show()
```

---

## §7 — Common Pitfalls Across All DR Methods

These are the mistakes that appear again and again in production pipelines. Each one is easy to avoid once you know to look for it.

---

### ⚠️ Pitfall 1: Forgetting to Scale

PCA, t-SNE, UMAP, ICA, and LDA all depend on distances or covariance. If one feature is in units of dollars (range 0–100,000) and another is a binary flag (0 or 1), the high-variance feature dominates every computation. Always apply `StandardScaler` (zero mean, unit variance) before any DR method — unless your features are already on the same scale by design, or you are using NMF (which requires non-negative data and should not be mean-centred).

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
# Now safe to pass to PCA, UMAP, t-SNE, LDA, ICA
```

---

### ⚠️ Pitfall 2: Data Leakage — Fit on Train, Transform Both

The scaler and the DR model must be fit **only on training data**, then applied to test data. If you fit on the full dataset before splitting, your test set has influenced the embedding and your evaluation is optimistic.

```python
# WRONG — leakage
X_embedded = PCA(50).fit_transform(X_all)
X_train, X_test = train_test_split(X_embedded)

# RIGHT
X_train, X_test = train_test_split(X_all)
pca = PCA(50).fit(X_train)
X_train_emb = pca.transform(X_train)
X_test_emb  = pca.transform(X_test)
```

Use `sklearn.pipeline.Pipeline` to make this automatic and foolproof.

---

### ⚠️ Pitfall 3: Interpreting t-SNE Inter-Cluster Distances as Meaningful

The most common t-SNE misinterpretation: "Cluster A is far from Cluster B, so they are very different." This is **not guaranteed to be true**. t-SNE's cost function only preserves local neighbourhoods. The positions of clusters relative to each other in the 2-D plot depend on random initialisation and are not interpretable.

Also: the **size of a cluster in t-SNE has no meaning**. Denser clusters in high dimensions may appear larger or smaller depending on local perplexity. If you want to make statements about global structure, use PCA or UMAP instead.

---

### ⚠️ Pitfall 4: Using DR for Downstream ML Without Checking If It Helps

DR is not free. You are discarding information. Before adding a DR step to your pipeline, always benchmark:

1. Baseline model on raw (scaled) features
2. Model on DR features
3. Model on raw + DR features concatenated

Often the raw features alone beat the DR features, especially for tree-based models (XGBoost, Random Forest) that handle high dimensionality well and do their own implicit feature selection.

---

### ⚠️ Pitfall 5: Choosing n_components Without Justification

Picking `n_components=2` because "it's easy to plot" and then using those features for a classifier is a common error. Two dimensions may throw away 90% of the variance. Use one of these principled strategies instead:

- **Scree plot elbow** (PCA): find where the explained variance curve bends
- **95% variance threshold** (PCA): `pca = PCA(n_components=0.95)` in sklearn
- **Reconstruction error vs k curve**: plot MSE against number of components and find diminishing returns
- **Cross-validation**: sweep `n_components` and optimise downstream metric

```python
# PCA: keep enough components to explain 95% of variance
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, svd_solver="full")
X_train_emb = pca.fit_transform(X_train_scaled)
print(f"Components retained: {pca.n_components_}")
```

---

## Course File Index

| File | Algorithm | Type |
|---|---|---|
| [pca.md](pca.md) | Principal Component Analysis | Linear, unsupervised |
| [lda.md](lda.md) | Linear Discriminant Analysis | Linear, supervised |
| [nmf.md](nmf.md) | Non-negative Matrix Factorisation | Linear, unsupervised |
| [ica.md](ica.md) | Independent Component Analysis | Linear, unsupervised |
| [tsne.md](tsne.md) | t-SNE | Nonlinear, unsupervised |
| [umap.md](umap.md) | UMAP | Nonlinear, unsupervised/semi-supervised |
| [manifold-methods.md](manifold-methods.md) | Isomap, LLE, Spectral, MDS, PHATE | Nonlinear, unsupervised |
| [clustering-as-dr.md](clustering-as-dr.md) | k-Means / GMM as DR | Nonlinear, unsupervised |
| [autoencoder-dr.md](autoencoder-dr.md) | Autoencoders and VAE | Deep nonlinear, unsupervised |
| [random-projections.md](random-projections.md) | Gaussian / Sparse Random Projections | Linear, unsupervised |

---

*Start with [pca.md](pca.md). Come back to this page whenever you are choosing between methods.*
