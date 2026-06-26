# PCA and All Its Variants: The Complete Masterclass

> **Why this algorithm matters:** PCA is the algorithm that made high-dimensional data tractable.
> In 1991, Turk and Pentland reduced 10,000-pixel face images to 50 numbers and achieved face
> recognition accuracy that shocked the computer vision community. In 2006, geneticists applied PCA
> to one million SNP measurements per person and recovered — with no labels — the entire migration
> history of modern humans across continents. Today, PCA is the mandatory preprocessing step in
> single-cell RNA sequencing, the backbone of financial risk models at every major bank, and the
> go-to tool for debugging neural network representations. Every data scientist who learns PCA
> superficially (`pca.fit_transform(X)` and move on) is leaving enormous insight on the table.
> This chapter will make you the person who truly understands what is happening inside that call.

---

## §0 — Python Ecosystem & Package Guide

Before the mathematics, you need to know the landscape. PCA is deceptively well-served by Python
packages — deceptive because most of them do subtly different things, and choosing the wrong one
for your use case costs you either accuracy, speed, interpretability, or all three.

### Complete Package Table

| Package | Import | Version (mid-2026) | What It Provides | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.decomposition import PCA` | ~1.5.x | Full PCA family: exact, randomized, incremental, kernel, sparse, factor analysis | BSD-3 |
| `prince` | `import prince` | ~0.19.x | MCA (categorical), FAMD (mixed data), MFA (grouped), interactive Altair plots, supplementary projections | MIT |
| `sklearn.decomposition.TruncatedSVD` | `from sklearn.decomposition import TruncatedSVD` | ~1.5.x | SVD without centering — critical for sparse matrices (text/genomics) | BSD-3 |
| `PyParSVD` | `from PyParSVD.streaming import StreamingSVD` | ~1.0.x | Distributed/streaming SVD across nodes; HPC environments | MIT |
| `qrpca` | `from qrpca import QRPCA` | ~0.1.x | GPU-accelerated randomized PCA; 15× speedup for n_features > 10K | MIT |
| `fbpca` | `from fbpca import pca` | 1.0 (archived) | Fast randomized PCA (Facebook); **deprecated — use sklearn** | BSD |
| `pyDRMetrics` | `from pyDRMetrics.pyDRMetrics import DRMetrics` | ~0.3.x | Trustworthiness, continuity, LCMC, co-k-NN evaluation metrics | MIT |
| `optuna` | `import optuna` | ~3.x | Hyperparameter optimization for n_components, gamma, alpha | MIT |

### Top Picks — Recommendation Table

| Package | Best For | Approach | Modern / Legacy |
|---|---|---|---|
| **`sklearn.PCA`** | 95% of use cases: dense data, moderate size, linear structure | Exact SVD / Randomized SVD / covariance eigendecomp | **Modern — first choice** |
| **`sklearn.IncrementalPCA`** | Large datasets that don't fit in RAM; streaming/online data | Incremental SVD with correct mean tracking | Modern |
| **`sklearn.KernelPCA`** | Nonlinear data structure; manifold-shaped data | Kernel eigendecomposition (dual space) | Modern |
| **`sklearn.TruncatedSVD`** | Sparse matrices (text, TF-IDF, genomics counts) | Randomized truncated SVD, no centering | Modern — **mandatory for sparse** |
| **`prince`** | Mixed/categorical data; interactive biplots; supplementary rows | PCA + MCA + FAMD with FactoMineR-validated math | Modern |

> 💡 **Intuition:** Think of these packages as four different cameras pointing at the same scene.
> `sklearn.PCA` is a high-quality standard camera — reliable, fast, works in most situations.
> `TruncatedSVD` is the same camera with a "don't touch the brightness" mode — essential when
> centering would destroy information (sparse matrices). `KernelPCA` is a fisheye lens — it distorts
> the input space to reveal curved structure. `prince` is a cinema camera with built-in color
> grading — it handles mixed data types and produces beautiful output. Pick the right camera
> before you shoot.

### The Same Transform Across Top Packages

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load a shared dataset (used throughout this chapter)
digits = load_digits()
X, y = digits.data, digits.target  # (1797, 64) — 8x8 pixel images

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Option 1: sklearn.PCA — the everyday workhorse ────────────────────────────
from sklearn.decomposition import PCA

pca_sklearn = PCA(
    n_components=40,         # Keep 40 principal components
    svd_solver='auto',       # sklearn picks: 'randomized' for this size
    whiten=False,
    random_state=42
)
X_train_pca = pca_sklearn.fit_transform(X_train_sc)
X_test_pca  = pca_sklearn.transform(X_test_sc)
print(f"sklearn PCA shape: {X_train_pca.shape}")
print(f"Variance explained: {pca_sklearn.explained_variance_ratio_.sum():.3f}")

# ── Option 2: IncrementalPCA — for large data ──────────────────────────────────
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=40, batch_size=200)
ipca.fit(X_train_sc)  # Internally splits into 200-sample batches
X_train_ipca = ipca.transform(X_train_sc)
print(f"IncrementalPCA variance explained: {ipca.explained_variance_ratio_.sum():.3f}")

# ── Option 3: TruncatedSVD — for sparse matrices ────────────────────────────────
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

X_sparse = csr_matrix(X_train)  # Simulate a sparse matrix
tsvd = TruncatedSVD(n_components=40, random_state=42)
X_train_tsvd = tsvd.fit_transform(X_sparse)
print(f"TruncatedSVD shape: {X_train_tsvd.shape}")
# NOTE: No centering happens here — critical for real sparse data

# ── Option 4: KernelPCA — for nonlinear structure ─────────────────────────────
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(
    n_components=40,
    kernel='rbf',
    gamma=1 / X_train_sc.shape[1],  # Median heuristic: 1/n_features
    fit_inverse_transform=True,      # Enable reconstruction error
    random_state=42,
    n_jobs=-1                        # Use all cores for kernel matrix computation
)
X_train_kpca = kpca.fit_transform(X_train_sc)
print(f"KernelPCA shape: {X_train_kpca.shape}")

# ── Option 5: prince — for mixed / categorical data ───────────────────────────
# import prince  # pip install prince
# pca_prince = prince.PCA(n_components=2, n_iter=3, random_state=42)
# pca_prince = pca_prince.fit(X_df)  # Expects a DataFrame
# chart = pca_prince.plot(X_df)      # Returns interactive Altair chart
# (Commented: requires pandas DataFrame input; uncomment if installed)
```

> 🔧 **In practice:** For 99% of new projects, start with `PCA(n_components=None, svd_solver='auto')`,
> look at the scree plot, then set `n_components` based on the elbow or variance threshold. Only
> switch to a specialized variant if you have a concrete reason: data doesn't fit in RAM (→ IPCA),
> sparse input (→ TruncatedSVD), nonlinear structure (→ KernelPCA), categorical/mixed types
> (→ prince).

### GPU-Accelerated PCA

For datasets with n_features > 10,000 (genomics, imaging, large NLP):

```python
# qrpca: GPU-accelerated randomized PCA
# pip install qrpca  # Requires CUDA GPU
# from qrpca import QRPCA
# gpu_pca = QRPCA(n_components=50)
# gpu_pca.fit(X_large)  # Claims 15x speedup over sklearn for n_features > 10K
# arXiv: 2206.06797

# cuML (RAPIDS): GPU PCA via NVIDIA ecosystem
# from cuml.decomposition import PCA as cuPCA
# cu_pca = cuPCA(n_components=50)  # Same API as sklearn
# cu_pca.fit(X_large_gpu)          # Data must be on GPU memory
```

> ⚠️ **Pitfall:** `fbpca` (Facebook's fast PCA) is frequently recommended in older tutorials.
> It was excellent in 2014 when sklearn's randomized SVD was less mature. As of 2026, sklearn's
> randomized PCA (with `n_oversamples` and `power_iteration_normalizer` tuning added in v1.1) is
> as fast and more accurate. The `fbpca` repository is archived — do not use it for new projects.

### Streaming / Distributed SVD

For data that cannot fit on a single machine:

```python
# PyParSVD: distributed streaming SVD
# pip install PyParSVD
# from PyParSVD.streaming import StreamingSVD
# 
# svd = StreamingSVD(k=50, ff=1.0)  # k=rank, ff=forgetting_factor
# for batch in data_generator:
#     svd.streaming_svd(batch)      # Updates decomposition with each batch
#
# arXiv: 2108.08845 — designed for MPI/HPC environments
```

### Solver Decision Tree

```
Is your data sparse (TF-IDF, count matrix)?
  YES → TruncatedSVD (NEVER PCA — centering destroys sparsity)
  NO  ↓
Does your data have nonlinear structure (Swiss roll, rings, manifolds)?
  YES → KernelPCA (choose kernel carefully — see §4)
  NO  ↓
Does your data fit in RAM?
  NO  → IncrementalPCA (partial_fit batch by batch)
  YES ↓
n_samples >> n_features AND n_features < 1000?
  YES → PCA(svd_solver='covariance_eigh')  [NEW in sklearn 1.5]
  NO  ↓
n_components ≪ min(n_samples, n_features)?
  YES → PCA(svd_solver='randomized')       [Default for large matrices]
  NO  → PCA(svd_solver='full')             [Exact; needed for n_components='mle']
```

---

## §1 — The Origin: The Paper That Started It All

> 📜 **Origin/Citation:** Pearson, K. (1901). "LIII. On lines and planes of closest fit to
> systems of points in space." *The London, Edinburgh, and Dublin Philosophical Magazine and
> Journal of Science*, 2(11), 559–572. DOI: 10.1080/14786440109462720

### The Problem Pearson Was Solving

In 1901, Karl Pearson was not thinking about machine learning. He was thinking about geometry.
Given a cloud of points in three-dimensional space — say, measurements of skull dimensions from
an anthropology study — he asked: **what is the single line, or plane, that comes closest to all
the points simultaneously?**

This is a precise geometric question. The "closest" in Pearson's formulation meant minimizing the
sum of squared perpendicular distances from each point to the line (not the vertical distances
used in ordinary linear regression). This distinction matters enormously and is still confused by
practitioners today.

The problem that Pearson was reacting against was simple linear regression, which minimizes
residuals in one direction only — implying one variable is "the response" and the others are
"predictors." For Pearson's anthropological data, all variables were symmetric — no single
measurement was the dependent variable. He needed a method that treated all dimensions equally.

**His core insight:** The line of closest fit to a cloud of points passes through the centroid
of the data and points in the direction of maximum spread (variance). There is nothing arbitrary
about this direction — it is the unique direction that minimizes the sum of squared projection
residuals. Pearson showed, by calculus, that this direction is the eigenvector corresponding to
the largest eigenvalue of the data's covariance matrix.

### The Hotelling Reformulation (1933)

> 📜 **Citation:** Hotelling, H. (1933). "Analysis of a complex of statistical variables into
> principal components." *Journal of Educational Psychology*, 24(6), 417–441.

Pearson's paper was geometric. Harold Hotelling's 1933 paper turned PCA into a statistical method.
Hotelling was working with correlated psychological test scores and asked a different question:
**how can we replace a set of correlated measurements with a smaller set of uncorrelated ones?**

Hotelling introduced the term "principal components" and defined them as linear combinations of
the original variables that:
1. Are uncorrelated with each other
2. Explain maximum variance in descending order

He showed that the solution is the eigendecomposition of the correlation matrix (or covariance
matrix), and he used the term "latent roots" for what we now call eigenvalues. This is the
formulation that sklearn implements.

> 💡 **Intuition:** These two formulations — Pearson's "minimize projection residuals" and
> Hotelling's "maximize variance of projections" — are mathematically equivalent. They produce
> identical principal components. But they give you different mental models for when PCA is
> appropriate. Pearson's view: use PCA when you want the best lower-dimensional approximation
> to your data. Hotelling's view: use PCA when you want to find uncorrelated factors that
> explain most of the variation.

### The Formal Algorithm (Both Formulations)

**Formulation 1 — Variance Maximization (Hotelling):**

Given centered data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ (subtract the mean from each feature), find:

$$\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{X}\mathbf{w}) = \arg\max_{\|\mathbf{w}\|=1} \mathbf{w}^T \mathbf{C} \mathbf{w}$$

where $\mathbf{C} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$ is the sample covariance matrix. The second
principal component $\mathbf{w}_2$ maximizes the same objective subject to $\mathbf{w}_2 \perp \mathbf{w}_1$,
and so on.

**Formulation 2 — Reconstruction Error Minimization (Pearson):**

Find the $d$-dimensional subspace (represented by an orthonormal basis matrix
$\mathbf{W} \in \mathbb{R}^{p \times d}$) that minimizes:

$$\min_{\mathbf{W}^T\mathbf{W}=\mathbf{I}} \|\mathbf{X} - \mathbf{X}\mathbf{W}\mathbf{W}^T\|_F^2$$

**The solution to both problems:** Eigendecomposition of the covariance matrix.

$$\mathbf{C} \mathbf{w}_k = \lambda_k \mathbf{w}_k, \quad k = 1, \ldots, p$$

The principal components are the eigenvectors $\mathbf{w}_k$, ordered by descending eigenvalue
$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$. The eigenvalue $\lambda_k$ equals
the variance of the data along principal component $k$.

### Original PCA Algorithm (Pseudocode)

```
Algorithm: Classical PCA
Input:  X ∈ R^{n×p}, target dimensionality d ≤ p
Output: Z ∈ R^{n×d} (projections), W ∈ R^{p×d} (loadings)

1. Center: X̃ = X - μ  where μ = (1/n) Σ_i x_i  (column means)
2. Compute covariance: C = (1/(n-1)) X̃^T X̃  ∈ R^{p×p}
3. Eigendecompose: C = V Λ V^T  (V columns are eigenvectors, Λ diagonal eigenvalues)
4. Sort: order eigenvectors by decreasing eigenvalue
5. Select: W = V[:, :d]  (top-d eigenvectors as columns)
6. Project: Z = X̃ W    (n×d projection scores)
7. Return Z, W
```

In practice, sklearn computes PCA via the SVD of $\tilde{\mathbf{X}}$ directly (not the covariance
matrix), because SVD is more numerically stable when $n < p$ or when the covariance matrix is
ill-conditioned. The relationship is:

$$\tilde{\mathbf{X}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where $\mathbf{V}$'s columns are the principal axes (stored as rows in `pca.components_`),
$\boldsymbol{\Sigma}$'s diagonal entries are the singular values, and
$\lambda_k = \sigma_k^2 / (n-1)$ gives you the eigenvalues. The projection scores are
$\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V} = \mathbf{U}\boldsymbol{\Sigma}$, which is
`pca.transform(X)`.

### What Came Before?

Before PCA, scientists dealt with high-dimensional correlated data through:
- **Simple correlation analysis:** Pearson's own correlation coefficient (1895), which only
  captures pairwise relationships, not the multivariate structure.
- **Multiple regression:** Galton and Pearson's earlier work assumed one response variable.
- **Ad hoc variable selection:** Choose variables by intuition or domain knowledge.

PCA was revolutionary because it required no labels, no prior variable ordering, and no
assumptions about which variables "matter" — the data itself reveals its principal axes.

> 🏆 **Best practice:** Read both the Pearson (1901) and Hotelling (1933) papers if you ever
> teach PCA. They are short (13 pages and 25 pages respectively) and give you the two mental
> models — geometric compression vs. statistical decorrelation — that explain when PCA works
> and when it fails. Both intuitions will come up in the industry applications in §9.

---

## §2 — The Algorithm, Deeply Explained

### Mathematical Foundations

Let's derive PCA from scratch and understand every step.

**Setup:** We have $n$ data points $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^p$, assembled
into a data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$. Our goal is to find a $d$-dimensional
linear subspace ($d \ll p$) onto which we can project the data while losing as little information
as possible.

**Step 1: Center the Data**

PCA always works on mean-centered data. Let $\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$
be the column means. Define $\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}\boldsymbol{\mu}^T$, where
$\mathbf{1}$ is the all-ones vector.

Why center? Because we want to capture deviations from the mean, not absolute values. The covariance
matrix measures dispersion around the mean. If you don't center, the first PC will almost always
point roughly toward the mean vector, which carries no information about structure.

**Step 2: The Optimization Objective**

We want to find a unit vector $\mathbf{w} \in \mathbb{R}^p$, $\|\mathbf{w}\| = 1$, such that
the projected values $z_i = \mathbf{w}^T\tilde{\mathbf{x}}_i$ have maximum variance:

$$\text{Var}(\mathbf{z}) = \frac{1}{n-1}\sum_{i=1}^n z_i^2 = \frac{1}{n-1}\|\tilde{\mathbf{X}}\mathbf{w}\|^2 = \mathbf{w}^T \mathbf{C} \mathbf{w}$$

where $\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} \in \mathbb{R}^{p \times p}$
is the sample covariance matrix.

**Step 3: The Lagrangian**

Using a Lagrange multiplier to enforce $\|\mathbf{w}\|^2 = 1$:

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T \mathbf{C} \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

Taking the derivative and setting to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbf{C}\mathbf{w} - 2\lambda\mathbf{w} = 0$$

$$\boxed{\mathbf{C}\mathbf{w} = \lambda \mathbf{w}}$$

This is exactly the eigenvalue equation! The optimal direction $\mathbf{w}$ is an eigenvector of
the covariance matrix $\mathbf{C}$, and the variance along that direction equals the eigenvalue
$\lambda$. To maximize variance, we choose $\mathbf{w}$ to be the eigenvector with the **largest**
eigenvalue $\lambda_1$.

**Step 4: Finding All Components**

The second principal component is the unit vector $\mathbf{w}_2 \perp \mathbf{w}_1$ that maximizes
the remaining variance. By the same Lagrangian argument (with an additional constraint
$\mathbf{w}_2^T\mathbf{w}_1 = 0$), $\mathbf{w}_2$ is the eigenvector of $\mathbf{C}$ with the second
largest eigenvalue $\lambda_2$.

More generally, the $k$-th principal component is the eigenvector $\mathbf{w}_k$ corresponding to
the $k$-th largest eigenvalue $\lambda_k$. The eigenvectors of a symmetric positive semidefinite
matrix (which $\mathbf{C}$ always is) are automatically orthogonal, so the constraint
$\mathbf{w}_k \perp \mathbf{w}_j$ for $k \neq j$ is automatically satisfied.

> 🔬 **Deep dive:** Why is $\mathbf{C}$ positive semidefinite? Because for any vector $\mathbf{v}$:
> $\mathbf{v}^T\mathbf{C}\mathbf{v} = \frac{1}{n-1}\mathbf{v}^T\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}\mathbf{v} = \frac{1}{n-1}\|\tilde{\mathbf{X}}\mathbf{v}\|^2 \geq 0$.
> This guarantees all eigenvalues are non-negative, which means variances are non-negative —
> a sanity check that the math is self-consistent.

### The SVD Connection

In practice, sklearn never computes the covariance matrix explicitly for large $p$. Instead, it
uses the **Singular Value Decomposition** of the centered data matrix:

$$\tilde{\mathbf{X}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{n \times n}$: left singular vectors (one per sample)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times p}$: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $\mathbf{V} \in \mathbb{R}^{p \times p}$: right singular vectors (one per feature direction)

The connection to PCA:

$$\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} = \frac{1}{n-1}\mathbf{V}\boldsymbol{\Sigma}^T\mathbf{U}^T\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T = \mathbf{V} \frac{\boldsymbol{\Sigma}^2}{n-1} \mathbf{V}^T$$

So the right singular vectors $\mathbf{V}$ are the principal axes (eigenvectors of $\mathbf{C}$),
and the eigenvalues are $\lambda_k = \sigma_k^2 / (n-1)$.

**Why SVD instead of eigendecomposition of C?**

1. **Numerical stability:** When $p > n$ or when two eigenvalues are close, forming $\mathbf{C}$
   explicitly and then eigendecomposing it squares the condition number. SVD on $\tilde{\mathbf{X}}$
   is more stable.
2. **Memory efficiency:** $\tilde{\mathbf{X}} \in \mathbb{R}^{n \times p}$ vs.
   $\mathbf{C} \in \mathbb{R}^{p \times p}$. For $p = 50,000$ (genomics), $\mathbf{C}$ requires
   20GB; $\tilde{\mathbf{X}}$ for $n = 1000$ samples requires only 400MB.
3. **Randomized algorithms:** The best fast PCA methods (Halko et al. 2011) operate on
   $\tilde{\mathbf{X}}$ directly, not on $\mathbf{C}$.

### Geometric Intuition

Sit with this picture for a moment. Imagine your data as a cloud of points in 2D: a long,
thin ellipse tilted at 45 degrees. PCA finds:

- **PC1:** The long axis of the ellipse — the direction of maximum spread.
- **PC2:** The short axis — perpendicular to PC1, capturing the remaining spread.

If you project all points onto PC1 alone, you get a 1D compression that retains as much variance
(information) as possible. The points compressed onto PC1 look like a 1D Gaussian. The
perpendicular "error" from each point to the PC1 line is what's lost — and PCA minimizes the
total squared error of these perpendicular residuals.

> 💡 **Intuition:** PCA is finding the axes of the data's ellipsoid of variation. For a $p$-dimensional
> Gaussian, the principal components are the eigenvectors of the covariance matrix — which are
> exactly the semi-axes of the confidence ellipsoid. If your data is a long thin ellipsoid, PCA
> finds the major axis (PC1) and minor axes (PC2, PC3, ...) in the right order.

**The rotation interpretation:** PCA is a rigid rotation of the coordinate system. You are not
distorting or warping the data — you are finding the natural coordinate system aligned with the
data's structure. The projection $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V}$ is simply the data
expressed in the new rotated coordinates, with axes sorted by decreasing variance.

**The subspace interpretation:** The $d$ retained principal components define a $d$-dimensional
affine subspace in $\mathbb{R}^p$ (the "PC subspace"). Every data point is decomposed into:
- **Signal:** Its projection onto the PC subspace (the $d$-dimensional representation)
- **Noise:** Its perpendicular distance from the subspace (the reconstruction error)

### The Exact Transform Formula

```python
# Given fitted pca object:
# pca.mean_          = μ ∈ R^p  (column means from training data)
# pca.components_    = W^T ∈ R^{d×p}  (d rows, each is a principal axis)
# pca.explained_variance_  = [λ_1, ..., λ_d]  (eigenvalues)

# Forward transform (fit_transform or transform):
Z = (X - pca.mean_) @ pca.components_.T   # shape: (n, d)
# Equivalent to: pca.transform(X)

# If whiten=True, also divides by sqrt(eigenvalues):
# Z_white = Z / np.sqrt(pca.explained_variance_)

# Inverse transform (reconstruction):
X_reconstructed = Z @ pca.components_ + pca.mean_   # shape: (n, p)
# Equivalent to: pca.inverse_transform(Z)

# Reconstruction error:
error = np.mean((X - X_reconstructed) ** 2)
```

### Computational Complexity

| Operation | Exact SVD ('full') | Randomized SVD ('randomized') | Covariance Eigh ('covariance_eigh') |
|---|---|---|---|
| **Fitting** | $O(\min(n,p)^2 \cdot \max(n,p))$ | $O(npd + (n+p)d^2)$ | $O(np^2 + p^3)$ |
| **Transform** | $O(nd)$ | $O(nd)$ | $O(nd)$ |
| **Memory** | $O(np)$ | $O(np)$ | $O(p^2)$ |
| **Best when** | $n, p < 500$ or $d \approx p$ | $d \ll \min(n,p)$ | $n \gg p$, $p < 1000$ |

The `'randomized'` solver is the game-changer for large data. For a $1000 \times 10000$ matrix
keeping $d=50$ components, it reduces fitting from $O(10^{10})$ to $O(10^8)$ operations — a
100× speedup. The new `'covariance_eigh'` solver in sklearn 1.5 exploits the tall-skinny case
($n \gg p$) by working with the $p \times p$ covariance matrix rather than the full $n \times p$
matrix.

### Implicit Assumptions — and Where They Break

PCA encodes these assumptions about your data:

| Assumption | What It Means | When It Breaks |
|---|---|---|
| **Linearity** | Relevant structure lies in a linear subspace | Swiss roll, circles, manifolds — use KPCA or UMAP |
| **Variance = Information** | Directions of high variance contain the signal | When high-variance directions are noise (e.g., sensor drift) |
| **Gaussian-like structure** | Data cluster around a mean, spread ellipsoidally | Multimodal data, long-tailed distributions |
| **Second-order statistics sufficient** | Covariance captures all meaningful structure | When higher-order statistics matter (use ICA for non-Gaussian sources) |
| **Isotropic noise** | All features have equal noise level | When some features are noisier (use Factor Analysis) |
| **No missing values** | All entries of X are observed | When data has missing values (impute first, or use probabilistic PCA) |

> ⚠️ **Pitfall:** The most common misconception is that PCA extracts "the most important features."
> PCA extracts the directions of **maximum variance** — which is not the same as maximum
> discriminability or maximum predictive utility. A feature with high variance could be measurement
> noise. A feature with low variance could be the crucial signal for your downstream task. This
> is why PCA preprocessing doesn't always improve classifier performance — sometimes the
> discriminative structure is in the low-variance dimensions.

### The Probabilistic View (Tipping & Bishop, 1999)

There is a generative model underlying PCA, discovered by Tipping and Bishop in 1999:

$$\mathbf{x}_i = \mathbf{W}\mathbf{z}_i + \boldsymbol{\mu} + \boldsymbol{\epsilon}_i$$

where $\mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ are latent factors, $\mathbf{W} \in \mathbb{R}^{p \times d}$ are loading vectors, and $\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I}_p)$ is **isotropic** noise with equal variance in all directions.

The maximum likelihood estimate of $\mathbf{W}$ gives exactly the PCA solution in the limit
$\sigma^2 \to 0$. This probabilistic interpretation lets you:
- Handle missing data (via EM, not sklearn)
- Compute a principled model selection criterion for $d$
- Use Minka's MLE (`n_components='mle'`) for automatic dimensionality selection

The key assumption is **isotropic noise** ($\sigma^2\mathbf{I}$). When features have different noise
levels, this is wrong, and Factor Analysis (which uses diagonal noise $\boldsymbol{\Psi}$) is
more appropriate.

---

## §3 — The Full Evolution: All Major Variants

PCA has one of the richest variant trees in machine learning. Each variant addresses a concrete
failure mode of classical PCA — too slow, too large, too linear, too dense, too noisy.

### §3.1 — Randomized PCA (Halko, Martinsson, Tropp, 2011)

**Problem with classical PCA:** Full SVD on a large matrix is $O(np\min(n,p))$. For a genomics
dataset with $n=5000$ samples and $p=500,000$ features, this is impractical.

> 📜 **Citation:** Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). "Finding structure with
> randomness: Probabilistic algorithms for constructing approximate matrix decompositions."
> *SIAM Review*, 53(2), 217–288. DOI: 10.1137/090771806. arXiv: 0909.4061

**The key insight:** If we want only the top $d$ principal components, we don't need the full
SVD. We can first compress the matrix to a small sketch that captures the $d$-dimensional range,
then run exact SVD on the tiny sketch.

**Algorithm (Halko et al. 2011, Algorithm 4.3 as implemented in sklearn):**

```
1. Draw Ω ∈ R^{p×(d+l)}  — a Gaussian random matrix (l=n_oversamples, default 10)
2. Compute sketch Y = X̃Ω ∈ R^{n×(d+l)}
3. Power iteration (q times):
     for i in range(q):
         Y = X̃ (X̃^T Y)    # alternatively: Y = (X̃ X̃^T)^q X̃ Ω
         Y, _ = QR(Y)       # re-orthogonalize (QR or LU)
4. QR decompose Y = QR  →  keep Q ∈ R^{n×(d+l)}
5. Form B = Q^T X̃ ∈ R^{(d+l)×p}  (small!)
6. Compute SVD of B: B = Û Σ V^T
7. U = Q Û  →  top-d columns give principal components
```

**Why random projection works:** The Johnson-Lindenstrauss lemma guarantees that a Gaussian
random matrix approximately preserves pairwise distances. The sketch $\mathbf{Y} = \tilde{\mathbf{X}}\boldsymbol{\Omega}$ captures the column space of $\tilde{\mathbf{X}}$ with high probability.
The power iterations $(q = $ `iterated_power`)improve this approximation by amplifying the signal
relative to the noise, which is critical when the singular values decay slowly.

**Error guarantee:** $\|\tilde{\mathbf{X}} - \mathbf{U}_d\boldsymbol{\Sigma}_d\mathbf{V}_d^T\|_F \leq (1+\varepsilon)\|\tilde{\mathbf{X}} - \tilde{\mathbf{X}}_d\|_F$ with high probability, where $\tilde{\mathbf{X}}_d$ is the best rank-$d$ approximation.

**When to prefer:** Any time `n_components ≪ min(n, p)` and `min(n, p) > 500`. The speedup
is roughly $d \cdot (n+p) / (n \cdot p)$ — the smaller $d$ relative to the matrix dimensions, the
larger the speedup. For $d=50$, $n=p=5000$, this is a ~100× speedup.

```python
# Explicit randomized PCA with tuning
from sklearn.decomposition import PCA

pca_rand = PCA(
    n_components=50,
    svd_solver='randomized',
    n_oversamples=10,                    # Extra sketch columns for accuracy
    power_iteration_normalizer='QR',     # Most accurate normalization
    iterated_power=4,                    # Number of power iterations
    random_state=42
)
# Compare to exact:
pca_exact = PCA(n_components=50, svd_solver='full')
# For large matrices, pca_rand is indistinguishable from pca_exact
```

> 🔧 **In practice:** Increase `iterated_power` to 6-7 if you see instability across random
> seeds (components flip signs or reorder with different `random_state`). Increase `n_oversamples`
> to 20-50 for matrices with slowly decaying singular value spectra (data without clear
> low-rank structure).

---

### §3.2 — Incremental PCA (Ross et al., 2008)

**Problem with classical PCA:** The entire dataset must fit in RAM simultaneously, because SVD
requires all data at once.

> 📜 **Citation:** Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008). "Incremental learning
> for robust visual tracking." *International Journal of Computer Vision*, 77(1–3), 125–141.
> DOI: 10.1007/s11263-007-0075-7

**Key algorithmic insight:** You cannot simply run PCA on each batch separately and average the
results — that's mathematically wrong. You must track cumulative statistics: the current SVD
decomposition, the current sample mean (which changes as new batches arrive), and the total
sample count. The Ross et al. paper provides the correct incremental SVD update formulas that
sklearn's `IncrementalPCA` implements.

**The mean update problem:** Suppose after $n_1$ samples you have mean $\boldsymbol{\mu}_1$. After
seeing $n_2$ new samples with mean $\boldsymbol{\mu}_2$, the combined mean is:

$$\boldsymbol{\mu}_{1+2} = \frac{n_1 \boldsymbol{\mu}_1 + n_2 \boldsymbol{\mu}_2}{n_1 + n_2}$$

This mean correction must be applied to the existing SVD before merging the new batch. Failing
to do this correctly produces wrong principal components.

**When to prefer:**
- Dataset is too large for RAM (multi-GB CSV files, genomic databases)
- Data arrives as a stream (financial tick data, sensor feeds, video frames)
- Online learning scenarios where PCA must update as new data arrives

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

# Scenario: 1M samples, can only load 10K at a time
ipca = IncrementalPCA(n_components=50)

# Method 1: Manual batching
for start in range(0, len(X_full), 10_000):
    batch = X_full[start:start + 10_000]
    ipca.partial_fit(batch)

print(f"Samples seen: {ipca.n_samples_seen_}")
print(f"Cumulative variance: {ipca.explained_variance_ratio_.sum():.3f}")

# Method 2: Let sklearn handle batching (in-core but memory-conscious)
ipca2 = IncrementalPCA(n_components=50, batch_size=10_000)
ipca2.fit(X_full)  # Internally calls partial_fit in chunks

# Transform in chunks to avoid memory spike
X_reduced = []
for start in range(0, len(X_full), 10_000):
    X_reduced.append(ipca.transform(X_full[start:start + 10_000]))
X_reduced = np.vstack(X_reduced)
```

> ⚠️ **Pitfall:** `n_components` must be less than or equal to the batch size. If you call
> `partial_fit` with batches of 100 samples and set `n_components=200`, sklearn will raise a
> `ValueError`. Set `batch_size ≥ n_components` always.

---

### §3.3 — Kernel PCA (Schölkopf, Smola, Müller, 1998)

**Problem with classical PCA:** PCA can only find linear structure. If your data lies on a
nonlinear manifold (a circle, a Swiss roll, concentric rings), PCA cannot separate it.

> 📜 **Citation:** Schölkopf, B., Smola, A., & Müller, K.-R. (1998). "Nonlinear component
> analysis as a kernel eigenvalue problem." *Neural Computation*, 10(5), 1299–1319.
> DOI: 10.1162/089976698300017467

**The kernel trick:** Instead of working in the original space $\mathbb{R}^p$, implicitly map
data to a high-dimensional (possibly infinite-dimensional) feature space $\mathcal{H}$ via
$\phi: \mathbb{R}^p \to \mathcal{H}$. Run PCA in $\mathcal{H}$. Because we only need inner
products $\langle\phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle$, we never need to compute
$\phi$ explicitly — we just evaluate $k(\mathbf{x}_i, \mathbf{x}_j) = \langle\phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle$.

**The algorithm:**

```
1. Compute kernel matrix: K ∈ R^{n×n},  K_{ij} = k(x_i, x_j)
2. Center K: K̃ = K - 1_n K - K 1_n + 1_n K 1_n   (where 1_n = (1/n) 11^T)
3. Eigendecompose K̃ = V Λ V^T  (Λ diagonal, λ₁ ≥ λ₂ ≥ ... ≥ 0)
4. Normalize: α_k = V_k / √λ_k  (columns of coefficient matrix A)
5. Project new point x*: z_k = Σᵢ α_{ik} k(xᵢ, x*)
```

**Key limitation:** The kernel matrix $\mathbf{K} \in \mathbb{R}^{n \times n}$ must be computed
and stored. For $n=10,000$, this is 800MB. For $n=100,000$, it's 80GB — impractical.

**Common kernels and their implicit feature spaces:**

| Kernel | Formula | Implicit Space | Good For |
|---|---|---|---|
| Linear | $\mathbf{x}_i^T\mathbf{x}_j$ | Same as input | Baseline (= linear PCA) |
| RBF/Gaussian | $\exp(-\gamma\|\mathbf{x}_i-\mathbf{x}_j\|^2)$ | Infinite-dimensional | General nonlinear structure |
| Polynomial | $(\mathbf{x}_i^T\mathbf{x}_j + c_0)^d$ | Degree-$d$ polynomial features | Polynomial interactions |
| Cosine | $\frac{\mathbf{x}_i^T\mathbf{x}_j}{\|\mathbf{x}_i\|\|\mathbf{x}_j\|}$ | Normalized sphere | Text, TF-IDF features |

```python
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# RBF KPCA for nonlinear structure
kpca = KernelPCA(
    n_components=20,
    kernel='rbf',
    gamma=0.05,                 # Critical: tune with cross-validation
    fit_inverse_transform=True, # Enables reconstruction error computation
    eigen_solver='randomized',  # Fast for large n
    random_state=42,
    n_jobs=-1
)
X_kpca = kpca.fit_transform(X_scaled)

# Reconstruction error (approximate for nonlinear kernels)
X_reconstructed = kpca.inverse_transform(X_kpca)
recon_error = np.mean((X_scaled - X_reconstructed) ** 2)
```

---

### §3.4 — Sparse PCA (Zou, Hastie, Tibshirani, 2006)

**Problem with classical PCA:** Each principal component loads on all $p$ features simultaneously,
making interpretation difficult. For gene expression data with 20,000 genes, PC1 has nonzero
loading for all 20,000 genes — what does it mean biologically?

> 📜 **Citation:** Zou, H., Hastie, T., & Tibshirani, R. (2006). "Sparse principal component
> analysis." *Journal of Computational and Graphical Statistics*, 15(2), 265–286.
> DOI: 10.1198/106186006X113430

**The key innovation:** Reformulate PCA as a regression problem, then add L1 regularization
(LASSO) to force loading vectors to be sparse — loading on only a few features per component.

**Algorithm (dictionary learning):**

$$\min_{\mathbf{A}, \mathbf{B}} \|\tilde{\mathbf{X}} - \tilde{\mathbf{X}}\mathbf{B}\mathbf{A}^T\|_F^2 + \alpha\sum_j\|\mathbf{b}_j\|_1$$

subject to $\mathbf{A}^T\mathbf{A} = \mathbf{I}$. Iterate:
- Fix $\mathbf{A}$ (current loadings), solve for sparse $\mathbf{B}$ via LASSO/LARS
- Fix $\mathbf{B}$ (scores), solve for orthonormal $\mathbf{A}$ via SVD

**Important property:** Unlike PCA, Sparse PCA components are **NOT orthogonal**. You gain
interpretability at the cost of orthogonality.

```python
from sklearn.decomposition import SparsePCA, MiniBatchSparsePCA

# SparsePCA: exact but slow
spca = SparsePCA(
    n_components=20,
    alpha=0.5,          # L1 sparsity strength — key hyperparameter
    ridge_alpha=0.01,   # Ridge for the transform step
    method='lars',      # 'lars' faster for high sparsity; 'cd' more stable
    n_jobs=-1,
    random_state=42
)
X_sparse_pca = spca.fit_transform(X_scaled)

# Check sparsity
sparsity = np.mean(spca.components_ == 0)
print(f"Sparsity of components: {sparsity:.2%}")  # Aim for 70-95%

# MiniBatchSparsePCA: faster approximation for larger datasets
mb_spca = MiniBatchSparsePCA(
    n_components=20,
    alpha=0.5,
    max_iter=500,
    tol=1e-3,
    max_no_improvement=10,   # Early stopping
    random_state=42
)
```

> ⚠️ **Pitfall:** SparsePCA is 100–1000× slower than regular PCA. For $n=1000$, $p=500$,
> $d=20$, expect 30–120 seconds. Always prototype with `MiniBatchSparsePCA` first.

---

### §3.5 — Factor Analysis: The Probabilistic Generalization

**Problem with classical PCA:** PCA assumes isotropic noise — all features have the same noise
level. In practice, some measurements (e.g., some EEG channels, some sensors) are noisier
than others.

Factor Analysis (FA) generalizes probabilistic PCA by allowing per-feature noise:

$$\mathbf{x}_i = \mathbf{W}\mathbf{z}_i + \boldsymbol{\mu} + \boldsymbol{\epsilon}_i, \quad \boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi})$$

where $\boldsymbol{\Psi} = \text{diag}(\psi_1, \ldots, \psi_p)$ is a **diagonal** matrix of
per-feature noise variances estimated from data. PCA is the special case $\boldsymbol{\Psi} = \sigma^2\mathbf{I}$.

FA is fit via EM (expectation-maximization), optimizing the marginal log-likelihood
$\log p(\mathbf{X})$ rather than variance explained.

```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(
    n_components=10,
    rotation='varimax',  # Post-fit rotation for interpretability
    max_iter=1000,
    tol=0.01,
    random_state=0       # Note: default is 0, not None
)
X_fa = fa.fit_transform(X_scaled)

# Per-feature noise estimates:
print(fa.noise_variance_)  # shape (n_features,) — which features are noisy?

# Model selection via cross-validated log-likelihood (FA has a proper likelihood)
from sklearn.model_selection import cross_val_score
fa_scores = []
for n in range(1, 20):
    fa_n = FactorAnalysis(n_components=n, random_state=0)
    fa_scores.append(cross_val_score(fa_n, X_scaled, cv=5).mean())
best_n = np.argmax(fa_scores) + 1
print(f"Best n_components by FA log-likelihood: {best_n}")
```

**When FA beats PCA:**
- Features have heterogeneous noise (sensors, clinical measurements)
- You need a proper generative model with likelihood (for model selection, hypothesis testing)
- You want varimax rotation for psychometric interpretation (simple structure)

---

### §3.6 — Truncated SVD / Latent Semantic Analysis

**Problem:** You have a sparse document-term matrix with hundreds of thousands of terms. PCA's
centering step destroys sparsity, blowing up memory from gigabytes to terabytes.

`TruncatedSVD` computes the top-$d$ SVD of $\mathbf{X}$ **without centering**. For text data,
this is LSA (Latent Semantic Analysis). For genomics count data, it enables PCA-like compression
on sparse count matrices.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# LSA pipeline for text
lsa = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50_000, sublinear_tf=True)),
    ('svd',   TruncatedSVD(n_components=200, random_state=42))
])
X_lsa = lsa.fit_transform(documents)
# NOTE: TruncatedSVD does NOT expose explained_variance_ratio_ by default for sparse
# Use: tsvd.explained_variance_ratio_.sum() after fitting
```

> 🔧 **In practice:** The rule is simple — if your data matrix is sparse (scipy.sparse), use
> `TruncatedSVD`. If it is dense, use `PCA`. Never use `PCA` on a sparse matrix: it materializes
> a dense centered copy and runs out of memory.

---

## §4 — Hyperparameters: The Complete Guide

### 4.1 `n_components` — The Master Hyperparameter

This is the single most consequential decision you make when fitting PCA. Every other parameter
is second-order compared to this one.

**What it controls:** The dimensionality of the output space — how many principal components you
retain. All information not captured by the top-$d$ components is discarded during the forward
transform.

**The three modes:**
- `int` (e.g., `n_components=50`): Keep exactly 50 components.
- `float` in $(0, 1)$ (e.g., `n_components=0.95`): Keep enough components to explain 95% of
  total variance. Only works with `svd_solver='full'`.
- `'mle'`: Use Minka's Bayesian MLE to automatically select dimensionality. Only with `svd_solver='full'`.

**Selection strategies (in priority order for new problems):**

**Strategy 1 — Cumulative Variance Threshold (most common):**

```python
pca_full = PCA(svd_solver='full').fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# Find thresholds
for threshold in [0.80, 0.90, 0.95, 0.99]:
    n = np.searchsorted(cumvar, threshold) + 1
    print(f"{threshold*100:.0f}% variance: {n} components")
```

Standard thresholds: 80% (exploratory), 90% (balanced), 95% (conservative), 99% (near-lossless).

**Strategy 2 — Scree Plot Elbow:** Plot `explained_variance_ratio_` per component and find
the "elbow" where the curve flattens. The components to the left of the elbow capture signal;
those to the right capture noise. Subjective but robust for datasets with clear structure.

**Strategy 3 — Minka's MLE:** Bayesian model selection. Best for moderate dimensions
($p < 1000$) and when you have no prior on $d$.

```python
pca_mle = PCA(n_components='mle', svd_solver='full')
pca_mle.fit(X_scaled)
print(f"MLE selected: {pca_mle.n_components_} components")
```

**Strategy 4 — Kaiser Rule:** Retain components with eigenvalue > 1.0 (when data is
standardized). A classical heuristic from factor analysis that breaks down in high dimensions
where many eigenvalues cluster near 1.

**Strategy 5 — Downstream Task Cross-Validation (most principled for supervised tasks):**

```python
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def objective(trial):
    n_components = trial.suggest_int('n_components', 2, min(X.shape) // 2)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('clf', LogisticRegression(max_iter=500, random_state=42))
    ])
    return cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f"Best n_components: {study.best_params['n_components']}")
```

**"Too high" symptoms:** Reconstruction error stops decreasing; downstream model performance
plateaus or degrades (overfitting on noise dimensions); eigenvalues are very small and equal.

**"Too low" symptoms:** High reconstruction error; poor downstream performance; classes overlap
in the 2D visualization.

---

### 4.2 `svd_solver` — Choosing the Right Algorithm

| Solver | When to Use | Key Trade-offs |
|---|---|---|
| `'auto'` | Default; let sklearn decide | Best first choice; calibrated heuristic |
| `'full'` | Small data; need exact results; `n_components='mle'` or float | Most accurate; slowest for large data |
| `'covariance_eigh'` | $n \gg p$ (e.g., 100K samples, 50 features) | Fastest in tall-skinny case; **new in sklearn 1.5** |
| `'arpack'` | Sparse input; very few components ($d < 10$) | Iterative; good for very small $d$ |
| `'randomized'` | Large dense data; $d \ll \min(n,p)$ | 10–100× speedup; slight approximation error |

The `'auto'` policy (from sklearn 1.5 source):
1. If `n_components='mle'` or float → `'full'`
2. If both $n$ and $p$ > 500 AND `n_components` < 80% of min(n,p) → `'randomized'`
3. Otherwise → `'full'`

> 🔧 **In practice:** Trust `'auto'` for most cases. Force `'randomized'` explicitly when you
> want control. Force `'covariance_eigh'` when $n \gg p$ and you've timed the other solvers.

---

### 4.3 `iterated_power` and `n_oversamples` (Randomized Solver Only)

These two parameters control the accuracy-speed trade-off of the randomized SVD.

**`iterated_power` (default `'auto'`):**
- Number of power iterations $q$ in the Halko et al. algorithm.
- Each iteration: $\mathbf{Y} \leftarrow \tilde{\mathbf{X}}(\tilde{\mathbf{X}}^T\mathbf{Y})$, amplifying the singular value signal.
- More iterations = higher accuracy, but $O(q \cdot np)$ additional work.
- Default `'auto'`: sklearn sets $q=4$ for `n_components/min(n,p) < 0.1`, else $q=2$.
- Increase to 6–7 if you see instability across random seeds.
- Rarely need $q > 7$ — diminishing returns after convergence.

**`n_oversamples` (default 10):**
- Extra random columns in the sketch beyond the target rank: sketch size = $d + l$.
- More oversampling = lower approximation error.
- The default of 10 is sufficient for matrices with fast-decaying singular values (clear structure).
- Increase to 20–50 for matrices with slow decay (noisy, diffuse structure).
- Very rarely helps to exceed 50.

**`power_iteration_normalizer` (default `'auto'`):**
- `'QR'`: Most accurate; re-orthogonalizes via QR at each power step.
- `'LU'`: Faster than QR; slightly less accurate.
- `'none'`: No stabilization; can suffer numerical errors for many iterations.
- Default `'auto'`: uses `'LU'` for `iterated_power < 5`, else `'QR'`.

---

### 4.4 `whiten` — When Scale Information Matters

**Default:** `False`

When `whiten=True`, the projected components are divided by their singular values:
$$\mathbf{Z}_{\text{white}} = \mathbf{Z} / \sqrt{\boldsymbol{\Lambda}}$$

This produces components with **unit variance each**, removing the information about relative
importance (PC1 is no longer larger than PC2 in variance).

| Setting | Use When | Effect |
|---|---|---|
| `whiten=False` | Linear models, preserving relative importance | PC1 variance > PC2 variance > ... |
| `whiten=True` | Distance-based models (SVM-RBF, k-NN, k-means) | All components have variance 1 |

> 💡 **Intuition:** SVM with an RBF kernel computes $\exp(-\gamma\|\mathbf{z}_i - \mathbf{z}_j\|^2)$.
> Without whitening, PC1 (high variance) dominates this distance; PC10 (low variance) contributes
> almost nothing. Whitening equalizes the contribution of all retained components, making the
> RBF kernel respect all dimensions equally.

---

### 4.5 KernelPCA: `kernel` and `gamma`

**`kernel` selection:**

```python
# Test linear kernel first (should match sklearn.PCA)
kpca_lin = KernelPCA(kernel='linear', n_components=d)
# If linear gives good results, PCA is sufficient — don't use KPCA

# RBF for general nonlinear structure
kpca_rbf = KernelPCA(kernel='rbf', gamma=gamma_val, n_components=d)

# Polynomial for interaction features
kpca_poly = KernelPCA(kernel='poly', degree=3, coef0=1.0, n_components=d)

# Cosine for text/normalized features
kpca_cos = KernelPCA(kernel='cosine', n_components=d)
```

**`gamma` (for RBF kernel) — the most important KPCA hyperparameter:**

- Default: `1 / n_features`. Often a good starting point.
- Too high: Kernel is very peaked; only very nearby points interact; tends to overfit.
- Too low: Kernel is flat; behaves nearly like linear PCA.
- **Best initialization:** Median heuristic:

```python
from sklearn.metrics import pairwise_distances
import numpy as np

# On a random subsample for speed
X_sample = X_scaled[:500]
D = pairwise_distances(X_sample)
gamma_init = 1 / (2 * np.median(D[D > 0]) ** 2)
print(f"Median-heuristic gamma: {gamma_init:.4f}")

# Then use Optuna to tune around this value:
import optuna
def kpca_objective(trial):
    gamma = trial.suggest_float('gamma', gamma_init / 10, gamma_init * 10, log=True)
    n_comp = trial.suggest_int('n_components', 5, 50)
    kpca = KernelPCA(kernel='rbf', gamma=gamma, n_components=n_comp,
                     fit_inverse_transform=True, n_jobs=-1)
    from sklearn.linear_model import LogisticRegression
    pipe = Pipeline([('kpca', kpca), ('clf', LogisticRegression(max_iter=500))])
    return cross_val_score(pipe, X_scaled, y, cv=5).mean()
```

---

### 4.6 SparsePCA: `alpha`

`alpha` controls sparsity strength (L1 regularization on loading vectors):

| `alpha` | Component behavior | Use when |
|---|---|---|
| Very low (0.01) | Dense loadings, like PCA | Baseline comparison |
| Moderate (0.5–2) | 70–90% zero loadings | Gene expression, factor interpretation |
| High (10+) | Nearly all loadings zero | Rarely useful; too sparse |

**Diagnostic:** `np.mean(spca.components_ == 0)` — target 70–95% for interpretable components.

---

### §4 Tuning Playbook

| Parameter | Estimator | Default | Typical Search Space | Priority |
|---|---|---|---|---|
| `n_components` | All | `None` (all) | `[2, min(n,p)//2]` | **CRITICAL** |
| `svd_solver` | PCA | `'auto'` | `{'auto','randomized','full','covariance_eigh'}` | High |
| `iterated_power` | PCA (randomized) | `'auto'` | `{2, 4, 6, 7}` | Medium |
| `n_oversamples` | PCA (randomized) | `10` | `[10, 50]` | Low |
| `whiten` | PCA, IPCA | `False` | `{True, False}` | Medium |
| `kernel` | KernelPCA | `'linear'` | `{'linear','rbf','poly','cosine'}` | **HIGH** |
| `gamma` | KernelPCA (rbf) | `1/p` | `[1e-4, 10]` log-scale | **HIGH** |
| `degree` | KernelPCA (poly) | `3` | `[2, 5]` | Medium |
| `alpha` | SparsePCA | `1` | `[0.01, 10]` log-scale | **HIGH** |
| `batch_size` | IPCA | `5*n_features` | `[n_components, n]` | Medium |
| `rotation` | FactorAnalysis | `None` | `{None, 'varimax', 'quartimax'}` | Medium |

---

## §5 — Strengths

**1. Optimality: Best Linear Compression**

PCA is not just a good linear compression method — it is the *provably best* one. For any choice
of $d < p$, PCA finds the $d$-dimensional linear subspace that minimizes the mean squared
reconstruction error. No other linear projection can do better. This optimality is why PCA
remains the benchmark that all other methods are compared to.

**2. Computational Efficiency**

For dense data of moderate size, PCA via randomized SVD is extremely fast. A $10000 \times 1000$
matrix with $d=50$ components takes under 1 second on a modern CPU. The randomized algorithm
scales nearly linearly with $d$, making it tractable even for large inputs.

**3. Deterministic on Clean Data**

When using `svd_solver='full'` or `'covariance_eigh'`, PCA is fully deterministic (up to sign
flips). Given the same data, the same components are always returned. This reproducibility is
important in production systems where model drift must be tracked.

**4. Interpretable Loadings**

The principal axes (`pca.components_`, shape `(d, p)`) are vectors in the original feature
space. Each row is a loading vector that tells you how much each original feature contributes
to that principal component. You can plot these as images (eigenfaces), bar charts (gene
expression), or maps (geographic features), making PCA deeply interpretable for domain experts.

**5. Decorrelates Features**

After PCA transformation, the output features are **uncorrelated by construction** (they
are projections onto orthogonal axes). This is valuable for downstream algorithms that assume
uncorrelated features (e.g., Naive Bayes, diagonal covariance Gaussian models, factor analysis).

**6. Noise Reduction**

Discarding the last $p - d$ components removes the low-variance directions, which often
correspond to measurement noise. For image data, this produces visually cleaner reconstructions.
For scientific data, it separates signal (large eigenvalues) from noise floor (small, equal
eigenvalues). The `noise_variance_` attribute of PCA estimates the average noise level in the
discarded dimensions.

**7. Variance Explained is Measurable**

Unlike most other DR algorithms, PCA gives you a direct, interpretable metric for how much
information you retained: `explained_variance_ratio_`. You can say "I retained 92% of the
variance in 40 dimensions." No other standard DR algorithm provides such a clean measure.

**8. Pipeline Composability**

PCA slots into sklearn pipelines without any special handling. It's invertible
(`inverse_transform`), supports `fit`/`transform`/`fit_transform`, and works with
`GridSearchCV`, `cross_val_score`, and all sklearn utilities out of the box.

---

## §6 — Weaknesses & Failure Modes

**1. Linearity: Cannot Capture Manifold Structure**

*Mechanism:* PCA finds the best linear subspace. If data lies on a nonlinear manifold (a circle,
a Swiss roll, a sphere), the linear subspace that PCA finds is a chord through the manifold —
it collapses what should be distinct parts of the manifold into the same region.

*Example:* Two concentric rings in 2D. PCA sees the combined cloud as a disk and projects both
rings onto a single 1D axis, merging them.

*Detection:* After 2D PCA, points from different classes (or different parts of a known manifold)
overlap in the scatter plot.

*Mitigation:* Use Kernel PCA (RBF kernel), UMAP, or t-SNE for nonlinear structure.

**2. Variance ≠ Relevance**

*Mechanism:* PCA maximizes variance, but the most predictively useful dimensions may not be the
highest-variance ones. A low-variance feature (e.g., a rare but perfectly diagnostic biomarker)
will be discarded.

*Example:* In medical imaging, the overall brightness (high variance, unrelated to pathology)
dominates PC1, while the subtle intensity pattern of a lesion (low variance, highly diagnostic)
is buried in PC10+.

*Detection:* Downstream classifier performance improves when you use more components than the
"natural" elbow suggests, or when LDA on the same data gives much higher accuracy than
PCA + classifier.

*Mitigation:* Use supervised dimensionality reduction (LDA, UMAP with labels) when labels are
available. Or increase `n_components` and let the downstream model select relevant features.

**3. Outlier Sensitivity**

*Mechanism:* The covariance matrix is sensitive to outliers. One extreme outlier can dominate
PC1, pulling it toward the outlier direction and contaminating all downstream analyses.

*Example:* A single bad sensor reading produces a value 100× larger than normal. PC1 now points
toward this outlier.

*Detection:* PC1 loadings concentrate on a single feature; the explained variance of PC1 is
disproportionately large; removing one sample dramatically changes PC1.

*Mitigation:* Preprocess with `RobustScaler` (uses median and IQR, not mean and std). Winsorize
extreme values. Use Robust PCA (e.g., `sklearn.decomposition.FactorAnalysis` with trimmed input,
or specialized robust PCA libraries).

> ⚠️ **Pitfall:** `StandardScaler` does NOT protect against outliers. It uses mean and std,
> which are themselves influenced by outliers. Use `RobustScaler` or clip extreme values before
> fitting PCA.

**4. Requires Standardization When Features Have Different Units**

*Mechanism:* PCA maximizes variance. Features measured in large units (income: 10K–200K) have
much larger variance than features in small units (age: 20–80). Without standardization, PCA
finds "directions of large-unit-variation," not meaningful patterns.

*Example:* Income (variance ~$10^9$) and age (variance ~$400$) in the same dataset. Without
`StandardScaler`, PC1 is entirely determined by income. Age is invisible.

*Detection:* `explained_variance_ratio_[0]` is very close to 1.0; loading vectors concentrate
on high-variance features.

*Mitigation:* Always `StandardScaler().fit_transform(X)` before PCA when features have
different units or scales. Exception: when all features are inherently comparable (pixel values,
percentage returns, standardized gene expression).

**5. Memory Constraints for KernelPCA**

*Mechanism:* KPCA requires computing and storing the full $n \times n$ kernel matrix.

*Threshold:* For $n=10,000$, $\mathbf{K}$ is 800MB. For $n=100,000$, it is 80GB.

*Mitigation:* Use `eigen_solver='randomized'` + a small `n_components`. For very large $n$,
use Nyström approximation or approximate KPCA methods.

**6. Cannot Handle Missing Data Natively**

*Mechanism:* SVD requires complete matrices. `NaN` values cause immediate failure.

*Mitigation:* Impute with `SimpleImputer` before PCA. For principled handling, use
probabilistic PCA with EM (not directly in sklearn; available in `pyro` or `tensorflow-probability`).

**7. Sign and Rotation Ambiguity**

*Mechanism:* A principal component $\mathbf{w}_k$ and its negative $-\mathbf{w}_k$ are
mathematically equivalent (they span the same 1D subspace). sklearn uses `svd_flip` to
enforce a convention (largest absolute element is positive), but the sign can vary across
sklearn versions or if you switch solvers.

*Effect:* Code that checks the sign of a loading vector will break silently across runs or
environments.

*Mitigation:* Never rely on the absolute sign of a loading vector. When comparing PCA results
across experiments, align signs explicitly using `svd_flip` or by checking the correlation
between corresponding components.

**8. PCA on Categorical or Mixed Data is Wrong**

*Mechanism:* PCA on binary (0/1) or ordinal features computes a Euclidean covariance that
has no geometric meaning for such variables.

*Mitigation:* Use `prince.MCA` for purely categorical data, `prince.FAMD` for mixed data.

---

## §7 — What I Think About When Fitting

This section is my internal monologue. Not the formal checklist — the real one, the one that
runs in my head every time I open a dataset and consider PCA.

### Before I Touch `fit()`

**What is the data type?** I look at my feature types immediately. If I see a mix of numeric
and categorical columns, PCA is wrong. I stop and reach for `prince.FAMD`. If it's a sparse
matrix (TF-IDF, one-hot, count data), I use `TruncatedSVD`. If it's all numeric and dense,
I continue.

**Does the data need scaling?** I check `X.describe()` and look at the range of each feature.
If the ranges differ by more than a factor of 10 across features, I always use `StandardScaler`.
No exceptions. The one exception: if all features are genuinely on the same scale (all pixel
intensities 0–255, all log-expression values, all z-scored already), I might skip scaling or
use just centering.

**How many samples vs. features?** If $n \gg p$ (tall-skinny), I note that `svd_solver='covariance_eigh'`
in sklearn 1.5 will be fastest. If $p > n$ (wide matrices), I know randomized SVD is essential.
If both are large, I check whether the data fits in RAM — if not, I switch to `IncrementalPCA`
before anything else.

**Are there missing values?** `X.isna().sum().sum()`. If nonzero, I stop and decide on imputation
strategy. For small fractions of missing data, `SimpleImputer(strategy='mean')`. For larger
fractions, `IterativeImputer`. For structural missingness (MCAR/MAR/MNAR), I think carefully.

**Are there outliers?** I quickly check `X.describe()` and plot a couple of box plots. If the
max/min are far from the 99th/1st percentile, I either winsorize or use `RobustScaler`. This
is the step most people skip, and it's the step that most often produces garbage PC1.

**Data leakage check:** I confirm I have a train/test split and that PCA will be fit *only*
on training data. If there's any cross-validation coming, I wrap everything in a `Pipeline`.
Non-negotiable.

### During and Immediately After `fit()`

**First thing I check:** `explained_variance_ratio_`. I look at the full spectrum:
- If PC1 explains > 50%: strong low-rank structure, PCA is appropriate
- If the first 10 PCs all explain ~equal variance: data is approximately spherical; PCA may
  not compress well
- If there's a clear elbow: the number of components before the elbow is my starting point

**The cumulative curve:** I set my threshold — usually 90% or 95% — and note how many
components it takes to get there. If it takes more than $\min(n, p) / 2$ components, I ask
whether PCA is the right tool. Maybe the data is fundamentally high-dimensional.

**Do components look interpretable?** I glance at `pca.components_[0]` — the loadings for PC1.
If it's a roughly uniform vector (all loadings similar magnitude), it often means "overall
level" or "average" — which is usually uninteresting. If some features dominate, I inspect
which ones and whether that makes domain sense.

**Sign check:** I never rely on the sign of a loading. But I note whether PC1 loadings are
mostly positive or mostly negative. If I'm comparing against a previous run, I check whether
the sign flipped (which is fine) before concluding that the model "changed."

**Reconstruction sanity check:** I always compute reconstruction error on a held-out sample:

```python
X_val_pca = pca.transform(X_val_scaled)
X_val_recon = pca.inverse_transform(X_val_pca)
val_error = np.mean((X_val_scaled - X_val_recon) ** 2)
total_var = np.var(X_val_scaled)
print(f"Relative reconstruction error: {val_error / total_var:.3f}")
# Should be ≈ 1 - cumulative_explained_variance_ratio
```

If the reconstruction error is much higher than `1 - sum(explained_variance_ratio_)`, the test
distribution has shifted from training — a red flag.

### Diagnosing Problems

**Scenario: PC1 explains 95% of variance**

Something is wrong. Either one feature has huge absolute scale (forgot to standardize), or
there is a near-constant vector in the data (batch effect, experimental artifact). Inspect
`pca.components_[0]` — which features load heavily?

**Scenario: All eigenvalues are equal**

Your data is isotropic (spherical Gaussian noise). PCA finds nothing. This is either a
preprocessing problem (you've already whitened the data elsewhere), a feature engineering
problem (your features are random noise), or a sign that you need a nonlinear method.

**Scenario: Downstream classifier is worse with PCA than without**

Either $d$ is too small (you discarded discriminative dimensions), or PCA is the wrong tool
(the discriminative structure is nonlinear or low-variance). Try:
1. Increase `n_components` significantly (to 95% variance)
2. Try `whiten=True`
3. Try supervised DR: LDA, UMAP with `y=labels`

**Scenario: Randomized PCA gives different components on each run**

Increase `iterated_power` (try 6–7). Set a fixed `random_state`. If components still differ,
the matrix doesn't have a clean low-rank structure — the "true" components aren't well-separated.
This is data, not a bug.

**Scenario: `partial_fit` raises ValueError**

Almost always: `n_components > batch_size`. Check your batch size and ensure
`batch_size ≥ n_components`.

---

## §8 — Diagnostic Plots & Evaluation

A fitted PCA model is not validated until you've run the right diagnostics. Here are the
essential plots and evaluation metrics, with complete code for each.

### Plot 1: Scree Plot + Cumulative Variance

The first plot you always make. Shows how variance is distributed across components and where
to set `n_components`.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

digits = load_digits()
X_scaled = StandardScaler().fit_transform(digits.data)

pca_full = PCA(svd_solver='full').fit(X_scaled)
evr = pca_full.explained_variance_ratio_
cumevr = np.cumsum(evr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot (individual variance)
ax1.bar(range(1, 21), evr[:20], color='steelblue', alpha=0.7)
ax1.axhline(y=evr[0]/10, color='red', linestyle='--', alpha=0.5, label='Noise floor heuristic')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Scree Plot — Individual Variance')
ax1.legend()

# Cumulative variance
thresholds = [0.80, 0.90, 0.95, 0.99]
ax2.plot(range(1, len(cumevr) + 1), cumevr, 'b-o', markersize=3, label='Cumulative variance')
for t in thresholds:
    n = np.searchsorted(cumevr, t) + 1
    ax2.axhline(t, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(n, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(f'{t*100:.0f}%: {n} PCs', xy=(n, t),
                 xytext=(n + 2, t - 0.04), fontsize=9)
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Variance Curve')
ax2.set_ylim(0, 1.05)
ax2.legend()

plt.tight_layout()
plt.savefig('scree_cumvar.png', dpi=150)
plt.show()
```

**Reading the plot:** A clear elbow in the scree plot identifies the number of "signal"
components. A flat tail (many components with equal small variance) is the noise floor.
In the cumulative plot, choose the threshold where the curve starts to flatten — this is
often more informative than the elbow.

---

### Plot 2: 2D PCA Scatter (Colored by Label)

The sanity check for any labeled dataset: do the classes separate in the first two PCs?

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

df_plot = pd.DataFrame({
    'PC1': X_2d[:, 0],
    'PC2': X_2d[:, 1],
    'Digit': digits.target.astype(str)
})

fig, ax = plt.subplots(figsize=(10, 8))
palette = sns.color_palette('tab10', 10)
for i, digit in enumerate(sorted(df_plot['Digit'].unique())):
    mask = df_plot['Digit'] == digit
    ax.scatter(df_plot.loc[mask, 'PC1'], df_plot.loc[mask, 'PC2'],
               c=[palette[i]], label=f'Digit {digit}', alpha=0.6, s=20)

ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)')
ax.set_title('Digits Dataset — 2D PCA Projection')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.show()
```

**Reading the plot:** Well-separated clusters → PCA captures discriminative structure. Severe
overlap → either use more components (for modeling, not visualization) or try a nonlinear
method (UMAP/t-SNE) for visualization.

---

### Plot 3: Loading Heatmap (Biplot-Style)

Understand what each PC means in terms of original features.

```python
import seaborn as sns
import pandas as pd

pca_n = PCA(n_components=10, random_state=42)
pca_n.fit(X_scaled)

# Loadings heatmap
loadings = pd.DataFrame(
    pca_n.components_,
    columns=[f'Pixel {i}' for i in range(X_scaled.shape[1])],
    index=[f'PC{i+1}' for i in range(10)]
)

fig, ax = plt.subplots(figsize=(16, 4))
sns.heatmap(loadings, cmap='RdBu_r', center=0, ax=ax,
            xticklabels=False, yticklabels=True)
ax.set_title('PCA Loading Heatmap — Which pixels define each PC?')
ax.set_xlabel('Pixel Features')
ax.set_ylabel('Principal Components')
plt.tight_layout()
plt.show()
```

**Reading the plot:** Red regions = features with positive loading (PC increases when these
increase). Blue = negative loading. Concentrated red/blue patterns → interpretable PC.
Uniform low-level color across all features → PC is a noise-averaging component.

---

### Plot 4: Reconstruction Error vs. n_components

Shows the information loss curve, helping you choose `n_components`.

```python
n_range = range(1, min(X_scaled.shape) + 1, 2)
errors = []
for n in n_range:
    pca_n = PCA(n_components=n, svd_solver='full', random_state=42)
    X_rec = pca_n.inverse_transform(pca_n.fit_transform(X_scaled))
    errors.append(np.mean((X_scaled - X_rec) ** 2))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(n_range), errors, 'b-', linewidth=1.5)
ax.set_xlabel('n_components')
ax.set_ylabel('Mean Squared Reconstruction Error')
ax.set_title('Reconstruction Error vs. Number of Components')

# Mark common thresholds
for t, style in [(0.90, '--'), (0.95, '-.')]:
    n_t = np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), t) + 1
    ax.axvline(n_t, color='red', linestyle=style, alpha=0.7,
               label=f'{t*100:.0f}% variance: {n_t} PCs')
ax.legend()
plt.tight_layout()
plt.show()
```

---

### Plot 5: Trustworthiness vs. n_components

For each n_components choice, how well are local neighborhoods preserved?

```python
from sklearn.manifold import trustworthiness

n_range_tw = [5, 10, 20, 30, 40, 50]
tw_scores = []
for n in n_range_tw:
    pca_n = PCA(n_components=n, random_state=42)
    X_red = pca_n.fit_transform(X_scaled)
    tw_scores.append(trustworthiness(X_scaled, X_red, n_neighbors=10))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_range_tw, tw_scores, 'g-o')
ax.axhline(0.95, color='red', linestyle='--', label='0.95 threshold')
ax.set_xlabel('n_components')
ax.set_ylabel('Trustworthiness (k=10)')
ax.set_title('Neighborhood Preservation vs. n_components')
ax.legend()
plt.tight_layout()
plt.show()
```

**Reading the plot:** Trustworthiness > 0.95 is excellent. For linear PCA on well-structured
data, trustworthiness typically rises quickly and plateaus. If trustworthiness stays low even
with many components, the data has nonlinear structure.

---

### Evaluation Metrics Summary

```python
from sklearn.manifold import trustworthiness

def evaluate_pca(pca, X_original, X_scaled):
    """Comprehensive PCA evaluation."""
    X_reduced = pca.transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)

    # 1. Reconstruction error
    recon_error = np.mean((X_scaled - X_reconstructed) ** 2)
    relative_error = 1 - pca.explained_variance_ratio_.sum()

    # 2. Explained variance
    cumvar = pca.explained_variance_ratio_.sum()

    # 3. Trustworthiness
    tw = trustworthiness(X_scaled, X_reduced, n_neighbors=10)

    # 4. Residual variance (global distance preservation)
    from sklearn.metrics import pairwise_distances
    if len(X_scaled) <= 2000:  # Expensive for large n
        D_high = pairwise_distances(X_scaled)
        D_low  = pairwise_distances(X_reduced)
        idx = np.triu_indices(len(X_scaled), k=1)
        resid_var = 1 - np.corrcoef(D_high[idx], D_low[idx])[0, 1] ** 2
    else:
        resid_var = float('nan')

    print(f"Components:          {pca.n_components_}")
    print(f"Explained variance:  {cumvar:.4f} ({cumvar*100:.1f}%)")
    print(f"Reconstruction MSE:  {recon_error:.4f}")
    print(f"Relative error:      {relative_error:.4f}")
    print(f"Trustworthiness:     {tw:.4f}")
    print(f"Residual variance:   {resid_var:.4f}")
    return dict(cumvar=cumvar, recon_error=recon_error, trustworthiness=tw)

pca_eval = PCA(n_components=40, random_state=42)
pca_eval.fit(X_scaled)
metrics = evaluate_pca(pca_eval, digits.data, X_scaled)
```

---

## §9 — Innovative Industry Applications

### Application 1: Genomics — Population Stratification Control

> 🏭 **Industry application:** Used at 23andMe, UK Biobank, and every major GWAS study globally.

**Domain:** Computational genomics / epidemiology

**Problem:** A genome-wide association study (GWAS) tries to find genetic variants (SNPs) that
correlate with a disease. But different human ancestries have different baseline allele frequencies
— a variant common in East Asians is rare in Europeans. If your study has more East Asians in
the case group (disease) than the control group, you'll falsely identify thousands of ancestry-
associated SNPs as "disease genes."

**Why PCA:** The top 10–20 PCs of a matrix of $n$ individuals × $p$ SNPs (where $p \sim 10^6$)
capture the entire population structure of the dataset. PC1 vs. PC2 perfectly separates European,
African, East Asian, and South Asian ancestries. Including these PCs as covariates in the GWAS
regression removes the population stratification confound.

```python
from sklearn.decomposition import PCA
import numpy as np

# snp_matrix: (n_individuals, n_snps) — e.g., (5000, 500_000)
# Use randomized SVD — full SVD is impossible at this scale
pca_gwas = PCA(n_components=20, svd_solver='randomized', random_state=42)
pop_pcs = pca_gwas.fit_transform(snp_matrix)  # (n_individuals, 20)

# Include in regression as covariates:
# phenotype ~ beta_snp * snp + beta_pc1 * PC1 + ... + beta_pc20 * PC20
# (Logistic or linear regression, per SNP)
```

**Reference:** Price et al. (2006), *Nature Genetics*. doi: 10.1038/ng1847

---

### Application 2: Quantitative Finance — Statistical Risk Factor Decomposition

> 🏭 **Industry application:** Used at BlackRock, JP Morgan, Goldman Sachs, and the Federal Reserve.

**Domain:** Portfolio risk management

**Problem:** A portfolio of 500 stocks has a covariance matrix with 125,000 parameters to
estimate from limited historical data. The naive covariance estimate is numerically unstable and
produces extreme, unreliable portfolio weights in mean-variance optimization.

**Why PCA:** The covariance matrix of stock returns has a remarkably low-rank structure.
The first 5–10 PCs explain 50–70% of all variance. PC1 is almost always "market risk"
(all loadings positive, proportional to market cap weights). PC2 often captures growth vs.
value, PC3 captures size, etc. — mirroring the Fama-French factors but discovered purely
from data.

```python
from sklearn.decomposition import PCA
import numpy as np

# returns: (n_days, n_stocks), e.g., (252, 500)
pca_risk = PCA(n_components=20)
pca_risk.fit(returns)

# Decompose covariance into systematic + idiosyncratic
systematic_cov = (pca_risk.components_.T
                  @ np.diag(pca_risk.explained_variance_)
                  @ pca_risk.components_)

residuals = returns - pca_risk.inverse_transform(pca_risk.transform(returns))
idiosyncratic_var = np.var(residuals, axis=0, ddof=1)
regularized_cov = systematic_cov + np.diag(idiosyncratic_var)
# This regularized covariance is numerically stable and well-conditioned

# Factor exposures (betas to each statistical factor):
factor_exposures = pca_risk.transform(returns)  # (n_days, 20)
```

**Non-obvious insight:** Each "eigen-portfolio" (row of `pca.components_`) is theoretically a
mean-variance efficient portfolio if returns follow a linear factor model. PC1 is approximately
the market-cap-weighted index.

**Reference:** Federal Reserve (2025). *Portfolio Margining Using PCA Latent Factors.*
https://www.federalreserve.gov/econres/feds/files/2025016pap.pdf

---

### Application 3: Manufacturing Quality Control — PCA Anomaly Detection

> 🏭 **Industry application:** Used in semiconductor fabs, steel mills, pharmaceutical production.

**Domain:** Industrial quality inspection

**Problem:** A semiconductor fab's wafer inspection system generates 10,000-dimensional
optical reflectance vectors. A rule-based system would require thousands of thresholds.
You want an unsupervised approach that flags any measurement that's unusual — not
just unusual in one dimension, but unusual in the multivariate sense.

**Why PCA:** Normal wafers form a low-dimensional manifold (their variation is systematic and
correlated). Defective wafers deviate from this manifold in unpredictable ways. The
**Q-statistic** (reconstruction error from PCA fit on normal wafers) captures this: normal
wafers have near-zero reconstruction error; defective ones have large error because their
anomalous variation is not captured by the "normal" principal components.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Train on known-good wafers only
scaler = StandardScaler()
X_normal_sc = scaler.fit_transform(X_normal_wafers)

pca_qc = PCA(n_components=0.95, svd_solver='full')  # Keep 95% of normal variance
pca_qc.fit(X_normal_sc)

def q_statistic(X_new):
    """PCA-based anomaly score: reconstruction error per sample."""
    X_sc = scaler.transform(X_new)
    X_rec = pca_qc.inverse_transform(pca_qc.transform(X_sc))
    return np.sum((X_sc - X_rec) ** 2, axis=1)  # Per-sample Hotelling Q

# Set threshold at 99th percentile of normal distribution
threshold = np.percentile(q_statistic(X_normal_wafers), 99)
anomaly_flags = q_statistic(X_test_wafers) > threshold
print(f"Flagged {anomaly_flags.mean()*100:.1f}% of test wafers")
```

**Extension:** For in-distribution anomalies (defects that are linear combinations of normal
variation), use the **Hotelling T² statistic** in the PC subspace instead of Q. A complete
system uses both: T² for in-subspace anomalies, Q for out-of-subspace anomalies.

---

### Application 4: Computer Vision — Eigenfaces and Face Recognition

> 🏭 **Industry application:** Historical gold standard (1991–2000); teaches reconstruction-error
> anomaly detection still used in modern systems.

**Domain:** Biometric recognition

PCA on flattened face images produces "eigenfaces" — the principal directions of face variation.
The key insight from Turk & Pentland (1991): face images lie on a low-dimensional linear
manifold in pixel space. New images are classified by nearest neighbor in eigenface space.

```python
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X_faces = faces.data      # (n_images, n_pixels)
y_faces = faces.target    # Person labels
h, w = faces.images.shape[1:]

pca_faces = PCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42)
X_faces_pca = pca_faces.fit_transform(X_faces)
eigenfaces = pca_faces.components_.reshape((150, h, w))

# Plot first 6 eigenfaces
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces[i], cmap='gray')
    ax.set_title(f'Eigenface {i+1}')
    ax.axis('off')
plt.suptitle('Eigenfaces — Principal Modes of Face Variation', fontsize=14)
plt.tight_layout()
plt.show()
```

**Why `whiten=True` here:** Face recognition uses nearest-neighbor distance in eigenface space.
Without whitening, PC1 (illumination) would dominate all distances. Whitening equalizes the
contribution of all eigenfaces, improving recognition.

---

### Application 5: NLP — Latent Semantic Analysis

> 🏭 **Industry application:** Used in early search engines; the conceptual ancestor of word2vec, GloVe, and transformers.

**Domain:** Information retrieval / document similarity

PCA (via TruncatedSVD) on TF-IDF matrices maps synonyms to nearby points in latent semantic
space, even if they never co-occur in the same document.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

lsa = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50_000, sublinear_tf=True, min_df=5)),
    ('svd',   TruncatedSVD(n_components=300, algorithm='randomized', random_state=42))
])
# CRITICAL: Use TruncatedSVD, NOT PCA — centering destroys sparsity
X_lsa = lsa.fit_transform(documents)

# Query: find documents most similar to a query
query_vec = lsa.transform([query_text])  # (1, 300)
similarities = X_lsa @ query_vec.T       # Cosine similarity (after L2 normalize)
top_k = np.argsort(similarities.ravel())[::-1][:5]
```

---

### Application 6: Quantum Chemistry — Molecular Descriptor Compression

> 🏭 **Industry application:** Used at pharma companies (Novartis, Pfizer) and AI drug discovery startups.

**Domain:** Computational chemistry / QSAR modeling

Molecules are described by thousands of quantum-mechanical descriptors (partial charges, HOMO/LUMO
energies, polarizability). QSAR models on all descriptors severely overfit. PCA compresses correlated
molecular descriptors into 20–50 interpretable principal factors.

**Non-obvious insight:** The PCA loadings often correspond to interpretable chemical concepts:
PC1 typically tracks molecular size, PC2 polarity, PC3 ring aromaticity. A chemist can
annotate PCA axes in a way they can't annotate raw descriptor indices.

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# molecular_descriptors: (n_molecules, n_descriptors)
qsar_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.90, svd_solver='full')),
    ('model', GradientBoostingRegressor(n_estimators=200))
])
qsar_pipe.fit(molecular_descriptors, activity_values)
```

---

### Application 7: Neuroscience — EEG/BCI Signal Preprocessing

> 🏭 **Industry application:** Used in consumer EEG devices (Emotiv, Muse) and clinical BCI research.

**Domain:** Brain-computer interfaces

EEG signals from 64+ channels are high-dimensional and highly correlated. Eye blinks and cardiac
artifacts dominate the highest-variance components. PCA preprocessing:
1. Fits PCA on contaminated data
2. Identifies artifact components by their temporal pattern (blinks = sharp spikes in time)
3. Reconstructs signal without artifact components (`inverse_transform` on artifact-free scores)

```python
from sklearn.decomposition import PCA
import numpy as np

# eeg_data: (n_timepoints, n_channels)
pca_eeg = PCA(n_components=0.99, svd_solver='full')
eeg_pca = pca_eeg.fit_transform(eeg_data)  # (n_timepoints, n_components)

# Identify artifact components: PC with very high kurtosis (non-Gaussian → artifact)
from scipy.stats import kurtosis
kurt_scores = kurtosis(eeg_pca, axis=0)
artifact_mask = kurt_scores > 5.0  # Threshold (domain-specific)

# Zero out artifact components and reconstruct
eeg_pca_clean = eeg_pca.copy()
eeg_pca_clean[:, artifact_mask] = 0
eeg_reconstructed = pca_eeg.inverse_transform(eeg_pca_clean)
```

---

### Application 8: Materials Science — High-Entropy Alloy Discovery

> 🏭 **Industry application:** Active learning for materials discovery at national labs (Argonne, NIST).

**Domain:** Materials science / active learning

High-entropy alloys (HEAs) have 5+ principal elements with continuous composition ranges. PCA on
a database of alloy compositions + properties reveals the principal axes of composition variation.
Active learning systems use PCA embeddings to select the next experiment: choose the alloy composition
farthest from all previously measured points in PCA space.

The loadings (`pca.components_`) are often directly interpretable as "alloying substitution groups"
— which elements tend to substitute for each other in maintaining a given crystal structure.

---

## §10 — Complete Python Worked Example

Three datasets. Three distinct PCA use cases: visualization + classification, image compression
with Eigenfaces, and feature engineering for regression.

---

### Example A: Digits Dataset — Visualization and Classification

**The problem:** The `digits` dataset has 1,797 images of handwritten digits (0–9), each
represented as a $8 \times 8$ pixel grid flattened to 64 features. We want to:
1. Understand the manifold structure via PCA visualization
2. Build an efficient classifier using PCA-compressed features
3. Diagnose how many components are actually needed

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.manifold import trustworthiness

# ─── 1. Load and Explore ─────────────────────────────────────────────────────
digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset shape: {X.shape}")   # (1797, 64)
print(f"Classes: {np.unique(y)}")
print(f"Feature range: {X.min():.0f} – {X.max():.0f}")  # Pixel values 0-16

# Brief EDA: pixel value distributions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for digit, ax in zip(range(10), axes.flat):
    mean_img = X[y == digit].mean(axis=0).reshape(8, 8)
    ax.imshow(mean_img, cmap='gray_r')
    ax.set_title(f'Mean digit {digit}')
    ax.axis('off')
plt.suptitle('Mean image per digit class')
plt.tight_layout()
plt.show()

# ─── 2. Preprocessing ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)   # fit only on train!

# ─── 3. Scree Analysis (on train only) ────────────────────────────────────────
pca_full = PCA(svd_solver='full').fit(X_train_sc)
evr = pca_full.explained_variance_ratio_
cumevr = np.cumsum(evr)

# How many components for key thresholds?
for t in [0.80, 0.90, 0.95, 0.99]:
    n = np.searchsorted(cumevr, t) + 1
    print(f"{t*100:.0f}% variance → {n} components")
# Typical output:
# 80% variance → 12 components
# 90% variance → 21 components
# 95% variance → 29 components
# 99% variance → 41 components

# ─── 4. 2D Visualization ─────────────────────────────────────────────────────
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_train_sc)

fig, ax = plt.subplots(figsize=(10, 8))
palette = plt.cm.get_cmap('tab10', 10)
for d in range(10):
    mask = y_train == d
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[palette(d)], label=str(d), alpha=0.6, s=15)
ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)')
ax.set_title('Digits: 2D PCA Projection (train set)')
ax.legend(title='Digit', ncol=2, loc='upper right')
plt.tight_layout()
plt.show()

# ─── 5. Eigendigits: The PCA Components ──────────────────────────────────────
pca_eigen = PCA(n_components=16, random_state=42)
pca_eigen.fit(X_train_sc)

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for i, ax in enumerate(axes.flat):
    comp_img = pca_eigen.components_[i].reshape(8, 8)
    ax.imshow(comp_img, cmap='RdBu_r')
    ax.set_title(f'PC{i+1}\n({evr[i]*100:.1f}%)', fontsize=8)
    ax.axis('off')
plt.suptitle('First 16 PCA Components — The "Eigendigits"')
plt.tight_layout()
plt.show()

# ─── 6. Classification: PCA + Logistic Regression ────────────────────────────
# Build pipelines for different n_components
results = {}
for n in [2, 5, 10, 20, 29, 41, 64]:  # 64 = no compression
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n, random_state=42) if n < 64 else 'passthrough'),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    results[n] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}

print(f"\n{'n_comp':>8} | {'Accuracy':>10} | {'±':>8}")
print('-' * 32)
for n, r in results.items():
    label = 'baseline' if n == 64 else ''
    print(f"{n:>8} | {r['mean']:.4f} | ±{r['std']:.4f}  {label}")

# ─── 7. Trustworthiness at Different Compressions ────────────────────────────
tw_results = {}
for n in [2, 5, 10, 20, 30, 40]:
    pca_n = PCA(n_components=n, random_state=42)
    X_n = pca_n.fit_transform(X_train_sc)
    tw_results[n] = trustworthiness(X_train_sc, X_n, n_neighbors=10)

print("\nTrustworthiness (k=10):")
for n, tw in tw_results.items():
    print(f"  {n} components: {tw:.4f}")
```

**What we learn:** The digits dataset has strong low-rank structure — 95% of variance is captured
in just 29 of 64 features. Classification accuracy with 20 components (90% variance) is nearly
identical to using all 64 features. The 2D visualization shows clear clusters for "1" (isolated),
"0" (upper right), and overlapping clusters for "3/8" (which are visually similar).

---

### Example B: Labeled Faces in the Wild — Eigenfaces and Image Compression

**The problem:** Face images of celebrities, $62 \times 47 = 2914$ pixels per image. We want
to demonstrate image compression via PCA reconstruction and the Eigenfaces visualization.

```python
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X_faces = faces.data     # (1140, 1850) — 1140 images, 50×37 pixels each
y_faces = faces.target
target_names = faces.target_names
h, w = faces.images.shape[1], faces.images.shape[2]

print(f"Dataset: {X_faces.shape[0]} images, {X_faces.shape[1]} features")
print(f"People: {target_names}")
print(f"Image dimensions: {h}×{w}")

# ─── 2. Train/Test Split ──────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_faces, y_faces, test_size=0.25, random_state=42, stratify=y_faces
)

# ─── 3. PCA — Fit on Training Faces Only ─────────────────────────────────────
n_components = 150
pca_faces = PCA(n_components=n_components, svd_solver='randomized',
                whiten=True, random_state=42)
X_tr_pca = pca_faces.fit_transform(X_tr)
X_te_pca = pca_faces.transform(X_te)

print(f"\nVariance captured by {n_components} components: "
      f"{pca_faces.explained_variance_ratio_.sum()*100:.1f}%")

# ─── 4. Eigenfaces Visualization ─────────────────────────────────────────────
eigenfaces = pca_faces.components_.reshape((n_components, h, w))

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces[i], cmap='gray')
    ax.set_title(f'Eigenface {i+1}\n({pca_faces.explained_variance_ratio_[i]*100:.1f}%)',
                 fontsize=9)
    ax.axis('off')
plt.suptitle('First 15 Eigenfaces — Principal Modes of Face Variation', fontsize=14)
plt.tight_layout()
plt.show()

# ─── 5. Image Compression: Reconstruct at Different Qualities ────────────────
test_face_idx = 0
test_face = X_te[test_face_idx:test_face_idx+1]
test_face_sc = (test_face - pca_faces.mean_)  # Manual centering for demo

compression_levels = [5, 20, 50, 100, 150]
fig, axes = plt.subplots(1, len(compression_levels) + 1, figsize=(16, 4))

axes[0].imshow(test_face.reshape(h, w), cmap='gray')
axes[0].set_title('Original\n(1850 px)')
axes[0].axis('off')

for ax, n in zip(axes[1:], compression_levels):
    pca_n = PCA(n_components=n, svd_solver='randomized', whiten=True, random_state=42)
    pca_n.fit(X_tr)
    reconstructed = pca_n.inverse_transform(pca_n.transform(test_face))
    ax.imshow(reconstructed.reshape(h, w), cmap='gray')
    var = pca_n.explained_variance_ratio_.sum()
    ax.set_title(f'{n} components\n({var*100:.0f}% var)')
    ax.axis('off')

plt.suptitle('Face Reconstruction Quality vs. Number of PCA Components', fontsize=13)
plt.tight_layout()
plt.show()

# ─── 6. Face Recognition with SVM ────────────────────────────────────────────
# Eigenfaces + SVM: classic baseline (not deep learning, but instructive)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

svm_clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm_clf.fit(X_tr_pca, y_tr)
y_pred = svm_clf.predict(X_te_pca)
print(classification_report(y_te, y_pred, target_names=target_names))

# Hyperparameter tuning via grid search (PCA already fit; tune SVM only)
param_grid = {'C': [1e3, 5e3, 1e4], 'gamma': [0.0001, 0.0005, 0.001]}
gs = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                  param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
gs.fit(X_tr_pca, y_tr)
print(f"Best params: {gs.best_params_}, F1: {gs.best_score_:.3f}")
```

**What we learn:** Eigenface 1 captures overall illumination (all pixels light vs. dark).
Eigenfaces 2–5 capture left-right lighting asymmetry, hair color, and expression. With just
50 components (out of 1850 pixels), faces are recognizable. The SVM classifier on 150
eigenfaces achieves 70–80% macro-F1 on the 7-person classification task — reasonable for a
linear subspace method.

---

### Example C: California Housing — PCA for Feature Engineering

**The problem:** The California housing dataset has 8 numerical features (median income, housing
age, average rooms, etc.) with moderate correlation. We want to demonstrate PCA as a
feature engineering technique before regression, and show when it helps vs. hurts.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ─── 1. Load and Explore ─────────────────────────────────────────────────────
housing = fetch_california_housing()
X_h, y_h = housing.data, housing.target
feature_names = housing.feature_names

print(f"Shape: {X_h.shape}")  # (20640, 8)
print(f"Features: {feature_names}")

# Correlation matrix — are features correlated?
df_h = pd.DataFrame(X_h, columns=feature_names)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_h.corr(), annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True)
ax.set_title('Feature Correlation Matrix — California Housing')
plt.tight_layout()
plt.show()

# ─── 2. Scree Analysis ───────────────────────────────────────────────────────
scaler_h = StandardScaler()
X_h_sc = scaler_h.fit_transform(X_h)

pca_h = PCA(svd_solver='full').fit(X_h_sc)
print("\nExplained variance per component:")
for i, (evr, ev) in enumerate(zip(pca_h.explained_variance_ratio_,
                                    pca_h.explained_variance_)):
    print(f"  PC{i+1}: {evr*100:.1f}%  (eigenvalue={ev:.3f})")

# ─── 3. Loading Analysis ─────────────────────────────────────────────────────
loadings_df = pd.DataFrame(
    pca_h.components_,
    columns=feature_names,
    index=[f'PC{i+1}' for i in range(8)]
)
print("\nPC Loadings:")
print(loadings_df.round(3))

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax)
ax.set_title('PCA Loadings — California Housing Features')
plt.tight_layout()
plt.show()

# ─── 4. Compare Regression Performance ───────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Baseline: Ridge regression on all features
pipe_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])
score_ridge = cross_val_score(pipe_ridge, X_h, y_h, cv=cv,
                              scoring='r2').mean()

# PCA + Ridge: at different compression levels
pca_ridge_scores = {}
for n in [2, 3, 4, 5, 6, 7, 8]:
    pipe_pca = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n, svd_solver='full')),
        ('model', Ridge())
    ])
    pca_ridge_scores[n] = cross_val_score(pipe_pca, X_h, y_h, cv=cv,
                                           scoring='r2').mean()

# Gradient Boosting baseline (nonlinear — PCA may hurt)
pipe_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(n_estimators=200, random_state=42))
])
score_gb = cross_val_score(pipe_gb, X_h, y_h, cv=cv, scoring='r2').mean()

# PCA + GBM
score_pca_gb = cross_val_score(Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('model', GradientBoostingRegressor(n_estimators=200, random_state=42))
]), X_h, y_h, cv=cv, scoring='r2').mean()

print(f"\nR² Scores:")
print(f"  Ridge (all 8 features): {score_ridge:.4f}")
for n, s in pca_ridge_scores.items():
    print(f"  PCA({n}) + Ridge:         {s:.4f}")
print(f"  GBM (all features):    {score_gb:.4f}")
print(f"  PCA(5) + GBM:          {score_pca_gb:.4f}")

# Plot performance vs. n_components
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(pca_ridge_scores.keys()),
        list(pca_ridge_scores.values()),
        'b-o', label='PCA + Ridge')
ax.axhline(score_ridge, color='blue', linestyle='--', label='Ridge (no PCA)')
ax.axhline(score_gb, color='green', linestyle='--', label='GBM (no PCA)')
ax.axhline(score_pca_gb, color='green', linestyle=':', label='PCA(5) + GBM')
ax.set_xlabel('n_components')
ax.set_ylabel('R² (cross-validated)')
ax.set_title('PCA as Feature Engineering — Does it Help Regression?')
ax.legend()
ax.set_ylim(0.4, 0.7)
plt.tight_layout()
plt.show()
```

**What we learn:** The California housing features are moderately correlated. PC1 (~35% variance)
captures a "neighborhood quality" factor (median income + house age). PC2 captures population
density vs. house size. For Ridge regression, PCA with 5–6 components nearly matches the
full-feature baseline (R²≈0.59) — showing that ~5 latent factors capture most of the linear
predictive signal. For Gradient Boosting (a nonlinear model), PCA hurts: GBM without PCA
achieves R²≈0.80, while PCA(5)+GBM drops to ≈0.72 because PCA discards some nonlinear
signal combinations that GBM exploits.

> 💡 **Intuition:** PCA helps linear models by reducing collinearity and noise. It hurts or
> has no effect for nonlinear ensemble models (GBM, Random Forest) that already handle
> correlated features well internally. Use PCA before linear models; be skeptical before
> nonlinear ones.

---

## §11 — When to Use This Algorithm

### Decision Flowchart

```
START: Do you need to reduce dimensionality?
│
├─ Is your data categorical or mixed (numeric + categorical)?
│   YES → prince.MCA (categorical) or prince.FAMD (mixed). NOT PCA.
│
├─ Is your data sparse (TF-IDF, count matrix, one-hot)?
│   YES → TruncatedSVD (LSA). NOT PCA (centering destroys sparsity).
│
├─ Does your data have nonlinear manifold structure (circles, spirals, clusters on a curve)?
│   YES → Kernel PCA (RBF kernel), UMAP, or t-SNE (visualization only)
│
├─ Do you need to VISUALIZE the data?
│   YES, labels available → UMAP (preserves global + local), t-SNE (local clusters)
│   YES, labels unavailable → PCA (most trustworthy global structure), then UMAP
│   PCA for initial global view is ALWAYS a good first step before UMAP/t-SNE
│
├─ Do you need interpretable, sparse loading vectors?
│   YES → SparsePCA / MiniBatchSparsePCA
│
├─ Do features have heterogeneous noise levels?
│   YES → FactorAnalysis (per-feature noise estimation via EM)
│
├─ Does your data fit in RAM?
│   NO → IncrementalPCA (partial_fit)
│
└─ Dense data, linear structure, fits in RAM?
    n_components/min(n,p) ≪ 1? → PCA(svd_solver='randomized')  [fastest]
    n ≫ p and p < 1000?        → PCA(svd_solver='covariance_eigh')  [new in 1.5]
    Small data or need exact?  → PCA(svd_solver='full')
```

### Comparison Table: PCA vs. Close Alternatives

| Dimension | PCA | UMAP | t-SNE | LDA | ICA |
|---|---|---|---|---|---|
| **Type** | Linear | Nonlinear | Nonlinear | Linear (supervised) | Linear (unsupervised) |
| **Objective** | Max variance | Preserve topology | Preserve local neighborhoods | Max class separability | Statistical independence |
| **Needs labels** | No | Optional | No | Yes | No |
| **Preserves global structure** | Yes | Partially | No | Yes | Partially |
| **Out-of-sample transform** | Yes | Yes (parametric) | No (native) | Yes | Yes |
| **Reconstruction possible** | Yes (exact for linear) | No | No | Yes | Yes |
| **Interpretable components** | Yes (loadings) | No | No | Yes (discriminant axes) | Yes (mixing matrix) |
| **Speed (large data)** | Fast | Moderate | Slow | Fast | Moderate |
| **Best for** | Feature engineering, compression, preprocessing | Exploration, cluster visualization | Publication cluster plots | Supervised feature extraction | Blind source separation |

### Red Flags — Do NOT Use PCA When:

- Your features are binary, ordinal, or categorical (use MCA/FAMD)
- Your data matrix is sparse (use TruncatedSVD)
- You need guaranteed independence (uncorrelated ≠ independent — use ICA)
- Your goal is purely cluster visualization and clusters are nonlinear (use UMAP or t-SNE)
- Your labels define the important structure and you have enough data (use LDA — it knows about classes)
- You have severe outliers that cannot be removed (use Robust PCA or RPCA-based methods)

### Green Flags — PCA Is Likely Your First Choice When:

- Dense numerical features, moderate size, unclear structure → start here always
- You need a reproducible, invertible dimensionality reduction
- You want to understand feature correlations and latent structure
- Preprocessing before a linear downstream model (Ridge, Logistic Regression, SVM)
- Anomaly detection via reconstruction error
- Computational budget is tight (PCA is by far the fastest DR algorithm)
- You need to communicate results to stakeholders (loadings are interpretable)

---

## §12 — Top Papers to Study

These are the 8 papers that will take you from "I know how to call `PCA.fit()`" to genuine
mastery. Read them in this order.

---

### Tier 1 — Start Here

**Paper 1: Pearson (1901) — The Origin**

> Pearson, K. (1901). LIII. On lines and planes of closest fit to systems of points in space.
> *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11), 559–572.
> DOI: 10.1080/14786440109462720

**Annotation:** This 13-page paper introduces PCA as a geometric problem — finding the line or
plane of minimum perpendicular distance to a scatter of points. Reading it gives you Pearson's
geometric mental model (compression, not variance maximization), which is crucial for understanding
reconstruction error. The notation is classical but the math is simple. Read this first.

**What you'll learn:** The geometric interpretation of PCA. Why the two formulations (variance
maximization vs. reconstruction error minimization) are equivalent. Historical context of how
statistics originated from geometry.

---

**Paper 2: Hotelling (1933) — The Statistical Formalization**

> Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components.
> *Journal of Educational Psychology*, 24(6), 417–441.
> URL: http://cda.psych.uiuc.edu/hotelling_principal_components.pdf

**Annotation:** Hotelling reformulates PCA from a statistics perspective, introducing the
covariance matrix eigendecomposition. This is the version sklearn implements. Reading both
Pearson and Hotelling gives you the two mental models that together explain all of PCA's
strengths and weaknesses.

**What you'll learn:** The statistical formulation (variance decomposition, latent roots),
orthogonality of components, and the historical context connecting PCA to factor analysis.

---

**Paper 3: Turk & Pentland (1991) — Eigenfaces**

> Turk, M., & Pentland, A. (1991). Eigenfaces for recognition.
> *Journal of Cognitive Neuroscience*, 3(1), 71–86.
> DOI: 10.1162/jocn.1991.3.1.71
> URL: https://www.face-rec.org/algorithms/PCA/jcn.pdf

**Annotation:** The most readable PCA application paper ever written. Shows that face images
live on a low-dimensional linear subspace, that PCA finds the "eigenfaces" spanning this
subspace, and that reconstruction error can classify faces and detect non-faces. Established
the reconstruction-error paradigm for anomaly detection. Extremely clear writing with strong
intuition.

**What you'll learn:** How to apply PCA to image data, the eigenfaces visualization, the
reconstruction-error anomaly detection idea, and why whitening matters for distance-based
nearest-neighbor classification.

---

### Tier 2 — Read Next

**Paper 4: Halko, Martinsson, Tropp (2011) — Randomized PCA**

> Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness:
> Probabilistic algorithms for constructing approximate matrix decompositions.
> *SIAM Review*, 53(2), 217–288.
> DOI: 10.1137/090771806
> arXiv: https://arxiv.org/abs/0909.4061

**Annotation:** The foundational paper for all modern fast PCA and SVD. The algorithm box
(Algorithm 4.3) is exactly what sklearn's `svd_solver='randomized'` implements. The paper
proves rigorous error bounds, explains why power iterations are needed for slowly-decaying
spectra, and provides comprehensive benchmarks. If you work with large matrices, this paper
tells you exactly what accuracy you're getting and at what cost.

**What you'll learn:** The random projection + power iteration algorithm, error guarantees,
when to increase `iterated_power` and `n_oversamples`, and the theoretical foundation for
all fast matrix decomposition methods in sklearn.

---

**Paper 5: Schölkopf, Smola, Müller (1998) — Kernel PCA**

> Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis as a kernel
> eigenvalue problem. *Neural Computation*, 10(5), 1299–1319.
> DOI: 10.1162/089976698300017467

**Annotation:** Introduces Kernel PCA by applying the kernel trick to the PCA eigenvalue
problem. The paper proves that eigenvectors of the centered kernel matrix correspond to
principal components in the (possibly infinite-dimensional) feature space. Requires Mercer's
theorem background but the algorithm boxes are self-contained. Essential for understanding
why KPCA can find nonlinear structure without explicitly computing feature maps.

**What you'll learn:** The dual formulation of PCA, the kernel centering trick, how to project
new points, and the pre-image problem (why inverse transform is approximate for nonlinear kernels).

---

**Paper 6: Zou, Hastie, Tibshirani (2006) — Sparse PCA**

> Zou, H., Hastie, T., & Tibshirani, R. (2006). Sparse principal component analysis.
> *Journal of Computational and Graphical Statistics*, 15(2), 265–286.
> DOI: 10.1198/106186006X113430
> URL: https://hastie.su.domains/Papers/spc_jcgs.pdf

**Annotation:** Shows how to reformulate PCA as a regression problem, enabling L1
regularization for sparse loading vectors. The alternating optimization algorithm (LASSO for
sparse loadings, ridge/SVD for scores) is elegant and efficient. Essential for anyone applying
PCA to gene expression, financial factors, or any domain where component interpretability matters.

**What you'll learn:** The SCoTLASS formulation, the SPCA regression reformulation, the
alternating optimization algorithm, and when sparse loading vectors provide genuinely better
scientific interpretation than dense PCA components.

---

### Tier 3 — Deep Mastery

**Paper 7: Ross, Lim, Lin, Yang (2008) — Incremental PCA**

> Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008). Incremental learning for robust
> visual tracking. *International Journal of Computer Vision*, 77(1–3), 125–141.
> DOI: 10.1007/s11263-007-0075-7
> URL: https://www.cs.ait.ac.th/~mdailey/cvreadings/Ross-Incremental.pdf

**Annotation:** Provides the correct incremental SVD update formulas that sklearn's
`IncrementalPCA` implements. The key subtlety is the mean update: as new data arrives, the
running mean changes, and the existing SVD must be corrected. This paper gives you the exact
formulas and proves convergence. Read this if you use IncrementalPCA on non-stationary streams.

**What you'll learn:** The correct incremental SVD algorithm, mean tracking, forgetting factors
for non-stationary data, and the convergence properties of approximate online PCA.

---

**Paper 8: Fang, Tao, Lv (2024) — Kernel PCA for Out-of-Distribution Detection**

> Fang, K., Tao, Q., & Lv, K. (2024). Kernel PCA for Out-of-Distribution Detection.
> *NeurIPS 2024*.
> arXiv: https://arxiv.org/abs/2402.02949
> GitHub: https://github.com/fanghenshaometeor/ood-kernel-pca

**Annotation:** Modern application of KPCA to deep learning. Shows that linear PCA fails to
separate in-distribution from out-of-distribution (OoD) features in the penultimate layers of
deep networks, but KPCA with task-specific kernels achieves state-of-the-art OoD detection.
The reconstruction error in kernel feature space is the anomaly score. Bridges classical PCA
theory with modern deep learning pipelines.

**What you'll learn:** Why linear PCA is insufficient for OoD detection in DNN feature spaces,
how to design task-specific kernels, and how to scale KPCA to large feature dimensions using
randomized approximations.

---

## §13 — Resources & Further Reading

### Books

**Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.**
The definitive reference. Chapter 2 (mathematical properties), Chapter 6 (choosing components),
Chapter 9 (outlier and influential observations), Chapter 11 (PCA in regression). Comprehensive
coverage of all PCA variants known as of 2002.

**Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 12.**
Covers probabilistic PCA and factor analysis in the generative modeling framework. Excellent
treatment of the EM algorithm for PPCA, the connection between PCA and maximum likelihood,
and the Bayesian PCA extension. Free PDF available from Microsoft Research.
URL: https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf

**Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
(2nd ed.). Springer. Chapter 14.5.**
Concise treatment of PCA, PPCA, Sparse PCA, and the connection to SVD. The authors of SparsePCA
explain it in their own words. Free PDF: https://hastie.su.domains/ElemStatLearn/

---

### Documentation and Tutorials

**scikit-learn PCA documentation**
The most up-to-date reference for all sklearn PCA parameters. Includes worked examples, solver
comparison benchmarks, and version change log.
URL: https://scikit-learn.org/stable/modules/decomposition.html#pca

**scikit-learn Decomposition user guide**
Covers PCA, KernelPCA, SparsePCA, IncrementalPCA, TruncatedSVD, FactorAnalysis in one place
with comparison examples.
URL: https://scikit-learn.org/stable/modules/decomposition.html

**Distill.pub: "A Visual Introduction to Machine Learning" / "Feature Visualization"**
Not PCA-specific, but the interactive visualizations clarify the geometric interpretation.
URL: https://distill.pub

**Jake VanderPlas: "In Depth: Principal Component Analysis" (Python Data Science Handbook)**
The best introductory treatment with clean matplotlib visualizations.
URL: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

---

### Research Papers (Beyond §12)

**Minka, T. P. (2001). "Automatic choice of dimensionality for PCA." NeurIPS 2001.**
The paper behind `n_components='mle'`. Derives the Bayesian model comparison approach for
selecting $d$. Elegant and short (7 pages).
URL: https://vismod.media.mit.edu/tech-reports/TR-514.pdf

**Tipping, M. E., & Bishop, C. M. (1999). "Probabilistic principal component analysis."
*Journal of the Royal Statistical Society: Series B*, 61(3), 611–622.**
The foundational paper for probabilistic PCA. Derives PCA from a Gaussian latent variable model,
enabling EM-based fitting with missing data.
DOI: 10.1111/1467-9868.00196

**Deerwester, S., et al. (1990). "Indexing by latent semantic analysis." *JASIS*, 41(6).**
The LSA paper — applying SVD to document-term matrices. The conceptual ancestor of word2vec,
GloVe, and transformer pre-training objectives.
DOI: 10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9

---

### Specialized Packages — Documentation Links

| Package | Documentation | What to Read |
|---|---|---|
| `prince` | https://github.com/MaxHalford/prince | MCA, FAMD examples |
| `pyDRMetrics` | https://pypi.org/project/pyDRMetrics/ | Evaluation metric API |
| `qrpca` | https://github.com/BergerLab/QRPCA | GPU PCA benchmarks |
| `PyParSVD` | https://github.com/Shaokun-X/PyParSVD | Streaming SVD usage |
| `openTSNE` | https://opentsne.readthedocs.io | For comparison with t-SNE |

---

*This chapter is part of the Dimensionality Reduction Masterclass. The next chapter covers
Linear Discriminant Analysis (LDA) — the supervised extension of PCA that maximizes
class separability rather than variance.*

