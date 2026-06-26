# Linear Discriminant Analysis: The Complete Masterclass

> **Why this algorithm matters:** In 1936, Ronald Fisher published a paper on classifying iris
> flowers. He was trying to answer a biologist's question — can you tell three species apart from
> petal measurements alone? — and in doing so, he invented a method that now runs inside medical
> diagnostic systems, brain-computer interfaces, fraud detection pipelines, and the feature
> refinement layers of modern deep learning applications. A 2026 controlled study found that
> inserting a single LDA step after a frozen ResNet-50 improved classification accuracy by up to
> 4.6 percentage points over using the raw CNN features. That is the power of supervised
> dimensionality reduction: when you have labels, you should use them. LDA is the cleanest,
> most theoretically grounded way to do exactly that.

---

## §0 — Python Ecosystem & Package Guide

Before touching any mathematics, you need to understand the landscape of tools. LDA has a
surprisingly rich ecosystem — not just sklearn, but dedicated packages for educational
introspection, statistical validation, Riemannian extensions for EEG data, and visualization.
Choosing the right package can save you hours of debugging and unlock capabilities that the
main sklearn API deliberately hides behind parameter names.

### Complete Package Table

| Package | Import | Version (mid-2026) | What it provides | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` | ~1.5.x–1.9.x | Full LDA with three solvers (SVD/LSQR/Eigen), shrinkage, custom covariance estimators, QDA. Pipeline-compatible. | BSD-3 |
| `mlxtend` | `from mlxtend.feature_extraction import LinearDiscriminantAnalysis` | ~0.23.x | Educational, step-by-step scatter-matrix → eigen decomposition implementation. Exposes raw eigenvectors and eigenvalues for inspection. | BSD-3 |
| `pingouin` | `import pingouin as pg` | ~0.5.x | Statistical testing around LDA assumptions: Mardia's multivariate normality, Box's M test for covariance homogeneity, MANOVA. | GPL-3.0 |
| `pyriemann` | `from pyriemann.classification import MDM` | ~0.7.x | Riemannian geometry LDA for covariance-matrix data (EEG/BCI). Operates on the manifold of symmetric positive definite (SPD) matrices. | BSD-3 |
| `prince` | `import prince` | ~0.13.x | Pandas-friendly LDA wrapper with seaborn-style biplots of the discriminant space. Good for rapid visualization. | MIT |
| `sklearn.covariance` | `from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance` | same as sklearn | Covariance estimators passed to LDA via `covariance_estimator=`. Critical companions for regularized LDA. | BSD-3 |

### Top Picks — Recommendation Table

| Package | Best for | Modern/Legacy | Notes |
|---|---|---|---|
| **`sklearn.LinearDiscriminantAnalysis`** | Production pipelines, all standard use cases | Modern | First choice. Three solvers, shrinkage, Pipeline-compatible, well-tested. |
| **`sklearn.QuadraticDiscriminantAnalysis`** | When class covariances differ substantially | Modern | Drop-in replacement for LDA when equal-covariance assumption fails. |
| **`mlxtend.LinearDiscriminantAnalysis`** | Learning, debugging, inspecting eigenvectors | Legacy-educational | Use to see the raw scatter matrices and eigenvectors. Not for production. |
| **`pingouin`** | Pre-fit assumption checking | Modern | Run `pg.multivariate_normality()` and Box's M before fitting any LDA. |
| **`pyriemann`** | EEG / BCI / covariance-matrix input data | Specialized | Only relevant when your data is naturally SPD matrices, not feature vectors. |

### The Same Fit Across Top Packages

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset Fisher himself introduced
data = load_iris()
X, y = data.data, data.target  # (150, 4), 3 classes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Option 1: sklearn (the workhorse) ──────────────────────────────────────
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda_sk = LinearDiscriminantAnalysis(solver='svd', n_components=2)
lda_sk.fit(X_train_sc, y_train)
X_2d = lda_sk.transform(X_train_sc)           # (120, 2) — LD1 and LD2
print(f"sklearn accuracy:          {lda_sk.score(X_test_sc, y_test):.4f}")
print(f"Explained variance ratio:  {lda_sk.explained_variance_ratio_}")

# ── Option 2: sklearn with Ledoit-Wolf regularization ─────────────────────
lda_reg = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=2)
lda_reg.fit(X_train_sc, y_train)
X_2d_reg = lda_reg.transform(X_train_sc)
print(f"Regularized LDA accuracy:  {lda_reg.score(X_test_sc, y_test):.4f}")

# ── Option 3: sklearn with OAS covariance estimator ───────────────────────
from sklearn.covariance import OAS

lda_oas = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=OAS())
lda_oas.fit(X_train_sc, y_train)
# Note: lsqr has no transform() — classification only
print(f"OAS-regularized accuracy:  {lda_oas.score(X_test_sc, y_test):.4f}")

# ── Option 4: mlxtend (educational — exposes eigenvectors directly) ────────
# pip install mlxtend
from mlxtend.feature_extraction import LinearDiscriminantAnalysis as mlxtLDA

lda_mlxt = mlxtLDA(n_discriminants=2)
lda_mlxt.fit(X_train_sc, y_train)
X_2d_mlxt = lda_mlxt.transform(X_train_sc)
print(f"mlxtend eigenvalues:       {lda_mlxt.e_vals_[:2]}")  # raw eigenvalues

# ── Option 5: QDA (when covariances differ) ───────────────────────────────
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda.fit(X_train_sc, y_train)
print(f"QDA accuracy:              {qda.score(X_test_sc, y_test):.4f}")
```

> 💡 **Intuition:** Think of these packages as different lenses on the same mathematical object.
> sklearn is a production telescope — calibrated, reliable, built for the field. mlxtend is a
> laboratory microscope — slower, but lets you examine the scatter matrices and eigenvectors at
> the cellular level. pingouin is a diagnostic instrument that tells you whether your data even
> satisfies the preconditions for LDA to work correctly. pyriemann is a specialized spectrometer
> for a completely different type of signal. Know which instrument you need before you start.

> 🔧 **In practice:** For 95% of real-world work, `sklearn.LinearDiscriminantAnalysis` with
> `solver='svd'` is your starting point. The only reasons to deviate: (1) you need regularization
> AND `transform()` → switch to `solver='eigen'`; (2) you need regularization but only
> classification → `solver='lsqr'` is fastest; (3) you want OAS or a custom covariance estimator
> → use `covariance_estimator=` with `lsqr` or `eigen`.

### Critical API Constraints You Must Memorize

These are the four rules that trip up almost every new LDA user in sklearn:

**Rule 1: Shrinkage requires a non-SVD solver.**
```python
# WRONG — raises ValueError
LinearDiscriminantAnalysis(solver='svd', shrinkage='auto')

# CORRECT
LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')   # no transform
LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')  # with transform
```

**Rule 2: LSQR cannot `transform()`.**
```python
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lda.fit(X, y)
lda.transform(X)  # raises error — use solver='eigen' if you need transform
```

**Rule 3: `n_components` is hard-bounded by K-1.**
```python
# For 3-class problem (K=3), maximum is 2 components
lda = LinearDiscriminantAnalysis(n_components=5)  # silently clips or raises
# Safe pattern:
n_comps = min(desired, n_classes - 1)
lda = LinearDiscriminantAnalysis(n_components=n_comps)
```

**Rule 4: `shrinkage` and `covariance_estimator` are mutually exclusive.**
```python
# WRONG
LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto',
                           covariance_estimator=OAS())  # ValueError

# CORRECT: pick one
LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')         # Ledoit-Wolf
LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=OAS())  # OAS
```

> ⚠️ **Pitfall:** The most common sklearn LDA error in production is trying to call `transform()`
> after fitting with `solver='lsqr'`. The error message is cryptic. The fix is always: if you
> need both regularization and `transform()`, use `solver='eigen'`.

### GPU-Accelerated and Large-Scale Options

sklearn's LDA runs on CPU only. For very large datasets:

- **RAPIDS cuML** (`from cuml.discriminant_analysis import LinearDiscriminantAnalysis`): GPU-accelerated LDA via CUDA. Drop-in sklearn-compatible API. Use when $n > 10^6$ or $p > 10^4$ and you have an NVIDIA GPU. Verify availability and version in current cuML docs.
- **Dask-ML** does not have a native LDA, but you can use Dask arrays with sklearn's `IncrementalPCA` first, then LDA on the reduced representation.
- **Practical ceiling:** For typical tabular ML problems ($n < 10^5$, $p < 10^3$), sklearn is fast enough that GPU acceleration is rarely necessary. LDA fitting is O($np^2$) and transform is O($nd$) — both are fast for typical data sizes.

---

## §1 — The Origin: The Paper That Started It All

> 📜 **Origin/Citation:** Fisher, R.A. (1936). "The Use of Multiple Measurements in Taxonomic
> Problems." *Annals of Eugenics*, 7(2), 179–188.
> DOI: https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

### The Problem Fisher Was Solving

It is 1936. C.R. Rao has not yet generalized the method to multiple classes. Ronald Fisher is not
a machine learning researcher — that field does not exist. He is a statistician and geneticist,
collaborating with Edgar Anderson, a botanist who has measured four features (sepal length, sepal
width, petal length, petal width) on 150 iris flowers from three species: *Iris setosa*, *Iris
versicolor*, and *Iris virginica*. The question is taxonomic: can we tell these species apart from
these measurements alone?

What came before Fisher's solution was straightforward and unsatisfying: you could pick one
measurement and threshold it. Petal length alone separates *setosa* cleanly. But *versicolor* and
*virginica* overlap on any single measurement. The natural extension — use all four measurements —
raises an immediate question: *how do you combine them*? Simple averaging is arbitrary. What
linear combination $w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4$ best separates the species?

Fisher's innovation was to pose this as an *optimization problem*. Do not hand-pick the weights.
Find the weights that maximize class separation in a mathematically precise sense.

### The Core Insight: Discriminant Ratios

Fisher's insight, stated in modern notation: find a weight vector $\mathbf{w}$ such that the
projected variable $z = \mathbf{w}^T \mathbf{x}$ maximizes the ratio of **between-class variance**
to **within-class variance** along the projection direction.

In the 2-class case (which Fisher originally solved), let $\boldsymbol{\mu}_1$ and $\boldsymbol{\mu}_2$ be the means of the two classes, and let $\mathbf{S}_W$ be the pooled within-class scatter matrix:

$$\mathbf{S}_W = \sum_{i \in C_1} (\mathbf{x}_i - \boldsymbol{\mu}_1)(\mathbf{x}_i - \boldsymbol{\mu}_1)^T + \sum_{i \in C_2} (\mathbf{x}_i - \boldsymbol{\mu}_2)(\mathbf{x}_i - \boldsymbol{\mu}_2)^T$$

After projection onto $\mathbf{w}$, the between-class variance is:

$$(\mathbf{w}^T \boldsymbol{\mu}_1 - \mathbf{w}^T \boldsymbol{\mu}_2)^2 = \mathbf{w}^T \underbrace{(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T}_{\mathbf{S}_B} \mathbf{w}$$

and the within-class variance is $\mathbf{w}^T \mathbf{S}_W \mathbf{w}$. Fisher's criterion is:

$$J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$

Taking the gradient and setting it to zero yields the generalized eigenvalue problem, whose solution is:

$$\mathbf{w}^* = \mathbf{S}_W^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$$

This is elegant: the optimal projection direction is the inverse-covariance-weighted difference of
class means. If all features are uncorrelated and equally variable, $\mathbf{S}_W^{-1}$ is the
identity, and you simply point in the direction between the two means. If some features are highly
variable within classes, $\mathbf{S}_W^{-1}$ down-weights those features. The covariance tells you
which differences in means are *actually meaningful* versus those swamped by within-class noise.

> 💡 **Intuition:** Imagine two clouds of points in 2D. You want to find a 1D line to project them
> onto such that the projected blobs are as far apart as possible *relative to their individual
> spreads*. Fisher's criterion is exactly the signal-to-noise ratio of this projection — a wide
> street (large between-class distance) divided by narrow sidewalks (small within-class spread).
> $\mathbf{S}_W^{-1}$ stretches the space to make all within-class directions equally "expensive"
> to move along — then you point toward the stretched-mean difference.

### Fisher's Original Algorithm (2-Class Case)

1. Compute class means: $\boldsymbol{\mu}_1 = \frac{1}{n_1}\sum_{i \in C_1} \mathbf{x}_i$, $\boldsymbol{\mu}_2 = \frac{1}{n_2}\sum_{i \in C_2} \mathbf{x}_i$
2. Compute within-class scatter: $\mathbf{S}_W = \sum_{k=1}^{2} \sum_{i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$
3. Compute optimal projection: $\mathbf{w}^* = \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$
4. Project all points: $z_i = (\mathbf{w}^*)^T \mathbf{x}_i$
5. Classify by threshold: $z_i > \theta$ implies class 1, else class 2, where $\theta = \frac{(\mathbf{w}^*)^T(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2)}{2}$

### The Multiclass Generalization: Rao (1948)

Fisher's 1936 paper solved only the 2-class case. The generalization to K classes — what sklearn implements — came from C.R. Rao, working at the Indian Statistical Institute:

> 📜 **Origin/Citation:** Rao, C.R. (1948). "The Utilization of Multiple Measurements in Problems
> of Biological Classification." *Journal of the Royal Statistical Society, Series B*, 10(2),
> 159–203. DOI: https://doi.org/10.1111/j.2517-6161.1948.tb00008.x

Rao's generalization replaces the single between-class scatter term with the full matrix:

$$\mathbf{S}_B = \sum_{k=1}^{K} n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$$

where $\boldsymbol{\mu}$ is the global mean. The generalized Fisher criterion is:

$$J(\mathbf{W}) = \frac{|\mathbf{W}^T \mathbf{S}_B \mathbf{W}|}{|\mathbf{W}^T \mathbf{S}_W \mathbf{W}|}$$

where $|\cdot|$ denotes the matrix determinant and $\mathbf{W}$ is now a matrix of projection
vectors. The solution is the set of generalized eigenvectors satisfying:

$$\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w} \quad \Longleftrightarrow \quad \mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}$$

The eigenvectors with the K-1 largest eigenvalues are the **linear discriminant directions**. The K-1 ceiling is a hard mathematical fact: K class means span at most a (K-1)-dimensional affine subspace (they live in a simplex), so $\mathbf{S}_B$ has rank at most K-1, meaning at most K-1 nonzero eigenvalues.

> 💡 **Intuition:** K class means in $\mathbb{R}^p$ form a simplex with K-1 degrees of freedom.
> LDA projects onto exactly those K-1 directions — the minimum-dimensional subspace capturing all
> between-class spread. Any additional dimensions are mathematically guaranteed to contain zero
> between-class signal.

### Historical Timeline

| Year | Event |
|---|---|
| 1936 | Fisher publishes 2-class discriminant analysis using the Iris dataset |
| 1948 | Rao generalizes to K classes; introduces scatter matrix formalism |
| 1960s–70s | LDA codified in Duda & Hart (1973) as the linear Bayes classifier |
| 1989 | Friedman introduces regularized discriminant analysis — LDA and QDA as a continuum |
| 1997 | Belhumeur et al. — Fisherfaces; reveal the small-sample-size problem |
| 2004 | Ledoit-Wolf provides analytic optimal shrinkage, making auto-regularization practical |
| 2024 | Nature Reviews Methods Primers: Zhao et al. comprehensive modern LDA review |
| 2026 | arxiv 2604.03928: LDA on frozen CNN features improves accuracy by up to 4.6pp |

---

## §2 — The Algorithm, Deeply Explained

### 2.1 The Gaussian Generative Model

LDA is most cleanly understood not from the optimization angle (Fisher's original approach) but
from the probabilistic angle introduced by subsequent statisticians. These two perspectives give
the same answer, and understanding both deepens your intuition.

**The probabilistic model:** Assume that each class $k$ generates data according to a multivariate Gaussian, and crucially, *all classes share the same covariance matrix* $\boldsymbol{\Sigma}$:

$$P(\mathbf{x} \mid y = k) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma})$$

$$P(y = k) = \pi_k \quad \text{(class priors, } \sum_k \pi_k = 1\text{)}$$

By Bayes' theorem, the posterior probability of class $k$ given observation $\mathbf{x}$ is:

$$\log P(y = k \mid \mathbf{x}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2} \boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log \pi_k + \text{const}$$

Notice that the quadratic term $-\frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x}$ appears
in every class's log-posterior and therefore cancels when you compare classes. The remaining
expression is **linear in $\mathbf{x}$** — that is the source of the word "linear" in Linear
Discriminant Analysis.

The decision boundary between class $j$ and class $k$ is the set of $\mathbf{x}$ where
$\log P(y=j|\mathbf{x}) = \log P(y=k|\mathbf{x})$, which simplifies to:

$$(\boldsymbol{\mu}_j - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} \mathbf{x} = \frac{1}{2}(\boldsymbol{\mu}_j^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_j - \boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k) - \log\frac{\pi_j}{\pi_k}$$

This is a hyperplane in $\mathbb{R}^p$ — a linear boundary. With equal priors and equal class
sizes, the boundary passes through the midpoint between $\boldsymbol{\mu}_j$ and $\boldsymbol{\mu}_k$ in the Mahalanobis metric.

> 💡 **Intuition:** LDA is the Bayes-optimal classifier when your generative model is
> "Gaussian blobs with the same shape but different centers." The decision boundary is placed
> exactly where Bayes' rule places it under this model. Fisher arrived at the same answer from a
> purely geometric optimization perspective — a beautiful convergence of two different lines of
> thought.

### 2.2 The Mahalanobis Distance Connection

Assign $\mathbf{x}$ to the class $k$ maximizing:

$$\delta_k(\mathbf{x}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2} \boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log \pi_k$$

With equal priors, this is equivalent to assigning to the nearest class mean in
**Mahalanobis distance**: $\arg\min_k (\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)$.

Mahalanobis distance adjusts for correlations between features. If two features are highly
correlated, moving along that direction gives very little new information — the Mahalanobis
metric accounts for this by stretching space inversely to the covariance. This is why LDA
naturally handles correlated features: it never treats them as independent.

### 2.3 The Scatter Matrices in Detail

**Within-class scatter matrix $\mathbf{S}_W$:**

$$\mathbf{S}_W = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$$

Note: sklearn computes this as $\sum_k \pi_k \mathbf{C}_k$ where $\pi_k$ is the class prior and
$\mathbf{C}_k$ is the class-$k$ sample covariance. This weights larger classes more heavily.

**Between-class scatter matrix $\mathbf{S}_B$:**

$$\mathbf{S}_B = \sum_{k=1}^{K} n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$$

$\mathbf{S}_B$ has rank at most $\min(K-1, p)$. Its column space is exactly the affine subspace
spanned by the class mean differences from the global mean.

**Total scatter:** $\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B$. The within-class and between-class
scatter partition the total scatter — a useful identity for verification.

**The generalized eigenvalue problem:**

$$\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$$

equivalently (when $\mathbf{S}_W$ is invertible):

$$\mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}$$

The eigenvectors $\mathbf{w}_1, \ldots, \mathbf{w}_{K-1}$ sorted by decreasing eigenvalue $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_{K-1}$ form the **linear discriminant directions**. Each eigenvalue $\lambda_j$ equals the Fisher ratio (between-class / within-class variance) along direction $\mathbf{w}_j$.

The explained-between-class variance for component $j$ is:

$$\text{EVR}_j = \frac{\lambda_j}{\sum_{i=1}^{K-1} \lambda_i}$$

This is what `lda.explained_variance_ratio_` returns in sklearn.

### 2.4 The SVD Implementation (sklearn's Default)

When you call `LinearDiscriminantAnalysis(solver='svd')`, sklearn does *not* compute $\mathbf{S}_W^{-1}\mathbf{S}_B$ directly. Why? Because when $n < p$, $\mathbf{S}_W$ is singular and cannot be inverted. The SVD solver avoids this by working with a whitened representation.

**SVD algorithm (conceptually):**

1. **Center within classes:** For each class $k$, subtract $\boldsymbol{\mu}_k$ from each sample.
2. **Stack and weight:** Build the matrix $\tilde{\mathbf{X}}$ where class $k$'s rows are scaled by $\sqrt{\pi_k}$.
3. **Thin SVD of $\tilde{\mathbf{X}}$:** $\tilde{\mathbf{X}} = \mathbf{U} \mathbf{D} \mathbf{V}^T$. The columns of $\mathbf{V}$ span the column space of $\mathbf{S}_W$.
4. **Sphere the class means:** Project class means into the whitened space: $\mathbf{M}_k = \mathbf{D}^{-1} \mathbf{V}^T \boldsymbol{\mu}_k$.
5. **Thin SVD of sphered means:** The left singular vectors give the discriminant directions in the whitened space.
6. **Map back:** Compose the two SVDs to get the projection matrix $\mathbf{W} \in \mathbb{R}^{p \times (K-1)}$.

The `transform()` call then computes: $\mathbf{Z} = (\mathbf{X} - \bar{\mathbf{x}}) \mathbf{W}$

where $\bar{\mathbf{x}}$ is the global mean (stored as `lda.xbar_`).

> 🔧 **In practice:** You never need to call this algorithm manually. Just know that SVD gives
> numerically identical results to the direct eigendecomposition approach when $\mathbf{S}_W$ is
> invertible, but remains numerically stable when $\mathbf{S}_W$ is ill-conditioned. The `tol`
> parameter controls the threshold for discarding near-zero singular values.

### 2.5 Geometric Intuition: What Is LDA Doing in Data Space?

Here is the mental model to carry with you. Start with your data in $\mathbb{R}^p$. The class
means $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K$ are K points in this space. They define a
$(K-1)$-dimensional affine subspace (a plane, line, or hyperplane depending on K). LDA finds the
projection from $\mathbb{R}^p$ onto this subspace that is *optimal* in the sense of maximizing
the Fisher ratio.

But LDA does not project naively onto the subspace of means. It first "spheres" or "whitens" the
within-class distribution using $\mathbf{S}_W$. In the whitened space, all within-class
variation becomes isotropic (unit sphere). In this whitened space, LDA then projects onto the
directions connecting the class means. This two-step process — whiten, then project onto
mean-differences — is the complete geometric description of LDA.

> 💡 **Intuition (the two-step picture):**
> Step 1: Stretch the data so that the within-class distribution becomes a unit sphere. This makes
> all features equally "noisy" regardless of their original scale or correlation.
> Step 2: In this sphered space, find the directions in which the class means are maximally
> separated. These are Fisher's discriminant directions.
> Un-stretch back to original space. That is your projection.

### 2.6 Implicit Assumptions and Where They Break

LDA makes three assumptions. Understanding when they hold — and when they fail — is the most
practical knowledge in this section.

**Assumption 1: Gaussian within-class distributions.**

If your within-class distributions are heavy-tailed (Student-t), skewed (log-normal), or
multimodal, the Bayes-optimal boundary is no longer linear. LDA's boundary will be a
misspecification.

*When it matters:* Bimodal within-class distributions (one class has two sub-populations) cause
LDA to average the two modes, creating a mean that represents neither. Heavy-tailed features make
LDA sensitive to outliers.

*When it doesn't matter much:* Mild skew, light tails, and non-Gaussianity with moderate n are
generally tolerable. LDA is known to be surprisingly robust — many practitioners use it
successfully on binary features, count data, and mild skew.

*Detection:* Q-Q plots per feature per class; Mardia's test (`pingouin.multivariate_normality()`).

**Assumption 2: Equal covariance matrices across all classes.**

If class $j$ has a very different covariance structure from class $k$ (e.g., one is spread along
feature 1, the other along feature 2), the pooled $\mathbf{S}_W$ is a misleading average of both.
Decision boundaries based on this pooled covariance will be biased.

*Detection:* Fit QDA (per-class covariance) and compare cross-validated accuracy to LDA. If QDA
is significantly better, the equal-covariance assumption is violated.

*Mitigation:* Use QDA. Or use regularized QDA (`QuadraticDiscriminantAnalysis(reg_param=0.1)`).

**Assumption 3: Linearity.**

LDA can only create linear decision boundaries. If the true boundary is curved (e.g., one class
is a sphere surrounded by the other), LDA will fail no matter how well the other assumptions hold.

*Detection:* Visualize in 2D using the LDA projection itself. If the projected classes are not
linearly separable, the original data likely requires a nonlinear method.

*Mitigation:* Kernel LDA (see §3), QDA, SVM with nonlinear kernel, gradient boosting.

### 2.7 Computational Complexity

| Operation | Solver | Complexity |
|---|---|---|
| Fitting (SVD) | SVD | $O(n p^2)$ for scatter matrices; $O(p^2 K)$ for SVD steps |
| Fitting (Eigen) | Eigen | $O(n p^2)$ scatter matrices + $O(p^3)$ eigendecomposition |
| Fitting (LSQR) | LSQR | $O(n p^2)$ scatter matrices; fast for classification |
| Transform | Any | $O(n_{\text{new}} \cdot p \cdot (K-1))$ — matrix multiply |
| Predict | Any | $O(n_{\text{new}} \cdot p \cdot K)$ — compute linear scores per class |

**Practical implications:**
- For $p \leq 1000$, fitting is essentially instantaneous.
- For $p > 10^4$, the $O(np^2)$ scatter matrix computation dominates. Consider PCA preprocessing first.
- The transform step is always cheap once the model is fitted.
- QDA fitting is $O(K \cdot n_k \cdot p^2)$ — scales poorly with both K and p.

> 🔬 **Deep dive:** The $O(p^3)$ eigendecomposition in the eigen solver can become the bottleneck
> for large $p$. The SVD solver avoids the full $p \times p$ eigendecomposition by working with
> the $n \times K$ matrix of whitened class means instead — making its scaling dominated by
> $O(np^2)$ for the scatter matrix, then $O(n K^2)$ for the smaller SVD. Since $K \ll p$ in
> practice, this is significantly faster for large-$p$ problems.

---

## §3 — The Full Evolution: Major Variants of LDA

The classic LDA has inspired a rich family of variants, each targeting a specific failure mode of
the original. Here is the full lineage.

### 3.1 Quadratic Discriminant Analysis (QDA)

**Problem it solves:** Classic LDA assumes all classes share one covariance matrix $\boldsymbol{\Sigma}$. When class covariances genuinely differ — one class is elongated along a different axis than another — LDA misspecifies the decision boundary.

**Key algorithmic change:** Allow each class its own covariance matrix $\boldsymbol{\Sigma}_k$. The log-posterior becomes:

$$\log P(y=k|\mathbf{x}) = -\frac{1}{2}\log|\boldsymbol{\Sigma}_k| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) + \log \pi_k$$

Now the $-\frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}_k^{-1} \mathbf{x}$ terms differ per class and do NOT cancel. The decision boundary is a **quadratic surface** (hence QDA).

**Cost:** QDA requires estimating K covariance matrices instead of one. Parameter count goes from $p(p+1)/2$ to $K \cdot p(p+1)/2$. For $K=10$, $p=100$, that is 10x more parameters. QDA is reliable only when $n_k \gg p$ for all $k$.

**When to prefer QDA:**
- Per-class sample sizes $n_k > 5p$ (very rough rule)
- Visual inspection shows classes with different shapes in scatter plots
- Cross-validated QDA accuracy significantly exceeds LDA accuracy

**Python:**
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis(reg_param=0.1)  # reg_param=0.0 is pure QDA
qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)
```

> 📜 **Origin/Citation:** QDA was already implicit in the Gaussian model setup. Formal treatment
> in Hastie, Tibshirani & Friedman (2009), §4.3, pp. 106–119.

### 3.2 Regularized Discriminant Analysis (RDA)

**Problem it solves:** Both LDA and QDA can fail when data is limited. LDA fails when the equal-covariance assumption is violated. QDA fails when per-class sample sizes are too small. RDA provides a principled continuum between them.

> 📜 **Origin/Citation:** Friedman, J.H. (1989). "Regularized Discriminant Analysis." *Journal of
> the American Statistical Association*, 84(405), 165–175.
> DOI: https://doi.org/10.1080/01621459.1989.10478752

**Key algorithmic change:** Friedman introduced two regularization parameters:

$$\hat{\boldsymbol{\Sigma}}_k(\alpha, \gamma) = \frac{(1-\gamma)\left[(1-\alpha)\hat{\boldsymbol{\Sigma}}_k + \alpha \hat{\boldsymbol{\Sigma}}\right] + \gamma \frac{\text{tr}(\hat{\boldsymbol{\Sigma}})}{p} \mathbf{I}}{1}$$

- $\gamma \in [0,1]$: interpolates between QDA ($\gamma=0$, per-class covariance) and LDA ($\gamma=1$, pooled covariance)
- $\alpha \in [0,1]$: shrinks the covariance toward a scaled identity matrix ($\alpha=1$ is nearest-centroid)

**When to prefer RDA:** When you are uncertain whether the equal-covariance assumption holds, or when per-class $n_k$ is moderate. Tune $\gamma$ and $\alpha$ by cross-validation.

**In sklearn:** The closest approximation is `QuadraticDiscriminantAnalysis(reg_param=alpha)`, which implements the $\alpha$ shrinkage. The full $(\alpha, \gamma)$ parameterization is not directly available in sklearn — for that you need a custom implementation or `mlxtend`.

### 3.3 Shrinkage LDA (Ledoit-Wolf and OAS)

**Problem it solves:** When $n_{\text{per class}} < p$, the empirical within-class covariance matrix $\hat{\mathbf{S}}_W$ is singular or near-singular. Inverting it amplifies estimation noise catastrophically.

> 📜 **Origin/Citation (Ledoit-Wolf):** Ledoit, O. and Wolf, M. (2004). "A Well-Conditioned
> Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*,
> 88(2), 365–411. DOI: https://doi.org/10.1016/S0047-259X(03)00096-4

**Key algorithmic change:** Replace $\hat{\mathbf{S}}_W$ with a shrinkage estimator:

$$\hat{\boldsymbol{\Sigma}}_{\text{shrunk}} = (1 - \rho) \hat{\boldsymbol{\Sigma}}_{\text{empirical}} + \rho \frac{\text{tr}(\hat{\boldsymbol{\Sigma}})}{p} \mathbf{I}$$

Ledoit-Wolf provides the analytically optimal $\rho^*$ — computable from the data without cross-validation. OAS (Oracle Approximating Shrinkage) uses a different analytic formula that typically has lower MSE when the data is Gaussian.

**In sklearn:**
```python
# Ledoit-Wolf shrinkage (analytic optimal rho)
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

# Or equivalently with OAS:
from sklearn.covariance import OAS
lda = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=OAS())

# For regularization + transform():
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=K-1)
```

**When to prefer:** Whenever $n_{\text{per class}} < 5p$, always use shrinkage. For high-dimensional data ($p > 100$), shrinkage is almost always beneficial.

### 3.4 The PCA + LDA Pipeline (Fisherfaces)

**Problem it solves:** The "small sample size" problem — when $n < p$, the within-class scatter matrix $\mathbf{S}_W$ is rank-deficient (singular). The eigen solver cannot invert it.

> 📜 **Origin/Citation:** Belhumeur, P.N., Hespanha, J.P. and Kriegman, D.J. (1997).
> "Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection." *IEEE
> Transactions on Pattern Analysis and Machine Intelligence*, 19(7), 711–720.
> DOI: https://doi.org/10.1109/34.598228

**Key algorithmic change:** First project data to a lower-dimensional PCA subspace (keeping at most $n - K$ components to ensure $\mathbf{S}_W$ is non-singular in the PCA space), then apply LDA in that subspace.

**Why it works:** The null space of $\mathbf{S}_W$ (directions with zero within-class variance) contains no useful discriminant information — *by definition*, no within-class variance means all classes look identical along those directions. Projecting out this null space first via PCA removes the problem without losing any discriminant signal.

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Classic Fisherfaces pipeline
fisherfaces = Pipeline([
    ('pca', PCA(n_components=min(n_train - n_classes, n_features), whiten=True)),
    ('lda', LinearDiscriminantAnalysis(n_components=n_classes - 1))
])
fisherfaces.fit(X_train, y_train)
```

**When to prefer:** Any high-dimensional setting where $n < p$: images, genomics, text embeddings (after dense projection), spectral data.

### 3.5 Kernel LDA (KDA)

**Problem it solves:** LDA can only create linear decision boundaries. For nonlinearly separable classes, this is a fundamental limitation.

**Key algorithmic change:** Apply the kernel trick. Map data to a high-dimensional feature space $\phi: \mathbb{R}^p \to \mathcal{H}$ implicitly via a kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$. Perform LDA in $\mathcal{H}$. The optimization is expressible entirely in terms of kernel matrix evaluations.

**In practice:** Kernel LDA is not directly in sklearn. The closest approach:
- Use `sklearn.preprocessing.KernelCenterer` + manual kernel matrix construction
- Or use `sklearn.svm.SVC` with `kernel='rbf'` for a nonlinear discriminative classifier
- For kernel LDA specifically: `mlxtend` has a `KernelLDA` or implement manually

```python
# Approximate kernel LDA via Nystrom approximation
from sklearn.kernel_approximation import Nystroem
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

kernel_lda = Pipeline([
    ('nystrom', Nystroem(kernel='rbf', gamma=0.1, n_components=100, random_state=42)),
    ('lda', LinearDiscriminantAnalysis(solver='svd'))
])
kernel_lda.fit(X_train, y_train)
```

**When to prefer:** When class boundaries are genuinely nonlinear but you want the interpretability framework of LDA. In practice, gradient boosted trees or SVMs are usually preferred for nonlinear boundaries.

### 3.6 Adaptive LDA (for Non-Stationary Data)

**Problem it solves:** In many real-time applications (EEG/BCI, sensor streams, user behavior modeling), the statistical properties of the data change over time. A static LDA trained on historical data degrades as the distribution shifts.

> 📜 **Origin/Citation:** For EEG BCI: Vidaurre, C. et al. (2011). *Towards a Cure for BCI
> Illiteracy.* Brain Topogr. 23, 194–198. Adaptive LDA for imagined syllables (2024):
> MDPI Brain Sciences, doi:10.3390/brainsci14030196.

**Key algorithmic change:** Update the LDA parameters incrementally as new labeled samples arrive. Two common strategies:
1. **Sliding window:** Refit LDA on the most recent $W$ samples only.
2. **Exponential forgetting:** Weight recent samples more heavily: $\hat{\boldsymbol{\mu}}_k \leftarrow (1-\eta)\hat{\boldsymbol{\mu}}_k + \eta \mathbf{x}_{\text{new}}$.

In sklearn, you can approximate this with `partial_fit` patterns on a sliding buffer. There is no native `partial_fit` for `LinearDiscriminantAnalysis`, but you can implement the scatter matrix update:

```python
# Adaptive LDA via incremental scatter matrix updates
# (simplified sketch — production implementations use more stable formulas)
class AdaptiveLDA:
    def __init__(self, forget_factor=0.99):
        self.lam = forget_factor  # exponential forgetting
        self.S_W = None
        self.means = {}
        self.counts = {}

    def update(self, x, label):
        if label not in self.means:
            self.means[label] = x.copy()
            self.counts[label] = 1
        else:
            # Exponentially weighted mean update
            self.means[label] = self.lam * self.means[label] + (1 - self.lam) * x
            self.counts[label] += 1
        # ... refit scatter matrices and recompute discriminant directions
```

**When to prefer:** Real-time BCI classification, stream-based fraud detection, any setting where the training distribution shifts slowly over time.

### 3.7 Sparse LDA

**Problem it solves:** In high-dimensional settings (genomics, text), it is desirable to find discriminant directions that use only a small subset of features. Standard LDA assigns nonzero weights to all features, making interpretation and feature selection difficult.

**Key algorithmic change:** Add an L1 penalty to the LDA criterion, enforcing sparsity in the discriminant weights:

$$J_{\text{sparse}}(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}} - \lambda \|\mathbf{w}\|_1$$

In practice, implemented via elastic net regularization of the LDA regression formulation. The `sklearn.linear_model.SGDClassifier` with `loss='modified_huber'` and L1 penalty is one approximation.

**In sklearn:** Not directly available. Use the `slda` package (verify availability) or reformulate as penalized regression.

### 3.8 Heteroscedastic LDA (HDA)

**Problem it solves:** In some applications (speech recognition), the within-class covariance changes across classes in a structured way — but you still want a linear boundary. HDA relaxes the equal-covariance assumption while keeping the search space manageable.

**Key idea:** Model each class covariance as a combination of a shared component and class-specific diagonal: $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}_{\text{shared}} + \mathbf{D}_k$. The HDA criterion maximizes inter-class distance while accounting for this heteroscedastic noise.

**Python:** Not in sklearn. Specialized implementations exist in speech/audio processing libraries.

### Summary: Which Variant When?

| Variant | Use When | Python |
|---|---|---|
| Classic LDA | $n \gg p$, classes share covariance, need simple projection | `sklearn.LinearDiscriminantAnalysis` |
| QDA | $n_k \gg p$ for all $k$, covariances clearly differ | `sklearn.QuadraticDiscriminantAnalysis` |
| Shrinkage LDA (LW/OAS) | $n_{\text{per class}} < 5p$, high-d tabular | `solver='lsqr'`, `shrinkage='auto'` |
| PCA + LDA (Fisherfaces) | $n < p$, images, genomics | `Pipeline([PCA, LDA])` |
| Kernel LDA | Nonlinear boundaries needed | Nystroem + LDA pipeline |
| Adaptive LDA | Non-stationary streams, EEG/BCI | Custom incremental implementation |
| Sparse LDA | Feature selection needed, interpretability | Custom or specialized packages |

---

## §4 — Hyperparameters: The Complete Guide

LDA has fewer tunable hyperparameters than most ML algorithms — but the ones it has are
surprisingly subtle. The interaction between `solver`, `shrinkage`, and `n_components` trips up
experienced practitioners regularly.

### 4.1 `solver` — The Algorithmic Engine

**What it controls:** Which numerical algorithm solves for the discriminant directions.

| Solver | Supports `shrinkage` | Supports `transform()` | When to use |
|---|---|---|---|
| `'svd'` (default) | No | Yes | Default. Most numerically stable. Works when $n < p$. Large $p$. |
| `'lsqr'` | Yes | No | Need regularization, classification only (no transform needed). |
| `'eigen'` | Yes | Yes | Need both regularization AND `transform()`. |

**Decision flowchart:**
1. Do you need `transform()` (DR)? If yes → `svd` or `eigen`.
2. If yes to DR and need shrinkage → `eigen`.
3. If no to DR and need shrinkage → `lsqr` (faster than eigen for classification).
4. No shrinkage, need DR → `svd` (fastest, most stable).

**No Optuna needed:** `solver` is a structural decision based on your requirements, not a value to search over.

> ⚠️ **Pitfall:** The most common error: `solver='svd', shrinkage='auto'`. This raises a
> `ValueError`. The SVD solver does not compute an explicit covariance matrix and therefore
> cannot apply shrinkage to it. Always switch to `'lsqr'` or `'eigen'` when you need shrinkage.

### 4.2 `shrinkage` — Regularizing the Covariance

**What it controls:** How much the empirical within-class covariance is regularized toward a scaled identity matrix:

$$\hat{\boldsymbol{\Sigma}}_{\text{shrunk}} = (1 - \rho) \hat{\boldsymbol{\Sigma}} + \rho \cdot \frac{\text{tr}(\hat{\boldsymbol{\Sigma}})}{p} \mathbf{I}$$

- `shrinkage=None` (default): No regularization. Uses empirical covariance directly.
- `shrinkage='auto'`: Ledoit-Wolf analytic optimal $\rho^*$, computed from data without CV.
- `shrinkage=float` in $[0, 1]$: Manual specification.

**When shrinkage matters:** The ratio $n_{\text{per class}} / p$ is the key signal.

| $n_{\text{per class}} / p$ | Recommendation |
|---|---|
| > 10 | `shrinkage=None` is likely fine |
| 3–10 | Try `shrinkage='auto'`; compare CV scores |
| 1–3 | `shrinkage='auto'` strongly recommended |
| < 1 | `shrinkage='auto'` or PCA + LDA; raw LDA will be unstable |

**Too low (shrinkage → 0.0):** Uses empirical covariance. Singular or near-singular when $n < p$. Classification can collapse.

**Too high (shrinkage → 1.0):** Within-class directions all treated equally (spherical covariance). Equivalent to nearest-centroid classifier — ignores feature correlations entirely. Useful as a robust baseline.

**Auto vs. OAS:** `shrinkage='auto'` uses Ledoit-Wolf. For Gaussian data, OAS often has lower MSE:
```python
from sklearn.covariance import OAS
lda = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=OAS())
```

**Optuna search (when manual tuning is warranted):**
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    shrinkage = trial.suggest_float('shrinkage', 0.0, 1.0)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
    return cross_val_score(lda, X, y, cv=5, scoring='balanced_accuracy').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best shrinkage: {study.best_params['shrinkage']:.3f}")
```

### 4.3 `n_components` — The Dimensionality of the Output

**What it controls:** How many linear discriminant components `transform()` returns.

**Hard constraint:** `n_components <= min(n_classes - 1, n_features)`. This is a mathematical ceiling, not a software limitation. You cannot get more than K-1 components.

**This parameter does NOT affect classification.** `predict()` and `predict_proba()` always use all K-1 discriminant directions internally. `n_components` only controls the dimensionality of `transform()` output.

**Practical guidance:**
- For 2D visualization: `n_components=2`
- For 3D visualization: `n_components=3`  
- For downstream classification: use all K-1 (`n_components=None`)
- For selecting a meaningful subset: use `explained_variance_ratio_`

```python
# Select n_components by explained variance
lda = LinearDiscriminantAnalysis(solver='svd', n_components=None)
lda.fit(X_train, y_train)
cumvar = np.cumsum(lda.explained_variance_ratio_)
n_keep = int(np.searchsorted(cumvar, 0.95)) + 1
print(f"Components for 95% between-class variance: {n_keep}")

# Refit with selected n_components
lda_final = LinearDiscriminantAnalysis(solver='svd', n_components=n_keep)
lda_final.fit(X_train, y_train)
```

**Too few components:** You discard between-class discriminant information. For K=10 classes, using only 1 component discards 8 orthogonal directions of class separation.

**All K-1 components:** Usually correct for downstream classification. Only reduce further for visualization or if some components visually capture noise.

### 4.4 `priors` — Adjusting for Class Imbalance

**What it controls:** The prior probability $\pi_k$ used in the Bayes decision rule. Shifts decision boundaries without changing the discriminant directions.

**Default (`None`):** Priors are estimated from training class frequencies. If training data is 90% class 0, that class gets prior 0.9 — which biases classification toward the majority class.

**When to change priors:**
- Training data is oversampled (SMOTE) but deployment is naturally imbalanced → set priors to real-world rates
- Different classes have different misclassification costs → inflate the prior of the "expensive-to-miss" class
- Domain knowledge tells you the true prevalence differs from training distribution

```python
import numpy as np

# Medical screening: true disease prevalence 5%, training dataset is 50/50
lda = LinearDiscriminantAnalysis(
    solver='svd',
    priors=np.array([0.95, 0.05])  # [P(healthy), P(disease)]
)
lda.fit(X_train_balanced, y_train_balanced)
# Now predict() uses correct real-world priors
```

> ⚠️ **Pitfall:** Priors must sum to 1.0 exactly. A common mistake is passing counts instead of
> probabilities: `priors=np.array([450, 50])` raises a ValueError. Always normalize:
> `priors = counts / counts.sum()`.

> 💡 **Intuition:** Changing priors is equivalent to shifting the classification threshold.
> High prior for class $k$ means "unless the evidence strongly suggests otherwise, guess class $k$."
> In fraud detection, setting a low fraud prior (0.01) makes the model conservative about
> predicting fraud — appropriate when false positives are costly.

### 4.5 `store_covariance` — Accessing the Pooled Covariance

**What it controls:** Whether the (SVD solver) explicitly computes and stores the pooled within-class covariance matrix in `lda.covariance_`.

**Default:** `False`. The SVD solver does not need to form the explicit covariance matrix.

**When to set True:** When you need `lda.covariance_` for diagnostics (checking condition number, plotting covariance structure, computing Mahalanobis distances manually).

```python
lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
lda.fit(X, y)
print(f"Covariance condition number: {np.linalg.cond(lda.covariance_):.1f}")
# A condition number > 1e6 suggests near-singularity → add shrinkage
```

### 4.6 `tol` — Singular Value Threshold (SVD Solver)

**What it controls:** Singular values of the centered data matrix below `tol` are treated as zero and the corresponding dimensions are discarded before computing discriminant directions.

**Default:** `1e-4`. Usually fine.

**When to change:**
- Nearly collinear features (VIF > 100): increase to `1e-3` or `1e-2`
- Features with very small but real variance: decrease to `1e-5`
- Diagnostic: check `lda.scalings_.shape[0]` after fitting — the row count tells you how many dimensions survived the threshold

```python
lda = LinearDiscriminantAnalysis(solver='svd', tol=1e-3)
lda.fit(X, y)
print(f"Retained input dimensions: {lda.scalings_.shape[0]} of {X.shape[1]}")
```

### 4.7 `covariance_estimator` — Custom Covariance Estimation

**What it controls:** Replaces the default empirical covariance with any sklearn-compatible covariance estimator (must have `.fit()` and `.covariance_` attribute).

**Cannot be combined with `shrinkage`**. Requires `solver='lsqr'` or `solver='eigen'`.

Available estimators and when to use each:

| Estimator | When to use |
|---|---|
| `LedoitWolf()` | Same as `shrinkage='auto'`. Asymptotically optimal. |
| `OAS()` | Better MSE than LW for Gaussian data and moderate $p/n$. |
| `ShrunkCovariance(shrinkage=0.3)` | When you want to specify shrinkage manually as an estimator object. |
| `EmpiricalCovariance()` | Default behavior; no regularization. |

```python
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance

# Compare estimators
for name, est in [('LedoitWolf', LedoitWolf()), ('OAS', OAS()),
                  ('Shrunk-0.1', ShrunkCovariance(shrinkage=0.1))]:
    lda = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=est)
    score = cross_val_score(lda, X, y, cv=5, scoring='balanced_accuracy').mean()
    print(f"{name}: {score:.4f}")
```

### 4.8 QDA `reg_param` — Per-Class Covariance Regularization

For `QuadraticDiscriminantAnalysis`, the `reg_param` parameter regularizes each class covariance:

$$\hat{\boldsymbol{\Sigma}}_k^{\text{reg}} = (1 - \text{reg\_param}) \hat{\boldsymbol{\Sigma}}_k + \text{reg\_param} \cdot \mathbf{I}$$

**Default:** 0.0 (pure QDA). **Range:** [0.0, 1.0]. If any class has $n_k < p$, pure QDA will fail with a singular matrix. Start with `reg_param=0.1` and increase if needed.

### Tuning Playbook Table

| Parameter | Default | Tune? | Range | Method | Red flag |
|---|---|---|---|---|---|
| `solver` | `'svd'` | No (structural choice) | `{'svd','lsqr','eigen'}` | Logic: see §4.1 | Mismatched shrinkage/solver → ValueError |
| `shrinkage` | `None` | Yes, if $n_k < 5p$ | `[0.0, 1.0]` or `'auto'` | Start with `'auto'`; Optuna if needed | 0.0 with small $n$ → singular covariance |
| `n_components` | `None` (=K-1) | Yes, for visualization | `[1, K-1]` | Scree plot of EVR | >K-1 → error; too few → information loss |
| `priors` | `None` (from data) | Yes, for imbalanced data | Probabilities summing to 1 | Domain knowledge | Default on SMOTE-balanced data → biased |
| `tol` | `1e-4` | Rarely | `[1e-6, 1e-2]` | Collinearity diagnostic | Too large → discards real signal |
| `store_covariance` | `False` | No (diagnostic flag) | `bool` | Set True for diagnostics | — |
| `covariance_estimator` | `None` | Alternative to shrinkage | Any sklearn covariance estimator | Compare LW vs. OAS by CV | Cannot combine with shrinkage |
| QDA `reg_param` | `0.0` | Yes, if QDA is unstable | `[0.0, 1.0]` | Optuna or grid search | 0.0 with $n_k < p$ → LinAlgError |

---

## §5 — Strengths

### 5.1 It Uses Labels — The Fundamental Advantage

Every unsupervised DR method (PCA, t-SNE, UMAP, Isomap) is blind to your class structure. They
project onto directions of maximum variance, maximum neighbor preservation, or maximum connectivity
— none of which is the same as maximum class separability. LDA directly optimizes the quantity
that matters for classification: the ratio of between-class to within-class scatter.

*Mechanism:* The Fisher criterion $J(\mathbf{W}) = |\mathbf{W}^T \mathbf{S}_B \mathbf{W}| / |\mathbf{W}^T \mathbf{S}_W \mathbf{W}|$ is exactly the signal-to-noise ratio of the projection for classification. No unsupervised method optimizes this.

*Consequence:* LDA can discard dimensions that have high total variance but low discriminant power. In face recognition, lighting direction is the largest source of variance — PCA captures it. LDA ignores it because it adds noise within each person's class, not signal between persons.

### 5.2 Dimensionality Is Bounded by Problem Complexity

With K classes, you need at most K-1 dimensions to represent all between-class structure. LDA
enforces this bound automatically. You never get a 100-dimensional embedding for a 3-class problem
— you get at most 2 dimensions. This is not a limitation; it is a feature. The embedding
dimensionality scales with the complexity of the class structure, not the original feature space.

### 5.3 Built-In Classifier + Probabilistic Output

LDA gives you three outputs from a single fit:
- `transform(X)`: the low-dimensional projection (DR)
- `predict(X)`: class labels (classification)
- `predict_proba(X)`: posterior class probabilities

The probabilistic output is calibrated under the Gaussian assumption — if the assumption holds,
these are true posterior probabilities, not just scores. This is in contrast to SVMs, which
require Platt scaling to produce probabilities.

### 5.4 Speed and Scalability

Transform is a simple matrix multiply: $O(n_{\text{new}} \cdot p \cdot K)$. Fitting is $O(np^2)$
for computing scatter matrices. For typical tabular data ($n < 10^5$, $p < 10^3$), both are
essentially instantaneous. The fitted model is just a few matrices — tiny memory footprint.

### 5.5 Handles Correlated Features Naturally

Unlike naive Bayes (which assumes feature independence), LDA accounts for feature correlations
through the covariance matrix $\mathbf{S}_W$. The Mahalanobis-distance interpretation means that
correlated features are effectively treated as single informative dimensions — avoiding the
double-counting that plagues methods based on marginal distributions.

### 5.6 Excellent Regularization Story

With Ledoit-Wolf or OAS shrinkage, LDA degrades gracefully as $p/n$ grows. The analytic optimal
shrinkage coefficient means you get well-calibrated regularization without cross-validation. This
is a significant practical advantage in genomics and clinical settings where $n$ is small.

### 5.7 Interpretability

The discriminant directions $\mathbf{w}_j$ are vectors in the original feature space. Each
component of $\mathbf{w}_j$ tells you how much that feature contributes to discriminating classes
along the $j$-th direction. In medical applications, these weights can be examined by domain
experts and mapped to known biomarkers.

---

## §6 — Weaknesses & Failure Modes

### 6.1 The K-1 Component Ceiling

**Mechanism:** $\mathbf{S}_B$ has rank at most K-1. No more than K-1 discriminant directions exist.

**Failure mode:** You have 100 classes but want a 50-dimensional embedding. LDA gives you exactly 99 dimensions — which is fine. But if you want to keep only 2 dimensions for visualization with 100 classes, you are discarding 97 directions of between-class information.

**Detection:** Check `lda.explained_variance_ratio_` — if you need many components to capture 95% of between-class variance, truncation to 2D will be lossy.

**Mitigation:** For visualization with many classes, use t-SNE or UMAP on the full K-1 dimensional LDA projection instead of truncating at 2D. Use LDA to get K-1 dimensions first, then t-SNE on that.

### 6.2 Singular Within-Class Scatter (Small Sample Size Problem)

**Mechanism:** When $n_{\text{per class}} < p$, the within-class scatter matrix $\mathbf{S}_W$ is rank-deficient. Its inverse does not exist.

**Detection:** `np.linalg.matrix_rank(S_W) < p` or `np.linalg.cond(S_W) > 1e12`. The SVD solver handles this via `tol`, but the eigen solver will fail with a `LinAlgError`.

**Mitigation:**
1. Use `solver='svd'` with appropriate `tol`.
2. Add shrinkage: `solver='eigen', shrinkage='auto'`.
3. PCA preprocessing: `Pipeline([PCA(n_components=n_train - n_classes), LDA])`.

### 6.3 Gaussian and Equal-Covariance Assumptions

**Mechanism:** LDA is the Bayes-optimal classifier *only* under Gaussian class conditionals with shared covariance. When these assumptions fail:
- Heavy-tailed within-class distributions → decision boundary biased by outliers
- Bimodal within-class distributions → class mean is unrepresentative
- Unequal covariances → pooled covariance averages meaninglessly

**Detection:**
- Q-Q plots per feature per class for normality
- Compare LDA vs. QDA cross-validated accuracy — large gap (>5%) indicates unequal covariance
- Box's M test (via `pingouin`) for covariance homogeneity
- Visualize class-conditional distributions with `sns.kdeplot` per feature per class

**Mitigation:**
- Non-Gaussian: transform features (log, Box-Cox, QuantileTransformer)
- Unequal covariance: use QDA or regularized QDA
- Multimodal within-class: split into sub-classes, or use mixture-model-based DA

### 6.4 Linear Decision Boundaries Only

**Mechanism:** LDA can only produce hyperplanar decision boundaries. Any genuinely nonlinear class structure will be mishandled.

**Classic failure case:** Two concentric rings (one class is a donut surrounding the other). The class means are both near the center, $\mathbf{S}_B$ is near-zero, and LDA produces a random projection with no discriminant power.

**Detection:** If `lda.score(X_test, y_test)` is near the majority-class baseline despite the classes being visually separable in the original space, linearity is the issue.

**Mitigation:** Kernel LDA, SVM with nonlinear kernel, gradient boosted trees, neural networks.

### 6.5 Sensitivity to Feature Scaling

**Mechanism:** $\mathbf{S}_W$ is computed in the original feature space. Features measured in different units (e.g., age in years vs. income in dollars) will dominate the scatter matrices simply by having larger numerical values.

**Detection:** Check feature variances — if max/min variance ratio exceeds 100, unscaled LDA is unreliable.

**Mitigation:** Always `StandardScaler()` before LDA. This is a mandatory preprocessing step, not optional.

> ⚠️ **Pitfall:** The most common preprocessing mistake in production LDA pipelines: fitting the
> scaler on the full dataset before train/test split. Always use a `Pipeline`:
> `Pipeline([('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis())])`.

### 6.6 Class Imbalance and Prior Misspecification

**Mechanism:** Default priors are estimated from training data. If training data is artificially balanced (e.g., via SMOTE) but deployment data is naturally imbalanced, LDA decision boundaries will be miscalibrated.

**Detection:** Large discrepancy between training class proportions and expected deployment proportions.

**Mitigation:** Set `priors=` to reflect the real deployment distribution. Or use `predict_proba()` with an external threshold calibrated by business costs.

### 6.7 It Cannot Extract Nonlinear Features

**Mechanism:** LDA is a linear projection — the output is a linear function of the inputs. It cannot discover curved manifolds, local cluster structure, or nonlinear interaction effects.

**Consequence:** If the useful information in your data is encoded in nonlinear interactions (e.g., $x_1 \cdot x_2$ distinguishes classes, but $x_1$ and $x_2$ individually do not), LDA will miss it entirely.

**Mitigation:** Feature engineering (explicit nonlinear features), kernel LDA, or use UMAP for nonlinear supervised DR.

### Weakness Summary Table

| Weakness | Mechanism | Detection | Mitigation |
|---|---|---|---|
| K-1 component ceiling | Rank of $\mathbf{S}_B$ | EVR plot | Accept ceiling or use LDA+t-SNE |
| Singular $\mathbf{S}_W$ | $n < p$ | Condition number | SVD solver, shrinkage, PCA+LDA |
| Gaussian assumption | Generative model mismatch | Q-Q plots, Mardia test | Transform features, use QDA |
| Equal covariance assumption | Pooled covariance distorts boundaries | LDA vs QDA CV comparison | Use QDA or RDA |
| Linearity only | Hyperplanar boundaries only | Accuracy vs. majority baseline | Kernel LDA, tree methods |
| Scale sensitivity | Scatter matrices scale-dependent | Variance ratio check | Always StandardScaler first |
| Prior misspecification | Training ≠ deployment distribution | EVS pipeline audit | Set `priors=` explicitly |

---

## §7 — What I Think About When Fitting

This section is my internal monologue when I am about to fit an LDA model. Run through this
checklist before every fit, during fitting (watch for warnings), and after (run the diagnostics).

### Before Fitting

**1. Count K and compute max components.**
K classes → K-1 maximum components. If K=2 and I am hoping for a 2D visualization: I need a
different method. If K=10 and I want a 5D embedding: fine. I check this before writing any code.

**2. Compute the $n_{\text{per class}} / p$ ratio.**
I look at `y.value_counts()` and `X.shape[1]`. If the minimum per-class count divided by $p$ is
less than 5, I immediately go to `solver='eigen', shrinkage='auto'` or a PCA+LDA pipeline. I do
not debate this — singular covariance is painful to debug after the fact.

**3. Check class balance explicitly.**
Are the training proportions representative of deployment? If I trained on SMOTE-balanced data, I
need to set `priors=` to the real deployment rates. I ask this question at every project start.

**4. Check feature scaling.**
I run `X.describe()` and look at the std row. If the ratio of max to min std exceeds about 10,
I StandardScale. I always StandardScale unless there is a specific reason not to (e.g., all
features are proportions on the same scale). LDA without scaling on mixed-unit data is wrong.

**5. Spot-check Gaussianity.**
I plot `sns.histplot` for each feature per class. Not a formal test — just looking for severe
bimodality or extreme skew. If I see it, I try `np.log1p` or `QuantileTransformer`. I run
`pingouin.multivariate_normality(X_class_k)` if the stakes are high enough to care about the
formal test result.

**6. Check covariance homogeneity.**
I compute the per-class correlation matrices (for small $p$) and see if they look qualitatively
similar. If one class has a strong positive correlation between features A and B and another class
has a negative one, the pooled covariance is meaningless and I should use QDA.

**7. Decide: am I using LDA as a classifier, a projector, or both?**
This dictates the solver. Projection only: `svd` is fine. Classification only with regularization:
`lsqr`. Both: `eigen`. The API forces this choice — I make it consciously before fitting.

### During Fitting — Signals to Watch

**Warning: collinear features.** If sklearn emits a `CollinearityWarning` or similar, increase
`tol` to `1e-3`. Check `lda.scalings_.shape[0]` — fewer rows than features means dimensions were
dropped.

**Warning: singular matrix with eigen solver.** This means $\mathbf{S}_W$ is rank-deficient. Fix:
switch to `shrinkage='auto'` or use SVD solver. Do not suppress the warning.

**Unexpected `ValueError: n_components cannot be larger than...`** I forgot the K-1 ceiling.
Fix: `n_components = min(n_components_desired, n_classes - 1)`.

**`NotFittedError` on `transform()` with `lsqr` solver.** I am trying to call `transform()` on
an LSQR model. Fix: switch to `solver='eigen'`.

### After Fitting — Diagnostics I Always Run

**1. Explained variance ratio.**
```python
print(lda.explained_variance_ratio_)
```
If the first component captures >95% of between-class variance for a K>3 problem, the classes are
essentially 1D-separable — a simple threshold might work as well as LDA. If the ratios are
roughly equal across all K-1 components, the class structure is genuinely K-1-dimensional.

**2. Scatter plot in LD space.**
I always plot `X_lda[:, 0]` vs `X_lda[:, 1]` colored by class. If I cannot see reasonably
separated clusters, LDA is not working — either the assumption is violated, the data is not
linearly separable, or I have preprocessing issues. Clean cluster separation is the primary
sanity check.

**3. LDA vs. QDA cross-validated accuracy.**
```python
score_lda = cross_val_score(lda, X, y, cv=5).mean()
score_qda = cross_val_score(QDA, X, y, cv=5).mean()
```
If QDA is more than 3–5% better, the equal-covariance assumption is likely violated.

**4. Fisher ratio in projected space.**
I compute $B/W$ — between-class scatter divided by within-class scatter in the projected space.
I want this to be much greater than 1. If it is close to 1, LDA is not achieving much separation.

**5. Compare projected features to raw features with a downstream classifier.**
```python
pipe_lda = Pipeline([('lda', lda), ('knn', KNeighborsClassifier())])
pipe_raw = Pipeline([('knn', KNeighborsClassifier())])
```
If `pipe_lda` performs comparably or better with K-1 features instead of p features, LDA is
compressing without information loss.

### Decision Tree: If I See X, I Do Y

| What I observe | What I do |
|---|---|
| `ValueError: shrinkage is not supported with 'svd'` | Switch to `solver='lsqr'` or `solver='eigen'` |
| `ValueError: n_components > min(n_classes-1, n_features)` | `n_components = min(wanted, K-1)` |
| Singular matrix error with eigen solver | Add `shrinkage='auto'` or switch to SVD |
| QDA accuracy >> LDA accuracy (>5%) | Use QDA; classes have different covariance structures |
| LD1 explains >95% variance | Classes nearly 1D-separable; 2D visualization will show almost 1 axis |
| Overlapping clusters in LD scatter plot | Check: normality, covariance homogeneity, linearity; consider kernel LDA or tree methods |
| LDA worse than raw features on downstream classifier | LDA is discarding useful non-discriminant structure; unusual — check for data leakage or preprocessing errors |
| Very different class sizes in scatter plot | Set `priors=` to reflect deployment rates |

### Common "Smell Tests"

A well-fitted LDA should show:
- Clean, roughly Gaussian blobs in LD scatter plots, not chaotic overlapping clouds
- `explained_variance_ratio_` values that decrease — LD1 should always explain more variance than LD2
- Cross-validated accuracy above the majority-class baseline by a meaningful margin
- Fisher ratio in projected space significantly greater than 1
- Consistent results across different random seeds (LDA is deterministic — if results vary, you have a data leakage or CV bug)

---

## §8 — Diagnostic Plots & Evaluation

Here is the complete set of diagnostics I run for every LDA model, with code.

### 8.1 Scatter Plot in Discriminant Space (Primary Diagnostic)

**What it shows:** The projection of all samples onto the first two linear discriminant axes. This
is the direct visualization of what LDA is doing.

**How to read it:** Well-separated, roughly Gaussian, roughly equal-spread blobs = LDA assumptions
satisfied, model working. Overlapping blobs = classes not linearly separable in this space, or
assumptions violated. One blob much more spread than others = unequal covariance (consider QDA).

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

data = load_iris()
X, y = data.data, data.target
class_names = data.target_names

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

lda = LinearDiscriminantAnalysis(solver='svd', n_components=2)
X_lda = lda.fit_transform(X_sc, y)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#E74C3C', '#2ECC71', '#3498DB']
for i, (cls, name) in enumerate(zip(np.unique(y), class_names)):
    mask = y == cls
    ax.scatter(X_lda[mask, 0], X_lda[mask, 1], c=colors[i],
               label=name, alpha=0.8, edgecolors='white', linewidths=0.5, s=80)

# Plot class mean positions
for i, (cls, name) in enumerate(zip(np.unique(y), class_names)):
    mean = X_lda[y == cls].mean(0)
    ax.scatter(*mean, c=colors[i], marker='*', s=300, edgecolors='black', linewidths=1.5, zorder=5)

ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}% between-class variance)")
ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}% between-class variance)")
ax.set_title("LDA Projection — Iris Dataset")
ax.legend()
plt.tight_layout()
plt.show()
```

### 8.2 Explained Variance Ratio (Scree Plot)

**What it shows:** How much between-class variance each discriminant axis captures. Analogous to PCA's scree plot, but specifically for between-class structure.

**How to read it:** A steep elbow means one or two directions carry most discriminant information. A flat spectrum means class separation is spread across many directions — you should keep more components.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar plot of EVR
axes[0].bar(range(1, len(lda.explained_variance_ratio_) + 1),
            lda.explained_variance_ratio_, color='steelblue', edgecolor='white')
axes[0].set_xlabel("Linear Discriminant Component")
axes[0].set_ylabel("Explained Between-Class Variance Ratio")
axes[0].set_title("Scree Plot — Between-Class Variance")
axes[0].set_xticks(range(1, len(lda.explained_variance_ratio_) + 1))

# Cumulative EVR
cumvar = np.cumsum(lda.explained_variance_ratio_)
axes[1].plot(range(1, len(cumvar) + 1), cumvar, 'o-', color='steelblue', linewidth=2)
axes[1].axhline(0.95, color='red', linestyle='--', label='95% threshold')
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance Ratio")
axes[1].set_title("Cumulative Between-Class Variance")
axes[1].legend()
axes[1].set_ylim([0, 1.05])
plt.tight_layout()
plt.show()

print(f"EVR per component: {lda.explained_variance_ratio_.round(4)}")
print(f"Cumulative EVR:    {cumvar.round(4)}")
```

### 8.3 Per-Class Covariance Eigenvalue Spectra (Assumption Check)

**What it shows:** The shape of each class's distribution. If all classes have similar eigenvalue spectra, the equal-covariance assumption holds. Divergent spectra → use QDA.

```python
fig, axes = plt.subplots(1, len(np.unique(y)), figsize=(12, 4))
for k, (cls, name) in enumerate(zip(np.unique(y), class_names)):
    X_k = X_sc[y == cls]
    cov_k = np.cov(X_k.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov_k))[::-1]
    axes[k].semilogy(eigvals, 'o-', color=colors[k])
    axes[k].set_title(f"{name}\nEigenvalue Spectrum")
    axes[k].set_xlabel("Component")
    axes[k].set_ylabel("Eigenvalue (log scale)")
plt.suptitle("Per-Class Covariance Eigenvalue Spectra\n(Similar shapes → LDA assumption holds)")
plt.tight_layout()
plt.show()
```

### 8.4 Fisher Ratio in Projected Space

**What it shows:** The actual between-class / within-class scatter ratio that LDA maximized. This confirms the optimization is working.

```python
X_proj = lda.transform(X_sc)

# Between-class scatter: variance of class means
class_means_proj = np.array([X_proj[y == k].mean(0) for k in np.unique(y)])
global_mean_proj = X_proj.mean(0)
B = sum(np.sum((y == k)) * np.dot(class_means_proj[k] - global_mean_proj,
                                   class_means_proj[k] - global_mean_proj)
        for k in range(len(np.unique(y))))

# Within-class scatter: sum of intra-class variances
W = sum(np.sum((X_proj[y == k] - class_means_proj[k])**2)
        for k in range(len(np.unique(y))))

print(f"Between-class scatter B: {B:.2f}")
print(f"Within-class scatter  W: {W:.2f}")
print(f"Fisher ratio B/W:        {B/W:.2f}  (want >> 1)")
```

### 8.5 Confusion Matrix and Classification Report

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.3,
                                                      random_state=42, stratify=y)
lda_eval = LinearDiscriminantAnalysis(solver='svd', n_components=2)
lda_eval.fit(X_train, y_train)

print(classification_report(y_test, lda_eval.predict(X_test), target_names=class_names))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(lda_eval, X_test, y_test,
                                       display_labels=class_names, ax=ax, colorbar=False)
ax.set_title("LDA Confusion Matrix")
plt.tight_layout()
plt.show()
```

### 8.6 Trustworthiness Score (for DR evaluation)

**What it shows:** Whether the 5 nearest neighbors in the original space are also close in the LDA projection. A score of 1.0 = perfect local preservation. Below 0.9 = notable distortion.

```python
from sklearn.manifold import trustworthiness

trust = trustworthiness(X_sc, X_proj, n_neighbors=5)
print(f"Trustworthiness (k=5): {trust:.4f}")
# For LDA: expect high trustworthiness between class means; may be lower within overlapping classes
```

### 8.7 LDA vs. Raw Features — Downstream Classifier Comparison

The most practical diagnostic: does LDA actually help?

```python
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_with_lda = Pipeline([
    ('lda', LinearDiscriminantAnalysis(solver='svd')),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipe_without_lda = Pipeline([
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

scores_lda = cross_val_score(pipe_with_lda, X_sc, y, cv=cv, scoring='balanced_accuracy')
scores_raw = cross_val_score(pipe_without_lda, X_sc, y, cv=cv, scoring='balanced_accuracy')

print(f"With LDA:    {scores_lda.mean():.4f} ± {scores_lda.std():.4f}")
print(f"Without LDA: {scores_raw.mean():.4f} ± {scores_raw.std():.4f}")
```

> 🏆 **Best practice:** Run this comparison on every LDA project. If LDA does not improve or
> match raw-feature performance, the assumptions may be violated or the data is not linearly
> separable. Do not use LDA blindly as preprocessing — verify that it helps.

---

## §9 — Innovative Industry Applications

### 9.1 Face Recognition — Fisherfaces (Computer Vision)

> 🏭 **Industry application:** Security systems, phone unlock, border control.

The canonical application. PCA (Eigenfaces) maximizes total variance — dominated by lighting
changes, not by person identity. LDA (Fisherfaces) maximizes inter-person variance relative to
intra-person variance under different illumination and expression.

**Why LDA is the right choice:** The supervised signal "this image is person #47" is exactly what
you want. LDA ignores lighting-induced variance (high within-person) and amplifies identity-
discriminating variance (high between-person). The Belhumeur et al. (1997) result: <5% error vs.
~25% for Eigenfaces under severe illumination change.

**The pipeline matters:** PCA first (avoid singular $\mathbf{S}_W$ when $n < p = 64 \times 64 = 4096$), then LDA.

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# n_persons = K; keep at most n_train - K PCA components for S_W to be full-rank
fisherfaces = Pipeline([
    ('pca', PCA(n_components=150, whiten=True)),
    ('lda', LinearDiscriminantAnalysis(n_components=None))  # keeps K-1 components
])
fisherfaces.fit(X_faces_train, y_person_id)
```

### 9.2 Brain-Computer Interface (BCI) — Real-Time EEG Classification

> 🏭 **Industry application:** Prosthetic limb control, assistive communication for ALS patients, neurofeedback.

LDA is the dominant classifier in motor-imagery BCI for concrete reasons: prediction in <1ms (a
matrix multiply), tiny memory footprint (deployable on microcontrollers), interpretable weights
that neuroscientists can examine, and robust performance with the small datasets that patients
can provide (typically 100–300 trials per class).

**Non-obvious insight:** The LDA discriminant weights in EEG feature space can be directly
visualized as spatial filters — a specific electrode pattern — giving neuroscientists interpretable
information about which brain regions drive the classification.

A 2024 paper (MDPI Brain Sciences, doi:10.3390/brainsci14030196) showed adaptive LDA significantly
outperforms static LDA for imagined syllable classification, because EEG signals are non-stationary
and models need incremental updating.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# EEG: small n_per_class (~200), moderate p (~50 CSP features) → shrinkage essential
bci_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
])
bci_pipeline.fit(X_csp_features, y_movement_class)
# For real-time: bci_pipeline.predict(x_new) executes in microseconds
```

### 9.3 Cancer Subtype Classification from Gene Expression

> 🏭 **Industry application:** Oncology, guiding treatment selection (e.g., PAM50 for breast cancer).

High-dimensional, small-n genomic data ($p \approx 20,000$ genes, $n \approx 500$ patients) is
LDA's most demanding setting. The key: feature selection first, then regularized LDA.

**Why LDA over deep learning here:** With 500 samples and 20,000 features, a neural network will
massively overfit. LDA with shrinkage has a controlled number of effective parameters and is
interpretable — the discriminant weights map directly to gene expression signatures that
oncologists can validate against known biology.

**The PAM50 connection:** The PAM50 breast cancer subtype classifier (which guides chemotherapy
decisions for millions of patients) uses a nearest-centroid method — effectively a shrinkage-based
version of LDA where the class centroids are shrunken toward the global mean.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 5 breast cancer subtypes → max 4 LD components
cancer_lda = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(f_classif, k=200)),        # reduce from 20k genes
    ('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=4))
])
cancer_lda.fit(X_gene_expr, y_subtype)
```

### 9.4 Financial Fraud Scoring

> 🏭 **Industry application:** Credit card fraud detection, AML transaction monitoring.

For binary fraud detection, LDA's `transform()` produces a 1-dimensional fraud score — the linear
combination of features that maximally separates fraud from non-fraud under Gaussian assumptions.
This score is directly the log-likelihood ratio between fraud and non-fraud hypotheses (under the
model). It is interpretable, auditable, and fast enough to evaluate in real time.

**Non-obvious advantage:** LDA's linear score is expressible as a SQL query:
`fraud_score = 0.23 * amount + 0.18 * velocity - 0.45 * merchant_age + ...`. This is deployable
in any system without an ML runtime, satisfying compliance requirements for model explainability.

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1% fraud rate → set priors explicitly (training data may be balanced)
lda_fraud = LinearDiscriminantAnalysis(
    solver='svd', n_components=1,
    priors=np.array([0.99, 0.01])
)
lda_fraud.fit(X_transactions, y_fraud)

# 1D fraud score: higher = more fraud-like
fraud_scores = lda_fraud.transform(X_new).ravel()

# The linear equation (deployable as SQL/rules engine):
print("Fraud score coefficients:")
print(dict(zip(feature_names, lda_fraud.scalings_[:, 0].round(4))))
```

### 9.5 NLP — Supervised Document Space Visualization

> 🏭 **Industry application:** Content recommendation, news categorization, customer feedback routing.

**Important disambiguation:** This is Fisher LDA (Linear Discriminant Analysis), NOT Latent
Dirichlet Allocation (also abbreviated LDA in NLP). They are completely different algorithms.

After extracting TF-IDF or dense embedding features for labeled documents, LDA finds the projection
that maximally separates document categories. PCA finds the directions of maximum total vocabulary
variance (dominated by high-frequency stopwords and common phrases). LDA finds directions that
discriminate the categories — directly aligned with the classification goal.

**The pipeline:** Sparse TF-IDF → TruncatedSVD (LSA, to dense) → Fisher LDA (supervised projection).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

# 6 news categories → max 5 LD components
text_lda = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, min_df=2)),
    ('lsa', TruncatedSVD(n_components=100, random_state=42)),
    ('lda', LinearDiscriminantAnalysis(n_components=5))
])
X_supervised = text_lda.fit_transform(documents, y_category)
# Now visualize in 2D or use X_supervised as input to downstream classifier
```

### 9.6 Transfer Learning Feature Refinement (Modern Innovation, 2026)

> 🏭 **Industry application:** Any fine-tuning scenario with frozen pretrained CNN or language model features.

This is the freshest application, grounded in a 2026 arxiv paper. When you extract features from
a frozen pretrained CNN (e.g., ResNet-50 gives 2048-dim features per image), those features are
optimized for ImageNet class discrimination — not for your specific downstream task. They contain
"general-purpose" variance (textures, colors, shapes) irrelevant to your classification goal.

Inserting an LDA step between the frozen feature extractor and the final classifier:
- Discards dimensions irrelevant to your target classes
- Projects to a subspace optimally aligned with your class structure
- Reduces dimensionality to K-1, preventing overfitting in the final layer

The 2026 study (arXiv:2604.03928) showed accuracy improvements of up to 4.6 percentage points
across ResNet-18, ResNet-50, MobileNetV3, and EfficientNet-B0 on CIFAR-100 and Tiny ImageNet.
LDA outperformed PCA in 7 of 8 settings.

```python
import torch
import torchvision.models as models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Extract frozen CNN features
backbone = models.resnet50(pretrained=True)
backbone.fc = torch.nn.Identity()  # remove final classification head
backbone.eval()

with torch.no_grad():
    feats_train = backbone(X_train_tensor).numpy()  # (n_train, 2048)
    feats_test  = backbone(X_test_tensor).numpy()

# Supervised refinement: project to K-1 dimensions
n_classes = len(np.unique(y_train))
lda_refine = LinearDiscriminantAnalysis(solver='svd', n_components=n_classes - 1)
feats_lda_train = lda_refine.fit_transform(feats_train, y_train)
feats_lda_test  = lda_refine.transform(feats_test)

# Final linear classifier on compressed features
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000).fit(feats_lda_train, y_train)
print(f"Accuracy with LDA refinement: {clf.score(feats_lda_test, y_test):.4f}")
```

### 9.7 Telecom Churn Prediction

> 🏭 **Industry application:** Telecom CRM, subscription retention campaigns.

LDA reduces a multi-feature churn model to a single "churn propensity score" — the optimal linear
combination of usage features (call duration, data consumption, support tickets, top-up frequency)
under Gaussian assumptions. This score is fast to compute, easy to explain to non-technical
stakeholders, and directly usable as the feature for a downstream gradient boosting model.

A 2024 study (Scientific Reports, PMC11161656) demonstrates competitive performance of LDA-based
churn models against gradient boosting when features are carefully preprocessed.

**Non-obvious application:** The LDA churn score is directly usable as a "willingness to stay"
metric that can be passed to a pricing optimization engine to determine the appropriate retention
offer for each customer.

### 9.8 Spectral Data Analysis (Chemistry, Agriculture)

> 🏭 **Industry application:** Food quality control, pharmaceutical batch release, soil analysis.

Near-infrared (NIR) spectroscopy produces hundreds of wavelength features per sample. LDA reduces
this high-dimensional spectrum to K-1 discriminant directions — separating, for example, genuine
olive oil from adulterated olive oil, or authentic Scotch whisky from impostors. LDA is standard
in chemometrics for exactly this use case: high-p, interpretable features, small n per class.

The discriminant loadings (weights per wavelength) map directly to specific chemical bonds,
giving chemists physically interpretable models.

---

## §10 — Complete Python Worked Example

We run three datasets end-to-end: **Iris** (3-class canonical), **Wine** (13 chemical features,
3 cultivars), and **Digits** (multiclass 10-class, high-dimensional, great for showing K-1 ceiling).

### 10.1 Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.manifold import trustworthiness
import optuna
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
```

### 10.2 Dataset 1: Iris (3-Class) — Classical LDA

```python
# ── Load ──────────────────────────────────────────────────────────────────
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
print(f"Iris: {X_iris.shape[0]} samples, {X_iris.shape[1]} features, {len(np.unique(y_iris))} classes")
print(f"Class distribution: {dict(zip(iris.target_names, np.bincount(y_iris)))}")
print(f"Max LDA components: {len(np.unique(y_iris)) - 1}")  # 2

# ── Preprocessing ─────────────────────────────────────────────────────────
X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(
    X_iris, y_iris, test_size=0.25, random_state=42, stratify=y_iris
)
scaler_i = StandardScaler()
X_tr_i_sc = scaler_i.fit_transform(X_tr_i)
X_te_i_sc = scaler_i.transform(X_te_i)

# ── Fit LDA ───────────────────────────────────────────────────────────────
lda_iris = LinearDiscriminantAnalysis(solver='svd', n_components=2)
lda_iris.fit(X_tr_i_sc, y_tr_i)

print(f"\nExplained between-class variance ratio:")
for i, evr in enumerate(lda_iris.explained_variance_ratio_):
    print(f"  LD{i+1}: {evr*100:.1f}%")

# ── Evaluate ──────────────────────────────────────────────────────────────
y_pred_i = lda_iris.predict(X_te_i_sc)
print(f"\nTest accuracy: {balanced_accuracy_score(y_te_i, y_pred_i):.4f}")
print(classification_report(y_te_i, y_pred_i, target_names=iris.target_names))

# ── Visualize ─────────────────────────────────────────────────────────────
X_lda_full_i = lda_iris.transform(scaler_i.transform(X_iris))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2D scatter
colors = ['#E74C3C', '#2ECC71', '#3498DB']
for i, (cls, name) in enumerate(zip(np.unique(y_iris), iris.target_names)):
    mask = y_iris == cls
    axes[0].scatter(X_lda_full_i[mask, 0], X_lda_full_i[mask, 1],
                    c=colors[i], label=name, alpha=0.8, s=70, edgecolors='white')
    # Mark class mean
    mean = X_lda_full_i[mask].mean(0)
    axes[0].scatter(*mean, c=colors[i], marker='*', s=250, edgecolors='black', zorder=5)

axes[0].set_xlabel(f"LD1 ({lda_iris.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"LD2 ({lda_iris.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("Iris — LDA Projection (2 Components)")
axes[0].legend()

# Scree plot
axes[1].bar(['LD1', 'LD2'], lda_iris.explained_variance_ratio_ * 100,
            color=['#3498DB', '#E74C3C'], edgecolor='white')
axes[1].set_ylabel("% Between-Class Variance")
axes[1].set_title("Iris — Between-Class Variance by Component")
for i, v in enumerate(lda_iris.explained_variance_ratio_ * 100):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("iris_lda.png", dpi=150, bbox_inches='tight')
plt.show()
```

**What we learn:** LD1 captures ~99% of between-class variance for Iris. The three species form
clean, well-separated clusters in 2D. Setosa is perfectly separated from the other two; Versicolor
and Virginica show slight overlap in LD2, which is expected from the original feature distributions.
The star markers show that the class means are far apart relative to within-class spreads.

### 10.3 Dataset 2: Wine (13 Features, 3 Cultivars) — Regularization + Comparison

```python
# ── Load ──────────────────────────────────────────────────────────────────
wine = load_wine()
X_wine, y_wine = wine.data, wine.target
print(f"\nWine: {X_wine.shape[0]} samples, {X_wine.shape[1]} features, {len(np.unique(y_wine))} classes")
print(f"n/p ratio: {X_wine.shape[0]/X_wine.shape[1]:.1f}  (n_per_class ≈ {X_wine.shape[0]//3}/{X_wine.shape[1]})")

X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(
    X_wine, y_wine, test_size=0.25, random_state=42, stratify=y_wine
)
scaler_w = StandardScaler()
X_tr_w_sc = scaler_w.fit_transform(X_tr_w)
X_te_w_sc = scaler_w.transform(X_te_w)

# ── Compare solvers and regularization ────────────────────────────────────
from sklearn.covariance import OAS, LedoitWolf

results = {}
configs = {
    'LDA (SVD, no shrinkage)':
        LinearDiscriminantAnalysis(solver='svd', n_components=2),
    'LDA (Eigen, LW shrinkage)':
        LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=2),
    'LDA (LSQR, OAS)':
        LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=OAS()),
    'QDA (reg=0.1)':
        QuadraticDiscriminantAnalysis(reg_param=0.1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in configs.items():
    # Classification CV
    scores = cross_val_score(
        Pipeline([('sc', StandardScaler()), ('model', model)]),
        X_wine, y_wine, cv=cv, scoring='balanced_accuracy'
    )
    results[name] = scores
    print(f"{name:35s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ── Best model: fit and project ───────────────────────────────────────────
lda_wine = LinearDiscriminantAnalysis(solver='svd', n_components=2)
lda_wine.fit(X_tr_w_sc, y_tr_w)
X_lda_w = lda_wine.transform(scaler_w.transform(X_wine))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2D projection
for i, (cls, name) in enumerate(zip(np.unique(y_wine), wine.target_names)):
    mask = y_wine == cls
    axes[0].scatter(X_lda_w[mask, 0], X_lda_w[mask, 1],
                    c=colors[i], label=name, alpha=0.8, s=70, edgecolors='white')
axes[0].set_xlabel(f"LD1 ({lda_wine.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"LD2 ({lda_wine.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("Wine — LDA Projection (2 Components)")
axes[0].legend()

# CV comparison plot
ax = axes[1]
names_short = [n.split('(')[0].strip() for n in results.keys()]
means = [v.mean() for v in results.values()]
stds  = [v.std() for v in results.values()]
bars = ax.barh(names_short, means, xerr=stds, color='steelblue', alpha=0.8, capsize=5)
ax.set_xlabel("Balanced Accuracy (5-fold CV)")
ax.set_title("Wine — Model Comparison")
ax.set_xlim([0.8, 1.01])
for bar, mean in zip(bars, means):
    ax.text(mean + 0.002, bar.get_y() + bar.get_height()/2,
            f"{mean:.4f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig("wine_lda.png", dpi=150, bbox_inches='tight')
plt.show()

# ── Discriminant loadings (feature importance) ────────────────────────────
print("\nTop features by |LD1 loading|:")
ld1_loadings = pd.Series(lda_wine.scalings_[:, 0], index=wine.feature_names)
print(ld1_loadings.abs().sort_values(ascending=False).head(5))
```

**What we learn:** Wine is a clean, nearly-linearly-separable 3-class problem. All LDA variants
achieve >98% balanced accuracy with 5-fold CV. The SVD (no shrinkage) and eigen (LW shrinkage)
variants perform comparably — because $n/p = 178/13 \approx 14$, which is comfortably above the
threshold where shrinkage becomes critical. The LD1 loadings reveal which chemical measurements
most strongly separate the cultivars.

### 10.4 Dataset 3: Digits (10-Class) — K-1 Ceiling and Multiclass Visualization

```python
# ── Load ──────────────────────────────────────────────────────────────────
digits = load_digits()
X_dig, y_dig = digits.data, digits.target
print(f"\nDigits: {X_dig.shape[0]} samples, {X_dig.shape[1]} features, {len(np.unique(y_dig))} classes")
print(f"Max LDA components: {len(np.unique(y_dig)) - 1}")  # 9 (K=10 → K-1=9)
print(f"n/p ratio: {X_dig.shape[0]/X_dig.shape[1]:.1f}")  # ~28, good for LDA

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X_dig, y_dig, test_size=0.25, random_state=42, stratify=y_dig
)
scaler_d = StandardScaler()
X_tr_d_sc = scaler_d.fit_transform(X_tr_d)
X_te_d_sc = scaler_d.transform(X_te_d)

# ── Fit full LDA (all 9 components) ───────────────────────────────────────
lda_dig = LinearDiscriminantAnalysis(solver='svd', n_components=9)
lda_dig.fit(X_tr_d_sc, y_tr_d)

print(f"\nDigits LDA — Explained between-class variance ratio:")
cumvar = np.cumsum(lda_dig.explained_variance_ratio_)
for i, (evr, cv_) in enumerate(zip(lda_dig.explained_variance_ratio_, cumvar)):
    print(f"  LD{i+1}: {evr*100:.1f}%  (cumulative: {cv_*100:.1f}%)")

# ── Visualize in 2D (LD1 vs LD2) ──────────────────────────────────────────
X_lda_d = lda_dig.transform(scaler_d.transform(X_dig))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cmap = plt.cm.get_cmap('tab10', 10)

scatter = axes[0].scatter(X_lda_d[:, 0], X_lda_d[:, 1],
                           c=y_dig, cmap='tab10', alpha=0.6, s=15, vmin=-0.5, vmax=9.5)
plt.colorbar(scatter, ax=axes[0], ticks=range(10), label='Digit Class')
axes[0].set_xlabel(f"LD1 ({lda_dig.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"LD2 ({lda_dig.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("Digits — LDA Projection (LD1 vs LD2)")

# Scree plot with cumulative
ax2 = axes[1]
x_pos = np.arange(1, 10)
ax2.bar(x_pos, lda_dig.explained_variance_ratio_ * 100, color='steelblue',
        alpha=0.7, label='Per-component')
ax2_twin = ax2.twinx()
ax2_twin.plot(x_pos, cumvar * 100, 'o-', color='tomato', linewidth=2, label='Cumulative')
ax2_twin.axhline(95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
ax2.set_xlabel("Linear Discriminant Component")
ax2.set_ylabel("% Between-Class Variance (bars)")
ax2_twin.set_ylabel("Cumulative % (line)")
ax2_twin.set_ylim([0, 105])
ax2.set_title("Digits — Between-Class Variance by Component")

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.tight_layout()
plt.savefig("digits_lda.png", dpi=150, bbox_inches='tight')
plt.show()

# ── Classification accuracy vs. n_components ──────────────────────────────
from sklearn.neighbors import KNeighborsClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
component_scores = []
for n in range(1, 10):  # 1 to 9 = K-1 for K=10 digits
    pipe = Pipeline([
        ('sc', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(solver='svd', n_components=n)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    score = cross_val_score(pipe, X_dig, y_dig, cv=cv, scoring='balanced_accuracy').mean()
    component_scores.append(score)
    print(f"  n_components={n}: {score:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 10), component_scores, 'o-', color='steelblue', linewidth=2, markersize=8)
ax.set_xlabel("n_components (LDA)")
ax.set_ylabel("Balanced Accuracy (5-fold CV, KNN downstream)")
ax.set_title("Digits — Accuracy vs. Number of LDA Components")
ax.set_xticks(range(1, 10))
ax.axvline(np.argmax(component_scores) + 1, color='red', linestyle='--',
           label=f"Best: {np.argmax(component_scores)+1} components")
ax.legend()
plt.tight_layout()
plt.savefig("digits_lda_ncomponents.png", dpi=150, bbox_inches='tight')
plt.show()
```

**What we learn from Digits:** With K=10 digit classes, LDA gives exactly 9 discriminant
components. The first two components (LD1 and LD2) already show excellent class separation in
2D — most digit classes form distinct clusters. The scree plot reveals that the first 4–5
components capture most between-class variance. The n_components accuracy curve typically shows
diminishing returns after 5–7 components: downstream KNN accuracy plateaus, showing that the
later discriminant directions add little useful information for classification. This is the
practical argument for truncating at fewer than K-1 components even when you have them all.

### 10.5 Optuna Hyperparameter Tuning (Wine Dataset)

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

n_classes_wine = len(np.unique(y_wine))
max_comps = n_classes_wine - 1  # = 2 for wine

def objective(trial):
    solver = trial.suggest_categorical('solver', ['svd', 'eigen'])

    if solver == 'svd':
        lda = LinearDiscriminantAnalysis(
            solver='svd',
            n_components=trial.suggest_int('n_components', 1, max_comps),
            tol=trial.suggest_float('tol', 1e-6, 1e-2, log=True)
        )
    else:  # eigen
        lda = LinearDiscriminantAnalysis(
            solver='eigen',
            shrinkage=trial.suggest_float('shrinkage', 0.0, 1.0),
            n_components=trial.suggest_int('n_components', 1, max_comps)
        )

    pipe = Pipeline([('sc', StandardScaler()), ('lda', lda)])
    cv_score = cross_val_score(
        pipe, X_wine, y_wine,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='balanced_accuracy'
    ).mean()
    return cv_score

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=80, show_progress_bar=False)

print(f"\nOptuna best params: {study.best_params}")
print(f"Best CV balanced accuracy: {study.best_value:.4f}")
```

### 10.6 Full Diagnostic Suite

```python
# Fit final model on Wine
lda_final = LinearDiscriminantAnalysis(solver='svd', n_components=2)
lda_final.fit(X_tr_w_sc, y_tr_w)
X_lda_w_train = lda_final.transform(X_tr_w_sc)
X_lda_w_test = lda_final.transform(X_te_w_sc)

# Fisher ratio in projected space
class_means_proj = np.array([X_lda_w_train[y_tr_w == k].mean(0) for k in np.unique(y_tr_w)])
global_mean_proj = X_lda_w_train.mean(0)
B = sum(np.sum(y_tr_w == k) *
        np.dot(class_means_proj[k] - global_mean_proj, class_means_proj[k] - global_mean_proj)
        for k in range(3))
W = sum(np.sum((X_lda_w_train[y_tr_w == k] - class_means_proj[k])**2) for k in range(3))
print(f"Fisher ratio in LDA space: {B/W:.2f}  (raw features Fisher ratio: ~1)")

# Trustworthiness
trust = trustworthiness(X_tr_w_sc, X_lda_w_train, n_neighbors=5)
print(f"Trustworthiness (k=5): {trust:.4f}")

# Final test set performance
y_pred_w = lda_final.predict(X_te_w_sc)
print(f"\nWine test set:")
print(classification_report(y_te_w, y_pred_w, target_names=wine.target_names))
```

> 🏆 **Best practice:** Always run the Fisher ratio check. If the ratio in projected LDA space
> is not dramatically higher than in the raw feature space, LDA is not adding value. The ratio
> should be at least 3–5 to justify the projection. For well-separated classes like Wine, expect
> ratios in the dozens.

---

## §11 — When to Use This Algorithm (Decision Guide)

### The Primary Question: Do You Have Labels?

LDA is a **supervised** method. If you do not have class labels for the data you want to reduce,
LDA is not applicable. Stop here and go to PCA, t-SNE, or UMAP.

If you have labels, LDA is the natural first choice for supervised DR. Now ask the secondary
questions.

### Decision Flowchart

```
Have labels? ─── NO ──► Use PCA / t-SNE / UMAP
     │
    YES
     │
Need more than K-1 components? ─── YES ──► Use Supervised UMAP or combine: LDA (K-1 dims) → t-SNE/UMAP
     │
    NO
     │
n_per_class > 5p? ─── NO ──► Use shrinkage LDA (solver='eigen', shrinkage='auto')
     │                        or PCA+LDA pipeline if n << p
    YES
     │
Classes likely Gaussian with equal covariance?
     │
    YES ──► Standard LDA (solver='svd')
     │
    NO ─── Covariances unequal? ──► QDA (reg_param=0.1 for safety)
           Nonlinear boundary? ──► Kernel LDA, gradient boosting, SVM
           Both? ──► Gradient boosting + SHAP for interpretation
```

### LDA vs. Alternatives: When Does Each Win?

| Scenario | Best choice | Why |
|---|---|---|
| Labels available, classes roughly Gaussian | **LDA** | Directly optimizes class separability; provably optimal under assumptions |
| Labels available, class covariances differ | **QDA** | Per-class covariance captures different spreads |
| Labels available, nonlinear boundaries | **Gradient boosting / SVM-RBF** | LDA cannot model curves |
| Labels available, $p \gg n$ | **Shrinkage LDA or PCA+LDA** | Regularization prevents singular covariance |
| No labels, need 2D/3D visualization | **t-SNE / UMAP** | Unsupervised, nonlinear, preserves neighborhood structure |
| No labels, linear reconstruction | **PCA** | Optimal linear unsupervised DR |
| Labels available, want nonlinear + supervised | **Supervised UMAP** | Nonlinear extension of LDA's supervised principle |
| Need fast prediction (<1ms), interpretable weights | **LDA** | Transform is one matrix multiply; weights are human-readable |
| Feature selection (sparse weights) | **Sparse LDA** or **Lasso-regularized classifier** | LDA gives dense weights |

### Red Flags: When NOT to Use LDA

1. **No class labels exist.** LDA cannot be fitted without labels.
2. **K=2 and you need more than 1D projection.** Binary LDA gives exactly 1 component.
3. **Class boundaries are visually nonlinear.** (One class surrounds another, XOR pattern.)
4. **Within-class distributions are bimodal.** The class mean is unrepresentative.
5. **Classes have wildly different covariance structures.** QDA is better.
6. **You need a completely unsupervised exploration.** Labels bias the projection toward known class structure; you might miss unexpected structure.
7. **Test class distribution will differ from training.** LDA's discriminant directions are calibrated to training class structure; severe covariate shift undermines them.

### LDA vs. PCA: The Core Comparison

These are the two linear DR methods you will use most. The choice is usually clear:

| Question | Favors PCA | Favors LDA |
|---|---|---|
| Do you have class labels? | No | Yes |
| Goal: maximize reconstructability? | Yes | No |
| Goal: maximize class separability? | No | Yes |
| n > p per class? | Either | Either |
| n < p? | PCA (then maybe LDA) | PCA first, then LDA |
| Interpretability of directions? | Both equally linear | Both equally linear |
| Classification downstream? | Neutral | LDA is better starting point |
| Visualization without labels? | PCA is natural | Not applicable |

> 💡 **The key insight:** PCA and LDA are not competitors — they are often best used in sequence.
> PCA removes the null space of the within-class scatter matrix (solving the singularity problem)
> and LDA then finds maximally discriminative directions in the reduced space. The Fisherfaces
> pipeline (PCA → LDA) is the canonical implementation of this idea.

---

## §12 — Top Papers to Study

Ranked from "start here" to "deep mastery."

### Tier 1: Start Here

**1. Fisher (1936) — The Original**

Fisher, R.A. (1936). "The Use of Multiple Measurements in Taxonomic Problems." *Annals of Eugenics*, 7(2), 179–188.
DOI: https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

*Why read it:* Remarkably readable for a 90-year-old paper. Fisher derives the 2-class discriminant criterion from scratch, shows the full calculation on iris flowers, and explains the intuition without matrix notation (he uses coordinates). Reading the original forces you to understand the idea at its core before layers of abstraction hide it. Also: Fisher introduced the Iris dataset in this paper.

**2. Zhao et al. (2024) — The Modern Overview**

Zhao, S., Zhang, B., Yang, J., Zhou, J. and Xu, Y. (2024). "Linear Discriminant Analysis." *Nature Reviews Methods Primers*, 4(1), 70.
DOI: https://doi.org/10.1038/s43586-024-00357-9

*Why read it:* The best current comprehensive reference. Covers the theory, numerical implementation, graphical interpretation, all major variants, applications across medicine/genomics/neuroscience, reproducibility considerations, and future directions. Peer-reviewed, authoritative, written for practitioners. Start here if you want a single modern document that covers everything.

### Tier 2: Read Next

**3. Belhumeur, Hespanha & Kriegman (1997) — Fisherfaces**

Belhumeur, P.N., Hespanha, J.P. and Kriegman, D.J. (1997). "Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 19(7), 711–720.
DOI: https://doi.org/10.1109/34.598228
PDF: https://cseweb.ucsd.edu/classes/wi14/cse152-a/fisherface-pami97.pdf

*Why read it:* The paper that showed LDA could be applied to raw images. Introduces the "small sample size problem" clearly and motivates the PCA+LDA pipeline. The face recognition context is vivid and the results (5% vs. 25% error rate under illumination change) are striking. After reading this, you will understand exactly why supervised DR beats unsupervised DR when labels are available.

**4. Rao (1948) — Multiclass Generalization**

Rao, C.R. (1948). "The Utilization of Multiple Measurements in Problems of Biological Classification." *Journal of the Royal Statistical Society, Series B*, 10(2), 159–203.
DOI: https://doi.org/10.1111/j.2517-6161.1948.tb00008.x

*Why read it:* This paper introduces the scatter matrix formalism (S_B and S_W), proves the generalized eigenvalue problem for K classes, and establishes the connection to Mahalanobis distance. Dense but important. Rao's scatter matrices are the mathematical substrate of all modern LDA implementations — understanding their derivation is understanding the algorithm.

**5. Friedman (1989) — Regularized Discriminant Analysis**

Friedman, J.H. (1989). "Regularized Discriminant Analysis." *Journal of the American Statistical Association*, 84(405), 165–175.
DOI: https://doi.org/10.1080/01621459.1989.10478752

*Why read it:* Shows that LDA and QDA are endpoints on a regularization continuum parameterized by two intuitive knobs. The insight that shrinking toward the identity matrix is a form of regularization — making classification more robust by assuming features are more independent than they appear — is both theoretically clean and practically useful. This paper motivated all modern shrinkage approaches.

### Tier 3: Deep Mastery

**6. Ledoit & Wolf (2004) — Analytic Optimal Shrinkage**

Ledoit, O. and Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365–411.
DOI: https://doi.org/10.1016/S0047-259X(03)00096-4

*Why read it:* When sklearn does `shrinkage='auto'`, this is the formula it uses. Understanding the paper means understanding when auto-shrinkage is optimal (asymptotically, when $p/n \to c > 0$) and when it might not be (heavy tails, extremely sparse data). The derivation uses random matrix theory — challenging but illuminating.

**7. Hastie, Tibshirani & Friedman (2009), §4.3 — ESL**

Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Free PDF: https://hastie.su.domains/ElemStatLearn/

*Why read it:* §4.3 (pp. 106–119) is the canonical graduate-textbook treatment of LDA. Derives LDA as the Bayes-optimal classifier under Gaussian assumptions, explains the geometric "sphering" interpretation, compares to logistic regression, and covers regularized DA. Essential for understanding the probabilistic foundation.

**8. Anonymous (2026) — LDA on CNN Features**

"Supervised Dimensionality Reduction Revisited: Why LDA on Frozen CNN Features Deserves a Second Look." arXiv:2604.03928.
URL: https://arxiv.org/abs/2604.03928

*Why read it:* The most modern application paper in this list. Rigorously controlled study showing LDA improves CNN feature quality across architectures and datasets. Makes the case for classical LDA in the deep learning era. If you are building transfer learning pipelines, this paper justifies adding an LDA step between your feature extractor and classifier.

---

## §13 — Resources & Further Reading

### Books

| Book | Chapters | Why Read |
|---|---|---|
| *The Elements of Statistical Learning* — Hastie, Tibshirani & Friedman (2009) | §4.3, §12.6 | Canonical statistical treatment. Free PDF at hastie.su.domains/ElemStatLearn/ |
| *Pattern Classification* — Duda, Hart & Stork (2001) | §2.6, §3.2 | Pattern recognition perspective; Mahalanobis distance connection; decision-theoretic framing |
| *Machine Learning with PyTorch and Scikit-Learn* — Raschka, Liu & Mirjalili (2022) | Ch. 5 | Accessible, code-heavy treatment; author created mlxtend LDA implementation |
| *The Matrix Cookbook* — Petersen & Pedersen (2012) | §1, §8 | Quick reference for matrix derivatives and eigenvalue identities needed to derive LDA by hand. Free at www2.imm.dtu.dk/pubdb/pubs/3274-full.html |
| *Applied Multivariate Statistical Analysis* — Johnson & Wichern (2007) | Ch. 11–12 | Classical statistician's view; LDA in the context of MANOVA and discriminant analysis |

### Documentation

- **sklearn LDA user guide:** https://scikit-learn.org/stable/modules/lda_qda.html
- **sklearn LDA API reference:** https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
- **sklearn covariance estimators:** https://scikit-learn.org/stable/modules/covariance.html
- **mlxtend LDA:** https://rasbt.github.io/mlxtend/user_guide/feature_extraction/LinearDiscriminantAnalysis/
- **pyriemann docs:** https://pyriemann.readthedocs.io/

### Tutorials and Blog Posts

- **Sebastian Raschka — Linear Discriminant Analysis:** https://sebastianraschka.com/Articles/2014_python_lda.html
  — Step-by-step Python implementation from scatter matrices to eigenvectors. Ideal companion to reading Fisher (1936). Raschka is the author of mlxtend.

- **scikit-learn example — LDA vs. PCA for visualization:**
  https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html
  — Official sklearn comparison of PCA and LDA on the Iris dataset with clean matplotlib code.

- **Distill.pub (related):** While there is no Distill post specifically on LDA, the post "Understanding Umap" (https://pair-code.github.io/understanding-umap/) gives excellent context for comparing supervised vs. unsupervised DR methods.

- **Adaptive LDA for BCI (MDPI Brain Sciences 2024):**
  https://www.mdpi.com/2076-3425/14/3/196
  — For practitioners interested in the EEG/BCI application domain.

- **pingouin statistical testing tutorial:**
  https://pingouin-stats.org/build/html/api.html
  — Documentation for Mardia's test and Box's M via the `pingouin` package.

### Online Courses

- **StatQuest with Josh Starmer — LDA Playlist** (YouTube): Excellent visual explanations of the scatter matrix construction and Fisher criterion. Free.
- **Andrew Ng's CS229 Lecture Notes (Stanford):** Chapter on Generative Learning Algorithms covers GDA (Gaussian Discriminant Analysis), which is LDA from a probabilistic perspective. Available free at cs229.stanford.edu.
- **fast.ai Practical Deep Learning:** Part 2 covers transfer learning feature spaces, which connects to the 2026 LDA+CNN application.

---

*Chapter completed: Linear Discriminant Analysis — Dimensionality Reduction Masterclass.*
*Datasets: Iris (3-class), Wine (13-feature chemical), Digits (10-class, K-1 ceiling demo).*

