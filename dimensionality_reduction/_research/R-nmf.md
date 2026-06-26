# Research Dossier: Non-negative Matrix Factorization (NMF)

**For:** Dimensionality Reduction Masterclass — Chapter on NMF  
**Prepared:** 2026-06-26  
**Chapter target:** 18,000+ tokens  
**Writing agent note:** This dossier is your sole context source. Every code snippet uses verified sklearn 1.5.x syntax. API uncertainties are explicitly flagged.

---

## TABLE OF CONTENTS

1. Conceptual Foundations and History
2. Mathematical Formulation
3. Algorithmic Variants (MU vs CD vs ANLS)
4. NMF Uniqueness and Identifiability
5. Key Papers — Full Citations
6. Sklearn API Reference (1.5.x verified)
7. Specialized Python Packages
8. NMF Variants (Sparse, Robust, Online, Semi-, Convex, Deep)
9. Innovative Real-World Applications (6-8 domains)
10. Hyperparameter Tuning Guidance
11. Practitioner Mental Model / Fitting Checklist
12. Evaluation Metrics

---

## 1. CONCEPTUAL FOUNDATIONS AND HISTORY

### Pre-history
- **Paatero & Tapper (1994)**: "Positive Matrix Factorization" — the true algorithmic precursor, applied to environmental science and astrophysics (air pollution source apportionment). Required all matrix entries to be non-negative. Largely unknown outside chemometrics.
- **Lee & Seung (1999)**: The paper that made NMF famous. Published in *Nature*, it showed that NMF learns **parts-based representations** — a critical conceptual contribution. When you apply NMF to face images (the CBCL dataset), you get eyes, noses, mouths as separate components. PCA gives you "eigenfaces" (holistic, ghostly blends). NMF gives you parts.

### The Core Idea
NMF decomposes a non-negative matrix **X** (n_samples × n_features) into two non-negative matrices:

```
X ≈ W · H
```

Where:
- **W** is (n_samples × k) — the "encoding" or "activation" matrix. Each row is a sample's recipe.
- **H** is (k × n_features) — the "dictionary" or "component" matrix. Each row is a learned basis vector.
- **k** = n_components (the rank of the factorization, a hyperparameter)

The non-negativity constraint on both W and H is what distinguishes NMF:
- Non-negativity → **additive combinations only** (no cancellation of parts)
- This is why NMF learns parts: you can only add features, never subtract them
- Contrast with PCA: eigenfaces are linear combinations with positive and negative coefficients, producing holistic (ghostly) faces

### Intuition: Why Parts Emerge
In PCA, the first component accounts for as much variance as possible — often the "average face." PCA then uses subtraction (negative loadings) to remove the average from subsequent components. NMF cannot subtract. So each component must represent an additive feature that when combined with others builds a sample. This forces the algorithm to discover localized, interpretable parts.

### The Matrix Approximation Problem
NMF minimizes a reconstruction objective. The standard choice is the squared Frobenius norm (least squares):

```
minimize ||X - WH||²_F  subject to W ≥ 0, H ≥ 0
```

Alternative objectives:
- **Kullback-Leibler (KL) divergence**: Treats columns of X as probability distributions. Better for count data (text, gene expression read counts).
- **Itakura-Saito (IS) divergence**: Scale-invariant. Used in audio spectrograms (important: audio energy is scale-invariant — a note sounds like itself at any volume).

The general **beta-divergence** family (Cichocki 2009) unifies all three:
- beta=2: Frobenius (squared Euclidean)
- beta=1: KL divergence
- beta=0: Itakura-Saito divergence

---

## 2. MATHEMATICAL FORMULATION

### Problem Statement
Given: **X** ∈ ℝ^(m×n)_+  (all entries ≥ 0)
Find: **W** ∈ ℝ^(m×k)_+, **H** ∈ ℝ^(k×n)_+

Such that: **X ≈ WH**

Minimize D_β(X || WH) where D_β is the beta-divergence.

### Scale Ambiguity
NMF has a scaling ambiguity: if you scale one column of W by constant c and divide the corresponding row of H by c, the product WH is unchanged. This is fundamental — it means there is **no unique factorization** in general (see Section 4).

### Initialization Sensitivity
NMF is **non-convex** in (W, H) jointly, though convex in W alone (H fixed) or H alone (W fixed). This means:
- Local minima abound
- Solution depends heavily on initialization
- Multiple random restarts may be needed
- NNDSVD initialization (Section 6) was specifically designed to address this

---

## 3. ALGORITHMIC VARIANTS

### 3.1 Multiplicative Update Rules (MU) — Lee & Seung 2001

The original algorithm. Update rules derived by ensuring each multiplicative step cannot decrease the objective:

```
H ← H ⊙ (W^T X) / (W^T W H)
W ← W ⊙ (X H^T) / (W H H^T)
```

Where ⊙ is element-wise multiplication and / is element-wise division.

**Convergence:** Monotonically non-increasing but **not guaranteed to converge to a stationary point**. The update can stall near zero (the zero-locking problem: if an entry hits 0, it stays 0 forever because the multiplicative rule multiplies by a factor that cannot resurrect it).

**Practical performance:** Slow. Research shows CD converges **5-24x faster** than MU in wall-clock time.

**When to use MU in sklearn:** When solver='mu'. Required if beta_loss != 'frobenius' (i.e., for KL divergence or Itakura-Saito). For Frobenius loss, prefer 'cd'.

**KL divergence update rules:**
```
H ← H ⊙ (W^T (X / WH)) / (sum of W columns)
W ← W ⊙ ((X / WH) H^T) / (sum of H rows)
```

**IS divergence update rules:** Similar multiplicative form but with different power in the denominator (β=0 case).

### 3.2 Coordinate Descent (CD) — Chou & Lin 2012

Sklearn's default solver. Cycles through each element or block of W and H and minimizes the objective exactly with respect to that variable while holding everything else fixed.

**Algorithm:**
1. Fix H, solve for W column by column using nonnegative least squares (NNLS)
2. Fix W, solve for H row by row using NNLS
3. Repeat until convergence

**Convergence guarantee:** Converges to a stationary point (local minimum or saddle point). More rigorous than MU.

**Performance:** 5-24x faster than MU in practice. Excellent on sparse data (updates only active coordinates).

**Zero-locking advantage:** CD can move an entry away from zero because it uses NNLS (which can set an entry to exactly zero or away from it based on the gradient).

**When to use:** Default choice. Only works with Frobenius loss in sklearn.

### 3.3 Alternating Nonnegative Least Squares (ANLS) — Kim & Park 2008

The ANLS framework is a block coordinate descent approach. It alternates between:
1. Solve nonneg LS for H (fix W)  
2. Solve nonneg LS for W (fix H)

The key insight from Kim & Park 2008 (SIAM Journal on Matrix Analysis and Applications, 30:713-730): use the **block principal pivoting method** for the inner NNLS problem, which converges faster than active set methods.

**Not directly in sklearn** (sklearn's CD is close but not identical). Available in nimfa.

### 3.4 Projected Gradient Descent

Hoyer (2004) used this for sparse NMF: take gradient step, then project onto the constraint set (non-negativity, and optionally a fixed L1/L2 norm ratio for sparsity). Less common in practice than MU or CD.

### 3.5 Stochastic / Mini-Batch (Online NMF)

Process mini-batches of data and update H after each batch:
- Maintains a running estimate of dictionary H
- Forget factor controls how much old information decays
- Enables streaming / out-of-core processing
- sklearn's MiniBatchNMF (added in v1.1)

---

## 4. NMF UNIQUENESS AND IDENTIFIABILITY

### The Bad News
In general, **NMF is not unique**. Given a solution (W, H), you can produce infinitely many other solutions by:
- Scaling: (WD, D⁻¹H) for any positive diagonal D
- Permuting columns of W and rows of H correspondingly
- More complex transformations that preserve non-negativity

### Sufficient Conditions for Uniqueness

**Sufficiently Scattered Condition (SSC):** The main modern uniqueness theory. If the columns of W^T and the rows of H are "sufficiently scattered" across the non-negative orthant (meaning they approximate covering the cone well), then NMF is essentially unique (up to scaling and permutation).

**Sparsity-based uniqueness:** The Data-Based Uniqueness (DBU) theorem from chemometrics: if each column of X is a "pure" sample (has non-zero weight for only one component), then NMF is unique. Practically this means if every topic has at least one document that discusses only that topic.

**Separability assumption (Donoho & Stodden 2003, Arora et al. 2012):** A matrix X is separable if for every column of H (every "topic"), there exists at least one row of X (one "word") that appears in only that topic. Such words are called **anchor words**. Under separability:
- NMF can be solved in **polynomial time** (Arora et al. 2012)
- The solution is essentially unique
- This is the basis for "anchor-word algorithms" like SPAMS

**Partial identifiability (SIAM 2023):** Recent work (Lauritzen, Richardson, Robins in SIAM Journal on Matrix Analysis, 2023) shows that even without full identifiability, some individual factors or their combinations may be identifiable.

### Practical Implications for Practitioners
1. Run NMF multiple times with different random seeds — if results are consistent, you have (near-)unique factorization
2. Add L1 sparsity regularization (encourages uniqueness through Hoyer's theory)
3. Don't over-interpret component order or scale — these are arbitrary
4. For critical applications (e.g., genomics), report consensus factorizations from many runs

---

## 5. KEY PAPERS — FULL CITATIONS

### Paper 1: Lee & Seung 1999 (THE SEMINAL PAPER)
- **Authors:** Daniel D. Lee, H. Sebastian Seung
- **Title:** Learning the parts of objects by non-negative matrix factorization
- **Venue:** Nature
- **Year:** 1999
- **Volume/Pages:** 401(6755):788-791
- **DOI:** 10.1038/44565
- **URL:** https://www.nature.com/articles/44565
- **What you'll learn:** (1) The conceptual breakthrough: NMF learns parts-based decompositions, not holistic ones like PCA. (2) Three experiments: face images (parts = eye, nose, mouth), text (parts = topics), demonstrations on CBCL dataset. (3) The original multiplicative update rules are introduced here. (4) Why non-negativity matters: it induces sparsity and interpretability through additive-only combinations.
- **Difficulty:** Low-medium. Beautifully written, accessible to ML practitioners. Read this first.
- **Historical note:** This 3-page Nature paper single-handedly launched NMF as a mainstream ML tool. Prior work by Paatero (1994) was largely ignored outside environmental science.

### Paper 2: Lee & Seung 2001 (THE ALGORITHMS PAPER)
- **Authors:** Daniel D. Lee, H. Sebastian Seung
- **Title:** Algorithms for Non-negative Matrix Factorization
- **Venue:** Advances in Neural Information Processing Systems 13 (NIPS 2000, proceedings published 2001)
- **Pages:** 556-562
- **URL:** https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization
- **Note:** Sometimes cited as NIPS 2000, sometimes as 2001 (proceedings publication). Both year citations appear in the literature.
- **What you'll learn:** (1) The formal derivation of multiplicative update rules for both Frobenius loss and KL divergence. (2) Convergence proof using auxiliary functions (analogous to the EM algorithm convergence proof). (3) Two algorithms presented: least-squares NMF and KL-NMF. A 2025 arXiv supplement (2501.11341) provides a more detailed proof guide for readers who want rigor.
- **Difficulty:** Medium. Requires understanding of divergences and auxiliary functions. The proof technique is elegant and worth learning.

### Paper 3: Cichocki et al. 2009 (THE COMPREHENSIVE BOOK)
- **Authors:** Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, Shun-ichi Amari
- **Title:** Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-way Data Analysis and Blind Source Separation
- **Publisher:** John Wiley & Sons
- **Year:** 2009
- **ISBN:** 978-0-470-74666-0
- **Wiley page:** https://www.wiley.com/en-us/Nonnegative+Matrix+and+Tensor+Factorizations
- **What you'll learn:** (1) The full zoo of NMF algorithms: multiplicative rules, projected gradient, ALS, ANLS, and many others. (2) The beta-divergence family unifying Frobenius, KL, and Itakura-Saito. (3) Tensor extensions (NTF) for multi-way data. (4) Signal processing applications in depth: blind source separation, audio, brain imaging. (5) Comprehensive benchmarks.
- **Difficulty:** High. This is the graduate textbook. 500+ pages. Use as reference, not cover-to-cover.

### Paper 4: Hoyer 2004 (SPARSE NMF)
- **Author:** Patrik O. Hoyer
- **Title:** Non-negative matrix factorization with sparseness constraints
- **Venue:** Journal of Machine Learning Research (JMLR)
- **Year:** 2004
- **Volume/Pages:** 5:1457-1469
- **URL:** https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf
- **What you'll learn:** (1) Defines a principled sparsity measure for NMF (Hoyer measure: ranges [0,1], invariant to L2 norm). (2) Projects gradient updates onto a feasible set with controlled sparsity. (3) Shows that sparsity constraints improve uniqueness and interpretability.
- **Difficulty:** Medium.

### Paper 5: Kim & Park 2008 (ANLS / FAST NMF)
- **Authors:** H. Kim, H. Park
- **Title:** Nonnegative Matrix Factorization Based on Alternating Nonnegativity Constrained Least Squares and Active Set Method
- **Venue:** SIAM Journal on Matrix Analysis and Applications
- **Year:** 2008
- **Volume/Pages:** 30:713-730
- **DOI:** 10.1137/07069239X
- **What you'll learn:** (1) ANLS framework with block principal pivoting. (2) Convergence theory for block coordinate descent in NMF. (3) Benchmarks showing ANLS outperforms MU by large margins.
- **Difficulty:** High (requires numerical linear algebra background).

### Paper 6: Boutsidis & Gallopoulos 2008 (NNDSVD INITIALIZATION)
- **Authors:** C. Boutsidis, E. Gallopoulos
- **Title:** SVD based initialization: A head start for nonnegative matrix factorization
- **Venue:** Pattern Recognition
- **Year:** 2008
- **Volume/Pages:** 41(4):1350-1362
- **What you'll learn:** (1) NNDSVD: uses the SVD of X to construct a deterministic, non-negative initialization for W and H. (2) Three variants: NNDSVD (zeroing small values), NNDSVDa (setting zeros to mean), NNDSVDar (setting zeros to small random values). (3) Substantially better initialization → faster convergence and better final solutions than random init.
- **Difficulty:** Medium.

### Paper 7: Févotte & Idier 2011 (BETA-DIVERGENCE NMF)
- **Authors:** Cédric Févotte, Jérôme Idier
- **Title:** Algorithms for Nonnegative Matrix Factorization with the β-Divergence
- **Venue:** Neural Computation
- **Year:** 2011
- **Volume/Pages:** 23(9):2421-2456
- **DOI:** 10.1162/NECO_a_00168
- **What you'll learn:** (1) Multiplicative update rules for the full beta-divergence family. (2) Theoretical justification for IS divergence in audio. (3) EM-style interpretation.
- **Difficulty:** High (requires signal processing background for IS interpretation).

### Paper 8: Arora et al. 2012 (SEPARABLE NMF / ANCHOR WORDS)
- **Authors:** Sanjeev Arora, Rong Ge, Ravi Kannan, Ankur Moitra
- **Title:** Computing a Nonnegative Matrix Factorization — Provably
- **Venue:** STOC 2012
- **Year:** 2012
- **URL:** http://arxiv.org/abs/1111.0952
- **What you'll learn:** (1) Under the separability assumption, NMF can be solved in polynomial time. (2) Anchor words in topic models. (3) Connects NMF to computational geometry (conical hull algorithms).
- **Difficulty:** High (theoretical CS/algorithms background needed).

---

## 6. SKLEARN API REFERENCE (verified 1.5.x)

### 6.1 `sklearn.decomposition.NMF`

```python
from sklearn.decomposition import NMF

model = NMF(
    n_components=None,       # int | 'auto' | None
    init=None,               # str | None
    solver='cd',             # 'cd' | 'mu'
    beta_loss='frobenius',   # str | float
    tol=1e-4,                # float
    max_iter=200,            # int
    random_state=None,       # int | RandomState | None
    alpha_W=0.0,             # float
    alpha_H='same',          # float | 'same'
    l1_ratio=0.0,            # float [0, 1]
    verbose=0,               # int
    shuffle=False,           # bool
)
```

#### Parameter-by-Parameter Reference

**`n_components`**
- Type: `int | 'auto' | None`
- Default: `None` (uses min(n_samples, n_features))
- Controls: The rank k of the factorization. The most critical hyperparameter.
- 'auto' option: New in v1.4. Infers from shapes of custom W/H passed at fit time (only useful with init='custom').
- Practical note: Setting to None uses all features — gives perfect reconstruction but no compression. Always set this explicitly.

**`init`**
- Type: `{'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'} | None`
- Default: `None` → resolves to 'nndsvda' if n_components ≤ min(n_samples, n_features), else 'random'
- **VERSION CHANGE: In v1.1, default changed from 'nndsvd' to 'nndsvda'** (because 'nndsvd' produces many zeros that can cause issues with MU solver)
- Options:
  - `'random'`: W and H initialized from uniform distribution. Requires multiple restarts. Needed when n_components > min(n_samples, n_features).
  - `'nndsvd'`: Nonnegative Double SVD (Boutsidis & Gallopoulos 2008). Deterministic. Zero entries where SVD component is negative. **Best for sparse data.** Incompatible with MU solver if zeros appear (MU gets stuck at 0).
  - `'nndsvda'`: Like nndsvd but zeros replaced by mean of X. **Best general-purpose default.** Avoids zero-locking.
  - `'nndsvdar'`: Like nndsvd but zeros replaced by small random values. A compromise between nndsvd and random.
  - `'custom'`: Pass your own W and H via `fit(X, W=W_init, H=H_init)`. Requires n_components='auto' or consistent shape.

**`solver`**
- Type: `{'cd', 'mu'}`
- Default: `'cd'`
- `'cd'` = Coordinate Descent. Faster (5-24x), convergence guarantee. Only works with `beta_loss='frobenius'`.
- `'mu'` = Multiplicative Update. Required for `beta_loss` ∈ {'kullback-leibler', 'itakura-saito', or any float != 2.0}. Slower but handles all divergences.
- Recommendation: Use 'cd' for speed unless you need KL or IS divergence.

**`beta_loss`**
- Type: `float | {'frobenius', 'kullback-leibler', 'itakura-saito'}`
- Default: `'frobenius'`
- Named aliases map to floats: 'frobenius'=2.0, 'kullback-leibler'=1.0, 'itakura-saito'=0.0
- Can also pass arbitrary float for generalized beta-divergence
- **Constraint:** Only used with `solver='mu'`. With `solver='cd'`, this is ignored (always uses Frobenius).
- **Constraint for IS:** `beta_loss ≤ 0` → input X cannot contain zeros.
- When to use KL: Count data (bag-of-words, read counts), where Poisson noise model is appropriate.
- When to use IS: Audio/music spectrograms where scale-invariance matters.
- When to use Frobenius: Most other cases, especially when using 'cd' solver.

**`tol`**
- Type: `float`
- Default: `1e-4`
- Controls: Convergence threshold on relative reconstruction error change between iterations.
- Setting 0.0 forces running until `max_iter`.

**`max_iter`**
- Type: `int`
- Default: `200`
- Practical note: 200 is often insufficient. For large datasets or when `init='random'`, may need 500-2000. Check `n_iter_` attribute after fitting; if it equals `max_iter`, the algorithm did not converge.
- **API FLAG:** If `n_iter_ == max_iter`, sklearn will emit a `ConvergenceWarning`.

**`random_state`**
- Type: `int | RandomState | None`
- Default: `None`
- Used for: 'random' and 'nndsvdar' initializations, and shuffle in CD solver.
- For reproducibility, always set when using 'random' init.

**`alpha_W`**
- Type: `float`
- Default: `0.0` (no regularization)
- New in: v1.0 (replaced deprecated `alpha` parameter)
- Controls L1 and L2 regularization strength on W matrix.
- Regularization applied: `l1_ratio * alpha_W * ||W||_1 + (1-l1_ratio)/2 * alpha_W * ||W||_F²`

**`alpha_H`**
- Type: `float | 'same'`
- Default: `'same'` (uses alpha_W value)
- New in: v1.0
- Controls regularization on H matrix. Set to 0.0 to regularize only W, or to a different value to regularize W and H differently.
- Common pattern: regularize only H (the encoding matrix) to encourage sparse representations: `alpha_W=0, alpha_H=0.1, l1_ratio=1.0`

**`l1_ratio`**
- Type: `float` in [0, 1]
- Default: `0.0` (pure L2 regularization)
- Controls mix of L1 vs L2: `0.0` = L2 ridge, `1.0` = L1 lasso, in between = elastic net
- For sparsity in components: use `l1_ratio=1.0` (pure L1)

**`verbose`**
- Type: `int`
- Default: `0`
- 1 or higher: prints convergence information at each iteration.

**`shuffle`**
- Type: `bool`
- Default: `False`
- Only for CD solver: if True, randomizes the order of coordinate updates each iteration. Can help escape local minima.

#### Key Attributes After Fitting

```python
model.components_          # H matrix, shape (n_components, n_features)
model.reconstruction_err_  # Final loss value (Frobenius or beta-divergence)
model.n_iter_              # Number of iterations actually performed
model.n_features_in_       # n_features seen during fit (v0.24+)
```

#### Inference

```python
W = model.fit_transform(X)   # Fit and get W (encoding), shape (n_samples, n_components)
W = model.transform(X_new)   # Encode new data using learned H
X_reconstructed = W @ model.components_  # Reconstruct
```

#### **DEPRECATED PARAMETERS** (v1.0 removed `alpha` and `regularization`)
- Old: `alpha=0.1, regularization='both'`
- New: `alpha_W=0.1, alpha_H=0.1, l1_ratio=0.0`
- This was a **breaking change** in v1.0. If you see old code using `alpha=`, it needs updating.

---

### 6.2 `sklearn.decomposition.MiniBatchNMF`

Added in sklearn v1.1. Handles datasets too large to fit in memory.

```python
from sklearn.decomposition import MiniBatchNMF

model = MiniBatchNMF(
    n_components=None,            # int | 'auto' | None
    init=None,                    # str | None
    batch_size=1024,              # int
    beta_loss='frobenius',        # str | float
    tol=1e-4,                     # float
    max_no_improvement=10,        # int | None
    max_iter=200,                 # int
    alpha_W=0.0,                  # float
    alpha_H='same',               # float | 'same'
    l1_ratio=0.0,                 # float
    forget_factor=0.7,            # float
    fresh_restarts=False,         # bool
    fresh_restarts_max_iter=30,   # int
    transform_max_iter=None,      # int | None
    random_state=None,            # int | RandomState | None
    verbose=False,                # bool
)
```

#### MiniBatchNMF-Specific Parameters

**`batch_size`**
- Default: `1024`
- Number of samples per mini-batch.
- Larger = better long-term convergence (more stable gradient), slower iterations.
- Smaller = faster iterations but noisier updates.
- Rule of thumb: Start with 1024. Scale with dataset size (keep batches at 5-10% of data for datasets < 100K rows; use 1024-4096 for millions).

**`max_no_improvement`**
- Default: `10`
- Early stopping: stop after this many consecutive mini-batch iterations without improvement.
- Set to `None` to disable early stopping and always run `max_iter`.

**`forget_factor`**
- Default: `0.7`
- Range: (0, 1]
- Controls how quickly old mini-batch information is forgotten.
- `1.0` = equal weight to all past batches (good for finite, static datasets).
- `< 1.0` = exponential decay — more recent batches weighted more (good for true streaming / concept drift).
- Typical values: 0.5-0.9 for online streaming; 1.0 for finite large datasets.

**`fresh_restarts`**
- Default: `False`
- If `True`, fully solves for W (the per-sample encoding) at each mini-batch, not just one step.
- Slower but produces better encodings. Use when reconstruction quality is critical.

**`fresh_restarts_max_iter`**
- Default: `30`
- Maximum iterations for the inner W optimization when `fresh_restarts=True`.

**`transform_max_iter`**
- Default: `None` → uses `max_iter`
- Maximum iterations used when calling `transform()` on new data after fitting.

#### MiniBatchNMF Usage Pattern

```python
from sklearn.decomposition import MiniBatchNMF

# For streaming/online scenario
model = MiniBatchNMF(n_components=20, forget_factor=0.7, random_state=42)

# Partial fit on batches
for batch in data_batches:
    model.partial_fit(batch)

# Encode new data
W_new = model.transform(X_new)
```

---

### 6.3 Sklearn NMF Example Code (Verified 1.5.x Syntax)

#### Basic NMF with NNDSVDA initialization

```python
from sklearn.decomposition import NMF
import numpy as np

# X must be non-negative
X = np.abs(np.random.randn(100, 50))

model = NMF(
    n_components=10,
    init='nndsvda',   # Best general-purpose init
    solver='cd',      # Fastest solver
    max_iter=500,     # More iterations for safety
    random_state=42,
    tol=1e-4,
)

W = model.fit_transform(X)  # (100, 10)
H = model.components_        # (10, 50)

print(f"Reconstruction error: {model.reconstruction_err_:.4f}")
print(f"Converged in {model.n_iter_} iterations")
```

#### NMF with KL divergence (for text/counts)

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [...]  # list of text strings
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(docs)  # TF-IDF is non-negative

model = NMF(
    n_components=20,      # Number of topics
    init='nndsvda',
    solver='mu',          # Required for KL divergence
    beta_loss='kullback-leibler',
    max_iter=1000,
    random_state=42,
    alpha_H=0.1,          # Regularize encoding
    l1_ratio=1.0,         # L1 for sparsity
)

W = model.fit_transform(X)   # Document-topic matrix
H = model.components_         # Topic-word matrix
feature_names = vectorizer.get_feature_names_out()

# Top words per topic
for i, topic_vec in enumerate(H):
    top_idx = topic_vec.argsort()[-10:][::-1]
    print(f"Topic {i}: {', '.join(feature_names[top_idx])}")
```

#### Sparse NMF with L1 regularization

```python
model = NMF(
    n_components=15,
    init='nndsvda',
    solver='cd',
    alpha_W=0.0,    # Don't regularize W (patterns)
    alpha_H=0.5,    # Regularize H (encodings → sparse)
    l1_ratio=1.0,   # Pure L1 → true sparsity
    max_iter=500,
)
```

#### Checking convergence

```python
model = NMF(n_components=20, max_iter=300, random_state=42)
W = model.fit_transform(X)

if model.n_iter_ == model.max_iter:
    import warnings
    print("WARNING: NMF did not converge! Consider increasing max_iter.")
    # Increase to 500 or 1000 and refit
```

---

## 7. SPECIALIZED PYTHON PACKAGES

### 7.1 nimfa

- **PyPI name:** `nimfa`
- **Install:** `pip install nimfa`
- **Version (mid-2026):** 1.4.0 (last release ~2021; project mature but low maintenance activity)
- **GitHub:** https://github.com/marinkaz/nimfa
- **License:** BSD
- **Maintenance:** Low activity. No major releases since 2021. Works with Python 3.x. Dependencies: NumPy ≥ 1.7, SciPy ≥ 0.12.
- **What it does better than sklearn:**
  - 17+ NMF algorithms: LSNMF, NSNMF (Nonsmooth NMF), PMF (Probabilistic), SNMF (Sparse), PSMF, BD (Bayesian), etc.
  - Supports both dense and sparse matrix representations (scipy.sparse)
  - Built-in quality metrics: cophenetic correlation, dispersion, residuals, etc.
  - Consensus NMF over multiple runs for stability analysis
- **Key import:**
  ```python
  import nimfa
  nmf = nimfa.Nmf(V, rank=10, seed='nndsvd', max_iter=200)
  nmf_fit = nmf()
  W = nmf_fit.basis()
  H = nmf_fit.coef()
  ```
- **Best for:** Research requiring multiple NMF variants, consensus analysis, or bioinformatics (was designed with genomics in mind).

### 7.2 PyTorch NMF (torchnmf)

- **PyPI name:** `torchnmf`
- **Install:** `pip install torchnmf`
- **Version (mid-2026):** ~0.3.4
- **GitHub:** https://github.com/yoyolicoris/pytorch-NMF
- **Documentation:** https://pytorch-nmf.readthedocs.io/
- **License:** MIT
- **Maintenance:** Active
- **What it does better than sklearn:**
  - **GPU acceleration** via PyTorch (critical for large matrices: tens of thousands × tens of thousands)
  - Convolutional NMF (CNMF) — not available anywhere else in Python
  - PLCA (Probabilistic Latent Component Analysis) variants
  - All standard beta-divergences (Frobenius, KL, IS)
  - Can run batched NMF across multiple GPUs
- **Key import:**
  ```python
  import torch
  from torchnmf.nmf import NMF

  # X must be a non-negative torch.Tensor
  X = torch.abs(torch.randn(1000, 500)).cuda()
  model = NMF(X.shape, rank=20).cuda()
  model.fit(X, beta=2)  # beta=2: Frobenius, beta=1: KL, beta=0: IS
  W = model.H  # (1000, 20) — torchnmf uses H for activations
  H = model.W  # (20, 500) — torchnmf uses W for dictionary (note: inverted!)
  ```
- **Note on naming convention:** torchnmf uses H for the sample matrix (sklearn's W) and W for the component matrix (sklearn's H). This is the audio/signal processing convention — be careful.

### 7.3 Robust NMF (robust-nmf)

- **GitHub:** https://github.com/neel-dey/robust-nmf
- **Install:** `pip install robust-nmf` (or clone from GitHub)
- **License:** MIT
- **Maintenance:** Research code (academic), may have limited active maintenance
- **What it does:** PyTorch (GPU) and NumPy (CPU) implementation of Févotte & Dobigeon's Robust NMF algorithm from "Nonlinear hyperspectral unmixing with robust nonnegative matrix factorization." Handles outliers via L2,1 norm on the noise component.
- **Best for:** Hyperspectral imaging with pixel contamination, face recognition with occlusions.

### 7.4 GeneNMF

- **GitHub:** https://github.com/carmonalab/GeneNMF
- **Install:** R package — `install.packages('GeneNMF')` (R ecosystem, not Python)
- **What it does:** NMF for single-cell omics data, integrates directly with Seurat objects, discovers gene programs across multiple samples, performs multi-sample consensus NMF.
- **Relevance:** Worth mentioning for scRNA-seq applications; Python users should use sklearn NMF or nimfa on scRNA-seq count matrices.

### 7.5 sklearn NMF (baseline comparison)

- **PyPI name:** `scikit-learn`
- **Install:** `pip install scikit-learn`
- **Version (mid-2026):** 1.5.x (stable) or 1.6/1.7/1.8+ (newer stable)
- **License:** BSD
- **What it does:** Production-quality NMF with CD and MU solvers, NNDSVD init, MiniBatchNMF for large data. Best integrated with sklearn pipelines, cross-validation, etc.
- **Limitation:** No GPU support, limited to standard beta-divergences, no convolutional NMF, no consensus analysis.

### 7.6 Beta NMF (minibatchNMF by rserizel)

- **GitHub:** https://github.com/rserizel/minibatchNMF
- **Install:** Clone and install
- **What it does:** Mini-batch multiplicative update rules specifically for beta-NMF. Academic implementation from a 2014 paper. Slower than sklearn's MiniBatchNMF but pedagogically useful.
- **Best for:** Understanding mini-batch NMF theory, audio applications with IS divergence.

### 7.7 ESAT (Environmental Source Apportionment Toolkit)

- **Reference:** PMC12180922 (2025 paper)
- **Focused application:** Air quality / environmental science source apportionment. Implements EPA's PMF (Positive Matrix Factorization, the Paatero-Tapper ancestor of NMF) methodology.
- **Best for:** Atmospheric science, environmental monitoring.

---

## 8. NMF VARIANTS

### 8.1 Sparse NMF (Hoyer 2004)

**What it is:** Adds an explicit sparsity constraint to NMF. Instead of just requiring W, H ≥ 0, it additionally requires a target sparsity level (fraction of zeros) for each column of W and/or row of H.

**Hoyer sparseness measure:** For a vector x:
```
sparseness(x) = (√n - ||x||₁/||x||₂) / (√n - 1)
```
Ranges from 0 (dense, all equal) to 1 (sparse, only one nonzero).

**Algorithm:** Projected gradient descent: take gradient step, project onto cone of non-negative vectors with the desired L1/L2 norm ratio.

**In sklearn:** Approximate sparse NMF via regularization:
```python
# This promotes sparsity but doesn't guarantee a specific sparseness level
NMF(n_components=k, alpha_H=0.5, l1_ratio=1.0)  # L1 on H → sparse H
```

**When to use:** When you want maximally interpretable components (e.g., topic models where each document loads heavily on just 1-2 topics).

### 8.2 Robust NMF

**What it is:** Decomposes X into a low-rank part (WH) plus a sparse outlier matrix E:
```
X = WH + E, with W,H ≥ 0, E sparse
```
**Motivation:** NMF breaks down when data has gross corruptions (occlusions in faces, bad pixels in hyperspectral images, gross noise in audio). Standard NMF tries to fit the outliers and distorts the components.

**Algorithm:** Alternating minimization with L2,1 norm on E (robust to column-sparse corruptions) or L1 norm on E (robust to arbitrary corruptions).

**Applications:** Robust face recognition with partial occlusions, hyperspectral unmixing with bad scan lines, audio denoising.

**In Python:** `robust-nmf` package (Févotte & Dobigeon 2014).

### 8.3 Online NMF / MiniBatchNMF

**What it is:** Processes data in mini-batches. Updates H (the dictionary) incrementally, without revisiting old data.

**Key idea (Mairal et al. 2010 — online dictionary learning):** At each step t, observe mini-batch X_t, solve for current encodings W_t, then update H using a surrogate function that approximates the full dataset objective.

**The forget factor:** Controls contribution of old data:
- At time t, weight of mini-batch s is `forget_factor^(t-s)`
- With forget_factor=1: all batches weighted equally (like full batch)
- With forget_factor<1: old data gradually forgotten

**When to use over full NMF:**
- Dataset exceeds available RAM (cannot fit in memory)
- Streaming scenario (new data arriving continuously)
- Want faster approximate solution on very large datasets
- Datasets with millions of samples (e.g., web-scale text)

**sklearn implementation:** `MiniBatchNMF` (added v1.1). Use `partial_fit` for true streaming:
```python
model = MiniBatchNMF(n_components=20, forget_factor=1.0)
for batch in chunks(X, size=1024):
    model.partial_fit(batch)
```

### 8.4 Semi-NMF (Ding et al. 2010)

**What it is:** Relaxes the non-negativity constraint on W (the encoding matrix), keeping H ≥ 0. The data matrix X can contain mixed-sign entries.

**Factorization:** X ≈ WH where H ≥ 0 but W is unrestricted (can be negative).

**Why it matters:** Real data often contains negative values (e.g., mean-subtracted data, centered embeddings). Standard NMF cannot handle this. Semi-NMF can.

**Deep Semi-NMF:** Stacks multiple Semi-NMF layers to learn hierarchical representations:
```
X ≈ W₁ H₁
H₁ ≈ W₂ H₂  (H₁ ≥ 0)
...
```
The final H_n is non-negative and interpretable; intermediate W matrices can have mixed signs. This is a form of deep feature learning predating deep learning's dominance.

**Recent work (2023-2024):** DNSRF (Deep Network Semi-NMF Representation Framework, ACM TIST 2024) combines deep semi-NMF with discriminative learning.

### 8.5 Convex NMF (Ding et al. 2008, ICDM)

**What it is:** Constrains the columns of W to lie in the convex hull of the data points. H remains non-negative.

**Factorization:** W = X G, where G ≥ 0. So W is literally a non-negative mixture of data points.

**Why it matters:**
1. Guarantees physical interpretability: components are convex combinations of actual data points, not abstract latent vectors
2. Enables NMF on data with mixed-sign entries (since W is formed from X, and X might have negatives, but G ≥ 0 ensures W is meaningful)
3. Connects to archetypal analysis and k-means (as special cases)

**Applications:** When you want components to be identifiable as "typical examples" or "archetypes" (customer segments that correspond to real customer profiles, not abstract blends).

### 8.6 Deep NMF

**What it is:** Multiple layers of NMF stacked together:
```
X ≈ W₁ H₁
H₁ ≈ W₂ H₂   (H₁,H₂ ≥ 0, W₁,W₂ ≥ 0)
...
```
Each layer learns more abstract representations. Bridges NMF and deep learning.

**Key difference from Semi-NMF:** All matrices are non-negative (pure NMF stacking). This leads to hierarchical parts-based representations at multiple scales.

**Training:** Usually greedy layer-wise pretraining (train layer 1, freeze H₁, use as input to layer 2, etc.). Can also be trained end-to-end via backpropagation (Deep NMF with PyTorch autograd).

**Developments (2023-2024):**
- Generalized Discriminative Deep NMF (IJCAI 2023): adds discriminative loss to deep NMF
- Deep Approximately Orthogonal NMF: encourages orthogonality at each layer for disentanglement
- Sparse Deep NMF (arXiv 1707.09316): combines depth with sparsity constraints

**Applications:** Multi-scale image decomposition, hierarchical topic modeling, layered audio decomposition.

---

## 9. INNOVATIVE REAL-WORLD APPLICATIONS

### 9.1 Mutational Signatures in Cancer Genomics (HIGHEST IMPACT APPLICATION)

**Domain:** Computational oncology / cancer genomics
**The problem:** Tumors accumulate somatic mutations over a patient's lifetime. Different mutagenic processes (UV radiation, smoking, BRCA dysfunction, chemotherapy exposure) leave distinctive patterns in the mutation spectrum. We want to decompose observed mutations into these underlying processes.
**Data:** A 96-dimensional vector per tumor sample (6 possible base substitutions × 16 trinucleotide contexts). Matrix X is (n_tumors × 96).
**Why NMF:** The mutation count matrix is non-negative (counts can't be negative). Each mutation in a tumor is caused by some underlying process. NMF decomposes X = WH where H contains "signatures" (rows: the 96-dim patterns of each mutagenic process) and W contains "exposures" (columns: how much each process contributed to each tumor). This is literally an additive parts model.
**Current state:** The COSMIC database (https://cancer.sanger.ac.uk/signatures/) catalogs 78+ validated SBS signatures. Tools: SigProfiler (official COSMIC tool), MuSiCal (Nature Genetics 2024). Recent work (bayesNMF, arXiv 2502.18674) uses Bayesian NMF with automatic rank selection for this problem.
**Impact:** Has identified signature SBS4 (smoking), SBS7a/7b (UV radiation), SBS3 (BRCA/HRD deficiency), SBS11 (temozolomide chemotherapy). Directly actionable clinically — affects treatment selection.
**Why not PCA:** Mutations can't be "negative." PCA components would have negative weights, meaningless for mutation counts. KL-divergence NMF is preferred (counts follow Poisson distribution).

### 9.2 Audio Source Separation (Music Transcription / Denoising)

**Domain:** Music information retrieval, audio engineering
**The problem:** Given a mixture audio signal, separate it into component sources (drums, bass, vocals, piano). Or transcribe polyphonic music (identify which notes are playing at each time).
**Data:** Short-Time Fourier Transform (STFT) magnitude spectrogram: matrix X is (n_freq_bins × n_time_frames). Each entry is a non-negative magnitude.
**Why NMF with IS divergence:** Each spectral column of H represents a prototype "template" of one instrument or note. Each column of W represents how active that template is over time. Itakura-Saito divergence is used because audio perception is scale-invariant (a note sounds the same loud or quiet), and IS is scale-invariant. The power spectrogram at any frequency bin is the sum of contributions from each source.
**Approach:** NMF with IS divergence + continuity and sparsity constraints. The columns of H are encouraged to be harmonic (peaks at integer multiples of the fundamental frequency) for music.
**Tools:** torchnmf (GPU), custom implementations.
**Companies:** Used in iZotope (audio software), LANDR (mastering), academic tools like REPET.
**Limitation:** Modern deep learning (U-Net, Demucs) has largely surpassed NMF for audio source separation on most benchmarks. NMF remains competitive for limited-resource scenarios and interpretable musical transcription.
**Reference:** "Single-Channel Audio Source Separation with NMF: Divergences, Constraints and Algorithms" (Springer 2018).

### 9.3 Hyperspectral Image Unmixing (Remote Sensing / Geology)

**Domain:** Remote sensing, environmental monitoring, mineral exploration
**The problem:** A hyperspectral satellite or airborne sensor captures hundreds of spectral bands per pixel. Each ground pixel is typically a mixture of several materials ("endmembers": water, vegetation, soil, rock, urban). We want to estimate which materials are present and in what proportions.
**Data:** Hyperspectral data cube → unfolded into X (n_pixels × n_spectral_bands). Non-negative (reflectance values).
**Why NMF:** The "linear mixing model" says pixel = weighted sum of endmember spectra. This is exactly NMF: H rows = endmember spectra, W rows = per-pixel abundances. Non-negativity is physical (abundances sum to 1, each non-negative; reflectance is non-negative).
**NMF variants used:**
- Constrained NMF: add sum-to-one constraint on abundances
- Sparse NMF: each pixel contains only a few materials
- Robust NMF: bad scan lines, atmospheric correction errors treated as outliers
**Applications:** Mineral mapping for mining exploration, monitoring crop types for precision agriculture, urban mapping for city planning, snow/ice coverage for climate science.
**Review paper:** "Hyperspectral Unmixing Based on Nonnegative Matrix Factorization: A Comprehensive Review" (arXiv 2205.09933, 2022).

### 9.4 Microbial Community Structure (Metagenomics)

**Domain:** Microbial ecology, clinical microbiome research
**The problem:** Characterize the composition of microbial communities (gut, soil, ocean) from DNA sequencing data. Different environments/conditions (healthy gut vs. diseased gut) may have different "community types." We want to discover these latent community structures.
**Data:** OTU (Operational Taxonomic Unit) table or species abundance table: X is (n_samples × n_species). Non-negative counts.
**Why NMF:** Samples are mixtures of latent "sub-communities" (ecological guilds). NMF decomposes X into sub-community patterns (H: prototypical species compositions) and sample weights (W: how much each sub-community is present in each sample). Since species abundances are non-negative counts, NMF's non-negativity constraint is natural.
**Application (2024):** "Towards more accurate microbial source tracking via NMF" (ISMB 2024): uses NMF to track the source of microbial contamination (e.g., which animal species is contaminating water).
**Functional metagenomics (PLoS Comp Bio 2017):** NMF applied to fiber-degradation functional genes in human gut microbiota identified latent guilds corresponding to fiber-fermenting microbes vs. protein degraders.
**Tools:** sklearn NMF with KL divergence (counts), nimfa for consensus analysis.

### 9.5 Drug Combination Discovery (Pharmacogenomics)

**Domain:** Drug discovery, precision medicine
**The problem:** Drug repurposing and synergistic combination discovery. Given a large matrix of drug-gene interaction data (how each drug affects gene expression), find groups of drugs with similar mechanisms and identify novel target pathways.
**Data:** Drug-gene response matrix: X is (n_drugs × n_genes). Values are expression fold-changes (made non-negative by taking absolute values or splitting into up/down matrices).
**Why NMF:** Drugs tend to cluster by mechanism of action. NMF components identify drug "programs" (sets of co-regulated genes) and drug clusters (groups with similar programs). The non-negative decomposition gives interpretable gene signatures associated with each drug program.
**Specific application — Connectivity Map (CMap/LINCS):** The Broad Institute's CMap project profiles ~30,000 drugs against ~1,000 cell lines. NMF has been applied to identify drug mechanism clusters, enabling repurposing (drugs active on Alzheimer's targets being discovered in depression datasets).
**Activity cliffs and NMF:** In QSAR (quantitative structure-activity relationship), NMF on molecular fingerprint matrices can identify structural sub-patterns associated with activity. Activity cliffs (small structural change → large activity change) are challenging for any method.
**Reference:** PLoS Comp Bio 2008: NMF applied to pharmacogenomics profiling.

### 9.6 Single-Cell RNA-seq Gene Program Discovery

**Domain:** Computational biology, single-cell genomics
**The problem:** Single-cell RNA sequencing profiles gene expression in thousands of individual cells. Each cell is a sample; each gene is a feature. We want to find "gene programs" — co-regulated sets of genes that define cell states, cell types, or responses.
**Data:** Count matrix: X is (n_cells × n_genes). Non-negative integer counts (number of RNA molecules detected).
**Why NMF over PCA:** scRNA-seq data is non-negative (cannot have negative RNA counts). PCA components have negative loadings, which are biologically uninterpretable. NMF's non-negative factorization gives:
- H rows = gene programs (which genes are co-regulated in each program)
- W columns = cell-program weights (which programs are active in each cell)
**NMF variant:** KL-divergence NMF preferred (count data → Poisson noise model).
**Tool: GeneNMF (R):** Designed specifically for scRNA-seq on Seurat objects. Performs consensus NMF across multiple runs and multiple samples. Identifies robust gene programs.
**Applications:** Identifying cancer subclones, discovering cell state transitions in development, finding stress response programs, understanding tumor microenvironment.
**Challenge:** scRNA-seq data is very sparse (most genes are 0 in most cells). This is actually a feature for NMF, but requires careful preprocessing (log-normalization or Pearson residuals before NMF).

### 9.7 Text Topic Modeling (NMF vs LDA vs LSA Comparison)

**Domain:** Natural language processing, information retrieval
**The problem:** Given a corpus of documents, discover latent topics. What are the major themes in a large collection of news articles, scientific papers, or customer reviews?
**Data:** TF-IDF matrix or raw count matrix: X is (n_docs × n_vocab). Non-negative.

**NMF approach:**
- H rows = topics (word distributions for each topic)
- W rows = topic mixtures per document (how much of each topic is in each doc)

**Comparison with LSA/LSI:**
| Property | NMF | LSA/LSI |
|----------|-----|---------|
| Decomposition | W, H ≥ 0 | Singular vectors (can be negative) |
| Topic interpretability | High: additive, parts-based | Low: holistic, negative loadings |
| Uniqueness | Non-unique (without constraints) | Unique (SVD is unique) |
| Scalability | Slower | Fast (truncated SVD via ARPACK) |
| Statistical model | None (deterministic) | None (deterministic) |
| Polysemy handling | Moderate | Moderate |
| Sparsity | Naturally sparse with L1 reg | Not sparse |
| Implementation | sklearn NMF | sklearn TruncatedSVD |

**Comparison with LDA:**
- LDA: generative probabilistic model (Dirichlet-Multinomial). Strong statistical foundations. Topic words have proper probability distributions.
- NMF: no statistical model. But often produces **more coherent topics** empirically (Comparative studies 2023-2024 in IIETA).
- NMF: faster to train. LDA: requires sampling or variational inference.
- NMF: handles new documents easily (just call transform()). LDA: inference on new docs requires approximate methods.

**When to prefer NMF for topics:**
- Want interpretable, non-overlapping topics
- Have TF-IDF input (not raw counts) — TF-IDF is not count data, doesn't fit Poisson model → NMF with Frobenius better than LDA
- Need fast computation
- Documents are long (LDA struggles with very short texts)

**Practical code:**
```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(documents)  # sparse, non-negative

nmf = NMF(n_components=20, init='nndsvda', solver='cd', max_iter=500)
W = nmf.fit_transform(X)

feature_names = vectorizer.get_feature_names_out()
for i, comp in enumerate(nmf.components_):
    top_words = [feature_names[j] for j in comp.argsort()[:-11:-1]]
    print(f"Topic {i}: {', '.join(top_words)}")
```

### 9.8 Recommender Systems (Collaborative Filtering)

**Domain:** E-commerce, streaming services, social media
**The problem:** Predict which items (movies, products, songs) a user will like based on historical interactions.
**Data:** User-item interaction matrix X (n_users × n_items). Entries are ratings (1-5) or implicit feedback (0 or 1 for watch/no-watch). Non-negative.
**Why NMF:** Each user's preferences can be decomposed into a mixture of latent "taste profiles" (e.g., "likes action movies" or "prefers indie films"). NMF naturally gives non-negative user profiles (W) and non-negative item profiles (H), meaning every latent factor only adds to the predicted affinity — a user cannot have "negative affinity" for a genre, only zero.

**Contrast with SVD-based CF (used by Netflix Prize winner):**
- SVD (matrix factorization with negative values): More flexible, can represent anti-preferences. Used in PMF, SVD++.
- NMF: Enforces non-negativity → more interpretable components but potentially less accurate.

**Practical use in 2024:** While deep learning (two-tower models, autoencoders, transformer-based) has largely replaced NMF in production at large scale, NMF remains competitive for:
- Small/medium-scale recommenders (< 1M users, < 100K items)
- Scenarios where interpretability matters (explain why something was recommended)
- Cold start with side information

**sklearn implementation:**
```python
from sklearn.decomposition import NMF
import numpy as np

# R: user-item rating matrix (fill 0 for unobserved)
R = np.array([...])  # shape (n_users, n_items), non-negative

model = NMF(n_components=30, init='random', solver='mu',
            beta_loss='frobenius', max_iter=500, random_state=42)
W = model.fit_transform(R)  # user factors
H = model.components_        # item factors

# Predict rating for user u, item i
predicted_rating = W[u] @ H[:, i]
```

### 9.9 Wi-Fi / Mobile Traffic Pattern Discovery (Smart Cities)

**Domain:** Urban computing, network engineering, smart cities
**The problem:** Understand population movement patterns in cities from Wi-Fi user count data. Discover recurring daily patterns (commute rush, lunch hour, evening activity).
**Data:** Count matrix X (n_locations × n_time_slots). Non-negative counts of connected users.
**Why NMF:** Each location's daily traffic is a mixture of a few basic "activity patterns" (morning rush, lunch peak, evening rush, always-busy airport). NMF components = archetypal patterns; encodings = how much each archetype appears at each location.
**Reference:** "Identifying Population Movements with Non-Negative Matrix Factorization from Wi-Fi User Counts in Smart and Connected Cities" (arXiv 2111.10459).
**Application:** City planners use these patterns to optimize transit schedules, retail operators use them to staff stores.

---

## 10. HYPERPARAMETER TUNING GUIDANCE

### 10.1 `n_components` (THE MOST IMPORTANT HYPERPARAMETER)

**What it controls:** The rank of the factorization. The number of latent components/topics/patterns.

**Recommended ranges:**
- Text topic modeling: 5-50 (typically 10-30)
- Image/signal analysis: 5-100
- Genomics (mutational signatures): 3-15
- scRNA-seq gene programs: 10-50
- Recommender systems: 10-200

**Tuning methods:**

1. **Reconstruction error elbow plot:**
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

errors = []
n_components_range = range(2, 50)
for k in n_components_range:
    model = NMF(n_components=k, init='nndsvda', max_iter=500, random_state=42)
    model.fit(X)
    errors.append(model.reconstruction_err_)

plt.plot(n_components_range, errors, 'bo-')
plt.xlabel('n_components')
plt.ylabel('Reconstruction Error')
plt.title('NMF Reconstruction Error vs. n_components')
```
Look for an "elbow" where adding more components gives diminishing returns.

2. **Consensus matrix (cophenetic correlation) — for stability:**
- Run NMF K times with different random seeds for each value of k
- Build consensus matrix (fraction of runs where two samples are in the same dominant component)
- Compute cophenetic correlation of consensus matrix
- Choose k that maximizes cophenetic correlation (implemented in nimfa)

3. **Cross-validated reconstruction error:**
- Hold out a random subset of matrix entries (set to 0)
- Fit NMF and check reconstruction error on held-out entries
- Analogous to matrix completion cross-validation

4. **Domain knowledge:** Often the best guide. If you know there are 5 soil types in your hyperspectral data, use n_components=5.

**Too low:** Under-fitting. The model can't capture all meaningful variation. High reconstruction error. Topics are too broad/generic.

**Too high:** Redundant components (two similar topics covering the same theme). Individual topics become hard to interpret. Risk of overfitting. Computational cost increases linearly.

**Optuna search space:**
```python
import optuna
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score

def objective(trial):
    k = trial.suggest_int('n_components', 5, 50, log=True)
    alpha_H = trial.suggest_float('alpha_H', 1e-4, 10.0, log=True)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    model = NMF(
        n_components=k,
        init='nndsvda',
        solver='cd',
        alpha_W=0.0,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        max_iter=500,
        random_state=42,
    )
    W = model.fit_transform(X_train)
    return model.reconstruction_err_  # minimize

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### 10.2 `init` (INITIALIZATION STRATEGY)

**Recommended defaults:**
- First try: `'nndsvda'` (sklearn default for small k)
- For sparse data / sparse components: `'nndsvd'` (more zeros)
- With MU solver: `'nndsvda'` or `'random'` (MU gets stuck at zeros if init='nndsvd')
- For reproducibility + robustness: Run 3-5 times with `'random'` and different seeds, take best reconstruction error

**When random restarts matter:**
```python
best_err = float('inf')
best_model = None
for seed in range(5):
    model = NMF(n_components=20, init='random', random_state=seed, max_iter=500)
    W = model.fit_transform(X)
    if model.reconstruction_err_ < best_err:
        best_err = model.reconstruction_err_
        best_model = model
```

**Too few restarts:** Risk of stuck in poor local minimum. Symptoms: very high reconstruction error, degenerate components (one component captures nearly all variance).

### 10.3 `alpha_W`, `alpha_H`, `l1_ratio` (REGULARIZATION)

**No regularization (defaults: alpha_W=0, alpha_H='same'):**
- Appropriate when interpretability is not critical
- For dense data where components aren't expected to be sparse

**L1 regularization (`l1_ratio=1.0`):**
- Promotes sparsity — each sample uses few components, each component uses few features
- Numerically: soft-thresholds entries toward zero
- Good for: topic models, source separation, anything where "parts" should be sparse

**L2 regularization (`l1_ratio=0.0`):**
- Prevents component magnitudes from growing unbounded
- Smoother components
- Not often used alone for NMF (less interpretable)

**Elastic net (`0 < l1_ratio < 1`):**
- Combination: sparse but stable
- Useful when exact zeros are not needed but sparsity is preferred

**Typical ranges:**
- `alpha_H` (regularize encodings): 0.001 to 1.0 (start with 0.1)
- `l1_ratio`: 0.0, 0.5, or 1.0 (discrete choice, tune as categorical)

**Too high alpha:** Components collapse (H → all zeros, W absorbs all variance or vice versa). Reconstruction error skyrockets.

**Too low alpha (near 0):** No regularization effect. Dense, non-sparse components.

### 10.4 `max_iter` (CONVERGENCE)

**Default 200 is often insufficient:**
- MU solver with KL divergence: needs 500-2000 iterations
- CD solver with Frobenius: usually fine with 200-500
- Very large matrices: need more iterations

**Practical check:**
```python
if model.n_iter_ == model.max_iter:
    print(f"WARNING: Did not converge! Try max_iter={model.max_iter * 3}")
```

**Computational tradeoff:** Each additional 100 iterations adds linear time cost. Balance convergence vs. speed.

### 10.5 `beta_loss` (LOSS FUNCTION)

**Decision tree:**
```
Is your data counts (integers ≥ 0)?
├── Yes: Use beta_loss='kullback-leibler' with solver='mu'
│   └── Data is audio spectrogram? Use beta_loss='itakura-saito' with solver='mu'
└── No (real-valued, already non-negative):
    └── Use beta_loss='frobenius' with solver='cd' (fastest)
```

**Speed ranking:** 'frobenius' with 'cd' >> 'kullback-leibler' with 'mu' >> 'itakura-saito' with 'mu'

### 10.6 `solver` (ALGORITHM)

**Decision:**
- Always use `'cd'` unless you need KL or IS divergence
- `'mu'` required for non-Frobenius beta_loss
- `'mu'` also needed if you have a custom beta_loss float value

### 10.7 MiniBatchNMF `batch_size` and `forget_factor`

**batch_size:**
- Default 1024 is reasonable
- Scale with dataset: for 10M samples, consider batch_size=4096-8192
- Larger = more stable gradients, slower per-epoch progress

**forget_factor:**
- Finite static large dataset: 0.9-1.0
- Online streaming (concept drift possible): 0.5-0.7
- Quick adaptation to new distribution: 0.2-0.5

---

## 11. PRACTITIONER MENTAL MODEL / FITTING CHECKLIST

### Before Fitting

**1. Check non-negativity of input:**
```python
assert (X >= 0).all(), "NMF requires non-negative input! Check your data."
# For sparse matrices:
assert X.data.min() >= 0, "NMF requires non-negative sparse input!"
```
**Common sources of negative values:**
- Mean-subtracted data (use Semi-NMF instead, or add abs())
- Log-ratio transformed data
- Difference matrices

**2. Check for missing values:**
```python
import numpy as np
print(f"NaN count: {np.isnan(X.toarray()).sum()}")  # NMF cannot handle NaN natively
# Impute with 0 or use matrix completion NMF (nimfa)
```

**3. Scale/preprocessing decisions:**
- Raw counts: Good for KL-NMF. Or apply log(1+x) for more balanced scale.
- TF-IDF: Good for Frobenius-NMF with text. (TF-IDF is non-negative by construction)
- Pixel values: Normalize to [0,1]. Already non-negative.
- Gene expression: Log-normalize (standard scRNA-seq: log(CPM+1)) or use Pearson residuals.
- **Do NOT standardize (zero-mean) before NMF** — this introduces negative values.

**4. Choose n_components heuristic:**
- Start with n_components = √(n_features / 2) as a rough estimate
- For topic models: 10-20 as default starting point
- For source separation: roughly equal to expected number of sources
- Use reconstruction error curve to refine

**5. Choose init and solver:**
- Default: init='nndsvda', solver='cd' — works in most cases
- For count data with KL: init='nndsvda', solver='mu', beta_loss='kullback-leibler'
- For audio: init='random', solver='mu', beta_loss='itakura-saito'

### During Fitting

**6. Monitor convergence:**
```python
model = NMF(n_components=20, verbose=1, max_iter=500)  # verbose=1 prints progress
W = model.fit_transform(X)
print(f"Iterations used: {model.n_iter_}")
print(f"Converged: {model.n_iter_ < model.max_iter}")
```

**7. Track reconstruction error across trials (if using random init):**
```python
trials = []
for seed in range(5):
    m = NMF(n_components=k, init='random', random_state=seed, max_iter=500)
    m.fit_transform(X)
    trials.append((m.reconstruction_err_, m))
trials.sort(key=lambda x: x[0])
best_model = trials[0][1]
```

### After Fitting

**8. Inspect component quality:**
```python
H = model.components_  # (n_components, n_features)

# For text: print top words per topic
for i, topic in enumerate(H):
    top_idx = topic.argsort()[-10:][::-1]
    top_terms = [vocab[j] for j in top_idx]
    print(f"Topic {i}: {', '.join(top_terms)}")

# For images: visualize each component as an image
import matplotlib.pyplot as plt
for i, comp in enumerate(H):
    plt.subplot(4, 5, i+1)
    plt.imshow(comp.reshape(image_shape), cmap='gray')
plt.suptitle('NMF Components')
```

**9. Check sparsity of encodings:**
```python
W = model.fit_transform(X)
sparsity = (W == 0).mean()
print(f"Encoding sparsity: {sparsity:.1%}")
# If too dense (< 10% zeros), consider adding L1 regularization
# If too sparse (> 90% zeros), reduce regularization
```

**10. Check reconstruction quality:**
```python
X_reconstructed = W @ model.components_
relative_error = np.linalg.norm(X - X_reconstructed, 'fro') / np.linalg.norm(X, 'fro')
print(f"Relative reconstruction error: {relative_error:.4f}")
# Typical acceptable range: 0.01 - 0.20 depending on task
# Topic models: 0.05-0.30 is typical (TF-IDF not perfectly factorizable)
# Image compression: 0.01-0.10 is typical
```

### Common Pitfalls

**Pitfall 1: Degenerate solutions (one component dominates)**
- Symptom: One row of H has very large values; all others near zero. W has one dominant column.
- Cause: Bad initialization (all mass goes to one component)
- Fix: Use nndsvda init, or run multiple random restarts and pick lowest reconstruction error

**Pitfall 2: Component duplication**
- Symptom: Two or more components are nearly identical
- Cause: n_components is too high, or regularization too low
- Fix: Reduce n_components, or add L1 regularization

**Pitfall 3: Non-convergence with MU solver**
- Symptom: n_iter_ == max_iter, reconstruction error still decreasing
- Fix: Increase max_iter (try 1000+), or switch to solver='cd' if using Frobenius loss

**Pitfall 4: Negative values cause errors**
- Symptom: `ValueError: NMF (input X) is not non-negative`
- Fix: Check preprocessing pipeline for subtraction or centering steps

**Pitfall 5: Memory errors with full NMF on large data**
- Symptom: MemoryError or OOM during NNDSVD initialization (computes full SVD)
- Fix: Use init='random' for very large matrices (NNDSVD requires computing SVD of X), or use MiniBatchNMF

**Pitfall 6: Comparing solutions across runs**
- Problem: Components may be permuted differently in different runs (column permutation ambiguity)
- Fix: Match components by cosine similarity between runs. Or use consensus NMF (nimfa).

**Pitfall 7: ConvergenceWarning silently ignored**
```python
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('error', category=ConvergenceWarning)
# Now convergence failures will raise errors, not silently pass
```

---

## 12. EVALUATION METRICS

### 12.1 Reconstruction Error (Primary Metric)

**Frobenius reconstruction error:**
```python
import numpy as np
W = model.fit_transform(X)
H = model.components_
err = np.linalg.norm(X - W @ H, 'fro')
# sklearn also stores this: model.reconstruction_err_
```

**Relative reconstruction error:**
```python
rel_err = np.linalg.norm(X - W @ H, 'fro') / np.linalg.norm(X, 'fro')
# Normalize by scale of X for comparability across datasets
```

**Cross-validated reconstruction error (for model selection):**
```python
from sklearn.utils import check_random_state
rng = check_random_state(42)

# Mask random entries
mask = rng.random(X.shape) > 0.8  # 20% held out
X_masked = X.copy()
X_masked[mask] = 0

model = NMF(n_components=k, init='nndsvda', max_iter=500)
W = model.fit_transform(X_masked)
X_pred = W @ model.components_

# Error on held-out entries only
held_out_err = np.sqrt(((X[mask] - X_pred[mask]) ** 2).mean())
```

### 12.2 Topic Coherence (for Text NMF)

Measures whether the top words in each topic are semantically related. The C_v metric is standard.

```python
# Using gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# Convert top words per topic to gensim format
topics_words = []
for topic_vec in model.components_:
    top_idx = topic_vec.argsort()[-10:][::-1]
    topics_words.append([feature_names[i] for i in top_idx])

# Build gensim corpus
texts = [doc.split() for doc in documents]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

cm = CoherenceModel(topics=topics_words, texts=texts,
                    dictionary=dictionary, coherence='c_v')
coherence_score = cm.get_coherence()
print(f"Topic coherence (C_v): {coherence_score:.4f}")
# Higher is better. Typical good scores: 0.50-0.70
```

### 12.3 Topic Diversity (for Text NMF)

Fraction of unique words across top-10 words of all topics. Ensures topics are distinct, not redundant.

```python
all_top_words = []
for topic_vec in model.components_:
    top_idx = topic_vec.argsort()[-10:][::-1]
    all_top_words.extend([feature_names[i] for i in top_idx])

diversity = len(set(all_top_words)) / len(all_top_words)
print(f"Topic diversity: {diversity:.4f}")
# Range [0,1]; higher = more diverse topics. Good: > 0.7
```

### 12.4 Perplexity (Generative Model Analog)

Primarily for LDA, but can be approximated for NMF by treating normalized W rows as document-topic probabilities:

```python
# Normalize W to probabilities
W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)
H_norm = H / (H.sum(axis=1, keepdims=True) + 1e-10)

# Compute log-likelihood under document reconstruction
log_likelihood = np.sum(X * np.log(W_norm @ H_norm + 1e-10))
n_tokens = X.sum()
perplexity = np.exp(-log_likelihood / n_tokens)
# Lower is better. Compare NMF perplexity vs LDA perplexity on same corpus.
```
**Note:** Perplexity for NMF is not as well-defined as for LDA (which has a proper generative model). C_v coherence is a better metric.

### 12.5 Trustworthiness and Continuity (Neighborhood Preservation)

Standard metrics for dimensionality reduction: do nearby points in the original space stay nearby in the low-dimensional encoding (W)?

```python
from sklearn.manifold import trustworthiness

# X: original data (n_samples, n_features)
# W: NMF encoding (n_samples, n_components)
trust = trustworthiness(X, W, n_neighbors=10)
print(f"Trustworthiness (k=10): {trust:.4f}")
# Range [0,1]; 1.0 = perfect neighborhood preservation
```

**Interpretation:** Trustworthiness penalizes when points that are far in X appear close in W (false neighbors). Continuity penalizes when points that are close in X appear far in W (missing neighbors). NMF typically has lower trustworthiness than t-SNE/UMAP for visualization purposes, but that's not NMF's primary goal (it's a parts-decomposition, not a manifold embedding).

### 12.6 Silhouette Score (Cluster Quality of Encodings)

If you have class labels, check how well NMF encodings separate classes:

```python
from sklearn.metrics import silhouette_score

# Use dominant component assignment as cluster label
cluster_labels = W.argmax(axis=1)
score = silhouette_score(W, cluster_labels)
print(f"Silhouette score: {score:.4f}")
# Range [-1, 1]; higher is better. > 0.5 is good clustering.
```

### 12.7 Cophenetic Correlation (Stability Metric)

For selecting n_components: run NMF multiple times and measure stability.

```python
# nimfa implements this natively
import nimfa

def consensus_nmf(X, k, n_runs=10):
    """Run NMF k times and return consensus matrix."""
    C = np.zeros((X.shape[0], X.shape[0]))
    for _ in range(n_runs):
        nmf = nimfa.Nmf(X, rank=k, seed='random', max_iter=500)
        fit = nmf()
        W = np.array(fit.basis())
        labels = W.argmax(axis=1)
        # Add co-occurrence to consensus
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if labels[i] == labels[j]:
                    C[i, j] += 1
    return C / n_runs

# Cophenetic correlation: how well does consensus matrix
# match the hierarchical clustering of it?
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

C = consensus_nmf(X, k=15)
Z = linkage(squareform(1 - C), method='average')
coph_corr, _ = cophenet(Z, squareform(1 - C))
print(f"Cophenetic correlation for k=15: {coph_corr:.4f}")
# Higher (closer to 1.0) = more stable factorization at this k
```

### 12.8 Reconstruction Error on Heldout Data vs. Training Data

Track both to detect overfitting:

```python
from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

model = NMF(n_components=20, init='nndsvda', max_iter=500)
W_train = model.fit_transform(X_train)
W_val = model.transform(X_val)  # Use learned H to encode validation

train_err = np.linalg.norm(X_train - W_train @ model.components_, 'fro')
val_err = np.linalg.norm(X_val - W_val @ model.components_, 'fro')

print(f"Train reconstruction: {train_err:.4f}")
print(f"Val reconstruction: {val_err:.4f}")
# If val_err >> train_err, consider reducing n_components or increasing regularization
```

### 12.9 Interpretability Score (Manual / Downstream)

**For text:** Ask humans to evaluate topic coherence on a 1-5 scale. Or use automatic measures like PMI (Pointwise Mutual Information) between top topic words.

**For images:** Visualize components and check if they correspond to recognizable parts.

**For genomics:** Check if discovered signatures match known COSMIC signatures using cosine similarity.

**For recommender systems:** RMSE on held-out ratings:
```python
# Compare predicted vs actual ratings on held-out (user, item) pairs
test_pairs = [(user, item, rating) for ...]
predictions = [W[u] @ model.components_[:, i] for u, i, r in test_pairs]
actuals = [r for u, i, r in test_pairs]
rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
print(f"Recommendation RMSE: {rmse:.4f}")
```

---

## APPENDIX A: KEY VERSION CHANGES (sklearn)

| sklearn Version | Change | Impact |
|----------------|--------|--------|
| v0.17 | Added solver='cd', l1_ratio | CD solver introduced |
| v0.19 | Added solver='mu', beta_loss | MU solver and KL/IS divergences added |
| v0.24 | Added n_features_in_ attribute | Minor |
| v1.0 | Removed alpha, regularization; added alpha_W, alpha_H | **BREAKING CHANGE** — old code breaks |
| v1.1 | Default init changed from 'nndsvd' to 'nndsvda'; MiniBatchNMF added | Behavior change + new class |
| v1.4 | n_components='auto' option added | New feature |

---

## APPENDIX B: MATHEMATICAL GLOSSARY

| Symbol | Meaning |
|--------|---------|
| X | Input data matrix, shape (n_samples, n_features). Must be ≥ 0. |
| W | Sample encoding matrix, shape (n_samples, n_components). Must be ≥ 0. |
| H | Component/dictionary matrix, shape (n_components, n_features). Must be ≥ 0. sklearn stores this as `model.components_`. |
| k | Number of components (rank of factorization). sklearn parameter: `n_components`. |
| β | Beta-divergence parameter. β=2: Frobenius, β=1: KL, β=0: IS. |
| α_W | Regularization strength on W. sklearn: `alpha_W`. |
| α_H | Regularization strength on H. sklearn: `alpha_H`. |
| ρ | L1 ratio (elastic net mixing). sklearn: `l1_ratio`. 0=L2, 1=L1. |
| ⊙ | Elementwise (Hadamard) product |
| || · ||_F | Frobenius norm (square root of sum of squared elements) |
| || · ||_1 | L1 norm (sum of absolute values) |

---

## APPENDIX C: QUICK DECISION GUIDE

```
What is your goal?
├── Parts-based decomposition (faces, signals, images)
│   → NMF with Frobenius loss, CD solver, nndsvda init
│
├── Text topic modeling
│   ├── Have TF-IDF features → Frobenius + CD + nndsvda
│   └── Have raw counts → KL divergence + MU + nndsvda
│
├── Audio source separation
│   → Itakura-Saito + MU + random init (or 'nndsvda')
│
├── Genomics (scRNA, mutational signatures)
│   → KL divergence + MU + nndsvda + consensus over multiple runs
│
├── Large dataset (millions of samples)
│   → MiniBatchNMF with appropriate forget_factor
│
├── Interpretable, parts-based but data has mixed signs
│   → Semi-NMF (not in sklearn; use custom implementation)
│
└── Want archetypes / convex combinations
    → Convex NMF (not in sklearn; use custom or nimfa)
```

---

*End of Research Dossier. Writing agent: use this dossier as your sole factual reference. All sklearn code uses verified 1.5.x syntax. Flag any implementation details not covered here before writing.*
