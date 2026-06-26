# Research Dossier: Independent Component Analysis (ICA)
## For Dimensionality Reduction Masterclass — Chapter Writing Agent

**Prepared by:** Deep Research Agent  
**Date:** 2026-06-26  
**Target output:** 18,000+ token expert chapter on ICA  
**Writing agent instructions:** You have ONLY this dossier and the course bible. Every claim below is sourced. Use this as your sole reference.

---

## SECTION 1 — PAPER LOOKUPS (Full Citations)

### Paper 1: Bell & Sejnowski 1995 (InfoMax / ICA founding paper)

**Full Citation:**  
Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. *Neural Computation*, 7(6), 1129–1159.

**DOI:** 10.1162/neco.1995.7.6.1129  
**ResearchGate PDF:** https://www.researchgate.net/publication/15614030_An_Information-Maximization_Approach_to_Blind_Separation_and_Blind_Deconvolution  
**Semantic Scholar:** https://www.semanticscholar.org/paper/An-Information-Maximization-Approach-to-Blind-and-Bell-Sejnowski/1d7d0e8c4791700defd4b0df82a26b50055346e0

**What you learn from it (2–3 sentences):**  
Bell & Sejnowski derived a self-organizing learning algorithm that maximizes the mutual information (Shannon entropy) flowing through a network of nonlinear units — this is the InfoMax principle. The key insight is that maximizing output entropy (rather than minimizing squared reconstruction error as PCA does) implicitly forces the outputs to be statistically independent when passed through appropriate nonlinearities. The algorithm makes no assumptions about input distributions and learns the mixing matrix purely from observed data — this is "blind" source separation.

**What it introduced:**  
- The InfoMax framework for ICA
- Learning rule: ΔW ∝ [I − φ(y)yᵀ]W, where φ is a nonlinear activation derivative
- Later shown to be equivalent to maximum likelihood ICA and mutual information minimization
- The algorithm was subsequently recognized as implementing a particular form of ICA

**Historical context:**  
Prior art included PCA (Pearson 1901), which only removes second-order correlations (covariance). Bell & Sejnowski's insight: to truly separate independent sources, you need to go beyond covariance to higher-order statistics.

**Difficulty level:** Graduate (requires information theory background). The math is elegant but demands familiarity with entropy, mutual information, and the chain rule for information.

---

### Paper 2: Hyvärinen & Oja 1997 FastICA

**Full Citation:**  
Hyvärinen, A., & Oja, E. (1997). A fast fixed-point algorithm for independent component analysis. *Neural Computation*, 9(7), 1483–1492.

**DOI:** 10.1162/neco.1997.9.7.1483  
**MIT Press direct:** https://direct.mit.edu/neco/article/9/7/1483/6120/A-Fast-Fixed-Point-Algorithm-for-Independent  
**ACM DL:** https://dl.acm.org/doi/10.1162/neco.1997.9.7.1483

**What you learn from it (2–3 sentences):**  
This paper transforms the ICA learning rule into a fixed-point iteration (Newton step in disguise) that converges quadratically rather than linearly, making ICA practical on real datasets of thousands of samples. The algorithm finds one independent component at a time (deflation) or all simultaneously (parallel), requires zero user-defined learning rate, and is orders of magnitude faster than gradient methods. It has no free parameters other than the contrast function (nonlinearity), which encodes assumptions about source distributions.

**Core algorithm (fixed-point form):**  
```
w ← E{x · g(wᵀx)} − E{g'(wᵀx)} · w
w ← w / ||w||
```
where g is the derivative of G (the negentropy approximation function), and x is the whitened data.

**Difficulty level:** Advanced undergraduate / early graduate. The fixed-point derivation is elegant once you see the Newton step, but the whitening preprocessing needs explaining first.

---

### Paper 3: Hyvärinen 1999 Survey

**Full Citation:**  
Hyvärinen, A. (1999). Survey on independent component analysis. *Neural Computing Surveys*, 2, 94–128.

**Aalto repository:** https://research.aalto.fi/en/publications/survey-on-independent-component-analysis/

**What you learn from it (2–3 sentences):**  
This is the definitive survey of the field at the time of consolidation, covering all ICA definitions, all major estimation principles (negentropy, mutual information minimization, maximum likelihood, tensorial methods), and all major algorithms (FastICA, InfoMax, JADE, SOBI). It is the paper to read for a rigorous treatment of the relationship between these seemingly different formulations (they are equivalent under mild conditions). It also covers the ambiguities of ICA (scale, sign, permutation indeterminacies) and the conditions for identifiability.

**Difficulty level:** Advanced graduate. Highly mathematical, covering cumulants, differential entropy, and matrix calculus. Excellent as a reference once you understand the basics.

---

### Paper 4: Hyvärinen & Oja 2000 (Review paper — most cited)

**Full Citation:**  
Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4–5), 411–430.

**DOI:** 10.1016/S0893-6080(00)00026-5  
**ScienceDirect:** https://www.sciencedirect.com/science/article/abs/pii/S0893608000000265

**What you learn from it:**  
This is the workhorse reference — more accessible than the 1999 survey but still rigorous. Covers the model, ambiguities, preprocessing (centering, whitening), and all major estimation approaches. Serves as sklearn's own reference in its documentation.

---

### Paper 5: Cardoso & Souloumiac 1993 — JADE

**Full Citation:**  
Cardoso, J.-F., & Souloumiac, A. (1993). Blind beamforming for non-Gaussian signals. *IEE Proceedings F — Radar and Signal Processing*, 140(6), 362–370.

**Also relevant:** Cardoso, J.-F. (1989). Source separation using higher order moments. *ICASSP 1989*. [This introduced FOBI, the precursor]

**What you learn from it (2–3 sentences):**  
JADE (Joint Approximate Diagonalization of Eigenmatrices) exploits fourth-order cumulants (kurtosis-related statistics) that vanish for Gaussian distributions — it simultaneously diagonalizes a set of cumulant matrices via Jacobi rotations. Unlike gradient-based methods, JADE converges reliably and handles both sub- and super-Gaussian sources equally well. JADE is particularly robust in the low-sample regime and is the algorithm underlying MNE-Python's ICA `method='infomax'` with `extended=True` variant.

---

## SECTION 2 — THEORETICAL FOUNDATIONS (For the Writing Agent)

### The ICA Model

The generative model for ICA is:

```
x = A · s
```

Where:
- **x** ∈ ℝⁿ = observed mixed signals (n sensors)
- **s** ∈ ℝⁿ = independent source signals (n sources)
- **A** ∈ ℝⁿˣⁿ = unknown mixing matrix (to be estimated)

The goal is to find the **unmixing matrix W = A⁻¹** such that:

```
y = W · x ≈ s
```

**The three fundamental ambiguities of ICA** (always mention these — they are intrinsic, not bugs):
1. **Scale ambiguity**: Cannot determine the variance of each source (arbitrary scaling factor D absorbed into A: A = A·D⁻¹ paired with s = D·s gives same x)
2. **Sign ambiguity**: Cannot determine the sign of each source (−1 is just a special case of scale)
3. **Permutation ambiguity**: Cannot determine the ordering of sources

**Identifiability condition**: ICA is identifiable if and only if at most one source is Gaussian. This is the key constraint. Two or more Gaussian sources cannot be separated because any rotation of Gaussian variables is still Gaussian.

---

### The Cocktail Party Problem

The canonical motivation: imagine a cocktail party with N speakers and N microphones placed at different locations. Each microphone records a different linear mixture of all speakers' voices:

```
x_i(t) = a_{i1}·s₁(t) + a_{i2}·s₂(t) + ... + a_{iN}·sN(t)
```

ICA recovers each speaker's voice from only the microphone recordings, without knowing the room geometry (mixing matrix A) or even the number of speakers — purely from the statistical structure of the recordings.

**Why PCA fails here**: PCA finds orthogonal axes of maximum variance, which are linear combinations of sources. PCA can decorrelate but cannot separate, because independence is a stronger condition than uncorrelation. Two variables can be uncorrelated (zero covariance) but still statistically dependent (e.g., X ~ N(0,1) and Y = X²).

**Why non-Gaussianity is key** (Central Limit Theorem reversal): When you mix independent non-Gaussian sources, the mixture becomes "more Gaussian" than any individual source (CLT). Therefore, to undo the mixing and recover independent sources, you look for directions where the projected distribution is maximally non-Gaussian. This is the fundamental intuition.

---

### Measures of Non-Gaussianity

**1. Kurtosis (fourth-order cumulant):**
```
kurt(y) = E{y⁴} − 3(E{y²})²
```
- Gaussian variables have kurtosis = 0
- Super-Gaussian (heavy-tailed, e.g., Laplace): kurtosis > 0
- Sub-Gaussian (platykurtic, e.g., uniform): kurtosis < 0
- FastICA maximizes |kurtosis| to find non-Gaussian directions

**Problem with kurtosis**: Extremely sensitive to outliers. A single data point far from zero can dominate the fourth moment.

**2. Negentropy (more robust):**
```
J(y) = H(y_Gaussian) − H(y)
```
Where H is differential entropy. Negentropy is zero for Gaussian, always non-negative. Measures how "far from Gaussian" a distribution is.

**Approximation (Hyvärinen 1997–1999):**
```
J(y) ≈ [E{G(y)} − E{G(ν)}]²
```
Where ν ~ N(0,1) and G is a non-quadratic function. In sklearn, `fun` parameter controls G:
- `'logcosh'`: G(u) = (1/α)log(cosh(α·u)) — smooth approximation, best general choice
- `'exp'`: G(u) = −exp(−u²/2) — emphasizes super-Gaussian sources
- `'cube'`: G(u) = u⁴/4 — equivalent to kurtosis (unstable for outliers)

**3. Mutual Information (most principled):**
```
I(y₁, y₂, ..., yₙ) = Σᵢ H(yᵢ) − H(y)
```
ICA minimizes mutual information between components. Shown to be equivalent to negentropy maximization under whitening.

---

### The InfoMax Connection

Bell & Sejnowski's InfoMax maximizes:
```
H(f(Wx)) = log|det W| + Σᵢ H(f(wᵢᵀx))
```
Where f is a nonlinear squashing function. The key insight: if f = CDF of the true source distribution, then InfoMax = maximum likelihood = mutual information minimization. All roads lead to the same mountain.

**Learning rule** (gradient ascent on mutual information):
```
ΔW ∝ (Wᵀ)⁻¹ + g(y)xᵀ
```
Where g(y) = f'(y)/f(y) (score function of the source density).

---

### JADE Algorithm Details

JADE uses cumulant matrices (fourth-order statistics):

1. **Whiten** the data: Z = K·X where K is the whitening matrix (from PCA/EVD)
2. **Compute** the set of all cumulant matrices Qᵢⱼ of the whitened data
3. **Joint approximate diagonalization**: Find orthogonal matrix V such that V diagonalizes all Qᵢⱼ simultaneously
4. The Jacobi sweep algorithm iterates rotations to minimize off-diagonal elements

**Advantages over FastICA:**
- No iteration convergence issues (Jacobi sweeps always converge)
- Works on both sub- and super-Gaussian sources without choosing a nonlinearity
- More reliable in low-sample settings
- Used in EEGLAB and MNE as the `'infomax'` extended method

**Python availability**: The `jade` package on PyPI, or via MNE's built-in implementation.

---

### ICA vs PCA: The Definitive Comparison

| Aspect | PCA | ICA |
|--------|-----|-----|
| **Objective** | Maximize variance | Maximize statistical independence |
| **Constraint** | Orthogonality of components | Independence (higher-order stats) |
| **Statistics used** | 2nd order (covariance) | Higher-order (3rd, 4th moments) |
| **Distribution assumption** | None (works for any) | Sources must be non-Gaussian |
| **Result uniqueness** | Unique (eigenvectors) | Up to sign + permutation |
| **Preprocessing** | Just center | Must whiten (PCA step inside ICA) |
| **Components ordering** | By variance explained | Arbitrary order |
| **Use case** | Compression, noise reduction | Source separation, feature extraction |
| **Gaussian data** | Works fine | Fails (cannot separate) |

**Critical relationship**: ICA *uses* PCA internally. The whitening step (sphering) in FastICA is essentially PCA — it removes second-order dependencies and equalizes variances. ICA then rotates in the whitened space to remove higher-order dependencies. Think of it as: PCA takes you to an uncorrelated basis; ICA rotates further to an independent basis.

---

## SECTION 3 — SKLEARN 1.5.x API (Verified from official docs)

**Source:** https://scikit-learn.org/1.5/modules/generated/sklearn.decomposition.FastICA.html

### Class Signature
```python
sklearn.decomposition.FastICA(
    n_components=None,
    *,
    algorithm='parallel',
    whiten='unit-variance',
    fun='logcosh',
    fun_args=None,
    max_iter=200,
    tol=0.0001,
    w_init=None,
    whiten_solver='svd',
    random_state=None
)
```

### Complete Parameter Reference

**`n_components`** — `int` or `None`, default `None`
- Number of ICA components to extract
- If `None`, uses all features (n_features); this is rarely what you want
- Should be ≤ n_features (constraint from the linear ICA model)
- Practical guidance: set explicitly. Start with domain knowledge (e.g., for EEG: 0.9 × n_channels)
- Note: when `whiten=False`, n_components must equal n_features

**`algorithm`** — `{'parallel', 'deflation'}`, default `'parallel'`
- `'parallel'`: All components extracted simultaneously. Faster in practice (vectorized), but may converge to a different local optimum on each run
- `'deflation'`: Extracts one component at a time by projecting out already-found components via Gram-Schmidt orthogonalization. Slower but potentially more robust for a small number of target components
- **Practical guidance**: Use `'parallel'` for most tasks. Use `'deflation'` when you only need the top few components and stability matters more than speed

**`whiten`** — `str` or `bool`, default `'unit-variance'`
- `'unit-variance'`: Whitens AND rescales so each recovered source has unit variance. **(DEFAULT since v1.3)**
- `'arbitrary-variance'`: Whitens but does not rescale. The classic behavior before v1.3.
- `False`: Assumes data is already whitened (no preprocessing). You must supply pre-whitened data.
- **VERSION CHANGE — CRITICAL**: In v1.2 and earlier, default was `True` (equivalent to `'unit-variance'`). In v1.3, `True` was deprecated; use string options. Check your version.
- **Practical guidance**: Almost always use `'unit-variance'` (default). Use `False` only if you are calling FastICA multiple times and want to avoid redundant whitening.

**`fun`** — `{'logcosh', 'exp', 'cube'}` or callable, default `'logcosh'`
- Specifies the G function for negentropy approximation
- `'logcosh'`: G(u) = log(cosh(u)). Smooth, robust, good general default. Handles both sub- and super-Gaussian sources.
- `'exp'`: G(u) = −exp(−u²/2). Emphasizes finding super-Gaussian components (sparse). Slightly faster per iteration.
- `'cube'`: G(u) = u⁴/4. Equivalent to kurtosis maximization. Fast but VERY sensitive to outliers. Avoid unless data is clean and you know sources are super-Gaussian.
- Custom callable: Must return a tuple `(G(x), g(x))` where g = G'. Example:
  ```python
  def my_g(x):
      return x**3, (3 * x**2).mean(axis=-1)
  ```
- **Practical guidance**: Start with `'logcosh'` (default). Switch to `'exp'` for sparse signals (text, spike trains). Never use `'cube'` on real data.

**`fun_args`** — `dict` or `None`, default `None`
- Arguments passed to the G function
- When `fun='logcosh'`, defaults to `{'alpha': 1.0}`. Alpha controls the shape:
  - α close to 1: Emphasizes sub-Gaussian sources
  - α close to 2 (max): Emphasizes super-Gaussian sources
- For `'exp'` and `'cube'`: no extra arguments needed
- **Practical guidance**: Leave at `None` for logcosh. Tune `alpha` (range 1.0–2.0) if you know your source type.

**`max_iter`** — `int`, default `200`
- Maximum number of fixed-point iterations
- If convergence warning fires, increase to 500 or 1000
- **Practical guidance**: If you see `ICA did not converge. Consider increasing tolerance or number of iterations`, first try increasing to 500. If that fails, try a different `fun` or check your data.

**`tol`** — `float`, default `1e-4`
- Convergence tolerance: iteration stops when `||W_new − W_old||_max < tol`
- Larger tol → faster but less precise
- Smaller tol → slower but more precise
- **Practical guidance**: Default is usually fine. For high-precision source separation, try `1e-6`. For exploratory work, `1e-3` is fine.

**`w_init`** — `ndarray` of shape `(n_components, n_components)` or `None`, default `None`
- Initial unmixing matrix. If `None`, drawn from N(0,1)
- **Use case**: If you know the approximate solution (e.g., from a prior run), pass it here for faster convergence or to avoid a bad local optimum
- **Practical guidance**: Leave as `None` unless you have domain-specific initialization

**`whiten_solver`** — `{'eigh', 'svd'}`, default `'svd'`  
**[ADDED in v1.2 — flag this for the reader]**
- Controls the eigendecomposition method for the whitening step
- `'svd'`: Uses SVD of the data matrix. More numerically stable. Faster when n_samples ≤ n_features (wide data).
- `'eigh'`: Uses eigendecomposition of the covariance matrix. More memory efficient. Faster when n_samples ≥ 50 × n_features (tall data, many observations).
- **Practical guidance**: Default `'svd'` is safe for almost all cases. Switch to `'eigh'` only for very tall matrices (e.g., 100,000 EEG time points × 32 channels).

**`random_state`** — `int`, `RandomState`, or `None`, default `None`
- Seed for initializing `w_init` when `w_init=None`
- Set explicitly for reproducible results
- **Practical guidance**: ALWAYS set `random_state=42` (or any fixed int) in production code and comparisons.

### Key Attributes (Post-Fit)

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `components_` | `(n_components, n_features)` | Unmixing vectors — apply to data to get sources |
| `mixing_` | `(n_features, n_components)` | Pseudo-inverse of components_ — maps sources back to data space |
| `mean_` | `(n_features,)` | Column means. Only set if `whiten != False` |
| `whitening_` | `(n_components, n_features)` | Pre-whitening matrix (PCA step). Only set if `whiten != False` |
| `n_iter_` | `int` | Iterations to convergence (parallel) or max across components (deflation) |
| `n_features_in_` | `int` | Number of features seen during fit. Added v0.24 |
| `feature_names_in_` | `ndarray` | Feature names if X had string column names. Added v1.0 |

### Key Methods with Signatures

```python
# Fit and transform in one step (most common)
X_sources = ica.fit_transform(X)   # shape: (n_samples, n_components)

# Fit only
ica.fit(X)

# Transform new data (using learned W)
X_sources_new = ica.transform(X_test)

# Reconstruct original space from sources (for artifact removal)
X_reconstructed = ica.inverse_transform(X_sources)
```

### Complete Working Example (sklearn 1.5.x)

```python
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

# --- Step 1: Simulate cocktail party problem ---
rng = np.random.RandomState(42)
n_samples, n_sources = 2000, 3

# Three independent non-Gaussian sources
t = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * t)                          # sinusoid (sub-Gaussian)
s2 = np.sign(np.sin(3 * t))                 # square wave (sub-Gaussian)
s3 = rng.standard_cauchy(n_samples)         # Cauchy noise (super-Gaussian)
s3 = np.clip(s3, -4, 4)
S = np.vstack([s1, s2, s3]).T               # (2000, 3)
S /= S.std(axis=0)                          # normalize

# Random mixing matrix
A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])
X_mixed = S @ A.T                           # (2000, 3) observed mixtures

# --- Step 2: Fit FastICA ---
ica = FastICA(
    n_components=3,
    algorithm='parallel',
    whiten='unit-variance',
    fun='logcosh',
    max_iter=500,
    tol=1e-4,
    random_state=42
)
X_sources_hat = ica.fit_transform(X_mixed)  # (2000, 3) recovered sources

# --- Step 3: Examine output ---
print("Mixing matrix (estimated):\n", ica.mixing_)
print("n_iter_:", ica.n_iter_)

# --- Step 4: Reconstruct (for artifact removal workflow) ---
# Zero out component 0 (e.g., detected as artifact)
X_sources_hat_cleaned = X_sources_hat.copy()
X_sources_hat_cleaned[:, 0] = 0
X_reconstructed = X_sources_hat_cleaned @ ica.mixing_.T + ica.mean_
# Equivalent to: ica.inverse_transform(X_sources_hat_cleaned)
```

---

## SECTION 4 — SPECIALIZED PYTHON PACKAGES

### Package 1: scikit-learn FastICA
- **PyPI name:** `scikit-learn`
- **Install:** `pip install scikit-learn`
- **Version (mid-2026):** ~1.5.2 (stable) / 1.8+ (latest)
- **Key import:** `from sklearn.decomposition import FastICA`
- **What it does:** Standard FastICA with parallel and deflation algorithms. Full sklearn pipeline compatibility.
- **Strengths:** Pipeline integration, preprocessing utilities, cross-validation, no extra dependencies
- **Weaknesses:** CPU only, single-threaded, no Picard algorithm, no infomax variant
- **License:** BSD-3
- **Maintenance:** Active (scikit-learn core team)

### Package 2: python-picard (Picard & Picard-O)
- **PyPI name:** `python-picard`
- **Install:** `pip install python-picard`
- **GitHub:** https://github.com/pierreablin/picard
- **Version (mid-2026):** ~0.7
- **Key import:** `from picard import picard`
- **What it does:** Preconditioned ICA for Real Data. Uses preconditioned L-BFGS over orthogonal matrices. Two variants:
  - `picard()`: Standard Picard (max-likelihood, equivalent to Infomax)
  - `picard(ortho=True)`: Picard-O (orthogonal constraint, equivalent to FastICA)
- **Strengths:** Faster than FastICA on real data (not simulated). More robust convergence. Extended Infomax capable. sklearn-compatible API available.
- **Weaknesses:** Less well-known. Slightly more complex API than sklearn.
- **Why it beats sklearn on real data**: FastICA's rate of convergence is severely impaired on real data because the underlying Hessian approximation is far from the truth. Picard's Hessian preconditioning remains accurate.
- **License:** BSD-3
- **Maintenance:** Active (Pierre Ablin, MIND-INRIA team)
- **Reference:** Ablin, Cardoso, Gramfort (2018). "Faster ICA under orthogonal constraint." ICASSP 2018.

```python
from picard import picard
import numpy as np

# X shape: (n_sources, n_samples) — NOTE: transposed vs sklearn!
K, W, Y = picard(X.T, n_components=3, ortho=True, random_state=42)
# K: whitening matrix (n_components, n_features)
# W: orthogonal unmixing matrix (n_components, n_components)
# Y: estimated sources (n_components, n_samples)

# sklearn-compatible API (X shape: n_samples, n_features)
from picard import FastICA as PicardFastICA
ica = PicardFastICA(n_components=3, random_state=42)
X_sources = ica.fit_transform(X)
```

### Package 3: MNE-Python (mne.preprocessing.ICA)
- **PyPI name:** `mne`
- **Install:** `pip install mne`
- **Version (mid-2026):** ~1.8 (stable) / 1.12+ (latest)
- **Key import:** `from mne.preprocessing import ICA`
- **What it does:** ICA tailored for EEG/MEG signal processing. Supports FastICA, Picard, and Infomax (extended). Includes specialized methods for automatic artifact detection.
- **Unique features:**
  - `find_bads_eog()`: Detects eye movement artifacts via correlation with EOG channel
  - `find_bads_ecg()`: Detects heartbeat artifacts via CTPS or correlation with ECG channel
  - `find_bads_muscle()`: Detects muscle artifacts via spectral slope
  - `plot_components()`: Interactive topomap visualization
  - `plot_sources()`: Time-series of components
  - Works directly on MNE Raw, Epochs, and Evoked objects
- **Preprocessing requirement**: Data MUST be high-pass filtered ≥1 Hz before ICA (documented in MNE)
- **License:** BSD-3
- **Maintenance:** Active (MNE-Tools core team)

```python
import mne
from mne.preprocessing import ICA

# Assumes raw: mne.io.Raw object, already filtered 1-40 Hz
ica = ICA(
    n_components=20,          # or 0.95 for 95% variance
    method='fastica',         # or 'picard' (recommended), 'infomax'
    max_iter='auto',          # 1000 for fastica, 500 for picard/infomax
    random_state=42
)
ica.fit(raw, picks='eeg')

# Auto-detect eye blink artifacts
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices

# Auto-detect heartbeat artifacts
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation')
ica.exclude += ecg_indices

# Apply: remove artifacts from raw
raw_clean = ica.apply(raw.copy())
```

### Package 4: EEGLAB (MATLAB) — mentioned for completeness
- The dominant ICA tool in academic neuroscience for EEG
- Uses Infomax (Bell & Sejnowski) by default, JADE and other algorithms available
- The `binica` and `runica` functions implement the original InfoMax
- MNE-Python's `method='infomax'` is the Python equivalent

### Package 5: PyTorch-based ICA (for GPU)
No dominant single package exists as of mid-2026, but:
- **`torchica`** (experimental): FastICA in PyTorch for GPU acceleration
- **Alternative**: Implement FastICA manually in PyTorch (fixed-point iteration vectorizes naturally)
- For large-scale problems (fMRI, hyperspectral), GPU-based batch ICA is active research

### Package Recommendation Table

| Use Case | Best Package | Reason |
|----------|-------------|--------|
| General ML / Pipeline | `sklearn.FastICA` | Pipeline compatible, no extra install |
| Best convergence / real data | `python-picard` | Faster, more robust on real-world data |
| EEG/MEG artifact removal | `mne.preprocessing.ICA` | Built-in artifact detection methods |
| Extended Infomax | `mne` with `method='infomax'` | Best Infomax implementation in Python |
| Research / custom G function | `sklearn.FastICA` with callable `fun` | Most flexible API for custom nonlinearities |

---

## SECTION 5 — HYPERPARAMETER TUNING GUIDANCE

### Hyperparameter 1: `n_components`

**What it controls**: How many independent sources you attempt to extract.

**Too few**: Miss important signal structure. Residual variance is still mixed in the discarded components. For EEG, common eye-blink ICs may not be captured.

**Too many**: Attempts to extract more components than the data supports. Leads to convergence failures, numerical instability, or "split" components (one true source appearing as two ICs). If n_components > true rank of data, the algorithm must fabricate structure.

**Heuristics by domain:**
- **EEG (64-channel)**: 40–50 components (rule of thumb: ~0.7 × n_channels). MNE default: n_components satisfying 99.9999% explained variance.
- **fMRI**: 20–70 components. For resting-state network discovery, 10–30 "low-dimensional ICA" captures major networks; 70–100 "high-dimensional ICA" captures sub-network parcellation.
- **Audio BSS**: Equal to known number of sources. If unknown, use model selection.
- **General DR**: Start with PCA — look at explained variance curve (elbow). Use that number.

**How to search with Optuna:**
```python
import optuna

def objective(trial):
    n_components = trial.suggest_int('n_components', 2, X.shape[1])
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    X_sources = ica.fit_transform(X)
    # Use reconstruction error as proxy metric
    X_reconstructed = ica.inverse_transform(X_sources)
    return np.mean((X - X_reconstructed)**2)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

---

### Hyperparameter 2: `fun` (Contrast function)

**Too few options, but high impact.**

| `fun` value | Source type | Stability | Outlier sensitivity |
|-------------|-------------|-----------|---------------------|
| `'logcosh'` | Any (best general) | High | Low |
| `'exp'` | Super-Gaussian, sparse | Medium | Medium |
| `'cube'` | Super-Gaussian ONLY | Low | Very High |

**How to choose:**
1. Default: `'logcosh'`
2. If sources are known sparse (document word counts, spike trains, spike sorting): try `'exp'`
3. Never use `'cube'` unless data is perfectly clean and Gaussian noise has been removed

**Optuna search space:**
```python
fun = trial.suggest_categorical('fun', ['logcosh', 'exp'])
# Skip 'cube' in production search
```

**`fun_args` for logcosh:**
```python
alpha = trial.suggest_float('alpha', 1.0, 2.0, step=0.1)
```

---

### Hyperparameter 3: `algorithm`

**`'parallel'`** (default): All components estimated simultaneously. Converges to different local optima depending on initialization. Run multiple times and pick best (lowest reconstruction error or highest negentropy).

**`'deflation'`**: Estimated component-by-component. More deterministic given same random_state. Useful when you only need top k components.

**Decision rule**: If you need stable top-1 or top-2 components for artifact removal, use `'deflation'`. For general source separation, use `'parallel'`.

**Optuna:**
```python
algorithm = trial.suggest_categorical('algorithm', ['parallel', 'deflation'])
```

---

### Hyperparameter 4: `max_iter` and `tol`

**`max_iter`**:
- Too low: Algorithm terminates before convergence, quality degrades
- Too high: Wastes compute on already-converged model
- Red flag: sklearn `ConvergenceWarning: FastICA did not converge`

**`tol`**:
- Too loose (e.g., 1e-2): Components are rough approximations
- Too tight (e.g., 1e-8): May never converge on noisy data; wastes compute

**Practical pairing:**
```
Exploratory: max_iter=200, tol=1e-4 (defaults)
Production:  max_iter=500, tol=1e-5
High-quality EEG: max_iter=1000, tol=1e-6
```

**Optuna:**
```python
max_iter = trial.suggest_int('max_iter', 200, 2000, log=True)
tol = trial.suggest_float('tol', 1e-6, 1e-2, log=True)
```

---

### Hyperparameter 5: `whiten_solver`

- `'svd'` (default): Use for most cases. n_samples ≤ n_features.
- `'eigh'`: Use when n_samples >> n_features (e.g., n_samples=100,000, n_features=64 EEG channels).

**Not worth tuning via Optuna** — choose based on data dimensions.

---

### Complete Optuna Search Space (ICA)

```python
def suggest_ica_params(trial):
    return {
        'n_components': trial.suggest_int('n_components', 2, min(50, X.shape[1])),
        'algorithm': trial.suggest_categorical('algorithm', ['parallel', 'deflation']),
        'fun': trial.suggest_categorical('fun', ['logcosh', 'exp']),
        'fun_args': {'alpha': trial.suggest_float('alpha', 1.0, 2.0)} 
                    if trial.params.get('fun') == 'logcosh' else None,
        'max_iter': trial.suggest_int('max_iter', 200, 1000, log=True),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
        'random_state': 42,
        'whiten': 'unit-variance'
    }
```

---

## SECTION 6 — EXPERT MENTAL MODEL / PRACTITIONER CHECKLIST

### Before Fitting ICA — Verify These

**1. Check source non-Gaussianity**
```python
from scipy import stats
# Test each channel for normality — ICA only works for non-Gaussian sources
for col in range(X.shape[1]):
    stat, p = stats.normaltest(X[:, col])
    if p > 0.05:
        print(f"Column {col} may be Gaussian — ICA might not separate it")
```
If ALL channels are Gaussian, ICA will fail. This is the single most common cause of nonsense results.

**2. Check for linear mixing assumption**
ICA assumes x = As (linear). If your mixing is nonlinear (e.g., room acoustics with reverb), linear ICA will give poor results. Check: does inverse_transform approximately reconstruct X?

**3. Centering and scaling**
```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
# OR just center (ICA handles variance via whitening)
X_centered = X - X.mean(axis=0)
```
ICA internally centers and whitens, but explicit centering reduces numerical issues.

**4. Check for temporal correlations (time series)**
ICA assumes i.i.d. samples. For time series, adjacent samples ARE correlated. This is fine for audio/EEG (FastICA treats each time point as a sample), but be aware that for financial data, you may want to preprocess (e.g., difference the series to remove autocorrelation).

**5. Handle outliers before fitting**
Kurtosis-based approaches (`fun='cube'`) and even logcosh can be corrupted by extreme outliers. For EEG:
```python
# Clip extreme values before ICA
X_clipped = np.clip(X, np.percentile(X, 1), np.percentile(X, 99))
```

**6. Check n_samples >> n_components rule of thumb**
ICA is a maximum-likelihood method: it needs sufficient data. Rule of thumb: n_samples ≥ 50 × n_components for reliable estimation.

---

### During Fitting — Watch For

**1. ConvergenceWarning**
```
ConvergenceWarning: FastICA did not converge. 
Consider increasing tolerance or the number of iterations.
```
Action: First increase `max_iter` to 500, then 1000. If still failing, change `fun`. If still failing, your data may not have non-Gaussian structure separable by ICA.

**2. Check n_iter_ after fitting**
```python
print(f"Converged in {ica.n_iter_} iterations (max: {ica.max_iter})")
```
If `n_iter_ == max_iter`, convergence was NOT reached.

**3. Multiple random restarts**
ICA is not convex — different initializations give different (but valid) solutions (up to permutation/sign). For important analyses, run 5–10 times and select the run with:
- Lowest reconstruction error
- Highest total negentropy (most non-Gaussian components)
```python
best_ica, best_score = None, np.inf
for seed in range(10):
    ica = FastICA(n_components=k, random_state=seed, max_iter=500)
    X_src = ica.fit_transform(X)
    recon_err = np.mean((X - ica.inverse_transform(X_src))**2)
    if recon_err < best_score:
        best_score = recon_err
        best_ica = ica
```

---

### After Fitting — Diagnostics

**1. Reconstruction error**
```python
X_recon = ica.inverse_transform(ica.transform(X))
recon_err = np.mean((X - X_recon)**2)
print(f"Reconstruction MSE: {recon_err:.6f}")
```
For `whiten='unit-variance'`, near-zero error = components span the data. Large error = something wrong.

**2. Non-Gaussianity of recovered sources**
```python
from scipy import stats
X_src = ica.transform(X)
for i in range(X_src.shape[1]):
    k = stats.kurtosis(X_src[:, i])
    print(f"IC {i}: kurtosis = {k:.3f} ({'super' if k > 0 else 'sub'}-Gaussian)")
```
Components near kurtosis=0 are residual Gaussian noise — may not be meaningful.

**3. Amari distance (if ground truth available)**
```python
def amari_distance(W_est, A_true):
    """Lower is better. 0 = perfect separation (up to permutation/scale)."""
    P = W_est @ A_true
    n = P.shape[0]
    return (np.sum(np.sum(np.abs(P), axis=1) / np.max(np.abs(P), axis=1) - 1) +
            np.sum(np.sum(np.abs(P), axis=0) / np.max(np.abs(P), axis=0) - 1)) / (2 * n * (n - 1))
```

**4. Visual inspection (domain-dependent)**
- EEG: Topomap should show spatially coherent pattern (eye, muscle, heart)
- Audio: Power spectrum of component should be interpretable (voice vs. music vs. noise)
- fMRI: Spatial map should show a known brain network

**5. Signal-to-distortion ratio (for BSS with ground truth)**
```python
def sdr(s_true, s_est):
    """Signal-to-Distortion Ratio in dB. Higher is better."""
    noise = s_true - s_est
    return 10 * np.log10(np.sum(s_true**2) / np.sum(noise**2))
```

---

### Common Pitfalls with Concrete Examples

**Pitfall 1: "Why are my components garbage on financial data?"**  
*Cause*: Stock returns are approximately Gaussian (central limit theorem at daily scale). ICA cannot separate Gaussian sources.  
*Fix*: Try higher-frequency returns (intraday, minutes), which have fat-tail distributions, or use returns minus drift (differencing).

**Pitfall 2: "ICA keeps giving different results on each run"**  
*Cause*: `random_state=None` (default). ICA's optimization landscape has many valid local minima.  
*Fix*: Set `random_state=42`. Also note: different orderings are all mathematically valid.

**Pitfall 3: "I set n_components = n_features but got convergence failure"**  
*Cause*: Data matrix may be rank-deficient (e.g., 64 EEG channels but rank 58 due to reference electrode or interpolation).  
*Fix*: Set n_components to the effective rank:
```python
rank = np.linalg.matrix_rank(X)
ica = FastICA(n_components=min(n_desired, rank - 1))
```

**Pitfall 4: "Components look like noise, not sources"**  
*Cause*: Using `fun='cube'` on data with outliers. Kurtosis is dominated by a few extreme samples.  
*Fix*: Switch to `fun='logcosh'`. Clip outliers before fitting.

**Pitfall 5: "EEG ICA didn't remove eye blinks"**  
*Cause*: Raw data was not high-pass filtered before ICA. Low-frequency drifts prevent convergence to meaningful components.  
*Fix*: `raw.filter(l_freq=1.0, h_freq=None)` before ICA fit. MNE documentation explicitly requires this.

**Pitfall 6: "inverse_transform gives a different shape than expected"**  
*Cause*: You fit with `n_components < n_features`. inverse_transform maps back to n_features dimensions, which is correct — it reconstructs in the original space.

---

## SECTION 7 — REAL-WORLD INNOVATIVE APPLICATIONS

### Application 1: EEG Artifact Removal (Neuroscience / Clinical BCI)
**Domain:** Neuroimaging, brain-computer interfaces  
**Problem:** EEG signals are contaminated by eye blinks (EOG) causing 10–50× larger amplitude than neural signals, heartbeats (ECG), muscle noise (EMG), and power line interference. These mask true neural activity.  
**Why ICA:** The artifacts come from spatially fixed physical sources (eyes, heart, muscles) whose scalp projections are linearly mixed across channels. ICA recovers these as separate components, which can then be removed before back-projecting to channel space.  
**How it works:**
1. Bandpass filter 1–40 Hz (removes low-frequency drift that confounds ICA)
2. Fit ICA to EEG data (each time point is a "sample", each channel a "feature")
3. Identify artifact ICs by correlation with EOG/ECG channels or using automated IC classifiers (MNE-ICALabel, ICLabel in EEGLAB)
4. Zero out artifact ICs, back-project — cleaned EEG
**Packages:** MNE-Python with `method='picard'` or `method='fastica'`  
**Reference:** EEGLAB tutorials at eeglab.org; MNE tutorial at mne.tools/stable  
**Scale:** Used in essentially all EEG research labs worldwide; standard preprocessing in 90%+ of EEG papers

---

### Application 2: fMRI Resting-State Network Discovery
**Domain:** Cognitive neuroscience, clinical neuroimaging (Alzheimer's, depression)  
**Problem:** During "resting state" fMRI (subject lying still with eyes closed), the brain shows spontaneous correlated activity in spatially distributed networks. How do you identify these networks without a task paradigm?  
**Why ICA:** ICA on fMRI data decomposes the spatial × time matrix into spatially independent components (ICs), each with its own time course. The major resting-state networks (Default Mode Network, Sensorimotor, Visual, etc.) emerge as ICs because their activity patterns are statistically independent from each other.  
**High-dimensional vs Low-dimensional ICA:**
- Low-dimensional (10–30 ICs): Captures major networks (Default Mode, Sensorimotor, Visual)
- High-dimensional (70–100 ICs): Sub-network parcellation; can detect within-network damage in Alzheimer's
**Tools:** FSL MELODIC (MATLAB), nilearn ICA (Python), GroupICA + dual regression  
**Key paper:** Beckmann & Smith (2004), FSL ICA for fMRI, NeuroImage  
**Innovation:** ICA-based biomarkers for Alzheimer's — high-dimensional ICA detects within-Default-Mode-Network connectivity damage earlier than structural MRI

---

### Application 3: Cocktail Party Audio Source Separation
**Domain:** Audio processing, voice assistants, hearing aids  
**Problem:** Multiple simultaneous speakers recorded on multiple microphones. Extract each speaker's voice.  
**Why ICA:** The physics of sound propagation creates linear mixtures (at least approximately in near-field conditions). ICA recovers independent source signals from mixed microphone recordings.  
**Limitations:** Standard ICA works for instantaneous (non-convolutive) mixtures. Real rooms have reverberation (convolutive mixing) — requires frequency-domain ICA with permutation alignment across frequency bins.  
**Companies using this:** Hearing aid manufacturers (Phonak, Siemens Healthineers), noise-cancellation systems, smart speakers  
**Alternative packages:** `python_speech_features`, custom frequency-domain ICA  
**Performance:** In anechoic conditions, ICA achieves near-perfect separation. In reverberant rooms, SDR ~10–15 dB (vs. ~30 dB in ideal conditions)

---

### Application 4: Fetal ECG Extraction
**Domain:** Obstetrics, neonatal monitoring  
**Problem:** Fetal heart rate monitoring requires ECG signals from the fetal heart, but electrodes placed on the maternal abdomen record a superposition of maternal ECG (much stronger, ~10× larger amplitude) and fetal ECG (weaker), plus noise. Invasive scalp electrodes have infection risk.  
**Why ICA:** With 4–8 abdominal electrodes, ICA separates maternal and fetal cardiac signals. The maternal heartbeat and fetal heartbeat are statistically independent (they beat at different rates and are uncorrelated), satisfying ICA's assumptions.  
**Implementation:** minimum 2 electrodes needed; 4–8 preferred. Each electrode is a "channel". Time points are samples. FastICA with `fun='exp'` (ECG is sparse/spiky — super-Gaussian).  
**Real-world use:** Research hospitals; FDA-cleared commercial devices from companies like Monica Healthcare use related BSS techniques.  
**Reference:** PMC11117810 — "Development and Implementation of Innovative BSS Techniques for Real-Time Extraction and Analysis of Fetal ECG Signals" (2024)

---

### Application 5: Financial Time Series Decomposition
**Domain:** Quantitative finance, risk management  
**Problem:** Stock returns in a portfolio reflect multiple latent market factors (global market risk, sector rotation, currency effects, earnings surprises). PCA gives orthogonal factors but they don't correspond to interpretable economic forces. Can we recover independent market drivers?  
**Why ICA:** Unlike PCA factors, ICA factors are statistically independent (not just uncorrelated), which better reflects the independence of distinct market regimes. ICA identifies "shocks" — infrequent large moves vs. frequent small fluctuations.  
**Key finding (Back & Weigend 1997):** Applied to 28 Japanese stocks, ICA recovered two types of ICs: (1) large infrequent shocks that explain most price movement, (2) small frequent fluctuations. PCA factors were linear combinations that didn't cleanly separate these.  
**Modern use:** Factor analysis in systematic trading; risk factor decomposition; pairs trading signal generation  
**Caveat:** Daily stock returns ARE approximately Gaussian at the aggregate level — use intraday data or longer windows for fat-tailed behavior that ICA can exploit  
**Reference:** Back & Weigend (1997). "A First Application of ICA to Extracting Structure from Stock Returns." *International Journal of Neural Systems*, 8.

---

### Application 6: Hyperspectral Image Unmixing (Remote Sensing)
**Domain:** Satellite remote sensing, mineral exploration, environmental monitoring  
**Problem:** Each pixel in a hyperspectral image records radiance across 200+ spectral bands. In practice, many pixels contain mixtures of multiple materials (endmembers) — bare soil, vegetation, water, mineral deposits. Spectral unmixing estimates the proportion of each material in each pixel.  
**Why ICA:** If the spectral signatures of pure materials (endmembers) are independent, ICA can recover them as components. ICA is unsupervised — no need to know the endmembers in advance.  
**Limitation:** The linear mixture model (abundances sum to 1, non-negative) is not strictly enforced by ICA. For constrained unmixing, Non-negative Matrix Factorization (NMF) is often preferred or ICA is combined with NMF.  
**Applications:** Mineral mapping (AVIRIS sensor, HYPERION), vegetation stress detection, urban material mapping, planetary surface analysis (Mars CRISM instrument)  
**Reference:** IEEE publications; PubMed 20707164 (ICA for spectral unmixing review)

---

### Application 7: Telecommunications — MIMO Channel Equalization (Innovative)
**Domain:** Wireless communications, 5G, Wi-Fi  
**Problem:** In MIMO (Multiple Input Multiple Output) systems, multiple transmit antennas create interfering signal paths. Channel equalization requires estimating and inverting the channel mixing matrix. Traditional methods need pilot signals (overhead cost). Can we do blind equalization?  
**Why ICA:** The transmitted symbol streams from different antennas are statistically independent (they carry different data). ICA recovers the unmixing matrix from received signals alone — "blind" equalization without pilot overhead.  
**Performance:** Bit error rate performance close to perfect channel state information cases in quasi-static channels. Performance degrades in fast-fading channels (ICA's stationarity assumption violated).  
**Advantage:** Eliminates pilot overhead → increases spectral efficiency by ~10–15%  
**Reference:** Springer Wireless Personal Communications, "ICA Based MIMO Transceiver For Time Varying Wireless Channels" (2017); ResearchGate 224563934

---

### Application 8: Wastewater Treatment Process Monitoring (Innovative Industrial)
**Domain:** Environmental engineering, process control  
**Problem:** Wastewater treatment plants generate high-dimensional, multivariate, non-Gaussian sensor data (influent flow, temperature, dissolved oxygen, ammonia, turbidity, etc.). Fault detection and process optimization require extracting independent underlying process variables from these correlated measurements.  
**Why ICA:** Unlike PCA (which extracts correlated principal components), ICA extracts statistically independent factors representing distinct underlying processes (biological oxidation, settling, chemical addition). Non-Gaussian sensor data suits ICA's assumptions well.  
**What ICA finds:** Independent process components correspond to interpretable operational modes — normal biological activity, shock loads, aeration failures, sludge bulking events.  
**Reference:** ScienceDirect article pii/S2214714425017209 — "Independent component analysis in wastewater treatment plants: Unlocking process understanding and performance optimization" (2025)  
**Scale:** Active area of industrial IoT + process control research; relevant to municipal utilities worldwide.

---

## SECTION 8 — EVALUATION METRICS FOR ICA

### Metric 1: Amari Distance (Gold Standard — when ground truth available)

The Amari distance (also called Amari error index) measures how well the estimated demixing matrix W̃ agrees with the true mixing matrix A:

```python
def amari_distance(W_est, A_true):
    """
    Amari distance — measures quality of ICA separation.
    0 = perfect separation (modulo permutation and scaling).
    Larger = worse.
    
    W_est: estimated unmixing matrix (n_components x n_components)
    A_true: true mixing matrix (n_components x n_components)
    """
    P = W_est @ A_true   # Should be close to a permutation+scale matrix
    n = P.shape[0]
    
    row_term = np.sum(
        np.sum(np.abs(P), axis=1) / np.max(np.abs(P), axis=1) - 1
    )
    col_term = np.sum(
        np.sum(np.abs(P), axis=0) / np.max(np.abs(P), axis=0) - 1
    )
    return (row_term + col_term) / (2 * n * (n - 1))
```

**Interpretation:**
- Amari distance = 0: Perfect separation
- Amari distance < 0.1: Good separation
- Amari distance > 0.5: Poor separation
- Available in scipy as `scipy.stats.spearmanr` doesn't apply here — implement manually

---

### Metric 2: Signal-to-Noise Ratio (SNR) / Signal-to-Interference Ratio (SIR)

Used in audio BSS and ECG separation:

```python
def sir_metric(s_true, s_est):
    """
    Signal-to-Interference Ratio for one component.
    Assumes correct permutation alignment.
    """
    # Optimal scaling
    alpha = np.dot(s_true, s_est) / np.dot(s_est, s_est)
    interference = s_true - alpha * s_est
    return 10 * np.log10(np.sum(s_true**2) / np.sum(interference**2))
```

---

### Metric 3: Reconstruction Error

For exploratory ICA (no ground truth):

```python
X_sources = ica.fit_transform(X)
X_reconstructed = ica.inverse_transform(X_sources)

# Mean Squared Error
mse = np.mean((X - X_reconstructed)**2)

# Relative reconstruction error
rel_err = np.linalg.norm(X - X_reconstructed, 'fro') / np.linalg.norm(X, 'fro')
print(f"Relative reconstruction error: {rel_err:.4f}")
```

**Note**: For `whiten='unit-variance'`, reconstruction is not perfect when n_components < n_features — the whitening step reduces dimensionality. This is expected and the reconstruction error measures information loss, not ICA quality.

---

### Metric 4: Negentropy of Recovered Sources

Higher negentropy = more non-Gaussian = better separation:

```python
from scipy.stats import differential_entropy

def negentropy(x):
    """Approximated negentropy using logcosh."""
    x_std = x / x.std()
    gaussian_reference = np.random.normal(0, 1, len(x))
    return (np.mean(np.log(np.cosh(x_std))) - 
            np.mean(np.log(np.cosh(gaussian_reference))))**2

X_src = ica.fit_transform(X)
for i in range(X_src.shape[1]):
    nJ = negentropy(X_src[:, i])
    print(f"IC {i}: negentropy = {nJ:.4f}")
```

A component with near-zero negentropy is Gaussian residual noise — probably not a meaningful source.

---

### Metric 5: Kurtosis

```python
from scipy.stats import kurtosis

X_src = ica.fit_transform(X)
for i in range(X_src.shape[1]):
    k = kurtosis(X_src[:, i])
    print(f"IC {i}: excess kurtosis = {k:.3f}")
```

- |kurtosis| > 1: Likely a meaningful non-Gaussian source
- |kurtosis| < 0.5: Likely noise (Gaussian)

---

### Metric 6: Explained Variance (Per Component)

Unlike PCA, ICA components are not ordered by variance. But you can compute variance each component contributes:

```python
X_src = ica.fit_transform(X)
# Variance of each source in original space
var_explained = []
for i in range(X_src.shape[1]):
    mask = np.zeros(X_src.shape)
    mask[:, i] = X_src[:, i]
    x_recon_i = ica.inverse_transform(mask)
    var_explained.append(np.var(x_recon_i) / np.var(X) * 100)

# Sort components by variance explained for interpretability
order = np.argsort(var_explained)[::-1]
print("Components sorted by variance explained (%):", 
      [f"{v:.1f}" for v in sorted(var_explained, reverse=True)])
```

---

### Metric 7: Temporal Autocorrelation (for EEG/time series)

Meaningful neural ICs often have smooth, structured temporal dynamics. Artifact ICs (eye blinks, heartbeats) have characteristic spectral profiles.

```python
from scipy.signal import welch

fs = 1000  # sampling rate
X_src = ica.fit_transform(X)
for i in range(X_src.shape[1]):
    freqs, psd = welch(X_src[:, i], fs=fs, nperseg=256)
    # High power in 0.1-1 Hz: likely eye artifact
    # High power in >20 Hz: likely muscle artifact
    low_power = np.mean(psd[(freqs < 1)])
    high_power = np.mean(psd[(freqs > 20)])
    print(f"IC {i}: low/high freq power ratio = {low_power/high_power:.2f}")
```

---

## SECTION 9 — ADDITIONAL TECHNICAL DETAILS FOR WRITER

### The Whitening / Sphering Step (Preprocessing Inside ICA)

FastICA's internal preprocessing:
1. **Center**: X_c = X − mean(X, axis=0)
2. **Whiten**: Find matrix K such that E{(Kx)(Kx)ᵀ} = I
   - Compute covariance: C = (1/n)X_cᵀX_c
   - Eigendecompose: C = EDE where D = diag(d₁, ..., dₙ)
   - Whitening matrix: K = D^{-1/2}Eᵀ
   - Whitened data: Z = KX_c (now Z has identity covariance)
3. **FastICA on Z**: Find orthogonal W such that y = WZ = (WK)X_c

After whitening, the problem reduces to finding an orthogonal rotation W (n² → n(n-1)/2 free parameters), which is much easier than finding an arbitrary invertible matrix (n² parameters). This is the key computational trick that makes FastICA fast.

---

### The Fixed-Point Iteration Derivation

The FastICA update rule can be derived as a Newton step in the space of unit vectors:

**Objective**: Maximize J(w) ≈ [E{G(wᵀz)} − E{G(ν)}]²  
where ν ~ N(0,1) and z is whitened data.

**Gradient**: ∂J/∂w = E{z · g(wᵀz)}  
**Hessian**: E{g'(wᵀz)} · I − E{z zᵀ g'(wᵀz)}

Since z is whitened, E{zzᵀ} = I, so:  
Newton step: w_new = w − [gradient] / [approx Hessian]

This simplifies (after algebra) to:
```
w_new ← E{z · g(wᵀz)} − E{g'(wᵀz)} · w
w_new ← w_new / ||w_new||   # orthonormalize
```

This is the core FastICA update. It converges cubically (third-order convergence) to a local optimum.

---

### Connection Between ICA Formulations

All these formulations are equivalent (proven in Hyvärinen 1999 survey):

| Formulation | Objective |
|-------------|-----------|
| InfoMax (Bell & Sejnowski) | Maximize H(f(Wx)) |
| Maximum likelihood | Maximize log p(x\|W) = log\|det W\| + Σ log pᵢ(wᵢᵀx) |
| Mutual information min. | Minimize I(y₁, ..., yₙ) |
| Negentropy max. | Maximize Σᵢ J(yᵢ) subject to whitened y |
| Kurtosis max. (FastICA with cube) | Maximize \|E{(wᵀz)⁴}\| |

They all find the same solution; they differ only in numerical properties, convergence speed, and sensitivity to outliers.

---

### ICA Identifiability Theorem (Precise Statement)

ICA is identifiable (i.e., A can be recovered up to permutation and scaling) if and only if:
1. The mixing matrix A is square and invertible
2. At most one of the sᵢ has a Gaussian distribution
3. The sources sᵢ are mutually independent

This theorem (proved by Comon 1994) explains:
- Why two or more Gaussian sources cannot be separated (any rotation of a Gaussian vector is still Gaussian — the rotation is unidentifiable)
- Why overcomplete ICA (more sources than mixtures) is fundamentally harder
- Why ICA works even without knowing the number of sources (up to n = number of sensors)

---

### ICA for Dimensionality Reduction vs Source Separation

**Source separation** (square case): n mixtures, n sources, full A is square. Goal: recover all sources.

**Dimensionality reduction** (truncated case): n mixtures, k << n sources. Use FastICA with n_components=k. This is a form of blind source separation that assumes only k sources are present (or that the remaining sources have negligible signal).

**ICA for feature extraction**: Fit ICA, then use the components_ matrix as a projection. The features (columns of components_) capture non-Gaussian structure. Unlike PCA, ICA features are NOT ordered by importance — you must assess them via kurtosis or reconstruction contribution.

---

## SECTION 10 — HISTORICAL LINEAGE

```
1985: Hérault & Jutten — First neural network approach to BSS (early ICA)
1989: Cardoso — FOBI (Fourth-Order Blind Identification) — first cumulant method
1991: Comon — Formal definition of ICA, identifiability theorem
1993: Cardoso & Souloumiac — JADE algorithm
1995: Bell & Sejnowski — InfoMax ICA (most cited ICA paper)
1996: Amari et al. — Natural gradient version of InfoMax
1997: Hyvärinen & Oja — FastICA (fixed-point algorithm)
1997: Back & Weigend — First ICA application to financial data
1999: Hyvärinen — Survey paper, unifies all approaches
2000: Hyvärinen & Oja — "Algorithms and Applications" review (most readable)
2001: Hyvärinen, Karhunen, Oja — Book "Independent Component Analysis" (Wiley)
2004: Beckmann & Smith — ICA for fMRI (MELODIC, FSL)
2018: Ablin, Cardoso, Gramfort — Picard algorithm (faster than FastICA on real data)
2026: ICA remains foundational in EEG/MEG preprocessing, fMRI RSN analysis, and BSS
```

---

## SECTION 11 — QUICK REFERENCE CODE SNIPPETS

### Minimal Working Example
```python
from sklearn.decomposition import FastICA
import numpy as np

rng = np.random.RandomState(42)
X = rng.randn(1000, 5)  # 1000 samples, 5 mixed signals

ica = FastICA(n_components=5, random_state=42, max_iter=500)
X_ica = ica.fit_transform(X)   # shape: (1000, 5)
X_back = ica.inverse_transform(X_ica)   # shape: (1000, 5)
```

### EEG Artifact Removal (MNE)
```python
import mne
from mne.preprocessing import ICA

# raw: mne.io.Raw, already loaded and filtered 1-40 Hz
ica = ICA(n_components=20, method='picard', random_state=42)
ica.fit(raw, picks='eeg')

# Detect artifacts automatically
eog_idx, _ = ica.find_bads_eog(raw, threshold=3.0)
ecg_idx, _ = ica.find_bads_ecg(raw, method='correlation')
ica.exclude = eog_idx + ecg_idx

raw_clean = ica.apply(raw.copy())
```

### Picard (faster alternative for real data)
```python
from picard import picard

# X: (n_sources, n_samples) — transposed vs sklearn
K, W, Y = picard(X.T, n_components=10, ortho=True, random_state=42)
# Y: recovered sources, shape (n_components, n_samples)
```

### Custom G function
```python
from sklearn.decomposition import FastICA
import numpy as np

def logcosh_tanh(x):
    """Manual logcosh with custom alpha."""
    alpha = 1.5
    gx = np.tanh(alpha * x)           # g(x) = tanh(alpha*x)
    g_x = alpha * (1 - gx**2)         # g'(x)
    return gx, g_x.mean(axis=-1)      # sklearn expects mean of g'

ica = FastICA(fun=logcosh_tanh, n_components=5, random_state=42)
```

---

## KEY SOURCES FOR WRITING AGENT

1. Bell & Sejnowski 1995 — DOI: 10.1162/neco.1995.7.6.1129
2. Hyvärinen & Oja 1997 — DOI: 10.1162/neco.1997.9.7.1483 (https://direct.mit.edu/neco/article/9/7/1483)
3. Hyvärinen 1999 Survey — Neural Computing Surveys 2:94–128
4. Hyvärinen & Oja 2000 — DOI: 10.1016/S0893-6080(00)00026-5
5. sklearn 1.5 docs — https://scikit-learn.org/1.5/modules/generated/sklearn.decomposition.FastICA.html
6. MNE-Python ICA docs — https://mne.tools/stable/generated/mne.preprocessing.ICA.html
7. Picard package — https://github.com/pierreablin/picard
8. Back & Weigend 1997 — https://www.worldscientific.com/doi/abs/10.1142/S0129065797000458
9. Wastewater ICA 2025 — https://www.sciencedirect.com/science/article/abs/pii/S2214714425017209
10. Fetal ECG BSS 2024 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11117810/
