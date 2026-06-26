# Independent Component Analysis: The Complete Masterclass

> **Why this algorithm matters:** At a cocktail party with five simultaneous conversations, your
> auditory system effortlessly picks out one speaker from the cacophony — even when every
> microphone in the room records a different blend of all five voices. This is the problem
> Independent Component Analysis was built to solve. Beyond cocktail parties, ICA is the engine
> behind artifact removal in every serious EEG/MEG laboratory worldwide, the method that discovers
> resting-state brain networks in fMRI, the signal processing backbone of modern hearing aids, and
> the tool that separates fetal heartbeats from maternal ECG without a single surgical probe.
> What makes ICA special isn't just what it does — it's what it reveals: the hidden statistical
> structure that makes reality tick, the independent generating processes behind every dataset you
> will ever encounter. After this chapter, you will understand not just how to call
> `FastICA().fit_transform()`, but why it works, when it breaks, and how to think like a
> practitioner who trusts their ICA output.

---

## §0 — Python Ecosystem & Package Guide

Before touching a single line of math, map the terrain. ICA has a richer ecosystem than most
dimensionality reduction algorithms because it spans so many domains — neuroscience, audio,
finance, remote sensing — each of which has developed specialized tooling. Here is every
Python package you need to know, verified for mid-2026.

### Complete Package Table

| Package | Import | Version (mid-2026) | What it provides | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.decomposition import FastICA` | ~1.5.x | Standard FastICA, pipeline-compatible, deflation + parallel algorithms | BSD-3 |
| `python-picard` | `from picard import picard` | ~0.7 | Preconditioned ICA (Picard + Picard-O), faster on real data, sklearn API | BSD-3 |
| `mne` | `from mne.preprocessing import ICA` | ~1.8 | EEG/MEG-specialized ICA, built-in artifact detection, FastICA + Picard + Infomax | BSD-3 |
| `optuna` | `import optuna` | ~3.x | Hyperparameter search over `n_components`, `fun`, `algorithm`, `tol` | MIT |
| `matplotlib` | `import matplotlib.pyplot as plt` | ~3.9.x | All diagnostic plots: source time series, mixing matrices, kurtosis bars | PSF |
| `scipy` | `from scipy import stats` | ~1.13.x | Kurtosis, normality tests, Welch PSD for evaluating IC quality | BSD-3 |
| `numpy` | `import numpy as np` | ~2.0.x | Fixed-point iteration, Amari distance, reconstruction error | BSD-3 |
| `seaborn` | `import seaborn as sns` | ~0.13.x | Heatmaps (mixing matrix visualization), pairplots of ICA components | BSD-3 |

> 🔧 **In practice:** For pure machine learning pipelines, `sklearn.FastICA` is your workhorse —
> it requires zero extra installations and slots into any `Pipeline`. For EEG/MEG work, install
> `mne` first; it wraps FastICA, Picard, and Infomax with domain-specific artifact detection.
> For any serious real-world dataset (not simulated), install `python-picard`: it converges
> faster and more reliably than sklearn's FastICA on data with heterogeneous source distributions.

---

### Package Deep-Dives

#### scikit-learn `FastICA` — The Universal Starting Point

`sklearn.decomposition.FastICA` implements the Hyvärinen & Oja (1997) fixed-point algorithm
with both parallel and deflation extraction strategies. It is the right default because it
integrates cleanly with `Pipeline`, `GridSearchCV`, and every sklearn utility.

```python
from sklearn.decomposition import FastICA
import numpy as np

# Same API as every sklearn transformer
ica = FastICA(
    n_components=3,
    algorithm='parallel',       # or 'deflation'
    whiten='unit-variance',     # default since v1.3 — always use this string form
    fun='logcosh',              # negentropy contrast function
    max_iter=500,
    tol=1e-4,
    random_state=42
)

X_sources = ica.fit_transform(X_mixed)        # shape: (n_samples, n_components)
X_reconstructed = ica.inverse_transform(X_sources)  # back to original feature space
```

**Key attributes after fitting:**
- `ica.components_`: shape `(n_components, n_features)` — the unmixing vectors
- `ica.mixing_`: shape `(n_features, n_components)` — the estimated mixing matrix (pseudo-inverse of `components_`)
- `ica.mean_`: shape `(n_features,)` — column means used for centering
- `ica.whitening_`: shape `(n_components, n_features)` — the PCA whitening matrix
- `ica.n_iter_`: number of iterations to convergence

> ⚠️ **Pitfall:** In sklearn v1.2 and earlier, `whiten=True` was the default. In v1.3, `True`
> was deprecated in favour of the string `'unit-variance'`. In v1.5+, passing `whiten=True` raises
> a `FutureWarning`. Always use `whiten='unit-variance'` or `whiten='arbitrary-variance'` in
> any code you write today.

---

#### `python-picard` — The Speed Champion for Real Data

The Picard algorithm (Ablin, Cardoso & Gramfort 2018) reframes ICA as a preconditioned
L-BFGS optimization over the space of orthogonal matrices. On simulated data with perfect
Gaussian mixtures, FastICA and Picard are comparable. On real-world data with heterogeneous
source distributions, Picard typically converges **5–10× faster** with fewer iterations to
a lower final objective.

Why? FastICA's Hessian approximation is accurate only near the solution. Picard uses a
preconditioning matrix that remains a good Hessian approximation throughout optimization,
giving reliable quasi-Newton steps even far from the minimum.

```python
from picard import picard
import numpy as np

# NOTE: picard() expects shape (n_sources, n_samples) — TRANSPOSED vs sklearn
K, W, Y = picard(
    X.T,                    # (n_features, n_samples)
    n_components=3,
    ortho=True,             # Picard-O: orthogonal constraint, equivalent to FastICA
    random_state=42,
    max_iter=500,
    tol=1e-7
)
# K: whitening matrix (n_components, n_features)
# W: orthogonal unmixing matrix (n_components, n_components)
# Y: estimated sources (n_components, n_samples)

# Sklearn-compatible API (same shape convention as sklearn)
from picard import FastICA as PicardFastICA
ica_picard = PicardFastICA(n_components=3, random_state=42)
X_sources = ica_picard.fit_transform(X)   # (n_samples, n_components)
```

> 📜 **Origin/Citation:** Ablin, P., Cardoso, J.-F., & Gramfort, A. (2018). Faster ICA under
> orthogonal constraint. *ICASSP 2018*. https://github.com/pierreablin/picard

---

#### `mne.preprocessing.ICA` — Domain Expert for Brain Signals

MNE-Python's ICA wraps FastICA, Picard, and Infomax into a neuroscience-native interface that
operates directly on `Raw`, `Epochs`, and `Evoked` objects. Its killer features are automated
artifact detection methods that leverage MNE's physiological signal knowledge:

```python
import mne
from mne.preprocessing import ICA

# Prerequisites: raw must be filtered >= 1 Hz (documented MNE requirement)
raw.filter(l_freq=1.0, h_freq=40.0)

ica = ICA(
    n_components=20,            # or 0.95 for 95% explained variance threshold
    method='picard',            # recommended in modern MNE; or 'fastica', 'infomax'
    max_iter='auto',            # 500 for picard, 1000 for fastica
    random_state=42
)
ica.fit(raw, picks='eeg')

# Automated artifact detection
eog_indices, eog_scores = ica.find_bads_eog(raw)        # eye blinks
ecg_indices, ecg_scores = ica.find_bads_ecg(
    raw, method='correlation'                             # heartbeat
)
ica.exclude = eog_indices + ecg_indices

# Apply: remove artifact components, project back to sensor space
raw_clean = ica.apply(raw.copy())
```

> 🏆 **Best practice:** In MNE, always high-pass filter at ≥1 Hz before ICA. Low-frequency
> drifts create a strong non-Gaussian component that "absorbs" the first IC and prevents
> the algorithm from converging to meaningful neural components. This is not optional — MNE's
> own documentation flags it as a hard requirement.

---

### GPU and Streaming Options

**GPU-accelerated ICA:** As of mid-2026, there is no dominant production-ready GPU ICA package.
The fixed-point iteration vectorizes naturally in PyTorch or JAX — for large-scale fMRI or
hyperspectral data, consider implementing the whitening step with `torch.linalg.svd` and the
fixed-point loop with batched matrix operations. `torchica` exists on PyPI but is experimental.

**Streaming / Online ICA:** For non-stationary environments (e.g., real-time EEG in a BCI),
online ICA algorithms update the unmixing matrix as new data arrives. The `orica` (Online
Recursive ICA) and variants of natural gradient ICA support streaming. These are primarily
research implementations — check recent GitHub repositories rather than PyPI packages.

---

### Top 5 Picks — Recommendation Table

| Use case | Best package | Reasoning | Modern/Legacy |
|---|---|---|---|
| General ML pipeline, feature extraction | `sklearn.FastICA` | Zero extra deps, Pipeline-compatible, battle-tested | Modern |
| Real-world data, best convergence | `python-picard` | 5–10× faster than FastICA on heterogeneous sources | Modern |
| EEG/MEG artifact removal | `mne.preprocessing.ICA` | Built-in EOG/ECG detection, works on MNE data objects | Modern |
| Extended Infomax (sub + super-Gaussian) | `mne` with `method='infomax'` | Only reliable Python Infomax implementation | Modern |
| Research with custom G functions | `sklearn.FastICA` + callable `fun` | Most flexible API for nonlinearity customization | Modern |

---

### Direct Comparison: Same Data, Four Implementations

```python
import numpy as np
import time
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

# ── Simulate cocktail party data ──────────────────────────────────────────
rng = np.random.RandomState(42)
n_samples, n_sources = 5000, 5

t = np.linspace(0, 10, n_samples)
s1 = np.sin(2 * np.pi * 1.5 * t)                     # 1.5 Hz sinusoid
s2 = np.sign(np.sin(2 * np.pi * 2.0 * t))            # 2 Hz square wave
s3 = rng.standard_cauchy(n_samples).clip(-4, 4)       # Cauchy noise
s4 = (t % 1.0) - 0.5                                  # sawtooth
s5 = rng.laplace(0, 0.5, n_samples)                   # Laplace noise
S = np.column_stack([s1, s2, s3, s4, s5])
S /= S.std(axis=0)

A = rng.randn(n_sources, n_sources)                   # random mixing
X_mixed = S @ A.T
X_mixed = StandardScaler().fit_transform(X_mixed)

# ── Option 1: sklearn FastICA (parallel) ──────────────────────────────────
t0 = time.perf_counter()
ica_sk = FastICA(n_components=5, algorithm='parallel',
                 fun='logcosh', max_iter=500, random_state=42)
X_sk = ica_sk.fit_transform(X_mixed)
print(f"sklearn FastICA:  {time.perf_counter()-t0:.3f}s, "
      f"n_iter={ica_sk.n_iter_}")

# ── Option 2: sklearn FastICA (deflation) ─────────────────────────────────
t0 = time.perf_counter()
ica_def = FastICA(n_components=5, algorithm='deflation',
                  fun='logcosh', max_iter=500, random_state=42)
X_def = ica_def.fit_transform(X_mixed)
print(f"sklearn Deflation: {time.perf_counter()-t0:.3f}s, "
      f"n_iter={ica_def.n_iter_}")

# ── Option 3: python-picard ───────────────────────────────────────────────
try:
    from picard import FastICA as PicardFastICA
    t0 = time.perf_counter()
    ica_pc = PicardFastICA(n_components=5, random_state=42)
    X_pc = ica_pc.fit_transform(X_mixed)
    print(f"Picard:          {time.perf_counter()-t0:.3f}s")
except ImportError:
    print("picard not installed: pip install python-picard")

# ── Reconstruction error (all should be near 0 if n_comp == n_feat) ───────
for name, ica_obj, X_src in [
    ('sklearn-parallel', ica_sk, X_sk),
    ('sklearn-deflation', ica_def, X_def),
]:
    X_recon = ica_obj.inverse_transform(X_src)
    err = np.mean((X_mixed - X_recon)**2)
    print(f"{name:20s}  recon MSE = {err:.2e}")
```

> 💡 **Intuition:** All three implementations find the same mathematical solution — they differ
> only in the path they take through the optimization landscape. Think of it like three climbers
> finding the same mountain summit via different trails. The Picard trail uses better trail maps
> (preconditioning) and tends to be shorter on rugged real-world terrain, even though on smooth
> synthetic mountains they're equivalent.

---

## §1 — The Origin: The Paper That Started It All

### The Lineage Before 1995

Independent Component Analysis did not spring fully-formed from a single paper. It emerged
from a decade of converging ideas in neural computation, signal processing, and information
theory. The intellectual genealogy matters because it explains why ICA has so many equivalent
formulations:

**1985 — Hérault & Jutten:** The first neural network approach to blind source separation
(BSS). They proposed a feedback network that could, under certain conditions, decorrelate its
inputs — a crude precursor to ICA. The algorithm lacked theoretical grounding but demonstrated
that separation was possible from observations alone.

**1989 — Cardoso (FOBI):** Jean-François Cardoso introduced FOBI (Fourth-Order Blind
Identification), the first algorithm to explicitly use fourth-order statistics (kurtosis-like
measures) for source separation. FOBI diagonalizes one fourth-order cumulant matrix, which
works when all sources have distinct kurtosis values — a limitation that JADE would later
remove.

**1991 — Comon:** Pierre Comon published the formal definition of statistical independence for
ICA and proved the identifiability theorem: ICA can recover the mixing matrix up to permutation
and scaling **if and only if** at most one source is Gaussian. This mathematical foundation
transformed BSS from engineering heuristic into a rigorous statistical problem.

**1993 — Cardoso & Souloumiac (JADE):** The Joint Approximate Diagonalization of
Eigenmatrices algorithm simultaneously diagonalized a set of fourth-order cumulant matrices,
overcoming FOBI's limitation. JADE became the standard algorithm for the next two years — until
Bell and Sejnowski changed everything.

---

### Bell & Sejnowski 1995 — The Information-Maximization Bombshell

> 📜 **Origin/Citation:** Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization
> approach to blind separation and blind deconvolution. *Neural Computation*, 7(6), 1129–1159.
> DOI: [10.1162/neco.1995.7.6.1129](https://doi.org/10.1162/neco.1995.7.6.1129)

Anthony Bell and Terry Sejnowski arrived at ICA from a completely different direction: they were
studying self-organizing neural networks and the principle of maximum information transfer. The
question they asked was deceptively simple: **given a network of neurons with nonlinear activation
functions, what should the synaptic weights be to maximize the information flowing through?**

Their answer — the **InfoMax principle** — turned out to be mathematically equivalent to
independent component analysis, although they did not initially recognise it as such.

**The problem they were solving:**

Prior to 1995, signal processing had PCA for decorrelation, but decorrelation only removes
*second-order* statistical dependencies (covariance). Two signals can have zero covariance yet
remain statistically dependent in higher-order senses. For example, if $X \sim \mathcal{N}(0,1)$
and $Y = X^2$, then $\text{Cov}(X, Y) = 0$ but $X$ and $Y$ are clearly dependent. PCA cannot
disentangle this relationship. Bell & Sejnowski needed a method that removed *all* statistical
dependencies, not just linear ones.

**The core insight — Maximizing information flow:**

Consider a single-layer neural network $\mathbf{y} = f(\mathbf{W}\mathbf{x})$ where $\mathbf{x}$
is the input, $\mathbf{W}$ is the weight matrix, and $f$ is a nonlinear activation function applied
elementwise. The output entropy is:

$$H(\mathbf{y}) = H(f(\mathbf{W}\mathbf{x})) = \log|\det \mathbf{W}| + \sum_i H(f_i(w_i^\top \mathbf{x}))$$

Maximizing $H(\mathbf{y})$ with respect to $\mathbf{W}$ maximizes information transfer through the
network. The gradient ascent update rule is:

$$\Delta \mathbf{W} \propto (\mathbf{W}^\top)^{-1} + \mathbf{g}(\mathbf{y})\mathbf{x}^\top$$

where $\mathbf{g}(\mathbf{y}) = f'(\mathbf{y})/f(\mathbf{y})$ is the score function of the
source density. When $f = \text{sigmoid}$ (CDF of a logistic distribution), this becomes:

$$\Delta \mathbf{W} \propto (\mathbf{W}^\top)^{-1} + (1 - 2\mathbf{y})\mathbf{x}^\top$$

> 💡 **Intuition:** Think of the network as a communication channel. You want to maximize the
> channel's information throughput. If the output units are correlated, you're wasting bandwidth
> — correlated outputs carry redundant information. Maximizing entropy forces outputs apart into
> maximally informative, independent signals. This is Shannon's channel capacity argument applied
> to neurons.

**The crucial equivalence (shown later by MacKay, Amari, and others):**

The InfoMax objective is equivalent to maximum likelihood estimation when $f$ is chosen as the
CDF of the true source distribution. It is also equivalent to minimizing mutual information
between outputs. This trinity — InfoMax, maximum likelihood, mutual information minimization —
converges to the same answer and explains why ICA papers from different communities look
superficially different but compute the same thing.

---

### The Cocktail Party Problem — The Canonical Motivating Example

Imagine $N$ speakers talking simultaneously in a room. $N$ microphones are placed at different
positions, each recording a different linear mixture of all voices:

$$x_i(t) = a_{i1} s_1(t) + a_{i2} s_2(t) + \cdots + a_{iN} s_N(t), \quad i = 1, \ldots, N$$

In matrix form: $\mathbf{x}(t) = \mathbf{A} \mathbf{s}(t)$, where:

- $\mathbf{x}(t) \in \mathbb{R}^N$ = microphone recordings at time $t$ (observed)
- $\mathbf{s}(t) \in \mathbb{R}^N$ = individual speaker voices (unknown sources)
- $\mathbf{A} \in \mathbb{R}^{N \times N}$ = mixing matrix encoding room geometry (unknown)

The goal is to find the **unmixing matrix** $\mathbf{W} = \mathbf{A}^{-1}$ such that
$\mathbf{y}(t) = \mathbf{W}\mathbf{x}(t) \approx \mathbf{s}(t)$, using *only* the observed
signals — without knowing $\mathbf{A}$, without training signals, and without any prior
knowledge of the speakers' voices.

**Why PCA fails here:** PCA finds directions of maximum variance and makes components
orthogonal. Applied to microphone data, it produces components that are linear combinations of
speakers — still mixed, just decorrelated. Decorrelation removes second-order dependencies but
leaves higher-order dependencies intact. ICA goes further: it finds directions where the
projected distribution is *maximally non-Gaussian*, which (by a reversal of the Central Limit
Theorem) corresponds to recovering independent sources.

---

### The Three Fundamental Ambiguities of ICA

These are mathematically proven limitations — not implementation bugs:

**1. Scale ambiguity:** If $\mathbf{W}\mathbf{x} \approx \mathbf{s}$, then for any diagonal
$\mathbf{D}$, $\mathbf{D}^{-1}\mathbf{W}\mathbf{x} \approx \mathbf{D}^{-1}\mathbf{s}$ is also
valid. You cannot determine source variances independently.

**2. Sign ambiguity:** The sign of each component is undetermined ($-1$ is scale with $D=-I$).
ICA might return the negative of the true source — this is mathematically valid but requires
care in interpretation.

**3. Permutation ambiguity:** Component ordering is arbitrary. IC index 0 in one run may
correspond to IC index 2 in another run with a different random seed. Always label ICA
components by their *content*, not their position.

> ⚠️ **Pitfall:** These ambiguities are permanent. Always label ICA components by what they
> represent — examine topomaps (EEG), listen to extracted signals (audio), or check correlation
> with known factors (finance) — rather than relying on positional indices.

---

## §2 — The Algorithm, Deeply Explained

### The ICA Generative Model

Let's be precise about what we're assuming. The ICA model is:

$$\mathbf{x} = \mathbf{A}\mathbf{s}$$

where $\mathbf{x} \in \mathbb{R}^p$ is the observed mixture vector, $\mathbf{s} \in \mathbb{R}^d$
is the vector of latent independent sources, and $\mathbf{A} \in \mathbb{R}^{p \times d}$ is the
unknown mixing matrix. In the standard (square) case, $p = d$ and $\mathbf{A}$ is invertible.

The model has these explicit assumptions:

1. **Independence:** The sources $s_1, s_2, \ldots, s_d$ are mutually statistically independent:
   $p(\mathbf{s}) = \prod_{i=1}^d p_i(s_i)$
2. **Non-Gaussianity:** At most one source may be Gaussian (identifiability theorem, Comon 1994)
3. **Linear mixing:** The observed signals are linear combinations of sources
4. **Square and full-rank mixing:** $\mathbf{A}$ is invertible (standard case)
5. **Zero mean:** Sources are zero-mean (achieved by centering data)

The goal is to find $\mathbf{W} = \mathbf{A}^{-1}$ such that $\hat{\mathbf{s}} = \mathbf{W}\mathbf{x}$
recovers the original sources up to the three fundamental ambiguities.

---

### Why Non-Gaussianity Is the Key — The CLT Reversal

The Central Limit Theorem (CLT) says: **sums of independent random variables become more Gaussian
as the number of terms grows**. ICA exploits this in reverse.

Suppose we project the observed mixture $\mathbf{x} = \mathbf{A}\mathbf{s}$ onto a unit vector $\mathbf{w}$:

$$y = \mathbf{w}^\top \mathbf{x} = \mathbf{w}^\top \mathbf{A} \mathbf{s} = \mathbf{b}^\top \mathbf{s}$$

where $\mathbf{b} = \mathbf{A}^\top \mathbf{w}$. The projection $y$ is a weighted sum of the
independent sources $s_i$. By the CLT, as $\mathbf{b}$ spreads weight across multiple sources,
$y$ becomes more Gaussian. The only directions $\mathbf{w}$ where $y$ is *least* Gaussian are
those where $\mathbf{b}$ concentrates all weight on a single source (i.e., $\mathbf{b} = \mathbf{e}_i$,
a coordinate vector in the source space).

**This gives us the fundamental ICA algorithm: find the projection directions $\mathbf{w}$ that
maximize non-Gaussianity of $\mathbf{w}^\top \mathbf{x}$.**

> 💡 **Intuition:** Think of mixing independent voices as blending colors. If you mix red, green,
> and blue paint, you get brown — a "more average" color. To un-mix, you look for the "most
> extreme" (most saturated) color directions. Gaussianity is the probability-space equivalent of
> "average" and non-Gaussianity is the equivalent of "saturated" or "extreme." ICA finds the
> pure colors by searching for maximum extremity.

---

### Measuring Non-Gaussianity

**Option 1: Kurtosis**

The excess kurtosis is the standardized fourth cumulant:

$$\text{kurt}(y) = E\{y^4\} - 3(E\{y^2\})^2$$

For a Gaussian variable, $\text{kurt}(y) = 0$. For sub-Gaussian variables (uniform, bounded),
$\text{kurt}(y) < 0$. For super-Gaussian variables (Laplace, Cauchy, sparse signals),
$\text{kurt}(y) > 0$.

FastICA with `fun='cube'` maximizes $|E\{(\mathbf{w}^\top \mathbf{x})^4\}|$, which is
proportional to $|\text{kurt}(\mathbf{w}^\top \mathbf{x})|$.

**Problem with kurtosis:** It is the fourth moment, making it extremely sensitive to outliers.
A single sample with value $x = 10$ contributes $10^4 = 10000$ to the fourth moment but only
$10^2 = 100$ to the variance. One corrupted data point can dominate the kurtosis estimate.

**Option 2: Negentropy (robust, preferred)**

The differential entropy of a random variable $Y$ with density $p(y)$ is:

$$H(Y) = -\int p(y) \log p(y) \, dy$$

Negentropy is defined as:

$$J(Y) = H(Y_{\text{Gaussian}}) - H(Y)$$

where $Y_{\text{Gaussian}}$ is a Gaussian with the same variance as $Y$. The Gaussian distribution
has maximum entropy among all distributions with the same variance, so $J(Y) \geq 0$ always,
with equality if and only if $Y$ is Gaussian. Negentropy is a principled, robust measure of
non-Gaussianity.

Computing negentropy exactly is intractable (requires knowing $p(y)$). Hyvärinen (1997, 1999)
proposed the approximation:

$$J(Y) \approx [E\{G(Y)\} - E\{G(\nu)\}]^2$$

where $\nu \sim \mathcal{N}(0,1)$ and $G$ is a non-quadratic function. The choice of $G$
corresponds to the `fun` parameter in sklearn:

| `fun` value | $G(u)$ | $g(u) = G'(u)$ | $g'(u) = G''(u)$ |
|-------------|---------|-----------------|-------------------|
| `'logcosh'` | $\frac{1}{\alpha}\log\cosh(\alpha u)$ | $\tanh(\alpha u)$ | $\alpha(1 - \tanh^2(\alpha u))$ |
| `'exp'` | $-\exp(-u^2/2)$ | $u \exp(-u^2/2)$ | $(1 - u^2)\exp(-u^2/2)$ |
| `'cube'` | $u^4/4$ | $u^3$ | $3u^2$ |

The `'logcosh'` default is a smooth, bounded approximation that handles both sub- and
super-Gaussian sources robustly. The `'exp'` function emphasizes super-Gaussian (sparse)
structure. The `'cube'` function is pure kurtosis — fast but numerically dangerous.

---

### The Whitening Preprocessing Step — Why It Matters

Before running the fixed-point iterations, FastICA applies a crucial preprocessing step:
**whitening** (also called sphering). This transforms the data so that its covariance matrix
is the identity:

$$E\{\mathbf{z}\mathbf{z}^\top\} = \mathbf{I}$$

**Step 1: Center.** Subtract column means: $\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{x}}\mathbf{1}^\top$

**Step 2: Compute covariance.** $\mathbf{C} = \frac{1}{n}\mathbf{X}_c^\top\mathbf{X}_c$

**Step 3: Eigendecompose.** $\mathbf{C} = \mathbf{E}\mathbf{D}\mathbf{E}^\top$ where $\mathbf{D} = \text{diag}(d_1, \ldots, d_p)$

**Step 4: Whiten.** $\mathbf{K} = \mathbf{D}^{-1/2}\mathbf{E}^\top$ is the whitening matrix.
The whitened data is $\mathbf{Z} = \mathbf{K}\mathbf{X}_c$.

> 💡 **Intuition:** Whitening is essentially PCA. It rotates the data to the principal component
> axes *and* scales each axis to unit variance. After whitening, the data cloud is a sphere (or
> hypersphere in $p$ dimensions) with no preferred orientation. All remaining structure that ICA
> can exploit lives in the *shape* of the data distribution within this sphere, not in the
> covariance.

**Why whitening makes ICA tractable:** After whitening, the unmixing matrix $\mathbf{W}$ must
satisfy $\mathbf{W}\mathbf{W}^\top = \mathbf{I}$ (it must be orthogonal). The problem shrinks
from finding an arbitrary invertible matrix ($p^2$ free parameters) to finding an orthogonal
rotation ($p(p-1)/2$ free parameters). For $p = 100$, that's 10,000 parameters vs. 4,950 —
roughly a 2× reduction, but more importantly, the constraint set (orthogonal matrices) has a
well-understood geometry (the Stiefel manifold) that enables efficient optimization.

**The combined transform:** FastICA ultimately finds the full demixing matrix as:

$$\hat{\mathbf{W}}_{\text{total}} = \hat{\mathbf{W}}_{\text{orthogonal}} \cdot \mathbf{K}$$

where $\hat{\mathbf{W}}_{\text{orthogonal}}$ is the orthogonal rotation found by the fixed-point
iterations and $\mathbf{K}$ is the whitening matrix. This is stored in `ica.components_` (shape:
`n_components × n_features`).

---

### The FastICA Fixed-Point Algorithm — Full Derivation

> 📜 **Origin/Citation:** Hyvärinen, A., & Oja, E. (1997). A fast fixed-point algorithm for
> independent component analysis. *Neural Computation*, 9(7), 1483–1492.
> DOI: [10.1162/neco.1997.9.7.1483](https://doi.org/10.1162/neco.1997.9.7.1483)

Given whitened data $\mathbf{Z}$ (rows are samples), we want to find a unit vector
$\mathbf{w}$ that maximizes $J(\mathbf{w}^\top \mathbf{z}) \approx [E\{G(\mathbf{w}^\top \mathbf{z})\} - E\{G(\nu)\}]^2$.

**Gradient of the objective:**

$$\nabla_\mathbf{w} J(\mathbf{w}) \propto E\{\mathbf{z} \cdot g(\mathbf{w}^\top \mathbf{z})\}$$

where $g = G'$ (first derivative of the contrast function).

**Hessian approximation:**

$$\mathbf{H} \approx E\{g'(\mathbf{w}^\top \mathbf{z})\} \cdot \mathbf{I} - E\{\mathbf{z}\mathbf{z}^\top g'(\mathbf{w}^\top \mathbf{z})\}$$

Since $\mathbf{z}$ is whitened, $E\{\mathbf{z}\mathbf{z}^\top\} = \mathbf{I}$, and under the
independence assumption at the optimum:

$$\mathbf{H} \approx \left(E\{g'(\mathbf{w}^\top \mathbf{z})\} - 1\right) \cdot \mathbf{I}$$

**Newton step** (gradient divided by scalar Hessian):

$$\mathbf{w}_{\text{new}} = \mathbf{w} - \frac{E\{\mathbf{z} \cdot g(\mathbf{w}^\top \mathbf{z})\}}{E\{g'(\mathbf{w}^\top \mathbf{z})\} - 1}$$

Rearranging (multiply numerator and denominator by $-1$):

$$\boxed{\mathbf{w}_{\text{new}} = E\{\mathbf{z} \cdot g(\mathbf{w}^\top \mathbf{z})\} - E\{g'(\mathbf{w}^\top \mathbf{z})\} \cdot \mathbf{w}}$$

followed by normalization: $\mathbf{w}_{\text{new}} \leftarrow \mathbf{w}_{\text{new}} / \|\mathbf{w}_{\text{new}}\|$

This is the **FastICA update rule** — a Newton step disguised as a simple fixed-point iteration.
The name "fixed-point" refers to the fact that at convergence, $\mathbf{w}_{\text{new}} = \mathbf{w}$
(up to sign), satisfying $E\{\mathbf{z} \cdot g(\mathbf{w}^\top \mathbf{z})\} - E\{g'(\mathbf{w}^\top \mathbf{z})\} \cdot \mathbf{w} = c\mathbf{w}$
for some scalar $c$.

**Convergence rate:** Third-order (cubic) — the convergence basin is large and the algorithm
reaches high precision in very few iterations compared to gradient methods (which converge
linearly). This is what makes FastICA "fast."

---

### Parallel vs Deflation Extraction

**Parallel (simultaneous) extraction:**

All $d$ component vectors $\mathbf{W} = [\mathbf{w}_1, \ldots, \mathbf{w}_d]^\top$ are updated
simultaneously. After each update, apply symmetric orthogonalization:

$$\mathbf{W} \leftarrow \mathbf{W}(\mathbf{W}^\top \mathbf{W})^{-1/2}$$

to prevent all vectors from converging to the same component. This is the default
`algorithm='parallel'` in sklearn.

**Deflation (sequential) extraction:**

Extract components one at a time. After finding $\mathbf{w}_k$, project it out of the data:

$$\mathbf{Z}_{\text{deflated}} = \mathbf{Z} - \mathbf{w}_k \mathbf{w}_k^\top \mathbf{Z}$$

Then find $\mathbf{w}_{k+1}$ from the deflated data. This is Gram-Schmidt orthogonalization.

| Property | Parallel | Deflation |
|---|---|---|
| Speed | Faster (vectorized) | Slower (sequential) |
| Stability | May converge to different local optima | More deterministic for top-k |
| Use case | All $d$ components needed | Only top-k components needed |
| sklearn default | Yes | No |

---

### Computational Complexity

Let $n$ = number of samples, $p$ = number of features, $d$ = number of components.

**Whitening step (PCA):** $O(np^2 + p^3)$ — dominated by SVD or eigendecomposition.

**FastICA fixed-point iterations:** Each iteration costs $O(ndp)$ for the matrix multiplication
$\mathbf{W}\mathbf{Z}$ plus $O(nd)$ for the expectation terms. For $K$ iterations:
$O(K \cdot ndp)$.

**Total:** $O(np^2 + K \cdot ndp)$

In practice:
- For large $n$, small $p$ (tall data, many EEG time points): whitening dominates
- For large $p$, small $n$ (wide data, hyperspectral images): use `whiten_solver='svd'`
- FastICA converges typically in $K = 10$–$100$ iterations for well-conditioned problems

**Transform (apply to new data):** $O(nd)$ — just a matrix multiplication
$\mathbf{Z}_{\text{new}} = (\hat{\mathbf{W}}_{\text{total}}) \mathbf{X}_{\text{new}} - \mathbf{b}$

---

### The ICA vs PCA Comparison — Definitive Table

| Aspect | PCA | ICA |
|--------|-----|-----|
| **Objective** | Maximize variance | Maximize statistical independence |
| **Statistics used** | 2nd-order (covariance) | Higher-order (3rd, 4th cumulants) |
| **Components** | Orthogonal directions of max variance | Independent non-Gaussian directions |
| **Ordering** | By variance explained (decreasing) | Arbitrary — no natural ordering |
| **Uniqueness** | Unique (eigenvectors) | Unique up to sign + permutation |
| **Gaussian data** | Works fine | Fails — cannot separate Gaussian sources |
| **Preprocessing** | Center only | Must whiten (PCA step inside ICA) |
| **Reconstruction** | Perfect with $d = p$ | Perfect with $d = p$ (full square case) |
| **Key assumption** | None (decorrelation always valid) | Non-Gaussian, independent sources |
| **Primary use** | Compression, noise reduction, visualization | Source separation, artifact removal, feature extraction |

**Critical relationship:** ICA *uses* PCA internally. Whitening is PCA. ICA then rotates in
the whitened (PCA) space to achieve independence. PCA takes you to an uncorrelated basis;
ICA rotates further to an independent basis. You cannot have ICA without PCA, but you can have
PCA without ICA.

> 🔬 **Deep dive:** Why does whitening reduce the problem to finding an orthogonal matrix?
> After whitening, $\mathbf{Z} = \mathbf{K}\mathbf{x} = \mathbf{K}\mathbf{A}\mathbf{s}$.
> Let $\tilde{\mathbf{A}} = \mathbf{K}\mathbf{A}$. Then $E\{\mathbf{Z}\mathbf{Z}^\top\} = \tilde{\mathbf{A}} E\{\mathbf{s}\mathbf{s}^\top\} \tilde{\mathbf{A}}^\top = \tilde{\mathbf{A}}\tilde{\mathbf{A}}^\top = \mathbf{I}$
> (since sources have unit variance and whitened data has identity covariance). A matrix
> satisfying $\mathbf{M}\mathbf{M}^\top = \mathbf{I}$ is orthogonal by definition. So finding
> the unmixing matrix in whitened space is equivalent to finding an orthogonal rotation — a
> much more constrained and well-structured problem.

---

### Where the Assumptions Break

Understanding failure modes requires understanding which assumption was violated:

| Violated assumption | Symptom | What to do |
|---|---|---|
| Sources are Gaussian | Components have near-zero kurtosis, random-looking | Use PCA instead; ICA is non-identifiable |
| Sources are not independent | Recovered components still correlated | Check data generation process; try NMF for non-negative sources |
| Mixing is nonlinear | Reconstruction error is large; components don't separate | Use nonlinear BSS or deep learning approach |
| Sample size too small ($n < 50d$) | Unstable estimates, high variance across seeds | Collect more data; reduce $d$ |
| Data has outliers | Kurtosis-based methods dominated by outliers | Use `fun='logcosh'`, clip extreme values |
| Rank-deficient data | ConvergenceWarning or near-zero components | Set `n_components < rank(X)` |

---

## §3 — The Full Evolution: Every Major Variant

### 3.1 — InfoMax ICA (Bell & Sejnowski 1995) — The Founding Algorithm

> 📜 **Citation:** Bell, A. J., & Sejnowski, T. J. (1995). Neural Computation, 7(6), 1129–1159.

**Problem solved:** First practical algorithm for blind source separation using information theory.

**Key algorithmic idea:** Maximize entropy flowing through a single-layer neural network with
sigmoid nonlinearities. The weight update rule (natural gradient form, Amari 1996):

$$\Delta \mathbf{W} \propto \mathbf{W}^{-\top} + \mathbf{g}(\mathbf{W}\mathbf{x})\mathbf{x}^\top$$

where $\mathbf{g}(u) = 1 - 2\sigma(u)$ for sigmoid $\sigma$.

**Key weakness:** The sigmoid assumes all sources are super-Gaussian (sparse). Sub-Gaussian
sources (e.g., uniformly distributed signals) produce incorrect unmixing because the score
function has the wrong sign.

**When to prefer:** When you know all sources are super-Gaussian (speech, images). Historical
interest only for general use — FastICA has superseded it.

**Python:** `mne.preprocessing.ICA(method='infomax')` — the most reliable Python Infomax
implementation, using the Bell & Sejnowski original with optional sigmoid or tanh nonlinearity.

---

### 3.2 — Extended InfoMax (Lee, Girolami & Sejnowski 1999) — Handling Both Source Types

> 📜 **Citation:** Lee, T.-W., Girolami, M., & Sejnowski, T. J. (1999). Independent component
> analysis using an extended infomax algorithm for mixed subgaussian and supergaussian sources.
> *Neural Computation*, 11(2), 417–441.

**Problem solved:** Standard InfoMax assumes all sources are super-Gaussian (sigmoid nonlinearity).
Real-world datasets often contain a mix of sub-Gaussian (uniform, bounded) and super-Gaussian
(sparse, heavy-tailed) sources. InfoMax misidentifies sub-Gaussian sources as super-Gaussian.

**Key algorithmic change:** Adaptively switch the nonlinearity per component:

$$g_i(y) = \begin{cases} 1 - 2\sigma(y) & \text{if component } i \text{ is super-Gaussian} \\ 2\sigma(y) - 1 & \text{if component } i \text{ is sub-Gaussian} \end{cases}$$

The switching criterion uses the kurtosis sign: positive kurtosis → super-Gaussian → sigmoid;
negative kurtosis → sub-Gaussian → tanh.

**When to prefer:** When you have mixed source types and need the Infomax framework (e.g.,
EEG where both artifact components (super-Gaussian eye blinks) and oscillatory neural
components (sub-Gaussian sinusoids) coexist).

**Python:** `mne.preprocessing.ICA(method='infomax', fit_params={'extended': True})`

```python
import mne
from mne.preprocessing import ICA

ica = ICA(
    n_components=20,
    method='infomax',
    fit_params=dict(extended=True),   # Extended InfoMax
    random_state=42
)
ica.fit(raw, picks='eeg')
```

---

### 3.3 — FastICA (Hyvärinen & Oja 1997) — The Speed Revolution

> 📜 **Citation:** Hyvärinen, A., & Oja, E. (1997). Neural Computation, 9(7), 1483–1492.
> DOI: 10.1162/neco.1997.9.7.1483

**Problem solved:** InfoMax's gradient-based learning is slow (linear convergence) and requires
tuning a learning rate. For datasets with thousands of samples and dozens of sources, this was
impractical.

**Key algorithmic change:** Replace gradient ascent with a Newton fixed-point iteration
(derived in §2). The key insight: in whitened space, the Hessian simplifies to a scalar times
the identity, making Newton steps trivial to compute.

**Result:** Cubic convergence rate. No learning rate to tune. No free parameters beyond the
choice of contrast function $G$. Typically converges in 10–100 iterations.

**Variants:**
- **Parallel FastICA** (sklearn default): All components simultaneously, symmetric orthogonalization
- **Deflation FastICA**: One component at a time, Gram-Schmidt

**When to prefer:** General-purpose ICA. The right default for 90% of use cases.

**Python:** `sklearn.decomposition.FastICA` — the canonical Python implementation.

```python
from sklearn.decomposition import FastICA

ica = FastICA(
    n_components=5,
    algorithm='parallel',   # or 'deflation'
    fun='logcosh',          # contrast function G
    fun_args={'alpha': 1.0},
    max_iter=500,
    tol=1e-4,
    whiten='unit-variance',
    random_state=42
)
X_sources = ica.fit_transform(X)
```

---

### 3.4 — JADE (Cardoso & Souloumiac 1993/1996) — The Cumulant Approach

> 📜 **Citation:** Cardoso, J.-F., & Souloumiac, A. (1993). Blind beamforming for non-Gaussian
> signals. *IEE Proceedings F*, 140(6), 362–370.

**Problem solved:** Both InfoMax and FastICA require iterative optimization and can get stuck.
JADE (Joint Approximate Diagonalization of Eigenmatrices) is a batch algebraic method that
always converges.

**Key algorithmic idea:** Fourth-order cumulants of any multivariate distribution can be
organized into a set of matrices. For independent sources, these cumulant matrices are
simultaneously diagonal (in the source basis). JADE finds the orthogonal transformation that
*jointly approximately diagonalizes* all cumulant matrices:

**Algorithm:**
1. **Whiten** the data to get $\mathbf{Z}$
2. **Compute** all $p^2$ cumulant matrices $\mathbf{Q}_{ij}$:
   $$Q_{ij,kl} = \text{cum}(z_i, z_j, z_k, z_l)$$
3. **Stack** the most informative subset into a matrix $\mathbf{M}$
4. **Jointly diagonalize** via Jacobi sweeps (iterative Givens rotations):
   $$\mathbf{V} = \arg\min_{\mathbf{V} \in O(p)} \sum_{ij} \|\text{off}(\mathbf{V}\mathbf{Q}_{ij}\mathbf{V}^\top)\|_F^2$$
5. The unmixing matrix is $\hat{\mathbf{W}} = \mathbf{V}^\top \mathbf{K}$ where $\mathbf{K}$ is
   the whitening matrix

**Advantages over FastICA:**
- No convergence failure — Jacobi sweeps always converge
- No contrast function to choose — uses all fourth-order information
- Handles both sub- and super-Gaussian sources without switching
- More reliable in low-sample regimes ($n < 1000$)

**Disadvantages:**
- $O(p^4)$ complexity for cumulant computation — impractical for high-dimensional data
- No sklearn implementation — must use the `jade` package or MNE's internal implementation

**When to prefer:** Low-dimensional BSS with heterogeneous source distributions; when
FastICA convergence is unreliable; for rigorous theoretical guarantees.

**Python:**
```python
# pip install jade  (or use MNE's internal implementation)
# JADE is available via MNE's 'infomax' extended method which uses related cumulant approach
# For pure JADE:
from jade import jadeR   # pip install jade; verify in current docs
W = jadeR(X.T)           # X.T shape: (n_features, n_samples)
X_sources = (W @ X.T).T  # shape: (n_samples, n_components)
```

---

### 3.5 — Natural Gradient ICA (Amari et al. 1996) — Riemannian Optimization

> 📜 **Citation:** Amari, S., Cichocki, A., & Yang, H. H. (1996). A new learning algorithm for
> blind signal separation. *NIPS 1996*.

**Problem solved:** Standard gradient on weight matrices is poorly conditioned — the loss
landscape has vastly different curvatures in different directions, causing oscillations and
slow convergence (especially near saddle points).

**Key insight:** The parameter space of ICA is not Euclidean — it is the curved manifold of
invertible matrices. The Riemannian gradient (natural gradient) accounts for this curvature
and is invariant to the scale and orientation of the current estimate.

**Natural gradient update:**

$$\Delta \mathbf{W} \propto \left[\mathbf{I} + \mathbf{g}(\mathbf{y})\mathbf{y}^\top\right] \mathbf{W}$$

The key difference from standard gradient: multiply by $\mathbf{W}^\top \mathbf{W}$ (the
Fisher information metric for ICA). This makes the update equivariant — multiplying the data
by a constant matrix doesn't change the algorithm's trajectory.

**Equivariance property:** Natural gradient ICA is **equivariant** — if you premultiply the
data by any invertible matrix, the unmixing matrix adjusts accordingly. Standard gradient ICA
is sensitive to data conditioning (highly correlated inputs → slow convergence). Natural
gradient ICA is not.

**When to prefer:** Online ICA (streaming data), low-SNR environments, data with strong
correlations between channels. Used in the `binica` EEGLAB implementation.

**Python:** Available in MNE's `method='infomax'` implementation, which uses natural gradient.

---

### 3.6 — Picard & Picard-O (Ablin, Cardoso & Gramfort 2018) — The Modern Standard

> 📜 **Citation:** Ablin, P., Cardoso, J.-F., & Gramfort, A. (2018). Faster ICA under
> orthogonal constraint. *ICASSP 2018*. arXiv: 1611.05267.
> GitHub: https://github.com/pierreablin/picard

**Problem solved:** FastICA's quadratic convergence is conditional on the Hessian approximation
being accurate. On *real data* (not well-conditioned simulated mixtures), FastICA's Hessian is
often a poor approximation, leading to many iterations and potential divergence. FastICA's
claimed "cubic convergence" applies at the limit, not in practice with real heterogeneous data.

**Key algorithmic idea:** Use a *preconditioned* L-BFGS optimizer on the Stiefel manifold
(orthogonal matrices). The preconditioner is a better-quality Hessian approximation, derived
from the second-order ICA geometry. For the non-orthogonal case (Picard vs Picard-O), optimize
over all invertible matrices, not just orthogonal ones.

**Two variants:**
- **Picard-O** (`ortho=True`): Orthogonal constraint — equivalent to FastICA, but faster
- **Picard** (`ortho=False`): No orthogonal constraint — equivalent to Infomax/maximum likelihood,
  can handle the general (non-orthogonal) case more flexibly

**Performance advantage on real data:**

| Algorithm | Typical iterations to convergence (real EEG, 64ch, 100k samples) |
|---|---|
| FastICA (sklearn) | 200–500 |
| Picard-O | 30–80 |
| Picard (full) | 20–60 |

**When to prefer:** Any time you're working with real data and computational cost matters.
MNE-Python's documentation now recommends `method='picard'` over `method='fastica'` as the
default for EEG preprocessing.

```python
from picard import picard

# Standard picard call (expects n_features × n_samples)
K, W, Y = picard(
    X.T,                    # shape: (n_features, n_samples)
    n_components=20,
    ortho=True,             # Picard-O (FastICA equivalent, but faster)
    random_state=42,
    max_iter=500,
    tol=1e-7
)
# Y shape: (n_components, n_samples)
sources = Y.T               # back to (n_samples, n_components)

# sklearn-compatible wrapper
from picard import FastICA as PicardFastICA  # verify in current docs
ica = PicardFastICA(n_components=20, random_state=42)
sources = ica.fit_transform(X)  # X shape: (n_samples, n_features)
```

---

### 3.7 — Second-Order Blind Identification (SOBI) — Exploiting Temporal Structure

> 📜 **Citation:** Belouchrani, A., Abed-Meraim, K., Cardoso, J.-F., & Moulines, E. (1997).
> A blind source separation technique using second-order statistics. *IEEE Transactions on Signal
> Processing*, 45(2), 434–444.

**Problem solved:** Standard ICA ignores temporal structure — it treats each time point as an
independent sample. In many real signals (EEG, speech, finance), adjacent samples are
correlated. SOBI exploits this temporal autocorrelation structure as additional information
for source separation.

**Key idea:** Instead of using higher-order statistics (kurtosis, negentropy), SOBI jointly
diagonalizes time-lagged covariance matrices $\mathbf{R}(\tau) = E\{\mathbf{x}(t)\mathbf{x}(t-\tau)^\top\}$
at multiple lags $\tau = 1, 2, \ldots, T$. For independent sources, these matrices are
simultaneously diagonal in the source basis — just like JADE uses spatial cumulants, SOBI
uses temporal covariances.

**Advantage:** Works even when all sources are *Gaussian* (standard ICA fails here), provided
sources have different autocorrelation structures (different spectral profiles).

**When to prefer:**
- Time series where sources have distinct temporal dynamics but may be Gaussian
- Low-SNR scenarios where higher-order statistics are unreliable
- EEG/MEG resting state analysis where sources have characteristic frequency bands

**Python:** Available in MNE-Python as `method='sobi'` in some versions (verify in current
docs), or via the `mne-realtime` package and dedicated `sobi` Python packages.

```python
# SOBI via joint diagonalization (illustrative sketch)
import numpy as np
from scipy.linalg import eigh

def sobi(X, n_lags=100, n_components=None):
    """Simplified SOBI implementation."""
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    # Step 1: Whiten
    C = np.cov(X.T)
    vals, vecs = eigh(C)
    K = vecs[:, -n_components:] @ np.diag(vals[-n_components:]**-0.5)
    Z = X @ K                    # whitened data

    # Step 2: Compute time-lagged covariance matrices
    R_list = []
    for tau in range(1, n_lags + 1):
        R_tau = (Z[tau:].T @ Z[:-tau]) / (n_samples - tau)
        R_list.append((R_tau + R_tau.T) / 2)  # symmetrize

    # Step 3: Joint diagonalization (Jacobi sweeps — simplified)
    # In practice, use a proper joint diag library
    return Z, R_list  # placeholder
```

---

### 3.8 — Overcomplete / Underdetermined ICA — More Sources Than Mixtures

All formulations above assume $d \leq p$ (at most as many sources as sensors). **Overcomplete ICA**
($d > p$) addresses the case where there are *more* sources than observations — mathematically
harder because the system is underdetermined ($\mathbf{A}$ is wide, not square).

**Approaches:**
1. **Sparse ICA / Basis Pursuit:** Assume sources are sparse. Use $\ell_1$ minimization to
   find the sparsest representation: $\hat{\mathbf{s}} = \arg\min \|\mathbf{s}\|_1$ subject to
   $\mathbf{A}\mathbf{s} \approx \mathbf{x}$
2. **Dictionary Learning:** Learn both $\mathbf{A}$ and $\mathbf{s}$ jointly with sparsity
   constraints (similar to sparse coding)

**Python:** `sklearn.decomposition.SparsePCA`, `sklearn.decomposition.DictionaryLearning`,
or specialized sparse BSS implementations.

**When relevant:** Audio source separation (more speakers than microphones),
neural spike sorting (many neurons, few electrodes), text topic modeling (many topics, few
co-occurrence statistics).

---

## §4 — Hyperparameters: The Complete Guide

FastICA has a small but impactful set of hyperparameters. Every single one matters in at
least one practical scenario. Here is the complete guide.

---

### `n_components` — How Many Sources to Extract

**What it controls:** The number of independent components ICA will attempt to recover.
Internally, this also determines the rank of the whitening step (truncated SVD/PCA).

**Default:** `None` — which uses ALL features. This is almost never what you want. Always
set this explicitly.

**Mechanism:** When `n_components=k`, FastICA:
1. Keeps the top $k$ principal components during whitening (discards the rest)
2. Performs the fixed-point iteration in the $k$-dimensional whitened subspace
3. Returns $k$ independent components

**Too few components:** You miss meaningful signal structure. In EEG, you might not capture
the eye-blink or heartbeat artifact component, leaving those artifacts in the data.

**Too many components:** Attempts to find more structure than the data contains. Leads to:
- Convergence failures (ConvergenceWarning)
- "Split" components: one true source appearing as two ICs
- Numerical instability from near-zero eigenvalues in whitening

**Domain heuristics:**

| Domain | Guideline | Rationale |
|---|---|---|
| EEG (64 channels) | $0.7 \times n_{\text{channels}}$ (≈45) | Reference electrode and interpolation reduce effective rank |
| EEG (MNE default) | `n_components` achieving 99.9999% variance | Data-driven rank selection |
| fMRI (resting state) | 10–30 (low-dim), 70–100 (high-dim) | Major networks vs. sub-network parcellation |
| Audio BSS | = number of speakers | Known problem size |
| General DR | PCA elbow method | Match number of meaningful PCs |
| Small $n$ | $< \sqrt{n / 50}$ | Ensure $n > 50 \times d$ for stable estimation |

**Tuning with PCA elbow:**
```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Run PCA to find natural dimensionality
pca = PCA().fit(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)

# Step 2: Find elbow or 95%/99% threshold
n_95 = np.argmax(cumvar >= 0.95) + 1
n_99 = np.argmax(cumvar >= 0.99) + 1
print(f"95% variance: {n_95} components | 99% variance: {n_99} components")

# Step 3: Check rank
rank = np.linalg.matrix_rank(X)
n_components = min(n_99, rank - 1)  # safety margin
print(f"Safe n_components for ICA: {n_components}")
```

**Tuning with Optuna:**
```python
import optuna
from sklearn.decomposition import FastICA
import numpy as np

def objective(trial):
    n_components = trial.suggest_int('n_components', 2, min(50, X.shape[1] - 1))
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500,
                  whiten='unit-variance')
    X_sources = ica.fit_transform(X)
    X_recon = ica.inverse_transform(X_sources)
    # Relative reconstruction error as proxy metric
    return np.linalg.norm(X - X_recon, 'fro') / np.linalg.norm(X, 'fro')

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
print(f"Best n_components: {study.best_params['n_components']}")
```

> ⚠️ **Pitfall:** If your data is rank-deficient (e.g., EEG with interpolated channels or
> re-referenced data), setting `n_components >= rank(X)` will cause a `LinAlgError` or
> `ConvergenceWarning`. Always check `np.linalg.matrix_rank(X)` first.

---

### `fun` — The Contrast Function (Non-Gaussianity Measure)

**What it controls:** The function $G$ used to approximate negentropy. This is the most
conceptually important hyperparameter — it encodes your prior belief about source distributions.

**Default:** `'logcosh'` — the right default for almost all cases.

**The three built-in options:**

**`'logcosh'`:** $G(u) = \frac{1}{\alpha}\log\cosh(\alpha u)$, $g(u) = \tanh(\alpha u)$

- Smooth, bounded contrast function
- Robust to outliers (tanh is bounded, so extreme values don't dominate)
- Works well for both sub- and super-Gaussian sources
- The $\alpha$ parameter (`fun_args={'alpha': 1.0}`) controls shape:
  - $\alpha = 1$: balanced, slightly sub-Gaussian emphasis
  - $\alpha = 2$: more super-Gaussian emphasis (sparser sources)

**`'exp'`:** $G(u) = -\exp(-u^2/2)$, $g(u) = u\exp(-u^2/2)$

- Better for recovering sparse (super-Gaussian) sources
- Slightly more sensitive to outliers than logcosh but better than cube
- Useful for: spike trains, speech (sparse in time-frequency domain), sparse coding applications

**`'cube'`:** $G(u) = u^4/4$, $g(u) = u^3$

- Exactly equivalent to maximizing $|$kurtosis$|$
- Fastest per iteration (simple polynomial)
- Extremely sensitive to outliers — **avoid on real data**
- Use only when: data is perfectly clean, sources are known to be super-Gaussian, and
  outliers have been removed

**Custom callable:**

```python
import numpy as np
from sklearn.decomposition import FastICA

def my_logcosh_alpha15(x):
    """Custom logcosh with alpha=1.5."""
    alpha = 1.5
    gx = np.tanh(alpha * x)                    # g(x) = tanh(alpha * x)
    g_x = alpha * (1 - gx ** 2)               # g'(x)
    return gx, g_x.mean(axis=-1)              # sklearn expects mean of g'(x)

ica = FastICA(fun=my_logcosh_alpha15, n_components=5, random_state=42)
X_sources = ica.fit_transform(X)
```

**Decision flowchart for `fun`:**

```
What do you know about your sources?
    |
    ├── Unknown distribution / mixed types → 'logcosh' (default)
    |
    ├── Sparse signals (speech, spikes, images) → 'exp'
    |
    ├── Known super-Gaussian, clean data → 'cube' (use with caution)
    |
    └── Custom prior knowledge → callable G function
```

| `fun` | Source type | Outlier robustness | Speed | When to use |
|---|---|---|---|---|
| `'logcosh'` | Any | High | Medium | Default; first choice |
| `'exp'` | Super-Gaussian / sparse | Medium | Medium | Known sparse sources |
| `'cube'` | Super-Gaussian ONLY | Very Low | Fastest | Clean data only — avoid in production |

---

### `algorithm` — Parallel vs Deflation Extraction

**Default:** `'parallel'`

**`'parallel'`:** All $d$ component vectors updated simultaneously with symmetric
orthogonalization after each step. Faster (vectorized), but the symmetric orthogonalization
can interfere when the algorithm is near convergence, potentially increasing iteration count.

**`'deflation'`:** Extracts components one at a time using Gram-Schmidt orthogonalization.
Slower but more deterministic — once component $k$ is found, it stays fixed while
components $k+1, k+2, \ldots$ are extracted.

**Practical decision rule:**
- Need all $d$ components for downstream analysis (source separation, feature matrix): `'parallel'`
- Need only top-1 or top-2 components for artifact removal or inspection: `'deflation'`
- Parallel gives poor results (convergence issues): try `'deflation'`

> 🔧 **In practice:** Run with `'parallel'` first. If you see `ConvergenceWarning`, try
> increasing `max_iter`. If that fails, switch to `'deflation'` — it's more stable at the
> cost of speed.

---

### `whiten` — Preprocessing Mode

**Default:** `'unit-variance'` (since sklearn v1.3)

**Options:**

- `'unit-variance'`: Whitens AND rescales so recovered sources have unit variance. Best choice
  for most applications — ensures scale-comparable components.
- `'arbitrary-variance'`: Whitens but does not normalize component scales. Use when you want
  the absolute scale information preserved (rare).
- `False`: Assumes data is already whitened. Use when calling FastICA multiple times on the
  same pre-whitened dataset to save computation.

> ⚠️ **Pitfall (API change):** In sklearn ≤ v1.2, `whiten=True` was the default. In v1.3,
> string options were introduced and `whiten=True` triggers a `FutureWarning`. In v1.5+,
> passing `True` may raise an error. Always use the string form in new code.

---

### `whiten_solver` — Eigendecomposition Backend

**Default:** `'svd'` (added in sklearn v1.2 — flag this if using older versions)

**`'svd'`:** Computes SVD of the data matrix directly. More numerically stable. Faster when
$n \leq p$ (wide data — more features than samples).

**`'eigh'`:** Computes eigendecomposition of the covariance matrix $\mathbf{X}^\top\mathbf{X}$.
More memory efficient. Faster when $n \gg p$ (tall data, e.g., 100,000 EEG time points × 64
channels).

**Decision rule:**

| Data shape | Recommended `whiten_solver` |
|---|---|
| $n \leq p$ (wide) | `'svd'` (default) |
| $n \gg 50p$ (very tall) | `'eigh'` |
| Otherwise | `'svd'` (default) |

---

### `max_iter` and `tol` — Convergence Control

**`max_iter`** (default: 200): Maximum fixed-point iterations.

If `ica.n_iter_ == max_iter` after fitting, convergence was NOT achieved. Increase to 500, then
1000. If still failing, change `fun` or reduce `n_components`.

**`tol`** (default: 1e-4): Convergence threshold. Iteration stops when
$\|\mathbf{W}_{\text{new}} - \mathbf{W}_{\text{old}}\|_\infty < \text{tol}$.

**Recommended pairing by use case:**

| Context | `max_iter` | `tol` |
|---|---|---|
| Exploratory / first run | 200 | 1e-4 |
| Production ML pipeline | 500 | 1e-4 |
| EEG artifact removal (high quality) | 1000 | 1e-6 |
| fMRI network analysis | 500 | 1e-5 |

---

### `random_state` — Reproducibility

**Default:** `None` — different results every run.

**Always set this** in any code you intend to share, publish, or compare across runs.
`random_state=42` is conventional. The random state seeds the initial $\mathbf{W}$
initialization. Different seeds find the same sources but in different orderings (permutation
ambiguity) — all equally valid.

> 🏆 **Best practice:** Run with 5–10 different seeds in production and select the run with
> lowest reconstruction MSE. ICA's landscape has many valid local optima (all permutations of
> the same source set are valid), so multiple seeds help explore this landscape.

---

### Complete Tuning Playbook Table

| Parameter | Default | Effect of increasing | Effect of decreasing | When to tune | Search space |
|---|---|---|---|---|---|
| `n_components` | `None` | More structure captured; risk of instability if > rank | Faster; risk of missing sources | Almost always | PCA elbow → Optuna int |
| `fun` | `'logcosh'` | — (categorical) | — | Source distribution known | `['logcosh', 'exp']` |
| `fun_args['alpha']` | 1.0 | Emphasizes super-Gaussian | Emphasizes sub-Gaussian | When using logcosh | 1.0 → 2.0 |
| `algorithm` | `'parallel'` | — (categorical) | — | Stability vs. speed trade-off | `['parallel', 'deflation']` |
| `max_iter` | 200 | More likely to converge | May not converge | ConvergenceWarning fires | 200 → 2000 (log) |
| `tol` | 1e-4 | Looser convergence (faster) | Tighter convergence (slower) | Precision requirements | 1e-6 → 1e-2 (log) |
| `whiten_solver` | `'svd'` | — (categorical) | — | Very tall matrices | `['svd', 'eigh']` |
| `random_state` | `None` | — | — | Reproducibility | Fix at 42; try 5–10 seeds |

---

### Complete Optuna Search Space

```python
import optuna
from sklearn.decomposition import FastICA
import numpy as np

def ica_objective(trial, X):
    n_feat = X.shape[1]
    rank = np.linalg.matrix_rank(X)

    fun = trial.suggest_categorical('fun', ['logcosh', 'exp'])
    fun_args = None
    if fun == 'logcosh':
        alpha = trial.suggest_float('alpha', 1.0, 2.0, step=0.1)
        fun_args = {'alpha': alpha}

    params = {
        'n_components': trial.suggest_int('n_components', 2, min(rank - 1, n_feat - 1)),
        'algorithm': trial.suggest_categorical('algorithm', ['parallel', 'deflation']),
        'fun': fun,
        'fun_args': fun_args,
        'max_iter': trial.suggest_int('max_iter', 200, 1000, log=True),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
        'whiten': 'unit-variance',
        'random_state': 42,
    }

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ica = FastICA(**params)
        X_sources = ica.fit_transform(X)

    X_recon = ica.inverse_transform(X_sources)
    return np.linalg.norm(X - X_recon, 'fro') / np.linalg.norm(X, 'fro')

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: ica_objective(trial, X), n_trials=50)
print("Best params:", study.best_params)
```

---

## §5 — Strengths

### Strength 1: Recovers Statistically Independent Sources — Beyond Decorrelation

PCA removes linear correlations (second-order dependencies). ICA removes *all* statistical
dependencies — first, second, third, fourth, and higher-order. This is a fundamentally stronger
guarantee.

**Mechanism:** ICA maximizes non-Gaussianity, which is equivalent to maximizing statistical
independence under the linear mixing model. Two outputs $y_1, y_2$ are independent if
$p(y_1, y_2) = p(y_1)p(y_2)$ — not just $E\{y_1 y_2\} = 0$ (uncorrelated).

**When this matters:** Any time the underlying generating processes are genuinely independent:
brain networks, individual speakers, separate physical processes in industrial sensors.

---

### Strength 2: Blind Source Separation — No Reference Signals Needed

ICA recovers independent sources using *only* the statistical structure of the observed mixtures.
No reference signal, no training labels, no ground-truth sources, no knowledge of the mixing
process.

**Mechanism:** The identifiability theorem guarantees that the mixing matrix can be recovered
uniquely (up to permutation/scale) from the distribution of the observations alone, provided
sources are non-Gaussian and independent.

**When this matters:** Any real-world sensor array where the mixing is unknown: microphone
arrays, electrode arrays (EEG/MEG), antenna arrays (MIMO).

---

### Strength 3: Interpretable Components — Each IC Maps to a Physical Process

Unlike PCA components (abstract variance-maximizing directions) or t-SNE embeddings
(non-interpretable 2D coordinates), ICA components are designed to correspond to real
independent generating processes.

**Mechanism:** The independence constraint forces each component to represent a distinct
physical signal: an eye blink (EEG), a specific speaker (audio), a geological formation
(hyperspectral). This interpretability is often more valuable than the dimensionality reduction.

**When this matters:** Any application where you need to understand and label the discovered
components — clinical EEG analysis, fMRI network discovery, financial factor decomposition.

---

### Strength 4: Excellent Reconstruction — Perfect With $d = p$

In the square case (number of components = number of features), ICA is a rotation of the data
plus scaling — it is perfectly invertible. `ica.inverse_transform()` returns the original data
exactly (up to numerical precision).

**Mechanism:** `mixing_ = pseudo_inverse(components_)`. When `n_components == n_features`,
this is a true inverse. `inverse_transform(transform(X)) = X`.

**When this matters:** Artifact removal in EEG — you reconstruct the clean signal by zeroing
out artifact ICs and back-projecting into sensor space.

---

### Strength 5: Scales to High-Dimensional Time Series

For data with many time points and moderate feature count (e.g., EEG: 100,000 samples × 64
channels), the whitening step is $O(np^2)$ and the fixed-point iterations are $O(K \cdot ndp)$
with small $K$ — making ICA competitive even at substantial data scales.

---

## §6 — Weaknesses & Failure Modes

### Failure Mode 1: Gaussian Sources — The Fundamental Blocker

**Mechanism:** The identifiability theorem (Comon 1994) proves that ICA *cannot* separate
two or more Gaussian sources. Under whitening, any orthogonal rotation of Gaussian sources
produces output that looks identically Gaussian — there is no information to distinguish
the original orientation from any rotation.

**Detection:**
```python
from scipy import stats

for col in range(X.shape[1]):
    stat, p = stats.normaltest(X[:, col])
    if p > 0.05:
        print(f"Feature {col}: potentially Gaussian (p={p:.3f}) — ICA may fail")
```

**Mitigation:** If sources are Gaussian, PCA is the right tool. If only some sources are
Gaussian, ICA recovers the non-Gaussian ones but cannot separate the Gaussian mix.

---

### Failure Mode 2: Outliers Corrupting Kurtosis

**Mechanism:** `fun='cube'` uses $g(u) = u^3$, contributing $u^4$ to the expectation.
A single sample at $u = 10$ contributes 10,000 to the fourth moment vs. 100 from variance —
completely dominating the estimate. Even `fun='logcosh'` (bounded to $\tanh$) is somewhat
affected by clusters of outliers.

**Detection:**
```python
import numpy as np

# Check for extreme values
for col in range(X.shape[1]):
    q99 = np.percentile(np.abs(X[:, col]), 99)
    q50 = np.percentile(np.abs(X[:, col]), 50)
    if q99 / q50 > 10:
        print(f"Feature {col}: potential outliers (99th/50th percentile ratio = {q99/q50:.1f})")
```

**Mitigation:** Use `fun='logcosh'`, clip extreme values before fitting, or use robust
whitening (MCD estimator) instead of standard PCA whitening.

---

### Failure Mode 3: Non-Stationary Mixing

**Mechanism:** ICA assumes a fixed, time-invariant mixing matrix $\mathbf{A}$. If the mixing
changes over time (e.g., moving speakers, varying electrode impedance, shifting source
locations), a single $\mathbf{A}$ cannot describe the data — the recovered "sources" will
be artifacts of the non-stationarity.

**Detection:** Fit ICA on the first and second halves of your data separately. If components
change substantially between fits (different topomaps, different power spectra), the mixing
is non-stationary.

**Mitigation:** Use a sliding window approach, or online ICA methods that adapt $\mathbf{W}$
over time. For EEG, segment data into stationary epochs first.

---

### Failure Mode 4: More Sources Than Sensors (Underdetermined Case)

**Mechanism:** When there are more sources than observed channels ($d > p$), the mixing matrix
$\mathbf{A}$ has more columns than rows — it is wide (not square) and thus not invertible. The
standard ICA model is not identifiable.

**Detection:** This is a problem specification issue. If you have 2 microphones and 5 speakers,
standard ICA will give nonsensical results.

**Mitigation:** Overcomplete ICA with sparsity constraints (see §3.8), or reduce the number of
target components $d$ to $\leq p$.

---

### Failure Mode 5: Permutation and Sign Ambiguity in Automated Pipelines

**Mechanism:** ICA components are unordered and unsigned. If you run ICA twice with different
random seeds (or on slightly different data), component 0 in run A may correspond to component
2 in run B, with opposite sign. Automated pipelines that assume stable component indices will
produce inconsistent results.

**Detection:** Run ICA with 5 different seeds. Compare component matrices using Hungarian
algorithm matching on absolute cosine similarity. If top match scores are consistently high
(> 0.9), components are stable. If they're low (< 0.7), the solution is unstable.

**Mitigation:** Always label components by content (kurtosis, topography, power spectrum).
Use stability analysis (ICASSO — run ICA many times and cluster the components) to identify
robust vs. spurious ICs.

---

### Failure Mode 6: Convergence Failure on Rank-Deficient Data

**Mechanism:** EEG data with interpolated channels, re-referenced data, or data with repeated
features can be rank-deficient. The whitening step requires inverting eigenvalues, which becomes
numerically unstable for zero or near-zero eigenvalues.

**Detection:**
```python
rank = np.linalg.matrix_rank(X)
n_desired = ica.n_components_
if n_desired >= rank:
    print(f"WARNING: n_components={n_desired} >= rank(X)={rank}. "
          f"Reduce to {rank - 1}.")
```

**Mitigation:**
```python
rank = np.linalg.matrix_rank(X)
ica = FastICA(n_components=min(n_desired, rank - 1), ...)
```

---

### Summary of Failure Modes

| Failure mode | Symptom | Detection | Fix |
|---|---|---|---|
| Gaussian sources | Near-zero kurtosis components | Normality test | Use PCA instead |
| Outliers | Dominant single-sample kurtosis | 99th/50th ratio | `fun='logcosh'`; clip data |
| Non-stationary mixing | Different components on data halves | Split-half comparison | Online ICA; segment data |
| Underdetermined ($d > p$) | Nonsensical results | Problem specification | Reduce $d$; sparse ICA |
| Permutation instability | Inconsistent indices across seeds | Multi-seed comparison | Label by content; ICASSO |
| Rank-deficient data | ConvergenceWarning; LinAlgError | `matrix_rank` check | Set `n_components < rank` |

---

## §7 — What I Think About When Fitting

This section is the internal monologue I run through every time I apply ICA to a new dataset.
It is the difference between getting output and trusting output.

---

### Before Fitting — The Pre-Flight Checks

**"Is ICA even the right tool here?"**

ICA requires non-Gaussian independent sources with linear mixing. Before touching the code,
I ask: is this data the result of additive mixing of distinct processes? EEG: yes (brain
sources mix linearly across the scalp). Stock returns: probably not (returns are correlated,
tend toward Gaussian, and mixing is not physically linear). Text: possibly (documents are
mixtures of topics, but non-negativity matters — consider NMF first).

**"What does the marginal distribution of each feature look like?"**

```python
from scipy import stats
import numpy as np

for col in range(min(X.shape[1], 20)):   # check first 20 channels
    kurt = stats.kurtosis(X[:, col])
    _, p_norm = stats.normaltest(X[:, col])
    print(f"  col {col:2d}: kurtosis={kurt:+.2f}, normality p={p_norm:.3f} "
          f"{'[GAUSSIAN?]' if p_norm > 0.05 else ''}")
```

If most features look Gaussian (kurtosis ≈ 0, high normality p-value), I stop and use PCA.

**"What is the effective rank?"**

```python
rank = np.linalg.matrix_rank(X, tol=1e-6)
print(f"Data rank: {rank} / {X.shape[1]} features")
# If rank < n_features, I set n_components = rank - 1 as my upper limit
```

**"Are there obvious outliers?"**

```python
# Per-feature outlier check
for col in range(X.shape[1]):
    z = np.abs((X[:, col] - X[:, col].mean()) / X[:, col].std())
    n_extreme = (z > 5).sum()
    if n_extreme > 0:
        print(f"  col {col}: {n_extreme} samples >5σ — consider clipping")
```

For EEG, I always apply artifact rejection (amplitude thresholding, peak-to-peak rejection)
before ICA. For audio, I check for clipping. For financial data, I check for split-adjusted
and dividend-adjusted prices — unadjusted data has artificial outliers at split dates.

**"How many samples do I have relative to the number of components?"**

Rule of thumb: $n_{\text{samples}} \geq 50 \times n_{\text{components}}$ for stable ICA.
With 1,000 time points and 64 EEG channels, I'd use at most 20 components. With 10,000 time
points, I'm comfortable with 40–50 components.

---

### Choosing n_components — My Personal Workflow

1. Run PCA, look at scree plot (cumulative explained variance)
2. Note where the curve elbows (usually 70–90% variance explains the "signal")
3. Cross-check with `matrix_rank(X)` — never exceed rank − 1
4. Start conservatively (fewer components), run ICA, inspect components
5. If components look like noise (near-Gaussian, flat spectra), reduce further
6. If good components are being merged (you see two artifacts in one IC), increase

---

### During Fitting — What to Monitor

**"Did it converge?"**

```python
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    ica.fit(X)
    if any('did not converge' in str(warning.message) for warning in w):
        print(f"CONVERGENCE FAILURE! n_iter_={ica.n_iter_}, max_iter={ica.max_iter}")
        print("Actions: (1) increase max_iter, (2) change fun, (3) reduce n_components")

print(f"Converged in {ica.n_iter_} / {ica.max_iter} iterations")
```

If `n_iter_ == max_iter`, convergence was not achieved — the output is unreliable.

**"Multiple seeds — which is the best run?"**

For any important analysis, I never trust a single run:

```python
best_ica, best_err = None, np.inf
for seed in range(10):
    ica = FastICA(n_components=d, random_state=seed, max_iter=500, fun='logcosh')
    X_src = ica.fit_transform(X)
    X_recon = ica.inverse_transform(X_src)
    err = np.linalg.norm(X - X_recon, 'fro')
    if err < best_err:
        best_err = err
        best_ica = ica

print(f"Best seed reconstruction error: {best_err:.4f}")
```

This takes 10× the compute but gives me far more confidence in the result.

---

### After Fitting — The Diagnostic Pass

**"Do the recovered sources look non-Gaussian?"**

```python
from scipy.stats import kurtosis

X_src = best_ica.fit_transform(X)
for i in range(X_src.shape[1]):
    k = kurtosis(X_src[:, i])
    flag = "" if abs(k) > 0.5 else " [NEAR GAUSSIAN — check!]"
    print(f"  IC {i}: kurtosis = {k:+.3f}{flag}")
```

Near-Gaussian components (|kurtosis| < 0.5) are likely noise residuals or a sign that the
algorithm failed to find a meaningful non-Gaussian direction.

**"Does reconstruction work?"**

```python
X_recon = best_ica.inverse_transform(X_src)
rel_err = np.linalg.norm(X - X_recon, 'fro') / np.linalg.norm(X, 'fro')
print(f"Relative reconstruction error: {rel_err:.4f}")
# Near 0: good (d == p case or the whitening space captures all variance)
# Large: you've lost information in the whitening step (expected when d << p)
```

**"Do the components make physical sense?"**

This is domain-specific but critical:
- **EEG:** Plot topomaps. An eye-blink IC should show frontal polarity. A heart IC should show
  regular pulses in its time series. A muscle IC should have flat, high-frequency power spectrum.
- **Audio:** Listen to each component. A speech IC should sound like a voice. A background noise
  IC should sound like static.
- **Finance:** Correlate each IC with known market factors (VIX, sector ETFs, FX rates). ICs
  that don't correlate with anything known may be data artefacts.

**"If I see a Gaussian-looking IC at position 0..."**

I don't trust it. I investigate: was the data truly non-Gaussian? Was there a convergence
warning I missed? Did I have enough samples for the number of components I requested?

**"If components vary wildly across seeds..."**

I use stability analysis (run ICA 20× times with different seeds, cluster the resulting components
with hierarchical clustering, keep only stable clusters). In EEG analysis, this is called
ICASSO. A component that appears in less than 50% of runs is probably not a genuine source.

---

## §8 — Diagnostic Plots & Evaluation

Evaluating ICA output requires a different mindset from supervised learning. There is no
accuracy score. Instead, you assess: did the algorithm find statistically independent,
non-Gaussian components that correspond to meaningful physical processes?

---

### Diagnostic 1: Source Time Series Plot

**What it shows:** The time series of each recovered independent component.

**How to read it:** Meaningful sources have characteristic temporal patterns:
- Sinusoidal: oscillatory process (neural alpha oscillation, 50 Hz power line)
- Square wave: switching process (visual stimulation, digital artifact)
- Sparse spikes: super-Gaussian source (eye blinks, heartbeats, neural spikes)
- White noise: likely Gaussian residual — not a meaningful source

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

def plot_ica_sources(X, ica, n_components=None, title="ICA Source Time Series"):
    """Plot time series of all recovered ICA components."""
    X_sources = ica.transform(X)
    d = X_sources.shape[1] if n_components is None else n_components
    t = np.arange(X_sources.shape[0])

    fig, axes = plt.subplots(d, 1, figsize=(14, 2 * d), sharex=True)
    if d == 1:
        axes = [axes]

    from scipy.stats import kurtosis as scipy_kurtosis
    for i, ax in enumerate(axes):
        k = scipy_kurtosis(X_sources[:, i])
        ax.plot(t, X_sources[:, i], lw=0.7, color=f'C{i}')
        ax.set_ylabel(f'IC {i}\nkurt={k:.2f}', fontsize=9)
        ax.axhline(0, color='gray', lw=0.5, linestyle='--')

    axes[-1].set_xlabel("Time (samples)")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

# Usage
ica = FastICA(n_components=5, random_state=42, max_iter=500)
ica.fit(X)
plot_ica_sources(X, ica)
```

**Good:** Clear temporal patterns, smooth envelope, characteristic shape specific to each IC.
**Bad:** All components look like white noise, or all look identical.

---

### Diagnostic 2: Mixing Matrix Heatmap

**What it shows:** How each source contributes to each observed channel (the estimated $\hat{\mathbf{A}}$, stored as `ica.mixing_`).

**How to read it:** Each column of `mixing_` is the "spatial pattern" of one IC across all
observed features. For EEG, this would be a topomap. For sensor arrays, it shows which sensors
are most affected by each source.

```python
import seaborn as sns

def plot_mixing_matrix(ica, feature_names=None, figsize=(10, 6)):
    """Heatmap of the estimated mixing matrix A."""
    A = ica.mixing_   # shape: (n_features, n_components)
    n_feat, n_comp = A.shape

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        A,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        xticklabels=[f'IC {i}' for i in range(n_comp)],
        yticklabels=feature_names if feature_names is not None else range(n_feat),
        annot=(n_feat * n_comp <= 100),  # only annotate small matrices
        fmt='.2f'
    )
    ax.set_title("Estimated Mixing Matrix $\\hat{A}$ (columns = ICA spatial patterns)")
    ax.set_xlabel("Independent Component")
    ax.set_ylabel("Observed Feature / Channel")
    plt.tight_layout()
    plt.show()

plot_mixing_matrix(ica)
```

**Good:** Each column has a clear structure — a few large values and many near-zero values
(sparse spatial pattern), or a smooth spatial pattern (for EEG).
**Bad:** All columns look random (uniform values across features) — ICA didn't find structure.

---

### Diagnostic 3: Kurtosis Bar Chart

**What it shows:** The excess kurtosis of each recovered component, measuring its non-Gaussianity.

**How to read it:**
- |kurtosis| > 1.0: Strongly non-Gaussian — likely a meaningful source
- |kurtosis| 0.5–1.0: Moderately non-Gaussian — borderline
- |kurtosis| < 0.5: Near-Gaussian — likely noise residual or failed separation

```python
from scipy.stats import kurtosis as scipy_kurtosis

def plot_kurtosis_bars(X, ica):
    """Bar chart of IC kurtosis values."""
    X_sources = ica.transform(X)
    d = X_sources.shape[1]
    kurts = [scipy_kurtosis(X_sources[:, i]) for i in range(d)]

    colors = ['steelblue' if abs(k) > 0.5 else 'salmon' for k in kurts]

    fig, ax = plt.subplots(figsize=(max(8, d * 0.8), 4))
    bars = ax.bar(range(d), kurts, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(0.5, color='green', linewidth=0.8, linestyle='--', label='|kurt|=0.5 threshold')
    ax.axhline(-0.5, color='green', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Independent Component")
    ax.set_ylabel("Excess Kurtosis")
    ax.set_title("IC Non-Gaussianity: Kurtosis by Component\n"
                 "(blue = meaningful, red = near-Gaussian)")
    ax.set_xticks(range(d))
    ax.set_xticklabels([f'IC {i}' for i in range(d)])
    ax.legend()
    plt.tight_layout()
    plt.show()

    return kurts

kurts = plot_kurtosis_bars(X, ica)
```

---

### Diagnostic 4: Distribution Histograms (Non-Gaussianity Visual)

**What it shows:** The marginal distribution of each IC compared to a Gaussian reference.

```python
from scipy.stats import norm

def plot_ic_distributions(X, ica, n_cols=3):
    """Grid of IC histograms with Gaussian overlay."""
    X_sources = ica.transform(X)
    d = X_sources.shape[1]
    n_rows = (d + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if d > 1 else [axes]

    for i in range(d):
        ax = axes[i]
        data = X_sources[:, i]
        k = scipy_kurtosis(data)
        ax.hist(data, bins=50, density=True, alpha=0.7, color=f'C{i}', label='IC')
        xrange = np.linspace(data.min(), data.max(), 200)
        ax.plot(xrange, norm.pdf(xrange, 0, 1), 'r--', lw=1.5, label='N(0,1)')
        ax.set_title(f'IC {i}  (kurt={k:.2f})', fontsize=10)
        ax.legend(fontsize=7)

    for i in range(d, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("IC Marginal Distributions vs Gaussian", fontsize=12)
    plt.tight_layout()
    plt.show()

plot_ic_distributions(X, ica)
```

---

### Diagnostic 5: Reconstruction Error Analysis

**What it shows:** How much signal is preserved when using $k$ components vs. original data.

```python
def plot_reconstruction_error(X, max_components=None):
    """
    Reconstruction error vs number of ICA components.
    Shows the trade-off between compression and information loss.
    """
    if max_components is None:
        max_components = X.shape[1]

    errors = []
    components_range = range(1, max_components + 1)

    for d in components_range:
        ica = FastICA(n_components=d, random_state=42, max_iter=500,
                      whiten='unit-variance')
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_src = ica.fit_transform(X)
        X_recon = ica.inverse_transform(X_src)
        rel_err = np.linalg.norm(X - X_recon, 'fro') / np.linalg.norm(X, 'fro')
        errors.append(rel_err)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(components_range, errors, 'o-', color='steelblue', markersize=4)
    ax.axhline(0.05, color='red', linestyle='--', label='5% error threshold')
    ax.set_xlabel("Number of ICA Components")
    ax.set_ylabel("Relative Reconstruction Error")
    ax.set_title("ICA Reconstruction Error vs Number of Components")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return errors

errors = plot_reconstruction_error(X, max_components=min(20, X.shape[1]))
```

---

### Diagnostic 6: Amari Distance (When Ground Truth Available)

Use this metric when you have synthesized data and know the true mixing matrix:

```python
def amari_distance(W_est, A_true):
    """
    Amari distance — measures ICA separation quality against ground truth.
    0 = perfect. < 0.1 = good. > 0.5 = poor.

    Parameters
    ----------
    W_est : ndarray, shape (n_components, n_features)
        Estimated unmixing matrix (ica.components_ after fitting whitened data).
    A_true : ndarray, shape (n_features, n_components)
        True mixing matrix used to generate the data.
    """
    P = W_est @ A_true   # Should approximate a permutation+scale matrix
    n = P.shape[0]
    row_term = np.sum(np.sum(np.abs(P), axis=1) / np.max(np.abs(P), axis=1) - 1)
    col_term = np.sum(np.sum(np.abs(P), axis=0) / np.max(np.abs(P), axis=0) - 1)
    return (row_term + col_term) / (2 * n * (n - 1))

# Example usage with synthesized data
# A_true: the mixing matrix used to generate X
# ica.components_: the estimated unmixing vectors (whitened space)
# For the full-space unmixing, use: W_full = ica.components_ @ ica.whitening_... etc.
# See §10 worked example for complete implementation
```

---

### Summary Evaluation Table

| Metric | When to use | What good looks like |
|---|---|---|
| Reconstruction error | Always | < 5% relative error (if $d = p$: ≈ 0) |
| Kurtosis per IC | Always | $|$kurtosis$| > 0.5$ for all retained ICs |
| IC distribution histograms | Always | Non-bell-shaped, distinct from Gaussian |
| Source time series | Always | Clear temporal patterns (not white noise) |
| Mixing matrix heatmap | When features are interpretable | Sparse or structured column patterns |
| Amari distance | Simulated data only | < 0.1 for good separation |
| Multi-seed stability | Any critical analysis | > 0.85 cosine similarity across seeds |

---

## §9 — Innovative Industry Applications

ICA is one of the most domain-versatile algorithms in the ML toolkit. Here are eight
applications ranging from the canonical to the genuinely surprising.

---

### Application 1: EEG Artifact Removal in Neuroscience & Clinical BCI

> 🏭 **Industry application:** Used in essentially every serious EEG research laboratory
> worldwide; standard preprocessing in 90%+ of published EEG papers.

**The problem:** EEG electrodes on the scalp record voltage differences at 1,000+ samples/second
across 32–256 channels. The neural signals of interest (microvolts) are overwhelmed by:
- Eye blinks: 50–200 µV, 10–50× larger than neural signals
- Heartbeats: regular QRS complex contaminating frontal channels
- Muscle tension (EMG): broadband high-frequency noise
- Power line interference: 50/60 Hz contamination

**Why ICA:** The scalp's electrical conductivity creates precisely the linear spatial mixing
that ICA requires. A blink generates a fixed voltage pattern across the scalp (its "spatial
topography") that is statistically independent of ongoing neural activity. ICA recovers this
as a single component that can be identified and removed.

**The workflow:**
1. High-pass filter ≥1 Hz (remove slow drift)
2. Fit ICA with 0.7 × n_channels components
3. Automatically identify artifacts via `find_bads_eog()` (eye), `find_bads_ecg()` (heart)
4. Zero out artifact ICs
5. Back-project to sensor space

```python
import mne
from mne.preprocessing import ICA

# Assume raw is an MNE Raw object, already loaded
raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

# Fit ICA with Picard (fastest for real EEG)
ica = ICA(
    n_components=20,
    method='picard',
    random_state=42,
    max_iter='auto'
)
ica.fit(raw, picks='eeg')

# Auto-detect and exclude artifacts
eog_idx, _ = ica.find_bads_eog(raw, threshold=3.0)
ecg_idx, _ = ica.find_bads_ecg(raw, method='correlation', threshold=0.25)
ica.exclude = eog_idx + ecg_idx
print(f"Excluding {len(ica.exclude)} artifact ICs: {ica.exclude}")

# Apply artifact removal
raw_clean = ica.apply(raw.copy())
```

---

### Application 2: fMRI Resting-State Brain Network Discovery

> 🏭 **Industry application:** FSL's MELODIC ICA is the standard analysis pipeline for
> resting-state fMRI used at thousands of neuroimaging sites worldwide; ICA biomarkers for
> Alzheimer's are an active clinical diagnostic research area.

**The problem:** During "resting state" fMRI (subject lying still), the brain shows spontaneous
BOLD signal correlations in spatially distributed networks. Traditional analysis requires a
task paradigm — how do you discover brain networks without one?

**Why ICA:** Spatial ICA decomposes the $T \times V$ fMRI matrix (T timepoints × V voxels)
into spatially independent components, each with its own time course. Brain networks
(Default Mode, Sensorimotor, Visual) emerge as ICs because their activity is statistically
independent from other networks — they operate on different temporal scales and spatial patterns.

**High-dim vs. low-dim ICA:**
- **Low-dim (10–30 ICs):** Recovers major networks — Default Mode, Sensorimotor, Visual, etc.
- **High-dim (70–100 ICs):** Sub-parcellates networks — detects within-Default-Mode damage
  in Alzheimer's earlier than structural MRI can

```python
# Conceptual sketch — fMRI ICA uses nilearn in practice
from nilearn.decomposition import CanICA
from nilearn import datasets, plotting

# Load example fMRI data
dataset = datasets.fetch_atlas_destrieux_2009()  # or fetch_development_fmri()

# Spatial Group ICA
canica = CanICA(
    n_components=20,
    random_state=42,
    memory='nilearn_cache',
    verbose=0
)
# canica.fit(fmri_filenames)  # list of NIfTI files
# components = canica.components_img_  # spatial maps
```

---

### Application 3: Fetal ECG Extraction — Non-Invasive Obstetric Monitoring

> 🏭 **Industry application:** Research hospitals worldwide use ICA-based BSS for non-invasive
> fetal monitoring. FDA-cleared commercial fetal monitors use related signal separation techniques.

**The problem:** Monitoring fetal heart rate traditionally requires either invasive scalp
electrodes (infection risk) or the inherently noisy cardiotocography (CTG). Abdominal ECG
electrodes offer a non-invasive alternative, but the maternal ECG (10× stronger) completely
masks the fetal signal.

**Why ICA:** With 4–8 abdominal electrodes, each records a different linear combination of:
maternal heart (strong), fetal heart (weak but independent — different rate, independent
rhythm), and noise. ICA separates these because the maternal and fetal hearts beat independently.

**Key setup detail:** ECG signals are super-Gaussian (sparse spikes) — use `fun='exp'` for
better separation of spike-like sources than the default `'logcosh'`.

```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt

def extract_fetal_ecg(abdominal_signals, fs=1000):
    """
    Extract fetal ECG from abdominal electrode recordings.

    Parameters
    ----------
    abdominal_signals : ndarray, shape (n_samples, n_electrodes)
        Raw abdominal ECG recordings from multiple electrodes.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    ica_components : ndarray, shape (n_samples, n_electrodes)
        ICA components — inspect for fetal ECG (typically smaller amplitude,
        higher rate ~140 bpm vs maternal ~70 bpm).
    ica : FastICA
        Fitted ICA object for back-projection.
    """
    # Bandpass filter 1-250 Hz (ECG range)
    b, a = butter(4, [1, 250], btype='bandpass', fs=fs)
    X_filtered = filtfilt(b, a, abdominal_signals, axis=0)

    # ICA — use 'exp' for super-Gaussian ECG signals (sparse spikes)
    ica = FastICA(
        n_components=abdominal_signals.shape[1],
        fun='exp',            # better for sparse spike-like sources
        whiten='unit-variance',
        max_iter=1000,
        random_state=42
    )
    ica_components = ica.fit_transform(X_filtered)

    return ica_components, ica

# User then inspects components for:
# - Regular high-rate (~140 bpm) spike pattern → fetal ECG
# - Regular lower-rate (~70 bpm) spike pattern → maternal ECG
# - Broadband noise → noise components
```

> 📜 **Reference:** PMC11117810 — "Development and Implementation of Innovative BSS Techniques
> for Real-Time Extraction and Analysis of Fetal ECG Signals" (2024).
> https://pmc.ncbi.nlm.nih.gov/articles/PMC11117810/

---

### Application 4: Hyperspectral Image Unmixing — Satellite Remote Sensing

> 🏭 **Industry application:** NASA's AVIRIS and HYPERION instruments generate hyperspectral
> imagery used for mineral mapping, vegetation stress, and planetary analysis (Mars CRISM).

**The problem:** Each pixel in a hyperspectral image records reflectance across 200+ spectral
bands. In practice, most pixels contain mixtures of multiple materials (bare soil, vegetation,
water, minerals). Spectral unmixing estimates what fraction of each material each pixel contains.

**Why ICA:** If pure material spectra (endmembers) are statistically independent — which they
often are, since soil reflectance is independent of vegetation reflectance — ICA recovers
them as components without needing a spectral library.

**Limitation:** NMF (Non-negative Matrix Factorization) is often preferred because it
enforces non-negativity constraints (abundances can't be negative). ICA and NMF are
complementary: use ICA first for exploration, NMF for constrained quantitative unmixing.

```python
import numpy as np
from sklearn.decomposition import FastICA

def hyperspectral_ica_unmixing(image_cube, n_endmembers):
    """
    ICA-based spectral unmixing of a hyperspectral image.

    Parameters
    ----------
    image_cube : ndarray, shape (n_rows, n_cols, n_bands)
        Hyperspectral image.
    n_endmembers : int
        Number of endmember materials to extract.

    Returns
    -------
    abundance_maps : ndarray, shape (n_rows, n_cols, n_endmembers)
        Spatial abundance maps for each endmember.
    endmember_spectra : ndarray, shape (n_bands, n_endmembers)
        Recovered spectral signatures.
    """
    n_rows, n_cols, n_bands = image_cube.shape
    X = image_cube.reshape(-1, n_bands)   # (n_pixels, n_bands)

    ica = FastICA(
        n_components=n_endmembers,
        fun='exp',              # hyperspectral endmembers are often sparse
        whiten='unit-variance',
        max_iter=500,
        random_state=42
    )
    abundances = ica.fit_transform(X)   # (n_pixels, n_endmembers)
    endmember_spectra = ica.mixing_     # (n_bands, n_endmembers)

    abundance_maps = abundances.reshape(n_rows, n_cols, n_endmembers)
    return abundance_maps, endmember_spectra
```

---

### Application 5: Financial Market Factor Decomposition

> 🏭 **Industry application:** Back & Weigend (1997) first demonstrated ICA on Japanese stock
> data; modern quant funds use ICA-based factor models for alternative risk premia extraction.

**The problem:** Portfolio returns reflect multiple latent market forces — global macro risk,
sector rotation, currency effects, earnings quality. PCA finds orthogonal factors but they
don't separate into interpretable economic drivers. Can we find *independent* market forces?

**Why ICA (and its limits):** Unlike PCA factors, ICA factors are statistically independent —
better reflecting the independence of distinct market regimes. ICA identifies "shocks" —
infrequent large moves (super-Gaussian, captured well by ICA) vs. frequent small fluctuations.

**Critical caveat:** Daily aggregate stock returns tend toward Gaussian (CLT applies at the
daily scale). Use intraday returns, longer windows, or returns minus GARCH-fitted conditional
mean to get fat-tailed distributions that ICA can exploit.

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

def financial_ica_factors(returns_df, n_factors=5):
    """
    Extract independent market factors from portfolio returns.

    Parameters
    ----------
    returns_df : DataFrame, shape (n_days, n_assets)
        Asset return series.
    n_factors : int
        Number of independent factors to extract.

    Returns
    -------
    factor_returns : DataFrame, shape (n_days, n_factors)
        Time series of independent market factors.
    loadings : DataFrame, shape (n_assets, n_factors)
        Asset loadings on each factor.
    """
    # Standardize (ICA is scale-invariant but centering helps)
    scaler = StandardScaler()
    X = scaler.fit_transform(returns_df.values)

    ica = FastICA(
        n_components=n_factors,
        fun='logcosh',          # returns have mixed sub/super-Gaussian structure
        whiten='unit-variance',
        max_iter=500,
        random_state=42
    )
    factors = ica.fit_transform(X)   # (n_days, n_factors)

    # Compute kurtosis to identify factor type
    from scipy.stats import kurtosis
    for i in range(n_factors):
        k = kurtosis(factors[:, i])
        print(f"Factor {i}: kurtosis={k:.2f} "
              f"({'shock/event-driven' if k > 2 else 'smooth/trend'})")

    factor_returns = pd.DataFrame(
        factors,
        index=returns_df.index,
        columns=[f'IC_{i}' for i in range(n_factors)]
    )
    loadings = pd.DataFrame(
        ica.mixing_,
        index=returns_df.columns,
        columns=[f'IC_{i}' for i in range(n_factors)]
    )
    return factor_returns, loadings
```

> 📜 **Reference:** Back, A. D., & Weigend, A. S. (1997). A first application of ICA to
> extracting structure from stock returns. *International Journal of Neural Systems*, 8.

---

### Application 6: MIMO Telecommunications Channel Equalization (Non-obvious!)

> 🏭 **Industry application:** Active research application in 5G/6G MIMO systems;
> demonstrated in IEEE publications for blind beamforming.

**The problem:** In MIMO (Multiple Input, Multiple Output) wireless systems, multiple
transmit antennas create interfering signal paths at multiple receive antennas. Channel
equalization (separating the transmitted streams) normally requires pilot signals — overhead
that reduces throughput.

**Why ICA:** Transmitted symbol streams from different antennas carry independent data
streams. ICA recovers the unmixing matrix (channel equalization) from received signals
alone — "blind" equalization that eliminates pilot overhead and increases spectral efficiency
by ~10–15%.

**Limitation:** Works for quasi-static channels (slow fading). Fast-fading channels violate
ICA's stationarity assumption. Performance degrades in highly reverberant environments.

> 📜 **Reference:** Springer Wireless Personal Communications — "ICA Based MIMO Transceiver
> For Time Varying Wireless Channels" (2017).

---

### Application 7: Industrial Process Monitoring — Wastewater Treatment (Non-obvious!)

> 🏭 **Industry application:** Research published in 2025 demonstrates ICA for fault
> detection and process optimization at wastewater treatment plants.

**The problem:** Wastewater treatment plants generate continuous streams of multivariate,
non-Gaussian sensor data: influent flow, temperature, dissolved oxygen, ammonia, turbidity,
pH, and dozens more. Process faults (aeration failure, sludge bulking, shock loads) need
early detection. PCA misses important non-Gaussian fault signals.

**Why ICA:** Unlike PCA (which extracts correlated principal components), ICA extracts
statistically independent factors representing distinct underlying processes — biological
oxidation, settling, chemical addition. These non-Gaussian sensor patterns match ICA's
assumptions far better than PCA's variance-maximization.

**What ICA finds:** Interpretable independent process components corresponding to normal
biological activity, shock loads, aeration failures — enabling earlier fault detection and
root-cause analysis.

> 📜 **Reference:** ScienceDirect (2025) — "Independent component analysis in wastewater
> treatment plants: Unlocking process understanding and performance optimization."
> https://www.sciencedirect.com/science/article/abs/pii/S2214714425017209

---

### Application 8: Image Feature Extraction — Face Recognition

> 🏭 **Industry application:** ICA basis functions for faces outperform PCA (eigenfaces) on
> face verification tasks — demonstrated in Bartlett et al. (2002).

**The problem:** PCA eigenfaces represent global luminance patterns — eigenvectors that look
like blurry composites of many faces. They capture variance but not the locally independent
facial features (eyes, nose, mouth) that are actually diagnostic for recognition.

**Why ICA:** ICA basis functions are sparse and localized — each IC represents a local facial
feature (an eye, a nose tip, a cheek shadow) rather than a global pattern. These localized
features are more robust to partial occlusion and correspond better to the visual system's
processing hierarchy.

```python
from sklearn.decomposition import FastICA, PCA
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np

def compare_pca_vs_ica_faces():
    """Extract and visualize PCA eigenfaces vs ICA basis functions."""
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data      # (400, 4096) — 400 images, 64×64 pixels
    n_components = 16

    # PCA: global variance-maximizing components
    pca = PCA(n_components=n_components, random_state=42, whiten=True)
    pca.fit(X)
    eigenfaces = pca.components_.reshape(n_components, 64, 64)

    # ICA: statistically independent local features
    ica = FastICA(n_components=n_components, whiten='unit-variance',
                  max_iter=500, random_state=42)
    ica.fit(X)
    ica_faces = ica.components_.reshape(n_components, 64, 64)

    fig, axes = plt.subplots(2, n_components, figsize=(2 * n_components, 5))
    for i in range(n_components):
        axes[0, i].imshow(eigenfaces[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(ica_faces[i], cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('PCA\nEigenfaces', fontsize=10)
    axes[1, 0].set_ylabel('ICA\nBasis', fontsize=10)
    fig.suptitle('PCA vs ICA: Face Representation', fontsize=12)
    plt.tight_layout()
    plt.show()

compare_pca_vs_ica_faces()
```

**What to observe:** PCA eigenfaces look like ghostly face composites. ICA basis functions
show sparse, localized features: individual eye patterns, nose shapes, facial texture patches.

---

## §10 — Complete Python Worked Example

This section runs three complete end-to-end examples: the canonical cocktail party simulation,
the Olivetti faces dataset, and the digits dataset. Each demonstrates a different facet of ICA.

---

### Part A: The Cocktail Party Problem — Source Separation with Ground Truth

This is the pedagogically perfect example because we know the true sources and can compute
the Amari distance to measure separation quality.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Simulate the cocktail party problem
# ═══════════════════════════════════════════════════════════════════════════

rng = np.random.RandomState(42)
n_samples = 4000
t = np.linspace(0, 8, n_samples)

# Three independent non-Gaussian sources
s1 = np.sin(2 * np.pi * 1.5 * t)                         # 1.5 Hz sinusoid (sub-Gaussian)
s2 = np.sign(np.sin(2 * np.pi * 2.0 * t))                # 2 Hz square wave (sub-Gaussian)
s3 = rng.standard_cauchy(n_samples)                        # Cauchy noise (super-Gaussian)
s3 = np.clip(s3, -4, 4)

S_true = np.column_stack([s1, s2, s3])
S_true /= S_true.std(axis=0)   # unit variance sources

print("True source statistics:")
for i in range(3):
    k = kurtosis(S_true[:, i])
    print(f"  Source {i}: kurtosis={k:+.2f}, std={S_true[:, i].std():.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Mix the signals
# ═══════════════════════════════════════════════════════════════════════════

# Random mixing matrix (unknown to the algorithm)
A_true = np.array([[1.0, 1.0, 1.0],
                   [0.5, 2.0, 1.0],
                   [1.5, 1.0, 2.0]])

X_mixed = S_true @ A_true.T    # (4000, 3) — "microphone recordings"
X_mixed = StandardScaler().fit_transform(X_mixed)

print(f"\nMixed signal shape: {X_mixed.shape}")
print(f"Mixed signal correlations:\n{np.round(np.corrcoef(X_mixed.T), 3)}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Run FastICA with multiple seeds
# ═══════════════════════════════════════════════════════════════════════════

best_ica, best_err = None, np.inf
for seed in range(10):
    ica = FastICA(
        n_components=3,
        algorithm='parallel',
        fun='logcosh',
        fun_args={'alpha': 1.0},
        whiten='unit-variance',
        max_iter=500,
        tol=1e-5,
        random_state=seed
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X_src = ica.fit_transform(X_mixed)

    X_recon = ica.inverse_transform(X_src)
    err = np.linalg.norm(X_mixed - X_recon, 'fro')
    if err < best_err:
        best_err = err
        best_ica = ica
        best_seed = seed

print(f"\nBest seed: {best_seed}, reconstruction error: {best_err:.4f}")
print(f"Iterations to convergence: {best_ica.n_iter_}")

X_sources_hat = best_ica.fit_transform(X_mixed)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Evaluate separation quality
# ═══════════════════════════════════════════════════════════════════════════

def amari_distance(W_est, A_true):
    """Amari distance: 0 = perfect, higher = worse."""
    P = W_est @ A_true
    n = P.shape[0]
    row_term = np.sum(np.sum(np.abs(P), axis=1) / np.max(np.abs(P), axis=1) - 1)
    col_term = np.sum(np.sum(np.abs(P), axis=0) / np.max(np.abs(P), axis=0) - 1)
    return (row_term + col_term) / (2 * n * (n - 1))

# Compute Amari distance
# Note: ica.components_ is the whitened-space unmixing matrix
# For Amari distance, use the full demixing (whitening + rotation)
W_full = best_ica.components_ @ best_ica.whitening_  # (n_comp, n_features)
amari = amari_distance(W_full, A_true)
print(f"\nAmari distance: {amari:.4f} (0=perfect, <0.1=good)")

print("\nRecovered source statistics:")
for i in range(3):
    k = kurtosis(X_sources_hat[:, i])
    print(f"  IC {i}: kurtosis={k:+.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Visualize — side-by-side comparison
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 3, figsize=(15, 8))
time_window = slice(0, 400)   # show first 400 samples

for i in range(3):
    # True sources (left)
    axes[i, 0].plot(S_true[time_window, i], lw=0.8, color='darkblue')
    axes[i, 0].set_title(f'True Source {i}' if i == 0 else '')
    axes[i, 0].set_ylabel(f'Source {i}')

    # Mixed signals (middle)
    axes[i, 1].plot(X_mixed[time_window, i], lw=0.8, color='darkorange')
    axes[i, 1].set_title('Mixed Signal' if i == 0 else '')

    # Recovered ICA components (right)
    axes[i, 2].plot(X_sources_hat[time_window, i], lw=0.8, color='darkgreen')
    axes[i, 2].set_title(f'ICA Recovered (Amari={amari:.3f})' if i == 0 else '')

axes[0, 0].set_title('True Sources', fontsize=11)
axes[0, 1].set_title('Observed Mixtures', fontsize=11)
axes[0, 2].set_title('ICA Recovered Sources', fontsize=11)
for ax in axes[-1, :]:
    ax.set_xlabel('Time (samples)')

plt.suptitle('Cocktail Party: ICA Source Separation', fontsize=13)
plt.tight_layout()
plt.savefig('ica_cocktail_party.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Also run PCA for comparison
# ═══════════════════════════════════════════════════════════════════════════

pca = PCA(n_components=3, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_mixed)

print("\nPCA components kurtosis (should be near 0 — not separated):")
for i in range(3):
    k = kurtosis(X_pca[:, i])
    print(f"  PC {i}: kurtosis={k:+.2f}")

print("\nICA components kurtosis (should be far from 0 — separated):")
for i in range(3):
    k = kurtosis(X_sources_hat[:, i])
    print(f"  IC {i}: kurtosis={k:+.2f}")
```

**What to observe:** PCA principal components have near-zero kurtosis — they are still
mixtures of the non-Gaussian sources, smeared back toward Gaussian by the mixing. ICA
components have high |kurtosis| and visually resemble the original sinusoid, square wave,
and Cauchy noise. The Amari distance should be < 0.05 for well-separated sources.

---

### Part B: Olivetti Faces — ICA vs PCA Basis Functions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load Olivetti faces dataset
# ═══════════════════════════════════════════════════════════════════════════

print("Loading Olivetti faces dataset...")
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data          # (400, 4096): 400 images, 64×64 pixels
y = faces.target        # (400,): person IDs 0–39

print(f"Dataset shape: {X.shape}")
print(f"Pixel value range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Unique subjects: {len(np.unique(y))}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Fit ICA and PCA
# ═══════════════════════════════════════════════════════════════════════════

n_components = 20

# PCA baseline
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_pca = pca.fit_transform(X)
print(f"\nPCA: {pca.explained_variance_ratio_.sum()*100:.1f}% variance explained")

# FastICA
ica = FastICA(
    n_components=n_components,
    fun='logcosh',
    whiten='unit-variance',
    max_iter=500,
    tol=1e-4,
    random_state=42
)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    X_ica = ica.fit_transform(X)

# Reconstruction error
X_recon_ica = ica.inverse_transform(X_ica)
X_recon_pca = pca.inverse_transform(X_pca)
print(f"ICA reconstruction error: {np.mean((X - X_recon_ica)**2):.4f}")
print(f"PCA reconstruction error: {np.mean((X - X_recon_pca)**2):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Visualize basis functions (eigenfaces vs ICA faces)
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, n_components, figsize=(25, 8))

for i in range(n_components):
    # PCA eigenfaces (row 0)
    eigenface = pca.components_[i].reshape(64, 64)
    axes[0, i].imshow(eigenface, cmap='gray', vmin=eigenface.min(), vmax=eigenface.max())
    axes[0, i].axis('off')

    # ICA basis faces (row 1)
    ica_face = ica.components_[i].reshape(64, 64)
    axes[1, i].imshow(ica_face, cmap='gray', vmin=ica_face.min(), vmax=ica_face.max())
    axes[1, i].axis('off')

    # Kurtosis comparison (row 2: bar)
    k_ica = kurtosis(X_ica[:, i])
    k_pca = kurtosis(X_pca[:, i])
    axes[2, i].bar(['PCA', 'ICA'], [abs(k_pca), abs(k_ica)],
                   color=['steelblue', 'darkorange'])
    axes[2, i].set_title(f'|kurt|', fontsize=6)
    axes[2, i].tick_params(labelsize=6)

axes[0, 0].set_ylabel('PCA\nEigenfaces', fontsize=10)
axes[1, 0].set_ylabel('ICA\nBasis', fontsize=10)
axes[2, 0].set_ylabel('|Kurtosis|', fontsize=10)
plt.suptitle('Olivetti Faces: PCA Eigenfaces vs ICA Independent Basis Functions\n'
             'ICA basis functions are sparser and more localized', fontsize=11)
plt.tight_layout()
plt.savefig('ica_faces_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nKurtosis comparison:")
print(f"  Mean |kurtosis| — PCA components: {np.mean([abs(kurtosis(X_pca[:, i])) for i in range(n_components)]):.2f}")
print(f"  Mean |kurtosis| — ICA components: {np.mean([abs(kurtosis(X_ica[:, i])) for i in range(n_components)]):.2f}")
```

**What to observe:** PCA eigenfaces look like blurry, global face patterns. ICA basis
functions are sparse and localized — individual eye, nose, and facial shadow patterns emerge.
ICA components have substantially higher mean absolute kurtosis than PCA components,
confirming they capture more non-Gaussian structure.

---

### Part C: Digits Dataset — ICA for Feature Extraction + Downstream Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import warnings

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load and inspect the digits dataset
# ═══════════════════════════════════════════════════════════════════════════

digits = load_digits()
X, y = digits.data, digits.target   # (1797, 64), labels 0–9
print(f"Digits dataset: {X.shape[0]} samples, {X.shape[1]} features (8×8 pixels)")
print(f"Classes: {np.unique(y)} ({len(np.unique(y))} digits)")

# Check non-Gaussianity of raw features
print("\nNon-Gaussianity check (first 8 pixels):")
for col in range(8):
    k = kurtosis(X[:, col])
    print(f"  Pixel {col}: kurtosis={k:+.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Compare PCA and ICA as preprocessing for classification
# ═══════════════════════════════════════════════════════════════════════════

n_components_list = [5, 10, 15, 20, 30, 40]
pca_scores = []
ica_scores = []

for n_comp in n_components_list:
    # PCA pipeline
    pca_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp, random_state=42)),
        ('clf', LogisticRegression(max_iter=500, random_state=42))
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pca_acc = cross_val_score(pca_pipe, X, y, cv=5, scoring='accuracy').mean()

    # ICA pipeline
    ica_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ica', FastICA(n_components=n_comp, whiten='unit-variance',
                        fun='logcosh', max_iter=500, random_state=42)),
        ('clf', LogisticRegression(max_iter=500, random_state=42))
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ica_acc = cross_val_score(ica_pipe, X, y, cv=5, scoring='accuracy').mean()

    pca_scores.append(pca_acc)
    ica_scores.append(ica_acc)
    print(f"n_components={n_comp:3d}: PCA={pca_acc:.3f}, ICA={ica_acc:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Plot classification accuracy vs components
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_components_list, pca_scores, 'o-', label='PCA + LR', color='steelblue')
ax.plot(n_components_list, ica_scores, 's-', label='ICA + LR', color='darkorange')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cross-validation Accuracy (5-fold)')
ax.set_title('Digits Classification: PCA vs ICA Preprocessing\n'
             '(Logistic Regression downstream)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ica_digits_classification.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Visualize ICA components (digit basis functions)
# ═══════════════════════════════════════════════════════════════════════════

n_comp = 20
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ica = FastICA(n_components=n_comp, whiten='unit-variance',
              fun='logcosh', max_iter=500, random_state=42)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    X_ica = ica.fit_transform(X_scaled)

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
axes = axes.flatten()
for i in range(n_comp):
    ic = ica.components_[i].reshape(8, 8)
    k = kurtosis(X_ica[:, i])
    axes[i].imshow(ic, cmap='RdBu_r', vmin=-ic.std()*3, vmax=ic.std()*3)
    axes[i].set_title(f'IC {i}\nkurt={k:.1f}', fontsize=8)
    axes[i].axis('off')

plt.suptitle(f'ICA Basis Functions for Digits (n_components={n_comp})\n'
             'Each IC represents an independent stroke/feature', fontsize=11)
plt.tight_layout()
plt.savefig('ica_digits_basis.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Reconstruction demonstration
# ═══════════════════════════════════════════════════════════════════════════

X_recon = scaler.inverse_transform(ica.inverse_transform(X_ica))
recon_mse = np.mean((X - X_recon)**2)
recon_rel = np.linalg.norm(X - X_recon, 'fro') / np.linalg.norm(X, 'fro')
print(f"\nReconstruction (n_components={n_comp}):")
print(f"  MSE: {recon_mse:.4f}")
print(f"  Relative error: {recon_rel:.4f} ({recon_rel*100:.1f}%)")
```

**Key findings from this example:**
1. ICA and PCA achieve similar classification accuracy on digits — they are both linear
   feature extractors, and the classification downstream doesn't directly benefit from
   independence vs. uncorrelation
2. ICA basis functions look like pen stroke fragments — more interpretable as visual primitives
   than PCA eigendigits
3. Reconstruction error is expected since `n_components=20 < 64 features` — the whitening
   discards dimensions, just like PCA

**The important lesson:** ICA beats PCA when you need *independent* components for
artifact removal or source separation. For pure feature extraction and downstream classification,
the advantage is less clear — use the diagnostic plots to understand what each algorithm gives you.

---

## §11 — When to Use ICA: Decision Guide

### The Core Decision Tree

```
START: Do you need to separate mixed signals into their original components?
│
├── YES: Do you have reason to believe the mixing is LINEAR?
│   │
│   ├── YES: Are the sources statistically independent and non-Gaussian?
│   │   │
│   │   ├── YES → USE ICA ✓
│   │   │   (cocktail party, EEG artifact removal, fMRI RSN, fetal ECG)
│   │   │
│   │   └── Sources may be Gaussian → USE PCA (ICA is non-identifiable for Gaussian sources)
│   │
│   └── NO (nonlinear mixing): USE deep BSS / NNICA / VAE-based separation
│
└── NO: What is your actual goal?
    │
    ├── Visualization (2D/3D) → USE t-SNE or UMAP
    ├── Compression / reconstruction → USE PCA or Autoencoder
    ├── Non-negative parts-based representation → USE NMF
    ├── Supervised dimensionality reduction → USE LDA
    ├── Classification preprocessing → Try PCA first; ICA if domain-driven
    └── Noise reduction → PCA (remove low-variance components)
```

---

### ICA vs Closest Alternatives

| Situation | Use ICA | Use PCA | Use NMF | Use t-SNE/UMAP |
|---|---|---|---|---|
| **Need statistical independence** | ✓ | — | — | — |
| **Source separation (BSS)** | ✓ | — | — | — |
| **Gaussian sources** | — | ✓ | — | — |
| **Non-negative parts (e.g., topics, spectra)** | — | — | ✓ | — |
| **Compression + perfect reconstruction** | ✓ (d=p) | ✓ | ✓ | — |
| **2D visualization** | partial | ✓ (first 2 PCs) | — | ✓ |
| **No domain knowledge needed** | ✓ | ✓ | ✓ | ✓ |
| **Large $n$, large $p$** | slow | ✓ | ✓ | very slow |
| **Need ordered components** | — | ✓ | — | — |
| **EEG artifact removal** | ✓ | — | — | — |
| **fMRI network discovery** | ✓ | — | — | — |
| **Image compression** | — | ✓ | — | — |
| **Topic modeling (text)** | — | — | ✓ | — |

---

### Red Flags — When NOT to Use ICA

1. **Your sources are approximately Gaussian** (daily stock returns, measurement noise):
   ICA is non-identifiable. Use PCA.

2. **You need ordered components** (most important first): ICA components have no natural
   ordering. Use PCA.

3. **You need non-negative components** (spectral abundances, document topics, image patches):
   ICA allows negative values. Use NMF.

4. **Your sample size is small** ($n < 50d$): ICA's higher-order moment estimates are
   unreliable. Use PCA.

5. **You need global topology preservation for visualization**: ICA doesn't optimize for
   neighborhood structure. Use t-SNE or UMAP.

6. **The mixing is nonlinear** (reverberant audio, nonlinear sensor fusion): Linear ICA
   will give poor results. Use deep learning BSS.

7. **You need interpretable variance explained**: ICA has no clean variance decomposition
   (components are not ordered). Use PCA for explained variance analysis.

---

## §12 — Top Papers to Study

These are the papers that will take you from "I can call FastICA" to genuinely understanding
the field. Ranked from essential to deep mastery.

---

### START HERE

**1. Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications.**
*Neural Networks*, 13(4–5), 411–430.
DOI: [10.1016/S0893-6080(00)00026-5](https://doi.org/10.1016/S0893-6080(00)00026-5)
ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0893608000000265

**Annotation:** This is the most readable comprehensive treatment of ICA. It covers the
generative model, identifiability, preprocessing (centering, whitening), the negentropy
approximation, FastICA, and multiple application domains. If you read only one ICA paper,
read this one. It is also the reference cited in sklearn's own documentation.
*Why read it:* The clearest bridge between theory and practice; everything you need to use
ICA correctly in one paper.

---

**2. Hyvärinen, A., & Oja, E. (1997). A fast fixed-point algorithm for independent component analysis.**
*Neural Computation*, 9(7), 1483–1492.
DOI: [10.1162/neco.1997.9.7.1483](https://doi.org/10.1162/neco.1997.9.7.1483)
MIT Press: https://direct.mit.edu/neco/article/9/7/1483/6120/

**Annotation:** The original FastICA paper. Derives the fixed-point update from first
principles as a Newton step in unit-vector space. Shows cubic convergence. Introduces the
contrast function framework. Short (10 pages), dense, beautiful. Reading this makes the
sklearn API completely transparent.
*Why read it:* After this paper, every hyperparameter in `FastICA()` makes geometric sense.

---

### READ NEXT

**3. Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution.**
*Neural Computation*, 7(6), 1129–1159.
DOI: [10.1162/neco.1995.7.6.1129](https://doi.org/10.1162/neco.1995.7.6.1129)
ResearchGate: https://www.researchgate.net/publication/15614030

**Annotation:** The founding paper. Connects maximum information transmission through
nonlinear networks to blind source separation. The InfoMax algorithm and learning rule are
derived here. Historical reading — shows how a neuroscience question accidentally solved
a signal processing problem. Required for understanding why ICA and maximum entropy are
equivalent, and why the `fun` parameter in FastICA encodes source distribution assumptions.
*Why read it:* The intellectual origin story. Essential for deep understanding of the InfoMax connection.

---

**4. Hyvärinen, A. (1999). Survey on independent component analysis.**
*Neural Computing Surveys*, 2, 94–128.
Aalto repository: https://research.aalto.fi/en/publications/survey-on-independent-component-analysis/

**Annotation:** The most comprehensive theoretical treatment of ICA, written when the field
was consolidating. Proves the equivalence of InfoMax, maximum likelihood, mutual information
minimization, and negentropy maximization. Covers all major algorithms (FastICA, InfoMax,
JADE, SOBI) and their relationships. Contains the identifiability theorem with full proof.
*Why read it:* If you need to explain WHY all the different ICA formulations give the same answer.

---

**5. Cardoso, J.-F., & Souloumiac, A. (1993). Blind beamforming for non-Gaussian signals.**
*IEE Proceedings F*, 140(6), 362–370.

**Annotation:** Introduces the JADE algorithm — joint approximate diagonalization of
fourth-order cumulant matrices via Jacobi sweeps. JADE is theoretically cleaner than
FastICA (no contrast function choice, guaranteed convergence). Understanding JADE illuminates
the cumulant-based view of ICA that complements the negentropy view.
*Why read it:* Essential if you work with low-sample regimes or need guaranteed convergence.

---

### DEEP MASTERY

**6. Ablin, P., Cardoso, J.-F., & Gramfort, A. (2018). Faster ICA under orthogonal constraint.**
*ICASSP 2018*. arXiv: 1611.05267.
GitHub: https://github.com/pierreablin/picard

**Annotation:** Introduces Picard-O, which reframes ICA as preconditioned L-BFGS on the
Stiefel manifold. Demonstrates that FastICA's Hessian approximation degrades on real data,
explaining empirically observed slow convergence. Shows 5–10× speedup on real EEG datasets.
The modern practitioner's algorithm for real-world ICA.
*Why read it:* If you use ICA on real data and care about computational efficiency.

---

**7. Comon, P. (1994). Independent component analysis, a new concept?**
*Signal Processing*, 36(3), 287–314.
DOI: 10.1016/0165-1684(94)90029-9

**Annotation:** The paper that gave ICA its name and proved the identifiability theorem
rigorously. Comon formally defined statistical independence for the ICA context and proved
that at most one Gaussian source is allowable. The mathematical foundation without which
everything else is heuristic.
*Why read it:* For the formal proof of why Gaussian sources cannot be separated.

---

**8. Bartlett, M. S., Movellan, J. R., & Sejnowski, T. J. (2002). Face recognition by independent component analysis.**
*IEEE Transactions on Neural Networks*, 13(6), 1450–1464.

**Annotation:** Classic paper demonstrating ICA's superiority over PCA eigenfaces for face
recognition. Introduces two ICA architectures (statistical independence of features vs.
of images). Shows ICA basis functions are sparse and localized, matching V1 receptive fields.
Bridges ICA theory to computer vision practice.
*Why read it:* The definitive demonstration of ICA for image representation.

---

**9. Beckmann, C. F., & Smith, S. M. (2004). Probabilistic independent component analysis for functional magnetic resonance imaging.**
*IEEE Transactions on Medical Imaging*, 23(2), 137–152.

**Annotation:** The paper that made ICA the standard method for fMRI resting-state analysis.
Introduces MELODIC — the FSL ICA tool used worldwide. Addresses the noise model for fMRI,
order selection for ICA components, and the statistical testing of spatial maps.
*Why read it:* Essential if you work with neuroimaging data.

---

## §13 — Resources & Further Reading

### Books

**Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component Analysis*. Wiley.**
The definitive textbook. Chapter 6 (negentropy), Chapter 7 (FastICA), Chapter 8 (convergence
proofs), and Chapter 17 (applications) are the most relevant. Dense but complete.
Read Chapters 1–3 first for foundations.

**Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.**
Chapter 12.4 covers ICA within the probabilistic latent variable model framework.
The perspective is different from Hyvärinen — ICA as inference in a non-Gaussian latent
variable model. Excellent for connecting ICA to the broader probabilistic ML framework.

**Murphy, K. P. (2022). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.**
Chapter 32 covers ICA, blind deconvolution, and topic models within a unified framework.
Available free at https://probml.github.io/pml-book/book2.html

---

### Online Resources

**sklearn.decomposition.FastICA — Official Documentation**
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
The canonical API reference. The User Guide section on signal decomposition is worth reading
in full: https://scikit-learn.org/stable/modules/decomposition.html#ica

**MNE-Python ICA Tutorial**
https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
The best practical tutorial for ICA in neuroscience. Step-by-step EEG artifact removal with
code. The recommended workflow for any EEG/MEG practitioner.

**Picard Package Documentation**
https://github.com/pierreablin/picard
https://pierre.ablin.io/
Short, clear README. See the benchmark notebooks comparing Picard vs FastICA on real data.

**sklearn Faces Decomposition Example**
https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html
Side-by-side comparison of PCA, ICA, NMF, and other methods on the Olivetti faces dataset.
Extremely useful for building intuition about what each method extracts.

---

### Tutorials and Blog Posts

**Aapo Hyvärinen's ICA tutorial pages (Aalto University)**
https://research.aalto.fi/en/persons/aapo-hyvarinen
The original author's course materials and slides are publicly available. The lecture slides
for his "Advanced Machine Learning" course (Helsinki) are particularly clear.

**EEGLAB Wiki — ICA Overview**
https://eeglab.org/tutorials/06_RejectArtifacts/RunICA.html
Comprehensive practical guide to ICA for EEG. Explains the EEGLAB workflow, component
labelling, and artifact removal with many figures. Essential for neuroscience practitioners.

**Towards Data Science — ICA Explained**
Multiple high-quality articles on TDS explain ICA from different angles (information theory,
geometry, practical sklearn). Search "ICA independent component analysis" on TDS.

---

### Package Documentation Links

| Package | Documentation URL |
|---|---|
| `sklearn.FastICA` | https://scikit-learn.org/stable/modules/decomposition.html#ica |
| `python-picard` | https://github.com/pierreablin/picard |
| `mne.preprocessing.ICA` | https://mne.tools/stable/generated/mne.preprocessing.ICA.html |
| `nilearn CanICA` | https://nilearn.github.io/stable/modules/generated/nilearn.decomposition.CanICA.html |
| `optuna` | https://optuna.readthedocs.io/ |

---

*This chapter was authored for the Dimensionality Reduction Masterclass following the course
bible specification. All Python code uses sklearn ~1.5.x syntax. The `picard` sklearn-compatible
`FastICA` wrapper (`from picard import FastICA`) should be verified against the current
python-picard release, as the API may differ across minor versions.*

