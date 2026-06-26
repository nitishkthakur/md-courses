# Non-negative Matrix Factorization: The Complete Masterclass

> **Why this algorithm matters:** In 1999, Daniel Lee and Sebastian Seung published a three-page
> paper in *Nature* that changed how we think about representation learning. They showed that when
> you constrain a matrix factorization to use only non-negative values, something remarkable
> happens: the algorithm stops learning holistic blends and starts learning *parts*. Apply PCA to
> face images and you get "eigenfaces" — ghostly, photographic negatives of the average human
> face. Apply NMF to the same images and you get eyes, a nose, a mouth, cheekbones — actual
> building blocks that combine to form a face. That single insight — **non-negativity enforces
> parts-based decomposition** — has made NMF the algorithm of choice in cancer genomics (where
> it discovers mutational signatures that identify UV damage, smoking, and BRCA mutations),
> audio source separation (where it unmixes instruments from a recording), single-cell RNA
> sequencing (where it finds gene programs defining cell identity), and hyperspectral remote
> sensing (where it estimates the composition of every pixel on Earth's surface from satellite
> data). This chapter will take you from the 1999 Nature paper all the way to deep NMF, robust
> NMF, and MiniBatchNMF — and will make you the practitioner who knows not just *how* to run NMF
> but *why* it works, *when* to trust it, and *when* to walk away.

---

## §0 — Python Ecosystem & Package Guide

Before the mathematics, you need to know the landscape. NMF has a clear hierarchy of Python
implementations, and choosing the right one for your domain matters more than for most algorithms.

### Complete Package Table

| Package | Import | Version (mid-2026) | What It Provides | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.decomposition import NMF, MiniBatchNMF` | ~1.5.x | Production-quality NMF with CD and MU solvers, NNDSVD initialization, MiniBatchNMF for out-of-core data; full sklearn pipeline compatibility | BSD-3 |
| `nimfa` | `import nimfa` | 1.4.0 (stable, low maintenance since 2021) | 17+ NMF algorithms (LSNMF, NSNMF, PMF, SNMF, BD, PSMF, and more); consensus NMF over multiple runs; built-in stability metrics (cophenetic correlation, dispersion); sparse matrix support | BSD |
| `torchnmf` | `from torchnmf.nmf import NMF` | ~0.3.4 | GPU-accelerated NMF via PyTorch; Convolutional NMF (CNMF); PLCA variants; all beta-divergences (Frobenius, KL, IS); multi-GPU batched NMF | MIT |
| `robust-nmf` | `from robustnmf.nmf import robust_nmf` | research release | Robust NMF: decomposes X = WH + E where E is a sparse outlier matrix; PyTorch GPU + NumPy CPU; handles corrupted data (occlusions, bad pixels) | MIT |
| `GeneNMF` | R: `install.packages('GeneNMF')` | R ecosystem | Consensus NMF for single-cell omics on Seurat objects; multi-sample gene program discovery; Python users use sklearn/nimfa on scRNA-seq count matrices | GPL |

> ⚠️ **Pitfall:** `nimfa` has low maintenance activity since 2021 but remains functional with
> Python 3.x and modern NumPy. If you hit installation issues, pin `numpy<2.0`. For production
> pipelines, prefer sklearn; use nimfa only when you need consensus NMF or a specialized variant
> like NSNMF that sklearn does not implement.

> 🔧 **In practice:** `torchnmf` uses the **opposite naming convention** from sklearn: it calls
> the sample encoding matrix `H` and the dictionary matrix `W`. This is the audio/signal
> processing convention (originating from the Virtanen 2007 notation). When reading torchnmf
> output, remember: `model.H` corresponds to sklearn's `W` (per-sample activations) and
> `model.W` corresponds to sklearn's `model.components_`. Flag this in code comments or you
> will confuse yourself at 2am.

### GPU-Accelerated Options

If your data matrix is large — tens of thousands of rows and tens of thousands of features (e.g.,
hyperspectral cubes, large-scale genomics, audio spectrograms) — sklearn's CPU implementation
will bottleneck. Your options:

1. **`torchnmf`** — The most practical GPU option. Works with CUDA and MPS (Apple Silicon). Install
   `pip install torchnmf` and move tensors to GPU before fitting.
2. **Custom PyTorch autograd NMF** — Write the NMF objective in PyTorch, use `torch.clamp(min=0)`
   to enforce non-negativity, and train with Adam. Not as efficient as specialized multiplicative
   updates but trivially extensible to any loss and constraint.
3. **cuML NMF** (RAPIDS) — NVIDIA's RAPIDS ecosystem includes GPU NMF. Install via conda
   (`conda install -c rapidsai cuml`). API is close to sklearn. # verify in current docs

### Streaming / Out-of-Core Options

For data that does not fit in RAM:

- **`sklearn.decomposition.MiniBatchNMF`** — Added in sklearn v1.1. Processes data in mini-batches
  with a forget factor. Supports `partial_fit()` for true streaming.
- **Custom online NMF** — Mairal et al. (2010) described online dictionary learning; you can
  implement this with sklearn's MiniBatchNMF as the backbone.

### Top Picks — Recommendation Table

| Package | Best For | Internal Solver | Modern / Legacy |
|---|---|---|---|
| **`sklearn.NMF`** | 95% of production use cases; dense or sparse TF-IDF matrices; sklearn pipelines | CD (Frobenius) or MU (KL, IS) | **Modern — default first choice** |
| **`sklearn.MiniBatchNMF`** | Datasets > available RAM; true streaming pipelines | Mini-batch MU with forget factor | Modern |
| **`nimfa`** | Research requiring consensus NMF; stability analysis; genomics workflows | 17+ solvers including ANLS, NSNMF | Modern for research |
| **`torchnmf`** | GPU-accelerated NMF; audio/hyperspectral (large matrices); convolutional NMF | GPU MU (all beta-divergences) | Modern for GPU |
| **`robust-nmf`** | Corrupted data with gross outliers; hyperspectral bad pixels; occluded faces | Alternating L2,1 minimization | Specialized / research |

### The Same Fit Across Top Packages

Here is the same factorization task — decompose a non-negative matrix into 20 components —
expressed in each of the top packages. This direct comparison shows you the API differences
so you can switch between them confidently.

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Build a shared TF-IDF matrix (used throughout this chapter)
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=5)
X = vectorizer.fit_transform(newsgroups.data)   # scipy sparse, (11314, 5000), non-negative
X_dense = X.toarray()                            # dense copy for packages requiring it
feature_names = vectorizer.get_feature_names_out()

N_COMPONENTS = 20

# ── Option 1: sklearn.NMF — the production workhorse ──────────────────────────
from sklearn.decomposition import NMF

nmf_sklearn = NMF(
    n_components=N_COMPONENTS,
    init='nndsvda',       # Best general-purpose initialization (sklearn default)
    solver='cd',          # Coordinate descent — fastest, requires Frobenius loss
    beta_loss='frobenius',
    max_iter=500,
    random_state=42,
    tol=1e-4,
)
W_sklearn = nmf_sklearn.fit_transform(X)          # (11314, 20) — document-topic matrix
H_sklearn = nmf_sklearn.components_               # (20, 5000) — topic-word matrix
print(f"[sklearn] err={nmf_sklearn.reconstruction_err_:.4f}, iters={nmf_sklearn.n_iter_}")

# Top words per topic
for i, topic in enumerate(H_sklearn[:3]):         # show first 3 topics
    top = [feature_names[j] for j in topic.argsort()[:-11:-1]]
    print(f"  Topic {i}: {', '.join(top)}")

# ── Option 2: sklearn.MiniBatchNMF — out-of-core / streaming ──────────────────
from sklearn.decomposition import MiniBatchNMF

nmf_mb = MiniBatchNMF(
    n_components=N_COMPONENTS,
    init='nndsvda',
    batch_size=1024,
    forget_factor=1.0,    # Finite dataset: weight all batches equally
    max_iter=200,
    random_state=42,
)
W_mb = nmf_mb.fit_transform(X)
print(f"[MiniBatch] reconstruction_err={nmf_mb.reconstruction_err_:.4f}")

# ── Option 3: nimfa — research-grade, consensus analysis ──────────────────────
import nimfa                          # pip install nimfa

# nimfa expects (n_features, n_samples) — transpose of sklearn convention!
V = X_dense.T                         # (5000, 11314)

nmf_nimfa = nimfa.Nmf(
    V,
    rank=N_COMPONENTS,
    seed='nndsvd',        # NNDSVD initialization
    max_iter=200,
    update='euclidean',   # Frobenius / euclidean loss
)
fit = nmf_nimfa()
W_nimfa = np.array(fit.basis()).T     # (11314, 20) after transposing back
H_nimfa = np.array(fit.coef())       # (20, 5000)
print(f"[nimfa] residuals={fit.distance(metric='euclidean'):.4f}")

# ── Option 4: torchnmf — GPU-accelerated ──────────────────────────────────────
import torch
from torchnmf.nmf import NMF as TorchNMF   # pip install torchnmf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_tensor = torch.tensor(X_dense, dtype=torch.float32).to(device)

tnmf = TorchNMF(X_tensor.shape, rank=N_COMPONENTS).to(device)
tnmf.fit(X_tensor, beta=2)             # beta=2 → Frobenius loss

# NOTE: torchnmf uses INVERTED naming vs sklearn
W_torch = tnmf.H.cpu().detach().numpy()   # activations (sklearn's W)
H_torch = tnmf.W.cpu().detach().numpy()   # dictionary (sklearn's components_)
print(f"[torchnmf] run on device={device}")
```

> 🏆 **Best practice:** For text, genomics, or image data in a production sklearn pipeline,
> start with `sklearn.NMF(init='nndsvda', solver='cd')`. Switch to `MiniBatchNMF` when data
> exceeds ~500MB uncompressed. Switch to `torchnmf` when your matrix has more than ~50K features
> (e.g., full hyperspectral cube unfolded) and a GPU is available. Use `nimfa` only when you
> need stability analysis (cophenetic correlation across multiple random runs) — it provides
> this out of the box and sklearn does not.

---

## §1 — The Origin: The Paper That Started It All

> 📜 **Origin/Citation:** D. D. Lee and H. S. Seung, "Learning the parts of objects by
> non-negative matrix factorization," *Nature*, vol. 401, no. 6755, pp. 788–791, Oct. 1999.
> DOI: [10.1038/44565](https://doi.org/10.1038/44565). Three pages. One figure. Cited over
> 20,000 times. Read this paper — it is one of the most beautifully written ML papers ever
> published.

### What Came Before: The Limitation of Prior Art

To understand what Lee and Seung did, you first need to understand why existing matrix
factorization methods were unsatisfying for many scientific problems.

By 1999, the dominant approaches to decomposing data matrices were:

- **PCA / SVD** — Finds directions of maximum variance. Components are orthogonal and globally
  defined. The issue: SVD components can have both positive and negative entries. When you apply
  PCA to face images, your components look like photographic negatives of themselves — the famous
  "eigenfaces." They capture global variation, not local parts.
- **Vector Quantization (VQ)** — Assigns each data point to exactly one prototype. Extremely
  sparse (each sample uses *one* component) but rigid — no room for partial membership.
- **ICA** — Finds statistically independent components. Also allows negative values, and the
  independence constraint leads to very different representations from parts-based ones.

The deeper problem was conceptual. These methods were all asking: *what linear combination of
basis vectors reconstructs this data point?* But the basis vectors themselves could be
non-physical — mixtures of positive and negative weights with no clear semantic meaning.

For many scientific domains, the data has a clear **additive structure**: a face is made of
parts, a document is made of topics, a tumor's mutations are caused by separate mutagenic
processes. In all of these, you want an algorithm that says "this sample is composed of part A
plus part B plus part C" — not "this sample is 3.2 standard deviations in direction 1 minus
1.7 in direction 2."

### The True Precursor: Paatero & Tapper (1994)

Before Lee and Seung, Pentti Paatero and Unto Tapper published a 1994 paper in *Environmetrics*
describing "Positive Matrix Factorization" (PMF) — a method requiring all matrix entries to be
non-negative, applied to air pollution source apportionment. They wanted to decompose a matrix
of pollution measurements into contributions from different emission sources (traffic, industry,
sea salt). Since contributions cannot be negative (a source cannot remove pollution), non-negativity
was physically motivated.

PMF worked, but it remained almost entirely unknown outside environmental science and
chemometrics. It was Lee and Seung's 1999 *Nature* paper that brought the non-negative
factorization idea to the mainstream machine learning community — backed by three compelling
experiments and a crisp conceptual explanation that resonated far beyond air quality research.

### The Core Insight: Non-negativity Enforces Parts

The Lee-Seung insight is elegant and worth sitting with.

Given a data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}_+$ (all entries non-negative),
NMF seeks two non-negative matrices:

$$\mathbf{X} \approx \mathbf{W} \mathbf{H}$$

where $\mathbf{W} \in \mathbb{R}^{n \times k}_+$ and $\mathbf{H} \in \mathbb{R}^{k \times p}_+$,
with $k \ll \min(n, p)$.

Every entry of $\mathbf{X}$ is approximated as:

$$x_{ij} \approx \sum_{r=1}^{k} w_{ir} h_{rj}$$

Because $w_{ir} \geq 0$ and $h_{rj} \geq 0$, this is a **purely additive combination**. You
can only add parts together, never subtract them. This single constraint — no negatives — is
what forces the algorithm to find interpretable parts.

> 💡 **Intuition:** Think of building a LEGO model. You can only add bricks, never remove them.
> If you want to represent a face under this constraint, you must find bricks (components) that
> are recognizable sub-parts of faces — an eye module, a nose module, a forehead module — because
> there is no other way to combine them into something face-like. PCA, without this constraint,
> can use subtraction: it defines "what makes a face different from the average," which produces
> ghost-like images. NMF must define the face as a sum of positive parts.

### Three Experiments from the Paper

Lee and Seung demonstrated their insight with three experiments:

**Experiment 1: Face Images (CBCL Dataset)**
Applied PCA, VQ, and NMF to 2,429 face images (19×19 pixels each). PCA produced eigenfaces —
holistic, global features that look like entire face images. VQ produced 49 whole face prototypes.
NMF produced eyes, noses, mouths, and cheekbones — actual facial parts. A person's face could
then be represented as a sum of these parts, each with a non-negative coefficient.

**Experiment 2: Text Data**
Applied NMF to a corpus of news articles. Each article was represented as a bag-of-words. NMF
discovered topics — semantically coherent word clusters — without any topic labels. This predated
LDA (2003) and showed that NMF was a viable topic model.

**Experiment 3: Speech**
Applied NMF to spectrogram data from speech recordings. NMF learned spectral templates
corresponding to phonemes.

### The Original Algorithm: Multiplicative Update Rules

Lee and Seung's companion algorithms paper (NIPS 2001) derived the now-canonical **multiplicative
update rules** for NMF. These rules are derived by finding a gradient descent update that is
guaranteed to be non-negative.

The objective for **Frobenius-norm NMF** (least squares) is:

$$\mathcal{L}_F = \|\mathbf{X} - \mathbf{W}\mathbf{H}\|_F^2 = \sum_{i,j} (x_{ij} - [\mathbf{W}\mathbf{H}]_{ij})^2$$

The multiplicative update rules are:

$$\mathbf{H} \leftarrow \mathbf{H} \odot \frac{\mathbf{W}^\top \mathbf{X}}{\mathbf{W}^\top \mathbf{W} \mathbf{H}}$$

$$\mathbf{W} \leftarrow \mathbf{W} \odot \frac{\mathbf{X} \mathbf{H}^\top}{\mathbf{W} \mathbf{H} \mathbf{H}^\top}$$

where $\odot$ denotes element-wise multiplication and division is also element-wise.

**Why do these rules preserve non-negativity?** If $\mathbf{W}$ and $\mathbf{H}$ are initialized
with non-negative values, the numerator and denominator of each update are both non-negative
(they are matrix products of non-negative matrices), so the ratio is non-negative, and the
element-wise product with the current value stays non-negative.

**Convergence:** Lee and Seung proved that these updates are *monotonically non-increasing* for
the objective — each step either decreases $\mathcal{L}_F$ or leaves it unchanged. The proof
uses an **auxiliary function** technique analogous to the convergence proof of the EM algorithm.
However, convergence to a stationary point (not just a limit of the objective) was not proven;
this is a theoretical gap that motivated later work (coordinate descent, ANLS).

For **KL-divergence NMF**, the objective is:

$$\mathcal{L}_{KL} = \sum_{i,j} \left( x_{ij} \log \frac{x_{ij}}{[\mathbf{W}\mathbf{H}]_{ij}} - x_{ij} + [\mathbf{W}\mathbf{H}]_{ij} \right)$$

With KL-divergence update rules:

$$\mathbf{H} \leftarrow \mathbf{H} \odot \frac{\mathbf{W}^\top (\mathbf{X} \oslash \mathbf{W}\mathbf{H})}{\mathbf{1}^\top \mathbf{W}}$$

$$\mathbf{W} \leftarrow \mathbf{W} \odot \frac{(\mathbf{X} \oslash \mathbf{W}\mathbf{H}) \mathbf{H}^\top}{\mathbf{H} \mathbf{1}}$$

where $\oslash$ is element-wise division. This variant is appropriate for count data, where the
Poisson noise model motivates the KL divergence.

> 📜 **Origin/Citation:** The algorithms paper is: D. D. Lee and H. S. Seung, "Algorithms for
> Non-negative Matrix Factorization," *Advances in Neural Information Processing Systems 13
> (NIPS 2000)*, pp. 556–562, 2001.
> URL: https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization

---

## §2 — The Algorithm, Deeply Explained

### Problem Setup and Notation

Let $\mathbf{X} \in \mathbb{R}^{n \times p}_+$ be your data matrix with $n$ samples and $p$
features. Every entry must be non-negative: $x_{ij} \geq 0$ for all $i, j$. You choose a rank
$k$ (the number of components, `n_components`). NMF finds:

$$\mathbf{X} \approx \mathbf{W} \mathbf{H}$$

where:
- $\mathbf{W} \in \mathbb{R}^{n \times k}_+$ — the **encoding matrix** (sklearn calls this the
  output of `fit_transform(X)`). Row $i$ tells you how much sample $i$ uses each component.
- $\mathbf{H} \in \mathbb{R}^{k \times p}_+$ — the **component matrix** (stored as
  `model.components_` in sklearn). Row $r$ is the $r$-th latent component — a non-negative
  vector in the feature space.

The approximation quality is measured by a **beta-divergence** loss:

$$\mathcal{L}_\beta(\mathbf{X} \| \mathbf{W}\mathbf{H}) = \sum_{i,j} d_\beta(x_{ij}, [\mathbf{W}\mathbf{H}]_{ij})$$

The scalar beta-divergence $d_\beta(x, y)$ is defined as:

$$d_\beta(x, y) = \begin{cases}
\frac{1}{\beta(\beta-1)}\left(x^\beta + (\beta-1)y^\beta - \beta x y^{\beta-1}\right) & \beta \notin \{0,1\} \\
x \log \frac{x}{y} - x + y & \beta = 1 \text{ (KL divergence)} \\
\frac{x}{y} - \log \frac{x}{y} - 1 & \beta = 0 \text{ (Itakura-Saito)}
\end{cases}$$

The three canonical choices:
- **$\beta = 2$ (Frobenius):** $d_2(x,y) = \frac{1}{2}(x-y)^2$. Minimizes squared error. Good
  for continuous non-negative data with Gaussian-like noise. This is the sklearn default.
- **$\beta = 1$ (KL divergence):** Appropriate when entries are counts (bag-of-words, read
  counts, mutation counts). Assumes Poisson noise. Required with `solver='mu'`.
- **$\beta = 0$ (Itakura-Saito):** Scale-invariant. A prediction that is off by factor 2 is
  penalized equally whether the true value is 1 or 1000. This is why it is the right choice for
  audio spectrograms, where energy can span many orders of magnitude.

### The Non-convexity Problem

NMF is **non-convex**. The joint optimization over $(\mathbf{W}, \mathbf{H})$ has many local
minima. However, the subproblem of optimizing $\mathbf{W}$ alone with $\mathbf{H}$ fixed (or
vice versa) is convex — it reduces to a non-negative least squares problem with a unique global
minimum. This structure motivates alternating optimization algorithms.

> 💡 **Intuition:** Imagine two people are blindfolded and each holding one end of a rubber band
> stretched over a bumpy landscape. Each person, in turn, walks downhill while the other stays
> still. They alternate. Each person's move is a convex step (they're walking down a smooth bowl),
> but the joint landscape they're navigating is bumpy. This is why initialization matters: where
> you start on the bumpy joint landscape determines which valley you fall into.

### Scale Ambiguity

NMF has a fundamental **scaling ambiguity**: for any positive diagonal matrix $\mathbf{D}$,
$(\mathbf{W}\mathbf{D})(\mathbf{D}^{-1}\mathbf{H}) = \mathbf{W}\mathbf{H}$. So you can always
rescale columns of $\mathbf{W}$ and corresponding rows of $\mathbf{H}$ without changing the
product. Similarly, you can permute the component order.

The practical consequence: **NMF components have no natural ordering** (unlike PCA's principal
components, which are ordered by variance explained), and their individual scales are arbitrary.
Don't read too much into which topic is "number 1."

### Solver 1: Coordinate Descent (sklearn default)

**Algorithm (Chou & Lin 2012):**

```
Initialize W, H ≥ 0 (via NNDSVD or random)
Repeat until convergence:
    For each column c of W:
        Solve: min_{w_c ≥ 0} ||X - W H||_F²  (NNLS sub-problem, H fixed)
        Update w_c with the solution
    For each row r of H:
        Solve: min_{h_r ≥ 0} ||X - W H||_F²  (NNLS sub-problem, W fixed)
        Update h_r with the solution
```

Each sub-problem is a **Non-Negative Least Squares** (NNLS) problem, which is convex and has
a unique solution. The coordinate descent algorithm solves it efficiently using an active-set
method.

**Key advantages of CD:**
1. **Convergence guarantee:** CD converges to a stationary point (local minimum or saddle point)
   — a stronger guarantee than the multiplicative update rules.
2. **Speed:** 5–24× faster than multiplicative updates in wall-clock time (benchmarks from
   Kim & Park 2008).
3. **Zero handling:** Unlike MU, CD can *move entries away from zero*, because the NNLS solver
   decides the optimal value including the possibility of setting entries back above zero.

**Limitation:** Only works with Frobenius loss. If you need KL or IS divergence, you must use
the MU solver.

### Solver 2: Multiplicative Updates (required for KL/IS)

**Algorithm (Lee & Seung 2001):**

```
Initialize W, H ≥ 0
Repeat until convergence:
    H ← H ⊙ (W^T X) / (W^T W H)        # element-wise multiply/divide
    W ← W ⊙ (X H^T) / (W H H^T)
```

The fraction in each update is element-wise. The multiplicative structure guarantees that if
any entry of $\mathbf{W}$ or $\mathbf{H}$ starts non-negative, it stays non-negative.

**The zero-locking problem:** If any entry of $\mathbf{W}$ or $\mathbf{H}$ is initialized to
exactly zero, it will **remain zero forever** — the multiplicative update multiplies by a factor
that is itself a ratio involving the current value, so it cannot resurrect a zero. This is why
`init='nndsvd'` (which can produce many exact zeros) is problematic with the MU solver. Use
`'nndsvda'` or `'random'` instead when `solver='mu'`.

### Initialization: NNDSVD

Boutsidis & Gallopoulos (2008) showed that a clever SVD-based initialization dramatically
improves NMF solutions. The **NNDSVD** (Non-Negative Double SVD) algorithm works as follows:

1. Compute the truncated SVD of $\mathbf{X}$: $\mathbf{X} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$
   (keeping $k$ singular vectors)
2. For each singular triplet $(\sigma_r, \mathbf{u}_r, \mathbf{v}_r)$, split into positive
   and negative parts:
   - $\mathbf{u}_r^+ = \max(\mathbf{u}_r, 0)$, $\mathbf{u}_r^- = \max(-\mathbf{u}_r, 0)$
   - $\mathbf{v}_r^+ = \max(\mathbf{v}_r, 0)$, $\mathbf{v}_r^- = \max(-\mathbf{v}_r, 0)$
3. Set the $r$-th component using whichever pair $(u^+, v^+)$ or $(u^-, v^-)$ has larger
   combined norm.

**Three NNDSVD variants in sklearn:**
- `'nndsvd'`: Use zeros where the SVD component is negative. Best for sparse data. Can cause
  zero-locking with MU solver.
- `'nndsvda'` (sklearn default for $k \leq \min(n,p)$): Replace zeros with the mean of
  $\mathbf{X}$. Avoids zero-locking. **Best general-purpose default.**
- `'nndsvdar'`: Replace zeros with small random values. A compromise.

> ⚠️ **Pitfall:** NNDSVD requires computing a full truncated SVD of $\mathbf{X}$, which for a
> matrix with millions of rows can exhaust memory or be very slow. For very large matrices
> (n > 100K, p > 10K), use `init='random'` or switch to `MiniBatchNMF`.

### Computational Complexity

- **Fitting (each iteration):** $O(nkp)$ — dominated by the matrix multiplications
  $\mathbf{W}^\top \mathbf{X}$ (cost $nkp$) and $\mathbf{W}\mathbf{H}\mathbf{H}^\top$
  (cost $k^2 p + nk^2$). For sparse $\mathbf{X}$ with density $\rho$, the cost is
  $O(\rho \cdot n \cdot k \cdot p)$ — much cheaper for TF-IDF matrices.
- **Fitting (total):** $O(T \cdot nkp)$ where $T$ is the number of iterations (typically
  100–500).
- **Transform (new data):** $O(n_\text{new} \cdot k \cdot p)$ — holds $\mathbf{H}$ fixed and
  solves for the encoding of new samples via NNLS.
- **NNDSVD initialization:** $O(np \min(n,p))$ — can be significant for large matrices.
- **Memory:** $O(nk + kp)$ for $\mathbf{W}$ and $\mathbf{H}$; $O(np)$ for dense $\mathbf{X}$.

### Implicit Assumptions and Where They Break

NMF makes several implicit assumptions about your data:

1. **Non-negativity:** All entries of $\mathbf{X}$ must be $\geq 0$. Negative values cause
   a hard error. If your data has negative values (log-ratios, mean-subtracted data, signed
   residuals), NMF is not appropriate without modification.

2. **Additive structure:** The data is actually composed of parts that add together. This is
   true for spectra, counts, images of opaque objects, and similar domains. It is *not* true for
   data where features interact multiplicatively or where subtraction is physically meaningful.

3. **Poisson or Gaussian noise** (depending on beta-divergence choice): KL divergence assumes
   Poisson counting noise. Frobenius assumes equal-variance Gaussian noise. If your data has
   heavy-tailed noise or outliers, consider Robust NMF.

4. **Low effective rank:** The assumption $k \ll \min(n,p)$ must hold — the data must be
   well-described by a small number of parts. If the true rank is high (data is intrinsically
   high-dimensional), NMF will have poor reconstruction and hard-to-interpret components.

> ⚠️ **Pitfall:** The most common mistake is applying NMF to data that has been
> **mean-centered or standardized** (as you would before PCA). Centering introduces negative
> values and violates the non-negativity assumption. If you must normalize, use
> min-max scaling to $[0, 1]$ or divide by the row/column sums — operations that preserve
> non-negativity.

---

## §3 — The Full Evolution of NMF

Standard NMF is powerful but has known failure modes: non-uniqueness, sensitivity to
initialization, breakdown under outliers, inability to handle mixed-sign data, and scalability
limits. The research community has addressed each of these with targeted variants.

### 3.1 Sparse NMF — Hoyer (2004)

> 📜 **Origin/Citation:** P. O. Hoyer, "Non-negative matrix factorization with sparseness
> constraints," *Journal of Machine Learning Research*, vol. 5, pp. 1457–1469, 2004.
> URL: https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf

**Problem it solves:** Standard NMF produces components and encodings that may be dense — every
document might load on every topic, every pixel might use every component. This can hurt both
interpretability and uniqueness.

**Key idea:** Add an explicit sparsity constraint using the **Hoyer sparseness measure**:

$$\text{sparseness}(\mathbf{x}) = \frac{\sqrt{n} - \|\mathbf{x}\|_1 / \|\mathbf{x}\|_2}{\sqrt{n} - 1}$$

This measure ranges from 0 (all entries equal — maximum density) to 1 (only one non-zero entry
— maximum sparsity). It is invariant to scaling of $\mathbf{x}$, which makes it a principled
way to measure sparsity independently of component magnitude.

**Algorithm:** Projected gradient descent. Take a gradient step, then project the result onto
the set of non-negative vectors with the desired L1/L2 norm ratio. This projection has a
closed-form solution.

**When to prefer:** When you want maximally interpretable topics (each document uses only 1-2
topics), or when uniqueness is important (sparse solutions are more likely to be unique).

**In sklearn:** Exact Hoyer sparsity constraints are not implemented. The approximation is
L1 regularization on H or W:

```python
from sklearn.decomposition import NMF

# Promote sparsity in encodings (W) via L1 penalty on H — sklearn convention
model = NMF(
    n_components=20,
    init='nndsvda',
    solver='cd',
    alpha_W=0.0,       # Don't regularize W (the dictionary)
    alpha_H=0.5,       # Regularize H (the encodings — promote sparsity)
    l1_ratio=1.0,      # Pure L1 → true sparsity (not just small values)
    max_iter=500,
)
```

### 3.2 Robust NMF — Févotte & Dobigeon (2014)

**Problem it solves:** Standard NMF assumes noise is Gaussian (Frobenius) or Poisson (KL).
Real data often contains **gross corruptions**: a bad scan line in a hyperspectral image, an
occluded region in a face image, a corrupted audio frame. Standard NMF tries to fit these
outliers and distorts all components in the process.

**Key idea:** Decompose $\mathbf{X}$ into a low-rank non-negative part plus an explicit outlier
matrix:

$$\mathbf{X} = \mathbf{W}\mathbf{H} + \mathbf{E}, \quad \mathbf{W}, \mathbf{H} \geq 0$$

The outlier matrix $\mathbf{E}$ is penalized with an $\ell_{2,1}$ norm (sum of column
$\ell_2$ norms), which encourages column-sparse corruptions (entire pixels or time frames are
outliers, not individual entries). Alternatively, an $\ell_1$ penalty on $\mathbf{E}$ handles
entry-wise corruptions.

**In Python:** The `robust-nmf` package provides PyTorch (GPU) and NumPy (CPU) implementations.
Use it when you have hyperspectral imagery with bad pixels, face images with occlusion, or any
data with expected gross corruptions.

```python
# pip install robust-nmf  (or clone from https://github.com/neel-dey/robust-nmf)
from robustnmf.nmf import robust_nmf   # verify installation name in current docs

W, H, E, loss = robust_nmf(
    X,                    # non-negative data matrix
    rank=20,              # n_components
    beta=1.5,             # beta-divergence for WH term
    reg_val=0.1,          # regularization on E (outlier penalty strength)
    n_iter=500,
)
```

### 3.3 Online NMF / MiniBatchNMF — Mairal et al. (2010) + sklearn v1.1

**Problem it solves:** For datasets with millions of samples, loading all data into memory and
computing the full matrix products $\mathbf{W}^\top \mathbf{X}$ is infeasible.

**Key idea (Mairal et al. 2010):** At each step, observe a mini-batch $\mathbf{X}_t$, solve for
its encoding $\mathbf{W}_t$ with current $\mathbf{H}$ fixed, then update $\mathbf{H}$ using a
surrogate function that approximates the full dataset objective using an exponentially
down-weighted history.

The **forget factor** $\rho \in (0, 1]$ controls how quickly old mini-batches are forgotten.
With $\rho = 1$, all batches contribute equally (equivalent to cycling through data). With
$\rho < 1$, older batches decay exponentially — enabling adaptation to concept drift.

**sklearn implementation (`MiniBatchNMF`, added v1.1):**

```python
from sklearn.decomposition import MiniBatchNMF

model = MiniBatchNMF(
    n_components=20,
    init='nndsvda',
    batch_size=1024,
    forget_factor=1.0,     # 1.0 for finite datasets; < 1.0 for streaming
    max_iter=200,
    max_no_improvement=10, # early stopping if no improvement
    random_state=42,
)

# Standard fit on large dataset
W = model.fit_transform(X_large)

# True streaming (new data arrives over time)
model2 = MiniBatchNMF(n_components=20, forget_factor=0.7)
for batch in stream_batches():
    model2.partial_fit(batch)
```

### 3.4 Semi-NMF — Ding et al. (2010)

**Problem it solves:** Many real datasets contain negative values (e.g., mean-subtracted data,
centered word embeddings, log-ratio microarray data). Standard NMF cannot handle these.

**Key idea:** Relax the non-negativity constraint on $\mathbf{W}$ (the encoding matrix), while
keeping $\mathbf{H} \geq 0$ (the dictionary matrix):

$$\mathbf{X} \approx \mathbf{W}\mathbf{H}, \quad \mathbf{H} \geq 0, \quad \mathbf{W} \text{ unrestricted}$$

$\mathbf{X}$ can now contain negative entries. The dictionary $\mathbf{H}$ remains non-negative
and interpretable. The encodings $\mathbf{W}$ indicate how much of each component is present,
but can also indicate "opposition" (negative loading).

**Deep Semi-NMF:** Stack multiple Semi-NMF layers for hierarchical feature learning:
$\mathbf{X} \approx \mathbf{W}_1 \mathbf{H}_1$, $\mathbf{H}_1 \approx \mathbf{W}_2 \mathbf{H}_2$, etc.
Only the final $\mathbf{H}$ level needs to be non-negative. This was an early form of deep
feature learning, predating the dominance of neural networks.

**Not in sklearn.** Requires custom implementation or the research code from Ding et al.

### 3.5 Convex NMF — Ding et al. (2008)

**Key idea:** Constrain the dictionary matrix $\mathbf{H}$ to be in the convex hull of the data:
$\mathbf{H} = \mathbf{G}\mathbf{X}$ where $\mathbf{G} \geq 0$. The components are now non-negative
mixtures of actual data points — they are **archetypes**, not abstract latent vectors.

**Why this matters:** You can always point to a real data point (or convex combination of
real points) that exemplifies each component. This is useful in customer segmentation (each
segment corresponds to a real customer profile, not an abstract mixture) and genomics (each
cell state corresponds to a real cell).

**Connection to archetypal analysis:** Convex NMF is closely related to Archetypal Analysis
(Cutler & Breiman 1994), which finds the corners of a polytope containing most of the data.

### 3.6 Beta-NMF — Févotte & Idier (2011)

> 📜 **Origin/Citation:** C. Févotte and J. Idier, "Algorithms for Nonnegative Matrix
> Factorization with the β-Divergence," *Neural Computation*, vol. 23, no. 9, pp. 2421–2456,
> 2011. DOI: 10.1162/NECO_a_00168

**Key idea:** Unify all three classical NMF objectives ($\beta=2, 1, 0$) and any intermediate
value into a single family of multiplicative update rules. For any $\beta$, the update rules
become:

$$\mathbf{H} \leftarrow \mathbf{H} \odot \frac{\mathbf{W}^\top ([\mathbf{W}\mathbf{H}]^{\beta-2} \odot \mathbf{X})}{\mathbf{W}^\top [\mathbf{W}\mathbf{H}]^{\beta-1}}$$

And symmetrically for $\mathbf{W}$.

**When non-integer beta matters:** Values between 0 and 2 interpolate between IS and Frobenius.
For audio with moderate dynamic range, $\beta \approx 0.5$ can outperform both IS and Frobenius.
In sklearn, you can pass a float for `beta_loss`:

```python
model = NMF(
    n_components=20,
    solver='mu',
    beta_loss=0.5,    # beta-NMF, intermediate between IS and KL
    max_iter=1000,
)
```

### 3.7 Deep NMF

**Key idea:** Stack multiple NMF layers to learn hierarchical, multi-scale parts:

$$\mathbf{X} \approx \mathbf{W}_1 \mathbf{H}_1, \quad \mathbf{H}_1 \approx \mathbf{W}_2 \mathbf{H}_2, \quad \ldots$$

Each layer factorizes the output of the previous layer. All matrices are non-negative, so the
result is a hierarchical parts-based representation: layer 1 learns fine parts (edges, phonemes),
layer 2 learns combinations of those (contours, syllables), and so on.

**Training:** Typically greedy layer-wise pretraining: train layer 1, freeze $\mathbf{H}_1$, use
it as input to layer 2, and so on. Can also be trained end-to-end with PyTorch autograd by
expressing the entire stack as a computational graph with non-negativity enforced by `F.relu`.

**Recent developments:**
- Generalized Discriminative Deep NMF (IJCAI 2023): Adds a discriminative loss at the top layer.
- Deep Approximately Orthogonal NMF: Encourages orthogonal components at each layer for
  disentanglement.
- Sparse Deep NMF (arXiv 1707.09316): Combines hierarchical depth with Hoyer sparsity.

**Not in sklearn.** Implement with PyTorch or nimfa (nimfa supports some stacked variants).

> 🔧 **In practice:** For most production NLP and genomics tasks, the ROI from deep NMF is
> modest compared to the engineering complexity. Start with standard NMF + appropriate
> regularization. Add depth only if you have evidence that your data has multi-scale structure
> and you have the engineering time to validate it.

---

## §4 — Hyperparameters: The Complete Guide

NMF in sklearn has nine user-facing hyperparameters. Understanding each one deeply — not just
its name and type but the mechanism behind it — is what separates practitioners who get useful
factorizations from those who wonder why their topics don't make sense.

### `n_components` — The Most Critical Hyperparameter

**What it controls:** The rank $k$ of the factorization. Directly controls the trade-off between
reconstruction quality (more components → lower error) and interpretability (fewer components →
broader, clearer topics).

**Default:** `None` — resolves to `min(n_samples, n_features)`. This gives perfect
reconstruction (if the rank is sufficient) but zero compression. **Always set this explicitly.**

**The 'auto' option (sklearn ≥ v1.4):** Only useful when passing a custom initialization via
`init='custom'` — it infers `n_components` from the shapes of the provided `W_init` and
`H_init`.

**Tuning methods:**

1. **Elbow on reconstruction error:** Plot error vs. $k$ and look for the elbow.
2. **Cophenetic correlation (stability):** Run NMF multiple times per $k$, build a consensus
   matrix, compute cophenetic correlation. The $k$ that maximizes stability is a good choice.
3. **Cross-validated held-out reconstruction:** Mask 20% of entries, fit on observed, measure
   error on masked. The $k$ that minimizes held-out error without degrading training error
   avoids overfitting.
4. **Domain knowledge:** Often most reliable — if you know there are ~10 topics or ~5 sources,
   start there.

**Too low:** High reconstruction error. Topics too broad and generic. Multiple distinct themes
lumped into single components.

**Too high:** Redundant components (two very similar topics). Hard to interpret. Risk of
fitting noise. Linear increase in compute cost.

**Typical ranges by domain:**
| Domain | Typical `n_components` |
|---|---|
| Text topic modeling | 5–50 (start: 10–20) |
| Face/image parts | 10–100 |
| Mutational signatures (genomics) | 3–15 |
| scRNA-seq gene programs | 10–50 |
| Audio source separation | 2–10 |
| Recommender systems | 10–200 |

### `init` — Initialization Strategy

**Default:** `None` → resolves to `'nndsvda'` if $k \leq \min(n,p)$, else `'random'`.

**VERSION CHANGE (sklearn v1.1):** Default changed from `'nndsvd'` to `'nndsvda'`. If you have
old code or old model configs that relied on `'nndsvd'` being the default, behavior has changed.

**Options:**
- `'nndsvda'` **(recommended for most cases):** NNDSVD with zeros replaced by `mean(X)`.
  Avoids zero-locking with MU solver. Fast convergence. Works for any solver.
- `'nndsvd'` **(for sparse data/sparse output):** More zeros in initial W, H. Better if you
  want sparse components. Incompatible with MU solver (zero-locking). Use with `solver='cd'`.
- `'nndsvdar'` **(middle ground):** NNDSVD zeros replaced by small random values. Good with
  MU solver on sparse data.
- `'random'` **(for robustness via restarts):** Pure random initialization. Non-deterministic.
  Should be run multiple times; take the run with lowest `reconstruction_err_`. Also the only
  option when $k > \min(n,p)$ (rank exceeds the number of singular values available).
- `'custom'` **(expert use):** Pass your own matrices via `fit(X, W=W_init, H=H_init)`.
  Requires `n_components='auto'` or consistent shape.

**Interaction with solver:**
- `solver='cd'` + any init: works well.
- `solver='mu'` + `init='nndsvd'`: danger — zero entries created by NNDSVD will be locked at
  zero forever by multiplicative updates. Use `'nndsvda'` or `'random'` with MU.

### `solver` — Optimization Algorithm

**Options:** `'cd'` (coordinate descent, default) or `'mu'` (multiplicative updates).

**`'cd'`:** Use this unless you need a non-Frobenius beta-divergence. It is 5–24× faster than
MU and provides a stronger convergence guarantee (converges to a stationary point). Works only
with `beta_loss='frobenius'`.

**`'mu'`:** Required when `beta_loss='kullback-leibler'`, `'itakura-saito'`, or any float
value of `beta_loss`. Also the only option for custom beta values.

**Decision rule:**
```
Is your data counts (text BOW, gene expression, mutation counts)?
  → solver='mu', beta_loss='kullback-leibler'
Is your data audio spectrograms?
  → solver='mu', beta_loss='itakura-saito'
Everything else?
  → solver='cd', beta_loss='frobenius'  (fastest and safest)
```

### `beta_loss` — Reconstruction Objective

**Type:** `float | {'frobenius', 'kullback-leibler', 'itakura-saito'}`

Named aliases map to floats: `'frobenius'` = 2.0, `'kullback-leibler'` = 1.0, `'itakura-saito'` = 0.0.

**Constraint:** Only used when `solver='mu'`. With `solver='cd'`, this parameter is ignored
and Frobenius loss is always used.

**Constraint for IS ($\beta \leq 0$):** $\mathbf{X}$ cannot contain zeros (log of zero is
undefined). Pre-process by adding a small epsilon: `X = X + 1e-10`.

**When each divergence is appropriate:**
- **Frobenius ($\beta=2$):** Minimizes squared error. Assumes equal-variance Gaussian noise.
  Appropriate for continuous real-valued data (pixel intensities, normalized features).
- **KL ($\beta=1$):** Minimizes relative entropy. Assumes Poisson counting noise. Appropriate
  for count data (word counts, read counts, mutation counts). Treats small counts differently
  from large ones — a miss on a count of 1 is penalized differently than a miss on a count
  of 1000.
- **IS ($\beta=0$):** Scale-invariant. A prediction that misses by a factor of 2 is penalized
  equally whether the true value is 0.001 or 1000. This makes it appropriate for audio
  spectrograms where power spans many orders of magnitude.
- **Intermediate $\beta$:** Values between 0 and 2 interpolate between these behaviors.
  $\beta = 0.5$ is sometimes used for audio with moderate dynamic range.

### `alpha_W`, `alpha_H`, `l1_ratio` — Regularization Triad

These three parameters jointly control regularization. The penalty added to the objective is:

$$\text{penalty} = \rho \cdot \alpha_W \|\mathbf{W}\|_1 + \frac{1-\rho}{2} \cdot \alpha_W \|\mathbf{W}\|_F^2 + \rho \cdot \alpha_H \|\mathbf{H}\|_1 + \frac{1-\rho}{2} \cdot \alpha_H \|\mathbf{H}\|_F^2$$

where $\rho$ = `l1_ratio`.

**Defaults:** `alpha_W=0.0`, `alpha_H='same'` (inherits `alpha_W`), `l1_ratio=0.0`. No
regularization by default.

**`alpha_H='same'`** means: apply the same alpha as `alpha_W` to both matrices. To regularize
only H (encodings), set `alpha_W=0, alpha_H=0.1`.

**Breaking change (sklearn v1.0):** The old `alpha` and `regularization` parameters were
removed. Update old code:
- Old: `NMF(alpha=0.1, regularization='both')` → New: `NMF(alpha_W=0.1, alpha_H=0.1, l1_ratio=0.0)`
- Old: `NMF(alpha=0.1, regularization='components')` → New: `NMF(alpha_W=0.0, alpha_H=0.1)`

**Practical patterns:**
```python
# Pattern 1: Sparse topic-word matrix (H sparse → focused topics)
NMF(alpha_W=0.0, alpha_H=0.1, l1_ratio=1.0)

# Pattern 2: Sparse document-topic encodings (W sparse → each doc in few topics)
NMF(alpha_W=0.1, alpha_H=0.0, l1_ratio=1.0)

# Pattern 3: Both sparse (classic sparse NMF approximation)
NMF(alpha_W=0.1, alpha_H=0.1, l1_ratio=1.0)

# Pattern 4: Elastic net (sparse but stable)
NMF(alpha_W=0.0, alpha_H=0.1, l1_ratio=0.5)
```

**Too high alpha:** Components collapse (H → all zeros; W tries to compensate but can't). Watch
for `reconstruction_err_` skyrocketing while components look nearly zero.

**Too low alpha (near 0):** Dense components. May be fine for image data; will produce overlapping
topics in text.

### `max_iter` and `tol` — Convergence Control

**`max_iter`** (default: 200): Maximum number of iterations. 200 is often insufficient.

| Scenario | Recommended `max_iter` |
|---|---|
| CD solver, well-conditioned data | 200–500 |
| MU solver, KL divergence | 500–2000 |
| Large sparse matrices (text) | 500–1000 |
| `init='random'`, multiple restarts | 500+ |

**Always check:** `model.n_iter_ == model.max_iter` after fitting → did not converge. Sklearn
emits a `ConvergenceWarning`.

**`tol`** (default: 1e-4): Stop when relative change in reconstruction error falls below this
threshold. Setting `tol=0.0` forces running until `max_iter`.

### `shuffle` (CD solver only)

**Default:** `False`. If `True`, randomizes the order of coordinate updates each iteration.
Can help escape poor local minima at the cost of slightly more compute. Worth trying if you
observe degenerate components with default settings.

### Tuning Playbook Table

| Hyperparameter | Start here | Search range (Optuna) | When to change |
|---|---|---|---|
| `n_components` | 20 | `suggest_int('k', 5, 50, log=True)` | Always tune |
| `init` | `'nndsvda'` | Fixed (categorical) | Use `'random'` if degenerate solutions |
| `solver` | `'cd'` | Fixed (data-driven choice) | Use `'mu'` for KL/IS |
| `beta_loss` | `'frobenius'` | Categorical: frob/KL/IS | By data type |
| `alpha_H` | 0.0 | `suggest_float('alpha_H', 1e-4, 10, log=True)` | When you want sparse encodings |
| `alpha_W` | 0.0 | `suggest_float('alpha_W', 1e-4, 10, log=True)` | When you want sparse components |
| `l1_ratio` | 0.0 | `suggest_categorical('l1', [0, 0.5, 1.0])` | When adding any alpha > 0 |
| `max_iter` | 500 | Fixed (monitor convergence) | Increase if `n_iter_ == max_iter` |
| `tol` | 1e-4 | Fixed | Decrease for tighter convergence |

---

## §5 — Strengths

### 1. Interpretable, Parts-Based Components

NMF's defining strength is that its components are human-readable. Because all entries are
non-negative, each component is literally a description of which features "are present" in a
pattern — not a complicated signed contrast. A topic in a text NMF is a set of co-occurring
words, all positively weighted. A component in a face NMF is a set of pixels that form a
recognizable facial part. No other standard linear DR method achieves this level of direct
interpretability.

This is why NMF is the go-to method in scientific domains where interpretability is as
important as predictive accuracy: genomics (mutational signatures), ecology (microbial community
types), neuroscience (neural activity patterns), and chemistry (spectral decomposition).

### 2. Natural Fit for Non-Negative Data

Many of the most important data types in science and industry are non-negative by construction:
word counts, pixel intensities, gene expression counts, spectral measurements, user-item
interaction logs, audio power spectrograms. For all of these, NMF's non-negativity constraint
is not a restriction — it is an *alignment with reality*. PCA on count data produces negative
loadings that are biologically or physically meaningless. NMF components correspond to actual
contributions.

### 3. Flexible Loss Functions (Beta-Divergence)

Through the beta-divergence family, NMF can match its reconstruction objective to the noise
model of your data. This is a concrete, mechanistic advantage:
- Frobenius for Gaussian noise in continuous measurements.
- KL divergence for Poisson noise in counts.
- Itakura-Saito for log-scale (multiplicative) noise in audio.

No other standard sklearn DR method offers this range of noise model flexibility.

### 4. Scalability via MiniBatchNMF

With `MiniBatchNMF`, NMF can operate on datasets that do not fit in memory, with `partial_fit`
enabling true streaming. This is practically important for web-scale text, large genomics
matrices, and continuous monitoring systems. PCA has `IncrementalPCA` for the same purpose, but
for non-negative data, `MiniBatchNMF` is the right tool.

### 5. Transform on New Data

Unlike t-SNE and UMAP (which require re-running the full algorithm when new data arrives), NMF
has a natural, efficient `transform()` operation: given a new sample, hold $\mathbf{H}$ fixed
and solve for the encoding via NNLS. This is $O(kp)$ per sample, making NMF suitable for online
production inference pipelines. A topic model trained on a corpus can encode new documents in
milliseconds.

### 6. Simple, Well-Understood Theory

Unlike autoencoders and variational methods, NMF has no architecture decisions, no activation
function tuning, no need for backpropagation. The algorithm is transparent, the loss function
is explicit, and every hyperparameter has a clear mechanistic interpretation. This makes NMF
easier to debug, explain, and audit — important in regulated industries (healthcare, finance).

---

## §6 — Weaknesses & Failure Modes

### 1. Non-uniqueness

NMF solutions are not unique in general. Two runs from different initializations may find
completely different components that achieve similar reconstruction error. You cannot simply
say "topic 3 is the sports topic" — topic 3 in run 1 may correspond to topic 7 in run 2.

**Detection:** Run NMF 10 times from random initializations. Compute pairwise cosine similarity
between components across runs. If components are not consistently matched (similarity < 0.9),
your factorization is unstable.

**Mitigation:**
- Use NNDSVD initialization (more deterministic starting point).
- Add L1 regularization (sparse solutions tend to be more unique).
- Use consensus NMF (nimfa) to aggregate many runs.
- For critical applications, report only components that are stable across ≥ 80% of runs.

### 2. Sensitivity to Initialization and Local Minima

Because NMF is non-convex, the solution found depends on where you start. A bad initialization
can lead to **degenerate solutions**: one component captures nearly all variance, while the
rest collapse to near-zero. Or two components become nearly identical duplicates.

**Detection:** Check `reconstruction_err_` across multiple runs. Check if any component has
near-zero mean across all features. Check pairwise cosine similarity between components —
values > 0.95 indicate duplication.

**Mitigation:** Use `init='nndsvda'` as the default. If degenerate solutions persist, run
multiple restarts with `init='random'` and take the best.

### 3. Requires Non-Negative Input (Hard Constraint)

NMF will raise `ValueError` if any entry of $\mathbf{X}$ is negative. This rules out:
- Mean-centered data (standard preprocessing for PCA)
- Log-ratio transformed data (common in microbiome analysis)
- Difference matrices, residuals with mixed signs

**Mitigation:** Use Semi-NMF for mixed-sign data. Or min-max normalize to $[0,1]$. Or add a
constant shift: `X_shifted = X - X.min()` — though this changes the interpretation.

### 4. Slow Convergence with MU Solver

The multiplicative update rules converge slowly. For large matrices with KL divergence, you may
need 1,000–5,000 iterations to converge, with each iteration costing $O(nkp)$. For a matrix
with $n=100K$, $p=10K$, $k=50$, this is expensive.

**Mitigation:** Use `solver='cd'` with Frobenius loss whenever possible. Switch to
`MiniBatchNMF` for large datasets. Accept slightly higher reconstruction error in exchange for
speed with fewer iterations.

### 5. No Probabilistic Interpretation

NMF has no generative model (unlike LDA, which models documents as Dirichlet mixtures over
topics). This means:
- No natural way to compute document-topic posterior probabilities.
- No principled uncertainty quantification.
- Perplexity comparisons with LDA are not strictly valid.

**When this matters:** If you need calibrated probability estimates for downstream Bayesian
inference, NMF is not the right tool. Use LDA (from `gensim` or `sklearn.decomposition.LatentDirichletAllocation`)
or Probabilistic Matrix Factorization instead.

### 6. Poor Visualization (NMF is not a visualization method)

NMF with $k=2$ or $k=3$ can be used for visualization, but it is not designed for this. The
components are not optimized to maximize pairwise distances between samples in the embedding —
they are optimized to reconstruct the data. As a result:
- NMF embeddings often show clusters pressed against the axes (due to non-negativity).
- Neighborhood structure is not preserved as faithfully as t-SNE or UMAP.
- Trustworthiness scores for NMF embeddings are typically lower than for manifold methods.

**When this matters:** If visualization is your goal, use t-SNE or UMAP on the NMF encoding
(run NMF first for compression, then t-SNE for visualization). This is a common and effective
two-stage pipeline.

---

## §7 — What I Think About When Fitting

This section is the internal monologue of an expert fitting NMF. Walk through this checklist
every time you apply NMF to a new dataset. It will save you from the most common mistakes.

### Before Fitting: Data Audit

**1. Verify non-negativity. This is non-negotiable.**

```python
import numpy as np
from scipy.sparse import issparse

if issparse(X):
    assert X.data.min() >= 0, f"NMF: sparse X has negative values (min={X.data.min():.4f})"
else:
    assert X.min() >= 0, f"NMF: dense X has negative values (min={X.min():.4f})"
```

Common traps: you ran StandardScaler (subtracts mean → negatives), you log-transformed and
some values are < 1 (log < 0), or you computed a ratio and the numerator can exceed the
denominator. If you find negatives, trace back your preprocessing pipeline.

**2. Check for NaNs and Infs. NMF will silently produce garbage with NaNs.**

```python
if issparse(X):
    assert not np.any(np.isnan(X.data)), "NMF: NaN in sparse X.data"
else:
    assert not np.any(np.isnan(X)) and not np.any(np.isinf(X)), "NMF: NaN or Inf in X"
```

**3. Check the scale of your data.** Does your matrix span many orders of magnitude (e.g.,
some cells are 0, some are 1,000,000)? If so:
- For count data: apply `log1p`: `X_log = np.log1p(X)` — this stabilizes variance.
- For audio spectrograms: use IS divergence (`beta_loss='itakura-saito'`) — it is scale-invariant.
- For images: normalize to [0, 1].

**4. Do NOT apply StandardScaler or zero-mean centering before NMF.**
This is the single most common mistake. Write it on a sticky note if you have to.

**5. Choose `n_components` heuristic.** Start here:
- Text: 10–20 topics as a starting point.
- Genomics: look at published work for your data type (mutational signatures: 3–15).
- Images: try $k = \sqrt{p/2}$ as a rough estimate.
- Source separation: roughly equal to expected number of sources.

**6. Choose loss function based on data type:**
- Raw counts → `beta_loss='kullback-leibler'`, `solver='mu'`
- Continuous, pre-normalized → `beta_loss='frobenius'`, `solver='cd'`
- Audio/spectrogram → `beta_loss='itakura-saito'`, `solver='mu'`

### During Fitting: Monitoring

**7. Always set `verbose=1` on your first fit to see convergence progress.** You want to see
the reconstruction error decreasing consistently. If it plateaus early, increase `max_iter` or
check for initialization issues.

**8. After fitting, immediately check convergence:**

```python
model = NMF(n_components=20, init='nndsvda', solver='cd', max_iter=500, random_state=42)
W = model.fit_transform(X)

if model.n_iter_ == model.max_iter:
    print(f"WARNING: Did not converge in {model.max_iter} iterations!")
    print(f"Consider: max_iter={model.max_iter * 3}, or check your data scale.")
else:
    print(f"Converged in {model.n_iter_} iterations. Reconstruction error: {model.reconstruction_err_:.6f}")
```

**9. For `init='random'`, run multiple restarts:**

```python
best_err = float('inf')
best_model = None
for seed in range(5):
    m = NMF(n_components=20, init='random', random_state=seed, max_iter=500)
    m.fit_transform(X)
    if m.reconstruction_err_ < best_err:
        best_err = m.reconstruction_err_
        best_model = m
print(f"Best reconstruction error across 5 restarts: {best_err:.4f}")
```

### After Fitting: Quality Checks

**10. Compute relative reconstruction error.**

```python
W = model.fit_transform(X)
X_hat = W @ model.components_
if issparse(X):
    X_dense = X.toarray()
else:
    X_dense = X
rel_err = np.linalg.norm(X_dense - X_hat, 'fro') / np.linalg.norm(X_dense, 'fro')
print(f"Relative reconstruction error: {rel_err:.4f}")
```

Typical ranges: text TF-IDF 0.05–0.30, image pixels 0.01–0.10, genomics counts 0.05–0.20.
A very high error (> 0.5) means too few components or a poor factorization.

**11. Inspect the components. Always visualize or print them.**

```python
H = model.components_   # (n_components, n_features)

# Text: top words per topic
for i, topic_vec in enumerate(H):
    top_idx = topic_vec.argsort()[:-11:-1]
    top_terms = [feature_names[j] for j in top_idx]
    print(f"Topic {i:2d}: {', '.join(top_terms)}")

# Image: reshape each component as an image and display
# Genomics: match to COSMIC signatures by cosine similarity
```

**12. Check sparsity of encodings (W).**

```python
sparsity_W = (W == 0).mean()
sparsity_H = (model.components_ == 0).mean()
print(f"Sparsity of W (encodings): {sparsity_W:.1%}")
print(f"Sparsity of H (components): {sparsity_H:.1%}")
```

If W is too dense (< 5% zeros) and you want sparse, topic-like assignments, add
`alpha_H` regularization with `l1_ratio=1.0`.

**13. Smell tests — if any of these are true, something is wrong:**

- One component has values 10× larger than all others → likely a degenerate initialization.
  Re-run with `init='nndsvda'` or multiple random restarts.
- Two components have cosine similarity > 0.95 → too many components. Reduce `n_components`.
- Reconstruction error is nearly zero → too many components, or `n_components ≥ rank(X)`.
  Reduce `n_components`.
- All components look nearly identical → strong positive correlation in features; consider
  normalizing columns of X before factorization.

**14. Decision tree based on what you see:**

```
High reconstruction error (> 0.3)?
  → Increase n_components, or check if data truly has low-rank structure
Components are unintelligible/noisy?
  → Add L1 regularization (alpha_H=0.1, l1_ratio=1.0)
Components are duplicated?
  → Reduce n_components
W is too dense (all samples use all components)?
  → Add alpha_W regularization
Algorithm did not converge?
  → Increase max_iter, or switch from MU to CD if using Frobenius
Getting different components on each run?
  → Use init='nndsvda', or run 10 random restarts and check stability
```

---

## §8 — Diagnostic Plots & Evaluation

Visual diagnostics for NMF are different from PCA because NMF has no concept of "explained
variance." The key questions are: (1) how well does the factorization reconstruct the data?
(2) how stable are the components? (3) are the components interpretable and non-redundant?
Here is the complete diagnostic suite with full, runnable Python code.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import trustworthiness
from scipy.sparse import issparse

# ── Shared setup ──────────────────────────────────────────────────────────────
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', min_df=5)
X = vectorizer.fit_transform(newsgroups.data)
feature_names = vectorizer.get_feature_names_out()
K_FIT = 20    # number of components for the main model

model = NMF(n_components=K_FIT, init='nndsvda', solver='cd', max_iter=500, random_state=42)
W = model.fit_transform(X)
H = model.components_

# ── Plot 1: Reconstruction Error vs. n_components (Elbow Plot) ────────────────
k_range = range(2, 41, 2)
errors = []
for k in k_range:
    m = NMF(n_components=k, init='nndsvda', solver='cd', max_iter=300, random_state=42)
    m.fit(X)
    errors.append(m.reconstruction_err_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(k_range), errors, 'bo-', linewidth=2, markersize=6)
ax.set_xlabel('n_components (k)', fontsize=12)
ax.set_ylabel('Reconstruction Error (Frobenius)', fontsize=12)
ax.set_title('NMF: Reconstruction Error vs. Number of Components', fontsize=13)
ax.axvline(x=K_FIT, color='red', linestyle='--', alpha=0.7, label=f'Selected k={K_FIT}')
ax.legend()
plt.tight_layout()
plt.savefig('nmf_elbow.png', dpi=150)
plt.show()
# READ: Look for the "elbow" — the point where additional components give
# diminishing reduction in error. If the curve is smooth with no elbow,
# your data may not have a clear low-rank structure.

# ── Plot 2: Component Heatmap (Topic-Word Matrix) ─────────────────────────────
# Show top 10 words per topic as a heatmap
n_top_words = 10
top_word_indices = []
top_word_labels = []
for topic_vec in H:
    idx = topic_vec.argsort()[:-n_top_words-1:-1]
    top_word_indices.append(idx)
    top_word_labels.extend([feature_names[i] for i in idx])

# Build a (n_components, n_top_words) value matrix
heatmap_data = np.array([H[i, top_word_indices[i]] for i in range(K_FIT)])

fig, ax = plt.subplots(figsize=(14, 8))
all_labels = [feature_names[i] for i in top_word_indices[0]]  # first topic's words
# For display, just show the full H[:, :50] normalized
H_norm = H / (H.max(axis=1, keepdims=True) + 1e-10)
sns.heatmap(H_norm[:, :50], ax=ax, cmap='YlOrRd', xticklabels=feature_names[:50],
            yticklabels=[f'T{i}' for i in range(K_FIT)])
ax.set_title('NMF Component Matrix H (first 50 features, normalized per row)', fontsize=12)
ax.set_xlabel('Features (words)')
ax.set_ylabel('Components (topics)')
plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()
plt.savefig('nmf_component_heatmap.png', dpi=150)
plt.show()
# READ: Each row is a topic. Dark cells = words strongly associated with that topic.
# If two rows look similar, you have duplicate components → reduce n_components.
# If all rows look the same, your alpha_H is too low → add L1 regularization.

# ── Plot 3: Sparsity Analysis ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Sparsity of W (document-topic matrix)
w_zeros_per_row = (W == 0).sum(axis=1) / K_FIT
axes[0].hist(w_zeros_per_row, bins=30, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Fraction of zero components per document')
axes[0].set_ylabel('Number of documents')
axes[0].set_title(f'W Sparsity (mean: {w_zeros_per_row.mean():.1%} zeros)')
axes[0].axvline(x=w_zeros_per_row.mean(), color='red', linestyle='--')

# Sparsity of H (component-word matrix)
h_zeros_per_row = (H == 0).sum(axis=1) / H.shape[1]
axes[1].bar(range(K_FIT), h_zeros_per_row, color='darkorange', edgecolor='white')
axes[1].set_xlabel('Component index')
axes[1].set_ylabel('Fraction of zero entries')
axes[1].set_title('H Row Sparsity (fraction of zero words per topic)')

plt.suptitle('NMF Sparsity Diagnostics', fontsize=13)
plt.tight_layout()
plt.savefig('nmf_sparsity.png', dpi=150)
plt.show()
# READ: Left plot: most documents should use only a few topics (right-skewed distribution
# when using L1 regularization). Right plot: each topic should focus on a subset of words.
# If W sparsity is near 0 (dense), add alpha_H L1 regularization.

# ── Plot 4: Dominant Topic Distribution ───────────────────────────────────────
dominant_topic = W.argmax(axis=1)
topic_counts = np.bincount(dominant_topic, minlength=K_FIT)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(K_FIT), topic_counts, color='teal', edgecolor='white')
ax.set_xlabel('Topic index')
ax.set_ylabel('Number of documents')
ax.set_title('Document Count per Dominant Topic')
plt.tight_layout()
plt.savefig('nmf_topic_distribution.png', dpi=150)
plt.show()
# READ: Balanced distribution = good topic coverage. If one topic dominates (>50% of docs),
# you may have a degenerate solution. If many topics have zero documents, reduce n_components.

# ── Plot 5: Component Inter-similarity (Redundancy Check) ─────────────────────
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(H)
np.fill_diagonal(sim_matrix, 0)   # zero out diagonal for visualization

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(sim_matrix, ax=ax, cmap='coolwarm', center=0, vmin=0, vmax=1,
            xticklabels=[f'T{i}' for i in range(K_FIT)],
            yticklabels=[f'T{i}' for i in range(K_FIT)])
ax.set_title('Topic-Topic Cosine Similarity (Redundancy Check)', fontsize=12)
plt.tight_layout()
plt.savefig('nmf_topic_similarity.png', dpi=150)
plt.show()
# READ: Off-diagonal values close to 0 = good (distinct topics).
# Values > 0.7 between topic pairs = redundant components — reduce n_components
# or add more regularization.

# ── Plot 6: Reconstruction Quality Per Sample ─────────────────────────────────
X_dense = X.toarray() if issparse(X) else X
X_hat = W @ H
sample_errors = np.linalg.norm(X_dense - X_hat, axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(sample_errors, bins=40, color='slateblue', edgecolor='white')
ax.set_xlabel('Per-sample reconstruction error (Frobenius norm)')
ax.set_ylabel('Number of samples')
ax.set_title('Distribution of Per-Sample Reconstruction Errors')
ax.axvline(x=np.percentile(sample_errors, 95), color='red', linestyle='--',
           label='95th percentile')
ax.legend()
plt.tight_layout()
plt.savefig('nmf_sample_errors.png', dpi=150)
plt.show()
# READ: Long right tail = some documents are hard to reconstruct. These may be:
# - Very short documents (few non-zero features)
# - Off-topic documents not well covered by discovered topics
# - Outlier documents that would benefit from Robust NMF
```

> 🏆 **Best practice:** Run all six of these diagnostics before concluding your NMF model is
> good. The elbow plot tells you whether your `k` is in the right range. The component heatmap
> tells you if topics are interpretable. The sparsity plots tell you if regularization is
> working. The similarity matrix catches redundancy. Never skip the redundancy check —
> duplicate topics are the most common silent failure of NMF in production.

---

## §9 — Innovative Industry Applications

NMF is one of those algorithms whose true impact becomes clear when you look beyond text mining.
Here are six domains where NMF is doing genuinely important scientific and commercial work.

### 9.1 Mutational Signatures in Cancer Genomics (Highest-Impact Application)

> 🏭 **Industry application:** Used at the Sanger Institute, Broad Institute, and major cancer
> centers to characterize the mutational history of tumor genomes.

When a cell acquires a somatic mutation, the mutation's identity (which base changed to which,
in what trinucleotide context) reveals something about the mutagenic process that caused it.
UV radiation leaves a characteristic signature of C→T transitions at dipyrimidines. Smoking
causes C→A transversions. Chemotherapy drugs (temozolomide, platinum) each leave their own
mark.

A tumor sample is represented as a 96-dimensional vector: 6 base substitution types × 16
trinucleotide contexts. A cohort of tumors forms a matrix $\mathbf{X} \in \mathbb{R}^{n \times 96}$
of non-negative mutation counts. NMF decomposes this into:
- $\mathbf{H} \in \mathbb{R}^{k \times 96}$ — the $k$ mutational signatures (each a pattern
  over 96 mutation types)
- $\mathbf{W} \in \mathbb{R}^{n \times k}$ — the exposure of each tumor to each signature

The COSMIC database catalogs 78+ validated SBS (Single Base Substitution) signatures, all
discovered or confirmed via NMF. This directly affects clinical decision-making: detecting
BRCA-related signature SBS3 suggests HRD (homologous recombination deficiency) and sensitivity
to PARP inhibitors. Detecting SBS4 (smoking) in a lung cancer patient confirms etiology.

```python
from sklearn.decomposition import NMF
import numpy as np

# X: (n_tumors, 96) mutation count matrix. Use KL divergence (counts).
model = NMF(
    n_components=5,        # Expected number of signatures in this cohort
    init='nndsvda',
    solver='mu',
    beta_loss='kullback-leibler',   # Poisson model for counts
    max_iter=2000,
    random_state=42,
    alpha_H=0.0,
    l1_ratio=0.0,
)
W = model.fit_transform(X_mutations)   # (n_tumors, 5) — exposures
H = model.components_                   # (5, 96) — signatures

# Match discovered signatures to COSMIC using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
# cosmic_signatures: (78, 96) — load from COSMIC website
# sim = cosine_similarity(H, cosmic_signatures)
# best_match = sim.argmax(axis=1)
```

> 🔧 **In practice:** For production-grade signature analysis, use the SigProfilerExtractor
> tool (Python) built on NMF, which handles rank selection, uncertainty quantification, and
> COSMIC matching. For exploratory work, sklearn NMF with KL divergence and multiple random
> restarts is sufficient.

### 9.2 Audio Source Separation and Music Transcription

> 🏭 **Industry application:** Used in iZotope's RX audio repair suite, LANDR's mastering
> pipeline, and academic music information retrieval.

A music recording contains multiple simultaneous instruments. The Short-Time Fourier Transform
(STFT) magnitude spectrogram $\mathbf{X} \in \mathbb{R}^{F \times T}_+$ (frequency bins ×
time frames) is non-negative. Under the linear mixing assumption, each frequency bin at each
time is a sum of contributions from each instrument.

NMF decomposes this as $\mathbf{X} \approx \mathbf{W}\mathbf{H}$ where:
- $\mathbf{H} \in \mathbb{R}^{k \times F}$ — $k$ spectral templates (each represents one
  instrument's timbre, with harmonic peaks at integer multiples of its fundamental frequency)
- $\mathbf{W} \in \mathbb{R}^{T \times k}$ — time-varying activations (how loud each
  instrument is at each time frame)

IS divergence ($\beta=0$) is preferred because audio perception is logarithmic — a prediction
off by factor 2 matters equally at any amplitude.

```python
import librosa
import numpy as np
from torchnmf.nmf import NMF as TorchNMF
import torch

# Load audio and compute STFT magnitude spectrogram
y, sr = librosa.load('mixed_music.wav', sr=22050)
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))  # (1025, T), non-negative

S_tensor = torch.tensor(S, dtype=torch.float32)
model = TorchNMF(S_tensor.shape, rank=4)  # 4 instruments
model.fit(S_tensor, beta=0)               # beta=0 → Itakura-Saito divergence

# Separate sources using Wiener filtering
H_templates = model.W.detach().numpy()    # (4, 1025) — spectral templates
W_activations = model.H.detach().numpy()  # (1025, 4) — note: torchnmf naming

# Reconstruct each source's spectrogram
for k in range(4):
    source_spectrogram = np.outer(H_templates[k], W_activations[:, k])
    # Apply Wiener mask to original complex STFT for phase recovery
```

### 9.3 Hyperspectral Image Unmixing in Remote Sensing

> 🏭 **Industry application:** Used by geological survey agencies, ESA (European Space Agency),
> USGS, and mining exploration companies.

A hyperspectral sensor captures images across hundreds of spectral bands (visible + infrared).
Each ground pixel is a mixture of several materials ("endmembers"): bare soil, green vegetation,
dry grass, rock, urban concrete. The **linear mixing model** says:

$$\text{pixel spectrum} = \sum_r a_r \cdot \text{endmember spectrum}_r$$

where $a_r \geq 0$ are abundances (summing to 1) and endmember spectra are non-negative
reflectance curves. This is NMF: the pixel matrix $\mathbf{X} \in \mathbb{R}^{n_\text{pix} \times n_\text{bands}}_+$
factorizes into abundances $\mathbf{W}$ and endmember spectra $\mathbf{H}$.

```python
from sklearn.decomposition import NMF
import numpy as np

# X: (n_pixels, n_bands) — unfolded hyperspectral cube, reflectance values in [0, 1]
n_endmembers = 5   # estimated number of land cover types

model = NMF(
    n_components=n_endmembers,
    init='nndsvda',
    solver='cd',
    max_iter=1000,
    random_state=42,
)
W = model.fit_transform(X_hyperspectral)   # (n_pixels, 5) — abundance maps
H = model.components_                       # (5, n_bands) — endmember spectra

# Visualize abundance map for endmember 0 (e.g., vegetation)
abundance_map = W[:, 0].reshape(height, width)   # fold back into image shape
```

### 9.4 Single-Cell RNA Sequencing Gene Program Discovery

> 🏭 **Industry application:** Used in computational biology labs at the Broad Institute,
> Sanger Institute, EMBL-EBI, and in pharma R&D (Genentech, Pfizer) for drug target discovery.

A single-cell RNA-seq experiment profiles gene expression across thousands of individual cells.
The count matrix $\mathbf{X} \in \mathbb{Z}^{n_\text{cells} \times n_\text{genes}}_+$ contains
non-negative integer RNA counts. NMF decomposes this into:
- $\mathbf{H}$ (gene programs) — co-regulated gene sets defining cell states or types
- $\mathbf{W}$ (cell-program weights) — which programs are active in each cell

PCA on scRNA-seq gives components with negative gene loadings, which are biologically
meaningless (RNA counts cannot be negative). NMF's non-negative gene loadings directly
correspond to sets of co-expressed genes.

```python
from sklearn.decomposition import NMF
import numpy as np

# X_scrna: (n_cells, n_genes) — log-normalized count matrix (log1p of CPM)
# Apply log1p to compress dynamic range while preserving non-negativity
X_log = np.log1p(X_counts_normalized)    # still non-negative

model = NMF(
    n_components=15,       # Number of gene programs to discover
    init='nndsvda',
    solver='mu',
    beta_loss='kullback-leibler',   # KL preferred for count-like data
    max_iter=1000,
    random_state=42,
)
W_cells = model.fit_transform(X_log)     # (n_cells, 15) — cell-program weights
H_genes = model.components_              # (15, n_genes) — gene programs

# Top genes per program
gene_names = np.array(adata.var_names)  # if using AnnData
for i, prog in enumerate(H_genes):
    top_genes = gene_names[prog.argsort()[:-11:-1]]
    print(f"Program {i}: {', '.join(top_genes)}")
```

### 9.5 Text Topic Modeling: NMF vs LDA

> 🏭 **Industry application:** Used in news aggregation (Reuters, AP), customer feedback
> analysis (SurveyMonkey, Qualtrics), scientific literature mining (PubMed, Semantic Scholar).

NMF and Latent Dirichlet Allocation (LDA) are the two workhorses of topic modeling. Knowing
when to prefer NMF is a practical skill.

| Property | NMF | LDA |
|---|---|---|
| Components | Non-negative loadings | Dirichlet-distributed probabilities |
| Topic interpretability | High (additive parts) | High (proper probability distributions) |
| Statistical model | None (deterministic) | Generative (Dirichlet-Multinomial) |
| Best input | TF-IDF (non-counts) | Raw word counts |
| Speed | Faster | Slower (sampling or VI) |
| New document inference | Fast (`transform()` via NNLS) | Approximate (VI or Gibbs) |
| Topic coherence (C_v) | Often higher empirically | Slightly lower on some benchmarks |
| Short text handling | Struggles | Also struggles (document length assumption) |

**Rule of thumb:** Use NMF with TF-IDF for speed and good empirical coherence. Use LDA with
raw counts when you need a proper probabilistic model and calibrated uncertainty.

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', min_df=5)
X_tfidf = vectorizer.fit_transform(documents)

model = NMF(n_components=20, init='nndsvda', solver='cd', max_iter=500, random_state=42)
W = model.fit_transform(X_tfidf)
feature_names = vectorizer.get_feature_names_out()

for i, topic in enumerate(model.components_):
    top = [feature_names[j] for j in topic.argsort()[:-11:-1]]
    print(f"Topic {i:2d}: {', '.join(top)}")
```

### 9.6 Urban Mobility Pattern Discovery

> 🏭 **Industry application:** "Identifying Population Movements with Non-Negative Matrix
> Factorization from Wi-Fi User Counts in Smart and Connected Cities" (arXiv 2111.10459).
> Used by city planners and transit authorities to understand population flow.

A matrix $\mathbf{X} \in \mathbb{R}^{n_\text{locations} \times n_\text{time\_slots}}_+$
records how many Wi-Fi users are connected at each location at each time of day. NMF discovers
archetypal temporal patterns:
- A "morning rush hour" pattern peaking at 8–9am at transit hubs
- A "lunch break" pattern peaking at 12pm at restaurants and food courts
- An "evening entertainment" pattern peaking at 7–9pm at restaurants and bars
- An "always-busy" pattern flat across all times at airports and hospitals

Each location's usage is a weighted combination of these archetypes. City planners use this to
optimize transit schedules; retailers use it to staff stores; emergency services use it to
pre-position resources.

```python
# X: (n_locations, n_time_slots) — Wi-Fi user counts per location per 15-min interval
model = NMF(
    n_components=6,        # Expect ~6 archetypal daily patterns
    init='nndsvda',
    solver='cd',
    max_iter=500,
    random_state=42,
)
W_locations = model.fit_transform(X_wifi)  # (n_locations, 6) — which patterns at each site
H_patterns = model.components_             # (6, n_time_slots) — the temporal archetypes

# Visualize each temporal archetype
time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in (0, 15, 30, 45)]
for i, pattern in enumerate(H_patterns):
    plt.plot(pattern, label=f'Pattern {i}')
plt.xlabel('Time of day'); plt.ylabel('Relative activity')
plt.title('NMF Temporal Archetypes in Urban Wi-Fi Data')
plt.legend()
```

---

## §10 — Complete Python Worked Example

We will run two end-to-end examples: (1) topic modeling on the 20 Newsgroups dataset with
Optuna hyperparameter tuning, and (2) face parts decomposition on the Olivetti Faces dataset.
Both use only real, named sklearn datasets and verified 1.5.x syntax.

### Example A: 20 Newsgroups Topic Modeling with Optuna

```python
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ── 1. Data loading and EDA ───────────────────────────────────────────────────
newsgroups_all = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes'),
    random_state=42,
)
print(f"Total documents: {len(newsgroups_all.data)}")
print(f"Categories: {len(newsgroups_all.target_names)}")
# 18,846 documents, 20 categories

# Split for cross-validated model selection
docs_train, docs_val = train_test_split(
    newsgroups_all.data, test_size=0.2, random_state=42
)
print(f"Train: {len(docs_train)}, Val: {len(docs_val)}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    min_df=5,           # ignore very rare words
    max_df=0.85,        # ignore very common words (near stopwords)
    ngram_range=(1, 1), # unigrams only for NMF topic models
    sublinear_tf=True,  # log-scale TF — reduces influence of very common terms
)
X_train = vectorizer.fit_transform(docs_train)   # sparse, (15077, 8000), non-negative
X_val = vectorizer.transform(docs_val)
feature_names = vectorizer.get_feature_names_out()

print(f"Train matrix shape: {X_train.shape}")
print(f"Sparsity: {1 - X_train.nnz / np.prod(X_train.shape):.1%}")

# ── 3. Baseline NMF fit ───────────────────────────────────────────────────────
baseline_model = NMF(
    n_components=20,
    init='nndsvda',
    solver='cd',
    max_iter=500,
    random_state=42,
    tol=1e-4,
)
W_train_base = baseline_model.fit_transform(X_train)
W_val_base = baseline_model.transform(X_val)

print(f"\nBaseline model:")
print(f"  n_components=20, reconstruction_err={baseline_model.reconstruction_err_:.4f}")
print(f"  Converged: {baseline_model.n_iter_ < baseline_model.max_iter} ({baseline_model.n_iter_} iters)")

# Validation reconstruction error
import numpy as np
X_val_dense = X_val.toarray()
X_val_hat = W_val_base @ baseline_model.components_
val_err = np.linalg.norm(X_val_dense - X_val_hat, 'fro')
print(f"  Validation Frobenius error: {val_err:.4f}")

# ── 4. Hyperparameter tuning with Optuna ─────────────────────────────────────
def nmf_objective(trial):
    k = trial.suggest_int('n_components', 5, 40, log=True)
    alpha_H = trial.suggest_float('alpha_H', 1e-5, 5.0, log=True)
    l1_ratio = trial.suggest_categorical('l1_ratio', [0.0, 0.5, 1.0])

    model = NMF(
        n_components=k,
        init='nndsvda',
        solver='cd',
        alpha_W=0.0,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        max_iter=400,
        random_state=42,
        tol=1e-4,
    )
    W_t = model.fit_transform(X_train)
    W_v = model.transform(X_val)
    X_v_hat = W_v @ model.components_
    val_frobenius = np.linalg.norm(X_val.toarray() - X_v_hat, 'fro')
    return val_frobenius

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(nmf_objective, n_trials=30, show_progress_bar=True)

best = study.best_params
print(f"\nBest hyperparameters: {best}")
print(f"Best validation error: {study.best_value:.4f}")

# ── 5. Refit best model ───────────────────────────────────────────────────────
best_model = NMF(
    n_components=best['n_components'],
    init='nndsvda',
    solver='cd',
    alpha_W=0.0,
    alpha_H=best['alpha_H'],
    l1_ratio=best['l1_ratio'],
    max_iter=1000,    # more iterations for final model
    random_state=42,
)
W_best = best_model.fit_transform(X_train)
print(f"\nFinal model: k={best['n_components']}, err={best_model.reconstruction_err_:.4f}")

# ── 6. Inspect discovered topics ─────────────────────────────────────────────
print("\n=== Discovered Topics ===")
k_best = best['n_components']
for i, topic_vec in enumerate(best_model.components_):
    top_idx = topic_vec.argsort()[:-11:-1]
    top_words = [feature_names[j] for j in top_idx]
    print(f"Topic {i:2d}: {', '.join(top_words)}")

# ── 7. Visualization: Topic importance per document ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Distribution of dominant topics
dominant = W_best.argmax(axis=1)
counts = np.bincount(dominant, minlength=k_best)
axes[0].bar(range(k_best), counts, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Topic index'); axes[0].set_ylabel('# Documents')
axes[0].set_title('Documents per Dominant Topic')

# Right: Optuna optimization history
vals = [t.value for t in study.trials]
axes[1].plot(vals, 'o-', markersize=4, color='darkorange', alpha=0.7)
axes[1].axhline(y=study.best_value, color='red', linestyle='--',
                label=f'Best: {study.best_value:.2f}')
axes[1].set_xlabel('Trial'); axes[1].set_ylabel('Validation Frobenius Error')
axes[1].set_title('Optuna Optimization History'); axes[1].legend()

plt.suptitle('20 Newsgroups NMF Topic Modeling', fontsize=13)
plt.tight_layout()
plt.savefig('nmf_newsgroups_results.png', dpi=150)
plt.show()
```

### Example B: Olivetti Faces — Parts Decomposition

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Load data ──────────────────────────────────────────────────────────────
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = faces.data     # (400, 4096) — 400 faces, 64x64 pixels, values in [0,1]
print(f"Faces shape: {X_faces.shape}, min={X_faces.min():.3f}, max={X_faces.max():.3f}")
# Already in [0, 1], non-negative — no preprocessing needed

# ── 2. Fit NMF with k=36 components ──────────────────────────────────────────
k = 36
model = NMF(n_components=k, init='nndsvda', solver='cd', max_iter=1000, random_state=42)
W_faces = model.fit_transform(X_faces)
H_faces = model.components_   # (36, 4096) — 36 face parts

print(f"Convergence: {model.n_iter_} iterations, err={model.reconstruction_err_:.4f}")

rel_err = np.linalg.norm(X_faces - W_faces @ H_faces, 'fro') / np.linalg.norm(X_faces, 'fro')
print(f"Relative reconstruction error: {rel_err:.4f}")

# ── 3. Visualize discovered face parts ────────────────────────────────────────
fig, axes = plt.subplots(6, 6, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    component = H_faces[i].reshape(64, 64)
    ax.imshow(component, cmap='gray', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'Part {i}', fontsize=7)
plt.suptitle(f'NMF Face Parts (k={k}) — Olivetti Faces Dataset', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('nmf_olivetti_parts.png', dpi=150)
plt.show()

# ── 4. Reconstruct and compare ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 6, figsize=(14, 7))
for col_idx in range(6):
    face_idx = col_idx * 5
    original = X_faces[face_idx].reshape(64, 64)
    reconstructed = (W_faces[face_idx] @ H_faces).reshape(64, 64)

    axes[0, col_idx].imshow(original, cmap='gray')
    axes[0, col_idx].axis('off')
    axes[0, col_idx].set_title(f'Face {face_idx}', fontsize=8)

    axes[1, col_idx].imshow(reconstructed, cmap='gray')
    axes[1, col_idx].axis('off')
    axes[1, col_idx].set_title(f'NMF Recon.', fontsize=8)

    # Difference map (absolute error per pixel)
    axes[2, col_idx].imshow(np.abs(original - reconstructed), cmap='hot')
    axes[2, col_idx].axis('off')
    err_i = np.linalg.norm(original - reconstructed, 'fro')
    axes[2, col_idx].set_title(f'err={err_i:.3f}', fontsize=8)

axes[0, 0].set_ylabel('Original', rotation=90, fontsize=9)
axes[1, 0].set_ylabel('NMF Recon.', rotation=90, fontsize=9)
axes[2, 0].set_ylabel('|Error|', rotation=90, fontsize=9)
plt.suptitle('NMF Face Reconstruction Quality (Olivetti Faces)', fontsize=12)
plt.tight_layout()
plt.savefig('nmf_olivetti_reconstruction.png', dpi=150)
plt.show()

# ── 5. Interpretation ─────────────────────────────────────────────────────────
# Each component (row of H) corresponds to a face part:
# Some components should show eyes, others noses, foreheads, chins, glasses.
# The encoding (row of W) tells you how much of each part a given face uses.
dominant_parts = W_faces.argmax(axis=1)
print(f"\nMost common dominant part: {np.bincount(dominant_parts).argmax()}")
print(f"Each face uses on average {(W_faces > 0.01).sum(axis=1).mean():.1f} of {k} parts")
```

**What we learn from these examples:**

In the 20 Newsgroups experiment, Optuna will typically select `n_components` in the range
15–25 (matching the 20 known newsgroup categories is not a coincidence) and a small but
non-zero `alpha_H` with `l1_ratio=1.0`, producing sparse topic assignments where each document
loads primarily on 1–3 topics. The discovered topics correspond to recognizable newsgroup
themes: computer hardware, religion, sports, automotive, space exploration.

In the Olivetti Faces experiment with $k=36$, the NMF components visually correspond to local
facial features: forehead patches, eye regions, nose tips, mouth corners, cheek shading. This
is the parts-based decomposition that Lee and Seung showed in 1999 — and it still works
beautifully today. A relative reconstruction error of 0.05–0.10 is typical, meaning NMF
captures 90–95% of the face information with just 36 non-negative parts.

---

## §11 — When to Use NMF (Decision Guide)

### Use NMF when:

| Condition | Reason |
|---|---|
| Your data is non-negative (counts, intensities, proportions) | NMF's constraint matches the data's physical properties |
| You need interpretable, additive components | NMF produces "parts" — PCA does not |
| You are modeling topics, sources, or archetypes | NMF's structure is exactly this factorization |
| Your features are physical quantities that cannot be negative | Non-negativity is semantically meaningful |
| You need to transform new samples efficiently | `transform()` via NNLS is fast and principled |
| Your data is very sparse (text, genomics) | CD solver on sparse data is very efficient |

### Do NOT use NMF when:

| Condition | Better alternative |
|---|---|
| Data has negative values (after centering, log-ratios, residuals) | PCA, ICA, or Semi-NMF |
| You need visualization with good neighborhood preservation | t-SNE or UMAP (possibly on top of NMF) |
| You need a probabilistic generative model | LDA (`sklearn.decomposition.LatentDirichletAllocation`) |
| Data is dense, continuous, Gaussian-noise | PCA (faster, unique, explained variance available) |
| Supervised setting with class labels | LDA (discriminant analysis) |
| You need guaranteed globally optimal decomposition | No DR method guarantees this, but PCA's SVD is unique |

### NMF vs. PCA vs. ICA vs. LDA (Quick Reference)

| Property | NMF | PCA | ICA | LDA |
|---|---|---|---|---|
| Component sign | Non-negative | Mixed | Mixed | Mixed |
| Interpretability | Parts-based (high) | Variance-based (medium) | Source signals (high) | Discriminative (high) |
| Input requirement | Non-negative | Any | Any | Any, needs labels |
| Uniqueness | Non-unique (generally) | Unique (SVD) | Unique (up to permutation/scale) | Unique |
| Statistical model | None | Gaussian covariance | Statistical independence | Gaussian class-conditional |
| Scale invariance | No | No | Yes | No |
| Computational cost | Medium (iterative) | Low (SVD) | Medium (iterative) | Low (eigendecomp) |
| Best for | Count/spectral data, topic models, source sep. | Continuous data, noise reduction, visualization | BSS, EEG, audio | Supervised classification DR |

### Red Flags

Stop and reconsider if you observe any of these:
1. After fitting, `(X < 0).any()` — your input wasn't truly non-negative.
2. `model.n_iter_ == model.max_iter` repeatedly even after tripling `max_iter`.
3. One component has weight 50× larger than others — degenerate initialization.
4. Components from two separate runs have cosine similarity < 0.5 — highly unstable.
5. Your goal is visualization of cluster structure — NMF is not the right tool.

---

## §12 — Top Papers to Study

### Start Here

**Paper 1 (Must read first)**

Lee, D. D. and Seung, H. S. "Learning the parts of objects by non-negative matrix
factorization." *Nature*, 401(6755):788–791, 1999. DOI: [10.1038/44565](https://doi.org/10.1038/44565)

*Why:* Three pages. The conceptual breakthrough. Read this to understand why non-negativity
matters. The face experiments are unforgettable. This paper will change how you think about
what a "feature" in a representation should be.

**Paper 2**

Lee, D. D. and Seung, H. S. "Algorithms for Non-negative Matrix Factorization." *NIPS 2000
Proceedings*, pp. 556–562, 2001.
URL: https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization

*Why:* The companion paper deriving the multiplicative update rules. Includes the formal
convergence proof via auxiliary functions — a technique worth understanding because it also
explains the EM algorithm's convergence. Read this second, immediately after the 1999 paper.

### Read Next

**Paper 3**

Boutsidis, C. and Gallopoulos, E. "SVD based initialization: A head start for nonnegative
matrix factorization." *Pattern Recognition*, 41(4):1350–1362, 2008.

*Why:* NNDSVD initialization is the default in sklearn. This paper explains why it works:
the SVD provides a warm start that already captures the dominant variance structure, and the
non-negative projection of SVD vectors is surprisingly good at preserving this structure.
If you have ever wondered why `init='nndsvda'` is the sklearn default, this paper answers it.

**Paper 4**

Hoyer, P. O. "Non-negative matrix factorization with sparseness constraints." *Journal of
Machine Learning Research*, 5:1457–1469, 2004.
URL: https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf

*Why:* Introduces the principled sparseness measure and shows that sparse NMF solutions are
more unique and interpretable. The Hoyer sparseness measure (L1/L2 norm ratio) is widely used
in signal processing. This paper is the foundation for understanding why L1 regularization on
NMF helps and when it is the right choice.

**Paper 5**

Kim, H. and Park, H. "Nonnegative Matrix Factorization Based on Alternating Nonnegativity
Constrained Least Squares and Active Set Method." *SIAM Journal on Matrix Analysis and
Applications*, 30:713–730, 2008. DOI: [10.1137/07069239X](https://doi.org/10.1137/07069239X)

*Why:* Proves that coordinate descent (ANLS variant) converges to a stationary point —
a stronger guarantee than multiplicative updates. Also provides the benchmark data showing
CD is 5–24× faster. This is why sklearn switched to CD as the default solver.

### Deep Mastery

**Paper 6**

Févotte, C. and Idier, J. "Algorithms for Nonnegative Matrix Factorization with the
β-Divergence." *Neural Computation*, 23(9):2421–2456, 2011. DOI: [10.1162/NECO_a_00168](https://doi.org/10.1162/NECO_a_00168)

*Why:* The definitive paper on beta-NMF. Derives multiplicative updates for the full
beta-divergence family, provides the EM interpretation, and justifies IS divergence for audio
via the scale-invariance argument. Read this before applying NMF to any audio or spectroscopic
data.

**Paper 7**

Arora, S., Ge, R., Kannan, R., and Moitra, A. "Computing a Nonnegative Matrix Factorization
— Provably." *STOC 2012*, 2012. arXiv: [1111.0952](http://arxiv.org/abs/1111.0952)

*Why:* Shows that NMF is NP-hard in general but solvable in polynomial time under the
separability assumption. Introduces anchor words as a computational geometric concept. Relevant
if you want to understand the theoretical limits of NMF and when fast, provably correct
algorithms exist.

**Book**

Cichocki, A., Zdunek, R., Phan, A. H., and Amari, S. "Nonnegative Matrix and Tensor
Factorizations: Applications to Exploratory Multi-way Data Analysis and Blind Source
Separation." Wiley, 2009. ISBN: 978-0-470-74666-0.

*Why:* The graduate-level reference. 500+ pages covering every NMF algorithm, the tensor
extension, and deep applications in signal processing. Use as a reference, not cover-to-cover
reading. Chapter 2 (algorithms) and Chapter 5 (audio BSS) are the most practically useful.

---

## §13 — Resources & Further Reading

### Official Documentation

- **sklearn NMF:** https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
- **sklearn MiniBatchNMF:** https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html
- **sklearn decomposition user guide:** https://scikit-learn.org/stable/modules/decomposition.html#nmf
- **nimfa documentation:** http://nimfa.biolab.si/
- **torchnmf documentation:** https://pytorch-nmf.readthedocs.io/
- **COSMIC mutational signatures:** https://cancer.sanger.ac.uk/signatures/

### Best Blog Posts & Tutorials

- **scikit-learn example — NMF for topic extraction:**
  https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
  — Official sklearn example comparing NMF and LDA on the 20 Newsgroups dataset. Direct,
  runnable code with output interpretation. Start here for text applications.

- **"The Why and How of Nonnegative Matrix Factorization" (arXiv 1401.5226):**
  Gillis (2014). A 50-page tutorial review that reads like the best parts of a textbook.
  Covers theory, algorithms, uniqueness, and applications. Freely available on arXiv.
  The most comprehensive short introduction to NMF available.

- **Distill.pub — "Visualizing Neural Machine Translation Mechanisms" (2019):**
  Not directly about NMF, but shows how non-negative matrix factorization concepts appear in
  attention mechanism analysis. Worth reading to see how NMF thinking transfers to modern
  deep learning.

### Books (Specific Chapters)

- **"Pattern Recognition and Machine Learning" (Bishop 2006):** Chapter 12 (Continuous Latent
  Variables) gives the PCA context from which NMF departs. Reading Chapter 12 before this
  chapter sharpens the contrast between the methods.
- **"Mining of Massive Datasets" (Leskovec, Rajaraman, Ullman):** Chapter 11 (Dimensionality
  Reduction) covers CUR decomposition and NMF together. Free PDF at
  http://www.mmds.org/ — Chapter 11 is 40 pages and excellent for practitioners.
- **"Speech and Language Processing" (Jurafsky & Martin, 3rd ed.):** Chapter 18 covers
  word vectors and latent semantic analysis. NMF appears as an alternative to LSA/SVD for
  creating non-negative word embeddings.

### Online Courses

- **fast.ai Practical Deep Learning for Coders:** Covers matrix factorization for recommender
  systems (a generalization of NMF). The notebook for Collaborative Filtering is the most
  hands-on treatment of NMF-adjacent methods available freely online.
- **Coursera "Mining Trends in Healthcare Data" (JHU):** Applied NMF to EHR and clinical
  data — demonstrates the genomics and clinical applications in a structured course format.

### Key Version Notes for sklearn NMF

| sklearn Version | Change |
|---|---|
| v0.19 | `solver='mu'`, `beta_loss` parameters added |
| v1.0 | **BREAKING:** `alpha`/`regularization` removed → use `alpha_W`, `alpha_H`, `l1_ratio` |
| v1.1 | Default `init` changed `'nndsvd'` → `'nndsvda'`; `MiniBatchNMF` added |
| v1.4 | `n_components='auto'` option added |

