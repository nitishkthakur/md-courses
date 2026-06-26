# Autoencoder-Based Dimensionality Reduction: The Complete Masterclass

> **Why this algorithm matters:** In 2006, Geoffrey Hinton and Ruslan Salakhutdinov published a
> four-page paper in *Science* showing that a deep neural network with a 30-neuron bottleneck
> could compress MNIST digits more meaningfully than PCA — not just lower reconstruction error,
> but genuinely better separation of digit classes in the compressed space. That paper restarted
> the deep learning revolution. The autoencoder was the vehicle. Today, variants of the
> autoencoder are used to generate new drug molecules, detect credit card fraud without a single
> fraud label, discover cell types from millions of single-cell genomes, compress images beyond
> JPEG quality, and power voice conversion that sounds indistinguishable from the target speaker.
> It is one of the most versatile architectures in all of machine learning — not because it is
> magic, but because the idea is elegant: *force the network to compress data through a narrow
> bottleneck, and the bottleneck is forced to learn what matters*.

---

## §0 — Python Ecosystem & Package Guide

The autoencoder landscape in Python is fragmented in a useful way: different packages dominate
different use cases, and choosing the right one saves you days of implementation work. Here is the
complete map, as of mid-2026.

### Complete Package Table

| Package | Import | Version (mid-2026) | What It Provides | License |
|---|---|---|---|---|
| `torch` | `import torch; import torch.nn as nn` | 2.4.x | Custom deep AE, VAE, all variants; full architectural control; GPU | BSD |
| `scikit-learn` | `from sklearn.neural_network import MLPRegressor` | 1.5.x | Shallow AE via regression trick; no GPU; prototyping only | BSD-3 |
| `pythae` | `from pythae.models import VAE, BetaVAE, WAE_MMD` | 0.1.2 | 19+ VAE variants with unified API; benchmarking; W&B integration | Apache-2.0 |
| `pyod` | `from pyod.models.auto_encoder import AutoEncoder` | 3.6.1 | AE + VAE for anomaly detection; automatic threshold; 60+ detectors | BSD-2 |
| `scvi-tools` | `import scvi` | 1.4.3 | VAE for single-cell RNA-seq; count likelihoods; batch correction | BSD-3 |
| `umap-learn` | `from umap.parametric_umap import ParametricUMAP` | 0.5.8 | Parametric UMAP: neural encoder trained on UMAP loss; out-of-sample | BSD-3 |
| `lightning` | `from lightning import LightningModule, Trainer` | 2.6.x | Training framework for production AE/VAE; auto GPU; checkpointing | Apache-2.0 |

### Top Picks — Recommendation Table

| Package | Best For | Modern / Legacy | GPU? |
|---|---|---|---|
| **`torch`** | Deep AE, VAE, beta-VAE, research, production | Modern | Yes |
| **`pythae`** | Comparing 19+ VAE variants systematically | Modern | Yes (via torch) |
| **`pyod`** | Anomaly detection with AE/VAE, production threshold | Modern | Yes (via keras) |
| **`scvi-tools`** | Single-cell genomics, count data, batch correction | Modern | Yes (via torch) |
| **`sklearn.MLPRegressor`** | Quick prototyping, shallow AE, no GPU available | Legacy-ish | No |

> ⚠️ **Pitfall:** sklearn's `MLPRegressor` is the *only* option here that does not have a native
> `.encode()` or `.transform()` method. When you call `ae.predict(X)`, you get the
> **reconstruction** — same shape as input — not the bottleneck representation. Getting the latent
> codes requires manually implementing a forward pass through encoder layers using `coefs_` and
> `intercepts_`. Every other package (torch, pythae, pyod, scvi-tools) gives you explicit
> encoder/decoder objects. Plan accordingly.

### The Same Autoencoder Across Top Packages

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load digits dataset (64-dim images of hand-written digits 0-9)
digits = load_digits()
X, y = digits.data, digits.target   # X: (1797, 64), float64

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

LATENT_DIM = 8   # compress 64 → 8 dimensions

# ── Option 1: sklearn MLPRegressor (shallow AE, prototyping) ─────────────────
from sklearn.neural_network import MLPRegressor

ae_sklearn = MLPRegressor(
    hidden_layer_sizes=(32, LATENT_DIM, 32),  # encoder: 64→32→8, decoder: 8→32→64
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42,
)
ae_sklearn.fit(X_train_sc, X_train_sc)   # X == y for autoencoder

# Extract bottleneck (encoder forward pass)
def encode_sklearn(model, X, n_encoder_layers=2):
    """Forward pass through encoder layers only (stops at bottleneck)."""
    act = {'relu': lambda x: np.maximum(0, x),
           'tanh': np.tanh,
           'logistic': lambda x: 1 / (1 + np.exp(-x)),
           'identity': lambda x: x}[model.activation]
    h = X
    for i in range(n_encoder_layers):
        h = h @ model.coefs_[i] + model.intercepts_[i]
        h = act(h)
    return h

Z_train_sk = encode_sklearn(ae_sklearn, X_train_sc, n_encoder_layers=2)
Z_test_sk  = encode_sklearn(ae_sklearn, X_test_sc,  n_encoder_layers=2)
print(f"sklearn latent shape: {Z_test_sk.shape}")   # (360, 8)

# ── Option 2: PyTorch (deep AE, production) ──────────────────────────────────
import torch
import torch.nn as nn

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = DeepAutoencoder(input_dim=64, latent_dim=LATENT_DIM).to(device)

X_tr = torch.tensor(X_train_sc, dtype=torch.float32).to(device)
X_te = torch.tensor(X_test_sc,  dtype=torch.float32).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

train_losses = []
for epoch in range(300):
    model.train()
    x_recon, z = model(X_tr)
    loss = criterion(x_recon, X_tr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

model.eval()
with torch.no_grad():
    Z_test_torch = model.encoder(X_te).cpu().numpy()
print(f"PyTorch latent shape: {Z_test_torch.shape}")   # (360, 8)

# ── Option 3: PyOD AutoEncoder (anomaly detection) ───────────────────────────
# pip install pyod
from pyod.models.auto_encoder import AutoEncoder  # verify in current docs

clf = AutoEncoder(
    hidden_neurons=[32, LATENT_DIM, LATENT_DIM, 32],  # symmetric bottleneck
    hidden_activation='relu',
    output_activation='sigmoid',
    epochs=100,
    batch_size=32,
    lr=1e-3,
    contamination=0.05,   # expected fraction of anomalies
    random_state=42,
)
clf.fit(X_train_sc)
anomaly_scores = clf.decision_scores_    # reconstruction error per training sample
y_pred = clf.predict(X_test_sc)         # 0=normal, 1=outlier

# ── Option 4: Parametric UMAP with AE loss ───────────────────────────────────
# pip install 'umap-learn[parametric_umap]'
# from umap.parametric_umap import ParametricUMAP  # verify in current docs
# embedder = ParametricUMAP(n_components=2, autoencoder_loss=True, n_epochs=200)
# Z_pumap = embedder.fit_transform(X_train_sc)
# Z_new   = embedder.transform(X_test_sc)   # out-of-sample! unlike standard UMAP
```

> 🔧 **In practice:** For any production system or research experiment, use PyTorch. The encoder
> and decoder are explicit `nn.Module` objects — you call `model.encoder(x)` to get the latent
> codes directly. The sklearn approach is fine for quick experiments but the manual `coefs_`
> gymnastics gets painful fast, and there's no dropout, no learning rate scheduling, and no GPU.

### GPU-Accelerated and Streaming Options

**GPU:** PyTorch (move tensors to `.to('cuda')`), pythae (PyTorch backend), pyod (Keras backend,
set `CUDA_VISIBLE_DEVICES`), scvi-tools (PyTorch via GPU-accelerated training with `.train()`).
sklearn has no GPU support.

**Streaming / Online:** sklearn's `MLPRegressor` supports `partial_fit()` for mini-batch online
training (SGD/Adam only, not LBFGS). PyTorch naturally supports online/streaming via custom
DataLoaders and incremental forward-backward passes.

**Large-Scale:** PyTorch Lightning (`lightning` package) wraps PyTorch training loops with
automatic multi-GPU, gradient accumulation, and model checkpointing — the right choice when your
dataset does not fit in memory or you need distributed training.

---

## §1 — The Origin: The Paper That Started It All

### Before Autoencoders: The Problem With PCA

In the early 1980s, the dominant tool for dimensionality reduction was Principal Component
Analysis — linear, interpretable, and elegant. PCA finds the hyperplane of maximum variance: the
directions along which the data spreads most. For many datasets, this is exactly what you want.
But PCA has a fundamental limitation: it is *linear*. If your data lives on a curved manifold —
a sphere, a Swiss roll, a helix — PCA cannot see the manifold's intrinsic structure. The first
principal component of a circle is not the circle's "direction" — there is no single direction.
PCA would see a blob.

By the mid-1980s, the parallel distributed processing (PDP) research group at UC San Diego,
led by Rumelhart, McClelland, and Hinton, was building a theory of how brains learn internal
representations. The question they posed: can a network learn compressed representations of its
inputs *without supervision* — without labels, without a target other than reconstructing what
it sees?

### The Original Paper

> 📜 **Origin/Citation:** Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). "Learning
> Internal Representations by Error Propagation." In Rumelhart & McClelland (Eds.), *Parallel
> Distributed Processing: Explorations in the Microstructure of Cognition*, Vol. 1, Ch. 8,
> pp. 318–362. MIT Press.

The core idea in Chapter 8 of the PDP volumes is disarmingly simple: take a feedforward network,
make the middle layer narrower than the input, and train it to output its own input. The
narrow middle layer — the **bottleneck** — must find a compressed representation that preserves
enough information to reconstruct the original. If the bottleneck has $k$ neurons and the input
has $p > k$ dimensions, the network is forced to discover the $k$ most informative directions in
the data.

What makes this possible is **backpropagation**, described formally in the companion Nature paper:

> 📜 Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). "Learning Representations by
> Back-propagating Errors." *Nature*, 323, 533–536. DOI: 10.1038/323533a0

Backprop gives the gradient of the reconstruction error with respect to every weight in the
network, including those in the bottleneck. Gradient descent on these weights teaches the network
what to compress and what to discard.

### The Formal Algorithm (Original)

Given training data $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ with $\mathbf{x}_i \in \mathbb{R}^p$:

1. **Define architecture:** Input layer ($p$ units) → hidden bottleneck ($k \ll p$ units) → output layer ($p$ units)
2. **Forward pass:** $\hat{\mathbf{x}}_i = g_\phi(f_\theta(\mathbf{x}_i))$
   - Encoder: $\mathbf{z}_i = f_\theta(\mathbf{x}_i) = \sigma(\mathbf{W}_e \mathbf{x}_i + \mathbf{b}_e)$
   - Decoder: $\hat{\mathbf{x}}_i = g_\phi(\mathbf{z}_i) = \mathbf{W}_d \mathbf{z}_i + \mathbf{b}_d$
3. **Loss:** $\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$
4. **Backprop:** Compute $\partial \mathcal{L} / \partial \theta$ and $\partial \mathcal{L} / \partial \phi$ via chain rule
5. **Update:** $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$, $\phi \leftarrow \phi - \eta \nabla_\phi \mathcal{L}$
6. **Repeat** steps 2–5 until convergence
7. **Extract representation:** $\mathbf{Z} = f_\theta(\mathbf{X})$ — the bottleneck activations

### The Historical Gap: 1986 to 2006

Here is something subtle about the history that most textbooks skip: the 1986 paper described
autoencoders and backpropagation, but *nobody successfully trained a deep autoencoder for nearly
two decades*. Shallow autoencoders (one hidden layer) worked fine. But deeper autoencoders with
3, 4, or 5 hidden layers consistently trained poorly — the gradients vanished before reaching the
early layers, and the network converged to useless solutions.

The field largely gave up on deep autoencoders and instead worked with shallow networks, kernel
methods, and eventually SVMs throughout the 1990s. Then came the 2006 landmark:

> 📜 **Citation:** Hinton, G.E., Salakhutdinov, R.R. (2006). "Reducing the Dimensionality of
> Data with Neural Networks." *Science*, 313(5786), pp. 504–507. DOI: 10.1126/science.1127647.
> PDF: https://www.cs.toronto.edu/~hinton/absps/science.pdf

Hinton and Salakhutdinov's insight: the reason deep autoencoders failed was not because they
were fundamentally flawed, but because *random weight initialization* placed the network in a bad
part of the loss landscape where gradient descent could not navigate. Their solution: use
**Restricted Boltzmann Machines (RBMs)** to pre-train each layer greedily — initializing the
weights near a meaningful solution — then fine-tune the entire autoencoder with backpropagation.

The result was remarkable. A 784→1000→500→250→2 autoencoder trained on MNIST found a 2D code
where the ten digit classes were visibly separated — with no labels, purely from reconstruction
pressure. The same architecture on a document dataset (20 Newsgroups) outperformed Latent
Semantic Analysis (PCA on the TF-IDF matrix) by a wide margin in information retrieval tasks.

> 💡 **Intuition:** Think of the RBM pre-training as finding a good starting valley in a
> mountainous landscape before you let gradient descent roll downhill. Once you're near a good
> basin, gradient descent can do the fine-tuning. This insight — that deep networks can be
> trained if initialized carefully — restarted the entire deep learning revolution.

Today we no longer need RBM pre-training. Better weight initialization schemes (Xavier/He init),
batch normalization, skip connections, and the ReLU activation (which resists vanishing gradients
far better than sigmoid) all solve the original training instability. But the 2006 paper remains
a landmark — it proved that deep representations are qualitatively different from and better than
shallow ones.

---

## §2 — The Algorithm, Deeply Explained

### The Core Architecture

An autoencoder is a neural network trained with reconstruction as its objective. It has two
conceptual halves:

$$
f_\theta : \mathbb{R}^p \to \mathbb{R}^d \quad \text{(encoder)}
$$
$$
g_\phi : \mathbb{R}^d \to \mathbb{R}^p \quad \text{(decoder)}
$$

where $p$ is the original dimensionality and $d \ll p$ is the bottleneck (latent) dimension. The
composed function $g_\phi \circ f_\theta$ is trained to approximate the identity: $g_\phi(f_\theta(\mathbf{x})) \approx \mathbf{x}$.

The loss is reconstruction error. For continuous inputs (MSE):

$$
\mathcal{L}_{\text{recon}}(\theta, \phi) = \frac{1}{n} \sum_{i=1}^n \|\mathbf{x}_i - g_\phi(f_\theta(\mathbf{x}_i))\|^2
$$

For binary inputs (binary cross-entropy):

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^p \left[ x_{ij} \log \hat{x}_{ij} + (1 - x_{ij}) \log(1 - \hat{x}_{ij}) \right]
$$

> 💡 **Intuition:** The autoencoder is playing a game of telephone with itself. Input → compress
> into $d$ numbers → reconstruct original from those $d$ numbers. The only way to win the game
> with a small $d$ is to figure out what's *essential* about the input. Noise, redundancy, and
> correlated features get factored out. The bottleneck forces a kind of information prioritization.

### Linear Autoencoders Recover PCA

This is one of the most important theoretical results in the autoencoder literature. If you use
*linear* activation functions (identity) in all layers and a single hidden layer, the
autoencoder's optimal solution is exactly the PCA subspace.

More precisely: with a linear encoder $\mathbf{z} = \mathbf{W}_e \mathbf{x}$ and linear decoder
$\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z}$, the encoder weights $\mathbf{W}_e$ at the global
minimum of MSE span the same subspace as the top-$d$ PCA eigenvectors. The individual weight
vectors are not necessarily the eigenvectors themselves (they can be any orthogonal basis for
that subspace), but the projected representations $\mathbf{Z}$ are equivalent to PCA projections.

**Proof sketch:** The optimal $\mathbf{W}_e^* \mathbf{W}_d^*$ is the rank-$d$ matrix that
minimizes the Frobenius norm $\|\mathbf{X} - \mathbf{X}\mathbf{W}_e\mathbf{W}_d\|_F^2$, which
by the Eckart-Young theorem is achieved by the truncated SVD — the PCA projection.

> 💡 **Intuition:** This means that when you add nonlinear activations (ReLU, tanh), you are
> strictly extending PCA. The bottleneck can now represent *curved* low-dimensional manifolds,
> not just hyperplanes. Every percentage point of reconstruction improvement over linear-AE
> reflects genuine nonlinear structure captured.

### Geometric Picture

Think about your data as a cloud of points in $p$-dimensional space. PCA finds a flat
$d$-dimensional hyperplane that the cloud is closest to — the "best flat floor" under the data.

An autoencoder finds the best *curved* $d$-dimensional surface. If the data truly lies on a
nonlinear manifold (a Swiss roll, a sphere, a face image manifold), the autoencoder can learn
to "unfold" it into the bottleneck coordinates. The encoder is the unfolding function; the
decoder is the re-embedding.

The latent space $\mathbf{Z} \in \mathbb{R}^{n \times d}$ is a coordinate chart on this learned
manifold. Nearby points in $\mathbb{R}^p$ that look similar should map to nearby points in
$\mathbb{R}^d$ — but this is only guaranteed approximately. Unlike Isomap or UMAP, the
autoencoder does not explicitly optimize for neighborhood preservation; it optimizes
reconstruction. Neighborhoods are preserved only insofar as it helps reconstruction.

### The Optimization

Both $\theta$ (encoder parameters) and $\phi$ (decoder parameters) are trained simultaneously
by minimizing $\mathcal{L}$ via stochastic gradient descent (usually Adam). The key operations:

**Forward pass:** Compute $\hat{\mathbf{x}}_i$ for each sample in the mini-batch.

**Loss computation:** Compute MSE (or BCE) between $\mathbf{x}_i$ and $\hat{\mathbf{x}}_i$.

**Backward pass:** Backpropagate gradients through the decoder, through the bottleneck, and
through the encoder. The bottleneck is not special — gradients flow through it just like any
other layer.

**Update:** Apply Adam update to all weights in $\theta$ and $\phi$.

### Computational Complexity

Let $L$ = number of layers per encoder, $w$ = maximum layer width, $n$ = number of samples,
$p$ = input dimension, $d$ = bottleneck dimension.

**Training complexity:** $\mathcal{O}(n \cdot L \cdot w^2)$ per epoch. Each epoch requires a
forward and backward pass through all layers for all samples. The $w^2$ factor comes from the
matrix multiplications at each fully-connected layer.

**Inference (encoding) complexity:** $\mathcal{O}(n \cdot L \cdot w \cdot p)$ for the full
forward pass through the encoder. For a single new point: $\mathcal{O}(L \cdot w \cdot p)$ —
a constant-time operation regardless of training set size. This is a major advantage over
non-parametric methods like t-SNE (which must recompute the full affinity matrix) or standard
UMAP (which requires approximate nearest-neighbor search over the training set).

**Memory:** $\mathcal{O}(L \cdot w^2)$ for weights. For a typical tabular AE (input=100,
hidden=[256, 128, 32, 128, 256], output=100), total parameters ≈ 100×256 + 256×128 + 128×32 +
32×128 + 128×256 + 256×100 ≈ 182K. Tiny.

### Assumptions the Autoencoder Makes

1. **Low-dimensional manifold hypothesis:** The data lives near a $d$-dimensional manifold
   in $\mathbb{R}^p$. If the data is genuinely $p$-dimensional (no manifold structure), the AE
   will have high reconstruction error for any small $d$.

2. **Smoothness:** Nearby inputs should map to nearby latent codes. The autoencoder learns a
   *continuous* mapping, so this is implicit. But it is not explicitly enforced — a standard AE
   can learn a discontinuous-looking latent space with holes and gaps.

3. **Reconstruction as sufficient proxy for representation:** We train the AE to reconstruct
   inputs, and hope the bottleneck learns a useful representation. This proxy can fail: an AE can
   learn to reconstruct well while learning latent codes that are not useful for downstream tasks.
   For example, it might memorize local texture statistics and fail to capture class-level structure.

4. **Feature scale equivalence:** MSE treats all features equally. A feature with variance 1000
   and a feature with variance 0.01 are weighted equally in the loss. This means the AE will
   prioritize reconstructing high-variance features. Always standardize features before fitting.

### Where the Assumptions Break

- **Discrete or count data (genomics, text):** MSE is the wrong loss. Use BCE for binary, or
  negative binomial likelihood (scVI) for count data. Applying a Gaussian-likelihood AE to raw
  gene expression counts produces systematically poor representations.
- **Very high-dimensional, sparse data (text BoW):** The AE spends most of its capacity
  reconstructing zeros. Better: use NMF (for count/non-negative), LSA (linear), or a VAE with
  multinomial likelihood (Mult-VAE for collaborative filtering).
- **Very small datasets (n < 500):** The AE will overfit. The decoder can memorize the training
  set without the encoder learning a meaningful representation. Use PCA instead.
- **Data with no low-dimensional structure:** If the intrinsic dimensionality is close to $p$,
  any bottleneck will lose information. Reconstruction error will be high, and downstream task
  performance will suffer.

---

## §3 — The Full Evolution: Major Variants

### 3.1 Undercomplete Autoencoder (The Baseline)

The standard architecture described in §2. Bottleneck dimension $d < p$. The compression
constraint forces the encoder to learn a compact representation.

**When to use:** Feature extraction for downstream tasks; initialization for clustering;
out-of-sample embedding for streaming data.

**Key limitation:** Without additional regularization, a sufficiently wide or deep undercomplete
AE can still learn degenerate solutions (e.g., mapping everything to a small region of the
bottleneck). The bottleneck alone is not always sufficient regularization.

---

### 3.2 Denoising Autoencoder (DAE)

> 📜 Vincent, P., Larochelle, H., Bengio, Y., Manzagol, P.A. (2008). "Extracting and Composing
> Robust Features with Denoising Autoencoders." *ICML 2008*, pp. 1096–1103.
> https://dl.acm.org/doi/10.1145/1390156.1390294

**The problem it solves:** A standard AE can learn to cheat — if the bottleneck is wide enough
relative to the data complexity, it can learn near-identity mappings without discovering
meaningful structure. Even with an undercomplete bottleneck, the AE is only exposed to
clean data and may learn brittle features.

**The algorithmic change:** Corrupt the input before feeding it to the encoder, then train the
decoder to reconstruct the *original clean* input:

$$
\tilde{\mathbf{x}}_i \sim q_\mathcal{D}(\tilde{\mathbf{x}} | \mathbf{x}_i)
$$
$$
\mathcal{L}_{\text{DAE}} = \frac{1}{n} \sum_{i=1}^n \|\mathbf{x}_i - g_\phi(f_\theta(\tilde{\mathbf{x}}_i))\|^2
$$

Common corruption strategies:
- **Gaussian noise:** $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$
- **Masking (dropout) noise:** Randomly zero out a fraction $\nu$ of input features: $\tilde{x}_j = x_j \cdot b_j$, $b_j \sim \text{Bernoulli}(1-\nu)$
- **Salt-and-pepper:** Randomly set features to 0 or 1 (for binary inputs)

**Why it works — the deep result:** Minimizing the DAE loss is mathematically equivalent to
estimating the score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ — the gradient of the
log data density. This is the same quantity estimated by score-matching methods and is
connected to diffusion models. The DAE learns to "denoise" by pushing points toward regions of
high data probability. This gives the learned representation a genuine probabilistic grounding.

**In practice:** Masking with $\nu = 0.3$ (mask 30% of features) is a robust default. This
adds strong regularization pressure without requiring careful noise level tuning.

```python
import torch
import torch.nn as nn

def add_masking_noise(x, mask_fraction=0.3):
    """Randomly zero out mask_fraction of features."""
    mask = (torch.rand_like(x) > mask_fraction).float()
    return x * mask

# DAE training loop
for epoch in range(300):
    model.train()
    x_corrupted = add_masking_noise(X_tr, mask_fraction=0.3)
    x_recon, z = model(x_corrupted)
    loss = criterion(x_recon, X_tr)   # reconstruct CLEAN from corrupted
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**When to prefer:** Whenever you have noisy data, incomplete features, or want more robust
representations that transfer to downstream tasks. DAE almost always outperforms a standard AE
of the same architecture for representation learning quality.

---

### 3.3 Sparse Autoencoder

**The problem it solves:** The undercomplete AE forces compression by architecture. But what
if you want an *overcomplete* bottleneck (latent dim > input dim) that still learns useful
representations? You need a different pressure: sparsity.

**The algorithmic change:** Add a sparsity penalty on bottleneck activations:

$$
\mathcal{L}_{\text{sparse}} = \mathcal{L}_{\text{recon}} + \beta_s \sum_{j=1}^d \text{KL}(\rho \| \hat{\rho}_j)
$$

where $\rho$ is a target sparsity (e.g., 0.05 — each neuron should be active 5% of the time),
$\hat{\rho}_j = \frac{1}{n}\sum_i z_{ij}$ is the average activation of neuron $j$, and
$\text{KL}(\rho \| \hat{\rho}_j) = \rho \log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}$ is the KL divergence between Bernoulli distributions. Alternatively, use an L1 penalty: $\mathcal{L}_{\text{sparse}} = \mathcal{L}_{\text{recon}} + \lambda \sum_j |z_j|$.

**Why it works:** Each latent neuron is forced to be selective — active for only a small fraction
of inputs. This produces *dictionary-like* representations where each latent dimension codes a
specific feature (e.g., a specific edge orientation in images, a specific co-occurring word
pattern in text). Sparse autoencoders have recently become prominent in mechanistic
interpretability research for large language models (Anthropic's work on feature decomposition).

**When to prefer:** When you want interpretable latent features; when the bottleneck dimension
is larger than the input (overcomplete dictionary learning); when studying neural network
representations via LLM probing.

---

### 3.4 Contractive Autoencoder (CAE)

> 📜 Rifai, S., Vincent, P., Muller, X., Glorot, X., Bengio, Y. (2011). "Contractive
> Autoencoders: Explicit Invariance During Feature Extraction." *ICML 2011*, pp. 833–840.

**The algorithmic change:** Add a penalty on the Frobenius norm of the Jacobian of the encoder:

$$
\mathcal{L}_{\text{CAE}} = \mathcal{L}_{\text{recon}} + \lambda_c \left\| \frac{\partial f_\theta(\mathbf{x})}{\partial \mathbf{x}} \right\|_F^2
$$

**Why it works:** The Jacobian norm measures how sensitive the latent code is to small
perturbations of the input. Penalizing this forces the encoder to learn representations that
are *locally invariant* — robust to small input variations. The learned manifold is explicitly
smooth. CAE is theoretically elegant but computationally expensive (computing Jacobians requires
$d \times p$ backward passes per sample). Rarely used in practice today — denoising AE achieves
similar benefits more efficiently.

---

### 3.5 Variational Autoencoder (VAE)

> 📜 Kingma, D.P., Welling, M. (2014). "Auto-Encoding Variational Bayes." *ICLR 2014*.
> arXiv:1312.6114. https://arxiv.org/abs/1312.6114

This is the most important variant. The VAE replaces the deterministic bottleneck with a
probabilistic one — and in doing so, gains the ability to *generate* new data points, produces
a smoother and more regular latent space, and provides a principled probabilistic framework.

**The core insight:** In a standard AE, the encoder maps each input to a *point* in latent
space: $\mathbf{z}_i = f_\theta(\mathbf{x}_i)$. This means the latent space can be sparsely
populated — there are large gaps between the training points' latent codes, and decoding a
point in the gap produces garbage. The VAE fixes this by making the encoder map each input to
a *distribution* over latent space.

**The model:**

Encoder: $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2(\mathbf{x})))$

The encoder network outputs two vectors: $\boldsymbol{\mu} \in \mathbb{R}^d$ and $\boldsymbol{\sigma} \in \mathbb{R}^d_{>0}$ (or equivalently $\log \boldsymbol{\sigma}^2$).

Prior: $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$

Decoder: $p_\phi(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\hat{\mathbf{x}}; g_\phi(\mathbf{z}), \sigma_x^2 \mathbf{I})$

**The ELBO loss:**

The VAE is trained to maximize the Evidence Lower BOund (ELBO):

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\phi(\mathbf{x}|\mathbf{z})]}_{\text{reconstruction term}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL regularization term}}
$$

The first term rewards accurate reconstruction. The second term penalizes the posterior
$q_\phi(\mathbf{z}|\mathbf{x})$ from deviating from the unit Gaussian prior — this is what
fills in the gaps and makes the latent space regular.

**The reparameterization trick:** We need to backpropagate through the sampling step
$\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$. Sampling is not differentiable. The trick:

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Now $\mathbf{z}$ is a deterministic function of $\boldsymbol{\mu}$, $\boldsymbol{\sigma}$, and
the fixed noise $\boldsymbol{\epsilon}$. Gradients flow through $\boldsymbol{\mu}$ and
$\boldsymbol{\sigma}$ cleanly.

**Closed-form KL:** For Gaussian encoder and Gaussian prior, the KL divergence has a closed form:

$$
D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \| \mathcal{N}(\mathbf{0}, \mathbf{I})) = \frac{1}{2} \sum_{j=1}^d \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

```python
class VAE(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(16, latent_dim)
        self.fc_log_var = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def encode(self, x):
        h = self.encoder_base(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

def vae_loss(x_recon, x, mu, log_var, beta=1.0):
    recon = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + beta * kl, recon, kl
```

**For DR specifically:** Use $\boldsymbol{\mu}$ (not the sampled $\mathbf{z}$) as the latent
representation. The mean is the most likely code for each input. The sampled $\mathbf{z}$ adds
noise useful for training but not for embeddings.

---

### 3.6 beta-VAE (Disentangled Representations)

> 📜 Higgins, I. et al. (2017). "beta-VAE: Learning Basic Visual Concepts with a Constrained
> Variational Framework." *ICLR 2017*.

A single hyperparameter change with profound consequences. Replace the VAE loss with:

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi}[\log p_\phi(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

With $\beta > 1$, the KL term is weighted more heavily, pushing each latent dimension to encode
exactly one independent factor of variation. The result: a *disentangled* latent space where
dimension 1 might correspond to shape, dimension 2 to orientation, dimension 3 to scale, etc.
Latent traversal (varying one $z_j$ while holding others fixed, then decoding) produces clean,
isolated attribute changes.

**Tradeoff:** Higher $\beta$ → more disentanglement, worse reconstruction quality (blurrier
outputs). $\beta = 1$ recovers the standard VAE. Typical range: $\beta \in [2, 10]$.

**When to prefer:** When you need interpretable latent axes — drug discovery (latent dims
correspond to chemical properties), biology (latent dims correspond to cell types), generative
design (controllable attribute manipulation).

---

### 3.7 Wasserstein Autoencoder (WAE)

> 📜 Tolstikhin, I., Bousquet, O., Gelly, S., Schölkopf, B. (2018). "Wasserstein Auto-Encoders."
> *ICLR 2018*. https://openreview.net/forum?id=HkL7n1-0b

**The problem it solves:** The VAE's per-sample KL divergence can lead to blurry
reconstructions (the decoder must generate outputs for the entire spread of $q_\phi(\mathbf{z}|\mathbf{x})$)
and posterior collapse (the encoder collapses to the prior, contributing no information).

**The insight:** Instead of penalizing the per-sample posterior $q_\phi(\mathbf{z}|\mathbf{x})$
from the prior, penalize the *aggregate* posterior $q_\mathcal{Z} = \int q_\phi(\mathbf{z}|\mathbf{x}) p_{\text{data}}(\mathbf{x}) \, d\mathbf{x}$
from the prior. This is grounded in optimal transport (Wasserstein distance) theory.

**WAE-MMD loss:**
$$
\mathcal{L}_{\text{WAE}} = \frac{1}{n}\sum_i \|\mathbf{x}_i - g_\phi(\mathbf{z}_i)\|^2 + \lambda \cdot \text{MMD}(q_\mathcal{Z} \| p(\mathbf{z}))
$$

where MMD (Maximum Mean Discrepancy) with an RBF kernel measures the distance between the
aggregate encoding distribution and the prior. WAE produces sharper reconstructions than VAE
and avoids posterior collapse. Available in `pythae` as `WAE_MMD`.

---

### 3.8 Parametric UMAP

> 📜 Sainburg, T., McInnes, L., Gentner, T.Q. (2021). "Parametric UMAP Embeddings for
> Representation and Semisupervised Learning." *Neural Computation*, 33(11), 2881–2907.
> https://pmc.ncbi.nlm.nih.gov/articles/PMC8516496/

**The problem it solves:** Standard UMAP produces an embedding for the training set but has
no parametric function to embed new points. Every new point requires rerunning the UMAP
graph-construction and optimization — expensive.

**The algorithmic change:** Train a neural network encoder to minimize the same graph-layout
loss that UMAP optimizes. The encoder learns a function $f_\theta : \mathbb{R}^p \to \mathbb{R}^d$
that can instantly embed any new point via a forward pass. The autoencoder variant adds a
decoder and jointly trains on UMAP loss + reconstruction loss.

**Key advantage:** Out-of-sample embedding in $\mathcal{O}(L \cdot w \cdot p)$ time per point
(a single forward pass), compared to $\mathcal{O}(n \cdot \log n)$ for rerunning UMAP.
Also enables streaming embeddings and semi-supervised extension (add cross-entropy loss on
labeled samples).

**When to prefer:** Any real-time or production setting where new points arrive continuously
and must be embedded without rerunning the full algorithm.

---

### 3.9 Contrastive Learning (SimCLR, MoCo)

> 📜 Chen, T. et al. (2020). "A Simple Framework for Contrastive Learning of Visual
> Representations." *ICML 2020*. arXiv:2002.05709. https://github.com/google-research/simclr

**The problem it solves:** Reconstruction-based AEs must preserve every pixel/feature of the
input, even irrelevant ones (background, lighting, camera noise). This wastes representation
capacity on irrelevance and limits representation quality.

**The insight:** You do not need a decoder. Instead, train an encoder so that different
*augmented views* of the same input map to similar representations, while different inputs
map to dissimilar representations. The NT-Xent (Normalized Temperature-scaled Cross Entropy)
loss for a batch of $N$ samples with $2N$ augmented views:

$$
\mathcal{L}(i,j) = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}
$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^T \mathbf{v} / (\|\mathbf{u}\| \|\mathbf{v}\|)$ is cosine similarity and $\tau$ is temperature.

**Why contrastive beats reconstructive for vision:** Images have massive low-level
variability (pixel-level noise, lighting) that is semantically meaningless. A reconstruction
AE must model this. A contrastive encoder discards it by construction — two crops of the same
image are positive pairs regardless of crop position, color jitter, etc.

**When to prefer:** Self-supervised pretraining on images or audio; when you have large
unlabeled datasets; when downstream tasks care about semantic similarity, not reconstruction.
**Not suitable for tabular DR** — the augmentation strategy must be domain-appropriate.

---

## §4 — Hyperparameters: The Complete Guide

### 4.1 Latent Dimension (Bottleneck Size)

**What it controls:** The number of information channels allowed between encoder and decoder —
the fundamental trade-off between compression and reconstruction quality.

**Mechanism:** Smaller $d$ = higher compression = more information discarded. Larger $d$ =
lower compression = more reconstruction fidelity but less meaningful compression.

**Default:** No default — you must choose. A common starting point is $d = \lceil \sqrt{p} \rceil$,
so for 64-dim digits: start at $d = 8$.

**Tuning strategy:**
1. Fit an AE with several values of $d$: $[2, 4, 8, 16, 32, 64]$
2. Plot validation reconstruction loss vs. $d$ on a log-log scale
3. Look for an **elbow** — the point where loss stops dropping sharply. This approximates the
   data's intrinsic dimensionality
4. As a cross-check: run PCA and note how many components explain 95% of variance — treat this
   as an upper bound on meaningful $d$

**Too small:** High reconstruction error. Downstream task performance drops. Blurry/distorted
reconstructions. Features systematically poorly reconstructed (check per-feature MSE).

**Too large:** Near-identity mapping. Latent dimensions have low variance. Out-of-sample
reconstruction not better than training set (no compression pressure). Downstream task accuracy
plateaus.

```python
import optuna

def objective(trial):
    latent_dim = trial.suggest_categorical("latent_dim", [2, 4, 8, 16, 32])
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    alpha      = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)

    from sklearn.neural_network import MLPRegressor
    hidden = (32, latent_dim, 32)
    ae = MLPRegressor(
        hidden_layer_sizes=hidden, activation='relu', solver='adam',
        learning_rate_init=lr, alpha=alpha, max_iter=500,
        early_stopping=True, n_iter_no_change=20, random_state=42
    )
    ae.fit(X_train_sc, X_train_sc)
    X_recon_val = ae.predict(X_val_sc)
    val_mse = np.mean((X_val_sc - X_recon_val) ** 2)
    return val_mse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print(f"Best latent_dim: {study.best_params['latent_dim']}")
```

---

### 4.2 Architecture (Depth and Width)

**What it controls:** The expressiveness of the encoder and decoder functions.

**Depth (number of encoder layers):** More layers = more complex nonlinear mappings. For
tabular data with <200 features, 2–3 encoder layers is usually sufficient. For images, 4–6
layers of convolutions. Going deeper beyond this rarely helps for DR and increases training
time and overfitting risk.

**Width (neurons per layer):** Should decrease monotonically from input to bottleneck
(hourglass shape). A geometric taper works well: each layer ≈ half the previous. For input
dimension $p$: try $[p/2, p/4, d]$ for a 3-layer encoder.

**In sklearn:** Specify as `hidden_layer_sizes=(w1, w2, ..., d, ..., w2, w1)` where the middle
element is the bottleneck. Decoder layers are mirrored (symmetric architecture).

**Too deep:** Vanishing gradients on early encoder layers; loss barely moves in first epochs.
Solution: use batch normalization, residual connections, or GELU activations (in PyTorch).

**Too wide:** Overfitting on small datasets. High memory usage. Solution: increase L2 alpha.

---

### 4.3 Activation Function

**What it controls:** The nonlinearity at each hidden unit. This determines what manifolds the
AE can represent.

| Activation | Default in | Best For | Notes |
|---|---|---|---|
| `relu` | sklearn | Tabular data, deep nets | Fast, sparse activations; can cause dead neurons |
| `tanh` | — | When outputs in $[-1,1]$; RNNs | Saturates for large inputs; slower than relu |
| `logistic` (sigmoid) | — | Binary reconstruction targets | Use in output layer only for binary data |
| `identity` | — | Linear AE (recovers PCA) | No nonlinearity = linear mapping |
| `gelu` | PyTorch | Transformers, modern deep nets | Smoother than relu, often better for depth |

**Rule:** Use `relu` as default for encoder hidden layers. For the decoder output layer: `identity` for continuous inputs (sklearn default), `sigmoid` for binary. Never use `sigmoid`/`tanh` in deep encoder hidden layers — saturating activations cause vanishing gradients.

---

### 4.4 Learning Rate

**What it controls:** Step size for the Adam optimizer per update.

**Default (sklearn):** 0.001. **Default (Adam PyTorch):** 0.001.

**Tuning range:** Log-uniform in $[10^{-5}, 10^{-2}]$.

**Too high (>0.01):** Training loss oscillates; never converges; reconstruction quality is noisy.

**Too low (<1e-5):** Training is glacially slow; effectively no learning in reasonable epochs.

**Best practice (PyTorch):** Use cosine annealing:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
```
Or reduce-on-plateau:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
```

---

### 4.5 L2 Regularization (alpha / weight_decay)

**What it controls:** Penalty on large weight magnitudes — prevents overfitting, encourages
smooth encoder functions.

**Default (sklearn alpha):** 0.0001. **PyTorch equivalent:** `weight_decay` in `Adam`.

**Too high:** Underfitting — all weights near zero, high reconstruction error across train and
test set uniformly.

**Too low / zero:** Overfitting — training MSE << test MSE; latent space memorizes training set.

**Tuning range:** Log-uniform in $[10^{-6}, 10^{-2}]$.

---

### 4.6 Batch Size

**What it controls:** Number of samples per gradient update.

**Default (sklearn):** `'auto'` = min(200, n\_samples).

**For tabular AE:** 128–512 is the practical range. For very small datasets (n<500): full-batch
(batch\_size = n\_samples). For SimCLR: 1024–4096 required (more negatives = better).

**Large batch → smoother gradients, faster epochs, may converge to flatter minima**.
**Small batch → noisier gradients (regularizing effect), often better generalization**.

---

### 4.7 Epochs (max\_iter)

**Default (sklearn):** 200. **This is almost always insufficient for deep AEs.**

**Recommended:** Set `max_iter=2000` and use `early_stopping=True` with `n_iter_no_change=30`.
Always plot `ae.loss_curve_` after training. If loss is still decreasing at the final iteration,
you stopped too early.

**Diagnostic:**
```python
import matplotlib.pyplot as plt
plt.plot(ae.loss_curve_)
plt.xlabel("Epoch"); plt.ylabel("Training MSE")
plt.title("Autoencoder Training Loss Curve")
plt.yscale('log')
plt.show()
```

---

### 4.8 KL Weight Beta (VAE-specific)

**What it controls:** The balance between reconstruction fidelity and latent space regularity.

**Standard VAE:** $\beta = 1.0$ (equal weighting of reconstruction and KL terms).

**beta-VAE:** $\beta > 1$ for disentanglement. Range: 2–10 typical.

**Too high:** Blurry reconstructions; posterior collapse risk; latent space well-regularized
but encoding little information. **Too low:** Sharp reconstructions; entangled, irregular latent
space; poor generation quality.

**KL annealing trick:** Start $\beta = 0$ and linearly increase to 1.0 over the first 50–100
epochs. This allows reconstruction loss to stabilize before KL pressure is applied, significantly
reducing posterior collapse.

```python
for epoch in range(n_epochs):
    beta = min(1.0, epoch / 50)   # linear warmup over 50 epochs
    x_recon, mu, log_var = vae(X_batch)
    loss, recon, kl = vae_loss(x_recon, X_batch, mu, log_var, beta=beta)
```

---

### 4.9 Corruption Level (DAE-specific)

**What it controls:** How much input is corrupted before encoding.

**Masking fraction:** 0.1–0.5 of features. Start at 0.3.
**Gaussian noise std:** 0.1–0.5 times feature std.

**Too high:** AE must reconstruct from almost no information; training diverges.
**Too low:** Essentially a standard AE; no regularization benefit.

---

### Hyperparameter Tuning Playbook

| Parameter | Default | Search Space | Direction | Key Interaction |
|---|---|---|---|---|
| `latent_dim` | — | `[2, 4, 8, 16, 32, 64]` categorical | ↑ = less compression | Primary knob; tune first |
| Architecture depth | 1–2 layers | `[1, 2, 3, 4]` int | ↑ = more expressive | Interact with L2 alpha |
| Layer width | — | `[32, 64, 128, 256]` categorical (first hidden) | ↑ = more capacity | Interact with depth |
| `learning_rate_init` | 0.001 | $[10^{-5}, 10^{-2}]$ log | — | Check loss curve convergence |
| `alpha` (L2) | 0.0001 | $[10^{-6}, 10^{-2}]$ log | ↑ = more regularization | Interact with n_samples |
| `batch_size` | auto | `[32, 64, 128, 256, 512]` categorical | ↑ = faster epochs | Interact with LR |
| `max_iter` | 200 | 500–2000 with early stopping | ↑ = more training | Use `loss_curve_` to check |
| `beta` (VAE) | 1.0 | $[0.5, 10]$ log | ↑ = more disentanglement | Tradeoff vs. recon quality |
| `mask_fraction` (DAE) | — | $[0.1, 0.5]$ uniform | ↑ = harder denoising | Tradeoff vs. training stability |

---

## §5 — Strengths

### 5.1 Handles Nonlinear Manifolds

The fundamental strength: autoencoders can learn *curved* low-dimensional manifolds, not just
flat hyperplanes. PCA fails on a Swiss roll; a deep AE does not. This matters in every domain
where the natural structure of data is nonlinear — face image variation is nonlinear (rotation,
expression), molecular conformation space is nonlinear, cell type differentiation trajectories
are nonlinear.

**Mechanism:** Nonlinear activations (ReLU, GELU) give the encoder the capacity to warp the
input space, folding and stretching it so that the manifold lies along the coordinate axes of
latent space.

### 5.2 Parametric Mapping (Instant Out-of-Sample Embedding)

Once trained, encoding a new point is a single forward pass: $\mathcal{O}(L \cdot w \cdot p)$.
This is independent of $n$ (the training set size). Compare to t-SNE (no principled
out-of-sample method) or standard UMAP (requires rerunning neighbor graph construction).

**Mechanism:** The encoder $f_\theta$ is a learned function, not a table lookup. New points
are handled by the same function.

### 5.3 Scalable to Very High Dimensions

MLPs scale to millions of input features (with appropriate sparse layers). Convolutional AEs
scale to high-resolution images by exploiting local structure. This is impossible for kernel
methods (which require an $n \times n$ matrix) or non-parametric manifold methods.

### 5.4 Unified Framework for Multiple Tasks

The same trained model supports: dimensionality reduction (bottleneck), reconstruction/denoising
(decoder), anomaly detection (reconstruction error), and — for VAEs — generation. One training
run supports all downstream uses.

### 5.5 Domain Adaptable

By changing the loss function, you adapt the AE to different data types: MSE for continuous,
BCE for binary, negative binomial for count data (scVI), spectral loss for audio, perceptual
loss for images. The architecture stays the same; only the loss changes.

---

## §6 — Weaknesses & Failure Modes

### 6.1 Posterior Collapse in VAEs

**Symptom:** The KL divergence term collapses to ~0 during training. The decoder learns to
ignore the latent variable $\mathbf{z}$ and becomes a standalone generative model. Sampling
$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and decoding produces the same blurry
average output regardless of $\mathbf{z}$.

**Detection:** Monitor KL term during training. If it drops below 0.01 per dimension and stays
there, you have collapse.

**Cause:** The decoder is powerful enough to model $p(\mathbf{x})$ without conditioning on $\mathbf{z}$.
The optimizer finds the easy solution of setting $q_\phi(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z})$.

**Mitigation:** KL annealing; Free Bits (enforce minimum KL per dimension); reduce decoder
capacity; use WAE-MMD instead.

### 6.2 No Neighborhood Preservation Guarantee

**Symptom:** The 2D AE embedding looks like a random cloud — no visible cluster structure —
even though the data has clear clusters.

**Mechanism:** AEs optimize reconstruction, not neighborhood structure. Two similar inputs might
map to distant latent codes if reconstruction pressure doesn't require proximity. t-SNE and UMAP
explicitly optimize for neighborhood preservation.

**Detection:** Compute trustworthiness score (`sklearn.manifold.trustworthiness`). Values below
0.8 suggest poor neighborhood structure.

**Mitigation:** Use Parametric UMAP (optimizes neighborhood loss in the encoder). Or post-process:
apply t-SNE on the AE bottleneck codes (use AE as a pre-processor for t-SNE, which dramatically
speeds up t-SNE on high-dimensional data).

### 6.3 Blurry Reconstructions in VAEs

**Symptom:** Reconstructed images are blurry even when reconstruction loss is low.

**Mechanism:** The VAE decoder models $p(\mathbf{x}|\mathbf{z})$ as a Gaussian, and MSE loss
is the Gaussian log-likelihood. Averaging over the posterior uncertainty $q_\phi(\mathbf{z}|\mathbf{x})$
tends to produce average (blurry) outputs.

**Mitigation:** Use a perceptual loss (VGG feature matching) instead of pixel-wise MSE; use
a GAN decoder (VAE-GAN); use WAE-MMD (produces sharper outputs than VAE at same architecture);
increase decoder capacity.

### 6.4 Requires Careful Preprocessing

**Symptom:** One feature dominates the reconstruction loss; other features are ignored.

**Mechanism:** MSE treats all features equally in absolute scale. A feature with range [0, 1000]
contributes 10^6 times more to MSE than a feature in [0, 1].

**Detection:** Compute per-feature reconstruction MSE. Features with high individual MSE but
low variance in original space are ignored.

**Mitigation:** Always apply `StandardScaler` before fitting. For count data (genomics): use
log1p normalization or specialized likelihoods.

### 6.5 Overfitting on Small Datasets

**Symptom:** Training MSE is excellent; test MSE is 10–100× worse.

**Mechanism:** The encoder-decoder has millions of parameters but only hundreds of training
samples. The network memorizes the training set.

**Detection:** Compare `ae.loss_` (training) to reconstruction MSE computed on test set.

**Mitigation:** Increase L2 regularization (`alpha`); add dropout (PyTorch only); use denoising
AE; reduce network size; use `early_stopping=True`.

### 6.6 Latent Space Axes are Not Interpretable by Default

**Symptom:** Each latent dimension is a mix of multiple original features with no clear meaning.

**Mechanism:** Unlike PCA (which maximizes variance along orthogonal directions with a
mathematical interpretation) or NMF (which enforces non-negativity and produces part-based
representations), standard AEs have no interpretability constraint on individual latent dimensions.

**Mitigation:** Use beta-VAE for disentangled axes; apply ICA or PCA on the latent codes
post-hoc to rotate them toward interpretable directions.

---

## §7 — What I Think About When Fitting

This is the internal monologue I run every time I train an autoencoder. Follow this checklist
and you will avoid 90% of the common mistakes.

### Before Fitting

**On the data:**
- What is $p$? If $p < 50$, ask yourself whether PCA would suffice. The AE overhead is
  only justified if you expect the manifold to be nonlinear.
- Check for missing values — autoencoders require complete inputs. Impute first, or explicitly
  design a masking strategy.
- Check feature scales. Run `X.std(axis=0).describe()` — if the range of standard deviations
  spans more than 10×, standardize. Always standardize.
- For count data (RNA-seq, word counts): log1p transform or use count-appropriate likelihood.
- What is my downstream task? Clustering? Classification? Generation? Anomaly detection? This
  determines the correct variant:
  - Anomaly detection → undercomplete AE or VAE via pyod
  - Generation/sampling → VAE (pythae)
  - Interpretable factors → beta-VAE
  - Streaming/online embedding → Parametric UMAP or any AE
  - Noisy/incomplete input → DAE

**On architecture:**
- Start from PCA as a benchmark. Run PCA and note how many components explain 95% variance.
  This is your upper bound on meaningful latent dim.
- For sklearn: remember `hidden_layer_sizes` includes the bottleneck. A symmetric 64→32→8→32→64
  AE is `hidden_layer_sizes=(32, 8, 32)` — sklearn adds the input (64) and output (64) layers
  automatically.
- Choose `early_stopping=True` and `n_iter_no_change=30` — always. The default 200 iterations
  is insufficient.
- Set `random_state=42` for reproducibility.

### During Fitting

**Watch the loss curve:**
- After fitting sklearn: `plt.plot(ae.loss_curve_)`. It should decrease monotonically and
  plateau. If it's still steep at the end, increase `max_iter`.
- Spikes in the loss curve → learning rate too high; reduce `learning_rate_init` by 3–5×.
- Loss not moving in first 10 epochs → learning rate too low or vanishing gradients.
- For VAE: plot reconstruction term and KL term separately. KL should increase gradually from 0.
  If KL → 0 and stays there, you have posterior collapse — apply KL annealing.

**Check gradient flow (PyTorch):**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```
Encoder early layers should have non-trivial gradient norms (>1e-4). Near-zero gradients in
early layers = vanishing gradient problem.

### After Fitting

**Reconstruction quality check:**
```python
X_recon = ae.predict(X_test_sc)
per_sample_mse = np.mean((X_test_sc - X_recon)**2, axis=1)
per_feat_mse   = np.mean((X_test_sc - X_recon)**2, axis=0)
print(f"Mean test MSE: {per_sample_mse.mean():.4f}")
print(f"Worst feature: {per_feat_mse.argmax()}, MSE={per_feat_mse.max():.4f}")
```
- Look at the 5 worst-reconstructed samples. Do they look like outliers or noise? If yes,
  your AE is working correctly. If they look like typical samples, your bottleneck is too small.
- Per-feature MSE: identify which features are poorly reconstructed. These are features where
  the AE finds compression difficult — often the most informative ones.

**Latent space structure:**
```python
Z = encode_sklearn(ae, X_test_sc)   # shape (n_test, d)
# If d > 2: apply PCA or t-SNE on Z for visualization
from sklearn.decomposition import PCA
Z_2d = PCA(n_components=2).fit_transform(Z)
plt.scatter(Z_2d[:,0], Z_2d[:,1], c=y_test, cmap='tab10', alpha=0.7)
```
- Do class labels (if available) form coherent clusters in the latent space? If yes, the AE
  has learned class-discriminative structure unsupervised — excellent.
- If the latent space is one amorphous blob regardless of class: the bottleneck may be too
  small, or the AE needs more capacity.

**Downstream task benchmark:**
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1, max_iter=1000)
lr.fit(Z_train, y_train)
print(f"AE latent accuracy: {lr.score(Z_test, y_test):.4f}")
# Compare to PCA baseline:
from sklearn.decomposition import PCA
Z_pca = PCA(n_components=LATENT_DIM).fit_transform(X_train_sc)
lr_pca = LogisticRegression(C=0.1, max_iter=1000).fit(Z_pca, y_train)
Z_pca_test = PCA(n_components=LATENT_DIM).fit(X_train_sc).transform(X_test_sc)
print(f"PCA latent accuracy: {lr_pca.score(Z_pca_test, y_test):.4f}")
```
If the AE does not beat PCA on the downstream task with equal $d$, ask yourself: is the
data's structure actually nonlinear? Sometimes it isn't, and PCA is the right tool.

**The sniff test:** Decode a few latent space interpolations. Pick two samples from different
classes, linearly interpolate their latent codes, decode each intermediate point. If the
interpolated reconstructions transition smoothly and meaningfully — congrats, you have a good
manifold. If they collapse to noise or produce impossible-looking samples, the latent space
has holes (standard AE issue) — consider a VAE.

---

## §8 — Diagnostic Plots & Evaluation

### 8.1 Training Loss Curve

**What it shows:** Whether the optimizer converged.
**Good:** Monotonically decreasing, clear plateau.
**Bad:** Oscillating, still steeply decreasing at the end.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

ae = MLPRegressor(
    hidden_layer_sizes=(32, 8, 32), activation='relu', solver='adam',
    learning_rate_init=0.001, max_iter=500, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20, random_state=42
)
ae.fit(X_train_sc, X_train_sc)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ae.loss_curve_, label='Train Loss')
if ae.validation_scores_ is not None:
    axes[0].plot([-s for s in ae.validation_scores_], label='Val Loss (neg R²)', alpha=0.7)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
axes[0].set_title("Training Curve"); axes[0].legend(); axes[0].set_yscale('log')

# Per-sample test reconstruction error distribution
X_recon = ae.predict(X_test_sc)
per_sample_mse = np.mean((X_test_sc - X_recon)**2, axis=1)
axes[1].hist(per_sample_mse, bins=30, edgecolor='k')
axes[1].set_xlabel("Reconstruction MSE"); axes[1].set_ylabel("Count")
axes[1].set_title("Reconstruction Error Distribution (Test Set)")
plt.tight_layout(); plt.show()
print(f"Mean test MSE: {per_sample_mse.mean():.4f} ± {per_sample_mse.std():.4f}")
```

### 8.2 Reconstruction Quality Visualization

**What it shows:** Whether the AE preserves visually meaningful information.

```python
def encode_sklearn(model, X, n_enc=2):
    act = {'relu': lambda x: np.maximum(0, x), 'tanh': np.tanh,
           'logistic': lambda x: 1/(1+np.exp(-x)), 'identity': lambda x: x}[model.activation]
    h = X
    for i in range(n_enc):
        h = h @ model.coefs_[i] + model.intercepts_[i]
        h = act(h)
    return h

n_show = 8
fig, axes = plt.subplots(2, n_show, figsize=(14, 3))
X_recon = ae.predict(X_test_sc)
X_orig_raw = scaler.inverse_transform(X_test_sc)
X_recon_raw = scaler.inverse_transform(X_recon)

for i in range(n_show):
    axes[0, i].imshow(X_orig_raw[i].reshape(8, 8), cmap='gray_r', vmin=0, vmax=16)
    axes[0, i].axis('off')
    axes[1, i].imshow(X_recon_raw[i].reshape(8, 8), cmap='gray_r', vmin=0, vmax=16)
    axes[1, i].axis('off')
axes[0, 0].set_ylabel("Original", rotation=0, ha='right', fontsize=10)
axes[1, 0].set_ylabel("Reconstructed", rotation=0, ha='right', fontsize=10)
plt.suptitle("Autoencoder Reconstruction Quality (digits, latent_dim=8)")
plt.tight_layout(); plt.show()
```

### 8.3 Latent Space Visualization

**What it shows:** Whether the compressed representation captures semantic structure.

```python
from sklearn.decomposition import PCA as SklearnPCA

Z_test = encode_sklearn(ae, X_test_sc, n_enc=2)  # shape (n_test, 8)

# If latent_dim > 2: project to 2D with PCA for visualization
if Z_test.shape[1] > 2:
    Z_2d = SklearnPCA(n_components=2).fit_transform(Z_test)
else:
    Z_2d = Z_test

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y_test, cmap='tab10', alpha=0.7, s=30)
plt.colorbar(scatter, ax=ax, label='Digit class')
ax.set_xlabel("Latent dim 1"); ax.set_ylabel("Latent dim 2")
ax.set_title("Autoencoder Latent Space (8→2 via PCA), colored by digit class")
plt.tight_layout(); plt.show()
```

### 8.4 Reconstruction Error vs. Latent Dimension (Elbow Plot)

```python
latent_dims = [2, 4, 8, 16, 32, 48, 64]
train_mses, test_mses = [], []

for d in latent_dims:
    enc_size = max(16, d * 2)
    ae_d = MLPRegressor(
        hidden_layer_sizes=(enc_size, d, enc_size), activation='relu', solver='adam',
        learning_rate_init=0.001, max_iter=500, early_stopping=True,
        n_iter_no_change=20, random_state=42
    )
    ae_d.fit(X_train_sc, X_train_sc)
    train_mses.append(np.mean((X_train_sc - ae_d.predict(X_train_sc))**2))
    test_mses.append(np.mean((X_test_sc   - ae_d.predict(X_test_sc))**2))

plt.figure(figsize=(7, 4))
plt.plot(latent_dims, train_mses, 'o-', label='Train MSE')
plt.plot(latent_dims, test_mses,  's--', label='Test MSE')
plt.xlabel("Latent Dimension $d$"); plt.ylabel("Reconstruction MSE")
plt.title("Elbow Plot: Reconstruction Loss vs. Bottleneck Size")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

> 🔧 **In practice:** The test MSE elbow tells you where you stop gaining meaningful
> compression. If test MSE drops sharply from $d=2$ to $d=8$ but barely from $d=8$ to $d=32$,
> your data has an intrinsic dimensionality near 8.

### 8.5 Trustworthiness Score

```python
from sklearn.manifold import trustworthiness

Z_test = encode_sklearn(ae, X_test_sc, n_enc=2)
T = trustworthiness(X_test_sc, Z_test, n_neighbors=10)
print(f"Trustworthiness@10: {T:.4f}")
# Interpretation: >0.95 = excellent, 0.90–0.95 = good, <0.85 = poor
```

### 8.6 Downstream Task Linear Probe

```python
from sklearn.linear_model import LogisticRegression

Z_train = encode_sklearn(ae, X_train_sc, n_enc=2)
Z_test  = encode_sklearn(ae, X_test_sc,  n_enc=2)

lr_probe = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_probe.fit(Z_train, y_train)
ae_acc = lr_probe.score(Z_test, y_test)

# PCA baseline at same dimensionality
from sklearn.decomposition import PCA as SklearnPCA
pca = SklearnPCA(n_components=8)
Z_pca_train = pca.fit_transform(X_train_sc)
Z_pca_test  = pca.transform(X_test_sc)
lr_pca = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_pca.fit(Z_pca_train, y_train)
pca_acc = lr_pca.score(Z_pca_test, y_test)

print(f"AE  linear probe accuracy (d=8): {ae_acc:.4f}")
print(f"PCA linear probe accuracy (d=8): {pca_acc:.4f}")
```

---

## §9 — Innovative Industry Applications

### 9.1 Drug Discovery — Molecular Generation with VAE

**Domain:** Pharmaceutical research / computational chemistry.

**Problem:** The chemical space of drug-like molecules is estimated at $10^{60}$ compounds.
High-throughput screening can test maybe $10^7$ per campaign. We need to *generate* candidates
with desired properties, not just screen them.

**Why VAE:** Encode SMILES strings (molecular representations) into a continuous latent space.
The smooth, regularized latent space enables gradient-based optimization: decode a molecule →
predict binding affinity → backpropagate through the decoder to find the latent direction
of improvement → decode the improved latent code. This "latent space optimization" is only
possible because the VAE makes the latent space geometrically regular.

> 🏭 **Industry application:** Gómez-Bombarelli et al. (ACS Central Science, 2018) showed that
> a character-level LSTM VAE on SMILES could generate novel valid molecules and be steered toward
> high predicted binding affinity. The TGVAE (Transformer-Graph VAE) extended this in 2025.

```python
# Sketch: continuous latent space optimization for drug-like molecules
# (pseudo-code; full implementation requires RDKit and a molecular property predictor)

def latent_optimization(vae, property_predictor, z_init, n_steps=100, lr=0.01):
    z = z_init.clone().detach().requires_grad_(True)
    optimizer_z = torch.optim.Adam([z], lr=lr)
    for step in range(n_steps):
        molecule = vae.decoder(z)         # decode to molecular fingerprint
        prop = property_predictor(molecule)  # predict binding affinity
        loss = -prop                       # maximize property (minimize negative)
        optimizer_z.zero_grad(); loss.backward(); optimizer_z.step()
    return z.detach()
```

### 9.2 Credit Card Fraud Detection — Unsupervised via Reconstruction Error

**Domain:** Financial services.

**Problem:** Fraud is 0.1% of transactions — extreme class imbalance. Labeled fraud data is
sparse and often proprietary. Supervised models struggle.

**Why AE:** Train only on normal transactions. The AE learns what "normal" looks like.
Fraudulent transactions, by definition different from training distribution, produce
unusually high reconstruction error. Reconstruction error = anomaly score. No labels needed.

```python
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import roc_auc_score, classification_report

# Train only on normal transactions (y_train == 0)
X_normal = X_train_sc[y_train == 0]

clf = AutoEncoder(
    hidden_neurons=[64, 32, 32, 64],
    epochs=100, batch_size=64, lr=1e-3,
    contamination=0.001,  # 0.1% expected fraud rate
    random_state=42,
)
clf.fit(X_normal)

fraud_scores = clf.decision_function(X_test_sc)  # reconstruction error per sample
auc = roc_auc_score(y_test, fraud_scores)
print(f"AUC-ROC: {auc:.4f}")  # target: >0.93 for production use
```

> 🏭 **Industry application:** AE-based fraud detection is now standard at major payment
> processors. Production systems often combine AE-based first-stage filtering with a supervised
> LightGBM specialist on the flagged subset. This hybrid approach balances recall (AE catches
> novel fraud patterns not in training data) with precision (LightGBM reduces false positives).

### 9.3 Industrial Predictive Maintenance — LSTM Autoencoder

**Domain:** Manufacturing, oil and gas, aviation.

**Problem:** Equipment failure is expensive and dangerous. Sensor data from motors, pumps,
and compressors is high-dimensional multivariate time series. Failure modes are rare and
labeled failure examples are scarce.

**Why LSTM AE:** The LSTM encoder captures temporal dependencies across sensor readings over
time. Trained on normal operating conditions, it fails to reconstruct anomalous patterns that
precede failure (bearing wear signatures, vibration changes, temperature spikes). Gives a
continuous anomaly score over time.

```python
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, latent_dim, seq_len):
        super().__init__()
        self.encoder = nn.LSTM(n_features, latent_dim, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, n_features, batch_first=True)
        self.seq_len = seq_len

    def forward(self, x):  # x: (batch, seq_len, n_features)
        _, (h, _) = self.encoder(x)
        # Repeat hidden state across sequence to initialize decoder
        z = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        x_recon, _ = self.decoder(z)
        return x_recon, h[-1]  # reconstruction, latent code
```

### 9.4 Single-Cell Genomics — Cell Type Discovery with scVI

**Domain:** Computational biology.

**Problem:** Single-cell RNA-seq measures 20,000+ gene expression levels per cell across
tens of thousands of cells. PCA assumptions (Gaussian noise, linear structure) are violated
by count data with massive zero-inflation. Batch effects across experiments dwarf biological
variation.

**Why VAE (scVI):** scVI uses a VAE with a negative binomial likelihood, directly modeling
the discrete count nature of sequencing data. The VAE latent space (20–50 dimensions)
simultaneously learns biologically meaningful embeddings AND corrects for batch effects by
conditioning the decoder on batch ID. The result: cells of the same type cluster together
across different experimental batches.

```python
import scvi
import anndata

# Load single-cell count data in AnnData format
adata = anndata.read_h5ad("pbmc_10k.h5ad")  # 10,000 PBMCs, 20,000 genes

# Setup: specify raw count layer and batch variable
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")

# Train the VAE
model = scvi.model.SCVI(adata, n_latent=20, n_layers=2, gene_likelihood="nb")
model.train(max_epochs=400, early_stopping=True)

# Extract 20-dimensional latent representation
latent = model.get_latent_representation()   # (n_cells, 20)
adata.obsm["X_scVI"] = latent
```

> 🏭 **Industry application:** scVI and its extensions (scANVI for semi-supervised, totalVI for
> protein+RNA) have become the standard for large-scale cell atlas projects. The CELLxGENE
> Census from the Chan Zuckerberg Initiative uses scVI to integrate data across millions of
> cells from hundreds of experiments.

### 9.5 Voice Conversion — Disentangling Content from Identity

**Domain:** Speech technology.

**Problem:** Convert one speaker's voice to sound like another without parallel utterance data
(no recordings of both speakers saying the same sentences).

**Why VQ-VAE:** The bottleneck of a VQ-VAE (Vector Quantized VAE) forces the model to
represent speech as a sequence of discrete codes drawn from a learned codebook. When the
bottleneck is small enough, speaker-specific prosody and timbre are dropped, and only phonetic
content remains. A separate speaker embedding (from a speaker encoder) is concatenated at
decoding time — enabling any content code to be decoded in any speaker's voice.

> 🏭 **Industry application:** CycleVAE was the baseline system in the Voice Conversion
> Challenge 2020. Commercial applications include dubbing (generating a translated voice that
> sounds like the original speaker) and privacy-preserving speech (anonymizing speaker identity
> in speech datasets while preserving linguistic content).

### 9.6 Recommendation Systems — VAE for Collaborative Filtering

**Domain:** E-commerce, streaming, social media.

**Problem:** Traditional matrix factorization (SVD-based) cannot generalize to new users at
inference time without reoptimization. The item-interaction matrix is sparse and high-dimensional.

**Why Mult-VAE:** Netflix researchers showed (Liang et al., WWW 2018) that a VAE with
multinomial likelihood for implicit feedback data outperforms all traditional collaborative
filtering methods. The VAE learns a regularized latent user preference space. New users are
embedded by a single forward pass through the encoder. The regularization improves
generalization over deterministic AEs.

```python
# Sketch: Mult-VAE for collaborative filtering
# Input: binary user-item interaction vector (1=interacted, 0=not)
# Loss: multinomial log-likelihood (softmax output) + KL divergence

class MultVAE(nn.Module):
    def __init__(self, n_items, latent_dim=200):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_items, 600), nn.Tanh(),
        )
        self.fc_mu  = nn.Linear(600, latent_dim)
        self.fc_var = nn.Linear(600, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 600), nn.Tanh(),
            nn.Linear(600, n_items)
        )

    def forward(self, x):
        h   = self.encoder(x)
        mu  = self.fc_mu(h); log_var = self.fc_var(h)
        z   = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return torch.log_softmax(self.decoder(z), dim=-1), mu, log_var
```

### 9.7 Domain Adaptation — Domain-Invariant Representations

**Domain:** Medical imaging, autonomous driving, multi-sensor fusion.

**Problem:** A model trained on source domain data (e.g., imaging center A) fails on target
domain (imaging center B) due to scanner differences. The distributions differ systematically.

**Why AE:** An adversarial AE trains the encoder to produce representations that fool a domain
discriminator — making it impossible to tell which domain a representation came from. The
classifier trained on source domain latent representations then generalizes to the target domain.
This enables cross-hospital model transfer without sharing raw patient data.

### 9.8 Learned Image Compression

**Domain:** Media delivery, medical imaging archival, satellite data.

**Problem:** JPEG uses hand-crafted DCT transforms. For domain-specific images (chest X-rays,
satellite multispectral, hyperspectral), JPEG's generic transforms are suboptimal and produce
visible artifacts at high compression ratios.

**Why convolutional AE:** End-to-end optimization of rate (bits used by quantized latent codes)
and distortion (reconstruction quality) for the specific image distribution. A convolutional
encoder + quantizer + decoder learns data-adaptive transforms that outperform JPEG at equivalent
bit rates. Particularly dramatic gains for medical images where JPEG artifacts can obscure
clinically relevant features.

---

## §10 — Complete Python Worked Example

End-to-end on the **digits dataset** (sklearn built-in, 1797 8×8 digit images, 64 features,
10 classes). Three progressively powerful approaches: (1) shallow sklearn AE, (2) deep PyTorch
AE, (3) PyTorch VAE.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
import warnings; warnings.filterwarnings('ignore')

# ─── 1. DATA LOADING & EDA ────────────────────────────────────────────────────
digits = load_digits()
X, y = digits.data.astype(np.float32), digits.target
print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"Feature range: [{X.min():.1f}, {X.max():.1f}]")

# EDA: how much variance does PCA explain?
pca_full = PCA(n_components=64)
pca_full.fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
d_95 = np.searchsorted(cumvar, 0.95) + 1
d_99 = np.searchsorted(cumvar, 0.99) + 1
print(f"PCA: {d_95} components for 95% variance, {d_99} for 99%")
# → This tells us the data has ~23-dim structure: meaningful compression target is d≈8-20

# ─── 2. PREPROCESSING ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

LATENT_DIM = 8

# ─── 3. SHALLOW sklearn AUTOENCODER ───────────────────────────────────────────
from sklearn.neural_network import MLPRegressor

ae = MLPRegressor(
    hidden_layer_sizes=(32, LATENT_DIM, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=25,
    random_state=42,
)
ae.fit(X_train_sc, X_train_sc)
print(f"sklearn AE trained: {ae.n_iter_} epochs, final loss: {ae.loss_:.5f}")

def encode_sklearn(model, X, n_enc=2):
    act = {'relu': lambda x: np.maximum(0, x), 'tanh': np.tanh,
           'logistic': lambda x: 1/(1+np.exp(-x)), 'identity': lambda x: x}[model.activation]
    h = X.copy()
    for i in range(n_enc):
        h = h @ model.coefs_[i] + model.intercepts_[i]
        h = act(h)
    return h

Z_train_sk = encode_sklearn(ae, X_train_sc, n_enc=2)
Z_test_sk  = encode_sklearn(ae, X_test_sc,  n_enc=2)

X_recon_sk = ae.predict(X_test_sc)
mse_sk = np.mean((X_test_sc - X_recon_sk)**2)
trust_sk = trustworthiness(X_test_sc, Z_test_sk, n_neighbors=10)

lr_sk = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_sk.fit(Z_train_sk, y_train)
acc_sk = lr_sk.score(Z_test_sk, y_test)
print(f"\n── sklearn AE (d={LATENT_DIM}) ──")
print(f"  Test MSE:        {mse_sk:.4f}")
print(f"  Trustworthiness: {trust_sk:.4f}")
print(f"  Linear probe:    {acc_sk:.4f}")

# PCA baseline
pca_base = PCA(n_components=LATENT_DIM)
Z_pca_tr = pca_base.fit_transform(X_train_sc)
Z_pca_te = pca_base.transform(X_test_sc)
lr_pca = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_pca.fit(Z_pca_tr, y_train)
acc_pca = lr_pca.score(Z_pca_te, y_test)
mse_pca = np.mean((X_test_sc - pca_base.inverse_transform(Z_pca_te))**2)
print(f"\n── PCA baseline (d={LATENT_DIM}) ──")
print(f"  Test MSE:        {mse_pca:.4f}")
print(f"  Linear probe:    {acc_pca:.4f}")

# ─── 4. DEEP PYTORCH AUTOENCODER ──────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DeepAE(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48), nn.ReLU(),
            nn.Linear(48, 24),        nn.ReLU(),
            nn.Linear(24, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.ReLU(),
            nn.Linear(24, 48),         nn.ReLU(),
            nn.Linear(48, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

device = 'cuda' if torch.cuda.is_available() else 'cpu'
deep_ae = DeepAE(64, LATENT_DIM).to(device)
opt = torch.optim.Adam(deep_ae.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)

X_tr_t = torch.tensor(X_train_sc, dtype=torch.float32).to(device)
X_te_t = torch.tensor(X_test_sc,  dtype=torch.float32).to(device)
dataset = TensorDataset(X_tr_t)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

train_losses_deep = []
for epoch in range(400):
    deep_ae.train()
    epoch_loss = 0
    for (xb,) in loader:
        xr, z = deep_ae(xb)
        loss = nn.functional.mse_loss(xr, xb)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * len(xb)
    train_losses_deep.append(epoch_loss / len(X_tr_t))
    scheduler.step()

deep_ae.eval()
with torch.no_grad():
    Z_train_deep = deep_ae.encoder(X_tr_t).cpu().numpy()
    Z_test_deep  = deep_ae.encoder(X_te_t).cpu().numpy()
    X_recon_deep = deep_ae.decoder(torch.tensor(Z_test_deep, device=device)).cpu().numpy()

mse_deep = np.mean((X_test_sc - X_recon_deep)**2)
trust_deep = trustworthiness(X_test_sc, Z_test_deep, n_neighbors=10)
lr_deep = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_deep.fit(Z_train_deep, y_train)
acc_deep = lr_deep.score(Z_test_deep, y_test)
print(f"\n── Deep PyTorch AE (d={LATENT_DIM}) ──")
print(f"  Test MSE:        {mse_deep:.4f}")
print(f"  Trustworthiness: {trust_deep:.4f}")
print(f"  Linear probe:    {acc_deep:.4f}")

# ─── 5. PYTORCH VAE ───────────────────────────────────────────────────────────
class VAEDigits(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.enc_base = nn.Sequential(nn.Linear(input_dim, 48), nn.ReLU(),
                                       nn.Linear(48, 24), nn.ReLU())
        self.fc_mu    = nn.Linear(24, latent_dim)
        self.fc_lv    = nn.Linear(24, latent_dim)
        self.decoder  = nn.Sequential(nn.Linear(latent_dim, 24), nn.ReLU(),
                                      nn.Linear(24, 48), nn.ReLU(),
                                      nn.Linear(48, input_dim))

    def encode(self, x):
        h = self.enc_base(x)
        return self.fc_mu(h), self.fc_lv(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
        return self.decoder(z), mu, lv

vae = VAEDigits(64, LATENT_DIM).to(device)
opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-4)

def vae_loss_fn(xr, x, mu, lv, beta=1.0):
    recon = nn.functional.mse_loss(xr, x, reduction='sum') / len(x)
    kl    = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return recon + beta * kl, recon.item(), kl.item()

recon_hist, kl_hist = [], []
for epoch in range(400):
    vae.train()
    beta = min(1.0, epoch / 100)    # KL annealing: warm up over 100 epochs
    for (xb,) in loader:
        xr, mu, lv = vae(xb)
        loss, r, k = vae_loss_fn(xr, xb, mu, lv, beta=beta)
        opt_vae.zero_grad(); loss.backward(); opt_vae.step()
    if epoch % 50 == 0:
        recon_hist.append(r); kl_hist.append(k)

vae.eval()
with torch.no_grad():
    mu_train, _ = vae.encode(X_tr_t)
    mu_test,  _ = vae.encode(X_te_t)
    Z_train_vae = mu_train.cpu().numpy()
    Z_test_vae  = mu_test.cpu().numpy()

trust_vae = trustworthiness(X_test_sc, Z_test_vae, n_neighbors=10)
lr_vae = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_vae.fit(Z_train_vae, y_train)
acc_vae = lr_vae.score(Z_test_vae, y_test)
print(f"\n── PyTorch VAE (d={LATENT_DIM}) ──")
print(f"  Trustworthiness: {trust_vae:.4f}")
print(f"  Linear probe:    {acc_vae:.4f}")

# ─── 6. VISUALIZATION ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (Z, title) in zip(axes, [
    (Z_test_sk,   f"sklearn AE (d={LATENT_DIM})"),
    (Z_test_deep, f"Deep PyTorch AE (d={LATENT_DIM})"),
    (Z_test_vae,  f"VAE mu (d={LATENT_DIM})")
]):
    Z_2d = PCA(n_components=2).fit_transform(Z)
    sc = ax.scatter(Z_2d[:,0], Z_2d[:,1], c=y_test, cmap='tab10', alpha=0.7, s=20)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

plt.colorbar(sc, ax=axes[-1], label='Digit class')
plt.suptitle("Latent Space Comparison: sklearn AE vs Deep AE vs VAE (digits dataset)")
plt.tight_layout(); plt.show()

# ─── 7. SUMMARY TABLE ─────────────────────────────────────────────────────────
print("\n── Summary (d=8 bottleneck, digits test set) ──────────────────────────")
print(f"{'Method':<22} {'Test MSE':>10} {'Trust@10':>10} {'LinearProbe':>12}")
print("-" * 58)
print(f"{'PCA':<22} {mse_pca:>10.4f} {'—':>10} {acc_pca:>12.4f}")
print(f"{'sklearn AE':<22} {mse_sk:>10.4f} {trust_sk:>10.4f} {acc_sk:>12.4f}")
print(f"{'Deep PyTorch AE':<22} {mse_deep:>10.4f} {trust_deep:>10.4f} {acc_deep:>12.4f}")
print(f"{'VAE (mu)':<22} {'—':>10} {trust_vae:>10.4f} {acc_vae:>12.4f}")
```

> 💡 **Intuition:** On the digits dataset with $d=8$, you will typically see the deep AE
> outperform PCA on downstream classification (because digit structure is nonlinear — rotation,
> stroke thickness, pixel arrangement). The VAE may trade slight accuracy for a smoother, more
> interpolable latent space. The sklearn AE sits between them. These relative rankings can flip
> for small $n$ or linear datasets.

---

## §11 — When to Use This Algorithm

### Decision Guide

| Your situation | Recommendation |
|---|---|
| Need fast baseline; data is small or linear | **PCA first** — 10× simpler, explainable |
| Data is nonlinear, $n > 2{,}000$, no GPU | **sklearn AE** (MLPRegressor trick) |
| Data is nonlinear, GPU available, research | **Deep PyTorch AE or VAE** |
| Need to generate new data samples | **VAE** (pythae) |
| Need interpretable/disentangled latent axes | **beta-VAE** (pythae BetaVAE) |
| Anomaly detection, no fraud labels | **AE or VAE via pyod** |
| Single-cell RNA-seq, genomics count data | **scvi-tools SCVI** |
| Out-of-sample embedding for streaming data | **Parametric UMAP** or any AE |
| Self-supervised vision pretraining, large data | **SimCLR / MoCo** (contrastive) |
| Need neighborhood-preserving 2D visualization | **t-SNE or UMAP** (not AE alone) |
| Data has $n < 500$ | **PCA** (AE will overfit) |
| Text data, bag-of-words, NLP | **LSA or Mult-VAE** |

### When NOT to Use an Autoencoder

- **$n < 500$:** Overfitting. PCA or kernel PCA.
- **You need p-values or statistical inference:** AE has no closed-form uncertainty.
- **You need the 2D visualization to be globally meaningful:** t-SNE or UMAP preserves
  neighborhoods explicitly; AE does not.
- **Binary or count data with MSE loss:** The wrong likelihood. Use scVI (counts) or BCE output.
- **Training time is severely constrained:** A PCA fit takes milliseconds; an AE takes minutes.

### Comparison with Closest Alternatives

| Criterion | AE | PCA | t-SNE | UMAP |
|---|---|---|---|---|
| Nonlinear manifolds | Yes | No | Yes | Yes |
| Out-of-sample (new points) | Yes (instant) | Yes (instant) | No (approximations exist) | No (standard) |
| Generation / sampling | VAE: Yes | No | No | No |
| Global structure preserved | Partially | Yes | No | Partially |
| Neighborhood structure | Not guaranteed | No | Yes | Yes |
| Scalable to $n > 10^6$ | Yes (mini-batch) | Yes (incremental PCA) | No | Yes (approx) |
| Interpretable dimensions | No (unless beta-VAE) | Yes (top PCs) | No | No |

---

## §12 — Top Papers to Study

Papers are ranked from "Start here" to "Deep mastery":

### Tier 1: Start Here

**1. Hinton & Salakhutdinov (2006) — The Deep AE Landmark**
- Hinton, G.E., Salakhutdinov, R.R. "Reducing the Dimensionality of Data with Neural Networks."
  *Science*, 313(5786), pp. 504–507. DOI: 10.1126/science.1127647.
  https://www.cs.toronto.edu/~hinton/absps/science.pdf
- **Why read it:** The paper that restarted deep learning. Shows deep AE beats PCA on MNIST and
  document retrieval. Introduces layer-wise RBM pretraining. Even if the pretraining method
  is obsolete, the experiments and intuition are foundational. Start here.

**2. Kingma & Welling (2014) — VAE Foundation**
- Kingma, D.P., Welling, M. "Auto-Encoding Variational Bayes." *ICLR 2014*. arXiv:1312.6114.
  https://arxiv.org/abs/1312.6114
- **Why read it:** The reparameterization trick and the ELBO objective are derived clearly.
  Section 3 (the core contribution) is 2 pages of dense math that is worth studying carefully.
  After reading this paper you will understand every VAE variant, because they all modify the
  ELBO in some way.

### Tier 2: Read Next

**3. Vincent et al. (2008) — Denoising Autoencoders**
- Vincent, P., Larochelle, H., Bengio, Y., Manzagol, P.A. "Extracting and Composing Robust
  Features with Denoising Autoencoders." *ICML 2008*, pp. 1096–1103.
  https://dl.acm.org/doi/10.1145/1390156.1390294
  PDF: https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
- **Why read it:** The connection between DAE loss and score matching (learning the data density
  gradient) is one of the most beautiful results in unsupervised representation learning.
  Modern diffusion models descend directly from this insight.

**4. Higgins et al. (2017) — beta-VAE**
- Higgins, I. et al. "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
  Framework." *ICLR 2017*.
- **Why read it:** One hyperparameter change ($\beta > 1$) yields disentangled representations.
  The paper defines the disentanglement problem formally and provides the first quantitative
  metric (the beta-VAE score). Critical reading for anyone doing interpretable latent space work.

**5. Chen et al. (2020) — SimCLR**
- Chen, T., Kornblith, S., Norouzi, M., Hinton, G. "A Simple Framework for Contrastive
  Learning of Visual Representations." *ICML 2020*. arXiv:2002.05709.
  https://github.com/google-research/simclr
- **Why read it:** SimCLR beat supervised ImageNet pretraining on many downstream tasks using
  no labels at all. The ablation studies (which design choices matter) are exceptionally
  pedagogical. Read this after VAE to understand the contrastive learning alternative.

### Tier 3: Deep Mastery

**6. Tolstikhin et al. (2018) — Wasserstein Autoencoder**
- Tolstikhin, I., Bousquet, O., Gelly, S., Schölkopf, B. "Wasserstein Auto-Encoders."
  *ICLR 2018*. https://openreview.net/forum?id=HkL7n1-0b
- **Why read it:** Grounds the AE objective in optimal transport theory. WAE-MMD is practically
  better than VAE in many settings. Understanding this paper requires Wasserstein distance and
  MMD background — worth the investment.

**7. Sainburg et al. (2021) — Parametric UMAP**
- Sainburg, T., McInnes, L., Gentner, T.Q. "Parametric UMAP Embeddings for Representation
  and Semisupervised Learning." *Neural Computation*, 33(11), 2881–2907.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8516496/
- **Why read it:** The bridge between graph-based manifold methods (UMAP) and neural
  autoencoders. Enables streaming embeddings and semi-supervised extensions. Essential if
  you work in production settings where new data arrives continuously.

**8. Rumelhart, Hinton, Williams (1986) — The Origin**
- Rumelhart, D.E., Hinton, G.E., Williams, R.J. "Learning Internal Representations by Error
  Propagation." In *Parallel Distributed Processing*, Vol. 1, Ch. 8. MIT Press, 1986.
- **Why read it:** Historical foundation. The clarity of reasoning in 1986 is remarkable given
  how little was known. Gives you perspective on which problems are genuinely hard (they knew
  deep nets were hard to train) and which were eventually solved by better tools.

---

## §13 — Resources & Further Reading

### Books

- **Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), Chapter 14** —
  The canonical textbook treatment of autoencoders. Covers undercomplete, regularized (sparse,
  denoising, contractive), and representation learning theory. Free PDF: deeplearningbook.org

- **Bishop & Bishop — *Deep Learning: Foundations and Concepts* (Springer, 2024), Ch. 17–18** —
  Updated treatment including VAEs and normalizing flows. More mathematically precise than
  Goodfellow on the probabilistic generative model view.

- **Murphy — *Probabilistic Machine Learning: Advanced Topics* (MIT Press, 2023), Ch. 20–21** —
  The most rigorous treatment of VAEs, VAE variants, and the ELBO from a probabilistic
  perspective. Read this when you want the full Bayesian derivation.

### Blog Posts & Tutorials

- **Lilian Weng, "From Autoencoder to Beta-VAE"** (2018) —
  https://lilianweng.github.io/posts/2018-08-12-vae/
  The best single blog post on VAEs and their variants. Clear derivations, excellent diagrams.
  Essential reading.

- **Jeremy Jordan, "Introduction to Autoencoders"** —
  https://www.jeremyjordan.me/autoencoders/
  Clear walkthrough of the autoencoder family with intuitive explanations. Good starting point.

- **distill.pub, "Feature Visualization"** (Olah et al., 2017) —
  https://distill.pub/2017/feature-visualization/
  Not specifically about AEs but foundational for understanding what encoder features represent.

- **UvA Deep Learning Tutorials — Tutorial 9: Deep Autoencoders** —
  https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
  Full PyTorch Lightning implementation of AE and VAE on CIFAR-10. One of the best practical
  tutorials available.

- **PyTorch VAE GitHub repository** —
  https://github.com/AntixK/PyTorch-VAE
  Implements 17 VAE variants in clean PyTorch. The code is clean enough to read as documentation.

### Package Documentation

- **sklearn MLPRegressor** (1.5.x): https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPRegressor.html
- **PyOD documentation**: https://pyod.readthedocs.io/en/latest/
- **pythae (benchmark VAE)**: https://github.com/clementchadebec/benchmark_VAE
- **scvi-tools**: https://docs.scvi-tools.org/en/stable/
- **Parametric UMAP**: https://umap-learn.readthedocs.io/en/latest/parametric_umap.html
- **Keras molecule generation example**: https://keras.io/examples/generative/molecule_generation/

### Online Courses

- **fast.ai Practical Deep Learning for Coders** (lesson on generative models) —
  https://course.fast.ai/ — pragmatic PyTorch implementation approach

- **Coursera Deep Learning Specialization (Andrew Ng), Course 4** —
  Covers encoder-decoder architectures in the context of sequence-to-sequence models.
  Good for the architectural intuition.

- **Stanford CS231n (CNNs for Visual Recognition)** — Lecture 13 covers autoencoders, VAEs,
  and GANs. Slides freely available at cs231n.stanford.edu.
