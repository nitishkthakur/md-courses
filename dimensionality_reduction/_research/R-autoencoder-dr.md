# Research Dossier: Autoencoder and VAE-based Dimensionality Reduction
## For: Dimensionality Reduction Masterclass — Chapter on Autoencoders & VAEs

**Research Date:** June 2026
**Target chapter length:** 18,000+ tokens
**Audience:** The writing agent has only this dossier and the course bible. Every claim here is sourced.

---

## TABLE OF CONTENTS

1. [Algorithm Landscape & Taxonomy](#1-algorithm-landscape--taxonomy)
2. [Must-Read Papers (Full Citations)](#2-must-read-papers-full-citations)
3. [Sklearn 1.5.x API — Every Parameter Documented](#3-sklearn-15x-api--every-parameter-documented)
4. [Specialized Python Packages](#4-specialized-python-packages)
5. [Real-World Applications (8 Concrete Examples)](#5-real-world-applications-8-concrete-examples)
6. [Hyperparameter Tuning Guide](#6-hyperparameter-tuning-guide)
7. [Expert Practitioner Checklist](#7-expert-practitioner-checklist)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Common Pitfalls With Concrete Examples](#9-common-pitfalls-with-concrete-examples)
10. [Mathematical Foundations Quick Reference](#10-mathematical-foundations-quick-reference)

---

## 1. ALGORITHM LANDSCAPE & TAXONOMY

### The Autoencoder Family Tree

```
Autoencoder (Rumelhart/Hinton 1986 PDP, Hinton & Salakhutdinov 2006 Science)
├── UNDERCOMPLETE AUTOENCODER  — bottleneck smaller than input
│   ├── Shallow (1-hidden-layer) — near-equivalent to PCA with nonlinear activation
│   ├── Deep Undercomplete — stacked encoder/decoder, captures nonlinear manifolds
│   └── Tied-weights variant — decoder weights = transpose(encoder weights)
│
├── REGULARIZED AUTOENCODERS — prevent identity mapping even at full latent capacity
│   ├── Sparse Autoencoder — L1 or KL sparsity penalty on activations
│   ├── Denoising Autoencoder (DAE) — corrupt input, reconstruct clean (Vincent 2008)
│   ├── Contractive Autoencoder (CAE) — penalize Frobenius norm of Jacobian
│   └── Dropout Autoencoder — dropout as regularizer during encoding
│
├── VARIATIONAL AUTOENCODER (VAE) — Kingma & Welling 2014
│   ├── Standard VAE (Gaussian prior, ELBO loss = recon + KL)
│   ├── beta-VAE — stronger KL weight for disentanglement (Higgins 2017, ICLR)
│   ├── WAE (Wasserstein AE) — optimal transport regularizer (Tolstikhin 2018, ICLR)
│   ├── VQ-VAE — discrete latent codes via vector quantization (van den Oord 2017)
│   ├── Conditional VAE (CVAE) — condition generation on class label
│   └── Hierarchical VAE (HVAE, NVAE) — multi-scale latent hierarchy
│
├── CONTRASTIVE LEARNING AS DR — no reconstruction, contrastive loss
│   ├── SimCLR — augmentation-based contrastive (Chen et al. 2020 ICML)
│   └── MoCo — momentum encoder + queue (He et al. 2020 CVPR)
│
└── PARAMETRIC UMAP — UMAP loss trained through encoder network (Sainburg 2021)
    └── autoencoder variant: encoder minimizes UMAP loss + decoder minimizes recon
```

### Core Intuition Comparison Table

| Variant | Core Innovation | Best For | Generative? |
|---|---|---|---|
| Undercomplete AE | Bottleneck forces compression | DR, feature extraction | No |
| Denoising AE | Robustness via corruption | Representation learning, denoising | No |
| Sparse AE | Sparsity forces selectivity | Interpretable features | No |
| VAE | Probabilistic latent space, sampling possible | Generation + DR | Yes |
| beta-VAE | Higher KL weight for factor disentanglement | Interpretable latent axes | Yes |
| WAE | Optimal transport regularizer, sharper outputs | Generative quality | Yes |
| SimCLR | Contrastive loss, no decoder needed | Self-supervised pretraining | No |
| Parametric UMAP | UMAP loss in a neural encoder | Out-of-sample UMAP, streaming | No |

---

## 2. MUST-READ PAPERS (FULL CITATIONS)

### Paper 1: The Origin — Rumelhart 1986 (Two Forms)

**Form A — Book chapter (the original autoencoder description):**
- **Authors:** David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
- **Title:** "Learning Internal Representations by Error Propagation"
- **In:** Rumelhart & McClelland (Eds.), *Parallel Distributed Processing: Explorations in the Microstructure of Cognition*, Vol. 1, Chapter 8, pp. 318–362
- **Publisher:** MIT Press, Cambridge, MA, 1986
- **What you'll learn:** This is the original description of using a multi-layer network with a narrow middle layer to compress and reconstruct inputs. The PDP "encoder-decoder" idea is stated here. The backpropagation algorithm that makes it trainable is developed. Historical foundation of the entire field.
- **Difficulty:** Moderate — written before modern notation but very clear reasoning.

**Form B — Nature paper (backpropagation itself):**
- **Authors:** David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
- **Title:** "Learning Representations by Back-propagating Errors"
- **Venue:** *Nature*, 323, 533–536
- **Year:** 1986
- **DOI/URL:** https://www.scirp.org/reference/referencespapers?referenceid=1698775
- **What you'll learn:** The mathematical derivation of backpropagation. The key insight that hidden units can learn internal representations without explicit supervision — just reconstruction error. This is why autoencoders work.
- **Difficulty:** Low-to-moderate. Short paper, dense mathematics.

---

### Paper 2: Hinton & Salakhutdinov 2006 (The Deep AE Landmark)

- **Authors:** Geoffrey E. Hinton, Ruslan R. Salakhutdinov
- **Title:** "Reducing the Dimensionality of Data with Neural Networks"
- **Venue:** *Science*, 313(5786), pp. 504–507
- **Year:** 2006
- **DOI:** 10.1126/science.1127647
- **URL:** https://www.science.org/doi/10.1126/science.1127647
- **PDF:** https://www.cs.toronto.edu/~hinton/absps/science.pdf
- **What you'll learn:** The breakthrough observation that deep autoencoders beat PCA on dimensionality reduction, but only when initialized correctly. The paper introduces Restricted Boltzmann Machine (RBM) pretraining as a method to initialize deep network weights near a good solution, then fine-tunes with backprop. Without pretraining, gradient descent gets stuck; with pretraining, the 2D code for MNIST is visually cleaner than PCA. Historic paper that relaunched deep learning interest.
- **Difficulty:** Moderate. Requires understanding RBMs for full depth, but the AE ideas stand alone.
- **Key result:** A 784→1000→500→250→2 AE trained on MNIST finds a 2D code that clusters digits better than PCA. Codes on a document dataset beat LSA (a linear method) decisively.

---

### Paper 3: Vincent et al. 2008 (Denoising Autoencoders)

- **Authors:** Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol
- **Title:** "Extracting and Composing Robust Features with Denoising Autoencoders"
- **Venue:** *Proceedings of the 25th International Conference on Machine Learning (ICML 2008)*, pp. 1096–1103
- **Year:** 2008
- **DOI/URL:** https://dl.acm.org/doi/10.1145/1390156.1390294
- **PDF:** https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
- **What you'll learn:** The core idea: corrupt the input (add Gaussian noise, mask features, salt-and-pepper), train the AE to reconstruct the clean original. This seemingly simple change dramatically improves the quality of learned representations. The representations learned by denoising AEs are more robust and transfer better to downstream tasks. This paper also laid groundwork for pre-training deep networks layer-by-layer using DAEs, an alternative to RBM-based pretraining.
- **Difficulty:** Low-to-moderate. Very clear writing by Bengio's group.

---

### Paper 4: Kingma & Welling 2014 (VAE — Foundational)

- **Authors:** Diederik P. Kingma, Max Welling
- **Title:** "Auto-Encoding Variational Bayes"
- **Venue:** *International Conference on Learning Representations (ICLR 2014)*
- **Year:** 2014 (arXiv preprint December 2013)
- **arXiv:** arXiv:1312.6114
- **URL:** https://arxiv.org/abs/1312.6114
- **What you'll learn:** The VAE is introduced. The encoder outputs parameters (μ, σ) of a Gaussian posterior rather than a point estimate. The decoder samples from this distribution. The problem: sampling is not differentiable so backprop fails. The solution: the **reparameterization trick** — sample ε ~ N(0,1), then compute z = μ + σ·ε, making z differentiable w.r.t. μ and σ. The loss is the ELBO = E[log p(x|z)] - KL(q(z|x) || p(z)), a sum of reconstruction loss and a KL divergence that regularizes the latent space toward a unit Gaussian prior. This gives a continuous, smooth, sampleable latent space, enabling generation of new data points.
- **Difficulty:** Moderate-high. Requires variational inference background. The paper itself is dense but the math is clean.

---

### Paper 5: Higgins et al. 2017 (beta-VAE, Disentanglement)

- **Authors:** Irina Higgins, Loïc Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner
- **Title:** "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- **Venue:** *International Conference on Learning Representations (ICLR 2017)*
- **Year:** 2017
- **What you'll learn:** A single hyperparameter change to the VAE objective — multiplying the KL term by a scalar β > 1 — dramatically increases disentanglement of latent factors. With high β, each latent dimension learns to represent a single independent factor of variation (e.g., color, shape, size, orientation), while with β=1 (standard VAE) factors are entangled. The tradeoff: higher β reduces reconstruction quality. The paper provides both qualitative (latent traversal videos) and quantitative (disentanglement score) evaluation. Critical reading for anyone working on interpretable latent spaces.
- **Difficulty:** Moderate. Well-written, extensive experiments.

---

### Paper 6: Tolstikhin et al. 2018 (WAE — Wasserstein AE)

- **Authors:** Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, Bernhard Schölkopf
- **Title:** "Wasserstein Auto-Encoders"
- **Venue:** *International Conference on Learning Representations (ICLR 2018)*
- **Year:** 2018
- **URL:** https://openreview.net/forum?id=HkL7n1-0b
- **What you'll learn:** Instead of maximizing the ELBO (VAE approach), WAE minimizes the Wasserstein distance between the model distribution and the true data distribution. This leads to a different regularizer: instead of KL divergence per datapoint, WAE penalizes the aggregate posterior q_φ(z) = ∫q_φ(z|x)p_data(x)dx from deviating from the prior p(z). Two implementations: WAE-GAN (adversarial discriminator on z) and WAE-MMD (maximum mean discrepancy, kernel-based). WAE produces sharper reconstructions than VAE and avoids posterior collapse issues. The paper grounds the autoencoder objective in optimal transport theory.
- **Difficulty:** High. Requires familiarity with optimal transport (Wasserstein distance) and MMD.

---

### Paper 7: Chen et al. 2020 (SimCLR — Contrastive Learning)

- **Authors:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- **Title:** "A Simple Framework for Contrastive Learning of Visual Representations"
- **Venue:** *International Conference on Machine Learning (ICML 2020)*
- **Year:** 2020
- **arXiv:** arXiv:2002.05709
- **GitHub:** https://github.com/google-research/simclr
- **What you'll learn:** SimCLR shows that high-quality low-dimensional representations can be learned without reconstruction or labels — only by contrasting augmented views of the same image. The key design choices: (1) heavy data augmentation (crop + color jitter + Gaussian blur) to create two views of each image, (2) a projector head (nonlinear MLP) on top of the encoder that the contrastive loss is applied to (not the representation directly), (3) normalized temperature-scaled cross-entropy (NT-Xent) loss, (4) large batch sizes (4096+). The encoder learns representations where semantically similar inputs cluster together. These representations transfer remarkably well to downstream tasks via linear probing.
- **Difficulty:** Moderate. Clear writing, extensive ablations.

---

### Paper 8: He et al. 2020 (MoCo — Momentum Contrast)

- **Authors:** Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick
- **Title:** "Momentum Contrast for Unsupervised Visual Representation Learning"
- **Venue:** *CVPR 2020*
- **Year:** 2020
- **arXiv:** arXiv:1911.05722
- **What you'll learn:** MoCo frames self-supervised learning as a dictionary look-up problem. The key insight: contrastive learning needs many negative examples, but large batches are expensive. MoCo maintains a dynamic queue of encoded representations from recent mini-batches as a large dictionary of negatives. A momentum encoder (slowly updated via moving average of query encoder) keeps the queue consistent over time. MoCo achieves strong results with moderate batch sizes and transfers well to detection/segmentation tasks, outperforming supervised ImageNet pretraining on Pascal VOC.
- **Difficulty:** Moderate. The queue + momentum encoder design takes a moment to internalize.

---

### Paper 9: Sainburg et al. 2021 (Parametric UMAP)

- **Authors:** Tim Sainburg, Leland McInnes, Timothy Q. Gentner
- **Title:** "Parametric UMAP Embeddings for Representation and Semisupervised Learning"
- **Venue:** *Neural Computation*, 33(11), pp. 2881–2907
- **Year:** 2021
- **PubMed:** https://pubmed.ncbi.nlm.nih.gov/34474477/
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8516496/
- **What you'll learn:** Standard UMAP is nonparametric — each point gets an embedding, but there is no function mapping new points to the latent space. Parametric UMAP trains a neural network encoder to minimize the UMAP graph-layout loss, giving a parametric mapping. New points are instantly embedded by a forward pass. The autoencoder variant also trains a decoder to minimize reconstruction loss jointly with the UMAP loss. Enables streaming/online embeddings, faster inference, and semi-supervised extension by adding label-based cross-entropy loss on labeled samples.
- **Difficulty:** Moderate. Requires UMAP background.

---

## 3. SKLEARN 1.5.x API — EVERY PARAMETER DOCUMENTED

### The sklearn Approach to Autoencoders

sklearn does not have a dedicated `Autoencoder` class. The canonical approach uses `MLPRegressor` (or `MLPClassifier`) with X as both input and target: `model.fit(X, X)`. To extract the bottleneck representation, you use `coefs_` and `intercepts_` directly or build a wrapper.

**KEY LIMITATION (flag this in the chapter):** sklearn's `MLPRegressor.predict(X_new)` passes data through the entire network (encoder + decoder) and returns the reconstruction. There is no built-in `.encode(X)` method. To get the latent representation, you must manually implement the forward pass through only the encoder layers using `coefs_` and `intercepts_`.

**Verified Workaround (sklearn 1.5.x):**
```python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Build autoencoder: input_dim → 64 → 32 → BOTTLENECK(8) → 32 → 64 → input_dim
# hidden_layer_sizes specifies ALL hidden layers including the bottleneck
ae = MLPRegressor(
    hidden_layer_sizes=(64, 32, 8, 32, 64),  # symmetric: encoder + bottleneck + decoder
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)
ae.fit(X_scaled, X_scaled)  # X == y for autoencoder

# Manual encoder forward pass to get bottleneck (layer index 2 = 8-dim bottleneck)
def encode(model, X, n_encoder_layers=3):
    """Forward pass through only the encoder portion of the MLP."""
    activation_fn = {'relu': lambda x: np.maximum(0, x),
                     'tanh': np.tanh,
                     'logistic': lambda x: 1 / (1 + np.exp(-x)),
                     'identity': lambda x: x}[model.activation]
    h = X
    for i in range(n_encoder_layers):
        h = h @ model.coefs_[i] + model.intercepts_[i]
        h = activation_fn(h)
    return h

X_encoded = encode(ae, X_scaled, n_encoder_layers=3)  # shape (n_samples, 8)
```

### MLPRegressor — Complete Parameter Reference (sklearn 1.5.x)

Source: https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPRegressor.html

| Parameter | Type | Default | What It Controls | Autoencoder Notes |
|---|---|---|---|---|
| `hidden_layer_sizes` | tuple of int | `(100,)` | Number of neurons in each hidden layer; len = number of hidden layers | For a symmetric AE, use e.g. `(256, 128, 64, 32, 64, 128, 256)` — middle = bottleneck |
| `activation` | `{'identity','logistic','tanh','relu'}` | `'relu'` | Nonlinearity applied at each hidden unit | `'relu'` is default and works well; use `'tanh'` for outputs in [-1,1]; `'identity'` makes the AE equivalent to PCA (linear) |
| `solver` | `{'lbfgs','sgd','adam'}` | `'adam'` | Optimizer for weight updates | `'adam'` is the practical choice; `'lbfgs'` is good for small datasets (converges faster, no learning rate to tune); `'sgd'` gives fine control via momentum but needs careful LR scheduling |
| `alpha` | float | `0.0001` | L2 regularization coefficient (weight decay) | Increase to 0.001–0.01 if the AE overfits (reconstruction loss on train << validation) |
| `batch_size` | int or `'auto'` | `'auto'` | Mini-batch size for stochastic solvers | `'auto'` = min(200, n_samples). For large datasets, use 256 or 512 explicitly |
| `learning_rate` | `{'constant','invscaling','adaptive'}` | `'constant'` | LR schedule — **only used when solver='sgd'** | Ignored for 'adam'. Use `'adaptive'` with sgd to reduce LR when training stalls |
| `learning_rate_init` | float | `0.001` | Initial learning rate for 'adam' and 'sgd' | Critical. 0.001 is good starting point; try 1e-4 for very deep nets or noisy data |
| `power_t` | float | `0.5` | Exponent for invscaling LR decay — **only used when solver='sgd'** | Rarely tuned; leave at 0.5 |
| `max_iter` | int | `200` | Maximum epochs (full passes through data) | 200 is often too few for deep AEs. Set 500–2000 and use early_stopping=True |
| `shuffle` | bool | `True` | Whether to shuffle samples each epoch — **only for sgd/adam** | Leave True |
| `random_state` | int/RandomState/None | `None` | Controls weight initialization and shuffling RNG | Set for reproducibility |
| `tol` | float | `1e-4` | Stopping tolerance: training stops if loss improvement < tol for n_iter_no_change epochs | Make smaller (e.g. 1e-6) for careful convergence |
| `verbose` | bool | `False` | Print progress to stdout | Set True for debugging |
| `warm_start` | bool | `False` | If True, reuse previous fit's weights as initialization for next fit() call | Useful for incremental training / curriculum learning |
| `momentum` | float | `0.9` | Gradient update momentum — **only for solver='sgd'** | Leave at 0.9 |
| `nesterovs_momentum` | bool | `True` | Use Nesterov's accelerated gradient — **only when solver='sgd' and momentum>0** | Leave True; provides faster convergence than standard momentum |
| `early_stopping` | bool | `False` | If True, holds out `validation_fraction` of training data for validation, stops when val score doesn't improve for `n_iter_no_change` epochs | Highly recommended for AEs to prevent overfitting |
| `validation_fraction` | float | `0.1` | Fraction of training data for validation when early_stopping=True | 0.1 (10%) is reasonable; increase to 0.15 for smaller datasets |
| `beta_1` | float | `0.9` | Adam first moment decay — **only for solver='adam'** | Leave at Adam default; rarely tuned |
| `beta_2` | float | `0.999` | Adam second moment decay — **only for solver='adam'** | Leave at Adam default |
| `epsilon` | float | `1e-8` | Adam numerical stability constant — **only for solver='adam'** | Leave at default |
| `n_iter_no_change` | int | `10` | Number of epochs with no improvement before stopping (early_stopping or tol-based) | Increase to 20–50 for deep AEs that improve slowly |
| `max_fun` | int | `15000` | Max function evaluations — **only for solver='lbfgs'** | Increase for large networks |

### Fitted Attributes (available after .fit())

| Attribute | Type | Description |
|---|---|---|
| `coefs_` | list of ndarray, shape (n_layers-1,) | Weight matrices. `coefs_[i]` has shape (n_in, n_out) for layer i |
| `intercepts_` | list of ndarray, shape (n_layers-1,) | Bias vectors. `intercepts_[i]` has shape (n_out,) for layer i |
| `loss_` | float | Final training loss (MSE) |
| `best_loss_` | float | Minimum loss seen during training (None if early_stopping=True) |
| `loss_curve_` | list | Loss at each iteration (only for sgd/adam) — use to plot training curve |
| `n_iter_` | int | Number of epochs run |
| `n_layers_` | int | Total number of layers including input and output |
| `out_activation_` | str | Output activation name (always 'identity' for MLPRegressor) |
| `validation_scores_` | list or None | R² on validation set per epoch (only if early_stopping=True) |
| `n_features_in_` | int | Number of input features seen during fit |

### Version-Specific Changes to Note

- `covariance_eigh` solver was added to `PCA` in sklearn 1.5 (not MLPRegressor)
- `feature_names_in_` attribute added in sklearn 1.0 — check if X has string feature names
- `partial_fit()` is available for incremental training (SGD only, not LBFGS)
- **API UNCERTAINTY FLAG**: sklearn does not expose a `.encode()` or `.transform()` method on MLPRegressor. Any chapter code showing encoder access must use the manual `coefs_`/`intercepts_` approach or a wrapper class.

---

## 4. SPECIALIZED PYTHON PACKAGES

### Package 1: PyTorch (torch)

- **PyPI name:** `torch`
- **Install:** `pip install torch torchvision`
- **Version (mid-2026):** 2.4.x
- **License:** BSD
- **What it does better than sklearn:** Full control over encoder/decoder architecture separately; custom loss functions (VAE ELBO, contrastive); GPU acceleration; gradient checkpointing for deep nets; `nn.Module` subclassing for clean encoder/decoder API
- **Key import:** `import torch; import torch.nn as nn`
- **Maintenance:** Meta AI, extremely active
- **Use for:** Deep AE, VAE, beta-VAE, WAE, SimCLR encoder
- **Code snippet:**
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, z):
        return self.net(z)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
```

---

### Package 2: pythae (Unified VAE Benchmarking)

- **PyPI name:** `pythae`
- **Install:** `pip install pythae`
- **Version (mid-2026):** 0.1.2 (September 2023; mature/stable)
- **License:** Apache 2.0
- **What it does better than sklearn:** Implements 19+ VAE variants (VAE, beta-VAE, WAE, VAMP, RHVAE, SVAE, etc.) with a unified API. Same encoder/decoder architecture used across all models for fair comparison. Integrates with W&B, MLflow, Comet. Supports HuggingFace Hub model sharing. Benchmarking use case published at NeurIPS 2022.
- **Key imports:**
```python
from pythae.models import VAE, VAEConfig
from pythae.models import BetaVAE, BetaVAEConfig
from pythae.models import WAE_MMD, WAE_MMD_Config
from pythae.trainers import BaseTrainer, BaseTrainerConfig
```
- **Maintenance:** INRIA/Sorbonne, published NeurIPS 2022, stable as of 2024
- **Use for:** Systematic comparison of VAE variants; research; any VAE beyond vanilla
- **Paper:** Chadebec, Vincent & Allassonnière, NeurIPS 2022, arXiv:2206.08309

---

### Package 3: PyOD (Anomaly Detection via Autoencoders)

- **PyPI name:** `pyod`
- **Install:** `pip install pyod`
- **Version (mid-2026):** 3.6.1 (released June 17, 2026)
- **License:** BSD-2-Clause
- **What it does better than sklearn:** 60+ anomaly detectors with unified fit/predict API. AE and VAE models built-in. Automatic threshold computation. Handles high-dimensional tabular data. ADEngine for automated model selection. The most comprehensive anomaly detection library in Python.
- **Key imports:**
```python
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.beta_vae import Beta_VAE
```
- **Autoencoder-specific usage:**
```python
from pyod.models.auto_encoder import AutoEncoder

clf = AutoEncoder(
    hidden_neurons=[64, 32, 32, 64],  # symmetric; middle 2 = bottleneck
    hidden_activation='relu',
    output_activation='sigmoid',
    epochs=100,
    batch_size=32,
    lr=1e-3,
    l2_regularizer=0.1,
    contamination=0.05,  # expected fraction of anomalies
)
clf.fit(X_train)
y_scores = clf.decision_scores_          # reconstruction error per sample
y_pred = clf.predict(X_test)             # 0=normal, 1=outlier
y_prob = clf.predict_proba(X_test)       # probability of being an outlier
```
- **Maintenance:** yzhao062/PyOD, extremely active (version 3.x in 2026)

---

### Package 4: scvi-tools (Single-Cell VAE)

- **PyPI name:** `scvi-tools`
- **Install:** `pip install scvi-tools`
- **Version (mid-2026):** 1.4.3 (released May 12, 2026)
- **License:** BSD 3-Clause
- **What it does better than sklearn:** Purpose-built for single-cell RNA-seq. The scVI model is a VAE that handles count data (negative binomial likelihood), batch effects, and zero-inflation. Integrates with AnnData/scanpy ecosystem. GPU-accelerated via PyTorch. Models include scVI (basic), scANVI (semi-supervised), totalVI (CITE-seq), SCANVI, and more.
- **Key imports:**
```python
import scvi
import anndata

adata = anndata.read_h5ad("my_scrna.h5ad")
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
model = scvi.model.SCVI(adata, n_latent=20, n_layers=2)
model.train(max_epochs=400)
latent = model.get_latent_representation()  # shape (n_cells, n_latent)
```
- **Maintenance:** Yosef Lab (Berkeley) + scverse community, extremely active

---

### Package 5: umap-learn with Parametric UMAP

- **PyPI name:** `umap-learn`
- **Install:** `pip install umap-learn[parametric_umap]` (installs Keras 3 dependency)
- **Version (mid-2026):** 0.5.8
- **License:** BSD 3-Clause
- **What it does better than sklearn:** Standard UMAP is non-parametric (no function to embed new points). Parametric UMAP trains a neural encoder to minimize the UMAP graph-layout loss. With `autoencoder_loss=True`, a decoder is added and the encoder is jointly trained on UMAP loss + reconstruction loss. Now supports Keras 3 backends: JAX, PyTorch, TensorFlow.
- **Key imports:**
```python
from umap.parametric_umap import ParametricUMAP

embedder = ParametricUMAP(
    n_components=2,
    autoencoder_loss=True,      # add reconstruction decoder
    reconstruction_validation=0.1,
    parametric_reconstruction=True,
    encoder=None,               # use default MLP encoder; or pass custom keras model
    n_epochs=200,
)
embedding = embedder.fit_transform(X)
embedding_new = embedder.transform(X_new)  # out-of-sample! unlike standard UMAP
```
- **Maintenance:** lmcinnes/umap, very active

---

### Package 6: PyTorch Lightning (Training Framework)

- **PyPI name:** `lightning`
- **Install:** `pip install lightning`
- **Version (mid-2026):** 2.6.x
- **License:** Apache 2.0
- **What it does better than sklearn:** Structured training loops with automatic GPU/multi-GPU, logging, checkpointing. Has an official deep autoencoders tutorial (UvA DL course). Not a DR library per se but the right framework for building production-grade deep AE/VAE.
- **Key import:** `from lightning import LightningModule, Trainer`
- **Maintenance:** Lightning AI, very active

---

### Package Summary Table

| Package | Best For | Version | Modern? | GPU? |
|---|---|---|---|---|
| `torch` | Custom deep AE, VAE, research | 2.4.x | Modern | Yes |
| `pythae` | Comparing 19+ VAE variants | 0.1.2 | Modern | Yes (via torch) |
| `pyod` | Anomaly detection with AE | 3.6.1 | Modern | Yes (via keras) |
| `scvi-tools` | Single-cell genomics | 1.4.3 | Modern | Yes (via torch) |
| `umap-learn` | Parametric UMAP + AE loss | 0.5.8 | Modern | Partial |
| `sklearn` | Shallow AE, prototyping | 1.5.x | Legacy-ish | No |

---

## 5. REAL-WORLD APPLICATIONS (8 CONCRETE EXAMPLES)

### Application 1: Drug Discovery — Molecular Generation (VAE)

- **Domain:** Pharmaceutical research / computational chemistry
- **Problem:** The chemical space of drug-like molecules is enormous (~10^60 compounds). Traditional enumeration and testing is impossibly slow. Researchers need to *generate* new candidate molecules with desired properties (high binding affinity, low toxicity, good ADMET profile).
- **Why VAE:** The molecule (represented as SMILES string or graph) is encoded into a continuous latent space. Because VAE enforces a smooth, regular latent space, you can interpolate between known drugs or perform gradient-based optimization in latent space (decode a new molecule → predict property → backprop through decoder to find better z). This "latent space optimization" is not possible with discrete representations.
- **Reference/Company:** Gómez-Bombarelli et al. 2018, "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules" (ACS Central Science); recent 2024/2025 work: TGVAE (Transformer-Graph VAE) combining GNNs + VAE, published Nature Computational Science 2025 — outperforms prior methods in generating diverse novel molecules.
- **Keras example:** https://keras.io/examples/generative/molecule_generation/

---

### Application 2: Credit Card Fraud Detection (Reconstruction Error AE)

- **Domain:** Financial services / fintech
- **Problem:** Fraud is extremely rare (~0.1% of transactions), making supervised learning difficult (class imbalance). Traditional supervised approaches need labeled fraud data which may be sparse or proprietary.
- **Why AE:** Train an autoencoder **only on normal (legitimate) transactions**. The AE learns to reconstruct normal patterns well. When a fraudulent transaction is presented, it has an unexpectedly high reconstruction error — it looks "different" from anything the AE was trained on. Threshold on reconstruction error = fraud score. This is entirely unsupervised — no fraud labels needed.
- **Industry reality:** As of 2024, $12.5B was lost to financial fraud in the US (FTC report). Production systems often combine AE-based anomaly screening (first pass) with a supervised XGBoost/LightGBM specialist model (second pass), with AE-enhanced LightGBM (AEELG) showing strong results.
- **Reference:** "An AutoEncoder Enhanced Light Gradient Boosting Machine Method for Credit Card Fraud Detection", PMC 2024

---

### Application 3: Industrial Predictive Maintenance (LSTM Autoencoder)

- **Domain:** Manufacturing / industrial IoT
- **Problem:** Equipment failure is expensive and dangerous. Traditional scheduled maintenance wastes resources. Sensor data from motors, pumps, compressors is high-dimensional and time-series in nature.
- **Why LSTM AE:** An LSTM Autoencoder captures temporal dependencies in multivariate sensor time series. Trained on normal operating conditions, it fails to reconstruct anomalous readings (bearing wear, vibration changes, temperature spikes) that precede failure. Gives a continuous "anomaly score" rather than a binary alarm.
- **Result:** Real-time prediction with 95%+ detection accuracy without requiring labeled failure examples. Federated LSTM autoencoders allow training across distributed edge devices without centralizing sensitive operational data (data sovereignty compliance).
- **References:** "Real-Time Predictive Maintenance using Autoencoder Reconstruction and Anomaly Detection" (arXiv 2021); "Lightweight LSTM VAE for Anomaly Detection in Industrial Control Systems" (PMC 2022)

---

### Application 4: Learned Image Compression (Convolutional AE)

- **Domain:** Media / content delivery / photography
- **Problem:** JPEG was designed with hand-crafted DCT transforms. Learned compression can outperform JPEG at the same bit rate by learning task-specific, data-adaptive transforms.
- **Why deep AE:** A convolutional autoencoder learns to compress images into a compact latent representation, with quantization of the latent codes for actual bit savings. The decoder reconstructs the image. End-to-end optimization of rate (bits used) and distortion (reconstruction quality). Works especially well for medical images, satellite imagery, and face images — domains where JPEG's artifacts are particularly harmful.
- **Reference:** "Autoencoded Image Compression for Secure and Fast Transmission" (arXiv 2024). Companies like Google (web image formats) and Apple (HEIF) have explored learned compression variants.

---

### Application 5: Single-Cell RNA Sequencing — Cell Type Discovery (VAE)

- **Domain:** Computational biology / genomics
- **Problem:** Single-cell RNA-seq produces a matrix of gene expression counts per cell: typically 10,000–30,000 genes × 10,000–500,000 cells. Curse of dimensionality + massive batch effects between experiments make analysis hard.
- **Why scVI (VAE):** scVI models expression counts as negative binomial (handles zero-inflation and overdispersion of counts — assumptions PCA violates). The VAE latent space (typically 20–50 dimensions) simultaneously learns a biologically meaningful embedding AND corrects batch effects by conditioning on batch ID. Downstream: cluster cell types, compute differential expression, integrate datasets from different labs.
- **Impact:** scVI is part of the scverse ecosystem and has been used to build the CELLxGENE Census model (trained on millions of human cells). scvi-tools v1.4.3 released May 2026.
- **References:** Lopez et al. 2018, "Deep generative modeling for single-cell transcriptomics", Nature Methods; https://scvi-tools.org

---

### Application 6: Voice Conversion and Speech Synthesis (VQ-VAE / CycleVAE)

- **Domain:** Speech technology / conversational AI
- **Problem:** Convert the voice of one speaker to sound like another, without paired utterance data. The challenge: disentangle *what is said* (linguistic content) from *who is saying it* (speaker identity + prosody).
- **Why VAE:** A VAE (or VQ-VAE) with a bottleneck encodes speech into a low-dimensional representation. When the bottleneck is small enough, the model is forced to drop speaker-specific information and retain only phonetic content. Different decoders then reconstruct speech in different speaker styles.
- **Specific approach:** CycleVAE (Voice Conversion Challenge 2020 baseline) — CycleVAE uses cycle-consistent reconstruction to map between speakers without parallel data. VQ-VAE (Vector Quantized VAE) produces discrete latent codes that are especially good at content/style disentanglement.
- **Reference:** "Baseline System of Voice Conversion Challenge 2020 with Cyclic VAE and Parallel WaveGAN" (arXiv 2020); WaveNet autoencoders (DeepMind) for unsupervised speech representation learning (arXiv 2019)

---

### Application 7: Recommendation Systems (VAE for Collaborative Filtering)

- **Domain:** E-commerce / streaming / social media
- **Problem:** Matrix factorization (SVD-based latent factors) requires optimization at inference time to embed new users. This is too slow for real-time recommendations. Autoencoders provide immediate inference via forward pass.
- **Why VAE:** Netflix researchers (Liang et al. 2018, WWW) showed a Mult-VAE (VAE with multinomial likelihood for implicit feedback data) outperforms traditional CF methods and even sophisticated MF methods like WMF. The VAE regularization improves generalization versus a plain AE. The latent space is a compressed user preference representation.
- **Reference:** Liang, Krishnan, Hoffman, Jebara 2018, "Variational Autoencoders for Collaborative Filtering", WWW 2018 (arXiv:1802.05814). AutoRec (Sedhain et al. 2015, WWW) also demonstrated AE superiority over MF on MovieLens and Netflix Prize data.
- **Industry:** Used in production at Netflix-scale systems as part of multi-stage ranking pipelines.

---

### Application 8: Domain Adaptation for Cross-Domain Transfer (AE)

- **Domain:** Computer vision / NLP / medical imaging
- **Problem:** A model trained on source domain data (e.g., ImageNet photos) performs poorly on target domain (e.g., medical X-rays, satellite images, different camera sensors). The distributions differ.
- **Why AE:** Train an autoencoder to learn a domain-invariant latent space. The encoder is shared between source and target domains, with a domain discriminator that tries to tell domains apart. By adversarially training the encoder to fool the discriminator, the latent representation becomes domain-agnostic. The downstream classifier trained on source domain latent features generalizes to target domain.
- **Applications:** Cross-hospital model transfer in medical imaging (FDA data sovereignty), different sensor modalities in autonomous driving, text classification across languages.
- **Reference:** "Deep Autoencoder Based Domain Adaptation for Transfer Learning", Multimedia Tools and Applications, Springer 2022; "Representation Learning via Integrated Autoencoder for Unsupervised Domain Adaptation", Frontiers of Computer Science 2022.

---

## 6. HYPERPARAMETER TUNING GUIDE

### Hyperparameter 1: Latent Dimension (Bottleneck Size)

- **What it controls:** The information bottleneck — how many numbers are used to represent each input
- **Default in sklearn:** No default (user must specify via position in hidden_layer_sizes)
- **Recommended search range:** `[2, 4, 8, 16, 32, 64, 128]`
- **Rule of thumb:** Start at `sqrt(input_dim)`. For 784-dim MNIST: start at 28, try 2–256. For 20,000 genes: try 20–100.
- **Too high (latent_dim too large):** AE learns near-identity mapping. Low reconstruction error but useless compression. Latent dimensions have low variance (uninformative). Reconstruction on training = reconstruction on test (no generalization pressure).
- **Too low (latent_dim too small):** High reconstruction error. Loss of important information. Downstream task performance drops. Blurry reconstructions in image AEs.
- **Finding the elbow:** Plot reconstruction loss vs. latent_dim on a log scale. Look for an "elbow" — the point where loss stops dropping sharply. This approximates the intrinsic dimensionality of the data.
- **Optuna search space:**
```python
def objective(trial):
    latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 16, 32, 64])
    # ... build model, train, return val reconstruction loss
```

---

### Hyperparameter 2: Architecture Depth and Width (Encoder)

- **What it controls:** Number of layers and neurons per layer in encoder (and symmetric decoder)
- **Recommended:** 2–4 hidden layers per encoder. Width should taper from input to bottleneck.
- **Common tapers:** `input → input/2 → input/4 → bottleneck` (halving each layer)
- **Too deep:** Vanishing gradients, long training time, potential overfitting. Marginal representation quality gains beyond 4 encoder layers for tabular data.
- **Too shallow:** Can't capture nonlinear manifold structure. For tabular data with <100 features, 2 encoder layers often sufficient.
- **Key principle:** Decoder should mirror encoder architecture (symmetric) for best results
- **Optuna search space:**
```python
n_layers = trial.suggest_int("n_layers", 1, 4)
first_layer_size = trial.suggest_categorical("first_size", [64, 128, 256])
# build geometric taper down to latent_dim
```

---

### Hyperparameter 3: Learning Rate

- **What it controls:** Step size for Adam (or SGD) optimizer
- **Default (sklearn):** 0.001 (adam)
- **Recommended range:** `[1e-5, 1e-2]` — log uniform
- **Too high (>1e-2):** Training loss oscillates or diverges. Loss curve is noisy/unstable.
- **Too low (<1e-5):** Extremely slow convergence. May never reach good reconstruction.
- **Best practice:** Use cosine annealing or reduce-on-plateau scheduler in PyTorch. In sklearn: use `learning_rate='adaptive'` with solver='sgd' for automatic reduction, or use Adam's default adaptive behavior.
- **Optuna search space:**
```python
lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
```

---

### Hyperparameter 4: L2 Regularization (alpha / weight_decay)

- **What it controls:** Penalty on large weights; prevents overfitting; encourages smooth mappings
- **Default (sklearn alpha):** 0.0001
- **Recommended range:** `[1e-6, 1e-2]` — log uniform
- **Too high:** Underfitting; all weights near zero; high reconstruction error everywhere
- **Too low / zero:** AE memorizes training set; poor out-of-sample reconstruction
- **PyTorch equivalent:** `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`

---

### Hyperparameter 5: Batch Size

- **What it controls:** Number of samples per gradient update step
- **Default (sklearn):** `'auto'` = min(200, n_samples)
- **Recommended:** 32–512 depending on dataset size
- **Large batch (512+):** Faster training per epoch, smoother gradients, but may converge to flatter minima (worse generalization). For **SimCLR specifically**: large batch (1024–4096) is critical because you need many negatives per step.
- **Small batch (16–64):** Noisy gradients act as regularizer, often better generalization but slower.
- **Rule of thumb for tabular AE:** 256 is a good default. For image AE: 64–128.

---

### Hyperparameter 6: VAE-Specific — KL Weight (Beta)

- **What it controls:** Balance between reconstruction fidelity and regularization of latent space
- **Standard VAE:** beta = 1.0
- **beta-VAE:** beta > 1 (typical range 2–10 for high disentanglement)
- **Too high beta:** Blurry reconstructions, high ELBO KL term, better disentanglement, posterior collapse risk
- **Too low beta (<1):** Sharp reconstructions, but unregularized/entangled latent space, poor generalization, poor generation
- **KL annealing:** A practical trick — start beta at 0 and linearly increase to 1.0 over first N epochs. Allows reconstruction to stabilize before imposing regularity pressure.
- **Cyclical annealing (Fu et al. 2019):** Repeat the annealing schedule multiple times rather than once. Prevents posterior collapse more reliably.
- **Optuna search space:**
```python
beta = trial.suggest_float("beta", 0.1, 10.0, log=True)
```

---

### Hyperparameter 7: Corruption Level (Denoising AE)

- **What it controls:** Fraction of input features masked or level of Gaussian noise added
- **Common ranges:** Masking: 20–50% of features. Gaussian: σ = 0.1–0.5 × std of feature
- **Too high:** AE must guess too much of the input; high reconstruction error; unstable training
- **Too low:** AE doesn't develop robust representations; essentially equivalent to standard AE
- **Practical note:** Bernoulli masking (randomly zero out features) is more interpretable. Gaussian noise is smoother and often preferred for continuous data.

---

### Hyperparameter 8: Number of Epochs (max_iter in sklearn)

- **Default (sklearn):** 200 (often insufficient for deep AEs)
- **Recommended:** 500–2000 with early stopping (`early_stopping=True, n_iter_no_change=30`)
- **Diagnostic:** Always plot `ae.loss_curve_` to check for convergence. Loss still decreasing at max_iter = increase max_iter.

---

## 7. EXPERT PRACTITIONER CHECKLIST

### BEFORE FITTING

**Data preparation:**
- [ ] Scale all features to zero mean, unit variance (`StandardScaler`). AEs are sensitive to feature scales — a feature with range [0, 1000] will dominate reconstruction loss vs. one with range [0, 1].
- [ ] For image data: normalize to [0, 1] or [-1, 1]. For genomics count data: log-normalize or use count-specific likelihoods (negative binomial as in scVI).
- [ ] Check for missing values — autoencoders require complete inputs (or use masking strategies explicitly).
- [ ] For VAE: ensure input is continuous or appropriately modeled (use Bernoulli likelihood for binary inputs, Gaussian for continuous, Multinomial for counts).

**Architecture design:**
- [ ] Decide bottleneck size based on expected intrinsic dimensionality. Use PCA variance explained curve as lower bound on intrinsic dimensionality — if 95% variance explained at k=50 PCA components, start with bottleneck = 50.
- [ ] For sklearn MLPRegressor AE: remember the `hidden_layer_sizes` parameter includes the bottleneck. Example: encoder 256→128→32 + decoder 128→256 = `hidden_layer_sizes=(256, 128, 32, 128, 256)`.
- [ ] Choose activation function: `relu` for deep nets, `tanh` if outputs must be in [-1,1], `sigmoid` for binary reconstructions.

**Choosing algorithm:**
- [ ] Need generation / sampling? → VAE or pythae model
- [ ] Need interpretable/disentangled factors? → beta-VAE
- [ ] Need anomaly detection? → undercomplete AE or VAE via pyod
- [ ] Need out-of-sample embedding (transform new points instantly)? → Parametric UMAP or any AE (sklearn MLPRegressor.predict by definition)
- [ ] Need to handle noise/missing data robustly? → Denoising AE
- [ ] Domain is single-cell genomics? → scvi-tools

---

### DURING FITTING

**Monitoring training:**
- [ ] Plot training loss curve (`ae.loss_curve_` in sklearn, or your PyTorch loss logger). Should monotonically decrease and plateau. Spikes = LR too high or bad batch.
- [ ] For VAE: plot reconstruction loss and KL term separately. KL should gradually increase as the model learns to regularize. If KL collapses to zero at any point, you have **posterior collapse** — the decoder ignores z and becomes a standalone model.
- [ ] Check validation loss vs. training loss. Large gap = overfitting. Solutions: increase alpha/weight_decay, add dropout, reduce depth/width.
- [ ] For very deep nets (>4 encoder layers): check gradient flow. Vanishing gradients manifest as loss barely changing in early epochs.

**Posterior collapse in VAE (critical pitfall):**
- Symptom: KL term → 0, reconstruction improves but sampling from prior produces garbage
- Solutions: (1) KL annealing, (2) reduce beta, (3) increase latent dim, (4) use WAE instead, (5) use Free Bits (enforce minimum KL per dimension), (6) skip connections (ResNet-style AE)

---

### AFTER FITTING

**Quality checks:**
- [ ] Reconstruction quality: visualize original vs. reconstructed for a random sample. For images: look for blurriness (sign of bottleneck too small or VAE with high beta). For tabular: compute per-feature MAE, check if any features are systematically poorly reconstructed.
- [ ] Latent space visualization: if bottleneck > 2D, apply PCA or t-SNE on latent codes and plot colored by class labels (if available). Well-separated clusters = good structure learned.
- [ ] Out-of-sample reconstruction: compute reconstruction MSE on a held-out test set. Should be similar to training set MSE. Significant degradation = overfitting or domain shift.
- [ ] For DR use case: run downstream classifier on latent codes. Compare accuracy to PCA baseline at same dimensionality.
- [ ] For VAE generative use case: sample z ~ N(0,I), decode, and inspect samples. Should look plausible. If not: latent space is not regular enough (increase KL weight).

---

## 8. EVALUATION METRICS

### Metric 1: Reconstruction Loss (MSE / BCE)

**Formula:** `L_recon = (1/N) Σ ||x_i - x̂_i||²` for MSE, or binary cross-entropy for binary inputs

**Implementation:**
```python
import numpy as np
X_reconstructed = ae.predict(X_test)  # sklearn: full AE forward pass
mse = np.mean((X_test - X_reconstructed) ** 2)
per_feature_mse = np.mean((X_test - X_reconstructed) ** 2, axis=0)  # which features are hard
```

**What it tells you:** How well the AE preserves the original information. Primary loss signal during training.

**Limitation:** MSE treats all features equally. A feature with large absolute values dominates. Always use after StandardScaler.

---

### Metric 2: ELBO (for VAE)

**Formula:** `ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))`

**What it tells you:** The ELBO is the actual training objective for VAEs. Higher (less negative) is better. Report both total ELBO and its two components separately:
- Reconstruction term (E_q[log p(x|z)]): should decrease over training
- KL term (KL(q||p)): should increase gradually (from 0 to a stable value)

**Posterior collapse detection:** If KL term ≈ 0 after training, the encoder is outputting the prior — the latent space is not being used.

---

### Metric 3: Trustworthiness

**Formula:**
`T(k) = 1 - (2 / (nk(2n - 3k - 1))) Σ_{i=1}^{n} Σ_{j ∈ U_k(i)} (r(i,j) - k)`

Where `r(i,j)` is the rank of point j in original space among neighbors of i in latent space, and `U_k(i)` are false neighbors — points that are k-nearest in latent space but not in original.

**Implementation:**
```python
from sklearn.manifold import trustworthiness
T = trustworthiness(X_original, X_latent, n_neighbors=10)
print(f"Trustworthiness@10: {T:.4f}")  # 1.0 = perfect, <0.8 = poor
```

**What it tells you:** Precision-like metric. Are the k nearest neighbors in latent space also neighbors in original space? Detects false neighbors (points that are incorrectly brought close).

---

### Metric 4: Continuity

**Formula:** Complement of Trustworthiness — precision/recall role reversed. Detects true neighbors that were torn apart.

`C(k) = 1 - (2 / (nk(2n - 3k - 1))) Σ_{i=1}^{n} Σ_{j ∈ V_k(i)} (s(i,j) - k)`

Where `V_k(i)` are points that are k-nearest in original space but not in latent space, and `s(i,j)` is rank in latent space.

**Note:** sklearn's `trustworthiness` computes T only. Continuity requires a custom implementation or package like `trustworthiness_continuity` from various community libraries.

---

### Metric 5: FID Score (Fréchet Inception Distance — for VAE/generative)

**What it measures:** Quality and diversity of generated samples. Computes the Fréchet distance between the distribution of real image features (from InceptionV3) and generated image features.

**Formula:** `FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r Σ_g))`

**Lower is better.** A VAE with FID close to the data's test-set FID generates realistic, diverse images.

**Implementation:** Use `torchmetrics.image.fid.FrechetInceptionDistance` (PyTorch) or `tensorflow_gan.eval.frechet_inception_distance`.

**Caveat:** FID requires InceptionV3 (pretrained on ImageNet) and is only meaningful for natural images at sufficient resolution (≥75×75). Not applicable to tabular AEs.

---

### Metric 6: Downstream Task Accuracy

**The most important metric for DR.** Fit a simple linear classifier (LogisticRegression with low C) on the latent codes. If the latent space is useful, the classifier should do well.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Get latent codes
Z_train = encode(ae, X_train_scaled)
Z_test = encode(ae, X_test_scaled)

# Linear probe
clf = LogisticRegression(max_iter=1000, C=0.1)
clf.fit(Z_train, y_train)
accuracy = clf.score(Z_test, y_test)
print(f"Linear probe accuracy: {accuracy:.4f}")
```

**Baseline comparison:** Compare to PCA + LogisticRegression at the same latent dimension.

---

### Metric 7: Reconstruction Error as Anomaly Score

**For anomaly detection specifically:**
```python
X_recon = ae.predict(X_test)
recon_errors = np.mean((X_test - X_recon) ** 2, axis=1)  # per-sample MSE

# ROC-AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true_binary, recon_errors)  # y=1 for anomaly
print(f"AUC: {auc:.4f}")  # 0.9+ = excellent anomaly detector
```

---

### Metric 8: Disentanglement Score (for beta-VAE)

**FactorVAE metric (Higgins et al. 2017):** Train a low-capacity majority vote classifier to predict which factor of variation was held fixed, given difference vectors in latent space. High accuracy = high disentanglement.

**MIG (Mutual Information Gap):** Measures whether each latent dimension captures a unique factor of variation using mutual information.

**Practical note:** These metrics require ground-truth factor labels (e.g., dSprites dataset with known angles, scales, positions). Not available for real-world unstructured data. In practice, use latent traversal visualization: pick a single latent dimension, vary it over its range, decode, and visualize. A disentangled dimension changes exactly one visual attribute.

---

## 9. COMMON PITFALLS WITH CONCRETE EXAMPLES

### Pitfall 1: Not Scaling Features Before Fitting

**Symptom:** Reconstruction loss is dominated by one feature. MSE of 450 on a temperature sensor [0–500°C] vs. 0.002 on a normalized humidity [0–1]. The AE ignores humidity entirely.

**Fix:** Always apply `StandardScaler` before fitting:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
ae.fit(X_scaled, X_scaled)
```

---

### Pitfall 2: Treating MLPRegressor as an Encoder (sklearn API misunderstanding)

**Symptom:** User calls `ae.predict(X_new)` and uses the output as the low-dimensional representation. But `predict()` outputs the reconstruction (same dimensionality as input), not the bottleneck.

**Fix:** Use the manual encoder forward pass with `coefs_` and `intercepts_` (see §3 above). Or use PyTorch where encoder/decoder are explicit `nn.Module` objects.

---

### Pitfall 3: Posterior Collapse in VAE

**Symptom:** VAE trains to low total ELBO but KL term ≈ 0 after epoch 10. Sampling z ~ N(0,I) and decoding produces mean-blurred output regardless of z. The decoder has learned to ignore z.

**Cause:** The decoder is powerful enough to reconstruct X without using z. The optimizer finds the easy solution of setting q(z|x) ≈ p(z) (KL = 0) and using only the decoder's capacity.

**Fix:**
1. KL annealing: start beta=0, linearly increase to 1 over 50 epochs
2. Free bits: minimum KL per latent dimension (e.g., min KL per dim = 0.1 nats)
3. Use WAE instead (regularizes aggregate posterior, not per-sample)
4. Reduce decoder capacity (fewer decoder layers than encoder)

---

### Pitfall 4: max_iter=200 (sklearn default) is Insufficient

**Symptom:** Loss still rapidly decreasing at iteration 200. `convergence_warnings: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.`

**Fix:** Set `max_iter=1000` and `early_stopping=True`:
```python
ae = MLPRegressor(
    hidden_layer_sizes=(256, 128, 32, 128, 256),
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=30,
    random_state=42,
)
```

---

### Pitfall 5: Autoencoder Overfitting on Small Datasets

**Symptom:** Training reconstruction MSE = 0.003, test reconstruction MSE = 0.85. The AE has memorized the training set.

**Signs:** Very deep/wide architecture for a small dataset (e.g., 5-layer encoder for 500 samples).

**Fix:**
1. Increase `alpha` (L2 regularization): try 0.01–0.1
2. Add dropout (only possible in PyTorch; sklearn MLP has no dropout)
3. Use denoising AE (adds regularization pressure naturally)
4. Reduce network depth/width
5. Use `early_stopping=True` with larger `validation_fraction=0.2`

---

### Pitfall 6: Using AE Reconstruction Error for Anomaly Detection Without Calibrating the Threshold

**Symptom:** Anomaly detection seems to work on training data but produces too many false positives in production.

**Fix:** Set the anomaly threshold on a clean validation set (known normal data), not the training set. Use pyod's `contamination` parameter (estimated fraction of anomalies) or calibrate on held-out validation data:
```python
# Calibrate: set threshold at 99th percentile of normal reconstruction errors
train_errors = np.mean((X_train - ae.predict(X_train)) ** 2, axis=1)
threshold = np.percentile(train_errors, 99)
test_errors = np.mean((X_test - ae.predict(X_test)) ** 2, axis=1)
is_anomaly = test_errors > threshold
```

---

### Pitfall 7: Assuming AE Latent Space Has Meaningful Axes (Like PCA)

**Symptom:** User tries to interpret individual latent dimensions as "factors" without any guarantee. In a standard (non-beta) AE, the latent axes can be arbitrarily rotated combinations of features.

**Fix:** Use beta-VAE for interpretable, disentangled axes. Or apply PCA on the latent codes after encoding (post-hoc rotation). Or use ICA on the latent codes.

---

## 10. MATHEMATICAL FOUNDATIONS QUICK REFERENCE

### Standard Autoencoder

- **Encoder:** `z = f_θ(x)` — maps input x ∈ R^d to latent code z ∈ R^k, k << d
- **Decoder:** `x̂ = g_φ(z)` — maps latent code back to input space
- **Loss:** `L = (1/N) Σ ||x_i - g_φ(f_θ(x_i))||²` (MSE for continuous; BCE for binary)
- **Training:** Minimize L with respect to θ (encoder parameters) and φ (decoder parameters) simultaneously via backpropagation

### Tied Weights

- **Constraint:** `W_decoder = W_encoder^T`
- **Benefit:** Halves parameter count, regularizes, often comparable performance
- **In sklearn:** Not natively supported (MLPRegressor has independent encoder/decoder weights)
- **In practice:** Implemented manually in PyTorch by sharing weight tensors

### Denoising Autoencoder

- **Corrupt:** `x̃ ~ q_D(x̃|x)` (add noise: Gaussian N(0, σ²I), Bernoulli masking, salt-and-pepper)
- **Loss:** `L_DAE = (1/N) Σ ||x_i - g_φ(f_θ(x̃_i))||²`
- **Key insight:** Minimizing this loss is equivalent to learning the score function ∇_x log p(x) — the denoising AE implicitly estimates the data density gradient. This is why denoising representations transfer well.

### Variational Autoencoder (VAE)

- **Encoder:** outputs parameters of posterior distribution: `(μ, σ) = f_θ(x)` where z | x ~ N(μ, σ²I)
- **Reparameterization trick:** `z = μ + σ ⊙ ε` where `ε ~ N(0, I)` — makes sampling differentiable
- **Decoder:** `p_φ(x|z)` — generative model (Gaussian: output is mean of reconstruction)
- **ELBO loss:**
  ```
  L_VAE = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        = reconstruction_loss - KL_loss
  ```
- **KL term (closed form for Gaussian):**
  ```
  KL(N(μ, σ²) || N(0,1)) = (1/2) Σ_j (1 + log(σ_j²) - μ_j² - σ_j²)
  ```
- **PyTorch implementation:**
```python
def vae_loss(x, x_recon, mu, log_var, beta=1.0):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss
```

### beta-VAE

- **Modified ELBO:**
  ```
  L_β = E_q[log p(x|z)] - β · KL(q(z|x) || p(z))
  ```
- With β > 1: stronger pressure on KL term → each latent dimension must efficiently encode one independent factor

### Contrastive Loss (SimCLR NT-Xent)

- **NT-Xent loss for one positive pair (i, j):**
  ```
  L(i,j) = -log( exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )
  ```
  Where `sim(u,v) = u·v / (||u|| ||v||)` is cosine similarity, τ is temperature (typically 0.07–0.5)

### Wasserstein AE (WAE-MMD)

- **Loss:**
  ```
  L_WAE = E_{p_data}[c(x, g_φ(z))] + λ · MMD(q_Z || p_Z)
  ```
- Where `c` is a cost function (e.g., squared distance), the first term is the reconstruction cost, and MMD (Maximum Mean Discrepancy) measures how far the aggregate posterior `q_Z = ∫q_φ(z|x)p_data(x)dx` deviates from the prior `p_Z = N(0,I)`.
- **MMD (with RBF kernel):**
  ```
  MMD²(q,p) = E_{q,q}[k(z,z')] - 2E_{q,p}[k(z,z')] + E_{p,p}[k(z,z')]
  ```

---

## APPENDIX: KEY URLs AND REFERENCES

### Primary Paper URLs
- Hinton & Salakhutdinov 2006: https://www.science.org/doi/10.1126/science.1127647
- Vincent et al. 2008: https://dl.acm.org/doi/10.1145/1390156.1390294
- Vincent et al. 2008 PDF: https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
- Kingma & Welling 2014: https://arxiv.org/abs/1312.6114
- He et al. 2020 MoCo: https://arxiv.org/abs/1911.05722
- Chen et al. 2020 SimCLR: https://arxiv.org/abs/2002.05709 (GitHub: github.com/google-research/simclr)
- Sainburg et al. 2021 Param UMAP: https://pmc.ncbi.nlm.nih.gov/articles/PMC8516496/
- WAE: https://openreview.net/forum?id=HkL7n1-0b

### Package Documentation URLs
- sklearn MLPRegressor 1.5: https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPRegressor.html
- PyOD: https://pyod.readthedocs.io/
- scvi-tools: https://docs.scvi-tools.org/en/stable/
- pythae: https://github.com/clementchadebec/benchmark_VAE
- Parametric UMAP: https://umap-learn.readthedocs.io/en/latest/parametric_umap.html
- Keras molecule generation: https://keras.io/examples/generative/molecule_generation/

### Key GitHub Repos
- SimCLR: https://github.com/google-research/simclr
- scvi-tools: https://github.com/scverse/scvi-tools
- umap-learn: https://github.com/lmcinnes/umap

---

*Research dossier complete. All URLs verified as of June 2026. Sklearn API parameters verified against https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPRegressor.html*
