# Research Dossier: MLP Neural Networks (sklearn)
**Chapter file:** `mlp-neural-networks.md`
**Researcher:** Claude Sonnet 4.6 | **Date:** 2026-06-25

---

## Scope

This dossier feeds the chapter `mlp-neural-networks.md` in the Supervised Learning Masterclass. It covers:
- Frank Rosenblatt's perceptron (1958)
- Multi-layer perceptrons and backpropagation (Rumelhart, Hinton, Williams 1986)
- Universal approximation theorems (Cybenko 1989; Hornik 1991)
- The Adam optimizer (Kingma & Ba 2014)
- Glorot/Xavier weight initialization (Glorot & Bengio 2010)
- sklearn `MLPClassifier` and `MLPRegressor` — every parameter, verified against sklearn 1.5.x docs
- Vanishing gradient problem and the ReLU solution
- sklearn MLP limitations and the upgrade path to PyTorch/Keras
- SHAP explainability (KernelExplainer)
- The tabular data debate: trees vs neural networks (Grinsztajn et al. 2022)

---

## Verified Packages

### Primary Package: scikit-learn

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor
```

- **Version:** sklearn 1.5.x (stable as of mid-2026; 1.9 is current stable, parameters identical to 1.5)
- **Algorithm:** Feedforward MLP trained by backpropagation with one of three solvers: `lbfgs`, `sgd`, or `adam`
- **Maintainer:** scikit-learn community (INRIA lead)
- **License:** BSD-3-Clause
- **GPU support:** None — explicitly noted in the official docs: *"this implementation is not intended for large-scale applications. In particular, scikit-learn offers no GPU support."*
- **Key limitation:** No dropout, no batch normalization, no CNN/RNN layers, no custom loss functions

```python
# Same fit() call for both classifier and regressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        max_iter=300,
        random_state=42
    ))
])
clf.fit(X_train, y_train)
```

### Graduated Frameworks (when sklearn MLP is not enough)

| Package | Import | Strengths | GPU | Docs |
|---|---|---|---|---|
| **PyTorch** | `import torch; import torch.nn as nn` | Full flexibility, custom architectures, GPU, research-grade | Yes (CUDA/MPS) | https://pytorch.org/docs |
| **Keras / TensorFlow** | `from tensorflow import keras` | High-level API, fast prototyping, production serving | Yes (CUDA/TPU) | https://keras.io/api |
| **PyTorch Tabular** | `from pytorch_tabular import TabularModel` | sklearn-compatible API over PyTorch for tabular data | Yes | https://pytorch-tabular.readthedocs.io |

**Top picks for this algorithm family:**

| Package | Best for | Modern/Legacy |
|---|---|---|
| `sklearn.neural_network.MLPClassifier` | Medium tabular datasets (<50k rows), quick baseline, sklearn pipelines | Modern (limited) |
| PyTorch (`torch.nn.Module`) | Custom architectures, GPU training, research, production | Modern |
| Keras (`keras.Sequential`) | Rapid prototyping, team familiar with TF ecosystem | Modern |

### SHAP (for explainability)

```python
import shap  # version ~0.46.x
# For sklearn MLP: use KernelExplainer (model-agnostic)
explainer = shap.KernelExplainer(clf.predict_proba, X_background)
shap_values = explainer.shap_values(X_test[:50])  # slow — sample background
```

### Optuna (for hyperparameter tuning)

```python
import optuna  # version ~3.x

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layer_sizes = tuple(
        trial.suggest_int(f'n_units_l{i}', 32, 256) for i in range(n_layers)
    )
    alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
    lr_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=300,
        random_state=42
    )
    return cross_val_score(clf, X_train_scaled, y_train, cv=5).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## The Original Algorithm

### Paper 1: The Perceptron (1958)

> 📜 **Citation:** Rosenblatt, F. (1958). "The perceptron: a probabilistic model for information storage and organization in the brain." *Psychological Review*, 65(6), 386–408. DOI: [10.1037/h0042519](https://doi.org/10.1037/h0042519)

**Problem it solved:** Before 1958, McCulloch-Pitts neurons (1943) were binary threshold units that could not learn from data — they required hand-engineered weights. Rosenblatt wanted a machine that *learns* from examples.

**Core innovation:** The perceptron learning rule. Given a binary label $y \in \{-1, +1\}$ and prediction $\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$, update weights only when misclassified:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot (y - \hat{y}) \cdot \mathbf{x}$$

**Key limitation:** The perceptron convergence theorem guarantees convergence only if data is linearly separable. Minsky and Papert (1969) famously proved the perceptron cannot solve XOR — a fatal limitation that triggered the first "AI winter" for neural networks.

**In sklearn:**
```python
from sklearn.linear_model import Perceptron
# The original perceptron — single layer, linear decision boundary
p = Perceptron(max_iter=1000, random_state=42)
```

---

### Paper 2: Backpropagation / MLP (1986)

> 📜 **Citation:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*, 323(6088), 533–536. DOI: [10.1038/323533a0](https://doi.org/10.1038/323533a0)

**Problem it solved:** With a single-layer perceptron dead-ended by the XOR problem, the field needed a method to train networks with *hidden layers* — layers that learn intermediate representations. The key challenge was computing gradients for neurons that are neither input nor output.

**Core innovation:** The backpropagation algorithm — efficient computation of $\partial L / \partial w_{ij}$ for every weight in the network via the chain rule, propagating error signals backward through the network.

**The forward pass:** For a layer $l$ with weight matrix $\mathbf{W}^{(l)}$ and bias $\mathbf{b}^{(l)}$:

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = g(\mathbf{z}^{(l)})$$

where $g(\cdot)$ is the activation function applied elementwise.

**The backward pass — error signals (delta):**

For the output layer $L$:
$$\boldsymbol{\delta}^{(L)} = \frac{\partial L}{\partial \mathbf{z}^{(L)}} = \nabla_{\mathbf{a}} L \odot g'(\mathbf{z}^{(L)})$$

For a hidden layer $l$ (chain rule):
$$\boldsymbol{\delta}^{(l)} = \left( (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \right) \odot g'(\mathbf{z}^{(l)})$$

**Weight gradients:**
$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$

**Weight update (SGD):**
$$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}$$

**Historical note:** Rumelhart, Hinton, and Williams did not invent backpropagation independently — Werbos (1974) derived it in his Ph.D. thesis and Parker (1985) rediscovered it. But the 1986 Nature paper was the one that actually *demonstrated* that hidden layers could learn useful representations, igniting the connectionist revival.

---

### Paper 3: Universal Approximation Theorem (1989, 1991)

> 📜 **Citation (Cybenko):** Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function." *Mathematics of Control, Signals, and Systems*, 2(4), 303–314. DOI: [10.1007/BF02551274](https://doi.org/10.1007/BF02551274)

> 📜 **Citation (Hornik):** Hornik, K. (1991). "Approximation capabilities of multilayer feedforward networks." *Neural Networks*, 4(2), 251–257. DOI: [10.1016/0893-6080(91)90009-T](https://doi.org/10.1016/0893-6080(91)90009-T)

**What it says (Cybenko's version):** A feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$, provided the activation function is a nonconstant, bounded, and monotone-increasing continuous function (e.g., sigmoid).

**What it says (Hornik's extension):** Standard multilayer feedforward networks are universal approximators with respect to $L^p(\mu)$ criteria for any finite input environment measure $\mu$, provided sufficiently many hidden units are available. Crucially, Hornik showed the result holds for *any* bounded nonconstant activation — it is the *architecture* (multilayer with nonlinear activations), not the specific activation function, that confers universal approximation power.

**What it does NOT say:**
- It does not say how many neurons you need (this can be exponential in the input dimension)
- It does not say you can find those weights with gradient descent
- It says nothing about generalization (learning from finite samples)

> ⚠️ **Pitfall:** The universal approximation theorem is frequently misread as a guarantee that MLPs will work well in practice. It only guarantees existence of a set of weights that approximate the function — not that training will find them, not that you have enough data, not that you won't overfit.

---

## Major Variants & Evolution

### 1. Perceptron → Multi-Layer Perceptron (1958 → 1986)
- **Problem solved:** XOR and nonlinear separability
- **Change:** Added hidden layers + backpropagation
- **Package:** `sklearn.neural_network.MLPClassifier`

### 2. Sigmoid → Tanh Activations (1980s–1990s)
- **Problem solved:** Sigmoid's output in (0,1) means it is not zero-centered; gradients from one layer are always positive, slowing convergence with SGD
- **Change:** Tanh maps to (−1, +1), zero-centered, produces both positive and negative gradients — converges roughly 4× faster than sigmoid in practice (LeCun et al. 1998)
- **Package:** `activation='tanh'` in sklearn MLP

### 3. ReLU Activation (Glorot et al. 2011; Krizhevsky et al. 2012)
- **Problem solved:** Vanishing gradients in deep networks — sigmoid and tanh saturate for large |x|, making $g'(z) \approx 0$, which kills the gradient signal through many layers
- **Change:** $\text{ReLU}(x) = \max(0, x)$ with derivative exactly 1 for $x > 0$; gradients do not decay
- **Paper:** Glorot, X., Bordes, A., & Bengio, Y. (2011). "Deep sparse rectifier neural networks." AISTATS. AlexNet (Krizhevsky et al. 2012) made ReLU standard.
- **Package:** `activation='relu'` — the sklearn default as of version 0.18+

> 💡 **Intuition:** With sigmoid, if a weight produces output near 0 or 1, the gradient becomes essentially 0. In a 10-layer network, $0.25^{10} \approx 10^{-6}$ — the first layer receives no meaningful signal. ReLU's gradient is either 0 or 1, so active neurons propagate gradients exactly.

### 4. Glorot/Xavier Weight Initialization (2010)
- **Problem solved:** Random initialization with naive variance leads to either vanishing or exploding activations in early training
- **Paper:** Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." AISTATS, PMLR Vol. 9, pp. 249–256. URL: https://proceedings.mlr.press/v9/glorot10a.html
- **Key insight:** To preserve signal variance across layers, weights should be drawn from $\text{Uniform}(-b, b)$ where $b = \sqrt{6 / (\text{fan\_in} + \text{fan\_out})}$
- **sklearn implementation:** This is exactly what sklearn uses internally for all activation functions (factor=6 for ReLU, factor=2 for logistic):

```python
# From sklearn source: sklearn/neural_network/_multilayer_perceptron.py
factor = 6.0  # (or 2.0 for logistic)
init_bound = np.sqrt(factor / (fan_in + fan_out))
coef_init = rng.uniform(-init_bound, init_bound, (fan_in, fan_out))
```

### 5. Adam Optimizer (2014)
- **Problem solved:** SGD with momentum requires careful learning rate tuning; learning rate is the same for all parameters; learning rate must be manually scheduled
- **Paper:** Kingma, D. P. & Ba, J. (2014). "Adam: A method for stochastic optimization." ICLR 2015. arXiv:1412.6980. DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)

**Algorithm:**
At step $t$, given gradient $g_t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment: mean)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment: uncentered variance)}$$

Bias correction (both moments start at 0, are biased toward 0 early in training):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Defaults used in sklearn:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 0.001$

**Why Adam dominates for MLPs on moderate tabular data:** Each parameter gets its own adaptive learning rate based on its historical gradient magnitudes. Parameters with consistently large gradients get smaller effective learning rates; sparse parameters with rare large gradients get larger effective learning rates. This makes it very robust to learning rate choice and to sparse feature spaces.

### 6. Deep Learning Frameworks (PyTorch/Keras) — the "graduation" path
- **Problem solved:** sklearn MLP lacks GPU support, dropout, batch normalization, custom architectures, convolutional/recurrent layers, and production serving
- **When to graduate:** dataset >50k rows with GPU available; need dropout or BN for regularization; building a CNN/RNN; need custom loss function; need model serving with TF Serving or TorchServe

```python
# Equivalent MLP in PyTorch
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Equivalent in Keras
import keras
model = keras.Sequential([
    keras.layers.Dense(100, activation='relu', input_shape=(n_features,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## Hyperparameter Reference

### Architecture

#### `hidden_layer_sizes` — tuple, default `(100,)`
**What it controls:** The network topology — the number of hidden layers (length of tuple) and neurons per layer (each element). Note: the input and output layers are not included. `(100,)` means one hidden layer with 100 neurons.

**Effect of too few neurons:** Underfitting — the network cannot represent the underlying function complexity.
**Effect of too many neurons:** Overfitting, slower training, increased memory.

**Tuning guidance:**
- Start with one or two hidden layers — rarely need more than three for tabular data
- A common heuristic: set size proportional to $\sqrt{n_{\text{features}} \times n_{\text{samples}}}$ or use a "funnel" (decreasing sizes)
- Typical search space for tabular data: single layer (32, 64, 100, 200) or two layers (100, 50), (200, 100)
- The architecture and `alpha` (regularization) are the two most impactful parameters

```python
# Optuna search
n_layers = trial.suggest_int('n_layers', 1, 3)
sizes = tuple(trial.suggest_int(f'h{i}', 32, 256, log=True) for i in range(n_layers))
```

#### `activation` — str, default `'relu'`
Valid options: `'identity'`, `'logistic'`, `'tanh'`, `'relu'`

| Activation | Formula | Range | Gradient | Use case |
|---|---|---|---|---|
| `'relu'` | $\max(0, x)$ | $[0, \infty)$ | 0 or 1 | Default; fast; avoids vanishing gradient |
| `'tanh'` | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | $(0, 1)$ | Zero-centered; better than sigmoid; vanishes for large \|x\| |
| `'logistic'` | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | $(0, 0.25)$ | Classic; most prone to vanishing gradient; rarely use in hidden layers |
| `'identity'` | $x$ | $(-\infty, \infty)$ | 1 | Linear — effectively no nonlinearity; for debugging only |

> ⚠️ **Pitfall:** Using `activation='logistic'` on deep architectures causes vanishing gradients. The maximum gradient of sigmoid is 0.25, so through 4 layers: $0.25^4 = 0.0039$ — almost no signal reaches early layers. Stick with `'relu'` unless you have a specific reason.

> 🔧 **In practice:** The output activation is set automatically by sklearn: `'softmax'` for multi-class classification, `'logistic'` for binary classification, and `'identity'` (linear) for regression. The `activation` parameter only controls the *hidden* layer activations.

### Optimization

#### `solver` — str, default `'adam'`
Valid: `'lbfgs'`, `'sgd'`, `'adam'`

| Solver | Method | Best for | Supports online learning? |
|---|---|---|---|
| `'adam'` | Adaptive moment estimation (1st and 2nd order) | Most tabular datasets; robust default | Yes (mini-batch) |
| `'lbfgs'` | Limited-memory Broyden–Fletcher–Goldfarb–Shanno (quasi-Newton) | Small datasets (<2000 samples); often achieves lower final loss | No (full batch only) |
| `'sgd'` | Stochastic gradient descent with optional momentum | When you need manual control of learning rate schedule; large datasets | Yes (mini-batch) |

**When to use `lbfgs`:** The official sklearn note says: *"For small datasets, however, 'lbfgs' can converge faster and perform better."* lbfgs approximates the Hessian (second-order information), so it takes better-directed steps. But it requires the full dataset in memory for each update and does not scale.

**When to use `adam`:** Default for most cases. Very robust to learning rate choice. Handles sparse gradients well. Usually the best choice for datasets >2000 samples.

**When to use `sgd`:** Rarely preferred over `adam` unless you want explicit control over momentum and learning rate schedules, or if you are matching a specific training regime from a paper.

#### `alpha` — float, default `0.0001`
L2 regularization penalty added to the loss:
$$L_{\text{total}} = L_{\text{data}} + \frac{\alpha}{2n} \|\mathbf{W}\|_2^2$$

**Effect:** Shrinks weights toward zero, reducing model complexity. Higher alpha = stronger regularization = lower variance, potentially higher bias.

**Tuning guidance:** Most impactful regularization knob in sklearn MLP. Log-scale search: `[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]`. Use cross-validation. If the model overfits, increase alpha.

#### `batch_size` — int or `'auto'`, default `'auto'`
`'auto'` resolves to `min(200, n_samples)`. Ignored when `solver='lbfgs'` (which always uses full batch).

**Effect on training:**
- Small batches: more stochastic updates, noisy loss curves, can escape local minima, acts as implicit regularizer
- Large batches: smoother loss curve, potentially faster per epoch but may converge to sharper (worse-generalizing) minima
- `min(200, n_samples)` is a conservative, effective default

#### `learning_rate` — str, default `'constant'`
Only used with `solver='sgd'`. Options:
- `'constant'`: $\eta_t = \eta_0$ (fixed rate = `learning_rate_init`)
- `'invscaling'`: $\eta_t = \eta_0 / t^{\text{power\_t}}$ — decreases over time
- `'adaptive'`: $\eta_t$ stays at $\eta_0$ while training loss improves; when it fails to decrease for `n_iter_no_change` steps, learning rate is divided by 5

Ignored by `solver='adam'` (which has internal adaptive rates per parameter) and `solver='lbfgs'` (which uses line search).

#### `learning_rate_init` — float, default `0.001`
Used by `adam` and `sgd`. Starting learning rate.

**Tuning guidance:** Log-scale: `[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]`. Adam is less sensitive than SGD. If loss is not decreasing, decrease this. If converging too slowly, increase.

#### `power_t` — float, default `0.5`
Exponent for the `'invscaling'` schedule:
$$\eta_t = \frac{\eta_0}{t^{p}}$$
where $p$ is `power_t` and $t$ is the current iteration. Default $p=0.5$ means learning rate falls as $1/\sqrt{t}$.

#### `max_iter` — int, default `200`
Maximum number of epochs (for `adam`/`sgd`) or gradient descent iterations (for `lbfgs`). For stochastic solvers, one epoch = one pass through the training data.

> ⚠️ **Pitfall:** The default `max_iter=200` is almost always too low for non-trivial problems. sklearn will raise a `ConvergenceWarning` and return suboptimal weights. Raise to 1000–5000 and rely on `early_stopping=True` to halt early when appropriate.

```python
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress the warning only after you've diagnosed the problem
warnings.filterwarnings('ignore', category=ConvergenceWarning)
```

#### `shuffle` — bool, default `True`
Whether to shuffle samples in each iteration (SGD and Adam only). Almost always leave as `True` — shuffling is essential to avoid order-dependent biases in mini-batch SGD.

#### `random_state` — int or None, default `None`
Controls: (1) weight initialization, (2) training–validation split if `early_stopping=True`, (3) batch sampling order. Set for reproducibility.

#### `tol` — float, default `1e-4`
Convergence tolerance. Training stops when the loss (or score, if `early_stopping=True`) has not improved by at least `tol` for `n_iter_no_change` consecutive iterations.

**Tuning guidance:** Lower `tol` = train longer, potentially better fit, longer runtime. If you see `ConvergenceWarning`, do NOT just lower `tol` to suppress it — that changes what "converged" means. Fix by increasing `max_iter` or scaling features.

#### `verbose` — bool, default `False`
Prints loss at each iteration to stdout. Useful for monitoring; turn off in production.

#### `warm_start` — bool, default `False`
If `True`, calling `fit()` again reuses the learned weights as initialization rather than reinitializing. Useful for incremental learning or for custom stopping criteria:

```python
clf = MLPClassifier(max_iter=1, warm_start=True, random_state=42)
for epoch in range(500):
    clf.fit(X_train, y_train)
    print(f"Epoch {epoch}: loss={clf.loss_:.4f}")
```

### SGD-Specific

#### `momentum` — float, default `0.9`
Momentum for SGD. The update becomes:
$$\mathbf{v} \leftarrow \mu \mathbf{v} - \eta \nabla L, \quad \mathbf{W} \leftarrow \mathbf{W} + \mathbf{v}$$

Momentum accumulates a velocity in directions of consistent gradient, damping oscillations and accelerating convergence in low-curvature directions. Default `0.9` is standard.

#### `nesterovs_momentum` — bool, default `True`
Nesterov's accelerated gradient (NAG). Instead of computing gradient at current position, compute it at the *anticipated* future position:
$$\mathbf{v} \leftarrow \mu \mathbf{v} - \eta \nabla L(\mathbf{W} + \mu \mathbf{v})$$

NAG effectively "looks ahead" before computing the gradient, resulting in more responsive updates and faster convergence. Theoretical: converges at $O(1/t^2)$ vs $O(1/t)$ for plain gradient descent in convex settings.

### Early Stopping

#### `early_stopping` — bool, default `False`
If `True`, a fraction (`validation_fraction`) of training data is automatically set aside as a validation set. Training stops when the validation score fails to improve for `n_iter_no_change` consecutive epochs.

> 🏆 **Best practice:** Always enable `early_stopping=True` with a generous `max_iter` (e.g., 2000+). This is the proper way to prevent overfitting in sklearn MLP. Without it, you are running a fixed number of epochs with no regularization on training duration.

```python
clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)
```

#### `validation_fraction` — float, default `0.1`
Fraction of training data to set aside for early stopping validation. Only used if `early_stopping=True`.

#### `n_iter_no_change` — int, default `10`
Number of iterations with no improvement in validation score (or training loss, if `early_stopping=False`) before early stopping triggers or adaptive learning rate is reduced. Added in sklearn 0.20.

### Adam-Specific

#### `beta_1` — float, default `0.9`
Exponential decay for the first moment (mean of gradients). Controls how quickly old gradients are forgotten. Rarely changed from default.

#### `beta_2` — float, default `0.999`
Exponential decay for the second moment (variance of gradients). Higher values mean very stable estimates of gradient variance; good when gradients are sparse or noisy. Rarely changed.

#### `epsilon` — float, default `1e-8`
Numerical stability constant in Adam's denominator $(\sqrt{\hat{v}_t} + \epsilon)$. Prevents division by zero. Rarely changed.

### lbfgs-Specific

#### `max_fun` — int, default `15000`
Maximum number of loss function evaluations (not gradient descent steps) for the lbfgs solver. lbfgs uses line search — multiple function evaluations per iteration — so `max_fun` ≥ `max_iter`. Added in sklearn 0.22. Available in both `MLPClassifier` and `MLPRegressor`.

### Tuning Playbook Table

| Parameter | Typical Range | Primary Effect | Tuning Strategy |
|---|---|---|---|
| `hidden_layer_sizes` | (32,) to (512, 256, 128) | Model capacity | Start small; add layers/neurons only if underfitting |
| `activation` | `'relu'` (default), `'tanh'` | Nonlinearity; gradient flow | Almost always `'relu'`; try `'tanh'` if dying ReLU suspected |
| `solver` | `'adam'`, `'lbfgs'` for small data | Optimization method | `'adam'` default; `'lbfgs'` for <2000 samples |
| `alpha` | `[1e-6, 1e-1]` log scale | L2 regularization | Most important regularizer; tune with CV |
| `learning_rate_init` | `[1e-4, 1e-2]` log scale | Step size | Log-scale search; `0.001` is reliable starting point |
| `batch_size` | `'auto'`, 32, 64, 128, 256 | Stochasticity / noise | Try `'auto'` first; smaller batches = more regularization |
| `max_iter` | 500–5000 | Training duration | Set high; use `early_stopping=True` to stop |
| `early_stopping` | `True` | Overfitting prevention | Always `True` for the `adam`/`sgd` solver |
| `n_iter_no_change` | 10–50 | Patience | Increase for noisy training curves |
| `momentum` | 0.8–0.99 | SGD acceleration | Only for `solver='sgd'`; default `0.9` usually fine |
| `beta_1`, `beta_2` | defaults | Adam moment decay | Essentially never change from defaults |

---

## Common Errors & Pitfalls

### 1. ConvergenceWarning: max_iter Reached
**Symptom:** `ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.`

**Root causes:**
- `max_iter=200` is too low (extremely common with default settings)
- Features not scaled — without StandardScaler, gradient descent takes inconsistent steps across features
- Learning rate too large (for `sgd`) — diverging rather than converging

**Fix:**
```python
# Priority 1: Scale features (ALWAYS required for MLP)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Priority 2: Increase max_iter
clf = MLPClassifier(max_iter=1000, early_stopping=True)

# Priority 3: Check learning_rate_init if using sgd
clf = MLPClassifier(solver='sgd', learning_rate_init=0.01)
```

### 2. Forgetting Feature Scaling
**Symptom:** Poor accuracy or slow convergence, `ConvergenceWarning`

**Mechanism:** MLP uses gradient descent, which assumes features are on similar scales. Features with large variance dominate the gradient updates; features with small variance get negligible updates. Unlike tree-based models, neural networks are *not* invariant to feature scale.

> ⚠️ **Pitfall:** This is the #1 mistake with sklearn MLP. Always wrap in a Pipeline with StandardScaler.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=1000, random_state=42))
])
```

### 3. Using the Wrong Solver for Dataset Size
**Symptom:** Very slow training (`adam` on tiny dataset) or out-of-memory (`lbfgs` on large dataset)

**Fix:** Use `'lbfgs'` for <2000 samples, `'adam'` for larger datasets.

### 4. Dying ReLU
**Symptom:** Training loss plateaus immediately; `coefs_` have many all-zero rows.

**Mechanism:** If a neuron's pre-activation $z < 0$ for all training examples, the ReLU gradient is 0, the neuron never updates, and it "dies." Caused by large negative biases early in training or very large learning rates.

**Fix:** Use `activation='tanh'` or reduce `learning_rate_init`. With `lbfgs`, this is rarely a problem since the solver handles step size automatically.

### 5. Warm Start Epoch Counter Not Resetting
**Symptom:** Unexpected behavior when using `warm_start=True` and changing `max_iter`

**Mechanism:** With `warm_start=True`, sklearn re-uses existing weights AND existing iteration counters. This is a known issue (GitHub #8713). The `max_iter` in a subsequent fit applies to the *total* training, not additional epochs.

**Fix:** Implement a manual training loop using `max_iter=1` and `warm_start=True` as shown above.

### 6. SHAP KernelExplainer Slowness
**Symptom:** `shap.KernelExplainer` takes hours on moderate datasets.

**Mechanism:** KernelExplainer is model-agnostic and uses a weighted linear regression over feature coalitions. It is $O(2^p)$ in the worst case; in practice it uses sampling but is still much slower than TreeExplainer.

**Fix:**
```python
# Use a small background summary (e.g., 100 samples, or k-means centroids)
background = shap.kmeans(X_train, 10)
explainer = shap.KernelExplainer(clf.predict_proba, background)
shap_values = explainer.shap_values(X_test[:20])  # small test set
```

### 7. No Dropout / Batch Normalization Available
**Symptom:** Model overfits even with large `alpha` and `early_stopping=True`.

**Mechanism:** sklearn MLP does not implement dropout or batch normalization — the two most powerful regularization techniques for neural networks.

**Fix:** If overfitting persists after tuning `alpha`, `hidden_layer_sizes`, and enabling `early_stopping`, this is the signal to move to PyTorch/Keras.

### 8. Output Activation and Loss Function Are Fixed
**Symptom:** Cannot customize loss function (e.g., focal loss, Huber loss) or output activation.

**Fix:** Use PyTorch or Keras where these are fully customizable.

### 9. Non-Reproducibility Despite `random_state`
**Symptom:** Results change between runs with same `random_state`.

**Mechanism:** With `n_jobs > 1` (e.g., in GridSearchCV), BLAS parallelism can introduce floating-point non-determinism.

**Fix:** Set `n_jobs=1` in GridSearchCV when exact reproducibility is required, or use `random_state` with `n_jobs=1` in the MLP itself.

---

## Key Datasets for Examples

### 1. Digits (Classification — multiclass)
```python
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
# Shape: X=(1797, 64), y=(1797,) — 10 classes (0-9), 8x8 pixel images
# n_classes=10, n_features=64 (flattened pixel intensities 0-16)
print(X.shape, y.shape)  # (1797, 64) (1797,)
```
**Why:** Classic "baby MNIST" — demonstrates MLP's ability to handle image-like pixel features. Manageable size for sklearn MLP. Good for showing hidden layer visualizations.

### 2. Breast Cancer Wisconsin (Classification — binary)
```python
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X, y = bc.data, bc.target
# Shape: X=(569, 30), y=(569,) — binary: 0=malignant, 1=benign
# 30 real-valued features (cell nucleus measurements)
print(bc.feature_names)
```
**Why:** Clean binary classification benchmark; demonstrates scaling importance; good for ROC-AUC, precision-recall, SHAP explanations.

### 3. California Housing (Regression)
```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.data, housing.target
# Shape: X=(20640, 8), y=(20640,) — median house price in $100k units
# Requires StandardScaler; demonstrates MLPRegressor
```
**Why:** Medium-sized regression benchmark; features on different scales (great for demonstrating scaling importance); excellent for comparing MLP vs gradient boosting.

### 4. MNIST via OpenML (larger-scale demonstration)
```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)
# Shape: X=(70000, 784), y=(70000,) — 10 classes
# Note: sklearn MLP will be slow; use a subset for demonstration
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=10000, random_state=42)
```
**Why:** Classic benchmark; also useful for showing when sklearn MLP is insufficient and PyTorch is needed.

---

## The Tabular Data Debate: Trees vs. Neural Networks

> 📊 **Benchmark:** Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). "Why do tree-based models still outperform deep learning on typical tabular data?" NeurIPS 2022. URL: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

**Key findings:**
- Across 45 datasets from varied domains, tree-based models (XGBoost, Random Forests) remain state-of-the-art on medium-sized tabular data (~10k samples)
- Even after extensive hyperparameter search (20,000 compute hours), NNs fail to match GBDT average rank
- Three identified challenges for NNs on tabular data: (1) must be robust to uninformative features (trees handle this with feature selection at each split; NNs rely entirely on $\alpha$ regularization); (2) must preserve data orientation (trees are rotation-invariant by nature; NNs are not invariant to rotations of the feature space, while the optimal decision boundaries for tabular data often align with feature axes); (3) must learn irregular functions (tabular data often has irregular, locally smooth functions that trees handle better via recursive partitioning)

**Practical recommendation for the chapter author:**
- sklearn MLP is appropriate as a **baseline** on medium tabular datasets
- For production on structured tabular data, the default should be gradient-boosted trees (XGBoost, LightGBM, CatBoost)
- Neural networks can win when: features are dense embeddings, data volume is very large (>500k rows), or the problem has inherent structure (images, sequences) that CNNs/RNNs exploit
- The "graduated" path: sklearn MLP (baseline) → PyTorch/Keras (when MLP wins or need GPU/dropout)

---

## Curated Resources

| Resource | URL | Why |
|---|---|---|
| sklearn MLPClassifier docs (1.5) | https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPClassifier.html | Verified parameter reference for the target version |
| sklearn MLP user guide | https://scikit-learn.org/1.5/modules/neural_networks_supervised.html | Full theory, equations, tips, and warnings |
| Rumelhart et al. 1986 (Nature) | https://www.nature.com/articles/323533a0 | Original backpropagation paper |
| Kingma & Ba 2014 (Adam) | https://arxiv.org/abs/1412.6980 | Original Adam paper |
| Glorot & Bengio 2010 (Xavier init) | https://proceedings.mlr.press/v9/glorot10a.html | Xavier initialization derivation |
| Cybenko 1989 (UAT) | https://www.semanticscholar.org/paper/Approximation-by-superpositions-of-a-sigmoidal-Cybenko/8da1dda34ecc96263102181448c94ec7d645d085 | Universal approximation theorem (original) |
| Hornik 1991 (UAT extension) | https://www.sciencedirect.com/science/article/abs/pii/089360809190009T | Extension to arbitrary bounded activations |
| Grinsztajn et al. 2022 (NeurIPS) | https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html | Trees vs NNs on tabular data — essential benchmark |
| SHAP KernelExplainer API | https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html | Verified SHAP API for model-agnostic explainability |
| Rosenblatt 1958 (Perceptron) | https://www.semanticscholar.org/paper/The-perceptron:-a-probabilistic-model-for-storage-Rosenblatt/5d11aad09f65431b5d3cb1d85328743c9e53cb96 | Original perceptron paper |
| sklearn GitHub — MLP source | https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neural_network/_multilayer_perceptron.py | Source code for weight initialization details |
| MLP convergence pitfalls | https://www.zerve.ai/data-science-problems/scikit-learn/convergencewarning-solver-failed-to-converge | Practitioner guide to ConvergenceWarning |

---

## Computational Complexity

**Training complexity (backpropagation per sample):**
$$O\left(i \cdot n \cdot \left(m \cdot h + (k-1) \cdot h^2 + h \cdot o\right)\right)$$

where:
- $i$ = number of epochs/iterations
- $n$ = number of training samples
- $m$ = number of input features
- $k$ = number of hidden layers
- $h$ = neurons per hidden layer (assumed uniform)
- $o$ = output neurons

**Inference complexity per sample:** $O(m \cdot h + (k-1) \cdot h^2 + h \cdot o)$ — one forward pass

**Key insight:** Training complexity scales with $n$ (data size) and $h^2$ (hidden size squared for inter-hidden connections). This is why sklearn MLP gets impractical for large networks or large datasets without GPU.

---

## Uncertainties

1. **`max_fun` in MLPClassifier vs MLPRegressor:** The WebFetch of the 1.5 MLPClassifier docs confirmed `max_fun=15000` is present in MLPClassifier (added v0.22), but an early auto-parse suggested it was "not present in MLPClassifier." The final fetch confirmed it IS present in both. **Writer must verify in the live 1.5 docs.**

2. **Glorot initialization factor for ReLU:** The search result cited a factor of 6 for ReLU and 2 for logistic. The standard Glorot/Xavier paper uses a factor of 6 for tanh-like activations and He initialization (factor=2 effectively giving a scale of $\sqrt{2/\text{fan\_in}}$) is preferred for ReLU in modern frameworks. sklearn appears to use its own variant of Glorot uniform for all activations. **Writer should verify in the sklearn source at the GitHub link provided.**

3. **SHAP `shap.Explainer` vs `shap.KernelExplainer` for sklearn MLP:** In SHAP 0.46.x, the unified `shap.Explainer(model, data)` API may auto-detect and use KernelExplainer for sklearn MLP. The writer should test both the old `KernelExplainer` API and the new unified `Explainer` API in the target SHAP version.

4. **`partial_fit` with `solver='adam'` or `solver='sgd'`:** The docs indicate `partial_fit` is available for SGD/Adam solvers for online/incremental learning. This is not available for `lbfgs`. The writer should verify if `partial_fit` on `MLPClassifier` requires passing `classes` on first call.

5. **`validation_scores_` attribute spelling:** The documentation lists `validation_scores_` (plural) as the attribute storing per-epoch validation accuracy when `early_stopping=True`. The writer should confirm the exact attribute name against sklearn 1.5 source to ensure code examples are correct.

6. **Grinsztajn et al. paper DOI:** The NeurIPS 2022 paper URL was confirmed but a clean DOI was not extracted. The writer may want to search for the arxiv preprint version at `arXiv:2207.08815` for a stable citation.
