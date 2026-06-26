# MLP Neural Networks: The Complete Masterclass

> **Why this algorithm matters:** In 1986, a three-page paper in *Nature* cracked open a problem
> that had paralyzed the field for nearly two decades: how to train a network with hidden layers.
> The backpropagation algorithm that Rumelhart, Hinton, and Williams published that year is the
> intellectual ancestor of every deep learning system in existence today — every large language
> model, every image classifier, every protein structure predictor. You are reading this because
> neural networks power the most economically significant technology of the 21st century. But before
> you can use deep learning wisely, you must understand the foundational architecture: the
> Multi-Layer Perceptron. This chapter teaches you the MLP from first principles — the 1958
> perceptron that started it all, the 1986 backpropagation breakthrough, the universal
> approximation theorem, every activation function, every optimizer, every hyperparameter in
> sklearn's `MLPClassifier`, and the honest truth about when you should use a neural network
> versus a gradient-boosted tree.

---

## §0 — Python Ecosystem & Package Guide

The MLP exists at a fascinating intersection: it is both the simplest neural network architecture
and the gateway to the entire deep learning ecosystem. Your package choice here carries more weight
than for almost any other algorithm in this course, because the gap between sklearn's MLP and
PyTorch/Keras is not just an API difference — it is the difference between a teaching tool and a
production-grade deep learning framework. Let's map the terrain carefully.

### Complete Package Table

| Package | Import | Version (mid-2026) | Internals | GPU | License |
|---|---|---|---|---|---|
| `scikit-learn` | `from sklearn.neural_network import MLPClassifier, MLPRegressor` | ~1.5.x (1.9 current) | Feedforward MLP, backprop, solvers: adam/sgd/lbfgs | None | BSD-3 |
| `PyTorch` | `import torch; import torch.nn as nn` | ~2.4.x | Full framework: autograd, custom layers, GPU, distributed | CUDA/MPS | BSD-3 |
| `Keras / TensorFlow` | `from tensorflow import keras` or `import keras` (standalone) | TF ~2.17.x / Keras ~3.x | High-level API over TF/JAX/PyTorch backends | CUDA/TPU | Apache-2 |
| `PyTorch Tabular` | `from pytorch_tabular import TabularModel` | ~1.1.x | sklearn-compatible API over PyTorch, tabular-optimized architectures | Yes | MIT |
| `FastAI` | `from fastai.tabular.all import *` | ~2.7.x | High-level API over PyTorch, tabular learner | Yes | Apache-2 |
| `skorch` | `from skorch import NeuralNetClassifier` | ~0.15.x | sklearn-compatible wrapper around PyTorch | Yes (via PyTorch) | BSD-3 |
| `SHAP` | `import shap` | ~0.46.x | KernelExplainer for model-agnostic SHAP on sklearn MLP | — | MIT |
| `Optuna` | `import optuna` | ~3.x | Hyperparameter optimization | — | MIT |

> ⚠️ **Pitfall:** sklearn's `MLPClassifier` documentation explicitly states: *"This implementation
> is not intended for large-scale applications. In particular, scikit-learn offers no GPU support."*
> It also lacks dropout, batch normalization, custom loss functions, and CNN/RNN layers. Know what
> you are buying before you buy it.

### Top Picks — Recommendation Table

| Package | Best for | Modern/Legacy | Notes |
|---|---|---|---|
| **`sklearn.MLPClassifier`** | Small/medium tabular datasets (<50k rows), quick baselines, sklearn Pipelines | Modern (limited scope) | **Start here**; zero-friction integration with preprocessing pipelines |
| **`PyTorch` (`torch.nn`)** | Custom architectures, research, GPU training, production systems | Modern (dominant) | **Graduate here** when sklearn MLP is not enough |
| **`Keras`** | Rapid prototyping, TF ecosystem, production serving (TF Serving) | Modern | Strong option if your team knows TF; Keras 3 is backend-agnostic |
| **`skorch`** | When you want sklearn-compatible API (GridSearchCV, Pipeline) but PyTorch power | Modern | Best of both worlds for tabular PyTorch users |
| **`PyTorch Tabular`** | Tabular-specific architectures (TabNet, SAINT, FT-Transformer) | Modern | Worth knowing when you want to go beyond vanilla MLP on tabular data |

### The Same Fit Across Top Packages

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Shared data prep — used throughout this chapter
digits = load_digits()
X, y = digits.data, digits.target          # (1797, 64), 10-class digit recognition
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Option 1: sklearn (the baseline workhorse) ──────────────────────────────
from sklearn.neural_network import MLPClassifier

clf_sklearn = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    max_iter=1000,
    early_stopping=True,
    random_state=42
)
clf_sklearn.fit(X_train_sc, y_train)
print(f"sklearn MLP accuracy: {clf_sklearn.score(X_test_sc, y_test):.4f}")

# ── Option 2: PyTorch (the graduate path) ───────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, n_classes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=64, hidden_sizes=[100, 50], n_classes=10)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

X_t = torch.tensor(X_train_sc, dtype=torch.float32)
y_t = torch.tensor(y_train, dtype=torch.long)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_t), y_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_test_sc, dtype=torch.float32)).argmax(1).numpy()
print(f"PyTorch MLP accuracy: {(preds == y_test).mean():.4f}")

# ── Option 3: Keras (rapid prototyping) ─────────────────────────────────────
# import keras                                         # pip install keras tensorflow
# keras_model = keras.Sequential([
#     keras.layers.Input(shape=(64,)),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(50,  activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(10,  activation='softmax')
# ])
# keras_model.compile(optimizer='adam',
#                     loss='sparse_categorical_crossentropy',
#                     metrics=['accuracy'])
# keras_model.fit(X_train_sc, y_train, epochs=100, validation_split=0.1, verbose=0)
# print(f"Keras MLP accuracy: {keras_model.evaluate(X_test_sc, y_test, verbose=0)[1]:.4f}")

# ── Option 4: skorch (sklearn API + PyTorch backend) ────────────────────────
# from skorch import NeuralNetClassifier              # pip install skorch
# net = NeuralNetClassifier(
#     MLP,
#     module__input_dim=64, module__hidden_sizes=[100,50], module__n_classes=10,
#     optimizer=optim.Adam, lr=1e-3, max_epochs=100,
#     iterator_train__shuffle=True,
# )
# net.fit(X_train_sc.astype(np.float32), y_train)
# print(f"skorch MLP accuracy: {net.score(X_test_sc.astype(np.float32), y_test):.4f}")
```

> 💡 **Intuition:** Think of sklearn's MLP as a fully-assembled bicycle — perfect for getting
> from A to B on a normal road, minimal setup required. PyTorch is a machine shop where you can
> build any vehicle you can imagine, but you need to know how gears work. Keras is a car with
> power steering — faster to drive than raw PyTorch but less control. skorch is a bolt-on motor
> kit for the bicycle — you get the speed without fully rebuilding. Choose the right vehicle for
> the terrain.

> 🔧 **In practice:** For any new MLP project on tabular data: (1) start with sklearn's
> `MLPClassifier` in a `Pipeline` with `StandardScaler` — this is your baseline, takes 5 minutes
> to set up; (2) if accuracy is competitive and the dataset fits in memory, tune it with Optuna;
> (3) if you need dropout, batch normalization, GPU, or custom losses, graduate to PyTorch or Keras.

---

## §1 — The Origin: The Papers That Started It All

### Before the Perceptron: McCulloch-Pitts Neurons (1943)

To understand where MLPs come from, you need to go back to 1943 — before there was such a thing
as a "machine learning algorithm." Warren McCulloch, a neurophysiologist, and Walter Pitts, a
mathematician, published "A Logical Calculus of the Ideas Immanent in Nervous Activity" in the
*Bulletin of Mathematical Biophysics*. Their contribution was a mathematical model of a neuron:
a binary threshold unit that fires (outputs 1) if the weighted sum of its binary inputs exceeds
a threshold, and is silent (outputs 0) otherwise:

$$\text{output} = \begin{cases} 1 & \text{if } \sum_j w_j x_j \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

McCulloch and Pitts showed that networks of such units could, in principle, compute any logical
function. But here was the fatal limitation: the weights $w_j$ and threshold $\theta$ had to be
set by hand. There was no learning — no algorithm to find the right weights from data. These were
human-engineered boolean computers, not learning machines.

### Rosenblatt's Perceptron (1958): The First Learning Algorithm

> 📜 **Citation:** Rosenblatt, F. (1958). "The perceptron: a probabilistic model for information
> storage and organization in the brain." *Psychological Review*, 65(6), 386–408.
> DOI: [10.1037/h0042519](https://doi.org/10.1037/h0042519)

Frank Rosenblatt was a psychologist at Cornell who wanted to build a machine that *learns* from
examples the way humans do. In 1958 he introduced the **perceptron** — a single-layer neural
network with a simple but revolutionary learning rule. The problem he set out to solve: given
labeled examples, automatically adjust weights so the neuron correctly classifies new inputs.

**The Perceptron Model:**

Given an input vector $\mathbf{x} = (x_1, \ldots, x_p)$ and weight vector
$\mathbf{w} = (w_1, \ldots, w_p)$ with bias $b$:

$$\hat{y} = \text{sign}\!\left(\mathbf{w}^T \mathbf{x} + b\right) = \begin{cases} +1 & \text{if } \mathbf{w}^T \mathbf{x} + b > 0 \\ -1 & \text{if } \mathbf{w}^T \mathbf{x} + b \leq 0 \end{cases}$$

**The Perceptron Learning Rule:**

For each training example $(\mathbf{x}_i, y_i)$ where $y_i \in \{-1, +1\}$:

1. Compute prediction: $\hat{y}_i = \text{sign}(\mathbf{w}^T \mathbf{x}_i + b)$
2. If correctly classified ($\hat{y}_i = y_i$): no update
3. If misclassified ($\hat{y}_i \neq y_i$): update weights:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y_i \cdot \mathbf{x}_i$$
$$b \leftarrow b + \eta \cdot y_i$$

where $\eta > 0$ is the learning rate. The update rule has an elegant geometric interpretation:
when the model misclassifies example $i$, it adjusts the weight vector toward $\mathbf{x}_i$ (if
$y_i = +1$) or away from $\mathbf{x}_i$ (if $y_i = -1$), rotating the decision hyperplane to
reduce future misclassifications.

**The Perceptron Convergence Theorem:** Rosenblatt proved that if the data is linearly separable
(a hyperplane exists that perfectly classifies all examples), the perceptron algorithm will converge
to a correct solution in a finite number of updates. This was a mathematically rigorous guarantee,
and it caused enormous excitement. The *New York Times* ran a headline in 1958: *"New Navy Device
Learns by Doing."*

> 💡 **Intuition:** The perceptron is doing something beautifully simple — it is finding a
> hyperplane in feature space that separates the two classes. Every update nudges the hyperplane a
> little closer to correctly separating all points. When the data is separable, this process must
> eventually succeed. When it is not separable, it never stops.

**The Fatal Limitation — Minsky and Papert (1969):**

In 1969, Marvin Minsky and Seymour Papert published *Perceptrons* (MIT Press), a rigorous
mathematical analysis of what perceptrons can and cannot compute. Their most famous result: the
perceptron cannot solve the XOR problem.

The XOR function:
- $f(0, 0) = 0$
- $f(0, 1) = 1$
- $f(1, 0) = 1$
- $f(1, 1) = 0$

Try to draw a single straight line separating the 1s from the 0s in $(x_1, x_2)$ space — it is
impossible. XOR is not linearly separable. Since the perceptron can only learn linear decision
boundaries, it is fundamentally incapable of learning XOR.

More broadly, Minsky and Papert showed that many practically useful functions require nonlinear
separators. The single-layer perceptron is, in a deep sense, limited to linear classification.
Their analysis was so influential (and, in hindsight, overstated in its pessimism about adding
hidden layers) that it triggered the first **AI Winter** for neural networks — funding dried up,
interest evaporated, and the field lay dormant for over a decade.

```python
from sklearn.linear_model import Perceptron
import numpy as np

# The original single-layer perceptron
p = Perceptron(max_iter=1000, random_state=42, eta0=1.0)

# XOR problem — the perceptron's Waterloo
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

p.fit(X_xor, y_xor)
print(f"XOR accuracy: {p.score(X_xor, y_xor):.2f}")  # 0.50 or 0.75 — never 1.0
```

### Rumelhart, Hinton, and Williams (1986): Backpropagation

> 📜 **Citation:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning
> representations by back-propagating errors." *Nature*, 323(6088), 533–536.
> DOI: [10.1038/323533a0](https://doi.org/10.1038/323533a0)

The question that haunted the field after Minsky and Papert: if you add hidden layers to a network
to gain nonlinear power, how do you train them? The output layer is easy — you know what the
correct output should be, so you can compute an error. But hidden layers produce *intermediate*
representations, not final outputs. There is no direct supervision signal for a hidden neuron.
You do not know what a hidden neuron "should" output — only whether the final network output is
correct.

The solution is the **backpropagation algorithm**, which computes the gradient of the loss with
respect to every weight in the network by applying the chain rule systematically, propagating
error signals backward from the output layer through each hidden layer to the input.

> 📜 **Historical note:** Backpropagation was not invented by Rumelhart et al. Paul Werbos derived
> it in his 1974 Harvard Ph.D. thesis (*Beyond Regression: New Tools for Prediction and Analysis
> in the Behavioral Sciences*). David Parker independently rediscovered it in 1985. But the 1986
> *Nature* paper was the first to demonstrate convincingly that hidden layers could learn useful
> representations — that they discovered internal features of the data (like the edges and curves
> that form handwritten digits) rather than being mysterious black boxes. This demonstration of
> *what* hidden layers learn was as important as the algorithm itself.

**What the paper actually showed:** Rumelhart et al. trained a 2-layer MLP on the XOR problem
(trivially solved by a hidden layer), on a symmetry detection task, and on the famous
encoder-decoder task where the network had to compress 8 inputs down to 3 hidden units and
reconstruct all 8 outputs — effectively learning a binary encoding. When they visualized the
learned representations, the hidden units had discovered something interpretable. The algorithm
was not just mathematically correct; it learned *meaningful* internal structure.

**The Network Architecture:**

An MLP with $L$ layers (input layer 0, $L-1$ hidden layers, output layer $L$) is defined by:

- Weight matrices: $\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ for $l = 1, \ldots, L$
- Bias vectors: $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$
- Activation function: $g(\cdot)$ applied elementwise

where $n_l$ is the number of neurons in layer $l$.

**The Forward Pass:**

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = g\!\left(\mathbf{z}^{(l)}\right)$$

with $\mathbf{a}^{(0)} = \mathbf{x}$ (the input). The final output $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$.

**The Loss Function:**

For binary classification with sigmoid output:
$$L = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

For multiclass with softmax output:
$$L = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \hat{y}_{ik}$$

For regression with identity output:
$$L = \frac{1}{2n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2$$

**The Backward Pass — Deriving the Gradient:**

The key quantity is the **error signal** $\boldsymbol{\delta}^{(l)}$, defined as the partial
derivative of the loss with respect to the pre-activation $\mathbf{z}^{(l)}$:

$$\boldsymbol{\delta}^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}}$$

**Step 1: Output layer error signal.**

For cross-entropy loss with softmax, the output error simplifies beautifully:
$$\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$$

This is the prediction minus the label — the "residual" in the neural network sense.

For mean squared error with identity output:
$$\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$$

For other loss/activation combinations, the chain rule gives:
$$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}} L \odot g'\!\left(\mathbf{z}^{(L)}\right)$$

where $\odot$ denotes elementwise multiplication.

**Step 2: Hidden layer error signals (the backpropagation recurrence).**

$$\boldsymbol{\delta}^{(l)} = \left(\left(\mathbf{W}^{(l+1)}\right)^T \boldsymbol{\delta}^{(l+1)}\right) \odot g'\!\left(\mathbf{z}^{(l)}\right)$$

Read this carefully. The error at layer $l$ is computed by:
1. Taking the error from the next layer $\boldsymbol{\delta}^{(l+1)}$
2. Multiplying by the transposed weight matrix $(\mathbf{W}^{(l+1)})^T$ — this distributes the
   upstream error back through the weights that caused it
3. Multiplying elementwise by $g'(\mathbf{z}^{(l)})$ — the activation derivative at this layer,
   which tells us how sensitive each neuron's output was to its input

This is the chain rule in matrix form, applied layer by layer from output to input.

**Step 3: Weight and bias gradients.**

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} \left(\mathbf{a}^{(l-1)}\right)^T$$
$$\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

**Step 4: Gradient descent update (vanilla SGD).**

$$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}$$
$$\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{b}^{(l)}}$$

> 🔬 **Deep dive:** Why is the weight gradient $\boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$?
> By the chain rule: $\frac{\partial L}{\partial w^{(l)}_{jk}} = \frac{\partial L}{\partial z^{(l)}_j}
> \cdot \frac{\partial z^{(l)}_j}{\partial w^{(l)}_{jk}} = \delta^{(l)}_j \cdot a^{(l-1)}_k$.
> Written for all $j, k$ simultaneously in matrix form: $\boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$.
> This outer product structure is why the weight gradient has the same shape as $\mathbf{W}^{(l)}$.

**Pseudocode for one training step:**

```
for each mini-batch B = {(x_1,y_1), ..., (x_m,y_m)}:
    # Forward pass
    a[0] = mean(x_i for i in B)               # batch input
    for l in 1..L:
        z[l] = W[l] @ a[l-1] + b[l]
        a[l] = g(z[l])                         # hidden: relu/tanh; output: softmax/sigmoid/identity

    # Backward pass
    delta[L] = a[L] - y_mean                   # output error
    for l in L-1..1:
        delta[l] = (W[l+1].T @ delta[l+1]) * g_prime(z[l])

    # Gradient update
    for l in 1..L:
        W[l] -= lr * delta[l] @ a[l-1].T / m
        b[l] -= lr * mean(delta[l])
```

What was uncertain in 1986: Rumelhart et al. were not sure hidden layers would reliably learn
useful representations in practice on complex real-world tasks. They demonstrated it worked on toy
tasks. Scaling to larger problems with deeper networks took another two decades of algorithmic
improvements: better initialization, better activations (ReLU), regularization (dropout), more
data, and GPU hardware.

### The Universal Approximation Theorem (1989, 1991)

> 📜 **Citation (Cybenko):** Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal
> function." *Mathematics of Control, Signals, and Systems*, 2(4), 303–314.
> DOI: [10.1007/BF02551274](https://doi.org/10.1007/BF02551274)

> 📜 **Citation (Hornik):** Hornik, K. (1991). "Approximation capabilities of multilayer
> feedforward networks." *Neural Networks*, 4(2), 251–257.
> DOI: [10.1016/0893-6080(91)90009-T](https://doi.org/10.1016/0893-6080(91)90009-T)

After backpropagation revived interest in neural networks, theorists asked: what can these things
actually represent? In 1989, George Cybenko proved the Universal Approximation Theorem (UAT) for
sigmoidal activations:

**Theorem (Cybenko 1989):** Let $g: \mathbb{R} \to \mathbb{R}$ be any nonconstant, bounded, and
continuous sigmoidal function (i.e., $g(t) \to 1$ as $t \to +\infty$ and $g(t) \to 0$ as
$t \to -\infty$). Then for any continuous function $f: [0,1]^n \to \mathbb{R}$ and any
$\epsilon > 0$, there exists a finite sum:

$$F(\mathbf{x}) = \sum_{j=1}^N \alpha_j \, g\!\left(\mathbf{w}_j^T \mathbf{x} + \theta_j\right)$$

such that $\|F(\mathbf{x}) - f(\mathbf{x})\| < \epsilon$ for all $\mathbf{x} \in [0,1]^n$.

In plain language: a single hidden layer with sufficiently many sigmoid neurons can approximate
any continuous function arbitrarily well.

**Hornik's 1991 extension** showed that the result holds for *any* bounded nonconstant activation
function — it is the architecture (multilayer with nonlinear activations), not the specific
activation, that confers universal approximation power. He also extended the result to
$L^p(\mu)$ criteria.

> ⚠️ **Pitfall:** The UAT is frequently misread as a practical guarantee. It is not. It states
> three things: (1) an approximating network *exists*, (2) one hidden layer is sufficient in
> principle, (3) the number of neurons needed may be exponential in the input dimension. It says
> nothing about whether gradient descent will find those weights, whether you have enough data,
> or whether the network will generalize. The UAT is a statement about *representational capacity*,
> not about *learnability* or *generalization*.

> 💡 **Intuition:** Think of the UAT as saying "a neural network is a universal function
> approximator" in the same way that "polynomials are universal function approximators" (by the
> Weierstrass theorem). That's true but doesn't mean fitting a polynomial of degree 1000 to your
> 100 data points is a good idea. In practice, deeper networks with fewer neurons per layer often
> generalize better than very wide single-layer networks.

---

## §2 — The Algorithm, Deeply Explained

### The Architecture: What an MLP Actually Is

An MLP is a directed acyclic computation graph organized into layers. Information flows in one
direction: from input to output. There is no feedback, no recurrence, no memory between inputs.
This makes the forward pass simple — a sequence of matrix multiplications and elementwise
nonlinearities — and makes backpropagation tractable.

**Layer structure:** An MLP with $L$ layers has:
- **Input layer:** No computation, just holds the feature vector $\mathbf{x} \in \mathbb{R}^p$
- **Hidden layers** $1, \ldots, L-1$: Each applies $\mathbf{a}^{(l)} = g(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})$
- **Output layer** $L$: Same formula but with a task-specific activation:
  - Binary classification: sigmoid $\sigma(z) = 1/(1+e^{-z})$ → outputs probability in $(0, 1)$
  - Multiclass: softmax $\text{softmax}(\mathbf{z})_k = e^{z_k} / \sum_j e^{z_j}$ → probability distribution over $K$ classes
  - Regression: identity $g(z) = z$ → unbounded real value

**Total parameters:**

$$\text{params} = \sum_{l=1}^{L} \left( n_{l-1} \cdot n_l + n_l \right) = \sum_{l=1}^{L} n_l (n_{l-1} + 1)$$

For a network with input dimension 64, two hidden layers of 100 and 50 neurons, and 10 output
classes:

$$\text{params} = (64 \cdot 100 + 100) + (100 \cdot 50 + 50) + (50 \cdot 10 + 10) = 6500 + 5050 + 510 = 12{,}060$$

That is 12,060 parameters learned from 1,797 training samples in the digits dataset. Already more
parameters than samples — deep learning's characteristic overparameterized regime, where
regularization and implicit biases of gradient descent carry the generalization load.

### Activation Functions: The Engine of Nonlinearity

Without activation functions, every layer computes a linear transformation, and any composition
of linear transformations is itself linear. The entire network would collapse to a single matrix
multiplication — no more expressive than logistic regression. Nonlinear activations are what give
MLPs their power to represent complex functions.

**Sigmoid (logistic):** $\sigma(z) = \frac{1}{1 + e^{-z}}$

- Range: $(0, 1)$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \leq 0.25$
- The maximum gradient is 0.25, achieved at $z = 0$. For $|z| > 3$, $\sigma'(z) < 0.05$.
- **Vanishing gradient problem:** In a network with $L$ layers all using sigmoid, the gradient at
  layer 1 is scaled by $\prod_{l=1}^L \sigma'(z^{(l)}) \leq 0.25^L$. For $L = 4$:
  $0.25^4 = 3.9 \times 10^{-3}$ — the first layer receives nearly zero gradient. Learning stops.
- **Not zero-centered:** Outputs always positive. This means gradients for weights in the next
  layer are all the same sign, leading to zigzag updates in gradient descent.

**Tanh:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$

- Range: $(-1, 1)$ — zero-centered; fixes the "same-sign gradient" problem of sigmoid
- Derivative: $\tanh'(z) = 1 - \tanh^2(z) \leq 1$
- Maximum gradient is 1 at $z = 0$; still saturates for large $|z|$; still vanishes
- Better than sigmoid for hidden layers; about 4× faster convergence in practice (LeCun et al. 1998)

**ReLU (Rectified Linear Unit):** $\text{ReLU}(z) = \max(0, z)$

- Range: $[0, \infty)$
- Derivative: $\text{ReLU}'(z) = \mathbb{1}[z > 0]$ — exactly 0 or 1
- **No vanishing gradient** for active neurons (where $z > 0$): the gradient passes through
  unchanged. This is the key property that enables training deep networks.
- **Computationally trivial:** A single comparison operation per neuron
- **Sparse activations:** Roughly half of neurons are inactive (output 0) at any given time —
  a form of implicit regularization that also happens to match sparse biological neuron coding
- **Dying ReLU problem:** If a neuron's pre-activation is always negative (due to large negative
  bias or aggressive weight updates), it always outputs 0 and receives zero gradient — it "dies"
  and never recovers. Its incoming weights never update.

> 💡 **Intuition:** ReLU is a valve. When open ($z > 0$), it lets the signal pass with no
> attenuation. When closed ($z < 0$), it blocks entirely. A sigmoid is a dimmer switch — it always
> attenuates the signal, more so at extremes. In a 10-layer network of dimmer switches, the signal
> at the input barely moves. In a network of valves, the signal either flows fully or does not,
> preserving gradient magnitude through many layers.

**GELU (Gaussian Error Linear Unit):** $\text{GELU}(z) = z \cdot \Phi(z)$

where $\Phi$ is the standard normal CDF. Approximation: $\text{GELU}(z) \approx 0.5z(1 + \tanh[\sqrt{2/\pi}(z + 0.044715z^3)])$.

- Smooth, differentiable everywhere (unlike ReLU's kink at 0)
- Stochastic interpretation: $\text{GELU}(z) = z \cdot \mathbb{P}(X \leq z)$ where $X \sim \mathcal{N}(0,1)$ — weights inputs by the probability of being "above noise"
- Standard in Transformers (BERT, GPT-2+), outperforms ReLU on NLP tasks
- Not available in sklearn MLP; available in PyTorch (`nn.GELU()`) and Keras

**Leaky ReLU:** $\text{LeakyReLU}(z) = \max(\alpha z, z)$ where $\alpha \approx 0.01$

- Fixes dying ReLU: negative inputs get a small nonzero gradient $\alpha$ instead of 0
- Not available in sklearn MLP; available in PyTorch (`nn.LeakyReLU(0.01)`)

**ELU (Exponential Linear Unit):**
$$\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

- Mean activations closer to zero than ReLU, reducing bias shift
- Not available in sklearn MLP; available in PyTorch (`nn.ELU()`)

### Computational Complexity

**Training complexity** for one epoch (one pass through $n$ samples):

$$O\!\left(n \cdot \left(p \cdot h + (k-1) \cdot h^2 + h \cdot o\right)\right)$$

where $p$ = input features, $h$ = neurons per hidden layer (assumed uniform), $k$ = number of
hidden layers, $o$ = output neurons. Over $T$ epochs total: multiply by $T$.

**Key scaling laws:**
- Linear in $n$: doubling the dataset doubles training time
- Quadratic in $h$: doubling neuron width quadruples the cost of inter-hidden connections
- Linear in $k$: adding layers increases cost linearly (for fixed $h$)
- For sklearn (CPU only): a network with 3 hidden layers of 256 neurons each on 50k samples
  takes minutes on modern hardware. For 500k samples or 1024-neuron layers, you need GPU.

**Inference complexity** per sample:

$$O\!\left(p \cdot h + (k-1) \cdot h^2 + h \cdot o\right)$$

Just one forward pass — no backward pass needed. Inference is fast once the model is trained.

### Implicit Assumptions

The MLP makes several implicit assumptions that practitioners often overlook:

1. **Feature scaling required:** Gradient descent assumes all features contribute roughly equally
   to the gradient. Features with 10,000× larger scale dominate weight updates for the first
   layer. The model will technically converge but may use many more iterations than necessary.
   Always scale to zero mean, unit variance with `StandardScaler`.

2. **i.i.d. assumption:** Training examples are assumed to be independent and identically
   distributed. If your data has temporal autocorrelation, spatial structure, or group membership
   (patients within hospitals), standard MLP training ignores this structure — use time-series
   architectures (LSTM, Transformer) or mixed-effects models instead.

3. **Stationarity:** The distribution of $(\mathbf{x}, y)$ is assumed constant across the
   training set. Concept drift (distribution shift over time) will silently degrade performance.

4. **Continuous features preferred:** MLPs process continuous inputs naturally. High-cardinality
   categorical features need embedding layers (which sklearn MLP does not support) or careful
   encoding (target encoding, one-hot for low cardinality).

5. **No built-in feature selection:** Unlike tree-based models that split on individual features
   and thus naturally ignore uninformative ones, an MLP must suppress uninformative features
   through $\alpha$ (L2 regularization). With many irrelevant features, either use strong $\alpha$
   or pre-select features before training.

> 🔬 **Deep dive:** Why does L2 regularization ($\alpha > 0$) help with irrelevant features?
> The loss becomes $L_{\text{total}} = L_{\text{data}} + \frac{\alpha}{2n}\|\mathbf{W}\|_2^2$.
> Weights for features uncorrelated with the target get zero gradient from the data term but a
> nonzero gradient $-\alpha w_{ij}/n$ from the regularization term, pushing them toward zero.
> With strong enough $\alpha$, the network effectively ignores uninformative features.

---

## §3 — The Full Evolution: From Perceptron to Modern Deep Networks

The history of MLPs is a history of solving one problem at a time. Each innovation in this
section emerged because practitioners hit a specific wall — training failed, generalized poorly,
or couldn't scale — and the community invented a targeted fix. Understanding the chain of problems
and solutions is the fastest way to internalize why each component of a modern neural network is
the way it is.

### 3.1 Single-Layer Perceptron → Multi-Layer Perceptron (1958 → 1986)

**Problem:** Linear decision boundaries cannot solve XOR or any nonlinearly separable task.

**Solution:** Hidden layers with nonlinear activations, trained by backpropagation.

**Key change:** Adding one or more hidden layers between input and output, with the
backpropagation algorithm (§1) to compute gradients for all weights.

**When to prefer:** Always, for any nonlinear problem. The single-layer perceptron is a
historical artifact — never use it for practical classification except as a linear baseline.

```python
# Single-layer perceptron (linear):
from sklearn.linear_model import Perceptron
p = Perceptron()

# Multi-layer perceptron (nonlinear):
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,))   # one hidden layer of 100 neurons
```

### 3.2 Sigmoid → Tanh Activations (1980s–1990s)

> 📜 **Citation:** LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (1998). "Efficient
> BackProp." In *Neural Networks: Tricks of the Trade*, Lecture Notes in Computer Science.
> Springer, Berlin. URL: https://link.springer.com/chapter/10.1007/3-540-49430-8_2

**Problem:** Sigmoid's output range $(0, 1)$ is not zero-centered. When all activations are
positive, the gradient of the downstream weights is always the same sign — all positive or all
negative — depending on $\delta^{(l+1)}$. This forces the weight updates to move in correlated
directions, causing inefficient zigzag search in parameter space. LeCun et al. called this the
"saturation and slow learning" problem and recommended tanh as the standard hidden activation.

**Key change:** Replace $\sigma(z)$ with $\tanh(z)$. Tanh maps to $(-1, +1)$, is zero-centered,
and has maximum gradient 1 (vs 0.25 for sigmoid), making it roughly 4× faster in practice.

**When to prefer:** Legacy code or when you need a hidden layer output in $(-1, 1)$. For new
models, ReLU dominates.

```python
mlp_tanh = MLPClassifier(activation='tanh', hidden_layer_sizes=(100,), max_iter=500)
```

### 3.3 Tanh → ReLU Activations (2011–2012)

> 📜 **Citation:** Glorot, X., Bordes, A., & Bengio, Y. (2011). "Deep sparse rectifier neural
> networks." In *Proceedings of the 14th International Conference on Artificial Intelligence and
> Statistics (AISTATS)*, PMLR 15, 315–323.
> URL: http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf

> 📜 **Citation (AlexNet):** Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet
> classification with deep convolutional neural networks." *NeurIPS*, 25, 1097–1105.

**Problem:** Both sigmoid and tanh saturate — their derivatives approach 0 for large $|z|$. In
deep networks (5+ layers), the product of derivatives along the chain rule shrinks exponentially:
for sigmoid, $\prod_{l=1}^L \sigma'(z^{(l)}) \leq 0.25^L$. At 10 layers, this is $10^{-6}$.
The first layers receive essentially zero gradient and do not learn. This was the **vanishing
gradient problem**.

**Key change:** $\text{ReLU}(z) = \max(0, z)$. The derivative is exactly 1 for $z > 0$ and 0
for $z \leq 0$. Active neurons propagate gradients without any attenuation. This single change
enabled training networks 10× deeper than before.

The 2011 Glorot et al. paper provided the first rigorous study of ReLU in deep networks,
showing it produces sparser activations and better-conditioned gradients than tanh. The 2012
AlexNet paper (winning ImageNet by a dramatic ~10% accuracy margin) made ReLU the industry
standard.

**When to prefer:** Almost always. ReLU is the default for hidden layers in all modern networks.

```python
mlp_relu = MLPClassifier(activation='relu')   # default in sklearn since v0.18
```

### 3.4 Glorot/Xavier Weight Initialization (2010)

> 📜 **Citation:** Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training
> deep feedforward neural networks." In *Proceedings of the 13th International Conference on
> Artificial Intelligence and Statistics (AISTATS)*, PMLR 9, 249–256.
> URL: https://proceedings.mlr.press/v9/glorot10a.html

**Problem:** If weights are initialized too large, activations explode (each layer amplifies the
signal), leading to saturated sigmoid/tanh neurons and vanishing gradients. If initialized too
small, activations shrink to zero (signal dying). Either way, early training is dysfunctional.

**Core insight:** For the variance of activations to remain constant across layers during the
forward pass, and for the variance of gradients to remain constant during the backward pass, the
weights in layer $l$ should have variance:

$$\text{Var}(w^{(l)}) = \frac{2}{n_{l-1} + n_l}$$

where $n_{l-1}$ is the number of inputs (fan-in) and $n_l$ is the number of outputs (fan-out).

**Glorot uniform initialization:** Draw weights from:
$$w \sim \text{Uniform}\!\left(-\sqrt{\frac{6}{n_{l-1} + n_l}},\ +\sqrt{\frac{6}{n_{l-1} + n_l}}\right)$$

**He/Kaiming initialization (for ReLU):**

> 📜 **Citation:** He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving deep into rectifiers:
> Surpassing human-level performance on ImageNet classification." *ICCV 2015*.
> arXiv:1502.01852. URL: https://arxiv.org/abs/1502.01852

Since ReLU zeros out half the activations, the effective variance is halved. He et al. showed
that the correct initialization for ReLU is:

$$\text{Var}(w^{(l)}) = \frac{2}{n_{l-1}} \quad \Rightarrow \quad w \sim \mathcal{N}\!\left(0, \sqrt{\frac{2}{n_{l-1}}}\right)$$

**sklearn's implementation:** sklearn uses a variant of Glorot uniform for all activations
(factor 6 for relu/tanh, factor 2 for logistic). This is good but not strictly He initialization
for ReLU. In PyTorch, `torch.nn.init.kaiming_uniform_` implements He initialization and is
the default for `nn.Linear` with ReLU.

```python
# PyTorch explicit He initialization:
import torch.nn as nn

layer = nn.Linear(128, 64)
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(layer.bias)
```

### 3.5 The Adam Optimizer (2014)

> 📜 **Citation:** Kingma, D. P. & Ba, J. (2014). "Adam: A method for stochastic optimization."
> *International Conference on Learning Representations (ICLR 2015)*.
> arXiv:1412.6980. DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)

**Problem:** Plain SGD with momentum requires careful, manual tuning of the learning rate. A
single global learning rate is applied to all parameters equally, but different parameters have
very different gradient magnitudes — some weights might receive gradients of 0.001 while others
receive 10.0. A learning rate that works for one is disastrous for the other.

**Solution:** Adam maintains **per-parameter adaptive learning rates** based on estimates of
first and second moments of the gradient history.

**Algorithm at step $t$, for each parameter $\theta$ with gradient $g_t$:**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias correction (both moment estimates are initialized to 0 and are biased toward 0 at early
steps — this correction accounts for that):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Defaults:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 0.001$

**Why it works:** The effective learning rate for parameter $\theta$ is
$\eta / (\sqrt{\hat{v}_t} + \epsilon)$. Parameters with consistently large gradients (large
$\hat{v}_t$) get smaller effective learning rates — they are moving fast enough and don't need
large steps. Parameters with small or sparse gradients get larger effective learning rates —
they need bigger nudges to move meaningfully. Adam is self-tuning in this sense.

> 💡 **Intuition:** Adam is like an experienced mountain climber with a topographic memory. It
> remembers not just the current slope (gradient) but also the average slope ($m_t$) and the
> steepness variance ($v_t$) of every direction it has walked. In rugged terrain (high $v_t$),
> it takes cautious steps. In smooth valleys (low $v_t$), it takes bold strides. SGD is a
> climber who always takes the same size step regardless of terrain.

**When to prefer:** Adam over SGD: virtually always for tabular MLPs, CNNs, and Transformers.
SGD with momentum (often with learning rate schedules) is preferred in some image classification
pipelines where it achieves marginally better final generalization at the cost of more tuning.

```python
# Adam is the default solver in sklearn MLPClassifier
clf = MLPClassifier(solver='adam', learning_rate_init=0.001, beta_1=0.9, beta_2=0.999)
```

### 3.6 L-BFGS Solver for Small Datasets

**Problem:** Adam and SGD are first-order optimizers — they use only gradient information. For
small datasets (<2000 samples), the optimization landscape is smooth enough to exploit
second-order curvature information, which dramatically accelerates convergence.

**Solution:** L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), a quasi-Newton method
that approximates the inverse Hessian using the last $m$ gradient difference vectors.

**Why it wins on small data:** L-BFGS takes curvature-aware steps that are much more
precisely directed than gradient descent steps. On small smooth objectives, it converges in
far fewer iterations. The limitation: it requires the full dataset (no mini-batches), which
becomes intractable for large $n$.

```python
# Use lbfgs for small datasets
clf_small = MLPClassifier(solver='lbfgs', max_iter=2000, hidden_layer_sizes=(50,))
# On n < 2000 samples, lbfgs often achieves lower final loss than adam
```

### 3.7 Dropout Regularization (2012–2014)

> 📜 **Citation:** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.
> (2014). "Dropout: A simple way to prevent neural networks from overfitting." *Journal of
> Machine Learning Research*, 15(1), 1929–1958. URL: https://jmlr.org/papers/v15/srivastava14a.html

**Problem:** Large MLPs dramatically overfit training data, especially with limited data.
L2 regularization ($\alpha$) helps but is insufficient for very overparameterized networks.

**Solution:** During training, randomly zero out each neuron's activation with probability $p$
(the dropout rate, typically 0.2–0.5). This prevents neurons from co-adapting — no neuron can
rely on any specific other neuron being present. The network is forced to learn redundant
representations, each of which is independently predictive.

**At test time:** All neurons are active, but their outputs are scaled by $(1-p)$ to compensate
for the fact that during training only $(1-p)$ fraction were active on average (or equivalently,
in the inverted dropout convention, test-time scaling is done at training time).

**The ensemble interpretation:** Training with dropout is equivalent to training an exponential
ensemble of $2^N$ different "thinned" networks (where $N$ is the number of droppable neurons)
and averaging their predictions at test time. This is the core reason it reduces variance.

> ⚠️ **Pitfall:** sklearn's `MLPClassifier` does not implement dropout. If you need dropout for
> regularization, you must use PyTorch or Keras. This is the most common reason to graduate from
> sklearn MLP.

```python
# Dropout in PyTorch
model = nn.Sequential(
    nn.Linear(64, 100), nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(100, 50),  nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(50, 10)
)
```

### 3.8 Batch Normalization (2015)

> 📜 **Citation:** Ioffe, S. & Szegedy, C. (2015). "Batch normalization: Accelerating deep
> network training by reducing internal covariate shift." *ICML 2015*. arXiv:1502.03167.
> URL: https://arxiv.org/abs/1502.03167

**Problem:** As weights update during training, the distribution of each layer's inputs shifts
— a phenomenon called **internal covariate shift**. This means each layer must continuously
adapt to a moving input distribution, slowing convergence and requiring careful learning rate
tuning.

**Solution:** Before each activation, normalize the layer's pre-activations to zero mean and
unit variance across the batch, then apply learnable scale $\gamma$ and shift $\beta$:

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{z}_i + \beta$$

where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance.

**Benefits:** Enables 10× higher learning rates, acts as regularization (reducing the need for
dropout in some architectures), and makes training dramatically more stable. Batch norm is
standard in modern deep learning.

> ⚠️ **Pitfall:** sklearn's `MLPClassifier` does not implement batch normalization. Graduate to
> PyTorch (`nn.BatchNorm1d`) or Keras (`layers.BatchNormalization`) when you need it.

### 3.9 The Graduated Path: When sklearn MLP Hits Its Limits

sklearn's MLP is explicitly documented as "not intended for large-scale applications." The
decision tree for graduation:

| Need | sklearn MLP | PyTorch/Keras |
|---|---|---|
| Quick tabular baseline | ✓ | — |
| GPU acceleration | ✗ | ✓ |
| Dropout regularization | ✗ | ✓ |
| Batch normalization | ✗ | ✓ |
| Custom loss function | ✗ | ✓ |
| Learning rate scheduling | ✗ (limited) | ✓ |
| CNN/RNN/Transformer layers | ✗ | ✓ |
| >50k samples efficiently | ✗ | ✓ |
| Production serving | ✗ | ✓ |
| sklearn Pipeline integration | ✓ | Via skorch |

### 3.10 Tabular-Specific Architectures (2019–present)

**TabNet (2019):**

> 📜 **Citation:** Arik, S. Ö. & Pfister, T. (2019). "TabNet: Attentive interpretable tabular
> learning." *AAAI 2021*. arXiv:1908.07442. URL: https://arxiv.org/abs/1908.07442

Uses sequential attention to select relevant features at each decision step — combining the
feature selection of trees with the representational power of neural networks. Available in
`pytorch-tabular` and the standalone `pytorch-tabnet` package.

**FT-Transformer (Feature Tokenization Transformer, 2021):**

> 📜 **Citation:** Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). "Revisiting
> deep learning models for tabular data." *NeurIPS 2021*. arXiv:2106.11959.
> URL: https://arxiv.org/abs/2106.11959

Tokenizes each feature into an embedding and applies self-attention (Transformer blocks). On
many tabular benchmarks, matches or slightly exceeds gradient boosted trees — the first
architecture to seriously challenge trees at scale.

**The honest benchmark:**

> 📊 **Benchmark:** Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). "Why do tree-based
> models still outperform deep learning on typical tabular data?" *NeurIPS 2022*.
> URL: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

Across 45 tabular datasets, gradient-boosted trees (XGBoost, Random Forests) outperform deep
learning models in the majority of cases at ~10k sample scale. The gap narrows at larger scales
and when features are dense embeddings. This is the most important benchmark paper for
positioning MLPs in the ML practitioner's toolkit.

---

## §4 — Hyperparameters: The Complete Guide

Every hyperparameter in sklearn's `MLPClassifier` and `MLPRegressor` is listed below with its
mechanism, default value, failure modes, and tuning strategy.

### Architecture Parameters

#### `hidden_layer_sizes` — tuple, default `(100,)`

**What it controls:** The network topology. Length of the tuple = number of hidden layers.
Each element = number of neurons in that layer. The input and output layers are determined
automatically from the data.

**Mechanism:** More neurons per layer = more parameters = higher capacity to fit complex
functions. More layers = deeper compositional representations = ability to learn hierarchical
features. But both come at the cost of: longer training, higher risk of overfitting, and
increased optimization difficulty.

**Effect of too few:** Underfitting — the network cannot represent the underlying function.
Manifests as high training error.

**Effect of too many:** Overfitting — the network memorizes training noise. High training
accuracy, poor test accuracy. Also slower training with no accuracy benefit.

**Tuning guidance:**
- Start with one hidden layer: try $(32,)$, $(64,)$, $(100,)$, $(200,)$
- If one layer underfits, add a second: try $(100, 50)$, $(200, 100)$, $(256, 128)$
- For tabular data: rarely need more than 2–3 hidden layers
- A useful heuristic for hidden size: $h \approx 2/3 \cdot (p + o)$ or $h \in [p, 2p]$
- The architecture and `alpha` are the two most impactful hyperparameters

```python
# Optuna architecture search
n_layers = trial.suggest_int('n_layers', 1, 3)
hidden_layer_sizes = tuple(
    trial.suggest_int(f'h_{i}', 32, 512, log=True) for i in range(n_layers)
)
```

#### `activation` — str, default `'relu'`

**What it controls:** The nonlinear activation function applied to all hidden layers.
The output layer activation is set automatically (softmax for multiclass, sigmoid for binary,
identity for regression) and is NOT controlled by this parameter.

| Value | Function | Range | Key property | When to use |
|---|---|---|---|---|
| `'relu'` | $\max(0, x)$ | $[0, \infty)$ | No vanishing gradient for active neurons | Default; almost always correct |
| `'tanh'` | $\tanh(x)$ | $(-1, 1)$ | Zero-centered; vanishes for large \|x\| | If dying ReLU is suspected; legacy |
| `'logistic'` | $1/(1+e^{-x})$ | $(0,1)$ | Maximum gradient 0.25; vanishes | Rarely for hidden layers |
| `'identity'` | $x$ | $(-\infty,\infty)$ | No nonlinearity | Debugging only; equivalent to linear model |

> ⚠️ **Pitfall:** `activation='logistic'` (sigmoid) in hidden layers will cause vanishing
> gradients in any network deeper than ~3 layers. The maximum gradient of sigmoid is 0.25, so
> through 4 layers: $0.25^4 \approx 0.004$. Stick with `'relu'` unless you have a specific reason.

### Optimization Parameters

#### `solver` — str, default `'adam'`

**What it controls:** The optimization algorithm used to find the weights.

| Value | Method | Best for | Mini-batch? | Memory |
|---|---|---|---|---|
| `'adam'` | Adaptive moment estimation | Most datasets; robust default | Yes | $O(\text{params})$ |
| `'lbfgs'` | Limited-memory quasi-Newton | Small datasets (<2000 samples) | No (full batch) | $O(m \cdot \text{params})$ |
| `'sgd'` | SGD + optional momentum/Nesterov | Manual schedule control | Yes | $O(\text{params})$ |

**`'lbfgs'` caveat:** Requires the full dataset in memory each step and does not scale.
The sklearn docs state: *"For small datasets, however, 'lbfgs' can converge faster and perform
better."* Use it when $n < 2000$ and you need the best possible final loss.

**`'sgd'` use case:** When you want explicit control over the learning rate schedule (e.g., for
matching a specific training regime from a paper). Requires tuning `learning_rate`, `momentum`,
and `learning_rate_init`. Adam is almost always easier.

#### `alpha` — float, default `0.0001`

**What it controls:** The L2 regularization strength. The total loss becomes:

$$L_{\text{total}} = L_{\text{data}} + \frac{\alpha}{2n} \|\mathbf{W}\|_2^2$$

where the norm is over all weights in all layers (biases are NOT regularized in sklearn).

**Mechanism:** Larger $\alpha$ pulls all weights toward zero, reducing model complexity.

**Effect of too low ($\alpha \to 0$):** Overfitting — the network can grow weights arbitrarily
large to memorize training data.

**Effect of too high:** Underfitting — weights are forced so close to zero that the network
cannot represent the data. Manifests as both high training and test error.

**Tuning guidance:** This is the single most important regularization hyperparameter in
sklearn MLP. Search on a log scale: $[10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 0.1]$.
Use cross-validation. The right value depends heavily on dataset size: larger datasets can
tolerate smaller $\alpha$; small datasets need larger $\alpha$.

```python
# Log-scale grid search for alpha
from sklearn.model_selection import GridSearchCV
param_grid = {'mlp__alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]}
gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
gs.fit(X_train, y_train)
```

#### `learning_rate_init` — float, default `0.001`

**What it controls:** The initial learning rate for `adam` and `sgd` solvers. Ignored by
`lbfgs` (which uses line search).

**Effect of too high:** Divergence or oscillation — loss bounces or increases. Manifests as
unstable loss curve.

**Effect of too low:** Very slow convergence — loss decreases but painfully slowly. Risk of
getting stuck in a poor local minimum.

**Tuning guidance:** Adam is relatively robust to learning rate choice. Log-scale search:
$[10^{-4}, 5 \times 10^{-4}, 10^{-3}, 5 \times 10^{-3}, 10^{-2}]$. The default 0.001 works
well for most problems with Adam.

#### `batch_size` — int or `'auto'`, default `'auto'`

**What it controls:** The number of training samples per gradient update. `'auto'` resolves
to `min(200, n_samples)`. Ignored for `lbfgs`.

**Effect of small batches (e.g., 32):**
- More gradient update steps per epoch — faster early learning
- Noisier gradient estimates — acts as implicit regularizer
- Can escape sharp local minima (noise helps exploration)
- Slower wall-clock time per epoch (less parallelism)

**Effect of large batches (e.g., 512+):**
- Smoother, more accurate gradient estimates
- Faster per epoch (more parallel)
- May converge to sharper, less-generalizing minima (the "sharp minima" generalization gap)

**Tuning guidance:** The `'auto'` default (`min(200, n_samples)`) is a good compromise.
Try smaller batches (32, 64) if overfitting; try larger (256, 512) if training is too slow.

#### `max_iter` — int, default `200`

**What it controls:** Maximum number of training epochs for stochastic solvers (adam/sgd) or
maximum number of gradient descent steps for lbfgs.

> ⚠️ **Pitfall:** The default `max_iter=200` is almost always too low for non-trivial problems.
> You will get a `ConvergenceWarning` and the returned model will be undertrained. The fix is
> to set `max_iter=2000` (or higher) and enable `early_stopping=True` — let the convergence
> criterion stop training, not the epoch limit.

```python
# The correct pattern:
clf = MLPClassifier(
    max_iter=2000,          # generous ceiling
    early_stopping=True,    # stop early if val score plateaus
    n_iter_no_change=20,    # patience
    random_state=42
)
```

#### `early_stopping` — bool, default `False`

**What it controls:** Whether to reserve a validation split and stop training when validation
score stops improving.

**Mechanism:** If `True`, sklearn sets aside `validation_fraction` of training data as a
held-out validation set. After each epoch, it evaluates the model on this set. If the
validation score fails to improve by at least `tol` for `n_iter_no_change` consecutive epochs,
training stops.

> 🏆 **Best practice:** Always enable `early_stopping=True` with a generous `max_iter` (2000+)
> for adam and sgd. Without early stopping, you are committing to a fixed number of epochs with
> no mechanism to prevent overfitting. The combination `early_stopping=True` +
> `n_iter_no_change=20` + `max_iter=2000` is the recommended default for any real training run.

#### `validation_fraction` — float, default `0.1`

10% of training data reserved for early stopping validation. Only active when
`early_stopping=True`. If your dataset is small (<500 samples), consider reducing to 0.05 to
preserve more training data.

#### `n_iter_no_change` — int, default `10`

**What it controls:** Patience — how many consecutive epochs of no improvement before stopping.

**Tuning guidance:** The default 10 can be too aggressive for noisy training curves. If you see
training stopping prematurely (loss still decreasing but with occasional flat spots), increase to
20–50.

#### `tol` — float, default `1e-4`

Minimum improvement required to count as "improving." Applies to either training loss (if
`early_stopping=False`) or validation score (if `early_stopping=True`).

> ⚠️ **Pitfall:** If training stops early with a `ConvergenceWarning`, do NOT lower `tol` to
> suppress the warning. That changes the convergence criterion semantically. The correct fix is
> to increase `max_iter`.

#### `shuffle` — bool, default `True`

Whether to shuffle training samples before each epoch. Almost always leave as `True`. Shuffling
prevents the gradient from being biased by the order of training examples, which is especially
important if your data has temporal or class-ordered structure.

#### `warm_start` — bool, default `False`

If `True`, subsequent calls to `fit()` reuse the existing weights rather than reinitializing.
Useful for incremental training or for custom epoch-by-epoch monitoring:

```python
# Custom training loop with warm_start
clf = MLPClassifier(max_iter=1, warm_start=True, random_state=42)
train_losses = []
for epoch in range(500):
    clf.fit(X_train_sc, y_train)
    train_losses.append(clf.loss_)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={clf.loss_:.4f}, val_acc={clf.score(X_val_sc, y_val):.4f}")
```

> ⚠️ **Pitfall:** With `warm_start=True` and `solver='adam'`, the internal epoch counter is NOT
> reset between `fit()` calls (sklearn GitHub issue #8713). This can cause unexpected behavior
> when relying on `max_iter`. The manual loop pattern above bypasses this.

#### `random_state` — int or None, default `None`

Controls weight initialization, training-validation split (if `early_stopping=True`), and
mini-batch sampling order. Always set for reproducibility.

#### `verbose` — bool, default `False`

Prints loss at every iteration. Useful for debugging convergence; turn off in production.

### SGD-Specific Parameters

#### `momentum` — float, default `0.9`

Momentum coefficient for SGD. The update becomes:

$$\mathbf{v}_t = \mu \mathbf{v}_{t-1} - \eta \nabla L(\mathbf{W}_{t-1}), \quad \mathbf{W}_t = \mathbf{W}_{t-1} + \mathbf{v}_t$$

Momentum accumulates velocity in directions of consistent gradient, damping oscillations
and accelerating convergence in elongated loss landscape valleys. The default 0.9 is standard
and rarely needs changing.

#### `nesterovs_momentum` — bool, default `True`

Nesterov's Accelerated Gradient (NAG): instead of computing the gradient at the current
position, compute it at the anticipated future position $\mathbf{W} + \mu \mathbf{v}$:

$$\mathbf{v}_t = \mu \mathbf{v}_{t-1} - \eta \nabla L(\mathbf{W}_{t-1} + \mu \mathbf{v}_{t-1})$$

NAG "looks ahead" before computing the gradient — it corrects its trajectory before it overshoots.
In convex settings, NAG converges at $O(1/t^2)$ vs $O(1/t)$ for plain momentum. Essentially
always better than plain momentum; keep the default `True`.

#### `learning_rate` — str, default `'constant'` (SGD only)

| Value | Schedule | Formula | When to use |
|---|---|---|---|
| `'constant'` | Fixed | $\eta_t = \eta_0$ | Default; good with early stopping |
| `'invscaling'` | Decreasing | $\eta_t = \eta_0 / t^{p}$ | Theoretical convergence guarantees |
| `'adaptive'` | Reduce on plateau | Halves $\eta$ when loss stops improving | Alternative to early stopping |

#### `power_t` — float, default `0.5`

Exponent for `'invscaling'` schedule: $\eta_t = \eta_0 / t^{0.5}$. The $t^{-0.5}$ decay
satisfies the Robbins-Monro conditions for SGD convergence in stochastic optimization:
$\sum_t \eta_t = \infty$ (enough total learning) and $\sum_t \eta_t^2 < \infty$ (small enough
eventually). Rarely changed.

### Adam-Specific Parameters

#### `beta_1` — float, default `0.9`

Exponential decay for first moment (moving average of gradient). Controls how quickly old
gradients are downweighted. $\beta_1 = 0.9$ means the gradient from 10 steps ago has weight
$0.9^{10} \approx 0.35$. Almost never changed from 0.9.

#### `beta_2` — float, default `0.999`

Exponential decay for second moment (moving average of squared gradient). Higher = more stable
estimate of gradient variance; useful for sparse features where gradient magnitudes vary widely.
Almost never changed from 0.999.

#### `epsilon` — float, default `1e-8`

Numerical stability constant in the denominator $\sqrt{\hat{v}_t} + \epsilon$. Prevents
division by zero when gradients are near zero. Sometimes increased to $10^{-7}$ or $10^{-6}$
for numerical stability on float32 hardware. Essentially never needs tuning.

### lbfgs-Specific Parameter

#### `max_fun` — int, default `15000`

Maximum number of loss function evaluations for the lbfgs solver. lbfgs uses line search, which
requires multiple function evaluations per gradient step, so `max_fun` should be significantly
larger than `max_iter`. Added in sklearn 0.22. Available in both `MLPClassifier` and
`MLPRegressor`. Note: verify in the sklearn 1.5 docs whether this is present in both estimators
(the research dossier confirms it is).

### Tuning Playbook

| Parameter | Typical Range | Primary Effect | Recommended Strategy |
|---|---|---|---|
| `hidden_layer_sizes` | `(32,)` to `(512, 256, 128)` | Model capacity | Start `(100,)`; add layers if underfitting |
| `activation` | `'relu'` (default), `'tanh'` | Nonlinearity quality | Almost always `'relu'` |
| `solver` | `'adam'` or `'lbfgs'` | Optimization algorithm | `'adam'` default; `'lbfgs'` for $n < 2000$ |
| `alpha` | $[10^{-6}, 10^{-1}]$ log | L2 regularization | Log-scale search; most impactful regularizer |
| `learning_rate_init` | $[10^{-4}, 10^{-2}]$ log | Step size | Default 0.001 usually fine for Adam |
| `batch_size` | `'auto'`, 32, 64, 128 | Stochasticity | `'auto'` first; smaller if overfitting |
| `max_iter` | 1000–5000 | Training ceiling | Set high; rely on `early_stopping=True` |
| `early_stopping` | `True` | Overfitting prevention | **Always `True`** for adam/sgd |
| `n_iter_no_change` | 10–50 | Patience | Increase for noisy curves |
| `validation_fraction` | 0.05–0.2 | Val set size | 0.1 default; reduce for small datasets |
| `momentum` | 0.8–0.99 | SGD acceleration | Only for sgd; default 0.9 fine |
| `beta_1`, `beta_2` | defaults | Adam moments | Never change from 0.9, 0.999 |

---

## §5 — Strengths

### 5.1 Nonlinear Function Approximation of Arbitrary Complexity

**Mechanism:** The composition of multiple nonlinear layers allows MLPs to learn complex,
highly nonlinear decision boundaries. By the universal approximation theorem (§1), a sufficiently
wide MLP can approximate any continuous function. In practice, depth (multiple layers) allows
networks to learn compositional features — low-level patterns in early layers that combine into
high-level concepts in later layers — far more efficiently than width alone.

**Why this matters:** Many real-world problems have nonlinear structure that linear models
fundamentally cannot capture. Logistic regression will asymptote at ~78% on MNIST; an MLP with
two hidden layers reaches 97%+ on the same data. The nonlinearity is not a nice-to-have — it
is the core value proposition.

### 5.2 Feature Learning / Representation Learning

**Mechanism:** Hidden layers do not just classify — they transform the input into successively
better representations for the task. Each hidden layer projects the input into a new feature
space where the classification problem becomes easier. By the final hidden layer, even
nonlinearly separable input data may be linearly separable in the learned representation space.

**Why this matters:** Feature engineering is often the most time-consuming part of an ML
project. An MLP amortizes this cost by learning features directly from data. For pixel inputs
(images), frequency data (audio), or dense embeddings (text), hand-engineering features is
virtually impossible — representation learning is the only practical approach.

### 5.3 Works Directly on Dense Feature Vectors

**Mechanism:** MLPs natively consume dense floating-point vectors. Unlike trees, they do not
need to enumerate split candidates or handle categorical features specially (with appropriate
encoding). Dense inputs from image pixels, audio spectrograms, or pre-computed embeddings are
handled naturally.

**Why this matters:** For unstructured data processed into dense vectors, MLPs are the natural
choice. On truly tabular data with mixed categorical/numerical features, this advantage is
weaker (trees handle mixed types better natively).

### 5.4 Smooth, Differentiable Decision Boundary

**Mechanism:** The MLP's decision boundary is a smooth (except at ReLU kinks) function of the
input. This smoothness makes the model appropriate when the true decision boundary is expected
to be continuous, and makes gradient-based explanation methods (integrated gradients, vanilla
gradients) meaningful.

**Practical implication:** Calibrated probability estimates. An MLP's output probabilities can
be well-calibrated with temperature scaling, making them useful for downstream decision-making
(e.g., in systems that need to set decision thresholds based on class probabilities).

### 5.5 Principled Probabilistic Output

**Mechanism:** The softmax output layer + cross-entropy loss is derived from the maximum
likelihood principle. The output $\hat{y}_k = P(Y = k \mid \mathbf{x})$ has a well-defined
probabilistic interpretation. Unlike, say, a hard-margin SVM, the MLP directly models class
probabilities.

### 5.6 Highly Parallelizable

**Mechanism:** The matrix multiplications in forward and backward passes are perfectly
parallelizable across GPUs. For large networks, GPUs provide 100–1000× speedup over CPU
training. sklearn's CPU-only MLP is the slow lane — PyTorch on a modern GPU handles 10M
sample training in minutes.

### 5.7 Scales to Massive Data

**Mechanism:** With mini-batch SGD (or Adam), the MLP processes one mini-batch at a time. The
memory requirement is $O(\text{batch\_size} \times p)$, not $O(n \times p)$. You can train on
datasets far larger than GPU memory by streaming mini-batches.

**Contrast with sklearn MLP:** sklearn stores the full dataset in memory (not a streaming
implementation). For large-scale training, use PyTorch with DataLoader.

---

## §6 — Weaknesses & Failure Modes

### 6.1 Requires Feature Scaling

**Mechanism:** Gradient descent on a loss surface with wildly different feature scales is
inefficient. Features with large variance contribute disproportionately to the weight update
magnitude in the first layer, while features with small variance receive negligible gradient
signal. The optimization landscape becomes elongated ellipsoids rather than spheres.

**Detection:** `ConvergenceWarning` even with high `max_iter`. Slow, noisy loss curves.

**Mitigation:** Always wrap in a `Pipeline` with `StandardScaler`. This is non-negotiable for
sklearn MLP.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=1000, early_stopping=True, random_state=42))
])
```

> ⚠️ **Pitfall:** This is the #1 mistake made by practitioners using sklearn MLP. Unlike
> tree-based models, which are invariant to monotone feature transformations, MLPs are
> profoundly sensitive to feature scale.

### 6.2 Prone to Overfitting with Limited Data

**Mechanism:** An MLP with 100k parameters trained on 1k samples has 100 parameters per
sample — a massively overparameterized regime. Without regularization, the network will
memorize training noise. sklearn MLP provides only L2 regularization (`alpha`) and early
stopping — no dropout, no batch normalization (the two most effective regularizers for deep
networks). For severely limited data (<1000 samples), even well-regularized MLPs often
underperform simpler models.

**Detection:** Large gap between training accuracy and validation accuracy. Training accuracy
near 100%, test accuracy significantly lower.

**Mitigation:**
1. Increase `alpha` (L2 regularization)
2. Reduce `hidden_layer_sizes` (smaller capacity)
3. Enable `early_stopping=True`
4. If these fail: use PyTorch/Keras with dropout; or switch to gradient-boosted trees

### 6.3 Hyperparameter Sensitivity

**Mechanism:** MLP performance depends on at least 5 interdependent hyperparameters:
architecture (`hidden_layer_sizes`), regularization (`alpha`), learning rate
(`learning_rate_init`), batch size, and training duration (`max_iter`/`early_stopping`).
The interactions are nonlinear — a good architecture with a bad learning rate performs worse
than a mediocre architecture with a good learning rate. This makes random search or Optuna
essentially mandatory for getting the best out of an MLP.

**Detection:** Large variance in cross-validation scores across random seeds. Wide spread in
learning curves for different hyperparameter configurations.

**Mitigation:** Use Optuna (§9) with a budget of 50–100 trials. Focus on `alpha`,
`hidden_layer_sizes`, and `learning_rate_init` — these three account for most of the variance.

### 6.4 No Native Handling of Mixed Data Types

**Mechanism:** MLPs operate on continuous inputs. Categorical features must be encoded before
use (one-hot, ordinal, or target encoding), which can be lossy (one-hot for high-cardinality
features is impractical) and inflates feature dimensionality. Tree models handle categoricals
naturally at the split level.

**Detection:** Poor performance on datasets with many categorical features or high-cardinality
categoricals, where tree-based models perform well.

**Mitigation:** Target encoding for high-cardinality features; entity embeddings (learnable
dense representations for categoricals) in PyTorch/Keras; or simply use a gradient-boosted
tree model for datasets dominated by categorical features.

### 6.5 Not Robust to Irrelevant Features

**Mechanism:** Decision trees and random forests inherently perform feature selection at each
split — they ignore features that do not reduce impurity. An MLP must learn to suppress
irrelevant features through L2 regularization, which only pushes weights toward zero but does
not zero them out exactly (unlike L1). In high-dimensional spaces with many irrelevant features,
this is unreliable without strong regularization.

**Detection:** Performance significantly worse than trees on datasets with many irrelevant
features. SHAP values showing many features with non-negligible but noisy importance.

**Mitigation:** Pre-select features using tree-based importance or mutual information;
increase `alpha` to push irrelevant feature weights to near-zero.

### 6.6 Dying ReLU Problem

**Mechanism:** During training with ReLU, if a neuron's pre-activation $z = \mathbf{w}^T
\mathbf{a} + b < 0$ for every example in the training set, the ReLU gradient is 0, the weight
gradient for that neuron is 0, and it never updates. It is "dead." This can happen early in
training if the weight initialization produces large negative biases, or if an aggressive
learning rate causes large negative weight updates.

**Detection:** `clf.coefs_` contains weight matrices with rows that are all zeros or
near-zeros. Training loss plateaus immediately at a high value.

**Mitigation:**
1. Reduce `learning_rate_init` (the most common fix)
2. Switch to `activation='tanh'` (tanh does not die)
3. In PyTorch, switch to Leaky ReLU or ELU
4. Check for extreme feature values (outliers can produce extreme initial activations)

### 6.7 Local Minima and Optimization Challenges

**Mechanism:** The MLP loss surface is non-convex and has many local minima, saddle points,
and plateaus. Different random initializations can lead to substantially different final models.
In practice, most local minima on overparameterized networks generalize similarly (a result
of the "loss landscape" literature), but there are unlucky initializations.

**Detection:** Large variance in performance across runs with different `random_state` values.
Loss curves that plateau at different loss values.

**Mitigation:**
1. Always set `random_state` and test multiple seeds
2. With `solver='lbfgs'`, try multiple restarts
3. Use larger `batch_size` for smoother gradients
4. Use Adam (adaptive learning rates navigate the loss landscape more robustly than SGD)

### 6.8 Black-Box Interpretability

**Mechanism:** An MLP with 10,000+ parameters has no native interpretability mechanism
analogous to a decision tree's decision path or a linear model's coefficients. Each prediction
is the result of thousands of nonlinear transformations. Post-hoc explanation methods (SHAP,
LIME) approximate the model locally but are not exact.

**Detection:** Not a failure mode per se — a design property. Becomes a problem in regulated
industries (healthcare, finance) where model explainability is legally required.

**Mitigation:**
- Use SHAP's `KernelExplainer` for local feature importance (§8)
- Use permutation importance for global feature importance
- Use partial dependence plots (PDPs) for feature effect visualization
- If exact interpretability is required, consider using a tree model or a distillation approach
  (train a tree to mimic the MLP's predictions)

### 6.9 sklearn MLP-Specific Limitations

| Limitation | Mechanism | Mitigation |
|---|---|---|
| No GPU support | CPU-only implementation | Graduate to PyTorch/Keras |
| No dropout | Not implemented | Tune `alpha` more aggressively or graduate |
| No batch normalization | Not implemented | Graduate to PyTorch/Keras |
| No custom loss | Fixed loss functions (cross-entropy, MSE) | Graduate to PyTorch/Keras |
| Fixed output activations | softmax/sigmoid/identity auto-selected | No override possible in sklearn |
| No `partial_fit` for lbfgs | Full-batch solver cannot process streaming data | Use `adam` or `sgd` with `partial_fit` |
| No learning rate scheduling | Only for sgd; adam's internal schedule is fixed | Graduate for custom schedules |
| `ConvergenceWarning` by default | `max_iter=200` too low | Set `max_iter=1000+`, `early_stopping=True` |

---

## §7 — Diagnostic Plots & Evaluation

Knowing *how* to read an MLP's training behavior is as important as knowing how to configure it.
These are the diagnostic plots you should generate for every MLP you train.

### 7.1 Training Loss Curve

**What it shows:** The evolution of training loss (and optionally validation score) across
epochs. This is the primary window into the optimization process.

**How to read it:**
- Smooth, monotonically decreasing → healthy convergence
- Oscillating wildly → learning rate too high; reduce `learning_rate_init`
- Plateauing immediately → dying ReLU, feature scaling problem, or learning rate too low
- Training loss decreases but validation stalls early → overfitting; increase `alpha`

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load and prepare data
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Train with early stopping (captures validation scores)
clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False
)
clf.fit(X_train_sc, y_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training loss curve
axes[0].plot(clf.loss_curve_, label='Training loss', color='steelblue', lw=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (cross-entropy)')
axes[0].set_title('Training Loss Curve')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Validation scores (only available when early_stopping=True)
if hasattr(clf, 'validation_scores_'):
    axes[1].plot(clf.validation_scores_, label='Validation accuracy', color='darkorange', lw=2)
    axes[1].axvline(np.argmax(clf.validation_scores_), color='red', ls='--',
                    label=f'Best epoch: {np.argmax(clf.validation_scores_)}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Validation Score Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_loss_curves.png', dpi=150)
plt.show()

print(f"Best validation score: {max(clf.validation_scores_):.4f}")
print(f"n_iter_: {clf.n_iter_} epochs trained")
print(f"best_loss_: {clf.best_loss_:.6f}")
```

> 🔧 **In practice:** The `validation_scores_` attribute (note the plural — verify spelling in
> sklearn 1.5 docs) is only populated when `early_stopping=True`. Without it, you only have
> `loss_curve_` which shows training loss. For a full picture of overfitting, you need early
> stopping enabled.

### 7.2 Learning Curves (Sample Complexity)

**What it shows:** How training and validation score change as a function of training set size.
This is the definitive diagnostic for bias vs variance.

**How to read it:**
- Both curves plateau at high accuracy → model is performing near its potential; more data
  won't help; try a more complex architecture
- Training high, validation low, large gap → high variance (overfitting); need more data,
  more regularization, or smaller network
- Both curves plateau at low accuracy → high bias (underfitting); need more capacity
  (larger `hidden_layer_sizes`) or better features

```python
from sklearn.model_selection import learning_curve

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(100, 50), alpha=1e-4, max_iter=500,
        early_stopping=True, n_iter_no_change=15, random_state=42
    ))
])

train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training accuracy')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.2, color='steelblue')
ax.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation accuracy')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.2, color='darkorange')
ax.set_xlabel('Training set size')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curves — MLP on Digits')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_learning_curves.png', dpi=150)
plt.show()
```

### 7.3 Validation Curve (Hyperparameter Sensitivity)

**What it shows:** How training and validation score respond as you vary a single hyperparameter.
The "sweet spot" is where the validation curve peaks.

```python
from sklearn.model_selection import validation_curve

alphas = np.logspace(-6, 0, 10)   # 1e-6 to 1.0

train_scores, val_scores = validation_curve(
    pipe, X, y,
    param_name='mlp__alpha',
    param_range=alphas,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

fig, ax = plt.subplots(figsize=(9, 6))
ax.semilogx(alphas, train_scores.mean(axis=1), 'o-', color='steelblue', label='Training')
ax.fill_between(alphas, train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.2, color='steelblue')
ax.semilogx(alphas, val_scores.mean(axis=1), 'o-', color='darkorange', label='Validation')
ax.fill_between(alphas, val_scores.mean(axis=1) - val_scores.std(axis=1),
                         val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.2, color='darkorange')
ax.set_xlabel('alpha (L2 regularization)')
ax.set_ylabel('Accuracy')
ax.set_title('Validation Curve — alpha')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_validation_curve.png', dpi=150)
plt.show()
```

> 💡 **Intuition:** The validation curve for `alpha` shows the classic bias-variance trade-off.
> Too small: training accuracy high, validation lower (high variance). Too large: both accuracies
> drop (high bias). The crossover point is your optimal alpha.

### 7.4 Confusion Matrix (Classification)

**What it shows:** The full breakdown of correct and incorrect predictions by class pair —
which classes are being confused with which.

```python
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

y_pred = pipe.predict(X_test)

fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=digits.target_names,
    cmap='Blues',
    ax=ax
)
ax.set_title('Confusion Matrix — MLP on Digits Test Set')
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png', dpi=150)
plt.show()

print(classification_report(y_test, y_pred, target_names=[str(c) for c in digits.target_names]))
```

**What "good" looks like:** Dark diagonal, light off-diagonal. The digits most commonly confused
are 4/9, 3/5, 7/9 — visually similar pairs.

### 7.5 ROC Curves and AUC (Binary and Multiclass)

```python
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.preprocessing import label_binarize

# For multiclass: one-vs-rest ROC curves
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)
y_score    = pipe.predict_proba(X_test)

fig, ax = plt.subplots(figsize=(9, 7))
for i, cls in enumerate(classes):
    RocCurveDisplay.from_predictions(
        y_test_bin[:, i], y_score[:, i],
        name=f'Digit {cls}',
        ax=ax,
        alpha=0.7
    )
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_title('ROC Curves — MLP on Digits (One-vs-Rest)')
ax.legend(loc='lower right', fontsize=7)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_roc_curves.png', dpi=150)
plt.show()

# Macro-average OVR AUC
auc_macro = roc_auc_score(y_test, y_score, multi_class='ovr', average='macro')
print(f"Macro OVR AUC: {auc_macro:.4f}")
```

### 7.6 Residual Plots (Regression)

For `MLPRegressor`, residual analysis is essential to diagnose model quality.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

housing = fetch_california_housing()
X_reg, y_reg = housing.data, housing.target
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

pipe_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(100, 50), alpha=1e-4,
        max_iter=1000, early_stopping=True, random_state=42
    ))
])
pipe_reg.fit(X_tr, y_tr)
y_pred_reg = pipe_reg.predict(X_te)

residuals = y_te - y_pred_reg
r2 = r2_score(y_te, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_te, y_pred_reg))
print(f"R² = {r2:.4f}, RMSE = {rmse:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Predicted vs Actual
axes[0].scatter(y_te, y_pred_reg, alpha=0.3, s=10, color='steelblue')
lims = [min(y_te.min(), y_pred_reg.min()), max(y_te.max(), y_pred_reg.max())]
axes[0].plot(lims, lims, 'r--', lw=2)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title(f'Predicted vs Actual (R²={r2:.3f})')
axes[0].grid(alpha=0.3)

# Plot 2: Residuals vs Predicted
axes[1].scatter(y_pred_reg, residuals, alpha=0.3, s=10, color='darkorange')
axes[1].axhline(0, color='red', lw=2)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Residual')
axes[1].set_title('Residuals vs Predicted')
axes[1].grid(alpha=0.3)

# Plot 3: Residual histogram
axes[2].hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[2].axvline(0, color='red', lw=2)
axes[2].set_xlabel('Residual')
axes[2].set_ylabel('Count')
axes[2].set_title('Residual Distribution')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_residuals.png', dpi=150)
plt.show()
```

**What "good" looks like:**
- Predicted vs Actual: points clustered along the 45° diagonal, no systematic curves
- Residuals vs Predicted: random scatter around 0, no fan shape (heteroskedasticity), no U-curve (systematic bias)
- Residual distribution: approximately normal, centered at 0

**What "bad" looks like:**
- Systematic curve in residuals vs predicted → the model is missing a nonlinear pattern (underfit)
- Fan shape → heteroskedastic residuals; the model's error varies with the predicted value
- Heavy tails in residual distribution → outliers or the model is badly wrong on some examples

### 7.7 Calibration Curve (Probability Quality)

**What it shows:** Whether the model's predicted probabilities match actual frequencies.
A perfectly calibrated model predicts 80% probability means the event occurs 80% of the time.

```python
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X_bc, y_bc = bc.data, bc.target
X_tr_bc, X_te_bc, y_tr_bc, y_te_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

pipe_bc = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), alpha=1e-3,
                          max_iter=1000, early_stopping=True, random_state=42))
])
pipe_bc.fit(X_tr_bc, y_tr_bc)

fig, ax = plt.subplots(figsize=(8, 7))
CalibrationDisplay.from_estimator(pipe_bc, X_te_bc, y_te_bc, n_bins=10, ax=ax,
                                   name='MLP (before calibration)')
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.set_title('Calibration Curve — MLP on Breast Cancer')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_calibration.png', dpi=150)
plt.show()
```

**What "good" looks like:** The curve follows the diagonal closely. MLPs tend to be reasonably
well-calibrated due to the cross-entropy loss, but can be overconfident in some regimes.

**If miscalibrated:** Apply Platt scaling or isotonic regression via
`sklearn.calibration.CalibratedClassifierCV`.

---

## §8 — Explainability & Interpretable AI

MLPs are often described as "black boxes," but this understates what we can recover with
modern explanation tools. The key is knowing which tool is right for this model class and
understanding what those tools can and cannot say.

### 8.1 What Built-In Interpretability the MLP Provides

Unlike linear models (where coefficients directly quantify feature effects) or decision trees
(where the path through the tree explains a prediction), the MLP provides essentially no
native interpretability. There are no coefficients that map cleanly to "feature X increases the
prediction by Y." The `coefs_` attributes (one weight matrix per layer) are difficult to
interpret directly because each feature's effect is routed through multiple layers of nonlinear
transformations before reaching the output.

The one native diagnostic is **weight visualization**: for the first hidden layer, you can
visualize each neuron's weight vector as an image (if the input is image-like), revealing what
visual patterns each neuron detects. This is meaningful for MNIST but not for arbitrary tabular
data.

```python
import matplotlib.pyplot as plt
import numpy as np

# Visualize first hidden layer weights for digit data
# Each row of coefs_[0] is the weight vector of one hidden neuron
# Shape: (n_features=64, n_hidden_1=100)
fig, axes = plt.subplots(10, 10, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i < clf.coefs_[0].shape[1]:
        weight_image = clf.coefs_[0][:, i].reshape(8, 8)
        ax.imshow(weight_image, cmap='RdBu', aspect='auto',
                  vmin=-np.abs(clf.coefs_[0]).max(),
                  vmax=np.abs(clf.coefs_[0]).max())
    ax.axis('off')
plt.suptitle('First Hidden Layer Weights (8×8 receptive fields)', fontsize=14)
plt.tight_layout()
plt.savefig('mlp_weight_visualization.png', dpi=150)
plt.show()
```

### 8.2 SHAP with KernelExplainer

**Which explainer:** For sklearn's `MLPClassifier`, use `shap.KernelExplainer`. This is
model-agnostic — it works by treating the model as a black box and approximating SHAP values
via a weighted linear regression over all feature subsets (sampling-based approximation).

> 📜 **Citation:** Lundberg, S. M. & Lee, S. I. (2017). "A unified approach to interpreting
> model predictions." *NeurIPS*, 30. URL: https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

**Cost:** `KernelExplainer` is slow — $O(2^p)$ in theory, $O(T \cdot n_{\text{background}} \cdot n_{\text{test}})$
in practice with sampling ($T$ = number of evaluations per sample). Always use a small background
summary (k-means or a small random sample) and compute SHAP values on a small test subset.

```python
import shap
import numpy as np

# Use the breast cancer binary classification example for cleaner SHAP plots
# pipe_bc is already fitted above

# Summarize background with k-means (10 cluster centroids)
# This approximates the marginal distribution efficiently
X_tr_bc_sc = pipe_bc.named_steps['scaler'].transform(X_tr_bc)
background = shap.kmeans(X_tr_bc_sc, 10)

# Create explainer on the final MLP step (inputs are already scaled)
mlp_step = pipe_bc.named_steps['mlp']
explainer = shap.KernelExplainer(mlp_step.predict_proba, background)

# Compute SHAP values for 50 test samples (slow — may take minutes)
X_te_bc_sc = pipe_bc.named_steps['scaler'].transform(X_te_bc)
shap_values = explainer.shap_values(X_te_bc_sc[:50], nsamples=100)
# shap_values is a list of length n_classes; each array is (n_samples, n_features)
# For binary, shap_values[1] is SHAP values for class 1 (malignant→benign)
```

> ⚠️ **Pitfall:** `shap.KernelExplainer` expects raw model inputs (already scaled). If you
> pass `X_test` directly to `KernelExplainer` but the `predict_proba` function internally
> calls the full pipeline (including scaler), the background data and test data must be on the
> same scale. The cleanest approach: either pass the full pipeline's `predict_proba` with
> unscaled data, or pass the MLP's `predict_proba` directly with pre-scaled data. The code
> above uses the second approach.

**Alternative: use the full pipeline with unscaled data:**
```python
# Simpler but slightly slower (background will be in unscaled space)
background_raw = shap.kmeans(X_tr_bc, 10)
explainer_pipe = shap.KernelExplainer(pipe_bc.predict_proba, background_raw)
shap_values_pipe = explainer_pipe.shap_values(X_te_bc[:50], nsamples=100)
```

#### SHAP Summary Plot (Global Importance)

```python
# Summary beeswarm: each dot is one sample; position=SHAP value; color=feature value
# For binary classification, plot SHAP values for class 1
shap.summary_plot(
    shap_values[1],          # SHAP values for the positive class
    X_te_bc_sc[:50],
    feature_names=bc.feature_names,
    plot_type='beeswarm',    # shows distribution
    show=False
)
plt.title('SHAP Summary (Beeswarm) — MLP on Breast Cancer')
plt.tight_layout()
plt.savefig('mlp_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
```

**How to read the beeswarm:** Each row is a feature. The x-axis is the SHAP value — positive
means the feature pushed the prediction toward class 1 (benign), negative means toward class 0
(malignant). Color is feature value (red = high, blue = low). A wide spread means the feature
has high, variable impact. Dense clustering near zero means low importance.

#### SHAP Waterfall Plot (Local Explanation)

```python
# Explain a single prediction
sample_idx = 0
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[1][sample_idx],
        base_values=explainer.expected_value[1],
        data=X_te_bc_sc[sample_idx],
        feature_names=bc.feature_names
    ),
    show=False
)
plt.title(f'SHAP Waterfall — Sample {sample_idx} (True class: {y_te_bc[sample_idx]})')
plt.tight_layout()
plt.savefig('mlp_shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
```

**How to read the waterfall:** Starting from the base value (average model output), each bar
shows how much each feature pushed the prediction up (red) or down (blue). The final predicted
probability is the base value plus all SHAP contributions.

#### SHAP Bar Plot (Mean Absolute Importance)

```python
# Mean |SHAP value| per feature — simple global feature importance
mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(range(15), mean_abs_shap[sorted_idx[:15]][::-1], color='steelblue')
ax.set_yticks(range(15))
ax.set_yticklabels([bc.feature_names[i] for i in sorted_idx[:15]][::-1], fontsize=10)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Top 15 Features by Mean |SHAP| — MLP')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_shap_bar.png', dpi=150)
plt.show()
```

### 8.3 Permutation Importance

Permutation importance is a model-agnostic, post-hoc importance measure. For each feature,
it measures how much the validation score drops when that feature's values are randomly shuffled
(breaking its relationship with the target). A large drop means the feature is important; no
drop means the model can predict equally well without it.

**Advantage over SHAP:** Much faster. **Disadvantage:** Can miss interactions and is affected
by correlated features (correlated features may show lower importance because shuffling one
doesn't break the information carried by the other).

```python
from sklearn.inspection import permutation_importance

# Compute permutation importance on the test set
perm_result = permutation_importance(
    pipe_bc, X_te_bc, y_te_bc,
    n_repeats=10,
    random_state=42,
    scoring='roc_auc'
)

perm_sorted_idx = perm_result.importances_mean.argsort()[::-1][:15]

fig, ax = plt.subplots(figsize=(9, 7))
ax.boxplot(
    perm_result.importances[perm_sorted_idx].T,
    vert=False,
    labels=[bc.feature_names[i] for i in perm_sorted_idx]
)
ax.axvline(0, color='red', ls='--')
ax.set_xlabel('Decrease in AUC when feature permuted')
ax.set_title('Permutation Importance — MLP on Breast Cancer')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_permutation_importance.png', dpi=150)
plt.show()
```

### 8.4 Partial Dependence Plots (PDP) and ICE Curves

**PDP:** Shows the marginal effect of one or two features on the predicted outcome, averaging
over all other features. Reveals the direction and shape of a feature's relationship with the
prediction.

**ICE (Individual Conditional Expectation):** Shows the same thing for each individual sample —
the predicted outcome as the focal feature varies, holding all others constant. ICE reveals
heterogeneity: if all ICE lines have the same shape as the PDP, the feature effect is
homogeneous; if ICE lines diverge, there are interactions.

```python
from sklearn.inspection import PartialDependenceDisplay

# Top 2 features by permutation importance
top_features = perm_sorted_idx[:2].tolist()

fig, ax = plt.subplots(figsize=(12, 5))
PartialDependenceDisplay.from_estimator(
    pipe_bc,
    X_te_bc,
    features=top_features,
    feature_names=bc.feature_names,
    kind='both',       # 'both' shows PDP line + ICE curves
    subsample=100,     # show 100 ICE curves
    ax=ax,
    random_state=42
)
plt.suptitle('PDP + ICE Curves — Top 2 Features')
plt.tight_layout()
plt.savefig('mlp_pdp_ice.png', dpi=150)
plt.show()
```

**How to read ICE + PDP:**
- If ICE curves are parallel (all same shape) → no interactions; feature effect is homogeneous
- If ICE curves cross or diverge → interaction effects; the feature's impact depends on other features
- The PDP is the average of all ICE curves — if ICE diverges, PDP can be misleading

### 8.5 LIME (Local Interpretable Model-Agnostic Explanations)

LIME fits a simple local linear model in the neighborhood of a single test point and uses its
coefficients as an explanation. For tabular data, it perturbs features around the point, gets
the MLP's predictions, and fits a weighted linear regression.

```python
# pip install lime
from lime.lime_tabular import LimeTabularExplainer

explainer_lime = LimeTabularExplainer(
    training_data=X_tr_bc,
    feature_names=bc.feature_names,
    class_names=['malignant', 'benign'],
    mode='classification',
    random_state=42
)

# Explain one test sample
sample_idx = 5
exp = explainer_lime.explain_instance(
    data_row=X_te_bc[sample_idx],
    predict_fn=pipe_bc.predict_proba,
    num_features=10
)
exp.show_in_notebook(show_table=True)   # In Jupyter
# Or: exp.as_pyplot_figure()           # For script
```

### 8.6 Caveats and Limitations of MLP Explanations

1. **KernelExplainer SHAP values are approximations.** Unlike `TreeExplainer` (which computes
   exact SHAP values in $O(T \cdot L \cdot 2^D)$ time), `KernelExplainer` uses sampling and
   approximation. With too few samples (`nsamples` too low), values are noisy.

2. **SHAP for neural networks has a better option: `GradientExplainer` or `DeepExplainer`.**
   For PyTorch/Keras models, `shap.GradientExplainer` uses backpropagation to compute SHAP
   values much faster than `KernelExplainer`. For sklearn MLP, these are not available
   (sklearn models don't expose gradients). If you need fast, accurate SHAP for neural
   networks, use PyTorch + `shap.GradientExplainer`.

3. **PDP assumes feature independence.** If two features are highly correlated, averaging over
   one while varying the other can produce combinations that never appear in the data —
   unrealistic extrapolations. Use individual conditional expectation (ICE) or accumulated
   local effects (ALE, via `PyALE` package) when features are correlated.

4. **Local explanations (SHAP waterfall, LIME) apply only to the specific sample.** Do not
   generalize a single sample's explanation to the whole model without checking the global
   summary.

5. **Explanations inherit model errors.** If the MLP makes a wrong prediction, the SHAP
   explanation tells you why the model was wrong, not why the correct answer is what it is.

---

## §9 — Complete Python Worked Example

This section walks through two complete end-to-end workflows: classification on the Digits
dataset and regression on California Housing. Every step from loading data to final SHAP
interpretation is included.

### Part A: Classification — Digits Dataset (10-class)

```python
# ============================================================
# MLP CLASSIFICATION WORKED EXAMPLE — sklearn digits dataset
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 1. DATA LOADING & EDA ────────────────────────────────────────────────────
digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Features: {X.shape[1]} (8×8 pixel intensities, 0–16)")
print(f"Classes: {np.unique(y)} ({len(np.unique(y))} digits)")
print(f"Class distribution:\n{pd.Series(y).value_counts().sort_index()}")

# Visualize a sample of each digit
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    axes[0, digit].imshow(X[idx].reshape(8, 8), cmap='gray_r')
    axes[0, digit].set_title(f'{digit}', fontsize=12)
    axes[0, digit].axis('off')
    # Second row: another example
    idx2 = np.where(y == digit)[0][1]
    axes[1, digit].imshow(X[idx2].reshape(8, 8), cmap='gray_r')
    axes[1, digit].axis('off')
plt.suptitle('Two examples of each digit (8×8 pixels)', fontsize=13)
plt.tight_layout()
plt.savefig('digits_samples.png', dpi=150)
plt.show()

# Feature statistics
print(f"\nPixel intensity range: [{X.min():.1f}, {X.max():.1f}]")
print(f"Mean pixel intensity: {X.mean():.2f}")
print(f"Features with zero variance: {(X.std(0) == 0).sum()}")

# ── 2. TRAIN/VAL/TEST SPLIT ──────────────────────────────────────────────────
X_train_raw, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_raw, X_val, y_train, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.125, random_state=42, stratify=y_train
)
# Splits: ~70% train / 12.5% val / 20% test
print(f"\nTrain: {X_train_raw.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ── 3. PREPROCESSING PIPELINE ────────────────────────────────────────────────
# For MLP: mandatory StandardScaler — neural networks REQUIRE feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)   # fit on train only!
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

print(f"\nPost-scaling: mean≈{X_train.mean():.4f}, std≈{X_train.std():.4f}")

# ── 4. BASELINE MODEL ────────────────────────────────────────────────────────
baseline_clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False
)
baseline_clf.fit(X_train, y_train)

baseline_val_acc = baseline_clf.score(X_val_sc, y_val)
print(f"\nBaseline MLP validation accuracy: {baseline_val_acc:.4f}")
print(f"Epochs trained: {baseline_clf.n_iter_}")

# ── 5. HYPERPARAMETER TUNING WITH OPTUNA ─────────────────────────────────────
# We tune on train+val (using the scaled features; CV handles the split)
X_trainval = np.vstack([X_train, X_val_sc])
y_trainval  = np.concatenate([y_train, y_val])

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layer_sizes = tuple(
        trial.suggest_int(f'n_units_l{i}', 32, 256, log=True) for i in range(n_layers)
    )
    alpha          = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
    lr_init        = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    activation     = trial.suggest_categorical('activation', ['relu', 'tanh'])

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver='adam',
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=42
    )
    # 5-fold stratified CV on train+val
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_trainval, y_trainval, cv=cv,
                             scoring='accuracy', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"\nOptuna best params: {study.best_params}")
print(f"Optuna best CV accuracy: {study.best_value:.4f}")

# ── 6. FINAL MODEL WITH BEST PARAMS ──────────────────────────────────────────
best = study.best_params
n_layers = best['n_layers']
hidden_layer_sizes = tuple(best[f'n_units_l{i}'] for i in range(n_layers))

final_clf = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=best['activation'],
    solver='adam',
    alpha=best['alpha'],
    learning_rate_init=best['learning_rate_init'],
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=25,
    random_state=42,
    verbose=False
)
final_clf.fit(X_trainval, y_trainval)   # fit on full train+val

test_acc = final_clf.score(X_test_sc, y_test)
print(f"\nFinal MLP test accuracy: {test_acc:.4f}")
print(f"Architecture: {hidden_layer_sizes}, activation: {best['activation']}")
print(f"alpha: {best['alpha']:.2e}, lr: {best['learning_rate_init']:.2e}")
print(f"Epochs trained: {final_clf.n_iter_}")

# ── 7. DIAGNOSTIC PLOTS ───────────────────────────────────────────────────────
y_pred = final_clf.predict(X_test_sc)
y_prob = final_clf.predict_proba(X_test_sc)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loss curve
axes[0].plot(final_clf.loss_curve_, color='steelblue', lw=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Training Loss Curve')
axes[0].grid(alpha=0.3)

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=digits.target_names,
    cmap='Blues',
    ax=axes[1]
)
axes[1].set_title(f'Confusion Matrix (test acc={test_acc:.3f})')

# Class-level precision-recall
report = classification_report(y_test, y_pred, output_dict=True)
classes = [str(c) for c in range(10)]
prec = [report[c]['precision'] for c in classes]
rec  = [report[c]['recall'] for c in classes]
x = np.arange(10)
w = 0.35
axes[2].bar(x - w/2, prec, w, label='Precision', color='steelblue')
axes[2].bar(x + w/2, rec,  w, label='Recall',    color='darkorange')
axes[2].set_xticks(x)
axes[2].set_xticklabels(classes)
axes[2].set_xlabel('Digit Class')
axes[2].set_ylabel('Score')
axes[2].set_title('Per-Class Precision & Recall')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_digits_diagnostics.png', dpi=150)
plt.show()

# Macro-average OVR AUC
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=np.arange(10))
macro_auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')
print(f"\nMacro OVR AUC: {macro_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ── 8. PERMUTATION IMPORTANCE ────────────────────────────────────────────────
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    final_clf, X_test_sc, y_test,
    n_repeats=5, random_state=42, scoring='accuracy'
)
top15_idx = perm.importances_mean.argsort()[::-1][:15]

fig, ax = plt.subplots(figsize=(9, 6))
ax.boxplot(perm.importances[top15_idx].T, vert=False,
           labels=[f'Pixel ({i//8},{i%8})' for i in top15_idx])
ax.axvline(0, color='red', ls='--')
ax.set_xlabel('Accuracy decrease when feature permuted')
ax.set_title('Top 15 Pixels by Permutation Importance')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_digits_perm_importance.png', dpi=150)
plt.show()

# ── 9. SHAP EXPLANATIONS ─────────────────────────────────────────────────────
# KernelExplainer: use k-means summary of background + small test set
background = shap.kmeans(X_trainval, 10)   # 10 k-means centroids
explainer = shap.KernelExplainer(final_clf.predict_proba, background)

# Compute SHAP for 30 test samples (slow — ~2-5 minutes on CPU)
shap_values = explainer.shap_values(X_test_sc[:30], nsamples=100)
# shap_values is a list of 10 arrays (one per class), each shape (30, 64)

# Global summary: use SHAP values for class 0 (digit "0")
# Each feature is a pixel; visualize as 8×8 heatmap
mean_abs_shap_class0 = np.abs(shap_values[0]).mean(axis=0)   # shape (64,)
shap_image = mean_abs_shap_class0.reshape(8, 8)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(shap_image, cmap='hot')
axes[0].set_title('Mean |SHAP| for class "0" — which pixels matter?')
axes[0].axis('off')
plt.colorbar(axes[0].images[0], ax=axes[0])

# SHAP summary beeswarm for class 0
shap.summary_plot(shap_values[0], X_test_sc[:30],
                  feature_names=[f'p({i//8},{i%8})' for i in range(64)],
                  max_display=15, show=False)
plt.title('SHAP Summary — Class "0"')
plt.tight_layout()
plt.savefig('mlp_digits_shap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== FINAL RESULTS ===")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Macro OVR AUC: {macro_auc:.4f}")
print(f"Architecture: {hidden_layer_sizes}")
```

### Part B: Regression — California Housing Dataset

```python
# ============================================================
# MLP REGRESSION WORKED EXAMPLE — California Housing
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 1. DATA LOADING & EDA ────────────────────────────────────────────────────
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names
# y = median house value in $100k units; features: MedInc, HouseAge, AveRooms, etc.

df = pd.DataFrame(X, columns=feature_names)
df['MedHouseVal'] = y

print(f"Shape: X={X.shape}, y={y.shape}")
print(f"\nFeature statistics:")
print(df.describe().round(2))

# Distribution of target
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(y, bins=50, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Median House Value ($100k)')
axes[0].set_title('Target Distribution')
axes[0].grid(alpha=0.3)

# Correlation heatmap
axes[1].remove()
ax_corr = fig.add_subplot(1, 3, 2)
corr = df.corr()
sns_colors = sns.color_palette('coolwarm', as_cmap=True)
sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax_corr, square=True,
            annot=True, fmt='.2f', fontsize=7)
ax_corr.set_title('Feature Correlations')

# Scatter: MedInc vs target
axes[2].scatter(df['MedInc'], y, alpha=0.1, s=2, color='steelblue')
axes[2].set_xlabel('Median Income (tens of thousands)')
axes[2].set_ylabel('Median House Value ($100k)')
axes[2].set_title('Income vs House Value')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('housing_eda.png', dpi=150)
plt.show()

# ── 2. SPLIT ─────────────────────────────────────────────────────────────────
X_train_raw, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train_raw.shape[0]} | Test: {X_test.shape[0]}")

# ── 3. PREPROCESSING ─────────────────────────────────────────────────────────
# California Housing features span very different scales:
# MedInc ~[0, 15], AveRooms ~[1, 100], Population ~[3, 35k]
# StandardScaler is essential.
pipe_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    ))
])

# ── 4. BASELINE CROSS-VALIDATION ─────────────────────────────────────────────
baseline_r2 = cross_val_score(pipe_reg, X_train_raw, y_train, cv=5,
                               scoring='r2', n_jobs=-1)
print(f"\nBaseline 5-fold CV R²: {baseline_r2.mean():.4f} ± {baseline_r2.std():.4f}")

# ── 5. HYPERPARAMETER TUNING ─────────────────────────────────────────────────
def objective_reg(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layer_sizes = tuple(
        trial.suggest_int(f'h{i}', 32, 256, log=True) for i in range(n_layers)
    )
    alpha    = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
    lr_init  = trial.suggest_float('lr_init', 1e-4, 1e-2, log=True)
    batch_sz = trial.suggest_categorical('batch_size', ['auto', 64, 128, 256])

    reg = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=lr_init,
            batch_size=batch_sz,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42
        ))
    ])
    scores = cross_val_score(reg, X_train_raw, y_train, cv=5,
                             scoring='r2', n_jobs=-1)
    return scores.mean()

study_reg = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_reg.optimize(objective_reg, n_trials=40, show_progress_bar=False)

print(f"\nOptuna best R²: {study_reg.best_value:.4f}")
print(f"Best params: {study_reg.best_params}")

# ── 6. FINAL MODEL ────────────────────────────────────────────────────────────
best_r = study_reg.best_params
n_l = best_r['n_layers']
best_sizes = tuple(best_r[f'h{i}'] for i in range(n_l))

pipe_best = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=best_sizes,
        alpha=best_r['alpha'],
        learning_rate_init=best_r['lr_init'],
        batch_size=best_r['batch_size'],
        max_iter=3000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42
    ))
])
pipe_best.fit(X_train_raw, y_train)

y_pred = pipe_best.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print(f"\nTest R²  = {r2:.4f}")
print(f"Test RMSE = {rmse:.4f} ($100k units)")
print(f"Test MAE  = {mae:.4f} ($100k units)")

# ── 7. RESIDUAL DIAGNOSTICS ───────────────────────────────────────────────────
residuals = y_test - y_pred
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].scatter(y_test, y_pred, alpha=0.2, s=5, color='steelblue')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, 'r--', lw=2)
axes[0].set_xlabel('Actual ($100k)')
axes[0].set_ylabel('Predicted ($100k)')
axes[0].set_title(f'Predicted vs Actual (R²={r2:.3f})')
axes[0].grid(alpha=0.3)

axes[1].scatter(y_pred, residuals, alpha=0.2, s=5, color='darkorange')
axes[1].axhline(0, color='red', lw=2)
axes[1].set_xlabel('Predicted ($100k)')
axes[1].set_ylabel('Residual')
axes[1].set_title('Residuals vs Predicted')
axes[1].grid(alpha=0.3)

axes[2].hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
axes[2].axvline(0, color='red', lw=2)
axes[2].set_xlabel('Residual ($100k)')
axes[2].set_title('Residual Distribution')
axes[2].grid(alpha=0.3)

plt.suptitle(f'MLP Regression Diagnostics — California Housing (RMSE={rmse:.3f})', fontsize=13)
plt.tight_layout()
plt.savefig('mlp_housing_residuals.png', dpi=150)
plt.show()

# ── 8. PARTIAL DEPENDENCE PLOTS ──────────────────────────────────────────────
# Top 2 features by correlation with target
top_feat_idx = np.argsort(np.abs(np.corrcoef(X_train_raw.T, y_train)[:-1, -1]))[::-1][:2]
print(f"\nTop 2 correlated features: {[feature_names[i] for i in top_feat_idx]}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
PartialDependenceDisplay.from_estimator(
    pipe_best,
    X_train_raw,
    features=top_feat_idx.tolist(),
    feature_names=feature_names,
    kind='both',        # PDP + ICE
    subsample=200,
    ax=axes,
    random_state=42
)
plt.suptitle('PDP + ICE — Top 2 Features')
plt.tight_layout()
plt.savefig('mlp_housing_pdp.png', dpi=150)
plt.show()

# ── 9. PERMUTATION IMPORTANCE ─────────────────────────────────────────────────
perm_reg = permutation_importance(
    pipe_best, X_test, y_test,
    n_repeats=10, random_state=42, scoring='r2'
)
sorted_feat_idx = perm_reg.importances_mean.argsort()[::-1]

fig, ax = plt.subplots(figsize=(9, 6))
ax.boxplot(
    perm_reg.importances[sorted_feat_idx].T,
    vert=False,
    labels=[feature_names[i] for i in sorted_feat_idx]
)
ax.axvline(0, color='red', ls='--')
ax.set_xlabel('R² decrease when feature permuted')
ax.set_title('Permutation Feature Importance — MLP California Housing')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_housing_perm_importance.png', dpi=150)
plt.show()

# ── 10. SHAP EXPLANATIONS ─────────────────────────────────────────────────────
import shap

X_tr_sc = pipe_best.named_steps['scaler'].transform(X_train_raw)
X_te_sc = pipe_best.named_steps['scaler'].transform(X_test)
mlp_reg_step = pipe_best.named_steps['mlp']

background = shap.kmeans(X_tr_sc, 10)
explainer_reg = shap.KernelExplainer(mlp_reg_step.predict, background)

# 30 test samples (slow)
shap_vals_reg = explainer_reg.shap_values(X_te_sc[:30], nsamples=100)

# Summary beeswarm
shap.summary_plot(
    shap_vals_reg, X_te_sc[:30],
    feature_names=feature_names,
    show=False
)
plt.title('SHAP Summary — MLP California Housing Regression')
plt.tight_layout()
plt.savefig('mlp_housing_shap.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 11. FINAL INTERPRETATION ──────────────────────────────────────────────────
print("\n=== INTERPRETATION ===")
print(f"The MLP achieves R²={r2:.3f} on the test set, explaining {r2*100:.1f}% of house")
print(f"price variance. RMSE={rmse:.3f} corresponds to a typical prediction error of")
print(f"${rmse*1e5:,.0f}. The SHAP analysis shows that MedInc (median income) and")
print(f"geographic features (Latitude, Longitude) are the dominant drivers, consistent")
print(f"with the domain understanding that location and wealth concentration drive")
print(f"housing prices in California.")
print(f"\nNote: XGBoost typically achieves R²≈0.84 on this dataset. The MLP achieves")
print(f"comparable performance at R²≈{r2:.2f}, confirming that for tabular regression,")
print(f"the models are competitive but trees often have a slight edge.")
```

---

## §10 — When to Use This Algorithm (Decision Guide)

Use this section as a decision framework. The first question is always: should you be using any
neural network at all, or is a gradient-boosted tree the right default?

### The Core Question: MLP vs Trees on Tabular Data

> 📊 **Benchmark:** Grinsztajn et al. (2022) found that on 45 tabular benchmark datasets at
> ~10k samples, gradient-boosted trees outperform all neural architectures on average — including
> after 20,000 compute hours of neural architecture search. The gap narrows for larger datasets
> and disappears for non-tabular problems (images, text, sequences). On truly tabular data with
> <500k rows, start with XGBoost/LightGBM, not MLPs.

### Decision Table

| Data Characteristic | Recommendation | Reason |
|---|---|---|
| **Dataset size: <500 samples** | Logistic Regression, SVM, or small tree | MLP will overfit; not enough data to learn representations |
| **Dataset size: 500–50k rows, tabular** | Try XGBoost first; MLP as comparison | Trees win on average (Grinsztajn 2022); MLP is a competitive baseline |
| **Dataset size: >500k rows, tabular** | MLP (GPU) or XGBoost with subsample | At scale, MLP begins to close the gap with GPU |
| **Tabular with mostly categorical features** | CatBoost, LightGBM | Trees handle categoricals natively; MLP needs encoding |
| **Tabular with many irrelevant features** | XGBoost, Random Forest | Trees perform implicit feature selection; MLP relies on regularization |
| **Dense continuous features** | MLP competitive | Where MLP's gradient-based weight learning shines |
| **Image classification** | CNN (PyTorch/Keras) | Convolutional architecture exploits spatial structure; vanilla MLP ignores it |
| **Sequence / time-series** | LSTM, Transformer (PyTorch/Keras) | MLP has no memory; recurrent/attention architectures are needed |
| **NLP / text** | Pre-trained Transformer (BERT, GPT) | Representation learning at scale; MLP cannot compete |
| **Small dataset with known structure** | Logistic Regression, Decision Tree | Interpretable; don't need the complexity of MLP |
| **Need strict interpretability** | Linear model, Decision Tree | If you need coefficient-level interpretability or decision paths |
| **Features are embeddings from another model** | MLP is a strong choice | MLPs excel at learning from dense embedding spaces |
| **Online/streaming learning** | `MLPClassifier` with `solver='adam'` + `partial_fit` | Supports incremental fitting; trees are batch-only |
| **Need uncertainty quantification** | Bayesian NN (PyTorch + Pyro), MC Dropout | Standard MLP gives point predictions; no native uncertainty |
| **Production serving, latency <10ms** | Small MLP (PyTorch TorchScript) | Deployable with PyTorch; sklearn MLP can be used via ONNX export |

### When to Use sklearn `MLPClassifier` Specifically

```
Start here if ALL of the following are true:
✓ Tabular data (not images/text/sequences)
✓ n_samples < 50,000 (otherwise GPU-based PyTorch is preferable)
✓ You need sklearn Pipeline compatibility (preprocessing + model in one object)
✓ You want a quick neural network baseline without PyTorch setup overhead
✓ You do NOT need: dropout, batch normalization, GPU, custom loss

If you need any of the following → graduate to PyTorch or Keras:
✗ Dataset > 50k rows AND you need fast training
✗ Dropout regularization
✗ Batch normalization
✗ Custom loss function
✗ CNN/RNN/Transformer layers
✗ Model serving in production
✗ Learning rate scheduling
```

### Flowchart: Which Package?

```
                     New supervised learning problem
                              │
              ┌───────────────┴───────────────┐
         Tabular data?                   Non-tabular?
              │                               │
     ┌────────┴───────┐               ┌───────┴────────┐
  n < 500?       n ≥ 500              Image/audio?   Text/sequence?
     │                │                    │               │
  LR/SVM/Tree    Try XGBoost           CNN             Transformer
                 first                 (PyTorch)       (PyTorch/HF)
                     │
              MLP comparable?
                     │
            ┌────────┴────────┐
          Yes                  No
            │                  │
     Need dropout/BN/GPU?   Stay with XGBoost
            │
     ┌──────┴──────┐
    Yes             No
     │               │
  PyTorch/Keras   sklearn MLP
```

### Checklist Before Training Any MLP

Before training, confirm:

1. **Features are scaled** → `StandardScaler` (or `MinMaxScaler` for bounded inputs)
2. **`max_iter` is set high** → at least 1000, ideally 2000+
3. **`early_stopping=True`** → prevents training for a fixed epoch count blindly
4. **`random_state` is set** → reproducibility
5. **You have a baseline** → always compare against logistic regression and a gradient-boosted tree
6. **`alpha` has been tuned** → log-scale search; most important regularizer
7. **No `ConvergenceWarning`** → if you see it, fix `max_iter` and scaling before interpreting results

---

## §11 — Resources & Further Reading

### The Original Papers (Essential)

| Paper | Citation | URL |
|---|---|---|
| Rosenblatt Perceptron (1958) | Rosenblatt, F. "The perceptron: a probabilistic model for information storage and organization in the brain." *Psychological Review*, 65(6), 386–408. | https://doi.org/10.1037/h0042519 |
| Backpropagation (1986) | Rumelhart, D. E., Hinton, G. E., & Williams, R. J. "Learning representations by back-propagating errors." *Nature*, 323(6088), 533–536. | https://doi.org/10.1038/323533a0 |
| Universal Approximation — Cybenko (1989) | Cybenko, G. "Approximation by superpositions of a sigmoidal function." *Mathematics of Control, Signals, and Systems*, 2(4), 303–314. | https://doi.org/10.1007/BF02551274 |
| Universal Approximation — Hornik (1991) | Hornik, K. "Approximation capabilities of multilayer feedforward networks." *Neural Networks*, 4(2), 251–257. | https://doi.org/10.1016/0893-6080(91)90009-T |

### Key Algorithmic Advances

| Paper | Citation | URL |
|---|---|---|
| Glorot/Xavier Initialization (2010) | Glorot, X. & Bengio, Y. "Understanding the difficulty of training deep feedforward neural networks." *AISTATS*, PMLR 9, 249–256. | https://proceedings.mlr.press/v9/glorot10a.html |
| ReLU for Deep Networks (2011) | Glorot, X., Bordes, A., & Bengio, Y. "Deep sparse rectifier neural networks." *AISTATS*, PMLR 15, 315–323. | http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf |
| AlexNet / ReLU at Scale (2012) | Krizhevsky, A., Sutskever, I., & Hinton, G. E. "ImageNet classification with deep convolutional neural networks." *NeurIPS*, 25. | https://dl.acm.org/doi/10.1145/3065386 |
| Dropout (2014) | Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. "Dropout: A simple way to prevent neural networks from overfitting." *JMLR*, 15(1), 1929–1958. | https://jmlr.org/papers/v15/srivastava14a.html |
| Adam Optimizer (2014) | Kingma, D. P. & Ba, J. "Adam: A method for stochastic optimization." *ICLR 2015*. | https://arxiv.org/abs/1412.6980 |
| Batch Normalization (2015) | Ioffe, S. & Szegedy, C. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML*. | https://arxiv.org/abs/1502.03167 |
| He/Kaiming Initialization (2015) | He, K., Zhang, X., Ren, S., & Sun, J. "Delving deep into rectifiers." *ICCV*. | https://arxiv.org/abs/1502.01852 |

### The Critical Tabular Data Benchmark

| Paper | Citation | URL |
|---|---|---|
| Trees vs NNs on Tabular Data (2022) | Grinsztajn, L., Oyallon, E., & Varoquaux, G. "Why do tree-based models still outperform deep learning on typical tabular data?" *NeurIPS 2022*. arXiv:2207.08815. | https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html |
| Revisiting DL for Tabular (FT-Transformer, 2021) | Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. "Revisiting deep learning models for tabular data." *NeurIPS 2021*. arXiv:2106.11959. | https://arxiv.org/abs/2106.11959 |
| TabNet (2019) | Arik, S. Ö. & Pfister, T. "TabNet: Attentive interpretable tabular learning." *AAAI 2021*. arXiv:1908.07442. | https://arxiv.org/abs/1908.07442 |

### Explainability

| Paper | Citation | URL |
|---|---|---|
| SHAP (2017) | Lundberg, S. M. & Lee, S. I. "A unified approach to interpreting model predictions." *NeurIPS*, 30. | https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html |

### Books (Essential Chapters)

| Book | Chapters | Notes |
|---|---|---|
| **Goodfellow, Bengio, Courville — *Deep Learning* (2016)** | Chapters 6 (Deep Feedforward Networks), 7 (Regularization), 8 (Optimization) | The standard textbook; free online at deeplearningbook.org |
| **Bishop — *Pattern Recognition and Machine Learning* (2006)** | Chapter 5 (Neural Networks) | Dense, rigorous; covers backprop derivation cleanly |
| **Bishop — *Deep Learning: Foundations and Concepts* (2024)** | Chapters 6–10 | Updated treatment; replaces PRML for modern DL |
| **Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (2009)** | Chapter 11 (Neural Networks) | Good for connecting NNs to statistical learning theory |
| **Géron — *Hands-On Machine Learning* 3rd ed. (2022)** | Chapters 10–12 | Practical, code-heavy sklearn + Keras treatment |

### Documentation (Verified)

| Resource | URL |
|---|---|
| sklearn MLPClassifier docs (1.5) | https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPClassifier.html |
| sklearn MLPRegressor docs (1.5) | https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPRegressor.html |
| sklearn Neural Networks User Guide | https://scikit-learn.org/1.5/modules/neural_networks_supervised.html |
| PyTorch `torch.nn` docs | https://pytorch.org/docs/stable/nn.html |
| Keras layers API | https://keras.io/api/layers/ |
| SHAP KernelExplainer API | https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html |
| SHAP GradientExplainer (PyTorch) | https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html |
| Optuna docs | https://optuna.readthedocs.io/ |

### Online Resources

| Resource | URL | Notes |
|---|---|---|
| *Deep Learning* book (free online) | https://www.deeplearningbook.org | Goodfellow et al.; chapter 6 is the definitive MLP chapter |
| distill.pub — "Why Momentum Really Works" | https://distill.pub/2017/momentum/ | Outstanding visual explanation of momentum and optimization |
| Andrej Karpathy — "The Unreasonable Effectiveness of RNNs" | http://karpathy.github.io/2015/05/21/rnn-effectiveness/ | Classic blog post on neural network intuition |
| fast.ai Practical Deep Learning | https://course.fast.ai | Top-down, code-first deep learning course |
| sklearn MLP source code | https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neural_network/_multilayer_perceptron.py | Weight initialization details; verified against dossier |

### Verified API Flags

The following items should be verified against sklearn 1.5 documentation before use:

1. **`validation_scores_` vs `best_validation_score_`:** The attribute storing per-epoch
   validation scores (plural) when `early_stopping=True`. The dossier confirms `validation_scores_`
   (plural) — verify against sklearn 1.5 source if behavior is unexpected.

2. **`max_fun` in `MLPClassifier`:** Confirmed present in both `MLPClassifier` and
   `MLPRegressor` as of sklearn 0.22+. Verify in sklearn 1.5 docs.

3. **`shap.KernelExplainer` vs `shap.Explainer` for sklearn MLP:** In SHAP 0.46.x, the
   unified `shap.Explainer(model.predict_proba, background)` may auto-detect and dispatch to
   `KernelExplainer`. Either API works; the explicit `KernelExplainer` call is more transparent
   about what algorithm is being used.

4. **`partial_fit` availability:** `MLPClassifier.partial_fit` requires passing `classes` on
   the first call: `clf.partial_fit(X_batch, y_batch, classes=np.unique(y))`. Only available
   with `solver='sgd'` or `solver='adam'`; not with `solver='lbfgs'`.






