# Support Vector Machines: The Complete Masterclass

> **Why this algorithm matters:** Support Vector Machines are the algorithm that proved machine
> learning has a principled mathematical foundation. When Vapnik and Cortes published their
> soft-margin SVM in 1995, they didn't just give the world a new classifier — they demonstrated
> that you could control a model's generalization error via a geometric quantity (the margin)
> derived directly from the training data's structure, with theoretical bounds grounded in
> Vapnik-Chervonenkis (VC) theory. The kernel trick — introduced in the very first SVM paper in
> 1992 — remains one of the most elegant ideas in all of machine learning: you can implicitly
> operate in an infinite-dimensional feature space by computing a simple inner product. SVMs
> dominated benchmark competitions through the early 2000s, were the state of the art for text
> classification and bioinformatics for over a decade, and the fundamental ideas they introduced —
> margin maximization, duality, kernel methods, sparsity of solutions — remain active ingredients
> in modern deep learning theory. On the right problem (small-to-medium $n$, high $p$, kernel
> structure), a well-tuned SVM still beats gradient boosting and shallow neural networks. More
> importantly, understanding SVMs deeply makes you a better machine learning practitioner across
> every algorithm you'll ever use.

---

## §0 — Python Ecosystem & Package Guide

Every SVM implementation in Python traces its lineage to two C libraries: **LIBSVM** (Chang & Lin,
NTU Taiwan) for kernel SVMs, and **LIBLINEAR** (Fan et al.) for large-scale linear SVMs. Everything
else is a wrapper, extension, or GPU acceleration of these two engines. Knowing this prevents
confusion when you see identical results from seemingly different packages.

### Complete Package Table

| Package | Import | Version (mid-2026) | Backend | Key Strengths | License |
|---|---|---|---|---|---|
| `scikit-learn` (SVC/SVR) | `from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM` | ~1.5.x | LIBSVM | Full kernel SVM suite; sklearn Pipeline compatible; probability calibration; OvO multiclass | BSD-3 |
| `scikit-learn` (LinearSVC) | `from sklearn.svm import LinearSVC, LinearSVR` | ~1.5.x | LIBLINEAR | Near-$O(n)$ for linear SVMs; L1/L2 penalties; handles millions of samples | BSD-3 |
| `ThunderSVM` | `from thundersvm import SVC, SVR` | ~0.3.x | CUDA/CPU | ~100× speedup over LIBSVM on GPU; identical API to sklearn | Apache-2 |
| `cuML` (RAPIDS) | `from cuml.svm import SVC, SVR` | ~24.x | CUDA | GPU-native; integrates with cuDF/CuPy; RAPIDS ecosystem | Apache-2 |
| `sklearn.kernel_approximation` | `from sklearn.kernel_approximation import Nystroem, RBFSampler` | ~1.5.x | NumPy | Scalable approximate kernel maps; enables LinearSVC on large $n$ with nonlinear decision boundaries | BSD-3 |
| `sklearn.calibration` | `from sklearn.calibration import CalibratedClassifierCV` | ~1.5.x | NumPy | Platt scaling or isotonic calibration for probability outputs from any SVM | BSD-3 |
| `libsvm` (Python bindings) | `import svmutil` | ~3.3x | LIBSVM (native) | Direct LIBSVM Python bindings; lower-level access | BSD-3 |
| `shap` | `import shap` | ~0.46.x | NumPy | KernelExplainer for model-agnostic SHAP; LinearExplainer for LinearSVC | MIT |
| `optuna` | `import optuna` | ~3.x | — | Efficient Bayesian HPO for the C/gamma grid | MIT |

### Top Picks — Recommendation Table

| Package | Best For | Modern/Legacy | Notes |
|---|---|---|---|
| **`sklearn.SVC`** | $n < 10{,}000$, nonlinear problems, kernels, benchmark tasks | Modern | First choice for kernel SVMs; LIBSVM backed; slow at large scale |
| **`sklearn.LinearSVC`** | $n > 10{,}000$, text classification, sparse features, linear problems | Modern | 100–1000× faster than SVC(kernel='linear'); first choice for large linear problems |
| **`Nystroem + LinearSVC`** | $10{,}000 < n < 500{,}000$, approximate nonlinear SVM at scale | Modern | Best of both worlds: nonlinear kernel + linear-speed training; interpretable `coef_` |
| **`ThunderSVM`** | GPU workstations, $n > 50{,}000$, need RBF/poly kernel | Modern | Drop-in sklearn replacement; requires CUDA — verify maintenance status before production use |
| **`sklearn.NuSVC`** | When you want interpretable fraction of SVs / margin errors | Modern | Use `nu` instead of `C`; otherwise same LIBSVM backend |

### The Same Fit Across Top Packages

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Shared setup ─────────────────────────────────────────────────────────────
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Option 1: sklearn SVC (kernel RBF) — small n, nonlinear ─────────────────
from sklearn.svm import SVC

pipe_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42))
])
pipe_svc.fit(X_train, y_train)
print(f"SVC accuracy:       {pipe_svc.score(X_test, y_test):.4f}")

# ── Option 2: sklearn LinearSVC — large n, linear boundary ──────────────────
from sklearn.svm import LinearSVC

pipe_lsvc = Pipeline([
    ('scaler', StandardScaler()),
    ('lsvc', LinearSVC(C=1.0, max_iter=5000, random_state=42))
])
pipe_lsvc.fit(X_train, y_train)
print(f"LinearSVC accuracy: {pipe_lsvc.score(X_test, y_test):.4f}")

# ── Option 3: Nystroem + LinearSVC — scalable nonlinear ─────────────────────
from sklearn.kernel_approximation import Nystroem

pipe_nys = Pipeline([
    ('scaler', StandardScaler()),
    ('nystroem', Nystroem(kernel='rbf', gamma=0.01, n_components=300, random_state=42)),
    ('lsvc', LinearSVC(C=1.0, max_iter=5000, random_state=42))
])
pipe_nys.fit(X_train, y_train)
print(f"Nystroem+LSVC acc:  {pipe_nys.score(X_test, y_test):.4f}")

# ── Option 4: ThunderSVM (GPU) — identical sklearn API ──────────────────────
# from thundersvm import SVC as ThunderSVC  # requires CUDA
# pipe_thunder = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svc', ThunderSVC(kernel='rbf', C=10.0, gamma='scale'))
# ])
# pipe_thunder.fit(X_train, y_train)
# print(f"ThunderSVM accuracy: {pipe_thunder.score(X_test, y_test):.4f}")

# ── Option 5: NuSVC — interpretable nu parameterization ─────────────────────
from sklearn.svm import NuSVC

pipe_nusvc = Pipeline([
    ('scaler', StandardScaler()),
    ('nusvc', NuSVC(nu=0.1, kernel='rbf', gamma='scale'))
])
pipe_nusvc.fit(X_train, y_train)
print(f"NuSVC accuracy:     {pipe_nusvc.score(X_test, y_test):.4f}")
# nu=0.1 → at most 10% of samples will be margin errors
```

> 🔧 **In practice:** For a new project, start with `SVC(kernel='rbf')` in a `Pipeline` with
> `StandardScaler`. If $n > 10{,}000$, switch to `LinearSVC` or `Nystroem + LinearSVC` immediately.
> Never run a kernel SVM without scaling first — it is the single most common mistake that
> causes mysteriously poor results.

> ⚠️ **Pitfall:** `SVC(kernel='linear')` and `LinearSVC` are NOT the same. They use different
> algorithms (LIBSVM dual vs. LIBLINEAR primal/dual), different defaults (`probability` support,
> multiclass strategy, loss function), and have very different scalability. On large datasets,
> `LinearSVC` is 100–10,000× faster than `SVC(kernel='linear')`. Always prefer `LinearSVC` for
> linear problems.

---

## §1 — The Origin: The Paper That Started It All

### Before SVMs: The "Architectural" Problem

To appreciate why SVMs were revolutionary, you need to understand what ML researchers were
struggling with in the early 1990s. Neural networks were ascendant — the backprop renaissance of
the late 1980s had produced remarkable results. But neural networks had a fundamental practical
problem: **architectural choices were made by hand**. How many hidden layers? How many neurons per
layer? How long to train? All of these were design decisions that controlled the model's capacity,
and practitioners tuned them through trial and error, with no principled guidance on how to
prevent overfitting.

The standard theoretical tool for understanding generalization was Vapnik-Chervonenkis (VC) theory,
which says the generalization error is bounded by:

$$\text{Err}_{\text{test}} \leq \text{Err}_{\text{train}} + O\!\left(\sqrt{\frac{h}{n}}\right)$$

where $h$ is the VC dimension (a measure of model complexity) and $n$ is the number of training
samples. For neural networks, $h$ grows with the number of weights — but the relationship was
complex and hard to use practically.

The key insight Boser, Guyon, and Vapnik brought was: **what if you chose the architecture by
maximizing the margin?** The margin directly controls the VC dimension of the classifier, meaning
maximum-margin classifiers have the lowest generalization bound, regardless of the ambient
dimensionality.

> 📊 **Benchmark:** In their 1992 paper, Boser et al. demonstrated on handwritten digit
> recognition benchmarks that their optimal margin classifier outperformed contemporary neural
> networks, using polynomial kernels of degree up to 4. This result, replicated and extended
> by Cortes & Vapnik (1995) on a larger digit dataset, contributed significantly to SVM adoption.

### Paper 1: The True Origin — Hard-Margin SVM + The Kernel Trick

> 📜 **Citation:**
> Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992). A training algorithm for optimal margin
> classifiers. *Proceedings of the Fifth Annual Workshop on Computational Learning Theory
> (COLT '92)*, Pittsburgh, PA, pp. 144–152. ACM.
> DOI: https://doi.org/10.1145/130385.130401

**The problem:** Given training data $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ with
$y_i \in \{-1, +1\}$ and $\mathbf{x}_i \in \mathbb{R}^p$, find a linear decision boundary
$\mathbf{w}^\top \mathbf{x} + b = 0$ that classifies all training points correctly **and** has the
largest possible margin between the two classes.

**The core insight:** If the data is linearly separable, there are infinitely many valid
hyperplanes. The unique best one — the one with the largest margin — has the smallest VC dimension
among all separating hyperplanes, and therefore achieves the smallest theoretical generalization
error bound.

**Setting up the geometry.** A hyperplane $\mathbf{w}^\top \mathbf{x} + b = 0$ divides space into
two half-spaces. For a correctly classified point $\mathbf{x}_i$ with label $y_i$:

$$y_i (\mathbf{w}^\top \mathbf{x}_i + b) > 0$$

The **functional margin** of point $i$ is $y_i (\mathbf{w}^\top \mathbf{x}_i + b)$. The
**geometric margin** — the Euclidean distance from point $\mathbf{x}_i$ to the hyperplane — is:

$$\frac{y_i (\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

We want every point to be at least distance $\gamma$ from the hyperplane:
$y_i (\mathbf{w}^\top \mathbf{x}_i + b) \geq \gamma \|\mathbf{w}\|$ for all $i$.

By rescaling $\mathbf{w}$ and $b$ (which doesn't change the hyperplane), we can always set the
functional margin of the closest points (the support vectors) to exactly 1:
$y_i (\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$ for all $i$. Then the geometric margin is
$1/\|\mathbf{w}\|$ and the total margin width (between the two class boundaries) is $2/\|\mathbf{w}\|$.

**The hard-margin primal problem:**

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i (\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i = 1, \ldots, n$$

Minimizing $\|\mathbf{w}\|^2$ is equivalent to maximizing the margin $2/\|\mathbf{w}\|$. The
$\frac{1}{2}$ is a convenience factor that makes the derivative cleaner. This is a convex
quadratic program with $p+1$ variables and $n$ inequality constraints.

**Deriving the dual.** Introducing Lagrange multipliers $\alpha_i \geq 0$ for each constraint, the
Lagrangian is:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i \left[ y_i (\mathbf{w}^\top \mathbf{x}_i + b) - 1 \right]$$

At the saddle point, partial derivatives vanish:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \implies \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i \tag{1}$$

$$\frac{\partial \mathcal{L}}{\partial b} = 0 \implies \sum_{i=1}^n \alpha_i y_i = 0 \tag{2}$$

Substituting (1) back and simplifying, the primal objective becomes:

$$\frac{1}{2}\|\mathbf{w}\|^2 = \frac{1}{2}\left(\sum_i \alpha_i y_i \mathbf{x}_i\right)^\top\!\!\left(\sum_j \alpha_j y_j \mathbf{x}_j\right) = \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

The Lagrangian reduces to:

$$\mathcal{L} = \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

The **hard-margin dual** is therefore:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

$$\text{subject to: } \alpha_i \geq 0 \; \forall i, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

This dual has $n$ variables (one per training sample) and two simple constraints, regardless of $p$.
For high-dimensional problems ($p \gg n$), the dual is smaller. For structured problems where you
need kernels, the dual is essential.

**KKT complementary slackness:** The KKT conditions at the optimum require:

$$\alpha_i \left[ y_i (\mathbf{w}^\top \mathbf{x}_i + b) - 1 \right] = 0 \quad \forall i$$

This means either $\alpha_i = 0$ (the constraint is inactive — the point is far from the margin,
irrelevant to the solution) or $y_i (\mathbf{w}^\top \mathbf{x}_i + b) = 1$ (the point lies
exactly on the margin boundary). Points with $\alpha_i > 0$ are the **support vectors** — the
only points that matter for the solution.

**The sparsity revelation:** Most $\alpha_i$ will be zero. The weight vector (from eq. 1) is:

$$\mathbf{w} = \sum_{i \in \text{SV}} \alpha_i y_i \mathbf{x}_i$$

And the prediction at any new point $\mathbf{x}$ is:

$$f(\mathbf{x}) = \text{sign}\!\left(\sum_{i \in \text{SV}} \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b\right)$$

The solution depends *only* on the support vectors. You could throw away every other training
point and the model would be identical.

**The kernel trick — the crown jewel.** Notice that both the dual objective and the prediction
formula depend on training data only through dot products $\mathbf{x}_i^\top \mathbf{x}_j$.
Boser, Guyon, and Vapnik recognized: what if instead of working in the original space
$\mathbb{R}^p$, we first map data to a higher-dimensional feature space via
$\phi: \mathbb{R}^p \to \mathcal{H}$? If $\mathcal{H}$ is nonlinear, the optimal hyperplane there
becomes a nonlinear boundary in the original space.

The computation would require computing $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$ for every
pair — potentially infinite-dimensional. But Mercer's theorem guarantees that any valid positive
semi-definite kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$ corresponds to an inner product
in *some* feature space:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$$

So we simply replace every $\mathbf{x}_i^\top \mathbf{x}_j$ in the dual with $K(\mathbf{x}_i, \mathbf{x}_j)$.
The prediction becomes:

$$f(\mathbf{x}) = \text{sign}\!\left(\sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

We never need to compute $\phi(\mathbf{x})$ explicitly — just the kernel function. The RBF kernel
$K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$ implicitly corresponds
to an infinite-dimensional feature space, yet every computation requires only a single scalar.

> 💡 **Intuition:** The kernel trick is like buying a house with a mortgage — you get access to
> resources (infinite-dimensional feature maps) that you could never afford to compute directly
> by paying only the "interest" (kernel evaluations). You never own the feature map outright, but
> you enjoy all its benefits.

### Paper 2: The Canonical SVM — Soft Margins (Cortes & Vapnik 1995)

> 📜 **Citation:**
> Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
> DOI: https://doi.org/10.1007/BF00994018

**The problem with hard margins:** Real data is noisy. Classes overlap. Even with a nonlinear
kernel, the hard-margin SVM fails — it demands perfect separability in feature space. With even a
single outlier that violates the margin, the problem becomes infeasible.

**The fix — slack variables.** Cortes and Vapnik introduced non-negative slack variables
$\xi_i \geq 0$ that allow individual points to violate the margin:

$$y_i (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

When $\xi_i = 0$: point is correctly classified and outside the margin. When $0 < \xi_i \leq 1$:
point is correctly classified but inside the margin (a margin violation). When $\xi_i > 1$: point
is misclassified. The total penalty for violations is $\sum_i \xi_i$.

**The soft-margin primal (C-SVM):**

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$

$$\text{s.t. } y_i (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 \quad \forall i$$

The parameter $C > 0$ controls the trade-off: large $C$ means violations are expensive, forcing
the model toward the hard-margin solution (low bias, high variance); small $C$ means violations
are cheap, allowing a wider margin with more misclassified points (high bias, low variance).

**The soft-margin dual:**

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

$$\text{s.t. } 0 \leq \alpha_i \leq C \quad \forall i, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

The only change from the hard-margin dual is the **box constraint** $\alpha_i \leq C$ (instead of
just $\alpha_i \geq 0$). The upper bound $C$ limits the influence of any single training point —
if $C$ is small, even support vectors can't have large $\alpha_i$ values, forcing the margin to
be wide.

**Three types of points:**
- $\alpha_i = 0$: non-support vectors; correctly classified, outside the margin; ignored
- $0 < \alpha_i < C$: support vectors on the margin boundary ($\xi_i = 0$, exactly at the margin)
- $\alpha_i = C$: support vectors that violate the margin ($\xi_i > 0$); "bounded support vectors"

**Historical note:** This paper was submitted in 1992 (the same year as Boser et al.) and took
three years to appear in print — partly because reviewers were skeptical that a quadratic program
could be practically solved for real datasets. The name "Support-Vector Networks" reflects the
authors' view that SVMs form a network-like structure where the support vectors are the hidden
nodes.

> 🔬 **Deep dive:** The primal formulation uses the **hinge loss** $\ell(y, f(\mathbf{x})) = \max(0, 1 - y \cdot f(\mathbf{x}))$. The slack variable $\xi_i = \max(0, 1 - y_i f(\mathbf{x}_i))$ is exactly the hinge loss evaluated at each training point. So the soft-margin SVM objective is:
> $$\frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \max(0, 1 - y_i f(\mathbf{x}_i))$$
> This is $L_2$ regularization + hinge loss, a fact that connects SVMs directly to regularized
> risk minimization theory and explains why the C-SVM and ridge-regularized logistic regression
> often give similar results (they differ only in hinge vs. log loss, and implicitly in margin
> geometry).

---

## §2 — The Algorithm, Deeply Explained

### The Mathematical Foundations

Let's build the complete picture. You have $n$ training pairs $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$
with $\mathbf{x}_i \in \mathbb{R}^p$ and $y_i \in \{-1, +1\}$. You apply a feature map
$\phi: \mathbb{R}^p \to \mathcal{H}$ (which could be the identity if $K$ is linear) and seek to
find $\mathbf{w} \in \mathcal{H}$ and $b \in \mathbb{R}$ that solve the soft-margin primal.

**What the algorithm actually optimizes.** Rephrasing with hinge loss, the SVM solves:

$$\min_{\mathbf{w}, b} \underbrace{\frac{1}{2}\|\mathbf{w}\|^2}_{\text{regularizer}} + C \sum_{i=1}^n \underbrace{\max(0, 1 - y_i (\mathbf{w}^\top \phi(\mathbf{x}_i) + b))}_{\text{hinge loss on sample } i}$$

This is a trade-off between **complexity** (small $\|\mathbf{w}\|^2$ = wide margin) and
**empirical risk** (small hinge loss = few misclassifications). $C$ weights this trade-off.
Note: in sklearn convention, $C$ multiplies the loss term — large $C$ means more penalty for
misclassification, so you fit the data more tightly.

**Why convexity matters.** The hinge loss $\max(0, 1 - t)$ is convex in $t$. The regularizer
$\|\mathbf{w}\|^2$ is strictly convex. Their sum is strictly convex in $(\mathbf{w}, b)$, which
guarantees a **unique global minimum**. There are no local minima, no saddle points that trap
optimization, no sensitivity to random initialization. Given the same training data and
hyperparameters, SVM will always converge to the same solution.

**The dual problem in full.** Through the Lagrangian derivation shown in §1, the equivalent
dual is:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

$$\text{s.t.} \quad 0 \leq \alpha_i \leq C \; \forall i, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

The matrix $Q_{ij} = y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$ is the **kernel matrix** (also called
the Gram matrix), scaled by labels. When $K$ is a valid Mercer kernel, $Q$ is positive
semi-definite, making the dual a **convex QP** (the objective is concave, the feasible set is
convex).

**Strong duality.** By Slater's condition (the primal has a strictly feasible point for $C > 0$),
strong duality holds: the primal and dual optimal values are equal. This means you can solve
either form and get the same optimal $(\mathbf{w}^*, b^*)$.

**Recovering the primal solution from the dual:**

From $\mathbf{w} = \sum_i \alpha_i y_i \phi(\mathbf{x}_i)$, and for any support vector on the
margin (where $0 < \alpha_i < C$, so $\xi_i = 0$ and $y_i f(\mathbf{x}_i) = 1$):

$$b = y_i - \sum_{j \in \text{SV}} \alpha_j y_j K(\mathbf{x}_j, \mathbf{x}_i)$$

In practice, $b$ is computed as the average over all margin support vectors for numerical stability.

**The prediction formula:**

$$f(\mathbf{x}) = \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$$

$$\hat{y}(\mathbf{x}) = \text{sign}(f(\mathbf{x}))$$

In sklearn, `model.decision_function(X)` returns $f(\mathbf{x})$ and `model.predict(X)` returns
$\hat{y}$.

### Multiple Intuitions for Understanding SVMs

**Intuition 1: The widest street.** Imagine two classes of data as red and blue houses on either
side of a road. A linear SVM finds the widest possible road that separates them, with the edges
of the road touching the closest houses on each side (the support vectors). When data is noisy
and houses occasionally appear on the wrong side, the soft-margin SVM allows a small number of
transgressors, penalized by $C$.

> 💡 **Intuition:** The support vectors are the data points that "define" the decision boundary.
> Remove any non-support-vector point and the model is unchanged. Move a support vector and the
> entire boundary moves. This extreme sensitivity to support vectors is both a strength (the
> model uses only the most informative points) and a weakness (outliers that become support
> vectors can dramatically perturb the solution).

**Intuition 2: The kernel as a similarity function.** The RBF kernel
$K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$ is 1 when
$\mathbf{x} = \mathbf{x}'$ and decays to 0 as the points move apart. The parameter $\gamma$
controls how quickly this decay happens — large $\gamma$ means only very close points are
considered "similar". The prediction $f(\mathbf{x}) = \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x})$ is essentially a weighted vote, where each support vector contributes a "yes" or "no" vote scaled by its similarity to the query point.

**Intuition 3: The feature space view.** The polynomial kernel
$K(\mathbf{x}, \mathbf{x}') = (\gamma\langle\mathbf{x}, \mathbf{x}'\rangle + r)^d$ implicitly
maps to a space containing all monomials of degree up to $d$. For 2D input and $d=2$, the feature
map is roughly $\phi(x_1, x_2) = (x_1^2, \sqrt{2}x_1 x_2, x_2^2, \sqrt{2\gamma}x_1, \sqrt{2\gamma}x_2, 1)$.
A linear boundary in this 6D space corresponds to a conic section (ellipse, hyperbola, parabola)
in the original 2D space. The kernel trick lets you find this nonlinear boundary without ever
computing the 6D coordinates.

**Intuition 4: Regularization and the bias-variance tradeoff.** The margin $2/\|\mathbf{w}\|$ is
inversely related to the norm of the weight vector. Forcing $\|\mathbf{w}\|$ to be small (via
the regularization term) forces the margin to be wide, which reduces the model's sensitivity to
small perturbations in the training data — this is exactly what reduces variance. The parameter
$C$ mediates this: small $C$ imposes strong regularization (wide margin, high bias, low variance);
large $C$ relaxes regularization (narrow margin, low bias, high variance).

### How Training Works — The SMO Algorithm

Solving the SVM dual is a Quadratic Programming problem. The naive QP solver requires computing
and storing the full $n \times n$ kernel matrix $Q$ (100 GB for $n = 10^5$) and running an
$O(n^3)$ algorithm. This is infeasible for any serious dataset.

The breakthrough was Platt's **Sequential Minimal Optimization (SMO)** algorithm (1998), which
is what sklearn's LIBSVM uses internally.

> 📜 **Citation:**
> Platt, J. C. (1998). Sequential minimal optimization: A fast algorithm for training support
> vector machines. *Microsoft Research Technical Report MSR-TR-98-14*.
> URL: https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/

**SMO's insight:** The smallest possible QP sub-problem that can be solved while respecting the
equality constraint $\sum_i \alpha_i y_i = 0$ is a **2-variable** problem. With two variables
$\alpha_i$ and $\alpha_j$, the constraint $\alpha_i y_i + \alpha_j y_j = \text{const}$ allows
you to express $\alpha_i$ as a function of $\alpha_j$, reducing to a 1D constrained optimization
with a closed-form solution.

**SMO algorithm sketch:**

```
Initialize: α = 0, b = 0, error_cache = {E_i = -y_i for all i}

Repeat until convergence (KKT violations < tol):
    1. Select two variables α_i, α_j to optimize:
       - Outer loop: cycle through all samples, pick α_i violating KKT conditions
       - Inner loop: pick α_j maximizing |E_i - E_j| (maximum step heuristic)

    2. Solve the 2-variable QP analytically:
       η = K(x_i,x_i) + K(x_j,x_j) - 2·K(x_i,x_j)    # curvature; must be > 0
       α_j_new_unc = α_j + y_j·(E_i - E_j) / η          # unconstrained optimum

       # Clip to box constraints [L, H]:
       if y_i == y_j:  L = max(0, α_i+α_j-C),   H = min(C, α_i+α_j)
       else:           L = max(0, α_j-α_i),      H = min(C, C+α_j-α_i)
       α_j_new = clip(α_j_new_unc, L, H)
       α_i_new = α_i + y_i·y_j·(α_j - α_j_new)

    3. Update bias b using KKT conditions on the two updated variables
    4. Update error cache E_i = f(x_i) - y_i for modified points
```

**Why SMO is fast:** It avoids matrix factorization entirely. The kernel cache stores the most
recently computed $K(\mathbf{x}_i, \mathbf{x}_j)$ values (default 200 MB in sklearn), reducing
redundant kernel evaluations. On sparse linear data, SMO converges in $O(n)$; on dense nonlinear
problems, empirically $O(n^2)$ to $O(n^3)$.

> 🔧 **In practice:** Increase `cache_size` from the default 200 MB to 2000 MB (or more) for
> large datasets. The kernel cache is the single biggest factor in practical training speed after
> choosing the right kernel.

### Computational Complexity

| Operation | LIBSVM (SVC/SVR) | LIBLINEAR (LinearSVC) |
|---|---|---|
| Training | $O(n^2 p)$ to $O(n^3 p)$ | $O(n p)$ approx. |
| Memory | $O(n_{SV}^2)$ kernel cache + $O(n)$ | $O(p)$ for `coef_` |
| Inference | $O(n_{SV} \cdot p)$ per sample | $O(p)$ per sample |
| Practical $n$ limit | ~10,000–50,000 | Millions |

**Inference bottleneck:** Unlike tree-based models where inference is $O(\text{depth})$, kernel
SVM inference requires $n_{SV}$ kernel evaluations per test point. If your SVM has 8,000 support
vectors and $p = 100$ features, each prediction requires 800,000 multiply-adds. This matters
significantly in real-time serving.

> ⚠️ **Pitfall:** Monitor `model.n_support_` after training. If more than 50% of your training
> points are support vectors, something is wrong — either $C$ is too small (too many margin
> violations allowed), your kernel is poorly tuned, or the features need scaling. A high support
> vector ratio means slow inference and often indicates an underfit model.

### Implicit Assumptions

SVMs make fewer explicit distributional assumptions than many other algorithms, but they have
strong implicit requirements:

1. **Feature scale matters.** The RBF kernel uses $\|\mathbf{x} - \mathbf{x}'\|^2$, which mixes
   all features equally. An unscaled feature with range [0, 10,000] will dominate the distance
   computation and effectively make all other features irrelevant. This is the most common SVM
   mistake in practice.

2. **The kernel must be appropriate.** Using a linear kernel when the true boundary is highly
   nonlinear will underfit. Using an RBF kernel with poorly tuned $\gamma$ will either overfit
   (large $\gamma$) or underfit (small $\gamma$). The kernel is a structural assumption about
   the problem geometry.

3. **The training set is representative.** SVMs are sensitive to outliers — extreme points can
   become bounded support vectors ($\alpha_i = C$) that drag the boundary toward them.

4. **The number of features $p$ is fixed.** Unlike online learners, SVMs don't gracefully handle
   streaming data with new features at test time.

5. **Classes are balanced for C-SVM.** With imbalanced classes, the majority class can dominate
   the margin. Use `class_weight='balanced'` to correct this.

### The Theoretical Foundation: VC Theory and Margin Bounds

The deepest reason SVMs work — the reason the maximum-margin principle is not just a heuristic
but a theoretically grounded strategy — comes from Vapnik-Chervonenkis (VC) theory. Sit with
this argument for a moment: it's one of the most beautiful in machine learning.

**The VC dimension of linear classifiers.** Consider the class of linear classifiers
$\mathcal{H} = \{\text{sign}(\mathbf{w}^\top\mathbf{x} + b) : \mathbf{w} \in \mathbb{R}^p, b \in \mathbb{R}\}$.
The VC dimension of this class is $h = p + 1$ — it can shatter any $p + 1$ points in general
position. The VC bound on generalization error says:

$$P(\text{err}_{\text{test}}) \leq \text{err}_{\text{train}} + O\!\left(\sqrt{\frac{h(\ln(2n/h) + 1) - \ln(\delta/4)}{n}}\right)$$

with probability $1 - \delta$. This bound grows with the VC dimension $h$. For $p = 100$ features
with $n = 200$ samples, the VC bound is vacuous — it says the test error could be nearly 1.

**The margin-based bound.** But here's the key insight Vapnik formalized: for linear classifiers
constrained to have a large margin $\gamma = 1/\|\mathbf{w}\|$, the effective VC dimension is
not $p + 1$ but something much smaller. Specifically, if all data lies within a ball of radius $R$:

$$h \leq \left\lfloor \frac{R^2}{\gamma^2} \right\rfloor + 1 = \left\lfloor R^2 \|\mathbf{w}\|^2 \right\rfloor + 1$$

This bound is **independent of $p$**. A large-margin classifier in 1,000,000 dimensions has the
same VC dimension as one in 2 dimensions — provided $R^2\|\mathbf{w}\|^2$ is the same. This is
the theoretical reason why maximum-margin SVMs can generalize even when $p \gg n$.

> 🔬 **Deep dive:** The quantity $R^2\|\mathbf{w}\|^2$ has a direct geometric interpretation:
> it's the ratio of the data's scale to the margin squared. A wide margin (small $\|\mathbf{w}\|$)
> relative to the data's extent (large $R$) means low VC dimension, which means low capacity,
> which means low variance, which means good generalization. Minimizing $\|\mathbf{w}\|^2$ in
> the SVM objective is not an arbitrary regularization choice — it's directly minimizing the
> VC-theoretic capacity of the classifier.

**The connection to structural risk minimization.** Vapnik's structural risk minimization (SRM)
principle says: choose the hypothesis class that minimizes the bound on test error, not just
training error. The SVM implements SRM by selecting the classifier with the smallest capacity
(widest margin) among all classifiers that achieve zero (or low) training error.

**Kernel VC theory.** In kernel space, the margin bound becomes:

$$h \leq \left\lfloor R_\phi^2 \|\mathbf{w}_\phi\|_\mathcal{H}^2 \right\rfloor + 1$$

where $R_\phi$ is the radius in feature space and $\|\mathbf{w}_\phi\|_\mathcal{H}$ is the RKHS
norm. For the RBF kernel, $\|\mathbf{w}_\phi\|_\mathcal{H}^2 = \boldsymbol{\alpha}^\top K \boldsymbol{\alpha}$
(the kernel matrix weighted by dual variables). Minimizing the SVM objective also minimizes this
RKHS norm, giving the minimum-norm interpolant in feature space — a concept that connects SVMs
to Gaussian processes and modern regularization theory.

### The Representer Theorem and RKHS

The mathematical setting that makes SVMs rigorous is the theory of **Reproducing Kernel Hilbert
Spaces (RKHS)**. Every kernel function $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$
satisfying Mercer's condition defines a unique Hilbert space $\mathcal{H}_K$ of functions on
$\mathcal{X}$, equipped with an inner product such that $K(\cdot, \mathbf{x})$ is an element of
$\mathcal{H}_K$ for each fixed $\mathbf{x}$, and the **reproducing property** holds:

$$f(\mathbf{x}) = \langle f, K(\cdot, \mathbf{x}) \rangle_{\mathcal{H}_K} \quad \forall f \in \mathcal{H}_K$$

The **Representer Theorem** (Schölkopf et al., 2001) states: for any regularized empirical risk
minimization problem of the form:

$$\min_{f \in \mathcal{H}_K} \sum_{i=1}^n L(y_i, f(\mathbf{x}_i)) + \Omega(\|f\|_{\mathcal{H}_K})$$

with $\Omega$ strictly increasing, the solution has the form:

$$f^*(\mathbf{x}) = \sum_{i=1}^n c_i K(\mathbf{x}_i, \mathbf{x})$$

for some coefficients $c_i \in \mathbb{R}$. This is profound: even though we're optimizing over
an infinite-dimensional function space $\mathcal{H}_K$, the optimal solution lives in the
$n$-dimensional subspace spanned by $\{K(\mathbf{x}_i, \cdot)\}_{i=1}^n$. For SVMs, the
Representer Theorem directly gives us the form $f^*(\mathbf{x}) = \sum_i \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$.

> 💡 **Intuition:** The Representer Theorem is why the "dual formulation" is not just a
> computational convenience but a fundamental structural result. The optimal SVM prediction
> function is always a linear combination of kernel evaluations at training points — you can't
> do better by considering other functional forms. The sparsity (only support vectors have
> $\alpha_i \neq 0$) is a consequence of the specific hinge loss; other loss functions give
> non-sparse solutions (all $c_i \neq 0$) but still of the same form.

### The Connection to Ridge Regression and Gaussian Processes

The SVM is part of a larger family of kernel-based methods. Understanding these connections
deepens your intuition for when to use SVMs vs. alternatives.

**SVM vs. Kernel Ridge Regression (KRR):**

| Property | C-SVM (hinge loss) | KRR (squared loss) |
|---|---|---|
| Loss function | Hinge: $\max(0, 1 - y f(\mathbf{x}))$ | Squared: $(y - f(\mathbf{x}))^2$ |
| RKHS regularizer | $\|f\|_{\mathcal{H}}^2$ | $\|f\|_{\mathcal{H}}^2$ |
| Solution sparsity | Sparse (only SVs have $\alpha_i \neq 0$) | Dense (all $\alpha_i \neq 0$ generally) |
| Robustness to outliers | More robust (hinge saturates) | Sensitive (squared loss penalizes quadratically) |
| sklearn equivalent | `SVC` | `KernelRidge` |

Both solve regularized empirical risk minimization in an RKHS — the only difference is the loss
function. SVMs use hinge loss (produces sparse solutions); KRR uses squared loss (dense, but often
has a closed-form solution and is easier to compute).

**SVM vs. Gaussian Process Classification:** A Gaussian Process classifier places a GP prior over
$f$ and computes the posterior. The kernel in a GP plays exactly the same role as in an SVM: it
encodes similarity between inputs. For the same kernel, the GP and the SVM are solving related
problems — the GP does Bayesian inference, the SVM does regularized risk minimization. In practice,
SVMs are faster to train; GPs provide uncertainty quantification. For small $n$, GPs are often
worth the extra computational cost.

---

## §3 — The Full Evolution

### 3.1 ν-SVM — A More Interpretable Parameterization

> 📜 **Citation:**
> Schölkopf, B., Smola, A. J., Williamson, R. C., & Bartlett, P. L. (2000). New support vector
> algorithms. *Neural Computation*, 12(5), 1207–1245.
> DOI: https://doi.org/10.1162/089976600300015565

**The problem with C:** The parameter $C$ has no direct interpretation. Its optimal value depends
on the scale of labels, the dataset size, and the noise level. A $C$ that works well on one
dataset is meaningless for another.

**The ν-SVM fix:** Replace $C$ with $\nu \in (0, 1]$ that has a clean probabilistic interpretation:
- $\nu$ is an **upper bound** on the fraction of margin errors (misclassified or within-margin points)
- $\nu$ is a **lower bound** on the fraction of support vectors

**NuSVC primal:**

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \rho} \frac{1}{2}\|\mathbf{w}\|^2 - \nu\rho + \frac{1}{n}\sum_{i=1}^n \xi_i$$

$$\text{s.t. } y_i(\mathbf{w}^\top\phi(\mathbf{x}_i) + b) \geq \rho - \xi_i, \quad \xi_i \geq 0, \quad \rho \geq 0$$

The margin $\rho$ is now a free variable that's also optimized, rather than being fixed to 1 as in
C-SVM. When $\nu \to 0$, $\rho$ is large (hard margin). When $\nu = 1$, almost all points become
support vectors.

```python
from sklearn.svm import NuSVC, NuSVR

# nu=0.1 → at most 10% margin errors; at least 10% support vectors
nusvc = NuSVC(nu=0.1, kernel='rbf', gamma='scale', decision_function_shape='ovr')

# Note: NuSVC will raise ValueError if nu is infeasible given class imbalance
# Feasibility requirement: nu <= 2 * min(n_pos, n_neg) / n
```

> ⚠️ **Pitfall:** `NuSVC` raises `ValueError: specified nu is infeasible` when $\nu > 2\min(n_+, n_-)/n$. With a 1:99 class ratio, $\nu$ must be below 0.02. Always check class balance before using NuSVC.

**When to use:** When you want to control the fraction of the dataset that becomes support
vectors, or when you're comparing models across datasets with different sizes and need a
scale-free regularization parameter. In practice, C-SVM (`SVC`) is preferred for most tasks
as it's numerically more stable.

### 3.2 Support Vector Regression (SVR) — Extending to Continuous Targets

> 📜 **Citation:**
> Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A., & Vapnik, V. (1997). Support vector
> regression machines. *Advances in Neural Information Processing Systems*, 9, 155–161.

**The extension.** Instead of classifying $y_i \in \{-1, +1\}$, SVR fits a continuous target
$y_i \in \mathbb{R}$. The key innovation is the **ε-insensitive loss**:

$$L_\varepsilon(y, f(\mathbf{x})) = \max(0, |y - f(\mathbf{x})| - \varepsilon)$$

No loss is incurred for predictions within an ε-tube around the true value. This produces sparse
solutions where only the points *outside* the ε-tube become support vectors.

**SVR primal:**

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n(\xi_i + \xi_i^*)$$

$$\text{s.t. } y_i - f(\mathbf{x}_i) \leq \varepsilon + \xi_i, \quad f(\mathbf{x}_i) - y_i \leq \varepsilon + \xi_i^*, \quad \xi_i, \xi_i^* \geq 0$$

Two slack variables per point: $\xi_i$ for points above the tube, $\xi_i^*$ for points below.

**Prediction:**

$$f(\mathbf{x}) = \sum_{i \in \text{SV}} (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b$$

The coefficients $(\alpha_i - \alpha_i^*)$ can be positive or negative depending on whether point
$i$ is above or below the tube.

```python
from sklearn.svm import SVR, NuSVR

# SVR: specify epsilon explicitly
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')

# NuSVR: nu controls fraction of SVs (epsilon is automatically determined)
nusvr = NuSVR(kernel='rbf', C=1.0, nu=0.5, gamma='scale')
```

> 💡 **Intuition:** SVR is like fitting a "fat" regression line rather than a thin one. The
> thickness of the line is 2ε. Any point that falls within the fat tube is considered "well
> predicted" and contributes no loss. Only points outside the tube are support vectors and drive
> the optimization.

**Tuning ε.** Start with ε proportional to the noise level in your target. If you standardize
targets to mean 0, std 1, start with `epsilon=0.1`. Too small ε → every point is a support
vector → overfitting. Too large ε → model ignores fine structure → underfitting.

### 3.3 LinearSVC / LinearSVR — Scaling to Millions of Samples

> 📜 **Citation:**
> Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008). LIBLINEAR: A library
> for large linear classification. *Journal of Machine Learning Research*, 9, 1871–1874.
> URL: https://www.jmlr.org/papers/volume9/fan08a/fan08a.pdf

**The motivation.** LIBSVM's SMO is $O(n^2)$ to $O(n^3)$. For text classification where
$n > 10^6$ and $p > 10^5$ (news categorization, spam detection), this is completely impractical.
Fan et al. developed LIBLINEAR, which solves the linear SVM primal (or dual) using coordinate
descent, achieving near-$O(n)$ complexity.

**Key differences from SVC(kernel='linear'):**

| Feature | `SVC(kernel='linear')` | `LinearSVC` |
|---|---|---|
| Backend | LIBSVM (SMO) | LIBLINEAR (coord. descent) |
| Complexity | $O(n^2)$–$O(n^3)$ | $O(n)$ approx. |
| Multiclass | OvO | OvR (default) |
| Default loss | Hinge | Squared hinge |
| `predict_proba` | Yes (probability=True) | No (needs CalibratedClassifierCV) |
| Practical $n$ | Up to ~50,000 | Millions |
| `coef_` | No (implicit via dual) | Yes — directly interpretable |

```python
from sklearn.svm import LinearSVC

# Good defaults for text classification
lsvc = LinearSVC(
    C=1.0,
    loss='squared_hinge',   # default; smoother than hinge
    penalty='l2',           # standard; use 'l1' for sparse coef_
    dual='auto',            # sklearn 1.3+: auto-selects dual vs primal
    max_iter=5000,          # increase from 1000 default for text
    random_state=42
)

# L1 penalty for sparse coef_ (automatic feature selection):
lsvc_l1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=0.1, max_iter=10000)
```

> 🏆 **Best practice:** For text classification (bag-of-words, TF-IDF with $n > 10^4$),
> `LinearSVC` is typically the best SVM choice. It matches or exceeds kernel SVC in accuracy
> while being orders of magnitude faster. The `coef_` attribute gives direct feature importance.

### 3.4 Kernel Approximation + LinearSVC — Scalable Nonlinear SVMs

> 📜 **Citation:**
> Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *NIPS*, 20.
>
> Williams, C. K. I., & Seeger, M. (2001). Using the Nyström method to speed up kernel machines.
> *NIPS*, 13, 661–667.

**The problem:** For $n > 50{,}000$ with a nonlinear kernel, LIBSVM becomes impractical. But
`LinearSVC` sacrifices all nonlinearity. The solution: **approximate the kernel map**
$\phi(\mathbf{x})$ with an explicit low-dimensional map $\tilde{\phi}(\mathbf{x}) \in \mathbb{R}^D$,
then apply `LinearSVC` on the transformed features.

**Two approaches:**

1. **RBFSampler (Random Kitchen Sinks):** Draw $D$ random frequencies from $\mathcal{N}(0, \gamma\mathbf{I})$ and compute trigonometric features. Approximates the RBF kernel via Bochner's theorem.

2. **Nystroem:** Select $D$ landmark points from training data and use kernel submatrix eigendecomposition. More accurate than random features for the same $D$.

```python
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Nystroem — more accurate, recommended
pipe_nys = Pipeline([
    ('scaler', StandardScaler()),
    ('nystroem', Nystroem(kernel='rbf', gamma=0.01, n_components=500, random_state=42)),
    ('svc', LinearSVC(C=1.0, max_iter=5000, random_state=42))
])

# RBFSampler — faster for very large datasets
pipe_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('rbf', RBFSampler(gamma=0.01, n_components=500, random_state=42)),
    ('svc', LinearSVC(C=1.0, max_iter=5000, random_state=42))
])
```

**Trade-off:** As $D$ increases, approximation accuracy improves but training time grows. In
practice, $D = 100$ to $1000$ components often suffices to closely approximate a full kernel SVM
while training in seconds on datasets with millions of samples.

> 🔬 **Deep dive:** The Nystroem approximation uses a subset of training points as basis vectors. Formally, if the landmark points are $\{z_1, \ldots, z_D\}$ and $K_D = [K(z_i, z_j)]_{D\times D}$, the approximation is $\tilde{\phi}(\mathbf{x}) = K_D^{-1/2} [K(\mathbf{x}, z_1), \ldots, K(\mathbf{x}, z_D)]^\top$. This gives both interpretable `coef_` and fast SHAP via `LinearExplainer`.

### 3.5 ThunderSVM — GPU-Accelerated SVM

> 📜 **Citation:**
> Wen, Z., Shi, J., Li, Q., He, B., & Chen, J. (2018). ThunderSVM: A fast SVM library on GPUs
> and CPUs. *Journal of Machine Learning Research*, 19, 1–5.
> URL: https://www.jmlr.org/papers/v19/17-740.html

For datasets where $n = 10{,}000$ to $500{,}000$ and you want a full kernel SVM, ThunderSVM
parallelizes the SMO algorithm across GPU cores with a reported ~100× speedup over LIBSVM.

```python
# Requires CUDA toolkit — verify compatibility before production use
# pip install thundersvm
from thundersvm import SVC  # identical API to sklearn SVC

model = SVC(kernel='rbf', C=10.0, gamma='scale')
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
```

> ⚠️ **Pitfall:** Verify ThunderSVM maintenance status before committing to it in production.
> Check https://github.com/Xtra-Computing/thundersvm/releases for the latest Python/CUDA
> compatibility. RAPIDS `cuml.svm` may be better maintained for GPU deployments.

### 3.6 Platt Scaling — Calibrated Probabilities

> 📜 **Citation:**
> Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to
> regularized likelihood methods. *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74.

**The problem.** SVMs produce a decision function value $f(\mathbf{x})$ that is not a probability.
For decision-making under uncertainty (e.g., "only alert if P(fraud) > 0.95"), you need
calibrated probabilities.

**Platt's solution.** Fit a sigmoid $P(y=1|\mathbf{x}) = 1/(1 + \exp(A \cdot f(\mathbf{x}) + B))$
by minimizing log-loss on held-out out-of-fold predictions.

```python
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Method 1: SVC(probability=True) — Platt scaling via 5-fold internal CV
svc_prob = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svc_prob.fit(X_train_sc, y_train)
probs = svc_prob.predict_proba(X_test_sc)

# Method 2: CalibratedClassifierCV — more flexible and preferred
base_svc = SVC(kernel='rbf', C=1.0)
cal_svc = CalibratedClassifierCV(base_svc, cv=5, method='sigmoid')   # Platt
cal_svc.fit(X_train_sc, y_train)
probs = cal_svc.predict_proba(X_test_sc)
```

> ⚠️ **Pitfall:** `SVC(probability=True)` adds ~5× training overhead and causes `predict()` and
> `predict_proba()` to disagree near the boundary (they use different internal functions). Use
> `CalibratedClassifierCV` for more control and avoid the inconsistency.

### 3.7 One-Class SVM — Anomaly Detection

The **One-Class SVM** (Schölkopf et al., 1999) learns a boundary around the "normal" class from
unlabeled data. It finds the hyperplane separating mapped data from the origin with the largest
margin, which corresponds geometrically to finding the smallest hypersphere enclosing the data
in feature space (SVDD).

```python
from sklearn.svm import OneClassSVM

# nu: upper bound on the fraction of outliers (training anomaly rate)
oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
oc_svm.fit(X_normal_train)          # fit only on "normal" class
predictions = oc_svm.predict(X_test)
# Returns +1 (inlier) or -1 (outlier)
scores = oc_svm.score_samples(X_test)   # anomaly score
```

> 🔧 **In practice:** For anomaly detection on tabular data, Isolation Forest and Local Outlier
> Factor often outperform One-Class SVM because they don't require kernel tuning. One-Class SVM
> shines when you have strong prior knowledge about the kernel structure (genomics, text).

### 3.8 String Kernels and Custom Domain Kernels — SVMs in Bioinformatics and NLP

One of the most powerful features of the SVM framework is the ability to inject domain knowledge
through custom kernel functions. The entire SVM theory carries over unchanged — you just need a
positive semi-definite kernel matrix, and the rest follows from the representer theorem.

**String kernels for text and biological sequences:**

The **spectrum kernel** (Leslie et al., 2002) counts the number of shared $k$-grams between
two strings, giving a natural similarity measure for DNA sequences, protein sequences, or
text:

$$K_k(s, t) = \sum_{u \in \Sigma^k} \langle \phi_u(s), \phi_u(t) \rangle$$

where $\phi_u(s)$ is the count of $k$-mer $u$ in string $s$ and $\Sigma$ is the alphabet.

```python
from sklearn.svm import SVC
import numpy as np
from collections import Counter

def spectrum_kernel(s1, s2, k=3):
    """k-spectrum kernel: count shared k-mers between two strings."""
    def get_kmers(s, k):
        return Counter(s[i:i+k] for i in range(len(s) - k + 1))
    
    kmers1 = get_kmers(s1, k)
    kmers2 = get_kmers(s2, k)
    
    # Dot product in k-mer count space
    shared = set(kmers1.keys()) & set(kmers2.keys())
    return sum(kmers1[u] * kmers2[u] for u in shared)

# Example: DNA sequence classification
sequences_train = ['ACGTAGCTAGCG', 'GCTAGCTAGCTAG', 'TTACGTAGCTAG',
                   'AATTCCGGAATT', 'CCGGAATTCCGG', 'TTAACCGGTTAA']
labels_train = [0, 0, 0, 1, 1, 1]   # 0=promoter, 1=exon (simplified)
sequences_test = ['ACGTAGCTAGAT', 'CCGGAATTCCAA']

# Build Gram matrices
n_train = len(sequences_train)
n_test  = len(sequences_test)
K_train = np.array([[spectrum_kernel(s1, s2, k=3) for s2 in sequences_train]
                    for s1 in sequences_train])
K_test  = np.array([[spectrum_kernel(s1, s2, k=3) for s2 in sequences_train]
                    for s1 in sequences_test])

# Normalize kernel matrix (recommended for better numerical conditioning)
D = np.sqrt(np.diag(K_train))
K_train_norm = K_train / np.outer(D, D)
K_test_norm  = K_test  / np.outer(np.sqrt([spectrum_kernel(s,s,k=3) for s in sequences_test]), D)

# Train SVM with precomputed kernel
svc = SVC(kernel='precomputed', C=1.0)
svc.fit(K_train_norm, labels_train)
y_pred = svc.predict(K_test_norm)
print(f"Predicted labels: {y_pred}")
print(f"Support vector indices: {svc.support_}")
```

**Graph kernels for molecular property prediction:**

For molecular graphs (atoms = nodes, bonds = edges), the Weisfeiler-Leman (WL) graph kernel
computes similarity based on neighborhood aggregation — the same principle behind Graph Neural
Networks, but framed as a kernel:

```python
# Graph kernels: use grakel library
# pip install grakel
from grakel import Graph, GraphKernel
from grakel.kernels import WeisfeilerLehman, VertexHistogram

# Example: molecular graph classification
# molecules represented as (edge_set, node_labels) pairs
# ...full example requires domain-specific graph construction
# This is the general pattern:
# K = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5},
#                         {"name": "vertex_histogram"}])
# K_train = K.fit_transform(graphs_train)
# K_test  = K.transform(graphs_test)
# svc = SVC(kernel='precomputed', C=1.0).fit(K_train, y_train)
```

> 📜 **Origin/Citation:**
> Leslie, C., Eskin, E., & Noble, W. S. (2002). The spectrum kernel: A string kernel for SVM protein classification. *Pacific Symposium on Biocomputing*, 7, 564–575.
>
> Lodhi, H., Saunders, C., Shawe-Taylor, J., Cristianini, N., & Watkins, C. (2002). Text classification using string kernels. *Journal of Machine Learning Research*, 2, 419–444.
> URL: https://www.jmlr.org/papers/volume2/lodhi02a/lodhi02a.pdf

**When to use custom kernels:**
- You have structured inputs (sequences, graphs, trees) where Euclidean distance is not meaningful
- You have domain-specific similarity measures (edit distance, BLAST score, graph isomorphism)
- You want to incorporate prior knowledge about which inputs should be similar

> 🔧 **In practice:** With `kernel='precomputed'`, `support_vectors_` is NOT stored — only
> `support_` (indices). To retrieve support vectors: `support_sequences = [sequences_train[i] for i in svc.support_]`. For large datasets, precomputing the full Gram matrix may be memory-intensive; consider using an approximation or chunked computation.

### 3.9 SVM for Large-Scale Problems — A Practical Migration Guide

As your dataset grows, you need to migrate your SVM approach. Here's a concrete guide:

```
Dataset size → Recommended approach
─────────────────────────────────────────────────────────
n < 1,000         SVC(kernel='rbf', cache_size=500)
                  Full kernel SVM, fast, cache whole matrix
                  
1,000–10,000      SVC(kernel='rbf', cache_size=2000)
                  Still full kernel SVM, increase cache
                  Optuna tuning: 50 trials, ~5 minutes
                  
10,000–100,000    Nystroem(n_components=500) + LinearSVC
                  Approximate kernel, near-linear speed
                  OR ThunderSVM on GPU (100× speedup)
                  
100,000–1,000,000 LinearSVC(C=1.0) for linear problems
                  SGDClassifier(loss='hinge') for online
                  For nonlinear: consider XGBoost/LightGBM
                  
> 1,000,000       Do NOT use SVM for nonlinear problems
                  LinearSVC with stochastic optimization
                  or neural networks
```

```python
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

# For n > 10,000 but < 100,000 — Nystroem + LinearSVC
def build_scalable_svm(n_samples, kernel='rbf', C=1.0, gamma=0.01, n_components=300):
    if n_samples < 10_000:
        from sklearn.svm import SVC
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel=kernel, C=C, gamma=gamma, cache_size=2000))
        ])
    elif n_samples < 100_000:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('approx', Nystroem(kernel=kernel, gamma=gamma,
                                n_components=n_components, random_state=42)),
            ('clf',    LinearSVC(C=C, max_iter=5000, random_state=42))
        ])
    else:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LinearSVC(C=C, max_iter=10000, random_state=42))
        ])

# For true online/streaming SVM: SGDClassifier with hinge loss
online_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(
        loss='hinge',         # SVM hinge loss
        penalty='l2',         # L2 regularization (like SVM)
        alpha=1.0/(1000 * 1), # alpha = 1/(n * C), so C=1 with n=1000 → alpha=0.001
        max_iter=1000,
        random_state=42
    ))
])
# SGDClassifier.partial_fit() for true incremental learning
```

> 💡 **Intuition:** `SGDClassifier(loss='hinge')` is an online SVM that updates weights
> stochastically. It's mathematically equivalent to a linear SVM in the limit of many passes
> through the data, but trains in $O(n)$ time per epoch. Use it when data doesn't fit in
> memory, when you need to retrain incrementally as new data arrives, or when $n > 10^6$.

---

## §4 — Hyperparameters: The Complete Guide

### 4.1 C — The Regularization Parameter

**What it controls:** $C$ is the penalty per unit of hinge loss. Mathematically, $C$ is the
coefficient of the slack variable sum $\sum_i \xi_i$ in the primal. Large $C$: misclassifications
are expensive, model tries hard to classify every point correctly (low bias, high variance,
narrow margin). Small $C$: misclassifications are cheap, model accepts more violations in exchange
for a wider margin (high bias, low variance).

**Default:** 1.0 (SVC, SVR, LinearSVC, NuSVC).

**Mechanism:**
- $C \to \infty$: approaches hard-margin SVM; may not converge if data not separable in kernel space
- $C = 1$: balanced; reasonable starting point
- $C \to 0$: all points become support vectors; margin is maximal; model ignores labels

**How to tune:** Always log-scale. Joint search with `gamma` for kernel SVMs.

```python
from sklearn.model_selection import GridSearchCV
import numpy as np

param_grid = {
    'svc__C':     np.logspace(-2, 3, 6),    # [0.01, 0.1, 1, 10, 100, 1000]
    'svc__gamma': np.logspace(-4, 1, 6),    # [1e-4, 1e-3, 0.01, 0.1, 1, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
```

> 💡 **Intuition:** Think of $1/C$ as the ridge regularization strength in linear models. High $C$ ≈ small $\lambda$ (low regularization, fits training data tightly). Small $C$ ≈ large $\lambda$ (strong regularization, smooth boundary). The analogy is exact for LinearSVC vs. Ridge regression.

**The C-gamma interaction (critical):** For RBF SVMs, C and gamma must be tuned jointly. The
optimal region forms a diagonal band in the (log C, log gamma) space. As gamma increases (narrower
kernels), you typically need smaller C to prevent overfitting. Searching them independently misses
the optimal region. The sklearn documentation has an excellent visualization of this at:
https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

### 4.2 kernel — The Decision Boundary Shape

**What it controls:** The kernel determines what kinds of decision boundaries the SVM can
represent. It's the most consequential architectural choice in SVM design.

| Kernel | Formula | Parameters | When to use |
|---|---|---|---|
| `'linear'` | $\langle\mathbf{x}, \mathbf{x}'\rangle$ | None | Linearly separable data, high $p$, text; use `LinearSVC` instead |
| `'rbf'` | $\exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)$ | `gamma` | **Default choice**; universal approximator; try first |
| `'poly'` | $(\gamma\langle\mathbf{x},\mathbf{x}'\rangle + r)^d$ | `gamma`, `coef0`, `degree` | Images, multiplicative feature interactions |
| `'sigmoid'` | $\tanh(\gamma\langle\mathbf{x},\mathbf{x}'\rangle + r)$ | `gamma`, `coef0` | Rarely competitive; not always Mercer (use with caution) |
| `'precomputed'` | Custom $K_{ij}$ | — | Graph kernels, string kernels, custom domain similarity |

**Mercer's condition:** Valid kernels must produce positive semi-definite (PSD) Gram matrices.
The RBF kernel is always PSD. The sigmoid kernel is **not** always PSD — it's only PSD for
certain $(\gamma, r)$ combinations. Using a non-PSD kernel can produce unstable optimization
with no theoretical guarantees.

**Precomputed kernels:**
```python
# Compute Gram matrices from domain-specific similarity
K_train = np.array([[domain_kernel(s1, s2) for s2 in X_train] for s1 in X_train])
K_test  = np.array([[domain_kernel(s1, s2) for s2 in X_train] for s1 in X_test])

svc = SVC(kernel='precomputed')
svc.fit(K_train, y_train)    # pass n_train × n_train Gram matrix
pred = svc.predict(K_test)   # pass n_test × n_train kernel matrix
```

### 4.3 gamma — The RBF Bandwidth

**What it controls:** $\gamma$ in the RBF kernel $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$ controls how quickly kernel similarity decays with distance. Large $\gamma$: only very nearby points are "similar" (tight Gaussians → complex, wiggly boundary → overfitting risk). Small $\gamma$: distant points are still similar (broad Gaussians → smooth boundary → underfitting risk).

**Default:** `'scale'` since sklearn 0.22. Formula: $\gamma = 1 / (p \cdot \text{Var}(\mathbf{X}))$ where $\text{Var}(\mathbf{X})$ is computed over the full training matrix. This accounts for both the number of features and the overall scale of the data.

- `gamma='scale'` (recommended): $\gamma = 1/(p \cdot \text{Var}(\mathbf{X}))$ — accounts for scale
- `gamma='auto'` (pre-0.22 default): $\gamma = 1/p$ — ignores feature variance

> ⚠️ **Pitfall:** If you're running code written before sklearn 0.22 that relied on the default, behavior changed. Always specify `gamma='scale'` after scaling, or tune gamma explicitly.

**Tuning:** After StandardScaler, `gamma='scale'` gives $\gamma = 1/p$, a reasonable starting
point. Search on a log scale: `np.logspace(-4, 1, 12)`.

### 4.4 degree — Polynomial Kernel Degree

**Default:** 3 (ignored for non-polynomial kernels).

Higher degree allows more complex boundaries but increases overfitting risk. Search `[2, 3, 4, 5]`.
Degree 2 (quadratic) is usually the sweet spot for image data. Degrees above 5 rarely help.

### 4.5 coef0 — Free Term in Poly/Sigmoid Kernels

**Default:** 0.0. In the polynomial kernel $(\gamma\langle\mathbf{x}, \mathbf{x}'\rangle + r)^d$,
setting $r = 0$ gives a homogeneous polynomial; $r > 0$ includes all degrees up to $d$. For
polynomial kernels, try $r \in \{0, 1\}$.

### 4.6 shrinking — SMO Heuristic

**Default:** `True`. When enabled, points that clearly satisfy KKT conditions are temporarily
removed from the working set, giving 2–10× speedup without affecting the solution. Set
`shrinking=False` only for debugging.

### 4.7 probability — Platt Scaling

**Default:** `False`. When `True`, adds ~5× training time via internal 5-fold CV to fit a
sigmoid for probability outputs. Prefer `CalibratedClassifierCV` for more control.

### 4.8 tol — Stopping Criterion

**Default:** `1e-3` (SVC/SVR); `1e-4` (LinearSVC). Smaller `tol` = more precise optimization
but slower convergence. If you get `ConvergenceWarning`, first try scaling features; then
increase `max_iter`; then tighten `tol` to `1e-6`.

### 4.9 cache_size — Kernel Cache

**Default:** 200 MB. For $n > 5{,}000$, increase to at least 1000 MB. For $n = 50{,}000$, use
4000+ MB if RAM allows. The speedup can be 5–10×.

```python
SVC(kernel='rbf', C=1.0, cache_size=2000)  # 2 GB cache
```

### 4.10 class_weight — Handling Imbalance

**Default:** `None` (equal weights). With `class_weight='balanced'`, effective $C$ for class $k$
becomes $C \cdot n / (K \cdot n_k)$. Use whenever the rarest class is < 20% of the dataset.

```python
# Imbalanced binary: 95% class 0, 5% class 1
svc = SVC(kernel='rbf', C=1.0, class_weight='balanced')
# Minority class gets C scaled up by factor ~19 — forces the margin to
# penalize minority misclassifications much more heavily
```

### 4.11 max_iter — Iteration Limit

**Default:** -1 (unlimited) for SVC/SVR; 1000 for LinearSVC. For LinearSVC on text, use
`max_iter=10000`. If `ConvergenceWarning` persists after scaling, try `tol=1e-3` first.

### 4.12 decision_function_shape — Multiclass Output

**Default:** `'ovr'`. SVC trains OvO internally but reshapes output to $(n, K)$ for sklearn
compatibility. Does NOT change training. Use `break_ties=True` for deterministic tiebreaking.

### 4.13 Optuna-Based Hyperparameter Search (Recommended for Efficiency)

Grid search over a 6×6 grid requires 36 × 5 folds = 180 SVM fits. With Optuna's Tree Parzen
Estimator (TPE), you can often find better hyperparameters in 50 trials by focusing on promising
regions of the search space:

```python
import optuna
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

def svc_objective(trial):
    C     = trial.suggest_float('C',     1e-3, 1e3, log=True)
    gamma = trial.suggest_float('gamma', 1e-5, 1e1, log=True)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
    
    params = dict(kernel=kernel, C=C, gamma=gamma, random_state=42)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 4)
        params['coef0']  = trial.suggest_float('coef0', 0.0, 1.0)
    
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(**params))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(svc_objective, n_trials=60, show_progress_bar=True)

print(f"Best params: {study.best_params}")
print(f"Best CV accuracy: {study.best_value:.4f}")

# Visualize optimization history (requires optuna.visualization)
try:
    import optuna.visualization as vis
    fig = vis.plot_param_importances(study)
    fig.show()
    fig2 = vis.plot_contour(study, params=['C', 'gamma'])
    fig2.show()
except ImportError:
    pass   # plotly not installed

# Build final model with best params
best_p = study.best_params
final_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(
        kernel=best_p['kernel'],
        C=best_p['C'],
        gamma=best_p['gamma'],
        degree=best_p.get('degree', 3),
        coef0=best_p.get('coef0', 0.0),
        random_state=42
    ))
])
final_svc.fit(X_train, y_train)
print(f"Test accuracy (Optuna-tuned): {final_svc.score(X_test, y_test):.4f}")
```

> 🏆 **Best practice:** For kernel SVM tuning, prefer Optuna with TPE over pure grid search.
> TPE samples promising regions more densely after each trial, reaching near-optimal performance
> in ~50 trials vs. 36+ for a 6×6 grid with the same or better coverage of the search space.
> The log-uniform prior on $C$ and $\gamma$ is critical — never use a linear prior for SVM
> hyperparameters.

### Multiclass Strategies — OvO vs. OvR in Depth

Understanding how sklearn handles multiclass SVMs prevents subtle bugs and performance surprises.

**One-vs-One (OvO) — used by SVC and NuSVC:**
- Trains $\binom{K}{2} = K(K-1)/2$ binary SVMs
- Each classifier gets a subset of training data (only samples from the two classes)
- Prediction: majority vote among all $K(K-1)/2$ classifiers
- For $K = 10$: 45 binary SVMs, each trained on $\approx 2n/K$ samples

**One-vs-Rest (OvR) — used by LinearSVC:**
- Trains $K$ binary SVMs
- Each classifier sees all $n$ training samples
- Prediction: argmax of $K$ decision function values
- For $K = 10$: 10 binary SVMs, each trained on $n$ samples

**Which is better?** OvO is usually preferred for kernel SVMs:
1. Each OvO binary problem is smaller (fewer samples → faster)
2. Each OvO problem is more balanced (roughly equal class sizes)
3. OvR can suffer when one class dominates the training data

```python
from sklearn.svm import SVC
import numpy as np

# OvO (default for SVC): decision_function shape
svc_ovo = SVC(kernel='rbf', C=1.0, decision_function_shape='ovo')
svc_ovo.fit(X_train_sc, y_train)
# decision_function shape: (n_test, K*(K-1)/2) = (n, 45) for K=10
df_ovo = svc_ovo.decision_function(X_test_sc)
print(f"OvO decision function shape: {df_ovo.shape}")

# OvR output (default display shape, but same OvO training!)
svc_ovr = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr')
svc_ovr.fit(X_train_sc, y_train)
# decision_function shape: (n_test, K) = (n, 10) for K=10
# but TRAINING is still OvO internally
df_ovr = svc_ovr.decision_function(X_test_sc)
print(f"OvR display decision function shape: {df_ovr.shape}")
# predict() uses OvO voting, NOT argmax(df_ovr) — they can differ!
```

> ⚠️ **Pitfall:** Even with `decision_function_shape='ovr'`, `predict()` uses **OvO voting** internally. The OvR-shaped decision function is created by summing OvO votes per class. This means `np.argmax(model.decision_function(X), axis=1)` may differ from `model.predict(X)` when there are ties. Use `break_ties=True` for deterministic prediction.

### Hyperparameter Tuning Playbook

| Hyperparameter | Typical Range | Effect | Strategy |
|---|---|---|---|
| `C` | $[10^{-3}, 10^3]$ log-scale | Bias-variance tradeoff | Joint log-scale grid with `gamma` |
| `gamma` (RBF) | $[10^{-4}, 10^1]$ log-scale after scaling | Boundary smoothness | Joint log-scale grid with `C` |
| `kernel` | `'rbf'`, `'linear'`, `'poly'` | Hypothesis class | Try `'rbf'` first; `'linear'`+LinearSVC for large $n$ |
| `degree` (poly) | $[2, 3, 4, 5]$ | Polynomial complexity | Grid search integers |
| `epsilon` (SVR) | $[10^{-3}, 1]$ log-scale | Tube width / sparsity | Proportional to target noise level |
| `nu` (NuSVC/NuSVR) | $(0, 0.5]$ | Fraction of SVs/errors | Grid; check feasibility first |
| `class_weight` | `None`, `'balanced'` | Imbalance correction | Use `'balanced'` if recall matters |
| `cache_size` | $[200, 8000]$ MB | Training speed | Max out within RAM limits |

---

## §5 — Strengths

### 5.1 Effective in High-Dimensional Spaces

SVMs excel when $p \gg n$. The VC dimension of the maximum-margin classifier is bounded by
$h \leq \min(p, \lfloor R^2/\gamma^2 \rfloor) + 1$ where $R$ is the data's enclosing ball radius
and $\gamma$ is the margin. This bound depends on $R/\gamma$ (a dimensionless ratio), **not** on
$p$ directly. A large-margin SVM in 10,000 dimensions can have the same generalization bound as
one in 2 dimensions — if the margin is the same.

> 📊 **Benchmark:** Text classification benchmarks on 20 Newsgroups (19,997 documents, ~100,000
> vocabulary features) consistently show LinearSVC achieving 85–90% accuracy, competitive with
> logistic regression and dramatically outperforming naive Bayes on many categories.

### 5.2 Memory Efficient — Only Support Vectors Are Stored

After training, the SVM only stores support vectors and their $\alpha_i$ coefficients. In
practice, this is often 10–20% of the training set. For inference, you only need
$n_{SV} \ll n$ kernel evaluations.

> 🔧 **In practice:** Check `model.n_support_` (per-class SV count) and `model.support_vectors_`
> after training. If the ratio of SVs to training samples is low (say, 10–20%), your model is
> sparse and memory-efficient.

### 5.3 Global Optimum Guaranteed

The SVM training problem is a convex QP with a unique global minimum. Unlike neural networks
and gradient boosting, there are no local minima and no sensitivity to random initialization.
Given the same data and hyperparameters, LIBSVM will always produce the same solution (to within
`tol` tolerance).

**Mechanism:** The primal objective $\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i$ is strictly
convex in $\mathbf{w}$ and convex in $\boldsymbol{\xi}$. The dual is a bounded concave QP.
Both are solved to global optimality by SMO.

### 5.4 The Kernel Trick — Implicit Infinite-Dimensional Feature Engineering

The RBF kernel is a **universal approximator** — given sufficient training data and optimal $\gamma$
and $C$, it can approximate any continuous classification function to arbitrary accuracy. No manual
feature engineering is required; the kernel implicitly performs it.

**Mechanism:** The RBF kernel corresponds to a dot product in a Hilbert space of infinite
dimension. Any continuous function on a compact domain can be approximated by linear combinations
of RBF kernels (Micchelli's theorem). The margin constraint prevents overfitting despite the
infinite dimensionality.

### 5.5 Works Well with Domain-Specific Kernels

For domains where Euclidean geometry is not the right similarity metric (protein sequences, DNA,
graphs, molecules), custom kernels allow injecting domain knowledge directly. No other commonly
used algorithm accommodates custom similarity functions as cleanly — you just pass a precomputed
Gram matrix and everything else follows from the SVM framework.

### 5.6 Sparse, Interpretable Linear Models (LinearSVC)

With `LinearSVC(penalty='l1')`, many coefficients are exactly zero — automatic feature selection
with a model that's fully interpretable via `coef_`. This combination is exceptionally valuable
for text classification and bioinformatics where the number of features is large but only a small
fraction are relevant.

### 5.7 Robustness to the Curse of Dimensionality

Most distance-based algorithms degrade catastrophically in high dimensions — all pairwise
distances converge to the same value (a phenomenon known as distance concentration). The RBF
kernel SVM is relatively resistant to this because its generalization is bounded not by the
ambient dimension $p$ but by the margin $\gamma$ and the data radius $R$. In practice, this
means kernel SVMs can work surprisingly well at $p = 1000$ or $p = 10000$ dimensions where KNN
completely fails.

**Why this works:** The SVM finds a hyperplane in the RKHS that has maximum margin. In high
dimensions, the RKHS provides a rich enough representation that a separating hyperplane with
large margin exists. The regularization (via the RKHS norm constraint) prevents the model from
memorizing the training set despite the high dimensionality.

> 📊 **Benchmark:** In the canonical text classification benchmark (Reuters-21578), kernel
> SVMs achieved state-of-the-art results with $p > 50{,}000$ TF-IDF features and $n \approx
> 8{,}000$ training documents. This high-$p$/low-$n$ regime is precisely where SVM's
> margin-based capacity control shines compared to models whose capacity grows with $p$.

### 5.8 Excellent Performance on Clean, Medium-Sized Datasets

On clean, well-preprocessed tabular datasets with $n = 500$ to $n = 10{,}000$ and moderate
$p$, properly tuned kernel SVMs frequently match or beat gradient boosting and neural networks.
The keys are correct feature scaling, joint C/gamma tuning, and appropriate kernel choice.

> 📊 **Benchmark:** In the original Cortes & Vapnik (1995) paper, SVMs achieved 0.8% error
> on the USPS handwritten digit test set — better than neural networks and boosted trees of
> the same era. Modern RBF SVMs on the same dataset still achieve around 0.5% error with
> proper tuning, competitive with convolutional networks on this relatively simple task.

---

## §6 — Weaknesses & Failure Modes

### 6.1 Does Not Scale to Large Datasets

**Mechanism:** LIBSVM's SMO is $O(n^2)$ to $O(n^3)$ in the number of training samples. For
$n = 100{,}000$ with an RBF kernel, training can take hours or days.

**Detection:** Training time exceeds a few minutes on $n > 10{,}000$. No progress output.

**Mitigation:**
```
n < 5,000:         SVC(kernel='rbf')                    — fine
5,000–50,000:      Nystroem + LinearSVC                 — recommended
n > 50,000:        LinearSVC  or  GBM/neural network    — kernel SVM impractical
```

### 6.2 Sensitive to Feature Scaling

**Mechanism:** The RBF kernel computes $\exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$. If one
feature ranges 0–10,000 and another 0–1, the first dominates the distance by a factor of $10^8$,
effectively making the second feature invisible to the model.

**Detection:** Wide spread of feature scales in `X.describe()`. Poor performance without scaling.

**Mitigation:** Always use `StandardScaler` inside a `Pipeline`. Never fit the scaler on test data.

```python
# ALWAYS use a Pipeline — never fit scaler separately:
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf'))])
pipe.fit(X_train, y_train)   # scaler fitted on train only
```

### 6.3 Requires Careful Joint Hyperparameter Tuning

**Mechanism:** Performance is highly sensitive to the joint $(C, \gamma)$ region. Their
interaction means independent tuning misses the optimum. The search space spans ~6 orders of
magnitude in each direction.

**Detection:** Large performance discrepancy across nearby hyperparameter values in grid search.

**Mitigation:** Joint log-scale 2D grid search with 5-fold CV, or Optuna with log-uniform priors.

### 6.4 No Native Probability Outputs

**Mechanism:** SVMs optimize hinge loss, not log-loss. The decision function value $f(\mathbf{x})$
is not calibrated — its scale depends on the data and hyperparameters.

**Mitigation:** Use `CalibratedClassifierCV`. Check calibration with a reliability diagram.
Even with Platt scaling, kernel SVM calibration can be imperfect.

### 6.5 Poor Interpretability for Kernel SVMs

**Mechanism:** The weight vector $\mathbf{w}$ lives in an implicit high-dimensional feature space.
There is no way to directly inspect feature importances in the original input space.

**Mitigation:** Use permutation importance (slow but general) or switch to Nystroem + LinearSVC
for interpretable `coef_` and fast SHAP LinearExplainer.

### 6.6 Multiclass is Indirect and Slow for Many Classes

**Mechanism:** SVMs are fundamentally binary. OvO trains $K(K-1)/2$ classifiers. For $K=100$
classes, that's 4,950 binary SVMs. Prediction requires voting among all of them.

**Mitigation:** For many-class problems ($K > 10$), prefer `LinearSVC` with `multi_class='ovr'`
or `multi_class='crammer_singer'`, which handles multiclass more directly.

### 6.7 Sensitive to Outliers

**Mechanism:** Outliers that violate the margin become bounded support vectors with $\alpha_i = C$.
A single extreme outlier can shift the decision boundary significantly when $C$ is large.

**Detection:** Training with/without suspected outliers gives very different results.

**Mitigation:** Robust preprocessing (`RobustScaler` instead of `StandardScaler`). Decrease $C$
to reduce the influence of any single point. Investigate large-margin-violation points.

### 6.8 Missing Values Not Supported

**Mechanism:** LIBSVM and LIBLINEAR require a complete input matrix.

**Mitigation:** Impute before fitting with `SimpleImputer` or `KNNImputer` inside a Pipeline.

### 6.9 Feature Engineering Responsibility Falls on You

**Mechanism:** Unlike gradient boosting (which can learn feature interactions via tree splits)
or neural networks (which learn feature representations), a kernel SVM with a fixed kernel
learns boundaries in a space determined by your choice of kernel. If you choose `kernel='rbf'`
but the true pattern requires features that haven't been extracted, the SVM can't invent them.

**Example:** For time-series classification, using raw pixel values as features for an SVM is
suboptimal. The model needs you to engineer frequency-domain features (FFT coefficients),
statistics (mean, variance, autocorrelation), or domain-specific features first. Then the
RBF kernel over these hand-engineered features can be highly effective.

**Mitigation:** For structured inputs (images, text, time series), extract appropriate features
first. For raw inputs, consider neural networks or tree-based methods that learn feature
representations automatically.

### 6.10 Instability in the Presence of Duplicate Features

**Mechanism:** If you have two identical or nearly identical features, the SVM will split
weight between them in a way that's numerically unstable and non-unique. The resulting
`coef_` (for LinearSVC) becomes hard to interpret, and the kernel matrix may become near-singular.

**Detection:** High variance in `coef_` across different random states. Near-zero `svc.score_samples()` variance across test points.

**Mitigation:** Run PCA or remove duplicate/near-duplicate features before fitting. With
`LinearSVC(penalty='l1')`, the L1 penalty automatically selects one from a group of correlated
features — a natural deduplication.

### Failure Mode Diagnostic Checklist

When your SVM performs poorly, work through this checklist in order:

```python
# Diagnostic checklist — run after any poor SVM performance

# 1. Are features scaled?
print(f"Feature means:   {X_train.mean(axis=0)[:5]}")   # should be ~0
print(f"Feature stds:    {X_train.std(axis=0)[:5]}")    # should be ~1
# Fix: add StandardScaler() to pipeline

# 2. Are there missing values?
print(f"NaN count: {np.isnan(X_train).sum()}")
# Fix: add SimpleImputer to pipeline

# 3. What's the support vector ratio?
svc.fit(X_train_sc, y_train)
sv_ratio = svc.n_support_.sum() / len(X_train)
print(f"SV ratio: {sv_ratio:.1%}")
# If > 50%: C too small or kernel poorly tuned
# If < 5%: C too large (overfit) for noisy data

# 4. Training vs validation accuracy gap
train_acc = svc.score(X_train_sc, y_train)
val_acc   = cross_val_score(pipe, X_train, y_train, cv=5).mean()
print(f"Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}")
# Large gap → overfitting → reduce C or gamma
# Both low → underfitting → increase C, try different kernel

# 5. Did the optimizer converge?
# Look for ConvergenceWarning in output
# Fix: increase max_iter, decrease tol, scale features

# 6. Is C/gamma on a good range?
# Run the 2D heatmap from §7.3 to visualize the search space
```

---

## §7 — Diagnostic Plots & Evaluation

### 7.1 Key Metrics

**For classification (SVC):**
- **Accuracy:** Good when classes are balanced.
- **ROC-AUC:** Use `decision_function` for AUC: `roc_auc_score(y_test, svc.decision_function(X_test))` for binary; `decision_function_shape='ovr'` for multiclass macro-AUC.
- **F1 / Precision-Recall AUC:** Essential when classes are imbalanced.

**For regression (SVR):**
- **MAE / RMSE:** Standard. Points within ε-tube contribute zero loss by design.
- **R²:** Global goodness of fit.
- **SVR-specific:** Track support vector ratio — if > 80% are SVs, ε is too small or C too large.

### 7.2 Learning Curves

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10, gamma=0.01))])

train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='blue', label='Training accuracy')
plt.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                 train_scores.mean(1) + train_scores.std(1), alpha=0.15, color='blue')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='red', label='Validation accuracy')
plt.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                 val_scores.mean(1) + val_scores.std(1), alpha=0.15, color='red')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('SVM Learning Curve')
plt.legend()
plt.tight_layout()
plt.show()
```

**Reading the plot:**
- High train accuracy, low val accuracy, large gap → Overfitting → decrease $C$
- Low both, small gap → Underfitting → increase $C$, use more complex kernel, add features
- Both converge high → Good fit; more data won't help much

### 7.3 C-Gamma Validation Heatmap

```python
from sklearn.model_selection import GridSearchCV
import seaborn as sns

param_grid = {'svc__C': np.logspace(-2, 3, 6), 'svc__gamma': np.logspace(-4, 1, 6)}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

scores = grid.cv_results_['mean_test_score'].reshape(6, 6)
plt.figure(figsize=(8, 6))
sns.heatmap(
    scores, annot=True, fmt='.3f',
    xticklabels=[f'{g:.0e}' for g in np.logspace(-4, 1, 6)],
    yticklabels=[f'{c:.0e}' for c in np.logspace(-2, 3, 6)],
    cmap='YlOrRd'
)
plt.xlabel('gamma')
plt.ylabel('C')
plt.title('C-Gamma Validation Accuracy Heatmap')
plt.tight_layout()
plt.show()
```

**Reading the heatmap:** The optimal region forms a diagonal band (top-right = overfit; bottom-left = underfit). A plateau in the middle of the diagonal indicates a robust, well-tuned model.

### 7.4 Decision Boundary Visualization (2D)

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

svc = SVC(kernel='rbf', C=5.0, gamma=0.5)
svc.fit(X_sc, y)

x_min, x_max = X_sc[:, 0].min() - 0.5, X_sc[:, 0].max() + 0.5
y_min, y_max = X_sc[:, 1].min() - 0.5, X_sc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap='RdBu', alpha=0.6)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k',
            linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
plt.scatter(X_sc[y==0, 0], X_sc[y==0, 1], c='red',  edgecolors='k', s=40)
plt.scatter(X_sc[y==1, 0], X_sc[y==1, 1], c='blue', edgecolors='k', s=40)
sv = svc.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none', edgecolors='k',
            linewidths=2, label=f'Support vectors (n={len(sv)})')
plt.legend()
plt.title(f'Decision Boundary  C={svc.C}, γ={svc.gamma}')
plt.tight_layout()
plt.show()
```

**Reading the plot:** The solid black line is the decision boundary ($f = 0$). Dashed lines are
margin boundaries ($f = \pm 1$). Circled points are support vectors — they lie on or within the
margin. A wide gap between dashed lines = good margin.

### 7.5 SVR Residual Plot

```python
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

pipe_svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR(C=10, epsilon=0.1, gamma='scale'))])
pipe_svr.fit(X_train, y_train)
y_pred = pipe_svr.predict(X_test)
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred, y_test, alpha=0.4, s=20)
lims = [min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max())]
axes[0].plot(lims, lims, 'r--', lw=2)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('Predicted vs Actual')

axes[1].scatter(y_pred, residuals, alpha=0.4, s=20)
axes[1].axhline(0, color='r', linestyle='--', lw=2)
eps = pipe_svr.named_steps['svr'].epsilon
axes[1].axhline(eps,  color='g', linestyle=':', label=f'+ε={eps}')
axes[1].axhline(-eps, color='g', linestyle=':', label=f'-ε={eps}')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residual')
axes[1].set_title('SVR Residuals (green = ε-tube boundary)')
axes[1].legend()
plt.tight_layout()
plt.show()
```

**Reading the SVR residual plot:** Points within ε (between green lines) have zero hinge loss
and are not support vectors. Points outside are support vectors driving the fit. Systematic
patterns in residuals indicate missing nonlinearity or wrong kernel/ε.

### 7.6 Support Vector Count vs C

```python
import matplotlib.pyplot as plt
import numpy as np

C_values = np.logspace(-2, 3, 15)
n_sv_list, val_acc_list = [], []

for C in C_values:
    svc_c = SVC(kernel='rbf', C=C, gamma=0.01)
    svc_c.fit(X_train_sc, y_train)
    n_sv_list.append(svc_c.n_support_.sum())
    val_acc_list.append(svc_c.score(X_val_sc, y_val))

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.semilogx(C_values, n_sv_list, 'b-o', label='Support vectors')
ax2.semilogx(C_values, val_acc_list, 'r-s', label='Val accuracy')
ax1.set_xlabel('C'); ax1.set_ylabel('Number of Support Vectors', color='blue')
ax2.set_ylabel('Validation Accuracy', color='red')
plt.title('Support Vectors vs C')
plt.tight_layout()
plt.show()
```

**Reading:** As $C$ decreases, SVs increase (softer margin). Ideal: few SVs + high accuracy.
If accuracy plateaus but SVs keep growing, you've passed the optimal $C$.

### 7.9 Probability Calibration Curve

Even when `SVC(probability=True)` is used, the calibration (alignment between predicted
probabilities and actual frequencies) should be verified:

```python
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Compare calibration: uncalibrated, Platt scaling, isotonic regression
fig, ax = plt.subplots(figsize=(7, 6))

# Baseline (raw decision function transformed via sigmoid — just for illustration)
# SVC with probability=True (Platt scaling)
svc_platt = CalibratedClassifierCV(SVC(kernel='rbf', C=1.0), cv=5, method='sigmoid')
svc_platt.fit(X_train_sc, y_train)
CalibrationDisplay.from_estimator(svc_platt, X_test_sc, y_test,
                                  n_bins=10, ax=ax, name='Platt scaling')

# Isotonic regression calibration
svc_iso = CalibratedClassifierCV(SVC(kernel='rbf', C=1.0), cv=5, method='isotonic')
svc_iso.fit(X_train_sc, y_train)
CalibrationDisplay.from_estimator(svc_iso, X_test_sc, y_test,
                                  n_bins=10, ax=ax, name='Isotonic')

ax.set_title('SVM Probability Calibration')
plt.tight_layout()
plt.show()
```

**Reading the calibration plot:** The diagonal (dashed line) represents perfect calibration —
a predicted probability of 0.7 means 70% of such predictions are positive. Points above the
diagonal: underconfident (predicted 0.5, actual 0.7 → model is more confident than it says).
Points below: overconfident. For well-calibrated SVMs, the Platt sigmoid should follow the
diagonal closely. Isotonic regression is more flexible but needs more calibration data (use when
the test set has > 1,000 samples per class).

### 7.10 ROC Curve and Precision-Recall Curve

```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve (use decision_function, not predict_proba, for kernel SVMs)
y_score = svc_tuned.decision_function(X_test_sc)  # binary: shape (n,)
RocCurveDisplay.from_predictions(y_test, y_score, ax=axes[0], name='SVC (RBF)')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_title('ROC Curve — SVM')

# Precision-Recall Curve (critical for imbalanced datasets)
PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=axes[1], name='SVC (RBF)')
axes[1].set_title('Precision-Recall Curve — SVM')

plt.tight_layout()
plt.show()
```

**When to use ROC-AUC vs PR-AUC:**
- **ROC-AUC**: Default when classes are roughly balanced. Measures ability to separate classes across all thresholds.
- **PR-AUC**: When positive class is rare (< 10%). ROC-AUC can look deceptively high even with poor precision on the minority class. PR-AUC reveals the actual precision vs. recall trade-off.

For fraud detection (0.1% positive rate), a model with ROC-AUC = 0.95 might have PR-AUC = 0.30 — telling a very different story about real-world utility.

### 7.11 Threshold Optimization

SVMs produce a decision function $f(\mathbf{x})$, not a probability. The default classification
threshold is $f(\mathbf{x}) = 0$. For imbalanced problems, the optimal threshold may not be 0:

```python
from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np

# Get decision function scores
y_score = svc_tuned.decision_function(X_test_sc)

# Find threshold that maximizes F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores[:-1])]
best_f1 = f1_scores.max()
print(f"Best threshold: {best_threshold:.4f}")
print(f"Best F1:        {best_f1:.4f}")

# Apply optimal threshold
y_pred_optimal = (y_score >= best_threshold).astype(int)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_optimal))
```

> 🏆 **Best practice:** For imbalanced binary classification with an SVM, always compute the
> optimal threshold on a validation set rather than using 0. The F1-optimal threshold can
> substantially improve precision and recall balance compared to the default.

---

## §8 — Explainability & Interpretable AI

SVMs present an explainability challenge: for kernel SVMs, the feature weights live in an
implicit high-dimensional space making direct interpretation impossible. For LinearSVC, `coef_`
gives exact feature weights. Choose your approach based on your SVM variant.

### 8.1 LinearSVC — Direct Weight Inspection

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([('scaler', StandardScaler()), ('lsvc', LinearSVC(C=1.0, max_iter=5000))])
pipe.fit(X_train, y_train)

# For binary classification:
coef = pipe.named_steps['lsvc'].coef_[0]   # shape (n_features,)

top_k = 20
top_idx = np.argsort(np.abs(coef))[-top_k:]
plt.figure(figsize=(10, 6))
colors = ['red' if c < 0 else 'steelblue' for c in coef[top_idx]]
plt.barh(range(top_k), coef[top_idx], color=colors)
plt.yticks(range(top_k), [feature_names[i] for i in top_idx])
plt.xlabel('Coefficient (positive = class +1, negative = class -1)')
plt.title('LinearSVC Feature Coefficients')
plt.axvline(0, color='k', linestyle='--', lw=0.8)
plt.tight_layout()
plt.show()
```

> ⚠️ **Pitfall:** `coef_` gives weights in the scaled feature space. To compare with raw-unit
> importances, divide by the scaler's `scale_`: `raw_coef = coef / scaler.scale_`. For ranking
> purposes, scaled coefficients are usually more meaningful.

### 8.2 SHAP for Kernel SVMs — KernelExplainer

For kernel SVMs (RBF, poly), `shap.KernelExplainer` is the only option for SHAP values:

```python
import shap

# Use a small background (SHAP runs n_background × n_explain × n_features evaluations)
background = shap.sample(X_train_sc, 100)
explainer = shap.KernelExplainer(
    lambda X_sc: pipe_svc.named_steps['svc'].decision_function(X_sc),
    background
)

# Keep n_explain VERY small — this is slow
shap_values = explainer.shap_values(X_test_sc[:10])
shap.summary_plot(shap_values, X_test_sc[:10], feature_names=feature_names)
```

> ⚠️ **Pitfall:** KernelExplainer with RBF SVM is confirmed extremely slow (GitHub issue #3747).
> Each test sample requires $n_{\text{background}} \times 2^p$ (or Monte Carlo approximation)
> SVM evaluations. Keep `n_explain` ≤ 20 and `n_background` ≤ 100. For production explainability
> at scale, use the Nystroem + LinearSVC approach below.

### 8.3 SHAP for LinearSVC — LinearExplainer (Fast)

```python
import shap
from sklearn.svm import LinearSVC

pipe = Pipeline([('scaler', StandardScaler()), ('lsvc', LinearSVC(C=1.0, max_iter=5000))])
pipe.fit(X_train, y_train)
X_train_sc = pipe.named_steps['scaler'].transform(X_train)
X_test_sc  = pipe.named_steps['scaler'].transform(X_test)

# LinearExplainer: instantaneous for linear models
explainer = shap.LinearExplainer(
    pipe.named_steps['lsvc'],
    X_train_sc,
    feature_perturbation='interventional'
)
shap_values = explainer.shap_values(X_test_sc)

# Summary plot
shap.summary_plot(shap_values, X_test_sc, feature_names=feature_names)

# Waterfall for a single prediction
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test_sc[0],
    feature_names=feature_names
))
```

> 🏆 **Best practice:** For interpretability + SVM performance, use Nystroem + LinearSVC as a
> drop-in replacement for kernel SVC. This pattern gives you:
> 1. Near-identical accuracy for $n < 500{,}000$
> 2. Interpretable `coef_` in transformed Nystroem space
> 3. Fast SHAP via `LinearExplainer`
> 4. ~100× faster SHAP than KernelExplainer

### 8.4 Permutation Importance — Model-Agnostic

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    pipe_svc, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)
sorted_idx = result.importances_mean.argsort()[::-1][:20]

plt.figure(figsize=(10, 6))
plt.boxplot(
    [result.importances[i] for i in sorted_idx],
    vert=False,
    labels=[feature_names[i] for i in sorted_idx]
)
plt.xlabel('Accuracy decrease')
plt.title('Permutation Feature Importance (SVM)')
plt.tight_layout()
plt.show()
```

> ⚠️ **Pitfall:** Permutation importance measured on the **training set** measures memorization,
> not genuine importance. Always compute it on a **held-out test set**.

### 8.5 Partial Dependence Plots (PDP) and ICE Plots

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    pipe_svc, X_train,
    features=[0, 1, (0, 1)],   # individual features + 2D interaction
    feature_names=feature_names,
    kind='both',                # 'average' (PDP) + 'individual' (ICE)
    subsample=100,
    n_cols=3
)
plt.suptitle('PDP + ICE — SVM', y=1.02)
plt.tight_layout()
plt.show()
```

**PDP:** Marginal effect of a feature averaged over all others. **ICE:** Individual-level effects.
For RBF SVMs, PDPs are typically smooth S-shaped curves. Spread in ICE curves indicates
heterogeneous feature effects across subgroups.

### 8.6 Global vs Local Explanations — Summary

| Method | Speed | Scope | SVM Variant |
|---|---|---|---|
| `coef_` (direct) | Instant | Global | LinearSVC only |
| `shap.LinearExplainer` | Fast | Local + global | LinearSVC only |
| `shap.KernelExplainer` | Very slow | Local + global | Any SVM |
| Permutation importance | Moderate | Global | Any SVM |
| PDP/ICE | Moderate | Global | Any SVM (via pipeline) |
| Nystroem + LinearSVC | Fast | Local + global | Approx. any kernel SVM |

---

## §9 — Complete Python Worked Example

We'll work through two complete end-to-end examples:
1. **Classification:** `load_digits` (1797 samples, 64 features, 10 classes)
2. **Regression:** `fetch_california_housing` (first 2000 samples for SVR tractability)

### Part A: SVM Classification on Digits

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score)
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & Brief EDA ──────────────────────────────────────────────────────
digits = load_digits()
X, y = digits.data, digits.target
feature_names = [f'pixel_{i}' for i in range(64)]

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution (balanced): {np.bincount(y)}")
print(f"Feature range: [{X.min():.1f}, {X.max():.1f}]")

# Visualize one digit per class
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y == i)[0][0]
    ax.imshow(digits.images[idx], cmap='gray_r')
    ax.set_title(f'Digit: {i}')
    ax.axis('off')
plt.suptitle('Sample Digits from Each Class')
plt.tight_layout()
plt.show()

# ── 2. Train/Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 3. Baseline SVC (no tuning) ──────────────────────────────────────────────
pipe_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])
pipe_svc.fit(X_train, y_train)
print(f"\nBaseline SVC accuracy: {pipe_svc.score(X_test, y_test):.4f}")

svc_base = pipe_svc.named_steps['svc']
print(f"Support vectors: {svc_base.n_support_.sum()} / {len(X_train)} "
      f"({svc_base.n_support_.sum()/len(X_train):.1%})")

# ── 4. Joint C/gamma Grid Search ─────────────────────────────────────────────
param_grid = {
    'svc__C':     np.logspace(-1, 3, 5),    # [0.1, 1, 10, 100, 1000]
    'svc__gamma': np.logspace(-4, 0, 5),    # [1e-4, 1e-3, 0.01, 0.1, 1.0]
}
grid_search = GridSearchCV(
    pipe_svc, param_grid,
    cv=5, scoring='accuracy', n_jobs=-1, verbose=0,
    return_train_score=True
)
grid_search.fit(X_train, y_train)
print(f"\nBest params:  {grid_search.best_params_}")
print(f"Best CV acc:  {grid_search.best_score_:.4f}")

# C-gamma heatmap
scores = grid_search.cv_results_['mean_test_score'].reshape(5, 5)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    scores, annot=True, fmt='.3f', cmap='YlOrRd',
    xticklabels=[f'{v:.0e}' for v in np.logspace(-4, 0, 5)],
    yticklabels=[f'{v:.0e}' for v in np.logspace(-1, 3, 5)],
    ax=ax
)
ax.set_xlabel('gamma'); ax.set_ylabel('C')
ax.set_title('5-Fold CV Accuracy: C × gamma Grid — Digits')
plt.tight_layout()
plt.show()

# ── 5. Evaluate Best Model ───────────────────────────────────────────────────
best_model = grid_search.best_estimator_
y_pred  = best_model.predict(X_test)
y_score = best_model.decision_function(X_test)   # shape (n, K)

print(f"\nTest accuracy (tuned): {best_model.score(X_test, y_test):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

roc_auc_macro = roc_auc_score(y_test, y_score, multi_class='ovr', average='macro')
print(f"Macro ROC-AUC (OvR): {roc_auc_macro:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(10)]).plot(
    cmap='Blues', colorbar=False, ax=ax)
ax.set_title('Tuned SVC Confusion Matrix — Digits')
plt.tight_layout()
plt.show()

# ── 6. Support Vector Analysis ───────────────────────────────────────────────
svc_best = best_model.named_steps['svc']
print(f"\nSupport vectors per class: {svc_best.n_support_}")
print(f"Total SVs: {svc_best.n_support_.sum()} ({svc_best.n_support_.sum()/len(X_train):.1%})")
print(f"Best C={svc_best.C}, gamma={svc_best.gamma}")

# ── 7. Learning Curve ────────────────────────────────────────────────────────
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5, scoring='accuracy', n_jobs=-1
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores.mean(1), 'o-', color='blue', label='Train')
plt.fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1),
                 train_scores.mean(1)+train_scores.std(1), alpha=0.15, color='blue')
plt.plot(train_sizes, val_scores.mean(1), 'o-', color='red',  label='Validation')
plt.fill_between(train_sizes, val_scores.mean(1)-val_scores.std(1),
                 val_scores.mean(1)+val_scores.std(1), alpha=0.15, color='red')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('SVM Learning Curve — Digits')
plt.legend()
plt.tight_layout()
plt.show()

# ── 8. Permutation Importance ────────────────────────────────────────────────
print("\nComputing permutation importance (~30 seconds)...")
result = permutation_importance(
    best_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
)
sorted_idx = result.importances_mean.argsort()[::-1][:20]

plt.figure(figsize=(10, 5))
plt.boxplot([result.importances[i] for i in sorted_idx], vert=False,
            labels=[feature_names[i] for i in sorted_idx])
plt.xlabel('Accuracy decrease')
plt.title('Top 20 Pixel Importances (Permutation) — RBF SVC')
plt.tight_layout()
plt.show()

# Visualize importance as 8×8 pixel heatmap
importance_image = result.importances_mean.reshape(8, 8)
plt.figure(figsize=(5, 4))
plt.imshow(importance_image, cmap='hot', interpolation='nearest')
plt.colorbar(label='Mean accuracy decrease')
plt.title('Pixel Importance Heatmap — RBF SVC')
plt.tight_layout()
plt.show()

# ── 9. SHAP via Nystroem + LinearSVC ─────────────────────────────────────────
# Approximate the RBF SVM for fast, interpretable SHAP
pipe_approx = Pipeline([
    ('scaler', StandardScaler()),
    ('nystroem', Nystroem(kernel='rbf', gamma=svc_best.gamma,
                          n_components=300, random_state=42)),
    ('lsvc', LinearSVC(C=svc_best.C, max_iter=10000, random_state=42))
])
pipe_approx.fit(X_train, y_train)
approx_acc = pipe_approx.score(X_test, y_test)
print(f"\nNystroem+LinearSVC test accuracy: {approx_acc:.4f}")

X_train_nys = pipe_approx[:-1].transform(X_train)
X_test_nys  = pipe_approx[:-1].transform(X_test)
lsvc = pipe_approx.named_steps['lsvc']

# LinearExplainer — fast, no perturbations
explainer = shap.LinearExplainer(lsvc, X_train_nys)
shap_values = explainer.shap_values(X_test_nys[:50])

# Summary plot (class 0 vs rest for illustration)
sv_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
plt.figure(figsize=(10, 5))
shap.summary_plot(
    sv_to_plot, X_test_nys[:50],
    feature_names=[f'nys_{i}' for i in range(X_test_nys.shape[1])],
    show=False
)
plt.title('SHAP Summary — Nystroem+LinearSVC (class 0 vs rest)')
plt.tight_layout()
plt.show()
print("SHAP analysis complete.")
```

### Part B: SVR on California Housing (Subset)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data (subset for SVR tractability) ───────────────────────────────
housing = fetch_california_housing()
X_full, y_full = housing.data, housing.target
feature_names = list(housing.feature_names)

# First 2000 samples — SVR is O(n^3); full 20k would take many hours
X, y = X_full[:2000], y_full[:2000]
print(f"Dataset: {X.shape}, target range: [{y.min():.2f}, {y.max():.2f}] (×$100k)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Baseline SVR ──────────────────────────────────────────────────────────
pipe_svr = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'))
])
pipe_svr.fit(X_train, y_train)
y_pred_base = pipe_svr.predict(X_test)
print(f"Baseline SVR — RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_base)):.4f}, "
      f"R²: {r2_score(y_test, y_pred_base):.4f}")

# ── 3. Tune C and epsilon ────────────────────────────────────────────────────
param_grid_svr = {
    'svr__C':       [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.05, 0.1, 0.5],
    'svr__gamma':   ['scale', 0.01, 0.1],
}
grid_svr = GridSearchCV(pipe_svr, param_grid_svr, cv=5, scoring='r2', n_jobs=-1)
grid_svr.fit(X_train, y_train)
print(f"\nBest SVR params: {grid_svr.best_params_}")
print(f"Best CV R²:      {grid_svr.best_score_:.4f}")

# ── 4. Evaluate Best SVR ─────────────────────────────────────────────────────
best_svr = grid_svr.best_estimator_
y_pred   = best_svr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"\nTuned SVR — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

svr = best_svr.named_steps['svr']
print(f"Support vectors: {len(svr.support_)} ({len(svr.support_)/len(X_train):.1%})")

# ── 5. Diagnostic Plots ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].scatter(y_pred, y_test, alpha=0.4, s=15, color='steelblue')
lims = [min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max())]
axes[0].plot(lims, lims, 'r--', lw=2)
axes[0].set_xlabel('Predicted (×$100k)'); axes[0].set_ylabel('Actual')
axes[0].set_title(f'Predicted vs Actual (R²={r2:.3f})')

residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.4, s=15, color='steelblue')
axes[1].axhline(0, color='r', lw=2, linestyle='--')
eps = svr.epsilon
axes[1].axhline( eps, color='g', lw=1.5, linestyle=':', label=f'+ε={eps}')
axes[1].axhline(-eps, color='g', lw=1.5, linestyle=':', label=f'-ε={eps}')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residual')
axes[1].set_title('SVR Residuals'); axes[1].legend()

axes[2].hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[2].axvline(0, color='r', lw=2, linestyle='--')
axes[2].set_xlabel('Residual'); axes[2].set_ylabel('Count')
axes[2].set_title(f'Residual Distribution (RMSE={rmse:.3f})')

plt.suptitle('SVR Diagnostics — California Housing (n=2000 subset)', y=1.02)
plt.tight_layout()
plt.show()

# ── 6. Permutation Importance ────────────────────────────────────────────────
result = permutation_importance(best_svr, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()[::-1]

plt.figure(figsize=(8, 5))
plt.boxplot([result.importances[i] for i in sorted_idx], vert=False,
            labels=[feature_names[i] for i in sorted_idx])
plt.xlabel('R² decrease')
plt.title('SVR Feature Importance (Permutation) — California Housing')
plt.tight_layout()
plt.show()

# ── 7. Partial Dependence Plots ──────────────────────────────────────────────
top3 = sorted_idx[:3].tolist()
fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(
    best_svr, X_train, features=top3,
    feature_names=feature_names, kind='average', n_cols=3, ax=ax
)
plt.suptitle('Partial Dependence — Top 3 SVR Features', y=1.02)
plt.tight_layout()
plt.show()

# ── 8. Final Interpretation ──────────────────────────────────────────────────
print("\n" + "="*60)
print("Final Model Interpretation")
print("="*60)
print(f"Best SVR: C={svr.C}, ε={svr.epsilon}, γ={svr.gamma}")
print(f"Test R²: {r2:.4f} (explains {r2*100:.1f}% of variance in house prices)")
print(f"Test RMSE: ${rmse*100:.2f}k median error")
print(f"SVs: {len(svr.support_)}/{len(X_train)} ({len(svr.support_)/len(X_train):.1%})")
print(f"Most important feature: {feature_names[sorted_idx[0]]}")
print(f"  Dropping it reduces R² by {result.importances_mean[sorted_idx[0]]:.4f}")
```

### Interpretation of Results

After tuning, the RBF SVC on digits achieves approximately 98–99% accuracy — near ceiling
performance on this clean 10-class dataset — with roughly 10–15% of training points becoming
support vectors. The confusion matrix typically shows 0 errors on most digits; confusion occurs
primarily between 4/9 and 3/8, which look similar in 8×8 pixel format. The pixel importance
heatmap reveals that interior pixels (rows 2–6) carry more discriminative information than edge
pixels, consistent with how handwritten digits are structured.

The SVR on the 2000-sample California Housing subset achieves R² ≈ 0.72–0.78. Permutation
importance consistently ranks `MedInc` (median income) as the strongest predictor, followed by
latitude/longitude (geographic location) and `AveOccup` (average occupancy). The SVR residual
plot shows systematic underprediction for very high-value homes (above $4–5 × $100k), suggesting
the RBF kernel's smooth interpolation misses the sharp price discontinuities at the luxury end
of the market. This is a genuine failure mode of SVR that tree-based models handle better.

---

## §10 — When to Use Support Vector Machines (Decision Guide)

### Primary Decision Flowchart

```
Start: I have a supervised learning problem.
│
├── Is n > 100,000 samples?
│   ├── YES → Do NOT use kernel SVM.
│   │          Options: LinearSVC (linear), GBM/neural net (nonlinear)
│   │          Exception: ThunderSVM/cuML on GPU if you have CUDA hardware
│   └── NO  → Continue ↓
│
├── Is n > 10,000 samples?
│   ├── YES → Is a linear boundary sufficient?
│   │         ├── YES → LinearSVC (fast, interpretable, LIBLINEAR)
│   │         └── NO  → Nystroem + LinearSVC (scalable approx kernel SVM)
│   └── NO (n ≤ 10,000) → Continue ↓
│
├── Is the boundary nonlinear (or unknown)?
│   ├── YES → SVC(kernel='rbf') — tune C and gamma jointly
│   └── NO  → LinearSVC or SVC(kernel='linear')
│
└── Is interpretability required?
    ├── YES → LinearSVC or Nystroem + LinearSVC (for coef_ and SHAP)
    └── NO  → SVC(kernel='rbf') — full kernel SVM performance
```

### Comprehensive Decision Table

| Criterion | SVM is a Good Choice | SVM is a Poor Choice |
|---|---|---|
| **Dataset size** | $n < 10{,}000$ | $n > 100{,}000$ |
| **Dimensionality** | High $p$ (text, genomics, images with few samples) | Very low $p$ with large $n$ — use tree models |
| **Feature type** | Numeric, continuous, scaled | Mostly categorical (use GBM) |
| **Missing values** | None (impute first) | Many missing values (GBM handles natively) |
| **Class balance** | Moderate imbalance (use `class_weight`) | Severe imbalance ≥1:100 |
| **Kernel structure** | Domain-specific kernel available | Purely tabular, no clear kernel motivation |
| **Probability output** | Acceptable overhead for Platt scaling | Real-time probability at high throughput |
| **Interpretability** | LinearSVC: fully interpretable | RBF kernel SVM: black box |
| **Training speed** | Hours to tune is acceptable | Online/streaming learning |
| **Inference speed** | Low-throughput serving | Millions of predictions per second |

### Head-to-Head: SVM vs Common Alternatives

| Algorithm | SVM Wins When... | Alternative Wins When... |
|---|---|---|
| SVM vs Logistic Regression | Nonlinear boundary, small $n$, custom kernel | Speed, probabilities, large $n$, interpretability |
| SVM vs Random Forest | Small $n$, very high $p$, clear kernel structure | Missing values, categorical features, large $n$ |
| SVM vs Gradient Boosting | Small $n$ (<5k), high-dimensional sparse features | Medium/large $n$, mixed types, auto handles scale |
| SVM vs Neural Network | Small/medium $n$, high $p$, no GPU available | $n > 100{,}000$, raw images/audio/text |
| SVM vs KNN | High $p$, nonlinear, sparse domains | Low $p$, exact nearest neighbors needed |
| LinearSVC vs Logistic Reg | Sparse text, L1 feature selection, hinge loss | When calibrated probabilities are critical |

### When SVMs Definitively Win

1. **Bioinformatics/genomics** ($p \gg n$): Protein structure prediction, gene expression with $n < 1{,}000$ samples and $p = 10{,}000+$ features. Maximum-margin regularization is theoretically and empirically optimal here.

2. **Text classification (linear SVM)**: For TF-IDF/bag-of-words features, `LinearSVC` consistently competes with logistic regression and beats it on smaller datasets. Faster than neural networks for the same quality on classical NLP benchmarks.

3. **Domains with precomputed kernels**: String kernels for proteins/DNA, graph kernels for molecular graphs, tree kernels for NLP parse trees. No other widely-used algorithm accepts custom similarity matrices this naturally.

4. **When a global optimum guarantee matters**: Clinical and regulatory applications where reproducibility is critical. The convexity guarantee means identical results every run.

5. **Small datasets with clear nonlinearity**: $n < 5{,}000$ with a complex boundary that trees overfit to — RBF SVM often finds the right smooth boundary.

### When to Definitively Avoid SVMs

1. **Streaming/online learning**: SVMs require batch training. Use `SGDClassifier(loss='hinge')` for online SVMs.
2. **$n > 100{,}000$ with nonlinear requirements**: Gradient boosting or neural networks.
3. **Mixed categorical features**: CatBoost, LightGBM, Random Forest handle these natively.
4. **High-throughput inference**: A linear model or compiled tree ensemble is faster.
5. **Label noise > 20%**: GBM with early stopping is more robust.

---

## §11 — Resources & Further Reading

### Original Papers (Primary Citations)

1. **Boser, Guyon & Vapnik (1992) — Hard-Margin SVM + Kernel Trick**
   Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992). A training algorithm for optimal margin classifiers. *Proc. COLT '92*, pp. 144–152. ACM.
   DOI: https://doi.org/10.1145/130385.130401
   Full text: https://dl.acm.org/doi/10.1145/130385.130401

2. **Cortes & Vapnik (1995) — Soft-Margin C-SVM (the canonical reference)**
   Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
   DOI: https://doi.org/10.1007/BF00994018
   Semantic Scholar PDF: https://www.semanticscholar.org/paper/Support-Vector-Networks-Cortes-Vapnik/52b7bf3ba59b31f362aa07f957f1543a29a4279e

3. **Platt (1998) — SMO Algorithm (makes sklearn SVC tractable)**
   Platt, J. C. (1998). Sequential minimal optimization: A fast algorithm for training support vector machines. *Microsoft Research Technical Report MSR-TR-98-14*.
   URL: https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/

4. **Drucker et al. (1997) — Support Vector Regression**
   Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A., & Vapnik, V. (1997). Support vector regression machines. *NIPS*, 9, 155–161.
   URL: https://proceedings.neurips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html

5. **Schölkopf et al. (2000) — ν-SVM**
   Schölkopf, B., Smola, A. J., Williamson, R. C., & Bartlett, P. L. (2000). New support vector algorithms. *Neural Computation*, 12(5), 1207–1245.

6. **Platt (1999) — Platt Scaling (calibrated probabilities)**
   Platt, J. C. (1999). Probabilistic outputs for support vector machines. *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74.

7. **Chang & Lin (2011) — LIBSVM**
   Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. *ACM TIST*, 2(3), 27:1–27:27.
   DOI: https://doi.org/10.1145/1961189.1961199 | Software: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

8. **Fan et al. (2008) — LIBLINEAR (backend for LinearSVC)**
   Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008). LIBLINEAR: A library for large linear classification. *JMLR*, 9, 1871–1874.
   URL: https://www.jmlr.org/papers/volume9/fan08a/fan08a.pdf

9. **Wen et al. (2018) — ThunderSVM**
   Wen, Z., Shi, J., Li, Q., He, B., & Chen, J. (2018). ThunderSVM: A fast SVM library on GPUs and CPUs. *JMLR*, 19, 1–5.
   URL: https://www.jmlr.org/papers/v19/17-740.html

10. **Rahimi & Recht (2007) — RBFSampler (Random Kitchen Sinks)**
    Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *NIPS*, 20.

### Books

11. **Schölkopf, B. & Smola, A. J. (2002). *Learning with Kernels*. MIT Press. ISBN: 978-0-262-19475-4**
    The definitive theoretical reference. Chapters 2–7 cover all SVM foundations (duality, kernels, regularization, VC bounds). Required reading for deep theory.
    MIT Press: https://mitpress.mit.edu/9780262536578/learning-with-kernels/

12. **Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.**
    Chapter 7 covers SVMs with clear derivations. Free PDF: https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/

13. **Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.**
    Chapter 12: SVMs connected to regularized classification theory. Free PDF: https://hastie.su.domains/ElemStatLearn/

### Official Documentation

14. **sklearn SVM User Guide (1.5.x):** https://scikit-learn.org/1.5/modules/svm.html
15. **sklearn SVC API reference:** https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html
16. **sklearn LinearSVC API reference:** https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html
17. **sklearn RBF Parameter Tuning Example:** https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
18. **sklearn Kernel Approximation docs:** https://scikit-learn.org/stable/modules/kernel_approximation.html
19. **LIBSVM Homepage:** https://www.csie.ntu.edu.tw/~cjlin/libsvm/
20. **LIBSVM Practical Guide (highly recommended for tuning heuristics):** https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
21. **ThunderSVM docs:** https://thundersvm.readthedocs.io/en/latest/
22. **SHAP KernelExplainer docs:** https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html

### Key GitHub Issues / Practical Notes

23. **SHAP KernelExplainer slowness with SVM:** https://github.com/shap/shap/issues/3747 — Community-confirmed performance bottleneck; contains practical alternatives.

### Uncertainties & Verification Flags

> ⚠️ **Verify before production use:**
>
> 1. **`probability=True` deprecation:** The dossier notes this may be deprecated in a future sklearn version in favor of `CalibratedClassifierCV`. Check sklearn 1.5.x changelog: https://scikit-learn.org/1.5/whats_new/v1.5.html
>
> 2. **ThunderSVM maintenance status:** Verify current Python/CUDA compatibility at https://github.com/Xtra-Computing/thundersvm/releases before using in production.
>
> 3. **cuML SVM API:** RAPIDS cuML may require specific CUDA driver versions. Verify at https://docs.rapids.ai/api/cuml/stable/api/#support-vector-machines
>
> 4. **`gamma='scale'` exact formula:** The formula $\gamma = 1/(p \cdot \text{Var}(\mathbf{X}))$ uses the total variance over the full training matrix. Verify the exact computation in `sklearn/svm/_base.py` if your scaling behavior seems unexpected.
>
> 5. **NuSVC `class_weight` support:** Verify whether `class_weight` is supported in `NuSVC` for your sklearn version by checking https://scikit-learn.org/1.5/modules/generated/sklearn.svm.NuSVC.html
>
> 6. **SHAP LinearExplainer with LinearSVC:** Verify `shap.LinearExplainer` accepts `LinearSVC` in SHAP 0.46.x. The API has evolved; check https://shap.readthedocs.io/en/latest/ for the current signature.

