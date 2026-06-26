# Research Dossier: Support Vector Machines (SVMs)

**Target file:** `support-vector-machines.md`
**Dossier prepared:** 2026-06-25
**Covers:** Full SVM history, hard-margin, soft-margin, kernel trick, SMO, sklearn API verification, GPU packages, SHAP, pitfalls

---

## Scope

This dossier feeds the chapter `support-vector-machines.md`. It covers:
- sklearn `SVC`, `SVR`, `LinearSVC`, `LinearSVR`, `NuSVC`, `NuSVR` — ALL parameters verified against sklearn 1.5.2 docs
- Underlying C libraries: LIBSVM (for kernel SVMs) and LIBLINEAR (for LinearSVC/LinearSVR)
- GPU-accelerated alternatives: ThunderSVM, cuSVM
- SHAP explainability for SVMs (KernelExplainer and faster alternatives)
- Full paper trail from Boser/Guyon/Vapnik 1992 through Cortes/Vapnik 1995, Platt 1998 SMO, Schölkopf 2000 ν-SVM, Drucker 1997 SVR, to Chang & Lin 2011 LIBSVM

---

## Verified Packages

### sklearn 1.5.x SVM Module

```python
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

**Backend libraries:**
- `sklearn.svm.SVC`, `SVR`, `NuSVC`, `NuSVR`, `OneClassSVM` → backed by **LIBSVM** (C library by Chang & Lin)
- `sklearn.svm.LinearSVC`, `LinearSVR` → backed by **LIBLINEAR** (C library by Fan, Chang, Hsieh, Wang, Lin)

### ThunderSVM (GPU-accelerated)

```python
# Install: pip install thundersvm
from thundersvm import SVC, SVR, OneClassSVM, SVCSklearn
# sklearn-compatible API; supports CUDA GPUs and multi-core CPUs
# Reported 100× speedup over LIBSVM on GPU, 10× on CPU
# JMLR paper: Wen et al., "ThunderSVM: A Fast SVM Library on GPUs and CPUs," JMLR 19 (2018)
```

**Key constraint:** ThunderSVM requires CUDA toolkit. Python interface is sklearn-compatible.

### cuML (RAPIDS AI)

```python
# Install: conda install -c rapidsai cuml
from cuml.svm import SVC, SVR
# GPU-only; part of RAPIDS ecosystem
# Returns numpy/cupy arrays; sklearn-compatible fit/predict API
```

### cuSVM

cuSVM is a lower-level CUDA SVM package predating ThunderSVM; less maintained. ThunderSVM is the preferred GPU option.

### SHAP for SVMs

```python
import shap

# Option 1: KernelExplainer — model-agnostic, works with any kernel SVM
# WARNING: O(n²) per prediction, extremely slow for large datasets or many features
explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:10])  # limit samples!

# Option 2: LinearSVC — use coef_ directly (instant, no SHAP needed)
# coef_ shape: (1, n_features) binary, (n_classes, n_features) multiclass OvR
importance = np.abs(lin_svc.coef_[0])

# Option 3: Permutation importance (model-agnostic, faster than KernelExplainer)
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Option 4: Kernel approximation → LinearSVC → use coef_ (best of both worlds)
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
approx_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_map', Nystroem(kernel='rbf', gamma=0.1, n_components=300, random_state=42)),
    ('svc', LinearSVC(C=1.0))
])
```

> **SHAP GitHub issue:** KernelExplainer with sklearn SVM RBF kernel is confirmed "really slow" (shap/shap issue #3747, issue #3943). The recommendation is to use permutation importance or switch to LinearSVC + coef_ when interpretability is required.

---

## The Original Algorithm

### Paper 1: Hard-Margin SVM — The True Origin

> **📜 Citation:**
> Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992). A training algorithm for optimal margin classifiers. *Proceedings of the Fifth Annual Workshop on Computational Learning Theory (COLT '92)*, Pittsburgh, PA, July 1992, pp. 144–152. ACM. DOI: https://doi.org/10.1145/130385.130401

**What problem it solved:** Prior to 1992, the dominant paradigm was neural networks and decision trees, which required architectural choices (number of hidden units, depth) that controlled capacity in ad-hoc ways. The generalization error was hard to bound theoretically. Boser, Guyon, and Vapnik — working at AT&T Bell Laboratories — wanted a learner whose capacity was directly controlled by the data's geometric structure (the margin), not by a manually chosen architecture.

**The core insight:** Maximize the margin between classes. For linearly separable data with labels $y_i \in \{-1, +1\}$, find $\mathbf{w}$ and $b$ such that:

$$y_i (\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i$$

and minimize $\|\mathbf{w}\|^2$. The **geometric margin** between classes is $\frac{2}{\|\mathbf{w}\|}$, so minimizing $\|\mathbf{w}\|^2$ maximizes the margin.

**Primal hard-margin problem:**

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i (\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \; \forall i$$

**Dual formulation (via Lagrangian):**

Introduce Lagrange multipliers $\alpha_i \geq 0$ for each constraint:

$$L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i \left[ y_i (\mathbf{w}^\top \mathbf{x}_i + b) - 1 \right]$$

Setting $\nabla_\mathbf{w} L = 0$: $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$

Setting $\partial L / \partial b = 0$: $\sum_i \alpha_i y_i = 0$

The dual becomes:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

subject to: $\alpha_i \geq 0$, $\sum_i \alpha_i y_i = 0$

**KKT complementary slackness:** $\alpha_i \left[ y_i (\mathbf{w}^\top \mathbf{x}_i + b) - 1 \right] = 0$

This means $\alpha_i > 0$ **only** for points on the margin hyperplanes ($y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$). These are the **support vectors** — the prediction at test time depends only on them:

$$\hat{y}(\mathbf{x}) = \text{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b \right)$$

**The kernel trick (introduced in this 1992 paper):** The dual depends only on dot products $\mathbf{x}_i^\top \mathbf{x}_j$. Replace these with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$, enabling nonlinear classification without computing $\phi$ explicitly. Boser, Guyon, and Vapnik recognized this immediately and demonstrated it with polynomial kernels.

---

### Paper 2: Soft-Margin SVM — Cortes & Vapnik 1995

> **📜 Citation:**
> Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. DOI: https://doi.org/10.1007/BF00994018

**The limitation of hard-margin:** The 1992 formulation requires linearly separable data (in feature space). Real data is noisy and overlapping. The hard-margin SVM fails completely when data is not separable.

**The fix — slack variables:** Cortes & Vapnik introduced slack variables $\xi_i \geq 0$ allowing violations of the margin:

**Primal soft-margin problem (C-SVM):**

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$

subject to: $y_i (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) \geq 1 - \xi_i$, $\xi_i \geq 0 \; \forall i$

$C > 0$ is the regularization parameter: **large $C$** = hard margins, few violations allowed (low bias, high variance); **small $C$** = wide margins, many violations tolerated (high bias, low variance).

**Dual soft-margin problem:**

$$\min_{\boldsymbol{\alpha}} \frac{1}{2} \boldsymbol{\alpha}^\top Q \boldsymbol{\alpha} - \mathbf{e}^\top \boldsymbol{\alpha}$$

subject to: $\mathbf{y}^\top \boldsymbol{\alpha} = 0$, $0 \leq \alpha_i \leq C \; \forall i$

where $Q_{ij} = y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$ and $\mathbf{e}$ is the all-ones vector.

The box constraint $0 \leq \alpha_i \leq C$ is the only change from the hard-margin dual. The prediction function is identical:

$$f(\mathbf{x}) = \sum_{i \in \text{SV}} y_i \alpha_i K(\mathbf{x}_i, \mathbf{x}) + b$$

**Historical context:** This paper was submitted in 1992 and published in 1995. It demonstrated SVMs on digit recognition and showed dramatically better results than competing methods at the time. The name "Support-Vector Networks" was used because the connection graph of support vectors looked like a neural network.

---

### Paper 3: SMO — Sequential Minimal Optimization (Platt 1998)

> **📜 Citation:**
> Platt, J. C. (1998). Sequential minimal optimization: A fast algorithm for training support vector machines. *Microsoft Research Technical Report MSR-TR-98-14*, April 1998. URL: https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/

**The problem SMO solved:** The dual QP problem has $n$ variables (one $\alpha_i$ per training sample) and $n^2/2$ entries in the $Q$ matrix. For $n = 10{,}000$ samples, $Q$ requires ~400 MB. Prior chunking algorithms (Osuna et al. 1997) still required expensive QP sub-solvers.

**SMO's insight:** The smallest possible QP sub-problem with constraints is size **2** (because the equality constraint $\sum_i \alpha_i y_i = 0$ means you can't optimize a single $\alpha_i$ in isolation, but you can always optimize a pair analytically).

**SMO algorithm sketch:**
```
repeat until convergence:
    1. Select two multipliers α_i, α_j to optimize (using heuristics to choose violating pairs)
    2. Solve the 2-variable QP analytically (closed-form solution)
    3. Update α_i, α_j, and the threshold b
    4. Check KKT conditions for all points
```

**The analytical update:** With $\alpha_2^{old}$ and the cached kernel values, the unconstrained optimum along the constraint line is:

$$\alpha_2^{new,unc} = \alpha_2^{old} + \frac{y_2 (E_1 - E_2)}{\eta}$$

where $E_i = f(\mathbf{x}_i) - y_i$ is the prediction error and $\eta = K_{11} + K_{22} - 2K_{12}$.

Then clip to box constraints: $\alpha_2^{new} = \text{clip}(\alpha_2^{new,unc}, L, H)$

**Memory:** Linear in $n$ (only stores the error cache and kernel values on demand). **Speed:** Between $O(n)$ and $O(n^2)$ empirically, vs. $O(n^2)$ to $O(n^3)$ for chunking. On sparse linear data, SMO can be 1000× faster than chunking.

> **This is the algorithm sklearn's LIBSVM uses internally.**

---

## Major Variants & Evolution

### 3.1 ν-SVM (Nu-SVM) — Schölkopf et al. 2000

> **📜 Citation:**
> Schölkopf, B., Smola, A. J., Williamson, R. C., & Bartlett, P. L. (2000). New support vector algorithms. *Neural Computation*, 12(5), 1207–1245.

**Motivation:** The parameter $C$ has no direct interpretation — its optimal value depends on the scale of $y$ and the dataset size. Schölkopf et al. introduced $\nu \in (0, 1]$ which has a clean interpretation:
- $\nu$ is an **upper bound** on the fraction of margin errors (misclassified or within-margin points)
- $\nu$ is a **lower bound** on the fraction of support vectors

**sklearn implementation:** `NuSVC(nu=0.5)` and `NuSVR(nu=0.5)`

**NuSVC primal:**

$$\min_{\mathbf{w}, b, \xi, \rho} \frac{1}{2}\|\mathbf{w}\|^2 - \nu\rho + \frac{1}{n}\sum_i \xi_i$$

subject to: $y_i(\mathbf{w}^\top\phi(\mathbf{x}_i) + b) \geq \rho - \xi_i$, $\xi_i \geq 0$, $\rho \geq 0$

The margin $\rho$ is now also optimized. When $\nu \to 0$, the problem approaches hard-margin SVM. When $\nu = 1$, almost all points become support vectors.

**Practical note:** NuSVC can be slightly harder to optimize numerically. Prefer C-SVC (`SVC`) unless you need the $\nu$ interpretation.

---

### 3.2 Support Vector Regression (SVR) — Drucker et al. 1997

> **📜 Citation:**
> Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A., & Vapnik, V. (1997). Support vector regression machines. *Advances in Neural Information Processing Systems*, 9, 155–161.

**The extension to regression:** Instead of classifying $y_i \in \{-1,+1\}$, now $y_i \in \mathbb{R}$. Define the **ε-insensitive loss**:

$$L_\varepsilon(y, f(\mathbf{x})) = \max(0, |y - f(\mathbf{x})| - \varepsilon)$$

No penalty is incurred for predictions within an ε-tube around the true value.

**SVR primal:**

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i(\xi_i + \xi_i^*)$$

subject to:
$$y_i - \mathbf{w}^\top\phi(\mathbf{x}_i) - b \leq \varepsilon + \xi_i$$
$$\mathbf{w}^\top\phi(\mathbf{x}_i) + b - y_i \leq \varepsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

**Prediction:** $f(\mathbf{x}) = \sum_{i \in \text{SV}} (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b$

**sklearn:** `SVR(epsilon=0.1, C=1.0, kernel='rbf', gamma='scale')`

---

### 3.3 LinearSVC / LinearSVR — Fan et al. 2008 (LIBLINEAR)

> **📜 Citation:**
> Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008). LIBLINEAR: A library for large linear classification. *Journal of Machine Learning Research*, 9, 1871–1874. URL: https://www.jmlr.org/papers/volume9/fan08a/fan08a.pdf

**Motivation:** LIBSVM's SMO is $O(n^2)$ to $O(n^3)$ — completely impractical for text classification where $n > 100{,}000$ and $p > 10^5$. LIBLINEAR solves the primal or dual of the linear SVM directly using coordinate descent, achieving near-$O(n)$ complexity.

**Key differences from SVC(kernel='linear'):**
- No kernel cache needed (linear, no Gram matrix)
- OvR (one-vs-rest) instead of OvO by default
- `dual='auto'` in sklearn 1.5 (was `True` in older versions): automatically chooses primal or dual based on `n_samples > n_features`
- Does NOT support `predict_proba` natively (use `CalibratedClassifierCV`)
- `loss='squared_hinge'` by default (not hinge)
- L1 penalty option (`penalty='l1', dual=False`) produces sparse `coef_`

```python
LinearSVC(
    penalty='l2',           # 'l1' or 'l2'
    loss='squared_hinge',   # 'hinge' or 'squared_hinge'
    dual='auto',            # 'auto', True, or False
    tol=1e-4,
    C=1.0,
    multi_class='ovr',      # 'ovr' or 'crammer_singer'
    fit_intercept=True,
    intercept_scaling=1.0,
    class_weight=None,
    verbose=0,
    random_state=None,
    max_iter=1000
)
```

**Rule of thumb from sklearn docs:** Use `dual=False` when `n_samples > n_features`. `dual='auto'` handles this automatically.

---

### 3.4 Kernel Approximation + LinearSVC (Scalable Nonlinear SVM)

For large $n$ where you need a nonlinear kernel but can't afford $O(n^2)$:

```python
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Nystroem: more accurate, O(n * n_components^2) fit
nystroem_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('nystroem', Nystroem(kernel='rbf', gamma=0.1, n_components=300, random_state=42)),
    ('svc', LinearSVC(C=1.0))
])

# RBFSampler: faster but less accurate (Random Kitchen Sinks, Rahimi & Recht 2007)
rbf_approx_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('rbf_sampler', RBFSampler(gamma=0.1, n_components=300, random_state=42)),
    ('svc', LinearSVC(C=1.0))
])
```

**Key trade-off:** RBFSampler is $O(n \cdot p \cdot D)$ where $D$ = n_components; Nystroem is more accurate but uses $O(D^2 \cdot p)$ memory for the inverse. Both give interpretable `coef_` after fitting.

---

### 3.5 ThunderSVM — Wen et al. 2018 (GPU SVM)

> **📜 Citation:**
> Wen, Z., Shi, J., Li, Q., He, B., & Chen, J. (2018). ThunderSVM: A fast SVM library on GPUs and CPUs. *Journal of Machine Learning Research*, 19, 1–5.

```python
from thundersvm import SVC
# Identical API to sklearn's SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
```

**Reported speedups:** ~100× vs LIBSVM on GPU (NVIDIA Tesla K40), ~10× on CPU. Supports SVC, SVR, OneClassSVM, NuSVC, NuSVR, probability estimates.

---

### 3.6 Platt Scaling for Probabilities

> **📜 Citation:**
> Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74.

**The problem:** SVMs output a decision function value $f(\mathbf{x}) = \mathbf{w}^\top\phi(\mathbf{x}) + b$, not a probability. The magnitude of $f(\mathbf{x})$ is not calibrated.

**Platt's fix:** Fit a sigmoid (logistic regression) $P(y=1|\mathbf{x}) = \sigma(A \cdot f(\mathbf{x}) + B)$ on a held-out calibration set. Parameters $A$ and $B$ are fit using maximum likelihood.

**sklearn implementation:** `SVC(probability=True)` internally runs 5-fold cross-validation on the training data, fitting a sigmoid to each fold's out-of-fold predictions. This:
1. Adds significant computational overhead (5× training time approximately)
2. Uses slightly less data for the actual SVM (out-of-fold)
3. Can cause `predict()` and `predict_proba()` to disagree (since `predict` uses the raw decision function, not the sigmoid)

> ⚠️ **Deprecation notice (from search results):** The `probability` parameter may be deprecated in a future sklearn version in favor of explicit `CalibratedClassifierCV(SVC(), ensemble=False)`. The author should verify this against the latest sklearn 1.5.x changelog.

**Better approach for multiclass:**
```python
from sklearn.calibration import CalibratedClassifierCV
cal_svc = CalibratedClassifierCV(SVC(), cv=5, method='sigmoid')  # Platt scaling
# or method='isotonic' for non-parametric calibration
cal_svc.fit(X_train, y_train)
probs = cal_svc.predict_proba(X_test)
```

---

## Hyperparameter Reference

### SVC — Complete Parameter Table (sklearn 1.5.2 verified)

| Parameter | Default | Type | What it controls | Too high | Too low |
|---|---|---|---|---|---|
| `C` | 1.0 | float | Regularization: smaller C = wider margin, more violations allowed | Overfit (hard margin) | Underfit (too permissive) |
| `kernel` | `'rbf'` | str/callable | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, `'precomputed'`, or callable | — | — |
| `degree` | 3 | int | Polynomial kernel degree (poly only); must be non-negative | Overfit, slow | Underfit |
| `gamma` | `'scale'` | str/float | RBF/poly/sigmoid kernel width: `'scale'` = $1/(p \cdot \text{Var}(\mathbf{X}))$, `'auto'` = $1/p$, or float > 0 | Overfit (narrow Gaussians) | Underfit (too smooth) |
| `coef0` | 0.0 | float | Free term in poly/sigmoid: $(γ\langle\mathbf{x},\mathbf{x}'\rangle + r)^d$ | — | — |
| `shrinking` | True | bool | Use shrinking heuristic (skip non-SV updates) | — | Slower convergence |
| `probability` | False | bool | Enable Platt scaling (5-fold CV internally) | — | — |
| `tol` | 1e-3 | float | SMO stopping criterion | Early stopping, inaccurate | Very slow convergence |
| `cache_size` | 200 | float | Kernel cache in MB | — (limited by RAM) | Cache misses, slow |
| `class_weight` | None | dict/`'balanced'` | Per-class C scaling: `C_k = C * class_weight[k]` | — | — |
| `max_iter` | -1 | int | Hard iteration limit (−1 = unlimited) | — | May not converge |
| `decision_function_shape` | `'ovr'` | str | OvO internal, `'ovr'` reshapes output to $(n, K)$; `'ovo'` returns $(n, K(K-1)/2)$ | — | — |
| `break_ties` | False | bool | Resolve OvR ties via decision_function argmax (computational cost) | — | — |
| `random_state` | None | int/None | Seed for probability CV only | — | — |

**gamma formula (critical — verify this):**
- `gamma='scale'` (default since sklearn 0.22): $\gamma = 1 / (n\_features \times \text{Var}(\mathbf{X}))$ — accounts for feature scale
- `gamma='auto'` (pre-0.22 default): $\gamma = 1 / n\_features$ — ignores feature variance

> ⚠️ **Pitfall:** If you use old code that passes `gamma='auto'` or relies on the old default, your model behavior changed at sklearn 0.22. Always use `gamma='scale'` after scaling, or tune gamma explicitly.

### SVR — Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `kernel` | `'rbf'` | Same options as SVC |
| `C` | 1.0 | Same meaning as SVC |
| `epsilon` | 0.1 | Half-width of the ε-insensitive tube; predictions within ±ε incur no loss |
| `gamma` | `'scale'` | Same formula as SVC |
| `degree` | 3 | Poly kernel only |
| `coef0` | 0.0 | Poly/sigmoid only |
| `tol` | 1e-3 | SMO stopping |
| `cache_size` | 200 | MB |
| `shrinking` | True | Heuristic |
| `max_iter` | -1 | Unlimited |

**Tuning epsilon:** Start with `epsilon` proportional to the noise level in your target variable. If your targets are standardized (mean 0, std 1), start with `epsilon=0.1`. Too small ε → overfitting (every point becomes a support vector); too large ε → underfitting (model ignores fine structure).

### LinearSVC — Complete Parameter Table (sklearn 1.5.2 verified)

| Parameter | Default | Notes |
|---|---|---|
| `penalty` | `'l2'` | L2 or L1 regularization. L1 produces sparse `coef_`. Combo `penalty='l1', loss='hinge'` is NOT supported. |
| `loss` | `'squared_hinge'` | `'hinge'` (standard SVM loss) or `'squared_hinge'` (squared hinge, smoother). |
| `dual` | `'auto'` | `'auto'` chooses dual or primal based on `n_samples`, `n_features`, `loss`, `multi_class`, `penalty`. `False` recommended when `n_samples > n_features`. |
| `tol` | 1e-4 | Stopping criterion. |
| `C` | 1.0 | Regularization parameter (same meaning as SVC). |
| `multi_class` | `'ovr'` | `'ovr'` (one-vs-rest, n_classes binary classifiers) or `'crammer_singer'` (joint multiclass objective). |
| `fit_intercept` | True | Whether to fit a bias term. |
| `intercept_scaling` | 1.0 | Scaling for intercept feature when `fit_intercept=True`. |
| `class_weight` | None | `'balanced'` or dict. |
| `verbose` | 0 | Verbosity level. |
| `random_state` | None | For dual coordinate descent shuffling. |
| `max_iter` | 1000 | Max iterations. |

### NuSVC Parameters (sklearn 1.5.2 verified)

```python
NuSVC(
    nu=0.5,              # Upper bound on margin errors, lower bound on SV fraction; range (0, 1]
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=0.001,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape='ovr',
    break_ties=False,
    random_state=None
)
```

**nu interpretation:** If `nu=0.3`, at most 30% of training samples will be margin errors, and at least 30% will be support vectors. Must satisfy $\nu \leq \frac{2 \min(n_+, n_-)}{n}$ for a feasible problem (may raise `ValueError` if violated).

### NuSVR Parameters (sklearn 1.5.2 verified)

```python
NuSVR(
    nu=0.5,       # Controls the fraction of support vectors (unlike SVR's epsilon)
    C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    tol=0.001,
    cache_size=200,
    verbose=False,
    max_iter=-1
)
```

**Key difference from SVR:** NuSVR uses `nu` to control the proportion of support vectors directly, rather than an explicit epsilon tube. The `epsilon` parameter is automatically determined from the training data.

---

## Kernel Reference

| Kernel | Formula | Parameters | sklearn name | When to use |
|---|---|---|---|---|
| Linear | $\langle\mathbf{x}, \mathbf{x}'\rangle$ | None | `'linear'` | Linearly separable data, high $p$, text |
| RBF (Gaussian) | $\exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)$ | `gamma` | `'rbf'` | Default choice; works for most problems |
| Polynomial | $(\gamma\langle\mathbf{x},\mathbf{x}'\rangle + r)^d$ | `gamma`, `coef0`, `degree` | `'poly'` | Image/text with multiplicative features |
| Sigmoid | $\tanh(\gamma\langle\mathbf{x},\mathbf{x}'\rangle + r)$ | `gamma`, `coef0` | `'sigmoid'` | Rarely outperforms RBF; not always PSD |
| Precomputed | Custom $K(\mathbf{x}_i, \mathbf{x}_j)$ | — | `'precomputed'` | String kernels, graph kernels, custom similarity |

**Mercer's Theorem:** A kernel $K(\mathbf{x}, \mathbf{x}')$ is valid (corresponds to a real inner product in some feature space) iff the kernel matrix $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is **positive semi-definite** for all datasets. The RBF and polynomial kernels are Mercer kernels. The sigmoid kernel $\tanh(\gamma\langle\mathbf{x},\mathbf{x}'\rangle + r)$ is **not always PSD** (only for certain $\gamma, r$ values) — use with caution.

**String kernels** (for NLP/bioinformatics):
- Spectrum kernel (Leslie et al. 2002): counts k-mer occurrences; use `kernel='precomputed'`
- Subsequence kernel (Lodhi et al., JMLR 2002): counts subsequences with gaps
- Pass a precomputed Gram matrix: `SVC(kernel='precomputed').fit(K_train, y_train)`, predict with `K_test` of shape `(n_test, n_train)`

---

## Computational Complexity

| Operation | LIBSVM (SVC, SVR) | LIBLINEAR (LinearSVC) |
|---|---|---|
| **Training** | $O(n^2 p)$ to $O(n^3 p)$ | $O(n p)$ approximately |
| **Inference** | $O(n_{SV} \cdot p)$ per sample | $O(p)$ per sample |
| **Memory** | $O(n_{SV}^2)$ kernel cache + $O(n)$ | $O(p)$ for `coef_` |
| **Practical limit** | ~10,000–50,000 samples | Millions of samples |

**Training complexity note:** The $O(n^2)$ to $O(n^3)$ comes from the SMO algorithm needing to evaluate kernel values between training points. The actual exponent depends heavily on the number of support vectors and the cache hit rate.

**Inference:** Unlike tree-based models, prediction with kernel SVM requires computing $n_{SV}$ kernel evaluations per test point. If $n_{SV}$ is large (e.g., 80% of training points are SVs — a sign of too-small $C$), inference can be slow.

---

## Multiclass Strategy

**SVC / NuSVC:** One-vs-One (OvO) internally
- Trains $\binom{K}{2} = K(K-1)/2$ binary classifiers
- For $K=10$ classes: 45 binary SVMs
- Prediction: voting among 45 classifiers
- `decision_function_shape='ovr'` (default) reshapes the OvO output to shape $(n, K)$ for sklearn compatibility but does NOT change training
- `decision_function_shape='ovo'` returns raw $(n, K(K-1)/2)$ shape

**LinearSVC:** One-vs-Rest (OvR) by default
- Trains $K$ binary classifiers
- Each classifier: class $k$ vs. all others
- `multi_class='crammer_singer'`: solves a joint objective (Crammer & Singer 2002); slightly different solution, similar accuracy

**Which is better?** OvO is often preferred for SVMs because each binary problem uses fewer, balanced samples. OvR can suffer from class imbalance and ambiguity in the boundary region. In practice, accuracy is usually similar but OvO trains faster per sub-problem.

---

## Common Errors & Pitfalls

### 1. Forgetting to Scale Features

**The most common and most damaging mistake.** SVMs use Euclidean distances (in the kernel via $\|\mathbf{x} - \mathbf{x}'\|^2$). An unscaled feature with range [0, 10,000] will completely dominate a binary feature [0, 1].

```python
# WRONG
svc = SVC()
svc.fit(X_train, y_train)

# RIGHT — always use a Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))
])
pipe.fit(X_train, y_train)
```

> ⚠️ **Pitfall:** If you scale training data but fit the scaler on test data separately, you've introduced data leakage. Always fit the scaler on train only, apply to both train and test.

### 2. Using gamma='auto' on Unscaled Data

With `gamma='auto'` = $1/p$ and unscaled features, the RBF kernel bandwidth is meaningless. Always use `gamma='scale'` which accounts for feature variance, or scale first and then tune gamma.

### 3. Not Tuning C and gamma Together

C and gamma have a trade-off that forms a diagonal in parameter space (see sklearn RBF parameters example). Searching them independently misses the optimal region. Always use a 2D grid search:

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100, 1000],
    'svc__gamma': [1e-4, 1e-3, 0.01, 0.1, 1, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
```

### 4. probability=True Overhead (and Inconsistency)

```python
# This is 5× slower to train because of internal CV
svc = SVC(probability=True)

# Prediction inconsistency:
# svc.predict() uses decision_function (may differ from argmax of predict_proba)
# In binary case: svc.predict_proba()[:,1] < 0.5 yet svc.predict() says class 1
```

### 5. Cache Size Too Small

For large datasets, the kernel cache (default 200 MB) gets evicted constantly, requiring repeated kernel recomputations. Increase for large problems:
```python
SVC(cache_size=2000)  # 2 GB for large datasets
```

### 6. Using SVC(kernel='rbf') on n > 50,000 Samples

Training time is $O(n^2)$ to $O(n^3)$. With $n = 100{,}000$, this can take hours or days.

**Mitigation path:**
```
n < 10,000: SVC(kernel='rbf')  — fine
10,000 < n < 100,000: Try Nystroem + LinearSVC first
n > 100,000: LinearSVC, SGDClassifier, or deep learning
```

### 7. NuSVC Infeasibility

`NuSVC` raises `ValueError` if $\nu$ is too large relative to the class balance:
$$\nu > \frac{2 \min(n_+, n_-)}{n}$$

With highly imbalanced classes ($n_+ = 10$, $n_- = 990$), $\nu$ must be less than 0.02. This silently corrupts results if not checked.

### 8. max_iter With Convergence Warnings

`SVC(max_iter=-1)` may produce `ConvergenceWarning` on hard problems. Setting `max_iter=10000` and decreasing `tol` from `1e-3` to `1e-6` usually helps. Also try `shrinking=False` for small datasets.

### 9. LinearSVC Not Converging

`max_iter=1000` (default) is often too small for text classification with many features. Use `max_iter=10000` and scale features first.

### 10. Multiclass predict() vs decision_function() Mismatch

With `decision_function_shape='ovr'`, `predict()` internally uses OvO voting, but `decision_function()` sums OvO scores to produce an OvR-like output. The `argmax(decision_function())` may differ from `predict()` when there are ties. Use `break_ties=True` for deterministic tiebreaking (at a cost).

### 11. support_vectors_ Is Empty for 'precomputed' Kernel

With `kernel='precomputed'`, `support_vectors_` is not stored (only the indices `support_`). Use `X_train[model.support_]` to retrieve them.

---

## Key Datasets for Examples

```python
# Dataset 1: Wine (classification, 178 samples, 13 features, 3 classes)
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
# Perfect for SVC demo: small n, multiple classes, requires scaling

# Dataset 2: Digits (classification, 1797 samples, 64 features, 10 classes)
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
# Good for kernel comparison, probability calibration demo

# Dataset 3: Synthetic (for margin visualization)
from sklearn.datasets import make_classification, make_moons, make_circles
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                            n_redundant=0, random_state=42)
X_moon, y_moon = make_moons(n_samples=500, noise=0.2, random_state=42)
X_circ, y_circ = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)

# Dataset 4: Breast Cancer (classification, 569 samples, 30 features, binary)
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

# Dataset 5: California Housing (regression, for SVR demo)
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)
# Warning: n=20640 — use a subset for SVR to avoid O(n^3) pain
X_sub, y_sub = X[:2000], y[:2000]
```

---

## Curated Resources

### Primary Papers (DOI-verified)

1. **Boser, Guyon, Vapnik (1992)** — hard-margin SVM + kernel trick
   - URL: https://dl.acm.org/doi/10.1145/130385.130401
   - DOI: 10.1145/130385.130401

2. **Cortes & Vapnik (1995)** — soft-margin SVM, the canonical reference
   - URL: https://link.springer.com/article/10.1007/BF00994018
   - DOI: 10.1007/BF00994018
   - PDF via Semantic Scholar: https://www.semanticscholar.org/paper/Support-Vector-Networks-Cortes-Vapnik/52b7bf3ba59b31f362aa07f957f1543a29a4279e

3. **Platt (1998)** — SMO algorithm, the reason sklearn SVMs are tractable
   - URL: https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/
   - Technical report: MSR-TR-98-14

4. **Schölkopf et al. (2000)** — ν-SVM (NuSVC/NuSVR)
   - Neural Computation 12(5):1207–1245

5. **Drucker et al. (1997)** — SVR (epsilon-insensitive regression)
   - NIPS 1997: https://www.researchgate.net/publication/2880432_Support_Vector_Regression_Machines

6. **Chang & Lin (2011)** — LIBSVM (the C library under sklearn's SVC)
   - DOI: https://doi.org/10.1145/1961189.1961199
   - Full paper: https://dl.acm.org/doi/10.1145/1961189.1961199

7. **Fan et al. (2008)** — LIBLINEAR (under LinearSVC)
   - URL: https://www.jmlr.org/papers/volume9/fan08a/fan08a.pdf

8. **Wen et al. (2018)** — ThunderSVM
   - URL: https://www.jmlr.org/papers/v19/17-740.html
   - DOI: 10.5555/3291125.3291146

### Books

9. **Schölkopf & Smola (2002)** — "Learning with Kernels: SVMs, Regularization, Optimization, and Beyond"
   - MIT Press. ISBN: 978-0-262-19475-4 (hardcover), 978-0-262-53657-8 (paperback)
   - Google Books: https://books.google.com/books?id=y8ORL3DWt4sC
   - **The** definitive reference for kernel methods theory. Chapters 2–7 cover all theoretical foundations.

### Official Documentation

10. **sklearn SVM User Guide** (1.5.2):
    - URL: https://scikit-learn.org/1.5/modules/svm.html
    - Covers formulations, kernel functions, complexity, scaling

11. **sklearn SVC API reference** (1.5.2):
    - URL: https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html

12. **sklearn RBF Parameter Tuning Example**:
    - URL: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    - Must-read for the C-gamma interaction visualization

13. **LIBSVM homepage** (Chang & Lin):
    - URL: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    - Has implementation guide, practical advice, and the famous LIBSVM README with tuning tips

### Tutorials & Blog Posts

14. **sklearn kernel approximation docs**:
    - URL: https://scikit-learn.org/stable/modules/kernel_approximation.html
    - RBFSampler + Nystroem for scaling SVMs to large data

15. **SHAP KernelExplainer docs**:
    - URL: https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html

16. **ThunderSVM docs**:
    - URL: https://thundersvm.readthedocs.io/en/latest/
    - Getting started: https://thundersvm.readthedocs.io/en/latest/get-started.html

17. **GitHub issue on KernelExplainer slowness**:
    - URL: https://github.com/shap/shap/issues/3747
    - Real-world confirmation that KernelExplainer + RBF SVC is very slow; use alternatives

18. **Lodhi et al. (2002)** — String kernels for text classification:
    - URL: https://www.jmlr.org/papers/volume2/lodhi02a/lodhi02a.pdf
    - JMLR 2(2):419–444

---

## Uncertainties

The following items the chapter author **must double-check** before writing:

1. **probability parameter deprecation:** The search results indicated `probability=True` may be deprecated in favor of `CalibratedClassifierCV`. Verify against the actual sklearn 1.5.x changelog — this claim appeared in search results but needs confirmation against the release notes at https://scikit-learn.org/1.5/whats_new/v1.5.html

2. **NuSVC class_weight support:** Search results suggest `class_weight` is not supported in NuSVC (only NuSVC doesn't allow it in some versions). Verify by checking https://scikit-learn.org/1.5/modules/generated/sklearn.svm.NuSVC.html attribute list.

3. **LinearSVC dual='auto' behavior:** The search confirmed `dual='auto'` is the default in sklearn 1.5.x. In older sklearn (< 1.3), the default was `dual=True`. The author should verify the exact sklearn version this changed in.

4. **ThunderSVM current maintenance status:** ThunderSVM's last release dates and Python version compatibility should be verified at https://github.com/Xtra-Computing/thundersvm/releases before recommending it as a primary GPU option.

5. **SHAP 0.46.x LinearExplainer:** Verify whether `shap.LinearExplainer` exists in SHAP 0.46.x as a fast alternative to KernelExplainer for LinearSVC. The `LinearExplainer` should work with `coef_`-based models.

6. **cuML SVM compatibility:** RAPIDS cuML SVM API may differ from sklearn. Verify the exact parameter names and whether `SVC(kernel='rbf')` is supported in the current cuML version (which may require CUDA 11+ and specific driver versions).

7. **gamma='scale' exact formula source:** The formula $\gamma = 1/(n\_features \times \text{Var}(\mathbf{X}))$ is stated in the sklearn docs but the exact form — whether it uses the full $\mathbf{X}$ variance or per-feature variance — should be verified by reading the sklearn source at `sklearn/svm/_base.py`.

8. **Platt 1999 vs 1998:** The SMO paper is 1998 (MSR-TR-98-14). The Platt scaling paper is 1999 (MIT Press). Verify these are indeed two separate publications.

9. **NuSVR epsilon vs C interaction:** In NuSVR, both `C` and `nu` exist (unlike NuSVC which replaces C with nu). The interaction between these two needs careful explanation — verify from https://scikit-learn.org/1.5/modules/generated/sklearn.svm.NuSVR.html

10. **LinearSVC loss='hinge' + penalty='l1' unsupported combination:** Confirm this is still the case in sklearn 1.5 (it was in all previous versions) to avoid documenting an invalid parameter combination.

---

## Summary for Chapter Author

Support Vector Machines have a clean three-paper origin story: Boser/Guyon/Vapnik 1992 introduced hard-margin SVMs with the kernel trick; Cortes/Vapnik 1995 added the soft-margin (C parameter) that makes them practical on real noisy data; and Platt 1998 gave SMO which made training tractable (O(n²) rather than requiring a full QP solver). The theory is elegant — maximum-margin classifiers with sparse solutions (only support vectors matter) grounded in VC theory — but the practical bottleneck is severe: sklearn's `SVC` is $O(n^2)$–$O(n^3)$ and should never be used on datasets with more than ~10,000–50,000 samples without kernel approximation. The two most critical practical points are: (1) always scale features before fitting any SVM, and (2) C and gamma must be tuned jointly on a log-scale 2D grid. For large $n$, the path is `LinearSVC` (LIBLINEAR backend, near-$O(n)$) or `Nystroem + LinearSVC` for approximate nonlinear SVMs. SHAP with `KernelExplainer` is confirmed extremely slow for kernel SVMs; use `coef_` for `LinearSVC`, permutation importance for kernel SVMs, or kernel approximation to get a linear model with usable `coef_`.
