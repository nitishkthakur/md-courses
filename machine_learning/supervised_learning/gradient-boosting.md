# Gradient Boosting Machines: The Complete Masterclass

> **Why this algorithm matters:** Gradient Boosting Machines are the dominant algorithm on
> structured tabular data — the kind that fills the world's spreadsheets, databases, and data
> warehouses. In the decade from 2014 to 2024, GBM variants (XGBoost, LightGBM, CatBoost) won
> more Kaggle competitions than every other algorithm combined. At Netflix, Airbnb, LinkedIn, and
> virtually every major technology company, a GBM is the first model trained on any new tabular
> prediction problem. The reason is not mystery: gradient boosting achieves near-optimal
> bias-variance tradeoff on tabular data through a beautifully principled idea — gradient descent
> in function space. After a 2001 paper by Jerome Friedman, what had been an ad-hoc ensemble
> trick became a fully general optimization framework that works with any differentiable loss
> function. Twenty-five years of engineering refinements later, the family of algorithms descended
> from that paper can train on 100 million rows in minutes on a single GPU. This chapter takes
> you from the original AdaBoost intuition through the full mathematical framework to every
> practical hyperparameter decision you will face in production.

---

## §0 — Python Ecosystem & Package Guide

The GBM ecosystem is unusually rich and fragmented. Four major libraries each have distinct
algorithmic innovations, performance profiles, and hyperparameter naming conventions. Picking
the wrong one for your problem costs real performance. Here is the complete landscape.

### Complete Package Table

| Package | Import | Version (mid-2026) | Algorithm variant | License | GPU |
|---|---|---|---|---|---|
| `scikit-learn` | `from sklearn.ensemble import GradientBoostingClassifier` | ~1.5.x | Friedman MART (2001), exact sort-based split finding | BSD-3 | No |
| `scikit-learn (Hist)` | `from sklearn.ensemble import HistGradientBoostingClassifier` | ~1.5.x | Histogram-based, LightGBM-inspired, native missing values & categoricals | BSD-3 | No |
| `xgboost` | `import xgboost as xgb` | ~3.x | XGBoost: 2nd-order Taylor + L1/L2/gamma regularization + DART | Apache-2 | Yes (CUDA) |
| `lightgbm` | `import lightgbm as lgb` | ~4.x | LightGBM: leaf-wise growth + GOSS sampling + EFB feature bundling | MIT | Yes |
| `catboost` | `from catboost import CatBoostClassifier` | ~1.2.x | Ordered boosting + symmetric oblivious trees + native categoricals | Apache-2 | Yes |
| `ngboost` | `from ngboost import NGBClassifier` | ~0.4.x | Natural Gradient Boosting for full distributional output | Apache-2 | No |

### Top Picks — Recommendation Table

| Package | Best for | Status | Notes |
|---|---|---|---|
| **`HistGradientBoostingClassifier`** | Quick, correct baselines; mixed data types; missing values | Modern, recommended | Default go-to for sklearn pipelines. Handles NaN and categoricals natively. |
| **`XGBClassifier` (xgboost)** | Kaggle competition work; GPU acceleration; custom objectives | Modern, dominant | Set `tree_method='hist', device='cuda'` for large data. Best ecosystem support. |
| **`LGBMClassifier` (lightgbm)** | Very large datasets (>1M rows); fastest CPU training | Modern, fast | Primary complexity control is `num_leaves`, not `max_depth`. Watch the subsample pitfall. |
| **`CatBoostClassifier` (catboost)** | High-cardinality categorical features; want zero preprocessing | Modern, specialized | Pass `cat_features` directly; no encoding needed. Uses `random_seed` not `random_state`. |
| **`NGBRegressor` (ngboost)** | Probabilistic regression; uncertainty quantification needed | Research-grade | Outputs a distribution object. Slower. Use when point predictions aren't enough. |

> 🔧 **In practice:** For 90% of new tabular problems, start with `HistGradientBoostingClassifier`
> (fast, correct, zero preprocessing needed) as your baseline, then switch to XGBoost or
> LightGBM when you need GPU scaling or finer regularization control. Reach for CatBoost
> specifically when you have many string categorical columns — its ordered target statistics
> genuinely outperform manual encodings.

### Canonical Import Block

```python
# sklearn (pedagogically clear; slow on n > 50k)
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)

# XGBoost (~3.x as of mid-2026)
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# LightGBM (~4.x)
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

# CatBoost (~1.2.x)
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# NGBoost (~0.4.x) — probabilistic boosting
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Normal, LogNormal, Bernoulli, k_categorical
from ngboost.scores import LogScore, CRPScore

# Explainability & HPO
import shap
import optuna
# Note: in Optuna 3.x, integrations may require: pip install optuna-integration
# from optuna.integration import XGBoostPruningCallback  # verify namespace
```

### The Same Fit Across Top Packages

Comparing GBM libraries is only meaningful when you control for the same learning rate and
approximate model capacity. Here is breast cancer (a familiar benchmark) fitted identically
across all four major packages:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── sklearn GradientBoosting (slow, exact, pedagogic baseline) ────────────
from sklearn.ensemble import GradientBoostingClassifier
m1 = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, random_state=42
)
m1.fit(X_train, y_train)
print(f"sklearn GB  AUC: {roc_auc_score(y_test, m1.predict_proba(X_test)[:,1]):.4f}")

# ── sklearn HistGradientBoosting (fast, production-ready) ─────────────────
from sklearn.ensemble import HistGradientBoostingClassifier
m2 = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=0.1, random_state=42
)
m2.fit(X_train, y_train)
print(f"sklearn Hist AUC: {roc_auc_score(y_test, m2.predict_proba(X_test)[:,1]):.4f}")

# ── XGBoost (regularized, 2nd-order, CUDA-capable) ────────────────────────
from xgboost import XGBClassifier
m3 = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    tree_method='hist', device='cpu',
    eval_metric='logloss', random_state=42, verbosity=0
)
m3.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"XGBoost     AUC: {roc_auc_score(y_test, m3.predict_proba(X_test)[:,1]):.4f}")

# ── LightGBM (leaf-wise, fastest CPU) ────────────────────────────────────
from lightgbm import LGBMClassifier
m4 = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    subsample=0.8, subsample_freq=1,          # NOTE: must set subsample_freq!
    colsample_bytree=0.8, random_state=42
)
m4.fit(X_train, y_train)
print(f"LightGBM    AUC: {roc_auc_score(y_test, m4.predict_proba(X_test)[:,1]):.4f}")

# ── CatBoost (ordered boosting, zero encoding needed) ─────────────────────
from catboost import CatBoostClassifier
m5 = CatBoostClassifier(
    iterations=300, learning_rate=0.05, depth=4,
    loss_function='Logloss', eval_metric='AUC',
    verbose=False, random_seed=42             # NOTE: random_seed, not random_state
)
m5.fit(X_train, y_train)
print(f"CatBoost    AUC: {roc_auc_score(y_test, m5.predict_proba(X_test)[:,1]):.4f}")
```

> ⚠️ **Pitfall:** Two silent gotchas in the cross-package comparison above. (1) LightGBM's
> `subsample` parameter is silently ignored unless `subsample_freq > 0`. The line
> `LGBMClassifier(subsample=0.8)` without `subsample_freq=1` uses full-sample training.
> (2) CatBoost uses `random_seed` not `random_state` — passing `random_state=42` gives no error
> but is silently ignored. These are genuine traps in production code.

---

## §1 — The Origin: The Papers That Started It All

### 1.1 The Prologue — AdaBoost and the Weak Learnability Hypothesis

The story of gradient boosting begins not with gradients but with a theoretical question posed
in the late 1980s by Michael Kearns: can a learning algorithm that is barely better than random
guessing be boosted into an arbitrarily accurate predictor?

> 📜 **Origin/Citation:** Freund, Y. and Schapire, R.E. (1997). "A Decision-Theoretic
> Generalization of On-Line Learning and an Application to Boosting." *Journal of Computer and
> System Sciences*, 55(1), 119–139. DOI: 10.1006/jcss.1997.1504.
> URL: https://dl.acm.org/doi/10.1006/jcss.1997.1504

In 1996–1997, Yoav Freund and Robert Schapire answered yes with **AdaBoost** (Adaptive Boosting).
The core idea was elegantly simple: train a sequence of weak classifiers, where each successive
learner focuses more on the examples the previous ones got wrong. After $T$ rounds, take a
weighted majority vote. The algorithm reweights training examples at each step, upweighting
misclassified examples so the next learner "pays attention" to the hard cases.

**The AdaBoost.M1 algorithm (binary classification, $y \in \{-1, +1\}$):**

$$\text{Initialize: } w_i^{(1)} = \frac{1}{n}, \quad i = 1, \ldots, n$$

For $t = 1, 2, \ldots, T$:

1. Train weak learner $h_t$ on the weighted dataset with weights $\{w_i^{(t)}\}$
2. Compute weighted misclassification error:
$$\epsilon_t = \sum_{i=1}^{n} w_i^{(t)} \cdot \mathbf{1}[h_t(\mathbf{x}_i) \neq y_i]$$
3. Compute learner weight:
$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
4. Update example weights:
$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp\left(-\alpha_t \cdot y_i \cdot h_t(\mathbf{x}_i)\right)$$
5. Normalize: $w_i^{(t+1)} \leftarrow w_i^{(t+1)} / \sum_j w_j^{(t+1)}$

Final prediction:
$$H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(\mathbf{x})\right)$$

Notice the learner weight formula. If $\epsilon_t = 0.5$ (random guessing), $\alpha_t = 0$ —
the learner contributes nothing. If $\epsilon_t \approx 0$ (near-perfect), $\alpha_t \to \infty$
— the learner dominates. If $\epsilon_t > 0.5$ (worse than random), $\alpha_t < 0$ — we flip
the learner's sign. This is exquisitely calibrated.

AdaBoost won the **Gödel Prize in 2003** — the highest honor in theoretical computer science.
Empirically, it worked spectacularly well on face detection (Viola-Jones, 2001) and many
classification benchmarks of the era.

**What wasn't understood at the time:** Why does AdaBoost not overfit catastrophically even after
thousands of rounds? The empirical observation was clear — test error kept dropping even as training
error hit zero. Schapire and colleagues explained this with **margin theory** (larger margins
generalize better), but the full story took years to unravel.

**The crucial retroactive insight (Friedman, Hastie, Tibshirani, 2000):** AdaBoost is implicitly
performing **coordinate-wise gradient descent on the exponential loss**:

$$L(y, F) = e^{-y \cdot F(\mathbf{x})}$$

This was not understood by Freund and Schapire when they designed AdaBoost — it was discovered
retroactively. This connection was the seed of Friedman's generalization.

### 1.2 The Foundational Paper — Friedman's Gradient Boosting Machine

> 📜 **Origin/Citation:** Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient
> Boosting Machine." *The Annals of Statistics*, 29(5), 1189–1232.
> DOI: 10.1214/aos/1013203451.
> Open Access URL: https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full

Jerome Friedman asked: what if AdaBoost's reweighting trick is actually an instance of a more
general principle? What if we could do boosting with *any* loss function, not just the
exponential loss that AdaBoost implicitly minimizes?

**The problem with AdaBoost as a starting point:**
- It only works for binary classification
- Its exponential loss is highly sensitive to outliers (it never stops penalizing misclassified
  examples, regardless of how far from the boundary they are)
- There is no principled way to extend it to regression, multi-class, or survival analysis
- Its connection to gradient descent was not formalized

**Friedman's core insight — gradient descent in function space:**

Instead of searching for the best parameter vector $\boldsymbol{\theta}$ in a fixed model class,
we search for the best *function* $F(\mathbf{x})$ in some function space. The goal is to minimize
the expected loss:

$$F^* = \arg\min_{F} \mathbb{E}_{y, \mathbf{x}}[L(y, F(\mathbf{x}))]$$

We cannot solve this directly (the function space is infinite-dimensional), but we can approximate
it greedily. At each iteration, we ask: given our current estimate $F_{t-1}(\mathbf{x})$, in which
functional direction should we move to decrease the loss most rapidly?

The answer from calculus of variations: the direction of steepest descent in function space is
the **negative gradient** of the loss, evaluated at the current prediction. For the $i$-th
training example:

$$r_{ti} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{t-1}}$$

Friedman calls these $r_{ti}$ the **pseudo-residuals**. The word "pseudo" is important: for
squared error loss, $r_{ti} = y_i - F_{t-1}(\mathbf{x}_i)$ — the actual residuals. For other
losses, they generalize the residual concept.

> 💡 **Intuition:** Think of the current model $F_{t-1}$ as your best current approximation to
> the truth. The pseudo-residuals tell you, for each training point, in which direction your
> prediction is wrong and by how much. You then fit a new weak learner (a shallow tree) to predict
> these pseudo-residuals — this new tree is a "correction model" that learns to fix the current
> model's systematic errors. You add a shrunken version of this correction to your ensemble.
> Repeat. Each round, the ensemble gets a little closer to the true function.

**The General GBM Framework (MART — Multiple Additive Regression Trees):**

**Initialization:**

$$F_0(\mathbf{x}) = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, \gamma)$$

This is the best constant prediction. For squared error loss, $F_0 = \bar{y}$ (the mean).
For log-loss (binary classification), $F_0 = \log\left(\frac{\bar{p}}{1-\bar{p}}\right)$
(the log-odds of the marginal class probability).

**For $t = 1, 2, \ldots, T$:**

**Step 1 — Compute pseudo-residuals:**
$$r_{ti} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{t-1}}, \quad i = 1, \ldots, n$$

**Step 2 — Fit a regression tree to pseudo-residuals:**

Fit a $J$-leaf regression tree $h_t$ to the dataset $\{(\mathbf{x}_i, r_{ti})\}_{i=1}^n$,
producing $J$ terminal regions $\{R_{tj}\}_{j=1}^J$.

**Step 3 — Line search for optimal leaf values:**

For each leaf $j$, find the optimal step size by solving a 1D optimization within the leaf:
$$\gamma_{tj} = \arg\min_{\gamma} \sum_{\mathbf{x}_i \in R_{tj}} L\left(y_i, F_{t-1}(\mathbf{x}_i) + \gamma\right)$$

For squared error loss this has a closed form ($\gamma_{tj}$ = mean of pseudo-residuals in leaf
$j$). For log-loss it requires a Newton step: $\gamma_{tj} \approx \sum r_{ti} / \sum |r_{ti}|(1 - |r_{ti}|)$.

**Step 4 — Update ensemble:**
$$F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \nu \cdot \sum_{j=1}^{J} \gamma_{tj} \cdot \mathbf{1}[\mathbf{x} \in R_{tj}]$$

where $\nu \in (0, 1]$ is the **learning rate** (also called **shrinkage**).

**Return $F_T(\mathbf{x})$.**

### The Final Prediction Formula

$$\hat{F}(\mathbf{x}) = F_0 + \sum_{t=1}^{T} \nu \cdot h_t(\mathbf{x})$$

This is additive model estimation — the prediction is the initial constant plus $T$ shrunken
tree corrections. For regression, this is the raw output. For classification, we apply a
link function: $\hat{p} = \sigma(F_T(\mathbf{x})) = 1/(1 + e^{-F_T(\mathbf{x})})$.

### Loss Functions and Their Gradients

| Loss name | $L(y, F)$ | Negative gradient $r_i$ | Leaf optimal $\gamma_j$ | Use case |
|---|---|---|---|---|
| Squared error (L2) | $\frac{1}{2}(y - F)^2$ | $y_i - F(\mathbf{x}_i)$ | Mean of residuals in leaf | Regression (default) |
| Absolute error (L1) | $|y - F|$ | $\text{sign}(y_i - F(\mathbf{x}_i))$ | Median of residuals in leaf | Outlier-robust regression |
| Huber | (see below) | Huber-clipped residual | Weighted median | Robust regression |
| Deviance/log-loss | $\log(1 + e^{-yF})$ | $y_i - \sigma(F(\mathbf{x}_i))$ | Newton-Raphson step | Binary classification |
| Exponential | $e^{-yF}$ | $y_i e^{-y_iF(\mathbf{x}_i)}$ | Weighted log-odds | AdaBoost equivalent |
| Multinomial deviance | $-\sum_k y_k \log p_k$ | $y_{ik} - p_k(\mathbf{x}_i)$ | Per-class Newton step | Multi-class classification |

> 🔬 **Deep dive:** The **Huber loss** for robust regression blends L2 for small residuals
> and L1 for large ones, controlled by the threshold parameter $\alpha$:
> $$L_\delta(y, F) = \begin{cases} \frac{1}{2}(y-F)^2 & \text{if } |y-F| \leq \delta \\ \delta\left(|y-F| - \frac{\delta}{2}\right) & \text{otherwise} \end{cases}$$
> For squared error, a single large outlier can dominate all pseudo-residuals, pulling trees
> to fit the outlier. Huber loss caps the gradient at $\delta$, preventing any one point from
> steering the model too far. Use `loss='huber'` in sklearn's `GradientBoostingRegressor`
> when you have heavy-tailed target distributions.

### What Was Uncertain at Publication

Friedman was honest about what he didn't know: how to choose the optimal tree depth $J$.
He suggested empirically $4 \leq J \leq 8$, with $J = 6$ as a safe default, but there was
no theoretical justification. He also noted that the interaction between learning rate $\nu$
and number of trees $T$ was clear qualitatively (lower $\nu$ requires larger $T$) but left
the precise tradeoff as an empirical question.

### The 2002 Follow-up — Stochastic Gradient Boosting

> 📜 **Origin/Citation:** Friedman, J.H. (2002). "Stochastic Gradient Boosting." *Computational
> Statistics & Data Analysis*, 38(4), 367–378. DOI: 10.1016/S0167-9473(01)00065-2.
> URL: https://ideas.repec.org/a/eee/csdana/v38y2002i4p367-378.html

Friedman made a critical practical improvement: before fitting each tree, subsample a random
fraction $\eta$ of the training data *without replacement*:

$$F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \nu \cdot h_t(\mathbf{x}; \mathcal{S}_t)$$

where $|\mathcal{S}_t| = \lfloor \eta n \rfloor$ is a random subsample drawn without
replacement. This introduced the `subsample` hyperparameter ($\eta$, typically 0.5–0.8).

The dual benefit is remarkable: (1) faster training (fewer samples per tree), and (2) *better
generalization* — the stochasticity acts as a variance-reduction regularizer, similar to
bagging but within the sequential boosting framework. Friedman reported that subsampling
often reduced test error by 10–30% and always reduced training time.

---

## §2 — The Algorithm, Deeply Explained

### 2.1 The Mathematical Foundation — Steepest Descent in Function Space

Let us build the foundation carefully. In ordinary gradient descent, we have a parameterized
model $f(\mathbf{x}; \boldsymbol{\theta})$ and minimize an empirical risk by updating parameters:

$$\boldsymbol{\theta}^{(t)} = \boldsymbol{\theta}^{(t-1)} - \rho \cdot \nabla_{\boldsymbol{\theta}} \hat{R}(\boldsymbol{\theta})$$

Gradient boosting replaces the parameter vector $\boldsymbol{\theta}$ with the function
$F$ itself, evaluated at the training points. Think of the vector of predictions
$\mathbf{F} = (F(\mathbf{x}_1), F(\mathbf{x}_2), \ldots, F(\mathbf{x}_n))^T$ as the
"parameters." The empirical risk is:

$$\hat{R}(F) = \sum_{i=1}^{n} L(y_i, F(\mathbf{x}_i))$$

The negative gradient of this with respect to $\mathbf{F}$ evaluated at the current estimate:

$$-\mathbf{g}^{(t)} = \left(-\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right)_{i=1}^{n}$$

In pure gradient descent, we would update $\mathbf{F} \leftarrow \mathbf{F} + \rho \cdot (-\mathbf{g}^{(t)})$.
But this only updates the function at the training points — it gives us no generalization to
new $\mathbf{x}$ values.

The **gradient boosting trick**: fit a function (regression tree) $h_t$ to the negative
gradient values. The tree learns a *generalizable* functional relationship from features to
gradient directions. Then add $\nu \cdot h_t$ to the ensemble. The tree fitting is the
"generalization step" — it extrapolates the gradient direction to all of input space.

> 💡 **Intuition 1 — The Residual Fitter:** For squared error loss, the pseudo-residuals
> ARE the ordinary residuals: $r_{ti} = y_i - F_{t-1}(\mathbf{x}_i)$. GBM with squared error
> loss literally fits each new tree to the unexplained residuals of the current ensemble.
> The ensemble becomes a sum of models, each trained to explain what the previous ones missed.

> 💡 **Intuition 2 — The Photographer Analogy:** Imagine you are photographing a landscape
> in stages. Your first photo captures the broad features — mountains, sky, forest. Your second
> photo captures the fine detail your first one blurred. Your third photo sharpens the subtle
> lighting gradients the first two missed. Each photo is a "correction" to the image so far.
> The final composite image (your ensemble) is more detailed than any single photo could be.
> The learning rate $\nu$ controls how much each photo contributes — low $\nu$ means each
> photo is slightly under-exposed, requiring more photos (iterations) but producing a more
> carefully blended final image.

> 💡 **Intuition 3 — The Expert Committee:** Think of each tree as an expert who specializes
> in the cases the committee has gotten wrong so far. The first expert gives a broad opinion.
> The second expert is specifically trained on the most controversial cases — the ones the
> first expert missed. The third expert focuses on the remaining controversies. The final
> verdict is a weighted sum of all expert opinions, with more weight on the earlier, more
> confident experts (captured by the shrinkage $\nu$).

### 2.2 Why Trees as Base Learners?

Friedman chose regression trees for a specific reason: trees naturally partition input space
into rectangular regions, and within each region, the prediction is a constant. This gives
GBM a natural decomposition:

$$F_T(\mathbf{x}) = F_0 + \sum_{t=1}^{T} \nu \sum_{j=1}^{J} \gamma_{tj} \cdot \mathbf{1}[\mathbf{x} \in R_{tj}]$$

This is a **sum of piecewise constant functions** — a nonparametric regression in product
space. The tree structure captures interactions between features (a split on feature $A$
followed by a split on feature $B$ is an $A \times B$ interaction), while the depth controls
the order of interactions. A depth-3 tree captures up to 3-way interactions.

Trees are also computationally convenient: fitting a regression tree to the pseudo-residuals
is an $O(n \cdot p \cdot d)$ operation (for exact methods), and the prediction of a tree
ensemble is $O(T \cdot d)$ at inference time.

In principle, any differentiable model could serve as a base learner — Friedman also considered
linear models and smoothing splines. But trees dominate in practice because they handle mixed
feature types, interact naturally with boosting's additive structure, and have well-understood
complexity controls.

### 2.3 The Role of Shrinkage (Learning Rate)

The shrinkage parameter $\nu$ is arguably the single most important hyperparameter in GBM.
Its role is not simply "learning speed" — it is a fundamental regularization mechanism.

Without shrinkage ($\nu = 1$), each tree takes a full gradient step. A few hundred trees are
typically enough to drive training error to near-zero, but generalization suffers because each
step is "too confident" — individual tree noise is amplified.

With small $\nu$ (e.g., 0.01), each tree takes a tiny step, adding a small correction to the
ensemble. This requires far more trees (compensating by increasing $T$), but each tree has
much lower influence on the final model. The ensemble becomes an average over many small
corrections — much more robust to individual tree idiosyncrasies.

The fundamental tradeoff:

$$\text{Bias} \propto \nu \cdot T, \quad \text{Variance} \propto \nu^2 \cdot T$$

Decreasing $\nu$ and proportionally increasing $T$ (keeping $\nu T$ approximately constant)
*decreases variance without increasing bias*, at the cost of training time. This is why the
canonical recommendation is: "lower learning rate + early stopping = better generalization."

### 2.4 Computational Complexity

**Training:**

- **Exact sort-based (sklearn `GradientBoostingClassifier`):** At each tree, for each feature,
  sort all $n$ values to find best splits. Cost per tree: $O(n \cdot p \cdot \log n)$ (due to
  sorting). Total: $O(T \cdot n \cdot p \cdot \log n)$.
  
- **Histogram-based (HistGBM, XGBoost hist, LightGBM):** Pre-bin features into $B \leq 255$
  bins. Finding the best split requires scanning $B$ thresholds per feature instead of $n$.
  Cost per tree: $O(n \cdot p + B \cdot p)$ (build histograms + scan them). Total:
  $O(T \cdot (n + B) \cdot p)$. For large $n$, this is $O(T \cdot B \cdot p)$ — **independent
  of $n$ for the split-finding step**. This is the core reason histogram methods are 10–20x
  faster on large datasets.

**Inference:**

Each prediction requires traversing $T$ trees of depth $d$: $O(T \cdot d)$ per sample.
For a typical configuration (500 trees, depth 6): 3,000 node evaluations per sample. This is
extremely fast on modern hardware — sub-millisecond for a single sample.

> 📊 **Benchmark:** sklearn's own documentation reports: `GradientBoostingRegressor` (200 trees,
> California housing dataset) trains in ~7.2 seconds; `HistGradientBoostingRegressor` (same
> settings) trains in ~0.5 seconds — a 14x speedup. The gap widens with larger $n$.

### 2.5 What GBM Assumes About Your Data

Unlike many other models, GBM makes very few explicit assumptions:

1. **The loss is differentiable** with respect to $F(\mathbf{x})$. (Trivially satisfied for
   all standard choices.)

2. **The relationship between features and target is approximable by a sum of piecewise
   constant functions.** Since any smooth function can be approximated by step functions as
   $T \to \infty$, this is a very weak assumption — essentially no functional form assumption.

3. **No distributional assumption** on the features. GBM handles non-Gaussian, mixed-type,
   skewed, and heavy-tailed feature distributions naturally.

4. **No scale invariance** — unlike SVMs or linear models, GBM does not require feature
   scaling. A feature measured in dollars has the same effect as one measured in cents because
   splits find the optimal threshold regardless of scale.

**The implicit assumptions that matter in practice:**

- **Smoothness of the target function.** GBM approximates smooth functions with piecewise
  constant trees. Very jagged, discontinuous target functions may require extremely many trees.

- **Existence of finite-variance noise.** If the noise is heavy-tailed (Cauchy distribution),
  squared error loss explodes and you must switch to absolute error or Huber loss.

- **Training distribution = test distribution.** GBM (like all supervised learning) is not
  distribution-shift robust without explicit mechanisms.

---

## §3 — The Full Evolution: From AdaBoost to Modern Industrial GBM

The story of gradient boosting is a story of engineering solving theoretical limitations at
scale. Each generation addressed a concrete bottleneck of the previous one.

### 3.1 AdaBoost (1996/1997) — The Seed

> 📜 **Citation:** Freund & Schapire (1997), *J. Computer and System Sciences*, 55(1).

**Problem it solved:** Can weak learners be combined into strong ones?

**Key algorithm:** Exponential loss minimization via adaptive reweighting.

**Limitation:** Only binary classification with exponential loss. No regression, no multi-class,
no extension to other loss functions. Sensitive to outliers (exponential loss is unbounded).

**When to use it today:** Rarely. AdaBoost is available in sklearn as
`AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))`. Its main modern
value is pedagogical — it shows the mechanism of adaptive weighting clearly. On real problems,
Friedman's GBM universally outperforms it.

### 3.2 Friedman MART / GBM (2001) — The Foundation

> 📜 **Citation:** Friedman (2001), *The Annals of Statistics*, 29(5). The most-cited paper
> in the boosting lineage. ~12,000 citations as of mid-2026.

**Problem it solved:** AdaBoost was not generalizable. MART unified AdaBoost, regression
boosting, and multi-class boosting under a single gradient descent framework.

**Key algorithmic change:** Replaced exponential loss + reweighting with *any* differentiable
loss + pseudo-residual fitting. Introduced the concept of fitting trees to gradients.

**When to prefer:** The sklearn `GradientBoostingClassifier/Regressor` implements this faithfully.
Use it when you need the closest implementation to the original paper, for teaching purposes,
or when `n < 50,000` and you want fine-grained control over the MART algorithm.

**Python:**
```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# Friedman MART, exact sort-based split finding
# Hyperparameters: n_estimators, learning_rate, max_depth (default 3),
#                  subsample, loss, min_samples_leaf, max_features
```

### 3.3 Stochastic Gradient Boosting (2002) — The Regularizer

> 📜 **Citation:** Friedman (2002), *Computational Statistics & Data Analysis*, 38(4).

**Problem it solved:** Full-sample tree fitting has high variance and is slow.

**Key change:** At each iteration, subsample $\eta n$ examples without replacement before
fitting the tree. This adds Breiman-style variance reduction to the sequential boosting framework.

$$F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \nu \cdot h_t(\mathbf{x}; \mathcal{S}_t), \quad |\mathcal{S}_t| = \lfloor \eta n \rfloor$$

**When to prefer:** Always. Setting `subsample=0.7` (or any value < 1.0) in
`GradientBoostingClassifier` makes it stochastic. This virtually always improves
generalization at the cost of some noise in the training curve. The parameter $\eta = 0.5$
was Friedman's recommendation; modern practice uses 0.6–0.8.

### 3.4 sklearn `HistGradientBoosting*` (2020) — The Speed Fix

**Problem it solved:** The exact sort-based split finding of sklearn's `GradientBoosting*` is
$O(n \cdot p)$ per node, making it unusably slow on datasets with $n > 50{,}000$.

**Key algorithmic change:** Pre-bin each continuous feature into at most `max_bins=255`
discrete bins before training. Finding the best split then requires scanning only 255
thresholds instead of $n$ unique values. This reduces per-node work from $O(n)$ to $O(255)$,
giving an asymptotic speedup linear in $n/B$.

**Additional innovations over the base `GradientBoosting*`:**
- **Native missing value handling:** A dedicated bin is reserved for NaN values; the best
  direction (left or right) is learned during split finding. No imputation needed.
- **Native categorical features:** Pass `categorical_features=['col_name']` or
  `categorical_features='from_dtype'` (auto-detects pandas Categorical). Categories are
  handled with histogram-based optimal splits, not ordinal encoding.
- **Monotonic constraints:** `monotonic_cst={'feature_name': +1}` forces predictions to be
  monotonically non-decreasing with respect to that feature. Essential for business logic
  (e.g., loan amount should not decrease predicted default probability).
- **Interaction constraints:** `interaction_cst="pairwise"` or custom interaction groups.
  Prevents the model from learning spurious feature interactions.

> ⚠️ **Pitfall:** In sklearn 1.5.x, `monotonic_cst` and `categorical_features` cannot be
> applied to the same feature simultaneously — this raises a `ValueError`. Verify whether this
> is resolved in your version (tracked as GitHub issue #28898).

**When to prefer:** This is the default choice for sklearn users working with moderately large
data ($n$ up to ~1M), mixed feature types, or missing values. Cleaner API, faster training,
and zero preprocessing overhead.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=31,          # primary complexity control
    min_samples_leaf=20,        # regularization
    l2_regularization=0.1,      # leaf weight shrinkage
    categorical_features='from_dtype',   # auto-detect pandas Categorical
    early_stopping=True,        # uses internal validation split
    n_iter_no_change=20,
    random_state=42
)
```

### 3.5 XGBoost (2016) — The Regularization Leap

> 📜 **Citation:** Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting
> System." *KDD 2016*, 785–794. DOI: 10.1145/2939672.2939785.
> arXiv: https://arxiv.org/abs/1603.02754

**Problem it solved:** Friedman's GBM used only first-order gradients, giving suboptimal step
sizes. There was no principled regularization beyond shrinkage and subsampling.

**Key innovation 1 — Second-order Taylor expansion:**

Friedman approximated the loss at each tree step as:
$$\tilde{L}^{(t)} \approx \sum_i \left[L(y_i, F_{t-1}(\mathbf{x}_i)) + g_i h_t(\mathbf{x}_i)\right]$$

where $g_i = \partial L / \partial \hat{y}_i$ is the first-order gradient.

XGBoost adds the Hessian term:
$$\tilde{L}^{(t)} \approx \sum_i \left[g_i h_t(\mathbf{x}_i) + \frac{1}{2} h_i h_t(\mathbf{x}_i)^2\right] + \Omega(h_t)$$

where $h_i = \partial^2 L / \partial \hat{y}_i^2$ is the second-order gradient (Hessian), and
$\Omega(h_t)$ is the tree complexity penalty.

**Key innovation 2 — Regularized objective:**

$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|$$

where $T$ is the number of leaves, $w_j$ are leaf weights, $\gamma$ is a penalty on the
number of leaves, $\lambda$ is L2 regularization, and $\alpha$ is L1 regularization.

**Key innovation 3 — Closed-form optimal leaf weights:**

With the quadratic approximation, the optimal weight for leaf $j$ has a closed form:
$$w_j^* = -\frac{G_j}{H_j + \lambda}, \quad G_j = \sum_{i \in R_j} g_i, \quad H_j = \sum_{i \in R_j} h_i$$

**Key innovation 4 — Split gain formula:**

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

Splits are only accepted when Gain > 0. The $\gamma$ parameter is therefore a natural, principled
minimum-gain threshold — not just a heuristic pruning parameter.

**Key innovation 5 — Column subsampling:**

`colsample_bytree`, `colsample_bylevel`, `colsample_bynode` add three levels of feature
randomization: per-tree, per-depth-level, and per-split-node respectively. These reduce
correlation between trees analogously to Random Forests.

**When to prefer XGBoost:** When you need GPU acceleration on large datasets, require custom
loss functions (via `obj` parameter), compete in machine learning competitions, or need the
widest ecosystem support (SHAP, Optuna, sklearn, Ray, Dask all have first-class XGBoost
integrations).

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    gamma=0,                    # min split gain (acts like pruning)
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,                # L1
    reg_lambda=1,               # L2 (default is 1, not 0!)
    tree_method='hist',         # always use hist for speed
    device='cpu',               # 'cuda' for GPU
    eval_metric='logloss',
    early_stopping_rounds=50,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration}")
```

> ⚠️ **Pitfall:** XGBoost's default `learning_rate` (called `eta`) is **0.3**, not 0.1.
> This is a very aggressive learning rate — much higher than sklearn's default of 0.1. When
> benchmarking XGBoost against other libraries at default settings, you are comparing different
> regimes. Always set `learning_rate=0.05` or `0.1` for fair comparisons. Also note:
> `reg_lambda=1` by default — XGBoost is *always* L2-regularized unless you set it to 0.

### 3.6 LightGBM (2017) — The Speed Record

> 📜 **Citation:** Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting
> Decision Tree." *NeurIPS 2017*, 3149–3157.
> URL: https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

**Problem it solved:** Even histogram-based XGBoost is slow on datasets with millions of rows
and thousands of features. Two bottlenecks remained: (1) all data points participate in
histogram construction, and (2) all features must be scanned even when most are sparse/useless.

**Key innovation 1 — GOSS (Gradient-based One-Side Sampling):**

Standard subsampling (Friedman 2002) samples randomly. GOSS samples *strategically*:

1. Sort all samples by |gradient| in decreasing order
2. Keep the top $a \cdot 100\%$ of samples (large-gradient samples — the "hard" ones)
3. Randomly sample $b \cdot 100\%$ from the remaining samples (small-gradient samples)
4. Amplify the small-gradient samples by a factor of $\frac{1-a}{b}$ in the histogram
   construction to compensate for their under-representation

**Theoretical justification:** Samples with large gradients contribute most to the information
gain calculation. Dropping small-gradient samples (already well-fitted) with small probability
introduces much less approximation error than uniform subsampling.

**Key innovation 2 — EFB (Exclusive Feature Bundling):**

In high-dimensional sparse datasets (e.g., one-hot encoded categoricals, NLP bag-of-words
features), many features are *mutually exclusive* — they never take non-zero values
simultaneously (e.g., two one-hot columns from the same categorical variable). EFB bundles
such features together, reducing $p$ features to $O(p/k)$ bundles, where $k$ is the average
number of non-exclusive features per bundle.

Finding the optimal bundling is NP-hard (graph coloring), but LightGBM uses a greedy
approximation that works well in practice.

**Key innovation 3 — Leaf-wise (best-first) tree growth:**

All previous implementations (sklearn, XGBoost default) grow trees **level-wise** —
all nodes at depth $d$ are split before any node at depth $d+1$. LightGBM grows trees
**leaf-wise** — at each step, split the single leaf with the highest information gain, anywhere
in the tree. This produces unbalanced but more efficient trees.

Level-wise: `max_depth` controls complexity. Leaf-wise: `num_leaves` controls complexity.

> ⚠️ **Pitfall:** Setting `max_depth=6` in LightGBM and thinking it is equivalent to
> XGBoost's `max_depth=6` is wrong. In LightGBM, `max_depth` is a *ceiling*, not the actual
> structure. With `num_leaves=200`, the tree can have up to 200 leaves, each potentially at
> depth > 6, effectively ignoring `max_depth`. The primary tuning knob is `num_leaves`;
> `max_depth` is a safety guard.

**Speed claim:** Up to 20x faster than XGBoost on large datasets, with similar or better accuracy.

**When to prefer LightGBM:** Very large datasets ($n > 1\text{M}$), high-dimensional sparse
features, when raw training speed is critical, or when GPU memory is a constraint (LightGBM's
GPU footprint is smaller than XGBoost's for equivalent settings).

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,              # PRIMARY complexity control
    max_depth=-1,               # -1 = no limit; num_leaves governs depth
    min_child_samples=20,       # min data per leaf (regularization)
    subsample=0.8,
    subsample_freq=1,           # MUST set > 0 to enable subsampling!
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    class_weight='balanced',    # for imbalanced targets
    random_state=42
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)
```

### 3.7 CatBoost (2018) — The Categorical Feature Specialist

> 📜 **Citation:** Prokhorenkova, L. et al. (2018). "CatBoost: Unbiased Boosting with
> Categorical Features." *NeurIPS 2018*, 6639–6649.
> arXiv: https://arxiv.org/abs/1706.09516

**Problem it solved:** All prior GBM implementations suffer from **prediction shift** — a
subtle bias in the gradient estimates that accumulates over boosting rounds. Additionally,
categorical feature handling required manual encoding (label encoding introduces false ordinal
relationships; one-hot encoding is expensive; target encoding introduces leakage).

**Key innovation 1 — Ordered Boosting:**

In standard GBM, when computing the gradient for example $i$ at iteration $t$, we use
$F_{t-1}(\mathbf{x}_i)$, where $F_{t-1}$ was trained on data including example $i$.
This creates a **circular dependency**: the gradient at $i$ depends on a model that has
already partially "memorized" $i$'s label. The resulting gradient estimates are biased.

CatBoost eliminates this with ordered boosting:

1. Randomly permute the training examples (once, before training)
2. For example $i$ (at position $\sigma_i$ in the permutation), compute its gradient using
   a model trained *only on examples that appear before it* in the permutation:
   $r_{ti} = -\partial L(y_i, F_{t-1}^{(\sigma_i)}(\mathbf{x}_i)) / \partial F$

This is computationally expensive (requires maintaining $n$ different per-sample models), but
CatBoost implements it efficiently by sharing computation across permutation positions.

**Key innovation 2 — Ordered Target Statistics for categoricals:**

For categorical feature $c$ and example $i$ (at position $\sigma_i$), CatBoost encodes the
category as its in-sample target statistic, using *only* preceding examples:

$$\hat{x}_{i}^{c} = \frac{\sum_{j: \sigma_j < \sigma_i, x_j^c = x_i^c} y_j + P \cdot \bar{y}}{\sum_{j: \sigma_j < \sigma_i, x_j^c = x_i^c} 1 + P}$$

where $P$ is a prior weight (CatBoost uses $P = 1$ by default) and $\bar{y}$ is the global
mean. This is essentially an online ordered target encoding that prevents target leakage.

**Key innovation 3 — Symmetric (Oblivious) Trees:**

CatBoost uses oblivious decision trees as base learners by default. An oblivious tree applies
the *same* split criterion at all nodes of a given depth level:

- At depth 1: all nodes split on feature $f_1$ at threshold $t_1$
- At depth 2: all nodes split on feature $f_2$ at threshold $t_2$
- ...

This produces a binary tree that can be represented as a lookup table of $2^d$ values. At
inference time, classification becomes $d$ comparisons + one table lookup — extremely fast.
The oblivious constraint acts as a regularizer and often outperforms standard asymmetric trees
on structured data.

**When to prefer CatBoost:** When your dataset has many high-cardinality categorical features
(user IDs, product codes, geographic regions), when you want zero preprocessing (pass raw
category strings directly), when ordered boosting matters (small datasets with potential target
leakage), or when inference speed is critical (symmetric trees are very fast).

```python
from catboost import CatBoostClassifier, Pool

cat_features = ['country', 'product_category', 'user_segment']
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_features)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    cat_features=cat_features,
    loss_function='Logloss',
    eval_metric='AUC',
    od_type='Iter',         # early stopping via iteration count
    od_wait=50,
    task_type='CPU',        # 'GPU' for GPU training
    verbose=False,
    random_seed=42          # NOTE: random_seed, not random_state
)
model.fit(train_pool, eval_set=val_pool)
```

### 3.8 DART — Dropouts Meet MART (2015)

> 📜 **Citation:** Rashmi, K.V. and Gilad-Bachrach, R. (2015). "DART: Dropouts meet Multiple
> Additive Regression Trees." *AISTATS 2015*, 489–497.
> URL: https://proceedings.mlr.press/v38/korlakaivinayak15.pdf

**Problem it solved:** In standard GBM, later trees focus on refining tiny residuals, contributing
negligible increments to the prediction. This over-specialization gives early trees disproportionate
influence. The analogy to neural networks: "dead neurons" from over-reliance on strong features.

**Key change:** At each boosting iteration, randomly *drop* a subset of existing trees
(analogous to neural dropout). Fit the new tree on residuals from the remaining (non-dropped)
ensemble. After fitting, rescale contributions to maintain ensemble scale. This prevents
late trees from becoming irrelevant and forces all trees to contribute meaningfully.

**Available in:** XGBoost (`booster='dart'`, with parameters `rate_drop`, `skip_drop`,
`normalize_type='tree'/'forest'`) and LightGBM (`boosting='dart'`).

> ⚠️ **Pitfall:** DART is **incompatible with early stopping**. Because DART randomly drops
> trees, the validation loss is non-monotone across iterations — it can go up even when
> the model improves. XGBoost silently ignores `early_stopping_rounds` when `booster='dart'`.
> Always use a fixed `n_estimators` with DART. This makes DART harder to tune and limits its
> practical use.

### 3.9 NGBoost — Probabilistic Gradient Boosting (2020)

> 📜 **Citation:** Duan, T. et al. (2020). "NGBoost: Natural Gradient Boosting for
> Probabilistic Prediction." *ICML 2020*. arXiv: https://arxiv.org/abs/1910.03225

**Problem it solved:** All prior GBM variants output *point predictions*. In medicine, finance,
and weather forecasting, you need a full probability distribution $P(y | \mathbf{x})$ —
not just a mean estimate but also an uncertainty interval.

**Key insight:** Instead of boosting toward a scalar $\hat{y}$, boost toward the *parameters*
$\theta$ of a probability distribution. Use the **natural gradient** (gradient pre-conditioned
by the Fisher information matrix) to make updates scale-invariant:

$$\tilde{\nabla}_\theta S(\theta; y) = I(\theta)^{-1} \nabla_\theta S(\theta; y)$$

where $I(\theta)$ is the Fisher information matrix of the chosen distribution, and $S$ is a
proper scoring rule (log score or CRPS).

```python
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

ngb = NGBRegressor(
    Dist=Normal,
    Score=LogScore,
    n_estimators=500,
    learning_rate=0.01,
    verbose=False
)
ngb.fit(X_train, y_train)
y_dist = ngb.pred_dist(X_test)
print(y_dist.mean[:5])                # point predictions
print(y_dist.interval(0.95)[:5])      # 95% prediction intervals
```

**When to prefer NGBoost:** When calibrated uncertainty quantification is the primary goal
(clinical risk scoring, financial VaR, scientific emulation). Be aware: it is significantly
slower than XGBoost/LightGBM and has less active maintenance as of mid-2026.

### Evolution Summary Table

| Variant | Year | Key innovation | Primary gain | Sklearn? |
|---|---|---|---|---|
| AdaBoost | 1997 | Adaptive sample reweighting | Classification boosting | `AdaBoostClassifier` |
| MART/GBM | 2001 | Gradient descent in function space | General loss functions | `GradientBoosting*` |
| Stochastic GB | 2002 | Row subsampling per iteration | Variance reduction | `subsample` param |
| DART | 2015 | Dropout on trees | Anti-over-specialization | via XGBoost/LightGBM |
| XGBoost | 2016 | 2nd-order gradients + regularization | Better generalization | `XGBClassifier` |
| HistGBM (sklearn) | 2020 | Histogram binning | Speed, NaN/cat support | `HistGradientBoosting*` |
| LightGBM | 2017 | GOSS + EFB + leaf-wise | Extreme scale speed | `LGBMClassifier` |
| CatBoost | 2018 | Ordered boosting + symmetric trees | Categorical features | `CatBoostClassifier` |
| NGBoost | 2020 | Natural gradient + distributions | Probabilistic output | `NGBClassifier` |

---

## §4 — Hyperparameters: The Complete Guide

GBM has more hyperparameters than most algorithms, and they interact in important ways. This
section covers every significant parameter, explains its mechanism, and gives concrete tuning
guidance. We organize by package, starting with the universal parameters that appear in all
implementations.

### 4.1 Universal Parameters (all GBM implementations)

#### `n_estimators` / `num_iterations` / `iterations`

**Mechanism:** The number of boosting rounds — equivalently, the number of trees. Each
additional tree corrects the residuals of the previous ensemble. More trees = more capacity =
longer training = potential overfitting.

**Defaults:** sklearn `GradientBoosting*`: 100. sklearn `HistGradientBoosting*`: 100.
XGBoost (sklearn wrapper): 100. LightGBM: 100. CatBoost: 1000 (much higher default!).

**The key interaction:** `n_estimators` and `learning_rate` are inextricably coupled.
Halving the learning rate *requires* roughly doubling `n_estimators` to achieve the same
training loss. They are not independent tuning axes.

**Tuning strategy:** Never tune `n_estimators` with grid search. Instead:
1. Set a low `learning_rate` (0.01–0.05)
2. Set `n_estimators` very high (1000–5000)
3. Use early stopping to find the optimal value automatically
4. The optimal `n_estimators` found by early stopping is your answer

**Too low:** High bias — the model has not converged.
**Too high without early stopping:** Overfitting — the model fits noise in the training set.

#### `learning_rate` / `eta` / `shrinkage`

**Mechanism:** Multiplies each tree's contribution to the ensemble:
$F_t = F_{t-1} + \nu \cdot h_t$. Smaller $\nu$ means each tree has less influence, requiring
more trees to fit the data but producing a smoother, less overfit final model.

**Default:** sklearn: 0.1. XGBoost: 0.3 (much higher!). LightGBM: 0.1. CatBoost: auto (~0.03).

**Tuning strategy:** Start at 0.1. If you have time budget for more iterations, reduce to 0.05
or 0.01. The literature consistently shows that smaller learning rates with more trees
generalize better. On Kaggle competitions, winning models often use `learning_rate=0.01` with
3,000–10,000 trees.

**Typical range:** 0.005–0.3. Log-scale search: `trial.suggest_float('lr', 0.01, 0.3, log=True)`.

**Too high (> 0.3):** Noisy training curve, premature convergence to a suboptimal solution.
**Too low (< 0.005):** Very slow training; early stopping may not trigger within your budget.

> 🏆 **Best practice:** Use `learning_rate=0.05` with early stopping as your default. This
> is the single most effective change you can make from out-of-box settings. If final model
> performance matters more than training speed, try `0.01`.

#### `max_depth` / `depth`

**Mechanism:** Maximum depth of individual trees. Controls the maximum order of feature
interactions a single tree can capture. A depth-$d$ tree captures at most $d$-way interactions.

**Default:** sklearn `GradientBoosting*`: 3. XGBoost: 6. LightGBM: -1 (unlimited, controlled
by `num_leaves`). CatBoost: 6.

**Interaction with base learner role:** GBM deliberately uses *shallow* trees (weak learners).
A depth-3 tree (8 leaves max) is appropriate. A depth-10 tree is effectively a strong learner —
one tree already overfit — and the boosting framework breaks down.

**Tuning range:** 3–8 for most problems. If your interactions are genuinely complex (many
high-order feature interactions), consider depth 6–8 for XGBoost/CatBoost.

**Too deep:** Each tree overfits, the ensemble has high variance. Early trees fit noise,
leaving little to correct in later rounds.
**Too shallow (depth 1 = stumps):** The classic AdaBoost weak learner. Slow convergence,
misses interactions. Use depth 2–3 minimum for most problems.

> ⚠️ **Pitfall:** In LightGBM, `max_depth` is NOT the primary complexity control — `num_leaves`
> is. Setting `max_depth=6` with `num_leaves=100` allows a tree with 100 leaves and depth
> potentially > 6 (LightGBM grows leaf-wise, not level-wise). See §3.6.

#### `subsample` (Row subsampling)

**Mechanism:** Fraction of training samples selected uniformly at random (without replacement)
to fit each tree. Values < 1.0 implement stochastic gradient boosting (Friedman 2002).

**Default:** 1.0 in all implementations (no subsampling).

**Effect:** Reduces variance (fewer overfit trees) and speeds training. The variance reduction
is analogous to bagging but applied within the sequential framework.

**Typical range:** 0.5–0.9. Values below 0.5 may introduce too much noise.

**Tuning:** Generally 0.6–0.8 is a safe default improvement over 1.0. Worth including in
Optuna search: `trial.suggest_float('subsample', 0.5, 1.0)`.

> ⚠️ **Pitfall (LightGBM):** Setting `subsample=0.8` in LightGBM does **nothing** unless you
> also set `subsample_freq` (alias `bagging_freq`) to a positive integer:
> ```python
> # WRONG — subsample silently ignored
> LGBMClassifier(subsample=0.8)
> # CORRECT
> LGBMClassifier(subsample=0.8, subsample_freq=1)
> ```
> This is the most common silent misconfiguration in LightGBM.

#### Column subsampling (`max_features` / `colsample_bytree`)

**Mechanism:** Fraction (or count) of features considered when choosing each split.
Analogous to Random Forest's feature subsampling — reduces correlation between trees.

**Default:** sklearn: `None` (all features). XGBoost: 1.0. LightGBM: 1.0.

**Typical range:** 0.5–1.0. In XGBoost, there are three levels: `colsample_bytree` (per tree),
`colsample_bylevel` (per depth level), `colsample_bynode` (per split). Start with
`colsample_bytree=0.8` for most problems.

#### Minimum leaf size (`min_samples_leaf` / `min_child_samples` / `min_child_weight`)

**Mechanism:** Prevents splits that produce very small leaves. In sklearn, `min_samples_leaf`
is the count of samples; XGBoost uses `min_child_weight` (minimum sum of Hessians in leaf);
LightGBM uses `min_child_samples` (minimum sample count, default 20).

**Effect:** Primary regularization parameter for the tree structure. Larger values reduce
overfitting by preventing the model from fitting individual points in small leaves.

**Default:** sklearn: 1 (very low — practically no constraint). LightGBM: 20 (well-regularized
by default). XGBoost: `min_child_weight=1`.

**Tuning:** For sklearn, increase from 1 to 5–50. For LightGBM, the default 20 is reasonable;
try 10–100. For XGBoost, try `min_child_weight` 1–10.

### 4.2 XGBoost-Specific Parameters

#### `gamma` / `min_split_loss`

**Mechanism:** The minimum gain required for a split to be accepted. Comes directly from
the split gain formula: if $\text{Gain} < \gamma$, the split is rejected.

**Default:** 0 (any positive gain accepted). **Typical range:** 0–5.

**Effect:** Acts as a pruning threshold — higher values produce sparser trees. Start at 0;
add gradually (0.1, 0.5, 1.0) if overfitting is a problem.

#### `reg_lambda` (L2 regularization on leaf weights)

**Default:** 1. **Range:** 0.1–10. Note: XGBoost is *always* L2-regularized by default.
Higher values shrink leaf weights toward zero, analogous to Ridge regression on tree leaves.

#### `reg_alpha` (L1 regularization on leaf weights)

**Default:** 0. **Range:** 0–10. Induces sparsity in leaf weights. Rarely useful unless you
have reason to believe many leaves should be zero.

#### `scale_pos_weight`

**Mechanism:** Multiplier for the gradient of positive class examples. Set to
`n_negative / n_positive` for imbalanced binary classification. This adjusts the effective
sample weights, focusing gradient steps more on the rare class.

**Default:** 1. **Caution:** Setting this improves recall on the minority class but degrades
probability calibration. If you need well-calibrated probabilities, use `scale_pos_weight=1`
and handle imbalance post-hoc with Platt scaling.

### 4.3 LightGBM-Specific Parameters

#### `num_leaves`

**The primary complexity parameter for LightGBM.** Unlike `max_depth` (which limits depth),
`num_leaves` directly controls the maximum number of leaves a tree can have. Since LightGBM
grows leaf-wise, this is the only reliable complexity control.

**Default:** 31. **Rule of thumb:** `num_leaves < 2^(max_depth)` to prevent overfitting.
For small datasets (n < 10,000): 15–31. For large datasets: 31–255.

**Tuning:** This is the first parameter to tune in LightGBM. Large values (> 100) with small
datasets cause severe overfitting.

#### `min_child_samples` / `min_data_in_leaf`

**Default:** 20 (much more conservative than sklearn's default of 1). This is a key reason
LightGBM generalizes well out of the box on larger datasets. For very large datasets ($n > 1$M),
increase to 50–200.

#### `path_smooth`

**Mechanism:** Smooths predictions along the path from root to leaf, preventing extreme values
in leaves with few samples. A value of `path_smooth=1.0` effectively adds a virtual "average"
sample to each leaf prediction.

**Default:** 0 (no smoothing). **Typical value:** 0.5–2.0.

**Usage note:** This is a native LightGBM parameter passed as a `**kwarg` in the sklearn wrapper:
```python
LGBMClassifier(path_smooth=1.0)  # passes as native LGB parameter
```

### 4.4 CatBoost-Specific Parameters

#### `depth` vs `max_depth`

In CatBoost, `depth` controls the depth of symmetric trees. Because symmetric trees apply the
same split at each level, a depth-6 symmetric tree has exactly $2^6 = 64$ leaves. This is
different from XGBoost's `max_depth=6`, which allows up to $2^6 = 64$ leaves but may have
fewer due to the gain threshold.

**Default:** 6. **Typical range:** 4–10.

#### `l2_leaf_reg` (equivalent to `reg_lambda`)

**Default:** 3.0. CatBoost is more aggressively regularized by default than XGBoost
(which defaults to 1.0). This is part of why CatBoost often performs well with fewer tuning
iterations — its defaults are already conservative.

#### `boosting_type`

**`'Ordered'`** (default for small datasets): ordered boosting — eliminates prediction shift,
slower but more accurate. **`'Plain'`**: standard gradient boosting, faster.

**When to use Plain:** On large datasets ($n > 10{,}000$) where prediction shift bias is
negligible and ordered boosting's overhead is not worth it. On very small datasets, ordered
boosting is essential.

### 4.5 Tuning Playbook Table

| Hyperparameter | Library | Typical range | Effect | Tuning strategy |
|---|---|---|---|---|
| `learning_rate` | All | 0.01–0.3 (log scale) | Regularization + convergence speed | Start 0.05; use early stopping for `n_estimators` |
| `n_estimators` | All | 100–5000 | Model capacity | Set high; use early stopping |
| `max_depth` / `depth` | sklearn, XGB, CB | 3–8 | Interaction order | 4–6 for most; tune after learning rate |
| `num_leaves` | LightGBM | 15–255 | Model complexity | Primary LGB parameter; tune first |
| `subsample` | All | 0.5–1.0 | Variance reduction | 0.7–0.8 almost always helps |
| `subsample_freq` | LightGBM | 1 (enable) / 0 (disable) | Enables bagging | Must set ≥1 to activate `subsample` |
| `colsample_bytree` | XGB, LGB | 0.5–1.0 | Feature de-correlation | 0.7–0.9 typical |
| `min_child_samples` | LGB | 10–200 | Leaf regularization | Increase for large $n$ |
| `min_child_weight` | XGB | 1–10 | Leaf regularization | Increase to reduce overfitting |
| `reg_lambda` | XGB, HistGB | 0.1–10.0 | L2 on leaf weights | Try 0.1, 1.0, 10.0 |
| `gamma` | XGB | 0–5 | Split pruning | Start 0; add if overfitting |
| `l2_leaf_reg` | CatBoost | 1–10 | L2 on leaf weights | Default 3 is good; try 1–5 |
| `max_leaf_nodes` | HistGB | 15–255 | Complexity | Like `num_leaves` in LGB |
| `l2_regularization` | HistGB | 0.0–1.0 | L2 regularization | Try 0.0, 0.1, 1.0 |

> 🏆 **Best practice (Optuna HPO):** Use a two-phase tuning strategy. Phase 1: fix
> `learning_rate=0.05`, use early stopping to find the optimal `n_estimators`, and run Optuna
> on `max_depth`/`num_leaves`, `min_child_samples`, `subsample`, and `colsample_bytree`.
> Phase 2: with the best structure found, reduce `learning_rate` to 0.01 and re-run early
> stopping to find the final `n_estimators`. This two-phase approach is standard in
> competition-grade pipelines and consistently outperforms single-phase search.

```python
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

def objective(trial):
    params = {
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators':     500,
        'tree_method':      'hist',
        'device':           'cpu',
        'eval_metric':      'logloss',
        'random_state':     42,
    }
    model = XGBClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)
print("Best params:", study.best_params)
print("Best AUC:",   study.best_value)
```

---

## §5 — Strengths

### 5.1 State-of-the-Art Performance on Tabular Data

GBM variants dominate structured tabular prediction benchmarks. In a 2023 large-scale benchmark
(Grinsztajn et al., arXiv:2207.08815; see also arXiv:2305.17094) evaluating 45 datasets against
deep learning methods, tree ensembles (primarily GBM variants) outperformed neural networks on
tabular data in the majority of settings. The mechanisms are specific:

- **Invariance to monotone transformations of inputs:** Trees split on thresholds, so a feature
  measured in dollars vs. log-dollars gives equivalent models. Neural networks require careful
  normalization.
- **Robustness to irrelevant features:** Trees simply don't split on uninformative features.
  Neural networks mix all features in every layer's matrix multiplication.
- **Handling of irregular decision boundaries:** Tabular data often has complex, non-smooth
  interaction patterns. Trees partition input space explicitly; neural networks must learn smooth
  approximations of sharp boundaries.

> 📊 **Benchmark:** On the OpenML-CC18 benchmark suite (72 classification tasks), XGBoost
> and LightGBM outperform or match deep learning models on 80%+ of datasets with default
> hyperparameters, and on 90%+ after tuning. (Shwartz-Ziv & Armon, 2022.)

### 5.2 Handles Mixed Feature Types Natively

Modern GBM implementations handle numerics, categoricals, and missing values within a single
model with zero preprocessing:

- `HistGradientBoostingClassifier(categorical_features='from_dtype')` auto-detects pandas
  Categorical columns and handles them with histogram-based optimal splits.
- CatBoost accepts raw string categories directly with `cat_features=` — no encoding needed.
- All histogram implementations handle NaN natively — a dedicated bin is allocated, and the
  model learns which branch NaN values should follow during split finding.

**Mechanism:** In histogram-based GBMs, a missing value gets its own histogram bin. During split
finding, the algorithm evaluates both directions (left or right) for the missing value bin and
chooses whichever reduces loss more. This is a learned, principled imputation embedded in the
tree structure — not a heuristic fill value.

### 5.3 Principled Regularization with Many Knobs

GBM has more regularization mechanisms than virtually any other algorithm:
- Shrinkage (learning rate $\nu$) — reduces each tree's contribution
- Stochastic subsampling (row and column) — variance reduction via randomization
- Tree depth / leaf count — controls model capacity directly
- Minimum leaf size — prevents fitting individual points
- L1/L2 regularization on leaf weights (XGBoost, HistGB) — shrinks leaf values
- Minimum split gain (XGBoost `gamma`) — principled pruning threshold
- Interaction constraints — prevents specific feature pairs from interacting
- Monotonic constraints — enforces domain knowledge directly

This richness means an experienced practitioner can precisely control the bias-variance
tradeoff in ways that simpler algorithms cannot match.

### 5.4 Automatic Feature Selection

GBM implicitly performs feature selection through its tree-splitting mechanism. Irrelevant
features simply never get selected for splits (or get selected rarely). The result:

- GBM is robust to having hundreds of noise features added to the dataset
- Feature importances from GBM provide a meaningful signal for feature engineering
- In high-dimensional settings ($p \gg n$), GBM degrades gracefully rather than catastrophically

> ⚠️ **Caveat:** The feature *importance* metric is sensitive to feature cardinality and
> correlation — see §6. But the model's *predictive performance* is robust to noise features
> even if the importance scores are misleading.

### 5.5 No Feature Scaling Required

Trees split on thresholds, not inner products. A feature with values in [0, 1] and a feature
with values in [0, 1,000,000] are handled identically — the split threshold adapts to the
feature's scale. This eliminates a major source of preprocessing error and makes GBMs suitable
for raw, unprocessed data.

### 5.6 Flexible Loss Functions and Custom Objectives

Any differentiable loss function can be plugged into the GBM framework. Standard libraries
support: squared error, absolute error, Huber, log-loss, multinomial deviance, Poisson,
Tweedie, gamma, quantile regression, and survival models. XGBoost additionally allows
fully custom `obj` and `eval` callbacks — you can implement any loss you can differentiate.
This makes GBMs applicable to non-standard prediction targets (counts, revenues, durations).

### 5.7 Built-in Uncertainty Proxies

- **Quantile regression:** sklearn `GradientBoostingRegressor(loss='quantile', alpha=0.9)`
  predicts the 90th percentile of the conditional distribution — an asymmetric loss that trains
  the model to estimate quantiles. Useful for prediction intervals.
- **Multiple quantile prediction:** Fit separate GBMs for $\alpha = 0.1$ and $\alpha = 0.9$ to
  get an 80% prediction interval.
- **NGBoost:** Full distributional output with proper calibration.

---

## §6 — Weaknesses & Failure Modes

### 6.1 Sequential Training — No Parallelism Across Trees

**Mechanism:** Each tree must wait for the previous one to finish, because tree $t$ uses
the residuals computed from tree $t-1$. This is fundamentally sequential. Parallelism within
a single tree (across features and split points) is possible and used by XGBoost/LightGBM,
but the *cross-tree* parallelism available to Random Forests is not.

**Impact:** On very large datasets with time budgets measured in wall-clock hours, Random
Forests can train in parallel and finish faster than GBM. Random Forests also support
perfect parallelism across workers; GBM boosting rounds cannot be distributed across machines
without approximations (XGBoost distributed and LightGBM distributed use data parallelism
on histograms — partial, not perfect, parallelism).

**Detection:** Training time scales linearly with `n_estimators`. If you need 2,000+ trees
and training time is the bottleneck, this is the issue.

**Mitigation:** Use histogram-based methods (LightGBM, XGBoost hist, HistGB) which maximize
within-tree parallelism. Use early stopping to minimize `n_estimators`. For extreme scale,
consider LightGBM's distributed training or XGBoost's Dask/Ray integration.

### 6.2 Overfitting on Small Datasets

**Mechanism:** With sufficient trees and depth, GBM can memorize any training set. On small
datasets ($n < 1{,}000$), even a single deep tree can overfit badly, and stacking hundreds of
them compounds this.

**Detection:** Large gap between training loss and validation loss. Training loss continues to
decrease while validation loss plateaus or increases (classic overfitting).

**Mitigation:**
- Aggressive regularization: high `min_samples_leaf` (or `min_child_samples`), low `max_depth`,
  high `l2_regularization`.
- Reduce `n_estimators` with strong early stopping patience (`n_iter_no_change=5`).
- Use CatBoost with `boosting_type='Ordered'` — its ordered boosting specifically addresses
  bias on small datasets.
- Consider simpler models (logistic regression, Random Forest) for $n < 500$.

### 6.3 Slow Inference for Very Deep Ensembles

**Mechanism:** Inference requires traversing $T$ trees of depth $d$: $O(T \cdot d)$ per sample.
A model with 5,000 trees of depth 6 requires 30,000 node evaluations per sample — this may be
~1–2ms per sample in pure Python, but still orders of magnitude slower than a linear model.

**Detection:** Profile your prediction pipeline. GBM inference is typically 1–100ms per sample
depending on ensemble size; linear models are 0.01ms.

**Mitigation:**
- Use CatBoost's symmetric trees — inference is a $d$-step lookup table operation.
- Reduce `n_estimators` via early stopping.
- Use `tree_method='hist'` in XGBoost, which has highly optimized C++ prediction.
- For ultra-low-latency serving, compile the model to ONNX or use Treelite for model compilation.
- Consider model distillation: train a GBM, then train a linear model or small neural network
  to mimic it for fast serving.

### 6.4 Extrapolation Failure

**Mechanism:** Trees partition the training feature space into rectangular regions. For test
points *outside* the range of training features (e.g., the model is trained on house prices
from 2015–2020 and asked to predict 2025 prices), the model returns the prediction for the
nearest training region — a flat extrapolation. Unlike linear models, GBMs cannot extrapolate
linearly beyond the training distribution.

**Detection:** When your test data has feature values outside training ranges (e.g., economic
data with a regime change, a new product category not seen in training).

**Mitigation:**
- Monitor feature distributions for distributional shift in production.
- For time series, include time as a feature to allow the model to extrapolate along the time
  axis.
- Hybrid models: use a linear model for the extrapolation component and GBM for the residuals.
- Use domain knowledge to constrain predictions with monotonic constraints.

### 6.5 Split-Based Feature Importance Is Biased

**Mechanism:** The default feature importance in sklearn, XGBoost, and LightGBM counts either
(a) how many times a feature is used in splits ("split importance") or (b) the total gain from
splits on a feature ("gain importance"). Both are biased toward high-cardinality features:
a continuous variable with 1,000 unique values has many more candidate split points than a
binary variable, giving it more opportunities to be selected.

**Detection:** If your most important features by split count are continuous high-cardinality
variables, and your SHAP values show a different ranking, you have importance bias.

**Mitigation:** Always use SHAP values (`shap.TreeExplainer`) or permutation importance
(`sklearn.inspection.permutation_importance`) as your primary feature importance measure.
Split-based importance is a useful quick sanity check but not a reliable ranking.

### 6.6 Probability Calibration

**Mechanism:** GBM probability outputs are not automatically well-calibrated. The model
optimizes log-loss (which is proper), so in the limit of infinite data and capacity, it
converges to true probabilities. But with finite data and regularization, the outputs can
be systematically over- or under-confident.

**Detection:** Plot a calibration curve (`sklearn.calibration.CalibrationDisplay`). Well-calibrated
models show points near the diagonal (predicted probability = observed frequency).

**Mitigation:** Post-hoc calibration with `sklearn.calibration.CalibratedClassifierCV` using
Platt scaling (`method='sigmoid'`) or isotonic regression (`method='isotonic'`). Apply on a
held-out calibration set.

### 6.7 Correlated Features and SHAP Attribution

**Mechanism:** When two features are highly correlated ($r > 0.9$), SHAP values are split
between them. The model might use either feature interchangeably, and the attribution depends
on which the tree happened to use. Permutation importance also behaves oddly — swapping one
highly correlated feature barely hurts performance because the other serves as a backup.

**Detection:** Spearman correlation matrix between important features. High pairwise correlations
signal potential attribution instability.

**Mitigation:** Use SHAP interaction values to understand how correlated features jointly
contribute. For feature selection, use hierarchical clustering on features and select one
from each cluster, rather than ranking by individual importance.

### 6.8 Class Imbalance: Calibration vs. Discrimination Tradeoff

**Mechanism:** With highly imbalanced classes (1:100 or worse), the GBM's log-loss objective
is dominated by the majority class. Predictions concentrate near zero (predicting majority
class) with near-zero probabilities for the minority class.

**Mitigation options (each with tradeoffs):**
- XGBoost `scale_pos_weight = n_neg/n_pos`: rescales gradients for positive class.
  Better sensitivity, but degrades calibration.
- LightGBM `is_unbalance=True` or `class_weight='balanced'`: similar effect.
- CatBoost `auto_class_weights='Balanced'`: automatic class reweighting.
- Oversample minority class (SMOTE) before training: improves sensitivity but risks
  learning from synthetic data patterns.
- **Best for calibrated probabilities:** train without reweighting, then post-hoc calibrate
  with isotonic regression on a balanced calibration set.

> 🏆 **Best practice:** For imbalanced classification where calibrated probabilities matter
> (e.g., clinical risk scoring), use standard training with no class reweighting, then
> calibrate post-hoc. Use `scale_pos_weight` / `is_unbalance` only when recall/sensitivity
> is more important than probability accuracy.

---

## §7 — Diagnostic Plots & Evaluation

A GBM model without diagnostics is a black box you are flying blind in. This section gives you
the complete toolkit: the right metrics, the right plots, and how to read each one.

### 7.1 Choosing Your Evaluation Metric

**For regression:**
- **RMSE** (`mean_squared_error(squared=False)`): penalizes large errors quadratically. Good
  when large prediction errors are disproportionately costly.
- **MAE** (`mean_absolute_error`): robust to outliers; easier to interpret as average error size.
- **R²** (`r2_score`): proportion of variance explained. Useful for communicating results.

**For classification:**
- **ROC-AUC**: discrimination ability across all thresholds. Primary metric for imbalanced data.
  Insensitive to class imbalance and threshold choice.
- **PR-AUC** (average precision): better than ROC-AUC for severe imbalance (1:100+).
- **Log-loss**: penalizes miscalibrated probabilities. Use when probability quality matters.

> 🔧 **In practice:** For most tabular classification problems, optimize ROC-AUC during
> hyperparameter tuning. Use log-loss as a secondary metric to ensure probabilities are not
> miscalibrated. Set your decision threshold based on precision/recall requirements, not the
> default 0.5.

### 7.2 Learning Curves — Diagnosing Fit Quality

Learning curves plot train and validation loss against boosting rounds. They are the single
most informative diagnostic for GBM.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

train_losses, val_losses = [], []
n_estimators_range = range(10, 501, 10)

model = GradientBoostingClassifier(
    learning_rate=0.05, max_depth=4, subsample=0.8,
    random_state=42, warm_start=True
)

for n_est in n_estimators_range:
    model.set_params(n_estimators=n_est)
    model.fit(X_train, y_train)
    train_losses.append(log_loss(y_train, model.predict_proba(X_train)))
    val_losses.append(log_loss(y_val,   model.predict_proba(X_val)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(n_estimators_range), train_losses, label='Train log-loss', color='steelblue')
ax.plot(list(n_estimators_range), val_losses,   label='Val log-loss',   color='darkorange')
best_n = list(n_estimators_range)[np.argmin(val_losses)]
ax.axvline(best_n, ls='--', color='red', label=f'Best n_estimators={best_n}')
ax.set_xlabel('Number of boosting rounds')
ax.set_ylabel('Log-loss')
ax.set_title('GBM Learning Curves')
ax.legend()
plt.tight_layout()
plt.show()
```

**Reading the learning curve:**
- **Good fit:** Train and val losses decrease together, then val loss plateaus. The optimal
  `n_estimators` is the val loss minimum.
- **Overfitting:** Val loss starts rising while train loss continues falling. You need more
  regularization — reduce `max_depth`, increase `min_samples_leaf`, increase `l2_regularization`.
- **Underfitting:** Both curves are high. Increase `n_estimators`, increase `max_depth` or
  `num_leaves`, or increase `learning_rate`.
- **Noisy val curve:** Low `subsample` causing high variance. Increase `subsample` or use a
  larger validation set.

For XGBoost, the eval history is tracked automatically:
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8,
    tree_method='hist', device='cpu',
    eval_metric=['logloss', 'auc'], early_stopping_rounds=50,
    random_state=42, verbosity=0
)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

results = model.evals_result()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ['logloss', 'auc']):
    ax.plot(results['validation_0'][metric], label='Train')
    ax.plot(results['validation_1'][metric], label='Val')
    ax.axvline(model.best_iteration, ls='--', color='red',
               label=f'Best={model.best_iteration}')
    ax.set_xlabel('Boosting round'); ax.set_ylabel(metric.upper())
    ax.legend(); ax.set_title(f'XGBoost {metric}')
plt.tight_layout(); plt.show()
```

### 7.3 Feature Importance Plots

Always compare at least two importance types — they measure fundamentally different things.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# ---------- Split-based importance (fast but biased) ----------
feat_imp = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=True).tail(20)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
feat_imp.plot.barh(ax=axes[0], color='steelblue')
axes[0].set_title('Split-count Importance\n(biased toward high-cardinality features)')
axes[0].set_xlabel('Importance')

# ---------- Permutation importance (unbiased, slower) ----------
perm_result = permutation_importance(
    model, X_val, y_val, n_repeats=10,
    scoring='roc_auc', random_state=42
)
perm_imp = pd.DataFrame({
    'mean': perm_result.importances_mean,
    'std':  perm_result.importances_std
}, index=feature_names).sort_values('mean', ascending=True).tail(20)

axes[1].barh(perm_imp.index, perm_imp['mean'],
             xerr=perm_imp['std'], color='darkorange', alpha=0.7)
axes[1].set_title('Permutation Importance\n(measures actual AUC degradation on val set)')
axes[1].set_xlabel('Mean AUC decrease')
plt.suptitle('Feature Importance Comparison', fontsize=13)
plt.tight_layout(); plt.show()
```

**Reading importance plots:**
- If split importance and permutation importance agree on top features: trust both.
- If they disagree: prefer permutation importance for feature selection decisions.
- Features important by split count but unimportant by permutation are likely high-cardinality
  features exploited structurally without genuine predictive value.
- Negative permutation importance = the feature adds noise; strongly consider removing it.

### 7.4 Residual Plots (Regression)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(y_pred, y_test, alpha=0.3, s=8)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title(f'Predicted vs Actual (R²={r2_score(y_test,y_pred):.3f})')

axes[1].scatter(y_pred, residuals, alpha=0.3, s=8)
axes[1].axhline(0, color='red', lw=2)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Predicted')

axes[2].hist(residuals, bins=60, edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Residual'); axes[2].set_ylabel('Count')
axes[2].set_title('Residual Distribution')

plt.suptitle(f'HistGBM Regression Diagnostics (RMSE={rmse:.3f})')
plt.tight_layout(); plt.show()
```

**Reading residual plots:**
- **Good:** Uniform cloud around zero residuals, symmetric distribution.
- **Heteroscedasticity (fan shape):** Model is less accurate at extreme predicted values.
  Consider log-transforming the target or using Huber loss.
- **Systematic curvature (arch):** Missing a nonlinear relationship. Increase depth or add
  interaction features.
- **Heavy tails:** Extreme residuals on a subset of data — investigate those points.

### 7.5 Calibration Curves (Classification)

```python
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV

fig, ax = plt.subplots(figsize=(7, 6))
CalibrationDisplay.from_estimator(
    model, X_test, y_test, n_bins=10, ax=ax, name='GBM (uncalibrated)'
)

# Post-hoc isotonic calibration on a held-out calibration split
calib_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calib_model.fit(X_cal, y_cal)
CalibrationDisplay.from_estimator(
    calib_model, X_test, y_test, n_bins=10, ax=ax, name='GBM + isotonic calibration'
)

ax.set_title('Calibration Curves'); plt.legend(loc='upper left')
plt.tight_layout(); plt.show()
```

**Reading calibration curves:**
- **Perfect calibration:** Points on the diagonal (predicted prob = observed frequency).
- **Over-confident (below diagonal):** Model assigns probabilities too high.
- **Under-confident (above diagonal):** Model assigns probabilities too low.
- **S-shaped curve:** Classic uncalibrated GBM output — apply Platt scaling or isotonic regression.

---

## §8 — Explainability & Interpretable AI

GBMs have built-in feature importance, but the real explainability toolkit is SHAP. For
tree ensembles, SHAP's TreeExplainer provides mathematically exact Shapley values in polynomial
time — no approximation, no sampling noise.

### 8.1 Built-in Feature Importance — What It Measures and Why It's Insufficient

All GBM libraries expose `model.feature_importances_`. But the value of this attribute depends
on what metric underlies it:

- **Split count** (LightGBM default `importance_type='split'`, XGBoost default): counts how
  many times each feature is used in a split across all trees. Biased toward high-cardinality
  and frequently-used features.
- **Gain-weighted** (set `importance_type='gain'` in LightGBM/XGBoost): total information gain
  attributed to each feature's splits. Less biased than split count, but still affected by
  feature scale and range.

Neither is a reliable feature ranking for feature selection. Use SHAP instead.

### 8.2 SHAP TreeExplainer — Exact Shapley Values

> 📜 **Citation:** Lundberg, S.M. et al. (2020). "From Local Explanations to Global Understanding
> with Explainable AI for Trees." *Nature Machine Intelligence*, 2, 56–67.
> TreeSHAP complexity: $O(T \cdot D \cdot 2^D)$ where $T$ = trees, $D$ = max depth.

SHAP TreeExplainer computes exact Shapley values for tree ensembles. For a GBM with 500 trees
of depth 6, this takes seconds — not minutes.

```python
import shap
import matplotlib.pyplot as plt

# --- Option 1: Auto-detect (recommended for all GBM variants) ---
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)   # Returns Explanation object

# --- Option 2: Explicit TreeExplainer ---
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_test)  # numpy array, shape (n, p)
```

#### SHAP Beeswarm Summary Plot — Global Feature Importance

```python
# Beeswarm: shows distribution of SHAP values for each feature
shap.plots.beeswarm(shap_values)
# OR (older API):
# shap.summary_plot(shap_values_raw, X_test, feature_names=feature_names)
```

**Reading the beeswarm plot:**
- Each dot = one prediction. X-axis = SHAP value (impact on log-odds/prediction).
- Color = feature value (red = high, blue = low).
- Features sorted by mean absolute SHAP value (most important at top).
- A feature whose dots spread widely has high average impact on predictions.
- Red dots on the right: high feature value → higher predicted output (positive correlation).
- Blue dots on the right: low feature value → higher predicted output (negative correlation).

#### SHAP Waterfall Plot — Single Prediction Explanation

```python
# Waterfall: breaks down one prediction into feature contributions
shap.plots.waterfall(shap_values[0])
```

**Reading the waterfall plot:**
- Each bar = one feature's contribution to shifting the prediction from the base value (mean
  prediction) to the final prediction for this sample.
- Red bars push the prediction up; blue bars push it down.
- Sum of all bars = final prediction − base value.
- The "base value" at the left is $E[F(\mathbf{x})] = $ expected model output over training data.

#### SHAP Force Plot — Interactive Single Prediction

```python
# Force plot: compact visualization of one prediction
shap.plots.force(shap_values[0])
# For multiple predictions (interactive):
shap.plots.force(shap_values[:100])  # shows 100 predictions stacked
```

#### SHAP Dependence Plot — Feature Interaction Visualization

```python
# Dependence plot: SHAP value vs feature value, colored by an interaction feature
feature_idx = 0  # index of feature to examine
shap.dependence_plot(
    feature_idx,
    shap_values_raw,
    X_test,
    feature_names=feature_names,
    interaction_index='auto'   # auto-detects strongest interaction feature
)
```

**Reading the dependence plot:**
- X-axis: raw feature value. Y-axis: SHAP value for that feature.
- Color: value of the auto-detected interaction feature.
- A smooth curve = the feature has a monotone main effect.
- Color gradients show that the feature's effect *differs* depending on another feature —
  this is an interaction effect.

#### SHAP for Multi-class Classification

```python
# Multi-class: shap_values has shape (n_classes, n_samples, n_features) for some models
# With XGBoost multi-class:
explainer = shap.TreeExplainer(multiclass_model)
shap_vals = explainer.shap_values(X_test)  # list of length n_classes

# Plot for class 0
shap.summary_plot(shap_vals[0], X_test, feature_names=feature_names,
                  title='SHAP values for class 0')
```

### 8.3 LIME — Local Surrogate Explanations

LIME (Local Interpretable Model-Agnostic Explanations) fits a linear model locally around each
prediction. It is model-agnostic and slower than SHAP TreeExplainer, but can be useful when
you need a simple linear explanation for a single prediction.

```python
# pip install lime
import lime
import lime.lime_tabular

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['malignant', 'benign'],
    mode='classification'
)

# Explain one prediction
exp = explainer_lime.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=10
)
exp.show_in_notebook()
# or: exp.as_pyplot_figure()
```

**When to use LIME over SHAP:** LIME is useful for models where TreeSHAP is not available
(non-tree models, ensembles with non-tree base learners, NGBoost). For standard GBM variants,
prefer SHAP TreeExplainer — it is faster, more consistent, and mathematically exact.

### 8.4 Partial Dependence Plots (PDP) and ICE Curves

PDPs show the marginal effect of one or two features on predictions, averaged over all other
features. ICE (Individual Conditional Expectation) curves show the same relationship for each
individual training example — revealing heterogeneous effects.

```python
from sklearn.inspection import PartialDependenceDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1D PDP: marginal effect of feature 0
PartialDependenceDisplay.from_estimator(
    model, X_test,
    features=[0, 2],          # feature indices to plot
    feature_names=feature_names,
    kind='both',              # 'average' = PDP, 'individual' = ICE, 'both' = both
    subsample=200,            # use 200 samples for ICE (faster)
    ax=axes,
    random_state=42
)
plt.suptitle('Partial Dependence Plots with ICE Curves', fontsize=12)
plt.tight_layout(); plt.show()

# 2D PDP (interaction heatmap) — show interaction between two features
fig, ax = plt.subplots(figsize=(7, 6))
PartialDependenceDisplay.from_estimator(
    model, X_test,
    features=[(0, 2)],        # tuple = 2D interaction plot
    feature_names=feature_names,
    ax=ax
)
plt.tight_layout(); plt.show()
```

**Reading PDP/ICE plots:**
- **PDP line (yellow/average):** The average marginal effect. A flat line = the feature has
  no effect on average. A rising line = positive marginal effect. A non-monotone line =
  complex nonlinear relationship.
- **ICE lines (individual):** Each line = one sample. When ICE lines diverge or cross,
  you have **heterogeneous treatment effects** — the feature's impact differs across subgroups.
  Crossing ICE lines reveal interaction effects.
- **SHAP dependence plots are generally more informative than 1D PDPs** because they also
  show the interaction feature.

### 8.5 Global vs Local Explanations — Summary

| Tool | Type | Correctness | Speed | When to use |
|---|---|---|---|---|
| `feature_importances_` | Global | Biased (split count) | Instant | Quick sanity check only |
| `permutation_importance` | Global | Unbiased | Slow | Feature selection, ranking |
| SHAP beeswarm | Global | Exact (TreeSHAP) | Fast | Primary global explanation |
| SHAP waterfall/force | Local | Exact (TreeSHAP) | Fast | Single prediction explanation |
| SHAP dependence | Local→Global | Exact (TreeSHAP) | Fast | Feature effect + interactions |
| PDP | Global | Marginal average | Medium | Main effects, simple reports |
| ICE | Local→Global | Marginal per-sample | Medium | Heterogeneous effects |
| LIME | Local | Approximate | Slow | When SHAP TreeExplainer unavailable |

### 8.6 Caveats and Pitfalls in Interpretation

> ⚠️ **Pitfall (correlated features):** SHAP splits attribution between correlated features.
> If `income` and `credit_score` are highly correlated ($r=0.9$), SHAP will show both as
> important but with attribution that depends on which one the tree happened to split on first.
> The correct interpretation is "this cluster of correlated features is collectively important"
> — not a precise ranking of individual feature contributions.

> ⚠️ **Pitfall (PDP marginal assumption):** PDPs compute $E_{\mathbf{x}_{-j}}[F(x_j, \mathbf{x}_{-j})]$
> by averaging over the empirical marginal distribution. If $x_j$ and other features are
> correlated, this marginalizes over *impossible* feature combinations (e.g., house with
> 10 bedrooms and 400 sq ft). PDPs can be misleading in the presence of correlations; SHAP
> conditional expectation is more principled but harder to compute.

> ⚠️ **Pitfall (SHAP + CatBoost):** For CatBoost's symmetric trees, some SHAP versions require
> `shap.TreeExplainer(model, data=background_data, model_output='raw')` to get margin-level
> (logit-space) SHAP values rather than probability-space SHAP values. Verify with your exact
> shap and catboost versions — shapes and interpretations differ between these modes.

---

## §9 — Complete Python Worked Example

Two complete examples: one regression (California Housing) and one classification (Adult Income).
Each runs through the full pipeline: load → EDA → preprocess → fit → tune → diagnose → SHAP.

### 9.1 Regression: California Housing with HistGradientBoosting + XGBoost

```python
# ============================================================
# GBM Regression Masterclass — California Housing Dataset
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
df = housing.frame
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']
feature_names = X.columns.tolist()

print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")
print(X.describe().round(2))

# ─────────────────────────────────────────────
# 2. EDA — Quick distributions and correlations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.flat, feature_names):
    ax.hist(X[col], bins=40, edgecolor='none', alpha=0.7)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel('Value', fontsize=7)
plt.suptitle('Feature Distributions — California Housing', fontsize=12)
plt.tight_layout()
plt.show()

# Target distribution: right-skewed, clipped at 5.0
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(y, bins=50, edgecolor='none')
axes[0].set_title('Target (Median House Value) Distribution')
axes[0].set_xlabel('$100k USD')
axes[1].hist(np.log(y), bins=50, edgecolor='none', color='darkorange')
axes[1].set_title('Log-transformed Target Distribution')
axes[1].set_xlabel('log(MedHouseVal)')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 3. TRAIN/VAL/TEST SPLIT
# ─────────────────────────────────────────────
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42
)
print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 4. BASELINE MODEL (HistGradientBoosting)
# ─────────────────────────────────────────────
baseline = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.1,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.1,
    random_state=42
)
baseline.fit(X_train_full, y_train_full)
y_pred_base = baseline.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base  = mean_absolute_error(y_test, y_pred_base)
r2_base   = r2_score(y_test, y_pred_base)
print(f"\nBaseline HistGBM:")
print(f"  RMSE = {rmse_base:.4f}  MAE = {mae_base:.4f}  R² = {r2_base:.4f}")
print(f"  Stopped at iteration: {baseline.n_iter_}")

# ─────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING WITH OPTUNA
# ─────────────────────────────────────────────
from xgboost import XGBRegressor

def objective(trial):
    params = {
        'n_estimators':     500,          # fixed high; early stopping finds optimum
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        'gamma':            trial.suggest_float('gamma', 0.0, 2.0),
        'tree_method':      'hist',
        'device':           'cpu',
        'eval_metric':      'rmse',
        'early_stopping_rounds': 30,
        'random_state':     42,
        'verbosity':        0,
    }
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, timeout=300, show_progress_bar=False)
print(f"\nOptuna best RMSE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# ─────────────────────────────────────────────
# 6. FINAL TUNED MODEL
# ─────────────────────────────────────────────
best_params = study.best_params.copy()
best_params.update({
    'n_estimators': 2000,
    'tree_method': 'hist',
    'device': 'cpu',
    'eval_metric': 'rmse',
    'early_stopping_rounds': 50,
    'random_state': 42,
    'verbosity': 0,
})

tuned_model = XGBRegressor(**best_params)
tuned_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_tuned = tuned_model.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mae_tuned  = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned   = r2_score(y_test, y_pred_tuned)

print(f"\nTuned XGBoost:")
print(f"  RMSE = {rmse_tuned:.4f}  MAE = {mae_tuned:.4f}  R² = {r2_tuned:.4f}")
print(f"  Best iteration: {tuned_model.best_iteration}")
print(f"  Improvement over baseline: {(rmse_base - rmse_tuned)/rmse_base*100:.1f}% RMSE reduction")

# ─────────────────────────────────────────────
# 7. DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────
residuals = y_test.values - y_pred_tuned
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(y_pred_tuned, y_test, alpha=0.3, s=8, color='steelblue')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Predicted ($100k)'); axes[0].set_ylabel('Actual ($100k)')
axes[0].set_title(f'Predicted vs Actual (R²={r2_tuned:.3f})')

axes[1].scatter(y_pred_tuned, residuals, alpha=0.3, s=8, color='darkorange')
axes[1].axhline(0, color='red', lw=2)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residuals')
axes[1].set_title(f'Residuals vs Predicted (RMSE={rmse_tuned:.3f})')

axes[2].hist(residuals, bins=60, edgecolor='none', alpha=0.8, color='mediumseagreen')
axes[2].axvline(0, color='red', lw=2)
axes[2].set_xlabel('Residual'); axes[2].set_ylabel('Count')
axes[2].set_title('Residual Distribution')

plt.suptitle('XGBoost Regression Diagnostics — California Housing', fontsize=12)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 8. SHAP EXPLANATIONS
# ─────────────────────────────────────────────
X_test_sample = X_test.iloc[:500]   # use a sample for speed

explainer = shap.Explainer(tuned_model, X_train)
shap_values = explainer(X_test_sample)

# Global: beeswarm summary
plt.figure()
shap.plots.beeswarm(shap_values, max_display=8)
plt.title('SHAP Beeswarm — Global Feature Importance')
plt.tight_layout()
plt.show()

# Local: waterfall for one prediction
idx = 0
print(f"\nSHAP Explanation — Sample {idx}")
print(f"  True value: ${y_test.iloc[idx]*100:.0f}k, Predicted: ${y_pred_tuned[idx]*100:.0f}k")
shap.plots.waterfall(shap_values[idx])

# Dependence plot for most important feature
most_imp_feat = feature_names[np.argmax(np.abs(shap_values.values).mean(0))]
print(f"\nMost important feature by mean |SHAP|: {most_imp_feat}")
shap.dependence_plot(
    most_imp_feat,
    shap_values.values,
    X_test_sample,
    interaction_index='auto'
)

# ─────────────────────────────────────────────
# 9. PARTIAL DEPENDENCE PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
PartialDependenceDisplay.from_estimator(
    tuned_model, X_test_sample,
    features=['MedInc', 'Latitude'],
    feature_names=feature_names,
    kind='both',
    subsample=100,
    ax=axes,
    random_state=42
)
plt.suptitle('Partial Dependence Plots with ICE Curves', fontsize=12)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 10. FINAL INTERPRETATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY — California Housing")
print("="*60)
print(f"Baseline HistGBM (default params):  RMSE={rmse_base:.4f}, R²={r2_base:.4f}")
print(f"Tuned XGBoost (Optuna 50 trials):   RMSE={rmse_tuned:.4f}, R²={r2_tuned:.4f}")

# SHAP feature ranking
mean_abs_shap = pd.Series(
    np.abs(shap_values.values).mean(0),
    index=feature_names
).sort_values(ascending=False)
print("\nSHAP Feature Importance Ranking (mean |SHAP|):")
for feat, imp in mean_abs_shap.items():
    print(f"  {feat:20s}: {imp:.4f}")
print("\nInterpretation:")
print("  MedInc (median income) is the dominant predictor — the SHAP")
print("  beeswarm shows high income strongly predicts high house values,")
print("  but the dependence plot reveals diminishing returns at high incomes.")
print("  Latitude and Longitude interact: coastal areas (low latitude, mid-")
print("  longitude for CA) command price premiums — this geographic interaction")
print("  is captured by the 2D PDP. AveOccup (average occupancy) has a")
print("  nonlinear negative effect: very high occupancy (crowding) reduces")
print("  predicted values sharply.")
```

### 9.2 Classification: Adult Income with LightGBM

```python
# ============================================================
# GBM Classification — Adult Income Dataset
# (Predict whether income > $50k)
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, RocCurveDisplay,
                              PrecisionRecallDisplay)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibrationDisplay
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
X_raw = adult.data.copy()
y_raw = (adult.target == '>50K').astype(int)

print(f"Dataset: {X_raw.shape[0]:,} samples, {X_raw.shape[1]} features")
print(f"Class balance: {y_raw.mean()*100:.1f}% positive (>50K)")
print(f"\nFeature dtypes:\n{X_raw.dtypes.value_counts()}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# LightGBM handles categoricals natively with pd.Categorical dtype
# ─────────────────────────────────────────────
cat_cols = X_raw.select_dtypes(include=['category', 'object']).columns.tolist()
num_cols = X_raw.select_dtypes(include=['number']).columns.tolist()

for col in cat_cols:
    X_raw[col] = X_raw[col].astype('category')

print(f"\nCategorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numeric features ({len(num_cols)}): {num_cols}")

# Missing values — LightGBM handles NaN natively; no imputation needed
print(f"\nMissing values per feature:\n{X_raw.isnull().sum()[X_raw.isnull().sum() > 0]}")

# ─────────────────────────────────────────────
# 3. TRAIN/VAL/TEST SPLIT (stratified)
# ─────────────────────────────────────────────
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_raw, y_raw, test_size=0.15, random_state=42, stratify=y_raw
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)
print(f"\nTrain: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 4. BASELINE MODEL
# ─────────────────────────────────────────────
baseline_lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,       # MUST enable subsample
    colsample_bytree=0.8,
    random_state=42
)
baseline_lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(50), log_evaluation(-1)]
)

y_prob_base = baseline_lgbm.predict_proba(X_test)[:, 1]
auc_base  = roc_auc_score(y_test, y_prob_base)
pr_base   = average_precision_score(y_test, y_prob_base)
print(f"\nBaseline LightGBM:")
print(f"  ROC-AUC = {auc_base:.4f}  PR-AUC = {pr_base:.4f}")
print(f"  Best iteration: {baseline_lgbm.best_iteration_}")

# ─────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING WITH OPTUNA
# ─────────────────────────────────────────────
def lgbm_objective(trial):
    params = {
        'n_estimators':      1000,
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq':    1,
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 0.5),
        'random_state':      42,
    }
    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(30, verbose=False), log_evaluation(-1)]
    )
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(lgbm_objective, n_trials=50, timeout=300)
print(f"\nOptuna best AUC: {study_lgbm.best_value:.4f}")

# ─────────────────────────────────────────────
# 6. FINAL TUNED MODEL ON FULL TRAINING DATA
# ─────────────────────────────────────────────
final_params = study_lgbm.best_params.copy()
final_params.update({
    'n_estimators': 2000, 'subsample_freq': 1, 'random_state': 42
})

final_model = LGBMClassifier(**final_params)
final_model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)]
)

y_prob_final = final_model.predict_proba(X_test)[:, 1]
y_pred_final = final_model.predict(X_test)

auc_final = roc_auc_score(y_test, y_prob_final)
pr_final  = average_precision_score(y_test, y_prob_final)
print(f"\nTuned LightGBM (on full train set):")
print(f"  ROC-AUC = {auc_final:.4f}  PR-AUC = {pr_final:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['<=50K', '>50K']))

# ─────────────────────────────────────────────
# 7. DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ROC Curve
RocCurveDisplay.from_estimator(final_model, X_test, y_test, ax=axes[0], name='LightGBM')
axes[0].set_title(f'ROC Curve (AUC={auc_final:.3f})')

# Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(final_model, X_test, y_test, ax=axes[1], name='LightGBM')
axes[1].set_title(f'PR Curve (AP={pr_final:.3f})')

# Calibration Curve
CalibrationDisplay.from_estimator(
    final_model, X_test, y_test, n_bins=10, ax=axes[2], name='LightGBM'
)
axes[2].set_title('Calibration Curve')

plt.suptitle('LightGBM Classification Diagnostics — Adult Income', fontsize=12)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 8. SHAP EXPLANATIONS
# ─────────────────────────────────────────────
# LightGBM: use TreeExplainer; SHAP values are in log-odds (margin) space
X_test_sample = X_test.iloc[:500]

explainer = shap.TreeExplainer(final_model)
shap_values_lgbm = explainer.shap_values(X_test_sample)

# For binary classification, LightGBM TreeExplainer returns a list [class_0, class_1]
# Take class 1 (>50K)
if isinstance(shap_values_lgbm, list):
    sv = shap_values_lgbm[1]
else:
    sv = shap_values_lgbm

# Beeswarm (needs Explanation object for plots.beeswarm)
shap_explanation = shap.Explanation(
    values=sv,
    base_values=explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value,
    data=X_test_sample.values,
    feature_names=X_test_sample.columns.tolist()
)
plt.figure()
shap.plots.beeswarm(shap_explanation, max_display=10)
plt.title('SHAP Beeswarm — LightGBM Adult Income')
plt.tight_layout()
plt.show()

# Waterfall for a high-income prediction
high_income_idx = np.where(y_test.values == 1)[0][0]
shap.plots.waterfall(shap_explanation[high_income_idx])

# Permutation importance comparison
from sklearn.inspection import permutation_importance
perm = permutation_importance(
    final_model, X_test, y_test,
    n_repeats=5, scoring='roc_auc', random_state=42
)
perm_df = pd.DataFrame({
    'feature': X_test.columns,
    'shap_imp': np.abs(sv).mean(0),
    'perm_imp': perm.importances_mean
}).sort_values('shap_imp', ascending=False)
print(f"\nFeature Importance Comparison (top 10):")
print(perm_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 9. INTERPRETATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL INTERPRETATION — Adult Income Dataset")
print("="*60)
print(f"ROC-AUC: {auc_final:.4f} — Excellent discrimination")
print(f"PR-AUC:  {pr_final:.4f} — Good precision-recall balance")
print()
print("Top predictive features (by mean |SHAP|):")
for _, row in perm_df.head(5).iterrows():
    print(f"  {row['feature']:25s} SHAP={row['shap_imp']:.4f}  Perm={row['perm_imp']:.4f}")
print()
print("Key findings from SHAP:")
print("  - Capital gain and capital loss are the strongest predictors —")
print("    any nonzero capital gain/loss indicates investment activity,")
print("    strongly associated with higher income.")
print("  - Education and age show gradual positive effects.")
print("  - Occupation and marital status are important categorical features.")
print("  - Hours-per-week has a threshold effect: working > 40h/week")
print("    predicts higher income, but the marginal effect plateaus.")
```

---

## §10 — When to Use This Algorithm (Decision Guide)

### 10.1 Decision Flowchart

Use this flowchart to decide whether GBM is right for your problem, and which variant to use:

```
Is your data TABULAR (rows = samples, columns = features)?
├── NO  → Consider neural networks (images, text, audio, graphs)
└── YES → Continue

Do you have VERY FEW SAMPLES (n < 200)?
├── YES → Consider logistic regression, Ridge, SVM (GBM will overfit)
└── NO  → Continue

Is INTERPRETABILITY the ONLY requirement (regulator demands linear model)?
├── YES → Use logistic regression or linear/Ridge regression
└── NO  → Continue

Do you need a FAST BASELINE in < 10 minutes with minimal preprocessing?
├── YES → Use HistGradientBoosting* (handles NaN + categoricals natively)
└── NO  → Continue

Do you have MANY HIGH-CARDINALITY CATEGORICAL features (user IDs, geo, SKUs)?
├── YES → Use CatBoost (native ordered target statistics, no encoding needed)
└── NO  → Continue

Is YOUR DATASET VERY LARGE (n > 1 million rows)?
├── YES → Use LightGBM (fastest CPU training, GOSS subsampling)
│         or XGBoost with device='cuda' (GPU, excellent ecosystem)
└── NO  → Continue

Do you need CUSTOM LOSS FUNCTIONS or GPU acceleration?
├── YES → Use XGBoost (custom obj/eval, CUDA, Dask/Ray distributed)
└── NO  → Continue

Do you need FULL PROBABILITY DISTRIBUTIONS (prediction intervals, uncertainty)?
├── YES → Use NGBoost (Normal, LogNormal, Exponential, k_categorical)
└── NO  → Use XGBoost or LightGBM with standard tuning
```

### 10.2 When GBM Is the Right Tool

| Scenario | Recommended variant | Why |
|---|---|---|
| Kaggle tabular competition | XGBoost or LightGBM | Dominant track record, best ecosystem |
| Fast prototyping, mixed types | HistGradientBoosting* | Zero preprocessing, handles NaN/cat |
| Large dataset (>1M rows), CPU | LightGBM | GOSS + EFB = fastest CPU training |
| Large dataset, GPU available | XGBoost (device='cuda') | Excellent CUDA implementation |
| Many categorical features | CatBoost | Ordered TS eliminates encoding overhead |
| Clinical/financial risk scores | XGBoost + SHAP + calibration | Best explainability + calibration ecosystem |
| Time series (tabular features) | LightGBM or XGBoost | Handles temporal features well with lag features |
| Regression with outliers | GBM with Huber/quantile loss | Robust loss functions |
| Probabilistic regression | NGBoost | Outputs distributional predictions |

### 10.3 When GBM Is NOT the Right Tool

| Scenario | Problem | Better alternative |
|---|---|---|
| Image/text/audio data | GBM can't process raw pixels/tokens | CNN, Transformer, BERT |
| Very small data (n < 300) | Overfitting, poor generalization | Logistic regression, SVM, Ridge |
| Online/streaming learning | Sequential trees require all data | River (online ML), SGD |
| Exact probabilistic guarantees | GBM probabilities need calibration | Calibrated GBM, Bayesian models |
| Regulatory linear model required | Decision trees are not linear | Logistic regression, scorecard |
| Extremely low-latency (<0.1ms) | 1000-tree ensemble is slow | Distill to linear model, ONNX/Treelite |
| Sparse high-dimensional text | Bag-of-words = 100k features, GBM slow | Logistic with L1 (Lasso), linear SVM |
| Pure extrapolation needed | Trees cannot extrapolate | Linear/polynomial regression |

### 10.4 GBM vs Random Forest — When Each Wins

Both are tree ensembles. The practical choice:

| Factor | Favor GBM | Favor Random Forest |
|---|---|---|
| **Accuracy on tabular data** | GBM almost always wins with tuning | RF is competitive out-of-the-box |
| **Tuning effort** | GBM requires careful HPO | RF is robust to hyperparameters |
| **Training time** | Comparable (HistGBM); RF parallelizes better | RF with many cores |
| **Out-of-bag estimates** | Not available | Built-in OOB error + feature importance |
| **Noisy/outlier-heavy data** | Sensitive to outliers (L2 loss) | More robust (averaging) |
| **Small datasets** | Overfits more easily | More conservative |
| **Prototype speed** | Requires learning rate tuning | `RandomForest(n_estimators=200)` just works |

**Rule of thumb:** When in doubt on a new tabular problem, fit both a GBM (HistGBM or XGBoost
with `n_estimators=300, learning_rate=0.05`) and a Random Forest side by side. GBM usually
wins on accuracy after tuning; RF wins on simplicity and robustness.

---

## §11 — Resources & Further Reading

### Original Papers (Essential Reading)

1. **Freund & Schapire (1997) — AdaBoost**
   "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting."
   *Journal of Computer and System Sciences*, 55(1), 119–139.
   DOI: 10.1006/jcss.1997.1504
   URL: https://dl.acm.org/doi/10.1006/jcss.1997.1504

2. **Friedman (2001) — The Founding GBM Paper** ★★★
   "Greedy Function Approximation: A Gradient Boosting Machine."
   *The Annals of Statistics*, 29(5), 1189–1232.
   DOI: 10.1214/aos/1013203451
   Open Access: https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full

3. **Friedman (2002) — Stochastic Gradient Boosting**
   "Stochastic Gradient Boosting." *Computational Statistics & Data Analysis*, 38(4), 367–378.
   DOI: 10.1016/S0167-9473(01)00065-2
   URL: https://ideas.repec.org/a/eee/csdana/v38y2002i4p367-378.html

4. **Chen & Guestrin (2016) — XGBoost** ★★★
   "XGBoost: A Scalable Tree Boosting System." *KDD 2016*, 785–794.
   DOI: 10.1145/2939672.2939785 | arXiv: https://arxiv.org/abs/1603.02754

5. **Ke et al. (2017) — LightGBM** ★★★
   "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 2017*, 3149–3157.
   URL: https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf

6. **Prokhorenkova et al. (2018) — CatBoost**
   "CatBoost: Unbiased Boosting with Categorical Features." *NeurIPS 2018*, 6639–6649.
   arXiv: https://arxiv.org/abs/1706.09516

7. **Rashmi & Gilad-Bachrach (2015) — DART**
   "DART: Dropouts meet Multiple Additive Regression Trees." *AISTATS 2015*, 489–497.
   URL: https://proceedings.mlr.press/v38/korlakaivinayak15.pdf

8. **Duan et al. (2020) — NGBoost**
   "NGBoost: Natural Gradient Boosting for Probabilistic Prediction." *ICML 2020*.
   arXiv: https://arxiv.org/abs/1910.03225

9. **Lundberg et al. (2020) — TreeSHAP**
   "From Local Explanations to Global Understanding with Explainable AI for Trees."
   *Nature Machine Intelligence*, 2, 56–67.
   URL: https://www.nature.com/articles/s42256-019-0138-9

10. **GBM Benchmark (2023)** — Comprehensive comparison of GBM variants on 100+ datasets
    arXiv: https://arxiv.org/pdf/2305.17094

### Package Documentation (Verified URLs)

- **sklearn GradientBoostingClassifier 1.5:**
  https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
- **sklearn HistGradientBoostingClassifier 1.5:**
  https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- **XGBoost Parameters (stable):**
  https://xgboost.readthedocs.io/en/stable/parameter.html
- **XGBoost Introduction to Boosted Trees (official tutorial with math):**
  https://xgboost.readthedocs.io/en/stable/tutorials/model.html
- **LightGBM Parameters (stable):**
  https://lightgbm.readthedocs.io/en/stable/Parameters.html
- **LightGBM Parameters Tuning Guide:**
  https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
- **CatBoost Python API:**
  https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier
- **SHAP API Reference:**
  https://shap.readthedocs.io/en/latest/api.html

### Best Tutorials & Deep Dives

- **Explained.ai — Gradient Boosting** (Terence Parr & Jeremy Howard) ★★★
  The best intuitive + mathematical walkthrough of GBM for L2, L1, and general losses.
  Three-part series: https://explained.ai/gradient-boosting/

- **XGBoost Official Tutorial** — The XGBoost team's own explanation of the math.
  https://xgboost.readthedocs.io/en/stable/tutorials/model.html

- **Towards Data Science — Complete Guide to GBM** — Good practical survey.
  https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502

- **SHAP documentation and examples:**
  https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html

### Books (Specific Chapters)

- **"The Elements of Statistical Learning" (Hastie, Tibshirani, Friedman, 2009)**
  Chapter 10: Boosting and Additive Trees. The canonical textbook treatment of AdaBoost
  and MART by two of the three authors. Free PDF: https://hastie.su.domains/ElemStatLearn/

- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Géron, 3rd ed.)**
  Chapter 7: Ensemble Learning and Random Forests — covers gradient boosting, XGBoost.

- **"Introduction to Statistical Learning" (James et al., 2nd ed.)**
  Chapter 8: Tree-Based Methods — covers boosting with accessible mathematics.
  Free PDF: https://www.statlearning.com/

### Uncertainties and Version Flags

The following items were flagged in the research dossier and should be verified against
current documentation:

1. **XGBoost version:** As of mid-2026, XGBoost may be at 3.x. The 2.x API is backward-
   compatible. Verify `tree_method='hist', device='cuda'` is the correct GPU syntax for
   your version (`gpu_id` + `tree_method='gpu_hist'` was the pre-2.x API, now deprecated).

2. **`optuna.integration` namespace:** In Optuna 3.x, integrations were moved to a separate
   package `optuna-integration`. The import
   `from optuna.integration import XGBoostPruningCallback` may require
   `pip install optuna-integration`. Verify in current Optuna changelog.

3. **LightGBM `path_smooth`:** This native parameter is passed via `**kwargs` in the sklearn
   wrapper (`LGBMClassifier(path_smooth=1.0)`). It works but may trigger a warning in some
   LightGBM 4.x versions. Verify the exact behavior.

4. **SHAP + CatBoost symmetric trees:** May require
   `shap.TreeExplainer(model, data=background, model_output='raw')` for correct margin-level
   values. Test with your exact shap and catboost versions.

5. **sklearn `monotonic_cst` + `categorical_features` incompatibility (issue #28898):**
   Cannot be used for the same feature in sklearn 1.5.x. Verify in your version.

6. **NGBoost maintenance:** Limited activity post-2021. Verify that `pip install ngboost`
   installs cleanly and the `Bernoulli` distribution is available for your Python version.

7. **LightGBM `boosting_type='goss'`:** In LightGBM 4.x, verify whether GOSS is specified
   as `boosting_type='goss'` or as a separate data sampling parameter. The documentation
   organization changed between major versions.

