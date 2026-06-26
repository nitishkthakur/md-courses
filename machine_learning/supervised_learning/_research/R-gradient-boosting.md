# Research Dossier: Gradient Boosting Machines
**Target file:** `gradient-boosting.md`
**Prepared for:** Chapter author of the Supervised Learning Masterclass
**Date:** June 2026
**Research agent model:** claude-sonnet-4-6

---

## Scope

This dossier feeds `gradient-boosting.md` — the GBM chapter covering the complete lineage from AdaBoost (1996/1997) through Friedman's MART/GBM (2001/2002) through the modern industrial implementations: XGBoost 2.x, LightGBM 4.x, CatBoost 1.2.x, and NGBoost. It includes SHAP TreeExplainer, Optuna HPO integration, and practitioner pitfalls.

---

## Verified Packages

### Package Overview Table

| Package | Import | Version (mid-2026) | Algorithm variant | License | GPU |
|---|---|---|---|---|---|
| scikit-learn | `from sklearn.ensemble import GradientBoostingClassifier` | ~1.5.x | Friedman MART, exact sort | BSD-3 | No |
| scikit-learn (Hist) | `from sklearn.ensemble import HistGradientBoostingClassifier` | ~1.5.x | Histogram-based, LightGBM-inspired | BSD-3 | No |
| xgboost | `import xgboost as xgb` | ~2.x / 3.x | XGBoost (2nd-order Taylor + regularization) | Apache-2 | Yes (CUDA) |
| lightgbm | `import lightgbm as lgb` | ~4.x | Leaf-wise + GOSS + EFB | MIT | Yes |
| catboost | `from catboost import CatBoostClassifier` | ~1.2.x | Ordered boosting + symmetric trees | Apache-2 | Yes |
| ngboost | `from ngboost import NGBClassifier` | ~0.4.x | Natural Gradient + distributional output | Apache-2 | No |

### Canonical Import Block

```python
# sklearn (slow, exact)
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# LightGBM
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

# CatBoost
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# NGBoost
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Normal, LogNormal, Bernoulli, k_categorical
from ngboost.scores import LogScore, CRPScore

# Explainability
import shap

# HPO
import optuna
from optuna.integration import XGBoostPruningCallback  # verify current namespace
import optuna.integration.lightgbm as lgb_optuna  # LightGBMTuner
```

### Same `fit()` call across packages (classification, breast cancer)

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# sklearn (slow but pedagogically clear)
from sklearn.ensemble import GradientBoostingClassifier
m1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
m1.fit(X_train, y_train)

# sklearn Hist (fast, handles missing values and categoricals)
from sklearn.ensemble import HistGradientBoostingClassifier
m2 = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, random_state=42)
m2.fit(X_train, y_train)

# XGBoost
from xgboost import XGBClassifier
m3 = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                   tree_method='hist', device='cpu', random_state=42, eval_metric='logloss')
m3.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# LightGBM
from lightgbm import LGBMClassifier
m4 = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42)
m4.fit(X_train, y_train)

# CatBoost
from catboost import CatBoostClassifier
m5 = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6,
                        loss_function='Logloss', verbose=False, random_seed=42)
m5.fit(X_train, y_train)
```

---

## The Original Algorithm

### AdaBoost (the seed)

> **📜 Citation:** Freund, Y. and Schapire, R.E. (1997). "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting." *Journal of Computer and System Sciences*, 55(1), 119–139.
> **DOI:** 10.1006/jcss.1997.1504
> **URL:** https://dl.acm.org/doi/10.1006/jcss.1997.1504

**Problem solved:** The weak learnability hypothesis — can weak classifiers (barely better than random) be combined into a strong one?

**Key insight:** Reweight training examples at each round. Misclassified examples get higher weight so the next learner focuses on them. Final prediction is a weighted majority vote.

**AdaBoost.M1 algorithm (binary classification):**

```
Initialize: w_i = 1/n for all i = 1..n
For t = 1..T:
  1. Train weak learner h_t on weighted dataset (weights w_i)
  2. Compute weighted error: ε_t = Σ w_i · 1[h_t(x_i) ≠ y_i]
  3. Compute learner weight: α_t = (1/2) ln((1 - ε_t) / ε_t)
  4. Update weights: w_i ← w_i · exp(-α_t · y_i · h_t(x_i))
  5. Normalize: w_i ← w_i / Σ w_i
Final: H(x) = sign(Σ_t α_t · h_t(x))
```

**Key equations:**

$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

$$H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(\mathbf{x})\right)$$

AdaBoost minimizes the **exponential loss** $L(y, f) = \exp(-y \cdot f(\mathbf{x}))$, which Friedman later showed (2001) is the population minimizer for AdaBoost. This connection was not understood at the time of Freund & Schapire — it was a surprising retroactive discovery.

**Historical context:** Before AdaBoost (1996/1997), the predominant methods were decision trees (CART, 1984), neural networks, and SVMs. Ensemble methods existed (bagging, Breiman 1996) but combining *adaptive* weighted learners was new. AdaBoost won the Gödel Prize in 2003.

---

### Friedman's GBM — The Foundational Paper

> **📜 Citation:** Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *The Annals of Statistics*, 29(5), 1189–1232.
> **DOI:** 10.1214/aos/1013203451
> **URL (Open Access):** https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full

**Problem solved:** AdaBoost worked only for classification with exponential loss and had no framework for extending to regression or other loss functions. Friedman asked: what if we view boosting as *gradient descent in function space*?

**Core insight — functional gradient descent:** Instead of optimizing parameters in a fixed model, we optimize over the *space of functions*. At each iteration, the negative gradient of the loss with respect to the current ensemble prediction $F_{t-1}(\mathbf{x})$ (evaluated at the training points) gives the direction of steepest descent. We fit a weak learner (regression tree) to this gradient, then take a step in that direction.

**General GBM Framework:**

```
Initialize: F_0(x) = argmin_γ Σ L(y_i, γ)  [constant prediction]

For t = 1..T:
  1. Compute negative gradient (pseudo-residuals):
     r_{ti} = -[ ∂L(y_i, F(x_i)) / ∂F(x_i) ]_{F=F_{t-1}}

  2. Fit regression tree h_t to pseudo-residuals {(x_i, r_{ti})}
     (tree with J terminal regions R_{tj}, j=1..J)

  3. Line search for optimal step in each leaf region:
     γ_{tj} = argmin_γ Σ_{x_i ∈ R_{tj}} L(y_i, F_{t-1}(x_i) + γ)

  4. Update: F_t(x) = F_{t-1}(x) + ν · Σ_j γ_{tj} · 1[x ∈ R_{tj}]
     (ν = learning rate / shrinkage)

Return F_T(x)
```

**Loss functions and their negative gradients:**

| Loss | $L(y, F)$ | Negative gradient $r_i$ | Use case |
|---|---|---|---|
| Squared error (L2) | $(y - F)^2 / 2$ | $y_i - F(x_i)$ | Regression |
| Absolute error (L1) | $|y - F|$ | $\text{sign}(y_i - F(x_i))$ | Robust regression |
| Huber | (hybrid) | Huber-clipped residual | Outlier-robust regression |
| Log-loss (deviance) | $\log(1 + e^{-yF})$ | $y_i - \sigma(F(x_i))$ | Binary classification |
| Exponential | $e^{-yF}$ | $y_i \cdot e^{-y_i F(x_i)}$ | AdaBoost as special case |

**Key equation — prediction:**

$$\hat{F}(\mathbf{x}) = F_0 + \sum_{t=1}^{T} \nu \cdot h_t(\mathbf{x})$$

where $\nu \in (0, 1]$ is the learning rate (shrinkage), $h_t$ is the $t$-th regression tree, and $F_0$ is the initial constant prediction (often the mean for regression, log-odds for classification).

**What was uncertain at publication:** How to choose the optimal tree depth $J$. Friedman suggested $4 \leq J \leq 8$ for most problems based on empirical observation, not theory.

**Terminology note:** Friedman called this MART (Multiple Additive Regression Trees). The acronym GBM/GBDT/GBRT are all the same algorithm; confusion is endemic.

---

### Stochastic Gradient Boosting

> **📜 Citation:** Friedman, J.H. (2002). "Stochastic Gradient Boosting." *Computational Statistics & Data Analysis*, 38(4), 367–378.
> **DOI:** 10.1016/S0167-9473(01)00065-2
> **URL:** https://ideas.repec.org/a/eee/csdana/v38y2002i4p367-378.html

**Key change:** At each iteration, instead of fitting the tree to all $n$ training examples, randomly subsample a fraction $\eta$ (typically 0.5) *without replacement*. This provides two benefits: (1) faster training (fewer samples per tree), (2) variance reduction analogous to bagging, often significantly better generalization.

$$F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \nu \cdot h_t(\mathbf{x}; \mathcal{S}_t)$$

where $\mathcal{S}_t$ is a random subsample of $\lfloor \eta n \rfloor$ examples drawn without replacement.

This paper introduced the `subsample` hyperparameter (called $\eta$ in the paper). Setting `subsample < 1.0` transforms vanilla GBM into stochastic GBM — almost always better in practice.

---

## Major Variants & Evolution

### Variant 1: sklearn `GradientBoostingClassifier/Regressor`

**Year:** Available since sklearn 0.10 (2012), actively maintained
**What it is:** Pure Python + NumPy/Cython implementation of Friedman 2001/2002 MART. Uses exact best-split search — scans all split points for all features. This is $O(n \cdot p \cdot d)$ per tree where $d$ = tree depth.
**Weakness:** Slow on $n > 50{,}000$. No native missing value handling. No categorical support. No GPU.
**When to use:** Teaching, small datasets, when you need warm_start for incremental training studies.

### Variant 2: sklearn `HistGradientBoostingClassifier/Regressor`

**Year:** Graduated to stable API in sklearn 0.23 (2020), inspired by LightGBM
**Paper inspiration:** Ke et al. 2017 (LightGBM)
**Key change:** Bin continuous features into `max_bins` (default 255) histogram bins before training. Finding the best split becomes scanning 255 candidate thresholds instead of $n$ unique values. This reduces per-tree complexity from $O(n \cdot p)$ to $O(B \cdot p)$ where $B \leq 255$.

**Additional features over regular GB:**
- Native missing value support (a dedicated bin)
- Native categorical feature support (`categorical_features` parameter)
- Monotonic constraints (`monotonic_cst`)
- Interaction constraints (`interaction_cst`)
- Parallelism via OpenMP

**Speed benchmark (sklearn docs):** GradientBoostingRegressor (200 trees) = 7.2 seconds; HistGradientBoostingRegressor (200 trees) = 0.5 seconds on same data.

### Variant 3: XGBoost

> **📜 Citation:** Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, San Francisco, 785–794.
> **DOI:** 10.1145/2939672.2939785
> **URL:** https://dl.acm.org/doi/10.1145/2939672.2939785
> **arXiv:** https://arxiv.org/abs/1603.02754

**What it changed over Friedman 2001:**

1. **Second-order Taylor expansion:** Friedman used only first-order gradients. XGBoost approximates the loss function with both the gradient $g_i = \partial L / \partial \hat{y}_i$ and the Hessian $h_i = \partial^2 L / \partial \hat{y}_i^2$, giving a more accurate local quadratic approximation.

2. **Regularized objective:** Added explicit L1 ($\alpha$) and L2 ($\lambda$) regularization on leaf weights, plus a penalty $\gamma$ on the number of leaves:

$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

3. **Closed-form optimal leaf weight:**

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

where $G_j = \sum_{i \in R_j} g_i$, $H_j = \sum_{i \in R_j} h_i$.

4. **Split gain formula:**

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

Splits are only accepted when Gain > 0, making $\gamma$ a natural pruning threshold.

5. **Column subsampling** (`colsample_bytree`, `colsample_bylevel`, `colsample_bynode`)
6. **Built-in cross-validation, early stopping, parallel split finding**
7. **DART booster** (Rashmi & Gilad-Bachrach, AISTATS 2015): applies dropout to trees, preventing late trees from over-specializing. Set `booster='dart'`.

**Why XGBoost won Kaggle (2014–2016):** The regularized objective + second-order gradients + column subsampling combination gave substantially better generalization than sklearn's GBM, especially on tabular data.

### Variant 4: LightGBM

> **📜 Citation:** Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 3149–3157.
> **URL:** https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
> **PDF:** https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf

**Two key algorithmic innovations:**

**1. GOSS (Gradient-based One-Side Sampling):** Instead of uniform subsampling, keep all large-gradient examples and randomly subsample small-gradient examples. Rationale: examples with large gradients contribute most to the information gain calculation. Large-gradient samples capture the hard-to-learn patterns; small-gradient samples are already well-approximated. This gives XGBoost-class accuracy with far less data processed per tree.

**2. EFB (Exclusive Feature Bundling):** In high-dimensional sparse data (e.g., one-hot encoded features), many features are mutually exclusive (they never both take nonzero values simultaneously). Bundle such features into a single feature, reducing feature count from $p$ to $O(p / k)$ where $k$ is the exclusivity ratio. Finding optimal bundles is NP-hard, but a greedy graph-coloring algorithm works well.

**3. Leaf-wise (best-first) growth:** XGBoost and sklearn GBM grow trees level-by-level (depth-wise). LightGBM grows the single leaf with the highest gain at each step, anywhere in the tree. This produces unbalanced but more efficient trees. The tradeoff: leaf-wise growth is faster to reach a given loss level, but prone to overfitting on small datasets. Control with `num_leaves` (not `max_depth`).

**Speed claim:** Up to 20x faster than XGBoost on large datasets, with similar or better accuracy.

### Variant 5: CatBoost

> **📜 Citation:** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., and Gulin, A. (2018). "CatBoost: Unbiased Boosting with Categorical Features." *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, 6639–6649.
> **arXiv:** https://arxiv.org/abs/1706.09516
> **NeurIPS proceedings:** https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

**Problem with all prior GBM implementations:** When computing gradients for example $i$, the model has been trained on data including $i$. This causes **prediction shift** — the gradient estimates are biased because the model has partially memorized the targets. The bias is $O(1/n)$ per example.

**Two key innovations:**

**1. Ordered Boosting:** Instead of computing all gradients at step $t$ using the full model $F_{t-1}$ trained on all data, use a per-example model trained only on data points that appeared *before* example $i$ in a random permutation. This mimics inference (where target is unknown) and eliminates prediction shift bias. An ordered permutation is chosen once and reused.

**2. Ordered Target Statistics (Ordered TS) for categoricals:** For categorical feature $c$ at example $i$ (position $\sigma_i$ in permutation), the encoding is:

$$\hat{x}_i^c = \frac{\sum_{j: \sigma_j < \sigma_i} [x_j^c = x_i^c] \cdot y_j + \text{prior} \cdot P}{\sum_{j: \sigma_j < \sigma_i} [x_j^c = x_i^c] + P}$$

where $P$ is a prior weight (prevents unstable estimates for rare categories). This is ordered target encoding with no target leakage.

**3. Symmetric (oblivious) trees:** CatBoost uses oblivious decision trees as base learners by default. An oblivious tree applies the same split criterion at all nodes at a given depth level, producing a lookup table of $2^d$ leaf values. This makes inference extremely fast (bit masking operations), training more regularized, and reduces overfitting. However, it can be suboptimal for some datasets — `grow_policy='Depthwise'` or `'Lossguide'` are alternatives.

**4. Native categorical handling:** Pass `cat_features=[list of column names or indices]` to the constructor or `fit()`. No manual encoding needed. Works out of the box on string or integer categorical columns.

### Variant 6: NGBoost

> **📜 Citation:** Duan, T., Avati, A., Ding, D.Y., Thai, K.K., Basu, S., Ng, A.Y., and Schuler, A. (2020). "NGBoost: Natural Gradient Boosting for Probabilistic Prediction." *Proceedings of the 37th ICML*.
> **arXiv:** https://arxiv.org/abs/1910.03225
> **Code:** https://github.com/stanfordmlgroup/ngboost

**What problem it solves:** All prior GBM variants output *point predictions* ($\hat{y}$). In medicine, finance, and weather forecasting, you need a full *probability distribution* $P(y | \mathbf{x})$. NGBoost outputs distributional parameters directly.

**Key idea:** For a distribution with parameters $\theta$ (e.g., Normal with $\mu$ and $\sigma^2$), treat $\theta$ as the prediction target. Boost over $\theta$ using the natural gradient of the proper scoring rule (log score or CRPS):

$$\tilde{\nabla}_\theta S(\theta; y) = I(\theta)^{-1} \nabla_\theta S(\theta; y)$$

where $I(\theta)$ is the Fisher information matrix. The natural gradient corrects for the geometry of the parameter space, making updates scale-invariant.

**Available distributions:**
- `Normal` — Gaussian regression (mean + variance)
- `LogNormal` — positive-valued outcomes
- `Exponential` — survival / time-to-event
- `k_categorical(k)` — classification over $k$ classes

**Scoring rules:** `LogScore` (negative log-likelihood), `CRPScore` (Continuous Ranked Probability Score for regression)

**API:**
```python
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

ngb = NGBRegressor(Dist=Normal, Score=LogScore, n_estimators=500, learning_rate=0.01)
ngb.fit(X_train, y_train)
y_dist = ngb.pred_dist(X_test)   # returns distribution object
y_dist.mean   # point predictions
y_dist.interval(0.95)  # 95% prediction interval
```

**Limitation:** Significantly slower than XGBoost/LightGBM. No GPU. Base learner is by default a shallow decision tree (depth 3).

### Variant 7: DART (Dropouts meet MART)

> **📜 Citation:** Rashmi, K.V. and Gilad-Bachrach, R. (2015). "DART: Dropouts meet Multiple Additive Regression Trees." *Proceedings of the 18th AISTATS*, 489–497.
> **URL:** https://proceedings.mlr.press/v38/korlakaivinayak15.pdf

**Problem:** In standard GBM, later trees focus on refining small corrections, contributing tiny increments. This leads to "over-specialization" and can hurt generalization. Early trees dominate.

**Solution:** At each iteration, randomly drop a subset of existing trees (analogous to neural network dropout). Fit the new tree on the residuals from the remaining trees. After fitting, rescale contributions to maintain ensemble scale.

**Available in:** XGBoost (`booster='dart'`, parameters: `rate_drop`, `skip_drop`, `normalize_type`), LightGBM (`boosting='dart'`).

**Caution:** Early stopping is unreliable with DART because removing trees changes predictions unpredictably. Do not use `early_stopping_rounds` with DART.

---

## Hyperparameter Reference

### sklearn `GradientBoostingClassifier` / `GradientBoostingRegressor`

**Verified against sklearn 1.5.2:** https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

| Parameter | Default | Type | Effect | Tuning strategy |
|---|---|---|---|---|
| `loss` | `'log_loss'` (clf) / `'squared_error'` (reg) | str | Loss function being minimized | `'log_loss'` for classification; `'exponential'` = AdaBoost loss; `'huber'` / `'quantile'` for robust regression |
| `n_estimators` | `100` | int | Number of boosting rounds (trees) | Tune jointly with `learning_rate`. Range: 100–5000. Use `n_iter_no_change` for early stopping. |
| `learning_rate` | `0.1` | float | Shrinkage multiplier per tree | Lower = better generalization, need more trees. Rule of thumb: halve `learning_rate`, double `n_estimators`. Range: 0.01–0.3. |
| `max_depth` | `3` | int or None | Max depth of individual trees | 3–6 is typical. Shallow trees = weak learners (correct). Deep trees violate GBM design. |
| `subsample` | `1.0` | float | Fraction of samples per tree (Stochastic GB) | Values 0.5–0.8 often improve both speed and accuracy. Enables variance reduction. |
| `min_samples_split` | `2` | int or float | Min samples to split an internal node | Increase to prevent overfitting. Range: 2–20 or 0.01–0.1 (fraction). |
| `min_samples_leaf` | `1` | int or float | Min samples per leaf | Increase to prevent overfitting. Range: 1–20. More impactful than `min_samples_split`. |
| `min_weight_fraction_leaf` | `0.0` | float | Min weighted fraction at leaf | Alternative to `min_samples_leaf` when sample weights are used. |
| `max_features` | `None` (all) | str / int / float | Features considered per split | `'sqrt'` or `'log2'` adds randomization (like Random Forest). Often helps. |
| `max_leaf_nodes` | `None` | int | Max leaves (enables best-first growth) | If set, tree grows in best-first mode, overriding `max_depth`. |
| `min_impurity_decrease` | `0.0` | float | Min impurity reduction for split | Rarely tuned. Use regularization via tree structure instead. |
| `init` | `None` | estimator / `'zero'` | Initial prediction | `'zero'` = start from 0, not mean. Useful for custom initialization. |
| `criterion` | `'friedman_mse'` | str | Split quality measure | Keep default `'friedman_mse'` (Friedman's improved MSE with intra-node variance). |
| `warm_start` | `False` | bool | Reuse previous fit and add more trees | Useful for studying learning curves without refitting. |
| `validation_fraction` | `0.1` | float | Holdout for early stopping | Only used when `n_iter_no_change` is set. |
| `n_iter_no_change` | `None` | int | Early stopping patience | Set to 10–20. Requires `validation_fraction`. |
| `tol` | `1e-4` | float | Early stopping tolerance | Rarely needs changing. |
| `ccp_alpha` | `0.0` | float | Cost-complexity pruning parameter | Post-pruning. Small positive values (0.001–0.01) can help. |
| `random_state` | `None` | int | Seed for reproducibility | Always set for experiments. |

### sklearn `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor`

**Verified against sklearn 1.5.2:** https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

| Parameter | Default | Effect | Notes |
|---|---|---|---|
| `max_iter` | `100` | Number of boosting iterations | Equivalent to `n_estimators`. |
| `learning_rate` | `0.1` | Shrinkage | Same as GradientBoosting. |
| `max_leaf_nodes` | `31` | Max leaves per tree | Primary complexity control. XGBoost default analogue: `max_leaves=31`. |
| `max_depth` | `None` | Max depth | Less important than `max_leaf_nodes`. |
| `min_samples_leaf` | `20` | Min samples per leaf | Default 20 is safe but high. Try 10–50. |
| `l2_regularization` | `0.0` | L2 regularization on leaf values | Equivalent to XGBoost's `reg_lambda`. Try 0.1–1.0. |
| `max_features` | `1.0` | Fraction of features per node | Values < 1.0 add randomization. |
| `max_bins` | `255` | Number of histogram bins | Maximum is 255. Rarely needs changing. |
| `categorical_features` | `None` | Categorical feature specification | Use `"from_dtype"` to detect from pandas dtype automatically. |
| `monotonic_cst` | `None` | Monotonic constraints per feature | Dict `{feature_name: +1/-1/0}` or array. **Cannot combine with categorical features** (GitHub issue #28898). |
| `interaction_cst` | `None` | Feature interaction constraints | `"pairwise"`, `"no_interactions"`, or list of sets. |
| `early_stopping` | `'auto'` | Automatic early stopping | `'auto'` activates if n_samples >= 10,000. |
| `scoring` | `'loss'` | Early stopping metric | Can pass a sklearn scorer string. |
| `validation_fraction` | `0.1` | Holdout fraction for early stopping | Or absolute sample count if >= 1. |
| `n_iter_no_change` | `10` | Early stopping patience | |
| `tol` | `1e-7` | Early stopping tolerance | |
| `warm_start` | `False` | Add iterations incrementally | |
| `class_weight` | `None` | Class weighting | `'balanced'` for imbalanced datasets. |

> **⚠️ Pitfall:** `monotonic_cst` and `categorical_features` cannot be used simultaneously in sklearn 1.5.x — this raises an error (GitHub issue #28898). This limitation may be lifted in later versions; verify in current docs.

### XGBoost `XGBClassifier` / `XGBRegressor`

**Verified against XGBoost docs (parameter.html):** https://xgboost.readthedocs.io/en/stable/parameter.html

Note: XGBoost is at 3.x as of mid-2026. The 2.x API is backward-compatible. The sklearn wrapper adds `n_estimators`; the native API uses `num_boost_round`.

| Parameter | Default (XGB native) | Notes |
|---|---|---|
| `n_estimators` | `100` (sklearn wrapper) | `num_boost_round` in native API |
| `learning_rate` / `eta` | `0.3` | Note: XGBoost default is 0.3, not 0.1! Much higher than sklearn. |
| `max_depth` | `6` | Deeper default than sklearn (3). |
| `min_child_weight` | `1` | Minimum sum of Hessian in a leaf. Analogous to `min_samples_leaf` but weighted. Increase to regularize. |
| `gamma` / `min_split_loss` | `0` | Minimum gain for a split (the $\gamma$ in split gain formula). Values 0–5. |
| `subsample` | `1` | Row subsampling. Set 0.6–0.8 typically. |
| `colsample_bytree` | `1` | Column subsample per tree. Set 0.6–0.8. |
| `colsample_bylevel` | `1` | Column subsample per tree level. |
| `colsample_bynode` | `1` | Column subsample per split node. Most granular. |
| `reg_lambda` / `lambda` | `1` | L2 regularization on leaf weights. **Default is 1, not 0!** |
| `reg_alpha` / `alpha` | `0` | L1 regularization on leaf weights. |
| `tree_method` | `'auto'` | Use `'hist'` for all practical purposes (same as HistGB internally). `'exact'` is slow. |
| `device` | `'cpu'` | `'cuda'` for GPU. `'cuda:0'`, `'cuda:1'` for specific devices. |
| `booster` | `'gbtree'` | `'dart'` for dropout trees, `'gblinear'` for linear base learners. |
| `monotone_constraints` | `None` | Monotonic constraints per feature. Dict or tuple of +1/-1/0. |
| `interaction_constraints` | `None` | Nested list of feature groups allowed to interact. |
| `scale_pos_weight` | `1` | Ratio of negative to positive for imbalanced classification. |
| `n_jobs` | `1` | Parallel threads. Set to -1 for all CPUs. |
| `eval_metric` | auto | `'logloss'`, `'auc'`, `'rmse'` etc. |
| `early_stopping_rounds` | `None` | Stop if no improvement for N rounds. Requires `eval_set`. |

**Early stopping example:**
```python
model = XGBClassifier(n_estimators=1000, learning_rate=0.05, tree_method='hist',
                      device='cpu', early_stopping_rounds=50, eval_metric='logloss')
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
print(f"Best iteration: {model.best_iteration}")
```

> **⚠️ Pitfall:** XGBoost's default `learning_rate=0.3` is much higher than sklearn's 0.1. When benchmarking, use the same `learning_rate` across packages. Also, `reg_lambda=1` by default (not 0), so XGBoost is implicitly regularized even with no explicit regularization settings.

### LightGBM `LGBMClassifier` / `LGBMRegressor`

**Verified against LightGBM 4.x docs:** https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html and https://lightgbm.readthedocs.io/en/stable/Parameters.html

The sklearn wrapper uses `colsample_bytree`, `subsample`, `subsample_freq` as aliases. The native parameter names are different — both work.

| Parameter | Default | Native alias | Notes |
|---|---|---|---|
| `n_estimators` / `num_iterations` | `100` | `num_boost_round` | |
| `learning_rate` | `0.1` | `shrinkage_rate` | |
| `num_leaves` | `31` | — | **The primary complexity parameter for leaf-wise growth.** Keep < $2^{max\_depth}$ to avoid overfitting. |
| `max_depth` | `-1` (unlimited) | — | Sets a ceiling on leaf-wise growth. |
| `min_child_samples` / `min_data_in_leaf` | `20` | — | Min data points per leaf. Increase for regularization. |
| `min_child_weight` / `min_sum_hessian_in_leaf` | `0.001` | — | Min Hessian sum per leaf. |
| `subsample` / `bagging_fraction` | `1.0` | — | Row subsampling. |
| `subsample_freq` / `bagging_freq` | `0` | — | Apply bagging every k iterations. Must set > 0 to enable `bagging_fraction`. |
| `colsample_bytree` / `feature_fraction` | `1.0` | — | Column subsampling per tree. |
| `reg_alpha` / `lambda_l1` | `0.0` | — | L1 regularization. |
| `reg_lambda` / `lambda_l2` | `0.0` | — | L2 regularization. |
| `min_split_gain` / `min_gain_to_split` | `0.0` | — | Min gain for split. |
| `path_smooth` | `0` | — | Smoothing on path from root to leaf. Prevents overfitting on leaves with few samples. Non-zero values (e.g., 1.0) smooth predictions along the tree path. |
| `boosting_type` | `'gbdt'` | `boosting` | `'dart'` for dropout, `'goss'` for GOSS (now default behavior in LightGBM 4.x uses GOSS internally). |
| `objective` | auto | — | `'binary'`, `'multiclass'`, `'regression'`, etc. |
| `is_unbalance` | `False` | — | Alternative to `class_weight='balanced'` for imbalanced data. |
| `class_weight` | `None` | — | sklearn-style class weights. |
| `importance_type` | `'split'` | — | `'gain'` for gain-based importance. |

> **⚠️ Pitfall (LightGBM subsample):** Setting `subsample < 1.0` alone does nothing. You **must also set** `subsample_freq` (or `bagging_freq`) to a positive integer (e.g., 1). This is a common silent misconfiguration — the model silently ignores `bagging_fraction` when `bagging_freq=0`.

> **⚠️ Pitfall (path_smooth):** The `path_smooth` parameter is a native LightGBM parameter passed via `**kwargs` in the sklearn wrapper. It is not listed in the sklearn-style constructor signature. Pass it as a kwarg: `LGBMClassifier(path_smooth=1.0)` — this works but may trigger warnings.

### CatBoost `CatBoostClassifier` / `CatBoostRegressor`

**Verified against CatBoost 1.2.x docs:** https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier

| Parameter | Default | Notes |
|---|---|---|
| `iterations` / `n_estimators` | `1000` | Note: much higher default than other libraries. |
| `learning_rate` / `eta` | auto (0.03 typically) | Auto-selected based on dataset size. |
| `depth` / `max_depth` | `6` | For symmetric trees, depth is exactly equal across all branches. |
| `l2_leaf_reg` / `reg_lambda` | `3.0` | L2 regularization. Default is higher than XGBoost/LightGBM. |
| `border_count` / `max_bin` | `254` | Number of histogram bins for numeric features. |
| `cat_features` | `None` | Column indices or names to treat as categorical. Key differentiator. |
| `loss_function` / `objective` | auto | `'Logloss'` (binary), `'MultiClass'`, `'RMSE'`, `'MAE'`, etc. |
| `boosting_type` | auto | `'Ordered'` (default for small datasets), `'Plain'` (standard gradient descent). |
| `grow_policy` | `'SymmetricTree'` | Also: `'Depthwise'`, `'Lossguide'` (leaf-wise). SymmetricTree is CatBoost's default and fastest for inference. |
| `od_type` | `None` | Overfitting detector type: `'IncToDec'` or `'Iter'`. |
| `od_wait` | `20` | Early stopping patience. |
| `random_seed` | `None` | Seed (uses `random_seed`, not `random_state`!). |
| `task_type` | `'CPU'` | `'GPU'` for GPU training. |
| `devices` | `None` | GPU device indices, e.g., `'0:1'`. |
| `verbose` | `1` | 0 to suppress output. |

**Cat features example:**
```python
from catboost import CatBoostClassifier, Pool

cat_features = ['country', 'product_type', 'user_segment']
train_pool = Pool(X_train, y_train, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    loss_function='Logloss',
    eval_metric='AUC',
    od_type='Iter',
    od_wait=50,
    verbose=False,
    random_seed=42
)
model.fit(train_pool, eval_set=Pool(X_val, y_val, cat_features=cat_features))
```

> **⚠️ Pitfall (CatBoost random_seed):** CatBoost uses `random_seed`, not `random_state`. This is different from all sklearn-compatible estimators. Easy bug when wrapping in pipelines.

---

## Common Errors & Pitfalls

### 1. Learning Rate / n_estimators Coupling

The most common mistake: tuning `learning_rate` and `n_estimators` independently. They are tightly coupled — a lower learning rate requires proportionally more trees. The correct workflow:

```
1. Fix a small learning_rate (0.05 or 0.01)
2. Find optimal n_estimators via early stopping
3. Then optionally tune tree structure (depth, num_leaves, etc.)
4. Optionally further reduce learning_rate and increase n_estimators proportionally
```

### 2. XGBoost Default learning_rate=0.3

XGBoost's `eta` defaults to 0.3, not 0.1. When users benchmark XGBoost against sklearn GBM "out of the box," they're comparing lr=0.3/depth=6 against lr=0.1/depth=3 — entirely different regime. Always align defaults when comparing.

### 3. LightGBM bagging silently disabled

`LGBMClassifier(subsample=0.8)` does nothing by default because `subsample_freq=0` disables bagging. Must set:
```python
LGBMClassifier(subsample=0.8, subsample_freq=1)  # bagging every iteration
```

### 4. sklearn monotonic_cst + categorical_features incompatibility

In sklearn 1.5.x, `HistGradientBoostingClassifier` raises an error if you specify both `monotonic_cst` and `categorical_features` for the same features. Verify in your sklearn version.

### 5. DART + early stopping

`booster='dart'` is incompatible with `early_stopping_rounds`. The DART mechanism randomly drops trees, making validation loss non-monotone. XGBoost silently ignores early stopping with DART — you get no error, just wrong behavior.

### 6. Class imbalance: scale_pos_weight vs is_unbalance

- XGBoost: Use `scale_pos_weight = n_negative / n_positive`
- LightGBM: Use `is_unbalance=True` OR `class_weight='balanced'` (not both)
- CatBoost: Use `class_weights={0: 1, 1: w}` or `auto_class_weights='Balanced'`

Caution: `is_unbalance=True` and `scale_pos_weight` both produce poorly-calibrated probability estimates. If you need well-calibrated probabilities, use `class_weight` variants and apply Platt scaling or isotonic regression post-hoc.

### 7. Overfit detection: feature importance inflation

Split-based feature importance (default in LightGBM and XGBoost) is biased toward high-cardinality features. A continuous feature with 1000 unique values appears important even if weakly predictive, because there are more split points to choose from. **Always prefer SHAP or gain-weighted importance over count-based split importance.**

### 8. Collinearity and SHAP disagreement

When two features are highly correlated, SHAP splits credit between them. Permutation importance shows low importance for both (since swapping one still leaves the other as backup). These are measuring different things — SHAP measures prediction attribution, permutation importance measures prediction degradation. Both are valid. Conflating them causes misleading "feature ranking" stories.

### 9. CatBoost with ordered boosting on small datasets

Ordered boosting requires enough data to estimate ordered target statistics reliably. On small datasets (n < 1000), it can actually hurt performance. Test `boosting_type='Plain'` as a baseline.

### 10. max_depth vs num_leaves in LightGBM

Setting `max_depth=6` in LightGBM doesn't constrain the tree the same way as in XGBoost. LightGBM is leaf-wise: depth 6 with `num_leaves=200` creates a very deep, unbalanced tree. The primary control is `num_leaves`. A `num_leaves` larger than $2^{max\_depth}$ effectively ignores the depth constraint.

### 11. Prediction shift in target encoding without ordered TS

If you manually target-encode categorical features *before* training (using the full training set), you introduce target leakage — the gradient at each point includes information from that point's own label. This is the problem CatBoost's ordered TS solves. The symptom: training metrics look great, but val/test metrics disappoint.

---

## Key Datasets for Examples

### 1. `breast_cancer` — Binary Classification Benchmark

```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
# 569 samples, 30 features (all numeric), binary labels (0=malignant, 1=benign)
# Good for: comparing algorithms, showing overfitting/underfitting behavior
```

### 2. `california_housing` — Regression Benchmark

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X, y = housing.data, housing.target
# 20,640 samples, 8 features, target = median house value (in $100k)
# Good for: regression, large enough to show HistGBM speed advantage
```

### 3. Higgs Boson Dataset — Large-Scale Classification

```python
from sklearn.datasets import fetch_openml
higgs = fetch_openml(name='higgs', version=1, as_frame=True, parser='auto')
X, y = higgs.data, higgs.target
# 11,000,000 samples (use subset), 28 features (physics observables)
# This is the canonical dataset for LightGBM/XGBoost speed benchmarks
# The HistGradientBoosting sklearn documentation uses this dataset
# Caution: downloading takes time, cache with data_home= parameter
```

### 4. `make_hastie_10_2` — Synthetic Binary Classification

```python
from sklearn.datasets import make_hastie_10_2
X, y = make_hastie_10_2(n_samples=12000, random_state=42)
# 10 features, binary, no noise, from ESL textbook
# Good for: demonstrating GBM algorithm, comparing train/test curves
```

---

## SHAP TreeExplainer

**Verified API (SHAP 0.46.x):** https://shap.readthedocs.io/en/latest/api.html

TreeExplainer implements the polynomial-time TreeSHAP algorithm (Lundberg et al. 2020) for exact Shapley value computation for tree ensembles. Supported models: XGBoost, LightGBM, CatBoost, sklearn GradientBoosting/HistGradientBoosting, RandomForest, ExtraTrees.

```python
import shap

# Option 1: Auto-detect model type (recommended)
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)  # Returns Explanation object

# Option 2: Explicit TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # Returns numpy array

# Plots
shap.summary_plot(shap_values, X_test, feature_names=feature_names)  # Beeswarm
shap.plots.waterfall(shap_values[0])  # Single prediction breakdown
shap.plots.force(shap_values[0])      # Force plot for one prediction
shap.plots.beeswarm(shap_values)      # Summary beeswarm plot
shap.dependence_plot('feature_name', shap_values.values, X_test)  # Dependence

# For multiclass (XGBoost/LightGBM): shap_values is a list of arrays, one per class
# Use shap_values[:, :, class_index] with new Explanation object API
```

**TreeSHAP computational complexity:** $O(T \cdot D \cdot 2^D)$ where $T$ = number of trees, $D$ = max tree depth. For typical GBM (100–1000 trees, depth 3–6), this is fast.

> **⚠️ Pitfall (SHAP interaction with CatBoost symmetric trees):** CatBoost's symmetric tree structure requires passing `model_output='raw'` in some versions of SHAP to get correct values. Verify with your exact versions.

---

## Optuna Integration for HPO

```python
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': 'hist',
        'eval_metric': 'logloss',
        'random_state': 42,
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=100, timeout=600)
print("Best params:", study.best_params)
print("Best AUC:", study.best_value)

# Optuna's built-in LightGBM Tuner (uses expert heuristics)
import optuna.integration.lightgbm as lgb_optuna
# verify: optuna.integration namespace may change with optuna version
```

**Optuna + XGBoost with pruning (early stopping integration):**
```python
from optuna.integration import XGBoostPruningCallback  # verify namespace in optuna 3.x

def objective_with_pruning(trial):
    pruning_callback = XGBoostPruningCallback(trial, 'validation-logloss')
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model.best_score
```

> **⚠️ Verify:** The `optuna.integration.XGBoostPruningCallback` namespace has moved between Optuna versions. In Optuna 3.x, check `optuna.integration.XGBoostPruningCallback` — if missing, look for `optuna_integration.xgboost`. The standalone `optuna-integration` package was separated from core `optuna` in version 3.x.

---

## Curated Resources

Each resource verified as real and accessible:

1. **Friedman 2001 (Open Access PDF)** — The founding paper. Read sections 3–5 for the full algorithm.
   https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full

2. **Friedman 2002 Stochastic GB** — The `subsample` paper. Explains why randomization helps.
   https://ideas.repec.org/a/eee/csdana/v38y2002i4p367-378.html

3. **Freund & Schapire 1997 (AdaBoost)** — ACM DL, DOI 10.1006/jcss.1997.1504
   https://dl.acm.org/doi/10.1006/jcss.1997.1504

4. **Chen & Guestrin 2016 (XGBoost)** — KDD paper. Section 2 has the full math.
   https://dl.acm.org/doi/10.1145/2939672.2939785 | arXiv: https://arxiv.org/abs/1603.02754

5. **Ke et al. 2017 (LightGBM)** — NeurIPS PDF. GOSS and EFB algorithms.
   https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf

6. **Prokhorenkova et al. 2018 (CatBoost)** — NeurIPS. Ordered boosting math.
   https://arxiv.org/abs/1706.09516

7. **Duan et al. 2020 (NGBoost)** — ICML. Natural gradient + probabilistic output.
   https://arxiv.org/abs/1910.03225

8. **sklearn GradientBoostingClassifier 1.5 docs** — Verified hyperparameter reference.
   https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

9. **sklearn HistGradientBoostingClassifier 1.5 docs** — Histogram variant.
   https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

10. **XGBoost Parameters (stable)** — All native parameter names and defaults.
    https://xgboost.readthedocs.io/en/stable/parameter.html

11. **LightGBM Parameters (stable)** — Native parameter reference.
    https://lightgbm.readthedocs.io/en/stable/Parameters.html

12. **LightGBM Parameters Tuning Guide** — Official tuning strategies.
    https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html

13. **CatBoost Python API** — CatBoostClassifier parameters.
    https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier

14. **Explained.ai Gradient Boosting** — Terence Parr & Jeremy Howard. Best visual/intuitive explanation of the math. Three-part series on L2, L1, and general loss.
    https://explained.ai/gradient-boosting/

15. **XGBoost Introduction to Boosted Trees** — Official tutorial with the full math.
    https://xgboost.readthedocs.io/en/stable/tutorials/model.html

16. **SHAP TreeExplainer API**
    https://shap.readthedocs.io/en/latest/api.html

17. **Rashmi & Gilad-Bachrach 2015 (DART)** — AISTATS proceedings PDF.
    https://proceedings.mlr.press/v38/korlakaivinayak15.pdf

18. **Benchmarking GBM Algorithms (2023)** — arXiv benchmark paper.
    https://arxiv.org/pdf/2305.17094

---

## Uncertainties

The chapter author must double-check these:

1. **XGBoost version:** As of mid-2026, XGBoost may be at 3.x, not 2.x. The sklearn API is largely backward-compatible but verify `device` parameter behavior (it was added in 2.0; in older versions, `gpu_id` and `tree_method='gpu_hist'` were used — now deprecated). Confirm: `tree_method='hist', device='cuda'` is the correct 2.x+ syntax.

2. **`optuna.integration` namespace:** In Optuna 3.x, integrations were moved to a separate package `optuna-integration`. The import path `from optuna.integration import XGBoostPruningCallback` may require `pip install optuna-integration`. Verify current state.

3. **`path_smooth` in LightGBM sklearn wrapper:** This native parameter is passed via `**kwargs` in `LGBMClassifier`. It works but may trigger a warning about unknown parameters in some LightGBM 4.x versions. Verify the exact behavior.

4. **SHAP + CatBoost:** The exact SHAP API for CatBoost symmetric trees may require `shap.TreeExplainer(model, data=background_data, model_output='raw')` to get proper margin-level SHAP values vs probability-level. Test with a small example and verify `shap_values` shapes.

5. **NGBoost maintenance:** The `stanfordmlgroup/ngboost` repo has had limited activity post-2021. Verify the package still installs cleanly and the `Bernoulli` distribution for classification is available (some versions only have `k_categorical`). Check pypi.org/project/ngboost for current version.

6. **HistGradientBoosting monotonic + categorical constraint:** GitHub issue #28898 documented an error when combining these. Verify whether this is fixed in sklearn 1.5.x or 1.6.x.

7. **LightGBM GOSS vs explicit `boosting_type='goss'`:** In LightGBM 4.x, GOSS is a data sampling strategy, not a separate boosting type. The `boosting_type` can be set to `'goss'` but verify whether in 4.x it's now a separate parameter or still under `boosting_type`. The docs may have reorganized this.

8. **CatBoost `auto_class_weights` parameter:** Verify exact spelling and behavior in CatBoost 1.2.x. Some sources reference `class_weights` as a dict; others reference `auto_class_weights='Balanced'`. Both may exist; confirm which takes precedence.

9. **Higgs dataset size:** `fetch_openml(name='higgs')` downloads 11M rows by default. For examples, subsample to 100k–500k. The chapter should note this prominently to avoid making readers download 8+ GB.

10. **NGBoost ICML 2020 publication venue:** The arXiv paper says "Accepted for ICML 2020" but also appears in NeurIPS 2019 (as a workshop paper). The primary citation should be ICML 2020 Proceedings. Verify the exact conference proceedings citation.

---

## 3-Sentence Summary

Gradient Boosting Machines span a 25-year lineage from AdaBoost's 1997 weighted ensemble to today's industrial-scale histogram implementations, all united by a single insight — fit each new weak learner to the negative gradient of the loss in function space. The critical hierarchy for practitioners is: use sklearn's `HistGradientBoosting*` for quick baselines with missing values and categoricals; reach for XGBoost (`tree_method='hist', device='cuda'`) or LightGBM 4.x for large-scale tabular problems where speed and regularization depth matter; use CatBoost when your data has native categorical features at scale and you want to avoid all preprocessing. SHAP's `TreeExplainer` provides exact, polynomial-time Shapley values for all tree-based GBM variants and should always accompany GBM deployments; the most common practitioner pitfalls are the `learning_rate`/`n_estimators` coupling, LightGBM's silent subsample misconfiguration, and split-count feature importance bias.
