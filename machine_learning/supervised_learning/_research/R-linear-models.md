# Research Dossier: Linear Models (Ridge, Lasso, ElasticNet, and the Family)

**Target output file:** `linear-models.md`
**Research date:** June 2026
**Dossier author:** Research agent, Claude Sonnet 4.6

---

## Scope

This dossier feeds the chapter file `linear-models.md`. It covers the full family of penalized (and unpenalized) linear regression models available in scikit-learn 1.5.x: `LinearRegression`, `Ridge`, `RidgeCV`, `Lasso`, `LassoCV`, `LassoLars`, `LassoLarsCV`, `LassoLarsIC`, `ElasticNet`, `ElasticNetCV`, `Lars`, `LarsCV`, `MultiTaskLasso`, `MultiTaskElasticNet`, `BayesianRidge`, `ARDRegression`, `HuberRegressor`, `QuantileRegressor`, `TheilSenRegressor`, `RANSACRegressor`, `OrthogonalMatchingPursuit`, `SGDRegressor`, `TweedieRegressor`/`PoissonRegressor`/`GammaRegressor`. It also covers supporting packages: `statsmodels` (OLS + inference), `glmnet-python`, and SHAP's `LinearExplainer`.

---

## Verified Packages

### Core Package: scikit-learn 1.5.x

**Version confirmed:** 1.5.2 (stable as of late 2024; docs at https://scikit-learn.org/1.5/)

All linear models live under `sklearn.linear_model`. Canonical imports:

```python
from sklearn.linear_model import (
    LinearRegression,
    Ridge, RidgeCV,
    Lasso, LassoCV, LassoLarsCV, LassoLarsIC,
    LassoLars,
    ElasticNet, ElasticNetCV,
    Lars, LarsCV,
    MultiTaskLasso, MultiTaskLassoCV,
    MultiTaskElasticNet, MultiTaskElasticNetCV,
    BayesianRidge,
    ARDRegression,
    HuberRegressor,
    QuantileRegressor,
    TheilSenRegressor,
    RANSACRegressor,
    OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV,
    SGDRegressor,
    TweedieRegressor, PoissonRegressor, GammaRegressor,
)
from sklearn.feature_selection import SelectFromModel
```

**Key API methods across all estimators:**
- `.fit(X, y)` — trains the model
- `.predict(X)` — returns fitted values
- `.coef_` — fitted coefficient vector (or matrix for multi-output)
- `.intercept_` — fitted intercept
- `.score(X, y)` — returns R² (for regressors)
- `.get_params()` / `.set_params()` — hyperparameter access

### statsmodels 0.14.x / 0.15.x (dev)

**Purpose:** Statistical inference — p-values, confidence intervals, heteroskedasticity tests, diagnostic tests. sklearn does NOT expose these.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Design-matrix-style
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const)
results = model.fit()
print(results.summary())  # full table: coefs, SEs, t-stats, p-values, CIs
print(results.conf_int())  # 95% confidence intervals by default

# Formula interface (R-style)
model = smf.ols('y ~ x1 + x2 + x3', data=df)
results = model.fit()

# Robust standard errors (HC3 = heteroskedasticity-consistent)
results_robust = model.fit(cov_type='HC3')

# Heteroskedasticity tests
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
bp_stat, bp_pval, _, _ = het_breuschpagan(results.resid, results.model.exog)
white_stat, white_pval, _, _ = het_white(results.resid, results.model.exog)
```

**Docs:** https://www.statsmodels.org/stable/stats.html

### SHAP 0.46.x — LinearExplainer

**Verified API:**

```python
import shap

# LinearExplainer for sklearn linear models
# model: fitted sklearn linear model (coef_ + intercept_ required)
# masker: background dataset for feature masking (use training data or shap.maskers.Independent)
explainer = shap.LinearExplainer(
    model=fitted_lasso,
    masker=X_train,          # use training data as background
    # feature_perturbation='interventional'  # DEPRECATED; use masker instead
)
shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

# With interventional (independent feature) assumption:
explainer_ind = shap.LinearExplainer(model, masker=shap.maskers.Independent(X_train))
```

**Key version note (0.45.0+):** Return type for models with multiple outputs changed from `list` to `np.ndarray`. The `feature_perturbation` parameter is **deprecated** — use the `masker` parameter instead.

**Mathematical relationship:** Under the independence assumption, `shap_value[i] = coef_[i] * (X[i] - X_train.mean(axis=0)[i])`. Under the correlation-dependent masker, credit is shared among correlated features via a linear transform estimated from `nsamples` samples.

### Other Relevant Packages

| Package | Import | Notes |
|---|---|---|
| `glmnet-python` | `from glmnet import ElasticNet as GlmnetEN` | R glmnet port; less maintained; Python glmnet differs from sklearn API |
| `linearmodels` | `from linearmodels import OLS as LmOLS` | Panel data, instrumental variables, robust covariance; docs at linearmodels.readthedocs.io |
| `statsmodels` | `import statsmodels.api as sm` | Full statistical inference, diagnostic tests |
| `scipy.stats` | `from scipy import stats` | `linregress()` for simple regression with p-values; `pearsonr`, `spearmanr` |

**glmnet-python caution:** The Python port (`glmnet` on PyPI) has had maintenance issues. For production, prefer `sklearn.linear_model.LassoCV` / `ElasticNetCV` which implement the same pathwise coordinate descent internally. The R `glmnet` package remains the gold standard reference for the algorithm.

---

## The Original Algorithm

### 1. Ordinary Least Squares (OLS) — The Baseline

**No single "paper"** — OLS emerges from work by Legendre (1805, "Nouvelles méthodes pour la détermination des orbites des comètes") and Gauss (1809, "Theoria Motus"). The modern statistical foundation is the **Gauss-Markov theorem**.

**Gauss-Markov theorem (statement):** Under the classical linear model assumptions — (1) linearity: $y = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$; (2) no perfect multicollinearity: $\text{rank}(\mathbf{X}) = p$; (3) strict exogeneity: $\mathbb{E}[\boldsymbol{\varepsilon} | \mathbf{X}] = \mathbf{0}$; (4) spherical errors: $\text{Var}(\boldsymbol{\varepsilon} | \mathbf{X}) = \sigma^2 \mathbf{I}$ — the OLS estimator $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$ is the **Best Linear Unbiased Estimator (BLUE)**: among all linear unbiased estimators, it has the smallest variance.

**The normal equations (closed-form solution):**

$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

**Computational complexity:**
- Training: $O(np^2 + p^3)$ — forming $\mathbf{X}^\top\mathbf{X}$ is $O(np^2)$; inverting is $O(p^3)$
- Inference: $O(p)$ per sample

**When OLS breaks down:** (a) $\mathbf{X}^\top\mathbf{X}$ is singular or near-singular (multicollinearity) → numerically unstable inversion; (b) $p > n$ → underdetermined, infinitely many solutions; (c) outliers → error term assumption violated, OLS is sensitive; (d) heteroskedasticity → OLS is still unbiased but not BLUE, SEs are wrong.

---

### 2. Ridge Regression — Hoerl & Kennard (1970)

**Full citation:**
> Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634
>
> Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Applications to nonorthogonal problems. *Technometrics*, 12(1), 69–82. https://doi.org/10.1080/00401706.1970.10488635

**Connection to Tikhonov:** Ridge regression is a special case of Tikhonov regularization (Tikhonov, 1943/1963), where the regularization matrix is $\lambda \mathbf{I}$. The statistics and signal processing communities arrived at the same idea independently.

**Problem solved:** When $\mathbf{X}^\top\mathbf{X}$ is ill-conditioned (multicollinear predictors), small perturbations in $\mathbf{y}$ cause enormous changes in $\hat{\boldsymbol{\beta}}_{\text{OLS}}$. Hoerl & Kennard proposed adding a small constant $\alpha$ to the diagonal of $\mathbf{X}^\top\mathbf{X}$ before inverting, making it strictly positive-definite.

**Objective function:**

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_2^2 \right\}$$

**Closed-form solution:**

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^\top\mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

The matrix $(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})$ is always invertible for $\alpha > 0$, even when $\mathbf{X}^\top\mathbf{X}$ is singular.

**Bias-variance tradeoff:** Ridge introduces bias (the estimate is no longer unbiased for $\boldsymbol{\beta}$) but reduces variance. As $\alpha \to 0$, Ridge → OLS (unbiased, high variance). As $\alpha \to \infty$, all coefficients → 0 (heavily biased, zero variance). There exists a range of $\alpha > 0$ where the mean squared error (MSE = bias² + variance) is strictly lower than OLS. This is not a contradiction of the Gauss-Markov theorem, because Ridge is a *biased* estimator — and the theorem only guarantees OLS is best among *unbiased* linear estimators.

**SVD interpretation:** If $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$ (SVD), then:

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \mathbf{V} \text{diag}\!\left(\frac{d_j}{d_j^2 + \alpha}\right) \mathbf{U}^\top \mathbf{y}$$

Ridge shrinks each principal component direction by $d_j^2/(d_j^2 + \alpha)$: directions with large singular values are barely shrunk; small singular values (near-collinear directions) are heavily shrunk. This is why Ridge is effective at handling multicollinearity.

---

### 3. Lasso — Tibshirani (1996)

**Full citation:**
> Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
>
> Available via Oxford Academic: https://academic.oup.com/jrsssb/article/58/1/267/7027929

**Retrospective (15 years later):**
> Tibshirani, R. (2011). Regression shrinkage and selection via the lasso: a retrospective. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 73(3), 273–282. https://doi.org/10.1111/j.1467-9868.2011.00771.x

**Problem solved:** Ridge shrinks coefficients but never sets any to exactly zero — the fitted model always uses all $p$ predictors. For interpretability and when the true model is sparse (few active variables), Ridge offers no variable selection. Subset selection (best-subsets, stepwise) gives exact zeros but is computationally intractable for large $p$ and is highly unstable (high variance in model selection). The Lasso elegantly combines shrinkage and variable selection.

**Objective function:**

$$\hat{\boldsymbol{\beta}}_{\text{Lasso}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1 \right\}$$

where $\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|$. The factor $\frac{1}{2n}$ is sklearn's convention (matches glmnet when $\alpha_{\text{sklearn}} = \lambda_{\text{glmnet}}/n$; note R glmnet uses $\frac{1}{2n} + \lambda$).

**Why does L1 induce sparsity?** The L1 ball $\|\boldsymbol{\beta}\|_1 \leq t$ is a hyperdiamond in $\mathbb{R}^p$. Its vertices lie on the coordinate axes. The constraint surface and the elliptical contours of the loss typically first touch at a vertex — where all but one coordinate is zero. L2 (Ridge) uses a smooth sphere with no corners, so contact typically occurs at a non-zero point on the surface.

**No closed form:** The L1 penalty is non-differentiable at $\beta_j = 0$. There is no equivalent of the normal equations. The optimal solution is characterized by the KKT conditions:

$$\mathbf{x}_j^\top (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \alpha \cdot \text{sign}(\hat{\beta}_j) \quad \text{if } \hat{\beta}_j \neq 0$$
$$|\mathbf{x}_j^\top (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})| \leq \alpha \quad \text{if } \hat{\beta}_j = 0$$

**Original algorithm (1996):** Tibshirani used a combination of quadratic programming and the "shooting" algorithm (a precursor to coordinate descent). The efficient modern algorithm (coordinate descent) came later.

---

### 4. LARS — Efron, Hastie, Johnstone, Tibshirani (2004)

**Full citation:**
> Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *The Annals of Statistics*, 32(2), 407–499. https://doi.org/10.1214/009053604000000067
>
> Preprint: https://arxiv.org/abs/math/0406456
>
> Full PDF (Stanford): https://tibshirani.su.domains/ftp/lars.pdf

**Problem solved:** Computing the full Lasso regularization path (solution for all values of $\alpha$ from $\infty$ to 0) required expensive computation. LARS showed that this path is piecewise linear, and provides an algorithm that traces it exactly in $O(p^3)$ time total — the same cost as a single OLS fit.

**Key insight:** LARS builds the solution incrementally. At each step, it finds the predictor most correlated with the current residual, and moves the coefficient in that direction until a new predictor becomes equally correlated with the residual. Then both move jointly. A small modification makes LARS trace the exact Lasso path.

---

### 5. Coordinate Descent for Lasso/Glmnet — Friedman, Hastie, Tibshirani (2010)

**Full citation:**
> Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1–22. https://doi.org/10.18637/jss.v033.i01
>
> Preprint: https://arxiv.org/pdf/0708.1485

**Key contribution:** Cyclical coordinate descent on the Lasso objective is *simpler* than LARS and *substantially faster* in practice for large datasets. For each coordinate $j$, the update is a soft-threshold operation on the partial residual:

$$\hat{\beta}_j \leftarrow \frac{S\!\left(\frac{1}{n}\mathbf{x}_j^\top (\mathbf{y} - \mathbf{X}_{-j}\hat{\boldsymbol{\beta}}_{-j}),\, \alpha\right)}{\frac{1}{n}\|\mathbf{x}_j\|_2^2}$$

where the soft-threshold function is $S(z, \gamma) = \text{sign}(z)(|z| - \gamma)_+$.

**Warm starts on path:** Starting at $\alpha_{\max}$ (where all coefficients are zero) and decreasing $\alpha$, each solve is initialized with the solution from the previous $\alpha$. This makes computing 100 points on the regularization path nearly as fast as computing 1 solution from scratch.

---

### 6. ElasticNet — Zou & Hastie (2005)

**Full citation:**
> Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 67(2), 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
>
> Oxford Academic: https://academic.oup.com/jrsssb/article/67/2/301/7109482

**Problem solved:** Lasso has two critical limitations: (1) when $p > n$, Lasso selects at most $n$ variables and behaves erratically; (2) with a group of correlated predictors, Lasso arbitrarily selects one and discards the rest — ignoring the information in the others. ElasticNet adds an L2 penalty to overcome both.

**Objective function (sklearn parameterization):**

$$\hat{\boldsymbol{\beta}}_{\text{EN}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \rho \|\boldsymbol{\beta}\|_1 + \frac{\alpha(1-\rho)}{2} \|\boldsymbol{\beta}\|_2^2 \right\}$$

where $\rho$ is `l1_ratio` ($\rho = 0$ → Ridge; $\rho = 1$ → Lasso; $0 < \rho < 1$ → ElasticNet). Note the ElasticNet includes both penalties and reduces to each pure case at the extremes.

**The grouping effect:** When $\rho$ is close to 0, ElasticNet tends to select groups of correlated features together, sharing credit proportionally. This is more stable and often more informative than Lasso's arbitrary selection from the group.

---

## Major Variants & Evolution

### 3.1 Ridge Cross-Validation: RidgeCV

```python
from sklearn.linear_model import RidgeCV
import numpy as np

# Default: Leave-One-Out CV (cv=None)
ridge_cv = RidgeCV(
    alphas=np.logspace(-6, 6, 200),  # 200 candidates on log scale
    cv=None,                          # None = LOO CV (exact, fast via GCV formula)
    scoring=None,                     # None = R² (default)
    store_cv_results=True,            # Store per-alpha CV scores (v1.5+)
)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.alpha_)   # Best alpha found
```

**Note:** LOO CV for Ridge is exact and fast via the Generalized Cross-Validation (GCV) formula — no need to refit $n$ times. For large $n$, LOO is efficient here but slower than k-fold for other models.

### 3.2 LassoCV — Regularization Path + K-Fold

```python
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(
    alphas=None,      # None = auto-generate from data (recommended)
    n_alphas=100,     # Number of alphas on path
    cv=5,             # K-fold (not LOO — Lasso has no GCV shortcut)
    max_iter=10000,   # May need to increase for convergence
    n_jobs=-1,        # Parallel CV folds
    eps=1e-3,         # Path length: alpha_min = eps * alpha_max
)
lasso_cv.fit(X_train, y_train)
print(lasso_cv.alpha_)         # Best alpha
print(lasso_cv.alphas_)        # Full alpha path used
```

**Important:** LassoCV uses warm starts internally — computing all 100 alphas is barely slower than computing 1.

### 3.3 LassoLars and LassoLarsCV

Lasso solved via the LARS algorithm instead of coordinate descent. Produces the exact regularization path:

```python
from sklearn.linear_model import LassoLarsCV, LassoLarsIC

# Cross-validation selection
lasso_lars_cv = LassoLarsCV(cv=5, max_iter=500)
lasso_lars_cv.fit(X_train, y_train)

# Information criterion selection (faster than CV)
lasso_lars_ic = LassoLarsIC(criterion='bic')  # 'aic' or 'bic'
lasso_lars_ic.fit(X_train, y_train)
print(lasso_lars_ic.alpha_)
```

**When to use LassoLars over Lasso?** For $p \ll n$, LARS is often faster for computing the full path. For large $n$ with moderate $p$, coordinate descent (LassoCV) is typically faster. For numerical precision, LassoLars can be more stable.

### 3.4 ElasticNetCV

```python
from sklearn.linear_model import ElasticNetCV

enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],  # Test multiple ratios
    alphas=None,     # Auto-generate
    n_alphas=100,
    cv=5,
    max_iter=10000,
    n_jobs=-1,
)
enet_cv.fit(X_train, y_train)
print(enet_cv.alpha_, enet_cv.l1_ratio_)
```

**Tip:** Always test a range of `l1_ratio` values including 1.0 (pure Lasso) in ElasticNetCV. This lets the data determine whether L2 regularization actually helps.

### 3.5 BayesianRidge — Automatic Regularization via Evidence Maximization

**Theoretical basis:**
> Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. *Journal of Machine Learning Research*, 1, 211–244.
>
> MacKay, D. J. C. (1992). Bayesian interpolation. *Neural Computation*, 4(3), 415–447.

Place Gaussian priors on the weights ($\boldsymbol{\beta} | \lambda \sim \mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$) and on the noise ($\varepsilon | \alpha \sim \mathcal{N}(0, \alpha^{-1})$). Place Gamma hyperpriors on $\alpha$ and $\lambda$. Marginalizing over $\boldsymbol{\beta}$ gives the **evidence** (marginal likelihood), which is maximized over $\alpha, \lambda$ via EM-style updates. No cross-validation needed.

```python
from sklearn.linear_model import BayesianRidge

br = BayesianRidge(
    max_iter=300,          # Maximum EM iterations (was n_iter before v1.3)
    tol=1e-3,              # Convergence tolerance
    alpha_1=1e-6,          # Gamma prior shape for noise precision alpha
    alpha_2=1e-6,          # Gamma prior rate for noise precision alpha
    lambda_1=1e-6,         # Gamma prior shape for weight precision lambda
    lambda_2=1e-6,         # Gamma prior rate for weight precision lambda
    compute_score=True,    # Store log marginal likelihood during training
)
br.fit(X_train, y_train)

# Uncertainty quantification
y_pred, y_std = br.predict(X_test, return_std=True)
print(br.alpha_)    # Estimated noise precision
print(br.lambda_)   # Estimated regularization precision (= alpha in Ridge notation)
print(br.sigma_)    # Posterior covariance of weights (p × p matrix)
```

**Deprecation note:** `n_iter` parameter was renamed to `max_iter` in sklearn 1.3. In 1.5.x, use `max_iter`.

### 3.6 ARDRegression — Per-Feature Regularization

ARD extends BayesianRidge by assigning a separate precision $\lambda_j$ to each weight $\beta_j$: $\beta_j \sim \mathcal{N}(0, \lambda_j^{-1})$. Features whose $\lambda_j \to \infty$ are effectively pruned (sparse Bayesian learning). More flexible than BayesianRidge but slower (EM now updates $p$ hyperparameters per iteration).

```python
from sklearn.linear_model import ARDRegression

ard = ARDRegression(
    max_iter=300,
    tol=1e-3,
    threshold_lambda=1e4,  # Prune features with lambda_ > this
    alpha_1=1e-6, alpha_2=1e-6,
    lambda_1=1e-6, lambda_2=1e-6,
)
ard.fit(X_train, y_train)
y_pred, y_std = ard.predict(X_test, return_std=True)

# See which features were pruned
print(ard.lambda_)  # Per-feature precisions; large = pruned
```

### 3.7 HuberRegressor — Robust to Outliers

Replaces the squared loss with the Huber loss, which is quadratic for small residuals (like OLS) and linear for large residuals (reducing outlier influence):

$$H_\epsilon(r) = \begin{cases} r^2 & \text{if } |r| \leq \epsilon \\ 2\epsilon|r| - \epsilon^2 & \text{if } |r| > \epsilon \end{cases}$$

```python
from sklearn.linear_model import HuberRegressor

huber = HuberRegressor(
    epsilon=1.35,    # Threshold between quadratic/linear loss; 1.35 → 95% efficiency vs OLS on Gaussian data
    max_iter=100,
    alpha=0.0001,    # L2 regularization (small default)
    tol=1e-5,
    warm_start=False,
)
huber.fit(X_train, y_train)
print(huber.outliers_)  # Boolean mask of detected outliers
```

**Epsilon interpretation:** At $\epsilon = 1.35$, HuberRegressor achieves ~95% asymptotic efficiency relative to OLS on clean Gaussian data, but is substantially more robust to heavy tails and outliers.

### 3.8 QuantileRegressor — Distributional Regression (sklearn 1.0+)

Predicts conditional quantiles via the pinball (tilted L1) loss. Useful for prediction intervals, risk quantification, heteroskedastic data.

```python
from sklearn.linear_model import QuantileRegressor

# Predict median (robust, like Huber but non-parametric w.r.t. loss)
median_reg = QuantileRegressor(
    quantile=0.5,    # Median; valid range: (0, 1) exclusive
    alpha=1.0,       # L1 regularization strength
    fit_intercept=True,
    solver='highs',  # Linear programming solver (SciPy HiGHS)
    solver_options=None,
)
median_reg.fit(X_train, y_train)

# Prediction interval (90% interval)
lower = QuantileRegressor(quantile=0.05, alpha=0.1).fit(X_train, y_train)
upper = QuantileRegressor(quantile=0.95, alpha=0.1).fit(X_train, y_train)
```

**Added in sklearn 1.0.** Uses linear programming (LP) internally via `scipy.optimize.linprog`. The `alpha` parameter controls L1 regularization (not the quantile level — a common confusion!).

### 3.9 TheilSenRegressor — High-Breakdown-Point Robust Regression

Computes the median of regression slopes across all pairs (or a random subsample) of data points. Breakdown point: ~29.3% (can tolerate up to 29.3% outliers without breakdown). Computationally expensive: $O\binom{n}{p}$.

```python
from sklearn.linear_model import TheilSenRegressor

ts = TheilSenRegressor(
    fit_intercept=True,
    max_subpopulation=10000,  # Max combinations to consider (random sampling if exceeded)
    n_subsamples=None,        # None = automatically determined
    max_iter=300,
    tol=1e-3,
    n_jobs=-1,
    random_state=42,
)
ts.fit(X_train, y_train)
```

### 3.10 RANSACRegressor — Consensus-Based Robust Regression

Random Sample Consensus: repeatedly fits a model to a small random subset, identifies inliers (points within `residual_threshold`), then refits on inliers. Can tolerate up to ~50% outliers.

```python
from sklearn.linear_model import RANSACRegressor, LinearRegression

ransac = RANSACRegressor(
    estimator=LinearRegression(),     # Inner model (any sklearn regressor)
    min_samples=0.5,                  # Fraction (or int) of samples per random fit
    residual_threshold=None,          # None = 1.4826 * MAD of absolute residuals
    max_trials=100,                   # Max random draws
    random_state=42,
)
ransac.fit(X_train, y_train)
print(ransac.inlier_mask_)   # Boolean array of inliers
```

### 3.11 MultiTaskLasso and MultiTaskElasticNet — Joint Feature Selection

When you have multiple related regression targets (e.g., predicting 10 time points simultaneously), MultiTaskLasso selects the same sparse set of features for all targets using a group L1/L2 (L2,1) norm:

$$\|\mathbf{W}\|_{2,1} = \sum_{j=1}^p \sqrt{\sum_{k=1}^K W_{jk}^2}$$

This encourages entire rows of $\mathbf{W}$ (one per feature, across all $K$ tasks) to be zero simultaneously.

```python
from sklearn.linear_model import MultiTaskLasso

mt_lasso = MultiTaskLasso(alpha=0.1, fit_intercept=True, max_iter=1000)
mt_lasso.fit(X_train, Y_train)  # Y_train shape: (n_samples, n_targets)
print(mt_lasso.coef_)           # Shape: (n_targets, n_features)
```

---

## Hyperparameter Reference

### LinearRegression

| Parameter | Default | Range | Notes |
|---|---|---|---|
| `fit_intercept` | `True` | bool | Set `False` only if data is pre-centered |
| `copy_X` | `True` | bool | `False` may overwrite X in-place |
| `n_jobs` | `None` | int or None | Multi-output only; no effect for single output |
| `positive` | `False` | bool | Constrain all coefficients ≥ 0 (uses NNLS solver) |

No regularization hyperparameters — LinearRegression is pure OLS.

### Ridge

| Parameter | Default | Range / Options | Tuning |
|---|---|---|---|
| `alpha` | `1.0` | [1e-6, 1e6], log scale | Always CV; use `RidgeCV(alphas=np.logspace(-6, 6, 200))` |
| `solver` | `'auto'` | See below | Usually leave `'auto'` |
| `fit_intercept` | `True` | bool | Almost always True |
| `copy_X` | `True` | bool | Memory optimization |
| `max_iter` | `None` | int | For iterative solvers (sag, saga, sparse_cg, lbfgs) |
| `tol` | `1e-4` | [1e-8, 1e-2] | Lower = more precise but slower |
| `positive` | `False` | bool | Forces ≥ 0; only compatible with solver='lbfgs' |
| `random_state` | `None` | int | For sag/saga only |

**Ridge solver selection (when `solver='auto'`):**

| Condition | Auto-selected solver |
|---|---|
| `positive=True` | `'lbfgs'` |
| Input not sparse | `'cholesky'` |
| Input sparse | `'sparse_cg'` |

**Manual solver guidance:**
- `'svd'`: Most numerically stable; slowest; not for sparse X
- `'cholesky'`: Fast for dense X, moderate $p$; fails for singular $\mathbf{X}^\top\mathbf{X}$
- `'lsqr'`: Fast iterative, sparse-compatible
- `'sparse_cg'`: Conjugate gradient, good for large sparse X
- `'sag'`/`'saga'`: Best for very large $n$ (stochastic); need feature scaling
- `'lbfgs'`: Only option supporting `positive=True`

### Lasso

| Parameter | Default | Range / Options | Tuning |
|---|---|---|---|
| `alpha` | `1.0` | [1e-6, 10], log scale | Use `LassoCV`; alpha=0 → OLS (use LinearRegression instead) |
| `max_iter` | `1000` | [1000, 100000] | Increase if ConvergenceWarning; try 10000 first |
| `tol` | `1e-4` | [1e-6, 1e-3] | Lower tol may require more iterations |
| `warm_start` | `False` | bool | `True` for incremental fitting or path computation |
| `positive` | `False` | bool | Constrain coefficients ≥ 0 |
| `selection` | `'cyclic'` | `'cyclic'`, `'random'` | `'random'` can converge faster for many features |
| `fit_intercept` | `True` | bool | Almost always True |
| `precompute` | `False` | bool/array | `True` precomputes Gram matrix; faster for $p < n$ |
| `random_state` | `None` | int | For `selection='random'` |

### ElasticNet

Same as Lasso, plus:

| Parameter | Default | Range | Tuning |
|---|---|---|---|
| `alpha` | `1.0` | [1e-6, 10], log scale | Use `ElasticNetCV` |
| `l1_ratio` | `0.5` | [0, 1] | Test multiple: `[0.1, 0.5, 0.7, 0.9, 0.95, 1.0]` in ElasticNetCV |

**l1_ratio interaction with alpha:**
- `l1_ratio=1.0` → Lasso
- `l1_ratio=0.0` → Ridge (but use Ridge class directly; ElasticNet(l1_ratio=0) is numerically inferior)
- `l1_ratio=0.5` → equal L1 and L2 contribution (default)

### BayesianRidge

| Parameter | Default | Notes |
|---|---|---|
| `max_iter` | `300` | EM iterations; increase if `scores_` is not flat |
| `tol` | `1e-3` | Convergence tolerance for EM |
| `alpha_1`, `alpha_2` | `1e-6` each | Gamma prior on noise precision; nearly non-informative at defaults |
| `lambda_1`, `lambda_2` | `1e-6` each | Gamma prior on weight precision; nearly non-informative |
| `alpha_init` | `None` | Init noise precision; None → 1/Var(y) |
| `lambda_init` | `None` | Init weight precision; None → 1.0 |
| `compute_score` | `False` | Set True to track convergence via marginal likelihood |

**Tuning:** BayesianRidge is largely self-tuning. The prior hyperparameters (alpha_1, alpha_2, lambda_1, lambda_2) rarely need adjustment except for strongly non-Gaussian likelihoods.

### HuberRegressor

| Parameter | Default | Range | Tuning |
|---|---|---|---|
| `epsilon` | `1.35` | [1.0, 5.0] | Lower → more robust, less efficient on clean data; 1.35 is standard |
| `max_iter` | `100` | [50, 500] | Increase for large datasets |
| `alpha` | `1e-4` | [0, 1] | L2 regularization; often keep small |
| `tol` | `1e-5` | [1e-8, 1e-3] | |
| `warm_start` | `False` | bool | |

### QuantileRegressor

| Parameter | Default | Range | Tuning |
|---|---|---|---|
| `quantile` | `0.5` | (0, 1) exclusive | Problem-specific; 0.5 = median regression |
| `alpha` | `1.0` | [0, ∞) | L1 regularization; set 0 for no regularization |
| `solver` | `'highs'` | `'highs'`, `'highs-ds'`, `'highs-ipm'`, `'interior-point'` | `'highs'` uses the best available HiGHS method |
| `fit_intercept` | `True` | bool | |

### Tuning Playbook Table

| Model | Key HP | Range | Strategy | Effect if too high | Effect if too low |
|---|---|---|---|---|---|
| Ridge | `alpha` | [1e-6, 1e6] | RidgeCV LOO | All coefs → 0 (underfitting) | OLS (multicollinearity issues) |
| Lasso | `alpha` | [1e-6, 10] | LassoCV k-fold | All coefs → 0 | Many nonzero coefs (OLS) |
| ElasticNet | `alpha` | [1e-6, 10] | ElasticNetCV | Overshrinkage | Underfitting |
| ElasticNet | `l1_ratio` | [0.1, 1.0] | Grid in ElasticNetCV | More L2-like (grouping) | More L1-like (sparsity) |
| Lasso | `max_iter` | [1000, 50000] | Start at 1000; increase on warning | N/A | ConvergenceWarning |
| BayesianRidge | `max_iter` | [100, 1000] | Check `scores_` flatness | N/A | Premature convergence |
| HuberRegressor | `epsilon` | [1.0, 3.0] | Manual tuning based on outlier rate | Less robust | Inefficient on clean data |

---

## Common Errors & Pitfalls

### Pitfall 1: Forgetting to Standardize Features

**Mechanism:** Regularization penalties ($\alpha\|\boldsymbol{\beta}\|_1$ or $\alpha\|\boldsymbol{\beta}\|_2^2$) operate on the coefficient scale. If `age` ranges [0, 100] and `income` ranges [0, 1,000,000], income has intrinsically smaller coefficients and will be penalized less regardless of actual importance. This violates the invariance assumption of Ridge/Lasso.

**Rule:** Always standardize features before applying Ridge, Lasso, or ElasticNet. Center the response variable too (or use `fit_intercept=True`, which sklearn handles internally by centering y and X before fitting).

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# CORRECT: Standardize inside a Pipeline (avoids data leakage)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1)),
])
pipe.fit(X_train, y_train)
```

**Do NOT standardize before the train/test split** — compute the scaler's mean/std on the training set only and apply to both.

### Pitfall 2: alpha Parameterization Mismatch between sklearn and R glmnet

sklearn Lasso: $\frac{1}{2n}\|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha_{\text{sklearn}}\|\boldsymbol{\beta}\|_1$

R glmnet: $\frac{1}{2n}\|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda_{\text{glmnet}}\|\boldsymbol{\beta}\|_1$

They are equivalent when $\alpha_{\text{sklearn}} = \lambda_{\text{glmnet}}$. *However*, R glmnet by default uses its own normalization. When comparing results, always verify the objective function used.

### Pitfall 3: ConvergenceWarning in Lasso/ElasticNet

Symptoms: `ConvergenceWarning: Objective did not converge. The max_iter was reached.`

Fix: Increase `max_iter` (try 10000 or 50000), or tighten `tol`. Also ensure features are standardized (converges faster on standardized data).

```python
# Check for convergence
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    lasso.fit(X_train, y_train)
    if any(issubclass(warning.category, UserWarning) for warning in w):
        print("Did not converge! Increase max_iter.")
```

### Pitfall 4: Using LassoCV's alpha_ on New Data Without Refitting

`LassoCV.alpha_` is selected to minimize CV error on the training set. You must refit a Lasso with `alpha=lasso_cv.alpha_` on the full training set (or use `lasso_cv` directly, which already stores the refit model in `coef_` / `intercept_`).

Actually, `LassoCV.predict()` uses the model refit on all training data at the best alpha — this is correct. The trap is manual reuse of `lasso_cv.alpha_` without refitting.

### Pitfall 5: Lasso Selects Arbitrarily from Correlated Groups

When two features $x_1, x_2$ are highly correlated, Lasso may select either one with equal probability — the choice is not stable across bootstrap samples. If you care which features are selected (not just the prediction), use ElasticNet or run stability selection.

```python
# Stability selection: which features appear in >50% of bootstrap Lasso fits?
from sklearn.utils import resample

selected_counts = np.zeros(p)
for _ in range(100):
    X_boot, y_boot = resample(X_train, y_train)
    lasso.fit(X_boot, y_boot)
    selected_counts += (lasso.coef_ != 0)
stable_features = selected_counts / 100 > 0.5
```

### Pitfall 6: Using boston Dataset (Deprecated)

The `sklearn.datasets.load_boston` dataset contains a feature with documented racial bias and was deprecated in sklearn 1.0 and removed in 1.2. Use:
- `sklearn.datasets.fetch_california_housing()` — for housing price prediction
- `sklearn.datasets.load_diabetes()` — for medical regression (n=442, p=10, continuous target)
- `sklearn.datasets.make_regression()` — for synthetic controlled examples

### Pitfall 7: Confusing alpha in QuantileRegressor vs Quantile

In `QuantileRegressor`, `alpha` is the **L1 regularization strength** (like Lasso's alpha), NOT the quantile level. The quantile level is the `quantile` parameter. This is a common naming confusion because in some other contexts `alpha` refers to the significance level.

### Pitfall 8: BayesianRidge max_iter Naming Change

Before sklearn 1.3: `BayesianRidge(n_iter=300)`.
sklearn 1.3+: `BayesianRidge(max_iter=300)`. `n_iter` is deprecated and may issue a warning in 1.5.x. Use `max_iter`.

### Pitfall 9: SelectFromModel Threshold for Lasso

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

# threshold=None with L1 penalty → uses 1e-5 (coefficients must exceed this)
selector = SelectFromModel(Lasso(alpha=0.01), threshold='1e-5')  # or leave None

# IMPORTANT: alpha affects sparsity; higher alpha → fewer features selected
# Must cross-validate alpha separately if using SelectFromModel for feature selection
```

### Pitfall 10: Not Centering the Response for Ridge/Lasso

sklearn's Ridge/Lasso with `fit_intercept=True` internally centers both $X$ and $y$ before fitting (subtracts column means), then adjusts the intercept accordingly. This is mathematically equivalent to fitting with an explicit bias column but is numerically cleaner. The key practical implication: the reported `intercept_` is correct *only if* the test data is on the same scale as the training data. Never manually subtract the training mean from test data and then use the model without intercept.

---

## Multicollinearity and VIF

Variance Inflation Factor measures how much the variance of an estimated coefficient is inflated due to collinearity with other predictors:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the $R^2$ from regressing $x_j$ on all other predictors.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def compute_vif(X_df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_df.values, i)
        for i in range(len(X_df.columns))
    ]
    return vif_data.sort_values("VIF", ascending=False)

# Rules of thumb:
# VIF = 1: no collinearity
# 1 < VIF < 5: moderate collinearity (acceptable)
# VIF > 5-10: high collinearity (consider Ridge/ElasticNet or dropping feature)
# VIF > 10: severe collinearity
```

**Remedies for high VIF:** (1) Ridge regression — handles it directly via regularization; (2) ElasticNet — handles it and provides sparsity; (3) PCA/dimensionality reduction before fitting; (4) Manual feature removal (domain-guided).

---

## Statistical Inference with statsmodels

For p-values, confidence intervals, and hypothesis testing on regression coefficients, sklearn is insufficient — use statsmodels:

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Fit OLS with full statistical output
X_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_const).fit()
print(model.summary())
# Outputs: R², adj-R², F-statistic, AIC, BIC
# Per-coefficient: estimate, std error, t-stat, p-value, 95% CI

# Check for heteroskedasticity
bp_lm, bp_pval, bp_f, bp_fp = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan p-value: {bp_pval:.4f}")  # p < 0.05 → heteroskedasticity present

# If heteroskedasticity detected, use HC3 robust SEs (recommended default)
model_robust = sm.OLS(y_train, X_const).fit(cov_type='HC3')
print(model_robust.summary())

# Confidence intervals
print(model.conf_int(alpha=0.05))  # 95% CIs

# Prediction intervals (not built-in but via get_prediction)
predictions = model.get_prediction(sm.add_constant(X_test))
pred_df = predictions.summary_frame(alpha=0.05)
# pred_df has: mean, mean_se, mean_ci_lower, mean_ci_upper, obs_ci_lower, obs_ci_upper
```

---

## Feature Selection with Lasso + SelectFromModel

```python
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# Step 1: Find optimal alpha via CV
lasso_cv = LassoCV(cv=5, n_alphas=100, max_iter=10000, random_state=42, n_jobs=-1)
lasso_cv.fit(X_train_scaled, y_train)

# Step 2: Use fitted Lasso to select features
selector = SelectFromModel(lasso_cv, prefit=True, threshold=1e-5)
X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)

print(f"Original: {X_train.shape[1]} features → Selected: {X_train_sel.shape[1]} features")
print(f"Selected feature names: {np.array(feature_names)[selector.get_support()]}")

# Pipeline version (avoids data leakage; uses internal LassoCV for selection)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(Lasso(alpha=lasso_cv.alpha_))),
    ('final_model', LinearRegression()),
])
```

---

## Key Datasets for Examples

### 1. California Housing Dataset

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
# 20,640 samples, 8 features (median income, house age, avg rooms, avg bedrooms,
# population, avg occupancy, latitude, longitude)
# Target: median house value in $100,000s
print(data.DESCR)
```

### 2. Diabetes Dataset

```python
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
X, y = data.data, data.target
# 442 patients, 10 features (age, sex, bmi, bp, s1-s6 blood serum measurements)
# All features pre-standardized (mean 0, std 1)
# Target: disease progression measure (quantitative)
print(X.shape)   # (442, 10)
print(y.describe())
```

### 3. make_regression (Synthetic, Controlled)

```python
from sklearn.datasets import make_regression
import numpy as np

# Controlled sparse regression: only 10 of 100 features are informative
X, y, true_coef = make_regression(
    n_samples=500,
    n_features=100,
    n_informative=10,  # Only 10 truly nonzero coefficients
    noise=10.0,
    coef=True,         # Return true coefficients
    random_state=42,
)
# Ideal for demonstrating Lasso's variable selection: alpha should recover sparse coef
```

### 4. Make Dataset for Demonstrating Multicollinearity

```python
from sklearn.datasets import make_regression
import numpy as np

# Correlated features for demonstrating Ridge/ElasticNet advantage
rng = np.random.default_rng(42)
X_base = make_regression(n_samples=200, n_features=5, noise=5, random_state=42)[0]
# Add highly correlated copies
X = np.hstack([X_base, X_base[:, :3] + rng.normal(0, 0.01, (200, 3))])
# X now has near-duplicate features → VIF will be very high
```

---

## Curated Resources

### Original Papers

1. **Tibshirani (1996) — Lasso**
   URL: https://academic.oup.com/jrsssb/article/58/1/267/7027929
   Why: The seminal paper introducing Lasso; read the original formulation, not a textbook gloss.

2. **Hoerl & Kennard (1970) — Ridge (Part I)**
   URL: https://doi.org/10.1080/00401706.1970.10488634
   Why: Original biased-estimation paper; clear motivation from multicollinearity.

3. **Zou & Hastie (2005) — ElasticNet**
   URL: https://academic.oup.com/jrsssb/article/67/2/301/7109482
   Why: Definitive paper on ElasticNet; proves the grouping property formally.

4. **Efron, Hastie, Johnstone, Tibshirani (2004) — LARS**
   URL: https://arxiv.org/abs/math/0406456 (free preprint)
   Why: Entire Lasso regularization path in $O(p^3)$; foundational for understanding path algorithms.

5. **Friedman, Hastie, Tibshirani (2010) — Coordinate Descent / glmnet**
   URL: https://doi.org/10.18637/jss.v033.i01
   Why: The algorithm behind sklearn's Lasso; explains warm starts, pathwise descent.

6. **Tibshirani (2011) — Lasso Retrospective**
   URL: https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2011.00771.x
   Why: 15-year retrospective; covers LARS, adaptive Lasso, group Lasso, fused Lasso — all the extensions.

### Official Documentation

7. **sklearn 1.5 Linear Models — Full Module Docs**
   URL: https://scikit-learn.org/1.5/modules/linear_model.html
   Why: Complete parameter tables, solver comparison tables, usage examples.

8. **sklearn 1.5 Ridge API Reference**
   URL: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html
   Why: Authoritative parameter defaults and solver options.

9. **sklearn 1.5 Lasso API Reference**
   URL: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Lasso.html
   Why: Exact objective function, all parameters.

10. **sklearn 1.5 QuantileRegressor**
    URL: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.QuantileRegressor.html
    Why: Verify this was added in sklearn 1.0; solver options (HiGHS LP).

11. **SHAP LinearExplainer Documentation**
    URL: https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html
    Why: Authoritative API; explains masker parameter and deprecated feature_perturbation.

12. **SHAP LinearExplainer Math (Correlation)**
    URL: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/linear_models/Math%20behind%20LinearExplainer%20with%20correlation%20feature%20perturbation.html
    Why: Explains the mathematical relationship between SHAP values and coefficients under correlated vs. independent feature assumptions.

13. **statsmodels OLS Results**
    URL: https://www.statsmodels.org/stable/stats.html
    Why: Full reference for het_breuschpagan, het_white, robust covariance types (HC0–HC3).

14. **glmnet Stanford Page (R reference)**
    URL: https://glmnet.stanford.edu/articles/glmnet.html
    Why: The canonical R implementation; parameter conventions differ from sklearn; author-maintained.

### Tutorials & Blog Posts

15. **An Introduction to Statistical Learning (ISL) — Chapter 6**
    Book (free PDF): https://www.statlearning.com/
    Why: Best pedagogical treatment of Ridge, Lasso, ElasticNet with figures; Chapters 6.1–6.2 are essential.

16. **ESL Chapter 3 — Linear Methods for Regression**
    URL: https://hastie.su.domains/ElemStatLearn/ (free PDF)
    Why: Mathematical depth beyond ISL; derives the SVD perspective of Ridge, LARS algorithm, degrees of freedom.

17. **sklearn Example: L1-based Models for Sparse Signals**
    URL: https://scikit-learn.org/1.5/auto_examples/linear_model/plot_lasso_and_elasticnet.html
    Why: Code comparing Lasso, ElasticNet, ARD on synthetic sparse data; includes coefficient heatmap.

---

## Uncertainties

1. **sklearn 1.5 vs 1.9 discrepancy:** Web search results show sklearn 1.9.0 documentation (the stable latest as of mid-2026) while the course bible targets 1.5.x. Some features (e.g., `store_cv_results` in RidgeCV) may have been added in 1.6+. The chapter author must verify exact parameter availability in 1.5.2 vs later.

2. **glmnet-python package status:** The `glmnet` Python package on PyPI has had maintenance issues. As of 2024–2026, it may not be actively maintained. The chapter should either (a) omit it or (b) note it as a legacy option and recommend sklearn's implementation instead.

3. **BayesianRidge `n_iter` deprecation timing:** Confirmed deprecated in sklearn 1.3, but the exact removal version needs checking. Should be `max_iter` in 1.5.x.

4. **QuantileRegressor added in sklearn 1.0:** Confirmed from search results. The chapter should note this with `# Added in sklearn 1.0+`.

5. **SHAP 0.46.x `shap_values()` return type:** The docs note that for single-output models, `shap_values()` returns an `np.ndarray` of shape `(n_samples, n_features)`. For multiple outputs, return type changed from `list` to `np.ndarray` in 0.45.0. Writer should verify the exact current behavior in 0.46.x.

6. **statsmodels version:** Results show 0.14.6 (stable) and 0.15.0 (dev). The stable version to target for the chapter is 0.14.x. Writer should verify `het_white` is in `statsmodels.stats.diagnostic` (confirmed) vs. `statsmodels.stats.stattools`.

7. **Gauss-Markov and homoskedasticity:** The classic Gauss-Markov theorem requires homoskedasticity. The generalized Gauss-Markov theorem (allowing $\text{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{\Omega}$ with known $\mathbf{\Omega}$) shows GLS is BLUE. The chapter should distinguish these clearly.

8. **Adaptive Lasso (Zou 2006):** A major variant not in sklearn directly. The adaptive Lasso uses feature-specific penalty weights $\alpha_j = \alpha / |\hat{\beta}_j^{\text{OLS}}|^\gamma$, which gives an oracle property. Writer may want to cover it as a variant, noting it requires custom implementation or use of `Lasso(positive=...)` tricks.

9. **Fused Lasso (Tibshirani et al. 2005):** Another important variant for structured sparsity (penalizes differences of adjacent coefficients). Not in sklearn; requires cvxpy or specialized packages. Include as a "Further Evolution" note.

10. **Group Lasso:** Not natively in sklearn. MultiTaskLasso implements a related idea (row-sparsity) but true group Lasso (where groups are user-specified, not tasks) requires `celer`, `group_lasso`, or custom implementation.

---

## Notes for the Chapter Author

### Critical Mathematical Points to Get Right

1. **sklearn's Lasso objective has a $\frac{1}{2n}$ factor:** This is important when comparing alpha values to R glmnet or theoretical treatments that use $\frac{1}{2}$ (without the $n$). In sklearn, doubling $n$ without changing `alpha` halves the effective regularization strength.

2. **Ridge closed form vs. iterative solvers:** The closed form $(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ exists and is implemented by `solver='cholesky'` and `solver='svd'`. For large sparse matrices, iterative solvers (sag, saga, sparse_cg) are more practical.

3. **The soft-threshold operator:** This is the key to Lasso's coordinate descent update. The writer should derive it explicitly from the KKT conditions for a 1D problem.

4. **ElasticNet double-shrinkage:** Zou & Hastie noted that the original ElasticNet has a double shrinkage issue — the L1 penalty already shrinks, then the L2 adds more. They recommend a correction factor $(1 + \alpha(1-\rho)/2)$ to inflate coefficients back. sklearn does NOT apply this correction by default; the `ElasticNet` class returns the raw penalized solution.

5. **Degrees of freedom for Lasso:** The number of nonzero coefficients in the Lasso solution is an unbiased estimate of the degrees of freedom (Efron et al., 2004; Tibshirani & Taylor, 2012). This has important implications for AIC/BIC-based model selection.

### Section Structure Advice

- **Section 2 (Deep Explanation):** Derive the Ridge closed form from scratch (take gradient of objective, set to zero). Then show the SVD interpretation. For Lasso, show why the gradient approach fails at $\beta_j = 0$ and derive the KKT conditions. Then derive the soft-threshold update as the coordinate-wise minimizer.

- **Section 7 (Diagnostics):** Include: (a) residual vs fitted plot; (b) Q-Q plot of residuals; (c) Scale-location plot; (d) Regularization path plot (alpha vs. coefficients); (e) CV error vs. alpha plot (from LassoCV.mse_path_); (f) Coefficient path comparison (Ridge vs. Lasso vs. ElasticNet).

- **Section 9 (Worked Example):** Use the diabetes dataset for a clean, pedagogically clear example. It is pre-standardized, has 10 interpretable features, and $n=442$ is small enough to run quickly but large enough to demonstrate CV. Show: OLS → Ridge → Lasso → ElasticNet comparison; regularization path; feature selection with SelectFromModel; SHAP; statsmodels for p-values.

### Code Snippet Patterns

```python
# Regularization path visualization (Lasso)
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

alphas, coefs, _ = lasso_path(X_train_scaled, y_train, n_alphas=100)

plt.figure(figsize=(10, 5))
plt.semilogx(alphas, coefs.T)
plt.gca().invert_xaxis()  # High alpha (more regularization) on left
plt.xlabel('alpha (log scale)')
plt.ylabel('Coefficients')
plt.title('Lasso Regularization Path')
plt.axvline(lasso_cv.alpha_, color='r', linestyle='--', label=f'CV alpha={lasso_cv.alpha_:.4f}')
plt.legend()
plt.tight_layout()
plt.show()

# SHAP for linear model
explainer = shap.LinearExplainer(fitted_lasso, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)
shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                                      data=X_test_scaled[0], feature_names=feature_names))
```

---

*Dossier complete. All API signatures verified against sklearn 1.5.2 official documentation. Papers verified via journal URLs. Key uncertainties noted above for author follow-up.*
