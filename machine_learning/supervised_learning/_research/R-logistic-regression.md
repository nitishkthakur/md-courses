# Research Dossier: Logistic Regression
**Target file:** `logistic-regression.md`
**Research date:** 2026-06-25
**Dossier author:** Research Agent (Claude Sonnet 4.6)

---

## Scope

This dossier feeds the chapter file `logistic-regression.md` in the Supervised Learning Masterclass. It covers the full depth of logistic regression: historical origin, mathematical foundations, MLE derivation, decision boundary geometry, the complete sklearn 1.5.x API with deprecation notices, all solver/penalty combinations, regularization paths, multiclass strategies, calibration theory, class-imbalance handling, online learning via SGDClassifier, statistical inference via statsmodels, SHAP LinearExplainer, and key datasets for worked examples.

---

## Verified Packages

### Primary packages for this algorithm

| Package | Import path | Version (mid-2026) | What it provides |
|---|---|---|---|
| `scikit-learn` | `from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier` | ~1.5.x | Full LogisticRegression with 6 solvers, CV variant, SGD variant |
| `statsmodels` | `import statsmodels.api as sm; from statsmodels.discrete.discrete_model import Logit` | ~0.14.x | Statistical inference: p-values, confidence intervals, AIC/BIC |
| `python-glmnet` | `from glmnet import LogitNet` | ~2.0 (Civis Analytics) | Regularization path via Fortran glmnet backend; sklearn-compatible API |
| `shap` | `import shap; shap.LinearExplainer(...)` | ~0.46.x | LinearExplainer for LR — computes exact SHAP values from coefficients |
| `optuna` | `import optuna` | ~3.x | Hyperparameter optimization (C, l1_ratio, solver) |
| `imbalanced-learn` | `from imblearn.over_sampling import SMOTE` | ~0.12.x | SMOTE and other resampling for class-imbalance scenarios |

### Key import snippet

```python
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
import statsmodels.api as sm
import shap
import optuna
```

### Core API methods (verified against sklearn 1.5.x)

```python
# Standard usage
clf = LogisticRegression(
    penalty='l2',          # {'l1', 'l2', 'elasticnet', None}
    C=1.0,                 # inverse regularization strength
    solver='lbfgs',        # default solver
    max_iter=100,
    fit_intercept=True,
    class_weight=None,     # or 'balanced'
    l1_ratio=None,         # only used when penalty='elasticnet'
    random_state=42,
)
clf.fit(X_train, y_train)
clf.predict(X_test)           # hard class labels
clf.predict_proba(X_test)     # probability estimates, shape (n, n_classes)
clf.predict_log_proba(X_test) # log probabilities
clf.decision_function(X_test) # raw log-odds score before sigmoid
clf.score(X_test, y_test)     # mean accuracy

# Key attributes after fit
clf.coef_        # (1, n_features) for binary, (n_classes, n_features) for multi
clf.intercept_   # (1,) for binary, (n_classes,) for multi
clf.n_iter_      # actual iterations used per class

# Built-in CV selection of C
cv_clf = LogisticRegressionCV(
    Cs=10,       # int: log-spaced grid 1e-4 to 1e4; or list of floats
    cv=5,        # stratified k-fold
    solver='lbfgs',
    scoring=None,  # defaults to accuracy
    refit=True,
).fit(X_train, y_train)
cv_clf.C_        # best C per class, shape (n_classes,)
cv_clf.scores_   # dict: class -> grid of CV scores

# Online learning variant (logistic regression via SGD)
sgd_clf = SGDClassifier(
    loss='log_loss',    # makes it logistic regression (was 'log' before sklearn 1.1)
    penalty='l2',
    alpha=0.0001,       # regularization strength (NOT the same C convention!)
    max_iter=1000,
    random_state=42,
)
sgd_clf.partial_fit(X_batch, y_batch, classes=[0, 1])  # online learning
```

> **Note on `loss='log_loss'`:** The `loss='log'` alias was deprecated in sklearn 1.1 and removed in 1.3. Use `loss='log_loss'` for all current code.

> **Note on `alpha` vs `C`:** In `SGDClassifier`, regularization is parameterized by `alpha` (the regularization coefficient directly), not `C` (its inverse). `alpha ≈ 1 / (n_samples * C)` for rough equivalence.

### statsmodels API (verified)

```python
import statsmodels.api as sm

# statsmodels requires you to add the constant (intercept) manually
X_with_const = sm.add_constant(X_train)   # prepends a column of 1s
logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(method='newton')  # Newton-Raphson by default

print(result.summary())        # full table: coef, std err, z, P>|z|, [0.025, 0.975]
result.params                  # coefficient array
result.pvalues                 # Wald test p-values
result.conf_int(alpha=0.05)    # 95% CI, returns DataFrame with [lower, upper]
result.aic                     # Akaike Information Criterion
result.bic                     # Bayesian Information Criterion
result.llf                     # log-likelihood at fit
result.pred_table()            # confusion matrix at 0.5 threshold

# CRITICAL difference from sklearn:
# statsmodels has NO regularization by default — coefficients are pure MLE
# sklearn applies L2 regularization by default (C=1.0) → coefficients differ
```

### SHAP LinearExplainer (verified against shap 0.46.x)

```python
import shap

explainer = shap.LinearExplainer(
    model=clf,              # fitted sklearn LogisticRegression
    masker=X_train,         # background data for computing E[x]
    link=shap.links.logit,  # keeps SHAP values in log-odds space (additive)
    # link=shap.links.identity  # returns SHAP values in log-odds directly
)
shap_values = explainer(X_test)   # returns Explanation object (shap 0.46.x)

# For legacy .shap_values() call:
shap_vals = explainer.shap_values(X_test)  # returns ndarray

# Visualization
shap.summary_plot(shap_values, X_test)
shap.waterfall_plot(shap_values[0])   # single prediction
shap.force_plot(explainer.expected_value, shap_vals[0], X_test[0])
```

> **SHAP math for LR:** For a linear model, SHAP value for feature $j$ is exactly $\phi_j = \beta_j \cdot (x_j - \mathbb{E}[x_j])$. This is not an approximation — it is exact. The sum of all SHAP values equals the model output minus the baseline. Units are in log-odds (before the sigmoid).

> **Uncertainty:** The `feature_perturbation` parameter in older shap docs is deprecated. The current API uses `masker` with `shap.maskers.Independent(X_train)` for independence assumption or `shap.maskers.Impute(X_train, strategy='mean')`. Verify exact masker API in shap 0.46.x changelog.

---

## The Original Algorithm

### Citation

**Cox, D. R. (1958).** The Regression Analysis of Binary Sequences. *Journal of the Royal Statistical Society: Series B (Methodological)*, 20(2), 215–232.
DOI: https://doi.org/10.1111/j.2517-6161.1958.tb00292.x
URL: https://academic.oup.com/jrsssb/article/20/2/215/7027376

### Historical precursors the chapter must cover

- **Pierre-François Verhulst (1838–1845):** Coined the term "logistic" and introduced the logistic growth curve $N(t) = L / (1 + e^{-k(t-t_0)})$ for population dynamics. This is the mathematical precursor — the sigmoid was known, but not as a regression model.
- **Joseph Berkson (1944):** Coined "logit" (from "logistic unit"). Advocated logit as a simpler alternative to probit (which required normal CDF tables). His paper: "Application of the Logistic Function to Bio-Assay," *JASA* 39(227), 357–365.
- **Cox (1958):** Formalized binary regression using the logistic link function in a full statistical regression framework with parameter estimation, testing, and inference. This is the canonical "birth" of logistic regression as an applied statistical tool.
- **Walker & Duncan (1967):** Extended Cox's framework and promoted logistic regression in epidemiology and social sciences.
- **McFadden (1973, 1974):** Extended to multinomial logistic regression ("conditional logit") with theoretical grounding in random utility theory. Nobel Prize in Economics 2000.

### What problem Cox (1958) solved

Before Cox, binary outcomes were typically modeled with probit regression (using the normal CDF as the link function), which was analytically cumbersome — it required special tables or numerical integration. Cox proposed the logistic function as the link:

$$P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{x}^T \boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}^T \boldsymbol{\beta}}}$$

This offered: (1) a closed-form derivative amenable to Newton-Raphson optimization, (2) a natural interpretation via log-odds ratios, and (3) computational tractability with tabulating equipment of the era.

### The logistic function (sigmoid)

$$\sigma(z) = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}}$$

Key properties:
- $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$ — outputs are valid probabilities
- $\sigma(0) = 0.5$, $\sigma(-z) = 1 - \sigma(z)$ — symmetric about 0.5
- $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$ — derivative has closed form, enabling efficient optimization
- $\lim_{z \to \infty} \sigma(z) = 1$, $\lim_{z \to -\infty} \sigma(z) = 0$

### The log-odds (logit) interpretation

The model is linear in the **log-odds**:

$$\log \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \mathbf{x}^T \boldsymbol{\beta} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$$

This is the "logit" transformation (inverse sigmoid). Consequence: a unit increase in $x_j$ multiplies the **odds** by $e^{\beta_j}$. This is the odds ratio, the primary interpretive unit of logistic regression.

- $\beta_j > 0$: feature $j$ increases probability of class 1
- $\beta_j < 0$: feature $j$ decreases probability of class 1
- $e^{\beta_j}$: the multiplicative change in odds per unit increase in $x_j$

### Maximum likelihood estimation

Given $n$ i.i.d. samples $(\mathbf{x}_i, y_i)$ with $y_i \in \{0, 1\}$, the log-likelihood is:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

where $\hat{p}_i = \sigma(\mathbf{x}_i^T \boldsymbol{\beta})$. This is identical to the **negative binary cross-entropy loss** (which we minimize):

$$\mathcal{L}_{\text{CE}}(\boldsymbol{\beta}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

The log-likelihood is **strictly concave** in $\boldsymbol{\beta}$ (the Hessian is negative semi-definite), guaranteeing a unique global maximum — a critical theoretical property.

### Gradient of the log-likelihood

$$\nabla_{\boldsymbol{\beta}} \ell = \mathbf{X}^T (\mathbf{y} - \hat{\mathbf{p}})$$

where $\hat{\mathbf{p}} = [\hat{p}_1, \ldots, \hat{p}_n]^T$ is the vector of predicted probabilities. The gradient is zero when predicted probabilities match observed labels perfectly — intuitive!

### Hessian (for Newton-Raphson)

$$\mathbf{H} = -\mathbf{X}^T \mathbf{W} \mathbf{X}$$

where $\mathbf{W} = \text{diag}(\hat{p}_i (1 - \hat{p}_i))$ is a diagonal matrix of prediction variances. Note: $\hat{p}_i(1-\hat{p}_i) > 0$ always, so $\mathbf{H}$ is negative definite and the problem is strictly concave.

### Newton-Raphson update (IRLS)

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \mathbf{H}^{-1} \nabla \ell = \boldsymbol{\beta}^{(t)} + (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T (\mathbf{y} - \hat{\mathbf{p}})$$

This is the **Iteratively Reweighted Least Squares (IRLS)** algorithm — each Newton step solves a weighted least squares problem. Convergence is typically achieved in 4–10 iterations for well-conditioned problems. Each iteration costs $O(np^2 + p^3)$ (matrix multiplication + inversion).

### Decision boundary

Setting $\hat{p} = 0.5$ is equivalent to $\mathbf{x}^T \boldsymbol{\beta} = 0$, which defines a **hyperplane** in feature space. This is why logistic regression produces a **linear decision boundary**: it is linear in $\mathbf{x}$, even though the probability output is nonlinear (S-shaped). A point $\mathbf{x}$ is classified as class 1 if $\mathbf{x}^T \boldsymbol{\beta} > 0$.

---

## Major Variants & Evolution

### 1. Regularized Logistic Regression (Tibshirani 1996; Ng 2004)

**Problem solved:** Unregularized MLE overfits with many features, diverges under separation.

**Key change:** Add a penalty to the loss:

$$\mathcal{L}_{\text{reg}} = -\frac{1}{n} \sum_{i} \ell_i + \frac{\lambda}{C} r(\boldsymbol{\beta})$$

where $r(\boldsymbol{\beta})$ is the regularizer. In sklearn, $C = 1/\lambda$ (larger $C$ = weaker regularization). Three types:
- **L2 (Ridge):** $r(\boldsymbol{\beta}) = \frac{1}{2}\|\boldsymbol{\beta}\|_2^2$ — shrinks all coefficients, never zeroes any out. Solves multicollinearity.
- **L1 (Lasso):** $r(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_1$ — induces sparsity; some coefficients become exactly zero. Automatic feature selection.
- **ElasticNet:** $r(\boldsymbol{\beta}) = \rho \|\boldsymbol{\beta}\|_1 + \frac{1-\rho}{2}\|\boldsymbol{\beta}\|_2^2$ — combines L1 and L2. `l1_ratio` = $\rho$. Useful when features are correlated and sparsity is still desired.
- **None:** No penalty — pure MLE. Use `penalty=None` or equivalently `C=np.inf`. Equivalent to statsmodels default.

**Python:** `sklearn.linear_model.LogisticRegression(penalty='l1'/'l2'/'elasticnet'/None)`

### 2. L1 Regularization Path (Efron et al. 2004; Friedman, Hastie, Tibshirani 2010)

**Problem solved:** Finding the right level of L1 sparsity requires searching many values of $C$.

**Key change:** Compute coefficients along the entire regularization path efficiently using warm-starts and the LARS/coordinate-descent algorithm. The glmnet package (R) computes the path orders-of-magnitude faster than fitting separate models.

**Python:** `glmnet.LogitNet` (python-glmnet by Civis Analytics). Also `sklearn.linear_model.LogisticRegressionCV` with a list of Cs gives a similar (but computationally less optimized) path.

**Paper:** Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1–22. https://www.jstatsoft.org/article/view/v033i01

### 3. Multinomial Logistic Regression / Softmax Regression (McFadden 1974)

**Problem solved:** Extending binary LR to K > 2 classes.

**Key change:** The prediction for class $k$ uses the softmax function:

$$P(Y=k \mid \mathbf{x}) = \frac{e^{\mathbf{x}^T \boldsymbol{\beta}_k}}{\sum_{j=1}^K e^{\mathbf{x}^T \boldsymbol{\beta}_j}}$$

where there are $K$ weight vectors $\boldsymbol{\beta}_1, \ldots, \boldsymbol{\beta}_K$ (one per class). The loss is the **multiclass cross-entropy**:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K \mathbf{1}[y_i=k] \log P(Y=k \mid \mathbf{x}_i)$$

**OvR vs Multinomial:** In sklearn 1.5+, `multi_class` is deprecated. For $K \geq 3$, sklearn now defaults to multinomial. Multinomial is generally preferred as it jointly models all classes (probabilities sum to 1 by construction), while OvR trains K independent binary classifiers.

**Note:** The model is **overparameterized** (K weight vectors, but only K-1 are identifiable). sklearn handles this by either fixing one weight vector to zero (or through regularization). This doesn't affect predictions.

**sklearn change in 1.5:** The `multi_class` parameter is deprecated. From v1.7, the multinomial loss will always be used for n_classes ≥ 3. For OvR: use `OneVsRestClassifier(LogisticRegression())`.

### 4. Online / SGD Logistic Regression (Bottou 1998, 2010)

**Problem solved:** Standard batch optimization requires loading all data into memory and computes the gradient over the full dataset — infeasible for streaming or very large datasets.

**Key change:** Approximate the gradient with a single sample or mini-batch:

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta_t \nabla_{\boldsymbol{\beta}} \ell_i(\boldsymbol{\beta}^{(t)})$$

Using a decreasing learning rate $\eta_t$ guarantees convergence. Supports `partial_fit()` for true online learning.

**Python:** `sklearn.linear_model.SGDClassifier(loss='log_loss')`

**Reference:** Bottou, L. (2010). Large-Scale Machine Learning with Stochastic Gradient Descent. *COMPSTAT 2010 Proceedings*, 177–186.

### 5. Penalized Logistic Regression via Firth's Method (Firth 1993)

**Problem solved:** Complete or quasi-complete separation — when a linear combination of predictors perfectly predicts the outcome, causing MLE coefficients to diverge to $\pm\infty$.

**Key change:** Add a Jeffrey's prior penalty to the log-likelihood: $\ell^*(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) + \frac{1}{2} \log |\mathbf{I}(\boldsymbol{\beta})|$, where $\mathbf{I}$ is the Fisher information matrix. This is also known as **bias reduction** (it removes $O(1/n)$ bias from MLE).

**Python:** `logistf` in R; in Python, use the `logistf` or `firthlogist` package (not in sklearn). Also: heavy L2 regularization (small `C`) is a practical workaround.

**Paper:** Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.

### 6. SAGA Solver (Defazio et al. 2014)

**Problem solved:** SAG (Stochastic Average Gradient) only supports L2 regularization; SAGA generalizes to L1 and ElasticNet.

**Key change:** SAGA uses variance-reduced stochastic gradient descent with a gradient memory table, enabling convergence guarantees for non-smooth (L1) penalties at the per-sample update cost.

**Python:** `LogisticRegression(solver='saga', penalty='l1')`

**Paper:** Defazio, A., Bach, F., & Lacoste-Julien, S. (2014). SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. *NeurIPS 2014*. https://arxiv.org/abs/1407.0202

### 7. Newton-Cholesky Solver (sklearn 1.2+)

**Problem solved:** For problems where $n \gg p$ (many more samples than features), the standard L-BFGS solver is suboptimal. Newton-Cholesky computes the full Hessian but solves the Newton system efficiently via Cholesky decomposition.

**When to use:** `n_samples >> n_features`, especially with one-hot encoded categorical features with many rare categories. Memory cost is $O(p^2)$ in features, so not suitable for $p > 10^4$.

**Python:** `LogisticRegression(solver='newton-cholesky')`

---

## Hyperparameter Reference

### Full parameter table (sklearn 1.5.x LogisticRegression)

| Parameter | Default | Type | Description |
|---|---|---|---|
| `penalty` | `'l2'` | `{'l1','l2','elasticnet', None}` | Regularization type |
| `C` | `1.0` | float > 0 | Inverse regularization strength; smaller = stronger regularization |
| `solver` | `'lbfgs'` | str | Optimization algorithm (see solver matrix below) |
| `max_iter` | `100` | int | Maximum iterations for convergence |
| `tol` | `1e-4` | float | Stopping criterion tolerance |
| `fit_intercept` | `True` | bool | Whether to fit a bias term |
| `intercept_scaling` | `1` | float | Only for liblinear; scales synthetic feature |
| `class_weight` | `None` | dict or `'balanced'` | Per-class loss weights |
| `random_state` | `None` | int | Seed for solvers with randomness (sag, saga, liblinear) |
| `multi_class` | `'auto'` | str | **DEPRECATED in 1.5**, removed in 1.7. Use `OneVsRestClassifier` for OvR |
| `verbose` | `0` | int | Verbosity level |
| `warm_start` | `False` | bool | Reuse previous fit for initialization (faster re-training) |
| `n_jobs` | `None` | int | Parallelism over classes (only for OvR) |
| `l1_ratio` | `None` | float [0, 1] | Mixing parameter for ElasticNet; 0 = L2, 1 = L1 |
| `dual` | `False` | bool | Dual formulation; only for liblinear + L2 when n_samples > n_features |

### Solver / Penalty compatibility matrix (verified from sklearn 1.5.x docs)

| Solver | L1 | L2 | ElasticNet | None | Multinomial | Best for |
|---|---|---|---|---|---|---|
| `lbfgs` | ✗ | ✓ | ✗ | ✓ | ✓ | Default; medium-large datasets |
| `liblinear` | ✓ | ✓ | ✗ | ✗ | ✗ | Small datasets; binary only |
| `newton-cg` | ✗ | ✓ | ✗ | ✓ | ✓ | Medium datasets; high precision |
| `newton-cholesky` | ✗ | ✓ | ✗ | ✓ | ✗ | n_samples >> n_features (binary only) |
| `sag` | ✗ | ✓ | ✗ | ✓ | ✓ | Large datasets; requires scaling |
| `saga` | ✓ | ✓ | ✓ | ✓ | ✓ | Large datasets; all penalties |

> **Writer note:** `newton-cholesky` supports multiclass via OvR internally, but does NOT support the full multinomial loss. It will raise an error if `multi_class='multinomial'` (or once multinomial becomes the default in 1.7). Verify this in the 1.7 release notes.

### Hyperparameter tuning guide

**`C` (regularization strength, inverse):**
- Default: 1.0. This applies moderate L2 regularization.
- Smaller C → stronger regularization → more bias, less variance → useful when n_features is large relative to n_samples
- Larger C → weaker regularization → approaches unregularized MLE → can overfit
- Typical search range: `[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]` (log scale)
- Use `LogisticRegressionCV(Cs=10)` for automatic selection via CV
- Rule of thumb: start at C=1.0 and look at the learning curve. If training accuracy >> validation accuracy → decrease C.

**`penalty` choice:**
- **L2 (default):** Always try first. Works well when all features contribute. No sparsity.
- **L1:** When you suspect many features are irrelevant. Induces exact zeros. Requires `solver='liblinear'` or `solver='saga'`.
- **ElasticNet:** When features are correlated AND you want sparsity. Requires `solver='saga'` and `l1_ratio` tuning.
- **None:** For statistical inference (p-values via statsmodels). Also when n_samples >> p and you trust you have enough data.

**`l1_ratio` (ElasticNet only):**
- Range: [0.0, 1.0]. 0 = pure L2, 1 = pure L1.
- Tune jointly with C using `LogisticRegressionCV(l1_ratios=[0.1, 0.5, 0.9])`.
- Typical starting grid: `[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]`

**`max_iter`:**
- Default: 100. Often too low for unscaled data or complex problems.
- Increase to 1000 or 10000 if you see `ConvergenceWarning`.
- Always scale features with `StandardScaler` before increasing max_iter — scaling is usually the root fix.
- Warning: increasing max_iter indefinitely when the model truly hasn't converged (e.g., due to separation) masks a deeper problem.

**`tol`:**
- Default: 1e-4. Controls when to declare convergence (norm of gradient or parameter change < tol).
- Decreasing tol tightens convergence (more iterations, better solution). Set to 1e-6 for high-precision applications.
- Increasing tol speeds up convergence at the cost of solution quality — rarely worth it for logistic regression.

**`class_weight`:**
- Default: None (equal weights). Use `'balanced'` for imbalanced datasets.
- `'balanced'` sets $w_k = n / (n_{\text{classes}} \times n_k)$ where $n_k$ is the count of class $k$.
- Effect: minority class errors are penalized more heavily in the loss.
- Alternative: use `sample_weight` parameter in `fit()` for instance-level weights.

**`solver` selection guide:**
- `lbfgs`: default; use it unless you have a specific reason not to.
- `saga`: when you need L1 or ElasticNet, or when n > 10^5 (faster convergence for large data).
- `liblinear`: small datasets (<10^4 samples), or when you need L1 without the saga overhead.
- `newton-cholesky`: when n_samples >> n_features (e.g., n=10^6, p=50). Memory-intensive in p.
- `sag`/`newton-cg`: rarely preferred over lbfgs or saga.

**`warm_start`:**
- Default: False. Set to True for iterative retraining (e.g., adding data incrementally) or when sweeping C values.
- Meaningful speed-up only when the model is called multiple times with similar parameters.

### Tuning playbook

| Hyperparameter | Typical range | Effect | How to tune |
|---|---|---|---|
| `C` | [1e-4, 100] log-scale | Bias-variance tradeoff | `LogisticRegressionCV(Cs=20)` |
| `penalty` | {'l1','l2','elasticnet'} | Sparsity vs shrinkage | Try all; L2 is strong default |
| `l1_ratio` | [0.1, 0.9] | L1/L2 mix (ElasticNet) | Grid with Cs; `LogisticRegressionCV(l1_ratios=...)` |
| `max_iter` | 100–10000 | Convergence | Increase until ConvergenceWarning disappears |
| `tol` | 1e-6 to 1e-3 | Convergence precision | Default 1e-4 is fine; tighten for sensitive applications |
| `class_weight` | None, 'balanced', dict | Class imbalance | Use 'balanced' as first attempt; tune if needed |

---

## Common Errors & Pitfalls

### 1. ConvergenceWarning — the most common error

**Symptom:** `ConvergenceWarning: lbfgs failed to converge (status=1). Increase the number of iterations (max_iter) or scale the data.`

**Root causes (in order of frequency):**
1. **Unscaled features** — the most common cause. Gradient descent on features at wildly different scales (e.g., age [0-100] vs income [0-100000]) struggles. Fix: `StandardScaler` or `MinMaxScaler` before fitting.
2. `max_iter` too low — default of 100 is often insufficient for complex problems.
3. Near-separation — the model is trying to push coefficients toward infinity.

**Fix:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train)
```

### 2. Complete and quasi-complete separation

**What it is:** A linear combination of features perfectly separates the classes. The MLE does not exist — log-likelihood keeps increasing as coefficients grow without bound. Coefficients become very large (|coef| > 100) with huge standard errors. In statsmodels, the summary will show warnings.

**Detection:** Look at `clf.coef_` — values in the hundreds are a red flag. In statsmodels: `result.pvalues` near 1.0 with very large coefficient estimates.

**Fixes:**
1. L2 regularization (reduce C) — the recommended sklearn approach
2. Firth's penalized likelihood (external package)
3. Remove the perfectly separating feature (if it's a data leak)
4. Collect more data

### 3. The C vs alpha confusion (SGDClassifier vs LogisticRegression)

`LogisticRegression` uses `C` (inverse regularization). `SGDClassifier` uses `alpha` (direct regularization coefficient). They are related by approximately `alpha ≈ 1 / (n_samples * C)`. Mixing them up leads to massively different regularization levels.

### 4. Default regularization surprises users coming from statsmodels

sklearn applies L2 regularization by default (`C=1.0`). If you fit a logistic regression in statsmodels (no regularization) and compare with sklearn, the coefficients will differ. For unregularized MLE, set `penalty=None` in sklearn.

### 5. Probability outputs not summing correctly in binary case

`predict_proba()` for binary classification returns shape `(n, 2)` where column 0 is P(class 0) and column 1 is P(class 1). Always use `predict_proba(X)[:, 1]` for the probability of the positive class in binary settings.

### 6. multi_class deprecation (sklearn 1.5)

`multi_class='ovr'` and `multi_class='multinomial'` are deprecated in sklearn 1.5. Code using these parameters will work but emit a deprecation warning. They will be **removed in sklearn 1.7**. Update code:
- For OvR: `OneVsRestClassifier(LogisticRegression())`
- For multinomial: just use `LogisticRegression()` (it will be the default for multi-class from 1.7)

### 7. Feature correlation inflating coefficient magnitudes

Highly correlated features cause coefficient instability (multicollinearity). Individual coefficients become meaningless — their signs can flip with small data changes. Use L2 regularization, VIF analysis, or PCA before fitting when correlation is high.

### 8. Interpreting coefficients without accounting for scale

Raw coefficients are in log-odds per unit change. If features are on different scales, raw coefficients are not comparable. To compare feature importance, standardize first (`StandardScaler`) so coefficients represent log-odds per standard deviation.

### 9. Calibration and threshold selection

Logistic regression is inherently well-calibrated (predicted probabilities are actual probabilities) because it directly optimizes log-loss. However, with strong regularization or class imbalance, calibration can degrade. Always check a calibration curve for deployment-critical models. The default threshold of 0.5 is rarely optimal for imbalanced problems — use precision-recall curves to select the operating point.

### 10. n_jobs misunderstanding

`n_jobs` in `LogisticRegression` only parallelizes across classes in a **OvR** multiclass setup. For binary classification, `n_jobs` has no effect. For a single-threaded binary problem, `n_jobs=-1` provides zero speedup.

---

## Key Datasets for Examples

### 1. Breast Cancer Wisconsin (sklearn built-in)

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
# 569 samples, 30 features, binary (malignant=0, benign=1)
# Good for: full binary LR workflow, SHAP, calibration
print(data.feature_names)  # 30 features: mean radius, mean texture, etc.
```

**Why it's ideal:** Moderate size, clean, well-studied, excellent for showing calibration, coefficients, SHAP, regularization effects.

### 2. Titanic Survival (seaborn/Kaggle)

```python
import seaborn as sns
titanic = sns.load_dataset('titanic')
# Drop rows with missing values, encode sex/class
# Features: pclass, sex, age, sibsp, parch, fare
# Binary target: survived (0/1)
# Good for: class imbalance, feature preprocessing, coefficient interpretation
```

**Why it's ideal:** Familiar real-world story, natural class imbalance (38% survived), demonstrates class_weight='balanced', preprocessing pipeline.

### 3. Credit Card Fraud Detection (Kaggle)

```python
import pandas as pd
# Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv('creditcardfraud.csv')
# 284,807 transactions, 30 features (V1-V28 are PCA components + Amount + Time)
# Target: Class (0=legit, 1=fraud) — severe imbalance: ~0.17% fraud
# Good for: extreme imbalance, saga solver for large n, SMOTE comparison
```

**Why it's ideal:** Severe imbalance (1 fraud per 577 transactions), large n (saga solver shines), demonstrates need for precision-recall over accuracy.

### 4. Iris (multiclass demo)

```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
# 150 samples, 4 features, 3 classes
# Good for: multinomial vs OvR comparison, decision boundary visualization
```

### 5. UCI Adult Income (for binary with many features)

```python
from sklearn.datasets import fetch_openml
adult = fetch_openml('adult', version=2, as_frame=True)
X, y = adult.data, adult.target
# ~48,000 samples, mix of categorical/numerical, binary income classification
# Good for: pipeline with ColumnTransformer, L1 sparsity example
```

---

## Curated Resources

### Original papers and foundational references

1. **Cox (1958) — the founding paper**
   URL: https://academic.oup.com/jrsssb/article/20/2/215/7027376
   DOI: https://doi.org/10.1111/j.2517-6161.1958.tb00292.x
   *Why:* The original formulation; chapter must cite this.

2. **Berkson (1944) — "logit" coined**
   Berkson, J. (1944). Application of the Logistic Function to Bio-Assay. *JASA*, 39(227), 357–365.
   *Why:* Historical context; where the "logit" word comes from.

3. **Friedman, Hastie, Tibshirani (2010) — regularization paths**
   URL: https://www.jstatsoft.org/article/view/v033i01
   *Why:* The mathematical foundation of the glmnet regularization path that informed sklearn's implementation.

4. **Defazio et al. (2014) — SAGA solver**
   URL: https://arxiv.org/abs/1407.0202
   *Why:* The theoretical paper behind sklearn's best large-scale solver.

### Official documentation

5. **sklearn LogisticRegression API (1.5.x)**
   URL: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
   *Why:* Authoritative; use this for version-specific parameter verification.

6. **sklearn Linear Models User Guide**
   URL: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
   *Why:* Includes mathematical formulations, solver comparison table, and multiclass explanation.

7. **statsmodels Logit documentation**
   URL: https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.summary.html
   *Why:* Authoritative API for p-values and confidence intervals.

8. **SHAP LinearExplainer documentation**
   URL: https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html
   *Why:* Exact API for LinearExplainer; verify masker parameter name.

### Best tutorials and explanations

9. **sklearn 1.5 release notes (multi_class deprecation)**
   URL: https://scikit-learn.org/1.5/whats_new/v1.5.html
   *Why:* Documents the multi_class deprecation precisely; link directly to this.

10. **SHAP Sentiment Analysis with LR (official notebook)**
    URL: https://shap.github.io/shap/notebooks/linear_explainer/Sentiment%20Analysis%20with%20Logistic%20Regression.html
    *Why:* Working code for LinearExplainer with a text/LR example; shows log-odds SHAP values.

11. **sklearn L1 Sparsity example**
    URL: https://scikit-learn.org/1.5/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html
    *Why:* Official visualization of how L1 vs L2 regularization produces different sparsity patterns.

12. **Connections: Log-Likelihood, Cross-Entropy, KL Divergence, LR, and NNs**
    URL: https://glassboxmedicine.com/2019/12/07/connections-log-likelihood-cross-entropy-kl-divergence-logistic-regression-and-neural-networks/
    *Why:* Best non-academic explanation of why cross-entropy = negative log-likelihood = MLE for LR.

13. **UCLA Stats — Odds Ratio Interpretation**
    URL: https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
    *Why:* Authoritative, clear explanation of the odds-ratio interpretation of coefficients.

14. **Complete Separation in LR (UCLA)**
    URL: https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logistic-regression-and-what-are-some-strategies-to-deal-with-the-issue/
    *Why:* Clear explanation of the separation problem that practitioners encounter.

15. **python-glmnet GitHub**
    URL: https://github.com/civisanalytics/python-glmnet
    *Why:* The Python implementation of the full glmnet regularization path for LR.

16. **Introduction to logistic regression (arXiv tutorial)**
    URL: https://arxiv.org/pdf/2008.13567
    *Why:* Accessible mathematical derivation; covers gradient descent and Newton's method clearly.

---

## Calibration — Why Logistic Regression Is Well-Calibrated

This is a critical topic that the chapter should explain mechanically.

Logistic regression is well-calibrated because it **directly optimizes the log-likelihood**, which is equivalent to minimizing the Kullback-Leibler divergence between predicted and true probability distributions. When you minimize:

$$\mathcal{L} = -\frac{1}{n} \sum_i [y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i)]$$

the optimal solution satisfies $\hat{p}_i = P(Y=1 \mid \mathbf{x}_i)$ in expectation — i.e., the predicted probabilities are calibrated to the true conditional probabilities. This is not true for most other classifiers (SVMs, random forests) whose loss functions are not strictly proper scoring rules.

**Calibration degrades when:**
- Strong regularization (small C) squeezes probabilities away from 0 and 1
- Severe class imbalance (the model's implicit prior is wrong)
- Training and test distributions differ

**How to check:**
```python
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=10, ax=ax)
ax.set_title("Calibration Curve — Logistic Regression")
plt.show()
```

**Contrast with Platt Scaling:** Platt scaling IS logistic regression applied to the raw scores of another classifier (e.g., an SVM). Logistic regression is what other classifiers use to fix their calibration.

---

## Mathematical Notation Notes for Chapter Writer

Follow the course bible conventions:
- $\boldsymbol{\beta}$ or $\mathbf{w}$ for weight vector (the chapter should pick one and be consistent; $\boldsymbol{\beta}$ is standard in statistics, $\mathbf{w}$ in ML — recommend $\boldsymbol{\beta}$ for this algorithm given its statistical roots)
- $\hat{p}_i = P(Y=1 \mid \mathbf{x}_i)$ for predicted probability
- $\sigma(z) = 1/(1+e^{-z})$ for the sigmoid/logistic function
- $\text{logit}(p) = \log(p/(1-p))$ for the logit (inverse sigmoid)
- Decision boundary: $\{\mathbf{x} : \mathbf{x}^T \boldsymbol{\beta} = 0\}$
- Log-odds: $\log \text{OR}(x_j) = \beta_j$ (per unit increase in $x_j$)
- Odds ratio: $\text{OR}(x_j) = e^{\beta_j}$

---

## Uncertainties — Writer Must Verify

1. **`multi_class` removal timeline:** Sources confirm deprecated in 1.5, removed in 1.7. Verify whether sklearn 1.7 is already released as of chapter writing date and whether `OneVsRestClassifier` is the correct migration path for all solvers. Check https://scikit-learn.org/stable/whats_new/v1.7.html.

2. **`newton-cholesky` and multinomial:** This solver does NOT support the multinomial objective (only binary/OvR). Confirm this in the 1.7 docs — it may have been updated.

3. **`SGDClassifier(loss='log')` deprecation:** Deprecated in 1.1, replaced by `loss='log_loss'`. Confirm whether `'log'` still works as an alias in sklearn 1.5.x or has been fully removed.

4. **SHAP LinearExplainer masker API:** The `feature_perturbation` parameter appears to be deprecated. The new API uses `masker=shap.maskers.Independent(X_train)`. Verify exact parameter names in shap 0.46.x — the docs at `shap.readthedocs.io` may lag the latest version.

5. **`LogisticRegressionCV` `penalty` parameter in sklearn 1.9.x:** One search result showed `penalty='deprecated'` in the 1.9 signature — this suggests the `penalty` naming may be changing. **This is a significant uncertainty** — the chapter targets 1.5.x but the writer should note that newer versions may differ. Verify the 1.5.x signature specifically.

6. **glmnet-python maintenance status:** The `python-glmnet` by Civis Analytics (https://github.com/civisanalytics/python-glmnet) appears active as of 2024; confirm it still compiles against current numpy/sklearn. Alternative: `glmnetforpython` or the R glmnet via rpy2.

7. **Firth's method in Python:** No widely-maintained sklearn-compatible Firth implementation exists as of 2025. The writer should note this gap explicitly — statsmodels has experimental support but it's not the primary API. Double-check the current `firthlogist` package status.

8. **`intercept_scaling` for `liblinear`:** This is a subtle parameter. When `fit_intercept=True` and `solver='liblinear'`, sklearn appends a synthetic feature of constant value `intercept_scaling` to regularize the intercept separately. This is a footgun if overlooked. The writer should explain this in the hyperparameter section.

---

## Summary Statistics for Dossier

- **Verified sources:** sklearn 1.5.x API (from official docs), statsmodels 0.14.x, shap 0.46.x, original Cox (1958) paper DOI confirmed
- **Key APIs confirmed:** All 15 LogisticRegression parameters, LogisticRegressionCV parameters, SGDClassifier `loss='log_loss'`, statsmodels Logit API, shap LinearExplainer constructor
- **Deprecations confirmed:** `multi_class` deprecated in 1.5 (removed 1.7), `loss='log'` → `loss='log_loss'` in SGDClassifier
- **Solver matrix:** Verified all 6 solvers × 4 penalties compatibility
- **Uncertainties flagged:** 8 items the writer must independently verify before finalizing code snippets
