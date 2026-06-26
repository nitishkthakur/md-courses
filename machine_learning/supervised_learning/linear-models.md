# Linear Models: Ridge, Lasso, ElasticNet — The Complete Masterclass

> **Why this algorithm matters:** Every machine learning practitioner learns linear regression.
> Almost none of them truly understand it. The gap between calling `LinearRegression().fit()` and
> genuinely knowing the geometry of the L1 ball, the SVD interpretation of ridge shrinkage, the
> exact form of the soft-threshold operator, the bias-variance frontier, and the conditions under
> which each variant dominates — that gap is enormous, and it is the gap this chapter closes.
>
> Linear models with regularization are not a beginner tool you graduate away from. Ridge
> regression is the backbone of genomics (where $p \gg n$ is the norm and you must handle
> hundreds of thousands of SNPs), Lasso powers variable selection in economics and epidemiology,
> and ElasticNet is the default first-pass model at many high-stakes production systems. When
> Netflix engineers estimate viewing propensity, when epidemiologists identify risk factors from
> electronic health records, when quant funds build factor models, linear models with proper
> regularization are often the starting point — and frequently the ending point. Understanding
> them at the level of Tikhonov, Hoerl, and Tibshirani is not academic luxury. It is professional
> necessity.

---

## §0 — Python Ecosystem & Package Guide

Before touching mathematics, you need to know the landscape. The linear model family in Python is
vast — sklearn alone exposes over 20 estimators under `sklearn.linear_model`, and there are
important capabilities in `statsmodels`, `glmnet`, and SHAP that sklearn does not cover. Choosing
the wrong package means missing p-values, missing the regularization path, or worse, using an
API that silently parameterizes alpha differently from what you expect.

### Complete Package Table

| Package | Import | Version (mid-2026) | Key Capability | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.linear_model import Ridge, Lasso, ElasticNet` | ~1.5.x | Full family: OLS, Ridge, Lasso, ElasticNet, Bayesian, robust; Pipeline-compatible | BSD-3 |
| `statsmodels` | `import statsmodels.api as sm` | ~0.14.x | Statistical inference: p-values, CIs, F-tests, AIC/BIC, diagnostic tests | BSD-3 |
| `glmnet` (Python) | `from glmnet import ElasticNet as GlmnetEN` | Legacy (unmaintained) | R-style regularization path via Fortran backend; use sklearn instead | BSD-3 |
| `linearmodels` | `from linearmodels import OLS as LmOLS` | ~6.x | Panel data OLS, IV regression, robust/cluster-robust covariance | Apache-2 |
| `scipy.stats` | `from scipy import stats` | ~1.13.x | `linregress()` for simple regression with p-values; correlation tests | BSD-3 |
| `shap` | `import shap` | ~0.46.x | `LinearExplainer` for exact SHAP values from any linear model | MIT |
| `optuna` | `import optuna` | ~3.x | Bayesian hyperparameter search for `alpha`, `l1_ratio` | MIT |

### Top Picks — Recommendation Table

| Package | Best for | Modern/Legacy | Notes |
|---|---|---|---|
| **`sklearn.linear_model`** (Ridge/Lasso/ElasticNet) | Production ML pipelines, regularization, feature selection | **Modern — first choice** | Fast, Pipeline-ready, 20+ estimator variants; 95% of use cases |
| **`sklearn.linear_model.RidgeCV / LassoCV / ElasticNetCV`** | Automatic alpha selection via CV | **Modern — always use these over bare Ridge/Lasso** | Built-in path algorithm; much faster than GridSearchCV |
| **`statsmodels.OLS`** | Statistical inference, p-values, diagnostic tests | **Modern — use alongside sklearn** | Use when you need CIs, hypothesis testing, residual diagnostics |
| **`sklearn.linear_model.BayesianRidge`** | Uncertainty quantification, auto-regularization | **Modern** | No CV needed; outputs predictive uncertainty via `.predict(return_std=True)` |
| **`statsmodels` + `sklearn` pipeline** | Full workflow: inference + prediction | **Modern — best of both worlds** | Fit statsmodels for inference; sklearn for production scoring |

### The Same Fit Across Top Packages

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data (used throughout this chapter)
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize (mandatory for regularized models)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Option 1: sklearn OLS (baseline, no regularization) ───────────────────
from sklearn.linear_model import LinearRegression

ols = LinearRegression()
ols.fit(X_train_sc, y_train)
print(f"OLS R²:     {ols.score(X_test_sc, y_test):.4f}")

# ── Option 2: sklearn Ridge ────────────────────────────────────────────────
from sklearn.linear_model import RidgeCV
import numpy as np

ridge_cv = RidgeCV(alphas=np.logspace(-3, 5, 200), cv=None)  # LOO via GCV
ridge_cv.fit(X_train_sc, y_train)
print(f"Ridge R²:   {ridge_cv.score(X_test_sc, y_test):.4f}  alpha={ridge_cv.alpha_:.4f}")

# ── Option 3: sklearn Lasso (with CV path) ────────────────────────────────
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(n_alphas=100, cv=5, max_iter=10000, n_jobs=-1, random_state=42)
lasso_cv.fit(X_train_sc, y_train)
print(f"Lasso R²:   {lasso_cv.score(X_test_sc, y_test):.4f}  alpha={lasso_cv.alpha_:.4f}")
print(f"  Nonzero coefs: {(lasso_cv.coef_ != 0).sum()}/{len(lasso_cv.coef_)}")

# ── Option 4: sklearn ElasticNet (with CV path) ───────────────────────────
from sklearn.linear_model import ElasticNetCV

enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    n_alphas=100, cv=5, max_iter=10000, n_jobs=-1, random_state=42
)
enet_cv.fit(X_train_sc, y_train)
print(f"ElasticNet R²: {enet_cv.score(X_test_sc, y_test):.4f}  "
      f"alpha={enet_cv.alpha_:.4f}  l1_ratio={enet_cv.l1_ratio_:.2f}")

# ── Option 5: statsmodels OLS (inference) ────────────────────────────────
import statsmodels.api as sm

X_const = sm.add_constant(X_train_sc)
sm_model = sm.OLS(y_train, X_const).fit()
print(sm_model.summary().tables[0])  # F-stat, R², AIC/BIC
# sm_model.summary().tables[1]  # coefficient table with p-values

# ── Option 6: BayesianRidge (auto-regularization + uncertainty) ───────────
from sklearn.linear_model import BayesianRidge

br = BayesianRidge(max_iter=300, compute_score=True)
br.fit(X_train_sc, y_train)
y_pred_br, y_std_br = br.predict(X_test_sc, return_std=True)
from sklearn.metrics import r2_score
print(f"BayesianRidge R²: {r2_score(y_test, y_pred_br):.4f}  "
      f"avg uncertainty: {y_std_br.mean():.2f}")
```

> 💡 **Intuition:** Think of these packages as instruments at different levels of the
> measurement hierarchy. sklearn is your production lab balance — calibrated, repeatable, fast,
> integrates with everything. statsmodels is your analytical chemistry setup — slower, but gives
> you error bars, test statistics, and confidence intervals that a practitioner publishing results
> cannot omit. BayesianRidge is your Bayesian uncertainty meter — it tells you not just the
> estimate but your epistemic uncertainty about it. Pick based on what question you are answering:
> "what is the best prediction?" (sklearn), "is this coefficient statistically significant?"
> (statsmodels), or "how confident should I be in this forecast?" (BayesianRidge).

> 🔧 **In practice:** The single most important setup decision: always use the `CV` variants
> (`RidgeCV`, `LassoCV`, `ElasticNetCV`) rather than bare `Ridge`, `Lasso`, `ElasticNet`. The CV
> variants use warm-started path algorithms — computing 100 alpha values is barely slower than
> computing 1. There is almost never a reason to manually grid-search alpha for linear models.

> ⚠️ **Pitfall:** The `glmnet` Python package on PyPI has had maintenance issues as of 2024–2026
> and may not be actively maintained. For all practical purposes, sklearn's coordinate descent
> implementation is equivalent and far more production-ready. Do not use the Python `glmnet`
> package for new projects.

---

## §1 — The Origin: A 200-Year Story in Three Acts

The story of linear models with regularization spans over two centuries, from the orbits of
comets to the genetics of cancer. Understanding this history is not nostalgia — it reveals *why*
each method was invented, and why the design choices (L1 vs. L2, closed-form vs. iterative)
that seem arbitrary are in fact deeply motivated.

### Act I — Ordinary Least Squares: Legendre and Gauss (1805–1809)

**The problem:** In the early 1800s, astronomers needed to predict the orbits of newly discovered
celestial bodies from a handful of noisy measurements. Given $n$ observations of position and
$p$ unknown orbital parameters, they needed to fit a linear system $\mathbf{y} \approx \mathbf{X}\boldsymbol{\beta}$
where $n > p$ (overdetermined) and the observations were contaminated by measurement error.

**The innovation:** Adrien-Marie Legendre published the method of least squares in 1805 in
*"Nouvelles méthodes pour la détermination des orbites des comètes"* — he proposed minimizing
the sum of squared residuals. Carl Friedrich Gauss claimed independent prior discovery and
provided the statistical justification in 1809's *"Theoria Motus Corporum Coelestium"*, showing
that least squares is the maximum likelihood estimator when errors are normally distributed.

> 📜 **Origin/Citation:** Legendre, A. M. (1805). *Nouvelles méthodes pour la détermination
> des orbites des comètes*. Paris: Courcier. — The first published statement of the
> least-squares principle.
>
> Gauss, C. F. (1809). *Theoria Motus Corporum Coelestium*. Hamburg: Perthes & Besser.
> — Statistical justification via the normal distribution.

**The Gauss-Markov theorem (the bedrock result):** Under four classical assumptions — (1)
linearity: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$; (2) full rank:
$\text{rank}(\mathbf{X}) = p$; (3) strict exogeneity: $\mathbb{E}[\boldsymbol{\varepsilon} | \mathbf{X}] = \mathbf{0}$;
(4) spherical errors: $\text{Var}(\boldsymbol{\varepsilon} | \mathbf{X}) = \sigma^2 \mathbf{I}$ — the
OLS estimator

$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

is the **Best Linear Unbiased Estimator (BLUE)**: among all *linear unbiased* estimators of
$\boldsymbol{\beta}$, OLS has the smallest variance for every linear combination $\mathbf{c}^\top\boldsymbol{\beta}$.

This is a powerful result, but notice the qualifying adjective: *unbiased*. The Gauss-Markov
theorem says nothing about biased estimators. A biased estimator could have *lower mean squared
error* than OLS by trading some bias for a substantial reduction in variance. This loophole is
exactly what Ridge regression exploits.

**When OLS fails catastrophically:**
1. **Multicollinearity:** When columns of $\mathbf{X}$ are nearly linearly dependent,
   $\mathbf{X}^\top\mathbf{X}$ is near-singular. Its smallest eigenvalue $\lambda_{\min} \approx 0$.
   The condition number $\kappa = \lambda_{\max}/\lambda_{\min} \gg 1$. A tiny perturbation in
   $\mathbf{y}$ (e.g., measurement noise) causes enormous swings in $\hat{\boldsymbol{\beta}}$.
   Variance explodes.
2. **$p > n$:** The system is underdetermined — infinitely many $\boldsymbol{\beta}$ minimize the
   squared loss. $\mathbf{X}^\top\mathbf{X}$ is singular by construction. OLS has no unique solution.
3. **Outliers:** The squared loss magnifies the influence of observations with large residuals.
   A single extreme outlier can dominate the entire fit.
4. **Heteroskedasticity:** OLS is still unbiased but the standard errors are wrong, making
   hypothesis tests unreliable.

Each of these failures motivated a different branch of the linear model family tree.

### Act II — Ridge Regression: Hoerl & Kennard (1970)

By the 1960s, multicollinearity was recognized as a serious practical problem in regression.
Social scientists, chemists, and econometricians were fitting regression models where predictors
were highly correlated — chemical concentrations, economic indicators, biological measurements —
and finding that OLS produced wildly unstable coefficient estimates with enormous standard errors.

Arthur Hoerl (DuPont) and Robert Kennard (Drexel) proposed a strikingly simple fix in two
companion papers in *Technometrics* (1970): add a small constant $\alpha$ to the diagonal of
$\mathbf{X}^\top\mathbf{X}$ before inverting.

> 📜 **Full Citation — Ridge (I):** Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression:
> Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
> https://doi.org/10.1080/00401706.1970.10488634
>
> **Full Citation — Ridge (II):** Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression:
> Applications to nonorthogonal problems. *Technometrics*, 12(1), 69–82.
> https://doi.org/10.1080/00401706.1970.10488635
>
> **Tikhonov connection:** This is a special case of *Tikhonov regularization*, independently
> developed in the Soviet Union: Tikhonov, A. N. (1943/1963). On the stability of inverse
> problems. *Doklady Akademii Nauk SSSR*, 39(5), 195–198.

**The Ridge objective:**

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}} \underbrace{\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2}_{\text{data fit}} + \underbrace{\alpha \|\boldsymbol{\beta}\|_2^2}_{\text{L2 penalty}}$$

**The closed-form solution:**

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^\top\mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

The beauty: $(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})$ is *always* positive-definite for
$\alpha > 0$, even when $\mathbf{X}^\top\mathbf{X}$ is singular. Adding $\alpha$ to the diagonal
literally pushes the problematic near-zero eigenvalues away from zero, making the matrix safely
invertible.

**The key insight about bias and variance:** Ridge is a *biased* estimator. The expected value
of $\hat{\boldsymbol{\beta}}_{\text{Ridge}}$ is not $\boldsymbol{\beta}$ (unlike OLS). But its
variance is smaller — and for a suitable range of $\alpha > 0$, the reduction in variance
dominates the introduced bias, making the MSE lower than OLS. This does not contradict
Gauss-Markov, because that theorem only guarantees OLS is best among *unbiased* estimators.

Hoerl and Kennard showed that there always exists an $\alpha^* > 0$ such that the total MSE of
Ridge is strictly less than that of OLS. The challenge is that $\alpha^*$ depends on the unknown
$\boldsymbol{\beta}$ and $\sigma^2$ — which is why we use cross-validation to select it.

**What was new and uncertain in 1970:** The theoretical existence of a better $\alpha$ was proven,
but the paper offered no practical guidance on *how* to choose $\alpha$ beyond plotting the "ridge
trace" (coefficients vs. $\alpha$) and looking for stability. Cross-validation for this purpose
was not systematically explored until the 1980s.

### Act III — The Lasso: Tibshirani (1996)

Ridge solves the multicollinearity and $p > n$ problems elegantly, but it has one persistent
limitation: **it never sets any coefficient to exactly zero**. A Ridge-fit model always uses all
$p$ predictors. For interpretability — "which 10 genes out of 20,000 are associated with cancer
progression?" — Ridge offers nothing. You would need to inspect the magnitude of each coefficient
and make a subjective judgment about what constitutes "essentially zero."

The alternatives in 1996 were:
- **Best subset selection:** Exhaustively search all $2^p$ subsets. Optimal in principle, but
  NP-hard for $p > 30$, and famously unstable — a small change in data can flip the selected
  subset entirely.
- **Stepwise selection:** Forward/backward/stepwise. Fast but highly unstable, ignores multiple
  comparisons, and has no regularization.

Robert Tibshirani's 1996 paper introduced the **Lasso** — Least Absolute Shrinkage and Selection
Operator — combining shrinkage (like Ridge) with variable selection (like subset selection) in a
single convex optimization problem.

> 📜 **Full Citation — Lasso:** Tibshirani, R. (1996). Regression shrinkage and selection via
> the lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1),
> 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
> Free access: https://academic.oup.com/jrsssb/article/58/1/267/7027929

**The Lasso objective (sklearn parameterization):**

$$\hat{\boldsymbol{\beta}}_{\text{Lasso}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1 \right\}$$

where $\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|$. The $\frac{1}{2n}$ factor is sklearn's
normalization convention — it makes the optimal $\alpha$ roughly scale-invariant with respect to
$n$.

**Why does L1 give exact zeros, while L2 does not?** This is one of the most important geometric
insights in modern statistics. The L2 penalty $\|\boldsymbol{\beta}\|_2^2 \leq t$ defines a
smooth sphere in $\mathbb{R}^p$. The L1 penalty $\|\boldsymbol{\beta}\|_1 \leq t$ defines a
*hyperdiamond* (cross-polytope) — a polytope whose corners point along the coordinate axes.

When we minimize the sum of the loss function (with elliptical contours) subject to the
constraint $\|\boldsymbol{\beta}\|_1 \leq t$, the optimal point is where the ellipsoid
*first touches* the constraint set. For the sphere, the touch point is generically on the smooth
surface — not at a corner — so all coordinates can be nonzero. For the hyperdiamond, the touch
point is *typically at a corner*, where all but one (or a few) coordinates are zero. This is
exactly the sparsity property.

**What was experimental/uncertain in 1996:** Tibshirani proved the objective is convex, gave a
coordinate-shooting algorithm, and showed impressive empirical results. But several questions were
open: theoretical guarantees on *when* Lasso correctly identifies the true support (sign consistency,
now called the "oracle property") were not established until the mid-2000s. The efficient
coordinate descent algorithm (Friedman et al. 2010) was still 14 years away. The paper originally
noted: "At present the LASSO seems especially well suited for use when the true model is known to
be sparse," which was prescient but not yet rigorously proven.

---

## §2 — The Algorithm, Deeply Explained

### 2.1 Ordinary Least Squares — The Foundation

Let $\mathbf{X} \in \mathbb{R}^{n \times p}$ be the design matrix (rows are samples, columns are
features), $\mathbf{y} \in \mathbb{R}^n$ the target vector, and $\boldsymbol{\beta} \in \mathbb{R}^p$
the coefficient vector. We seek $\boldsymbol{\beta}$ that minimizes the **residual sum of squares**:

$$\mathcal{L}_{\text{OLS}}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 = \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2$$

**Derivation of the normal equations:** Take the gradient, set to zero:

$$\nabla_{\boldsymbol{\beta}} \mathcal{L}_{\text{OLS}} = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}$$

$$\mathbf{X}^\top\mathbf{X} \boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y}$$

$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} \quad \text{(when } \mathbf{X}^\top\mathbf{X} \text{ is invertible)}$$

**Computational complexity:**
- Forming $\mathbf{X}^\top\mathbf{X}$: $O(np^2)$
- Inverting $\mathbf{X}^\top\mathbf{X}$ (Cholesky): $O(p^3)$
- Total training: $O(np^2 + p^3)$ — dominated by $O(np^2)$ when $n \gg p$
- Inference per sample: $O(p)$

For $n = 100{,}000$ and $p = 1{,}000$: training costs $\approx 10^{11}$ operations — expensive but
feasible. For $p = 100{,}000$ (genomics): $O(p^3) = 10^{15}$ — impossible without dimensionality
reduction or iterative solvers.

### 2.2 Ridge Regression — Full Derivation

**Objective:**

$$\mathcal{L}_{\text{Ridge}}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_2^2$$

**Derivation:**

$$\nabla_{\boldsymbol{\beta}} \mathcal{L}_{\text{Ridge}} = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + 2\alpha\boldsymbol{\beta} = \mathbf{0}$$

$$(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}$$

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^\top\mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

The matrix $(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})$ has eigenvalues $\lambda_j + \alpha$
where $\lambda_j \geq 0$ are eigenvalues of $\mathbf{X}^\top\mathbf{X}$. Since $\alpha > 0$,
*all* eigenvalues of the penalized matrix are strictly positive — the matrix is always
positive-definite and invertible, regardless of whether $\mathbf{X}^\top\mathbf{X}$ is singular.

**The SVD perspective (the deepest intuition for Ridge):**

Let $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$ be the thin SVD, where:
- $\mathbf{U} \in \mathbb{R}^{n \times p}$: left singular vectors (sample space)
- $\mathbf{D} = \text{diag}(d_1, \ldots, d_p)$: singular values $d_1 \geq \cdots \geq d_p \geq 0$
- $\mathbf{V} \in \mathbb{R}^{p \times p}$: right singular vectors (feature space)

The OLS solution in terms of SVD:

$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = \mathbf{V}\mathbf{D}^{-1}\mathbf{U}^\top\mathbf{y} = \sum_{j=1}^p \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j} \mathbf{v}_j$$

Small singular values $d_j \approx 0$ correspond to near-collinear directions. Their reciprocals
$1/d_j$ explode — making OLS wildly sensitive to noise in those directions.

The Ridge solution:

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \mathbf{V} \text{diag}\!\left(\frac{d_j}{d_j^2 + \alpha}\right) \mathbf{U}^\top \mathbf{y} = \sum_{j=1}^p \frac{d_j}{d_j^2 + \alpha} (\mathbf{u}_j^\top \mathbf{y}) \mathbf{v}_j$$

Each principal component direction $\mathbf{v}_j$ is scaled by the **shrinkage factor**
$d_j^2/(d_j^2 + \alpha)$:

- When $d_j^2 \gg \alpha$: shrinkage factor $\approx 1$ — direction is barely shrunk.
- When $d_j^2 \ll \alpha$: shrinkage factor $\approx d_j^2/\alpha \approx 0$ — direction is
  nearly zeroed out.

**Ridge shrinks near-collinear directions (small $d_j$) heavily and leaves strongly supported
directions (large $d_j$) nearly unchanged.** This is why Ridge is so effective at handling
multicollinearity: it surgically suppresses the problematic near-null-space directions.

> 💡 **Intuition:** Imagine your data lies mostly in a 3-dimensional subspace of a 50-dimensional
> feature space. Ridge finds that subspace (via SVD) and essentially ignores the remaining 47
> dimensions — they get shrunk toward zero. OLS would assign large, noisy coefficients to those
> 47 dimensions because the least-squares fit is indifferent between the signal subspace and the
> noise.

**Bias-variance tradeoff for Ridge:**

$$\text{MSE}(\hat{\boldsymbol{\beta}}_{\text{Ridge}}) = \underbrace{\text{Bias}^2}_{\alpha^2 \boldsymbol{\beta}^\top(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})^{-2}\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta}} + \underbrace{\text{Variance}}_{\sigma^2 \text{tr}\!\left[(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})^{-2}\mathbf{X}^\top\mathbf{X}\right]}$$

As $\alpha$ increases: Bias² increases (Ridge pulls estimates toward zero) but Variance
decreases. The optimal $\alpha$ balances these. Hoerl and Kennard proved that the minimum-MSE
$\alpha^* = p\sigma^2 / \|\boldsymbol{\beta}\|_2^2 > 0$ always exists — but it depends on
unknowns, so we cross-validate.

### 2.3 Lasso Regression — Why the Gradient Breaks Down

**Objective:**

$$\mathcal{L}_{\text{Lasso}}(\boldsymbol{\beta}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1$$

The squared-loss term is smooth (infinitely differentiable). But $|\beta_j|$ is **not
differentiable at $\beta_j = 0$**. Setting a gradient to zero the Ridge way fails — there is no
gradient at the candidate optimum.

**Subgradient conditions (KKT):** The Lasso is a convex optimization problem. The KKT conditions
(necessary and sufficient for convexity) give:

$$\frac{1}{n}\mathbf{x}_j^\top (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \alpha \cdot s_j$$

where $s_j$ is the subgradient of $|\beta_j|$: $s_j = \text{sign}(\hat{\beta}_j)$ if
$\hat{\beta}_j \neq 0$, and $s_j \in [-1, 1]$ if $\hat{\beta}_j = 0$.

Equivalently:
$$\hat{\beta}_j \neq 0 \implies \frac{1}{n}\mathbf{x}_j^\top (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \alpha \cdot \text{sign}(\hat{\beta}_j)$$

$$\hat{\beta}_j = 0 \implies \left|\frac{1}{n}\mathbf{x}_j^\top (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})\right| \leq \alpha$$

**Interpretation:** A predictor is excluded from the Lasso model if and only if its inner product
with the current residual (its "correlation with the residual") has magnitude at most $\alpha$.
The threshold $\alpha$ is the Lasso's *activity gate* — features whose signal is weaker than
$\alpha$ get set to zero.

### 2.4 The Soft-Threshold Operator — The Engine of Lasso

For a 1D Lasso problem, or for a single coordinate in coordinate descent:

$$\min_{\beta} \frac{1}{2}(z - \beta)^2 + \alpha|\beta|$$

Taking the subgradient and setting to zero:

- If $z > \alpha$: optimal $\beta = z - \alpha$ (shrink from above)
- If $z < -\alpha$: optimal $\beta = z + \alpha$ (shrink from below)
- If $|z| \leq \alpha$: optimal $\beta = 0$ (threshold to zero)

This is the **soft-threshold operator**:

$$\mathcal{S}(z, \alpha) = \text{sign}(z)(|z| - \alpha)_+ = \begin{cases} z - \alpha & z > \alpha \\ 0 & |z| \leq \alpha \\ z + \alpha & z < -\alpha \end{cases}$$

> 💡 **Intuition:** Soft-thresholding is like a signal denoiser. If the signal is strong enough
> (above $\alpha$), it passes through — reduced by $\alpha$ (shrunk toward zero, not to zero).
> If the signal is weak (below $\alpha$), it is killed entirely. This is fundamentally different
> from hard-thresholding (which would pass the signal unchanged or kill it entirely) — soft-
> thresholding gives a continuous, differentiable-in-its-domain solution.

**Coordinate descent for Lasso** (Friedman, Hastie, Tibshirani, 2010): Cycle through coordinates
$j = 1, 2, \ldots, p, 1, 2, \ldots$ until convergence. At each step, compute the **partial
residual** holding all other coordinates fixed:

$$r_j = \mathbf{y} - \sum_{k \neq j} \mathbf{x}_k \hat{\beta}_k = \mathbf{y} - \mathbf{X}_{-j}\hat{\boldsymbol{\beta}}_{-j}$$

Then update:

$$\hat{\beta}_j \leftarrow \frac{\mathcal{S}\!\left(\frac{1}{n}\mathbf{x}_j^\top r_j,\; \alpha\right)}{\frac{1}{n}\|\mathbf{x}_j\|_2^2}$$

This is a closed-form update per coordinate. For standardized features ($\|\mathbf{x}_j\|_2^2 = n$),
the denominator is 1 and the update simplifies to pure soft-thresholding. Each cycle through all
$p$ coordinates costs $O(np)$. Convergence to global optimum is guaranteed (for convex objectives
with coordinate-separable regularization).

### 2.5 ElasticNet — Combining L1 and L2

**Objective:**

$$\mathcal{L}_{\text{EN}}(\boldsymbol{\beta}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha\rho\|\boldsymbol{\beta}\|_1 + \frac{\alpha(1-\rho)}{2}\|\boldsymbol{\beta}\|_2^2$$

where `l1_ratio` $\rho \in [0, 1]$ controls the balance. The coordinate update for ElasticNet
modifies the denominator only:

$$\hat{\beta}_j \leftarrow \frac{\mathcal{S}\!\left(\frac{1}{n}\mathbf{x}_j^\top r_j,\; \alpha\rho\right)}{\frac{1}{n}\|\mathbf{x}_j\|_2^2 + \alpha(1-\rho)}$$

The L2 penalty adds $\alpha(1-\rho)$ to the denominator, which *shrinks* the effective update
(additional Ridge-like shrinkage on top of the soft-threshold). This is what produces the
**grouping effect**: correlated features share their soft-threshold credit instead of one being
selected arbitrarily.

> 💡 **Intuition:** Think of Lasso as a harsh selector — it picks one representative from each
> group of correlated features and discards the others. ElasticNet is more diplomatic — it says
> "these features contribute similarly, let's give them proportional credit." The L2 penalty
> introduces a preference for similar-magnitude coefficients among correlated features.

### 2.6 Assumptions of the Linear Model Family

All models in this family share an implicit set of assumptions. Understanding when they are
violated tells you when to trust the model and when to reach for something else.

| Assumption | What it means | What breaks if violated | Detection | Mitigation |
|---|---|---|---|---|
| **Linearity** | $\mathbb{E}[y \mid \mathbf{x}] = \mathbf{x}^\top\boldsymbol{\beta}$ | Systematic pattern in residuals vs. fitted | Residuals vs. fitted plot | Feature engineering: polynomials, interactions, transforms |
| **Independence** | Errors $\varepsilon_i$ are uncorrelated | OLS estimates still unbiased but SEs wrong; time series fail badly | ACF/PACF of residuals | Time-series models (ARIMA); clustered SEs |
| **Homoskedasticity** | $\text{Var}(\varepsilon_i) = \sigma^2$ (constant) | OLS still unbiased; hypothesis tests unreliable | Scale-location plot; Breusch-Pagan test | Robust SEs (HC3); WLS; transform target |
| **Normality of errors** | $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ | Only affects inference (CIs, p-values), not point estimates | Q-Q plot of residuals | Huber regression; quantile regression |
| **No perfect multicollinearity** | $\text{rank}(\mathbf{X}) = p$ | OLS not uniquely defined | VIF > 10; condition number >> 1 | Ridge (handles gracefully); drop/combine features |
| **Correct specification** | True model is linear in $\boldsymbol{\beta}$ | Consistent bias | Ramsey RESET test | Nonlinear models; generalized additive models |

> ⚠️ **Pitfall:** The linearity assumption is in the *parameters* $\boldsymbol{\beta}$, not in
> the *features* $\mathbf{x}$. The model $y = \beta_0 + \beta_1 x + \beta_2 x^2$ is linear in
> $\boldsymbol{\beta}$ and is perfectly valid as a linear model. "Linear model" does not mean
> "linear relationship with the original features."

### 2.7 Computational Complexity Summary

| Model | Training | Inference | Path (100 alphas) | Notes |
|---|---|---|---|---|
| OLS | $O(np^2 + p^3)$ | $O(p)$ | N/A | Cholesky decomposition |
| Ridge | $O(np^2 + p^3)$ | $O(p)$ | ~same (closed form) | SVD: $O(\min(np^2, n^2p))$; iterative for large sparse |
| Lasso | $O(np \cdot T_\text{iter})$ | $O(p)$ | $O(np)$ (warm starts) | $T_\text{iter}$ = iterations to converge; path with warm starts |
| ElasticNet | $O(np \cdot T_\text{iter})$ | $O(p)$ | $O(np)$ | Same as Lasso |
| LARS | $O(p \cdot \min(n,p)^2)$ | $O(p)$ | Exact path | Faster than Lasso when $p \ll n$ |

> 🔬 **Deep dive:** The "warm start" path algorithm for Lasso is the reason `LassoCV` can compute
> 100 regularization values almost as fast as 1. Starting at $\alpha_{\max}$ (where all
> coefficients are zero), each successive smaller $\alpha$ is solved starting from the previous
> solution, which is nearby. Active set sizes grow slowly along the path, so most iterations are
> cheap. This is why the coordinate descent Lasso is so practical for production use.

---

## §3 — The Full Evolution: From Ridge to the Modern Family

The original Ridge and Lasso papers opened a research program that has produced dozens of
important variants. This section traces the major branches, each solving a specific limitation
of what came before.

### 3.1 RidgeCV — Automatic Alpha via Generalized Cross-Validation

> 📜 **Paper:** Golub, G. H., Heath, M., & Wahba, G. (1979). Generalized cross-validation as a
> method for choosing a good ridge parameter. *Technometrics*, 21(2), 215–223.

**Problem solved:** Ridge solves multicollinearity, but choosing $\alpha$ by hand is subjective.
Golub, Heath, and Wahba showed that for Ridge specifically, leave-one-out (LOO) cross-validation
has a computationally cheap equivalent: the **Generalized Cross-Validation (GCV)** criterion,

$$\text{GCV}(\alpha) = \frac{1}{n}\frac{\|\mathbf{y} - \hat{\mathbf{y}}_\alpha\|_2^2}{(1 - \text{df}(\alpha)/n)^2}$$

where $\text{df}(\alpha) = \sum_j d_j^2/(d_j^2 + \alpha)$ is the effective degrees of freedom.
This can be computed for any $\alpha$ from a single SVD of $\mathbf{X}$ without refitting —
making LOO CV for Ridge computationally free.

```python
from sklearn.linear_model import RidgeCV
import numpy as np

# cv=None → LOO via GCV (exact, no refitting required!)
ridge_cv = RidgeCV(
    alphas=np.logspace(-6, 6, 200),  # 200 log-spaced candidates
    cv=None,                          # None = GCV (fast LOO)
    scoring=None,                     # None = R² (default for regressors)
    store_cv_results=True,            # Store per-alpha scores (sklearn 1.5+)
)
ridge_cv.fit(X_train_sc, y_train)
print(f"Best alpha: {ridge_cv.alpha_:.4f}")
# Access CV scores for all alphas:
if hasattr(ridge_cv, 'cv_results_'):
    print(f"CV results shape: {ridge_cv.cv_results_.shape}")  # (n_samples, n_alphas)
```

> 🔧 **In practice:** `RidgeCV` with `cv=None` uses GCV — it is the default and usually the
> best choice. For large $n$, GCV is computationally free (one SVD). You only need `cv=5` or
> similar if you want k-fold cross-validation for a specific scoring metric other than R².

### 3.2 LARS — Least Angle Regression: The Exact Path Algorithm

> 📜 **Paper:** Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle
> regression. *The Annals of Statistics*, 32(2), 407–499.
> https://doi.org/10.1214/009053604000000067
> Free preprint: https://arxiv.org/abs/math/0406456

**Problem solved:** In 2004, computing the full Lasso path (solutions at all $\alpha$ values)
was expensive — you would naively need to solve the Lasso from scratch at each of hundreds of
$\alpha$ values. Efron et al. showed the Lasso regularization path is **piecewise linear** in
$\alpha$, and provided LARS: an algorithm that traces the *entire* path in $O(p \cdot \min(n,p)^2)$
time — the same order as a single OLS fit on $\min(n,p)$ features.

**Key insight:** LARS builds the active set incrementally. At each step:
1. Find the predictor $x_j$ most correlated with the current residual $r = y - \hat{y}$.
2. Move $\hat{\beta}_j$ in the direction of its correlation with $r$, at a rate that keeps it the
   *most* correlated predictor.
3. Continue until a second predictor becomes equally correlated with the residual. Add it to the
   active set.
4. Now both active predictors move together, at rates that keep them equally correlated.
5. Continue until all $p$ predictors are active (OLS solution) or a stopping criterion is met.

A small modification — when a coefficient would pass through zero, remove it from the active set
and change direction — makes LARS trace the exact Lasso path.

```python
from sklearn.linear_model import Lars, LarsCV, LassoLars, LassoLarsCV, LassoLarsIC

# Exact LARS solution path
lars = Lars(n_nonzero_coefs=10)  # Fit until 10 nonzero coefficients
lars.fit(X_train_sc, y_train)
print(f"Path shape (alphas, coefs): {lars.alphas_.shape}, {lars.coef_path_.shape}")
# coef_path_: (p, n_steps) — coefficients at each LARS step

# Lasso via LARS algorithm (identical solution, different algorithm)
lasso_lars = LassoLars(alpha=0.1)
lasso_lars.fit(X_train_sc, y_train)

# Lasso-LARS with CV selection
lasso_lars_cv = LassoLarsCV(cv=5, max_iter=500, n_jobs=-1)
lasso_lars_cv.fit(X_train_sc, y_train)
print(f"Best alpha (LARS-CV): {lasso_lars_cv.alpha_:.6f}")

# Information criterion selection (AIC or BIC) — much faster than CV
lasso_lars_ic = LassoLarsIC(criterion='bic')  # 'aic' or 'bic'
lasso_lars_ic.fit(X_train_sc, y_train)
print(f"Best alpha (BIC): {lasso_lars_ic.alpha_:.6f}")
```

**When to prefer LassoLars over LassoCV:**
- When $p \ll n$ and you need the exact path (not an approximation)
- When you want AIC/BIC model selection (faster than CV, good for large $n$)
- When reproducibility of exact path matters
- For small-to-medium $p$ (say $p < 10{,}000$); for large $p$, coordinate descent is faster

> 📊 **Benchmark:** For $n=1{,}000$, $p=500$: LARS path $\approx$ 3× faster than LassoCV with
> warm starts. For $n=10{,}000$, $p=500$: LassoCV catches up and becomes comparable or faster
> due to its vectorized coordinate descent implementation.

### 3.3 Coordinate Descent + Warm Starts: The Modern Workhorse

> 📜 **Paper:** Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
> generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1),
> 1–22. https://doi.org/10.18637/jss.v033.i01

**Problem solved:** LARS is elegant but becomes slow for large $n$ (the $n$-factor in its
complexity hurts). Friedman et al. showed that **cyclical coordinate descent** on the Lasso/ElasticNet
objective, with warm starts along the regularization path, is both simpler and faster in practice
for large datasets. The `glmnet` R package implemented this and became the go-to tool for
penalized regression. sklearn's `Lasso`, `LassoCV`, `ElasticNet`, `ElasticNetCV` all use this
algorithm internally.

**Key algorithmic contribution:** Pathwise coordinate descent:
1. Generate a sequence $\alpha_1 > \alpha_2 > \cdots > \alpha_K$ (log-spaced from $\alpha_{\max}$
   to $\alpha_{\min} = \epsilon \cdot \alpha_{\max}$)
2. At $\alpha_1 = \alpha_{\max}$: all coefficients are exactly zero (start here for free)
3. For each subsequent $\alpha_{k+1}$: initialize coordinates at $\hat{\boldsymbol{\beta}}(\alpha_k)$
   (the previous solution) and run coordinate descent to convergence
4. Because $\hat{\boldsymbol{\beta}}(\alpha_{k+1})$ is close to $\hat{\boldsymbol{\beta}}(\alpha_k)$,
   few iterations are needed

This is why computing 100 path values is barely slower than computing 1 value from scratch.
The total work is roughly the cost of a single OLS fit, amortized over the whole path.

> 💡 **Intuition:** Think of the regularization path as a homotopy — a continuous deformation
> from the trivially simple solution (zero vector) to the complex solution (full OLS). Warm starts
> follow this path, taking tiny steps. At each step, very little has changed, so very little
> work is needed. You are not solving $K$ independent problems; you are solving one long problem
> with $K$ waypoints.

### 3.4 ElasticNet and the Grouping Property

> 📜 **Paper:** Zou, H., & Hastie, T. (2005). Regularization and variable selection via the
> elastic net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*,
> 67(2), 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

**Problem solved:** Lasso has two critical failure modes:
1. When $p > n$: Lasso can select at most $n$ variables (the problem becomes degenerate). This
   is problematic in genomics where $p$ might be 50,000 and $n$ is 200.
2. **Correlated predictors:** If $x_1 \approx x_2$, Lasso will randomly select one and set the
   other to zero, even if both are genuinely important. The selection is unstable — different
   bootstrap samples may select $x_1$ one time and $x_2$ the next. This is not a bug; it is the
   geometry of the L1 ball in the presence of near-duplicate features.

ElasticNet's **grouping property** (Zou & Hastie, Theorem 1): If two features satisfy
$|\mathbf{x}_i^\top \mathbf{x}_j|/n \approx 1$ (nearly identical), then their ElasticNet
coefficients satisfy:

$$|\hat{\beta}_i - \hat{\beta}_j| \leq \frac{2}{\alpha(1-\rho)}\sqrt{1 - r_{ij}^2} \cdot \|\mathbf{y}\|_2$$

where $r_{ij}$ is the correlation between $x_i$ and $x_j$. As $r_{ij} \to 1$, the bound
$\to 0$ — perfectly correlated features get *identical* coefficients. The L2 penalty enforces
this grouping, distributing credit across the correlated group rather than arbitrarily
concentrating it on one member.

```python
# Demonstrating the grouping effect
from sklearn.linear_model import Lasso, ElasticNet
import numpy as np

np.random.seed(42)
n, p = 100, 10
# Create dataset with highly correlated features
X = np.random.randn(n, 3)
X = np.column_stack([X, X[:, 0] + 0.01*np.random.randn(n),   # x4 ≈ x1
                        X[:, 1] + 0.01*np.random.randn(n)])   # x5 ≈ x2
X = np.hstack([X, np.random.randn(n, 5)])  # 5 noise features
y = 2*X[:, 0] + 1.5*X[:, 1] + np.random.randn(n)

lasso = Lasso(alpha=0.1).fit(X, y)
enet  = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)

print("Lasso coefs (first 5):", lasso.coef_[:5].round(3))
# → e.g., [2.1, 0.0, 0.0, 1.3, 0.0] — picks x1, x4 (or x2, x5) arbitrarily
print("ElasticNet coefs (first 5):", enet.coef_[:5].round(3))
# → e.g., [1.3, 0.9, 0.0, 1.2, 0.8] — shares credit between x1/x4 and x2/x5
```

> ⚠️ **Pitfall — ElasticNet double shrinkage:** Zou & Hastie noted that ElasticNet applies both
> L1 and L2 shrinkage, resulting in coefficients that are more shrunk than either pure Lasso or
> pure Ridge. They recommended a correction factor of $(1 + \alpha(1-\rho))$ applied to the
> ElasticNet coefficients. sklearn does **not** apply this correction by default. If you are
> comparing ElasticNet coefficient magnitudes to OLS or standalone Ridge/Lasso estimates, be
> aware of this double-shrinkage effect.

### 3.5 BayesianRidge — Self-Tuning Regularization via Evidence Maximization

> 📜 **Papers:** MacKay, D. J. C. (1992). Bayesian interpolation. *Neural Computation*, 4(3),
> 415–447. and Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine.
> *JMLR*, 1, 211–244.

**Problem solved:** Ridge and Lasso require cross-validation to select $\alpha$ — expensive and
noisy with small datasets. Bayesian Ridge places a Gaussian prior on the weights and a Gamma
prior on the regularization strength, then learns both via **evidence maximization** (also called
Type II maximum likelihood or Empirical Bayes):

$$\boldsymbol{\beta} \sim \mathcal{N}\!\left(\mathbf{0}, \lambda^{-1}\mathbf{I}\right), \quad \varepsilon \sim \mathcal{N}\!\left(0, \alpha^{-1}\right)$$

$$\text{precision of noise: } \alpha, \quad \text{precision of weights: } \lambda$$

Marginalizing over $\boldsymbol{\beta}$ gives the **marginal likelihood** (evidence):

$$\log p(\mathbf{y} | \alpha, \lambda) = \text{const} - \frac{\alpha}{2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}^*\|^2 - \frac{\lambda}{2}\|\boldsymbol{\beta}^*\|^2 - \frac{1}{2}\log|\mathbf{S}^{-1}|$$

where $\mathbf{S}^{-1} = \alpha\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ is the posterior
precision. Updating $\alpha$ and $\lambda$ to maximize this evidence via EM-style updates gives
self-tuning regularization — *no cross-validation needed*.

**Posterior inference:** The posterior over $\boldsymbol{\beta}$ is Gaussian:

$$p(\boldsymbol{\beta} | \mathbf{y}, \alpha, \lambda) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

$$\boldsymbol{\mu} = \alpha \boldsymbol{\Sigma} \mathbf{X}^\top \mathbf{y}, \quad \boldsymbol{\Sigma} = (\alpha\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}$$

This gives a full posterior distribution over coefficients — not just point estimates but
*uncertainty estimates*. Predictive variance for a new point $\mathbf{x}_*$:

$$\text{Var}(\hat{y}_*) = \frac{1}{\alpha} + \mathbf{x}_*^\top \boldsymbol{\Sigma} \mathbf{x}_*$$

```python
from sklearn.linear_model import BayesianRidge
import numpy as np

br = BayesianRidge(
    max_iter=300,       # EM iterations (renamed from n_iter in sklearn 1.3)
    tol=1e-3,
    alpha_1=1e-6,       # Gamma prior on noise precision: shape
    alpha_2=1e-6,       # Gamma prior on noise precision: rate
    lambda_1=1e-6,      # Gamma prior on weight precision: shape
    lambda_2=1e-6,      # Gamma prior on weight precision: rate
    compute_score=True, # Track log marginal likelihood during EM
)
br.fit(X_train_sc, y_train)

# Point prediction + uncertainty
y_pred, y_std = br.predict(X_test_sc, return_std=True)
# y_std: predictive standard deviation per sample

# Posterior summary
print(f"Estimated noise precision (alpha): {br.alpha_:.4f}")
print(f"Estimated weight precision (lambda): {br.lambda_:.4f}")
print(f"Effective regularization alpha/lambda: {br.lambda_/br.alpha_:.4f}")
print(f"Posterior covariance shape: {br.sigma_.shape}")  # (p, p)

# 95% prediction interval
import scipy.stats
z = scipy.stats.norm.ppf(0.975)
lower = y_pred - z * y_std
upper = y_pred + z * y_std
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"95% PI coverage: {coverage:.2%}")
```

> 🔧 **In practice:** BayesianRidge is excellent for small datasets ($n < 500$) where cross-
> validation is noisy, and for cases where you need calibrated uncertainty estimates (e.g., for
> downstream decision-making under uncertainty). For large datasets, sklearn Ridge/RidgeCV is
> faster and produces essentially the same point predictions.

> ⚠️ **Pitfall — `n_iter` deprecation:** Before sklearn 1.3, BayesianRidge used `n_iter=300`.
> In sklearn 1.3+, this was renamed to `max_iter=300`. Using `n_iter` in 1.5.x will raise a
> deprecation warning. Always use `max_iter`.

### 3.6 ARDRegression — Per-Feature Relevance (Sparse Bayesian Learning)

**Problem solved:** BayesianRidge uses a *single* regularization precision $\lambda$ shared
across all features. ARD (Automatic Relevance Determination) assigns a *separate* precision
$\lambda_j$ to each coefficient $\beta_j$:

$$\beta_j \sim \mathcal{N}(0, \lambda_j^{-1})$$

During EM, features whose $\lambda_j$ grows very large (their prior contracts to a spike at zero)
are effectively pruned from the model. This gives a *sparsity mechanism* analogous to Lasso, but
from a Bayesian perspective without the non-differentiability issues.

```python
from sklearn.linear_model import ARDRegression

ard = ARDRegression(
    max_iter=300,
    tol=1e-3,
    threshold_lambda=1e4,  # Features with lambda_ > this are effectively pruned
    alpha_1=1e-6, alpha_2=1e-6,
    lambda_1=1e-6, lambda_2=1e-6,
    compute_score=True,
)
ard.fit(X_train_sc, y_train)

# See which features survived
active_mask = ard.lambda_ < 1e4
print(f"ARD active features: {active_mask.sum()} / {len(active_mask)}")
print(f"ARD lambda per feature (log): {np.log10(ard.lambda_).round(1)}")

y_pred_ard, y_std_ard = ard.predict(X_test_sc, return_std=True)
```

**ARD vs. Lasso:** ARD is slower (updates $p$ hyperparameters per EM step vs. $O(p)$ coordinate
descent steps per cycle, but EM steps are more expensive) and rarely outperforms Lasso + CV for
point prediction. Its advantage is genuine Bayesian uncertainty quantification and the ability to
prune irrelevant features in a probabilistically grounded way.

### 3.7 HuberRegressor — Robust Linear Regression

> 📜 **Paper:** Huber, P. J. (1964). Robust estimation of a location parameter. *The Annals of
> Mathematical Statistics*, 35(1), 73–101.

**Problem solved:** OLS squares the residuals — a single large outlier can devastate the fit.
The **Huber loss** is a hybrid: quadratic for small residuals (like OLS, efficient under
normality) but linear for large residuals (like MAE, robust to outliers):

$$H_\epsilon(r) = \begin{cases} r^2 & \text{if } |r| \leq \epsilon \\ 2\epsilon|r| - \epsilon^2 & \text{if } |r| > \epsilon \end{cases}$$

The transition point $\epsilon$ (sklearn: `epsilon` parameter) controls robustness. At
$\epsilon = 1.35$, the Huber estimator achieves ~95% asymptotic efficiency relative to OLS on
clean Gaussian data — you pay only 5% efficiency to get substantial outlier resistance.

```python
from sklearn.linear_model import HuberRegressor
import numpy as np

# Introduce outliers
X_noisy = X_train_sc.copy()
y_noisy = y_train.copy()
outlier_idx = np.random.choice(len(y_noisy), size=20, replace=False)
y_noisy[outlier_idx] += 500  # Extreme outliers

# Compare OLS vs. Huber
ols_clean = LinearRegression().fit(X_train_sc, y_train)
ols_noisy = LinearRegression().fit(X_noisy, y_noisy)
huber_noisy = HuberRegressor(epsilon=1.35, max_iter=200, alpha=0.0001).fit(X_noisy, y_noisy)

from sklearn.metrics import mean_absolute_error
y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
print(f"OLS (clean) MAE:  {mean_absolute_error(y_test_array, ols_clean.predict(X_test_sc)):.2f}")
print(f"OLS (noisy) MAE:  {mean_absolute_error(y_test_array, ols_noisy.predict(X_test_sc)):.2f}")
print(f"Huber (noisy) MAE:{mean_absolute_error(y_test_array, huber_noisy.predict(X_test_sc)):.2f}")

# Identify outliers detected by Huber
print(f"Outliers detected: {huber_noisy.outliers_.sum()}")
```

### 3.8 QuantileRegressor — Distributional Regression

**Problem solved:** OLS estimates the conditional *mean* $\mathbb{E}[y|\mathbf{x}]$. For
heteroskedastic data (variance changes with $\mathbf{x}$), asymmetric distributions, or when you
need prediction intervals, the mean is not what you want.

QuantileRegressor (added in sklearn 1.0) estimates any conditional quantile $Q_\tau(y|\mathbf{x})$
by minimizing the **pinball loss**:

$$\mathcal{L}_\tau(r) = \begin{cases} \tau \cdot r & r \geq 0 \\ (1 - \tau) \cdot (-r) & r < 0 \end{cases}$$

At $\tau = 0.5$ this is the MAE loss (median regression). This is solved as a linear program.

```python
from sklearn.linear_model import QuantileRegressor  # Added sklearn 1.0+

# 90% prediction interval: fit 5th and 95th percentiles
q_low  = QuantileRegressor(quantile=0.05, alpha=0.01, solver='highs').fit(X_train_sc, y_train)
q_med  = QuantileRegressor(quantile=0.50, alpha=0.01, solver='highs').fit(X_train_sc, y_train)
q_high = QuantileRegressor(quantile=0.95, alpha=0.01, solver='highs').fit(X_train_sc, y_train)

y_lo   = q_low.predict(X_test_sc)
y_hi   = q_high.predict(X_test_sc)
coverage = np.mean((y_test_array >= y_lo) & (y_test_array <= y_hi))
avg_width = np.mean(y_hi - y_lo)
print(f"90% PI coverage: {coverage:.2%}  avg width: {avg_width:.2f}")
# Good calibration: coverage should be close to 90%
```

> ⚠️ **Pitfall — alpha vs. quantile confusion:** In `QuantileRegressor`, `alpha` is the
> **L1 regularization strength** (like Lasso's `alpha`). The quantile level is the `quantile`
> parameter (default 0.5). Do not confuse these — `alpha=0.95` does NOT mean the 95th percentile.

### 3.9 RANSACRegressor and TheilSenRegressor — High-Breakdown-Point Methods

For truly adversarial outliers (not just heavy-tailed noise, but actual data contamination),
HuberRegressor may not be enough. Two high-breakdown-point estimators are available:

**RANSACRegressor** (Random Sample Consensus): Fits the model on a random subset of points,
identifies inliers, refits on inliers. Can tolerate up to 50% outliers.

**TheilSenRegressor**: Computes the median of slopes over all subpopulation pairs. Breakdown
point ~29.3% (robust to just under a third of the data being corrupted). More statistically
efficient than RANSAC but computationally expensive: $O\binom{n}{p}$.

```python
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor

ransac = RANSACRegressor(
    estimator=LinearRegression(),
    min_samples=0.5,          # Use 50% of data for each random subset
    residual_threshold=None,  # Auto: 1.4826 * MAD of absolute residuals
    max_trials=100,
    random_state=42,
)
ransac.fit(X_noisy, y_noisy)
print(f"RANSAC inliers: {ransac.inlier_mask_.sum()} / {len(y_noisy)}")

ts = TheilSenRegressor(max_subpopulation=10000, n_jobs=-1, random_state=42)
ts.fit(X_noisy, y_noisy)
```

### 3.10 MultiTaskLasso — Joint Sparsity Across Multiple Targets

**Problem solved:** When predicting $K$ related targets (e.g., scores on $K$ test items from the
same student, or gene expression at $K$ time points), fitting $K$ separate Lassos may select
different features for each target — losing the statistical power that comes from sharing
information across tasks.

MultiTaskLasso uses the $\ell_{2,1}$ norm (row-sparsity):

$$\min_{\mathbf{W}} \frac{1}{2n}\|\mathbf{Y} - \mathbf{X}\mathbf{W}^\top\|_F^2 + \alpha \|\mathbf{W}\|_{2,1}$$

where $\|\mathbf{W}\|_{2,1} = \sum_{j=1}^p \|\mathbf{w}_{j,\cdot}\|_2$ (the L2 norm of each row,
summed over all rows). This encourages **entire rows** of $\mathbf{W}$ (one row per feature,
containing that feature's coefficient for all $K$ targets) to be zero simultaneously. All tasks
share the same sparse support.

```python
from sklearn.linear_model import MultiTaskLasso, MultiTaskLassoCV

# Y_train: (n_samples, n_targets)
# Create a synthetic multi-target example
Y_train_mt = np.column_stack([y_train, y_train + 10*np.random.randn(len(y_train))])
Y_test_mt  = np.column_stack([y_test,  y_test  + 10*np.random.randn(len(y_test))])

mt_lasso = MultiTaskLasso(alpha=1.0, fit_intercept=True, max_iter=1000)
mt_lasso.fit(X_train_sc, Y_train_mt)
print(f"MTLasso coef shape: {mt_lasso.coef_.shape}")  # (n_targets, n_features)

# Rows that are entirely zero = jointly selected to be inactive
active_features = (np.abs(mt_lasso.coef_) > 1e-10).any(axis=0)
print(f"Jointly selected features: {active_features.sum()} / {X_train_sc.shape[1]}")
```

### 3.11 Adaptive Lasso — The Oracle Property

> 📜 **Paper:** Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the
> American Statistical Association*, 101(476), 1418–1429.

**Problem solved:** The standard Lasso is a biased estimator and does not have the *oracle
property* in general: it cannot simultaneously achieve correct variable selection (sign
consistency) and optimal estimation rates. Zou (2006) introduced the **adaptive Lasso**, which
uses feature-specific weights derived from an initial OLS or Ridge estimate:

$$\hat{\boldsymbol{\beta}}_{\text{AdaLasso}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \sum_{j=1}^p \frac{|\beta_j|}{|\hat{\beta}_j^{(0)}|^\gamma} \right\}$$

where $\hat{\boldsymbol{\beta}}^{(0)}$ is an initial (e.g., Ridge) estimate and $\gamma > 0$
controls adaptivity (often $\gamma = 1$ or $\gamma = 2$). Features that are already small
(likely zero in truth) get larger penalties $1/|\hat{\beta}_j^{(0)}|^\gamma$ — the penalty is
*adaptive* to the estimated importance of each feature.

**Not natively in sklearn**, but straightforward to implement by manually setting penalty weights:

```python
from sklearn.linear_model import Lasso, RidgeCV
import numpy as np

# Step 1: Initial Ridge estimate (or OLS if p < n)
init_ridge = RidgeCV(alphas=np.logspace(-3, 3, 100)).fit(X_train_sc, y_train)
beta_init = np.abs(init_ridge.coef_)

# Step 2: Adaptive weights (protect against division by zero)
gamma = 1.0
weights = 1.0 / (beta_init + 1e-6)**gamma

# Step 3: Scale features by weights (equiv. to adaptive penalty)
X_weighted = X_train_sc / weights[np.newaxis, :]  # (n, p) / (1, p)

# Step 4: Fit standard Lasso on weighted features
from sklearn.linear_model import LassoCV
ada_lasso_cv = LassoCV(cv=5, max_iter=10000, n_jobs=-1).fit(X_weighted, y_train)

# Step 5: Recover original-scale coefficients
ada_coef = ada_lasso_cv.coef_ / weights
print(f"Adaptive Lasso nonzero: {(ada_coef != 0).sum()}")
```

> 📜 **Also see:** Group Lasso (Yuan & Lin, 2006) for user-specified groups; Fused Lasso
> (Tibshirani et al., 2005) for structured sparsity with penalties on differences of adjacent
> coefficients. Neither is natively in sklearn; implementations are available in the `celer`
> and `group-lasso` Python packages.

---

## §4 — Hyperparameters: The Complete Guide

### 4.1 The Master Parameter: `alpha` (Regularization Strength)

`alpha` ($\alpha$, also written $\lambda$ in some texts) is the single most important
hyperparameter in the regularized linear model family. It controls the strength of the penalty
relative to the data-fit term, and therefore the entire bias-variance tradeoff.

**What it controls:** The relative weight of the regularization penalty vs. the squared loss.
Larger $\alpha$ → more regularization → smaller coefficients → higher bias, lower variance.
Smaller $\alpha$ → less regularization → approaches OLS → lower bias, higher variance.

**The alpha landscape across models:**

| $\alpha \to 0$ | $\alpha = 1$ (default) | $\alpha \to \infty$ |
|---|---|---|
| Ridge → OLS (unbiased, possibly high-variance) | Moderate shrinkage | All coefs → 0 (constant prediction = $\bar{y}$) |
| Lasso → OLS (dense model) | Some sparsity (data-dependent) | All coefs → 0 (null model) |

> ⚠️ **Pitfall — The default `alpha=1.0` is almost never right:** The default `alpha=1.0` in
> sklearn Ridge, Lasso, and ElasticNet is a placeholder, not a tuned value. On standardized data,
> `alpha=1.0` may massively over-regularize or under-regularize depending on the problem scale.
> *Always* use the CV variants (`RidgeCV`, `LassoCV`, `ElasticNetCV`) to select `alpha`.

**The $\alpha_{\max}$ formula for Lasso:** The smallest value of $\alpha$ that sets all
coefficients to zero:

$$\alpha_{\max} = \frac{1}{n}\|\mathbf{X}^\top \mathbf{y}\|_\infty = \frac{1}{n}\max_j |\mathbf{x}_j^\top \mathbf{y}|$$

`LassoCV` with `alphas=None` auto-generates a grid from $\alpha_{\max} \cdot \epsilon$ to
$\alpha_{\max}$. You almost never need to specify the grid manually.

**Interpretation across scales:** Because sklearn uses the $\frac{1}{2n}$ factor in the Lasso
objective, a dataset with 10,000 samples and another with 100 samples require different $\alpha$
values for equivalent regularization strength. When comparing across datasets of different sizes,
keep in mind that the effective penalty strength scales as $n \cdot \alpha$.

**Tuning strategy:**

```python
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
import numpy as np

# For Ridge: use GCV-based LOO (cv=None)
ridge_cv = RidgeCV(alphas=np.logspace(-4, 6, 300), cv=None)
ridge_cv.fit(X_train_sc, y_train)

# For Lasso: auto-generate grid, 5-fold CV
lasso_cv = LassoCV(
    n_alphas=200,     # More points → smoother CV curve
    cv=5,
    max_iter=50000,   # Generous; avoid ConvergenceWarning
    n_jobs=-1,
    random_state=42,
)
lasso_cv.fit(X_train_sc, y_train)

# For ElasticNet: grid over both alpha and l1_ratio
enet_cv = ElasticNetCV(
    l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    n_alphas=200,
    cv=5,
    max_iter=50000,
    n_jobs=-1,
    random_state=42,
)
enet_cv.fit(X_train_sc, y_train)
```

### 4.2 `l1_ratio` (ElasticNet only)

`l1_ratio` ($\rho$) controls the mix of L1 and L2 regularization in ElasticNet.

- `l1_ratio=1.0` → pure Lasso (exact zeros, arbitrary selection from correlated groups)
- `l1_ratio=0.0` → pure Ridge (no zeros, but use `Ridge` class directly — `ElasticNet(l1_ratio=0)` is numerically inferior)
- `l1_ratio=0.5` → equal L1 and L2 contribution (default)
- `l1_ratio=0.9` → mostly Lasso with mild grouping effect

**How it interacts with alpha:** The L1 effective strength is $\alpha\rho$ and the L2 effective
strength is $\alpha(1-\rho)/2$. Reducing `l1_ratio` reduces sparsity (fewer exact zeros) but
increases stability in the presence of correlated features.

**Tuning strategy:** Always test a range including 1.0 (pure Lasso). Let the data decide whether
the L2 component actually helps. Values to test: `[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]`.
Test this via `ElasticNetCV(l1_ratio=[...], ...)` — it runs a 2D grid over `alpha` × `l1_ratio`
simultaneously.

> 🔧 **In practice:** If `ElasticNetCV` selects `l1_ratio=1.0`, the data are telling you that
> pure Lasso is better and the L2 component is not helping. If it selects a lower value like
> `l1_ratio=0.5`, you likely have correlated features and ElasticNet's grouping effect is
> genuinely useful.

### 4.3 `fit_intercept`

**Default:** `True` for all models.

**What it controls:** Whether to compute an intercept $\beta_0$ (bias term). When `True`, sklearn
internally centers $\mathbf{X}$ (subtracts column means) and $\mathbf{y}$ (subtracts the mean),
fits the regularized model on centered data, then adjusts the intercept as
$\hat{\beta}_0 = \bar{y} - \bar{\mathbf{x}}^\top\hat{\boldsymbol{\beta}}$.

**Critical implication:** The intercept is *not penalized* — it is computed after penalization of
the centered model. This is correct behavior: you should not regularize the intercept, as doing
so would bias predictions toward zero rather than toward $\bar{y}$.

**When to set `False`:** Only when data is already centered (zero mean features and zero mean
target), or for special-purpose uses like fitting without intercept for theoretical reasons.

### 4.4 `solver` (Ridge)

The `solver` parameter for Ridge determines the numerical algorithm used to solve the linear
system $(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}$.

| Solver | Best for | Dense/Sparse | Notes |
|---|---|---|---|
| `'auto'` | Default (auto-selects) | Both | Picks `'cholesky'` for dense, `'sparse_cg'` for sparse |
| `'svd'` | Maximum numerical stability | Dense only | Computes full SVD; slowest but most stable; also gives shrinkage factors |
| `'cholesky'` | Small-to-medium dense $p$ | Dense | Fast; fails if $\mathbf{X}^\top\mathbf{X}+\alpha\mathbf{I}$ is numerically singular |
| `'lsqr'` | Large sparse data | Both | Regularized least-squares iterative; efficient |
| `'sparse_cg'` | Very large sparse data | Both | Conjugate gradient; good when $p$ is huge and sparse |
| `'sag'` | Very large $n$, few features | Dense | Stochastic gradient; needs feature scaling |
| `'saga'` | Like `sag`, $\ell_1$ option too | Dense | Use for non-smooth penalties; needs scaling |
| `'lbfgs'` | Only when `positive=True` | Dense | L-BFGS-B; the only solver supporting non-negative constraint |

> 🔧 **In practice:** Leave `solver='auto'` unless: (1) you need `positive=True` coefficients
> (use `'lbfgs'`), (2) you have very large sparse data (use `'sparse_cg'` or `'sag'`), or (3)
> you are debugging numerical issues (use `'svd'` for maximum stability).

### 4.5 `max_iter` (Lasso, ElasticNet, BayesianRidge)

**Default:** `1000` for Lasso/ElasticNet, `300` for BayesianRidge.

**What it controls:** The maximum number of coordinate descent iterations (Lasso/ElasticNet) or
EM iterations (BayesianRidge) before the solver stops, regardless of convergence.

**The ConvergenceWarning:** If the solver exits due to reaching `max_iter` before meeting the
convergence tolerance `tol`, sklearn issues:
```
ConvergenceWarning: Objective did not converge. The max_iter was reached.
```

This means your model may have suboptimal coefficients. **Do not ignore this warning.**

**Fix:**
```python
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

# Check for convergence explicitly
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    lasso = Lasso(alpha=0.01, max_iter=1000).fit(X_train_sc, y_train)

for w in caught_warnings:
    if issubclass(w.category, ConvergenceWarning):
        print("⚠️  Not converged! Increase max_iter.")
        # Fix: try max_iter=10000 or max_iter=50000
```

**When does convergence take many iterations?**
- Features not standardized (high condition number → slow coordinate descent)
- Very small `alpha` (many active features → many nonzero coordinates to update)
- Very small `tol` (convergence criterion very tight)
- Very large $p$ (each full cycle over all coordinates is expensive)

> 🏆 **Best practice:** Start with `max_iter=10000` for Lasso/ElasticNet. On standardized
> data, 10,000 iterations is almost always sufficient. If still not converged, go to 50,000.

### 4.6 `tol` (Convergence Tolerance)

**Default:** `1e-4` for Lasso/ElasticNet, `1e-3` for BayesianRidge.

**What it controls:** Coordinate descent stops when the maximum coefficient change in a full
cycle is less than `tol`:

$$\max_j |\hat{\beta}_j^{(t+1)} - \hat{\beta}_j^{(t)}| < \text{tol}$$

**Guidance:** The default `1e-4` is sufficient for most use cases. Tighter tolerances
(`1e-6`, `1e-7`) may be needed for: stability selection (bootstrapped feature selection),
computing solution paths for visualization, or situations where small coefficient changes matter
for downstream decision-making.

### 4.7 `warm_start`

**Default:** `False` for Lasso, ElasticNet, Ridge.

**What it controls:** When `True`, the model's current `coef_` is used as the starting point
for the next `fit()` call, rather than starting from scratch. This is the engine that makes
path algorithms fast internally — but you can also use it manually for incremental training.

```python
from sklearn.linear_model import Lasso

# Manual regularization path with warm starts
lasso = Lasso(warm_start=True, max_iter=10000)
coefs = []
alphas_manual = np.logspace(2, -4, 100)

for alpha in alphas_manual:
    lasso.set_params(alpha=alpha)
    lasso.fit(X_train_sc, y_train)
    coefs.append(lasso.coef_.copy())

coefs = np.array(coefs)  # Shape: (100, n_features)
```

### 4.8 `selection` (Lasso/ElasticNet)

**Default:** `'cyclic'`

**Options:** `'cyclic'` (update features in order 1, 2, ..., p, 1, 2, ...) or `'random'`
(sample a random feature to update at each step).

**When random selection helps:** For models with many features ($p > 1000$), random coordinate
selection can converge faster because it avoids the pattern where cyclic selection keeps revisiting
features that don't need updating. Set `random_state` for reproducibility when using `'random'`.

```python
# For large p, random coordinate selection can be faster
lasso_random = Lasso(alpha=0.01, selection='random', random_state=42, max_iter=10000)
```

### 4.9 `positive` (Ridge, Lasso, ElasticNet, LinearRegression)

**Default:** `False`

**What it controls:** Constrains all coefficients to be non-negative ($\hat{\beta}_j \geq 0$).
Useful in problems with known non-negative structure: word counts (NMF-style), physical
concentration measurements, dose-response models.

**Implementation details:**
- For `Ridge`: only compatible with `solver='lbfgs'`
- For `LinearRegression`: uses NNLS (Non-Negative Least Squares, scipy.optimize.nnls)
- For `Lasso`/`ElasticNet`: coordinate descent with a one-sided constraint (threshold at 0)

### 4.10 Complete Tuning Playbook

| Model | Parameter | Typical Range | Tuning Strategy | Too High | Too Low |
|---|---|---|---|---|---|
| **Ridge** | `alpha` | [1e-6, 1e6] log scale | `RidgeCV(alphas=np.logspace(-6, 6, 200), cv=None)` | All coefs → 0 (underfitting) | OLS (multicollinearity, overfit) |
| **Lasso** | `alpha` | [auto from data] | `LassoCV(n_alphas=100, cv=5, max_iter=10000)` | Null model (all zeros) | Dense OLS (no selection) |
| **ElasticNet** | `alpha` | [auto from data] | `ElasticNetCV(l1_ratio=[...], n_alphas=100, cv=5)` | Null model | Dense model |
| **ElasticNet** | `l1_ratio` | [0.1, 1.0] | Grid `[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]` | More L2-like (grouping) | More L1-like (sparsity) |
| **Lasso** | `max_iter` | [1000, 100000] | Start 10000; increase on ConvergenceWarning | N/A | ConvergenceWarning |
| **HuberRegressor** | `epsilon` | [1.0, 5.0] | Start 1.35; lower for more outliers | Less robust | Lower efficiency on clean data |
| **BayesianRidge** | `max_iter` | [100, 1000] | Check `scores_` is flat; usually 300 is enough | N/A | Premature convergence |
| **QuantileRegressor** | `quantile` | (0, 1) | Problem-specific (0.5 = median) | N/A (not a regularization param) | N/A |
| **QuantileRegressor** | `alpha` | [0, 1.0] | Start at 0 (no reg) or 0.1 | Over-regularized | Overfit (sparse data) |

---

## §5 — Strengths

### 5.1 Interpretability: Coefficients as Causal Language

The prediction of a linear model is $\hat{y} = \beta_0 + \sum_j \beta_j x_j$. Each coefficient
$\beta_j$ has a direct, unambiguous interpretation: a one-unit increase in $x_j$, holding all
other features constant, is associated with a change of $\beta_j$ units in the predicted target.
This is the **ceteris paribus** interpretation familiar from economics and epidemiology.

**Why this is a genuine strength and not just simplicity:** In high-stakes applications —
clinical guidelines, legal decisions, financial reporting, regulatory submissions — you often
*must* explain your model. A Ridge regression coefficient can be shown to a cardiologist and
discussed meaningfully. A random forest's feature importances cannot. The interpretability of
linear models is not a limitation to overcome; it is a design requirement that linear models
meet, and tree ensembles do not.

**The Lasso enhancement:** Lasso's variable selection makes the interpretation even cleaner. A
Lasso model that selects 5 features out of 50 tells a compelling story: "these 5 factors matter,
the other 45 do not (or are redundant with the 5)."

### 5.2 Computational Efficiency: Scales Where Others Cannot

Linear models are among the cheapest supervised learning algorithms at both training and inference
time. Ridge training costs $O(np^2 + p^3)$ — with the closed-form solution, there is no
iterative optimization at all. Lasso and ElasticNet use coordinate descent, which is empirically
very fast (typically $< 1$ second for $n = 10{,}000$, $p = 1{,}000$ on a laptop).

**Inference:** $O(p)$ per sample — literally a dot product and addition. At inference time, there
is nothing faster in the supervised learning world. A linear model deployed in a microservice can
score millions of records per second on a single CPU core.

**Memory:** Storing a linear model requires only $p + 1$ floats (the coefficients and intercept).
For $p = 10{,}000$: 80 kilobytes. A gradient-boosted tree ensemble may require gigabytes for the
same $p$.

> 📊 **Benchmark:** On the 20-newsgroups dataset ($n = 18{,}000$ documents, $p \approx 130{,}000$
> sparse TF-IDF features), `SGDClassifier` (a stochastic linear model) trains in ~2 seconds.
> A kernel SVM on the same data takes ~10 minutes. The entire point of linear models for
> high-dimensional text is that their $O(p)$ inference is untouchable.

### 5.3 High Dimensionality: When $p \gg n$, Linear Models Still Work

For very high-dimensional problems where $p > n$ — genomics (50,000 SNPs, 200 patients),
text classification (100,000 vocabulary terms, 1,000 documents), NLP features — tree-based
models overfit catastrophically, and even SVMs with kernels become computationally infeasible.

Ridge (which has a closed-form solution even when $p > n$, via the dual formulation) and Lasso/
ElasticNet (which impose sparsity and effectively use only a small active subset of features)
are specifically designed for this regime.

**The dual Ridge solution for $p > n$:** Instead of the primal $(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ (which involves an $p \times p$ matrix), use the dual:

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top + \alpha\mathbf{I})^{-1}\mathbf{y}$$

This involves an $n \times n$ matrix — far cheaper when $p \gg n$.

### 5.4 Convexity: Global Optimum Guaranteed

All regularized linear models (Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor,
QuantileRegressor) have *convex* objective functions. This means:
- There is no local minimum problem
- The solution is globally optimal
- The optimization is reliable and reproducible across random restarts
- Convergence is guaranteed by coordinate descent theory (for separable regularization)

This is a dramatic contrast with deep neural networks, which have highly non-convex loss
landscapes and no global optimality guarantees.

### 5.5 Natural Handling of Multicollinearity (Ridge/ElasticNet)

When features are highly correlated, OLS coefficients are unstable and have huge standard errors.
Ridge regularization *directly addresses this* by the SVD mechanism described in §2.2: it
identifies near-collinear directions in feature space and shrinks them, without needing any
user intervention. You do not need to diagnose multicollinearity, drop features, or orthogonalize.
Just use Ridge.

### 5.6 Automatic Variable Selection (Lasso/ElasticNet)

Lasso and ElasticNet produce *sparse* coefficient vectors. You can use them directly as feature
selection tools for downstream modeling:

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(cv=5, n_alphas=100, max_iter=10000, n_jobs=-1).fit(X_train_sc, y_train)
selector = SelectFromModel(lasso_cv, prefit=True, threshold=1e-5)
X_train_selected = selector.transform(X_train_sc)
X_test_selected  = selector.transform(X_test_sc)
print(f"Selected {X_train_selected.shape[1]} features from {X_train_sc.shape[1]}")
```

The resulting selection is *regularized* (unlike stepwise selection) and *simultaneous*
(all features selected at once, not sequentially). This has important advantages for
stability and multiple comparisons control.

### 5.7 Statistical Inference with statsmodels

Linear models are the *only* mainstream ML algorithm family where you get rigorously valid
statistical inference — p-values, confidence intervals, F-tests, AIC/BIC — without any
asymptotic approximations beyond standard regression theory. For research, clinical reports,
economic analyses, and regulatory submissions, this is not optional.

### 5.8 No Feature Scaling Required for OLS (But Required for Regularized Models)

Pure OLS is scale-invariant: multiplying a feature by 2 multiplies its coefficient by 0.5,
leaving predictions identical. For regularized models, scaling is required (the penalty treats
all features equally in coefficient space). But the scaling requirement is a *feature*, not a
bug — it forces you to think about meaningful units for your features, and `StandardScaler` in
a `Pipeline` makes this trivially easy.

---

## §6 — Weaknesses & Failure Modes

### 6.1 Linearity Assumption: The Fundamental Constraint

**The failure:** Linear models assume $\mathbb{E}[y|\mathbf{x}] = \mathbf{x}^\top\boldsymbol{\beta}$.
When the true relationship is nonlinear — quadratic, exponential, piecewise, with interactions —
a linear model will have systematic residual patterns regardless of how much data you have.

**Mechanism:** No amount of regularization or data can make a linear model capture a genuinely
nonlinear signal. The model class is simply too restrictive. Adding more features (Ridge with
many features) does not help; the features themselves are linearly combined.

**Detection:**
```python
import matplotlib.pyplot as plt

y_pred_ols = ols.predict(X_test_sc)
residuals = y_test - y_pred_ols

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred_ols, residuals, alpha=0.5)
ax.axhline(0, color='red', lw=2)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs. Fitted — Look for Patterns')
# A U-shape, funnel, or systematic curve → nonlinearity
```

**Mitigation:**
- Feature engineering: add polynomial terms, log-transforms, interaction terms
- Use a nonlinear model (gradient boosting, neural networks) if the complexity is justified
- Generalized additive models (GAMs via `pygam` or `mgcv`) — nonlinear, but still interpretable

### 6.2 Lasso Instability with Correlated Features

**The failure:** When $x_1 \approx x_2$ (highly correlated predictors), Lasso may select $x_1$
in one bootstrap sample and $x_2$ in another, with near-equal CV error for both. The selection
is *statistically unstable* — small perturbations in the data flip the selection. This is not
a numerical bug; it is the mathematical consequence of the L1 geometry.

**Detection:** Run **stability selection** — bootstrap Lasso 100 times and count how often each
feature is selected. Features selected in less than 50% of bootstrap runs are unstable.

```python
from sklearn.utils import resample
import numpy as np

n_bootstrap = 100
selection_freq = np.zeros(X_train_sc.shape[1])

for _ in range(n_bootstrap):
    X_boot, y_boot = resample(X_train_sc, y_train, random_state=None)
    lasso_boot = LassoCV(cv=3, n_alphas=50, max_iter=5000, n_jobs=1).fit(X_boot, y_boot)
    selection_freq += (lasso_boot.coef_ != 0).astype(float)

selection_freq /= n_bootstrap

print("Stable features (selected >70% of runs):")
stable = selection_freq > 0.7
for i, f in enumerate(feature_names):
    if stable[i]:
        print(f"  {f}: {selection_freq[i]:.0%}")
```

**Mitigation:** Use ElasticNet (grouping effect) or stability selection to identify reliably
selected features.

### 6.3 The $\alpha$ Sensitivity Problem

**The failure:** Linear model performance is extremely sensitive to the choice of $\alpha$.
On many datasets, a factor-of-10 change in $\alpha$ changes the model from well-regularized
to catastrophically underfit or overfit. The "default" `alpha=1.0` is almost never correct.

**Mechanism:** The optimal $\alpha$ depends on the signal-to-noise ratio, the number of active
features, and the feature correlations — none of which are known beforehand.

**Mitigation:** Always use CV variants. Never use bare `Lasso(alpha=1.0)` in production.

### 6.4 Collinear Features and Ridge Coefficient Interpretability

**The failure:** When two features $x_1$ and $x_2$ are perfectly collinear ($x_1 = cx_2$), Ridge
will distribute the coefficient arbitrarily between them (subject to the L2 constraint). The
*sum* of their effects is correct, but the individual coefficients are not uniquely meaningful.

**Detection:** Compute VIF (Variance Inflation Factor):

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def compute_vif(X_df):
    """Compute VIF for each feature in a DataFrame."""
    vif_data = pd.DataFrame({
        'feature': X_df.columns,
        'VIF': [variance_inflation_factor(X_df.values, i)
                for i in range(X_df.shape[1])]
    })
    return vif_data.sort_values('VIF', ascending=False)

# Rules of thumb:
# VIF = 1: no collinearity
# 1 < VIF < 5: moderate (acceptable)
# 5 < VIF < 10: high (investigate)
# VIF > 10: severe (Ridge/ElasticNet strongly recommended; coefficients not interpretable as-is)

vif_df = compute_vif(pd.DataFrame(X_train_sc, columns=feature_names))
print(vif_df)
```

**Mitigation:** Use Ridge (for stability, not interpretability of individual coefficients), or
perform PCA first (orthogonal features → interpretable Ridge coefficients).

### 6.5 Outliers Destroy OLS (And Can Damage Ridge/Lasso)

**The failure:** The squared loss $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$ assigns
quadratically increasing cost to large residuals. A single observation with $|y_i - \hat{y}_i| = 10$
contributes 100 times as much to the loss as one with $|y_i - \hat{y}_i| = 1$. Outliers in
$\mathbf{y}$ (vertical outliers) can dominate the regression. Ridge and Lasso, using the same
squared loss, inherit this sensitivity.

**Detection:**
- Standardized residuals: flag observations where $|e_i^*| > 2.5$
- Cook's distance: identifies influential observations
- `HuberRegressor.outliers_`: direct Boolean mask of detected outliers

**Mitigation:** Use `HuberRegressor` (moderate robustness), `RANSACRegressor` (up to 50%
outliers), or `TheilSenRegressor` (up to 29.3% corruption). Or: carefully examine and clean
outliers before fitting.

### 6.6 No Uncertainty Quantification (OLS/Ridge/Lasso/ElasticNet)

**The failure:** sklearn's `Ridge`, `Lasso`, and `ElasticNet` return only point predictions.
They do not provide prediction intervals or confidence intervals. For many applications
(medical diagnosis, financial risk, autonomous systems), a prediction without uncertainty is
not actionable.

**Mitigation:**
- Use `BayesianRidge` for posterior predictive uncertainty
- Use `statsmodels.OLS` for classical confidence intervals and prediction intervals
- Use `QuantileRegressor` for quantile-based prediction intervals
- Use conformal prediction (e.g., `MAPIE` library) for model-agnostic calibrated prediction
  intervals on top of any sklearn estimator

### 6.7 Cannot Handle Missing Values Natively

All sklearn linear models require complete-case data. Passing a matrix with NaN values raises
a `ValueError`. You must impute missing values before fitting.

**Mitigation:**
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # or 'mean', 'most_frequent', KNNImputer
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=np.logspace(-4, 6, 200))),
])
pipe.fit(X_train, y_train)
```

### 6.8 Lasso Selects At Most $n$ Features When $p > n$

**The failure:** When $p > n$, Lasso can select at most $n$ features (the underdetermined system
guarantees this — you cannot have more nonzero coefficients than constraints in the KKT system).
For $p = 50{,}000$ (genomics) and $n = 200$ patients, Lasso is limited to selecting at most 200
SNPs, even if 500 are truly relevant.

**Mechanism:** This is a consequence of the geometry of the L1 minimization in underdetermined
systems — the solution lives on a face of dimension at most $\min(n, p)$ of the L1 ball.

**Mitigation:** ElasticNet (the L2 component removes the degeneracy and allows up to $p$ nonzero
features); ARDRegression; dimensionality reduction + Lasso in stages; group Lasso.

---

## §7 — Diagnostic Plots & Evaluation

Fitting a linear model is the beginning, not the end. Before trusting any conclusion from a
linear model, you should run a standard battery of diagnostic checks. Each plot tests a specific
assumption or reveals a specific failure mode.

### 7.1 Residual vs. Fitted Plot — Testing Linearity and Homoskedasticity

The most important diagnostic. Plot the residuals $e_i = y_i - \hat{y}_i$ against the fitted
values $\hat{y}_i$.

**What good looks like:** A horizontal band of roughly constant spread around zero. No pattern.

**What bad looks like:**
- U-shape or systematic curve → **nonlinearity**: the true relationship is not linear in $x$
- Funnel (increasing spread with fitted values) → **heteroskedasticity**: variance is not constant
- Outlying points far from the band → **influential outliers**

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = {
    'OLS': ols,
    'Ridge': ridge_cv,
    'Lasso': lasso_cv,
}

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_sc)
    residuals = y_test_array - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
    ax.axhline(0, color='red', linestyle='--', lw=2)
    # Add lowess smoothing line to reveal trends
    from scipy.stats import pearsonr
    ax.set_xlabel('Fitted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f'{name} — Residuals vs. Fitted', fontsize=13)
    r, _ = pearsonr(y_pred, np.abs(residuals))
    ax.text(0.02, 0.95, f'corr(fitted, |resid|) = {r:.3f}',
            transform=ax.transAxes, fontsize=10, va='top')

plt.tight_layout()
plt.show()
```

### 7.2 Q-Q Plot of Residuals — Testing Normality

The quantile-quantile (Q-Q) plot compares the distribution of residuals to the standard normal
distribution. If residuals are normally distributed (as assumed for valid inference), points
should fall on the diagonal line.

**What bad looks like:**
- S-shape → heavy tails (more extreme residuals than Gaussian predicts): consider robust regression
- One tail curving off → right-skewed or left-skewed residuals: consider log-transforming target
- Discrete steps → discretized target variable

```python
import scipy.stats as stats

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, models.items()):
    residuals = y_test_array - model.predict(X_test_sc)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals)
    ax.scatter(osm, osr, alpha=0.6, s=20)
    ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(f'{name} — Normal Q-Q Plot')
    ax.text(0.02, 0.95, f'R = {r:.4f}', transform=ax.transAxes, va='top')

plt.tight_layout()
plt.show()
```

### 7.3 Scale-Location Plot — Detecting Heteroskedasticity

Plot $\sqrt{|e_i|}$ against $\hat{y}_i$. Under homoskedasticity (constant variance), the lowess
curve should be flat. An upward slope indicates that residual variance increases with the fitted
value (multiplicative heteroskedasticity — consider log-transforming the target).

```python
fig, ax = plt.subplots(figsize=(8, 5))
y_pred = lasso_cv.predict(X_test_sc)
residuals = y_test_array - y_pred
sqrt_abs_resid = np.sqrt(np.abs(residuals))

ax.scatter(y_pred, sqrt_abs_resid, alpha=0.5)
# Lowess trend line
from statsmodels.nonparametric.smoothers_lowess import lowess
smoothed = lowess(sqrt_abs_resid, y_pred, frac=0.5)
ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', lw=2, label='Lowess')
ax.set_xlabel('Fitted Values')
ax.set_ylabel('√|Residuals|')
ax.set_title('Scale-Location Plot (Homoskedasticity Check)')
ax.legend()
plt.tight_layout()
plt.show()
```

### 7.4 Regularization Path Plots — The Most Important Linear Model Diagnostic

The regularization path shows how each coefficient changes as $\alpha$ varies from high (null
model, all zero) to low (dense OLS). This plot reveals:
- Which features are selected first (most important)
- Which features are related (enter/exit together)
- The stable range of $\alpha$ where the selected support doesn't change

```python
from sklearn.linear_model import lasso_path, enet_path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Lasso path ──────────────────────────────────────────────────────────────
alphas_l, coefs_l, _ = lasso_path(X_train_sc, y_train, n_alphas=100)
ax = axes[0]
colors = cm.tab10(np.linspace(0, 1, X_train_sc.shape[1]))
for j in range(coefs_l.shape[0]):
    ax.plot(alphas_l, coefs_l[j, :], color=colors[j % 10],
            label=feature_names[j] if len(feature_names) <= 10 else None)
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xlabel('alpha (log scale, decreasing →)')
ax.set_ylabel('Coefficients')
ax.set_title('Lasso Regularization Path')
ax.axvline(lasso_cv.alpha_, color='black', linestyle='--', lw=2,
           label=f'CV alpha = {lasso_cv.alpha_:.4f}')
if len(feature_names) <= 10:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
else:
    ax.legend(['CV alpha'], loc='upper left')

# ── Ridge path (via SVD shrinkage factors) ───────────────────────────────────
ax = axes[1]
alphas_r = np.logspace(-3, 5, 100)
from sklearn.linear_model import Ridge
ridge_coefs = []
for a in alphas_r:
    r = Ridge(alpha=a)
    r.fit(X_train_sc, y_train)
    ridge_coefs.append(r.coef_)
ridge_coefs = np.array(ridge_coefs)  # (n_alphas, n_features)

for j in range(ridge_coefs.shape[1]):
    ax.plot(alphas_r, ridge_coefs[:, j], color=colors[j % 10],
            label=feature_names[j] if len(feature_names) <= 10 else None)
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xlabel('alpha (log scale, decreasing →)')
ax.set_ylabel('Coefficients')
ax.set_title('Ridge Regularization Path')
ax.axvline(ridge_cv.alpha_, color='black', linestyle='--', lw=2,
           label=f'CV alpha = {ridge_cv.alpha_:.1f}')
if len(feature_names) <= 10:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()
```

> 💡 **Intuition:** The Lasso path looks like ramps that suddenly become flat at zero (features
> "entering" the model as alpha decreases). The Ridge path looks like smooth curves converging
> to the OLS values — no exact zeros, but some features remain much smaller than others.

### 7.5 CV Error vs. Alpha Plot — Reading the Optimal Window

`LassoCV` stores the cross-validation MSE at each alpha in `mse_path_`. Plotting this reveals:
- The location and uncertainty of the optimal alpha
- Whether the CV curve is flat (robust to alpha choice) or peaked (sensitive)
- Whether using the "1SE rule" (selecting the largest alpha within 1 SE of the minimum)
  is appropriate for sparser models

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# ── Lasso CV curve ────────────────────────────────────────────────────────
ax = axes[0]
alphas_path = lasso_cv.alphas_
mse_path = lasso_cv.mse_path_      # Shape: (n_alphas, n_folds)
mse_mean = mse_path.mean(axis=1)
mse_std  = mse_path.std(axis=1)

ax.semilogx(alphas_path, mse_mean, 'b-o', ms=3, label='Mean CV MSE')
ax.fill_between(alphas_path, mse_mean - mse_std, mse_mean + mse_std,
                alpha=0.2, color='blue', label='±1 SD')
ax.axvline(lasso_cv.alpha_, color='red', lw=2, linestyle='--',
           label=f'Optimal alpha = {lasso_cv.alpha_:.4f}')
# 1-SE rule alpha (largest alpha within 1 SE of minimum)
min_mse_idx = np.argmin(mse_mean)
threshold_1se = mse_mean[min_mse_idx] + mse_std[min_mse_idx]
alpha_1se_idx = np.where(mse_mean <= threshold_1se)[0][0]  # largest valid alpha
ax.axvline(alphas_path[alpha_1se_idx], color='orange', lw=2, linestyle=':',
           label=f'1-SE alpha = {alphas_path[alpha_1se_idx]:.4f}')
ax.set_xlabel('alpha (log scale)')
ax.set_ylabel('CV MSE')
ax.set_title('Lasso: CV Error vs. Alpha')
ax.legend()
ax.invert_xaxis()

# ── Number of nonzero features vs alpha ──────────────────────────────────
ax = axes[1]
alphas_l, coefs_l, _ = lasso_path(X_train_sc, y_train, n_alphas=200)
n_nonzero = (coefs_l != 0).sum(axis=0)
ax2 = ax.twinx()
ax.semilogx(alphas_l, n_nonzero, 'g-', lw=2, label='# Nonzero Features')
ax.invert_xaxis()
ax.set_xlabel('alpha (log scale, decreasing →)')
ax.set_ylabel('Number of Nonzero Coefficients', color='green')
ax.axvline(lasso_cv.alpha_, color='red', lw=2, linestyle='--',
           label=f'CV alpha → {(lasso_cv.coef_ != 0).sum()} features')
ax.set_title('Lasso: Feature Sparsity vs. Alpha')
ax.legend()
plt.tight_layout()
plt.show()
```

### 7.6 Coefficient Bar Plot — Comparing Models

```python
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
model_items = [('OLS', ols), ('Ridge', ridge_cv), ('Lasso', lasso_cv)]

for ax, (name, model) in zip(axes, model_items):
    coef = model.coef_
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    n_show = min(len(feature_names), 15)
    ax.barh(
        range(n_show),
        coef[sorted_idx[:n_show]],
        color=['steelblue' if c > 0 else 'tomato' for c in coef[sorted_idx[:n_show]]],
    )
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[:n_show]], fontsize=9)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Coefficient Value')
    ax.set_title(f'{name} Coefficients')

plt.tight_layout()
plt.show()
```

### 7.7 Actual vs. Predicted Plot

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_sc)
    from sklearn.metrics import r2_score, mean_absolute_error
    r2  = r2_score(y_test_array, y_pred)
    mae = mean_absolute_error(y_test_array, y_pred)
    ax.scatter(y_test_array, y_pred, alpha=0.5, s=20)
    lims = [min(y_test_array.min(), y_pred.min()),
            max(y_test_array.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{name}\nR²={r2:.3f}  MAE={mae:.1f}')
    ax.legend()

plt.tight_layout()
plt.show()
```

### 7.8 Breusch-Pagan Heteroskedasticity Test

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

X_const = sm.add_constant(X_train_sc)
sm_ols = sm.OLS(y_train, X_const).fit()

# Breusch-Pagan test
bp_lm, bp_pval, bp_f, bp_fpval = het_breuschpagan(sm_ols.resid, sm_ols.model.exog)
print(f"Breusch-Pagan test: LM={bp_lm:.3f}, p={bp_pval:.4f}")
if bp_pval < 0.05:
    print("  → Heteroskedasticity detected (p < 0.05)")
    print("  → Consider: log-transform target, WLS, or robust standard errors (HC3)")
else:
    print("  → No evidence of heteroskedasticity")

# If heteroskedastic, use robust standard errors
sm_ols_robust = sm.OLS(y_train, X_const).fit(cov_type='HC3')
print("\nWith HC3 robust standard errors:")
print(sm_ols_robust.summary().tables[1])
```

### 7.9 Evaluation Metrics Guide

| Metric | Formula | Use when | Pitfall |
|---|---|---|---|
| **R²** (coefficient of determination) | $1 - \text{SS}_{\text{res}}/\text{SS}_{\text{tot}}$ | Universal baseline; compare models on same data | Can be negative for out-of-sample; inflated by adding features (use adj-R²) |
| **Adjusted R²** | $1 - (1-R^2)\frac{n-1}{n-p-1}$ | Model comparison with different # features | Only valid for OLS; not for regularized models |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Same units as target; widely reported | Sensitive to outliers; must know scale of target |
| **MAE** | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Outlier-robust; same units as target | Less analytically tractable; ignores error distribution |
| **MAPE** | $\frac{100}{n}\sum\frac{|y_i - \hat{y}_i|}{|y_i|}$ | Percentage error; scale-free | Undefined or infinite when $y_i = 0$ |
| **MASE** | MAE / (MAE of naïve baseline) | Scale-free comparison across datasets | Requires defining a meaningful baseline |

> 🏆 **Best practice:** Report R², RMSE, and MAE together. Use R² for model-to-model comparison
> on the same dataset. Use MAE for communicating accuracy to non-technical stakeholders (it is
> in target units and is outlier-robust). Use RMSE when large errors are particularly costly
> (squared loss amplifies them).

---

## §8 — Explainability & Interpretable AI

Linear models are the most interpretable class of models in mainstream machine learning. But
"interpretable" does not mean "no need for explanation tools." The coefficient table explains
*global* behavior, but many real-world questions are *local* ("why did the model give this
specific prediction?"). SHAP, permutation importance, and PDP/ICE fill this gap.

### 8.1 Built-in Interpretability: Coefficients

The primary interpretability tool for linear models is the coefficient vector itself:

```python
import pandas as pd
import numpy as np

# Coefficient table with confidence intervals (requires statsmodels for CIs)
import statsmodels.api as sm

X_const = sm.add_constant(X_train_sc)
sm_fit = sm.OLS(y_train, X_const).fit()

coef_df = pd.DataFrame({
    'feature': ['intercept'] + feature_names,
    'coefficient': sm_fit.params.values,
    'std_error': sm_fit.bse.values,
    't_stat': sm_fit.tvalues.values,
    'p_value': sm_fit.pvalues.values,
    'ci_lower': sm_fit.conf_int().iloc[:, 0].values,
    'ci_upper': sm_fit.conf_int().iloc[:, 1].values,
})
coef_df = coef_df.sort_values('p_value').reset_index(drop=True)
print(coef_df.to_string(index=False))
```

**Standardized coefficients as importance:** On standardized features, the coefficient magnitude
directly represents the change in the target for a 1-SD change in the feature — a natural
measure of feature importance. Features with large $|\hat{\beta}_j|$ matter more.

**Lasso as built-in variable selection:** After Lasso fit, the zero/nonzero pattern of
`coef_` directly tells you which features matter. Features with `coef_ == 0` are excluded.
This is a model-intrinsic form of feature importance.

### 8.2 SHAP: LinearExplainer — Exact Values from Coefficients

For linear models, SHAP values have a precise mathematical relationship to the model
coefficients. The `LinearExplainer` computes exact SHAP values efficiently without any sampling.

**Mathematical connection:** Under the **independence assumption** (features are independent):

$$\phi_j(\mathbf{x}) = \hat{\beta}_j \cdot (x_j - \mathbb{E}[X_j])$$

Each SHAP value is the coefficient times the deviation of the feature from its training mean.
This is a direct, interpretable formula. The sum of SHAP values equals the prediction minus the
baseline (expected prediction):

$$\hat{y}(\mathbf{x}) - \hat{y}_{\text{base}} = \sum_{j=1}^p \phi_j(\mathbf{x})$$

Under the **correlation-aware assumption** (using `masker=X_train` — the default), SHAP
distributes credit among correlated features using a linear transform estimated from the
training data. This is more accurate when features are correlated.

```python
import shap
import numpy as np
import matplotlib.pyplot as plt

# ── Setup LinearExplainer ─────────────────────────────────────────────────
# Use the Lasso model (sparser → cleaner SHAP plot)
fitted_lasso = lasso_cv  # Already fitted

# Independence assumption (fast, ignores correlations)
explainer_ind = shap.LinearExplainer(
    fitted_lasso,
    masker=shap.maskers.Independent(X_train_sc, max_samples=200)
)
shap_values_ind = explainer_ind.shap_values(X_test_sc)
print(f"SHAP values shape: {shap_values_ind.shape}")  # (n_test, n_features)

# Verify: SHAP additivity
# sum(shap_values[i]) + expected_value ≈ prediction[i]
pred_check = shap_values_ind[0].sum() + explainer_ind.expected_value
print(f"SHAP sum + E[f] = {pred_check:.4f}")
print(f"Model prediction = {fitted_lasso.predict(X_test_sc[[0]])[0]:.4f}")
# Should be equal (up to numerical precision)

# Correlation-aware assumption (recommended for correlated features)
explainer_corr = shap.LinearExplainer(
    fitted_lasso,
    masker=X_train_sc,  # Full training set as background
)
shap_values_corr = explainer_corr.shap_values(X_test_sc)
```

**SHAP Summary Plot (Global Feature Importance):**

```python
# Summary plot: each dot = one test sample; x-axis = SHAP value (+ means pushes prediction up)
# Color = feature value (red = high, blue = low)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_corr,
    X_test_sc,
    feature_names=feature_names,
    plot_type='dot',     # 'dot' (beeswarm), 'bar' (mean absolute SHAP), 'violin'
    show=False,
    max_display=10,
)
plt.title('SHAP Summary Plot (Lasso Model)', fontsize=14)
plt.tight_layout()
plt.show()

# Bar version (easier to read, less information)
plt.figure(figsize=(10, 5))
shap.summary_plot(
    shap_values_corr,
    X_test_sc,
    feature_names=feature_names,
    plot_type='bar',
    show=False,
    max_display=10,
)
plt.title('SHAP Mean Absolute Values (Global Importance)', fontsize=14)
plt.tight_layout()
plt.show()
```

**SHAP Waterfall Plot (Local — Single Prediction Explanation):**

```python
# Explain a single prediction in detail
sample_idx = 0
explanation = shap.Explanation(
    values=shap_values_corr[sample_idx],
    base_values=explainer_corr.expected_value,
    data=X_test_sc[sample_idx],
    feature_names=feature_names,
)

plt.figure(figsize=(10, 6))
shap.waterfall_plot(explanation, max_display=10, show=False)
plt.title(f'SHAP Waterfall: Sample {sample_idx}\n'
          f'Actual={y_test_array[sample_idx]:.1f}  '
          f'Predicted={fitted_lasso.predict(X_test_sc[[sample_idx]])[0]:.1f}')
plt.tight_layout()
plt.show()
```

**SHAP Force Plot (Alternative Local Visualization):**

```python
# Force plot: horizontal bar showing which features push prediction up/down
shap.force_plot(
    explainer_corr.expected_value,
    shap_values_corr[0],
    X_test_sc[0],
    feature_names=feature_names,
    matplotlib=True,
    show=False,
)
plt.title('SHAP Force Plot: Sample 0')
plt.tight_layout()
plt.show()
```

> ⚠️ **Key SHAP API note (0.45.0+):** The `feature_perturbation` parameter in `LinearExplainer`
> is deprecated. Use the `masker` parameter instead:
> - `masker=shap.maskers.Independent(X_train)` → independence assumption
> - `masker=X_train` (numpy array) → correlation-aware (uses training data to estimate correlations)
>
> For linear models, the independence assumption gives `shap_values[j] = coef_[j] * (x_j - mean_j)`.
> This is exact for models with no feature interactions (which is all standard linear models).

### 8.3 SHAP Dependence Plot — Feature-Level Relationships

```python
# Dependence plot for a specific feature
# Shows SHAP value for that feature vs. feature value
# Colored by another feature (auto-detected or user-specified)

fig, axes = plt.subplots(2, min(5, len(feature_names)//2), figsize=(18, 8))
axes = axes.flatten()

for i, fname in enumerate(feature_names[:len(axes)]):
    shap.dependence_plot(
        fname,
        shap_values_corr,
        X_test_sc,
        feature_names=feature_names,
        ax=axes[i],
        show=False,
    )

plt.suptitle('SHAP Dependence Plots (Linear Model)', fontsize=14)
plt.tight_layout()
plt.show()
```

### 8.4 Permutation Importance — Model-Agnostic Feature Importance

While coefficients (and SHAP) give intrinsic feature importance for linear models, permutation
importance is useful as a **validation check**: it measures the actual decrease in model
performance when a feature's values are randomly shuffled.

```python
from sklearn.inspection import permutation_importance

# Compute permutation importance on test set
perm_imp = permutation_importance(
    lasso_cv,
    X_test_sc,
    y_test,
    n_repeats=30,           # 30 permutations per feature
    random_state=42,
    scoring='r2',           # metric to use
    n_jobs=-1,
)

# Sort and plot
sorted_idx = np.argsort(perm_imp.importances_mean)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(
    perm_imp.importances[sorted_idx[:12]].T,
    vert=False,
    labels=[feature_names[i] for i in sorted_idx[:12]],
)
ax.axvline(0, color='red', lw=1, linestyle='--')
ax.set_xlabel('Decrease in R² (mean ± std over 30 permutations)')
ax.set_title('Permutation Feature Importance (Lasso Model)')
plt.tight_layout()
plt.show()
```

> 💡 **When permutation importance disagrees with coefficients:** This can happen with
> correlated features. Two correlated features $x_1 \approx x_2$: if $x_1$ is in the Lasso
> model with coefficient 2.0 and $x_2$ is zeroed out, permuting $x_1$ alone will degrade
> performance significantly — but permuting $x_2$ (coefficient 0) will also degrade performance
> slightly because $x_1$ partially explains the information in $x_2$. Permutation importance
> reflects the actual prediction contribution including correlation effects; coefficients reflect
> the model's parameter choice.

### 8.5 Partial Dependence Plots (PDP) and ICE

For linear models, PDP is a straight line by construction (the model is linear). But visualizing
it is still useful to confirm the linearity and to understand the marginal effect magnitude.

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(16, 5))
display = PartialDependenceDisplay.from_estimator(
    lasso_cv,
    X_test_sc,
    features=list(range(min(5, X_test_sc.shape[1]))),
    feature_names=feature_names,
    kind='both',            # 'average' (PDP) or 'individual' (ICE) or 'both'
    subsample=50,           # For ICE: number of individual lines to plot
    random_state=42,
    ax=ax,
    grid_resolution=50,
)
display.figure_.suptitle('Partial Dependence Plots (Lasso)', fontsize=14)
plt.tight_layout()
plt.show()
```

> 💡 **Interpretation for linear models:** Since the model is $\hat{y} = \beta_0 + \sum_j \beta_j x_j$,
> the partial dependence of $y$ on $x_j$ is $\mathbb{E}_{\mathbf{x}_{-j}}[\hat{y}] = \bar{y} + \beta_j(x_j - \bar{x}_j)$
> — a perfect straight line with slope $\beta_j$. ICE curves are also all parallel straight lines
> with the same slope. Any deviation from this in practice indicates interactions or nonlinearity
> that your linear model is not capturing.

### 8.6 LIME for Local Explanations

LIME (Local Interpretable Model-agnostic Explanations) fits a local linear approximation to any
model's predictions in the neighborhood of a specific point. For an already-linear model, LIME
produces explanations identical to the model coefficients — but it is useful for comparing a
linear model's explanations to those of a nonlinear model on the same data.

```python
# pip install lime
import lime
import lime.lime_tabular

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_sc,
    feature_names=feature_names,
    mode='regression',
    discretize_continuous=False,
    random_state=42,
)

# Explain a single prediction
lime_exp = lime_explainer.explain_instance(
    data_row=X_test_sc[0],
    predict_fn=lasso_cv.predict,
    num_features=10,
    num_samples=1000,
)
lime_exp.show_in_notebook(show_table=True)

# For a standalone plot:
lime_vals = lime_exp.as_list()
features_lime = [x[0] for x in lime_vals]
values_lime   = [x[1] for x in lime_vals]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['steelblue' if v > 0 else 'tomato' for v in values_lime]
ax.barh(features_lime, values_lime, color=colors)
ax.axvline(0, color='black', lw=0.8)
ax.set_title(f'LIME Explanation: Sample 0\n'
             f'Prediction = {lasso_cv.predict(X_test_sc[[0]])[0]:.1f}')
plt.tight_layout()
plt.show()
```

---

## §9 — Complete Python Worked Example: Diabetes Dataset

We now build a complete, production-quality linear model pipeline on the sklearn `diabetes`
dataset — a medical regression benchmark predicting disease progression from clinical measurements.
The dataset has $n = 442$ patients and $p = 10$ features (age, sex, BMI, blood pressure, and six
serum measurements), with a continuous numeric target (disease progression score one year after
baseline). The features are pre-standardized in the original dataset, but we demonstrate the full
preprocessing workflow regardless.

This example covers: EDA → VIF analysis → OLS baseline → Ridge/Lasso/ElasticNet comparison →
CV-selected regularization → regularization path visualization → feature selection → SHAP
explanations → statistical inference with statsmodels.

### 9.1 Setup and Data Loading

```python
# ─────────────────────────────────────────────────────────────────────────────
# LINEAR MODELS WORKED EXAMPLE — Diabetes Dataset
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV,
    Lasso, LassoCV, ElasticNet, ElasticNetCV,
    BayesianRidge, HuberRegressor,
)

np.random.seed(42)

# Load data
data = load_diabetes(as_frame=True)
X_df = data.data
y    = data.target

print(f"Dataset: Diabetes | n={X_df.shape[0]}, p={X_df.shape[1]}")
print(f"Target: disease progression — "
      f"mean={y.mean():.1f}, std={y.std():.1f}, "
      f"range=[{y.min():.0f}, {y.max():.0f}]")
print(f"\nFeatures: {list(X_df.columns)}")
```

### 9.2 Exploratory Data Analysis

```python
# Correlation with target
corrs = X_df.corrwith(y).sort_values(ascending=False)
print("\nFeature-target correlations:")
print(corrs.round(3))

# Pair-wise feature correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = pd.concat([X_df, y.rename('target')], axis=1).corr()
sns.heatmap(
    corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
    center=0, vmin=-1, vmax=1, ax=ax, square=True,
    annot_kws={'size': 8}, mask=np.triu(np.ones_like(corr_matrix, dtype=bool))
)
ax.set_title('Feature Correlation Matrix (Diabetes)', fontsize=13)
plt.tight_layout()
plt.show()

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y, bins=25, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Disease Progression')
axes[0].set_ylabel('Count')
axes[0].set_title('Target Distribution')

(osm, osr), (slope, intercept, r) = stats.probplot(y)
axes[1].scatter(osm, osr, alpha=0.6, s=20)
axes[1].plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2)
axes[1].set_title(f'Q-Q Plot of Target  R={r:.4f}')
axes[1].set_xlabel('Theoretical Quantiles')
axes[1].set_ylabel('Sample Quantiles')
plt.tight_layout()
plt.show()
```

### 9.3 Preprocessing and VIF Analysis

```python
# Train/Test split
feature_names = list(X_df.columns)
X     = X_df.values
y_arr = y.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_arr, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# Standardize (fit on train, transform both)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# VIF analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_df = pd.DataFrame({
    'feature': feature_names,
    'VIF': [variance_inflation_factor(X_train_sc, i)
            for i in range(len(feature_names))]
}).sort_values('VIF', ascending=False)
print("\nVariance Inflation Factors (VIF):")
print(vif_df.to_string(index=False))
print("\nVIF guide: <5 = OK, 5-10 = investigate, >10 = severe collinearity")
```

### 9.4 Model Training and Comparison

```python
# ── OLS baseline ─────────────────────────────────────────────────────────────
ols = LinearRegression().fit(X_train_sc, y_train)

# ── Ridge with GCV ───────────────────────────────────────────────────────────
ridge_cv = RidgeCV(alphas=np.logspace(-4, 6, 300), cv=None).fit(X_train_sc, y_train)

# ── Lasso with 5-fold CV + path ───────────────────────────────────────────────
lasso_cv = LassoCV(n_alphas=200, cv=5, max_iter=50000, n_jobs=-1,
                   random_state=42).fit(X_train_sc, y_train)

# ── ElasticNet with 2D CV ──────────────────────────────────────────────────────
enet_cv = ElasticNetCV(
    l1_ratio=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    n_alphas=200, cv=5, max_iter=50000, n_jobs=-1, random_state=42,
).fit(X_train_sc, y_train)

# ── BayesianRidge ────────────────────────────────────────────────────────────
br = BayesianRidge(max_iter=300, compute_score=True).fit(X_train_sc, y_train)

# ── Huber (robust baseline) ───────────────────────────────────────────────────
huber = HuberRegressor(epsilon=1.35, max_iter=300, alpha=0.0001).fit(X_train_sc, y_train)

# Results table
def eval_model(name, model):
    y_pred = model.predict(X_test_sc)
    return {
        'Model': name,
        'R²':  round(r2_score(y_test, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'MAE':  round(mean_absolute_error(y_test, y_pred), 2),
        'Nonzero coefs': int((np.abs(model.coef_) > 1e-5).sum()),
    }

results = pd.DataFrame([
    eval_model('OLS',         ols),
    eval_model('Ridge',       ridge_cv),
    eval_model('Lasso',       lasso_cv),
    eval_model('ElasticNet',  enet_cv),
    eval_model('BayesianRidge', br),
    eval_model('Huber',       huber),
])
print("\n=== Test-Set Performance ===")
print(results.to_string(index=False))

print(f"\nRidge best alpha:    {ridge_cv.alpha_:.4f}")
print(f"Lasso best alpha:    {lasso_cv.alpha_:.6f} "
      f"→ {(lasso_cv.coef_ != 0).sum()} nonzero features: "
      f"{[f for f, c in zip(feature_names, lasso_cv.coef_) if c != 0]}")
print(f"ElasticNet: alpha={enet_cv.alpha_:.6f}, l1_ratio={enet_cv.l1_ratio_:.2f}")
```

### 9.5 Regularization Path Visualization

```python
from sklearn.linear_model import lasso_path
import matplotlib.cm as cm

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
cmap = cm.tab10(np.linspace(0, 1, len(feature_names)))

# ── Lasso coefficient path ────────────────────────────────────────────────────
ax = axes[0]
alphas_path, coefs_path, _ = lasso_path(X_train_sc, y_train, n_alphas=200)
for j, fname in enumerate(feature_names):
    ax.semilogx(alphas_path, coefs_path[j], color=cmap[j], label=fname, lw=1.5)
ax.invert_xaxis()
ax.axvline(lasso_cv.alpha_, color='black', lw=2.5, linestyle='--',
           label=f'CV opt. → {(lasso_cv.coef_ != 0).sum()} features')
ax.set_xlabel('alpha (decreasing →)')
ax.set_ylabel('Coefficient value')
ax.set_title('Lasso Regularization Path (Diabetes)')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

# ── CV error curve with 1SE rule ──────────────────────────────────────────────
ax = axes[1]
mse_mean = lasso_cv.mse_path_.mean(axis=1)
mse_std  = lasso_cv.mse_path_.std(axis=1)
ax.semilogx(lasso_cv.alphas_, mse_mean, 'b-', lw=1.5, label='Mean CV MSE')
ax.fill_between(lasso_cv.alphas_, mse_mean - mse_std, mse_mean + mse_std,
                alpha=0.25, color='blue', label='±1 SD')
ax.axvline(lasso_cv.alpha_, color='red', lw=2, linestyle='--',
           label=f'Optimal alpha={lasso_cv.alpha_:.5f}')
min_idx = np.argmin(mse_mean)
thresh_1se = mse_mean[min_idx] + mse_std[min_idx]
valid_1se = np.where(mse_mean <= thresh_1se)[0]
ax.axvline(lasso_cv.alphas_[valid_1se[0]], color='orange', lw=2, linestyle=':',
           label='1-SE rule alpha')
ax.invert_xaxis()
ax.set_xlabel('alpha')
ax.set_ylabel('CV MSE')
ax.set_title('Lasso: CV Error Curve')
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
```

### 9.6 Residual Diagnostics

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
model_list_diag = [('OLS', ols), ('Ridge', ridge_cv), ('Lasso', lasso_cv)]

for col, (name, model) in enumerate(model_list_diag):
    y_pred    = model.predict(X_test_sc)
    residuals = y_test - y_pred

    # Residuals vs. Fitted
    ax = axes[0, col]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='none')
    ax.axhline(0, color='red', lw=2, linestyle='--')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{name}: Residuals vs. Fitted')

    # Q-Q plot
    ax = axes[1, col]
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals)
    ax.scatter(osm, osr, alpha=0.6, s=20)
    ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(f'{name}: Normal Q-Q  (R={r:.3f})')

plt.tight_layout()
plt.show()
```

### 9.7 SHAP Explanations

```python
import shap

# Fit on the Lasso model
explainer = shap.LinearExplainer(lasso_cv, masker=X_train_sc)
shap_vals = explainer.shap_values(X_test_sc)

# Verify SHAP additivity
pred_check = shap_vals[0].sum() + explainer.expected_value
model_pred = lasso_cv.predict(X_test_sc[[0]])[0]
print(f"SHAP additivity check: sum(phi)+E[f]={pred_check:.3f}, model={model_pred:.3f}")

# Summary plot
plt.figure(figsize=(9, 5))
shap.summary_plot(shap_vals, X_test_sc, feature_names=feature_names,
                  plot_type='dot', show=False)
plt.title('SHAP Beeswarm — Lasso on Diabetes Dataset')
plt.tight_layout()
plt.show()

# Waterfall for a single sample
best_idx = np.argmin(np.abs(y_test - lasso_cv.predict(X_test_sc)))
exp = shap.Explanation(
    values=shap_vals[best_idx],
    base_values=explainer.expected_value,
    data=X_test_sc[best_idx],
    feature_names=feature_names,
)
plt.figure(figsize=(9, 5))
shap.waterfall_plot(exp, max_display=10, show=False)
plt.title(f'SHAP Waterfall (best-predicted sample)\n'
          f'Actual={y_test[best_idx]:.1f}  '
          f'Predicted={lasso_cv.predict(X_test_sc[[best_idx]])[0]:.1f}')
plt.tight_layout()
plt.show()
```

### 9.8 Statistical Inference (statsmodels)

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

X_const = sm.add_constant(X_train_sc)
sm_ols = sm.OLS(y_train, X_const).fit()
print(sm_ols.summary())

bp_lm, bp_pval, _, _ = het_breuschpagan(sm_ols.resid, sm_ols.model.exog)
print(f"\nBreusch-Pagan heteroskedasticity test: p={bp_pval:.4f}")
if bp_pval < 0.05:
    print("  → Heteroskedasticity detected. Using HC3 robust standard errors.")
    sm_robust = sm.OLS(y_train, X_const).fit(cov_type='HC3')
    print(sm_robust.summary().tables[1])

# 95% Prediction intervals
preds = sm_ols.get_prediction(sm.add_constant(X_test_sc))
pred_df = preds.summary_frame(alpha=0.05)
coverage = ((y_test >= pred_df['obs_ci_lower'].values) &
            (y_test <= pred_df['obs_ci_upper'].values)).mean()
print(f"\n95% Prediction Interval empirical coverage: {coverage:.2%}  "
      f"(target: 95%)")
```

---






## §10 — When to Use This Algorithm (Decision Guide)

### 10.1 Decision Flowchart

```
START: You have a regression problem with numeric features
│
├─► Is your target continuous (not a probability)?
│     No → Use logistic regression, GLMs (Poisson, Gamma), etc.
│     Yes ↓
│
├─► Do you need a quick, interpretable baseline?
│     Yes → Start with Ridge or Lasso. Always.
│     No ↓
│
├─► Is p >> n (many more features than samples)?
│     Yes ↓
│     ├─► Are features likely sparse (true model uses few features)?
│     │     Yes → Lasso or ElasticNet
│     │     No (dense signal) → Ridge (or ElasticNet with low l1_ratio)
│     No ↓
│
├─► Do you have severe multicollinearity (VIF > 10)?
│     Yes → Ridge (handles it natively via SVD shrinkage)
│             OR ElasticNet (if you also want some sparsity)
│     No ↓
│
├─► Do you need variable selection (sparse model, fewer features)?
│     Yes → Lasso or ElasticNet
│     No ↓
│
├─► Do you have significant outliers (contaminated response)?
│     Yes (moderate, ~5-15%) → HuberRegressor
│          (severe, up to 50%) → RANSACRegressor
│          (breakdown point ~29%) → TheilSenRegressor
│     No ↓
│
├─► Do you need calibrated uncertainty (prediction intervals)?
│     Yes, Bayesian style → BayesianRidge (auto-regularized)
│     Yes, frequentist → statsmodels OLS + get_prediction()
│     Yes, distribution-free → QuantileRegressor or conformal prediction
│     No ↓
│
├─► Is the true relationship clearly nonlinear?
│     Yes → Feature engineering + linear model, or switch to nonlinear model
│     No → Linear model is likely competitive
│
└─► Use cross-validated linear model: RidgeCV / LassoCV / ElasticNetCV
```

### 10.2 Decision Table: Which Linear Model to Use?

| Scenario | Recommended Model | Reason |
|---|---|---|
| **Quick baseline for any regression** | `RidgeCV` | Fast, stable, always works, one CV call |
| **Variable selection required** | `LassoCV` | Exact zeros, built-in feature selection |
| **Correlated features + selection** | `ElasticNetCV` | Grouping effect + sparsity |
| **$p > n$ (genomics, text, etc.)** | `ElasticNetCV` (l1_ratio < 1) | L2 removes degeneracy |
| **Statistical inference (p-values, CIs)** | `statsmodels.OLS` | Full inference machinery |
| **Self-tuning, no CV needed** | `BayesianRidge` | Evidence maximization selects alpha |
| **Predictive uncertainty needed** | `BayesianRidge` | Posterior predictive variance |
| **Outliers in response variable** | `HuberRegressor` | Huber loss reduces outlier influence |
| **Extreme outlier contamination** | `RANSACRegressor` | Up to 50% outlier tolerance |
| **Conditional quantiles / PI** | `QuantileRegressor` | Pinball loss, any quantile |
| **Multiple related targets (multitask)** | `MultiTaskLassoCV` | Joint sparsity across tasks |
| **Massive $n$, needs speed** | `SGDRegressor` | Online learning, stochastic gradient |
| **Maximum statistical stability (small n)** | `BayesianRidge` | No overfitting, uncertainty-aware |
| **Interpretability is paramount** | `Lasso` | Sparse coefficients tell a clean story |

### 10.3 When NOT to Use Linear Models

| Situation | Why Linear Models Fail | What to Use Instead |
|---|---|---|
| **Clear nonlinear relationship** | Model class too restrictive; systematic residual patterns | Gradient boosting, neural nets, GAMs |
| **Complex feature interactions** | Cannot capture $x_1 \times x_2$ effects without manual engineering | Tree ensembles (detect interactions automatically) |
| **Image, text, or audio raw input** | High-dimensional unstructured data needs representation learning | CNNs, transformers, pretrained embeddings |
| **Time series with temporal patterns** | Ignores autocorrelation; poor extrapolation | ARIMA, Prophet, temporal GBMs |
| **Very heterogeneous data** | Different feature relationships in different regions | Local models, mixture models, tree methods |
| **Categorical target with $k > 2$** | Multiclass requires multinomial extension (not covered here) | Multinomial logistic regression, tree methods |

### 10.4 Performance Expectations

On tabular regression benchmarks (the kind used in Kaggle competitions):

- **Linear models** typically achieve 70–90% of the R² of a well-tuned gradient boosting model
  when the true relationship is approximately linear and features are well-engineered.
- On text classification with TF-IDF features, **linear models are often competitive with neural
  networks** and far faster to train and deploy.
- In genomics GWAS studies, **Ridge and Lasso** are the dominant methods — gradient boosting
  struggles with $p > n$ regimes where linear models excel.
- In economics and epidemiology, **linear models are preferred by convention** for interpretability,
  even when they sacrifice some predictive performance.

> 🏆 **Best practice:** Always fit a `LassoCV` or `RidgeCV` before trying any nonlinear model.
> If the linear model achieves acceptable performance, the gain from a more complex model may
> not justify the interpretability cost. If the linear model performs poorly, the residual plots
> will tell you *why* (nonlinearity? outliers? heteroskedasticity?) and guide your next step.

---

## §11 — Resources & Further Reading

### 11.1 The Foundational Papers (Must-Read)

1. **Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.**
   *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288.
   URL: https://academic.oup.com/jrsssb/article/58/1/267/7027929
   *The original Lasso paper. Read it. Tibshirani's writing is remarkably clear for a statistics
   paper. The geometric intuition in Section 2 is the best 5-page explanation of L1 sparsity
   ever written.*

2. **Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems.**
   *Technometrics*, 12(1), 55–67.
   URL: https://doi.org/10.1080/00401706.1970.10488634
   *The original Ridge paper. Read Part I for the theoretical result; Part II for the applications.
   Remarkable that a 1970 paper is still directly relevant.*

3. **Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net.**
   *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 67(2), 301–320.
   URL: https://academic.oup.com/jrsssb/article/67/2/301/7109482
   *Definitive paper on ElasticNet. Theorem 1 (grouping property) is the key result. Section 4
   on the geometric picture is illuminating.*

4. **Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression.**
   *The Annals of Statistics*, 32(2), 407–499.
   URL: https://arxiv.org/abs/math/0406456 (free preprint)
   *The LARS paper. 90 pages, but Section 2 (LARS algorithm) and Section 3 (connection to Lasso)
   are the essential 20 pages. Section 6 on degrees of freedom is important for AIC/BIC.*

5. **Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized
   linear models via coordinate descent.**
   *Journal of Statistical Software*, 33(1), 1–22.
   URL: https://doi.org/10.18637/jss.v033.i01
   *The glmnet paper. This is the algorithm behind sklearn's Lasso. Read Section 2 for the
   coordinate descent derivation and Section 3 for the warm-start path trick.*

### 11.2 Important Follow-Up Papers

6. **Tibshirani, R. (2011). Regression shrinkage and selection via the lasso: A retrospective.**
   *Journal of the Royal Statistical Society: Series B*, 73(3), 273–282.
   URL: https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2011.00771.x
   *Tibshirani reviews 15 years of progress: LARS, adaptive Lasso, group Lasso, fused Lasso,
   sign consistency conditions. Essential for understanding where the field went.*

7. **Zou, H. (2006). The adaptive lasso and its oracle properties.**
   *Journal of the American Statistical Association*, 101(476), 1418–1429.
   *Proves that adaptive Lasso has the oracle property (correct variable selection and optimal
   estimation rates simultaneously). The standard Lasso does not.*

8. **Meinshausen, N., & Bühlmann, P. (2006). High-dimensional graphs and variable selection
   with the lasso.**
   *The Annals of Statistics*, 34(3), 1436–1462.
   *The first rigorous conditions under which Lasso correctly identifies the true sparse support
   (the "sign consistency" result). Answers when Lasso is statistically reliable.*

9. **Bickel, P. J., Ritov, Y., & Tsybakov, A. B. (2009). Simultaneous analysis of Lasso and
   Dantzig selector.**
   *The Annals of Statistics*, 37(4), 1705–1732.
   *The restricted eigenvalue condition (REC) — the cleanest set of sufficient conditions for
   Lasso estimation and prediction bounds in high dimensions.*

10. **Meinshausen, N., & Bühlmann, P. (2010). Stability selection.**
    *Journal of the Royal Statistical Society: Series B*, 72(4), 417–473.
    *Stability selection: bootstrapped Lasso for reliable feature selection with finite-sample
    error control. The recommended approach when interpretability of selected features matters.*

### 11.3 Books

11. **Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
    (ESL), 2nd edition. Springer.**
    Free PDF: https://hastie.su.domains/ElemStatLearn/
    *Chapter 3 (Linear Methods for Regression) is the mathematical deep-dive. Section 3.4 on
    Ridge (with SVD interpretation), 3.4.2 on LASSO, 3.4.4 on LARS, 3.8 on path algorithms.
    The reference if you want derivations.*

12. **James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). An Introduction to
    Statistical Learning (ISL), 2nd edition. Springer.**
    Free PDF: https://www.statlearning.com/
    *Chapter 6 (Linear Model Selection and Regularization) is the pedagogically gentler version.
    Excellent figures for geometric intuition of Ridge vs. Lasso. Chapters 6.1–6.2 are essential.*

13. **Bühlmann, P., & van de Geer, S. (2011). Statistics for High-Dimensional Data: Methods,
    Theory and Applications. Springer.**
    *The theoretical reference for Lasso in $p \gg n$ settings: oracle inequalities, restricted
    eigenvalue conditions, sign consistency. Not for the faint of heart, but essential for
    genomics and high-dimensional inference.*

14. **Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical
    Models. Cambridge University Press.**
    *Practical guide to regression from a Bayesian perspective. Excellent treatment of
    standardization, interpretation, and model checking. Chapter 3–4 on linear regression
    and Chapter 12 on multilevel models.*

### 11.4 Official Package Documentation

15. **sklearn Linear Models — Complete Reference**
    URL: https://scikit-learn.org/stable/modules/linear_model.html
    *Complete documentation for all sklearn linear estimators: parameter tables, solver
    comparison, code examples. The canonical reference for sklearn 1.5.x API.*

16. **sklearn Ridge API**
    URL: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

17. **sklearn Lasso API**
    URL: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

18. **sklearn ElasticNet API**
    URL: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

19. **SHAP LinearExplainer Documentation**
    URL: https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html
    *Authoritative API reference including the masker parameter and the mathematical derivation
    of SHAP values for linear models under independence and correlation assumptions.*

20. **statsmodels OLS and Diagnostic Tests**
    URL: https://www.statsmodels.org/stable/stats.html
    *Complete reference for het_breuschpagan, het_white, OLS summary output, robust covariance
    types (HC0–HC3).*

### 11.5 Best Tutorials and Blog Posts

21. **sklearn Linear Model Selection Practical Guide**
    URL: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html
    *Official sklearn example comparing L1 and ElasticNet on synthetic sparse data.*

22. **Distill.pub: Feature Visualization** (conceptual background for interpreting coefficients)
    URL: https://distill.pub/2017/feature-visualization/

23. **glmnet Stanford Official Page (R reference implementation)**
    URL: https://glmnet.stanford.edu/articles/glmnet.html
    *Author-maintained documentation for the R glmnet package — the canonical reference
    implementation. Parameter conventions differ from sklearn; important when comparing results.*

24. **SHAP Documentation: Linear Models Math**
    URL: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/linear_models/Math%20behind%20LinearExplainer%20with%20correlation%20feature%20perturbation.html
    *Mathematical derivation of correlated-feature SHAP values for linear models.*

### 11.6 Online Courses

25. **Stanford CS229: Machine Learning (Andrew Ng) — Problem Set on Linear Models**
    URL: https://cs229.stanford.edu/
    *The theoretical foundations; problem sets include Ridge, bias-variance decomposition.*

26. **fast.ai Practical Deep Learning — Linear models as baselines**
    URL: https://course.fast.ai/
    *Practical treatment emphasizing that linear models should always be your first step.*

---

*Chapter complete. Next in the Supervised Learning Masterclass:*
*→ Decision Trees (`decision-trees.md`)*
*→ Support Vector Machines (`support-vector-machines.md`)*
*→ K-Nearest Neighbors (`k-nearest-neighbors.md`)*
