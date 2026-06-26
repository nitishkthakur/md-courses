# Logistic Regression: The Complete Masterclass

> **Why this algorithm matters:** Logistic regression is the bedrock of predictive modeling.
> It powers credit-risk scoring at every major bank, clinical trial analysis at every pharmaceutical
> company, and the output layer of nearly every modern deep neural network. In medicine alone,
> logistic regression has been the standard model for binary outcome prediction — survival, disease
> presence, treatment response — for over six decades. A 2019 review in *NEJM* found that logistic
> regression remains the most commonly used statistical model in clinical research papers, used in
> more than 40% of all studies. Understanding it deeply — not just calling `clf.fit()`, but
> understanding every Greek letter, every solver, every pitfall — is the single most efficient
> investment you can make as a machine learning practitioner.

---

## §0 — Python Ecosystem & Package Guide

Before we dive into the mathematics, you need to know the landscape. Logistic regression is one
of those rare algorithms where the implementation choice carries real consequences: the solver you
pick determines which regularization penalties are available; the package you pick determines
whether you get p-values, calibration, or Bayesian posteriors. Let's map the terrain.

### Complete Package Table

| Package | Import | Version (mid-2026) | What it provides | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.linear_model import LogisticRegression` | ~1.5.x | Full LR with 6 solvers, CV variant, SGD variant, calibration tools | BSD-3 |
| `statsmodels` | `import statsmodels.api as sm` | ~0.14.x | Statistical inference: p-values, CIs, AIC/BIC, deviance residuals | BSD-3 |
| `python-glmnet` | `from glmnet import LogitNet` | ~2.0 (Civis Analytics) | Full L1/L2/ElasticNet regularization path via Fortran glmnet backend | BSD-3 |
| `tensorflow-probability` | `import tensorflow_probability as tfp` | ~0.24.x | Bayesian LR with MCMC/VI posterior inference over weights | Apache-2 |
| `PyMC` | `import pymc as pm` | ~5.x | Full Bayesian LR with NUTS sampler, posterior predictive checks | Apache-2 |
| `imbalanced-learn` | `from imblearn.over_sampling import SMOTE` | ~0.12.x | SMOTE and resampling for class-imbalance scenarios | MIT |
| `shap` | `import shap` | ~0.46.x | LinearExplainer for exact SHAP values from LR coefficients | MIT |
| `optuna` | `import optuna` | ~3.x | Hyperparameter optimization (C, l1_ratio, solver) | MIT |

### Top Picks — Recommendation Table

| Package | Best for | Approach | Notes |
|---|---|---|---|
| **`sklearn.LogisticRegression`** | Production ML pipelines, standard classification | Numerical optimization (6 solvers) | **First choice** for 95% of use cases; fast, robust, Pipeline-compatible |
| **`statsmodels.Logit`** | Statistical inference, p-values, clinical research | Newton-Raphson / IRLS | The choice when you need CIs, hypothesis tests, AIC/BIC |
| **`sklearn.LogisticRegressionCV`** | Automatic C-selection via cross-validation | Same solvers as LR | Use this instead of GridSearchCV when tuning only C |
| **`glmnet.LogitNet`** | High-dimensional, sparse feature selection, regularization path | Coordinate descent (Fortran) | Fastest for full L1-path computation; sklearn-compatible API |
| **`PyMC`** | Uncertainty quantification, small data, hierarchical models | MCMC (NUTS) | When you need full posterior distributions over coefficients |

### The Same Fit Across Top Packages

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and split data (used throughout this chapter)
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ── Option 1: sklearn (the workhorse) ──────────────────────────────────────
from sklearn.linear_model import LogisticRegression

lr_sklearn = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42,
)
lr_sklearn.fit(X_train_sc, y_train)
print(f"sklearn accuracy: {lr_sklearn.score(X_test_sc, y_test):.4f}")
print(f"sklearn coef shape: {lr_sklearn.coef_.shape}")  # (1, 30) for binary

# ── Option 2: statsmodels (for inference) ─────────────────────────────────
import statsmodels.api as sm

X_train_const = sm.add_constant(X_train_sc)   # adds intercept column
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit(method='newton', disp=False)
print(result.summary().tables[1])  # table of coefficients + p-values

# ── Option 3: LogisticRegressionCV (built-in C-selection) ─────────────────
from sklearn.linear_model import LogisticRegressionCV

lr_cv = LogisticRegressionCV(
    Cs=20,          # 20 values log-spaced from 1e-4 to 1e4
    cv=5,           # 5-fold stratified CV
    penalty='l2',
    solver='lbfgs',
    scoring='roc_auc',   # optimize AUC, not accuracy
    max_iter=1000,
    random_state=42,
)
lr_cv.fit(X_train_sc, y_train)
print(f"Best C: {lr_cv.C_[0]:.4f}")
print(f"LR-CV accuracy: {lr_cv.score(X_test_sc, y_test):.4f}")

# ── Option 4: python-glmnet (regularization path) ─────────────────────────
# from glmnet import LogitNet  # pip install python-glmnet
# logitnet = LogitNet(alpha=1.0, n_splits=5)   # alpha=1 -> L1 path
# logitnet.fit(X_train_sc, y_train)
# print(f"glmnet accuracy: {logitnet.score(X_test_sc, y_test):.4f}")
# (Commented: requires Fortran build; uncomment if installed)
```

> 💡 **Intuition:** Think of these packages as different instruments measuring the same
> underlying quantity. sklearn is a production-grade digital scale — fast, reliable, Pipeline-ready.
> statsmodels is a precision analytical balance — slower, but gives you the measurement uncertainty
> (standard errors). glmnet is a spectrometer — shows you the full spectrum of coefficient paths
> as regularization varies. PyMC is a probabilistic sensor — tells you not just the value but
> your full uncertainty distribution over it. Pick the right instrument for the job.

> 🔧 **In practice:** For a new project, always start with `LogisticRegression(max_iter=1000)` in
> a `Pipeline` with `StandardScaler`. If you need p-values for a report or paper, fit the same
> model in statsmodels with `penalty=None` in sklearn first to confirm results match.

---

## §1 — The Origin: Cox (1958) and the Birth of Logistic Regression

### Before Cox: The Problem with Linear Probability Models

Imagine the year is 1957. You are a statistician at a British research institute, and you want to
model the probability that a patient survives surgery based on their age and blood pressure. The
natural tool of the era is linear regression:

$$\hat{p}_i = \beta_0 + \beta_1 \cdot \text{age}_i + \beta_2 \cdot \text{bp}_i$$

This is the **Linear Probability Model (LPM)**, and it has a catastrophic flaw: for extreme values
of age or blood pressure, $\hat{p}_i$ will exceed 1 or drop below 0. Probabilities outside [0, 1]
are mathematically incoherent. You cannot say that a 90-year-old with hypertension has a 140%
chance of survival. Even within the valid range, the linear model assumes a constant marginal
effect of, say, age on probability — but in reality, moving from age 20 to 25 has a much smaller
effect on mortality probability than moving from age 75 to 80. The effect should be S-shaped, not
linear.

The alternative known before Cox was **probit regression**, introduced by Chester Bliss in 1934
and formalized by C. I. Bliss and Ronald Fisher. Probit uses the normal CDF $\Phi$ as the link
function:

$$P(Y=1 \mid \mathbf{x}) = \Phi(\mathbf{x}^T \boldsymbol{\beta})$$

Probit has excellent theoretical justification (it arises from a latent normal variable model) but
a painful practical problem: computing $\Phi$ requires numerical tables or integration — there is
no closed-form derivative. In 1957, before electronic computers were commonplace, this was a
significant computational burden.

### Enter Verhulst, Berkson, and the Logistic Function

The mathematical object that Cox would use had been known for over a century. **Pierre-François
Verhulst** (1838, 1845) invented the logistic growth curve to model population dynamics:

$$N(t) = \frac{L}{1 + e^{-k(t - t_0)}}$$

This S-shaped curve, which Verhulst named "logistique," maps the real line to $(0, L)$. The
sigmoid shape naturally captures diminishing returns: populations grow quickly when small and slow
as they approach carrying capacity $L$. Verhulst was modeling ecology, not statistics, but the
mathematical form was identical to what would later become the logistic regression link function.

The bridge to statistics came from **Joseph Berkson** in 1944. Berkson introduced the term
**"logit"** (a portmanteau of "logistic unit") and proposed it as a simpler alternative to the
probit for bioassay analysis — experiments where you expose organisms to different doses and
measure the fraction that respond:

> 📜 **Origin/Citation:** Berkson, J. (1944). Application of the Logistic Function to Bio-Assay.
> *Journal of the American Statistical Association*, 39(227), 357–365.

Berkson argued that logit was computationally far simpler than probit: the logistic function has
a clean algebraic form and a closed-form derivative. The logit transformation (log-odds) was also
computable by hand using logarithm tables. Berkson's logit vs. probit debate was one of the
liveliest statistical controversies of the 1940s-50s.

### Cox (1958): The Founding Paper

In 1958, **David Roxbee Cox** — then at Birkbeck College, London — published the paper that
unified and formalized binary regression using the logistic link:

> 📜 **Origin/Citation:** Cox, D. R. (1958). The Regression Analysis of Binary Sequences.
> *Journal of the Royal Statistical Society: Series B (Methodological)*, 20(2), 215–232.
> DOI: https://doi.org/10.1111/j.2517-6161.1958.tb00292.x
> URL: https://academic.oup.com/jrsssb/article/20/2/215/7027376

Cox's contribution was not the logistic function itself (Verhulst, Berkson), nor the idea of
binary regression (Bliss, probit). His breakthrough was **the complete statistical framework**:
parameter estimation via maximum likelihood, a test for goodness of fit, methods for comparing
nested models, and — crucially — clear guidance on when and how to use logistic regression in
applied research.

Cox motivated the logistic function with a simple but profound observation. Suppose the odds of
success in a binary trial are multiplicative in the covariates:

$$\frac{P(Y=1 \mid x)}{P(Y=0 \mid x)} = e^{\beta_0 + \beta_1 x}$$

This is a natural multiplicative model — just as linear regression says effects on the *scale* of
the response are additive, logistic regression says effects on the *log-odds* are additive. Taking
the log of both sides:

$$\log \frac{P}{1-P} = \beta_0 + \beta_1 x$$

Solving for $P$:

$$P = \frac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}} = \sigma(\beta_0 + \beta_1 x)$$

where $\sigma(z) = 1/(1+e^{-z})$ is the **sigmoid function**. The model is linear in the log-odds
(logit scale) but nonlinear in the probability scale — exactly the S-shaped behavior we want.

### Why the Logistic Function is the *Natural* Choice

There are at least three independent theoretical justifications for the logistic link, each coming
from a completely different direction. Together they make logistic regression not a lucky choice,
but the *inevitable* choice.

**Justification 1: Maximum Entropy (Jaynes 1957)**

Suppose you observe a binary outcome $Y$ and you know the linear constraint $\mathbb{E}[Y] = \sigma(\mathbf{x}^T \boldsymbol{\beta})$. Among all probability distributions consistent with this
constraint, the **maximum entropy distribution** is the logistic model. This is the information-
theoretic argument: you are assuming as little as possible beyond what the data tells you. Any
other link function would impose additional structure not justified by the linear constraint alone.

**Justification 2: Generalized Linear Models (GLM Framework)**

Nelder and Wedderburn (1972) unified regression models by observing that many distributions from
the exponential family have a *canonical link function* — a link that makes the GLM equations
particularly clean.

> 📜 **Origin/Citation:** Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized Linear
> Models. *Journal of the Royal Statistical Society: Series A*, 135(3), 370–384.

For the Bernoulli distribution ($Y \in \{0,1\}$), the natural parameter is the log-odds:
$\theta = \log[p/(1-p)]$. The canonical link function maps the mean $\mu = p$ to this natural
parameter: $g(\mu) = \text{logit}(\mu)$. Using the canonical link produces the cleanest
optimization equations (the score equations have the simple form $\mathbf{X}^T(\mathbf{y} -
\hat{\mathbf{p}}) = \mathbf{0}$) and guarantees that the Fisher information matrix equals the
expected second derivative of the log-likelihood — a property that simplifies inference.

**Justification 3: Random Utility Model (McFadden 1974)**

In economics, Daniel McFadden showed that if an individual chooses between two options (say,
taking a car vs. taking transit), and if the unobserved utility components are independently
Gumbel-distributed, then the probability of choosing option 1 follows exactly the logistic model.
This gives logistic regression a behavioral-economics interpretation: you are modeling choice
behavior under rationality + Gumbel noise.

> 📜 **Origin/Citation:** McFadden, D. (1974). Conditional Logit Analysis of Qualitative Choice
> Behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105–142). Academic Press.
> (McFadden received the Nobel Prize in Economics in 2000 partly for this work.)

### The Key Properties of the Sigmoid

Let's carefully examine the logistic function $\sigma(z) = \frac{1}{1+e^{-z}}$:

$$\sigma(z) \in (0, 1) \quad \forall z \in \mathbb{R}$$

This is the fundamental property: outputs are always valid probabilities, unlike the LPM.

$$\sigma(0) = 0.5 \qquad \sigma(-z) = 1 - \sigma(z)$$

Symmetry around zero: if the log-odds are zero (equal likelihood of both classes), the probability
is exactly 0.5. The function is antisymmetric: flipping the sign of the input flips the
probability around 0.5.

$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

This beautiful self-referential derivative is why logistic regression is so computationally
tractable. The derivative at any point is computable from the value itself — no need for tables
or numerical integration. At $z=0$, the slope is $\sigma(0)(1-\sigma(0)) = 0.25$, the maximum
slope. The function flattens out as $|z| \to \infty$, which corresponds to the fact that
extreme log-odds values (very confident predictions) are less sensitive to further changes.

$$\lim_{z \to +\infty} \sigma(z) = 1, \qquad \lim_{z \to -\infty} \sigma(z) = 0$$

The function asymptotically approaches but never reaches 0 or 1 — it is never *certain* of any
outcome, just arbitrarily confident. This is philosophically appropriate for a probabilistic model.

### Historical Context: What Was State of the Art in 1958?

Before Cox, the practical toolkit for binary outcomes was:
1. **2×2 contingency tables** (chi-squared tests) — for one binary predictor only
2. **Probit regression** — theoretically sound but computationally demanding
3. **Linear probability models** — fast but nonsensical predictions outside [0,1]
4. **Discriminant analysis** (Fisher 1936) — assumes multivariate normality within classes

Cox's logistic regression was immediately superior: it handled multiple continuous predictors,
produced valid probability outputs, had a clean optimization algorithm (what we now call Newton-
Raphson or IRLS), and gave naturally interpretable parameters (odds ratios). It spread through
epidemiology in the 1960s, biostatistics in the 1970s, economics via McFadden's multinomial
extension, and by the 1980s was the dominant method for binary outcomes across all quantitative
sciences.

---

## §2 — The Algorithm, Deeply Explained

### The Logistic Model: Building Up from First Principles

We have $n$ training samples $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ where:
- $\mathbf{x}_i \in \mathbb{R}^p$ is the feature vector (with $p$ features)
- $y_i \in \{0, 1\}$ is the binary label (0 = negative, 1 = positive)

We want to learn a function $f(\mathbf{x}) = P(Y=1 \mid \mathbf{x})$ — the conditional
probability of the positive class given the features. Logistic regression models this as:

$$\boxed{P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{x}^T \boldsymbol{\beta} + \beta_0) = \frac{1}{1 + e^{-(\boldsymbol{\beta}^T \mathbf{x} + \beta_0)}}}$$

where $\boldsymbol{\beta} \in \mathbb{R}^p$ is the weight vector and $\beta_0 \in \mathbb{R}$ is
the bias (intercept). We often absorb the intercept by appending a 1 to $\mathbf{x}$:
$\tilde{\mathbf{x}} = [1, x_1, \ldots, x_p]^T$ and $\tilde{\boldsymbol{\beta}} = [\beta_0, \beta_1, \ldots, \beta_p]^T$,
giving $P(Y=1 \mid \mathbf{x}) = \sigma(\tilde{\mathbf{x}}^T \tilde{\boldsymbol{\beta}})$.

Complementarily, $P(Y=0 \mid \mathbf{x}) = 1 - \sigma(z) = \sigma(-z) = \frac{1}{1+e^z}$,
confirming probabilities sum to 1.

### Log-Odds and the Logit Transformation

The key insight of Cox's model is that it is **linear in the log-odds**, not linear in the
probability. Define the **odds** of class 1 as the ratio of probabilities:

$$\text{Odds}(Y=1 \mid \mathbf{x}) = \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \frac{\hat{p}}{1 - \hat{p}}$$

For example, odds of 3 means "3 times as likely to be class 1 as class 0." Odds of 0.25 means
"4 times as likely to be class 0 as class 1." Taking the natural log:

$$\log \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \text{logit}(\hat{p}) = \boldsymbol{\beta}^T \mathbf{x} + \beta_0$$

This is the **logit function** — the inverse of the sigmoid. The logistic model is linear on the
logit (log-odds) scale. This linearity has a beautiful implication for interpretation:

**A unit increase in feature $x_j$, holding all other features fixed, adds $\beta_j$ to the
log-odds, which multiplies the odds by $e^{\beta_j}$.**

This multiplicative factor $e^{\beta_j}$ is the **odds ratio (OR)** — the single most important
interpretive quantity in logistic regression. Examples:

- $\beta_j = 0.693$: $e^{0.693} \approx 2$ — feature $j$ doubles the odds per unit increase
- $\beta_j = -0.405$: $e^{-0.405} \approx 0.67$ — feature $j$ reduces odds by 33% per unit increase
- $\beta_j = 0$: $e^0 = 1$ — feature $j$ has no effect on odds

> 💡 **Intuition:** Think of odds as a betting multiplier. If the odds of disease are 1:4 (0.25),
> that means for every person with disease, 4 people don't have it. An odds ratio of 2 for smoking
> means smokers have twice the odds of disease as non-smokers. Odds ratios are multiplicative:
> if smoking has OR=2 and obesity has OR=1.5, a smoking obese person has OR = 2 × 1.5 = 3
> (relative to a non-smoking, non-obese person) — but *only if* the logistic model is correctly
> specified with no interaction terms.

### Maximum Likelihood Estimation

We want to find the $\boldsymbol{\beta}$ that makes the observed data most probable. Each
observation $y_i$ is a Bernoulli trial with success probability $\hat{p}_i = \sigma(\mathbf{x}_i^T \boldsymbol{\beta})$.
The probability of observing label $y_i$ given the features is:

$$P(Y = y_i \mid \mathbf{x}_i; \boldsymbol{\beta}) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1-y_i}$$

This is a compact way of writing: if $y_i = 1$, the probability is $\hat{p}_i$; if $y_i = 0$,
the probability is $1 - \hat{p}_i$. Assuming i.i.d. samples, the **likelihood** is the product:

$$\mathcal{L}(\boldsymbol{\beta}) = \prod_{i=1}^n \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1-y_i}$$

Taking the log (monotone transformation, preserves argmax):

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

We want to *maximize* this. Equivalently, we minimize the **negative log-likelihood**, which is
the **binary cross-entropy loss**:

$$\mathcal{L}_{\text{CE}}(\boldsymbol{\beta}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

Why the negative? By convention, optimization algorithms minimize (not maximize), so we flip the sign. The $1/n$ factor normalizes by dataset size (common in ML, not always in statistics).

> 🔬 **Deep dive:** Why is cross-entropy the right loss? There are two equivalent perspectives.
> First, MLE perspective: cross-entropy IS the negative log-likelihood; minimizing it is
> equivalent to maximum likelihood. Second, information-theoretic perspective: cross-entropy
> $H(P_\text{true}, P_\text{model}) = H(P_\text{true}) + D_\text{KL}(P_\text{true} \| P_\text{model})$.
> Since $H(P_\text{true})$ is a constant (the entropy of the true labels), minimizing cross-entropy
> is equivalent to minimizing the KL divergence between the model's predicted distribution and
> the true distribution. You are finding the $\boldsymbol{\beta}$ that makes the predicted
> distribution closest to the true distribution in information-theoretic terms. See resource [12]
> for an excellent extended explanation.

### Why There Is No Closed Form: The Fundamental Difference from Linear Regression

In linear regression, the MLE has a closed-form solution: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$. This exists because the gradient of the squared error loss with respect to $\boldsymbol{\beta}$ gives a *linear* equation. Setting it to zero can be solved algebraically.

For logistic regression, the gradient of the log-likelihood is:

$$\nabla_{\boldsymbol{\beta}} \ell = \mathbf{X}^T (\mathbf{y} - \hat{\mathbf{p}})$$

where $\hat{\mathbf{p}} = [\sigma(\mathbf{x}_1^T \boldsymbol{\beta}), \ldots, \sigma(\mathbf{x}_n^T \boldsymbol{\beta})]^T$. Setting this to zero:

$$\mathbf{X}^T (\mathbf{y} - \hat{\mathbf{p}}(\boldsymbol{\beta})) = \mathbf{0}$$

The problem is that $\hat{\mathbf{p}}(\boldsymbol{\beta})$ is a *nonlinear* function of $\boldsymbol{\beta}$ (through the sigmoid). The score equation is a system of nonlinear equations — there is no algebraic solution. We must use iterative numerical optimization.

### The Hessian: Proving Convexity

The Hessian (matrix of second partial derivatives) of the log-likelihood is:

$$\mathbf{H}(\boldsymbol{\beta}) = -\mathbf{X}^T \mathbf{W} \mathbf{X}$$

where $\mathbf{W} = \text{diag}(w_1, w_2, \ldots, w_n)$ with weights $w_i = \hat{p}_i(1 - \hat{p}_i)$.

Note that $w_i = \hat{p}_i(1-\hat{p}_i) > 0$ for all $\hat{p}_i \in (0,1)$ — each weight is
strictly positive. Therefore, for any non-zero vector $\mathbf{v}$:

$$\mathbf{v}^T \mathbf{H} \mathbf{v} = -\mathbf{v}^T \mathbf{X}^T \mathbf{W} \mathbf{X} \mathbf{v} = -\|\mathbf{W}^{1/2} \mathbf{X} \mathbf{v}\|^2 \leq 0$$

The Hessian is **negative semi-definite** (negative definite when $\mathbf{X}$ has full column rank),
which means the log-likelihood is **strictly concave** in $\boldsymbol{\beta}$. Equivalently, the
cross-entropy loss is **strictly convex**. This is a gold-standard property:

- There is exactly one global minimum (the MLE)
- No local minima to get trapped in
- Gradient descent is guaranteed to converge to the optimal solution
- The curvature (Hessian) can be used directly for optimization (Newton's method)

This convexity is what separates logistic regression from neural networks (non-convex) and makes
it the algorithm of choice when you need *guaranteed* optimal estimation.

> 💡 **Intuition:** The Hessian weights $w_i = \hat{p}_i(1-\hat{p}_i)$ have a beautiful
> interpretation. This is the **variance of a Bernoulli($\hat{p}_i$) random variable**. Points
> near the decision boundary (where $\hat{p}_i \approx 0.5$) have high variance $\approx 0.25$
> and contribute most to the curvature — they are the most informative about the direction of
> the decision boundary. Points far from the boundary ($\hat{p}_i \to 0$ or $\hat{p}_i \to 1$)
> have low variance $\to 0$ and contribute less — the model is already confident about them.

### Newton-Raphson: IRLS — The Canonical Optimization Algorithm

Newton's method for optimization updates the parameters using both the gradient and curvature
(Hessian). For the log-likelihood:

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \mathbf{H}^{-1} \nabla_{\boldsymbol{\beta}} \ell$$

Substituting our gradient and Hessian:

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + (\mathbf{X}^T \mathbf{W}^{(t)} \mathbf{X})^{-1} \mathbf{X}^T (\mathbf{y} - \hat{\mathbf{p}}^{(t)})$$

This can be rearranged into the **Iteratively Reweighted Least Squares (IRLS)** form. Define the
**adjusted response** $\mathbf{z}^{(t)} = \mathbf{X} \boldsymbol{\beta}^{(t)} + (\mathbf{W}^{(t)})^{-1}(\mathbf{y} - \hat{\mathbf{p}}^{(t)})$. Then the update is:

$$\boldsymbol{\beta}^{(t+1)} = (\mathbf{X}^T \mathbf{W}^{(t)} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W}^{(t)} \mathbf{z}^{(t)}$$

This is exactly a **weighted least squares** problem with design matrix $\mathbf{X}$, weights
$\mathbf{W}^{(t)}$, and response $\mathbf{z}^{(t)}$. At each Newton step, we solve a WLS problem
with updated weights and response. The "iterative" in IRLS refers to the fact that weights are
recomputed at each step based on current predictions.

**Convergence:** IRLS typically converges in 4–10 iterations for well-conditioned problems.
Each iteration costs $O(np^2 + p^3)$: $O(np^2)$ to form $\mathbf{X}^T \mathbf{W} \mathbf{X}$
and $O(p^3)$ to solve the linear system. For $n \gg p$ (many more samples than features), the
$O(np^2)$ term dominates.

This IRLS algorithm is exactly what the `statsmodels.Logit(method='newton')` solver uses and is
conceptually related to what `newton-cg` and `newton-cholesky` in sklearn implement.

### Gradient Descent: An Alternative Route

For large $n$ or $p$, forming and inverting the $p \times p$ Hessian is too expensive. Pure
gradient descent updates:

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + \eta \cdot \mathbf{X}^T (\mathbf{y} - \hat{\mathbf{p}}^{(t)})$$

where $\eta > 0$ is the learning rate. Gradient descent requires more iterations than Newton
(hundreds vs. single digits) but each step is much cheaper: $O(np)$ vs. $O(np^2 + p^3)$.

**L-BFGS** (the sklearn default) is a quasi-Newton method that approximates the Hessian inverse
using the history of gradient vectors, without forming the full Hessian. It combines the fast
convergence of Newton's method with the memory efficiency of gradient descent. L-BFGS typically
converges in tens to low-hundreds of iterations and scales well to millions of features.

### The Decision Boundary

A binary prediction is made by thresholding: predict class 1 if $\hat{p} > 0.5$. Since
$\sigma(z) > 0.5 \Leftrightarrow z > 0$, the decision boundary is the set of points where:

$$\boldsymbol{\beta}^T \mathbf{x} + \beta_0 = 0$$

This is a **hyperplane** in $\mathbb{R}^p$. In 2D, it is a line; in 3D, a plane. In general,
it is a $p-1$ dimensional affine subspace of $\mathbb{R}^p$.

This linearity is both a strength (simple, interpretable, low variance) and a fundamental limitation
(any problem whose true decision boundary is curved — XOR, concentric circles, most real-world
classification problems with complex feature interactions — cannot be solved exactly by logistic
regression without feature engineering).

> 💡 **Intuition:** Imagine you're in a dark room with two piles of objects — red balls and blue
> balls — and you can only separate them with a single flat sheet of paper (a hyperplane). Logistic
> regression is finding the *best* position for that sheet, optimizing probability calibration.
> If the two groups are spirals or concentric rings, no flat sheet can perfectly separate them.
> That's the fundamental limit.

### Multiclass Logistic Regression: OvR and Softmax

Binary logistic regression generalizes to $K > 2$ classes in two ways:

**One-vs-Rest (OvR):** Train $K$ separate binary classifiers, each distinguishing class $k$
from all others. At prediction time, pick the class with the highest probability:
$\hat{y} = \arg\max_k P_k(Y=k \mid \mathbf{x})$. Simple, parallelizable, but the $K$ classifiers
are trained independently — there is no guarantee that the $K$ probabilities sum to 1.

**Multinomial (Softmax):** Use a single joint model with $K$ weight vectors $\boldsymbol{\beta}_1, \ldots, \boldsymbol{\beta}_K$:

$$P(Y=k \mid \mathbf{x}) = \frac{e^{\mathbf{x}^T \boldsymbol{\beta}_k}}{\sum_{j=1}^K e^{\mathbf{x}^T \boldsymbol{\beta}_j}}$$

This is the **softmax function** — a generalization of the sigmoid to $K$ classes. The
probabilities sum to 1 by construction. The loss is the multiclass cross-entropy:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K \mathbf{1}[y_i=k] \log P(Y=k \mid \mathbf{x}_i)$$

The softmax model has $K \times p$ parameters but is **overparameterized**: you can add any
constant vector to all $\boldsymbol{\beta}_k$ without changing the predictions (because the
softmax depends only on differences). In practice, this is handled by regularization or by
fixing $\boldsymbol{\beta}_K = \mathbf{0}$ (the reference class convention).

**sklearn (1.5+) defaults:** The `multi_class` parameter is deprecated in sklearn 1.5 and will
be removed in 1.7. For $K \geq 3$, sklearn now defaults to multinomial (softmax) for all solvers
that support it. For OvR, use `OneVsRestClassifier(LogisticRegression())` explicitly.

> ⚠️ **Pitfall:** If you are using `multi_class='ovr'` or `multi_class='multinomial'` in your
> code, you will get `DeprecationWarning` in sklearn 1.5 and your code **will break** in sklearn
> 1.7. Update your code now. See https://scikit-learn.org/1.5/whats_new/v1.5.html.

### Regularization: L1, L2, and ElasticNet

Without regularization, the MLE $\hat{\boldsymbol{\beta}}$ is the unique maximum of the concave
log-likelihood. But for large $p$ (many features) or when features are correlated, the MLE can
have high variance — small changes in the training data lead to large changes in coefficients.
Regularization adds a **penalty** to the loss that shrinks coefficients toward zero:

$$\mathcal{L}_{\text{reg}}(\boldsymbol{\beta}) = \underbrace{-\frac{1}{n} \sum_i \ell_i}_{\text{data fit}} + \underbrace{\frac{1}{C} \cdot r(\boldsymbol{\beta})}_{\text{penalty}}$$

where $C > 0$ is sklearn's inverse regularization parameter and $r(\boldsymbol{\beta})$ is the
regularizer. Note that sklearn writes this with $C = 1/\lambda$, so larger $C$ means weaker
regularization (more like unregularized MLE) and smaller $C$ means stronger regularization.

**L2 (Ridge):** $r(\boldsymbol{\beta}) = \frac{1}{2}\|\boldsymbol{\beta}\|_2^2 = \frac{1}{2}\sum_j \beta_j^2$

The L2 penalty has a smooth, differentiable gradient that scales with $\beta_j$: large coefficients
incur a large gradient penalty pushing them toward zero. L2 *shrinks* all coefficients but never
makes any exactly zero. It is equivalent to placing an **isotropic Gaussian prior** $\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, C\mathbf{I})$ and taking the MAP estimate. L2 solves the
multicollinearity problem by distributing weight among correlated features.

**L1 (Lasso):** $r(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_1 = \sum_j |\beta_j|$

The L1 penalty is non-differentiable at $\beta_j = 0$ but produces **exact sparsity**: many
coefficients become exactly zero (not just small — exactly zero). L1 performs automatic feature
selection. It is equivalent to a **Laplace prior** $\beta_j \sim \text{Laplace}(0, C)$. The
L1 penalty requires coordinate descent or subgradient methods (sklearn: `solver='liblinear'` or
`solver='saga'`).

**ElasticNet:** $r(\boldsymbol{\beta}) = \rho\|\boldsymbol{\beta}\|_1 + \frac{1-\rho}{2}\|\boldsymbol{\beta}\|_2^2$

Combines L1 (sparsity) and L2 (handles correlated features). The `l1_ratio` hyperparameter in
sklearn controls $\rho$: 0 → pure L2, 1 → pure L1. Requires `solver='saga'`.

**None (unregularized):** Pure MLE. Use `penalty=None`. Available with `lbfgs`, `newton-cg`,
`newton-cholesky`, `sag`, `saga`. Equivalent to `statsmodels.Logit` default.

> 🔧 **In practice:** Start with L2 (the default). Switch to L1 if you suspect many features
> are irrelevant and want a sparse model. Use ElasticNet when features are correlated AND you
> want sparsity. Use `penalty=None` only if you have very good reason to believe you have enough
> data and no overfitting risk, or when you specifically need statistical inference (p-values).

### The Six Solvers: When to Use Each

sklearn implements six optimization algorithms for logistic regression. Choosing the right one
depends on dataset size, required penalty, and precision needs:

| Solver | Penalty support | Algorithm | Best for |
|---|---|---|---|
| `lbfgs` | L2, None | L-BFGS quasi-Newton | **Default; medium-large datasets; most use cases** |
| `liblinear` | L1, L2 | Coordinate descent | Small datasets (<10^4); binary classification |
| `newton-cg` | L2, None | Conjugate gradient | Medium datasets; high precision |
| `newton-cholesky` | L2, None | Newton + Cholesky factorization | $n \gg p$; $p < 10^4$ |
| `sag` | L2, None | Stochastic Average Gradient | Large datasets; requires scaling |
| `saga` | L1, L2, EN, None | SAGA (variance-reduced SGD) | Large datasets; all penalties |

**lbfgs** (the default): L-BFGS stores the last $m$ gradient vectors to approximate the inverse
Hessian. Very memory efficient and typically converges in tens of iterations. Can't handle L1
(non-smooth). For most classification problems with scaled features, this is the right choice.

**liblinear**: Uses coordinate descent on the dual problem. Extremely fast for small $n$ and
binary classification. The only solver supporting L1 for small data. Does not support multinomial
loss natively (implements OvR internally). Has an `intercept_scaling` quirk: when
`fit_intercept=True`, sklearn appends a column of `intercept_scaling` to $\mathbf{X}$ and
regularizes it alongside other features. If `intercept_scaling=1` (default), the intercept
gets the same L2 regularization as other features — potentially problematic for very large or
very small intercepts.

**saga**: A variance-reduced stochastic gradient descent algorithm. Each update uses a random
sample gradient *corrected by a stored gradient table* to reduce variance. This gives
convergence guarantees for L1 (non-smooth) penalties, unlike vanilla SGD. The memory cost is
$O(np)$ (storing one gradient per training sample) — be mindful for very large $n$.

**newton-cholesky** (sklearn 1.2+): Solves the full Newton system using Cholesky decomposition
of the $p \times p$ Hessian matrix. Extremely accurate but memory cost is $O(p^2)$ in features.
Use for $n \gg p$ (e.g., $n = 10^6$, $p = 50$). Does not support multinomial loss or L1.

### Computational Complexity

| Operation | Complexity |
|---|---|
| Training (full Newton, IRLS) | $O(T \cdot (np^2 + p^3))$ where $T \sim 4$–$10$ iterations |
| Training (L-BFGS) | $O(T \cdot np)$ where $T \sim 100$ iterations |
| Training (saga/sag) | $O(T \cdot p)$ per iteration, $T$ can be large |
| Inference (single sample) | $O(p)$ — just a dot product and sigmoid |
| Inference (batch of $n$) | $O(np)$ |

Logistic regression is one of the fastest algorithms at inference time: a single prediction is
a dot product followed by a scalar sigmoid — trivially parallelizable on any hardware. This
is why logistic regression is deployed in latency-critical production systems (real-time ad
bidding, fraud detection at transaction time) where tree ensembles would be too slow.

### What the Model Assumes

Logistic regression makes several implicit assumptions. Violating them doesn't make the model
*wrong*, but it degrades interpretability and coefficient reliability:

1. **Linearity in log-odds:** The logit of the conditional probability is a linear function of
   the features. Violation: nonlinear feature effects. Mitigation: polynomial features,
   splines, interaction terms.

2. **No perfect multicollinearity:** If features are linearly dependent, the design matrix is
   singular and MLE doesn't exist (or is numerically unstable). Near-multicollinearity inflates
   coefficient variances (large standard errors, unstable estimates). L2 regularization shrinks
   the condition number, stabilizing estimates.

3. **Independence of observations:** The log-likelihood assumes i.i.d. samples. Violation:
   clustered data (multiple observations per patient, longitudinal data). Mitigation: mixed-
   effects logistic regression, GEEs, clustered standard errors.

4. **No perfect or quasi-complete separation:** If a linear combination of features perfectly
   predicts the outcome in the training data, MLE doesn't exist (coefficients → ∞). Mitigation:
   L2 regularization (most practical), Firth's penalized likelihood, data augmentation.

5. **Sufficient sample size per predictor:** A rule of thumb from statistics literature is
   "at least 10 events per predictor variable" (EPV ≥ 10). For rare outcomes, this can require
   very large datasets. With small EPV, estimates are biased (see: Firth's method).

```python
# Demonstrating the model's components
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate 2D data for visualization
X_vis, y_vis = make_classification(
    n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    random_state=42, class_sep=1.2
)

clf_vis = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
clf_vis.fit(X_vis, y_vis)

print("Coefficients (β):", clf_vis.coef_[0])
print("Intercept (β₀):", clf_vis.intercept_[0])

# The decision boundary: β₀ + β₁x₁ + β₂x₂ = 0
# → x₂ = -(β₀ + β₁x₁) / β₂
beta = clf_vis.coef_[0]
beta0 = clf_vis.intercept_[0]

x1_range = np.linspace(X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1, 100)
x2_boundary = -(beta0 + beta[0] * x1_range) / beta[1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Decision boundary
ax = axes[0]
colors = ['#2196F3', '#F44336']
for cls, color in zip([0, 1], colors):
    mask = y_vis == cls
    ax.scatter(X_vis[mask, 0], X_vis[mask, 1], c=color, alpha=0.6, s=30,
               label=f'Class {cls}')
ax.plot(x1_range, x2_boundary, 'k-', lw=2, label='Decision boundary (p=0.5)')

# Shading confidence contours
xx1, xx2 = np.meshgrid(np.linspace(X_vis[:,0].min()-1, X_vis[:,0].max()+1, 200),
                        np.linspace(X_vis[:,1].min()-1, X_vis[:,1].max()+1, 200))
Z = clf_vis.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:, 1].reshape(xx1.shape)
ax.contourf(xx1, xx2, Z, alpha=0.15, cmap='RdBu_r', levels=20)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Logistic Regression: Linear Decision Boundary')
ax.legend()

# Plot 2: Sigmoid function
ax = axes[1]
z = np.linspace(-6, 6, 200)
sigma = 1 / (1 + np.exp(-z))
ax.plot(z, sigma, 'b-', lw=2.5, label=r'$\sigma(z) = 1/(1+e^{-z})$')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
ax.fill_between(z, sigma, 0.5, where=z > 0, alpha=0.1, color='blue')
ax.fill_between(z, sigma, 0.5, where=z < 0, alpha=0.1, color='red')
ax.set_xlabel('Log-odds (z = βᵀx + β₀)')
ax.set_ylabel('Probability P(Y=1|x)')
ax.set_title('The Sigmoid Function')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression_basics.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nOdds ratios: {np.exp(clf_vis.coef_[0])}")
```

---

## §3 — The Full Evolution: From Cox (1958) to Modern Variants

Logistic regression has been continuously enriched over 65 years. Each variant emerged to solve
a specific failure mode of the original. Understanding this evolution means understanding *when*
to use each tool.

### 3a — Penalized Logistic Regression: L1, L2, and ElasticNet (Tibshirani 1996; Ng 2004)

**Problem solved:** The original Cox (1958) formulation used pure MLE — no regularization. This
works well when $n \gg p$ and features are well-separated, but fails in the modern high-dimensional
setting (gene expression data with $p = 20,000$ features and $n = 200$ samples; text classification
with $p = 10^6$ features; ad-click prediction with $p = 10^9$ sparse features).

Without regularization, MLE overfits badly when $p$ is large relative to $n$. Worse, when any
linear combination of features perfectly separates the training classes (perfect separation),
MLE doesn't exist at all — coefficients diverge to infinity.

**The Ridge solution (L2):** Robert Hoerl and Robert Kennard (1970) introduced ridge regression
for linear models; Andrew Ng (2004) and others showed its benefit for logistic regression.
Ridge adds $\frac{1}{2C}\|\boldsymbol{\beta}\|_2^2$ to the loss. The solution is unique even
under multicollinearity, and the L2 penalty has a clean Bayesian interpretation: it is the MAP
estimate under a $\mathcal{N}(0, C\mathbf{I})$ prior on the weights.

**The Lasso solution (L1):** Robert Tibshirani's 1996 paper introduced Lasso (Least Absolute
Shrinkage and Selection Operator) for linear regression:

> 📜 **Origin/Citation:** Tibshirani, R. (1996). Regression Shrinkage and Selection via the
> Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.

L1 regularization for logistic regression adds $\frac{1}{C}\|\boldsymbol{\beta}\|_1$ to the
loss. The key difference from L2: the L1 penalty is *not differentiable* at zero, and the
subgradient condition at $\beta_j = 0$ can be satisfied by the data gradient alone, making
the coefficient *exactly* zero rather than merely small. This geometric fact (the L1 ball has
corners at the coordinate axes; the solution often lands on a corner) is why Lasso produces
sparsity.

**The ElasticNet:** Zou and Hastie (2005) introduced ElasticNet:

> 📜 **Origin/Citation:** Zou, H., & Hastie, T. (2005). Regularization and Variable Selection
> via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.

ElasticNet combines L1 and L2: $r = \rho\|\boldsymbol{\beta}\|_1 + \frac{1-\rho}{2}\|\boldsymbol{\beta}\|_2^2$.
It addresses a key weakness of Lasso: when features are highly correlated, Lasso arbitrarily
selects one from a correlated group. ElasticNet tends to include or exclude the whole group together.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Demonstrating regularization effects on coefficients
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target

# Scale the data (CRITICAL for regularization to work properly)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Compare regularization types
C = 0.1  # moderate regularization
configs = [
    ('L2 Ridge', dict(penalty='l2', C=C, solver='lbfgs')),
    ('L1 Lasso', dict(penalty='l1', C=C, solver='liblinear')),
    ('ElasticNet', dict(penalty='elasticnet', C=C, solver='saga', l1_ratio=0.5)),
    ('No penalty', dict(penalty=None, solver='lbfgs')),
]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, params) in zip(axes.flatten(), configs):
    clf = LogisticRegression(max_iter=1000, random_state=42, **params)
    clf.fit(X_sc, y)
    coefs = clf.coef_[0]
    colors = ['#2196F3' if c > 0 else '#F44336' for c in coefs]
    ax.bar(range(len(coefs)), coefs, color=colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    n_zero = np.sum(np.abs(coefs) < 1e-6)
    ax.set_title(f'{name} (C={C})\n{n_zero} zero coefficients out of {len(coefs)}')
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Coefficient value')
    ax.tick_params(axis='x', labelsize=7)
plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3b — The Regularization Path: glmnet (Friedman, Hastie, Tibshirani 2010)

**Problem solved:** Fitting a single model for one value of $C$ tells you nothing about how
the model changes as regularization varies. To understand feature importance and select $C$,
you want to see the entire **regularization path** — how each coefficient evolves as $C$
sweeps from $\infty$ (unregularized MLE) to $0$ (all coefficients zero).

Naive solution: fit 100 models at 100 values of $C$. This is expensive. The glmnet algorithm
(Friedman, Hastie, Tibshirani 2010) computes the entire L1 path efficiently using coordinate
descent with warm-starts and active-set screening, making the full path computation only 2–10x
more expensive than fitting a single model:

> 📜 **Origin/Citation:** Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization
> Paths for Generalized Linear Models via Coordinate Descent. *Journal of Statistical Software*,
> 33(1), 1–22. https://www.jstatsoft.org/article/view/v033i01

```python
# Regularization path visualization with sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X_sc = StandardScaler().fit_transform(data.data)
y = data.target

# Sweep C values (large C = weak regularization)
Cs = np.logspace(-4, 2, 50)
coef_paths = []

for C in Cs:
    clf = LogisticRegression(penalty='l1', C=C, solver='liblinear',
                              max_iter=1000, random_state=42)
    clf.fit(X_sc, y)
    coef_paths.append(clf.coef_[0].copy())

coef_paths = np.array(coef_paths)  # shape (50, 30)

plt.figure(figsize=(12, 6))
for j in range(coef_paths.shape[1]):
    plt.plot(np.log10(Cs), coef_paths[:, j], alpha=0.6, linewidth=1)
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='C=1 (default)')
plt.xlabel('log₁₀(C)  →  stronger regularization (left) to weaker (right)')
plt.ylabel('Coefficient value')
plt.title('L1 Regularization Path — Breast Cancer Dataset\n'
          '(Each line = one feature; lines reaching zero = Lasso elimination)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('l1_regularization_path.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3c — SGD Logistic Regression: Online and Large-Scale Learning (Bottou 1998, 2010)

**Problem solved:** Standard logistic regression requires loading all data into memory and
computing the full gradient at each step — infeasible for streaming data or datasets with
$n > 10^7$ samples.

Léon Bottou's work on stochastic gradient descent for large-scale ML showed that approximating
the gradient with a single randomly sampled term (or mini-batch) is sufficient for convergence
with a decaying learning rate:

> 📜 **Origin/Citation:** Bottou, L. (2010). Large-Scale Machine Learning with Stochastic
> Gradient Descent. *COMPSTAT 2010 Proceedings*, 177–186.

In sklearn, `SGDClassifier(loss='log_loss')` implements logistic regression via SGD. The
`partial_fit()` method enables true **online learning**: the model can be updated one batch at
a time, making it suitable for streaming applications where data arrives continuously.

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Online learning demo: fit in mini-batches
sgd_lr = SGDClassifier(
    loss='log_loss',       # logistic regression (was 'log' before sklearn 1.1)
    penalty='l2',
    alpha=1e-4,            # regularization (note: alpha, NOT C like LogisticRegression)
    learning_rate='optimal',  # decays as 1/(alpha * (t + t0))
    max_iter=1,            # each partial_fit is one epoch
    random_state=42,
)

# Simulate streaming: feed data in chunks of 100
batch_size = 100
classes = np.array([0, 1])
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X_sc = StandardScaler().fit_transform(data.data)
y = data.target

for i in range(0, len(X_sc) - batch_size, batch_size):
    X_batch = X_sc[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    sgd_lr.partial_fit(X_batch, y_batch, classes=classes)

# Final evaluation
from sklearn.metrics import roc_auc_score
proba = sgd_lr.predict_proba(X_sc[-batch_size:])[:, 1]
print(f"SGD LR AUC (holdout): {roc_auc_score(y[-batch_size:], proba):.4f}")

# Key difference: regularization convention
# LogisticRegression: C = 1.0 means regularization strength = 1/C = 1.0
# SGDClassifier: alpha = 0.0001 means regularization strength = alpha directly
# Roughly: alpha ≈ 1 / (n_samples * C)
# For breast_cancer (n=569, C=1.0): alpha ≈ 1/(569*1.0) ≈ 0.00176
```

> ⚠️ **Pitfall:** The `loss='log'` string was deprecated in sklearn 1.1 and **removed in
> sklearn 1.3**. If you have old code using `loss='log'`, it will fail with a ValueError. Use
> `loss='log_loss'` always.

> ⚠️ **Pitfall:** `SGDClassifier` uses `alpha` (direct regularization coefficient) while
> `LogisticRegression` uses `C` (inverse). They are NOT the same thing.
> `alpha ≈ 1 / (n_samples * C)`. Mixing them up leads to wildly different regularization levels.

### 3d — Generalized Linear Models (Nelder & Wedderburn 1972)

**Problem solved:** After Cox (1958), practitioners needed a unified framework connecting logistic
regression to Poisson regression, Gamma regression, and other models for non-Gaussian outcomes.

Nelder and Wedderburn (1972) created the GLM framework, showing that all these models share
a common structure:

> 📜 **Origin/Citation:** Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized Linear
> Models. *Journal of the Royal Statistical Society: Series A*, 135(3), 370–384.

A GLM consists of three components:
1. **Random component:** $Y \mid \mathbf{x}$ follows a distribution from the exponential family
   (Bernoulli, Poisson, Gamma, Normal, etc.)
2. **Systematic component:** Linear predictor $\eta = \mathbf{x}^T \boldsymbol{\beta}$
3. **Link function:** $g(\mu) = \eta$, where $\mu = \mathbb{E}[Y \mid \mathbf{x}]$

For logistic regression:
- Random: $Y \sim \text{Bernoulli}(p)$
- Systematic: $\eta = \mathbf{x}^T \boldsymbol{\beta}$
- Link: $g(p) = \text{logit}(p) = \log[p/(1-p)]$ — the canonical link for Bernoulli

The canonical link is special: it makes the score equations take the simple form $\nabla_{\boldsymbol{\beta}} \ell = \mathbf{X}^T(\mathbf{y} - \hat{\boldsymbol{\mu}})$, and the Fisher information equals the observed information.

```python
# GLM logistic regression via statsmodels (the canonical GLM implementation)
import statsmodels.api as sm

data_sm = load_breast_cancer()
X_sm = StandardScaler().fit_transform(data_sm.data)
y_sm = data_sm.target
X_sm_const = sm.add_constant(X_sm)

# GLM with Binomial family and logit link (identical to Logit but more general API)
glm_model = sm.GLM(
    y_sm,
    X_sm_const,
    family=sm.families.Binomial(link=sm.families.links.Logit())
)
glm_result = glm_model.fit()
print(glm_result.summary().tables[0])  # model-level statistics
print(f"Deviance: {glm_result.deviance:.4f}")
print(f"AIC: {glm_result.aic:.4f}")
print(f"Pearson chi2: {glm_result.pearson_chi2:.4f}")
```

### 3e — Multinomial Logistic Regression / Softmax (McFadden 1974)

**Problem solved:** Binary logistic regression only models two classes. Extending to $K > 2$
classes in a principled, jointly optimal way requires the multinomial (softmax) model.

Daniel McFadden formalized the multinomial logit model in economics as the "conditional logit"
model, grounded in random utility theory. His work showed that if the unobserved utility is
i.i.d. Gumbel (Type I extreme value) distributed, the resulting choice probabilities are exactly
the softmax:

$$P(Y=k \mid \mathbf{x}) = \frac{e^{\mathbf{x}^T \boldsymbol{\beta}_k}}{\sum_{j=1}^K e^{\mathbf{x}^T \boldsymbol{\beta}_j}}$$

In sklearn 1.5+, the `multi_class` parameter is deprecated. The recommendation is:
- **For multinomial (softmax):** `LogisticRegression()` with `lbfgs`, `newton-cg`, `sag`, or
  `saga` — these automatically use softmax for $K \geq 3$
- **For OvR:** `OneVsRestClassifier(LogisticRegression())`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_iris, y_iris = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.2,
                                            stratify=y_iris, random_state=42)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

# Multinomial (softmax) — jointly models all classes
lr_softmax = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',        # lbfgs supports multinomial
    max_iter=1000,
    random_state=42,
)
lr_softmax.fit(X_tr_sc, y_tr)
print(f"Softmax coef shape: {lr_softmax.coef_.shape}")  # (3, 4) — 3 classes, 4 features
print(f"Softmax accuracy: {lr_softmax.score(X_te_sc, y_te):.4f}")

# OvR — 3 independent binary classifiers
lr_ovr = OneVsRestClassifier(
    LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
)
lr_ovr.fit(X_tr_sc, y_tr)
print(f"OvR accuracy: {lr_ovr.score(X_te_sc, y_te):.4f}")
```

### 3f — Mixed-Effects / Hierarchical Logistic Regression

**Problem solved:** In medical, educational, and social science research, observations are
not i.i.d. — they are grouped (students in schools, patients in hospitals, transactions
by customer). Standard logistic regression ignores this group structure, leading to
anti-conservative standard errors (standard errors that are too small, inflating significance).

**Mixed-effects logistic regression** (also called multilevel or hierarchical LR) adds
random effects to the model:

$$\text{logit}(P(Y=1 \mid \mathbf{x}_i, \mathbf{z}_i)) = \mathbf{x}_i^T \boldsymbol{\beta} + \mathbf{z}_i^T \mathbf{u}_j$$

where $\mathbf{u}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{G})$ are random effects for group $j$.
Fixed effects $\boldsymbol{\beta}$ capture population-average effects; random effects $\mathbf{u}_j$
capture group-level deviations.

```python
# Mixed-effects logistic regression via statsmodels
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Simulated grouped data: 100 groups, 20 observations per group
np.random.seed(42)
n_groups = 50
n_per_group = 20
n = n_groups * n_per_group

group_ids = np.repeat(np.arange(n_groups), n_per_group)
x1 = np.random.randn(n)
group_intercepts = np.random.randn(n_groups) * 0.8  # random intercepts per group
log_odds = 0.5 * x1 + group_intercepts[group_ids]
y_sim = (np.random.rand(n) < 1 / (1 + np.exp(-log_odds))).astype(int)

df_sim = pd.DataFrame({'y': y_sim, 'x1': x1, 'group': group_ids.astype(str)})

# Mixed-effects logistic regression
melogit = smf.mixedlm(
    "y ~ x1",
    df_sim,
    groups=df_sim["group"],
)
result_me = melogit.fit(method='powell', disp=False)  # # verify in current docs
print(result_me.summary())
# Note: statsmodels MixedLM fits a LMM (linear), not logistic mixed model.
# For true mixed-effects logistic regression, use R lme4, or
# BayesLR via PyMC (Section 3g), or statsmodels BinomialBayesMixedGLM
```

### 3g — Bayesian Logistic Regression (1990s–present)

**Problem solved:** Standard logistic regression gives a point estimate $\hat{\boldsymbol{\beta}}$
but says nothing about uncertainty in the coefficients. Confidence intervals from statsmodels
are based on asymptotic normal approximations, which can be poor for small samples or when
priors matter. Bayesian logistic regression puts a prior over $\boldsymbol{\beta}$ and computes
the posterior distribution:

$$P(\boldsymbol{\beta} \mid \mathbf{X}, \mathbf{y}) \propto P(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\beta}) \cdot P(\boldsymbol{\beta})$$

The posterior predictive distribution gives principled uncertainty over predictions, not just
a point estimate. This is especially valuable for:
- Small datasets (where asymptotic approximations fail)
- High-stakes decisions requiring calibrated uncertainty
- Hierarchical models with partial pooling

```python
# Bayesian Logistic Regression with PyMC
# pip install pymc  # requires version ~5.x
import numpy as np
try:
    import pymc as pm
    import arviz as az

    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    data_bc = load_breast_cancer()
    # Use only first 5 features for a manageable Bayesian model
    X_bay = StandardScaler().fit_transform(data_bc.data[:, :5])
    y_bay = data_bc.target

    with pm.Model() as logistic_model:
        # Priors: weakly informative Normal priors on coefficients
        beta0 = pm.Normal('intercept', mu=0, sigma=5)
        beta = pm.Normal('coefficients', mu=0, sigma=2, shape=X_bay.shape[1])

        # Likelihood
        log_odds_bay = beta0 + pm.math.dot(X_bay, beta)
        p = pm.math.sigmoid(log_odds_bay)
        obs = pm.Bernoulli('obs', p=p, observed=y_bay)

        # Inference via NUTS (No-U-Turn Sampler)
        trace = pm.sample(
            draws=1000,
            tune=500,
            chains=2,
            target_accept=0.9,
            random_seed=42,
            progressbar=True,
        )

    # Posterior summary
    print(az.summary(trace, var_names=['intercept', 'coefficients']))
    az.plot_posterior(trace, var_names=['coefficients'])

except ImportError:
    print("PyMC not installed. Install with: pip install pymc arviz")
```

### 3h — Firth's Penalized MLE for Rare Events and Separation (Firth 1993)

**Problem solved:** When a predictor perfectly or near-perfectly separates the two classes
in the training data (complete or quasi-complete separation), the MLE for logistic regression
does not exist — the log-likelihood keeps increasing as the coefficient for the separating
variable grows to $\pm\infty$. This is an extremely common problem with rare events (e.g.,
modeling mortality in ICU patients where "age > 90 AND critical diagnosis" → always dies)
or small datasets.

David Firth (1993) proposed a penalized log-likelihood that adds the Jeffreys invariant prior:

$$\ell^*(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) + \frac{1}{2} \log |\mathbf{I}(\boldsymbol{\beta})|$$

where $|\mathbf{I}(\boldsymbol{\beta})| = |\mathbf{X}^T \mathbf{W} \mathbf{X}|$ is the
determinant of the Fisher information matrix. This penalty "pushes" the solution away from
infinity even under separation, producing finite estimates with reduced bias.

> 📜 **Origin/Citation:** Firth, D. (1993). Bias Reduction of Maximum Likelihood Estimates.
> *Biometrika*, 80(1), 27–38.

```python
# Detecting separation and the practical fix in sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

# Simulate perfect separation
np.random.seed(42)
n = 100
X_sep = np.random.randn(n, 2)
# Perfectly separating: class = 1 if x[0] > 0
y_sep = (X_sep[:, 0] > 0).astype(int)

# sklearn's response: heavy coefficient → sign of separation problem
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    clf_nosep = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
    clf_nosep.fit(X_sep, y_sep)
    print(f"Unregularized coef[0]: {clf_nosep.coef_[0, 0]:.2f}")  # will be very large

# sklearn's L2 fix: regularization forces finite solution
clf_sep_l2 = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', max_iter=1000)
clf_sep_l2.fit(X_sep, y_sep)
print(f"L2 regularized coef[0] (C=0.01): {clf_sep_l2.coef_[0, 0]:.4f}")  # finite

# For Firth's method in Python, use the 'firthlogist' package:
# pip install firthlogist  # # verify in current docs
try:
    from firthlogist import FirthLogisticRegression
    firth_lr = FirthLogisticRegression(max_iter=1000)
    firth_lr.fit(X_sep, y_sep)
    print(f"Firth coef[0]: {firth_lr.coef_[0]:.4f}")  # finite, less biased
except ImportError:
    print("firthlogist not installed. The practical workaround: use L2 with small C.")

# Detection heuristics
def detect_separation(clf):
    """Returns True if any coefficient is suspiciously large (sign of separation)."""
    return np.any(np.abs(clf.coef_) > 50)

print(f"\nSeparation detected (unregularized)?: {detect_separation(clf_nosep)}")
print(f"Separation detected (L2 C=0.01)?: {detect_separation(clf_sep_l2)}")
```

### 3i — Calibration and the Historical Role of Logistic Regression (Platt 1999)

**Problem solved:** Most classifiers (SVMs, random forests, neural networks) are not inherently
well-calibrated — their output scores don't directly correspond to probabilities. John Platt
(1999) showed that fitting a logistic regression model to the output scores of an SVM produces
well-calibrated probabilities. This is called "Platt scaling":

$$P(Y=1 \mid f_\text{SVM}(\mathbf{x})) = \sigma(A \cdot f_\text{SVM}(\mathbf{x}) + B)$$

where $A$ and $B$ are fit by logistic regression on the SVM scores. This places logistic
regression in a historically unusual position: it is the algorithm that *other* classifiers
use to fix their calibration.

Why is logistic regression well-calibrated in the first place? Because it directly minimizes
cross-entropy loss, which is a **strictly proper scoring rule**. A proper scoring rule is
maximized (in expectation) when the predicted probabilities equal the true probabilities.
Minimizing cross-entropy is therefore equivalent to matching predicted probabilities to true
empirical probabilities — calibration is built into the objective function.

Calibration can degrade for logistic regression when:
1. **Strong regularization (small C):** Pushes probabilities away from 0 and 1, understating
   confidence for extreme predictions
2. **Class imbalance:** The model's implicit prior assumes equal class frequencies
3. **Distribution shift:** Training and test distributions differ

```python
# Calibration comparison: LR vs Random Forest vs SVM
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = load_breast_cancer()
X_cal, y_cal = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X_cal, y_cal, test_size=0.3,
                                            stratify=y_cal, random_state=42)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

lr_cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
rf_cal = RandomForestClassifier(n_estimators=100, random_state=42)
svm_cal = SVC(probability=False, kernel='rbf', random_state=42)  # raw SVM
svm_platt = CalibratedClassifierCV(SVC(kernel='rbf', random_state=42), method='sigmoid')

for clf in [lr_cal, rf_cal, svm_platt]:
    clf.fit(X_tr_s, y_tr)

fig, ax = plt.subplots(figsize=(8, 7))
CalibrationDisplay.from_estimator(lr_cal, X_te_s, y_te, n_bins=10, ax=ax,
                                   name='Logistic Regression')
CalibrationDisplay.from_estimator(rf_cal, X_te_s, y_te, n_bins=10, ax=ax,
                                   name='Random Forest')
CalibrationDisplay.from_estimator(svm_platt, X_te_s, y_te, n_bins=10, ax=ax,
                                   name='SVM + Platt (LR) calibration')
ax.set_title('Calibration Curves — Breast Cancer Dataset\n'
             '(Diagonal = perfect calibration)')
plt.tight_layout()
plt.savefig('calibration_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3j — The Neural Logistic Regression: Softmax as Output Layer

The connection between logistic regression and deep learning is not incidental — it is structural.
The output layer of virtually every classification neural network IS logistic regression:

- **Binary classification:** The final layer produces a single scalar $z = \mathbf{w}^T \mathbf{h} + b$
  (where $\mathbf{h}$ is the last hidden layer) and applies sigmoid: $P(Y=1) = \sigma(z)$.
  The cross-entropy loss is minimized. This IS logistic regression on features $\mathbf{h}$.

- **Multiclass classification:** The final layer produces $K$ scalars $\mathbf{z} = \mathbf{W}\mathbf{h} + \mathbf{b}$
  and applies softmax. The cross-entropy loss is minimized. This IS multinomial logistic regression
  on features $\mathbf{h}$.

The key insight: deep learning is hierarchical feature learning followed by logistic regression
on the learned features. When you train a neural network end-to-end, you are jointly:
1. Learning a nonlinear feature transformation $\phi(\mathbf{x}) = \mathbf{h}$
2. Learning a logistic regression classifier on $\mathbf{h}$

This is also the basis of **transfer learning**: take a pretrained network's penultimate layer
activations as features, then fit a simple logistic regression on your downstream task.

```python
# Transfer learning with logistic regression as the classification head
# This is standard practice for fine-tuning pretrained models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Simulate pretrained features (in practice, from ResNet, BERT, etc.)
np.random.seed(42)
n_train, n_test, n_features = 500, 100, 512  # typical BERT-base hidden dim
X_pretrained_train = np.random.randn(n_train, n_features)
X_pretrained_test = np.random.randn(n_test, n_features)
y_transfer_train = (X_pretrained_train[:, 0] + X_pretrained_train[:, 1] > 0).astype(int)
y_transfer_test = (X_pretrained_test[:, 0] + X_pretrained_test[:, 1] > 0).astype(int)

# Logistic regression as a linear probe on pretrained features
linear_probe = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
sc_transfer = StandardScaler()
X_tr_tf = sc_transfer.fit_transform(X_pretrained_train)
X_te_tf = sc_transfer.transform(X_pretrained_test)
linear_probe.fit(X_tr_tf, y_transfer_train)
print(f"Linear probe accuracy: {linear_probe.score(X_te_tf, y_transfer_test):.4f}")
print("(In practice, replace random features with BERT/ResNet embeddings)")
```

---

## §4 — Hyperparameters: The Complete Guide

This section is your tuning reference. We cover every parameter of `sklearn.linear_model.LogisticRegression`
(version 1.5.x), with the mechanism, default, interaction effects, and tuning strategy for each.

### 4.1 — `C` (Inverse Regularization Strength)

**What it controls:** $C = 1/\lambda$ where $\lambda$ is the regularization coefficient in
the penalized objective $\mathcal{L}_\text{reg} = \mathcal{L}_\text{CE} + \frac{\lambda}{2}\|\boldsymbol{\beta}\|^2$.
A smaller $C$ means a larger penalty $\lambda$ — stronger regularization.

**Default:** `C=1.0` — This is a moderate L2 regularization. Not too strong, not too weak.
It's a reasonable starting point but rarely the optimal value for a specific dataset.

**Effect when too small (C → 0):** Very strong regularization → all coefficients shrink
toward zero → model is essentially predicting the marginal class probability (prevalence)
for every sample → high bias, low variance, underfitting.

**Effect when too large (C → ∞):** No regularization → pure MLE → overfits when $p$ is
large relative to $n$, diverges under separation.

**Tuning strategy:**
```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      stratify=y, random_state=42)
sc = StandardScaler()
X_tr = sc.fit_transform(X_train)
X_te = sc.transform(X_test)

# Method 1: LogisticRegressionCV — built-in, efficient C selection
lr_cv = LogisticRegressionCV(
    Cs=np.logspace(-4, 3, 30),  # 30 log-spaced values from 1e-4 to 1e3
    cv=5,
    penalty='l2',
    solver='lbfgs',
    scoring='roc_auc',   # optimize AUC — better than accuracy for imbalanced data
    max_iter=2000,
    n_jobs=-1,
    random_state=42,
)
lr_cv.fit(X_tr, y_train)
print(f"Best C (per class): {lr_cv.C_}")
print(f"Test AUC: {roc_auc_score(y_test, lr_cv.predict_proba(X_te)[:, 1]):.4f}")

# Method 2: Optuna for joint C + solver + penalty optimization
import optuna
from sklearn.model_selection import cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
    solver = 'saga' if penalty == 'l1' else 'lbfgs'
    C = trial.suggest_float('C', 1e-4, 100, log=True)
    clf = LogisticRegression(
        penalty=penalty, C=C, solver=solver, max_iter=2000, random_state=42
    )
    scores = cross_val_score(clf, X_tr, y_train, cv=5, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"\nBest params: {study.best_params}")
print(f"Best CV AUC: {study.best_value:.4f}")
```

### 4.2 — `penalty` (Regularization Type)

**What it controls:** The mathematical form of the regularization term added to the objective.

| Value | Effect | When to use |
|---|---|---|
| `'l2'` (default) | Shrinks all coefficients, never zeroes any | Default; all features contribute |
| `'l1'` | Exact zeros; automatic feature selection | Many irrelevant features; want sparse model |
| `'elasticnet'` | Compromise: sparsity + handles correlation | Correlated features + desired sparsity |
| `None` | Pure MLE, no regularization | Statistical inference; very large $n/p$ ratio |

> 🏆 **Best practice:** Start with `penalty='l2'`. Switch to `'l1'` if you have high-dimensional
> data with many potentially irrelevant features and want to identify the most important ones.
> Use `penalty=None` only when you need statistical inference (p-values) or when you have
> confirmed via CV that regularization hurts performance (large $n$, low $p$).

### 4.3 — `solver` (Optimization Algorithm)

The solver determines the optimization algorithm used to find $\hat{\boldsymbol{\beta}}$. This
is the most commonly misconfigured parameter in practice because users don't read the penalty
compatibility matrix.

```python
# Solver selection — all producing valid configurations
configs = [
    dict(solver='lbfgs',           penalty='l2',        C=1.0),          # default, general purpose
    dict(solver='liblinear',        penalty='l1',        C=1.0),          # L1, small data
    dict(solver='liblinear',        penalty='l2',        C=1.0),          # L2, small data
    dict(solver='saga',             penalty='l1',        C=1.0),          # L1, large data
    dict(solver='saga',             penalty='elasticnet', C=1.0, l1_ratio=0.5),  # ElasticNet
    dict(solver='saga',             penalty=None),                         # unregularized, large data
    dict(solver='newton-cholesky',  penalty='l2',        C=1.0),          # n >> p, L2 only
    dict(solver='newton-cg',        penalty='l2',        C=1.0),          # L2, high precision
]

# INVALID (will raise ValueError):
# dict(solver='lbfgs',     penalty='l1')   — lbfgs can't handle non-smooth L1
# dict(solver='liblinear', penalty='elasticnet')  — liblinear doesn't support EN
# dict(solver='liblinear', penalty=None)   — liblinear always regularizes

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_tr_s = StandardScaler().fit_transform(data.data)

for cfg in configs:
    clf = LogisticRegression(max_iter=1000, random_state=42, **cfg)
    clf.fit(X_tr_s, data.target)
    acc = clf.score(X_tr_s, data.target)
    penalty_str = cfg.get('penalty', 'l2')
    print(f"solver={cfg['solver']:20s} penalty={str(penalty_str):12s} → train_acc={acc:.4f}")
```

### 4.4 — `max_iter` (Maximum Iterations)

**Default:** `100` — This default is surprisingly low. For unscaled or complex data, you will
almost certainly hit convergence warnings with this default.

**The ConvergenceWarning:** The most common error beginners encounter. It means the solver
did not converge within `max_iter` steps. The fix is almost always:
1. First: **scale your features** with `StandardScaler` — this is the root cause 80% of the time
2. Then: **increase `max_iter`** to 1000 or 10000

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

# The correct pattern: always use Pipeline with StandardScaler
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,       # 1000 is a safe default; increase to 10000 if needed
        tol=1e-4,
        random_state=42,
    ))
])

# Check for convergence warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    pipe.fit(data.data, data.target)
    if len(w) > 0:
        print(f"Convergence warning: {w[0].message}")
    else:
        print(f"Converged in {pipe.named_steps['lr'].n_iter_} iterations")
```

### 4.5 — `tol` (Convergence Tolerance)

**Default:** `1e-4` — Controls when to declare convergence. Specifically, the solver stops
when the norm of the gradient (or parameter change, depending on solver) is less than `tol`.

**When to change:** Decrease to `1e-6` for high-precision applications (when you need
coefficients precise to 6 decimal places for coefficient interpretation). Increasing `tol`
is rarely worthwhile — logistic regression is already fast and the default is reasonable.

### 4.6 — `class_weight` (Handling Class Imbalance)

**Default:** `None` — All samples have equal weight in the loss.

**`'balanced'`:** sklearn computes weights as $w_k = \frac{n}{n_\text{classes} \times n_k}$
where $n_k$ is the count of class $k$. A class with 10% prevalence gets 5× more weight than
a class with 50% prevalence. This is equivalent to resampling the minority class up to 50%.

**Why it matters:** With severe class imbalance (e.g., 1% fraud rate), a model predicting
"never fraud" for every sample achieves 99% accuracy but 0% recall on the minority class.
`class_weight='balanced'` reweights the loss to penalize minority-class errors more.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

# Simulate imbalanced dataset: 95% class 0, 5% class 1
np.random.seed(42)
n = 1000
X_imb = np.random.randn(n, 5)
# Class 1 only when multiple features are high
log_odds_imb = 0.5*X_imb[:,0] + 0.3*X_imb[:,1] - 2.5  # bias toward class 0
y_imb = (np.random.rand(n) < 1/(1+np.exp(-log_odds_imb))).astype(int)
print(f"Class distribution: {np.bincount(y_imb)}")  # roughly [950, 50]

X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(
    X_imb, y_imb, test_size=0.3, stratify=y_imb, random_state=42
)
sc_i = StandardScaler()
X_tr_si = sc_i.fit_transform(X_tr_i)
X_te_si = sc_i.transform(X_te_i)

# Without class_weight
lr_no_weight = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
lr_no_weight.fit(X_tr_si, y_tr_i)
print("\nWithout class_weight='balanced':")
print(classification_report(y_te_i, lr_no_weight.predict(X_te_si), digits=3))

# With class_weight='balanced'
lr_balanced = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                  class_weight='balanced')
lr_balanced.fit(X_tr_si, y_tr_i)
print("\nWith class_weight='balanced':")
print(classification_report(y_te_i, lr_balanced.predict(X_te_si), digits=3))
```

### 4.7 — `l1_ratio` (ElasticNet Mixing Parameter)

**What it controls:** The mixing ratio between L1 and L2 in ElasticNet: `l1_ratio=0` → pure
L2, `l1_ratio=1` → pure L1. Only used when `penalty='elasticnet'` and `solver='saga'`.

**Tuning:** Use `LogisticRegressionCV` with `l1_ratios` parameter and `penalty='elasticnet'`:

```python
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

# Tune both C and l1_ratio jointly
lr_en_cv = LogisticRegressionCV(
    Cs=np.logspace(-3, 2, 10),
    l1_ratios=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    penalty='elasticnet',
    solver='saga',
    cv=5,
    scoring='roc_auc',
    max_iter=5000,
    random_state=42,
    n_jobs=-1,
)

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
sc_en = StandardScaler()
X_en = sc_en.fit_transform(data.data)

lr_en_cv.fit(X_en, data.target)
print(f"Best C: {lr_en_cv.C_[0]:.4f}")
print(f"Best l1_ratio: {lr_en_cv.l1_ratio_[0]:.2f}")

n_zero = np.sum(np.abs(lr_en_cv.coef_[0]) < 1e-6)
print(f"Zero coefficients: {n_zero} / {data.data.shape[1]}")
```

### 4.8 — `fit_intercept` and `intercept_scaling`

**`fit_intercept=True` (default):** Fits an intercept $\beta_0$. Set to `False` only if
your data is already centered (zero mean) or you have a specific theoretical reason to force
the decision boundary through the origin.

**`intercept_scaling` (liblinear only):** A subtle parameter. When using `solver='liblinear'`
with `fit_intercept=True`, sklearn appends a synthetic column of constant value
`intercept_scaling` to the feature matrix. The intercept is then regularized with the same
strength as other features, scaled by `intercept_scaling`. At the default value of 1.0, the
intercept is regularized equally with other features — this can unduly shrink the intercept
toward zero in datasets with very unequal class prevalence. Set `intercept_scaling` larger
(e.g., 10) to reduce intercept regularization relative to feature regularization.

> ⚠️ **Pitfall:** The `intercept_scaling` parameter is specific to `solver='liblinear'` and
> has zero effect for all other solvers. Misuse is common when copy-pasting configurations.

### 4.9 — `warm_start`

**Default:** `False`. When `True`, the previous `fit()` solution is used as the initialization
for the next `fit()` call (instead of the default zero initialization).

**When it helps:** When sweeping C values in a manual loop, or when incrementally adding
data. Starting from a nearby solution dramatically reduces iterations needed for convergence.

```python
# warm_start for efficient C sweep
Cs = [100.0, 10.0, 1.0, 0.1, 0.01]  # decreasing order: start from unregularized
lr_warm = LogisticRegression(
    C=Cs[0], solver='lbfgs', max_iter=1000, warm_start=True, random_state=42
)

from sklearn.datasets import load_breast_cancer
X_ws = StandardScaler().fit_transform(load_breast_cancer().data)
y_ws = load_breast_cancer().target

for C in Cs:
    lr_warm.C = C
    lr_warm.fit(X_ws, y_ws)
    print(f"C={C:.3f}: n_iter={lr_warm.n_iter_[0]}, acc={lr_warm.score(X_ws, y_ws):.4f}")
```

### 4.10 — The Hyperparameter Tuning Playbook

| Parameter | Typical range | Effect | Tuning strategy |
|---|---|---|---|
| `C` | `[1e-4, 1e3]` log-scale | Bias-variance tradeoff | `LogisticRegressionCV(Cs=30)` or Optuna |
| `penalty` | `{'l1','l2','elasticnet',None}` | Sparsity vs. shrinkage | L2 first; L1 if high-dimensional |
| `l1_ratio` | `[0.1, 0.9]` | L1/L2 mix | Grid with `LogisticRegressionCV(l1_ratios=...)` |
| `max_iter` | `100–10000` | Convergence | Start 1000; increase until no ConvergenceWarning |
| `tol` | `1e-6` to `1e-3` | Convergence precision | Default `1e-4` usually fine |
| `class_weight` | `None`, `'balanced'`, dict | Class imbalance | Try `'balanced'`; tune via PR-AUC |
| `solver` | (see matrix) | Algorithm | `lbfgs` default; `saga` for large $n$ or L1 |
| `intercept_scaling` | `[1, 100]` | Intercept regularization | Increase if intercept is being over-shrunk |

