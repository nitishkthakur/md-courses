# R6 — GLMs + Bambi (research dossier)

> Reference material for the chapter writer. Verified against PyMC v5 (5.2x), ArviZ, and
> Bambi (current line ~0.13–0.16) docs in June 2026. Where I could not 100% confirm an
> exact symbol I say so explicitly. Cross-check anything marked ⚠️ against the live docs
> before shipping code.

## Scope & which chapters use this

Primary consumer: **Chapter 09 `09-glms-and-bambi.md`**. Secondary touch-points:
- Ch 10 (hierarchical) reuses the Bambi `(1|g)` group-specific syntax and the binomial/logistic
  patterns — keep terminology identical here so Ch 10 can cross-ref.
- Ch 13 (mixtures / zero-inflation) builds on Poisson/NegBin and `pm.ZeroInflated*`.
- Ch 14 (splines) reuses the `bikes` count GLM + Bambi spline term.
- Ch 16 (missing data) reuses penguins/titanic.

Topic coverage required: linear regression (PyMC + Bambi); logistic (Bernoulli & aggregated
Binomial, logit link, log-odds → odds ratios); Poisson & Negative Binomial (log link, IRR,
overdispersion, exposure/offset); robust regression (StudentT); ordinal (cumulative / sratio)
and categorical/softmax; link functions & coefficient interpretation on each scale; interactions;
the full Bambi formula API with raw-PyMC equivalents; `bmb.interpret` predictions / comparisons /
slopes (marginal effects & contrasts).

## Verified API & idioms

### Bambi `Model` constructor (confirmed from api/Model.html)
```python
import bambi as bmb
bmb.Model(
    formula, data, family="gaussian", priors=None, link=None,
    categorical=None, potentials=None, dropna=False,
    auto_scale=True, noncentered=True, center_predictors=True,
    extra_namespace=None,
)
```
- **Valid family strings** (confirmed from `bambi/defaults/families.py` `BUILTIN_FAMILIES`):
  `"gaussian"`, `"t"`, `"bernoulli"`, `"binomial"`, `"poisson"`, `"negativebinomial"`,
  `"gamma"`, `"beta"`, `"wald"`, `"categorical"`, `"cumulative"` (ordinal), `"sratio"`
  (stopping-ratio ordinal). Note: there is **no** `"stoppingratio"` or `"ordinal"` string key —
  use `"cumulative"` / `"sratio"`. Ordinal families added in **Bambi 0.12.0** (PR #678).
- **Valid link strings**: `"identity"`, `"log"`, `"logit"`, `"probit"`, `"cloglog"`,
  `"inverse"`, `"inverse_squared"`, `"softmax"`. Each family has a sensible default link
  (bernoulli/binomial→logit, poisson/negativebinomial→log, gaussian→identity), so passing
  `link=` is usually optional.
- `auto_scale=True` and `center_predictors=True` are **on by default** — Bambi computes
  weakly-informative default priors scaled to the data. Mention this prominently: it's why a
  bare `bmb.Model("y ~ x", df)` already samples sanely. Inspect with `model.plot_priors()` or
  print `model` (after `model.build()`).

### Bambi `.fit()` (confirmed)
```python
idata = model.fit(
    draws=1000, tune=1000, chains=4, target_accept=0.9,
    random_seed=RANDOM_SEED,
    inference_method="pymc",            # also "nuts_numpyro", "nuts_blackjax", "vi", "laplace"
    idata_kwargs={"log_likelihood": True},  # needed for az.loo / az.compare
)
```
Returns an **`az.InferenceData`**. Backend PyMC model is `model.backend.model`. Use
`model.graph()` to draw it. ⚠️ Arg name for the NumPyro backend has been `inference_method=`
across recent versions; in some pre-0.13 builds the value spelling was `"nuts_numpyro"` — verify
the exact string in the installed version's `model.fit` docstring.

### Formula idioms (confirmed)
- Intercept implicit; suppress with `"y ~ 0 + x"`.
- Group-specific (random) effects: `"y ~ x + (1|g)"`, random slope `"y ~ x + (x|g)"`.
- Interaction: `"y ~ a*b"` expands to `a + b + a:b`; `a:b` is the interaction-only term.
- Categorical predictor coded automatically; force with `C(x)`; pick reference with
  `C(x, Treatment("level"))`.
- **Bernoulli with a non-0/1 / multi-string outcome**: pick the success level in the formula,
  e.g. `"vote['clinton'] ~ party_id"` — the single-bracket `['level']` syntax marks the event.
- **Aggregated Binomial** (successes out of trials): `"p(successes, trials) ~ x"` with
  `family="binomial"`. The `p(y, n)` proportion helper is the confirmed current syntax.
- **Offset / exposure** (Poisson/NegBin): `"y ~ x + offset(log(exposure))"`. ⚠️ Confirmed by
  Bambi maintainer (tcapretto) on PyMC Discourse and present in tests, but he flagged it as
  "just in the tests," lightly documented — verify it imports/runs in the pinned version; if it
  errors, fall back to raw PyMC with `pm.math.log(exposure)` added to the linear predictor.

### `bmb.interpret` — marginal effects (confirmed signatures)
```python
# Adjusted predictions on the response scale
bmb.interpret.predictions(model, idata, conditional, average_by=None, ...)
bmb.interpret.plot_predictions(model, idata, conditional, ...)

# Contrasts / comparisons (differences or ratios -> odds ratios, IRR)
bmb.interpret.comparisons(model, idata, contrast, conditional,
                          average_by=None, comparison="diff")   # "diff" | "ratio"
# Slopes / marginal effects (dy/dx, elasticities)
bmb.interpret.slopes(model, idata, wrt, conditional=None,
                     average_by=None, slope="dydx", eps=1e-4)   # dydx|eyex|eydx|dyex
bmb.interpret.plot_slopes(model, idata, wrt, conditional=None, ...)
```
- `contrast={"persons":[1,4]}` and `comparison="ratio"` gives **odds ratios** (logistic) or
  **rate ratios / IRR** (Poisson) directly. `comparison="diff"` gives differences on the
  response scale (e.g. probability differences — proper average marginal effects).
- `wrt` = "with respect to" variable for slopes; `slope="eyex"` = elasticity.
- Each returns a `Result` named-tuple: `.summary` (DataFrame with mean + HDI) and `.draws`
  (InferenceData with the full posterior of the quantity). This is the Bayesian analogue of
  R's **marginaleffects** package — say so; the design is consciously modeled on it.
- These compute over the posterior, so the credible interval on a contrast is exact (not a
  delta-method approximation). That's a strong selling point vs frequentist `margins`.

### Raw PyMC v5 equivalents (all confirmed spellings)
```python
import pymc as pm, numpy as np
RANDOM_SEED = 8927

# --- Linear ---
with pm.Model(coords={"obs": df.index}) as lin:
    x = pm.Data("x", df.x.values, dims="obs")
    a = pm.Normal("a", 0, 1)
    b = pm.Normal("b", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    mu = pm.Deterministic("mu", a + b * x, dims="obs")
    pm.Normal("y", mu=mu, sigma=sigma, observed=df.y.values, dims="obs")

# --- Logistic (Bernoulli, logit link) ---
with pm.Model() as logit:
    a = pm.Normal("a", 0, 1.5); b = pm.Normal("b", 0, 0.5)
    p = pm.Deterministic("p", pm.math.invlogit(a + b * x))   # = sigmoid
    pm.Bernoulli("y", p=p, observed=y01)
    # equivalently, pass the linear predictor directly: pm.Bernoulli("y", logit_p=a + b*x, ...)

# --- Aggregated Binomial ---
pm.Binomial("y", n=trials, p=pm.math.invlogit(eta), observed=successes)
# or pm.Binomial("y", n=trials, logit_p=eta, observed=successes)

# --- Poisson (log link) ---
with pm.Model() as pois:
    lam = pm.math.exp(a + b * x + log_exposure)   # offset = log_exposure added on log scale
    pm.Poisson("y", mu=lam, observed=counts)

# --- Negative Binomial (overdispersed counts) ---
with pm.Model() as nb:
    mu = pm.math.exp(a + b * x)
    alpha = pm.Exponential("alpha", 1.0)          # dispersion; small alpha = MORE overdispersed
    pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=counts)

# --- Robust (StudentT likelihood) ---
nu = pm.Exponential("nu", 1/29) + 1              # McElreath/Stan convention; nu>1
pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y)

# --- Ordinal (OrderedLogistic) ---
with pm.Model() as ord_m:
    cut = pm.Normal("cutpoints", mu=np.linspace(-2, 2, K-1), sigma=1,
                    transform=pm.distributions.transforms.univariate_ordered,
                    initval=np.linspace(-2, 2, K-1))
    eta = b * x                                   # NOTE: no intercept (absorbed by cutpoints)
    pm.OrderedLogistic("y", eta=eta, cutpoints=cut, observed=y_int)  # y_int in 0..K-1
```
- `pm.math.invlogit` == `pm.math.sigmoid` (both exist). `pm.Bernoulli`/`pm.Binomial` accept
  **`logit_p=`** directly (numerically safer than building `p` then passing it).
- `pm.OrderedLogistic(name, eta, cutpoints, compute_p=True)` and `pm.OrderedProbit` exist
  (confirmed in PyMC 5.x docs). Outcome must be integer-coded `0..K-1`. The
  `transform=pm.distributions.transforms.univariate_ordered` (alias historically
  `pm.distributions.transforms.ordered`) enforces monotone cutpoints — **must** pass `initval`
  or sampling can start on a zero-prob point. ⚠️ Confirm `univariate_ordered` vs `ordered`
  spelling in the pinned version; both have existed.

## Key concepts & equations the chapter must nail

- **GLM anatomy:** linear predictor $\eta_i = \mathbf{x}_i^\top\boldsymbol\beta$; link $g(\mu_i)=\eta_i$;
  $\mu_i = g^{-1}(\eta_i)$; likelihood family for $y_i\mid\mu_i$. Three knobs: family, link, predictor.
- **Logit link & odds:** $\text{logit}(p)=\log\frac{p}{1-p}=\eta$, inverse
  $p=\frac{1}{1+e^{-\eta}}=\sigma(\eta)$. A one-unit rise in $x_j$ multiplies the **odds** by
  $e^{\beta_j}$ → **odds ratio** $=e^{\beta_j}$. Coefficients live on the log-odds scale; they are
  **not** probability changes (the probability effect depends on the baseline — the "divide-by-4
  rule": the max slope of $p$ w.r.t. $x_j$ is $\beta_j/4$, a quick upper bound on the prob change).
- **Log link & rates (Poisson):** $\log\mu=\eta$, $\mu=e^{\eta}$. $e^{\beta_j}$ = **incidence/rate
  ratio (IRR)**: multiplicative effect on the expected count. Poisson assumes
  $\mathrm{Var}(y)=\mathrm{E}(y)=\mu$ (equidispersion).
- **Overdispersion → NegBin:** real counts usually have $\mathrm{Var}(y)>\mu$. PyMC NegBin:
  $\mathrm{Var}(y)=\mu+\mu^2/\alpha$. **Small $\alpha$ ⇒ heavy overdispersion; $\alpha\to\infty$ ⇒
  Poisson.** ⚠️ This is the *inverse* of the textbook NB2 dispersion $\phi$ where
  $\mathrm{Var}=\mu+\phi\mu^2$; here $\alpha_{\text{PyMC}}=1/\phi$. Flag this footgun loudly.
  PyMC parametrizes by `(mu, alpha)`; also accepts `(p, n)` with $p=\alpha/(\mu+\alpha)$, $n=\alpha$.
- **Exposure/offset:** when units have different exposure $u_i$ (person-years, road-km, bin width),
  model the *rate*: $\log\mu_i=\log u_i + \mathbf{x}_i^\top\beta$. The $\log u_i$ enters as a fixed
  **offset** (coefficient pinned to 1), not a free parameter.
- **Robust regression:** swap Normal for **StudentT** likelihood; small $\nu$ (3–7) gives heavy
  tails so outliers don't drag the line. As $\nu\to\infty$, StudentT → Normal. Prior on $\nu$:
  `pm.Exponential("nu", 1/29)` shifted by 1 (Stan/McElreath convention), or `pm.Gamma("nu",2,0.1)`.
- **Ordinal — cumulative logit:** model cumulative probabilities with **shared cutpoints**
  $\alpha_1<\dots<\alpha_{K-1}$: $P(y\le k)=\sigma(\alpha_k-\eta)$, so $\eta$ shifts the whole
  latent distribution. **No intercept** (cutpoints play that role). `sratio` (stopping/continuation
  ratio) models the conditional "stop at category $k$ given $\ge k$" — different link, same ordered
  outcome. Use ordinal, **never** treat a Likert 1–5 as numeric or as unordered categorical.
- **Categorical/softmax:** $K$ outcomes, reference category fixed; for class $k$
  $P(y=k)=\text{softmax}(\eta_k)$, $\eta_k=\mathbf{x}^\top\beta_k$ with $\beta_{\text{ref}}=0$.
  $K-1$ coefficient vectors are estimated. Bambi `family="categorical"`, link `softmax`.
- **Interactions:** coefficient on $a{:}b$ = how the slope of $a$ changes per unit $b$. On a
  **link scale** (logit/log) even a model *without* an interaction term is non-additive on the
  response scale — interpret with `interpret.comparisons/slopes`, not raw coefficients. The rugged
  example is the canonical "the slope of ruggedness on log-GDP flips sign inside vs outside Africa."

## Common errors → causes → fixes

| Symptom | Cause | Fix |
|---|---|---|
| Logistic posterior won't converge / coefficient runs to ±∞, huge `r_hat` | **(Quasi-)separation**: a predictor perfectly splits the classes; flat/wide prior gives no shrinkage | Standardize predictors; use a weakly-informative prior (Gelman: `StudentT(nu=4, 0, 1)` or `Normal(0, 1–2.5)` on standardized x); the prior *is* the regularizer |
| NegBin/Poisson: `mu` overflows, `nan` logp, sampler stuck | Building `mu` then `exp` of an un-centered large $\eta$; or passing `mu` already exponentiated where a log-scale value expected | Center/standardize predictors; keep priors tight on the **log** scale (a Normal(0,1) on a log-rate slope is already a 2.7× IRR per SD); exponentiate inside the model only once |
| Poisson fits but PPC undershoots the tail / variance | **Overdispersion** — data variance ≫ mean | Switch to `pm.NegativeBinomial` / `family="negativebinomial"`; compare with `az.loo`/`az.compare` |
| Excess zeros beyond NegBin | Zero-inflation / hurdle process | `pm.ZeroInflatedPoisson` / `pm.ZeroInflatedNegativeBinomial` (Ch 13) |
| Counts implausible per unit | Forgot exposure | Add `offset(log(exposure))` (Bambi) or `+ log_exposure` to $\eta$ (PyMC) |
| Ordinal: `SamplingError: Initial evaluation ... -inf`, or divergences | Cutpoints not ordered at init; missing `transform`/`initval` | Use `transform=...univariate_ordered` **and** pass sorted `initval`; code outcome as ints `0..K-1` (not 1..K) |
| `dy/dx` / coefficients misread as probability/count changes | Reading link-scale coefs as response-scale effects | Report odds ratios `exp(beta)` / IRR, or use `bmb.interpret` on the response scale; teach the divide-by-4 rule |
| `az.loo` errors "no log_likelihood group" | Forgot to store pointwise log-lik | Bambi: `fit(idata_kwargs={"log_likelihood": True})`; PyMC: `pm.compute_log_likelihood(idata, model)` |
| Bambi default priors "too tight," posterior barely moves | `auto_scale` rescaled priors to standardized data; you compared to unstandardized intuition | Inspect `model.plot_priors()`; override via `priors={...}`; remember predictors are centered internally so the intercept is at the predictor mean |
| Bambi Bernoulli "could not determine success level" | Outcome is string/multi-level | Use `"y['success_level'] ~ ..."` bracket syntax |
| StudentT robust model: `nu` wanders to huge values | Data has no outliers; likelihood ≈ Normal | Fine — it's telling you robustness isn't needed; or fix/regularize `nu` |

## Datasets & load snippets

```python
import seaborn as sns, pandas as pd, bambi as bmb

# Penguins — logistic (sex/species), robust regression, missing data
penguins = sns.load_dataset("penguins").dropna()      # cols: species, island,
# bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, (year)
# (palmerpenguins.load_penguins() also works; has a `year` column)

# Titanic — logistic with genuine missing data (age has NaNs)
titanic = sns.load_dataset("titanic")                 # survived, pclass, sex, age, fare, ...

# Tips — Gaussian/robust GLM + interactions
tips = sns.load_dataset("tips")                        # total_bill, tip, sex, smoker, day, time, size

# Bikes — Poisson / NegBin count GLM (Bambi built-in)
bikes = bmb.load_data("bikes")                         # hourly bike counts + weather/temp/hour
# Confirmed: "bikes" IS a registered bmb.load_data dataset. Full list:
#  my_data, adults, ANES, ESCS, carclaims, batting, cherry_blossoms,
#  sleepstudy, periwinkles, admissions, bikes, mtcars, kidney

# ANES — Bambi's canonical logistic example
anes = bmb.load_data("ANES")                           # vote, party_id, age

# Rugged — interactions / log-GDP (McElreath rethinking repo)
rugged = pd.read_csv(
    "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/rugged.csv", sep=";")
rugged = rugged.dropna(subset=["rgdppc_2000"]).assign(
    log_gdp=lambda d: np.log(d["rgdppc_2000"]))
```
For ordinal, McElreath's `Trolley` (`.../data/Trolley.csv`, sep=";", outcome `response` 1–7) is the
classic example; or synthesize ordinal data with known cutpoints for parameter recovery.

## Curated resources

- **Bambi Model API** — https://bambinos.github.io/bambi/api/Model.html — exact constructor
  args, family/link strings, `.fit/.predict/.set_priors`. The authority for §"Verified API."
- **Bambi `interpret`: Plot Slopes** — https://bambinos.github.io/bambi/notebooks/plot_slopes.html
  — `slopes`/`plot_slopes` signatures, `wrt`/`conditional`/`average_by`/`slope` semantics (the well-
  arsenic example). Use for marginal-effects section.
- **Bambi `interpret`: Plot Predictions** —
  https://bambinos.github.io/bambi/notebooks/plot_predictions.html — adjusted predictions API.
- **Bambi `interpret`: Plot Comparisons** —
  https://bambinos.github.io/bambi/notebooks/plot_comparisons.html — `comparisons` with
  `comparison="ratio"` for odds ratios / IRR (titanic & fish examples).
- **Bambi Logistic Regression (ANES)** —
  https://bambinos.github.io/bambi/notebooks/logistic_regression.html — canonical
  `vote['clinton'] ~ party_id + party_id:age`, `family="bernoulli"`, log-odds interpretation,
  `model.predict`. Mirror this almost verbatim for the chapter's logistic walkthrough.
- **Bambi Hierarchical Binomial** —
  https://bambinos.github.io/bambi/notebooks/hierarchical_binomial_bambi.html — `p(y, n)`
  aggregated-binomial syntax + `(1|g)`; bridges to Ch 10.
- **Bambi Categorical Regression** —
  https://bambinos.github.io/bambi/notebooks/categorical_regression.html — `family="categorical"`,
  softmax, n−1 coefficient sets.
- **Bambi 0.12.0 release (ordinal)** — https://github.com/bambinos/bambi/releases/tag/0.12.0 —
  introduces `"cumulative"` & `"sratio"` ordinal families + predictions on new groups.
- **Bambi paper (Capretto et al. 2022, JSS v103i15)** — https://www.jstatsoft.org/article/view/v103i15
  / arXiv:2012.10754 — design rationale, formula grammar, default-prior construction. Cite for "why
  Bambi" and the auto-prior story.
- **PyMC `OrderedLogistic`** —
  https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.OrderedLogistic.html
  — `eta`, `cutpoints`, K−1 cutpoints, integer outcome.
- **PyMC example: Ordinal regression** —
  https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-ordinal-regression.html
  — `univariate_ordered` transform + Dirichlet-cutpoint alternative; full v5 model.
- **PyMC `NegativeBinomial`** —
  https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.NegativeBinomial.html
  — `(mu, alpha)` parametrization, `Var=mu+mu^2/alpha`. Anchor the α-vs-φ warning here.
- **PyMC example: Negative Binomial regression** —
  https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-negative-binomial-regression.html
  — overdispersion diagnosis + offset discussion.
- **PyMC example: Binomial regression** —
  https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-binomial-regression.html
  — `pm.math.invlogit` + aggregated `pm.Binomial(n=, p=)`.
- **Gelman et al. (2008), weakly-informative prior for logistic** — https://arxiv.org/pdf/0901.4011 —
  the StudentT(4,0,1)/Cauchy(2.5) default and the separation argument. Use in the priors/errors box.
- **PyMC Discourse: offset/exposure in Bambi** —
  https://discourse.pymc.io/t/how-to-specify-exposure-offset-in-a-bambi-model/13667 — maintainer's
  `offset(log(...))` answer (and its "lightly documented" caveat).
- **McElreath, *Statistical Rethinking* 2e** — Ch 8 (rugged interaction), Ch 11 (Poisson/binomial,
  log-odds, IRR), Ch 12 (ordered categorical, monsters & mixtures). PyMC port:
  https://github.com/pymc-devs/pymc-resources (Rethinking_2). The intuition backbone for this chapter.
- **Gelman, Hill & Vehtari, *Regression and Other Stories* (free PDF)** —
  https://avehtari.github.io/ROS-Examples/ — Ch 13–15 logistic, Ch 15 Poisson/overdispersion, and
  the divide-by-4 rule; arsenic-wells example matches Bambi's `slopes` demo.

## Uncertainties / things the writer should double-check

1. **Ordered transform spelling.** I confirmed `pm.distributions.transforms.univariate_ordered`
   from the PyMC ordinal example, but older docs use `transforms.ordered`. Verify against the
   pinned PyMC 5.x version; both may be importable but prefer `univariate_ordered`.
2. **Bambi offset term.** `offset(log(x))` in the formula is confirmed by the maintainer and tests
   but he called it lightly documented. Run a smoke test; if it fails, use the raw-PyMC offset.
3. **`bmb.interpret` exact kwarg names across versions.** `comparison=` (singular) vs older
   `comparison_type=` has drifted between Bambi releases; `slope=` values `dydx/eyex/eydx/dyex`
   confirmed. Check the installed version's signature before pasting.
4. **NumPyro backend string** in `model.fit(inference_method=...)`: confirmed values include
   `"nuts_numpyro"`/`"nuts_blackjax"` in recent versions, but the precise spelling changed around
   0.12–0.13. Verify.
5. **`p(y, n)` vs `proportion(y, n)`.** Current docs/tests use `p(successes, trials)`; the
   alias `proportion(...)` has appeared in discussion — confirm the live alias before using it.
6. **`auto_scale` default behavior** can surprise users comparing to hand-set PyMC priors; show
   `model.plot_priors()` so the equivalence (or difference) is visible, rather than asserting they
   match exactly.
7. NegBin α-vs-φ inversion: triple-check the variance formula direction in whatever PyMC version is
   pinned — it is the single most common interpretation bug in this chapter.
