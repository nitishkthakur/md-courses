# R2 — PRIORS (research dossier)

Reference material for the chapter writers. Verified against PyMC v5 docs (stable), the Stan
Prior-Choice wiki, ArviZ, Bambi, and the primary literature (Gelman 2006; Simpson et al. 2017;
Piironen & Vehtari 2017; Kallioinen et al. 2023). Where I could not verify an exact symbol I say so.

## Scope & which chapters use this

- **Chapter 02** (`02-priors-1-concepts-and-common-priors.md`): what a prior *is*; flat/improper vs
  proper; conjugate families with analytic posteriors; the weakly-informative / informative /
  regularizing taxonomy; PC priors; the catalog of common distributions and *when* to reach for each;
  priors on **location vs scale**.
- **Chapter 03** (`03-priors-2-choosing-building-and-checking.md`): prior predictive checks as the
  decision procedure; schools of thought (subjective / objective / empirical Bayes / weakly-informative);
  standardization & scaling so unit-scale priors make sense; scale/variance priors in depth (HalfNormal,
  HalfCauchy, Exponential, InverseGamma pitfalls, Gelman 2006); LKJ for correlations; horseshoe /
  regularized horseshoe; hyperparameter choice; "should I look at the data?" debate; sensitivity analysis.
- Feeds forward to Ch 09 (GLM coefficient priors), Ch 10 (LKJ + hyperpriors for varying effects),
  Ch 12 (GP lengthscale/amplitude priors), Ch 14 (sparsity/horseshoe). Cross-ref those, don't duplicate.

## Verified API & idioms (confirmed against PyMC v5 stable docs)

**Distributions (all confirmed v5 spellings):** `pm.Normal("mu", mu=0, sigma=1)`,
`pm.HalfNormal("s", sigma=1)`, `pm.Exponential("s", lam=1)` (rate param is `lam`),
`pm.Gamma("g", alpha=2, beta=0.1)` *or* `pm.Gamma("g", mu=..., sigma=...)`,
`pm.Beta("p", alpha=2, beta=2)`, `pm.StudentT("b", nu=3, mu=0, sigma=1)`,
`pm.HalfCauchy("s", beta=1)`, `pm.Cauchy`, `pm.InverseGamma("v", alpha=3, beta=0.5)`,
`pm.Dirichlet("w", a=np.ones(k))`, `pm.LKJCorr`, `pm.LKJCholeskyCov`, `pm.MvNormal`.
NB the bible warns (correctly): **PyMC/Stan parameterize Normal by σ (sd), not variance, not
precision** — BUGS/JAGS used precision `tau`. Make this warning explicit in Ch 02.

**Prior predictive (verified signature, PyMC stable):**
```python
pm.sample_prior_predictive(
    draws=500, model=None, var_names=None, random_seed=None,
    return_inferencedata=True, idata_kwargs=None, compile_kwargs=None,
)
```
Default `draws=500`. Returns `InferenceData` with groups **`prior`** (unobserved RVs +
deterministics) and **`prior_predictive`** (observed RVs sampled forward). Note: the arg is
`draws`, **not** `samples` (the `samples` name was deprecated/removed — do not use it). Confirmed
working idiom from the official posterior_predictive core notebook:
```python
def standardize(series):
    return (series - series.mean()) / series.std()

with pm.Model() as model_1:
    a = pm.Normal("a", 0.0, 0.5)          # intercept prior, std-scale
    b = pm.Normal("b", 0.0, 1.0)          # slope prior, std-scale
    mu = a + b * predictor_scaled
    sigma = pm.Exponential("sigma", 1.0)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=outcome_scaled)
    idata = pm.sample_prior_predictive(draws=50, random_seed=rng)
# access: idata.prior["a"], idata.prior_predictive["obs"]
```
Plot with `az.plot_ppc(idata, group="prior")` and visualize implied regression lines by overlaying
`a + b*x` draws. The official notebook's teaching point: flat priors (`sigma=10`) let the outcome
range from −40 to +40 SD — absurd on standardized data — which *motivates* weakly-informative priors.

**LKJCholeskyCov (verified, stable docs):**
```python
with pm.Model() as m:
    sd_dist = pm.Exponential.dist(1.0, size=n)        # note .dist() — NOT a named RV
    chol, corr, sigmas = pm.LKJCholeskyCov(
        "chol_cov", eta=4, n=n, sd_dist=sd_dist, compute_corr=True
    )
    vals = pm.MvNormal("vals", mu=np.zeros(n), chol=chol)
```
- `eta > 0` is the LKJ shape. **`eta = 1` → uniform over correlation matrices**; **larger eta → more
  mass on near-identity (low-correlation) matrices** (stronger shrinkage toward independence). This is
  the single most-confused point — nail it.
- `sd_dist` MUST be built with the `.dist()` classmethod (an unregistered distribution) with
  `shape[-1] == n`. Scalars auto-broadcast.
- `compute_corr=True` (default) returns the tuple `(chol, corr, stds)`. With `store_in_trace=True`
  (default) the corr and stds are saved as `{name}_corr` and `{name}_stds`. If `compute_corr=False`
  you get a *packed* triangular vector; unpack with `pm.expand_packed_triangular(n, packed, lower=True)`.
- `pm.LKJCorr` exists too (returns the correlation matrix directly) but the Cholesky form is the
  recommended, better-conditioned, faster-sampling default for varying-slope models (Ch 10).

**Bambi (verified):** `import bambi as bmb`; `Model("y ~ x1 + x2", data=df)` auto-generates
**weakly-informative, data-scaled priors** (rstanarm-style, Westfall 2017, arXiv:1702.01201).
Override with `priors={"x1": bmb.Prior("Normal", mu=0, sigma=1)}` passed to `Model(...)`. The
`Prior` `name` must be a PyMC distribution string ("Normal", "StudentT", ...). Passing explicit
`Prior` objects disables auto-scaling for those terms. Inspect generated priors with `model.build()`
then `model` repr, or `model.set_priors(...)`. **Always also show the raw-PyMC equivalent** (bible rule).

## Key concepts & equations the chapter must nail

**Conjugate families (verified update rules — bayesrulesbook Ch 5, John D. Cook compendium):**
- **Beta–Binomial.** Prior $\theta\sim\mathrm{Beta}(\alpha,\beta)$; data $Y=y$ successes in $n$
  Bernoulli/Binomial trials. Posterior $\theta\mid y \sim \mathrm{Beta}(\alpha+y,\ \beta+n-y)$.
  Posterior mean $=\frac{\alpha+y}{\alpha+\beta+n}$ — a convex combination of prior mean and MLE;
  $(\alpha+\beta)$ is "prior pseudo-counts." $\mathrm{Beta}(1,1)=\mathrm{Uniform}(0,1)$;
  $\mathrm{Beta}(0.5,0.5)$ = Jeffreys prior for the binomial.
- **Gamma–Poisson.** Prior $\lambda\sim\mathrm{Gamma}(s,r)$ (shape $s$, **rate** $r$); data
  $y_1,\dots,y_n$. Posterior $\lambda\mid y\sim\mathrm{Gamma}\!\big(s+\sum y_i,\ r+n\big)$.
  Posterior mean $=\frac{s+\sum y_i}{r+n}$. (Match PyMC: `pm.Gamma(alpha=s, beta=r)`.)
- **Normal–Normal (known data variance $\sigma^2$).** Prior on the mean
  $\mu\sim\mathrm{Normal}(\theta,\tau^2)$; $n$ obs with sample mean $\bar y$. Posterior
  $\mu\mid y\sim\mathrm{Normal}(\mu_n,\ \tau_n^2)$ with
  $$\mu_n=\frac{\theta\,\sigma^2 + \bar y\, n\tau^2}{n\tau^2+\sigma^2},\qquad
  \tau_n^2=\frac{\tau^2\sigma^2}{n\tau^2+\sigma^2}.$$
  Cleaner in **precision** form: posterior precision = sum of precisions
  $1/\tau_n^2 = 1/\tau^2 + n/\sigma^2$; posterior mean is the precision-weighted average of prior
  mean and $\bar y$. This is the cleanest derivation to *show* (it previews shrinkage and the
  hierarchical partial-pooling formula in Ch 10).

**Flat / improper vs proper.** A *proper* prior integrates to 1. A *flat/improper* prior
($p(\theta)\propto 1$ on $\mathbb R$, or $p(\sigma)\propto 1/\sigma$) does not. Improper priors can
still yield a proper posterior, but they (a) can produce *improper posteriors* silently (esp. in
hierarchical variance params and separable logistic regression), (b) are not invariant to
reparameterization, (c) break prior predictive checks and marginal-likelihood/Bayes-factor methods.
Jeffreys priors ($p(\theta)\propto\sqrt{\det I(\theta)}$) are the classic "objective" choice and are
reparameterization-invariant, but are often improper and rarely what you want in applied multilevel work.

**The taxonomy (be precise, these get conflated):**
- *Informative*: encodes real domain knowledge; narrow; can dominate small data. e.g. `Normal(0.4, 0.2)`.
- *Weakly informative*: rules out absurd values, stays agnostic within the plausible range; mild
  regularization for stability. e.g. `Normal(0, 1)` on standardized predictors.
- *Regularizing*: deliberately shrinks toward a null (ridge ↔ Normal prior; lasso ↔ Laplace prior;
  horseshoe ↔ sparsity). Bayesian penalization = MAP under a prior. Tie this to L2/L1 the ML reader knows.
- *Flat/noninformative/"objective"* (Jeffreys, reference priors): try to "let the data speak";
  fragile in hierarchical models.

**Stan Prior-Choice hierarchy (verified, quote it):** (1) flat — not recommended; (2) super-vague proper
`normal(0,1e6)` — not recommended; (3) very weak `normal(0,10)`; (4) **generic weakly informative
`normal(0,1)`** (Gelman) or `student_t(3,0,1)` (Vehtari, heavier tails/robustness); (5) specific
informative. **Regression coefficients (predictors scaled to mean 0, SD 0.5):**
`student_t(nu,0,s)` with `3<nu<7`; rstanarm default `normal(0,2.5)`. The historic Cauchy
(`student_t(1,0,2.5)`) is now discouraged — tails too heavy → sampling trouble with weakly-informative data.

**Scale/variance priors (the heart of Ch 03).** For a positive scale $\sigma$ use
**HalfNormal**, **Exponential**, or **half-Student-t / HalfCauchy**, NOT InverseGamma on the variance.
- *Gelman (2006), "Prior distributions for variance parameters in hierarchical models", Bayesian
  Analysis 1(3):515–534.* Two key results: (i) the conjugate **`InvGamma(ε, ε)` prior on $\sigma^2$
  is pathological** for hierarchical variance — as $\varepsilon\to0$ there is no proper limiting
  posterior, and small $\varepsilon$ artificially pulls the group-level SD toward 0 (spurious
  shrinkage), with inference highly sensitive to $\varepsilon$, *especially when the number of groups
  is small or the true $\sigma$ is near 0*. (ii) Recommended default: a **half-Cauchy** (member of
  the folded-/half-t family) on $\sigma$ itself, weakly informative with a heavy tail (e.g. scale 25
  for 8-schools $\tau$). Stan wiki refinements: `half-normal(0,1)` or `half-t(4,0,1)` when groups are
  few; `Gamma(2, 0)` ("$\propto\sigma$") as a boundary-avoiding choice for penalized/modal estimation.
- **Exponential(1)** is a clean, light-tailed, single-parameter default (PyMC docs use it). HalfNormal
  is similar but with even lighter tails. HalfCauchy permits large scales but its very heavy tail can
  slow NUTS and occasionally needs a finite-scale regularizer. Rule of thumb for the writer: prefer
  HalfNormal/Exponential by default; reach for half-t/half-Cauchy when you genuinely expect the scale
  might be large and don't want to rule it out.

**LKJ for correlation matrices.** $\mathrm{LKJ}(\eta)$ is a prior over $n\times n$ correlation
matrices with density $\propto \det(R)^{\eta-1}$. $\eta=1$ uniform; $\eta>1$ concentrates on the
identity (independence); $\eta<1$ favors strong correlations. Default for varying-slope models:
$\eta\approx 2$ (mild regularization toward independence — analogue of `Beta(2,2)`). Use
`LKJCholeskyCov` (Cholesky-parameterized) for sampling efficiency. See Ch 10.

**Horseshoe / regularized horseshoe (sparsity).** Original horseshoe (Carvalho, Polson, Scott 2010):
$\beta_j\sim N(0,\tau^2\lambda_j^2)$, $\lambda_j\sim\mathrm{HalfCauchy}(0,1)$,
$\tau\sim\mathrm{HalfCauchy}(0,\tau_0)$. Global $\tau$ shrinks everything; local $\lambda_j$ "spikes"
let true signals escape — a continuous spike-and-slab. *Piironen & Vehtari (2017), "Sparsity
information and regularization in the horseshoe and other shrinkage priors", EJS 11(2):5018–5051
(arXiv:1707.01694)*: two fixes — (a) set $\tau_0$ from a guess $p_0$ of the number of nonzero
coefficients ($\tau_0 = \frac{p_0}{p-p_0}\frac{\sigma}{\sqrt n}$), and (b) add a **slab** of finite
width $c$ via $\tilde\lambda_j^2 = \frac{c^2\lambda_j^2}{c^2+\tau^2\lambda_j^2}$ so large coefficients
are regularized (finite-width slab vs infinite-width for plain horseshoe). The slab width gets its own
prior, e.g. $c^2\sim\mathrm{InvGamma}$. Reparameterize non-centered to sample well. PyMC build pattern
(Austin Rochford's writeup is the canonical worked example).

## Common errors → causes → fixes

| Symptom | Cause | Fix |
|---|---|---|
| `sd_dist` error / wrong shape in `LKJCholeskyCov` | Passed a *named* RV or wrong shape | Build with `.dist()` (e.g. `pm.Exponential.dist(1., size=n)`), ensure `shape[-1]==n` |
| Prior predictive `obs` ranges to ±40 SD; nonsensical $y$ | Priors too wide for the (standardized) scale | Tighten to unit-scale weakly-informative; standardize predictors & outcome first |
| Divergences / funnel that priors can't cure | Centered hierarchical parameterization, not the prior per se | Non-centered reparam (Ch 7/10); but also avoid `InvGamma(ε,ε)` on the variance |
| Group SD collapses to ~0, very sensitive to settings | `InverseGamma(ε,ε)` prior on variance (Gelman 2006) | Use `pm.HalfNormal`/`pm.Exponential`/half-t on the SD `σ`, never IG(ε,ε) on `σ²` |
| Heavy-tailed posterior, slow/stuck NUTS on a scale param | HalfCauchy tail too heavy / weak likelihood | Switch to HalfNormal/Exponential, or half-t(4); add `target_accept=0.95` |
| Posterior barely moves from prior on small data | Prior too informative (unintentionally) | Run prior predictive + prior sensitivity (power-scaling); widen |
| Improper posterior / sampler explodes | Flat/improper prior on a variance or under separation | Use a proper weakly-informative prior |
| `sample_prior_predictive(samples=...)` TypeError | `samples` arg removed in v5 | Use `draws=` |
| LKJ "all correlations ~0" unexpectedly | `eta` too large (concentrates on identity) | Lower `eta` toward 1–2; remember larger eta ⇒ less correlation |
| Coefficients on raw (unscaled) predictors need huge/tiny priors | Predictors on wildly different scales | Standardize (mean 0, SD 0.5 per Gelman) so a single unit-scale prior fits all |

## Datasets & load snippets

- **Howell !Kung** (slope/intercept priors, the canonical prior-predictive teaching set):
  `df = pd.read_csv("https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv", sep=";")`
  then `adult = df[df.age >= 18]`. Standardize: `weight_s = (adult.weight - adult.weight.mean())/adult.weight.std()`.
- **Eight Schools** (hierarchical scale prior, half-Cauchy on $\tau$): hardcode
  `y = np.array([28,8,-3,7,-1,1,18,12]); sigma = np.array([15,10,16,11,9,11,10,18])`.
- **Synthetic regression with known truth** (ENCOURAGED for showing prior pull / shrinkage):
  `rng=np.random.default_rng(8927); x=rng.normal(size=100); y=1.5+2.0*x+rng.normal(scale=1,size=100)`
  then show recovery and how priors bias estimates under small `n`.
- **Diabetes / many-predictor** for horseshoe demo: `from sklearn.datasets import load_diabetes`
  (10 features) or a synthetic sparse design (`beta_true` mostly zeros) to demonstrate selection.

## Curated resources (each verified URL + why + pointer)

- **Stan Prior Choice Recommendations wiki** — https://github.com/stan-dev/stan/wiki/prior-choice-recommendations
  — the single most concrete, opinionated source; quote its 5-level hierarchy, the
  `normal(0,1)`/`student_t(3,0,1)` defaults, scale-param advice, and the `inv_gamma(0.4,0.3)` NB-phi prior.
- **Gelman (2006), "Prior distributions for variance parameters in hierarchical models"** —
  https://sites.stat.columbia.edu/gelman/research/published/taumain.pdf — *the* citation for why
  InvGamma(ε,ε) is bad and half-Cauchy is the default; Bayesian Analysis 1(3):515–534.
- **Simpson, Rue, Riebler, Martins, Sørbye (2017), "Penalising model component complexity" (PC priors)** —
  https://arxiv.org/abs/1403.4630 (Statistical Science 32(1):1–28) — four principles (Occam, complexity
  measure, constant-rate penalization, user scaling); base-model thinking; PC prior = Exponential on the
  KL-distance scale → Exponential/Gumbel-type priors. Use for the PC-priors section of Ch 02/03.
- **Piironen & Vehtari (2017), regularized horseshoe** — https://arxiv.org/abs/1707.01694 — derivation of
  $\tau_0$ from expected #nonzeros and the finite slab; the reference for the sparsity section (Ch 14 deep-dive).
- **Austin Rochford, "The Hierarchical Regularized Horseshoe Prior in PyMC"** —
  https://austinrochford.com/posts/2021-05-29-horseshoe-pymc3.html — worked PyMC build (porting to v5 is
  mechanical); best code template for the horseshoe example.
- **PyMC core notebook: Prior and Posterior Predictive Checks** —
  https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html — the
  `standardize`/`sample_prior_predictive` idiom and the ±40-SD flat-prior cautionary tale; copy this pattern.
- **PyMC `LKJCholeskyCov` API** —
  https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html —
  verified params (`eta`,`n`,`sd_dist`,`compute_corr`) and the runnable example; cite for Ch 10 too.
- **PyMC `sample_prior_predictive` API** —
  https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_prior_predictive.html —
  verified `draws=500` default and `prior`/`prior_predictive` groups.
- **Kallioinen, Paananen, Bürkner, Vehtari (2023), "Detecting and diagnosing prior and likelihood
  sensitivity with power-scaling"** — https://arxiv.org/abs/2107.14054 (Stat. & Comp. 34:57) +
  package https://n-kall.github.io/priorsense/ — the modern, principled prior **sensitivity** method;
  `powerscale_sensitivity`. (R-only today; for Python, frame conceptually and do manual re-fits.)
- **Bayes Rules! Ch 5 "Conjugate Families"** — https://www.bayesrulesbook.com/chapter-5 — clean, verified
  conjugate update formulas and pseudo-count intuition; good learner-facing reference.
- **John D. Cook, "A Compendium of Conjugate Priors"** —
  https://www.johndcook.com/CompendiumOfConjugatePriors.pdf — exhaustive table of conjugate pairs for the
  appendix cheat-sheet.
- **Bambi default-priors paper (Westfall 2017)** — https://arxiv.org/abs/1702.01201 — explains exactly how
  Bambi auto-scales weakly-informative priors (rstanarm-style); cite when showing Bambi defaults.
- **McElreath, *Statistical Rethinking* 2e, Ch 4** (PyMC port: `pymc-devs/pymc-resources`) — the
  prior-predictive-for-height example this course's Howell section mirrors; build-intuition source.
- **BDA3 (Gelman et al. 2013), Ch 2–5** — http://www.stat.columbia.edu/~gelman/book/ — conjugacy,
  noninformative priors, weakly-informative priors, hierarchical variance priors; the reference text.

## Uncertainties / things the writer should double-check

- **`priorsense` power-scaling has no first-class Python/PyMC port** that I could verify; treat it as a
  *concept* in Ch 03 and demonstrate sensitivity by re-fitting under 2–3 alternative priors, or apply
  power-scaling manually via importance weights on the log-prior. Do not claim a `pm.`/`az.` function exists.
- **Exact PC-prior closed forms** beyond "Exponential on the distance scale" are model-specific (PC for a
  precision → a type-2 Gumbel / Exponential-on-`σ`-like form). If writing the formula for a specific
  parameter (e.g. GP range, AR coefficient), re-derive from Simpson 2017 §2–3 rather than guessing.
- The Simpson et al. (2017) **arXiv id 1403.4630** vs the Statistical Science version — confirm the link
  resolves; both should point to the same paper. (Search also surfaced a "Some comments…" reply,
  arXiv:1609.06968 — that is a *commentary*, not the original.)
- **Bambi's exact default scale factors** evolve across versions; show defaults by inspecting the built
  model (`model.build(); print(model)`) rather than hardcoding numbers that may drift.
- Whether the course wants `pm.HalfStudentT` (verify spelling in current PyMC; half-t is often done via
  `pm.StudentT` truncated or `pm.HalfCauchy`/`pm.HalfNormal` — confirm `HalfStudentT` exists in installed
  version before using it). HalfNormal, HalfCauchy, Exponential, InverseGamma are all confirmed present.
- The `student_t(3,0,1)` "Vehtari robustness" default and the `3<nu<7` coefficient guidance come from the
  Stan wiki's evolving text; quote as "current Stan-wiki guidance" rather than a fixed standard.
