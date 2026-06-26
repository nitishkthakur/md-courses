# Appendix A — Cheat-Sheets & Resources

### The quick-reference you keep open in the other monitor

> *A note from me to you.*
>
> *Every working Bayesian I know keeps a scratchpad of half-remembered facts: which PyMC distribution takes a rate and which takes a scale, what R̂ value is "good enough," what to do the moment NUTS spits out "There were 47 divergences after tuning." You learned all of this across the eighteen chapters of this course — slowly, with intuition and derivations and worked examples, the way it should be learned. This appendix is the other half of that pedagogy: the place you come back to **after** you understand, when you just need the answer fast at 11pm with a deadline tomorrow.*
>
> *Treat it as a map, not a textbook. Every entry is a pointer back into a chapter where the idea was earned. If a table cell surprises you — "wait, why is `eta` larger meaning *less* correlation?" — that surprise is the signal to go re-read the chapter, not to memorize the cell. A cheat-sheet you don't understand is just superstition with good formatting.*
>
> *I have tried to make this genuinely useful as a standalone reference — the kind of thing you could print and tape next to your desk. Use it that way.*

---

## What you'll be able to do after this chapter

- **Pick a distribution** for any prior or likelihood in seconds, with the correct PyMC v5 spelling and parameterization (σ, not precision; rate vs scale).
- **Choose a default prior** for any parameter *type* — intercept, standardized slope, scale, correlation matrix, probability, overdispersion, GP lengthscale — and say *why* in one sentence.
- **Read every diagnostic** against a single consistent threshold table (R̂ ≤ 1.01, ESS ≳ 400, k̂ < 0.7, BFMI > 0.3, 0 divergences, treedepth not maxed).
- **Debug fast** with one master troubleshooting table that aggregates every symptom → cause → fix from the whole course.
- **Look up any term** in a glossary that fixes the course's vocabulary.
- **Find the canonical source** — book chapter, paper, doc page, or blog — for any topic, organized so you can go deeper.
- **Plan what to learn next** with a roadmap into the territory this course deliberately left at its edges.

A note on how to read the code-shaped cells: PyMC spellings are written as you'd type them, e.g. `pm.Normal("mu", mu=0, sigma=1)`. Throughout, assume the course-standard header:

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

---

## 1. Distribution cheat-sheet

This is the zoo. Every distribution you reached for in this course, with its **support** (where it lives), its **PyMC v5 parameterization** (the spelling that actually runs — and we always parameterize by **standard deviation σ, never variance, never precision τ**; BUGS/JAGS used precision and that mismatch is a classic footgun), its **typical use**, and a **note** capturing the one thing people get wrong.

> 💡 **Intuition:** A distribution is a small machine for generating numbers. When you choose one as a *prior*, you are saying "before I see data, here is the kind of number this parameter plausibly is." When you choose one as a *likelihood*, you are saying "given the parameter, here is how the data is generated." Same machines, two jobs.

### Continuous distributions

| Distribution (PyMC) | Support | Parameters (PyMC spelling) | Typical use | Notes / footguns |
|---|---|---|---|---|
| `pm.Normal` | $(-\infty,\infty)$ | `mu`, `sigma` | Intercepts, standardized slopes, Gaussian likelihood, non-centered offsets | **`sigma` is SD, not variance, not precision.** The default workhorse. |
| `pm.StudentT` | $(-\infty,\infty)$ | `nu`, `mu`, `sigma` | Robust regression/likelihood; outlier-tolerant slope priors | Smaller `nu` → heavier tails; `nu→∞` → Normal. `nu≈3–7` is the robust sweet spot. |
| `pm.Cauchy` | $(-\infty,\infty)$ | `alpha` (loc), `beta` (scale) | Very heavy-tailed prior (rare) | = StudentT with `nu=1`; no mean/variance. Tails so heavy NUTS can struggle. |
| `pm.Laplace` | $(-\infty,\infty)$ | `mu`, `b` (scale) | Bayesian **lasso** / L1-style sparsity | MAP under Laplace = L1 penalty. Sharper peak than Normal. |
| `pm.HalfNormal` | $[0,\infty)$ | `sigma` | **Default scale/SD prior**; group-level σ | Light tails. First reach for scale params on standardized data. |
| `pm.HalfCauchy` | $[0,\infty)$ | `beta` (scale) | Scale prior when you won't rule out large σ (Gelman 2006) | Heavy tail can slow NUTS; finite scale helps. |
| `pm.Exponential` | $[0,\infty)$ | `lam` (**rate** λ) | Scale/SD prior; PyMC's clean one-param default | **`lam` is a rate**: mean $=1/\lambda$. `Exponential(1)` ⇒ mean 1. |
| `pm.Gamma` | $(0,\infty)$ | `alpha`,`beta` (rate) **or** `mu`,`sigma` | Positive params; rate priors; NB overdispersion; conjugate to Poisson | `beta` is a **rate**. Has a `mu,sigma` parameterization too — pick one. |
| `pm.InverseGamma` | $(0,\infty)$ | `alpha`, `beta` | GP lengthscale priors; (historically) variance priors | **Avoid `InvGamma(ε,ε)` on hierarchical variances** (Gelman 2006). Fine for GP range. |
| `pm.Beta` | $(0,1)$ | `alpha`, `beta` | Probabilities; mixing weights; conjugate to Binomial | `Beta(1,1)=Uniform(0,1)`; `Beta(0.5,0.5)`=Jeffreys. Pseudo-counts. |
| `pm.Uniform` | $[lower,upper]$ | `lower`, `upper` | Bounded params with a hard range; *rarely* a good prior | Hard edges distort posteriors; prefer soft (Normal/Half) priors. |
| `pm.LogNormal` | $(0,\infty)$ | `mu`, `sigma` (of log) | Positive, right-skewed quantities (incomes, durations) | `mu`,`sigma` are on the **log** scale. Median $=e^\mu$. |
| `pm.Weibull` | $(0,\infty)$ | `alpha` (shape), `beta` (scale) | Survival / time-to-event | Generalizes Exponential (`alpha=1`). |
| `pm.VonMises` | circular | `mu`, `kappa` | Angles / circular data | `kappa` is a concentration (like precision). |

### Multivariate & matrix-valued

| Distribution (PyMC) | Support | Parameters | Typical use | Notes |
|---|---|---|---|---|
| `pm.MvNormal` | $\mathbb{R}^k$ | `mu`, and `cov` **or** `chol` **or** `tau` | Correlated parameters; GP latent values | Prefer `chol=` (Cholesky) for speed/conditioning over `cov=`. |
| `pm.Dirichlet` | simplex | `a` (concentration vector) | Mixture weights; categorical probabilities | `a=np.ones(k)` is flat on the simplex; larger `a` → near-uniform weights. |
| `pm.LKJCorr` | corr. matrices | `eta`, `n` | Correlation matrix prior; **prefer `LKJCholeskyCov` for modeling** (next row) | Returns the **upper-triangular** off-diagonal entries (a flat vector), *not* an $(n,n)$ matrix — pass `return_matrix=True` (recent PyMC) for the full matrix. `eta=1` uniform; **larger `eta` ⇒ *more* shrinkage to identity** (less correlation). |
| `pm.LKJCholeskyCov` | covariance | `eta`, `n`, `sd_dist` | **Preferred** corr+scale prior for varying slopes (Ch 10) | `sd_dist` must be built with `.dist()`. `compute_corr=True` returns `(chol, corr, sigmas)`. |
| `pm.Wishart` / `pm.LKJCholeskyCov` | covariance | — | Covariance priors | Prefer LKJ-Cholesky over Wishart in practice (better behaved). |

### Discrete distributions

| Distribution (PyMC) | Support | Parameters | Typical use | Notes |
|---|---|---|---|---|
| `pm.Bernoulli` | $\{0,1\}$ | `p` **or** `logit_p` | Binary outcomes (logistic) | Use `logit_p=` to avoid a manual sigmoid and improve numerics. |
| `pm.Binomial` | $\{0,\dots,n\}$ | `n`, `p` | Counts of successes in `n` trials | Conjugate prior is Beta. |
| `pm.Poisson` | $\{0,1,2,\dots\}$ | `mu` (= rate λ) | Count outcomes (log link) | **Mean = variance**. If data over-dispersed → use NegBinomial. |
| `pm.NegativeBinomial` | $\{0,1,\dots\}$ | `mu`, `alpha` | **Over-dispersed** counts | `alpha`→∞ recovers Poisson; small `alpha` ⇒ heavy over-dispersion. |
| `pm.Categorical` | $\{0,\dots,k{-}1\}$ | `p` (vector) | Multinomial logit / mixture component labels | Often marginalized out in mixtures (Ch 13). |
| `pm.ZeroInflatedPoisson` | $\{0,1,\dots\}$ | `psi`, `mu` | Excess zeros (structural + sampling zeros) | `psi` = prob of the "count" component. See also hurdle models. |
| `pm.ZeroInflatedNegativeBinomial` | $\{0,1,\dots\}$ | `psi`, `mu`, `alpha` | Excess zeros **and** over-dispersion | The pragmatic default for messy count data. |

> ⚠️ **Pitfall:** The three most common parameterization mistakes, in order of how often they bite: (1) passing a **variance or precision** where PyMC wants `sigma` (a standard deviation) — your priors will be wildly wrong; (2) forgetting that `pm.Exponential`'s `lam` and `pm.Gamma`'s `beta` are **rates**, so `Exponential(0.1)` has mean 10, not 0.1; (3) building `LKJCholeskyCov`'s `sd_dist` as a *named* RV instead of with `.dist()`. When in doubt, run `pm.sample_prior_predictive` and *look* at the numbers the machine produces.

### Conjugate pairs (the analytic shortcuts from Chapter 04)

A handful of prior–likelihood pairs give you the posterior in closed form — no MCMC needed. They are worth memorizing because they build the intuition that *every* posterior is a compromise between prior and data, and because they make great unit tests (fit with MCMC, check you recover the analytic answer).

| Likelihood | Conjugate prior | Posterior | Posterior mean reads as |
|---|---|---|---|
| Binomial / Bernoulli ($y$ successes in $n$) | $\theta\sim\mathrm{Beta}(\alpha,\beta)$ | $\theta\mid y\sim\mathrm{Beta}(\alpha+y,\ \beta+n-y)$ | $\frac{\alpha+y}{\alpha+\beta+n}$ — prior pseudo-counts $\alpha,\beta$ blended with data |
| Poisson ($\sum y_i$ over $n$ obs) | $\lambda\sim\mathrm{Gamma}(s, r)$ (rate $r$) | $\lambda\mid y\sim\mathrm{Gamma}(s+\sum y_i,\ r+n)$ | $\frac{s+\sum y_i}{r+n}$ |
| Normal, known $\sigma^2$ (mean unknown) | $\mu\sim\mathrm{Normal}(\theta,\tau^2)$ | $\mu\mid y\sim\mathrm{Normal}(\mu_n,\tau_n^2)$ | precision-weighted average: $1/\tau_n^2 = 1/\tau^2 + n/\sigma^2$ |
| Normal, known mean (variance unknown) | $\sigma^2\sim\mathrm{InvGamma}(\alpha,\beta)$ | $\sigma^2\mid y\sim\mathrm{InvGamma}(\alpha+\tfrac n2,\ \beta+\tfrac12\sum (y_i-\mu)^2)$ | (note the IG-on-variance caveat below) |

> 💡 **Intuition:** Read every conjugate posterior as **"prior pseudo-data + real data."** The Beta–Binomial says: pretend you'd already seen $\alpha$ successes and $\beta$ failures before the experiment. The Normal–Normal says: posterior precision is the *sum* of prior precision and data precision — adding information always sharpens. That precision-weighted-average formula is exactly the partial-pooling formula you met in Chapter 10; conjugacy is hierarchical shrinkage with one level.

---

## 2. Prior cheat-sheet — by parameter *type*

Stop choosing priors distribution-by-distribution and start choosing them by **what the parameter is**. This table is the one I use most. It assumes the course default: **predictors are standardized** (mean 0, SD ≈ 0.5 per Gelman, or SD 1) and where sensible the outcome is standardized too, so that a single unit-scale prior is sane across models. Establish that habit (Chapter 03) and these defaults just work.

| Parameter type | Recommended default prior | Why (one sentence) |
|---|---|---|
| **Intercept** (centered/standardized outcome) | `pm.Normal("a", 0, 1)` (or `1.5`) | After centering, the intercept is "the outcome at the average predictor," which sits near 0 with unit-ish spread. |
| **Standardized slope / regression coef** | `pm.Normal("b", 0, 1)`; robust: `pm.StudentT("b", nu=3, mu=0, sigma=1)` | On standardized predictors, a 1-SD change rarely moves the outcome by many SDs; `Normal(0,1)` rules out the absurd while staying agnostic. StudentT adds outlier tolerance (current Stan-wiki guidance, `3<nu<7`). |
| **Scale / standard deviation** ($\sigma$, residual or group-level) | `pm.HalfNormal("s", 1)` or `pm.Exponential("s", 1)` | Light-tailed, single-parameter, defined on $[0,\infty)$; on standardized data the scale is order-1. **Never `InvGamma(ε,ε)` on the variance.** |
| **Scale you won't bound from above** (might be large) | `pm.HalfCauchy("s", 1)` or half-$t$(4) | Heavy tail keeps large values possible; Gelman (2006) default for hierarchical $\tau$. Watch for slow NUTS. |
| **Correlation matrix** (varying slopes) | `pm.LKJCholeskyCov(..., eta=2, ...)` | `eta=2` mildly regularizes toward independence (the matrix analogue of `Beta(2,2)`); Cholesky form samples best. **Larger `eta` ⇒ less correlation.** |
| **Probability** $p\in(0,1)$ | `pm.Beta("p", 2, 2)`; or `Normal(0,1.5)` on the **logit** | `Beta(2,2)` is gently centered, avoids the 0/1 edges; modeling on the logit scale plays nicer with covariates. |
| **Simplex / mixture weights** | `pm.Dirichlet("w", a=np.ones(k))` | Flat over the simplex; raise `a` to pull weights toward equal, lower to allow dominance. |
| **Over-dispersion** (NegBinomial `alpha`) | `pm.Gamma("alpha", 2, 0.1)` or `InvGamma`-style; Stan-wiki suggests `InvGamma` on `1/sqrt(alpha)` | Keeps `alpha` away from 0 (which would blow up the variance) while allowing the Poisson limit. |
| **Degrees of freedom** (StudentT `nu`) | `pm.Gamma("nu", 2, 0.1)` (mean ≈ 20) | Puts almost no mass below ~2 — not a hard floor, but the Student-t variance is effectively always finite — while letting the data choose tail heaviness. |
| **GP lengthscale** $\ell$ | `pm.InverseGamma("ls", ...)` tuned to your domain; or a PC prior | IG suppresses both implausibly tiny and infinite lengthscales — the two pathologies of GP fitting (Ch 12). Set it from the range of your inputs. |
| **GP amplitude / marginal SD** $\eta$ | `pm.HalfNormal("eta", 1)` (on standardized outcome) | Same logic as any scale prior; bounds how wiggly the function's *output* can be. |
| **Horseshoe global shrinkage** $\tau$ | `HalfCauchy` with $\tau_0$ set from expected #nonzeros | Global $\tau$ shrinks everything; local $\lambda_j$ let true signals escape — a continuous spike-and-slab (Piironen & Vehtari 2017). |

> 🔧 **In practice:** The recipe behind this whole table is just three moves. **(1) Standardize** so "order 1" is meaningful. **(2) Pick a family by support** — real line → Normal/StudentT; positive → HalfNormal/Exponential; bounded → Beta; matrix → LKJ. **(3) Set the scale by a prior predictive check** (Chapter 03): simulate outcomes from the prior, plot them, and ask "are these the kind of data I could conceivably see?" If your prior-predictive heights run from −40 to +40 standard deviations, your prior is too wide — tighten it. That single check, not a lookup table, is the real decision procedure.

> 📜 **Citation/Origin:** The "level 1–5" prior hierarchy (flat → super-vague → weak → generic weakly-informative `Normal(0,1)`/`StudentT(3,0,1)` → specific informative) is from the **Stan Prior Choice Recommendations** wiki. The half-Cauchy-on-σ recommendation and the case against `InvGamma(ε,ε)` are **Gelman (2006)**. The `eta=2` LKJ default and the regularized horseshoe are from the LKJ paper and **Piironen & Vehtari (2017)** respectively. Full links in §6.

---

## 3. Diagnostic thresholds — the green/red table

After every fit you run the same checklist. This is it, with the **one-line meaning** of each number and the **ArviZ symbol** that produces it. These thresholds are used identically throughout the course (Chapters 07, 08, 11) — quote them consistently and you'll never contradict yourself.

| Diagnostic | Target | What it means | How to get it |
|---|---|---|---|
| **R̂ (R-hat)** | **≤ 1.01** | Chains agree with each other *and* with themselves — convergence to a common distribution. Above 1.01, your chains haven't found the same posterior. | `az.summary(idata)["r_hat"]`, `az.rhat(idata)` |
| **ESS (bulk)** | **≳ 400** total (≈100/chain) | Effective independent draws for the *center* of the distribution (means, medians). Low bulk-ESS → noisy point estimates. | `az.summary` `ess_bulk`; `az.ess(idata, method="bulk")` |
| **ESS (tail)** | **≳ 400** total | Effective draws for the *tails* (5%/95% quantiles, interval endpoints). A model can mix in the bulk but not the tails. | `az.summary` `ess_tail`; `az.ess(idata, method="tail")` |
| **Divergences** | **exactly 0** | Trajectories where the leapfrog integrator failed at high curvature; they mark regions the sampler systematically *fails to explore* → biased posterior. | `int(idata.sample_stats["diverging"].sum())` |
| **BFMI (E-BFMI)** | **> 0.3** | Energy Bayesian Fraction of Missing Information — whether momentum resampling can traverse the full energy range. Below 0.3 → heavy tails / poor energy exploration. | `az.bfmi(idata)` (per chain); `az.plot_energy(idata)` |
| **Pareto k̂** | **< 0.7** | Reliability of the PSIS-LOO estimate at each data point; also flags influential/outlier points. ≥ 0.7 unreliable, ≥ 1.0 the estimand has no finite mean. ArviZ also reports an adaptive threshold `good_k = min(1 − 1/log10(S), 0.7)` (S = #draws) on the `ELPDData` object — use it when present. | `az.loo(idata, pointwise=True).pareto_k`; adaptive: `az.loo(idata).good_k` |
| **Max treedepth** | **not maxed** | If many draws hit `max_treedepth` (default 10 → 1024 leapfrog steps), NUTS is truncating trajectories — an *efficiency* smell (poorly-scaled metric), not a bias. | `(idata.sample_stats["tree_depth"] == 10).mean()` |
| **MCSE** | small vs `sd` | Monte Carlo standard error of an *estimate*: how many digits of a posterior mean you can trust. `MCSE ≈ sd/√ESS`. | `az.summary` `mcse_mean`, `mcse_sd`; `az.mcse(idata)` |

> 🩺 **Diagnostic:** Read these in order and **stop at the first red light**. (1) Divergences first — even a handful means "do not trust this posterior yet," because they signal a region the sampler can't reach. (2) R̂ next — if chains disagree, nothing downstream means anything. (3) ESS — once converged, do you have *enough* draws? (4) BFMI / treedepth — efficiency and geometry smells. (5) Only when sampling is clean do you turn to **predictive** checks (k̂, LOO-PIT, PPC) to ask whether the model is any *good*. A clean sampler of a wrong model is still a wrong model.

> 🧮 **The math:** **Rank-normalized split-R̂** (Vehtari et al. 2021) fixes classic Gelman–Rubin R̂ by (1) *splitting* each chain in half to catch within-chain trends, (2) *rank-normalizing* the pooled draws so infinite-variance marginals don't break it, and (3) reporting the **max** of R̂ on the draws and "folded" R̂ (absolute deviations from the median), which senses differences in **scale**, not just location. That is exactly what `az.rhat(..., method="rank")` and the `r_hat` column compute. **Bulk-ESS** uses rank-normalized draws (central tendency); **tail-ESS** is the min of ESS at the 5% and 95% quantiles (interval reliability). Both come from $\mathrm{ESS}=N/(1+2\sum_t\rho_t)$, the autocorrelation-discounted draw count.

> ⚠️ **Pitfall:** **R̂ is `NaN` with a single chain** — it needs ≥ 2 chains to compare. Always sample `chains=4`. And note two threshold footnotes: the older R̂ ≤ 1.1 is *too loose* (use 1.01); Stan historically warned on BFMI < 0.2 while this course standardizes on **< 0.3** as the flag (so don't panic if Stan stays quiet where ArviZ would warn).

---

## 4. Master troubleshooting table

This is the big one — the aggregate of every "common errors & fixes" box from every chapter, organized so you can scan by **symptom** (the message or number you actually see) and jump to **cause** and **fix**. When NUTS yells at you, start here.

### Sampling & geometry (Chapters 05, 07, 10)

| Symptom (message / number) | Cause | Fix |
|---|---|---|
| `There were N divergences after tuning` | Step size too large for a high-curvature region (funnel neck); pathological geometry | Raise `target_accept` (0.9 → 0.95 → 0.99); **reparameterize non-centered**; tighten/regularize scale priors; standardize predictors |
| Divergences only at small `tau`, invisible on a linear axis | Neal's funnel — the neck is squeezed | Add `pm.Deterministic("log_tau", pm.math.log(tau))` and `az.plot_pair(..., var_names=["theta","log_tau"], divergences=True)` |
| `r_hat > 1.01` (e.g. 1.3) | Chains not converged / multimodal / too little tuning / a stuck chain | More `tune`+`draws`; better `init`; reparameterize; inspect `az.plot_trace`/`az.plot_rank` for the offending chain |
| `ess_bulk` / `ess_tail` tiny (e.g. 30) | High autocorrelation / poor geometry / strong correlations | Non-center; raise `target_accept`; longer chains; rescale predictors; `init="jitter+adapt_full"` for correlated posteriors |
| `BFMI < 0.3` / energy warning | Heavy tails; momentum resampling can't traverse the energy range | Reparameterize; lighter-tailed or tighter priors; revisit the scale-parameter prior |
| Many `Maximum tree depth reached` | Trajectories truncated — metric poorly scaled or step size tiny | `init="jitter+adapt_full"`; standardize; raise `tune`; *only then* `nuts={"max_treedepth": 12}` |
| `The acceptance probability does not match the target` | Adaptation didn't converge — too few tuning steps | Increase `tune` to 2000–3000; set `init="jitter+adapt_diag"` explicitly |
| `r_hat = NaN` in summary | Only 1 chain (R̂ needs ≥2), or a constant/deterministic | Sample `chains=4`; ignore for fixed Deterministics |
| Sampling very slow, low ESS, **no** divergences | Strong posterior correlations a diagonal metric can't capture | `init="jitter+adapt_full"` (dense metric), or reparameterize to decorrelate |
| `SamplingError: Initial evaluation ... results in -inf` | Bad initial point / value impossible under the prior (obs outside support, log of ≤0) | Set `initvals=`; fix priors/transforms; check the `observed` data domain; use `logit_p`/`log` parameterizations |
| `BlasError` / multiprocessing hangs on some OS | mp-context or BLAS thread contention | `cores=1` to debug; `mp_ctx="spawn"`; `blas_cores=1` |
| Discrete params + NUTS: "Cannot sample" | NUTS needs gradients; discrete vars unsupported directly | **Marginalize** the discrete var (Ch 13), or rely on PyMC's auto `CompoundStep` (`*GibbsMetropolis`) |
| Metropolis "works" but ESS tiny in high-D | Random-walk diffusion — the typical-set problem | Use NUTS (gradient-based). This *is* the lesson of Chapter 05. |

### Priors & model specification (Chapters 02, 03, 09, 10)

| Symptom | Cause | Fix |
|---|---|---|
| Prior-predictive `obs` ranges to ±40 SD; nonsensical $y$ | Priors too wide for the (standardized) scale | Tighten to unit-scale weakly-informative; standardize predictors **and** outcome first |
| Group SD collapses to ~0, very sensitive to settings | `InverseGamma(ε,ε)` prior on the variance (Gelman 2006) | Use `pm.HalfNormal`/`pm.Exponential`/half-$t$ on the SD `σ` — never IG(ε,ε) on `σ²` |
| Posterior barely moves from the prior on small data | Prior unintentionally too informative | Run a prior predictive check + sensitivity (re-fit under 2–3 priors); widen |
| Improper posterior / sampler explodes | Flat/improper prior on a variance, or under separation | Use a proper weakly-informative prior |
| `sd_dist` error / wrong shape in `LKJCholeskyCov` | Passed a *named* RV or wrong shape | Build with `.dist()` (`pm.Exponential.dist(1., size=n)`); ensure `shape[-1]==n` |
| LKJ gives "all correlations ≈ 0" unexpectedly | `eta` too large (concentrates on the identity) | Lower `eta` toward 1–2; remember **larger `eta` ⇒ less correlation** |
| Logistic coefs run off to ±∞; perfect in-sample fit | **Separation** — a predictor perfectly splits the classes | Add a weakly-informative prior on coefs (`Normal(0, 1)` / `StudentT(7,0,2.5)`); this regularizes the divergence away |
| Coefs on raw predictors need absurd priors | Predictors on wildly different scales | Standardize (mean 0, SD ≈ 0.5) so one unit-scale prior fits all |
| `sample_prior_predictive(samples=...)` TypeError | `samples` arg removed/renamed in v5 | Use `draws=` |

### Comparison, calibration & predictive checks (Chapters 08, 11)

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: log_likelihood not found` in `az.loo`/`az.waic` | PyMC v5 doesn't store pointwise log-lik by default | `pm.sample(..., idata_kwargs={"log_likelihood": True})` **or** `with model: pm.compute_log_likelihood(idata)` |
| log_likelihood missing when likelihood is a `pm.Potential` | `compute_log_likelihood` only covers *observed* RVs, not `Potential` | Express the likelihood as observed `pm.<Dist>("y", ..., observed=y)`; for custom densities use `pm.CustomDist` with a `logp` (not a bare `Potential`) |
| "Pareto k̂ > 0.7" warning from `az.loo` | Heavy-tailed LOO importance ratios / influential points | Refit those points with **K-fold CV** or PSIS moment-matching; report which observations; treat as a robustness flag |
| `az.compare` errors about mismatched `n_data_points` | Models fit on different data / transformed `y` / different obs var | Compare only models on the **same observations, same scale**; set `var_name=` if multiple observed vars |
| Comparing a model on `log(y)` vs `y` and one "wins" | ELPD is on different scales; a Jacobian is missing | Put both on the *same* response; add the change-of-variables Jacobian if you transform `y` |
| WAIC and LOO disagree a lot | `p_waic` unstable (var > 0.4 points), small n, flexible model | Trust **LOO**; investigate high-k̂ points; consider K-fold |
| Over-trusting a 0.3-ELPD "winner" | `elpd_diff < 4`, well within `dse` | Report "no meaningful difference"; don't select on noise |
| `weight = 0` for a model with small `elpd_diff` | Stacking allocates all weight to a near-equivalent better model | Expected; for softer weights try `method="BB-pseudo-BMA"` |
| Bayes factor flips when you widen a prior | Marginal likelihood = an average over the *prior* | Don't use BFs here; use ELPD/LOO, or commit to proper informative priors + **bridge sampling** |
| `az.plot_ppc`: "no posterior_predictive group" | Forgot `sample_posterior_predictive` | `pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=RANDOM_SEED)` |
| PPC looks perfect but the model is wrong | Double-dipping; test statistic uninformative | Use **LOO-PIT** + informative `T` (min, max, sd, #zeros), not location stats |
| LOO-PIT is ∪-shaped | Predictions under-dispersed / over-confident | Add over-dispersion (Poisson→NegBinomial), heavier-tailed likelihood, or a missing variance component |
| LOO-PIT is ∩-shaped | Predictions over-dispersed / under-confident | Tighten the likelihood; remove an over-flexible component |

### Latent variables, mixtures & special structure (Chapters 13, 16)

| Symptom | Cause | Fix |
|---|---|---|
| **Label switching** — components swap identity across chains; bimodal `az.plot_trace` | Mixture likelihood is invariant to permuting component labels | Impose an **ordered** constraint (`pm.Normal(..., transform=pm.distributions.transforms.ordered)` on the means) so components are identifiable |
| NUTS can't sample a mixture with a discrete component label | Discrete latent + gradient sampler | **Marginalize** the label (`pm.Mixture` / `log_sum_exp`); PyMC's `pm.Mixture` does this for you |
| Imputation params explode or won't sample | Missingness modeled on the wrong scale; MNAR treated as MAR | Model imputed values as parameters with sane priors; if MNAR, model the missingness *mechanism* (selection model, Ch 16) |
| Funnel returns in a non-centered model | A *different* parameter is now the tight one, or data is informative per group | Centered can mix *better* with strong per-group data; profile and switch the parameterization per parameter |

> 🔧 **In practice:** Ninety percent of real sampling pain reduces to three moves: **standardize your data, non-center your hierarchy, and raise `target_accept`** — in that order. Reach for `max_treedepth`, dense metrics, and exotic priors only after those three have failed. And always — *always* — run a prior predictive check before you blame the sampler; half of "divergences" are really a prior that put mass on impossible regions.

---

## The 60-second worked example: the whole appendix on one dataset

A cheat-sheet earns its keep when it turns a real fit into a checklist. Let me show you the entire appendix exercised on the one dataset that appears in more chapters than any other — **Eight Schools** (Ch 07, 08, 10, 11). Eight schools ran an SAT-coaching program; we have an estimated effect $y_j$ and its standard error $\sigma_j$ for each. The question — *is there one true effect, or eight, or something in between?* — is the canonical motivation for hierarchical modeling, and the centered version is the canonical funnel. This single run touches every table above: the prior cheat-sheet, the diagnostic thresholds, the troubleshooting table, and model comparison.

```python
import numpy as np
import pymc as pm
import arviz as az

RANDOM_SEED = 8927
J = 8
y     = np.array([28., 8., -3., 7., -1., 1., 18., 12.])   # estimated effects
sigma = np.array([15., 10., 16., 11., 9., 11., 10., 18.])  # their standard errors
coords = {"school": [f"S{j}" for j in range(J)]}

# --- Model A: CENTERED hierarchy (the funnel; expect divergences) ---
with pm.Model(coords=coords) as centered:
    mu    = pm.Normal("mu", mu=0, sigma=5)            # population mean: location -> Normal
    tau   = pm.HalfCauchy("tau", beta=5)             # group SD: scale we won't bound -> HalfCauchy (§2, Gelman 2006)
    pm.Deterministic("log_tau", pm.math.log(tau))    # log-axis handle so the funnel neck is visible
    theta = pm.Normal("theta", mu=mu, sigma=tau, dims="school")  # <-- centered
    pm.Normal("obs", mu=theta, sigma=sigma, observed=y, dims="school")
    idata_c = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                        random_seed=RANDOM_SEED, idata_kwargs={"log_likelihood": True})

# --- Model B: NON-CENTERED hierarchy (the cure) ---
with pm.Model(coords=coords) as noncentered:
    mu      = pm.Normal("mu", mu=0, sigma=5)
    tau     = pm.HalfCauchy("tau", beta=5)
    theta_t = pm.Normal("theta_t", mu=0, sigma=1, dims="school")     # raw std-normal offset
    theta   = pm.Deterministic("theta", mu + tau * theta_t, dims="school")  # affine recover
    pm.Normal("obs", mu=theta, sigma=sigma, observed=y, dims="school")
    idata_nc = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                         random_seed=RANDOM_SEED, idata_kwargs={"log_likelihood": True})
```

Now run the **diagnostic thresholds table** (§3) on each, in the order §3 prescribes:

```python
for name, idata in [("centered", idata_c), ("non-centered", idata_nc)]:
    n_div = int(idata.sample_stats["diverging"].sum())
    print(name, "divergences:", n_div)          # centered: typically 20-100+; non-centered: ~0
print(az.summary(idata_nc, var_names=["mu", "tau"]))   # scan r_hat<=1.01, ess_bulk/tail>=400
az.plot_energy(idata_nc)                                # BFMI>0.3, the two energy densities overlap
```

**How to read what comes back.** The centered model will report a clutch of divergences and a `tau` whose `ess_bulk` is embarrassingly low — exactly the **"There were N divergences after tuning"** row of the troubleshooting table, whose fix is *"reparameterize non-centered."* That is precisely what Model B does, and its summary should show `r_hat = 1.00`, `ess_bulk`/`ess_tail` comfortably in the thousands, and zero divergences. To *see* the cause, plot the funnel on a log axis — that is exactly why the centered model carries `pm.Deterministic("log_tau", pm.math.log(tau))`: on a *linear* `tau` axis the divergences pile up against 0 and stay invisible (the troubleshooting row *"Divergences only at small `tau`, invisible on a linear axis"*), whereas pairing `theta` against `log_tau` stretches the neck open so the cluster is unmistakable:

```python
az.plot_pair(idata_c, var_names=["theta", "log_tau"],
             coords={"school": ["S0"]}, divergences=True)
# newer ArviZ: az.plot_pair(..., visuals={"divergence": True})
```

The divergent draws (highlighted) pile up at the **bottom of the funnel**, the small-`tau` neck where curvature explodes and a fixed leapfrog step can't keep up. That picture *is* the geometry behind the divergence count.

Before trusting any comparison, close the loop with a **posterior predictive check** (§4, predictive-checks rows; full machinery in Ch 08/11) — and, if you want the whole prior→fit→diagnose→PPC→compare chain, a prior predictive check too:

```python
with noncentered:
    # prior predictive: do the priors imply sane school effects *before* seeing y?
    idata_nc.extend(pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED))
    # posterior predictive: do replicated y look like the observed effects?
    pm.sample_posterior_predictive(idata_nc, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
az.plot_ppc(idata_nc)   # observed y vs replicated draws; should overlap, not systematically miss
```

The `plot_ppc` density should sit comfortably inside the cloud of replicated curves — for Eight Schools the eight noisy effects are easily covered, so the PPC is unremarkable (the *point* here is that you ran it). Note `pm.sample_prior_predictive(draws=...)` uses `draws=`, not the retired v3 `samples=` (troubleshooting table), and `extend_inferencedata=True` writes the `posterior_predictive` group `az.plot_ppc` needs.

Finally, **model comparison** (§4, comparison block; full machinery in Ch 11). Both models describe the same eight observations, so LOO is a fair comparison:

```python
df = az.compare({"centered": idata_c, "non_centered": idata_nc})
print(df)              # columns: rank, elpd_loo, p_loo, elpd_diff, weight, se, dse, warning, scale
az.plot_compare(df)
```

> 🩺 **Diagnostic:** Here the two models are *statistically the same model* — they differ only in parameterization, not in their implied posterior — so you should expect `elpd_diff` well under 4 and `dse` to swamp it. That is the table teaching you its own most important rule: **a tiny `elpd_diff` within a `dse` is not a real difference.** The non-centered model doesn't predict *better*; it *samples* better. This is the cleanest illustration of why "fixed the sampler" and "improved the model" are different claims — and why you check divergences (§3) before you ever look at a comparison table.

> ⚠️ **Pitfall:** Notice the `idata_kwargs={"log_likelihood": True}` on both `pm.sample` calls. Without it, recent PyMC v5 does **not** store the pointwise log-likelihood, and `az.compare`/`az.loo` will raise the `KeyError: log_likelihood not found` from the troubleshooting table. The post-hoc fix is `with model: pm.compute_log_likelihood(idata)`.

---

## 5. Glossary

The vocabulary this course uses, fixed so the chapters never drift. Terms are grouped loosely by where they live in the workflow.

**Foundations & probability**

- **Prior** $p(\theta)$ — your beliefs about parameters *before* seeing data.
- **Likelihood** $p(y\mid\theta)$ — the data-generating model as a function of $\theta$.
- **Posterior** $p(\theta\mid y)$ — updated beliefs *after* data; $\propto$ prior × likelihood.
- **Evidence / marginal likelihood** $p(y)=\int p(y\mid\theta)p(\theta)\,d\theta$ — the normalizing constant; the average likelihood over the prior. Drives Bayes factors (and their fragility).
- **Posterior predictive** $p(\tilde y\mid y)$ — distribution of *new* data, integrating over posterior uncertainty.
- **Prior predictive** — same, integrating over the *prior*; the basis of prior predictive checks.
- **Conjugate prior** — a prior whose family is preserved in the posterior (Beta–Binomial, Gamma–Poisson), giving a closed-form update.
- **Generative model** — a model you can *simulate forward* from; the small-world description of how data arise.

**Priors**

- **Weakly informative prior** — rules out absurd values, stays agnostic within the plausible range (`Normal(0,1)` on standardized data). The course default.
- **Regularizing prior** — deliberately shrinks toward a null; Bayesian L2 (Normal/ridge), L1 (Laplace/lasso), sparsity (horseshoe).
- **Improper / flat prior** — doesn't integrate to 1; can silently yield improper posteriors and breaks predictive checks and Bayes factors.
- **PC prior (penalized complexity)** — a prior built by penalizing distance from a simple base model at a constant rate (Simpson et al. 2017).
- **LKJ prior** — prior over correlation matrices; `eta=1` uniform, larger `eta` concentrates on the identity (independence).
- **Horseshoe** — a continuous spike-and-slab sparsity prior: a global scale shrinks everything, local scales let true signals escape.
- **Standardization** — centering and scaling predictors (mean 0, SD ≈ 0.5 or 1) so a single unit-scale prior is sane. Course default.

**Inference & MCMC**

- **MCMC** — Markov Chain Monte Carlo; build a Markov chain whose stationary distribution is the posterior.
- **Detailed balance** — $p(\theta)T(\theta'\mid\theta)=p(\theta')T(\theta\mid\theta')$; a sufficient condition for the target to be stationary.
- **Metropolis–Hastings** — propose, then accept/reject to enforce balance; needs only the *unnormalized* posterior.
- **Gibbs sampling** — MH with full-conditional proposals (acceptance 1); needs conjugacy, struggles under correlation.
- **Typical set** — the thin shell where posterior *mass* (density × volume) concentrates in high dimensions; *not* the mode. The thing the sampler must explore.
- **HMC** — Hamiltonian Monte Carlo; augments parameters with momentum and follows gradient-driven dynamics along the typical set.
- **Leapfrog** — the symplectic, time-reversible integrator HMC uses (kick-drift-kick); $O(\epsilon^2)$ error per step.
- **NUTS** — No-U-Turn Sampler; HMC that auto-tunes trajectory length by recursive doubling until the path starts to U-turn.
- **`target_accept`** — the NUTS step-size knob; higher → smaller steps → fewer divergences but slower. Course default 0.9, raise to 0.95–0.99 for trouble.
- **Mass matrix / metric** — the kinetic-energy covariance NUTS adapts so the typical set looks isotropic; `adapt_diag` (default) vs `adapt_full` (dense).
- **Warmup / tuning** — the discarded phase where step size (dual averaging) and the metric are adapted.
- **Divergence** — a flagged leapfrog failure at high curvature; a truth-teller marking unexplored, biasing regions.
- **Non-centered parameterization** — reparameterizing $\theta_j=\mu+\tau\tilde\theta_j$ with $\tilde\theta_j\sim N(0,1)$ to decouple the geometry; the default funnel cure (Ch 07/10).
- **Funnel** — Neal's funnel; the squeezed neck (small group scale $\tau$) where centered hierarchies diverge.
- **VI / ADVI** — Variational Inference; approximate the posterior by optimizing an ELBO instead of sampling. Fast, but can be over-confident.
- **Pathfinder** — a fast variational/optimization-path posterior approximation, good for initialization.

**Diagnostics**

- **R̂ (R-hat)** — rank-normalized split potential-scale-reduction; convergence measure, target ≤ 1.01.
- **ESS (bulk / tail)** — effective sample size for central / tail quantities; autocorrelation-discounted draw counts, target ≳ 400.
- **MCSE** — Monte Carlo standard error of an estimate; $\approx \mathrm{sd}/\sqrt{\mathrm{ESS}}$.
- **BFMI / E-BFMI** — energy fraction of missing information; energy-exploration health, target > 0.3.
- **InferenceData** — the xarray-backed container PyMC returns; groups: `posterior`, `sample_stats`, `posterior_predictive`, `prior`, `prior_predictive`, `observed_data`, `log_likelihood`, `constant_data`. (The v3 name "trace" is retired.)
- **Rank plot** — per-chain histogram of pooled ranks; uniform = good mixing (a robust alternative to trace plots).
- **Energy plot** — overlays marginal vs transition energy densities; narrow transitions = poor exploration.

**Checking, comparison & calibration**

- **PPC (posterior predictive check)** — compare simulated $\tilde y$ to observed $y$; `az.plot_ppc`.
- **Bayesian p-value** — $\Pr(T(\tilde y)\ge T(y)\mid y)$ for a test statistic $T$; ≈ 0.5 is well-calibrated, near 0/1 flags misfit.
- **LOO-PIT** — leave-one-out probability integral transform; uniform if calibrated, ∪ = over-confident, ∩ = under-confident.
- **ELPD** — expected log pointwise predictive density; the predictive-fit target that LOO/WAIC estimate. Higher = better.
- **PSIS-LOO** — Pareto-smoothed importance-sampling leave-one-out CV; the recommended ELPD estimator.
- **Pareto k̂** — per-point reliability/influence diagnostic for PSIS-LOO; < 0.7 trustworthy. Modern ArviZ/PSIS (Vehtari et al. 2024) also exposes a sample-size-dependent `good_k = min(1 − 1/log10(S), 0.7)` (S = #draws) on the `ELPDData` object; prefer it when reported.
- **WAIC** — Widely Applicable Information Criterion; an alternative ELPD estimator, asymptotically equal to LOO but less robust.
- **`p_loo` / `p_waic`** — effective number of parameters; a misspecification diagnostic.
- **Stacking** — averaging predictive distributions with weights that maximize the LOO log score (M-open optimal).
- **Pseudo-BMA+** — Akaike-style LOO weights with a Bayesian-bootstrap SE correction; cheaper than stacking.
- **Bayes factor** — ratio of marginal likelihoods; powerful for genuine hypothesis tests but fragile to priors (Jeffreys–Lindley).
- **SBC (simulation-based calibration)** — validates the *inference algorithm* by checking that posterior ranks of prior draws are uniform.

**Models**

- **GLM** — generalized linear model; linear predictor + link function + non-Gaussian likelihood.
- **Hierarchical / multilevel model** — shared hyperpriors across groups producing **partial pooling** and **shrinkage**.
- **Partial pooling / shrinkage** — group estimates pulled toward the global mean by an amount the data decides.
- **GP (Gaussian process)** — a prior over functions defined by a mean and covariance (kernel); `pm.gp.*`.
- **HSGP** — Hilbert-Space approximate GP; a fast basis-function approximation for scale.
- **BART** — Bayesian Additive Regression Trees; a nonparametric sum-of-trees regressor (`pymc_bart`).
- **Mixture model** — a weighted sum of component densities; needs an ordered constraint to avoid label switching.
- **State-space / structural time series** — latent-state models (random walk, AR, trend+seasonality, stochastic volatility).
- **MCAR / MAR / MNAR** — missing completely at random / at random / not at random; determines whether you must model the missingness mechanism.

---

## 6. Master resource list

The consolidated library, organized by topic so you can go deeper on anything in this course. These are the sources the chapters were built from — every link is real. Start with the **books** for breadth, reach for the **papers** when you need the precise statement, and live in the **docs/blogs** when you're actually coding.

### Books (with the chapters that matter most)

- **McElreath, *Statistical Rethinking*, 2nd ed. (2020)** — the best intuition-first introduction in the field; golems, the typical set, generative thinking. Free lecture videos on YouTube; PyMC port at **pymc-devs/pymc-resources**. Ch 2–4 (priors, first models), Ch 7 "Ulysses' Compass" (overfitting, WAIC/PSIS), Ch 13–14 (multilevel).
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python* (CRC, 2021)** — the PyMC/ArviZ companion to *this* course; free online at **https://bayesiancomputationbook.com**. Ch 2 "Exploratory Analysis of Bayesian Models" is the canonical ArviZ walkthrough (summary, trace/rank, divergences, LOO, PPC, LOO-PIT).
- **Martin, *Bayesian Analysis with Python*, 3rd ed. (2024)** — clean, idiomatic, example-driven PyMC.
- **Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, *Bayesian Data Analysis* (BDA3, 2013)** — the reference. Free PDF at **http://www.stat.columbia.edu/~gelman/book/**. Ch 2–5 (conjugacy, noninformative & weakly-informative priors, hierarchical variance priors); pp.151–153 for Bayesian p-values.
- **Gelman, Hill & Vehtari, *Regression and Other Stories* (2020)** — free PDF; the applied-regression on-ramp. Plus Gelman & Hill (2007) for multilevel models.
- **Kruschke, *Doing Bayesian Data Analysis*, 2nd ed.** — the "puppy book"; gentle, thorough, JAGS/Stan-flavored.

### Priors (Chapters 02–03, 10, 14)

- **Stan Prior Choice Recommendations wiki** — https://github.com/stan-dev/stan/wiki/prior-choice-recommendations — the single most concrete, opinionated source; the 5-level hierarchy and scale-parameter advice.
- **Gelman (2006), "Prior distributions for variance parameters in hierarchical models,"** *Bayesian Analysis* 1(3):515–534 — https://sites.stat.columbia.edu/gelman/research/published/taumain.pdf — *the* citation for why `InvGamma(ε,ε)` is bad and half-Cauchy is the default.
- **Simpson, Rue, Riebler, Martins, Sørbye (2017), "Penalising model component complexity" (PC priors),"** *Statistical Science* 32(1):1–28 — https://arxiv.org/abs/1403.4630.
- **Piironen & Vehtari (2017), regularized horseshoe,** *EJS* 11(2):5018–5051 — https://arxiv.org/abs/1707.01694 — derivation of $\tau_0$ and the finite slab.
- **Austin Rochford, "The Hierarchical Regularized Horseshoe Prior in PyMC"** — https://austinrochford.com/posts/2021-05-29-horseshoe-pymc3.html — the best worked PyMC horseshoe template (porting to v5 is mechanical).
- **Kallioinen, Paananen, Bürkner, Vehtari (2023), power-scaling prior/likelihood sensitivity** — https://arxiv.org/abs/2107.14054 + **priorsense** https://n-kall.github.io/priorsense/ — the modern sensitivity method (R-only today; do manual re-fits in Python).
- **Bayes Rules! Ch 5 "Conjugate Families"** — https://www.bayesrulesbook.com/chapter-5 — clean conjugate update formulas.
- **John D. Cook, "A Compendium of Conjugate Priors"** — https://www.johndcook.com/CompendiumOfConjugatePriors.pdf — exhaustive conjugate-pair table.

### MCMC, HMC & NUTS (Chapters 05, 17)

- **Betancourt, "A Conceptual Introduction to Hamiltonian Monte Carlo,"** arXiv:1701.02434 — https://arxiv.org/abs/1701.02434 — *the* source for the typical set, why random-walk fails, and HMC geometry. Read §2–5.
- **Neal, "MCMC using Hamiltonian dynamics,"** *Handbook of MCMC* (2011), arXiv:1206.1901 — https://arxiv.org/pdf/1206.1901 — definitive leapfrog + correctness derivation.
- **Hoffman & Gelman, "The No-U-Turn Sampler,"** *JMLR* 15 (2014) — https://jmlr.org/papers/v15/hoffman14a.html — NUTS + dual averaging; Algorithm 6 is the canonical pseudocode.
- **Betancourt, "Markov Chain Monte Carlo in Practice"** — https://betanalpha.github.io/assets/case_studies/markov_chain_monte_carlo.html — typical set, concentration of measure, in plain language.
- **Stan Reference Manual, HMC/NUTS chapter** — https://mc-stan.org/docs/reference-manual/mcmc.html — independent authoritative description of metric, step size, treedepth.
- **Chi Feng's interactive MCMC demo** — https://chi-feng.github.io/mcmc-demo/ — *watch* random-walk vs HMC vs NUTS on the funnel and banana. Worth ten pages of prose.

### Diagnostics, PPC & SBC (Chapters 07, 08)

- **Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), "Rank-normalization, folding, and localization: An improved R̂,"** *Bayesian Analysis* 16(2):667–718 — https://projecteuclid.org/journals/bayesian-analysis/volume-16/issue-2/Rank-Normalization-Folding-and-Localization--An-Improved-Rhat-for/10.1214/20-BA1221.full — source of R̂ ≤ 1.01 and bulk/tail-ESS. Online appendix with copy-pasteable formulas: https://avehtari.github.io/rhat_ess/rhat_ess.html.
- **Talts, Betancourt, Simpson, Vehtari, Gelman (2018), "Validating Bayesian Inference Algorithms with Simulation-Based Calibration"** — https://arxiv.org/pdf/1804.06788 — the SBC algorithm and ∪/∩ rank-histogram interpretation.
- **Gabry, Simpson, Vehtari, Betancourt, Gelman (2019), "Visualization in Bayesian workflow,"** *JRSS-A* 182(2):389–402 — https://doi.org/10.1111/rssa.12378 — origin of the energy plot, PPC, and LOO-PIT visual idioms. Code: https://github.com/jgabry/bayes-vis-paper.
- **PyMC gallery, "Diagnosing Biased Inference with Divergences"** — https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html — the exact centered/non-centered Eight Schools code; the single best hands-on diagnostics reference.
- **PyMC gallery, "Sampler Statistics"** — https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/sampler-stats.html — every `sample_stats` field.
- **ArviZ InferenceData schema** — https://python.arviz.org/en/stable/schema/schema.html — authoritative group semantics.
- **`simuk` (PyPI) — SBC for PyMC/Bambi** — https://pypi.org/project/simuk/ — the `SBC(model, num_simulations, sample_kwargs)` API.

### Model comparison (Chapter 11)

- **Vehtari, Gelman, Gabry (2017), "Practical Bayesian model evaluation using LOO-CV and WAIC,"** *Stat. Comput.* 27:1413–1432 — https://link.springer.com/article/10.1007/s11222-016-9696-4 (preprint https://arxiv.org/abs/1507.04544) — THE reference for LOO/WAIC/PSIS.
- **Vehtari, Simpson, Gelman, Yao, Gabry (2024), "Pareto Smoothed Importance Sampling,"** *JMLR* — https://arxiv.org/abs/1507.02646 — defines PSIS and the sample-size-dependent k̂ threshold.
- **Yao, Vehtari, Simpson, Gelman (2018), "Using Stacking to Average Bayesian Predictive Distributions,"** *Bayesian Analysis* 13(3):917–1007 — https://arxiv.org/abs/1704.02030 — stacking vs pseudo-BMA+, the M-open argument.
- **Vehtari, Cross-validation FAQ** — https://users.aalto.fi/~ave/CV-FAQ.html — the practical bible: the "elpd_diff < 4 is small" rule, when to use K-fold, WAIC-vs-LOO guidance.
- **loo R package: Pareto-k diagnostic** — https://mc-stan.org/loo/reference/pareto-k-diagnostic.html — exact threshold language.
- **PyMC "Model comparison" core notebook** — https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html — `idata_kwargs={"log_likelihood": True}`, `az.loo/compare/plot_compare`.
- **Gronau, Singmann, Wagenmakers (2017), "bridgesampling,"** arXiv:1710.08162 — https://arxiv.org/abs/1710.08162 — marginal likelihoods for Bayes factors and why naive estimators fail.

### Workflow & the big picture

- **Gelman et al. (2020), "Bayesian Workflow,"** arXiv:2011.01808 — https://arxiv.org/abs/2011.01808 — the comprehensive map of the whole process.
- **Betancourt, "Towards A Principled Bayesian Workflow"** and the full case-study series — https://betanalpha.github.io/writing/ — the rigorous, geometry-first treatment of every step.

### Docs, libraries & blogs (where you live while coding)

- **PyMC docs + example gallery** — https://www.pymc.io/projects/docs/ and https://www.pymc.io/projects/examples/ — plus the PyMC Discourse forum for real questions.
- **ArviZ docs** — https://python.arviz.org/ — `summary`, `plot_trace`, `plot_rank`, `plot_energy`, `loo`, `compare`, `plot_ppc`, `plot_loo_pit`.
- **Bambi docs** — https://bambinos.github.io/bambi — formula-API GLMs/multilevel; always pair with the raw-PyMC equivalent.
- **Stan User's Guide + Reference Manual** — https://mc-stan.org/docs/ — even if you live in PyMC, Stan's docs are a superb statistics reference.
- **NumPyro** (https://num.pyro.ai), **BlackJAX** (https://blackjax-devs.github.io/blackjax/), **nutpie** (https://github.com/pymc-devs/nutpie) — JAX/Rust NUTS backends for scale (`pm.sample(nuts_sampler="numpyro"|"nutpie"|"blackjax")`).
- **PyMC-BART** — https://www.pymc.io/projects/bart — Bayesian Additive Regression Trees.
- **Blogs:** Thomas Wiecki (https://twiecki.io, "While My MCMC Gently Samples"), Austin Rochford (https://austinrochford.com), the PyMC Labs blog (https://www.pymc-labs.com/blog-posts), and Junpeng Lao's and Ravin Kumar's writing — production-grade Bayesian judgment from the people who build the tools.

---

## 7. What to learn next — a roadmap beyond this course

This course gave you the full applied core: priors, inference, diagnostics, the workflow, the model zoo. But every chapter had an edge where I deliberately stopped. Here is where to walk past those edges, and *why* each direction is worth your time.

- **Deeper simulation-based calibration.** You met SBC (Ch 08) as the fake-data-recovery step. Go further: SBC with the **uniformity ECDF test** and confidence bands (Säilynoja, Bürkner, Vehtari 2022) instead of eyeballing histograms; SBC at scale for *every* parameter of a complex model; and using SBC to debug your *own* custom likelihoods and samplers. This is the discipline that separates "my model fit" from "my inference is *correct*."
- **Bayesian causal inference.** Everything in this course was associational. The next leap is to ask *interventional* and *counterfactual* questions: do-calculus and DAGs (Pearl), the potential-outcomes framework (Rubin), Bayesian treatment of confounding, instrumental variables, and difference-in-differences as Bayesian models. McElreath's later *Statistical Rethinking* chapters are the gentle on-ramp; from there, the causal-inference literature proper. This is arguably the highest-leverage skill on this list.
- **Advanced Gaussian processes.** You learned GP regression and HSGP (Ch 12). Beyond: **deep GPs**, multi-output and coregionalization kernels, GPs for spatial statistics, scalable inducing-point methods, and the GP–neural-network correspondence (infinite-width limits, neural tangent kernels). The GP view of flexible function-fitting connects probabilistic ML to deep learning.
- **Probabilistic-programming internals.** You *used* NUTS; now build it. Implement HMC and a no-U-turn loop from scratch; learn automatic differentiation deeply; study how PyTensor/JAX trace and compile a model graph; read the BlackJAX source for clean, modular MCMC. Understanding the machine makes you fearless when it breaks.
- **Production & MLOps for Bayesian models.** A fitted `InferenceData` is not a deployed system. Learn to serialize and version models and posteriors; cache and reuse warmup; serve fast posterior-predictive inference (JAX-compiled, batched); monitor calibration drift in production; schedule refits; and reason about latency budgets when "uncertainty quantification" must return in 50ms. The scaling chapter (Ch 17) is the doorway; the rest is engineering discipline.
- **Bayesian decision theory.** The posterior is the *means*, not the end. Decision theory closes the loop: define a **loss/utility function**, compute the **expected posterior loss** of each action, and choose the action that minimizes it. This is where uncertainty quantification finally *pays off* — in A/B test stopping rules, portfolio allocation, experimental design (maximizing expected information gain), and any setting where a number must become a decision. BDA3 Ch 9 and Berger's *Statistical Decision Theory* are the canonical starts.

> 💡 **Intuition:** Notice the shape of this roadmap. This course taught you to *build a trustworthy posterior*. The next layer of expertise is everything you *do* with one: validate the inference rigorously (SBC), reason about cause (causal inference), model richer functions (GPs), understand and extend the machinery (internals), ship it (MLOps), and turn it into action (decision theory). You are no longer learning Bayesian *modeling*; you are learning Bayesian *practice*. The modeling was the easy half.

---

## 🧪 Exercises

These are deliberately reference-exercising: each one sends you back into a table to *use* it, which is the only way a cheat-sheet becomes muscle memory.

1. **Parameterization audit (conceptual).** For each, write the correct PyMC v5 call and state what's wrong with the naive version: (a) a residual standard deviation you believe is around 3, written as `pm.Exponential("s", 3)`; (b) a Poisson over-dispersion you want to allow, written with `pm.Poisson`; (c) a varying-slope correlation prior whose `sd_dist` you defined as `sd = pm.Exponential("sd", 1)`. *Hint: §1's footgun column; rate vs mean for Exponential; `.dist()` for `sd_dist`.*

2. **Prior-by-type drill (conceptual).** A logistic GLM on standardized predictors has: an intercept, three slopes, and you suspect one slope perfectly separates a rare class. From §2 and §4, write a sane prior for each parameter and explain in one sentence how the prior on the separating slope prevents the coefficient from running to ±∞.

3. **Threshold triage (coding).** Take any model you fit earlier in the course. Write a single function `report(idata)` that prints, in the §3 order, the divergence count, max R̂, min ess_bulk, min ess_tail, per-chain BFMI, and (if a `log_likelihood` group exists) the max Pareto k̂ — and prints a ✅/❌ next to each against the course thresholds. *Hint: `idata.sample_stats["diverging"].sum()`, `az.summary`, `az.bfmi`, `az.loo(idata, pointwise=True).pareto_k.max()`.*

4. **Funnel reproduction (coding).** Run the §worked-example Eight Schools models. Confirm the centered model produces divergences and the non-centered one does not, then reproduce the funnel pair-plot on a log-`tau` axis. Explain, referencing the troubleshooting table, *why* the fix is "non-center" and not "raise `target_accept` to 0.99."

5. **Comparison sanity (coding).** Fit a Poisson and a Negative-Binomial model to any over-dispersed count dataset (e.g. `bikes`). Run `az.compare` and `az.plot_loo_pit` on both. Use §4's comparison rows to (a) confirm the NegBinomial wins on ELPD and (b) read the Poisson's ∪-shaped LOO-PIT as the *cause*. State the `elpd_diff`/`dse` and whether the difference is meaningful.

6. **Build your own cheat-sheet (synthesis).** Pick the model family you most expect to use in your work (hierarchical GLM, GP, time series, …). On one page, distill from this appendix: its 2–3 key distributions, the prior you'd use for each parameter type, the diagnostics that fail *first* for that family, and the two most likely troubleshooting rows. Tape it next to your monitor. *This is the real deliverable of the whole course — the judgment, compressed.*

---

## 📚 Resources & further reading

This appendix *is* the consolidated resource list — see **§6** above for the full, topic-organized bibliography with live links. The three sources to keep permanently open while you work:

- **Vehtari's Cross-validation FAQ** — https://users.aalto.fi/~ave/CV-FAQ.html — answers almost every model-comparison question you'll ever have.
- **The Stan Prior Choice wiki** — https://github.com/stan-dev/stan/wiki/prior-choice-recommendations — the prior cheat-sheet's source of truth.
- **The PyMC example gallery** — https://www.pymc.io/projects/examples/ — runnable, current, idiomatic code for every model in this course.

And the two books that, between them, contain almost everything: **McElreath's *Statistical Rethinking*** for intuition and **Martin/Kumar/Lao's *Bayesian Modeling and Computation in Python*** (free at https://bayesiancomputationbook.com) for the PyMC/ArviZ practice that mirrors this course.

---

## ➡️ What's next

There is no "next chapter" — you've reached the end of the course. So let me hand you back to the **beginning**, with new eyes. Re-open **Chapter 00 (Overview & Setup)** and **Chapter 08 (The Bayesian Workflow)**, and read them as the *loop* they always were: prior predictive → fit → diagnose → posterior predictive → compare → calibrate → iterate. Everything in this appendix is a station on that loop. The course taught you to walk it once, slowly, with a guide. Now go walk it on a dataset that is *yours* — one where nobody knows the answer, including you. That is where the Bayesian workflow stops being a syllabus and becomes a way of thinking. Keep this appendix open while you do. You've got this.

