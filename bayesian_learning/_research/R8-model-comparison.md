# R8 — Model Comparison & Evaluation (Research Dossier)

> Reference material for the chapter writers. Verified against ArviZ 0.21/0.22, PyMC v5 (5.28),
> and the loo R package docs in June 2026. Where an API could not be 100% confirmed it is flagged.

## Scope & which chapters use this

Primary consumer: **Chapter 11 — Model Comparison & Evaluation**. Secondary touchpoints:
Ch 08 (Bayesian Workflow: "model expansion/comparison" step), Ch 03 (prior sensitivity ↔ why
Bayes factors are fragile), Ch 07 (the k̂ diagnostic complements R̂/ESS in the diagnostic loop),
Ch 18 (capstones report LOO/`az.compare` tables), and Appendix A (diagnostic thresholds table:
k̂ < 0.7). The chapter must teach the **predictive (ELPD) view** of evaluation, PSIS-LOO-CV, WAIC,
the Pareto-k̂ diagnostic, `az.loo`/`az.waic`/`az.compare`/`az.plot_compare`, LOO-PIT calibration,
stacking & pseudo-BMA+ averaging, posterior predictive checks as comparison, and a clear-eyed
section on **why Bayes factors are fragile** (prior sensitivity, Jeffreys–Lindley) and when bridge
sampling is the right tool.

## Verified API & idioms (confirmed)

**PyMC must compute the pointwise log-likelihood before LOO/WAIC.** It is NOT stored by default in
v5. Two confirmed ways:

```python
# (A) at sample time
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                  random_seed=RANDOM_SEED, idata_kwargs={"log_likelihood": True})

# (B) after the fact — inside the model context, or pass model=
with model:
    pm.compute_log_likelihood(idata)   # extends idata in place by default
```

Verified signature (PyMC v5):
```python
pymc.compute_log_likelihood(idata, *, var_names=None, extend_inferencedata=True,
                            model=None, sample_dims=("chain", "draw"),
                            progressbar=True, compile_kwargs=None)
```
This adds a `log_likelihood` group to the InferenceData. ArviZ reads from that group.

**`az.loo` (PSIS-LOO-CV).** Verified signature (ArviZ 0.21):
```python
arviz.loo(data, pointwise=None, var_name=None, reff=None, scale=None)
```
Returns an `ELPDData` (a pandas.Series subclass) with fields: `elpd_loo`, `se`, `p_loo`,
`n_samples`, `n_data_points`, `warning` (bool), and — when `pointwise=True` — `loo_i`,
`pareto_k`, plus `scale` and `good_k`. `scale` ∈ {`"log"` (default, higher=better),
`"negative_log"`, `"deviance"`}. `reff` is the relative MCMC efficiency (auto-computed from ESS if
omitted). Use `pointwise=True` to get per-observation `pareto_k` for diagnostics:
```python
loo = az.loo(idata, pointwise=True)
loo.pareto_k        # xarray DataArray of k̂ per observation
```

**`az.waic`.** Verified signature (ArviZ 0.21):
```python
arviz.waic(data, pointwise=None, var_name=None, scale=None, dask_kwargs=None)
```
Returns `elpd_waic`, `se`, `p_waic`, `n_samples`, `n_data_points`, `warning`, and `waic_i`
(if `pointwise=True`). The `warning` flag trips when a pointwise posterior variance of the
log-likelihood exceeds ~0.4 (a sign WAIC's penalty term is unstable for that point).

**`az.compare`.** Verified signature (ArviZ 0.21):
```python
arviz.compare(compare_dict, ic=None, method="stacking", b_samples=1000,
              alpha=1, seed=None, scale=None, var_name=None)
```
- `compare_dict`: `{"name": idata_or_ELPDData, ...}` — you can pass InferenceData objects OR
  pre-computed `az.loo(...)` results.
- `ic`: `"loo"` (default/recommended) or `"waic"`.
- `method`: `"stacking"` (default), `"BB-pseudo-BMA"` (Bayesian-bootstrap pseudo-BMA = **pseudo-BMA+**),
  or `"pseudo-BMA"` (no bootstrap; discouraged). This argument ONLY changes the `weight` column.
- Returns a DataFrame sorted best→worst. **Column names changed across ArviZ versions** — newer
  ArviZ uses the IC-agnostic spellings `rank, elpd_loo (or "elpd"), p_loo (or "pIC"), elpd_diff,
  weight, se, dse, warning, scale`. Writer note: in recent ArviZ the index/columns are
  `rank, elpd_loo, p_loo, elpd_diff, weight, se, dse, warning, scale`. Confirm exact spellings
  against the installed version (see Uncertainties).

**`az.plot_compare`.** Confirmed parameters:
```python
arviz.plot_compare(comp_df, insample_dev=False, plot_standard_error=True,
                   plot_ic_diff=True, order_by_rank=True, legend=True, title=True,
                   figsize=None, textsize=None, labeller=None, plot_kwargs=None,
                   ax=None, backend=None, backend_kwargs=None, show=None)
```
Empty circle = ELPD with its SE (grey error bar); filled = in-sample deviance if `insample_dev=True`;
triangle + bar = `dse` (SE of the difference vs best). `order_by_rank=True` puts the best model on top.

**Canonical end-to-end call (verified pattern, from PyMC's Model comparison notebook):**
```python
pooled_loo = az.loo(trace_p)
hierarchical_loo = az.loo(trace_h)
df = az.compare({"hierarchical": hierarchical_loo, "pooled": pooled_loo})
az.plot_compare(df)
```

**LOO-PIT calibration.** Verified:
```python
arviz.loo_pit(idata=None, y=None, y_hat=None, log_weights=None)   # returns PIT values in [0,1]
az.plot_loo_pit(idata, y="y")            # KDE of PIT vs Uniform(0,1) + confidence band
az.plot_loo_pit(idata, y="y", ecdf=True) # Δ-ECDF view (preferred, easier to read)
```
LOO-PIT value for point i is `p(ỹ_i ≤ y_i | y_{-i})`; under a well-calibrated model these are
~Uniform(0,1). Requires `posterior_predictive` (or pass `y_hat`) AND `log_likelihood` groups.

## Key concepts & equations the chapter must nail

**ELPD — the target.** Out-of-sample predictive fit is the expected log pointwise predictive
density for a new dataset:
$$\mathrm{elpd} = \sum_{i=1}^{n} \int p_t(\tilde y_i)\, \log p(\tilde y_i \mid y)\, d\tilde y_i$$
where $p(\tilde y_i\mid y)=\int p(\tilde y_i\mid\theta)p(\theta\mid y)d\theta$ is the posterior
predictive and $p_t$ the (unknown) true data process. Everything else (LOO, WAIC, AIC, DIC) is an
estimator of elpd. Higher elpd = better. The **log score** is a *strictly proper scoring rule*, which
is why it's the principled target (motivate this).

**LOO-CV.** The leave-one-out estimate:
$$\mathrm{elpd}_{\mathrm{loo}} = \sum_{i=1}^n \log p(y_i \mid y_{-i}),\qquad
p(y_i\mid y_{-i}) = \int p(y_i\mid\theta)\,p(\theta\mid y_{-i})\,d\theta.$$
Refitting $n$ times is infeasible, so we **reuse the full-data posterior via importance sampling**:
the LOO weight for draw $s$ at point $i$ is $r_i^{(s)} \propto 1/p(y_i\mid\theta^{(s)})$. Raw IS
weights have heavy tails → **PSIS** fits a generalized Pareto distribution to the largest weights and
replaces them with order-statistic-based smoothed values, drastically reducing variance.

**The Pareto k̂ diagnostic.** The estimated GPD shape parameter k̂ at each point measures the tail
heaviness of that point's importance ratios; it equals (roughly) the number of finite moments being
$\lfloor 1/k\rfloor$. Interpretation (state both the classic fixed thresholds AND the new adaptive one):
- Classic: **k̂ < 0.5 good, 0.5–0.7 ok, > 0.7 unreliable, > 1 the LOO estimand has non-finite mean.**
- Modern adaptive (Vehtari et al. 2024, used by ArviZ as `good_k`):
  $$\hat k_{\text{threshold}} = \min\!\Big(1 - \tfrac{1}{\log_{10} S},\; 0.7\Big),$$
  S = number of posterior draws. Below threshold = reliable; between threshold and 0.7 may improve
  with more draws (S > ~2200); 0.7–1 large bias; ≥ 1 PSIS ill-defined.
- k̂ also measures **influence**: high-k̂ points are influential/outliers. The fix when many points
  exceed the threshold is **K-fold CV** (`az.compare` can't do this directly — see datasets/notes)
  or PSIS **moment matching**.

**WAIC.** A fully-Bayesian, pointwise estimator:
$$\widehat{\mathrm{elpd}}_{\mathrm{waic}} = \underbrace{\sum_i \log\Big(\tfrac1S\sum_s p(y_i\mid\theta^{(s)})\Big)}_{\text{lppd}} - \underbrace{\sum_i \mathrm{Var}_s\big[\log p(y_i\mid\theta^{(s)})\big]}_{p_{\text{waic}}}.$$
WAIC is asymptotically equal to LOO, but **LOO is preferred in practice**: it's more robust for finite
samples/weak priors and — crucially — PSIS-LOO comes with the per-point k̂ diagnostic that tells you
*when it has failed*; WAIC's only warning is the var > 0.4 flag and it tends to be over-optimistic.

**p_loo as a diagnostic.** The effective number of parameters $p_{\text{loo}}=\mathrm{lppd}-\mathrm{elpd}_{\text{loo}}$.
Well-behaved: $p_{\text{loo}} < p$ (actual params) and $< n$. If many k̂ > 0.7 AND $p_{\text{loo}}\ll p$
→ likely **model misspecification / outliers** (PPCs will usually flag it too); if $p_{\text{loo}}$ is
close to $p$ with high k̂ → flexible-but-OK model, just needs K-fold.

**Reading `az.compare`.** `elpd_diff` is each model's ELPD minus the best model's ELPD (best = 0).
`dse` is the SE of *that difference* (computed pairwise on pointwise elpd, so it's smaller and more
reliable than treating the two `se` columns as independent). **Rule of thumb (Vehtari CV-FAQ):** if
`|elpd_diff| < 4`, the difference is small/noisy — don't over-interpret. If `elpd_diff > 4`, compare it
to `dse`; a difference of several `dse` (e.g. `elpd_diff/dse` ≳ 2, loosely a z-score) is meaningful.
`weight` is the stacking/pseudo-BMA weight (loosely, posterior model probability) — note weights can be
0 for dominated models even when their `elpd_diff` is modest.

**Stacking & pseudo-BMA+ (Yao et al. 2018).** Don't just pick one model — average predictive
distributions. **Stacking** finds weights $w$ on the simplex maximizing the LOO log score of the mixture
$\sum_k w_k\, p(\tilde y\mid M_k)$ — optimal for prediction in the **M-open** world (true model not in
the set). **Pseudo-BMA+** uses Akaike-style weights from elpd_loo with a Bayesian-bootstrap correction
for the SEs (`method="BB-pseudo-BMA"`); cheaper but generally dominated by stacking. Classic BMA weights
(via marginal likelihood) are **not** recommended here because they assume M-closed and inherit Bayes-
factor prior sensitivity.

**Why Bayes factors are fragile.** $BF_{12}=p(y\mid M_1)/p(y\mid M_2)$ uses the **marginal likelihood**
$p(y\mid M)=\int p(y\mid\theta)p(\theta)d\theta$, which is an average over the *prior*. Consequences:
(1) **Extreme prior sensitivity** — widening a vague prior on a parameter that exists only under $M_1$
shrinks $p(y\mid M_1)$ and pushes the BF toward $M_2$, *regardless of the data*. (2) **Jeffreys–Lindley
paradox** — as the prior under the alternative gets more diffuse (and as $n\to\infty$ with a fixed,
just-significant effect), the BF favors the null arbitrarily strongly; frequentist and Bayesian tests
can flatly disagree. (3) Improper priors make BFs **undefined** (the normalizing constant is arbitrary).
Contrast with LOO/ELPD, which is far less prior-sensitive because it conditions on the data through the
posterior predictive. **When BFs are legitimate:** genuine M-closed hypothesis testing with carefully
elicited, *proper, informative* priors. Then you need the marginal likelihood, computed by **bridge
sampling** (the `bridgestan`/`bridgesampling` lineage) since naive harmonic-mean estimators are
notoriously unstable. PyMC has `pm.sample_smc` (Sequential Monte Carlo) which returns a marginal-
likelihood estimate as a byproduct — mention but note bridge sampling is the gold standard.

**Posterior predictive checks as comparison.** PPCs (`az.plot_ppc`, `pm.sample_posterior_predictive`)
and **LOO-PIT** are the qualitative complement to the numeric ELPD: a model can have higher elpd_loo yet
still fail calibration. LOO-PIT non-uniformity patterns: ∪-shape (PIT mass at 0 and 1) = predictions
**under-dispersed/overconfident**; ∩-shape = over-dispersed; slope/shift = biased mean.

## Common errors → causes → fixes

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError`/`KeyError`: "log_likelihood not found in InferenceData" when calling `az.loo`/`az.waic` | PyMC v5 doesn't store pointwise log-lik by default | `pm.sample(..., idata_kwargs={"log_likelihood": True})` OR `with model: pm.compute_log_likelihood(idata)` |
| log_likelihood missing when likelihood is a `pm.Potential` | `compute_log_likelihood` only covers *observed* RVs, not `Potential` | Express the likelihood as an observed `pm.<Dist>("y", ..., observed=y)`; for custom densities use `pm.CustomDist` with a `logp`, not a bare `Potential` |
| ArviZ warns "Estimated shape parameter of Pareto distribution is > 0.7" | Heavy-tailed LOO importance ratios / influential points | Refit those points exactly with **K-fold CV**, use PSIS **moment matching**, or treat as a robustness flag; report which points |
| `az.compare` raises about mismatched observations / different `n_data_points` | Models fit on different data, transformed `y`, or different obs var | Compare only models with the **same observations on the same scale**; set `var_name` if multiple observed vars |
| Comparing a model on `log(y)` vs `y` and concluding one "wins" | ELPD is on different scales; a Jacobian is missing | Put both likelihoods on the *same* response; add the change-of-variables Jacobian if you transform `y` |
| `weight` is 0 for a model with small `elpd_diff` | Stacking allocates all weight to a near-equivalent better model | Expected behavior; if you want softer weights inspect `method="BB-pseudo-BMA"` |
| WAIC and LOO disagree a lot | WAIC `p_waic` unstable (var>0.4 points), small n, or flexible model | Trust **LOO**; investigate high-k̂ points; consider K-fold |
| Over-trusting a 0.3-elpd "winner" | `elpd_diff < 4`, well within `dse` | Report it as "no meaningful difference"; don't select |
| Bayes factor flips when prior widened | Marginal-likelihood = prior average | Don't use BFs for this; use ELPD/LOO, or fix proper informative priors + bridge sampling |

## Datasets & load snippets

LOO/compare needs ≥ 2 nested-or-rival models on the **same observations**. Suggested examples:

- **Eight Schools** (matches the PyMC notebook — pooled vs hierarchical, the canonical compare):
  ```python
  y = np.array([28, 8, -3, 7, -1, 1, 18, 12]); sigma = np.array([15,10,16,11,9,11,10,18]); J = len(y)
  ```
- **Radon** (Ch 10 flagship): complete-pooling vs no-pooling vs partial-pooling → `az.compare`
  is a perfect demonstration that partial pooling wins. Load via `pm.get_data("radon.csv")` or
  the Gelman/Hill CSV (verify the exact `pm.get_data` filename in the installed PyMC).
- **Howell !Kung height** (Ch 1–3): linear vs polynomial vs spline → overfitting story; k̂ stays low.
- **Roaches / bike-sharing counts** (Ch 9): Poisson vs Negative-Binomial — LOO + LOO-PIT cleanly show
  the NB wins because Poisson is under-dispersed (∪-shaped LOO-PIT).
- **Synthetic with known truth** (encouraged): generate from a known model, fit the true model plus an
  under- and over-fit rival, show LOO recovers ordering and k̂ stays green.

K-fold note: ArviZ does not auto-refit; do K-fold manually (loop folds, refit, accumulate pointwise
`log_likelihood`, then `az.compare`) or use the `loo` R package's `kfold()` for reference.

## Curated resources

- **Vehtari, Gelman, Gabry (2017), "Practical Bayesian model evaluation using LOO-CV and WAIC,"
  *Stat. Comput.* 27:1413–1432** — https://link.springer.com/article/10.1007/s11222-016-9696-4
  (preprint https://arxiv.org/abs/1507.04544). THE reference for LOO/WAIC/PSIS; §2–3 give the elpd,
  LOO and WAIC formulas the chapter must reproduce.
- **Vehtari, Simpson, Gelman, Yao, Gabry (2024), "Pareto Smoothed Importance Sampling," *JMLR*** —
  https://arxiv.org/abs/1507.02646 — defines PSIS and the sample-size-dependent k̂ threshold `good_k`.
- **Yao, Vehtari, Simpson, Gelman (2018), "Using Stacking to Average Bayesian Predictive
  Distributions," *Bayesian Analysis* 13(3):917–1007** — https://doi.org/10.1214/17-BA1091 (preprint
  https://arxiv.org/abs/1704.02030). Source for stacking vs pseudo-BMA+ and the M-open argument.
- **Vehtari, Cross-validation FAQ** — https://users.aalto.fi/~ave/CV-FAQ.html — the practical bible:
  the "elpd_diff < 4 is small, otherwise compare to SE" rule, when SE is unreliable, when to use K-fold,
  WAIC-vs-LOO guidance. Cite specific Q&A entries.
- **PyMC "Model comparison" core notebook** —
  https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html — verbatim
  `idata_kwargs={"log_likelihood": True}`, `pm.compute_log_likelihood`, `az.loo/compare/plot_compare`
  on Eight Schools; the table-interpretation paragraphs (elpd_diff/dse/weight).
- **ArviZ `arviz.loo` docs** — https://python.arviz.org/en/stable/api/generated/arviz.loo.html — signature, ELPDData fields, k̂.
- **ArviZ `arviz.compare` docs** — https://python.arviz.org/en/stable/api/generated/arviz.compare.html — methods (stacking/BB-pseudo-BMA/pseudo-BMA), columns.
- **ArviZ `arviz.plot_compare` docs** — https://python.arviz.org/en/stable/api/generated/arviz.plot_compare.html — plot parameters and how to read the figure.
- **ArviZ LOO-PIT ECDF example** — https://python.arviz.org/en/stable/examples/plot_loo_pit_ecdf.html — `az.plot_loo_pit(..., ecdf=True)` calibration plot.
- **loo R package: Pareto-k diagnostic** — https://mc-stan.org/loo/reference/pareto-k-diagnostic.html — exact threshold language and `k ≥ 1` non-finite-mean caveat.
- **loo R package: glossary** — https://mc-stan.org/loo/reference/loo-glossary.html — definitive p_loo / k̂ interpretation (misspecification vs flexible model).
- **loo vignette: Bayesian stacking & pseudo-BMA weights** — https://mc-stan.org/loo/articles/loo2-weights.html — worked weights, stacking vs pseudo-BMA+.
- **Bayesian Modeling and Computation in Python (Martin, Kumar, Lao), Ch. 2 "Exploratory Analysis of
  Bayesian Models"** — https://bayesiancomputationbook.com/markdown/chp_02.html — the course's
  companion treatment of LOO, `az.compare`, LOO-PIT, and PPCs in idiomatic ArviZ.
- **Statistical Rethinking (McElreath) 2nd ed., Ch. 7 "Ulysses' Compass"** — overfitting, information
  theory, WAIC/PSIS, the regularization-vs-information-criteria framing; PyMC port in
  https://github.com/pymc-devs/pymc-resources.
- **Gronau, Singmann, Wagenmakers, "bridgesampling" (2017), arXiv:1710.08162** —
  https://arxiv.org/abs/1710.08162 — bridge sampling for marginal likelihoods/Bayes factors and why
  naive estimators fail (for the BF-fragility section).

## Uncertainties / things the writer should double-check

- **`az.compare` exact column spellings vary by ArviZ version.** Older docs show `loo, p_loo, d_loo,
  weight, se, dse, warning, loo_scale`; newer 0.21 docs show IC-agnostic `elpd_loo`/`elpd`, `p_loo`/`pIC`,
  `elpd_diff`, `se`/`SE`, `dse`/`dSE`. Print `df.columns` against the installed version before asserting
  names in code/prose. The semantics (best=rank 0, elpd_diff vs best, dse of the difference) are stable.
- **`good_k` field on ELPDData** is present in recent ArviZ but may be absent in older installs; the
  fixed 0.5/0.7 thresholds are always safe to teach alongside the adaptive one.
- **`az.plot_compare` parameter defaults** (`plot_ic_diff`, `insample_dev`) have flipped across versions;
  verify defaults rather than asserting them.
- **`pm.get_data` filenames** (e.g. `"radon.csv"`, `"coal.csv"`) — confirm the exact strings in the
  installed PyMC; the API (`pm.get_data`) is correct.
- **K-fold in ArviZ**: there is no single `az.kfold` convenience; confirm whether the installed ArviZ
  exposes a helper or whether the chapter should show the manual fold loop (recommended to show manually).
- **SMC marginal likelihood**: `pm.sample_smc` returns a log marginal likelihood in `idata.sample_stats`;
  confirm the exact key name in the installed version before showing a Bayes-factor-via-SMC snippet.
