# R5 — Diagnostics, PPC, SBC, ArviZ (research dossier)

> Reference material for the chapter writers. Verified against ArviZ ~0.18–0.22 (legacy
> `arviz` package, `import arviz as az`) and PyMC v5. Course stack is FROZEN on the legacy
> `az.*` API that PyMC v5 returns and uses. There is a NEW next-gen ecosystem
> (`arviz-plots` / `arviz-stats`, v1.x, with a `visuals=` dict API) — flagged below as a
> migration hazard but NOT the API to teach. Where I could not 100% confirm a symbol I say so.

## Scope & which chapters use this

This dossier feeds three chapters:

- **Ch 07 — MCMC diagnostics & debugging.** The core consumer. R̂ (rank-normalized
  split), ESS bulk/tail, MCSE, trace/rank/forest plots, divergences (geometry + plots),
  energy/BFMI, target_accept tuning, Neal's funnel, centered ↔ non-centered, the ArviZ
  diagnostic loop.
- **Ch 08 — The Bayesian workflow.** Uses the diagnostic loop as a workflow stage, plus
  posterior predictive checks (PPC) and **simulation-based calibration (SBC)** as the
  algorithm-validation / fake-data-recovery step.
- **Ch 11 — Model comparison & evaluation.** Uses PPC, Bayesian p-values, and **LOO-PIT**
  as calibration checks alongside ELPD/LOO (LOO/PSIS machinery itself is R8's job; here we
  only need `az.loo_pit` / `az.plot_loo_pit` / `az.plot_bpv`).

## Verified API & idioms (confirmed against current docs)

**`az.summary`** — confirmed signature (ArviZ 0.21):
```python
az.summary(data, var_names=None, filter_vars=None, group=None, fmt='wide',
           kind='all', round_to=None, circ_var_names=None, stat_focus='mean',
           stat_funcs=None, extend=True, hdi_prob=None, skipna=False,
           labeller=None, coords=None, index_origin=None, order=None)
```
Default columns with `stat_focus='mean'`: **`mean, sd, hdi_3%, hdi_97%, mcse_mean,
mcse_sd, ess_bulk, ess_tail, r_hat`**. With `stat_focus='median'` you instead get
`median, mad, eti_3%, eti_97%, mcse_median, ess_median, ess_tail, r_hat`.
`kind` ∈ `{'all','stats','diagnostics'}`. **`r_hat` is only computed with ≥2 chains.**
`filter_vars` ∈ `{None,'like','regex'}`; exclude vars with a `~` prefix in `var_names`.

**Stand-alone diagnostic functions** (confirmed via az.summary "See Also" and the
`arviz.stats.diagnostics` module): `az.rhat(idata, var_names=..., method='rank')`,
`az.ess(idata, method='bulk'|'tail'|'mean'|'median'|'quantile', prob=...)`,
`az.mcse(idata, method='mean'|'sd'|'quantile', prob=...)`, `az.bfmi(idata)` (returns one
value per chain). `az.rhat` default `method='rank'` is exactly the Vehtari et al. 2021
rank-normalized split-R̂.

**Plots — all confirmed as top-level legacy symbols:**
```python
az.plot_trace(idata, var_names=["mu","tau"])           # left=KDE per chain, right=trace
az.plot_rank(idata, var_names=["mu"], kind="bars")      # kind ∈ {"bars","vlines"}
az.plot_energy(idata)                                   # marginal vs transition energy + BFMI
az.plot_forest(idata, var_names=["theta"], combined=True, r_hat=True, ess=True)
az.plot_posterior(idata, var_names=["mu"], hdi_prob=0.94)
az.plot_pair(idata, var_names=["theta","tau"], divergences=True)   # legacy spelling
az.plot_ppc(idata, kind="kde", num_pp_samples=100)      # kind ∈ {"kde","cumulative","scatter"}
az.plot_loo_pit(idata, y="obs")                         # LOO-PIT KDE vs uniform band
az.plot_bpv(idata, kind="p_value")                      # kind ∈ {"u_value","p_value","t_stat"}
```

**`az.plot_ppc`** confirmed signature (0.21): key args `kind='kde'`, `mean=True`,
`observed=None` (defaults True for posterior group, False for prior), `num_pp_samples`,
`data_pairs={'y':'y_pred'}` (maps observed var → predictive var when names differ),
`group='posterior'` (set `group='prior'` to plot a prior predictive check from the same
function), `flatten`, `flatten_pp`. Needs groups `posterior_predictive` + `observed_data`.

**`az.plot_pair` divergence spelling — VERIFY-AND-WARN.** In the legacy `arviz` package
(what PyMC v5 ships against) it is `divergences=True` (a bool; only honored when
`group` is prior/posterior). In the **new** `arviz-plots` v1.x package it became
`visuals={"divergence": True}`. The PyMC example gallery's current rendering uses the new
`visuals=` form. **Teach `divergences=True`** (matches the frozen stack) but add a one-line
note that newer ArviZ uses `visuals={"divergence": True}`. Do not mix them in one snippet.

**`az.plot_rank`** — `kind="bars"` (default) draws one rank histogram per chain;
`kind="vlines"` draws deviations from uniform as vertical lines. Uniform-looking bars =
good mixing; a chain skewed high/low = different location/scale (a robust visual alternative
to trace plots, which the Vehtari 2021 paper recommends over trace plots).

**Sampler stats PyMC writes into `idata.sample_stats`** (confirmed): `diverging` (bool;
this is the divergence flag — `idata.sample_stats.diverging`), `energy`, `tree_depth`,
`max_energy_error`, `energy_error`, `step_size`, `step_size_bar`, `acceptance_rate`,
`n_steps`, `reached_max_treedepth`, `index_in_trajectory`, `lp`. Count divergences with
`int(idata.sample_stats["diverging"].sum())`. Mean of `acceptance_rate` should sit near
`target_accept`.

**Canonical centered vs non-centered (Eight Schools / Neal's funnel) — confirmed from the
PyMC "Diagnosing Biased Inference with Divergences" gallery notebook:**
```python
import pymc as pm, numpy as np, arviz as az
RANDOM_SEED = 8927
J = 8
y = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
sigma = np.array([15., 10., 16., 11., 9., 11., 10., 18.])

# CENTERED — funnels, diverges at small tau
with pm.Model() as centered:
    mu  = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfCauchy("tau", beta=5)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)   # <-- centered
    obs   = pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
    idata_c = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                        random_seed=RANDOM_SEED)

# NON-CENTERED — the funnel cure
with pm.Model() as noncentered:
    mu  = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfCauchy("tau", beta=5)
    theta_tilde = pm.Normal("theta_t", mu=0, sigma=1, shape=J)      # raw std-normal
    theta = pm.Deterministic("theta", mu + tau * theta_tilde)       # affine recover
    obs   = pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
    idata_nc = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                         random_seed=RANDOM_SEED)
```
Diagnose: `az.plot_pair(idata_c, var_names=["theta","tau"], coords={"theta_dim_0":0},
divergences=True)` — divergences cluster in the funnel **neck** (small `tau`). Plotting
`tau` on a **log scale** makes the funnel visible; the gallery adds
`pm.Deterministic("log_tau", pm.math.log(tau))` and pairs `theta` against `log_tau`.

**Prior/posterior predictive sampling (PyMC v5 idiom):**
```python
with model:
    idata.extend(pm.sample_prior_predictive(samples=500, random_seed=RANDOM_SEED))
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=RANDOM_SEED)
```
`pm.sample_posterior_predictive(..., extend_inferencedata=True)` writes the
`posterior_predictive` group in place. To compute LOO-PIT you also need a `log_likelihood`
group: pass `idarea = pm.sample(..., idata_kwargs={"log_likelihood": True})` (in recent
PyMC v5 log-likelihood is NOT computed by default — **verify on the installed version**;
older v5 computed it automatically). You can also add it post-hoc with
`pm.compute_log_likelihood(idata, model=model)`.

**SBC via `simuk` (confirmed PyPI API)** — the maintained PyMC/Bambi-compatible SBC tool:
```python
from simuk import SBC
sbc = SBC(model,                                 # a PyMC or Bambi model
          num_simulations=1000,                  # ≥1000 recommended
          sample_kwargs={"draws": 25, "tune": 50, "progressbar": False})
sbc.run_simulations()
sbc.plot_results()        # SBC rank histogram / ECDF; want uniform within the envelope
```
(Requires Python ≥3.11.) If `simuk` is unavailable, SBC is short to hand-roll: loop over
prior draws θ̃ ~ p(θ), simulate ỹ ~ p(y|θ̃), refit, and rank θ̃ among the L posterior draws;
collect ranks and histogram them.

## Key concepts & equations the chapter must nail

**Rank-normalized split-R̂ (Vehtari et al. 2021).** Classic Gelman–Rubin R̂ compares
between-chain variance B to within-chain variance W: `R̂ = sqrt( ((N-1)/N)·W + (1/N)·B ) / W`.
It breaks when marginals are heavy-tailed/infinite-variance and is blind to within-chain
non-stationarity. The 2021 fix: (1) **split** each chain in half (catches trends within a
chain), (2) **rank-normalize** the pooled draws (replace each draw by its rank → inverse-
normal transform; kills the infinite-variance problem), then compute split-R̂ on the
transformed values, and (3) report the **max** of R̂ on the draws and R̂ on the
folded/absolute-centered draws ("folded R̂", which senses differences in **scale/tails**,
not just location). **Threshold: R̂ ≤ 1.01** (the paper's recommendation; the old 1.1 is too
loose). This is exactly `az.rhat(..., method='rank')` and the `r_hat` column in `az.summary`.

**Bulk-ESS vs Tail-ESS.** Effective sample size discounts autocorrelation:
`ESS = N / (1 + 2 Σ_t ρ_t)`. **Bulk-ESS** is computed on rank-normalized draws → reliability
of central tendency (mean/median, the bulk of the distribution). **Tail-ESS** is the minimum
of the ESS at the **5% and 95% quantiles** → reliability of interval endpoints / tail
quantities. You need *both*: a model can mix fine in the bulk but terribly in the tails.
**Thresholds:** aim **ESS ≥ 400** total for stable estimates (and the paper's rule of thumb
is **≥ 50 effective draws per chain**, so ≥ 200 with 4 chains as a floor; the course quotes
ESS ≥ 400, ≈100/chain — keep that consistent with the bible).

**MCSE (Monte Carlo standard error).** The sampling error of a posterior **estimate** from
finite draws: `MCSE_mean ≈ sd / sqrt(ESS)`. Report it so the learner sees how many digits of
a posterior mean are trustworthy: if `mean=2.43, mcse_mean=0.05`, the second decimal is
noise. Columns `mcse_mean`, `mcse_sd` in `az.summary`; function `az.mcse`.

**Divergences — the geometry.** NUTS integrates Hamiltonian dynamics with a finite leapfrog
step. Where the posterior has **extreme curvature** (the narrow neck of a funnel), the
fixed step size can't resolve the geometry; the simulated energy error blows up, the
trajectory "diverges," and that point is flagged. Divergences are **truth-tellers**
(Betancourt): they mark regions the sampler systematically *fails to explore*, biasing the
posterior. **Target: 0 divergences.** They cluster spatially — `az.plot_pair(...,
divergences=True)` shows them piled at the funnel neck (small group-scale `tau`).

**Energy plot & BFMI.** NUTS alternates a momentum resample with Hamiltonian flow. The
**energy plot** overlays two densities: the marginal energy distribution `π_E` and the
distribution of **energy transitions** `π_{ΔE}` between successive samples. If the transition
distribution is **narrower** than the marginal, momentum resampling can't move the chain
across the full energy range → slow exploration of heavy tails. **E-BFMI** (energy Bayesian
fraction of missing information) quantifies this:
`E-BFMI = E[ (E_n − E_{n-1})^2 ] / Var[E_n]` (numerator over successive draws, denominator
the marginal energy variance). **Threshold: BFMI > 0.3** is healthy; **< 0.3** flags poor
energy exploration (often heavy tails → reparameterize or change priors). (PyMC/Stan's
`check_energy` historically warned below ~0.2 — note both numbers; the course standard is
0.3.) `az.plot_energy(idata)` shows it; `az.bfmi(idata)` returns per-chain values.

**Centered vs non-centered (the funnel cure).** In a hierarchical model, the centered form
`θ_j ~ Normal(μ, τ)` couples `θ_j` to `τ`: when `τ` is small the `θ_j` are squeezed into a
tiny region (Neal's funnel), curvature explodes, divergences appear. The **non-centered**
form decouples them: sample `θ̃_j ~ Normal(0,1)` on a fixed, well-conditioned geometry and
recover `θ_j = μ + τ·θ̃_j` deterministically. This is the single most important
reparameterization in the course; introduced here (Ch 07) and used as the default cure in
Ch 10. Caveat: non-centered is best in the **weak-data / small-τ** regime; with strongly
informative data per group the **centered** form can actually mix better — mention this.

**Posterior predictive checks (PPC).** Draw `ỹ ~ p(ỹ|y) = ∫ p(ỹ|θ) p(θ|y) dθ` and compare
to observed `y`. `az.plot_ppc` overlays many predictive densities against the observed KDE.
**Bayesian p-value** for a test statistic `T`: `p_B = Pr(T(ỹ) ≥ T(y) | y)`; well-calibrated
≈ 0.5, values near 0 or 1 flag systematic misfit in that feature. Use *informative* `T`
(min, max, sd, # zeros) — location stats are near 0.5 even for bad models. `az.plot_bpv`
(`kind ∈ {"u_value","p_value","t_stat"}`; based on BDA3 pp.151–153).

**LOO-PIT.** PPCs reuse the data twice (double-dipping → over-optimistic). LOO-PIT fixes
this with leave-one-out: `u_i = Pr(ỹ_i ≤ y_i | y_{-i})`, computed cheaply via PSIS rather
than n refits. If the model is calibrated, the `u_i` are **Uniform(0,1)**. `az.plot_loo_pit`
overlays the LOO-PIT KDE/ECDF on the uniform band: a ∩ shape = over-dispersed (too
uncertain), a ∪ shape = under-dispersed (over-confident), tilt = bias. Needs a
`log_likelihood` group in the InferenceData.

**SBC (Talts et al. 2018).** Validates the *inference algorithm itself*, not a single fit.
Key identity: if you draw `θ̃ ~ p(θ)`, then `ỹ ~ p(y|θ̃)`, then `{θ^(1..L)} ~ p(θ|ỹ)`, the
**rank** of `θ̃` among the L posterior draws is **Uniform{0,…,L}** when inference is correct.
Aggregate ranks over many simulations → a **rank histogram**: uniform = calibrated;
**∪-shape** = posterior too narrow (over-confident); **∩-shape** = too wide (under-confident);
**slope/spike at an edge** = biased. This is the rigorous "fake-data recovery" the workflow
(Ch 08) demands before trusting a new/complex model.

**The InferenceData object & standard groups** (confirmed against the ArviZ schema spec):
`posterior` (the draws), `sample_stats` (per-draw diagnostics: `diverging`, `energy`,
`lp`, `tree_depth`, `acceptance_rate`, …), `posterior_predictive`, `prior`,
`prior_predictive`, `observed_data`, `constant_data`, `log_likelihood`. Dims are
`(chain, draw, *var_dims)`. xarray-backed; select with `idata.posterior["mu"].sel(chain=0)`.

**The standard ArviZ diagnostic loop (Ch 07 spine):**
```python
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED)
print(int(idata.sample_stats["diverging"].sum()))   # want 0
az.summary(idata)                                   # scan r_hat<=1.01, ess_bulk/tail>=400
az.plot_trace(idata)                                # fuzzy caterpillars, chains overlap
az.plot_rank(idata)                                 # uniform bars per chain
az.plot_energy(idata)                               # BFMI>0.3, distributions overlap
az.plot_pair(idata, divergences=True)               # where do divergences cluster?
# then PPC:
az.plot_ppc(idata)                                  # predictive vs observed
```

## Common errors → causes → fixes

| Symptom (message / number) | Cause | Fix |
|---|---|---|
| `There were N divergences after tuning` | High curvature (funnel neck); step size too large | Raise `target_accept=0.95–0.99`; **non-center** the hierarchy; tighten/regularize scale priors |
| `r_hat > 1.01` (e.g. 1.3) | Chains not converged / multimodal / not enough tuning | More `tune`/`draws`; better init; reparameterize; check `az.plot_trace`/`plot_rank` for a stuck chain |
| `ess_bulk` or `ess_tail` tiny (e.g. 30) | High autocorrelation / poor geometry | Non-center; raise `target_accept`; longer chains; reparameterize; rescale predictors |
| `BFMI < 0.3` / energy warning | Heavy tails; momentum resampling can't traverse energy range | Reparameterize; lighter-tailed or tighter priors; sometimes change scale-param prior |
| `r_hat = NaN` in summary | Only 1 chain (R̂ needs ≥2), or a constant/deterministic | Sample `chains=4`; ignore for fixed Deterministics |
| Many `Maximum tree depth reached` | Trajectories truncated → inefficient (not bias per se) | Increase `max_treedepth` via `nuts={"max_treedepth": 12}`; usually a geometry smell — reparameterize |
| `az.plot_ppc` raises "no posterior_predictive group" | Didn't run `sample_posterior_predictive` | `pm.sample_posterior_predictive(idata, extend_inferencedata=True)` |
| `az.loo_pit`/`plot_loo_pit` errors on missing log-likelihood | `log_likelihood` group absent | `pm.compute_log_likelihood(idata, model=model)` or sample with `idata_kwargs={"log_likelihood": True}` |
| `plot_pair(..., divergences=True)` raises TypeError | You're on `arviz-plots` v1.x | Use legacy `arviz`, or switch to `visuals={"divergence": True}` |
| PPC looks perfect but model is wrong | Double-dipping; test stat uninformative | Use **LOO-PIT** + informative `T` (min/max/sd) |
| Divergences only at small `tau`, hidden on linear axis | Funnel | Plot `log(tau)`; `pm.Deterministic("log_tau", pm.math.log(tau))` then pair |

## Datasets & load snippets

- **Eight Schools** (canonical funnel/non-centering, Ch 07/08): hardcode
  `y = np.array([28,8,-3,7,-1,1,18,12.]); sigma = np.array([15,10,16,11,9,11,10,18.]); J=8`.
- **Radon (Minnesota)** (hierarchy that funnels in real life, cross-ref Ch 10): available
  via PyMC example data; common load
  `pd.read_csv("https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/radon.csv")`
  (**verify the exact path** in the current pymc-examples repo before relying on it).
- **Synthetic-with-known-truth** is ideal here: simulate from a generative process, fit,
  and show recovery + SBC. Encouraged by the bible for estimator/diagnostic teaching.
- For PPC demos, any GLM dataset (Howell heights, bikes counts) works; the diagnostic
  machinery is dataset-agnostic.

## Curated resources (real URLs, with why + pointer)

- **Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), "Rank-normalization, folding, and
  localization: An improved R̂"**, *Bayesian Analysis* 16(2):667–718 —
  https://projecteuclid.org/journals/bayesian-analysis/volume-16/issue-2/Rank-Normalization-Folding-and-Localization--An-Improved-R%CB%86-for/10.1214/20-BA1221.full
  The source of R̂≤1.01, bulk/tail-ESS, rank plots. Read §3 (R̂), §4 (ESS), §5 (rank plots).
- **Online appendix + code** (same paper) — https://avehtari.github.io/rhat_ess/rhat_ess.html
  Worked formulas and the exact recommended thresholds; copy-pasteable definitions.
- **`az.summary` docs (0.21)** —
  https://python.arviz.org/en/v0.21.0/api/generated/arviz.summary.html
  Confirms columns, `kind`, `stat_focus`, "r_hat needs ≥2 chains".
- **`az.plot_ppc` docs (0.21)** —
  https://python.arviz.org/en/v0.21.0/api/generated/arviz.plot_ppc.html
  Confirms `kind`, `data_pairs`, `num_pp_samples`, required groups.
- **`az.plot_rank` docs** —
  https://python.arviz.org/en/latest/api/generated/arviz.plot_rank.html
  `kind={"bars","vlines"}`; the rank-uniformity interpretation.
- **`az.plot_loo_pit` docs (0.19)** —
  https://python.arviz.org/en/v0.19.0/api/generated/arviz.plot_loo_pit.html
  LOO-PIT ECDF interpretation and args.
- **`az.bfmi` docs** — https://python.arviz.org/en/latest/api/generated/arviz.bfmi.html
  Per-chain BFMI; the 0.3 rule of thumb provenance.
- **PyMC example gallery — "Diagnosing Biased Inference with Divergences"** —
  https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html
  The exact centered/non-centered Eight Schools code, `target_accept`, `sample_stats.diverging`,
  pair-plot of divergences in the funnel. The single best hands-on reference for Ch 07.
- **PyMC example gallery — "Sampler Statistics"** —
  https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/sampler-stats.html
  Lists every `sample_stats` field and how to access them.
- **ArviZ InferenceData schema spec** —
  https://python.arviz.org/en/stable/schema/schema.html
  Authoritative list/semantics of every group (posterior, sample_stats, log_likelihood, …).
- **Talts, Betancourt, Simpson, Vehtari, Gelman (2018), "Validating Bayesian Inference
  Algorithms with Simulation-Based Calibration"** — https://arxiv.org/pdf/1804.06788
  The SBC algorithm + rank-histogram interpretation (Algorithm 1, Figs of ∪/∩ shapes).
- **`simuk` (PyPI) — SBC for PyMC/Bambi** — https://pypi.org/project/simuk/
  The `SBC(model, num_simulations, sample_kwargs).run_simulations().plot_results()` API.
- **Gabry, Simpson, Vehtari, Betancourt, Gelman (2019), "Visualization in Bayesian
  workflow"**, *JRSS-A* 182(2):389–402 — https://doi.org/10.1111/rssa.12378 (code:
  https://github.com/jgabry/bayes-vis-paper ) — origin of the energy plot, PPC and LOO-PIT
  visual idioms used by ArviZ.
- **Betancourt, "Robust Statistical Workflow with PyStan"** —
  https://betanalpha.github.io/assets/case_studies/pystan_workflow.html
  The `check_energy`/`check_rhat`/`check_n_eff`/`check_div` diagnostic functions and the
  geometric story of divergences and BFMI (Stan idioms, same concepts).
- **EABM (ArviZ) — "Prior and Posterior predictive checks"** —
  https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html
  Clean modern treatment of PPC, LOO-PIT, Bayesian p-values with ArviZ code.
- **Bayesian Modeling and Computation in Python — Ch 2 "Exploratory Analysis of Bayesian
  Models"** — https://bayesiancomputationbook.com/notebooks/chp_02.html
  Course-aligned ArviZ walkthrough: `az.summary`, trace/rank, divergences, PPC, `az.plot_bpv`.

## Uncertainties / things the writer should double-check

1. **`az.plot_pair` divergence kwarg.** Legacy `arviz` = `divergences=True`; new
   `arviz-plots` v1.x = `visuals={"divergence": True}`. Confirm which is installed and teach
   `divergences=True` (frozen stack) with a one-line note. The live PyMC gallery now renders
   the `visuals=` form — do not copy it blindly.
2. **log-likelihood default.** Recent PyMC v5 may NOT compute `log_likelihood` during
   `pm.sample` by default. For LOO/LOO-PIT either pass
   `idata_kwargs={"log_likelihood": True}` to `pm.sample` or call
   `pm.compute_log_likelihood(idata, model=model)` afterward. Verify on the pinned version.
2b. **`pm.sample_prior_predictive` arg name.** Older v5 used `samples=`; some versions accept
    `draws=`. Verify `samples` vs `draws` on the installed PyMC.
3. **BFMI threshold number.** Course standard is **0.3**; Stan's historical `check_energy`
   warned below ~0.2. State 0.3 as the course threshold and mention 0.2 as the older Stan
   default so the learner isn't confused by Stan output.
4. **ESS floor.** Bible says ESS ≥ 400 (≈100/chain); Vehtari 2021's stricter rule is ≥50
   *effective* draws per chain. Keep the bible's 400 as the headline; cite the per-chain rule.
5. **`simuk` API exactness.** Confirmed `SBC` class + `run_simulations()` + `plot_results()`
   from PyPI, but the plot method name and whether it returns ECDF vs histogram should be
   re-checked against the installed `simuk` version; the hand-rolled SBC fallback is safe.
6. **Radon CSV URL** in pymc-examples shifts between repo reorganizations — verify the raw
   path (or use `pd.read_csv` from the PyMC example-data mirror) before shipping the snippet.
7. ArviZ is mid-migration to the v1.x `arviz-stats`/`arviz-plots` split (e.g.
   `az.plot_ppc_pit` appears in newest docs). Everything taught here is the **legacy `az.*`**
   API that PyMC v5 returns and that the course froze on — do not import `arviz_plots`.
