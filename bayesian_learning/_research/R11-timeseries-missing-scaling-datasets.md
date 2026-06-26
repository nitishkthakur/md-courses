# R11 — Time Series + Missing Data + Scaling/Backends + Dataset Index

Research dossier feeding **Ch 15** (time series & state space), **Ch 16** (missing data), **Ch 17**
(scaling & advanced computation), and the **dataset index** in **Ch 00 / Ch 18**.

> Verification status: all PyMC v5, ArviZ, Stan, NumPyro symbols below were checked against
> current official docs (pymc.io, mc-stan.org, pymc-extras) in June 2026. Where I could not
> confirm an exact spelling I say so explicitly. Treat anything marked **VERIFY** as needing a
> quick docs check before it ships in a chapter.

---

## Scope & which chapters use this

- **Cluster A — Time series & state space → Ch 15.** AR(p), Gaussian random walk, structural time
  series (local level/trend + seasonality), stochastic volatility, change-points (coal mining),
  Kalman/state-space view, the `pymc_extras.statespace` module, forecasting via `pm.Data` swap.
- **Cluster B — Missing data → Ch 16.** Rubin's MCAR/MAR/MNAR, imputation-as-parameters, PyMC
  automatic imputation (NaN / masked arrays), marginalizing discrete missingness, modeling the
  missingness mechanism (selection models), censoring (`pm.Censored`) & truncation (`pm.Truncated`),
  vs multiple imputation.
- **Cluster C — Scaling / backends → Ch 17.** `pm.sample(nuts_sampler=...)` for numpyro/nutpie/
  blackjax, GPU via JAX, within-chain parallelism (`reduce_sum`) in Stan, minibatch ADVI
  (`pm.Minibatch` + `total_size`), marginalization for speed, profiling, picking a backend.
- **Cluster D — Dataset catalog → Ch 00 master index & Ch 18 capstones.** Verified load snippets.

---

## Verified API & idioms (confirmed spellings + short snippets)

### A1. Time-series distributions (`pymc.distributions.timeseries`)
Confirmed present in v5: **`AR`**, **`GaussianRandomWalk`**, **`MvGaussianRandomWalk`**,
**`MvStudentTRandomWalk`**, **`EulerMaruyama`**, **`GARCH11`**, and a base **`RandomWalk`**.

**`pm.AR`** — signature `pm.AR(name, rho, *, sigma=1.0, init_dist=None, constant=False, ar_order=None, steps=None, ...)`.
- `rho`: tensor of AR coefficients; last-dim entry *n* is the lag-*n* coefficient. If `constant=True`,
  the **first** element of `rho` is the intercept (so `rho` has length `ar_order + 1`).
- `init_dist`: an **unnamed** distribution from the `.dist()` API for the first `ar_order` values;
  defaults to `pm.Normal.dist(0, 100, shape=...)` (a wide default — usually override it).
- Process: `x_t = ρ₀ + ρ₁ x_{t-1} + … + ρ_p x_{t-p} + ε_t`, `ε_t ~ N(0, σ²)`.

```python
with pm.Model() as ar2:
    rho = pm.Normal("rho", 0.0, 0.5, shape=3)        # constant + 2 lags
    sigma = pm.HalfNormal("sigma", 1.0)
    init = pm.Normal.dist(0.0, 1.0, shape=2)          # init for the 2 lags
    y = pm.AR("y", rho, sigma=sigma, init_dist=init,
              constant=True, ar_order=2, observed=data)
```

**`pm.GaussianRandomWalk`** — `pm.GaussianRandomWalk(name, mu=0, sigma=1, *, init_dist=None, steps=None)`.
- `init_dist` must be an **unnamed univariate `.dist()`**; `steps` needed only if `shape` isn't given.
- Modern v5 idiom (the v3 `pm.GaussianRandomWalk("x", sd=..., shape=N)` form is **removed** — `sd`
  is gone, use `sigma`; `init` keyword renamed to `init_dist` and now takes a `.dist()`).

```python
with pm.Model() as rw:
    sigma = pm.Exponential("sigma", 1.0)
    x = pm.GaussianRandomWalk("x", sigma=sigma,
                              init_dist=pm.Normal.dist(0, 1), steps=len(y) - 1)
```

### A2. Stochastic volatility (the canonical PyMC example)
Source: `pymc.io/projects/examples/.../case_studies/stochastic_volatility.html` (S&P 500 daily
returns). The verified model shape:

```python
with pm.Model(coords={"time": returns.index}) as sv:
    step_size = pm.Exponential("step_size", 10.0)
    volatility = pm.GaussianRandomWalk("volatility", sigma=step_size,
                                       init_dist=pm.Normal.dist(0, 100),
                                       dims="time")
    nu = pm.Exponential("nu", 0.1)
    returns_obs = pm.StudentT("returns", nu=nu, lam=pm.math.exp(-2 * volatility),
                              observed=returns["change"], dims="time")
    idata = pm.sample(nuts_sampler="numpyro", random_seed=RANDOM_SEED)
```
Key teaching points: latent log-volatility is a GRW; returns are heavy-tailed (StudentT) with
time-varying scale `exp(volatility)`; `lam` is precision so `exp(-2·vol)`. Plot `exp(volatility)`
posterior bands over the absolute returns to show volatility clustering (2008 crisis visible).

### A3. Change-point / switchpoint (coal mining disasters)
Data: `disaster_data = pd.read_csv(pm.get_data("coal.csv"), header=None)` (UK 1851–1962; **two NaN
years** → triggers automatic imputation, a nice bridge to Ch 16). Verified pattern:

```python
with pm.Model() as disaster:
    switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())
    early_rate = pm.Exponential("early_rate", 1.0)
    late_rate  = pm.Exponential("late_rate", 1.0)
    rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
    disasters = pm.Poisson("disasters", rate, observed=disaster_data)
    idata = pm.sample()   # uses CompoundStep: NUTS for rates + Metropolis for switchpoint
```
> ⚠️ `switchpoint` is **discrete**, so PyMC assigns it a Metropolis step (not NUTS). This is the
> classic teaching example for "PyMC can mix samplers in one model." Mention that NUTS samplers
> (numpyro/nutpie) cannot handle the discrete RV directly — you'd marginalize it out instead.

### A4. State-space / structural time series — `pymc_extras.statespace`
This module **exists and is current** (formerly in `pymc_experimental`, now `pymc_extras`). It is
the modern, recommended way to do Kalman-filter-based structural TS in PyMC. Two usage tiers:

**(i) Pre-built models** — `from pymc_extras.statespace import BayesianSARIMA, BayesianVARMAX`
(also `structural` components). Confirmed classes: `BayesianSARIMA` / `BayesianSARIMAX`,
`BayesianVARMAX`, `BayesianETS` (**VERIFY ETS name**).

**(ii) Structural components** — `from pymc_extras.statespace import structural as st` (**VERIFY the
exact import alias**; docs path is `pymc_extras.statespace.models.structural`). Confirmed component
classes: **`LevelTrendComponent`**, **`TimeSeasonality`**, **`FrequencySeasonality`**,
**`CycleComponent`**, **`AutoregressiveComponent`**, **`MeasurementError`**, **`RegressionComponent`**.
Components are **added** then **built**, then registered inside a `pm.Model`:

```python
# pseudocode-ish; verify exact arg names in current pymc_extras docs
trend    = st.LevelTrendComponent(order=2, innovations_order=1)
seasonal = st.TimeSeasonality(season_length=12, innovations=True)
error    = st.MeasurementError()
ss_mod   = (trend + seasonal + error).build(name="my_sts")

with pm.Model(coords=ss_mod.coords) as m:
    # user places PRIORS on the named parameters the component requested
    ss_mod.build_statespace_graph(data=y)        # wires Kalman filter into the logp
    idata = pm.sample(nuts_sampler="nutpie")

forecasts = ss_mod.forecast(idata, start=-1, periods=24, filter_output="smoothed")
```
Confirmed core API on the state-space object: **`build_statespace_graph(data=...)`** (call inside a
`pm.Model` context), **`.forecast(idata, start=..., periods=..., filter_output=..., scenario=...)`**,
and **`.sample_conditional_posterior(...)`** for filtered/smoothed states. The base class is
**`PyMCStateSpace`** (subclass it for custom models; implement `make_symbolic_graph`, `param_names`,
etc.). A worked example is the "Forecasting Hurricane Trajectories with State Space Models" gallery
case study.

> 🔧 The component model asks YOU to supply priors for the parameters it declares (it prints the
> required parameter names + dims). This is the main "gotcha" — you don't write the Kalman recursion;
> you only write priors, then call `build_statespace_graph`.

### A5. Forecasting via `pm.Data` swap + posterior predictive
v5 idiom: `pm.Data` is **mutable by default** (the old `pm.MutableData` / `pm.ConstantData` split is
deprecated — just use `pm.Data`). Out-of-sample workflow:

```python
with pm.Model(coords={"obs": idx}) as m:
    x = pm.Data("x", x_train, dims="obs")
    # ... model ...
    idata = pm.sample()

with m:
    pm.set_data({"x": x_future}, coords={"obs": future_idx})
    ppc = pm.sample_posterior_predictive(idata, predictions=True,
                                         extend_inferencedata=False)
```
`predictions=True` stores results in `ppc.predictions` (separate from in-sample `posterior_predictive`).

### B1. Missing data — automatic imputation
**Verified behavior:** in v5 you pass `observed` data containing **`np.nan`** (or an
`np.ma.masked_array`); PyMC detects it, emits an **`ImputationWarning`**, and creates two RVs in the
InferenceData: a `<name>_observed` (the non-missing data) and a `<name>_unobserved` (the imputed
missing values, treated as free parameters sampled from the likelihood). Both NaN and masked-array
forms work; **NaN is the modern, simpler route** (masked arrays no longer required).

```python
import numpy as np
y = np.array([1.2, np.nan, 0.7, np.nan, 2.1])
with pm.Model() as m:
    mu = pm.Normal("mu", 0, 1); sigma = pm.HalfNormal("sigma", 1)
    obs = pm.Normal("obs", mu, sigma, observed=y)   # ImputationWarning; obs_unobserved created
    idata = pm.sample()
```
**Hard constraints to teach:**
- Automatic imputation works **only for the OBSERVED likelihood variable**, NOT for missing
  *predictors/covariates*. For missing predictors you must build an explicit model for the predictor
  (impute-as-parameter): give `x` its own distribution with `observed=x_with_nan`, then use the
  resulting (partially latent) `x` downstream. This is the "joint model of X and Y" pattern.
- **Automatic imputation FAILS (often silently) if the masked/NaN data is wrapped in `pm.Data`.**
  Pass the raw NaN numpy array directly to `observed=`, not `observed=pm.Data(...)`. (Known issue
  pymc#6626.)
- For **discrete** missing data, automatic imputation triggers a Metropolis step and is fragile;
  prefer **marginalizing** the discrete latent out (sum over its support), then recover it with
  `pm.sample_posterior_predictive` if needed.

### B2. Censoring & truncation (relatives of missing data)
**`pm.Censored(name, dist, lower=..., upper=..., observed=...)`** — wraps an **unnamed** base
`.dist()`; values beyond a bound are known to be *at least/at most* the bound (mass piles at the
bound). Use `lower=None` or `upper=None` to disable one side; use `lower=-np.inf`/`upper=np.inf`
likewise. Continuous censored dists are **likelihood-only**.

```python
with pm.Model() as m:
    mu = pm.Normal("mu", 0, 5); sigma = pm.HalfNormal("sigma", 5)
    base = pm.Normal.dist(mu, sigma)
    obs = pm.Censored("obs", base, lower=None, upper=upper_bound, observed=y_censored)
```
**`pm.Truncated(name, dist, lower=..., upper=..., observed=...)`** — for data where values outside
the bounds are **never observed/recorded** (the distribution is renormalized over the interval).
Teaching contrast: **censoring = you see the bound; truncation = you don't see it at all.** Gallery
ref: "Bayesian regression with truncated or censored data."

### C1. Backends via `pm.sample(nuts_sampler=...)`
Confirmed values: `"pymc"` (default, C/Numba), `"nutpie"` (Rust; Numba **or** JAX compile backend),
`"numpyro"` (JAX), `"blackjax"` (JAX).

```python
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                  nuts_sampler="numpyro", random_seed=RANDOM_SEED)

idata = pm.sample(nuts_sampler="nutpie",
                  nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"})
```
- Backend-specific kwargs go through **`nuts_sampler_kwargs={...}`** (e.g. `chain_method`,
  `postprocessing_backend` for numpyro; `backend`/`gradient_backend` for nutpie).
- There are also low-level direct entry points: **`pymc.sampling.jax.sample_numpyro_nuts(...)`** and
  **`pymc.sampling.jax.sample_blackjax_nuts(...)`** (import `import pymc.sampling.jax`).
- **GPU:** only the JAX backends (numpyro/blackjax, and nutpie's JAX mode) use GPU; install
  `jax[cuda12]` per JAX's instructions; set `chain_method="vectorized"` to run chains in parallel on
  one GPU. nutpie's **Numba** backend is CPU-only but often the fastest CPU option.
- **Picking:** nutpie (Numba) → great default CPU speed, fast compile; numpyro → best for GPU and
  large vectorized models; blackjax → research/flexibility; `"pymc"` → most compatible (handles
  discrete RVs, mixtures of step methods that JAX backends can't). JAX/nutpie backends require a
  **fully continuous** model graph (no discrete RVs needing Metropolis).

### C2. Within-chain parallelism in Stan — `reduce_sum`
For Stan models whose log-likelihood is a sum of conditionally independent terms. Compile with
threading and pass `threads_per_chain`:

```python
from cmdstanpy import CmdStanModel
model = CmdStanModel(stan_file="m.stan", cpp_options={"STAN_THREADS": True})
fit = model.sample(data=data, parallel_chains=4, threads_per_chain=4)  # 16 threads total
```
```stan
functions {
  real partial_sum_lpmf(array[] int y_slice, int start, int end, vector mu) {
    return bernoulli_logit_lupmf(y_slice | mu[start:end]);
  }
}
model {
  target += reduce_sum(partial_sum_lupmf, y, grainsize, mu);  // grainsize=1 → auto
}
```
Total threads = `parallel_chains * threads_per_chain`. `grainsize=1` lets the scheduler auto-tune
chunk size. `map_rect` is the older, harder alternative.

### C3. Minibatch ADVI — `pm.Minibatch` + `total_size`
```python
X_mb, y_mb = pm.Minibatch(X_data, y_data, batch_size=128)
with pm.Model() as m:
    # ... priors ...
    obs = pm.Normal("obs", mu, sigma, observed=y_mb, total_size=len(y_data))
    approx = pm.fit(20000, method="advi")   # returns Approximation
    idata = approx.sample(1000)
```
`total_size=N` rescales the minibatch log-likelihood by `N/batch_size` so the ELBO is unbiased.
`pm.Minibatch(*tensors, batch_size=...)` returns minibatched, index-synchronized tensors. `pm.fit`
defaults to ADVI; `method="fullrank_advi"` for correlated posteriors. Use for very large *N* where
full-data MCMC is infeasible; warn that ADVI underestimates variance (cross-ref Ch 06).

### C4. Marginalization for speed
`pymc_extras` provides **`MarginalModel`** (**VERIFY exact current name** — it has moved between
`pymc_experimental`/`pymc_extras`) which can analytically sum out discrete latents (mixture
assignments, missing discretes), giving NUTS-able continuous models that sample far better than
Metropolis-on-discretes. In raw Stan you do this by hand with `log_sum_exp` (cross-ref Ch 13).

---

## Key concepts & equations the chapter must nail

- **AR(1) stationarity:** `|ρ| < 1`; stationary variance `σ²/(1-ρ²)`. GRW is the `ρ=1` boundary
  (nonstationary, variance grows linearly in *t*). Put priors that respect this (e.g. a prior on ρ
  concentrated in (-1,1)).
- **Local level (random walk + noise):** `μ_t = μ_{t-1} + η_t`, `y_t = μ_t + ε_t`. **Local linear
  trend** adds a slope state: `μ_t = μ_{t-1} + δ_{t-1} + η_t`, `δ_t = δ_{t-1} + ζ_t`. These are the
  building blocks of structural TS.
- **State-space / Kalman:** state eq `x_t = T x_{t-1} + R η_t`, obs eq `y_t = Z x_t + ε_t`. The
  Kalman filter computes `p(y)` exactly by marginalizing the Gaussian states — this is *why*
  state-space models sample so much better than writing the latent path explicitly. `pymc_extras`
  does this for you.
- **Stochastic volatility:** `log σ_t` is a latent GRW; `y_t ~ StudentT(ν, 0, exp(σ_t))`. Captures
  volatility clustering that constant-variance models miss.
- **Rubin's missingness taxonomy:** **MCAR** P(miss)⊥everything; **MAR** P(miss) depends only on
  *observed* data — Bayesian likelihood imputation is valid and you can **ignore the missingness
  mechanism**; **MNAR** P(miss) depends on the *unobserved* value itself — you **must** model the
  mechanism jointly (a **selection model**: joint `p(y, R | θ, ψ)` with missingness indicator `R`).
- **Why Bayesian imputation ≈ multiple imputation:** each posterior draw of `<name>_unobserved` is a
  proper multiple imputation; uncertainty propagates automatically (no Rubin's-rules pooling needed).
- **Censoring likelihood:** for right-censoring at `c`, the contribution is `P(Y > c) = 1 - F(c)`
  (survival), not the density — exactly what `pm.Censored` encodes.
- **ELBO / total_size scaling:** minibatch gradient is unbiased iff likelihood term is scaled by
  `N/batch_size`.

---

## Common errors → causes → fixes

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: ... unexpected keyword 'sd'` / `'init'` on GRW | v3 API | Use `sigma=` and `init_dist=pm.Normal.dist(...)` |
| `GaussianRandomWalk` shape mismatch | `steps` vs `shape`/`dims` confusion | Give `steps=len(y)-1` **or** `dims=...`, not both inconsistently |
| AR posterior wanders / nonstationary | default `init_dist=Normal(0,100)` too wide; ρ unconstrained | Tighten `init_dist`; prior on ρ in (-1,1) |
| Discrete switchpoint won't sample with numpyro/nutpie | JAX backends can't do Metropolis/discrete RVs | Use default `nuts_sampler="pymc"`, or marginalize the discrete out |
| Imputation silently not happening | NaN data wrapped in `pm.Data` (issue #6626) | Pass raw NaN numpy array to `observed=`, not `pm.Data(...)` |
| `ImputationWarning` but you wanted imputed **predictors** | auto-impute only covers the observed likelihood | Model the predictor explicitly: `pm.Normal("x", ..., observed=x_nan)` |
| Slow / Metropolis on imputed discrete missing | discrete latent imputed by sampler | Marginalize the discrete latent (sum over support) |
| `pm.Censored` "can't sample" | tried to use it as a prior / non-likelihood | Censored/Truncated dists are likelihood-only (`observed=`) |
| numpyro/blackjax import error | JAX backend not installed | `pip install numpyro` (or `blackjax`, `nutpie`); for GPU install `jax[cuda12]` |
| nutpie JAX mode fails | missing JAX deps | `nuts_sampler_kwargs={"backend":"jax","gradient_backend":"jax"}` needs jax installed |
| Minibatch ELBO biased / wrong scale | forgot `total_size` | add `total_size=N` to the observed node |
| `reduce_sum` gives no speedup | model compiled without threading | `cpp_options={"STAN_THREADS": True}` + `threads_per_chain>1` |
| Forecast shapes wrong after `set_data` | coords resized (issue #7064) | resize coords in `set_data(..., coords=...)`; use `predictions=True` |

---

## Datasets & load snippets (Cluster D — the consolidated catalog)

```python
import seaborn as sns
titanic  = sns.load_dataset("titanic")    # logistic + genuine missing (age) → Ch 9, 16
penguins = sns.load_dataset("penguins")   # GLM, robust regression, missing → Ch 9, 16
tips     = sns.load_dataset("tips")       # GLMs, interactions → Ch 9
geyser   = sns.load_dataset("geyser")     # Old Faithful; mixtures/density → Ch 13
diamonds = sns.load_dataset("diamonds")   # GLMs, log models → Ch 9
flights  = sns.load_dataset("flights")    # monthly airline passengers 1949-60; trend+seasonality → Ch 15
```
> `sns.load_dataset("flights")` IS the classic AirPassengers series (long format: year/month/
> passengers) — pivot it for a time index. Verified available names include: titanic, penguins,
> tips, geyser, diamonds, flights, fmri, mpg, iris, planets, car_crashes, dowjones, seaice.

```python
import pymc as pm, pandas as pd
# Coal mining disasters (built-in; has 2 NaN years → triggers auto-imputation)
coal = pd.read_csv(pm.get_data("coal.csv"), header=None)
# Radon (built-in raw files; filter to Minnesota "MN")
srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))
cty   = pd.read_csv(pm.get_data("cty.dat"))
```

```python
# Statistical Rethinking raw CSVs (semicolon-separated!) from rmcelreath/rethinking
base = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/"
howell      = pd.read_csv(base + "Howell1.csv",     sep=";")  # height/weight → Ch 1-3, 9
rugged      = pd.read_csv(base + "rugged.csv",      sep=";")  # terrain/GDP interactions → Ch 9
chimpanzees = pd.read_csv(base + "chimpanzees.csv", sep=";")  # hierarchical logistic → Ch 10
```

```python
# Mauna Loa CO2 (GPs / structural TS) → Ch 12, 15
import statsmodels.api as sm
co2 = sm.datasets.co2.load_pandas().data   # weekly CO2, 1958-2001; .resample("MS").mean() to monthly
```

Eight Schools (hardcode): `y=[28,8,-3,7,-1,1,18,12]`, `sigma=[15,10,16,11,9,11,10,18]`.

> When a URL is uncertain, prefer seaborn built-ins, `pm.get_data`, the rethinking raw CSVs, or
> **synthetic data with a known generative process** (encouraged for showing parameter recovery —
> especially good for AR/GRW where you can simulate then re-estimate ρ, σ).

---

## Curated resources (each with a real URL + why + pointer)

- **Stochastic Volatility model** — https://www.pymc.io/projects/examples/en/stable/case_studies/stochastic_volatility.html
  — the canonical SV example (S&P 500, GRW log-vol, StudentT returns); copy its model verbatim for Ch 15.
- **PyMC Timeseries distributions API** — https://www.pymc.io/projects/docs/en/stable/api/distributions/timeseries.html
  — authoritative list/signatures for AR, GaussianRandomWalk, EulerMaruyama, GARCH11; verify args here.
- **pymc.AR API** — https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.AR.html
  — exact `rho/constant/ar_order/init_dist/steps` semantics + the AR(2) example.
- **pymc_extras structural components** — https://www.pymc.io/projects/extras/en/latest/statespace/models/structural.html
  — LevelTrend/TimeSeasonality/FrequencySeasonality/Cycle/Autoregressive/MeasurementError classes for Ch 15 structural TS.
- **PyMCStateSpace base class** — https://www.pymc.io/projects/extras/en/latest/statespace/generated/pymc_extras.statespace.core.PyMCStateSpace.html
  — `build_statespace_graph`, `.forecast`, `.sample_conditional_posterior` — the state-space workflow.
- **Forecasting Hurricane Trajectories (state space case study)** — https://www.pymc.io/projects/examples/en/latest/case_studies/ssm_hurricane_tracking.html
  — full worked `pymc_extras.statespace` forecast pipeline incl. `scenario=` exogenous forecasting.
- **Structural Timeseries Modeling notebook** — https://github.com/pymc-devs/pymc-extras/blob/main/notebooks/Structural%20Timeseries%20Modeling.ipynb
  — end-to-end local-level-trend + seasonality build; the reference for combining components.
- **Automatic Missing Data Imputation with PyMC (Strong Inference, Fonnesbeck)** — http://stronginference.com/missing-data-imputation.html
  — explains masked-array / NaN auto-imputation mechanics and the `_unobserved` node (note: TLS cert
  warning on the https variant; cite the canonical http page).
- **Missing Data (ISYE 6420, BUGS→PyMC, Aaron Reding)** — https://areding.github.io/6420-pymc/unit6/Unit6-missingdata.html
  — MAR/MCAR worked PyMC v5 imputation examples; also the coal/switchpoint port at /unit10/Unit10-disasters.html.
- **Bayesian regression with truncated or censored data** — https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-truncated-censored-regression.html
  — definitive `pm.Censored` vs `pm.Truncated` regression example for Ch 16.
- **Censored Data Models (survival)** — https://www.pymc.io/projects/examples/en/latest/survival_analysis/censored_data.html
  — second censoring example tying to survival/likelihood-of-bound intuition.
- **Other NUTS Samplers (pymc-marketing)** — https://www.pymc-marketing.io/en/stable/notebooks/general/other_nuts_samplers.html
  — side-by-side numpyro/nutpie/blackjax usage + install notes for Ch 17.
- **Faster Sampling with JAX and Numba** — https://www.pymc.io/projects/examples/en/latest/samplers/fast_sampling_with_jax_and_numba.html
  — official backend benchmark/how-to; the page to cite for "how to pick a backend."
- **GLM: Mini-batch ADVI on hierarchical regression** — https://www.pymc.io/projects/examples/en/latest/variational_inference/GLM-hierarchical-advi-minibatch.html
  — the canonical `pm.Minibatch` + `total_size` example for Ch 17.
- **Stan reduce_sum tutorial** — https://mc-stan.org/learn-stan/case-studies/reduce_sum_tutorial.html
  — minimal `partial_sum` + `reduce_sum` walkthrough for the within-chain parallelism section.
- **CmdStan Parallelization guide** — https://mc-stan.org/docs/cmdstan-guide/parallelization.html
  — `STAN_THREADS`, `threads_per_chain`, total-thread math.
- **A Primer on Bayesian Methods for Multilevel Modeling (radon)** — https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html
  — definitive radon load (`pm.get_data("srrs2.dat")`/`cty.dat`) and processing.
- **statsmodels CO2 dataset** — https://www.statsmodels.org/dev/datasets/generated/co2.html
  — `sm.datasets.co2.load_pandas()` for Mauna Loa structural-TS/GP example.
- **PyMC sample API** — https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html
  — confirm `nuts_sampler`, `nuts_sampler_kwargs`, `compile_kwargs` argument names.

---

## Uncertainties / things the writer should double-check

1. **`pymc_extras.statespace` import aliases & exact class names.** Docs show `LevelTrendComponent`,
   `TimeSeasonality`, `FrequencySeasonality`, `CycleComponent`, `AutoregressiveComponent`,
   `MeasurementError`, `RegressionComponent`, but a summary view abbreviated some (e.g. "LevelTrend",
   "Cycle"). Confirm the precise class names and whether the convenience import is
   `from pymc_extras.statespace import structural as st` before pasting code. The `.build(name=...)`
   vs `.build()` signature and whether components combine with `+` then `.build()` should be verified
   against the current Structural Timeseries notebook.
2. **`MarginalModel` location/name** (`pymc_extras` vs `pymc_experimental`) — it has moved; verify the
   current import path before using in Ch 16/17.
3. **`BayesianETS`** existence — I confirmed `BayesianSARIMA(X)` and `BayesianVARMAX`; an ETS class
   may or may not exist yet. Verify.
4. **`pm.Data` mutability default** — v5 unified `MutableData`/`ConstantData` into `pm.Data` (mutable);
   confirm the installed version doesn't still warn or require `pm.MutableData` for `set_data`.
5. **AR `constant` ordering** — confirm the intercept is the *first* element of `rho` (docs example
   uses `constant=True` with a length-3 `rho` for AR(2)); state it explicitly in the chapter.
6. **GRW `init_dist` default scale** is 100 (very wide) in some versions — check it hasn't changed,
   and always override it in teaching examples.
7. **Coal `pm.get_data` filename** — confirm it's `"coal.csv"` (some older examples used
   `"disasters.txt"`/a hardcoded array). The ISYE port and gallery both use the built-in.
8. **GPU specifics** (`chain_method="vectorized"`, `jax[cuda12]`) evolve fast — verify against current
   JAX install docs at sample-writing time.
