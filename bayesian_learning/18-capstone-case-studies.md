# Chapter 18 — Capstone Case Studies

### Three full, honest, end-to-end analyses — the whole workflow, three times, no skipped steps

> *A note from me to you.*
>
> *Every chapter before this one taught you a part. This is the chapter where you watch the parts become a craft. We are going to do three complete Bayesian analyses — a grouped count model, a structural time-series forecast, and a logistic model with real missing data — and I am going to refuse to skip a single step. Not because skipping is hard, but because skipping is exactly the habit that produces confident, wrong analyses. You have seen each tool in isolation: priors in Ch 02–03, NUTS in Ch 05, diagnostics in Ch 07, the workflow spine in Ch 08, GLMs in Ch 09, hierarchy and non-centering in Ch 10, LOO in Ch 11, GPs and HSGP in Ch 12, missing data in Ch 16. What you have not yet seen is all of them deployed together, in order, on a real problem, with the judgment calls narrated out loud.*
>
> *That narration is the point. Anyone can paste a model. The expert move is the running commentary: "I'm standardizing here because otherwise my prior is on the wrong scale," "that divergence is the funnel, so I'm non-centering, not just cranking `target_accept`," "the forecast interval has to widen as I push past the data, and here is why it does." I want you to finish this chapter able to sit down with a dataset you have never seen and run the loop on instinct: frame → data → prior → prior-predictive → model → fit → diagnose → posterior-predictive → compare/expand → interpret/decide. Three times through, and it becomes muscle memory.*
>
> *One promise and one warning. The promise: by the end you will have a template you can lift, almost verbatim, onto your own work. The warning: do not lift the model — lift the **method**. The right model for your data is one you discover by running this loop, not one you copy from a textbook. Let's run it.*

---

## What you'll be able to do after this chapter

- **Execute the full Bayesian workflow end to end**, visibly hitting every stage from Ch 08, on three genuinely different problem types — and recognize which stage you're in at any moment.
- **Build a hierarchical Negative-Binomial count model** with non-centered varying effects, diagnose its funnel, compare it against Poisson and complete-pooling baselines with LOO, and read its posterior predictive checks for overdispersion and zeros.
- **Fit a structural / Gaussian-process time-series model** with HSGP, decompose it into trend + seasonal + noise, and produce a forecast whose intervals **honestly widen** as you extrapolate past the data.
- **Handle missing data inside a logistic regression** — imputing both the outcome's covariates and modeling the missingness sensibly — then check **calibration** with LOO-PIT and turn the posterior into a **decision** with a threshold and a cost.
- **Narrate the judgment calls**: every standardization, every prior, every reparameterization, every model expansion, justified out loud the way you'd defend it to a skeptical colleague.
- **Assemble a reusable analysis template** you can carry to your own data.

A word on how to read this chapter. Each case study is structured as the **same eight-step loop**, with the step name as a sub-heading, so you can see the skeleton repeat. The steps are exactly Ch 08's spine:

```
 1. FRAME the question        (what decision rides on this?)
 2. DATA & preprocessing      (load, clean, standardize, index)
 3. PRIORS + prior predictive (are the priors physically sane?)
 4. MODEL                     (the generative story in PyMC)
 5. FIT + diagnostics         (R̂, ESS, divergences, energy)
 6. POSTERIOR predictive      (does it retrodict the data?)
 7. COMPARE / EXPAND          (LOO, alternatives, did a check demand more?)
 8. INTERPRET & DECIDE        (intervals, contrasts, the actual answer)
```

Watch that skeleton appear three times. The datasets change, the likelihoods change, the failure modes change — but the loop does not. That invariance *is* the expertise.

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

> 🔧 **In practice:** Every model below returns an `az.InferenceData` object (never call it a "trace" — that was the v3 name). Every sample call uses the course-standard `pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED)`, and I raise `target_accept` only when a *diagnostic* tells me to. Predictors get standardized by default (Ch 03) so my priors live on a known scale. None of this is decoration — it is the discipline that makes the three analyses comparable and reproducible.

---
---

# Case Study A — A Hierarchical Negative-Binomial Count Model

### Bicycle counts by hour and weather, pooled across days

> *The question behind this one: "How many bikes will cross this bridge in a given hour, and what drives the variation?" It looks like a regression. It is really a lesson in three things at once: counts are not Gaussian (you need Poisson, then Negative-Binomial), groups share information (you need hierarchy), and the hierarchy will funnel (you need non-centering). We will hit all three, in order, because each one is forced on us by a check that fails.*

## A.1 — FRAME the question

Let me start where you should always start: **what decision rides on this?** Imagine you run a bike-share operator. You want to staff rebalancing crews and predict demand by hour of day, accounting for weather. So the quantities you actually care about are:

1. The **expected count** in each hour-of-day, and how much it swings with temperature and weather.
2. The **predictive spread** — not just the mean, but a calibrated interval, because you staff for the busy tail, not the average.
3. Whether the variation across days is large enough that you should model days as a **group** rather than pretending every day is exchangeable noise.

That third point is the hierarchical hook. We have repeated measurements (hours) nested in days, and days will differ — a rainy Tuesday is not a sunny Saturday. A flat regression that ignores days throws away that structure and, worse, gives you overconfident intervals. So our generative story is: *counts in an hour depend on hour-of-day and weather, with a day-level offset drawn from a population of days.*

> 💡 **Intuition:** Framing is not paperwork. The frame tells you the **likelihood** (counts ⇒ Poisson/NegBin), the **structure** (hours-in-days ⇒ varying intercept by day), and the **quantity to report** (predictive interval, not just a coefficient). Get the frame right and the model almost writes itself; get it wrong and no amount of sampling saves you.

## A.2 — DATA & preprocessing

We'll use the **`bikes`** dataset that ships with Bambi (it's in the shared catalog, used again from Ch 09 and Ch 14). It has hourly counts with temperature, humidity, weather, and hour-of-day.

```python
import bambi as bmb

bikes = bmb.load_data("bikes")     # registered Bambi dataset; hourly counts + weather
print(bikes.shape)
print(bikes.head())
print(bikes.columns.tolist())
# typical columns: count, hour, temperature, humidity, weather (categorical), ...
```

The dataset as shipped does not always carry an explicit day index, so to make the hierarchy concrete and pedagogically clean I'll construct a **`day` grouping** from the row blocks (in your real data this would be a genuine calendar day). The mechanics of indexing groups are identical either way, and the lesson — partial pooling across a grouping factor — is the same.

```python
# --- build a clean modeling frame ---
df = bikes.copy()

# Ensure an integer hour-of-day 0..23 and a 'day' group.
# (In real data, parse a timestamp; here we derive a group to illustrate hierarchy.)
df["hour"] = df["hour"].astype(int)
n_per_day = 24
df["day"] = (np.arange(len(df)) // n_per_day).astype(int)   # contiguous day blocks

# Standardize the continuous predictor(s). Standardization is the DEFAULT (Ch 03):
# it puts the slope prior on a "per standard deviation" scale we can reason about.
df["temp_z"] = (df["temperature"] - df["temperature"].mean()) / df["temperature"].std()

# Weather as a small categorical -> integer codes for a fixed effect.
df["weather"] = df["weather"].astype("category")
weather_levels = df["weather"].cat.categories.tolist()
df["weather_idx"] = df["weather"].cat.codes.values

# Hour-of-day as a cyclic-ish categorical effect (24 levels). We model it as a
# varying intercept too, but to start simple we treat it as a fixed factor.
hours = np.sort(df["hour"].unique())

# Group index for days, built with an explicit lookup to avoid the silent
# "wrong group->row mapping" bug (see Ch 10 errors table).
day_levels = np.sort(df["day"].unique())
day_lookup = {d: i for i, d in enumerate(day_levels)}
df["day_code"] = df["day"].map(day_lookup).astype(int)

y = df["count"].to_numpy()
print("counts: mean=%.2f  var=%.2f  ->  var/mean=%.2f"
      % (y.mean(), y.var(), y.var() / y.mean()))
```

That last print is the **single most important diagnostic before you fit anything to count data.** If `var/mean` is close to 1, a Poisson might do. If it is well above 1 — and for bike counts it always is, often 3–10× — your data are **overdispersed** and a Poisson will be badly overconfident in its tails. Let me make the reasoning explicit.

> 🧮 **The math:** Poisson forces $\mathrm{Var}(y)=\mathrm{E}(y)=\mu$ (equidispersion). The Negative-Binomial relaxes this: in PyMC's `(mu, alpha)` parametrization, $\mathrm{Var}(y)=\mu + \mu^2/\alpha$. **Small $\alpha$ ⇒ heavy overdispersion; $\alpha\to\infty$ ⇒ back to Poisson.** Watch the direction — it is the inverse of the textbook NB2 dispersion $\phi$ where $\mathrm{Var}=\mu+\phi\mu^2$, so $\alpha_{\text{PyMC}} = 1/\phi$. This sign-flip is the single most common interpretation bug in count modeling; I will say it twice so you never make it.

So our plan, built up the way the workflow demands — simplest first, expand only when a check fails:

- **Model 0 (baseline):** complete-pooling Poisson. Expect it to fail the overdispersion check.
- **Model 1:** complete-pooling Negative-Binomial. Fixes dispersion, ignores days.
- **Model 2:** **hierarchical** Negative-Binomial with a varying intercept by day, **non-centered**. The model we actually want.

We'll fit all three and let LOO and the PPC adjudicate.

## A.3 — PRIORS + prior predictive check

Counts enter through a **log link**: $\log \mu_i = \alpha + \beta_{\text{temp}}\, z_i + \gamma_{w[i]} + u_{d[i]}$, where $z_i$ is standardized temperature, $\gamma_w$ is a weather effect, and $u_d$ is the day offset. Because everything lives on the **log scale**, priors that look "wide" can be insane after exponentiation. This is where juniors blow up `mu` to `1e17` and the sampler chokes. Let me set priors deliberately.

```python
# Reason on the log scale. exp(alpha) is the baseline expected count at
# average temperature, baseline weather, average day.
#   alpha ~ Normal(log(mean_count), 1)   -> centered at the empirical log-mean
#   beta  ~ Normal(0, 0.5)               -> a 1-SD temp change scales the rate by
#                                           exp(0.5) ~ 1.6x at the +/-1 sigma edge
#   gamma_w ~ Normal(0, 0.5)             -> weather shifts the log-rate modestly
log_mean = np.log(y.mean())
print("intercept prior centered at log-mean =", round(log_mean, 3))
```

> ⚠️ **Pitfall:** A `Normal(0, 10)` prior on a log-rate slope is *not* weakly informative — it implies the rate can multiply by $e^{10}\approx 22{,}000$ per unit of a standardized predictor. On the count scale that is physically absurd and it will give you `nan` logp and a stuck sampler. On a log link, keep slope priors **tight**: a `Normal(0, 0.5)` already permits a 1.6× rate change per standard deviation, which is a lot. Tightness here is not timidity; it is putting the prior on the scale the link actually uses.

Now the **prior predictive check** — the step almost everyone skips and almost everyone regrets skipping. We draw datasets *from the prior alone*, before touching the likelihood's fit, and ask: are these plausible bike counts?

```python
coords = {
    "obs": np.arange(len(df)),
    "weather": weather_levels,
    "day": day_levels,
}

with pm.Model(coords=coords) as prior_check_model:
    temp_z = pm.Data("temp_z", df["temp_z"].to_numpy(), dims="obs")
    weather_idx = pm.Data("weather_idx", df["weather_idx"].to_numpy(), dims="obs")

    alpha = pm.Normal("alpha", mu=log_mean, sigma=1.0)
    beta_temp = pm.Normal("beta_temp", mu=0.0, sigma=0.5)
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5, dims="weather")
    alpha_nb = pm.Exponential("alpha_nb", 1.0)      # NB dispersion

    log_mu = alpha + beta_temp * temp_z + gamma[weather_idx]
    mu = pm.Deterministic("mu", pm.math.exp(log_mu), dims="obs")
    pm.NegativeBinomial("y", mu=mu, alpha=alpha_nb, observed=y, dims="obs")

    idata_prior = pm.sample_prior_predictive(samples=200, random_seed=RANDOM_SEED)
    # NOTE: arg is `samples=` on this PyMC version; some versions use `draws=`.
    #       # verify in current PyMC docs

# What do prior-implied counts look like?
prior_y = idata_prior.prior_predictive["y"].values.ravel()
print("prior predictive counts:  median=%.0f  99th pct=%.0f  max=%.0f"
      % (np.median(prior_y), np.percentile(prior_y, 99), prior_y.max()))
```

> 🩺 **Diagnostic:** Read the prior predictive like a sanity gate. You want the **bulk** of prior-implied counts to overlap the plausible range of real hourly bike counts (a few up to a few hundred) and the **tail** not to explode to astronomically impossible numbers. If your 99th percentile is in the millions, your priors are too wide — tighten them and redraw. We are not fitting the data here; we are checking that the *generative story* is capable of producing data that looks like the world. If it can't, no posterior will save you. This is McElreath's "the model is a small world" made operational: make sure the small world can contain the large one.

A quick visual you'll run:

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.clip(prior_y, 0, 800), bins=50, density=True, alpha=0.7)
ax.axvline(y.max(), color="k", ls="--", label="observed max count")
ax.set(xlabel="prior-predictive count", ylabel="density",
       title="Case A — prior predictive check (clipped at 800)")
ax.legend()
```

If the histogram piles its mass in the 0–300 region with a tail that reaches but doesn't wildly overshoot the observed max, the priors are doing their job: regularizing without dictating. Good. We proceed.

## A.4 — MODEL

Now the three models. I'll write each in raw PyMC so you see the machinery, and show the Bambi one-liner alongside (bible §10: whenever you show Bambi, show the PyMC it generates).

**Model 0 — complete-pooling Poisson (the baseline we expect to break):**

```python
with pm.Model(coords=coords) as m0_poisson:
    temp_z = pm.Data("temp_z", df["temp_z"].to_numpy(), dims="obs")
    weather_idx = pm.Data("weather_idx", df["weather_idx"].to_numpy(), dims="obs")

    alpha = pm.Normal("alpha", mu=log_mean, sigma=1.0)
    beta_temp = pm.Normal("beta_temp", 0.0, 0.5)
    gamma = pm.Normal("gamma", 0.0, 0.5, dims="weather")

    mu = pm.Deterministic("mu", pm.math.exp(alpha + beta_temp * temp_z + gamma[weather_idx]),
                          dims="obs")
    pm.Poisson("y", mu=mu, observed=y, dims="obs")

    idata_m0 = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                         random_seed=RANDOM_SEED,
                         idata_kwargs={"log_likelihood": True})  # needed for LOO
```

**Model 1 — complete-pooling Negative-Binomial:**

```python
with pm.Model(coords=coords) as m1_nb:
    temp_z = pm.Data("temp_z", df["temp_z"].to_numpy(), dims="obs")
    weather_idx = pm.Data("weather_idx", df["weather_idx"].to_numpy(), dims="obs")

    alpha = pm.Normal("alpha", mu=log_mean, sigma=1.0)
    beta_temp = pm.Normal("beta_temp", 0.0, 0.5)
    gamma = pm.Normal("gamma", 0.0, 0.5, dims="weather")
    alpha_nb = pm.Exponential("alpha_nb", 1.0)      # dispersion; small => more overdispersed

    mu = pm.Deterministic("mu", pm.math.exp(alpha + beta_temp * temp_z + gamma[weather_idx]),
                          dims="obs")
    pm.NegativeBinomial("y", mu=mu, alpha=alpha_nb, observed=y, dims="obs")

    idata_m1 = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                         random_seed=RANDOM_SEED,
                         idata_kwargs={"log_likelihood": True})
```

**Model 2 — hierarchical Negative-Binomial, varying intercept by day, non-centered.** This is the model the frame asked for. The day offsets $u_d$ are drawn from a population $\mathrm{Normal}(0, \sigma_u)$, and — critically — I write them **non-centered** from the start, because a varying-intercept model with a learned group scale is the textbook funnel (Ch 07, Ch 10).

```python
with pm.Model(coords=coords) as m2_hier:
    temp_z = pm.Data("temp_z", df["temp_z"].to_numpy(), dims="obs")
    weather_idx = pm.Data("weather_idx", df["weather_idx"].to_numpy(), dims="obs")
    day_idx = pm.Data("day_idx", df["day_code"].to_numpy(), dims="obs")

    # population-level (fixed) effects
    alpha = pm.Normal("alpha", mu=log_mean, sigma=1.0)
    beta_temp = pm.Normal("beta_temp", 0.0, 0.5)
    gamma = pm.Normal("gamma", 0.0, 0.5, dims="weather")

    # day-level varying intercept, NON-CENTERED (the Ch 10 default cure)
    sigma_u = pm.Exponential("sigma_u", 1.0)              # scale of day-to-day variation
    pm.Deterministic("log_sigma_u", pm.math.log(sigma_u))  # the funnel is visible on log scale
    z_u = pm.Normal("z_u", 0.0, 1.0, dims="day")          # std-normal offsets
    u = pm.Deterministic("u", z_u * sigma_u, dims="day")  # u_d = sigma_u * z_d

    alpha_nb = pm.Exponential("alpha_nb", 1.0)

    log_mu = alpha + beta_temp * temp_z + gamma[weather_idx] + u[day_idx]
    mu = pm.Deterministic("mu", pm.math.exp(log_mu), dims="obs")
    pm.NegativeBinomial("y", mu=mu, alpha=alpha_nb, observed=y, dims="obs")

    idata_m2 = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                         random_seed=RANDOM_SEED,
                         idata_kwargs={"log_likelihood": True})
```

> 💡 **Intuition (why non-centered, said plainly):** In the *centered* form you'd write `u = pm.Normal("u", 0, sigma_u, dims="day")`. That directly couples each `u_d` to `sigma_u`: when `sigma_u` is small the `u_d` are crushed into a narrow neck, the local curvature of the posterior explodes, and NUTS's fixed step size can't navigate it — divergences appear, clustered at small `sigma_u`. The non-centered form `u = z * sigma_u` with `z ~ Normal(0,1)` makes `z` and `sigma_u` *a priori independent*, unrolling the funnel into a roughly isotropic Gaussian the sampler glides through. This is the "Matt trick" (named for Matt Hoffman). It is the single most important reparameterization in the course, and for a sparse-data hierarchy it is the default, not an afterthought.

And the Bambi equivalent of Model 2, for when you want the one-liner:

```python
# Bambi version of the hierarchical NegBin. Bambi non-centers by default.
m2_bambi = bmb.Model(
    "count ~ temp_z + weather + (1|day)",
    data=df, family="negativebinomial",
)
idata_m2_bambi = m2_bambi.fit(
    draws=1000, tune=1000, chains=4, target_accept=0.9,
    random_seed=RANDOM_SEED,
    idata_kwargs={"log_likelihood": True},   # so az.loo/az.compare work
)
# `(1|day)` IS the varying intercept; family="negativebinomial" IS the NB likelihood.
# Bambi's noncentered=True default gives you the same reparameterization we wrote by hand.
# Bambi's .fit forwards sampler kwargs (here target_accept) through to pm.sample; this
# pass-through has been version-sensitive, so verify it in your pinned Bambi version.
# The Case-C Bambi call below passes the SAME kwargs, deliberately, for consistency.
```

> 🔧 **In practice:** I always write the raw-PyMC model *and* the Bambi one. The Bambi line is what I'd ship; the PyMC block is what I reach for the moment I need something Bambi can't express (a custom non-linearity, a bespoke missingness model, a weird likelihood). Knowing both — and knowing they produce the *same* model — is what lets you move fast without losing control.

## A.5 — FIT + diagnostics

Sampling returns three `InferenceData` objects. Now we **diagnose before we interpret** — non-negotiable. The order is fixed: divergences, then $\hat R$ and ESS, then the energy/funnel visuals.

```python
def quick_diag(idata, name):
    n_div = int(idata.sample_stats["diverging"].sum())
    s = az.summary(idata, var_names=["alpha", "beta_temp", "gamma"])
    print(f"\n=== {name} ===")
    print("divergences:", n_div)
    print("max r_hat:  ", round(float(s["r_hat"].max()), 4))
    print("min ess_bulk:", int(s["ess_bulk"].min()),
          " min ess_tail:", int(s["ess_tail"].min()))

quick_diag(idata_m0, "M0 Poisson")
quick_diag(idata_m1, "M1 NegBin")
quick_diag(idata_m2, "M2 Hierarchical NegBin")
```

> 🩺 **Diagnostic (the thresholds, reused from Ch 07):** You want **0 divergences**, **$\hat R \le 1.01$**, and both **ESS_bulk and ESS_tail $\gtrsim 400$** (≈100 per chain across 4 chains). If `M2` shows a handful of divergences, that is the funnel reminding you it exists. Our first response is *not* to crank `target_accept` blindly — it's to confirm we already non-centered (we did) and then, only if a few stragglers remain, raise `target_accept` to 0.95–0.99 and lengthen tuning. If we had written the *centered* form, we'd expect dozens of divergences piled at small `sigma_u`, and the cure would be reparameterization, not knob-twiddling.

If `M2` still shows a few divergences after non-centering, here is the escalation, exactly as Ch 10's error table prescribes:

```python
with m2_hier:
    idata_m2 = pm.sample(1000, tune=2000, chains=4, target_accept=0.97,
                         random_seed=RANDOM_SEED,
                         idata_kwargs={"log_likelihood": True})
print("divergences after target_accept=0.97:",
      int(idata_m2.sample_stats["diverging"].sum()))
```

The visual confirmations you'll run on the hierarchical model:

```python
az.plot_trace(idata_m2, var_names=["alpha", "beta_temp", "sigma_u", "alpha_nb"])
az.plot_energy(idata_m2)                      # BFMI > 0.3, the two energy curves overlap
# The funnel check: pair log(sigma_u) against a RECONSTRUCTED day effect u (the Deterministic),
# colored by divergences. Crucially we pair against `u`, NOT `z_u`: in the non-centered
# parameterization z_u is a priori independent of sigma_u, so (sigma_u, z_u) is a clean blob
# *by construction* and can never show the funnel. The funnel lives in (log_sigma_u, u),
# the reconstructed effect — and only on the LOG scale of sigma_u (Ch 07).
az.plot_pair(idata_m2, var_names=["log_sigma_u", "u"],
             coords={"day": [day_levels[0]]}, divergences=True)
# newer ArviZ: divergences -> visuals={"divergence": True}
```

> 🩺 **Diagnostic:** `az.plot_trace` should show four fuzzy, overlapping "caterpillars" per parameter — no chain wandering off, no flat stretches. `az.plot_energy` overlays the marginal energy and the energy-transition distributions; if the transition curve is much narrower than the marginal, the chain can't traverse the energy range (BFMI < 0.3) and you should suspect heavy tails or a residual funnel. The `plot_pair` of `log_sigma_u` vs the reconstructed day effect `u` is the **money plot** for hierarchy: in a healthy non-centered fit the divergence points (if any) are scattered, not piled into a neck at small `sigma_u`. Seeing them funnel into that neck — the characteristic narrowing of `u`'s spread as `log_sigma_u` shrinks — is the unmistakable signature that you forgot to non-center something. (Pairing `log_sigma_u` against `z_u` instead would *always* look like a clean blob, because the non-centered offsets are independent of the scale by construction; you must use the reconstructed `u` to see the funnel.)

## A.6 — POSTERIOR predictive check

Diagnostics tell us the sampler explored the posterior faithfully. They say **nothing** about whether the model fits. For that we draw replicated datasets $\tilde y \sim p(\tilde y \mid y)$ and compare to the real counts — with test statistics chosen to expose the failure mode we care about: **overdispersion and zeros**.

```python
for idata, model, name in [(idata_m0, m0_poisson, "M0 Poisson"),
                           (idata_m1, m1_nb, "M1 NegBin"),
                           (idata_m2, m2_hier, "M2 Hier NegBin")]:
    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                       random_seed=RANDOM_SEED)

az.plot_ppc(idata_m0, num_pp_samples=100)     # Poisson: predictive too narrow in the tail
az.plot_ppc(idata_m2, num_pp_samples=100)     # Hier NegBin: should track the observed KDE
```

The headline visual is `az.plot_ppc`: it overlays ~100 predictive count distributions (thin lines) against the observed distribution (thick line). For the **Poisson** you will see the predictive distributions hug the mean too tightly — the observed data has a fatter right tail than any Poisson replicate, the classic overdispersion signature. For the **hierarchical NegBin** the predictive cloud should envelop the observed curve, tail included.

But the *quantitative* check is a Bayesian p-value on a tail-sensitive statistic. Location statistics (the mean) look fine even for a broken model; you must probe the **variance, the max, and the count of large values**.

```python
def bayes_p_value(idata, stat_fn, name):
    # Assumes posterior_predictive["y"] has dims (chain, draw, obs); we stack the two
    # sampling dims into one and apply stat_fn to each replicate dataset. The loop is
    # O(n_draws) (one stat_fn call per draw) — fine for a few thousand draws, but note
    # it materializes one (n_obs,) array at a time rather than vectorizing.
    obs = stat_fn(idata.observed_data["y"].values)
    pp = idata.posterior_predictive["y"].stack(s=("chain", "draw"))
    rep = np.array([stat_fn(pp.isel(s=i).values) for i in range(pp.sizes["s"])])
    p = float((rep >= obs).mean())
    print(f"{name:18s}  T(y)={obs:10.2f}   p_B={p:.3f}")

print("--- Bayesian p-values on standard deviation (tail sensitivity) ---")
bayes_p_value(idata_m0, np.std, "M0 Poisson")
bayes_p_value(idata_m1, np.std, "M1 NegBin")
bayes_p_value(idata_m2, np.std, "M2 Hier NegBin")
```

> 🩺 **Diagnostic (how to read $p_B$):** A well-calibrated model gives a Bayesian p-value near **0.5** for the test statistic. For the **Poisson**, the standard-deviation p-value will be near **0** — the observed spread exceeds that of *almost every* Poisson replicate (a $p_B$ of, say, 0.002 means roughly 2 in 1000 replicates reach the observed SD, not literally zero), a flat-out rejection of equidispersion. The **NegBin** models pull $p_B$ back toward 0.5. This is the workflow working as designed: a check failed (overdispersion), so we expanded the model (Poisson → NegBin), and the next check confirms the expansion fixed the specific defect. We did not guess; the data told us. (`az.plot_bpv(idata, kind="p_value")` draws this for you.)

## A.7 — COMPARE / EXPAND

We have three models. Which generalizes best? This is exactly the predictive-accuracy question Ch 11 answers with **PSIS-LOO** (leave-one-out cross-validation estimated cheaply via Pareto-smoothed importance sampling). We computed `log_likelihood` at sample time, so:

```python
loo_m0 = az.loo(idata_m0)
loo_m1 = az.loo(idata_m1)
loo_m2 = az.loo(idata_m2)

comparison = az.compare({"poisson": idata_m0, "nb": idata_m1, "hier_nb": idata_m2})
print(comparison)
az.plot_compare(comparison)
```

> 🩺 **Diagnostic (reading `az.compare`):** The table is sorted best-first by ELPD (expected log predictive density). Read three columns. **`elpd_loo`**: higher is better. **`elpd_diff` / `dse`**: each model's ELPD *minus the best model's* (so the best row is 0) and the standard error *of that difference* — if a model's `elpd_diff` exceeds roughly 2× its `dse`, the difference is real, not noise. (Older ArviZ versions spelled `elpd_diff` as `d_loo`; print `comparison.columns` to confirm the names in your installed version.) **`p_loo`**: the effective number of parameters; if it balloons past the actual parameter count, suspect model misspecification or influential points. You should see the **Poisson dead last by a wide margin** (overdispersion murders its out-of-sample density), the NegBin a large jump better, and the **hierarchical NegBin best** — because borrowing strength across days both improves fit and, crucially, gives honest predictive uncertainty.

> ⚠️ **Pitfall:** Always check the Pareto-$\hat k$ diagnostic that `az.loo` prints. **$\hat k < 0.7$** means the PSIS-LOO estimate is trustworthy; values $\ge 0.7$ flag observations whose importance weights are too heavy-tailed for the approximation, and $\ge 1.0$ is unreliable. A few high-$\hat k$ points usually mean a handful of influential outliers (a freak high-traffic hour, say). When `az.loo` warns, look at *which* points and whether your likelihood should be even heavier-tailed — but don't silently trust an ELPD number sitting on bad $\hat k$ values.

This is also where you'd ask: *did a check demand more complexity?* If the PPC still showed excess zeros beyond what the NegBin produces, the next expansion would be a **zero-inflated** or **hurdle** NegBin (Ch 13). If the temperature effect looked non-linear in a residual plot, you'd add a spline (Ch 14). The discipline is identical every time: **expand only because a check failed, and re-run the loop.**

## A.8 — INTERPRET & DECIDE

Diagnostics pass, the PPC is honest, LOO crowned the hierarchical NegBin. *Now* — and only now — we interpret. And we interpret on the scale stakeholders understand, not the log scale.

```python
post = idata_m2.posterior

# Temperature effect as an Incidence Rate Ratio (IRR): exp(beta) per +1 SD of temp.
beta_draws = post["beta_temp"].values.ravel()
irr = np.exp(beta_draws)
print("Temperature IRR per +1 SD:  median=%.2f  94%% HDI=[%.2f, %.2f]"
      % (np.median(irr), *az.hdi(irr, hdi_prob=0.94)))

# How big is day-to-day variation? sigma_u on the log scale; exp(sigma_u) ~ multiplicative.
sigma_u_draws = post["sigma_u"].values.ravel()
print("Day-level SD (log scale):   median=%.2f  94%% HDI=[%.2f, %.2f]"
      % (np.median(sigma_u_draws), *az.hdi(sigma_u_draws, hdi_prob=0.94)))
```

> 🧮 **The math (interpreting log-link coefficients):** A coefficient $\beta$ on the log scale becomes a **multiplicative** effect on the count: a one-unit (here one-standard-deviation) increase in the predictor multiplies the expected count by $e^{\beta}$ — the **incidence rate ratio (IRR)**. So if `beta_temp` has median 0.30, warmer-by-one-SD days see about $e^{0.30}\approx 1.35$, a 35% higher expected ridership. Report the IRR with its HDI, never the raw log coefficient, because no stakeholder thinks in log-rates.

The decision-relevant quantity — what you'd actually staff against — is the **predictive count for a specific scenario**, *with its uncertainty*. Suppose we want the predicted count for hour 17 (rush hour), warm weather, on a *new, unobserved day*. The hierarchical model shines here: an unseen day's offset is a fresh draw from the learned population $u_{\text{new}} \sim \mathrm{Normal}(0, \sigma_u)$, so $\sigma_u$ automatically **inflates** the predictive interval for a day we've never measured (Ch 10's headline advantage).

```python
# Predictive count for a NEW day at a chosen scenario, propagating ALL uncertainty.
alpha_d = post["alpha"].values.ravel()
beta_d = post["beta_temp"].values.ravel()
# pick a warm temperature scenario (+1 SD) and a baseline weather level (index 0)
gamma_d = post["gamma"].values[..., 0].ravel()
sig_u_d = post["sigma_u"].values.ravel()
alpha_nb_d = post["alpha_nb"].values.ravel()

temp_scn = 1.0
u_new = rng.normal(0.0, sig_u_d)                 # fresh day offset from the population
log_mu_scn = alpha_d + beta_d * temp_scn + gamma_d + u_new
mu_scn = np.exp(log_mu_scn)
# draw actual counts from the NegBin predictive (mu, alpha) -> propagate count noise too
y_scn = rng.negative_binomial(alpha_nb_d, alpha_nb_d / (alpha_nb_d + mu_scn))

print("Predicted count, new warm rush-hour day:")
print("  median=%.0f   80%% interval=[%.0f, %.0f]   95%% interval=[%.0f, %.0f]"
      % (np.median(y_scn),
         *np.percentile(y_scn, [10, 90]),
         *np.percentile(y_scn, [2.5, 97.5])))
```

> 💡 **Intuition (the decision):** Notice we propagated **three** layers of uncertainty: posterior uncertainty in the coefficients, between-day uncertainty via a fresh $u_{\text{new}}$, and the NegBin count noise. A frequentist point forecast would hand you a single number; we hand you a full predictive distribution. If you staff to the **90th percentile** of `y_scn`, you'll be under-resourced one rush-hour-day in ten — a decision you can now make *quantitatively*, trading the cost of idle crews against the cost of a stockout. That is the entire payoff of going Bayesian for a decision problem: the uncertainty is the deliverable, not an afterthought.

**What Case A taught.** Counts forced a non-Gaussian likelihood; overdispersion (a *check*, not a hunch) forced Poisson → NegBin; nested structure forced a hierarchy; the hierarchy forced non-centering; and LOO + PPC adjudicated the whole sequence. Every piece of complexity was *demanded by a failed check* — the workflow's prime directive. Hold that pattern; you'll see it again, differently dressed, in the next two case studies.

---
---

# Case Study B — A Structural / Gaussian-Process Time-Series Forecast

### Mauna Loa CO₂: trend + seasonality, and a forecast that widens honestly

> *The question behind this one: "Where is atmospheric CO₂ headed, and how sure are we?" This is the case study that teaches the hardest lesson in forecasting — that your uncertainty must grow as you extrapolate, and that a model which forecasts with constant-width bands is lying to you. We'll decompose the series into a smooth rising trend plus an annual cycle plus noise, fit it with the Hilbert-Space Gaussian Process (HSGP) approximation so it actually scales, and then watch the predictive interval fan out as we push past the last observation. If you internalize one figure from this chapter, make it that fanning interval.*

## B.1 — FRAME the question

What decision rides on a CO₂ forecast? Policy, sensor calibration, climate-model boundary conditions — but for our purposes the decision is **"how wide should my honest uncertainty band be 5 years out?"** because that is the question every forecasting model gets wrong in the same direction: too confident. The Mauna Loa series has three obvious structural pieces, and naming them *is* the modeling:

1. A **long, smooth, accelerating upward trend** (the Keeling curve).
2. A **strong annual seasonal cycle** (Northern-Hemisphere plants inhale in summer, exhale in winter).
3. **Short-scale noise** and measurement wiggle.

A Gaussian process gives us a principled language for "smooth function of unknown shape," and — this is the elegant part — **adding kernels adds structure**: an ExpQuad kernel for the slow trend *plus* a periodic kernel for the seasonality *plus* a noise term. The GP's covariance is the sum, and the posterior automatically decomposes the data into those components. That additive-kernel idea is the GP view of a structural time series (Ch 12, Ch 15).

> 💡 **Intuition:** A GP is a **distribution over functions**. Instead of guessing a parametric form (a cubic? a sine?), you specify *how correlated* the function's values are at nearby inputs — the kernel — and let the data pick the function. The lengthscale knob controls wiggliness: short lengthscale = wiggly, long lengthscale = nearly flat. For a slow trend you want a long lengthscale; for the annual cycle you want a periodic kernel with a one-year period. The model is a small world of plausible smooth curves, and the data carves out the posterior subset that fits.

## B.2 — DATA & preprocessing

```python
import statsmodels.api as sm

co2_raw = sm.datasets.co2.load_pandas().data    # weekly CO2 ppm, 1958-2001
co2 = co2_raw.dropna().resample("MS").mean().dropna()    # monthly means
print(co2.head())
print("n months:", len(co2))

# x = years since start (centered later for HSGP). y = CO2 ppm.
t = (co2.index - co2.index[0]).days.values / 365.25       # years since first obs
y_raw = co2["co2"].to_numpy()

# Standardize y so amplitude/lengthscale priors live on a known scale (Ch 03).
y_mean, y_std = y_raw.mean(), y_raw.std()
y = (y_raw - y_mean) / y_std

# Center x for the HSGP boundary logic (HSGP wants data centered near 0).
x_mean = t.mean()
x = (t - x_mean)[:, None]            # GP inputs MUST be 2-D: shape (n, 1)
print("x range:", x.min(), "to", x.max(), "  shape", x.shape)
```

> ⚠️ **Pitfall:** GP input arrays must be **two-dimensional**, shape `(n, 1)` for a 1-D input — the single most common GP error is passing a flat `(n,)` array and getting a cryptic shape error from the covariance function. Always `x[:, None]`. And standardize `y`: it lets you reason about the amplitude prior (the GP's marginal SD) as "a fraction of the data's SD," which is a scale you can actually think about.

**Train/forecast split.** To *demonstrate* honest widening, we deliberately hold out the last several years and forecast them, so we can see both the interval fan out and whether the truth stays inside it.

```python
n_test = 60                       # hold out the last 5 years (60 months)
x_train, x_test = x[:-n_test], x[-n_test:]
y_train, y_test = y[:-n_test], y[-n_test:]
print("train:", x_train.shape[0], " forecast horizon:", x_test.shape[0], "months")
```

## B.3 — PRIORS + prior predictive check

The GP's priors are priors on **lengthscale, amplitude, and noise** — and the lengthscale prior is where GPs go to die if you're careless. With finite data the likelihood is *flat toward short lengthscales*: a GP can "explain" any dataset by wiggling through every point, which overfits and, worse, creates vicious posterior geometry and divergences. The cure (Betancourt's robust-GP advice, Ch 12) is an **InverseGamma** or **Gamma** lengthscale prior: zero density at 0 (kills infinitesimal lengthscales) with a heavy right tail (still allows long, smooth trends).

First, a word on the architecture, because it is a genuine trap. The *intuition* is "one GP whose covariance is `cov_trend + cov_seas`," and if we were using a full latent GP we could literally write `cov = cov_trend + cov_seas` and hand it to `pm.gp.Latent`. But **plain `pm.gp.HSGP` cannot take a covariance that contains a `Periodic` kernel** — HSGP needs each kernel's spectral density, and `Periodic` does not have one in the form HSGP requires. So the scalable design is **two GP objects**: an `HSGP` for the smooth trend and a separate `HSGPPeriodic` for the seasonal cycle, with their latent functions *summed inside the model*. That two-component form is what we prior-check here and fit in B.4.

```python
# --- the INTUITION (pseudocode — does NOT run as an HSGP) ---
# The mental model is a single GP whose covariance is the SUM of a trend kernel
# and a seasonal kernel:
#     cov = eta_trend**2 * ExpQuad(...)  +  eta_seas**2 * Periodic(...)
# With a full `pm.gp.Latent(cov_func=cov)` this is literally valid. But
# `pm.gp.HSGP(cov_func=cov)` will RAISE, because HSGP needs a spectral density for
# every kernel and `Periodic` doesn't provide one. So below we build it as TWO
# scalable GP objects (HSGP trend + HSGPPeriodic seasonal) and sum the functions.

# --- the RUNNABLE two-component prior-predictive model ---
# Prior reasoning (on standardized y):
#   eta (amplitude) ~ HalfNormal(...) : function deviations as a fraction of y's SD
#   ell_t ~ InverseGamma centered well above the seasonal scale (years)
#   ell_s : within-period wiggliness; period fixed at 1.0 year
#   sigma (noise) ~ HalfNormal(0.5)   : a fraction of y's SD

with pm.Model() as prior_gp:
    # trend: smooth ExpQuad via HSGP (amplitude folded into cov_func as eta**2 * k)
    eta_trend = pm.HalfNormal("eta_trend", 2.0)
    ell_trend = pm.InverseGamma("ell_trend", mu=5.0, sigma=2.0)   # ~5-year trend scale
    cov_trend = eta_trend**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_trend)
    gp_trend = pm.gp.HSGP(m=[200], c=1.5, cov_func=cov_trend)
    f_trend = gp_trend.prior("f_trend", X=x_train)

    # seasonal: periodic via HSGPPeriodic (amplitude passed as scale=, kernel is BARE)
    eta_seas = pm.HalfNormal("eta_seas", 1.0)
    ell_seas = pm.InverseGamma("ell_seas", mu=1.0, sigma=0.5)
    cov_seas = pm.gp.cov.Periodic(input_dim=1, period=1.0, ls=ell_seas)
    gp_seas = pm.gp.HSGPPeriodic(m=20, scale=eta_seas, cov_func=cov_seas)
    f_seas = gp_seas.prior("f_seas", X=x_train)

    f = pm.Deterministic("f", f_trend + f_seas)
    sigma = pm.HalfNormal("sigma", 0.5)
    pm.Normal("y", mu=f, sigma=sigma, observed=y_train)

    idata_prior_gp = pm.sample_prior_predictive(samples=50, random_seed=RANDOM_SEED)
```

> ⚠️ **Pitfall (HSGP and periodic kernels):** Plain `pm.gp.HSGP` supports `ExpQuad`, `Matern32`, and `Matern52` — kernels with a known spectral density — but **not** an arbitrary `Periodic`; passing a `cov_trend + cov_seas` sum that contains `Periodic` to `pm.gp.HSGP` *raises*, which is exactly why the pseudocode above is flagged as non-runnable and the actual prior-predictive call sits on the two-object form. For a periodic component the dedicated class is **`pm.gp.HSGPPeriodic`**, whose constructor is `HSGPPeriodic(m, scale=1.0, *, mean_func, cov_func)` — note `scale` *is* the amplitude (the GP's marginal SD), so its `cov_func` must be a **bare** `Periodic(...)`, not `eta**2 * Periodic(...)`. The clean, robust design for Mauna Loa is therefore: an **HSGP with an ExpQuad (or Matérn-5/2) trend** plus a **separate `HSGPPeriodic` seasonal component**, summed inside the model — exactly the runnable block above, and the form B.4 fits. The lesson stands regardless of which approximation engine you pick: trend kernel + seasonal kernel + noise.

Here is the **prior predictive check** read: draw functions from the prior and confirm they look like *plausible CO₂-shaped curves* — smoothly rising with an annual ripple — not white noise and not flat lines.

```python
fig, ax = plt.subplots(figsize=(8, 4))
fdraws = idata_prior_gp.prior["f"].stack(s=("chain", "draw"))
for i in range(min(30, fdraws.sizes["s"])):
    ax.plot(x_train.ravel(), fdraws.isel(s=i).values, color="C0", alpha=0.25, lw=0.8)
ax.plot(x_train.ravel(), y_train, "k.", ms=2, label="observed (standardized)")
ax.set(xlabel="centered years", ylabel="standardized CO2",
       title="Case B — prior predictive functions (should look smooth + rising-capable)")
ax.legend()
```

> 🩺 **Diagnostic:** Prior GP draws should be **smooth** (the lengthscale prior is doing its job — no infinitesimal-lengthscale spaghetti) and **flexible enough** to rise and ripple like CO₂, with amplitude comparable to the data's spread. If the draws are wild and jagged, your lengthscale prior allows too-short lengthscales — push more InverseGamma mass to longer scales. If they're dead-flat lines, your amplitude is too small. We're tuning the *space of functions the model can imagine* before we ask it to fit.

## B.4 — MODEL

The model that actually scales. We use HSGP for the trend (cheap, $O(nm)$ instead of $O(n^3)$) and HSGPPeriodic for the seasonal cycle, sum the latent functions, and put a Gaussian observation noise on top.

```python
with pm.Model() as gp_model:
    x_data = pm.Data("x", x_train)          # 2-D inputs

    # --- trend component: smooth Matern-5/2 via HSGP ---
    eta_t = pm.HalfNormal("eta_t", 2.0)
    ell_t = pm.InverseGamma("ell_t", mu=5.0, sigma=2.0)     # ~years
    cov_t = eta_t**2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_t)
    gp_trend = pm.gp.HSGP(m=[200], c=2.0, cov_func=cov_t)   # c>=1.2; data well inside [-L, L]
    f_trend = gp_trend.prior("f_trend", X=x_data)

    # --- seasonal component: periodic via HSGPPeriodic ---
    # HSGPPeriodic(m, scale=1.0, *, mean_func, cov_func): `scale` IS the GP's amplitude
    # (its marginal SD), so the kernel must be a BARE Periodic — do NOT also bake eta_s
    # into cov_func, or you double-count the amplitude. (Contrast plain HSGP, which folds
    # the amplitude into cov_func as eta**2 * k; HSGPPeriodic takes it as `scale=` instead.)
    eta_s = pm.HalfNormal("eta_s", 1.0)
    ell_s = pm.InverseGamma("ell_s", mu=1.0, sigma=0.5)
    cov_s = pm.gp.cov.Periodic(input_dim=1, period=1.0, ls=ell_s)   # bare kernel, no eta_s**2
    gp_seas = pm.gp.HSGPPeriodic(m=20, scale=eta_s, cov_func=cov_s)
    f_seas = gp_seas.prior("f_seas", X=x_data)

    # --- combine + observation noise ---
    f = pm.Deterministic("f", f_trend + f_seas)
    sigma = pm.HalfNormal("sigma", 0.5)
    pm.Normal("y", mu=f, sigma=sigma, observed=y_train)

    idata_gp = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                         random_seed=RANDOM_SEED,
                         idata_kwargs={"log_likelihood": True})
```

> 🔧 **In practice (why HSGP, not a plain `Latent` GP):** A full latent GP is $O(n^3)$ from the Cholesky of an $n\times n$ covariance — with ~520 monthly points that's already slow, and any real series buries it. HSGP approximates the kernel by its spectral density on a fixed set of basis functions; cost drops to $O(nm)$ and — the killer feature — **the basis is independent of the kernel hyperparameters**, so nothing recomputes during sampling. Two knobs: `m` (number of basis functions; more `m` captures *shorter* lengthscales) and `c` (boundary factor; larger `c` captures *longer* lengthscales but needs larger `m`). Riutort-Mayol et al. recommend `c ≥ 1.2`, and your train **and** forecast points must sit well inside the boundary $[-L, L]$ — which matters enormously for forecasting, as we'll see in a moment.

> ⚠️ **Pitfall (the forecast-boundary trap):** Because we will forecast *beyond* `x_train`, the forecast inputs `x_test` must still fall inside the HSGP boundary `[-L, L]` where `L = c · max|x_centered|`. If you set `c` too small, your extrapolation runs off the edge of the basis and the forecast goes haywire (truncated, biased near the edge). I use `c=2.0` and will verify the test points are inside. This is a forecasting-specific subtlety the in-sample tutorials never warn you about.

## B.5 — FIT + diagnostics

```python
n_div = int(idata_gp.sample_stats["diverging"].sum())
print("divergences:", n_div)
print(az.summary(idata_gp, var_names=["eta_t", "ell_t", "eta_s", "ell_s", "sigma"]))

az.plot_trace(idata_gp, var_names=["ell_t", "ell_s", "sigma"])
az.plot_energy(idata_gp)
```

> 🩺 **Diagnostic (GP-specific reads):** The danger sign in any GP fit is the **lengthscale wandering toward zero** with a swarm of divergences — the non-identifiability we priored against. Check that `ell_t`'s posterior sits comfortably away from 0 (our InverseGamma should guarantee it) and that `r_hat ≤ 1.01`, `ess ≳ 400`, `0 divergences`. If you *do* see lengthscale-collapse divergences, the fixes in order are: confirm the InverseGamma prior is actually pushing mass off zero, raise `target_accept` to 0.95–0.99, and if a Cholesky `LinAlgError` appears, bump the `jitter` in `.conditional` to `1e-4`. The energy plot should show overlapping marginal/transition curves and BFMI > 0.3.

A lovely sanity check unique to additive GPs: pull out the **trend** and **seasonal** posteriors separately and confirm the decomposition makes physical sense.

```python
ft = idata_gp.posterior["f_trend"].mean(("chain", "draw")).values
fs = idata_gp.posterior["f_seas"].mean(("chain", "draw")).values
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(x_train.ravel(), ft); axes[0].set_title("recovered TREND component")
axes[1].plot(x_train.ravel(), fs); axes[1].set_title("recovered SEASONAL component")
axes[1].set_xlabel("centered years")
```

> 🩺 **Diagnostic:** The trend panel should be a smooth, monotonically rising curve (the Keeling curve, de-seasonalized); the seasonal panel should be a clean ~1-year oscillation of small amplitude. If the "trend" is wiggling annually, your trend lengthscale is too short and it's stealing the seasonality — push `ell_t` longer. This component-recovery plot is the GP analogue of residual analysis: it tells you the model is attributing variation to the *right* structure, not fitting noise with the wrong kernel.

## B.6 — POSTERIOR predictive + the forecast that widens

Now the centerpiece. We extend the GP to the forecast inputs with `.conditional`, which gives the posterior of the latent function at new $x$ — and because we ask for the **predictive of observations** (function + noise), the bands include both the GP's extrapolation uncertainty *and* the observation noise.

```python
x_all = x                                    # full timeline incl. held-out tail
with gp_model:
    # latent function over the WHOLE timeline (train + forecast horizon)
    f_trend_pred = gp_trend.conditional("f_trend_pred", Xnew=x_all)
    f_seas_pred = gp_seas.conditional("f_seas_pred", Xnew=x_all)
    f_pred = pm.Deterministic("f_pred", f_trend_pred + f_seas_pred)
    # predictive for OBSERVATIONS = latent f + Gaussian noise
    y_pred = pm.Normal("y_pred", mu=f_pred, sigma=sigma, shape=x_all.shape[0])

    idata_gp.extend(pm.sample_posterior_predictive(
        idata_gp, var_names=["f_pred", "y_pred"], random_seed=RANDOM_SEED))
```

```python
# Un-standardize back to ppm for an honest plot.
def unstd(z):
    return z * y_std + y_mean

yp = idata_gp.posterior_predictive["y_pred"].stack(s=("chain", "draw"))
ypp = unstd(yp.values)                                   # (n_all, n_samples)
med = np.median(ypp, axis=1)
lo50, hi50 = np.percentile(ypp, [25, 75], axis=1)
lo95, hi95 = np.percentile(ypp, [2.5, 97.5], axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(t, lo95, hi95, alpha=0.20, label="95% predictive")
ax.fill_between(t, lo50, hi50, alpha=0.35, label="50% predictive")
ax.plot(t, med, "C0", lw=1.5, label="posterior median")
ax.plot(t[:-n_test], unstd(y_train), "k.", ms=2, label="train")
ax.plot(t[-n_test:], unstd(y_test), "r.", ms=4, label="held-out truth")
ax.axvline(t[-n_test], color="grey", ls="--", label="forecast start")
ax.set(xlabel="years since 1958", ylabel="CO2 (ppm)",
       title="Case B — Mauna Loa forecast: intervals FAN OUT past the data")
ax.legend(loc="upper left")
```

> 💡 **Intuition (the single most important figure in forecasting):** Look at what happens to the band at the dashed forecast line. To the *left*, where we have data, the 95% interval is tight — the GP is pinned down by observations. To the *right*, past the last datum, the band **fans out**: with no data to constrain it, the GP's extrapolation uncertainty grows, and grows *faster* the further you push. This is the honest behavior every naive forecaster gets wrong. A linear regression with a fitted slope hands you a constant-width prediction band forever; the GP *knows* it knows less about 2010 than about 2001, and it says so by widening. If your held-out red points stay inside the fanning band, your forecast uncertainty is **calibrated** — which is the only kind of forecast worth shipping.

> 🩺 **Diagnostic (was the forecast honest?):** Two checks. (1) **Coverage**: roughly 95% of the held-out red points should fall inside the 95% band, ~50% inside the 50% band. If far fewer do, the model is overconfident (bands too narrow — often a too-short trend lengthscale or underestimated noise); if essentially all do with room to spare, it may be under-confident. (2) **The fanning itself**: confirm the band width at the forecast horizon is *visibly larger* than at the last training point. A flat-width forecast band is a red flag that you've accidentally extrapolated a rigid parametric trend instead of a GP.

## B.7 — COMPARE / EXPAND

How do we know the trend-plus-seasonal structure earns its keep? Compare against a **trend-only** GP (drop the seasonal component) on in-sample LOO, and — more honest for a forecasting task — compare **out-of-sample predictive accuracy** on the held-out tail directly.

```python
# (a) In-sample LOO: does seasonality improve generalization?
# Fit a trend-only variant (same code, omit f_seas) -> idata_trend_only, then:
# comparison = az.compare({"trend+seasonal": idata_gp, "trend_only": idata_trend_only})
# print(comparison)    # expect trend+seasonal to win decisively (CO2 seasonality is huge)

# (b) Honest forecast scoring on the held-out tail: log predictive density of y_test.
yp_test = ypp[-n_test:, :]                       # predictive draws over the horizon
truth = unstd(y_test)
# crude pointwise log-score via a Gaussian fit per horizon step:
mu_h = yp_test.mean(axis=1); sd_h = yp_test.std(axis=1)
from scipy import stats
logscore = stats.norm.logpdf(truth, mu_h, sd_h).sum()
print("held-out log predictive score (higher=better):", round(logscore, 1))
```

> 🩺 **Diagnostic:** `az.compare` (Ch 11) on the in-sample fits will reward the seasonal component because CO₂'s annual cycle is enormous and dropping it forces the noise term to absorb it — wrecking the fit. But for forecasting, the **held-out log-score** is the more honest currency: it scores the *actual* predictive distribution against *actual* future values. A model can win in-sample LOO and still forecast worse if it over-fits the trend; computing both keeps you honest. If the trend-only model's forecast band fails to fan correctly, that shows up as a brutal log-score the moment a seasonal trough lands outside its interval.

This is also the natural place to consider expansions the checks might demand: a **changepoint** in the trend (the Keeling curve accelerates — a `ScaledCov` or a `LevelTrendComponent` from `pymc_extras.statespace`, Ch 15), a **Matérn** instead of ExpQuad for slightly rougher realism, or a heavier-tailed observation noise if the residuals show outliers. As always: expand only when a check fails, then re-run the loop.

## B.8 — INTERPRET & DECIDE

```python
# The decision quantity: forecast CO2 5 years out, with honest uncertainty.
horizon_idx = -1                                  # last forecast month
fc = ypp[horizon_idx, :]
print("CO2 forecast at horizon (ppm):")
print("  median=%.1f   80%% interval=[%.1f, %.1f]   95%% interval=[%.1f, %.1f]"
      % (np.median(fc), *np.percentile(fc, [10, 90]), *np.percentile(fc, [2.5, 97.5])))

# trend slope: how fast is CO2 rising, in ppm/year, with uncertainty?
# Finite-difference the posterior TREND component (de-seasonalized) over the last
# few years of training data, then un-standardize back to ppm/year.
#   - f_trend is standardized; multiply by y_std to get ppm.
#   - x is in centered YEARS, so a difference in x is already in years.
f_tr = idata_gp.posterior["f_trend"].stack(s=("chain", "draw")).values   # (n_train, n_draws)
xt = x_train.ravel()                                  # centered years
yrs_window = 5.0                                       # estimate slope over last ~5 yr
i_lo = int(np.searchsorted(xt, xt[-1] - yrs_window))
slope_draws = (f_tr[-1, :] - f_tr[i_lo, :]) * y_std / (xt[-1] - xt[i_lo])  # ppm/year
print("CO2 trend slope over the last %.0f yr of training data:" % yrs_window)
print("  median=%.2f ppm/yr   94%% HDI=[%.2f, %.2f]"
      % (np.median(slope_draws), *az.hdi(slope_draws, hdi_prob=0.94)))
```

> 💡 **Intuition (the deliverable):** The answer to "where is CO₂ headed" is **not** a number — it is a distribution that gets wider with horizon. We report the median forecast *and* the 95% interval, and we state plainly that the interval at 5 years is wider than at 1 year *because that is true*. The trend-slope computation above turns the de-seasonalized trend into the decision-relevant rate "≈ N ppm/year, 94% HDI [·,·]" — note it carries an HDI, because even *how fast* CO₂ is rising is uncertain, and a point slope would hide that. A stakeholder who wants "the number" is asking the wrong question; your job is to make the uncertainty legible and let them decide against it (e.g., plan for the upper tail of the band). The GP's fanning interval is the quantitative honesty that constant-width forecasts lack.

**What Case B taught.** Time series forced us to think in *components* (trend + seasonal + noise), which the additive-kernel GP expresses natively. Scale forced HSGP over a full latent GP. The lengthscale non-identifiability forced an InverseGamma prior. And forecasting forced the single non-negotiable lesson: **uncertainty must widen as you extrapolate, and a model that doesn't widen is hiding its ignorance.** The boundary factor `c`, the held-out coverage check, and the fanning band were all in service of that one honest forecast. Now to the third loop — where the data themselves are incomplete.

---
---

# Case Study C — Logistic Regression with Missing Data

### Titanic survival: imputing covariates, checking calibration, and turning a posterior into a decision

> *The question behind this one: "Given a passenger's attributes, what is the probability they survived — and how do we make a yes/no call from that probability when some attributes are missing?" This case study braids together three things the textbooks usually teach separately: missing data done the Bayesian way (impute the missing values as parameters, propagating their uncertainty instead of throwing rows away), calibration checking (a model can be accurate-on-average and still systematically over- or under-confident — LOO-PIT catches it), and decision-making (a probability is not a decision; you need a threshold and a cost). This is the analysis that looks most like the messy real world, because real data always has holes.*

## C.1 — FRAME the question

The decision is concrete: **classify each passenger as survived / not**, but more usefully, **report a calibrated survival probability** so a downstream user can pick their own threshold against their own costs. The catch is the data: the Titanic `age` column has genuine missing values — about 20% of passengers have no recorded age. The wrong move (and the default in most ML pipelines) is to **drop those rows** or **mean-impute** age. Both are statistically dishonest:

- **Dropping** discards real information from every *other* column of those passengers, and if age-missingness correlates with survival (it does — missingness is informative here), it **biases** your estimates.
- **Mean-imputation** pretends you know the missing ages exactly, which **understates uncertainty** — your downstream probabilities will be falsely confident.

The Bayesian move is to treat each missing age as **a parameter with its own posterior**, inferred jointly with everything else. Each posterior draw fills the holes with a *different* plausible age, so the uncertainty about those ages flows automatically into the survival probabilities. This is, mathematically, proper multiple imputation — without Rubin's-rules bookkeeping (Ch 16).

> 💡 **Intuition:** "Missing" is not "delete." A missing value is just **a quantity you don't know**, and Bayesian inference is the calculus of quantities you don't know. So you give it a prior, condition on everything you *do* observe, and read its posterior. The missing ages and the survival coefficients are inferred in the same breath, each informing the other. That is the whole idea, and it is beautiful: the same machinery that estimates a slope estimates a hole in your spreadsheet.

## C.2 — DATA & preprocessing

```python
import seaborn as sns

titanic = sns.load_dataset("titanic")
cols = ["survived", "pclass", "sex", "age", "fare"]
df = titanic[cols].copy()

print("rows:", len(df))
print("missing per column:\n", df.isna().sum())
# 'age' has ~177 NaNs; 'fare' occasionally one. survived/pclass/sex complete.

# Encode predictors. Standardize the continuous ones (Ch 03 default).
df["male"] = (df["sex"] == "male").astype(float)
df["pclass"] = df["pclass"].astype(int)
# fill the rare missing fare with median BEFORE standardizing (or impute it too);
# we focus the imputation lesson on AGE, the column with substantial missingness.
df["fare"] = df["fare"].fillna(df["fare"].median())
df["fare_z"] = (df["fare"] - df["fare"].mean()) / df["fare"].std()

# AGE: keep the NaNs! We will impute them. Standardize using observed-age moments.
age_obs_mean = df["age"].mean()       # mean over OBSERVED ages
age_obs_std = df["age"].std()
df["age_z"] = (df["age"] - age_obs_mean) / age_obs_std   # NaNs survive standardization

y = df["survived"].to_numpy().astype(int)
n_missing_age = int(df["age_z"].isna().sum())
print("age values to impute:", n_missing_age)
```

> ⚠️ **Pitfall (the imputation footgun, from Ch 16):** PyMC's automatic imputation triggers when you pass `observed=` an array **containing `np.nan`**. It then creates two nodes in the InferenceData: `<name>_observed` (the present values) and `<name>_unobserved` (the imputed ones, sampled as free parameters). But this **fails silently if you wrap the NaN array in `pm.Data`** (issue #6626). So: pass the raw NaN numpy array directly to `observed=`, never `observed=pm.Data(...)`. Also note — automatic imputation works for the *observed likelihood variable*, and to impute a missing **predictor** like age you must give age its **own distribution** with `observed=age_with_nan`, then use the resulting partially-latent age downstream. That "joint model of X and Y" is exactly what we build next.

> 🔧 **In practice:** I standardize age using the **observed-age** mean and SD, which is fine because I'm about to give the latent age a prior on that standardized scale (a `Normal(0,1)` is a sensible population model for standardized age). If age-missingness were strongly informative in a way that shifts the distribution (MNAR), I'd model the missingness mechanism explicitly — more on that in the expansion step.

## C.3 — PRIORS + prior predictive check

Two sets of priors now: the **logistic regression coefficients** (on the log-odds scale) and the **imputation model for age** (a distribution the missing ages are drawn from).

```python
# Coefficient priors on the LOG-ODDS scale. Standardized predictors => Normal(0, 1.5)
# on the intercept and Normal(0, 1) on slopes are weakly informative and REGULARIZING
# (they prevent the separation blow-up classic to logistic regression).
#   - intercept ~ Normal(0, 1.5): baseline log-odds; invlogit(0)=0.5, +/-1.5 covers ~0.18-0.82
#   - slopes ~ Normal(0, 1): a 1-SD change shifts log-odds by ~1 => meaningful but bounded
#
# Age imputation model: latent ages ~ Normal(mu_age, sigma_age) on the standardized scale.
```

> ⚠️ **Pitfall (separation):** Logistic regression's signature failure is **(quasi-)separation** — a predictor that perfectly splits survivors from non-survivors drives its coefficient to $\pm\infty$ under a flat prior, and the sampler never converges (huge $\hat R$, coefficient runs away). The cure is *the prior itself*: a weakly-informative `Normal(0, 1)`–ish prior on standardized coefficients **regularizes** the estimate, pulling it back from infinity. Gelman et al. (2008) recommend exactly this (a `StudentT(4,0,1)` or `Normal(0, 1–2.5)`). This is a case where the prior is not a nuisance — it is load-bearing. Standardize, then prior-regularize, and separation stops being scary.

The **prior predictive check** here asks: do prior-implied survival probabilities span a sensible range (not all crushed at 0 or 1), and do prior-implied ages look like plausible ages?

```python
with pm.Model() as prior_logit:
    b0 = pm.Normal("b0", 0.0, 1.5)
    b_male = pm.Normal("b_male", 0.0, 1.0)
    b_pclass = pm.Normal("b_pclass", 0.0, 1.0)
    b_age = pm.Normal("b_age", 0.0, 1.0)
    b_fare = pm.Normal("b_fare", 0.0, 1.0)

    # imputation model for standardized age
    mu_age = pm.Normal("mu_age", 0.0, 1.0)
    sigma_age = pm.HalfNormal("sigma_age", 1.0)
    age_latent = pm.Normal("age", mu_age, sigma_age,
                           observed=df["age_z"].to_numpy())   # NaNs -> auto-imputed

    eta = (b0 + b_male * df["male"].to_numpy()
           + b_pclass * (df["pclass"].to_numpy() - 2)         # center pclass at 2
           + b_age * age_latent
           + b_fare * df["fare_z"].to_numpy())
    p = pm.Deterministic("p", pm.math.invlogit(eta))
    pm.Bernoulli("y", p=p, observed=y)

    idata_prior_c = pm.sample_prior_predictive(samples=200, random_seed=RANDOM_SEED)

p_prior = idata_prior_c.prior["p"].values.ravel()
print("prior survival prob:  1st pct=%.2f  median=%.2f  99th pct=%.2f"
      % (np.percentile(p_prior, 1), np.median(p_prior), np.percentile(p_prior, 99)))
```

> 🩺 **Diagnostic:** You want the prior-implied survival probabilities to be **spread across (0,1)**, not piled at the extremes. If your priors push every prior-predictive probability to ~0 or ~1, the coefficient priors are too wide (each draw produces near-certain outcomes) — tighten them. A healthy prior predictive here has a median near 0.5 and tails that reach but don't saturate the boundaries, meaning the model is *a priori open-minded* about who survived. We are, again, checking that the small world can contain the large one *before* fitting.

## C.4 — MODEL

The full model: logistic likelihood for survival, joint with an imputation model for the missing standardized ages.

```python
with pm.Model() as logit_impute:
    # --- coefficient priors (log-odds scale) ---
    b0 = pm.Normal("b0", 0.0, 1.5)
    b_male = pm.Normal("b_male", 0.0, 1.0)
    b_pclass = pm.Normal("b_pclass", 0.0, 1.0)
    b_age = pm.Normal("b_age", 0.0, 1.0)
    b_fare = pm.Normal("b_fare", 0.0, 1.0)

    # --- imputation model for missing standardized age ---
    mu_age = pm.Normal("mu_age", 0.0, 1.0)
    sigma_age = pm.HalfNormal("sigma_age", 1.0)
    # passing the NaN-containing array to observed= triggers automatic imputation:
    #   creates age_observed (data) and age_unobserved (free params).
    age = pm.Normal("age", mu_age, sigma_age, observed=df["age_z"].to_numpy())

    # --- linear predictor & Bernoulli likelihood ---
    eta = (b0 + b_male * df["male"].to_numpy()
           + b_pclass * (df["pclass"].to_numpy() - 2)
           + b_age * age
           + b_fare * df["fare_z"].to_numpy())
    p = pm.Deterministic("p", pm.math.invlogit(eta))
    pm.Bernoulli("y", logit_p=eta, observed=y)     # logit_p is numerically safer than p=

    idata_c = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                        random_seed=RANDOM_SEED,
                        idata_kwargs={"log_likelihood": True})
```

> 🧮 **The math (why this is proper multiple imputation):** The joint model factorizes as $p(y, \text{age}_{\text{mis}}, \theta \mid \text{age}_{\text{obs}}, X) \propto p(y \mid X, \text{age}, \theta)\, p(\text{age} \mid \mu_{\text{age}}, \sigma_{\text{age}})\, p(\theta)$. MCMC samples the missing ages and the coefficients *jointly* from this posterior. Each posterior draw $m$ carries a complete imputed age vector $\text{age}^{(m)}_{\text{mis}}$ — and that *is* the $m$-th multiple imputation. Averaging any quantity over the draws automatically applies Rubin's rules (within- plus between-imputation variance) without you computing anything. Under **MAR** (missingness depends only on observed data) this is exactly valid and you can ignore the missingness mechanism. That's a remarkable amount of correctness for free.

The Bambi equivalent, for contrast (Bambi can fit the logistic part with one line, but automatic *covariate* imputation is a raw-PyMC strength — so this is the place where dropping to PyMC earns its keep):

```python
# Bambi handles the logistic regression itself in one line (it would drop NaN rows
# unless you impute first). Use it as the no-missingness baseline:
m_bambi = bmb.Model(
    "survived ~ male + pclass + age_z + fare_z",
    data=df.dropna(subset=["age_z"]),     # complete-case: the thing we're improving on
    family="bernoulli",
)
idata_bambi = m_bambi.fit(draws=1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=RANDOM_SEED,
                          idata_kwargs={"log_likelihood": True})
# Compare this COMPLETE-CASE Bambi fit against our IMPUTED PyMC fit in step C.7.
```

> 🔧 **In practice:** This is the clearest "raw PyMC over Bambi" moment in the whole chapter. Bambi is glorious for the regression, but joint covariate imputation — building a distribution for a missing *predictor* and threading the latent values into the linear predictor — is something you reach into PyMC to express. Knowing *when* to drop from the formula API to the modeling API is itself a senior skill.

## C.5 — FIT + diagnostics

```python
n_div = int(idata_c.sample_stats["diverging"].sum())
print("divergences:", n_div)
print(az.summary(idata_c, var_names=["b0", "b_male", "b_pclass", "b_age", "b_fare",
                                     "mu_age", "sigma_age"]))

# inspect a few imputed ages: the age_unobserved node carries their posteriors
print(az.summary(idata_c, var_names=["age_unobserved"]).head())
az.plot_trace(idata_c, var_names=["b_male", "b_pclass", "b_age"])
```

> 🩺 **Diagnostic:** Same thresholds as always — **0 divergences, $\hat R \le 1.01$, ESS $\gtrsim 400$**. Two things specific to this model. First, if a coefficient's posterior is enormous with a huge $\hat R$, suspect **separation** — but our regularizing priors should have prevented it; if not, tighten them further or standardize a missed predictor. Second, look at `az.summary(idata_c, var_names=["age_unobserved"])`: each imputed age has a posterior *mean and a credible interval*. That interval is the point — a complete-case analysis throws it away; mean-imputation collapses it to zero width; here it is honest and propagated. Skim a few to confirm they're plausible standardized ages (roughly within ±3), not runaway values.

## C.6 — POSTERIOR predictive + calibration (LOO-PIT)

A logistic model's posterior predictive is trickier to eyeball than a continuous one — the outcome is just 0/1. The check that matters most for a probabilistic classifier is **calibration**: when the model says "70% chance of survival," do roughly 70% of such passengers actually survive? An ordinary PPC reuses the data twice (it over-states fit). **LOO-PIT** fixes this with leave-one-out, computed cheaply via PSIS, and is the right calibration tool here (Ch 11).

```python
# Standard PPC overlay (categorical: compares predicted vs observed survival rate spread).
with logit_impute:
    pm.sample_posterior_predictive(idata_c, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
az.plot_ppc(idata_c, num_pp_samples=100)

# Calibration via LOO-PIT (needs the log_likelihood group, which we computed):
az.plot_loo_pit(idata_c, y="y")
# newer ArviZ also offers az.plot_loo_pit(idata_c, y="y", ecdf=True)
```

> 🩺 **Diagnostic (reading LOO-PIT — memorize the shapes):** LOO-PIT computes $u_i = \Pr(\tilde y_i \le y_i \mid y_{-i})$ for each point; if the model is calibrated these $u_i$ are **Uniform(0,1)**, so `az.plot_loo_pit` overlays their density on a uniform reference band. Read the shape:
> - **Flat / inside the band** ⇒ well-calibrated. 
> - **∩ shape (hump in the middle)** ⇒ the model is **over-dispersed / under-confident** (its predictive intervals are too wide).
> - **∪ shape (mass at the edges)** ⇒ **under-dispersed / over-confident** (intervals too narrow — the dangerous direction for a classifier; you'd act on probabilities that are too extreme).
> - **Tilt / ramp** ⇒ **bias** (systematically predicting too high or too low).
>
> For our Titanic model you're hoping for a roughly flat LOO-PIT inside the band. If it shows a ∪, the model is over-confident and you should suspect a missing predictor or interaction (e.g., the famous `sex × pclass` interaction — women in first class survived at very different rates than the additive model expects). That diagnosis would *demand* a model expansion, which is exactly the next step.

> ⚠️ **Pitfall:** If `az.plot_loo_pit` errors with "no log_likelihood group," you forgot `idata_kwargs={"log_likelihood": True}` at sample time. Fix it post-hoc with `pm.compute_log_likelihood(idata_c, model=logit_impute)` — don't refit.

## C.7 — COMPARE / EXPAND

Three comparisons earn their place here. **(1)** Imputation vs complete-case: does keeping the missing-age rows (with imputation) actually help? **(2)** Additive vs interaction model: does LOO-PIT's calibration verdict demand the `sex × pclass` interaction? **(3)** LOO across the contenders.

```python
# Expansion the calibration check may demand: add the sex x pclass interaction.
with pm.Model() as logit_interact:
    b0 = pm.Normal("b0", 0.0, 1.5)
    b_male = pm.Normal("b_male", 0.0, 1.0)
    b_pclass = pm.Normal("b_pclass", 0.0, 1.0)
    b_age = pm.Normal("b_age", 0.0, 1.0)
    b_fare = pm.Normal("b_fare", 0.0, 1.0)
    b_male_pclass = pm.Normal("b_male_pclass", 0.0, 1.0)     # interaction term

    mu_age = pm.Normal("mu_age", 0.0, 1.0)
    sigma_age = pm.HalfNormal("sigma_age", 1.0)
    age = pm.Normal("age", mu_age, sigma_age, observed=df["age_z"].to_numpy())

    male = df["male"].to_numpy(); pcl = df["pclass"].to_numpy() - 2
    eta = (b0 + b_male * male + b_pclass * pcl + b_age * age
           + b_fare * df["fare_z"].to_numpy()
           + b_male_pclass * male * pcl)
    pm.Bernoulli("y", logit_p=eta, observed=y)
    idata_interact = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=RANDOM_SEED,
                               idata_kwargs={"log_likelihood": True})

comparison_c = az.compare({"additive_imputed": idata_c,
                           "interaction_imputed": idata_interact})
print(comparison_c)
az.plot_compare(comparison_c)
```

> 🩺 **Diagnostic:** If the interaction model wins on LOO with `elpd_diff` larger than ~2× its `dse`, the `sex × pclass` structure is real and you keep it — and you'd expect its LOO-PIT to be flatter (better-calibrated) than the additive model's. This is the workflow's full circle in miniature: a *calibration check* (LOO-PIT ∪-shape) hypothesized a missing interaction; a *model expansion* added it; a *comparison* (LOO) and a *re-checked calibration* confirmed it earned its place. Note also the `p_loo` and Pareto-$\hat k$ columns — with imputed values and possible influential passengers, watch for $\hat k \ge 0.7$ and inspect those rows.

On imputation-vs-complete-case: the imputed model uses **every** passenger, so its coefficient posteriors are tighter and unbiased under MAR, whereas the complete-case Bambi fit silently conditions on "age was recorded," which biases estimates if missingness correlates with survival. The honest comparison isn't just ELPD — it's that the imputed model *answers the question for the passengers the complete-case model deleted.*

> 🔧 **In practice (MNAR and the limits of MAR):** Everything above assumes **MAR** — missingness depends only on *observed* data. If age were missing *because of* the unobserved age itself (say, ages were systematically unrecorded for the very old), that's **MNAR**, and ignoring the mechanism biases you. The fix is a **selection model**: jointly model survival, age, *and* a missingness indicator $R$ whose probability depends on the latent age (Ch 16). It's more work and the data rarely identify the mechanism well, so the honest practice is: assume MAR, state it as an assumption, and run a **sensitivity analysis** under a plausible MNAR alternative to see if your conclusions move. Naming your missingness assumption out loud is itself the senior move.

## C.8 — INTERPRET & DECIDE

First, interpret coefficients as **odds ratios** — the natural scale for log-odds — then turn probabilities into an actual **decision** with a threshold and a cost.

```python
post = idata_interact.posterior

def odds_ratio(name):
    d = post[name].values.ravel()
    return np.median(np.exp(d)), az.hdi(np.exp(d), hdi_prob=0.94)

for nm in ["b_male", "b_pclass", "b_age", "b_fare"]:
    med, hdi = odds_ratio(nm)
    print(f"{nm:12s}  OR median={med:.2f}  94% HDI=[{hdi[0]:.2f}, {hdi[1]:.2f}]")
```

> 🧮 **The math (odds ratios & the divide-by-4 rule):** A logistic coefficient $\beta$ lives on the log-odds scale; $e^{\beta}$ is the **odds ratio** — the multiplicative change in the odds of survival per unit of the predictor. So `b_male` with median $-2.5$ means $e^{-2.5}\approx 0.08$: being male multiplies the odds of survival by ~0.08, a brutal reduction ("women and children first" in one number). For a quick read on the *probability* scale, the **divide-by-4 rule**: the steepest slope of $p$ with respect to a predictor is $\beta/4$, an upper bound on the probability change per unit. Never report the raw log-odds coefficient to a stakeholder — report the odds ratio (with HDI) or a probability contrast.

Now the **decision**. A calibrated probability is not yet a classification; you need a threshold, and the right threshold depends on the **relative cost** of the two error types. Suppose (illustratively) a false negative — predicting "died" for someone who'd survive — costs you 3× a false positive. Then you don't threshold at 0.5; you threshold where expected cost is minimized.

```python
# Posterior survival probabilities per passenger (use the better model).
with logit_interact:
    pm.sample_posterior_predictive(idata_interact, var_names=["y"],
                                   extend_inferencedata=True, random_seed=RANDOM_SEED)
male = df["male"].to_numpy(); pcl = df["pclass"].to_numpy() - 2

# Reconstruct each passenger's ACTUAL standardized age. This is the whole point of the
# imputation: the decision MUST use real ages, not a constant. For observed passengers
# use df["age_z"]; for the missing ones plug in the posterior-mean of age_unobserved
# (its uncertainty is small relative to the population spread, and the per-passenger
# decision needs a single age — for full propagation, loop the cost over draws instead).
age_full = df["age_z"].to_numpy().copy()
missing_rows = np.where(df["age_z"].isna().to_numpy())[0]
age_imp_mean = idata_interact.posterior["age_unobserved"].mean(
    ("chain", "draw")).values                              # one value per missing row, in order
age_full[missing_rows] = age_imp_mean                      # fill the holes with imputed posteriors

def expected_cost(threshold, cost_fn=3.0, cost_fp=1.0):
    # use posterior-mean probabilities for the decision; for a fuller treatment,
    # average the cost over the FULL posterior of p (propagates uncertainty into the decision).
    pm_draws = post  # alias
    eta = (pm_draws["b0"].values[..., None]
           + pm_draws["b_male"].values[..., None] * male
           + pm_draws["b_pclass"].values[..., None] * pcl
           + pm_draws["b_age"].values[..., None] * age_full   # REAL ages (observed + imputed)
           + pm_draws["b_fare"].values[..., None] * df["fare_z"].to_numpy()
           + pm_draws["b_male_pclass"].values[..., None] * male * pcl)
    p = 1 / (1 + np.exp(-eta))
    p_mean = p.reshape(-1, p.shape[-1]).mean(axis=0)       # posterior-mean prob per passenger
    pred = (p_mean >= threshold).astype(int)
    fn = ((pred == 0) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    return cost_fn * fn + cost_fp * fp

thresholds = np.linspace(0.1, 0.9, 17)
costs = [expected_cost(t) for t in thresholds]
best_t = thresholds[int(np.argmin(costs))]
print("cost-minimizing threshold (FN 3x FP):", round(best_t, 2))
```

> 💡 **Intuition (probability → decision):** A model that stops at "here's $p$" has done half the job. The *decision* — survived or not, treat or not, ship or not — requires a **loss function**: what does each kind of mistake cost you? With an asymmetric loss (false negatives 3× worse), the optimal threshold drops **below** 0.5, because you'd rather over-predict survival than miss a survivor. The Bayesian payoff compounds here: because we have the *full posterior* of each probability, we can minimize **expected** cost over that posterior, propagating model uncertainty straight into the decision boundary. That is the entire arc of applied Bayesian inference — from a question, through a calibrated posterior, to a defensible action — and it's why we did all the careful work upstream. A miscalibrated probability (the ∪-shaped LOO-PIT we guarded against) would make this threshold optimization quietly wrong.

> 🔧 **In practice:** Notice the decision uses each passenger's **actual** standardized age — observed ages where recorded, the posterior-mean imputed age where missing — *not* a constant. Setting age to its mean for everyone would silently discard the very information the imputation worked to recover, and would mis-rank exactly the passengers whose ages we had to infer; using the real age vector keeps the decision faithful to the model. Two refinements push this further: (1) don't collapse to posterior-mean *probabilities* before thresholding — average the **cost itself** over the full posterior, so a passenger whose probability is uncertain contributes that uncertainty to the expected cost; (2) for the imputed ages, propagate their full posterior rather than the posterior mean by looping the cost over draws (drawing a fresh imputed age vector each draw). The code above takes the posterior-mean shortcut on both for clarity. Either way, the threshold falls out of a **cost**, never out of habit (0.5 is a default, not a decision).

**What Case C taught.** Missing data forced joint imputation (impute the hole as a parameter; never delete or mean-fill). Logistic regression forced regularizing priors against separation. The probabilistic-classifier goal forced a *calibration* check (LOO-PIT), whose ∪-shape verdict could *demand* an interaction expansion. And the word "classify" forced a decision analysis — a threshold derived from a cost, optimized over the posterior. Every step answered to a check or a decision, never to habit. That is the loop, run a third time, on the messiest data of the three.

---
---

## What the three case studies share — the invariant loop

Step back and look at the skeleton, not the flesh. Three completely different problems — overdispersed counts, a smooth time series, a binary outcome with holes — and yet the *method* never changed:

| Workflow step | Case A (Hier. NegBin) | Case B (GP / time series) | Case C (Logistic + missing) |
|---|---|---|---|
| **1. Frame** | demand by hour/weather, staff the tail | where is CO₂ headed, how wide the band | calibrated survival prob → classify |
| **2. Data** | standardize temp, index days | standardize y, center x, 2-D inputs | standardize, *keep* NaN ages |
| **3. Priors + prior-pred** | tight log-scale slopes | InverseGamma lengthscale | regularizing log-odds priors |
| **4. Model** | non-centered varying intercept | additive HSGP kernels | joint impute + Bernoulli |
| **5. Fit + diagnose** | funnel → non-center, then R̂/ESS | lengthscale-collapse watch | separation watch |
| **6. Posterior-pred** | PPC + tail Bayesian p-value | fanning forecast band, coverage | LOO-PIT calibration |
| **7. Compare/expand** | LOO: Poisson < NB < hier-NB | seasonal earns keep, held-out score | LOO + interaction expansion |
| **8. Interpret/decide** | IRR + predictive interval | forecast distribution | odds ratios + cost threshold |

> 💡 **Intuition (the meta-lesson):** Notice that the *failure modes* are what made each case distinctive — the funnel, the lengthscale collapse, separation, the over-confident LOO-PIT — and in every case the failure was caught by a **check** and cured by a **principled response** (non-center, robust prior, regularize, expand), never by brute force. The expert is not someone who picks the right model on the first try. The expert is someone who runs this loop calmly, reads each check honestly, and lets the failed checks dictate the next move. You now have that loop. Run it on your own data and you are doing real Bayesian modeling — not imitating it.

Three more cross-cutting habits that showed up in all three:

- **Standardize predictors by default** (Ch 03). It put every prior on a scale we could reason about — log-rate slopes in A, amplitude/lengthscale in B, log-odds in C. Without it, prior choice is guesswork.
- **Propagate uncertainty all the way to the deliverable.** A's predictive interval, B's fanning band, C's cost-weighted threshold all carried full posterior uncertainty into the *decision*. The uncertainty is the product, not a footnote.
- **Expand only when a check demands it.** Poisson→NegBin (PPC), trend→trend+seasonal (fit/score), additive→interaction (LOO-PIT). Every added piece of complexity had a named check behind it. This is the discipline that separates a defensible model from a baroque one.

---

## ⚠️ Common errors & how to fix them

These are the specific traps that bit (or nearly bit) us across the three case studies. Symptom → cause → fix.

| Symptom | Cause | Fix |
|---|---|---|
| Count model: `mu` overflows, `nan` logp, sampler stuck | wide prior on a **log-link** slope (e.g. `Normal(0,10)`) → `exp` explodes | keep log-scale slope priors tight (`Normal(0, 0.5)`); standardize predictors; exponentiate once inside the model |
| Poisson PPC undershoots the tail; SD Bayesian p-value ≈ 0 | **overdispersion** (Var ≫ mean) | switch to `pm.NegativeBinomial`; compare with `az.loo`/`az.compare` |
| NegBin dispersion read backwards | confusing PyMC `alpha` with textbook `φ` | remember `Var = mu + mu²/alpha`, so **small `alpha` = MORE overdispersed**, `alpha_PyMC = 1/φ` |
| Many divergences clustered at small `sigma_u` | **centered** hierarchy funnel | write **non-centered** `u = z*sigma_u`, `z~Normal(0,1)`; only then raise `target_accept` to 0.95–0.99 |
| GP: lengthscale wanders to ~0, divergences | flat likelihood toward short lengthscales (non-identifiability) | use **InverseGamma/Gamma** lengthscale prior (Betancourt); raise `target_accept` |
| GP: `ValueError`/shape error from covariance | `X` passed as 1-D `(n,)` | reshape to 2-D: `X[:, None]` |
| GP: `LinAlgError: not positive definite` | near-duplicate inputs, tiny ℓ, no jitter | raise `jitter` (e.g. `1e-4`) in `.conditional`; standardize inputs; firmer ℓ prior |
| Forecast band is constant width / goes haywire past data | rigid parametric trend, or forecast points outside HSGP `[-L, L]` | use a GP (it fans naturally); set `c` large enough (`c=2–4`) that `x_test` is well inside the boundary |
| `HSGP` rejects a `Periodic` kernel | plain HSGP needs a known spectral density (ExpQuad/Matérn only) | use **`HSGPPeriodic`** for the seasonal component; sum it with the HSGP trend |
| Imputation silently doesn't happen | NaN array wrapped in `pm.Data` (issue #6626) | pass the **raw NaN numpy array** to `observed=`, not `observed=pm.Data(...)` |
| Want to impute a missing **predictor**, auto-impute only covers the outcome | auto-imputation works on the observed *likelihood* variable | give the predictor its **own** distribution: `pm.Normal("age", ..., observed=age_nan)`, then use it downstream |
| Logistic coefficient → ±∞, huge `r_hat` | **(quasi-)separation** under a flat prior | standardize; use a regularizing prior (`Normal(0, 1)`–`StudentT(4,0,1)`) — the prior *is* the fix |
| `az.loo` / `az.plot_loo_pit` errors: "no log_likelihood group" | log-likelihood not stored | sample with `idata_kwargs={"log_likelihood": True}` or `pm.compute_log_likelihood(idata, model=...)` |
| `az.compare` ranks look unstable; `p_loo` huge | influential points; high Pareto-$\hat k$ | check $\hat k < 0.7$; inspect high-$\hat k$ rows; consider a heavier-tailed likelihood |
| LOO-PIT is ∪-shaped | model **over-confident** (predictive too narrow) | add a missing predictor/interaction; check for unmodeled overdispersion/heterogeneity |
| Discrete RV won't sample with `nuts_sampler="numpyro"` | JAX backends can't do Metropolis on discretes | use default `nuts_sampler="pymc"`, or marginalize the discrete out |

---

## 🧪 Exercises

Work these in a notebook. The first few cement the loop; the later ones push you into the judgment calls that make an expert.

**1 (conceptual — the loop).** For each of the three case studies, name the *specific check* that forced each model expansion (Poisson→NegBin, trend→trend+seasonal, additive→interaction). Then state, in one sentence each, what you would have concluded *wrongly* if you had skipped that check and shipped the simpler model. Hint: think about the tail in A, the forecast band in B, and calibration in C.

**2 (coding — overdispersion).** Refit Case A's complete-pooling Poisson and NegBin, and compute the Bayesian p-value for **three** test statistics: the standard deviation, the maximum count, and the number of hours with count above the 95th percentile. Which statistic most sharply rejects the Poisson? Hint: location statistics (the mean) stay near $p_B=0.5$ even for a broken model — that's *why* you probe the tail.

**3 (coding — the funnel, made visible).** Take Case A's hierarchical model and deliberately write the **centered** version (`u = pm.Normal("u", 0, sigma_u, dims="day")`). Sample it, count divergences, and make `az.plot_pair(idata, var_names=["sigma_u", "u"], coords={"day":[day_levels[0]]}, divergences=True)` with `sigma_u` on a log scale. Confirm the divergences pile into the neck at small `sigma_u`. Then non-center and watch them vanish. Hint: add `pm.Deterministic("log_sigma_u", pm.math.log(sigma_u))` to see the funnel clearly.

**4 (coding — honest forecasting).** In Case B, refit with a deliberately *too-small* HSGP boundary factor (`c=1.1`) and a *too-short* trend lengthscale prior. Forecast the held-out tail and overlay both forecasts. Where does the bad model's interval fail — does it fan too little, or run off the basis boundary? Compute held-out coverage (fraction of red points inside the 95% band) for both. Hint: a model that *doesn't* widen will have terrible tail coverage exactly where you most need it.

**5 (coding — calibration and decisions).** In Case C, compute and plot LOO-PIT for the additive model and the interaction model side by side. Which is better calibrated? Then, for the better model, sweep the decision threshold from 0.1 to 0.9 under **two** cost ratios (FN = 1×FP, and FN = 5×FP) and plot expected cost vs threshold. How far does the optimal threshold move? Hint: asymmetric costs pull the threshold away from 0.5 toward the cheaper error.

**6 (judgment — MNAR sensitivity).** Case C assumed age is **MAR**. Construct a simple MNAR scenario: suppose ages are missing *more often* for passengers who actually died. Simulate this on a copy of the data, refit the imputation model, and see whether the survival coefficients shift relative to the true (fully observed) values. What does this tell you about the cost of wrongly assuming MAR? Hint: under MNAR the imputation model for age is misspecified because it ignores the mechanism; a proper fix is a joint selection model (Ch 16).

**7 (stretch — a fourth case, your own).** Pick a dataset from the shared catalog you haven't modeled (radon, penguins, coal-mining disasters) and run the full eight-step loop on it from scratch, writing one sentence of narration per step explaining your judgment call. This is the real exam: can you run the loop unaided?

---

## 📚 Resources & further reading

The case studies braided together material from across the course; here are the primary sources, organized by case, every link real.

**The workflow spine (all three cases):**
- Gelman, Vehtari, Simpson, Carpenter, Bürkner et al., **"Bayesian Workflow"** (2020), arXiv:2011.01808 — the descriptive map of the whole loop; §5 (the folk theorem), §6 (model comparison), §7 (model expansion / topology of models).
- Betancourt, **"Towards A Principled Bayesian Workflow"** — https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html — the prescriptive four-question version; pairs with Ch 08.
- Martin, Kumar, Lao, **Bayesian Modeling and Computation in Python**, Ch 9 "End-to-End Bayesian Workflows" — https://bayesiancomputationbook.com/notebooks/chp_09.html — the closest analogue to this chapter, free online.

**Case A — hierarchical counts:**
- **PyMC — A Primer on Bayesian Methods for Multilevel Modeling (radon)** — https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html — the canonical non-centered varying-intercept template; copy the offset-trick block.
- Thomas Wiecki, **"Why hierarchical models are awesome, tricky, and Bayesian"** — https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/ — the clearest funnel/non-centering explanation in print.
- **PyMC — Negative Binomial regression** — https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-negative-binomial-regression.html — overdispersion diagnosis and the `(mu, alpha)` parametrization (anchor the `alpha`-vs-`φ` warning here).
- **PyMC — Diagnosing Biased Inference with Divergences** — https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html — divergences in the funnel neck and the non-centered cure.
- Vehtari, Gelman, Gabry, **"Practical Bayesian model evaluation using LOO-CV and WAIC"** (2017), *Stat. Comput.* — the LOO/PSIS machinery behind `az.compare`; the Pareto-$\hat k$ diagnostic.

**Case B — GP / structural time series:**
- **PyMC — Mauna Loa CO₂ (part 1)** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MaunaLoa.html — additive kernel design and component prediction; **(part 2)** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MaunaLoa2.html — the full multi-component model with Gamma lengthscale priors.
- **PyMC — HSGP Basic / First Steps** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html — rules of thumb for `m`, `c`, and `approx_hsgp_hyperparams`; and **HSGP Advanced** — https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Advanced.html — boundary intuition.
- Betancourt, **"Robust Gaussian Process Modeling"** — https://betanalpha.github.io/assets/case_studies/gaussian_processes.html — *the* reference for lengthscale non-identifiability and the InverseGamma fix.
- Riutort-Mayol, Bürkner, Andersen, Solin, Vehtari (2022), **"Practical Hilbert space approximate Bayesian Gaussian processes"**, arXiv:2004.11408 — the `m`/`c`/`L` theory behind `pm.gp.HSGP`.
- Rasmussen & Williams, **Gaussian Processes for Machine Learning** (free PDF) — https://gaussianprocess.org/gpml/chapters/RW.pdf — Ch 2 (regression equations), Ch 4 (kernels).
- **pymc_extras structural time series** — https://www.pymc.io/projects/extras/en/latest/statespace/models/structural.html — `LevelTrendComponent`/`TimeSeasonality` for the Kalman-filter route to the same decomposition (Ch 15).

**Case C — logistic + missing data:**
- **Bambi — Logistic Regression (ANES)** — https://bambinos.github.io/bambi/notebooks/logistic_regression.html — the formula-API logistic template and log-odds interpretation.
- **Bambi — `interpret`: Plot Comparisons** — https://bambinos.github.io/bambi/notebooks/plot_comparisons.html — odds ratios via `comparison="ratio"`, the Bayesian analogue of R's *marginaleffects*.
- **Automatic Missing Data Imputation with PyMC (Fonnesbeck)** — http://stronginference.com/missing-data-imputation.html — the `_observed`/`_unobserved` node mechanics.
- **Missing Data (ISYE 6420, Aaron Reding)** — https://areding.github.io/6420-pymc/unit6/Unit6-missingdata.html — MAR/MCAR worked PyMC v5 imputation examples.
- Gelman, Jakulin, Pittau, Su (2008), **"A weakly informative default prior distribution for logistic and other regression models"** — https://arxiv.org/pdf/0901.4011 — the StudentT(4,0,1)/Cauchy(2.5) default and the separation argument.
- **ArviZ — LOO-PIT** docs — https://python.arviz.org/en/latest/api/generated/arviz.plot_loo_pit.html — calibration via `az.plot_loo_pit`; the ∪/∩/tilt interpretation.
- **EABM (ArviZ) — Prior and Posterior predictive checks** — https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html — modern PPC + LOO-PIT + Bayesian p-value treatment.

**Books to keep open:** McElreath, *Statistical Rethinking* 2e (the intuition backbone — Ch 11 counts, Ch 13–14 hierarchy, Ch 15 missing data/measurement error); Gelman, Hill & Vehtari, *Regression and Other Stories* (free PDF, https://avehtari.github.io/ROS-Examples/) for the divide-by-4 rule, logistic interpretation, and Poisson overdispersion; BDA3 (Gelman et al. 2013) Ch 6 for posterior predictive checking and Ch 18 for missing data.

---

## ➡️ What's next

You have now run the full Bayesian workflow three times, on three different problem shapes, with every step visible and every judgment call narrated. That is the capstone: proof that you can take a question and a messy dataset and produce a model you can defend — frame, prior-check, fit, diagnose, posterior-check, compare, decide.

The final stop is the **Appendix — Cheatsheets & Resources** (`appendix-cheatsheets-and-resources.md`): the distribution cheat-sheet, the prior cheat-sheet (parameter → recommended prior → why), the diagnostic-thresholds table ($\hat R \le 1.01$, ESS $\ge 400$, $\hat k < 0.7$, BFMI $> 0.3$, 0 divergences) all in one place, the troubleshooting table, a glossary, the master resource list, and a "what to learn next" roadmap. Keep it pinned next to your editor — it is the quick-reference distillation of everything these eighteen chapters built. Go run the loop on your own data; that is the only exercise that finishes the course.

