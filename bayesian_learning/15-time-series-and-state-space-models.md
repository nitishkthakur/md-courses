# Chapter 15 — Time Series & State-Space Models

### Modeling dependence over time the Bayesian way — and why honest forecasts must fan out

> *A note from me to you.*
>
> *Every model you have built so far has quietly assumed that your data points are exchangeable — that the order of the rows doesn't matter, that observation 5 carries no information about observation 6 beyond what the shared parameters already say. For most of this course that assumption was fine, even liberating. But the moment your index is **time**, it is a lie, and a dangerous one. Yesterday's stock price tells you a great deal about today's. This month's airline traffic is built on last month's plus a seasonal wobble plus a slow upward drift. The temperature now is not a fresh independent draw — it is mostly the temperature an hour ago. Dependence over time is not noise to be averaged away; it is the **signal**, and a model that ignores it will be confidently, expensively wrong.*
>
> *Here is the second, deeper reason this chapter matters. Classical time-series courses hand you point forecasts — a single line marching off into the future — and maybe a confidence band bolted on by an asymptotic formula nobody trusts. The Bayesian view makes the one thing that actually matters about a forecast **explicit**: as you predict further ahead, you know less, and the model must **say so**. A correct probabilistic forecast is a cone that widens into the future, not a laser pointer. When a stakeholder asks "what will sales be in Q4?" the honest answer is a distribution, and the width of that distribution — the part naive methods hide — is exactly the part that should drive the decision. By the end of this chapter you will build forecasts that fan out for the right reasons, and you'll be able to point at the band and say precisely which sources of uncertainty it contains.*
>
> *We'll build up from the two atoms of Bayesian time series — the autoregression and the random walk — to the interpretable workhorse (additive structural models: trend + seasonality + noise), to the finance classic (stochastic volatility), to change-points, and finally to the state-space / Kalman view that ties it all together. Every model is generative; every forecast carries its uncertainty. Let's give your golem a memory.*

---

## What you'll be able to do after this chapter

- **Explain why time-ordered data breaks exchangeability**, and choose a dependence structure (AR, random walk, structural, state-space) that matches the mechanism you believe generated the series.
- **Build and fit autoregressive models** $\mathrm{AR}(p)$ in PyMC v5 with `pm.AR`, set priors that *enforce or relax* stationarity ($|\rho|<1$), and recover the true coefficients from synthetic data.
- **Use the Gaussian random walk** (`pm.GaussianRandomWalk`) as a local-level / time-varying mean, and understand it as the $\rho=1$ boundary where stationarity is deliberately abandoned.
- **Assemble a structural time-series model additively** — trend + seasonal + noise as separate, interpretable components you can read off the posterior — the single most useful time-series model in applied practice.
- **Fit the stochastic-volatility model** (latent log-volatility random walk driving a heavy-tailed return) and read volatility clustering straight off the posterior.
- **Fit a change-point model** (coal-mining disasters: a discrete switchpoint between two Poisson rates) and understand why PyMC quietly switches step methods to handle the discrete parameter.
- **State the state-space / Kalman view** (latent state equation + observation equation) and know when to reach for `pymc_extras.statespace` instead of writing the latent path by hand.
- **Forecast with honest, widening uncertainty** — using the `pm.Data` swap when the future depends on fixed inputs, and **forward-simulating the latent recursion** (or `statespace.forecast`) when it extends a free random-walk/AR state — and explain exactly which uncertainties the band contains.

I assume you are fluent with everything through **Chapter 08 (The Bayesian Workflow)**: you write `pm.Model` blocks with named `coords`, set standardized priors (the standardization-by-default habit from **Chapter 03**), run `pm.sample`, and gate on the diagnostics from **Chapter 07** ($\hat R \le 1.01$, ESS$_\text{bulk}$ & ESS$_\text{tail} \gtrsim 400$, zero divergences). Those gates apply to every model here too — time-series models are *notorious* for funnels and slow mixing, so the non-centered reparameterization reflex from **Chapter 07 / 10** will earn its keep. We also lean on the Gaussian-process view from **Chapter 12 (Gaussian Processes)**: a structural time series and a GP with a periodic-plus-trend kernel are two windows onto the same idea.

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

---

## 1. Why time breaks everything you've been assuming

Let me make the problem concrete before any model. Suppose you have a series $y_1, y_2, \dots, y_T$ and you fit it as if the points were independent and identically distributed — say $y_t \sim \mathrm{Normal}(\mu, \sigma)$. The likelihood factorizes:

$$
p(y_{1:T} \mid \mu, \sigma) \;=\; \prod_{t=1}^{T} \mathrm{Normal}(y_t \mid \mu, \sigma).
$$

That product is a statement of faith: *given the parameters, knowing $y_{t-1}$ tells you nothing about $y_t$.* For a time series this is almost always false. Run that i.i.d. model on a series with an upward trend and it will report a $\mu$ somewhere in the middle, a $\sigma$ inflated to cover the whole sweep, and — fatally — a forecast that is a **flat horizontal line** at $\mu$ with a band that does **not** widen into the future. The model has no concept of "later," so it cannot express "I'm less sure about later."

> 💡 **Intuition:** The defining feature of a time series is that the joint distribution does *not* factorize into independent marginals. Instead we factorize it along the **arrow of time** using the chain rule of probability:
> $$
> p(y_{1:T}) \;=\; p(y_1)\,\prod_{t=2}^{T} p(y_t \mid y_{1:t-1}).
> $$
> Every time-series model in this chapter is a different, tractable choice for the conditional $p(y_t \mid y_{1:t-1})$ — that is, a different *theory of memory*. AR says "the past $p$ values matter, linearly." A random walk says "only the immediately previous *state* matters." A structural model says "the world is a slowly drifting trend plus a repeating season plus noise." Pick the memory that matches your mechanism.

The Bayesian machinery does not change at all. We still write a generative model, put priors on its parameters, condition on the data, and sample the posterior. What changes is the *structure* of the likelihood — it now has temporal dependence baked in — and, crucially, what a **forecast** means. A forecast is just the posterior predictive distribution $p(\tilde y_{T+1:T+h} \mid y_{1:T})$ for future times, and because each future value depends on the (uncertain) previous one, prediction error **compounds**: the further out you go, the wider the predictive distribution becomes. That widening is the headline feature, and we will engineer it on purpose.

> ⚠️ **Pitfall:** The single most common time-series mistake — across *every* framework, not just Bayesian — is reporting a point forecast (the posterior predictive **mean**) and dropping the spread. The mean of a forecast that fans from $\pm 2$ units next month to $\pm 20$ units next year looks reassuringly like a tidy line. Shipping that line without its band is how teams get blindsided. In this chapter the band is never optional.

A note on **stationarity**, the concept that organizes the whole zoo. A series is (weakly) stationary if its mean, variance, and autocovariance don't depend on $t$ — statistically, the future looks like the past. Stationary models (AR with $|\rho|<1$) **mean-revert**: shocks decay, forecasts return to a long-run level, and the predictive band converges to a finite width. Non-stationary models (the random walk) **never forget**: shocks are permanent, forecasts have no anchor, and the band grows without bound. Neither is "right" — they encode different beliefs about how the world remembers. Knowing which you've assumed is the difference between a forecast that calms down and one that explodes, and you should *choose* it, not stumble into it.

---

## 2. The autoregressive model: memory as a weighted sum of the past

The autoregressive model of order $p$, written $\mathrm{AR}(p)$, is the most direct way to say "the present is a linear combination of the recent past plus a shock." Its generative recipe:

$$
y_t \;=\; \rho_0 + \rho_1\,y_{t-1} + \rho_2\,y_{t-2} + \cdots + \rho_p\,y_{t-p} + \varepsilon_t,
\qquad \varepsilon_t \sim \mathrm{Normal}(0, \sigma).
$$

In ASCII, the AR(2) case:

```
y_t = rho0 + rho1*y_{t-1} + rho2*y_{t-2} + eps_t,   eps_t ~ Normal(0, sigma)
```

Here $\rho_0$ is an optional **intercept** (a constant), $\rho_1, \dots, \rho_p$ are the **AR coefficients** (how strongly each lag pulls), and $\sigma$ is the **innovation** (shock) standard deviation. The whole behavior of the process is dictated by the $\rho$'s.

### 2.1 The AR(1) intuition and the stationarity condition

Start with the simplest case, $p=1$, dropping the intercept:

$$
y_t = \rho\, y_{t-1} + \varepsilon_t.
$$

Unroll it: $y_t = \rho^t y_0 + \sum_{k=0}^{t-1}\rho^k \varepsilon_{t-k}$. The effect of an old shock $\varepsilon_{t-k}$ is multiplied by $\rho^k$. Three regimes:

- **$|\rho| < 1$:** old shocks decay geometrically. The process is **stationary** and mean-reverting. It has a finite long-run variance.
- **$\rho = 1$:** every shock is remembered *forever* with full weight. This is the **Gaussian random walk** (§3) — the boundary of stationarity.
- **$|\rho| > 1$:** shocks are *amplified* over time. The process **explodes**; this is almost never what you want.

> 🧮 **The math:** For a stationary AR(1) with $|\rho|<1$, take variances of $y_t = \rho y_{t-1} + \varepsilon_t$. In the stationary regime $\mathrm{Var}(y_t)=\mathrm{Var}(y_{t-1})=:v$, and since $\varepsilon_t$ is independent of $y_{t-1}$,
> $$
> v = \rho^2 v + \sigma^2 \quad\Longrightarrow\quad v = \frac{\sigma^2}{1 - \rho^2}.
> $$
> Read this formula. As $\rho \to 1$, the denominator $\to 0$ and the variance **blows up** — the process loses its anchor. As $\rho \to 0$, $v \to \sigma^2$ and you're back to i.i.d. noise. The lag-$k$ autocorrelation is $\rho^k$: a clean geometric decay you can eyeball in an ACF plot. This single equation is the whole personality of an AR(1).

So the prior you put on $\rho$ is not cosmetic — it encodes whether you believe the world mean-reverts. If you want to **enforce stationarity**, keep $\rho$ inside $(-1,1)$; if you want to **relax** it (allow near-unit-root or explosive behavior because the science demands it), widen the prior. A `Normal(0, 0.5)` prior on $\rho$ puts ~95% of its mass in roughly $(-1, 1)$ and gently discourages the explosive region without hard-banning it — a sensible weakly-informative default. If you truly need a hard constraint, transform a $\mathrm{Beta}$ or use a bounded prior, but in practice the soft prior plus real data is almost always enough.

### 2.2 `pm.AR`: the built-in, and the exact argument semantics

PyMC ships the AR process as a first-class distribution, `pm.AR`. Its signature has a few semantics worth pinning down precisely, because getting them wrong is the classic AR footgun:

```python
# pm.AR(name, rho, *, sigma=1.0, init_dist=None, constant=False, ar_order=None, steps=None, ...)
```

- **`rho`** — the coefficient tensor. The last-dimension entry $n$ is the lag-$n$ coefficient.
- **`constant`** — if `True`, the **first** element of `rho` is the intercept $\rho_0$, so `rho` has length `ar_order + 1` (the gallery AR(2) example uses a length-3 `rho` with `constant=True`). The remaining entries are the lag coefficients in order: `rho[1]` is lag-1, `rho[2]` is lag-2, and so on.
- **`ar_order`** — the order $p$. Set it explicitly; don't rely on inference from shapes.
- **`init_dist`** — an **unnamed** distribution from the `.dist()` API giving the prior for the first `ar_order` values. The default is a *very* wide `Normal(0, 100)` — almost always override it, or your initial conditions will let the chain wander.

Here is an AR(2) with a constant, written the way you should write it — explicit order, explicit init:

```python
with pm.Model() as ar2_model:
    # rho[0] = intercept (constant=True), rho[1] = lag-1, rho[2] = lag-2
    rho = pm.Normal("rho", mu=0.0, sigma=0.5, shape=3)
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    init = pm.Normal.dist(0.0, 1.0, shape=2)          # prior for the first 2 values
    y_ar = pm.AR(
        "y_ar", rho, sigma=sigma, init_dist=init,
        constant=True, ar_order=2, observed=y_data,
    )
    idata_ar = pm.sample(
        1000, tune=1000, chains=4, target_accept=0.9, random_seed=RANDOM_SEED
    )
```

> 🔧 **In practice:** Three habits keep `pm.AR` healthy. **(1)** Always pass a sane `init_dist` — `pm.Normal.dist(0, 1)` if your series is standardized, scaled to your series otherwise — never the wide default. **(2)** Standardize your series first (subtract mean, divide by SD); then `Normal(0, 0.5)` on the coefficients and `HalfNormal(1)` on `sigma` are calibrated to that scale, exactly as in **Chapter 03**. **(3)** Give the order explicitly with `ar_order=`. The shape of `rho` is `ar_order + constant`; mismatches here are the #1 `pm.AR` error.

### 2.3 Worked: simulate an AR(2) and recover it

Nothing teaches a model like watching it recover a truth you planted. Let's simulate a genuine AR(2), fit it, and check the posterior covers the true coefficients — the synthetic-data-with-known-truth pattern the bible recommends.

```python
# --- Simulate an AR(2) with known coefficients ---
true_rho = np.array([0.5, -0.3])     # lag-1 = 0.5, lag-2 = -0.3 (stationary)
true_sigma = 1.0
T = 300
y_data = np.zeros(T)
eps = rng.normal(0, true_sigma, size=T)
for t in range(2, T):
    y_data[t] = true_rho[0] * y_data[t-1] + true_rho[1] * y_data[t-2] + eps[t]

# (Then fit the ar2_model from §2.2, but with constant=False and shape=2 on rho.)
with pm.Model() as ar2_recover:
    rho = pm.Normal("rho", 0.0, 0.5, shape=2)          # two lags, no intercept
    sigma = pm.HalfNormal("sigma", 1.0)
    init = pm.Normal.dist(0.0, 1.0, shape=2)
    pm.AR("y", rho, sigma=sigma, init_dist=init,
          constant=False, ar_order=2, observed=y_data)
    idata_ar = pm.sample(1000, tune=1000, chains=4,
                         target_accept=0.9, random_seed=RANDOM_SEED)

az.summary(idata_ar, var_names=["rho", "sigma"])
```

When you run `az.summary`, you should see something like `rho[0] ≈ 0.50`, `rho[1] ≈ -0.30`, `sigma ≈ 1.0`, each with a 94% HDI comfortably bracketing the truth, `r_hat` of 1.00, and ESS in the thousands. That is the model passing its calibration test: **fed data it itself could have generated, it recovers the generator.**

> 🩺 **Diagnostic:** For AR models, before anything else plot the **ACF of the residuals** (`az.plot_autocorr` on the posterior-mean residuals, or just `np.correlate`). If the AR order is right, residual autocorrelation should be flat — all the temporal structure has been absorbed into the $\rho$'s. Leftover autocorrelation at lag $k$ is a sign you need a higher order, a seasonal term, or a different structure entirely. Also eyeball the posterior of $\rho$: if its mass crowds against $\pm 1$, your series may have a unit root and an AR model is fighting a random walk — switch to §3.

---

## 3. The Gaussian random walk: a time-varying mean that never forgets

The autoregression assumes a *fixed* relationship to a stable past. But often the thing that's drifting is the **level itself** — the local mean of the series wanders, and observations are noisy readings around that wandering mean. For that, the right atom is the **Gaussian random walk** (GRW):

$$
x_t \;=\; x_{t-1} + \eta_t, \qquad \eta_t \sim \mathrm{Normal}(0, \sigma).
$$

In ASCII: `x_t = x_{t-1} + eta_t,  eta_t ~ Normal(0, sigma)`. Each step is the previous value plus an independent Gaussian nudge. This is exactly the AR(1) with $\rho = 1$ — the boundary case from §2.1 — and it behaves completely differently from a stationary AR:

- **It is non-stationary.** $\mathrm{Var}(x_t) = t\,\sigma^2$ grows linearly with time. There is no long-run mean to revert to; the process has *permanent memory*.
- **Shocks are forever.** A single large $\eta_t$ shifts the entire future path up by that amount. Nothing pulls it back.
- **It is the maximally flexible smooth trend.** With a *small* $\sigma$, a GRW is a slowly-varying curve that can adapt to whatever shape the data demand — without you specifying a functional form. That's its superpower as a building block.

> 💡 **Intuition:** Think of the GRW as a "local level" — the model's running estimate of "where the series is right now." It is the Bayesian answer to "fit a trend, but don't tell me the trend's shape in advance." The prior on the step size $\sigma$ is the **smoothness knob**: small $\sigma$ ⇒ a stiff, almost-straight level; large $\sigma$ ⇒ a wiggly level that chases every bump (and overfits). Choosing $\sigma$'s prior is choosing how much you'll let the level move from one step to the next. This is the same bias–variance tension you met with the GP lengthscale in **Chapter 12** — here it lives in a single scale parameter.

### 3.1 `pm.GaussianRandomWalk` and the v5 API

The modern signature:

```python
# pm.GaussianRandomWalk(name, mu=0, sigma=1, *, init_dist=None, steps=None)
```

Two API notes that trip people up, straight from the dossier's "things that changed" list:

- **`sigma`, not `sd`.** The v3 `sd=` keyword is **gone**. Using it raises `TypeError`.
- **`init_dist`, not `init`.** The starting value $x_0$ is given by an **unnamed** `.dist()`, e.g. `init_dist=pm.Normal.dist(0, 1)`. The old `init=` keyword is gone.
- **`steps` *or* a `dims`/`shape`, not both inconsistently.** Either give `steps=len(y)-1` (the GRW has one more value than it has steps) or attach `dims=` and let the coord set the length — but don't supply contradictory lengths.

```python
with pm.Model(coords={"time": np.arange(T)}) as rw_model:
    sigma = pm.Exponential("sigma", 1.0)                 # step-size / smoothness
    level = pm.GaussianRandomWalk(
        "level", sigma=sigma,
        init_dist=pm.Normal.dist(0, 1),
        dims="time",
    )
    obs_sigma = pm.HalfNormal("obs_sigma", 1.0)
    pm.Normal("y", mu=level, sigma=obs_sigma, observed=y_data, dims="time")
    idata_rw = pm.sample(1000, tune=1000, chains=4,
                         target_accept=0.95, random_seed=RANDOM_SEED)
```

This is the **local-level model**: a latent random-walk level `level`, observed through Gaussian measurement noise `obs_sigma`. Notice we now have *two* variances doing different jobs — the **process** noise $\sigma$ (how fast the true level drifts) and the **observation** noise `obs_sigma` (how noisily we measure it). Separating "the world changed" from "our measurement was noisy" is the whole point of state-space thinking (§7), and you can already see it here in two lines.

> ⚠️ **Pitfall:** A GRW with $T$ latent states is a $T$-dimensional correlated parameter — a long, thin ridge in posterior space, prone to **funnels** exactly like the hierarchical models of **Chapter 10**. Symptoms: divergences, low ESS, a `sigma` that won't mix. The cures are the same: raise `target_accept` to 0.95–0.99, and if that's not enough, **reparameterize non-centered**. The non-centered GRW writes the innovations as standard normals and accumulates them:
> ```python
> # non-centered GRW (the funnel cure for random walks)
> z = pm.Normal("z", 0, 1, dims="time")               # standardized innovations
> level = pm.Deterministic("level",
>             z.cumsum() * sigma, dims="time")          # scale, then accumulate
> ```
> Same terminology, same move as Eight Schools — separate the scale $\sigma$ from the shape $z$ so the geometry stops pinching. Reach for this the instant a random-walk model misbehaves.

### 3.2 Reading the local level

After sampling, the latent `level` is a full posterior *path* — at each time $t$ you have a distribution over where the true level was. Plotting its posterior mean with an HDI band gives you a **smoothed** version of your noisy series, with honest uncertainty about the smoothing:

```python
# pseudocode for the plot you'll run
post_level = idata_rw.posterior["level"]               # (chain, draw, time)
mean_level = post_level.mean(dim=["chain", "draw"])
hdi_level  = az.hdi(idata_rw, var_names=["level"])["level"]   # Dataset → DataArray, dim 'hdi'
# lo, hi = hdi_level.sel(hdi="lower"), hdi_level.sel(hdi="higher")   # NOT .lo/.hi
# plt.plot(y_data, ".", alpha=.4); plt.plot(mean_level)
# plt.fill_between(np.arange(T), lo, hi, alpha=.3)
```

The band is *narrow in the middle of the data* (lots of neighbors to pin the level down) and **widens at the edges**, especially as you extrapolate past the last observation — there, the only thing holding the level is the prior on $\sigma$, so the GRW fans out at rate $\sqrt{h}\,\sigma$ for a forecast horizon $h$. That fanning is not a bug; it is the random walk being honest that it has no idea where the level is going.

---

## 4. Structural time series: trend + seasonality + noise, built by addition

Now we reach the workhorse — the model I reach for first on almost any real business or scientific series, and the one you should master cold. The idea is gloriously simple and the bible's §7 catalog points us straight at it: **decompose the series into interpretable components and add them up.**

$$
y_t \;=\; \underbrace{\mu_t}_{\text{trend / level}} \;+\; \underbrace{s_t}_{\text{seasonality}} \;+\; \underbrace{\varepsilon_t}_{\text{noise}}.
$$

In ASCII: `y_t = trend_t + seasonal_t + noise_t`. Each piece is a small generative model you understand on its own, and because they add, the posterior hands you each component *separately* — you can plot "the trend," "the seasonal pattern," and "the residual noise" as three distinct, interpretable curves. This decomposability is why structural models dominate applied forecasting (it's the engine behind Facebook's Prophet, BSTS, and `statsmodels`' UnobservedComponents): a stakeholder can *look* at the trend and the season and nod. A black box can forecast as well, but it can't be argued with.

### 4.1 The components, one at a time

**The trend / level.** Use a Gaussian random walk for a wandering local level (§3), or a **local linear trend** when you believe the series has momentum — a *slope* that itself drifts:

$$
\mu_t = \mu_{t-1} + \delta_{t-1} + \eta_t, \qquad \delta_t = \delta_{t-1} + \zeta_t.
$$

Here $\mu_t$ is the level and $\delta_t$ is a slowly-changing slope; $\eta_t$ and $\zeta_t$ are the level and slope innovations. With $\zeta = 0$ the slope is constant (a straight line); let it wander and the trend can bend. This two-state local-linear-trend is the standard structural trend.

**The seasonality.** A repeating pattern with period $L$ (12 for monthly-with-yearly-season, 7 for daily-with-weekly-season). Two ways to encode it:

1. **Dummy / time-domain seasonality:** one effect per season-of-cycle, $s_t^{(1)}, \dots, s_t^{(L)}$, constrained to **sum to zero** over a full cycle (so it doesn't fight the trend for the overall level). The sum-to-zero constraint is the same identifiability fix you used for categorical effects in **Chapter 09** — here you can impose it with a `ZeroSumNormal` or by construction.
2. **Fourier / frequency-domain seasonality:** approximate the seasonal shape with a few sines and cosines, $s_t = \sum_{k=1}^{K} \big[a_k \sin(2\pi k t/L) + b_k \cos(2\pi k t/L)\big]$. A handful of harmonics ($K$ small) gives a smooth season with far fewer parameters than $L$ dummies — the GP-flavored, parsimonious choice, and what Prophet uses.

**The noise.** The leftover, $\varepsilon_t \sim \mathrm{Normal}(0, \sigma_\text{obs})$ — or `StudentT` if you expect outliers (the robust move from **Chapter 09**). This is the irreducible observation error after trend and season are accounted for.

> 🔧 **In practice:** Start additive and *simple*: random-walk level + a small Fourier season + Normal noise. Add a drifting slope only if the trend visibly bends; add more harmonics only if the seasonal residuals show structure. Each component you add is another variance parameter the data must support — and structural models are famous for **only-weakly-identified variances** (the level noise and observation noise can trade off; see the pitfall below). Resist the urge to throw every component in at once. Build the owl one feather at a time.

> ⚠️ **Pitfall — the variance trade-off.** In a local-level-plus-noise model, the process variance $\sigma^2_\text{level}$ and the observation variance $\sigma^2_\text{obs}$ are only **jointly** identified: a wiggly true level read cleanly looks much like a flat level read noisily. The likelihood has a long ridge connecting them. If your sampler crawls and these two won't separate, that's why. Cures: an informative prior on one of them (often you *do* know your measurement precision), tighter `Exponential`/`HalfNormal` scales, more data, or the non-centered trend. Don't be surprised when these two variances are correlated in the posterior — pair-plot them and you'll see the ridge.

### 4.2 Building it in PyMC by hand (the additive recipe)

Here is a complete structural model — random-walk level + Fourier seasonality + Normal noise — written from scratch so you see every component. We'll deploy it on a real dataset in §6, but study the skeleton first:

```python
# pseudocode-flavored skeleton; the full runnable version is in §6
def fourier_features(t, period, n_harmonics):
    """Design matrix of sin/cos pairs for seasonality of given period."""
    k = np.arange(1, n_harmonics + 1)
    arg = 2 * np.pi * np.outer(t, k) / period        # (T, K)
    return np.concatenate([np.sin(arg), np.cos(arg)], axis=1)  # (T, 2K)

with pm.Model(coords={"time": time_idx, "fourier": fourier_names}) as sts:
    # --- Trend: a Gaussian random-walk level (non-centered for safety) ---
    sigma_level = pm.Exponential("sigma_level", 1.0)
    z = pm.Normal("z", 0, 1, dims="time")
    level = pm.Deterministic("level", z.cumsum() * sigma_level, dims="time")

    # --- Seasonality: a few Fourier coefficients (regularized to be smooth) ---
    beta_f = pm.Normal("beta_f", 0, 1, dims="fourier")
    seasonal = pm.Deterministic("seasonal", X_fourier @ beta_f, dims="time")

    # --- Noise + likelihood ---
    sigma_obs = pm.HalfNormal("sigma_obs", 1.0)
    mu = pm.Deterministic("mu", level + seasonal, dims="time")
    pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y_std, dims="time")
```

Read the structure: `mu = level + seasonal` is literally the additive decomposition, and because we wrapped each piece in `pm.Deterministic`, the InferenceData will contain posterior paths for `level`, `seasonal`, and `mu` separately. You can plot the trend without the season, the season without the trend — exactly the interpretability that makes structural models worth their weight.

> 💡 **Intuition:** A structural time series and a **Gaussian process** (Chapter 12) are two dialects of the same sentence. A GP with a *linear-plus-periodic* kernel produces trend-plus-seasonality too; the difference is bookkeeping. The structural model writes the components explicitly as states (great for interpretability and for the Kalman filter of §7); the GP writes them implicitly as a covariance function (great for smoothness control and uncertainty). When someone asks "should I use a GP or a structural model for this series?", the honest answer is usually "either — pick the one whose *knobs* you'd rather tune."

### 4.3 The `pymc_extras.statespace` shortcut (and when to use it)

Writing the level recursion and seasonality by hand, as above, is the right way to *learn* — you see every moving part. But for production structural models there is a faster, numerically superior path: the **`pymc_extras.statespace`** module (formerly in `pymc_experimental`). It builds the model as a proper state-space system and runs a **Kalman filter** under the hood, which marginalizes the latent states analytically — so the sampler explores a small, well-conditioned parameter space instead of a $T$-dimensional latent path. That is dramatically better geometry (§7 explains *why*).

You assemble components, **add** them, **build**, and supply priors:

```python
# pseudocode — pymc_extras.statespace APIs move between releases; see the callout below
from pymc_extras.statespace import structural as st

trend    = st.LevelTrendComponent(order=2, innovations_order=1)   # local linear trend
seasonal = st.TimeSeasonality(season_length=12, innovations=True) # yearly season
error    = st.MeasurementError()
ss_mod   = (trend + seasonal + error).build(name="airline_sts")

with pm.Model(coords=ss_mod.coords) as m:
    # YOU supply priors for the parameter names the components requested.
    # ss_mod prints exactly which names/dims it needs — that's the main "gotcha".
    # ... pm.Normal / pm.HalfNormal for each requested parameter ...
    ss_mod.build_statespace_graph(data=y)               # wires the Kalman filter into logp
    idata = pm.sample(nuts_sampler="nutpie", random_seed=RANDOM_SEED)

forecasts = ss_mod.forecast(idata, start=-1, periods=24, filter_output="smoothed")
```

> 🔧 **In practice:** The component model declares which parameters it needs and asks *you* to place the priors — it prints the required names and dims. You never write the Kalman recursion; you write priors, call `build_statespace_graph(data=...)` **inside** the `pm.Model` context, sample, and then use `.forecast(...)` and `.sample_conditional_posterior(...)` for smoothed/filtered states. I'll do the worked example in §6 with the hand-built model (always runnable) and point you to statespace as the production upgrade.

> 🛠️ **One verification note (applies to every `pymc_extras.statespace` block in this chapter).** This module is younger and faster-moving than core PyMC: the exact class names (`LevelTrendComponent`, `TimeSeasonality`, `FrequencySeasonality`, `CycleComponent`, `AutoregressiveComponent`, `MeasurementError`, `RegressionComponent`), the `from pymc_extras.statespace import structural as st` convenience alias, and the `.build()` / `.build_statespace_graph()` / `.forecast()` signatures can drift between releases. Every statespace code block here is marked `# pseudocode` for that reason — **check the current `pymc_extras` structural-components docs before pasting.** The hand-built models (§2, §3, §5, §6) use only stable core-PyMC APIs and are meant to run as-is; the statespace path is the production upgrade you reach for once you've confirmed the spellings.

---

## 5. Stochastic volatility: when the *variance* has a memory

So far the thing drifting over time has been the **mean**. In finance — and anywhere the *spread* of your data clusters in time — it's the **variance** that drifts. Look at any daily-returns series and you'll see it: long calm stretches, then a violent cluster of big moves (a crash, a crisis), then calm again. This is **volatility clustering**, and a constant-$\sigma$ model is blind to it. It reports one average volatility, badly underpredicting risk during turbulent periods and overpredicting it during calm ones — a potentially ruinous miscalibration.

The **stochastic volatility** (SV) model — a genuine PyMC classic — fixes this by letting the (log) standard deviation itself be a latent random walk:

$$
\log\sigma_t \;=\; \log\sigma_{t-1} + \text{(step)}, \qquad
y_t \;\sim\; \mathrm{StudentT}\!\big(\nu,\; 0,\; \exp(\log\sigma_t)\big).
$$

Two design choices to savor. First, the latent **log-volatility is a Gaussian random walk** (§3) — so the volatility wanders smoothly and persistently, which is exactly the clustering we see. We model it on the *log* scale so $\sigma_t = \exp(\cdot)$ is automatically positive. Second, the returns are **Student-$t$**, not Normal: even after we let the scale vary, real returns have fatter tails than Gaussian, and $\nu$ (degrees of freedom) absorbs that excess kurtosis — the robust-likelihood idea from **Chapter 09**, here doing double duty.

### 5.1 The canonical PyMC SV model

This is essentially the model in PyMC's own case study (S&P 500 daily returns). Note the `lam` parametrization: PyMC's `StudentT` takes precision `lam = 1/scale²`, so a scale of $\exp(\text{vol})$ means $\lambda = \exp(-2\cdot\text{vol})$.

```python
with pm.Model(coords={"time": returns.index}) as sv_model:
    step_size = pm.Exponential("step_size", 10.0)          # GRW step for log-vol
    volatility = pm.GaussianRandomWalk(
        "volatility", sigma=step_size,
        init_dist=pm.Normal.dist(0, 100),                  # deliberately diffuse first log-vol state (see note)
        dims="time",
    )
    nu = pm.Exponential("nu", 0.1)                          # Student-t tail thickness
    returns_obs = pm.StudentT(
        "returns", nu=nu,
        lam=pm.math.exp(-2 * volatility),                  # lam = 1/scale^2, scale = exp(vol)
        observed=returns["change"], dims="time",
    )
    idata_sv = pm.sample(
        1000, tune=1000, target_accept=0.9,
        nuts_sampler="numpyro",                            # this model is big & continuous → JAX backend shines
        random_seed=RANDOM_SEED,
    )
```

> ⚠️ **Pitfall — the one wide `init_dist` we tolerate.** Everywhere else this chapter preaches "never use the diffuse default init; tighten it to a sane scale." Here the canonical case study sets `init_dist=pm.Normal.dist(0, 100)` on the *first* log-volatility state, and — be honest about this — a std of 100 on the **log** scale is enormous on the natural scale ($\exp(100)$ is astronomically large), so this is *not* "safe because it's on the log scale"; the log scale makes it *more* extreme, not less. It samples fine anyway for one specific reason: it's a prior on a single state at $t=0$, and with thousands of daily returns the GRW recursion plus the likelihood overwhelm that one diffuse starting point almost immediately. This is the exception that proves the rule — tolerable only because the series is long. On a short series, tighten it (e.g. `pm.Normal.dist(0, 5)`), exactly as you would anywhere else.

> 🔧 **In practice:** The SV model has one latent state per day — thousands of correlated parameters. It is slow and funnel-prone under the default sampler. Two moves help enormously: **(1)** raise `target_accept` to 0.9+; **(2)** use a JAX backend (`nuts_sampler="numpyro"` or `"nutpie"`), which vectorizes the latent path and is often 5–20× faster on big continuous models like this — a preview of **Chapter 17 (Scaling)**. The model is *fully continuous* (no discrete RVs), so the JAX backends apply cleanly. If you see funnels in `step_size`, non-center the volatility random walk exactly as in §3.1.

### 5.2 Reading volatility clustering off the posterior

The payoff plot: transform the latent log-volatility back to the data scale, $\exp(\text{volatility})$, and overlay its posterior band on the absolute returns.

```python
# pseudocode for the money plot
exp_vol  = np.exp(idata_sv.posterior["volatility"])         # back to return-scale sigma
vol_mean = exp_vol.mean(dim=["chain", "draw"])
vol_hdi  = az.hdi(exp_vol)["volatility"]                    # DataArray with an 'hdi' dim
# lo, hi = vol_hdi.sel(hdi="lower"), vol_hdi.sel(hdi="higher")   # NOT .lo/.hi
# plt.plot(np.abs(returns["change"]), ".", alpha=.3)        # |returns| as a volatility proxy
# plt.plot(vol_mean); plt.fill_between(returns.index, lo, hi, alpha=.3)
```

> 🩺 **Diagnostic:** On an equities series, the posterior $\exp(\text{volatility})$ should be **low and flat in calm years and spike sharply during crises** (the 2008 crash, March 2020). If your data span those periods, the spikes are unmistakable and the band is tight precisely where there's lots of large-move data to pin the volatility down. This is the model *seeing* the clustering that a constant-$\sigma$ Normal could never represent. Sanity check `nu`: a posterior for $\nu$ well below ~30 confirms the tails are genuinely heavier than Gaussian even after the time-varying scale — i.e., you needed both the random-walk volatility *and* the Student-$t$ tails.

> 📜 **Citation/Origin:** Stochastic volatility models trace to Taylor (1982/1986) and the econometrics of Jacquier, Polson & Rossi (1994); the GRW-log-vol-plus-Student-$t$ formulation is the standard Bayesian treatment. The PyMC implementation here is adapted from the official "Stochastic Volatility" case study, which fits it on S&P 500 returns — copy that notebook to see the full crisis-spike plot.

---

## 6. Worked example: a structural model on airline passengers, with a forecast that fans out

Time to assemble the whole machine on a real, beloved dataset: the **monthly airline-passenger counts, 1949–1960** — the classic series with a clear upward trend *and* a strong yearly season whose amplitude grows over time. Seaborn ships it as `flights` (it *is* the AirPassengers series), so the load is trivial. We'll build trend + seasonality + noise, sample, check, and then forecast two years ahead with honest widening bands by forward-simulating the random-walk level from the posterior (§6.5 explains why that, and not a naive `pm.Data` swap, is the right move for a free latent path).

### 6.1 Load and look

```python
import seaborn as sns
flights = sns.load_dataset("flights")          # columns: year, month, passengers
flights["month_idx"] = (flights["year"] - 1949) * 12 + (flights["month"].cat.codes)
flights = flights.sort_values("month_idx").reset_index(drop=True)

y_raw = flights["passengers"].to_numpy().astype(float)
T = len(y_raw)                                  # 144 months
t = np.arange(T)
```

Plot `y_raw` against `t` and you'll see three things at once: a roughly linear **upward trend**, a repeating **12-month season** (summer peaks), and — importantly — the seasonal swings get *bigger* as the level rises. That last fact (multiplicative seasonality) is a hint: modeling `log(passengers)` turns the growing-amplitude season into a roughly *constant*-amplitude season on the log scale, which an additive model handles beautifully. This is a recurring trick — **work on the log scale when variation scales with level.**

```python
y_log = np.log(y_raw)
y_mean, y_std_sd = y_log.mean(), y_log.std()
y_std = (y_log - y_mean) / y_std_sd            # standardize (Chapter 03 default)
```

We standardize the (logged) series so our `Normal(0,1)`-flavored priors are calibrated, exactly the habit from **Chapter 03**. Everything downstream lives on the standardized-log scale; we'll invert both transforms at the very end to report passenger counts.

### 6.2 Build the additive structural model

```python
def fourier_features(time, period, n_harmonics):
    k = np.arange(1, n_harmonics + 1)
    arg = 2 * np.pi * np.outer(time, k) / period            # (len(time), K)
    return np.concatenate([np.sin(arg), np.cos(arg)], axis=1)  # (len(time), 2K)

N_HARM = 4
X_fourier_train = fourier_features(t, period=12, n_harmonics=N_HARM)
fourier_names = [f"{fn}{kk}" for fn in ["sin", "cos"] for kk in range(1, N_HARM + 1)]

coords = {"time": t, "fourier": fourier_names}
with pm.Model(coords=coords) as airline_sts:
    # Fourier design matrix as a data container (keeps the graph clean; the season is
    # a deterministic function of these fixed features and the learned coefficients).
    Xf = pm.Data("Xf", X_fourier_train, dims=("time", "fourier"))

    # --- Trend: non-centered Gaussian random-walk level ---
    sigma_level = pm.Exponential("sigma_level", 5.0)        # small step ⇒ smooth trend
    z = pm.Normal("z", 0, 1, dims="time")
    level = pm.Deterministic("level", z.cumsum() * sigma_level, dims="time")

    # --- Seasonality: Fourier coefficients (regularized smooth season) ---
    beta_f = pm.Normal("beta_f", 0, 0.5, dims="fourier")
    seasonal = pm.Deterministic("seasonal", Xf @ beta_f, dims="time")

    # --- Noise + likelihood ---
    sigma_obs = pm.HalfNormal("sigma_obs", 0.5)
    mu = pm.Deterministic("mu", level + seasonal, dims="time")
    pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y_std, dims="time")
```

Every modeling decision here is one we've justified: a **non-centered** random-walk level (the §3.1 funnel cure, used pre-emptively because random walks always risk funnels), a **Fourier** season with 4 harmonics (smooth and parsimonious — 8 coefficients instead of 12 dummies), regularizing `Normal(0, 0.5)` priors on the seasonal coefficients (standardized scale), and a `HalfNormal` observation noise. The latent `level`, `seasonal`, and `mu` are all wrapped in `pm.Deterministic`, so the posterior hands them back as separate paths — and `level`'s last value is the launch point we'll forward-simulate from in §6.5.

### 6.3 Prior predictive check — before you fit anything

The workflow gate from **Chapter 08**: simulate from the prior and make sure the model can produce series that *look like they could be airline data* (on the standardized-log scale).

```python
with airline_sts:
    prior_pred = pm.sample_prior_predictive(samples=200, random_seed=RANDOM_SEED)
# Plot a handful of prior-draw "y" series; they should be wandering, gently seasonal,
# and on roughly the right scale (a few standardized units) — NOT exploding to ±50.
```

> 🩺 **Diagnostic:** If prior-predictive draws blow up to absurd magnitudes, `sigma_level` is too loose (the random walk wanders off) or the Fourier priors are too wide (giant seasonal swings). Tighten them. If every prior draw is a near-flat line, you've over-regularized and the model can't trend or season — loosen. You want prior draws that *bracket* plausible series without asserting any particular one. This five-minute check catches mis-scaled priors before they waste an hour of sampling.

### 6.4 Sample and diagnose

```python
with airline_sts:
    idata_sts = pm.sample(
        1000, tune=1500, chains=4, target_accept=0.95,    # 0.95: random walks funnel
        random_seed=RANDOM_SEED,
    )

az.summary(idata_sts, var_names=["sigma_level", "sigma_obs", "beta_f"])
```

> 🩺 **Diagnostic:** Gate on the **Chapter 07** thresholds: $\hat R \le 1.01$ for every parameter, ESS$_\text{bulk}$ and ESS$_\text{tail} \gtrsim 400$, and **zero divergences**. For this model, watch `sigma_level` and `sigma_obs` specifically — they're the variance pair that can trade off (the §4.1 ridge). If you see divergences, an `az.plot_pair(idata_sts, var_names=["sigma_level", "sigma_obs"])` will often reveal a banana/ridge; raising `target_accept` to 0.99 or tightening one prior usually clears it. A rank plot (`az.plot_rank`) on `sigma_level` should look like uniform static, not stripes.

Now the **posterior predictive check** — does the fitted model reproduce the in-sample series?

```python
with airline_sts:
    pm.sample_posterior_predictive(idata_sts, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
az.plot_ppc(idata_sts, num_pp_samples=100)
```

You should see the posterior-predictive cloud hug the data, and — more tellingly for a time series — the posterior `level` should trace a smooth rising trend while `seasonal` traces a clean repeating wave. Plot those two `Deterministic`s separately; their interpretability is the whole reason we built it additively.

### 6.5 Forecast 24 months ahead — forward-simulate the random walk

Here is the move that produces an honest forecast. The temptation is to "just resize the `time` coord with `pm.set_data` and call `sample_posterior_predictive` again." **Do not do this with the hand-built model**, and it's worth understanding exactly why, because it's a trap an expert will walk into. Our latent `level` is built as `z.cumsum() * sigma_level`, and `z` is a **free random variable** stored in `idata_sts.posterior` with shape `(chain, draw, time=144)`. When you resize `time` to 168 and run posterior-predictive, PyMC conditions on the *stored* posterior draws of `z` — which are length 144 — against a graph that now expects length 168. You get a shape/broadcasting error, or (depending on version) PyMC re-draws the **entire** `z` fresh from its `Normal(0,1)` prior, **throwing away the fitted in-sample trend** and replacing it with a brand-new prior random walk. Either way you do not get "continue the fitted trend, then fan out." The `pm.Data` swap is the right tool for forecasting a model whose future depends only on *fixed inputs* (a regression on future covariates); it is the **wrong** tool when the thing you must extend is a *free latent path*.

The correct, always-runnable move is to **forward-simulate the random walk in numpy from the posterior draws**. We already have everything we need in `idata_sts`: posterior draws of `sigma_level` (the step size), `beta_f` (the seasonal coefficients), `sigma_obs` (observation noise), and the in-sample `level` path (whose *last* value is where the future trend starts). For each posterior draw we (1) take the last fitted level, (2) add fresh Gaussian innovations to walk it forward `H` steps, (3) add the Fourier season evaluated at the future times, and (4) add observation noise. The spread across draws *is* the forecast uncertainty, and it fans out because the future innovations accumulate.

```python
H = 24                                                       # forecast horizon: 2 years
t_future = np.arange(T, T + H)                               # future indices only
Xf_future = fourier_features(t_future, period=12, n_harmonics=N_HARM)  # (H, 2K)

# --- pull posterior draws, flattened over (chain, draw) ---
post = idata_sts.posterior.stack(sample=("chain", "draw"))
sigma_level = post["sigma_level"].values                     # (S,)
beta_f      = post["beta_f"].values.T                        # (S, 2K)
sigma_obs   = post["sigma_obs"].values                       # (S,)
last_level  = post["level"].values[-1, :]                    # (S,) last in-sample level per draw
S = sigma_level.shape[0]

# --- forward-simulate the level random walk H steps for every draw ---
rng_fc = np.random.default_rng(RANDOM_SEED)
innov = rng_fc.normal(size=(S, H)) * sigma_level[:, None]    # fresh innovations, scaled per draw
level_future = last_level[:, None] + np.cumsum(innov, axis=1)  # (S, H): RW continues from last level

# --- add seasonality and observation noise, all on the standardized-log scale ---
seasonal_future = beta_f @ Xf_future.T                       # (S, H)
mu_future   = level_future + seasonal_future                # (S, H)
eps         = rng_fc.normal(size=(S, H)) * sigma_obs[:, None]
y_future_std = mu_future + eps                               # (S, H) predictive draws
```

> 💡 **Why this is the honest forecast.** The level starts at each draw's *fitted* last value `last_level` (so we keep the trend we worked to estimate), then walks forward with fresh `Normal(0, sigma_level)` innovations. Beyond the data the level is governed only by `sigma_level`, so its variance grows like $h\,\sigma_\text{level}^2$ and the band fans at rate $\sqrt{h}\,\sigma_\text{level}$ — exactly the random walk admitting it doesn't know where the trend is headed. Parameter uncertainty enters because we loop over *every* posterior draw of `sigma_level`, `beta_f`, `sigma_obs` and `last_level`; process uncertainty enters through the fresh `innov` and `eps`. Both sources from §9 are present, by construction.

> 🔧 **In practice — the production shortcut.** Forward-simulating by hand is transparent and always runs, but for a real structural model you'd let the **state-space machinery** do it correctly for you. With `pymc_extras.statespace` (§4.3), a 24-month forecast is one line — `ss_mod.forecast(idata, start=-1, periods=24)` — and it propagates the random-walk level (and any trend/season states) through the Kalman recursion with the right covariance, no manual `cumsum` bookkeeping. That `.forecast` is built for exactly this; reach for it once you've confirmed the API (see the §4.3 verification callout). The hand-built forward-simulation above is your ground-truth check that you understand what `.forecast` is doing under the hood.

### 6.6 Plot the forecast on the original scale

Finally, invert standardization and the log to get back to passengers, and draw the cone. **One subtlety the careful reader must respect:** the back-transform `exp(·)` is *nonlinear*, so the mean of the exponentiated draws is **not** the exp of the mean — it's a biased estimate of the predictive mean on the count scale (it carries a $+\tfrac{1}{2}\sigma^2$ lognormal correction). Summarize the central forecast with the **posterior median** instead, which behaves cleanly because the median is preserved under any monotone transform. The HDI is likewise safe: quantiles are monotone-invariant, so the back-transformed interval is the correct interval.

```python
# back-transform is real; the plt.* lines are commented sketches of the figure
pred_log_future   = y_future_std * y_std_sd + y_mean         # undo standardization, (S, H)
pred_pass_future  = np.exp(pred_log_future)                  # undo log → counts, (S, H)

# central line = MEDIAN (clean under the nonlinear exp), not the mean
median_fc = np.median(pred_pass_future, axis=0)             # (H,)
# 94% predictive band; az.hdi wants an (sample, ...) array → add a chain axis of 1
hdi_fc = az.hdi(pred_pass_future[None, :, :], hdi_prob=0.94)  # shape (H, 2): [:,0]=low, [:,1]=high

# plt.plot(t, y_raw, "k.")                          # historical data
# plt.plot(t_future, median_fc)                     # forecast median
# plt.fill_between(t_future, hdi_fc[:, 0], hdi_fc[:, 1], alpha=.3)   # low, high columns
# plt.axvline(T, ls="--")                           # mark where data ends and forecast begins
```

Note the band accessor: `az.hdi` returns the bounds on an `hdi` axis with coordinates `lower`/`higher` (for a labelled `DataArray` you'd write `.sel(hdi="lower")` / `.sel(hdi="higher")`); on a plain numpy array it's the trailing axis, so we index `[:, 0]` (low) and `[:, 1]` (high). There is **no** `.lo`/`.hi` attribute — reaching for those is a common `AttributeError`.

The picture you'll get is the whole point of the chapter: a forecast that **continues the trend and the season**, with a predictive band that is **tight over the historical data and flares open after month 144**. The flare is the random-walk level admitting it doesn't know where the trend is headed, *compounded* with the irreducible observation noise. A naive `statsmodels` point forecast would give you the center line and hide the flare. Your Bayesian forecast hands the decision-maker the flare — which, for any real decision (how many planes to buy, how much inventory to stock), is the part that actually matters.

> 💡 **Intuition — what's *in* the band?** Be precise about this when you present it. The predictive band contains (a) **observation noise** `sigma_obs` (present even in-sample), (b) **trend uncertainty** from the random-walk level fanning out, (c) **seasonal-coefficient uncertainty** (small here), and (d) **parameter uncertainty** in all of the above, integrated over the posterior. It does *not* contain model-structure uncertainty — if the true world has a regime change or a pandemic, this model can't know. Naming what the band does and doesn't cover is the mark of someone who understands their forecast rather than just running it.

---

## 7. Change-points: when the world switches regime

Sometimes a series isn't smoothly drifting — it **breaks**. A policy changes, a machine is replaced, a safety regulation passes, and the data-generating process before the break is genuinely different from after. A **change-point** (or switchpoint) model says exactly that: there is an unknown time $\tau$ at which one regime ends and another begins, and the position of $\tau$ is itself something to infer.

The canonical teaching example — old as PyMC itself — is the **coal-mining disasters** series: yearly counts of severe coal-mining accidents in the UK from 1851 to 1962. By eye, the disaster rate is high in the Victorian era and drops noticeably in the late 1880s–1890s (around when safety legislation tightened). We model this with a **discrete switchpoint** between two Poisson rates:

$$
\tau \sim \mathrm{DiscreteUniform}(t_\min, t_\max), \qquad
r_t = \begin{cases} \lambda_\text{early} & t < \tau \\ \lambda_\text{late} & t \ge \tau \end{cases}, \qquad
y_t \sim \mathrm{Poisson}(r_t).
$$

The switchpoint $\tau$ is a *parameter*, so the posterior over $\tau$ tells you **when** the regime changed and how sure you are about the year — a beautiful illustration of inference over structure, not just over rates.

### 7.1 The full model, with the discrete-sampler subtlety

```python
disaster_data = pd.read_csv(pm.get_data("coal.csv"), header=None)   # built-in; 1851–1962
# coal.csv has two NaN years → a natural bridge to automatic imputation (Chapter 16)
years = np.arange(1851, 1962)

with pm.Model() as coal_model:
    switchpoint = pm.DiscreteUniform("switchpoint",
                                     lower=years.min(), upper=years.max())
    early_rate = pm.Exponential("early_rate", 1.0)
    late_rate  = pm.Exponential("late_rate", 1.0)

    # pm.math.switch picks the rate per year based on the switchpoint
    rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)

    disasters = pm.Poisson("disasters", rate, observed=disaster_data.values.ravel())
    idata_coal = pm.sample(1000, tune=1000, chains=4,
                           random_seed=RANDOM_SEED)
```

> ⚠️ **Pitfall — the discrete parameter changes the sampler.** `switchpoint` is a **discrete** random variable. NUTS works on continuous gradients and cannot move a discrete parameter, so PyMC quietly assigns `switchpoint` a **Metropolis** step while keeping NUTS for the continuous rates — a `CompoundStep`. This "just works," and it's the canonical demonstration that **PyMC can mix step methods in one model**. But it has two consequences you must internalize: **(1)** the JAX backends (`nuts_sampler="numpyro"`/`"nutpie"`) **cannot sample this model** — they require a fully continuous graph (cross-ref **Chapter 17**). You must use the default `nuts_sampler="pymc"`. **(2)** Metropolis-on-discrete mixes more slowly than NUTS-on-continuous, so check the switchpoint's ESS specifically. If you needed the JAX backends (for speed), you would **marginalize** $\tau$ out — sum the likelihood over all possible switch years — turning it into a continuous model (the marginalization idea from **Chapter 13**).

### 7.2 Reading the change-point posterior

```python
az.summary(idata_coal, var_names=["switchpoint", "early_rate", "late_rate"])
az.plot_posterior(idata_coal, var_names=["switchpoint"])
```

> 🩺 **Diagnostic:** The `switchpoint` posterior is the headline — a (possibly multi-modal) distribution over **years**. For the coal data it concentrates around **1889–1891**, with `early_rate ≈ 3.0` disasters/year before and `late_rate ≈ 1.0` after — a clear, interpretable regime change with quantified uncertainty about *when* it happened. If the posterior over `switchpoint` is broad or bimodal, that's the model honestly telling you the break isn't sharp. Also note: those two NaN years in `coal.csv` trigger PyMC's **automatic imputation** — you'll see an `ImputationWarning` and a `disasters_unobserved` node appear in the InferenceData, with the missing counts filled in from the Poisson likelihood. That's a free preview of **Chapter 16 (Missing Data)**: every posterior draw of the unobserved years is a proper imputation.

> 📜 **Citation/Origin:** The coal-mining disasters change-point analysis dates to Jarrett (1979) and was popularized as the flagship example in the BUGS manual and Carlin, Gelfand & Smith (1992). It became *the* introductory PyMC example precisely because it shows three ideas at once: inference over a structural parameter, mixed step methods, and (thanks to the two NaN years) automatic missing-data imputation.

---

## 8. The state-space / Kalman view: the idea that unifies the chapter

Everything in this chapter — AR, random walk, structural, stochastic volatility — is a special case of one framework: the **linear Gaussian state-space model**. It has exactly two equations:

$$
\underbrace{x_t = T\,x_{t-1} + R\,\eta_t}_{\text{state (transition) equation}}, \qquad
\underbrace{y_t = Z\,x_t + \varepsilon_t}_{\text{observation equation}},
$$

with $\eta_t \sim \mathrm{Normal}(0, Q)$ and $\varepsilon_t \sim \mathrm{Normal}(0, H)$. In words: there is a hidden **state** $x_t$ that evolves by a linear rule $T$ driven by process noise $\eta_t$; you never see $x_t$ directly, only a linear, noisy readout $y_t = Z x_t + \varepsilon_t$. Choosing the matrices $T, R, Z, Q, H$ *is* choosing the model:

| Model | Hidden state $x_t$ | Transition $T$ | Observation $Z$ |
|---|---|---|---|
| Local level (§3) | the level $\mu_t$ | $1$ (carry forward) | $1$ (observe level + noise) |
| Local linear trend (§4.1) | $(\mu_t, \delta_t)$ | $\left[\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right]$ | $(1, 0)$ |
| AR($p$) (§2) | last $p$ values | companion matrix of $\rho$ | $(1,0,\dots,0)$ |
| Structural (§4) | level + slope + season states | block-diagonal | sum the components |

> 💡 **Intuition:** The state-space view is just the §1 chain-rule factorization made into linear algebra. "Only the previous *state* matters" is the **Markov** property, and it's why these models are so composable: stack states, stack transitions, and you've combined trend and season into one system. This is the same latent-variable thinking from **Chapter 13** — there's a hidden process generating what you see — applied along the time axis.

### 8.1 Why state-space sampling is so much better

Here is the deep reason `pymc_extras.statespace` (§4.3) exists, and it's worth understanding even if you mostly hand-build models. When you write the latent path explicitly (a `GaussianRandomWalk` with $T$ states), the sampler must explore a **$T$-dimensional** correlated posterior — the long thin ridge, the funnel risk, the slow mixing we've fought all chapter. But for *linear Gaussian* models, the **Kalman filter** computes the exact marginal likelihood $p(y_{1:T} \mid \theta)$ by analytically integrating out *all* the latent states:

$$
p(y_{1:T}\mid\theta) \;=\; \prod_{t=1}^{T} p(y_t \mid y_{1:t-1}, \theta),
$$

where each one-step-ahead term is a Gaussian whose mean and variance the filter propagates recursively. The sampler then only has to explore the handful of **hyperparameters** $\theta$ (the variances and coefficients) — a tiny, well-conditioned space — while the states are recovered afterward by the **Kalman smoother**. This is the time-series cousin of marginalizing discrete latents in mixtures (**Chapter 13**): integrate out the high-dimensional nuisance analytically, sample only the low-dimensional thing you care about. It is *dramatically* better geometry, and it's why a structural model that crawls when hand-built flies through `statespace`.

> 🔧 **In practice:** Hand-build the latent path (as in §6) when you're learning, when the model is small, or when it's non-Gaussian/nonlinear (the Kalman filter needs linear-Gaussian; stochastic volatility, for instance, is nonlinear in the state so it's typically hand-built or handled with a particle/extended filter). Reach for `pymc_extras.statespace` when the model *is* linear-Gaussian (most structural trend+season models), when $T$ is large, or when the hand-built version funnels. The `PyMCStateSpace` base class lets you subclass for custom systems (`make_symbolic_graph`, `param_names`); `build_statespace_graph`, `.forecast`, and `.sample_conditional_posterior` are the workflow surface (see the single verification callout in §4.3 — those are the APIs that move between releases).

> 🧮 **The math — one Kalman step, conceptually.** Carrying a Gaussian belief $x_{t-1}\mid y_{1:t-1} \sim \mathrm{Normal}(m_{t-1}, P_{t-1})$, the filter does two moves each step: **predict** (push the belief through the transition: $m_t^- = T m_{t-1}$, $P_t^- = T P_{t-1} T^\top + RQR^\top$) and **update** (correct it with the new observation via the Kalman gain). Both moves keep the belief Gaussian — *that's* the magic of linear-Gaussian models, and exactly why everything stays closed-form and the marginal likelihood drops out for free. You won't write this loop (the library does), but knowing it's a predict–update recursion over Gaussians demystifies what `build_statespace_graph` is wiring into your `logp`.

---

## 9. Forecasting, the Bayesian way: the band is the product

We've now forecast twice (the local level in §3.2, the structural model in §6.5). Let me consolidate the *principle*, because it's the most important transferable skill in the chapter and the place naive practitioners go wrong.

A Bayesian forecast is **not** a point and a formula-derived interval. It is the full posterior predictive distribution over future observations,

$$
p(\tilde y_{T+1:T+h} \mid y_{1:T}) \;=\; \int p(\tilde y_{T+1:T+h} \mid \theta)\; p(\theta \mid y_{1:T})\; d\theta,
$$

which we approximate by, for **each** posterior draw $\theta^{(s)}$, simulating the model forward $h$ steps and collecting the resulting future trajectories. The spread of those trajectories *is* the forecast uncertainty, and it bundles together two distinct sources you should be able to name separately:

1. **Parameter uncertainty** — we don't know $\theta$ exactly; the integral over $p(\theta\mid y)$ averages forecasts across every plausible parameter setting. This is the part frequentist point forecasts drop entirely.
2. **Process / innovation uncertainty** — even with $\theta$ known, the future has fresh shocks ($\varepsilon_t$, $\eta_t$) we haven't seen. Each forward simulation draws new ones.

For a **mean-reverting** model (stationary AR), the band widens to a finite asymptote — the stationary variance $\sigma^2/(1-\rho^2)$ from §2.1. For a **random-walk-based** model (local level, structural trend), the band widens *without bound* like $\sqrt{h}$, because the level genuinely forgets. **The shape of your forecast cone is a direct readout of the memory you assumed.** A skeptic who sees your cone can reverse-engineer whether you used a stationary or a unit-root model — that's how diagnostic the picture is.

The mechanical recipe depends on **what the future depends on**, and getting this distinction right is the whole game:

```
CASE A — the future depends only on FIXED inputs (regression on future covariates,
         a deterministic Fourier season, a known calendar). Use the pm.Data swap:
  1. Wrap the future-known inputs in pm.Data(..., dims="time") at build time.
  2. Fit on the training window.
  3. pm.set_data({...future inputs...}, coords={"time": future_index})   # resize coord!
  4. pm.sample_posterior_predictive(idata, predictions=True)             # honest future draws

CASE B — the future extends a FREE LATENT PATH (a random-walk level, an AR state, a
         stochastic-volatility GRW). Do NOT resize-and-PPC a free RV — see §6.5. Instead:
  3'. Forward-simulate the recursion in numpy from posterior draws (continue the last
      fitted state with fresh innovations), OR use pymc_extras.statespace .forecast(...),
      which propagates the state covariance through the Kalman recursion for you.

THEN (both cases):
  5. Summarize: median (or mean for symmetric scales) = the line; az.hdi = the band.
     Report BOTH. After a nonlinear back-transform (e.g. exp), prefer the MEDIAN (§6.6).
```

> ⚠️ **Pitfall — never report only the mean.** I will say it a third time because it is the single most consequential habit in this chapter. The posterior predictive **mean** of a forecast is a tidy line that *looks* like a confident answer. Shipping it without the `az.hdi` band throws away the one thing Bayesian inference uniquely gives you — calibrated uncertainty that grows with the horizon. A forecast without its band is not a humble version of a Bayesian forecast; it's a different, worse object. Plot the band. Annotate where the data ends. Tell the stakeholder, in words, what the band contains (§6.6).

> 🩺 **Diagnostic — is my forecast calibrated?** Don't just trust the band; **back-test** it. Hold out the last $h$ observations, forecast them, and check what fraction fall inside your 50%/94% predictive intervals — they should be roughly 50%/94%. This is **time-series cross-validation**, and it's the honest analogue of the LOO machinery from **Chapter 11** (note: vanilla LOO assumes exchangeability and is *not* directly valid for time series — use leave-*future*-out / one-step-ahead validation instead). If your 94% band only catches 70% of held-out points, your model is overconfident — usually too little process noise or a missing component (a seasonal term, a regime change).

---

## ⚠️ Common errors & how to fix them

The symptom → cause → fix table to keep open while you debug time-series models. It extends the general diagnostics table from **Chapter 07** with the failure modes specific to AR, random walks, structural models, stochastic volatility, change-points, and forecasting.

| Symptom (message / behavior) | Cause | Fix (in order) |
|---|---|---|
| `TypeError: ... unexpected keyword 'sd'` or `'init'` on `GaussianRandomWalk` | v3 API — `sd` and `init` were removed in v5 | Use `sigma=` and `init_dist=pm.Normal.dist(...)` |
| `GaussianRandomWalk` shape mismatch / off-by-one | `steps` vs `shape`/`dims` confusion (a GRW has one more *value* than *steps*) | Give `steps=len(y)-1` **or** a `dims=`/`shape` of length `len(y)`, never inconsistent versions of both |
| `pm.AR` shape error on `rho` | `rho` length ≠ `ar_order + constant` | `len(rho) = ar_order` if `constant=False`, `ar_order + 1` if `constant=True`; set `ar_order=` explicitly |
| AR posterior wanders, $\rho$ piles against $\pm 1$, nonstationary | default `init_dist=Normal(0,100)` too wide; $\rho$ prior unconstrained; series has a unit root | Tighten `init_dist` (e.g. `Normal.dist(0,1)` on a standardized series); use `Normal(0,0.5)` on $\rho$; if it *is* a unit root, switch to a `GaussianRandomWalk` |
| Random walk / structural model: divergences, low ESS, `sigma` won't mix | the $T$-dim latent path is a funnel (same geometry as a hierarchical model) | Raise `target_accept` to 0.95–0.99; **non-center** the random walk (`z~Normal(0,1)`, `level = (z.cumsum())*sigma`); standardize the series |
| `sigma_level` and `sigma_obs` strongly correlated, slow mixing | the process-vs-observation variance ridge (only jointly identified) | Informative prior on the one you know (often measurement precision); tighter `Exponential`/`HalfNormal`; more data |
| Stochastic-volatility model crawls / funnels | thousands of correlated latent vol states; default sampler struggles | `nuts_sampler="numpyro"` or `"nutpie"` (it's fully continuous, so JAX applies); raise `target_accept`; non-center the vol GRW |
| `StudentT` SV model: variance way off | forgot `lam` is **precision** not scale | `lam=pm.math.exp(-2*volatility)` (scale `exp(vol)` ⇒ `lam = 1/scale² = exp(-2·vol)`) |
| Discrete `switchpoint` won't sample with `numpyro`/`nutpie` | JAX backends can't do Metropolis / discrete RVs | Use default `nuts_sampler="pymc"` (it mixes NUTS + Metropolis automatically), **or** marginalize the discrete switchpoint out |
| Change-point: `switchpoint` ESS tiny while rates are fine | Metropolis-on-discrete mixes slower than NUTS-on-continuous | More draws; thin less; check the posterior isn't genuinely multimodal (a real soft break) |
| `ImputationWarning` on the coal data | `coal.csv` has 2 NaN years → automatic imputation kicked in | Expected — a `disasters_unobserved` node appears; that's a feature (Chapter 16), not a bug |
| Forecast shapes wrong after `set_data` | the `time` coord wasn't resized (issue #7064) | Pass `coords={"time": future_index}` inside `set_data`; use `predictions=True` |
| Out-of-sample draws overwrite in-sample PPC | called `sample_posterior_predictive` without `predictions=True` | Use `predictions=True` → results land in `idata.predictions`, separate from `posterior_predictive` |
| Forecast band doesn't widen into the future | extrapolated a *deterministic* trend, or reused the last state with no process noise | Forward-simulate the random walk with **fresh** innovations past the data (§6.5), or use `statespace.forecast`; never just carry the last level forward unchanged |
| Forecast errors out / trend vanishes after `set_data` + PPC | tried to resize a **free latent path** (`z`/RW state) and re-run posterior-predictive — stored `z` is the old length, or gets redrawn from the prior | Don't `set_data`-swap a free RW path; forward-simulate from posterior draws (§6.5) or use `statespace.forecast`. The `pm.Data` swap is only for **fixed** future inputs |
| Auto-imputation silently not happening | NaN data wrapped in `pm.Data` (issue #6626) | Pass the **raw** NaN array to `observed=`, not `observed=pm.Data(...)` |

---

## 🧪 Exercises

1. **(Conceptual — memory and the cone.)** Without code, sketch the forecast band you'd expect from (a) a stationary AR(1) with $\rho=0.6$, and (b) a Gaussian random walk, each forecast 50 steps ahead. Mark the asymptote (if any) and the rate of widening. Then explain, in two sentences, how someone looking only at your forecast cone could tell which model produced it. *Hint: §2.1's $\sigma^2/(1-\rho^2)$ vs §3's $t\sigma^2$.*

2. **(Coding — recover an AR(2) and break it.)** Using the §2.3 simulator, fit the AR(2) and confirm the 94% HDIs cover `(0.5, -0.3)`. Then re-simulate with `true_rho = [0.95, 0.0]` (near unit root) and refit *with the same priors*. Report what happens to the **lag-1 coefficient `rho[0]`** posterior (the $\rho_1$ in our notation — remember §2.3 uses `constant=False`, so `rho[0]` is lag-1, `rho[1]` is lag-2; there is no intercept), the ESS, and any divergences, and explain why a near-unit-root series stresses a stationary AR prior. *Hint: watch `rho[0]`'s mass pile toward 1 and the stationary variance $\sigma^2/(1-\rho_1^2)$ explode as $\rho_1\to 1$.*

3. **(Coding — the non-centering payoff.)** Fit the §3 local-level model on a noisy synthetic random walk **twice**: once centered (`pm.GaussianRandomWalk`) and once non-centered (`z~Normal(0,1)`, `level = z.cumsum()*sigma`). At `target_accept=0.9`, compare divergence counts and the ESS of `sigma`. Then push both to `target_accept=0.99` and compare again. Argue from the numbers why reparameterization, not a higher `target_accept`, is the real cure — exactly as in the Eight Schools funnel from **Chapter 07**.

4. **(Coding — structural model end-to-end + back-test.)** Take the airline-passengers model from §6, but **hold out the last 24 months** as a test set. Fit on the first 120, forecast the held-out 24 with the `pm.Data` swap, and compute the fraction of held-out points inside your 50% and 94% predictive bands. Is the model calibrated? If it's overconfident, add a drifting slope (local linear trend) or another Fourier harmonic and re-check. *Hint: this is leave-future-out validation; report coverage, not just RMSE.*

5. **(Coding — the change-point posterior.)** Fit the §7 coal-mining model. Plot the posterior of `switchpoint` and report its 94% HDI in *years*. Then compute the posterior of the **ratio** `early_rate / late_rate` (a `pm.Deterministic`) and interpret it as "disasters were $k\times$ more frequent before the break." Finally, confirm the `disasters_unobserved` node exists and report the posterior mean of the two imputed years. *Hint: the discrete switchpoint uses a Metropolis step — check its ESS.*

6. **(Conceptual/coding — stochastic volatility reading.)** Fit the §5 SV model on any daily-returns series you can grab (or simulate one with a deliberate "crisis" — a stretch of large `step_size`). Plot $\exp(\text{volatility})$ with its HDI over the absolute returns. Identify the high-volatility cluster, and report the posterior of $\nu$. In one paragraph, explain what a *constant-$\sigma$ Normal* model would have gotten wrong about this series, and which posterior quantity proves the tails are heavier than Gaussian even after the time-varying scale.

7. **(Stretch — state-space upgrade.)** Re-express the §6 structural model using `pymc_extras.statespace` components (`LevelTrendComponent` + `TimeSeasonality` + `MeasurementError`). Compare sampling time and ESS against the hand-built version, and use `.forecast(...)` for the 24-month-ahead bands. Explain, in terms of §8.1, *why* the state-space version samples better. *Hint: verify the current class names and `.build()` signature in the `pymc_extras` docs first; the hand-built model is your ground-truth check.*

---

## 📚 Resources & further reading

The sources behind every model in this chapter. Start with the stochastic-volatility and structural notebooks — they're the two you'll copy from most.

- **PyMC — "Stochastic Volatility" case study** — the canonical SV model (S&P 500 returns, GRW log-volatility, Student-$t$ returns); copy its model verbatim and reproduce the crisis-spike plot. https://www.pymc.io/projects/examples/en/stable/case_studies/stochastic_volatility.html
- **PyMC — Time-series distributions API** — authoritative signatures for `AR`, `GaussianRandomWalk`, `MvGaussianRandomWalk`, `EulerMaruyama`, `GARCH11`; the page to verify every argument against. https://www.pymc.io/projects/docs/en/stable/api/distributions/timeseries.html
- **PyMC — `pymc.AR` API** — exact `rho` / `constant` / `ar_order` / `init_dist` / `steps` semantics and the AR(2) worked example (with `constant=True` the intercept is `rho[0]` and `rho` has length `ar_order + 1`, as we use it in §2.2). https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.AR.html
- **`pymc_extras` — structural components docs** — `LevelTrendComponent`, `TimeSeasonality`, `FrequencySeasonality`, `CycleComponent`, `AutoregressiveComponent`, `MeasurementError`, `RegressionComponent` for building structural TS the state-space way. https://www.pymc.io/projects/extras/en/latest/statespace/models/structural.html
- **`pymc_extras` — `PyMCStateSpace` base class** — `build_statespace_graph`, `.forecast`, `.sample_conditional_posterior`; the full state-space workflow surface and how to subclass for custom systems. https://www.pymc.io/projects/extras/en/latest/statespace/generated/pymc_extras.statespace.core.PyMCStateSpace.html
- **PyMC — "Structural Timeseries Modeling" notebook** — end-to-end local-level-trend + seasonality build with `pymc_extras`; the reference for combining components with `+` and `.build()`. https://github.com/pymc-devs/pymc-extras/blob/main/notebooks/Structural%20Timeseries%20Modeling.ipynb
- **PyMC — "Forecasting Hurricane Trajectories with State Space Models"** — full `pymc_extras.statespace` forecast pipeline, including `scenario=` exogenous forecasting. https://www.pymc.io/projects/examples/en/latest/case_studies/ssm_hurricane_tracking.html
- **ISYE 6420 (Aaron Reding) — Coal-mining disasters / switchpoint in PyMC v5** — a clean modern port of the change-point model with the discrete-step discussion. https://areding.github.io/6420-pymc/unit10/Unit10-disasters.html
- **PyMC — `pymc.sample` API** — confirm `nuts_sampler`, `nuts_sampler_kwargs`, and the `predictions=` / `set_data` forecasting arguments. https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html
- **statsmodels — Mauna Loa CO₂ dataset** — `sm.datasets.co2.load_pandas()`; an excellent second structural-TS / GP series (trend + clean yearly season) to redo §6 on. https://www.statsmodels.org/dev/datasets/generated/co2.html
- **Durbin & Koopman, *Time Series Analysis by State Space Methods* (2nd ed., 2012)** — the definitive reference for the state-space formulation, the Kalman filter/smoother, and structural models. Read Ch 2–3 for the filter, Ch 3 for structural components.
- **Harvey, *Forecasting, Structural Time Series Models and the Kalman Filter* (1989)** — the original, very readable account of additive structural decomposition (trend + season + cycle + irregular).
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python*, Ch 6 — "Time Series"** — the course-aligned PyMC/ArviZ treatment of AR, random walks, and structural models. Free online. https://bayesiancomputationbook.com/notebooks/chp_06.html
- **Taylor & Letham (2018), "Forecasting at Scale"** (Prophet) — the additive trend + Fourier-seasonality + holidays decomposition that made structural TS mainstream; useful intuition for §4. https://peerj.com/preprints/3190/
- **(Background) Vehtari et al. — leave-future-out cross-validation for time series** — why vanilla LOO (Chapter 11) is *not* valid for time series and what to do instead. See the `loo` package docs and Bürkner, Gabry, Vehtari (2020), "Approximate leave-future-out cross-validation."

---

## ➡️ What's next

You can now give your golem a memory — and, just as important, make it admit when that memory runs out. You've modeled dependence over time with autoregressions and random walks, decomposed real series into interpretable trend + season + noise, captured drifting variance with stochastic volatility, located regime changes with change-points, seen how the state-space / Kalman view unifies all of it, and — the headline skill — produced forecasts whose bands fan out for honest, nameable reasons.

Notice the thread we kept tugging: those two NaN years in the coal data, the latent states we never directly observed, the `_unobserved` node that quietly appeared. Time series is *riddled* with missingness — sensors drop out, surveys skip months, the future is just the limiting case of "not yet observed." **Chapter 16 — Missing Data the Bayesian Way** takes that thread and pulls hard: Rubin's MCAR/MAR/MNAR taxonomy, why Bayesian imputation-as-a-parameter is automatically a *proper multiple imputation*, PyMC's automatic NaN handling (the `disasters_unobserved` trick generalized), modeling the missingness *mechanism* when data are missing-not-at-random, and the close cousins censoring (`pm.Censored`) and truncation (`pm.Truncated`). The gaps in your time series were a preview; next we treat missingness as a first-class modeling problem.




