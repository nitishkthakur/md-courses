# Chapter 16 — Missing Data, the Bayesian Way

### Missing values are not holes to be filled — they are parameters to be inferred

> *A note from me to you.*
>
> *Here is a confession that should make you uncomfortable: almost every real dataset you will ever touch has holes in it, and almost every analysis you have ever read quietly papered over them. Someone dropped the rows with missing values (`df.dropna()`), or filled the gaps with the column mean (`df.fillna(df.mean())`), wrote one sentence in the methods section, and moved on. Both of those moves are statistical malpractice dressed up as data cleaning. Dropping rows throws away hard-won information and — far worse — can **bias** every estimate in your model if the rows that went missing are not a random sample. Mean-imputation is even more insidious: it invents a number out of thin air and then **lies about how certain you are of it**, because the model treats that fabricated mean as though you had measured it. Your standard errors shrink, your intervals tighten, and your golem struts around with a confidence it has not earned.*
>
> *This chapter is where we stop apologizing for missing data and start modeling it. The Bayesian view turns out to be almost magically clean here, and it rests on a single sentence that I want you to tattoo on the inside of your eyelids: **a missing value is just another unknown, and we already have a complete, principled machinery for unknowns — we put a distribution on it and infer it jointly with everything else.** A missing entry is not a defect in your data; it is a parameter you have not yet learned. You treat it exactly like a slope or an intercept: give it a model, let the likelihood and the other data inform it, and read its posterior. And because it is inferred jointly, its uncertainty flows automatically into every downstream quantity — no fudging, no fake precision, no Rubin's-rules bookkeeping bolted on afterward.*
>
> *We'll start with Rubin's taxonomy — MCAR, MAR, MNAR — because the single most expensive mistake in this entire field is wrongly assuming your missingness is benign. Then we'll see how PyMC turns a NaN into a latent random variable with almost no ceremony, how to handle a missing **predictor** (which needs a touch more care than a missing **outcome**), how to model the missingness **mechanism** itself when you can't ignore it, and how censoring and truncation are the same idea wearing different clothes. By the end you'll never `dropna()` reflexively again — you'll ask first what the gaps are trying to tell you.*

---

## What you'll be able to do after this chapter

- **State Rubin's taxonomy** — MCAR, MAR, MNAR — precisely, give a concrete real-world example of each, and say exactly what each assumption *permits* you to do and what it *forbids*.
- **Explain why the Bayesian treats missing values as parameters**, and why this automatically propagates imputation uncertainty into every estimate — unlike mean-imputation, which fabricates certainty.
- **Use PyMC's automatic imputation** for a missing **outcome**: pass `observed` data containing `np.nan` (or a masked array), understand the `ImputationWarning`, and read the `<name>_observed` / `<name>_unobserved` nodes that appear in the `InferenceData`.
- **Impute a missing predictor** correctly by building an explicit sub-model for the covariate — the "joint model of $X$ and $Y$" pattern — and know why automatic imputation does *not* cover this case.
- **Model the missingness mechanism** for MNAR data with a **selection model**, jointly modeling the data and the probability of being observed.
- **Handle censoring (`pm.Censored`) and truncation (`pm.Truncated`)** as close relatives of missingness, and articulate the crucial difference: *censoring — you see the bound; truncation — you don't see the point at all.*
- **Inspect the posterior over imputed values** as a sanity check, and contrast the whole approach with classical **multiple imputation**, seeing why one coherent Bayesian model is cleaner.

I'll assume you're fluent with everything through **Chapter 08 (The Bayesian Workflow)**: you write `pm.Model` blocks with named `coords`, set sane priors on **standardized** predictors (the standardization-by-default habit from **Chapter 03**), run `pm.sample`, and gate every model on the diagnostics from **Chapter 07** ($\hat R \le 1.01$, ESS$_\text{bulk}$ and ESS$_\text{tail} \gtrsim 400$, zero divergences). Missing-data models add *latent* parameters — the imputed values themselves — so those diagnostics now apply to the imputations too, and you'll watch them carefully. We also lean on the **latent-variable thinking** from **Chapter 13 (Mixtures & Latent Variables)**: a missing value is the cleanest latent variable there is.

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

## 1. Rubin's taxonomy: the three ways data goes missing

Before a single line of model code, you must answer one question, because everything downstream hinges on it: **why is this value missing?** Not "how do I fill it" — *why is it gone*. Donald Rubin gave us, in 1976, the vocabulary that has organized the entire field ever since. There are three regimes, and they form a ladder from "harmless" to "dangerous."

Let me set up notation we'll keep for the whole chapter. Let $Y$ be the complete data you *would* have observed if nothing were missing. Split it into the part you actually see and the part you don't:

$$
Y = (Y_\text{obs},\, Y_\text{mis}).
$$

Now introduce the **missingness indicator** $R$ — a matrix the same shape as $Y$, with $R_{ij} = 1$ if entry $(i,j)$ is observed and $0$ if it's missing. The whole taxonomy is a statement about what the distribution of $R$ — the probability of being missing — is allowed to depend on. We write that distribution $p(R \mid Y, \psi)$, where $\psi$ are parameters governing the missingness process.

> 🧮 **The math — the three definitions, stated precisely.**
>
> **MCAR (Missing Completely At Random):** missingness depends on *nothing*:
> $$ p(R \mid Y_\text{obs}, Y_\text{mis}, \psi) = p(R \mid \psi). $$
> The probability that an entry is missing is a constant (or depends only on $\psi$), independent of every value in the dataset — observed or not.
>
> **MAR (Missing At Random):** missingness depends *only on observed* values:
> $$ p(R \mid Y_\text{obs}, Y_\text{mis}, \psi) = p(R \mid Y_\text{obs}, \psi). $$
> Once you condition on what you *did* see, the missingness carries no further information about what you *didn't*. The name is a historical misnomer — it does **not** mean "missing in a random way"; it means "missing in a way fully explained by observed data."
>
> **MNAR (Missing Not At Random):** missingness depends on the *unobserved* values themselves:
> $$ p(R \mid Y_\text{obs}, Y_\text{mis}, \psi) \;\text{genuinely depends on}\; Y_\text{mis}. $$
> The very fact that a value is missing tells you something about what that value would have been. This is the hard case, and the dangerous one.

These are abstract. Let me make each one concrete with an example you'll never forget, because the entire art of missing-data analysis is **correctly classifying your situation** into one of these three buckets.

### 1.1 MCAR — the gods of randomness ate your data

A lab technician drops a tray and shatters a random subset of the blood samples. A network packet carrying a sensor reading is lost. A survey is printed double-sided and a stack of forms gets photocopied with one side blank — and which respondents landed in that stack has nothing to do with anything about them. In every case, the *probability* of going missing is the same for all units and unrelated to any value, seen or unseen.

> 💡 **Intuition:** Under MCAR, your observed data is a **simple random sub-sample** of the complete data. That's the strongest, cleanest assumption you can make. It means even the crude approaches — `dropna()` (called *complete-case analysis*) — give you **unbiased** estimates; you just lose efficiency because you threw away rows. MCAR is the assumption everyone *wishes* held. It rarely does.

The danger isn't in *believing* MCAR when it's true; it's in *assuming* it when it's false. If you `dropna()` under non-MCAR missingness, you bias everything, silently.

### 1.2 MAR — missingness explained by what you can see

This is the regime most well-designed analyses live in, and the one the Bayesian machinery handles for free. The key is that the *reason* for missingness, while not random, is **fully captured by variables you observed**.

Concrete example, the one we'll build the worked example around: in a study of body mass, suppose **older** participants are more likely to skip the weigh-in (maybe mobility, maybe modesty), and you have recorded everyone's **age**. Then whether `body_mass` is missing depends on `age` — *which you observed*. Conditional on age, the missingness tells you nothing more about the missing weight. That's MAR. Another: high-income respondents skip the "income" question more often, but you can predict income well from observed education and occupation; conditional on those, missingness is uninformative.

> 💡 **Intuition:** MAR says: *the model already has the variables it needs to explain who went missing.* You don't need to know the missing value to predict whether it's missing — the observed variables suffice. This is exactly the situation in which a Bayesian model that conditions on the observed data does the right thing automatically, as we'll prove in §2.

### 1.3 MNAR — the missingness is *about* the missing value

Now the trap. Suppose people with **very high** incomes refuse to report their income *precisely because it is high* — and there is no set of observed covariates that fully accounts for it. The probability of "income is missing" depends on income itself, which is the thing you don't have. Or: a scale that **can't register weights above 150 kg** simply records nothing — the missingness is caused by the value exceeding a threshold. Or, the canonical clinical example: patients drop out of a depression trial *because they got sicker*, and the very symptom severity that caused them to leave is the unrecorded outcome.

> ⚠️ **Pitfall — the cardinal sin of missing data.** The deadliest mistake in this whole field is **wrongly assuming MAR when the truth is MNAR.** Under MNAR, *no* amount of conditioning on observed variables recovers the truth, because the missing values are systematically different from the observed ones in a way your observed data cannot see. If high earners hide their income and you impute under MAR, you'll impute "typical" incomes into slots that should hold extreme ones, and your estimated mean income will be **biased downward** — confidently. The cruelty is that **you cannot detect MNAR from the data alone**: the observed data is, by construction, silent about the systematically-absent values. MNAR is an assumption you argue for with *domain knowledge*, not a hypothesis you test. The honest workflow is: assume MAR if you have a defensible story for it, then do a **sensitivity analysis** under plausible MNAR mechanisms to see how much your conclusions could move (§6.3).

Here is the ladder, summarized — keep this table next to you:

| Regime | $p(\text{missing})$ depends on… | Concrete example | `dropna()` (complete case) | What you must do |
|---|---|---|---|---|
| **MCAR** | nothing | shattered blood samples; lost packets | unbiased (but wasteful) | anything works; impute to regain efficiency |
| **MAR** | observed data only | older people skip the weigh-in (age recorded) | **biased when missingness relates to the outcome; unbiased if it depends only on covariates** (Little 1992) | condition on the explanatory observed vars; Bayesian imputation is valid, ignore the mechanism |
| **MNAR** | the missing value itself | high earners hide income; scale caps at 150 kg | biased | **model the mechanism jointly** (selection model) |

> 📜 **Citation/Origin:** The taxonomy is Donald B. Rubin, *"Inference and Missing Data,"* **Biometrika** 63(3), 1976 — one of the most cited papers in all of statistics. The book-length treatment is Little & Rubin, *Statistical Analysis with Missing Data* (3rd ed., 2019). The crucial theoretical result we lean on next — that under MAR the missingness mechanism is **ignorable** for likelihood-based and Bayesian inference — is in that paper and is the reason Bayesians get off so lightly.

---

## 2. The Bayesian insight: missing values are parameters

Now the payoff. The reason this chapter is shorter and cleaner than the equivalent chapter in a frequentist textbook is a single conceptual move that the Bayesian framework makes effortless.

In Bayesian inference, **everything you don't know is a parameter with a distribution.** You already accept this for $\mu$, for $\sigma$, for every slope and intercept. The leap — and it is a small one once you see it — is to notice that a **missing data value is also just something you don't know.** So you do the only thing you ever do with an unknown: put a probability model on it and infer it jointly with the rest.

> 💡 **Intuition:** There is no special "missing data theory" in the Bayesian world. There is only *the* theory — joint probability and conditioning — applied to a vector of unknowns that happens to include some data values alongside the usual parameters. The missing entries $Y_\text{mis}$ join $\theta$ in the list of things you infer. You write down the joint, you condition on what you observed, you read the posterior. That's it. The missing values get a posterior just like every parameter does.

Let me make that formal, because seeing the integral is what makes MAR's gift concrete.

### 2.1 Why MAR lets you ignore the mechanism

We want the posterior over our parameters $\theta$ given everything we actually observed — which is $Y_\text{obs}$ **and** the pattern $R$ of what's missing. The full joint model factorizes as

$$
p(Y_\text{obs}, Y_\text{mis}, R \mid \theta, \psi) \;=\; \underbrace{p(Y_\text{obs}, Y_\text{mis}\mid\theta)}_{\text{data model}}\;\underbrace{p(R \mid Y_\text{obs}, Y_\text{mis}, \psi)}_{\text{missingness model}}.
$$

To get the likelihood of what we *saw*, we integrate (marginalize) over the missing values we *didn't*:

$$
p(Y_\text{obs}, R \mid \theta, \psi) \;=\; \int p(Y_\text{obs}, Y_\text{mis}\mid\theta)\, p(R \mid Y_\text{obs}, Y_\text{mis}, \psi)\, dY_\text{mis}.
$$

Now invoke **MAR**: $p(R \mid Y_\text{obs}, Y_\text{mis}, \psi) = p(R \mid Y_\text{obs}, \psi)$. The missingness term no longer depends on $Y_\text{mis}$, so it slides straight out of the integral:

$$
p(Y_\text{obs}, R \mid \theta, \psi) \;=\; p(R \mid Y_\text{obs}, \psi)\,\int p(Y_\text{obs}, Y_\text{mis}\mid\theta)\, dY_\text{mis}
\;=\; p(R \mid Y_\text{obs}, \psi)\; p(Y_\text{obs}\mid\theta).
$$

Look at what happened. The likelihood **factored** into a piece about the missingness ($\psi$) and a piece about the data ($\theta$). If — and this is the second half of *ignorability* — $\theta$ and $\psi$ have independent priors, then the missingness factor $p(R\mid Y_\text{obs},\psi)$ is just a constant as far as inferring $\theta$ is concerned. It contributes nothing to the shape of $\theta$'s posterior. **You can drop it entirely and never model $R$ at all.**

> 🧮 **The math — the punchline.** Under MAR with distinct parameters, the posterior for the things you care about reduces to
> $$ p(\theta \mid Y_\text{obs}) \;\propto\; p(\theta)\, \int p(Y_\text{obs}, Y_\text{mis}\mid\theta)\, dY_\text{mis}. $$
> This is **exactly** the model you'd write if you simply treated $Y_\text{mis}$ as more parameters and let MCMC integrate over them. That integral is *automatic* in any sampler: every draw of $Y_\text{mis}$ from its conditional posterior, marginalized over, *is* the integral. So the recipe "treat missing values as parameters and sample them" is not a hack — it is the literal evaluation of the correct marginal likelihood. This is why Bayesians get MAR for free.

Under **MNAR**, that integral does *not* simplify — the $p(R\mid\cdots)$ term still carries $Y_\text{mis}$ — and you are forced to keep modeling the missingness mechanism explicitly. That's §6.

### 2.2 Why mean-imputation is a lie (and Bayesian imputation isn't)

Contrast two ways to "fill in" a missing weight.

**Mean-imputation.** You compute $\bar y$ from the observed weights and plug it into every gap. Then you fit your model as if those plugged-in numbers were real measurements. What did you do to your uncertainty? Two crimes at once. First, you **shrank the variance** of the weight variable — every imputed point sits exactly at the mean, so the imputed values have *zero* spread, artificially deflating the sample variance and any correlation involving weight. Second, and worse, you told the model **"I measured these"** — so it counts them as full observations, your effective sample size is inflated, and every standard error and credible interval comes out **too narrow.** You have manufactured confidence. A decision-maker reading your intervals will take risks they shouldn't, because you hid the fact that a chunk of your data was guessed.

> ⚠️ **Pitfall:** Mean-imputation (and its cousins, median- and mode-imputation, and naive `fillna`) is *single imputation*: one number, no uncertainty. The single deepest reason it is wrong is that **it replaces a distribution with a point.** A missing value is not a number; it is a distribution of plausible numbers, and any method that collapses it to one number before fitting throws away the very thing that makes the value "missing" rather than "known."

**Bayesian imputation.** You give the missing weight a model — say it shares the population distribution of weights — and let the sampler draw it jointly with everything else. Now each missing entry has a *posterior distribution*, not a point. In one MCMC draw it might be $61$ kg; in the next, $58$; in the next, $64$. That spread is exactly the imputation uncertainty, and because every downstream quantity (the regression slope, the predicted outcome) is computed *inside* each draw using that draw's imputed value, the uncertainty **propagates automatically** into the posterior of everything. Your intervals come out the right width — wider than if the data were complete, exactly as honesty demands. Nothing is faked; nothing is bolted on.

> 💡 **Intuition:** Think of each posterior draw as one *complete, plausible version* of the dataset, with the gaps filled by one coherent guess consistent with the model and the observed data. You have, in effect, infinitely many filled-in datasets, and your final answer averages over all of them. That averaging is what makes the uncertainty honest. Hold this picture — it's exactly the bridge to *multiple imputation* in §7.

---

## 3. Automatic imputation of a missing *outcome* in PyMC

PyMC makes the simplest case — a missing **observed outcome** — almost embarrassingly easy. You don't write any special syntax. You hand the `observed` argument a data array that contains `np.nan` for the missing entries, and PyMC does the rest: it notices the gaps, splits the node in two, and quietly creates latent random variables for the missing positions, sampled from the very likelihood you specified. Let me show it on a tiny example so the mechanics are unmistakable, then explain exactly what got created.

```python
# A handful of observations from a Normal; two are missing.
y = np.array([1.2, np.nan, 0.7, np.nan, 2.1, 1.5, 0.9, np.nan])

with pm.Model() as outcome_model:
    mu = pm.Normal("mu", mu=0, sigma=2)
    sigma = pm.HalfNormal("sigma", sigma=2)
    # Pass the RAW numpy array with NaNs directly to observed=.
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)   # <- triggers ImputationWarning
    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                      random_seed=RANDOM_SEED)
```

When you run this, PyMC prints an **`ImputationWarning`** — this is *informational*, not an error. It is PyMC telling you, "I found NaNs in your observed data and I'm going to impute them as free parameters." Read that warning as a confirmation, not a problem.

> 🔧 **In practice — what PyMC actually built.** Behind the scenes, PyMC split your `obs` node into two pieces, and you'll see both in the resulting `InferenceData`:
>
> - **`obs_observed`** — a `Deterministic`/observed node holding the non-missing entries, contributing the usual likelihood terms.
> - **`obs_unobserved`** — a genuine **latent random variable**, one dimension per missing entry, sampled from the same `Normal(mu, sigma)` likelihood. *This is the imputation.* It has a full posterior.
>
> So `idata.posterior["obs_unobserved"]` is a $(\text{chains} \times \text{draws} \times 3)$ array — the posterior for your three missing values. PyMC literally turned the holes into parameters, exactly as §2 promised. (The exact node names can vary across PyMC versions; the *behavior* — latent RVs for the gaps — is the constant. If a `var_names` lookup fails, list `idata.posterior` to see what your build actually called them.)

> 🩺 **Diagnostic — inspect the imputed posterior, always.** The imputed values are parameters, so they get the same scrutiny as any parameter. Run:
> ```python
> # verify the exact node name in your PyMC version: print(list(idata.posterior)) if this KeyErrors
> az.summary(idata, var_names=["obs_unobserved", "mu", "sigma"])
> ```
> and look at the imputed rows. Two sanity checks: (1) their posterior means should be sensible — for a plain `Normal(mu, sigma)` model with no other information, each imputed value's posterior is *just* the posterior predictive, so all three should center near $\hat\mu$ with spread near $\hat\sigma$; (2) their $\hat R$ and ESS should pass the usual gates ($\hat R \le 1.01$, ESS $\gtrsim 400$). If an imputed value has a pathological posterior, that's a signal your model for that variable is wrong.

### 3.1 NaN arrays vs. masked arrays — two routes to the same place

Historically, PyMC's imputation was triggered by a **NumPy masked array** (`np.ma.masked_array`), where you mark missing entries with a boolean mask. That route still works and you'll see it in older notebooks. The modern, simpler route is to just use `np.nan` in a plain float array, as above. Both produce identical models. Here they are side by side:

```python
# Route 1 (modern, preferred): NaN in a plain float array.
y_nan = np.array([1.2, np.nan, 0.7, np.nan, 2.1])

# Route 2 (classic): a masked array — mask True where the value is MISSING.
y_masked = np.ma.masked_array(
    data=[1.2, -999, 0.7, -999, 2.1],          # the -999 placeholders are ignored
    mask=[False, True, False, True, False],     # True = this entry is missing
)

# Either one can be passed to observed= and yields the same imputation behavior:
# pm.Normal("obs", mu, sigma, observed=y_nan)      # works
# pm.Normal("obs", mu, sigma, observed=y_masked)   # also works, identical model
```

> 🔧 **In practice:** Prefer the **NaN array** — it's less error-prone (no separate mask to keep in sync) and it's how your data already arrives from `pandas` (`df["col"].to_numpy()` turns missing cells into `np.nan` for float columns). Reach for masked arrays only when you have a sentinel value like `-999` and want to mark it missing without rewriting the array. Either way, **the entries must be of a float dtype** — integer arrays can't hold `np.nan`, so cast with `.astype(float)` first if needed.

> ⚠️ **Pitfall — do NOT wrap NaN data in `pm.Data`.** This one bites people. Automatic imputation triggers off the raw array you pass to `observed=`. If you wrap it — `observed=pm.Data("y", y_nan)` — PyMC's NaN-detection can **silently fail to fire**: no `ImputationWarning`, no `_unobserved` node, and (depending on version) NaNs leak into the `logp` and either error cryptically or poison your sampling. Pass the **raw NaN numpy array directly** to `observed=`. This is a known issue (pymc#6626). If you need `pm.Data` for out-of-sample swapping of *predictors*, that's fine — just keep the NaN-bearing **observed outcome** as a bare array.

> ⚠️ **Pitfall — discrete missing outcomes are fragile.** If your observed variable is *discrete* (a `pm.Poisson` or `pm.Bernoulli` outcome) and you feed it NaNs, PyMC will still impute — but the imputed values are **discrete latent parameters**, which NUTS can't move. PyMC falls back to a **Metropolis** step for them, which mixes poorly, and the JAX/nutpie backends (Chapter 17) can't handle them at all. The coal-mining change-point model from **Chapter 15** is exactly this case — its two NaN years get a Metropolis-imputed Poisson count. For a handful of missing discretes it's tolerable; for many, **marginalize** the discrete out instead (sum over its support, the `log_sum_exp` trick from **Chapter 13 (Mixtures & Latent Variables)**) and recover the imputed values afterward with `pm.sample_posterior_predictive` if you need them.

---

## 4. The harder, more common case: a missing *predictor*

The automatic machinery in §3 covers a missing **outcome** — the thing on the left of your likelihood, the `observed=` variable. But in practice the gaps are far more often in your **predictors** — the right-hand-side covariates. And here is a fact that trips up everyone the first time:

> ⚠️ **Pitfall — automatic imputation does NOT cover missing predictors.** PyMC's NaN-detection fires only for the *observed likelihood variable*. Predictors enter your model as ordinary tensors inside the linear predictor; PyMC doesn't treat them as random variables, so it has nothing to impute. If you put NaNs in a predictor and feed it into `mu = alpha + beta * x`, those NaNs propagate straight into the `logp` and your model breaks (NaN gradients, failed sampling) — *without* a friendly `ImputationWarning`. You have to do the imputation **yourself**, explicitly.

The fix is wonderfully principled and it's the same move from §2, made manifest: **build a model for the predictor.** If $x$ has missing entries, give $x$ its own distribution and hand *that* distribution the NaN-bearing array as `observed`. Now $x$ is a (partially latent) random variable — the observed entries inform its distribution, the missing entries become parameters — and you use the resulting tensor downstream in your regression exactly as if it were fully observed. This is the **joint model of $X$ and $Y$**: you write a generative story for *both* the predictor and the outcome, and the sampler infers the missing $X$'s using *both* sources of information at once.

> 💡 **Intuition:** Here's the part that makes the Bayesian approach genuinely *better*, not just tidier, than imputing $x$ in a separate pre-processing step. When you impute $x$ jointly with the outcome model, the missing $x$'s are informed from **two directions**: (1) the distribution of $x$ itself (the observed $x$'s tell us what $x$ values are plausible), and (2) the **outcome** $y$ — if we know the regression $y \approx \alpha + \beta x$ and we observed $y$ for a row with missing $x$, then $y$ *tells us about* the missing $x$ (invert the line!). A two-stage "impute then regress" pipeline can only use direction (1); it never lets the outcome inform the imputation, and it freezes the imputed values, killing the uncertainty. The joint model uses both directions and keeps every imputation uncertain. *That* is why "one coherent model" beats a pipeline.

### 4.1 The pattern, in code

Schematically, a linear regression $y \sim \alpha + \beta x$ where $x$ has missing entries:

```python
# x_data and y_data are numpy arrays; x_data contains some np.nan, y is complete.
x_data = x_with_nan.astype(float)     # ensure float dtype so it can hold NaN
y_data = y_complete

with pm.Model() as joint_model:
    # --- 1. A sub-model (prior) for the PREDICTOR x ---------------------
    # The observed x's inform this; the NaN entries become latent parameters.
    mu_x    = pm.Normal("mu_x", mu=0, sigma=1)
    sigma_x = pm.HalfNormal("sigma_x", sigma=1)
    x_imputed = pm.Normal("x", mu=mu_x, sigma=sigma_x,
                          observed=x_data)        # imputation fires here, on x
    # `x_imputed` now mixes observed values and latent (imputed) values.

    # --- 2. The OUTCOME model uses the (partially imputed) x downstream --
    alpha = pm.Normal("alpha", mu=0, sigma=2)
    beta  = pm.Normal("beta",  mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu_y  = alpha + beta * x_imputed              # missing x's flow in as parameters
    y_obs = pm.Normal("y", mu=mu_y, sigma=sigma, observed=y_data)

    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                      random_seed=RANDOM_SEED)
```

Notice the structure: `x` is *both* a thing with a prior (`mu_x`, `sigma_x`) *and* a thing fed `observed=`. That dual role is the whole trick. Where `x_data` has real values, the `observed=` clause pins `x` to the data and those values inform `mu_x`/`sigma_x`. Where `x_data` is NaN, PyMC creates a latent node (you'll see `x_unobserved` in the `InferenceData`) and infers it — informed by the prior `Normal(mu_x, sigma_x)` *and*, crucially, by the outcome through `mu_y = alpha + beta * x`.

> 🔧 **In practice — choosing the predictor's sub-model.** The model you put on $x$ is itself a modeling choice, and it should reflect what $x$ *is*:
> - A roughly-symmetric continuous predictor → `pm.Normal("x", mu_x, sigma_x, observed=...)` (after standardizing, $\mathrm{Normal}(0,1)$-ish priors are sane — the standardization habit from **Chapter 03** pays off again).
> - A skewed positive predictor (income, concentration) → model $\log x$ as Normal, or use `pm.Gamma`/`pm.LogNormal`.
> - A binary predictor → `pm.Bernoulli`; a count → `pm.Poisson` (but recall the discrete-imputation fragility from §3).
> - **Multiple correlated predictors** with missingness scattered across them → model them *jointly* with a `pm.MvNormal` (with an `pm.LKJCholeskyCov` prior on the correlation, à la **Chapter 10**) so that observed components of a row inform the missing components through their correlation. This is the genuinely powerful version and it's where joint Bayesian imputation crushes column-by-column mean-filling.

> 🩺 **Diagnostic — does the imputation use the outcome?** A quick, satisfying check: for a row with missing $x$ but a *large* observed $y$, the posterior of that row's imputed $x$ should shift toward the $x$ value the regression line would predict for that $y$ (if $\beta>0$, large $y$ ⇒ the imputed $x$ leans high). If you plot the imputed $x$ posteriors colored by their row's $y$, you'll *see* the outcome pulling them — visual proof that the joint model is using direction (2) from the intuition box. A two-stage pipeline would show no such pull.

---

## 5. Worked example: a MAR-missing predictor, imputed jointly

Time to put it all together on a full workflow and — the part that makes this *land* — to **compare three analyses** on the same data: (A) drop the incomplete rows, (B) mean-impute the predictor, and (C) model the missingness the Bayesian way. We'll do it on **synthetic data with a known generative process**, because that's the only setting where we can check who recovered the truth and who got biased. (After we've nailed the mechanism on synthetic data, §5.7 points you at the **Palmer Penguins** dataset, which has genuine missing predictor values, to repeat the exercise on real data.)

### 5.1 The generative story (so we know the right answer)

We simulate a clean linear relationship between a standardized predictor $x$ and an outcome $y$, with known parameters:

$$
x_i \sim \mathrm{Normal}(0, 1), \qquad y_i \sim \mathrm{Normal}(\alpha + \beta\, x_i,\; \sigma), \qquad \alpha = 1.0,\;\; \beta = 2.0,\;\; \sigma = 1.0.
$$

Then — and this is the heart of the demonstration — we make $x$ **MAR-missing in a way correlated with $x$ itself**, but *driven through the observed outcome*. Concretely, we'll let the probability that $x_i$ is missing rise with $y_i$ (which we always observe). Because missingness depends on the *observed* $y$, this is **MAR**, not MNAR — and it is exactly the regime where dropping rows biases you but Bayesian imputation does not. Watch the slope $\beta$: complete-case analysis will systematically mis-estimate it, because the rows we deleted are the high-$y$ (hence high-$x$, since $\beta>0$) rows — we've sawn off one end of the line.

> 🧮 **The math — *why* complete-case is biased here (a crucial refinement).** Be precise about what makes this example break: complete-case regression of $y$ on $x$ is biased **specifically because the missingness depends on the outcome $y$.** This is not a general property of MAR. A standard, slightly surprising result (Little 1992; White & Carlin 2010) is that complete-case regression is *unbiased for the slope* whenever missingness depends **only on the covariates** $x$ — even arbitrarily, even strongly — and becomes biased exactly when it depends on the outcome $y$ (or on unobserved values, i.e. MNAR). The intuition: deleting rows on the basis of $x$ just reshapes the design (the spread of $x$ you condition on), and least-squares / the regression likelihood condition on $x$ anyway, so the conditional $p(y\mid x)$ you're estimating is untouched; deleting rows on the basis of $y$ instead carves the *response* distribution at each $x$, which is precisely the thing the slope summarizes. So had we made $x$ missing as a function of some other observed covariate (not $y$), `dropna()` would have recovered $\beta$ just fine — and the whole §5 demonstration would have shown no bias. We route the missingness through $y$ on purpose, because that is the MAR case where complete-case actually fails.

```python
# ---- Simulate complete data with known truth ----------------------------
N = 200
ALPHA_TRUE, BETA_TRUE, SIGMA_TRUE = 1.0, 2.0, 1.0

x_full = rng.normal(0, 1, size=N)
y = rng.normal(ALPHA_TRUE + BETA_TRUE * x_full, SIGMA_TRUE)

# ---- Make x MAR-missing: P(x missing) increases with the OBSERVED y -------
# logistic in standardized y → higher y, more likely x is missing (MAR via y).
p_missing = 1.0 / (1.0 + np.exp(-(y - y.mean()) / y.std()))   # ~0.5 avg, rises with y
missing_mask = rng.random(N) < p_missing                       # True = x is missing
print(f"Fraction of x missing: {missing_mask.mean():.2f}")     # ~0.50

x_obs = x_full.copy()
x_obs[missing_mask] = np.nan        # the predictor we actually get to see
```

> 🔧 **In practice:** We deliberately made *half* the predictor missing and made the missingness strongly tied to $y$. That's an aggressive, almost adversarial setup — chosen so the bias from naive methods is large and visible. Real problems are usually milder, but the *direction* of the bias is the same, so seeing it big here trains your intuition for spotting it small in the wild.

### 5.2 Baseline A — complete-case analysis (the biased default)

First, the thing most people do without thinking: drop the rows where $x$ is missing and regress on what's left.

```python
keep = ~missing_mask                       # only rows with observed x
# Index the OBSERVABLE array, not the oracle x_full — an analyst never has x_full.
# (On the kept rows x_obs and x_full coincide, so this is honest, not a shortcut.)
x_cc, y_cc = x_obs[keep], y[keep]

with pm.Model() as model_completecase:
    alpha = pm.Normal("alpha", 0, 2)
    beta  = pm.Normal("beta",  0, 2)
    sigma = pm.HalfNormal("sigma", 2)
    mu = alpha + beta * x_cc
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_cc)
    idata_cc = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                         random_seed=RANDOM_SEED)

az.summary(idata_cc, var_names=["alpha", "beta", "sigma"])
```

> 🩺 **What you'll see (expected qualitative behavior).** Because we deleted the high-$y$/high-$x$ rows, the surviving data covers a *truncated* range of $x$, and the fitted slope should come out **biased toward zero** — you should see $\hat\beta$ pulled below the true $2.0$ (often well under $1.7$ at this missingness level), with the intercept compensated upward. The exact value depends on the seed and the realized missing fraction, so treat these as directions, not precise figures; at this aggressive ~50% missingness the 94% interval for $\beta$ may not even cover $2.0$. The model's diagnostics ($\hat R$, ESS) will look *perfect* — that's the trap. Clean diagnostics certify that the sampler explored *this* (wrong) model well; they say nothing about the data you threw away. Good diagnostics on a biased model is the most dangerous combination in applied Bayes.

### 5.3 Baseline B — mean-imputation (fake certainty)

Now the second naive default: fill the missing $x$'s with the mean of the observed $x$'s, then regress on the full $N$ rows as if nothing were missing.

```python
x_meanfill = x_obs.copy()
x_meanfill[missing_mask] = np.nanmean(x_obs)    # plug the observed-x mean into the gaps

with pm.Model() as model_meanfill:
    alpha = pm.Normal("alpha", 0, 2)
    beta  = pm.Normal("beta",  0, 2)
    sigma = pm.HalfNormal("sigma", 2)
    mu = alpha + beta * x_meanfill
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    idata_mean = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                           random_seed=RANDOM_SEED)
```

> 🩺 **What you'll see (expected qualitative behavior).** This is, if anything, *worse* than complete-case, for a subtler reason. Half the $x$ values now sit exactly at the mean, so the predictor's spread is artificially collapsed and its relationship with $y$ is **attenuated** — you should see $\hat\beta$ biased toward zero. (Whether it is biased *more* or *less* than the complete-case slope is not guaranteed across seeds and missingness fractions — it depends on how the collapsed central mass trades off against the truncation, so don't lean on a fixed ordering of the two.) The robust failures are these two: because we kept all $N$ rows, the model thinks it has a full sample, so the credible interval for $\beta$ is **too narrow** — confidently wrong; and the imputed rows (constant $x$, variable $y$) inflate the residual scale, so $\hat\sigma$ tends to come out too big. Mean-imputation manages to bias the slope *and* corrupt the noise estimate at once.

### 5.4 The right way — model C: joint Bayesian imputation

Now the joint model from §4 — and because this is our canonical fully-worked example, we run it through the *whole* Chapter 08 workflow: prior predictive check → sample → diagnostics → posterior predictive check. We give $x$ its own Normal sub-model (it's standardized, so $\mathrm{Normal}$ with weakly-informative hyperpriors is right), pass the NaN array to *its* `observed=`, and use the resulting partially-latent `x` in the outcome regression.

```python
with pm.Model() as model_impute:
    # --- sub-model for the predictor x (imputation fires on the NaNs) ------
    mu_x    = pm.Normal("mu_x", mu=0, sigma=1)
    sigma_x = pm.HalfNormal("sigma_x", sigma=1)
    x_imp = pm.Normal("x", mu=mu_x, sigma=sigma_x, observed=x_obs)  # x_obs has NaNs

    # --- outcome regression uses the (partly imputed) x -------------------
    alpha = pm.Normal("alpha", 0, 2)
    beta  = pm.Normal("beta",  0, 2)
    sigma = pm.HalfNormal("sigma", 2)
    mu = alpha + beta * x_imp
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    # --- WORKFLOW STAGE 1: prior predictive check (before looking at posteriors) ---
    idata_imp = pm.sample_prior_predictive(samples=200, random_seed=RANDOM_SEED)

    # --- WORKFLOW STAGE 2: sample the posterior ---------------------------
    idata_imp.extend(
        pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                  random_seed=RANDOM_SEED)
    )

    # --- WORKFLOW STAGE 3: posterior predictive check ---------------------
    pm.sample_posterior_predictive(idata_imp, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
```

You'll see the `ImputationWarning` fire (good — imputation is on). The `InferenceData` now contains an `x_unobserved` node with one dimension per missing entry — roughly $100$ imputed parameters, each with a full posterior.

> 🩺 **Prior predictive check (workflow stage 1).** Before trusting any posterior, eyeball what the *priors alone* imply for $y$:
> ```python
> az.plot_ppc(idata_imp, group="prior", num_pp_samples=100)
> ```
> With standardized data and these weakly-informative priors, the prior-predictive $y$'s should span a plausible range (roughly the $\pm$ tens, not $\pm 10^6$) and comfortably *cover* the observed $y$'s without being absurdly wider — confirming the priors are weakly-informative, not accidentally informative or pathologically vague. If the prior predictive looked nothing like the data's scale, that's your cue to revisit priors *before* spending compute on sampling.

> 🩺 **Posterior predictive check (workflow stage 3).** After sampling, confirm the fitted model can regenerate data that looks like what you saw:
> ```python
> az.plot_ppc(idata_imp, num_pp_samples=100)   # group="posterior_predictive" by default
> ```
> The posterior-predictive density of $y$ should overlay the observed-$y$ density closely. A systematic mismatch (e.g. the model can't reproduce the spread of $y$) would flag a misspecified likelihood — not something the imputation can paper over.

> 🩺 **What you'll see (expected qualitative behavior).** $\hat\beta$ should recover the truth — its posterior covers $2.0$ far more comfortably than the two baselines do, because the joint model uses the observed $y$ to reconstruct what the deleted $x$'s must have been. The interval should be **appropriately wide** — wider than a hypothetical complete-data fit (we genuinely have less information) but not fake-narrow like mean-fill. The intercept and $\sigma$ should also land near their true values. This is the headline result: *only the model that respects the missingness recovers the truth.* Check the diagnostics — they should pass the gates, including for the `x_unobserved` parameters. With this much missingness you might see a divergence or two; if so, bump `target_accept` to `0.95` (the Chapter 07 reflex).

### 5.5 The side-by-side that makes the point

Plot the three posteriors for $\beta$ on one axis and let the picture do the arguing:

```python
az.plot_forest(
    [idata_cc, idata_mean, idata_imp],
    model_names=["A: complete-case", "B: mean-impute", "C: joint Bayesian"],
    var_names=["beta"], combined=True, figsize=(8, 3),
)
plt.axvline(BETA_TRUE, color="k", linestyle="--", label="true β = 2.0")
plt.legend(); plt.title("Slope estimate under three treatments of missing x");
```

> 🩺 **Reading the forest plot (expected qualitative behavior).** You'll see three horizontal intervals against the dashed line at $\beta=2.0$. **A (complete-case)** should sit low, its interval possibly missing the truth — biased by deletion. **B (mean-impute)** should also sit low but with a *deceptively tight* interval — biased *and* overconfident, the worst combination (its exact position relative to A can vary by seed, but its over-narrow interval is the robust tell). **C (joint Bayesian)** should straddle the dashed line with an honestly wider interval — unbiased and calibrated. One plot, the whole moral of the chapter. Print the numbers too:
> ```python
> for name, idata in [("complete-case", idata_cc),
>                     ("mean-impute", idata_mean),
>                     ("joint Bayes", idata_imp)]:
>     b = az.summary(idata, var_names=["beta"]).loc["beta"]
>     print(f"{name:>14}: β = {b['mean']:.2f}  [{b['hdi_3%']:.2f}, {b['hdi_97%']:.2f}]")
> ```

### 5.6 Inspecting the imputed posterior as a sanity check

Because this is synthetic, we have a luxury we *never* have in real life: we know the true $x$ values that were deleted. Let's grade the imputations against them.

```python
# Posterior mean of each imputed x, in the order PyMC stored them.
imp_post = idata_imp.posterior["x_unobserved"]                 # (chain, draw, n_missing)
imp_mean = imp_post.mean(dim=("chain", "draw")).values
imp_hdi  = az.hdi(idata_imp, var_names=["x_unobserved"])["x_unobserved"].values

x_true_missing = x_full[missing_mask]                          # the values we hid

plt.errorbar(x_true_missing, imp_mean,
             yerr=np.abs(imp_hdi.T - imp_mean), fmt="o", alpha=0.5)
plt.plot([-3, 3], [-3, 3], "k--")                             # perfect recovery line
plt.xlabel("true (hidden) x"); plt.ylabel("imputed x  (posterior mean ± 94% HDI)")
plt.title("Did the joint model recover the deleted predictor values?");
```

> 🩺 **Reading the recovery plot:** Points scatter around the $45°$ line — the model recovers the hidden $x$'s *on average*, and each comes with an honest error bar (the imputation uncertainty). They won't sit *exactly* on the line (we deleted half the predictor — perfect recovery is impossible), and that's the point: the error bars **own** that uncertainty. Two things to verify: (1) no systematic tilt away from the diagonal (a tilt would betray a misspecified sub-model for $x$); (2) the error bars are wide enough that roughly 94% of the diagonal-line points fall inside their 94% HDIs — that's **calibration** of the imputations, the same idea as posterior-predictive calibration from **Chapter 11**. If the bars are too narrow to cover the truth that often, your imputations are overconfident.

> 💡 **Intuition — what to do when you have NO ground truth (i.e., always).** In real data you can't make this plot, so you sanity-check the imputations *indirectly*: (1) compare the **distribution** of imputed values to the distribution of observed values with `az.plot_dist` or a histogram overlay — they should be plausibly similar in shape (a posterior-predictive check *for the missing data*); (2) confirm imputed values respect known **bounds** (no negative ages, no imputed probabilities outside $[0,1]$) — if they don't, your sub-model needs a bounded distribution or a transform; (3) check the imputed posteriors mix well ($\hat R$, ESS). These three checks are your real-world substitute for the recovery plot.

### 5.7 Repeat on real data — Palmer Penguins

To see this on genuine missingness, load the **Palmer Penguins** dataset, which ships with real missing values in its body-measurement columns. Here is the *same* joint-imputation pipeline as §5.4, end to end, on real ecological data — standardize, build a Normal sub-model for the NaN-bearing predictor, regress, prior-predictive check, sample, diagnose, posterior-predictive check.

```python
import seaborn as sns
penguins = sns.load_dataset("penguins")     # 344 rows; several columns have real NaNs
print(penguins.isna().sum())                # flipper_length_mm, body_mass_g, sex, ... have gaps

# --- preprocessing -------------------------------------------------------
# Drop rows missing the OUTCOME (auto-imputation could handle these too, but here we
# isolate the lesson to a missing PREDICTOR); keep rows where flipper length is NaN.
peng = penguins.dropna(subset=["body_mass_g"]).copy()

def standardize(s):                          # Chapter 03 habit: standardize before modeling
    return (s - s.mean()) / s.std()

y_peng = standardize(peng["body_mass_g"]).to_numpy()
# Standardize flipper using observed-only moments, THEN expose its NaNs to PyMC.
flip = peng["flipper_length_mm"]
flip_std = ((flip - flip.mean()) / flip.std()).to_numpy()   # NaNs stay NaN
print(f"Missing flipper after dropping NaN mass: {np.isnan(flip_std).sum()}")

# --- the joint-imputation model (identical structure to §5.4) ------------
with pm.Model() as model_penguins:
    mu_x    = pm.Normal("mu_x", 0, 1)
    sigma_x = pm.HalfNormal("sigma_x", 1)
    x_imp = pm.Normal("flipper", mu=mu_x, sigma=sigma_x, observed=flip_std)  # NaNs impute

    alpha = pm.Normal("alpha", 0, 1)
    beta  = pm.Normal("beta",  0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    mu = alpha + beta * x_imp
    pm.Normal("mass", mu=mu, sigma=sigma, observed=y_peng)

    idata_peng = pm.sample_prior_predictive(samples=200, random_seed=RANDOM_SEED)
    idata_peng.extend(pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                                random_seed=RANDOM_SEED))
    pm.sample_posterior_predictive(idata_peng, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)

az.summary(idata_peng, var_names=["alpha", "beta", "sigma"])
az.plot_ppc(idata_peng, num_pp_samples=100)   # posterior predictive check on real data
```

> 🩺 **What you'll see (expected qualitative behavior).** Flipper length and body mass are strongly positively related, so on standardized scales $\hat\beta$ should land high (these two variables correlate around $0.87$ in this dataset, so expect a standardized slope in the high-$0.7$-to-$0.9$ range, seed- and preprocessing-dependent). The `flipper_unobserved` node holds the imputed standardized flipper lengths for the rows that were missing it; check their $\hat R$/ESS like any parameter. The posterior-predictive density of standardized mass should overlay the observed density closely. Compare the slope to a `dropna()`-on-flipper baseline: because Penguins has only a *couple* of missing flippers, the shift here is small (real problems are usually milder than §5's adversarial 50%) — but the pipeline is identical, and on a column with heavier missingness the widening intervals and shifted point estimate would reappear, exactly the §5 lesson, now on data you didn't manufacture.

---

## 6. When you can't ignore it: modeling the MNAR mechanism

Everything so far assumed **MAR** — and under MAR we proved (§2.1) you can *ignore* the missingness mechanism. When the data is **MNAR**, that proof collapses: the missingness term keeps $Y_\text{mis}$ inside it, won't factor out, and pretending otherwise biases you. The only honest fix is to **model the missingness mechanism jointly** with the data. This is a **selection model**: you write down not just a model for $Y$, but a model for *why each value is observed or not*, and you fit them together.

### 6.1 The selection-model structure

You augment your generative story with the missingness indicator $R_i \in \{0,1\}$ (1 = observed) and give it its own likelihood that is *allowed to depend on the value $y_i$ itself*:

$$
y_i \sim p(y \mid \theta), \qquad R_i \sim \mathrm{Bernoulli}\big(\pi_i\big), \qquad \mathrm{logit}(\pi_i) = \gamma_0 + \gamma_1\, y_i.
$$

The parameter $\gamma_1$ is the **MNAR knob**: it says how the *value* of $y_i$ changes its probability of being observed. If $\gamma_1 = 0$, observation doesn't depend on the value and you're back in MAR/MCAR. If $\gamma_1 \neq 0$, high (or low) values are systematically more likely to vanish, and the model *corrects* for that by knowing that the unobserved $y$'s are drawn preferentially from the suppressed region.

> 🧮 **The math — why this works.** The joint likelihood for one unit, with its $R_i$, is $p(y_i\mid\theta)\,p(R_i\mid y_i,\gamma)$. For a unit where $y_i$ is **missing**, we must marginalize over the unseen $y_i$:
> $$ p(R_i = 0) = \int p(y_i\mid\theta)\,\big(1-\pi(y_i)\big)\,dy_i. $$
> That integral *weights* the data distribution by the probability of being missing — so the model literally accounts for the fact that the missing values were drawn from the part of the distribution that tends to go missing. The observed $\gamma_1$ tells you the strength of the selection. The catch, and it's a deep one: this integral is only identified if you have **some** information pinning down $\gamma_1$ — either a defensible prior on it, or auxiliary data — because the data alone is consistent with many $(\theta, \gamma_1)$ trade-offs (a steeper selection plus a higher true mean can mimic the same observed sample).

### 6.2 A selection model in PyMC

Here's the pattern for an MNAR outcome where larger $y$ is more likely to be missing. To make the synthetic example genuinely MNAR, we now apply §5's `missing_mask` **to $y$ itself** (rather than to $x$) — so the value that drives its own missingness is the very value we don't get to see. We model the *observed* $y$'s, treat the *missing* $y$'s as explicit latent parameters, and add a Bernoulli likelihood for the **observation indicator** that depends on the $y$ value — observed or latent — that belongs to *that same row*.

> ⚠️ **Pitfall — do NOT trust the auto-impute handle to preserve row order.** It is tempting to write `y_latent = pm.Normal("y", ..., observed=y_with_nan)` and then `gamma1 * y_latent` against an indicator vector `r` in the original row order. **This is a bug.** When PyMC auto-imputes, it splits the node into a `y_observed` piece and a `y_unobserved` piece, and the Python handle it returns is a *concatenation* of those two — its element order is `[all observed rows, then all missing rows]`, **not** the original array order (this is exactly why the InferenceData shows separate `y_observed`/`y_unobserved` nodes). Multiplying that re-ordered vector against `r` (which *is* in original order) silently pairs each observation indicator with the **wrong** $y$, and the selection model fits garbage (or raises a shape error). The robust fix is to never rely on positional alignment: build the two pieces yourself and pair each $R$ term with the $y$ it belongs to.

```python
# Build the MNAR example: missingness is driven by y itself (genuinely MNAR).
y_with_nan = y.copy()
y_with_nan[missing_mask] = np.nan          # reuse §5's mask, now applied to y

obs_idx = np.where(~missing_mask)[0]        # rows where y is observed
mis_idx = np.where(missing_mask)[0]         # rows where y is missing
n_missing = mis_idx.size

with pm.Model() as selection_model:
    mu    = pm.Normal("mu", 0, 5)
    sigma = pm.HalfNormal("sigma", 5)

    # 1a. Data model for the OBSERVED y's (ordinary likelihood, original values).
    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y[obs_idx])

    # 1b. Data model for the MISSING y's: explicit latents, same Normal(mu, sigma).
    #     These are the imputed outcomes — we control their order, so no misalignment.
    y_mis = pm.Normal("y_mis", mu=mu, sigma=sigma, shape=n_missing)

    # 2. SELECTION model: P(observed) is a logistic function of THAT row's y.
    gamma0 = pm.Normal("gamma0", 0, 1.5)
    gamma1 = pm.Normal("gamma1", 0, 1.0)   # <- the MNAR knob; PRIOR matters a lot

    # Observed rows: R = 1, logit uses the OBSERVED y for that row.
    pm.Bernoulli("R_obs", logit_p=gamma0 + gamma1 * y[obs_idx],
                 observed=np.ones(obs_idx.size))
    # Missing rows: R = 0, logit uses the LATENT (imputed) y for that row.
    pm.Bernoulli("R_mis", logit_p=gamma0 + gamma1 * y_mis,
                 observed=np.zeros(n_missing))

    idata_sel = pm.sample(1000, tune=1000, chains=4, target_accept=0.95,
                          random_seed=RANDOM_SEED)
```

The magic is in the pairing: each `R_obs` term is tied to the *observed* $y$ of its row (and asserts "this row was seen"), while each `R_mis` term is tied to the *latent* `y_mis` of its row (and asserts "this row went missing"). For the missing rows, the sampler must choose `y_mis` values that are *simultaneously* consistent with the `Normal(mu, sigma)` data model **and** with having been likely to go missing — which drags them into the suppressed tail (high $y$, here), correcting the bias that a MAR fit would suffer. Because we assembled the observed and latent pieces by hand, every $R$ is guaranteed to sit beside the correct $y$. (If you *do* want to use auto-imputation's `y_unobserved` handle, first reassemble a full-order vector with `pt.set_subtensor` — `import pytensor.tensor as pt` — before forming any `logit_p`; never assume the handle is already in array order.)

> ⚠️ **Pitfall — selection models live and die by the prior on $\gamma_1$.** MNAR models are often **weakly identified**: the data can't fully separate "a high true mean with mild selection" from "a low true mean with strong selection." That means your prior on $\gamma_1$ (and your *structural* assumption about how selection depends on $y$ — linear in $y$? on $\log y$? a threshold?) is doing real inferential work, not just regularizing. This is not a bug to hide; it's the honest cost of MNAR. **Always** report how your conclusions change as you vary the $\gamma_1$ prior — that *is* the analysis.

### 6.3 Sensitivity analysis is the real deliverable

Because MNAR can't be confirmed from data, the professional move is rarely "fit one MNAR model and trust it." It's a **sensitivity analysis**: fit the MAR model, then fit selection models across a *range* of assumed selection strengths (e.g., fix $\gamma_1$ at several plausible values, or use priors of increasing strength), and report how much your headline estimate moves.

```python
# pseudocode — sensitivity sweep over assumed MNAR strength
for g1_fixed in [-1.0, -0.5, 0.0, 0.5, 1.0]:      # 0.0 == the MAR/ignorable case
    # refit with gamma1 PINNED at g1_fixed (pm.Data or a tight prior), record the
    # posterior mean of the quantity you care about, then plot estimate vs g1_fixed.
    ...
```

> 💡 **Intuition:** A sensitivity curve answers the question a decision-maker actually has: *"How wrong could my conclusion be if the missingness is worse than I hope?"* If your estimate barely budges across a wide range of plausible selection strengths, your conclusion is **robust** to MNAR and you can defend it. If it swings wildly, you've learned something vital — the data are too compromised to answer the question without stronger assumptions or better data collection. Either outcome is more honest than a single MAR fit that quietly assumes $\gamma_1 = 0$.

> 📜 **Citation/Origin:** Selection models trace to James Heckman's 1979 econometric correction for sample selection (the "Heckman correction," which won the 2000 Nobel). The Bayesian treatment and the pattern-mixture vs. selection-model dichotomy are covered in Little & Rubin (2019, Ch. 15) and in Daniels & Hogan, *Missing Data in Longitudinal Studies* (2008), which is the definitive reference for sensitivity analysis under MNAR.

---

## 7. Close relatives: censoring and truncation

Two phenomena look like missing data, are handled by their own dedicated PyMC tools, and confuse people endlessly because they sound alike. Let me draw the line sharply, because using the wrong one biases you.

> 💡 **Intuition — the one-sentence distinction.** **Censoring: you know a value exists and you know it's beyond a bound, but not its exact size** (you see the bound). **Truncation: values beyond a bound are never recorded at all — they're absent from your dataset and you don't even know how many there were** (you don't see the point). Censoring keeps the row with partial information; truncation deletes the row entirely and hides the deletion.

Concrete pairs:
- **Censoring.** A scale maxes out at $150$ kg: a $170$-kg person reads "$150+$" — you know they're *at least* $150$, just not the exact value. A clinical trial ends at $5$ years: a patient still alive is "censored at $5$" — survival time is *at least* $5$. The data point is there; its value is only known to be on one side of a threshold.
- **Truncation.** A study only enrolls people with income above $\$20{,}000$: anyone below simply never appears in your data, and you don't know how many you're missing. A particle detector that can't register energies below a threshold: low-energy events leave *no trace at all*.

### 7.1 Censoring with `pm.Censored`

For censored data, the likelihood contribution of a censored point isn't the *density* at the bound — it's the *probability mass beyond the bound*. For a value right-censored at $c$ (we know $Y \ge c$), the contribution is the survival probability:

$$
P(Y \ge c) = 1 - F(c),
$$

where $F$ is the CDF of the underlying distribution. Using the density there instead — the classic mistake of just plugging the bound in as if it were the true value — throws away the "it could be much larger" information and **biases your estimates** (it pulls your fitted distribution toward the bound). `pm.Censored` encodes the correct survival/CDF contribution for you:

```python
# y_cens: values, with anything at-or-above upper_bound recorded AS the bound.
# (e.g., the scale capped at 150 → those rows hold 150.0)
upper_bound = 150.0

with pm.Model() as censored_model:
    mu    = pm.Normal("mu", 0, 50)
    sigma = pm.HalfNormal("sigma", 50)
    base  = pm.Normal.dist(mu, sigma)                 # UNNAMED .dist() base distribution
    obs = pm.Censored("obs", base,
                      lower=None, upper=upper_bound,   # lower=None → not left-censored
                      observed=y_cens)
    idata_cens = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                           random_seed=RANDOM_SEED)
```

> 🔧 **In practice — the `pm.Censored` API.** It wraps an **unnamed** base distribution created with `.dist()` (note `pm.Normal.dist(mu, sigma)`, *not* `pm.Normal("...", ...)`). Set `lower=None` (or `-np.inf`) to disable left-censoring, `upper=None` (or `np.inf`) to disable right-censoring; pass both for interval-censored data. `pm.Censored` is **likelihood-only** — it must be given `observed=`; you cannot use it as a prior on a parameter. Internally it reassigns the probability mass beyond each active bound to a point mass at the bound, which is exactly the survival-probability contribution above. `# verify lower/upper kwarg behavior in current PyMC docs.`

> 🩺 **Diagnostic (expected qualitative behavior).** Fit the model *ignoring* censoring (treat the capped values as real $150$s) and compare to the `pm.Censored` fit. The naive fit should report a **smaller $\hat\mu$ and smaller $\hat\sigma$** — it believes the piled-up-at-$150$ mass is the true distribution rather than a stack of censored tails. The censored model should recover the true (larger) mean and spread because it knows those points represent "$\ge 150$." Seeing the two side by side is the cleanest demonstration of *why* you can't just plug in the bound.

### 7.2 Truncation with `pm.Truncated`

For truncated data, the underlying distribution must be **renormalized** to integrate to one over the *observed* region, because values outside it were never possible to record. The likelihood for a truncation to $[a, b]$ is

$$
p(y \mid a \le Y \le b) = \frac{p(y)}{F(b) - F(a)}, \qquad a \le y \le b,
$$

and zero outside. The denominator is what makes truncation different from censoring: it accounts for the *invisible* missing mass by inflating the density of what you *did* see. `pm.Truncated` handles it:

```python
# Only values within [lower_bound, upper_bound] ever entered the dataset.
lower_bound = 20.0

with pm.Model() as truncated_model:
    mu    = pm.Normal("mu", 0, 50)
    sigma = pm.HalfNormal("sigma", 50)
    base  = pm.Normal.dist(mu, sigma)                 # again an UNNAMED .dist()
    obs = pm.Truncated("obs", base,
                       lower=lower_bound, upper=None,  # left-truncated only
                       observed=y_trunc)
    idata_trunc = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                            random_seed=RANDOM_SEED)
```

> ⚠️ **Pitfall — using the wrong tool biases you in opposite directions.** Treat *truncated* data as if it were complete (no renormalization) and you'll **mis-estimate the mean and underestimate the spread** — the model doesn't know a whole tail was sliced off, so it shoves the distribution toward the truncation point. Treat *censored* data as truncated (or vice versa) and you'll be wrong differently. The decision rule is purely about your **data-collection process**: *Did values beyond the bound get recorded as the bound (→ censored, `pm.Censored`) or did they never appear at all (→ truncated, `pm.Truncated`)?* Answer that one question correctly and the modeling is easy; get it wrong and no amount of sampling saves you. Both `pm.Censored` and `pm.Truncated` are **likelihood-only** and wrap an unnamed `.dist()`.

> 📜 **Citation/Origin & where to go deeper:** The PyMC example gallery's *"Bayesian regression with truncated or censored data"* is the definitive worked notebook for both, and the *"Censored Data Models"* survival-analysis example ties censoring to the survival-likelihood view. Links in Resources below.

---

## 8. How this compares to Multiple Imputation

If you've done missing-data work in a frequentist shop, you've met **Multiple Imputation (MI)** — Rubin's own celebrated method, and the standard outside the Bayesian world. It's worth seeing how it relates, because the comparison reveals *why* the single coherent Bayesian model is cleaner, and because MI is, at its core, a Bayesian idea exported to non-Bayesian downstream analyses.

MI is a three-step recipe:
1. **Impute** — fit a model to the data (often itself Bayesian) and draw $m$ (say, $20$) *complete* datasets, each with the gaps filled by a different plausible draw. The variation *across* the $m$ datasets encodes the imputation uncertainty.
2. **Analyze** — run your downstream analysis (the regression, the t-test) separately on each of the $m$ completed datasets, giving $m$ sets of estimates and standard errors.
3. **Pool** — combine the $m$ results with **Rubin's rules**: average the point estimates, and combine the within-imputation variance with the between-imputation variance to get a total variance that reflects both ordinary sampling error *and* imputation uncertainty.

> 💡 **Intuition — MI and Bayesian imputation are the *same idea*, split into stages.** Look back at §2.2: each posterior draw in our joint model *is* one completed dataset, and the spread across draws *is* the imputation uncertainty. MI does exactly this, but it **freezes** the imputation after step 1 and bolts the uncertainty back on with an algebraic formula (Rubin's rules) in step 3. The Bayesian joint model never freezes anything — the imputed values and the analysis parameters are sampled *together*, so the uncertainty flows continuously and exactly, with no pooling formula required. Each MCMC draw of `x_unobserved` is, quite literally, one of MI's imputations; you just have thousands of them and you never had to leave the model.

Here's the comparison that should guide your judgment:

| | **Multiple Imputation (MI)** | **Joint Bayesian model (this chapter)** |
|---|---|---|
| Imputation model | separate, fit first | the *same* model, fit jointly |
| Does the **outcome** inform the imputation? | only if you're careful to include $y$ in the imputation model | **automatically and always** — they're one model |
| Uncertainty propagation | approximate, via **Rubin's rules** pooling | **exact**, flows through the joint posterior |
| Number of imputations | finite ($m\approx 20$); a Monte Carlo approximation | effectively the whole posterior (thousands of draws) |
| Congeniality risk | imputation & analysis models can **disagree** ("uncongenial"), biasing pooled results | impossible — there's only one model |
| Feedback ($y \to$ imputed $x$) | broken by the two-stage split | intact |
| Ease in PyMC | you'd build it as the joint model anyway | the natural, default formulation |

> 🔧 **In practice:** MI is the right tool when you *can't* fit one joint model — e.g., the downstream analysis is a black box (someone else's procedure, a fixed regulatory pipeline) and you can only hand it completed datasets. In that world, MI's modularity is a feature. But when **you** own the whole analysis, the joint Bayesian model is strictly cleaner: no congeniality worries (the imputation and analysis models are provably compatible because they're the *same* model), no Rubin's-rules arithmetic, and the outcome informs the imputations for free. If you ever need MI-style completed datasets to feed elsewhere, you can *extract* them from a Bayesian fit — just draw $m$ samples of `x_unobserved` and write out $m$ filled datasets. The Bayesian model is the more general object; MI is a special, staged case of it.

> 📜 **Citation/Origin:** Multiple Imputation is Rubin, *Multiple Imputation for Nonresponse in Surveys* (1987). The "congeniality" concern — that the imputation and analysis models must be compatible for MI's variance estimates to be valid — is Meng (1994), *"Multiple-Imputation Inferences with Uncongenial Sources of Input."* The joint-modeling view as the coherent ideal is laid out in Gelman et al., *Bayesian Data Analysis* (BDA3), Chapter 18, "Models for missing data."

---

## ⚠️ Common errors & how to fix them

| Symptom | Likely cause | Fix |
|---|---|---|
| No `ImputationWarning`, no `_unobserved` node, NaNs leak into `logp` | NaN data wrapped in `pm.Data(...)` (issue pymc#6626) | Pass the **raw NaN numpy array** straight to `observed=`, not `observed=pm.Data(...)`. |
| `ValueError`/`TypeError` about NaN or integer dtype | NaNs in an **integer** array (integers can't hold NaN) | Cast to float first: `x = x.astype(float)`. |
| NaNs in a **predictor** crash sampling, no imputation warning | Auto-imputation only covers the **observed likelihood** variable, not covariates | Build an explicit sub-model: `pm.Normal("x", mu_x, sigma_x, observed=x_nan)`, then use `x` downstream (§4). |
| Slope/intercept biased even though diagnostics are perfect | Complete-case (`dropna`) or mean-impute under non-MCAR missingness | Model the missingness: joint imputation (MAR) or selection model (MNAR). Clean diagnostics ≠ unbiased. |
| Imputed-value posteriors don't mix; many divergences | Discrete missing data imputed via Metropolis; or a funnel between scale params and imputed values | Marginalize discretes (Ch 13); raise `target_accept` to 0.95–0.99; non-center the predictor sub-model. |
| Selection model gives wildly different answers run-to-run / per prior | MNAR model **weakly identified** — data can't pin down $\gamma_1$ | This is expected. Use an informative prior on $\gamma_1$ and report a **sensitivity analysis** (§6.3); don't trust one fit. |
| `pm.Censored`/`pm.Truncated` errors "can't sample" / no `observed` | Tried to use it as a **prior** or forgot `observed=` | They are **likelihood-only**; always pass `observed=` and wrap an **unnamed** `.dist()` base. |
| Censored fit gives a mean that's too low/spread too small | Plugged censored values in as real numbers instead of using `pm.Censored` | Use `pm.Censored(..., upper=bound, observed=...)` so the bound contributes survival prob $1-F(c)$, not density. |
| Truncated fit biased toward the bound | Treated truncated data as complete (no renormalization) | Use `pm.Truncated`; it divides by $F(b)-F(a)$ to renormalize over the observed region. |
| JAX/nutpie backend refuses to sample the missing-data model | Discrete imputed RV (Metropolis) in the graph; JAX backends need a fully continuous graph | Use default `nuts_sampler="pymc"`, or marginalize the discrete out (Ch 13, Ch 17). |
| Imputed values violate bounds (negative ages, probs > 1) | Sub-model distribution is unbounded/wrong for that variable | Use a bounded/transformed sub-model (`pm.LogNormal`, `pm.Beta`, log-transform), not a plain `Normal`. |

---

## 🧪 Exercises

1. **(Conceptual — classify the mechanism.)** For each scenario, name the regime (MCAR / MAR / MNAR) and justify in one sentence, then say whether `dropna()` would bias the analysis: (a) a temperature logger's SD card has a corrupted sector that randomly drops 3% of readings; (b) in a wage survey, people with higher education skip the "weekly hours" question more, and education is recorded; (c) patients in a weight-loss trial stop reporting weight once they've regained the lost pounds; (d) a survey question about drug use is skipped more often by actual users. *Hint:* the test is always "does $P(\text{missing})$ depend on nothing, on observed variables, or on the missing value itself?"

2. **(Coding — see the bias yourself.)** Reproduce the §5 comparison from scratch with $N=300$, $\beta_\text{true}=1.5$, and missingness probability $\propto$ the observed $y$. Fit all three models (complete-case, mean-impute, joint Bayesian) and make the `az.plot_forest` of $\beta$ with a line at the truth. Then **vary the missingness fraction** from 10% to 70% and plot the three $\hat\beta$'s against the missing fraction. *Hint:* the complete-case and mean-impute biases should *grow* with the missing fraction while the joint model stays centered (with widening intervals). Explain in two sentences why the joint model's interval widens but its center holds.

3. **(Coding — inspect the imputations.)** In your joint model from Exercise 2, pull out `idata.posterior["x_unobserved"]` and (a) overlay a histogram of the imputed posterior means on a histogram of the *observed* $x$'s — are their distributions plausibly similar? (b) For the three rows with the largest observed $y$, plot the posterior of their imputed $x$ — do they lean high, as the regression ($\beta>0$) demands? *Hint:* this is the §4.1 diagnostic that proves the outcome is informing the imputation.

4. **(Coding — censoring vs. plug-in.)** Simulate $300$ draws from $\mathrm{Normal}(10, 4)$, then **right-censor** everything above $14$ (record those as exactly $14$). Fit two models: one that naively treats the $14$'s as real, and one using `pm.Censored(..., upper=14)`. Compare the posterior of $\mu$ and $\sigma$ from each to the truth. *Hint:* the naive model's $\hat\mu$ and $\hat\sigma$ should both be too small; explain why in terms of the survival-probability likelihood $1-F(14)$.

5. **(Coding — truncation.)** Now repeat Exercise 4 but **truncate** instead of censor: *discard* every draw above $14$ entirely (so your sample is smaller and you don't know how many you lost). Fit with `pm.Truncated(..., upper=14)` and with a naive `pm.Normal`. Which naive bias is worse, censoring-ignored or truncation-ignored, and why? *Hint:* think about what information each tool's renormalization recovers.

6. **(Stretch — selection model & sensitivity.)** Take the §5 synthetic data but now make the missingness depend on the **unobserved $x$** itself (so it's genuinely MNAR), e.g. high $x$ tends to go missing with no $y$-mediation. (a) Show that the §5.4 *MAR* joint model is now biased. (b) Build the §6.2 selection model with the missingness logit a function of the latent $x$, and run a sensitivity sweep over the prior scale of $\gamma_1$. (c) Plot $\hat\beta$ against the assumed selection strength and write a one-paragraph honest conclusion of the form "if selection is mild, $\beta\approx\ldots$; if strong, $\beta$ could be as large as $\ldots$." *Hint:* this is the real-world deliverable — a robustness statement, not a single number.

---

## 📚 Resources & further reading

**Foundational papers & books**
- **Rubin, D. B. (1976), "Inference and Missing Data," *Biometrika* 63(3):581–592** — the origin of MCAR/MAR/MNAR and the ignorability result. The one paper to read.
- **Little, R. J. A. & Rubin, D. B. (2019), *Statistical Analysis with Missing Data*, 3rd ed., Wiley** — the canonical book; Ch. 6 (likelihood/Bayes), Ch. 15 (MNAR / selection & pattern-mixture models).
- **Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, *Bayesian Data Analysis* (BDA3, 2013), Chapter 18, "Models for missing data"** — the Bayesian joint-modeling treatment; free PDF at stat.columbia.edu/~gelman/book/.
- **Rubin, D. B. (1987), *Multiple Imputation for Nonresponse in Surveys*** — the origin of MI and Rubin's rules.
- **Meng, X.-L. (1994), "Multiple-Imputation Inferences with Uncongenial Sources of Input," *Statistical Science* 9(4)** — why imputation and analysis models must be congenial (the risk the joint model avoids).
- **Little, R. J. A. (1992), "Regression with Missing X's: A Review," *JASA* 87(420):1227–1237** — the result behind §5's nuance: complete-case regression is unbiased when missingness depends only on the covariates, biased when it depends on the outcome.
- **White, I. R. & Carlin, J. B. (2010), "Bias and efficiency of multiple imputation compared with complete-case analysis for missing covariate values," *Statistics in Medicine* 29(28):2920–2931** — a modern, applied restatement of when complete-case is (and isn't) safe for missing covariates.
- **Daniels, M. J. & Hogan, J. W. (2008), *Missing Data in Longitudinal Studies: Strategies for Bayesian Modeling and Sensitivity Analysis*** — the reference for MNAR sensitivity analysis.

**PyMC docs, examples & tutorials**
- **Automatic Missing Data Imputation with PyMC (Chris Fonnesbeck, Strong Inference)** — http://stronginference.com/missing-data-imputation.html — the clearest explanation of the masked-array / NaN mechanism and the `_unobserved` node.
- **Missing Data (ISYE 6420, Aaron Reding — BUGS→PyMC port)** — https://areding.github.io/6420-pymc/unit6/Unit6-missingdata.html — worked MAR/MCAR imputation examples in current PyMC v5.
- **Bayesian regression with truncated or censored data (PyMC example gallery)** — https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-truncated-censored-regression.html — the definitive `pm.Censored` vs `pm.Truncated` regression notebook.
- **Censored Data Models (PyMC survival analysis gallery)** — https://www.pymc.io/projects/examples/en/latest/survival_analysis/censored_data.html — censoring through the survival-likelihood lens.
- **PyMC `pm.Censored` / `pm.Truncated` API** — https://www.pymc.io/projects/docs/en/stable/api/distributions.html — confirm `lower`/`upper`/`observed` semantics and the unnamed-`.dist()` requirement.

**Connections within this course**
- **McElreath, *Statistical Rethinking* (2nd ed., 2020), Chapter 15, "Missing Data and Other Opportunities"** — the best intuition-first treatment anywhere; builds the imputation-as-parameters idea with DAGs. PyMC port in `pymc-devs/pymc-resources`.
- **Martin, Kumar, Lao, *Bayesian Modeling and Computation in Python* (2021)** — free at **bayesiancomputationbook.com**; the ArviZ/PyMC workflow companion, including censored/truncated likelihoods.
- **Gelman et al., "Bayesian Workflow" (2020), arXiv:2011.01808** — frames sensitivity analysis and model expansion, the workflow scaffolding for the MNAR sensitivity sweep in §6.3.

---

## ➡️ What's next

You now treat a missing value the way you treat any unknown — as a parameter with a posterior — and you can tell, from the *mechanism*, whether you can impute it freely (MAR), must model the selection (MNAR), or are really facing its cousins, censoring and truncation. But notice what just happened to your models: every missing entry added a *latent parameter*, and a dataset with thousands of gaps adds thousands of them. Selection models add an entire second likelihood. These models get **big and slow** fast, and the default PyMC sampler — wonderful as it is — may no longer be enough.

That's the bridge to **Chapter 17 — Scaling & Advanced Computation**. There we swap in faster engines (NumPyro and BlackJAX on JAX, nutpie's Rust core) via the one-line `pm.sample(nuts_sampler=...)`, push models onto the GPU, parallelize *within* a chain in Stan with `reduce_sum`, and fit truly large datasets with minibatch ADVI. We'll also revisit the discrete-imputation problem from this chapter: the JAX backends can't sample discrete latents at all, which is one more reason marginalization (Chapter 13) is a scaling tool as much as a modeling one. You've learned to model honestly; next you'll learn to do it *fast*.
