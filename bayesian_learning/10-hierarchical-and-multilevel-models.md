# Chapter 10 — Hierarchical & Multilevel Models

### Partial pooling, shrinkage, varying effects, and the funnel that returns to test you

> *A note from me to you.*
>
> *If you take away one modeling idea from this entire course — one structural move that will change how you think about every dataset you ever touch — let it be this chapter. Almost all real data is **grouped**. Patients within hospitals, measurements within counties, repeated trials within subjects, sales within stores, students within schools. The naive practitioner faces a fork: either lump everyone together and pretend the groups don't exist (you lose all the local signal), or fit each group on its own and pretend they have nothing in common (you drown the small groups in noise). Both are wrong, and both are wrong in a way that **looks fine** until it bites you.*
>
> *The hierarchical model is the third path, and it is one of the deepest ideas in applied statistics. You let the groups be different — but you also let them **inform each other**, through a shared population they are all drawn from. The data themselves decide how much the groups should resemble each other. Small, noisy groups get pulled toward the crowd; large, well-measured groups stand their ground. This is **partial pooling**, and the pull is **shrinkage**, and once you see it you cannot unsee it. It is regularization, it is empirical Bayes, it is the James–Stein estimator, it is the reason your model generalizes — all the same idea wearing different hats.*
>
> *We will build it from the ground up on the radon dataset — the flagship example of multilevel modeling, Gelman and Hill's running case study — and we will watch small counties shrink toward the state mean. Then the villain from Chapter 07 returns: the **funnel**. Every hierarchical model is a funnel in disguise, and the cure is the move you already learned — **non-centering** — now applied as your default. By the end you will be able to write a correlated varying-slopes model with an LKJ prior from memory, predict for counties you've never seen, and explain to a skeptic exactly why your small-group estimates are more trustworthy than theirs. Let's build the crown jewel.*

---

## What you'll be able to do after this chapter

- **Place any grouped-data problem on the pooling spectrum** — complete pooling, no pooling, partial pooling — and explain the bias–variance tradeoff each makes.
- **Read the shrinkage picture** — see *why* small groups shrink toward the grand mean and large groups don't, and connect it to the James–Stein estimator and to regularization.
- **Build a varying-intercept model in PyMC v5** with named `coords`/`dims`, sensible hyperpriors on the group mean and group scale, and know why you choose `Exponential`/`HalfNormal` over `InverseGamma`.
- **Add a group-level predictor** (county uranium) that lives at the *intercept* level, and understand how covariates can enter at either level of the hierarchy.
- **Add varying slopes**, then make intercepts and slopes **correlated** via `pm.LKJCholeskyCov` + `pm.MvNormal`, and interpret the learned correlation.
- **Recognize and cure the hierarchical funnel** with the **non-centered parameterization** as your default — tied explicitly back to Chapter 07 — and know the honest exception: when each group has lots of data, **centered** can be better.
- **Predict for new, unseen groups** by drawing a fresh group effect from the learned population.
- **Use `pm.ZeroSumNormal`** to make group deflections identifiable against a global level.
- **Frame multilevel modeling as adaptive regularization** — the model that learns its own penalty.

Throughout, I assume you met the **funnel** and the **centered → non-centered reparameterization** as *concepts* and as a worked Eight Schools fix in **Chapter 07 (MCMC Diagnostics & Debugging)**; here we use them as everyday tools. I assume the GLM machinery and the Bambi formula API from **Chapter 09 (GLMs & Bambi)**. And I assume **standardization of predictors** as the default for setting priors, established in **Chapter 03 (Priors II)**.

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt          # v5: pytensor, NOT theano/aesara
import matplotlib.pyplot as plt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

---

## 1. The pooling spectrum: three ways to handle groups

Let me set the stage with the cleanest possible toy before the real data. Imagine you run a chain of coffee shops and you want to estimate the average rating of each shop. Some shops have 2,000 reviews; some opened last week and have 4. You want one number per shop. How do you get it?

There are exactly three philosophies, and they form a spectrum.

> 💡 **Intuition:** The whole question is *how much should one group's estimate borrow from the others?* The three classical answers are "borrow everything" (complete pooling), "borrow nothing" (no pooling), and "borrow an amount the data decide" (partial pooling). The last one is the hierarchical model, and it is almost always what you want.

Let the groups be indexed $j = 1, \dots, J$, with observations $y_{ij}$ for $i = 1, \dots, n_j$ in group $j$. We want a per-group location $\alpha_j$.

**Complete pooling — one global estimate.** Ignore the grouping entirely. Every observation is a draw from one shared distribution with one shared mean $\alpha$:

$$ y_{ij} \sim \mathrm{Normal}(\alpha, \sigma_y), \qquad \alpha \sim \mathrm{Normal}(0, 10). $$

```
complete pooling:   all groups  ──►  one α
```

This has **low variance** (you pool all the data into one estimate, so it's stable) but potentially **high bias** (if shops genuinely differ, you've erased every real difference). It throws away the entire reason you collected grouped data.

**No pooling — independent per group.** Fit each group on its own, as if the others never existed:

$$ y_{ij} \sim \mathrm{Normal}(\alpha_j, \sigma_y), \qquad \alpha_j \sim \mathrm{Normal}(0, 10)\ \text{independently for each } j. $$

```
no pooling:   group 1 ─► α₁     group 2 ─► α₂     ...     group J ─► αⱼ
              (no information flows between groups)
```

This has **low bias** (each group speaks for itself) but **high variance** for small groups (the 4-review shop's estimate is pure noise — one cranky customer swings it wildly). You will *overfit* every small group. The shop with 4 reviews and an unlucky 1-star will be ranked your worst location, which is nonsense.

**Partial pooling — the hierarchical compromise.** Here is the move. We say the group means are themselves *drawn from a common population*:

$$ y_{ij} \sim \mathrm{Normal}(\alpha_j, \sigma_y), \qquad \alpha_j \sim \mathrm{Normal}(\mu_\alpha, \sigma_\alpha). $$

```
partial pooling (hierarchical):

        μ_α, σ_α   (the population the groups share — estimated from data)
          │
          ├──► α₁    ├──► α₂    ...    ├──► αⱼ
          │          │                 │
        group 1    group 2           group J
```

Now $\mu_\alpha$ (the grand mean) and $\sigma_\alpha$ (how much groups differ) are **parameters we estimate from the data**, with their own *hyperpriors*. This is what makes the model *hierarchical*: parameters have priors, and those priors have parameters (hyperparameters) that themselves get priors. The structure is a small tower.

The magic is in $\sigma_\alpha$. If the data say the groups are nearly identical, $\sigma_\alpha$ is estimated small, and every $\alpha_j$ is pulled hard toward $\mu_\alpha$ — you approach **complete pooling**. If the data say the groups are wildly different, $\sigma_\alpha$ is large, the population prior is loose, and each $\alpha_j$ is free to follow its own data — you approach **no pooling**. The hierarchical model **interpolates between the two extremes, and the data choose where on that line you sit**, group by group.

> 🧮 **The math — partial pooling is a precision-weighted average.** For a single group with $n_j$ observations and sample mean $\bar y_j$, conditional on the hyperparameters the posterior mean of $\alpha_j$ is
> $$ \hat\alpha_j \;\approx\; \frac{\dfrac{n_j}{\sigma_y^2}\,\bar y_j \;+\; \dfrac{1}{\sigma_\alpha^2}\,\mu_\alpha}{\dfrac{n_j}{\sigma_y^2} \;+\; \dfrac{1}{\sigma_\alpha^2}}. $$
> Read this slowly — it is the entire chapter in one line. The estimate is a **weighted average of the group's own mean $\bar y_j$ and the population mean $\mu_\alpha$**, weighted by their *precisions* (inverse variances). The group's data carry weight $n_j/\sigma_y^2$ — proportional to its sample size. The population carries weight $1/\sigma_\alpha^2$. So:
> - **Large $n_j$** (data-rich group): the data weight dominates, $\hat\alpha_j \to \bar y_j$. The group barely moves. *No pooling, locally.*
> - **Small $n_j$** (data-poor group): the population weight dominates, $\hat\alpha_j \to \mu_\alpha$. The group is dragged toward the crowd. *Complete pooling, locally.*
>
> The same formula gives different amounts of pooling to different groups *in the same model*, automatically, with no tuning knob. That is the thing of beauty.

> 📜 **Citation/Origin:** This complete/no/partial trichotomy is the organizing frame of Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models* (2007), and of the canonical PyMC radon notebook (*A Primer on Bayesian Methods for Multilevel Modeling*). We'll follow their setup closely so you can read their material alongside this chapter.

---

## 2. Shrinkage, James–Stein, and why this is just regularization

The "pull toward the grand mean" has a name — **shrinkage** — and it is one of the most important and counterintuitive results in 20th-century statistics. Let me make it vivid with the example the field always uses: estimating a batch of means from small samples.

### The shrinkage picture

Suppose you have $J$ groups, each measured noisily, and you plot two things on a number line: the **no-pooling** estimate of each group (its own raw mean) and the **partial-pooling** estimate (after the hierarchical model). You connect each pair with an arrow. The picture always looks like this:

```
no pooling   (raw group means, spread out)
  •      •   •     •         •      •    •        •
  │      │   │     │         │      │    │        │
  ▼      ▼   ▼     ▼         ▼      ▼    ▼        ▼     ← arrows pull inward
     •   •   •  •      • •  •   •   •  •                 (toward grand mean μ_α)
partial pooling  (shrunk toward the middle)
                    ▲
                  grand mean μ_α
```

Every estimate moves **inward**, toward $\mu_\alpha$ — but not by the same amount. The arrows are *long* for small/noisy groups and *short* for large/precise groups. **The groups with the least information get pulled the most.** That is shrinkage, and it is exactly the precision-weighting formula above made visual.

> 💡 **Intuition — why shrinkage is *correct*, not a hack.** A small group with an extreme raw mean is almost certainly extreme *partly because it got unlucky*, not purely because it's truly extreme — that's regression to the mean. The hierarchical model knows this: it has seen all the other groups, it knows roughly how spread-out true group means are (that's $\sigma_\alpha$), and it knows how noisy each measurement is (that's $\sigma_y / \sqrt{n_j}$). So it discounts extreme small-sample estimates *exactly* in proportion to how likely they are to be flukes. The shrunken estimate is a **better guess about the true group value** than the raw mean — provably, on average.

### James–Stein: shrinkage dominates the obvious estimator

Here is the result that stunned statisticians in 1961. Suppose you want to estimate $J \ge 3$ unknown means from one noisy observation each. The "obvious" estimator is to just use each observation as your estimate (the MLE, the no-pooling estimate). Charles Stein proved that this obvious estimator is **inadmissible**: there exists another estimator that has *lower total squared error than the MLE, for every possible true set of means, simultaneously.* And that better estimator is a **shrinkage** estimator — it pulls all the observations toward a common point.

This was scandalous. It says: even if you are estimating the price of tea in China, the diameter of Saturn, and the batting average of a baseball player — three utterly unrelated quantities — you do better, in total squared error, by shrinking all three toward their common average than by reporting each measurement as-is. The "borrowing of strength" works even across unrelated quantities.

> 📜 **Citation/Origin:** Stein (1956) and James & Stein (1961). The hierarchical/empirical-Bayes *interpretation* — that James–Stein shrinkage is the posterior mean under a Normal population prior whose variance is estimated from the data — is due to Efron & Morris (1970s), who illustrated it with the now-classic **baseball batting averages** example: early-season averages from a handful of at-bats, shrunk toward the league mean, predict end-of-season averages far better than the raw early averages do. The radon model you're about to build is the same machine pointed at houses instead of hitters.

> 💡 **Intuition — the unifying view: multilevel modeling *is* adaptive regularization.** In ridge regression you add a penalty $\lambda \|\beta\|^2$ that pulls coefficients toward zero, and you tune $\lambda$ by cross-validation. The hierarchical prior $\alpha_j \sim \mathrm{Normal}(\mu_\alpha, \sigma_\alpha)$ does *exactly the same thing* — it pulls the $\alpha_j$ toward $\mu_\alpha$ with a strength governed by $1/\sigma_\alpha^2$, which plays the role of $\lambda$. The difference, and it is a beautiful one: you **don't tune the penalty by cross-validation — the model learns $\sigma_\alpha$ from the data, as a parameter, with full uncertainty propagated.** The hierarchy *is* a regularizer that fits its own strength. Hold onto this. It's the deepest sentence in the chapter: a multilevel model is a regularized model that learns how hard to regularize.

> ⚠️ **Pitfall — "but shrinkage biases my estimates!"** Yes, deliberately. Each individual $\hat\alpha_j$ is a *biased* estimate of that group's true value. But bias is not the enemy; **total error** is. Shrinkage trades a little bias for a large reduction in variance, and the variance reduction wins — that's the whole James–Stein theorem. If your stakeholder objects that "the small store's estimate was pulled up unfairly," the honest answer is: its raw estimate was almost certainly too low *by luck*, and the shrunk estimate will be closer to the truth more often than not. You are not being unfair; you are being calibrated.

---

## 3. The radon dataset: our flagship

Time to make this concrete on the dataset that *is* multilevel modeling in the textbooks. **Radon** is a radioactive, carcinogenic gas that seeps from uranium-bearing soil into buildings through the basement. The U.S. EPA measured radon levels in ~12,000 houses to understand the risk. We'll use the **Minnesota** subset (Gelman & Hill's running example): about 900 houses spread across **85 counties**.

Our questions:
1. What's the radon level in a typical house in each county? (a per-county intercept)
2. Does the floor of measurement matter? Basement readings should be higher than first-floor readings, since the gas enters from below. (a slope on `floor`)
3. Does the county's soil uranium predict its radon? (a *county-level* predictor — covariates can live at the group level too)

The grouping is **houses within counties**, and the counties vary enormously in sample size: some counties have dozens of measured houses, many have just one or two. That heterogeneity is *exactly* the regime where partial pooling earns its keep.

> 🔧 **In practice — loading and preprocessing radon.** The data ship with PyMC via `pm.get_data`. This is the canonical Gelman/Hill preprocessing; copy it carefully — the silent bugs in multilevel modeling almost always come from a botched county→row index.

```python
# --- Load the Minnesota radon data (ships with PyMC) ---
srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))
srrs2.columns = srrs2.columns.map(str.strip)              # tidy stray whitespace
srrs_mn = srrs2[srrs2.state == "MN"].copy()
srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips  # county FIPS code

# --- County-level uranium, merged in by FIPS ---
cty = pd.read_csv(pm.get_data("cty.dat"))
cty_mn = cty[cty.st == "MN"].copy()
cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips
srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips").drop_duplicates(subset="idnum")
srrs_mn.county = srrs_mn.county.map(str.strip)

# --- Build a clean, contiguous county index (this is where bugs hide) ---
mn_counties   = srrs_mn.county.unique()                    # 85 county names
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
county        = srrs_mn.county.replace(county_lookup).values   # integer 0..84 per house

# --- Outcomes and predictors ---
log_radon     = np.log(srrs_mn.activity + 0.1).values      # +0.1 avoids log(0)
floor_measure = srrs_mn.floor.values                       # 0 = basement, 1 = first floor
u             = np.log(srrs_mn.Uppm.values)                # county-level log-uranium

n_counties = len(mn_counties)
print(n_counties, "counties;", len(log_radon), "houses")   # -> 85 counties; ~919 houses
```

A few modeling decisions worth pausing on:

- **We model `log(radon)`.** Radon is a positive, right-skewed concentration. On the log scale it's roughly Normal, the floor effect is multiplicative (a sensible assumption for a gas concentration), and our Gaussian likelihood is appropriate. The `+0.1` is a standard nudge so that the handful of zero readings don't become $-\infty$.
- **`floor = 0` is basement, `floor = 1` is first floor.** We expect a **negative** slope: first-floor readings are *lower* because the gas enters from the ground. Don't trust me — let the model tell us, and check the sign.
- **`u = log(Uppm)` is a *county-level* predictor.** Every house in a county shares the same $u$. It will enter the model at the **intercept** level — it helps explain *why counties differ* — not at the house level. This is the canonical illustration that predictors can live at any level of a hierarchy.

> ⚠️ **Pitfall — the silent index bug.** The single most common way to get a *wrong but plausible-looking* hierarchical model is to mismatch the county index. If `county` is built from an unsorted or non-contiguous factor, then `alpha[county_idx]` silently pairs houses with the wrong county's intercept, and your model fits beautifully while being nonsense. The defense is the explicit `county_lookup = dict(zip(mn_counties, range(len(mn_counties))))` above, which guarantees a contiguous `0..J-1` index, plus a sanity check: spot-check that `mn_counties[county[k]] == srrs_mn.county.iloc[k]` for a few rows.

---

## 4. The three models, in PyMC v5, with coords and dims

Let's fit all three pooling regimes on radon so you can *see* the spectrum, not just read about it. From here on we use PyMC v5's **named `coords` and `dims`** religiously — they make every plot and summary self-documenting (you get `alpha[AITKIN]` instead of `alpha[3]`), and they prevent shape bugs. Set up the coordinate dictionary once:

```python
coords = {
    "county": mn_counties,                       # 85 county names
    "obs_id": np.arange(len(log_radon)),         # one entry per house
}
```

### 4a. Complete pooling — one intercept for all of Minnesota

```python
with pm.Model(coords=coords) as pooled_model:
    floor_idx = pm.Data("floor_idx", floor_measure, dims="obs_id")

    # One global intercept and one global slope, shared by every house in the state.
    alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
    beta  = pm.Normal("beta",  mu=0.0, sigma=10.0)
    sigma = pm.Exponential("sigma", 1.0)

    mu = alpha + beta * floor_idx
    y  = pm.Normal("y", mu=mu, sigma=sigma, observed=log_radon, dims="obs_id")

    idata_pooled = pm.sample(1000, tune=1000, chains=4,
                             target_accept=0.9, random_seed=RANDOM_SEED)
```

This model has exactly three parameters: a global intercept, a global floor slope, and a residual scale. Every county shares them. Aitkin County and Hennepin County (which contains Minneapolis) get the *identical* prediction. That's the complete-pooling assumption laid bare — and it's clearly too strong, because counties really do differ in their soil.

### 4b. No pooling — an independent intercept per county

```python
with pm.Model(coords=coords) as unpooled_model:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    floor_idx  = pm.Data("floor_idx", floor_measure, dims="obs_id")

    # An independent intercept for every county; NO shared population.
    alpha = pm.Normal("alpha", mu=0.0, sigma=10.0, dims="county")
    beta  = pm.Normal("beta",  mu=0.0, sigma=10.0)             # shared floor slope
    sigma = pm.Exponential("sigma", 1.0)

    mu = alpha[county_idx] + beta * floor_idx
    y  = pm.Normal("y", mu=mu, sigma=sigma, observed=log_radon, dims="obs_id")

    idata_unpooled = pm.sample(1000, tune=1000, chains=4,
                               target_accept=0.9, random_seed=RANDOM_SEED)
```

The single change that matters is `dims="county"` on `alpha`: now there are 85 intercepts, one per county, and `alpha[county_idx]` looks up the right one for each house. But crucially, the 85 intercepts have **independent** $\mathrm{Normal}(0, 10)$ priors — they cannot see each other. A county with one measured house gets an intercept estimated from that one house, with enormous uncertainty and no help from its neighbors.

> ⚠️ **Pitfall:** In the no-pooling model, counties with a single house produce posterior intercepts with huge credible intervals, and counties with *zero* houses can't be estimated at all. The diffuse $\mathrm{Normal}(0,10)$ prior barely regularizes. This is the overfitting-small-groups failure mode in the flesh — note it now, because partial pooling will fix it.

### 4c. Partial pooling — the hierarchical varying-intercept model

Now the real thing. We give the 85 county intercepts a **shared population** with estimated mean and scale:

$$
\begin{aligned}
y_i &\sim \mathrm{Normal}(\alpha_{j[i]} + \beta\,\mathrm{floor}_i,\ \sigma_y) \\
\alpha_j &\sim \mathrm{Normal}(\mu_\alpha,\ \sigma_\alpha) \qquad j = 1,\dots,85 \\
\mu_\alpha &\sim \mathrm{Normal}(0, 10), \quad \sigma_\alpha \sim \mathrm{Exponential}(1), \quad \sigma_y \sim \mathrm{Exponential}(1)
\end{aligned}
$$

```python
with pm.Model(coords=coords) as partial_centered:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    floor_idx  = pm.Data("floor_idx", floor_measure, dims="obs_id")

    # --- Hyperpriors: the population the county intercepts are drawn from ---
    mu_a    = pm.Normal("mu_a", mu=0.0, sigma=10.0)       # grand mean intercept
    sigma_a = pm.Exponential("sigma_a", 1.0)              # how much counties differ

    # --- County intercepts drawn from that population (CENTERED form) ---
    alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")

    beta    = pm.Normal("beta", mu=0.0, sigma=10.0)       # shared floor slope
    sigma_y = pm.Exponential("sigma_y", 1.0)              # within-county scatter

    mu = alpha[county_idx] + beta * floor_idx
    y  = pm.Normal("y", mu=mu, sigma=sigma_y, observed=log_radon, dims="obs_id")

    idata_partial = pm.sample(1000, tune=1000, chains=4,
                              target_accept=0.9, random_seed=RANDOM_SEED)
```

Look at what's new compared to no-pooling: the `alpha` intercepts no longer have a fixed `mu=0, sigma=10` prior. Instead their prior is `mu=mu_a, sigma=sigma_a` — **and those two are themselves random variables we estimate.** That single change wires all 85 counties together. They now share a population, and that population's scale `sigma_a` controls how much they shrink toward `mu_a`.

> ⚠️ **Pitfall — keep an eye on the divergence count.** This `partial_centered` model is written in the **centered** parameterization (the county intercept is drawn *directly* from `Normal(mu_a, sigma_a)`). On radon, with most counties having only a handful of houses, this is precisely the **funnel** regime from Chapter 07, and you may well see a divergence warning like `There were 31 divergences after tuning`. **Do not ignore it.** We are going to fix it properly in §6 with non-centering — but I want you to fit the centered form first and *see* the warning, because recognizing it in the wild is the skill that matters. (If your run happens to come back clean, count yourself lucky and reparameterize anyway; the funnel is lurking.)

### Comparing the three: the shrinkage plot

Now the payoff. Let's extract each model's county intercepts and plot the no-pooling estimates against the partial-pooling estimates, sized by county sample size. This is *the* picture of the chapter.

```python
# Per-county sample sizes
counts = srrs_mn.groupby("county").size().reindex(mn_counties).values

# No-pooling posterior-mean intercepts (one per county)
a_unpooled = idata_unpooled.posterior["alpha"].mean(("chain", "draw")).values
# Partial-pooling posterior-mean intercepts
a_partial  = idata_partial.posterior["alpha"].mean(("chain", "draw")).values
# The estimated grand mean (the point everything shrinks toward)
grand_mean = idata_partial.posterior["mu_a"].mean().item()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(counts, a_unpooled, label="no pooling",      alpha=0.6)
ax.scatter(counts, a_partial,  label="partial pooling", alpha=0.6, marker="x")
for c, au, ap in zip(counts, a_unpooled, a_partial):
    ax.plot([c, c], [au, ap], color="grey", lw=0.5)      # arrow: raw -> shrunk
ax.axhline(grand_mean, ls="--", color="k", label=r"grand mean $\mu_\alpha$")
ax.set_xscale("log"); ax.set_xlabel("houses in county (log scale)")
ax.set_ylabel("estimated county intercept"); ax.legend()
```

> 🩺 **Diagnostic — reading the shrinkage plot.** The x-axis is the county sample size (log scale); the y-axis is the estimated intercept. Each grey segment connects a county's no-pooling estimate to its partial-pooling estimate. You will see the unmistakable signature: **on the left (small counties), the segments are long — those estimates get yanked hard toward the dashed grand-mean line. On the right (large counties), the segments are tiny — those estimates barely move.** The no-pooling estimates for small counties are scattered all over the place (some absurdly high or low, from one or two unlucky houses); after partial pooling they collapse into a sensible band around the grand mean. **This is the precision-weighting formula from §1 rendered as a picture.** Small counties have little data, so the population prior dominates and pulls them in; large counties have lots of data, so their own data dominate and they stand firm. The model gave each county *exactly* the amount of regularization its sample size warranted — and you never told it to.

> 💡 **Intuition — which estimate would you bet on?** For a county with two measured houses, the no-pooling estimate is "whatever those two houses said." The partial-pooling estimate is "those two houses, tempered by what the other 83 counties told us about how spread-out counties typically are." If you had to predict the radon in a *third*, unmeasured house in that county, which estimate would you trust? The shrunken one, every time. That's not a philosophical preference — it's lower expected prediction error, and you can verify it with the LOO cross-validation tools from **Chapter 11 (Model Comparison & Evaluation)**.

---

## 5. Choosing hyperpriors, and a prior predictive sanity check

Before we go further, two things deserve a careful word: the **scale hyperprior** $\sigma_\alpha$, and the habit of *checking your priors before fitting*.

### The hyperprior on the group scale is the consequential choice

In a hierarchical model, $\sigma_\alpha$ is the single most important prior. It controls the degree of pooling: a tight prior near zero forces heavy pooling (toward complete pooling); a diffuse prior permits the groups to fly apart (toward no pooling). It deserves thought, not a reflex.

> 🔧 **In practice — what to put on a group scale $\sigma_\alpha$.** Use a **`HalfNormal`**, **`Exponential`**, or **`HalfCauchy`** — distributions with support on $(0, \infty)$ that put gentle mass near zero (allowing strong pooling) while letting the data pull the scale up if groups genuinely differ. On standardized data, `pm.Exponential("sigma_a", 1.0)` (mean 1) or `pm.HalfNormal("sigma_a", 1.0)` are excellent defaults. `HalfCauchy(5)` is the historically popular weakly-informative choice but its heavy tail can slow mixing — prefer `Exponential`/`HalfNormal` unless you have a reason.

> ⚠️ **Pitfall — never use `InverseGamma(ε, ε)` on a variance.** The old BUGS-era default `InverseGamma(0.001, 0.001)` (a "diffuse" prior on the *variance*) is actively harmful in hierarchical models: it has a spike near zero in a place that interacts badly with the likelihood, biases $\sigma_\alpha$ downward, and can wreck your sampling when the true between-group variance is small (which is *exactly* the hard, funnel-prone regime). Andrew Gelman's 2006 paper *"Prior distributions for variance parameters in hierarchical models"* is the definitive takedown; the practical upshot is **put your prior on the standard deviation $\sigma_\alpha$ directly, with a half-Normal / Exponential / half-Cauchy**, never `InverseGamma(ε,ε)` on the variance. (Recall from **Chapter 05** the precision/sd footgun: BUGS and JAGS parameterized Normals by *precision* $\tau = 1/\sigma^2$; PyMC and Stan use the **standard deviation** $\sigma$. State this loudly to anyone porting old code.)

> 📜 **Citation/Origin:** Gelman (2006), *Bayesian Analysis* 1(3). The folk-theorem reading: when the number of groups $J$ is small (say $J < 5$), the prior on $\sigma_\alpha$ genuinely matters and you should think hard; with $J = 85$ counties the data inform $\sigma_\alpha$ well and a sensible weakly-informative prior suffices.

### Prior predictive check

We never fit blind. Before sampling the posterior, draw from the *prior predictive* — simulate fake datasets from the priors alone — and check they're in a physically sane range. This is the **prior predictive check** from **Chapter 03 (Priors II)** and **Chapter 08 (The Bayesian Workflow)**, now applied to a hierarchical model.

```python
with partial_centered:
    # v5: the kwarg is `draws` (the v3/v4 `samples=` is deprecated and will be removed)
    prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)

az.plot_ppc(prior_pred, group="prior", num_pp_samples=200)
```

> 🩺 **Diagnostic — reading the prior predictive.** `az.plot_ppc(..., group="prior")` overlays many simulated `log_radon` densities drawn purely from the priors. You're not checking fit here (there's no data involved yet) — you're checking *plausibility of scale*. Recall observed `log_radon` runs roughly from $-1$ to $4$. If your prior predictive produced log-radon values in the hundreds, your priors are absurdly wide and you'd want to tighten them. Be honest about what `mu_a ~ Normal(0, 10)` actually implies: on the *log* scale a Normal(0, 10) intercept permits excursions of $\pm 20$ or more — radon spanning many orders of magnitude — so the prior predictive here is genuinely **very broad**, far wider than the observed range, not merely "comfortably covering" it. That is fine in *this* problem precisely because we have ~900 houses across 85 counties: the likelihood overwhelms such a diffuse prior, and the canonical Gelman/Hill radon notebook uses these same wide priors. The point of the check is to confirm the scale isn't *insane* (no values in the hundreds), not that it's tight. If you wanted a genuinely tighter, more self-documenting prior you would **standardize the predictors and the outcome** (the Chapter 03 default) and use something like `Normal(0, 2.5)` on standardized scales — re-running this prior predictive then gives a markedly sharper, more honest spread. The prior predictive is your honest reply to "did you peek at the answer to set your priors?" — these were chosen for *plausibility of scale*, then the data did the rest.

---

## 6. Adding a group-level predictor: uranium enters the intercept

Counties don't differ at random — they differ because their *soil* differs. We have a county-level measurement, log-uranium `u`, and physics says high-uranium soil should mean high radon. Let's let the model use it. The crucial point: `u` varies **by county, not by house**, so it belongs in the model of the **intercept**, not in the house-level equation. This is the textbook illustration that *predictors can enter at any level of a hierarchy*.

$$
\begin{aligned}
y_i &\sim \mathrm{Normal}(\alpha_{j[i]} + \beta\,\mathrm{floor}_i,\ \sigma_y) \\
\alpha_j &\sim \mathrm{Normal}(\gamma_0 + \gamma_1\, u_j,\ \sigma_\alpha)
\end{aligned}
$$

Read the second line: the county intercept's *population mean* is no longer a flat $\mu_\alpha$; it's a **regression on uranium**, $\gamma_0 + \gamma_1 u_j$. So $\sigma_\alpha$ now measures the *residual* between-county variation **after** uranium has explained what it can. We expect $\sigma_\alpha$ to *shrink* relative to the previous model — uranium soaks up some of the between-county differences — and we expect $\gamma_1 > 0$.

```python
# u is per-house in the dataframe but constant within county; collapse to one value per county:
u_county = srrs_mn.groupby("county")["Uppm"].first().reindex(mn_counties).values
u_county = np.log(u_county)                      # county-level log-uranium, length 85

# No new coord needed: `u_data` rides on the existing "county" dim, so reuse `coords` directly.
with pm.Model(coords=coords) as varying_intercept_u:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    floor_idx  = pm.Data("floor_idx", floor_measure, dims="obs_id")
    u_data     = pm.Data("u", u_county, dims="county")    # one uranium value per county

    # The county intercept's MEAN is now a line in uranium:
    gamma0  = pm.Normal("gamma0", 0.0, 10.0)             # intercept of the county-level line
    gamma1  = pm.Normal("gamma1", 0.0, 10.0)             # uranium slope (expect > 0)
    sigma_a = pm.Exponential("sigma_a", 1.0)             # residual between-county sd

    # --- NON-CENTERED already (see §7); we'll motivate this fully in a moment ---
    z_a   = pm.Normal("z_a", 0.0, 1.0, dims="county")
    alpha = pm.Deterministic("alpha", gamma0 + gamma1 * u_data + z_a * sigma_a, dims="county")

    beta    = pm.Normal("beta", 0.0, 10.0)
    sigma_y = pm.Exponential("sigma_y", 1.0)

    mu = alpha[county_idx] + beta * floor_idx
    y  = pm.Normal("y", mu=mu, sigma=sigma_y, observed=log_radon, dims="obs_id")

    idata_u = pm.sample(1000, tune=1000, chains=4,
                        target_accept=0.9, random_seed=RANDOM_SEED)

az.summary(idata_u, var_names=["gamma0", "gamma1", "sigma_a", "beta", "sigma_y"])
```

> 🩺 **Diagnostic — reading the uranium effect.** In the summary you'll find `gamma1` with a clearly positive mean and a 94% HDI that excludes zero — high-uranium counties really do have higher radon, as physics demands. And `sigma_a` will be **smaller** than in the model without uranium: once the county-level predictor explains the systematic part of the between-county differences, less residual variation is left for the random intercepts to absorb. That's a group-level predictor *doing its job* — it's not just decoration, it tightens the hierarchy. The `beta` (floor) slope will be solidly negative (around $-0.6$ on the log scale): first-floor readings are lower, exactly as the gas-from-below mechanism predicts.

> 💡 **Intuition — two levels, two kinds of predictor.** `floor` is a **house-level** predictor: it varies *within* a county (some measured rooms are basements, some first floors), so it sharpens predictions *inside* each county. `uranium` is a **county-level** predictor: it's constant within a county, so it can only explain why *counties* differ. Hierarchical models let you use both, cleanly, in their proper places. A flat (non-hierarchical) regression would force you to either ignore the county structure or to one-hot-encode 85 dummy variables and lose all pooling. The multilevel model is the principled home for "predictors at different levels of aggregation."

> ⚠️ **Pitfall — a county-level predictor does *not* reduce the need for the random intercept.** A common misread is "I added uranium, so I can drop the varying intercept." No: uranium explains the *mean* of the county intercepts, but there's still residual county-to-county variation ($\sigma_\alpha > 0$) that the random intercept must absorb — soil chemistry, construction styles, measurement campaigns. Keep both. The group-level predictor and the random intercept are partners, not substitutes.

---

## 7. The funnel returns — non-centering as your default

You'll have noticed I quietly wrote the uranium model in the **non-centered** form (the `z_a * sigma_a` line) without yet explaining why. Now we pay that debt. This is the most important *computational* section of the chapter, and it's the direct sequel to Chapter 07.

### Why every hierarchical model is a funnel

Go back to the centered partial-pooling model (§4c). The defining line was:

```python
alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")   # centered
```

The **width** of each `alpha[j]` is `sigma_a` — a parameter we are *also* estimating. That coupling is the funnel. When the sampler proposes a small `sigma_a`, the 85 `alpha` values are squeezed into a razor-thin neck near `mu_a`; when it proposes a large `sigma_a`, they spread out into a wide mouth. The joint posterior of $(\alpha_j, \log\sigma_\alpha)$ is the **funnel** geometry you met in Chapter 07: the local curvature changes by orders of magnitude from mouth to neck, so a single HMC step size cannot fit both. Steps tuned for the wide mouth overshoot catastrophically in the narrow neck, and the leapfrog integrator **diverges**.

```
 log σ_α
   │        .  .  .  .  .  .  .  .       ← wide mouth (large σ_α): α's spread out
   │       .                    .
   │        .  .  .  .  .  .  .
   │           .  .  .  .  .            ← curvature increasing as σ_α shrinks
   │              .  .  .
   │                ✦✦✦                 ← NECK (small σ_α): divergences cluster here
   └──────────────────────────► α_j
```

> 🩺 **Diagnostic — divergences are truth-tellers, and they cluster in the neck.** This is Betancourt's central message and the lesson of Chapter 07: divergences are **not** random noise to be silenced. They mark the region the sampler physically could not traverse — almost always the **funnel neck where $\sigma_\alpha$ is small.** Localize them the same way you did for Eight Schools: add `pm.Deterministic("log_sigma_a", pt.log(sigma_a))` to the model, then `az.plot_pair(idata_partial, var_names=["log_sigma_a", "alpha"], coords={"county": ["AITKIN"]}, divergences=True)`. The divergent draws (highlighted) will pile up in the bottom of the funnel — small `log_sigma_a` — proving the diagnosis. **The spatial pattern of divergences *is* the diagnosis.** A handful of divergences is not "close enough"; it means an entire low-variance region was systematically under-explored, biasing every estimate that depends on it (including, ironically, $\sigma_\alpha$ itself, which gets pushed away from the small values it can't reach).

### The cure: non-centered parameterization

The fix is the move you already own from Chapter 07. Don't draw `alpha` *directly* from `Normal(mu_a, sigma_a)` — which welds its width to `sigma_a`. Instead draw a **standardized** offset on a fixed unit-scale geometry, then *recover* `alpha` with a deterministic affine map:

$$
\tilde\alpha_j \sim \mathrm{Normal}(0, 1), \qquad \alpha_j = \mu_\alpha + \sigma_\alpha \, \tilde\alpha_j .
$$

In ASCII, the one line that cures the funnel:

```
alpha = mu_a + z_a * sigma_a       where  z_a ~ Normal(0, 1)
```

```python
with pm.Model(coords=coords) as partial_noncentered:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    floor_idx  = pm.Data("floor_idx", floor_measure, dims="obs_id")

    mu_a    = pm.Normal("mu_a", 0.0, 10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # --- NON-CENTERED: standardized offset, then affine recovery ---
    z_a   = pm.Normal("z_a", 0.0, 1.0, dims="county")               # unit-scale, fixed geometry
    alpha = pm.Deterministic("alpha", mu_a + z_a * sigma_a, dims="county")

    beta    = pm.Normal("beta", 0.0, 10.0)
    sigma_y = pm.Exponential("sigma_y", 1.0)

    mu = alpha[county_idx] + beta * floor_idx
    y  = pm.Normal("y", mu=mu, sigma=sigma_y, observed=log_radon, dims="obs_id")

    idata_nc = pm.sample(1000, tune=1000, chains=4,
                         target_accept=0.9, random_seed=RANDOM_SEED)
```

> 🧮 **The math — why this is the *same* model.** If $\tilde\alpha_j \sim \mathrm{Normal}(0,1)$ then the affine transform $\alpha_j = \mu_\alpha + \sigma_\alpha \tilde\alpha_j$ is distributed *exactly* $\mathrm{Normal}(\mu_\alpha, \sigma_\alpha)$ — that's the location–scale property of the Gaussian. So the posterior $p(\mu_\alpha, \sigma_\alpha, \alpha \mid y)$ is mathematically identical to the centered one; the marginal of every quantity you care about is unchanged. What changes is the *coordinates the sampler crawls over*. The parameter NUTS actually moves now is $\tilde\alpha_j$ (the `z_a`), and **its geometry is a fixed unit Gaussian that does not depend on $\sigma_\alpha$ at all.** The coupling that created the funnel is gone. We pay only one cheap deterministic multiply.

> 🩺 **Diagnostic — watch every number go green.** Re-run the diagnostic loop on `idata_nc`:
> ```python
> print("divergences:", int(idata_nc.sample_stats["diverging"].sum()))   # -> 0
> az.summary(idata_nc, var_names=["mu_a", "sigma_a", "beta", "sigma_y"])  # R-hat ~ 1.00, ESS high
> az.plot_energy(idata_nc)                                                # marginal & transition overlap
> ```
> The divergence count drops to **0**. $\hat R \le 1.01$ across the board. ESS_bulk and ESS_tail jump — often by a large factor on `sigma_a`, the parameter the centered sampler couldn't explore — into the thousands. The energy plot (BFMI) shows the marginal and transition energy distributions overlapping nicely (BFMI comfortably above $0.3$). And critically, the **posterior of `sigma_a` now reaches the small values** the centered sampler kept diverging away from — so your between-county variation is estimated honestly rather than biased upward. The model didn't change; the coordinates did, and that was enough.

> 💡 **Intuition — what you actually did.** You didn't change *what* you're inferring, you changed *how the sampler walks over it*. The centered coordinates are a map drawn on a sheet pinched to a point — the sampler keeps tumbling into the crease. The non-centered coordinates flatten the sheet back out, and the same landscape becomes an easy stroll. **Non-centering is the art of choosing coordinates in which the posterior geometry is benign**, and it is your *default* for hierarchical models. Reach for it first.

> 📜 **Citation/Origin:** The non-centered ("offset") parameterization is the **"Matt trick"** (named for Matt Hoffman on the stan-users list); the geometry is in Betancourt & Girolami, *"Hamiltonian Monte Carlo for Hierarchical Models"* (2013, arXiv:1312.0906), and Thomas Wiecki's blog post *"Why hierarchical models are awesome, tricky, and Bayesian"* is the friendliest visual walkthrough. Same algebra, cross-language, is in the Stan User's Guide reparameterization chapter. We introduced it in Chapter 07 on Eight Schools; here it's the everyday tool.

### The honest exception: when centered is actually *better*

Here is the nuance that separates someone who memorized "always non-center" from someone who understands it. **Non-centered is not universally better.** Which parameterization wins is governed by *whether the prior or the likelihood dominates each group's posterior* (Betancourt & Girolami):

| Regime | What it means | Best parameterization |
|---|---|---|
| **Prior-dominated** | Few observations per group, small $\sigma_\alpha$, sparse data | **Non-centered** wins decisively |
| **Likelihood-dominated** | Many observations per group, large $\sigma_\alpha$, data-rich | **Centered** is more efficient |

> 💡 **Intuition.** When a group has lots of data, the likelihood pins its $\alpha_j$ down tightly *regardless* of $\sigma_\alpha$ — the funnel never forms, the posterior is already well-conditioned, and the non-centered coordinates just introduce their *own* awkward geometry (now it's $\tilde\alpha_j$ that gets coupled to the data in an inconvenient way). In that regime centered samples faster with higher ESS. When a group is data-starved, the prior dominates, the funnel bites hard, and non-centered is the clear winner.
>
> **The practitioner's rule:** default to non-centered for sparse / weakly-informed hierarchies (the usual case, and the radon case — most counties have a handful of houses). But in *big-data-per-group* settings, **test the centered form** — it may sample faster. And with heterogeneous group sizes you can even **mix**: centered for the data-rich groups, non-centered for the data-poor ones (this is more advanced and rarely necessary, but it's the logical endpoint). Radon, with most counties tiny, is squarely a non-center-by-default problem — which is why every model in this chapter from here on is non-centered.

> ⚠️ **Pitfall — don't "fix" divergences by lowering `target_accept` until the warning stops.** That hides the bias instead of removing it. The legitimate responses to a hierarchical funnel are: (1) **non-center** (the real fix), then if a few divergences remain, (2) raise `target_accept` $0.9 \to 0.95 \to 0.99$ to shrink the step size, (3) lengthen `tune`. If you're at `target_accept=0.99` and *still* diverging after non-centering, suspect a *second* funnel — e.g. correlated varying slopes (next section) that *also* need non-centering — not a reason to crank `target_accept` to 0.999.

---

## 8. Varying slopes: letting the floor effect vary by county

So far only the **intercept** varies by county; the floor slope `beta` is shared statewide. But why should the floor effect be identical everywhere? Some counties' housing stock, soil, or basement construction might make the basement-vs-first-floor gap larger or smaller. Let's let the **slope vary by county too** — a *varying-slopes* model.

The naive way is to give each county its own intercept *and* its own slope, each with an independent hierarchical prior:

$$
\begin{aligned}
y_i &\sim \mathrm{Normal}(\alpha_{j[i]} + \beta_{j[i]}\,\mathrm{floor}_i,\ \sigma_y) \\
\alpha_j &\sim \mathrm{Normal}(\mu_\alpha, \sigma_\alpha), \qquad \beta_j \sim \mathrm{Normal}(\mu_\beta, \sigma_\beta).
\end{aligned}
$$

```python
with pm.Model(coords=coords) as varying_slopes:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    floor_idx  = pm.Data("floor_idx", floor_measure, dims="obs_id")

    mu_a, sigma_a = pm.Normal("mu_a", 0.0, 10.0), pm.Exponential("sigma_a", 1.0)
    mu_b, sigma_b = pm.Normal("mu_b", 0.0, 10.0), pm.Exponential("sigma_b", 1.0)

    # Non-centered for BOTH intercept and slope (two funnels => two cures):
    z_a   = pm.Normal("z_a", 0.0, 1.0, dims="county")
    z_b   = pm.Normal("z_b", 0.0, 1.0, dims="county")
    alpha = pm.Deterministic("alpha", mu_a + z_a * sigma_a, dims="county")
    beta  = pm.Deterministic("beta",  mu_b + z_b * sigma_b, dims="county")

    sigma_y = pm.Exponential("sigma_y", 1.0)
    mu = alpha[county_idx] + beta[county_idx] * floor_idx
    y  = pm.Normal("y", mu=mu, sigma=sigma_y, observed=log_radon, dims="obs_id")

    idata_vs = pm.sample(1000, tune=1000, chains=4,
                         target_accept=0.9, random_seed=RANDOM_SEED)
```

> ⚠️ **Pitfall — two random effects means two funnels.** Notice we non-centered *both* `alpha` and `beta`. A model with $K$ varying effects has $K$ potential funnels; non-center all of them, or the one you left centered will keep diverging. This is the "second funnel" warning from the end of §7, made concrete.

This works, but it makes a strong, usually-false assumption: that a county's intercept and its slope are **independent**. They rarely are. Counties with high baseline radon (high intercept) might also have a bigger basement-vs-floor gap (steeper slope), because there's simply more gas to differ. To capture that, we let the intercept and slope be **correlated** — and that requires a multivariate prior.

---

## 9. Correlated varying effects with `pm.LKJCholeskyCov`

This is the most sophisticated pattern in the chapter, and once you have it you can build essentially any multilevel model. The idea: stack each county's intercept and slope into a **2-vector** $(\alpha_j, \beta_j)$ and say the whole vector is drawn from a **bivariate Normal** population, with a full covariance matrix $\Sigma$ that lets intercept and slope co-vary:

$$
\begin{bmatrix}\alpha_j\\\beta_j\end{bmatrix} \sim \mathrm{MvNormal}\!\left(\begin{bmatrix}\mu_\alpha\\\mu_\beta\end{bmatrix},\ \Sigma\right), \qquad \Sigma = \mathrm{diag}(\boldsymbol\sigma)\, R\, \mathrm{diag}(\boldsymbol\sigma) = L L^\top .
$$

Here $\boldsymbol\sigma = (\sigma_\alpha, \sigma_\beta)$ are the standard deviations of the two effects, $R$ is the $2\times 2$ **correlation matrix**, and $L$ is the **Cholesky factor** of the full covariance ($\Sigma = LL^\top$). The job of the prior is to put a sensible distribution on $R$ (and $\boldsymbol\sigma$), and the right tool is the **LKJ distribution**.

### The LKJ prior on correlation matrices

> 🧮 **The math — what LKJ does.** The **LKJ** (Lewandowski–Kurowicka–Joe) distribution is a prior over valid correlation matrices, controlled by a single shape parameter $\eta > 0$:
> - $\eta = 1$: **uniform** over all valid correlation matrices — every correlation from $-1$ to $1$ is equally likely a priori.
> - $\eta > 1$: concentrates toward the **identity** — shrinks correlations toward zero (skeptical of strong correlation).
> - $\eta < 1$: pushes mass toward the **extremes** ($\pm 1$).
>
> A common weakly-informative choice is $\eta = 2$, which gently favors weaker correlations without forbidding strong ones — it says "correlations exist but I doubt they're enormous," which is usually right.

In PyMC, you don't build $R$ and $\boldsymbol\sigma$ separately and multiply — you use `pm.LKJCholeskyCov`, which puts an LKJ prior on the correlation *and* a prior on the standard deviations, and hands you back the Cholesky factor directly. Its verified v5 signature:

```python
pm.LKJCholeskyCov(name, eta, n, sd_dist, *, compute_corr=True, store_in_trace=True)
```

Two things trip everyone up, so let me flag them before the code:

> ⚠️ **Pitfall — `sd_dist` must be an *unregistered* distribution built with `.dist()`.** You pass the prior for the standard deviations as `sd_dist`, but it must be created with the **`.dist()` API**, e.g. `pm.Exponential.dist(1.0, shape=2)` — *not* a registered RV like `pm.Exponential("sd", 1)`. PyMC **clones** it internally (the docs warn: "sd_dist will be cloned, rendering it independent of the one passed as input"). Pass a registered RV and you'll get an error.

> ⚠️ **Pitfall — in v5, `compute_corr=True` returns a *3-tuple*.** With the v5 default `compute_corr=True`, the call returns `chol, corr, stds` — the unpacked Cholesky factor $L$, the correlation matrix $R$, and the std-dev vector $\boldsymbol\sigma$. This is a **v3→v5 change**: old v3 code returned only the *packed* Cholesky and you had to call `pm.expand_packed_triangular`. If you copy a v3 snippet you'll get a shape error. In v5, **unpack three values.**

### The full correlated varying-intercept-and-slope model

Here is the complete pattern — non-centered, as it must be. Read the shapes carefully; the orientation of the matrix product is the one thing easy to get wrong.

```python
coords_corr = {
    "county": mn_counties,
    "param":  ["alpha", "beta"],          # the two correlated effects
    "obs_id": np.arange(len(log_radon)),
}

with pm.Model(coords=coords_corr) as covariation_model:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    floor_idx  = pm.Data("floor_idx", floor_measure, dims="obs_id")

    # --- Population means for intercept and slope ---
    mu_ab = pm.Normal("mu_ab", mu=0.0, sigma=10.0, dims="param")     # (2,)

    # --- LKJ prior on the 2x2 covariance; sd_dist via .dist() (NOTE the .dist) ---
    sd_dist = pm.Exponential.dist(1.0, shape=2)
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol_cov", eta=2.0, n=2, sd_dist=sd_dist,
        compute_corr=True, store_in_trace=False,   # keep corr/stds out of the file; recompute below
    )                                              # chol is the (2,2) lower-triangular L

    # --- Non-centered correlated draws: ab_j = mu_ab + L @ z_j ---
    # z has dims (param, county) so that L (2,2) @ z (2, J) -> (2, J), then .T -> (J, 2)
    z  = pm.Normal("z", 0.0, 1.0, dims=("param", "county"))
    ab = pm.Deterministic("ab", (mu_ab[:, None] + pt.dot(chol, z)).T,
                          dims=("county", "param"))     # (J, 2): col 0 = alpha, col 1 = beta

    alpha = ab[:, 0]
    beta  = ab[:, 1]

    # Recover the correlation as a named deterministic so it's in the summary:
    pm.Deterministic("corr_ab", corr[0, 1])             # the intercept-slope correlation

    sigma_y = pm.Exponential("sigma_y", 1.0)
    mu = alpha[county_idx] + beta[county_idx] * floor_idx
    y  = pm.Normal("y", mu=mu, sigma=sigma_y, observed=log_radon, dims="obs_id")

    idata_corr = pm.sample(1000, tune=1000, chains=4,
                           target_accept=0.9, random_seed=RANDOM_SEED)
```

> 🔧 **In practice — the shape check that saves an afternoon.** The transpose orientation is the classic bug. Trace the shapes once and you'll never doubt it again: `chol` (i.e. $L$) is `(2, 2)`; `z` is `(2, J)` because its dims are `("param", "county")`; `pt.dot(chol, z)` is `(2, J)`; adding the broadcast `mu_ab[:, None]` of shape `(2, 1)` keeps it `(2, J)`; the final `.T` gives `(J, 2)` so that row $j$ is county $j$'s $(\alpha_j, \beta_j)$. The PyMC radon primer uses exactly this `pt.dot(chol, z).T` orientation. If you ever see a baffling broadcasting error here, print the shapes — it's always the transpose. (Older posts use the alias `at.dot` from aesara; in current PyMC v5 it is `pt` for `pytensor.tensor` — update the alias when transcribing.)

### Interpreting the correlation

```python
az.summary(idata_corr, var_names=["mu_ab", "corr_ab", "sigma_y"])
az.plot_posterior(idata_corr, var_names=["corr_ab"])
```

> 🩺 **Diagnostic — reading `corr_ab`.** This is the headline of the correlated model: the posterior of the **intercept–slope correlation**. Suppose it comes back centered around, say, $-0.3$ with most of its mass below zero. That tells a *story*: counties with **higher baseline radon (high intercept) tend to have a more negative floor slope** — i.e. in high-radon counties, moving from basement to first floor buys you a *bigger* reduction. The model learned this from the data and, crucially, **uses it to borrow strength across the two parameters**: a county with few first-floor measurements (so a poorly-determined slope) but a well-determined high intercept will have its slope nudged in the direction the correlation predicts. That's *strength borrowed across parameters*, not just across groups — a second layer of partial pooling. Always read the correlation's *uncertainty*: if the 94% HDI for `corr_ab` comfortably spans zero, the data don't strongly support a correlation, and the simpler uncorrelated model (§8) might be preferable on parsimony grounds (check with LOO in **Chapter 11**).

> 💡 **Intuition — why model the correlation at all?** Two reasons. First, *honesty*: pretending $\alpha_j$ and $\beta_j$ are independent when they're not understates your uncertainty and can bias predictions. Second, *power*: a real correlation lets a county's well-measured intercept inform its poorly-measured slope (and vice versa), which is exactly the borrowing-of-strength that makes hierarchies valuable, now operating across parameters within a group. The cost is a slightly more complex model and one more thing to get the shapes right on — usually worth it when you have several varying effects.

> 📜 **Citation/Origin:** The LKJ distribution is from Lewandowski, Kurowicka & Joe (2009). The clean PyMC v5 template here follows the official *"Primer on Bayesian Methods for Multilevel Modeling"* (radon) `covariation_intercept_slope` block and Tomi Capretto's worked LKJ post on the `sleepstudy` data — both linked in Resources. The `sleepstudy` dataset (reaction time vs sleep deprivation, `Reaction ~ Days + (Days|Subject)`) is the other classic place to practice this exact pattern, with a *strong, real* positive intercept–slope correlation.

---

## 10. Identifiability with `ZeroSumNormal`

There's a subtle identifiability issue lurking in every varying-intercept model that has *both* a global level `mu_a` and free deflections `alpha_j`. Watch:

$$ \alpha_j = \mu_\alpha + (\text{deflection}_j). $$

You can **add a constant $c$ to $\mu_\alpha$ and subtract $c$ from every deflection** and the predictions are *identical*. The likelihood can't tell the two apart — only the prior breaks the tie. This is a *soft* non-identifiability: the hierarchical prior on the deflections (they're $\mathrm{Normal}(0, \sigma_\alpha)$, centered at zero) does pin it down well enough that sampling usually works, but the global level `mu_a` and the deflections can trade off and slosh against each other, hurting ESS and making `mu_a` harder to interpret cleanly.

The clean fix is to **constrain the deflections to sum to exactly zero**, so they *cannot* absorb a constant shift. PyMC v5 has a purpose-built distribution for this: `pm.ZeroSumNormal`.

> 🧮 **The math — `ZeroSumNormal`.** It's a Normal whose values are constrained to sum to zero along a specified axis (the last axis by default). Verified v5 signature:
> ```python
> pm.ZeroSumNormal(name, sigma=1, *, n_zerosum_axes=1, dims=None, shape=None)
> ```
> Because the deflections are forced to sum to zero, the global `mu_a` becomes the *crisp, unambiguous* overall level — no constant can leak between the two. This both **makes `mu_a` cleanly interpretable** and **often improves sampling** by removing the sloshing degree of freedom. Note that `ZeroSumNormal`'s win is **identifiability, not a funnel cure** — see the caveat after the code.

```python
with pm.Model(coords=coords) as zsn_model:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    floor_idx  = pm.Data("floor_idx", floor_measure, dims="obs_id")

    a_bar   = pm.Normal("a_bar", 0.0, 10.0)               # the global level (now identifiable)
    sigma_a = pm.Exponential("sigma_a", 1.0)
    # Deflections constrained to sum to zero across counties:
    a_dev   = pm.ZeroSumNormal("a_dev", sigma=sigma_a, dims="county")
    alpha   = pm.Deterministic("alpha", a_bar + a_dev, dims="county")

    beta    = pm.Normal("beta", 0.0, 10.0)
    sigma_y = pm.Exponential("sigma_y", 1.0)
    mu = alpha[county_idx] + beta * floor_idx
    y  = pm.Normal("y", mu=mu, sigma=sigma_y, observed=log_radon, dims="obs_id")

    idata_zsn = pm.sample(1000, tune=1000, chains=4,
                          target_accept=0.9, random_seed=RANDOM_SEED)
```

> ⚠️ **Pitfall — with a hierarchical `sigma_a`, this `ZeroSumNormal` is *centered* and can re-introduce a funnel.** Look at the code: `a_dev = pm.ZeroSumNormal("a_dev", sigma=sigma_a, dims="county")` welds the deflections' width directly to the free scale `sigma_a` — that is the *centered* parameterization, the exact funnel structure §7 spent pages warning against. In the small-`sigma_a` regime (most counties data-poor, as in radon) you can therefore see divergences here, just as in §4c. `ZeroSumNormal`'s benefit is **identifiability and a cleaner global level, not a cure for the funnel.** And unlike the plain varying intercept, it does *not* non-center with the simple offset trick: the sum-to-zero constraint doesn't compose cleanly with `mu + z*sigma` (a free `Normal(0,1)` vector wouldn't sum to zero), so there's no one-line reparameterization. The practical defense: watch the divergence count, raise `target_accept` (0.95 → 0.99), and reserve `ZeroSumNormal` for the regime where it pays — either a **fixed** (non-hierarchical) `sigma` for the deflections, or a likelihood-dominated, data-rich setting where the funnel never forms. If you need *both* sum-to-zero identifiability *and* a non-centered hierarchical scale on radon-like sparse data, prefer the plain non-centered intercept of §7 and impose the zero-sum interpretation post hoc.

> 🔧 **In practice — when to reach for `ZeroSumNormal`.** It shines for **categorical fixed-effect-style factors with no natural baseline** — think "effect of each of 12 brands" or "each of 5 treatment arms," where you don't want one arbitrary level to be the reference. Constraining the effects to sum to zero gives every level a deflection *relative to the grand mean*, which is exactly the interpretation you usually want, and it removes the global/deflection trade-off. For radon's varying intercepts it's a clean, optional upgrade; for ANOVA-style models it's close to essential.

> ⚠️ **Pitfall — `sigma` cannot be a vector across the zero-sum axis.** The docs are explicit: **"sigma cannot have length > 1 across the zero-sum axes."** So you can't put per-group scales on the dimension you're zero-summing. A scalar `sigma_a` (as above) is fine; a length-$J$ vector of per-county scales on the `county` zero-sum axis is not. Keep the scale scalar on the constrained axis.

---

## 11. Predicting for new, unseen groups

Here's a question that exposes whether you *really* understand the hierarchy. You've fit the radon model on 85 counties. A colleague asks: "What's the radon distribution for a house in a county we never measured?" A no-pooling model is *helpless* here — it has no intercept for a county it never saw. The hierarchical model answers instantly and honestly, and *this is its headline advantage*.

The logic: an unseen county's intercept is just a **fresh draw from the learned population**:

$$ \alpha_{\text{new}} = \mu_\alpha + \sigma_\alpha \, \tilde\alpha_{\text{new}}, \qquad \tilde\alpha_{\text{new}} \sim \mathrm{Normal}(0, 1). $$

Because we propagate the *full posterior* of $\mu_\alpha$ and $\sigma_\alpha$, the predictive for a new county is automatically wider than for a measured one — the uncertainty includes "we don't even know which county this is." That inflation is *correct*, and it's something a non-hierarchical model literally cannot give you.

There are two correct routes; I'll show the cleaner one (a fresh population draw) and name the other.

**Route 1 — population draw (a brand-new group with no data yet).** Build a small model that reuses the *posterior* of the population parameters and draws a fresh offset for the new county:

```python
# pseudocode-ish: draw the new county's intercept from the fitted population posterior
post = idata_nc.posterior
mu_a_samps    = post["mu_a"].values.flatten()        # (n_samples,)
sigma_a_samps = post["sigma_a"].values.flatten()     # (n_samples,)

# Fresh standardized offset per posterior sample -> new-county intercept:
z_new      = rng.normal(size=mu_a_samps.shape)
alpha_new  = mu_a_samps + sigma_a_samps * z_new      # posterior predictive for a new county's intercept

# Predict log-radon in a basement (floor=0) in this new county, for each posterior draw:
beta_samps    = post["beta"].values.flatten()
sigma_y_samps = post["sigma_y"].values.flatten()
mu_new        = alpha_new + beta_samps * 0.0
log_radon_new = rng.normal(mu_new, sigma_y_samps)    # full predictive distribution

# `[None, :]` prepends a length-1 chain axis so ArviZ reads the array as (chain, draw);
# a bare 1-D array without that axis can mis-plot or error in convert_to_dataset.
az.plot_posterior({"log_radon_new_county": log_radon_new[None, :]})
```

> 💡 **Intuition — where the extra uncertainty comes from.** `alpha_new` mixes two sources of uncertainty: our uncertainty about the *population* (`mu_a`, `sigma_a` are themselves posteriors) **and** the genuine spread *between* counties (`sigma_a` itself). A measured county's intercept only carries the first kind plus its own data; a new county carries the first plus the *full* between-county spread. So `log_radon_new` is correctly wider. **This automatic, calibrated inflation of predictive uncertainty for unseen groups is the single most practically valuable property of hierarchical models** — it's why they generalize and why they don't embarrass you on new data.

**Route 2 — the in-model way with fresh offsets and `sample_posterior_predictive`.** The idiomatic PyMC route adds a coord for the new group, defines `z_new ~ Normal(0,1, dims="new")`, computes the deterministic intercept, and calls `pm.sample_posterior_predictive(idata, var_names=[...], predictions=True)`. The matching of model variables to posterior samples is **by name**, so the population parameters (`mu_a`, `sigma_a`, `beta`, `sigma_y`) are reused from the fitted `idata`, while the fresh `z_new` is sampled from its prior. For new *rows of existing* counties you don't even need new coords — just `pm.set_data({"county_idx": new_idx, "floor_idx": new_floor})` and resample the predictive.

> ⚠️ **Pitfall — predicting for new groups by name and shape.** `sample_posterior_predictive` matches variables **by name** between the fitting model and the prediction model; keep names, shapes, and coords compatible or you'll get cryptic shape/coord errors. For genuinely *new* groups the variable's `dims` length must change (more counties), so you must update the coord too — Route 1's explicit population draw sidesteps this entirely and is the less error-prone choice. (See PyMC Labs' *"Out-of-Model Predictions in PyMC"* for the full in-model recipe.) Also note: `pm.Data` is **mutable by default in v5** — do *not* pass the old `mutable=True` kwarg (removed); just `pm.set_data({...})` to swap.

---

## 12. The one-liner: Bambi

Everything in this chapter has a formula-API shortcut in **Bambi** (which you met in **Chapter 09**). Bambi compiles these formulas down to *exactly* the PyMC models we wrote by hand — and it defaults to the **non-centered** parameterization, so it's funnel-safe out of the box. Here are the radon models as one-liners:

```python
import bambi as bmb

# Varying intercept only  ==  §4c / §7 (non-centered partial pooling):
m1 = bmb.Model("log_radon ~ floor + (1|county)", data=srrs_mn)

# Varying intercept + varying slope of floor, CORRELATED by default  ==  §9:
m2 = bmb.Model("log_radon ~ floor + (floor|county)", data=srrs_mn)

# Uncorrelated varying intercept + slope  ==  §8:
m3 = bmb.Model("log_radon ~ 0 + floor + (0 + floor|county)", data=srrs_mn)

idata_bambi = m2.fit(draws=1000, tune=1000, target_accept=0.9,
                     random_seed=RANDOM_SEED)            # returns az.InferenceData
# Bambi non-centers by default; pass noncentered=False to force the centered form.
```

> 🔧 **In practice — read the formula like a sentence.** `(1|county)` means "a random **intercept** that varies by county" (the `1` is the intercept). `(floor|county)` means "a random intercept **and** a random `floor` slope, varying by county, **correlated**" — this is the LKJ-covariance model from §9, and Bambi sets up the `LKJCholeskyCov` for you. `(0 + floor|county)` drops the random intercept and keeps only a varying slope. The `noncentered=True` default means Bambi already applied the §7 cure. **Use Bambi for speed and for communicating intent; reach for raw PyMC when you need a custom structure Bambi's formula grammar can't express** (a non-standard likelihood, a bespoke group-level predictor structure, a measurement-error layer). Per the course rule, whenever you show Bambi, know the raw-PyMC model it generates — and now you do, because we built every one of them by hand above.

> 💡 **Intuition — Bambi is a model *compiler*, not magic.** It does nothing you couldn't write yourself in PyMC; it just writes the boilerplate (the `coords`, the non-centered offsets, the `LKJCholeskyCov`, the sensible default priors) so you can iterate at the speed of thought. The reason we built everything by hand *first* is so that when Bambi does something surprising — or when you hit a model it can't express — you can drop down a level and fix it. That's the whole philosophy of this course: understand the machine, then use the power tools.

---

## 13. Worked end-to-end: radon partial pooling, start to finish

Let's run the full workflow once, cleanly, on the **non-centered varying-intercept-plus-uranium** model — load → prior check → fit → diagnose → posterior predictive → interpret. This is the model you'd actually ship for radon, and walking it end to end ties every piece of the chapter together.

**Step 0 — Data (from §3, recapped).** We have `log_radon` (outcome), `floor_measure` (house-level, 0/1), `county` (integer index 0–84), `u_county` (county-level log-uranium, length 85), and `mn_counties` (85 names). 919 houses, 85 counties, wildly uneven group sizes.

**Step 1 — Build the model.** Varying intercept with a uranium group-level predictor, non-centered, shared floor slope:

```python
coords = {"county": mn_counties, "obs_id": np.arange(len(log_radon))}

with pm.Model(coords=coords) as radon_final:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    floor_idx  = pm.Data("floor_idx", floor_measure, dims="obs_id")
    u_data     = pm.Data("u", u_county, dims="county")

    # Group-level (county) model of the intercept: a line in uranium + residual spread
    gamma0  = pm.Normal("gamma0", 0.0, 5.0)              # county-level intercept
    gamma1  = pm.Normal("gamma1", 0.0, 5.0)             # uranium slope (expect > 0)
    sigma_a = pm.Exponential("sigma_a", 1.0)            # residual between-county sd

    z_a   = pm.Normal("z_a", 0.0, 1.0, dims="county")   # non-centered offsets
    alpha = pm.Deterministic("alpha", gamma0 + gamma1 * u_data + z_a * sigma_a, dims="county")

    beta    = pm.Normal("beta", 0.0, 5.0)               # house-level floor slope (expect < 0)
    sigma_y = pm.Exponential("sigma_y", 1.0)            # within-county scatter

    mu = alpha[county_idx] + beta * floor_idx
    y  = pm.Normal("y", mu=mu, sigma=sigma_y, observed=log_radon, dims="obs_id")
```

**Step 2 — Prior predictive check.** Make sure the priors imply plausible radon before touching the posterior:

```python
with radon_final:
    # v5: use `draws=` (the old `samples=` kwarg is deprecated)
    prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
az.plot_ppc(prior_pred, group="prior", num_pp_samples=200)
```

Confirm the simulated `log_radon` covers the observed $[-1, 4]$ range without exploding to absurd values. It will — and it will in fact *far exceed* that range, because these priors are deliberately diffuse (a `Normal(0, 5)` intercept on the log scale is wide; `floor` and `u` are unstandardized). That is acceptable here only because we have ample data to overwhelm the prior; the check confirms the *scale* isn't insane, not that it's tight. With standardized predictors you could use `Normal(0, 2.5)`-style priors and get a markedly tighter, more honest prior predictive (see §5).

**Step 3 — Sample, with the standard call.**

```python
with radon_final:
    idata = pm.sample(1000, tune=1000, chains=4,
                      target_accept=0.9, random_seed=RANDOM_SEED)
    pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                   random_seed=RANDOM_SEED)
```

**Step 4 — Diagnose (the Chapter 07 loop).** Never interpret before this passes:

```python
print("divergences:", int(idata.sample_stats["diverging"].sum()))     # target: 0
az.summary(idata, var_names=["gamma0", "gamma1", "sigma_a", "beta", "sigma_y"])
az.plot_energy(idata)                                                  # BFMI healthy?
az.plot_trace(idata, var_names=["gamma0", "gamma1", "sigma_a", "beta", "sigma_y"])
```

> 🩺 **Diagnostic — what a healthy fit looks like here.** Because we non-centered, expect **0 divergences**, **$\hat R \le 1.01$** on every parameter, and **ESS_bulk / ESS_tail in the thousands**. The trace plots are fuzzy caterpillars with four well-mixed, overlapping chains; the rank plots (`az.plot_rank`) would be flat and uniform. The energy plot's two distributions overlap (BFMI well above $0.3$). If *any* of these is off — say a few divergences linger — raise `target_accept` to $0.95$, then $0.99$, and lengthen `tune`; if they persist, hunt for a second un-non-centered effect. (Here there's only one varying effect and it's non-centered, so it should be clean.)

**Step 5 — Posterior predictive check.**

```python
az.plot_ppc(idata, num_pp_samples=200)
```

> 🩺 **Diagnostic — reading `az.plot_ppc`.** It overlays many posterior-predictive `log_radon` densities (light) against the observed density (bold). You want the observed curve sitting comfortably *within* the spread of predictive curves — the model can reproduce data like what you saw. For radon it should: the Gaussian-on-log-scale likelihood matches the roughly-symmetric log-radon distribution well. If the observed curve poked out of the predictive envelope (e.g. the model couldn't reproduce a skew or a spike), you'd have *model misspecification* even with perfect convergence — and you'd reach for a heavier-tailed likelihood or an extra structural term. For sharper, double-dipping-free calibration (LOO-PIT, Bayesian $p$-values) see **Chapters 08 and 11**; they need a `log_likelihood` group, which `pm.compute_log_likelihood(idata, model=radon_final)` adds if sampling didn't.

**Step 6 — Interpret and post-process.** Now the science. Pull the posteriors and read them as quantities people care about:

```python
summ = az.summary(idata, var_names=["gamma1", "beta", "sigma_a", "sigma_y"])
print(summ)

# Floor effect on the natural scale (multiplicative, since we modeled log-radon):
beta_samps = idata.posterior["beta"].values.flatten()
# A 94% *quantile* interval (percentiles 3/50/97) — NOT an HDI. For an HDI use
# az.hdi(idata, var_names=["beta"], hdi_prob=0.94) and exponentiate the bounds.
print("first-floor multiplies radon by (3%/50%/97% quantiles):",
      np.exp(np.percentile(beta_samps, [3, 50, 97])))   # roughly ~[0.49, 0.55, 0.62] (seed/BLAS-dependent)
```

> 🩺 **Diagnostic — reading the science.**
> - **`gamma1` > 0** (uranium slope): high-uranium counties have higher baseline radon — the soil signal is real, with an HDI excluding zero. This is a *group-level* covariate explaining *why counties differ*.
> - **`beta` ≈ −0.6** (floor slope): because we modeled `log_radon`, $e^{\beta} \approx e^{-0.6} \approx 0.55$, so a **first-floor reading is ~55% of a basement reading** — the gas-from-below mechanism, quantified. (Always exponentiate to report log-scale slopes as multiplicative effects; see **Chapter 09** on link/scale interpretation.)
> - **`sigma_a`** is the residual between-county spread *after* uranium — smaller than in the uranium-free model, because uranium absorbed part of it. The fact that it's still clearly positive means counties differ for reasons beyond uranium, which the random intercept dutifully captures.
> - **County intercepts** (`alpha`, 85 of them) are your shrunken, uranium-informed per-county estimates. Plot them with `az.plot_forest(idata, var_names=["alpha"], combined=True)` — small counties have wider intervals pulled toward the uranium-predicted line; large counties stand on their own data. That forest plot is the deliverable: a defensible radon estimate for *every* county, including the data-starved ones.

> 💡 **The payoff, stated plainly.** You now have a single coherent model that (a) gives a sensible estimate for *every* county — even the ones with one house — by borrowing strength; (b) quantifies the floor effect as a multiplicative factor with honest uncertainty; (c) explains part of the between-county variation with soil uranium; (d) predicts for *unmeasured* counties with correctly-inflated uncertainty (§11); and (e) sampled cleanly because we non-centered. That is the crown jewel of applied Bayesian modeling, fit end to end. Everything else in this chapter is a variation on this skeleton.

---

## ⚠️ Common errors & how to fix them

| Symptom | Cause | Fix |
|---|---|---|
| `There were N divergences after tuning`, clustered at small `sigma_a` | **Centered funnel** — curvature too high in the neck where the group scale is small | **Non-center** (`z ~ Normal(0,1)`; `param = mu + z*sigma`). *Then*, if a few remain, raise `target_accept` 0.9→0.95→0.99 |
| Divergences persist *after* non-centering | Step size still slightly too large, very tight neck — **or a second un-non-centered effect** | `target_accept=0.99`, more `tune`; non-center *every* varying effect (a varying-slopes model has two funnels) |
| `sigma_a` posterior piles up at 0, low ESS on `sigma_a` | Centered sampler **can't reach the neck** — it misses the low-variance solutions and biases `sigma_a` upward | Non-center; inspect `az.plot_pair(var_names=["log_sigma_a", "alpha"], divergences=True)` to confirm the funnel |
| `LKJCholeskyCov` raises about `sd_dist` | Passed a **registered** RV (`pm.Exponential("sd", 1)`) instead of a `.dist()` object | Use `pm.Exponential.dist(1.0, shape=n)` — note the **`.dist`**; it gets cloned internally |
| `ValueError` / shape error unpacking the LKJ output | v3 habit: assumed a single *packed* Cholesky | In v5 with `compute_corr=True`, **unpack three**: `chol, corr, stds = pm.LKJCholeskyCov(...)` |
| Baffling broadcast error in `pt.dot(chol, z)` | Wrong transpose / wrong axis as the matrix dimension | Shape-check: `L` is `(2,2)`, `z` dims `("param","county")` ⇒ `(2,J)`; `pt.dot(L,z)` ⇒ `(2,J)`; `.T` ⇒ `(J,2)` |
| `pm.Data(..., mutable=True)` errors or warns | `mutable` kwarg **removed**; `pm.Data` is mutable by default in v5 | Drop the kwarg; swap with `pm.set_data({...})` before `sample_posterior_predictive` |
| Posterior predictive for a **new group** errors on shape/coords | Coord length changed but the variable's `dims` didn't (matching is by name & shape) | Update the coord *and* the variable shape; or use the explicit **population draw** (§11 Route 1) — cleaner and less error-prone |
| Model fits beautifully but results are nonsense | **Silent county→row index bug** (unsorted/non-contiguous factor) | Build `county_lookup = dict(zip(mn_counties, range(len(mn_counties))))`; verify `mn_counties[county[k]] == df.county.iloc[k]` |
| `sigma_y` / `sigma_a` mixing slow, heavy tails | Diffuse scale prior (`HalfCauchy(5)`, `Uniform(0,100)`) or `InverseGamma(ε,ε)` | Prefer `Exponential(1)` or `HalfNormal`; **never** `InverseGamma(ε,ε)` on a variance (Gelman 2006) |
| `ZeroSumNormal` raises about `sigma` length | Passed a **vector** `sigma` across the zero-sum axis | Docs forbid `sigma` length > 1 on the zero-sum axis — use a **scalar** scale there |
| Ported BUGS/JAGS model gives crazy scales | BUGS/JAGS parameterize Normal by **precision** $\tau = 1/\sigma^2$ | PyMC/Stan use **standard deviation $\sigma$** — convert: $\sigma = 1/\sqrt{\tau}$ |
| Added a group-level predictor, then dropped the random intercept | Misconception that the predictor *replaces* the varying intercept | Keep both — the predictor explains the *mean* of the intercepts; `sigma_a > 0` residual variation still needs the random intercept |

---

## 🧪 Exercises

**1. (Conceptual — the pooling spectrum.)** In one paragraph each, describe a real dataset from your own work where (a) complete pooling would be roughly fine, (b) no pooling would be roughly fine, and (c) partial pooling is clearly the right call. For (c), identify which groups will shrink the most and why. *Hint: think about the per-group sample sizes and how genuinely different the groups are — the precision-weighting formula in §1 tells you the answer.*

**2. (Coding — fit and compare the three regimes.)** On the radon data, fit the complete-pooling, no-pooling, and partial-pooling (non-centered) varying-intercept models from §4. Reproduce the **shrinkage plot** (no-pooling vs partial-pooling intercepts against county sample size). Then pick the three smallest counties and the three largest, and report each one's intercept estimate and 94% HDI width under no-pooling vs partial-pooling. *Hint: small counties' HDIs should shrink dramatically under partial pooling; large counties' should barely change. Quantify it.*

**3. (Coding — feel the funnel.)** Fit the **centered** partial-pooling model and the **non-centered** one with identical sampler settings. For each, report the divergence count, the `sigma_a` ESS_bulk, and the minimum of `sigma_a`'s posterior draws. Then make the funnel pair plot: add `pm.Deterministic("log_sigma_a", pt.log(sigma_a))` to the centered model and run `az.plot_pair(..., var_names=["log_sigma_a", "alpha"], coords={"county": [<one small county>]}, divergences=True)`. *Hint: in the centered fit the divergences cluster at the bottom of the funnel and `sigma_a` can't reach its smallest values; the non-centered fit has zero divergences and a much smaller minimum `sigma_a`.*

**4. (Coding — correlated varying slopes.)** Extend the model to a correlated varying intercept *and* floor slope using `pm.LKJCholeskyCov` (§9). Report the posterior of the intercept–slope correlation `corr_ab` with its 94% HDI, and write one sentence interpreting its sign. Does the data support a non-zero correlation (does the HDI exclude zero)? *Hint: try `eta=2`; remember `sd_dist = pm.Exponential.dist(1.0, shape=2)` with the `.dist`, and unpack three values from `LKJCholeskyCov`.*

**5. (Coding — predict an unseen county.)** Using the fitted non-centered model, produce the **posterior predictive distribution of log-radon for a basement house in a brand-new, unmeasured county** (§11, Route 1). Overlay it on the predictive for a *measured* county of your choice. *Hint: the new-county predictive should be visibly wider — explain in one sentence which source of uncertainty makes it so.*

**6. (Conceptual + light coding — centered vs non-centered in the data-rich regime.)** The chapter claims centered can beat non-centered when groups are data-rich. Test it with synthetic data: simulate a hierarchical Gaussian with **only 4 groups but 500 observations each** and a **large** true `sigma_a`. Fit both parameterizations and compare ESS-per-second and divergences. *Hint: in this likelihood-dominated regime you should see the centered form sampling *more* efficiently — the opposite of radon. This is the Betancourt & Girolami result in miniature; don't be surprised, be enlightened.*

**7. (Stretch — `ZeroSumNormal`.)** Refit the varying-intercept model with `pm.ZeroSumNormal` deflections (§10). Compare the posterior of the global level (`a_bar` vs `mu_a`) and its ESS to the unconstrained version. Does constraining the deflections to sum to zero improve the ESS of the global level or sharpen its interpretation? *Hint: the zero-sum constraint removes the global/deflection trade-off; watch the global level's ESS and the width of its posterior.*

---

## 📚 Resources & further reading

- **PyMC — *A Primer on Bayesian Methods for Multilevel Modeling* (radon).** https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html — *The* canonical PyMC v5 radon notebook: complete/no/partial pooling, varying intercept & slope, the non-centered offset trick, and the `LKJCholeskyCov` + `pt.dot(chol, z).T` covariation model. Our data load and correlated-effects block follow it directly. **Read this one first.**
- **Thomas Wiecki — *Why hierarchical models are awesome, tricky, and Bayesian.*** https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/ — The friendliest plain-language funnel explanation, with the centered↔non-centered code and the funnel/reparameterized-funnel scatter plots that are the mental model for §7.
- **Betancourt — *Hierarchical Modeling* case study.** https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html (PDF mirror: `.../hierarchical_modeling.pdf`) — The rigorous theory: prior-vs-likelihood domination decides the parameterization, funnel geometry, divergences as truth-tellers. The source for §7's "honest exception."
- **Betancourt & Girolami — *Hamiltonian Monte Carlo for Hierarchical Models* (2013), arXiv:1312.0906.** https://arxiv.org/abs/1312.0906 — The primary source for "centered when the likelihood dominates, non-centered when the prior dominates." Cite for the high-data-regime nuance (and Exercise 6).
- **Betancourt — *Diagnosing Biased Inference with Divergences* (PyMC port).** https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html — Divergences clustering in the neck, the non-centered fix, `target_accept` tuning. Pairs with Chapter 07.
- **Tomi Capretto — *Hierarchical modeling with the LKJ prior in PyMC.*** https://tomicapretto.com/posts/2022-06-12_lkj-prior/ — The cleanest worked `LKJCholeskyCov(..., compute_corr=True, store_in_trace=False)` → non-centered `dot(L, z).T` on `sleepstudy`, with `sd_dist` via `.dist()`. Best template for §9 (uses the old `at` alias — read as `pt` in v5).
- **PyMC docs — `pymc.LKJCholeskyCov`.** https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html — Verified signature, the `sd_dist` `.dist()` requirement, and the 3-tuple return.
- **PyMC docs — `pymc.ZeroSumNormal`.** https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroSumNormal.html — Verified `n_zerosum_axes`, the "sigma length ≤ 1 across zero-sum axes" constraint, and the `dims` example.
- **Bambi — *Hierarchical Linear Regression* (radon).** https://bambinos.github.io/bambi/notebooks/radon_example.html — The formula syntax `(1|county)`, `(floor|county)`, the prior dict, the `noncentered=` flag, and `.fit()`→InferenceData for §12.
- **PyMC Labs — *Out-of-Model Predictions in PyMC.*** https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc — The in-model recipe for predicting *new/unseen groups* (§11, Route 2): fresh offsets passed to `sample_posterior_predictive`.
- **Gelman & Hill (2007), *Data Analysis Using Regression and Multilevel/Hierarchical Models.*** The textbook home of the radon example and the pooling trichotomy. Chapters 11–13 are the core multilevel material. See also Gelman, Hill & Vehtari, *Regression and Other Stories* (2020, free PDF).
- **McElreath, *Statistical Rethinking* (2nd ed), Chapters 13–14** ("Models With Memory", "Adventures in Covariance") — the best intuition-first treatment of varying effects and correlated slopes. PyMC ports: *Statistical Rethinking Lecture 12 (Multilevel Models)* https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/12-Multilevel_Models.html and *Lecture 13 (Multilevel Adventures)* https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/13-Multilevel_Adventures.html.
- **Gelman (2006), *Prior distributions for variance parameters in hierarchical models*, Bayesian Analysis 1(3).** The definitive case for `HalfNormal`/`HalfCauchy`/`Exponential` on group scales and against `InverseGamma(ε,ε)` (§5).
- **Stan User's Guide — Reparameterization.** https://mc-stan.org/docs/stan-users-guide/reparameterization.html — The cross-language reference statement of the non-centered ("Matt trick") parameterization; same algebra, confirms §7.
- **Efron & Morris (1975/1977) on Stein's estimator and baseball**, and **James & Stein (1961)** — the shrinkage and James–Stein backbone of §2.

---

## ➡️ What's next

You can now build, fit, diagnose, and interpret the full zoo of multilevel models — varying intercepts, varying slopes, correlated effects, group-level predictors — and you have the non-centering reflex that keeps them sampling cleanly. But a nagging question has surfaced repeatedly: *is the more complex model actually better?* Does adding the uranium predictor, or the correlated slope, or another level of hierarchy, genuinely improve **predictions** — or just fit the noise? We dodged it with hand-waves toward "check with LOO." It's time to answer it properly. **Chapter 11 — Model Comparison & Evaluation** takes the *predictive* view of model quality head-on: expected log predictive density (ELPD), leave-one-out cross-validation via PSIS, WAIC, the $\hat k$ diagnostic that tells you when LOO is trustworthy, `az.compare`, stacking and model averaging, and the honest trouble with Bayes factors. The shrinkage you just learned to love is *exactly* the thing that makes a model generalize — and in the next chapter we'll learn to measure that generalization, so you can defend "this model is better" with a number instead of a hunch.
